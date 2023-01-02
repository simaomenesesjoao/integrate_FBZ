#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 
#include <cmath>
#include <iostream>
#include <omp.h>


#define MIN       -4
//#define NEnergies 256

//#define NFermi    100
//#define minFermi  -4
//#define NPontos   512

#define TT        -1.0
#define DIM       4
#define a_cc      1
#define BETA      1000.0


Eigen::Matrix<std::complex<double>, DIM, DIM> hamiltonian(Eigen::Matrix<double, 1, 2>  k, double LAMBDA, double EX, double mag){
        Eigen::Matrix<std::complex<double>, DIM, DIM> H, Sx;
        std::complex<double> f, h1, h2, h3, Ephi1, Ephi2, Ephi3, h, g, zero(0,0), ii(0,1);
        std::complex<double> fx, hx, gx;
        std::complex<double> fy, hy, gy;
        Eigen::Matrix<double, 2, 1>  d1, d2, d3;
        Eigen::Matrix<double, 1, 2> Kappa, b1, b2;

        Sx << zero,  1.0, zero, zero, 
               1.0, zero, zero, zero, 
              zero, zero, zero,  1.0, 
              zero, zero,  1.0, zero;

        b1 <<  2*M_PI/sqrt(3), 2*M_PI/3; 
        b2 << -2*M_PI/sqrt(3), 2*M_PI/3;

        Kappa = (b1 - b2)/3.;

        d1 <<         0., -1.;
        d2 <<  sqrt(3)/2, 0.5;
        d3 << -sqrt(3)/2, 0.5; 

        h1 = exp(std::complex<double>(0, Kappa * d1));
        h2 = exp(std::complex<double>(0, Kappa * d2));
        h3 = exp(std::complex<double>(0, Kappa * d3));
        Ephi1 = exp(std::complex<double>(0.,static_cast<double>(k*d1)));
        Ephi2 = exp(std::complex<double>(0.,static_cast<double>(k*d2)));
        Ephi3 = exp(std::complex<double>(0.,static_cast<double>(k*d3)));

        // Full Hamiltonian
        f = (Ephi1 + Ephi2 + Ephi3) * TT;
        h = std::complex<double>(0., 2/3. * LAMBDA ) * (Ephi1 * h1 + Ephi2 * h2 + Ephi3 * h3);
        g = std::complex<double>(0., 2/3. * LAMBDA ) * (Ephi1 / h1 + Ephi2 / h2 + Ephi3 / h3);

        // velocity operator in the x direction
        fx = (Ephi1*d1[0] + Ephi2*d2[0] + Ephi3*d3[0]) * TT* std::complex<double>(0,1.0);
        hx = std::complex<double>(0., 2/3. * LAMBDA ) * (Ephi1 * h1*d1[0] + Ephi2 * h2*d2[0] + Ephi3 * h3*d3[0])* std::complex<double>(0,1.0);
        gx = std::complex<double>(0., 2/3. * LAMBDA ) * (Ephi1 / h1*d1[0] + Ephi2 / h2*d2[0] + Ephi3 / h3*d3[0])* std::complex<double>(0,1.0);

        // velocity operator in the y direction
        fy = (Ephi1*d1[1] + Ephi2*d2[1] + Ephi3*d3[1]) * TT* std::complex<double>(0,1.0);
        hy = std::complex<double>(0., 2/3. * LAMBDA ) * (Ephi1 * h1*d1[1] + Ephi2 * h2*d2[1] + Ephi3 * h3*d3[1])* std::complex<double>(0,1.0);
        gy = std::complex<double>(0., 2/3. * LAMBDA ) * (Ephi1 / h1*d1[1] + Ephi2 / h2*d2[1] + Ephi3 / h3*d3[1])* std::complex<double>(0,1.0);

        std::complex<double> ex(EX, 0);
        // Hamiltonian matrix
        H <<            ex,         zero,    f, h,
                      zero,          -ex,    g, f,
              std::conj(f), std::conj(g),   ex, zero,
              std::conj(h), std::conj(f), zero,  -ex;
        return H + mag*Sx;

}

int main(int argc, char **argv) {
    double EX, LAMBDA, minFermi, delta;
    unsigned NFermi, NPontos;
    if(argc == 7){
        LAMBDA   = atof(argv[1]);
        EX       = atof(argv[2]);
        minFermi = atof(argv[3]);
        NFermi   = atoi(argv[4]);
        NPontos  = atoi(argv[5]);
        delta    = atof(argv[6]);
    } else {
        std::cout << "Incorrect usage. Please pass the following arguments to the program:\n";
        std::cout << "LAMBDA EX minFermi NFermi NPontos delta\n";
            
        exit(1);
    }
    //
    //
    //



    double globalBarea; // area of the small section of the Brillouin zone

    Eigen::Array<std::complex<double>, -1, -1> global_cond(NFermi, 1);
    Eigen::Array<std::complex<double>, -1, -1> global_condK(NFermi, 1);
    Eigen::Array<std::complex<double>, -1, -1> global_condKp(NFermi, 1);
    Eigen::Array<std::complex<double>, -1, -1> global_condR(NFermi, 1);
    Eigen::Array<std::complex<double>, -1, -1> global_cond_conv(NFermi, 1); 



    // Definition and initialization of energy and fermi arrays
    Eigen::Array<std::complex<double>, -1, -1> fermis_global(NFermi, 1);
    for(unsigned i = 0; i < NFermi; i++)
        fermis_global(i,0) = std::complex<double>(minFermi - 2.*i*minFermi/(NFermi-1), 0);



    globalBarea = 0;

    global_cond.setZero();
    global_condK.setZero();
    global_condKp.setZero();
    global_condR.setZero();
    global_cond_conv.setZero();

    omp_set_num_threads(1);

#pragma omp parallel shared(global_cond, global_condK, global_condKp, global_condR, global_cond_conv) 
    {
        Eigen::Matrix<double, 1, 2> Kappa, Kappap, b1, b2, k;
        Eigen::Array<std::complex<double>, -1, -1> cond(NFermi, 1);
        Eigen::Array<std::complex<double>, -1, -1> condK(NFermi, 1);
        Eigen::Array<std::complex<double>, -1, -1> condKp(NFermi, 1);
        Eigen::Array<std::complex<double>, -1, -1> condR(NFermi, 1);
        Eigen::Array<std::complex<double>, -1, -1> cond_conv(NFermi, 1);
        double Barea = 0;
        //double a = sqrt(3)*a_cc;


        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<std::complex<double>, DIM, DIM>> es(DIM);
        Eigen::Array<std::complex<double>, -1, -1> fermis(NFermi, 1);
        fermis = fermis_global;

        cond.setZero();
        condK.setZero();
        condKp.setZero();
        condR.setZero();
        cond_conv.setZero();

        b1 <<  2*M_PI/sqrt(3), 2*M_PI/3; 
        b2 << -2*M_PI/sqrt(3), 2*M_PI/3;

        Kappa = (b1 - b2)/3.;
        Kappap = -Kappa;


        double dK, dKp;

#pragma omp for 
        for(unsigned ix = 0; ix < NPontos; ix++)
            for(unsigned iy = 0; iy < NPontos; iy++) {
                double norm = 4; // correction so that the integral is normalized to one

                double x = cos((2.*ix + 1.)/(2*NPontos) * M_PI);
                double y = cos((2.*iy + 1.)/(2*NPontos) * M_PI);
                double peso = sqrt(1. - x*x)*sqrt(1. - y*y) * pow(M_PI/NPontos,2)/norm;
                k = x*b1/2.0 + y*b2/2.0;

                // variables defined as if I had half the k points
                double x1 = cos((1.*ix + 1.)/(NPontos) * M_PI);
                double y1 = cos((1.*iy + 1.)/(NPontos) * M_PI);
                double peso1 = sqrt(1. - x1*x1)*sqrt(1. - y1*y1) * pow(2*M_PI/NPontos,2)/norm;

                // distance to the Dirac points
                dK = (k - Kappa).norm();
                dKp = (k - Kappap).norm();

                //double delta = 0.0001;
                // Fixing the units
                Eigen::Matrix<double, 1, 2>  delta_kx, delta_ky;
                delta_kx << delta, 0.0;
                delta_ky << 0.0, delta;

                Eigen::Matrix<std::complex<double>, DIM, DIM> H, Hx1, Hx2, Hy1, Hy2, Hm1, Hm2;
                double mag = 0;
                double delta_mag = delta;

                H = hamiltonian(k, LAMBDA, EX, mag);
                Hx1 = hamiltonian(k+delta_kx,   LAMBDA, EX, mag);
                Hx2 = hamiltonian(k+delta_kx/2, LAMBDA, EX, mag);
                Hy1 = hamiltonian(k+delta_ky,   LAMBDA, EX, mag);
                Hy2 = hamiltonian(k+delta_ky/2, LAMBDA, EX, mag);
                Hm1 = hamiltonian(k, LAMBDA, EX, mag + delta_mag);
                Hm2 = hamiltonian(k, LAMBDA, EX, mag + delta_mag/2);

                es.compute(H);
                Eigen::MatrixXd  e = es.eigenvalues();
                Eigen::MatrixXcd V = es.eigenvectors();

                es.compute(Hx1);
                Eigen::MatrixXd  ex1 = es.eigenvalues();
                Eigen::MatrixXcd Vx1 = es.eigenvectors();

                es.compute(Hx2);
                Eigen::MatrixXd  ex2 = es.eigenvalues();
                Eigen::MatrixXcd Vx2 = es.eigenvectors();

                es.compute(Hy1);
                Eigen::MatrixXd  ey1 = es.eigenvalues();
                Eigen::MatrixXcd Vy1 = es.eigenvectors();

                es.compute(Hy2);
                Eigen::MatrixXd  ey2 = es.eigenvalues();
                Eigen::MatrixXcd Vy2 = es.eigenvectors();

                es.compute(Hm1);
                Eigen::MatrixXd  em1 = es.eigenvalues();
                Eigen::MatrixXcd Vm1 = es.eigenvectors();

                es.compute(Hm2);
                Eigen::MatrixXd  em2 = es.eigenvalues();
                Eigen::MatrixXcd Vm2 = es.eigenvectors();

                for(unsigned n = 0; n < DIM; n++){
                    V.col(n)   *= std::abs(V(0,n))/V(0,n);
                    Vx1.col(n) *= std::abs(Vx1(0,n))/Vx1(0,n);
                    Vy1.col(n) *= std::abs(Vy1(0,n))/Vy1(0,n);
                    Vx2.col(n) *= std::abs(Vx2(0,n))/Vx2(0,n);
                    Vy2.col(n) *= std::abs(Vy2(0,n))/Vy2(0,n);

                    Vm1.col(n) *= std::abs(Vm1(0,n))/Vm1(0,n);
                    Vm2.col(n) *= std::abs(Vm2(0,n))/Vm2(0,n);
                    //Vx2.col(n) /= exp(std::complex<double>(0,std::arg(Vx2(0,n))));
                    //Vy2.col(n) /= exp(std::complex<double>(0,std::arg(Vy2(0,n))));
                }

                Eigen::MatrixXcd dVx1 = (Vx1-V)/delta;
                Eigen::MatrixXcd dVx2 = (Vx2-V)/(delta/2);
                Eigen::MatrixXcd dVy1 = (Vy1-V)/delta;
                Eigen::MatrixXcd dVy2 = (Vy2-V)/(delta/2);
                Eigen::MatrixXcd dVm1 = (Vm1-V)/delta_mag;
                Eigen::MatrixXcd dVm2 = (Vm2-V)/(delta_mag/2);


                //std::cout << "k: " << k << " ";
                //std::cout << e.transpose() << "\n";
                //std::cout << V << "\n\n";
                //std::cout << e1.transpose() << "\n";
                //std::cout << V1 << "\n\n\n";
                //std::cout << e2.transpose() << "\n";
                //std::cout << V2 << "\n\n\n";

                //std::cout << "difference\n";
                //std::cout << dV1 << "\n\n";
                //std::cout << dV2 << "\n\n";
                //std::cout << "____________________________________________\n";
                //std::cout << (Vx1-V).norm() << " ";
                //std::cout << (Vx2-V).norm() << " ";
                //std::cout << (Vy1-V).norm() << " ";
                //std::cout << (Vy2-V).norm() << "\n";

                //double norm1 = (Vx2-V).norm();
                //double norm2 = (Vy1-V).norm();
                //if(false){
                //if(norm1 > 0.1){
                    //std::cout << "k: " << k << " ";
                    //std::cout << norm1 << " ";
                    //std::cout << "\n";
                    //std::cout << ex1.transpose() << "\n";
                    //std::cout << Vx1 << "\n\n";
                    //std::cout << ex2.transpose() << "\n";
                    //std::cout << Vx2 << "\n\n";
                    //std::cout << e.transpose() << "\n";
                    //std::cout << V << "\n";
                //std::cout << "____________________________________________\n";
                //}


                Eigen::Array<std::complex<double>, -1, -1> temparray(NFermi, 1), fermarray(NFermi, 1);

                // Iterate over the eigenvalues
                temparray.setZero();

                
                // Kubo-Bastin with integration in energies
                for(int m = 0; m < DIM; m++) {
                    fermarray = (1.0 + exp((e(m) - fermis)*BETA)).inverse();
                    temparray += -(dVm2.col(m).adjoint()*dVx2.col(m) - dVx2.col(m).adjoint()*dVm2.col(m))(0,0)*fermarray;
                }
                // Fixing the units
                double Ac = 3*sqrt(3)/2.0;
                double factor = 2*M_PI/Ac;
                temparray = temparray*peso*factor;


                cond += temparray;
                double cutoff = 0.5;
                if(dK < cutoff){ 
                    condK += temparray; 
                    Barea += peso;
                }
                if(dKp < cutoff){ condKp += temparray; }
                if(dK > cutoff && dKp > cutoff){ condR += temparray; }

                // pretend there's only half the points
                if((ix%2==0) && (iy%2==0)){
                    cond_conv += temparray/peso*peso1;

                }


            }
#pragma omp critical
        {
            globalBarea += Barea;

        global_cond += cond;
        global_condK += condK;
        global_condKp += condKp;
        global_condR += condR;

        global_cond_conv += cond_conv;
        }
#pragma omp barrier
    };

    for(unsigned i = 0; i < NFermi; i++){
        std::cout << fermis_global(i).real()     << " ";
        std::cout << global_cond(i).real()      << " " << global_cond(i).imag()       << " ";
        std::cout << global_condK(i).real()     << " " << global_condK(i).imag()     << " ";
        std::cout << global_condKp(i).real()    << " " << global_condKp(i).imag()    << " ";
        std::cout << global_condR(i).real()     << " " << global_condR(i).imag()     << " ";
        std::cout << global_cond_conv(i).real()  << " " << global_cond_conv(i).imag()  << " ";
        std::cout << "\n";
    }

}
