//
// Created by tanmay on 8/10/22.
//

#include <stdio.h>
#include <stdlib.h>
#include "cpfloat_binary64.h"

double RHO=12;
double SIGMA=10;
double BETA=8/3;
double POINT_FIVE_PERCENT=0.005;

#define TYPE double
#define PRINT_PRECISION_FORMAT "%0.15lf"

void solve_lorenz(double *x_0, double *y_0, double *z_0) {
  // Allocate the data structure for target formats and rounding parameters.
  optstruct *fpopts = init_optstruct();

  // Set up the parameters for binary16 target format.
  fpopts->precision = 11;                 // Bits in the significand + 1.
  fpopts->emax = 15;                      // The maximum exponent value.
  fpopts->subnormal = CPFLOAT_SUBN_USE;   // Support for subnormals is on.
  fpopts->round = CPFLOAT_RND_NE;         // Round toward +infinity.
  fpopts->flip = CPFLOAT_NO_SOFTERR;      // Bit flips are off.
  fpopts->p = 0;                          // Bit flip probability (not used).
  fpopts->explim = CPFLOAT_EXPRANGE_TARG; // Limited exponent in target format.

  cpfloat(x_0, x_0, 1, fpopts);
  cpfloat(y_0, y_0, 1, fpopts);
  cpfloat(z_0, z_0, 1, fpopts);

  double temp_1, temp_2, temp_3;
  double x_1, y_1, z_1;
  x_1 = 0;
  y_1 = 0;
  z_1 = 0;

  cpfloat(&x_1, &x_1, 1, fpopts);
  cpfloat(&y_1, &y_1, 1, fpopts);
  cpfloat(&z_1, &z_1, 1, fpopts);

  int n = 4;

  char command_filename[] = "lorenz_ode_commands.txt";
  FILE *command_unit;
  char data_filename[] = "lorenz_ode_data.txt";
  FILE *data_unit;

  data_unit = fopen ( data_filename, "wt" );

  for(int i = 0; i < n; i++) {
    //    printf("x_0 = "PRINT_PRECISION_FORMAT"\n", x_0);
    //    printf("y_0 = "PRINT_PRECISION_FORMAT"\n", y_0);
    //    printf("z_0 = "PRINT_PRECISION_FORMAT"\n\n", z_0);

    fprintf ( data_unit, "  %d  %14.6g  %14.6g  %14.6g\n",
            2*i, *x_0, *y_0, *z_0 );


    //    x_1 = x_0 + SIGMA*(y_0-x_0)*0.005;
    cpf_sub(&temp_1, y_0, x_0, 1, fpopts);
    cpf_mul(&temp_2, &SIGMA, &temp_1, 1, fpopts);
    cpf_mul(&temp_1, &temp_2, &POINT_FIVE_PERCENT, 1, fpopts);
    cpf_add(&x_1, x_0, &temp_1, 1, fpopts);

//    y_1 = y_0 + (RHO*x_0 - y_0 - x_0*z_0)*0.005;
    cpf_mul(&temp_1, &RHO, x_0, 1, fpopts);
    cpf_sub(&temp_2, &temp_1, y_0, 1, fpopts);
    cpf_mul(&temp_3, x_0, z_0, 1, fpopts);
    cpf_sub(&temp_1, &temp_2, &temp_3, 1, fpopts);
    cpf_mul(&temp_2, &temp_1, &POINT_FIVE_PERCENT, 1, fpopts);
    cpf_add(&y_1, y_0, &temp_2, 1, fpopts);

//    z_1 = z_0 + (x_0*y_0 - BETA*z_0)*0.005;
    cpf_mul(&temp_1, x_0, y_0, 1, fpopts);
    cpf_mul(&temp_2, &BETA, z_0, 1, fpopts);
    cpf_sub(&temp_3, &temp_1, &temp_2, 1, fpopts);
    cpf_mul(&temp_1, &temp_3, &POINT_FIVE_PERCENT, 1, fpopts);
    cpf_add(&z_1, z_0, &temp_1, 1, fpopts);

    //    printf("x_1 = "PRINT_PRECISION_FORMAT"\n", x_1);
    //    printf("y_1 = "PRINT_PRECISION_FORMAT"\n", y_1);
    //    printf("z_1 = "PRINT_PRECISION_FORMAT"\n\n", z_1);

    fprintf ( data_unit, "  %d  %14.6g  %14.6g  %14.6g\n",
            2*i+1, x_1, y_1, z_1 );

//    x_0 = x_1 + SIGMA*(y_1-x_1)*0.005;
    cpf_sub(&temp_1, &y_1, &x_1, 1, fpopts);
    cpf_mul(&temp_2, &SIGMA, &temp_1, 1, fpopts);
    cpf_mul(&temp_1, &temp_2, &POINT_FIVE_PERCENT, 1, fpopts);
    cpf_add(x_0, &x_1, &temp_1, 1, fpopts);

//    y_0 = y_1 + (RHO*x_1 - y_1 - x_1*z_1)*0.005;
    cpf_mul(&temp_1, &RHO, &x_1, 1, fpopts);
    cpf_sub(&temp_2, &temp_1, &y_1, 1, fpopts);
    cpf_mul(&temp_3, &x_1, &z_1, 1, fpopts);
    cpf_sub(&temp_1, &temp_2, &temp_3, 1, fpopts);
    cpf_mul(&temp_2, &temp_1, &POINT_FIVE_PERCENT, 1, fpopts);
    cpf_add(y_0, &y_1, &temp_1, 1, fpopts);

//    z_0 = z_1 + (x_1*y_1 - BETA*z_1)*0.005;
    cpf_mul(&temp_1, &x_1, &y_1, 1, fpopts);
    cpf_mul(&temp_2, &BETA, &z_1, 1, fpopts);
    cpf_sub(&temp_3, &temp_1, &temp_2, 1, fpopts);
    cpf_mul(&temp_1, &temp_3, &POINT_FIVE_PERCENT, 1, fpopts);
    cpf_add(z_0, &z_1, &temp_1, 1, fpopts);
  }

//      printf("x_0 = "PRINT_PRECISION_FORMAT"\n", x_0);
//      printf("y_0 = "PRINT_PRECISION_FORMAT"\n", y_0);
//      printf("z_0 = "PRINT_PRECISION_FORMAT"\n\n", z_0);
//
//  fprintf ( data_unit, "  %d  %14.6g  %14.6g  %14.6g\n",
//          2*i, *x_0, *y_0, *z_0 );
//
//
//  //    x_1 = x_0 + SIGMA*(y_0-x_0)*0.005;
//  cpf_sub(&temp_1, y_0, x_0, 1, fpopts);
//  cpf_mul(&temp_2, &SIGMA, &temp_1, 1, fpopts);
//  cpf_mul(&temp_1, &temp_2, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(&x_1, x_0, &temp_1, 1, fpopts);
//
//  //    y_1 = y_0 + (RHO*x_0 - y_0 - x_0*z_0)*0.005;
//  cpf_mul(&temp_1, &RHO, x_0, 1, fpopts);
//  cpf_sub(&temp_2, &temp_1, y_0, 1, fpopts);
//  cpf_mul(&temp_3, x_0, z_0, 1, fpopts);
//  cpf_sub(&temp_1, &temp_2, &temp_3, 1, fpopts);
//  cpf_mul(&temp_2, &temp_1, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(&y_1, y_0, &temp_2, 1, fpopts);
//
//  //    z_1 = z_0 + (x_0*y_0 - BETA*z_0)*0.005;
//  cpf_mul(&temp_1, x_0, y_0, 1, fpopts);
//  cpf_mul(&temp_2, &BETA, z_0, 1, fpopts);
//  cpf_sub(&temp_3, &temp_1, &temp_2, 1, fpopts);
//  cpf_mul(&temp_1, &temp_3, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(&z_1, z_0, &temp_1, 1, fpopts);
//
//  //    printf("x_1 = "PRINT_PRECISION_FORMAT"\n", x_1);
//  //    printf("y_1 = "PRINT_PRECISION_FORMAT"\n", y_1);
//  //    printf("z_1 = "PRINT_PRECISION_FORMAT"\n\n", z_1);
//
////  fprintf ( data_unit, "  %d  %14.6g  %14.6g  %14.6g\n",
////          2*i+1, x_1, y_1, z_1 );
//
//  //    x_0 = x_1 + SIGMA*(y_1-x_1)*0.005;
//  cpf_sub(&temp_1, &y_1, &x_1, 1, fpopts);
//  cpf_mul(&temp_2, &SIGMA, &temp_1, 1, fpopts);
//  cpf_mul(&temp_1, &temp_2, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(x_0, &x_1, &temp_1, 1, fpopts);
//
//  //    y_0 = y_1 + (RHO*x_1 - y_1 - x_1*z_1)*0.005;
//  cpf_mul(&temp_1, &RHO, &x_1, 1, fpopts);
//  cpf_sub(&temp_2, &temp_1, &y_1, 1, fpopts);
//  cpf_mul(&temp_3, &x_1, &z_1, 1, fpopts);
//  cpf_sub(&temp_1, &temp_2, &temp_3, 1, fpopts);
//  cpf_mul(&temp_2, &temp_1, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(y_0, &y_1, &temp_1, 1, fpopts);
//
//  //    z_0 = z_1 + (x_1*y_1 - BETA*z_1)*0.005;
//  cpf_mul(&temp_1, &x_1, &y_1, 1, fpopts);
//  cpf_mul(&temp_2, &BETA, &z_1, 1, fpopts);
//  cpf_sub(&temp_3, &temp_1, &temp_2, 1, fpopts);
//  cpf_mul(&temp_1, &temp_3, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(z_0, &z_1, &temp_1, 1, fpopts);
//
//
//  //    printf("x_0 = "PRINT_PRECISION_FORMAT"\n", x_0);
//  //    printf("y_0 = "PRINT_PRECISION_FORMAT"\n", y_0);
//  //    printf("z_0 = "PRINT_PRECISION_FORMAT"\n\n", z_0);
//
////  fprintf ( data_unit, "  %d  %14.6g  %14.6g  %14.6g\n",
////          2*i, *x_0, *y_0, *z_0 );
//
//
//  //    x_1 = x_0 + SIGMA*(y_0-x_0)*0.005;
//  cpf_sub(&temp_1, y_0, x_0, 1, fpopts);
//  cpf_mul(&temp_2, &SIGMA, &temp_1, 1, fpopts);
//  cpf_mul(&temp_1, &temp_2, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(&x_1, x_0, &temp_1, 1, fpopts);
//
//  //    y_1 = y_0 + (RHO*x_0 - y_0 - x_0*z_0)*0.005;
//  cpf_mul(&temp_1, &RHO, x_0, 1, fpopts);
//  cpf_sub(&temp_2, &temp_1, y_0, 1, fpopts);
//  cpf_mul(&temp_3, x_0, z_0, 1, fpopts);
//  cpf_sub(&temp_1, &temp_2, &temp_3, 1, fpopts);
//  cpf_mul(&temp_2, &temp_1, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(&y_1, y_0, &temp_2, 1, fpopts);
//
//  //    z_1 = z_0 + (x_0*y_0 - BETA*z_0)*0.005;
//  cpf_mul(&temp_1, x_0, y_0, 1, fpopts);
//  cpf_mul(&temp_2, &BETA, z_0, 1, fpopts);
//  cpf_sub(&temp_3, &temp_1, &temp_2, 1, fpopts);
//  cpf_mul(&temp_1, &temp_3, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(&z_1, z_0, &temp_1, 1, fpopts);
//
//  //    printf("x_1 = "PRINT_PRECISION_FORMAT"\n", x_1);
//  //    printf("y_1 = "PRINT_PRECISION_FORMAT"\n", y_1);
//  //    printf("z_1 = "PRINT_PRECISION_FORMAT"\n\n", z_1);
//
////  fprintf ( data_unit, "  %d  %14.6g  %14.6g  %14.6g\n",
////          2*i+1, x_1, y_1, z_1 );
//
//  //    x_0 = x_1 + SIGMA*(y_1-x_1)*0.005;
//  cpf_sub(&temp_1, &y_1, &x_1, 1, fpopts);
//  cpf_mul(&temp_2, &SIGMA, &temp_1, 1, fpopts);
//  cpf_mul(&temp_1, &temp_2, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(x_0, &x_1, &temp_1, 1, fpopts);
//
//  //    y_0 = y_1 + (RHO*x_1 - y_1 - x_1*z_1)*0.005;
//  cpf_mul(&temp_1, &RHO, &x_1, 1, fpopts);
//  cpf_sub(&temp_2, &temp_1, &y_1, 1, fpopts);
//  cpf_mul(&temp_3, &x_1, &z_1, 1, fpopts);
//  cpf_sub(&temp_1, &temp_2, &temp_3, 1, fpopts);
//  cpf_mul(&temp_2, &temp_1, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(y_0, &y_1, &temp_1, 1, fpopts);
//
//  //    z_0 = z_1 + (x_1*y_1 - BETA*z_1)*0.005;
//  cpf_mul(&temp_1, &x_1, &y_1, 1, fpopts);
//  cpf_mul(&temp_2, &BETA, &z_1, 1, fpopts);
//  cpf_sub(&temp_3, &temp_1, &temp_2, 1, fpopts);
//  cpf_mul(&temp_1, &temp_3, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(z_0, &z_1, &temp_1, 1, fpopts);
//
//
//  //    printf("x_0 = "PRINT_PRECISION_FORMAT"\n", x_0);
//  //    printf("y_0 = "PRINT_PRECISION_FORMAT"\n", y_0);
//  //    printf("z_0 = "PRINT_PRECISION_FORMAT"\n\n", z_0);
//
////  fprintf ( data_unit, "  %d  %14.6g  %14.6g  %14.6g\n",
////          2*i, *x_0, *y_0, *z_0 );
//
//
//  //    x_1 = x_0 + SIGMA*(y_0-x_0)*0.005;
//  cpf_sub(&temp_1, y_0, x_0, 1, fpopts);
//  cpf_mul(&temp_2, &SIGMA, &temp_1, 1, fpopts);
//  cpf_mul(&temp_1, &temp_2, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(&x_1, x_0, &temp_1, 1, fpopts);
//
//  //    y_1 = y_0 + (RHO*x_0 - y_0 - x_0*z_0)*0.005;
//  cpf_mul(&temp_1, &RHO, x_0, 1, fpopts);
//  cpf_sub(&temp_2, &temp_1, y_0, 1, fpopts);
//  cpf_mul(&temp_3, x_0, z_0, 1, fpopts);
//  cpf_sub(&temp_1, &temp_2, &temp_3, 1, fpopts);
//  cpf_mul(&temp_2, &temp_1, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(&y_1, y_0, &temp_2, 1, fpopts);
//
//  //    z_1 = z_0 + (x_0*y_0 - BETA*z_0)*0.005;
//  cpf_mul(&temp_1, x_0, y_0, 1, fpopts);
//  cpf_mul(&temp_2, &BETA, z_0, 1, fpopts);
//  cpf_sub(&temp_3, &temp_1, &temp_2, 1, fpopts);
//  cpf_mul(&temp_1, &temp_3, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(&z_1, z_0, &temp_1, 1, fpopts);
//
//  //    printf("x_1 = "PRINT_PRECISION_FORMAT"\n", x_1);
//  //    printf("y_1 = "PRINT_PRECISION_FORMAT"\n", y_1);
//  //    printf("z_1 = "PRINT_PRECISION_FORMAT"\n\n", z_1);
//
////  fprintf ( data_unit, "  %d  %14.6g  %14.6g  %14.6g\n",
////          2*i+1, x_1, y_1, z_1 );
//
//  //    x_0 = x_1 + SIGMA*(y_1-x_1)*0.005;
//  cpf_sub(&temp_1, &y_1, &x_1, 1, fpopts);
//  cpf_mul(&temp_2, &SIGMA, &temp_1, 1, fpopts);
//  cpf_mul(&temp_1, &temp_2, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(x_0, &x_1, &temp_1, 1, fpopts);
//
//  //    y_0 = y_1 + (RHO*x_1 - y_1 - x_1*z_1)*0.005;
//  cpf_mul(&temp_1, &RHO, &x_1, 1, fpopts);
//  cpf_sub(&temp_2, &temp_1, &y_1, 1, fpopts);
//  cpf_mul(&temp_3, &x_1, &z_1, 1, fpopts);
//  cpf_sub(&temp_1, &temp_2, &temp_3, 1, fpopts);
//  cpf_mul(&temp_2, &temp_1, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(y_0, &y_1, &temp_1, 1, fpopts);
//
//  //    z_0 = z_1 + (x_1*y_1 - BETA*z_1)*0.005;
//  cpf_mul(&temp_1, &x_1, &y_1, 1, fpopts);
//  cpf_mul(&temp_2, &BETA, &z_1, 1, fpopts);
//  cpf_sub(&temp_3, &temp_1, &temp_2, 1, fpopts);
//  cpf_mul(&temp_1, &temp_3, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(z_0, &z_1, &temp_1, 1, fpopts);
//
//
//  //    printf("x_0 = "PRINT_PRECISION_FORMAT"\n", x_0);
//  //    printf("y_0 = "PRINT_PRECISION_FORMAT"\n", y_0);
//  //    printf("z_0 = "PRINT_PRECISION_FORMAT"\n\n", z_0);
//
////  fprintf ( data_unit, "  %d  %14.6g  %14.6g  %14.6g\n",
////          2*i, *x_0, *y_0, *z_0 );
//
//
//  //    x_1 = x_0 + SIGMA*(y_0-x_0)*0.005;
//  cpf_sub(&temp_1, y_0, x_0, 1, fpopts);
//  cpf_mul(&temp_2, &SIGMA, &temp_1, 1, fpopts);
//  cpf_mul(&temp_1, &temp_2, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(&x_1, x_0, &temp_1, 1, fpopts);
//
//  //    y_1 = y_0 + (RHO*x_0 - y_0 - x_0*z_0)*0.005;
//  cpf_mul(&temp_1, &RHO, x_0, 1, fpopts);
//  cpf_sub(&temp_2, &temp_1, y_0, 1, fpopts);
//  cpf_mul(&temp_3, x_0, z_0, 1, fpopts);
//  cpf_sub(&temp_1, &temp_2, &temp_3, 1, fpopts);
//  cpf_mul(&temp_2, &temp_1, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(&y_1, y_0, &temp_2, 1, fpopts);
//
//  //    z_1 = z_0 + (x_0*y_0 - BETA*z_0)*0.005;
//  cpf_mul(&temp_1, x_0, y_0, 1, fpopts);
//  cpf_mul(&temp_2, &BETA, z_0, 1, fpopts);
//  cpf_sub(&temp_3, &temp_1, &temp_2, 1, fpopts);
//  cpf_mul(&temp_1, &temp_3, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(&z_1, z_0, &temp_1, 1, fpopts);
//
//  //    printf("x_1 = "PRINT_PRECISION_FORMAT"\n", x_1);
//  //    printf("y_1 = "PRINT_PRECISION_FORMAT"\n", y_1);
//  //    printf("z_1 = "PRINT_PRECISION_FORMAT"\n\n", z_1);
//
////  fprintf ( data_unit, "  %d  %14.6g  %14.6g  %14.6g\n",
////          2*i+1, x_1, y_1, z_1 );
//
//  //    x_0 = x_1 + SIGMA*(y_1-x_1)*0.005;
//  cpf_sub(&temp_1, &y_1, &x_1, 1, fpopts);
//  cpf_mul(&temp_2, &SIGMA, &temp_1, 1, fpopts);
//  cpf_mul(&temp_1, &temp_2, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(x_0, &x_1, &temp_1, 1, fpopts);
//
//  //    y_0 = y_1 + (RHO*x_1 - y_1 - x_1*z_1)*0.005;
//  cpf_mul(&temp_1, &RHO, &x_1, 1, fpopts);
//  cpf_sub(&temp_2, &temp_1, &y_1, 1, fpopts);
//  cpf_mul(&temp_3, &x_1, &z_1, 1, fpopts);
//  cpf_sub(&temp_1, &temp_2, &temp_3, 1, fpopts);
//  cpf_mul(&temp_2, &temp_1, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(y_0, &y_1, &temp_1, 1, fpopts);
//
//  //    z_0 = z_1 + (x_1*y_1 - BETA*z_1)*0.005;
//  cpf_mul(&temp_1, &x_1, &y_1, 1, fpopts);
//  cpf_mul(&temp_2, &BETA, &z_1, 1, fpopts);
//  cpf_sub(&temp_3, &temp_1, &temp_2, 1, fpopts);
//  cpf_mul(&temp_1, &temp_3, &POINT_FIVE_PERCENT, 1, fpopts);
//  cpf_add(z_0, &z_1, &temp_1, 1, fpopts);

  printf("x_0 = "PRINT_PRECISION_FORMAT"\n", *x_0);
  printf("y_0 = "PRINT_PRECISION_FORMAT"\n", *y_0);
  printf("z_0 = "PRINT_PRECISION_FORMAT"\n\n", *z_0);

  fprintf ( data_unit, "  %d  %14.6g  %14.6g  %14.6g\n",
          2*n, *x_0, *y_0, *z_0 );
}

int main() {
  double x_0, y_0, z_0;

  x_0 = 1.2;
  y_0 = 1.3;
  z_0 = 1.4;

//  RHO = 12;
//  SIGMA = 10;
//  BETA = 8/3;
//  POINT_FIVE_PERCENT = 0.005;

  //  printf("Enter value of z: ");
  //  scanf("%lf", z_0);

  solve_lorenz(&x_0, &y_0, &z_0);

  //  printf("z_1 = "PRINT_PRECISION_FORMAT"\n\n", z_1);
}