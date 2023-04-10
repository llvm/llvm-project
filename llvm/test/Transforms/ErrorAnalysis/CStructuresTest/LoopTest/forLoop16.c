#include<stdio.h>
#include "cpfloat_binary64.h"

double temp(double a, double res) {
  double a_out, res_out;

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

  // Validate the parameters in fpopts.
  int retval = cpfloat_validate_optstruct(fpopts);
  printf("The validation function returned %d.\n", retval);

  // Round the values of X to the binary16 format and store in Z.
  cpfloat(&a_out, &a, 1, fpopts);
  printf("a rounded to binary16:\n  %.15e \n", a_out);

  // Round the values of X to the binary16 format and store in Z.
  cpfloat(&res, &res, 1, fpopts);
  printf("res rounded to binary16:\n  %.15e \n", res_out);


  cpf_sub(&res_out, &res, &a_out, 1, fpopts);
  printf("res_out rounded to binary16:\n  %.15e \n", res_out);
  cpf_sub(&res, &res_out, &a_out, 1, fpopts);
  printf("res_out rounded to binary16:\n  %.15e \n", res);
  cpf_sub(&res_out, &res, &a_out, 1, fpopts);
  printf("res_out rounded to binary16:\n  %.15e \n", res_out);
  cpf_sub(&res, &res_out, &a_out, 1, fpopts);
  printf("res_out rounded to binary16:\n  %.15e \n", res);
  cpf_sub(&res_out, &res, &a_out, 1, fpopts);
  printf("res_out rounded to binary16:\n  %.15e \n", res_out);


  return res_out;
}

int main() {
  double a, res;
  int b;
  res=0.1;

  printf("Multiply Accumulator\n");
  printf("Enter value of a: ");
  scanf("%lf", &a);

  res = temp(a, res);


  printf("Result = %0.15lf\n", res);
  //  fAFfp32markForResult(res);

  return 0;
}

