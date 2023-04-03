#include <hostexec.h>
#include <omp.h>
#include <stdarg.h>
#include <stdio.h>

int myintfn(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  int *a = va_arg(args, int *);
  int i2 = va_arg(args, int);
  int i3 = va_arg(args, int);
  va_end(args);
  int rv = i2 + i3;
  printf("  INSIDE myintfn:  fnptr:%p  &a:%p int arg2:%d  int arg3:%d rv:%d \n",
         fnptr, a, i2, i3, rv);
  return rv;
}
double mydoublefn(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  int *a = va_arg(args, int *);
  int i2 = va_arg(args, int);
  int i3 = va_arg(args, int);
  double rv = (double)(i2 + i3) * 1.1;
  printf(
      "  INSIDE mydoublefn:  fnptr:%p  &a:%p int arg2:%d  int arg3:%d rv:%f \n",
      fnptr, a, i2, i3, rv);
  va_end(args);
  return rv;
}
long mylongfn(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  int *a = va_arg(args, int *);
  int i2 = va_arg(args, int);
  int i3 = va_arg(args, int);
  long rv = -(long)(i2 + i3);
  printf(
      "  INSIDE mylongfn:  fnptr:%p  &a:%p int arg2:%d  int arg3:%d rv:%ld \n",
      fnptr, a, i2, i3, rv);
  va_end(args);
  return rv;
}
float myfloatfn(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  int *a = va_arg(args, int *);
  int i2 = va_arg(args, int);
  int i3 = va_arg(args, int);
  float rv = (float)(i2 + i3) * 1.1;
  printf(
      "  INSIDE myfloatfn:  fnptr:%p  &a:%p int arg2:%d  int arg3:%d rv:%f \n",
      fnptr, a, i2, i3, rv);
  va_end(args);
  return rv;
}
void my4argvoidfn(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  int *a = va_arg(args, int *);
  int i2 = va_arg(args, int);
  int i3 = va_arg(args, int);
  printf("  INSIDE my4argvoidfn:  fnptr:%p  &a:%p int arg2:%d  int arg3:%d  \n",
         fnptr, a, i2, i3);
  va_end(args);
}

int main() {
  int N = 4;
  int a[N];
  int b[N];
  for (int i = 0; i < N; i++) {
    a[i] = 0;
    b[i] = i;
  }

  printf("\nTesting native variadic function pointers fnptr:%p  &a:%p\n",
         myintfn, (void *)a);
  uint sim1 = myintfn(myintfn, &a, 2, 3);
  double sim1d = mydoublefn(mydoublefn, &a, 2, 3);
  my4argvoidfn(my4argvoidfn, &a, 2, 3);
  printf("Return values are %d  and %f \n", sim1, sim1d);

  printf("\nTesting hostexec wrappers/macro on host (no target region)\n");
  printf("  Results should be same as above  fn_ptr:%p &a:%p\n", myintfn, &a);
  uint sim2 = hostexec_int(myintfn, &a, 2, 3);
  double sim2d = hostexec_double(mydoublefn, &a, 2, (int)3);
  hostexec(my4argvoidfn, &a, 2, 3);
  printf("Return values are %d  and %f \n", sim2, sim2d);

  // function pointers are not captured so convert to variables
  hostexec_int_t *myintfn_var = myintfn;
  hostexec_double_t *mydoublefn_var = mydoublefn;
  hostexec_float_t *myfloatfn_var = myfloatfn;
  hostexec_long_t *mylongfn_var = mylongfn;
  hostexec_t *my4argvoidfn_var = my4argvoidfn;

  printf("\nTesting hostexec variadic functions in simple target region \n");
  printf("  Results should be same as above  fn_ptr:%p &a:%p\n", myintfn, &a);
#pragma omp target map(to : a[0 : N])
  {
    uint sim2 = hostexec_int(myintfn_var, &a, 2, 3);
    double sim2d = hostexec_double(mydoublefn_var, &a, 2, (int)3);
    hostexec(my4argvoidfn_var, &a, 2, 3);
    printf("target printf: Return values are %d  and %f \n", sim2, sim2d);
  }

  int failcode = 0;
  printf(
      "\nTesting hostexec variadic functions in omp parallel target region \n");
#pragma omp target parallel for map(from : a[0 : N]) map(to : b[0 : N])        \
    map(tofrom : failcode)
  for (int j = 0; j < N; j++) {
    a[j] = b[j];
    uint rv = hostexec_int(myintfn_var, &a, j, a[j]);
    double rvd = hostexec_double(mydoublefn_var, &a, j, a[j]);
    float rvf = hostexec_float(myfloatfn_var, &a, j, a[j]);
    long rvl = hostexec_long(mylongfn_var, &a, j, a[j]);
    hostexec(my4argvoidfn_var, &a, j, a[j]);
    printf("target printf: t:%d of %d :: j:%d a[j]:%d return_vals int:%d "
           "double:%f float:%f long:%ld \n",
           omp_get_thread_num(), omp_get_num_threads(), j, a[j], rv, rvd, rvf,
           rvl);
    if (rv != j + a[j]) {
      failcode++;
      fprintf(stderr, "hostexec_int failed\n");
    }
    if (rvd != (double)(j + a[j]) * 1.1) {
      failcode++;
      fprintf(stderr, "hostexec_double failed\n");
    }
    if (rvf - ((float)(j + a[j]) * 1.1) > 10e-8) {
      failcode++;
      fprintf(stderr, "hostexec_float failed rvf:%f answer:%f \n", rvf,
              (float)(j + a[j]) * 1.1);
    }
    if (rvd - ((double)(j + a[j]) * 1.1) > 10e-15) {
      failcode++;
      fprintf(stderr, "hostexec_double failed rvd:%f answer:%f \n", rvd,
              (double)(j + a[j]) * 1.1);
    }
    if (rvl != -(long)(j + a[j])) {
      failcode++;
      printf("hostexec_long FAILED %ld rvl:%ld\n", -(long)(j + a[j]), rvl);
    }
  }

  // Check that b was copied into a
  int rc = 0;
  for (int i = 0; i < N; i++)
    if (a[i] != b[i]) {
      rc++;
      printf("Wrong value: a[%d]=%d\n", i, a[i]);
    }

  if (!rc && !failcode) {
    printf("Success\n");
    return EXIT_SUCCESS;
  } else {
    printf("Failure %d\n", failcode);
    return EXIT_FAILURE;
  }
}
