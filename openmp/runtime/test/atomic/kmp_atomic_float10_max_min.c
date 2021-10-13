// RUN: %libomp-compile -mlong-double-80 && %libomp-run
// UNSUPPORTED: gcc

#include <stdio.h>
#include <omp.h>

#ifdef  __cplusplus
extern "C" {
#endif
typedef void* ident_t;
extern void __kmpc_atomic_float10_max(ident_t *id_ref, int gtid,
                                      long double *lhs, long double rhs);
extern void __kmpc_atomic_float10_min(ident_t *id_ref, int gtid,
                                      long double *lhs, long double rhs);
extern long double __kmpc_atomic_float10_max_cpt(ident_t *id_ref, int gtid,
                                                 long double *lhs,
                                                 long double rhs, int flag);
extern long double __kmpc_atomic_float10_min_cpt(ident_t *id_ref, int gtid,
                                                 long double *lhs,
                                                 long double rhs, int flag);
#ifdef  __cplusplus
}
#endif

int main() {
  int ret = 0;
  long double s = 012.3456; // small
  long double e = 123.4567; // middle
  long double d = 234.5678; // big
  long double x = 123.4567; // object
  long double v = 0.; // captured value

// initialize OpenMP runtime library
  omp_set_num_threads(4);

// max
//  #pragma omp atomic compare update
//    if (x < d) x = d;
  __kmpc_atomic_float10_max(NULL, 0, &x, d);
  if (x != d) {
    ret++;
    printf("Error max: %Lf != %Lf\n", x, d);
  }
  __kmpc_atomic_float10_max(NULL, 0, &x, s); // no-op
  if (x != d) {
    ret++;
    printf("Error max: %Lf != %Lf\n", x, d);
  }

// min
//  #pragma omp atomic compare update
//    if (x > s) x = s;
  __kmpc_atomic_float10_min(NULL, 0, &x, s);
  if (x != s) {
    ret++;
    printf("Error min: %Lf != %Lf\n", x, s);
  }
  __kmpc_atomic_float10_min(NULL, 0, &x, e); // no-op
  if (x != s) {
    ret++;
    printf("Error min: %Lf != %Lf\n", x, s);
  }

// max_cpt old
//  #pragma omp atomic compare update capture
//    { v = x; if (x < d) x = d; }
  v = __kmpc_atomic_float10_max_cpt(NULL, 0, &x, d, 0);
  if (x != d) {
    ret++;
    printf("Error max_cpt obj: %Lf != %Lf\n", x, d);
  }
  if (v != s) {
    ret++;
    printf("Error max_cpt cpt: %Lf != %Lf\n", v, s);
  }
  v = __kmpc_atomic_float10_max_cpt(NULL, 0, &x, e, 0); // no-op
  if (x != d) {
    ret++;
    printf("Error max_cpt obj: %Lf != %Lf\n", x, d);
  }
  if (v != d) {
    ret++;
    printf("Error max_cpt cpt: %Lf != %Lf\n", v, d);
  }

// min_cpt old
//  #pragma omp atomic compare update capture
//    { v = x; if (x > d) x = d; }
  v = __kmpc_atomic_float10_min_cpt(NULL, 0, &x, s, 0);
  if (x != s) {
    ret++;
    printf("Error min_cpt obj: %Lf != %Lf\n", x, s);
  }
  if (v != d) {
    ret++;
    printf("Error min_cpt cpt: %Lf != %Lf\n", v, d);
  }
  v = __kmpc_atomic_float10_min_cpt(NULL, 0, &x, e, 0); // no-op
  if (x != s) {
    ret++;
    printf("Error max_cpt obj: %Lf != %Lf\n", x, s);
  }
  if (v != s) {
    ret++;
    printf("Error max_cpt cpt: %Lf != %Lf\n", v, s);
  }

// max_cpt new
//  #pragma omp atomic compare update capture
//    { if (x < d) x = d; v = x; }
  v = __kmpc_atomic_float10_max_cpt(NULL, 0, &x, d, 1);
  if (x != d) {
    ret++;
    printf("Error max_cpt obj: %Lf != %Lf\n", x, d);
  }
  if (v != d) {
    ret++;
    printf("Error max_cpt cpt: %Lf != %Lf\n", v, d);
  }
  v = __kmpc_atomic_float10_max_cpt(NULL, 0, &x, e, 1); // no-op
  if (x != d) {
    ret++;
    printf("Error max_cpt obj: %Lf != %Lf\n", x, d);
  }
  if (v != d) {
    ret++;
    printf("Error max_cpt cpt: %Lf != %Lf\n", v, d);
  }

// min_cpt new
//  #pragma omp atomic compare update capture
//    { if (x > d) x = d; v = x; }
  v = __kmpc_atomic_float10_min_cpt(NULL, 0, &x, s, 1);
  if (x != s) {
    ret++;
    printf("Error min_cpt obj: %Lf != %Lf\n", x, s);
  }
  if (v != s) {
    ret++;
    printf("Error min_cpt cpt: %Lf != %Lf\n", v, s);
  }
  v = __kmpc_atomic_float10_min_cpt(NULL, 0, &x, e, 1); // no-op
  if (x != s) {
    ret++;
    printf("Error max_cpt obj: %Lf != %Lf\n", x, s);
  }
  if (v != s) {
    ret++;
    printf("Error max_cpt cpt: %Lf != %Lf\n", v, s);
  }

  if (ret == 0)
    printf("passed\n");
  return ret;
}
