// RUN: %libomp-compile-and-run

#include <stdio.h>
#include <stdbool.h>
#include <omp.h>

#ifdef  __cplusplus
extern "C" {
#endif
typedef void* ident_t;
extern bool
__kmpc_atomic_bool_1_cas(ident_t *loc, int gtid, char *x, char e, char d);
extern bool
__kmpc_atomic_bool_2_cas(ident_t *loc, int gtid, short *x, short e, short d);
extern bool
__kmpc_atomic_bool_4_cas(ident_t *loc, int gtid, int *x, int e, int d);
extern bool
__kmpc_atomic_bool_8_cas(ident_t *loc, int gtid, long long *x, long long e,
                         long long d);
extern char
__kmpc_atomic_val_1_cas(ident_t *loc, int gtid, char *x, char e, char d);
extern short
__kmpc_atomic_val_2_cas(ident_t *loc, int gtid, short *x, short e, short d);
extern int
__kmpc_atomic_val_4_cas(ident_t *loc, int gtid, int *x, int e, int d);
extern long long
__kmpc_atomic_val_8_cas(ident_t *loc, int gtid, long long *x, long long e,
                        long long d);
#ifdef  __cplusplus
}
#endif

int main() {
  int ret = 0;
  bool r;
  char c0 = 1;
  char c1 = 2;
  char c2 = 3;
  char co = 2;
  char cc = 0;
  short s0 = 11;
  short s1 = 12;
  short s2 = 13;
  short so = 12;
  short sc = 0;
  int i0 = 211;
  int i1 = 212;
  int i2 = 213;
  int io = 212;
  int ic = 0;
  long long l0 = 3111;
  long long l1 = 3112;
  long long l2 = 3113;
  long long lo = 3112;
  long long lc = 0;

// initialize OpenMP runtime library
  omp_set_dynamic(0);

//  #pragma omp atomic compare update capture
//    { r = x == e; if(r) { x = d; } }
// char, co == c1 initially, co == c2 finally
  r = __kmpc_atomic_bool_1_cas(NULL, 0, &co, c0, c2); // no-op
  if (co != c1) {
    ret++; printf("Error bool_1_cas no-op: %d != %d\n", co, c1); }
  if (r) { ret++; printf("Error bool_1_cas no-op ret: %d\n", r); }
  r = __kmpc_atomic_bool_1_cas(NULL, 0, &co, c1, c2);
  if (co != c2) {
    ret++; printf("Error bool_1_cas: %d != %d\n", co, c2); }
  if (!r) { ret++; printf("Error bool_1_cas ret: %d\n", r); }
// short
  r = __kmpc_atomic_bool_2_cas(NULL, 0, &so, s0, s2); // no-op
  if (so != s1) {
    ret++; printf("Error bool_2_cas no-op: %d != %d\n", so, s1); }
  if (r) { ret++; printf("Error bool_2_cas no-op ret: %d\n", r); }
  r = __kmpc_atomic_bool_2_cas(NULL, 0, &so, s1, s2);
  if (so != s2) {
    ret++; printf("Error bool_2_cas: %d != %d\n", so, s2); }
  if (!r) { ret++; printf("Error bool_2_cas ret: %d\n", r); }
// int
  r = __kmpc_atomic_bool_4_cas(NULL, 0, &io, i0, i2); // no-op
  if (io != i1) {
    ret++; printf("Error bool_4_cas no-op: %d != %d\n", io, i1); }
  if (r) { ret++; printf("Error bool_4_cas no-op ret: %d\n", r); }
  r = __kmpc_atomic_bool_4_cas(NULL, 0, &io, i1, i2);
  if (io != i2) {
    ret++; printf("Error bool_4_cas: %d != %d\n", io, i2); }
  if (!r) { ret++; printf("Error bool_4_cas ret: %d\n", r); }
// long long
  r = __kmpc_atomic_bool_8_cas(NULL, 0, &lo, l0, l2); // no-op
  if (lo != l1) {
    ret++; printf("Error bool_8_cas no-op: %lld != %lld\n", lo, l1); }
  if (r) { ret++; printf("Error bool_8_cas no-op ret: %d\n", r); }
  r = __kmpc_atomic_bool_8_cas(NULL, 0, &lo, l1, l2);
  if (lo != l2) {
    ret++; printf("Error bool_8_cas: %lld != %lld\n", lo, l2); }
  if (!r) { ret++; printf("Error bool_8_cas ret: %d\n", r); }

//  #pragma omp atomic compare update capture
//    { v = x; if (x == e) { x = d; } }
// char, co == c2 initially, co == c1 finally
  cc = __kmpc_atomic_val_1_cas(NULL, 0, &co, c0, c1); // no-op
  if (co != c2) {
    ret++; printf("Error val_1_cas no-op: %d != %d\n", co, c2); }
  if (cc != c2) {
    ret++; printf("Error val_1_cas no-op ret: %d != %d\n", cc, c2); }
  cc = __kmpc_atomic_val_1_cas(NULL, 0, &co, c2, c1);
  if (co != c1) {
    ret++; printf("Error val_1_cas: %d != %d\n", co, c1); }
  if (cc != c2) { ret++; printf("Error val_1_cas ret: %d != %d\n", cc, c2); }
// short
  sc = __kmpc_atomic_val_2_cas(NULL, 0, &so, s0, s1); // no-op
  if (so != s2) {
    ret++; printf("Error val_2_cas no-op: %d != %d\n", so, s2); }
  if (sc != s2) {
    ret++; printf("Error val_2_cas no-op ret: %d != %d\n", sc, s2); }
  sc = __kmpc_atomic_val_2_cas(NULL, 0, &so, s2, s1);
  if (so != s1) {
    ret++; printf("Error val_2_cas: %d != %d\n", so, s1); }
  if (sc != s2) {
    ret++; printf("Error val_2_cas ret: %d != %d\n", sc, s2); }
// int
  ic = __kmpc_atomic_val_4_cas(NULL, 0, &io, i0, i1); // no-op
  if (io != i2) {
    ret++; printf("Error val_4_cas no-op: %d != %d\n", io, i2); }
  if (ic != i2) {
    ret++; printf("Error val_4_cas no-op ret: %d != %d\n", ic, i2); }
  ic = __kmpc_atomic_val_4_cas(NULL, 0, &io, i2, i1);
  if (io != i1) {
    ret++; printf("Error val_4_cas: %d != %d\n", io, i1); }
  if (ic != i2) {
    ret++; printf("Error val_4_cas ret: %d != %d\n", ic, i2); }
// long long
  lc = __kmpc_atomic_val_8_cas(NULL, 0, &lo, l0, l1); // no-op
  if (lo != l2) {
    ret++; printf("Error val_8_cas no-op: %lld != %lld\n", lo, l2); }
  if (lc != l2) {
    ret++; printf("Error val_8_cas no-op ret: %lld != %lld\n", lc, l2); }
  lc = __kmpc_atomic_val_8_cas(NULL, 0, &lo, l2, l1);
  if (lo != l1) {
    ret++; printf("Error val_8_cas: %lld != %lld\n", lo, l1); }
  if (lc != l2) {
    ret++; printf("Error val_8_cas ret: %lld != %lld\n", lc, l2); }

// check in parallel
  i0 = 1;
  i1 = 0;
  for (io = 0; io < 5; ++io) {
    #pragma omp parallel num_threads(2) private(i2, ic, r)
    {
      if (omp_get_thread_num() == 0) {
        // th0 waits for th1 to increment i1, then th0 increments i0
        #pragma omp atomic read
          i2 = i1;
        ic = __kmpc_atomic_val_4_cas(NULL, 0, &i0, i2, i2 + 1);
        while(ic != i2) {
          #pragma omp atomic read
            i2 = i1;
          ic = __kmpc_atomic_val_4_cas(NULL, 0, &i0, i2, i2 + 1);
        }
      } else {
        // th1 increments i1 if it is equal to i0 - 1, letting th0 to proceed
        r = 0;
        while(!r) {
          #pragma omp atomic read
            i2 = i0;
          r = __kmpc_atomic_bool_4_cas(NULL, 0, &i1, i2 - 1, i2);
        }
      }
    }
  }
  if (i0 != 6 || i1 != 5) {
    ret++;
    printf("Error in parallel, %d != %d or %d != %d\n", i0, 6, i1, 5);
  }

  if (ret == 0)
    printf("passed\n");
  return ret;
}
