// RUN: %libomp-compile-and-run

#include <stdio.h>
#include <stdbool.h>
#include <omp.h>

// Used to detect architecture
#include "../../src/kmp_platform.h"

#ifdef  __cplusplus
extern "C" {
#endif
typedef void* ident_t;
extern bool
__kmpc_atomic_bool_1_cas_cpt(ident_t *loc, int gtid, char *x, char e, char d,
                             char *pv);
extern bool
__kmpc_atomic_bool_2_cas_cpt(ident_t *loc, int gtid, short *x, short e, short d,
                             short *pv);
extern bool
__kmpc_atomic_bool_4_cas_cpt(ident_t *loc, int gtid, int *x, int e, int d,
                             int *pv);
extern bool
__kmpc_atomic_bool_8_cas_cpt(ident_t *loc, int gtid, long long *x, long long e,
                             long long d, long long *pv);
extern char
__kmpc_atomic_val_1_cas_cpt(ident_t *loc, int gtid, char *x, char e, char d,
                            char *pv);
extern short
__kmpc_atomic_val_2_cas_cpt(ident_t *loc, int gtid, short *x, short e, short d,
                            short *pv);
extern int
__kmpc_atomic_val_4_cas_cpt(ident_t *loc, int gtid, int *x, int e, int d,
                            int *pv);
extern long long
__kmpc_atomic_val_8_cas_cpt(ident_t *loc, int gtid, long long *x, long long e,
                            long long d, long long *pv);
#ifdef  __cplusplus
}
#endif

int main() {
  int ret = 0;
#if KMP_ARCH_X86 || KMP_ARCH_X86_64
  bool r;
  char c0 = 1;
  char c1 = 2;
  char c2 = 3;
  char co = 2;
  char cc = 0;
  char cv = 0;
  short s0 = 11;
  short s1 = 12;
  short s2 = 13;
  short so = 12;
  short sc = 0;
  short sv = 0;
  int i0 = 211;
  int i1 = 212;
  int i2 = 213;
  int io = 212;
  int ic = 0;
  int iv = 0;
  long long l0 = 3111;
  long long l1 = 3112;
  long long l2 = 3113;
  long long lo = 3112;
  long long lc = 0;
  long long lv = 0;

// initialize OpenMP runtime library
  omp_set_dynamic(0);

//  #pragma omp atomic compare update capture
//    { r = x == e; if(r) { x = d; } else { v = x; } }
// char, co == c1 initially, co == c2 finally
  r = __kmpc_atomic_bool_1_cas_cpt(NULL, 0, &co, c0, c2, &cv); // no-op
  if (co != c1) {
    ret++; printf("Error bool_1_cas_cpt no-op: %d != %d\n", co, c1); }
  if (cv != co) {
    ret++; printf("Error bool_1_cas_cpt no-op cpt: %d != %d\n", cv, co); }
  if (r) { ret++; printf("Error bool_1_cas_cpt no-op ret: %d\n", r); }
  cv = 0;
  r = __kmpc_atomic_bool_1_cas_cpt(NULL, 0, &co, c1, c2, &cv);
  if (co != c2) { ret++; printf("Error bool_1_cas_cpt: %d != %d\n", co, c2); }
  if (cv != 0) { ret++; printf("Error bool_1_cas_cpt cpt: %d != %d\n", cv, 0); }
  if (!r) { ret++; printf("Error bool_1_cas_cpt ret: %d\n", r); }
// short
  r = __kmpc_atomic_bool_2_cas_cpt(NULL, 0, &so, s0, s2, &sv); // no-op
  if (so != s1) {
    ret++; printf("Error bool_2_cas_cpt no-op: %d != %d\n", so, s1); }
  if (sv != so) {
    ret++; printf("Error bool_2_cas_cpt no-op cpt: %d != %d\n", sv, so); }
  if (r) { ret++; printf("Error bool_2_cas_cpt no-op ret: %d\n", r); }
  sv = 0;
  r = __kmpc_atomic_bool_2_cas_cpt(NULL, 0, &so, s1, s2, &sv);
  if (so != s2) { ret++; printf("Error bool_2_cas_cpt: %d != %d\n", so, s2); }
  if (sv != 0) { ret++; printf("Error bool_2_cas_cpt cpt: %d != %d\n", sv, 0); }
  if (!r) { ret++; printf("Error bool_2_cas_cpt ret: %d\n", r); }
// int
  r = __kmpc_atomic_bool_4_cas_cpt(NULL, 0, &io, i0, i2, &iv); // no-op
  if (io != i1) {
    ret++; printf("Error bool_4_cas_cpt no-op: %d != %d\n", io, i1); }
  if (iv != io) {
    ret++; printf("Error bool_4_cas_cpt no-op cpt: %d != %d\n", iv, io); }
  if (r) { ret++; printf("Error bool_4_cas_cpt no-op ret: %d\n", r); }
  iv = 0;
  r = __kmpc_atomic_bool_4_cas_cpt(NULL, 0, &io, i1, i2, &iv);
  if (io != i2) { ret++; printf("Error bool_4_cas_cpt: %d != %d\n", io, i2); }
  if (iv != 0) { ret++; printf("Error bool_4_cas_cpt cpt: %d != %d\n", iv, 0); }
  if (!r) { ret++; printf("Error bool_4_cas_cpt ret: %d\n", r); }
// long long
  r = __kmpc_atomic_bool_8_cas_cpt(NULL, 0, &lo, l0, l2, &lv); // no-op
  if (lo != l1) {
    ret++; printf("Error bool_8_cas_cpt no-op: %lld != %lld\n", lo, l1); }
  if (lv != lo) {
    ret++; printf("Error bool_8_cas_cpt no-op cpt: %lld != %lld\n", lv, lo); }
  if (r) { ret++; printf("Error bool_8_cas_cpt no-op ret: %d\n", r); }
  lv = 0;
  r = __kmpc_atomic_bool_8_cas_cpt(NULL, 0, &lo, l1, l2, &lv);
  if (lo != l2) {
    ret++; printf("Error bool_8_cas_cpt: %lld != %lld\n", lo, l2); }
  if (lv != 0) { // should not be assigned
    ret++; printf("Error bool_8_cas_cpt cpt: %lld != %d\n", lv, 0); }
  if (!r) { ret++; printf("Error bool_8_cas_cpt ret: %d\n", r); }

//  #pragma omp atomic compare update capture
//    { if (x == e) { x = d; }; v = x; }
// char, co == c2 initially, co == c1 finally
  cc = __kmpc_atomic_val_1_cas_cpt(NULL, 0, &co, c0, c1, &cv); // no-op
  if (co != c2) {
    ret++; printf("Error val_1_cas_cpt no-op: %d != %d\n", co, c2); }
  if (cv != c2) {
    ret++; printf("Error val_1_cas_cpt no-op cpt: %d != %d\n", cv, c2); }
  if (cc != c2) {
    ret++; printf("Error val_1_cas_cpt no-op ret: %d != %d\n", cc, c2); }
  cc = __kmpc_atomic_val_1_cas_cpt(NULL, 0, &co, c2, c1, &cv);
  if (co != c1) { ret++; printf("Error val_1_cas_cpt: %d != %d\n", co, c1); }
  if (cv != c1) { ret++; printf("Error val_1_cas_cpt cpt: %d != %d\n", cv, c1); }
  if (cc != c2) { ret++; printf("Error val_1_cas_cpt ret: %d != %d\n", cc, c2); }
// short
  sc = __kmpc_atomic_val_2_cas_cpt(NULL, 0, &so, s0, s1, &sv); // no-op
  if (so != s2) {
    ret++; printf("Error val_2_cas_cpt no-op: %d != %d\n", so, s2); }
  if (sv != s2) {
    ret++; printf("Error val_2_cas_cpt no-op cpt: %d != %d\n", sv, s2); }
  if (sc != s2) {
    ret++; printf("Error val_2_cas_cpt no-op ret: %d != %d\n", sc, s2); }
  sc = __kmpc_atomic_val_2_cas_cpt(NULL, 0, &so, s2, s1, &sv);
  if (so != s1) { ret++; printf("Error val_2_cas_cpt: %d != %d\n", so, s1); }
  if (sv != s1) { ret++; printf("Error val_2_cas_cpt cpt: %d != %d\n", sv, s1); }
  if (sc != s2) { ret++; printf("Error val_2_cas_cpt ret: %d != %d\n", sc, s2); }
// int
  ic = __kmpc_atomic_val_4_cas_cpt(NULL, 0, &io, i0, i1, &iv); // no-op
  if (io != i2) {
    ret++; printf("Error val_4_cas_cpt no-op: %d != %d\n", io, i2); }
  if (iv != i2) {
    ret++; printf("Error val_4_cas_cpt no-op cpt: %d != %d\n", iv, i2); }
  if (ic != i2) {
    ret++; printf("Error val_4_cas_cpt no-op ret: %d != %d\n", ic, i2); }
  ic = __kmpc_atomic_val_4_cas_cpt(NULL, 0, &io, i2, i1, &iv);
  if (io != i1) { ret++; printf("Error val_4_cas_cpt: %d != %d\n", io, i1); }
  if (iv != i1) { ret++; printf("Error val_4_cas_cpt cpt: %d != %d\n", io, i1); }
  if (ic != i2) { ret++; printf("Error val_4_cas_cpt ret: %d != %d\n", ic, i2); }
// long long
  lc = __kmpc_atomic_val_8_cas_cpt(NULL, 0, &lo, l0, l1, &lv); // no-op
  if (lo != l2) {
    ret++; printf("Error val_8_cas_cpt no-op: %lld != %lld\n", lo, l2); }
  if (lv != l2) {
    ret++; printf("Error val_8_cas_cpt no-op cpt: %lld != %lld\n", lv, l2); }
  if (lc != l2) {
    ret++; printf("Error val_8_cas_cpt no-op ret: %lld != %lld\n", lc, l2); }
  lc = __kmpc_atomic_val_8_cas_cpt(NULL, 0, &lo, l2, l1, &lv);
  if (lo != l1) { ret++; printf("Error val_8_cas_cpt: %lld != %lld\n", lo, l1); }
  if (lv != l1) {
    ret++; printf("Error val_8_cas_cpt cpt: %lld != %lld\n", lv, l1); }
  if (lc != l2) {
    ret++; printf("Error val_8_cas_cpt ret: %lld != %lld\n", lc, l2); }

// check in parallel
  i0 = 1;
  i1 = 0;
  for (io = 0; io < 5; ++io) {
    #pragma omp parallel num_threads(2) private(i2, ic, r, iv)
    {
      if (omp_get_thread_num() == 0) {
        // th0 waits for th1 to increment i1, then th0 increments i0
        #pragma omp atomic read
          i2 = i1;
        ic = __kmpc_atomic_val_4_cas_cpt(NULL, 0, &i0, i2, i2 + 1, &iv);
        while(ic != i2) {
          if (iv != ic) {
            ret++;
            printf("Error 1 in parallel cpt, %d != %d\n", iv, ic);
          }
          #pragma omp atomic read
            i2 = i1;
          ic = __kmpc_atomic_val_4_cas_cpt(NULL, 0, &i0, i2, i2 + 1, &iv);
        }
        if (iv != i2 + 1) {
          ret++;
          printf("Error 2 in parallel cpt, %d != %d\n", iv, i2 + 1);
        }
      } else {
        // th1 increments i1 if it is equal to i0 - 1, letting th0 to proceed
        r = 0;
        while(!r) {
          #pragma omp atomic read
            i2 = i0;
          r = __kmpc_atomic_bool_4_cas_cpt(NULL, 0, &i1, i2 - 1, i2, &iv);
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
#else
  printf("Unsupported architecture, skipping test...\n");
#endif // KMP_ARCH_X86 || KMP_ARCH_X86_64
  return ret;
}
