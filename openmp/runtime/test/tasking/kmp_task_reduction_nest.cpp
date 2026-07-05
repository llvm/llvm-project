// RUN: %libomp-cxx-compile-and-run
// RUN: %libomp-cxx-compile -DFLG=1 && %libomp-run
// GCC-5 is needed for OpenMP 4.0 support (taskgroup)
// XFAIL: gcc-4
#include <cstdio>
#include <cmath>
#include <cassert>
#include <omp.h>

// Total number of loop iterations, should be multiple of T for this test
#define N 10000

// Flag to request lazy (1) or eager (0) allocation of reduction objects
#ifndef FLG
#define FLG 0
#endif

/*
  // initial user's code that corresponds to pseudo code of the test
  #pragma omp taskgroup task_reduction(+:i,j) task_reduction(*:x)
  {
    for( int l = 0; l < N; ++l ) {
      #pragma omp task firstprivate(l) in_reduction(+:i) in_reduction(*:x)
      {
        i += l;
        if( l%2 )
          x *= 1.0 / (l + 1);
        else
          x *= (l + 1);
      }
    }

    #pragma omp taskgroup task_reduction(-:i,k) task_reduction(+:y)
    {
      for( int l = 0; l < N; ++l ) {
        #pragma omp task firstprivate(l) in_reduction(+:j,y) \
            in_reduction(*:x) in_reduction(-:k)
        {
          j += l;
          k -= l;
          y += (double)l;
          if( l%2 )
            x *= 1.0 / (l + 1);
          else
            x *= (l + 1);
        }
        #pragma omp task firstprivate(l) in_reduction(+:y) in_reduction(-:i,k)
        {
          i -= l;
          k -= l;
          y += (double)l;
        }
        #pragma omp task firstprivate(l) in_reduction(+:j) in_reduction(*:x)
        {
          j += l;
          if( l%2 )
            x *= 1.0 / (l + 1);
          else
            x *= (l + 1);
        }
      }
    } // inner reduction

    for( int l = 0; l < N; ++l ) {
      #pragma omp task firstprivate(l) in_reduction(+:j)
        j += l;
    }
  } // outer reduction
*/

//------------------------------------------------
// OpenMP runtime library routines
#ifdef __cplusplus
extern "C" {
#endif
extern void* __kmpc_task_reduction_get_th_data(int gtid, void* tg, void* item);
extern void* __kmpc_task_reduction_init(int gtid, int num, void* data);
extern int __kmpc_global_thread_num(void*);
#ifdef __cplusplus
}
#endif

//------------------------------------------------
// Compiler-generated code

typedef struct _task_red_item {
    void       *shar; // shared reduction item
    size_t      size; // size of data item
    void       *f_init; // data initialization routine
    void       *f_fini; // data finalization routine
    void       *f_comb; // data combiner routine
    unsigned    flags;
} _task_red_item_t;

// int:+   no need in init/fini callbacks, valid for subtraction
void __red_int_add_comb(void *lhs, void *rhs) // combiner
{ *(int*)lhs += *(int*)rhs; }

// long long:+   no need in init/fini callbacks, valid for subtraction
void __red_llong_add_comb(void *lhs, void *rhs) // combiner
{ *(long long*)lhs += *(long long*)rhs; }

// double:*   no need in fini callback
void __red_dbl_mul_init(void *data) // initializer
{ *(double*)data = 1.0; }
void __red_dbl_mul_comb(void *lhs, void *rhs) // combiner
{ *(double*)lhs *= *(double*)rhs; }

// double:+   no need in init/fini callbacks
void __red_dbl_add_comb(void *lhs, void *rhs) // combiner
{ *(double*)lhs += *(double*)rhs; }

// ==============================

void calc_serial(int *pi, long long *pj, double *px, long long *pk, double *py)
{
    for( int l = 0; l < N; ++l ) {
        *pi += l;
        if( l%2 )
          *px *= 1.0 / (l + 1);
        else
          *px *= (l + 1);
    }
    for( int l = 0; l < N; ++l ) {
        *pj += l;
        *pk -= l;
        *py += (double)l;
        if( l%2 )
            *px *= 1.0 / (l + 1);
        else
            *px *= (l + 1);

        *pi -= l;
        *pk -= l;
        *py += (double)l;

        *pj += l;
        if( l%2 )
            *px *= 1.0 / (l + 1);
        else
            *px *= (l + 1);
    }
    for( int l = 0; l < N; ++l ) {
        *pj += l;
    }
}

//------------------------------------------------
// Test case
int main()
{
  int nthreads = omp_get_max_threads();
  int err = 0;
  void** ptrs = (void**)malloc(nthreads*sizeof(void*));

  // user's code ======================================
  // variables for serial calculations:
  int is = 3;
  long long js = -9999999;
  double xs = 99999.0;
  long long ks = 99999999;
  double ys = -99999999.0;
  // variables for parallel calculations:
  int ip = 3;
  long long jp = -9999999;
  double xp = 99999.0;
  long long kp = 99999999;
  double yp = -99999999.0;

  calc_serial(&is, &js, &xs, &ks, &ys);
  // ==================================================
  for (int i = 0; i < nthreads; ++i)
    ptrs[i] = NULL;
  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      // outer taskgroup reduces (i,j,x)
      #pragma omp taskgroup // task_reduction(+:i,j) task_reduction(*:x)
      {
        _task_red_item_t red_data[3];
        red_data[0].shar = &ip;
        red_data[0].size = sizeof(ip);
        red_data[0].f_init = NULL; // RTL will zero thread-specific objects
        red_data[0].f_fini = NULL; // no destructors needed
        red_data[0].f_comb = (void*)&__red_int_add_comb;
        red_data[0].flags = FLG;
        red_data[1].shar = &jp;
        red_data[1].size = sizeof(jp);
        red_data[1].f_init = NULL; // RTL will zero thread-specific objects
        red_data[1].f_fini = NULL; // no destructors needed
        red_data[1].f_comb = (void*)&__red_llong_add_comb;
        red_data[1].flags = FLG;
        red_data[2].shar = &xp;
        red_data[2].size = sizeof(xp);
        red_data[2].f_init = (void*)&__red_dbl_mul_init;
        red_data[2].f_fini = NULL; // no destructors needed
        red_data[2].f_comb = (void*)&__red_dbl_mul_comb;
        red_data[2].flags = FLG;
        int gtid = __kmpc_global_thread_num(NULL);
        void* tg1 = __kmpc_task_reduction_init(gtid, 3, red_data);

        for( int l = 0; l < N; l += 2 ) {
          // 2 iterations per task to get correct x value; actually any even
          // number of iters per task will work, otherwise x looses precision
          #pragma omp task firstprivate(l) //in_reduction(+:i) in_reduction(*:x)
          {
            int gtid = __kmpc_global_thread_num(NULL);
            int *p_ip = (int*)__kmpc_task_reduction_get_th_data(gtid, tg1, &ip);
            double *p_xp = (double*)__kmpc_task_reduction_get_th_data(
                                        gtid, tg1, &xp);
            if (!ptrs[gtid]) ptrs[gtid] = p_xp;

            // user's pseudo-code ==============================
            *p_ip += l;
            *p_xp *= (l + 1);

            *p_ip += l + 1;
            *p_xp *= 1.0 / (l + 2);
            // ==================================================
          }
        }
        // inner taskgroup reduces (i,k,y), i is same object as in outer one
        #pragma omp taskgroup // task_reduction(-:i,k) task_reduction(+:y)
        {
          _task_red_item_t red_data[3];
          red_data[0].shar = &ip;
          red_data[0].size = sizeof(ip);
          red_data[0].f_init = NULL; // RTL will zero thread-specific objects
          red_data[0].f_fini = NULL; // no destructors needed
          red_data[0].f_comb = (void*)&__red_int_add_comb;
          red_data[0].flags = FLG;
          red_data[1].shar = &kp;
          red_data[1].size = sizeof(kp);
          red_data[1].f_init = NULL; // RTL will zero thread-specific objects
          red_data[1].f_fini = NULL; // no destructors needed
          red_data[1].f_comb = (void*)&__red_llong_add_comb; // same for + and -
          red_data[1].flags = FLG;
          red_data[2].shar = &yp;
          red_data[2].size = sizeof(yp);
          red_data[2].f_init = NULL; // RTL will zero thread-specific objects
          red_data[2].f_fini = NULL; // no destructors needed
          red_data[2].f_comb = (void*)&__red_dbl_add_comb;
          red_data[2].flags = FLG;
          int gtid = __kmpc_global_thread_num(NULL);
          void* tg2 = __kmpc_task_reduction_init(gtid, 3, red_data);

          for( int l = 0; l < N; l += 2 ) {
            #pragma omp task firstprivate(l)
            // in_reduction(+:j,y) in_reduction(*:x) in_reduction(-:k)
            {
              int gtid = __kmpc_global_thread_num(NULL);
              long long *p_jp = (long long*)__kmpc_task_reduction_get_th_data(
                                                gtid, tg1, &jp);
              long long *p_kp = (long long*)__kmpc_task_reduction_get_th_data(
                                                gtid, tg2, &kp);
              double *p_xp = (double*)__kmpc_task_reduction_get_th_data(
                                          gtid, tg1, &xp);
              double *p_yp = (double*)__kmpc_task_reduction_get_th_data(
                                          gtid, tg2, &yp);
              // user's pseudo-code ==============================
              *p_jp += l;
              *p_kp -= l;
              *p_yp += (double)l;
              *p_xp *= (l + 1);

              *p_jp += l + 1;
              *p_kp -= l + 1;
              *p_yp += (double)(l + 1);
              *p_xp *= 1.0 / (l + 2);
              // =================================================
{
  // the following code is here just to check __kmpc_task_reduction_get_th_data:
  int tid = omp_get_thread_num();
  void *addr1;
  void *addr2;
  addr1 = __kmpc_task_reduction_get_th_data(gtid, tg1, &xp); // from shared
  addr2 = __kmpc_task_reduction_get_th_data(gtid, tg1, addr1); // from private
  if (addr1 != addr2) {
    #pragma omp atomic
      ++err;
    printf("Wrong thread-specific addresses %d s:%p p:%p\n", tid, addr1, addr2);
  }
  // from neighbour w/o taskgroup (should start lookup from current tg2)
  if (tid > 0) {
    if (ptrs[tid-1]) {
      addr2 = __kmpc_task_reduction_get_th_data(gtid, NULL, ptrs[tid-1]);
      if (addr1 != addr2) {
        #pragma omp atomic
          ++err;
        printf("Wrong thread-specific addresses %d s:%p n:%p\n",
               tid, addr1, addr2);
      }
    }
  } else {
    if (ptrs[nthreads-1]) {
      addr2 = __kmpc_task_reduction_get_th_data(gtid, NULL, ptrs[nthreads-1]);
      if (addr1 != addr2) {
        #pragma omp atomic
          ++err;
        printf("Wrong thread-specific addresses %d s:%p n:%p\n",
               tid, addr1, addr2);
      }
    }
  }
  // ----------------------------------------------
}
            }
            #pragma omp task firstprivate(l)
            // in_reduction(+:y) in_reduction(-:i,k)
            {
              int gtid = __kmpc_global_thread_num(NULL);
              int *p_ip = (int*)__kmpc_task_reduction_get_th_data(
                                    gtid, tg2, &ip);
              long long *p_kp = (long long*)__kmpc_task_reduction_get_th_data(
                                                gtid, tg2, &kp);
              double *p_yp = (double*)__kmpc_task_reduction_get_th_data(
                                          gtid, tg2, &yp);

              // user's pseudo-code ==============================
              *p_ip -= l;
              *p_kp -= l;
              *p_yp += (double)l;

              *p_ip -= l + 1;
              *p_kp -= l + 1;
              *p_yp += (double)(l + 1);
              // =================================================
            }
            #pragma omp task firstprivate(l)
            // in_reduction(+:j) in_reduction(*:x)
            {
              int gtid = __kmpc_global_thread_num(NULL);
              long long *p_jp = (long long*)__kmpc_task_reduction_get_th_data(
                                                gtid, tg1, &jp);
              double *p_xp = (double*)__kmpc_task_reduction_get_th_data(
                                          gtid, tg1, &xp);
              // user's pseudo-code ==============================
              *p_jp += l;
              *p_xp *= (l + 1);

              *p_jp += l + 1;
              *p_xp *= 1.0 / (l + 2);
              // =================================================
            }
          }
        } // inner reduction

        for( int l = 0; l < N; l += 2 ) {
          #pragma omp task firstprivate(l) // in_reduction(+:j)
          {
            int gtid = __kmpc_global_thread_num(NULL);
            long long *p_jp = (long long*)__kmpc_task_reduction_get_th_data(
                                              gtid, tg1, &jp);
            // user's pseudo-code ==============================
            *p_jp += l;
            *p_jp += l + 1;
            // =================================================
          }
        }
      } // outer reduction
    } // end single
  } // end parallel
  // check results
#if _DEBUG
  printf("reduction flags = %u\n", FLG);
#endif
  if (ip == is && jp == js && ks == kp &&
      fabs(xp - xs) < 0.01 && fabs(yp - ys) < 0.01)
    printf("passed\n");
  else
    printf("failed,\n ser:(%d %lld %f %lld %f)\n par:(%d %lld %f %lld %f)\n",
      is, js, xs, ks, ys,
      ip, jp, xp, kp, yp);
  return 0;
}
