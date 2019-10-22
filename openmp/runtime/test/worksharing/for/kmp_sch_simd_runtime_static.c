// RUN: %libomp-compile && %libomp-run
// RUN: %libomp-run 1 && %libomp-run 2
// REQUIRES: openmp-4.5

// The test checks schedule(simd:runtime)
// in combination with OMP_SCHEDULE=static[,chunk]
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#if defined(WIN32) || defined(_WIN32)
#include <windows.h>
#define delay() Sleep(1);
#define seten(a,b,c) _putenv_s((a),(b))
#else
#include <unistd.h>
#define delay() usleep(10);
#define seten(a,b,c) setenv((a),(b),(c))
#endif

#define SIMD_LEN 4
int err = 0;

// ---------------------------------------------------------------------------
// Various definitions copied from OpenMP RTL.
enum sched {
  kmp_sch_static_balanced_chunked = 45,
  kmp_sch_guided_simd = 46,
  kmp_sch_runtime_simd = 47,
};
typedef unsigned u32;
typedef long long i64;
typedef unsigned long long u64;
typedef struct {
  int reserved_1;
  int flags;
  int reserved_2;
  int reserved_3;
  char *psource;
} id;

#ifdef __cplusplus
extern "C" {
#endif
  int __kmpc_global_thread_num(id*);
  void __kmpc_barrier(id*, int gtid);
  void __kmpc_dispatch_init_4(id*, int, enum sched, int, int, int, int);
  void __kmpc_dispatch_init_8(id*, int, enum sched, i64, i64, i64, i64);
  int __kmpc_dispatch_next_4(id*, int, void*, void*, void*, void*);
  int __kmpc_dispatch_next_8(id*, int, void*, void*, void*, void*);
#ifdef __cplusplus
} // extern "C"
#endif
// End of definitions copied from OpenMP RTL.
// ---------------------------------------------------------------------------
static id loc = {0, 2, 0, 0, ";file;func;0;0;;"};

// ---------------------------------------------------------------------------
void
run_loop(
    int loop_lb,   // Loop lower bound.
    int loop_ub,   // Loop upper bound.
    int loop_st,   // Loop stride.
    int lchunk
) {
  static int volatile loop_sync = 0;
  int lb;   // Chunk lower bound.
  int ub;   // Chunk upper bound.
  int st;   // Chunk stride.
  int rc;
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();
  int gtid = __kmpc_global_thread_num(&loc);
  int last;
  int tc = (loop_ub - loop_lb) / loop_st + 1;
  int ch;
  int no_chunk = 0;
  if (lchunk == 0) {
    no_chunk = 1;
    lchunk = 1;
  }
  ch = lchunk * SIMD_LEN;
#if _DEBUG > 1
  printf("run_loop gtid %d tid %d (lb=%d, ub=%d, st=%d, ch=%d)\n",
         gtid, tid, (int)loop_lb, (int)loop_ub, (int)loop_st, lchunk);
#endif
  // Don't test degenerate cases that should have been discovered by codegen.
  if (loop_st == 0)
    return;
  if (loop_st > 0 ? loop_lb > loop_ub : loop_lb < loop_ub)
    return;
  __kmpc_dispatch_init_4(&loc, gtid, kmp_sch_runtime_simd,
                         loop_lb, loop_ub, loop_st, SIMD_LEN);
  {
    // Let the master thread handle the chunks alone.
    int chunk;      // No of current chunk.
    int last_ub;    // Upper bound of the last processed chunk.
    u64 cur;        // Number of interations in  current chunk.
    u64 max;        // Max allowed iterations for current chunk.
    int undersized = 0;
    last_ub = loop_ub;
    chunk = 0;
    max = (loop_ub - loop_lb) / loop_st + 1;
    // The first chunk can consume all iterations.
    while (__kmpc_dispatch_next_4(&loc, gtid, &last, &lb, &ub, &st)) {
      ++ chunk;
#if _DEBUG
      printf("th %d: chunk=%d, lb=%d, ub=%d ch %d\n",
             tid, chunk, (int)lb, (int)ub, (int)(ub-lb+1));
#endif
      // Check if previous chunk (it is not the final chunk) is undersized.
      if (undersized)
        printf("Error with chunk %d, th %d, err %d\n", chunk, tid, ++err);
      if (loop_st > 0) {
        if (!(ub <= loop_ub))
          printf("Error with ub %d, %d, ch %d, err %d\n",
                 (int)ub, (int)loop_ub, chunk, ++err);
        if (!(lb <= ub))
          printf("Error with bounds %d, %d, %d, err %d\n",
                 (int)lb, (int)ub, chunk, ++err);
      } else {
        if (!(ub >= loop_ub))
          printf("Error with ub %d, %d, %d, err %d\n",
                 (int)ub, (int)loop_ub, chunk, ++err);
        if (!(lb >= ub))
          printf("Error with bounds %d, %d, %d, err %d\n",
                 (int)lb, (int)ub, chunk, ++err);
      }; // if
      // Stride should not change.
      if (!(st == loop_st))
        printf("Error with st %d, %d, ch %d, err %d\n",
               (int)st, (int)loop_st, chunk, ++err);
      cur = ( ub - lb ) / loop_st + 1;
      // Guided scheduling uses FP computations, so current chunk may
      // be a bit bigger (+1) than allowed maximum.
      if (!( cur <= max + 1))
        printf("Error with iter %d, %d, err %d\n", cur, max, ++err);
      // Update maximum for the next chunk.
      if (last) {
        if (!no_chunk && cur > ch && nthreads > 1)
          printf("Error: too big last chunk %d (%d), tid %d, err %d\n",
                 (int)cur, ch, tid, ++err);
      } else {
        if (cur % ch)
          printf("Error with chunk %d, %d, ch %d, tid %d, err %d\n",
                 chunk, (int)cur, ch, tid, ++err);
      }
      if (cur < max)
        max = cur;
      last_ub = ub;
      undersized = (cur < ch);
#if _DEBUG > 1
      if (last)
        printf("under%d cur %d, ch %d, tid %d, ub %d, lb %d, st %d =======\n",
               undersized,cur,ch,tid,ub,lb,loop_st);
#endif
    } // while
    // Must have the right last iteration index.
    if (loop_st > 0) {
      if (!(last_ub <= loop_ub))
        printf("Error with last1 %d, %d, ch %d, err %d\n",
               (int)last_ub, (int)loop_ub, chunk, ++err);
      if (last && !(last_ub + loop_st > loop_ub))
        printf("Error with last2 %d, %d, %d, ch %d, err %d\n",
               (int)last_ub, (int)loop_st, (int)loop_ub, chunk, ++err);
    } else {
      if (!(last_ub >= loop_ub))
        printf("Error with last1 %d, %d, ch %d, err %d\n",
               (int)last_ub, (int)loop_ub, chunk, ++err);
      if (last && !(last_ub + loop_st < loop_ub))
        printf("Error with last2 %d, %d, %d, ch %d, err %d\n",
               (int)last_ub, (int)loop_st, (int)loop_ub, chunk, ++err);
    } // if
  }
  __kmpc_barrier(&loc, gtid);
} // run_loop

int main(int argc, char *argv[])
{
  int chunk = 0;
  if (argc > 1) {
    char *buf = malloc(8 + strlen(argv[1]));
    // expect chunk size as a parameter
    chunk = atoi(argv[1]);
    strcpy(buf,"static,");
    strcat(buf,argv[1]);
    seten("OMP_SCHEDULE",buf,1);
    printf("Testing schedule(simd:%s)\n", buf);
    free(buf);
  } else {
    seten("OMP_SCHEDULE","static",1);
    printf("Testing schedule(simd:static)\n");
  }
#pragma omp parallel// num_threads(num_th)
  run_loop(0, 26, 1, chunk);
  if (err) {
    printf("failed, err = %d\n", err);
    return 1;
  } else {
    printf("passed\n");
    return 0;
  }
}
