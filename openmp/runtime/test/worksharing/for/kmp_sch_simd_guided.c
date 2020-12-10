// RUN: %libomp-compile-and-run
/*
  Test for the 'schedule(simd:guided)' clause.
  Compiler needs to generate a dynamic dispatching and pass the schedule
  value 46 to the OpenMP RTL. Test uses numerous loop parameter combinations.
*/
#include <stdio.h>
#include <omp.h>

#if defined(WIN32) || defined(_WIN32)
#include <windows.h>
#define delay() Sleep(1);
#else
#include <unistd.h>
#define delay() usleep(10);
#endif

// uncomment for debug diagnostics:
//#define DEBUG

#define SIMD_LEN 4

// ---------------------------------------------------------------------------
// Various definitions copied from OpenMP RTL
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

extern int __kmpc_global_thread_num(id*);
extern void __kmpc_barrier(id*, int gtid);
extern void __kmpc_dispatch_init_4(id*, int, enum sched, int, int, int, int);
extern void __kmpc_dispatch_init_8(id*, int, enum sched, i64, i64, i64, i64);
extern int __kmpc_dispatch_next_4(id*, int, void*, void*, void*, void*);
extern int __kmpc_dispatch_next_8(id*, int, void*, void*, void*, void*);
// End of definitions copied from OpenMP RTL.
// ---------------------------------------------------------------------------
static id loc = {0, 2, 0, 0, ";file;func;0;0;;"};

// ---------------------------------------------------------------------------
int run_loop_64(i64 loop_lb, i64 loop_ub, i64 loop_st, int loop_chunk) {
  int err = 0;
  static int volatile loop_sync = 0;
  i64 lb;   // Chunk lower bound
  i64 ub;   // Chunk upper bound
  i64 st;   // Chunk stride
  int rc;
  int tid = omp_get_thread_num();
  int gtid = tid;
  int last;
#if DEBUG
  printf("run_loop_<%d>(lb=%d, ub=%d, st=%d, ch=%d)\n",
    (int)sizeof(i64), gtid, tid,
    (int)loop_lb, (int)loop_ub, (int)loop_st, loop_chunk);
#endif
  // Don't test degenerate cases that should have been discovered by codegen
  if (loop_st == 0)
    return 0;
  if (loop_st > 0 ? loop_lb > loop_ub : loop_lb < loop_ub)
    return 0;

  __kmpc_dispatch_init_8(&loc, gtid, kmp_sch_guided_simd,
                         loop_lb, loop_ub, loop_st, loop_chunk);
  if (tid == 0) {
    // Let the master thread handle the chunks alone
    int chunk;      // No of current chunk
    i64 next_lb;    // Lower bound of the next chunk
    i64 last_ub;    // Upper bound of the last processed chunk
    u64 cur;        // Number of interations in  current chunk
    u64 max;        // Max allowed iterations for current chunk
    int undersized = 0;

    chunk = 0;
    next_lb = loop_lb;
    max = (loop_ub - loop_lb) / loop_st + 1;
    // The first chunk can consume all iterations
    while (__kmpc_dispatch_next_8(&loc, gtid, &last, &lb, &ub, &st)) {
      ++ chunk;
#if DEBUG
      printf("chunk=%d, lb=%d, ub=%d\n", chunk, (int)lb, (int)ub);
#endif
      // Check if previous chunk (it is not the final chunk) is undersized
      if (undersized) {
        printf("Error with chunk %d\n", chunk);
        err++;
      }
      // Check lower and upper bounds
      if (lb != next_lb) {
        printf("Error with lb %d, %d, ch %d\n", (int)lb, (int)next_lb, chunk);
        err++;
      }
      if (loop_st > 0) {
        if (!(ub <= loop_ub)) {
          printf("Error with ub %d, %d, ch %d\n", (int)ub, (int)loop_ub, chunk);
          err++;
        }
        if (!(lb <= ub)) {
          printf("Error with bounds %d, %d, %d\n", (int)lb, (int)ub, chunk);
          err++;
        }
      } else {
        if (!(ub >= loop_ub)) {
          printf("Error with ub %d, %d, %d\n", (int)ub, (int)loop_ub, chunk);
          err++;
        }
        if (!(lb >= ub)) {
          printf("Error with bounds %d, %d, %d\n", (int)lb, (int)ub, chunk);
          err++;
        }
      }; // if
      // Stride should not change
      if (!(st == loop_st)) {
        printf("Error with st %d, %d, ch %d\n", (int)st, (int)loop_st, chunk);
        err++;
      }
      cur = (ub - lb) / loop_st + 1;
      // Guided scheduling uses FP computations, so current chunk may
      // be a bit bigger (+1) than allowed maximum
      if (!(cur <= max + 1)) {
        printf("Error with iter %llu, %llu\n", cur, max);
        err++;
      }
      // Update maximum for the next chunk
      if (cur < max)
        max = cur;
      next_lb = ub + loop_st;
      last_ub = ub;
      undersized = (cur < loop_chunk);
    }; // while
    // Must have at least one chunk
    if (!(chunk > 0)) {
      printf("Error with chunk %d\n", chunk);
      err++;
    }
    // Must have the right last iteration index
    if (loop_st > 0) {
      if (!(last_ub <= loop_ub)) {
        printf("Error with last1 %d, %d, ch %d\n",
               (int)last_ub, (int)loop_ub, chunk);
        err++;
      }
      if (!(last_ub + loop_st > loop_ub)) {
        printf("Error with last2 %d, %d, %d, ch %d\n",
               (int)last_ub, (int)loop_st, (int)loop_ub, chunk);
        err++;
      }
    } else {
      if (!(last_ub >= loop_ub)) {
        printf("Error with last1 %d, %d, ch %d\n",
               (int)last_ub, (int)loop_ub, chunk);
        err++;
      }
      if (!(last_ub + loop_st < loop_ub)) {
        printf("Error with last2 %d, %d, %d, ch %d\n",
               (int)last_ub, (int)loop_st, (int)loop_ub, chunk);
        err++;
      }
    }; // if
    // Let non-master threads go
    loop_sync = 1;
  } else {
    int i;
    // Workers wait for master thread to finish, then call __kmpc_dispatch_next
    for (i = 0; i < 1000000; ++ i) {
      if (loop_sync != 0) {
        break;
      }; // if
    }; // for i
    while (loop_sync == 0) {
      delay();
    }; // while
    // At this moment we do not have any more chunks -- all the chunks already
    // processed by master thread
    rc = __kmpc_dispatch_next_8(&loc, gtid, &last, &lb, &ub, &st);
    if (rc) {
      printf("Error return value\n");
      err++;
    }
  }; // if

  __kmpc_barrier(&loc, gtid);
  if (tid == 0) {
      loop_sync = 0;    // Restore original state
#if DEBUG
      printf("run_loop_64(): at the end\n");
#endif
  }; // if
  __kmpc_barrier(&loc, gtid);
  return err;
} // run_loop

// ---------------------------------------------------------------------------
int run_loop_32(int loop_lb, int loop_ub, int loop_st, int loop_chunk) {
  int err = 0;
  static int volatile loop_sync = 0;
  int lb;   // Chunk lower bound
  int ub;   // Chunk upper bound
  int st;   // Chunk stride
  int rc;
  int tid = omp_get_thread_num();
  int gtid = tid;
  int last;
#if DEBUG
  printf("run_loop_<%d>(lb=%d, ub=%d, st=%d, ch=%d)\n",
    (int)sizeof(int), gtid, tid,
    (int)loop_lb, (int)loop_ub, (int)loop_st, loop_chunk);
#endif
  // Don't test degenerate cases that should have been discovered by codegen
  if (loop_st == 0)
    return 0;
  if (loop_st > 0 ? loop_lb > loop_ub : loop_lb < loop_ub)
    return 0;

  __kmpc_dispatch_init_4(&loc, gtid, kmp_sch_guided_simd,
                         loop_lb, loop_ub, loop_st, loop_chunk);
  if (tid == 0) {
    // Let the master thread handle the chunks alone
    int chunk;      // No of current chunk
    int next_lb;    // Lower bound of the next chunk
    int last_ub;    // Upper bound of the last processed chunk
    u64 cur;        // Number of interations in  current chunk
    u64 max;        // Max allowed iterations for current chunk
    int undersized = 0;

    chunk = 0;
    next_lb = loop_lb;
    max = (loop_ub - loop_lb) / loop_st + 1;
    // The first chunk can consume all iterations
    while (__kmpc_dispatch_next_4(&loc, gtid, &last, &lb, &ub, &st)) {
      ++ chunk;
#if DEBUG
      printf("chunk=%d, lb=%d, ub=%d\n", chunk, (int)lb, (int)ub);
#endif
      // Check if previous chunk (it is not the final chunk) is undersized
      if (undersized) {
        printf("Error with chunk %d\n", chunk);
        err++;
      }
      // Check lower and upper bounds
      if (lb != next_lb) {
        printf("Error with lb %d, %d, ch %d\n", (int)lb, (int)next_lb, chunk);
        err++;
      }
      if (loop_st > 0) {
        if (!(ub <= loop_ub)) {
          printf("Error with ub %d, %d, ch %d\n", (int)ub, (int)loop_ub, chunk);
          err++;
        }
        if (!(lb <= ub)) {
          printf("Error with bounds %d, %d, %d\n", (int)lb, (int)ub, chunk);
          err++;
        }
      } else {
        if (!(ub >= loop_ub)) {
          printf("Error with ub %d, %d, %d\n", (int)ub, (int)loop_ub, chunk);
          err++;
        }
        if (!(lb >= ub)) {
          printf("Error with bounds %d, %d, %d\n", (int)lb, (int)ub, chunk);
          err++;
        }
      }; // if
      // Stride should not change
      if (!(st == loop_st)) {
        printf("Error with st %d, %d, ch %d\n", (int)st, (int)loop_st, chunk);
        err++;
      }
      cur = (ub - lb) / loop_st + 1;
      // Guided scheduling uses FP computations, so current chunk may
      // be a bit bigger (+1) than allowed maximum
      if (!(cur <= max + 1)) {
        printf("Error with iter %llu, %llu\n", cur, max);
        err++;
      }
      // Update maximum for the next chunk
      if (cur < max)
        max = cur;
      next_lb = ub + loop_st;
      last_ub = ub;
      undersized = (cur < loop_chunk);
    }; // while
    // Must have at least one chunk
    if (!(chunk > 0)) {
      printf("Error with chunk %d\n", chunk);
      err++;
    }
    // Must have the right last iteration index
    if (loop_st > 0) {
      if (!(last_ub <= loop_ub)) {
        printf("Error with last1 %d, %d, ch %d\n",
               (int)last_ub, (int)loop_ub, chunk);
        err++;
      }
      if (!(last_ub + loop_st > loop_ub)) {
        printf("Error with last2 %d, %d, %d, ch %d\n",
               (int)last_ub, (int)loop_st, (int)loop_ub, chunk);
        err++;
      }
    } else {
      if (!(last_ub >= loop_ub)) {
        printf("Error with last1 %d, %d, ch %d\n",
               (int)last_ub, (int)loop_ub, chunk);
        err++;
      }
      if (!(last_ub + loop_st < loop_ub)) {
        printf("Error with last2 %d, %d, %d, ch %d\n",
               (int)last_ub, (int)loop_st, (int)loop_ub, chunk);
        err++;
      }
    }; // if
    // Let non-master threads go
    loop_sync = 1;
  } else {
    int i;
    // Workers wait for master thread to finish, then call __kmpc_dispatch_next
    for (i = 0; i < 1000000; ++ i) {
      if (loop_sync != 0) {
        break;
      }; // if
    }; // for i
    while (loop_sync == 0) {
      delay();
    }; // while
    // At this moment we do not have any more chunks -- all the chunks already
    // processed by the master thread
    rc = __kmpc_dispatch_next_4(&loc, gtid, &last, &lb, &ub, &st);
    if (rc) {
      printf("Error return value\n");
      err++;
    }
  }; // if

  __kmpc_barrier(&loc, gtid);
  if (tid == 0) {
      loop_sync = 0;    // Restore original state
#if DEBUG
      printf("run_loop<>(): at the end\n");
#endif
  }; // if
  __kmpc_barrier(&loc, gtid);
  return err;
} // run_loop

// ---------------------------------------------------------------------------
int run_64(int num_th)
{
 int err = 0;
#pragma omp parallel num_threads(num_th)
 {
  int chunk;
  i64 st, lb, ub;
  for (chunk = SIMD_LEN; chunk <= 3*SIMD_LEN; chunk += SIMD_LEN) {
    for (st = 1; st <= 3; ++ st) {
      for (lb = -3 * num_th * st; lb <= 3 * num_th * st; ++ lb) {
        for (ub = lb; ub < lb + num_th * (chunk+1) * st; ++ ub) {
          err += run_loop_64(lb, ub,  st, chunk);
          err += run_loop_64(ub, lb, -st, chunk);
        }; // for ub
      }; // for lb
    }; // for st
  }; // for chunk
 }
 return err;
} // run_all

int run_32(int num_th)
{
 int err = 0;
#pragma omp parallel num_threads(num_th)
 {
  int chunk, st, lb, ub;
  for (chunk = SIMD_LEN; chunk <= 3*SIMD_LEN; chunk += SIMD_LEN) {
    for (st = 1; st <= 3; ++ st) {
      for (lb = -3 * num_th * st; lb <= 3 * num_th * st; ++ lb) {
        for (ub = lb; ub < lb + num_th * (chunk+1) * st; ++ ub) {
          err += run_loop_32(lb, ub,  st, chunk);
          err += run_loop_32(ub, lb, -st, chunk);
        }; // for ub
      }; // for lb
    }; // for st
  }; // for chunk
 }
 return err;
} // run_all

// ---------------------------------------------------------------------------
int main()
{
  int n, err = 0;
  for (n = 1; n <= 4; ++ n) {
    err += run_32(n);
    err += run_64(n);
  }; // for n
  if (err)
    printf("failed with %d errors\n", err);
  else
    printf("passed\n");
  return err;
}
