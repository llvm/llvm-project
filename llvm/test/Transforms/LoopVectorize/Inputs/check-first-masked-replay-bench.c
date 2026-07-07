// Benchmark: Scalar replay vs. Masked vector replay for check-first
// early-exit vectorization.
//
// Compile and run:
//   # Scalar replay (baseline):
//   clang -O2 -mllvm -enable-early-exit-vectorization \
//         -mllvm -enable-check-first-vectorization \
//         -o bench_scalar check-first-masked-replay-bench.c
//
//   # Masked replay:
//   clang -O2 -mllvm -enable-early-exit-vectorization \
//         -mllvm -enable-check-first-vectorization \
//         -mllvm -enable-check-first-masked-replay \
//         -o bench_masked check-first-masked-replay-bench.c
//
//   ./bench_scalar
//   ./bench_masked
//
// Expected: masked replay wins on short trip counts (N ~ 8-32) where
// the VF/2 scalar replay dominates.  On long trip counts the replay is
// amortized so the difference is smaller.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ALLOC_SIZE 4096
#define WARMUP_ITERS 100
#define BENCH_ITERS 100000

// memcpy-with-exit: copy bytes from src to dst until sentinel is found in chk.
// Returns the index of the sentinel (or N if not found).
__attribute__((noinline))
int64_t memcpy_exit(uint8_t *__restrict dst,
                    const uint8_t *__restrict src,
                    const uint8_t *__restrict chk,
                    int64_t N) {
  for (int64_t i = 0; i < N; i++) {
    dst[i] = src[i];
    if (chk[i] == 0)
      return i;
  }
  return N;
}

static double bench(int64_t N, int64_t exit_pos, int iters) {
  uint8_t *dst = (uint8_t *)aligned_alloc(64, ALLOC_SIZE);
  uint8_t *src = (uint8_t *)aligned_alloc(64, ALLOC_SIZE);
  uint8_t *chk = (uint8_t *)aligned_alloc(64, ALLOC_SIZE);

  memset(src, 0xAA, ALLOC_SIZE);
  memset(chk, 0xFF, ALLOC_SIZE);
  if (exit_pos < N && exit_pos < ALLOC_SIZE)
    chk[exit_pos] = 0;

  // Warmup
  volatile int64_t sink;
  for (int w = 0; w < WARMUP_ITERS; w++)
    sink = memcpy_exit(dst, src, chk, N);

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < iters; i++)
    sink = memcpy_exit(dst, src, chk, N);
  clock_gettime(CLOCK_MONOTONIC, &t1);

  double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
  free(dst);
  free(src);
  free(chk);
  return elapsed;
}

int main(void) {
  printf("%-12s %-12s %-12s %-12s\n", "N", "exit_pos", "time(ms)", "ns/iter");

  struct { int64_t N; int64_t exit_pos; const char *desc; } cases[] = {
    {    8,    3, "short, early exit" },
    {   16,    7, "short, mid exit"   },
    {   32,   15, "short, late exit"  },
    {   32,   32, "short, no exit"    },
    {  128,    5, "med, early exit"   },
    {  128,   64, "med, mid exit"     },
    {  128,  128, "med, no exit"      },
    { 1024,   10, "long, early exit"  },
    { 1024,  512, "long, mid exit"    },
    { 1024, 1024, "long, no exit"     },
  };
  int ncases = sizeof(cases) / sizeof(cases[0]);

  for (int c = 0; c < ncases; c++) {
    double t = bench(cases[c].N, cases[c].exit_pos, BENCH_ITERS);
    double ms = t * 1000.0;
    double ns_per = t * 1e9 / BENCH_ITERS;
    printf("%-12ld %-12ld %-12.3f %-12.1f  %s\n",
           (long)cases[c].N, (long)cases[c].exit_pos,
           ms, ns_per, cases[c].desc);
  }

  return 0;
}
