// RUN: %libomp-compile-and-run
// REQUIRES: linux
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Call the GNU ABI directly so this runtime test does not depend on the
// compiler accepting the OpenMP strict modifier or choosing the GNU ABI.
#include <omp.h>
#include <stdint.h>
#include <stdio.h>

enum { Capacity = 64 };
struct context {
  uint64_t start;
  int n, tasks, bad, seen[Capacity], sizes[Capacity];
};
struct signed_data { long lb, ub; struct context *c; };
struct unsigned_data { unsigned long long lb, ub; struct context *c; };
extern void GOMP_taskloop(void (*)(void *), void *, void (*)(void *, void *),
                         long, long, unsigned, unsigned long, int, long, long, long);
extern void GOMP_taskloop_ull(void (*)(void *), void *, void (*)(void *, void *),
                             long, long, unsigned, unsigned long, int,
                             unsigned long long, unsigned long long,
                             unsigned long long);

static void record(struct context *c, uint64_t lb, uint64_t ub) {
  uint64_t first = lb - c->start, size = ub - lb;
  int id = __atomic_fetch_add(&c->tasks, 1, __ATOMIC_RELAXED);
  if (id >= Capacity || first >= (uint64_t)c->n ||
      size == 0 || size > (uint64_t)c->n - first) {
    __atomic_fetch_add(&c->bad, 1, __ATOMIC_RELAXED);
    return;
  }
  c->sizes[id] = (int)size;
  for (uint64_t i = first; i < first + size; ++i)
    __atomic_fetch_add(&c->seen[i], 1, __ATOMIC_RELAXED);
}

static void signed_entry(void *p) {
  struct signed_data *d = p;
  record(d->c, d->lb, d->ub);
}
static void unsigned_entry(void *p) {
  struct unsigned_data *d = p;
  record(d->c, d->lb, d->ub);
}

static int run(int is_unsigned, int n, int grain, int sched, int strict,
               int if_val, int threads) {
  struct context c = {.start = is_unsigned ? (UINT64_C(1) << 63) + 17 : 7, .n = n};
  unsigned flags = (1u << 8) | (if_val ? 1u << 10 : 0) |
                   (sched == 1 ? 1u << 9 : 0) | (strict ? 1u << 14 : 0);
  #pragma omp parallel num_threads(threads) shared(c)
  {
    #pragma omp single
    {
      if (is_unsigned) {
        struct unsigned_data d = {0, 0, &c};
        GOMP_taskloop_ull(unsigned_entry, &d, NULL, sizeof(d),
                         _Alignof(struct unsigned_data), flags, grain, 0,
                         c.start, c.start + n, 1);
      } else {
        struct signed_data d = {0, 0, &c};
        GOMP_taskloop(signed_entry, &d, NULL, sizeof(d),
                     _Alignof(struct signed_data), flags, grain, 0,
                     (long)c.start, (long)c.start + n, 1);
      }
    }
  }
  int bad = c.bad;
  for (int i = 0; i < n; ++i)
    bad |= c.seen[i] != 1;
  int tasks = sched == 2 ? (grain < n ? grain : n) :
              strict ? (n + grain - 1) / grain : (n < grain ? 1 : n / grain);
  bad |= c.tasks != tasks;
  int tail_count = 0;
  for (int i = 0; i < c.tasks && i < Capacity; ++i) {
    if (sched == 1 && strict) {
      int tail = n % grain ? n % grain : grain;
      bad |= c.sizes[i] != grain && c.sizes[i] != tail;
      tail_count += c.sizes[i] != grain;
    } else {
      int small = n / tasks;
      bad |= c.sizes[i] != small && c.sizes[i] != small + 1;
    }
  }
  if (sched == 1 && strict)
    bad |= tail_count != (n % grain != 0);
  if (bad)
    printf("GNU strict failure: ull=%d n=%d grain=%d sched=%d strict=%d if=%d "
           "threads=%d tasks=%d/%d\n", is_unsigned, n, grain, sched, strict,
           if_val, threads, c.tasks, tasks);
  return !!bad;
}

int main(void) {
  const int counts[] = {1, 10, 11, 53, 60};
  const int grains[] = {3, 10, 64};
  int errors = 0, cases = 0;
  omp_set_dynamic(0);
  for (int threads = 1; threads <= 2; ++threads)
    for (int if_val = 0; if_val <= 1; ++if_val)
      for (int ull = 0; ull <= 1; ++ull)
        for (int n = 0; n < 5; ++n)
          for (int g = 0; g < 3; ++g)
            for (int sched = 1; sched <= 2; ++sched)
              for (int strict = 0; strict <= 1; ++strict) {
                errors += run(ull, counts[n], grains[g], sched, strict,
                              if_val, threads);
                ++cases;
              }
  printf("GNU strict: cases=%d errors=%d\n", cases, errors);
  return errors != 0;
}
