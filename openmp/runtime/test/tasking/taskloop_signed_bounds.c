// RUN: %libomp-compile
// RUN: env KMP_TASKLOOP_MIN_TASKS=1 %libomp-run
// RUN: env KMP_TASKLOOP_MIN_TASKS=64 %libomp-run
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exercise the runtime ABI directly, as in kmp_taskloop_5.c. Clang normally
// normalizes source loop coordinates, hiding these signed/descending bounds.
#include <omp.h>
#include <stdint.h>
#include <stdio.h>

enum { Capacity = 64 };
struct context {
  uint64_t start;
  int64_t step;
  int count, tasks, bad, last_count, last_index;
  int seen[Capacity], sizes[Capacity];
};
struct task {
  void *shareds;
  int (*routine)(int, struct task *);
  int part_id;
  uint64_t lb, ub;
  int last;
};
extern int __kmpc_global_thread_num(void *);
extern struct task *__kmpc_omp_task_alloc(void *, int, int, size_t, size_t,
                                          int (*)(int, struct task *));
extern void __kmpc_taskloop_5(void *, int, struct task *, int, uint64_t *,
                              uint64_t *, int64_t, int, int, uint64_t, int,
                              void *);

static void duplicate(struct task *dest, struct task *source, int last) {
  (void)source;
  dest->last = last;
}

static int execute(int gtid, struct task *task) {
  (void)gtid;
  struct context *c = *(struct context **)task->shareds;
  uint64_t stride = c->step > 0 ? (uint64_t)c->step : (uint64_t)-c->step;
  uint64_t delta = c->step > 0 ? task->lb - c->start : c->start - task->lb;
  uint64_t span = c->step > 0 ? task->ub - task->lb : task->lb - task->ub;
  uint64_t first = delta / stride, size = span / stride + 1;
  int id = __atomic_fetch_add(&c->tasks, 1, __ATOMIC_RELAXED);
  // Guard before iterating: a broken runtime can supply an enormous span.
  if (id >= Capacity || delta % stride || span % stride ||
      first >= (uint64_t)c->count || size > (uint64_t)c->count - first) {
    __atomic_fetch_add(&c->bad, 1, __ATOMIC_RELAXED);
    return 0;
  }
  c->sizes[id] = (int)size;
  for (uint64_t i = first; i < first + size; ++i)
    __atomic_fetch_add(&c->seen[i], 1, __ATOMIC_RELAXED);
  if (task->last) {
    __atomic_fetch_add(&c->last_count, 1, __ATOMIC_RELAXED);
    __atomic_store_n(&c->last_index, (int)(first + size - 1), __ATOMIC_RELAXED);
  }
  return 0;
}

static int run(uint64_t start, int64_t step, int n, int grain, int sched,
               int strict, int if_val, int threads, int unaligned) {
  struct context c = {
      .start = start, .step = step, .count = n, .last_index = -1};
  // Inclusive terminal may be off the stride lattice. Keep signed arithmetic
  // out of coordinate construction; the runtime ABI uses uint64_t bounds.
  uint64_t terminal = start + (uint64_t)step * (uint64_t)(n - 1);
  if (n && unaligned)
    terminal += step > 0 ? 1 : (uint64_t)-1;
#pragma omp parallel num_threads(threads) shared(c)
  {
#pragma omp single
    {
      if (omp_get_num_threads() != threads)
        ++c.bad;
      int gtid = __kmpc_global_thread_num(NULL);
      struct task *t = __kmpc_omp_task_alloc(NULL, gtid, 1, sizeof(*t),
                                             sizeof(struct context *), execute);
      *(struct context **)t->shareds = &c;
      t->lb = start;
      t->ub = terminal;
      t->last = 0;
      __kmpc_taskloop_5(NULL, gtid, t, if_val, &t->lb, &t->ub, step, 0, sched,
                        grain, strict, (void *)duplicate);
    }
  }
  if (!n)
    return c.tasks != 0 || c.last_count != 0 || c.bad != 0;
  int tasks = sched == 2 ? (grain < n ? grain : n)
              : strict   ? (n + grain - 1) / grain
                         : (n < grain ? 1 : n / grain);
  int bad =
      c.bad || c.tasks != tasks || c.last_count != 1 || c.last_index != n - 1;
  for (int i = 0; i < n; ++i)
    bad |= c.seen[i] != 1;
  int small = n / tasks, large_count = n % tasks, actual_large = 0;
  for (int i = 0; i < c.tasks && i < Capacity; ++i) {
    if (sched == 1 && strict) {
      int tail = n % grain ? n % grain : grain;
      bad |= c.sizes[i] != grain && c.sizes[i] != tail;
    } else {
      bad |= c.sizes[i] != small && c.sizes[i] != small + 1;
      actual_large += c.sizes[i] == small + 1;
    }
  }
  if (sched != 1 || !strict)
    bad |= actual_large != large_count;
  if (bad)
    printf(
        "bounds failure: start=%llu step=%lld n=%d grain=%d sched=%d strict=%d "
        "if=%d threads=%d tasks=%d/%d bad=%d last=%d/%d\n",
        (unsigned long long)start, (long long)step, n, grain, sched, strict,
        if_val, threads, c.tasks, tasks, c.bad, c.last_count, c.last_index);
  return !!bad;
}

int main(void) {
  const struct {
    uint64_t start;
    int64_t step;
    int count, grain, sched, strict, unaligned;
  } cases[] = {
      {17, -3, 53, 10, 1, 1, 1}, // Descending strict tail, crossing zero.
      {(uint64_t)-17, 1, 53, 10, 1, 1, 0}, // Ascending strict tail across zero.
      {(uint64_t)-17, 1, 53, 10, 1, 0, 0}, // Non-strict cross-zero loop.
      {1000, -3, 53, 10, 2, 0, 1}, // Descending num_tasks, off-stride bound.
      {(uint64_t)-200, 3, 53, 10, 1, 1, 1}, // Negative non-unit coordinates.
      {(UINT64_C(1) << 63) + 17, 3, 53, 10, 1, 1, 1}, // High unsigned bounds.
      {7, 1, 60, 10, 1, 1, 0}, // Exact grainsize control.
      {7, 1, 1, 10, 1, 1, 0}, // Single-iteration control.
      {7, 1, 0, 10, 1, 1, 0}, // Zero-trip control.
      {17, -3, 60, 10, 1, 1, 1}, // Exact descending loop.
      {17, -3, 53, 10, 1, 0, 1}, // Non-strict descending loop.
      {(uint64_t)-17, 1, 53, 10, 2, 1, 0}, // Strict num_tasks across zero.
  };
  int errors = 0, checked = 0;
  omp_set_dynamic(0);
  for (int threads = 1; threads <= 2; ++threads)
    for (int if_val = 0; if_val <= 1; ++if_val)
      for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        errors += run(cases[i].start, cases[i].step, cases[i].count,
                      cases[i].grain, cases[i].sched, cases[i].strict, if_val,
                      threads, cases[i].unaligned);
        ++checked;
      }
  printf("taskloop bounds: cases=%d errors=%d\n", checked, errors);
  return errors != 0;
}
