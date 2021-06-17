/*
 * kmp_barrier.h
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KMP_BARRIER_H
#define KMP_BARRIER_H

#include "kmp.h"

// Use four cache lines: MLC tends to prefetch the next or previous cache line
// creating a possible fake conflict between cores, so this is the only way to
// guarantee that no such prefetch can happen.
#ifndef KMP_FOURLINE_ALIGN_CACHE
#define KMP_FOURLINE_ALIGN_CACHE KMP_ALIGN(4 * CACHE_LINE)
#endif

#define KMP_OPTIMIZE_FOR_REDUCTIONS 0

class distributedBarrier {
  struct flags_s {
    kmp_uint32 volatile KMP_FOURLINE_ALIGN_CACHE stillNeed;
  };

  struct go_s {
    std::atomic<kmp_uint64> KMP_FOURLINE_ALIGN_CACHE go;
  };

  struct iter_s {
    kmp_uint64 volatile KMP_FOURLINE_ALIGN_CACHE iter;
  };

  struct sleep_s {
    std::atomic<bool> KMP_FOURLINE_ALIGN_CACHE sleep;
  };

  void init(size_t nthr);
  void resize(size_t nthr);
  void computeGo(size_t n);
  void computeVarsForN(size_t n);

public:
  enum {
    MAX_ITERS = 3,
    MAX_GOS = 8,
    IDEAL_GOS = 4,
    IDEAL_CONTENTION = 16,
  };

  flags_s *flags[MAX_ITERS];
  go_s *go;
  iter_s *iter;
  sleep_s *sleep;

  size_t KMP_ALIGN_CACHE num_threads; // number of threads in barrier
  size_t KMP_ALIGN_CACHE max_threads; // size of arrays in data structure
  // number of go signals each requiring one write per iteration
  size_t KMP_ALIGN_CACHE num_gos;
  // number of groups of gos
  size_t KMP_ALIGN_CACHE num_groups;
  // threads per go signal
  size_t KMP_ALIGN_CACHE threads_per_go;
  bool KMP_ALIGN_CACHE fix_threads_per_go;
  // threads per group
  size_t KMP_ALIGN_CACHE threads_per_group;
  // number of go signals in a group
  size_t KMP_ALIGN_CACHE gos_per_group;
  void *team_icvs;

  distributedBarrier() = delete;
  ~distributedBarrier() = delete;

  // Used instead of constructor to create aligned data
  static distributedBarrier *allocate(int nThreads) {
    distributedBarrier *d = (distributedBarrier *)_mm_malloc(
        sizeof(distributedBarrier), 4 * CACHE_LINE);
    d->num_threads = 0;
    d->max_threads = 0;
    for (int i = 0; i < MAX_ITERS; ++i)
      d->flags[i] = NULL;
    d->go = NULL;
    d->iter = NULL;
    d->sleep = NULL;
    d->team_icvs = NULL;
    d->fix_threads_per_go = false;
    // calculate gos and groups ONCE on base size
    d->computeGo(nThreads);
    d->init(nThreads);
    return d;
  }

  static void deallocate(distributedBarrier *db) { _mm_free(db); }

  void update_num_threads(size_t nthr) { init(nthr); }

  bool need_resize(size_t new_nthr) { return (new_nthr > max_threads); }
  size_t get_num_threads() { return num_threads; }
  kmp_uint64 go_release();
  void go_reset();
};

#endif // KMP_BARRIER_H
