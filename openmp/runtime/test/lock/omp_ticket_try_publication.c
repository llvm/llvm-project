// RUN: %libomp-compile
// RUN: env KMP_LOCK_KIND=ticket %libomp-run
// RUN: env KMP_LOCK_KIND=ticket KMP_CONSISTENCY_CHECK=all %libomp-run
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <omp.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

// The relaxed handshake arranges a failed try followed by a successful try.
// It must not provide the payload's release/acquire synchronization itself.
int main(void) {
  omp_lock_t lock;
  omp_nest_lock_t nested;
  atomic_int ready = 0, checked = 0;
  uint64_t payload = 0, count = 0, nested_count = 0;
  int errors = 0, team = 0;
  omp_set_dynamic(0);
  omp_init_lock(&lock);
  omp_init_nest_lock(&nested);

#pragma omp parallel num_threads(2) reduction(+ : errors) shared(team)
  {
    int tid = omp_get_thread_num();
#pragma omp single
    team = omp_get_num_threads();
    if (team == 2 && tid == 0) {
      omp_set_lock(&lock);
      payload = UINT64_C(0x12345678);
      atomic_store_explicit(&ready, 1, memory_order_relaxed);
      while (!atomic_load_explicit(&checked, memory_order_relaxed)) {
      }
      omp_unset_lock(&lock);
    } else if (team == 2 && tid == 1) {
      while (!atomic_load_explicit(&ready, memory_order_relaxed)) {
      }
      int acquired = omp_test_lock(&lock);
      errors += acquired != 0;
      if (acquired)
        omp_unset_lock(&lock);
      atomic_store_explicit(&checked, 1, memory_order_relaxed);
      while (!omp_test_lock(&lock)) {
      }
      errors += payload != UINT64_C(0x12345678);
      omp_unset_lock(&lock);
    }

#pragma omp barrier
#pragma omp single
    payload = 0;
    for (int i = 0; i < 512; ++i) {
      if (tid & 1)
        omp_set_lock(&lock);
      else
        while (!omp_test_lock(&lock)) {
        }
      errors += payload != count * UINT64_C(37);
      ++count;
      payload = count * UINT64_C(37);
      omp_unset_lock(&lock);
    }
    for (int i = 0; i < 128; ++i) {
      omp_set_nest_lock(&nested);
      errors += omp_test_nest_lock(&nested) != 2;
      omp_unset_nest_lock(&nested);
      ++nested_count;
      omp_unset_nest_lock(&nested);
    }
  }

  errors += team != 2 || count != (uint64_t)team * 512 ||
            nested_count != (uint64_t)team * 128;
  omp_destroy_nest_lock(&nested);
  omp_destroy_lock(&lock);
  printf("TICKET_TRY_PUBLICATION team=%d count=%llu nested=%llu errors=%d\n",
         team, (unsigned long long)count, (unsigned long long)nested_count, errors);
  return errors != 0;
}
