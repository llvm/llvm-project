/*
 * kmp_taskdeps.h
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KMP_TASKDEPS_H
#define KMP_TASKDEPS_H

#include "kmp.h"

#define KMP_ACQUIRE_DEPNODE(gtid, n) __kmp_acquire_lock(&(n)->dn.lock, (gtid))
#define KMP_RELEASE_DEPNODE(gtid, n) __kmp_release_lock(&(n)->dn.lock, (gtid))

static inline void __kmp_node_deref(kmp_info_t *thread, kmp_depnode_t *node) {
  if (!node)
    return;

  kmp_int32 n = KMP_ATOMIC_DEC(&node->dn.nrefs) - 1;
  if (n == 0) {
    KMP_ASSERT(node->dn.nrefs == 0);
#if USE_FAST_MEMORY
    __kmp_fast_free(thread, node);
#else
    __kmp_thread_free(thread, node);
#endif
  }
}

static inline void __kmp_depnode_list_free(kmp_info_t *thread,
                                           kmp_depnode_list *list) {
  kmp_depnode_list *next;

  for (; list; list = next) {
    next = list->next;

    __kmp_node_deref(thread, list->node);
#if USE_FAST_MEMORY
    __kmp_fast_free(thread, list);
#else
    __kmp_thread_free(thread, list);
#endif
  }
}

static inline void __kmp_dephash_free_entries(kmp_info_t *thread,
                                              kmp_dephash_t *h) {
  for (size_t i = 0; i < h->size; i++) {
    if (h->buckets[i]) {
      kmp_dephash_entry_t *next;
      for (kmp_dephash_entry_t *entry = h->buckets[i]; entry; entry = next) {
        next = entry->next_in_bucket;
        __kmp_depnode_list_free(thread, entry->last_ins);
        __kmp_depnode_list_free(thread, entry->last_mtxs);
        __kmp_node_deref(thread, entry->last_out);
        if (entry->mtx_lock) {
          __kmp_destroy_lock(entry->mtx_lock);
          __kmp_free(entry->mtx_lock);
        }
#if USE_FAST_MEMORY
        __kmp_fast_free(thread, entry);
#else
        __kmp_thread_free(thread, entry);
#endif
      }
      h->buckets[i] = 0;
    }
  }
}

static inline void __kmp_dephash_free(kmp_info_t *thread, kmp_dephash_t *h) {
  __kmp_dephash_free_entries(thread, h);
#if USE_FAST_MEMORY
  __kmp_fast_free(thread, h);
#else
  __kmp_thread_free(thread, h);
#endif
}

static inline void __kmp_release_deps(kmp_int32 gtid, kmp_taskdata_t *task) {
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_depnode_t *node = task->td_depnode;

  // Check mutexinoutset dependencies, release locks
  if (UNLIKELY(node && (node->dn.mtx_num_locks < 0))) {
    // negative num_locks means all locks were acquired
    node->dn.mtx_num_locks = -node->dn.mtx_num_locks;
    for (int i = node->dn.mtx_num_locks - 1; i >= 0; --i) {
      KMP_DEBUG_ASSERT(node->dn.mtx_locks[i] != NULL);
      __kmp_release_lock(node->dn.mtx_locks[i], gtid);
    }
  }

  if (task->td_dephash) {
    KA_TRACE(
        40, ("__kmp_release_deps: T#%d freeing dependencies hash of task %p.\n",
             gtid, task));
    __kmp_dephash_free(thread, task->td_dephash);
    task->td_dephash = NULL;
  }

  if (!node)
    return;

  KA_TRACE(20, ("__kmp_release_deps: T#%d notifying successors of task %p.\n",
                gtid, task));

  KMP_ACQUIRE_DEPNODE(gtid, node);
  node->dn.task =
      NULL; // mark this task as finished, so no new dependencies are generated
  KMP_RELEASE_DEPNODE(gtid, node);

  kmp_depnode_list_t *next;
  kmp_taskdata_t *next_taskdata;
  for (kmp_depnode_list_t *p = node->dn.successors; p; p = next) {
    kmp_depnode_t *successor = p->node;
    kmp_int32 npredecessors = KMP_ATOMIC_DEC(&successor->dn.npredecessors) - 1;

    // successor task can be NULL for wait_depends or because deps are still
    // being processed
    if (npredecessors == 0) {
      KMP_MB();
      if (successor->dn.task) {
        KA_TRACE(20, ("__kmp_release_deps: T#%d successor %p of %p scheduled "
                      "for execution.\n",
                      gtid, successor->dn.task, task));
        // If a regular task depending on a hidden helper task, when the
        // hidden helper task is done, the regular task should be executed by
        // its encountering team.
        if (KMP_HIDDEN_HELPER_THREAD(gtid)) {
          // Hidden helper thread can only execute hidden helper tasks
          KMP_ASSERT(task->td_flags.hidden_helper);
          next_taskdata = KMP_TASK_TO_TASKDATA(successor->dn.task);
          // If the dependent task is a regular task, we need to push to its
          // encountering thread's queue; otherwise, it can be pushed to its own
          // queue.
          if (!next_taskdata->td_flags.hidden_helper) {
            __kmp_omp_task(task->encountering_gtid, successor->dn.task, false);
          } else {
            __kmp_omp_task(gtid, successor->dn.task, false);
          }
        } else {
          __kmp_omp_task(gtid, successor->dn.task, false);
        }
      }
    }

    next = p->next;
    __kmp_node_deref(thread, p->node);
#if USE_FAST_MEMORY
    __kmp_fast_free(thread, p);
#else
    __kmp_thread_free(thread, p);
#endif
  }

  __kmp_node_deref(thread, node);

  KA_TRACE(
      20,
      ("__kmp_release_deps: T#%d all successors of %p notified of completion\n",
       gtid, task));
}

#endif // KMP_TASKDEPS_H
