/*
 * kmp_taskdeps.cpp
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//#define KMP_SUPPORT_GRAPH_OUTPUT 1

#include "kmp.h"
#include "kmp_io.h"
#include "kmp_wait_release.h"
#include "kmp_taskdeps.h"
#if OMPT_SUPPORT
#include "ompt-specific.h"
#endif

#if OMP_TASKGRAPH_EXPERIMENTAL
#include <bit>
#include <cstdlib>
#include <algorithm>
#include <cinttypes>
#endif

// TODO: Improve memory allocation? keep a list of pre-allocated structures?
// allocate in blocks? re-use list finished list entries?
// TODO: don't use atomic ref counters for stack-allocated nodes.
// TODO: find an alternate to atomic refs for heap-allocated nodes?
// TODO: Finish graph output support
// TODO: kmp_lock_t seems a tad to big (and heavy weight) for this. Check other
// runtime locks
// TODO: Any ITT support needed?

#ifdef KMP_SUPPORT_GRAPH_OUTPUT
static std::atomic<kmp_int32> kmp_node_id_seed = 0;
#endif

#undef DEBUG_TASKGRAPH

#ifdef DEBUG_TASKGRAPH
#define TGDBG(ARGS...) fprintf(stderr, ARGS)
#else
#define TGDBG(ARGS...)
#endif

static void __kmp_init_node(kmp_depnode_t *node, bool on_stack) {
  node->dn.successors = NULL;
  node->dn.task = NULL; // will point to the right task
  // once dependences have been processed
  for (int i = 0; i < MAX_MTX_DEPS; ++i)
    node->dn.mtx_locks[i] = NULL;
  node->dn.mtx_num_locks = 0;
  __kmp_init_lock(&node->dn.lock);
  // Init creates the first reference.  Bit 0 indicates that this node
  // resides on the stack.  The refcount is incremented and decremented in
  // steps of two, maintaining use of even numbers for heap nodes and odd
  // numbers for stack nodes.
  KMP_ATOMIC_ST_RLX(&node->dn.nrefs, on_stack ? 3 : 2);
#ifdef KMP_SUPPORT_GRAPH_OUTPUT
  node->dn.id = KMP_ATOMIC_INC(&kmp_node_id_seed);
#endif
#if OMP_TASKGRAPH_EXPERIMENTAL
  node->dn.set_membership = nullptr;
#endif
#if USE_ITT_BUILD && USE_ITT_NOTIFY
  __itt_sync_create(node, "OMP task dep node", NULL, 0);
#endif
}

static inline kmp_depnode_t *__kmp_node_ref(kmp_depnode_t *node) {
  KMP_ATOMIC_ADD(&node->dn.nrefs, 2);
  return node;
}

enum { KMP_DEPHASH_OTHER_SIZE = 97, KMP_DEPHASH_MASTER_SIZE = 997 };

size_t sizes[] = {997, 2003, 4001, 8191, 16001, 32003, 64007, 131071, 270029};
const size_t MAX_GEN = 8;

static inline size_t __kmp_dephash_hash(kmp_intptr_t addr, size_t hsize) {
  // TODO alternate to try: set = (((Addr64)(addrUsefulBits * 9.618)) %
  // m_num_sets );
  return ((addr >> 6) ^ (addr >> 2)) % hsize;
}

static kmp_dephash_t *__kmp_dephash_extend(kmp_info_t *thread,
                                           kmp_dephash_t *current_dephash) {
  kmp_dephash_t *h;

  size_t gen = current_dephash->generation + 1;
  if (gen >= MAX_GEN)
    return current_dephash;
  size_t new_size = sizes[gen];

  size_t size_to_allocate =
      new_size * sizeof(kmp_dephash_entry_t *) + sizeof(kmp_dephash_t);

#if USE_FAST_MEMORY
  h = (kmp_dephash_t *)__kmp_fast_allocate(thread, size_to_allocate);
#else
  h = (kmp_dephash_t *)__kmp_thread_malloc(thread, size_to_allocate);
#endif

  h->size = new_size;
  h->nelements = current_dephash->nelements;
  h->buckets = (kmp_dephash_entry **)(h + 1);
  h->generation = gen;
  h->nconflicts = 0;
  h->last_all = current_dephash->last_all;

  // make sure buckets are properly initialized
  for (size_t i = 0; i < new_size; i++) {
    h->buckets[i] = NULL;
  }

  // insert existing elements in the new table
  for (size_t i = 0; i < current_dephash->size; i++) {
    kmp_dephash_entry_t *next, *entry;
    for (entry = current_dephash->buckets[i]; entry; entry = next) {
      next = entry->next_in_bucket;
      // Compute the new hash using the new size, and insert the entry in
      // the new bucket.
      size_t new_bucket = __kmp_dephash_hash(entry->addr, h->size);
      entry->next_in_bucket = h->buckets[new_bucket];
      if (entry->next_in_bucket) {
        h->nconflicts++;
      }
      h->buckets[new_bucket] = entry;
    }
  }

  // Free old hash table
#if USE_FAST_MEMORY
  __kmp_fast_free(thread, current_dephash);
#else
  __kmp_thread_free(thread, current_dephash);
#endif

  return h;
}

static kmp_dephash_t *__kmp_dephash_create(kmp_info_t *thread,
                                           kmp_taskdata_t *current_task) {
  kmp_dephash_t *h;

  size_t h_size;

  if (current_task->td_flags.tasktype == TASK_IMPLICIT)
    h_size = KMP_DEPHASH_MASTER_SIZE;
  else
    h_size = KMP_DEPHASH_OTHER_SIZE;

  size_t size = h_size * sizeof(kmp_dephash_entry_t *) + sizeof(kmp_dephash_t);

#if USE_FAST_MEMORY
  h = (kmp_dephash_t *)__kmp_fast_allocate(thread, size);
#else
  h = (kmp_dephash_t *)__kmp_thread_malloc(thread, size);
#endif
  h->size = h_size;

  h->generation = 0;
  h->nelements = 0;
  h->nconflicts = 0;
  h->buckets = (kmp_dephash_entry **)(h + 1);
  h->last_all = NULL;

  for (size_t i = 0; i < h_size; i++)
    h->buckets[i] = 0;

  return h;
}

static kmp_dephash_entry *__kmp_dephash_find(kmp_info_t *thread,
                                             kmp_dephash_t **hash,
                                             kmp_intptr_t addr
#if OMP_TASKGRAPH_EXPERIMENTAL
                                             ,
                                             bool taskgraph_p
#endif
) {
  kmp_dephash_t *h = *hash;
  if (h->nelements != 0 && h->nconflicts / h->size >= 1) {
    *hash = __kmp_dephash_extend(thread, h);
    h = *hash;
  }
  size_t bucket = __kmp_dephash_hash(addr, h->size);

  kmp_dephash_entry_t *entry;
  for (entry = h->buckets[bucket]; entry; entry = entry->next_in_bucket)
    if (entry->addr == addr)
      break;

  if (entry == NULL) {
// create entry. This is only done by one thread so no locking required
#if USE_FAST_MEMORY
    entry = (kmp_dephash_entry_t *)__kmp_fast_allocate(
        thread, sizeof(kmp_dephash_entry_t));
#else
    entry = (kmp_dephash_entry_t *)__kmp_thread_malloc(
        thread, sizeof(kmp_dephash_entry_t));
#endif
    entry->addr = addr;
    if (!h->last_all) // no predecessor task with omp_all_memory dependence
      entry->last_out = NULL;
    else // else link the omp_all_memory depnode to the new entry
      entry->last_out = __kmp_node_ref(h->last_all);
    entry->last_set = NULL;
    entry->prev_set = NULL;
    entry->last_flag = 0;
#if OMP_TASKGRAPH_EXPERIMENTAL
    if (taskgraph_p)
      entry->set_num = -1;
    else
#endif
      entry->mtx_lock = NULL;
    entry->next_in_bucket = h->buckets[bucket];
    h->buckets[bucket] = entry;
    h->nelements++;
    if (entry->next_in_bucket)
      h->nconflicts++;
  }
  return entry;
}

template <bool refcounting>
static kmp_depnode_list_t *__kmp_add_node(kmp_info_t *thread,
                                          kmp_depnode_list_t *list,
                                          kmp_depnode_t *node) {
  kmp_depnode_list_t *new_head;

#if USE_FAST_MEMORY
  new_head = (kmp_depnode_list_t *)__kmp_fast_allocate(
      thread, sizeof(kmp_depnode_list_t));
#else
  new_head = (kmp_depnode_list_t *)__kmp_thread_malloc(
      thread, sizeof(kmp_depnode_list_t));
#endif

  if (refcounting) {
    new_head->node = __kmp_node_ref(node);
  } else {
    new_head->node = node;
  }
  new_head->next = list;

  return new_head;
}

static inline void __kmp_track_dependence(kmp_int32 gtid, kmp_depnode_t *source,
                                          kmp_depnode_t *sink,
                                          kmp_task_t *sink_task) {
#ifdef KMP_SUPPORT_GRAPH_OUTPUT
  kmp_taskdata_t *task_source = KMP_TASK_TO_TASKDATA(source->dn.task);
  // do not use sink->dn.task as that is only filled after the dependences
  // are already processed!
  kmp_taskdata_t *task_sink = KMP_TASK_TO_TASKDATA(sink_task);

  __kmp_printf("%d(%s) -> %d(%s)\n", source->dn.id,
               task_source->td_ident->psource, sink->dn.id,
               task_sink->td_ident->psource);
#endif
#if OMPT_SUPPORT && OMPT_OPTIONAL
  /* OMPT tracks dependences between task (a=source, b=sink) in which
     task a blocks the execution of b through the ompt_new_dependence_callback
     */
  if (ompt_enabled.ompt_callback_task_dependence) {
    kmp_taskdata_t *task_source = KMP_TASK_TO_TASKDATA(source->dn.task);
    ompt_data_t *sink_data;
    if (sink_task)
      sink_data = &(KMP_TASK_TO_TASKDATA(sink_task)->ompt_task_info.task_data);
    else
      sink_data = &__kmp_threads[gtid]->th.ompt_thread_info.task_data;

    ompt_callbacks.ompt_callback(ompt_callback_task_dependence)(
        &(task_source->ompt_task_info.task_data), sink_data);
  }
#endif /* OMPT_SUPPORT && OMPT_OPTIONAL */
}

kmp_base_depnode_t *__kmpc_task_get_depnode(kmp_task_t *task) {
  kmp_taskdata_t *td = KMP_TASK_TO_TASKDATA(task);
  return td->td_depnode ? &(td->td_depnode->dn) : NULL;
}

kmp_depnode_list_t *__kmpc_task_get_successors(kmp_task_t *task) {
  kmp_taskdata_t *td = KMP_TASK_TO_TASKDATA(task);
  return td->td_depnode->dn.successors;
}

static inline kmp_int32
__kmp_depnode_link_successor(kmp_int32 gtid, kmp_info_t *thread,
                             kmp_task_t *task, kmp_depnode_t *node,
                             kmp_depnode_list_t *plist) {
  if (!plist)
    return 0;
  kmp_int32 npredecessors = 0;
  // link node as successor of list elements
  for (kmp_depnode_list_t *p = plist; p; p = p->next) {
    kmp_depnode_t *dep = p->node;
    if (dep->dn.task) {
      KMP_ACQUIRE_DEPNODE(gtid, dep);
      if (dep->dn.task) {
        if (!dep->dn.successors || dep->dn.successors->node != node) {
          __kmp_track_dependence(gtid, dep, node, task);
          dep->dn.successors =
              __kmp_add_node<true>(thread, dep->dn.successors, node);
          KA_TRACE(40, ("__kmp_process_deps: T#%d adding dependence from %p to "
                        "%p\n",
                        gtid, KMP_TASK_TO_TASKDATA(dep->dn.task),
                        KMP_TASK_TO_TASKDATA(task)));
          npredecessors++;
        }
      }
      KMP_RELEASE_DEPNODE(gtid, dep);
    }
  }
  return npredecessors;
}

// Add the edge 'sink' -> 'source' in the task dependency graph
static inline kmp_int32 __kmp_depnode_link_successor(kmp_int32 gtid,
                                                     kmp_info_t *thread,
                                                     kmp_task_t *task,
                                                     kmp_depnode_t *source,
                                                     kmp_depnode_t *sink) {
  if (!sink)
    return 0;
  kmp_int32 npredecessors = 0;
  if (sink->dn.task) {
    // synchronously add source to sink' list of successors
    KMP_ACQUIRE_DEPNODE(gtid, sink);
    if (sink->dn.task) {
      if (!sink->dn.successors || sink->dn.successors->node != source) {
        __kmp_track_dependence(gtid, sink, source, task);
        sink->dn.successors =
            __kmp_add_node<true>(thread, sink->dn.successors, source);
        KA_TRACE(40, ("__kmp_process_deps: T#%d adding dependence from %p to "
                    "%p\n",
                    gtid, KMP_TASK_TO_TASKDATA(sink->dn.task),
                    KMP_TASK_TO_TASKDATA(task)));
        npredecessors++;
      }
    }
    KMP_RELEASE_DEPNODE(gtid, sink);
  }
  return npredecessors;
}

#if OMP_TASKGRAPH_EXPERIMENTAL
kmp_taskgraph_region_dep_t *__kmp_region_deplist_add(
    kmp_info_t *thread, kmp_taskgraph_region_dep_t **recycled_deps,
    kmp_taskgraph_region_t *region, kmp_taskgraph_region_dep_t *list) {
  kmp_taskgraph_region_dep_t *head;
  if (*recycled_deps) {
    head = *recycled_deps;
    *recycled_deps = (*recycled_deps)->next;
  } else
    head = (kmp_taskgraph_region_dep_t *)__kmp_fast_allocate(
        thread, sizeof(kmp_taskgraph_region_dep_t));
  head->region = region;
  head->next = list;
  return head;
}

kmp_taskgraph_region_t *
__kmp_region_worklist_reverse(kmp_taskgraph_region_t *list) {
  kmp_taskgraph_region_t *last = nullptr;
  while (list) {
    kmp_taskgraph_region_t *next = list->next;
    list->next = last;
    last = list;
    list = next;
  }
  return last;
}

static kmp_depnode_t *__kmp_find_in_depnode_list(kmp_depnode_t *node,
                                                 kmp_depnode_list_t *list) {
  for (; list; list = list->next)
    if (list->node == node)
      return list->node;
  return nullptr;
}

// A trivial fixed-size bitset implementation.

typedef struct kmp_bitset {
  kmp_uint64 *bits;
  kmp_size_t bitsize;
  kmp_size_t num_chunks;
} kmp_bitset_t;

static kmp_bitset_t *__kmp_bitset_alloc(kmp_info_t *thread,
                                        kmp_size_t bitsize) {
  kmp_size_t bytesize = (bitsize + 7) / 8;
  kmp_size_t num_chunks =
      (bytesize + sizeof(kmp_uint64) - 1) / sizeof(kmp_uint64);
  kmp_bitset_t *bitset = (kmp_bitset_t *)__kmp_fast_allocate(
      thread, sizeof(kmp_bitset_t) + sizeof(kmp_uint64) * num_chunks);
  bitset->bits = (kmp_uint64 *)&bitset[1];
  memset(bitset->bits, 0, sizeof(kmp_uint64) * num_chunks);
  bitset->bitsize = bitsize;
  bitset->num_chunks = num_chunks;
  return bitset;
}

static void __kmp_bitset_free(kmp_info_t *thread, kmp_bitset_t *bitset) {
  __kmp_fast_free(thread, bitset);
}

static void __kmp_bitset_set(kmp_bitset_t *bitset, kmp_size_t bitnum) {
  kmp_size_t chunk = bitnum / (8 * sizeof(kmp_uint64));
  if (bitnum < bitset->bitsize)
    bitset->bits[chunk] |= (kmp_uint64)1 << (bitnum & 63);
}

static void __kmp_bitset_clearall(kmp_bitset_t *bitset) {
  if (bitset)
    memset(bitset->bits, 0, sizeof(kmp_int64) * bitset->num_chunks);
}

static void __kmp_bitset_setall(kmp_bitset_t *bitset) {
  for (kmp_int32 chunk = 0; chunk < bitset->num_chunks - 1; chunk++)
    bitset->bits[chunk] = ~(kmp_uint64)0;
  kmp_int32 last_chunk_numbits = bitset->bitsize & 63;
  if (last_chunk_numbits > 0) {
    kmp_uint64 last_chunk_bits = ~((~(kmp_uint64)0) << last_chunk_numbits);
    bitset->bits[bitset->num_chunks - 1] = last_chunk_bits;
  }
}

static void __kmp_bitset_copy(kmp_bitset_t *dst, const kmp_bitset_t *src) {
  assert(dst->num_chunks == src->num_chunks);
  assert(dst->bitsize == src->bitsize);
  memcpy(dst->bits, src->bits, sizeof(kmp_uint64) * dst->num_chunks);
}

/// Return TRUE if \c b is a subset of \c a.

static bool __kmp_bitset_subset_p(const kmp_bitset_t *a,
                                  const kmp_bitset_t *b) {
  if (!b)
    return true;
  kmp_size_t chunk_max = std::max(a->num_chunks, b->num_chunks);
  for (kmp_size_t chunk = 0; chunk < chunk_max; chunk++) {
    kmp_uint64 a_bits = chunk < a->num_chunks ? a->bits[chunk] : 0;
    kmp_uint64 b_bits = chunk < b->num_chunks ? b->bits[chunk] : 0;
    if ((a_bits & b_bits) != b_bits)
      return false;
  }
  return true;
}

static void __kmp_bitset_and(kmp_bitset_t *a, kmp_bitset_t *b,
                             kmp_bitset_t *c) {
  kmp_size_t chunk_max = std::max(b->num_chunks, c->num_chunks);
  for (kmp_size_t chunk = 0; chunk < chunk_max; chunk++) {
    kmp_uint64 b_bits = chunk < b->num_chunks ? b->bits[chunk] : 0;
    kmp_uint64 c_bits = chunk < c->num_chunks ? c->bits[chunk] : 0;
    a->bits[chunk] = b_bits & c_bits;
  }
}

static void __kmp_bitset_and_not(kmp_bitset_t *a, kmp_bitset_t *b,
                                 kmp_bitset_t *c) {
  if (!c)
    __kmp_bitset_copy(a, b);
  else {
    kmp_size_t chunk_max = std::max(b->num_chunks, c->num_chunks);
    for (kmp_size_t chunk = 0; chunk < chunk_max; chunk++) {
      kmp_uint64 b_bits = chunk < b->num_chunks ? b->bits[chunk] : 0;
      kmp_uint64 c_bits = chunk < c->num_chunks ? c->bits[chunk] : 0;
      a->bits[chunk] = b_bits & ~c_bits;
    }
  }
}

static void __kmp_bitset_or(kmp_bitset_t *a, kmp_bitset_t *b, kmp_bitset_t *c) {
  if (!b && !c)
    __kmp_bitset_clearall(a);
  else if (!b)
    __kmp_bitset_copy(a, c);
  else if (!c)
    __kmp_bitset_copy(a, b);
  else {
    kmp_size_t chunk_max = std::max(b->num_chunks, c->num_chunks);
    for (kmp_size_t chunk = 0; chunk < chunk_max; chunk++) {
      kmp_uint64 b_bits = chunk < b->num_chunks ? b->bits[chunk] : 0;
      kmp_uint64 c_bits = chunk < c->num_chunks ? c->bits[chunk] : 0;
      a->bits[chunk] = b_bits | c_bits;
    }
  }
}

static bool __kmp_bitset_empty_p(kmp_bitset_t *bitset) {
  if (!bitset)
    return true;
  for (kmp_size_t chunk = 0; chunk < bitset->num_chunks; chunk++) {
    if (bitset->bits[chunk] != 0)
      return false;
  }
  return true;
}

/// Test two bitsets for equality.  Note that any unused bits at the end of the
/// last chunk are kept as zero.

static bool __kmp_bitset_equal(kmp_bitset_t *a, kmp_bitset_t *b) {
  if (!b)
    return __kmp_bitset_empty_p(a);
  kmp_size_t chunk_max = std::max(a->num_chunks, b->num_chunks);
  for (kmp_size_t chunk = 0; chunk < chunk_max; chunk++) {
    kmp_uint64 a_bits = chunk < a->num_chunks ? a->bits[chunk] : 0;
    kmp_uint64 b_bits = chunk < b->num_chunks ? b->bits[chunk] : 0;
    if (a_bits != b_bits)
      return false;
  }
  return true;
}

static bool __kmp_bitset_intersect_p(kmp_bitset_t *a, kmp_bitset_t *b) {
  if (!a || !b)
    return false;
  kmp_size_t chunk_max = std::max(a->num_chunks, b->num_chunks);
  for (kmp_size_t chunk = 0; chunk < chunk_max; chunk++) {
    kmp_uint64 a_bits = chunk < a->num_chunks ? a->bits[chunk] : 0;
    kmp_uint64 b_bits = chunk < b->num_chunks ? b->bits[chunk] : 0;
    if ((a_bits & b_bits) != 0)
      return true;
  }
  return false;
}

static kmp_int32 __kmp_bitset_popcount(kmp_bitset_t *bitset) {
  if (!bitset)
    return 0;
  kmp_int32 accum = 0;
  for (kmp_int32 c = 0; c < bitset->num_chunks; c++) {
    accum += std::__popcount(bitset->bits[c]);
  }
  return accum;
}

static kmp_int32 __kmp_taskgraph_add_dep(kmp_info_t *thread,
                                         kmp_depnode_t *node,
                                         kmp_depnode_list_t *plist) {
  kmp_int32 npredecessors = 0;
  for (; plist; plist = plist->next) {
    kmp_depnode_t *dep = plist->node;
    if (!dep->dn.successors ||
        !__kmp_find_in_depnode_list(node, dep->dn.successors)) {
      dep->dn.successors =
          __kmp_add_node<false>(thread, dep->dn.successors, node);
      npredecessors++;
    }
  }
  return npredecessors;
}

static kmp_int32 __kmp_taskgraph_add_dep(kmp_info_t *thread,
                                         kmp_depnode_t *source,
                                         kmp_depnode_t *sink) {
  if (!sink)
    return 0;
  kmp_int32 npredecessors = 0;
  if (!sink->dn.successors || sink->dn.successors->node != source) {
    if (!__kmp_find_in_depnode_list(source, sink->dn.successors)) {
      sink->dn.successors =
          __kmp_add_node<false>(thread, sink->dn.successors, source);
      npredecessors++;
    }
  }
  return npredecessors;
}
#endif

template <typename T>
static inline kmp_int32
__kmp_process_dep_all(kmp_int32 gtid, kmp_depnode_t *node, kmp_dephash_t *h,
                      bool dep_barrier, kmp_task_t *task) {
  KA_TRACE(30, ("__kmp_process_dep_all<%s>: T#%d processing dep_all, "
                "dep_barrier = %d\n",
                T::name, gtid, dep_barrier));
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_int32 npredecessors = 0;

  // process previous omp_all_memory node if any
  npredecessors += T::link_successor(gtid, thread, task, node, h->last_all);
  T::deref(thread, h->last_all);
  if (!dep_barrier) {
    h->last_all = T::ref(node);
  } else {
    // if this is a sync point in the serial sequence, then the previous
    // outputs are guaranteed to be completed after the execution of this
    // task so the previous output nodes can be cleared.
    h->last_all = NULL;
  }

  // process all regular dependences
  for (size_t i = 0; i < h->size; i++) {
    kmp_dephash_entry_t *info = h->buckets[i];
    if (!info) // skip empty slots in dephash
      continue;
    for (; info; info = info->next_in_bucket) {
      // for each entry the omp_all_memory works as OUT dependence
      kmp_depnode_t *last_out = info->last_out;
      kmp_depnode_list_t *last_set = info->last_set;
      kmp_depnode_list_t *prev_set = info->prev_set;
      if (last_set) {
        npredecessors += T::link_successor(gtid, thread, task, node, last_set);
        __kmp_depnode_list_free<T::rc>(thread, last_set);
        __kmp_depnode_list_free<T::rc>(thread, prev_set);
        info->last_set = NULL;
        info->prev_set = NULL;
        info->last_flag = 0; // no sets in this dephash entry
      } else {
        npredecessors += T::link_successor(gtid, thread, task, node, last_out);
      }
      T::deref(thread, last_out);
      if (!dep_barrier) {
        info->last_out = T::ref(node);
      } else {
        info->last_out = NULL;
      }
    }
  }
  KA_TRACE(30, ("__kmp_process_dep_all<%s>: T#%d found %d predecessors\n",
                T::name, gtid, npredecessors));
  return npredecessors;
}

template <typename T>
static inline kmp_int32
__kmp_process_deps(kmp_int32 gtid, kmp_depnode_t *node, kmp_dephash_t **hash,
                   bool dep_barrier, kmp_int32 ndeps,
                   kmp_depend_info_t *dep_list, kmp_task_t *task,
                   kmp_int32 &next_mutex_set, bool filter = true) {
  KA_TRACE(30, ("__kmp_process_deps<%s>: T#%d processing %d dependences : "
                "dep_barrier = %d, filter = %d\n",
                T::name, gtid, ndeps, dep_barrier, filter));

  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_int32 npredecessors = 0;
  for (kmp_int32 i = 0; i < ndeps; i++) {
    const kmp_depend_info_t *dep = &dep_list[i];

    if (filter && dep->base_addr == 0)
      continue; // skip filtered entries

#if OMP_TASKGRAPH_EXPERIMENTAL
    kmp_dephash_entry_t *info =
        __kmp_dephash_find(thread, hash, dep->base_addr, !T::rc);
#else
    kmp_dephash_entry_t *info =
        __kmp_dephash_find(thread, hash, dep->base_addr);
#endif
    kmp_depnode_t *last_out = info->last_out;
    kmp_depnode_list_t *last_set = info->last_set;
    kmp_depnode_list_t *prev_set = info->prev_set;

    if (dep->flags.out) { // out or inout --> clean lists if any
      if (last_set) {
        npredecessors += T::link_successor(gtid, thread, task, node, last_set);
        __kmp_depnode_list_free<T::rc>(thread, last_set);
        __kmp_depnode_list_free<T::rc>(thread, prev_set);
        info->last_set = NULL;
        info->prev_set = NULL;
        info->last_flag = 0; // no sets in this dephash entry
      } else {
        npredecessors += T::link_successor(gtid, thread, task, node, last_out);
      }
      T::deref(thread, last_out);
      if (!dep_barrier) {
        info->last_out = T::ref(node);
      } else {
        // if this is a sync point in the serial sequence, then the previous
        // outputs are guaranteed to be completed after the execution of this
        // task so the previous output nodes can be cleared.
        info->last_out = NULL;
      }
    } else { // either IN or MTX or SET
      if (info->last_flag == 0 || info->last_flag == dep->flag) {
        // last_set either didn't exist or of same dep kind
        // link node as successor of the last_out if any
        npredecessors += T::link_successor(gtid, thread, task, node, last_out);
        // link node as successor of all nodes in the prev_set if any
        npredecessors += T::link_successor(gtid, thread, task, node, prev_set);
        if (dep_barrier) {
          // clean last_out and prev_set if any; don't touch last_set
          T::deref(thread, last_out);
          info->last_out = NULL;
          __kmp_depnode_list_free<T::rc>(thread, prev_set);
          info->prev_set = NULL;
        }
      } else { // last_set is of different dep kind, make it prev_set
        // link node as successor of all nodes in the last_set
        npredecessors += T::link_successor(gtid, thread, task, node, last_set);
        // clean last_out if any
        T::deref(thread, last_out);
        info->last_out = NULL;
        // clean prev_set if any
        __kmp_depnode_list_free<T::rc>(thread, prev_set);
        if (!dep_barrier) {
          // move last_set to prev_set, new last_set will be allocated
          info->prev_set = last_set;
        } else {
          info->prev_set = NULL;
          info->last_flag = 0;
        }
        info->last_set = NULL;
      }
      // for dep_barrier last_flag value should remain:
      // 0 if last_set is empty, unchanged otherwise
      if (!dep_barrier) {
        info->last_flag = dep->flag; // store dep kind of the last_set
        info->last_set = __kmp_add_node<T::rc>(thread, info->last_set, node);
      }
      // check if we are processing MTX dependency
      if (dep->flag == KMP_DEP_MTX) {
        T::mutex_dep(thread, info, node, next_mutex_set);
      }
    }
  }
  KA_TRACE(30,
           ("__kmp_process_deps<%s>: T#%d found %d predecessors (filter: %d)\n",
            T::name, gtid, npredecessors, filter));
  return npredecessors;
}

struct normal_deps {
  static constexpr char name[] = "normal";
  static constexpr bool rc = true;
  static kmp_int32 link_successor(kmp_int32 gtid, kmp_info_t *thread,
                                  kmp_task_t *task, kmp_depnode_t *source,
                                  kmp_depnode_t *sink);
  static kmp_int32 link_successor(kmp_int32 gtid, kmp_info_t *thread,
                                  kmp_task_t *task, kmp_depnode_t *node,
                                  kmp_depnode_list_t *plist);
  static kmp_depnode_t *ref(kmp_depnode_t *node);
  static void deref(kmp_info_t *thread, kmp_depnode_t *node);
  static void mutex_dep(kmp_info_t *thread, kmp_dephash_entry_t *info,
                        kmp_depnode_t *node, kmp_int32 &next_mutex_set);
};

kmp_int32 normal_deps::link_successor(kmp_int32 gtid, kmp_info_t *thread,
                                      kmp_task_t *task, kmp_depnode_t *source,
                                      kmp_depnode_t *sink) {
  return __kmp_depnode_link_successor(gtid, thread, task, source, sink);
}

kmp_int32 normal_deps::link_successor(kmp_int32 gtid, kmp_info_t *thread,
                                      kmp_task_t *task, kmp_depnode_t *node,
                                      kmp_depnode_list_t *plist) {
  return __kmp_depnode_link_successor(gtid, thread, task, node, plist);
}

kmp_depnode_t *normal_deps::ref(kmp_depnode_t *node) {
  return __kmp_node_ref(node);
}

void normal_deps::deref(kmp_info_t *thread, kmp_depnode_t *node) {
  __kmp_node_deref(thread, node);
}

void normal_deps::mutex_dep(kmp_info_t *thread, kmp_dephash_entry_t *info,
                            kmp_depnode_t *node, kmp_int32 &next_mutex_set) {
  if (info->mtx_lock == NULL) {
    info->mtx_lock = (kmp_lock_t *)__kmp_allocate(sizeof(kmp_lock_t));
    __kmp_init_lock(info->mtx_lock);
  }
  KMP_DEBUG_ASSERT(node->dn.mtx_num_locks < MAX_MTX_DEPS);
  kmp_int32 m;
  // Save lock in node's array
  for (m = 0; m < MAX_MTX_DEPS; ++m) {
    // sort pointers in decreasing order to avoid potential livelock
    if (node->dn.mtx_locks[m] < info->mtx_lock) {
      KMP_DEBUG_ASSERT(!node->dn.mtx_locks[node->dn.mtx_num_locks]);
      for (int n = node->dn.mtx_num_locks; n > m; --n) {
        // shift right all lesser non-NULL pointers
        KMP_DEBUG_ASSERT(node->dn.mtx_locks[n - 1] != NULL);
        node->dn.mtx_locks[n] = node->dn.mtx_locks[n - 1];
      }
      node->dn.mtx_locks[m] = info->mtx_lock;
      break;
    }
  }
  KMP_DEBUG_ASSERT(m < MAX_MTX_DEPS); // must break from loop
  node->dn.mtx_num_locks++;
}

#if OMP_TASKGRAPH_EXPERIMENTAL
struct taskgraph_deps {
  static constexpr char name[] = "taskgraph";
  static constexpr bool rc = false;
  static kmp_int32 link_successor(kmp_int32 gtid, kmp_info_t *thread,
                                  kmp_task_t *task, kmp_depnode_t *source,
                                  kmp_depnode_t *sink);
  static kmp_int32 link_successor(kmp_int32 gtid, kmp_info_t *thread,
                                  kmp_task_t *task, kmp_depnode_t *node,
                                  kmp_depnode_list_t *plist);
  static kmp_depnode_t *ref(kmp_depnode_t *node) { return node; }
  static void deref(kmp_info_t *thread, kmp_depnode_t *node) {}
  static void mutex_dep(kmp_info_t *thread, kmp_dephash_entry_t *info,
                        kmp_depnode_t *node, kmp_int32 &next_mutex_set);
};

kmp_int32 taskgraph_deps::link_successor(kmp_int32 gtid, kmp_info_t *thread,
                                         kmp_task_t *task,
                                         kmp_depnode_t *source,
                                         kmp_depnode_t *sink) {
  return __kmp_taskgraph_add_dep(thread, source, sink);
}

kmp_int32 taskgraph_deps::link_successor(kmp_int32 gtid, kmp_info_t *thread,
                                         kmp_task_t *task, kmp_depnode_t *node,
                                         kmp_depnode_list_t *plist) {
  return __kmp_taskgraph_add_dep(thread, node, plist);
}

void taskgraph_deps::mutex_dep(kmp_info_t *thread, kmp_dephash_entry_t *info,
                               kmp_depnode_t *node, kmp_int32 &next_mutex_set) {
  if (info->set_num == -1) {
    info->set_num = next_mutex_set++;
  }
  if (!node->dn.set_membership) {
    node->dn.set_membership = __kmp_bitset_alloc(thread, 64);
  }
  __kmp_bitset_set(node->dn.set_membership, info->set_num);
}
#endif

/// Search for aliased (same base address) dependencies in \c dep_list, and
/// nullify duplicates.  Return TRUE if we have an 'all' dependency, FALSE
/// otherwise.  Return number of mutex dependencies in *N_MTXS.
static bool __kmp_filter_aliased_deps(kmp_int32 ndeps,
                                      kmp_depend_info_t *dep_list,
                                      kmp_task_t *task, int *n_mtxs) {
  *n_mtxs = 0;

  // Filter deps in dep_list
  // TODO: Different algorithm for large dep_list ( > 10 ? )
  for (int i = 0; i < ndeps; i++) {
    if (dep_list[i].base_addr != 0 &&
        dep_list[i].base_addr != (kmp_intptr_t)KMP_SIZE_T_MAX) {
      KMP_DEBUG_ASSERT(
          dep_list[i].flag == KMP_DEP_IN || dep_list[i].flag == KMP_DEP_OUT ||
          dep_list[i].flag == KMP_DEP_INOUT ||
          dep_list[i].flag == KMP_DEP_MTX || dep_list[i].flag == KMP_DEP_SET);
      for (int j = i + 1; j < ndeps; j++) {
        if (dep_list[i].base_addr == dep_list[j].base_addr) {
          if (dep_list[i].flag != dep_list[j].flag) {
            // two different dependences on same address work identical to OUT
            dep_list[i].flag = KMP_DEP_OUT;
          }
          dep_list[j].base_addr = 0; // Mark j element as void
        }
      }
      if (dep_list[i].flag == KMP_DEP_MTX) {
        // limit number of mtx deps to MAX_MTX_DEPS per node
        if (*n_mtxs < MAX_MTX_DEPS && task != NULL) {
          ++(*n_mtxs);
        } else {
          dep_list[i].flag = KMP_DEP_OUT; // downgrade mutexinoutset to inout
        }
      }
    } else if (dep_list[i].flag == KMP_DEP_ALL ||
               dep_list[i].base_addr == (kmp_intptr_t)KMP_SIZE_T_MAX) {
      // omp_all_memory dependence can be marked by compiler by either
      // (addr=0 && flag=0x80) (flag KMP_DEP_ALL), or (addr=-1).
      // omp_all_memory overrides all other dependences if any
      return true;
    }
  }
  return false;
}

#if OMP_TASKGRAPH_EXPERIMENTAL
// Round up a size to a power of two specified by val: Used to insert padding
// between structures co-allocated using a single malloc() call
// FIXME: We copy+pasted this, put it somewhere else instead.
static size_t __kmp_round_up_to_val(size_t size, size_t val) {
  if (size & (val - 1)) {
    size &= ~(val - 1);
    if (size <= KMP_SIZE_T_MAX - val) {
      size += val; // Round up if there is no overflow.
    }
  }
  return size;
} // __kmp_round_up_to_val

// FIXME: C++-ify this.
static kmp_taskgraph_region_t *__kmp_taskgraph_region_alloc(
    kmp_info_t *thread, kmp_taskgraph_record_t *taskgraph,
    kmp_taskgraph_region_t **&alloc_chain, kmp_taskgraph_node_t *node,
    kmp_taskgraph_region_t *parent) {
  kmp_taskgraph_region_t *region =
      (kmp_taskgraph_region_t *)__kmp_fast_allocate(
          thread, sizeof(kmp_taskgraph_region_t));
  region->owner = taskgraph;
  region->type = node ? TASKGRAPH_REGION_NODE : TASKGRAPH_REGION_WAIT;
  region->task.node = node;
  region->task.next_instance = region;
  region->mark = TASKGRAPH_UNMARKED;
  region->level = -1;
  region->timestamp = 0;
  region->next = nullptr;
  region->parent = parent;
  region->predecessors = nullptr;
  region->successors = nullptr;
  region->mutexset = nullptr;
  region->mutexset_parent = nullptr;
  region->reduce_input = nullptr;
  *alloc_chain = region;
  alloc_chain = &region->alloc_chain;
  return region;
}

// FIXME: This too.
static kmp_taskgraph_region_t *__kmp_taskgraph_region_alloc(
    kmp_info_t *thread, kmp_taskgraph_record_t *taskgraph,
    kmp_taskgraph_region_t **&alloc_chain, enum kmp_taskgraph_region_type type,
    kmp_int32 num_nodes, kmp_taskgraph_region_t *parent) {
  kmp_size_t size = sizeof(kmp_taskgraph_region_t) +
                    num_nodes * sizeof(kmp_taskgraph_region_t *);
  size = __kmp_round_up_to_val(size, sizeof(kmp_taskgraph_region_t *));
  kmp_taskgraph_region_t *region =
      (kmp_taskgraph_region_t *)__kmp_fast_allocate(thread, size);
  region->owner = taskgraph;
  region->type = type;
  region->inner.children = (kmp_taskgraph_region **)&region[1];
  region->inner.num_children = num_nodes;
  region->mark = TASKGRAPH_UNMARKED;
  region->level = -1;
  region->timestamp = 0;
  region->next = nullptr;
  region->parent = parent;
  region->predecessors = nullptr;
  region->successors = nullptr;
  region->mutexset = nullptr;
  region->mutexset_parent = nullptr;
  region->reduce_input = nullptr;
  *alloc_chain = region;
  alloc_chain = &region->alloc_chain;
  return region;
}

// This makes a mostly-deep copy of a region.  The region itself and children
// nodes are created new, but node pointers are shared.
static kmp_taskgraph_region_t *__kmp_taskgraph_region_clone(
    kmp_info_t *thread, kmp_taskgraph_record_t *taskgraph,
    kmp_taskgraph_region_t **&alloc_chain, kmp_taskgraph_region_t *from,
    kmp_taskgraph_region_t *parent, kmp_int32 indent = 0) {
  kmp_taskgraph_region_t *clone = nullptr;
  switch (from->type) {
  case TASKGRAPH_REGION_ENTRY:
  case TASKGRAPH_REGION_EXIT:
    clone = __kmp_taskgraph_region_alloc(thread, taskgraph, alloc_chain,
                                         nullptr, parent);
    clone->type = from->type;
    break;
  case TASKGRAPH_REGION_NODE:
  case TASKGRAPH_REGION_WAIT:
    clone = __kmp_taskgraph_region_alloc(thread, taskgraph, alloc_chain,
                                         from->task.node, parent);
    break;
  default: {
    clone =
        __kmp_taskgraph_region_alloc(thread, taskgraph, alloc_chain, from->type,
                                     from->inner.num_children, parent);
    for (kmp_int32 n = 0; n < from->inner.num_children; n++) {
      clone->inner.children[n] = __kmp_taskgraph_region_clone(
          thread, taskgraph, alloc_chain, from->inner.children[n], clone,
          indent + 2);
    }
  }
  }
  TGDBG("%*scloned region %p from region %p\n", indent, "", clone, from);
  return clone;
}

static kmp_int32
__kmp_taskgraph_topological_order(kmp_taskgraph_region_t *region,
                                  kmp_taskgraph_region_t **order_out,
                                  kmp_int32 *outidx) {
  if (region->mark == TASKGRAPH_PERMANENT_MARK)
    return region->level;

  assert(region->mark != TASKGRAPH_TEMP_MARK);

  region->mark = TASKGRAPH_TEMP_MARK;

  kmp_int32 max_level = -1;
  for (kmp_taskgraph_region_dep_t *s = region->predecessors; s; s = s->next) {
    kmp_int32 pred_level =
        __kmp_taskgraph_topological_order(s->region, order_out, outidx);
    max_level = pred_level > max_level ? pred_level : max_level;
  }

  region->level = max_level + 1;
  region->mark = TASKGRAPH_PERMANENT_MARK;
  order_out[(*outidx)++] = region;

  return region->level;
}

static void
__kmp_taskgraph_region_chain_clear_marks(kmp_taskgraph_region_t *region) {
  for (; region; region = region->next)
    region->mark = TASKGRAPH_UNMARKED;
}

static void
__kmp_taskgraph_region_chain_prune(kmp_taskgraph_region_t **region_p) {
  kmp_taskgraph_region_t *pruned_region = nullptr, *region = *region_p;
  kmp_taskgraph_region_t **pruned_region_p = &pruned_region;

  TGDBG("pruning worklist...\n");

  // NOTE: Pruning and deletion look the same here with respect to the handling
  // of the worklist, but deleted nodes are freed from the taskgraph structure
  // during cleanup, whereas combined nodes are retained.
  for (; region; region = region->next) {
    if (region->mark == TASKGRAPH_COMBINED || region->mark == TASKGRAPH_DELETED)
      *pruned_region_p = region->next;
    else {
      *pruned_region_p = region;
      pruned_region_p = &region->next;
    }
  }

  *pruned_region_p = nullptr;
  *region_p = pruned_region;
}

static kmp_int32 __kmp_region_deplist_len(kmp_taskgraph_region_dep_t *list) {
  kmp_int32 len = 0;
  for (; list; list = list->next)
    ++len;
  return len;
}

static void __kmp_region_deplist_free(kmp_info_t *thread,
                                      kmp_taskgraph_region_dep_t *list) {
  while (list) {
    kmp_taskgraph_region_dep_t *next = list->next;
    __kmp_fast_free(thread, list);
    list = next;
  }
}

static void __kmp_region_dep_recycle(kmp_taskgraph_region_dep_t **recycled,
                                     kmp_taskgraph_region_dep_t *dep) {
  dep->next = *recycled;
  *recycled = dep;
}

static void __kmp_region_deplist_recycle(kmp_taskgraph_region_dep_t **recycled,
                                         kmp_taskgraph_region_dep_t *list) {
  while (list) {
    kmp_taskgraph_region_dep_t *next = list->next;
    __kmp_region_dep_recycle(recycled, list);
    list = next;
  }
}

static bool __kmp_taskgraph_collapse_sequence(
    kmp_info_t *thread, kmp_taskgraph_record_t *taskgraph,
    kmp_taskgraph_region_t **&alloc_chain, kmp_taskgraph_region_t **region_p,
    kmp_taskgraph_region_t *parent, kmp_int32 &stamp) {
  kmp_taskgraph_region_t *region = *region_p;
  kmp_taskgraph_region_t *chain_start = region;
  kmp_taskgraph_region_t *chain_end = region;
  kmp_int32 chain_len = 1;

  if (region->type == TASKGRAPH_REGION_ENTRY)
    return false;

  while (__kmp_region_deplist_len(chain_end->successors) == 1) {
    kmp_taskgraph_region_t *past_end = chain_end->successors->region;
    if (__kmp_region_deplist_len(past_end->predecessors) == 1) {
      if (past_end->type == TASKGRAPH_REGION_EXIT)
        break;
      else {
        chain_end = past_end;
        ++chain_len;
      }
    } else
      break;
  }

  if (chain_len <= 1)
    return false;

  kmp_taskgraph_region_t *seq_region = __kmp_taskgraph_region_alloc(
      thread, taskgraph, alloc_chain, TASKGRAPH_REGION_SEQUENTIAL, chain_len,
      parent);
  TGDBG("allocated new seq region: %p (length %d)\n", seq_region, chain_len);
  kmp_taskgraph_region_t **worklist_p = region_p;
  *worklist_p = seq_region;
  seq_region->next = chain_start->next;
  kmp_int32 level = -1;
  for (kmp_int32 i = 0; i < chain_len; i++) {
    seq_region->inner.children[i] = chain_start;
    TGDBG("mark node %p as combined\n", chain_start);
    chain_start->mark = TASKGRAPH_COMBINED;
    chain_start->timestamp = stamp;
    chain_start->parent = seq_region;
    // The level of the sequence is the level of the first node.
    if (level == -1)
      level = chain_start->level;

    if (i < chain_len - 1) {
      chain_start = chain_start->successors->region;
    }
  }

  seq_region->level = level;
  seq_region->predecessors = seq_region->inner.children[0]->predecessors;
  seq_region->successors =
      seq_region->inner.children[chain_len - 1]->successors;
  seq_region->inner.children[0]->predecessors = nullptr;
  seq_region->inner.children[chain_len - 1]->successors = nullptr;

  // Update predecessors to point to new seq region.
  for (kmp_taskgraph_region_dep_t *pred = seq_region->predecessors; pred;
       pred = pred->next) {
    for (kmp_taskgraph_region_dep_t *succ = pred->region->successors; succ;
         succ = succ->next) {
      if (succ->region == seq_region->inner.children[0]) {
        succ->region = seq_region;
      }
    }
  }

  // Update successors to point back to new seq region.
  for (kmp_taskgraph_region_dep_t *succ = seq_region->successors; succ;
       succ = succ->next) {
    for (kmp_taskgraph_region_dep_t *pred = succ->region->predecessors; pred;
         pred = pred->next) {
      if (pred->region == seq_region->inner.children[chain_len - 1]) {
        pred->region = seq_region;
      }
    }
  }

  return true;
}

static const char *
__kmp_taskgraph_region_type_name(kmp_taskgraph_region_type type);

static void __kmp_taskgraph_region_dfs(kmp_taskgraph_region_t *region,
                                       kmp_taskgraph_region_t **order,
                                       kmp_int32 &idx, bool use_preds) {
  if (order) {
    region->timestamp = --idx;
    order[idx] = region;
  }
  region->mark = TASKGRAPH_TEMP_MARK;
  for (kmp_taskgraph_region_dep_t *reg = use_preds ? region->predecessors
                                                   : region->successors;
       reg; reg = reg->next) {
    if (reg->region->mark == TASKGRAPH_UNMARKED)
      __kmp_taskgraph_region_dfs(reg->region, order, idx, use_preds);
  }
}

#if defined(DEBUG_TASKGRAPH) && defined(CHECK_WORKLIST)

static void __kmp_taskgraph_region_gather_deps(
    kmp_info_t *thread, kmp_taskgraph_record_t *taskgraph,
    kmp_taskgraph_region_t *region, kmp_taskgraph_region_dep_t **deplist,
    bool &ok) {
  for (kmp_taskgraph_region_dep_t *dep = *deplist; dep; dep = dep->next) {
    if (dep->region == region)
      return;
  }

  *deplist = __kmp_region_deplist_add(thread, &taskgraph->recycled_deps, region,
                                      *deplist);

  for (kmp_taskgraph_region_dep_t *pred = region->predecessors; pred;
       pred = pred->next) {
    if (pred->region->mark == TASKGRAPH_DELETED) {
      fprintf(stderr, "*** Region %p's predecessor %p is a deleted node\n",
              region, pred->region);
      ok = false;
    }
    __kmp_taskgraph_region_gather_deps(thread, taskgraph, pred->region, deplist,
                                       ok);
  }

  for (kmp_taskgraph_region_dep_t *succ = region->successors; succ;
       succ = succ->next) {
    if (succ->region->mark == TASKGRAPH_DELETED) {
      fprintf(stderr, "*** Region %p's successor %p is a deleted node\n",
              region, succ->region);
      ok = false;
    }
    __kmp_taskgraph_region_gather_deps(thread, taskgraph, succ->region, deplist,
                                       ok);
  }
}

static bool __kmp_taskgraph_region_worklist_check(
    kmp_info_t *thread, kmp_taskgraph_record_t *taskgraph,
    kmp_taskgraph_region_t *region, const char *where) {
  kmp_taskgraph_region_dep_t *collected_nodes = nullptr;
  bool ok = true;
  __kmp_taskgraph_region_gather_deps(thread, taskgraph, region,
                                     &collected_nodes, ok);

  // Check all collected nodes are in the region's worklist.
  for (kmp_taskgraph_region_dep_t *cn = collected_nodes; cn; cn = cn->next) {
    bool in_list = false;
    for (kmp_taskgraph_region_t *r = region; r; r = r->next) {
      if (r == cn->region) {
        in_list = true;
        break;
      }
    }
    if (!in_list) {
      fprintf(stderr,
              "*** Region %p is in dependency graph but not worklist (%s)\n",
              cn->region, where);
      ok = false;
    }
  }

  for (kmp_taskgraph_region_t *r = region; r; r = r->next) {
    bool in_list = false;
    for (kmp_taskgraph_region_dep_t *cn = collected_nodes; cn; cn = cn->next) {
      if (r == cn->region) {
        in_list = true;
        break;
      }
    }
    if (!in_list) {
      fprintf(stderr,
              "*** Region %p is in worklist but not dependency graph (%s)\n", r,
              where);
      ok = false;
    }
  }

  __kmp_region_deplist_recycle(&taskgraph->recycled_deps, collected_nodes);

  return ok;
}
#else
static bool __kmp_taskgraph_region_worklist_check(
    kmp_info_t *thread, kmp_taskgraph_record_t *taskgraph,
    kmp_taskgraph_region_t *region, const char *where) {
  return true;
}
#endif

static kmp_taskgraph_region_t *__kmp_taskgraph_region_dom_intersect(
    kmp_taskgraph_region_t **order, kmp_taskgraph_region_t **doms,
    kmp_taskgraph_region_t *b1, kmp_taskgraph_region_t *b2) {
  kmp_int32 finger1 = b1->timestamp;
  kmp_int32 finger2 = b2->timestamp;
  while (finger1 != finger2) {
    while (finger1 < finger2)
      finger1 = doms[finger1]->timestamp;
    while (finger2 < finger1)
      finger2 = doms[finger2]->timestamp;
  }
  return order[finger1];
}

static void __kmp_taskgraph_region_doms(kmp_taskgraph_region_t **order,
                                        kmp_taskgraph_region_t **doms,
                                        kmp_int32 worklist_length,
                                        bool postdom) {
  bool changed = true;
  // Set doms[start_node] <- start_node
  doms[worklist_length - 1] = order[worklist_length - 1];
  order[worklist_length - 1]->mark = TASKGRAPH_PERMANENT_MARK;
  while (changed) {
    changed = false;
    for (int n = 0; n < worklist_length - 1; n++) {
      kmp_taskgraph_region_t *b = order[n];
      kmp_taskgraph_region_t *new_idom = nullptr;
      for (kmp_taskgraph_region_dep_t *pred = postdom ? b->successors
                                                      : b->predecessors;
           pred; pred = pred->next) {
        if (pred->region->mark == TASKGRAPH_PERMANENT_MARK) {
          new_idom = pred->region;
          break;
        }
      }
      for (kmp_taskgraph_region_dep_t *pred = postdom ? b->successors
                                                      : b->predecessors;
           pred; pred = pred->next) {
        if (pred->region == new_idom)
          continue;
        if (doms[pred->region->timestamp]) {
          new_idom = __kmp_taskgraph_region_dom_intersect(
              order, doms, pred->region, new_idom);
        }
      }
      if (doms[b->timestamp] != new_idom) {
        doms[b->timestamp] = new_idom;
        order[b->timestamp]->mark = TASKGRAPH_PERMANENT_MARK;
        changed = true;
      }
    }
  }
}

static bool __kmp_taskgraph_region_mutex_p(kmp_taskgraph_region_t *reg) {
  if (reg->type == TASKGRAPH_REGION_NODE)
    return reg->mutexset != nullptr;
  return false;
}

// This function collapses graph regions with forms like this:
//
//  1.    A(pp)    2.        A               3.     A(pp)
//       / \              /     \                 /   \
//      B   C           B(pp)    E(pp)           B(pp)  E
//       \ /           / \       / \            / \    /
//       D(*)         C   D     F   G           C  D  /
//                     \   \   /   /             \ | /
//                      `---H(*)--'               F(*)
//
// We look for a node with more than one predecessor (*), where each of those
// predecessors has a single successor and a single predecessor (pp).  We group
// nodes by which pp (predecessor-predecessor) they have: for (1), nodes B & C
// share a pp; for (2), C & D share a pp, and F & G share a pp; for (3), C & D
// share a pp, and E has a separate pp.
//
// We choose the pp the the highest level ("furthest down the graph"), and
// collapse the subgraph into a parallel region.

static bool __kmp_taskgraph_collapse_par_exclusive(
    kmp_info_t *thread, kmp_taskgraph_record_t *taskgraph,
    kmp_taskgraph_region_t **&alloc_chain, kmp_taskgraph_region_t **region_p,
    kmp_taskgraph_region_t *parent, kmp_int32 &stamp) {
  kmp_taskgraph_region_t *region = *region_p;
  kmp_int32 num_predecessors = __kmp_region_deplist_len(region->predecessors);

  TGDBG("predecessors %d, successors %d\n",
        __kmp_region_deplist_len(region->predecessors),
        __kmp_region_deplist_len(region->successors));

  if (num_predecessors <= 1)
    return false;

  TGDBG("found multiple predecessors, creating parallel/unordered region\n");
  kmp_taskgraph_region_dep_t *pred_preds = nullptr;
  kmp_int32 highest_level = -1;

  for (kmp_taskgraph_region_dep_t *pred = region->predecessors; pred;
       pred = pred->next) {
    TGDBG("consider predecessor: %p\n", pred->region);
    TGDBG("-- successors %d, predecessors %d\n",
          __kmp_region_deplist_len(pred->region->successors),
          __kmp_region_deplist_len(pred->region->predecessors));
    if (highest_level == -1 || pred->region->level > highest_level)
      highest_level = pred->region->level;
    kmp_taskgraph_region_t *pred_region = pred->region;
    if (__kmp_region_deplist_len(pred_region->successors) != 1)
      continue;
    if (__kmp_region_deplist_len(pred_region->predecessors) != 1)
      continue;
    bool in_list = false;
    TGDBG("pp region: %p (%s)\n", pred_region->predecessors->region,
          __kmp_taskgraph_region_type_name(
              pred_region->predecessors->region->type));
    kmp_taskgraph_region_t *pp_region = pred_region->predecessors->region;
    for (kmp_taskgraph_region_dep_t *pp = pred_preds; pp; pp = pp->next) {
      if (pp->region == pp_region) {
        in_list = true;
        break;
      }
    }
    if (!in_list) {
      pred_preds = __kmp_region_deplist_add(thread, &taskgraph->recycled_deps,
                                            pp_region, pred_preds);
      TGDBG("add %p to list: len(pred_preds)=%d\n", pp_region,
            __kmp_region_deplist_len(pred_preds));
    }
  }

  kmp_int32 num_pps = __kmp_region_deplist_len(pred_preds);
  if (num_pps == 0) {
    TGDBG("no collapsible regions, bailing out\n");
    return false;
  }
  TGDBG("found %d predecessor-predecessors\n", num_pps);
  TGDBG("highest pred level: %d\n", highest_level);

  kmp_int32 pp_idx = 0;

  bool changed = false;

  for (kmp_taskgraph_region_dep_t *pp = pred_preds; pp; pp = pp->next) {
    kmp_taskgraph_region_dep_t *par_succs = nullptr;
    kmp_taskgraph_region_dep_t *par_preds = nullptr;
    kmp_int32 preds_for_pp = 0;
    bool any_mutex_p = false;
    for (kmp_taskgraph_region_dep_t *pred = region->predecessors; pred;
         pred = pred->next) {
      kmp_taskgraph_region_t *pred_region = pred->region;
      if (!pred_region->predecessors)
        continue;
      if (pred_region->level < highest_level)
        continue;
      if (__kmp_region_deplist_len(pred_region->predecessors) != 1 ||
          __kmp_region_deplist_len(pred_region->successors) != 1)
        continue;
      TGDBG("counting pred region: %p (%s)\n", pred_region,
            __kmp_taskgraph_region_type_name(pred_region->type));
      if (pred_region->predecessors->region == pp->region) {
        ++preds_for_pp;
        if (__kmp_taskgraph_region_mutex_p(pred_region))
          any_mutex_p = true;
      }
    }
    TGDBG("found %d preds for pp region %p\n", preds_for_pp, pp->region);
    if (preds_for_pp < 2)
      continue;
    kmp_taskgraph_region_type region_type =
        any_mutex_p ? TASKGRAPH_REGION_EXCLUSIVE : TASKGRAPH_REGION_PARALLEL;
    kmp_taskgraph_region_t *par_region = __kmp_taskgraph_region_alloc(
        thread, taskgraph, alloc_chain, region_type, preds_for_pp, parent);
    changed = true;
    TGDBG("allocated %s region: %p\n",
          region_type == TASKGRAPH_REGION_EXCLUSIVE ? "exclusive" : "parallel",
          par_region);
    kmp_taskgraph_region_dep_t *pred = region->predecessors;
    kmp_int32 level = -1;
    bool found_reduction_data = false;
    for (kmp_int32 i = 0; pred; pred = pred->next) {
      kmp_taskgraph_region_t *pred_region = pred->region;
      TGDBG("considering pred region: %p\n", pred_region);
      if (!pred_region->predecessors) {
        TGDBG("bailing (no predecessors)\n");
        continue;
      }
      if (pred_region->predecessors->region != pp->region) {
        TGDBG("bailing (wrong pp region)\n");
        continue;
      }
      if (__kmp_region_deplist_len(pred_region->predecessors) != 1 ||
          __kmp_region_deplist_len(pred_region->successors) != 1) {
        TGDBG("bailing (non-unit pred/succ list length)\n");
        continue;
      }
      TGDBG("process region %p (%d/%d), level %d\n", pred->region, i + 1,
            preds_for_pp, pred_region->level);
      par_region->inner.children[i] = pred_region;
      pred_region->mark = TASKGRAPH_COMBINED;
      pred_region->timestamp = stamp;
      pred_region->parent = par_region;

      // Reduction handling.  The reduction input data is now attached to one
      // of the tasks participating in the reduction.  Move it to the enclosing
      // parallel region instead.
      if (pred_region->type == TASKGRAPH_REGION_NODE &&
          pred_region->task.node->reduce_input) {
        // We should only be doing this once per par region.
        assert(!par_region->reduce_input);
        par_region->reduce_input = pred_region->task.node->reduce_input;
        pred_region->task.node->reduce_input = nullptr;
        found_reduction_data = true;
      }

      // We expect all the predecessor regions to be at the same level.
      if (level == -1)
        level = pred_region->level;
      else
        assert(level == pred_region->level);
      if (!par_succs) {
        // Copy one list of predecessors/successors for the predecessor region.
        // We know these are of length one by checks above.  We'll re-use them
        // for the created parallel region.
        par_preds = pred_region->predecessors;
        par_succs = pred_region->successors;
        pred_region->predecessors = nullptr;
        pred_region->successors = nullptr;
      }
      i++;
    }
    par_region->level = level;
    par_region->predecessors = par_preds;
    par_region->successors = par_succs;

    if (region->type == TASKGRAPH_REGION_WAIT && !found_reduction_data) {
      // If we have no reduction data, we will not create a taskgroup for this
      // parallel region at replay time, so we don't need to terminate/discard
      // that region when we're done.  Clear the taskloop_task flag.
      region->task.node->taskloop_task = false;
    }

    // Add the new parallel region to the worklist. FIXME: We're reprocessing
    // the 'region' node here -- we don't need to do that if it's fully
    // consumed.)
    par_region->next = region->next;
    region->next = par_region;
  }

#ifdef DEBUG_TASKGRAPH
  TGDBG("before pred fixup:\n");
  for (kmp_taskgraph_region_dep_t *pred = region->predecessors; pred;
       pred = pred->next) {
    TGDBG("region %p, pred region: %p\n", region, pred->region);
  }
#endif

  // Now, fix up predecessor list for 'region', and successor lists for each
  // predecessor-predecessor.
  kmp_taskgraph_region_dep_t **dep_p = &region->predecessors;
  while (*dep_p) {
    kmp_taskgraph_region_dep_t *dep = *dep_p;
    if (dep->region->mark == TASKGRAPH_COMBINED) {
      if (!dep->region->successors) {
        dep->region = dep->region->parent;
        dep_p = &dep->next;
      } else {
        kmp_taskgraph_region_dep_t *next = dep->next;
        __kmp_region_dep_recycle(&taskgraph->recycled_deps, dep);
        *dep_p = next;
      }
    } else {
      dep_p = &dep->next;
    }
  }

#ifdef DEBUG_TASKGRAPH
  TGDBG("after pred fixup:\n");
  for (kmp_taskgraph_region_dep_t *pred = region->predecessors; pred;
       pred = pred->next) {
    TGDBG("region %p, pred region: %p\n", region, pred->region);
  }
#endif

  for (kmp_taskgraph_region_dep_t *pp = pred_preds; pp; pp = pp->next) {
    kmp_taskgraph_region_t *pp_region = pp->region;
    dep_p = &pp_region->successors;
    while (*dep_p) {
      kmp_taskgraph_region_dep_t *dep = *dep_p;
      if (dep->region->mark == TASKGRAPH_COMBINED) {
        if (!dep->region->predecessors) {
          dep->region = dep->region->parent;
          dep_p = &dep->next;
        } else {
          kmp_taskgraph_region_dep_t *next = dep->next;
          __kmp_region_dep_recycle(&taskgraph->recycled_deps, dep);
          *dep_p = next;
        }
      } else {
        dep_p = &dep->next;
      }
    }
  }

  return changed;
}

static void __kmp_taskgraph_region_dot(kmp_taskgraph_region_t *region,
                                       const char *name) {
  fprintf(stderr, "digraph %s {\n", name);
  for (kmp_taskgraph_region_t *r = region; r; r = r->next) {
    if (r->mark == TASKGRAPH_DELETED) {
      fprintf(stderr, "\"%p\" [shape=box, label=\"%p(%s) (deleted)\"]\n", r, r,
              __kmp_taskgraph_region_type_name(r->type));
    } else if (r->level == -1) {
      fprintf(stderr, "\"%p\" [shape=box, label=\"%p(%s) (new)\"]\n", r, r,
              __kmp_taskgraph_region_type_name(r->type));
    } else {
      fprintf(stderr, "\"%p\" [shape=box, label=\"%p(%s)\"]\n", r, r,
              __kmp_taskgraph_region_type_name(r->type));
    }
    for (kmp_taskgraph_region_dep_t *succ = r->successors; succ;
         succ = succ->next) {
      fprintf(stderr, "  \"%p\" -> \"%p\" [color=green]\n", r, succ->region);
    }
    for (kmp_taskgraph_region_dep_t *pred = r->predecessors; pred;
         pred = pred->next) {
      fprintf(stderr, "  \"%p\" -> \"%p\" [color=red, constraint=false]\n", r,
              pred->region);
    }
  }
  fprintf(stderr, "}\n");
}

static kmp_int32
__kmp_taskgraph_count_edges_to_dominator(kmp_taskgraph_region_t *reg,
                                         kmp_taskgraph_region_t *dom) {
  kmp_int32 count = __kmp_region_deplist_len(reg->successors) - 1;

  for (kmp_taskgraph_region_dep_t *pred = reg->predecessors; pred;
       pred = pred->next) {
    if (pred->region == dom)
      count++;
    else
      count += __kmp_taskgraph_count_edges_to_dominator(pred->region, dom) + 1;
  }
  count--;

  return count;
}

/// Extract/clone a subgraph of the dependency graph, and rewrite predecessor
/// and successor edges to point to the new cloned part.
//
// The function conceptually starts at the bottom (a list of predecessors
// with some particular dominator) and works up towards the entry point,
// stopping when it hits the aforementioned dominator.
//
// Say we have an irreducible graph like this (each letter represents a region,
// which could be a single task node or an already-processed nested region):
//
//          <S>         (S->A, S->B)
//        _/   \_
//       /       \
//      A         B     (A->C, A->D, B->F, B->G)
//     /  \      /  \
//    C    D    F    G
//    |\     \/     /|
//    | \    /\    / |  (C->H, C->I, D->I, F->H, G->H, G->J)
//    |  \ / __|__/  |
//     \ /\_/_ /     |
//      H__/  I      J
//       \__  |  ___/   (H->E, I->E, J->E)
//          \ | /
//           <E>
//
// We pick the exit node E which has more than one predecessor: H, I and J.
// In this case, H is immediately dominated by the start node, S.
// The 'preds_with_dom' list initially contains the node H.
// We clone the region H then call ourselves with its cloned predecessors,
// until we hit the dominator 'region_dom'.  After rewriting the original
// subgraph's (entering) predecessors and (leaving) successors, we obtain a
// graph like this:
//
//           __ <S>__          (S->A', S->B', S->A, S->B)
//        _/  /   \  \___
//       /  /      \     \
//      A'  B'      A     B    (A'->C', B'->F', B'->G', A->C, A->D, B->F, B->G)
//     /   / \     / \   / \
//    C'  F'  G'  C   D  F* G  (C'->H', F'->H', G'->H', C->I, D->I, G->J)
//     \_ | _/     \ /     /
//        H'        I     J    (H'->E, I->E, J->E)
//         \        |    /
//          \___   / __/
//               <E>
//
// The new cloned subgraph formed from nodes H', C', F', G', A', B' replaces
// the original predecessor of E, H.  Some nodes are now unreachable (F, marked
// with *), and can be deleted.  The start node S now has successors A, B, and
// the new clones A' and B'.
//
// In this way, irreducible graphs are turned into reducible graphs.  A
// critical point is what it means to clone a task node in this way: that is
// discussed in the commentary of __kmp_taskgraph_rewrite_irreducible.

static void __kmp_taskgraph_clone_subgraph(
    kmp_info_t *thread, kmp_taskgraph_record_t *taskgraph,
    kmp_taskgraph_region_t **&alloc_chain,
    kmp_taskgraph_region_t *cloned_nodes[], kmp_taskgraph_region_t *orig_region,
    kmp_taskgraph_region_t *doms[], kmp_taskgraph_region_dep_t *preds_with_dom,
    kmp_taskgraph_region_t *region_dom,
    kmp_taskgraph_region_t ***added_worklist) {
  for (kmp_taskgraph_region_dep_t *pred = preds_with_dom; pred;
       pred = pred->next) {
    kmp_taskgraph_region_t *pred_region = pred->region;
    if (pred_region == region_dom) {
      // NOTE: Adding the new subgraph entry point as a new successor for the
      // dominating block is done in the successor-adding post-pass.
      pred->region = region_dom;
    } else {
      // If we've already processed this predecessor, move on.
      if (cloned_nodes[pred_region->timestamp]) {
        pred->region = cloned_nodes[pred_region->timestamp];
        continue;
      }
      kmp_taskgraph_region_t *cloned_region = __kmp_taskgraph_region_clone(
          thread, taskgraph, alloc_chain, pred_region, nullptr);
      cloned_nodes[pred_region->timestamp] = cloned_region;

      **added_worklist = cloned_region;
      *added_worklist = &cloned_region->next;

      pred->region = cloned_region;
      // Now make a copy of the predecessor list and call ourselves recursively.
      kmp_taskgraph_region_dep_t *cloned_preds = nullptr;
      for (kmp_taskgraph_region_dep_t *p = pred_region->predecessors; p;
           p = p->next) {
        cloned_preds = __kmp_region_deplist_add(
            thread, &taskgraph->recycled_deps, p->region, cloned_preds);
      }
      cloned_region->predecessors = cloned_preds;
      // Note pred_region is the original predecessor region here, not the
      // newly-cloned one.
      __kmp_taskgraph_clone_subgraph(thread, taskgraph, alloc_chain,
                                     cloned_nodes, pred_region, doms,
                                     cloned_preds, region_dom, added_worklist);
    }
  }
}

/// This function uses several strategies to turn an irreducible taskgraph
/// into a reducible taskgraph.
//
// 1. If a node C depends on node B and also node A which dominates C,
//    and if B is also dominated by C, then the dependency of C on A can be
//    dropped.  That is, we know B must execute after A, so we can say
//    execution must proceed A->B->C, and we don't also need to specify the
//    transitive A->C dependency directly.
//
//            A          A
//           / \         |
//          B   )   ->   B
//           \ /         |
//            C          C
//
// 2. Two nodes with the same set of predecessors and successors are turned
//    into a parallel region.  This graph form can arise from use of
//    "inoutset" dependencies.
//
//        A  B  C                               A  B  C
//       / \/ \/ \                              |  |  |
//      /__/\ /\__\                             |  |  |
//      D____X____E  (A+B+C->D & A+B+C->E)  ->  par(D,E)
//       \       /                                 |
//        '--F--'                                  F
//
// 3. We find a node with >1 predecessor R, and group those predecessors by
//    their immediate dominators.  There are two subcases from here.
//
// 3a. If there is more than one group of predecessors (more than one
//     dominator), we pick the dominator with the highest topological-sort
//     level, and we clone the subgraph from that dominator to R.
//
// 3b. If all predecessors share a single dominator, we instead pick the
//     predecessor with the highest incoming/outgoing edge count, and we clone
//     the subgraph from that predecessor to the dominator.
//
// For details of how the subgraph cloning works, see the commentary for
// __kmp_taskgraph_clone_subgraph.
//
// In this way, irreducible edges are gradually "teased apart", and the graph
// thus becomes reducible.
//
// Cloning the subgraph means that task nodes can appear more than once in the
// taskgraph (multiple "instantiations").  The way this should be handled is
// left to later stages of execution, allowing for runtime or API-specific
// techniques to be used.
//
// Say the resulting graph clones a node N into N1 and N2.  Now:
//
//  - All of N1's predecessors and all of N2's predecessors must execute before
//    either N1 or N2 execute.
//  - Only N1 or N2 should execute, not both.
//  - All of N1's, and all of N2's, successors should execute after either N1
//    or N2 executes.
//
// For host execution, this is handled by __kmp_exec_descr_link_instances, etc.

static bool __kmp_taskgraph_rewrite_irreducible(
    kmp_info_t *thread, kmp_taskgraph_record_t *taskgraph,
    kmp_taskgraph_region_t **alloc_chain, kmp_taskgraph_region_t **region_p,
    kmp_taskgraph_region_t *exitregion) {
  kmp_taskgraph_region_t *entryregion = *region_p;
  bool changed = false;

  kmp_int32 worklist_length = 0;
  for (kmp_taskgraph_region_t *r = entryregion; r; r = r->next) {
    // Deleted regions stay deleted.  (We could actually remove these from
    // the worklist here, I think.)
    if (r->mark == TASKGRAPH_DELETED)
      continue;
    r->mark = TASKGRAPH_UNMARKED;
    worklist_length++;
  }

#ifdef DEBUG_TASKGRAPH
  TGDBG("worklist length: %d\n", worklist_length);

  __kmp_taskgraph_region_dot(entryregion, "PredsAndSuccs");
#endif

  kmp_taskgraph_region_t **order =
      (kmp_taskgraph_region_t **)__kmp_fast_allocate(
          thread, worklist_length * sizeof(kmp_taskgraph_region_t *));
  kmp_taskgraph_region_t **doms =
      (kmp_taskgraph_region_t **)__kmp_fast_allocate(
          thread, worklist_length * sizeof(kmp_taskgraph_region_t *));
  memset(doms, 0, worklist_length * sizeof(kmp_taskgraph_region_t *));
  kmp_int32 cursor = worklist_length;
  assert(entryregion->type == TASKGRAPH_REGION_ENTRY);
  __kmp_taskgraph_region_dfs(entryregion, order, cursor, false);
  assert(cursor == 0);
  __kmp_taskgraph_region_doms(order, doms, worklist_length, false);

#ifdef DEBUG_TASKGRAPH
  fprintf(stderr, "digraph {\n");
  for (kmp_int32 i = 0; i < worklist_length; i++) {
    kmp_taskgraph_region_t *b = order[i];
    for (kmp_taskgraph_region_dep_t *succ = b->successors; succ;
         succ = succ->next) {
      fprintf(stderr, "  \"%d\" -> \"%d\"\n", b->timestamp,
              succ->region->timestamp);
    }
    fprintf(stderr, "  \"%d\" -> \"%d\" [color=green, constraint=false]\n",
            b->timestamp, doms[b->timestamp]->timestamp);
  }
  fprintf(stderr, "}\n");
#endif

  // Irreducible regions are handled by duplicating regions, and those new
  // regions need adding to the worklist.  The added_worklist variable stores
  // the head of the new work to be added.
  kmp_taskgraph_region_t *added_worklist = nullptr;
  kmp_taskgraph_region_t **added_worklist_p = &added_worklist;

  bool dropped_preds_p = false;

  for (kmp_int32 i = 0; i < worklist_length; i++) {
    kmp_taskgraph_region_t *region = order[i];
    if (__kmp_region_deplist_len(region->predecessors) < 2)
      continue;
    TGDBG("checking region %p for redundant predecessors\n", region);
    kmp_taskgraph_region_dep_t **predp = &region->predecessors;
    while (*predp) {
      kmp_taskgraph_region_dep_t *pred = *predp;

      bool passes_pred = false;
      for (kmp_taskgraph_region_dep_t *rest = region->predecessors; rest;
           rest = rest->next) {
        if (rest->region == pred->region)
          continue;
        kmp_taskgraph_region_t *dom = doms[rest->region->timestamp];
        TGDBG("pred region: %p, next: %p\n", pred->region, rest->region);
        while (true) {
          TGDBG("check against dom: %p\n", dom);
          if (dom == pred->region) {
            passes_pred = true;
            break;
          } else if (dom == doms[dom->timestamp]) {
            break;
          } else {
            dom = doms[dom->timestamp];
          }
        }
        if (passes_pred)
          break;
      }

      if (passes_pred) {
        // We can drop this predecessor.
        TGDBG("dropping pred %p from region %p, dom %p\n", pred->region, region,
              doms[pred->region->timestamp]);
        kmp_taskgraph_region_dep_t *next = pred->next;
        kmp_taskgraph_region_dep_t **succp = &pred->region->successors;
        while (*succp) {
          kmp_taskgraph_region_dep_t *succ = *succp;
          if (succ->region == region) {
            kmp_taskgraph_region_dep_t *nexts = succ->next;
            __kmp_region_dep_recycle(&taskgraph->recycled_deps, succ);
            *succp = nexts;
          } else {
            succp = &succ->next;
          }
        }
        __kmp_region_dep_recycle(&taskgraph->recycled_deps, pred);
        *predp = next;
        dropped_preds_p = true;
      } else {
        predp = &pred->next;
      }
    }
  }

  if (dropped_preds_p)
    return true;

  kmp_bitset_t **pred_bitsets = nullptr;
  kmp_bitset_t **succ_bitsets = nullptr;

  bool regions_combined_p = false;

  for (kmp_int32 i = 0; i < worklist_length; i++) {
    kmp_taskgraph_region_t *region = order[i];
    struct {
      kmp_taskgraph_region_t *dom;
      kmp_int32 count;
    } dom_groups[worklist_length];
    kmp_int32 num_groups = 0;
    kmp_int32 npreds = __kmp_region_deplist_len(region->predecessors);
    if (npreds >= 2) {
      kmp_taskgraph_region_dep_t *pred;
      for (pred = region->predecessors; pred; pred = pred->next) {
        kmp_taskgraph_region_t *pred_region = pred->region;
        kmp_taskgraph_region_t *this_dom = doms[pred_region->timestamp];
#ifdef DEBUG_TASKGRAPH
        kmp_int32 edges_to_dom =
            __kmp_taskgraph_count_edges_to_dominator(pred_region, this_dom);
        TGDBG("this pred: %p, edges_to_dom=%d\n", pred_region, edges_to_dom);
#endif
        bool found = false;
        for (kmp_int32 grp = 0; grp < num_groups; grp++) {
          if (dom_groups[grp].dom == this_dom) {
            dom_groups[grp].count++;
            found = true;
            break;
          }
        }
        if (!found) {
          dom_groups[num_groups].dom = this_dom;
          dom_groups[num_groups].count = 1;
          num_groups++;
        }
      }

      if (num_groups == 1 && region->mark != TASKGRAPH_COMBINED) {
        TGDBG("region %p: all predecessors have a single dominator\n", region);

        if (!pred_bitsets) {
          pred_bitsets = (kmp_bitset_t **)__kmp_fast_allocate(
              thread, sizeof(kmp_bitset_t *) * worklist_length);
          succ_bitsets = (kmp_bitset_t **)__kmp_fast_allocate(
              thread, sizeof(kmp_bitset_t *) * worklist_length);

          for (kmp_int32 i = 0; i < worklist_length; i++) {
            pred_bitsets[i] = __kmp_bitset_alloc(thread, worklist_length);
            succ_bitsets[i] = __kmp_bitset_alloc(thread, worklist_length);
          }

          for (kmp_int32 j = 0; j < worklist_length; j++) {
            kmp_taskgraph_region_t *reg = order[j];

            for (pred = reg->predecessors; pred; pred = pred->next) {
              __kmp_bitset_set(pred_bitsets[reg->timestamp],
                               pred->region->timestamp);
            }

            for (kmp_taskgraph_region_dep_t *succ = reg->successors; succ;
                 succ = succ->next) {
              __kmp_bitset_set(succ_bitsets[reg->timestamp],
                               succ->region->timestamp);
            }
          }
        }

        kmp_taskgraph_region_dep_t *equal_deps_chain = nullptr;

        kmp_int32 same_preds_and_succs = 1;
        bool any_mutex_p = __kmp_taskgraph_region_mutex_p(region);
        // FIXME: We might be able to do a bit better than this by hashing.
        for (kmp_int32 j = i + 1; j < worklist_length; j++) {
          if (order[j]->mark != TASKGRAPH_COMBINED &&
              __kmp_bitset_equal(pred_bitsets[j], pred_bitsets[i]) &&
              __kmp_bitset_equal(succ_bitsets[j], succ_bitsets[i])) {
            TGDBG("regions %p and %p share all predecessors/successors\n",
                  order[i], order[j]);
            same_preds_and_succs++;
            equal_deps_chain = __kmp_region_deplist_add(
                thread, &taskgraph->recycled_deps, order[j], equal_deps_chain);
            if (__kmp_taskgraph_region_mutex_p(order[j]))
              any_mutex_p = true;
          }
        }
        if (same_preds_and_succs > 1) {
          kmp_taskgraph_region_type region_type =
              any_mutex_p ? TASKGRAPH_REGION_EXCLUSIVE
                          : TASKGRAPH_REGION_PARALLEL;
          kmp_taskgraph_region_t *par_region = __kmp_taskgraph_region_alloc(
              thread, taskgraph, alloc_chain, region_type, same_preds_and_succs,
              nullptr);
          par_region->inner.children[0] = region;
          region->mark = TASKGRAPH_COMBINED;
          region->parent = par_region;
          for (kmp_int32 j = 1; j < same_preds_and_succs; j++) {
            kmp_taskgraph_region_dep_t *next = equal_deps_chain->next;
            par_region->inner.children[j] = equal_deps_chain->region;
            equal_deps_chain->region->mark = TASKGRAPH_COMBINED;
            equal_deps_chain->region->parent = par_region;
            __kmp_region_dep_recycle(&taskgraph->recycled_deps,
                                     equal_deps_chain);
            equal_deps_chain = next;
          }
          par_region->predecessors =
              par_region->inner.children[0]->predecessors;
          par_region->inner.children[0]->predecessors = nullptr;
          par_region->successors = par_region->inner.children[0]->successors;
          par_region->inner.children[0]->successors = nullptr;

          // Redirect incoming deps to point to new parallel region.
          for (pred = par_region->predecessors; pred; pred = pred->next) {
            kmp_taskgraph_region_t *pred_region = pred->region;
            kmp_taskgraph_region_dep_t **succp = &pred_region->successors;
            while (*succp) {
              kmp_taskgraph_region_dep_t *succ = *succp;
              if (succ->region == par_region->inner.children[0]) {
                succ->region = par_region;
                succp = &succ->next;
              } else {
                bool found = false;
                for (kmp_int32 j = 1; j < same_preds_and_succs; j++) {
                  if (succ->region == par_region->inner.children[j]) {
                    found = true;
                    break;
                  }
                }
                if (found) {
                  kmp_taskgraph_region_dep_t *next = succ->next;
                  __kmp_region_dep_recycle(&taskgraph->recycled_deps, succ);
                  *succp = next;
                } else {
                  succp = &succ->next;
                }
              }
            }
          }

          for (kmp_taskgraph_region_dep_t *succ = par_region->successors; succ;
               succ = succ->next) {
            kmp_taskgraph_region_t *succ_region = succ->region;
            kmp_taskgraph_region_dep_t **predp = &succ_region->predecessors;
            while (*predp) {
              kmp_taskgraph_region_dep_t *pred = *predp;
              if (pred->region == par_region->inner.children[0]) {
                pred->region = par_region;
                predp = &pred->next;
              } else {
                bool found = false;
                for (kmp_int32 j = 1; j < same_preds_and_succs; j++) {
                  if (pred->region == par_region->inner.children[j]) {
                    found = true;
                    break;
                  }
                }
                if (found) {
                  kmp_taskgraph_region_dep_t *next = pred->next;
                  __kmp_region_dep_recycle(&taskgraph->recycled_deps, pred);
                  *predp = next;
                } else {
                  predp = &pred->next;
                }
              }
            }
          }

          par_region->next = region->next;
          region->next = par_region;

          regions_combined_p = true;
        }
      }

      if (regions_combined_p)
        continue;

      assert(num_groups >= 1);

      TGDBG("should split region %p (%d)\n", region, region->timestamp);
      TGDBG("clone graph to dominator: %p (%d, %s)\n", doms[region->timestamp],
            doms[region->timestamp]->timestamp,
            __kmp_taskgraph_region_type_name(doms[region->timestamp]->type));
      kmp_taskgraph_region_t *region_dom = doms[region->timestamp];
      kmp_int32 grp = -1;
      kmp_int32 highest_dom = -1;
      // Choose a dominator.  We pick one with the highest level, i.e.
      // with the largest chain of dependents.  Anything we pick should
      // be irreducible, because we've already tried the serial-parallel
      // decomposition.
      for (kmp_int32 j = 0; j < num_groups; j++) {
        if (dom_groups[j].dom->level > highest_dom) {
          grp = j;
          highest_dom = dom_groups[j].dom->level;
        }
      }

      // Separate out the predecessors with this dominator (identified by
      // grp).
      kmp_taskgraph_region_dep_t *preds_with_dom = nullptr;
      kmp_taskgraph_region_dep_t **pwd_tail = &preds_with_dom;
      kmp_taskgraph_region_dep_t **pred_cursor = &region->predecessors;
      TGDBG("before splitting we have %d preds\n",
            __kmp_region_deplist_len(region->predecessors));
      while (*pred_cursor) {
        kmp_taskgraph_region_dep_t *this_pred = *pred_cursor;
        kmp_taskgraph_region_t *dom = doms[this_pred->region->timestamp];
        if (dom == dom_groups[grp].dom) {
          *pwd_tail = this_pred;
          pwd_tail = &this_pred->next;
          *pred_cursor = this_pred->next;
        } else {
          pred_cursor = &this_pred->next;
        }
      }
      // Finish list.
      *pwd_tail = nullptr;

      if (!region->predecessors) {
        kmp_int32 highest = -1;
        kmp_taskgraph_region_dep_t **use_pred = nullptr;
        // This can only happen if...
        assert(num_groups == 1);
        region->predecessors = preds_with_dom;
        for (kmp_taskgraph_region_dep_t **rp = &region->predecessors; *rp;
             rp = &(*rp)->next) {
          kmp_int32 count = __kmp_taskgraph_count_edges_to_dominator(
              (*rp)->region, dom_groups[grp].dom);
          TGDBG("for pred %p, outgoing edges to dom = %d\n", (*rp)->region,
                count);
          if (count > highest) {
            highest = count;
            use_pred = rp;
          }
        }
        TGDBG("using pred %p\n", (*use_pred)->region);
        // Pick the single predecessor with the largest outgoing edge
        // count (the "most complicated" predecessor).
        preds_with_dom = *use_pred;
        *use_pred = (*use_pred)->next;
        preds_with_dom->next = nullptr;
      }

      kmp_taskgraph_region_dep_t *unlinked_successors = nullptr;

      // Unlink successors for preds_with_dom nodes, and record where they
      // came from.
      for (pred = preds_with_dom; pred; pred = pred->next) {
        kmp_taskgraph_region_dep_t **succp = &pred->region->successors;
        while (*succp) {
          kmp_taskgraph_region_dep_t *succ = *succp;
          kmp_taskgraph_region_t *succ_region = succ->region;
          if (succ_region == region) {
            kmp_taskgraph_region_dep_t *next = succ->next;
            __kmp_region_dep_recycle(&taskgraph->recycled_deps, succ);
            TGDBG("unlinking successor %p -> %p\n", pred->region, region);
            unlinked_successors =
                __kmp_region_deplist_add(thread, &taskgraph->recycled_deps,
                                         pred->region, unlinked_successors);
            *succp = next;
          } else {
            succp = &succ->next;
          }
        }
      }

      TGDBG("after splitting, # preds_with_dom=%d, others %d\n",
            __kmp_region_deplist_len(preds_with_dom),
            __kmp_region_deplist_len(region->predecessors));
      *pwd_tail = nullptr;
      kmp_taskgraph_region_t *cloned_nodes[worklist_length];
      memset(cloned_nodes, 0,
             sizeof(kmp_taskgraph_region_t *) * worklist_length);
      __kmp_taskgraph_clone_subgraph(thread, taskgraph, alloc_chain,
                                     cloned_nodes, region, doms, preds_with_dom,
                                     region_dom, &added_worklist_p);
      // Now fill in the successors for the cloned regions.
      for (kmp_int32 n = 0; n < worklist_length; n++) {
        kmp_taskgraph_region_t *cloned_region = cloned_nodes[n];
        if (!cloned_region)
          continue;
        for (kmp_taskgraph_region_dep_t *pred = cloned_region->predecessors;
             pred; pred = pred->next) {
          kmp_taskgraph_region_t *pred_region = pred->region;
          pred_region->successors =
              __kmp_region_deplist_add(thread, &taskgraph->recycled_deps,
                                       cloned_region, pred_region->successors);
        }
      }

#ifdef DEBUG_TASKGRAPH
      TGDBG("before appending:\n");
      for (pred = region->predecessors; pred; pred = pred->next) {
        TGDBG("region %p, pred: %p\n", region, pred);
      }
#endif

      // Re-attach redirected predecessor list to region's predecessors.
      pred = region->predecessors;
      if (pred) {
        while (pred && pred->next)
          pred = pred->next;
        pred->next = preds_with_dom;
      } else {
        region->predecessors = preds_with_dom;
      }

#ifdef DEBUG_TASKGRAPH
      TGDBG("after appending:\n");
      for (pred = region->predecessors; pred; pred = pred->next) {
        TGDBG("region %p, pred: %p\n", region, pred);
      }
#endif

      // Redirect the unlinked successors from the region's original
      // predecessors so that the new (cloned) predecessors still point to
      // the region.
      for (kmp_taskgraph_region_dep_t *succ = unlinked_successors; succ;) {
        kmp_taskgraph_region_t *cloned_reg =
            cloned_nodes[succ->region->timestamp];
        kmp_taskgraph_region_dep_t *next = succ->next;
        __kmp_region_dep_recycle(&taskgraph->recycled_deps, succ);
        TGDBG("add successor to cloned region: %p -> %p\n", cloned_reg, region);
        cloned_reg->successors = __kmp_region_deplist_add(
            thread, &taskgraph->recycled_deps, region, cloned_reg->successors);
        succ = next;
      }

      // Cloning subgraph invalidates e.g. the timestamp fields: just do
      // one round of transformation.  We could possibly do more if we
      // were careful.

      changed = true;
    }
    if (changed)
      break;
  }

  if (regions_combined_p)
    changed = true;

  if (pred_bitsets) {
    for (kmp_int32 j = 0; j < worklist_length; j++) {
      __kmp_bitset_free(thread, pred_bitsets[j]);
      __kmp_bitset_free(thread, succ_bitsets[j]);
    }
    __kmp_fast_free(thread, pred_bitsets);
    __kmp_fast_free(thread, succ_bitsets);
  }

  *added_worklist_p = nullptr;
  added_worklist = __kmp_region_worklist_reverse(added_worklist);

  kmp_taskgraph_region_t *last = exitregion;
  while (last && last->next)
    last = last->next;
  last->next = added_worklist;

  TGDBG("starting trim dead edges...\n");

  for (kmp_taskgraph_region_t *r = entryregion; r; r = r->next) {
    r->mark = TASKGRAPH_UNMARKED;
  }

  // Remove any regions which are now unreachable by DFS from the exit
  // region, and any connected dependency edges.
  int idx = 0;
  __kmp_taskgraph_region_dfs(exitregion, nullptr, idx, true);
  for (kmp_taskgraph_region_t *r = entryregion; r; r = r->next) {
    if (r->mark == TASKGRAPH_UNMARKED) {
      r->mark = TASKGRAPH_DELETED;

      __kmp_region_deplist_recycle(&taskgraph->recycled_deps, r->successors);
      r->successors = nullptr;

      // Delete predecessors for deleted nodes (and corresponding
      // successors).
      kmp_taskgraph_region_dep_t **predp = &r->predecessors;
      while (*predp) {
        kmp_taskgraph_region_dep_t *pred = *predp;
        if (pred->region->mark != TASKGRAPH_UNMARKED) {
          kmp_taskgraph_region_dep_t **succp = &pred->region->successors;
          while (*succp) {
            kmp_taskgraph_region_dep_t *succ = *succp;
            if (succ->region == r) {
              kmp_taskgraph_region_dep_t *next = succ->next;
              __kmp_region_dep_recycle(&taskgraph->recycled_deps, succ);
              *succp = next;
            } else {
              succp = &succ->next;
            }
          }
        }
        kmp_taskgraph_region_dep_t *next = pred->next;
        __kmp_region_dep_recycle(&taskgraph->recycled_deps, pred);
        *predp = next;
      }
    }
  }

  TGDBG("done trimming dead edges.\n");

  __kmp_taskgraph_region_chain_prune(&entryregion);
  __kmp_taskgraph_region_worklist_check(thread, taskgraph, entryregion,
                                        "after irreducible handling");

  worklist_length = 0;
  for (kmp_taskgraph_region_t *r = entryregion; r; r = r->next) {
    r->mark = TASKGRAPH_UNMARKED;
    worklist_length++;
  }

  // Recalculate topological sort
  kmp_int32 max_level = -1;
  kmp_taskgraph_region_t *r = entryregion;
  kmp_int32 outidx = 0;
  kmp_taskgraph_region_t *order_out[worklist_length];
  for (kmp_int32 i = 0; i < worklist_length; i++, r = r->next) {
    if (r->mark == TASKGRAPH_UNMARKED) {
      kmp_int32 level =
          __kmp_taskgraph_topological_order(r, order_out, &outidx);
      max_level = level > max_level ? level : max_level;
    }
  }

  // Re-sort worklist wrt. topological order calculated above.
  kmp_taskgraph_region_t **relink = &entryregion;
  for (kmp_int32 i = 0; i < worklist_length; i++) {
    *relink = order_out[i];
    relink = &order_out[i]->next;
  }
  *relink = nullptr;

#ifdef DEBUG_TASKGRAPH
  __kmp_taskgraph_region_dot(entryregion, "PredsAndSuccsAfter");
#endif

  return changed;
}

/// Build a nested region structure out of a recorded taskgraph.
//
// The algorithm proceeds by alternating two phases until a single top-level
// node is reached.  Briefly, and glossing over some details:
//
// 1. Serial-parallel decomposition.  Chains of single-successor,
//    single-predecessor nodes are collapsed into a "sequential" region, and
//    nodes with >1 predecessor, where each predecessor has a single
//    predecessor and a single successor, are collapsed into "parallel" regions.
//
// 2. Irreducible-graph processing.  Several techniques are used to turn graphs
//    not handled by step (1) into graphs that can be handled by that step.
//
// Notably, simple graphs that can be handled entirely by step (1) avoid doing
// much of the heavier processing involved in step (2), so the common case
// should be relatively fast.

static kmp_taskgraph_region_t *__kmp_taskgraph_build_regions(
    kmp_info_t *thread, kmp_taskgraph_record_t *taskgraph,
    kmp_taskgraph_region_t **&alloc_chain, kmp_taskgraph_region_t *entryregion,
    kmp_taskgraph_region_t *exitregion) {
  bool changed;
  kmp_int32 phase = 0;

#ifdef DEBUG_TASKGRAPH
  __kmp_taskgraph_region_dot(entryregion, "InitialPredsAndSuccs");
#endif

  __kmp_taskgraph_region_chain_clear_marks(entryregion);

  while (true) {
    do {
      changed = false;
      TGDBG("starting seq pass\n");
      for (kmp_taskgraph_region_t **seq_head = &entryregion; *seq_head;
           seq_head = &(*seq_head)->next) {
        TGDBG("consider %s region: %p\n",
              __kmp_taskgraph_region_type_name((*seq_head)->type), *seq_head);
        if ((*seq_head)->mark == TASKGRAPH_COMBINED) {
          TGDBG("already combined\n");
          continue;
        }
        changed |= __kmp_taskgraph_collapse_sequence(thread, taskgraph,
                                                     alloc_chain, seq_head,
                                                     /*parent=*/nullptr, phase);
        TGDBG("changed: %s\n", changed ? "true" : "false");
      }
      ++phase;
      __kmp_taskgraph_region_chain_prune(&entryregion);
      __kmp_taskgraph_region_worklist_check(thread, taskgraph, entryregion,
                                            "after seq collapse");
      TGDBG("starting par/unordered pass\n");
      for (kmp_taskgraph_region_t **par_head = &entryregion; *par_head;
           par_head = &(*par_head)->next) {
        TGDBG("consider %s region: %p\n",
              __kmp_taskgraph_region_type_name((*par_head)->type), *par_head);
        if ((*par_head)->mark == TASKGRAPH_COMBINED) {
          TGDBG("already combined\n");
          continue;
        }
        changed |= __kmp_taskgraph_collapse_par_exclusive(
            thread, taskgraph, alloc_chain, par_head, /*parent=*/nullptr,
            phase);
        TGDBG("changed: %s\n", changed ? "true" : "false");
      }
      ++phase;
      __kmp_taskgraph_region_chain_prune(&entryregion);
      __kmp_taskgraph_region_worklist_check(thread, taskgraph, entryregion,
                                            "after par collapse");
    } while (changed);

    if (entryregion->type == TASKGRAPH_REGION_ENTRY) {
      if (__kmp_region_deplist_len(entryregion->successors) == 1) {
        kmp_taskgraph_region_t *one_region = entryregion->successors->region;
        if (__kmp_region_deplist_len(one_region->successors) == 1) {
          kmp_taskgraph_region_t *maybe_exit = one_region->successors->region;
          if (maybe_exit->type == TASKGRAPH_REGION_EXIT)
            return one_region;
        }
      }
    } else {
      fprintf(stderr, "FIXME: Expected entry region!\n");
      return entryregion;
    }

    TGDBG("attempting to collapse irreducible regions\n");

    changed |= __kmp_taskgraph_rewrite_irreducible(
        thread, taskgraph, alloc_chain, &entryregion, exitregion);

    if (!changed) {
      fprintf(stderr, "FIXME: Failed to transform irreducible graph\n");
      return entryregion;
    }
  }

  return entryregion;
}

static void __kmp_taskgraph_count_nodes(kmp_taskgraph_region_t *region) {
  switch (region->type) {
  case TASKGRAPH_REGION_ENTRY:
  case TASKGRAPH_REGION_EXIT:
    return;
  case TASKGRAPH_REGION_NODE:
  case TASKGRAPH_REGION_WAIT: {
    TGDBG("process region %p\n", region);
    region->task.node->u.resolved.count++;
    kmp_taskgraph_region_t *last_region =
        region->task.node->u.resolved.last_region;
    TGDBG("last region: %p\n", last_region);
    if (last_region) {
      kmp_taskgraph_region_t *next = last_region->task.next_instance;
      TGDBG("next: %p\n", next);
      last_region->task.next_instance = region;
      region->task.next_instance = next;
    }
    region->task.node->u.resolved.last_region = region;
    return;
  }
  default:
    for (kmp_int32 n = 0; n < region->inner.num_children; n++) {
      __kmp_taskgraph_count_nodes(region->inner.children[n]);
    }
  }
}

static void __kmp_taskgraph_gather_mutex_sets(kmp_info_t *thread,
                                              kmp_taskgraph_region_t *region,
                                              const kmp_bitset_t *held) {
  switch (region->type) {
  case TASKGRAPH_REGION_ENTRY:
  case TASKGRAPH_REGION_EXIT:
  case TASKGRAPH_REGION_WAIT:
    return;
  case TASKGRAPH_REGION_NODE: {
#ifdef DEBUG_TASKGRAPH
    if (region->mutexset && __kmp_bitset_subset_p(held, region->mutexset)) {
      TGDBG("node is mutually exclusive with held: 0x%llx <: 0x%llx\n",
            (unsigned long long)region->mutexset->bits[0],
            (unsigned long long)held->bits[0]);
    }
#endif
    return;
  }
  case TASKGRAPH_REGION_SEQUENTIAL: {
    kmp_bitset_t *seq_held = __kmp_bitset_alloc(thread, held->bitsize);
    __kmp_bitset_clearall(seq_held);
    for (kmp_int32 child = 0; child < region->inner.num_children; child++) {
      __kmp_taskgraph_gather_mutex_sets(thread, region->inner.children[child],
                                        held);
      if (region->inner.children[child]->mutexset)
        __kmp_bitset_or(seq_held, seq_held,
                        region->inner.children[child]->mutexset);
    }
    region->mutexset = seq_held;
    return;
  }
  case TASKGRAPH_REGION_PARALLEL:
  case TASKGRAPH_REGION_EXCLUSIVE: {
    kmp_bitset_t *par_held = __kmp_bitset_alloc(thread, held->bitsize);
    kmp_bitset_t *conflicts = __kmp_bitset_alloc(thread, held->bitsize);
    while (true) {
      __kmp_bitset_clearall(par_held);
      for (kmp_int32 child = 0; child < region->inner.num_children; child++) {
        __kmp_bitset_clearall(conflicts);
        for (kmp_int32 other = 0; other < region->inner.num_children; other++) {
          if (other != child) {
            if (!region->inner.children[other]->mutexset)
              __kmp_taskgraph_gather_mutex_sets(
                  thread, region->inner.children[other], held);
            if (region->inner.children[other]->mutexset)
              __kmp_bitset_or(conflicts, conflicts,
                              region->inner.children[other]->mutexset);
          }
        }
        __kmp_taskgraph_gather_mutex_sets(thread, region->inner.children[child],
                                          conflicts);
        if (region->inner.children[child]->mutexset)
          __kmp_bitset_or(par_held, par_held,
                          region->inner.children[child]->mutexset);
      }
      if (!region->mutexset) {
        region->mutexset = par_held;
      } else if (__kmp_bitset_equal(region->mutexset, par_held)) {
        TGDBG("par mutexes stabilized, exiting loop\n");
        break;
      } else {
        TGDBG("par mutexes not stable, iterating\n");
        __kmp_bitset_copy(region->mutexset, par_held);
        __kmp_bitset_free(thread, par_held);
      }
    }
    __kmp_bitset_free(thread, conflicts);
    return;
  }
  }
}

static int __kmp_popcount_cmp(const void *a, const void *b) {
  const kmp_taskgraph_region_t *reg_a = *(kmp_taskgraph_region_t **)a;
  const kmp_taskgraph_region_t *reg_b = *(kmp_taskgraph_region_t **)b;
  kmp_int32 popc_a = 0, popc_b = 0;
  if (reg_a->mutexset)
    popc_a = __kmp_bitset_popcount(reg_a->mutexset);
  if (reg_b->mutexset)
    popc_b = __kmp_bitset_popcount(reg_b->mutexset);
  if (popc_a > popc_b)
    return -1;
  else if (popc_a < popc_b)
    return 1;
  return 0;
}

/// Find "mutexinoutset" regions that can be represented without explicit
// mutexes, i.e. using "TASKGRAPH_REGION_EXCLUSIVE".

static void __kmp_taskgraph_find_exclusive_regions(
    kmp_info_t *thread, kmp_taskgraph_record_t *taskgraph,
    kmp_taskgraph_region_t **&alloc_chain, kmp_taskgraph_region_t **region_p) {
  kmp_taskgraph_region_t *region = *region_p;
  switch (region->type) {
  case TASKGRAPH_REGION_ENTRY:
  case TASKGRAPH_REGION_EXIT:
  case TASKGRAPH_REGION_NODE:
  case TASKGRAPH_REGION_WAIT:
    break;
  case TASKGRAPH_REGION_SEQUENTIAL:
  case TASKGRAPH_REGION_PARALLEL: {
    for (kmp_int32 c = 0; c < region->inner.num_children; c++) {
      __kmp_taskgraph_find_exclusive_regions(thread, taskgraph, alloc_chain,
                                             &region->inner.children[c]);
    }
    break;
  }
  case TASKGRAPH_REGION_EXCLUSIVE: {
    qsort(region->inner.children, region->inner.num_children,
          sizeof(kmp_taskgraph_region_t *), __kmp_popcount_cmp);
    for (kmp_int32 c = 0; c < region->inner.num_children; c++) {
      TGDBG("building tree: region mutexset = 0x%llx\n",
            (unsigned long long)region->inner.children[c]->mutexset
                ? region->inner.children[c]->mutexset->bits[0]
                : 0);
      region->inner.children[c]->mark = TASKGRAPH_UNMARKED;
    }
    kmp_bitset_t *conflicts =
        __kmp_bitset_alloc(thread, region->mutexset->bitsize);
    kmp_bitset_t *subsets_cover =
        __kmp_bitset_alloc(thread, region->mutexset->bitsize);
    __kmp_bitset_copy(conflicts, region->mutexset);
    bool irregular = false;
    kmp_int32 combined_children = 0;
    for (kmp_int32 c = 0; c < region->inner.num_children; c++) {
      kmp_bitset_t *candidate = region->inner.children[c]->mutexset;
      if (__kmp_bitset_empty_p(candidate))
        continue;
      __kmp_bitset_clearall(subsets_cover);
      bool found_subset = false;
      bool other_overlaps = false;
      for (kmp_int32 d = c + 1; d < region->inner.num_children; d++) {
        // This could test for a subset in some cases, but that adds
        // complication for later processing.  Maybe revisit later if it
        // seems worthwhile.
        // E.g. if we have deps like this:
        //
        // #pragma omp task depend(mutexinoutset: deps[0], deps[1]) { /*a*/ }
        // #pragma omp task depend(mutexinoutset: deps[0]) { /*b*/ }
        // #pragma omp task depend(mutexinoutset: deps[1]) { /*c*/ }
        //
        // This could be represented as:
        //
        // exclusive {
        //   node: a
        //   parallel {
        //     node: b
        //     node: c
        //   }
        // }
        //
        // We're not doing that yet though.
        if (__kmp_bitset_equal(candidate,
                               region->inner.children[d]->mutexset)) {
          found_subset = true;
          __kmp_bitset_or(subsets_cover, subsets_cover,
                          region->inner.children[d]->mutexset);
        } else if (__kmp_bitset_intersect_p(
                       candidate, region->inner.children[d]->mutexset)) {
          other_overlaps = true;
          break;
        }
      }
      if (!found_subset || other_overlaps)
        continue;
      if (!__kmp_bitset_equal(subsets_cover, candidate)) {
        TGDBG("subsets cover: 0x%llx, candidate: 0x%llx\n",
              (unsigned long long)subsets_cover->bits[0],
              (unsigned long long)candidate->bits[0]);
        irregular = true;
        break;
      }
      for (kmp_int32 d = c + 1; d < region->inner.num_children; d++) {
        if (region->inner.children[d]->mutexset_parent)
          continue;
        // As above wrt. subsets.
        if (__kmp_bitset_equal(candidate,
                               region->inner.children[d]->mutexset)) {
          TGDBG("set index %d's parent to index %d\n", d, c);
          region->inner.children[d]->mutexset_parent =
              region->inner.children[c];
          combined_children++;
          __kmp_bitset_and_not(conflicts, conflicts, candidate);
        }
      }
    }
    TGDBG("irregular: %s\n", irregular ? "true" : "false");
    TGDBG("final conflicts: 0x%llx\n", (unsigned long long)conflicts->bits[0]);
    __kmp_bitset_free(thread, subsets_cover);
    region->type = TASKGRAPH_REGION_PARALLEL;
    if (!irregular && __kmp_bitset_empty_p(conflicts)) {
      TGDBG("transforming exclusive region %p\n", region);
      TGDBG("orig region children: %d\n", region->inner.num_children);
      TGDBG("combined children: %d\n", combined_children);
      if (region->inner.num_children == combined_children + 1) {
        region->type = TASKGRAPH_REGION_EXCLUSIVE;
      } else {
        kmp_taskgraph_region_t *new_par = __kmp_taskgraph_region_alloc(
            thread, taskgraph, alloc_chain, TASKGRAPH_REGION_PARALLEL,
            region->inner.num_children - combined_children, nullptr);
        for (kmp_int32 c = region->inner.num_children - 1; c >= 0; c--) {
          kmp_taskgraph_region_t *child = region->inner.children[c];
          // Make mutex set into a circular list.
          if (child->mutexset_parent && child->mark != TASKGRAPH_TEMP_MARK) {
            if (!child->mutexset_parent->mutexset_parent) {
              // child <-> parent
              child->mutexset_parent->mutexset_parent = child;
              child->mutexset_parent->mark = TASKGRAPH_TEMP_MARK;
            } else {
              kmp_taskgraph_region_t *parent = child->mutexset_parent;
              child->mutexset_parent = parent->mutexset_parent;
              parent->mutexset_parent = child;
              parent->mark = TASKGRAPH_TEMP_MARK;
            }
          }
        }
        kmp_int32 idx = 0;
        for (kmp_int32 c = 0; c < region->inner.num_children; c++) {
          kmp_taskgraph_region_t *child = region->inner.children[c];
          TGDBG("process child: %p\n", child);
          if (child->mutexset_parent && child->mark != TASKGRAPH_COMBINED) {
            kmp_int32 elems = 0;
            kmp_taskgraph_region_t *next = child;
            do {
              elems++;
              next = next->mutexset_parent;
            } while (next != child);
            TGDBG("make exclusive region with %d children\n", elems);
            kmp_taskgraph_region_t *excl_region = __kmp_taskgraph_region_alloc(
                thread, taskgraph, alloc_chain, TASKGRAPH_REGION_EXCLUSIVE,
                elems, nullptr);
            kmp_int32 excl_child = 0;
            next = child;
            do {
              excl_region->inner.children[excl_child++] = next;
              next->mark = TASKGRAPH_COMBINED;
              next = next->mutexset_parent;
            } while (next != child);
            assert(excl_child == excl_region->inner.num_children);
            new_par->inner.children[idx++] = excl_region;
          } else if (!child->mutexset_parent) {
            new_par->inner.children[idx++] = child;
          }
        }
        TGDBG("idx=%d, supposed to be %d\n", idx, new_par->inner.num_children);
        assert(idx == new_par->inner.num_children);
        *region_p = new_par;
        region->mark = TASKGRAPH_DELETED;
      }
    }
    __kmp_bitset_free(thread, conflicts);
    break;
  }
  default:
    assert(false && "unreachable");
  }
}

/// Strip mutex sets from taskgraph region, except those needed at runtime.

static kmp_int32
__kmp_taskgraph_strip_mutex_sets(kmp_info_t *thread,
                                 kmp_taskgraph_region_t *region,
                                 bool in_exclusive = false) {
  kmp_int32 mutexes_needed = 0;
  switch (region->type) {
  case TASKGRAPH_REGION_ENTRY:
  case TASKGRAPH_REGION_EXIT:
  case TASKGRAPH_REGION_WAIT:
    assert(!region->mutexset);
    break;
  case TASKGRAPH_REGION_NODE:
    if (region->mutexset) {
      if (in_exclusive) {
        __kmp_bitset_free(thread, region->mutexset);
        region->mutexset = nullptr;
      } else {
        // FIXME: This might be pessimistic -- the remaining mutex sets might
        // have holes or duplicates.  We could compact them.
        kmp_int32 m = region->mutexset->bitsize;
        mutexes_needed = std::max(mutexes_needed, m);
      }
    }
    break;
  case TASKGRAPH_REGION_EXCLUSIVE: {
    if (region->mutexset) {
      __kmp_bitset_free(thread, region->mutexset);
      region->mutexset = nullptr;
    }
    for (kmp_int32 c = 0; c < region->inner.num_children; c++) {
      kmp_int32 m = __kmp_taskgraph_strip_mutex_sets(
          thread, region->inner.children[c], true);
      mutexes_needed = std::max(mutexes_needed, m);
    }
    break;
  }
  default: {
    if (region->mutexset) {
      __kmp_bitset_free(thread, region->mutexset);
      region->mutexset = nullptr;
    }
    for (kmp_int32 c = 0; c < region->inner.num_children; c++) {
      kmp_int32 m = __kmp_taskgraph_strip_mutex_sets(
          thread, region->inner.children[c], in_exclusive);
      mutexes_needed = std::max(mutexes_needed, m);
    }
  }
  }
  return mutexes_needed;
}

static void __kmp_taskgraph_exclusive_regions(
    kmp_info_t *thread, kmp_taskgraph_record_t *taskgraph,
    kmp_taskgraph_region_t **&alloc_chain, kmp_taskgraph_region_t **region_p,
    kmp_int32 max_mutex) {
  kmp_bitset_t *top = __kmp_bitset_alloc(thread, max_mutex);
  __kmp_bitset_clearall(top);
  __kmp_taskgraph_gather_mutex_sets(thread, *region_p, top);
  __kmp_taskgraph_find_exclusive_regions(thread, taskgraph, alloc_chain,
                                         region_p);
  kmp_int32 num_mutexes = __kmp_taskgraph_strip_mutex_sets(thread, *region_p);
  taskgraph->num_mutexes = num_mutexes;
}

static const char *
__kmp_taskgraph_region_type_name(kmp_taskgraph_region_type type) {
  switch (type) {
  case TASKGRAPH_REGION_ENTRY:
    return "entry";
  case TASKGRAPH_REGION_EXIT:
    return "exit";
  case TASKGRAPH_REGION_NODE:
    return "node";
  case TASKGRAPH_REGION_WAIT:
    return "wait";
  case TASKGRAPH_REGION_PARALLEL:
    return "parallel";
  case TASKGRAPH_REGION_EXCLUSIVE:
    return "exclusive";
  case TASKGRAPH_REGION_SEQUENTIAL:
    return "sequential";
  case TASKGRAPH_REGION_IRREDUCIBLE:
    return "irreducible";
  default:
    return "<unknown>";
  }
}

#if defined(KMP_DEBUG) || defined(DEBUG_TASKGRAPH)
static void __kmp_dump_taskgraph_regions(FILE *f,
                                         kmp_taskgraph_region_t *region,
                                         int indent = 0) {
  switch (region->type) {
  case TASKGRAPH_REGION_ENTRY:
  case TASKGRAPH_REGION_EXIT:
    fprintf(f, "%*s%s node\n", indent, "",
            __kmp_taskgraph_region_type_name(region->type));
    break;
  case TASKGRAPH_REGION_NODE:
  case TASKGRAPH_REGION_WAIT: {
    char set_membership[40];
    if (region->mutexset)
      sprintf(set_membership, " [sets: 0x%llx]",
              (unsigned long long)region->mutexset->bits[0]);
    else
      strcpy(set_membership, "");
    if (region->task.node->u.resolved.count > 1)
      fprintf(f, "%*s%s: %p (* %d)%s\n", indent, "",
              __kmp_taskgraph_region_type_name(region->type), region->task.node,
              region->task.node->u.resolved.count, set_membership);
    else
      fprintf(f, "%*s%s: %p%s\n", indent, "",
              __kmp_taskgraph_region_type_name(region->type), region->task.node,
              set_membership);
    break;
  }
  default: {
    char set_membership[40];
    if (region->mutexset)
      sprintf(set_membership, " [sets: 0x%llx]",
              (unsigned long long)region->mutexset->bits[0]);
    else
      strcpy(set_membership, "");
    fprintf(f, "%*s%s%s {\n", indent, "",
            __kmp_taskgraph_region_type_name(region->type), set_membership);
    for (kmp_int32 c = 0; c < region->inner.num_children; c++) {
      __kmp_dump_taskgraph_regions(f, region->inner.children[c], indent + 2);
    }
    fprintf(f, "%*s}\n", indent, "");
  }
  }
}
#endif

#ifdef DEBUG_TASKGRAPH

static kmp_taskgraph_region_dep_t *
__kmp_dump_find_parent_regions(kmp_info *thd, kmp_taskgraph_record_t *taskgraph,
                               kmp_taskgraph_region_t *region, int numregions,
                               kmp_taskgraph_region_dep_t *list = nullptr) {
  for (int r = 0; r < numregions; r++) {
    if (!region[r].parent)
      continue;
    bool in_list = false;
    for (kmp_taskgraph_region_dep_t *dep = list; dep; dep = dep->next) {
      if (dep->region == region[r].parent) {
        in_list = true;
        break;
      }
    }
    if (!in_list) {
      list = __kmp_region_deplist_add(thd, &taskgraph->recycled_deps,
                                      region[r].parent, list);
      list = __kmp_dump_find_parent_regions(thd, taskgraph, region[r].parent, 1,
                                            list);
    }
  }
  return list;
}

static void __kmp_dump_raw_taskgraph_regions(FILE *f, kmp_info *thd,
                                             kmp_taskgraph_record_t *taskgraph,
                                             kmp_taskgraph_region_t *region,
                                             int numregions, int indent = 0) {
  kmp_taskgraph_region_dep_t *parentlist = nullptr;
  kmp_taskgraph_region_dep_t *printedlist = nullptr;
  for (int r = 0; r < numregions; r++) {
    int children = 0;
    if (region[r].type == TASKGRAPH_REGION_PARALLEL ||
        region[r].type == TASKGRAPH_REGION_SEQUENTIAL ||
        region[r].type == TASKGRAPH_REGION_EXCLUSIVE ||
        region[r].type == TASKGRAPH_REGION_IRREDUCIBLE)
      children = region[r].inner.num_children;
    fprintf(
        f,
        "%*sregion %d (%p): %s%s (%d children) parent %p succs %d preds %d\n",
        indent, "", r, &region[r],
        __kmp_taskgraph_region_type_name(region[r].type),
        region[r].mark == TASKGRAPH_COMBINED ? " (combined)" : "", children,
        region[r].parent, __kmp_region_deplist_len(region[r].successors),
        __kmp_region_deplist_len(region[r].predecessors));
    if (children > 0) {
      for (int c = 0; c < children; c++)
        __kmp_dump_raw_taskgraph_regions(
            f, thd, taskgraph, region->inner.children[c], 1, indent + 2);
    }
  }
  if (indent == 0) {
    parentlist =
        __kmp_dump_find_parent_regions(thd, taskgraph, region, numregions);
    fprintf(stderr, "%*sfound %d parent region(s):\n", indent, "",
            __kmp_region_deplist_len(parentlist));
    for (kmp_taskgraph_region_dep_t *p = parentlist; p; p = p->next) {
      __kmp_dump_raw_taskgraph_regions(f, thd, taskgraph, p->region, 1,
                                       indent + 2);
    }
    __kmp_region_deplist_recycle(&taskgraph->recycled_deps, parentlist);
  }
}
#endif

/// Build a nested region structure from a "raw" recorded taskgraph, and mark
/// the taskgraph ready for replay.
//
// The input to this function consists of tasks with *data* dependencies
// between them.  The output of the function is a nested tree structure: the
// dependencies between tasks implicitly become *control* dependencies.  In
// the common case, these ought to map straightforwardly to hardware-provided
// execution primitives (e.g. on a GPU), or to runtime-provided primitives (for
// the CPU).
//
// Here is an example taskgraph:
//
// #pragma omp taskgraph
// {
//   #pragma omp task depend(out: deps[2])
//   { }
//   #pragma omp task depend(out: deps[0], deps[1])
//   { }
//   #pragma omp task depend(inout: deps[0])
//   { }
//   #pragma omp task depend(inout: deps[1])
//   { }
//   #pragma omp task depend(inout: deps[2])
//   { }
//   #pragma omp task depend(in: deps[0], deps[1], deps[2])
//   { }
// }
//
// This dependency graph is "reducible", and the resulting tree looks like this:
//
// sequential {
//   parallel {
//     sequential {
//       node: 0x588aa11021b0
//       node: 0x588aa1102250
//     }
//     sequential {
//       node: 0x588aa11021d8
//       parallel {
//         node: 0x588aa1102228
//         node: 0x588aa1102200
//       }
//     }
//   }
//   node: 0x588aa1102278
// }
//
// Each node represents a task, and the containing parallel and sequential
// regions represent sub-regions that can be executed in parallel, or
// one-at-a-time, in order.
//
// In some cases, the data-dependency graph may not be trivially reducible to
// parallel and sequential regions.  In this case, several techniques are used
// to produce a reducible graph from an irreducible graph (see
// __kmp_taskgraph_rewrite_irreducible).
//
// For example in this graph:
//
// #pragma omp taskgraph
// {
//   #pragma omp task depend(out: deps[0], deps[1])
//   { }
//   #pragma omp task depend(out: deps[2], deps[3])
//   { }
//   #pragma omp task depend(inout: deps[0])
//   { }
//   #pragma omp task depend(inout: deps[1])
//   { }
//   #pragma omp task depend(inout: deps[2])
//   { }
//   #pragma omp task depend(inout: deps[3])
//   { }
//   #pragma omp task depend(in: deps[0], deps[1], deps[2], deps[3])
//   { }
//   #pragma omp task depend(in: deps[1], deps[2])
//   { }
// }
//
// The final two tasks overlap data dependencies in such a way that the
// resulting dependency graph cannot be trivially decomposed to parallel and
// sequential regions.  In this case, the graph is handled by duplicating task
// nodes so they appear in more than one place in the resulting nested region
// structure:
//
// parallel {
//   sequential {
//     parallel {
//       sequential {
//         node: 0x61bfca8ecfd8 (* 2)
//         node: 0x61bfca8ed050 (* 2)
//       }
//       sequential {
//         node: 0x61bfca8ed000 (* 2)
//         node: 0x61bfca8ed078 (* 2)
//       }
//     }
//     node: 0x61bfca8ed0f0
//   }
//   sequential {
//     parallel {
//       sequential {
//         node: 0x61bfca8ed000 (* 2)
//         parallel {
//           node: 0x61bfca8ed0a0
//           node: 0x61bfca8ed078 (* 2)
//         }
//       }
//       sequential {
//         node: 0x61bfca8ecfd8 (* 2)
//         parallel {
//           node: 0x61bfca8ed050 (* 2)
//           node: 0x61bfca8ed028
//         }
//       }
//     }
//     node: 0x61bfca8ed0c8
//   }
// }
//
// The "(* 2)" markers show that the task node appears "instantiated" in that
// number of places in the graph.  Care must be taken at replay time that all
// nodes preceding a multiply-instantiated node execute before the node, and
// that all nodes succeeding each "instantiation point" are executed once the
// task has executed.
//
// The final region type is "exclusive", which arises for "mutexinoutset"
// dependencies that are able to be abstracted away (we can't do this in all
// cases: when we can't, we still use explicit mutexes).
//
// An example of this:
//
// #pragma omp taskgraph
// {
//   #pragma omp task depend(mutexinoutset: deps[0])
//   { }
//   #pragma omp task depend(mutexinoutset: deps[1])
//   { }
//   #pragma omp task depend(mutexinoutset: deps[0])
//   { }
//   #pragma omp task depend(mutexinoutset: deps[1])
//   { }
//   #pragma omp task depend(mutexinoutset: deps[0])
//   { }
//   #pragma omp task depend(mutexinoutset: deps[1])
//   { }
//   #pragma omp task depend(mutexinoutset: deps[0])
//   { }
//   #pragma omp task depend(mutexinoutset: deps[1])
//   { }
// }
//
// Results in this structure:
//
// parallel {
//   exclusive {
//     node: 0x5c0c5c571120
//     node: 0x5c0c5c5710d0
//     node: 0x5c0c5c571080
//     node: 0x5c0c5c571030
//   }
//   exclusive {
//     node: 0x5c0c5c5710f8
//     node: 0x5c0c5c5710a8
//     node: 0x5c0c5c571058
//     node: 0x5c0c5c571008
//   }
// }
//
// The meaning of "exclusive" here is for each of the child regions (task
// nodes in this case) to be executed in some unspecified order, one at a
// time relative to the other regions in the structure.  E.g. a GPU
// implementation could try to dynamically schedule tasks such that they fit
// instantaneously-available execution resources.
//
// In cases where mutexes cannot be abstracted, each affected task node is
// annotated with a set of mutexes that must be held while executing the task.
// (Shown with [sets: 0xN] in dump output).

kmp_int32 __kmp_build_taskgraph(kmp_int32 gtid,
                                kmp_taskdata_t *current_taskdata,
                                kmp_taskgraph_record_t *taskgraph) {
  kmp_int32 numnodes = taskgraph->num_tasks;
  kmp_int32 numregions = numnodes + 2;
  kmp_taskgraph_node_t *nodes = taskgraph->record_map;
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_dephash_t *hash = __kmp_dephash_create(thread, current_taskdata);
  bool dep_barrier = false;

  // We need to take special care to align the all_depnodes array to the cache
  // line size, because kmp_depnode_t is marked as 64-byte aligned and
  // otherwise the compiler might generate faulting memory accesses based on
  // that alignment assumption.
  size_t all_depnodes_size = numregions * sizeof(kmp_depnode_t);
  // The maximum amount of padding we need is CACHE_LINE - 1 bytes.
  all_depnodes_size = all_depnodes_size + CACHE_LINE - 1;
  char *all_depnodes_misaligned =
      (char *)__kmp_thread_malloc(thread, all_depnodes_size);
  kmp_depnode_t *all_depnodes =
      (kmp_depnode_t *)((((intptr_t)all_depnodes_misaligned) + CACHE_LINE - 1) &
                        ~(CACHE_LINE - 1));
  kmp_int32 next_mutex_set = 0;

  for (kmp_int32 i = 0; i < numnodes; i++) {
    int n_mtxs = 0;
    bool dep_all;

    dep_all = __kmp_filter_aliased_deps(nodes[i].u.unresolved.ndeps,
                                        nodes[i].u.unresolved.dep_list,
                                        nodes[i].task, &n_mtxs);
    kmp_depnode_t *node = &all_depnodes[i];
    __kmp_init_node(node, /*on_stack=*/false);
    node->dn.task = nodes[i].task;
    dep_barrier = !nodes[i].task && nodes[i].taskloop_task;
    if (!dep_all) {
      __kmp_process_deps<taskgraph_deps>(
          gtid, node, &hash, dep_barrier, nodes[i].u.unresolved.ndeps,
          nodes[i].u.unresolved.dep_list, nodes[i].task, next_mutex_set);
    } else {
      __kmp_process_dep_all<taskgraph_deps>(gtid, node, hash, dep_barrier,
                                            nodes[i].task);
    }
  }

  kmp_taskgraph_region_t *order_out[numregions];
  kmp_int32 outidx = 0;

  kmp_taskgraph_region_t *initial_regions =
      (kmp_taskgraph_region_t *)__kmp_fast_allocate(
          thread, sizeof(kmp_taskgraph_region_t) * numregions);
  // FIXME: Something like 'placement new' here?
  memset(initial_regions, 0, sizeof(kmp_taskgraph_region_t) * numregions);

  kmp_taskgraph_region_t *cfg_barrier = nullptr;

  for (kmp_int32 i = 0; i < numnodes; i++) {
    initial_regions[i].type =
        nodes[i].task ? TASKGRAPH_REGION_NODE : TASKGRAPH_REGION_WAIT;
    initial_regions[i].task.node = &nodes[i];
    initial_regions[i].task.next_instance = &initial_regions[i];
    initial_regions[i].parent = nullptr;
    if (i < numnodes - 1) {
      initial_regions[i].next = &initial_regions[i + 1];
    } else {
      initial_regions[i].next = nullptr;
    }
    kmp_depnode_t *depnode = &all_depnodes[i];
    initial_regions[i].mutexset = depnode->dn.set_membership;
    for (kmp_depnode_list_t *succ = depnode->dn.successors; succ;
         succ = succ->next) {
      kmp_int32 succ_idx = succ->node - all_depnodes;
      kmp_taskgraph_region_t *tg_succ = &initial_regions[succ_idx];
      tg_succ->predecessors =
          __kmp_region_deplist_add(thread, &taskgraph->recycled_deps,
                                   &initial_regions[i], tg_succ->predecessors);
      initial_regions[i].successors =
          __kmp_region_deplist_add(thread, &taskgraph->recycled_deps, tg_succ,
                                   initial_regions[i].successors);
    }
    // Handle control flow dependencies.  If a node (e.g. a taskloop task) has
    // a wait after it corresponding to the end of an implicit taskgroup, join
    // the task to the wait.  The wait then becomes a barrier; any tasks after
    // it will depend on the barrier.
    if (nodes[i].u.unresolved.cfg_successor != -1) {
      kmp_int32 cfg_succ = nodes[i].u.unresolved.cfg_successor;
      initial_regions[i].successors = __kmp_region_deplist_add(
          thread, &taskgraph->recycled_deps, &initial_regions[cfg_succ],
          initial_regions[i].successors);
      initial_regions[cfg_succ].predecessors = __kmp_region_deplist_add(
          thread, &taskgraph->recycled_deps, &initial_regions[i],
          initial_regions[cfg_succ].predecessors);
    }
    if (nodes[i].taskloop_task && !nodes[i].task) {
      cfg_barrier = &initial_regions[i];
    } else if (cfg_barrier) {
      cfg_barrier->successors = __kmp_region_deplist_add(
          thread, &taskgraph->recycled_deps, &initial_regions[i],
          cfg_barrier->successors);
      initial_regions[i].predecessors = __kmp_region_deplist_add(
          thread, &taskgraph->recycled_deps, cfg_barrier,
          initial_regions[i].predecessors);
    }
  }

  __kmp_dephash_free<false>(thread, hash);
  __kmp_thread_free(thread, all_depnodes_misaligned);

  // We're done with the "unresolved" data now.  Initialise node count.
  for (kmp_int32 i = 0; i < numnodes; i++) {
    __kmp_thread_free(thread, nodes[i].u.unresolved.dep_list);
    nodes[i].u.resolved.last_region = nullptr;
    nodes[i].u.resolved.count = 0;
  }

  // Use these indices for the virtual entry and exit regions
  kmp_int32 entryregion = numnodes, exitregion = numnodes + 1;

  // Set entry/exit node types, and add to worklist
  initial_regions[entryregion].type = TASKGRAPH_REGION_ENTRY;
  initial_regions[entryregion].next = &initial_regions[0];
  initial_regions[exitregion].type = TASKGRAPH_REGION_EXIT;
  initial_regions[numnodes - 1].next = &initial_regions[exitregion];

  // Join entry and exit nodes up to the graph
  for (kmp_int32 i = 0; i < numnodes; i++) {
    kmp_taskgraph_region_t *region = &initial_regions[i];
    kmp_int32 npreds = __kmp_region_deplist_len(region->predecessors);
    kmp_int32 nsuccs = __kmp_region_deplist_len(region->successors);
    if (npreds == 0) {
      initial_regions[entryregion].successors =
          __kmp_region_deplist_add(thread, &taskgraph->recycled_deps, region,
                                   initial_regions[entryregion].successors);
      region->predecessors = __kmp_region_deplist_add(
          thread, &taskgraph->recycled_deps, &initial_regions[entryregion],
          region->predecessors);
    }
    if (nsuccs == 0) {
      initial_regions[exitregion].predecessors =
          __kmp_region_deplist_add(thread, &taskgraph->recycled_deps, region,
                                   initial_regions[exitregion].predecessors);
      region->successors = __kmp_region_deplist_add(
          thread, &taskgraph->recycled_deps, &initial_regions[exitregion],
          region->successors);
    }
    region->owner = taskgraph;
  }

  kmp_int32 max_level = -1;

  for (kmp_int32 i = 0; i < numregions; i++)
    initial_regions[i].timestamp = i;

  for (kmp_int32 i = 0; i < numregions; i++) {
    if (initial_regions[i].mark == TASKGRAPH_UNMARKED) {
      kmp_int32 level = __kmp_taskgraph_topological_order(&initial_regions[i],
                                                          order_out, &outidx);
      max_level = level > max_level ? level : max_level;
    }
  }

  assert(outidx == numregions);

#ifdef DEBUG_TASKGRAPH
  fprintf(stderr, "topological order (max level: %d):\n", max_level);

  for (kmp_int32 i = 0; i < outidx; i++) {
    fprintf(stderr, "node %d (region %p), level %d\n", order_out[i]->timestamp,
            order_out[i], order_out[i]->level);
  }
#endif

  kmp_taskgraph_region_t **alloc_chain = &initial_regions[0].alloc_chain;

  kmp_taskgraph_region_t *root_region = __kmp_taskgraph_build_regions(
      thread, taskgraph, alloc_chain, &initial_regions[entryregion],
      &initial_regions[exitregion]);

  __kmp_taskgraph_count_nodes(root_region);

  __kmp_taskgraph_exclusive_regions(thread, taskgraph, alloc_chain,
                                    &root_region, next_mutex_set);

  *alloc_chain = nullptr;

  taskgraph->root = root_region;
  taskgraph->alloc_root = initial_regions;

  // Free dependency lists and deleted regions.
  kmp_taskgraph_region_t **regp = &taskgraph->alloc_root;
  while (*regp) {
    kmp_taskgraph_region_t *reg = *regp;
    __kmp_region_deplist_free(thread, reg->predecessors);
    __kmp_region_deplist_free(thread, reg->successors);
    reg->predecessors = nullptr;
    reg->successors = nullptr;
    if (reg->mark == TASKGRAPH_DELETED) {
      kmp_taskgraph_region_t *chain_next = reg->alloc_chain;
      TGDBG("deleted region from alloc chain: %p\n", reg);
      __kmp_fast_free(thread, reg);
      *regp = chain_next;
    } else {
      regp = &reg->alloc_chain;
    }
  }
  // Free recycled dep list.  We could pass this along to the next invocation
  // of this function instead, but we don't do that yet (ownership/thread
  // safety needs careful consideration if we do that).
  for (kmp_taskgraph_region_dep_t *dep = taskgraph->recycled_deps; dep;) {
    kmp_taskgraph_region_dep_t *next = dep->next;
    TGDBG("free dep from recycled list\n");
    __kmp_fast_free(thread, dep);
    dep = next;
  }
  taskgraph->recycled_deps = nullptr;

  KG_TRACE(10, ("Processed taskgraph %p (graph_id %" PRIx64 "):\n", taskgraph,
                taskgraph->graph_id));
  KG_DUMP(10, __kmp_dump_taskgraph_regions(stderr, root_region));

#ifdef DEBUG_TASKGRAPH
//__kmp_dump_taskgraph_regions(stderr, root_region);
//__kmp_dump_raw_taskgraph_regions(stderr, thread, taskgraph,
//                                 &initial_regions[0], numregions);
#endif

  KMP_ATOMIC_ST_REL(&taskgraph->status, KMP_TDG_READY);

  return 0;
}
#endif

#define NO_DEP_BARRIER (false)
#define DEP_BARRIER (true)

// returns true if the task has any outstanding dependence
static bool __kmp_check_deps(kmp_int32 gtid, kmp_depnode_t *node,
                             kmp_task_t *task, kmp_dephash_t **hash,
                             bool dep_barrier, kmp_int32 ndeps,
                             kmp_depend_info_t *dep_list,
                             kmp_int32 ndeps_noalias,
                             kmp_depend_info_t *noalias_dep_list) {
  int n_mtxs = 0, dep_all = 0;
#if KMP_DEBUG
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
#endif
  KA_TRACE(20, ("__kmp_check_deps: T#%d checking dependences for task %p : %d "
                "possibly aliased dependences, %d non-aliased dependences : "
                "dep_barrier=%d .\n",
                gtid, taskdata, ndeps, ndeps_noalias, dep_barrier));

  dep_all = __kmp_filter_aliased_deps(ndeps, dep_list, task, &n_mtxs);

  // doesn't need to be atomic as no other thread is going to be accessing this
  // node just yet.
  // npredecessors is set -1 to ensure that none of the releasing tasks queues
  // this task before we have finished processing all the dependences
  node->dn.npredecessors = -1;

  // used to pack all npredecessors additions into a single atomic operation at
  // the end
  int npredecessors;
  kmp_int32 next_mutex = 0;

  if (!dep_all) { // regular dependences
    npredecessors = __kmp_process_deps<normal_deps>(
        gtid, node, hash, dep_barrier, ndeps, dep_list, task, next_mutex);
    npredecessors += __kmp_process_deps<normal_deps>(
        gtid, node, hash, dep_barrier, ndeps_noalias, noalias_dep_list, task,
        next_mutex, false);
  } else { // omp_all_memory dependence
    npredecessors = __kmp_process_dep_all<normal_deps>(gtid, node, *hash,
                                                       dep_barrier, task);
  }

  node->dn.task = task;
  KMP_MB();

  // Account for our initial fake value
  npredecessors++;

  // Update predecessors and obtain current value to check if there are still
  // any outstanding dependences (some tasks may have finished while we
  // processed the dependences)
  npredecessors =
      node->dn.npredecessors.fetch_add(npredecessors) + npredecessors;

  KA_TRACE(20, ("__kmp_check_deps: T#%d found %d predecessors for task %p \n",
                gtid, npredecessors, taskdata));

  // beyond this point the task could be queued (and executed) by a releasing
  // task...
  return npredecessors > 0 ? true : false;
}

/*!
@ingroup TASKING
@param loc_ref location of the original task directive
@param gtid Global Thread ID of encountering thread
@param new_task task thunk allocated by __kmp_omp_task_alloc() for the ''new
task''
@param ndeps Number of depend items with possible aliasing
@param dep_list List of depend items with possible aliasing
@param ndeps_noalias Number of depend items with no aliasing
@param noalias_dep_list List of depend items with no aliasing

@return Returns either TASK_CURRENT_NOT_QUEUED if the current task was not
suspended and queued, or TASK_CURRENT_QUEUED if it was suspended and queued

Schedule a non-thread-switchable task with dependences for execution
*/
kmp_int32 __kmpc_omp_task_with_deps(ident_t *loc_ref, kmp_int32 gtid,
                                    kmp_task_t *new_task, kmp_int32 ndeps,
                                    kmp_depend_info_t *dep_list,
                                    kmp_int32 ndeps_noalias,
                                    kmp_depend_info_t *noalias_dep_list) {

  kmp_taskdata_t *new_taskdata = KMP_TASK_TO_TASKDATA(new_task);
  KA_TRACE(10, ("__kmpc_omp_task_with_deps(enter): T#%d loc=%p task=%p\n", gtid,
                loc_ref, new_taskdata));
  __kmp_assert_valid_gtid(gtid);
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *current_task = thread->th.th_current_task;

#if OMPT_SUPPORT
  if (ompt_enabled.enabled) {
    if (!current_task->ompt_task_info.frame.enter_frame.ptr)
      current_task->ompt_task_info.frame.enter_frame.ptr =
          OMPT_GET_FRAME_ADDRESS(0);
    if (ompt_enabled.ompt_callback_task_create) {
      ompt_callbacks.ompt_callback(ompt_callback_task_create)(
          &(current_task->ompt_task_info.task_data),
          &(current_task->ompt_task_info.frame),
          &(new_taskdata->ompt_task_info.task_data),
          TASK_TYPE_DETAILS_FORMAT(new_taskdata), 1,
          OMPT_LOAD_OR_GET_RETURN_ADDRESS(gtid));
    }

    new_taskdata->ompt_task_info.frame.enter_frame.ptr =
        OMPT_GET_FRAME_ADDRESS(0);
  }

#if OMPT_OPTIONAL
  /* OMPT grab all dependences if requested by the tool */
  if (ndeps + ndeps_noalias > 0 && ompt_enabled.ompt_callback_dependences) {
    kmp_int32 i;

    int ompt_ndeps = ndeps + ndeps_noalias;
    ompt_dependence_t *ompt_deps = (ompt_dependence_t *)KMP_OMPT_DEPS_ALLOC(
        thread, (ndeps + ndeps_noalias) * sizeof(ompt_dependence_t));

    KMP_ASSERT(ompt_deps != NULL);

    for (i = 0; i < ndeps; i++) {
      ompt_deps[i].variable.ptr = (void *)dep_list[i].base_addr;
      if (dep_list[i].base_addr == (kmp_intptr_t)KMP_SIZE_T_MAX)
        ompt_deps[i].dependence_type = ompt_dependence_type_out_all_memory;
      else if (dep_list[i].flags.in && dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_inout;
      else if (dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_out;
      else if (dep_list[i].flags.in)
        ompt_deps[i].dependence_type = ompt_dependence_type_in;
      else if (dep_list[i].flags.mtx)
        ompt_deps[i].dependence_type = ompt_dependence_type_mutexinoutset;
      else if (dep_list[i].flags.set)
        ompt_deps[i].dependence_type = ompt_dependence_type_inoutset;
      else if (dep_list[i].flags.all)
        ompt_deps[i].dependence_type = ompt_dependence_type_out_all_memory;
    }
    for (i = 0; i < ndeps_noalias; i++) {
      ompt_deps[ndeps + i].variable.ptr = (void *)noalias_dep_list[i].base_addr;
      if (noalias_dep_list[i].base_addr == (kmp_intptr_t)KMP_SIZE_T_MAX)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_out_all_memory;
      else if (noalias_dep_list[i].flags.in && noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inout;
      else if (noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_out;
      else if (noalias_dep_list[i].flags.in)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_in;
      else if (noalias_dep_list[i].flags.mtx)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_mutexinoutset;
      else if (noalias_dep_list[i].flags.set)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inoutset;
      else if (noalias_dep_list[i].flags.all)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_out_all_memory;
    }
    ompt_callbacks.ompt_callback(ompt_callback_dependences)(
        &(new_taskdata->ompt_task_info.task_data), ompt_deps, ompt_ndeps);
    /* We can now free the allocated memory for the dependences */
    /* For OMPD we might want to delay the free until end of this function */
    KMP_OMPT_DEPS_FREE(thread, ompt_deps);
  }
#endif /* OMPT_OPTIONAL */
#endif /* OMPT_SUPPORT */

  bool serial = current_task->td_flags.team_serial ||
                current_task->td_flags.tasking_ser ||
                current_task->td_flags.final;
  kmp_task_team_t *task_team = thread->th.th_task_team;
  serial = serial &&
           !(task_team && (task_team->tt.tt_found_proxy_tasks ||
                           task_team->tt.tt_hidden_helper_task_encountered));

  if (!serial && (ndeps > 0 || ndeps_noalias > 0)) {
    /* if no dependences have been tracked yet, create the dependence hash */
    if (current_task->td_dephash == NULL)
      current_task->td_dephash = __kmp_dephash_create(thread, current_task);

#if USE_FAST_MEMORY
    kmp_depnode_t *node =
        (kmp_depnode_t *)__kmp_fast_allocate(thread, sizeof(kmp_depnode_t));
#else
    kmp_depnode_t *node =
        (kmp_depnode_t *)__kmp_thread_malloc(thread, sizeof(kmp_depnode_t));
#endif

    __kmp_init_node(node, /*on_stack=*/false);
    new_taskdata->td_depnode = node;

    if (__kmp_check_deps(gtid, node, new_task, &current_task->td_dephash,
                         NO_DEP_BARRIER, ndeps, dep_list, ndeps_noalias,
                         noalias_dep_list)) {
      KA_TRACE(10, ("__kmpc_omp_task_with_deps(exit): T#%d task had blocking "
                    "dependences: "
                    "loc=%p task=%p, return: TASK_CURRENT_NOT_QUEUED\n",
                    gtid, loc_ref, new_taskdata));
#if OMPT_SUPPORT
      if (ompt_enabled.enabled) {
        current_task->ompt_task_info.frame.enter_frame = ompt_data_none;
      }
#endif
      return TASK_CURRENT_NOT_QUEUED;
    }
  } else {
    KA_TRACE(10, ("__kmpc_omp_task_with_deps(exit): T#%d ignored dependences "
                  "for task (serialized) loc=%p task=%p\n",
                  gtid, loc_ref, new_taskdata));
  }

  KA_TRACE(10, ("__kmpc_omp_task_with_deps(exit): T#%d task had no blocking "
                "dependences : "
                "loc=%p task=%p, transferring to __kmp_omp_task\n",
                gtid, loc_ref, new_taskdata));

  kmp_int32 ret = __kmp_omp_task(gtid, new_task, true);
#if OMPT_SUPPORT
  if (ompt_enabled.enabled) {
    current_task->ompt_task_info.frame.enter_frame = ompt_data_none;
  }
#endif
  return ret;
}

#if OMPT_SUPPORT
void __ompt_taskwait_dep_finish(kmp_taskdata_t *current_task,
                                ompt_data_t *taskwait_task_data) {
  if (ompt_enabled.ompt_callback_task_schedule) {
    ompt_callbacks.ompt_callback(ompt_callback_task_schedule)(
        taskwait_task_data, ompt_taskwait_complete, NULL);
  }
  current_task->ompt_task_info.frame.enter_frame.ptr = NULL;
  *taskwait_task_data = ompt_data_none;
}
#endif /* OMPT_SUPPORT */

/*!
@ingroup TASKING
@param loc_ref location of the original task directive
@param gtid Global Thread ID of encountering thread
@param ndeps Number of depend items with possible aliasing
@param dep_list List of depend items with possible aliasing
@param ndeps_noalias Number of depend items with no aliasing
@param noalias_dep_list List of depend items with no aliasing

Blocks the current task until all specifies dependences have been fulfilled.
*/
void __kmpc_omp_wait_deps(ident_t *loc_ref, kmp_int32 gtid, kmp_int32 ndeps,
                          kmp_depend_info_t *dep_list, kmp_int32 ndeps_noalias,
                          kmp_depend_info_t *noalias_dep_list) {
  __kmpc_omp_taskwait_deps_51(loc_ref, gtid, ndeps, dep_list, ndeps_noalias,
                              noalias_dep_list, false);
}

/* __kmpc_omp_taskwait_deps_51 : Function for OpenMP 5.1 nowait clause.
                                 Placeholder for taskwait with nowait clause.
                                 Earlier code of __kmpc_omp_wait_deps() is now
                                 in this function.
*/
void __kmpc_omp_taskwait_deps_51(ident_t *loc_ref, kmp_int32 gtid,
                                 kmp_int32 ndeps, kmp_depend_info_t *dep_list,
                                 kmp_int32 ndeps_noalias,
                                 kmp_depend_info_t *noalias_dep_list,
                                 kmp_int32 has_no_wait) {
  KA_TRACE(10, ("__kmpc_omp_taskwait_deps(enter): T#%d loc=%p nowait#%d\n",
                gtid, loc_ref, has_no_wait));
  if (ndeps == 0 && ndeps_noalias == 0) {
    KA_TRACE(10, ("__kmpc_omp_taskwait_deps(exit): T#%d has no dependences to "
                  "wait upon : loc=%p\n",
                  gtid, loc_ref));
    return;
  }
  __kmp_assert_valid_gtid(gtid);
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *current_task = thread->th.th_current_task;

#if OMPT_SUPPORT
  // this function represents a taskwait construct with depend clause
  // We signal 4 events:
  //  - creation of the taskwait task
  //  - dependences of the taskwait task
  //  - schedule and finish of the taskwait task
  ompt_data_t *taskwait_task_data = &thread->th.ompt_thread_info.task_data;
  KMP_ASSERT(taskwait_task_data->ptr == NULL);
  if (ompt_enabled.enabled) {
    if (!current_task->ompt_task_info.frame.enter_frame.ptr)
      current_task->ompt_task_info.frame.enter_frame.ptr =
          OMPT_GET_FRAME_ADDRESS(0);
    if (ompt_enabled.ompt_callback_task_create) {
      ompt_callbacks.ompt_callback(ompt_callback_task_create)(
          &(current_task->ompt_task_info.task_data),
          &(current_task->ompt_task_info.frame), taskwait_task_data,
          ompt_task_taskwait | ompt_task_undeferred | ompt_task_mergeable, 1,
          OMPT_LOAD_OR_GET_RETURN_ADDRESS(gtid));
    }
  }

#if OMPT_OPTIONAL
  /* OMPT grab all dependences if requested by the tool */
  if (ndeps + ndeps_noalias > 0 && ompt_enabled.ompt_callback_dependences) {
    kmp_int32 i;

    int ompt_ndeps = ndeps + ndeps_noalias;
    ompt_dependence_t *ompt_deps = (ompt_dependence_t *)KMP_OMPT_DEPS_ALLOC(
        thread, (ndeps + ndeps_noalias) * sizeof(ompt_dependence_t));

    KMP_ASSERT(ompt_deps != NULL);

    for (i = 0; i < ndeps; i++) {
      ompt_deps[i].variable.ptr = (void *)dep_list[i].base_addr;
      if (dep_list[i].flags.in && dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_inout;
      else if (dep_list[i].flags.out)
        ompt_deps[i].dependence_type = ompt_dependence_type_out;
      else if (dep_list[i].flags.in)
        ompt_deps[i].dependence_type = ompt_dependence_type_in;
      else if (dep_list[i].flags.mtx)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_mutexinoutset;
      else if (dep_list[i].flags.set)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inoutset;
    }
    for (i = 0; i < ndeps_noalias; i++) {
      ompt_deps[ndeps + i].variable.ptr = (void *)noalias_dep_list[i].base_addr;
      if (noalias_dep_list[i].flags.in && noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inout;
      else if (noalias_dep_list[i].flags.out)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_out;
      else if (noalias_dep_list[i].flags.in)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_in;
      else if (noalias_dep_list[i].flags.mtx)
        ompt_deps[ndeps + i].dependence_type =
            ompt_dependence_type_mutexinoutset;
      else if (noalias_dep_list[i].flags.set)
        ompt_deps[ndeps + i].dependence_type = ompt_dependence_type_inoutset;
    }
    ompt_callbacks.ompt_callback(ompt_callback_dependences)(
        taskwait_task_data, ompt_deps, ompt_ndeps);
    /* We can now free the allocated memory for the dependences */
    /* For OMPD we might want to delay the free until end of this function */
    KMP_OMPT_DEPS_FREE(thread, ompt_deps);
    ompt_deps = NULL;
  }
#endif /* OMPT_OPTIONAL */
#endif /* OMPT_SUPPORT */

  // We can return immediately as:
  // - dependences are not computed in serial teams (except with proxy tasks)
  // - if the dephash is not yet created it means we have nothing to wait for
  bool ignore = current_task->td_flags.team_serial ||
                current_task->td_flags.tasking_ser ||
                current_task->td_flags.final;
  ignore =
      ignore && thread->th.th_task_team != NULL &&
      thread->th.th_task_team->tt.tt_found_proxy_tasks == FALSE &&
      thread->th.th_task_team->tt.tt_hidden_helper_task_encountered == FALSE;
  ignore = ignore || current_task->td_dephash == NULL;

  if (ignore) {
    KA_TRACE(10, ("__kmpc_omp_taskwait_deps(exit): T#%d has no blocking "
                  "dependences : loc=%p\n",
                  gtid, loc_ref));
#if OMPT_SUPPORT
    __ompt_taskwait_dep_finish(current_task, taskwait_task_data);
#endif /* OMPT_SUPPORT */
    return;
  }

  kmp_depnode_t node = {0};
  __kmp_init_node(&node, /*on_stack=*/true);

  if (!__kmp_check_deps(gtid, &node, NULL, &current_task->td_dephash,
                        DEP_BARRIER, ndeps, dep_list, ndeps_noalias,
                        noalias_dep_list)) {
    KA_TRACE(10, ("__kmpc_omp_taskwait_deps(exit): T#%d has no blocking "
                  "dependences : loc=%p\n",
                  gtid, loc_ref));
#if OMPT_SUPPORT
    __ompt_taskwait_dep_finish(current_task, taskwait_task_data);
#endif /* OMPT_SUPPORT */

    // There may still be references to this node here, due to task stealing.
    // Wait for them to be released.
    kmp_int32 nrefs;
    while ((nrefs = node.dn.nrefs) > 3) {
      KMP_DEBUG_ASSERT((nrefs & 1) == 1);
      KMP_YIELD(TRUE);
    }
    KMP_DEBUG_ASSERT(nrefs == 3);

    return;
  }

  int thread_finished = FALSE;
  kmp_flag_32<false, false> flag(
      (std::atomic<kmp_uint32> *)&node.dn.npredecessors, 0U);
  while (node.dn.npredecessors > 0) {
    flag.execute_tasks(thread, gtid, FALSE,
                       &thread_finished USE_ITT_BUILD_ARG(NULL),
                       __kmp_task_stealing_constraint);
  }

  // Wait until the last __kmp_release_deps is finished before we free the
  // current stack frame holding the "node" variable; once its nrefs count
  // reaches 3 (meaning 1, since bit zero of the refcount indicates a stack
  // rather than a heap address), we're sure nobody else can try to reference
  // it again.
  kmp_int32 nrefs;
  while ((nrefs = node.dn.nrefs) > 3) {
    KMP_DEBUG_ASSERT((nrefs & 1) == 1);
    KMP_YIELD(TRUE);
  }
  KMP_DEBUG_ASSERT(nrefs == 3);

#if OMPT_SUPPORT
  __ompt_taskwait_dep_finish(current_task, taskwait_task_data);
#endif /* OMPT_SUPPORT */
  KA_TRACE(10, ("__kmpc_omp_taskwait_deps(exit): T#%d finished waiting : loc=%p\
                \n",
                gtid, loc_ref));
}
