/*
 * kmp_taskdeps.cpp
 */

//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

//#define KMP_SUPPORT_GRAPH_OUTPUT 1

#include "kmp.h"
#include "kmp_io.h"
#include "kmp_wait_release.h"
#if OMPT_SUPPORT
#include "ompt-specific.h"
#endif

#if OMP_40_ENABLED

// TODO: Improve memory allocation? keep a list of pre-allocated structures?
// allocate in blocks? re-use list finished list entries?
// TODO: don't use atomic ref counters for stack-allocated nodes.
// TODO: find an alternate to atomic refs for heap-allocated nodes?
// TODO: Finish graph output support
// TODO: kmp_lock_t seems a tad to big (and heavy weight) for this. Check other
// runtime locks
// TODO: Any ITT support needed?

#ifdef KMP_SUPPORT_GRAPH_OUTPUT
static kmp_int32 kmp_node_id_seed = 0;
#endif

static void __kmp_init_node(kmp_depnode_t *node) {
  node->dn.task = NULL; // set to null initially, it will point to the right
  // task once dependences have been processed
  node->dn.successors = NULL;
  __kmp_init_lock(&node->dn.lock);
  node->dn.nrefs = 1; // init creates the first reference to the node
#ifdef KMP_SUPPORT_GRAPH_OUTPUT
  node->dn.id = KMP_TEST_THEN_INC32(&kmp_node_id_seed);
#endif
}

static inline kmp_depnode_t *__kmp_node_ref(kmp_depnode_t *node) {
  KMP_TEST_THEN_INC32(CCAST(kmp_int32 *, &node->dn.nrefs));
  return node;
}

static inline void __kmp_node_deref(kmp_info_t *thread, kmp_depnode_t *node) {
  if (!node)
    return;

  kmp_int32 n = KMP_TEST_THEN_DEC32(CCAST(kmp_int32 *, &node->dn.nrefs)) - 1;
  if (n == 0) {
    KMP_ASSERT(node->dn.nrefs == 0);
#if USE_FAST_MEMORY
    __kmp_fast_free(thread, node);
#else
    __kmp_thread_free(thread, node);
#endif
  }
}

#define KMP_ACQUIRE_DEPNODE(gtid, n) __kmp_acquire_lock(&(n)->dn.lock, (gtid))
#define KMP_RELEASE_DEPNODE(gtid, n) __kmp_release_lock(&(n)->dn.lock, (gtid))

static void __kmp_depnode_list_free(kmp_info_t *thread, kmp_depnode_list *list);

enum { KMP_DEPHASH_OTHER_SIZE = 97, KMP_DEPHASH_MASTER_SIZE = 997 };

static inline kmp_int32 __kmp_dephash_hash(kmp_intptr_t addr, size_t hsize) {
  // TODO alternate to try: set = (((Addr64)(addrUsefulBits * 9.618)) %
  // m_num_sets );
  return ((addr >> 6) ^ (addr >> 2)) % hsize;
}

static kmp_dephash_t *__kmp_dephash_create(kmp_info_t *thread,
                                           kmp_taskdata_t *current_task) {
  kmp_dephash_t *h;

  size_t h_size;

  if (current_task->td_flags.tasktype == TASK_IMPLICIT)
    h_size = KMP_DEPHASH_MASTER_SIZE;
  else
    h_size = KMP_DEPHASH_OTHER_SIZE;

  kmp_int32 size =
      h_size * sizeof(kmp_dephash_entry_t *) + sizeof(kmp_dephash_t);

#if USE_FAST_MEMORY
  h = (kmp_dephash_t *)__kmp_fast_allocate(thread, size);
#else
  h = (kmp_dephash_t *)__kmp_thread_malloc(thread, size);
#endif
  h->size = h_size;

#ifdef KMP_DEBUG
  h->nelements = 0;
  h->nconflicts = 0;
#endif
  h->buckets = (kmp_dephash_entry **)(h + 1);

  for (size_t i = 0; i < h_size; i++)
    h->buckets[i] = 0;

  return h;
}

void __kmp_dephash_free_entries(kmp_info_t *thread, kmp_dephash_t *h) {
  for (size_t i = 0; i < h->size; i++) {
    if (h->buckets[i]) {
      kmp_dephash_entry_t *next;
      for (kmp_dephash_entry_t *entry = h->buckets[i]; entry; entry = next) {
        next = entry->next_in_bucket;
        __kmp_depnode_list_free(thread, entry->last_ins);
        __kmp_node_deref(thread, entry->last_out);
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

void __kmp_dephash_free(kmp_info_t *thread, kmp_dephash_t *h) {
  __kmp_dephash_free_entries(thread, h);
#if USE_FAST_MEMORY
  __kmp_fast_free(thread, h);
#else
  __kmp_thread_free(thread, h);
#endif
}

static kmp_dephash_entry *
__kmp_dephash_find(kmp_info_t *thread, kmp_dephash_t *h, kmp_intptr_t addr) {
  kmp_int32 bucket = __kmp_dephash_hash(addr, h->size);

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
    entry->last_out = NULL;
    entry->last_ins = NULL;
    entry->next_in_bucket = h->buckets[bucket];
    h->buckets[bucket] = entry;
#ifdef KMP_DEBUG
    h->nelements++;
    if (entry->next_in_bucket)
      h->nconflicts++;
#endif
  }
  return entry;
}

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

  new_head->node = __kmp_node_ref(node);
  new_head->next = list;

  return new_head;
}

static void __kmp_depnode_list_free(kmp_info_t *thread,
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

static inline void __kmp_track_dependence(kmp_depnode_t *source,
                                          kmp_depnode_t *sink,
                                          kmp_task_t *sink_task) {
#ifdef KMP_SUPPORT_GRAPH_OUTPUT
  kmp_taskdata_t *task_source = KMP_TASK_TO_TASKDATA(source->dn.task);
  // do not use sink->dn.task as that is only filled after the dependencies
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
    kmp_taskdata_t *task_sink = KMP_TASK_TO_TASKDATA(sink_task);

    ompt_callbacks.ompt_callback(ompt_callback_task_dependence)(
        &(task_source->ompt_task_info.task_data),
        &(task_sink->ompt_task_info.task_data));
  }
#endif /* OMPT_SUPPORT && OMPT_OPTIONAL */
}

template <bool filter>
static inline kmp_int32
__kmp_process_deps(kmp_int32 gtid, kmp_depnode_t *node, kmp_dephash_t *hash,
                   bool dep_barrier, kmp_int32 ndeps,
                   kmp_depend_info_t *dep_list, kmp_task_t *task) {
  KA_TRACE(30, ("__kmp_process_deps<%d>: T#%d processing %d dependencies : "
                "dep_barrier = %d\n",
                filter, gtid, ndeps, dep_barrier));

  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_int32 npredecessors = 0;
  for (kmp_int32 i = 0; i < ndeps; i++) {
    const kmp_depend_info_t *dep = &dep_list[i];

    KMP_DEBUG_ASSERT(dep->flags.in);

    if (filter && dep->base_addr == 0)
      continue; // skip filtered entries

    kmp_dephash_entry_t *info =
        __kmp_dephash_find(thread, hash, dep->base_addr);
    kmp_depnode_t *last_out = info->last_out;

    if (dep->flags.out && info->last_ins) {
      for (kmp_depnode_list_t *p = info->last_ins; p; p = p->next) {
        kmp_depnode_t *indep = p->node;
        if (indep->dn.task) {
          KMP_ACQUIRE_DEPNODE(gtid, indep);
          if (indep->dn.task) {
            __kmp_track_dependence(indep, node, task);
            indep->dn.successors =
                __kmp_add_node(thread, indep->dn.successors, node);
            KA_TRACE(40, ("__kmp_process_deps<%d>: T#%d adding dependence from "
                          "%p to %p\n",
                          filter, gtid, KMP_TASK_TO_TASKDATA(indep->dn.task),
                          KMP_TASK_TO_TASKDATA(task)));
            npredecessors++;
          }
          KMP_RELEASE_DEPNODE(gtid, indep);
        }
      }

      __kmp_depnode_list_free(thread, info->last_ins);
      info->last_ins = NULL;

    } else if (last_out && last_out->dn.task) {
      KMP_ACQUIRE_DEPNODE(gtid, last_out);
      if (last_out->dn.task) {
        __kmp_track_dependence(last_out, node, task);
        last_out->dn.successors =
            __kmp_add_node(thread, last_out->dn.successors, node);
        KA_TRACE(
            40,
            ("__kmp_process_deps<%d>: T#%d adding dependence from %p to %p\n",
             filter, gtid, KMP_TASK_TO_TASKDATA(last_out->dn.task),
             KMP_TASK_TO_TASKDATA(task)));

        npredecessors++;
      }
      KMP_RELEASE_DEPNODE(gtid, last_out);
    }

    if (dep_barrier) {
      // if this is a sync point in the serial sequence, then the previous
      // outputs are guaranteed to be completed after
      // the execution of this task so the previous output nodes can be cleared.
      __kmp_node_deref(thread, last_out);
      info->last_out = NULL;
    } else {
      if (dep->flags.out) {
        __kmp_node_deref(thread, last_out);
        info->last_out = __kmp_node_ref(node);
      } else
        info->last_ins = __kmp_add_node(thread, info->last_ins, node);
    }
  }

  KA_TRACE(30, ("__kmp_process_deps<%d>: T#%d found %d predecessors\n", filter,
                gtid, npredecessors));

  return npredecessors;
}

#define NO_DEP_BARRIER (false)
#define DEP_BARRIER (true)

// returns true if the task has any outstanding dependence
static bool __kmp_check_deps(kmp_int32 gtid, kmp_depnode_t *node,
                             kmp_task_t *task, kmp_dephash_t *hash,
                             bool dep_barrier, kmp_int32 ndeps,
                             kmp_depend_info_t *dep_list,
                             kmp_int32 ndeps_noalias,
                             kmp_depend_info_t *noalias_dep_list) {
  int i;

#if KMP_DEBUG
  kmp_taskdata_t *taskdata = KMP_TASK_TO_TASKDATA(task);
#endif
  KA_TRACE(20, ("__kmp_check_deps: T#%d checking dependencies for task %p : %d "
                "possibly aliased dependencies, %d non-aliased depedencies : "
                "dep_barrier=%d .\n",
                gtid, taskdata, ndeps, ndeps_noalias, dep_barrier));

  // Filter deps in dep_list
  // TODO: Different algorithm for large dep_list ( > 10 ? )
  for (i = 0; i < ndeps; i++) {
    if (dep_list[i].base_addr != 0)
      for (int j = i + 1; j < ndeps; j++)
        if (dep_list[i].base_addr == dep_list[j].base_addr) {
          dep_list[i].flags.in |= dep_list[j].flags.in;
          dep_list[i].flags.out |= dep_list[j].flags.out;
          dep_list[j].base_addr = 0; // Mark j element as void
        }
  }

  // doesn't need to be atomic as no other thread is going to be accessing this
  // node just yet.
  // npredecessors is set -1 to ensure that none of the releasing tasks queues
  // this task before we have finished processing all the dependencies
  node->dn.npredecessors = -1;

  // used to pack all npredecessors additions into a single atomic operation at
  // the end
  int npredecessors;

  npredecessors = __kmp_process_deps<true>(gtid, node, hash, dep_barrier, ndeps,
                                           dep_list, task);
  npredecessors += __kmp_process_deps<false>(
      gtid, node, hash, dep_barrier, ndeps_noalias, noalias_dep_list, task);

  node->dn.task = task;
  KMP_MB();

  // Account for our initial fake value
  npredecessors++;

  // Update predecessors and obtain current value to check if there are still
  // any outstandig dependences (some tasks may have finished while we processed
  // the dependences)
  npredecessors =
      KMP_TEST_THEN_ADD32(CCAST(kmp_int32 *, &node->dn.npredecessors),
                          npredecessors) +
      npredecessors;

  KA_TRACE(20, ("__kmp_check_deps: T#%d found %d predecessors for task %p \n",
                gtid, npredecessors, taskdata));

  // beyond this point the task could be queued (and executed) by a releasing
  // task...
  return npredecessors > 0 ? true : false;
}

void __kmp_release_deps(kmp_int32 gtid, kmp_taskdata_t *task) {
  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_depnode_t *node = task->td_depnode;

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
  for (kmp_depnode_list_t *p = node->dn.successors; p; p = next) {
    kmp_depnode_t *successor = p->node;
    kmp_int32 npredecessors =
        KMP_TEST_THEN_DEC32(CCAST(kmp_int32 *, &successor->dn.npredecessors)) -
        1;
    // successor task can be NULL for wait_depends or because deps are still
    // being processed
    if (npredecessors == 0) {
      KMP_MB();
      if (successor->dn.task) {
        KA_TRACE(20, ("__kmp_release_deps: T#%d successor %p of %p scheduled "
                      "for execution.\n",
                      gtid, successor->dn.task, task));
        __kmp_omp_task(gtid, successor->dn.task, false);
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
suspendend and queued, or TASK_CURRENT_QUEUED if it was suspended and queued

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

  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *current_task = thread->th.th_current_task;

#if OMPT_SUPPORT
  if (ompt_enabled.enabled) {
    OMPT_STORE_RETURN_ADDRESS(gtid);
    if (!current_task->ompt_task_info.frame.enter_frame)
      current_task->ompt_task_info.frame.enter_frame =
          OMPT_GET_FRAME_ADDRESS(1);
    if (ompt_enabled.ompt_callback_task_create) {
      ompt_data_t task_data = ompt_data_none;
      ompt_callbacks.ompt_callback(ompt_callback_task_create)(
          current_task ? &(current_task->ompt_task_info.task_data) : &task_data,
          current_task ? &(current_task->ompt_task_info.frame) : NULL,
          &(new_taskdata->ompt_task_info.task_data),
          ompt_task_explicit | TASK_TYPE_DETAILS_FORMAT(new_taskdata), 1,
          OMPT_LOAD_RETURN_ADDRESS(gtid));
    }

    new_taskdata->ompt_task_info.frame.enter_frame = OMPT_GET_FRAME_ADDRESS(0);
  }

#if OMPT_OPTIONAL
  /* OMPT grab all dependences if requested by the tool */
  if (ndeps + ndeps_noalias > 0 &&
      ompt_enabled.ompt_callback_task_dependences) {
    kmp_int32 i;

    new_taskdata->ompt_task_info.ndeps = ndeps + ndeps_noalias;
    new_taskdata->ompt_task_info.deps =
        (ompt_task_dependence_t *)KMP_OMPT_DEPS_ALLOC(
            thread, (ndeps + ndeps_noalias) * sizeof(ompt_task_dependence_t));

    KMP_ASSERT(new_taskdata->ompt_task_info.deps != NULL);

    for (i = 0; i < ndeps; i++) {
      new_taskdata->ompt_task_info.deps[i].variable_addr =
          (void *)dep_list[i].base_addr;
      if (dep_list[i].flags.in && dep_list[i].flags.out)
        new_taskdata->ompt_task_info.deps[i].dependence_flags =
            ompt_task_dependence_type_inout;
      else if (dep_list[i].flags.out)
        new_taskdata->ompt_task_info.deps[i].dependence_flags =
            ompt_task_dependence_type_out;
      else if (dep_list[i].flags.in)
        new_taskdata->ompt_task_info.deps[i].dependence_flags =
            ompt_task_dependence_type_in;
    }
    for (i = 0; i < ndeps_noalias; i++) {
      new_taskdata->ompt_task_info.deps[ndeps + i].variable_addr =
          (void *)noalias_dep_list[i].base_addr;
      if (noalias_dep_list[i].flags.in && noalias_dep_list[i].flags.out)
        new_taskdata->ompt_task_info.deps[ndeps + i].dependence_flags =
            ompt_task_dependence_type_inout;
      else if (noalias_dep_list[i].flags.out)
        new_taskdata->ompt_task_info.deps[ndeps + i].dependence_flags =
            ompt_task_dependence_type_out;
      else if (noalias_dep_list[i].flags.in)
        new_taskdata->ompt_task_info.deps[ndeps + i].dependence_flags =
            ompt_task_dependence_type_in;
    }
    ompt_callbacks.ompt_callback(ompt_callback_task_dependences)(
        &(new_taskdata->ompt_task_info.task_data),
        new_taskdata->ompt_task_info.deps, new_taskdata->ompt_task_info.ndeps);
    /* We can now free the allocated memory for the dependencies */
    /* For OMPD we might want to delay the free until task_end */
    KMP_OMPT_DEPS_FREE(thread, new_taskdata->ompt_task_info.deps);
    new_taskdata->ompt_task_info.deps = NULL;
    new_taskdata->ompt_task_info.ndeps = 0;
  }
#endif /* OMPT_OPTIONAL */
#endif /* OMPT_SUPPORT */

  bool serial = current_task->td_flags.team_serial ||
                current_task->td_flags.tasking_ser ||
                current_task->td_flags.final;
#if OMP_45_ENABLED
  kmp_task_team_t *task_team = thread->th.th_task_team;
  serial = serial && !(task_team && task_team->tt.tt_found_proxy_tasks);
#endif

  if (!serial && (ndeps > 0 || ndeps_noalias > 0)) {
    /* if no dependencies have been tracked yet, create the dependence hash */
    if (current_task->td_dephash == NULL)
      current_task->td_dephash = __kmp_dephash_create(thread, current_task);

#if USE_FAST_MEMORY
    kmp_depnode_t *node =
        (kmp_depnode_t *)__kmp_fast_allocate(thread, sizeof(kmp_depnode_t));
#else
    kmp_depnode_t *node =
        (kmp_depnode_t *)__kmp_thread_malloc(thread, sizeof(kmp_depnode_t));
#endif

    __kmp_init_node(node);
    new_taskdata->td_depnode = node;

    if (__kmp_check_deps(gtid, node, new_task, current_task->td_dephash,
                         NO_DEP_BARRIER, ndeps, dep_list, ndeps_noalias,
                         noalias_dep_list)) {
      KA_TRACE(10, ("__kmpc_omp_task_with_deps(exit): T#%d task had blocking "
                    "dependencies: "
                    "loc=%p task=%p, return: TASK_CURRENT_NOT_QUEUED\n",
                    gtid, loc_ref, new_taskdata));
#if OMPT_SUPPORT
      if (ompt_enabled.enabled) {
        current_task->ompt_task_info.frame.enter_frame = NULL;
      }
#endif
      return TASK_CURRENT_NOT_QUEUED;
    }
  } else {
    KA_TRACE(10, ("__kmpc_omp_task_with_deps(exit): T#%d ignored dependencies "
                  "for task (serialized)"
                  "loc=%p task=%p\n",
                  gtid, loc_ref, new_taskdata));
  }

  KA_TRACE(10, ("__kmpc_omp_task_with_deps(exit): T#%d task had no blocking "
                "dependencies : "
                "loc=%p task=%p, transferring to __kmpc_omp_task\n",
                gtid, loc_ref, new_taskdata));

  kmp_int32 ret = __kmp_omp_task(gtid, new_task, true);
#if OMPT_SUPPORT
  if (ompt_enabled.enabled) {
    current_task->ompt_task_info.frame.enter_frame = NULL;
  }
#endif
  return ret;
}

/*!
@ingroup TASKING
@param loc_ref location of the original task directive
@param gtid Global Thread ID of encountering thread
@param ndeps Number of depend items with possible aliasing
@param dep_list List of depend items with possible aliasing
@param ndeps_noalias Number of depend items with no aliasing
@param noalias_dep_list List of depend items with no aliasing

Blocks the current task until all specifies dependencies have been fulfilled.
*/
void __kmpc_omp_wait_deps(ident_t *loc_ref, kmp_int32 gtid, kmp_int32 ndeps,
                          kmp_depend_info_t *dep_list, kmp_int32 ndeps_noalias,
                          kmp_depend_info_t *noalias_dep_list) {
  KA_TRACE(10, ("__kmpc_omp_wait_deps(enter): T#%d loc=%p\n", gtid, loc_ref));

  if (ndeps == 0 && ndeps_noalias == 0) {
    KA_TRACE(10, ("__kmpc_omp_wait_deps(exit): T#%d has no dependencies to "
                  "wait upon : loc=%p\n",
                  gtid, loc_ref));
    return;
  }

  kmp_info_t *thread = __kmp_threads[gtid];
  kmp_taskdata_t *current_task = thread->th.th_current_task;

  // We can return immediately as:
  // - dependences are not computed in serial teams (except with proxy tasks)
  // - if the dephash is not yet created it means we have nothing to wait for
  bool ignore = current_task->td_flags.team_serial ||
                current_task->td_flags.tasking_ser ||
                current_task->td_flags.final;
#if OMP_45_ENABLED
  ignore = ignore && thread->th.th_task_team != NULL &&
           thread->th.th_task_team->tt.tt_found_proxy_tasks == FALSE;
#endif
  ignore = ignore || current_task->td_dephash == NULL;

  if (ignore) {
    KA_TRACE(10, ("__kmpc_omp_wait_deps(exit): T#%d has no blocking "
                  "dependencies : loc=%p\n",
                  gtid, loc_ref));
    return;
  }

  kmp_depnode_t node;
  __kmp_init_node(&node);

  if (!__kmp_check_deps(gtid, &node, NULL, current_task->td_dephash,
                        DEP_BARRIER, ndeps, dep_list, ndeps_noalias,
                        noalias_dep_list)) {
    KA_TRACE(10, ("__kmpc_omp_wait_deps(exit): T#%d has no blocking "
                  "dependencies : loc=%p\n",
                  gtid, loc_ref));
    return;
  }

  int thread_finished = FALSE;
  kmp_flag_32 flag((volatile kmp_uint32 *)&(node.dn.npredecessors), 0U);
  while (node.dn.npredecessors > 0) {
    flag.execute_tasks(thread, gtid, FALSE, &thread_finished,
#if USE_ITT_BUILD
                       NULL,
#endif
                       __kmp_task_stealing_constraint);
  }

  KA_TRACE(10, ("__kmpc_omp_wait_deps(exit): T#%d finished waiting : loc=%p\n",
                gtid, loc_ref));
}

#endif /* OMP_40_ENABLED */
