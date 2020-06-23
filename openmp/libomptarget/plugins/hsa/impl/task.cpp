/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/

#include "task.h"
#include "amd_hostcall.h"
#include "data.h"
#include "internal.h"
#include "kernel.h"
#include "machine.h"
#include "queue.h"
#include "realtimer.h"
#include "rt.h"
#include "taskgroup.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <deque>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <malloc.h>
#include <map>
#include <pthread.h>
#include <set>
#include <stdarg.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <vector>
using core::Kernel;
using core::KernelImpl;
using core::RealTimer;
using core::TaskImpl;

pthread_mutex_t mutex_all_tasks_;
pthread_mutex_t mutex_readyq_;
#define NANOSECS 1000000000L

//  set NOTCOHERENT needs this include
#include "hsa_ext_amd.h"

extern bool handle_signal(hsa_signal_value_t value, void *arg);

void print_atl_kernel(const char *str, const int i);

std::vector<TaskImpl *> AllTasks;
std::queue<TaskImpl *> ReadyTaskQueue;
std::queue<hsa_signal_t> FreeSignalPool;

extern bool g_atmi_hostcall_required;

std::map<uint64_t, Kernel *> KernelImplMap;
bool setCallbackToggle = false;

atmi_task_handle_t ATMI_NULL_TASK_HANDLE = 0xFFFFFFFFFFFFFFFF;
atmi_place_t ATMI_DEFAULT_PLACE = {0, ATMI_DEVTYPE_GPU, 0, 0xFFFFFFFFFFFFFFFF};
atmi_mem_place_t ATMI_DEFAULT_MEM_PLACE = {0, ATMI_DEVTYPE_GPU, 0, 0};

extern RealTimer SignalAddTimer;
extern RealTimer HandleSignalTimer;
extern RealTimer HandleSignalInvokeTimer;
extern RealTimer TaskWaitTimer;
extern RealTimer TryLaunchTimer;
extern RealTimer ParamsInitTimer;
extern RealTimer TryLaunchInitTimer;
extern RealTimer ShouldDispatchTimer;
extern RealTimer RegisterCallbackTimer;
extern RealTimer LockTimer;
extern RealTimer TryDispatchTimer;
extern size_t max_ready_queue_sz;
extern size_t waiting_count;
extern size_t direct_dispatch;
extern size_t callback_dispatch;

extern ATLMachine g_atl_machine;

extern hsa_signal_t IdentityORSignal;
extern hsa_signal_t IdentityANDSignal;
extern hsa_signal_t IdentityCopySignal;
extern atl_context_t atlc;

namespace core {
extern void lock_set(const std::set<pthread_mutex_t *> &mutexes);
extern void unlock_set(const std::set<pthread_mutex_t *> &mutexes);
// this file will eventually be refactored into rt.cpp and other module-specific
// files

long int get_nanosecs(struct timespec start_time, struct timespec end_time) {
  long int nanosecs;
  if ((end_time.tv_nsec - start_time.tv_nsec) < 0)
    nanosecs =
        ((((long int)end_time.tv_sec - (long int)start_time.tv_sec) - 1) *
         NANOSECS) +
        (NANOSECS + (long int)end_time.tv_nsec - (long int)start_time.tv_nsec);
  else
    nanosecs =
        (((long int)end_time.tv_sec - (long int)start_time.tv_sec) * NANOSECS) +
        ((long int)end_time.tv_nsec - (long int)start_time.tv_nsec);
  return nanosecs;
}

void packet_store_release(uint32_t *packet, uint16_t header, uint16_t rest) {
  __atomic_store_n(packet, header | (rest << 16), __ATOMIC_RELEASE);
}

uint16_t create_header(hsa_packet_type_t type, int barrier,
                       atmi_task_fence_scope_t acq_fence,
                       atmi_task_fence_scope_t rel_fence) {
  uint16_t header = type << HSA_PACKET_HEADER_TYPE;
  header |= barrier << HSA_PACKET_HEADER_BARRIER;
  header |= (hsa_fence_scope_t) static_cast<int>(
      acq_fence << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE);
  header |= (hsa_fence_scope_t) static_cast<int>(
      rel_fence << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
  // __atomic_store_n((uint8_t*)(&header), (uint8_t)type, __ATOMIC_RELEASE);
  return header;
}

int get_task_handle_ID(atmi_task_handle_t t) {
#if 1
  return static_cast<int>(0xFFFFFFFF & t);
#else
  return t.lo;
#endif
}

void set_task_handle_ID(atmi_task_handle_t *t, int ID) {
#if 1
  unsigned long int task_handle = *t;
  task_handle |= 0xFFFFFFFFFFFFFFFF;
  task_handle &= ID;
  *t = task_handle;
#else
  t->lo = ID;
#endif
}

TaskImpl *getTaskImpl(atmi_task_handle_t t) {
  /* FIXME: node 0 only for now */
  TaskImpl *ret = NULL;
  if (t != ATMI_NULL_TASK_HANDLE) {
    lock(&mutex_all_tasks_);
    ret = AllTasks[get_task_handle_ID(t)];
    unlock(&mutex_all_tasks_);
  }
  return ret;
  // return AllTasks[t.lo];
}

atmi_taskgroup_handle_t get_taskgroup(atmi_task_handle_t t) {
  TaskImpl *task = getTaskImpl(t);
  atmi_taskgroup_handle_t ret;
  if (task) ret = task->taskgroup_;
  return ret;
}

TaskImpl *get_continuation_task(atmi_task_handle_t t) {
  /* FIXME: node 0 only for now */
  // return AllTasks[t.hi];
  return NULL;
}

extern void TaskImpl::wait() {
  TaskWaitTimer.start();
  while (state_ != ATMI_COMPLETED) {
  }

  /* Flag this task as completed */
  /* FIXME: How can HSA tell us if and when a task has failed? */
  set_state(ATMI_COMPLETED);

  TaskWaitTimer.stop();
  return;  // ATMI_STATUS_SUCCESS;
}

extern atmi_status_t Runtime::TaskWait(atmi_task_handle_t task) {
  DEBUG_PRINT("Waiting for task ID: %lu\n", task);
  atmi_status_t status = ATMI_STATUS_ERROR;
  TaskImpl *task_impl = getTaskImpl(task);
  if (task_impl) {
    task_impl->wait();
    status = ATMI_STATUS_SUCCESS;
  }
  return status;
}

void init_dag_scheduler() {
  if (atlc.g_mutex_dag_initialized == 0) {
    pthread_mutex_init(&mutex_all_tasks_, NULL);
    pthread_mutex_init(&mutex_readyq_, NULL);
    AllTasks.clear();
    AllTasks.reserve(500000);
    // PublicTaskMap.clear();
    atlc.g_mutex_dag_initialized = 1;
    DEBUG_PRINT("main tid = %lu\n", syscall(SYS_gettid));
  }
}

#if 0
// FIXME: this implementation may be needed to
// have a common lock/unlock for CPU/GPU
lock(int *mutex) {
    // atomic cas of mutex with 1 and return 0
}

unlock(int *mutex) {
    // atomic exch with 1
}
#endif

void TaskImpl::set_state(const atmi_state_t state) {
  state_ = state;
  // state_.store(state, std::memory_order_seq_cst);
  if (atmi_task_ != NULL) atmi_task_->state = state;
}

void TaskImpl::updateMetrics() {
  hsa_status_t err = HSA_STATUS_SUCCESS;
  // if(profile != NULL) {
  if (profilable_ == ATMI_TRUE) {
    hsa_signal_t signal = signal_;
    hsa_amd_profiling_dispatch_time_t metrics;
    if (devtype_ == ATMI_DEVTYPE_GPU) {
      err = hsa_amd_profiling_get_dispatch_time(get_compute_agent(place_),
                                                signal, &metrics);
      ErrorCheck(Profiling GPU dispatch, err);
      if (atmi_task_) {
        uint64_t freq;
        err = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq);
        ErrorCheck(Getting system timestamp frequency info, err);
        uint64_t start = metrics.start / (freq / NANOSECS);
        uint64_t end = metrics.end / (freq / NANOSECS);
        DEBUG_PRINT("Ticks: (%lu->%lu)\nFreq: %lu\nTime(ns: (%lu->%lu)\n",
                    metrics.start, metrics.end, freq, start, end);
        atmi_task_->profile.start_time = start;
        atmi_task_->profile.end_time = end;
        atmi_task_->profile.dispatch_time = start;
        atmi_task_->profile.ready_time = start;
      }
    } else {
      /* metrics for CPU tasks will be populated in the
       * worker pthread itself. No special function call */
    }
  }
}

pthread_mutex_t sort_mutex_2(pthread_mutex_t *addr1, pthread_mutex_t *addr2,
                             pthread_mutex_t **addr_low,
                             pthread_mutex_t **addr_hi) {
  if ((uint64_t)addr1 < (uint64_t)addr2) {
    *addr_low = addr1;
    *addr_hi = addr2;
  } else {
    *addr_hi = addr1;
    *addr_low = addr2;
  }
}

void lock(pthread_mutex_t *m) {
  // printf("Locked Mutex: %p\n", m);
  // LockTimer.start();
  pthread_mutex_lock(m);
  // LockTimer.stop();
}

void unlock(pthread_mutex_t *m) {
  // printf("Unlocked Mutex: %p\n", m);
  // LockTimer.start();
  pthread_mutex_unlock(m);
  // LockTimer.stop();
}

// lock set of mutexes in order of the set, but unlock them
// in the reverse order to not deadlock. NOTE that lock_set
// uses forward iterator whereas unlock uses reverse.
void lock_set(const std::set<pthread_mutex_t *> &mutexes) {
  DEBUG_PRINT("Locking [ ");
  for (std::set<pthread_mutex_t *>::iterator it = mutexes.begin();
       it != mutexes.end(); it++) {
    DEBUG_PRINT("%p ", *it);
    lock(*it);
  }
  DEBUG_PRINT("]\n");
}

// lock set of mutexes in order of the set, but unlock them
// in the reverse order to not deadlock. NOTE that lock_set
// uses forward iterator whereas unlock uses reverse.
void unlock_set(const std::set<pthread_mutex_t *> &mutexes) {
  DEBUG_PRINT("Unlocking [ ");
  for (std::set<pthread_mutex_t *>::reverse_iterator it = mutexes.rbegin();
       it != mutexes.rend(); it++) {
    DEBUG_PRINT("%p ", *it);
    unlock(*it);
  }
  DEBUG_PRINT("]\n");
}

void handle_signal_callback(TaskImpl *task) {
  // tasks without atmi_task handle should not be added to callbacks anyway
  assert(task->groupable_ != ATMI_TRUE);
  ComputeTaskImpl *compute_task = dynamic_cast<ComputeTaskImpl *>(task);

  lock(&(task->mutex_));
  task->set_state(ATMI_EXECUTED);
  unlock(&(task->mutex_));
  // after predecessor is done, decrement all successor's dependency count.
  // If count reaches zero, then add them to a 'ready' task list. Next,
  // dispatch all ready tasks in a round-robin manner to the available
  // GPU/CPU queues.
  // decrement reference count of its dependencies; add those with ref count = 0
  // to a 'ready' list.
  TaskImplVecTy &successors = task->and_successors_;
  DEBUG_PRINT("Deps list of %lu [%lu]: ", task->id_, successors.size());
  TaskImplVecTy temp_list;
  for (auto successor : successors) {
    // FIXME: should we be grabbing a lock on each successor before
    // decrementing their predecessor count? Currently, it may not be
    // required because there is only one callback thread, but what if there
    // were more?
    lock(&(successor->mutex_));
    DEBUG_PRINT(" %lu(%d) ", successor->id_, successor->num_predecessors_);
    successor->num_predecessors_--;
    if (successor->num_predecessors_ == 0) {
      // add to ready list
      temp_list.push_back(successor);
    }
    unlock(&(successor->mutex_));
  }
  std::set<pthread_mutex_t *> mutexes;
  // release the kernarg segment back to the kernarg pool
  Kernel *kernel = NULL;
  KernelImpl *kernel_impl = NULL;
  if (compute_task) {
    kernel = compute_task->kernel_;
    if (kernel) {
      kernel_impl = kernel->getKernelImpl(compute_task->kernel_id_);
      mutexes.insert(&(kernel_impl->mutex()));
    }
  }
  mutexes.insert(&mutex_readyq_);
  lock_set(mutexes);
  DEBUG_PRINT("\n");
  for (auto t : temp_list) {
    // FIXME: if groupable task, push it in the right taskgroup
    // else push it to readyqueue
    ReadyTaskQueue.push(t);
  }
  // release your own signal to the pool
  FreeSignalPool.push(task->signal_);
  // release your kernarg region to the pool
  if (kernel && compute_task) {
    DEBUG_PRINT("Freeing Kernarg Segment Id: %d\n",
                compute_task->kernarg_region_index_);
    kernel_impl->free_kernarg_segments().push(
        compute_task->kernarg_region_index_);
  }
  unlock_set(mutexes);
  DEBUG_PRINT(
      "[Handle Signal %lu ] Free Signal Pool Size: %lu; Ready Task Queue Size: "
      "%lu\n",
      task->id_, FreeSignalPool.size(), ReadyTaskQueue.size());
  // dispatch from ready queue if any task exists
  lock(&(task->mutex_));
  task->updateMetrics();
  task->set_state(ATMI_COMPLETED);
  unlock(&(task->mutex_));
  task->doProgress();
}

void handle_signal_barrier_pkt(TaskImpl *task,
                               TaskImplVecTy *dispatched_tasks_ptr) {
  // list of signals that can be reclaimed at every iteration
  std::vector<hsa_signal_t> temp_list;

  // since we attach barrier packet to sink tasks, at this point in the
  // callback,
  // we are guaranteed that all dispatched_tasks have completed execution.
  TaskImplVecTy dispatched_tasks = *dispatched_tasks_ptr;

  // TODO(ashwinma): check the performance implication of locking
  // across the entire loop vs locking across selected sections
  // of code inside the loop
  // lock(&mutex_readyq_);
  for (auto task : dispatched_tasks) {
    assert(task->groupable_ != ATMI_TRUE);
    ComputeTaskImpl *compute_task = dynamic_cast<ComputeTaskImpl *>(task);

    // This dispatched task is now executed
    lock(&(task->mutex_));
    task->set_state(ATMI_EXECUTED);
    unlock(&(task->mutex_));

    // now reclaim its resources (kernel args and signal)
    std::set<pthread_mutex_t *> mutexes;
    Kernel *kernel = NULL;
    KernelImpl *kernel_impl = NULL;
    if (compute_task) {
      kernel = compute_task->kernel_;
      if (kernel) {
        kernel_impl = kernel->getKernelImpl(compute_task->kernel_id_);
        mutexes.insert(&(kernel_impl->mutex()));
      }
    }
    mutexes.insert(&(task->mutex_));
    mutexes.insert(&mutex_readyq_);

    lock_set(mutexes);

    // release the kernarg segment back to the kernarg pool
    if (kernel && compute_task) {
      DEBUG_PRINT("Freeing Kernarg Segment Id: %d\n",
                  compute_task->kernarg_region_index_);
      kernel_impl->free_kernarg_segments().push(
          compute_task->kernarg_region_index_);
    }

    // release the signal back to the signal pool
    DEBUG_PRINT("Task %lu completed\n", task->id_);
    FreeSignalPool.push(task->signal_);

    task->updateMetrics();
    task->set_state(ATMI_COMPLETED);

    unlock_set(mutexes);
  }
  // TODO(ashwinma): check the performance implication of locking
  // across the entire loop vs locking across selected sections
  // of code inside the loop
  // unlock(&mutex_readyq_);

  // this was created on heap when sink tasks was cleared, so reclaim that
  // memory
  delete dispatched_tasks_ptr;

  task->doProgress();
}

bool handle_signal(hsa_signal_value_t value, void *arg) {
  // HandleSignalInvokeTimer.stop();
  static bool is_called = false;
  if (!is_called) {
    set_thread_affinity(1);
    /*
       int policy;
       struct sched_param param;
       pthread_getschedparam(pthread_self(), &policy, &param);
       param.sched_priority = sched_get_priority_max(SCHED_RR);
       printf("Setting Priority Policy for %d: %d\n", SCHED_RR,
       param.sched_priority);
       pthread_setschedparam(pthread_self(), SCHED_RR, &param);
       */
    is_called = true;
  }
  HandleSignalTimer.start();
  TaskImpl *task = NULL;
  TaskImplVecTy *dispatched_tasks_ptr = NULL;

  task = reinterpret_cast<TaskImpl *>(arg);

  DEBUG_PRINT("Handle signal from task %lu\n", task->id_);

  handle_signal_callback(task);

  HandleSignalTimer.stop();
  // HandleSignalInvokeTimer.start();
  return false;
}

bool handle_group_signal(hsa_signal_value_t value, void *arg) {
  // HandleSignalInvokeTimer.stop();
  static bool is_called = false;
  if (!is_called) {
    set_thread_affinity(1);
    /*
       int policy;
       struct sched_param param;
       pthread_getschedparam(pthread_self(), &policy, &param);
       param.sched_priority = sched_get_priority_max(SCHED_RR);
       printf("Setting Priority Policy for %d: %d\n", SCHED_RR,
       param.sched_priority);
       pthread_setschedparam(pthread_self(), SCHED_RR, &param);
       */
    is_called = true;
  }
  HandleSignalTimer.start();
  TaskgroupImpl *taskgroup = reinterpret_cast<TaskgroupImpl *>(arg);
  // TODO(ashwinma): what if the group task has dependents? this is not clear as
  // of now.
  // we need to define dependencies between tasks and task groups better...

  // release resources
  lock(&(taskgroup->group_mutex_));
  hsa_signal_wait_acquire(taskgroup->group_signal_, HSA_SIGNAL_CONDITION_EQ, 0,
                          UINT64_MAX, ATMI_WAIT_STATE);

  std::vector<TaskImpl *> group_tasks = taskgroup->running_groupable_tasks_;
  taskgroup->running_groupable_tasks_.clear();
  if (taskgroup->ordered_) {
    DEBUG_PRINT("Handling group signal callback for task group: %p size: %lu\n",
                taskgroup, group_tasks.size());
    for (int t = 0; t < group_tasks.size(); t++) {
      // does it matter which task was enqueued first, as long as we are
      // popping one task for every push?
      if (!taskgroup->running_ordered_tasks_.empty()) {
        DEBUG_PRINT("Removing task %lu with state: %d\n",
                    taskgroup->running_ordered_tasks_.front()->id_,
                    taskgroup->running_ordered_tasks_.front()->state_.load());
        taskgroup->running_ordered_tasks_.pop_front();
      }
    }
  }
  taskgroup->callback_started_.clear();
  unlock(&(taskgroup->group_mutex_));

  std::set<pthread_mutex_t *> mutexes;
  for (auto task : group_tasks) {
    ComputeTaskImpl *compute_task = dynamic_cast<ComputeTaskImpl *>(task);
    mutexes.insert(&(task->mutex_));
    Kernel *kernel = NULL;
    KernelImpl *kernel_impl = NULL;
    if (compute_task) {
      kernel = compute_task->kernel_;
      kernel_impl = kernel->getKernelImpl(compute_task->kernel_id_);
      mutexes.insert(&(kernel_impl->mutex()));
    }
  }
  mutexes.insert(&mutex_readyq_);
  lock_set(mutexes);
  TaskImpl *some_task = NULL;
  for (auto task : group_tasks) {
    ComputeTaskImpl *compute_task = dynamic_cast<ComputeTaskImpl *>(task);
    Kernel *kernel = NULL;
    KernelImpl *kernel_impl = NULL;
    if (compute_task) {
      kernel = compute_task->kernel_;
      kernel_impl = kernel->getKernelImpl(compute_task->kernel_id_);
      kernel_impl->free_kernarg_segments().push(
          compute_task->kernarg_region_index_);
    }
    {
      // TODO(ashwinma): we dont know how to specify dependencies between task
      // groups and
      // individual tasks yet. once we figure that out, we need to include
      // logic here to push the successor tasks to the ready task queue.
    }
    DEBUG_PRINT("Completed task %lu in group\n", task->id_);
    task->updateMetrics();
    task->set_state(ATMI_COMPLETED);
    some_task = task;
  }
  unlock_set(mutexes);

  // make progress on all other ready tasks that are in the same
  // taskgroup as some task in the collection group_tasks.
  if (some_task) some_task->doProgress();
  DEBUG_PRINT("Releasing %lu tasks from %p task group\n", group_tasks.size(),
              taskgroup);
  taskgroup->task_count_ -= group_tasks.size();
  {
    if (!taskgroup->task_count_.load()) {
      // after predecessor is done, decrement all successor's dependency count.
      // If count reaches zero, then add them to a 'ready' task list. Next,
      // dispatch all ready tasks in a round-robin manner to the available
      // GPU/CPU queues.
      // decrement reference count of its dependencies; add those with ref count
      // = 0 to a 'ready' list
      lock(&(taskgroup->group_mutex_));
      TaskImplVecTy &successors = taskgroup->and_successors_;
      taskgroup->and_successors_.clear();
      unlock(&(taskgroup->group_mutex_));
      DEBUG_PRINT("Deps list of %p [%lu]: ", taskgroup, successors.size());
      TaskImplVecTy temp_list;
      for (auto successor : successors) {
        // FIXME: should we be grabbing a lock on each successor before
        // decrementing their predecessor count? Currently, it may not be
        // required because there is only one callback thread, but what if there
        // were more?
        lock(&(successor->mutex_));
        DEBUG_PRINT(" %lu(%d) ", successor->id_, successor->num_predecessors_);
        successor->num_predecessors_--;
        if (successor->num_predecessors_ == 0) {
          // add to ready list
          temp_list.push_back(successor);
        }
        unlock(&(successor->mutex_));
      }
      lock(&mutex_readyq_);
      for (auto t : temp_list) {
        // FIXME: if groupable task, push it in the right taskgroup
        // else push it to readyqueue
        ReadyTaskQueue.push(t);
        some_task = t;
      }
      unlock(&mutex_readyq_);

      if (!temp_list.empty()) some_task->doProgress();
    }
  }
  HandleSignalTimer.stop();
  // HandleSignalInvokeTimer.start();
  // return true because we want the callback function to always be
  // 'registered' for any more incoming tasks belonging to this task group
  DEBUG_PRINT("Done handle_group_signal\n");
  return false;
}

void TaskImpl::doProgress() {
  {
    if (taskgroup_obj_->ordered_) {
      // pop from front of ordered task list and try dispatch as many as
      // possible
      bool should_dispatch = false;
      do {
        TaskImpl *ready_task = NULL;
        should_dispatch = false;
        lock(&taskgroup_obj_->group_mutex_);
        if (!taskgroup_obj_->running_ordered_tasks_.empty()) {
          ready_task = taskgroup_obj_->running_ordered_tasks_.front();
        }
        unlock(&taskgroup_obj_->group_mutex_);
        if (ready_task) {
          should_dispatch = ready_task->tryDispatch(NULL, /* callback */ true);
        }
      } while (should_dispatch);
    } else {
      lock(&mutex_readyq_);
      size_t queue_sz = ReadyTaskQueue.size();
      unlock(&mutex_readyq_);

      for (int i = 0; i < queue_sz; i++) {
        TaskImpl *ready_task = NULL;
        lock(&mutex_readyq_);
        if (!ReadyTaskQueue.empty()) {
          ready_task = ReadyTaskQueue.front();
          ReadyTaskQueue.pop();
        }
        unlock(&mutex_readyq_);

        if (ready_task) {
          ready_task->tryDispatch(NULL, /* callback */ true);
        }
      }
    }
  }
}

void *acquire_kernarg_segment(KernelImpl *impl, int *segment_id) {
  uint32_t kernel_segment_size = impl->kernarg_segment_size();
  void *ret_address = NULL;
  int free_idx = -1;
  lock(&(impl->mutex()));
  if (!(impl->free_kernarg_segments().empty())) {
    free_idx = impl->free_kernarg_segments().front();
    DEBUG_PRINT("Acquiring Kernarg Segment Id: %d\n", free_idx);
    ret_address = reinterpret_cast<void *>(
        reinterpret_cast<char *>(impl->kernarg_region()) +
        (free_idx * kernel_segment_size));
    impl->free_kernarg_segments().pop();
  } else {
    fprintf(
        stderr,
        "No free kernarg segments. Increase MAX_NUM_KERNELS and recompile.\n");
  }
  unlock(&(impl->mutex()));
  *segment_id = free_idx;
  return ret_address;
  // check if pool is empty by comparing the read and write indexes
  // if pool is empty
  //      extend existing pool
  //      if pool size is beyond a threshold max throw error
  // fi
  // get a free pool object and return
  //
  //
  // on handle signal, just return the free pool object?
}

std::vector<hsa_queue_t *> get_cpu_queues(atmi_place_t place) {
  ATLCPUProcessor &proc = get_processor<ATLCPUProcessor>(place);
  return proc.queues();
}

atmi_status_t ComputeTaskImpl::dispatch() {
  // DEBUG_PRINT("GPU Place Info: %d, %lx %lx\n", lparm->place.node_id,
  // lparm->place.cpu_set, lparm->place.gpu_set);

  TryDispatchTimer.start();
  TaskgroupImpl *taskgroup_obj = taskgroup_obj_;
  /* get this taskgroup's HSA queue (could be dynamically mapped or round
   * robin
   * if it is an unordered taskgroup */
  // FIXME: round robin for now, but may use some other load balancing algo
  // enqueue task's packet to that queue

  int proc_id = place_.device_id;
  if (proc_id == -1) {
    // user is asking runtime to pick a device
    // TODO(ashwinma): best device of this type? pick 0 for now
    proc_id = 0;
  }
  hsa_queue_t *this_Q = packets_[0].first;
  // printf("Task already populated with Q[%p, %lu]\n", this_Q,
  // packets[0].second);
  if (!this_Q) return ATMI_STATUS_ERROR;

  int ndim = -1;
  if (gridDim_[2] > 1)
    ndim = 3;
  else if (gridDim_[1] > 1)
    ndim = 2;
  else
    ndim = 1;
  if (devtype_ == ATMI_DEVTYPE_GPU) {
    hsa_kernel_dispatch_packet_t *this_aql = NULL;
    uint64_t index = 0ull;
    index = packets_[0].second;
    /* Find the queue index address to write the packet info into.  */
    const uint32_t queueMask = this_Q->size - 1;
    this_aql = &(((hsa_kernel_dispatch_packet_t
                       *)(this_Q->base_address))[index & queueMask]);
    DEBUG_PRINT("K for %p (%lu) %p [%lu]\n", this, id_, this_Q,
                index & queueMask);

    KernelImpl *kernel_impl = kernel_->getKernelImpl(kernel_id_);
    // this_aql->header = create_header(HSA_PACKET_TYPE_INVALID, ATMI_FALSE);
    // memset(this_aql, 0, sizeof(hsa_kernel_dispatch_packet_t));
    /*  FIXME: We need to check for queue overflow here. */
    // SignalAddTimer.start();
    if (groupable_ == ATMI_TRUE) {
      lock(&(taskgroup_obj->group_mutex_));
      taskgroup_obj->running_groupable_tasks_.push_back(this);
      unlock(&(taskgroup_obj->group_mutex_));
    }
    // SignalAddTimer.stop();
    this_aql->completion_signal = signal_;

    /* pass this task handle to the kernel as an argument */
    char *kargs = reinterpret_cast<char *>(kernarg_region_);
    // AMDGCN device libs assume that first three args beyond the kernel args
    // are grid
    // offsets in X, Y and Z dimensions
    atmi_implicit_args_t *impl_args = reinterpret_cast<atmi_implicit_args_t *>(
        kargs + (kernarg_region_size_ - sizeof(atmi_implicit_args_t)));
    impl_args->offset_x = 0;
    impl_args->offset_y = 0;
    impl_args->offset_z = 0;

    // assign a hostcall buffer for the selected Q
    {
      KernelImpl *kernel_impl = NULL;
      if (g_atmi_hostcall_required) {
        if (kernel_) {
          kernel_impl = kernel_->getKernelImpl(kernel_id_);
          // printf("Task Id: %lu, kernel name: %s\n", id_,
          // kernel_impl->kernel_name.c_str());
          char *kargs = reinterpret_cast<char *>(kernarg_region_);
          if (type() == ATL_KERNEL_EXECUTION && devtype_ == ATMI_DEVTYPE_GPU &&
              kernel_impl->platform_type() == AMDGCN) {
            atmi_implicit_args_t *impl_args =
                reinterpret_cast<atmi_implicit_args_t *>(
                    kargs +
                    (kernarg_region_size_ - sizeof(atmi_implicit_args_t)));
            impl_args->hostcall_ptr =
                atmi_hostcall_assign_buffer(this_Q, proc_id);
          }
        }
      }
    }

    /*  Process task values */
    /*  this_aql.dimensions=(uint16_t) ndim; */
    this_aql->setup |= (uint16_t)ndim
                       << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    this_aql->grid_size_x = gridDim_[0];
    this_aql->workgroup_size_x = groupDim_[0];
    if (ndim > 1) {
      this_aql->grid_size_y = gridDim_[1];
      this_aql->workgroup_size_y = groupDim_[1];
    } else {
      this_aql->grid_size_y = 1;
      this_aql->workgroup_size_y = 1;
    }
    if (ndim > 2) {
      this_aql->grid_size_z = gridDim_[2];
      this_aql->workgroup_size_z = groupDim_[2];
    } else {
      this_aql->grid_size_z = 1;
      this_aql->workgroup_size_z = 1;
    }

    /*  Bind kernel argument buffer to the aql packet.  */
    this_aql->kernarg_address = kernarg_region_;
    this_aql->kernel_object =
        dynamic_cast<GPUKernelImpl *>(kernel_impl)->kernel_objects_[proc_id];
    this_aql->private_segment_size = dynamic_cast<GPUKernelImpl *>(kernel_impl)
                                         ->private_segment_sizes_[proc_id];
    this_aql->group_segment_size = dynamic_cast<GPUKernelImpl *>(kernel_impl)
                                       ->group_segment_sizes_[proc_id];

    this_aql->reserved2 = id_;
    set_state(ATMI_DISPATCHED);
    /*  Prepare and set the packet header */
    packet_store_release(
        reinterpret_cast<uint32_t *>(this_aql),
        create_header(HSA_PACKET_TYPE_KERNEL_DISPATCH, taskgroup_obj->ordered_,
                      acquire_scope_, release_scope_),
        this_aql->setup);
    /* Increment write index and ring doorbell to dispatch the kernel.  */
    // hsa_queue_store_write_index_relaxed(this_Q, index+1);
    hsa_signal_store_relaxed(this_Q->doorbell_signal, index);
  } else if (devtype_ == ATMI_DEVTYPE_CPU) {
    fprintf(stderr, "Unsupported device type: %d\n", devtype_);
    return ATMI_STATUS_ERROR;
  }
  TryDispatchTimer.stop();
  DEBUG_PRINT("Task %lu (%d) Dispatched\n", id_, devtype_);
  return ATMI_STATUS_SUCCESS;
}

void ComputeTaskImpl::updateKernargRegion(void **args) {
  char *thisKernargAddress = reinterpret_cast<char *>(kernarg_region_);
  if (kernel_->num_args() && thisKernargAddress == NULL) {
    fprintf(stderr, "Unable to allocate/find free kernarg segment\n");
  }
  KernelImpl *kernel_impl = kernel_->getKernelImpl(kernel_id_);

  // Argument references will be copied to a contiguous memory region here
  // TODO(ashwinma): resolve all data affinities before copying, depending on
  // atmi_data_affinity_policy_t: ATMI_COPY, ATMI_NOCOPY
  for (int i = 0; i < kernel_->num_args(); i++) {
    memcpy(thisKernargAddress + kernel_impl->arg_offsets()[i], args[i],
           kernel_->arg_sizes()[i]);
    // hsa_memory_register(thisKernargAddress, ???
    DEBUG_PRINT("Arg[%d] = %p\n", i, *(void **)((char *)thisKernargAddress +
                                                kernel_impl->arg_offsets()[i]));
  }
}

void ComputeTaskImpl::acquireAqlPacket() {
  TaskgroupImpl *taskgroup_obj = taskgroup_obj_;
  // get AQL queue for GPU tasks and CPU tasks
  hsa_queue_t *this_Q = NULL;
  if (devtype_ == ATMI_DEVTYPE_GPU)
    this_Q = taskgroup_obj->chooseQueueFromPlace<ATLGPUProcessor>(place_);
  else if (devtype_ == ATMI_DEVTYPE_CPU) {
    fprintf(stderr, "Unsupported device type: %d\n", devtype_);
    return;
  }
  if (!this_Q) ATMIErrorCheck(Getting queue for dispatch, ATMI_STATUS_ERROR);

  if (devtype_ == ATMI_DEVTYPE_GPU) {
    hsa_signal_add_acq_rel(signal_, 1);
    /*  Obtain the current queue write index. increases with each call to
     * kernel  */
    // uint64_t index = hsa_queue_add_write_index_relaxed(this_Q, 1);
    // Atomically request a new packet ID.
    uint64_t index = hsa_queue_add_write_index_relaxed(this_Q, 1);
    // Wait until the queue is not full before writing the packet
    DEBUG_PRINT("Queue %p ordered? %d task: %lu index: %lu\n", this_Q,
                taskgroup_obj->ordered_, id_, index);
    // printf("%lu, %lu, K\n", id_, index);
    while (index - hsa_queue_load_read_index_acquire(this_Q) >= this_Q->size) {
    }
    packets_.push_back(std::make_pair(this_Q, index));
  }
}

bool TaskImpl::tryDispatchHostCallback(void **args) {
  bool should_tryDispatch = true;
  bool resources_available = true;
  bool predecessors_complete = true;
  bool should_dispatch = false;

  std::set<pthread_mutex_t *> req_mutexes;
  req_mutexes.clear();
  for (auto pred_task : predecessors_) {
    req_mutexes.insert(&(pred_task->mutex_));
  }
  req_mutexes.insert(&(mutex_));
  req_mutexes.insert(&mutex_readyq_);
  if (prev_ordered_task_) req_mutexes.insert(&(prev_ordered_task_->mutex_));
  TaskgroupImpl *taskgroup_obj = taskgroup_obj_;
  req_mutexes.insert(&(taskgroup_obj->group_mutex_));
  ComputeTaskImpl *compute_task = dynamic_cast<ComputeTaskImpl *>(this);

  KernelImpl *kernel_impl = NULL;
  if (compute_task) {
    kernel_impl =
        compute_task->kernel_->getKernelImpl(compute_task->kernel_id_);
    req_mutexes.insert(&(kernel_impl->mutex()));
  }
  lock_set(req_mutexes);

  if (state_ >= ATMI_READY) {
    // If someone else is trying to dispatch this task, give up
    unlock_set(req_mutexes);
    return false;
  }
  // do not add predecessor-successor link if task is already initialized using
  // create-activate pattern
  if (state_ < ATMI_INITIALIZED && should_tryDispatch) {
    if (!predecessors_.empty()) {
      // add to its predecessor's dependents list and return
      for (auto pred_task : predecessors_) {
        DEBUG_PRINT("Task %p depends on %p as predecessor ", this, pred_task);
        if (pred_task->state_ /*.load(std::memory_order_seq_cst)*/ <
            ATMI_EXECUTED) {
          should_tryDispatch = false;
          predecessors_complete = false;
          pred_task->and_successors_.push_back(this);
          num_predecessors_++;
          DEBUG_PRINT("(waiting)\n");
          waiting_count++;
        } else {
          DEBUG_PRINT("(completed)\n");
        }
      }
    }
    if (pred_taskgroup_objs_.size() > 0) {
      // add to its predecessor's dependents list and return
      DEBUG_PRINT("Task %lu has %lu predecessor task groups\n", id_,
                  pred_taskgroup_objs_.size());
      for (auto pred_tg : pred_taskgroup_objs_) {
        DEBUG_PRINT("Task %p depends on %p as predecessor task group ", this,
                    pred_tg);
        if (pred_tg && pred_tg->task_count_.load() > 0) {
          // predecessor task group is still running, so add yourself to its
          // successor list
          should_tryDispatch = false;
          predecessors_complete = false;
          pred_tg->and_successors_.push_back(this);
          num_predecessors_++;
          DEBUG_PRINT("(waiting)\n");
        } else {
          DEBUG_PRINT("(completed)\n");
        }
      }
    }
  }

  if (prev_ordered_task_ && should_tryDispatch) {
    DEBUG_PRINT("Task %lu depends on %lu as ordered predecessor ", id_,
                prev_ordered_task_->id_);
    // if this task is of a certain type and its previous task was also of the
    // same type,
    // then we can dispatch this task if the previous task has also been
    // dispatched
    // (received task signal and set its value)
    if (prev_ordered_task_->state_ < ATMI_READY &&
        ((prev_ordered_task_->type() == ATL_DATA_MOVEMENT &&
          type() == ATL_DATA_MOVEMENT) ||
         (prev_ordered_task_->type() == ATL_KERNEL_EXECUTION &&
          type() == ATL_KERNEL_EXECUTION &&
          ((prev_ordered_task_->devtype_ == ATMI_DEVTYPE_CPU &&
            devtype_ == ATMI_DEVTYPE_CPU) ||
           (prev_ordered_task_->devtype_ == ATMI_DEVTYPE_GPU &&
            devtype_ == ATMI_DEVTYPE_GPU))))) {
      should_tryDispatch = false;
      predecessors_complete = false;
      DEBUG_PRINT("(waiting)\n");
      waiting_count++;
    } else if (prev_ordered_task_->state_ < ATMI_EXECUTED &&
               ((prev_ordered_task_->type() == ATL_DATA_MOVEMENT &&
                 type() == ATL_KERNEL_EXECUTION) ||
                (prev_ordered_task_->type() == ATL_KERNEL_EXECUTION &&
                 type() == ATL_DATA_MOVEMENT) ||
                (prev_ordered_task_->devtype_ == ATMI_DEVTYPE_GPU &&
                 devtype_ == ATMI_DEVTYPE_CPU) ||
                (prev_ordered_task_->devtype_ == ATMI_DEVTYPE_CPU &&
                 devtype_ == ATMI_DEVTYPE_GPU))) {
      // if this task is of a certain type and its previous task
      // is of a different type, then we can dispatch this task
      // ONLY if the previous task has been at least
      // executed; and also add the previous task as a predecessor
      should_tryDispatch = false;
      predecessors_complete = false;
      if (state_ < ATMI_INITIALIZED) {
        prev_ordered_task_->and_successors_.push_back(this);
        num_predecessors_++;
      }
      DEBUG_PRINT("(waiting)\n");
      waiting_count++;
    } else {
      DEBUG_PRINT("(dispatched)\n");
    }
  }

  if (should_tryDispatch) {
    if ((kernel_impl && kernel_impl->free_kernarg_segments().empty()) ||
        (groupable_ == ATMI_FALSE && FreeSignalPool.empty())) {
      should_tryDispatch = false;
      resources_available = false;
    }
  }

  if (should_tryDispatch) {
    // try to dispatch if
    // a) you are using callbacks to resolve dependencies and all
    // your predecessors are done executing, OR
    // b) you are using barrier packets, in which case always try
    // to launch if you have a free signal at hand
    if (groupable_ == ATMI_TRUE) {
      signal_ = taskgroup_obj->signal();
    } else {
      // get a free signal
      hsa_signal_t new_signal = FreeSignalPool.front();
      signal_ = new_signal;
      DEBUG_PRINT("Before pop Signal handle: %" PRIu64
                  " Signal value:%ld (Signal pool sz: %lu)\n",
                  signal_.handle, hsa_signal_load_relaxed(signal_),
                  FreeSignalPool.size());
      FreeSignalPool.pop();
      DEBUG_PRINT(
          "[Try Dispatch] Free Signal Pool Size: %lu; Ready Task Queue Size: "
          "%lu\n",
          FreeSignalPool.size(), ReadyTaskQueue.size());
      // unlock(&mutex_readyq_);
    }
    if (compute_task) {
      // add itself to its kernel implementation's list of tasks
      // so that it can be waited upon for completion when the kernel
      // is terminated
      kernel_impl->launched_tasks().push_back(compute_task);
      // get kernarg resource
      uint32_t kernarg_segment_size = kernel_impl->kernarg_segment_size();
      int free_idx = kernel_impl->free_kernarg_segments().front();
      DEBUG_PRINT("Acquiring Kernarg Segment Id: %d\n", free_idx);
      compute_task->kernarg_region_index_ = free_idx;
      void *addr = reinterpret_cast<void *>(
          reinterpret_cast<char *>(kernel_impl->kernarg_region()) +
          (free_idx * kernarg_segment_size));
      kernel_impl->free_kernarg_segments().pop();
      assert(!(compute_task->kernel_->num_args()) ||
             (compute_task->kernel_->num_args() &&
              (compute_task->kernarg_region_ || args)));
      if (compute_task->kernarg_region_ != NULL) {
        // we had already created a memory region using malloc. Copy it
        // to the newly availed space
        size_t size_to_copy = compute_task->kernarg_region_size_;
        if (devtype_ == ATMI_DEVTYPE_GPU &&
            kernel_impl->platform_type() == AMDGCN) {
          // do not copy the implicit args from saved region
          // they are to be set/reset during task dispatch
          size_to_copy -= sizeof(atmi_implicit_args_t);
        }
        if (size_to_copy)
          memcpy(addr, compute_task->kernarg_region_, size_to_copy);
        // free existing region
        free(compute_task->kernarg_region_);
        compute_task->kernarg_region_ = addr;
      } else {
        // first time allocation/assignment
        compute_task->kernarg_region_ = addr;
        compute_task->updateKernargRegion(args);
      }
    }

    if (taskgroup_obj->ordered_)
      taskgroup_obj->running_ordered_tasks_.pop_front();

    acquireAqlPacket();

    // now the task (kernel/data movement) has the signal/kernarg resource
    // and can be set to the "ready" state
    set_state(ATMI_READY);
  } else {
    if (compute_task) {
      if (compute_task->kernel_ && compute_task->kernarg_region_ == NULL) {
        // first time allocation/assignment
        compute_task->kernarg_region_ =
            malloc(compute_task->kernarg_region_size_);
        // kernarg_region_copied = true;
        compute_task->updateKernargRegion(args);
      }
    }
    set_state(ATMI_INITIALIZED);
  }
  if (predecessors_complete == true && resources_available == false &&
      taskgroup_obj_->ordered_ == false) {
    // Ready task but no resources available. So, we push it to a
    // ready queue
    ReadyTaskQueue.push(this);
    max_ready_queue_sz++;
  }

  unlock_set(req_mutexes);
  return should_tryDispatch;
}

TaskImpl::TaskImpl()
    : id_(ATMI_NULL_TASK_HANDLE),
      place_(ATMI_DEFAULT_PLACE),
      devtype_(ATMI_DEVTYPE_ALL),
      state_(ATMI_UNINITIALIZED),
      atmi_task_(NULL),
      taskgroup_obj_(NULL),
      num_predecessors_(0),
      num_successors_(0),
      prev_ordered_task_(NULL),
      acquire_scope_(ATMI_FENCE_SCOPE_SYSTEM),
      release_scope_(ATMI_FENCE_SCOPE_SYSTEM),
      // is_continuation_(false),
      // continuation_task_(NULL)
      profilable_(false),
      groupable_(false),
      synchronous_(false) {
  pthread_mutex_init(&mutex_, NULL);
}

TaskImpl::~TaskImpl() {
  // packets_.clear();
}

ComputeTaskImpl::ComputeTaskImpl(Kernel *kernel)
    : TaskImpl(), kernel_(kernel), kernel_id_(-1), kernarg_region_(NULL) {
  lock(&mutex_all_tasks_);
  AllTasks.push_back(this);
  atmi_task_handle_t new_id;
  set_task_handle_ID(&new_id, AllTasks.size() - 1);
  unlock(&mutex_all_tasks_);
  id_ = new_id;
}

ComputeTaskImpl::ComputeTaskImpl(atmi_lparm_t *lparm, Kernel *kernel,
                                 int kernel_id)
    : TaskImpl(),
      kernel_(kernel),
      kernel_id_(kernel_id),
      kernarg_region_(NULL) {
  lock(&mutex_all_tasks_);
  AllTasks.push_back(this);
  atmi_task_handle_t new_id;
  set_task_handle_ID(&new_id, AllTasks.size() - 1);
  unlock(&mutex_all_tasks_);
  id_ = new_id;

  setParams(lparm);
}

void ComputeTaskImpl::setParams(const atmi_lparm_t *lparm) {
  static bool is_called = false;
  if (!is_called) {
    set_thread_affinity(0);

    /* int policy;
       struct sched_param param;
       pthread_getschedparam(pthread_self(), &policy, &param);
       param.sched_priority = sched_get_priority_min(policy);
       printf("Setting Priority Policy for %d: %d\n", policy,
       param.sched_priority);
       pthread_setschedparam(pthread_self(), policy, &param);
       */
    is_called = true;
  }

  KernelImpl *kernel_impl = kernel_->getKernelImpl(kernel_id_);
  kernarg_region_ = NULL;
  kernarg_region_size_ = kernel_impl->kernarg_segment_size();
  devtype_ = kernel_impl->devtype();
  profilable_ = lparm->profilable;
  groupable_ = lparm->groupable;
  atmi_task_ = lparm->task_info;

  // assign the memory scope
  acquire_scope_ = lparm->acquire_scope;
  release_scope_ = lparm->release_scope;

  // fill in from lparm
  for (int i = 0; i < 3 /* 3dims */; i++) {
    gridDim_[i] = lparm->gridDim[i];
    groupDim_[i] = lparm->groupDim[i];
  }
  // DEBUG_PRINT("Requires LHS: %p and RHS: %p\n", lparm.requires,
  // lparm->requires);
  // DEBUG_PRINT("Requires ThisTask: %p and ThisTask: %p\n",
  // lparm.task_info, lparm->task_info);

  /* Add row to taskgroup table for purposes of future synchronizations */
  taskgroup_ = lparm->group;
  taskgroup_obj_ = getTaskgroupImpl(taskgroup_);

  place_ = lparm->place;
  synchronous_ = lparm->synchronous;
  // DEBUG_PRINT("Taskgroup LHS: %p and RHS: %p\n", lparm.group,
  // lparm->group);
  num_predecessors_ = 0;
  num_successors_ = 0;

  /* For dependent child tasks, add dependent parent kernels to barriers.  */
  DEBUG_PRINT("Pif %s requires %d task\n", kernel_impl->name().c_str(),
              lparm->num_required);

  predecessors_.clear();
  // predecessors.resize(lparm->num_required);
  for (int idx = 0; idx < lparm->num_required; idx++) {
    TaskImpl *pred_task = getTaskImpl(lparm->requires[idx]);
    // assert(pred_task != NULL);
    if (pred_task) {
      DEBUG_PRINT("Task %lu adding %lu task as predecessor\n", id_,
                  pred_task->id_);
      predecessors_.push_back(pred_task);
    }
  }
  pred_taskgroup_objs_.clear();
  pred_taskgroup_objs_.resize(lparm->num_required_groups);
  for (int idx = 0; idx < lparm->num_required_groups; idx++) {
    pred_taskgroup_objs_[idx] = getTaskgroupImpl(lparm->required_groups[idx]);
  }

  lock(&(taskgroup_obj_->group_mutex_));
  if (taskgroup_obj_->ordered_) {
    taskgroup_obj_->running_ordered_tasks_.push_back(this);
    prev_ordered_task_ = taskgroup_obj_->last_task_;
    taskgroup_obj_->last_task_ = this;
  } else {
    taskgroup_obj_->running_default_tasks_.push_back(this);
  }
  unlock(&(taskgroup_obj_->group_mutex_));
  if (groupable_) {
    DEBUG_PRINT("Add ref_cnt 1 to task group %p\n", taskgroup_obj_);
    (taskgroup_obj_->task_count_)++;
  }
}

ComputeTaskImpl *createComputeTaskImpl(atmi_lparm_t *lparm,
                                       atmi_kernel_t atmi_kernel) {
  ComputeTaskImpl *task = NULL;
  // get kernel impl object and set all relevant
  // task params
  int kernel_id = -1;
  Kernel *kernel = get_kernel_obj(atmi_kernel);
  if (kernel) {
    kernel_id = kernel->getKernelImplId(lparm);
  }
  if (kernel_id == -1) return NULL;

  /*lock(&(kernel_impl->mutex));
    if(kernel_impl->free_kernarg_segments().empty()) {
  // no free kernarg segments -- allocate some more?
  // FIXME: realloc instead? HSA realloc?
  }
  unlock(&(kernel_impl->mutex));
  */
  task = new ComputeTaskImpl(lparm, kernel, kernel_id);

  return task;
}


bool TaskImpl::tryDispatch(void **args, bool isCallback) {
  ShouldDispatchTimer.start();
  TaskImpl *returned_task = this;

  bool should_dispatch = tryDispatchHostCallback(args);
  ShouldDispatchTimer.stop();

  if (should_dispatch) {
    bool register_task_callback = (returned_task->groupable_ != ATMI_TRUE);
    // direct_dispatch++;
    DEBUG_PRINT("Dispatching task %lu\n", returned_task->id_);
    ATMIErrorCheck(Dispatch compute kernel, returned_task->dispatch());

    RegisterCallbackTimer.start();
    if (register_task_callback) {
      {
        DEBUG_PRINT("Registering callback for task %luthis, \n", id_);
        hsa_status_t err = hsa_amd_signal_async_handler(
            returned_task->signal_, HSA_SIGNAL_CONDITION_EQ, 0, handle_signal,
            reinterpret_cast<void *>(returned_task));
        ErrorCheck(Creating signal handler, err);
      }
    } else {
      if (!returned_task->taskgroup_obj_->callback_started_.test_and_set()) {
        DEBUG_PRINT("Registering callback for task groups\n");
        hsa_status_t err = hsa_amd_signal_async_handler(
            returned_task->signal_, HSA_SIGNAL_CONDITION_EQ, 0,
            handle_group_signal,
            reinterpret_cast<void *>(returned_task->taskgroup_obj_));
        ErrorCheck(Creating signal handler, err);
      }
    }
    RegisterCallbackTimer.stop();
  }
  if (synchronous_ == ATMI_TRUE) { /*  Sychronous execution */
    /* For default synchrnous execution, wait til kernel is finished.  */
    //    TaskWaitTimer.start();
    if (groupable_ != ATMI_TRUE) {
      wait();
    } else {
      taskgroup_obj_->sync();
    }
    updateMetrics();
    // set_task_metrics(ret);
    set_state(ATMI_COMPLETED);
    //  TaskWaitTimer.stop();
    // std::cout << "Task Wait Interim Timer " << TaskWaitTimer << std::endl;
    // std::cout << "Launch Time: " << TryLaunchTimer << std::endl;
    // Done with dispatch, do not dispatch more.
    should_dispatch = false;
  }
  return should_dispatch;
}

atmi_task_handle_t ComputeTaskImpl::tryLaunchKernel(void **args) {
  atmi_task_handle_t task_handle = ATMI_NULL_TASK_HANDLE;
  // set_task_params(ret, lparm, kernel_id, args);
  // perhaps second arg here should be null because it is
  // already set for both HC and BP?
  tryDispatch(args, /* callback */ false);
  task_handle = id_;
  return task_handle;
}

Kernel *get_kernel_obj(atmi_kernel_t atmi_kernel) {
  std::map<uint64_t, Kernel *>::iterator map_iter;
  map_iter = KernelImplMap.find(atmi_kernel.handle);
  if (map_iter == KernelImplMap.end()) {
    DEBUG_PRINT("ERROR: Kernel/PIF %lu not found\n", atmi_kernel.handle);
    return NULL;
  }
  return map_iter->second;
}

atmi_task_handle_t Runtime::LaunchTask(
    atmi_lparm_t *lparm, atmi_kernel_t atmi_kernel,
    void **args /*, more params for place info? */) {
  ParamsInitTimer.start();

  assert(!(lparm->place.type & ATMI_DEVTYPE_CPU));
  atmi_task_handle_t task_handle = ATMI_NULL_TASK_HANDLE;
  if ((lparm->place.type & ATMI_DEVTYPE_GPU && !atlc.g_gpu_initialized) ||
      (lparm->place.type & ATMI_DEVTYPE_CPU))
    return task_handle;

  /*lock(&(kernel_impl->mutex));
    if(kernel_impl->free_kernarg_segments().empty()) {
  // no free kernarg segments -- allocate some more?
  // FIXME: realloc instead? HSA realloc?
  }
  unlock(&(kernel_impl->mutex));
  */
  // TaskImpl *task = atl_trycreate_task(kernel);

  ComputeTaskImpl *compute_task = createComputeTaskImpl(lparm, atmi_kernel);
  if (compute_task) {
    // Save kernel args because it will be activated
    // later. This way, the user will be able to
    // reuse their kernarg region that was created
    // in the application space.
    if (compute_task->kernel_ && compute_task->kernarg_region_ == NULL) {
      // first time allocation/assignment
      compute_task->kernarg_region_ =
          malloc(compute_task->kernarg_region_size_);
      // task_handle->kernarg_region_copied = true;
      compute_task->updateKernargRegion(args);
    }
    task_handle = compute_task->tryLaunchKernel(args);
    // task_handle = atl_trylaunch_kernel(lparm, task, kernel_id, args);
    DEBUG_PRINT("[Returned Task: %lu]\n", task_handle);
  }
  return task_handle;
}

}  // namespace core
