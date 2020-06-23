/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/

#ifndef SRC_RUNTIME_INCLUDE_TASK_H_
#define SRC_RUNTIME_INCLUDE_TASK_H_

#include <utility>
#include <vector>
#include "internal.h"

namespace core {
class TaskImpl;
using TaskImplVecTy = std::vector<TaskImpl *>;

class TaskImpl {
 public:
  TaskImpl();
  virtual ~TaskImpl();

  virtual atl_task_type_t type() const = 0;
  void set_state(const atmi_state_t state);
  atmi_state_t state() const { return state_; }
  void updateMetrics();
  void wait();

  bool tryDispatch(void **args, bool callback = false);

  void doProgress();

 private:
  bool tryDispatchBarrierPacket(void **args, TaskImpl **returned_task);
  bool tryDispatchHostCallback(void **args);
  virtual atmi_status_t dispatch() = 0;
  virtual void acquireAqlPacket() = 0;

 public:
  // track progress via HSA signal
  hsa_signal_t signal_;

  // combination of queue and index determines the AQL packet that was
  // used for this task. GPU tasks (currently) point to a single packet
  // but CPU tasks can have multiple AQL packets on different queues,
  // where each packet computes a sub-space of the problem on possibly
  // a different CPU thread.
  // TODO(ashwinma): check if the same can be conceptually extended to
  // multiple GPU setups as well.
  // TODO(ashwinma): this data structure may be reused for SDMA packets
  // if and when they become AQL-like user mode queues.
  std::vector<std::pair<hsa_queue_t *, uint64_t> > packets_;

  atmi_task_handle_t id_;

  // place object on which this task should execute; a default
  // place object implies that the runtime can finally choose the
  // location to execute the task.
  atmi_place_t place_;

  // if the device type does not match with the place
  // of the task, the behavior is undefined.
  atmi_devtype_t devtype_;

  // TODO(ashwinma): check to see if we really need atomic object
  // or are we using locks anyway?
  std::atomic<atmi_state_t> state_;

  // userspace task structure that holds profiling data and task state.
  atmi_task_t *atmi_task_;

  // taskgroup information
  core::TaskgroupImpl *taskgroup_obj_;
  atmi_taskgroup_handle_t taskgroup_;
  // atmi_taskgroup_t group_;

  // task dependency information
  uint32_t num_predecessors_;
  uint32_t num_successors_;
  // FIXME: queue or vector?
  // list of dependents
  // predecessors_ is a collection of all predecessor tasks to this task.
  TaskImplVecTy predecessors_;
  // and_predecessors_ and and_successors_ capture the dynamic pred/successor
  // lists of this task. If a pred task has already completed, that task
  // pointer will still exist in predecessors_ but will not be present in
  // and_predecessors_
  TaskImplVecTy and_successors_;
  TaskImplVecTy and_predecessors_;
  // Predecessor taskgroup. Any task from this taskgroup waits for
  // all tasks in the predecessor taskgroup to complete before launching.
  // This assumes that tasks are groupable so that a single signal value
  // is applied to each of the taskgroups (for groupable tasks only).
  std::vector<core::TaskgroupImpl *> pred_taskgroup_objs_;
  // Task pointer of predecessor task (for ordered task groups only).
  TaskImpl *prev_ordered_task_;

  // memory scope for the task
  atmi_task_fence_scope_t acquire_scope_;
  atmi_task_fence_scope_t release_scope_;

  // other miscellaneous flags
  bool profilable_;
  bool groupable_;
  bool synchronous_;

  // per task mutex to reduce contention
  pthread_mutex_t mutex_;

  // TODO(ashwinma): experimental flags to differentiate between
  // a regular task and a continuation
  // FIXME: probably make this a class hierarchy?
  boolean is_continuation_;
  TaskImpl *continuation_task_;
};  // class TaskImpl

class ComputeTaskImpl : public TaskImpl {
 public:
  // partial construction allowed for the experimental task template creation
  // and activation.
  explicit ComputeTaskImpl(Kernel *kernel);
  ComputeTaskImpl(atmi_lparm_t *lparm, Kernel *kernel, int kernel_id);
  ~ComputeTaskImpl() {}

  atl_task_type_t type() const override { return ATL_KERNEL_EXECUTION; };

  void updateKernargRegion(void **args);
  // if the construction of this object is split so that we dont
  // assign all params at once, we have a function to help set the
  // remaining params -- used by the experimental template task creation
  // and activation.
  void setParams(const atmi_lparm_t *lparm);

 private:
  atmi_status_t dispatch() override;
  void acquireAqlPacket() override;

 public:
  atmi_task_handle_t tryLaunchKernel(void **args);

  // the kernel object that defines what this task will execute
  core::Kernel *kernel_;
  // A kernel may have different implementations. kernel_id_ indexes
  // into the implementation of the kernel (via an id map).
  uint32_t kernel_id_;
  // malloced or acquired from a pool
  void *kernarg_region_;
  size_t kernarg_region_size_;
  // this is an index into the free kernarg segment pool
  int kernarg_region_index_;
  bool kernarg_region_copied_;
  // dimensions of the compute grid to be launched
  unsigned long gridDim_[3];
  unsigned long groupDim_[3];
};  // class ComputeTaskImpl


}  // namespace core
#endif  // SRC_RUNTIME_INCLUDE_TASK_H_
