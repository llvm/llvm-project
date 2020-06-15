/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#include "data.h"
#include <hsa.h>
#include <hsa_ext_amd.h>
#include <stdio.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include "atmi_runtime.h"
#include "internal.h"
#include "machine.h"
#include "rt.h"
#include "task.h"
#include "taskgroup.h"

using core::DataTaskImpl;
using core::TaskImpl;
extern ATLMachine g_atl_machine;
extern hsa_signal_t IdentityCopySignal;

namespace core {
#ifndef USE_ROCR_PTR_INFO
ATLPointerTracker g_data_map;  // Track all am pointer allocations.
#endif
void allow_access_to_all_gpu_agents(void *ptr);
// std::map<void *, ATLData *> MemoryMap;

const char *getPlaceStr(atmi_devtype_t type) {
  switch (type) {
    case ATMI_DEVTYPE_CPU:
      return "CPU";
    case ATMI_DEVTYPE_GPU:
      return "GPU";
    default:
      return NULL;
  }
}

std::ostream &operator<<(std::ostream &os, const ATLData *ap) {
  atmi_mem_place_t place = ap->place();
  os << "hostPointer:" << ap->host_aliasptr() << " devicePointer:" << ap->ptr()
     << " sizeBytes:" << ap->size() << " place:(" << getPlaceStr(place.dev_type)
     << ", " << place.dev_id << ", " << place.mem_id << ")";
  return os;
}

#ifndef USE_ROCR_PTR_INFO
void ATLPointerTracker::insert(void *pointer, ATLData *p) {
  std::lock_guard<std::mutex> l(mutex_);

  DEBUG_PRINT("insert: %p + %zu\n", pointer, p->size());
  tracker_.insert(std::make_pair(ATLMemoryRange(pointer, p->size()), p));
}

void ATLPointerTracker::remove(void *pointer) {
  std::lock_guard<std::mutex> l(mutex_);
  DEBUG_PRINT("remove: %p\n", pointer);
  tracker_.erase(ATLMemoryRange(pointer, 1));
}

ATLData *ATLPointerTracker::find(const void *pointer) {
  std::lock_guard<std::mutex> l(mutex_);
  ATLData *ret = NULL;
  auto iter = tracker_.find(ATLMemoryRange(pointer, 1));
  DEBUG_PRINT("find: %p\n", pointer);
  if (iter != tracker_.end())  // found
    ret = iter->second;
  return ret;
}
#endif

ATLProcessor &get_processor_by_compute_place(atmi_place_t place) {
  int dev_id = place.device_id;
  switch (place.type) {
    case ATMI_DEVTYPE_CPU:
      return g_atl_machine.processors<ATLCPUProcessor>()[dev_id];
    case ATMI_DEVTYPE_GPU:
      return g_atl_machine.processors<ATLGPUProcessor>()[dev_id];
  }
}

ATLProcessor &get_processor_by_mem_place(atmi_mem_place_t place) {
  int dev_id = place.dev_id;
  switch (place.dev_type) {
    case ATMI_DEVTYPE_CPU:
      return g_atl_machine.processors<ATLCPUProcessor>()[dev_id];
    case ATMI_DEVTYPE_GPU:
      return g_atl_machine.processors<ATLGPUProcessor>()[dev_id];
  }
}

hsa_agent_t get_compute_agent(atmi_place_t place) {
  return get_processor_by_compute_place(place).agent();
}

hsa_agent_t get_mem_agent(atmi_mem_place_t place) {
  return get_processor_by_mem_place(place).agent();
}

hsa_amd_memory_pool_t get_memory_pool_by_mem_place(atmi_mem_place_t place) {
  ATLProcessor &proc = get_processor_by_mem_place(place);
  return get_memory_pool(proc, place.mem_id);
}

void register_allocation(void *ptr, size_t size, atmi_mem_place_t place) {
#ifndef USE_ROCR_PTR_INFO
  ATLData *data = new ATLData(ptr, NULL, size, place, ATMI_IN_OUT);
  g_data_map.insert(ptr, data);
#else
  ATLData *data = new ATLData(ptr, NULL, size, place, ATMI_IN_OUT);

  hsa_status_t err = hsa_amd_pointer_info_set_userdata(ptr, data);
  ErrorCheck(Setting pointer info with user data, err);
#endif
  if (place.dev_type == ATMI_DEVTYPE_CPU) allow_access_to_all_gpu_agents(ptr);
  // TODO(ashwinma): what if one GPU wants to access another GPU?
}

atmi_status_t Runtime::Malloc(void **ptr, size_t size, atmi_mem_place_t place) {
  atmi_status_t ret = ATMI_STATUS_SUCCESS;
  hsa_amd_memory_pool_t pool = get_memory_pool_by_mem_place(place);
  hsa_status_t err = hsa_amd_memory_pool_allocate(pool, size, 0, ptr);
  ErrorCheck(atmi_malloc, err);
  DEBUG_PRINT("Malloced [%s %d] %p\n",
              place.dev_type == ATMI_DEVTYPE_CPU ? "CPU" : "GPU", place.dev_id,
              *ptr);
  if (err != HSA_STATUS_SUCCESS) ret = ATMI_STATUS_ERROR;

  register_allocation(*ptr, size, place);

  return ret;
}

atmi_status_t Runtime::Memfree(void *ptr) {
  atmi_status_t ret = ATMI_STATUS_SUCCESS;
  hsa_status_t err;
#ifndef USE_ROCR_PTR_INFO
  ATLData *data = g_data_map.find(ptr);
#else
  hsa_amd_pointer_info_t ptr_info;
  ptr_info.size = sizeof(hsa_amd_pointer_info_t);
  err = hsa_amd_pointer_info(reinterpret_cast<void *>(ptr), &ptr_info,
                             NULL,  /* alloc fn ptr */
                             NULL,  /* num_agents_accessible */
                             NULL); /* accessible agents */
  ErrorCheck(Checking pointer info, err);

  ATLData *data = reinterpret_cast<ATLData *>(ptr_info.userData);
#endif
  if (!data)
    ErrorCheck(Checking pointer info userData,
               HSA_STATUS_ERROR_INVALID_ALLOCATION);

#ifndef USE_ROCR_PTR_INFO
  g_data_map.remove(ptr);
#else
// is there a way to unset a userdata with AMD pointer before deleting 'data'?
#endif
  delete data;

  err = hsa_amd_memory_pool_free(ptr);
  ErrorCheck(atmi_free, err);
  DEBUG_PRINT("Freed %p\n", ptr);

  if (err != HSA_STATUS_SUCCESS || !data) ret = ATMI_STATUS_ERROR;
  return ret;
}

DataTaskImpl::DataTaskImpl(atmi_cparm_t *lparm, void *dest, const void *src,
                           const size_t size)
    : TaskImpl(),
      data_dest_ptr_(dest),
      data_src_ptr_(const_cast<void *>(src)),
      data_size_(size) {
  lock(&mutex_all_tasks_);
  AllTasks.push_back(this);
  atmi_task_handle_t new_id;
  set_task_handle_ID(&new_id, AllTasks.size() - 1);
  unlock(&mutex_all_tasks_);
  id_ = new_id;

  taskgroup_ = lparm->group;
  taskgroup_obj_ = getTaskgroupImpl(taskgroup_);

  profilable_ = lparm->profilable;
  groupable_ = lparm->groupable;
  atmi_task_ = lparm->task_info;

  // FIXME: assign the memory scope differently if it makes sense to have an
  // API for doing data copies in non-system scope
  // acquire_scope_ = ATMI_FENCE_SCOPE_SYSTEM;
  // release_scopei_ = ATMI_FENCE_SCOPE_SYSTEM;

  // TODO(ashwinma): performance fix if there are more CPU agents to improve
  // locality
  place_ = ATMI_PLACE_GPU(0, 0);

  predecessors_.resize(lparm->num_required);
  for (int idx = 0; idx < lparm->num_required; idx++) {
    TaskImpl *pred_task = getTaskImpl(lparm->requires[idx]);
    assert(pred_task != NULL);
    predecessors_[idx] = pred_task;
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

atmi_task_handle_t Runtime::MemcpyAsync(atmi_cparm_t *lparm, void *dest,
                                        const void *src, size_t size) {
  // TaskImpl *task = get_new_task();
  DataTaskImpl *task = new DataTaskImpl(lparm, dest, src, size);
  atl_dep_sync_t dep_sync_type =
      (atl_dep_sync_t)core::Runtime::getInstance().getDepSyncType();
  if (dep_sync_type == ATL_SYNC_BARRIER_PKT) {
    lock(&(task->taskgroup_obj_->group_mutex_));
    task->taskgroup_obj_->created_tasks_.push_back(task);
    unlock(&(task->taskgroup_obj_->group_mutex_));
  }
  task->tryDispatch(NULL);

  return task->id_;
}

void DataTaskImpl::acquireAqlPacket() {
  // get signal for SDMA and increment it accordingly
  hsa_status_t err;
  void *src = data_src_ptr_;
  void *dest = data_dest_ptr_;
  size_t size = data_size_;
#ifndef USE_ROCR_PTR_INFO
  ATLData *volatile src_data = g_data_map.find(src);
  ATLData *volatile dest_data = g_data_map.find(dest);
#else
  hsa_amd_pointer_info_t src_ptr_info;
  hsa_amd_pointer_info_t dest_ptr_info;
  src_ptr_info.size = sizeof(hsa_amd_pointer_info_t);
  dest_ptr_info.size = sizeof(hsa_amd_pointer_info_t);
  err = hsa_amd_pointer_info(reinterpret_cast<void *>(src), &src_ptr_info,
                             NULL,  /* alloc fn ptr */
                             NULL,  /* num_agents_accessible */
                             NULL); /* accessible agents */
  ErrorCheck(Checking src pointer info, err);
  err = hsa_amd_pointer_info(reinterpret_cast<void *>(dest), &dest_ptr_info,
                             NULL,  /* alloc fn ptr */
                             NULL,  /* num_agents_accessible */
                             NULL); /* accessible agents */
  ErrorCheck(Checking dest pointer info, err);
  ATLData *volatile src_data =
      reinterpret_cast<ATLData *>(src_ptr_info.userData);
  ATLData *volatile dest_data =
      reinterpret_cast<ATLData *>(dest_ptr_info.userData);
#endif
  bool is_src_host =
      (!src_data || src_data->place().dev_type == ATMI_DEVTYPE_CPU);
  bool is_dest_host =
      (!dest_data || dest_data->place().dev_type == ATMI_DEVTYPE_CPU);
  void *temp_host_ptr;
  const void *src_ptr = src;
  void *dest_ptr = dest;
  volatile Direction type;
  if (is_src_host && is_dest_host) {
    type = Direction::ATMI_H2H;
  } else if (src_data && !dest_data) {
    type = Direction::ATMI_D2H;
  } else if (!src_data && dest_data) {
    type = Direction::ATMI_H2D;
  } else {
    type = Direction::ATMI_D2D;
  }

  if (type == Direction::ATMI_H2D || type == Direction::ATMI_D2H)
    hsa_signal_add_acq_rel(signal_, 2);
  else
    hsa_signal_add_acq_rel(signal_, 1);
}

atmi_status_t DataTaskImpl::dispatch() {
  atmi_status_t ret;
  hsa_status_t err;

  void *dest = data_dest_ptr_;
  const void *src = data_src_ptr_;
  const size_t size = data_size_;

  TaskgroupImpl *taskgroup_obj = taskgroup_obj_;
  atl_dep_sync_t dep_sync_type =
      (atl_dep_sync_t)core::Runtime::getInstance().getDepSyncType();
  std::vector<hsa_signal_t> dep_signals;
  int val = 0;
  DEBUG_PRINT("(");
  for (auto pred_task : and_predecessors_) {
    dep_signals.push_back(pred_task->signal_);
    if (pred_task->state_ < ATMI_DISPATCHED) val++;
    assert(pred_task->state_ >= ATMI_DISPATCHED);
    DEBUG_PRINT("%lu ", pred_task->id_);
  }
  DEBUG_PRINT(")\n");
  if (val > 0)
    DEBUG_PRINT("Task[%lu] has %d not-dispatched predecessor tasks\n", id_,
                val);

#ifndef USE_ROCR_PTR_INFO
  ATLData *volatile src_data = g_data_map.find(src);
  ATLData *volatile dest_data = g_data_map.find(dest);
#else
  hsa_amd_pointer_info_t src_ptr_info;
  hsa_amd_pointer_info_t dest_ptr_info;
  src_ptr_info.size = sizeof(hsa_amd_pointer_info_t);
  dest_ptr_info.size = sizeof(hsa_amd_pointer_info_t);
  err = hsa_amd_pointer_info(reinterpret_cast<void *>(src), &src_ptr_info,
                             NULL,  /* alloc fn ptr */
                             NULL,  /* num_agents_accessible */
                             NULL); /* accessible agents */
  ErrorCheck(Checking src pointer info, err);
  err = hsa_amd_pointer_info(reinterpret_cast<void *>(dest), &dest_ptr_info,
                             NULL,  /* alloc fn ptr */
                             NULL,  /* num_agents_accessible */
                             NULL); /* accessible agents */
  ErrorCheck(Checking dest pointer info, err);
  ATLData *volatile src_data =
      reinterpret_cast<ATLData *>(src_ptr_info.userData);
  ATLData *volatile dest_data =
      reinterpret_cast<ATLData *>(dest_ptr_info.userData);
#endif
  bool is_src_host =
      (!src_data || src_data->place().dev_type == ATMI_DEVTYPE_CPU);
  bool is_dest_host =
      (!dest_data || dest_data->place().dev_type == ATMI_DEVTYPE_CPU);
  atmi_mem_place_t cpu = ATMI_MEM_PLACE_CPU_MEM(0, 0, 0);
  hsa_agent_t cpu_agent = get_mem_agent(cpu);
  hsa_agent_t src_agent;
  hsa_agent_t dest_agent;
  void *temp_host_ptr;
  const void *src_ptr = src;
  void *dest_ptr = dest;
  volatile Direction type;
  if (is_src_host && is_dest_host) {
    type = Direction::ATMI_H2H;
    src_agent = cpu_agent;
    dest_agent = cpu_agent;
    src_ptr = src;
    dest_ptr = dest;
  } else if (src_data && !dest_data) {
    type = Direction::ATMI_D2H;
    src_agent = get_mem_agent(src_data->place());
    dest_agent = src_agent;
    src_ptr = src;
    dest_ptr = dest;
  } else if (!src_data && dest_data) {
    type = Direction::ATMI_H2D;
    dest_agent = get_mem_agent(dest_data->place());
    src_agent = dest_agent;
    src_ptr = src;
    dest_ptr = dest;
  } else {
    type = Direction::ATMI_D2D;
    src_agent = get_mem_agent(src_data->place());
    dest_agent = get_mem_agent(dest_data->place());
    src_ptr = src;
    dest_ptr = dest;
  }
  DEBUG_PRINT("Memcpy source agent: %lu\n", src_agent.handle);
  DEBUG_PRINT("Memcpy dest agent: %lu\n", dest_agent.handle);

  if (type == Direction::ATMI_H2D || type == Direction::ATMI_D2H) {
    if (groupable_ == ATMI_TRUE) {
      lock(&(taskgroup_obj->group_mutex_));
      // barrier pkt already sets the signal values when the signal resource
      // is available
      taskgroup_obj->running_groupable_tasks_.push_back(this);
      unlock(&(taskgroup_obj->group_mutex_));
    }
    // For malloc'ed buffers, additional atmi_malloc/memcpy/free
    // steps are needed. So, fire and forget a copy thread with
    // signal count = 2 (one for actual host-device copy and another
    // for H2H copy to setup the device copy.
    std::thread(
        [](void *dst, const void *src, size_t size, hsa_agent_t agent,
           Direction type, atmi_mem_place_t cpu, hsa_signal_t signal,
           std::vector<hsa_signal_t> dep_signals, TaskImpl *task) {
          atmi_status_t ret;
          hsa_status_t err;
          atl_dep_sync_t dep_sync_type =
              (atl_dep_sync_t)core::Runtime::getInstance().getDepSyncType();
          void *temp_host_ptr;
          const void *src_ptr = src;
          void *dest_ptr = dst;
          ret = atmi_malloc(&temp_host_ptr, size, cpu);
          if (type == Direction::ATMI_H2D) {
            memcpy(temp_host_ptr, src, size);
            src_ptr = (const void *)temp_host_ptr;
            dest_ptr = dst;
          } else {
            src_ptr = src;
            dest_ptr = temp_host_ptr;
          }

          if (dep_sync_type == ATL_SYNC_BARRIER_PKT && !dep_signals.empty()) {
            DEBUG_PRINT("SDMA-host for %p (%lu) with %lu dependencies\n", task,
                        task->id_, dep_signals.size());
            err = hsa_amd_memory_async_copy(dest_ptr, agent, src_ptr, agent,
                                            size, dep_signals.size(),
                                            &(dep_signals[0]), signal);
            ErrorCheck(Copy async between memory pools, err);
          } else {
            DEBUG_PRINT("SDMA-host for %p (%lu)\n", task, task->id_);
            err = hsa_amd_memory_async_copy(dest_ptr, agent, src_ptr, agent,
                                            size, 0, NULL, signal);
            ErrorCheck(Copy async between memory pools, err);
          }
          task->set_state(ATMI_DISPATCHED);
          hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_EQ, 1,
                                  UINT64_MAX, ATMI_WAIT_STATE);
          // cleanup for D2H and H2D
          if (type == Direction::ATMI_D2H) {
            memcpy(dst, temp_host_ptr, size);
          }
          atmi_free(temp_host_ptr);
          hsa_signal_subtract_acq_rel(signal, 1);
        },
        dest, src, size, src_agent, type, cpu, signal_, dep_signals, this)
        .detach();
  } else {
    if (groupable_ == ATMI_TRUE) {
      lock(&(taskgroup_obj_->group_mutex_));
      // barrier pkt already sets the signal values when the signal resource
      // is available
      taskgroup_obj_->running_groupable_tasks_.push_back(this);
      unlock(&(taskgroup_obj_->group_mutex_));
    }

    // set task state to dispatched; then dispatch
    set_state(ATMI_DISPATCHED);

    if (dep_sync_type == ATL_SYNC_BARRIER_PKT && !dep_signals.empty()) {
      DEBUG_PRINT("SDMA for %p (%lu) with %lu dependencies\n", this, id_,
                  dep_signals.size());
      err = hsa_amd_memory_async_copy(dest_ptr, dest_agent, src_ptr, src_agent,
                                      size, dep_signals.size(),
                                      &(dep_signals[0]), signal_);
      ErrorCheck(Copy async between memory pools, err);
    } else {
      DEBUG_PRINT("SDMA for %p (%lu)\n", this, id_);
      err = hsa_amd_memory_async_copy(dest_ptr, dest_agent, src_ptr, src_agent,
                                      size, 0, NULL, signal_);
      ErrorCheck(Copy async between memory pools, err);
    }
  }
  return ATMI_STATUS_SUCCESS;
}

atmi_status_t Runtime::Memcpy(void *dest, const void *src, size_t size) {
  atmi_status_t ret;
  hsa_status_t err;

#ifndef USE_ROCR_PTR_INFO
  ATLData *volatile src_data = g_data_map.find(src);
  ATLData *volatile dest_data = g_data_map.find(dest);
#else
  hsa_amd_pointer_info_t src_ptr_info;
  hsa_amd_pointer_info_t dest_ptr_info;
  src_ptr_info.size = sizeof(hsa_amd_pointer_info_t);
  dest_ptr_info.size = sizeof(hsa_amd_pointer_info_t);
  err = hsa_amd_pointer_info(reinterpret_cast<void *>(src), &src_ptr_info,
                             NULL,  /* alloc fn ptr */
                             NULL,  /* num_agents_accessible */
                             NULL); /* accessible agents */
  ErrorCheck(Checking src pointer info, err);
  err = hsa_amd_pointer_info(reinterpret_cast<void *>(dest), &dest_ptr_info,
                             NULL,  /* alloc fn ptr */
                             NULL,  /* num_agents_accessible */
                             NULL); /* accessible agents */
  ErrorCheck(Checking dest pointer info, err);
  ATLData *volatile src_data =
      reinterpret_cast<ATLData *>(src_ptr_info.userData);
  ATLData *volatile dest_data =
      reinterpret_cast<ATLData *>(dest_ptr_info.userData);
#endif
  atmi_mem_place_t cpu = ATMI_MEM_PLACE_CPU_MEM(0, 0, 0);
  hsa_agent_t cpu_agent = get_mem_agent(cpu);
  hsa_agent_t src_agent;
  hsa_agent_t dest_agent;
  void *temp_host_ptr;
  const void *src_ptr = src;
  void *dest_ptr = dest;
  volatile Direction type;
  if (src_data && !dest_data) {
    type = Direction::ATMI_D2H;
    src_agent = get_mem_agent(src_data->place());
    dest_agent = src_agent;
    // dest_agent = cpu_agent; // FIXME: can the two agents be the GPU agent
    // itself?
    ret = atmi_malloc(&temp_host_ptr, size, cpu);
    // err = hsa_amd_agents_allow_access(1, &src_agent, NULL, temp_host_ptr);
    // ErrorCheck(Allow access to ptr, err);
    src_ptr = src;
    dest_ptr = temp_host_ptr;
  } else if (!src_data && dest_data) {
    type = Direction::ATMI_H2D;
    dest_agent = get_mem_agent(dest_data->place());
    // src_agent = cpu_agent; // FIXME: can the two agents be the GPU agent
    // itself?
    src_agent = dest_agent;
    ret = atmi_malloc(&temp_host_ptr, size, cpu);
    memcpy(temp_host_ptr, src, size);
    // FIXME: ideally lock would be the better approach, but we need to try to
    // understand why the h2d copy segfaults if we dont have the below lines
    // err = hsa_amd_agents_allow_access(1, &dest_agent, NULL, temp_host_ptr);
    // ErrorCheck(Allow access to ptr, err);
    src_ptr = (const void *)temp_host_ptr;
    dest_ptr = dest;
  } else if (!src_data && !dest_data) {
    type = Direction::ATMI_H2H;
    src_agent = cpu_agent;
    dest_agent = cpu_agent;
    src_ptr = src;
    dest_ptr = dest;
  } else {
    type = Direction::ATMI_D2D;
    src_agent = get_mem_agent(src_data->place());
    dest_agent = get_mem_agent(dest_data->place());
    src_ptr = src;
    dest_ptr = dest;
  }
  DEBUG_PRINT("Memcpy source agent: %lu\n", src_agent.handle);
  DEBUG_PRINT("Memcpy dest agent: %lu\n", dest_agent.handle);
  hsa_signal_store_release(IdentityCopySignal, 1);
  // hsa_signal_add_acq_rel(IdentityCopySignal, 1);
  err = hsa_amd_memory_async_copy(dest_ptr, dest_agent, src_ptr, src_agent,
                                  size, 0, NULL, IdentityCopySignal);
  ErrorCheck(Copy async between memory pools, err);
  hsa_signal_wait_acquire(IdentityCopySignal, HSA_SIGNAL_CONDITION_EQ, 0,
                          UINT64_MAX, ATMI_WAIT_STATE);

  // cleanup for D2H and H2D
  if (type == Direction::ATMI_D2H) {
    memcpy(dest, temp_host_ptr, size);
    ret = atmi_free(temp_host_ptr);
  } else if (type == Direction::ATMI_H2D) {
    ret = atmi_free(temp_host_ptr);
  }
  if (err != HSA_STATUS_SUCCESS || ret != ATMI_STATUS_SUCCESS)
    ret = ATMI_STATUS_ERROR;
  return ret;
}

}  // namespace core
