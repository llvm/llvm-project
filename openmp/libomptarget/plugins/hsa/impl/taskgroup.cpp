#include "taskgroup.h"
#include <cassert>
#include "realtimer.h"
using core::RealTimer;
extern RealTimer TaskWaitTimer;

/* Taskgroup specific globals */
atmi_taskgroup_handle_t ATMI_DEFAULT_TASKGROUP_HANDLE = {0ull};

namespace core {

/* Taskgroup vector to hold the runtime state of the
 * taskgroup and its tasks. Which was the latest
 * device used, latest queue used and also a
 * pool of tasks for synchronization if need be */
std::vector<core::TaskgroupImpl *> AllTaskgroups;

TaskgroupImpl *getTaskgroupImpl(atmi_taskgroup_handle_t t) {
  TaskgroupImpl *taskgroup_obj = NULL;
  lock(&mutex_all_tasks_);
  if (t < AllTaskgroups.size()) {
    taskgroup_obj = AllTaskgroups[t];
  }
  unlock(&mutex_all_tasks_);
  return taskgroup_obj;
}

atmi_status_t Runtime::TaskGroupCreate(atmi_taskgroup_handle_t *group_handle,
                                       bool ordered, atmi_place_t place) {
  atmi_status_t status = ATMI_STATUS_ERROR;
  if (group_handle) {
    TaskgroupImpl *taskgroup_obj = new TaskgroupImpl(ordered, place);
    // add to global taskgroup vector
    lock(&mutex_all_tasks_);
    AllTaskgroups.push_back(taskgroup_obj);
    // the below assert is always true because we insert into the
    // vector but dont delete that slot upon release.
    assert((AllTaskgroups.size() - 1) == taskgroup_obj->id_ &&
           "Taskgroup ID and vec size mismatch");
    *group_handle = taskgroup_obj->id_;
    unlock(&mutex_all_tasks_);
    status = ATMI_STATUS_SUCCESS;
  }
  return status;
}

atmi_status_t Runtime::TaskGroupRelease(atmi_taskgroup_handle_t group_handle) {
  atmi_status_t status = ATMI_STATUS_ERROR;
  TaskgroupImpl *taskgroup_obj = getTaskgroupImpl(group_handle);
  if (taskgroup_obj) {
    lock(&mutex_all_tasks_);
    delete taskgroup_obj;
    AllTaskgroups[group_handle] = NULL;
    unlock(&mutex_all_tasks_);
    status = ATMI_STATUS_SUCCESS;
  }
  return status;
}

void TaskgroupImpl::sync() {
  DEBUG_PRINT("Waiting for %u group tasks to complete\n", task_count_.load());
  // if tasks are groupable
  while (task_count_.load() != 0) {
  }

  if (ordered_ == ATMI_TRUE) {
    // if tasks are ordered, just wait for the last launched/created task in
    // this group
    last_task_->wait();
    lock(&(group_mutex_));
    last_task_ = NULL;
    unlock(&(group_mutex_));
  } else {
    // if tasks are neither groupable or ordered, wait for every task in the
    // taskgroup
    for (auto task : running_default_tasks_) {
      task->wait();
    }
  }
  clearSavedTasks();
}

atmi_status_t TaskgroupImpl::clearSavedTasks() {
  lock(&group_mutex_);
  running_ordered_tasks_.clear();
  running_default_tasks_.clear();
  running_groupable_tasks_.clear();
  unlock(&group_mutex_);
  return ATMI_STATUS_SUCCESS;
}

int TaskgroupImpl::getBestQueueID(atmi_scheduler_t sched) {
  int ret = 0;
  switch (sched) {
    case ATMI_SCHED_NONE:
      ret = __atomic_load_n(&next_best_queue_id_, __ATOMIC_ACQUIRE);
      break;
    case ATMI_SCHED_RR:
      ret = __atomic_fetch_add(&next_best_queue_id_, 1, __ATOMIC_ACQ_REL);
      break;
  }
  return ret;
}

/*
 * do not use the below because they will not work if we want to sort mutexes
atmi_status_t get_taskgroup_mutex(atl_taskgroup_t *taskgroup_obj,
pthread_mutex_t *m) {
    *m = taskgroup_obj->group_mutex;
    return ATMI_STATUS_SUCCESS;
}

atmi_status_t set_taskgroup_mutex(atl_taskgroup_t *taskgroup_obj,
pthread_mutex_t *m) {
    taskgroup_obj->group_mutex = *m;
    return ATMI_STATUS_SUCCESS;
}
*/

// constructor
core::TaskgroupImpl::TaskgroupImpl(bool ordered, atmi_place_t place)
    : ordered_(ordered),
      first_created_tasks_dispatched_(false),
      place_(place),
      next_best_queue_id_(0),
      last_task_(NULL),
      cpu_queue_(NULL),
      gpu_queue_(NULL) {
  static unsigned int taskgroup_id = 0;
  id_ = taskgroup_id++;

  running_groupable_tasks_.clear();
  running_ordered_tasks_.clear();
  running_default_tasks_.clear();
  and_successors_.clear();
  task_count_.store(0);
  callback_started_.clear();

  pthread_mutex_init(&(group_mutex_), NULL);

  // create the group signal with initial value 0; task dispatch is
  // then responsible for incrementing this value before ringing the
  // doorbell.
  hsa_status_t err = hsa_signal_create(0, 0, NULL, &group_signal_);
  ErrorCheck(Taskgroup signal creation, err);
}

// destructor
core::TaskgroupImpl::~TaskgroupImpl() {
  hsa_status_t err = hsa_signal_destroy(group_signal_);
  ErrorCheck(Taskgroup signal destruction, err);

  running_groupable_tasks_.clear();
  running_ordered_tasks_.clear();
  running_default_tasks_.clear();
  and_successors_.clear();
}

}  // namespace core
