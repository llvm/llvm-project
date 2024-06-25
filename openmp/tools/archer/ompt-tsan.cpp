/*
 * ompt-tsan.cpp -- Archer runtime library, TSan annotations for Archer
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <inttypes.h>
#include <iostream>
#include <list>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "omp-tools.h"

// Define attribute that indicates that the fall through from the previous
// case label is intentional and should not be diagnosed by a compiler
//   Code from libcxx/include/__config
// Use a function like macro to imply that it must be followed by a semicolon
#if __cplusplus > 201402L && __has_cpp_attribute(fallthrough)
#define KMP_FALLTHROUGH() [[fallthrough]]
// icc cannot properly tell this attribute is absent so force off
#elif defined(__INTEL_COMPILER)
#define KMP_FALLTHROUGH() ((void)0)
#elif __has_cpp_attribute(clang::fallthrough)
#define KMP_FALLTHROUGH() [[clang::fallthrough]]
#elif __has_attribute(fallthrough) || __GNUC__ >= 7
#define KMP_FALLTHROUGH() __attribute__((__fallthrough__))
#else
#define KMP_FALLTHROUGH() ((void)0)
#endif

static int hasReductionCallback;

namespace {
class ArcherFlags {
public:
#if (LLVM_VERSION) >= 40
  int flush_shadow{0};
#endif
  int print_max_rss{0};
  int verbose{0};
  int enabled{1};
  int report_data_leak{0};
  int ignore_serial{0};
  std::atomic<int> all_memory{0};

  ArcherFlags(const char *env) {
    if (env) {
      std::vector<std::string> tokens;
      std::string token;
      std::string str(env);
      std::istringstream iss(str);
      int tmp_int;
      while (std::getline(iss, token, ' '))
        tokens.push_back(token);

      for (std::vector<std::string>::iterator it = tokens.begin();
           it != tokens.end(); ++it) {
#if (LLVM_VERSION) >= 40
        if (sscanf(it->c_str(), "flush_shadow=%d", &flush_shadow))
          continue;
#endif
        if (sscanf(it->c_str(), "print_max_rss=%d", &print_max_rss))
          continue;
        if (sscanf(it->c_str(), "verbose=%d", &verbose))
          continue;
        if (sscanf(it->c_str(), "report_data_leak=%d", &report_data_leak))
          continue;
        if (sscanf(it->c_str(), "enable=%d", &enabled))
          continue;
        if (sscanf(it->c_str(), "ignore_serial=%d", &ignore_serial))
          continue;
        if (sscanf(it->c_str(), "all_memory=%d", &tmp_int)) {
          all_memory = tmp_int;
          continue;
        }
        std::cerr << "Illegal values for ARCHER_OPTIONS variable: " << token
                  << std::endl;
      }
    }
  }
};

class TsanFlags {
public:
  int ignore_noninstrumented_modules;

  TsanFlags(const char *env) : ignore_noninstrumented_modules(0) {
    if (env) {
      std::vector<std::string> tokens;
      std::string str(env);
      auto end = str.end();
      auto it = str.begin();
      auto is_sep = [](char c) {
        return c == ' ' || c == ',' || c == ':' || c == '\n' || c == '\t' ||
               c == '\r';
      };
      while (it != end) {
        auto next_it = std::find_if(it, end, is_sep);
        tokens.emplace_back(it, next_it);
        it = next_it;
        if (it != end) {
          ++it;
        }
      }

      for (const auto &token : tokens) {
        // we are interested in ignore_noninstrumented_modules to print a
        // warning
        if (sscanf(token.c_str(), "ignore_noninstrumented_modules=%d",
                   &ignore_noninstrumented_modules))
          continue;
      }
    }
  }
};
} // namespace

#if (LLVM_VERSION) >= 40
extern "C" {
int __attribute__((weak)) __archer_get_omp_status();
void __attribute__((weak)) __tsan_flush_memory() {}
}
#endif
static ArcherFlags *archer_flags;

#ifndef TsanHappensBefore

template <typename... Args> static void __ompt_tsan_func(Args...) {}

#define DECLARE_TSAN_FUNCTION(name, ...)                                       \
  static void (*name)(__VA_ARGS__) = __ompt_tsan_func<__VA_ARGS__>;

// Thread Sanitizer is a tool that finds races in code.
// See http://code.google.com/p/data-race-test/wiki/DynamicAnnotations .
// tsan detects these exact functions by name.
extern "C" {
DECLARE_TSAN_FUNCTION(AnnotateHappensAfter, const char *, int,
                      const volatile void *)
DECLARE_TSAN_FUNCTION(AnnotateHappensBefore, const char *, int,
                      const volatile void *)
DECLARE_TSAN_FUNCTION(AnnotateIgnoreWritesBegin, const char *, int)
DECLARE_TSAN_FUNCTION(AnnotateIgnoreWritesEnd, const char *, int)
DECLARE_TSAN_FUNCTION(AnnotateNewMemory, const char *, int,
                      const volatile void *, size_t)
DECLARE_TSAN_FUNCTION(__tsan_func_entry, const void *)
DECLARE_TSAN_FUNCTION(__tsan_func_exit)

// RunningOnValgrind is used to detect absence of TSan and must intentionally be a nullptr.
static int (*RunningOnValgrind)(void);
}

// This marker is used to define a happens-before arc. The race detector will
// infer an arc from the begin to the end when they share the same pointer
// argument.
#define TsanHappensBefore(cv) AnnotateHappensBefore(__FILE__, __LINE__, cv)

// This marker defines the destination of a happens-before arc.
#define TsanHappensAfter(cv) AnnotateHappensAfter(__FILE__, __LINE__, cv)

// Ignore any races on writes between here and the next TsanIgnoreWritesEnd.
#define TsanIgnoreWritesBegin() AnnotateIgnoreWritesBegin(__FILE__, __LINE__)

// Resume checking for racy writes.
#define TsanIgnoreWritesEnd() AnnotateIgnoreWritesEnd(__FILE__, __LINE__)

// We don't really delete the clock for now
#define TsanDeleteClock(cv)

// newMemory
#define TsanNewMemory(addr, size)                                              \
  AnnotateNewMemory(__FILE__, __LINE__, addr, size)
#define TsanFreeMemory(addr, size)                                             \
  AnnotateNewMemory(__FILE__, __LINE__, addr, size)
#endif

// Function entry/exit
#define TsanFuncEntry(pc) __tsan_func_entry(pc)
#define TsanFuncExit() __tsan_func_exit()

/// Required OMPT inquiry functions.
static ompt_get_parallel_info_t ompt_get_parallel_info;
static ompt_get_thread_data_t ompt_get_thread_data;

typedef char ompt_tsan_clockid;

static uint64_t my_next_id() {
  static uint64_t ID = 0;
  uint64_t ret = __sync_fetch_and_add(&ID, 1);
  return ret;
}

static int pagesize{0};

// Data structure to provide a threadsafe pool of reusable objects.
// DataPool<Type of objects>
namespace {
template <typename T> struct DataPool final {
  static __thread DataPool<T> *ThreadDataPool;
  std::mutex DPMutex{};

  // store unused objects
  std::vector<T *> DataPointer{};
  std::vector<T *> RemoteDataPointer{};

  // store all allocated memory to finally release
  std::list<void *> memory;

  // count remotely returned data (RemoteDataPointer.size())
  std::atomic<int> remote{0};

  // totally allocated data objects in pool
  int total{0};
#ifdef DEBUG_DATA
  int remoteReturn{0};
  int localReturn{0};

  int getRemote() { return remoteReturn + remote; }
  int getLocal() { return localReturn; }
#endif
  int getTotal() { return total; }
  int getMissing() {
    return total - DataPointer.size() - RemoteDataPointer.size();
  }

  // fill the pool by allocating a page of memory
  void newDatas() {
    if (remote > 0) {
      const std::lock_guard<std::mutex> lock(DPMutex);
      // DataPointer is empty, so just swap the vectors
      DataPointer.swap(RemoteDataPointer);
      remote = 0;
      return;
    }
    // calculate size of an object including padding to cacheline size
    size_t elemSize = sizeof(T);
    size_t paddedSize = (((elemSize - 1) / 64) + 1) * 64;
    // number of padded elements to allocate
    int ndatas = pagesize / paddedSize;
    char *datas = (char *)malloc(ndatas * paddedSize);
    memory.push_back(datas);
    for (int i = 0; i < ndatas; i++) {
      DataPointer.push_back(new (datas + i * paddedSize) T(this));
    }
    total += ndatas;
  }

  // get data from the pool
  T *getData() {
    T *ret;
    if (DataPointer.empty())
      newDatas();
    ret = DataPointer.back();
    DataPointer.pop_back();
    return ret;
  }

  // accesses to the thread-local datapool don't need locks
  void returnOwnData(T *data) {
    DataPointer.emplace_back(data);
#ifdef DEBUG_DATA
    localReturn++;
#endif
  }

  // returning to a remote datapool using lock
  void returnData(T *data) {
    const std::lock_guard<std::mutex> lock(DPMutex);
    RemoteDataPointer.emplace_back(data);
    remote++;
#ifdef DEBUG_DATA
    remoteReturn++;
#endif
  }

  ~DataPool() {
    // we assume all memory is returned when the thread finished / destructor is
    // called
    if (archer_flags->report_data_leak && getMissing() != 0) {
      printf("ERROR: While freeing DataPool (%s) we are missing %i data "
             "objects.\n",
             __PRETTY_FUNCTION__, getMissing());
      exit(-3);
    }
    for (auto i : DataPointer)
      if (i)
        i->~T();
    for (auto i : RemoteDataPointer)
      if (i)
        i->~T();
    for (auto i : memory)
      if (i)
        free(i);
  }
};

template <typename T> struct DataPoolEntry {
  DataPool<T> *owner;

  static T *New() { return DataPool<T>::ThreadDataPool->getData(); }

  void Delete() {
    static_cast<T *>(this)->Reset();
    if (owner == DataPool<T>::ThreadDataPool)
      owner->returnOwnData(static_cast<T *>(this));
    else
      owner->returnData(static_cast<T *>(this));
  }

  DataPoolEntry(DataPool<T> *dp) : owner(dp) {}
};

struct DependencyData;
typedef DataPool<DependencyData> DependencyDataPool;
template <>
__thread DependencyDataPool *DependencyDataPool::ThreadDataPool = nullptr;

/// Data structure to store additional information for task dependency.
struct DependencyData final : DataPoolEntry<DependencyData> {
  ompt_tsan_clockid in;
  ompt_tsan_clockid out;
  ompt_tsan_clockid inoutset;
  void *GetInPtr() { return &in; }
  void *GetOutPtr() { return &out; }
  void *GetInoutsetPtr() { return &inoutset; }

  void Reset() {}

  static DependencyData *New() { return DataPoolEntry<DependencyData>::New(); }

  DependencyData(DataPool<DependencyData> *dp)
      : DataPoolEntry<DependencyData>(dp) {}
};

struct TaskDependency {
  void *inPtr;
  void *outPtr;
  void *inoutsetPtr;
  ompt_dependence_type_t type;
  TaskDependency(DependencyData *depData, ompt_dependence_type_t type)
      : inPtr(depData->GetInPtr()), outPtr(depData->GetOutPtr()),
        inoutsetPtr(depData->GetInoutsetPtr()), type(type) {}
  void AnnotateBegin() {
    if (type == ompt_dependence_type_out ||
        type == ompt_dependence_type_inout ||
        type == ompt_dependence_type_mutexinoutset) {
      TsanHappensAfter(inPtr);
      TsanHappensAfter(outPtr);
      TsanHappensAfter(inoutsetPtr);
    } else if (type == ompt_dependence_type_in) {
      TsanHappensAfter(outPtr);
      TsanHappensAfter(inoutsetPtr);
    } else if (type == ompt_dependence_type_inoutset) {
      TsanHappensAfter(inPtr);
      TsanHappensAfter(outPtr);
    }
  }
  void AnnotateEnd() {
    if (type == ompt_dependence_type_out ||
        type == ompt_dependence_type_inout ||
        type == ompt_dependence_type_mutexinoutset) {
      TsanHappensBefore(outPtr);
    } else if (type == ompt_dependence_type_in) {
      TsanHappensBefore(inPtr);
    } else if (type == ompt_dependence_type_inoutset) {
      TsanHappensBefore(inoutsetPtr);
    }
  }
};

struct ParallelData;
typedef DataPool<ParallelData> ParallelDataPool;
template <>
__thread ParallelDataPool *ParallelDataPool::ThreadDataPool = nullptr;

/// Data structure to store additional information for parallel regions.
struct ParallelData final : DataPoolEntry<ParallelData> {

  // Parallel fork is just another barrier, use Barrier[1]

  /// Two addresses for relationships with barriers.
  ompt_tsan_clockid Barrier[2];

  const void *codePtr;

  void *GetParallelPtr() { return &(Barrier[1]); }

  void *GetBarrierPtr(unsigned Index) { return &(Barrier[Index]); }

  ParallelData *Init(const void *codeptr) {
    codePtr = codeptr;
    return this;
  }

  void Reset() {}

  static ParallelData *New(const void *codeptr) {
    return DataPoolEntry<ParallelData>::New()->Init(codeptr);
  }

  ParallelData(DataPool<ParallelData> *dp) : DataPoolEntry<ParallelData>(dp) {}
};

static inline ParallelData *ToParallelData(ompt_data_t *parallel_data) {
  return reinterpret_cast<ParallelData *>(parallel_data->ptr);
}

struct Taskgroup;
typedef DataPool<Taskgroup> TaskgroupPool;
template <> __thread TaskgroupPool *TaskgroupPool::ThreadDataPool = nullptr;

/// Data structure to support stacking of taskgroups and allow synchronization.
struct Taskgroup final : DataPoolEntry<Taskgroup> {
  /// Its address is used for relationships of the taskgroup's task set.
  ompt_tsan_clockid Ptr;

  /// Reference to the parent taskgroup.
  Taskgroup *Parent;

  void *GetPtr() { return &Ptr; }

  Taskgroup *Init(Taskgroup *parent) {
    Parent = parent;
    return this;
  }

  void Reset() {}

  static Taskgroup *New(Taskgroup *Parent) {
    return DataPoolEntry<Taskgroup>::New()->Init(Parent);
  }

  Taskgroup(DataPool<Taskgroup> *dp) : DataPoolEntry<Taskgroup>(dp) {}
};

enum ArcherTaskFlag { ArcherTaskFulfilled = 0x00010000 };

struct TaskData;
typedef DataPool<TaskData> TaskDataPool;
template <> __thread TaskDataPool *TaskDataPool::ThreadDataPool = nullptr;

/// Data structure to store additional information for tasks.
struct TaskData final : DataPoolEntry<TaskData> {
  /// Its address is used for relationships of this task.
  ompt_tsan_clockid Task{0};

  /// Child tasks use its address to declare a relationship to a taskwait in
  /// this task.
  ompt_tsan_clockid Taskwait{0};

  /// Child tasks use its address to model omp_all_memory dependencies
  ompt_tsan_clockid AllMemory[2]{0};

  /// Index of which barrier to use next.
  char BarrierIndex{0};

  /// Whether this task is currently executing a barrier.
  bool InBarrier{false};

  /// Whether this task is an included task.
  int TaskType{0};

  /// count execution phase
  int execution{0};

  /// Count how often this structure has been put into child tasks + 1.
  std::atomic_int RefCount{1};

  /// Reference to the parent that created this task.
  TaskData *Parent{nullptr};

  /// Reference to the team of this task.
  ParallelData *Team{nullptr};

  /// Reference to the current taskgroup that this task either belongs to or
  /// that it just created.
  Taskgroup *TaskGroup{nullptr};

  /// Dependency information for this task.
  TaskDependency *Dependencies{nullptr};

  /// Number of dependency entries.
  unsigned DependencyCount{0};

  // The dependency-map stores DependencyData objects representing
  // the dependency variables used on the sibling tasks created from
  // this task
  // We expect a rare need for the dependency-map, so alloc on demand
  std::unordered_map<void *, DependencyData *> *DependencyMap{nullptr};

#ifdef DEBUG
  int freed{0};
#endif

  bool isIncluded() { return TaskType & ompt_task_undeferred; }
  bool isUntied() { return TaskType & ompt_task_untied; }
  bool isFinal() { return TaskType & ompt_task_final; }
  bool isMergable() { return TaskType & ompt_task_mergeable; }
  bool isMerged() { return TaskType & ompt_task_merged; }

  bool isExplicit() { return TaskType & ompt_task_explicit; }
  bool isImplicit() { return TaskType & ompt_task_implicit; }
  bool isInitial() { return TaskType & ompt_task_initial; }
  bool isTarget() { return TaskType & ompt_task_target; }

  bool isFulfilled() { return TaskType & ArcherTaskFulfilled; }
  void setFulfilled() { TaskType |= ArcherTaskFulfilled; }

  void setAllMemoryDep() { AllMemory[0] = 1; }
  bool hasAllMemoryDep() { return AllMemory[0]; }

  void *GetTaskPtr() { return &Task; }

  void *GetTaskwaitPtr() { return &Taskwait; }

  void *GetLastAllMemoryPtr() { return AllMemory; }
  void *GetNextAllMemoryPtr() { return AllMemory + 1; }

  TaskData *Init(TaskData *parent, int taskType) {
    TaskType = taskType;
    Parent = parent;
    Team = Parent->Team;
    BarrierIndex = Parent->BarrierIndex;
    if (Parent != nullptr) {
      Parent->RefCount++;
      // Copy over pointer to taskgroup. This task may set up its own stack
      // but for now belongs to its parent's taskgroup.
      TaskGroup = Parent->TaskGroup;
    }
    return this;
  }

  TaskData *Init(ParallelData *team, int taskType) {
    TaskType = taskType;
    execution = 1;
    Team = team;
    return this;
  }

  void Reset() {
    InBarrier = false;
    TaskType = 0;
    execution = 0;
    BarrierIndex = 0;
    RefCount = 1;
    Parent = nullptr;
    Team = nullptr;
    TaskGroup = nullptr;
    if (DependencyMap) {
      for (auto i : *DependencyMap)
        i.second->Delete();
      delete DependencyMap;
    }
    DependencyMap = nullptr;
    if (Dependencies)
      free(Dependencies);
    Dependencies = nullptr;
    DependencyCount = 0;
#ifdef DEBUG
    freed = 0;
#endif
  }

  static TaskData *New(TaskData *parent, int taskType) {
    return DataPoolEntry<TaskData>::New()->Init(parent, taskType);
  }

  static TaskData *New(ParallelData *team, int taskType) {
    return DataPoolEntry<TaskData>::New()->Init(team, taskType);
  }

  TaskData(DataPool<TaskData> *dp) : DataPoolEntry<TaskData>(dp) {}
};
} // namespace

static inline TaskData *ToTaskData(ompt_data_t *task_data) {
  if (task_data)
    return reinterpret_cast<TaskData *>(task_data->ptr);
  return nullptr;
}

/// Store a mutex for each wait_id to resolve race condition with callbacks.
static std::unordered_map<ompt_wait_id_t, std::mutex> Locks;
static std::mutex LocksMutex;

static void ompt_tsan_thread_begin(ompt_thread_t thread_type,
                                   ompt_data_t *thread_data) {
  ParallelDataPool::ThreadDataPool = new ParallelDataPool;
  TsanNewMemory(ParallelDataPool::ThreadDataPool,
                sizeof(ParallelDataPool::ThreadDataPool));
  TaskgroupPool::ThreadDataPool = new TaskgroupPool;
  TsanNewMemory(TaskgroupPool::ThreadDataPool,
                sizeof(TaskgroupPool::ThreadDataPool));
  TaskDataPool::ThreadDataPool = new TaskDataPool;
  TsanNewMemory(TaskDataPool::ThreadDataPool,
                sizeof(TaskDataPool::ThreadDataPool));
  DependencyDataPool::ThreadDataPool = new DependencyDataPool;
  TsanNewMemory(DependencyDataPool::ThreadDataPool,
                sizeof(DependencyDataPool::ThreadDataPool));
  thread_data->value = my_next_id();
}

static void ompt_tsan_thread_end(ompt_data_t *thread_data) {
  TsanIgnoreWritesBegin();
  delete ParallelDataPool::ThreadDataPool;
  delete TaskgroupPool::ThreadDataPool;
  delete TaskDataPool::ThreadDataPool;
  delete DependencyDataPool::ThreadDataPool;
  TsanIgnoreWritesEnd();
}

/// OMPT event callbacks for handling parallel regions.

static void ompt_tsan_parallel_begin(ompt_data_t *parent_task_data,
                                     const ompt_frame_t *parent_task_frame,
                                     ompt_data_t *parallel_data,
                                     uint32_t requested_team_size, int flag,
                                     const void *codeptr_ra) {
  ParallelData *Data = ParallelData::New(codeptr_ra);
  parallel_data->ptr = Data;

  TsanHappensBefore(Data->GetParallelPtr());
  if (archer_flags->ignore_serial && ToTaskData(parent_task_data)->isInitial())
    TsanIgnoreWritesEnd();
}

static void ompt_tsan_parallel_end(ompt_data_t *parallel_data,
                                   ompt_data_t *task_data, int flag,
                                   const void *codeptr_ra) {
  if (archer_flags->ignore_serial && ToTaskData(task_data)->isInitial())
    TsanIgnoreWritesBegin();
  ParallelData *Data = ToParallelData(parallel_data);
  TsanHappensAfter(Data->GetBarrierPtr(0));
  TsanHappensAfter(Data->GetBarrierPtr(1));

  Data->Delete();

#if (LLVM_VERSION >= 40)
  if (&__archer_get_omp_status) {
    if (__archer_get_omp_status() == 0 && archer_flags->flush_shadow)
      __tsan_flush_memory();
  }
#endif
}

static void ompt_tsan_implicit_task(ompt_scope_endpoint_t endpoint,
                                    ompt_data_t *parallel_data,
                                    ompt_data_t *task_data,
                                    unsigned int team_size,
                                    unsigned int thread_num, int type) {
  switch (endpoint) {
  case ompt_scope_begin:
    if (type & ompt_task_initial) {
      parallel_data->ptr = ParallelData::New(nullptr);
    }
    task_data->ptr = TaskData::New(ToParallelData(parallel_data), type);
    TsanHappensAfter(ToParallelData(parallel_data)->GetParallelPtr());
    TsanFuncEntry(ToParallelData(parallel_data)->codePtr);
    break;
  case ompt_scope_end: {
    TaskData *Data = ToTaskData(task_data);
#ifdef DEBUG
    assert(Data->freed == 0 && "Implicit task end should only be called once!");
    Data->freed = 1;
#endif
    assert(Data->RefCount == 1 &&
           "All tasks should have finished at the implicit barrier!");
    if (type & ompt_task_initial) {
      Data->Team->Delete();
    }
    Data->Delete();
    TsanFuncExit();
    break;
  }
  case ompt_scope_beginend:
    // Should not occur according to OpenMP 5.1
    // Tested in OMPT tests
    break;
  }
}

static void ompt_tsan_sync_region(ompt_sync_region_t kind,
                                  ompt_scope_endpoint_t endpoint,
                                  ompt_data_t *parallel_data,
                                  ompt_data_t *task_data,
                                  const void *codeptr_ra) {
  TaskData *Data = ToTaskData(task_data);
  switch (endpoint) {
  case ompt_scope_begin:
  case ompt_scope_beginend:
    TsanFuncEntry(codeptr_ra);
    switch (kind) {
    case ompt_sync_region_barrier_implementation:
    case ompt_sync_region_barrier_implicit:
    case ompt_sync_region_barrier_explicit:
    case ompt_sync_region_barrier_implicit_parallel:
    case ompt_sync_region_barrier_implicit_workshare:
    case ompt_sync_region_barrier_teams:
    case ompt_sync_region_barrier: {
      char BarrierIndex = Data->BarrierIndex;
      TsanHappensBefore(Data->Team->GetBarrierPtr(BarrierIndex));

      if (hasReductionCallback < ompt_set_always) {
        // We ignore writes inside the barrier. These would either occur during
        // 1. reductions performed by the runtime which are guaranteed to be
        // race-free.
        // 2. execution of another task.
        // For the latter case we will re-enable tracking in task_switch.
        Data->InBarrier = true;
        TsanIgnoreWritesBegin();
      }

      break;
    }

    case ompt_sync_region_taskwait:
      break;

    case ompt_sync_region_taskgroup:
      Data->TaskGroup = Taskgroup::New(Data->TaskGroup);
      break;

    case ompt_sync_region_reduction:
      // should never be reached
      break;
    }
    if (endpoint == ompt_scope_begin)
      break;
    KMP_FALLTHROUGH();
  case ompt_scope_end:
    TsanFuncExit();
    switch (kind) {
    case ompt_sync_region_barrier_implementation:
    case ompt_sync_region_barrier_implicit:
    case ompt_sync_region_barrier_explicit:
    case ompt_sync_region_barrier_implicit_parallel:
    case ompt_sync_region_barrier_implicit_workshare:
    case ompt_sync_region_barrier_teams:
    case ompt_sync_region_barrier: {
      if (hasReductionCallback < ompt_set_always) {
        // We want to track writes after the barrier again.
        Data->InBarrier = false;
        TsanIgnoreWritesEnd();
      }

      char BarrierIndex = Data->BarrierIndex;
      // Barrier will end after it has been entered by all threads.
      if (parallel_data)
        TsanHappensAfter(Data->Team->GetBarrierPtr(BarrierIndex));

      // It is not guaranteed that all threads have exited this barrier before
      // we enter the next one. So we will use a different address.
      // We are however guaranteed that this current barrier is finished
      // by the time we exit the next one. So we can then reuse the first
      // address.
      Data->BarrierIndex = (BarrierIndex + 1) % 2;
      break;
    }

    case ompt_sync_region_taskwait: {
      if (Data->execution > 1)
        TsanHappensAfter(Data->GetTaskwaitPtr());
      break;
    }

    case ompt_sync_region_taskgroup: {
      assert(Data->TaskGroup != nullptr &&
             "Should have at least one taskgroup!");

      TsanHappensAfter(Data->TaskGroup->GetPtr());

      // Delete this allocated taskgroup, all descendent task are finished by
      // now.
      Taskgroup *Parent = Data->TaskGroup->Parent;
      Data->TaskGroup->Delete();
      Data->TaskGroup = Parent;
      break;
    }

    case ompt_sync_region_reduction:
      // Should not occur according to OpenMP 5.1
      // Tested in OMPT tests
      break;
    }
    break;
  }
}

static void ompt_tsan_reduction(ompt_sync_region_t kind,
                                ompt_scope_endpoint_t endpoint,
                                ompt_data_t *parallel_data,
                                ompt_data_t *task_data,
                                const void *codeptr_ra) {
  switch (endpoint) {
  case ompt_scope_begin:
    switch (kind) {
    case ompt_sync_region_reduction:
      TsanIgnoreWritesBegin();
      break;
    default:
      break;
    }
    break;
  case ompt_scope_end:
    switch (kind) {
    case ompt_sync_region_reduction:
      TsanIgnoreWritesEnd();
      break;
    default:
      break;
    }
    break;
  case ompt_scope_beginend:
    // Should not occur according to OpenMP 5.1
    // Tested in OMPT tests
    // Would have no implications for DR detection
    break;
  }
}

/// OMPT event callbacks for handling tasks.

static void ompt_tsan_task_create(
    ompt_data_t *parent_task_data,    /* id of parent task            */
    const ompt_frame_t *parent_frame, /* frame data for parent task   */
    ompt_data_t *new_task_data,       /* id of created task           */
    int type, int has_dependences,
    const void *codeptr_ra) /* pointer to outlined function */
{
  TaskData *Data;
  assert(new_task_data->ptr == NULL &&
         "Task data should be initialized to NULL");
  if (type & ompt_task_initial) {
    ompt_data_t *parallel_data;
    int team_size = 1;
    ompt_get_parallel_info(0, &parallel_data, &team_size);
    ParallelData *PData = ParallelData::New(nullptr);
    parallel_data->ptr = PData;

    Data = TaskData::New(PData, type);
    new_task_data->ptr = Data;
  } else if (type & ompt_task_undeferred) {
    Data = TaskData::New(ToTaskData(parent_task_data), type);
    new_task_data->ptr = Data;
  } else if (type & ompt_task_explicit || type & ompt_task_target) {
    Data = TaskData::New(ToTaskData(parent_task_data), type);
    new_task_data->ptr = Data;

    // Use the newly created address. We cannot use a single address from the
    // parent because that would declare wrong relationships with other
    // sibling tasks that may be created before this task is started!
    TsanHappensBefore(Data->GetTaskPtr());
    ToTaskData(parent_task_data)->execution++;
  }
}

static void freeTask(TaskData *task) {
  while (task != nullptr && --task->RefCount == 0) {
    TaskData *Parent = task->Parent;
    task->Delete();
    task = Parent;
  }
}

// LastAllMemoryPtr marks the beginning of an all_memory epoch
// NextAllMemoryPtr marks the end of an all_memory epoch
// All tasks with depend begin execution after LastAllMemoryPtr
// and end before NextAllMemoryPtr
static void releaseDependencies(TaskData *task) {
  if (archer_flags->all_memory) {
    if (task->hasAllMemoryDep()) {
      TsanHappensBefore(task->Parent->GetLastAllMemoryPtr());
      TsanHappensBefore(task->Parent->GetNextAllMemoryPtr());
    } else if (task->DependencyCount)
      TsanHappensBefore(task->Parent->GetNextAllMemoryPtr());
  }
  for (unsigned i = 0; i < task->DependencyCount; i++) {
    task->Dependencies[i].AnnotateEnd();
  }
}

static void acquireDependencies(TaskData *task) {
  if (archer_flags->all_memory) {
    if (task->hasAllMemoryDep())
      TsanHappensAfter(task->Parent->GetNextAllMemoryPtr());
    else if (task->DependencyCount)
      TsanHappensAfter(task->Parent->GetLastAllMemoryPtr());
  }
  for (unsigned i = 0; i < task->DependencyCount; i++) {
    task->Dependencies[i].AnnotateBegin();
  }
}

static void completeTask(TaskData *FromTask) {
  if (!FromTask)
    return;
  // Task-end happens after a possible omp_fulfill_event call
  if (FromTask->isFulfilled())
    TsanHappensAfter(FromTask->GetTaskPtr());
  // Included tasks are executed sequentially, no need to track
  // synchronization
  if (!FromTask->isIncluded()) {
    // Task will finish before a barrier in the surrounding parallel region
    // ...
    ParallelData *PData = FromTask->Team;
    TsanHappensBefore(PData->GetBarrierPtr(FromTask->BarrierIndex));

    // ... and before an eventual taskwait by the parent thread.
    TsanHappensBefore(FromTask->Parent->GetTaskwaitPtr());

    if (FromTask->TaskGroup != nullptr) {
      // This task is part of a taskgroup, so it will finish before the
      // corresponding taskgroup_end.
      TsanHappensBefore(FromTask->TaskGroup->GetPtr());
    }
  }
  // release dependencies
  releaseDependencies(FromTask);
}

static void suspendTask(TaskData *FromTask) {
  if (!FromTask)
    return;
  // Task may be resumed at a later point in time.
  TsanHappensBefore(FromTask->GetTaskPtr());
}

static void switchTasks(TaskData *FromTask, TaskData *ToTask) {
  // Legacy handling for missing reduction callback
  if (hasReductionCallback < ompt_set_always) {
    if (FromTask && FromTask->InBarrier) {
      // We want to ignore writes in the runtime code during barriers,
      // but not when executing tasks with user code!
      TsanIgnoreWritesEnd();
    }
    if (ToTask && ToTask->InBarrier) {
      // We want to ignore writes in the runtime code during barriers,
      // but not when executing tasks with user code!
      TsanIgnoreWritesBegin();
    }
  }
  //// Not yet used
  //  if (FromTask)
  //    FromTask->deactivate();
  //  if (ToTask)
  //    ToTask->activate();
}

static void endTask(TaskData *FromTask) {
  if (!FromTask)
    return;
}

static void startTask(TaskData *ToTask) {
  if (!ToTask)
    return;
  // Handle dependencies on first execution of the task
  if (ToTask->execution == 0) {
    ToTask->execution++;
    acquireDependencies(ToTask);
  }
  // 1. Task will begin execution after it has been created.
  // 2. Task will resume after it has been switched away.
  TsanHappensAfter(ToTask->GetTaskPtr());
}

static void ompt_tsan_task_schedule(ompt_data_t *first_task_data,
                                    ompt_task_status_t prior_task_status,
                                    ompt_data_t *second_task_data) {

  //
  //  The necessary action depends on prior_task_status:
  //
  //    ompt_task_early_fulfill = 5,
  //     -> ignored
  //
  //    ompt_task_late_fulfill  = 6,
  //     -> first completed, first freed, second ignored
  //
  //    ompt_task_complete      = 1,
  //    ompt_task_cancel        = 3,
  //     -> first completed, first freed, second starts
  //
  //    ompt_taskwait_complete = 8,
  //     -> first starts, first completes, first freed, second ignored
  //
  //    ompt_task_detach        = 4,
  //    ompt_task_yield         = 2,
  //    ompt_task_switch        = 7
  //     -> first suspended, second starts
  //

  TaskData *FromTask = ToTaskData(first_task_data);
  TaskData *ToTask = ToTaskData(second_task_data);

  switch (prior_task_status) {
  case ompt_task_early_fulfill:
    TsanHappensBefore(FromTask->GetTaskPtr());
    FromTask->setFulfilled();
    return;
  case ompt_task_late_fulfill:
    TsanHappensAfter(FromTask->GetTaskPtr());
    completeTask(FromTask);
    freeTask(FromTask);
    return;
  case ompt_taskwait_complete:
    acquireDependencies(FromTask);
    freeTask(FromTask);
    return;
  case ompt_task_complete:
    completeTask(FromTask);
    endTask(FromTask);
    switchTasks(FromTask, ToTask);
    freeTask(FromTask);
    return;
  case ompt_task_cancel:
    completeTask(FromTask);
    endTask(FromTask);
    switchTasks(FromTask, ToTask);
    freeTask(FromTask);
    startTask(ToTask);
    return;
  case ompt_task_detach:
    endTask(FromTask);
    suspendTask(FromTask);
    switchTasks(FromTask, ToTask);
    startTask(ToTask);
    return;
  case ompt_task_yield:
    suspendTask(FromTask);
    switchTasks(FromTask, ToTask);
    startTask(ToTask);
    return;
  case ompt_task_switch:
    suspendTask(FromTask);
    switchTasks(FromTask, ToTask);
    startTask(ToTask);
    return;
  }
}

static void ompt_tsan_dependences(ompt_data_t *task_data,
                                  const ompt_dependence_t *deps, int ndeps) {
  if (ndeps > 0) {
    // Copy the data to use it in task_switch and task_end.
    TaskData *Data = ToTaskData(task_data);
    if (!Data->Parent) {
      // Return since doacross dependences are not supported yet.
      return;
    }
    if (!Data->Parent->DependencyMap)
      Data->Parent->DependencyMap =
          new std::unordered_map<void *, DependencyData *>();
    Data->Dependencies =
        (TaskDependency *)malloc(sizeof(TaskDependency) * ndeps);
    Data->DependencyCount = ndeps;
    for (int i = 0, d = 0; i < ndeps; i++, d++) {
      if (deps[i].dependence_type == ompt_dependence_type_out_all_memory ||
          deps[i].dependence_type == ompt_dependence_type_inout_all_memory) {
        Data->setAllMemoryDep();
        Data->DependencyCount--;
        if (!archer_flags->all_memory) {
          printf("The application uses omp_all_memory, but Archer was\n"
                 "started to not consider omp_all_memory. This can lead\n"
                 "to false data race alerts.\n"
                 "Include all_memory=1 in ARCHER_OPTIONS to consider\n"
                 "omp_all_memory from the beginning.\n");
          archer_flags->all_memory = 1;
        }
        d--;
        continue;
      }
      auto ret = Data->Parent->DependencyMap->insert(
          std::make_pair(deps[i].variable.ptr, nullptr));
      if (ret.second) {
        ret.first->second = DependencyData::New();
      }
      new ((void *)(Data->Dependencies + d))
          TaskDependency(ret.first->second, deps[i].dependence_type);
    }

    // This callback is executed before this task is first started.
    TsanHappensBefore(Data->GetTaskPtr());
  }
}

/// OMPT event callbacks for handling locking.
static void ompt_tsan_mutex_acquired(ompt_mutex_t kind, ompt_wait_id_t wait_id,
                                     const void *codeptr_ra) {

  // Acquire our own lock to make sure that
  // 1. the previous release has finished.
  // 2. the next acquire doesn't start before we have finished our release.
  LocksMutex.lock();
  std::mutex &Lock = Locks[wait_id];
  LocksMutex.unlock();

  Lock.lock();
  TsanHappensAfter(&Lock);
}

static void ompt_tsan_mutex_released(ompt_mutex_t kind, ompt_wait_id_t wait_id,
                                     const void *codeptr_ra) {
  LocksMutex.lock();
  std::mutex &Lock = Locks[wait_id];
  LocksMutex.unlock();
  TsanHappensBefore(&Lock);

  Lock.unlock();
}

// callback , signature , variable to store result , required support level
#define SET_OPTIONAL_CALLBACK_T(event, type, result, level)                    \
  do {                                                                         \
    ompt_callback_##type##_t tsan_##event = &ompt_tsan_##event;                \
    result = ompt_set_callback(ompt_callback_##event,                          \
                               (ompt_callback_t)tsan_##event);                 \
    if (result < level)                                                        \
      printf("Registered callback '" #event "' is not supported at " #level    \
             " (%i)\n",                                                        \
             result);                                                          \
  } while (0)

#define SET_CALLBACK_T(event, type)                                            \
  do {                                                                         \
    int res;                                                                   \
    SET_OPTIONAL_CALLBACK_T(event, type, res, ompt_set_always);                \
  } while (0)

#define SET_CALLBACK(event) SET_CALLBACK_T(event, event)

#define findTsanFunction(f, fSig)                                              \
  do {                                                                         \
    void *fp = dlsym(RTLD_DEFAULT, #f);                                        \
    if (fp)                                                                    \
      f = fSig fp;                                                             \
    else                                                                       \
      printf("Unable to find TSan function " #f ".\n");                        \
  } while (0)

#define findTsanFunctionSilent(f, fSig) f = fSig dlsym(RTLD_DEFAULT, #f)

static int ompt_tsan_initialize(ompt_function_lookup_t lookup, int device_num,
                                ompt_data_t *tool_data) {
  const char *options = getenv("TSAN_OPTIONS");
  TsanFlags tsan_flags(options);

  ompt_set_callback_t ompt_set_callback =
      (ompt_set_callback_t)lookup("ompt_set_callback");
  if (ompt_set_callback == NULL) {
    std::cerr << "Could not set callback, exiting..." << std::endl;
    std::exit(1);
  }
  ompt_get_parallel_info =
      (ompt_get_parallel_info_t)lookup("ompt_get_parallel_info");
  ompt_get_thread_data = (ompt_get_thread_data_t)lookup("ompt_get_thread_data");

  if (ompt_get_parallel_info == NULL) {
    fprintf(stderr, "Could not get inquiry function 'ompt_get_parallel_info', "
                    "exiting...\n");
    exit(1);
  }

  findTsanFunction(AnnotateHappensAfter,
                   (void (*)(const char *, int, const volatile void *)));
  findTsanFunction(AnnotateHappensBefore,
                   (void (*)(const char *, int, const volatile void *)));
  findTsanFunction(AnnotateIgnoreWritesBegin, (void (*)(const char *, int)));
  findTsanFunction(AnnotateIgnoreWritesEnd, (void (*)(const char *, int)));
  findTsanFunction(
      AnnotateNewMemory,
      (void (*)(const char *, int, const volatile void *, size_t)));
  findTsanFunction(__tsan_func_entry, (void (*)(const void *)));
  findTsanFunction(__tsan_func_exit, (void (*)(void)));

  SET_CALLBACK(thread_begin);
  SET_CALLBACK(thread_end);
  SET_CALLBACK(parallel_begin);
  SET_CALLBACK(implicit_task);
  SET_CALLBACK(sync_region);
  SET_CALLBACK(parallel_end);

  SET_CALLBACK(task_create);
  SET_CALLBACK(task_schedule);
  SET_CALLBACK(dependences);

  SET_CALLBACK_T(mutex_acquired, mutex);
  SET_CALLBACK_T(mutex_released, mutex);
  SET_OPTIONAL_CALLBACK_T(reduction, sync_region, hasReductionCallback,
                          ompt_set_never);

  if (!tsan_flags.ignore_noninstrumented_modules)
    fprintf(stderr,
            "Warning: please export "
            "TSAN_OPTIONS='ignore_noninstrumented_modules=1' "
            "to avoid false positive reports from the OpenMP runtime!\n");
  if (archer_flags->ignore_serial)
    TsanIgnoreWritesBegin();

  return 1; // success
}

static void ompt_tsan_finalize(ompt_data_t *tool_data) {
  if (archer_flags->ignore_serial)
    TsanIgnoreWritesEnd();
  if (archer_flags->print_max_rss) {
    struct rusage end;
    getrusage(RUSAGE_SELF, &end);
    printf("MAX RSS[KBytes] during execution: %ld\n", end.ru_maxrss);
  }

  if (archer_flags)
    delete archer_flags;
}

extern "C" ompt_start_tool_result_t *
ompt_start_tool(unsigned int omp_version, const char *runtime_version) {
  const char *options = getenv("ARCHER_OPTIONS");
  archer_flags = new ArcherFlags(options);
  if (!archer_flags->enabled) {
    if (archer_flags->verbose)
      std::cout << "Archer disabled, stopping operation" << std::endl;
    delete archer_flags;
    return NULL;
  }

  pagesize = getpagesize();

  static ompt_start_tool_result_t ompt_start_tool_result = {
      &ompt_tsan_initialize, &ompt_tsan_finalize, {0}};

  // The OMPT start-up code uses dlopen with RTLD_LAZY. Therefore, we cannot
  // rely on dlopen to fail if TSan is missing, but would get a runtime error
  // for the first TSan call. We use RunningOnValgrind to detect whether
  // an implementation of the Annotation interface is available in the
  // execution or disable the tool (by returning NULL).

  findTsanFunctionSilent(RunningOnValgrind, (int (*)(void)));
  if (!RunningOnValgrind) // if we are not running on TSAN, give a different
                          // tool the chance to be loaded
  {
    if (archer_flags->verbose)
      std::cout << "Archer detected OpenMP application without TSan; "
                   "stopping operation"
                << std::endl;
    delete archer_flags;
    return NULL;
  }

  if (archer_flags->verbose)
    std::cout << "Archer detected OpenMP application with TSan, supplying "
                 "OpenMP synchronization semantics"
              << std::endl;
  return &ompt_start_tool_result;
}
