/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#ifndef INCLUDE_ATMI_H_
#define INCLUDE_ATMI_H_

#define ATMI_VERSION 0
#define ATMI_RELEASE 7
#define ATMI_PATCH 0
#define ATMI_VRM ((ATMI_VERSION * 65536) + (ATMI_RELEASE * 256) + ATMI_PATCH)
#define ROCM_VERSION_MAJOR 3
#define ROCM_VERSION_MINOR 2

/** \defgroup enumerations Enumerated Types
 * @{
 */

#define ATMI_TRUE 1
#define ATMI_FALSE 0

/**
 * @brief Status codes.
 */
typedef enum atmi_status_t {
  /**
   * The function has been executed successfully.
   */
  ATMI_STATUS_SUCCESS = 0,
  /**
   * A undocumented error has occurred.
   */
  ATMI_STATUS_UNKNOWN = 1,
  /**
   * A generic error has occurred.
   */
  ATMI_STATUS_ERROR = 2,
  /**
   * Too many kernel/task types have been created.
   */
  ATMI_STATUS_KERNELCOUNT_OVERFLOW = 3
} atmi_status_t;

/**
 * @brief Platform Types.
 */
typedef enum {
  /**
   * \deprecated Target Platform is BRIG
   */
  BRIG = 0,
  /**
   * Target Platform is AMD GCN (default)
   */
  AMDGCN = 1,
  /*
   * Target Platform is AMD GCN compiled either from HIP, CL or OpenMP Language
   * frontends.
   * ATMI runtime treats the target platform in the same way irrespective of the
   * high level language used to generate the code object.
   */
  AMDGCN_HIP = 1,
  AMDGCN_CL = 1,
  AMDGCN_OMP = 1,
  /*
   * Target platform is CPU
   */
  X86 = 2,
  /* -- support in the future? --
  PTX
  */
} atmi_platform_type_t;

/**
 * @brief Device Types.
 */
typedef enum atmi_devtype_s {
  ATMI_DEVTYPE_CPU = 0x0001,
  ATMI_DEVTYPE_iGPU = 0x0010,                                // Integrated GPU
  ATMI_DEVTYPE_dGPU = 0x0100,                                // Discrete GPU
  ATMI_DEVTYPE_GPU = ATMI_DEVTYPE_iGPU | ATMI_DEVTYPE_dGPU,  // Any GPU
  ATMI_DEVTYPE_ALL = 0x111  // Union of all device types
} atmi_devtype_t;

/**
 * @brief Memory Access Type.
 */
typedef enum atmi_memtype_s {
  ATMI_MEMTYPE_FINE_GRAINED = 0,
  ATMI_MEMTYPE_COARSE_GRAINED = 1,
  ATMI_MEMTYPE_ANY
} atmi_memtype_t;

/**
 * @brief Task States.
 */
typedef enum atmi_state_s {
  ATMI_UNINITIALIZED = -1,
  ATMI_INITIALIZED = 0,
  ATMI_READY = 1,
  ATMI_DISPATCHED = 2,
  ATMI_EXECUTED = 3,
  ATMI_COMPLETED = 4,
  ATMI_FAILED = 9999
} atmi_state_t;

/**
 * @brief Scheduler Types.
 */
typedef enum atmi_scheduler_s {
  ATMI_SCHED_NONE = 0,  // No scheduler, all tasks go to the same queue
  ATMI_SCHED_RR         // Round-robin tasks across queues
} atmi_scheduler_t;

/**
 * @brief ATMI data arg types.
 */
typedef enum atmi_arg_type_s { ATMI_IN, ATMI_OUT, ATMI_IN_OUT } atmi_arg_type_t;

/**
 * @brief ATMI Memory Fences for Tasks.
 */
typedef enum atmi_task_fence_scope_s {
  /**
   * No memory fence applied; external fences have to be applied around the task
   * launch/completion.
   */
  ATMI_FENCE_SCOPE_NONE = 0,
  /**
   * The fence is applied to the device.
   */
  ATMI_FENCE_SCOPE_DEVICE = 1,
  /**
   * The fence is applied to the entire system.
   */
  ATMI_FENCE_SCOPE_SYSTEM = 2
} atmi_task_fence_scope_t;

/** @} */
typedef char boolean;

/** \defgroup common Common ATMI Structures
 *  @{
 */
/**
 * @brief ATMI Task Profile Data Structure
 */
typedef struct atmi_tprofile_s {
  /**
   * Timestamp of task dispatch.
   */
  unsigned long int dispatch_time;
  /**
   * Timestamp when the task's dependencies were all met and ready to be
   * dispatched.
   */
  unsigned long int ready_time;
  /**
   * Timstamp when the task started execution.
   */
  unsigned long int start_time;
  /**
   * Timestamp when the task completed execution.
   */
  unsigned long int end_time;
} atmi_tprofile_t;

/**
 * @brief ATMI Compute Place
 */
typedef struct atmi_place_s {
  /**
   * The node in a cluster where computation should occur.
   * Default is node_id = 0 for local computations.
   */
  unsigned int node_id;
  /**
   * Device type: CPU, GPU or DSP
   */
  atmi_devtype_t type;
  /**
   * The device ordinal number ordered by runtime; -1 for any
   */
  int device_id;
  /**
   * Compute Unit Mask (advanced feature)
   */
  unsigned long cu_mask;
} atmi_place_t;

/**
 * @brief ATMI Memory Place
 */
typedef struct atmi_mem_place_s {
  /**
   * The node in a cluster where computation should occur.
   * Default is node_id = 0 for local computations.
   */
  unsigned int node_id;
  /**
   * Device type: CPU, GPU or DSP
   */
  atmi_devtype_t dev_type;
  /**
   * The device ordinal number ordered by runtime; -1 for any
   */
  int dev_id;
  // atmi_memtype_t mem_type;        // Fine grained or Coarse grained
  /**
   * The memory space/region ordinal number ordered by runtime; -1 for any
   */
  int mem_id;
} atmi_mem_place_t;

/**
 * @brief ATMI Memory Space/region Structure
 */
typedef struct atmi_memory_s {
  /**
   * Memory capacity
   */
  unsigned long int capacity;
  /**
   * Memory type
   */
  atmi_memtype_t type;
} atmi_memory_t;

/**
 * @brief ATMI Device Structure
 */
typedef struct atmi_device_s {
  /**
   * Device type: CPU, GPU or DSP
   */
  atmi_devtype_t type;
  /**
   * The number of compute cores
   */
  unsigned int core_count;
  /**
   * The number of memory spaces/regions that are accessible
   * from this device
   */
  unsigned int memory_count;
  /**
   * Array of memory spaces/regions that are accessible
   * from this device.
   */
  atmi_memory_t* memories;
} atmi_device_t;

/**
 * @brief ATMI Machine Structure
 */
typedef struct atmi_machine_s {
  /**
   * The number of devices categorized by the device type
   */
  unsigned int device_count_by_type[ATMI_DEVTYPE_ALL];
  /**
   * The device structures categorized by the device type
   */
  atmi_device_t* devices_by_type[ATMI_DEVTYPE_ALL];
} atmi_machine_t;

/**
 * @brief ATMI Task info structure
 */
typedef struct atmi_task_s {
  /**
   * Previously consistent state of task
   */
  atmi_state_t state;
  /**
   * Previously consistent task profile
   */
  atmi_tprofile_t profile;
} atmi_task_t;

/**
 * @brief The ATMI task handle.
 */
typedef unsigned long int atmi_task_handle_t;

/**
 * @brief The ATMI taskgroup handle.
 *
 * @details ATMI task groups can be a collection of compute and memory tasks.
 * They can have different properties like being ordered or
 * belonging to the same compute/memory place.
 *
 */
typedef unsigned long int atmi_taskgroup_handle_t;

/**
 * @brief The special NULL task handle.
 */
extern atmi_task_handle_t ATMI_NULL_TASK_HANDLE;
/**
 * @brief The special default taskgroup handle.
 */
extern atmi_taskgroup_handle_t ATMI_DEFAULT_TASKGROUP_HANDLE;
/**
 * @brief The special default compute place (GPU 0).
 */
extern atmi_place_t ATMI_DEFAULT_PLACE;

/**
 * @brief The ATMI Task Launch Parameter Data Structure
 */
typedef struct atmi_lparm_s {
  /**
   * Number of global threads/workitems in each dimension
   */
  unsigned long gridDim[3];
  /**
   * Workgroup size in each dimension
   */
  unsigned long groupDim[3];
  /** Taskgroup to which this task belongs.
   * Default = @p ATMI_DEFAULT_TASKGROUP_HANDLE. The runtime
   * enables several optimizations for tasks within the same
   * taskgroup (e.g., ordered taskgroups can execute in the
   * same queue, or tasks from the same taskgroup can be scheduled
   * independently from tasks from another taskgroup, and so on).
   */
  atmi_taskgroup_handle_t group;
  /**
   * Indicates whether to share completion signal objects with
   * other tasks in the same taskgroup (optimization opportunity).
   * Default = @p false
   */
  boolean groupable;
  /**
   * Default = @p false (asynchronous with respect to launching agent).
   */
  boolean synchronous;
  /**
   * Memory acquire semantics for this task.
   * Default = ATMI_FENCE_SCOPE_SYSTEM
   */
  atmi_task_fence_scope_t acquire_scope;
  /**
   * Memory release semantics for this task.
   * Default = ATMI_FENCE_SCOPE_SYSTEM
   */
  atmi_task_fence_scope_t release_scope;
  /**
   * Number of predecessor (parent) tasks required to be completed
   * before this task may begin execution. Default = 0.
   */
  int num_required;
  /**
   * Array of predecessor (parent) task handles required to be completed
   * before this task may begin execution. Default = NULL.
   */
  atmi_task_handle_t* requires;
  /**
   * Number of predecessor (parent) taskgroups required to be completed
   * before any task from this taskgroup may begin execution.
   * Default = 0
   */
  int num_required_groups;
  /**
   * Array of predecessor (parent) taskgroup handles required to be completed
   * before any task from this taskgroup may begin execution. Default = NULL.
   */
  atmi_taskgroup_handle_t* required_groups;
  /**
   * Indicates if metrics need to be collected about this task or not. If
   * @p true, then the profiling information will be returned in
   * task_info->profile structure, so this needs task_info to not be NULL.
   * Default = @p false.
   */
  boolean profilable;
  /**
   * \deprecated Constant that can be queried for the library vesion number.
   */
  int atmi_id;
  /**
   * Kernel implementation identifier if more than one kernel implementation
   * is defined for this task's kernel.
   */
  int kernel_id;
  /**
   * Compute location to launch this task.
   */
  atmi_place_t place;
  /**
   * Optional user-created structure to store executed task's information
   * like task's state or task's time profile.
   */
  atmi_task_t* task_info;
  /**
   * (Experimental) The continuation task of the current task.
   */
  atmi_task_handle_t continuation_task;
} atmi_lparm_t;

/**
 * @brief The ATMI Data Copy Parameter Data Structure
 */
typedef struct atmi_cparm_s {
  /** Taskgroup to which this task belongs.
   * Default = @p ATMI_DEFAULT_TASKGROUP_HANDLE. The runtime
   * enables several optimizations for tasks within the same
   * taskgroup (e.g., ordered taskgroups can execute in the
   * same queue, or tasks from the same taskgroup can be scheduled
   * independently from tasks from another taskgroup, and so on).
   */
  atmi_taskgroup_handle_t group;
  /**
   * Indicates whether to share completion signal objects with
   * other tasks in the same taskgroup (optimization opportunity).
   * Default = @p false
   */
  boolean groupable;
  /**
   * Indicates if metrics need to be collected about this task or not. If
   * @p true, then the profiling information will be returned in
   * task_info->profile structure, so this needs task_info to not be NULL.
   * Default = @p false.
   */
  boolean profilable;
  /**
   * Default = @p false (asynchronous with respect to launching agent).
   */
  boolean synchronous;
  /**
   * Number of predecessor (parent) tasks required to be completed
   * before this task may begin execution. Default = 0.
   */
  int num_required;
  /**
   * Array of predecessor (parent) task handles required to be completed
   * before this task may begin execution. Default = NULL.
   */
  atmi_task_handle_t* requires;
  /**
   * Number of predecessor (parent) taskgroups required to be completed
   * before any task from this taskgroup may begin execution.
   * Default = 0
   */
  int num_required_groups;
  /**
   * Array of predecessor (parent) taskgroup handles required to be completed
   * before any task from this taskgroup may begin execution. Default = NULL.
   */
  atmi_taskgroup_handle_t* required_groups;
  /**
   * Optional user-created structure to store executed task's information
   * like task's state or task's time profile.
   */
  atmi_task_t* task_info;
} atmi_cparm_t;

/**
 * @brief (Experimental) High-level data abstraction
 */
typedef struct atmi_data_s {
  /**
   * The data pointer
   */
  void* ptr;

  /**
   * Data size
   */
  unsigned int size;
  /**
   * The memory placement of the data
   */
  atmi_mem_place_t place;
  // TODO(ashwinma): what other information can be part of data?
} atmi_data_t;
/** @} */

// Below are some helper macros that can be used to setup
// some of the ATMI data structures.
#define ATMI_PLACE_ANY(node)                                    \
  {                                                             \
    .node_id = node, .type = ATMI_DEVTYPE_ALL, .device_id = -1, \
    .cu_mask = 0xFFFFFFFFFFFFFFFF                               \
  }
#define ATMI_PLACE_ANY_CPU(node)                                \
  {                                                             \
    .node_id = node, .type = ATMI_DEVTYPE_CPU, .device_id = -1, \
    .cu_mask = 0xFFFFFFFFFFFFFFFF                               \
  }
#define ATMI_PLACE_ANY_GPU(node)                                \
  {                                                             \
    .node_id = node, .type = ATMI_DEVTYPE_GPU, .device_id = -1, \
    .cu_mask = 0xFFFFFFFFFFFFFFFF                               \
  }
#define ATMI_PLACE_CPU(node, cpu_id)                                \
  {                                                                 \
    .node_id = node, .type = ATMI_DEVTYPE_CPU, .device_id = cpu_id, \
    .cu_mask = 0xFFFFFFFFFFFFFFFF                                   \
  }
#define ATMI_PLACE_GPU(node, gpu_id)                                \
  {                                                                 \
    .node_id = node, .type = ATMI_DEVTYPE_GPU, .device_id = gpu_id, \
    .cu_mask = 0xFFFFFFFFFFFFFFFF                                   \
  }
#define ATMI_PLACE_CPU_MASK(node, cpu_id, cpu_mask)                \
  {                                                                \
    .node_id = node, .type = ATMI_DEVTYPE_CPU, device_id = cpu_id, \
    .cu_mask = (0x0 | cpu_mask)                                    \
  }
#define ATMI_PLACE_GPU_MASK(node, gpu_id, gpu_mask)                \
  {                                                                \
    .node_id = node, .type = ATMI_DEVTYPE_GPU, device_id = gpu_id, \
    .cu_mask = (0x0 | gpu_mask)                                    \
  }
#define ATMI_PLACE(node, dev_type, dev_id, mask) \
  { .node_id = node, .type = dev_type, .device_id = dev_id, .cu_mask = mask }

#define ATMI_MEM_PLACE_ANY(node) \
  { .node_id = node, .dev_type = ATMI_DEVTYPE_ALL, .dev_id = -1, .mem_id = -1 }
#define ATMI_MEM_PLACE_ANY_CPU(node) \
  { .node_id = node, .dev_type = ATMI_DEVTYPE_CPU, .dev_id = -1, .mem_id = -1 }
#define ATMI_MEM_PLACE_ANY_GPU(node) \
  { .node_id = node, .dev_type = ATMI_DEVTYPE_GPU, .dev_id = -1, .mem_id = -1 }
#define ATMI_MEM_PLACE_CPU(node, cpu_id)                             \
  {                                                                  \
    .node_id = node, .dev_type = ATMI_DEVTYPE_CPU, .dev_id = cpu_id, \
    .mem_id = -1                                                     \
  }
#define ATMI_MEM_PLACE_GPU(node, gpu_id)                             \
  {                                                                  \
    .node_id = node, .dev_type = ATMI_DEVTYPE_GPU, .dev_id = gpu_id, \
    .mem_id = -1                                                     \
  }
#define ATMI_MEM_PLACE_CPU_MEM(node, cpu_id, cpu_mem_id)             \
  {                                                                  \
    .node_id = node, .dev_type = ATMI_DEVTYPE_CPU, .dev_id = cpu_id, \
    .mem_id = cpu_mem_id                                             \
  }
#define ATMI_MEM_PLACE_GPU_MEM(node, gpu_id, gpu_mem_id)             \
  {                                                                  \
    .node_id = node, .dev_type = ATMI_DEVTYPE_GPU, .dev_id = gpu_id, \
    .mem_id = gpu_mem_id                                             \
  }
#define ATMI_MEM_PLACE(d_type, d_id, m_id) \
  { .node_id = 0, .dev_type = d_type, .dev_id = d_id, .mem_id = m_id }
#define ATMI_MEM_PLACE_NODE(node, d_type, d_id, m_id) \
  { .node_id = node, .dev_type = d_type, .dev_id = d_id, .mem_id = m_id }

#define ATMI_DATA(X, PTR, COUNT, PLACE) \
  atmi_data_t X;                        \
  X.ptr = PTR;                          \
  X.size = COUNT;                       \
  X.place = PLACE;

#define WORKITEMS gridDim[0]
#define WORKITEMS2D gridDim[1]
#define WORKITEMS3D gridDim[2]

#define ATMI_CPARM(X)                                          \
  atmi_cparm_t* X;                                             \
  atmi_cparm_t _##X = {.group = ATMI_DEFAULT_TASKGROUP_HANDLE, \
                       .groupable = ATMI_FALSE,                \
                       .profilable = ATMI_FALSE,               \
                       .synchronous = ATMI_FALSE,              \
                       .num_required = 0,                      \
                       .requires = NULL,                       \
                       .num_required_groups = 0,               \
                       .required_groups = NULL,                \
                       .task_info = NULL};                     \
  X = &_##X;

#ifndef __OPENCL_C_VERSION__
#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)
#define MAKE_UNIQUE(x) CONCATENATE(x, __LINE__)
#define ATMI_PARM_SET_NAMED_DEPENDENCIES(X, HANDLES, ...)   \
  atmi_task_handle_t(HANDLES)[] = {__VA_ARGS__};            \
  (X)->num_required = sizeof(HANDLES) / sizeof(HANDLES[0]); \
  (X)->requires = HANDLES;

#define ATMI_PARM_SET_DEPENDENCIES(X, ...) \
  ATMI_PARM_SET_NAMED_DEPENDENCIES(X, MAKE_UNIQUE(handles), __VA_ARGS__)
#endif

#define ATMI_LPARM(X)                           \
  atmi_lparm_t* X;                              \
  atmi_lparm_t _##X;                            \
  _##X.gridDim[0] = 1;                          \
  _##X.gridDim[1] = 1;                          \
  _##X.gridDim[2] = 1;                          \
  _##X.groupDim[0] = 1;                         \
  _##X.groupDim[1] = 1;                         \
  _##X.groupDim[2] = 1;                         \
  _##X.group = 0ull;                            \
  _##X.groupable = ATMI_FALSE;                  \
  _##X.synchronous = ATMI_FALSE;                \
  _##X.acquire_scope = ATMI_FENCE_SCOPE_SYSTEM; \
  _##X.release_scope = ATMI_FENCE_SCOPE_SYSTEM; \
  _##X.num_required = 0;                        \
  _##X.requires = NULL;                         \
  _##X.num_required_groups = 0;                 \
  _##X.required_groups = NULL;                  \
  _##X.profilable = ATMI_FALSE;                 \
  _##X.atmi_id = ATMI_VRM;                      \
  _##X.kernel_id = -1;                          \
  _##X.place = (atmi_place_t)ATMI_PLACE_ANY(0); \
  _##X.task_info = NULL;                        \
  X = &_##X;

#define ATMI_LPARM_1D(X, Y) \
  ATMI_LPARM(X);            \
  X->gridDim[0] = Y;        \
  X->groupDim[0] = 64;

#define ATMI_LPARM_GPU_1D(X, GPU, Y) \
  ATMI_LPARM_1D(X, Y);               \
  X->place = (atmi_place_t)ATMI_PLACE_GPU(0, GPU);

#define ATMI_LPARM_CPU(X, CPU) \
  ATMI_LPARM(X);               \
  X->place = (atmi_place_t)ATMI_PLACE_CPU(0, CPU);

#define ATMI_LPARM_CPU_1D(X, CPU, Y) \
  ATMI_LPARM(X);                     \
  X->gridDim[0] = Y;                 \
  X->place = (atmi_place_t)ATMI_PLACE_CPU(0, CPU);

#define ATMI_LPARM_2D(X, Y, Z) \
  ATMI_LPARM(X);               \
  X->gridDim[0] = Y;           \
  X->gridDim[1] = Z;           \
  X->groupDim[0] = 64;         \
  X->groupDim[1] = 8;

#define ATMI_LPARM_3D(X, Y, Z, V) \
  ATMI_LPARM(X);                  \
  X->gridDim[0] = Y;              \
  X->gridDim[1] = Z;              \
  X->gridDim[2] = V;              \
  X->groupDim[0] = 8;             \
  X->groupDim[1] = 8;             \
  X->groupDim[2] = 8;

#endif  // INCLUDE_ATMI_H_
