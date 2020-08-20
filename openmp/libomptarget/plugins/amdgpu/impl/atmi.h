/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#ifndef INCLUDE_ATMI_H_
#define INCLUDE_ATMI_H_

#define ROCM_VERSION_MAJOR 3
#define ROCM_VERSION_MINOR 2

/** \defgroup enumerations Enumerated Types
 * @{
 */

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
} atmi_status_t;

/**
 * @brief Device Types.
 */
typedef enum atmi_devtype_s {
  ATMI_DEVTYPE_CPU = 0x0001,
  ATMI_DEVTYPE_iGPU = 0x0010,                               // Integrated GPU
  ATMI_DEVTYPE_dGPU = 0x0100,                               // Discrete GPU
  ATMI_DEVTYPE_GPU = ATMI_DEVTYPE_iGPU | ATMI_DEVTYPE_dGPU, // Any GPU
  ATMI_DEVTYPE_ALL = 0x111 // Union of all device types
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

/** \defgroup common Common ATMI Structures
 *  @{
 */

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
  atmi_memory_t *memories;
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
  atmi_device_t *devices_by_type[ATMI_DEVTYPE_ALL];
} atmi_machine_t;

// Below are some helper macros that can be used to setup
// some of the ATMI data structures.
#define ATMI_PLACE_CPU(node, cpu_id)                                           \
  { .node_id = node, .type = ATMI_DEVTYPE_CPU, .device_id = cpu_id }
#define ATMI_PLACE_GPU(node, gpu_id)                                           \
  { .node_id = node, .type = ATMI_DEVTYPE_GPU, .device_id = gpu_id }
#define ATMI_MEM_PLACE_CPU(node, cpu_id)                                       \
  {                                                                            \
    .node_id = node, .dev_type = ATMI_DEVTYPE_CPU, .dev_id = cpu_id,           \
    .mem_id = -1                                                               \
  }
#define ATMI_MEM_PLACE_GPU(node, gpu_id)                                       \
  {                                                                            \
    .node_id = node, .dev_type = ATMI_DEVTYPE_GPU, .dev_id = gpu_id,           \
    .mem_id = -1                                                               \
  }
#define ATMI_MEM_PLACE_CPU_MEM(node, cpu_id, cpu_mem_id)                       \
  {                                                                            \
    .node_id = node, .dev_type = ATMI_DEVTYPE_CPU, .dev_id = cpu_id,           \
    .mem_id = cpu_mem_id                                                       \
  }
#define ATMI_MEM_PLACE_GPU_MEM(node, gpu_id, gpu_mem_id)                       \
  {                                                                            \
    .node_id = node, .dev_type = ATMI_DEVTYPE_GPU, .dev_id = gpu_id,           \
    .mem_id = gpu_mem_id                                                       \
  }
#define ATMI_MEM_PLACE(d_type, d_id, m_id)                                     \
  { .node_id = 0, .dev_type = d_type, .dev_id = d_id, .mem_id = m_id }

#endif // INCLUDE_ATMI_H_
