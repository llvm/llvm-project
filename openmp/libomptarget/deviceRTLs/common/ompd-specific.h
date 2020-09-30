#ifndef __OMPD_SPECIFIC_H__
#define __OMPD_SPECIFIC_H__

#ifdef OMPD_SUPPORT

#include "state-queue.h"
#include <stdint.h>



__device__ void ompd_init( void );
extern "C" __device__ void ompd_bp_parallel_begin ( void );
extern "C" __device__ void ompd_bp_parallel_end ( void );
extern "C" __device__ void ompd_bp_task_begin ( void );
extern "C" __device__ void ompd_bp_task_end ( void );
extern "C" __device__ void ompd_bp_thread_begin ( void );
extern "C" __device__ void ompd_bp_thread_end ( void );


#define OMPD_FOREACH_ACCESS(OMPD_ACCESS) \
  OMPD_ACCESS(omptarget_nvptx_ThreadPrivateContext, topTaskDescr) \
  OMPD_ACCESS(omptarget_nvptx_ThreadPrivateContext,teamContext) \
  OMPD_ACCESS(omptarget_nvptx_ThreadPrivateContext,ompd_levelZeroParallelInfo) \
  OMPD_ACCESS(omptarget_nvptx_TaskDescr,ompd_thread_info) \
  OMPD_ACCESS(omptarget_nvptx_TaskDescr,prev) \
  OMPD_ACCESS(omptarget_nvptx_TeamDescr,levelZeroTaskDescr) \
  OMPD_ACCESS(ompd_nvptx_thread_info_t,state) \
  OMPD_ACCESS(ompd_nvptx_thread_info_t,threadIdx_x) \
  OMPD_ACCESS(ompd_nvptx_thread_info_t,enclosed_parallel)  \
  OMPD_ACCESS(ompd_nvptx_thread_info_t,task_function) \
  OMPD_ACCESS(ompd_nvptx_thread_info_t,task_implicit) \
  OMPD_ACCESS(ompd_nvptx_parallel_info_t,level) \
  OMPD_ACCESS(ompd_nvptx_parallel_info_t,parallel_tasks)


#define OMPD_FOREACH_SIZEOF(OMPD_SIZEOF) \
  OMPD_SIZEOF(omptarget_nvptx_ThreadPrivateContext)\
  OMPD_SIZEOF(omptarget_nvptx_TaskDescr) \
  OMPD_SIZEOF(ompd_nvptx_thread_info_t)


/* we only support work states for the moment */
typedef enum {
  omp_state_undefined      = 0x102,
  omp_state_work_serial    = 0x000,
  omp_state_work_parallel  = 0x001,
  omp_state_work_reduction = 0x002
} omp_state_t;

class omptarget_nvptx_TaskDescr;

__device__ void ompd_init_thread_master();
__device__ void ompd_set_device_specific_thread_state(
    omptarget_nvptx_TaskDescr *taskDescr, omp_state_t state);
__device__ void ompd_set_device_thread_state(omp_state_t state);
__device__ void ompd_init_thread_parallel();
__device__ void ompd_init_explicit_task(void *task_func);

INLINE void ompd_reset_device_thread_state() {
  ompd_set_device_thread_state(omp_state_work_serial);
}

/* We store parallel info in the threadPrivateContext the same way that task
 * descriptors are stored. Currently there is no support for nested
 * parallelism (TODO: there will probably be in the future), so we store one
 * parallel descriptor in the threadPrivateContext for the outermost parallel
 * region and additonally one descriptor in each thread in case of serialized
 * inner parallel regions
 */
typedef struct {
  uint16_t level;
  /* If level = 0, parallel_tasks points just to the master task descriptor
   * if level = 1, parallel_tasks points to threadPrivateContext->levelOneTaskDescr
   * if level > 1, we are in a serialized parallel region and parallel_tasks points
   * to the single task in the parallel region.
   */
  omptarget_nvptx_TaskDescr *parallel_tasks;
} ompd_nvptx_parallel_info_t;

typedef struct {
  uint64_t state; // In the host runtime we use the OMPT state.
                  // Here we need to have our own place to store it.
  uint16_t blockIdx_x; // Libomptarget should only schedule task in one dimension.
                  // To store a unique identifier for the current thread, we
                  // simply store ThreadIdx.x and BlockIdx.x
  uint16_t threadIdx_x;
  ompd_nvptx_parallel_info_t enclosed_parallel;
  void *task_function;
  uint8_t task_implicit;
} ompd_nvptx_thread_info_t;

#endif /* OMPD_SUPPORT */
#endif
