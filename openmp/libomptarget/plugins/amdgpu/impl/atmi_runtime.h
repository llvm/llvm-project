/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#ifndef INCLUDE_ATMI_RUNTIME_H_
#define INCLUDE_ATMI_RUNTIME_H_

#include "atmi.h"
#include <inttypes.h>
#include <stdlib.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** \defgroup handlers Function Handlers
 * This module includes all function handler types and handler
 * registration functions.
 * @{
 */
/**
 * @brief A generic function pointer representing CPU tasks.
 */
typedef void (*atmi_generic_fp)(void);

/** @} */

/** \defgroup context_functions ATMI Context Setup and Finalize
 *  @{
 */
/**
 * @brief Initialize the ATMI runtime environment.
 *
 * @detal All ATMI runtime functions will fail if this function is not called
 * at least once. The user may initialize difference device types at different
 * regions in the program in order for optimization purposes.
 *
 * @retval ::ATMI_STATUS_SUCCESS The function has executed successfully.
 *
 * @retval ::ATMI_STATUS_ERROR The function encountered errors.
 *
 * @retval ::ATMI_STATUS_UNKNOWN The function encountered errors.
 */
atmi_status_t atmi_init();

/**
 * @brief Finalize the ATMI runtime environment.
 *
 * @detail ATMI runtime functions will fail if called after finalize.
 *
 * @retval ::ATMI_STATUS_SUCCESS The function has executed successfully.
 *
 * @retval ::ATMI_STATUS_ERROR The function encountered errors.
 *
 * @retval ::ATMI_STATUS_UNKNOWN The function encountered errors.
 */
atmi_status_t atmi_finalize();
/** @} */

/** \defgroup module_functions ATMI Module
 * @{
 */

/**
 * @brief Register the ATMI code module from memory on to a specific place
 * (device).
 *
 * @detail Currently, only GPU devices need explicit module registration because
 * of their specific ISAs that require a separate compilation phase. On the
 * other
 * hand, CPU devices execute regular x86 functions that are compiled with the
 * host program.
 *
 * @param[in] module_bytes A memory region that contains the GPU modules
 * targeting ::AMDGCN platform types. Value cannot be NULL.
 *
 * @param[in] module_size Size of module region
 *
 * @param[in] place Denotes the execution place (device) on which the module
 * should be registered and loaded.
 *
 * @retval ::ATMI_STATUS_SUCCESS The function has executed successfully.
 *
 * @retval ::ATMI_STATUS_ERROR The function encountered errors.
 *
 * @retval ::ATMI_STATUS_UNKNOWN The function encountered errors.
 *
 */
atmi_status_t atmi_module_register_from_memory_to_place(void *module_bytes,
                                                        size_t module_size,
                                                        atmi_place_t place);

/** @} */

/** \defgroup machine ATMI Machine
 * @{
 */
/**
 * @brief ATMI's device discovery function to get the current machine's
 * topology.
 *
 * @detail The @p atmi_machine_t structure is a tree-based representation of the
 * compute and memory elements in the current node. Once ATMI is initialized,
 * this function can be called to retrieve the pointer to this global structure.
 *
 * @return Returns a pointer to a global structure of tyoe @p atmi_machine_t.
 * Returns NULL if ATMI is not initialized.
 */
atmi_machine_t *atmi_machine_get_info();
/** @} */

/** \defgroup memory_functions ATMI Data Management
 * @{
 */
/**
 * @brief Allocate memory from the specified memory place.
 *
 * @detail This function allocates memory from the specified memory place. If
 * the memory
 * place belongs primarily to the CPU, then the memory will be accessible by
 * other GPUs and CPUs in the system. If the memory place belongs primarily to a
 * GPU,
 * then it cannot be accessed by other devices in the system.
 *
 * @param[in] ptr The pointer to the memory that will be allocated.
 *
 * @param[in] size The size of the allocation in bytes.
 *
 * @param[in] place The memory place in the system to perform the allocation.
 *
 * @retval ::ATMI_STATUS_SUCCESS The function has executed successfully.
 *
 * @retval ::ATMI_STATUS_ERROR The function encountered errors.
 *
 * @retval ::ATMI_STATUS_UNKNOWN The function encountered errors.
 *
 */
atmi_status_t atmi_malloc(void **ptr, size_t size, atmi_mem_place_t place);

/**
 * @brief Frees memory that was previously allocated.
 *
 * @detail This function frees memory that was previously allocated by calling
 * @p atmi_malloc. It throws an error otherwise. It is illegal to access a
 * pointer after a call to this function.
 *
 * @param[in] ptr The pointer to the memory that has to be freed.
 *
 * @retval ::ATMI_STATUS_SUCCESS The function has executed successfully.
 *
 * @retval ::ATMI_STATUS_ERROR The function encountered errors.
 *
 * @retval ::ATMI_STATUS_UNKNOWN The function encountered errors.
 *
 */
atmi_status_t atmi_free(void *ptr);

/**
 * @brief Syncrhonously copy memory from the source to destination memory
 * locations.
 *
 * @detail This function assumes that the source and destination regions are
 * non-overlapping. The runtime determines the memory place of the source and
 * the
 * destination and executes the appropriate optimized data movement methodology.
 *
 * @param[in] dest The destination pointer previously allocated by a system
 * allocator or @p atmi_malloc.
 *
 * @param[in] src The source pointer previously allocated by a system
 * allocator or @p atmi_malloc.
 *
 * @param[in] size The size of the data to be copied in bytes.
 *
 * @retval ::ATMI_STATUS_SUCCESS The function has executed successfully.
 *
 * @retval ::ATMI_STATUS_ERROR The function encountered errors.
 *
 * @retval ::ATMI_STATUS_UNKNOWN The function encountered errors.
 *
 */
atmi_status_t atmi_memcpy(void *dest, const void *src, size_t size);

/** @} */

/** \defgroup cpu_dev_runtime ATMI CPU Device Runtime
 * @{
 */

/**
 * @brief Retrieve the global thread ID of
 * the currently running task. This function is valid
 * only within the body of a CPU task.
 *
 * @param[in] dim The dimension of the CPU task. Valid
 * dimensions are 0, 1 and 2.
 *
 * @return The global thread ID of the ATMI CPU task.
 *
 */
unsigned long get_global_id(unsigned int dim);

/**
 * @brief Retrieve the global thread count of
 * the currently running task. This function is valid
 * only within the body of a CPU task.
 *
 * @param[in] dim The dimension of the CPU task. Valid
 * dimensions are 0, 1 and 2.
 *
 * @return The global thread count of the ATMI CPU task.
 *
 */
unsigned long get_global_size(unsigned int dim);

/**
 * @brief Retrieve the local thread ID of
 * the currently running task. This function is valid
 * only within the body of a CPU task.
 *
 * @param[in] dim The dimension of the CPU task. Valid
 * dimensions are 0, 1 and 2.
 *
 * @return The local thread ID of the ATMI CPU task. The
 * current ATMI CPU task model assumes the workgroup size
 * of 1 at all times for all dimensions, so this call
 * always returns 0.
 */
unsigned long get_local_id(unsigned int dim);

/**
 * @brief Retrieve the local thread count of
 * the currently running task. This function is valid
 * only within the body of a CPU task.
 *
 * @param[in] dim The dimension of the CPU task. Valid
 * dimensions are 0, 1 and 2.
 *
 * @return The local thread count of the ATMI CPU task. The
 * current ATMI CPU task model assumes the workgroup size
 * of 1 at all times for all dimensions, so this call
 * always returns 1.
 *
 */
unsigned long get_local_size(unsigned int dim);

/**
 * @brief Retrieve the thread workgroup ID of
 * the currently running task. This function is valid
 * only within the body of a CPU task.
 *
 * @param[in] dim The dimension of the CPU task. Valid
 * dimensions are 0, 1 and 2.
 *
 * @return The thread workgroup ID of the ATMI CPU task. The
 * current ATMI CPU task model assumes the workgroup size
 * of 1 at all times for all dimensions, so this call
 * is equivalent to calling @p get_global_id.
 */
unsigned long get_group_id(unsigned int dim);

/**
 * @brief Retrieve the thread workgroup count of
 * the currently running task. This function is valid
 * only within the body of a CPU task.
 *
 * @param[in] dim The dimension of the CPU task. Valid
 * dimensions are 0, 1 and 2.
 *
 * @return The thread workgroup count of the ATMI CPU task. The
 * current ATMI CPU task model assumes the workgroup size
 * of 1 at all times for all dimensions, so this call
 * is equivalent to calling @p get_global_size.
 */
unsigned long get_num_groups(unsigned int dim);
/** @} */

#ifdef __cplusplus
}
#endif

#endif // INCLUDE_ATMI_RUNTIME_H_
