/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#ifndef INCLUDE_ATMI_RUNTIME_H_
#define INCLUDE_ATMI_RUNTIME_H_

#include "atmi.h"
#include "hsa.h"
#include <inttypes.h>
#include <stdlib.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

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
 * @param[in] on_deserialized_data Callback run on deserialized code object,
 * before loading it
 *
 * @param[in] cb_state void* passed to on_deserialized_data callback
 *
 * @retval ::HSA_STATUS_SUCCESS The function has executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR The function encountered errors.
 *
 */
hsa_status_t atmi_module_register_from_memory_to_place(
    void *module_bytes, size_t module_size, int DeviceId,
    hsa_status_t (*on_deserialized_data)(void *data, size_t size,
                                         void *cb_state),
    void *cb_state);

/** @} */

hsa_status_t atmi_memcpy_h2d(hsa_signal_t signal, void *deviceDest,
                             const void *hostSrc, size_t size,
                             hsa_agent_t agent);

hsa_status_t atmi_memcpy_d2h(hsa_signal_t sig, void *hostDest,
                             const void *deviceSrc, size_t size,
                             hsa_agent_t agent);

/** @} */

#ifdef __cplusplus
}
#endif

#endif // INCLUDE_ATMI_RUNTIME_H_
