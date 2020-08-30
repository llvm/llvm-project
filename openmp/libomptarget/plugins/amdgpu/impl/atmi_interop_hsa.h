/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#ifndef INCLUDE_ATMI_INTEROP_HSA_H_
#define INCLUDE_ATMI_INTEROP_HSA_H_

#include "atmi_runtime.h"
#include "hsa.h"
#include "hsa_ext_amd.h"

#ifdef __cplusplus
extern "C" {
#endif
/** \defgroup interop_hsa_functions ATMI-HSA Interop
 *  @{
 */

/**
 * @brief Get the device address and size of an HSA global symbol
 *
 * @detail Use this function to query the device address and size of an HSA
 * global symbol.
 * The symbol can be set at by the compiler or by the application writer in a
 * language-specific manner. This function is meaningful only after calling one
 * of the @p atmi_module_register functions.
 *
 * @param[in] place The ATMI memory place
 *
 * @param[in] symbol Pointer to a non-NULL global symbol name
 *
 * @param[in] var_addr Pointer to a non-NULL @p void* variable that will
 * hold the device address of the global symbol object.
 *
 * @param[in] var_size Pointer to a non-NULL @p uint variable that will
 * hold the size of the global symbol object.
 *
 * @retval ::ATMI_STATUS_SUCCESS The function has executed successfully.
 *
 * @retval ::ATMI_STATUS_ERROR If @p symbol, @p var_addr or @p var_size are
 * invalid
 * location in the current node, or if ATMI is not initialized.
 *
 * @retval ::ATMI_STATUS_UNKNOWN The function encountered errors.
 */
atmi_status_t atmi_interop_hsa_get_symbol_info(atmi_mem_place_t place,
                                               const char *symbol,
                                               void **var_addr,
                                               unsigned int *var_size);

/**
 * @brief Get the HSA-specific kernel info from a kernel name
 *
 * @detail Use this function to query the HSA-specific kernel info from the
 * kernel name.
 * This function is meaningful only after calling one
 * of the @p atmi_module_register functions.
 *
 * @param[in] place The ATMI memory place
 *
 * @param[in] kernel_name Pointer to a char array with the kernel name
 *
 * @param[in] info The different possible kernel properties
 *
 * @param[in] value Pointer to a non-NULL @p uint variable that will
 * hold the return value of the kernel property.
 *
 * @retval ::ATMI_STATUS_SUCCESS The function has executed successfully.
 *
 * @retval ::ATMI_STATUS_ERROR If @p symbol, @p var_addr or @p var_size are
 * invalid
 * location in the current node, or if ATMI is not initialized.
 *
 * @retval ::ATMI_STATUS_UNKNOWN The function encountered errors.
 */
atmi_status_t atmi_interop_hsa_get_kernel_info(
    atmi_mem_place_t place, const char *kernel_name,
    hsa_executable_symbol_info_t info, uint32_t *value);
/** @} */

#ifdef __cplusplus
}
#endif

#endif // INCLUDE_ATMI_INTEROP_HSA_H_
