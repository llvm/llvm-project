/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#include "rt.h"
/*
 * Initialize/Finalize
 */
atmi_status_t atmi_init() { return core::Runtime::getInstance().Initialize(); }

atmi_status_t atmi_finalize() {
  return core::Runtime::getInstance().Finalize();
}

/*
 * Machine Info
 */
atmi_machine_t *atmi_machine_get_info() {
  return core::Runtime::getInstance().GetMachineInfo();
}

/*
 * Modules
 */
atmi_status_t atmi_module_register_from_memory_to_place(void *module_bytes,
                                                        size_t module_size,
                                                        atmi_place_t place) {
  return core::Runtime::getInstance().RegisterModuleFromMemory(
      module_bytes, module_size, place);
}

/*
 * Data
 */
atmi_status_t atmi_memcpy(hsa_signal_t sig, void *dest, const void *src,
                          size_t size) {
  return core::Runtime::Memcpy(sig, dest, src, size);
}

atmi_status_t atmi_free(void *ptr) {
  return core::Runtime::getInstance().Memfree(ptr);
}

atmi_status_t atmi_malloc(void **ptr, size_t size, atmi_mem_place_t place) {
  return core::Runtime::getInstance().Malloc(ptr, size, place);
}
