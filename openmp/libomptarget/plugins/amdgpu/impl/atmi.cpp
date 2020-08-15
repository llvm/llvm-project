/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#include "rt.h"
/*
 * Initialize/Finalize
 */
atmi_status_t atmi_init() { return core::Runtime::Initialize(); }

atmi_status_t atmi_finalize() { return core::Runtime::Finalize(); }

/*
 * Machine Info
 */
atmi_machine_t *atmi_machine_get_info() {
  return core::Runtime::GetMachineInfo();
}

/*
 * Modules
 */
atmi_status_t atmi_module_register_from_memory_to_place(
    void *module_bytes, size_t module_size, atmi_place_t place,
    atmi_status_t (*on_deserialized_data)(void *data, size_t size,
                                          void *cb_state),
    void *cb_state) {
  return core::Runtime::getInstance().RegisterModuleFromMemory(
      module_bytes, module_size, place, on_deserialized_data, cb_state);
}

/*
 * Data
 */
atmi_status_t atmi_memcpy(void *dest, const void *src, size_t size) {
  return core::Runtime::Memcpy(dest, src, size);
}

atmi_status_t atmi_free(void *ptr) { return core::Runtime::Memfree(ptr); }

atmi_status_t atmi_malloc(void **ptr, size_t size, atmi_mem_place_t place) {
  return core::Runtime::Malloc(ptr, size, place);
}
