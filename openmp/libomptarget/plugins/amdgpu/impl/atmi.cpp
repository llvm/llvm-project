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
atmi_status_t atmi_memcpy(hsa_signal_t sig, void *dest, const void *src,
                          size_t size) {
  hsa_status_t rc = hsa_memory_copy(dest, src, size);

  // hsa_memory_copy sometimes fails in situations where
  // allocate + copy succeeds. Looks like it might be related to
  // locking part of a read only segment. Fall back for now.
  if (rc == HSA_STATUS_SUCCESS) {
    return ATMI_STATUS_SUCCESS;
  }

  return core::Runtime::Memcpy(sig, dest, src, size);
}

atmi_status_t atmi_memcpy_h2d(hsa_signal_t sig, void *device_dest,
                              const void *host_src, size_t size) {
  return atmi_memcpy(sig, device_dest, host_src, size);
}

atmi_status_t atmi_memcpy_d2h(hsa_signal_t sig, void *host_dest,
                              const void *device_src, size_t size) {
  return atmi_memcpy(sig, host_dest, device_src, size);
}

atmi_status_t atmi_free(void *ptr) { return core::Runtime::Memfree(ptr); }

atmi_status_t atmi_malloc(void **ptr, size_t size, atmi_mem_place_t place) {
  return core::Runtime::Malloc(ptr, size, place);
}
