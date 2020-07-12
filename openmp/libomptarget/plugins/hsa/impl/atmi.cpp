/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#include "rt.h"
/*
 * Initialize/Finalize
 */
atmi_status_t atmi_init() {
  return core::Runtime::getInstance().Initialize();
}

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
 * Kernels
 */
atmi_status_t atmi_kernel_create(atmi_kernel_t *atmi_kernel, const int num_args,
                                 const size_t *arg_sizes, const int num_impls,
                                 ...) {
  va_list arguments;
  va_start(arguments, num_impls);
  return core::Runtime::getInstance().CreateKernel(
      atmi_kernel, num_args, arg_sizes, num_impls, arguments);
  va_end(arguments);
}

atmi_status_t atmi_kernel_release(atmi_kernel_t atmi_kernel) {
  return core::Runtime::getInstance().ReleaseKernel(atmi_kernel);
}

atmi_status_t atmi_kernel_create_empty(atmi_kernel_t *atmi_kernel,
                                       const int num_args,
                                       const size_t *arg_sizes) {
  return core::Runtime::getInstance().CreateEmptyKernel(atmi_kernel, num_args,
                                                        arg_sizes);
}

atmi_status_t atmi_kernel_add_gpu_impl(atmi_kernel_t atmi_kernel,
                                       const char *impl,
                                       const unsigned int ID) {
  return core::Runtime::getInstance().AddGPUKernelImpl(atmi_kernel, impl, ID);
}

/*
 * Synchronize
 */

atmi_status_t atmi_task_wait(atmi_task_handle_t task) {
  return core::Runtime::getInstance().TaskWait(task);
}

/*
 * Tasks
 */

atmi_task_handle_t atmi_task_launch(
    atmi_lparm_t *lparm, atmi_kernel_t atmi_kernel,
    void **args /*, more params for place info? */) {
  return core::Runtime::getInstance().LaunchTask(lparm, atmi_kernel, args);
}

/*
 * Data
 */
atmi_status_t atmi_memcpy(void *dest, const void *src, size_t size) {
  return core::Runtime::getInstance().Memcpy(dest, src, size);
}

atmi_status_t atmi_free(void *ptr) {
  return core::Runtime::getInstance().Memfree(ptr);
}

atmi_status_t atmi_malloc(void **ptr, size_t size, atmi_mem_place_t place) {
  return core::Runtime::getInstance().Malloc(ptr, size, place);
}
