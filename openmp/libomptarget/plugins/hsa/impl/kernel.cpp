/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/

#include "kernel.h"
#include "internal.h"
#include "machine.h"

extern std::vector<hsa_amd_memory_pool_t> atl_gpu_kernarg_pools;
extern std::map<uint64_t, core::Kernel *> KernelImplMap;

namespace core {
void allow_access_to_all_gpu_agents(void *ptr);
atmi_status_t Runtime::CreateEmptyKernel(atmi_kernel_t *atmi_kernel,
                                         const int num_args,
                                         const size_t *arg_sizes) {
  static uint64_t counter = 0;
  uint64_t k_id = ++counter;
  atmi_kernel->handle = (uint64_t)k_id;
  Kernel *kernel = new Kernel(k_id, num_args, arg_sizes);
  KernelImplMap[k_id] = kernel;
  return ATMI_STATUS_SUCCESS;
}

atmi_status_t Runtime::ReleaseKernel(atmi_kernel_t atmi_kernel) {
  uint64_t k_id = atmi_kernel.handle;
  delete KernelImplMap[k_id];
  KernelImplMap.erase(k_id);
  return ATMI_STATUS_SUCCESS;
}

atmi_status_t Runtime::CreateKernel(atmi_kernel_t *atmi_kernel,
                                    const int num_args, const size_t *arg_sizes,
                                    const char * impl) {
  atmi_status_t status;
  hsa_status_t err;
  if (!atl_is_atmi_initialized()) return ATMI_STATUS_ERROR;
  status = atmi_kernel_create_empty(atmi_kernel, num_args, arg_sizes);
  ATMIErrorCheck(Creating kernel object, status);

  int impl_id = 0;
  {
    status = atmi_kernel_add_gpu_impl(*atmi_kernel, impl, impl_id);
    ATMIErrorCheck(Adding GPU kernel implementation, status);
    DEBUG_PRINT("GPU kernel %s added [%u]\n", impl, impl_id);
    // rest of kernel impl fields will be populated at first kernel launch
  }
  return ATMI_STATUS_SUCCESS;
}

atmi_status_t Runtime::AddGPUKernelImpl(atmi_kernel_t atmi_kernel,
                                        const char *impl,
                                        const unsigned int ID) {
  if (!atl_is_atmi_initialized() || KernelInfoTable.empty())
    return ATMI_STATUS_ERROR;
  uint64_t k_id = atmi_kernel.handle;
  Kernel *kernel = KernelImplMap[k_id];
  if (kernel->id_map().find(ID) != kernel->id_map().end()) {
    fprintf(stderr, "Kernel ID %d already found\n", ID);
    return ATMI_STATUS_ERROR;
  }
  std::vector<ATLGPUProcessor> &gpu_procs =
      g_atl_machine.processors<ATLGPUProcessor>();
  int gpu_count = gpu_procs.size();

  std::string hsaco_name = std::string(impl);
  bool some_success = false;
  for (int gpu = 0; gpu < gpu_count; gpu++) {
    if (KernelInfoTable[gpu].find(hsaco_name) != KernelInfoTable[gpu].end()) {
      DEBUG_PRINT("Found kernel %s for GPU %d\n", hsaco_name.c_str(), gpu);
      some_success = true;
    } else {
      DEBUG_PRINT("Did NOT find kernel %s for GPU %d\n", hsaco_name.c_str(),
                  gpu);
      continue;
    }
  }
  if (!some_success) return ATMI_STATUS_ERROR;

  KernelImpl *kernel_impl =
      new GPUKernelImpl(ID, hsaco_name, AMDGCN, *kernel);

  kernel->id_map()[ID] = kernel->impls().size();

  kernel->impls().push_back(kernel_impl);
  // rest of kernel impl fields will be populated at first kernel launch
  return ATMI_STATUS_SUCCESS;
}

KernelImpl::KernelImpl(unsigned int id, const std::string &name,
                       atmi_platform_type_t platform_type, const Kernel &kernel,
                       atmi_devtype_t devtype = ATMI_DEVTYPE_ALL)
    : id_(id),
      name_(name),
      platform_type_(platform_type),
      kernel_(kernel),
      devtype_(devtype) {}

GPUKernelImpl::GPUKernelImpl(unsigned int id, const std::string &name,
                             atmi_platform_type_t platform_type,
                             const Kernel &kernel)
    : KernelImpl(id, name, platform_type, kernel, ATMI_DEVTYPE_GPU) {
  std::vector<ATLGPUProcessor> &gpu_procs =
      g_atl_machine.processors<ATLGPUProcessor>();
  int gpu_count = gpu_procs.size();

  kernel_objects_.reserve(gpu_count);
  group_segment_sizes_.reserve(gpu_count);
  private_segment_sizes_.reserve(gpu_count);
  int max_kernarg_segment_size = 0;
  arg_offsets_.reserve(kernel.num_args());
  bool args_offsets_set = false;
  for (int gpu = 0; gpu < gpu_count; gpu++) {
    if (KernelInfoTable[gpu].find(name_) != KernelInfoTable[gpu].end()) {
      atl_kernel_info_t info = KernelInfoTable[gpu][name_];
      // save the rest of the kernel info metadata
      kernel_objects_[gpu] = info.kernel_object;
      group_segment_sizes_[gpu] = info.group_segment_size;
      private_segment_sizes_[gpu] = info.private_segment_size;
      if (max_kernarg_segment_size < info.kernel_segment_size)
        max_kernarg_segment_size = info.kernel_segment_size;

      // cache this value to retrieve arg offsets
      // TODO(ashwinma): will arg offsets change per device?
      if (!args_offsets_set) {
        for (int i = 0; i < kernel.num_args(); i++) {
          arg_offsets_[i] = info.arg_offsets[i];
        }
        args_offsets_set = true;
      }
    }
  }
  kernarg_segment_size_ = max_kernarg_segment_size;

  /* create kernarg memory */
  kernarg_region_ = NULL;
  if (kernarg_segment_size_ > 0) {
    DEBUG_PRINT("New kernarg segment size: %u\n", kernarg_segment_size_);
    hsa_status_t err = hsa_amd_memory_pool_allocate(
        atl_gpu_kernarg_pools[0], kernarg_segment_size_ * MAX_NUM_KERNELS, 0,
        &kernarg_region_);
      ErrorCheck(Allocating memory for the executable-kernel, err);
      allow_access_to_all_gpu_agents(kernarg_region_);

      for (int k = 0; k < MAX_NUM_KERNELS; k++) {
        atmi_implicit_args_t *impl_args =
            reinterpret_cast<atmi_implicit_args_t *>(
                reinterpret_cast<char *>(kernarg_region_) +
                (((k + 1) * kernarg_segment_size_) -
                 sizeof(atmi_implicit_args_t)));
        impl_args->offset_x = 0;
        impl_args->offset_y = 0;
        impl_args->offset_z = 0;
      }
  }

  for (int i = 0; i < MAX_NUM_KERNELS; i++) {
    free_kernarg_segments_.push(i);
  }
  pthread_mutex_init(&mutex_, NULL);
}

KernelImpl::~KernelImpl() {
  // wait for all task instances of all kernel_impl of this kernel
  for (auto &task : launched_tasks_) {
    if (task->state() < ATMI_COMPLETED) task->wait();
  }
  launched_tasks_.clear();

  arg_offsets_.clear();
  clear_container(&free_kernarg_segments_);
}

GPUKernelImpl::~GPUKernelImpl() {
  lock(&mutex_);
  ErrorCheck(Memory pool free, hsa_amd_memory_pool_free(kernarg_region_));
  kernel_objects_.clear();
  group_segment_sizes_.clear();
  private_segment_sizes_.clear();
  unlock(&mutex_);
}

bool Kernel::isValidId(unsigned int kernel_id) {
  std::map<unsigned int, unsigned int>::iterator it = id_map_.find(kernel_id);
  if (it == id_map_.end()) {
    fprintf(stderr, "ERROR: Kernel not found\n");
    return false;
  }
  int idx = it->second;
  if (idx >= impls_.size()) {
    fprintf(stderr, "Kernel ID %d out of bounds (%lu)\n", kernel_id,
            impls_.size());
    return false;
  }
  return true;
}

int Kernel::getKernelIdMapIndex(unsigned int kernel_id) {
  if (!isValidId(kernel_id)) {
    return -1;
  }
  return id_map_[kernel_id];
}

KernelImpl *Kernel::getKernelImpl(unsigned int kernel_id) {
  int idx = getKernelIdMapIndex(kernel_id);
  if (idx < 0) {
    fprintf(stderr, "Incorrect Kernel ID %d\n", kernel_id);
    return NULL;
  }
  return impls_[idx];
}

int Kernel::getKernelImplId(atmi_lparm_t *lparm) {
  int kernel_id = lparm->kernel_id;
  if (kernel_id == -1) {
    // choose the first available kernel for the given devtype
    for (auto kernel_impl : impls_) {
      if (kernel_impl->devtype() == lparm->place.type) {
        kernel_id = kernel_impl->id();
        break;
      }
    }
    if (kernel_id == -1) {
      fprintf(stderr,
              "ERROR: Kernel/PIF %lu doesn't have any implementations\n", id());
      return -1;
    }
  } else {
    if (!isValidId(kernel_id)) {
      DEBUG_PRINT("ERROR: Kernel ID %d not found\n", kernel_id);
      return -1;
    }
  }
  KernelImpl *kernel_impl = getKernelImpl(kernel_id);
  if (num_args_ && kernel_impl->kernarg_region() == NULL) {
    fprintf(stderr, "ERROR: Kernel Arguments not initialized for Kernel %s\n",
            kernel_impl->name().c_str());
    return -1;
  }

  return kernel_id;
}

Kernel::Kernel(uint64_t id, const int num_args, const size_t *arg_sizes)
    : id_(id), num_args_(num_args) {
  id_map_.clear();
  arg_sizes_.clear();
  impls_.clear();
  for (int i = 0; i < num_args; i++) {
    arg_sizes_.push_back(arg_sizes[i]);
  }
}

Kernel::~Kernel() {
  for (auto kernel_impl : impls_) {
    delete kernel_impl;
  }
  impls_.clear();
  arg_sizes_.clear();
  id_map_.clear();
}
}  // namespace core
