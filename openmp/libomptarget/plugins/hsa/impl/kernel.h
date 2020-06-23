/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/

#ifndef SRC_RUNTIME_INCLUDE_KERNEL_H_
#define SRC_RUNTIME_INCLUDE_KERNEL_H_

#include <map>
#include <queue>
#include <string>
#include <vector>
#include "internal.h"
#include "task.h"

namespace core {
class Kernel;
class KernelImpl {
 public:
  // constructor/destructor
  KernelImpl(unsigned int id, const std::string& name,
             atmi_platform_type_t platform_type, const Kernel& kernel,
             atmi_devtype_t devtype);
  virtual ~KernelImpl();

  // accessors
  atmi_devtype_t devtype() const { return devtype_; }
  unsigned int id() const { return id_; }
  std::string name() const { return name_; }
  atmi_platform_type_t platform_type() const { return platform_type_; }
  void* kernarg_region() const { return kernarg_region_; }
  void set_kernarg_region(void* p) {
    if (p) kernarg_region_ = p;
  }

  uint32_t kernarg_segment_size() const { return kernarg_segment_size_; }
  void set_kernarg_segment_size(uint32_t s) { kernarg_segment_size_ = s; }

  std::queue<int>& free_kernarg_segments() { return free_kernarg_segments_; }

  pthread_mutex_t& mutex() { return mutex_; }

  std::vector<uint64_t>& arg_offsets() { return arg_offsets_; }

  std::vector<TaskImpl*>& launched_tasks() { return launched_tasks_; }
  // functions
 protected:
  // FIXME: would anyone need to reverse engineer the
  // user-specified ID from the impls index?
  unsigned int id_;
  std::string name_;
  atmi_devtype_t devtype_;
  atmi_platform_type_t platform_type_;
  // reference to parent kernel
  const Kernel& kernel_;

  std::vector<uint64_t> arg_offsets_;

  pthread_mutex_t mutex_;  // to lock changes to the free pool
  void* kernarg_region_;
  std::queue<int> free_kernarg_segments_;
  uint32_t kernarg_segment_size_;  // differs for CPU vs GPU

  // potential running tasks that may need to be waited upon for
  // completion so that we can reclaim all their resources cleanly
  std::vector<TaskImpl*> launched_tasks_;
};  // class KernelImpl

class GPUKernelImpl : public KernelImpl {
 public:
  // constructor/destructor
  GPUKernelImpl(unsigned int id, const std::string& name,
                atmi_platform_type_t platform_type, const Kernel& kernel);
  ~GPUKernelImpl();

  // accessors

 public:
  // size of the below vectors should equal the number of
  // active GPUs in the process. Each GPU will have its own
  // kernel object compiled to its own ISA
  std::vector<uint64_t> kernel_objects_;
  std::vector<uint32_t> group_segment_sizes_;
  std::vector<uint32_t> private_segment_sizes_;

 private:
};

class Kernel {
 public:
  // constructor/destructor
  Kernel(uint64_t id, const int num_args, const size_t* arg_sizes);
  ~Kernel();

  // create GPU and CPU kernel implementation objects
  /* KernelImpl* createGPUKernelImpl(uint64_t id, const std::string& name,
                                  atmi_platform_type_t platform_type);
  */
  // accessors
  uint64_t id() const { return id_; }
  int num_args() const { return num_args_; }
  std::vector<size_t>& arg_sizes() { return arg_sizes_; }
  const std::vector<size_t>& arg_sizes() const { return arg_sizes_; }
  std::vector<KernelImpl*>& impls() { return impls_; }
  std::map<unsigned int, unsigned int>& id_map() { return id_map_; }

  // functions
  bool isValidId(unsigned int id);
  KernelImpl* getKernelImpl(unsigned int id);
  int getKernelIdMapIndex(unsigned int kernel_id);
  int getKernelImplId(atmi_lparm_t* lparm);

 private:
  // ID
  uint64_t id_;

  int num_args_;
  std::vector<size_t> arg_sizes_;
  std::vector<KernelImpl*> impls_;
  std::map<unsigned int, unsigned int> id_map_;
};  // class Kernel

}  // namespace core
#endif  // SRC_RUNTIME_INCLUDE_KERNEL_H_
