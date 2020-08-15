/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#ifndef SRC_RUNTIME_INCLUDE_RT_H_
#define SRC_RUNTIME_INCLUDE_RT_H_

#include "atmi_runtime.h"
#include "hsa.h"
#include <cstdarg>
#include <string>

namespace core {

#define DEFAULT_MAX_QUEUE_SIZE 4096
#define DEFAULT_MAX_KERNEL_TYPES 32
#define DEFAULT_NUM_GPU_QUEUES -1 // computed in code
#define DEFAULT_NUM_CPU_QUEUES -1 // computed in code
#define DEFAULT_DEBUG_MODE 0
class Environment {
public:
  Environment()
      : max_queue_size_(DEFAULT_MAX_QUEUE_SIZE),
        max_kernel_types_(DEFAULT_MAX_KERNEL_TYPES),
        num_gpu_queues_(DEFAULT_NUM_GPU_QUEUES),
        num_cpu_queues_(DEFAULT_NUM_CPU_QUEUES),
        debug_mode_(DEFAULT_DEBUG_MODE) {
    GetEnvAll();
  }

  ~Environment() {}

  void GetEnvAll();

  int getMaxQueueSize() const { return max_queue_size_; }
  int getMaxKernelTypes() const { return max_kernel_types_; }
  int getNumGPUQueues() const { return num_gpu_queues_; }
  int getNumCPUQueues() const { return num_cpu_queues_; }
  // TODO(ashwinma): int may change to enum if we have more debug modes
  int getDebugMode() const { return debug_mode_; }
  // TODO(ashwinma): int may change to enum if we have more profile modes

private:
  std::string GetEnv(const char *name) {
    char *env = getenv(name);
    std::string ret;
    if (env) {
      ret = env;
    }
    return ret;
  }

  int max_queue_size_;
  int max_kernel_types_;
  int num_gpu_queues_;
  int num_cpu_queues_;
  int debug_mode_;
};

class Runtime final {
public:
  static Runtime &getInstance() {
    static Runtime instance;
    return instance;
  }

  // init/finalize
  static atmi_status_t Initialize();
  static atmi_status_t Finalize();

  // modules
  static atmi_status_t RegisterModuleFromMemory(
      void *, size_t, atmi_place_t,
      atmi_status_t (*on_deserialized_data)(void *data, size_t size,
                                            void *cb_state),
      void *cb_state);

  // machine info
  static atmi_machine_t *GetMachineInfo();

  // data
  static atmi_status_t Memcpy(void *, const void *, size_t);
  static atmi_status_t Memfree(void *);
  static atmi_status_t Malloc(void **, size_t, atmi_mem_place_t);

  // environment variables
  int getMaxQueueSize() const { return env_.getMaxQueueSize(); }
  int getMaxKernelTypes() const { return env_.getMaxKernelTypes(); }
  int getNumGPUQueues() const { return env_.getNumGPUQueues(); }
  int getNumCPUQueues() const { return env_.getNumCPUQueues(); }
  // TODO(ashwinma): int may change to enum if we have more debug modes
  int getDebugMode() const { return env_.getDebugMode(); }

protected:
  Runtime() = default;
  ~Runtime() = default;
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

protected:
  // variable to track environment variables
  Environment env_;
};

} // namespace core

#endif // SRC_RUNTIME_INCLUDE_RT_H_
