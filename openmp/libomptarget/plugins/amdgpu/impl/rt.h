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
#define DEFAULT_DEBUG_MODE 0
class Environment {
public:
  Environment()
      : max_queue_size_(DEFAULT_MAX_QUEUE_SIZE),
        debug_mode_(DEFAULT_DEBUG_MODE) {
    GetEnvAll();
  }

  void GetEnvAll();

  int getMaxQueueSize() const { return max_queue_size_; }

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
  // machine info
  static atmi_machine_t *GetMachineInfo();
  // modules
  static atmi_status_t RegisterModuleFromMemory(
      void *, size_t, atmi_place_t,
      atmi_status_t (*on_deserialized_data)(void *data, size_t size,
                                            void *cb_state),
      void *cb_state);

  // data
  static atmi_status_t Memcpy(hsa_signal_t, void *, const void *, size_t);
  static atmi_status_t Memfree(void *);
  static atmi_status_t Malloc(void **, size_t, atmi_mem_place_t);

  // environment variables
  int getMaxQueueSize() const { return env_.getMaxQueueSize(); }

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
