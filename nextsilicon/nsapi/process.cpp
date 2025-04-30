#include "process.h"
#include "internal/nsapi_process.hpp"

int nsapi_multiprocess_set(uint32_t proc_index, uint32_t total_processes) {
  auto cmd = NSAPICommandMultiprocessSet(proc_index, total_processes);
  return nsapi_get_current_handler()->Execute(cmd);
}
