/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#include "internal.h"
#include "rt.h"

#include <stdio.h>
#include <string>

const char *get_error_string(hsa_status_t err) {
  const char *res;
  hsa_status_t rc = hsa_status_string(err, &res);
  return (rc == HSA_STATUS_SUCCESS) ? res : "HSA_STATUS UNKNOWN.";
}

namespace core {
/*
 * Environment variables
 */
void Environment::GetEnvAll() {
  std::string var = GetEnv("ATMI_HELP");
  if (!var.empty()) {
    printf("ATMI_MAX_HSA_QUEUE_SIZE : positive integer\n"
           "ATMI_DEBUG : 1 for printing out trace/debug info\n");
  }

  var = GetEnv("ATMI_MAX_HSA_QUEUE_SIZE");
  if (!var.empty())
    max_queue_size_ = std::stoi(var);

  var = GetEnv("ATMI_DEBUG");
  if (!var.empty())
    debug_mode_ = std::stoi(var);
}
} // namespace core
