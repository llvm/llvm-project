//===------------------- llvm-advisor-plugin.h - LLVM Advisor
//-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// C ABI for external capability plugins. Include this header in plugin
// implementations. No C++ headers required — pure C interface.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct AdvisorCapabilitySpec {
  const char *capability_id;
  const char *version;
  const char *description;
  const char *execution_mode;
  const char *cost_class;
  const char *readiness_level;
  const char **required_inputs;
  const char **depends_on;
} AdvisorCapabilitySpec;

typedef struct AdvisorRunContext {
  const char *unit_id;
  const char *snapshot_id;
  const char *data_dir;
  int (*read_blob)(const char *address, char **out_data, size_t *out_size);
  void (*free_blob)(char *data);
  volatile int *cancellation_token;
} AdvisorRunContext;

typedef struct AdvisorCapabilityResult {
  int success;
  const char *output_json;
  const char *error_message;
} AdvisorCapabilityResult;

void llvm_advisor_register_capabilities(AdvisorCapabilitySpec **specs_out,
                                        int *count_out);
AdvisorCapabilityResult
llvm_advisor_run_capability(const char *capability_id,
                            const AdvisorRunContext *ctx);
void llvm_advisor_free_result(AdvisorCapabilityResult *result);

#ifdef __cplusplus
}
#endif // __cplusplus
