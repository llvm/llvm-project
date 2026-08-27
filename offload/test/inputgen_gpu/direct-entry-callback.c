// Verify generate, replay, and pass-through behavior of the post-load callback.
// RUN: %clang -I%inputgen-gpu-src -I%inputgen-gpu-interface-include \
// RUN:   -I%inputgen-gpu-llvm-include %inputgen-gpu-src/inputgen_gpu_entry_state.c \
// RUN:   %inputgen-gpu-src/inputgen_gpu_entry_callbacks.c %s -o %t
// RUN: %t | FileCheck %s

#include "inputgen_gpu_entry_internal.h"

#include <stdint.h>
#include <stdio.h>

int64_t __ig_post_load(int64_t value, int64_t value_size, int32_t value_type_id,
                       int32_t id);

int main(void) {
  int Buffer[2] = {0, 17};
  InputGenEntryBuffer = Buffer;
  InputGenEntryBufferSize = sizeof(Buffer);
  InputGenEntryBufferOffset = 0;

  InputGenEntryMode = INPUTGEN_MODE_GENERATE;
  int64_t Generated = __ig_post_load(123, 4, IntegerTyID, 11);
  printf("generate=%lld buffer0=%d\n", (long long)Generated, Buffer[0]);

  Buffer[0] = 42;
  InputGenEntryMode = INPUTGEN_MODE_REPLAY;
  printf("replay=%lld\n", (long long)__ig_post_load(123, 4, IntegerTyID, 12));
  printf("passthrough-size=%lld\n",
         (long long)__ig_post_load(55, 8, IntegerTyID, 13));
  printf("passthrough-type=%lld\n", (long long)__ig_post_load(66, 4, 15, 14));
  return 0;
}

// CHECK: generate=9 buffer0=9
// CHECK: replay=42
// CHECK: passthrough-size=55
// CHECK: passthrough-type=66
