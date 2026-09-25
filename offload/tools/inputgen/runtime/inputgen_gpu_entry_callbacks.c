//===-- InputGen GPU Entry Runtime Device Callbacks ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "inputgen_gpu_entry_internal.h"

int64_t __ig_post_load(int64_t value, int64_t value_size, int32_t value_type_id,
                       int32_t id) {
  (void)id;
  if (value_type_id != IntegerTyID || value_size != 4)
    return value;
  if (!InputGenEntryBuffer ||
      InputGenEntryBufferOffset + sizeof(int) > InputGenEntryBufferSize)
    __builtin_trap();

  int *slot = (int *)((char *)InputGenEntryBuffer + InputGenEntryBufferOffset);
  if (InputGenEntryMode == INPUTGEN_MODE_GENERATE) {
    int generated = inputgen_entry_random();
    *slot = generated;
    return generated;
  }
  if (InputGenEntryMode == INPUTGEN_MODE_REPLAY)
    return *slot;
  __builtin_trap();
}
