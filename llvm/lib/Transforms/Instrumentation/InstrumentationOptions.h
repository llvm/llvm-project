//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_INSTRUMENTATION_INSTRUMENTATIONOPTIONS_H
#define LLVM_LIB_TRANSFORMS_INSTRUMENTATION_INSTRUMENTATIONOPTIONS_H

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/ProfileData/ProfCorrelatorKind.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"
#include <optional>

namespace llvm {
// Mode for selecting how to insert frame record info into the stack ring
// buffer.
enum class RecordStackHistoryMode {
  // Do not record frame record info.
  none,

  // Insert instructions into the prologue for storing into the stack ring
  // buffer directly.
  instr,

  // Add a call to __hwasan_add_frame_record in the runtime.
  libcall,
};
} // namespace llvm

#define OPTIONS_STRUCT_DECL
#include "InstrumentationOptions.inc"

#endif // LLVM_LIB_TRANSFORMS_INSTRUMENTATION_INSTRUMENTATIONOPTIONS_H
