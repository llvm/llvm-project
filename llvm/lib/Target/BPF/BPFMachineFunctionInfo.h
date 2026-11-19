//===-- BPFMachineFunctionInfo.h - BPF machine function info ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_BPFMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_BPF_BPFMACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"
#include <optional>

namespace llvm {

struct BPFMachineFunctionInfo : public MachineFunctionInfo {
  BPFMachineFunctionInfo(const Function &, const TargetSubtargetInfo *) {}

  // Whether every branch fits in a 16-bit offset in any layout.
  std::optional<bool> ShortBranches;
};

} // namespace llvm

#endif
