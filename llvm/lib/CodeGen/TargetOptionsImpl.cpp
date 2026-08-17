//===-- TargetOptionsImpl.cpp - Options that apply to all targets ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the methods in the TargetOptions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

/// DisableFramePointerElim - This returns true if frame pointer elimination
/// optimization should be disabled for the given machine function.
bool TargetOptions::DisableFramePointerElim(const MachineFunction &MF) const {
  FramePointerKind FP = MF.getFrameInfo().getFramePointerPolicy();
  switch (FP) {
  case FramePointerKind::All:
    return true;
  case FramePointerKind::NonLeaf:
  case FramePointerKind::NonLeafNoReserve:
    return MF.getFrameInfo().hasCalls();
  case FramePointerKind::None:
  case FramePointerKind::Reserved:
    return false;
  }
  llvm_unreachable("unknown frame pointer flag");
}

bool TargetOptions::FramePointerIsReserved(const MachineFunction &MF) const {
  FramePointerKind FP = MF.getFrameInfo().getFramePointerPolicy();
  switch (FP) {
  case FramePointerKind::All:
  case FramePointerKind::NonLeaf:
  case FramePointerKind::Reserved:
    return true;
  case FramePointerKind::NonLeafNoReserve:
    return MF.getFrameInfo().hasCalls();
  case FramePointerKind::None:
    return false;
  }
  llvm_unreachable("unknown frame pointer flag");
}

/// HonorSignDependentRoundingFPMath - Return true if the codegen must assume
/// that the rounding mode of the FPU can change from its default.
bool TargetOptions::HonorSignDependentRoundingFPMath() const {
  return HonorSignDependentRoundingFPMathOption;
}

/// NOTE: There are targets that still do not support the debug entry values
/// production and that is being controlled with the SupportsDebugEntryValues.
/// In addition, SCE debugger does not have the feature implemented, so prefer
/// not to emit the debug entry values in that case.
/// The EnableDebugEntryValues can be used for the testing purposes.
bool TargetOptions::ShouldEmitDebugEntryValues() const {
  return (SupportsDebugEntryValues && DebuggerTuning != DebuggerKind::SCE) ||
         EnableDebugEntryValues;
}
