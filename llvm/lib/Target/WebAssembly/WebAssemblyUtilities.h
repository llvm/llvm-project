//===-- WebAssemblyUtilities - WebAssembly Utility Functions ---*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the WebAssembly-specific
/// utility functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYUTILITIES_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYUTILITIES_H

#include "llvm/CodeGen/MachineBasicBlock.h"

namespace llvm {

class WebAssemblyFunctionInfo;

namespace WebAssembly {

bool isChild(const MachineInstr &MI, const WebAssemblyFunctionInfo &MFI);
bool mayThrow(const MachineInstr &MI);

// Exception-related function names
extern const char *const ClangCallTerminateFn;
extern const char *const CxaBeginCatchFn;
extern const char *const CxaRethrowFn;
extern const char *const StdTerminateFn;
extern const char *const PersonalityWrapperFn;

/// Return the "bottom" block of an entity, which can be either a MachineLoop or
/// WebAssemblyException. This differs from MachineLoop::getBottomBlock in that
/// it works even if the entity is discontiguous.
template <typename T> MachineBasicBlock *getBottom(const T *Unit) {
  MachineBasicBlock *Bottom = Unit->getHeader();
  for (MachineBasicBlock *MBB : Unit->blocks())
    if (MBB->getNumber() > Bottom->getNumber())
      Bottom = MBB;
  return Bottom;
}

} // end namespace WebAssembly

} // end namespace llvm

#endif
