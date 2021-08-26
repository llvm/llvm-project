//===- LiveDebugValues.cpp - Tracking Debug Value MIs ---------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_LIVEDEBUGVALUES_LIVEDEBUGVALUES_H
#define LLVM_LIB_CODEGEN_LIVEDEBUGVALUES_LIVEDEBUGVALUES_H

#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/TargetPassConfig.h"

namespace llvm {

// Inline namespace for types / symbols shared between different
// LiveDebugValues implementations.
inline namespace SharedLiveDebugValues {

// Expose a base class for LiveDebugValues interfaces to inherit from. This
// allows the generic LiveDebugValues pass handles to call into the
// implementation.
class LDVImpl {
public:
  virtual bool ExtendRanges(MachineFunction &MF, MachineDominatorTree *DomTree,
                            TargetPassConfig *TPC, unsigned InputBBLimit,
                            unsigned InputDbgValLimit) = 0;
  virtual ~LDVImpl() {}
};

} // namespace SharedLiveDebugValues

// Factory functions for LiveDebugValues implementations.
extern LDVImpl *makeVarLocBasedLiveDebugValues();
extern LDVImpl *makeInstrRefBasedLiveDebugValues();
extern LDVImpl *makeHeterogeneousLiveDebugValues();
} // namespace llvm

#endif // LLVM_LIB_CODEGEN_LIVEDEBUGVALUES_LIVEDEBUGVALUES_H
