//===- LiveDebugValues.cpp - Tracking Debug Value MIs ---------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
// Notified per clause 4(b) of the license.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_LIVEDEBUGVALUES_LIVEDEBUGVALUES_H
#define LLVM_LIB_CODEGEN_LIVEDEBUGVALUES_LIVEDEBUGVALUES_H

namespace llvm {
class MachineDominatorTree;
class MachineFunction;
class TargetPassConfig;
class Triple;

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
  virtual ~LDVImpl() = default;
};

} // namespace SharedLiveDebugValues

// Factory functions for LiveDebugValues implementations.
extern LDVImpl *makeVarLocBasedLiveDebugValues();
extern LDVImpl *makeInstrRefBasedLiveDebugValues();
extern LDVImpl *makeHeterogeneousLiveDebugValues();

extern bool debuginfoShouldUseDebugInstrRef(const Triple &T);

} // namespace llvm

#endif // LLVM_LIB_CODEGEN_LIVEDEBUGVALUES_LIVEDEBUGVALUES_H
