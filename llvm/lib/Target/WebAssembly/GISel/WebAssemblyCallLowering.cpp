//===-- WebAssemblyCallLowering.cpp - Call lowering for GlobalISel -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the lowering of LLVM calls to machine code calls for
/// GlobalISel.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyCallLowering.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyISelLowering.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyUtilities.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Value.h"

#define DEBUG_TYPE "wasm-call-lowering"

using namespace llvm;

WebAssemblyCallLowering::WebAssemblyCallLowering(
    const WebAssemblyTargetLowering &TLI)
    : CallLowering(&TLI) {}

bool WebAssemblyCallLowering::canLowerReturn(MachineFunction &MF,
                                             CallingConv::ID CallConv,
                                             SmallVectorImpl<BaseArgInfo> &Outs,
                                             bool IsVarArg) const {
  return WebAssembly::canLowerReturn(Outs.size(),
                                     &MF.getSubtarget<WebAssemblySubtarget>());
}

bool WebAssemblyCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                          const Value *Val,
                                          ArrayRef<Register> VRegs,
                                          FunctionLoweringInfo &FLI,
                                          Register SwiftErrorVReg) const {
  if (!Val)
    return true; // allow only void returns for now

  return false;
}

bool WebAssemblyCallLowering::lowerFormalArguments(
    MachineIRBuilder &MIRBuilder, const Function &F,
    ArrayRef<ArrayRef<Register>> VRegs, FunctionLoweringInfo &FLI) const {
  if (VRegs.empty())
    return true; // allow only empty signatures for now

  return false;
}

bool WebAssemblyCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                        CallLoweringInfo &Info) const {
  return false;
}
