//===-- WebAssemblySubtarget.cpp - WebAssembly Subtarget Information ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the WebAssembly-specific subclass of
/// TargetSubtarget.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblySubtarget.h"
#include "GISel/WebAssemblyCallLowering.h"
#include "GISel/WebAssemblyLegalizerInfo.h"
#include "GISel/WebAssemblyRegisterBankInfo.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblyInstrInfo.h"
#include "WebAssemblyTargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-subtarget"

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_TARGET_DESC
#include "WebAssemblyGenSubtargetInfo.inc"

WebAssemblySubtarget &
WebAssemblySubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                      StringRef FS) {
  // Determine default and user-specified characteristics
  LLVM_DEBUG(llvm::dbgs() << "initializeSubtargetDependencies\n");

  if (CPU.empty())
    CPU = "generic";

  ParseSubtargetFeatures(CPU, /*TuneCPU*/ CPU, FS);  
  
  FeatureBitset Bits = getFeatureBits();

  // bulk-memory implies bulk-memory-opt
  if (HasBulkMemory) {
    HasBulkMemoryOpt = true;
    Bits.set(WebAssembly::FeatureBulkMemoryOpt);
  }

  // gc implies reference-types
  if (HasGC) {
    HasReferenceTypes = true;
  }

  // reference-types implies call-indirect-overlong
  if (HasReferenceTypes) {
    HasCallIndirectOverlong = true;
    Bits.set(WebAssembly::FeatureCallIndirectOverlong);
  }

  // In case we changed any bits, update `MCSubtargetInfo`'s `FeatureBitset`.
  setFeatureBits(Bits);

  return *this;
}

WebAssemblySubtarget::WebAssemblySubtarget(const Triple &TT,
                                           const std::string &CPU,
                                           const std::string &FS,
                                           const TargetMachine &TM)
    : WebAssemblyGenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS),
      TargetTriple(TT), InstrInfo(initializeSubtargetDependencies(CPU, FS)),
      TLInfo(TM, *this) {
  CallLoweringInfo.reset(new WebAssemblyCallLowering(*getTargetLowering()));
  Legalizer.reset(new WebAssemblyLegalizerInfo(*this));
  auto *RBI = new WebAssemblyRegisterBankInfo(*getRegisterInfo());
  RegBankInfo.reset(RBI);

  InstSelector.reset(createWebAssemblyInstructionSelector(
      *static_cast<const WebAssemblyTargetMachine *>(&TM), *this, *RBI));
}

bool WebAssemblySubtarget::enableAtomicExpand() const {
  // If atomics are disabled, atomic ops are lowered instead of expanded
  return hasAtomics();
}

bool WebAssemblySubtarget::enableMachineScheduler() const {
  // Disable the MachineScheduler for now. Even with ShouldTrackPressure set and
  // enableMachineSchedDefaultSched overridden, it appears to have an overall
  // negative effect for the kinds of register optimizations we're doing.
  return false;
}

bool WebAssemblySubtarget::useAA() const { return true; }

const CallLowering *WebAssemblySubtarget::getCallLowering() const {
  return CallLoweringInfo.get();
}

InstructionSelector *WebAssemblySubtarget::getInstructionSelector() const {
  return InstSelector.get();
}

const LegalizerInfo *WebAssemblySubtarget::getLegalizerInfo() const {
  return Legalizer.get();
}

const RegisterBankInfo *WebAssemblySubtarget::getRegBankInfo() const {
  return RegBankInfo.get();
}
