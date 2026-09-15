//===-- SuperHSubtarget.h - Define Subtarget for SuperH ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SuperH specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "SuperHSubtarget.h"
#include "MCTargetDesc/SuperHBaseInfo.h"
#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "sh-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "SuperHGenSubtargetInfo.inc"

SuperHSubtarget::SuperHSubtarget(const StringRef &CPU, const StringRef &TuneCPU, 
                                 const StringRef &FS,  const TargetMachine &TM)
    : SuperHGenSubtargetInfo(TM.getTargetTriple(), CPU, TuneCPU, FS), TM(TM),
      InstrInfo(initializeSubtargetDependencies(CPU, TuneCPU, FS)), 
      TLInfo(TM, *this), TSInfo(), FrameLowering(*this) {
}

SuperHSubtarget::~SuperHSubtarget() = default;


SuperHSubtarget &SuperHSubtarget::initializeSubtargetDependencies(
    StringRef CPU, StringRef TuneCPU, StringRef FS) {
  const Triple &TT = getTargetTriple();
  // Determine default and user specified characteristics
  std::string CPUName = std::string(CPU);
  if (TuneCPU.empty())
    TuneCPU = CPUName;

  // Parse features string.
  ParseSubtargetFeatures(CPUName, TuneCPU, FS);
  return *this;
}




//===----------------------------------------------------------------------===//
//                             Classification Functions
//===----------------------------------------------------------------------===//

SHRefClass SuperHSubtarget::classifyBlockAddressReference() const {
  switch (TM.getCodeModel()) {
  default:
    llvm_unreachable("Unsupported code model");
  case CodeModel::Small:
  case CodeModel::Kernel: {
    return SHII::MO_PCREL;
  }
  case CodeModel::Medium:
  case CodeModel::Large: {
    return isPositionIndependent() ? 
           SHII::MO_PCREL : 
           SHII::MO_DIR;
  }
  }
}

SHRefClass SuperHSubtarget::classifyLocalReference(const GlobalValue *GV) const {
  switch (TM.getCodeModel()) {
  default:
    llvm_unreachable("Unsupported code model");
  case CodeModel::Small:
  case CodeModel::Kernel: {
    return isPositionIndependent() ? 
           SHII::MO_PCREL :
           SHII::MO_DIR;
  }
  case CodeModel::Medium: {
    return isPositionIndependent() ? 
           SHII::MO_GOTOFF : 
           SHII::MO_DIR;
  }
  case CodeModel::Large: {
    return isPositionIndependent() ? 
           SHII::MO_GOTOFF : 
           SHII::MO_DIR;
  }
  }
}

SHRefClass SuperHSubtarget::classifyExternalReference(const Module &M) const {
  if (TM.shouldAssumeDSOLocal(nullptr))
    return classifyLocalReference(nullptr);

  return isPositionIndependent() ? 
         SHII::MO_GOTPC : 
         SHII::MO_GOT;
}

SHRefClass SuperHSubtarget::classifyGlobalReference(const GlobalValue *GV) const {
  return classifyGlobalReference(GV, *GV->getParent());
}

SHRefClass SuperHSubtarget::classifyGlobalReference(const GlobalValue *GV,
                                   const Module &M) const {
  if (TM.shouldAssumeDSOLocal(GV))
    return classifyLocalReference(GV);

  switch (TM.getCodeModel()) {
  default:
    llvm_unreachable("Unsupported code model");
  case CodeModel::Small:
  case CodeModel::Kernel:
  case CodeModel::Medium: {
    return isPositionIndependent() ? 
           SHII::MO_GOTPC : 
           SHII::MO_DIR;
  }
  case CodeModel::Large: {
    return isPositionIndependent() ? 
           SHII::MO_GOTOFF : 
           SHII::MO_DIR;
  }
  }
}

SHRefClass SuperHSubtarget::classifyGlobalFunctionReference(const GlobalValue *GV,
                                           const Module &M) const {
  if (TM.shouldAssumeDSOLocal(GV))
    return SHII::MO_NO_FLAG;


  // If the function is marked as non-lazy, generate an indirect call
  // which loads from the GOT directly. This avoids run-time overhead
  // at the cost of eager binding.
  auto *F = dyn_cast_or_null<Function>(GV);
  if (F && F->hasFnAttribute(Attribute::NonLazyBind)) {
    return SHII::MO_GOTPC;
  }

  return isPositionIndependent() ? 
         SHII::MO_PLT : 
         SHII::MO_DIR;
}

SHRefClass SuperHSubtarget::classifyGlobalFunctionReference(const GlobalValue *GV) const {
  return classifyGlobalFunctionReference(GV, *GV->getParent());
}