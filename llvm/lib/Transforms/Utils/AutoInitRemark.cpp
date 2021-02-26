//===-- AutoInitRemark.cpp - Auto-init remark analysis---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the analysis for the "auto-init" remark.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/AutoInitRemark.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;
using namespace llvm::ore;

void AutoInitRemark::inspectStore(StoreInst &SI) {
  bool Volatile = SI.isVolatile();
  bool Atomic = SI.isAtomic();
  int64_t Size = DL.getTypeStoreSize(SI.getOperand(0)->getType());

  OptimizationRemarkMissed R(RemarkPass.data(), "AutoInitStore", &SI);
  R << "Store inserted by -ftrivial-auto-var-init.\nStore size: "
    << NV("StoreSize", Size) << " bytes.";
  if (Volatile)
    R << " Volatile: " << NV("StoreVolatile", true) << ".";
  if (Atomic)
    R << " Atomic: " << NV("StoreAtomic", true) << ".";
  // Emit StoreVolatile: false and StoreAtomic: false under ExtraArgs. This
  // won't show them in the remark message but will end up in the serialized
  // remarks.
  if (!Volatile || !Atomic)
    R << setExtraArgs();
  if (!Volatile)
    R << " Volatile: " << NV("StoreVolatile", false) << ".";
  if (!Atomic)
    R << " Atomic: " << NV("StoreAtomic", false) << ".";

  ORE.emit(R);
}

void AutoInitRemark::inspectUnknown(Instruction &I) {
  ORE.emit(OptimizationRemarkMissed(RemarkPass.data(),
                                    "AutoInitUnknownInstruction", &I)
           << "Initialization inserted by -ftrivial-auto-var-init.");
}
