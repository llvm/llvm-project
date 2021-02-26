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
#include "llvm/IR/IntrinsicInst.h"

using namespace llvm;
using namespace llvm::ore;

static void volatileOrAtomicWithExtraArgs(bool Volatile, bool Atomic,
                                          OptimizationRemarkMissed &R) {
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
}

void AutoInitRemark::inspectStore(StoreInst &SI) {
  bool Volatile = SI.isVolatile();
  bool Atomic = SI.isAtomic();
  int64_t Size = DL.getTypeStoreSize(SI.getOperand(0)->getType());

  OptimizationRemarkMissed R(RemarkPass.data(), "AutoInitStore", &SI);
  R << "Store inserted by -ftrivial-auto-var-init.\nStore size: "
    << NV("StoreSize", Size) << " bytes.";
  volatileOrAtomicWithExtraArgs(Volatile, Atomic, R);
  ORE.emit(R);
}

void AutoInitRemark::inspectUnknown(Instruction &I) {
  ORE.emit(OptimizationRemarkMissed(RemarkPass.data(),
                                    "AutoInitUnknownInstruction", &I)
           << "Initialization inserted by -ftrivial-auto-var-init.");
}

void AutoInitRemark::inspectIntrinsicCall(IntrinsicInst &II) {
  SmallString<32> CallTo;
  bool Atomic = false;
  switch (II.getIntrinsicID()) {
  case Intrinsic::memcpy:
    CallTo = "memcpy";
    break;
  case Intrinsic::memmove:
    CallTo = "memmove";
    break;
  case Intrinsic::memset:
    CallTo = "memset";
    break;
  case Intrinsic::memcpy_element_unordered_atomic:
    CallTo = "memcpy";
    Atomic = true;
    break;
  case Intrinsic::memmove_element_unordered_atomic:
    CallTo = "memmove";
    Atomic = true;
    break;
  case Intrinsic::memset_element_unordered_atomic:
    CallTo = "memset";
    Atomic = true;
    break;
  default:
    return inspectUnknown(II);
  }

  OptimizationRemarkMissed R(RemarkPass.data(), "AutoInitIntrinsic", &II);
  inspectCallee(StringRef(CallTo), /*KnownLibCall=*/true, R);
  inspectSizeOperand(II.getOperand(2), R);

  auto *CIVolatile = dyn_cast<ConstantInt>(II.getOperand(3));
  // No such thing as a memory intrinsic that is both atomic and volatile.
  bool Volatile = !Atomic && CIVolatile && CIVolatile->getZExtValue();
  volatileOrAtomicWithExtraArgs(Volatile, Atomic, R);
  ORE.emit(R);
}

void AutoInitRemark::inspectCall(CallInst &CI) {
  Function *F = CI.getCalledFunction();
  if (!F)
    return inspectUnknown(CI);

  LibFunc LF;
  bool KnownLibCall = TLI.getLibFunc(*F, LF) && TLI.has(LF);
  OptimizationRemarkMissed R(RemarkPass.data(), "AutoInitCall", &CI);
  inspectCallee(F, KnownLibCall, R);
  inspectKnownLibCall(CI, LF, R);
  ORE.emit(R);
}

template <typename FTy>
void AutoInitRemark::inspectCallee(FTy F, bool KnownLibCall,
                                   OptimizationRemarkMissed &R) {
  R << "Call to ";
  if (!KnownLibCall)
    R << NV("UnknownLibCall", "unknown") << " function ";
  R << NV("Callee", F) << " inserted by -ftrivial-auto-var-init.";
}

void AutoInitRemark::inspectKnownLibCall(CallInst &CI, LibFunc LF,
                                         OptimizationRemarkMissed &R) {
  switch (LF) {
  default:
    return;
  case LibFunc_bzero:
    inspectSizeOperand(CI.getOperand(1), R);
    break;
  }
}

void AutoInitRemark::inspectSizeOperand(Value *V, OptimizationRemarkMissed &R) {
  if (auto *Len = dyn_cast<ConstantInt>(V)) {
    uint64_t Size = Len->getZExtValue();
    R << " Memory operation size: " << NV("StoreSize", Size) << " bytes.";
  }
}
