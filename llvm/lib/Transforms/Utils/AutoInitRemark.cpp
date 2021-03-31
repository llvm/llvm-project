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
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Utils/Local.h"

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

static Optional<uint64_t> getSizeInBytes(Optional<uint64_t> SizeInBits) {
  if (!SizeInBits || *SizeInBits % 8 != 0)
    return None;
  return *SizeInBits / 8;
}

void AutoInitRemark::inspectStore(StoreInst &SI) {
  bool Volatile = SI.isVolatile();
  bool Atomic = SI.isAtomic();
  int64_t Size = DL.getTypeStoreSize(SI.getOperand(0)->getType());

  OptimizationRemarkMissed R(RemarkPass.data(), "AutoInitStore", &SI);
  R << "Store inserted by -ftrivial-auto-var-init.\nStore size: "
    << NV("StoreSize", Size) << " bytes.";
  inspectDst(SI.getOperand(1), R);
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
  inspectDst(II.getOperand(0), R);
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
    inspectDst(CI.getOperand(0), R);
    break;
  }
}

void AutoInitRemark::inspectSizeOperand(Value *V, OptimizationRemarkMissed &R) {
  if (auto *Len = dyn_cast<ConstantInt>(V)) {
    uint64_t Size = Len->getZExtValue();
    R << " Memory operation size: " << NV("StoreSize", Size) << " bytes.";
  }
}

void AutoInitRemark::inspectVariable(const Value *V,
                                     SmallVectorImpl<VariableInfo> &Result) {
  // If we find some information in the debug info, take that.
  bool FoundDI = false;
  // Try to get an llvm.dbg.declare, which has a DILocalVariable giving us the
  // real debug info name and size of the variable.
  for (const DbgVariableIntrinsic *DVI :
       FindDbgAddrUses(const_cast<Value *>(V))) {
    if (DILocalVariable *DILV = DVI->getVariable()) {
      Optional<uint64_t> DISize = getSizeInBytes(DILV->getSizeInBits());
      VariableInfo Var{DILV->getName(), DISize};
      if (!Var.isEmpty()) {
        Result.push_back(std::move(Var));
        FoundDI = true;
      }
    }
  }
  if (FoundDI) {
    assert(!Result.empty());
    return;
  }

  const auto *AI = dyn_cast<AllocaInst>(V);
  if (!AI)
    return;

  // If not, get it from the alloca.
  Optional<StringRef> Name = AI->hasName()
                                 ? Optional<StringRef>(AI->getName())
                                 : Optional<StringRef>(None);
  Optional<TypeSize> TySize = AI->getAllocationSizeInBits(DL);
  Optional<uint64_t> Size =
      TySize ? getSizeInBytes(TySize->getFixedSize()) : None;
  VariableInfo Var{Name, Size};
  if (!Var.isEmpty())
    Result.push_back(std::move(Var));
}

void AutoInitRemark::inspectDst(Value *Dst, OptimizationRemarkMissed &R) {
  // Find if Dst is a known variable we can give more information on.
  SmallVector<const Value *, 2> Objects;
  getUnderlyingObjects(Dst, Objects);
  SmallVector<VariableInfo, 2> VIs;
  for (const Value *V : Objects)
    inspectVariable(V, VIs);

  if (VIs.empty())
    return;

  R << "\nVariables: ";
  for (unsigned i = 0; i < VIs.size(); ++i) {
    const VariableInfo &VI = VIs[i];
    assert(!VI.isEmpty() && "No extra content to display.");
    if (i != 0)
      R << ", ";
    if (VI.Name)
      R << NV("VarName", *VI.Name);
    else
      R << NV("VarName", "<unknown>");
    if (VI.Size)
      R << " (" << NV("VarSize", *VI.Size) << " bytes)";
  }
  R << ".";
}
