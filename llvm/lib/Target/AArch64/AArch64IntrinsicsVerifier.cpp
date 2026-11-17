//===- AArch64IntrinsicsVerifier.cpp - Check SVE/SME intrinsic features ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IntrinsicFeatureGuard.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "aarch64-intrinsics-verifier"

namespace {

#include "AArch64GenIntrinsicFeatureChecks.inc"

// A streaming-compatible function can enter streaming mode with SME. These
// features are the minimum implied by SME and are available in that mode.
static constexpr FeatureBitset AArch64StreamingModeFeatures = {
    AArch64::FeatureBF16, AArch64::FeatureFPARMv8, AArch64::FeatureFullFP16,
    AArch64::FeatureNEON, AArch64::FeatureSME};

static bool matchesType(Type *Actual, Type *Expected, const DataLayout &DL) {
  if (Actual == Expected)
    return true;
  if (Expected->isPointerTy())
    return Actual->isPointerTy();
  if (auto *ActualVector = dyn_cast<VectorType>(Actual)) {
    if (auto *ExpectedVector = dyn_cast<VectorType>(Expected)) {
      auto *Pointer = dyn_cast<PointerType>(ActualVector->getElementType());
      return Pointer &&
             ExpectedVector->getElementType()->isIntegerTy(
                 DL.getPointerSizeInBits(Pointer->getAddressSpace())) &&
             ActualVector->getElementCount() ==
                 ExpectedVector->getElementCount();
    }
  }
  if (auto *ActualStruct = dyn_cast<StructType>(Actual)) {
    if (auto *ExpectedStruct = dyn_cast<StructType>(Expected)) {
      if (ActualStruct->getNumElements() != ExpectedStruct->getNumElements())
        return false;
      for (unsigned I = 0; I < ActualStruct->getNumElements(); ++I)
        if (!matchesType(ActualStruct->getElementType(I),
                         ExpectedStruct->getElementType(I), DL))
          return false;
      return true;
    }
  }
  return false;
}

static bool matchesSignature(const IntrinsicInst &Call,
                             const IntrinsicFeatureVariant &V) {
  if (V.AppliesToAllTypes)
    return true;
  FunctionType *Expected = getAArch64IntrinsicFunctionType(
      Call.getContext(), Call.getIntrinsicID(), V);
  FunctionType *Actual = Call.getFunctionType();
  if (Actual->getNumParams() != Expected->getNumParams() ||
      Actual->isVarArg() != Expected->isVarArg())
    return false;
  const DataLayout &DL = Call.getDataLayout();
  if (!matchesType(Actual->getReturnType(), Expected->getReturnType(), DL))
    return false;
  for (unsigned I = 0; I < Actual->getNumParams(); ++I)
    if (!matchesType(Actual->getParamType(I), Expected->getParamType(I), DL))
      return false;
  return true;
}

static bool hasFeatures(const IntrinsicFeatureGuard &Guard,
                        const FeatureBitset &Available) {
  for (const FeatureBitset &Required :
       getAArch64IntrinsicFeatureAlternatives(Guard)) {
    if ((Available & Required) == Required)
      return true;
  }
  return false;
}

struct IntrinsicCheckResult {
  bool HasGuards = false;
  bool FoundSignature = false;
  bool Valid = false;
  const IntrinsicFeatureGuard *Failed = nullptr;
  StringRef FailedMode;
};

static IntrinsicCheckResult checkIntrinsic(const IntrinsicInst &II,
                                           const AArch64Subtarget &ST) {
  IntrinsicCheckResult Result;
  ArrayRef<IntrinsicFeatureVariant> Variants =
      getAArch64IntrinsicVariants(II.getIntrinsicID());
  if (Variants.empty())
    return Result;
  Result.HasGuards = true;

  const FeatureBitset &Available = ST.getFeatureBits();
  const bool Streaming = ST.isStreaming();
  const bool BothModes = ST.isStreamingCompatible() && !Streaming;
  // A streaming-compatible function can enter streaming mode with SME.
  const FeatureBitset StreamingAvailable =
      BothModes ? Available | AArch64StreamingModeFeatures : Available;

  for (const IntrinsicFeatureVariant &V : Variants) {
    assert(V.NumGuards == 2 && "AArch64 variants need two mode guards");
    if (!matchesSignature(II, V))
      continue;
    Result.FoundSignature = true;
    // AArch64 uses guard 0 for non-streaming mode and guard 1 for
    // streaming mode.
    const IntrinsicFeatureGuard &NonStreamingGuard =
        getAArch64IntrinsicFeatureGuard(V, 0);
    const IntrinsicFeatureGuard &StreamingGuard =
        getAArch64IntrinsicFeatureGuard(V, 1);
    if ((!Streaming || BothModes) &&
        !hasFeatures(NonStreamingGuard, Available)) {
      Result.Failed = &NonStreamingGuard;
      Result.FailedMode = "non-streaming";
      continue;
    }
    if ((Streaming || BothModes) &&
        !hasFeatures(StreamingGuard, StreamingAvailable)) {
      Result.Failed = &StreamingGuard;
      Result.FailedMode = "streaming";
      continue;
    }
    Result.Valid = true;
    break;
  }
  return Result;
}

class AArch64IntrinsicsVerifier : public FunctionPass {
public:
  static char ID;

  AArch64IntrinsicsVerifier() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AArch64 Intrinsics Verifier";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesAll();
  }

  bool runOnFunction(Function &F) override {
    const auto &TM =
        getAnalysis<TargetPassConfig>().getTM<AArch64TargetMachine>();
    const AArch64Subtarget &ST = *TM.getSubtargetImpl(F);
    for (Instruction &I : instructions(F)) {
      auto *II = dyn_cast<IntrinsicInst>(&I);
      if (!II)
        continue;
      IntrinsicCheckResult Result = checkIntrinsic(*II, ST);
      if (!Result.HasGuards || Result.Valid)
        continue;

      StringRef Name = II->getCalledFunction()->getName();
      std::string Message = "AArch64 intrinsic '" + Name.str() +
                            "' in function '" + F.getName().str() + "' ";
      if (!Result.FoundSignature)
        Message += "has an unsupported type signature";
      else if (!Result.Failed->NumClauses)
        Message += "is unavailable in " + Result.FailedMode.str() + " mode";
      else
        Message += "requires " + std::string(Result.Failed->Expression) +
                   " in " + Result.FailedMode.str() + " mode";
      reportFatalUsageError(StringRef(Message));
    }
    return false;
  }
};

} // namespace

bool llvm::isAArch64IntrinsicAvailable(const IntrinsicInst &II,
                                       const AArch64Subtarget &ST) {
  return checkIntrinsic(II, ST).Valid;
}

char AArch64IntrinsicsVerifier::ID = 0;

INITIALIZE_PASS_BEGIN(AArch64IntrinsicsVerifier, DEBUG_TYPE,
                      "AArch64 Intrinsics Verifier", false, true)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AArch64IntrinsicsVerifier, DEBUG_TYPE,
                    "AArch64 Intrinsics Verifier", false, true)

FunctionPass *llvm::createAArch64IntrinsicsVerifierPass() {
  return new AArch64IntrinsicsVerifier();
}
