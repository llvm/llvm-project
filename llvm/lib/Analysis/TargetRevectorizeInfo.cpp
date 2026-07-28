//===- TargetRevectorizeInfo.cpp - Target revectorization info ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetRevectorizeInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

TargetRevectorizeInfoImplBase::~TargetRevectorizeInfoImplBase() = default;

InstructionCost
TargetRevectorizeInfoImplBase::getTargetIntrinsicVectorizationCost(
    Intrinsic::ID FromID, Type *WideRetTy, ArrayRef<Type *> WideArgTys,
    ElementCount VF) const {
  llvm_unreachable("Target intrinsic cost is not implemented");
}

Instruction *TargetRevectorizeInfoImplBase::vectorizeTargetIntrinsic(
    Intrinsic::ID, ArrayRef<Type *>, ArrayRef<Value *>, ElementCount,
    IRBuilderBase &) const {
  llvm_unreachable("Target intrinsic vectorization is not implemented");
}

TargetRevectorizeInfo::TargetRevectorizeInfo(const TargetTransformInfo &TTI)
    : TargetRevectorizeInfo(
          std::make_unique<TargetRevectorizeInfoImplBase>(TTI)) {}

TargetRevectorizeInfo::TargetRevectorizeInfo(
    std::unique_ptr<const TargetRevectorizeInfoImplBase> Impl)
    : Impl(std::move(Impl)) {
  assert(this->Impl && "Expected a valid TargetRevectorizeInfo implementation");
}

TargetRevectorizeInfo::TargetRevectorizeInfo(TargetRevectorizeInfo &&) =
    default;
TargetRevectorizeInfo &
TargetRevectorizeInfo::operator=(TargetRevectorizeInfo &&) = default;
TargetRevectorizeInfo::~TargetRevectorizeInfo() = default;

bool TargetRevectorizeInfo::invalidate(
    Function &F, const PreservedAnalyses &PA,
    FunctionAnalysisManager::Invalidator &Inv) {
  // The implementation holds a reference to TTI (result of TargetIRAnalysis),
  // so it gets invalided if TargetIRAnalysis itself is invalidated.
  return Inv.invalidate<TargetIRAnalysis>(F, PA);
}

bool TargetRevectorizeInfo::isTargetIntrinsicVectorizable(
    Intrinsic::ID ID) const {
  return Impl->isTargetIntrinsicVectorizable(ID);
}

InstructionCost TargetRevectorizeInfo::getTargetIntrinsicVectorizationCost(
    Intrinsic::ID FromID, Type *WideRetTy, ArrayRef<Type *> WideArgTys,
    ElementCount VF) const {
  return Impl->getTargetIntrinsicVectorizationCost(FromID, WideRetTy,
                                                   WideArgTys, VF);
}

Instruction *TargetRevectorizeInfo::vectorizeTargetIntrinsic(
    Intrinsic::ID VectorIID, ArrayRef<Type *> TysForDecl,
    ArrayRef<Value *> WideArgs, ElementCount VF, IRBuilderBase &Builder) const {
  return Impl->vectorizeTargetIntrinsic(VectorIID, TysForDecl, WideArgs, VF,
                                        Builder);
}

TargetRevectorizeWrapper::TargetRevectorizeWrapper()
    : Callback([](const Function &, const TargetTransformInfo &TTI) {
        return TargetRevectorizeInfo(TTI);
      }) {}

TargetRevectorizeWrapper::TargetRevectorizeWrapper(
    std::function<Result(const Function &, const TargetTransformInfo &)>
        Callback)
    : Callback(std::move(Callback)) {}

TargetRevectorizeWrapper::Result
TargetRevectorizeWrapper::run(Function &F, FunctionAnalysisManager &AM) {
  assert(!F.isIntrinsic() && "Should not request TRVI for intrinsics");
  return Callback(F, AM.getResult<TargetIRAnalysis>(F));
}

AnalysisKey TargetRevectorizeWrapper::Key;
