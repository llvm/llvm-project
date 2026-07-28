//===- TargetRevectorizeInfo.h - Target revectorization info ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_TARGETREVECTORIZEINFO_H
#define LLVM_ANALYSIS_TARGETREVECTORIZEINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InstructionCost.h"
#include <functional>

namespace llvm {

class Function;
class Instruction;
class IRBuilderBase;
class TargetTransformInfo;
class Type;
class Value;

/// The type-erased interface implemented by targets that support
/// revectorization.
class LLVM_ABI TargetRevectorizeInfoImplBase {
protected:
  const TargetTransformInfo &TTI;

public:
  explicit TargetRevectorizeInfoImplBase(const TargetTransformInfo &TTI)
      : TTI(TTI) {}
  virtual ~TargetRevectorizeInfoImplBase();

  /// Return true if the target intrinsic can be revectorized.
  virtual bool isTargetIntrinsicVectorizable(Intrinsic::ID ID) const {
    return false;
  }

  virtual InstructionCost
  getTargetIntrinsicVectorizationCost(Intrinsic::ID FromID, Type *WideRetTy,
                                      ArrayRef<Type *> WideArgTys,
                                      ElementCount VF) const;

  virtual Instruction *vectorizeTargetIntrinsic(Intrinsic::ID FromID,
                                                ArrayRef<Type *> TysForDecl,
                                                ArrayRef<Value *> WideArgs,
                                                ElementCount VF,
                                                IRBuilderBase &Builder) const;
};

/// Provides access to target-specific revectorization hooks without exposing
/// the target's subtarget type.
class LLVM_ABI TargetRevectorizeInfo {
public:
  explicit TargetRevectorizeInfo(const TargetTransformInfo &TTI);
  explicit TargetRevectorizeInfo(
      std::unique_ptr<const TargetRevectorizeInfoImplBase> Impl);

  TargetRevectorizeInfo(TargetRevectorizeInfo &&);
  TargetRevectorizeInfo &operator=(TargetRevectorizeInfo &&);
  ~TargetRevectorizeInfo();

  bool invalidate(Function &F, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &Inv);

  /// Return whether the target intrinsic \p ID can be vectorised. This is meant
  /// as an early legality check, and it should be checked further for a given
  /// VF using \p getTargetIntrinsicVectorizationCost().
  bool isTargetIntrinsicVectorizable(Intrinsic::ID ID) const;

  /// Return the cost of vectorizing a target intrinsic with its argument types
  /// and return type widened by \p VF. The cost might be invalid.
  ///
  /// \pre isTargetIntrinsicVectorizable(FromID)
  InstructionCost
  getTargetIntrinsicVectorizationCost(Intrinsic::ID FromID, Type *WideRetTy,
                                      ArrayRef<Type *> WideArgTys,
                                      ElementCount VF) const;

  /// Return the vectorized intrinsic call after its operands have been
  /// widened for a given \p VF.
  ///
  /// \p FromID The original intrinsic ID that need vectorizing.
  /// \p TysForDecl The widened types used to overload FromID
  /// \p WideArgs The arguments to the intrinsic widened by \p VF
  Instruction *vectorizeTargetIntrinsic(Intrinsic::ID FromID,
                                        ArrayRef<Type *> TysForDecl,
                                        ArrayRef<Value *> WideArgs,
                                        ElementCount VF,
                                        IRBuilderBase &Builder) const;

private:
  std::unique_ptr<const TargetRevectorizeInfoImplBase> Impl;
};

using TRVI = TargetRevectorizeInfo;

class LLVM_ABI TargetRevectorizeWrapper
    : public AnalysisInfoMixin<TargetRevectorizeWrapper> {
public:
  using Result = TargetRevectorizeInfo;

  TargetRevectorizeWrapper();
  explicit TargetRevectorizeWrapper(
      std::function<Result(const Function &, const TargetTransformInfo &)>
          Callback);

  Result run(Function &F, FunctionAnalysisManager &);

private:
  friend AnalysisInfoMixin<TargetRevectorizeWrapper>;
  static AnalysisKey Key;

  std::function<Result(const Function &, const TargetTransformInfo &)> Callback;
};

} // namespace llvm

#endif // LLVM_ANALYSIS_TARGETREVECTORIZEINFO_H
