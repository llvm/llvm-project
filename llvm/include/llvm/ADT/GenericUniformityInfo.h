//===- GenericUniformityInfo.h ---------------------------*- C++ -*--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_GENERICUNIFORMITYINFO_H
#define LLVM_ADT_GENERICUNIFORMITYINFO_H

// #include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GenericCycleInfo.h"
// #include "llvm/ADT/SmallPtrSet.h"
// #include "llvm/ADT/Uniformity.h"
// #include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class TargetTransformInfo;

template <typename ContextT> class GenericUniformityAnalysisImpl;
template <typename ImplT> struct GenericUniformityAnalysisImplDeleter {
  // Ugly hack around the fact that recent (> 15.0) clang will run into an
  // is_invocable() check in some GNU libc++'s unique_ptr implementation
  // and reject this deleter if you just make it callable with an ImplT *,
  // whether or not the type of ImplT is spelled out.
  using pointer = ImplT *;
  void operator()(ImplT *Impl);
};

template <typename ContextT> class GenericUniformityInfo {
public:
  using BlockT = typename ContextT::BlockT;
  using FunctionT = typename ContextT::FunctionT;
  using ValueRefT = typename ContextT::ValueRefT;
  using ConstValueRefT = typename ContextT::ConstValueRefT;
  using InstructionT = typename ContextT::InstructionT;
  using DominatorTreeT = typename ContextT::DominatorTreeT;
  using ThisT = GenericUniformityInfo<ContextT>;

  using CycleInfoT = GenericCycleInfo<ContextT>;
  using CycleT = typename CycleInfoT::CycleT;

  GenericUniformityInfo(FunctionT &F, const DominatorTreeT &DT,
                        const CycleInfoT &CI,
                        const TargetTransformInfo *TTI = nullptr);
  GenericUniformityInfo() = default;
  GenericUniformityInfo(GenericUniformityInfo &&) = default;
  GenericUniformityInfo &operator=(GenericUniformityInfo &&) = default;

  /// Whether any divergence was detected.
  bool hasDivergence() const;

  /// The GPU kernel this analysis result is for
  const FunctionT &getFunction() const { return *F; }

  /// Whether \p V is divergent at its definition.
  bool isDivergent(ConstValueRefT V) const;

  /// Whether \p V is uniform/non-divergent.
  bool isUniform(ConstValueRefT V) const { return !isDivergent(V); }

  bool hasDivergentTerminator(const BlockT &B);

  void print(raw_ostream &Out) const;

private:
  using ImplT = GenericUniformityAnalysisImpl<ContextT>;

  FunctionT *F;
  std::unique_ptr<ImplT, GenericUniformityAnalysisImplDeleter<ImplT>> DA;

  GenericUniformityInfo(const GenericUniformityInfo &) = delete;
  GenericUniformityInfo &operator=(const GenericUniformityInfo &) = delete;
};

} // namespace llvm

#endif // LLVM_ADT_GENERICUNIFORMITYINFO_H
