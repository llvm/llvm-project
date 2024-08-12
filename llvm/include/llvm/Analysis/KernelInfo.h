//=- KernelInfo.h - Kernel Analysis -------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the KernelInfo, KernelInfoAnalysis, and KernelInfoPrinter
// classes used to extract function properties from a GPU kernel.
//
// See llvm/docs/KernelInfo.rst.
// ===---------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_KERNELINFO_H
#define LLVM_ANALYSIS_KERNELINFO_H

#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetTransformInfo.h"

namespace llvm {
class DominatorTree;
class Function;

/// Data structure holding function info for kernels.
class KernelInfo {
  void updateForBB(const BasicBlock &BB, int64_t Direction,
                   OptimizationRemarkEmitter &ORE,
                   const TargetTransformInfo &TTI);

public:
  static KernelInfo getKernelInfo(Function &F, FunctionAnalysisManager &FAM);

  bool operator==(const KernelInfo &FPI) const {
    return std::memcmp(this, &FPI, sizeof(KernelInfo)) == 0;
  }

  bool operator!=(const KernelInfo &FPI) const { return !(*this == FPI); }

  /// If false, nothing was recorded here because the supplied function didn't
  /// appear in a module compiled for a GPU.
  bool IsValid = false;

  /// Whether the function has external linkage and is not a kernel function.
  bool ExternalNotKernel = false;

  /// OpenMP Launch bounds.
  ///@{
  std::optional<int64_t> OmpTargetNumTeams;
  std::optional<int64_t> OmpTargetThreadLimit;
  ///@}

  /// AMDGPU launch bounds.
  ///@{
  std::optional<int64_t> AmdgpuMaxNumWorkgroupsX;
  std::optional<int64_t> AmdgpuMaxNumWorkgroupsY;
  std::optional<int64_t> AmdgpuMaxNumWorkgroupsZ;
  std::optional<int64_t> AmdgpuFlatWorkGroupSizeMin;
  std::optional<int64_t> AmdgpuFlatWorkGroupSizeMax;
  std::optional<int64_t> AmdgpuWavesPerEuMin;
  std::optional<int64_t> AmdgpuWavesPerEuMax;
  ///@}

  /// NVPTX launch bounds.
  ///@{
  std::optional<int64_t> Maxclusterrank;
  std::optional<int64_t> Maxntidx;
  ///@}

  /// The number of alloca instructions inside the function, the number of those
  /// with allocation sizes that cannot be determined at compile time, and the
  /// sum of the sizes that can be.
  ///
  /// With the current implementation for at least some GPU archs,
  /// AllocasDyn > 0 might not be possible, but we report AllocasDyn anyway in
  /// case the implementation changes.
  int64_t Allocas = 0;
  int64_t AllocasDyn = 0;
  int64_t AllocasStaticSizeSum = 0;

  /// Number of direct/indirect calls (anything derived from CallBase).
  int64_t DirectCalls = 0;
  int64_t IndirectCalls = 0;

  /// Number of direct calls made from this function to other functions
  /// defined in this module.
  int64_t DirectCallsToDefinedFunctions = 0;

  /// Number of calls of type InvokeInst.
  int64_t Invokes = 0;

  /// Number of addrspace(0) memory accesses (via load, store, etc.).
  int64_t AddrspaceZeroAccesses = 0;
};

/// Analysis class for KernelInfo.
class KernelInfoAnalysis : public AnalysisInfoMixin<KernelInfoAnalysis> {
public:
  static AnalysisKey Key;

  using Result = const KernelInfo;

  KernelInfo run(Function &F, FunctionAnalysisManager &FAM) {
    return KernelInfo::getKernelInfo(F, FAM);
  }
};

/// Printer pass for KernelInfoAnalysis.
///
/// It just calls KernelInfoAnalysis, which prints remarks if they are enabled.
class KernelInfoPrinter : public PassInfoMixin<KernelInfoPrinter> {
public:
  explicit KernelInfoPrinter() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    AM.getResult<KernelInfoAnalysis>(F);
    return PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};
} // namespace llvm
#endif // LLVM_ANALYSIS_KERNELINFO_H
