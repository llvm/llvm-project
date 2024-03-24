//===- IPO/OpenMPOpt.h - Collection of OpenMP optimizations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_OPENMPOPT_H
#define LLVM_TRANSFORMS_IPO_OPENMPOPT_H

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/IPO/Attributor.h"

namespace llvm {

namespace omp {

/// Summary of a kernel (=entry point for target offloading).
using Kernel = Function *;

/// Set of kernels in the module
using KernelSet = SetVector<Kernel>;

/// Helper to determine if \p M contains OpenMP.
bool containsOpenMP(Module &M);

/// Helper to determine if \p M is a OpenMP target offloading device module.
bool isOpenMPDevice(Module &M);

/// Return true iff \p Fn is an OpenMP GPU kernel; \p Fn has the "kernel"
/// attribute.
bool isOpenMPKernel(Function &Fn);

/// Get OpenMP device kernels in \p M.
KernelSet getDeviceKernels(Module &M);

} // namespace omp

/// OpenMP optimizations pass.
class OpenMPOptPass : public PassInfoMixin<OpenMPOptPass> {
public:
  OpenMPOptPass() = default;
  OpenMPOptPass(ThinOrFullLTOPhase LTOPhase) : LTOPhase(LTOPhase) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  const ThinOrFullLTOPhase LTOPhase = ThinOrFullLTOPhase::None;
};

class OpenMPOptCGSCCPass : public PassInfoMixin<OpenMPOptCGSCCPass> {
public:
  OpenMPOptCGSCCPass() = default;
  OpenMPOptCGSCCPass(ThinOrFullLTOPhase LTOPhase) : LTOPhase(LTOPhase) {}

  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &UR);

private:
  const ThinOrFullLTOPhase LTOPhase = ThinOrFullLTOPhase::None;
};

template <typename Ty, bool InsertInvalidates = true>
struct BooleanStateWithSetVector : public BooleanState {
  bool contains(const Ty &Elem) const { return Set.contains(Elem); }
  bool insert(const Ty &Elem) {
    if (InsertInvalidates)
      BooleanState::indicatePessimisticFixpoint();
    return Set.insert(Elem);
  }

  const Ty &operator[](int Idx) const { return Set[Idx]; }
  bool operator==(const BooleanStateWithSetVector &RHS) const {
    return BooleanState::operator==(RHS) && Set == RHS.Set;
  }
  bool operator!=(const BooleanStateWithSetVector &RHS) const {
    return !(*this == RHS);
  }

  bool empty() const { return Set.empty(); }
  size_t size() const { return Set.size(); }

  /// "Clamp" this state with \p RHS.
  BooleanStateWithSetVector &operator^=(const BooleanStateWithSetVector &RHS) {
    BooleanState::operator^=(RHS);
    Set.insert(RHS.Set.begin(), RHS.Set.end());
    return *this;
  }

private:
  /// A set to keep track of elements.
  SetVector<Ty> Set;

public:
  typename decltype(Set)::iterator begin() { return Set.begin(); }
  typename decltype(Set)::iterator end() { return Set.end(); }
  typename decltype(Set)::const_iterator begin() const { return Set.begin(); }
  typename decltype(Set)::const_iterator end() const { return Set.end(); }
};

template <typename Ty, bool InsertInvalidates = true>
using BooleanStateWithPtrSetVector =
    BooleanStateWithSetVector<Ty *, InsertInvalidates>;

struct KernelInfoState : AbstractState {
  /// Flag to track if we reached a fixpoint.
  bool IsAtFixpoint = false;

  /// The parallel regions (identified by the outlined parallel functions) that
  /// can be reached from the associated function.
  BooleanStateWithPtrSetVector<CallBase, /* InsertInvalidates */ false>
      ReachedKnownParallelRegions;

  /// State to track what parallel region we might reach.
  BooleanStateWithPtrSetVector<CallBase> ReachedUnknownParallelRegions;

  /// State to track if we are in SPMD-mode, assumed or know, and why we decided
  /// we cannot be. If it is assumed, then RequiresFullRuntime should also be
  /// false.
  BooleanStateWithPtrSetVector<Instruction, false> SPMDCompatibilityTracker;

  /// The __kmpc_target_init call in this kernel, if any. If we find more than
  /// one we abort as the kernel is malformed.
  CallBase *KernelInitCB = nullptr;

  /// The constant kernel environement as taken from and passed to
  /// __kmpc_target_init.
  ConstantStruct *KernelEnvC = nullptr;

  /// The __kmpc_target_deinit call in this kernel, if any. If we find more than
  /// one we abort as the kernel is malformed.
  CallBase *KernelDeinitCB = nullptr;

  /// Flag to indicate if the associated function is a kernel entry.
  bool IsKernelEntry = false;

  /// State to track what kernel entries can reach the associated function.
  BooleanStateWithPtrSetVector<Function, false> ReachingKernelEntries;

  /// State to indicate if we can track parallel level of the associated
  /// function. We will give up tracking if we encounter unknown caller or the
  /// caller is __kmpc_parallel_51.
  BooleanStateWithSetVector<uint8_t> ParallelLevels;

  /// Flag that indicates if the kernel has nested Parallelism
  bool NestedParallelism = false;

  /// Abstract State interface
  ///{

  KernelInfoState() = default;
  KernelInfoState(bool BestState) {
    if (!BestState)
      indicatePessimisticFixpoint();
  }

  /// See AbstractState::isValidState(...)
  bool isValidState() const override { return true; }

  /// See AbstractState::isAtFixpoint(...)
  bool isAtFixpoint() const override { return IsAtFixpoint; }

  /// See AbstractState::indicatePessimisticFixpoint(...)
  ChangeStatus indicatePessimisticFixpoint() override {
    IsAtFixpoint = true;
    ParallelLevels.indicatePessimisticFixpoint();
    ReachingKernelEntries.indicatePessimisticFixpoint();
    SPMDCompatibilityTracker.indicatePessimisticFixpoint();
    ReachedKnownParallelRegions.indicatePessimisticFixpoint();
    ReachedUnknownParallelRegions.indicatePessimisticFixpoint();
    NestedParallelism = true;
    return ChangeStatus::CHANGED;
  }

  /// See AbstractState::indicateOptimisticFixpoint(...)
  ChangeStatus indicateOptimisticFixpoint() override {
    IsAtFixpoint = true;
    ParallelLevels.indicateOptimisticFixpoint();
    ReachingKernelEntries.indicateOptimisticFixpoint();
    SPMDCompatibilityTracker.indicateOptimisticFixpoint();
    ReachedKnownParallelRegions.indicateOptimisticFixpoint();
    ReachedUnknownParallelRegions.indicateOptimisticFixpoint();
    return ChangeStatus::UNCHANGED;
  }

  /// Return the assumed state
  KernelInfoState &getAssumed() { return *this; }
  const KernelInfoState &getAssumed() const { return *this; }

  bool operator==(const KernelInfoState &RHS) const {
    if (SPMDCompatibilityTracker != RHS.SPMDCompatibilityTracker)
      return false;
    if (ReachedKnownParallelRegions != RHS.ReachedKnownParallelRegions)
      return false;
    if (ReachedUnknownParallelRegions != RHS.ReachedUnknownParallelRegions)
      return false;
    if (ReachingKernelEntries != RHS.ReachingKernelEntries)
      return false;
    if (ParallelLevels != RHS.ParallelLevels)
      return false;
    if (NestedParallelism != RHS.NestedParallelism)
      return false;
    return true;
  }

  /// Returns true if this kernel contains any OpenMP parallel regions.
  bool mayContainParallelRegion() {
    return !ReachedKnownParallelRegions.empty() ||
           !ReachedUnknownParallelRegions.empty();
  }

  /// Return empty set as the best state of potential values.
  static KernelInfoState getBestState() { return KernelInfoState(true); }

  static KernelInfoState getBestState(KernelInfoState &KIS) {
    return getBestState();
  }

  /// Return full set as the worst state of potential values.
  static KernelInfoState getWorstState() { return KernelInfoState(false); }

  /// "Clamp" this state with \p KIS.
  KernelInfoState operator^=(const KernelInfoState &KIS) {
    // Do not merge two different _init and _deinit call sites.
    if (KIS.KernelInitCB) {
      if (KernelInitCB && KernelInitCB != KIS.KernelInitCB)
        llvm_unreachable("Kernel that calls another kernel violates OpenMP-Opt "
                         "assumptions.");
      KernelInitCB = KIS.KernelInitCB;
    }
    if (KIS.KernelDeinitCB) {
      if (KernelDeinitCB && KernelDeinitCB != KIS.KernelDeinitCB)
        llvm_unreachable("Kernel that calls another kernel violates OpenMP-Opt "
                         "assumptions.");
      KernelDeinitCB = KIS.KernelDeinitCB;
    }
    if (KIS.KernelEnvC) {
      if (KernelEnvC && KernelEnvC != KIS.KernelEnvC)
        llvm_unreachable("Kernel that calls another kernel violates OpenMP-Opt "
                         "assumptions.");
      KernelEnvC = KIS.KernelEnvC;
    }
    SPMDCompatibilityTracker ^= KIS.SPMDCompatibilityTracker;
    ReachedKnownParallelRegions ^= KIS.ReachedKnownParallelRegions;
    ReachedUnknownParallelRegions ^= KIS.ReachedUnknownParallelRegions;
    NestedParallelism |= KIS.NestedParallelism;
    return *this;
  }

  KernelInfoState operator&=(const KernelInfoState &KIS) {
    return (*this ^= KIS);
  }
};

struct AAKernelInfo : public StateWrapper<KernelInfoState, AbstractAttribute> {
  using Base = StateWrapper<KernelInfoState, AbstractAttribute>;
  AAKernelInfo(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Public getter for ReachingKernelEntries
  virtual BooleanStateWithPtrSetVector<Function, false>
  getReachingKernels() = 0;

  /// Create an abstract attribute biew for the position \p IRP.
  static AAKernelInfo &createForPosition(const IRPosition &IRP, Attributor &A);

  /// This function should return true if the type of the \p AA is AAKernelInfo
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAKernelInfo"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  static const char ID;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_OPENMPOPT_H
