//===- TargetTransformInfo.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This pass exposes codegen information to IR-level passes. Every
/// transformation that uses codegen information is broken into three parts:
/// 1. The IR-level analysis pass.
/// 2. The IR-level transformation interface which provides the needed
///    information.
/// 3. Codegen-level implementation which uses target-specific hooks.
///
/// This file defines #2, which is the interface that IR-level transformations
/// use for querying the codegen.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_TARGETTRANSFORMINFO_H
#define LLVM_ANALYSIS_TARGETTRANSFORMINFO_H

#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/InstructionCost.h"
#include <functional>

namespace llvm {

namespace Intrinsic {
typedef unsigned ID;
}

class AssumptionCache;
class BlockFrequencyInfo;
class DominatorTree;
class BranchInst;
class CallBase;
class ExtractElementInst;
class Function;
class GlobalValue;
class InstCombiner;
class OptimizationRemarkEmitter;
class IntrinsicInst;
class LoadInst;
class LoopAccessInfo;
class Loop;
class LoopInfo;
class ProfileSummaryInfo;
class RecurrenceDescriptor;
class SCEV;
class ScalarEvolution;
class StoreInst;
class SwitchInst;
class TargetLibraryInfo;
class Type;
class User;
class Value;
class VPIntrinsic;
struct KnownBits;
template <typename T> class Optional;

/// Information about a load/store intrinsic defined by the target.
struct MemIntrinsicInfo {
  /// This is the pointer that the intrinsic is loading from or storing to.
  /// If this is non-null, then analysis/optimization passes can assume that
  /// this intrinsic is functionally equivalent to a load/store from this
  /// pointer.
  Value *PtrVal = nullptr;

  // Ordering for atomic operations.
  AtomicOrdering Ordering = AtomicOrdering::NotAtomic;

  // Same Id is set by the target for corresponding load/store intrinsics.
  unsigned short MatchingId = 0;

  bool ReadMem = false;
  bool WriteMem = false;
  bool IsVolatile = false;

  bool isUnordered() const {
    return (Ordering == AtomicOrdering::NotAtomic ||
            Ordering == AtomicOrdering::Unordered) &&
           !IsVolatile;
  }
};

/// Attributes of a target dependent hardware loop.
struct HardwareLoopInfo {
  HardwareLoopInfo() = delete;
  HardwareLoopInfo(Loop *L) : L(L) {}
  Loop *L = nullptr;
  BasicBlock *ExitBlock = nullptr;
  BranchInst *ExitBranch = nullptr;
  const SCEV *ExitCount = nullptr;
  IntegerType *CountType = nullptr;
  Value *LoopDecrement = nullptr; // Decrement the loop counter by this
                                  // value in every iteration.
  bool IsNestingLegal = false;    // Can a hardware loop be a parent to
                                  // another hardware loop?
  bool CounterInReg = false;      // Should loop counter be updated in
                                  // the loop via a phi?
  bool PerformEntryTest = false;  // Generate the intrinsic which also performs
                                  // icmp ne zero on the loop counter value and
                                  // produces an i1 to guard the loop entry.
  bool isHardwareLoopCandidate(ScalarEvolution &SE, LoopInfo &LI,
                               DominatorTree &DT, bool ForceNestedLoop = false,
                               bool ForceHardwareLoopPHI = false);
  bool canAnalyze(LoopInfo &LI);
};

class IntrinsicCostAttributes {
  const IntrinsicInst *II = nullptr;
  Type *RetTy = nullptr;
  Intrinsic::ID IID;
  SmallVector<Type *, 4> ParamTys;
  SmallVector<const Value *, 4> Arguments;
  FastMathFlags FMF;
  // If ScalarizationCost is UINT_MAX, the cost of scalarizing the
  // arguments and the return value will be computed based on types.
  InstructionCost ScalarizationCost = InstructionCost::getInvalid();

public:
  IntrinsicCostAttributes(
      Intrinsic::ID Id, const CallBase &CI,
      InstructionCost ScalarCost = InstructionCost::getInvalid());

  IntrinsicCostAttributes(
      Intrinsic::ID Id, Type *RTy, ArrayRef<Type *> Tys,
      FastMathFlags Flags = FastMathFlags(), const IntrinsicInst *I = nullptr,
      InstructionCost ScalarCost = InstructionCost::getInvalid());

  IntrinsicCostAttributes(Intrinsic::ID Id, Type *RTy,
                          ArrayRef<const Value *> Args);

  IntrinsicCostAttributes(
      Intrinsic::ID Id, Type *RTy, ArrayRef<const Value *> Args,
      ArrayRef<Type *> Tys, FastMathFlags Flags = FastMathFlags(),
      const IntrinsicInst *I = nullptr,
      InstructionCost ScalarCost = InstructionCost::getInvalid());

  Intrinsic::ID getID() const { return IID; }
  const IntrinsicInst *getInst() const { return II; }
  Type *getReturnType() const { return RetTy; }
  FastMathFlags getFlags() const { return FMF; }
  InstructionCost getScalarizationCost() const { return ScalarizationCost; }
  const SmallVectorImpl<const Value *> &getArgs() const { return Arguments; }
  const SmallVectorImpl<Type *> &getArgTypes() const { return ParamTys; }

  bool isTypeBasedOnly() const {
    return Arguments.empty();
  }

  bool skipScalarizationCost() const { return ScalarizationCost.isValid(); }
};

class TargetTransformInfo;
typedef TargetTransformInfo TTI;

/// This pass provides access to the codegen interfaces that are needed
/// for IR-level transformations.
class TargetTransformInfo {
public:
  /// Construct a TTI object using a type implementing the \c Concept
  /// API below.
  ///
  /// This is used by targets to construct a TTI wrapping their target-specific
  /// implementation that encodes appropriate costs for their target.
  template <typename T> TargetTransformInfo(T Impl);

  /// Construct a baseline TTI object using a minimal implementation of
  /// the \c Concept API below.
  ///
  /// The TTI implementation will reflect the information in the DataLayout
  /// provided if non-null.
  explicit TargetTransformInfo(const DataLayout &DL);

  // Provide move semantics.
  TargetTransformInfo(TargetTransformInfo &&Arg);
  TargetTransformInfo &operator=(TargetTransformInfo &&RHS);

  // We need to define the destructor out-of-line to define our sub-classes
  // out-of-line.
  ~TargetTransformInfo();

  /// Handle the invalidation of this information.
  ///
  /// When used as a result of \c TargetIRAnalysis this method will be called
  /// when the function this was computed for changes. When it returns false,
  /// the information is preserved across those changes.
  bool invalidate(Function &, const PreservedAnalyses &,
                  FunctionAnalysisManager::Invalidator &) {
    // FIXME: We should probably in some way ensure that the subtarget
    // information for a function hasn't changed.
    return false;
  }

  /// \name Generic Target Information
  /// @{

  /// The kind of cost model.
  ///
  /// There are several different cost models that can be customized by the
  /// target. The normalization of each cost model may be target specific.
  enum TargetCostKind {
    TCK_RecipThroughput, ///< Reciprocal throughput.
    TCK_Latency,         ///< The latency of instruction.
    TCK_CodeSize,        ///< Instruction code size.
    TCK_SizeAndLatency   ///< The weighted sum of size and latency.
  };

  /// Query the cost of a specified instruction.
  ///
  /// Clients should use this interface to query the cost of an existing
  /// instruction. The instruction must have a valid parent (basic block).
  ///
  /// Note, this method does not cache the cost calculation and it
  /// can be expensive in some cases.
  InstructionCost getInstructionCost(const Instruction *I,
                                     enum TargetCostKind kind) const {
    InstructionCost Cost;
    switch (kind) {
    case TCK_RecipThroughput:
      Cost = getInstructionThroughput(I);
      break;
    case TCK_Latency:
      Cost = getInstructionLatency(I);
      break;
    case TCK_CodeSize:
    case TCK_SizeAndLatency:
      Cost = getUserCost(I, kind);
      break;
    }
    return Cost;
  }

  /// Underlying constants for 'cost' values in this interface.
  ///
  /// Many APIs in this interface return a cost. This enum defines the
  /// fundamental values that should be used to interpret (and produce) those
  /// costs. The costs are returned as an int rather than a member of this
  /// enumeration because it is expected that the cost of one IR instruction
  /// may have a multiplicative factor to it or otherwise won't fit directly
  /// into the enum. Moreover, it is common to sum or average costs which works
  /// better as simple integral values. Thus this enum only provides constants.
  /// Also note that the returned costs are signed integers to make it natural
  /// to add, subtract, and test with zero (a common boundary condition). It is
  /// not expected that 2^32 is a realistic cost to be modeling at any point.
  ///
  /// Note that these costs should usually reflect the intersection of code-size
  /// cost and execution cost. A free instruction is typically one that folds
  /// into another instruction. For example, reg-to-reg moves can often be
  /// skipped by renaming the registers in the CPU, but they still are encoded
  /// and thus wouldn't be considered 'free' here.
  enum TargetCostConstants {
    TCC_Free = 0,     ///< Expected to fold away in lowering.
    TCC_Basic = 1,    ///< The cost of a typical 'add' instruction.
    TCC_Expensive = 4 ///< The cost of a 'div' instruction on x86.
  };

  /// Estimate the cost of a GEP operation when lowered.
  InstructionCost
  getGEPCost(Type *PointeeType, const Value *Ptr,
             ArrayRef<const Value *> Operands,
             TargetCostKind CostKind = TCK_SizeAndLatency) const;

  /// \returns A value by which our inlining threshold should be multiplied.
  /// This is primarily used to bump up the inlining threshold wholesale on
  /// targets where calls are unusually expensive.
  ///
  /// TODO: This is a rather blunt instrument.  Perhaps altering the costs of
  /// individual classes of instructions would be better.
  unsigned getInliningThresholdMultiplier() const;

  /// \returns A value to be added to the inlining threshold.
  unsigned adjustInliningThreshold(const CallBase *CB) const;

  /// \returns Vector bonus in percent.
  ///
  /// Vector bonuses: We want to more aggressively inline vector-dense kernels
  /// and apply this bonus based on the percentage of vector instructions. A
  /// bonus is applied if the vector instructions exceed 50% and half that
  /// amount is applied if it exceeds 10%. Note that these bonuses are some what
  /// arbitrary and evolved over time by accident as much as because they are
  /// principled bonuses.
  /// FIXME: It would be nice to base the bonus values on something more
  /// scientific. A target may has no bonus on vector instructions.
  int getInlinerVectorBonusPercent() const;

  /// \return the expected cost of a memcpy, which could e.g. depend on the
  /// source/destination type and alignment and the number of bytes copied.
  InstructionCost getMemcpyCost(const Instruction *I) const;

  /// \return The estimated number of case clusters when lowering \p 'SI'.
  /// \p JTSize Set a jump table size only when \p SI is suitable for a jump
  /// table.
  unsigned getEstimatedNumberOfCaseClusters(const SwitchInst &SI,
                                            unsigned &JTSize,
                                            ProfileSummaryInfo *PSI,
                                            BlockFrequencyInfo *BFI) const;

  /// Estimate the cost of a given IR user when lowered.
  ///
  /// This can estimate the cost of either a ConstantExpr or Instruction when
  /// lowered.
  ///
  /// \p Operands is a list of operands which can be a result of transformations
  /// of the current operands. The number of the operands on the list must equal
  /// to the number of the current operands the IR user has. Their order on the
  /// list must be the same as the order of the current operands the IR user
  /// has.
  ///
  /// The returned cost is defined in terms of \c TargetCostConstants, see its
  /// comments for a detailed explanation of the cost values.
  InstructionCost getUserCost(const User *U, ArrayRef<const Value *> Operands,
                              TargetCostKind CostKind) const;

  /// This is a helper function which calls the two-argument getUserCost
  /// with \p Operands which are the current operands U has.
  InstructionCost getUserCost(const User *U, TargetCostKind CostKind) const {
    SmallVector<const Value *, 4> Operands(U->operand_values());
    return getUserCost(U, Operands, CostKind);
  }

  /// If a branch or a select condition is skewed in one direction by more than
  /// this factor, it is very likely to be predicted correctly.
  BranchProbability getPredictableBranchThreshold() const;

  /// Return true if branch divergence exists.
  ///
  /// Branch divergence has a significantly negative impact on GPU performance
  /// when threads in the same wavefront take different paths due to conditional
  /// branches.
  bool hasBranchDivergence() const;

  /// Return true if the target prefers to use GPU divergence analysis to
  /// replace the legacy version.
  bool useGPUDivergenceAnalysis() const;

  /// Returns whether V is a source of divergence.
  ///
  /// This function provides the target-dependent information for
  /// the target-independent LegacyDivergenceAnalysis. LegacyDivergenceAnalysis
  /// first builds the dependency graph, and then runs the reachability
  /// algorithm starting with the sources of divergence.
  bool isSourceOfDivergence(const Value *V) const;

  // Returns true for the target specific
  // set of operations which produce uniform result
  // even taking non-uniform arguments
  bool isAlwaysUniform(const Value *V) const;

  /// Returns the address space ID for a target's 'flat' address space. Note
  /// this is not necessarily the same as addrspace(0), which LLVM sometimes
  /// refers to as the generic address space. The flat address space is a
  /// generic address space that can be used access multiple segments of memory
  /// with different address spaces. Access of a memory location through a
  /// pointer with this address space is expected to be legal but slower
  /// compared to the same memory location accessed through a pointer with a
  /// different address space.
  //
  /// This is for targets with different pointer representations which can
  /// be converted with the addrspacecast instruction. If a pointer is converted
  /// to this address space, optimizations should attempt to replace the access
  /// with the source address space.
  ///
  /// \returns ~0u if the target does not have such a flat address space to
  /// optimize away.
  unsigned getFlatAddressSpace() const;

  /// Return any intrinsic address operand indexes which may be rewritten if
  /// they use a flat address space pointer.
  ///
  /// \returns true if the intrinsic was handled.
  bool collectFlatAddressOperands(SmallVectorImpl<int> &OpIndexes,
                                  Intrinsic::ID IID) const;

  bool isNoopAddrSpaceCast(unsigned FromAS, unsigned ToAS) const;

  /// Return true if globals in this address space can have initializers other
  /// than `undef`.
  bool canHaveNonUndefGlobalInitializerInAddressSpace(unsigned AS) const;

  unsigned getAssumedAddrSpace(const Value *V) const;

  /// Rewrite intrinsic call \p II such that \p OldV will be replaced with \p
  /// NewV, which has a different address space. This should happen for every
  /// operand index that collectFlatAddressOperands returned for the intrinsic.
  /// \returns nullptr if the intrinsic was not handled. Otherwise, returns the
  /// new value (which may be the original \p II with modified operands).
  Value *rewriteIntrinsicWithAddressSpace(IntrinsicInst *II, Value *OldV,
                                          Value *NewV) const;

  /// Test whether calls to a function lower to actual program function
  /// calls.
  ///
  /// The idea is to test whether the program is likely to require a 'call'
  /// instruction or equivalent in order to call the given function.
  ///
  /// FIXME: It's not clear that this is a good or useful query API. Client's
  /// should probably move to simpler cost metrics using the above.
  /// Alternatively, we could split the cost interface into distinct code-size
  /// and execution-speed costs. This would allow modelling the core of this
  /// query more accurately as a call is a single small instruction, but
  /// incurs significant execution cost.
  bool isLoweredToCall(const Function *F) const;

  struct LSRCost {
    /// TODO: Some of these could be merged. Also, a lexical ordering
    /// isn't always optimal.
    unsigned Insns;
    unsigned NumRegs;
    unsigned AddRecCost;
    unsigned NumIVMuls;
    unsigned NumBaseAdds;
    unsigned ImmCost;
    unsigned SetupCost;
    unsigned ScaleCost;
  };

  /// Parameters that control the generic loop unrolling transformation.
  struct UnrollingPreferences {
    /// The cost threshold for the unrolled loop. Should be relative to the
    /// getUserCost values returned by this API, and the expectation is that
    /// the unrolled loop's instructions when run through that interface should
    /// not exceed this cost. However, this is only an estimate. Also, specific
    /// loops may be unrolled even with a cost above this threshold if deemed
    /// profitable. Set this to UINT_MAX to disable the loop body cost
    /// restriction.
    unsigned Threshold;
    /// If complete unrolling will reduce the cost of the loop, we will boost
    /// the Threshold by a certain percent to allow more aggressive complete
    /// unrolling. This value provides the maximum boost percentage that we
    /// can apply to Threshold (The value should be no less than 100).
    /// BoostedThreshold = Threshold * min(RolledCost / UnrolledCost,
    ///                                    MaxPercentThresholdBoost / 100)
    /// E.g. if complete unrolling reduces the loop execution time by 50%
    /// then we boost the threshold by the factor of 2x. If unrolling is not
    /// expected to reduce the running time, then we do not increase the
    /// threshold.
    unsigned MaxPercentThresholdBoost;
    /// The cost threshold for the unrolled loop when optimizing for size (set
    /// to UINT_MAX to disable).
    unsigned OptSizeThreshold;
    /// The cost threshold for the unrolled loop, like Threshold, but used
    /// for partial/runtime unrolling (set to UINT_MAX to disable).
    unsigned PartialThreshold;
    /// The cost threshold for the unrolled loop when optimizing for size, like
    /// OptSizeThreshold, but used for partial/runtime unrolling (set to
    /// UINT_MAX to disable).
    unsigned PartialOptSizeThreshold;
    /// A forced unrolling factor (the number of concatenated bodies of the
    /// original loop in the unrolled loop body). When set to 0, the unrolling
    /// transformation will select an unrolling factor based on the current cost
    /// threshold and other factors.
    unsigned Count;
    /// Default unroll count for loops with run-time trip count.
    unsigned DefaultUnrollRuntimeCount;
    // Set the maximum unrolling factor. The unrolling factor may be selected
    // using the appropriate cost threshold, but may not exceed this number
    // (set to UINT_MAX to disable). This does not apply in cases where the
    // loop is being fully unrolled.
    unsigned MaxCount;
    /// Set the maximum unrolling factor for full unrolling. Like MaxCount, but
    /// applies even if full unrolling is selected. This allows a target to fall
    /// back to Partial unrolling if full unrolling is above FullUnrollMaxCount.
    unsigned FullUnrollMaxCount;
    // Represents number of instructions optimized when "back edge"
    // becomes "fall through" in unrolled loop.
    // For now we count a conditional branch on a backedge and a comparison
    // feeding it.
    unsigned BEInsns;
    /// Allow partial unrolling (unrolling of loops to expand the size of the
    /// loop body, not only to eliminate small constant-trip-count loops).
    bool Partial;
    /// Allow runtime unrolling (unrolling of loops to expand the size of the
    /// loop body even when the number of loop iterations is not known at
    /// compile time).
    bool Runtime;
    /// Allow generation of a loop remainder (extra iterations after unroll).
    bool AllowRemainder;
    /// Allow emitting expensive instructions (such as divisions) when computing
    /// the trip count of a loop for runtime unrolling.
    bool AllowExpensiveTripCount;
    /// Apply loop unroll on any kind of loop
    /// (mainly to loops that fail runtime unrolling).
    bool Force;
    /// Allow using trip count upper bound to unroll loops.
    bool UpperBound;
    /// Allow unrolling of all the iterations of the runtime loop remainder.
    bool UnrollRemainder;
    /// Allow unroll and jam. Used to enable unroll and jam for the target.
    bool UnrollAndJam;
    /// Threshold for unroll and jam, for inner loop size. The 'Threshold'
    /// value above is used during unroll and jam for the outer loop size.
    /// This value is used in the same manner to limit the size of the inner
    /// loop.
    unsigned UnrollAndJamInnerLoopThreshold;
    /// Don't allow loop unrolling to simulate more than this number of
    /// iterations when checking full unroll profitability
    unsigned MaxIterationsCountToAnalyze;
  };

  /// Get target-customized preferences for the generic loop unrolling
  /// transformation. The caller will initialize UP with the current
  /// target-independent defaults.
  void getUnrollingPreferences(Loop *L, ScalarEvolution &,
                               UnrollingPreferences &UP,
                               OptimizationRemarkEmitter *ORE) const;

  /// Query the target whether it would be profitable to convert the given loop
  /// into a hardware loop.
  bool isHardwareLoopProfitable(Loop *L, ScalarEvolution &SE,
                                AssumptionCache &AC, TargetLibraryInfo *LibInfo,
                                HardwareLoopInfo &HWLoopInfo) const;

  /// Query the target whether it would be prefered to create a predicated
  /// vector loop, which can avoid the need to emit a scalar epilogue loop.
  bool preferPredicateOverEpilogue(Loop *L, LoopInfo *LI, ScalarEvolution &SE,
                                   AssumptionCache &AC, TargetLibraryInfo *TLI,
                                   DominatorTree *DT,
                                   const LoopAccessInfo *LAI) const;

  /// Query the target whether lowering of the llvm.get.active.lane.mask
  /// intrinsic is supported.
  bool emitGetActiveLaneMask() const;

  // Parameters that control the loop peeling transformation
  struct PeelingPreferences {
    /// A forced peeling factor (the number of bodied of the original loop
    /// that should be peeled off before the loop body). When set to 0, the
    /// a peeling factor based on profile information and other factors.
    unsigned PeelCount;
    /// Allow peeling off loop iterations.
    bool AllowPeeling;
    /// Allow peeling off loop iterations for loop nests.
    bool AllowLoopNestsPeeling;
    /// Allow peeling basing on profile. Uses to enable peeling off all
    /// iterations basing on provided profile.
    /// If the value is true the peeling cost model can decide to peel only
    /// some iterations and in this case it will set this to false.
    bool PeelProfiledIterations;
  };

  /// Get target-customized preferences for the generic loop peeling
  /// transformation. The caller will initialize \p PP with the current
  /// target-independent defaults with information from \p L and \p SE.
  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             PeelingPreferences &PP) const;

  /// Targets can implement their own combinations for target-specific
  /// intrinsics. This function will be called from the InstCombine pass every
  /// time a target-specific intrinsic is encountered.
  ///
  /// \returns None to not do anything target specific or a value that will be
  /// returned from the InstCombiner. It is possible to return null and stop
  /// further processing of the intrinsic by returning nullptr.
  Optional<Instruction *> instCombineIntrinsic(InstCombiner &IC,
                                               IntrinsicInst &II) const;
  /// Can be used to implement target-specific instruction combining.
  /// \see instCombineIntrinsic
  Optional<Value *>
  simplifyDemandedUseBitsIntrinsic(InstCombiner &IC, IntrinsicInst &II,
                                   APInt DemandedMask, KnownBits &Known,
                                   bool &KnownBitsComputed) const;
  /// Can be used to implement target-specific instruction combining.
  /// \see instCombineIntrinsic
  Optional<Value *> simplifyDemandedVectorEltsIntrinsic(
      InstCombiner &IC, IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
      APInt &UndefElts2, APInt &UndefElts3,
      std::function<void(Instruction *, unsigned, APInt, APInt &)>
          SimplifyAndSetOp) const;
  /// @}

  /// \name Scalar Target Information
  /// @{

  /// Flags indicating the kind of support for population count.
  ///
  /// Compared to the SW implementation, HW support is supposed to
  /// significantly boost the performance when the population is dense, and it
  /// may or may not degrade performance if the population is sparse. A HW
  /// support is considered as "Fast" if it can outperform, or is on a par
  /// with, SW implementation when the population is sparse; otherwise, it is
  /// considered as "Slow".
  enum PopcntSupportKind { PSK_Software, PSK_SlowHardware, PSK_FastHardware };

  /// Return true if the specified immediate is legal add immediate, that
  /// is the target has add instructions which can add a register with the
  /// immediate without having to materialize the immediate into a register.
  bool isLegalAddImmediate(int64_t Imm) const;

  /// Return true if the specified immediate is legal icmp immediate,
  /// that is the target has icmp instructions which can compare a register
  /// against the immediate without having to materialize the immediate into a
  /// register.
  bool isLegalICmpImmediate(int64_t Imm) const;

  /// Return true if the addressing mode represented by AM is legal for
  /// this target, for a load/store of the specified type.
  /// The type may be VoidTy, in which case only return true if the addressing
  /// mode is legal for a load/store of any legal type.
  /// If target returns true in LSRWithInstrQueries(), I may be valid.
  /// TODO: Handle pre/postinc as well.
  bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                             bool HasBaseReg, int64_t Scale,
                             unsigned AddrSpace = 0,
                             Instruction *I = nullptr) const;

  /// Return true if LSR cost of C1 is lower than C1.
  bool isLSRCostLess(TargetTransformInfo::LSRCost &C1,
                     TargetTransformInfo::LSRCost &C2) const;

  /// Return true if LSR major cost is number of registers. Targets which
  /// implement their own isLSRCostLess and unset number of registers as major
  /// cost should return false, otherwise return true.
  bool isNumRegsMajorCostOfLSR() const;

  /// \returns true if LSR should not optimize a chain that includes \p I.
  bool isProfitableLSRChainElement(Instruction *I) const;

  /// Return true if the target can fuse a compare and branch.
  /// Loop-strength-reduction (LSR) uses that knowledge to adjust its cost
  /// calculation for the instructions in a loop.
  bool canMacroFuseCmp() const;

  /// Return true if the target can save a compare for loop count, for example
  /// hardware loop saves a compare.
  bool canSaveCmp(Loop *L, BranchInst **BI, ScalarEvolution *SE, LoopInfo *LI,
                  DominatorTree *DT, AssumptionCache *AC,
                  TargetLibraryInfo *LibInfo) const;

  enum AddressingModeKind {
    AMK_PreIndexed,
    AMK_PostIndexed,
    AMK_None
  };

  /// Return the preferred addressing mode LSR should make efforts to generate.
  AddressingModeKind getPreferredAddressingMode(const Loop *L,
                                                ScalarEvolution *SE) const;

  /// Return true if the target supports masked store.
  bool isLegalMaskedStore(Type *DataType, Align Alignment) const;
  /// Return true if the target supports masked load.
  bool isLegalMaskedLoad(Type *DataType, Align Alignment) const;

  /// Return true if the target supports nontemporal store.
  bool isLegalNTStore(Type *DataType, Align Alignment) const;
  /// Return true if the target supports nontemporal load.
  bool isLegalNTLoad(Type *DataType, Align Alignment) const;

  /// Return true if the target supports masked scatter.
  bool isLegalMaskedScatter(Type *DataType, Align Alignment) const;
  /// Return true if the target supports masked gather.
  bool isLegalMaskedGather(Type *DataType, Align Alignment) const;

  /// Return true if the target supports masked compress store.
  bool isLegalMaskedCompressStore(Type *DataType) const;
  /// Return true if the target supports masked expand load.
  bool isLegalMaskedExpandLoad(Type *DataType) const;

  /// Return true if we should be enabling ordered reductions for the target.
  bool enableOrderedReductions() const;

  /// Return true if the target has a unified operation to calculate division
  /// and remainder. If so, the additional implicit multiplication and
  /// subtraction required to calculate a remainder from division are free. This
  /// can enable more aggressive transformations for division and remainder than
  /// would typically be allowed using throughput or size cost models.
  bool hasDivRemOp(Type *DataType, bool IsSigned) const;

  /// Return true if the given instruction (assumed to be a memory access
  /// instruction) has a volatile variant. If that's the case then we can avoid
  /// addrspacecast to generic AS for volatile loads/stores. Default
  /// implementation returns false, which prevents address space inference for
  /// volatile loads/stores.
  bool hasVolatileVariant(Instruction *I, unsigned AddrSpace) const;

  /// Return true if target doesn't mind addresses in vectors.
  bool prefersVectorizedAddressing() const;

  /// Return the cost of the scaling factor used in the addressing
  /// mode represented by AM for this target, for a load/store
  /// of the specified type.
  /// If the AM is supported, the return value must be >= 0.
  /// If the AM is not supported, it returns a negative value.
  /// TODO: Handle pre/postinc as well.
  InstructionCost getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                                       int64_t BaseOffset, bool HasBaseReg,
                                       int64_t Scale,
                                       unsigned AddrSpace = 0) const;

  /// Return true if the loop strength reduce pass should make
  /// Instruction* based TTI queries to isLegalAddressingMode(). This is
  /// needed on SystemZ, where e.g. a memcpy can only have a 12 bit unsigned
  /// immediate offset and no index register.
  bool LSRWithInstrQueries() const;

  /// Return true if it's free to truncate a value of type Ty1 to type
  /// Ty2. e.g. On x86 it's free to truncate a i32 value in register EAX to i16
  /// by referencing its sub-register AX.
  bool isTruncateFree(Type *Ty1, Type *Ty2) const;

  /// Return true if it is profitable to hoist instruction in the
  /// then/else to before if.
  bool isProfitableToHoist(Instruction *I) const;

  bool useAA() const;

  /// Return true if this type is legal.
  bool isTypeLegal(Type *Ty) const;

  /// Returns the estimated number of registers required to represent \p Ty.
  InstructionCost getRegUsageForType(Type *Ty) const;

  /// Return true if switches should be turned into lookup tables for the
  /// target.
  bool shouldBuildLookupTables() const;

  /// Return true if switches should be turned into lookup tables
  /// containing this constant value for the target.
  bool shouldBuildLookupTablesForConstant(Constant *C) const;

  /// Return true if lookup tables should be turned into relative lookup tables.
  bool shouldBuildRelLookupTables() const;

  /// Return true if the input function which is cold at all call sites,
  ///  should use coldcc calling convention.
  bool useColdCCForColdCall(Function &F) const;

  /// Estimate the overhead of scalarizing an instruction. Insert and Extract
  /// are set if the demanded result elements need to be inserted and/or
  /// extracted from vectors.
  InstructionCost getScalarizationOverhead(VectorType *Ty,
                                           const APInt &DemandedElts,
                                           bool Insert, bool Extract) const;

  /// Estimate the overhead of scalarizing an instructions unique
  /// non-constant operands. The (potentially vector) types to use for each of
  /// argument are passes via Tys.
  InstructionCost getOperandsScalarizationOverhead(ArrayRef<const Value *> Args,
                                                   ArrayRef<Type *> Tys) const;

  /// If target has efficient vector element load/store instructions, it can
  /// return true here so that insertion/extraction costs are not added to
  /// the scalarization cost of a load/store.
  bool supportsEfficientVectorElementLoadStore() const;

  /// Don't restrict interleaved unrolling to small loops.
  bool enableAggressiveInterleaving(bool LoopHasReductions) const;

  /// Returns options for expansion of memcmp. IsZeroCmp is
  // true if this is the expansion of memcmp(p1, p2, s) == 0.
  struct MemCmpExpansionOptions {
    // Return true if memcmp expansion is enabled.
    operator bool() const { return MaxNumLoads > 0; }

    // Maximum number of load operations.
    unsigned MaxNumLoads = 0;

    // The list of available load sizes (in bytes), sorted in decreasing order.
    SmallVector<unsigned, 8> LoadSizes;

    // For memcmp expansion when the memcmp result is only compared equal or
    // not-equal to 0, allow up to this number of load pairs per block. As an
    // example, this may allow 'memcmp(a, b, 3) == 0' in a single block:
    //   a0 = load2bytes &a[0]
    //   b0 = load2bytes &b[0]
    //   a2 = load1byte  &a[2]
    //   b2 = load1byte  &b[2]
    //   r  = cmp eq (a0 ^ b0 | a2 ^ b2), 0
    unsigned NumLoadsPerBlock = 1;

    // Set to true to allow overlapping loads. For example, 7-byte compares can
    // be done with two 4-byte compares instead of 4+2+1-byte compares. This
    // requires all loads in LoadSizes to be doable in an unaligned way.
    bool AllowOverlappingLoads = false;
  };
  MemCmpExpansionOptions enableMemCmpExpansion(bool OptSize,
                                               bool IsZeroCmp) const;

  /// Enable matching of interleaved access groups.
  bool enableInterleavedAccessVectorization() const;

  /// Enable matching of interleaved access groups that contain predicated
  /// accesses or gaps and therefore vectorized using masked
  /// vector loads/stores.
  bool enableMaskedInterleavedAccessVectorization() const;

  /// Indicate that it is potentially unsafe to automatically vectorize
  /// floating-point operations because the semantics of vector and scalar
  /// floating-point semantics may differ. For example, ARM NEON v7 SIMD math
  /// does not support IEEE-754 denormal numbers, while depending on the
  /// platform, scalar floating-point math does.
  /// This applies to floating-point math operations and calls, not memory
  /// operations, shuffles, or casts.
  bool isFPVectorizationPotentiallyUnsafe() const;

  /// Determine if the target supports unaligned memory accesses.
  bool allowsMisalignedMemoryAccesses(LLVMContext &Context, unsigned BitWidth,
                                      unsigned AddressSpace = 0,
                                      Align Alignment = Align(1),
                                      bool *Fast = nullptr) const;

  /// Return hardware support for population count.
  PopcntSupportKind getPopcntSupport(unsigned IntTyWidthInBit) const;

  /// Return true if the hardware has a fast square-root instruction.
  bool haveFastSqrt(Type *Ty) const;

  /// Return true if it is faster to check if a floating-point value is NaN
  /// (or not-NaN) versus a comparison against a constant FP zero value.
  /// Targets should override this if materializing a 0.0 for comparison is
  /// generally as cheap as checking for ordered/unordered.
  bool isFCmpOrdCheaperThanFCmpZero(Type *Ty) const;

  /// Return the expected cost of supporting the floating point operation
  /// of the specified type.
  InstructionCost getFPOpCost(Type *Ty) const;

  /// Return the expected cost of materializing for the given integer
  /// immediate of the specified type.
  InstructionCost getIntImmCost(const APInt &Imm, Type *Ty,
                                TargetCostKind CostKind) const;

  /// Return the expected cost of materialization for the given integer
  /// immediate of the specified type for a given instruction. The cost can be
  /// zero if the immediate can be folded into the specified instruction.
  InstructionCost getIntImmCostInst(unsigned Opc, unsigned Idx,
                                    const APInt &Imm, Type *Ty,
                                    TargetCostKind CostKind,
                                    Instruction *Inst = nullptr) const;
  InstructionCost getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx,
                                      const APInt &Imm, Type *Ty,
                                      TargetCostKind CostKind) const;

  /// Return the expected cost for the given integer when optimising
  /// for size. This is different than the other integer immediate cost
  /// functions in that it is subtarget agnostic. This is useful when you e.g.
  /// target one ISA such as Aarch32 but smaller encodings could be possible
  /// with another such as Thumb. This return value is used as a penalty when
  /// the total costs for a constant is calculated (the bigger the cost, the
  /// more beneficial constant hoisting is).
  InstructionCost getIntImmCodeSizeCost(unsigned Opc, unsigned Idx,
                                        const APInt &Imm, Type *Ty) const;
  /// @}

  /// \name Vector Target Information
  /// @{

  /// The various kinds of shuffle patterns for vector queries.
  enum ShuffleKind {
    SK_Broadcast,        ///< Broadcast element 0 to all other elements.
    SK_Reverse,          ///< Reverse the order of the vector.
    SK_Select,           ///< Selects elements from the corresponding lane of
                         ///< either source operand. This is equivalent to a
                         ///< vector select with a constant condition operand.
    SK_Transpose,        ///< Transpose two vectors.
    SK_InsertSubvector,  ///< InsertSubvector. Index indicates start offset.
    SK_ExtractSubvector, ///< ExtractSubvector Index indicates start offset.
    SK_PermuteTwoSrc,    ///< Merge elements from two source vectors into one
                         ///< with any shuffle mask.
    SK_PermuteSingleSrc, ///< Shuffle elements of single source vector with any
                         ///< shuffle mask.
    SK_Splice            ///< Concatenates elements from the first input vector
                         ///< with elements of the second input vector. Returning
                         ///< a vector of the same type as the input vectors.
  };

  /// Additional information about an operand's possible values.
  enum OperandValueKind {
    OK_AnyValue,               // Operand can have any value.
    OK_UniformValue,           // Operand is uniform (splat of a value).
    OK_UniformConstantValue,   // Operand is uniform constant.
    OK_NonUniformConstantValue // Operand is a non uniform constant value.
  };

  /// Additional properties of an operand's values.
  enum OperandValueProperties { OP_None = 0, OP_PowerOf2 = 1 };

  /// \return the number of registers in the target-provided register class.
  unsigned getNumberOfRegisters(unsigned ClassID) const;

  /// \return the target-provided register class ID for the provided type,
  /// accounting for type promotion and other type-legalization techniques that
  /// the target might apply. However, it specifically does not account for the
  /// scalarization or splitting of vector types. Should a vector type require
  /// scalarization or splitting into multiple underlying vector registers, that
  /// type should be mapped to a register class containing no registers.
  /// Specifically, this is designed to provide a simple, high-level view of the
  /// register allocation later performed by the backend. These register classes
  /// don't necessarily map onto the register classes used by the backend.
  /// FIXME: It's not currently possible to determine how many registers
  /// are used by the provided type.
  unsigned getRegisterClassForType(bool Vector, Type *Ty = nullptr) const;

  /// \return the target-provided register class name
  const char *getRegisterClassName(unsigned ClassID) const;

  enum RegisterKind { RGK_Scalar, RGK_FixedWidthVector, RGK_ScalableVector };

  /// \return The width of the largest scalar or vector register type.
  TypeSize getRegisterBitWidth(RegisterKind K) const;

  /// \return The width of the smallest vector register type.
  unsigned getMinVectorRegisterBitWidth() const;

  /// \return The maximum value of vscale if the target specifies an
  ///  architectural maximum vector length, and None otherwise.
  Optional<unsigned> getMaxVScale() const;

  /// \return True if the vectorization factor should be chosen to
  /// make the vector of the smallest element type match the size of a
  /// vector register. For wider element types, this could result in
  /// creating vectors that span multiple vector registers.
  /// If false, the vectorization factor will be chosen based on the
  /// size of the widest element type.
  bool shouldMaximizeVectorBandwidth() const;

  /// \return The minimum vectorization factor for types of given element
  /// bit width, or 0 if there is no minimum VF. The returned value only
  /// applies when shouldMaximizeVectorBandwidth returns true.
  /// If IsScalable is true, the returned ElementCount must be a scalable VF.
  ElementCount getMinimumVF(unsigned ElemWidth, bool IsScalable) const;

  /// \return The maximum vectorization factor for types of given element
  /// bit width and opcode, or 0 if there is no maximum VF.
  /// Currently only used by the SLP vectorizer.
  unsigned getMaximumVF(unsigned ElemWidth, unsigned Opcode) const;

  /// \return True if it should be considered for address type promotion.
  /// \p AllowPromotionWithoutCommonHeader Set true if promoting \p I is
  /// profitable without finding other extensions fed by the same input.
  bool shouldConsiderAddressTypePromotion(
      const Instruction &I, bool &AllowPromotionWithoutCommonHeader) const;

  /// \return The size of a cache line in bytes.
  unsigned getCacheLineSize() const;

  /// The possible cache levels
  enum class CacheLevel {
    L1D, // The L1 data cache
    L2D, // The L2 data cache

    // We currently do not model L3 caches, as their sizes differ widely between
    // microarchitectures. Also, we currently do not have a use for L3 cache
    // size modeling yet.
  };

  /// \return The size of the cache level in bytes, if available.
  Optional<unsigned> getCacheSize(CacheLevel Level) const;

  /// \return The associativity of the cache level, if available.
  Optional<unsigned> getCacheAssociativity(CacheLevel Level) const;

  /// \return How much before a load we should place the prefetch
  /// instruction.  This is currently measured in number of
  /// instructions.
  unsigned getPrefetchDistance() const;

  /// Some HW prefetchers can handle accesses up to a certain constant stride.
  /// Sometimes prefetching is beneficial even below the HW prefetcher limit,
  /// and the arguments provided are meant to serve as a basis for deciding this
  /// for a particular loop.
  ///
  /// \param NumMemAccesses        Number of memory accesses in the loop.
  /// \param NumStridedMemAccesses Number of the memory accesses that
  ///                              ScalarEvolution could find a known stride
  ///                              for.
  /// \param NumPrefetches         Number of software prefetches that will be
  ///                              emitted as determined by the addresses
  ///                              involved and the cache line size.
  /// \param HasCall               True if the loop contains a call.
  ///
  /// \return This is the minimum stride in bytes where it makes sense to start
  ///         adding SW prefetches. The default is 1, i.e. prefetch with any
  ///         stride.
  unsigned getMinPrefetchStride(unsigned NumMemAccesses,
                                unsigned NumStridedMemAccesses,
                                unsigned NumPrefetches, bool HasCall) const;

  /// \return The maximum number of iterations to prefetch ahead.  If
  /// the required number of iterations is more than this number, no
  /// prefetching is performed.
  unsigned getMaxPrefetchIterationsAhead() const;

  /// \return True if prefetching should also be done for writes.
  bool enableWritePrefetching() const;

  /// \return The maximum interleave factor that any transform should try to
  /// perform for this target. This number depends on the level of parallelism
  /// and the number of execution units in the CPU.
  unsigned getMaxInterleaveFactor(unsigned VF) const;

  /// Collect properties of V used in cost analysis, e.g. OP_PowerOf2.
  static OperandValueKind getOperandInfo(const Value *V,
                                         OperandValueProperties &OpProps);

  /// This is an approximation of reciprocal throughput of a math/logic op.
  /// A higher cost indicates less expected throughput.
  /// From Agner Fog's guides, reciprocal throughput is "the average number of
  /// clock cycles per instruction when the instructions are not part of a
  /// limiting dependency chain."
  /// Therefore, costs should be scaled to account for multiple execution units
  /// on the target that can process this type of instruction. For example, if
  /// there are 5 scalar integer units and 2 vector integer units that can
  /// calculate an 'add' in a single cycle, this model should indicate that the
  /// cost of the vector add instruction is 2.5 times the cost of the scalar
  /// add instruction.
  /// \p Args is an optional argument which holds the instruction operands
  /// values so the TTI can analyze those values searching for special
  /// cases or optimizations based on those values.
  /// \p CxtI is the optional original context instruction, if one exists, to
  /// provide even more information.
  InstructionCost getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
      OperandValueKind Opd1Info = OK_AnyValue,
      OperandValueKind Opd2Info = OK_AnyValue,
      OperandValueProperties Opd1PropInfo = OP_None,
      OperandValueProperties Opd2PropInfo = OP_None,
      ArrayRef<const Value *> Args = ArrayRef<const Value *>(),
      const Instruction *CxtI = nullptr) const;

  /// \return The cost of a shuffle instruction of kind Kind and of type Tp.
  /// The exact mask may be passed as Mask, or else the array will be empty.
  /// The index and subtype parameters are used by the subvector insertion and
  /// extraction shuffle kinds to show the insert/extract point and the type of
  /// the subvector being inserted/extracted.
  /// NOTE: For subvector extractions Tp represents the source type.
  InstructionCost getShuffleCost(ShuffleKind Kind, VectorType *Tp,
                                 ArrayRef<int> Mask = None, int Index = 0,
                                 VectorType *SubTp = nullptr) const;

  /// Represents a hint about the context in which a cast is used.
  ///
  /// For zext/sext, the context of the cast is the operand, which must be a
  /// load of some kind. For trunc, the context is of the cast is the single
  /// user of the instruction, which must be a store of some kind.
  ///
  /// This enum allows the vectorizer to give getCastInstrCost an idea of the
  /// type of cast it's dealing with, as not every cast is equal. For instance,
  /// the zext of a load may be free, but the zext of an interleaving load can
  //// be (very) expensive!
  ///
  /// See \c getCastContextHint to compute a CastContextHint from a cast
  /// Instruction*. Callers can use it if they don't need to override the
  /// context and just want it to be calculated from the instruction.
  ///
  /// FIXME: This handles the types of load/store that the vectorizer can
  /// produce, which are the cases where the context instruction is most
  /// likely to be incorrect. There are other situations where that can happen
  /// too, which might be handled here but in the long run a more general
  /// solution of costing multiple instructions at the same times may be better.
  enum class CastContextHint : uint8_t {
    None,          ///< The cast is not used with a load/store of any kind.
    Normal,        ///< The cast is used with a normal load/store.
    Masked,        ///< The cast is used with a masked load/store.
    GatherScatter, ///< The cast is used with a gather/scatter.
    Interleave,    ///< The cast is used with an interleaved load/store.
    Reversed,      ///< The cast is used with a reversed load/store.
  };

  /// Calculates a CastContextHint from \p I.
  /// This should be used by callers of getCastInstrCost if they wish to
  /// determine the context from some instruction.
  /// \returns the CastContextHint for ZExt/SExt/Trunc, None if \p I is nullptr,
  /// or if it's another type of cast.
  static CastContextHint getCastContextHint(const Instruction *I);

  /// \return The expected cost of cast instructions, such as bitcast, trunc,
  /// zext, etc. If there is an existing instruction that holds Opcode, it
  /// may be passed in the 'I' parameter.
  InstructionCost
  getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                   TTI::CastContextHint CCH,
                   TTI::TargetCostKind CostKind = TTI::TCK_SizeAndLatency,
                   const Instruction *I = nullptr) const;

  /// \return The expected cost of a sign- or zero-extended vector extract. Use
  /// -1 to indicate that there is no information about the index value.
  InstructionCost getExtractWithExtendCost(unsigned Opcode, Type *Dst,
                                           VectorType *VecTy,
                                           unsigned Index = -1) const;

  /// \return The expected cost of control-flow related instructions such as
  /// Phi, Ret, Br, Switch.
  InstructionCost
  getCFInstrCost(unsigned Opcode,
                 TTI::TargetCostKind CostKind = TTI::TCK_SizeAndLatency,
                 const Instruction *I = nullptr) const;

  /// \returns The expected cost of compare and select instructions. If there
  /// is an existing instruction that holds Opcode, it may be passed in the
  /// 'I' parameter. The \p VecPred parameter can be used to indicate the select
  /// is using a compare with the specified predicate as condition. When vector
  /// types are passed, \p VecPred must be used for all lanes.
  InstructionCost
  getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                     CmpInst::Predicate VecPred,
                     TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
                     const Instruction *I = nullptr) const;

  /// \return The expected cost of vector Insert and Extract.
  /// Use -1 to indicate that there is no information on the index value.
  InstructionCost getVectorInstrCost(unsigned Opcode, Type *Val,
                                     unsigned Index = -1) const;

  /// \return The cost of Load and Store instructions.
  InstructionCost
  getMemoryOpCost(unsigned Opcode, Type *Src, Align Alignment,
                  unsigned AddressSpace,
                  TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
                  const Instruction *I = nullptr) const;

  /// \return The cost of masked Load and Store instructions.
  InstructionCost getMaskedMemoryOpCost(
      unsigned Opcode, Type *Src, Align Alignment, unsigned AddressSpace,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput) const;

  /// \return The cost of Gather or Scatter operation
  /// \p Opcode - is a type of memory access Load or Store
  /// \p DataTy - a vector type of the data to be loaded or stored
  /// \p Ptr - pointer [or vector of pointers] - address[es] in memory
  /// \p VariableMask - true when the memory access is predicated with a mask
  ///                   that is not a compile-time constant
  /// \p Alignment - alignment of single element
  /// \p I - the optional original context instruction, if one exists, e.g. the
  ///        load/store to transform or the call to the gather/scatter intrinsic
  InstructionCost getGatherScatterOpCost(
      unsigned Opcode, Type *DataTy, const Value *Ptr, bool VariableMask,
      Align Alignment, TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
      const Instruction *I = nullptr) const;

  /// \return The cost of the interleaved memory operation.
  /// \p Opcode is the memory operation code
  /// \p VecTy is the vector type of the interleaved access.
  /// \p Factor is the interleave factor
  /// \p Indices is the indices for interleaved load members (as interleaved
  ///    load allows gaps)
  /// \p Alignment is the alignment of the memory operation
  /// \p AddressSpace is address space of the pointer.
  /// \p UseMaskForCond indicates if the memory access is predicated.
  /// \p UseMaskForGaps indicates if gaps should be masked.
  InstructionCost getInterleavedMemoryOpCost(
      unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
      Align Alignment, unsigned AddressSpace,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput,
      bool UseMaskForCond = false, bool UseMaskForGaps = false) const;

  /// A helper function to determine the type of reduction algorithm used
  /// for a given \p Opcode and set of FastMathFlags \p FMF.
  static bool requiresOrderedReduction(Optional<FastMathFlags> FMF) {
    return FMF != None && !(*FMF).allowReassoc();
  }

  /// Calculate the cost of vector reduction intrinsics.
  ///
  /// This is the cost of reducing the vector value of type \p Ty to a scalar
  /// value using the operation denoted by \p Opcode. The FastMathFlags
  /// parameter \p FMF indicates what type of reduction we are performing:
  ///   1. Tree-wise. This is the typical 'fast' reduction performed that
  ///   involves successively splitting a vector into half and doing the
  ///   operation on the pair of halves until you have a scalar value. For
  ///   example:
  ///     (v0, v1, v2, v3)
  ///     ((v0+v2), (v1+v3), undef, undef)
  ///     ((v0+v2+v1+v3), undef, undef, undef)
  ///   This is the default behaviour for integer operations, whereas for
  ///   floating point we only do this if \p FMF indicates that
  ///   reassociation is allowed.
  ///   2. Ordered. For a vector with N elements this involves performing N
  ///   operations in lane order, starting with an initial scalar value, i.e.
  ///     result = InitVal + v0
  ///     result = result + v1
  ///     result = result + v2
  ///     result = result + v3
  ///   This is only the case for FP operations and when reassociation is not
  ///   allowed.
  ///
  InstructionCost getArithmeticReductionCost(
      unsigned Opcode, VectorType *Ty, Optional<FastMathFlags> FMF,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput) const;

  InstructionCost getMinMaxReductionCost(
      VectorType *Ty, VectorType *CondTy, bool IsUnsigned,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput) const;

  /// Calculate the cost of an extended reduction pattern, similar to
  /// getArithmeticReductionCost of an Add reduction with an extension and
  /// optional multiply. This is the cost of as:
  /// ResTy vecreduce.add(ext(Ty A)), or if IsMLA flag is set then:
  /// ResTy vecreduce.add(mul(ext(Ty A), ext(Ty B)). The reduction happens
  /// on a VectorType with ResTy elements and Ty lanes.
  InstructionCost getExtendedAddReductionCost(
      bool IsMLA, bool IsUnsigned, Type *ResTy, VectorType *Ty,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput) const;

  /// \returns The cost of Intrinsic instructions. Analyses the real arguments.
  /// Three cases are handled: 1. scalar instruction 2. vector instruction
  /// 3. scalar instruction which is to be vectorized.
  InstructionCost getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                        TTI::TargetCostKind CostKind) const;

  /// \returns The cost of Call instructions.
  InstructionCost getCallInstrCost(
      Function *F, Type *RetTy, ArrayRef<Type *> Tys,
      TTI::TargetCostKind CostKind = TTI::TCK_SizeAndLatency) const;

  /// \returns The number of pieces into which the provided type must be
  /// split during legalization. Zero is returned when the answer is unknown.
  unsigned getNumberOfParts(Type *Tp) const;

  /// \returns The cost of the address computation. For most targets this can be
  /// merged into the instruction indexing mode. Some targets might want to
  /// distinguish between address computation for memory operations on vector
  /// types and scalar types. Such targets should override this function.
  /// The 'SE' parameter holds pointer for the scalar evolution object which
  /// is used in order to get the Ptr step value in case of constant stride.
  /// The 'Ptr' parameter holds SCEV of the access pointer.
  InstructionCost getAddressComputationCost(Type *Ty,
                                            ScalarEvolution *SE = nullptr,
                                            const SCEV *Ptr = nullptr) const;

  /// \returns The cost, if any, of keeping values of the given types alive
  /// over a callsite.
  ///
  /// Some types may require the use of register classes that do not have
  /// any callee-saved registers, so would require a spill and fill.
  InstructionCost getCostOfKeepingLiveOverCall(ArrayRef<Type *> Tys) const;

  /// \returns True if the intrinsic is a supported memory intrinsic.  Info
  /// will contain additional information - whether the intrinsic may write
  /// or read to memory, volatility and the pointer.  Info is undefined
  /// if false is returned.
  bool getTgtMemIntrinsic(IntrinsicInst *Inst, MemIntrinsicInfo &Info) const;

  /// \returns The maximum element size, in bytes, for an element
  /// unordered-atomic memory intrinsic.
  unsigned getAtomicMemIntrinsicMaxElementSize() const;

  /// \returns A value which is the result of the given memory intrinsic.  New
  /// instructions may be created to extract the result from the given intrinsic
  /// memory operation.  Returns nullptr if the target cannot create a result
  /// from the given intrinsic.
  Value *getOrCreateResultFromMemIntrinsic(IntrinsicInst *Inst,
                                           Type *ExpectedType) const;

  /// \returns The type to use in a loop expansion of a memcpy call.
  Type *getMemcpyLoopLoweringType(LLVMContext &Context, Value *Length,
                                  unsigned SrcAddrSpace, unsigned DestAddrSpace,
                                  unsigned SrcAlign, unsigned DestAlign) const;

  /// \param[out] OpsOut The operand types to copy RemainingBytes of memory.
  /// \param RemainingBytes The number of bytes to copy.
  ///
  /// Calculates the operand types to use when copying \p RemainingBytes of
  /// memory, where source and destination alignments are \p SrcAlign and
  /// \p DestAlign respectively.
  void getMemcpyLoopResidualLoweringType(
      SmallVectorImpl<Type *> &OpsOut, LLVMContext &Context,
      unsigned RemainingBytes, unsigned SrcAddrSpace, unsigned DestAddrSpace,
      unsigned SrcAlign, unsigned DestAlign) const;

  /// \returns True if the two functions have compatible attributes for inlining
  /// purposes.
  bool areInlineCompatible(const Function *Caller,
                           const Function *Callee) const;

  /// \returns True if the caller and callee agree on how \p Args will be passed
  /// to the callee.
  /// \param[out] Args The list of compatible arguments.  The implementation may
  /// filter out any incompatible args from this list.
  bool areFunctionArgsABICompatible(const Function *Caller,
                                    const Function *Callee,
                                    SmallPtrSetImpl<Argument *> &Args) const;

  /// The type of load/store indexing.
  enum MemIndexedMode {
    MIM_Unindexed, ///< No indexing.
    MIM_PreInc,    ///< Pre-incrementing.
    MIM_PreDec,    ///< Pre-decrementing.
    MIM_PostInc,   ///< Post-incrementing.
    MIM_PostDec    ///< Post-decrementing.
  };

  /// \returns True if the specified indexed load for the given type is legal.
  bool isIndexedLoadLegal(enum MemIndexedMode Mode, Type *Ty) const;

  /// \returns True if the specified indexed store for the given type is legal.
  bool isIndexedStoreLegal(enum MemIndexedMode Mode, Type *Ty) const;

  /// \returns The bitwidth of the largest vector type that should be used to
  /// load/store in the given address space.
  unsigned getLoadStoreVecRegBitWidth(unsigned AddrSpace) const;

  /// \returns True if the load instruction is legal to vectorize.
  bool isLegalToVectorizeLoad(LoadInst *LI) const;

  /// \returns True if the store instruction is legal to vectorize.
  bool isLegalToVectorizeStore(StoreInst *SI) const;

  /// \returns True if it is legal to vectorize the given load chain.
  bool isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes, Align Alignment,
                                   unsigned AddrSpace) const;

  /// \returns True if it is legal to vectorize the given store chain.
  bool isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes, Align Alignment,
                                    unsigned AddrSpace) const;

  /// \returns True if it is legal to vectorize the given reduction kind.
  bool isLegalToVectorizeReduction(const RecurrenceDescriptor &RdxDesc,
                                   ElementCount VF) const;

  /// \returns True if the given type is supported for scalable vectors
  bool isElementTypeLegalForScalableVector(Type *Ty) const;

  /// \returns The new vector factor value if the target doesn't support \p
  /// SizeInBytes loads or has a better vector factor.
  unsigned getLoadVectorFactor(unsigned VF, unsigned LoadSize,
                               unsigned ChainSizeInBytes,
                               VectorType *VecTy) const;

  /// \returns The new vector factor value if the target doesn't support \p
  /// SizeInBytes stores or has a better vector factor.
  unsigned getStoreVectorFactor(unsigned VF, unsigned StoreSize,
                                unsigned ChainSizeInBytes,
                                VectorType *VecTy) const;

  /// Flags describing the kind of vector reduction.
  struct ReductionFlags {
    ReductionFlags() : IsMaxOp(false), IsSigned(false), NoNaN(false) {}
    bool IsMaxOp;  ///< If the op a min/max kind, true if it's a max operation.
    bool IsSigned; ///< Whether the operation is a signed int reduction.
    bool NoNaN;    ///< If op is an fp min/max, whether NaNs may be present.
  };

  /// \returns True if the target prefers reductions in loop.
  bool preferInLoopReduction(unsigned Opcode, Type *Ty,
                             ReductionFlags Flags) const;

  /// \returns True if the target prefers reductions select kept in the loop
  /// when tail folding. i.e.
  /// loop:
  ///   p = phi (0, s)
  ///   a = add (p, x)
  ///   s = select (mask, a, p)
  /// vecreduce.add(s)
  ///
  /// As opposed to the normal scheme of p = phi (0, a) which allows the select
  /// to be pulled out of the loop. If the select(.., add, ..) can be predicated
  /// by the target, this can lead to cleaner code generation.
  bool preferPredicatedReductionSelect(unsigned Opcode, Type *Ty,
                                       ReductionFlags Flags) const;

  /// \returns True if the target wants to expand the given reduction intrinsic
  /// into a shuffle sequence.
  bool shouldExpandReduction(const IntrinsicInst *II) const;

  /// \returns the size cost of rematerializing a GlobalValue address relative
  /// to a stack reload.
  unsigned getGISelRematGlobalCost() const;

  /// \returns True if the target supports scalable vectors.
  bool supportsScalableVectors() const;

  /// \name Vector Predication Information
  /// @{
  /// Whether the target supports the %evl parameter of VP intrinsic efficiently
  /// in hardware. (see LLVM Language Reference - "Vector Predication
  /// Intrinsics") Use of %evl is discouraged when that is not the case.
  bool hasActiveVectorLength() const;

  struct VPLegalization {
    enum VPTransform {
      // keep the predicating parameter
      Legal = 0,
      // where legal, discard the predicate parameter
      Discard = 1,
      // transform into something else that is also predicating
      Convert = 2
    };

    // How to transform the EVL parameter.
    // Legal:   keep the EVL parameter as it is.
    // Discard: Ignore the EVL parameter where it is safe to do so.
    // Convert: Fold the EVL into the mask parameter.
    VPTransform EVLParamStrategy;

    // How to transform the operator.
    // Legal:   The target supports this operator.
    // Convert: Convert this to a non-VP operation.
    // The 'Discard' strategy is invalid.
    VPTransform OpStrategy;

    bool shouldDoNothing() const {
      return (EVLParamStrategy == Legal) && (OpStrategy == Legal);
    }
    VPLegalization(VPTransform EVLParamStrategy, VPTransform OpStrategy)
        : EVLParamStrategy(EVLParamStrategy), OpStrategy(OpStrategy) {}
  };

  /// \returns How the target needs this vector-predicated operation to be
  /// transformed.
  VPLegalization getVPLegalizationStrategy(const VPIntrinsic &PI) const;
  /// @}

  /// @}

private:
  /// Estimate the latency of specified instruction.
  /// Returns 1 as the default value.
  InstructionCost getInstructionLatency(const Instruction *I) const;

  /// Returns the expected throughput cost of the instruction.
  /// Returns -1 if the cost is unknown.
  InstructionCost getInstructionThroughput(const Instruction *I) const;

  /// The abstract base class used to type erase specific TTI
  /// implementations.
  class Concept;

  /// The template model for the base class which wraps a concrete
  /// implementation in a type erased interface.
  template <typename T> class Model;

  std::unique_ptr<Concept> TTIImpl;
};

class TargetTransformInfo::Concept {
public:
  virtual ~Concept() = 0;
  virtual const DataLayout &getDataLayout() const = 0;
  virtual InstructionCost getGEPCost(Type *PointeeType, const Value *Ptr,
                                     ArrayRef<const Value *> Operands,
                                     TTI::TargetCostKind CostKind) = 0;
  virtual unsigned getInliningThresholdMultiplier() = 0;
  virtual unsigned adjustInliningThreshold(const CallBase *CB) = 0;
  virtual int getInlinerVectorBonusPercent() = 0;
  virtual InstructionCost getMemcpyCost(const Instruction *I) = 0;
  virtual unsigned
  getEstimatedNumberOfCaseClusters(const SwitchInst &SI, unsigned &JTSize,
                                   ProfileSummaryInfo *PSI,
                                   BlockFrequencyInfo *BFI) = 0;
  virtual InstructionCost getUserCost(const User *U,
                                      ArrayRef<const Value *> Operands,
                                      TargetCostKind CostKind) = 0;
  virtual BranchProbability getPredictableBranchThreshold() = 0;
  virtual bool hasBranchDivergence() = 0;
  virtual bool useGPUDivergenceAnalysis() = 0;
  virtual bool isSourceOfDivergence(const Value *V) = 0;
  virtual bool isAlwaysUniform(const Value *V) = 0;
  virtual unsigned getFlatAddressSpace() = 0;
  virtual bool collectFlatAddressOperands(SmallVectorImpl<int> &OpIndexes,
                                          Intrinsic::ID IID) const = 0;
  virtual bool isNoopAddrSpaceCast(unsigned FromAS, unsigned ToAS) const = 0;
  virtual bool
  canHaveNonUndefGlobalInitializerInAddressSpace(unsigned AS) const = 0;
  virtual unsigned getAssumedAddrSpace(const Value *V) const = 0;
  virtual Value *rewriteIntrinsicWithAddressSpace(IntrinsicInst *II,
                                                  Value *OldV,
                                                  Value *NewV) const = 0;
  virtual bool isLoweredToCall(const Function *F) = 0;
  virtual void getUnrollingPreferences(Loop *L, ScalarEvolution &,
                                       UnrollingPreferences &UP,
                                       OptimizationRemarkEmitter *ORE) = 0;
  virtual void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                                     PeelingPreferences &PP) = 0;
  virtual bool isHardwareLoopProfitable(Loop *L, ScalarEvolution &SE,
                                        AssumptionCache &AC,
                                        TargetLibraryInfo *LibInfo,
                                        HardwareLoopInfo &HWLoopInfo) = 0;
  virtual bool
  preferPredicateOverEpilogue(Loop *L, LoopInfo *LI, ScalarEvolution &SE,
                              AssumptionCache &AC, TargetLibraryInfo *TLI,
                              DominatorTree *DT, const LoopAccessInfo *LAI) = 0;
  virtual bool emitGetActiveLaneMask() = 0;
  virtual Optional<Instruction *> instCombineIntrinsic(InstCombiner &IC,
                                                       IntrinsicInst &II) = 0;
  virtual Optional<Value *>
  simplifyDemandedUseBitsIntrinsic(InstCombiner &IC, IntrinsicInst &II,
                                   APInt DemandedMask, KnownBits &Known,
                                   bool &KnownBitsComputed) = 0;
  virtual Optional<Value *> simplifyDemandedVectorEltsIntrinsic(
      InstCombiner &IC, IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
      APInt &UndefElts2, APInt &UndefElts3,
      std::function<void(Instruction *, unsigned, APInt, APInt &)>
          SimplifyAndSetOp) = 0;
  virtual bool isLegalAddImmediate(int64_t Imm) = 0;
  virtual bool isLegalICmpImmediate(int64_t Imm) = 0;
  virtual bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV,
                                     int64_t BaseOffset, bool HasBaseReg,
                                     int64_t Scale, unsigned AddrSpace,
                                     Instruction *I) = 0;
  virtual bool isLSRCostLess(TargetTransformInfo::LSRCost &C1,
                             TargetTransformInfo::LSRCost &C2) = 0;
  virtual bool isNumRegsMajorCostOfLSR() = 0;
  virtual bool isProfitableLSRChainElement(Instruction *I) = 0;
  virtual bool canMacroFuseCmp() = 0;
  virtual bool canSaveCmp(Loop *L, BranchInst **BI, ScalarEvolution *SE,
                          LoopInfo *LI, DominatorTree *DT, AssumptionCache *AC,
                          TargetLibraryInfo *LibInfo) = 0;
  virtual AddressingModeKind
    getPreferredAddressingMode(const Loop *L, ScalarEvolution *SE) const = 0;
  virtual bool isLegalMaskedStore(Type *DataType, Align Alignment) = 0;
  virtual bool isLegalMaskedLoad(Type *DataType, Align Alignment) = 0;
  virtual bool isLegalNTStore(Type *DataType, Align Alignment) = 0;
  virtual bool isLegalNTLoad(Type *DataType, Align Alignment) = 0;
  virtual bool isLegalMaskedScatter(Type *DataType, Align Alignment) = 0;
  virtual bool isLegalMaskedGather(Type *DataType, Align Alignment) = 0;
  virtual bool isLegalMaskedCompressStore(Type *DataType) = 0;
  virtual bool isLegalMaskedExpandLoad(Type *DataType) = 0;
  virtual bool enableOrderedReductions() = 0;
  virtual bool hasDivRemOp(Type *DataType, bool IsSigned) = 0;
  virtual bool hasVolatileVariant(Instruction *I, unsigned AddrSpace) = 0;
  virtual bool prefersVectorizedAddressing() = 0;
  virtual InstructionCost getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                                               int64_t BaseOffset,
                                               bool HasBaseReg, int64_t Scale,
                                               unsigned AddrSpace) = 0;
  virtual bool LSRWithInstrQueries() = 0;
  virtual bool isTruncateFree(Type *Ty1, Type *Ty2) = 0;
  virtual bool isProfitableToHoist(Instruction *I) = 0;
  virtual bool useAA() = 0;
  virtual bool isTypeLegal(Type *Ty) = 0;
  virtual InstructionCost getRegUsageForType(Type *Ty) = 0;
  virtual bool shouldBuildLookupTables() = 0;
  virtual bool shouldBuildLookupTablesForConstant(Constant *C) = 0;
  virtual bool shouldBuildRelLookupTables() = 0;
  virtual bool useColdCCForColdCall(Function &F) = 0;
  virtual InstructionCost getScalarizationOverhead(VectorType *Ty,
                                                   const APInt &DemandedElts,
                                                   bool Insert,
                                                   bool Extract) = 0;
  virtual InstructionCost
  getOperandsScalarizationOverhead(ArrayRef<const Value *> Args,
                                   ArrayRef<Type *> Tys) = 0;
  virtual bool supportsEfficientVectorElementLoadStore() = 0;
  virtual bool enableAggressiveInterleaving(bool LoopHasReductions) = 0;
  virtual MemCmpExpansionOptions
  enableMemCmpExpansion(bool OptSize, bool IsZeroCmp) const = 0;
  virtual bool enableInterleavedAccessVectorization() = 0;
  virtual bool enableMaskedInterleavedAccessVectorization() = 0;
  virtual bool isFPVectorizationPotentiallyUnsafe() = 0;
  virtual bool allowsMisalignedMemoryAccesses(LLVMContext &Context,
                                              unsigned BitWidth,
                                              unsigned AddressSpace,
                                              Align Alignment,
                                              bool *Fast) = 0;
  virtual PopcntSupportKind getPopcntSupport(unsigned IntTyWidthInBit) = 0;
  virtual bool haveFastSqrt(Type *Ty) = 0;
  virtual bool isFCmpOrdCheaperThanFCmpZero(Type *Ty) = 0;
  virtual InstructionCost getFPOpCost(Type *Ty) = 0;
  virtual InstructionCost getIntImmCodeSizeCost(unsigned Opc, unsigned Idx,
                                                const APInt &Imm, Type *Ty) = 0;
  virtual InstructionCost getIntImmCost(const APInt &Imm, Type *Ty,
                                        TargetCostKind CostKind) = 0;
  virtual InstructionCost getIntImmCostInst(unsigned Opc, unsigned Idx,
                                            const APInt &Imm, Type *Ty,
                                            TargetCostKind CostKind,
                                            Instruction *Inst = nullptr) = 0;
  virtual InstructionCost getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx,
                                              const APInt &Imm, Type *Ty,
                                              TargetCostKind CostKind) = 0;
  virtual unsigned getNumberOfRegisters(unsigned ClassID) const = 0;
  virtual unsigned getRegisterClassForType(bool Vector,
                                           Type *Ty = nullptr) const = 0;
  virtual const char *getRegisterClassName(unsigned ClassID) const = 0;
  virtual TypeSize getRegisterBitWidth(RegisterKind K) const = 0;
  virtual unsigned getMinVectorRegisterBitWidth() const = 0;
  virtual Optional<unsigned> getMaxVScale() const = 0;
  virtual bool shouldMaximizeVectorBandwidth() const = 0;
  virtual ElementCount getMinimumVF(unsigned ElemWidth,
                                    bool IsScalable) const = 0;
  virtual unsigned getMaximumVF(unsigned ElemWidth, unsigned Opcode) const = 0;
  virtual bool shouldConsiderAddressTypePromotion(
      const Instruction &I, bool &AllowPromotionWithoutCommonHeader) = 0;
  virtual unsigned getCacheLineSize() const = 0;
  virtual Optional<unsigned> getCacheSize(CacheLevel Level) const = 0;
  virtual Optional<unsigned> getCacheAssociativity(CacheLevel Level) const = 0;

  /// \return How much before a load we should place the prefetch
  /// instruction.  This is currently measured in number of
  /// instructions.
  virtual unsigned getPrefetchDistance() const = 0;

  /// \return Some HW prefetchers can handle accesses up to a certain
  /// constant stride.  This is the minimum stride in bytes where it
  /// makes sense to start adding SW prefetches.  The default is 1,
  /// i.e. prefetch with any stride.  Sometimes prefetching is beneficial
  /// even below the HW prefetcher limit, and the arguments provided are
  /// meant to serve as a basis for deciding this for a particular loop.
  virtual unsigned getMinPrefetchStride(unsigned NumMemAccesses,
                                        unsigned NumStridedMemAccesses,
                                        unsigned NumPrefetches,
                                        bool HasCall) const = 0;

  /// \return The maximum number of iterations to prefetch ahead.  If
  /// the required number of iterations is more than this number, no
  /// prefetching is performed.
  virtual unsigned getMaxPrefetchIterationsAhead() const = 0;

  /// \return True if prefetching should also be done for writes.
  virtual bool enableWritePrefetching() const = 0;

  virtual unsigned getMaxInterleaveFactor(unsigned VF) = 0;
  virtual InstructionCost getArithmeticInstrCost(
      unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
      OperandValueKind Opd1Info, OperandValueKind Opd2Info,
      OperandValueProperties Opd1PropInfo, OperandValueProperties Opd2PropInfo,
      ArrayRef<const Value *> Args, const Instruction *CxtI = nullptr) = 0;
  virtual InstructionCost getShuffleCost(ShuffleKind Kind, VectorType *Tp,
                                         ArrayRef<int> Mask, int Index,
                                         VectorType *SubTp) = 0;
  virtual InstructionCost getCastInstrCost(unsigned Opcode, Type *Dst,
                                           Type *Src, CastContextHint CCH,
                                           TTI::TargetCostKind CostKind,
                                           const Instruction *I) = 0;
  virtual InstructionCost getExtractWithExtendCost(unsigned Opcode, Type *Dst,
                                                   VectorType *VecTy,
                                                   unsigned Index) = 0;
  virtual InstructionCost getCFInstrCost(unsigned Opcode,
                                         TTI::TargetCostKind CostKind,
                                         const Instruction *I = nullptr) = 0;
  virtual InstructionCost getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                             Type *CondTy,
                                             CmpInst::Predicate VecPred,
                                             TTI::TargetCostKind CostKind,
                                             const Instruction *I) = 0;
  virtual InstructionCost getVectorInstrCost(unsigned Opcode, Type *Val,
                                             unsigned Index) = 0;
  virtual InstructionCost getMemoryOpCost(unsigned Opcode, Type *Src,
                                          Align Alignment,
                                          unsigned AddressSpace,
                                          TTI::TargetCostKind CostKind,
                                          const Instruction *I) = 0;
  virtual InstructionCost
  getMaskedMemoryOpCost(unsigned Opcode, Type *Src, Align Alignment,
                        unsigned AddressSpace,
                        TTI::TargetCostKind CostKind) = 0;
  virtual InstructionCost
  getGatherScatterOpCost(unsigned Opcode, Type *DataTy, const Value *Ptr,
                         bool VariableMask, Align Alignment,
                         TTI::TargetCostKind CostKind,
                         const Instruction *I = nullptr) = 0;

  virtual InstructionCost getInterleavedMemoryOpCost(
      unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
      Align Alignment, unsigned AddressSpace, TTI::TargetCostKind CostKind,
      bool UseMaskForCond = false, bool UseMaskForGaps = false) = 0;
  virtual InstructionCost
  getArithmeticReductionCost(unsigned Opcode, VectorType *Ty,
                             Optional<FastMathFlags> FMF,
                             TTI::TargetCostKind CostKind) = 0;
  virtual InstructionCost
  getMinMaxReductionCost(VectorType *Ty, VectorType *CondTy, bool IsUnsigned,
                         TTI::TargetCostKind CostKind) = 0;
  virtual InstructionCost getExtendedAddReductionCost(
      bool IsMLA, bool IsUnsigned, Type *ResTy, VectorType *Ty,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput) = 0;
  virtual InstructionCost
  getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                        TTI::TargetCostKind CostKind) = 0;
  virtual InstructionCost getCallInstrCost(Function *F, Type *RetTy,
                                           ArrayRef<Type *> Tys,
                                           TTI::TargetCostKind CostKind) = 0;
  virtual unsigned getNumberOfParts(Type *Tp) = 0;
  virtual InstructionCost
  getAddressComputationCost(Type *Ty, ScalarEvolution *SE, const SCEV *Ptr) = 0;
  virtual InstructionCost
  getCostOfKeepingLiveOverCall(ArrayRef<Type *> Tys) = 0;
  virtual bool getTgtMemIntrinsic(IntrinsicInst *Inst,
                                  MemIntrinsicInfo &Info) = 0;
  virtual unsigned getAtomicMemIntrinsicMaxElementSize() const = 0;
  virtual Value *getOrCreateResultFromMemIntrinsic(IntrinsicInst *Inst,
                                                   Type *ExpectedType) = 0;
  virtual Type *getMemcpyLoopLoweringType(LLVMContext &Context, Value *Length,
                                          unsigned SrcAddrSpace,
                                          unsigned DestAddrSpace,
                                          unsigned SrcAlign,
                                          unsigned DestAlign) const = 0;
  virtual void getMemcpyLoopResidualLoweringType(
      SmallVectorImpl<Type *> &OpsOut, LLVMContext &Context,
      unsigned RemainingBytes, unsigned SrcAddrSpace, unsigned DestAddrSpace,
      unsigned SrcAlign, unsigned DestAlign) const = 0;
  virtual bool areInlineCompatible(const Function *Caller,
                                   const Function *Callee) const = 0;
  virtual bool
  areFunctionArgsABICompatible(const Function *Caller, const Function *Callee,
                               SmallPtrSetImpl<Argument *> &Args) const = 0;
  virtual bool isIndexedLoadLegal(MemIndexedMode Mode, Type *Ty) const = 0;
  virtual bool isIndexedStoreLegal(MemIndexedMode Mode, Type *Ty) const = 0;
  virtual unsigned getLoadStoreVecRegBitWidth(unsigned AddrSpace) const = 0;
  virtual bool isLegalToVectorizeLoad(LoadInst *LI) const = 0;
  virtual bool isLegalToVectorizeStore(StoreInst *SI) const = 0;
  virtual bool isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes,
                                           Align Alignment,
                                           unsigned AddrSpace) const = 0;
  virtual bool isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes,
                                            Align Alignment,
                                            unsigned AddrSpace) const = 0;
  virtual bool isLegalToVectorizeReduction(const RecurrenceDescriptor &RdxDesc,
                                           ElementCount VF) const = 0;
  virtual bool isElementTypeLegalForScalableVector(Type *Ty) const = 0;
  virtual unsigned getLoadVectorFactor(unsigned VF, unsigned LoadSize,
                                       unsigned ChainSizeInBytes,
                                       VectorType *VecTy) const = 0;
  virtual unsigned getStoreVectorFactor(unsigned VF, unsigned StoreSize,
                                        unsigned ChainSizeInBytes,
                                        VectorType *VecTy) const = 0;
  virtual bool preferInLoopReduction(unsigned Opcode, Type *Ty,
                                     ReductionFlags) const = 0;
  virtual bool preferPredicatedReductionSelect(unsigned Opcode, Type *Ty,
                                               ReductionFlags) const = 0;
  virtual bool shouldExpandReduction(const IntrinsicInst *II) const = 0;
  virtual unsigned getGISelRematGlobalCost() const = 0;
  virtual bool supportsScalableVectors() const = 0;
  virtual bool hasActiveVectorLength() const = 0;
  virtual InstructionCost getInstructionLatency(const Instruction *I) = 0;
  virtual VPLegalization
  getVPLegalizationStrategy(const VPIntrinsic &PI) const = 0;
};

template <typename T>
class TargetTransformInfo::Model final : public TargetTransformInfo::Concept {
  T Impl;

public:
  Model(T Impl) : Impl(std::move(Impl)) {}
  ~Model() override {}

  const DataLayout &getDataLayout() const override {
    return Impl.getDataLayout();
  }

  InstructionCost
  getGEPCost(Type *PointeeType, const Value *Ptr,
             ArrayRef<const Value *> Operands,
             TargetTransformInfo::TargetCostKind CostKind) override {
    return Impl.getGEPCost(PointeeType, Ptr, Operands, CostKind);
  }
  unsigned getInliningThresholdMultiplier() override {
    return Impl.getInliningThresholdMultiplier();
  }
  unsigned adjustInliningThreshold(const CallBase *CB) override {
    return Impl.adjustInliningThreshold(CB);
  }
  int getInlinerVectorBonusPercent() override {
    return Impl.getInlinerVectorBonusPercent();
  }
  InstructionCost getMemcpyCost(const Instruction *I) override {
    return Impl.getMemcpyCost(I);
  }
  InstructionCost getUserCost(const User *U, ArrayRef<const Value *> Operands,
                              TargetCostKind CostKind) override {
    return Impl.getUserCost(U, Operands, CostKind);
  }
  BranchProbability getPredictableBranchThreshold() override {
    return Impl.getPredictableBranchThreshold();
  }
  bool hasBranchDivergence() override { return Impl.hasBranchDivergence(); }
  bool useGPUDivergenceAnalysis() override {
    return Impl.useGPUDivergenceAnalysis();
  }
  bool isSourceOfDivergence(const Value *V) override {
    return Impl.isSourceOfDivergence(V);
  }

  bool isAlwaysUniform(const Value *V) override {
    return Impl.isAlwaysUniform(V);
  }

  unsigned getFlatAddressSpace() override { return Impl.getFlatAddressSpace(); }

  bool collectFlatAddressOperands(SmallVectorImpl<int> &OpIndexes,
                                  Intrinsic::ID IID) const override {
    return Impl.collectFlatAddressOperands(OpIndexes, IID);
  }

  bool isNoopAddrSpaceCast(unsigned FromAS, unsigned ToAS) const override {
    return Impl.isNoopAddrSpaceCast(FromAS, ToAS);
  }

  bool
  canHaveNonUndefGlobalInitializerInAddressSpace(unsigned AS) const override {
    return Impl.canHaveNonUndefGlobalInitializerInAddressSpace(AS);
  }

  unsigned getAssumedAddrSpace(const Value *V) const override {
    return Impl.getAssumedAddrSpace(V);
  }

  Value *rewriteIntrinsicWithAddressSpace(IntrinsicInst *II, Value *OldV,
                                          Value *NewV) const override {
    return Impl.rewriteIntrinsicWithAddressSpace(II, OldV, NewV);
  }

  bool isLoweredToCall(const Function *F) override {
    return Impl.isLoweredToCall(F);
  }
  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               UnrollingPreferences &UP,
                               OptimizationRemarkEmitter *ORE) override {
    return Impl.getUnrollingPreferences(L, SE, UP, ORE);
  }
  void getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                             PeelingPreferences &PP) override {
    return Impl.getPeelingPreferences(L, SE, PP);
  }
  bool isHardwareLoopProfitable(Loop *L, ScalarEvolution &SE,
                                AssumptionCache &AC, TargetLibraryInfo *LibInfo,
                                HardwareLoopInfo &HWLoopInfo) override {
    return Impl.isHardwareLoopProfitable(L, SE, AC, LibInfo, HWLoopInfo);
  }
  bool preferPredicateOverEpilogue(Loop *L, LoopInfo *LI, ScalarEvolution &SE,
                                   AssumptionCache &AC, TargetLibraryInfo *TLI,
                                   DominatorTree *DT,
                                   const LoopAccessInfo *LAI) override {
    return Impl.preferPredicateOverEpilogue(L, LI, SE, AC, TLI, DT, LAI);
  }
  bool emitGetActiveLaneMask() override {
    return Impl.emitGetActiveLaneMask();
  }
  Optional<Instruction *> instCombineIntrinsic(InstCombiner &IC,
                                               IntrinsicInst &II) override {
    return Impl.instCombineIntrinsic(IC, II);
  }
  Optional<Value *>
  simplifyDemandedUseBitsIntrinsic(InstCombiner &IC, IntrinsicInst &II,
                                   APInt DemandedMask, KnownBits &Known,
                                   bool &KnownBitsComputed) override {
    return Impl.simplifyDemandedUseBitsIntrinsic(IC, II, DemandedMask, Known,
                                                 KnownBitsComputed);
  }
  Optional<Value *> simplifyDemandedVectorEltsIntrinsic(
      InstCombiner &IC, IntrinsicInst &II, APInt DemandedElts, APInt &UndefElts,
      APInt &UndefElts2, APInt &UndefElts3,
      std::function<void(Instruction *, unsigned, APInt, APInt &)>
          SimplifyAndSetOp) override {
    return Impl.simplifyDemandedVectorEltsIntrinsic(
        IC, II, DemandedElts, UndefElts, UndefElts2, UndefElts3,
        SimplifyAndSetOp);
  }
  bool isLegalAddImmediate(int64_t Imm) override {
    return Impl.isLegalAddImmediate(Imm);
  }
  bool isLegalICmpImmediate(int64_t Imm) override {
    return Impl.isLegalICmpImmediate(Imm);
  }
  bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                             bool HasBaseReg, int64_t Scale, unsigned AddrSpace,
                             Instruction *I) override {
    return Impl.isLegalAddressingMode(Ty, BaseGV, BaseOffset, HasBaseReg, Scale,
                                      AddrSpace, I);
  }
  bool isLSRCostLess(TargetTransformInfo::LSRCost &C1,
                     TargetTransformInfo::LSRCost &C2) override {
    return Impl.isLSRCostLess(C1, C2);
  }
  bool isNumRegsMajorCostOfLSR() override {
    return Impl.isNumRegsMajorCostOfLSR();
  }
  bool isProfitableLSRChainElement(Instruction *I) override {
    return Impl.isProfitableLSRChainElement(I);
  }
  bool canMacroFuseCmp() override { return Impl.canMacroFuseCmp(); }
  bool canSaveCmp(Loop *L, BranchInst **BI, ScalarEvolution *SE, LoopInfo *LI,
                  DominatorTree *DT, AssumptionCache *AC,
                  TargetLibraryInfo *LibInfo) override {
    return Impl.canSaveCmp(L, BI, SE, LI, DT, AC, LibInfo);
  }
  AddressingModeKind
    getPreferredAddressingMode(const Loop *L,
                               ScalarEvolution *SE) const override {
    return Impl.getPreferredAddressingMode(L, SE);
  }
  bool isLegalMaskedStore(Type *DataType, Align Alignment) override {
    return Impl.isLegalMaskedStore(DataType, Alignment);
  }
  bool isLegalMaskedLoad(Type *DataType, Align Alignment) override {
    return Impl.isLegalMaskedLoad(DataType, Alignment);
  }
  bool isLegalNTStore(Type *DataType, Align Alignment) override {
    return Impl.isLegalNTStore(DataType, Alignment);
  }
  bool isLegalNTLoad(Type *DataType, Align Alignment) override {
    return Impl.isLegalNTLoad(DataType, Alignment);
  }
  bool isLegalMaskedScatter(Type *DataType, Align Alignment) override {
    return Impl.isLegalMaskedScatter(DataType, Alignment);
  }
  bool isLegalMaskedGather(Type *DataType, Align Alignment) override {
    return Impl.isLegalMaskedGather(DataType, Alignment);
  }
  bool isLegalMaskedCompressStore(Type *DataType) override {
    return Impl.isLegalMaskedCompressStore(DataType);
  }
  bool isLegalMaskedExpandLoad(Type *DataType) override {
    return Impl.isLegalMaskedExpandLoad(DataType);
  }
  bool enableOrderedReductions() override {
    return Impl.enableOrderedReductions();
  }
  bool hasDivRemOp(Type *DataType, bool IsSigned) override {
    return Impl.hasDivRemOp(DataType, IsSigned);
  }
  bool hasVolatileVariant(Instruction *I, unsigned AddrSpace) override {
    return Impl.hasVolatileVariant(I, AddrSpace);
  }
  bool prefersVectorizedAddressing() override {
    return Impl.prefersVectorizedAddressing();
  }
  InstructionCost getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                                       int64_t BaseOffset, bool HasBaseReg,
                                       int64_t Scale,
                                       unsigned AddrSpace) override {
    return Impl.getScalingFactorCost(Ty, BaseGV, BaseOffset, HasBaseReg, Scale,
                                     AddrSpace);
  }
  bool LSRWithInstrQueries() override { return Impl.LSRWithInstrQueries(); }
  bool isTruncateFree(Type *Ty1, Type *Ty2) override {
    return Impl.isTruncateFree(Ty1, Ty2);
  }
  bool isProfitableToHoist(Instruction *I) override {
    return Impl.isProfitableToHoist(I);
  }
  bool useAA() override { return Impl.useAA(); }
  bool isTypeLegal(Type *Ty) override { return Impl.isTypeLegal(Ty); }
  InstructionCost getRegUsageForType(Type *Ty) override {
    return Impl.getRegUsageForType(Ty);
  }
  bool shouldBuildLookupTables() override {
    return Impl.shouldBuildLookupTables();
  }
  bool shouldBuildLookupTablesForConstant(Constant *C) override {
    return Impl.shouldBuildLookupTablesForConstant(C);
  }
  bool shouldBuildRelLookupTables() override {
    return Impl.shouldBuildRelLookupTables();
  }
  bool useColdCCForColdCall(Function &F) override {
    return Impl.useColdCCForColdCall(F);
  }

  InstructionCost getScalarizationOverhead(VectorType *Ty,
                                           const APInt &DemandedElts,
                                           bool Insert, bool Extract) override {
    return Impl.getScalarizationOverhead(Ty, DemandedElts, Insert, Extract);
  }
  InstructionCost
  getOperandsScalarizationOverhead(ArrayRef<const Value *> Args,
                                   ArrayRef<Type *> Tys) override {
    return Impl.getOperandsScalarizationOverhead(Args, Tys);
  }

  bool supportsEfficientVectorElementLoadStore() override {
    return Impl.supportsEfficientVectorElementLoadStore();
  }

  bool enableAggressiveInterleaving(bool LoopHasReductions) override {
    return Impl.enableAggressiveInterleaving(LoopHasReductions);
  }
  MemCmpExpansionOptions enableMemCmpExpansion(bool OptSize,
                                               bool IsZeroCmp) const override {
    return Impl.enableMemCmpExpansion(OptSize, IsZeroCmp);
  }
  bool enableInterleavedAccessVectorization() override {
    return Impl.enableInterleavedAccessVectorization();
  }
  bool enableMaskedInterleavedAccessVectorization() override {
    return Impl.enableMaskedInterleavedAccessVectorization();
  }
  bool isFPVectorizationPotentiallyUnsafe() override {
    return Impl.isFPVectorizationPotentiallyUnsafe();
  }
  bool allowsMisalignedMemoryAccesses(LLVMContext &Context, unsigned BitWidth,
                                      unsigned AddressSpace, Align Alignment,
                                      bool *Fast) override {
    return Impl.allowsMisalignedMemoryAccesses(Context, BitWidth, AddressSpace,
                                               Alignment, Fast);
  }
  PopcntSupportKind getPopcntSupport(unsigned IntTyWidthInBit) override {
    return Impl.getPopcntSupport(IntTyWidthInBit);
  }
  bool haveFastSqrt(Type *Ty) override { return Impl.haveFastSqrt(Ty); }

  bool isFCmpOrdCheaperThanFCmpZero(Type *Ty) override {
    return Impl.isFCmpOrdCheaperThanFCmpZero(Ty);
  }

  InstructionCost getFPOpCost(Type *Ty) override {
    return Impl.getFPOpCost(Ty);
  }

  InstructionCost getIntImmCodeSizeCost(unsigned Opc, unsigned Idx,
                                        const APInt &Imm, Type *Ty) override {
    return Impl.getIntImmCodeSizeCost(Opc, Idx, Imm, Ty);
  }
  InstructionCost getIntImmCost(const APInt &Imm, Type *Ty,
                                TargetCostKind CostKind) override {
    return Impl.getIntImmCost(Imm, Ty, CostKind);
  }
  InstructionCost getIntImmCostInst(unsigned Opc, unsigned Idx,
                                    const APInt &Imm, Type *Ty,
                                    TargetCostKind CostKind,
                                    Instruction *Inst = nullptr) override {
    return Impl.getIntImmCostInst(Opc, Idx, Imm, Ty, CostKind, Inst);
  }
  InstructionCost getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx,
                                      const APInt &Imm, Type *Ty,
                                      TargetCostKind CostKind) override {
    return Impl.getIntImmCostIntrin(IID, Idx, Imm, Ty, CostKind);
  }
  unsigned getNumberOfRegisters(unsigned ClassID) const override {
    return Impl.getNumberOfRegisters(ClassID);
  }
  unsigned getRegisterClassForType(bool Vector,
                                   Type *Ty = nullptr) const override {
    return Impl.getRegisterClassForType(Vector, Ty);
  }
  const char *getRegisterClassName(unsigned ClassID) const override {
    return Impl.getRegisterClassName(ClassID);
  }
  TypeSize getRegisterBitWidth(RegisterKind K) const override {
    return Impl.getRegisterBitWidth(K);
  }
  unsigned getMinVectorRegisterBitWidth() const override {
    return Impl.getMinVectorRegisterBitWidth();
  }
  Optional<unsigned> getMaxVScale() const override {
    return Impl.getMaxVScale();
  }
  bool shouldMaximizeVectorBandwidth() const override {
    return Impl.shouldMaximizeVectorBandwidth();
  }
  ElementCount getMinimumVF(unsigned ElemWidth,
                            bool IsScalable) const override {
    return Impl.getMinimumVF(ElemWidth, IsScalable);
  }
  unsigned getMaximumVF(unsigned ElemWidth, unsigned Opcode) const override {
    return Impl.getMaximumVF(ElemWidth, Opcode);
  }
  bool shouldConsiderAddressTypePromotion(
      const Instruction &I, bool &AllowPromotionWithoutCommonHeader) override {
    return Impl.shouldConsiderAddressTypePromotion(
        I, AllowPromotionWithoutCommonHeader);
  }
  unsigned getCacheLineSize() const override { return Impl.getCacheLineSize(); }
  Optional<unsigned> getCacheSize(CacheLevel Level) const override {
    return Impl.getCacheSize(Level);
  }
  Optional<unsigned> getCacheAssociativity(CacheLevel Level) const override {
    return Impl.getCacheAssociativity(Level);
  }

  /// Return the preferred prefetch distance in terms of instructions.
  ///
  unsigned getPrefetchDistance() const override {
    return Impl.getPrefetchDistance();
  }

  /// Return the minimum stride necessary to trigger software
  /// prefetching.
  ///
  unsigned getMinPrefetchStride(unsigned NumMemAccesses,
                                unsigned NumStridedMemAccesses,
                                unsigned NumPrefetches,
                                bool HasCall) const override {
    return Impl.getMinPrefetchStride(NumMemAccesses, NumStridedMemAccesses,
                                     NumPrefetches, HasCall);
  }

  /// Return the maximum prefetch distance in terms of loop
  /// iterations.
  ///
  unsigned getMaxPrefetchIterationsAhead() const override {
    return Impl.getMaxPrefetchIterationsAhead();
  }

  /// \return True if prefetching should also be done for writes.
  bool enableWritePrefetching() const override {
    return Impl.enableWritePrefetching();
  }

  unsigned getMaxInterleaveFactor(unsigned VF) override {
    return Impl.getMaxInterleaveFactor(VF);
  }
  unsigned getEstimatedNumberOfCaseClusters(const SwitchInst &SI,
                                            unsigned &JTSize,
                                            ProfileSummaryInfo *PSI,
                                            BlockFrequencyInfo *BFI) override {
    return Impl.getEstimatedNumberOfCaseClusters(SI, JTSize, PSI, BFI);
  }
  InstructionCost getArithmeticInstrCost(
      unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
      OperandValueKind Opd1Info, OperandValueKind Opd2Info,
      OperandValueProperties Opd1PropInfo, OperandValueProperties Opd2PropInfo,
      ArrayRef<const Value *> Args,
      const Instruction *CxtI = nullptr) override {
    return Impl.getArithmeticInstrCost(Opcode, Ty, CostKind, Opd1Info, Opd2Info,
                                       Opd1PropInfo, Opd2PropInfo, Args, CxtI);
  }
  InstructionCost getShuffleCost(ShuffleKind Kind, VectorType *Tp,
                                 ArrayRef<int> Mask, int Index,
                                 VectorType *SubTp) override {
    return Impl.getShuffleCost(Kind, Tp, Mask, Index, SubTp);
  }
  InstructionCost getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                                   CastContextHint CCH,
                                   TTI::TargetCostKind CostKind,
                                   const Instruction *I) override {
    return Impl.getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);
  }
  InstructionCost getExtractWithExtendCost(unsigned Opcode, Type *Dst,
                                           VectorType *VecTy,
                                           unsigned Index) override {
    return Impl.getExtractWithExtendCost(Opcode, Dst, VecTy, Index);
  }
  InstructionCost getCFInstrCost(unsigned Opcode, TTI::TargetCostKind CostKind,
                                 const Instruction *I = nullptr) override {
    return Impl.getCFInstrCost(Opcode, CostKind, I);
  }
  InstructionCost getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                                     CmpInst::Predicate VecPred,
                                     TTI::TargetCostKind CostKind,
                                     const Instruction *I) override {
    return Impl.getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind, I);
  }
  InstructionCost getVectorInstrCost(unsigned Opcode, Type *Val,
                                     unsigned Index) override {
    return Impl.getVectorInstrCost(Opcode, Val, Index);
  }
  InstructionCost getMemoryOpCost(unsigned Opcode, Type *Src, Align Alignment,
                                  unsigned AddressSpace,
                                  TTI::TargetCostKind CostKind,
                                  const Instruction *I) override {
    return Impl.getMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                CostKind, I);
  }
  InstructionCost getMaskedMemoryOpCost(unsigned Opcode, Type *Src,
                                        Align Alignment, unsigned AddressSpace,
                                        TTI::TargetCostKind CostKind) override {
    return Impl.getMaskedMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                      CostKind);
  }
  InstructionCost
  getGatherScatterOpCost(unsigned Opcode, Type *DataTy, const Value *Ptr,
                         bool VariableMask, Align Alignment,
                         TTI::TargetCostKind CostKind,
                         const Instruction *I = nullptr) override {
    return Impl.getGatherScatterOpCost(Opcode, DataTy, Ptr, VariableMask,
                                       Alignment, CostKind, I);
  }
  InstructionCost getInterleavedMemoryOpCost(
      unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
      Align Alignment, unsigned AddressSpace, TTI::TargetCostKind CostKind,
      bool UseMaskForCond, bool UseMaskForGaps) override {
    return Impl.getInterleavedMemoryOpCost(Opcode, VecTy, Factor, Indices,
                                           Alignment, AddressSpace, CostKind,
                                           UseMaskForCond, UseMaskForGaps);
  }
  InstructionCost
  getArithmeticReductionCost(unsigned Opcode, VectorType *Ty,
                             Optional<FastMathFlags> FMF,
                             TTI::TargetCostKind CostKind) override {
    return Impl.getArithmeticReductionCost(Opcode, Ty, FMF, CostKind);
  }
  InstructionCost
  getMinMaxReductionCost(VectorType *Ty, VectorType *CondTy, bool IsUnsigned,
                         TTI::TargetCostKind CostKind) override {
    return Impl.getMinMaxReductionCost(Ty, CondTy, IsUnsigned, CostKind);
  }
  InstructionCost getExtendedAddReductionCost(
      bool IsMLA, bool IsUnsigned, Type *ResTy, VectorType *Ty,
      TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput) override {
    return Impl.getExtendedAddReductionCost(IsMLA, IsUnsigned, ResTy, Ty,
                                            CostKind);
  }
  InstructionCost getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                        TTI::TargetCostKind CostKind) override {
    return Impl.getIntrinsicInstrCost(ICA, CostKind);
  }
  InstructionCost getCallInstrCost(Function *F, Type *RetTy,
                                   ArrayRef<Type *> Tys,
                                   TTI::TargetCostKind CostKind) override {
    return Impl.getCallInstrCost(F, RetTy, Tys, CostKind);
  }
  unsigned getNumberOfParts(Type *Tp) override {
    return Impl.getNumberOfParts(Tp);
  }
  InstructionCost getAddressComputationCost(Type *Ty, ScalarEvolution *SE,
                                            const SCEV *Ptr) override {
    return Impl.getAddressComputationCost(Ty, SE, Ptr);
  }
  InstructionCost getCostOfKeepingLiveOverCall(ArrayRef<Type *> Tys) override {
    return Impl.getCostOfKeepingLiveOverCall(Tys);
  }
  bool getTgtMemIntrinsic(IntrinsicInst *Inst,
                          MemIntrinsicInfo &Info) override {
    return Impl.getTgtMemIntrinsic(Inst, Info);
  }
  unsigned getAtomicMemIntrinsicMaxElementSize() const override {
    return Impl.getAtomicMemIntrinsicMaxElementSize();
  }
  Value *getOrCreateResultFromMemIntrinsic(IntrinsicInst *Inst,
                                           Type *ExpectedType) override {
    return Impl.getOrCreateResultFromMemIntrinsic(Inst, ExpectedType);
  }
  Type *getMemcpyLoopLoweringType(LLVMContext &Context, Value *Length,
                                  unsigned SrcAddrSpace, unsigned DestAddrSpace,
                                  unsigned SrcAlign,
                                  unsigned DestAlign) const override {
    return Impl.getMemcpyLoopLoweringType(Context, Length, SrcAddrSpace,
                                          DestAddrSpace, SrcAlign, DestAlign);
  }
  void getMemcpyLoopResidualLoweringType(
      SmallVectorImpl<Type *> &OpsOut, LLVMContext &Context,
      unsigned RemainingBytes, unsigned SrcAddrSpace, unsigned DestAddrSpace,
      unsigned SrcAlign, unsigned DestAlign) const override {
    Impl.getMemcpyLoopResidualLoweringType(OpsOut, Context, RemainingBytes,
                                           SrcAddrSpace, DestAddrSpace,
                                           SrcAlign, DestAlign);
  }
  bool areInlineCompatible(const Function *Caller,
                           const Function *Callee) const override {
    return Impl.areInlineCompatible(Caller, Callee);
  }
  bool areFunctionArgsABICompatible(
      const Function *Caller, const Function *Callee,
      SmallPtrSetImpl<Argument *> &Args) const override {
    return Impl.areFunctionArgsABICompatible(Caller, Callee, Args);
  }
  bool isIndexedLoadLegal(MemIndexedMode Mode, Type *Ty) const override {
    return Impl.isIndexedLoadLegal(Mode, Ty, getDataLayout());
  }
  bool isIndexedStoreLegal(MemIndexedMode Mode, Type *Ty) const override {
    return Impl.isIndexedStoreLegal(Mode, Ty, getDataLayout());
  }
  unsigned getLoadStoreVecRegBitWidth(unsigned AddrSpace) const override {
    return Impl.getLoadStoreVecRegBitWidth(AddrSpace);
  }
  bool isLegalToVectorizeLoad(LoadInst *LI) const override {
    return Impl.isLegalToVectorizeLoad(LI);
  }
  bool isLegalToVectorizeStore(StoreInst *SI) const override {
    return Impl.isLegalToVectorizeStore(SI);
  }
  bool isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes, Align Alignment,
                                   unsigned AddrSpace) const override {
    return Impl.isLegalToVectorizeLoadChain(ChainSizeInBytes, Alignment,
                                            AddrSpace);
  }
  bool isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes, Align Alignment,
                                    unsigned AddrSpace) const override {
    return Impl.isLegalToVectorizeStoreChain(ChainSizeInBytes, Alignment,
                                             AddrSpace);
  }
  bool isLegalToVectorizeReduction(const RecurrenceDescriptor &RdxDesc,
                                   ElementCount VF) const override {
    return Impl.isLegalToVectorizeReduction(RdxDesc, VF);
  }
  bool isElementTypeLegalForScalableVector(Type *Ty) const override {
    return Impl.isElementTypeLegalForScalableVector(Ty);
  }
  unsigned getLoadVectorFactor(unsigned VF, unsigned LoadSize,
                               unsigned ChainSizeInBytes,
                               VectorType *VecTy) const override {
    return Impl.getLoadVectorFactor(VF, LoadSize, ChainSizeInBytes, VecTy);
  }
  unsigned getStoreVectorFactor(unsigned VF, unsigned StoreSize,
                                unsigned ChainSizeInBytes,
                                VectorType *VecTy) const override {
    return Impl.getStoreVectorFactor(VF, StoreSize, ChainSizeInBytes, VecTy);
  }
  bool preferInLoopReduction(unsigned Opcode, Type *Ty,
                             ReductionFlags Flags) const override {
    return Impl.preferInLoopReduction(Opcode, Ty, Flags);
  }
  bool preferPredicatedReductionSelect(unsigned Opcode, Type *Ty,
                                       ReductionFlags Flags) const override {
    return Impl.preferPredicatedReductionSelect(Opcode, Ty, Flags);
  }
  bool shouldExpandReduction(const IntrinsicInst *II) const override {
    return Impl.shouldExpandReduction(II);
  }

  unsigned getGISelRematGlobalCost() const override {
    return Impl.getGISelRematGlobalCost();
  }

  bool supportsScalableVectors() const override {
    return Impl.supportsScalableVectors();
  }

  bool hasActiveVectorLength() const override {
    return Impl.hasActiveVectorLength();
  }

  InstructionCost getInstructionLatency(const Instruction *I) override {
    return Impl.getInstructionLatency(I);
  }

  VPLegalization
  getVPLegalizationStrategy(const VPIntrinsic &PI) const override {
    return Impl.getVPLegalizationStrategy(PI);
  }
};

template <typename T>
TargetTransformInfo::TargetTransformInfo(T Impl)
    : TTIImpl(new Model<T>(Impl)) {}

/// Analysis pass providing the \c TargetTransformInfo.
///
/// The core idea of the TargetIRAnalysis is to expose an interface through
/// which LLVM targets can analyze and provide information about the middle
/// end's target-independent IR. This supports use cases such as target-aware
/// cost modeling of IR constructs.
///
/// This is a function analysis because much of the cost modeling for targets
/// is done in a subtarget specific way and LLVM supports compiling different
/// functions targeting different subtargets in order to support runtime
/// dispatch according to the observed subtarget.
class TargetIRAnalysis : public AnalysisInfoMixin<TargetIRAnalysis> {
public:
  typedef TargetTransformInfo Result;

  /// Default construct a target IR analysis.
  ///
  /// This will use the module's datalayout to construct a baseline
  /// conservative TTI result.
  TargetIRAnalysis();

  /// Construct an IR analysis pass around a target-provide callback.
  ///
  /// The callback will be called with a particular function for which the TTI
  /// is needed and must return a TTI object for that function.
  TargetIRAnalysis(std::function<Result(const Function &)> TTICallback);

  // Value semantics. We spell out the constructors for MSVC.
  TargetIRAnalysis(const TargetIRAnalysis &Arg)
      : TTICallback(Arg.TTICallback) {}
  TargetIRAnalysis(TargetIRAnalysis &&Arg)
      : TTICallback(std::move(Arg.TTICallback)) {}
  TargetIRAnalysis &operator=(const TargetIRAnalysis &RHS) {
    TTICallback = RHS.TTICallback;
    return *this;
  }
  TargetIRAnalysis &operator=(TargetIRAnalysis &&RHS) {
    TTICallback = std::move(RHS.TTICallback);
    return *this;
  }

  Result run(const Function &F, FunctionAnalysisManager &);

private:
  friend AnalysisInfoMixin<TargetIRAnalysis>;
  static AnalysisKey Key;

  /// The callback used to produce a result.
  ///
  /// We use a completely opaque callback so that targets can provide whatever
  /// mechanism they desire for constructing the TTI for a given function.
  ///
  /// FIXME: Should we really use std::function? It's relatively inefficient.
  /// It might be possible to arrange for even stateful callbacks to outlive
  /// the analysis and thus use a function_ref which would be lighter weight.
  /// This may also be less error prone as the callback is likely to reference
  /// the external TargetMachine, and that reference needs to never dangle.
  std::function<Result(const Function &)> TTICallback;

  /// Helper function used as the callback in the default constructor.
  static Result getDefaultTTI(const Function &F);
};

/// Wrapper pass for TargetTransformInfo.
///
/// This pass can be constructed from a TTI object which it stores internally
/// and is queried by passes.
class TargetTransformInfoWrapperPass : public ImmutablePass {
  TargetIRAnalysis TIRA;
  Optional<TargetTransformInfo> TTI;

  virtual void anchor();

public:
  static char ID;

  /// We must provide a default constructor for the pass but it should
  /// never be used.
  ///
  /// Use the constructor below or call one of the creation routines.
  TargetTransformInfoWrapperPass();

  explicit TargetTransformInfoWrapperPass(TargetIRAnalysis TIRA);

  TargetTransformInfo &getTTI(const Function &F);
};

/// Create an analysis pass wrapper around a TTI object.
///
/// This analysis pass just holds the TTI instance and makes it available to
/// clients.
ImmutablePass *createTargetTransformInfoWrapperPass(TargetIRAnalysis TIRA);

} // namespace llvm

#endif
