//===- llvm/Analysis/ScalarEvolution.h - Scalar Evolution -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The ScalarEvolution class is an LLVM pass which can be used to analyze and
// categorize scalar expressions in loops.  It specializes in recognizing
// general induction variables, representing them with the abstract and opaque
// SCEV class.  Given this analysis, trip counts of loops and other important
// properties can be obtained.
//
// This analysis is primarily useful for induction variable substitution and
// strength reduction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SCALAREVOLUTION_H
#define LLVM_ANALYSIS_SCALAREVOLUTION_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ValueHandle.h"
#include <map>

namespace llvm {
  class APInt;
  class Constant;
  class ConstantInt;
  class DominatorTree;
  class Type;
  class ScalarEvolution;
  class DataLayout;
  class TargetLibraryInfo;
  class LLVMContext;
  class Loop;
  class LoopInfo;
  class Operator;
  class SCEVUnknown;
  class SCEV;
  template<> struct FoldingSetTrait<SCEV>;

  /// SCEV - This class represents an analyzed expression in the program.  These
  /// are opaque objects that the client is not allowed to do much with
  /// directly.
  ///
  class SCEV : public FoldingSetNode {
    friend struct FoldingSetTrait<SCEV>;

    /// FastID - A reference to an Interned FoldingSetNodeID for this node.
    /// The ScalarEvolution's BumpPtrAllocator holds the data.
    FoldingSetNodeIDRef FastID;

    // The SCEV baseclass this node corresponds to
    const unsigned short SCEVType;

  protected:
    /// SubclassData - This field is initialized to zero and may be used in
    /// subclasses to store miscellaneous information.
    unsigned short SubclassData;

  private:
    SCEV(const SCEV &) LLVM_DELETED_FUNCTION;
    void operator=(const SCEV &) LLVM_DELETED_FUNCTION;

  public:
    /// NoWrapFlags are bitfield indices into SubclassData.
    ///
    /// Add and Mul expressions may have no-unsigned-wrap <NUW> or
    /// no-signed-wrap <NSW> properties, which are derived from the IR
    /// operator. NSW is a misnomer that we use to mean no signed overflow or
    /// underflow.
    ///
    /// AddRec expression may have a no-self-wraparound <NW> property if the
    /// result can never reach the start value. This property is independent of
    /// the actual start value and step direction. Self-wraparound is defined
    /// purely in terms of the recurrence's loop, step size, and
    /// bitwidth. Formally, a recurrence with no self-wraparound satisfies:
    /// abs(step) * max-iteration(loop) <= unsigned-max(bitwidth).
    ///
    /// Note that NUW and NSW are also valid properties of a recurrence, and
    /// either implies NW. For convenience, NW will be set for a recurrence
    /// whenever either NUW or NSW are set.
    enum NoWrapFlags { FlagAnyWrap = 0,          // No guarantee.
                       FlagNW      = (1 << 0),   // No self-wrap.
                       FlagNUW     = (1 << 1),   // No unsigned wrap.
                       FlagNSW     = (1 << 2),   // No signed wrap.
                       NoWrapMask  = (1 << 3) -1 };

    explicit SCEV(const FoldingSetNodeIDRef ID, unsigned SCEVTy) :
      FastID(ID), SCEVType(SCEVTy), SubclassData(0) {}

    unsigned getSCEVType() const { return SCEVType; }

    /// getType - Return the LLVM type of this SCEV expression.
    ///
    Type *getType() const;

    /// isZero - Return true if the expression is a constant zero.
    ///
    bool isZero() const;

    /// isOne - Return true if the expression is a constant one.
    ///
    bool isOne() const;

    /// isAllOnesValue - Return true if the expression is a constant
    /// all-ones value.
    ///
    bool isAllOnesValue() const;

    /// isNonConstantNegative - Return true if the specified scev is negated,
    /// but not a constant.
    bool isNonConstantNegative() const;

    /// print - Print out the internal representation of this scalar to the
    /// specified stream.  This should really only be used for debugging
    /// purposes.
    void print(raw_ostream &OS) const;

    /// dump - This method is used for debugging.
    ///
    void dump() const;
  };

  // Specialize FoldingSetTrait for SCEV to avoid needing to compute
  // temporary FoldingSetNodeID values.
  template<> struct FoldingSetTrait<SCEV> : DefaultFoldingSetTrait<SCEV> {
    static void Profile(const SCEV &X, FoldingSetNodeID& ID) {
      ID = X.FastID;
    }
    static bool Equals(const SCEV &X, const FoldingSetNodeID &ID,
                       unsigned IDHash, FoldingSetNodeID &TempID) {
      return ID == X.FastID;
    }
    static unsigned ComputeHash(const SCEV &X, FoldingSetNodeID &TempID) {
      return X.FastID.ComputeHash();
    }
  };

  inline raw_ostream &operator<<(raw_ostream &OS, const SCEV &S) {
    S.print(OS);
    return OS;
  }

  /// SCEVCouldNotCompute - An object of this class is returned by queries that
  /// could not be answered.  For example, if you ask for the number of
  /// iterations of a linked-list traversal loop, you will get one of these.
  /// None of the standard SCEV operations are valid on this class, it is just a
  /// marker.
  struct SCEVCouldNotCompute : public SCEV {
    SCEVCouldNotCompute();

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static bool classof(const SCEV *S);
  };

  /// ScalarEvolution - This class is the main scalar evolution driver.  Because
  /// client code (intentionally) can't do much with the SCEV objects directly,
  /// they must ask this class for services.
  ///
  class ScalarEvolution : public FunctionPass {
  public:
    /// LoopDisposition - An enum describing the relationship between a
    /// SCEV and a loop.
    enum LoopDisposition {
      LoopVariant,    ///< The SCEV is loop-variant (unknown).
      LoopInvariant,  ///< The SCEV is loop-invariant.
      LoopComputable  ///< The SCEV varies predictably with the loop.
    };

    /// BlockDisposition - An enum describing the relationship between a
    /// SCEV and a basic block.
    enum BlockDisposition {
      DoesNotDominateBlock,  ///< The SCEV does not dominate the block.
      DominatesBlock,        ///< The SCEV dominates the block.
      ProperlyDominatesBlock ///< The SCEV properly dominates the block.
    };

    /// Convenient NoWrapFlags manipulation that hides enum casts and is
    /// visible in the ScalarEvolution name space.
    static SCEV::NoWrapFlags maskFlags(SCEV::NoWrapFlags Flags, int Mask) {
      return (SCEV::NoWrapFlags)(Flags & Mask);
    }
    static SCEV::NoWrapFlags setFlags(SCEV::NoWrapFlags Flags,
                                      SCEV::NoWrapFlags OnFlags) {
      return (SCEV::NoWrapFlags)(Flags | OnFlags);
    }
    static SCEV::NoWrapFlags clearFlags(SCEV::NoWrapFlags Flags,
                                        SCEV::NoWrapFlags OffFlags) {
      return (SCEV::NoWrapFlags)(Flags & ~OffFlags);
    }

  private:
    /// SCEVCallbackVH - A CallbackVH to arrange for ScalarEvolution to be
    /// notified whenever a Value is deleted.
    class SCEVCallbackVH : public CallbackVH {
      ScalarEvolution *SE;
      virtual void deleted();
      virtual void allUsesReplacedWith(Value *New);
    public:
      SCEVCallbackVH(Value *V, ScalarEvolution *SE = 0);
    };

    friend class SCEVCallbackVH;
    friend class SCEVExpander;
    friend class SCEVUnknown;

    /// F - The function we are analyzing.
    ///
    Function *F;

    /// LI - The loop information for the function we are currently analyzing.
    ///
    LoopInfo *LI;

    /// TD - The target data information for the target we are targeting.
    ///
    DataLayout *TD;

    /// TLI - The target library information for the target we are targeting.
    ///
    TargetLibraryInfo *TLI;

    /// DT - The dominator tree.
    ///
    DominatorTree *DT;

    /// CouldNotCompute - This SCEV is used to represent unknown trip
    /// counts and things.
    SCEVCouldNotCompute CouldNotCompute;

    /// ValueExprMapType - The typedef for ValueExprMap.
    ///
    typedef DenseMap<SCEVCallbackVH, const SCEV *, DenseMapInfo<Value *> >
      ValueExprMapType;

    /// ValueExprMap - This is a cache of the values we have analyzed so far.
    ///
    ValueExprMapType ValueExprMap;

    /// Mark predicate values currently being processed by isImpliedCond.
    DenseSet<Value*> PendingLoopPredicates;

    /// ExitLimit - Information about the number of loop iterations for
    /// which a loop exit's branch condition evaluates to the not-taken path.
    /// This is a temporary pair of exact and max expressions that are
    /// eventually summarized in ExitNotTakenInfo and BackedgeTakenInfo.
    struct ExitLimit {
      const SCEV *Exact;
      const SCEV *Max;

      /*implicit*/ ExitLimit(const SCEV *E) : Exact(E), Max(E) {}

      ExitLimit(const SCEV *E, const SCEV *M) : Exact(E), Max(M) {}

      /// hasAnyInfo - Test whether this ExitLimit contains any computed
      /// information, or whether it's all SCEVCouldNotCompute values.
      bool hasAnyInfo() const {
        return !isa<SCEVCouldNotCompute>(Exact) ||
          !isa<SCEVCouldNotCompute>(Max);
      }
    };

    /// ExitNotTakenInfo - Information about the number of times a particular
    /// loop exit may be reached before exiting the loop.
    struct ExitNotTakenInfo {
      AssertingVH<BasicBlock> ExitingBlock;
      const SCEV *ExactNotTaken;
      PointerIntPair<ExitNotTakenInfo*, 1> NextExit;

      ExitNotTakenInfo() : ExitingBlock(0), ExactNotTaken(0) {}

      /// isCompleteList - Return true if all loop exits are computable.
      bool isCompleteList() const {
        return NextExit.getInt() == 0;
      }

      void setIncomplete() { NextExit.setInt(1); }

      /// getNextExit - Return a pointer to the next exit's not-taken info.
      ExitNotTakenInfo *getNextExit() const {
        return NextExit.getPointer();
      }

      void setNextExit(ExitNotTakenInfo *ENT) { NextExit.setPointer(ENT); }
    };

    /// BackedgeTakenInfo - Information about the backedge-taken count
    /// of a loop. This currently includes an exact count and a maximum count.
    ///
    class BackedgeTakenInfo {
      /// ExitNotTaken - A list of computable exits and their not-taken counts.
      /// Loops almost never have more than one computable exit.
      ExitNotTakenInfo ExitNotTaken;

      /// Max - An expression indicating the least maximum backedge-taken
      /// count of the loop that is known, or a SCEVCouldNotCompute.
      const SCEV *Max;

    public:
      BackedgeTakenInfo() : Max(0) {}

      /// Initialize BackedgeTakenInfo from a list of exact exit counts.
      BackedgeTakenInfo(
        SmallVectorImpl< std::pair<BasicBlock *, const SCEV *> > &ExitCounts,
        bool Complete, const SCEV *MaxCount);

      /// hasAnyInfo - Test whether this BackedgeTakenInfo contains any
      /// computed information, or whether it's all SCEVCouldNotCompute
      /// values.
      bool hasAnyInfo() const {
        return ExitNotTaken.ExitingBlock || !isa<SCEVCouldNotCompute>(Max);
      }

      /// getExact - Return an expression indicating the exact backedge-taken
      /// count of the loop if it is known, or SCEVCouldNotCompute
      /// otherwise. This is the number of times the loop header can be
      /// guaranteed to execute, minus one.
      const SCEV *getExact(ScalarEvolution *SE) const;

      /// getExact - Return the number of times this loop exit may fall through
      /// to the back edge, or SCEVCouldNotCompute. The loop is guaranteed not
      /// to exit via this block before this number of iterations, but may exit
      /// via another block.
      const SCEV *getExact(BasicBlock *ExitingBlock, ScalarEvolution *SE) const;

      /// getMax - Get the max backedge taken count for the loop.
      const SCEV *getMax(ScalarEvolution *SE) const;

      /// Return true if any backedge taken count expressions refer to the given
      /// subexpression.
      bool hasOperand(const SCEV *S, ScalarEvolution *SE) const;

      /// clear - Invalidate this result and free associated memory.
      void clear();
    };

    /// BackedgeTakenCounts - Cache the backedge-taken count of the loops for
    /// this function as they are computed.
    DenseMap<const Loop*, BackedgeTakenInfo> BackedgeTakenCounts;

    /// ConstantEvolutionLoopExitValue - This map contains entries for all of
    /// the PHI instructions that we attempt to compute constant evolutions for.
    /// This allows us to avoid potentially expensive recomputation of these
    /// properties.  An instruction maps to null if we are unable to compute its
    /// exit value.
    DenseMap<PHINode*, Constant*> ConstantEvolutionLoopExitValue;

    /// ValuesAtScopes - This map contains entries for all the expressions
    /// that we attempt to compute getSCEVAtScope information for, which can
    /// be expensive in extreme cases.
    DenseMap<const SCEV *,
             std::map<const Loop *, const SCEV *> > ValuesAtScopes;

    /// LoopDispositions - Memoized computeLoopDisposition results.
    DenseMap<const SCEV *,
             std::map<const Loop *, LoopDisposition> > LoopDispositions;

    /// computeLoopDisposition - Compute a LoopDisposition value.
    LoopDisposition computeLoopDisposition(const SCEV *S, const Loop *L);

    /// BlockDispositions - Memoized computeBlockDisposition results.
    DenseMap<const SCEV *,
             std::map<const BasicBlock *, BlockDisposition> > BlockDispositions;

    /// computeBlockDisposition - Compute a BlockDisposition value.
    BlockDisposition computeBlockDisposition(const SCEV *S, const BasicBlock *BB);

    /// UnsignedRanges - Memoized results from getUnsignedRange
    DenseMap<const SCEV *, ConstantRange> UnsignedRanges;

    /// SignedRanges - Memoized results from getSignedRange
    DenseMap<const SCEV *, ConstantRange> SignedRanges;

    /// setUnsignedRange - Set the memoized unsigned range for the given SCEV.
    const ConstantRange &setUnsignedRange(const SCEV *S,
                                          const ConstantRange &CR) {
      std::pair<DenseMap<const SCEV *, ConstantRange>::iterator, bool> Pair =
        UnsignedRanges.insert(std::make_pair(S, CR));
      if (!Pair.second)
        Pair.first->second = CR;
      return Pair.first->second;
    }

    /// setUnsignedRange - Set the memoized signed range for the given SCEV.
    const ConstantRange &setSignedRange(const SCEV *S,
                                        const ConstantRange &CR) {
      std::pair<DenseMap<const SCEV *, ConstantRange>::iterator, bool> Pair =
        SignedRanges.insert(std::make_pair(S, CR));
      if (!Pair.second)
        Pair.first->second = CR;
      return Pair.first->second;
    }

    /// createSCEV - We know that there is no SCEV for the specified value.
    /// Analyze the expression.
    const SCEV *createSCEV(Value *V);

    /// createNodeForPHI - Provide the special handling we need to analyze PHI
    /// SCEVs.
    const SCEV *createNodeForPHI(PHINode *PN);

    /// createNodeForGEP - Provide the special handling we need to analyze GEP
    /// SCEVs.
    const SCEV *createNodeForGEP(GEPOperator *GEP);

    /// computeSCEVAtScope - Implementation code for getSCEVAtScope; called
    /// at most once for each SCEV+Loop pair.
    ///
    const SCEV *computeSCEVAtScope(const SCEV *S, const Loop *L);

    /// ForgetSymbolicValue - This looks up computed SCEV values for all
    /// instructions that depend on the given instruction and removes them from
    /// the ValueExprMap map if they reference SymName. This is used during PHI
    /// resolution.
    void ForgetSymbolicName(Instruction *I, const SCEV *SymName);

    /// getBECount - Subtract the end and start values and divide by the step,
    /// rounding up, to get the number of times the backedge is executed. Return
    /// CouldNotCompute if an intermediate computation overflows.
    const SCEV *getBECount(const SCEV *Start,
                           const SCEV *End,
                           const SCEV *Step,
                           bool NoWrap);

    /// getBackedgeTakenInfo - Return the BackedgeTakenInfo for the given
    /// loop, lazily computing new values if the loop hasn't been analyzed
    /// yet.
    const BackedgeTakenInfo &getBackedgeTakenInfo(const Loop *L);

    /// ComputeBackedgeTakenCount - Compute the number of times the specified
    /// loop will iterate.
    BackedgeTakenInfo ComputeBackedgeTakenCount(const Loop *L);

    /// ComputeExitLimit - Compute the number of times the backedge of the
    /// specified loop will execute if it exits via the specified block.
    ExitLimit ComputeExitLimit(const Loop *L, BasicBlock *ExitingBlock);

    /// ComputeExitLimitFromCond - Compute the number of times the backedge of
    /// the specified loop will execute if its exit condition were a conditional
    /// branch of ExitCond, TBB, and FBB.
    ExitLimit ComputeExitLimitFromCond(const Loop *L,
                                       Value *ExitCond,
                                       BasicBlock *TBB,
                                       BasicBlock *FBB,
                                       bool IsSubExpr);

    /// ComputeExitLimitFromICmp - Compute the number of times the backedge of
    /// the specified loop will execute if its exit condition were a conditional
    /// branch of the ICmpInst ExitCond, TBB, and FBB.
    ExitLimit ComputeExitLimitFromICmp(const Loop *L,
                                       ICmpInst *ExitCond,
                                       BasicBlock *TBB,
                                       BasicBlock *FBB,
                                       bool IsSubExpr);

    /// ComputeLoadConstantCompareExitLimit - Given an exit condition
    /// of 'icmp op load X, cst', try to see if we can compute the
    /// backedge-taken count.
    ExitLimit ComputeLoadConstantCompareExitLimit(LoadInst *LI,
                                                  Constant *RHS,
                                                  const Loop *L,
                                                  ICmpInst::Predicate p);

    /// ComputeExitCountExhaustively - If the loop is known to execute a
    /// constant number of times (the condition evolves only from constants),
    /// try to evaluate a few iterations of the loop until we get the exit
    /// condition gets a value of ExitWhen (true or false).  If we cannot
    /// evaluate the exit count of the loop, return CouldNotCompute.
    const SCEV *ComputeExitCountExhaustively(const Loop *L,
                                             Value *Cond,
                                             bool ExitWhen);

    /// HowFarToZero - Return the number of times an exit condition comparing
    /// the specified value to zero will execute.  If not computable, return
    /// CouldNotCompute.
    ExitLimit HowFarToZero(const SCEV *V, const Loop *L, bool IsSubExpr);

    /// HowFarToNonZero - Return the number of times an exit condition checking
    /// the specified value for nonzero will execute.  If not computable, return
    /// CouldNotCompute.
    ExitLimit HowFarToNonZero(const SCEV *V, const Loop *L);

    /// HowManyLessThans - Return the number of times an exit condition
    /// containing the specified less-than comparison will execute.  If not
    /// computable, return CouldNotCompute. isSigned specifies whether the
    /// less-than is signed.
    ExitLimit HowManyLessThans(const SCEV *LHS, const SCEV *RHS,
                               const Loop *L, bool isSigned, bool IsSubExpr);

    /// getPredecessorWithUniqueSuccessorForBB - Return a predecessor of BB
    /// (which may not be an immediate predecessor) which has exactly one
    /// successor from which BB is reachable, or null if no such block is
    /// found.
    std::pair<BasicBlock *, BasicBlock *>
    getPredecessorWithUniqueSuccessorForBB(BasicBlock *BB);

    /// isImpliedCond - Test whether the condition described by Pred, LHS, and
    /// RHS is true whenever the given FoundCondValue value evaluates to true.
    bool isImpliedCond(ICmpInst::Predicate Pred,
                       const SCEV *LHS, const SCEV *RHS,
                       Value *FoundCondValue,
                       bool Inverse);

    /// isImpliedCondOperands - Test whether the condition described by Pred,
    /// LHS, and RHS is true whenever the condition described by Pred, FoundLHS,
    /// and FoundRHS is true.
    bool isImpliedCondOperands(ICmpInst::Predicate Pred,
                               const SCEV *LHS, const SCEV *RHS,
                               const SCEV *FoundLHS, const SCEV *FoundRHS);

    /// isImpliedCondOperandsHelper - Test whether the condition described by
    /// Pred, LHS, and RHS is true whenever the condition described by Pred,
    /// FoundLHS, and FoundRHS is true.
    bool isImpliedCondOperandsHelper(ICmpInst::Predicate Pred,
                                     const SCEV *LHS, const SCEV *RHS,
                                     const SCEV *FoundLHS,
                                     const SCEV *FoundRHS);

    /// getConstantEvolutionLoopExitValue - If we know that the specified Phi is
    /// in the header of its containing loop, we know the loop executes a
    /// constant number of times, and the PHI node is just a recurrence
    /// involving constants, fold it.
    Constant *getConstantEvolutionLoopExitValue(PHINode *PN, const APInt& BEs,
                                                const Loop *L);

    /// isKnownPredicateWithRanges - Test if the given expression is known to
    /// satisfy the condition described by Pred and the known constant ranges
    /// of LHS and RHS.
    ///
    bool isKnownPredicateWithRanges(ICmpInst::Predicate Pred,
                                    const SCEV *LHS, const SCEV *RHS);

    /// forgetMemoizedResults - Drop memoized information computed for S.
    void forgetMemoizedResults(const SCEV *S);

  public:
    static char ID; // Pass identification, replacement for typeid
    ScalarEvolution();

    LLVMContext &getContext() const { return F->getContext(); }

    /// isSCEVable - Test if values of the given type are analyzable within
    /// the SCEV framework. This primarily includes integer types, and it
    /// can optionally include pointer types if the ScalarEvolution class
    /// has access to target-specific information.
    bool isSCEVable(Type *Ty) const;

    /// getTypeSizeInBits - Return the size in bits of the specified type,
    /// for which isSCEVable must return true.
    uint64_t getTypeSizeInBits(Type *Ty) const;

    /// getEffectiveSCEVType - Return a type with the same bitwidth as
    /// the given type and which represents how SCEV will treat the given
    /// type, for which isSCEVable must return true. For pointer types,
    /// this is the pointer-sized integer type.
    Type *getEffectiveSCEVType(Type *Ty) const;

    /// getSCEV - Return a SCEV expression for the full generality of the
    /// specified expression.
    const SCEV *getSCEV(Value *V);

    const SCEV *getConstant(ConstantInt *V);
    const SCEV *getConstant(const APInt& Val);
    const SCEV *getConstant(Type *Ty, uint64_t V, bool isSigned = false);
    const SCEV *getTruncateExpr(const SCEV *Op, Type *Ty);
    const SCEV *getZeroExtendExpr(const SCEV *Op, Type *Ty);
    const SCEV *getSignExtendExpr(const SCEV *Op, Type *Ty);
    const SCEV *getAnyExtendExpr(const SCEV *Op, Type *Ty);
    const SCEV *getAddExpr(SmallVectorImpl<const SCEV *> &Ops,
                           SCEV::NoWrapFlags Flags = SCEV::FlagAnyWrap);
    const SCEV *getAddExpr(const SCEV *LHS, const SCEV *RHS,
                           SCEV::NoWrapFlags Flags = SCEV::FlagAnyWrap) {
      SmallVector<const SCEV *, 2> Ops;
      Ops.push_back(LHS);
      Ops.push_back(RHS);
      return getAddExpr(Ops, Flags);
    }
    const SCEV *getAddExpr(const SCEV *Op0, const SCEV *Op1, const SCEV *Op2,
                           SCEV::NoWrapFlags Flags = SCEV::FlagAnyWrap) {
      SmallVector<const SCEV *, 3> Ops;
      Ops.push_back(Op0);
      Ops.push_back(Op1);
      Ops.push_back(Op2);
      return getAddExpr(Ops, Flags);
    }
    const SCEV *getMulExpr(SmallVectorImpl<const SCEV *> &Ops,
                           SCEV::NoWrapFlags Flags = SCEV::FlagAnyWrap);
    const SCEV *getMulExpr(const SCEV *LHS, const SCEV *RHS,
                           SCEV::NoWrapFlags Flags = SCEV::FlagAnyWrap)
    {
      SmallVector<const SCEV *, 2> Ops;
      Ops.push_back(LHS);
      Ops.push_back(RHS);
      return getMulExpr(Ops, Flags);
    }
    const SCEV *getMulExpr(const SCEV *Op0, const SCEV *Op1, const SCEV *Op2,
                           SCEV::NoWrapFlags Flags = SCEV::FlagAnyWrap) {
      SmallVector<const SCEV *, 3> Ops;
      Ops.push_back(Op0);
      Ops.push_back(Op1);
      Ops.push_back(Op2);
      return getMulExpr(Ops, Flags);
    }
    const SCEV *getUDivExpr(const SCEV *LHS, const SCEV *RHS);
    const SCEV *getAddRecExpr(const SCEV *Start, const SCEV *Step,
                              const Loop *L, SCEV::NoWrapFlags Flags);
    const SCEV *getAddRecExpr(SmallVectorImpl<const SCEV *> &Operands,
                              const Loop *L, SCEV::NoWrapFlags Flags);
    const SCEV *getAddRecExpr(const SmallVectorImpl<const SCEV *> &Operands,
                              const Loop *L, SCEV::NoWrapFlags Flags) {
      SmallVector<const SCEV *, 4> NewOp(Operands.begin(), Operands.end());
      return getAddRecExpr(NewOp, L, Flags);
    }
    const SCEV *getSMaxExpr(const SCEV *LHS, const SCEV *RHS);
    const SCEV *getSMaxExpr(SmallVectorImpl<const SCEV *> &Operands);
    const SCEV *getUMaxExpr(const SCEV *LHS, const SCEV *RHS);
    const SCEV *getUMaxExpr(SmallVectorImpl<const SCEV *> &Operands);
    const SCEV *getSMinExpr(const SCEV *LHS, const SCEV *RHS);
    const SCEV *getUMinExpr(const SCEV *LHS, const SCEV *RHS);
    const SCEV *getUnknown(Value *V);
    const SCEV *getCouldNotCompute();

    /// getSizeOfExpr - Return an expression for sizeof on the given type.
    ///
    const SCEV *getSizeOfExpr(Type *AllocTy);

    /// getAlignOfExpr - Return an expression for alignof on the given type.
    ///
    const SCEV *getAlignOfExpr(Type *AllocTy);

    /// getOffsetOfExpr - Return an expression for offsetof on the given field.
    ///
    const SCEV *getOffsetOfExpr(StructType *STy, unsigned FieldNo);

    /// getOffsetOfExpr - Return an expression for offsetof on the given field.
    ///
    const SCEV *getOffsetOfExpr(Type *CTy, Constant *FieldNo);

    /// getNegativeSCEV - Return the SCEV object corresponding to -V.
    ///
    const SCEV *getNegativeSCEV(const SCEV *V);

    /// getNotSCEV - Return the SCEV object corresponding to ~V.
    ///
    const SCEV *getNotSCEV(const SCEV *V);

    /// getMinusSCEV - Return LHS-RHS.  Minus is represented in SCEV as A+B*-1.
    const SCEV *getMinusSCEV(const SCEV *LHS, const SCEV *RHS,
                             SCEV::NoWrapFlags Flags = SCEV::FlagAnyWrap);

    /// getTruncateOrZeroExtend - Return a SCEV corresponding to a conversion
    /// of the input value to the specified type.  If the type must be
    /// extended, it is zero extended.
    const SCEV *getTruncateOrZeroExtend(const SCEV *V, Type *Ty);

    /// getTruncateOrSignExtend - Return a SCEV corresponding to a conversion
    /// of the input value to the specified type.  If the type must be
    /// extended, it is sign extended.
    const SCEV *getTruncateOrSignExtend(const SCEV *V, Type *Ty);

    /// getNoopOrZeroExtend - Return a SCEV corresponding to a conversion of
    /// the input value to the specified type.  If the type must be extended,
    /// it is zero extended.  The conversion must not be narrowing.
    const SCEV *getNoopOrZeroExtend(const SCEV *V, Type *Ty);

    /// getNoopOrSignExtend - Return a SCEV corresponding to a conversion of
    /// the input value to the specified type.  If the type must be extended,
    /// it is sign extended.  The conversion must not be narrowing.
    const SCEV *getNoopOrSignExtend(const SCEV *V, Type *Ty);

    /// getNoopOrAnyExtend - Return a SCEV corresponding to a conversion of
    /// the input value to the specified type. If the type must be extended,
    /// it is extended with unspecified bits. The conversion must not be
    /// narrowing.
    const SCEV *getNoopOrAnyExtend(const SCEV *V, Type *Ty);

    /// getTruncateOrNoop - Return a SCEV corresponding to a conversion of the
    /// input value to the specified type.  The conversion must not be
    /// widening.
    const SCEV *getTruncateOrNoop(const SCEV *V, Type *Ty);

    /// getUMaxFromMismatchedTypes - Promote the operands to the wider of
    /// the types using zero-extension, and then perform a umax operation
    /// with them.
    const SCEV *getUMaxFromMismatchedTypes(const SCEV *LHS,
                                           const SCEV *RHS);

    /// getUMinFromMismatchedTypes - Promote the operands to the wider of
    /// the types using zero-extension, and then perform a umin operation
    /// with them.
    const SCEV *getUMinFromMismatchedTypes(const SCEV *LHS,
                                           const SCEV *RHS);

    /// getPointerBase - Transitively follow the chain of pointer-type operands
    /// until reaching a SCEV that does not have a single pointer operand. This
    /// returns a SCEVUnknown pointer for well-formed pointer-type expressions,
    /// but corner cases do exist.
    const SCEV *getPointerBase(const SCEV *V);

    /// getSCEVAtScope - Return a SCEV expression for the specified value
    /// at the specified scope in the program.  The L value specifies a loop
    /// nest to evaluate the expression at, where null is the top-level or a
    /// specified loop is immediately inside of the loop.
    ///
    /// This method can be used to compute the exit value for a variable defined
    /// in a loop by querying what the value will hold in the parent loop.
    ///
    /// In the case that a relevant loop exit value cannot be computed, the
    /// original value V is returned.
    const SCEV *getSCEVAtScope(const SCEV *S, const Loop *L);

    /// getSCEVAtScope - This is a convenience function which does
    /// getSCEVAtScope(getSCEV(V), L).
    const SCEV *getSCEVAtScope(Value *V, const Loop *L);

    /// isLoopEntryGuardedByCond - Test whether entry to the loop is protected
    /// by a conditional between LHS and RHS.  This is used to help avoid max
    /// expressions in loop trip counts, and to eliminate casts.
    bool isLoopEntryGuardedByCond(const Loop *L, ICmpInst::Predicate Pred,
                                  const SCEV *LHS, const SCEV *RHS);

    /// isLoopBackedgeGuardedByCond - Test whether the backedge of the loop is
    /// protected by a conditional between LHS and RHS.  This is used to
    /// to eliminate casts.
    bool isLoopBackedgeGuardedByCond(const Loop *L, ICmpInst::Predicate Pred,
                                     const SCEV *LHS, const SCEV *RHS);

    /// getSmallConstantTripCount - Returns the maximum trip count of this loop
    /// as a normal unsigned value. Returns 0 if the trip count is unknown or
    /// not constant. This "trip count" assumes that control exits via
    /// ExitingBlock. More precisely, it is the number of times that control may
    /// reach ExitingBlock before taking the branch. For loops with multiple
    /// exits, it may not be the number times that the loop header executes if
    /// the loop exits prematurely via another branch.
    unsigned getSmallConstantTripCount(Loop *L, BasicBlock *ExitingBlock);

    /// getSmallConstantTripMultiple - Returns the largest constant divisor of
    /// the trip count of this loop as a normal unsigned value, if
    /// possible. This means that the actual trip count is always a multiple of
    /// the returned value (don't forget the trip count could very well be zero
    /// as well!). As explained in the comments for getSmallConstantTripCount,
    /// this assumes that control exits the loop via ExitingBlock.
    unsigned getSmallConstantTripMultiple(Loop *L, BasicBlock *ExitingBlock);

    // getExitCount - Get the expression for the number of loop iterations for
    // which this loop is guaranteed not to exit via ExitingBlock. Otherwise
    // return SCEVCouldNotCompute.
    const SCEV *getExitCount(Loop *L, BasicBlock *ExitingBlock);

    /// getBackedgeTakenCount - If the specified loop has a predictable
    /// backedge-taken count, return it, otherwise return a SCEVCouldNotCompute
    /// object. The backedge-taken count is the number of times the loop header
    /// will be branched to from within the loop. This is one less than the
    /// trip count of the loop, since it doesn't count the first iteration,
    /// when the header is branched to from outside the loop.
    ///
    /// Note that it is not valid to call this method on a loop without a
    /// loop-invariant backedge-taken count (see
    /// hasLoopInvariantBackedgeTakenCount).
    ///
    const SCEV *getBackedgeTakenCount(const Loop *L);

    /// getMaxBackedgeTakenCount - Similar to getBackedgeTakenCount, except
    /// return the least SCEV value that is known never to be less than the
    /// actual backedge taken count.
    const SCEV *getMaxBackedgeTakenCount(const Loop *L);

    /// hasLoopInvariantBackedgeTakenCount - Return true if the specified loop
    /// has an analyzable loop-invariant backedge-taken count.
    bool hasLoopInvariantBackedgeTakenCount(const Loop *L);

    /// forgetLoop - This method should be called by the client when it has
    /// changed a loop in a way that may effect ScalarEvolution's ability to
    /// compute a trip count, or if the loop is deleted.
    void forgetLoop(const Loop *L);

    /// forgetValue - This method should be called by the client when it has
    /// changed a value in a way that may effect its value, or which may
    /// disconnect it from a def-use chain linking it to a loop.
    void forgetValue(Value *V);

    /// GetMinTrailingZeros - Determine the minimum number of zero bits that S
    /// is guaranteed to end in (at every loop iteration).  It is, at the same
    /// time, the minimum number of times S is divisible by 2.  For example,
    /// given {4,+,8} it returns 2.  If S is guaranteed to be 0, it returns the
    /// bitwidth of S.
    uint32_t GetMinTrailingZeros(const SCEV *S);

    /// getUnsignedRange - Determine the unsigned range for a particular SCEV.
    ///
    ConstantRange getUnsignedRange(const SCEV *S);

    /// getSignedRange - Determine the signed range for a particular SCEV.
    ///
    ConstantRange getSignedRange(const SCEV *S);

    /// isKnownNegative - Test if the given expression is known to be negative.
    ///
    bool isKnownNegative(const SCEV *S);

    /// isKnownPositive - Test if the given expression is known to be positive.
    ///
    bool isKnownPositive(const SCEV *S);

    /// isKnownNonNegative - Test if the given expression is known to be
    /// non-negative.
    ///
    bool isKnownNonNegative(const SCEV *S);

    /// isKnownNonPositive - Test if the given expression is known to be
    /// non-positive.
    ///
    bool isKnownNonPositive(const SCEV *S);

    /// isKnownNonZero - Test if the given expression is known to be
    /// non-zero.
    ///
    bool isKnownNonZero(const SCEV *S);

    /// isKnownPredicate - Test if the given expression is known to satisfy
    /// the condition described by Pred, LHS, and RHS.
    ///
    bool isKnownPredicate(ICmpInst::Predicate Pred,
                          const SCEV *LHS, const SCEV *RHS);

    /// SimplifyICmpOperands - Simplify LHS and RHS in a comparison with
    /// predicate Pred. Return true iff any changes were made. If the
    /// operands are provably equal or unequal, LHS and RHS are set to
    /// the same value and Pred is set to either ICMP_EQ or ICMP_NE.
    ///
    bool SimplifyICmpOperands(ICmpInst::Predicate &Pred,
                              const SCEV *&LHS,
                              const SCEV *&RHS,
                              unsigned Depth = 0);

    /// getLoopDisposition - Return the "disposition" of the given SCEV with
    /// respect to the given loop.
    LoopDisposition getLoopDisposition(const SCEV *S, const Loop *L);

    /// isLoopInvariant - Return true if the value of the given SCEV is
    /// unchanging in the specified loop.
    bool isLoopInvariant(const SCEV *S, const Loop *L);

    /// hasComputableLoopEvolution - Return true if the given SCEV changes value
    /// in a known way in the specified loop.  This property being true implies
    /// that the value is variant in the loop AND that we can emit an expression
    /// to compute the value of the expression at any particular loop iteration.
    bool hasComputableLoopEvolution(const SCEV *S, const Loop *L);

    /// getLoopDisposition - Return the "disposition" of the given SCEV with
    /// respect to the given block.
    BlockDisposition getBlockDisposition(const SCEV *S, const BasicBlock *BB);

    /// dominates - Return true if elements that makes up the given SCEV
    /// dominate the specified basic block.
    bool dominates(const SCEV *S, const BasicBlock *BB);

    /// properlyDominates - Return true if elements that makes up the given SCEV
    /// properly dominate the specified basic block.
    bool properlyDominates(const SCEV *S, const BasicBlock *BB);

    /// hasOperand - Test whether the given SCEV has Op as a direct or
    /// indirect operand.
    bool hasOperand(const SCEV *S, const SCEV *Op) const;

    virtual bool runOnFunction(Function &F);
    virtual void releaseMemory();
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual void print(raw_ostream &OS, const Module* = 0) const;
    virtual void verifyAnalysis() const;

  private:
    FoldingSet<SCEV> UniqueSCEVs;
    BumpPtrAllocator SCEVAllocator;

    /// FirstUnknown - The head of a linked list of all SCEVUnknown
    /// values that have been allocated. This is used by releaseMemory
    /// to locate them all and call their destructors.
    SCEVUnknown *FirstUnknown;
  };
}

#endif
