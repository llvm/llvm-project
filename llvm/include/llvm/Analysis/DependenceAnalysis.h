//===-- llvm/Analysis/DependenceAnalysis.h -------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DependenceAnalysis is an LLVM pass that analyses dependences between memory
// accesses. Currently, it is an implementation of the approach described in
//
//            Practical Dependence Testing
//            Goff, Kennedy, Tseng
//            PLDI 1991
//
// There's a single entry point that analyzes the dependence between a pair
// of memory references in a function, returning either NULL, for no dependence,
// or a more-or-less detailed description of the dependence between them.
//
// This pass exists to support the DependenceGraph pass. There are two separate
// passes because there's a useful separation of concerns. A dependence exists
// if two conditions are met:
//
//    1) Two instructions reference the same memory location, and
//    2) There is a flow of control leading from one instruction to the other.
//
// DependenceAnalysis attacks the first condition; DependenceGraph will attack
// the second (it's not yet ready).
//
// Please note that this is work in progress and the interface is subject to
// change.
//
// Plausible changes:
//    Return a set of more precise dependences instead of just one dependence
//    summarizing all.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DEPENDENCEANALYSIS_H
#define LLVM_ANALYSIS_DEPENDENCEANALYSIS_H

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class AAResults;
template <typename T> class ArrayRef;
class Loop;
class LoopInfo;
class SCEVConstant;
class raw_ostream;

/// Dependence - This class represents a dependence between two memory
/// memory references in a function. It contains minimal information and
/// is used in the very common situation where the compiler is unable to
/// determine anything beyond the existence of a dependence; that is, it
/// represents a confused dependence (see also FullDependence). In most
/// cases (for output, flow, and anti dependences), the dependence implies
/// an ordering, where the source must precede the destination; in contrast,
/// input dependences are unordered.
///
/// When a dependence graph is built, each Dependence will be a member of
/// the set of predecessor edges for its destination instruction and a set
/// if successor edges for its source instruction. These sets are represented
/// as singly-linked lists, with the "next" fields stored in the dependence
/// itelf.
class LLVM_ABI Dependence {
protected:
  Dependence(Dependence &&) = default;
  Dependence &operator=(Dependence &&) = default;

public:
  Dependence(Instruction *Source, Instruction *Destination,
             const SCEVUnionPredicate &A)
      : Src(Source), Dst(Destination), Assumptions(A) {}
  virtual ~Dependence() = default;

  /// Dependence::DVEntry - Each level in the distance/direction vector
  /// has a direction (or perhaps a union of several directions), and
  /// perhaps a distance.
  /// The dependency information could be across a single loop level or across
  /// two separate levels that have the same trip count and nesting depth,
  /// which helps to provide information for loop fusion candidation.
  /// For example, loops b and c have the same iteration count and depth:
  ///    for (a = ...) {
  ///      for (b = 0; b < 10; b++) {
  ///      }
  ///      for (c = 0; c < 10; c++) {
  ///      }
  ///    }
  struct DVEntry {
    enum : unsigned char {
      NONE = 0,
      LT = 1,
      EQ = 2,
      LE = 3,
      GT = 4,
      NE = 5,
      GE = 6,
      ALL = 7
    };
    unsigned char Direction : 3; // Init to ALL, then refine.
    bool Scalar : 1;             // Init to true.
    bool PeelFirst : 1; // Peeling the first iteration will break dependence.
    bool PeelLast : 1;  // Peeling the last iteration will break the dependence.
    const SCEV *Distance = nullptr; // NULL implies no distance available.
    DVEntry()
        : Direction(ALL), Scalar(true), PeelFirst(false), PeelLast(false) {}
  };

  /// getSrc - Returns the source instruction for this dependence.
  Instruction *getSrc() const { return Src; }

  /// getDst - Returns the destination instruction for this dependence.
  Instruction *getDst() const { return Dst; }

  /// isInput - Returns true if this is an input dependence.
  bool isInput() const;

  /// isOutput - Returns true if this is an output dependence.
  bool isOutput() const;

  /// isFlow - Returns true if this is a flow (aka true) dependence.
  bool isFlow() const;

  /// isAnti - Returns true if this is an anti dependence.
  bool isAnti() const;

  /// isOrdered - Returns true if dependence is Output, Flow, or Anti
  bool isOrdered() const { return isOutput() || isFlow() || isAnti(); }

  /// isUnordered - Returns true if dependence is Input
  bool isUnordered() const { return isInput(); }

  /// isLoopIndependent - Returns true if this is a loop-independent
  /// dependence.
  virtual bool isLoopIndependent() const { return true; }

  /// isConfused - Returns true if this dependence is confused
  /// (the compiler understands nothing and makes worst-case assumptions).
  virtual bool isConfused() const { return true; }

  /// isConsistent - Returns true if this dependence is consistent
  /// (occurs every time the source and destination are executed).
  virtual bool isConsistent() const { return false; }

  /// getLevels - Returns the number of common loops surrounding the
  /// source and destination of the dependence.
  virtual unsigned getLevels() const { return 0; }

  /// getSameSDLevels - Returns the number of separate SameSD loops surrounding
  /// the source and destination of the dependence.
  virtual unsigned getSameSDLevels() const { return 0; }

  /// getDVEntry - Returns the DV entry associated with a regular or a
  /// SameSD level
  DVEntry getDVEntry(unsigned Level, bool IsSameSD) const;

  /// getDirection - Returns the direction associated with a particular
  /// common or SameSD level.
  virtual unsigned getDirection(unsigned Level, bool SameSD = false) const {
    return DVEntry::ALL;
  }

  /// getDistance - Returns the distance (or NULL) associated with a
  /// particular common or SameSD level.
  virtual const SCEV *getDistance(unsigned Level, bool SameSD = false) const {
    return nullptr;
  }

  /// Check if the direction vector is negative. A negative direction
  /// vector means Src and Dst are reversed in the actual program.
  virtual bool isDirectionNegative() const { return false; }

  /// If the direction vector is negative, normalize the direction
  /// vector to make it non-negative. Normalization is done by reversing
  /// Src and Dst, plus reversing the dependence directions and distances
  /// in the vector.
  virtual bool normalize(ScalarEvolution *SE) { return false; }

  /// isPeelFirst - Returns true if peeling the first iteration from
  /// this regular or SameSD loop level will break this dependence.
  virtual bool isPeelFirst(unsigned Level, bool SameSD = false) const {
    return false;
  }

  /// isPeelLast - Returns true if peeling the last iteration from
  /// this regular or SameSD loop level will break this dependence.
  virtual bool isPeelLast(unsigned Level, bool SameSD = false) const {
    return false;
  }

  /// inSameSDLoops - Returns true if this level is an SameSD level, i.e.,
  /// performed across two separate loop nests that have the Same Iteration and
  /// Depth.
  virtual bool inSameSDLoops(unsigned Level) const { return false; }

  /// isScalar - Returns true if a particular regular or SameSD level is
  /// scalar; that is, if no subscript in the source or destination mention
  /// the induction variable associated with the loop at this level.
  virtual bool isScalar(unsigned Level, bool SameSD = false) const;

  /// getNextPredecessor - Returns the value of the NextPredecessor field.
  const Dependence *getNextPredecessor() const { return NextPredecessor; }

  /// getNextSuccessor - Returns the value of the NextSuccessor field.
  const Dependence *getNextSuccessor() const { return NextSuccessor; }

  /// setNextPredecessor - Sets the value of the NextPredecessor
  /// field.
  void setNextPredecessor(const Dependence *pred) { NextPredecessor = pred; }

  /// setNextSuccessor - Sets the value of the NextSuccessor field.
  void setNextSuccessor(const Dependence *succ) { NextSuccessor = succ; }

  /// getRuntimeAssumptions - Returns the runtime assumptions under which this
  /// Dependence relation is valid.
  SCEVUnionPredicate getRuntimeAssumptions() const { return Assumptions; }

  /// dump - For debugging purposes, dumps a dependence to OS.
  void dump(raw_ostream &OS) const;

  /// dumpImp - For debugging purposes. Dumps a dependence to OS with or
  /// without considering the SameSD levels.
  void dumpImp(raw_ostream &OS, bool IsSameSD = false) const;

protected:
  Instruction *Src, *Dst;

private:
  SCEVUnionPredicate Assumptions;
  const Dependence *NextPredecessor = nullptr, *NextSuccessor = nullptr;
  friend class DependenceInfo;
};

/// FullDependence - This class represents a dependence between two memory
/// references in a function. It contains detailed information about the
/// dependence (direction vectors, etc.) and is used when the compiler is
/// able to accurately analyze the interaction of the references; that is,
/// it is not a confused dependence (see Dependence). In most cases
/// (for output, flow, and anti dependences), the dependence implies an
/// ordering, where the source must precede the destination; in contrast,
/// input dependences are unordered.
class LLVM_ABI FullDependence final : public Dependence {
public:
  FullDependence(Instruction *Source, Instruction *Destination,
                 const SCEVUnionPredicate &Assumes,
                 bool PossiblyLoopIndependent, unsigned Levels);

  /// isLoopIndependent - Returns true if this is a loop-independent
  /// dependence.
  bool isLoopIndependent() const override { return LoopIndependent; }

  /// isConfused - Returns true if this dependence is confused
  /// (the compiler understands nothing and makes worst-case
  /// assumptions).
  bool isConfused() const override { return false; }

  /// isConsistent - Returns true if this dependence is consistent
  /// (occurs every time the source and destination are executed).
  bool isConsistent() const override { return Consistent; }

  /// getLevels - Returns the number of common loops surrounding the
  /// source and destination of the dependence.
  unsigned getLevels() const override { return Levels; }

  /// getSameSDLevels - Returns the number of separate SameSD loops surrounding
  /// the source and destination of the dependence.
  unsigned getSameSDLevels() const override { return SameSDLevels; }

  /// getDVEntry - Returns the DV entry associated with a regular or a
  /// SameSD level.
  DVEntry getDVEntry(unsigned Level, bool IsSameSD) const {
    if (!IsSameSD) {
      assert(0 < Level && Level <= Levels && "Level out of range");
      return DV[Level - 1];
    } else {
      assert(Levels < Level &&
             Level <= static_cast<unsigned>(Levels) + SameSDLevels &&
             "isSameSD level out of range");
      return DVSameSD[Level - Levels - 1];
    }
  }

  /// getDirection - Returns the direction associated with a particular
  /// common or SameSD level.
  unsigned getDirection(unsigned Level, bool SameSD = false) const override;

  /// getDistance - Returns the distance (or NULL) associated with a
  /// particular common or SameSD level.
  const SCEV *getDistance(unsigned Level, bool SameSD = false) const override;

  /// Check if the direction vector is negative. A negative direction
  /// vector means Src and Dst are reversed in the actual program.
  bool isDirectionNegative() const override;

  /// If the direction vector is negative, normalize the direction
  /// vector to make it non-negative. Normalization is done by reversing
  /// Src and Dst, plus reversing the dependence directions and distances
  /// in the vector.
  bool normalize(ScalarEvolution *SE) override;

  /// isPeelFirst - Returns true if peeling the first iteration from
  /// this regular or SameSD loop level will break this dependence.
  bool isPeelFirst(unsigned Level, bool SameSD = false) const override;

  /// isPeelLast - Returns true if peeling the last iteration from
  /// this regular or SameSD loop level will break this dependence.
  bool isPeelLast(unsigned Level, bool SameSD = false) const override;

  /// inSameSDLoops - Returns true if this level is an SameSD level, i.e.,
  /// performed across two separate loop nests that have the Same Iteration and
  /// Depth.
  bool inSameSDLoops(unsigned Level) const override;

  /// isScalar - Returns true if a particular regular or SameSD level is
  /// scalar; that is, if no subscript in the source or destination mention
  /// the induction variable associated with the loop at this level.
  bool isScalar(unsigned Level, bool SameSD = false) const override;

private:
  unsigned short Levels;
  unsigned short SameSDLevels;
  bool LoopIndependent;
  bool Consistent; // Init to true, then refine.
  std::unique_ptr<DVEntry[]> DV;
  std::unique_ptr<DVEntry[]> DVSameSD; // DV entries on SameSD levels
  friend class DependenceInfo;
};

/// The property of monotonicity of a SCEV. To define the monotonicity, assume
/// a SCEV defined within N-nested loops. Let i_k denote the iteration number
/// of the k-th loop. Then we can regard the SCEV as an N-ary function:
///
///   F(i_1, i_2, ..., i_N)
///
/// For the domain of F, see the comment of SCEVMonotonicityDomain.
///
/// A function F is said to be "monotonically increasing with respect to the
/// k-th loop" if x <= y implies the following condition:
///
///   F(i_1, ..., i_{k-1}, x, i_{k+1}, ..., i_N) <=
///   F(i_1, ..., i_{k-1}, y, i_{k+1}, ..., i_N)
///
/// where i_1, ..., i_{k-1}, i_{k+1}, ..., i_N, x, and y are elements of their
/// respective domains.
///
/// Likewise F is "monotonically decreasing with respect to the k-th loop"
/// if x <= y implies
///
///   F(i_1, ..., i_{k-1}, x, i_{k+1}, ..., i_N) >=
///   F(i_1, ..., i_{k-1}, y, i_{k+1}, ..., i_N)
///
/// A function F that is monotonically increasing or decreasing with respect to
/// the k-th loop is simply called "monotonic with respect to k-th loop".
///
/// A function F is said to be "multivariate monotonic" when it is monotonic
/// with respect to all of the N loops.
///
/// Since integer comparison can be either signed or unsigned, we need to
/// distinguish monotonicity in the signed sense from that in the unsigned
/// sense. Note that the inequality "x <= y" merely indicates loop progression
/// and is not affected by the difference between signed and unsigned order.
///
/// Currently we only consider monotonicity in a signed sense.
enum class SCEVMonotonicityType {
  /// We don't know anything about the monotonicity of the SCEV.
  Unknown,

  /// The SCEV is loop-invariant with respect to the outermost loop. In other
  /// words, the function F corresponding to the SCEV is a constant function.
  Invariant,

  /// The function F corresponding to the SCEV is multivariate monotonic in a
  /// signed sense. Note that the multivariate monotonic function may also be a
  /// constant function. The order employed in the definition of monotonicity
  /// is not strict order.
  MultivariateSignedMonotonic,
};

/// The domain for checking monotonicity of a SCEV which is represented as a
/// function F(i_1, i_2, ..., i_N). Given a domain D, we say "F is multivariate
/// monotonic over the domain D" if F is multivariate monotonic over D.
enum class SCEVMonotonicityDomain {
  /// [0, BTC_1] x [0, BTC_2] x ... x [0, BTC_N], where BTC_k is the exact
  /// backedge-taken count for the k-th loop. This domain is well-defined only
  /// when all loops have exact backedge-taken counts.
  EntireDomain,

  /// [L_1, U_1] x [L_2, U_2] x ... x [L_N, U_N].
  /// When we say "F is multivariate monotonic over effective domain", it means:
  ///
  /// \exists L_k, U_k for all k, such that
  ///   - F is multivariate monotonic over [L_1, U_1] x ... x [L_N, U_N] and
  ///   - [L_1, U_1] x ... x [L_N, U_N] is a superset of where F is actually
  ///     executed.
  ///
  /// That is, we only implied the existence of such L_k and U_k, without any
  /// specific values.
  EffectiveDomain,
};

struct SCEVMonotonicity {
  SCEVMonotonicity(SCEVMonotonicityType Type,
                   const SCEV *FailurePoint = nullptr);

  SCEVMonotonicityType getType() const { return Type; }

  const SCEV *getFailurePoint() const { return FailurePoint; }

  bool isUnknown() const { return Type == SCEVMonotonicityType::Unknown; }

  void print(raw_ostream &OS, unsigned Depth) const;

private:
  SCEVMonotonicityType Type;

  /// The subexpression that caused Unknown. Mainly for debugging purpose.
  const SCEV *FailurePoint;
};

/// Check the monotonicity of a SCEV. Since dependence tests (SIV, MIV, etc.)
/// assume that subscript expressions are (multivariate) monotonic, we need to
/// verify this property before applying those tests. Violating this assumption
/// may cause them to produce incorrect results.
struct SCEVMonotonicityChecker
    : public SCEVVisitor<SCEVMonotonicityChecker, SCEVMonotonicity> {

  SCEVMonotonicityChecker(ScalarEvolution *SE) : SE(SE) {}

  /// Check the monotonicity of \p Expr. \p Expr must be integer type. If \p
  /// OutermostLoop is not null, \p Expr must be defined in \p OutermostLoop or
  /// one of its nested loops.
  SCEVMonotonicity checkMonotonicity(const SCEV *Expr,
                                     const Loop *OutermostLoop,
                                     SCEVMonotonicityDomain Domain);

private:
  ScalarEvolution *SE;

  struct Context {
    /// The outermost loop that DA is analyzing.
    const Loop *OutermostLoop;

    bool FoundInnermostAddRec = false;

    SCEVMonotonicityDomain Domain;

    void clear() {
      OutermostLoop = nullptr;
      FoundInnermostAddRec = false;
    }
  } Ctx;

  /// A helper to classify \p Expr as either Invariant or Unknown.
  SCEVMonotonicity invariantOrUnknown(const SCEV *Expr);

  /// Return true if \p Expr is loop-invariant with respect to the outermost
  /// loop.
  bool isLoopInvariant(const SCEV *Expr) const;

  /// A helper to create an Unknown SCEVMonotonicity.
  SCEVMonotonicity createUnknown(const SCEV *FailurePoint) {
    return SCEVMonotonicity(SCEVMonotonicityType::Unknown, FailurePoint);
  }

  SCEVMonotonicity visitAddRecExpr(const SCEVAddRecExpr *Expr);

  SCEVMonotonicity visitConstant(const SCEVConstant *) {
    return SCEVMonotonicity(SCEVMonotonicityType::Invariant);
  }
  SCEVMonotonicity visitVScale(const SCEVVScale *) {
    return SCEVMonotonicity(SCEVMonotonicityType::Invariant);
  }

  // TODO: Handle more cases.
  SCEVMonotonicity visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
    return invariantOrUnknown(Expr);
  }
  SCEVMonotonicity visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
    return invariantOrUnknown(Expr);
  }
  SCEVMonotonicity visitAddExpr(const SCEVAddExpr *Expr) {
    return invariantOrUnknown(Expr);
  }
  SCEVMonotonicity visitMulExpr(const SCEVMulExpr *Expr) {
    return invariantOrUnknown(Expr);
  }
  SCEVMonotonicity visitPtrToIntExpr(const SCEVPtrToIntExpr *Expr) {
    return invariantOrUnknown(Expr);
  }
  SCEVMonotonicity visitTruncateExpr(const SCEVTruncateExpr *Expr) {
    return invariantOrUnknown(Expr);
  }
  SCEVMonotonicity visitUDivExpr(const SCEVUDivExpr *Expr) {
    return invariantOrUnknown(Expr);
  }
  SCEVMonotonicity visitSMaxExpr(const SCEVSMaxExpr *Expr) {
    return invariantOrUnknown(Expr);
  }
  SCEVMonotonicity visitUMaxExpr(const SCEVUMaxExpr *Expr) {
    return invariantOrUnknown(Expr);
  }
  SCEVMonotonicity visitSMinExpr(const SCEVSMinExpr *Expr) {
    return invariantOrUnknown(Expr);
  }
  SCEVMonotonicity visitUMinExpr(const SCEVUMinExpr *Expr) {
    return invariantOrUnknown(Expr);
  }
  SCEVMonotonicity visitSequentialUMinExpr(const SCEVSequentialUMinExpr *Expr) {
    return invariantOrUnknown(Expr);
  }
  SCEVMonotonicity visitUnknown(const SCEVUnknown *Expr) {
    return invariantOrUnknown(Expr);
  }
  SCEVMonotonicity visitCouldNotCompute(const SCEVCouldNotCompute *Expr) {
    return invariantOrUnknown(Expr);
  }

  friend struct SCEVVisitor<SCEVMonotonicityChecker, SCEVMonotonicity>;
};

/// DependenceInfo - This class is the main dependence-analysis driver.
class DependenceInfo {
public:
  DependenceInfo(Function *F, AAResults *AA, ScalarEvolution *SE, LoopInfo *LI)
      : AA(AA), SE(SE), LI(LI), F(F), MonChecker(SE) {}

  /// Handle transitive invalidation when the cached analysis results go away.
  LLVM_ABI bool invalidate(Function &F, const PreservedAnalyses &PA,
                           FunctionAnalysisManager::Invalidator &Inv);

  /// depends - Tests for a dependence between the Src and Dst instructions.
  /// Returns NULL if no dependence; otherwise, returns a Dependence (or a
  /// FullDependence) with as much information as can be gleaned. By default,
  /// the dependence test collects a set of runtime assumptions that cannot be
  /// solved at compilation time. By default UnderRuntimeAssumptions is false
  /// for a safe approximation of the dependence relation that does not
  /// require runtime checks.
  LLVM_ABI std::unique_ptr<Dependence>
  depends(Instruction *Src, Instruction *Dst,
          bool UnderRuntimeAssumptions = false);

  Function *getFunction() const { return F; }

private:
  AAResults *AA;
  ScalarEvolution *SE;
  LoopInfo *LI;
  Function *F;
  SmallVector<const SCEVPredicate *, 4> Assumptions;
  SCEVMonotonicityChecker MonChecker;

  /// Subscript - This private struct represents a pair of subscripts from
  /// a pair of potentially multi-dimensional array references. We use a
  /// vector of them to guide subscript partitioning.
  struct Subscript {
    const SCEV *Src;
    const SCEV *Dst;
    enum ClassificationKind { ZIV, SIV, RDIV, MIV, NonLinear } Classification;
    SmallBitVector Loops;
    SmallBitVector GroupLoops;
    SmallBitVector Group;
  };

  struct CoefficientInfo {
    const SCEV *Coeff;
    const SCEV *PosPart;
    const SCEV *NegPart;
    const SCEV *Iterations;
  };

  struct BoundInfo {
    const SCEV *Iterations;
    const SCEV *Upper[8];
    const SCEV *Lower[8];
    unsigned char Direction;
    unsigned char DirSet;
  };

  /// Returns true if two loops have the Same iteration Space and Depth. To be
  /// more specific, two loops have SameSD if they are in the same nesting
  /// depth and have the same backedge count. SameSD stands for Same iteration
  /// Space and Depth.
  bool haveSameSD(const Loop *SrcLoop, const Loop *DstLoop) const;

  /// establishNestingLevels - Examines the loop nesting of the Src and Dst
  /// instructions and establishes their shared loops. Sets the variables
  /// CommonLevels, SrcLevels, and MaxLevels.
  /// The source and destination instructions needn't be contained in the same
  /// loop. The routine establishNestingLevels finds the level of most deeply
  /// nested loop that contains them both, CommonLevels. An instruction that's
  /// not contained in a loop is at level = 0. MaxLevels is equal to the level
  /// of the source plus the level of the destination, minus CommonLevels.
  /// This lets us allocate vectors MaxLevels in length, with room for every
  /// distinct loop referenced in both the source and destination subscripts.
  /// The variable SrcLevels is the nesting depth of the source instruction.
  /// It's used to help calculate distinct loops referenced by the destination.
  /// Here's the map from loops to levels:
  ///            0 - unused
  ///            1 - outermost common loop
  ///          ... - other common loops
  /// CommonLevels - innermost common loop
  ///          ... - loops containing Src but not Dst
  ///    SrcLevels - innermost loop containing Src but not Dst
  ///          ... - loops containing Dst but not Src
  ///    MaxLevels - innermost loop containing Dst but not Src
  /// Consider the follow code fragment:
  ///    for (a = ...) {
  ///      for (b = ...) {
  ///        for (c = ...) {
  ///          for (d = ...) {
  ///            A[] = ...;
  ///          }
  ///        }
  ///        for (e = ...) {
  ///          for (f = ...) {
  ///            for (g = ...) {
  ///              ... = A[];
  ///            }
  ///          }
  ///        }
  ///      }
  ///    }
  /// If we're looking at the possibility of a dependence between the store
  /// to A (the Src) and the load from A (the Dst), we'll note that they
  /// have 2 loops in common, so CommonLevels will equal 2 and the direction
  /// vector for Result will have 2 entries. SrcLevels = 4 and MaxLevels = 7.
  /// A map from loop names to level indices would look like
  ///     a - 1
  ///     b - 2 = CommonLevels
  ///     c - 3
  ///     d - 4 = SrcLevels
  ///     e - 5
  ///     f - 6
  ///     g - 7 = MaxLevels
  /// SameSDLevels counts the number of levels after common levels that are
  /// not common but have the same iteration space and depth. Internally this
  /// is checked using haveSameSD. Assume that in this code fragment, levels c
  /// and e have the same iteration space and depth, but levels d and f does
  /// not. Then SameSDLevels is set to 1. In that case the level numbers for the
  /// previous code look like
  ///     a   - 1
  ///     b   - 2
  ///     c,e - 3 = CommonLevels
  ///     d   - 4 = SrcLevels
  ///     f   - 5
  ///     g   - 6 = MaxLevels
  void establishNestingLevels(const Instruction *Src, const Instruction *Dst);

  unsigned CommonLevels, SrcLevels, MaxLevels, SameSDLevels;

  /// mapSrcLoop - Given one of the loops containing the source, return
  /// its level index in our numbering scheme.
  unsigned mapSrcLoop(const Loop *SrcLoop) const;

  /// mapDstLoop - Given one of the loops containing the destination,
  /// return its level index in our numbering scheme.
  unsigned mapDstLoop(const Loop *DstLoop) const;

  /// isLoopInvariant - Returns true if Expression is loop invariant
  /// in LoopNest.
  bool isLoopInvariant(const SCEV *Expression, const Loop *LoopNest) const;

  /// Makes sure all subscript pairs share the same integer type by
  /// sign-extending as necessary.
  /// Sign-extending a subscript is safe because getelementptr assumes the
  /// array subscripts are signed.
  void unifySubscriptType(ArrayRef<Subscript *> Pairs);

  /// removeMatchingExtensions - Examines a subscript pair.
  /// If the source and destination are identically sign (or zero)
  /// extended, it strips off the extension in an effort to
  /// simplify the actual analysis.
  void removeMatchingExtensions(Subscript *Pair);

  /// collectCommonLoops - Finds the set of loops from the LoopNest that
  /// have a level <= CommonLevels and are referred to by the SCEV Expression.
  void collectCommonLoops(const SCEV *Expression, const Loop *LoopNest,
                          SmallBitVector &Loops) const;

  /// checkSrcSubscript - Examines the SCEV Src, returning true iff it's
  /// linear. Collect the set of loops mentioned by Src.
  bool checkSrcSubscript(const SCEV *Src, const Loop *LoopNest,
                         SmallBitVector &Loops);

  /// checkDstSubscript - Examines the SCEV Dst, returning true iff it's
  /// linear. Collect the set of loops mentioned by Dst.
  bool checkDstSubscript(const SCEV *Dst, const Loop *LoopNest,
                         SmallBitVector &Loops);

  /// isKnownPredicate - Compare X and Y using the predicate Pred.
  /// Basically a wrapper for SCEV::isKnownPredicate,
  /// but tries harder, especially in the presence of sign and zero
  /// extensions and symbolics.
  bool isKnownPredicate(ICmpInst::Predicate Pred, const SCEV *X,
                        const SCEV *Y) const;

  /// collectUpperBound - All subscripts are the same type (on my machine,
  /// an i64). The loop bound may be a smaller type. collectUpperBound
  /// find the bound, if available, and zero extends it to the Type T.
  /// (I zero extend since the bound should always be >= 0.)
  /// If no upper bound is available, return NULL.
  const SCEV *collectUpperBound(const Loop *l, Type *T) const;

  /// collectConstantUpperBound - Calls collectUpperBound(), then
  /// attempts to cast it to SCEVConstant. If the cast fails,
  /// returns NULL.
  const SCEVConstant *collectConstantUpperBound(const Loop *l, Type *T) const;

  /// classifyPair - Examines the subscript pair (the Src and Dst SCEVs)
  /// and classifies it as either ZIV, SIV, RDIV, MIV, or Nonlinear.
  /// Collects the associated loops in a set.
  Subscript::ClassificationKind
  classifyPair(const SCEV *Src, const Loop *SrcLoopNest, const SCEV *Dst,
               const Loop *DstLoopNest, SmallBitVector &Loops);

  /// testZIV - Tests the ZIV subscript pair (Src and Dst) for dependence.
  /// Returns true if any possible dependence is disproved.
  /// If there might be a dependence, returns false.
  /// If the dependence isn't proven to exist,
  /// marks the Result as inconsistent.
  bool testZIV(const SCEV *Src, const SCEV *Dst, FullDependence &Result) const;

  /// testSIV - Tests the SIV subscript pair (Src and Dst) for dependence.
  /// Things of the form [c1 + a1*i] and [c2 + a2*j], where
  /// i and j are induction variables, c1 and c2 are loop invariant,
  /// and a1 and a2 are constant.
  /// Returns true if any possible dependence is disproved.
  /// If there might be a dependence, returns false.
  /// Sets appropriate direction vector entry and, when possible,
  /// the distance vector entry.
  /// If the dependence isn't proven to exist,
  /// marks the Result as inconsistent.
  bool testSIV(const SCEV *Src, const SCEV *Dst, unsigned &Level,
               FullDependence &Result, bool UnderRuntimeAssumptions);

  /// testRDIV - Tests the RDIV subscript pair (Src and Dst) for dependence.
  /// Things of the form [c1 + a1*i] and [c2 + a2*j]
  /// where i and j are induction variables, c1 and c2 are loop invariant,
  /// and a1 and a2 are constant.
  /// With minor algebra, this test can also be used for things like
  /// [c1 + a1*i + a2*j][c2].
  /// Returns true if any possible dependence is disproved.
  /// If there might be a dependence, returns false.
  /// Marks the Result as inconsistent.
  bool testRDIV(const SCEV *Src, const SCEV *Dst, FullDependence &Result) const;

  /// testMIV - Tests the MIV subscript pair (Src and Dst) for dependence.
  /// Returns true if dependence disproved.
  /// Can sometimes refine direction vectors.
  bool testMIV(const SCEV *Src, const SCEV *Dst, const SmallBitVector &Loops,
               FullDependence &Result) const;

  /// strongSIVtest - Tests the strong SIV subscript pair (Src and Dst)
  /// for dependence.
  /// Things of the form [c1 + a*i] and [c2 + a*i],
  /// where i is an induction variable, c1 and c2 are loop invariant,
  /// and a is a constant
  /// Returns true if any possible dependence is disproved.
  /// If there might be a dependence, returns false.
  /// Sets appropriate direction and distance.
  bool strongSIVtest(const SCEVAddRecExpr *Src, const SCEVAddRecExpr *Dst,
                     unsigned Level, FullDependence &Result,
                     bool UnderRuntimeAssumptions);

  /// weakCrossingSIVtest - Tests the weak-crossing SIV subscript pair
  /// (Src and Dst) for dependence.
  /// Things of the form [c1 + a*i] and [c2 - a*i],
  /// where i is an induction variable, c1 and c2 are loop invariant,
  /// and a is a constant.
  /// Returns true if any possible dependence is disproved.
  /// If there might be a dependence, returns false.
  /// Sets appropriate direction entry.
  /// Set consistent to false.
  bool weakCrossingSIVtest(const SCEV *SrcCoeff, const SCEV *SrcConst,
                           const SCEV *DstConst, const Loop *CurrentSrcLoop,
                           const Loop *CurrentDstLoop, unsigned Level,
                           FullDependence &Result) const;

  /// ExactSIVtest - Tests the SIV subscript pair
  /// (Src and Dst) for dependence.
  /// Things of the form [c1 + a1*i] and [c2 + a2*i],
  /// where i is an induction variable, c1 and c2 are loop invariant,
  /// and a1 and a2 are constant.
  /// Returns true if any possible dependence is disproved.
  /// If there might be a dependence, returns false.
  /// Sets appropriate direction entry.
  /// Set consistent to false.
  bool exactSIVtest(const SCEV *SrcCoeff, const SCEV *DstCoeff,
                    const SCEV *SrcConst, const SCEV *DstConst,
                    const Loop *CurrentSrcLoop, const Loop *CurrentDstLoop,
                    unsigned Level, FullDependence &Result) const;

  /// weakZeroSrcSIVtest - Tests the weak-zero SIV subscript pair
  /// (Src and Dst) for dependence.
  /// Things of the form [c1] and [c2 + a*i],
  /// where i is an induction variable, c1 and c2 are loop invariant,
  /// and a is a constant. See also weakZeroDstSIVtest.
  /// Returns true if any possible dependence is disproved.
  /// If there might be a dependence, returns false.
  /// Sets appropriate direction entry.
  /// Set consistent to false.
  /// If loop peeling will break the dependence, mark appropriately.
  bool weakZeroSrcSIVtest(const SCEV *DstCoeff, const SCEV *SrcConst,
                          const SCEV *DstConst, const Loop *CurrentSrcLoop,
                          const Loop *CurrentDstLoop, unsigned Level,
                          FullDependence &Result) const;

  /// weakZeroDstSIVtest - Tests the weak-zero SIV subscript pair
  /// (Src and Dst) for dependence.
  /// Things of the form [c1 + a*i] and [c2],
  /// where i is an induction variable, c1 and c2 are loop invariant,
  /// and a is a constant. See also weakZeroSrcSIVtest.
  /// Returns true if any possible dependence is disproved.
  /// If there might be a dependence, returns false.
  /// Sets appropriate direction entry.
  /// Set consistent to false.
  /// If loop peeling will break the dependence, mark appropriately.
  bool weakZeroDstSIVtest(const SCEV *SrcCoeff, const SCEV *SrcConst,
                          const SCEV *DstConst, const Loop *CurrentSrcLoop,
                          const Loop *CurrentDstLoop, unsigned Level,
                          FullDependence &Result) const;

  /// exactRDIVtest - Tests the RDIV subscript pair for dependence.
  /// Things of the form [c1 + a*i] and [c2 + b*j],
  /// where i and j are induction variable, c1 and c2 are loop invariant,
  /// and a and b are constants.
  /// Returns true if any possible dependence is disproved.
  /// Marks the result as inconsistent.
  /// Works in some cases that symbolicRDIVtest doesn't,
  /// and vice versa.
  bool exactRDIVtest(const SCEV *SrcCoeff, const SCEV *DstCoeff,
                     const SCEV *SrcConst, const SCEV *DstConst,
                     const Loop *SrcLoop, const Loop *DstLoop,
                     FullDependence &Result) const;

  /// symbolicRDIVtest - Tests the RDIV subscript pair for dependence.
  /// Things of the form [c1 + a*i] and [c2 + b*j],
  /// where i and j are induction variable, c1 and c2 are loop invariant,
  /// and a and b are constants.
  /// Returns true if any possible dependence is disproved.
  /// Marks the result as inconsistent.
  /// Works in some cases that exactRDIVtest doesn't,
  /// and vice versa. Can also be used as a backup for
  /// ordinary SIV tests.
  bool symbolicRDIVtest(const SCEVAddRecExpr *Src, const SCEVAddRecExpr *Dst);

  /// gcdMIVtest - Tests an MIV subscript pair for dependence.
  /// Returns true if any possible dependence is disproved.
  /// Marks the result as inconsistent.
  /// Can sometimes disprove the equal direction for 1 or more loops.
  //  Can handle some symbolics that even the SIV tests don't get,
  /// so we use it as a backup for everything.
  bool gcdMIVtest(const SCEV *Src, const SCEV *Dst,
                  FullDependence &Result) const;

  /// banerjeeMIVtest - Tests an MIV subscript pair for dependence.
  /// Returns true if any possible dependence is disproved.
  /// Marks the result as inconsistent.
  /// Computes directions.
  bool banerjeeMIVtest(const SCEV *Src, const SCEV *Dst,
                       const SmallBitVector &Loops,
                       FullDependence &Result) const;

  /// collectCoeffInfo - Walks through the subscript, collecting each
  /// coefficient, the associated loop bounds, and recording its positive and
  /// negative parts for later use.
  CoefficientInfo *collectCoeffInfo(const SCEV *Subscript, bool SrcFlag,
                                    const SCEV *&Constant) const;

  /// Given \p Expr of the form
  ///
  ///   c_0*X_0*i_0 + c_1*X_1*i_1 + ...c_n*X_n*i_n + C
  ///
  /// compute
  ///
  ///   RunningGCD = gcd(RunningGCD, c_0, c_1, ..., c_n)
  ///
  /// where c_0, c_1, ..., and c_n are the constant values. The result is stored
  /// in \p RunningGCD. Also, the initial value of \p RunningGCD affects the
  /// result. If we find a term like (c_k * X_k * i_k), where i_k is the
  /// induction variable of \p CurLoop, c_k is stored in \p CurLoopCoeff and not
  /// included in the GCD computation. Returns false if we fail to find a
  /// constant coefficient for some loop, e.g., when a term like (X+Y)*i is
  /// present. Otherwise returns true.
  bool accumulateCoefficientsGCD(const SCEV *Expr, const Loop *CurLoop,
                                 const SCEV *&CurLoopCoeff,
                                 APInt &RunningGCD) const;

  /// getPositivePart - X^+ = max(X, 0).
  const SCEV *getPositivePart(const SCEV *X) const;

  /// getNegativePart - X^- = min(X, 0).
  const SCEV *getNegativePart(const SCEV *X) const;

  /// getLowerBound - Looks through all the bounds info and
  /// computes the lower bound given the current direction settings
  /// at each level.
  const SCEV *getLowerBound(BoundInfo *Bound) const;

  /// getUpperBound - Looks through all the bounds info and
  /// computes the upper bound given the current direction settings
  /// at each level.
  const SCEV *getUpperBound(BoundInfo *Bound) const;

  /// exploreDirections - Hierarchically expands the direction vector
  /// search space, combining the directions of discovered dependences
  /// in the DirSet field of Bound. Returns the number of distinct
  /// dependences discovered. If the dependence is disproved,
  /// it will return 0.
  unsigned exploreDirections(unsigned Level, CoefficientInfo *A,
                             CoefficientInfo *B, BoundInfo *Bound,
                             const SmallBitVector &Loops,
                             unsigned &DepthExpanded, const SCEV *Delta) const;

  /// testBounds - Returns true iff the current bounds are plausible.
  bool testBounds(unsigned char DirKind, unsigned Level, BoundInfo *Bound,
                  const SCEV *Delta) const;

  /// findBoundsALL - Computes the upper and lower bounds for level K
  /// using the * direction. Records them in Bound.
  void findBoundsALL(CoefficientInfo *A, CoefficientInfo *B, BoundInfo *Bound,
                     unsigned K) const;

  /// findBoundsLT - Computes the upper and lower bounds for level K
  /// using the < direction. Records them in Bound.
  void findBoundsLT(CoefficientInfo *A, CoefficientInfo *B, BoundInfo *Bound,
                    unsigned K) const;

  /// findBoundsGT - Computes the upper and lower bounds for level K
  /// using the > direction. Records them in Bound.
  void findBoundsGT(CoefficientInfo *A, CoefficientInfo *B, BoundInfo *Bound,
                    unsigned K) const;

  /// findBoundsEQ - Computes the upper and lower bounds for level K
  /// using the = direction. Records them in Bound.
  void findBoundsEQ(CoefficientInfo *A, CoefficientInfo *B, BoundInfo *Bound,
                    unsigned K) const;

  /// Given a linear access function, tries to recover subscripts
  /// for each dimension of the array element access.
  bool tryDelinearize(Instruction *Src, Instruction *Dst,
                      SmallVectorImpl<Subscript> &Pair);

  /// Tries to delinearize \p Src and \p Dst access functions for a fixed size
  /// multi-dimensional array. Calls delinearizeFixedSizeArray() to delinearize
  /// \p Src and \p Dst separately,
  bool tryDelinearizeFixedSize(Instruction *Src, Instruction *Dst,
                               const SCEV *SrcAccessFn, const SCEV *DstAccessFn,
                               SmallVectorImpl<const SCEV *> &SrcSubscripts,
                               SmallVectorImpl<const SCEV *> &DstSubscripts);

  /// Tries to delinearize access function for a multi-dimensional array with
  /// symbolic runtime sizes.
  /// Returns true upon success and false otherwise.
  bool
  tryDelinearizeParametricSize(Instruction *Src, Instruction *Dst,
                               const SCEV *SrcAccessFn, const SCEV *DstAccessFn,
                               SmallVectorImpl<const SCEV *> &SrcSubscripts,
                               SmallVectorImpl<const SCEV *> &DstSubscripts);

  /// checkSubscript - Helper function for checkSrcSubscript and
  /// checkDstSubscript to avoid duplicate code
  bool checkSubscript(const SCEV *Expr, const Loop *LoopNest,
                      SmallBitVector &Loops, bool IsSrc);

  /// Returns true if \p Expr is multivariate monotonic over \p Domain.
  bool isMonotonic(const SCEV *Expr, SCEVMonotonicityDomain Domain,
                   const Loop *OutermostLoop);

  /// Returns true if both \p Src and \p Dst are monotonic over \p Domain.
  bool isMonotonicPair(const SCEV *Src, const SCEV *Dst,
                       SCEVMonotonicityDomain Domain,
                       const Loop *OutermostLoop);
}; // class DependenceInfo

/// AnalysisPass to compute dependence information in a function
class DependenceAnalysis : public AnalysisInfoMixin<DependenceAnalysis> {
public:
  typedef DependenceInfo Result;
  LLVM_ABI Result run(Function &F, FunctionAnalysisManager &FAM);

private:
  LLVM_ABI static AnalysisKey Key;
  friend struct AnalysisInfoMixin<DependenceAnalysis>;
}; // class DependenceAnalysis

/// Printer pass to dump DA results.
struct DependenceAnalysisPrinterPass
    : public PassInfoMixin<DependenceAnalysisPrinterPass> {
  DependenceAnalysisPrinterPass(raw_ostream &OS, bool NormalizeResults = false)
      : OS(OS), NormalizeResults(NormalizeResults) {}

  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);

  static bool isRequired() { return true; }

private:
  raw_ostream &OS;
  bool NormalizeResults;
}; // class DependenceAnalysisPrinterPass

/// Legacy pass manager pass to access dependence information
class LLVM_ABI DependenceAnalysisWrapperPass : public FunctionPass {
public:
  static char ID; // Class identification, replacement for typeinfo
  DependenceAnalysisWrapperPass();

  bool runOnFunction(Function &F) override;
  void releaseMemory() override;
  void getAnalysisUsage(AnalysisUsage &) const override;
  void print(raw_ostream &, const Module * = nullptr) const override;
  DependenceInfo &getDI() const;

private:
  std::unique_ptr<DependenceInfo> info;
}; // class DependenceAnalysisWrapperPass

/// createDependenceAnalysisPass - This creates an instance of the
/// DependenceAnalysis wrapper pass.
LLVM_ABI FunctionPass *createDependenceAnalysisWrapperPass();

} // namespace llvm

#endif
