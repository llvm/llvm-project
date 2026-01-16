//===-- DependenceAnalysis.cpp - DA Implementation --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DependenceAnalysis is an LLVM pass that analyses dependences between memory
// accesses. Currently, it is an (incomplete) implementation of the approach
// described in
//
//            Practical Dependence Testing
//            Goff, Kennedy, Tseng
//            PLDI 1991
//
// There's a single entry point that analyzes the dependence between a pair
// of memory references in a function, returning either NULL, for no dependence,
// or a more-or-less detailed description of the dependence between them.
//
// Since Clang linearizes some array subscripts, the dependence
// analysis is using SCEV->delinearize to recover the representation of multiple
// subscripts, and thus avoid the more expensive and less precise MIV tests. The
// delinearization is controlled by the flag -da-delinearize.
//
// We should pay some careful attention to the possibility of integer overflow
// in the implementation of the various tests. This could happen with Add,
// Subtract, or Multiply, with both APInt's and SCEV's.
//
// Some non-linear subscript pairs can be handled by the GCD test
// (and perhaps other tests).
// Should explore how often these things occur.
//
// Finally, it seems like certain test cases expose weaknesses in the SCEV
// simplification, especially in the handling of sign and zero extensions.
// It could be useful to spend time exploring these.
//
// Please note that this is work in progress and the interface is subject to
// change.
//
//===----------------------------------------------------------------------===//
//                                                                            //
//                   In memory of Ken Kennedy, 1945 - 2007                    //
//                                                                            //
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Delinearization.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "da"

//===----------------------------------------------------------------------===//
// statistics

STATISTIC(TotalArrayPairs, "Array pairs tested");
STATISTIC(NonlinearSubscriptPairs, "Nonlinear subscript pairs");
STATISTIC(ZIVapplications, "ZIV applications");
STATISTIC(ZIVindependence, "ZIV independence");
STATISTIC(StrongSIVapplications, "Strong SIV applications");
STATISTIC(StrongSIVsuccesses, "Strong SIV successes");
STATISTIC(StrongSIVindependence, "Strong SIV independence");
STATISTIC(WeakCrossingSIVapplications, "Weak-Crossing SIV applications");
STATISTIC(WeakCrossingSIVsuccesses, "Weak-Crossing SIV successes");
STATISTIC(WeakCrossingSIVindependence, "Weak-Crossing SIV independence");
STATISTIC(ExactSIVapplications, "Exact SIV applications");
STATISTIC(ExactSIVsuccesses, "Exact SIV successes");
STATISTIC(ExactSIVindependence, "Exact SIV independence");
STATISTIC(WeakZeroSIVapplications, "Weak-Zero SIV applications");
STATISTIC(WeakZeroSIVsuccesses, "Weak-Zero SIV successes");
STATISTIC(WeakZeroSIVindependence, "Weak-Zero SIV independence");
STATISTIC(ExactRDIVapplications, "Exact RDIV applications");
STATISTIC(ExactRDIVindependence, "Exact RDIV independence");
STATISTIC(SymbolicRDIVapplications, "Symbolic RDIV applications");
STATISTIC(SymbolicRDIVindependence, "Symbolic RDIV independence");
STATISTIC(GCDapplications, "GCD applications");
STATISTIC(GCDsuccesses, "GCD successes");
STATISTIC(GCDindependence, "GCD independence");
STATISTIC(BanerjeeApplications, "Banerjee applications");
STATISTIC(BanerjeeIndependence, "Banerjee independence");
STATISTIC(BanerjeeSuccesses, "Banerjee successes");
STATISTIC(SameSDLoopsCount, "Loops with Same iteration Space and Depth");

static cl::opt<bool>
    Delinearize("da-delinearize", cl::init(true), cl::Hidden,
                cl::desc("Try to delinearize array references."));
static cl::opt<bool> DisableDelinearizationChecks(
    "da-disable-delinearization-checks", cl::Hidden,
    cl::desc(
        "Disable checks that try to statically verify validity of "
        "delinearized subscripts. Enabling this option may result in incorrect "
        "dependence vectors for languages that allow the subscript of one "
        "dimension to underflow or overflow into another dimension."));

static cl::opt<unsigned> MIVMaxLevelThreshold(
    "da-miv-max-level-threshold", cl::init(7), cl::Hidden,
    cl::desc("Maximum depth allowed for the recursive algorithm used to "
             "explore MIV direction vectors."));

namespace {

/// Types of dependence test routines.
enum class DependenceTestType {
  All,
  StrongSIV,
  WeakCrossingSIV,
  ExactSIV,
  WeakZeroSIV,
  ExactRDIV,
  SymbolicRDIV,
  GCDMIV,
  BanerjeeMIV,
};

} // anonymous namespace

static cl::opt<DependenceTestType> EnableDependenceTest(
    "da-enable-dependence-test", cl::init(DependenceTestType::All),
    cl::ReallyHidden,
    cl::desc("Run only specified dependence test routine and disable others. "
             "The purpose is mainly to exclude the influence of other "
             "dependence test routines in regression tests. If set to All, all "
             "dependence test routines are enabled."),
    cl::values(clEnumValN(DependenceTestType::All, "all",
                          "Enable all dependence test routines."),
               clEnumValN(DependenceTestType::StrongSIV, "strong-siv",
                          "Enable only Strong SIV test."),
               clEnumValN(DependenceTestType::WeakCrossingSIV,
                          "weak-crossing-siv",
                          "Enable only Weak-Crossing SIV test."),
               clEnumValN(DependenceTestType::ExactSIV, "exact-siv",
                          "Enable only Exact SIV test."),
               clEnumValN(DependenceTestType::WeakZeroSIV, "weak-zero-siv",
                          "Enable only Weak-Zero SIV test."),
               clEnumValN(DependenceTestType::ExactRDIV, "exact-rdiv",
                          "Enable only Exact RDIV test."),
               clEnumValN(DependenceTestType::SymbolicRDIV, "symbolic-rdiv",
                          "Enable only Symbolic RDIV test."),
               clEnumValN(DependenceTestType::GCDMIV, "gcd-miv",
                          "Enable only GCD MIV test."),
               clEnumValN(DependenceTestType::BanerjeeMIV, "banerjee-miv",
                          "Enable only Banerjee MIV test.")));

// TODO: This flag is disabled by default because it is still under development.
// Enable it or delete this flag when the feature is ready.
static cl::opt<bool> EnableMonotonicityCheck(
    "da-enable-monotonicity-check", cl::init(false), cl::Hidden,
    cl::desc("Check if the subscripts are monotonic. If it's not, dependence "
             "is reported as unknown."));

static cl::opt<bool> DumpMonotonicityReport(
    "da-dump-monotonicity-report", cl::init(false), cl::Hidden,
    cl::desc(
        "When printing analysis, dump the results of monotonicity checks."));

//===----------------------------------------------------------------------===//
// basics

DependenceAnalysis::Result
DependenceAnalysis::run(Function &F, FunctionAnalysisManager &FAM) {
  auto &AA = FAM.getResult<AAManager>(F);
  auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  auto &LI = FAM.getResult<LoopAnalysis>(F);
  return DependenceInfo(&F, &AA, &SE, &LI);
}

AnalysisKey DependenceAnalysis::Key;

INITIALIZE_PASS_BEGIN(DependenceAnalysisWrapperPass, "da",
                      "Dependence Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(DependenceAnalysisWrapperPass, "da", "Dependence Analysis",
                    true, true)

char DependenceAnalysisWrapperPass::ID = 0;

DependenceAnalysisWrapperPass::DependenceAnalysisWrapperPass()
    : FunctionPass(ID) {}

FunctionPass *llvm::createDependenceAnalysisWrapperPass() {
  return new DependenceAnalysisWrapperPass();
}

bool DependenceAnalysisWrapperPass::runOnFunction(Function &F) {
  auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  info.reset(new DependenceInfo(&F, &AA, &SE, &LI));
  return false;
}

DependenceInfo &DependenceAnalysisWrapperPass::getDI() const { return *info; }

void DependenceAnalysisWrapperPass::releaseMemory() { info.reset(); }

void DependenceAnalysisWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<AAResultsWrapperPass>();
  AU.addRequiredTransitive<ScalarEvolutionWrapperPass>();
  AU.addRequiredTransitive<LoopInfoWrapperPass>();
}

namespace {

/// The property of monotonicity of a SCEV. To define the monotonicity, assume
/// a SCEV defined within N-nested loops. Let i_k denote the iteration number
/// of the k-th loop. Then we can regard the SCEV as an N-ary function:
///
///   F(i_1, i_2, ..., i_N)
///
/// The domain of i_k is the closed range [0, BTC_k], where BTC_k is the
/// backedge-taken count of the k-th loop.
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
                                     const Loop *OutermostLoop);

private:
  ScalarEvolution *SE;

  /// The outermost loop that DA is analyzing.
  const Loop *OutermostLoop;

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
  SCEVMonotonicity visitPtrToAddrExpr(const SCEVPtrToAddrExpr *Expr) {
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

/// A wrapper class for std::optional<APInt> that provides arithmetic operators
/// with overflow checking in a signed sense. This allows us to omit inserting
/// an overflow check at every arithmetic operation, which simplifies the code
/// if the operations are chained like `a + b + c + ...`.
///
/// If an calculation overflows, the result becomes "invalid" which is
/// internally represented by std::nullopt. If any operand of an arithmetic
/// operation is "invalid", the result will also be "invalid".
struct OverflowSafeSignedAPInt {
  OverflowSafeSignedAPInt() : Value(std::nullopt) {}
  OverflowSafeSignedAPInt(const APInt &V) : Value(V) {}
  OverflowSafeSignedAPInt(const std::optional<APInt> &V) : Value(V) {}

  OverflowSafeSignedAPInt operator+(const OverflowSafeSignedAPInt &RHS) const {
    if (!Value || !RHS.Value)
      return OverflowSafeSignedAPInt();
    bool Overflow;
    APInt Result = Value->sadd_ov(*RHS.Value, Overflow);
    if (Overflow)
      return OverflowSafeSignedAPInt();
    return OverflowSafeSignedAPInt(Result);
  }

  OverflowSafeSignedAPInt operator+(int RHS) const {
    if (!Value)
      return OverflowSafeSignedAPInt();
    return *this + fromInt(RHS);
  }

  OverflowSafeSignedAPInt operator-(const OverflowSafeSignedAPInt &RHS) const {
    if (!Value || !RHS.Value)
      return OverflowSafeSignedAPInt();
    bool Overflow;
    APInt Result = Value->ssub_ov(*RHS.Value, Overflow);
    if (Overflow)
      return OverflowSafeSignedAPInt();
    return OverflowSafeSignedAPInt(Result);
  }

  OverflowSafeSignedAPInt operator-(int RHS) const {
    if (!Value)
      return OverflowSafeSignedAPInt();
    return *this - fromInt(RHS);
  }

  OverflowSafeSignedAPInt operator*(const OverflowSafeSignedAPInt &RHS) const {
    if (!Value || !RHS.Value)
      return OverflowSafeSignedAPInt();
    bool Overflow;
    APInt Result = Value->smul_ov(*RHS.Value, Overflow);
    if (Overflow)
      return OverflowSafeSignedAPInt();
    return OverflowSafeSignedAPInt(Result);
  }

  OverflowSafeSignedAPInt operator-() const {
    if (!Value)
      return OverflowSafeSignedAPInt();
    if (Value->isMinSignedValue())
      return OverflowSafeSignedAPInt();
    return OverflowSafeSignedAPInt(-*Value);
  }

  operator bool() const { return Value.has_value(); }

  bool operator!() const { return !Value.has_value(); }

  const APInt &operator*() const {
    assert(Value && "Value is not available.");
    return *Value;
  }

  const APInt *operator->() const {
    assert(Value && "Value is not available.");
    return &*Value;
  }

private:
  /// Underlying value. std::nullopt means "unknown". An arithmetic operation on
  /// "unknown" always produces "unknown".
  std::optional<APInt> Value;

  OverflowSafeSignedAPInt fromInt(uint64_t V) const {
    assert(Value && "Value is not available.");
    return OverflowSafeSignedAPInt(
        APInt(Value->getBitWidth(), V, /*isSigned=*/true));
  }
};

} // anonymous namespace

// Used to test the dependence analyzer.
// Looks through the function, noting instructions that may access memory.
// Calls depends() on every possible pair and prints out the result.
// Ignores all other instructions.
static void dumpExampleDependence(raw_ostream &OS, DependenceInfo *DA,
                                  ScalarEvolution &SE, LoopInfo &LI,
                                  bool NormalizeResults) {
  auto *F = DA->getFunction();

  if (DumpMonotonicityReport) {
    SCEVMonotonicityChecker Checker(&SE);
    OS << "Monotonicity check:\n";
    for (Instruction &Inst : instructions(F)) {
      if (!isa<LoadInst>(Inst) && !isa<StoreInst>(Inst))
        continue;
      Value *Ptr = getLoadStorePointerOperand(&Inst);
      const Loop *L = LI.getLoopFor(Inst.getParent());
      const Loop *OutermostLoop = L ? L->getOutermostLoop() : nullptr;
      const SCEV *PtrSCEV = SE.getSCEVAtScope(Ptr, L);
      const SCEV *AccessFn = SE.removePointerBase(PtrSCEV);
      SCEVMonotonicity Mon = Checker.checkMonotonicity(AccessFn, OutermostLoop);
      OS.indent(2) << "Inst: " << Inst << "\n";
      OS.indent(4) << "Expr: " << *AccessFn << "\n";
      Mon.print(OS, 4);
    }
    OS << "\n";
  }

  for (inst_iterator SrcI = inst_begin(F), SrcE = inst_end(F); SrcI != SrcE;
       ++SrcI) {
    if (SrcI->mayReadOrWriteMemory()) {
      for (inst_iterator DstI = SrcI, DstE = inst_end(F); DstI != DstE;
           ++DstI) {
        if (DstI->mayReadOrWriteMemory()) {
          OS << "Src:" << *SrcI << " --> Dst:" << *DstI << "\n";
          OS << "  da analyze - ";
          if (auto D = DA->depends(&*SrcI, &*DstI,
                                   /*UnderRuntimeAssumptions=*/true)) {

#ifndef NDEBUG
            // Verify that the distance being zero is equivalent to the
            // direction being EQ.
            for (unsigned Level = 1; Level <= D->getLevels(); Level++) {
              const SCEV *Distance = D->getDistance(Level);
              bool IsDistanceZero = Distance && Distance->isZero();
              bool IsDirectionEQ =
                  D->getDirection(Level) == Dependence::DVEntry::EQ;
              assert(IsDistanceZero == IsDirectionEQ &&
                     "Inconsistent distance and direction.");
            }
#endif

            // Normalize negative direction vectors if required by clients.
            if (NormalizeResults && D->normalize(&SE))
              OS << "normalized - ";
            D->dump(OS);
          } else
            OS << "none!\n";
        }
      }
    }
  }
}

void DependenceAnalysisWrapperPass::print(raw_ostream &OS,
                                          const Module *) const {
  dumpExampleDependence(
      OS, info.get(), getAnalysis<ScalarEvolutionWrapperPass>().getSE(),
      getAnalysis<LoopInfoWrapperPass>().getLoopInfo(), false);
}

PreservedAnalyses
DependenceAnalysisPrinterPass::run(Function &F, FunctionAnalysisManager &FAM) {
  OS << "Printing analysis 'Dependence Analysis' for function '" << F.getName()
     << "':\n";
  dumpExampleDependence(OS, &FAM.getResult<DependenceAnalysis>(F),
                        FAM.getResult<ScalarEvolutionAnalysis>(F),
                        FAM.getResult<LoopAnalysis>(F), NormalizeResults);
  return PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
// Dependence methods

// Returns true if this is an input dependence.
bool Dependence::isInput() const {
  return Src->mayReadFromMemory() && Dst->mayReadFromMemory();
}

// Returns true if this is an output dependence.
bool Dependence::isOutput() const {
  return Src->mayWriteToMemory() && Dst->mayWriteToMemory();
}

// Returns true if this is an flow (aka true)  dependence.
bool Dependence::isFlow() const {
  return Src->mayWriteToMemory() && Dst->mayReadFromMemory();
}

// Returns true if this is an anti dependence.
bool Dependence::isAnti() const {
  return Src->mayReadFromMemory() && Dst->mayWriteToMemory();
}

// Returns true if a particular level is scalar; that is,
// if no subscript in the source or destination mention the induction
// variable associated with the loop at this level.
// Leave this out of line, so it will serve as a virtual method anchor
bool Dependence::isScalar(unsigned level, bool IsSameSD) const { return false; }

//===----------------------------------------------------------------------===//
// FullDependence methods

FullDependence::FullDependence(Instruction *Source, Instruction *Destination,
                               const SCEVUnionPredicate &Assumes,
                               bool PossiblyLoopIndependent,
                               unsigned CommonLevels)
    : Dependence(Source, Destination, Assumes), Levels(CommonLevels),
      LoopIndependent(PossiblyLoopIndependent) {
  Consistent = true;
  SameSDLevels = 0;
  if (CommonLevels)
    DV = std::make_unique<DVEntry[]>(CommonLevels);
}

// FIXME: in some cases the meaning of a negative direction vector
// may not be straightforward, e.g.,
// for (int i = 0; i < 32; ++i) {
//   Src:    A[i] = ...;
//   Dst:    use(A[31 - i]);
// }
// The dependency is
//   flow { Src[i] -> Dst[31 - i] : when i >= 16 } and
//   anti { Dst[i] -> Src[31 - i] : when i < 16 },
// -- hence a [<>].
// As long as a dependence result contains '>' ('<>', '<=>', "*"), it
// means that a reversed/normalized dependence needs to be considered
// as well. Nevertheless, current isDirectionNegative() only returns
// true with a '>' or '>=' dependency for ease of canonicalizing the
// dependency vector, since the reverse of '<>', '<=>' and "*" is itself.
bool FullDependence::isDirectionNegative() const {
  for (unsigned Level = 1; Level <= Levels; ++Level) {
    unsigned char Direction = DV[Level - 1].Direction;
    if (Direction == Dependence::DVEntry::EQ)
      continue;
    if (Direction == Dependence::DVEntry::GT ||
        Direction == Dependence::DVEntry::GE)
      return true;
    return false;
  }
  return false;
}

bool FullDependence::normalize(ScalarEvolution *SE) {
  if (!isDirectionNegative())
    return false;

  LLVM_DEBUG(dbgs() << "Before normalizing negative direction vectors:\n";
             dump(dbgs()););
  std::swap(Src, Dst);
  for (unsigned Level = 1; Level <= Levels; ++Level) {
    unsigned char Direction = DV[Level - 1].Direction;
    // Reverse the direction vector, this means LT becomes GT
    // and GT becomes LT.
    unsigned char RevDirection = Direction & Dependence::DVEntry::EQ;
    if (Direction & Dependence::DVEntry::LT)
      RevDirection |= Dependence::DVEntry::GT;
    if (Direction & Dependence::DVEntry::GT)
      RevDirection |= Dependence::DVEntry::LT;
    DV[Level - 1].Direction = RevDirection;
    // Reverse the dependence distance as well.
    if (DV[Level - 1].Distance != nullptr)
      DV[Level - 1].Distance = SE->getNegativeSCEV(DV[Level - 1].Distance);
  }

  LLVM_DEBUG(dbgs() << "After normalizing negative direction vectors:\n";
             dump(dbgs()););
  return true;
}

// The rest are simple getters that hide the implementation.

// getDirection - Returns the direction associated with a particular common or
// SameSD level.
unsigned FullDependence::getDirection(unsigned Level, bool IsSameSD) const {
  return getDVEntry(Level, IsSameSD).Direction;
}

// Returns the distance (or NULL) associated with a particular common or
// SameSD level.
const SCEV *FullDependence::getDistance(unsigned Level, bool IsSameSD) const {
  return getDVEntry(Level, IsSameSD).Distance;
}

// Returns true if a particular regular or SameSD level is scalar; that is,
// if no subscript in the source or destination mention the induction variable
// associated with the loop at this level.
bool FullDependence::isScalar(unsigned Level, bool IsSameSD) const {
  return getDVEntry(Level, IsSameSD).Scalar;
}

// Returns true if peeling the first iteration from this regular or SameSD
// loop level will break this dependence.
bool FullDependence::isPeelFirst(unsigned Level, bool IsSameSD) const {
  return getDVEntry(Level, IsSameSD).PeelFirst;
}

// Returns true if peeling the last iteration from this regular or SameSD
// loop level will break this dependence.
bool FullDependence::isPeelLast(unsigned Level, bool IsSameSD) const {
  return getDVEntry(Level, IsSameSD).PeelLast;
}

// inSameSDLoops - Returns true if this level is an SameSD level, i.e.,
// performed across two separate loop nests that have the Same iteration space
// and Depth.
bool FullDependence::inSameSDLoops(unsigned Level) const {
  assert(0 < Level && Level <= static_cast<unsigned>(Levels) + SameSDLevels &&
         "Level out of range");
  return Level > Levels;
}

//===----------------------------------------------------------------------===//
// SCEVMonotonicity

SCEVMonotonicity::SCEVMonotonicity(SCEVMonotonicityType Type,
                                   const SCEV *FailurePoint)
    : Type(Type), FailurePoint(FailurePoint) {
  assert(
      ((Type == SCEVMonotonicityType::Unknown) == (FailurePoint != nullptr)) &&
      "FailurePoint must be provided iff Type is Unknown");
}

void SCEVMonotonicity::print(raw_ostream &OS, unsigned Depth) const {
  OS.indent(Depth) << "Monotonicity: ";
  switch (Type) {
  case SCEVMonotonicityType::Unknown:
    assert(FailurePoint && "FailurePoint must be provided for Unknown");
    OS << "Unknown\n";
    OS.indent(Depth) << "Reason: " << *FailurePoint << "\n";
    break;
  case SCEVMonotonicityType::Invariant:
    OS << "Invariant\n";
    break;
  case SCEVMonotonicityType::MultivariateSignedMonotonic:
    OS << "MultivariateSignedMonotonic\n";
    break;
  }
}

bool SCEVMonotonicityChecker::isLoopInvariant(const SCEV *Expr) const {
  return !OutermostLoop || SE->isLoopInvariant(Expr, OutermostLoop);
}

SCEVMonotonicity SCEVMonotonicityChecker::invariantOrUnknown(const SCEV *Expr) {
  if (isLoopInvariant(Expr))
    return SCEVMonotonicity(SCEVMonotonicityType::Invariant);
  return createUnknown(Expr);
}

SCEVMonotonicity
SCEVMonotonicityChecker::checkMonotonicity(const SCEV *Expr,
                                           const Loop *OutermostLoop) {
  assert((!OutermostLoop || OutermostLoop->isOutermost()) &&
         "OutermostLoop must be outermost");
  assert(Expr->getType()->isIntegerTy() && "Expr must be integer type");
  this->OutermostLoop = OutermostLoop;
  return visit(Expr);
}

/// We only care about an affine AddRec at the moment. For an affine AddRec,
/// the monotonicity can be inferred from its nowrap property. For example, let
/// X and Y be loop-invariant, and assume Y is non-negative. An AddRec
/// {X,+.Y}<nsw> implies:
///
///   X <=s (X + Y) <=s ((X + Y) + Y) <=s ...
///
/// Thus, we can conclude that the AddRec is monotonically increasing with
/// respect to the associated loop in a signed sense. The similar reasoning
/// applies when Y is non-positive, leading to a monotonically decreasing
/// AddRec.
SCEVMonotonicity
SCEVMonotonicityChecker::visitAddRecExpr(const SCEVAddRecExpr *Expr) {
  if (!Expr->isAffine() || !Expr->hasNoSignedWrap())
    return createUnknown(Expr);

  const SCEV *Start = Expr->getStart();
  const SCEV *Step = Expr->getStepRecurrence(*SE);

  SCEVMonotonicity StartMon = visit(Start);
  if (StartMon.isUnknown())
    return StartMon;

  if (!isLoopInvariant(Step))
    return createUnknown(Expr);

  return SCEVMonotonicity(SCEVMonotonicityType::MultivariateSignedMonotonic);
}

//===----------------------------------------------------------------------===//
// DependenceInfo methods

// For debugging purposes. Dumps a dependence to OS.
void Dependence::dump(raw_ostream &OS) const {
  if (isConfused())
    OS << "confused";
  else {
    if (isConsistent())
      OS << "consistent ";
    if (isFlow())
      OS << "flow";
    else if (isOutput())
      OS << "output";
    else if (isAnti())
      OS << "anti";
    else if (isInput())
      OS << "input";
    dumpImp(OS);
    unsigned SameSDLevels = getSameSDLevels();
    if (SameSDLevels > 0) {
      OS << " / assuming " << SameSDLevels << " loop level(s) fused: ";
      dumpImp(OS, true);
    }
  }
  OS << "!\n";

  SCEVUnionPredicate Assumptions = getRuntimeAssumptions();
  if (!Assumptions.isAlwaysTrue()) {
    OS << "  Runtime Assumptions:\n";
    Assumptions.print(OS, 2);
  }
}

// For debugging purposes. Dumps a dependence to OS with or without considering
// the SameSD levels.
void Dependence::dumpImp(raw_ostream &OS, bool IsSameSD) const {
  unsigned Levels = getLevels();
  unsigned SameSDLevels = getSameSDLevels();
  bool OnSameSD = false;
  unsigned LevelNum = Levels;
  if (IsSameSD)
    LevelNum += SameSDLevels;
  OS << " [";
  for (unsigned II = 1; II <= LevelNum; ++II) {
    if (!OnSameSD && inSameSDLoops(II))
      OnSameSD = true;
    if (isPeelFirst(II, OnSameSD))
      OS << 'p';
    const SCEV *Distance = getDistance(II, OnSameSD);
    if (Distance)
      OS << *Distance;
    else if (isScalar(II, OnSameSD))
      OS << "S";
    else {
      unsigned Direction = getDirection(II, OnSameSD);
      if (Direction == DVEntry::ALL)
        OS << "*";
      else {
        if (Direction & DVEntry::LT)
          OS << "<";
        if (Direction & DVEntry::EQ)
          OS << "=";
        if (Direction & DVEntry::GT)
          OS << ">";
      }
    }
    if (isPeelLast(II, OnSameSD))
      OS << 'p';
    if (II < LevelNum)
      OS << " ";
  }
  if (isLoopIndependent())
    OS << "|<";
  OS << "]";
}

// Returns NoAlias/MayAliass/MustAlias for two memory locations based upon their
// underlaying objects. If LocA and LocB are known to not alias (for any reason:
// tbaa, non-overlapping regions etc), then it is known there is no dependecy.
// Otherwise the underlying objects are checked to see if they point to
// different identifiable objects.
static AliasResult underlyingObjectsAlias(AAResults *AA, const DataLayout &DL,
                                          const MemoryLocation &LocA,
                                          const MemoryLocation &LocB) {
  // Check the original locations (minus size) for noalias, which can happen for
  // tbaa, incompatible underlying object locations, etc.
  MemoryLocation LocAS =
      MemoryLocation::getBeforeOrAfter(LocA.Ptr, LocA.AATags);
  MemoryLocation LocBS =
      MemoryLocation::getBeforeOrAfter(LocB.Ptr, LocB.AATags);
  BatchAAResults BAA(*AA);
  BAA.enableCrossIterationMode();

  if (BAA.isNoAlias(LocAS, LocBS))
    return AliasResult::NoAlias;

  // Check the underlying objects are the same
  const Value *AObj = getUnderlyingObject(LocA.Ptr);
  const Value *BObj = getUnderlyingObject(LocB.Ptr);

  // If the underlying objects are the same, they must alias
  if (AObj == BObj)
    return AliasResult::MustAlias;

  // We may have hit the recursion limit for underlying objects, or have
  // underlying objects where we don't know they will alias.
  if (!isIdentifiedObject(AObj) || !isIdentifiedObject(BObj))
    return AliasResult::MayAlias;

  // Otherwise we know the objects are different and both identified objects so
  // must not alias.
  return AliasResult::NoAlias;
}

// Returns true if the load or store can be analyzed. Atomic and volatile
// operations have properties which this analysis does not understand.
static bool isLoadOrStore(const Instruction *I) {
  if (const LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->isUnordered();
  else if (const StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->isUnordered();
  return false;
}

// Returns true if two loops have the Same iteration Space and Depth. To be
// more specific, two loops have SameSD if they are in the same nesting
// depth and have the same backedge count. SameSD stands for Same iteration
// Space and Depth.
bool DependenceInfo::haveSameSD(const Loop *SrcLoop,
                                const Loop *DstLoop) const {
  if (SrcLoop == DstLoop)
    return true;

  if (SrcLoop->getLoopDepth() != DstLoop->getLoopDepth())
    return false;

  if (!SrcLoop || !SrcLoop->getLoopLatch() || !DstLoop ||
      !DstLoop->getLoopLatch())
    return false;

  const SCEV *SrcUB = nullptr, *DstUP = nullptr;
  if (SE->hasLoopInvariantBackedgeTakenCount(SrcLoop))
    SrcUB = SE->getBackedgeTakenCount(SrcLoop);
  if (SE->hasLoopInvariantBackedgeTakenCount(DstLoop))
    DstUP = SE->getBackedgeTakenCount(DstLoop);

  if (SrcUB != nullptr && DstUP != nullptr) {
    Type *WiderType = SE->getWiderType(SrcUB->getType(), DstUP->getType());
    SrcUB = SE->getNoopOrZeroExtend(SrcUB, WiderType);
    DstUP = SE->getNoopOrZeroExtend(DstUP, WiderType);

    if (SE->isKnownPredicate(ICmpInst::ICMP_EQ, SrcUB, DstUP))
      return true;
  }

  return false;
}

// Examines the loop nesting of the Src and Dst
// instructions and establishes their shared loops. Sets the variables
// CommonLevels, SrcLevels, and MaxLevels.
// The source and destination instructions needn't be contained in the same
// loop. The routine establishNestingLevels finds the level of most deeply
// nested loop that contains them both, CommonLevels. An instruction that's
// not contained in a loop is at level = 0. MaxLevels is equal to the level
// of the source plus the level of the destination, minus CommonLevels.
// This lets us allocate vectors MaxLevels in length, with room for every
// distinct loop referenced in both the source and destination subscripts.
// The variable SrcLevels is the nesting depth of the source instruction.
// It's used to help calculate distinct loops referenced by the destination.
// Here's the map from loops to levels:
//            0 - unused
//            1 - outermost common loop
//          ... - other common loops
// CommonLevels - innermost common loop
//          ... - loops containing Src but not Dst
//    SrcLevels - innermost loop containing Src but not Dst
//          ... - loops containing Dst but not Src
//    MaxLevels - innermost loops containing Dst but not Src
// Consider the follow code fragment:
//   for (a = ...) {
//     for (b = ...) {
//       for (c = ...) {
//         for (d = ...) {
//           A[] = ...;
//         }
//       }
//       for (e = ...) {
//         for (f = ...) {
//           for (g = ...) {
//             ... = A[];
//           }
//         }
//       }
//     }
//   }
// If we're looking at the possibility of a dependence between the store
// to A (the Src) and the load from A (the Dst), we'll note that they
// have 2 loops in common, so CommonLevels will equal 2 and the direction
// vector for Result will have 2 entries. SrcLevels = 4 and MaxLevels = 7.
// A map from loop names to loop numbers would look like
//     a - 1
//     b - 2 = CommonLevels
//     c - 3
//     d - 4 = SrcLevels
//     e - 5
//     f - 6
//     g - 7 = MaxLevels
// SameSDLevels counts the number of levels after common levels that are
// not common but have the same iteration space and depth. Internally this
// is checked using haveSameSD. Assume that in this code fragment, levels c and
// e have the same iteration space and depth, but levels d and f does not. Then
// SameSDLevels is set to 1. In that case the level numbers for the previous
// code look like
//     a   - 1
//     b   - 2
//     c,e - 3 = CommonLevels
//     d   - 4 = SrcLevels
//     f   - 5
//     g   - 6 = MaxLevels
void DependenceInfo::establishNestingLevels(const Instruction *Src,
                                            const Instruction *Dst) {
  const BasicBlock *SrcBlock = Src->getParent();
  const BasicBlock *DstBlock = Dst->getParent();
  unsigned SrcLevel = LI->getLoopDepth(SrcBlock);
  unsigned DstLevel = LI->getLoopDepth(DstBlock);
  const Loop *SrcLoop = LI->getLoopFor(SrcBlock);
  const Loop *DstLoop = LI->getLoopFor(DstBlock);
  SrcLevels = SrcLevel;
  MaxLevels = SrcLevel + DstLevel;
  SameSDLevels = 0;
  while (SrcLevel > DstLevel) {
    SrcLoop = SrcLoop->getParentLoop();
    SrcLevel--;
  }
  while (DstLevel > SrcLevel) {
    DstLoop = DstLoop->getParentLoop();
    DstLevel--;
  }

  // find the first common level and count the SameSD levels leading to it
  while (SrcLoop != DstLoop) {
    SameSDLevels++;
    if (!haveSameSD(SrcLoop, DstLoop))
      SameSDLevels = 0;
    SrcLoop = SrcLoop->getParentLoop();
    DstLoop = DstLoop->getParentLoop();
    SrcLevel--;
  }
  CommonLevels = SrcLevel;
  MaxLevels -= CommonLevels;
}

// Given one of the loops containing the source, return
// its level index in our numbering scheme.
unsigned DependenceInfo::mapSrcLoop(const Loop *SrcLoop) const {
  return SrcLoop->getLoopDepth();
}

// Given one of the loops containing the destination,
// return its level index in our numbering scheme.
unsigned DependenceInfo::mapDstLoop(const Loop *DstLoop) const {
  unsigned D = DstLoop->getLoopDepth();
  if (D > CommonLevels)
    // This tries to make sure that we assign unique numbers to src and dst when
    // the memory accesses reside in different loops that have the same depth.
    return D - CommonLevels + SrcLevels;
  else
    return D;
}

// Returns true if Expression is loop invariant in LoopNest.
bool DependenceInfo::isLoopInvariant(const SCEV *Expression,
                                     const Loop *LoopNest) const {
  // Unlike ScalarEvolution::isLoopInvariant() we consider an access outside of
  // any loop as invariant, because we only consier expression evaluation at a
  // specific position (where the array access takes place), and not across the
  // entire function.
  if (!LoopNest)
    return true;

  // If the expression is invariant in the outermost loop of the loop nest, it
  // is invariant anywhere in the loop nest.
  return SE->isLoopInvariant(Expression, LoopNest->getOutermostLoop());
}

// Finds the set of loops from the LoopNest that
// have a level <= CommonLevels and are referred to by the SCEV Expression.
void DependenceInfo::collectCommonLoops(const SCEV *Expression,
                                        const Loop *LoopNest,
                                        SmallBitVector &Loops) const {
  while (LoopNest) {
    unsigned Level = LoopNest->getLoopDepth();
    if (Level <= CommonLevels && !SE->isLoopInvariant(Expression, LoopNest))
      Loops.set(Level);
    LoopNest = LoopNest->getParentLoop();
  }
}

void DependenceInfo::unifySubscriptType(ArrayRef<Subscript *> Pairs) {

  unsigned widestWidthSeen = 0;
  Type *widestType;

  // Go through each pair and find the widest bit to which we need
  // to extend all of them.
  for (Subscript *Pair : Pairs) {
    const SCEV *Src = Pair->Src;
    const SCEV *Dst = Pair->Dst;
    IntegerType *SrcTy = dyn_cast<IntegerType>(Src->getType());
    IntegerType *DstTy = dyn_cast<IntegerType>(Dst->getType());
    if (SrcTy == nullptr || DstTy == nullptr) {
      assert(SrcTy == DstTy &&
             "This function only unify integer types and "
             "expect Src and Dst share the same type otherwise.");
      continue;
    }
    if (SrcTy->getBitWidth() > widestWidthSeen) {
      widestWidthSeen = SrcTy->getBitWidth();
      widestType = SrcTy;
    }
    if (DstTy->getBitWidth() > widestWidthSeen) {
      widestWidthSeen = DstTy->getBitWidth();
      widestType = DstTy;
    }
  }

  assert(widestWidthSeen > 0);

  // Now extend each pair to the widest seen.
  for (Subscript *Pair : Pairs) {
    const SCEV *Src = Pair->Src;
    const SCEV *Dst = Pair->Dst;
    IntegerType *SrcTy = dyn_cast<IntegerType>(Src->getType());
    IntegerType *DstTy = dyn_cast<IntegerType>(Dst->getType());
    if (SrcTy == nullptr || DstTy == nullptr) {
      assert(SrcTy == DstTy &&
             "This function only unify integer types and "
             "expect Src and Dst share the same type otherwise.");
      continue;
    }
    if (SrcTy->getBitWidth() < widestWidthSeen)
      // Sign-extend Src to widestType
      Pair->Src = SE->getSignExtendExpr(Src, widestType);
    if (DstTy->getBitWidth() < widestWidthSeen) {
      // Sign-extend Dst to widestType
      Pair->Dst = SE->getSignExtendExpr(Dst, widestType);
    }
  }
}

// removeMatchingExtensions - Examines a subscript pair.
// If the source and destination are identically sign (or zero)
// extended, it strips off the extension in an effect to simplify
// the actual analysis.
void DependenceInfo::removeMatchingExtensions(Subscript *Pair) {
  const SCEV *Src = Pair->Src;
  const SCEV *Dst = Pair->Dst;
  if ((isa<SCEVZeroExtendExpr>(Src) && isa<SCEVZeroExtendExpr>(Dst)) ||
      (isa<SCEVSignExtendExpr>(Src) && isa<SCEVSignExtendExpr>(Dst))) {
    const SCEVIntegralCastExpr *SrcCast = cast<SCEVIntegralCastExpr>(Src);
    const SCEVIntegralCastExpr *DstCast = cast<SCEVIntegralCastExpr>(Dst);
    const SCEV *SrcCastOp = SrcCast->getOperand();
    const SCEV *DstCastOp = DstCast->getOperand();
    if (SrcCastOp->getType() == DstCastOp->getType()) {
      Pair->Src = SrcCastOp;
      Pair->Dst = DstCastOp;
    }
  }
}

// Examine the scev and return true iff it's affine.
// Collect any loops mentioned in the set of "Loops".
bool DependenceInfo::checkSubscript(const SCEV *Expr, const Loop *LoopNest,
                                    SmallBitVector &Loops, bool IsSrc) {
  const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(Expr);
  if (!AddRec)
    return isLoopInvariant(Expr, LoopNest);

  // The AddRec must depend on one of the containing loops. Otherwise,
  // mapSrcLoop and mapDstLoop return indices outside the intended range. This
  // can happen when a subscript in one loop references an IV from a sibling
  // loop that could not be replaced with a concrete exit value by
  // getSCEVAtScope.
  const Loop *L = LoopNest;
  while (L && AddRec->getLoop() != L)
    L = L->getParentLoop();
  if (!L)
    return false;

  const SCEV *Start = AddRec->getStart();
  const SCEV *Step = AddRec->getStepRecurrence(*SE);
  if (!isLoopInvariant(Step, LoopNest))
    return false;
  if (IsSrc)
    Loops.set(mapSrcLoop(AddRec->getLoop()));
  else
    Loops.set(mapDstLoop(AddRec->getLoop()));
  return checkSubscript(Start, LoopNest, Loops, IsSrc);
}

// Examine the scev and return true iff it's linear.
// Collect any loops mentioned in the set of "Loops".
bool DependenceInfo::checkSrcSubscript(const SCEV *Src, const Loop *LoopNest,
                                       SmallBitVector &Loops) {
  return checkSubscript(Src, LoopNest, Loops, true);
}

// Examine the scev and return true iff it's linear.
// Collect any loops mentioned in the set of "Loops".
bool DependenceInfo::checkDstSubscript(const SCEV *Dst, const Loop *LoopNest,
                                       SmallBitVector &Loops) {
  return checkSubscript(Dst, LoopNest, Loops, false);
}

// Examines the subscript pair (the Src and Dst SCEVs)
// and classifies it as either ZIV, SIV, RDIV, MIV, or Nonlinear.
// Collects the associated loops in a set.
DependenceInfo::Subscript::ClassificationKind
DependenceInfo::classifyPair(const SCEV *Src, const Loop *SrcLoopNest,
                             const SCEV *Dst, const Loop *DstLoopNest,
                             SmallBitVector &Loops) {
  SmallBitVector SrcLoops(MaxLevels + 1);
  SmallBitVector DstLoops(MaxLevels + 1);
  if (!checkSrcSubscript(Src, SrcLoopNest, SrcLoops))
    return Subscript::NonLinear;
  if (!checkDstSubscript(Dst, DstLoopNest, DstLoops))
    return Subscript::NonLinear;
  Loops = SrcLoops;
  Loops |= DstLoops;
  unsigned N = Loops.count();
  if (N == 0)
    return Subscript::ZIV;
  if (N == 1)
    return Subscript::SIV;
  if (N == 2 && (SrcLoops.count() == 0 || DstLoops.count() == 0 ||
                 (SrcLoops.count() == 1 && DstLoops.count() == 1)))
    return Subscript::RDIV;
  return Subscript::MIV;
}

// All subscripts are all the same type.
// Loop bound may be smaller (e.g., a char).
// Should zero extend loop bound, since it's always >= 0.
// This routine collects upper bound and extends or truncates if needed.
// Truncating is safe when subscripts are known not to wrap. Cases without
// nowrap flags should have been rejected earlier.
// Return null if no bound available.
const SCEV *DependenceInfo::collectUpperBound(const Loop *L, Type *T) const {
  if (SE->hasLoopInvariantBackedgeTakenCount(L)) {
    const SCEV *UB = SE->getBackedgeTakenCount(L);
    return SE->getTruncateOrZeroExtend(UB, T);
  }
  return nullptr;
}

// Calls collectUpperBound(), then attempts to cast it to SCEVConstant.
// If the cast fails, returns NULL.
const SCEVConstant *DependenceInfo::collectConstantUpperBound(const Loop *L,
                                                              Type *T) const {
  if (const SCEV *UB = collectUpperBound(L, T))
    return dyn_cast<SCEVConstant>(UB);
  return nullptr;
}

/// Returns \p A - \p B if it guaranteed not to signed wrap. Otherwise returns
/// nullptr. \p A and \p B must have the same integer type.
static const SCEV *minusSCEVNoSignedOverflow(const SCEV *A, const SCEV *B,
                                             ScalarEvolution &SE) {
  if (SE.willNotOverflow(Instruction::Sub, /*Signed=*/true, A, B))
    return SE.getMinusSCEV(A, B);
  return nullptr;
}

/// Returns \p A * \p B if it guaranteed not to signed wrap. Otherwise returns
/// nullptr. \p A and \p B must have the same integer type.
static const SCEV *mulSCEVNoSignedOverflow(const SCEV *A, const SCEV *B,
                                           ScalarEvolution &SE) {
  if (SE.willNotOverflow(Instruction::Mul, /*Signed=*/true, A, B))
    return SE.getMulExpr(A, B);
  return nullptr;
}

/// Returns the absolute value of \p A. In the context of dependence analysis,
/// we need an absolute value in a mathematical sense. If \p A is the signed
/// minimum value, we cannot represent it unless extending the original type.
/// Thus if we cannot prove that \p A is not the signed minimum value, returns
/// nullptr.
static const SCEV *absSCEVNoSignedOverflow(const SCEV *A, ScalarEvolution &SE) {
  IntegerType *Ty = cast<IntegerType>(A->getType());
  if (!Ty)
    return nullptr;

  const SCEV *SMin =
      SE.getConstant(APInt::getSignedMinValue(Ty->getBitWidth()));
  if (!SE.isKnownPredicate(CmpInst::ICMP_NE, A, SMin))
    return nullptr;
  return SE.getAbsExpr(A, /*IsNSW=*/true);
}

/// Returns true iff \p Test is enabled.
static bool isDependenceTestEnabled(DependenceTestType Test) {
  if (EnableDependenceTest == DependenceTestType::All)
    return true;
  return EnableDependenceTest == Test;
}

// testZIV -
// When we have a pair of subscripts of the form [c1] and [c2],
// where c1 and c2 are both loop invariant, we attack it using
// the ZIV test. Basically, we test by comparing the two values,
// but there are actually three possible results:
// 1) the values are equal, so there's a dependence
// 2) the values are different, so there's no dependence
// 3) the values might be equal, so we have to assume a dependence.
//
// Return true if dependence disproved.
bool DependenceInfo::testZIV(const SCEV *Src, const SCEV *Dst,
                             FullDependence &Result) const {
  LLVM_DEBUG(dbgs() << "    src = " << *Src << "\n");
  LLVM_DEBUG(dbgs() << "    dst = " << *Dst << "\n");
  ++ZIVapplications;
  if (SE->isKnownPredicate(CmpInst::ICMP_EQ, Src, Dst)) {
    LLVM_DEBUG(dbgs() << "    provably dependent\n");
    return false; // provably dependent
  }
  if (SE->isKnownPredicate(CmpInst::ICMP_NE, Src, Dst)) {
    LLVM_DEBUG(dbgs() << "    provably independent\n");
    ++ZIVindependence;
    return true; // provably independent
  }
  LLVM_DEBUG(dbgs() << "    possibly dependent\n");
  Result.Consistent = false;
  return false; // possibly dependent
}

// strongSIVtest -
// From the paper, Practical Dependence Testing, Section 4.2.1
//
// When we have a pair of subscripts of the form [c1 + a*i] and [c2 + a*i],
// where i is an induction variable, c1 and c2 are loop invariant,
//  and a is a constant, we can solve it exactly using the Strong SIV test.
//
// Can prove independence. Failing that, can compute distance (and direction).
// In the presence of symbolic terms, we can sometimes make progress.
//
// If there's a dependence,
//
//    c1 + a*i = c2 + a*i'
//
// The dependence distance is
//
//    d = i' - i = (c1 - c2)/a
//
// A dependence only exists if d is an integer and abs(d) <= U, where U is the
// loop's upper bound. If a dependence exists, the dependence direction is
// defined as
//
//                { < if d > 0
//    direction = { = if d = 0
//                { > if d < 0
//
// Return true if dependence disproved.
bool DependenceInfo::strongSIVtest(const SCEV *Coeff, const SCEV *SrcConst,
                                   const SCEV *DstConst, const Loop *CurSrcLoop,
                                   const Loop *CurDstLoop, unsigned Level,
                                   FullDependence &Result,
                                   bool UnderRuntimeAssumptions) {
  if (!isDependenceTestEnabled(DependenceTestType::StrongSIV))
    return false;

  LLVM_DEBUG(dbgs() << "\tStrong SIV test\n");
  LLVM_DEBUG(dbgs() << "\t    Coeff = " << *Coeff);
  LLVM_DEBUG(dbgs() << ", " << *Coeff->getType() << "\n");
  LLVM_DEBUG(dbgs() << "\t    SrcConst = " << *SrcConst);
  LLVM_DEBUG(dbgs() << ", " << *SrcConst->getType() << "\n");
  LLVM_DEBUG(dbgs() << "\t    DstConst = " << *DstConst);
  LLVM_DEBUG(dbgs() << ", " << *DstConst->getType() << "\n");
  ++StrongSIVapplications;
  assert(0 < Level && Level <= CommonLevels && "level out of range");
  Level--;

  const SCEV *Delta = minusSCEVNoSignedOverflow(SrcConst, DstConst, *SE);
  if (!Delta) {
    Result.Consistent = false;
    return false;
  }
  LLVM_DEBUG(dbgs() << "\t    Delta = " << *Delta);
  LLVM_DEBUG(dbgs() << ", " << *Delta->getType() << "\n");

  // check that |Delta| < iteration count
  bool IsDeltaLarge = [&] {
    const SCEV *UpperBound = collectUpperBound(CurSrcLoop, Delta->getType());
    if (!UpperBound)
      return false;

    LLVM_DEBUG(dbgs() << "\t    UpperBound = " << *UpperBound);
    LLVM_DEBUG(dbgs() << ", " << *UpperBound->getType() << "\n");
    const SCEV *AbsDelta = absSCEVNoSignedOverflow(Delta, *SE);
    const SCEV *AbsCoeff = absSCEVNoSignedOverflow(Coeff, *SE);
    if (!AbsDelta || !AbsCoeff)
      return false;
    const SCEV *Product = mulSCEVNoSignedOverflow(UpperBound, AbsCoeff, *SE);
    if (!Product)
      return false;
    return SE->isKnownPredicate(CmpInst::ICMP_SGT, AbsDelta, Product);
  }();
  if (IsDeltaLarge) {
    // Distance greater than trip count - no dependence
    ++StrongSIVindependence;
    ++StrongSIVsuccesses;
    return true;
  }

  // Can we compute distance?
  if (isa<SCEVConstant>(Delta) && isa<SCEVConstant>(Coeff)) {
    APInt ConstDelta = cast<SCEVConstant>(Delta)->getAPInt();
    APInt ConstCoeff = cast<SCEVConstant>(Coeff)->getAPInt();
    APInt Distance = ConstDelta; // these need to be initialized
    APInt Remainder = ConstDelta;
    APInt::sdivrem(ConstDelta, ConstCoeff, Distance, Remainder);
    LLVM_DEBUG(dbgs() << "\t    Distance = " << Distance << "\n");
    LLVM_DEBUG(dbgs() << "\t    Remainder = " << Remainder << "\n");
    // Make sure Coeff divides Delta exactly
    if (Remainder != 0) {
      // Coeff doesn't divide Distance, no dependence
      ++StrongSIVindependence;
      ++StrongSIVsuccesses;
      return true;
    }
    Result.DV[Level].Distance = SE->getConstant(Distance);
    if (Distance.sgt(0))
      Result.DV[Level].Direction &= Dependence::DVEntry::LT;
    else if (Distance.slt(0))
      Result.DV[Level].Direction &= Dependence::DVEntry::GT;
    else
      Result.DV[Level].Direction &= Dependence::DVEntry::EQ;
    ++StrongSIVsuccesses;
  } else if (Delta->isZero()) {
    // Check if coefficient could be zero. If so, 0/0 is undefined and we
    // cannot conclude that only same-iteration dependencies exist.
    // When coeff=0, all iterations access the same location.
    if (SE->isKnownNonZero(Coeff)) {
      LLVM_DEBUG(
          dbgs() << "\t    Coefficient proven non-zero by SCEV analysis\n");
    } else {
      // Cannot prove at compile time, would need runtime assumption.
      if (UnderRuntimeAssumptions) {
        const SCEVPredicate *Pred = SE->getComparePredicate(
            ICmpInst::ICMP_NE, Coeff, SE->getZero(Coeff->getType()));
        Result.Assumptions = Result.Assumptions.getUnionWith(Pred, *SE);
        LLVM_DEBUG(dbgs() << "\t    Added runtime assumption: " << *Coeff
                          << " != 0\n");
      } else {
        // Cannot add runtime assumptions, this test cannot handle this case.
        // Let more complex tests try.
        LLVM_DEBUG(dbgs() << "\t    Would need runtime assumption " << *Coeff
                          << " != 0, but not allowed. Failing this test.\n");
        return false;
      }
    }
    // Since 0/X == 0 (where X is known non-zero or assumed non-zero).
    Result.DV[Level].Distance = Delta;
    Result.DV[Level].Direction &= Dependence::DVEntry::EQ;
    ++StrongSIVsuccesses;
  } else {
    if (Coeff->isOne()) {
      LLVM_DEBUG(dbgs() << "\t    Distance = " << *Delta << "\n");
      Result.DV[Level].Distance = Delta; // since X/1 == X
    } else {
      Result.Consistent = false;
    }

    // maybe we can get a useful direction
    bool DeltaMaybeZero = !SE->isKnownNonZero(Delta);
    bool DeltaMaybePositive = !SE->isKnownNonPositive(Delta);
    bool DeltaMaybeNegative = !SE->isKnownNonNegative(Delta);
    bool CoeffMaybePositive = !SE->isKnownNonPositive(Coeff);
    bool CoeffMaybeNegative = !SE->isKnownNonNegative(Coeff);
    // The double negatives above are confusing.
    // It helps to read !SE->isKnownNonZero(Delta)
    // as "Delta might be Zero"
    unsigned NewDirection = Dependence::DVEntry::NONE;
    if ((DeltaMaybePositive && CoeffMaybePositive) ||
        (DeltaMaybeNegative && CoeffMaybeNegative))
      NewDirection = Dependence::DVEntry::LT;
    if (DeltaMaybeZero)
      NewDirection |= Dependence::DVEntry::EQ;
    if ((DeltaMaybeNegative && CoeffMaybePositive) ||
        (DeltaMaybePositive && CoeffMaybeNegative))
      NewDirection |= Dependence::DVEntry::GT;
    if (NewDirection < Result.DV[Level].Direction)
      ++StrongSIVsuccesses;
    Result.DV[Level].Direction &= NewDirection;
  }
  return false;
}

// weakCrossingSIVtest -
// From the paper, Practical Dependence Testing, Section 4.2.2
//
// When we have a pair of subscripts of the form [c1 + a*i] and [c2 - a*i],
// where i is an induction variable, c1 and c2 are loop invariant,
// and a is a constant, we can solve it exactly using the
// Weak-Crossing SIV test.
//
// Given c1 + a*i = c2 - a*i', we can look for the intersection of
// the two lines, where i = i', yielding
//
//    c1 + a*i = c2 - a*i
//    2a*i = c2 - c1
//    i = (c2 - c1)/2a
//
// If i < 0, there is no dependence.
// If i > upperbound, there is no dependence.
// If i = 0 (i.e., if c1 = c2), there's a dependence with distance = 0.
// If i = upperbound, there's a dependence with distance = 0.
// If i is integral, there's a dependence (all directions).
// If the non-integer part = 1/2, there's a dependence (<> directions).
// Otherwise, there's no dependence.
//
// Can prove independence. Failing that,
// can sometimes refine the directions.
// Can determine iteration for splitting.
//
// Return true if dependence disproved.
bool DependenceInfo::weakCrossingSIVtest(const SCEV *Coeff,
                                         const SCEV *SrcConst,
                                         const SCEV *DstConst,
                                         const Loop *CurSrcLoop,
                                         const Loop *CurDstLoop, unsigned Level,
                                         FullDependence &Result) const {
  if (!isDependenceTestEnabled(DependenceTestType::WeakCrossingSIV))
    return false;

  LLVM_DEBUG(dbgs() << "\tWeak-Crossing SIV test\n");
  LLVM_DEBUG(dbgs() << "\t    Coeff = " << *Coeff << "\n");
  LLVM_DEBUG(dbgs() << "\t    SrcConst = " << *SrcConst << "\n");
  LLVM_DEBUG(dbgs() << "\t    DstConst = " << *DstConst << "\n");
  ++WeakCrossingSIVapplications;
  assert(0 < Level && Level <= CommonLevels && "Level out of range");
  Level--;
  Result.Consistent = false;
  const SCEV *Delta = SE->getMinusSCEV(DstConst, SrcConst);
  LLVM_DEBUG(dbgs() << "\t    Delta = " << *Delta << "\n");
  if (Delta->isZero()) {
    Result.DV[Level].Direction &= ~Dependence::DVEntry::LT;
    Result.DV[Level].Direction &= ~Dependence::DVEntry::GT;
    ++WeakCrossingSIVsuccesses;
    if (!Result.DV[Level].Direction) {
      ++WeakCrossingSIVindependence;
      return true;
    }
    Result.DV[Level].Distance = Delta; // = 0
    return false;
  }
  const SCEVConstant *ConstCoeff = dyn_cast<SCEVConstant>(Coeff);
  if (!ConstCoeff)
    return false;

  if (SE->isKnownNegative(ConstCoeff)) {
    ConstCoeff = dyn_cast<SCEVConstant>(SE->getNegativeSCEV(ConstCoeff));
    assert(ConstCoeff &&
           "dynamic cast of negative of ConstCoeff should yield constant");
    Delta = SE->getNegativeSCEV(Delta);
  }
  assert(SE->isKnownPositive(ConstCoeff) && "ConstCoeff should be positive");

  const SCEVConstant *ConstDelta = dyn_cast<SCEVConstant>(Delta);
  if (!ConstDelta)
    return false;

  // We're certain that ConstCoeff > 0; therefore,
  // if Delta < 0, then no dependence.
  LLVM_DEBUG(dbgs() << "\t    Delta = " << *Delta << "\n");
  LLVM_DEBUG(dbgs() << "\t    ConstCoeff = " << *ConstCoeff << "\n");
  if (SE->isKnownNegative(Delta)) {
    // No dependence, Delta < 0
    ++WeakCrossingSIVindependence;
    ++WeakCrossingSIVsuccesses;
    return true;
  }

  // We're certain that Delta > 0 and ConstCoeff > 0.
  // Check Delta/(2*ConstCoeff) against upper loop bound
  if (const SCEV *UpperBound =
          collectUpperBound(CurSrcLoop, Delta->getType())) {
    LLVM_DEBUG(dbgs() << "\t    UpperBound = " << *UpperBound << "\n");
    const SCEV *ConstantTwo = SE->getConstant(UpperBound->getType(), 2);
    const SCEV *ML =
        SE->getMulExpr(SE->getMulExpr(ConstCoeff, UpperBound), ConstantTwo);
    LLVM_DEBUG(dbgs() << "\t    ML = " << *ML << "\n");
    if (SE->isKnownPredicate(CmpInst::ICMP_SGT, Delta, ML)) {
      // Delta too big, no dependence
      ++WeakCrossingSIVindependence;
      ++WeakCrossingSIVsuccesses;
      return true;
    }
    if (SE->isKnownPredicate(CmpInst::ICMP_EQ, Delta, ML)) {
      // i = i' = UB
      Result.DV[Level].Direction &= ~Dependence::DVEntry::LT;
      Result.DV[Level].Direction &= ~Dependence::DVEntry::GT;
      ++WeakCrossingSIVsuccesses;
      if (!Result.DV[Level].Direction) {
        ++WeakCrossingSIVindependence;
        return true;
      }
      Result.DV[Level].Distance = SE->getZero(Delta->getType());
      return false;
    }
  }

  // check that Coeff divides Delta
  APInt APDelta = ConstDelta->getAPInt();
  APInt APCoeff = ConstCoeff->getAPInt();
  APInt Distance = APDelta; // these need to be initialzed
  APInt Remainder = APDelta;
  APInt::sdivrem(APDelta, APCoeff, Distance, Remainder);
  LLVM_DEBUG(dbgs() << "\t    Remainder = " << Remainder << "\n");
  if (Remainder != 0) {
    // Coeff doesn't divide Delta, no dependence
    ++WeakCrossingSIVindependence;
    ++WeakCrossingSIVsuccesses;
    return true;
  }
  LLVM_DEBUG(dbgs() << "\t    Distance = " << Distance << "\n");

  // if 2*Coeff doesn't divide Delta, then the equal direction isn't possible
  APInt Two = APInt(Distance.getBitWidth(), 2, true);
  Remainder = Distance.srem(Two);
  LLVM_DEBUG(dbgs() << "\t    Remainder = " << Remainder << "\n");
  if (Remainder != 0) {
    // Equal direction isn't possible
    Result.DV[Level].Direction &= ~Dependence::DVEntry::EQ;
    ++WeakCrossingSIVsuccesses;
  }
  return false;
}

// Kirch's algorithm, from
//
//        Optimizing Supercompilers for Supercomputers
//        Michael Wolfe
//        MIT Press, 1989
//
// Program 2.1, page 29.
// Computes the GCD of AM and BM.
// Also finds a solution to the equation ax - by = gcd(a, b).
// Returns true if dependence disproved; i.e., gcd does not divide Delta.
//
// We don't use OverflowSafeSignedAPInt here because it's known that this
// algorithm doesn't overflow.
static bool findGCD(unsigned Bits, const APInt &AM, const APInt &BM,
                    const APInt &Delta, APInt &G, APInt &X, APInt &Y) {
  APInt A0(Bits, 1, true), A1(Bits, 0, true);
  APInt B0(Bits, 0, true), B1(Bits, 1, true);
  APInt G0 = AM.abs();
  APInt G1 = BM.abs();
  APInt Q = G0; // these need to be initialized
  APInt R = G0;
  APInt::sdivrem(G0, G1, Q, R);
  while (R != 0) {
    // clang-format off
    APInt A2 = A0 - Q*A1; A0 = A1; A1 = A2;
    APInt B2 = B0 - Q*B1; B0 = B1; B1 = B2;
    G0 = G1; G1 = R;
    // clang-format on
    APInt::sdivrem(G0, G1, Q, R);
  }
  G = G1;
  LLVM_DEBUG(dbgs() << "\t    GCD = " << G << "\n");
  X = AM.slt(0) ? -A1 : A1;
  Y = BM.slt(0) ? B1 : -B1;

  // make sure gcd divides Delta
  R = Delta.srem(G);
  if (R != 0)
    return true; // gcd doesn't divide Delta, no dependence
  Q = Delta.sdiv(G);
  return false;
}

static OverflowSafeSignedAPInt
floorOfQuotient(const OverflowSafeSignedAPInt &OA,
                const OverflowSafeSignedAPInt &OB) {
  if (!OA || !OB)
    return OverflowSafeSignedAPInt();

  APInt A = *OA;
  APInt B = *OB;
  APInt Q = A; // these need to be initialized
  APInt R = A;
  APInt::sdivrem(A, B, Q, R);
  if (R == 0)
    return Q;
  if ((A.sgt(0) && B.sgt(0)) || (A.slt(0) && B.slt(0)))
    return Q;
  return OverflowSafeSignedAPInt(Q) - 1;
}

static OverflowSafeSignedAPInt
ceilingOfQuotient(const OverflowSafeSignedAPInt &OA,
                  const OverflowSafeSignedAPInt &OB) {
  if (!OA || !OB)
    return OverflowSafeSignedAPInt();

  APInt A = *OA;
  APInt B = *OB;
  APInt Q = A; // these need to be initialized
  APInt R = A;
  APInt::sdivrem(A, B, Q, R);
  if (R == 0)
    return Q;
  if ((A.sgt(0) && B.sgt(0)) || (A.slt(0) && B.slt(0)))
    return OverflowSafeSignedAPInt(Q) + 1;
  return Q;
}

/// Given an affine expression of the form A*k + B, where k is an arbitrary
/// integer, infer the possible range of k based on the known range of the
/// affine expression. If we know A*k + B is non-negative, i.e.,
///
///   A*k + B >= 0
///
/// we can derive the following inequalities for k when A is positive:
///
///   k >= -B / A
///
/// Since k is an integer, it means k is greater than or equal to the
/// ceil(-B / A).
///
/// If the upper bound of the affine expression \p UB is passed, the following
/// inequality can be derived as well:
///
///   A*k + B <= UB
///
/// which leads to:
///
///   k <= (UB - B) / A
///
/// Again, as k is an integer, it means k is less than or equal to the
/// floor((UB - B) / A).
///
/// The similar logic applies when A is negative, but the inequalities sign flip
/// while working with them.
///
/// Preconditions: \p A is non-zero, and we know A*k + B is non-negative.
static std::pair<OverflowSafeSignedAPInt, OverflowSafeSignedAPInt>
inferDomainOfAffine(OverflowSafeSignedAPInt A, OverflowSafeSignedAPInt B,
                    OverflowSafeSignedAPInt UB) {
  assert(A && B && "A and B must be available");
  assert(*A != 0 && "A must be non-zero");
  OverflowSafeSignedAPInt TL, TU;
  if (A->sgt(0)) {
    TL = ceilingOfQuotient(-B, A);
    LLVM_DEBUG(if (TL) dbgs() << "\t    Possible TL = " << *TL << "\n");

    // New bound check - modification to Banerjee's e3 check
    TU = floorOfQuotient(UB - B, A);
    LLVM_DEBUG(if (TU) dbgs() << "\t    Possible TU = " << *TU << "\n");
  } else {
    TU = floorOfQuotient(-B, A);
    LLVM_DEBUG(if (TU) dbgs() << "\t    Possible TU = " << *TU << "\n");

    // New bound check - modification to Banerjee's e3 check
    TL = ceilingOfQuotient(UB - B, A);
    LLVM_DEBUG(if (TL) dbgs() << "\t    Possible TL = " << *TL << "\n");
  }
  return std::make_pair(TL, TU);
}

// exactSIVtest -
// When we have a pair of subscripts of the form [c1 + a1*i] and [c2 + a2*i],
// where i is an induction variable, c1 and c2 are loop invariant, and a1
// and a2 are constant, we can solve it exactly using an algorithm developed
// by Banerjee and Wolfe. See Algorithm 6.2.1 (case 2.5) in:
//
//        Dependence Analysis for Supercomputing
//        Utpal Banerjee
//        Kluwer Academic Publishers, 1988
//
// It's slower than the specialized tests (strong SIV, weak-zero SIV, etc),
// so use them if possible. They're also a bit better with symbolics and,
// in the case of the strong SIV test, can compute Distances.
//
// Return true if dependence disproved.
//
// This is a modified version of the original Banerjee algorithm. The original
// only tested whether Dst depends on Src. This algorithm extends that and
// returns all the dependencies that exist between Dst and Src.
bool DependenceInfo::exactSIVtest(const SCEV *SrcCoeff, const SCEV *DstCoeff,
                                  const SCEV *SrcConst, const SCEV *DstConst,
                                  const Loop *CurSrcLoop,
                                  const Loop *CurDstLoop, unsigned Level,
                                  FullDependence &Result) const {
  if (!isDependenceTestEnabled(DependenceTestType::ExactSIV))
    return false;

  LLVM_DEBUG(dbgs() << "\tExact SIV test\n");
  LLVM_DEBUG(dbgs() << "\t    SrcCoeff = " << *SrcCoeff << " = AM\n");
  LLVM_DEBUG(dbgs() << "\t    DstCoeff = " << *DstCoeff << " = BM\n");
  LLVM_DEBUG(dbgs() << "\t    SrcConst = " << *SrcConst << "\n");
  LLVM_DEBUG(dbgs() << "\t    DstConst = " << *DstConst << "\n");
  ++ExactSIVapplications;
  assert(0 < Level && Level <= CommonLevels && "Level out of range");
  Level--;
  Result.Consistent = false;
  const SCEV *Delta = minusSCEVNoSignedOverflow(DstConst, SrcConst, *SE);
  if (!Delta)
    return false;
  LLVM_DEBUG(dbgs() << "\t    Delta = " << *Delta << "\n");
  const SCEVConstant *ConstDelta = dyn_cast<SCEVConstant>(Delta);
  const SCEVConstant *ConstSrcCoeff = dyn_cast<SCEVConstant>(SrcCoeff);
  const SCEVConstant *ConstDstCoeff = dyn_cast<SCEVConstant>(DstCoeff);
  if (!ConstDelta || !ConstSrcCoeff || !ConstDstCoeff)
    return false;

  // find gcd
  APInt G, X, Y;
  APInt AM = ConstSrcCoeff->getAPInt();
  APInt BM = ConstDstCoeff->getAPInt();
  APInt CM = ConstDelta->getAPInt();
  unsigned Bits = AM.getBitWidth();
  if (findGCD(Bits, AM, BM, CM, G, X, Y)) {
    // gcd doesn't divide Delta, no dependence
    ++ExactSIVindependence;
    ++ExactSIVsuccesses;
    return true;
  }

  LLVM_DEBUG(dbgs() << "\t    X = " << X << ", Y = " << Y << "\n");

  // since SCEV construction normalizes, LM = 0
  std::optional<APInt> UM;
  // UM is perhaps unavailable, let's check
  if (const SCEVConstant *CUB =
          collectConstantUpperBound(CurSrcLoop, Delta->getType())) {
    UM = CUB->getAPInt();
    LLVM_DEBUG(dbgs() << "\t    UM = " << *UM << "\n");
  }

  APInt TU(APInt::getSignedMaxValue(Bits));
  APInt TL(APInt::getSignedMinValue(Bits));
  APInt TC = CM.sdiv(G);
  APInt TX = X * TC;
  APInt TY = Y * TC;
  LLVM_DEBUG(dbgs() << "\t    TC = " << TC << "\n");
  LLVM_DEBUG(dbgs() << "\t    TX = " << TX << "\n");
  LLVM_DEBUG(dbgs() << "\t    TY = " << TY << "\n");

  APInt TB = BM.sdiv(G);
  APInt TA = AM.sdiv(G);

  // At this point, we have the following equations:
  //
  //   TA*i0 - TB*i1 = TC
  //
  // Also, we know that the all pairs of (i0, i1) can be expressed as:
  //
  //   (TX + k*TB, TY + k*TA)
  //
  // where k is an arbitrary integer.
  auto [TL0, TU0] = inferDomainOfAffine(TB, TX, UM);
  auto [TL1, TU1] = inferDomainOfAffine(TA, TY, UM);

  auto CreateVec = [](const OverflowSafeSignedAPInt &V0,
                      const OverflowSafeSignedAPInt &V1) {
    SmallVector<APInt, 2> Vec;
    if (V0)
      Vec.push_back(*V0);
    if (V1)
      Vec.push_back(*V1);
    return Vec;
  };

  SmallVector<APInt, 2> TLVec = CreateVec(TL0, TL1);
  SmallVector<APInt, 2> TUVec = CreateVec(TU0, TU1);

  LLVM_DEBUG(dbgs() << "\t    TA = " << TA << "\n");
  LLVM_DEBUG(dbgs() << "\t    TB = " << TB << "\n");

  if (TLVec.empty() || TUVec.empty())
    return false;
  TL = APIntOps::smax(TLVec.front(), TLVec.back());
  TU = APIntOps::smin(TUVec.front(), TUVec.back());
  LLVM_DEBUG(dbgs() << "\t    TL = " << TL << "\n");
  LLVM_DEBUG(dbgs() << "\t    TU = " << TU << "\n");

  if (TL.sgt(TU)) {
    ++ExactSIVindependence;
    ++ExactSIVsuccesses;
    return true;
  }

  // explore directions
  unsigned NewDirection = Dependence::DVEntry::NONE;
  OverflowSafeSignedAPInt LowerDistance, UpperDistance;
  OverflowSafeSignedAPInt OTY(TY), OTX(TX), OTA(TA), OTB(TB), OTL(TL), OTU(TU);
  // NOTE: It's unclear whether these calculations can overflow. At the moment,
  // we conservatively assume they can.
  if (TA.sgt(TB)) {
    LowerDistance = (OTY - OTX) + (OTA - OTB) * OTL;
    UpperDistance = (OTY - OTX) + (OTA - OTB) * OTU;
  } else {
    LowerDistance = (OTY - OTX) + (OTA - OTB) * OTU;
    UpperDistance = (OTY - OTX) + (OTA - OTB) * OTL;
  }

  if (!LowerDistance || !UpperDistance)
    return false;

  LLVM_DEBUG(dbgs() << "\t    LowerDistance = " << *LowerDistance << "\n");
  LLVM_DEBUG(dbgs() << "\t    UpperDistance = " << *UpperDistance << "\n");

  if (LowerDistance->sle(0) && UpperDistance->sge(0)) {
    NewDirection |= Dependence::DVEntry::EQ;
    ++ExactSIVsuccesses;
  }
  if (LowerDistance->slt(0)) {
    NewDirection |= Dependence::DVEntry::GT;
    ++ExactSIVsuccesses;
  }
  if (UpperDistance->sgt(0)) {
    NewDirection |= Dependence::DVEntry::LT;
    ++ExactSIVsuccesses;
  }

  // finished
  Result.DV[Level].Direction &= NewDirection;
  if (Result.DV[Level].Direction == Dependence::DVEntry::NONE)
    ++ExactSIVindependence;
  LLVM_DEBUG(dbgs() << "\t    Result = ");
  LLVM_DEBUG(Result.dump(dbgs()));
  return Result.DV[Level].Direction == Dependence::DVEntry::NONE;
}

// Return true if the divisor evenly divides the dividend.
static bool isRemainderZero(const SCEVConstant *Dividend,
                            const SCEVConstant *Divisor) {
  const APInt &ConstDividend = Dividend->getAPInt();
  const APInt &ConstDivisor = Divisor->getAPInt();
  return ConstDividend.srem(ConstDivisor) == 0;
}

// weakZeroSrcSIVtest -
// From the paper, Practical Dependence Testing, Section 4.2.2
//
// When we have a pair of subscripts of the form [c1] and [c2 + a*i],
// where i is an induction variable, c1 and c2 are loop invariant,
// and a is a constant, we can solve it exactly using the
// Weak-Zero SIV test.
//
// Given
//
//    c1 = c2 + a*i
//
// we get
//
//    (c1 - c2)/a = i
//
// If i is not an integer, there's no dependence.
// If i < 0 or > UB, there's no dependence.
// If i = 0, the direction is >= and peeling the
// 1st iteration will break the dependence.
// If i = UB, the direction is <= and peeling the
// last iteration will break the dependence.
// Otherwise, the direction is *.
//
// Can prove independence. Failing that, we can sometimes refine
// the directions. Can sometimes show that first or last
// iteration carries all the dependences (so worth peeling).
//
// (see also weakZeroDstSIVtest)
//
// Return true if dependence disproved.
bool DependenceInfo::weakZeroSrcSIVtest(const SCEV *DstCoeff,
                                        const SCEV *SrcConst,
                                        const SCEV *DstConst,
                                        const Loop *CurSrcLoop,
                                        const Loop *CurDstLoop, unsigned Level,
                                        FullDependence &Result) const {
  if (!isDependenceTestEnabled(DependenceTestType::WeakZeroSIV))
    return false;

  // For the WeakSIV test, it's possible the loop isn't common to
  // the Src and Dst loops. If it isn't, then there's no need to
  // record a direction.
  LLVM_DEBUG(dbgs() << "\tWeak-Zero (src) SIV test\n");
  LLVM_DEBUG(dbgs() << "\t    DstCoeff = " << *DstCoeff << "\n");
  LLVM_DEBUG(dbgs() << "\t    SrcConst = " << *SrcConst << "\n");
  LLVM_DEBUG(dbgs() << "\t    DstConst = " << *DstConst << "\n");
  ++WeakZeroSIVapplications;
  assert(0 < Level && Level <= MaxLevels && "Level out of range");
  Level--;
  Result.Consistent = false;
  const SCEV *Delta = SE->getMinusSCEV(SrcConst, DstConst);
  LLVM_DEBUG(dbgs() << "\t    Delta = " << *Delta << "\n");
  if (SE->isKnownPredicate(CmpInst::ICMP_EQ, SrcConst, DstConst)) {
    if (Level < CommonLevels) {
      Result.DV[Level].Direction &= Dependence::DVEntry::GE;
      Result.DV[Level].PeelFirst = true;
      ++WeakZeroSIVsuccesses;
    }
    return false; // dependences caused by first iteration
  }
  const SCEVConstant *ConstCoeff = dyn_cast<SCEVConstant>(DstCoeff);
  if (!ConstCoeff)
    return false;

  // Since ConstCoeff is constant, !isKnownNegative means it's non-negative.
  // TODO: Bail out if it's a signed minimum value.
  const SCEV *AbsCoeff = SE->isKnownNegative(ConstCoeff)
                             ? SE->getNegativeSCEV(ConstCoeff)
                             : ConstCoeff;
  const SCEV *NewDelta =
      SE->isKnownNegative(ConstCoeff) ? SE->getNegativeSCEV(Delta) : Delta;

  // check that Delta/SrcCoeff < iteration count
  // really check NewDelta < count*AbsCoeff
  if (const SCEV *UpperBound =
          collectUpperBound(CurSrcLoop, Delta->getType())) {
    LLVM_DEBUG(dbgs() << "\t    UpperBound = " << *UpperBound << "\n");
    const SCEV *Product = SE->getMulExpr(AbsCoeff, UpperBound);
    if (SE->isKnownPredicate(CmpInst::ICMP_SGT, NewDelta, Product)) {
      ++WeakZeroSIVindependence;
      ++WeakZeroSIVsuccesses;
      return true;
    }
    if (SE->isKnownPredicate(CmpInst::ICMP_EQ, NewDelta, Product)) {
      // dependences caused by last iteration
      if (Level < CommonLevels) {
        Result.DV[Level].Direction &= Dependence::DVEntry::LE;
        Result.DV[Level].PeelLast = true;
        ++WeakZeroSIVsuccesses;
      }
      return false;
    }
  }

  // check that Delta/SrcCoeff >= 0
  // really check that NewDelta >= 0
  if (SE->isKnownNegative(NewDelta)) {
    // No dependence, newDelta < 0
    ++WeakZeroSIVindependence;
    ++WeakZeroSIVsuccesses;
    return true;
  }

  // if SrcCoeff doesn't divide Delta, then no dependence
  if (isa<SCEVConstant>(Delta) &&
      !isRemainderZero(cast<SCEVConstant>(Delta), ConstCoeff)) {
    ++WeakZeroSIVindependence;
    ++WeakZeroSIVsuccesses;
    return true;
  }
  return false;
}

// weakZeroDstSIVtest -
// From the paper, Practical Dependence Testing, Section 4.2.2
//
// When we have a pair of subscripts of the form [c1 + a*i] and [c2],
// where i is an induction variable, c1 and c2 are loop invariant,
// and a is a constant, we can solve it exactly using the
// Weak-Zero SIV test.
//
// Given
//
//    c1 + a*i = c2
//
// we get
//
//    i = (c2 - c1)/a
//
// If i is not an integer, there's no dependence.
// If i < 0 or > UB, there's no dependence.
// If i = 0, the direction is <= and peeling the
// 1st iteration will break the dependence.
// If i = UB, the direction is >= and peeling the
// last iteration will break the dependence.
// Otherwise, the direction is *.
//
// Can prove independence. Failing that, we can sometimes refine
// the directions. Can sometimes show that first or last
// iteration carries all the dependences (so worth peeling).
//
// (see also weakZeroSrcSIVtest)
//
// Return true if dependence disproved.
bool DependenceInfo::weakZeroDstSIVtest(const SCEV *SrcCoeff,
                                        const SCEV *SrcConst,
                                        const SCEV *DstConst,
                                        const Loop *CurSrcLoop,
                                        const Loop *CurDstLoop, unsigned Level,
                                        FullDependence &Result) const {
  if (!isDependenceTestEnabled(DependenceTestType::WeakZeroSIV))
    return false;

  // For the WeakSIV test, it's possible the loop isn't common to the
  // Src and Dst loops. If it isn't, then there's no need to record a direction.
  LLVM_DEBUG(dbgs() << "\tWeak-Zero (dst) SIV test\n");
  LLVM_DEBUG(dbgs() << "\t    SrcCoeff = " << *SrcCoeff << "\n");
  LLVM_DEBUG(dbgs() << "\t    SrcConst = " << *SrcConst << "\n");
  LLVM_DEBUG(dbgs() << "\t    DstConst = " << *DstConst << "\n");
  ++WeakZeroSIVapplications;
  assert(0 < Level && Level <= SrcLevels && "Level out of range");
  Level--;
  Result.Consistent = false;
  const SCEV *Delta = SE->getMinusSCEV(DstConst, SrcConst);
  LLVM_DEBUG(dbgs() << "\t    Delta = " << *Delta << "\n");
  if (SE->isKnownPredicate(CmpInst::ICMP_EQ, DstConst, SrcConst)) {
    if (Level < CommonLevels) {
      Result.DV[Level].Direction &= Dependence::DVEntry::LE;
      Result.DV[Level].PeelFirst = true;
      ++WeakZeroSIVsuccesses;
    }
    return false; // dependences caused by first iteration
  }
  const SCEVConstant *ConstCoeff = dyn_cast<SCEVConstant>(SrcCoeff);
  if (!ConstCoeff)
    return false;

  // Since ConstCoeff is constant, !isKnownNegative means it's non-negative.
  // TODO: Bail out if it's a signed minimum value.
  const SCEV *AbsCoeff = SE->isKnownNegative(ConstCoeff)
                             ? SE->getNegativeSCEV(ConstCoeff)
                             : ConstCoeff;
  const SCEV *NewDelta =
      SE->isKnownNegative(ConstCoeff) ? SE->getNegativeSCEV(Delta) : Delta;

  // check that Delta/SrcCoeff < iteration count
  // really check NewDelta < count*AbsCoeff
  if (const SCEV *UpperBound =
          collectUpperBound(CurSrcLoop, Delta->getType())) {
    LLVM_DEBUG(dbgs() << "\t    UpperBound = " << *UpperBound << "\n");
    const SCEV *Product = SE->getMulExpr(AbsCoeff, UpperBound);
    if (SE->isKnownPredicate(CmpInst::ICMP_SGT, NewDelta, Product)) {
      ++WeakZeroSIVindependence;
      ++WeakZeroSIVsuccesses;
      return true;
    }
    if (SE->isKnownPredicate(CmpInst::ICMP_EQ, NewDelta, Product)) {
      // dependences caused by last iteration
      if (Level < CommonLevels) {
        Result.DV[Level].Direction &= Dependence::DVEntry::GE;
        Result.DV[Level].PeelLast = true;
        ++WeakZeroSIVsuccesses;
      }
      return false;
    }
  }

  // check that Delta/SrcCoeff >= 0
  // really check that NewDelta >= 0
  if (SE->isKnownNegative(NewDelta)) {
    // No dependence, newDelta < 0
    ++WeakZeroSIVindependence;
    ++WeakZeroSIVsuccesses;
    return true;
  }

  // if SrcCoeff doesn't divide Delta, then no dependence
  if (isa<SCEVConstant>(Delta) &&
      !isRemainderZero(cast<SCEVConstant>(Delta), ConstCoeff)) {
    ++WeakZeroSIVindependence;
    ++WeakZeroSIVsuccesses;
    return true;
  }
  return false;
}

// exactRDIVtest - Tests the RDIV subscript pair for dependence.
// Things of the form [c1 + a*i] and [c2 + b*j],
// where i and j are induction variable, c1 and c2 are loop invariant,
// and a and b are constants.
// Returns true if any possible dependence is disproved.
// Marks the result as inconsistent.
// Works in some cases that symbolicRDIVtest doesn't, and vice versa.
bool DependenceInfo::exactRDIVtest(const SCEV *SrcCoeff, const SCEV *DstCoeff,
                                   const SCEV *SrcConst, const SCEV *DstConst,
                                   const Loop *SrcLoop, const Loop *DstLoop,
                                   FullDependence &Result) const {
  if (!isDependenceTestEnabled(DependenceTestType::ExactRDIV))
    return false;

  LLVM_DEBUG(dbgs() << "\tExact RDIV test\n");
  LLVM_DEBUG(dbgs() << "\t    SrcCoeff = " << *SrcCoeff << " = AM\n");
  LLVM_DEBUG(dbgs() << "\t    DstCoeff = " << *DstCoeff << " = BM\n");
  LLVM_DEBUG(dbgs() << "\t    SrcConst = " << *SrcConst << "\n");
  LLVM_DEBUG(dbgs() << "\t    DstConst = " << *DstConst << "\n");
  ++ExactRDIVapplications;
  Result.Consistent = false;
  const SCEV *Delta = SE->getMinusSCEV(DstConst, SrcConst);
  LLVM_DEBUG(dbgs() << "\t    Delta = " << *Delta << "\n");
  const SCEVConstant *ConstDelta = dyn_cast<SCEVConstant>(Delta);
  const SCEVConstant *ConstSrcCoeff = dyn_cast<SCEVConstant>(SrcCoeff);
  const SCEVConstant *ConstDstCoeff = dyn_cast<SCEVConstant>(DstCoeff);
  if (!ConstDelta || !ConstSrcCoeff || !ConstDstCoeff)
    return false;

  // find gcd
  APInt G, X, Y;
  APInt AM = ConstSrcCoeff->getAPInt();
  APInt BM = ConstDstCoeff->getAPInt();
  APInt CM = ConstDelta->getAPInt();
  unsigned Bits = AM.getBitWidth();
  if (findGCD(Bits, AM, BM, CM, G, X, Y)) {
    // gcd doesn't divide Delta, no dependence
    ++ExactRDIVindependence;
    return true;
  }

  LLVM_DEBUG(dbgs() << "\t    X = " << X << ", Y = " << Y << "\n");

  // since SCEV construction seems to normalize, LM = 0
  std::optional<APInt> SrcUM;
  // SrcUM is perhaps unavailable, let's check
  if (const SCEVConstant *UpperBound =
          collectConstantUpperBound(SrcLoop, Delta->getType())) {
    SrcUM = UpperBound->getAPInt();
    LLVM_DEBUG(dbgs() << "\t    SrcUM = " << *SrcUM << "\n");
  }

  std::optional<APInt> DstUM;
  // UM is perhaps unavailable, let's check
  if (const SCEVConstant *UpperBound =
          collectConstantUpperBound(DstLoop, Delta->getType())) {
    DstUM = UpperBound->getAPInt();
    LLVM_DEBUG(dbgs() << "\t    DstUM = " << *DstUM << "\n");
  }

  APInt TU(APInt::getSignedMaxValue(Bits));
  APInt TL(APInt::getSignedMinValue(Bits));
  APInt TC = CM.sdiv(G);
  APInt TX = X * TC;
  APInt TY = Y * TC;
  LLVM_DEBUG(dbgs() << "\t    TC = " << TC << "\n");
  LLVM_DEBUG(dbgs() << "\t    TX = " << TX << "\n");
  LLVM_DEBUG(dbgs() << "\t    TY = " << TY << "\n");

  APInt TB = BM.sdiv(G);
  APInt TA = AM.sdiv(G);

  // At this point, we have the following equations:
  //
  //   TA*i - TB*j = TC
  //
  // Also, we know that the all pairs of (i, j) can be expressed as:
  //
  //   (TX + k*TB, TY + k*TA)
  //
  // where k is an arbitrary integer.
  auto [TL0, TU0] = inferDomainOfAffine(TB, TX, SrcUM);
  auto [TL1, TU1] = inferDomainOfAffine(TA, TY, DstUM);

  LLVM_DEBUG(dbgs() << "\t    TA = " << TA << "\n");
  LLVM_DEBUG(dbgs() << "\t    TB = " << TB << "\n");

  auto CreateVec = [](const OverflowSafeSignedAPInt &V0,
                      const OverflowSafeSignedAPInt &V1) {
    SmallVector<APInt, 2> Vec;
    if (V0)
      Vec.push_back(*V0);
    if (V1)
      Vec.push_back(*V1);
    return Vec;
  };

  SmallVector<APInt, 2> TLVec = CreateVec(TL0, TL1);
  SmallVector<APInt, 2> TUVec = CreateVec(TU0, TU1);
  if (TLVec.empty() || TUVec.empty())
    return false;

  TL = APIntOps::smax(TLVec.front(), TLVec.back());
  TU = APIntOps::smin(TUVec.front(), TUVec.back());
  LLVM_DEBUG(dbgs() << "\t    TL = " << TL << "\n");
  LLVM_DEBUG(dbgs() << "\t    TU = " << TU << "\n");

  if (TL.sgt(TU))
    ++ExactRDIVindependence;
  return TL.sgt(TU);
}

// symbolicRDIVtest -
// In Section 4.5 of the Practical Dependence Testing paper,the authors
// introduce a special case of Banerjee's Inequalities (also called the
// Extreme-Value Test) that can handle some of the SIV and RDIV cases,
// particularly cases with symbolics. Since it's only able to disprove
// dependence (not compute distances or directions), we'll use it as a
// fall back for the other tests.
//
// When we have a pair of subscripts of the form [c1 + a1*i] and [c2 + a2*j]
// where i and j are induction variables and c1 and c2 are loop invariants,
// we can use the symbolic tests to disprove some dependences, serving as a
// backup for the RDIV test. Note that i and j can be the same variable,
// letting this test serve as a backup for the various SIV tests.
//
// For a dependence to exist, c1 + a1*i must equal c2 + a2*j for some
//  0 <= i <= N1 and some 0 <= j <= N2, where N1 and N2 are the (normalized)
// loop bounds for the i and j loops, respectively. So, ...
//
// c1 + a1*i = c2 + a2*j
// a1*i - a2*j = c2 - c1
//
// To test for a dependence, we compute c2 - c1 and make sure it's in the
// range of the maximum and minimum possible values of a1*i - a2*j.
// Considering the signs of a1 and a2, we have 4 possible cases:
//
// 1) If a1 >= 0 and a2 >= 0, then
//        a1*0 - a2*N2 <= c2 - c1 <= a1*N1 - a2*0
//              -a2*N2 <= c2 - c1 <= a1*N1
//
// 2) If a1 >= 0 and a2 <= 0, then
//        a1*0 - a2*0 <= c2 - c1 <= a1*N1 - a2*N2
//                  0 <= c2 - c1 <= a1*N1 - a2*N2
//
// 3) If a1 <= 0 and a2 >= 0, then
//        a1*N1 - a2*N2 <= c2 - c1 <= a1*0 - a2*0
//        a1*N1 - a2*N2 <= c2 - c1 <= 0
//
// 4) If a1 <= 0 and a2 <= 0, then
//        a1*N1 - a2*0  <= c2 - c1 <= a1*0 - a2*N2
//        a1*N1         <= c2 - c1 <=       -a2*N2
//
// return true if dependence disproved
bool DependenceInfo::symbolicRDIVtest(const SCEV *A1, const SCEV *A2,
                                      const SCEV *C1, const SCEV *C2,
                                      const Loop *Loop1,
                                      const Loop *Loop2) const {
  if (!isDependenceTestEnabled(DependenceTestType::SymbolicRDIV))
    return false;

  ++SymbolicRDIVapplications;
  LLVM_DEBUG(dbgs() << "\ttry symbolic RDIV test\n");
  LLVM_DEBUG(dbgs() << "\t    A1 = " << *A1);
  LLVM_DEBUG(dbgs() << ", type = " << *A1->getType() << "\n");
  LLVM_DEBUG(dbgs() << "\t    A2 = " << *A2 << "\n");
  LLVM_DEBUG(dbgs() << "\t    C1 = " << *C1 << "\n");
  LLVM_DEBUG(dbgs() << "\t    C2 = " << *C2 << "\n");
  const SCEV *N1 = collectUpperBound(Loop1, A1->getType());
  const SCEV *N2 = collectUpperBound(Loop2, A1->getType());
  LLVM_DEBUG(if (N1) dbgs() << "\t    N1 = " << *N1 << "\n");
  LLVM_DEBUG(if (N2) dbgs() << "\t    N2 = " << *N2 << "\n");
  const SCEV *C2_C1 = SE->getMinusSCEV(C2, C1);
  const SCEV *C1_C2 = SE->getMinusSCEV(C1, C2);
  LLVM_DEBUG(dbgs() << "\t    C2 - C1 = " << *C2_C1 << "\n");
  LLVM_DEBUG(dbgs() << "\t    C1 - C2 = " << *C1_C2 << "\n");
  if (SE->isKnownNonNegative(A1)) {
    if (SE->isKnownNonNegative(A2)) {
      // A1 >= 0 && A2 >= 0
      if (N1) {
        // make sure that c2 - c1 <= a1*N1
        const SCEV *A1N1 = SE->getMulExpr(A1, N1);
        LLVM_DEBUG(dbgs() << "\t    A1*N1 = " << *A1N1 << "\n");
        if (SE->isKnownPredicate(CmpInst::ICMP_SGT, C2_C1, A1N1)) {
          ++SymbolicRDIVindependence;
          return true;
        }
      }
      if (N2) {
        // make sure that -a2*N2 <= c2 - c1, or a2*N2 >= c1 - c2
        const SCEV *A2N2 = SE->getMulExpr(A2, N2);
        LLVM_DEBUG(dbgs() << "\t    A2*N2 = " << *A2N2 << "\n");
        if (SE->isKnownPredicate(CmpInst::ICMP_SLT, A2N2, C1_C2)) {
          ++SymbolicRDIVindependence;
          return true;
        }
      }
    } else if (SE->isKnownNonPositive(A2)) {
      // a1 >= 0 && a2 <= 0
      if (N1 && N2) {
        // make sure that c2 - c1 <= a1*N1 - a2*N2
        const SCEV *A1N1 = SE->getMulExpr(A1, N1);
        const SCEV *A2N2 = SE->getMulExpr(A2, N2);
        const SCEV *A1N1_A2N2 = SE->getMinusSCEV(A1N1, A2N2);
        LLVM_DEBUG(dbgs() << "\t    A1*N1 - A2*N2 = " << *A1N1_A2N2 << "\n");
        if (SE->isKnownPredicate(CmpInst::ICMP_SGT, C2_C1, A1N1_A2N2)) {
          ++SymbolicRDIVindependence;
          return true;
        }
      }
      // make sure that 0 <= c2 - c1
      if (SE->isKnownNegative(C2_C1)) {
        ++SymbolicRDIVindependence;
        return true;
      }
    }
  } else if (SE->isKnownNonPositive(A1)) {
    if (SE->isKnownNonNegative(A2)) {
      // a1 <= 0 && a2 >= 0
      if (N1 && N2) {
        // make sure that a1*N1 - a2*N2 <= c2 - c1
        const SCEV *A1N1 = SE->getMulExpr(A1, N1);
        const SCEV *A2N2 = SE->getMulExpr(A2, N2);
        const SCEV *A1N1_A2N2 = SE->getMinusSCEV(A1N1, A2N2);
        LLVM_DEBUG(dbgs() << "\t    A1*N1 - A2*N2 = " << *A1N1_A2N2 << "\n");
        if (SE->isKnownPredicate(CmpInst::ICMP_SGT, A1N1_A2N2, C2_C1)) {
          ++SymbolicRDIVindependence;
          return true;
        }
      }
      // make sure that c2 - c1 <= 0
      if (SE->isKnownPositive(C2_C1)) {
        ++SymbolicRDIVindependence;
        return true;
      }
    } else if (SE->isKnownNonPositive(A2)) {
      // a1 <= 0 && a2 <= 0
      if (N1) {
        // make sure that a1*N1 <= c2 - c1
        const SCEV *A1N1 = SE->getMulExpr(A1, N1);
        LLVM_DEBUG(dbgs() << "\t    A1*N1 = " << *A1N1 << "\n");
        if (SE->isKnownPredicate(CmpInst::ICMP_SGT, A1N1, C2_C1)) {
          ++SymbolicRDIVindependence;
          return true;
        }
      }
      if (N2) {
        // make sure that c2 - c1 <= -a2*N2, or c1 - c2 >= a2*N2
        const SCEV *A2N2 = SE->getMulExpr(A2, N2);
        LLVM_DEBUG(dbgs() << "\t    A2*N2 = " << *A2N2 << "\n");
        if (SE->isKnownPredicate(CmpInst::ICMP_SLT, C1_C2, A2N2)) {
          ++SymbolicRDIVindependence;
          return true;
        }
      }
    }
  }
  return false;
}

// testSIV -
// When we have a pair of subscripts of the form [c1 + a1*i] and [c2 - a2*i]
// where i is an induction variable, c1 and c2 are loop invariant, and a1 and
// a2 are constant, we attack it with an SIV test. While they can all be
// solved with the Exact SIV test, it's worthwhile to use simpler tests when
// they apply; they're cheaper and sometimes more precise.
//
// Return true if dependence disproved.
bool DependenceInfo::testSIV(const SCEV *Src, const SCEV *Dst, unsigned &Level,
                             FullDependence &Result,
                             bool UnderRuntimeAssumptions) {
  LLVM_DEBUG(dbgs() << "    src = " << *Src << "\n");
  LLVM_DEBUG(dbgs() << "    dst = " << *Dst << "\n");
  const SCEVAddRecExpr *SrcAddRec = dyn_cast<SCEVAddRecExpr>(Src);
  const SCEVAddRecExpr *DstAddRec = dyn_cast<SCEVAddRecExpr>(Dst);
  if (SrcAddRec && DstAddRec) {
    const SCEV *SrcConst = SrcAddRec->getStart();
    const SCEV *DstConst = DstAddRec->getStart();
    const SCEV *SrcCoeff = SrcAddRec->getStepRecurrence(*SE);
    const SCEV *DstCoeff = DstAddRec->getStepRecurrence(*SE);
    const Loop *CurSrcLoop = SrcAddRec->getLoop();
    const Loop *CurDstLoop = DstAddRec->getLoop();
    assert(haveSameSD(CurSrcLoop, CurDstLoop) &&
           "Loops in the SIV test should have the same iteration space and "
           "depth");
    Level = mapSrcLoop(CurSrcLoop);
    bool disproven;
    if (SrcCoeff == DstCoeff)
      disproven =
          strongSIVtest(SrcCoeff, SrcConst, DstConst, CurSrcLoop, CurDstLoop,
                        Level, Result, UnderRuntimeAssumptions);
    else if (SrcCoeff == SE->getNegativeSCEV(DstCoeff))
      disproven = weakCrossingSIVtest(SrcCoeff, SrcConst, DstConst, CurSrcLoop,
                                      CurDstLoop, Level, Result);
    else
      disproven = exactSIVtest(SrcCoeff, DstCoeff, SrcConst, DstConst,
                               CurSrcLoop, CurDstLoop, Level, Result);
    return disproven || gcdMIVtest(Src, Dst, Result) ||
           symbolicRDIVtest(SrcCoeff, DstCoeff, SrcConst, DstConst, CurSrcLoop,
                            CurDstLoop);
  }
  if (SrcAddRec) {
    const SCEV *SrcConst = SrcAddRec->getStart();
    const SCEV *SrcCoeff = SrcAddRec->getStepRecurrence(*SE);
    const SCEV *DstConst = Dst;
    const Loop *CurSrcLoop = SrcAddRec->getLoop();
    Level = mapSrcLoop(CurSrcLoop);
    return weakZeroDstSIVtest(SrcCoeff, SrcConst, DstConst, CurSrcLoop,
                              CurSrcLoop, Level, Result) ||
           gcdMIVtest(Src, Dst, Result);
  }
  if (DstAddRec) {
    const SCEV *DstConst = DstAddRec->getStart();
    const SCEV *DstCoeff = DstAddRec->getStepRecurrence(*SE);
    const SCEV *SrcConst = Src;
    const Loop *CurDstLoop = DstAddRec->getLoop();
    Level = mapDstLoop(CurDstLoop);
    return weakZeroSrcSIVtest(DstCoeff, SrcConst, DstConst, CurDstLoop,
                              CurDstLoop, Level, Result) ||
           gcdMIVtest(Src, Dst, Result);
  }
  llvm_unreachable("SIV test expected at least one AddRec");
  return false;
}

// testRDIV -
// When we have a pair of subscripts of the form [c1 + a1*i] and [c2 + a2*j]
// where i and j are induction variables, c1 and c2 are loop invariant,
// and a1 and a2 are constant, we can solve it exactly with an easy adaptation
// of the Exact SIV test, the Restricted Double Index Variable (RDIV) test.
// It doesn't make sense to talk about distance or direction in this case,
// so there's no point in making special versions of the Strong SIV test or
// the Weak-crossing SIV test.
//
// With minor algebra, this test can also be used for things like
// [c1 + a1*i + a2*j][c2].
//
// Return true if dependence disproved.
bool DependenceInfo::testRDIV(const SCEV *Src, const SCEV *Dst,
                              FullDependence &Result) const {
  // we have 3 possible situations here:
  //   1) [a*i + b] and [c*j + d]
  //   2) [a*i + c*j + b] and [d]
  //   3) [b] and [a*i + c*j + d]
  // We need to find what we've got and get organized

  const SCEV *SrcConst, *DstConst;
  const SCEV *SrcCoeff, *DstCoeff;
  const Loop *SrcLoop, *DstLoop;

  LLVM_DEBUG(dbgs() << "    src = " << *Src << "\n");
  LLVM_DEBUG(dbgs() << "    dst = " << *Dst << "\n");
  const SCEVAddRecExpr *SrcAddRec = dyn_cast<SCEVAddRecExpr>(Src);
  const SCEVAddRecExpr *DstAddRec = dyn_cast<SCEVAddRecExpr>(Dst);
  if (SrcAddRec && DstAddRec) {
    SrcConst = SrcAddRec->getStart();
    SrcCoeff = SrcAddRec->getStepRecurrence(*SE);
    SrcLoop = SrcAddRec->getLoop();
    DstConst = DstAddRec->getStart();
    DstCoeff = DstAddRec->getStepRecurrence(*SE);
    DstLoop = DstAddRec->getLoop();
  } else if (SrcAddRec) {
    if (const SCEVAddRecExpr *tmpAddRec =
            dyn_cast<SCEVAddRecExpr>(SrcAddRec->getStart())) {
      SrcConst = tmpAddRec->getStart();
      SrcCoeff = tmpAddRec->getStepRecurrence(*SE);
      SrcLoop = tmpAddRec->getLoop();
      DstConst = Dst;
      DstCoeff = SE->getNegativeSCEV(SrcAddRec->getStepRecurrence(*SE));
      DstLoop = SrcAddRec->getLoop();
    } else
      llvm_unreachable("RDIV reached by surprising SCEVs");
  } else if (DstAddRec) {
    if (const SCEVAddRecExpr *tmpAddRec =
            dyn_cast<SCEVAddRecExpr>(DstAddRec->getStart())) {
      DstConst = tmpAddRec->getStart();
      DstCoeff = tmpAddRec->getStepRecurrence(*SE);
      DstLoop = tmpAddRec->getLoop();
      SrcConst = Src;
      SrcCoeff = SE->getNegativeSCEV(DstAddRec->getStepRecurrence(*SE));
      SrcLoop = DstAddRec->getLoop();
    } else
      llvm_unreachable("RDIV reached by surprising SCEVs");
  } else
    llvm_unreachable("RDIV expected at least one AddRec");
  return exactRDIVtest(SrcCoeff, DstCoeff, SrcConst, DstConst, SrcLoop, DstLoop,
                       Result) ||
         gcdMIVtest(Src, Dst, Result) ||
         symbolicRDIVtest(SrcCoeff, DstCoeff, SrcConst, DstConst, SrcLoop,
                          DstLoop);
}

// Tests the single-subscript MIV pair (Src and Dst) for dependence.
// Return true if dependence disproved.
// Can sometimes refine direction vectors.
bool DependenceInfo::testMIV(const SCEV *Src, const SCEV *Dst,
                             const SmallBitVector &Loops,
                             FullDependence &Result) const {
  LLVM_DEBUG(dbgs() << "    src = " << *Src << "\n");
  LLVM_DEBUG(dbgs() << "    dst = " << *Dst << "\n");
  Result.Consistent = false;
  return gcdMIVtest(Src, Dst, Result) ||
         banerjeeMIVtest(Src, Dst, Loops, Result);
}

/// Given a SCEVMulExpr, returns its first operand if its first operand is a
/// constant and the product doesn't overflow in a signed sense. Otherwise,
/// returns std::nullopt. For example, given (10 * X * Y)<nsw>, it returns 10.
/// Notably, if it doesn't have nsw, the multiplication may overflow, and if
/// so, it may not a multiple of 10.
static std::optional<APInt> getConstantCoefficient(const SCEV *Expr) {
  if (const auto *Constant = dyn_cast<SCEVConstant>(Expr))
    return Constant->getAPInt();
  if (const auto *Product = dyn_cast<SCEVMulExpr>(Expr))
    if (const auto *Constant = dyn_cast<SCEVConstant>(Product->getOperand(0)))
      if (Product->hasNoSignedWrap())
        return Constant->getAPInt();
  return std::nullopt;
}

bool DependenceInfo::accumulateCoefficientsGCD(const SCEV *Expr,
                                               const Loop *CurLoop,
                                               const SCEV *&CurLoopCoeff,
                                               APInt &RunningGCD) const {
  // If RunningGCD is already 1, exit early.
  // TODO: It might be better to continue the recursion to find CurLoopCoeff.
  if (RunningGCD == 1)
    return true;

  const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(Expr);
  if (!AddRec) {
    assert(isLoopInvariant(Expr, CurLoop) &&
           "Expected loop invariant expression");
    return true;
  }

  assert(AddRec->isAffine() && "Unexpected Expr");
  const SCEV *Start = AddRec->getStart();
  const SCEV *Step = AddRec->getStepRecurrence(*SE);
  if (AddRec->getLoop() == CurLoop) {
    CurLoopCoeff = Step;
  } else {
    std::optional<APInt> ConstCoeff = getConstantCoefficient(Step);

    // If the coefficient is the product of a constant and other stuff, we can
    // use the constant in the GCD computation.
    if (!ConstCoeff)
      return false;

    // TODO: What happens if ConstCoeff is the "most negative" signed number
    // (e.g. -128 for 8 bit wide APInt)?
    RunningGCD = APIntOps::GreatestCommonDivisor(RunningGCD, ConstCoeff->abs());
  }

  return accumulateCoefficientsGCD(Start, CurLoop, CurLoopCoeff, RunningGCD);
}

//===----------------------------------------------------------------------===//
// gcdMIVtest -
// Tests an MIV subscript pair for dependence.
// Returns true if any possible dependence is disproved.
// Marks the result as inconsistent.
// Can sometimes disprove the equal direction for 1 or more loops,
// as discussed in Michael Wolfe's book,
// High Performance Compilers for Parallel Computing, page 235.
//
// We spend some effort (code!) to handle cases like
// [10*i + 5*N*j + 15*M + 6], where i and j are induction variables,
// but M and N are just loop-invariant variables.
// This should help us handle linearized subscripts;
// also makes this test a useful backup to the various SIV tests.
//
// It occurs to me that the presence of loop-invariant variables
// changes the nature of the test from "greatest common divisor"
// to "a common divisor".
bool DependenceInfo::gcdMIVtest(const SCEV *Src, const SCEV *Dst,
                                FullDependence &Result) const {
  if (!isDependenceTestEnabled(DependenceTestType::GCDMIV))
    return false;

  LLVM_DEBUG(dbgs() << "starting gcd\n");
  ++GCDapplications;
  unsigned BitWidth = SE->getTypeSizeInBits(Src->getType());
  APInt RunningGCD = APInt::getZero(BitWidth);

  // Examine Src coefficients.
  // Compute running GCD and record source constant.
  // Because we're looking for the constant at the end of the chain,
  // we can't quit the loop just because the GCD == 1.
  const SCEV *Coefficients = Src;
  while (const SCEVAddRecExpr *AddRec =
             dyn_cast<SCEVAddRecExpr>(Coefficients)) {
    const SCEV *Coeff = AddRec->getStepRecurrence(*SE);
    // If the coefficient is the product of a constant and other stuff,
    // we can use the constant in the GCD computation.
    std::optional<APInt> ConstCoeff = getConstantCoefficient(Coeff);
    if (!ConstCoeff)
      return false;
    RunningGCD = APIntOps::GreatestCommonDivisor(RunningGCD, ConstCoeff->abs());
    Coefficients = AddRec->getStart();
  }
  const SCEV *SrcConst = Coefficients;

  // Examine Dst coefficients.
  // Compute running GCD and record destination constant.
  // Because we're looking for the constant at the end of the chain,
  // we can't quit the loop just because the GCD == 1.
  Coefficients = Dst;
  while (const SCEVAddRecExpr *AddRec =
             dyn_cast<SCEVAddRecExpr>(Coefficients)) {
    const SCEV *Coeff = AddRec->getStepRecurrence(*SE);
    // If the coefficient is the product of a constant and other stuff,
    // we can use the constant in the GCD computation.
    std::optional<APInt> ConstCoeff = getConstantCoefficient(Coeff);
    if (!ConstCoeff)
      return false;
    RunningGCD = APIntOps::GreatestCommonDivisor(RunningGCD, ConstCoeff->abs());
    Coefficients = AddRec->getStart();
  }
  const SCEV *DstConst = Coefficients;

  APInt ExtraGCD = APInt::getZero(BitWidth);
  const SCEV *Delta = minusSCEVNoSignedOverflow(DstConst, SrcConst, *SE);
  if (!Delta)
    return false;
  LLVM_DEBUG(dbgs() << "    Delta = " << *Delta << "\n");
  const SCEVConstant *Constant = dyn_cast<SCEVConstant>(Delta);
  if (!Constant)
    return false;
  APInt ConstDelta = cast<SCEVConstant>(Constant)->getAPInt();
  LLVM_DEBUG(dbgs() << "    ConstDelta = " << ConstDelta << "\n");
  if (ConstDelta == 0)
    return false;
  LLVM_DEBUG(dbgs() << "    RunningGCD = " << RunningGCD << "\n");
  APInt Remainder = ConstDelta.srem(RunningGCD);
  if (Remainder != 0) {
    ++GCDindependence;
    return true;
  }

  // Try to disprove equal directions.
  // For example, given a subscript pair [3*i + 2*j] and [i' + 2*j' - 1],
  // the code above can't disprove the dependence because the GCD = 1.
  // So we consider what happen if i = i' and what happens if j = j'.
  // If i = i', we can simplify the subscript to [2*i + 2*j] and [2*j' - 1],
  // which is infeasible, so we can disallow the = direction for the i level.
  // Setting j = j' doesn't help matters, so we end up with a direction vector
  // of [<>, *]
  //
  // Given A[5*i + 10*j*M + 9*M*N] and A[15*i + 20*j*M - 21*N*M + 5],
  // we need to remember that the constant part is 5 and the RunningGCD should
  // be initialized to ExtraGCD = 30.
  LLVM_DEBUG(dbgs() << "    ExtraGCD = " << ExtraGCD << '\n');

  bool Improved = false;
  Coefficients = Src;
  while (const SCEVAddRecExpr *AddRec =
             dyn_cast<SCEVAddRecExpr>(Coefficients)) {
    Coefficients = AddRec->getStart();
    const Loop *CurLoop = AddRec->getLoop();
    RunningGCD = ExtraGCD;
    const SCEV *SrcCoeff = AddRec->getStepRecurrence(*SE);
    const SCEV *DstCoeff = SE->getMinusSCEV(SrcCoeff, SrcCoeff);

    if (!accumulateCoefficientsGCD(Src, CurLoop, SrcCoeff, RunningGCD) ||
        !accumulateCoefficientsGCD(Dst, CurLoop, DstCoeff, RunningGCD))
      return false;

    Delta = SE->getMinusSCEV(SrcCoeff, DstCoeff);
    // If the coefficient is the product of a constant and other stuff,
    // we can use the constant in the GCD computation.
    std::optional<APInt> ConstCoeff = getConstantCoefficient(Delta);
    if (!ConstCoeff)
      // The difference of the two coefficients might not be a product
      // or constant, in which case we give up on this direction.
      continue;
    RunningGCD = APIntOps::GreatestCommonDivisor(RunningGCD, ConstCoeff->abs());
    LLVM_DEBUG(dbgs() << "\tRunningGCD = " << RunningGCD << "\n");
    if (RunningGCD != 0) {
      Remainder = ConstDelta.srem(RunningGCD);
      LLVM_DEBUG(dbgs() << "\tRemainder = " << Remainder << "\n");
      if (Remainder != 0) {
        unsigned Level = mapSrcLoop(CurLoop);
        Result.DV[Level - 1].Direction &= ~Dependence::DVEntry::EQ;
        Improved = true;
      }
    }
  }
  if (Improved)
    ++GCDsuccesses;
  LLVM_DEBUG(dbgs() << "all done\n");
  return false;
}

//===----------------------------------------------------------------------===//
// banerjeeMIVtest -
// Use Banerjee's Inequalities to test an MIV subscript pair.
// (Wolfe, in the race-car book, calls this the Extreme Value Test.)
// Generally follows the discussion in Section 2.5.2 of
//
//    Optimizing Supercompilers for Supercomputers
//    Michael Wolfe
//
// The inequalities given on page 25 are simplified in that loops are
// normalized so that the lower bound is always 0 and the stride is always 1.
// For example, Wolfe gives
//
//     LB^<_k = (A^-_k - B_k)^- (U_k - L_k - N_k) + (A_k - B_k)L_k - B_k N_k
//
// where A_k is the coefficient of the kth index in the source subscript,
// B_k is the coefficient of the kth index in the destination subscript,
// U_k is the upper bound of the kth index, L_k is the lower bound of the Kth
// index, and N_k is the stride of the kth index. Since all loops are normalized
// by the SCEV package, N_k = 1 and L_k = 0, allowing us to simplify the
// equation to
//
//     LB^<_k = (A^-_k - B_k)^- (U_k - 0 - 1) + (A_k - B_k)0 - B_k 1
//            = (A^-_k - B_k)^- (U_k - 1)  - B_k
//
// Similar simplifications are possible for the other equations.
//
// When we can't determine the number of iterations for a loop,
// we use NULL as an indicator for the worst case, infinity.
// When computing the upper bound, NULL denotes +inf;
// for the lower bound, NULL denotes -inf.
//
// Return true if dependence disproved.
bool DependenceInfo::banerjeeMIVtest(const SCEV *Src, const SCEV *Dst,
                                     const SmallBitVector &Loops,
                                     FullDependence &Result) const {
  if (!isDependenceTestEnabled(DependenceTestType::BanerjeeMIV))
    return false;

  LLVM_DEBUG(dbgs() << "starting Banerjee\n");
  ++BanerjeeApplications;
  LLVM_DEBUG(dbgs() << "    Src = " << *Src << '\n');
  const SCEV *A0;
  CoefficientInfo *A = collectCoeffInfo(Src, true, A0);
  LLVM_DEBUG(dbgs() << "    Dst = " << *Dst << '\n');
  const SCEV *B0;
  CoefficientInfo *B = collectCoeffInfo(Dst, false, B0);
  BoundInfo *Bound = new BoundInfo[MaxLevels + 1];
  const SCEV *Delta = SE->getMinusSCEV(B0, A0);
  LLVM_DEBUG(dbgs() << "\tDelta = " << *Delta << '\n');

  // Compute bounds for all the * directions.
  LLVM_DEBUG(dbgs() << "\tBounds[*]\n");
  for (unsigned K = 1; K <= MaxLevels; ++K) {
    Bound[K].Iterations = A[K].Iterations ? A[K].Iterations : B[K].Iterations;
    Bound[K].Direction = Dependence::DVEntry::ALL;
    Bound[K].DirSet = Dependence::DVEntry::NONE;
    findBoundsALL(A, B, Bound, K);
#ifndef NDEBUG
    LLVM_DEBUG(dbgs() << "\t    " << K << '\t');
    if (Bound[K].Lower[Dependence::DVEntry::ALL])
      LLVM_DEBUG(dbgs() << *Bound[K].Lower[Dependence::DVEntry::ALL] << '\t');
    else
      LLVM_DEBUG(dbgs() << "-inf\t");
    if (Bound[K].Upper[Dependence::DVEntry::ALL])
      LLVM_DEBUG(dbgs() << *Bound[K].Upper[Dependence::DVEntry::ALL] << '\n');
    else
      LLVM_DEBUG(dbgs() << "+inf\n");
#endif
  }

  // Test the *, *, *, ... case.
  bool Disproved = false;
  if (testBounds(Dependence::DVEntry::ALL, 0, Bound, Delta)) {
    // Explore the direction vector hierarchy.
    unsigned DepthExpanded = 0;
    unsigned NewDeps =
        exploreDirections(1, A, B, Bound, Loops, DepthExpanded, Delta);
    if (NewDeps > 0) {
      bool Improved = false;
      for (unsigned K = 1; K <= CommonLevels; ++K) {
        if (Loops[K]) {
          unsigned Old = Result.DV[K - 1].Direction;
          Result.DV[K - 1].Direction = Old & Bound[K].DirSet;
          Improved |= Old != Result.DV[K - 1].Direction;
          if (!Result.DV[K - 1].Direction) {
            Improved = false;
            Disproved = true;
            break;
          }
        }
      }
      if (Improved)
        ++BanerjeeSuccesses;
    } else {
      ++BanerjeeIndependence;
      Disproved = true;
    }
  } else {
    ++BanerjeeIndependence;
    Disproved = true;
  }
  delete[] Bound;
  delete[] A;
  delete[] B;
  return Disproved;
}

// Hierarchically expands the direction vector
// search space, combining the directions of discovered dependences
// in the DirSet field of Bound. Returns the number of distinct
// dependences discovered. If the dependence is disproved,
// it will return 0.
unsigned DependenceInfo::exploreDirections(unsigned Level, CoefficientInfo *A,
                                           CoefficientInfo *B, BoundInfo *Bound,
                                           const SmallBitVector &Loops,
                                           unsigned &DepthExpanded,
                                           const SCEV *Delta) const {
  // This algorithm has worst case complexity of O(3^n), where 'n' is the number
  // of common loop levels. To avoid excessive compile-time, pessimize all the
  // results and immediately return when the number of common levels is beyond
  // the given threshold.
  if (CommonLevels > MIVMaxLevelThreshold) {
    LLVM_DEBUG(dbgs() << "Number of common levels exceeded the threshold. MIV "
                         "direction exploration is terminated.\n");
    for (unsigned K = 1; K <= CommonLevels; ++K)
      if (Loops[K])
        Bound[K].DirSet = Dependence::DVEntry::ALL;
    return 1;
  }

  if (Level > CommonLevels) {
    // record result
    LLVM_DEBUG(dbgs() << "\t[");
    for (unsigned K = 1; K <= CommonLevels; ++K) {
      if (Loops[K]) {
        Bound[K].DirSet |= Bound[K].Direction;
#ifndef NDEBUG
        switch (Bound[K].Direction) {
        case Dependence::DVEntry::LT:
          LLVM_DEBUG(dbgs() << " <");
          break;
        case Dependence::DVEntry::EQ:
          LLVM_DEBUG(dbgs() << " =");
          break;
        case Dependence::DVEntry::GT:
          LLVM_DEBUG(dbgs() << " >");
          break;
        case Dependence::DVEntry::ALL:
          LLVM_DEBUG(dbgs() << " *");
          break;
        default:
          llvm_unreachable("unexpected Bound[K].Direction");
        }
#endif
      }
    }
    LLVM_DEBUG(dbgs() << " ]\n");
    return 1;
  }
  if (Loops[Level]) {
    if (Level > DepthExpanded) {
      DepthExpanded = Level;
      // compute bounds for <, =, > at current level
      findBoundsLT(A, B, Bound, Level);
      findBoundsGT(A, B, Bound, Level);
      findBoundsEQ(A, B, Bound, Level);
#ifndef NDEBUG
      LLVM_DEBUG(dbgs() << "\tBound for level = " << Level << '\n');
      LLVM_DEBUG(dbgs() << "\t    <\t");
      if (Bound[Level].Lower[Dependence::DVEntry::LT])
        LLVM_DEBUG(dbgs() << *Bound[Level].Lower[Dependence::DVEntry::LT]
                          << '\t');
      else
        LLVM_DEBUG(dbgs() << "-inf\t");
      if (Bound[Level].Upper[Dependence::DVEntry::LT])
        LLVM_DEBUG(dbgs() << *Bound[Level].Upper[Dependence::DVEntry::LT]
                          << '\n');
      else
        LLVM_DEBUG(dbgs() << "+inf\n");
      LLVM_DEBUG(dbgs() << "\t    =\t");
      if (Bound[Level].Lower[Dependence::DVEntry::EQ])
        LLVM_DEBUG(dbgs() << *Bound[Level].Lower[Dependence::DVEntry::EQ]
                          << '\t');
      else
        LLVM_DEBUG(dbgs() << "-inf\t");
      if (Bound[Level].Upper[Dependence::DVEntry::EQ])
        LLVM_DEBUG(dbgs() << *Bound[Level].Upper[Dependence::DVEntry::EQ]
                          << '\n');
      else
        LLVM_DEBUG(dbgs() << "+inf\n");
      LLVM_DEBUG(dbgs() << "\t    >\t");
      if (Bound[Level].Lower[Dependence::DVEntry::GT])
        LLVM_DEBUG(dbgs() << *Bound[Level].Lower[Dependence::DVEntry::GT]
                          << '\t');
      else
        LLVM_DEBUG(dbgs() << "-inf\t");
      if (Bound[Level].Upper[Dependence::DVEntry::GT])
        LLVM_DEBUG(dbgs() << *Bound[Level].Upper[Dependence::DVEntry::GT]
                          << '\n');
      else
        LLVM_DEBUG(dbgs() << "+inf\n");
#endif
    }

    unsigned NewDeps = 0;

    // test bounds for <, *, *, ...
    if (testBounds(Dependence::DVEntry::LT, Level, Bound, Delta))
      NewDeps += exploreDirections(Level + 1, A, B, Bound, Loops, DepthExpanded,
                                   Delta);

    // Test bounds for =, *, *, ...
    if (testBounds(Dependence::DVEntry::EQ, Level, Bound, Delta))
      NewDeps += exploreDirections(Level + 1, A, B, Bound, Loops, DepthExpanded,
                                   Delta);

    // test bounds for >, *, *, ...
    if (testBounds(Dependence::DVEntry::GT, Level, Bound, Delta))
      NewDeps += exploreDirections(Level + 1, A, B, Bound, Loops, DepthExpanded,
                                   Delta);

    Bound[Level].Direction = Dependence::DVEntry::ALL;
    return NewDeps;
  } else
    return exploreDirections(Level + 1, A, B, Bound, Loops, DepthExpanded,
                             Delta);
}

// Returns true iff the current bounds are plausible.
bool DependenceInfo::testBounds(unsigned char DirKind, unsigned Level,
                                BoundInfo *Bound, const SCEV *Delta) const {
  Bound[Level].Direction = DirKind;
  if (const SCEV *LowerBound = getLowerBound(Bound))
    if (SE->isKnownPredicate(CmpInst::ICMP_SGT, LowerBound, Delta))
      return false;
  if (const SCEV *UpperBound = getUpperBound(Bound))
    if (SE->isKnownPredicate(CmpInst::ICMP_SGT, Delta, UpperBound))
      return false;
  return true;
}

// Computes the upper and lower bounds for level K
// using the * direction. Records them in Bound.
// Wolfe gives the equations
//
//    LB^*_k = (A^-_k - B^+_k)(U_k - L_k) + (A_k - B_k)L_k
//    UB^*_k = (A^+_k - B^-_k)(U_k - L_k) + (A_k - B_k)L_k
//
// Since we normalize loops, we can simplify these equations to
//
//    LB^*_k = (A^-_k - B^+_k)U_k
//    UB^*_k = (A^+_k - B^-_k)U_k
//
// We must be careful to handle the case where the upper bound is unknown.
// Note that the lower bound is always <= 0
// and the upper bound is always >= 0.
void DependenceInfo::findBoundsALL(CoefficientInfo *A, CoefficientInfo *B,
                                   BoundInfo *Bound, unsigned K) const {
  Bound[K].Lower[Dependence::DVEntry::ALL] =
      nullptr; // Default value = -infinity.
  Bound[K].Upper[Dependence::DVEntry::ALL] =
      nullptr; // Default value = +infinity.
  if (Bound[K].Iterations) {
    Bound[K].Lower[Dependence::DVEntry::ALL] = SE->getMulExpr(
        SE->getMinusSCEV(A[K].NegPart, B[K].PosPart), Bound[K].Iterations);
    Bound[K].Upper[Dependence::DVEntry::ALL] = SE->getMulExpr(
        SE->getMinusSCEV(A[K].PosPart, B[K].NegPart), Bound[K].Iterations);
  } else {
    // If the difference is 0, we won't need to know the number of iterations.
    if (SE->isKnownPredicate(CmpInst::ICMP_EQ, A[K].NegPart, B[K].PosPart))
      Bound[K].Lower[Dependence::DVEntry::ALL] =
          SE->getZero(A[K].Coeff->getType());
    if (SE->isKnownPredicate(CmpInst::ICMP_EQ, A[K].PosPart, B[K].NegPart))
      Bound[K].Upper[Dependence::DVEntry::ALL] =
          SE->getZero(A[K].Coeff->getType());
  }
}

// Computes the upper and lower bounds for level K
// using the = direction. Records them in Bound.
// Wolfe gives the equations
//
//    LB^=_k = (A_k - B_k)^- (U_k - L_k) + (A_k - B_k)L_k
//    UB^=_k = (A_k - B_k)^+ (U_k - L_k) + (A_k - B_k)L_k
//
// Since we normalize loops, we can simplify these equations to
//
//    LB^=_k = (A_k - B_k)^- U_k
//    UB^=_k = (A_k - B_k)^+ U_k
//
// We must be careful to handle the case where the upper bound is unknown.
// Note that the lower bound is always <= 0
// and the upper bound is always >= 0.
void DependenceInfo::findBoundsEQ(CoefficientInfo *A, CoefficientInfo *B,
                                  BoundInfo *Bound, unsigned K) const {
  Bound[K].Lower[Dependence::DVEntry::EQ] =
      nullptr; // Default value = -infinity.
  Bound[K].Upper[Dependence::DVEntry::EQ] =
      nullptr; // Default value = +infinity.
  if (Bound[K].Iterations) {
    const SCEV *Delta = SE->getMinusSCEV(A[K].Coeff, B[K].Coeff);
    const SCEV *NegativePart = getNegativePart(Delta);
    Bound[K].Lower[Dependence::DVEntry::EQ] =
        SE->getMulExpr(NegativePart, Bound[K].Iterations);
    const SCEV *PositivePart = getPositivePart(Delta);
    Bound[K].Upper[Dependence::DVEntry::EQ] =
        SE->getMulExpr(PositivePart, Bound[K].Iterations);
  } else {
    // If the positive/negative part of the difference is 0,
    // we won't need to know the number of iterations.
    const SCEV *Delta = SE->getMinusSCEV(A[K].Coeff, B[K].Coeff);
    const SCEV *NegativePart = getNegativePart(Delta);
    if (NegativePart->isZero())
      Bound[K].Lower[Dependence::DVEntry::EQ] = NegativePart; // Zero
    const SCEV *PositivePart = getPositivePart(Delta);
    if (PositivePart->isZero())
      Bound[K].Upper[Dependence::DVEntry::EQ] = PositivePart; // Zero
  }
}

// Computes the upper and lower bounds for level K
// using the < direction. Records them in Bound.
// Wolfe gives the equations
//
//    LB^<_k = (A^-_k - B_k)^- (U_k - L_k - N_k) + (A_k - B_k)L_k - B_k N_k
//    UB^<_k = (A^+_k - B_k)^+ (U_k - L_k - N_k) + (A_k - B_k)L_k - B_k N_k
//
// Since we normalize loops, we can simplify these equations to
//
//    LB^<_k = (A^-_k - B_k)^- (U_k - 1) - B_k
//    UB^<_k = (A^+_k - B_k)^+ (U_k - 1) - B_k
//
// We must be careful to handle the case where the upper bound is unknown.
void DependenceInfo::findBoundsLT(CoefficientInfo *A, CoefficientInfo *B,
                                  BoundInfo *Bound, unsigned K) const {
  Bound[K].Lower[Dependence::DVEntry::LT] =
      nullptr; // Default value = -infinity.
  Bound[K].Upper[Dependence::DVEntry::LT] =
      nullptr; // Default value = +infinity.
  if (Bound[K].Iterations) {
    const SCEV *Iter_1 = SE->getMinusSCEV(
        Bound[K].Iterations, SE->getOne(Bound[K].Iterations->getType()));
    const SCEV *NegPart =
        getNegativePart(SE->getMinusSCEV(A[K].NegPart, B[K].Coeff));
    Bound[K].Lower[Dependence::DVEntry::LT] =
        SE->getMinusSCEV(SE->getMulExpr(NegPart, Iter_1), B[K].Coeff);
    const SCEV *PosPart =
        getPositivePart(SE->getMinusSCEV(A[K].PosPart, B[K].Coeff));
    Bound[K].Upper[Dependence::DVEntry::LT] =
        SE->getMinusSCEV(SE->getMulExpr(PosPart, Iter_1), B[K].Coeff);
  } else {
    // If the positive/negative part of the difference is 0,
    // we won't need to know the number of iterations.
    const SCEV *NegPart =
        getNegativePart(SE->getMinusSCEV(A[K].NegPart, B[K].Coeff));
    if (NegPart->isZero())
      Bound[K].Lower[Dependence::DVEntry::LT] = SE->getNegativeSCEV(B[K].Coeff);
    const SCEV *PosPart =
        getPositivePart(SE->getMinusSCEV(A[K].PosPart, B[K].Coeff));
    if (PosPart->isZero())
      Bound[K].Upper[Dependence::DVEntry::LT] = SE->getNegativeSCEV(B[K].Coeff);
  }
}

// Computes the upper and lower bounds for level K
// using the > direction. Records them in Bound.
// Wolfe gives the equations
//
//    LB^>_k = (A_k - B^+_k)^- (U_k - L_k - N_k) + (A_k - B_k)L_k + A_k N_k
//    UB^>_k = (A_k - B^-_k)^+ (U_k - L_k - N_k) + (A_k - B_k)L_k + A_k N_k
//
// Since we normalize loops, we can simplify these equations to
//
//    LB^>_k = (A_k - B^+_k)^- (U_k - 1) + A_k
//    UB^>_k = (A_k - B^-_k)^+ (U_k - 1) + A_k
//
// We must be careful to handle the case where the upper bound is unknown.
void DependenceInfo::findBoundsGT(CoefficientInfo *A, CoefficientInfo *B,
                                  BoundInfo *Bound, unsigned K) const {
  Bound[K].Lower[Dependence::DVEntry::GT] =
      nullptr; // Default value = -infinity.
  Bound[K].Upper[Dependence::DVEntry::GT] =
      nullptr; // Default value = +infinity.
  if (Bound[K].Iterations) {
    const SCEV *Iter_1 = SE->getMinusSCEV(
        Bound[K].Iterations, SE->getOne(Bound[K].Iterations->getType()));
    const SCEV *NegPart =
        getNegativePart(SE->getMinusSCEV(A[K].Coeff, B[K].PosPart));
    Bound[K].Lower[Dependence::DVEntry::GT] =
        SE->getAddExpr(SE->getMulExpr(NegPart, Iter_1), A[K].Coeff);
    const SCEV *PosPart =
        getPositivePart(SE->getMinusSCEV(A[K].Coeff, B[K].NegPart));
    Bound[K].Upper[Dependence::DVEntry::GT] =
        SE->getAddExpr(SE->getMulExpr(PosPart, Iter_1), A[K].Coeff);
  } else {
    // If the positive/negative part of the difference is 0,
    // we won't need to know the number of iterations.
    const SCEV *NegPart =
        getNegativePart(SE->getMinusSCEV(A[K].Coeff, B[K].PosPart));
    if (NegPart->isZero())
      Bound[K].Lower[Dependence::DVEntry::GT] = A[K].Coeff;
    const SCEV *PosPart =
        getPositivePart(SE->getMinusSCEV(A[K].Coeff, B[K].NegPart));
    if (PosPart->isZero())
      Bound[K].Upper[Dependence::DVEntry::GT] = A[K].Coeff;
  }
}

// X^+ = max(X, 0)
const SCEV *DependenceInfo::getPositivePart(const SCEV *X) const {
  return SE->getSMaxExpr(X, SE->getZero(X->getType()));
}

// X^- = min(X, 0)
const SCEV *DependenceInfo::getNegativePart(const SCEV *X) const {
  return SE->getSMinExpr(X, SE->getZero(X->getType()));
}

// Walks through the subscript,
// collecting each coefficient, the associated loop bounds,
// and recording its positive and negative parts for later use.
DependenceInfo::CoefficientInfo *
DependenceInfo::collectCoeffInfo(const SCEV *Subscript, bool SrcFlag,
                                 const SCEV *&Constant) const {
  const SCEV *Zero = SE->getZero(Subscript->getType());
  CoefficientInfo *CI = new CoefficientInfo[MaxLevels + 1];
  for (unsigned K = 1; K <= MaxLevels; ++K) {
    CI[K].Coeff = Zero;
    CI[K].PosPart = Zero;
    CI[K].NegPart = Zero;
    CI[K].Iterations = nullptr;
  }
  while (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(Subscript)) {
    const Loop *L = AddRec->getLoop();
    unsigned K = SrcFlag ? mapSrcLoop(L) : mapDstLoop(L);
    CI[K].Coeff = AddRec->getStepRecurrence(*SE);
    CI[K].PosPart = getPositivePart(CI[K].Coeff);
    CI[K].NegPart = getNegativePart(CI[K].Coeff);
    CI[K].Iterations = collectUpperBound(L, Subscript->getType());
    Subscript = AddRec->getStart();
  }
  Constant = Subscript;
#ifndef NDEBUG
  LLVM_DEBUG(dbgs() << "\tCoefficient Info\n");
  for (unsigned K = 1; K <= MaxLevels; ++K) {
    LLVM_DEBUG(dbgs() << "\t    " << K << "\t" << *CI[K].Coeff);
    LLVM_DEBUG(dbgs() << "\tPos Part = ");
    LLVM_DEBUG(dbgs() << *CI[K].PosPart);
    LLVM_DEBUG(dbgs() << "\tNeg Part = ");
    LLVM_DEBUG(dbgs() << *CI[K].NegPart);
    LLVM_DEBUG(dbgs() << "\tUpper Bound = ");
    if (CI[K].Iterations)
      LLVM_DEBUG(dbgs() << *CI[K].Iterations);
    else
      LLVM_DEBUG(dbgs() << "+inf");
    LLVM_DEBUG(dbgs() << '\n');
  }
  LLVM_DEBUG(dbgs() << "\t    Constant = " << *Subscript << '\n');
#endif
  return CI;
}

// Looks through all the bounds info and
// computes the lower bound given the current direction settings
// at each level. If the lower bound for any level is -inf,
// the result is -inf.
const SCEV *DependenceInfo::getLowerBound(BoundInfo *Bound) const {
  const SCEV *Sum = Bound[1].Lower[Bound[1].Direction];
  for (unsigned K = 2; Sum && K <= MaxLevels; ++K) {
    if (Bound[K].Lower[Bound[K].Direction])
      Sum = SE->getAddExpr(Sum, Bound[K].Lower[Bound[K].Direction]);
    else
      Sum = nullptr;
  }
  return Sum;
}

// Looks through all the bounds info and
// computes the upper bound given the current direction settings
// at each level. If the upper bound at any level is +inf,
// the result is +inf.
const SCEV *DependenceInfo::getUpperBound(BoundInfo *Bound) const {
  const SCEV *Sum = Bound[1].Upper[Bound[1].Direction];
  for (unsigned K = 2; Sum && K <= MaxLevels; ++K) {
    if (Bound[K].Upper[Bound[K].Direction])
      Sum = SE->getAddExpr(Sum, Bound[K].Upper[Bound[K].Direction]);
    else
      Sum = nullptr;
  }
  return Sum;
}

/// Check if we can delinearize the subscripts. If the SCEVs representing the
/// source and destination array references are recurrences on a nested loop,
/// this function flattens the nested recurrences into separate recurrences
/// for each loop level.
bool DependenceInfo::tryDelinearize(Instruction *Src, Instruction *Dst,
                                    SmallVectorImpl<Subscript> &Pair) {
  assert(isLoadOrStore(Src) && "instruction is not load or store");
  assert(isLoadOrStore(Dst) && "instruction is not load or store");
  Value *SrcPtr = getLoadStorePointerOperand(Src);
  Value *DstPtr = getLoadStorePointerOperand(Dst);
  Loop *SrcLoop = LI->getLoopFor(Src->getParent());
  Loop *DstLoop = LI->getLoopFor(Dst->getParent());
  const SCEV *SrcAccessFn = SE->getSCEVAtScope(SrcPtr, SrcLoop);
  const SCEV *DstAccessFn = SE->getSCEVAtScope(DstPtr, DstLoop);
  const SCEVUnknown *SrcBase =
      dyn_cast<SCEVUnknown>(SE->getPointerBase(SrcAccessFn));
  const SCEVUnknown *DstBase =
      dyn_cast<SCEVUnknown>(SE->getPointerBase(DstAccessFn));

  if (!SrcBase || !DstBase || SrcBase != DstBase)
    return false;

  SmallVector<const SCEV *, 4> SrcSubscripts, DstSubscripts;

  if (!tryDelinearizeFixedSize(Src, Dst, SrcAccessFn, DstAccessFn,
                               SrcSubscripts, DstSubscripts) &&
      !tryDelinearizeParametricSize(Src, Dst, SrcAccessFn, DstAccessFn,
                                    SrcSubscripts, DstSubscripts))
    return false;

  assert(isLoopInvariant(SrcBase, SrcLoop) &&
         isLoopInvariant(DstBase, DstLoop) &&
         "Expected SrcBase and DstBase to be loop invariant");

  int Size = SrcSubscripts.size();
  LLVM_DEBUG({
    dbgs() << "\nSrcSubscripts: ";
    for (int I = 0; I < Size; I++)
      dbgs() << *SrcSubscripts[I];
    dbgs() << "\nDstSubscripts: ";
    for (int I = 0; I < Size; I++)
      dbgs() << *DstSubscripts[I];
  });

  // The delinearization transforms a single-subscript MIV dependence test into
  // a multi-subscript SIV dependence test that is easier to compute. So we
  // resize Pair to contain as many pairs of subscripts as the delinearization
  // has found, and then initialize the pairs following the delinearization.
  Pair.resize(Size);
  SCEVMonotonicityChecker MonChecker(SE);
  const Loop *OutermostLoop = SrcLoop ? SrcLoop->getOutermostLoop() : nullptr;
  for (int I = 0; I < Size; ++I) {
    Pair[I].Src = SrcSubscripts[I];
    Pair[I].Dst = DstSubscripts[I];
    unifySubscriptType(&Pair[I]);

    if (EnableMonotonicityCheck) {
      if (MonChecker.checkMonotonicity(Pair[I].Src, OutermostLoop).isUnknown())
        return false;
      if (MonChecker.checkMonotonicity(Pair[I].Dst, OutermostLoop).isUnknown())
        return false;
    }
  }

  return true;
}

/// Try to delinearize \p SrcAccessFn and \p DstAccessFn if the underlying
/// arrays accessed are fixed-size arrays. Return true if delinearization was
/// successful.
bool DependenceInfo::tryDelinearizeFixedSize(
    Instruction *Src, Instruction *Dst, const SCEV *SrcAccessFn,
    const SCEV *DstAccessFn, SmallVectorImpl<const SCEV *> &SrcSubscripts,
    SmallVectorImpl<const SCEV *> &DstSubscripts) {
  LLVM_DEBUG({
    const SCEVUnknown *SrcBase =
        dyn_cast<SCEVUnknown>(SE->getPointerBase(SrcAccessFn));
    const SCEVUnknown *DstBase =
        dyn_cast<SCEVUnknown>(SE->getPointerBase(DstAccessFn));
    assert(SrcBase && DstBase && SrcBase == DstBase &&
           "expected src and dst scev unknowns to be equal");
  });

  const SCEV *ElemSize = SE->getElementSize(Src);
  assert(ElemSize == SE->getElementSize(Dst) && "Different element sizes");
  SmallVector<const SCEV *, 4> SrcSizes, DstSizes;
  if (!delinearizeFixedSizeArray(*SE, SE->removePointerBase(SrcAccessFn),
                                 SrcSubscripts, SrcSizes, ElemSize) ||
      !delinearizeFixedSizeArray(*SE, SE->removePointerBase(DstAccessFn),
                                 DstSubscripts, DstSizes, ElemSize))
    return false;

  // Check that the two size arrays are non-empty and equal in length and
  // value.  SCEV expressions are uniqued, so we can compare pointers.
  if (SrcSizes.size() != DstSizes.size() ||
      !std::equal(SrcSizes.begin(), SrcSizes.end(), DstSizes.begin())) {
    SrcSubscripts.clear();
    DstSubscripts.clear();
    return false;
  }

  assert(SrcSubscripts.size() == DstSubscripts.size() &&
         "Expected equal number of entries in the list of SrcSubscripts and "
         "DstSubscripts.");

  // In general we cannot safely assume that the subscripts recovered from GEPs
  // are in the range of values defined for their corresponding array
  // dimensions. For example some C language usage/interpretation make it
  // impossible to verify this at compile-time. As such we can only delinearize
  // iff the subscripts are positive and are less than the range of the
  // dimension.
  if (!DisableDelinearizationChecks) {
    if (!validateDelinearizationResult(*SE, SrcSizes, SrcSubscripts) ||
        !validateDelinearizationResult(*SE, DstSizes, DstSubscripts)) {
      SrcSubscripts.clear();
      DstSubscripts.clear();
      return false;
    }
  }
  LLVM_DEBUG({
    dbgs() << "Delinearized subscripts of fixed-size array\n"
           << "SrcGEP:" << *getLoadStorePointerOperand(Src) << "\n"
           << "DstGEP:" << *getLoadStorePointerOperand(Dst) << "\n";
  });
  return true;
}

bool DependenceInfo::tryDelinearizeParametricSize(
    Instruction *Src, Instruction *Dst, const SCEV *SrcAccessFn,
    const SCEV *DstAccessFn, SmallVectorImpl<const SCEV *> &SrcSubscripts,
    SmallVectorImpl<const SCEV *> &DstSubscripts) {

  const SCEVUnknown *SrcBase =
      dyn_cast<SCEVUnknown>(SE->getPointerBase(SrcAccessFn));
  const SCEVUnknown *DstBase =
      dyn_cast<SCEVUnknown>(SE->getPointerBase(DstAccessFn));
  assert(SrcBase && DstBase && SrcBase == DstBase &&
         "expected src and dst scev unknowns to be equal");

  const SCEV *ElementSize = SE->getElementSize(Src);
  if (ElementSize != SE->getElementSize(Dst))
    return false;

  const SCEV *SrcSCEV = SE->getMinusSCEV(SrcAccessFn, SrcBase);
  const SCEV *DstSCEV = SE->getMinusSCEV(DstAccessFn, DstBase);

  const SCEVAddRecExpr *SrcAR = dyn_cast<SCEVAddRecExpr>(SrcSCEV);
  const SCEVAddRecExpr *DstAR = dyn_cast<SCEVAddRecExpr>(DstSCEV);
  if (!SrcAR || !DstAR || !SrcAR->isAffine() || !DstAR->isAffine())
    return false;

  // First step: collect parametric terms in both array references.
  SmallVector<const SCEV *, 4> Terms;
  collectParametricTerms(*SE, SrcAR, Terms);
  collectParametricTerms(*SE, DstAR, Terms);

  // Second step: find subscript sizes.
  SmallVector<const SCEV *, 4> Sizes;
  findArrayDimensions(*SE, Terms, Sizes, ElementSize);

  // Third step: compute the access functions for each subscript.
  computeAccessFunctions(*SE, SrcAR, SrcSubscripts, Sizes);
  computeAccessFunctions(*SE, DstAR, DstSubscripts, Sizes);

  // Fail when there is only a subscript: that's a linearized access function.
  if (SrcSubscripts.size() < 2 || DstSubscripts.size() < 2 ||
      SrcSubscripts.size() != DstSubscripts.size())
    return false;

  // Statically check that the array bounds are in-range. The first subscript we
  // don't have a size for and it cannot overflow into another subscript, so is
  // always safe. The others need to be 0 <= subscript[i] < bound, for both src
  // and dst.
  // FIXME: It may be better to record these sizes and add them as constraints
  // to the dependency checks.
  if (!DisableDelinearizationChecks)
    if (!validateDelinearizationResult(*SE, Sizes, SrcSubscripts) ||
        !validateDelinearizationResult(*SE, Sizes, DstSubscripts))
      return false;

  return true;
}

//===----------------------------------------------------------------------===//

#ifndef NDEBUG
// For debugging purposes, dump a small bit vector to dbgs().
static void dumpSmallBitVector(SmallBitVector &BV) {
  dbgs() << "{";
  for (unsigned VI : BV.set_bits()) {
    dbgs() << VI;
    if (BV.find_next(VI) >= 0)
      dbgs() << ' ';
  }
  dbgs() << "}\n";
}
#endif

bool DependenceInfo::invalidate(Function &F, const PreservedAnalyses &PA,
                                FunctionAnalysisManager::Invalidator &Inv) {
  // Check if the analysis itself has been invalidated.
  auto PAC = PA.getChecker<DependenceAnalysis>();
  if (!PAC.preserved() && !PAC.preservedSet<AllAnalysesOn<Function>>())
    return true;

  // Check transitive dependencies.
  return Inv.invalidate<AAManager>(F, PA) ||
         Inv.invalidate<ScalarEvolutionAnalysis>(F, PA) ||
         Inv.invalidate<LoopAnalysis>(F, PA);
}

// depends -
// Returns NULL if there is no dependence.
// Otherwise, return a Dependence with as many details as possible.
// Corresponds to Section 3.1 in the paper
//
//            Practical Dependence Testing
//            Goff, Kennedy, Tseng
//            PLDI 1991
//
std::unique_ptr<Dependence>
DependenceInfo::depends(Instruction *Src, Instruction *Dst,
                        bool UnderRuntimeAssumptions) {
  SmallVector<const SCEVPredicate *, 4> Assume;
  bool PossiblyLoopIndependent = true;
  if (Src == Dst)
    PossiblyLoopIndependent = false;

  if (!(Src->mayReadOrWriteMemory() && Dst->mayReadOrWriteMemory()))
    // if both instructions don't reference memory, there's no dependence
    return nullptr;

  if (!isLoadOrStore(Src) || !isLoadOrStore(Dst)) {
    // can only analyze simple loads and stores, i.e., no calls, invokes, etc.
    LLVM_DEBUG(dbgs() << "can only handle simple loads and stores\n");
    return std::make_unique<Dependence>(Src, Dst,
                                        SCEVUnionPredicate(Assume, *SE));
  }

  const MemoryLocation &DstLoc = MemoryLocation::get(Dst);
  const MemoryLocation &SrcLoc = MemoryLocation::get(Src);

  switch (underlyingObjectsAlias(AA, F->getDataLayout(), DstLoc, SrcLoc)) {
  case AliasResult::MayAlias:
  case AliasResult::PartialAlias:
    // cannot analyse objects if we don't understand their aliasing.
    LLVM_DEBUG(dbgs() << "can't analyze may or partial alias\n");
    return std::make_unique<Dependence>(Src, Dst,
                                        SCEVUnionPredicate(Assume, *SE));
  case AliasResult::NoAlias:
    // If the objects noalias, they are distinct, accesses are independent.
    LLVM_DEBUG(dbgs() << "no alias\n");
    return nullptr;
  case AliasResult::MustAlias:
    break; // The underlying objects alias; test accesses for dependence.
  }

  if (DstLoc.Size != SrcLoc.Size || !DstLoc.Size.isPrecise() ||
      !SrcLoc.Size.isPrecise()) {
    // The dependence test gets confused if the size of the memory accesses
    // differ.
    LLVM_DEBUG(dbgs() << "can't analyze must alias with different sizes\n");
    return std::make_unique<Dependence>(Src, Dst,
                                        SCEVUnionPredicate(Assume, *SE));
  }

  Value *SrcPtr = getLoadStorePointerOperand(Src);
  Value *DstPtr = getLoadStorePointerOperand(Dst);
  const SCEV *SrcSCEV = SE->getSCEV(SrcPtr);
  const SCEV *DstSCEV = SE->getSCEV(DstPtr);
  LLVM_DEBUG(dbgs() << "    SrcSCEV = " << *SrcSCEV << "\n");
  LLVM_DEBUG(dbgs() << "    DstSCEV = " << *DstSCEV << "\n");
  const SCEV *SrcBase = SE->getPointerBase(SrcSCEV);
  const SCEV *DstBase = SE->getPointerBase(DstSCEV);
  if (SrcBase != DstBase) {
    // If two pointers have different bases, trying to analyze indexes won't
    // work; we can't compare them to each other. This can happen, for example,
    // if one is produced by an LCSSA PHI node.
    //
    // We check this upfront so we don't crash in cases where getMinusSCEV()
    // returns a SCEVCouldNotCompute.
    LLVM_DEBUG(dbgs() << "can't analyze SCEV with different pointer base\n");
    return std::make_unique<Dependence>(Src, Dst,
                                        SCEVUnionPredicate(Assume, *SE));
  }

  // Even if the base pointers are the same, they may not be loop-invariant. It
  // could lead to incorrect results, as we're analyzing loop-carried
  // dependencies. Src and Dst can be in different loops, so we need to check
  // the base pointer is invariant in both loops.
  Loop *SrcLoop = LI->getLoopFor(Src->getParent());
  Loop *DstLoop = LI->getLoopFor(Dst->getParent());
  if (!isLoopInvariant(SrcBase, SrcLoop) ||
      !isLoopInvariant(DstBase, DstLoop)) {
    LLVM_DEBUG(dbgs() << "The base pointer is not loop invariant.\n");
    return std::make_unique<Dependence>(Src, Dst,
                                        SCEVUnionPredicate(Assume, *SE));
  }

  uint64_t EltSize = SrcLoc.Size.toRaw();
  const SCEV *SrcEv = SE->getMinusSCEV(SrcSCEV, SrcBase);
  const SCEV *DstEv = SE->getMinusSCEV(DstSCEV, DstBase);

  // Check that memory access offsets are multiples of element sizes.
  if (!SE->isKnownMultipleOf(SrcEv, EltSize, Assume) ||
      !SE->isKnownMultipleOf(DstEv, EltSize, Assume)) {
    LLVM_DEBUG(dbgs() << "can't analyze SCEV with different offsets\n");
    return std::make_unique<Dependence>(Src, Dst,
                                        SCEVUnionPredicate(Assume, *SE));
  }

  // Runtime assumptions needed but not allowed.
  if (!Assume.empty() && !UnderRuntimeAssumptions)
    return std::make_unique<Dependence>(Src, Dst,
                                        SCEVUnionPredicate(Assume, *SE));

  unsigned Pairs = 1;
  SmallVector<Subscript, 2> Pair(Pairs);
  Pair[0].Src = SrcEv;
  Pair[0].Dst = DstEv;

  SCEVMonotonicityChecker MonChecker(SE);
  const Loop *OutermostLoop = SrcLoop ? SrcLoop->getOutermostLoop() : nullptr;
  if (EnableMonotonicityCheck)
    if (MonChecker.checkMonotonicity(Pair[0].Src, OutermostLoop).isUnknown() ||
        MonChecker.checkMonotonicity(Pair[0].Dst, OutermostLoop).isUnknown())
      return std::make_unique<Dependence>(Src, Dst,
                                          SCEVUnionPredicate(Assume, *SE));

  if (Delinearize) {
    if (tryDelinearize(Src, Dst, Pair)) {
      LLVM_DEBUG(dbgs() << "    delinearized\n");
      Pairs = Pair.size();
    }
  }

  // Establish loop nesting levels considering SameSD loops as common
  establishNestingLevels(Src, Dst);

  LLVM_DEBUG(dbgs() << "    common nesting levels = " << CommonLevels << "\n");
  LLVM_DEBUG(dbgs() << "    maximum nesting levels = " << MaxLevels << "\n");
  LLVM_DEBUG(dbgs() << "    SameSD nesting levels = " << SameSDLevels << "\n");

  // Modify common levels to consider the SameSD levels in the tests
  CommonLevels += SameSDLevels;
  MaxLevels -= SameSDLevels;
  if (SameSDLevels > 0) {
    // Not all tests are handled yet over SameSD loops
    // Revoke if there are any tests other than ZIV, SIV or RDIV
    for (unsigned P = 0; P < Pairs; ++P) {
      SmallBitVector Loops;
      Subscript::ClassificationKind TestClass =
          classifyPair(Pair[P].Src, LI->getLoopFor(Src->getParent()),
                       Pair[P].Dst, LI->getLoopFor(Dst->getParent()), Loops);

      if (TestClass != Subscript::ZIV && TestClass != Subscript::SIV &&
          TestClass != Subscript::RDIV) {
        // Revert the levels to not consider the SameSD levels
        CommonLevels -= SameSDLevels;
        MaxLevels += SameSDLevels;
        SameSDLevels = 0;
        break;
      }
    }
  }

  if (SameSDLevels > 0)
    SameSDLoopsCount++;

  FullDependence Result(Src, Dst, SCEVUnionPredicate(Assume, *SE),
                        PossiblyLoopIndependent, CommonLevels);
  ++TotalArrayPairs;

  for (unsigned P = 0; P < Pairs; ++P) {
    assert(Pair[P].Src->getType()->isIntegerTy() && "Src must be an integer");
    assert(Pair[P].Dst->getType()->isIntegerTy() && "Dst must be an integer");
    Pair[P].Loops.resize(MaxLevels + 1);
    Pair[P].GroupLoops.resize(MaxLevels + 1);
    Pair[P].Group.resize(Pairs);
    removeMatchingExtensions(&Pair[P]);
    Pair[P].Classification =
        classifyPair(Pair[P].Src, LI->getLoopFor(Src->getParent()), Pair[P].Dst,
                     LI->getLoopFor(Dst->getParent()), Pair[P].Loops);
    Pair[P].GroupLoops = Pair[P].Loops;
    Pair[P].Group.set(P);
    LLVM_DEBUG(dbgs() << "    subscript " << P << "\n");
    LLVM_DEBUG(dbgs() << "\tsrc = " << *Pair[P].Src << "\n");
    LLVM_DEBUG(dbgs() << "\tdst = " << *Pair[P].Dst << "\n");
    LLVM_DEBUG(dbgs() << "\tclass = " << Pair[P].Classification << "\n");
    LLVM_DEBUG(dbgs() << "\tloops = ");
    LLVM_DEBUG(dumpSmallBitVector(Pair[P].Loops));
  }

  // Test each subscript individually
  for (unsigned SI = 0; SI < Pairs; ++SI) {
    LLVM_DEBUG(dbgs() << "testing subscript " << SI);
    switch (Pair[SI].Classification) {
    case Subscript::NonLinear:
      // ignore these, but collect loops for later
      ++NonlinearSubscriptPairs;
      collectCommonLoops(Pair[SI].Src, LI->getLoopFor(Src->getParent()),
                         Pair[SI].Loops);
      collectCommonLoops(Pair[SI].Dst, LI->getLoopFor(Dst->getParent()),
                         Pair[SI].Loops);
      Result.Consistent = false;
      break;
    case Subscript::ZIV:
      LLVM_DEBUG(dbgs() << ", ZIV\n");
      if (testZIV(Pair[SI].Src, Pair[SI].Dst, Result))
        return nullptr;
      break;
    case Subscript::SIV: {
      LLVM_DEBUG(dbgs() << ", SIV\n");
      unsigned Level;
      if (testSIV(Pair[SI].Src, Pair[SI].Dst, Level, Result,
                  UnderRuntimeAssumptions))
        return nullptr;
      break;
    }
    case Subscript::RDIV:
      LLVM_DEBUG(dbgs() << ", RDIV\n");
      if (testRDIV(Pair[SI].Src, Pair[SI].Dst, Result))
        return nullptr;
      break;
    case Subscript::MIV:
      LLVM_DEBUG(dbgs() << ", MIV\n");
      if (testMIV(Pair[SI].Src, Pair[SI].Dst, Pair[SI].Loops, Result))
        return nullptr;
      break;
    }
  }

  // Make sure the Scalar flags are set correctly.
  SmallBitVector CompleteLoops(MaxLevels + 1);
  for (unsigned SI = 0; SI < Pairs; ++SI)
    CompleteLoops |= Pair[SI].Loops;
  for (unsigned II = 1; II <= CommonLevels; ++II)
    if (CompleteLoops[II])
      Result.DV[II - 1].Scalar = false;

  // Set the distance to zero if the direction is EQ.
  // TODO: Ideally, the distance should be set to 0 immediately simultaneously
  // with the corresponding direction being set to EQ.
  for (unsigned II = 1; II <= Result.getLevels(); ++II) {
    if (Result.getDirection(II) == Dependence::DVEntry::EQ) {
      if (Result.DV[II - 1].Distance == nullptr)
        Result.DV[II - 1].Distance = SE->getZero(SrcSCEV->getType());
      else
        assert(Result.DV[II - 1].Distance->isZero() &&
               "Inconsistency between distance and direction");
    }

#ifndef NDEBUG
    // Check that the converse (i.e., if the distance is zero, then the
    // direction is EQ) holds.
    const SCEV *Distance = Result.getDistance(II);
    if (Distance && Distance->isZero())
      assert(Result.getDirection(II) == Dependence::DVEntry::EQ &&
             "Distance is zero, but direction is not EQ");
#endif
  }

  if (SameSDLevels > 0) {
    // Extracting SameSD levels from the common levels
    // Reverting CommonLevels and MaxLevels to their original values
    assert(CommonLevels >= SameSDLevels);
    CommonLevels -= SameSDLevels;
    MaxLevels += SameSDLevels;
    std::unique_ptr<FullDependence::DVEntry[]> DV, DVSameSD;
    DV = std::make_unique<FullDependence::DVEntry[]>(CommonLevels);
    DVSameSD = std::make_unique<FullDependence::DVEntry[]>(SameSDLevels);
    for (unsigned Level = 0; Level < CommonLevels; ++Level)
      DV[Level] = Result.DV[Level];
    for (unsigned Level = 0; Level < SameSDLevels; ++Level)
      DVSameSD[Level] = Result.DV[CommonLevels + Level];
    Result.DV = std::move(DV);
    Result.DVSameSD = std::move(DVSameSD);
    Result.Levels = CommonLevels;
    Result.SameSDLevels = SameSDLevels;
    // Result is not consistent if it considers SameSD levels
    Result.Consistent = false;
  }

  if (PossiblyLoopIndependent) {
    // Make sure the LoopIndependent flag is set correctly.
    // All directions must include equal, otherwise no
    // loop-independent dependence is possible.
    for (unsigned II = 1; II <= CommonLevels; ++II) {
      if (!(Result.getDirection(II) & Dependence::DVEntry::EQ)) {
        Result.LoopIndependent = false;
        break;
      }
    }
  } else {
    // On the other hand, if all directions are equal and there's no
    // loop-independent dependence possible, then no dependence exists.
    // However, if there are runtime assumptions, we must return the result.
    bool AllEqual = true;
    for (unsigned II = 1; II <= CommonLevels; ++II) {
      if (Result.getDirection(II) != Dependence::DVEntry::EQ) {
        AllEqual = false;
        break;
      }
    }
    if (AllEqual && Result.Assumptions.getPredicates().empty())
      return nullptr;
  }

  return std::make_unique<FullDependence>(std::move(Result));
}
