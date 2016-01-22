//===----- ScopDetection.cpp  - Detect Scops --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Detect the maximal Scops of a function.
//
// A static control part (Scop) is a subgraph of the control flow graph (CFG)
// that only has statically known control flow and can therefore be described
// within the polyhedral model.
//
// Every Scop fullfills these restrictions:
//
// * It is a single entry single exit region
//
// * Only affine linear bounds in the loops
//
// Every natural loop in a Scop must have a number of loop iterations that can
// be described as an affine linear function in surrounding loop iterators or
// parameters. (A parameter is a scalar that does not change its value during
// execution of the Scop).
//
// * Only comparisons of affine linear expressions in conditions
//
// * All loops and conditions perfectly nested
//
// The control flow needs to be structured such that it could be written using
// just 'for' and 'if' statements, without the need for any 'goto', 'break' or
// 'continue'.
//
// * Side effect free functions call
//
// Only function calls and intrinsics that do not have side effects are allowed
// (readnone).
//
// The Scop detection finds the largest Scops by checking if the largest
// region is a Scop. If this is not the case, its canonical subregions are
// checked until a region is a Scop. It is now tried to extend this Scop by
// creating a larger non canonical region.
//
//===----------------------------------------------------------------------===//

#include "polly/ScopDetection.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopDetectionDiagnostic.h"
#include "polly/Support/SCEVValidator.h"
#include "polly/Support/ScopLocation.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Debug.h"
#include <set>
#include <stack>

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-detect"

// This option is set to a very high value, as analyzing such loops increases
// compile time on several cases. For experiments that enable this option,
// a value of around 40 has been working to avoid run-time regressions with
// Polly while still exposing interesting optimization opportunities.
static cl::opt<int> ProfitabilityMinPerLoopInstructions(
    "polly-detect-profitability-min-per-loop-insts",
    cl::desc("The minimal number of per-loop instructions before a single loop "
             "region is considered profitable"),
    cl::Hidden, cl::ValueRequired, cl::init(100000000), cl::cat(PollyCategory));

bool polly::PollyProcessUnprofitable;
static cl::opt<bool, true> XPollyProcessUnprofitable(
    "polly-process-unprofitable",
    cl::desc(
        "Process scops that are unlikely to benefit from Polly optimizations."),
    cl::location(PollyProcessUnprofitable), cl::init(false), cl::ZeroOrMore,
    cl::cat(PollyCategory));

static cl::opt<std::string> OnlyFunction(
    "polly-only-func",
    cl::desc("Only run on functions that contain a certain string"),
    cl::value_desc("string"), cl::ValueRequired, cl::init(""),
    cl::cat(PollyCategory));

static cl::opt<std::string> OnlyRegion(
    "polly-only-region",
    cl::desc("Only run on certain regions (The provided identifier must "
             "appear in the name of the region's entry block"),
    cl::value_desc("identifier"), cl::ValueRequired, cl::init(""),
    cl::cat(PollyCategory));

static cl::opt<bool>
    IgnoreAliasing("polly-ignore-aliasing",
                   cl::desc("Ignore possible aliasing of the array bases"),
                   cl::Hidden, cl::init(false), cl::ZeroOrMore,
                   cl::cat(PollyCategory));

bool polly::PollyUseRuntimeAliasChecks;
static cl::opt<bool, true> XPollyUseRuntimeAliasChecks(
    "polly-use-runtime-alias-checks",
    cl::desc("Use runtime alias checks to resolve possible aliasing."),
    cl::location(PollyUseRuntimeAliasChecks), cl::Hidden, cl::ZeroOrMore,
    cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool>
    ReportLevel("polly-report",
                cl::desc("Print information about the activities of Polly"),
                cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool>
    AllowNonAffine("polly-allow-nonaffine",
                   cl::desc("Allow non affine access functions in arrays"),
                   cl::Hidden, cl::init(false), cl::ZeroOrMore,
                   cl::cat(PollyCategory));

static cl::opt<bool> AllowNonAffineSubRegions(
    "polly-allow-nonaffine-branches",
    cl::desc("Allow non affine conditions for branches"), cl::Hidden,
    cl::init(true), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool>
    AllowNonAffineSubLoops("polly-allow-nonaffine-loops",
                           cl::desc("Allow non affine conditions for loops"),
                           cl::Hidden, cl::init(false), cl::ZeroOrMore,
                           cl::cat(PollyCategory));

static cl::opt<bool> AllowUnsigned("polly-allow-unsigned",
                                   cl::desc("Allow unsigned expressions"),
                                   cl::Hidden, cl::init(false), cl::ZeroOrMore,
                                   cl::cat(PollyCategory));

static cl::opt<bool, true>
    TrackFailures("polly-detect-track-failures",
                  cl::desc("Track failure strings in detecting scop regions"),
                  cl::location(PollyTrackFailures), cl::Hidden, cl::ZeroOrMore,
                  cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool> KeepGoing("polly-detect-keep-going",
                               cl::desc("Do not fail on the first error."),
                               cl::Hidden, cl::ZeroOrMore, cl::init(false),
                               cl::cat(PollyCategory));

static cl::opt<bool, true>
    PollyDelinearizeX("polly-delinearize",
                      cl::desc("Delinearize array access functions"),
                      cl::location(PollyDelinearize), cl::Hidden,
                      cl::ZeroOrMore, cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool>
    VerifyScops("polly-detect-verify",
                cl::desc("Verify the detected SCoPs after each transformation"),
                cl::Hidden, cl::init(false), cl::ZeroOrMore,
                cl::cat(PollyCategory));

/// @brief The minimal trip count under which loops are considered unprofitable.
static const unsigned MIN_LOOP_TRIP_COUNT = 8;

bool polly::PollyTrackFailures = false;
bool polly::PollyDelinearize = false;
StringRef polly::PollySkipFnAttr = "polly.skip.fn";

//===----------------------------------------------------------------------===//
// Statistics.

STATISTIC(ValidRegion, "Number of regions that a valid part of Scop");

class DiagnosticScopFound : public DiagnosticInfo {
private:
  static int PluginDiagnosticKind;

  Function &F;
  std::string FileName;
  unsigned EntryLine, ExitLine;

public:
  DiagnosticScopFound(Function &F, std::string FileName, unsigned EntryLine,
                      unsigned ExitLine)
      : DiagnosticInfo(PluginDiagnosticKind, DS_Note), F(F), FileName(FileName),
        EntryLine(EntryLine), ExitLine(ExitLine) {}

  virtual void print(DiagnosticPrinter &DP) const;

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == PluginDiagnosticKind;
  }
};

int DiagnosticScopFound::PluginDiagnosticKind = 10;

void DiagnosticScopFound::print(DiagnosticPrinter &DP) const {
  DP << "Polly detected an optimizable loop region (scop) in function '" << F
     << "'\n";

  if (FileName.empty()) {
    DP << "Scop location is unknown. Compile with debug info "
          "(-g) to get more precise information. ";
    return;
  }

  DP << FileName << ":" << EntryLine << ": Start of scop\n";
  DP << FileName << ":" << ExitLine << ": End of scop";
}

//===----------------------------------------------------------------------===//
// ScopDetection.

ScopDetection::ScopDetection() : FunctionPass(ID) {
  if (!PollyUseRuntimeAliasChecks)
    return;

  // Disable runtime alias checks if we ignore aliasing all together.
  if (IgnoreAliasing) {
    PollyUseRuntimeAliasChecks = false;
    return;
  }

  if (AllowNonAffine) {
    DEBUG(errs() << "WARNING: We disable runtime alias checks as non affine "
                    "accesses are enabled.\n");
    PollyUseRuntimeAliasChecks = false;
  }
}

template <class RR, typename... Args>
inline bool ScopDetection::invalid(DetectionContext &Context, bool Assert,
                                   Args &&... Arguments) const {

  if (!Context.Verifying) {
    RejectLog &Log = Context.Log;
    std::shared_ptr<RR> RejectReason = std::make_shared<RR>(Arguments...);

    if (PollyTrackFailures)
      Log.report(RejectReason);

    DEBUG(dbgs() << RejectReason->getMessage());
    DEBUG(dbgs() << "\n");
  } else {
    assert(!Assert && "Verification of detected scop failed");
  }

  return false;
}

bool ScopDetection::isMaxRegionInScop(const Region &R, bool Verify) const {
  if (!ValidRegions.count(&R))
    return false;

  if (Verify) {
    DetectionContextMap.erase(&R);
    const auto &It = DetectionContextMap.insert(
        std::make_pair(&R, DetectionContext(const_cast<Region &>(R), *AA,
                                            false /*verifying*/)));
    DetectionContext &Context = It.first->second;
    return isValidRegion(Context);
  }

  return true;
}

std::string ScopDetection::regionIsInvalidBecause(const Region *R) const {
  if (!RejectLogs.count(R))
    return "";

  // Get the first error we found. Even in keep-going mode, this is the first
  // reason that caused the candidate to be rejected.
  RejectLog Errors = RejectLogs.at(R);

  // This can happen when we marked a region invalid, but didn't track
  // an error for it.
  if (Errors.size() == 0)
    return "";

  RejectReasonPtr RR = *Errors.begin();
  return RR->getMessage();
}

bool ScopDetection::addOverApproximatedRegion(Region *AR,
                                              DetectionContext &Context) const {

  // If we already know about Ar we can exit.
  if (!Context.NonAffineSubRegionSet.insert(AR))
    return true;

  // All loops in the region have to be overapproximated too if there
  // are accesses that depend on the iteration count.
  for (BasicBlock *BB : AR->blocks()) {
    Loop *L = LI->getLoopFor(BB);
    if (AR->contains(L))
      Context.BoxedLoopsSet.insert(L);
  }

  return (AllowNonAffineSubLoops || Context.BoxedLoopsSet.empty());
}

bool ScopDetection::onlyValidRequiredInvariantLoads(
    InvariantLoadsSetTy &RequiredILS, DetectionContext &Context) const {
  Region &CurRegion = Context.CurRegion;

  for (LoadInst *Load : RequiredILS)
    if (!isHoistableLoad(Load, CurRegion, *LI, *SE))
      return false;

  Context.RequiredILS.insert(RequiredILS.begin(), RequiredILS.end());

  return true;
}

bool ScopDetection::isAffine(const SCEV *S, DetectionContext &Context,
                             Value *BaseAddress) const {

  InvariantLoadsSetTy AccessILS;
  if (!isAffineExpr(&Context.CurRegion, S, *SE, BaseAddress, &AccessILS))
    return false;

  if (!onlyValidRequiredInvariantLoads(AccessILS, Context))
    return false;

  return true;
}

bool ScopDetection::isValidSwitch(BasicBlock &BB, SwitchInst *SI,
                                  Value *Condition, bool IsLoopBranch,
                                  DetectionContext &Context) const {
  Loop *L = LI->getLoopFor(&BB);
  const SCEV *ConditionSCEV = SE->getSCEVAtScope(Condition, L);

  if (isAffine(ConditionSCEV, Context))
    return true;

  if (!IsLoopBranch && AllowNonAffineSubRegions &&
      addOverApproximatedRegion(RI->getRegionFor(&BB), Context))
    return true;

  if (IsLoopBranch)
    return false;

  return invalid<ReportNonAffBranch>(Context, /*Assert=*/true, &BB,
                                     ConditionSCEV, ConditionSCEV, SI);
}

bool ScopDetection::isValidBranch(BasicBlock &BB, BranchInst *BI,
                                  Value *Condition, bool IsLoopBranch,
                                  DetectionContext &Context) const {

  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(Condition)) {
    auto Opcode = BinOp->getOpcode();
    if (Opcode == Instruction::And || Opcode == Instruction::Or) {
      Value *Op0 = BinOp->getOperand(0);
      Value *Op1 = BinOp->getOperand(1);
      return isValidBranch(BB, BI, Op0, IsLoopBranch, Context) &&
             isValidBranch(BB, BI, Op1, IsLoopBranch, Context);
    }
  }

  // Non constant conditions of branches need to be ICmpInst.
  if (!isa<ICmpInst>(Condition)) {
    if (!IsLoopBranch && AllowNonAffineSubRegions &&
        addOverApproximatedRegion(RI->getRegionFor(&BB), Context))
      return true;
    return invalid<ReportInvalidCond>(Context, /*Assert=*/true, BI, &BB);
  }

  ICmpInst *ICmp = cast<ICmpInst>(Condition);
  // Unsigned comparisons are not allowed. They trigger overflow problems
  // in the code generation.
  //
  // TODO: This is not sufficient and just hides bugs. However it does pretty
  //       well.
  if (ICmp->isUnsigned() && !AllowUnsigned)
    return invalid<ReportUnsignedCond>(Context, /*Assert=*/true, BI, &BB);

  // Are both operands of the ICmp affine?
  if (isa<UndefValue>(ICmp->getOperand(0)) ||
      isa<UndefValue>(ICmp->getOperand(1)))
    return invalid<ReportUndefOperand>(Context, /*Assert=*/true, &BB, ICmp);

  // TODO: FIXME: IslExprBuilder is not capable of producing valid code
  //              for arbitrary pointer expressions at the moment. Until
  //              this is fixed we disallow pointer expressions completely.
  if (ICmp->getOperand(0)->getType()->isPointerTy())
    return false;

  Loop *L = LI->getLoopFor(ICmp->getParent());
  const SCEV *LHS = SE->getSCEVAtScope(ICmp->getOperand(0), L);
  const SCEV *RHS = SE->getSCEVAtScope(ICmp->getOperand(1), L);

  if (isAffine(LHS, Context) && isAffine(RHS, Context))
    return true;

  if (!IsLoopBranch && AllowNonAffineSubRegions &&
      addOverApproximatedRegion(RI->getRegionFor(&BB), Context))
    return true;

  if (IsLoopBranch)
    return false;

  return invalid<ReportNonAffBranch>(Context, /*Assert=*/true, &BB, LHS, RHS,
                                     ICmp);
}

bool ScopDetection::isValidCFG(BasicBlock &BB, bool IsLoopBranch,
                               bool AllowUnreachable,
                               DetectionContext &Context) const {
  Region &CurRegion = Context.CurRegion;

  TerminatorInst *TI = BB.getTerminator();

  if (AllowUnreachable && isa<UnreachableInst>(TI))
    return true;

  // Return instructions are only valid if the region is the top level region.
  if (isa<ReturnInst>(TI) && !CurRegion.getExit() && TI->getNumOperands() == 0)
    return true;

  Value *Condition = getConditionFromTerminator(TI);

  if (!Condition)
    return invalid<ReportInvalidTerminator>(Context, /*Assert=*/true, &BB);

  // UndefValue is not allowed as condition.
  if (isa<UndefValue>(Condition))
    return invalid<ReportUndefCond>(Context, /*Assert=*/true, TI, &BB);

  // Constant integer conditions are always affine.
  if (isa<ConstantInt>(Condition))
    return true;

  if (BranchInst *BI = dyn_cast<BranchInst>(TI))
    return isValidBranch(BB, BI, Condition, IsLoopBranch, Context);

  SwitchInst *SI = dyn_cast<SwitchInst>(TI);
  assert(SI && "Terminator was neither branch nor switch");

  return isValidSwitch(BB, SI, Condition, IsLoopBranch, Context);
}

bool ScopDetection::isValidCallInst(CallInst &CI) {
  if (CI.doesNotReturn())
    return false;

  if (CI.doesNotAccessMemory())
    return true;

  Function *CalledFunction = CI.getCalledFunction();

  // Indirect calls are not supported.
  if (CalledFunction == 0)
    return false;

  if (isIgnoredIntrinsic(&CI))
    return true;

  return false;
}

bool ScopDetection::isInvariant(const Value &Val, const Region &Reg) const {
  // A reference to function argument or constant value is invariant.
  if (isa<Argument>(Val) || isa<Constant>(Val))
    return true;

  const Instruction *I = dyn_cast<Instruction>(&Val);
  if (!I)
    return false;

  if (!Reg.contains(I))
    return true;

  if (I->mayHaveSideEffects())
    return false;

  // When Val is a Phi node, it is likely not invariant. We do not check whether
  // Phi nodes are actually invariant, we assume that Phi nodes are usually not
  // invariant. Recursively checking the operators of Phi nodes would lead to
  // infinite recursion.
  if (isa<PHINode>(*I))
    return false;

  for (const Use &Operand : I->operands())
    if (!isInvariant(*Operand, Reg))
      return false;

  return true;
}

MapInsnToMemAcc InsnToMemAcc;

/// @brief Remove smax of smax(0, size) expressions from a SCEV expression and
/// register the '...' components.
///
/// Array access expressions as they are generated by gfortran contain smax(0,
/// size) expressions that confuse the 'normal' delinearization algorithm.
/// However, if we extract such expressions before the normal delinearization
/// takes place they can actually help to identify array size expressions in
/// fortran accesses. For the subsequently following delinearization the smax(0,
/// size) component can be replaced by just 'size'. This is correct as we will
/// always add and verify the assumption that for all subscript expressions
/// 'exp' the inequality 0 <= exp < size holds. Hence, we will also verify
/// that 0 <= size, which means smax(0, size) == size.
struct SCEVRemoveMax : public SCEVVisitor<SCEVRemoveMax, const SCEV *> {
public:
  static const SCEV *remove(ScalarEvolution &SE, const SCEV *Expr,
                            std::vector<const SCEV *> *Terms = nullptr) {

    SCEVRemoveMax D(SE, Terms);
    return D.visit(Expr);
  }

  SCEVRemoveMax(ScalarEvolution &SE, std::vector<const SCEV *> *Terms)
      : SE(SE), Terms(Terms) {}

  const SCEV *visitTruncateExpr(const SCEVTruncateExpr *Expr) { return Expr; }

  const SCEV *visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
    return Expr;
  }

  const SCEV *visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
    return SE.getSignExtendExpr(visit(Expr->getOperand()), Expr->getType());
  }

  const SCEV *visitUDivExpr(const SCEVUDivExpr *Expr) { return Expr; }

  const SCEV *visitSMaxExpr(const SCEVSMaxExpr *Expr) {
    if ((Expr->getNumOperands() == 2) && Expr->getOperand(0)->isZero()) {
      auto Res = visit(Expr->getOperand(1));
      if (Terms)
        (*Terms).push_back(Res);
      return Res;
    }

    return Expr;
  }

  const SCEV *visitUMaxExpr(const SCEVUMaxExpr *Expr) { return Expr; }

  const SCEV *visitUnknown(const SCEVUnknown *Expr) { return Expr; }

  const SCEV *visitCouldNotCompute(const SCEVCouldNotCompute *Expr) {
    return Expr;
  }

  const SCEV *visitConstant(const SCEVConstant *Expr) { return Expr; }

  const SCEV *visitAddRecExpr(const SCEVAddRecExpr *Expr) {
    SmallVector<const SCEV *, 5> NewOps;
    for (const SCEV *Op : Expr->operands())
      NewOps.push_back(visit(Op));

    return SE.getAddRecExpr(NewOps, Expr->getLoop(), Expr->getNoWrapFlags());
  }

  const SCEV *visitAddExpr(const SCEVAddExpr *Expr) {
    SmallVector<const SCEV *, 5> NewOps;
    for (const SCEV *Op : Expr->operands())
      NewOps.push_back(visit(Op));

    return SE.getAddExpr(NewOps);
  }

  const SCEV *visitMulExpr(const SCEVMulExpr *Expr) {
    SmallVector<const SCEV *, 5> NewOps;
    for (const SCEV *Op : Expr->operands())
      NewOps.push_back(visit(Op));

    return SE.getMulExpr(NewOps);
  }

private:
  ScalarEvolution &SE;
  std::vector<const SCEV *> *Terms;
};

SmallVector<const SCEV *, 4>
ScopDetection::getDelinearizationTerms(DetectionContext &Context,
                                       const SCEVUnknown *BasePointer) const {
  SmallVector<const SCEV *, 4> Terms;
  for (const auto &Pair : Context.Accesses[BasePointer]) {
    std::vector<const SCEV *> MaxTerms;
    SCEVRemoveMax::remove(*SE, Pair.second, &MaxTerms);
    if (MaxTerms.size() > 0) {
      Terms.insert(Terms.begin(), MaxTerms.begin(), MaxTerms.end());
      continue;
    }
    // In case the outermost expression is a plain add, we check if any of its
    // terms has the form 4 * %inst * %param * %param ..., aka a term that
    // contains a product between a parameter and an instruction that is
    // inside the scop. Such instructions, if allowed at all, are instructions
    // SCEV can not represent, but Polly is still looking through. As a
    // result, these instructions can depend on induction variables and are
    // most likely no array sizes. However, terms that are multiplied with
    // them are likely candidates for array sizes.
    if (auto *AF = dyn_cast<SCEVAddExpr>(Pair.second)) {
      for (auto Op : AF->operands()) {
        if (auto *AF2 = dyn_cast<SCEVAddRecExpr>(Op))
          SE->collectParametricTerms(AF2, Terms);
        if (auto *AF2 = dyn_cast<SCEVMulExpr>(Op)) {
          SmallVector<const SCEV *, 0> Operands;

          for (auto *MulOp : AF2->operands()) {
            if (auto *Const = dyn_cast<SCEVConstant>(MulOp))
              Operands.push_back(Const);
            if (auto *Unknown = dyn_cast<SCEVUnknown>(MulOp)) {
              if (auto *Inst = dyn_cast<Instruction>(Unknown->getValue())) {
                if (!Context.CurRegion.contains(Inst))
                  Operands.push_back(MulOp);

              } else {
                Operands.push_back(MulOp);
              }
            }
          }
          if (Operands.size())
            Terms.push_back(SE->getMulExpr(Operands));
        }
      }
    }
    if (Terms.empty())
      SE->collectParametricTerms(Pair.second, Terms);
  }
  return Terms;
}

bool ScopDetection::hasValidArraySizes(DetectionContext &Context,
                                       SmallVectorImpl<const SCEV *> &Sizes,
                                       const SCEVUnknown *BasePointer) const {
  Value *BaseValue = BasePointer->getValue();
  Region &CurRegion = Context.CurRegion;
  for (const SCEV *DelinearizedSize : Sizes) {
    if (!isAffine(DelinearizedSize, Context, nullptr)) {
      Sizes.clear();
      break;
    }
    if (auto *Unknown = dyn_cast<SCEVUnknown>(DelinearizedSize)) {
      auto *V = dyn_cast<Value>(Unknown->getValue());
      if (auto *Load = dyn_cast<LoadInst>(V)) {
        if (Context.CurRegion.contains(Load) &&
            isHoistableLoad(Load, CurRegion, *LI, *SE))
          Context.RequiredILS.insert(Load);
        continue;
      }
    }
    if (hasScalarDepsInsideRegion(DelinearizedSize, &CurRegion))
      return invalid<ReportNonAffineAccess>(
          Context, /*Assert=*/true, DelinearizedSize,
          Context.Accesses[BasePointer].front().first, BaseValue);
  }

  // No array shape derived.
  if (Sizes.empty()) {
    if (AllowNonAffine)
      return true;

    for (const auto &Pair : Context.Accesses[BasePointer]) {
      const Instruction *Insn = Pair.first;
      const SCEV *AF = Pair.second;

      if (!isAffine(AF, Context, BaseValue)) {
        invalid<ReportNonAffineAccess>(Context, /*Assert=*/true, AF, Insn,
                                       BaseValue);
        if (!KeepGoing)
          return false;
      }
    }
    return false;
  }
  return true;
}

// We first store the resulting memory accesses in TempMemoryAccesses. Only
// if the access functions for all memory accesses have been successfully
// delinearized we continue. Otherwise, we either report a failure or, if
// non-affine accesses are allowed, we drop the information. In case the
// information is dropped the memory accesses need to be overapproximated
// when translated to a polyhedral representation.
bool ScopDetection::computeAccessFunctions(
    DetectionContext &Context, const SCEVUnknown *BasePointer,
    std::shared_ptr<ArrayShape> Shape) const {
  Value *BaseValue = BasePointer->getValue();
  bool BasePtrHasNonAffine = false;
  MapInsnToMemAcc TempMemoryAccesses;
  for (const auto &Pair : Context.Accesses[BasePointer]) {
    const Instruction *Insn = Pair.first;
    auto *AF = Pair.second;
    AF = SCEVRemoveMax::remove(*SE, AF);
    bool IsNonAffine = false;
    TempMemoryAccesses.insert(std::make_pair(Insn, MemAcc(Insn, Shape)));
    MemAcc *Acc = &TempMemoryAccesses.find(Insn)->second;

    if (!AF) {
      if (isAffine(Pair.second, Context, BaseValue))
        Acc->DelinearizedSubscripts.push_back(Pair.second);
      else
        IsNonAffine = true;
    } else {
      SE->computeAccessFunctions(AF, Acc->DelinearizedSubscripts,
                                 Shape->DelinearizedSizes);
      if (Acc->DelinearizedSubscripts.size() == 0)
        IsNonAffine = true;
      for (const SCEV *S : Acc->DelinearizedSubscripts)
        if (!isAffine(S, Context, BaseValue))
          IsNonAffine = true;
    }

    // (Possibly) report non affine access
    if (IsNonAffine) {
      BasePtrHasNonAffine = true;
      if (!AllowNonAffine)
        invalid<ReportNonAffineAccess>(Context, /*Assert=*/true, Pair.second,
                                       Insn, BaseValue);
      if (!KeepGoing && !AllowNonAffine)
        return false;
    }
  }

  if (!BasePtrHasNonAffine)
    InsnToMemAcc.insert(TempMemoryAccesses.begin(), TempMemoryAccesses.end());

  return true;
}

bool ScopDetection::hasBaseAffineAccesses(
    DetectionContext &Context, const SCEVUnknown *BasePointer) const {
  auto Shape = std::shared_ptr<ArrayShape>(new ArrayShape(BasePointer));

  auto Terms = getDelinearizationTerms(Context, BasePointer);

  SE->findArrayDimensions(Terms, Shape->DelinearizedSizes,
                          Context.ElementSize[BasePointer]);

  if (!hasValidArraySizes(Context, Shape->DelinearizedSizes, BasePointer))
    return false;

  return computeAccessFunctions(Context, BasePointer, Shape);
}

bool ScopDetection::hasAffineMemoryAccesses(DetectionContext &Context) const {
  for (const SCEVUnknown *BasePointer : Context.NonAffineAccesses)
    if (!hasBaseAffineAccesses(Context, BasePointer)) {
      if (KeepGoing)
        continue;
      else
        return false;
    }
  return true;
}

bool ScopDetection::isValidMemoryAccess(Instruction &Inst,
                                        DetectionContext &Context) const {
  Region &CurRegion = Context.CurRegion;

  Value *Ptr = getPointerOperand(Inst);
  Loop *L = LI->getLoopFor(Inst.getParent());
  const SCEV *AccessFunction = SE->getSCEVAtScope(Ptr, L);
  const SCEVUnknown *BasePointer;
  Value *BaseValue;

  BasePointer = dyn_cast<SCEVUnknown>(SE->getPointerBase(AccessFunction));

  if (!BasePointer)
    return invalid<ReportNoBasePtr>(Context, /*Assert=*/true, &Inst);

  BaseValue = BasePointer->getValue();

  if (isa<UndefValue>(BaseValue))
    return invalid<ReportUndefBasePtr>(Context, /*Assert=*/true, &Inst);

  // Check that the base address of the access is invariant in the current
  // region.
  if (!isInvariant(*BaseValue, CurRegion))
    return invalid<ReportVariantBasePtr>(Context, /*Assert=*/true, BaseValue,
                                         &Inst);

  AccessFunction = SE->getMinusSCEV(AccessFunction, BasePointer);

  const SCEV *Size = SE->getElementSize(&Inst);
  if (Context.ElementSize.count(BasePointer)) {
    if (Context.ElementSize[BasePointer] != Size)
      return invalid<ReportDifferentArrayElementSize>(Context, /*Assert=*/true,
                                                      &Inst, BaseValue);
  } else {
    Context.ElementSize[BasePointer] = Size;
  }

  bool isVariantInNonAffineLoop = false;
  SetVector<const Loop *> Loops;
  findLoops(AccessFunction, Loops);
  for (const Loop *L : Loops)
    if (Context.BoxedLoopsSet.count(L))
      isVariantInNonAffineLoop = true;

  if (PollyDelinearize && !isVariantInNonAffineLoop) {
    Context.Accesses[BasePointer].push_back({&Inst, AccessFunction});

    if (!isAffine(AccessFunction, Context, BaseValue))
      Context.NonAffineAccesses.insert(BasePointer);
  } else if (!AllowNonAffine) {
    if (isVariantInNonAffineLoop ||
        !isAffine(AccessFunction, Context, BaseValue))
      return invalid<ReportNonAffineAccess>(Context, /*Assert=*/true,
                                            AccessFunction, &Inst, BaseValue);
  }

  // FIXME: Think about allowing IntToPtrInst
  if (IntToPtrInst *Inst = dyn_cast<IntToPtrInst>(BaseValue))
    return invalid<ReportIntToPtr>(Context, /*Assert=*/true, Inst);

  if (IgnoreAliasing)
    return true;

  // Check if the base pointer of the memory access does alias with
  // any other pointer. This cannot be handled at the moment.
  AAMDNodes AATags;
  Inst.getAAMetadata(AATags);
  AliasSet &AS = Context.AST.getAliasSetForPointer(
      BaseValue, MemoryLocation::UnknownSize, AATags);

  if (!AS.isMustAlias()) {
    if (PollyUseRuntimeAliasChecks) {
      bool CanBuildRunTimeCheck = true;
      // The run-time alias check places code that involves the base pointer at
      // the beginning of the SCoP. This breaks if the base pointer is defined
      // inside the scop. Hence, we can only create a run-time check if we are
      // sure the base pointer is not an instruction defined inside the scop.
      // However, we can ignore loads that will be hoisted.
      for (const auto &Ptr : AS) {
        Instruction *Inst = dyn_cast<Instruction>(Ptr.getValue());
        if (Inst && CurRegion.contains(Inst)) {
          auto *Load = dyn_cast<LoadInst>(Inst);
          if (Load && isHoistableLoad(Load, CurRegion, *LI, *SE)) {
            Context.RequiredILS.insert(Load);
            continue;
          }

          CanBuildRunTimeCheck = false;
          break;
        }
      }

      if (CanBuildRunTimeCheck)
        return true;
    }
    return invalid<ReportAlias>(Context, /*Assert=*/true, &Inst, AS);
  }

  return true;
}

bool ScopDetection::isValidInstruction(Instruction &Inst,
                                       DetectionContext &Context) const {
  for (auto &Op : Inst.operands()) {
    auto *OpInst = dyn_cast<Instruction>(&Op);

    if (!OpInst)
      continue;

    if (isErrorBlock(*OpInst->getParent(), Context.CurRegion, *LI, *DT))
      return false;
  }

  // We only check the call instruction but not invoke instruction.
  if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
    if (isValidCallInst(*CI))
      return true;

    return invalid<ReportFuncCall>(Context, /*Assert=*/true, &Inst);
  }

  if (!Inst.mayWriteToMemory() && !Inst.mayReadFromMemory()) {
    if (!isa<AllocaInst>(Inst))
      return true;

    return invalid<ReportAlloca>(Context, /*Assert=*/true, &Inst);
  }

  // Check the access function.
  if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst)) {
    Context.hasStores |= isa<StoreInst>(Inst);
    Context.hasLoads |= isa<LoadInst>(Inst);
    if (auto *Load = dyn_cast<LoadInst>(&Inst))
      if (!Load->isSimple())
        return invalid<ReportNonSimpleMemoryAccess>(Context, /*Assert=*/true,
                                                    &Inst);
    if (auto *Store = dyn_cast<StoreInst>(&Inst))
      if (!Store->isSimple())
        return invalid<ReportNonSimpleMemoryAccess>(Context, /*Assert=*/true,
                                                    &Inst);

    return isValidMemoryAccess(Inst, Context);
  }

  // We do not know this instruction, therefore we assume it is invalid.
  return invalid<ReportUnknownInst>(Context, /*Assert=*/true, &Inst);
}

bool ScopDetection::canUseISLTripCount(Loop *L,
                                       DetectionContext &Context) const {
  // Ensure the loop has valid exiting blocks as well as latches, otherwise we
  // need to overapproximate it as a boxed loop.
  SmallVector<BasicBlock *, 4> LoopControlBlocks;
  L->getLoopLatches(LoopControlBlocks);
  L->getExitingBlocks(LoopControlBlocks);
  for (BasicBlock *ControlBB : LoopControlBlocks) {
    if (!isValidCFG(*ControlBB, true, false, Context))
      return false;
  }

  // We can use ISL to compute the trip count of L.
  return true;
}

bool ScopDetection::isValidLoop(Loop *L, DetectionContext &Context) const {
  if (canUseISLTripCount(L, Context))
    return true;

  if (AllowNonAffineSubLoops && AllowNonAffineSubRegions) {
    Region *R = RI->getRegionFor(L->getHeader());
    while (R != &Context.CurRegion && !R->contains(L))
      R = R->getParent();

    if (addOverApproximatedRegion(R, Context))
      return true;
  }

  const SCEV *LoopCount = SE->getBackedgeTakenCount(L);
  return invalid<ReportLoopBound>(Context, /*Assert=*/true, L, LoopCount);
}

/// @brief Return the number of loops in @p L (incl. @p L) that have a trip
///        count that is not known to be less than MIN_LOOP_TRIP_COUNT.
static int countBeneficialSubLoops(Loop *L, ScalarEvolution &SE) {
  auto *TripCount = SE.getBackedgeTakenCount(L);

  int count = 1;
  if (auto *TripCountC = dyn_cast<SCEVConstant>(TripCount))
    if (TripCountC->getType()->getScalarSizeInBits() <= 64)
      if (TripCountC->getValue()->getZExtValue() < MIN_LOOP_TRIP_COUNT)
        count -= 1;

  for (auto &SubLoop : *L)
    count += countBeneficialSubLoops(SubLoop, SE);

  return count;
}

int ScopDetection::countBeneficialLoops(Region *R) const {
  int LoopNum = 0;

  auto L = LI->getLoopFor(R->getEntry());
  L = L ? R->outermostLoopInRegion(L) : nullptr;
  L = L ? L->getParentLoop() : nullptr;

  auto SubLoops =
      L ? L->getSubLoopsVector() : std::vector<Loop *>(LI->begin(), LI->end());

  for (auto &SubLoop : SubLoops)
    if (R->contains(SubLoop))
      LoopNum += countBeneficialSubLoops(SubLoop, *SE);

  return LoopNum;
}

Region *ScopDetection::expandRegion(Region &R) {
  // Initial no valid region was found (greater than R)
  std::unique_ptr<Region> LastValidRegion;
  auto ExpandedRegion = std::unique_ptr<Region>(R.getExpandedRegion());

  DEBUG(dbgs() << "\tExpanding " << R.getNameStr() << "\n");

  while (ExpandedRegion) {
    const auto &It = DetectionContextMap.insert(std::make_pair(
        ExpandedRegion.get(),
        DetectionContext(*ExpandedRegion, *AA, false /*verifying*/)));
    DetectionContext &Context = It.first->second;
    DEBUG(dbgs() << "\t\tTrying " << ExpandedRegion->getNameStr() << "\n");
    // Only expand when we did not collect errors.

    if (!Context.Log.hasErrors()) {
      // If the exit is valid check all blocks
      //  - if true, a valid region was found => store it + keep expanding
      //  - if false, .tbd. => stop  (should this really end the loop?)
      if (!allBlocksValid(Context) || Context.Log.hasErrors()) {
        removeCachedResults(*ExpandedRegion);
        break;
      }

      // Store this region, because it is the greatest valid (encountered so
      // far).
      removeCachedResults(*LastValidRegion);
      LastValidRegion = std::move(ExpandedRegion);

      // Create and test the next greater region (if any)
      ExpandedRegion =
          std::unique_ptr<Region>(LastValidRegion->getExpandedRegion());

    } else {
      // Create and test the next greater region (if any)
      removeCachedResults(*ExpandedRegion);
      ExpandedRegion =
          std::unique_ptr<Region>(ExpandedRegion->getExpandedRegion());
    }
  }

  DEBUG({
    if (LastValidRegion)
      dbgs() << "\tto " << LastValidRegion->getNameStr() << "\n";
    else
      dbgs() << "\tExpanding " << R.getNameStr() << " failed\n";
  });

  return LastValidRegion.release();
}
static bool regionWithoutLoops(Region &R, LoopInfo *LI) {
  for (const BasicBlock *BB : R.blocks())
    if (R.contains(LI->getLoopFor(BB)))
      return false;

  return true;
}

unsigned ScopDetection::removeCachedResultsRecursively(const Region &R) {
  unsigned Count = 0;
  for (auto &SubRegion : R) {
    if (ValidRegions.count(SubRegion.get())) {
      removeCachedResults(*SubRegion.get());
      ++Count;
    } else
      Count += removeCachedResultsRecursively(*SubRegion);
  }
  return Count;
}

void ScopDetection::removeCachedResults(const Region &R) {
  ValidRegions.remove(&R);
  DetectionContextMap.erase(&R);
}

void ScopDetection::findScops(Region &R) {
  const auto &It = DetectionContextMap.insert(
      std::make_pair(&R, DetectionContext(R, *AA, false /*verifying*/)));
  DetectionContext &Context = It.first->second;

  bool RegionIsValid = false;
  if (!PollyProcessUnprofitable && regionWithoutLoops(R, LI)) {
    removeCachedResults(R);
    invalid<ReportUnprofitable>(Context, /*Assert=*/true, &R);
  } else
    RegionIsValid = isValidRegion(Context);

  bool HasErrors = !RegionIsValid || Context.Log.size() > 0;

  if (PollyTrackFailures && HasErrors)
    RejectLogs.insert(std::make_pair(&R, Context.Log));

  if (HasErrors) {
    removeCachedResults(R);
  } else {
    ++ValidRegion;
    ValidRegions.insert(&R);
    return;
  }

  for (auto &SubRegion : R)
    findScops(*SubRegion);

  // Try to expand regions.
  //
  // As the region tree normally only contains canonical regions, non canonical
  // regions that form a Scop are not found. Therefore, those non canonical
  // regions are checked by expanding the canonical ones.

  std::vector<Region *> ToExpand;

  for (auto &SubRegion : R)
    ToExpand.push_back(SubRegion.get());

  for (Region *CurrentRegion : ToExpand) {
    // Skip regions that had errors.
    bool HadErrors = RejectLogs.hasErrors(CurrentRegion);
    if (HadErrors)
      continue;

    // Skip invalid regions. Regions may become invalid, if they are element of
    // an already expanded region.
    if (!ValidRegions.count(CurrentRegion))
      continue;

    Region *ExpandedR = expandRegion(*CurrentRegion);

    if (!ExpandedR)
      continue;

    R.addSubRegion(ExpandedR, true);
    ValidRegions.insert(ExpandedR);
    removeCachedResults(*CurrentRegion);

    // Erase all (direct and indirect) children of ExpandedR from the valid
    // regions and update the number of valid regions.
    ValidRegion -= removeCachedResultsRecursively(*ExpandedR);
  }
}

bool ScopDetection::allBlocksValid(DetectionContext &Context) const {
  Region &CurRegion = Context.CurRegion;

  for (const BasicBlock *BB : CurRegion.blocks()) {
    Loop *L = LI->getLoopFor(BB);
    if (L && L->getHeader() == BB && (!isValidLoop(L, Context) && !KeepGoing))
      return false;
  }

  for (BasicBlock *BB : CurRegion.blocks()) {
    bool IsErrorBlock = isErrorBlock(*BB, CurRegion, *LI, *DT);

    // Also check exception blocks (and possibly register them as non-affine
    // regions). Even though exception blocks are not modeled, we use them
    // to forward-propagate domain constraints during ScopInfo construction.
    if (!isValidCFG(*BB, false, IsErrorBlock, Context) && !KeepGoing)
      return false;

    if (IsErrorBlock)
      continue;

    for (BasicBlock::iterator I = BB->begin(), E = --BB->end(); I != E; ++I)
      if (!isValidInstruction(*I, Context) && !KeepGoing)
        return false;
  }

  if (!hasAffineMemoryAccesses(Context))
    return false;

  return true;
}

bool ScopDetection::hasSufficientCompute(DetectionContext &Context,
                                         int NumLoops) const {
  int InstCount = 0;

  for (auto *BB : Context.CurRegion.blocks())
    if (Context.CurRegion.contains(LI->getLoopFor(BB)))
      InstCount += BB->size();

  InstCount = InstCount / NumLoops;

  return InstCount >= ProfitabilityMinPerLoopInstructions;
}

bool ScopDetection::isProfitableRegion(DetectionContext &Context) const {
  Region &CurRegion = Context.CurRegion;

  if (PollyProcessUnprofitable)
    return true;

  // We can probably not do a lot on scops that only write or only read
  // data.
  if (!Context.hasStores || !Context.hasLoads)
    return invalid<ReportUnprofitable>(Context, /*Assert=*/true, &CurRegion);

  int NumLoops = countBeneficialLoops(&CurRegion);
  int NumAffineLoops = NumLoops - Context.BoxedLoopsSet.size();

  // Scops with at least two loops may allow either loop fusion or tiling and
  // are consequently interesting to look at.
  if (NumAffineLoops >= 2)
    return true;

  // Scops that contain a loop with a non-trivial amount of computation per
  // loop-iteration are interesting as we may be able to parallelize such
  // loops. Individual loops that have only a small amount of computation
  // per-iteration are performance-wise very fragile as any change to the
  // loop induction variables may affect performance. To not cause spurious
  // performance regressions, we do not consider such loops.
  if (NumAffineLoops == 1 && hasSufficientCompute(Context, NumLoops))
    return true;

  return invalid<ReportUnprofitable>(Context, /*Assert=*/true, &CurRegion);
}

bool ScopDetection::isValidRegion(DetectionContext &Context) const {
  Region &CurRegion = Context.CurRegion;

  DEBUG(dbgs() << "Checking region: " << CurRegion.getNameStr() << "\n\t");

  if (CurRegion.isTopLevelRegion()) {
    DEBUG(dbgs() << "Top level region is invalid\n");
    return false;
  }

  if (!CurRegion.getEntry()->getName().count(OnlyRegion)) {
    DEBUG({
      dbgs() << "Region entry does not match -polly-region-only";
      dbgs() << "\n";
    });
    return false;
  }

  // SCoP cannot contain the entry block of the function, because we need
  // to insert alloca instruction there when translate scalar to array.
  if (CurRegion.getEntry() ==
      &(CurRegion.getEntry()->getParent()->getEntryBlock()))
    return invalid<ReportEntry>(Context, /*Assert=*/true, CurRegion.getEntry());

  if (!allBlocksValid(Context))
    return false;

  DebugLoc DbgLoc;
  if (!isReducibleRegion(CurRegion, DbgLoc))
    return invalid<ReportIrreducibleRegion>(Context, /*Assert=*/true,
                                            &CurRegion, DbgLoc);

  if (!isProfitableRegion(Context))
    return false;

  DEBUG(dbgs() << "OK\n");
  return true;
}

void ScopDetection::markFunctionAsInvalid(Function *F) const {
  F->addFnAttr(PollySkipFnAttr);
}

bool ScopDetection::isValidFunction(llvm::Function &F) {
  return !F.hasFnAttribute(PollySkipFnAttr);
}

void ScopDetection::printLocations(llvm::Function &F) {
  for (const Region *R : *this) {
    unsigned LineEntry, LineExit;
    std::string FileName;

    getDebugLocation(R, LineEntry, LineExit, FileName);
    DiagnosticScopFound Diagnostic(F, FileName, LineEntry, LineExit);
    F.getContext().diagnose(Diagnostic);
  }
}

void ScopDetection::emitMissedRemarksForValidRegions(const Function &F) {
  for (const Region *R : ValidRegions) {
    const Region *Parent = R->getParent();
    if (Parent && !Parent->isTopLevelRegion() && RejectLogs.count(Parent))
      emitRejectionRemarks(F, RejectLogs.at(Parent));
  }
}

void ScopDetection::emitMissedRemarksForLeaves(const Function &F,
                                               const Region *R) {
  for (const std::unique_ptr<Region> &Child : *R) {
    bool IsValid = DetectionContextMap.count(Child.get());
    if (IsValid)
      continue;

    bool IsLeaf = Child->begin() == Child->end();
    if (!IsLeaf)
      emitMissedRemarksForLeaves(F, Child.get());
    else {
      if (RejectLogs.count(Child.get())) {
        emitRejectionRemarks(F, RejectLogs.at(Child.get()));
      }
    }
  }
}

bool ScopDetection::isReducibleRegion(Region &R, DebugLoc &DbgLoc) const {
  BasicBlock *REntry = R.getEntry();
  BasicBlock *RExit = R.getExit();
  // Map to match the color of a BasicBlock during the DFS walk.
  DenseMap<const BasicBlock *, Color> BBColorMap;
  // Stack keeping track of current BB and index of next child to be processed.
  std::stack<std::pair<BasicBlock *, unsigned>> DFSStack;

  unsigned AdjacentBlockIndex = 0;
  BasicBlock *CurrBB, *SuccBB;
  CurrBB = REntry;

  // Initialize the map for all BB with WHITE color.
  for (auto *BB : R.blocks())
    BBColorMap[BB] = ScopDetection::WHITE;

  // Process the entry block of the Region.
  BBColorMap[CurrBB] = ScopDetection::GREY;
  DFSStack.push(std::make_pair(CurrBB, 0));

  while (!DFSStack.empty()) {
    // Get next BB on stack to be processed.
    CurrBB = DFSStack.top().first;
    AdjacentBlockIndex = DFSStack.top().second;
    DFSStack.pop();

    // Loop to iterate over the successors of current BB.
    const TerminatorInst *TInst = CurrBB->getTerminator();
    unsigned NSucc = TInst->getNumSuccessors();
    for (unsigned I = AdjacentBlockIndex; I < NSucc;
         ++I, ++AdjacentBlockIndex) {
      SuccBB = TInst->getSuccessor(I);

      // Checks for region exit block and self-loops in BB.
      if (SuccBB == RExit || SuccBB == CurrBB)
        continue;

      // WHITE indicates an unvisited BB in DFS walk.
      if (BBColorMap[SuccBB] == ScopDetection::WHITE) {
        // Push the current BB and the index of the next child to be visited.
        DFSStack.push(std::make_pair(CurrBB, I + 1));
        // Push the next BB to be processed.
        DFSStack.push(std::make_pair(SuccBB, 0));
        // First time the BB is being processed.
        BBColorMap[SuccBB] = ScopDetection::GREY;
        break;
      } else if (BBColorMap[SuccBB] == ScopDetection::GREY) {
        // GREY indicates a loop in the control flow.
        // If the destination dominates the source, it is a natural loop
        // else, an irreducible control flow in the region is detected.
        if (!DT->dominates(SuccBB, CurrBB)) {
          // Get debug info of instruction which causes irregular control flow.
          DbgLoc = TInst->getDebugLoc();
          return false;
        }
      }
    }

    // If all children of current BB have been processed,
    // then mark that BB as fully processed.
    if (AdjacentBlockIndex == NSucc)
      BBColorMap[CurrBB] = ScopDetection::BLACK;
  }

  return true;
}

bool ScopDetection::runOnFunction(llvm::Function &F) {
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  RI = &getAnalysis<RegionInfoPass>().getRegionInfo();
  if (!PollyProcessUnprofitable && LI->empty())
    return false;

  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  Region *TopRegion = RI->getTopLevelRegion();

  releaseMemory();

  if (OnlyFunction != "" && !F.getName().count(OnlyFunction))
    return false;

  if (!isValidFunction(F))
    return false;

  findScops(*TopRegion);

  // Only makes sense when we tracked errors.
  if (PollyTrackFailures) {
    emitMissedRemarksForValidRegions(F);
    emitMissedRemarksForLeaves(F, TopRegion);
  }

  if (ReportLevel)
    printLocations(F);

  assert(ValidRegions.size() == DetectionContextMap.size() &&
         "Cached more results than valid regions");
  return false;
}

bool ScopDetection::isNonAffineSubRegion(const Region *SubR,
                                         const Region *ScopR) const {
  const DetectionContext *DC = getDetectionContext(ScopR);
  assert(DC && "ScopR is no valid region!");
  return DC->NonAffineSubRegionSet.count(SubR);
}

const ScopDetection::DetectionContext *
ScopDetection::getDetectionContext(const Region *R) const {
  auto DCMIt = DetectionContextMap.find(R);
  if (DCMIt == DetectionContextMap.end())
    return nullptr;
  return &DCMIt->second;
}

const ScopDetection::BoxedLoopsSetTy *
ScopDetection::getBoxedLoops(const Region *R) const {
  const DetectionContext *DC = getDetectionContext(R);
  assert(DC && "ScopR is no valid region!");
  return &DC->BoxedLoopsSet;
}

const InvariantLoadsSetTy *
ScopDetection::getRequiredInvariantLoads(const Region *R) const {
  const DetectionContext *DC = getDetectionContext(R);
  assert(DC && "ScopR is no valid region!");
  return &DC->RequiredILS;
}

void polly::ScopDetection::verifyRegion(const Region &R) const {
  assert(isMaxRegionInScop(R) && "Expect R is a valid region.");

  DetectionContext Context(const_cast<Region &>(R), *AA, true /*verifying*/);
  isValidRegion(Context);
}

void polly::ScopDetection::verifyAnalysis() const {
  if (!VerifyScops)
    return;

  for (const Region *R : ValidRegions)
    verifyRegion(*R);
}

void ScopDetection::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  // We also need AA and RegionInfo when we are verifying analysis.
  AU.addRequiredTransitive<AAResultsWrapperPass>();
  AU.addRequiredTransitive<RegionInfoPass>();
  AU.setPreservesAll();
}

void ScopDetection::print(raw_ostream &OS, const Module *) const {
  for (const Region *R : ValidRegions)
    OS << "Valid Region for Scop: " << R->getNameStr() << '\n';

  OS << "\n";
}

void ScopDetection::releaseMemory() {
  RejectLogs.clear();
  ValidRegions.clear();
  InsnToMemAcc.clear();
  DetectionContextMap.clear();

  // Do not clear the invalid function set.
}

char ScopDetection::ID = 0;

Pass *polly::createScopDetectionPass() { return new ScopDetection(); }

INITIALIZE_PASS_BEGIN(ScopDetection, "polly-detect",
                      "Polly - Detect static control parts (SCoPs)", false,
                      false);
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass);
INITIALIZE_PASS_END(ScopDetection, "polly-detect",
                    "Polly - Detect static control parts (SCoPs)", false, false)
