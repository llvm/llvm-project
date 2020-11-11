//===- ScopBuilder.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Create a polyhedral description for a static control flow region.
//
// The pass creates a polyhedral description of the Scops detected by the SCoP
// detection derived from their LLVM-IR code.
//
//===----------------------------------------------------------------------===//

#include "polly/ScopBuilder.h"
#include "polly/Options.h"
#include "polly/ScopDetection.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ISLTools.h"
#include "polly/Support/SCEVValidator.h"
#include "polly/Support/ScopHelper.h"
#include "polly/Support/VirtualInstruction.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-scops"

STATISTIC(ScopFound, "Number of valid Scops");
STATISTIC(RichScopFound, "Number of Scops containing a loop");
STATISTIC(InfeasibleScops,
          "Number of SCoPs with statically infeasible context.");

bool polly::ModelReadOnlyScalars;

// The maximal number of dimensions we allow during invariant load construction.
// More complex access ranges will result in very high compile time and are also
// unlikely to result in good code. This value is very high and should only
// trigger for corner cases (e.g., the "dct_luma" function in h264, SPEC2006).
static int const MaxDimensionsInAccessRange = 9;

static cl::opt<bool, true> XModelReadOnlyScalars(
    "polly-analyze-read-only-scalars",
    cl::desc("Model read-only scalar values in the scop description"),
    cl::location(ModelReadOnlyScalars), cl::Hidden, cl::ZeroOrMore,
    cl::init(true), cl::cat(PollyCategory));

static cl::opt<int>
    OptComputeOut("polly-analysis-computeout",
                  cl::desc("Bound the scop analysis by a maximal amount of "
                           "computational steps (0 means no bound)"),
                  cl::Hidden, cl::init(800000), cl::ZeroOrMore,
                  cl::cat(PollyCategory));

static cl::opt<bool> PollyAllowDereferenceOfAllFunctionParams(
    "polly-allow-dereference-of-all-function-parameters",
    cl::desc(
        "Treat all parameters to functions that are pointers as dereferencible."
        " This is useful for invariant load hoisting, since we can generate"
        " less runtime checks. This is only valid if all pointers to functions"
        " are always initialized, so that Polly can choose to hoist"
        " their loads. "),
    cl::Hidden, cl::init(false), cl::cat(PollyCategory));

static cl::opt<bool>
    PollyIgnoreInbounds("polly-ignore-inbounds",
                        cl::desc("Do not take inbounds assumptions at all"),
                        cl::Hidden, cl::init(false), cl::cat(PollyCategory));

static cl::opt<unsigned> RunTimeChecksMaxArraysPerGroup(
    "polly-rtc-max-arrays-per-group",
    cl::desc("The maximal number of arrays to compare in each alias group."),
    cl::Hidden, cl::ZeroOrMore, cl::init(20), cl::cat(PollyCategory));

static cl::opt<int> RunTimeChecksMaxAccessDisjuncts(
    "polly-rtc-max-array-disjuncts",
    cl::desc("The maximal number of disjunts allowed in memory accesses to "
             "to build RTCs."),
    cl::Hidden, cl::ZeroOrMore, cl::init(8), cl::cat(PollyCategory));

static cl::opt<unsigned> RunTimeChecksMaxParameters(
    "polly-rtc-max-parameters",
    cl::desc("The maximal number of parameters allowed in RTCs."), cl::Hidden,
    cl::ZeroOrMore, cl::init(8), cl::cat(PollyCategory));

static cl::opt<bool> UnprofitableScalarAccs(
    "polly-unprofitable-scalar-accs",
    cl::desc("Count statements with scalar accesses as not optimizable"),
    cl::Hidden, cl::init(false), cl::cat(PollyCategory));

static cl::opt<std::string> UserContextStr(
    "polly-context", cl::value_desc("isl parameter set"),
    cl::desc("Provide additional constraints on the context parameters"),
    cl::init(""), cl::cat(PollyCategory));

static cl::opt<bool> DetectFortranArrays(
    "polly-detect-fortran-arrays",
    cl::desc("Detect Fortran arrays and use this for code generation"),
    cl::Hidden, cl::init(false), cl::cat(PollyCategory));

static cl::opt<bool> DetectReductions("polly-detect-reductions",
                                      cl::desc("Detect and exploit reductions"),
                                      cl::Hidden, cl::ZeroOrMore,
                                      cl::init(true), cl::cat(PollyCategory));

// Multiplicative reductions can be disabled separately as these kind of
// operations can overflow easily. Additive reductions and bit operations
// are in contrast pretty stable.
static cl::opt<bool> DisableMultiplicativeReductions(
    "polly-disable-multiplicative-reductions",
    cl::desc("Disable multiplicative reductions"), cl::Hidden, cl::ZeroOrMore,
    cl::init(false), cl::cat(PollyCategory));

enum class GranularityChoice { BasicBlocks, ScalarIndependence, Stores };

static cl::opt<GranularityChoice> StmtGranularity(
    "polly-stmt-granularity",
    cl::desc(
        "Algorithm to use for splitting basic blocks into multiple statements"),
    cl::values(clEnumValN(GranularityChoice::BasicBlocks, "bb",
                          "One statement per basic block"),
               clEnumValN(GranularityChoice::ScalarIndependence, "scalar-indep",
                          "Scalar independence heuristic"),
               clEnumValN(GranularityChoice::Stores, "store",
                          "Store-level granularity")),
    cl::init(GranularityChoice::ScalarIndependence), cl::cat(PollyCategory));

/// Helper to treat non-affine regions and basic blocks the same.
///
///{

/// Return the block that is the representing block for @p RN.
static inline BasicBlock *getRegionNodeBasicBlock(RegionNode *RN) {
  return RN->isSubRegion() ? RN->getNodeAs<Region>()->getEntry()
                           : RN->getNodeAs<BasicBlock>();
}

/// Return the @p idx'th block that is executed after @p RN.
static inline BasicBlock *
getRegionNodeSuccessor(RegionNode *RN, Instruction *TI, unsigned idx) {
  if (RN->isSubRegion()) {
    assert(idx == 0);
    return RN->getNodeAs<Region>()->getExit();
  }
  return TI->getSuccessor(idx);
}

static bool containsErrorBlock(RegionNode *RN, const Region &R, LoopInfo &LI,
                               const DominatorTree &DT) {
  if (!RN->isSubRegion())
    return isErrorBlock(*RN->getNodeAs<BasicBlock>(), R, LI, DT);
  for (BasicBlock *BB : RN->getNodeAs<Region>()->blocks())
    if (isErrorBlock(*BB, R, LI, DT))
      return true;
  return false;
}

///}

/// Create a map to map from a given iteration to a subsequent iteration.
///
/// This map maps from SetSpace -> SetSpace where the dimensions @p Dim
/// is incremented by one and all other dimensions are equal, e.g.,
///             [i0, i1, i2, i3] -> [i0, i1, i2 + 1, i3]
///
/// if @p Dim is 2 and @p SetSpace has 4 dimensions.
static isl::map createNextIterationMap(isl::space SetSpace, unsigned Dim) {
  isl::space MapSpace = SetSpace.map_from_set();
  isl::map NextIterationMap = isl::map::universe(MapSpace);
  for (unsigned u = 0; u < NextIterationMap.dim(isl::dim::in); u++)
    if (u != Dim)
      NextIterationMap =
          NextIterationMap.equate(isl::dim::in, u, isl::dim::out, u);
  isl::constraint C =
      isl::constraint::alloc_equality(isl::local_space(MapSpace));
  C = C.set_constant_si(1);
  C = C.set_coefficient_si(isl::dim::in, Dim, 1);
  C = C.set_coefficient_si(isl::dim::out, Dim, -1);
  NextIterationMap = NextIterationMap.add_constraint(C);
  return NextIterationMap;
}

/// Add @p BSet to set @p BoundedParts if @p BSet is bounded.
static isl::set collectBoundedParts(isl::set S) {
  isl::set BoundedParts = isl::set::empty(S.get_space());
  for (isl::basic_set BSet : S.get_basic_set_list())
    if (BSet.is_bounded())
      BoundedParts = BoundedParts.unite(isl::set(BSet));
  return BoundedParts;
}

/// Compute the (un)bounded parts of @p S wrt. to dimension @p Dim.
///
/// @returns A separation of @p S into first an unbounded then a bounded subset,
///          both with regards to the dimension @p Dim.
static std::pair<isl::set, isl::set> partitionSetParts(isl::set S,
                                                       unsigned Dim) {
  for (unsigned u = 0, e = S.n_dim(); u < e; u++)
    S = S.lower_bound_si(isl::dim::set, u, 0);

  unsigned NumDimsS = S.n_dim();
  isl::set OnlyDimS = S;

  // Remove dimensions that are greater than Dim as they are not interesting.
  assert(NumDimsS >= Dim + 1);
  OnlyDimS = OnlyDimS.project_out(isl::dim::set, Dim + 1, NumDimsS - Dim - 1);

  // Create artificial parametric upper bounds for dimensions smaller than Dim
  // as we are not interested in them.
  OnlyDimS = OnlyDimS.insert_dims(isl::dim::param, 0, Dim);

  for (unsigned u = 0; u < Dim; u++) {
    isl::constraint C = isl::constraint::alloc_inequality(
        isl::local_space(OnlyDimS.get_space()));
    C = C.set_coefficient_si(isl::dim::param, u, 1);
    C = C.set_coefficient_si(isl::dim::set, u, -1);
    OnlyDimS = OnlyDimS.add_constraint(C);
  }

  // Collect all bounded parts of OnlyDimS.
  isl::set BoundedParts = collectBoundedParts(OnlyDimS);

  // Create the dimensions greater than Dim again.
  BoundedParts =
      BoundedParts.insert_dims(isl::dim::set, Dim + 1, NumDimsS - Dim - 1);

  // Remove the artificial upper bound parameters again.
  BoundedParts = BoundedParts.remove_dims(isl::dim::param, 0, Dim);

  isl::set UnboundedParts = S.subtract(BoundedParts);
  return std::make_pair(UnboundedParts, BoundedParts);
}

/// Create the conditions under which @p L @p Pred @p R is true.
static isl::set buildConditionSet(ICmpInst::Predicate Pred, isl::pw_aff L,
                                  isl::pw_aff R) {
  switch (Pred) {
  case ICmpInst::ICMP_EQ:
    return L.eq_set(R);
  case ICmpInst::ICMP_NE:
    return L.ne_set(R);
  case ICmpInst::ICMP_SLT:
    return L.lt_set(R);
  case ICmpInst::ICMP_SLE:
    return L.le_set(R);
  case ICmpInst::ICMP_SGT:
    return L.gt_set(R);
  case ICmpInst::ICMP_SGE:
    return L.ge_set(R);
  case ICmpInst::ICMP_ULT:
    return L.lt_set(R);
  case ICmpInst::ICMP_UGT:
    return L.gt_set(R);
  case ICmpInst::ICMP_ULE:
    return L.le_set(R);
  case ICmpInst::ICMP_UGE:
    return L.ge_set(R);
  default:
    llvm_unreachable("Non integer predicate not supported");
  }
}

isl::set ScopBuilder::adjustDomainDimensions(isl::set Dom, Loop *OldL,
                                             Loop *NewL) {
  // If the loops are the same there is nothing to do.
  if (NewL == OldL)
    return Dom;

  int OldDepth = scop->getRelativeLoopDepth(OldL);
  int NewDepth = scop->getRelativeLoopDepth(NewL);
  // If both loops are non-affine loops there is nothing to do.
  if (OldDepth == -1 && NewDepth == -1)
    return Dom;

  // Distinguish three cases:
  //   1) The depth is the same but the loops are not.
  //      => One loop was left one was entered.
  //   2) The depth increased from OldL to NewL.
  //      => One loop was entered, none was left.
  //   3) The depth decreased from OldL to NewL.
  //      => Loops were left were difference of the depths defines how many.
  if (OldDepth == NewDepth) {
    assert(OldL->getParentLoop() == NewL->getParentLoop());
    Dom = Dom.project_out(isl::dim::set, NewDepth, 1);
    Dom = Dom.add_dims(isl::dim::set, 1);
  } else if (OldDepth < NewDepth) {
    assert(OldDepth + 1 == NewDepth);
    auto &R = scop->getRegion();
    (void)R;
    assert(NewL->getParentLoop() == OldL ||
           ((!OldL || !R.contains(OldL)) && R.contains(NewL)));
    Dom = Dom.add_dims(isl::dim::set, 1);
  } else {
    assert(OldDepth > NewDepth);
    int Diff = OldDepth - NewDepth;
    int NumDim = Dom.n_dim();
    assert(NumDim >= Diff);
    Dom = Dom.project_out(isl::dim::set, NumDim - Diff, Diff);
  }

  return Dom;
}

/// Compute the isl representation for the SCEV @p E in this BB.
///
/// @param BB               The BB for which isl representation is to be
/// computed.
/// @param InvalidDomainMap A map of BB to their invalid domains.
/// @param E                The SCEV that should be translated.
/// @param NonNegative      Flag to indicate the @p E has to be non-negative.
///
/// Note that this function will also adjust the invalid context accordingly.

__isl_give isl_pw_aff *
ScopBuilder::getPwAff(BasicBlock *BB,
                      DenseMap<BasicBlock *, isl::set> &InvalidDomainMap,
                      const SCEV *E, bool NonNegative) {
  PWACtx PWAC = scop->getPwAff(E, BB, NonNegative, &RecordedAssumptions);
  InvalidDomainMap[BB] = InvalidDomainMap[BB].unite(PWAC.second);
  return PWAC.first.release();
}

/// Build condition sets for unsigned ICmpInst(s).
/// Special handling is required for unsigned operands to ensure that if
/// MSB (aka the Sign bit) is set for an operands in an unsigned ICmpInst
/// it should wrap around.
///
/// @param IsStrictUpperBound holds information on the predicate relation
/// between TestVal and UpperBound, i.e,
/// TestVal < UpperBound  OR  TestVal <= UpperBound
__isl_give isl_set *ScopBuilder::buildUnsignedConditionSets(
    BasicBlock *BB, Value *Condition, __isl_keep isl_set *Domain,
    const SCEV *SCEV_TestVal, const SCEV *SCEV_UpperBound,
    DenseMap<BasicBlock *, isl::set> &InvalidDomainMap,
    bool IsStrictUpperBound) {
  // Do not take NonNeg assumption on TestVal
  // as it might have MSB (Sign bit) set.
  isl_pw_aff *TestVal = getPwAff(BB, InvalidDomainMap, SCEV_TestVal, false);
  // Take NonNeg assumption on UpperBound.
  isl_pw_aff *UpperBound =
      getPwAff(BB, InvalidDomainMap, SCEV_UpperBound, true);

  // 0 <= TestVal
  isl_set *First =
      isl_pw_aff_le_set(isl_pw_aff_zero_on_domain(isl_local_space_from_space(
                            isl_pw_aff_get_domain_space(TestVal))),
                        isl_pw_aff_copy(TestVal));

  isl_set *Second;
  if (IsStrictUpperBound)
    // TestVal < UpperBound
    Second = isl_pw_aff_lt_set(TestVal, UpperBound);
  else
    // TestVal <= UpperBound
    Second = isl_pw_aff_le_set(TestVal, UpperBound);

  isl_set *ConsequenceCondSet = isl_set_intersect(First, Second);
  return ConsequenceCondSet;
}

bool ScopBuilder::buildConditionSets(
    BasicBlock *BB, SwitchInst *SI, Loop *L, __isl_keep isl_set *Domain,
    DenseMap<BasicBlock *, isl::set> &InvalidDomainMap,
    SmallVectorImpl<__isl_give isl_set *> &ConditionSets) {
  Value *Condition = getConditionFromTerminator(SI);
  assert(Condition && "No condition for switch");

  isl_pw_aff *LHS, *RHS;
  LHS = getPwAff(BB, InvalidDomainMap, SE.getSCEVAtScope(Condition, L));

  unsigned NumSuccessors = SI->getNumSuccessors();
  ConditionSets.resize(NumSuccessors);
  for (auto &Case : SI->cases()) {
    unsigned Idx = Case.getSuccessorIndex();
    ConstantInt *CaseValue = Case.getCaseValue();

    RHS = getPwAff(BB, InvalidDomainMap, SE.getSCEV(CaseValue));
    isl_set *CaseConditionSet =
        buildConditionSet(ICmpInst::ICMP_EQ, isl::manage_copy(LHS),
                          isl::manage(RHS))
            .release();
    ConditionSets[Idx] = isl_set_coalesce(
        isl_set_intersect(CaseConditionSet, isl_set_copy(Domain)));
  }

  assert(ConditionSets[0] == nullptr && "Default condition set was set");
  isl_set *ConditionSetUnion = isl_set_copy(ConditionSets[1]);
  for (unsigned u = 2; u < NumSuccessors; u++)
    ConditionSetUnion =
        isl_set_union(ConditionSetUnion, isl_set_copy(ConditionSets[u]));
  ConditionSets[0] = isl_set_subtract(isl_set_copy(Domain), ConditionSetUnion);

  isl_pw_aff_free(LHS);

  return true;
}

bool ScopBuilder::buildConditionSets(
    BasicBlock *BB, Value *Condition, Instruction *TI, Loop *L,
    __isl_keep isl_set *Domain,
    DenseMap<BasicBlock *, isl::set> &InvalidDomainMap,
    SmallVectorImpl<__isl_give isl_set *> &ConditionSets) {
  isl_set *ConsequenceCondSet = nullptr;

  if (auto Load = dyn_cast<LoadInst>(Condition)) {
    const SCEV *LHSSCEV = SE.getSCEVAtScope(Load, L);
    const SCEV *RHSSCEV = SE.getZero(LHSSCEV->getType());
    bool NonNeg = false;
    isl_pw_aff *LHS = getPwAff(BB, InvalidDomainMap, LHSSCEV, NonNeg);
    isl_pw_aff *RHS = getPwAff(BB, InvalidDomainMap, RHSSCEV, NonNeg);
    ConsequenceCondSet = buildConditionSet(ICmpInst::ICMP_SLE, isl::manage(LHS),
                                           isl::manage(RHS))
                             .release();
  } else if (auto *PHI = dyn_cast<PHINode>(Condition)) {
    auto *Unique = dyn_cast<ConstantInt>(
        getUniqueNonErrorValue(PHI, &scop->getRegion(), LI, DT));

    if (Unique->isZero())
      ConsequenceCondSet = isl_set_empty(isl_set_get_space(Domain));
    else
      ConsequenceCondSet = isl_set_universe(isl_set_get_space(Domain));
  } else if (auto *CCond = dyn_cast<ConstantInt>(Condition)) {
    if (CCond->isZero())
      ConsequenceCondSet = isl_set_empty(isl_set_get_space(Domain));
    else
      ConsequenceCondSet = isl_set_universe(isl_set_get_space(Domain));
  } else if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(Condition)) {
    auto Opcode = BinOp->getOpcode();
    assert(Opcode == Instruction::And || Opcode == Instruction::Or);

    bool Valid = buildConditionSets(BB, BinOp->getOperand(0), TI, L, Domain,
                                    InvalidDomainMap, ConditionSets) &&
                 buildConditionSets(BB, BinOp->getOperand(1), TI, L, Domain,
                                    InvalidDomainMap, ConditionSets);
    if (!Valid) {
      while (!ConditionSets.empty())
        isl_set_free(ConditionSets.pop_back_val());
      return false;
    }

    isl_set_free(ConditionSets.pop_back_val());
    isl_set *ConsCondPart0 = ConditionSets.pop_back_val();
    isl_set_free(ConditionSets.pop_back_val());
    isl_set *ConsCondPart1 = ConditionSets.pop_back_val();

    if (Opcode == Instruction::And)
      ConsequenceCondSet = isl_set_intersect(ConsCondPart0, ConsCondPart1);
    else
      ConsequenceCondSet = isl_set_union(ConsCondPart0, ConsCondPart1);
  } else {
    auto *ICond = dyn_cast<ICmpInst>(Condition);
    assert(ICond &&
           "Condition of exiting branch was neither constant nor ICmp!");

    Region &R = scop->getRegion();

    isl_pw_aff *LHS, *RHS;
    // For unsigned comparisons we assumed the signed bit of neither operand
    // to be set. The comparison is equal to a signed comparison under this
    // assumption.
    bool NonNeg = ICond->isUnsigned();
    const SCEV *LeftOperand = SE.getSCEVAtScope(ICond->getOperand(0), L),
               *RightOperand = SE.getSCEVAtScope(ICond->getOperand(1), L);

    LeftOperand = tryForwardThroughPHI(LeftOperand, R, SE, LI, DT);
    RightOperand = tryForwardThroughPHI(RightOperand, R, SE, LI, DT);

    switch (ICond->getPredicate()) {
    case ICmpInst::ICMP_ULT:
      ConsequenceCondSet =
          buildUnsignedConditionSets(BB, Condition, Domain, LeftOperand,
                                     RightOperand, InvalidDomainMap, true);
      break;
    case ICmpInst::ICMP_ULE:
      ConsequenceCondSet =
          buildUnsignedConditionSets(BB, Condition, Domain, LeftOperand,
                                     RightOperand, InvalidDomainMap, false);
      break;
    case ICmpInst::ICMP_UGT:
      ConsequenceCondSet =
          buildUnsignedConditionSets(BB, Condition, Domain, RightOperand,
                                     LeftOperand, InvalidDomainMap, true);
      break;
    case ICmpInst::ICMP_UGE:
      ConsequenceCondSet =
          buildUnsignedConditionSets(BB, Condition, Domain, RightOperand,
                                     LeftOperand, InvalidDomainMap, false);
      break;
    default:
      LHS = getPwAff(BB, InvalidDomainMap, LeftOperand, NonNeg);
      RHS = getPwAff(BB, InvalidDomainMap, RightOperand, NonNeg);
      ConsequenceCondSet = buildConditionSet(ICond->getPredicate(),
                                             isl::manage(LHS), isl::manage(RHS))
                               .release();
      break;
    }
  }

  // If no terminator was given we are only looking for parameter constraints
  // under which @p Condition is true/false.
  if (!TI)
    ConsequenceCondSet = isl_set_params(ConsequenceCondSet);
  assert(ConsequenceCondSet);
  ConsequenceCondSet = isl_set_coalesce(
      isl_set_intersect(ConsequenceCondSet, isl_set_copy(Domain)));

  isl_set *AlternativeCondSet = nullptr;
  bool TooComplex =
      isl_set_n_basic_set(ConsequenceCondSet) >= MaxDisjunctsInDomain;

  if (!TooComplex) {
    AlternativeCondSet = isl_set_subtract(isl_set_copy(Domain),
                                          isl_set_copy(ConsequenceCondSet));
    TooComplex =
        isl_set_n_basic_set(AlternativeCondSet) >= MaxDisjunctsInDomain;
  }

  if (TooComplex) {
    scop->invalidate(COMPLEXITY, TI ? TI->getDebugLoc() : DebugLoc(),
                     TI ? TI->getParent() : nullptr /* BasicBlock */);
    isl_set_free(AlternativeCondSet);
    isl_set_free(ConsequenceCondSet);
    return false;
  }

  ConditionSets.push_back(ConsequenceCondSet);
  ConditionSets.push_back(isl_set_coalesce(AlternativeCondSet));

  return true;
}

bool ScopBuilder::buildConditionSets(
    BasicBlock *BB, Instruction *TI, Loop *L, __isl_keep isl_set *Domain,
    DenseMap<BasicBlock *, isl::set> &InvalidDomainMap,
    SmallVectorImpl<__isl_give isl_set *> &ConditionSets) {
  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI))
    return buildConditionSets(BB, SI, L, Domain, InvalidDomainMap,
                              ConditionSets);

  assert(isa<BranchInst>(TI) && "Terminator was neither branch nor switch.");

  if (TI->getNumSuccessors() == 1) {
    ConditionSets.push_back(isl_set_copy(Domain));
    return true;
  }

  Value *Condition = getConditionFromTerminator(TI);
  assert(Condition && "No condition for Terminator");

  return buildConditionSets(BB, Condition, TI, L, Domain, InvalidDomainMap,
                            ConditionSets);
}

bool ScopBuilder::propagateDomainConstraints(
    Region *R, DenseMap<BasicBlock *, isl::set> &InvalidDomainMap) {
  // Iterate over the region R and propagate the domain constrains from the
  // predecessors to the current node. In contrast to the
  // buildDomainsWithBranchConstraints function, this one will pull the domain
  // information from the predecessors instead of pushing it to the successors.
  // Additionally, we assume the domains to be already present in the domain
  // map here. However, we iterate again in reverse post order so we know all
  // predecessors have been visited before a block or non-affine subregion is
  // visited.

  ReversePostOrderTraversal<Region *> RTraversal(R);
  for (auto *RN : RTraversal) {
    // Recurse for affine subregions but go on for basic blocks and non-affine
    // subregions.
    if (RN->isSubRegion()) {
      Region *SubRegion = RN->getNodeAs<Region>();
      if (!scop->isNonAffineSubRegion(SubRegion)) {
        if (!propagateDomainConstraints(SubRegion, InvalidDomainMap))
          return false;
        continue;
      }
    }

    BasicBlock *BB = getRegionNodeBasicBlock(RN);
    isl::set &Domain = scop->getOrInitEmptyDomain(BB);
    assert(Domain);

    // Under the union of all predecessor conditions we can reach this block.
    isl::set PredDom = getPredecessorDomainConstraints(BB, Domain);
    Domain = Domain.intersect(PredDom).coalesce();
    Domain = Domain.align_params(scop->getParamSpace());

    Loop *BBLoop = getRegionNodeLoop(RN, LI);
    if (BBLoop && BBLoop->getHeader() == BB && scop->contains(BBLoop))
      if (!addLoopBoundsToHeaderDomain(BBLoop, InvalidDomainMap))
        return false;
  }

  return true;
}

void ScopBuilder::propagateDomainConstraintsToRegionExit(
    BasicBlock *BB, Loop *BBLoop,
    SmallPtrSetImpl<BasicBlock *> &FinishedExitBlocks,
    DenseMap<BasicBlock *, isl::set> &InvalidDomainMap) {
  // Check if the block @p BB is the entry of a region. If so we propagate it's
  // domain to the exit block of the region. Otherwise we are done.
  auto *RI = scop->getRegion().getRegionInfo();
  auto *BBReg = RI ? RI->getRegionFor(BB) : nullptr;
  auto *ExitBB = BBReg ? BBReg->getExit() : nullptr;
  if (!BBReg || BBReg->getEntry() != BB || !scop->contains(ExitBB))
    return;

  // Do not propagate the domain if there is a loop backedge inside the region
  // that would prevent the exit block from being executed.
  auto *L = BBLoop;
  while (L && scop->contains(L)) {
    SmallVector<BasicBlock *, 4> LatchBBs;
    BBLoop->getLoopLatches(LatchBBs);
    for (auto *LatchBB : LatchBBs)
      if (BB != LatchBB && BBReg->contains(LatchBB))
        return;
    L = L->getParentLoop();
  }

  isl::set Domain = scop->getOrInitEmptyDomain(BB);
  assert(Domain && "Cannot propagate a nullptr");

  Loop *ExitBBLoop = getFirstNonBoxedLoopFor(ExitBB, LI, scop->getBoxedLoops());

  // Since the dimensions of @p BB and @p ExitBB might be different we have to
  // adjust the domain before we can propagate it.
  isl::set AdjustedDomain = adjustDomainDimensions(Domain, BBLoop, ExitBBLoop);
  isl::set &ExitDomain = scop->getOrInitEmptyDomain(ExitBB);

  // If the exit domain is not yet created we set it otherwise we "add" the
  // current domain.
  ExitDomain = ExitDomain ? AdjustedDomain.unite(ExitDomain) : AdjustedDomain;

  // Initialize the invalid domain.
  InvalidDomainMap[ExitBB] = ExitDomain.empty(ExitDomain.get_space());

  FinishedExitBlocks.insert(ExitBB);
}

isl::set ScopBuilder::getPredecessorDomainConstraints(BasicBlock *BB,
                                                      isl::set Domain) {
  // If @p BB is the ScopEntry we are done
  if (scop->getRegion().getEntry() == BB)
    return isl::set::universe(Domain.get_space());

  // The region info of this function.
  auto &RI = *scop->getRegion().getRegionInfo();

  Loop *BBLoop = getFirstNonBoxedLoopFor(BB, LI, scop->getBoxedLoops());

  // A domain to collect all predecessor domains, thus all conditions under
  // which the block is executed. To this end we start with the empty domain.
  isl::set PredDom = isl::set::empty(Domain.get_space());

  // Set of regions of which the entry block domain has been propagated to BB.
  // all predecessors inside any of the regions can be skipped.
  SmallSet<Region *, 8> PropagatedRegions;

  for (auto *PredBB : predecessors(BB)) {
    // Skip backedges.
    if (DT.dominates(BB, PredBB))
      continue;

    // If the predecessor is in a region we used for propagation we can skip it.
    auto PredBBInRegion = [PredBB](Region *PR) { return PR->contains(PredBB); };
    if (std::any_of(PropagatedRegions.begin(), PropagatedRegions.end(),
                    PredBBInRegion)) {
      continue;
    }

    // Check if there is a valid region we can use for propagation, thus look
    // for a region that contains the predecessor and has @p BB as exit block.
    auto *PredR = RI.getRegionFor(PredBB);
    while (PredR->getExit() != BB && !PredR->contains(BB))
      PredR->getParent();

    // If a valid region for propagation was found use the entry of that region
    // for propagation, otherwise the PredBB directly.
    if (PredR->getExit() == BB) {
      PredBB = PredR->getEntry();
      PropagatedRegions.insert(PredR);
    }

    isl::set PredBBDom = scop->getDomainConditions(PredBB);
    Loop *PredBBLoop =
        getFirstNonBoxedLoopFor(PredBB, LI, scop->getBoxedLoops());
    PredBBDom = adjustDomainDimensions(PredBBDom, PredBBLoop, BBLoop);
    PredDom = PredDom.unite(PredBBDom);
  }

  return PredDom;
}

bool ScopBuilder::addLoopBoundsToHeaderDomain(
    Loop *L, DenseMap<BasicBlock *, isl::set> &InvalidDomainMap) {
  int LoopDepth = scop->getRelativeLoopDepth(L);
  assert(LoopDepth >= 0 && "Loop in region should have at least depth one");

  BasicBlock *HeaderBB = L->getHeader();
  assert(scop->isDomainDefined(HeaderBB));
  isl::set &HeaderBBDom = scop->getOrInitEmptyDomain(HeaderBB);

  isl::map NextIterationMap =
      createNextIterationMap(HeaderBBDom.get_space(), LoopDepth);

  isl::set UnionBackedgeCondition = HeaderBBDom.empty(HeaderBBDom.get_space());

  SmallVector<BasicBlock *, 4> LatchBlocks;
  L->getLoopLatches(LatchBlocks);

  for (BasicBlock *LatchBB : LatchBlocks) {
    // If the latch is only reachable via error statements we skip it.
    if (!scop->isDomainDefined(LatchBB))
      continue;

    isl::set LatchBBDom = scop->getDomainConditions(LatchBB);

    isl::set BackedgeCondition = nullptr;

    Instruction *TI = LatchBB->getTerminator();
    BranchInst *BI = dyn_cast<BranchInst>(TI);
    assert(BI && "Only branch instructions allowed in loop latches");

    if (BI->isUnconditional())
      BackedgeCondition = LatchBBDom;
    else {
      SmallVector<isl_set *, 8> ConditionSets;
      int idx = BI->getSuccessor(0) != HeaderBB;
      if (!buildConditionSets(LatchBB, TI, L, LatchBBDom.get(),
                              InvalidDomainMap, ConditionSets))
        return false;

      // Free the non back edge condition set as we do not need it.
      isl_set_free(ConditionSets[1 - idx]);

      BackedgeCondition = isl::manage(ConditionSets[idx]);
    }

    int LatchLoopDepth = scop->getRelativeLoopDepth(LI.getLoopFor(LatchBB));
    assert(LatchLoopDepth >= LoopDepth);
    BackedgeCondition = BackedgeCondition.project_out(
        isl::dim::set, LoopDepth + 1, LatchLoopDepth - LoopDepth);
    UnionBackedgeCondition = UnionBackedgeCondition.unite(BackedgeCondition);
  }

  isl::map ForwardMap = ForwardMap.lex_le(HeaderBBDom.get_space());
  for (int i = 0; i < LoopDepth; i++)
    ForwardMap = ForwardMap.equate(isl::dim::in, i, isl::dim::out, i);

  isl::set UnionBackedgeConditionComplement =
      UnionBackedgeCondition.complement();
  UnionBackedgeConditionComplement =
      UnionBackedgeConditionComplement.lower_bound_si(isl::dim::set, LoopDepth,
                                                      0);
  UnionBackedgeConditionComplement =
      UnionBackedgeConditionComplement.apply(ForwardMap);
  HeaderBBDom = HeaderBBDom.subtract(UnionBackedgeConditionComplement);
  HeaderBBDom = HeaderBBDom.apply(NextIterationMap);

  auto Parts = partitionSetParts(HeaderBBDom, LoopDepth);
  HeaderBBDom = Parts.second;

  // Check if there is a <nsw> tagged AddRec for this loop and if so do not add
  // the bounded assumptions to the context as they are already implied by the
  // <nsw> tag.
  if (scop->hasNSWAddRecForLoop(L))
    return true;

  isl::set UnboundedCtx = Parts.first.params();
  recordAssumption(&RecordedAssumptions, INFINITELOOP, UnboundedCtx,
                   HeaderBB->getTerminator()->getDebugLoc(), AS_RESTRICTION);
  return true;
}

void ScopBuilder::buildInvariantEquivalenceClasses() {
  DenseMap<std::pair<const SCEV *, Type *>, LoadInst *> EquivClasses;

  const InvariantLoadsSetTy &RIL = scop->getRequiredInvariantLoads();
  for (LoadInst *LInst : RIL) {
    const SCEV *PointerSCEV = SE.getSCEV(LInst->getPointerOperand());

    Type *Ty = LInst->getType();
    LoadInst *&ClassRep = EquivClasses[std::make_pair(PointerSCEV, Ty)];
    if (ClassRep) {
      scop->addInvariantLoadMapping(LInst, ClassRep);
      continue;
    }

    ClassRep = LInst;
    scop->addInvariantEquivClass(
        InvariantEquivClassTy{PointerSCEV, MemoryAccessList(), nullptr, Ty});
  }
}

bool ScopBuilder::buildDomains(
    Region *R, DenseMap<BasicBlock *, isl::set> &InvalidDomainMap) {
  bool IsOnlyNonAffineRegion = scop->isNonAffineSubRegion(R);
  auto *EntryBB = R->getEntry();
  auto *L = IsOnlyNonAffineRegion ? nullptr : LI.getLoopFor(EntryBB);
  int LD = scop->getRelativeLoopDepth(L);
  auto *S =
      isl_set_universe(isl_space_set_alloc(scop->getIslCtx().get(), 0, LD + 1));

  InvalidDomainMap[EntryBB] = isl::manage(isl_set_empty(isl_set_get_space(S)));
  isl::noexceptions::set Domain = isl::manage(S);
  scop->setDomain(EntryBB, Domain);

  if (IsOnlyNonAffineRegion)
    return !containsErrorBlock(R->getNode(), *R, LI, DT);

  if (!buildDomainsWithBranchConstraints(R, InvalidDomainMap))
    return false;

  if (!propagateDomainConstraints(R, InvalidDomainMap))
    return false;

  // Error blocks and blocks dominated by them have been assumed to never be
  // executed. Representing them in the Scop does not add any value. In fact,
  // it is likely to cause issues during construction of the ScopStmts. The
  // contents of error blocks have not been verified to be expressible and
  // will cause problems when building up a ScopStmt for them.
  // Furthermore, basic blocks dominated by error blocks may reference
  // instructions in the error block which, if the error block is not modeled,
  // can themselves not be constructed properly. To this end we will replace
  // the domains of error blocks and those only reachable via error blocks
  // with an empty set. Additionally, we will record for each block under which
  // parameter combination it would be reached via an error block in its
  // InvalidDomain. This information is needed during load hoisting.
  if (!propagateInvalidStmtDomains(R, InvalidDomainMap))
    return false;

  return true;
}

bool ScopBuilder::buildDomainsWithBranchConstraints(
    Region *R, DenseMap<BasicBlock *, isl::set> &InvalidDomainMap) {
  // To create the domain for each block in R we iterate over all blocks and
  // subregions in R and propagate the conditions under which the current region
  // element is executed. To this end we iterate in reverse post order over R as
  // it ensures that we first visit all predecessors of a region node (either a
  // basic block or a subregion) before we visit the region node itself.
  // Initially, only the domain for the SCoP region entry block is set and from
  // there we propagate the current domain to all successors, however we add the
  // condition that the successor is actually executed next.
  // As we are only interested in non-loop carried constraints here we can
  // simply skip loop back edges.

  SmallPtrSet<BasicBlock *, 8> FinishedExitBlocks;
  ReversePostOrderTraversal<Region *> RTraversal(R);
  for (auto *RN : RTraversal) {
    // Recurse for affine subregions but go on for basic blocks and non-affine
    // subregions.
    if (RN->isSubRegion()) {
      Region *SubRegion = RN->getNodeAs<Region>();
      if (!scop->isNonAffineSubRegion(SubRegion)) {
        if (!buildDomainsWithBranchConstraints(SubRegion, InvalidDomainMap))
          return false;
        continue;
      }
    }

    if (containsErrorBlock(RN, scop->getRegion(), LI, DT))
      scop->notifyErrorBlock();
    ;

    BasicBlock *BB = getRegionNodeBasicBlock(RN);
    Instruction *TI = BB->getTerminator();

    if (isa<UnreachableInst>(TI))
      continue;

    if (!scop->isDomainDefined(BB))
      continue;
    isl::set Domain = scop->getDomainConditions(BB);

    scop->updateMaxLoopDepth(isl_set_n_dim(Domain.get()));

    auto *BBLoop = getRegionNodeLoop(RN, LI);
    // Propagate the domain from BB directly to blocks that have a superset
    // domain, at the moment only region exit nodes of regions that start in BB.
    propagateDomainConstraintsToRegionExit(BB, BBLoop, FinishedExitBlocks,
                                           InvalidDomainMap);

    // If all successors of BB have been set a domain through the propagation
    // above we do not need to build condition sets but can just skip this
    // block. However, it is important to note that this is a local property
    // with regards to the region @p R. To this end FinishedExitBlocks is a
    // local variable.
    auto IsFinishedRegionExit = [&FinishedExitBlocks](BasicBlock *SuccBB) {
      return FinishedExitBlocks.count(SuccBB);
    };
    if (std::all_of(succ_begin(BB), succ_end(BB), IsFinishedRegionExit))
      continue;

    // Build the condition sets for the successor nodes of the current region
    // node. If it is a non-affine subregion we will always execute the single
    // exit node, hence the single entry node domain is the condition set. For
    // basic blocks we use the helper function buildConditionSets.
    SmallVector<isl_set *, 8> ConditionSets;
    if (RN->isSubRegion())
      ConditionSets.push_back(Domain.copy());
    else if (!buildConditionSets(BB, TI, BBLoop, Domain.get(), InvalidDomainMap,
                                 ConditionSets))
      return false;

    // Now iterate over the successors and set their initial domain based on
    // their condition set. We skip back edges here and have to be careful when
    // we leave a loop not to keep constraints over a dimension that doesn't
    // exist anymore.
    assert(RN->isSubRegion() || TI->getNumSuccessors() == ConditionSets.size());
    for (unsigned u = 0, e = ConditionSets.size(); u < e; u++) {
      isl::set CondSet = isl::manage(ConditionSets[u]);
      BasicBlock *SuccBB = getRegionNodeSuccessor(RN, TI, u);

      // Skip blocks outside the region.
      if (!scop->contains(SuccBB))
        continue;

      // If we propagate the domain of some block to "SuccBB" we do not have to
      // adjust the domain.
      if (FinishedExitBlocks.count(SuccBB))
        continue;

      // Skip back edges.
      if (DT.dominates(SuccBB, BB))
        continue;

      Loop *SuccBBLoop =
          getFirstNonBoxedLoopFor(SuccBB, LI, scop->getBoxedLoops());

      CondSet = adjustDomainDimensions(CondSet, BBLoop, SuccBBLoop);

      // Set the domain for the successor or merge it with an existing domain in
      // case there are multiple paths (without loop back edges) to the
      // successor block.
      isl::set &SuccDomain = scop->getOrInitEmptyDomain(SuccBB);

      if (SuccDomain) {
        SuccDomain = SuccDomain.unite(CondSet).coalesce();
      } else {
        // Initialize the invalid domain.
        InvalidDomainMap[SuccBB] = CondSet.empty(CondSet.get_space());
        SuccDomain = CondSet;
      }

      SuccDomain = SuccDomain.detect_equalities();

      // Check if the maximal number of domain disjunctions was reached.
      // In case this happens we will clean up and bail.
      if (SuccDomain.n_basic_set() < MaxDisjunctsInDomain)
        continue;

      scop->invalidate(COMPLEXITY, DebugLoc());
      while (++u < ConditionSets.size())
        isl_set_free(ConditionSets[u]);
      return false;
    }
  }

  return true;
}

bool ScopBuilder::propagateInvalidStmtDomains(
    Region *R, DenseMap<BasicBlock *, isl::set> &InvalidDomainMap) {
  ReversePostOrderTraversal<Region *> RTraversal(R);
  for (auto *RN : RTraversal) {

    // Recurse for affine subregions but go on for basic blocks and non-affine
    // subregions.
    if (RN->isSubRegion()) {
      Region *SubRegion = RN->getNodeAs<Region>();
      if (!scop->isNonAffineSubRegion(SubRegion)) {
        propagateInvalidStmtDomains(SubRegion, InvalidDomainMap);
        continue;
      }
    }

    bool ContainsErrorBlock = containsErrorBlock(RN, scop->getRegion(), LI, DT);
    BasicBlock *BB = getRegionNodeBasicBlock(RN);
    isl::set &Domain = scop->getOrInitEmptyDomain(BB);
    assert(Domain && "Cannot propagate a nullptr");

    isl::set InvalidDomain = InvalidDomainMap[BB];

    bool IsInvalidBlock = ContainsErrorBlock || Domain.is_subset(InvalidDomain);

    if (!IsInvalidBlock) {
      InvalidDomain = InvalidDomain.intersect(Domain);
    } else {
      InvalidDomain = Domain;
      isl::set DomPar = Domain.params();
      recordAssumption(&RecordedAssumptions, ERRORBLOCK, DomPar,
                       BB->getTerminator()->getDebugLoc(), AS_RESTRICTION);
      Domain = isl::set::empty(Domain.get_space());
    }

    if (InvalidDomain.is_empty()) {
      InvalidDomainMap[BB] = InvalidDomain;
      continue;
    }

    auto *BBLoop = getRegionNodeLoop(RN, LI);
    auto *TI = BB->getTerminator();
    unsigned NumSuccs = RN->isSubRegion() ? 1 : TI->getNumSuccessors();
    for (unsigned u = 0; u < NumSuccs; u++) {
      auto *SuccBB = getRegionNodeSuccessor(RN, TI, u);

      // Skip successors outside the SCoP.
      if (!scop->contains(SuccBB))
        continue;

      // Skip backedges.
      if (DT.dominates(SuccBB, BB))
        continue;

      Loop *SuccBBLoop =
          getFirstNonBoxedLoopFor(SuccBB, LI, scop->getBoxedLoops());

      auto AdjustedInvalidDomain =
          adjustDomainDimensions(InvalidDomain, BBLoop, SuccBBLoop);

      isl::set SuccInvalidDomain = InvalidDomainMap[SuccBB];
      SuccInvalidDomain = SuccInvalidDomain.unite(AdjustedInvalidDomain);
      SuccInvalidDomain = SuccInvalidDomain.coalesce();

      InvalidDomainMap[SuccBB] = SuccInvalidDomain;

      // Check if the maximal number of domain disjunctions was reached.
      // In case this happens we will bail.
      if (SuccInvalidDomain.n_basic_set() < MaxDisjunctsInDomain)
        continue;

      InvalidDomainMap.erase(BB);
      scop->invalidate(COMPLEXITY, TI->getDebugLoc(), TI->getParent());
      return false;
    }

    InvalidDomainMap[BB] = InvalidDomain;
  }

  return true;
}

void ScopBuilder::buildPHIAccesses(ScopStmt *PHIStmt, PHINode *PHI,
                                   Region *NonAffineSubRegion,
                                   bool IsExitBlock) {
  // PHI nodes that are in the exit block of the region, hence if IsExitBlock is
  // true, are not modeled as ordinary PHI nodes as they are not part of the
  // region. However, we model the operands in the predecessor blocks that are
  // part of the region as regular scalar accesses.

  // If we can synthesize a PHI we can skip it, however only if it is in
  // the region. If it is not it can only be in the exit block of the region.
  // In this case we model the operands but not the PHI itself.
  auto *Scope = LI.getLoopFor(PHI->getParent());
  if (!IsExitBlock && canSynthesize(PHI, *scop, &SE, Scope))
    return;

  // PHI nodes are modeled as if they had been demoted prior to the SCoP
  // detection. Hence, the PHI is a load of a new memory location in which the
  // incoming value was written at the end of the incoming basic block.
  bool OnlyNonAffineSubRegionOperands = true;
  for (unsigned u = 0; u < PHI->getNumIncomingValues(); u++) {
    Value *Op = PHI->getIncomingValue(u);
    BasicBlock *OpBB = PHI->getIncomingBlock(u);
    ScopStmt *OpStmt = scop->getIncomingStmtFor(PHI->getOperandUse(u));

    // Do not build PHI dependences inside a non-affine subregion, but make
    // sure that the necessary scalar values are still made available.
    if (NonAffineSubRegion && NonAffineSubRegion->contains(OpBB)) {
      auto *OpInst = dyn_cast<Instruction>(Op);
      if (!OpInst || !NonAffineSubRegion->contains(OpInst))
        ensureValueRead(Op, OpStmt);
      continue;
    }

    OnlyNonAffineSubRegionOperands = false;
    ensurePHIWrite(PHI, OpStmt, OpBB, Op, IsExitBlock);
  }

  if (!OnlyNonAffineSubRegionOperands && !IsExitBlock) {
    addPHIReadAccess(PHIStmt, PHI);
  }
}

void ScopBuilder::buildScalarDependences(ScopStmt *UserStmt,
                                         Instruction *Inst) {
  assert(!isa<PHINode>(Inst));

  // Pull-in required operands.
  for (Use &Op : Inst->operands())
    ensureValueRead(Op.get(), UserStmt);
}

// Create a sequence of two schedules. Either argument may be null and is
// interpreted as the empty schedule. Can also return null if both schedules are
// empty.
static isl::schedule combineInSequence(isl::schedule Prev, isl::schedule Succ) {
  if (!Prev)
    return Succ;
  if (!Succ)
    return Prev;

  return Prev.sequence(Succ);
}

// Create an isl_multi_union_aff that defines an identity mapping from the
// elements of USet to their N-th dimension.
//
// # Example:
//
//            Domain: { A[i,j]; B[i,j,k] }
//                 N: 1
//
// Resulting Mapping: { {A[i,j] -> [(j)]; B[i,j,k] -> [(j)] }
//
// @param USet   A union set describing the elements for which to generate a
//               mapping.
// @param N      The dimension to map to.
// @returns      A mapping from USet to its N-th dimension.
static isl::multi_union_pw_aff mapToDimension(isl::union_set USet, int N) {
  assert(N >= 0);
  assert(USet);
  assert(!USet.is_empty());

  auto Result = isl::union_pw_multi_aff::empty(USet.get_space());

  for (isl::set S : USet.get_set_list()) {
    int Dim = S.dim(isl::dim::set);
    auto PMA = isl::pw_multi_aff::project_out_map(S.get_space(), isl::dim::set,
                                                  N, Dim - N);
    if (N > 1)
      PMA = PMA.drop_dims(isl::dim::out, 0, N - 1);

    Result = Result.add_pw_multi_aff(PMA);
  }

  return isl::multi_union_pw_aff(isl::union_pw_multi_aff(Result));
}

void ScopBuilder::buildSchedule() {
  Loop *L = getLoopSurroundingScop(*scop, LI);
  LoopStackTy LoopStack({LoopStackElementTy(L, nullptr, 0)});
  buildSchedule(scop->getRegion().getNode(), LoopStack);
  assert(LoopStack.size() == 1 && LoopStack.back().L == L);
  scop->setScheduleTree(LoopStack[0].Schedule);
}

/// To generate a schedule for the elements in a Region we traverse the Region
/// in reverse-post-order and add the contained RegionNodes in traversal order
/// to the schedule of the loop that is currently at the top of the LoopStack.
/// For loop-free codes, this results in a correct sequential ordering.
///
/// Example:
///           bb1(0)
///         /     \.
///      bb2(1)   bb3(2)
///         \    /  \.
///          bb4(3)  bb5(4)
///             \   /
///              bb6(5)
///
/// Including loops requires additional processing. Whenever a loop header is
/// encountered, the corresponding loop is added to the @p LoopStack. Starting
/// from an empty schedule, we first process all RegionNodes that are within
/// this loop and complete the sequential schedule at this loop-level before
/// processing about any other nodes. To implement this
/// loop-nodes-first-processing, the reverse post-order traversal is
/// insufficient. Hence, we additionally check if the traversal yields
/// sub-regions or blocks that are outside the last loop on the @p LoopStack.
/// These region-nodes are then queue and only traverse after the all nodes
/// within the current loop have been processed.
void ScopBuilder::buildSchedule(Region *R, LoopStackTy &LoopStack) {
  Loop *OuterScopLoop = getLoopSurroundingScop(*scop, LI);

  ReversePostOrderTraversal<Region *> RTraversal(R);
  std::deque<RegionNode *> WorkList(RTraversal.begin(), RTraversal.end());
  std::deque<RegionNode *> DelayList;
  bool LastRNWaiting = false;

  // Iterate over the region @p R in reverse post-order but queue
  // sub-regions/blocks iff they are not part of the last encountered but not
  // completely traversed loop. The variable LastRNWaiting is a flag to indicate
  // that we queued the last sub-region/block from the reverse post-order
  // iterator. If it is set we have to explore the next sub-region/block from
  // the iterator (if any) to guarantee progress. If it is not set we first try
  // the next queued sub-region/blocks.
  while (!WorkList.empty() || !DelayList.empty()) {
    RegionNode *RN;

    if ((LastRNWaiting && !WorkList.empty()) || DelayList.empty()) {
      RN = WorkList.front();
      WorkList.pop_front();
      LastRNWaiting = false;
    } else {
      RN = DelayList.front();
      DelayList.pop_front();
    }

    Loop *L = getRegionNodeLoop(RN, LI);
    if (!scop->contains(L))
      L = OuterScopLoop;

    Loop *LastLoop = LoopStack.back().L;
    if (LastLoop != L) {
      if (LastLoop && !LastLoop->contains(L)) {
        LastRNWaiting = true;
        DelayList.push_back(RN);
        continue;
      }
      LoopStack.push_back({L, nullptr, 0});
    }
    buildSchedule(RN, LoopStack);
  }
}

void ScopBuilder::buildSchedule(RegionNode *RN, LoopStackTy &LoopStack) {
  if (RN->isSubRegion()) {
    auto *LocalRegion = RN->getNodeAs<Region>();
    if (!scop->isNonAffineSubRegion(LocalRegion)) {
      buildSchedule(LocalRegion, LoopStack);
      return;
    }
  }

  assert(LoopStack.rbegin() != LoopStack.rend());
  auto LoopData = LoopStack.rbegin();
  LoopData->NumBlocksProcessed += getNumBlocksInRegionNode(RN);

  for (auto *Stmt : scop->getStmtListFor(RN)) {
    isl::union_set UDomain{Stmt->getDomain()};
    auto StmtSchedule = isl::schedule::from_domain(UDomain);
    LoopData->Schedule = combineInSequence(LoopData->Schedule, StmtSchedule);
  }

  // Check if we just processed the last node in this loop. If we did, finalize
  // the loop by:
  //
  //   - adding new schedule dimensions
  //   - folding the resulting schedule into the parent loop schedule
  //   - dropping the loop schedule from the LoopStack.
  //
  // Then continue to check surrounding loops, which might also have been
  // completed by this node.
  size_t Dimension = LoopStack.size();
  while (LoopData->L &&
         LoopData->NumBlocksProcessed == getNumBlocksInLoop(LoopData->L)) {
    isl::schedule Schedule = LoopData->Schedule;
    auto NumBlocksProcessed = LoopData->NumBlocksProcessed;

    assert(std::next(LoopData) != LoopStack.rend());
    ++LoopData;
    --Dimension;

    if (Schedule) {
      isl::union_set Domain = Schedule.get_domain();
      isl::multi_union_pw_aff MUPA = mapToDimension(Domain, Dimension);
      Schedule = Schedule.insert_partial_schedule(MUPA);
      LoopData->Schedule = combineInSequence(LoopData->Schedule, Schedule);
    }

    LoopData->NumBlocksProcessed += NumBlocksProcessed;
  }
  // Now pop all loops processed up there from the LoopStack
  LoopStack.erase(LoopStack.begin() + Dimension, LoopStack.end());
}

void ScopBuilder::buildEscapingDependences(Instruction *Inst) {
  // Check for uses of this instruction outside the scop. Because we do not
  // iterate over such instructions and therefore did not "ensure" the existence
  // of a write, we must determine such use here.
  if (scop->isEscaping(Inst))
    ensureValueWrite(Inst);
}

/// Check that a value is a Fortran Array descriptor.
///
/// We check if V has the following structure:
/// %"struct.array1_real(kind=8)" = type { i8*, i<zz>, i<zz>,
///                                   [<num> x %struct.descriptor_dimension] }
///
///
/// %struct.descriptor_dimension = type { i<zz>, i<zz>, i<zz> }
///
/// 1. V's type name starts with "struct.array"
/// 2. V's type has layout as shown.
/// 3. Final member of V's type has name "struct.descriptor_dimension",
/// 4. "struct.descriptor_dimension" has layout as shown.
/// 5. Consistent use of i<zz> where <zz> is some fixed integer number.
///
/// We are interested in such types since this is the code that dragonegg
/// generates for Fortran array descriptors.
///
/// @param V the Value to be checked.
///
/// @returns True if V is a Fortran array descriptor, False otherwise.
bool isFortranArrayDescriptor(Value *V) {
  PointerType *PTy = dyn_cast<PointerType>(V->getType());

  if (!PTy)
    return false;

  Type *Ty = PTy->getElementType();
  assert(Ty && "Ty expected to be initialized");
  auto *StructArrTy = dyn_cast<StructType>(Ty);

  if (!(StructArrTy && StructArrTy->hasName()))
    return false;

  if (!StructArrTy->getName().startswith("struct.array"))
    return false;

  if (StructArrTy->getNumElements() != 4)
    return false;

  const ArrayRef<Type *> ArrMemberTys = StructArrTy->elements();

  // i8* match
  if (ArrMemberTys[0] != Type::getInt8PtrTy(V->getContext()))
    return false;

  // Get a reference to the int type and check that all the members
  // share the same int type
  Type *IntTy = ArrMemberTys[1];
  if (ArrMemberTys[2] != IntTy)
    return false;

  // type: [<num> x %struct.descriptor_dimension]
  ArrayType *DescriptorDimArrayTy = dyn_cast<ArrayType>(ArrMemberTys[3]);
  if (!DescriptorDimArrayTy)
    return false;

  // type: %struct.descriptor_dimension := type { ixx, ixx, ixx }
  StructType *DescriptorDimTy =
      dyn_cast<StructType>(DescriptorDimArrayTy->getElementType());

  if (!(DescriptorDimTy && DescriptorDimTy->hasName()))
    return false;

  if (DescriptorDimTy->getName() != "struct.descriptor_dimension")
    return false;

  if (DescriptorDimTy->getNumElements() != 3)
    return false;

  for (auto MemberTy : DescriptorDimTy->elements()) {
    if (MemberTy != IntTy)
      return false;
  }

  return true;
}

Value *ScopBuilder::findFADAllocationVisible(MemAccInst Inst) {
  // match: 4.1 & 4.2 store/load
  if (!isa<LoadInst>(Inst) && !isa<StoreInst>(Inst))
    return nullptr;

  // match: 4
  if (Inst.getAlignment() != 8)
    return nullptr;

  Value *Address = Inst.getPointerOperand();

  const BitCastInst *Bitcast = nullptr;
  // [match: 3]
  if (auto *Slot = dyn_cast<GetElementPtrInst>(Address)) {
    Value *TypedMem = Slot->getPointerOperand();
    // match: 2
    Bitcast = dyn_cast<BitCastInst>(TypedMem);
  } else {
    // match: 2
    Bitcast = dyn_cast<BitCastInst>(Address);
  }

  if (!Bitcast)
    return nullptr;

  auto *MallocMem = Bitcast->getOperand(0);

  // match: 1
  auto *MallocCall = dyn_cast<CallInst>(MallocMem);
  if (!MallocCall)
    return nullptr;

  Function *MallocFn = MallocCall->getCalledFunction();
  if (!(MallocFn && MallocFn->hasName() && MallocFn->getName() == "malloc"))
    return nullptr;

  // Find all uses the malloc'd memory.
  // We are looking for a "store" into a struct with the type being the Fortran
  // descriptor type
  for (auto user : MallocMem->users()) {
    /// match: 5
    auto *MallocStore = dyn_cast<StoreInst>(user);
    if (!MallocStore)
      continue;

    auto *DescriptorGEP =
        dyn_cast<GEPOperator>(MallocStore->getPointerOperand());
    if (!DescriptorGEP)
      continue;

    // match: 5
    auto DescriptorType =
        dyn_cast<StructType>(DescriptorGEP->getSourceElementType());
    if (!(DescriptorType && DescriptorType->hasName()))
      continue;

    Value *Descriptor = dyn_cast<Value>(DescriptorGEP->getPointerOperand());

    if (!Descriptor)
      continue;

    if (!isFortranArrayDescriptor(Descriptor))
      continue;

    return Descriptor;
  }

  return nullptr;
}

Value *ScopBuilder::findFADAllocationInvisible(MemAccInst Inst) {
  // match: 3
  if (!isa<LoadInst>(Inst) && !isa<StoreInst>(Inst))
    return nullptr;

  Value *Slot = Inst.getPointerOperand();

  LoadInst *MemLoad = nullptr;
  // [match: 2]
  if (auto *SlotGEP = dyn_cast<GetElementPtrInst>(Slot)) {
    // match: 1
    MemLoad = dyn_cast<LoadInst>(SlotGEP->getPointerOperand());
  } else {
    // match: 1
    MemLoad = dyn_cast<LoadInst>(Slot);
  }

  if (!MemLoad)
    return nullptr;

  auto *BitcastOperator =
      dyn_cast<BitCastOperator>(MemLoad->getPointerOperand());
  if (!BitcastOperator)
    return nullptr;

  Value *Descriptor = dyn_cast<Value>(BitcastOperator->getOperand(0));
  if (!Descriptor)
    return nullptr;

  if (!isFortranArrayDescriptor(Descriptor))
    return nullptr;

  return Descriptor;
}

void ScopBuilder::addRecordedAssumptions() {
  for (auto &AS : llvm::reverse(RecordedAssumptions)) {

    if (!AS.BB) {
      scop->addAssumption(AS.Kind, AS.Set, AS.Loc, AS.Sign,
                          nullptr /* BasicBlock */);
      continue;
    }

    // If the domain was deleted the assumptions are void.
    isl_set *Dom = scop->getDomainConditions(AS.BB).release();
    if (!Dom)
      continue;

    // If a basic block was given use its domain to simplify the assumption.
    // In case of restrictions we know they only have to hold on the domain,
    // thus we can intersect them with the domain of the block. However, for
    // assumptions the domain has to imply them, thus:
    //                     _              _____
    //   Dom => S   <==>   A v B   <==>   A - B
    //
    // To avoid the complement we will register A - B as a restriction not an
    // assumption.
    isl_set *S = AS.Set.copy();
    if (AS.Sign == AS_RESTRICTION)
      S = isl_set_params(isl_set_intersect(S, Dom));
    else /* (AS.Sign == AS_ASSUMPTION) */
      S = isl_set_params(isl_set_subtract(Dom, S));

    scop->addAssumption(AS.Kind, isl::manage(S), AS.Loc, AS_RESTRICTION, AS.BB);
  }
}

void ScopBuilder::addUserAssumptions(
    AssumptionCache &AC, DenseMap<BasicBlock *, isl::set> &InvalidDomainMap) {
  for (auto &Assumption : AC.assumptions()) {
    auto *CI = dyn_cast_or_null<CallInst>(Assumption);
    if (!CI || CI->getNumArgOperands() != 1)
      continue;

    bool InScop = scop->contains(CI);
    if (!InScop && !scop->isDominatedBy(DT, CI->getParent()))
      continue;

    auto *L = LI.getLoopFor(CI->getParent());
    auto *Val = CI->getArgOperand(0);
    ParameterSetTy DetectedParams;
    auto &R = scop->getRegion();
    if (!isAffineConstraint(Val, &R, L, SE, DetectedParams)) {
      ORE.emit(
          OptimizationRemarkAnalysis(DEBUG_TYPE, "IgnoreUserAssumption", CI)
          << "Non-affine user assumption ignored.");
      continue;
    }

    // Collect all newly introduced parameters.
    ParameterSetTy NewParams;
    for (auto *Param : DetectedParams) {
      Param = extractConstantFactor(Param, SE).second;
      Param = scop->getRepresentingInvariantLoadSCEV(Param);
      if (scop->isParam(Param))
        continue;
      NewParams.insert(Param);
    }

    SmallVector<isl_set *, 2> ConditionSets;
    auto *TI = InScop ? CI->getParent()->getTerminator() : nullptr;
    BasicBlock *BB = InScop ? CI->getParent() : R.getEntry();
    auto *Dom = InScop ? isl_set_copy(scop->getDomainConditions(BB).get())
                       : isl_set_copy(scop->getContext().get());
    assert(Dom && "Cannot propagate a nullptr.");
    bool Valid = buildConditionSets(BB, Val, TI, L, Dom, InvalidDomainMap,
                                    ConditionSets);
    isl_set_free(Dom);

    if (!Valid)
      continue;

    isl_set *AssumptionCtx = nullptr;
    if (InScop) {
      AssumptionCtx = isl_set_complement(isl_set_params(ConditionSets[1]));
      isl_set_free(ConditionSets[0]);
    } else {
      AssumptionCtx = isl_set_complement(ConditionSets[1]);
      AssumptionCtx = isl_set_intersect(AssumptionCtx, ConditionSets[0]);
    }

    // Project out newly introduced parameters as they are not otherwise useful.
    if (!NewParams.empty()) {
      for (isl_size u = 0; u < isl_set_n_param(AssumptionCtx); u++) {
        auto *Id = isl_set_get_dim_id(AssumptionCtx, isl_dim_param, u);
        auto *Param = static_cast<const SCEV *>(isl_id_get_user(Id));
        isl_id_free(Id);

        if (!NewParams.count(Param))
          continue;

        AssumptionCtx =
            isl_set_project_out(AssumptionCtx, isl_dim_param, u--, 1);
      }
    }
    ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "UserAssumption", CI)
             << "Use user assumption: " << stringFromIslObj(AssumptionCtx));
    isl::set newContext =
        scop->getContext().intersect(isl::manage(AssumptionCtx));
    scop->setContext(newContext);
  }
}

bool ScopBuilder::buildAccessMultiDimFixed(MemAccInst Inst, ScopStmt *Stmt) {
  Value *Val = Inst.getValueOperand();
  Type *ElementType = Val->getType();
  Value *Address = Inst.getPointerOperand();
  const SCEV *AccessFunction =
      SE.getSCEVAtScope(Address, LI.getLoopFor(Inst->getParent()));
  const SCEVUnknown *BasePointer =
      dyn_cast<SCEVUnknown>(SE.getPointerBase(AccessFunction));
  enum MemoryAccess::AccessType AccType =
      isa<LoadInst>(Inst) ? MemoryAccess::READ : MemoryAccess::MUST_WRITE;

  if (auto *BitCast = dyn_cast<BitCastInst>(Address)) {
    auto *Src = BitCast->getOperand(0);
    auto *SrcTy = Src->getType();
    auto *DstTy = BitCast->getType();
    // Do not try to delinearize non-sized (opaque) pointers.
    if ((SrcTy->isPointerTy() && !SrcTy->getPointerElementType()->isSized()) ||
        (DstTy->isPointerTy() && !DstTy->getPointerElementType()->isSized())) {
      return false;
    }
    if (SrcTy->isPointerTy() && DstTy->isPointerTy() &&
        DL.getTypeAllocSize(SrcTy->getPointerElementType()) ==
            DL.getTypeAllocSize(DstTy->getPointerElementType()))
      Address = Src;
  }

  auto *GEP = dyn_cast<GetElementPtrInst>(Address);
  if (!GEP)
    return false;

  SmallVector<const SCEV *, 4> Subscripts;
  SmallVector<int, 4> Sizes;
  SE.getIndexExpressionsFromGEP(GEP, Subscripts, Sizes);
  auto *BasePtr = GEP->getOperand(0);

  if (auto *BasePtrCast = dyn_cast<BitCastInst>(BasePtr))
    BasePtr = BasePtrCast->getOperand(0);

  // Check for identical base pointers to ensure that we do not miss index
  // offsets that have been added before this GEP is applied.
  if (BasePtr != BasePointer->getValue())
    return false;

  std::vector<const SCEV *> SizesSCEV;

  const InvariantLoadsSetTy &ScopRIL = scop->getRequiredInvariantLoads();

  Loop *SurroundingLoop = Stmt->getSurroundingLoop();
  for (auto *Subscript : Subscripts) {
    InvariantLoadsSetTy AccessILS;
    if (!isAffineExpr(&scop->getRegion(), SurroundingLoop, Subscript, SE,
                      &AccessILS))
      return false;

    for (LoadInst *LInst : AccessILS)
      if (!ScopRIL.count(LInst))
        return false;
  }

  if (Sizes.empty())
    return false;

  SizesSCEV.push_back(nullptr);

  for (auto V : Sizes)
    SizesSCEV.push_back(SE.getSCEV(
        ConstantInt::get(IntegerType::getInt64Ty(BasePtr->getContext()), V)));

  addArrayAccess(Stmt, Inst, AccType, BasePointer->getValue(), ElementType,
                 true, Subscripts, SizesSCEV, Val);
  return true;
}

bool ScopBuilder::buildAccessMultiDimParam(MemAccInst Inst, ScopStmt *Stmt) {
  if (!PollyDelinearize)
    return false;

  Value *Address = Inst.getPointerOperand();
  Value *Val = Inst.getValueOperand();
  Type *ElementType = Val->getType();
  unsigned ElementSize = DL.getTypeAllocSize(ElementType);
  enum MemoryAccess::AccessType AccType =
      isa<LoadInst>(Inst) ? MemoryAccess::READ : MemoryAccess::MUST_WRITE;

  const SCEV *AccessFunction =
      SE.getSCEVAtScope(Address, LI.getLoopFor(Inst->getParent()));
  const SCEVUnknown *BasePointer =
      dyn_cast<SCEVUnknown>(SE.getPointerBase(AccessFunction));

  assert(BasePointer && "Could not find base pointer");

  auto &InsnToMemAcc = scop->getInsnToMemAccMap();
  auto AccItr = InsnToMemAcc.find(Inst);
  if (AccItr == InsnToMemAcc.end())
    return false;

  std::vector<const SCEV *> Sizes = {nullptr};

  Sizes.insert(Sizes.end(), AccItr->second.Shape->DelinearizedSizes.begin(),
               AccItr->second.Shape->DelinearizedSizes.end());

  // In case only the element size is contained in the 'Sizes' array, the
  // access does not access a real multi-dimensional array. Hence, we allow
  // the normal single-dimensional access construction to handle this.
  if (Sizes.size() == 1)
    return false;

  // Remove the element size. This information is already provided by the
  // ElementSize parameter. In case the element size of this access and the
  // element size used for delinearization differs the delinearization is
  // incorrect. Hence, we invalidate the scop.
  //
  // TODO: Handle delinearization with differing element sizes.
  auto DelinearizedSize =
      cast<SCEVConstant>(Sizes.back())->getAPInt().getSExtValue();
  Sizes.pop_back();
  if (ElementSize != DelinearizedSize)
    scop->invalidate(DELINEARIZATION, Inst->getDebugLoc(), Inst->getParent());

  addArrayAccess(Stmt, Inst, AccType, BasePointer->getValue(), ElementType,
                 true, AccItr->second.DelinearizedSubscripts, Sizes, Val);
  return true;
}

bool ScopBuilder::buildAccessMemIntrinsic(MemAccInst Inst, ScopStmt *Stmt) {
  auto *MemIntr = dyn_cast_or_null<MemIntrinsic>(Inst);

  if (MemIntr == nullptr)
    return false;

  auto *L = LI.getLoopFor(Inst->getParent());
  auto *LengthVal = SE.getSCEVAtScope(MemIntr->getLength(), L);
  assert(LengthVal);

  // Check if the length val is actually affine or if we overapproximate it
  InvariantLoadsSetTy AccessILS;
  const InvariantLoadsSetTy &ScopRIL = scop->getRequiredInvariantLoads();

  Loop *SurroundingLoop = Stmt->getSurroundingLoop();
  bool LengthIsAffine = isAffineExpr(&scop->getRegion(), SurroundingLoop,
                                     LengthVal, SE, &AccessILS);
  for (LoadInst *LInst : AccessILS)
    if (!ScopRIL.count(LInst))
      LengthIsAffine = false;
  if (!LengthIsAffine)
    LengthVal = nullptr;

  auto *DestPtrVal = MemIntr->getDest();
  assert(DestPtrVal);

  auto *DestAccFunc = SE.getSCEVAtScope(DestPtrVal, L);
  assert(DestAccFunc);
  // Ignore accesses to "NULL".
  // TODO: We could use this to optimize the region further, e.g., intersect
  //       the context with
  //          isl_set_complement(isl_set_params(getDomain()))
  //       as we know it would be undefined to execute this instruction anyway.
  if (DestAccFunc->isZero())
    return true;

  auto *DestPtrSCEV = dyn_cast<SCEVUnknown>(SE.getPointerBase(DestAccFunc));
  assert(DestPtrSCEV);
  DestAccFunc = SE.getMinusSCEV(DestAccFunc, DestPtrSCEV);
  addArrayAccess(Stmt, Inst, MemoryAccess::MUST_WRITE, DestPtrSCEV->getValue(),
                 IntegerType::getInt8Ty(DestPtrVal->getContext()),
                 LengthIsAffine, {DestAccFunc, LengthVal}, {nullptr},
                 Inst.getValueOperand());

  auto *MemTrans = dyn_cast<MemTransferInst>(MemIntr);
  if (!MemTrans)
    return true;

  auto *SrcPtrVal = MemTrans->getSource();
  assert(SrcPtrVal);

  auto *SrcAccFunc = SE.getSCEVAtScope(SrcPtrVal, L);
  assert(SrcAccFunc);
  // Ignore accesses to "NULL".
  // TODO: See above TODO
  if (SrcAccFunc->isZero())
    return true;

  auto *SrcPtrSCEV = dyn_cast<SCEVUnknown>(SE.getPointerBase(SrcAccFunc));
  assert(SrcPtrSCEV);
  SrcAccFunc = SE.getMinusSCEV(SrcAccFunc, SrcPtrSCEV);
  addArrayAccess(Stmt, Inst, MemoryAccess::READ, SrcPtrSCEV->getValue(),
                 IntegerType::getInt8Ty(SrcPtrVal->getContext()),
                 LengthIsAffine, {SrcAccFunc, LengthVal}, {nullptr},
                 Inst.getValueOperand());

  return true;
}

bool ScopBuilder::buildAccessCallInst(MemAccInst Inst, ScopStmt *Stmt) {
  auto *CI = dyn_cast_or_null<CallInst>(Inst);

  if (CI == nullptr)
    return false;

  if (CI->doesNotAccessMemory() || isIgnoredIntrinsic(CI) || isDebugCall(CI))
    return true;

  bool ReadOnly = false;
  auto *AF = SE.getConstant(IntegerType::getInt64Ty(CI->getContext()), 0);
  auto *CalledFunction = CI->getCalledFunction();
  switch (AA.getModRefBehavior(CalledFunction)) {
  case FMRB_UnknownModRefBehavior:
    llvm_unreachable("Unknown mod ref behaviour cannot be represented.");
  case FMRB_DoesNotAccessMemory:
    return true;
  case FMRB_OnlyWritesMemory:
  case FMRB_OnlyWritesInaccessibleMem:
  case FMRB_OnlyWritesInaccessibleOrArgMem:
  case FMRB_OnlyAccessesInaccessibleMem:
  case FMRB_OnlyAccessesInaccessibleOrArgMem:
    return false;
  case FMRB_OnlyReadsMemory:
  case FMRB_OnlyReadsInaccessibleMem:
  case FMRB_OnlyReadsInaccessibleOrArgMem:
    GlobalReads.emplace_back(Stmt, CI);
    return true;
  case FMRB_OnlyReadsArgumentPointees:
    ReadOnly = true;
    LLVM_FALLTHROUGH;
  case FMRB_OnlyWritesArgumentPointees:
  case FMRB_OnlyAccessesArgumentPointees: {
    auto AccType = ReadOnly ? MemoryAccess::READ : MemoryAccess::MAY_WRITE;
    Loop *L = LI.getLoopFor(Inst->getParent());
    for (const auto &Arg : CI->arg_operands()) {
      if (!Arg->getType()->isPointerTy())
        continue;

      auto *ArgSCEV = SE.getSCEVAtScope(Arg, L);
      if (ArgSCEV->isZero())
        continue;

      auto *ArgBasePtr = cast<SCEVUnknown>(SE.getPointerBase(ArgSCEV));
      addArrayAccess(Stmt, Inst, AccType, ArgBasePtr->getValue(),
                     ArgBasePtr->getType(), false, {AF}, {nullptr}, CI);
    }
    return true;
  }
  }

  return true;
}

void ScopBuilder::buildAccessSingleDim(MemAccInst Inst, ScopStmt *Stmt) {
  Value *Address = Inst.getPointerOperand();
  Value *Val = Inst.getValueOperand();
  Type *ElementType = Val->getType();
  enum MemoryAccess::AccessType AccType =
      isa<LoadInst>(Inst) ? MemoryAccess::READ : MemoryAccess::MUST_WRITE;

  const SCEV *AccessFunction =
      SE.getSCEVAtScope(Address, LI.getLoopFor(Inst->getParent()));
  const SCEVUnknown *BasePointer =
      dyn_cast<SCEVUnknown>(SE.getPointerBase(AccessFunction));

  assert(BasePointer && "Could not find base pointer");
  AccessFunction = SE.getMinusSCEV(AccessFunction, BasePointer);

  // Check if the access depends on a loop contained in a non-affine subregion.
  bool isVariantInNonAffineLoop = false;
  SetVector<const Loop *> Loops;
  findLoops(AccessFunction, Loops);
  for (const Loop *L : Loops)
    if (Stmt->contains(L)) {
      isVariantInNonAffineLoop = true;
      break;
    }

  InvariantLoadsSetTy AccessILS;

  Loop *SurroundingLoop = Stmt->getSurroundingLoop();
  bool IsAffine = !isVariantInNonAffineLoop &&
                  isAffineExpr(&scop->getRegion(), SurroundingLoop,
                               AccessFunction, SE, &AccessILS);

  const InvariantLoadsSetTy &ScopRIL = scop->getRequiredInvariantLoads();
  for (LoadInst *LInst : AccessILS)
    if (!ScopRIL.count(LInst))
      IsAffine = false;

  if (!IsAffine && AccType == MemoryAccess::MUST_WRITE)
    AccType = MemoryAccess::MAY_WRITE;

  addArrayAccess(Stmt, Inst, AccType, BasePointer->getValue(), ElementType,
                 IsAffine, {AccessFunction}, {nullptr}, Val);
}

void ScopBuilder::buildMemoryAccess(MemAccInst Inst, ScopStmt *Stmt) {
  if (buildAccessMemIntrinsic(Inst, Stmt))
    return;

  if (buildAccessCallInst(Inst, Stmt))
    return;

  if (buildAccessMultiDimFixed(Inst, Stmt))
    return;

  if (buildAccessMultiDimParam(Inst, Stmt))
    return;

  buildAccessSingleDim(Inst, Stmt);
}

void ScopBuilder::buildAccessFunctions() {
  for (auto &Stmt : *scop) {
    if (Stmt.isBlockStmt()) {
      buildAccessFunctions(&Stmt, *Stmt.getBasicBlock());
      continue;
    }

    Region *R = Stmt.getRegion();
    for (BasicBlock *BB : R->blocks())
      buildAccessFunctions(&Stmt, *BB, R);
  }

  // Build write accesses for values that are used after the SCoP.
  // The instructions defining them might be synthesizable and therefore not
  // contained in any statement, hence we iterate over the original instructions
  // to identify all escaping values.
  for (BasicBlock *BB : scop->getRegion().blocks()) {
    for (Instruction &Inst : *BB)
      buildEscapingDependences(&Inst);
  }
}

bool ScopBuilder::shouldModelInst(Instruction *Inst, Loop *L) {
  return !Inst->isTerminator() && !isIgnoredIntrinsic(Inst) &&
         !canSynthesize(Inst, *scop, &SE, L);
}

/// Generate a name for a statement.
///
/// @param BB     The basic block the statement will represent.
/// @param BBIdx  The index of the @p BB relative to other BBs/regions.
/// @param Count  The index of the created statement in @p BB.
/// @param IsMain Whether this is the main of all statement for @p BB. If true,
///               no suffix will be added.
/// @param IsLast Uses a special indicator for the last statement of a BB.
static std::string makeStmtName(BasicBlock *BB, long BBIdx, int Count,
                                bool IsMain, bool IsLast = false) {
  std::string Suffix;
  if (!IsMain) {
    if (UseInstructionNames)
      Suffix = '_';
    if (IsLast)
      Suffix += "last";
    else if (Count < 26)
      Suffix += 'a' + Count;
    else
      Suffix += std::to_string(Count);
  }
  return getIslCompatibleName("Stmt", BB, BBIdx, Suffix, UseInstructionNames);
}

/// Generate a name for a statement that represents a non-affine subregion.
///
/// @param R    The region the statement will represent.
/// @param RIdx The index of the @p R relative to other BBs/regions.
static std::string makeStmtName(Region *R, long RIdx) {
  return getIslCompatibleName("Stmt", R->getNameStr(), RIdx, "",
                              UseInstructionNames);
}

void ScopBuilder::buildSequentialBlockStmts(BasicBlock *BB, bool SplitOnStore) {
  Loop *SurroundingLoop = LI.getLoopFor(BB);

  int Count = 0;
  long BBIdx = scop->getNextStmtIdx();
  std::vector<Instruction *> Instructions;
  for (Instruction &Inst : *BB) {
    if (shouldModelInst(&Inst, SurroundingLoop))
      Instructions.push_back(&Inst);
    if (Inst.getMetadata("polly_split_after") ||
        (SplitOnStore && isa<StoreInst>(Inst))) {
      std::string Name = makeStmtName(BB, BBIdx, Count, Count == 0);
      scop->addScopStmt(BB, Name, SurroundingLoop, Instructions);
      Count++;
      Instructions.clear();
    }
  }

  std::string Name = makeStmtName(BB, BBIdx, Count, Count == 0);
  scop->addScopStmt(BB, Name, SurroundingLoop, Instructions);
}

/// Is @p Inst an ordered instruction?
///
/// An unordered instruction is an instruction, such that a sequence of
/// unordered instructions can be permuted without changing semantics. Any
/// instruction for which this is not always the case is ordered.
static bool isOrderedInstruction(Instruction *Inst) {
  return Inst->mayHaveSideEffects() || Inst->mayReadOrWriteMemory();
}

/// Join instructions to the same statement if one uses the scalar result of the
/// other.
static void joinOperandTree(EquivalenceClasses<Instruction *> &UnionFind,
                            ArrayRef<Instruction *> ModeledInsts) {
  for (Instruction *Inst : ModeledInsts) {
    if (isa<PHINode>(Inst))
      continue;

    for (Use &Op : Inst->operands()) {
      Instruction *OpInst = dyn_cast<Instruction>(Op.get());
      if (!OpInst)
        continue;

      // Check if OpInst is in the BB and is a modeled instruction.
      auto OpVal = UnionFind.findValue(OpInst);
      if (OpVal == UnionFind.end())
        continue;

      UnionFind.unionSets(Inst, OpInst);
    }
  }
}

/// Ensure that the order of ordered instructions does not change.
///
/// If we encounter an ordered instruction enclosed in instructions belonging to
/// a different statement (which might as well contain ordered instructions, but
/// this is not tested here), join them.
static void
joinOrderedInstructions(EquivalenceClasses<Instruction *> &UnionFind,
                        ArrayRef<Instruction *> ModeledInsts) {
  SetVector<Instruction *> SeenLeaders;
  for (Instruction *Inst : ModeledInsts) {
    if (!isOrderedInstruction(Inst))
      continue;

    Instruction *Leader = UnionFind.getLeaderValue(Inst);
    // Since previous iterations might have merged sets, some items in
    // SeenLeaders are not leaders anymore. However, The new leader of
    // previously merged instructions must be one of the former leaders of
    // these merged instructions.
    bool Inserted = SeenLeaders.insert(Leader);
    if (Inserted)
      continue;

    // Merge statements to close holes. Say, we have already seen statements A
    // and B, in this order. Then we see an instruction of A again and we would
    // see the pattern "A B A". This function joins all statements until the
    // only seen occurrence of A.
    for (Instruction *Prev : reverse(SeenLeaders)) {
      // We are backtracking from the last element until we see Inst's leader
      // in SeenLeaders and merge all into one set. Although leaders of
      // instructions change during the execution of this loop, it's irrelevant
      // as we are just searching for the element that we already confirmed is
      // in the list.
      if (Prev == Leader)
        break;
      UnionFind.unionSets(Prev, Leader);
    }
  }
}

/// If the BasicBlock has an edge from itself, ensure that the PHI WRITEs for
/// the incoming values from this block are executed after the PHI READ.
///
/// Otherwise it could overwrite the incoming value from before the BB with the
/// value for the next execution. This can happen if the PHI WRITE is added to
/// the statement with the instruction that defines the incoming value (instead
/// of the last statement of the same BB). To ensure that the PHI READ and WRITE
/// are in order, we put both into the statement. PHI WRITEs are always executed
/// after PHI READs when they are in the same statement.
///
/// TODO: This is an overpessimization. We only have to ensure that the PHI
/// WRITE is not put into a statement containing the PHI itself. That could also
/// be done by
/// - having all (strongly connected) PHIs in a single statement,
/// - unite only the PHIs in the operand tree of the PHI WRITE (because it only
///   has a chance of being lifted before a PHI by being in a statement with a
///   PHI that comes before in the basic block), or
/// - when uniting statements, ensure that no (relevant) PHIs are overtaken.
static void joinOrderedPHIs(EquivalenceClasses<Instruction *> &UnionFind,
                            ArrayRef<Instruction *> ModeledInsts) {
  for (Instruction *Inst : ModeledInsts) {
    PHINode *PHI = dyn_cast<PHINode>(Inst);
    if (!PHI)
      continue;

    int Idx = PHI->getBasicBlockIndex(PHI->getParent());
    if (Idx < 0)
      continue;

    Instruction *IncomingVal =
        dyn_cast<Instruction>(PHI->getIncomingValue(Idx));
    if (!IncomingVal)
      continue;

    UnionFind.unionSets(PHI, IncomingVal);
  }
}

void ScopBuilder::buildEqivClassBlockStmts(BasicBlock *BB) {
  Loop *L = LI.getLoopFor(BB);

  // Extracting out modeled instructions saves us from checking
  // shouldModelInst() repeatedly.
  SmallVector<Instruction *, 32> ModeledInsts;
  EquivalenceClasses<Instruction *> UnionFind;
  Instruction *MainInst = nullptr, *MainLeader = nullptr;
  for (Instruction &Inst : *BB) {
    if (!shouldModelInst(&Inst, L))
      continue;
    ModeledInsts.push_back(&Inst);
    UnionFind.insert(&Inst);

    // When a BB is split into multiple statements, the main statement is the
    // one containing the 'main' instruction. We select the first instruction
    // that is unlikely to be removed (because it has side-effects) as the main
    // one. It is used to ensure that at least one statement from the bb has the
    // same name as with -polly-stmt-granularity=bb.
    if (!MainInst && (isa<StoreInst>(Inst) ||
                      (isa<CallInst>(Inst) && !isa<IntrinsicInst>(Inst))))
      MainInst = &Inst;
  }

  joinOperandTree(UnionFind, ModeledInsts);
  joinOrderedInstructions(UnionFind, ModeledInsts);
  joinOrderedPHIs(UnionFind, ModeledInsts);

  // The list of instructions for statement (statement represented by the leader
  // instruction).
  MapVector<Instruction *, std::vector<Instruction *>> LeaderToInstList;

  // The order of statements must be preserved w.r.t. their ordered
  // instructions. Without this explicit scan, we would also use non-ordered
  // instructions (whose order is arbitrary) to determine statement order.
  for (Instruction *Inst : ModeledInsts) {
    if (!isOrderedInstruction(Inst))
      continue;

    auto LeaderIt = UnionFind.findLeader(Inst);
    if (LeaderIt == UnionFind.member_end())
      continue;

    // Insert element for the leader instruction.
    (void)LeaderToInstList[*LeaderIt];
  }

  // Collect the instructions of all leaders. UnionFind's member iterator
  // unfortunately are not in any specific order.
  for (Instruction *Inst : ModeledInsts) {
    auto LeaderIt = UnionFind.findLeader(Inst);
    if (LeaderIt == UnionFind.member_end())
      continue;

    if (Inst == MainInst)
      MainLeader = *LeaderIt;
    std::vector<Instruction *> &InstList = LeaderToInstList[*LeaderIt];
    InstList.push_back(Inst);
  }

  // Finally build the statements.
  int Count = 0;
  long BBIdx = scop->getNextStmtIdx();
  for (auto &Instructions : LeaderToInstList) {
    std::vector<Instruction *> &InstList = Instructions.second;

    // If there is no main instruction, make the first statement the main.
    bool IsMain = (MainInst ? MainLeader == Instructions.first : Count == 0);

    std::string Name = makeStmtName(BB, BBIdx, Count, IsMain);
    scop->addScopStmt(BB, Name, L, std::move(InstList));
    Count += 1;
  }

  // Unconditionally add an epilogue (last statement). It contains no
  // instructions, but holds the PHI write accesses for successor basic blocks,
  // if the incoming value is not defined in another statement if the same BB.
  // The epilogue becomes the main statement only if there is no other
  // statement that could become main.
  // The epilogue will be removed if no PHIWrite is added to it.
  std::string EpilogueName = makeStmtName(BB, BBIdx, Count, Count == 0, true);
  scop->addScopStmt(BB, EpilogueName, L, {});
}

void ScopBuilder::buildStmts(Region &SR) {
  if (scop->isNonAffineSubRegion(&SR)) {
    std::vector<Instruction *> Instructions;
    Loop *SurroundingLoop =
        getFirstNonBoxedLoopFor(SR.getEntry(), LI, scop->getBoxedLoops());
    for (Instruction &Inst : *SR.getEntry())
      if (shouldModelInst(&Inst, SurroundingLoop))
        Instructions.push_back(&Inst);
    long RIdx = scop->getNextStmtIdx();
    std::string Name = makeStmtName(&SR, RIdx);
    scop->addScopStmt(&SR, Name, SurroundingLoop, Instructions);
    return;
  }

  for (auto I = SR.element_begin(), E = SR.element_end(); I != E; ++I)
    if (I->isSubRegion())
      buildStmts(*I->getNodeAs<Region>());
    else {
      BasicBlock *BB = I->getNodeAs<BasicBlock>();
      switch (StmtGranularity) {
      case GranularityChoice::BasicBlocks:
        buildSequentialBlockStmts(BB);
        break;
      case GranularityChoice::ScalarIndependence:
        buildEqivClassBlockStmts(BB);
        break;
      case GranularityChoice::Stores:
        buildSequentialBlockStmts(BB, true);
        break;
      }
    }
}

void ScopBuilder::buildAccessFunctions(ScopStmt *Stmt, BasicBlock &BB,
                                       Region *NonAffineSubRegion) {
  assert(
      Stmt &&
      "The exit BB is the only one that cannot be represented by a statement");
  assert(Stmt->represents(&BB));

  // We do not build access functions for error blocks, as they may contain
  // instructions we can not model.
  if (isErrorBlock(BB, scop->getRegion(), LI, DT))
    return;

  auto BuildAccessesForInst = [this, Stmt,
                               NonAffineSubRegion](Instruction *Inst) {
    PHINode *PHI = dyn_cast<PHINode>(Inst);
    if (PHI)
      buildPHIAccesses(Stmt, PHI, NonAffineSubRegion, false);

    if (auto MemInst = MemAccInst::dyn_cast(*Inst)) {
      assert(Stmt && "Cannot build access function in non-existing statement");
      buildMemoryAccess(MemInst, Stmt);
    }

    // PHI nodes have already been modeled above and terminators that are
    // not part of a non-affine subregion are fully modeled and regenerated
    // from the polyhedral domains. Hence, they do not need to be modeled as
    // explicit data dependences.
    if (!PHI)
      buildScalarDependences(Stmt, Inst);
  };

  const InvariantLoadsSetTy &RIL = scop->getRequiredInvariantLoads();
  bool IsEntryBlock = (Stmt->getEntryBlock() == &BB);
  if (IsEntryBlock) {
    for (Instruction *Inst : Stmt->getInstructions())
      BuildAccessesForInst(Inst);
    if (Stmt->isRegionStmt())
      BuildAccessesForInst(BB.getTerminator());
  } else {
    for (Instruction &Inst : BB) {
      if (isIgnoredIntrinsic(&Inst))
        continue;

      // Invariant loads already have been processed.
      if (isa<LoadInst>(Inst) && RIL.count(cast<LoadInst>(&Inst)))
        continue;

      BuildAccessesForInst(&Inst);
    }
  }
}

MemoryAccess *ScopBuilder::addMemoryAccess(
    ScopStmt *Stmt, Instruction *Inst, MemoryAccess::AccessType AccType,
    Value *BaseAddress, Type *ElementType, bool Affine, Value *AccessValue,
    ArrayRef<const SCEV *> Subscripts, ArrayRef<const SCEV *> Sizes,
    MemoryKind Kind) {
  bool isKnownMustAccess = false;

  // Accesses in single-basic block statements are always executed.
  if (Stmt->isBlockStmt())
    isKnownMustAccess = true;

  if (Stmt->isRegionStmt()) {
    // Accesses that dominate the exit block of a non-affine region are always
    // executed. In non-affine regions there may exist MemoryKind::Values that
    // do not dominate the exit. MemoryKind::Values will always dominate the
    // exit and MemoryKind::PHIs only if there is at most one PHI_WRITE in the
    // non-affine region.
    if (Inst && DT.dominates(Inst->getParent(), Stmt->getRegion()->getExit()))
      isKnownMustAccess = true;
  }

  // Non-affine PHI writes do not "happen" at a particular instruction, but
  // after exiting the statement. Therefore they are guaranteed to execute and
  // overwrite the old value.
  if (Kind == MemoryKind::PHI || Kind == MemoryKind::ExitPHI)
    isKnownMustAccess = true;

  if (!isKnownMustAccess && AccType == MemoryAccess::MUST_WRITE)
    AccType = MemoryAccess::MAY_WRITE;

  auto *Access = new MemoryAccess(Stmt, Inst, AccType, BaseAddress, ElementType,
                                  Affine, Subscripts, Sizes, AccessValue, Kind);

  scop->addAccessFunction(Access);
  Stmt->addAccess(Access);
  return Access;
}

void ScopBuilder::addArrayAccess(ScopStmt *Stmt, MemAccInst MemAccInst,
                                 MemoryAccess::AccessType AccType,
                                 Value *BaseAddress, Type *ElementType,
                                 bool IsAffine,
                                 ArrayRef<const SCEV *> Subscripts,
                                 ArrayRef<const SCEV *> Sizes,
                                 Value *AccessValue) {
  ArrayBasePointers.insert(BaseAddress);
  auto *MemAccess = addMemoryAccess(Stmt, MemAccInst, AccType, BaseAddress,
                                    ElementType, IsAffine, AccessValue,
                                    Subscripts, Sizes, MemoryKind::Array);

  if (!DetectFortranArrays)
    return;

  if (Value *FAD = findFADAllocationInvisible(MemAccInst))
    MemAccess->setFortranArrayDescriptor(FAD);
  else if (Value *FAD = findFADAllocationVisible(MemAccInst))
    MemAccess->setFortranArrayDescriptor(FAD);
}

/// Check if @p Expr is divisible by @p Size.
static bool isDivisible(const SCEV *Expr, unsigned Size, ScalarEvolution &SE) {
  assert(Size != 0);
  if (Size == 1)
    return true;

  // Only one factor needs to be divisible.
  if (auto *MulExpr = dyn_cast<SCEVMulExpr>(Expr)) {
    for (auto *FactorExpr : MulExpr->operands())
      if (isDivisible(FactorExpr, Size, SE))
        return true;
    return false;
  }

  // For other n-ary expressions (Add, AddRec, Max,...) all operands need
  // to be divisible.
  if (auto *NAryExpr = dyn_cast<SCEVNAryExpr>(Expr)) {
    for (auto *OpExpr : NAryExpr->operands())
      if (!isDivisible(OpExpr, Size, SE))
        return false;
    return true;
  }

  auto *SizeSCEV = SE.getConstant(Expr->getType(), Size);
  auto *UDivSCEV = SE.getUDivExpr(Expr, SizeSCEV);
  auto *MulSCEV = SE.getMulExpr(UDivSCEV, SizeSCEV);
  return MulSCEV == Expr;
}

void ScopBuilder::foldSizeConstantsToRight() {
  isl::union_set Accessed = scop->getAccesses().range();

  for (auto Array : scop->arrays()) {
    if (Array->getNumberOfDimensions() <= 1)
      continue;

    isl::space Space = Array->getSpace();
    Space = Space.align_params(Accessed.get_space());

    if (!Accessed.contains(Space))
      continue;

    isl::set Elements = Accessed.extract_set(Space);
    isl::map Transform = isl::map::universe(Array->getSpace().map_from_set());

    std::vector<int> Int;
    int Dims = Elements.dim(isl::dim::set);
    for (int i = 0; i < Dims; i++) {
      isl::set DimOnly = isl::set(Elements).project_out(isl::dim::set, 0, i);
      DimOnly = DimOnly.project_out(isl::dim::set, 1, Dims - i - 1);
      DimOnly = DimOnly.lower_bound_si(isl::dim::set, 0, 0);

      isl::basic_set DimHull = DimOnly.affine_hull();

      if (i == Dims - 1) {
        Int.push_back(1);
        Transform = Transform.equate(isl::dim::in, i, isl::dim::out, i);
        continue;
      }

      if (DimHull.dim(isl::dim::div) == 1) {
        isl::aff Diff = DimHull.get_div(0);
        isl::val Val = Diff.get_denominator_val();

        int ValInt = 1;
        if (Val.is_int()) {
          auto ValAPInt = APIntFromVal(Val);
          if (ValAPInt.isSignedIntN(32))
            ValInt = ValAPInt.getSExtValue();
        } else {
        }

        Int.push_back(ValInt);
        isl::constraint C = isl::constraint::alloc_equality(
            isl::local_space(Transform.get_space()));
        C = C.set_coefficient_si(isl::dim::out, i, ValInt);
        C = C.set_coefficient_si(isl::dim::in, i, -1);
        Transform = Transform.add_constraint(C);
        continue;
      }

      isl::basic_set ZeroSet = isl::basic_set(DimHull);
      ZeroSet = ZeroSet.fix_si(isl::dim::set, 0, 0);

      int ValInt = 1;
      if (ZeroSet.is_equal(DimHull)) {
        ValInt = 0;
      }

      Int.push_back(ValInt);
      Transform = Transform.equate(isl::dim::in, i, isl::dim::out, i);
    }

    isl::set MappedElements = isl::map(Transform).domain();
    if (!Elements.is_subset(MappedElements))
      continue;

    bool CanFold = true;
    if (Int[0] <= 1)
      CanFold = false;

    unsigned NumDims = Array->getNumberOfDimensions();
    for (unsigned i = 1; i < NumDims - 1; i++)
      if (Int[0] != Int[i] && Int[i])
        CanFold = false;

    if (!CanFold)
      continue;

    for (auto &Access : scop->access_functions())
      if (Access->getScopArrayInfo() == Array)
        Access->setAccessRelation(
            Access->getAccessRelation().apply_range(Transform));

    std::vector<const SCEV *> Sizes;
    for (unsigned i = 0; i < NumDims; i++) {
      auto Size = Array->getDimensionSize(i);

      if (i == NumDims - 1)
        Size = SE.getMulExpr(Size, SE.getConstant(Size->getType(), Int[0]));
      Sizes.push_back(Size);
    }

    Array->updateSizes(Sizes, false /* CheckConsistency */);
  }
}

void ScopBuilder::markFortranArrays() {
  for (ScopStmt &Stmt : *scop) {
    for (MemoryAccess *MemAcc : Stmt) {
      Value *FAD = MemAcc->getFortranArrayDescriptor();
      if (!FAD)
        continue;

      // TODO: const_cast-ing to edit
      ScopArrayInfo *SAI =
          const_cast<ScopArrayInfo *>(MemAcc->getLatestScopArrayInfo());
      assert(SAI && "memory access into a Fortran array does not "
                    "have an associated ScopArrayInfo");
      SAI->applyAndSetFAD(FAD);
    }
  }
}

void ScopBuilder::finalizeAccesses() {
  updateAccessDimensionality();
  foldSizeConstantsToRight();
  foldAccessRelations();
  assumeNoOutOfBounds();
  markFortranArrays();
}

void ScopBuilder::updateAccessDimensionality() {
  // Check all array accesses for each base pointer and find a (virtual) element
  // size for the base pointer that divides all access functions.
  for (ScopStmt &Stmt : *scop)
    for (MemoryAccess *Access : Stmt) {
      if (!Access->isArrayKind())
        continue;
      ScopArrayInfo *Array =
          const_cast<ScopArrayInfo *>(Access->getScopArrayInfo());

      if (Array->getNumberOfDimensions() != 1)
        continue;
      unsigned DivisibleSize = Array->getElemSizeInBytes();
      const SCEV *Subscript = Access->getSubscript(0);
      while (!isDivisible(Subscript, DivisibleSize, SE))
        DivisibleSize /= 2;
      auto *Ty = IntegerType::get(SE.getContext(), DivisibleSize * 8);
      Array->updateElementType(Ty);
    }

  for (auto &Stmt : *scop)
    for (auto &Access : Stmt)
      Access->updateDimensionality();
}

void ScopBuilder::foldAccessRelations() {
  for (auto &Stmt : *scop)
    for (auto &Access : Stmt)
      Access->foldAccessRelation();
}

void ScopBuilder::assumeNoOutOfBounds() {
  if (PollyIgnoreInbounds)
    return;
  for (auto &Stmt : *scop)
    for (auto &Access : Stmt) {
      isl::set Outside = Access->assumeNoOutOfBound();
      const auto &Loc = Access->getAccessInstruction()
                            ? Access->getAccessInstruction()->getDebugLoc()
                            : DebugLoc();
      recordAssumption(&RecordedAssumptions, INBOUNDS, Outside, Loc,
                       AS_ASSUMPTION);
    }
}

void ScopBuilder::ensureValueWrite(Instruction *Inst) {
  // Find the statement that defines the value of Inst. That statement has to
  // write the value to make it available to those statements that read it.
  ScopStmt *Stmt = scop->getStmtFor(Inst);

  // It is possible that the value is synthesizable within a loop (such that it
  // is not part of any statement), but not after the loop (where you need the
  // number of loop round-trips to synthesize it). In LCSSA-form a PHI node will
  // avoid this. In case the IR has no such PHI, use the last statement (where
  // the value is synthesizable) to write the value.
  if (!Stmt)
    Stmt = scop->getLastStmtFor(Inst->getParent());

  // Inst not defined within this SCoP.
  if (!Stmt)
    return;

  // Do not process further if the instruction is already written.
  if (Stmt->lookupValueWriteOf(Inst))
    return;

  addMemoryAccess(Stmt, Inst, MemoryAccess::MUST_WRITE, Inst, Inst->getType(),
                  true, Inst, ArrayRef<const SCEV *>(),
                  ArrayRef<const SCEV *>(), MemoryKind::Value);
}

void ScopBuilder::ensureValueRead(Value *V, ScopStmt *UserStmt) {
  // TODO: Make ScopStmt::ensureValueRead(Value*) offer the same functionality
  // to be able to replace this one. Currently, there is a split responsibility.
  // In a first step, the MemoryAccess is created, but without the
  // AccessRelation. In the second step by ScopStmt::buildAccessRelations(), the
  // AccessRelation is created. At least for scalar accesses, there is no new
  // information available at ScopStmt::buildAccessRelations(), so we could
  // create the AccessRelation right away. This is what
  // ScopStmt::ensureValueRead(Value*) does.

  auto *Scope = UserStmt->getSurroundingLoop();
  auto VUse = VirtualUse::create(scop.get(), UserStmt, Scope, V, false);
  switch (VUse.getKind()) {
  case VirtualUse::Constant:
  case VirtualUse::Block:
  case VirtualUse::Synthesizable:
  case VirtualUse::Hoisted:
  case VirtualUse::Intra:
    // Uses of these kinds do not need a MemoryAccess.
    break;

  case VirtualUse::ReadOnly:
    // Add MemoryAccess for invariant values only if requested.
    if (!ModelReadOnlyScalars)
      break;

    LLVM_FALLTHROUGH;
  case VirtualUse::Inter:

    // Do not create another MemoryAccess for reloading the value if one already
    // exists.
    if (UserStmt->lookupValueReadOf(V))
      break;

    addMemoryAccess(UserStmt, nullptr, MemoryAccess::READ, V, V->getType(),
                    true, V, ArrayRef<const SCEV *>(), ArrayRef<const SCEV *>(),
                    MemoryKind::Value);

    // Inter-statement uses need to write the value in their defining statement.
    if (VUse.isInter())
      ensureValueWrite(cast<Instruction>(V));
    break;
  }
}

void ScopBuilder::ensurePHIWrite(PHINode *PHI, ScopStmt *IncomingStmt,
                                 BasicBlock *IncomingBlock,
                                 Value *IncomingValue, bool IsExitBlock) {
  // As the incoming block might turn out to be an error statement ensure we
  // will create an exit PHI SAI object. It is needed during code generation
  // and would be created later anyway.
  if (IsExitBlock)
    scop->getOrCreateScopArrayInfo(PHI, PHI->getType(), {},
                                   MemoryKind::ExitPHI);

  // This is possible if PHI is in the SCoP's entry block. The incoming blocks
  // from outside the SCoP's region have no statement representation.
  if (!IncomingStmt)
    return;

  // Take care for the incoming value being available in the incoming block.
  // This must be done before the check for multiple PHI writes because multiple
  // exiting edges from subregion each can be the effective written value of the
  // subregion. As such, all of them must be made available in the subregion
  // statement.
  ensureValueRead(IncomingValue, IncomingStmt);

  // Do not add more than one MemoryAccess per PHINode and ScopStmt.
  if (MemoryAccess *Acc = IncomingStmt->lookupPHIWriteOf(PHI)) {
    assert(Acc->getAccessInstruction() == PHI);
    Acc->addIncoming(IncomingBlock, IncomingValue);
    return;
  }

  MemoryAccess *Acc = addMemoryAccess(
      IncomingStmt, PHI, MemoryAccess::MUST_WRITE, PHI, PHI->getType(), true,
      PHI, ArrayRef<const SCEV *>(), ArrayRef<const SCEV *>(),
      IsExitBlock ? MemoryKind::ExitPHI : MemoryKind::PHI);
  assert(Acc);
  Acc->addIncoming(IncomingBlock, IncomingValue);
}

void ScopBuilder::addPHIReadAccess(ScopStmt *PHIStmt, PHINode *PHI) {
  addMemoryAccess(PHIStmt, PHI, MemoryAccess::READ, PHI, PHI->getType(), true,
                  PHI, ArrayRef<const SCEV *>(), ArrayRef<const SCEV *>(),
                  MemoryKind::PHI);
}

void ScopBuilder::buildDomain(ScopStmt &Stmt) {
  isl::id Id = isl::id::alloc(scop->getIslCtx(), Stmt.getBaseName(), &Stmt);

  Stmt.Domain = scop->getDomainConditions(&Stmt);
  Stmt.Domain = Stmt.Domain.set_tuple_id(Id);
}

void ScopBuilder::collectSurroundingLoops(ScopStmt &Stmt) {
  isl::set Domain = Stmt.getDomain();
  BasicBlock *BB = Stmt.getEntryBlock();

  Loop *L = LI.getLoopFor(BB);

  while (L && Stmt.isRegionStmt() && Stmt.getRegion()->contains(L))
    L = L->getParentLoop();

  SmallVector<llvm::Loop *, 8> Loops;

  while (L && Stmt.getParent()->getRegion().contains(L)) {
    Loops.push_back(L);
    L = L->getParentLoop();
  }

  Stmt.NestLoops.insert(Stmt.NestLoops.begin(), Loops.rbegin(), Loops.rend());
}

/// Return the reduction type for a given binary operator.
static MemoryAccess::ReductionType getReductionType(const BinaryOperator *BinOp,
                                                    const Instruction *Load) {
  if (!BinOp)
    return MemoryAccess::RT_NONE;
  switch (BinOp->getOpcode()) {
  case Instruction::FAdd:
    if (!BinOp->isFast())
      return MemoryAccess::RT_NONE;
    LLVM_FALLTHROUGH;
  case Instruction::Add:
    return MemoryAccess::RT_ADD;
  case Instruction::Or:
    return MemoryAccess::RT_BOR;
  case Instruction::Xor:
    return MemoryAccess::RT_BXOR;
  case Instruction::And:
    return MemoryAccess::RT_BAND;
  case Instruction::FMul:
    if (!BinOp->isFast())
      return MemoryAccess::RT_NONE;
    LLVM_FALLTHROUGH;
  case Instruction::Mul:
    if (DisableMultiplicativeReductions)
      return MemoryAccess::RT_NONE;
    return MemoryAccess::RT_MUL;
  default:
    return MemoryAccess::RT_NONE;
  }
}

void ScopBuilder::checkForReductions(ScopStmt &Stmt) {
  SmallVector<MemoryAccess *, 2> Loads;
  SmallVector<std::pair<MemoryAccess *, MemoryAccess *>, 4> Candidates;

  // First collect candidate load-store reduction chains by iterating over all
  // stores and collecting possible reduction loads.
  for (MemoryAccess *StoreMA : Stmt) {
    if (StoreMA->isRead())
      continue;

    Loads.clear();
    collectCandidateReductionLoads(StoreMA, Loads);
    for (MemoryAccess *LoadMA : Loads)
      Candidates.push_back(std::make_pair(LoadMA, StoreMA));
  }

  // Then check each possible candidate pair.
  for (const auto &CandidatePair : Candidates) {
    bool Valid = true;
    isl::map LoadAccs = CandidatePair.first->getAccessRelation();
    isl::map StoreAccs = CandidatePair.second->getAccessRelation();

    // Skip those with obviously unequal base addresses.
    if (!LoadAccs.has_equal_space(StoreAccs)) {
      continue;
    }

    // And check if the remaining for overlap with other memory accesses.
    isl::map AllAccsRel = LoadAccs.unite(StoreAccs);
    AllAccsRel = AllAccsRel.intersect_domain(Stmt.getDomain());
    isl::set AllAccs = AllAccsRel.range();

    for (MemoryAccess *MA : Stmt) {
      if (MA == CandidatePair.first || MA == CandidatePair.second)
        continue;

      isl::map AccRel =
          MA->getAccessRelation().intersect_domain(Stmt.getDomain());
      isl::set Accs = AccRel.range();

      if (AllAccs.has_equal_space(Accs)) {
        isl::set OverlapAccs = Accs.intersect(AllAccs);
        Valid = Valid && OverlapAccs.is_empty();
      }
    }

    if (!Valid)
      continue;

    const LoadInst *Load =
        dyn_cast<const LoadInst>(CandidatePair.first->getAccessInstruction());
    MemoryAccess::ReductionType RT =
        getReductionType(dyn_cast<BinaryOperator>(Load->user_back()), Load);

    // If no overlapping access was found we mark the load and store as
    // reduction like.
    CandidatePair.first->markAsReductionLike(RT);
    CandidatePair.second->markAsReductionLike(RT);
  }
}

void ScopBuilder::verifyInvariantLoads() {
  auto &RIL = scop->getRequiredInvariantLoads();
  for (LoadInst *LI : RIL) {
    assert(LI && scop->contains(LI));
    // If there exists a statement in the scop which has a memory access for
    // @p LI, then mark this scop as infeasible for optimization.
    for (ScopStmt &Stmt : *scop)
      if (Stmt.getArrayAccessOrNULLFor(LI)) {
        scop->invalidate(INVARIANTLOAD, LI->getDebugLoc(), LI->getParent());
        return;
      }
  }
}

void ScopBuilder::hoistInvariantLoads() {
  if (!PollyInvariantLoadHoisting)
    return;

  isl::union_map Writes = scop->getWrites();
  for (ScopStmt &Stmt : *scop) {
    InvariantAccessesTy InvariantAccesses;

    for (MemoryAccess *Access : Stmt)
      if (isl::set NHCtx = getNonHoistableCtx(Access, Writes))
        InvariantAccesses.push_back({Access, NHCtx});

    // Transfer the memory access from the statement to the SCoP.
    for (auto InvMA : InvariantAccesses)
      Stmt.removeMemoryAccess(InvMA.MA);
    addInvariantLoads(Stmt, InvariantAccesses);
  }
}

/// Check if an access range is too complex.
///
/// An access range is too complex, if it contains either many disjuncts or
/// very complex expressions. As a simple heuristic, we assume if a set to
/// be too complex if the sum of existentially quantified dimensions and
/// set dimensions is larger than a threshold. This reliably detects both
/// sets with many disjuncts as well as sets with many divisions as they
/// arise in h264.
///
/// @param AccessRange The range to check for complexity.
///
/// @returns True if the access range is too complex.
static bool isAccessRangeTooComplex(isl::set AccessRange) {
  int NumTotalDims = 0;

  for (isl::basic_set BSet : AccessRange.get_basic_set_list()) {
    NumTotalDims += BSet.dim(isl::dim::div);
    NumTotalDims += BSet.dim(isl::dim::set);
  }

  if (NumTotalDims > MaxDimensionsInAccessRange)
    return true;

  return false;
}

bool ScopBuilder::hasNonHoistableBasePtrInScop(MemoryAccess *MA,
                                               isl::union_map Writes) {
  if (auto *BasePtrMA = scop->lookupBasePtrAccess(MA)) {
    return getNonHoistableCtx(BasePtrMA, Writes).is_null();
  }

  Value *BaseAddr = MA->getOriginalBaseAddr();
  if (auto *BasePtrInst = dyn_cast<Instruction>(BaseAddr))
    if (!isa<LoadInst>(BasePtrInst))
      return scop->contains(BasePtrInst);

  return false;
}

void ScopBuilder::addUserContext() {
  if (UserContextStr.empty())
    return;

  isl::set UserContext = isl::set(scop->getIslCtx(), UserContextStr.c_str());
  isl::space Space = scop->getParamSpace();
  if (Space.dim(isl::dim::param) != UserContext.dim(isl::dim::param)) {
    std::string SpaceStr = Space.to_str();
    errs() << "Error: the context provided in -polly-context has not the same "
           << "number of dimensions than the computed context. Due to this "
           << "mismatch, the -polly-context option is ignored. Please provide "
           << "the context in the parameter space: " << SpaceStr << ".\n";
    return;
  }

  for (unsigned i = 0; i < Space.dim(isl::dim::param); i++) {
    std::string NameContext =
        scop->getContext().get_dim_name(isl::dim::param, i);
    std::string NameUserContext = UserContext.get_dim_name(isl::dim::param, i);

    if (NameContext != NameUserContext) {
      std::string SpaceStr = Space.to_str();
      errs() << "Error: the name of dimension " << i
             << " provided in -polly-context "
             << "is '" << NameUserContext << "', but the name in the computed "
             << "context is '" << NameContext
             << "'. Due to this name mismatch, "
             << "the -polly-context option is ignored. Please provide "
             << "the context in the parameter space: " << SpaceStr << ".\n";
      return;
    }

    UserContext = UserContext.set_dim_id(isl::dim::param, i,
                                         Space.get_dim_id(isl::dim::param, i));
  }
  isl::set newContext = scop->getContext().intersect(UserContext);
  scop->setContext(newContext);
}

isl::set ScopBuilder::getNonHoistableCtx(MemoryAccess *Access,
                                         isl::union_map Writes) {
  // TODO: Loads that are not loop carried, hence are in a statement with
  //       zero iterators, are by construction invariant, though we
  //       currently "hoist" them anyway. This is necessary because we allow
  //       them to be treated as parameters (e.g., in conditions) and our code
  //       generation would otherwise use the old value.

  auto &Stmt = *Access->getStatement();
  BasicBlock *BB = Stmt.getEntryBlock();

  if (Access->isScalarKind() || Access->isWrite() || !Access->isAffine() ||
      Access->isMemoryIntrinsic())
    return nullptr;

  // Skip accesses that have an invariant base pointer which is defined but
  // not loaded inside the SCoP. This can happened e.g., if a readnone call
  // returns a pointer that is used as a base address. However, as we want
  // to hoist indirect pointers, we allow the base pointer to be defined in
  // the region if it is also a memory access. Each ScopArrayInfo object
  // that has a base pointer origin has a base pointer that is loaded and
  // that it is invariant, thus it will be hoisted too. However, if there is
  // no base pointer origin we check that the base pointer is defined
  // outside the region.
  auto *LI = cast<LoadInst>(Access->getAccessInstruction());
  if (hasNonHoistableBasePtrInScop(Access, Writes))
    return nullptr;

  isl::map AccessRelation = Access->getAccessRelation();
  assert(!AccessRelation.is_empty());

  if (AccessRelation.involves_dims(isl::dim::in, 0, Stmt.getNumIterators()))
    return nullptr;

  AccessRelation = AccessRelation.intersect_domain(Stmt.getDomain());
  isl::set SafeToLoad;

  auto &DL = scop->getFunction().getParent()->getDataLayout();
  if (isSafeToLoadUnconditionally(LI->getPointerOperand(), LI->getType(),
                                  LI->getAlign(), DL)) {
    SafeToLoad = isl::set::universe(AccessRelation.get_space().range());
  } else if (BB != LI->getParent()) {
    // Skip accesses in non-affine subregions as they might not be executed
    // under the same condition as the entry of the non-affine subregion.
    return nullptr;
  } else {
    SafeToLoad = AccessRelation.range();
  }

  if (isAccessRangeTooComplex(AccessRelation.range()))
    return nullptr;

  isl::union_map Written = Writes.intersect_range(SafeToLoad);
  isl::set WrittenCtx = Written.params();
  bool IsWritten = !WrittenCtx.is_empty();

  if (!IsWritten)
    return WrittenCtx;

  WrittenCtx = WrittenCtx.remove_divs();
  bool TooComplex = WrittenCtx.n_basic_set() >= MaxDisjunctsInDomain;
  if (TooComplex || !isRequiredInvariantLoad(LI))
    return nullptr;

  scop->addAssumption(INVARIANTLOAD, WrittenCtx, LI->getDebugLoc(),
                      AS_RESTRICTION, LI->getParent());
  return WrittenCtx;
}

static bool isAParameter(llvm::Value *maybeParam, const Function &F) {
  for (const llvm::Argument &Arg : F.args())
    if (&Arg == maybeParam)
      return true;

  return false;
}

bool ScopBuilder::canAlwaysBeHoisted(MemoryAccess *MA,
                                     bool StmtInvalidCtxIsEmpty,
                                     bool MAInvalidCtxIsEmpty,
                                     bool NonHoistableCtxIsEmpty) {
  LoadInst *LInst = cast<LoadInst>(MA->getAccessInstruction());
  const DataLayout &DL = LInst->getParent()->getModule()->getDataLayout();
  if (PollyAllowDereferenceOfAllFunctionParams &&
      isAParameter(LInst->getPointerOperand(), scop->getFunction()))
    return true;

  // TODO: We can provide more information for better but more expensive
  //       results.
  if (!isDereferenceableAndAlignedPointer(
          LInst->getPointerOperand(), LInst->getType(), LInst->getAlign(), DL))
    return false;

  // If the location might be overwritten we do not hoist it unconditionally.
  //
  // TODO: This is probably too conservative.
  if (!NonHoistableCtxIsEmpty)
    return false;

  // If a dereferenceable load is in a statement that is modeled precisely we
  // can hoist it.
  if (StmtInvalidCtxIsEmpty && MAInvalidCtxIsEmpty)
    return true;

  // Even if the statement is not modeled precisely we can hoist the load if it
  // does not involve any parameters that might have been specialized by the
  // statement domain.
  for (const SCEV *Subscript : MA->subscripts())
    if (!isa<SCEVConstant>(Subscript))
      return false;
  return true;
}

void ScopBuilder::addInvariantLoads(ScopStmt &Stmt,
                                    InvariantAccessesTy &InvMAs) {
  if (InvMAs.empty())
    return;

  isl::set StmtInvalidCtx = Stmt.getInvalidContext();
  bool StmtInvalidCtxIsEmpty = StmtInvalidCtx.is_empty();

  // Get the context under which the statement is executed but remove the error
  // context under which this statement is reached.
  isl::set DomainCtx = Stmt.getDomain().params();
  DomainCtx = DomainCtx.subtract(StmtInvalidCtx);

  if (DomainCtx.n_basic_set() >= MaxDisjunctsInDomain) {
    auto *AccInst = InvMAs.front().MA->getAccessInstruction();
    scop->invalidate(COMPLEXITY, AccInst->getDebugLoc(), AccInst->getParent());
    return;
  }

  // Project out all parameters that relate to loads in the statement. Otherwise
  // we could have cyclic dependences on the constraints under which the
  // hoisted loads are executed and we could not determine an order in which to
  // pre-load them. This happens because not only lower bounds are part of the
  // domain but also upper bounds.
  for (auto &InvMA : InvMAs) {
    auto *MA = InvMA.MA;
    Instruction *AccInst = MA->getAccessInstruction();
    if (SE.isSCEVable(AccInst->getType())) {
      SetVector<Value *> Values;
      for (const SCEV *Parameter : scop->parameters()) {
        Values.clear();
        findValues(Parameter, SE, Values);
        if (!Values.count(AccInst))
          continue;

        if (isl::id ParamId = scop->getIdForParam(Parameter)) {
          int Dim = DomainCtx.find_dim_by_id(isl::dim::param, ParamId);
          if (Dim >= 0)
            DomainCtx = DomainCtx.eliminate(isl::dim::param, Dim, 1);
        }
      }
    }
  }

  for (auto &InvMA : InvMAs) {
    auto *MA = InvMA.MA;
    isl::set NHCtx = InvMA.NonHoistableCtx;

    // Check for another invariant access that accesses the same location as
    // MA and if found consolidate them. Otherwise create a new equivalence
    // class at the end of InvariantEquivClasses.
    LoadInst *LInst = cast<LoadInst>(MA->getAccessInstruction());
    Type *Ty = LInst->getType();
    const SCEV *PointerSCEV = SE.getSCEV(LInst->getPointerOperand());

    isl::set MAInvalidCtx = MA->getInvalidContext();
    bool NonHoistableCtxIsEmpty = NHCtx.is_empty();
    bool MAInvalidCtxIsEmpty = MAInvalidCtx.is_empty();

    isl::set MACtx;
    // Check if we know that this pointer can be speculatively accessed.
    if (canAlwaysBeHoisted(MA, StmtInvalidCtxIsEmpty, MAInvalidCtxIsEmpty,
                           NonHoistableCtxIsEmpty)) {
      MACtx = isl::set::universe(DomainCtx.get_space());
    } else {
      MACtx = DomainCtx;
      MACtx = MACtx.subtract(MAInvalidCtx.unite(NHCtx));
      MACtx = MACtx.gist_params(scop->getContext());
    }

    bool Consolidated = false;
    for (auto &IAClass : scop->invariantEquivClasses()) {
      if (PointerSCEV != IAClass.IdentifyingPointer || Ty != IAClass.AccessType)
        continue;

      // If the pointer and the type is equal check if the access function wrt.
      // to the domain is equal too. It can happen that the domain fixes
      // parameter values and these can be different for distinct part of the
      // SCoP. If this happens we cannot consolidate the loads but need to
      // create a new invariant load equivalence class.
      auto &MAs = IAClass.InvariantAccesses;
      if (!MAs.empty()) {
        auto *LastMA = MAs.front();

        isl::set AR = MA->getAccessRelation().range();
        isl::set LastAR = LastMA->getAccessRelation().range();
        bool SameAR = AR.is_equal(LastAR);

        if (!SameAR)
          continue;
      }

      // Add MA to the list of accesses that are in this class.
      MAs.push_front(MA);

      Consolidated = true;

      // Unify the execution context of the class and this statement.
      isl::set IAClassDomainCtx = IAClass.ExecutionContext;
      if (IAClassDomainCtx)
        IAClassDomainCtx = IAClassDomainCtx.unite(MACtx).coalesce();
      else
        IAClassDomainCtx = MACtx;
      IAClass.ExecutionContext = IAClassDomainCtx;
      break;
    }

    if (Consolidated)
      continue;

    MACtx = MACtx.coalesce();

    // If we did not consolidate MA, thus did not find an equivalence class
    // for it, we create a new one.
    scop->addInvariantEquivClass(
        InvariantEquivClassTy{PointerSCEV, MemoryAccessList{MA}, MACtx, Ty});
  }
}

void ScopBuilder::collectCandidateReductionLoads(
    MemoryAccess *StoreMA, SmallVectorImpl<MemoryAccess *> &Loads) {
  ScopStmt *Stmt = StoreMA->getStatement();

  auto *Store = dyn_cast<StoreInst>(StoreMA->getAccessInstruction());
  if (!Store)
    return;

  // Skip if there is not one binary operator between the load and the store
  auto *BinOp = dyn_cast<BinaryOperator>(Store->getValueOperand());
  if (!BinOp)
    return;

  // Skip if the binary operators has multiple uses
  if (BinOp->getNumUses() != 1)
    return;

  // Skip if the opcode of the binary operator is not commutative/associative
  if (!BinOp->isCommutative() || !BinOp->isAssociative())
    return;

  // Skip if the binary operator is outside the current SCoP
  if (BinOp->getParent() != Store->getParent())
    return;

  // Skip if it is a multiplicative reduction and we disabled them
  if (DisableMultiplicativeReductions &&
      (BinOp->getOpcode() == Instruction::Mul ||
       BinOp->getOpcode() == Instruction::FMul))
    return;

  // Check the binary operator operands for a candidate load
  auto *PossibleLoad0 = dyn_cast<LoadInst>(BinOp->getOperand(0));
  auto *PossibleLoad1 = dyn_cast<LoadInst>(BinOp->getOperand(1));
  if (!PossibleLoad0 && !PossibleLoad1)
    return;

  // A load is only a candidate if it cannot escape (thus has only this use)
  if (PossibleLoad0 && PossibleLoad0->getNumUses() == 1)
    if (PossibleLoad0->getParent() == Store->getParent())
      Loads.push_back(&Stmt->getArrayAccessFor(PossibleLoad0));
  if (PossibleLoad1 && PossibleLoad1->getNumUses() == 1)
    if (PossibleLoad1->getParent() == Store->getParent())
      Loads.push_back(&Stmt->getArrayAccessFor(PossibleLoad1));
}

/// Find the canonical scop array info object for a set of invariant load
/// hoisted loads. The canonical array is the one that corresponds to the
/// first load in the list of accesses which is used as base pointer of a
/// scop array.
static const ScopArrayInfo *findCanonicalArray(Scop &S,
                                               MemoryAccessList &Accesses) {
  for (MemoryAccess *Access : Accesses) {
    const ScopArrayInfo *CanonicalArray = S.getScopArrayInfoOrNull(
        Access->getAccessInstruction(), MemoryKind::Array);
    if (CanonicalArray)
      return CanonicalArray;
  }
  return nullptr;
}

/// Check if @p Array severs as base array in an invariant load.
static bool isUsedForIndirectHoistedLoad(Scop &S, const ScopArrayInfo *Array) {
  for (InvariantEquivClassTy &EqClass2 : S.getInvariantAccesses())
    for (MemoryAccess *Access2 : EqClass2.InvariantAccesses)
      if (Access2->getScopArrayInfo() == Array)
        return true;
  return false;
}

/// Replace the base pointer arrays in all memory accesses referencing @p Old,
/// with a reference to @p New.
static void replaceBasePtrArrays(Scop &S, const ScopArrayInfo *Old,
                                 const ScopArrayInfo *New) {
  for (ScopStmt &Stmt : S)
    for (MemoryAccess *Access : Stmt) {
      if (Access->getLatestScopArrayInfo() != Old)
        continue;

      isl::id Id = New->getBasePtrId();
      isl::map Map = Access->getAccessRelation();
      Map = Map.set_tuple_id(isl::dim::out, Id);
      Access->setAccessRelation(Map);
    }
}

void ScopBuilder::canonicalizeDynamicBasePtrs() {
  for (InvariantEquivClassTy &EqClass : scop->InvariantEquivClasses) {
    MemoryAccessList &BasePtrAccesses = EqClass.InvariantAccesses;

    const ScopArrayInfo *CanonicalBasePtrSAI =
        findCanonicalArray(*scop, BasePtrAccesses);

    if (!CanonicalBasePtrSAI)
      continue;

    for (MemoryAccess *BasePtrAccess : BasePtrAccesses) {
      const ScopArrayInfo *BasePtrSAI = scop->getScopArrayInfoOrNull(
          BasePtrAccess->getAccessInstruction(), MemoryKind::Array);
      if (!BasePtrSAI || BasePtrSAI == CanonicalBasePtrSAI ||
          !BasePtrSAI->isCompatibleWith(CanonicalBasePtrSAI))
        continue;

      // we currently do not canonicalize arrays where some accesses are
      // hoisted as invariant loads. If we would, we need to update the access
      // function of the invariant loads as well. However, as this is not a
      // very common situation, we leave this for now to avoid further
      // complexity increases.
      if (isUsedForIndirectHoistedLoad(*scop, BasePtrSAI))
        continue;

      replaceBasePtrArrays(*scop, BasePtrSAI, CanonicalBasePtrSAI);
    }
  }
}

void ScopBuilder::buildAccessRelations(ScopStmt &Stmt) {
  for (MemoryAccess *Access : Stmt.MemAccs) {
    Type *ElementType = Access->getElementType();

    MemoryKind Ty;
    if (Access->isPHIKind())
      Ty = MemoryKind::PHI;
    else if (Access->isExitPHIKind())
      Ty = MemoryKind::ExitPHI;
    else if (Access->isValueKind())
      Ty = MemoryKind::Value;
    else
      Ty = MemoryKind::Array;

    // Create isl::pw_aff for SCEVs which describe sizes. Collect all
    // assumptions which are taken. isl::pw_aff objects are cached internally
    // and they are used later by scop.
    for (const SCEV *Size : Access->Sizes) {
      if (!Size)
        continue;
      scop->getPwAff(Size, nullptr, false, &RecordedAssumptions);
    }
    auto *SAI = scop->getOrCreateScopArrayInfo(Access->getOriginalBaseAddr(),
                                               ElementType, Access->Sizes, Ty);

    // Create isl::pw_aff for SCEVs which describe subscripts. Collect all
    // assumptions which are taken. isl::pw_aff objects are cached internally
    // and they are used later by scop.
    for (const SCEV *Subscript : Access->subscripts()) {
      if (!Access->isAffine() || !Subscript)
        continue;
      scop->getPwAff(Subscript, Stmt.getEntryBlock(), false,
                     &RecordedAssumptions);
    }
    Access->buildAccessRelation(SAI);
    scop->addAccessData(Access);
  }
}

/// Add the minimal/maximal access in @p Set to @p User.
///
/// @return True if more accesses should be added, false if we reached the
///         maximal number of run-time checks to be generated.
static bool buildMinMaxAccess(isl::set Set,
                              Scop::MinMaxVectorTy &MinMaxAccesses, Scop &S) {
  isl::pw_multi_aff MinPMA, MaxPMA;
  isl::pw_aff LastDimAff;
  isl::aff OneAff;
  unsigned Pos;

  Set = Set.remove_divs();
  polly::simplify(Set);

  if (Set.n_basic_set() > RunTimeChecksMaxAccessDisjuncts)
    Set = Set.simple_hull();

  // Restrict the number of parameters involved in the access as the lexmin/
  // lexmax computation will take too long if this number is high.
  //
  // Experiments with a simple test case using an i7 4800MQ:
  //
  //  #Parameters involved | Time (in sec)
  //            6          |     0.01
  //            7          |     0.04
  //            8          |     0.12
  //            9          |     0.40
  //           10          |     1.54
  //           11          |     6.78
  //           12          |    30.38
  //
  if (isl_set_n_param(Set.get()) >
      static_cast<isl_size>(RunTimeChecksMaxParameters)) {
    unsigned InvolvedParams = 0;
    for (unsigned u = 0, e = isl_set_n_param(Set.get()); u < e; u++)
      if (Set.involves_dims(isl::dim::param, u, 1))
        InvolvedParams++;

    if (InvolvedParams > RunTimeChecksMaxParameters)
      return false;
  }

  MinPMA = Set.lexmin_pw_multi_aff();
  MaxPMA = Set.lexmax_pw_multi_aff();

  MinPMA = MinPMA.coalesce();
  MaxPMA = MaxPMA.coalesce();

  // Adjust the last dimension of the maximal access by one as we want to
  // enclose the accessed memory region by MinPMA and MaxPMA. The pointer
  // we test during code generation might now point after the end of the
  // allocated array but we will never dereference it anyway.
  assert((!MaxPMA || MaxPMA.dim(isl::dim::out)) &&
         "Assumed at least one output dimension");

  Pos = MaxPMA.dim(isl::dim::out) - 1;
  LastDimAff = MaxPMA.get_pw_aff(Pos);
  OneAff = isl::aff(isl::local_space(LastDimAff.get_domain_space()));
  OneAff = OneAff.add_constant_si(1);
  LastDimAff = LastDimAff.add(OneAff);
  MaxPMA = MaxPMA.set_pw_aff(Pos, LastDimAff);

  if (!MinPMA || !MaxPMA)
    return false;

  MinMaxAccesses.push_back(std::make_pair(MinPMA, MaxPMA));

  return true;
}

/// Wrapper function to calculate minimal/maximal accesses to each array.
bool ScopBuilder::calculateMinMaxAccess(AliasGroupTy AliasGroup,
                                        Scop::MinMaxVectorTy &MinMaxAccesses) {
  MinMaxAccesses.reserve(AliasGroup.size());

  isl::union_set Domains = scop->getDomains();
  isl::union_map Accesses = isl::union_map::empty(scop->getParamSpace());

  for (MemoryAccess *MA : AliasGroup)
    Accesses = Accesses.add_map(MA->getAccessRelation());

  Accesses = Accesses.intersect_domain(Domains);
  isl::union_set Locations = Accesses.range();

  bool LimitReached = false;
  for (isl::set Set : Locations.get_set_list()) {
    LimitReached |= !buildMinMaxAccess(Set, MinMaxAccesses, *scop);
    if (LimitReached)
      break;
  }

  return !LimitReached;
}

static isl::set getAccessDomain(MemoryAccess *MA) {
  isl::set Domain = MA->getStatement()->getDomain();
  Domain = Domain.project_out(isl::dim::set, 0, Domain.n_dim());
  return Domain.reset_tuple_id();
}

bool ScopBuilder::buildAliasChecks() {
  if (!PollyUseRuntimeAliasChecks)
    return true;

  if (buildAliasGroups()) {
    // Aliasing assumptions do not go through addAssumption but we still want to
    // collect statistics so we do it here explicitly.
    if (scop->getAliasGroups().size())
      Scop::incrementNumberOfAliasingAssumptions(1);
    return true;
  }

  // If a problem occurs while building the alias groups we need to delete
  // this SCoP and pretend it wasn't valid in the first place. To this end
  // we make the assumed context infeasible.
  scop->invalidate(ALIASING, DebugLoc());

  LLVM_DEBUG(
      dbgs() << "\n\nNOTE: Run time checks for " << scop->getNameStr()
             << " could not be created as the number of parameters involved "
                "is too high. The SCoP will be "
                "dismissed.\nUse:\n\t--polly-rtc-max-parameters=X\nto adjust "
                "the maximal number of parameters but be advised that the "
                "compile time might increase exponentially.\n\n");
  return false;
}

std::tuple<ScopBuilder::AliasGroupVectorTy, DenseSet<const ScopArrayInfo *>>
ScopBuilder::buildAliasGroupsForAccesses() {
  AliasSetTracker AST(AA);

  DenseMap<Value *, MemoryAccess *> PtrToAcc;
  DenseSet<const ScopArrayInfo *> HasWriteAccess;
  for (ScopStmt &Stmt : *scop) {

    isl::set StmtDomain = Stmt.getDomain();
    bool StmtDomainEmpty = StmtDomain.is_empty();

    // Statements with an empty domain will never be executed.
    if (StmtDomainEmpty)
      continue;

    for (MemoryAccess *MA : Stmt) {
      if (MA->isScalarKind())
        continue;
      if (!MA->isRead())
        HasWriteAccess.insert(MA->getScopArrayInfo());
      MemAccInst Acc(MA->getAccessInstruction());
      if (MA->isRead() && isa<MemTransferInst>(Acc))
        PtrToAcc[cast<MemTransferInst>(Acc)->getRawSource()] = MA;
      else
        PtrToAcc[Acc.getPointerOperand()] = MA;
      AST.add(Acc);
    }
  }

  AliasGroupVectorTy AliasGroups;
  for (AliasSet &AS : AST) {
    if (AS.isMustAlias() || AS.isForwardingAliasSet())
      continue;
    AliasGroupTy AG;
    for (auto &PR : AS)
      AG.push_back(PtrToAcc[PR.getValue()]);
    if (AG.size() < 2)
      continue;
    AliasGroups.push_back(std::move(AG));
  }

  return std::make_tuple(AliasGroups, HasWriteAccess);
}

bool ScopBuilder::buildAliasGroups() {
  // To create sound alias checks we perform the following steps:
  //   o) We partition each group into read only and non read only accesses.
  //   o) For each group with more than one base pointer we then compute minimal
  //      and maximal accesses to each array of a group in read only and non
  //      read only partitions separately.
  AliasGroupVectorTy AliasGroups;
  DenseSet<const ScopArrayInfo *> HasWriteAccess;

  std::tie(AliasGroups, HasWriteAccess) = buildAliasGroupsForAccesses();

  splitAliasGroupsByDomain(AliasGroups);

  for (AliasGroupTy &AG : AliasGroups) {
    if (!scop->hasFeasibleRuntimeContext())
      return false;

    {
      IslMaxOperationsGuard MaxOpGuard(scop->getIslCtx().get(), OptComputeOut);
      bool Valid = buildAliasGroup(AG, HasWriteAccess);
      if (!Valid)
        return false;
    }
    if (isl_ctx_last_error(scop->getIslCtx().get()) == isl_error_quota) {
      scop->invalidate(COMPLEXITY, DebugLoc());
      return false;
    }
  }

  return true;
}

bool ScopBuilder::buildAliasGroup(
    AliasGroupTy &AliasGroup, DenseSet<const ScopArrayInfo *> HasWriteAccess) {
  AliasGroupTy ReadOnlyAccesses;
  AliasGroupTy ReadWriteAccesses;
  SmallPtrSet<const ScopArrayInfo *, 4> ReadWriteArrays;
  SmallPtrSet<const ScopArrayInfo *, 4> ReadOnlyArrays;

  if (AliasGroup.size() < 2)
    return true;

  for (MemoryAccess *Access : AliasGroup) {
    ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "PossibleAlias",
                                        Access->getAccessInstruction())
             << "Possibly aliasing pointer, use restrict keyword.");
    const ScopArrayInfo *Array = Access->getScopArrayInfo();
    if (HasWriteAccess.count(Array)) {
      ReadWriteArrays.insert(Array);
      ReadWriteAccesses.push_back(Access);
    } else {
      ReadOnlyArrays.insert(Array);
      ReadOnlyAccesses.push_back(Access);
    }
  }

  // If there are no read-only pointers, and less than two read-write pointers,
  // no alias check is needed.
  if (ReadOnlyAccesses.empty() && ReadWriteArrays.size() <= 1)
    return true;

  // If there is no read-write pointer, no alias check is needed.
  if (ReadWriteArrays.empty())
    return true;

  // For non-affine accesses, no alias check can be generated as we cannot
  // compute a sufficiently tight lower and upper bound: bail out.
  for (MemoryAccess *MA : AliasGroup) {
    if (!MA->isAffine()) {
      scop->invalidate(ALIASING, MA->getAccessInstruction()->getDebugLoc(),
                       MA->getAccessInstruction()->getParent());
      return false;
    }
  }

  // Ensure that for all memory accesses for which we generate alias checks,
  // their base pointers are available.
  for (MemoryAccess *MA : AliasGroup) {
    if (MemoryAccess *BasePtrMA = scop->lookupBasePtrAccess(MA))
      scop->addRequiredInvariantLoad(
          cast<LoadInst>(BasePtrMA->getAccessInstruction()));
  }

  //  scop->getAliasGroups().emplace_back();
  //  Scop::MinMaxVectorPairTy &pair = scop->getAliasGroups().back();
  Scop::MinMaxVectorTy MinMaxAccessesReadWrite;
  Scop::MinMaxVectorTy MinMaxAccessesReadOnly;

  bool Valid;

  Valid = calculateMinMaxAccess(ReadWriteAccesses, MinMaxAccessesReadWrite);

  if (!Valid)
    return false;

  // Bail out if the number of values we need to compare is too large.
  // This is important as the number of comparisons grows quadratically with
  // the number of values we need to compare.
  if (MinMaxAccessesReadWrite.size() + ReadOnlyArrays.size() >
      RunTimeChecksMaxArraysPerGroup)
    return false;

  Valid = calculateMinMaxAccess(ReadOnlyAccesses, MinMaxAccessesReadOnly);

  scop->addAliasGroup(MinMaxAccessesReadWrite, MinMaxAccessesReadOnly);
  if (!Valid)
    return false;

  return true;
}

void ScopBuilder::splitAliasGroupsByDomain(AliasGroupVectorTy &AliasGroups) {
  for (unsigned u = 0; u < AliasGroups.size(); u++) {
    AliasGroupTy NewAG;
    AliasGroupTy &AG = AliasGroups[u];
    AliasGroupTy::iterator AGI = AG.begin();
    isl::set AGDomain = getAccessDomain(*AGI);
    while (AGI != AG.end()) {
      MemoryAccess *MA = *AGI;
      isl::set MADomain = getAccessDomain(MA);
      if (AGDomain.is_disjoint(MADomain)) {
        NewAG.push_back(MA);
        AGI = AG.erase(AGI);
      } else {
        AGDomain = AGDomain.unite(MADomain);
        AGI++;
      }
    }
    if (NewAG.size() > 1)
      AliasGroups.push_back(std::move(NewAG));
  }
}

#ifndef NDEBUG
static void verifyUse(Scop *S, Use &Op, LoopInfo &LI) {
  auto PhysUse = VirtualUse::create(S, Op, &LI, false);
  auto VirtUse = VirtualUse::create(S, Op, &LI, true);
  assert(PhysUse.getKind() == VirtUse.getKind());
}

/// Check the consistency of every statement's MemoryAccesses.
///
/// The check is carried out by expecting the "physical" kind of use (derived
/// from the BasicBlocks instructions resides in) to be same as the "virtual"
/// kind of use (derived from a statement's MemoryAccess).
///
/// The "physical" uses are taken by ensureValueRead to determine whether to
/// create MemoryAccesses. When done, the kind of scalar access should be the
/// same no matter which way it was derived.
///
/// The MemoryAccesses might be changed by later SCoP-modifying passes and hence
/// can intentionally influence on the kind of uses (not corresponding to the
/// "physical" anymore, hence called "virtual"). The CodeGenerator therefore has
/// to pick up the virtual uses. But here in the code generator, this has not
/// happened yet, such that virtual and physical uses are equivalent.
static void verifyUses(Scop *S, LoopInfo &LI, DominatorTree &DT) {
  for (auto *BB : S->getRegion().blocks()) {
    for (auto &Inst : *BB) {
      auto *Stmt = S->getStmtFor(&Inst);
      if (!Stmt)
        continue;

      if (isIgnoredIntrinsic(&Inst))
        continue;

      // Branch conditions are encoded in the statement domains.
      if (Inst.isTerminator() && Stmt->isBlockStmt())
        continue;

      // Verify all uses.
      for (auto &Op : Inst.operands())
        verifyUse(S, Op, LI);

      // Stores do not produce values used by other statements.
      if (isa<StoreInst>(Inst))
        continue;

      // For every value defined in the block, also check that a use of that
      // value in the same statement would not be an inter-statement use. It can
      // still be synthesizable or load-hoisted, but these kind of instructions
      // are not directly copied in code-generation.
      auto VirtDef =
          VirtualUse::create(S, Stmt, Stmt->getSurroundingLoop(), &Inst, true);
      assert(VirtDef.getKind() == VirtualUse::Synthesizable ||
             VirtDef.getKind() == VirtualUse::Intra ||
             VirtDef.getKind() == VirtualUse::Hoisted);
    }
  }

  if (S->hasSingleExitEdge())
    return;

  // PHINodes in the SCoP region's exit block are also uses to be checked.
  if (!S->getRegion().isTopLevelRegion()) {
    for (auto &Inst : *S->getRegion().getExit()) {
      if (!isa<PHINode>(Inst))
        break;

      for (auto &Op : Inst.operands())
        verifyUse(S, Op, LI);
    }
  }
}
#endif

void ScopBuilder::buildScop(Region &R, AssumptionCache &AC) {
  scop.reset(new Scop(R, SE, LI, DT, *SD.getDetectionContext(&R), ORE,
                      SD.getNextID()));

  buildStmts(R);

  // Create all invariant load instructions first. These are categorized as
  // 'synthesizable', therefore are not part of any ScopStmt but need to be
  // created somewhere.
  const InvariantLoadsSetTy &RIL = scop->getRequiredInvariantLoads();
  for (BasicBlock *BB : scop->getRegion().blocks()) {
    if (isErrorBlock(*BB, scop->getRegion(), LI, DT))
      continue;

    for (Instruction &Inst : *BB) {
      LoadInst *Load = dyn_cast<LoadInst>(&Inst);
      if (!Load)
        continue;

      if (!RIL.count(Load))
        continue;

      // Invariant loads require a MemoryAccess to be created in some statement.
      // It is not important to which statement the MemoryAccess is added
      // because it will later be removed from the ScopStmt again. We chose the
      // first statement of the basic block the LoadInst is in.
      ArrayRef<ScopStmt *> List = scop->getStmtListFor(BB);
      assert(!List.empty());
      ScopStmt *RILStmt = List.front();
      buildMemoryAccess(Load, RILStmt);
    }
  }
  buildAccessFunctions();

  // In case the region does not have an exiting block we will later (during
  // code generation) split the exit block. This will move potential PHI nodes
  // from the current exit block into the new region exiting block. Hence, PHI
  // nodes that are at this point not part of the region will be.
  // To handle these PHI nodes later we will now model their operands as scalar
  // accesses. Note that we do not model anything in the exit block if we have
  // an exiting block in the region, as there will not be any splitting later.
  if (!R.isTopLevelRegion() && !scop->hasSingleExitEdge()) {
    for (Instruction &Inst : *R.getExit()) {
      PHINode *PHI = dyn_cast<PHINode>(&Inst);
      if (!PHI)
        break;

      buildPHIAccesses(nullptr, PHI, nullptr, true);
    }
  }

  // Create memory accesses for global reads since all arrays are now known.
  auto *AF = SE.getConstant(IntegerType::getInt64Ty(SE.getContext()), 0);
  for (auto GlobalReadPair : GlobalReads) {
    ScopStmt *GlobalReadStmt = GlobalReadPair.first;
    Instruction *GlobalRead = GlobalReadPair.second;
    for (auto *BP : ArrayBasePointers)
      addArrayAccess(GlobalReadStmt, MemAccInst(GlobalRead), MemoryAccess::READ,
                     BP, BP->getType(), false, {AF}, {nullptr}, GlobalRead);
  }

  buildInvariantEquivalenceClasses();

  /// A map from basic blocks to their invalid domains.
  DenseMap<BasicBlock *, isl::set> InvalidDomainMap;

  if (!buildDomains(&R, InvalidDomainMap)) {
    LLVM_DEBUG(
        dbgs() << "Bailing-out because buildDomains encountered problems\n");
    return;
  }

  addUserAssumptions(AC, InvalidDomainMap);

  // Initialize the invalid domain.
  for (ScopStmt &Stmt : scop->Stmts)
    if (Stmt.isBlockStmt())
      Stmt.setInvalidDomain(InvalidDomainMap[Stmt.getEntryBlock()]);
    else
      Stmt.setInvalidDomain(InvalidDomainMap[getRegionNodeBasicBlock(
          Stmt.getRegion()->getNode())]);

  // Remove empty statements.
  // Exit early in case there are no executable statements left in this scop.
  scop->removeStmtNotInDomainMap();
  scop->simplifySCoP(false);
  if (scop->isEmpty()) {
    LLVM_DEBUG(dbgs() << "Bailing-out because SCoP is empty\n");
    return;
  }

  // The ScopStmts now have enough information to initialize themselves.
  for (ScopStmt &Stmt : *scop) {
    collectSurroundingLoops(Stmt);

    buildDomain(Stmt);
    buildAccessRelations(Stmt);

    if (DetectReductions)
      checkForReductions(Stmt);
  }

  // Check early for a feasible runtime context.
  if (!scop->hasFeasibleRuntimeContext()) {
    LLVM_DEBUG(dbgs() << "Bailing-out because of unfeasible context (early)\n");
    return;
  }

  // Check early for profitability. Afterwards it cannot change anymore,
  // only the runtime context could become infeasible.
  if (!scop->isProfitable(UnprofitableScalarAccs)) {
    scop->invalidate(PROFITABLE, DebugLoc());
    LLVM_DEBUG(
        dbgs() << "Bailing-out because SCoP is not considered profitable\n");
    return;
  }

  buildSchedule();

  finalizeAccesses();

  scop->realignParams();
  addUserContext();

  // After the context was fully constructed, thus all our knowledge about
  // the parameters is in there, we add all recorded assumptions to the
  // assumed/invalid context.
  addRecordedAssumptions();

  scop->simplifyContexts();
  if (!buildAliasChecks()) {
    LLVM_DEBUG(dbgs() << "Bailing-out because could not build alias checks\n");
    return;
  }

  hoistInvariantLoads();
  canonicalizeDynamicBasePtrs();
  verifyInvariantLoads();
  scop->simplifySCoP(true);

  // Check late for a feasible runtime context because profitability did not
  // change.
  if (!scop->hasFeasibleRuntimeContext()) {
    LLVM_DEBUG(dbgs() << "Bailing-out because of unfeasible context (late)\n");
    return;
  }

#ifndef NDEBUG
  verifyUses(scop.get(), LI, DT);
#endif
}

ScopBuilder::ScopBuilder(Region *R, AssumptionCache &AC, AliasAnalysis &AA,
                         const DataLayout &DL, DominatorTree &DT, LoopInfo &LI,
                         ScopDetection &SD, ScalarEvolution &SE,
                         OptimizationRemarkEmitter &ORE)
    : AA(AA), DL(DL), DT(DT), LI(LI), SD(SD), SE(SE), ORE(ORE) {
  DebugLoc Beg, End;
  auto P = getBBPairForRegion(R);
  getDebugLocations(P, Beg, End);

  std::string Msg = "SCoP begins here.";
  ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "ScopEntry", Beg, P.first)
           << Msg);

  buildScop(*R, AC);

  LLVM_DEBUG(dbgs() << *scop);

  if (!scop->hasFeasibleRuntimeContext()) {
    InfeasibleScops++;
    Msg = "SCoP ends here but was dismissed.";
    LLVM_DEBUG(dbgs() << "SCoP detected but dismissed\n");
    RecordedAssumptions.clear();
    scop.reset();
  } else {
    Msg = "SCoP ends here.";
    ++ScopFound;
    if (scop->getMaxLoopDepth() > 0)
      ++RichScopFound;
  }

  if (R->isTopLevelRegion())
    ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "ScopEnd", End, P.first)
             << Msg);
  else
    ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "ScopEnd", End, P.second)
             << Msg);
}
