//===- LoopVectorize.cpp - A Loop Vectorizer ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the LLVM loop vectorizer. This pass modifies 'vectorizable' loops
// and generates target-independent LLVM-IR.
// The vectorizer uses the TargetTransformInfo analysis to estimate the costs
// of instructions in order to estimate the profitability of vectorization.
//
// The loop vectorizer combines consecutive loop iterations into a single
// 'wide' iteration. After this transformation the index is incremented
// by the SIMD vector width, and not by one.
//
// This pass has three parts:
// 1. The main loop pass that drives the different parts.
// 2. LoopVectorizationLegality - A unit that checks for the legality
//    of the vectorization.
// 3. InnerLoopVectorizer - A unit that performs the actual
//    widening of instructions.
// 4. LoopVectorizationCostModel - A unit that checks for the profitability
//    of vectorization. It decides on the optimal vector width, which
//    can be one, if vectorization is not profitable.
//
// There is a development effort going on to migrate loop vectorizer to the
// VPlan infrastructure and to introduce outer loop vectorization support (see
// docs/VectorizationPlan.rst and
// http://lists.llvm.org/pipermail/llvm-dev/2017-December/119523.html). For this
// purpose, we temporarily introduced the VPlan-native vectorization path: an
// alternative vectorization path that is natively implemented on top of the
// VPlan infrastructure. See EnableVPlanNativePath for enabling.
//
//===----------------------------------------------------------------------===//
//
// The reduction-variable vectorization is based on the paper:
//  D. Nuzman and R. Henderson. Multi-platform Auto-vectorization.
//
// Variable uniformity checks are inspired by:
//  Karrenberg, R. and Hack, S. Whole Function Vectorization.
//
// The interleaved access vectorization is based on the paper:
//  Dorit Nuzman, Ira Rosen and Ayal Zaks.  Auto-Vectorization of Interleaved
//  Data for SIMD
//
// Other ideas/concepts are from:
//  A. Zaks and D. Nuzman. Autovectorization in GCC-two years later.
//
//  S. Maleki, Y. Gao, M. Garzaran, T. Wong and D. Padua.  An Evaluation of
//  Vectorizing Compilers.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/LoopVectorize.h"
#include "LoopVectorizationPlanner.h"
#include "VPRecipeBuilder.h"
#include "VPlan.h"
#include "VPlanAnalysis.h"
#include "VPlanCFG.h"
#include "VPlanHelpers.h"
#include "VPlanPatternMatch.h"
#include "VPlanTransforms.h"
#include "VPlanUtils.h"
#include "VPlanVerifier.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolutionPatternMatch.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/InjectTLIMappings.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
#include "llvm/Transforms/Vectorize/LoopVectorizationLegality.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

using namespace llvm;
using namespace SCEVPatternMatch;

#define LV_NAME "loop-vectorize"
#define DEBUG_TYPE LV_NAME

#ifndef NDEBUG
const char VerboseDebug[] = DEBUG_TYPE "-verbose";
#endif

STATISTIC(LoopsVectorized, "Number of loops vectorized");
STATISTIC(LoopsAnalyzed, "Number of loops analyzed for vectorization");
STATISTIC(LoopsEpilogueVectorized, "Number of epilogues vectorized");
STATISTIC(LoopsEarlyExitVectorized, "Number of early exit loops vectorized");

static cl::opt<bool> EnableEpilogueVectorization(
    "enable-epilogue-vectorization", cl::init(true), cl::Hidden,
    cl::desc("Enable vectorization of epilogue loops."));

static cl::opt<unsigned> EpilogueVectorizationForceVF(
    "epilogue-vectorization-force-VF", cl::init(1), cl::Hidden,
    cl::desc("When epilogue vectorization is enabled, and a value greater than "
             "1 is specified, forces the given VF for all applicable epilogue "
             "loops."));

static cl::opt<unsigned> EpilogueVectorizationMinVF(
    "epilogue-vectorization-minimum-VF", cl::Hidden,
    cl::desc("Only loops with vectorization factor equal to or larger than "
             "the specified value are considered for epilogue vectorization."));

/// Loops with a known constant trip count below this number are vectorized only
/// if no scalar iteration overheads are incurred.
static cl::opt<unsigned> TinyTripCountVectorThreshold(
    "vectorizer-min-trip-count", cl::init(16), cl::Hidden,
    cl::desc("Loops with a constant trip count that is smaller than this "
             "value are vectorized only if no scalar iteration overheads "
             "are incurred."));

static cl::opt<unsigned> VectorizeMemoryCheckThreshold(
    "vectorize-memory-check-threshold", cl::init(128), cl::Hidden,
    cl::desc("The maximum allowed number of runtime memory checks"));

/// Option tail-folding-policy indicates that an epilogue is undesired, that
/// tail folding is preferred, and this lists all options. I.e., the vectorizer
/// will try to fold the tail-loop (epilogue) into the vector body and predicate
/// the instructions accordingly. If tail-folding fails, there are different
/// fallback strategies depending on these values:
enum class TailFoldingPolicyTy { None = 0, PreferFoldTail, MustFoldTail };

static cl::opt<TailFoldingPolicyTy> TailFoldingPolicy(
    "tail-folding-policy", cl::init(TailFoldingPolicyTy::None), cl::Hidden,
    cl::desc("Tail-folding preferences over creating an epilogue loop."),
    cl::values(
        clEnumValN(TailFoldingPolicyTy::None, "dont-fold-tail",
                   "Don't tail-fold loops."),
        clEnumValN(TailFoldingPolicyTy::PreferFoldTail, "prefer-fold-tail",
                   "prefer tail-folding, otherwise create an epilogue when "
                   "appropriate."),
        clEnumValN(TailFoldingPolicyTy::MustFoldTail, "must-fold-tail",
                   "always tail-fold, don't attempt vectorization if "
                   "tail-folding fails.")));

static cl::opt<TailFoldingStyle> ForceTailFoldingStyle(
    "force-tail-folding-style", cl::desc("Force the tail folding style"),
    cl::init(TailFoldingStyle::None),
    cl::values(
        clEnumValN(TailFoldingStyle::None, "none", "Disable tail folding"),
        clEnumValN(
            TailFoldingStyle::Data, "data",
            "Create lane mask for data only, using active.lane.mask intrinsic"),
        clEnumValN(TailFoldingStyle::DataWithoutLaneMask,
                   "data-without-lane-mask",
                   "Create lane mask with compare/stepvector"),
        clEnumValN(TailFoldingStyle::DataAndControlFlow, "data-and-control",
                   "Create lane mask using active.lane.mask intrinsic, and use "
                   "it for both data and control flow"),
        clEnumValN(TailFoldingStyle::DataWithEVL, "data-with-evl",
                   "Use predicated EVL instructions for tail folding. If EVL "
                   "is unsupported, fallback to data-without-lane-mask.")));

cl::opt<bool> llvm::EnableWideActiveLaneMask(
    "enable-wide-lane-mask", cl::init(false), cl::Hidden,
    cl::desc("Enable use of wide lane masks when used for control flow in "
             "tail-folded loops"));

static cl::opt<bool> EnableInterleavedMemAccesses(
    "enable-interleaved-mem-accesses", cl::init(false), cl::Hidden,
    cl::desc("Enable vectorization on interleaved memory accesses in a loop"));

/// An interleave-group may need masking if it resides in a block that needs
/// predication, or in order to mask away gaps.
static cl::opt<bool> EnableMaskedInterleavedMemAccesses(
    "enable-masked-interleaved-mem-accesses", cl::init(false), cl::Hidden,
    cl::desc("Enable vectorization on masked interleaved memory accesses in a loop"));

static cl::opt<unsigned> ForceTargetNumScalarRegs(
    "force-target-num-scalar-regs", cl::init(0), cl::Hidden,
    cl::desc("A flag that overrides the target's number of scalar registers."));

static cl::opt<unsigned> ForceTargetNumVectorRegs(
    "force-target-num-vector-regs", cl::init(0), cl::Hidden,
    cl::desc("A flag that overrides the target's number of vector registers."));

static cl::opt<unsigned> ForceTargetMaxScalarInterleaveFactor(
    "force-target-max-scalar-interleave", cl::init(0), cl::Hidden,
    cl::desc("A flag that overrides the target's max interleave factor for "
             "scalar loops."));

static cl::opt<unsigned> ForceTargetMaxVectorInterleaveFactor(
    "force-target-max-vector-interleave", cl::init(0), cl::Hidden,
    cl::desc("A flag that overrides the target's max interleave factor for "
             "vectorized loops."));

cl::opt<unsigned> llvm::ForceTargetInstructionCost(
    "force-target-instruction-cost", cl::init(0), cl::Hidden,
    cl::desc("A flag that overrides the target's expected cost for "
             "an instruction to a single constant value. Mostly "
             "useful for getting consistent testing."));

static cl::opt<unsigned> SmallLoopCost(
    "small-loop-cost", cl::init(20), cl::Hidden,
    cl::desc(
        "The cost of a loop that is considered 'small' by the interleaver."));

static cl::opt<bool> LoopVectorizeWithBlockFrequency(
    "loop-vectorize-with-block-frequency", cl::init(true), cl::Hidden,
    cl::desc("Enable the use of the block frequency analysis to access PGO "
             "heuristics minimizing code growth in cold regions and being more "
             "aggressive in hot regions."));

// Runtime interleave loops for load/store throughput.
static cl::opt<bool> EnableLoadStoreRuntimeInterleave(
    "enable-loadstore-runtime-interleave", cl::init(true), cl::Hidden,
    cl::desc(
        "Enable runtime interleaving until load/store ports are saturated"));

/// The number of stores in a loop that are allowed to need predication.
cl::opt<unsigned> NumberOfStoresToPredicate(
    "vectorize-num-stores-pred", cl::init(1), cl::Hidden,
    cl::desc("Max number of stores to be predicated behind an if."));

static cl::opt<bool> EnableIndVarRegisterHeur(
    "enable-ind-var-reg-heur", cl::init(true), cl::Hidden,
    cl::desc("Count the induction variable only once when interleaving"));

static cl::opt<unsigned> MaxNestedScalarReductionIC(
    "max-nested-scalar-reduction-interleave", cl::init(2), cl::Hidden,
    cl::desc("The maximum interleave count to use when interleaving a scalar "
             "reduction in a nested loop."));

static cl::opt<bool> ForceOrderedReductions(
    "force-ordered-reductions", cl::init(false), cl::Hidden,
    cl::desc("Enable the vectorisation of loops with in-order (strict) "
             "FP reductions"));

static cl::opt<bool> PreferPredicatedReductionSelect(
    "prefer-predicated-reduction-select", cl::init(false), cl::Hidden,
    cl::desc(
        "Prefer predicating a reduction operation over an after loop select."));

cl::opt<bool> llvm::EnableVPlanNativePath(
    "enable-vplan-native-path", cl::Hidden,
    cl::desc("Enable VPlan-native vectorization path with "
             "support for outer loop vectorization."));

cl::opt<bool>
    llvm::VerifyEachVPlan("vplan-verify-each",
#ifdef EXPENSIVE_CHECKS
                          cl::init(true),
#else
                          cl::init(false),
#endif
                          cl::Hidden,
                          cl::desc("Verify VPlans after VPlan transforms."));

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
cl::opt<bool> llvm::VPlanPrintAfterAll(
    "vplan-print-after-all", cl::init(false), cl::Hidden,
    cl::desc("Print VPlans after all VPlan transformations."));

cl::list<std::string> llvm::VPlanPrintAfterPasses(
    "vplan-print-after", cl::Hidden,
    cl::desc("Print VPlans after specified VPlan transformations (regexp)."));

cl::opt<bool> llvm::VPlanPrintVectorRegionScope(
    "vplan-print-vector-region-scope", cl::init(false), cl::Hidden,
    cl::desc("Limit VPlan printing to vector loop region in "
             "`-vplan-print-after*` if the plan has one."));
#endif

// This flag enables the stress testing of the VPlan H-CFG construction in the
// VPlan-native vectorization path. It must be used in conjuction with
// -enable-vplan-native-path. -vplan-verify-hcfg can also be used to enable the
// verification of the H-CFGs built.
static cl::opt<bool> VPlanBuildStressTest(
    "vplan-build-stress-test", cl::init(false), cl::Hidden,
    cl::desc(
        "Build VPlan for every supported loop nest in the function and bail "
        "out right after the build (stress test the VPlan H-CFG construction "
        "in the VPlan-native vectorization path)."));

cl::opt<bool> llvm::EnableLoopInterleaving(
    "interleave-loops", cl::init(true), cl::Hidden,
    cl::desc("Enable loop interleaving in Loop vectorization passes"));
cl::opt<bool> llvm::EnableLoopVectorization(
    "vectorize-loops", cl::init(true), cl::Hidden,
    cl::desc("Run the Loop vectorization passes"));

static cl::opt<cl::boolOrDefault> ForceSafeDivisor(
    "force-widen-divrem-via-safe-divisor", cl::Hidden,
    cl::desc(
        "Override cost based safe divisor widening for div/rem instructions"));

static cl::opt<bool> EnableEarlyExitVectorization(
    "enable-early-exit-vectorization", cl::init(true), cl::Hidden,
    cl::desc(
        "Enable vectorization of early exit loops with uncountable exits."));

// Likelyhood of bypassing the vectorized loop because there are zero trips left
// after prolog. See `emitIterationCountCheck`.
static constexpr uint32_t MinItersBypassWeights[] = {1, 127};

/// A helper function that returns true if the given type is irregular. The
/// type is irregular if its allocated size doesn't equal the store size of an
/// element of the corresponding vector type.
static bool hasIrregularType(Type *Ty, const DataLayout &DL) {
  // Determine if an array of N elements of type Ty is "bitcast compatible"
  // with a <N x Ty> vector.
  // This is only true if there is no padding between the array elements.
  return DL.getTypeAllocSizeInBits(Ty) != DL.getTypeSizeInBits(Ty);
}

/// A version of ScalarEvolution::getSmallConstantTripCount that returns an
/// ElementCount to include loops whose trip count is a function of vscale.
static ElementCount getSmallConstantTripCount(ScalarEvolution *SE,
                                              const Loop *L) {
  if (unsigned ExpectedTC = SE->getSmallConstantTripCount(L))
    return ElementCount::getFixed(ExpectedTC);

  const SCEV *BTC = SE->getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BTC))
    return ElementCount::getFixed(0);

  const SCEV *ExitCount = SE->getTripCountFromExitCount(BTC, BTC->getType(), L);
  if (isa<SCEVVScale>(ExitCount))
    return ElementCount::getScalable(1);

  const APInt *Scale;
  if (match(ExitCount, m_scev_Mul(m_scev_APInt(Scale), m_SCEVVScale())))
    if (cast<SCEVMulExpr>(ExitCount)->hasNoUnsignedWrap())
      if (Scale->getActiveBits() <= 32)
        return ElementCount::getScalable(Scale->getZExtValue());

  return ElementCount::getFixed(0);
}

/// Get the maximum trip count for \p L from the SCEV unsigned range, excluding
/// zero from the range. Only valid when not folding the tail, as the minimum
/// iteration count check guards against a zero trip count. Returns 0 if
/// unknown.
static unsigned getMaxTCFromNonZeroRange(PredicatedScalarEvolution &PSE,
                                         Loop *L) {
  const SCEV *BTC = PSE.getBackedgeTakenCount();
  if (isa<SCEVCouldNotCompute>(BTC))
    return 0;
  ScalarEvolution *SE = PSE.getSE();
  const SCEV *TripCount = SE->getTripCountFromExitCount(BTC, BTC->getType(), L);
  ConstantRange TCRange = SE->getUnsignedRange(TripCount);
  APInt MaxTCFromRange = TCRange.getUnsignedMax();
  if (!MaxTCFromRange.isZero() && MaxTCFromRange.getActiveBits() <= 32)
    return MaxTCFromRange.getZExtValue();
  return 0;
}

/// Returns "best known" trip count, which is either a valid positive trip count
/// or std::nullopt when an estimate cannot be made (including when the trip
/// count would overflow), for the specified loop \p L as defined by the
/// following procedure:
///   1) Returns exact trip count if it is known.
///   2) Returns expected trip count according to profile data if any.
///   3) Returns upper bound estimate if known, and if \p CanUseConstantMax.
///   4) Returns the maximum trip count from the SCEV range excluding zero,
///      if \p CanUseConstantMax and \p CanExcludeZeroTrips.
///   5) Returns std::nullopt if all of the above failed.
static std::optional<ElementCount>
getSmallBestKnownTC(PredicatedScalarEvolution &PSE, Loop *L,
                    bool CanUseConstantMax = true,
                    bool CanExcludeZeroTrips = false) {
  // Check if exact trip count is known.
  if (auto ExpectedTC = getSmallConstantTripCount(PSE.getSE(), L))
    return ExpectedTC;

  // Check if there is an expected trip count available from profile data.
  if (LoopVectorizeWithBlockFrequency)
    if (auto EstimatedTC = getLoopEstimatedTripCount(L))
      return ElementCount::getFixed(*EstimatedTC);

  if (!CanUseConstantMax)
    return std::nullopt;

  // Check if upper bound estimate is known.
  if (unsigned ExpectedTC = PSE.getSmallConstantMaxTripCount())
    return ElementCount::getFixed(ExpectedTC);

  // Get the maximum trip count from the SCEV range excluding zero. This is
  // only safe when not folding the tail, as the minimum iteration count check
  // prevents entering the vector loop with a zero trip count.
  if (CanUseConstantMax && CanExcludeZeroTrips)
    if (unsigned RefinedTC = getMaxTCFromNonZeroRange(PSE, L))
      return ElementCount::getFixed(RefinedTC);

  return std::nullopt;
}

namespace {
// Forward declare GeneratedRTChecks.
class GeneratedRTChecks;

using SCEV2ValueTy = DenseMap<const SCEV *, Value *>;
} // namespace

namespace llvm {

AnalysisKey ShouldRunExtraVectorPasses::Key;

/// InnerLoopVectorizer vectorizes loops which contain only one basic
/// block to a specified vectorization factor (VF).
/// This class performs the widening of scalars into vectors, or multiple
/// scalars. This class also implements the following features:
/// * It inserts an epilogue loop for handling loops that don't have iteration
///   counts that are known to be a multiple of the vectorization factor.
/// * It handles the code generation for reduction variables.
/// * Scalarization (implementation using scalars) of un-vectorizable
///   instructions.
/// InnerLoopVectorizer does not perform any vectorization-legality
/// checks, and relies on the caller to check for the different legality
/// aspects. The InnerLoopVectorizer relies on the
/// LoopVectorizationLegality class to provide information about the induction
/// and reduction variables that were found to a given vectorization factor.
class InnerLoopVectorizer {
public:
  InnerLoopVectorizer(Loop *OrigLoop, PredicatedScalarEvolution &PSE,
                      LoopInfo *LI, DominatorTree *DT,
                      const TargetTransformInfo *TTI, AssumptionCache *AC,
                      ElementCount VecWidth, unsigned UnrollFactor,
                      LoopVectorizationCostModel *CM,
                      GeneratedRTChecks &RTChecks, VPlan &Plan)
      : OrigLoop(OrigLoop), PSE(PSE), LI(LI), DT(DT), TTI(TTI), AC(AC),
        VF(VecWidth), UF(UnrollFactor), Builder(PSE.getSE()->getContext()),
        Cost(CM), RTChecks(RTChecks), Plan(Plan),
        VectorPHVPBB(cast<VPBasicBlock>(
            Plan.getVectorLoopRegion()->getSinglePredecessor())) {}

  virtual ~InnerLoopVectorizer() = default;

  /// Creates a basic block for the scalar preheader. Both
  /// EpilogueVectorizerMainLoop and EpilogueVectorizerEpilogueLoop overwrite
  /// the method to create additional blocks and checks needed for epilogue
  /// vectorization.
  virtual BasicBlock *createVectorizedLoopSkeleton();

  /// Fix the vectorized code, taking care of header phi's, and more.
  void fixVectorizedLoop(VPTransformState &State);

  /// Fix the non-induction PHIs in \p Plan.
  void fixNonInductionPHIs(VPTransformState &State);

protected:
  friend class LoopVectorizationPlanner;

  /// Create and return a new IR basic block for the scalar preheader whose name
  /// is prefixed with \p Prefix.
  BasicBlock *createScalarPreheader(StringRef Prefix);

  /// Allow subclasses to override and print debug traces before/after vplan
  /// execution, when trace information is requested.
  virtual void printDebugTracesAtStart() {}
  virtual void printDebugTracesAtEnd() {}

  /// The original loop.
  Loop *OrigLoop;

  /// A wrapper around ScalarEvolution used to add runtime SCEV checks. Applies
  /// dynamic knowledge to simplify SCEV expressions and converts them to a
  /// more usable form.
  PredicatedScalarEvolution &PSE;

  /// Loop Info.
  LoopInfo *LI;

  /// Dominator Tree.
  DominatorTree *DT;

  /// Target Transform Info.
  const TargetTransformInfo *TTI;

  /// Assumption Cache.
  AssumptionCache *AC;

  /// The vectorization SIMD factor to use. Each vector will have this many
  /// vector elements.
  ElementCount VF;

  /// The vectorization unroll factor to use. Each scalar is vectorized to this
  /// many different vector instructions.
  unsigned UF;

  /// The builder that we use
  IRBuilder<> Builder;

  // --- Vectorization state ---

  /// The profitablity analysis.
  LoopVectorizationCostModel *Cost;

  /// Structure to hold information about generated runtime checks, responsible
  /// for cleaning the checks, if vectorization turns out unprofitable.
  GeneratedRTChecks &RTChecks;

  VPlan &Plan;

  /// The vector preheader block of \p Plan, used as target for check blocks
  /// introduced during skeleton creation.
  VPBasicBlock *VectorPHVPBB;
};

/// Encapsulate information regarding vectorization of a loop and its epilogue.
/// This information is meant to be updated and used across two stages of
/// epilogue vectorization.
struct EpilogueLoopVectorizationInfo {
  ElementCount MainLoopVF = ElementCount::getFixed(0);
  unsigned MainLoopUF = 0;
  ElementCount EpilogueVF = ElementCount::getFixed(0);
  unsigned EpilogueUF = 0;
  BasicBlock *MainLoopIterationCountCheck = nullptr;
  BasicBlock *EpilogueIterationCountCheck = nullptr;
  Value *VectorTripCount = nullptr;
  VPlan &EpiloguePlan;

  EpilogueLoopVectorizationInfo(ElementCount MVF, unsigned MUF,
                                ElementCount EVF, unsigned EUF,
                                VPlan &EpiloguePlan)
      : MainLoopVF(MVF), MainLoopUF(MUF), EpilogueVF(EVF), EpilogueUF(EUF),
        EpiloguePlan(EpiloguePlan) {
    assert(EUF == 1 &&
           "A high UF for the epilogue loop is likely not beneficial.");
  }
};

/// An extension of the inner loop vectorizer that creates a skeleton for a
/// vectorized loop that has its epilogue (residual) also vectorized.
/// The idea is to run the vplan on a given loop twice, firstly to setup the
/// skeleton and vectorize the main loop, and secondly to complete the skeleton
/// from the first step and vectorize the epilogue.  This is achieved by
/// deriving two concrete strategy classes from this base class and invoking
/// them in succession from the loop vectorizer planner.
class InnerLoopAndEpilogueVectorizer : public InnerLoopVectorizer {
public:
  InnerLoopAndEpilogueVectorizer(
      Loop *OrigLoop, PredicatedScalarEvolution &PSE, LoopInfo *LI,
      DominatorTree *DT, const TargetTransformInfo *TTI, AssumptionCache *AC,
      EpilogueLoopVectorizationInfo &EPI, LoopVectorizationCostModel *CM,
      GeneratedRTChecks &Checks, VPlan &Plan, ElementCount VecWidth,
      ElementCount MinProfitableTripCount, unsigned UnrollFactor)
      : InnerLoopVectorizer(OrigLoop, PSE, LI, DT, TTI, AC, VecWidth,
                            UnrollFactor, CM, Checks, Plan),
        EPI(EPI), MinProfitableTripCount(MinProfitableTripCount) {}

  /// Holds and updates state information required to vectorize the main loop
  /// and its epilogue in two separate passes. This setup helps us avoid
  /// regenerating and recomputing runtime safety checks. It also helps us to
  /// shorten the iteration-count-check path length for the cases where the
  /// iteration count of the loop is so small that the main vector loop is
  /// completely skipped.
  EpilogueLoopVectorizationInfo &EPI;

protected:
  ElementCount MinProfitableTripCount;
};

/// A specialized derived class of inner loop vectorizer that performs
/// vectorization of *main* loops in the process of vectorizing loops and their
/// epilogues.
class EpilogueVectorizerMainLoop : public InnerLoopAndEpilogueVectorizer {
public:
  EpilogueVectorizerMainLoop(Loop *OrigLoop, PredicatedScalarEvolution &PSE,
                             LoopInfo *LI, DominatorTree *DT,
                             const TargetTransformInfo *TTI,
                             AssumptionCache *AC,
                             EpilogueLoopVectorizationInfo &EPI,
                             LoopVectorizationCostModel *CM,
                             GeneratedRTChecks &Check, VPlan &Plan)
      : InnerLoopAndEpilogueVectorizer(OrigLoop, PSE, LI, DT, TTI, AC, EPI, CM,
                                       Check, Plan, EPI.MainLoopVF,
                                       EPI.MainLoopVF, EPI.MainLoopUF) {}

protected:
  void printDebugTracesAtStart() override;
  void printDebugTracesAtEnd() override;
};

// A specialized derived class of inner loop vectorizer that performs
// vectorization of *epilogue* loops in the process of vectorizing loops and
// their epilogues.
class EpilogueVectorizerEpilogueLoop : public InnerLoopAndEpilogueVectorizer {
public:
  EpilogueVectorizerEpilogueLoop(Loop *OrigLoop, PredicatedScalarEvolution &PSE,
                                 LoopInfo *LI, DominatorTree *DT,
                                 const TargetTransformInfo *TTI,
                                 AssumptionCache *AC,
                                 EpilogueLoopVectorizationInfo &EPI,
                                 LoopVectorizationCostModel *CM,
                                 GeneratedRTChecks &Checks, VPlan &Plan)
      : InnerLoopAndEpilogueVectorizer(OrigLoop, PSE, LI, DT, TTI, AC, EPI, CM,
                                       Checks, Plan, EPI.EpilogueVF,
                                       EPI.EpilogueVF, EPI.EpilogueUF) {}
  /// Implements the interface for creating a vectorized skeleton using the
  /// *epilogue loop* strategy (i.e., the second pass of VPlan execution).
  BasicBlock *createVectorizedLoopSkeleton() final;

protected:
  void printDebugTracesAtStart() override;
  void printDebugTracesAtEnd() override;
};
} // end namespace llvm

/// Look for a meaningful debug location on the instruction or its operands.
static DebugLoc getDebugLocFromInstOrOperands(Instruction *I) {
  if (!I)
    return DebugLoc::getUnknown();

  DebugLoc Empty;
  if (I->getDebugLoc() != Empty)
    return I->getDebugLoc();

  for (Use &Op : I->operands()) {
    if (Instruction *OpInst = dyn_cast<Instruction>(Op))
      if (OpInst->getDebugLoc() != Empty)
        return OpInst->getDebugLoc();
  }

  return I->getDebugLoc();
}

/// Write a \p DebugMsg about vectorization to the debug output stream. If \p I
/// is passed, the message relates to that particular instruction.
#ifndef NDEBUG
static void debugVectorizationMessage(const StringRef Prefix,
                                      const StringRef DebugMsg,
                                      Instruction *I) {
  dbgs() << "LV: " << Prefix << DebugMsg;
  if (I != nullptr)
    dbgs() << " " << *I;
  else
    dbgs() << '.';
  dbgs() << '\n';
}
#endif

/// Create an analysis remark that explains why vectorization failed
///
/// \p PassName is the name of the pass (e.g. can be AlwaysPrint).  \p
/// RemarkName is the identifier for the remark.  If \p I is passed it is an
/// instruction that prevents vectorization.  Otherwise \p TheLoop is used for
/// the location of the remark. If \p DL is passed, use it as debug location for
/// the remark. \return the remark object that can be streamed to.
static OptimizationRemarkAnalysis
createLVAnalysis(const char *PassName, StringRef RemarkName,
                 const Loop *TheLoop, Instruction *I, DebugLoc DL = {}) {
  BasicBlock *CodeRegion = I ? I->getParent() : TheLoop->getHeader();
  // If debug location is attached to the instruction, use it. Otherwise if DL
  // was not provided, use the loop's.
  if (I && I->getDebugLoc())
    DL = I->getDebugLoc();
  else if (!DL)
    DL = TheLoop->getStartLoc();

  return OptimizationRemarkAnalysis(PassName, RemarkName, DL, CodeRegion);
}

namespace llvm {

/// Return the runtime value for VF.
Value *getRuntimeVF(IRBuilderBase &B, Type *Ty, ElementCount VF) {
  return B.CreateElementCount(Ty, VF);
}

void reportVectorizationFailure(const StringRef DebugMsg,
                                const StringRef OREMsg, const StringRef ORETag,
                                OptimizationRemarkEmitter *ORE, Loop *TheLoop,
                                Instruction *I) {
  LLVM_DEBUG(debugVectorizationMessage("Not vectorizing: ", DebugMsg, I));
  LoopVectorizeHints Hints(TheLoop, false /* doesn't matter */, *ORE);
  ORE->emit(
      createLVAnalysis(Hints.vectorizeAnalysisPassName(), ORETag, TheLoop, I)
      << "loop not vectorized: " << OREMsg);
}

void reportVectorizationInfo(const StringRef Msg, const StringRef ORETag,
                             OptimizationRemarkEmitter *ORE,
                             const Loop *TheLoop, Instruction *I, DebugLoc DL) {
  LLVM_DEBUG(debugVectorizationMessage("", Msg, I));
  LoopVectorizeHints Hints(TheLoop, false /* doesn't matter */, *ORE);
  ORE->emit(createLVAnalysis(Hints.vectorizeAnalysisPassName(), ORETag, TheLoop,
                             I, DL)
            << Msg);
}

/// Report successful vectorization of the loop. In case an outer loop is
/// vectorized, prepend "outer" to the vectorization remark.
static void reportVectorization(OptimizationRemarkEmitter *ORE, Loop *TheLoop,
                                VectorizationFactor VF, unsigned IC) {
  LLVM_DEBUG(debugVectorizationMessage(
      "Vectorizing: ", TheLoop->isInnermost() ? "innermost loop" : "outer loop",
      nullptr));
  StringRef LoopType = TheLoop->isInnermost() ? "" : "outer ";
  ORE->emit([&]() {
    return OptimizationRemark(LV_NAME, "Vectorized", TheLoop->getStartLoc(),
                              TheLoop->getHeader())
           << "vectorized " << LoopType << "loop (vectorization width: "
           << ore::NV("VectorizationFactor", VF.Width)
           << ", interleaved count: " << ore::NV("InterleaveCount", IC) << ")";
  });
}

} // end namespace llvm

namespace llvm {

// Loop vectorization cost-model hints how the epilogue/tail loop should be
// lowered.
enum EpilogueLowering {

  // The default: allowing epilogues.
  CM_EpilogueAllowed,

  // Vectorization with OptForSize: don't allow epilogues.
  CM_EpilogueNotAllowedOptSize,

  // A special case of vectorisation with OptForSize: loops with a very small
  // trip count are considered for vectorization under OptForSize, thereby
  // making sure the cost of their loop body is dominant, free of runtime
  // guards and scalar iteration overheads.
  CM_EpilogueNotAllowedLowTripLoop,

  // Loop hint indicating an epilogue is undesired, apply tail folding.
  CM_EpilogueNotNeededFoldTail,

  // Directive indicating we must either fold the epilogue/tail or not vectorize
  CM_EpilogueNotAllowedFoldTail
};

/// LoopVectorizationCostModel - estimates the expected speedups due to
/// vectorization.
/// In many cases vectorization is not profitable. This can happen because of
/// a number of reasons. In this class we mainly attempt to predict the
/// expected speedup/slowdowns due to the supported instruction set. We use the
/// TargetTransformInfo to query the different backends for the cost of
/// different operations.
class LoopVectorizationCostModel {
  friend class LoopVectorizationPlanner;

public:
  LoopVectorizationCostModel(EpilogueLowering SEL, Loop *L,
                             PredicatedScalarEvolution &PSE, LoopInfo *LI,
                             LoopVectorizationLegality *Legal,
                             const TargetTransformInfo &TTI,
                             const TargetLibraryInfo *TLI, AssumptionCache *AC,
                             OptimizationRemarkEmitter *ORE,
                             std::function<BlockFrequencyInfo &()> GetBFI,
                             const Function *F, const LoopVectorizeHints *Hints,
                             InterleavedAccessInfo &IAI,
                             VFSelectionContext &Config)
      : Config(Config), EpilogueLoweringStatus(SEL), TheLoop(L), PSE(PSE),
        LI(LI), Legal(Legal), TTI(TTI), TLI(TLI), AC(AC), ORE(ORE),
        GetBFI(GetBFI), TheFunction(F), Hints(Hints), InterleaveInfo(IAI) {}

  /// \return An upper bound for the vectorization factors (both fixed and
  /// scalable). If the factors are 0, vectorization and interleaving should be
  /// avoided up front.
  FixedScalableVFPair computeMaxVF(ElementCount UserVF, unsigned UserIC);

  /// Memory access instruction may be vectorized in more than one way.
  /// Form of instruction after vectorization depends on cost.
  /// This function takes cost-based decisions for Load/Store instructions
  /// and collects them in a map. This decisions map is used for building
  /// the lists of loop-uniform and loop-scalar instructions.
  /// The calculated cost is saved with widening decision in order to
  /// avoid redundant calculations.
  void setCostBasedWideningDecision(ElementCount VF);

  /// A call may be vectorized in different ways depending on whether we have
  /// vectorized variants available and whether the target supports masking.
  /// This function analyzes all calls in the function at the supplied VF,
  /// makes a decision based on the costs of available options, and stores that
  /// decision in a map for use in planning and plan execution.
  void setVectorizedCallDecision(ElementCount VF);

  /// Collect values we want to ignore in the cost model.
  void collectValuesToIgnore();

  /// \returns True if it is more profitable to scalarize instruction \p I for
  /// vectorization factor \p VF.
  bool isProfitableToScalarize(Instruction *I, ElementCount VF) const {
    assert(VF.isVector() &&
           "Profitable to scalarize relevant only for VF > 1.");
    assert(
        TheLoop->isInnermost() &&
        "cost-model should not be used for outer loops (in VPlan-native path)");

    auto Scalars = InstsToScalarize.find(VF);
    assert(Scalars != InstsToScalarize.end() &&
           "VF not yet analyzed for scalarization profitability");
    return Scalars->second.contains(I);
  }

  /// Returns true if \p I is known to be uniform after vectorization.
  bool isUniformAfterVectorization(Instruction *I, ElementCount VF) const {
    assert(
        TheLoop->isInnermost() &&
        "cost-model should not be used for outer loops (in VPlan-native path)");

    // If VF is scalar, then all instructions are trivially uniform.
    if (VF.isScalar())
      return true;

    // Pseudo probes must be duplicated per vector lane so that the
    // profiled loop trip count is not undercounted.
    if (isa<PseudoProbeInst>(I))
      return false;

    auto UniformsPerVF = Uniforms.find(VF);
    assert(UniformsPerVF != Uniforms.end() &&
           "VF not yet analyzed for uniformity");
    return UniformsPerVF->second.count(I);
  }

  /// Returns true if \p I is known to be scalar after vectorization.
  bool isScalarAfterVectorization(Instruction *I, ElementCount VF) const {
    assert(
        TheLoop->isInnermost() &&
        "cost-model should not be used for outer loops (in VPlan-native path)");
    if (VF.isScalar())
      return true;

    auto ScalarsPerVF = Scalars.find(VF);
    assert(ScalarsPerVF != Scalars.end() &&
           "Scalar values are not calculated for VF");
    return ScalarsPerVF->second.count(I);
  }

  /// \returns True if instruction \p I can be truncated to a smaller bitwidth
  /// for vectorization factor \p VF.
  bool canTruncateToMinimalBitwidth(Instruction *I, ElementCount VF) const {
    const auto &MinBWs = Config.getMinimalBitwidths();
    // Truncs must truncate at most to their destination type.
    if (isa_and_nonnull<TruncInst>(I) && MinBWs.contains(I) &&
        I->getType()->getScalarSizeInBits() < MinBWs.lookup(I))
      return false;
    return VF.isVector() && MinBWs.contains(I) &&
           !isProfitableToScalarize(I, VF) &&
           !isScalarAfterVectorization(I, VF);
  }

  /// Decision that was taken during cost calculation for memory instruction.
  enum InstWidening {
    CM_Unknown,
    CM_Widen,         // For consecutive accesses with stride +1.
    CM_Widen_Reverse, // For consecutive accesses with stride -1.
    CM_Interleave,
    CM_GatherScatter,
    CM_Scalarize,
    CM_VectorCall,
    CM_IntrinsicCall
  };

  /// Save vectorization decision \p W and \p Cost taken by the cost model for
  /// instruction \p I and vector width \p VF.
  void setWideningDecision(Instruction *I, ElementCount VF, InstWidening W,
                           InstructionCost Cost) {
    assert(VF.isVector() && "Expected VF >=2");
    WideningDecisions[{I, VF}] = {W, Cost};
  }

  /// Save vectorization decision \p W and \p Cost taken by the cost model for
  /// interleaving group \p Grp and vector width \p VF.
  void setWideningDecision(const InterleaveGroup<Instruction> *Grp,
                           ElementCount VF, InstWidening W,
                           InstructionCost Cost) {
    assert(VF.isVector() && "Expected VF >=2");
    /// Broadcast this decicion to all instructions inside the group.
    /// When interleaving, the cost will only be assigned one instruction, the
    /// insert position. For other cases, add the appropriate fraction of the
    /// total cost to each instruction. This ensures accurate costs are used,
    /// even if the insert position instruction is not used.
    InstructionCost InsertPosCost = Cost;
    InstructionCost OtherMemberCost = 0;
    if (W != CM_Interleave)
      OtherMemberCost = InsertPosCost = Cost / Grp->getNumMembers();
    ;
    for (auto *I : Grp->members()) {
      if (Grp->getInsertPos() == I)
        WideningDecisions[{I, VF}] = {W, InsertPosCost};
      else
        WideningDecisions[{I, VF}] = {W, OtherMemberCost};
    }
  }

  /// Return the cost model decision for the given instruction \p I and vector
  /// width \p VF. Return CM_Unknown if this instruction did not pass
  /// through the cost modeling.
  InstWidening getWideningDecision(Instruction *I, ElementCount VF) const {
    assert(VF.isVector() && "Expected VF to be a vector VF");
    assert(
        TheLoop->isInnermost() &&
        "cost-model should not be used for outer loops (in VPlan-native path)");

    std::pair<Instruction *, ElementCount> InstOnVF(I, VF);
    auto Itr = WideningDecisions.find(InstOnVF);
    if (Itr == WideningDecisions.end())
      return CM_Unknown;
    return Itr->second.first;
  }

  /// Return the vectorization cost for the given instruction \p I and vector
  /// width \p VF.
  InstructionCost getWideningCost(Instruction *I, ElementCount VF) {
    assert(VF.isVector() && "Expected VF >=2");
    std::pair<Instruction *, ElementCount> InstOnVF(I, VF);
    assert(WideningDecisions.contains(InstOnVF) &&
           "The cost is not calculated");
    return WideningDecisions[InstOnVF].second;
  }

  struct CallWideningDecision {
    InstWidening Kind;
    Function *Variant;
    Intrinsic::ID IID;
    std::optional<unsigned> MaskPos;
    InstructionCost Cost;
  };

  void setCallWideningDecision(CallInst *CI, ElementCount VF, InstWidening Kind,
                               Function *Variant, Intrinsic::ID IID,
                               std::optional<unsigned> MaskPos,
                               InstructionCost Cost) {
    assert(!VF.isScalar() && "Expected vector VF");
    CallWideningDecisions[{CI, VF}] = {Kind, Variant, IID, MaskPos, Cost};
  }

  CallWideningDecision getCallWideningDecision(CallInst *CI,
                                               ElementCount VF) const {
    assert(!VF.isScalar() && "Expected vector VF");
    auto I = CallWideningDecisions.find({CI, VF});
    if (I == CallWideningDecisions.end())
      return {CM_Unknown, nullptr, Intrinsic::not_intrinsic, std::nullopt, 0};
    return I->second;
  }

  /// Return True if instruction \p I is an optimizable truncate whose operand
  /// is an induction variable. Such a truncate will be removed by adding a new
  /// induction variable with the destination type.
  bool isOptimizableIVTruncate(Instruction *I, ElementCount VF) {
    // If the instruction is not a truncate, return false.
    auto *Trunc = dyn_cast<TruncInst>(I);
    if (!Trunc)
      return false;

    // Get the source and destination types of the truncate.
    Type *SrcTy = toVectorTy(Trunc->getSrcTy(), VF);
    Type *DestTy = toVectorTy(Trunc->getDestTy(), VF);

    // If the truncate is free for the given types, return false. Replacing a
    // free truncate with an induction variable would add an induction variable
    // update instruction to each iteration of the loop. We exclude from this
    // check the primary induction variable since it will need an update
    // instruction regardless.
    Value *Op = Trunc->getOperand(0);
    if (Op != Legal->getPrimaryInduction() && TTI.isTruncateFree(SrcTy, DestTy))
      return false;

    // If the truncated value is not an induction variable, return false.
    return Legal->isInductionPhi(Op);
  }

  /// Collects the instructions to scalarize for each predicated instruction in
  /// the loop.
  void collectInstsToScalarize(ElementCount VF);

  /// Collect values that will not be widened, including Uniforms, Scalars, and
  /// Instructions to Scalarize for the given \p VF.
  /// The sets depend on CM decision for Load/Store instructions
  /// that may be vectorized as interleave, gather-scatter or scalarized.
  /// Also make a decision on what to do about call instructions in the loop
  /// at that VF -- scalarize, call a known vector routine, or call a
  /// vector intrinsic.
  void collectNonVectorizedAndSetWideningDecisions(ElementCount VF) {
    // Do the analysis once.
    if (VF.isScalar() || Uniforms.contains(VF))
      return;
    setCostBasedWideningDecision(VF);
    collectLoopUniforms(VF);
    setVectorizedCallDecision(VF);
    collectLoopScalars(VF);
    collectInstsToScalarize(VF);
  }

  /// Given costs for both strategies, return true if the scalar predication
  /// lowering should be used for div/rem.  This incorporates an override
  /// option so it is not simply a cost comparison.
  bool isDivRemScalarWithPredication(InstructionCost ScalarCost,
                                     InstructionCost SafeDivisorCost) const {
    switch (ForceSafeDivisor) {
    case cl::BOU_UNSET:
      return ScalarCost < SafeDivisorCost;
    case cl::BOU_TRUE:
      return false;
    case cl::BOU_FALSE:
      return true;
    }
    llvm_unreachable("impossible case value");
  }

  /// Returns true if \p I is an instruction which requires predication and
  /// for which our chosen predication strategy is scalarization (i.e. we
  /// don't have an alternate strategy such as masking available).
  /// \p VF is the vectorization factor that will be used to vectorize \p I.
  bool isScalarWithPredication(Instruction *I, ElementCount VF);

  /// Wrapper function for LoopVectorizationLegality::isMaskRequired,
  /// that passes the Instruction \p I and if we fold tail.
  bool isMaskRequired(Instruction *I) const;

  /// Returns true if \p I is an instruction that needs to be predicated
  /// at runtime.  The result is independent of the predication mechanism.
  /// Superset of instructions that return true for isScalarWithPredication.
  bool isPredicatedInst(Instruction *I) const;

  /// A helper function that returns how much we should divide the cost of a
  /// predicated block by. Typically this is the reciprocal of the block
  /// probability, i.e. if we return X we are assuming the predicated block will
  /// execute once for every X iterations of the loop header so the block should
  /// only contribute 1/X of its cost to the total cost calculation, but when
  /// optimizing for code size it will just be 1 as code size costs don't depend
  /// on execution probabilities.
  ///
  /// Note that if a block wasn't originally predicated but was predicated due
  /// to tail folding, the divisor will still be 1 because it will execute for
  /// every iteration of the loop header.
  inline uint64_t
  getPredBlockCostDivisor(TargetTransformInfo::TargetCostKind CostKind,
                          const BasicBlock *BB);

  /// Returns true if an artificially high cost for emulated masked memrefs
  /// should be used.
  bool useEmulatedMaskMemRefHack(Instruction *I, ElementCount VF);

  /// Return the costs for our two available strategies for lowering a
  /// div/rem operation which requires speculating at least one lane.
  /// First result is for scalarization (will be invalid for scalable
  /// vectors); second is for the safe-divisor strategy.
  std::pair<InstructionCost, InstructionCost>
  getDivRemSpeculationCost(Instruction *I, ElementCount VF);

  /// Returns true if \p I is a memory instruction with consecutive memory
  /// access that can be widened.
  bool memoryInstructionCanBeWidened(Instruction *I, ElementCount VF);

  /// Returns true if \p I is a memory instruction in an interleaved-group
  /// of memory accesses that can be vectorized with wide vector loads/stores
  /// and shuffles.
  bool interleavedAccessCanBeWidened(Instruction *I, ElementCount VF) const;

  /// Check if \p Instr belongs to any interleaved access group.
  bool isAccessInterleaved(Instruction *Instr) const {
    return InterleaveInfo.isInterleaved(Instr);
  }

  /// Get the interleaved access group that \p Instr belongs to.
  const InterleaveGroup<Instruction> *
  getInterleavedAccessGroup(Instruction *Instr) const {
    return InterleaveInfo.getInterleaveGroup(Instr);
  }

  /// Returns true if we're required to use a scalar epilogue for at least
  /// the final iteration of the original loop.
  bool requiresScalarEpilogue(bool IsVectorizing) const {
    if (!isEpilogueAllowed()) {
      LLVM_DEBUG(dbgs() << "LV: Loop does not require scalar epilogue\n");
      return false;
    }
    // If we might exit from anywhere but the latch and early exit vectorization
    // is disabled, we must run the exiting iteration in scalar form.
    if (TheLoop->getExitingBlock() != TheLoop->getLoopLatch() &&
        !(EnableEarlyExitVectorization && Legal->hasUncountableEarlyExit())) {
      LLVM_DEBUG(dbgs() << "LV: Loop requires scalar epilogue: not exiting "
                           "from latch block\n");
      return true;
    }
    if (IsVectorizing && InterleaveInfo.requiresScalarEpilogue()) {
      LLVM_DEBUG(dbgs() << "LV: Loop requires scalar epilogue: "
                           "interleaved group requires scalar epilogue\n");
      return true;
    }
    LLVM_DEBUG(dbgs() << "LV: Loop does not require scalar epilogue\n");
    return false;
  }

  /// Returns true if an epilogue is allowed (e.g., not prevented by
  /// optsize or a loop hint annotation).
  bool isEpilogueAllowed() const {
    return EpilogueLoweringStatus == CM_EpilogueAllowed;
  }

  /// Returns true if tail-folding is preferred over an epilogue.
  bool preferTailFoldedLoop() const {
    return EpilogueLoweringStatus == CM_EpilogueNotNeededFoldTail ||
           EpilogueLoweringStatus == CM_EpilogueNotAllowedFoldTail;
  }

  /// Returns the TailFoldingStyle that is best for the current loop.
  TailFoldingStyle getTailFoldingStyle() const {
    return ChosenTailFoldingStyle;
  }

  /// Selects and saves TailFoldingStyle.
  /// \param IsScalableVF true if scalable vector factors enabled.
  /// \param UserIC User specific interleave count.
  void setTailFoldingStyle(bool IsScalableVF, unsigned UserIC) {
    assert(ChosenTailFoldingStyle == TailFoldingStyle::None &&
           "Tail folding must not be selected yet.");
    if (!Legal->canFoldTailByMasking()) {
      ChosenTailFoldingStyle = TailFoldingStyle::None;
      return;
    }

    // Default to TTI preference, but allow command line override.
    ChosenTailFoldingStyle = TTI.getPreferredTailFoldingStyle();
    if (ForceTailFoldingStyle.getNumOccurrences())
      ChosenTailFoldingStyle = ForceTailFoldingStyle.getValue();

    if (ChosenTailFoldingStyle != TailFoldingStyle::DataWithEVL)
      return;
    // Override EVL styles if needed.
    // FIXME: Investigate opportunity for fixed vector factor.
    bool EVLIsLegal = UserIC <= 1 && IsScalableVF &&
                      TTI.hasActiveVectorLength() && !EnableVPlanNativePath;
    if (EVLIsLegal)
      return;
    // If for some reason EVL mode is unsupported, fallback to an epilogue
    // if it's allowed, or DataWithoutLaneMask otherwise.
    if (EpilogueLoweringStatus == CM_EpilogueAllowed ||
        EpilogueLoweringStatus == CM_EpilogueNotNeededFoldTail)
      ChosenTailFoldingStyle = TailFoldingStyle::None;
    else
      ChosenTailFoldingStyle = TailFoldingStyle::DataWithoutLaneMask;

    LLVM_DEBUG(
        dbgs() << "LV: Preference for VP intrinsics indicated. Will "
                  "not try to generate VP Intrinsics "
               << (UserIC > 1
                       ? "since interleave count specified is greater than 1.\n"
                       : "due to non-interleaving reasons.\n"));
  }

  /// Returns true if all loop blocks should be masked to fold tail loop.
  bool foldTailByMasking() const {
    return getTailFoldingStyle() != TailFoldingStyle::None;
  }

  /// Returns true if the use of wide lane masks is requested and the loop is
  /// using tail-folding with a lane mask for control flow.
  bool useWideActiveLaneMask() const {
    if (!EnableWideActiveLaneMask)
      return false;

    return getTailFoldingStyle() == TailFoldingStyle::DataAndControlFlow;
  }

  /// Returns true if the instructions in this block requires predication
  /// for any reason, e.g. because tail folding now requires a predicate
  /// or because the block in the original loop was predicated.
  bool blockNeedsPredicationForAnyReason(BasicBlock *BB) const {
    return foldTailByMasking() || Legal->blockNeedsPredication(BB);
  }

  /// Returns true if VP intrinsics with explicit vector length support should
  /// be generated in the tail folded loop.
  bool foldTailWithEVL() const {
    return getTailFoldingStyle() == TailFoldingStyle::DataWithEVL;
  }

  /// Returns true if the predicated reduction select should be used to set the
  /// incoming value for the reduction phi.
  bool usePredicatedReductionSelect(RecurKind RecurrenceKind) const {
    // Force to use predicated reduction select since the EVL of the
    // second-to-last iteration might not be VF*UF.
    if (foldTailWithEVL())
      return true;

    // Note: For FindLast recurrences we prefer a predicated select to simplify
    // matching in handleFindLastReductions(), rather than handle multiple
    // cases.
    if (RecurrenceDescriptor::isFindLastRecurrenceKind(RecurrenceKind))
      return true;

    return PreferPredicatedReductionSelect ||
           TTI.preferPredicatedReductionSelect();
  }

  /// Estimate cost of an intrinsic call instruction CI if it were vectorized
  /// with factor VF.  Return the cost of the instruction, including
  /// scalarization overhead if it's needed.
  InstructionCost getVectorIntrinsicCost(CallInst *CI, ElementCount VF) const;

  /// Estimate cost of a call instruction CI if it were vectorized with factor
  /// VF. Return the cost of the instruction, including scalarization overhead
  /// if it's needed.
  InstructionCost getVectorCallCost(CallInst *CI, ElementCount VF) const;

  /// Invalidates decisions already taken by the cost model.
  void invalidateCostModelingDecisions() {
    WideningDecisions.clear();
    CallWideningDecisions.clear();
    Uniforms.clear();
    Scalars.clear();
  }

  /// Returns the expected execution cost. The unit of the cost does
  /// not matter because we use the 'cost' units to compare different
  /// vector widths. The cost that is returned is *not* normalized by
  /// the factor width.
  InstructionCost expectedCost(ElementCount VF);

  /// Returns true if epilogue vectorization is considered profitable, and
  /// false otherwise.
  /// \p VF is the vectorization factor chosen for the original loop.
  /// \p Multiplier is an aditional scaling factor applied to VF before
  /// comparing to EpilogueVectorizationMinVF.
  bool isEpilogueVectorizationProfitable(const ElementCount VF,
                                         const unsigned IC) const;

  /// Returns the execution time cost of an instruction for a given vector
  /// width. Vector width of one means scalar.
  InstructionCost getInstructionCost(Instruction *I, ElementCount VF);

  /// Return the cost of instructions in an inloop reduction pattern, if I is
  /// part of that pattern.
  std::optional<InstructionCost> getReductionPatternCost(Instruction *I,
                                                         ElementCount VF,
                                                         Type *VectorTy) const;

  /// Returns true if \p Op should be considered invariant and if it is
  /// trivially hoistable.
  bool shouldConsiderInvariant(Value *Op);

private:
  unsigned NumPredStores = 0;

  /// VF selection state independent of cost-modeling decisions.
  VFSelectionContext &Config;

  /// Calculate vectorization cost of memory instruction \p I.
  InstructionCost getMemoryInstructionCost(Instruction *I, ElementCount VF);

  /// The cost computation for scalarized memory instruction.
  InstructionCost getMemInstScalarizationCost(Instruction *I, ElementCount VF);

  /// The cost computation for interleaving group of memory instructions.
  InstructionCost getInterleaveGroupCost(Instruction *I, ElementCount VF);

  /// The cost computation for Gather/Scatter instruction.
  InstructionCost getGatherScatterCost(Instruction *I, ElementCount VF);

  /// The cost computation for widening instruction \p I with consecutive
  /// memory access.
  InstructionCost getConsecutiveMemOpCost(Instruction *I, ElementCount VF);

  /// The cost calculation for Load/Store instruction \p I with uniform pointer -
  /// Load: scalar load + broadcast.
  /// Store: scalar store + (loop invariant value stored? 0 : extract of last
  /// element)
  InstructionCost getUniformMemOpCost(Instruction *I, ElementCount VF);

  /// Estimate the overhead of scalarizing an instruction. This is a
  /// convenience wrapper for the type-based getScalarizationOverhead API.
  InstructionCost getScalarizationOverhead(Instruction *I,
                                           ElementCount VF) const;

  /// A type representing the costs for instructions if they were to be
  /// scalarized rather than vectorized. The entries are Instruction-Cost
  /// pairs.
  using ScalarCostsTy = MapVector<Instruction *, InstructionCost>;

  /// A set containing all BasicBlocks that are known to present after
  /// vectorization as a predicated block.
  DenseMap<ElementCount, SmallPtrSet<BasicBlock *, 4>>
      PredicatedBBsAfterVectorization;

  /// Records whether it is allowed to have the original scalar loop execute at
  /// least once. This may be needed as a fallback loop in case runtime
  /// aliasing/dependence checks fail, or to handle the tail/remainder
  /// iterations when the trip count is unknown or doesn't divide by the VF,
  /// or as a peel-loop to handle gaps in interleave-groups.
  /// Under optsize and when the trip count is very small we don't allow any
  /// iterations to execute in the scalar loop.
  EpilogueLowering EpilogueLoweringStatus = CM_EpilogueAllowed;

  /// Control finally chosen tail folding style.
  TailFoldingStyle ChosenTailFoldingStyle = TailFoldingStyle::None;

  /// A map holding scalar costs for different vectorization factors. The
  /// presence of a cost for an instruction in the mapping indicates that the
  /// instruction will be scalarized when vectorizing with the associated
  /// vectorization factor. The entries are VF-ScalarCostTy pairs.
  MapVector<ElementCount, ScalarCostsTy> InstsToScalarize;

  /// Holds the instructions known to be uniform after vectorization.
  /// The data is collected per VF.
  DenseMap<ElementCount, SmallPtrSet<Instruction *, 4>> Uniforms;

  /// Holds the instructions known to be scalar after vectorization.
  /// The data is collected per VF.
  DenseMap<ElementCount, SmallPtrSet<Instruction *, 4>> Scalars;

  /// Holds the instructions (address computations) that are forced to be
  /// scalarized.
  DenseMap<ElementCount, SmallPtrSet<Instruction *, 4>> ForcedScalars;

  /// Returns the expected difference in cost from scalarizing the expression
  /// feeding a predicated instruction \p PredInst. The instructions to
  /// scalarize and their scalar costs are collected in \p ScalarCosts. A
  /// non-negative return value implies the expression will be scalarized.
  /// Currently, only single-use chains are considered for scalarization.
  InstructionCost computePredInstDiscount(Instruction *PredInst,
                                          ScalarCostsTy &ScalarCosts,
                                          ElementCount VF);

  /// Collect the instructions that are uniform after vectorization. An
  /// instruction is uniform if we represent it with a single scalar value in
  /// the vectorized loop corresponding to each vector iteration. Examples of
  /// uniform instructions include pointer operands of consecutive or
  /// interleaved memory accesses. Note that although uniformity implies an
  /// instruction will be scalar, the reverse is not true. In general, a
  /// scalarized instruction will be represented by VF scalar values in the
  /// vectorized loop, each corresponding to an iteration of the original
  /// scalar loop.
  void collectLoopUniforms(ElementCount VF);

  /// Collect the instructions that are scalar after vectorization. An
  /// instruction is scalar if it is known to be uniform or will be scalarized
  /// during vectorization. collectLoopScalars should only add non-uniform nodes
  /// to the list if they are used by a load/store instruction that is marked as
  /// CM_Scalarize. Non-uniform scalarized instructions will be represented by
  /// VF values in the vectorized loop, each corresponding to an iteration of
  /// the original scalar loop.
  void collectLoopScalars(ElementCount VF);

  /// Keeps cost model vectorization decision and cost for instructions.
  /// Right now it is used for memory instructions only.
  using DecisionList = DenseMap<std::pair<Instruction *, ElementCount>,
                                std::pair<InstWidening, InstructionCost>>;

  DecisionList WideningDecisions;

  using CallDecisionList =
      DenseMap<std::pair<CallInst *, ElementCount>, CallWideningDecision>;

  CallDecisionList CallWideningDecisions;

  /// Returns true if \p V is expected to be vectorized and it needs to be
  /// extracted.
  bool needsExtract(Value *V, ElementCount VF) const {
    Instruction *I = dyn_cast<Instruction>(V);
    if (VF.isScalar() || !I || !TheLoop->contains(I) ||
        TheLoop->isLoopInvariant(I) ||
        getWideningDecision(I, VF) == CM_Scalarize ||
        (isa<CallInst>(I) &&
         getCallWideningDecision(cast<CallInst>(I), VF).Kind == CM_Scalarize))
      return false;

    // Assume we can vectorize V (and hence we need extraction) if the
    // scalars are not computed yet. This can happen, because it is called
    // via getScalarizationOverhead from setCostBasedWideningDecision, before
    // the scalars are collected. That should be a safe assumption in most
    // cases, because we check if the operands have vectorizable types
    // beforehand in LoopVectorizationLegality.
    return !Scalars.contains(VF) || !isScalarAfterVectorization(I, VF);
  };

  /// Returns a range containing only operands needing to be extracted.
  SmallVector<Value *, 4> filterExtractingOperands(Instruction::op_range Ops,
                                                   ElementCount VF) const {

    SmallPtrSet<const Value *, 4> UniqueOperands;
    SmallVector<Value *, 4> Res;
    for (Value *Op : Ops) {
      if (isa<Constant>(Op) || !UniqueOperands.insert(Op).second ||
          !needsExtract(Op, VF))
        continue;
      Res.push_back(Op);
    }
    return Res;
  }

public:
  /// The loop that we evaluate.
  Loop *TheLoop;

  /// Predicated scalar evolution analysis.
  PredicatedScalarEvolution &PSE;

  /// Loop Info analysis.
  LoopInfo *LI;

  /// Vectorization legality.
  LoopVectorizationLegality *Legal;

  /// Vector target information.
  const TargetTransformInfo &TTI;

  /// Target Library Info.
  const TargetLibraryInfo *TLI;

  /// Assumption cache.
  AssumptionCache *AC;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;

  /// A function to lazily fetch BlockFrequencyInfo. This avoids computing it
  /// unless necessary, e.g. when the loop isn't legal to vectorize or when
  /// there is no predication.
  std::function<BlockFrequencyInfo &()> GetBFI;
  /// The BlockFrequencyInfo returned from GetBFI.
  BlockFrequencyInfo *BFI = nullptr;
  /// Returns the BlockFrequencyInfo for the function if cached, otherwise
  /// fetches it via GetBFI. Avoids an indirect call to the std::function.
  BlockFrequencyInfo &getBFI() {
    if (!BFI)
      BFI = &GetBFI();
    return *BFI;
  }

  const Function *TheFunction;

  /// Loop Vectorize Hint.
  const LoopVectorizeHints *Hints;

  /// The interleave access information contains groups of interleaved accesses
  /// with the same stride and close to each other.
  InterleavedAccessInfo &InterleaveInfo;

  /// Values to ignore in the cost model.
  SmallPtrSet<const Value *, 16> ValuesToIgnore;

  /// Values to ignore in the cost model when VF > 1.
  SmallPtrSet<const Value *, 16> VecValuesToIgnore;
};
} // end namespace llvm

namespace {
/// Helper struct to manage generating runtime checks for vectorization.
///
/// The runtime checks are created up-front in temporary blocks to allow better
/// estimating the cost and un-linked from the existing IR. After deciding to
/// vectorize, the checks are moved back. If deciding not to vectorize, the
/// temporary blocks are completely removed.
class GeneratedRTChecks {
  /// Basic block which contains the generated SCEV checks, if any.
  BasicBlock *SCEVCheckBlock = nullptr;

  /// The value representing the result of the generated SCEV checks. If it is
  /// nullptr no SCEV checks have been generated.
  Value *SCEVCheckCond = nullptr;

  /// Basic block which contains the generated memory runtime checks, if any.
  BasicBlock *MemCheckBlock = nullptr;

  /// The value representing the result of the generated memory runtime checks.
  /// If it is nullptr no memory runtime checks have been generated.
  Value *MemRuntimeCheckCond = nullptr;

  DominatorTree *DT;
  LoopInfo *LI;
  TargetTransformInfo *TTI;

  SCEVExpander SCEVExp;
  SCEVExpander MemCheckExp;

  bool CostTooHigh = false;

  Loop *OuterLoop = nullptr;

  PredicatedScalarEvolution &PSE;

  /// The kind of cost that we are calculating
  TTI::TargetCostKind CostKind;

public:
  GeneratedRTChecks(PredicatedScalarEvolution &PSE, DominatorTree *DT,
                    LoopInfo *LI, TargetTransformInfo *TTI,
                    TTI::TargetCostKind CostKind)
      : DT(DT), LI(LI), TTI(TTI),
        SCEVExp(*PSE.getSE(), "scev.check", /*PreserveLCSSA=*/false),
        MemCheckExp(*PSE.getSE(), "scev.check", /*PreserveLCSSA=*/false),
        PSE(PSE), CostKind(CostKind) {}

  /// Generate runtime checks in SCEVCheckBlock and MemCheckBlock, so we can
  /// accurately estimate the cost of the runtime checks. The blocks are
  /// un-linked from the IR and are added back during vector code generation. If
  /// there is no vector code generation, the check blocks are removed
  /// completely.
  void create(Loop *L, const LoopAccessInfo &LAI,
              const SCEVPredicate &UnionPred, ElementCount VF, unsigned IC,
              OptimizationRemarkEmitter &ORE) {

    // Hard cutoff to limit compile-time increase in case a very large number of
    // runtime checks needs to be generated.
    // TODO: Skip cutoff if the loop is guaranteed to execute, e.g. due to
    // profile info.
    CostTooHigh =
        LAI.getNumRuntimePointerChecks() > VectorizeMemoryCheckThreshold;
    if (CostTooHigh) {
      // Mark runtime checks as never succeeding when they exceed the threshold.
      MemRuntimeCheckCond = ConstantInt::getTrue(L->getHeader()->getContext());
      SCEVCheckCond = ConstantInt::getTrue(L->getHeader()->getContext());
      ORE.emit([&]() {
        return OptimizationRemarkAnalysisAliasing(
                   DEBUG_TYPE, "TooManyMemoryRuntimeChecks", L->getStartLoc(),
                   L->getHeader())
               << "loop not vectorized: too many memory checks needed";
      });
      LLVM_DEBUG(dbgs() << "LV: Too many memory checks needed.\n");
      return;
    }

    BasicBlock *LoopHeader = L->getHeader();
    BasicBlock *Preheader = L->getLoopPreheader();

    // Use SplitBlock to create blocks for SCEV & memory runtime checks to
    // ensure the blocks are properly added to LoopInfo & DominatorTree. Those
    // may be used by SCEVExpander. The blocks will be un-linked from their
    // predecessors and removed from LI & DT at the end of the function.
    if (!UnionPred.isAlwaysTrue()) {
      SCEVCheckBlock = SplitBlock(Preheader, Preheader->getTerminator(), DT, LI,
                                  nullptr, "vector.scevcheck");

      SCEVCheckCond = SCEVExp.expandCodeForPredicate(
          &UnionPred, SCEVCheckBlock->getTerminator());
      if (isa<Constant>(SCEVCheckCond)) {
        // Clean up directly after expanding the predicate to a constant, to
        // avoid further expansions re-using anything left over from SCEVExp.
        SCEVExpanderCleaner SCEVCleaner(SCEVExp);
        SCEVCleaner.cleanup();
      }
    }

    const auto &RtPtrChecking = *LAI.getRuntimePointerChecking();
    if (RtPtrChecking.Need) {
      auto *Pred = SCEVCheckBlock ? SCEVCheckBlock : Preheader;
      MemCheckBlock = SplitBlock(Pred, Pred->getTerminator(), DT, LI, nullptr,
                                 "vector.memcheck");

      auto DiffChecks = RtPtrChecking.getDiffChecks();
      if (DiffChecks) {
        Value *RuntimeVF = nullptr;
        MemRuntimeCheckCond = addDiffRuntimeChecks(
            MemCheckBlock->getTerminator(), *DiffChecks, MemCheckExp,
            [VF, &RuntimeVF](IRBuilderBase &B, unsigned Bits) {
              if (!RuntimeVF)
                RuntimeVF = getRuntimeVF(B, B.getIntNTy(Bits), VF);
              return RuntimeVF;
            },
            IC);
      } else {
        MemRuntimeCheckCond = addRuntimeChecks(
            MemCheckBlock->getTerminator(), L, RtPtrChecking.getChecks(),
            MemCheckExp, VectorizerParams::HoistRuntimeChecks);
      }
      assert(MemRuntimeCheckCond &&
             "no RT checks generated although RtPtrChecking "
             "claimed checks are required");
    }

    SCEVExp.eraseDeadInstructions(SCEVCheckCond);

    if (!MemCheckBlock && !SCEVCheckBlock)
      return;

    // Unhook the temporary block with the checks, update various places
    // accordingly.
    if (SCEVCheckBlock)
      SCEVCheckBlock->replaceAllUsesWith(Preheader);
    if (MemCheckBlock)
      MemCheckBlock->replaceAllUsesWith(Preheader);

    if (SCEVCheckBlock) {
      SCEVCheckBlock->getTerminator()->moveBefore(
          Preheader->getTerminator()->getIterator());
      auto *UI = new UnreachableInst(Preheader->getContext(), SCEVCheckBlock);
      UI->setDebugLoc(DebugLoc::getTemporary());
      Preheader->getTerminator()->eraseFromParent();
    }
    if (MemCheckBlock) {
      MemCheckBlock->getTerminator()->moveBefore(
          Preheader->getTerminator()->getIterator());
      auto *UI = new UnreachableInst(Preheader->getContext(), MemCheckBlock);
      UI->setDebugLoc(DebugLoc::getTemporary());
      Preheader->getTerminator()->eraseFromParent();
    }

    DT->changeImmediateDominator(LoopHeader, Preheader);
    if (MemCheckBlock) {
      DT->eraseNode(MemCheckBlock);
      LI->removeBlock(MemCheckBlock);
    }
    if (SCEVCheckBlock) {
      DT->eraseNode(SCEVCheckBlock);
      LI->removeBlock(SCEVCheckBlock);
    }

    // Outer loop is used as part of the later cost calculations.
    OuterLoop = L->getParentLoop();
  }

  InstructionCost getCost() {
    if (SCEVCheckBlock || MemCheckBlock)
      LLVM_DEBUG(dbgs() << "Calculating cost of runtime checks:\n");

    if (CostTooHigh) {
      InstructionCost Cost;
      Cost.setInvalid();
      LLVM_DEBUG(dbgs() << "  number of checks exceeded threshold\n");
      return Cost;
    }

    InstructionCost RTCheckCost = 0;
    if (SCEVCheckBlock)
      for (Instruction &I : *SCEVCheckBlock) {
        if (SCEVCheckBlock->getTerminator() == &I)
          continue;
        InstructionCost C = TTI->getInstructionCost(&I, CostKind);
        LLVM_DEBUG(dbgs() << "  " << C << "  for " << I << "\n");
        RTCheckCost += C;
      }
    if (MemCheckBlock) {
      InstructionCost MemCheckCost = 0;
      for (Instruction &I : *MemCheckBlock) {
        if (MemCheckBlock->getTerminator() == &I)
          continue;
        InstructionCost C = TTI->getInstructionCost(&I, CostKind);
        LLVM_DEBUG(dbgs() << "  " << C << "  for " << I << "\n");
        MemCheckCost += C;
      }

      // If the runtime memory checks are being created inside an outer loop
      // we should find out if these checks are outer loop invariant. If so,
      // the checks will likely be hoisted out and so the effective cost will
      // reduce according to the outer loop trip count.
      if (OuterLoop) {
        ScalarEvolution *SE = MemCheckExp.getSE();
        // TODO: If profitable, we could refine this further by analysing every
        // individual memory check, since there could be a mixture of loop
        // variant and invariant checks that mean the final condition is
        // variant.
        const SCEV *Cond = SE->getSCEV(MemRuntimeCheckCond);
        if (SE->isLoopInvariant(Cond, OuterLoop)) {
          // It seems reasonable to assume that we can reduce the effective
          // cost of the checks even when we know nothing about the trip
          // count. Assume that the outer loop executes at least twice.
          unsigned BestTripCount = 2;

          // Get the best known TC estimate.
          if (auto EstimatedTC = getSmallBestKnownTC(
                  PSE, OuterLoop, /* CanUseConstantMax = */ false))
            if (EstimatedTC->isFixed())
              BestTripCount = EstimatedTC->getFixedValue();

          InstructionCost NewMemCheckCost = MemCheckCost / BestTripCount;

          // Let's ensure the cost is always at least 1.
          NewMemCheckCost = std::max(NewMemCheckCost.getValue(),
                                     (InstructionCost::CostType)1);

          if (BestTripCount > 1)
            LLVM_DEBUG(dbgs()
                       << "We expect runtime memory checks to be hoisted "
                       << "out of the outer loop. Cost reduced from "
                       << MemCheckCost << " to " << NewMemCheckCost << '\n');

          MemCheckCost = NewMemCheckCost;
        }
      }

      RTCheckCost += MemCheckCost;
    }

    if (SCEVCheckBlock || MemCheckBlock)
      LLVM_DEBUG(dbgs() << "Total cost of runtime checks: " << RTCheckCost
                        << "\n");

    return RTCheckCost;
  }

  /// Remove the created SCEV & memory runtime check blocks & instructions, if
  /// unused.
  ~GeneratedRTChecks() {
    SCEVExpanderCleaner SCEVCleaner(SCEVExp);
    SCEVExpanderCleaner MemCheckCleaner(MemCheckExp);
    bool SCEVChecksUsed = !SCEVCheckBlock || !pred_empty(SCEVCheckBlock);
    bool MemChecksUsed = !MemCheckBlock || !pred_empty(MemCheckBlock);
    if (SCEVChecksUsed)
      SCEVCleaner.markResultUsed();

    if (MemChecksUsed) {
      MemCheckCleaner.markResultUsed();
    } else {
      auto &SE = *MemCheckExp.getSE();
      // Memory runtime check generation creates compares that use expanded
      // values. Remove them before running the SCEVExpanderCleaners.
      for (auto &I : make_early_inc_range(reverse(*MemCheckBlock))) {
        if (MemCheckExp.isInsertedInstruction(&I))
          continue;
        SE.forgetValue(&I);
        I.eraseFromParent();
      }
    }
    MemCheckCleaner.cleanup();
    SCEVCleaner.cleanup();

    if (!SCEVChecksUsed)
      SCEVCheckBlock->eraseFromParent();
    if (!MemChecksUsed)
      MemCheckBlock->eraseFromParent();
  }

  /// Retrieves the SCEVCheckCond and SCEVCheckBlock that were generated as IR
  /// outside VPlan.
  std::pair<Value *, BasicBlock *> getSCEVChecks() const {
    using namespace llvm::PatternMatch;
    if (!SCEVCheckCond || match(SCEVCheckCond, m_ZeroInt()))
      return {nullptr, nullptr};

    return {SCEVCheckCond, SCEVCheckBlock};
  }

  /// Retrieves the MemCheckCond and MemCheckBlock that were generated as IR
  /// outside VPlan.
  std::pair<Value *, BasicBlock *> getMemRuntimeChecks() const {
    using namespace llvm::PatternMatch;
    if (MemRuntimeCheckCond && match(MemRuntimeCheckCond, m_ZeroInt()))
      return {nullptr, nullptr};
    return {MemRuntimeCheckCond, MemCheckBlock};
  }

  /// Return true if any runtime checks have been added
  bool hasChecks() const {
    return getSCEVChecks().first || getMemRuntimeChecks().first;
  }
};
} // namespace

static bool useActiveLaneMask(TailFoldingStyle Style) {
  return Style == TailFoldingStyle::Data ||
         Style == TailFoldingStyle::DataAndControlFlow;
}

static bool useActiveLaneMaskForControlFlow(TailFoldingStyle Style) {
  return Style == TailFoldingStyle::DataAndControlFlow;
}

// Return true if \p OuterLp is an outer loop annotated with hints for explicit
// vectorization. The loop needs to be annotated with #pragma omp simd
// simdlen(#) or #pragma clang vectorize(enable) vectorize_width(#). If the
// vector length information is not provided, vectorization is not considered
// explicit. Interleave hints are not allowed either. These limitations will be
// relaxed in the future.
// Please, note that we are currently forced to abuse the pragma 'clang
// vectorize' semantics. This pragma provides *auto-vectorization hints*
// (i.e., LV must check that vectorization is legal) whereas pragma 'omp simd'
// provides *explicit vectorization hints* (LV can bypass legal checks and
// assume that vectorization is legal). However, both hints are implemented
// using the same metadata (llvm.loop.vectorize, processed by
// LoopVectorizeHints). This will be fixed in the future when the native IR
// representation for pragma 'omp simd' is introduced.
static bool isExplicitVecOuterLoop(Loop *OuterLp,
                                   OptimizationRemarkEmitter *ORE) {
  assert(!OuterLp->isInnermost() && "This is not an outer loop");
  LoopVectorizeHints Hints(OuterLp, true /*DisableInterleaving*/, *ORE);

  // Only outer loops with an explicit vectorization hint are supported.
  // Unannotated outer loops are ignored.
  if (Hints.getForce() == LoopVectorizeHints::FK_Undefined)
    return false;

  Function *Fn = OuterLp->getHeader()->getParent();
  if (!Hints.allowVectorization(Fn, OuterLp,
                                true /*VectorizeOnlyWhenForced*/)) {
    LLVM_DEBUG(dbgs() << "LV: Loop hints prevent outer loop vectorization.\n");
    return false;
  }

  if (Hints.getInterleave() > 1) {
    // TODO: Interleave support is future work.
    LLVM_DEBUG(dbgs() << "LV: Not vectorizing: Interleave is not supported for "
                         "outer loops.\n");
    Hints.emitRemarkWithHints();
    return false;
  }

  return true;
}

static void collectSupportedLoops(Loop &L, LoopInfo *LI,
                                  OptimizationRemarkEmitter *ORE,
                                  SmallVectorImpl<Loop *> &V) {
  // Collect inner loops and outer loops without irreducible control flow. For
  // now, only collect outer loops that have explicit vectorization hints. If we
  // are stress testing the VPlan H-CFG construction, we collect the outermost
  // loop of every loop nest.
  if (L.isInnermost() || VPlanBuildStressTest ||
      (EnableVPlanNativePath && isExplicitVecOuterLoop(&L, ORE))) {
    LoopBlocksRPO RPOT(&L);
    RPOT.perform(LI);
    if (!containsIrreducibleCFG<const BasicBlock *>(RPOT, *LI)) {
      V.push_back(&L);
      // TODO: Collect inner loops inside marked outer loops in case
      // vectorization fails for the outer loop. Do not invoke
      // 'containsIrreducibleCFG' again for inner loops when the outer loop is
      // already known to be reducible. We can use an inherited attribute for
      // that.
      return;
    }
  }
  for (Loop *InnerL : L)
    collectSupportedLoops(*InnerL, LI, ORE, V);
}

//===----------------------------------------------------------------------===//
// Implementation of LoopVectorizationLegality, InnerLoopVectorizer and
// LoopVectorizationCostModel and LoopVectorizationPlanner.
//===----------------------------------------------------------------------===//

/// For the given VF and UF and maximum trip count computed for the loop, return
/// whether the induction variable might overflow in the vectorized loop. If not,
/// then we know a runtime overflow check always evaluates to false and can be
/// removed.
static bool isIndvarOverflowCheckKnownFalse(
    const LoopVectorizationCostModel *Cost,
    ElementCount VF, std::optional<unsigned> UF = std::nullopt) {
  // Always be conservative if we don't know the exact unroll factor.
  unsigned MaxUF = UF ? *UF : Cost->TTI.getMaxInterleaveFactor(VF);

  IntegerType *IdxTy = Cost->Legal->getWidestInductionType();
  APInt MaxUIntTripCount = IdxTy->getMask();

  // We know the runtime overflow check is known false iff the (max) trip-count
  // is known and (max) trip-count + (VF * UF) does not overflow in the type of
  // the vector loop induction variable.
  if (unsigned TC = Cost->PSE.getSmallConstantMaxTripCount()) {
    uint64_t MaxVF = VF.getKnownMinValue();
    if (VF.isScalable()) {
      std::optional<unsigned> MaxVScale =
          getMaxVScale(*Cost->TheFunction, Cost->TTI);
      if (!MaxVScale)
        return false;
      MaxVF *= *MaxVScale;
    }

    return (MaxUIntTripCount - TC).ugt(MaxVF * MaxUF);
  }

  return false;
}

// Return whether we allow using masked interleave-groups (for dealing with
// strided loads/stores that reside in predicated blocks, or for dealing
// with gaps).
static bool useMaskedInterleavedAccesses(const TargetTransformInfo &TTI) {
  // If an override option has been passed in for interleaved accesses, use it.
  if (EnableMaskedInterleavedMemAccesses.getNumOccurrences() > 0)
    return EnableMaskedInterleavedMemAccesses;

  return TTI.enableMaskedInterleavedAccessVectorization();
}

/// Replace \p VPBB with a VPIRBasicBlock wrapping \p IRBB. All recipes from \p
/// VPBB are moved to the end of the newly created VPIRBasicBlock. All
/// predecessors and successors of VPBB, if any, are rewired to the new
/// VPIRBasicBlock. If \p VPBB may be unreachable, \p Plan must be passed.
static VPIRBasicBlock *replaceVPBBWithIRVPBB(VPBasicBlock *VPBB,
                                             BasicBlock *IRBB,
                                             VPlan *Plan = nullptr) {
  if (!Plan)
    Plan = VPBB->getPlan();
  VPIRBasicBlock *IRVPBB = Plan->createVPIRBasicBlock(IRBB);
  auto IP = IRVPBB->begin();
  for (auto &R : make_early_inc_range(VPBB->phis()))
    R.moveBefore(*IRVPBB, IP);

  for (auto &R :
       make_early_inc_range(make_range(VPBB->getFirstNonPhi(), VPBB->end())))
    R.moveBefore(*IRVPBB, IRVPBB->end());

  VPBlockUtils::reassociateBlocks(VPBB, IRVPBB);
  // VPBB is now dead and will be cleaned up when the plan gets destroyed.
  return IRVPBB;
}

BasicBlock *InnerLoopVectorizer::createScalarPreheader(StringRef Prefix) {
  BasicBlock *VectorPH = OrigLoop->getLoopPreheader();
  assert(VectorPH && "Invalid loop structure");
  assert((OrigLoop->getUniqueLatchExitBlock() ||
          Cost->requiresScalarEpilogue(VF.isVector())) &&
         "loops not exiting via the latch without required epilogue?");

  // NOTE: The Plan's scalar preheader VPBB isn't replaced with a VPIRBasicBlock
  // wrapping the newly created scalar preheader here at the moment, because the
  // Plan's scalar preheader may be unreachable at this point. Instead it is
  // replaced in executePlan.
  return SplitBlock(VectorPH, VectorPH->getTerminator(), DT, LI, nullptr,
                    Twine(Prefix) + "scalar.ph");
}

/// Knowing that loop \p L executes a single vector iteration, add instructions
/// that will get simplified and thus should not have any cost to \p
/// InstsToIgnore.
static void addFullyUnrolledInstructionsToIgnore(
    Loop *L, const LoopVectorizationLegality::InductionList &IL,
    SmallPtrSetImpl<Instruction *> &InstsToIgnore) {
  auto *Cmp = L->getLatchCmpInst();
  if (Cmp)
    InstsToIgnore.insert(Cmp);
  for (const auto &KV : IL) {
    // Extract the key by hand so that it can be used in the lambda below.  Note
    // that captured structured bindings are a C++20 extension.
    const PHINode *IV = KV.first;

    // Get next iteration value of the induction variable.
    Instruction *IVInst =
        cast<Instruction>(IV->getIncomingValueForBlock(L->getLoopLatch()));
    if (all_of(IVInst->users(),
               [&](const User *U) { return U == IV || U == Cmp; }))
      InstsToIgnore.insert(IVInst);
  }
}

BasicBlock *InnerLoopVectorizer::createVectorizedLoopSkeleton() {
  // Create a new IR basic block for the scalar preheader.
  BasicBlock *ScalarPH = createScalarPreheader("");
  return ScalarPH->getSinglePredecessor();
}

namespace {

struct CSEDenseMapInfo {
  static bool canHandle(const Instruction *I) {
    return isa<InsertElementInst>(I) || isa<ExtractElementInst>(I) ||
           isa<ShuffleVectorInst>(I) || isa<GetElementPtrInst>(I);
  }

  static inline Instruction *getEmptyKey() {
    return DenseMapInfo<Instruction *>::getEmptyKey();
  }

  static inline Instruction *getTombstoneKey() {
    return DenseMapInfo<Instruction *>::getTombstoneKey();
  }

  static unsigned getHashValue(const Instruction *I) {
    assert(canHandle(I) && "Unknown instruction!");
    return hash_combine(I->getOpcode(),
                        hash_combine_range(I->operand_values()));
  }

  static bool isEqual(const Instruction *LHS, const Instruction *RHS) {
    if (LHS == getEmptyKey() || RHS == getEmptyKey() ||
        LHS == getTombstoneKey() || RHS == getTombstoneKey())
      return LHS == RHS;
    return LHS->isIdenticalTo(RHS);
  }
};

} // end anonymous namespace

/// FIXME: This legacy common-subexpression-elimination routine is scheduled for
/// removal, in favor of the VPlan-based one.
static void legacyCSE(BasicBlock *BB) {
  // Perform simple cse.
  SmallDenseMap<Instruction *, Instruction *, 4, CSEDenseMapInfo> CSEMap;
  for (Instruction &In : llvm::make_early_inc_range(*BB)) {
    if (!CSEDenseMapInfo::canHandle(&In))
      continue;

    // Check if we can replace this instruction with any of the
    // visited instructions.
    if (Instruction *V = CSEMap.lookup(&In)) {
      In.replaceAllUsesWith(V);
      In.eraseFromParent();
      continue;
    }

    CSEMap[&In] = &In;
  }
}

/// This function attempts to return a value that represents the ElementCount
/// at runtime. For fixed-width VFs we know this precisely at compile
/// time, but for scalable VFs we calculate it based on an estimate of the
/// vscale value.
static unsigned estimateElementCount(ElementCount VF,
                                     std::optional<unsigned> VScale) {
  unsigned EstimatedVF = VF.getKnownMinValue();
  if (VF.isScalable())
    if (VScale)
      EstimatedVF *= *VScale;
  assert(EstimatedVF >= 1 && "Estimated VF shouldn't be less than 1");
  return EstimatedVF;
}

InstructionCost
LoopVectorizationCostModel::getVectorCallCost(CallInst *CI,
                                              ElementCount VF) const {
  // We only need to calculate a cost if the VF is scalar; for actual vectors
  // we should already have a pre-calculated cost at each VF.
  if (!VF.isScalar())
    return getCallWideningDecision(CI, VF).Cost;

  Type *RetTy = CI->getType();
  if (RecurrenceDescriptor::isFMulAddIntrinsic(CI))
    if (auto RedCost = getReductionPatternCost(CI, VF, RetTy))
      return *RedCost;

  SmallVector<Type *, 4> Tys;
  for (auto &ArgOp : CI->args())
    Tys.push_back(ArgOp->getType());

  InstructionCost ScalarCallCost = TTI.getCallInstrCost(
      CI->getCalledFunction(), RetTy, Tys, Config.CostKind);

  // If this is an intrinsic we may have a lower cost for it.
  if (getVectorIntrinsicIDForCall(CI, TLI)) {
    InstructionCost IntrinsicCost = getVectorIntrinsicCost(CI, VF);
    return std::min(ScalarCallCost, IntrinsicCost);
  }
  return ScalarCallCost;
}

static Type *maybeVectorizeType(Type *Ty, ElementCount VF) {
  if (VF.isScalar() || !canVectorizeTy(Ty))
    return Ty;
  return toVectorizedTy(Ty, VF);
}

InstructionCost
LoopVectorizationCostModel::getVectorIntrinsicCost(CallInst *CI,
                                                   ElementCount VF) const {
  Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);
  assert(ID && "Expected intrinsic call!");
  Type *RetTy = maybeVectorizeType(CI->getType(), VF);
  FastMathFlags FMF;
  if (auto *FPMO = dyn_cast<FPMathOperator>(CI))
    FMF = FPMO->getFastMathFlags();

  SmallVector<const Value *> Arguments(CI->args());
  FunctionType *FTy = CI->getCalledFunction()->getFunctionType();
  SmallVector<Type *> ParamTys;
  std::transform(FTy->param_begin(), FTy->param_end(),
                 std::back_inserter(ParamTys),
                 [&](Type *Ty) { return maybeVectorizeType(Ty, VF); });

  IntrinsicCostAttributes CostAttrs(ID, RetTy, Arguments, ParamTys, FMF,
                                    dyn_cast<IntrinsicInst>(CI),
                                    InstructionCost::getInvalid());
  return TTI.getIntrinsicInstrCost(CostAttrs, Config.CostKind);
}

void InnerLoopVectorizer::fixVectorizedLoop(VPTransformState &State) {
  // Fix widened non-induction PHIs by setting up the PHI operands.
  fixNonInductionPHIs(State);

  // Don't apply optimizations below when no (vector) loop remains, as they all
  // require one at the moment.
  VPBasicBlock *HeaderVPBB =
      vputils::getFirstLoopHeader(*State.Plan, State.VPDT);
  if (!HeaderVPBB)
    return;

  BasicBlock *HeaderBB = State.CFG.VPBB2IRBB[HeaderVPBB];

  // Remove redundant induction instructions.
  legacyCSE(HeaderBB);
}

void InnerLoopVectorizer::fixNonInductionPHIs(VPTransformState &State) {
  auto Iter = vp_depth_first_shallow(Plan.getEntry());
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(Iter)) {
    for (VPRecipeBase &P : VPBB->phis()) {
      VPWidenPHIRecipe *VPPhi = dyn_cast<VPWidenPHIRecipe>(&P);
      if (!VPPhi)
        continue;
      PHINode *NewPhi = cast<PHINode>(State.get(VPPhi));
      // Make sure the builder has a valid insert point.
      Builder.SetInsertPoint(NewPhi);
      for (const auto &[Inc, VPBB] : VPPhi->incoming_values_and_blocks())
        NewPhi->addIncoming(State.get(Inc), State.CFG.VPBB2IRBB[VPBB]);
    }
  }
}

void LoopVectorizationCostModel::collectLoopScalars(ElementCount VF) {
  // We should not collect Scalars more than once per VF. Right now, this
  // function is called from collectUniformsAndScalars(), which already does
  // this check. Collecting Scalars for VF=1 does not make any sense.
  assert(VF.isVector() && !Scalars.contains(VF) &&
         "This function should not be visited twice for the same VF");

  // This avoids any chances of creating a REPLICATE recipe during planning
  // since that would result in generation of scalarized code during execution,
  // which is not supported for scalable vectors.
  if (VF.isScalable()) {
    Scalars[VF].insert_range(Uniforms[VF]);
    return;
  }

  SmallSetVector<Instruction *, 8> Worklist;

  // These sets are used to seed the analysis with pointers used by memory
  // accesses that will remain scalar.
  SmallSetVector<Instruction *, 8> ScalarPtrs;
  SmallPtrSet<Instruction *, 8> PossibleNonScalarPtrs;
  auto *Latch = TheLoop->getLoopLatch();

  // A helper that returns true if the use of Ptr by MemAccess will be scalar.
  // The pointer operands of loads and stores will be scalar as long as the
  // memory access is not a gather or scatter operation. The value operand of a
  // store will remain scalar if the store is scalarized.
  auto IsScalarUse = [&](Instruction *MemAccess, Value *Ptr) {
    InstWidening WideningDecision = getWideningDecision(MemAccess, VF);
    assert(WideningDecision != CM_Unknown &&
           "Widening decision should be ready at this moment");
    if (auto *Store = dyn_cast<StoreInst>(MemAccess))
      if (Ptr == Store->getValueOperand())
        return WideningDecision == CM_Scalarize;
    assert(Ptr == getLoadStorePointerOperand(MemAccess) &&
           "Ptr is neither a value or pointer operand");
    return WideningDecision != CM_GatherScatter;
  };

  // A helper that returns true if the given value is a getelementptr
  // instruction contained in the loop.
  auto IsLoopVaryingGEP = [&](Value *V) {
    return isa<GetElementPtrInst>(V) && !TheLoop->isLoopInvariant(V);
  };

  // A helper that evaluates a memory access's use of a pointer. If the use will
  // be a scalar use and the pointer is only used by memory accesses, we place
  // the pointer in ScalarPtrs. Otherwise, the pointer is placed in
  // PossibleNonScalarPtrs.
  auto EvaluatePtrUse = [&](Instruction *MemAccess, Value *Ptr) {
    // We only care about bitcast and getelementptr instructions contained in
    // the loop.
    if (!IsLoopVaryingGEP(Ptr))
      return;

    // If the pointer has already been identified as scalar (e.g., if it was
    // also identified as uniform), there's nothing to do.
    auto *I = cast<Instruction>(Ptr);
    if (Worklist.count(I))
      return;

    // If the use of the pointer will be a scalar use, and all users of the
    // pointer are memory accesses, place the pointer in ScalarPtrs. Otherwise,
    // place the pointer in PossibleNonScalarPtrs.
    if (IsScalarUse(MemAccess, Ptr) &&
        all_of(I->users(), IsaPred<LoadInst, StoreInst>))
      ScalarPtrs.insert(I);
    else
      PossibleNonScalarPtrs.insert(I);
  };

  // We seed the scalars analysis with three classes of instructions: (1)
  // instructions marked uniform-after-vectorization and (2) bitcast,
  // getelementptr and (pointer) phi instructions used by memory accesses
  // requiring a scalar use.
  //
  // (1) Add to the worklist all instructions that have been identified as
  // uniform-after-vectorization.
  Worklist.insert_range(Uniforms[VF]);

  // (2) Add to the worklist all bitcast and getelementptr instructions used by
  // memory accesses requiring a scalar use. The pointer operands of loads and
  // stores will be scalar unless the operation is a gather or scatter.
  // The value operand of a store will remain scalar if the store is scalarized.
  for (auto *BB : TheLoop->blocks())
    for (auto &I : *BB) {
      if (auto *Load = dyn_cast<LoadInst>(&I)) {
        EvaluatePtrUse(Load, Load->getPointerOperand());
      } else if (auto *Store = dyn_cast<StoreInst>(&I)) {
        EvaluatePtrUse(Store, Store->getPointerOperand());
        EvaluatePtrUse(Store, Store->getValueOperand());
      }
    }
  for (auto *I : ScalarPtrs)
    if (!PossibleNonScalarPtrs.count(I)) {
      LLVM_DEBUG(dbgs() << "LV: Found scalar instruction: " << *I << "\n");
      Worklist.insert(I);
    }

  // Insert the forced scalars.
  // FIXME: Currently VPWidenPHIRecipe() often creates a dead vector
  // induction variable when the PHI user is scalarized.
  auto ForcedScalar = ForcedScalars.find(VF);
  if (ForcedScalar != ForcedScalars.end())
    for (auto *I : ForcedScalar->second) {
      LLVM_DEBUG(dbgs() << "LV: Found (forced) scalar instruction: " << *I << "\n");
      Worklist.insert(I);
    }

  // Expand the worklist by looking through any bitcasts and getelementptr
  // instructions we've already identified as scalar. This is similar to the
  // expansion step in collectLoopUniforms(); however, here we're only
  // expanding to include additional bitcasts and getelementptr instructions.
  unsigned Idx = 0;
  while (Idx != Worklist.size()) {
    Instruction *Dst = Worklist[Idx++];
    if (!IsLoopVaryingGEP(Dst->getOperand(0)))
      continue;
    auto *Src = cast<Instruction>(Dst->getOperand(0));
    if (llvm::all_of(Src->users(), [&](User *U) -> bool {
          auto *J = cast<Instruction>(U);
          return !TheLoop->contains(J) || Worklist.count(J) ||
                 ((isa<LoadInst>(J) || isa<StoreInst>(J)) &&
                  IsScalarUse(J, Src));
        })) {
      Worklist.insert(Src);
      LLVM_DEBUG(dbgs() << "LV: Found scalar instruction: " << *Src << "\n");
    }
  }

  // An induction variable will remain scalar if all users of the induction
  // variable and induction variable update remain scalar.
  for (const auto &Induction : Legal->getInductionVars()) {
    auto *Ind = Induction.first;
    auto *IndUpdate = cast<Instruction>(Ind->getIncomingValueForBlock(Latch));

    // If tail-folding is applied, the primary induction variable will be used
    // to feed a vector compare.
    if (Ind == Legal->getPrimaryInduction() && foldTailByMasking())
      continue;

    // Returns true if \p Indvar is a pointer induction that is used directly by
    // load/store instruction \p I.
    auto IsDirectLoadStoreFromPtrIndvar = [&](Instruction *Indvar,
                                              Instruction *I) {
      return Induction.second.getKind() ==
                 InductionDescriptor::IK_PtrInduction &&
             (isa<LoadInst>(I) || isa<StoreInst>(I)) &&
             Indvar == getLoadStorePointerOperand(I) && IsScalarUse(I, Indvar);
    };

    // Determine if all users of the induction variable are scalar after
    // vectorization.
    bool ScalarInd = all_of(Ind->users(), [&](User *U) -> bool {
      auto *I = cast<Instruction>(U);
      return I == IndUpdate || !TheLoop->contains(I) || Worklist.count(I) ||
             IsDirectLoadStoreFromPtrIndvar(Ind, I);
    });
    if (!ScalarInd)
      continue;

    // If the induction variable update is a fixed-order recurrence, neither the
    // induction variable or its update should be marked scalar after
    // vectorization.
    auto *IndUpdatePhi = dyn_cast<PHINode>(IndUpdate);
    if (IndUpdatePhi && Legal->isFixedOrderRecurrence(IndUpdatePhi))
      continue;

    // Determine if all users of the induction variable update instruction are
    // scalar after vectorization.
    bool ScalarIndUpdate = all_of(IndUpdate->users(), [&](User *U) -> bool {
      auto *I = cast<Instruction>(U);
      return I == Ind || !TheLoop->contains(I) || Worklist.count(I) ||
             IsDirectLoadStoreFromPtrIndvar(IndUpdate, I);
    });
    if (!ScalarIndUpdate)
      continue;

    // The induction variable and its update instruction will remain scalar.
    Worklist.insert(Ind);
    Worklist.insert(IndUpdate);
    LLVM_DEBUG(dbgs() << "LV: Found scalar instruction: " << *Ind << "\n");
    LLVM_DEBUG(dbgs() << "LV: Found scalar instruction: " << *IndUpdate
                      << "\n");
  }

  Scalars[VF].insert_range(Worklist);
}

bool LoopVectorizationCostModel::isScalarWithPredication(Instruction *I,
                                                         ElementCount VF) {
  if (!isPredicatedInst(I))
    return false;

  // Do we have a non-scalar lowering for this predicated
  // instruction? No - it is scalar with predication.
  switch(I->getOpcode()) {
  default:
    return true;
  case Instruction::Call:
    if (VF.isScalar())
      return true;
    return getCallWideningDecision(cast<CallInst>(I), VF).Kind == CM_Scalarize;
  case Instruction::Load:
  case Instruction::Store: {
    auto *Ptr = getLoadStorePointerOperand(I);
    auto *Ty = getLoadStoreType(I);
    unsigned AS = getLoadStoreAddressSpace(I);
    Type *VTy = Ty;
    if (VF.isVector())
      VTy = VectorType::get(Ty, VF);
    const Align Alignment = getLoadStoreAlignment(I);
    return isa<LoadInst>(I)
               ? !(Config.isLegalMaskedLoad(Ty, Ptr, Alignment, AS) ||
                   TTI.isLegalMaskedGather(VTy, Alignment))
               : !(Config.isLegalMaskedStore(Ty, Ptr, Alignment, AS) ||
                   TTI.isLegalMaskedScatter(VTy, Alignment));
  }
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::URem: {
    // We have the option to use the safe-divisor idiom to avoid predication.
    // The cost based decision here will always select safe-divisor for
    // scalable vectors as scalarization isn't legal.
    const auto [ScalarCost, SafeDivisorCost] = getDivRemSpeculationCost(I, VF);
    return isDivRemScalarWithPredication(ScalarCost, SafeDivisorCost);
  }
  }
}

bool LoopVectorizationCostModel::isMaskRequired(Instruction *I) const {
  return Legal->isMaskRequired(I, foldTailByMasking());
}

// TODO: Fold into LoopVectorizationLegality::isMaskRequired.
bool LoopVectorizationCostModel::isPredicatedInst(Instruction *I) const {
  // TODO: We can use the loop-preheader as context point here and get
  // context sensitive reasoning for isSafeToSpeculativelyExecute.
  if (isSafeToSpeculativelyExecute(I) ||
      (isa<LoadInst, StoreInst, CallInst>(I) && !isMaskRequired(I)) ||
      isa<UncondBrInst, CondBrInst, SwitchInst, PHINode, AllocaInst>(I))
    return false;

  // If the instruction was executed conditionally in the original scalar loop,
  // predication is needed with a mask whose lanes are all possibly inactive.
  if (Legal->blockNeedsPredication(I->getParent()))
    return true;

  // If we're not folding the tail by masking, predication is unnecessary.
  if (!foldTailByMasking())
    return false;

  // All that remain are instructions with side-effects originally executed in
  // the loop unconditionally, but now execute under a tail-fold mask (only)
  // having at least one active lane (the first). If the side-effects of the
  // instruction are invariant, executing it w/o (the tail-folding) mask is safe
  // - it will cause the same side-effects as when masked.
  switch(I->getOpcode()) {
  default:
    llvm_unreachable(
        "instruction should have been considered by earlier checks");
  case Instruction::Call:
    // Side-effects of a Call are assumed to be non-invariant, needing a
    // (fold-tail) mask.
    assert(isMaskRequired(I) &&
           "should have returned earlier for calls not needing a mask");
    return true;
  case Instruction::Load:
    // If the address is loop invariant no predication is needed.
    return !Legal->isInvariant(getLoadStorePointerOperand(I));
  case Instruction::Store: {
    // For stores, we need to prove both speculation safety (which follows from
    // the same argument as loads), but also must prove the value being stored
    // is correct.  The easiest form of the later is to require that all values
    // stored are the same.
    return !(Legal->isInvariant(getLoadStorePointerOperand(I)) &&
             TheLoop->isLoopInvariant(cast<StoreInst>(I)->getValueOperand()));
  }
  case Instruction::UDiv:
  case Instruction::URem:
    // If the divisor is loop-invariant no predication is needed.
    return !Legal->isInvariant(I->getOperand(1));
  case Instruction::SDiv:
  case Instruction::SRem:
    // Conservative for now, since masked-off lanes may be poison and could
    // trigger signed overflow.
    return true;
  }
}

uint64_t LoopVectorizationCostModel::getPredBlockCostDivisor(
    TargetTransformInfo::TargetCostKind CostKind, const BasicBlock *BB) {
  if (CostKind == TTI::TCK_CodeSize)
    return 1;
  // If the block wasn't originally predicated then return early to avoid
  // computing BlockFrequencyInfo unnecessarily.
  if (!Legal->blockNeedsPredication(BB))
    return 1;

  uint64_t HeaderFreq =
      getBFI().getBlockFreq(TheLoop->getHeader()).getFrequency();
  uint64_t BBFreq = getBFI().getBlockFreq(BB).getFrequency();
  assert(HeaderFreq >= BBFreq &&
         "Header has smaller block freq than dominated BB?");
  return std::round((double)HeaderFreq / BBFreq);
}

std::pair<InstructionCost, InstructionCost>
LoopVectorizationCostModel::getDivRemSpeculationCost(Instruction *I,
                                                     ElementCount VF) {
  assert(I->getOpcode() == Instruction::UDiv ||
         I->getOpcode() == Instruction::SDiv ||
         I->getOpcode() == Instruction::SRem ||
         I->getOpcode() == Instruction::URem);
  assert(!isSafeToSpeculativelyExecute(I));

  // Scalarization isn't legal for scalable vector types
  InstructionCost ScalarizationCost = InstructionCost::getInvalid();
  if (!VF.isScalable()) {
    // Get the scalarization cost and scale this amount by the probability of
    // executing the predicated block. If the instruction is not predicated,
    // we fall through to the next case.
    ScalarizationCost = 0;

    // These instructions have a non-void type, so account for the phi nodes
    // that we will create. This cost is likely to be zero. The phi node
    // cost, if any, should be scaled by the block probability because it
    // models a copy at the end of each predicated block.
    ScalarizationCost += VF.getFixedValue() *
                         TTI.getCFInstrCost(Instruction::PHI, Config.CostKind);

    // The cost of the non-predicated instruction.
    ScalarizationCost +=
        VF.getFixedValue() * TTI.getArithmeticInstrCost(
                                 I->getOpcode(), I->getType(), Config.CostKind);

    // The cost of insertelement and extractelement instructions needed for
    // scalarization.
    ScalarizationCost += getScalarizationOverhead(I, VF);

    // Scale the cost by the probability of executing the predicated blocks.
    // This assumes the predicated block for each vector lane is equally
    // likely.
    ScalarizationCost =
        ScalarizationCost /
        getPredBlockCostDivisor(Config.CostKind, I->getParent());
  }

  InstructionCost SafeDivisorCost = 0;
  auto *VecTy = toVectorTy(I->getType(), VF);
  // The cost of the select guard to ensure all lanes are well defined
  // after we speculate above any internal control flow.
  SafeDivisorCost +=
      TTI.getCmpSelInstrCost(Instruction::Select, VecTy,
                             toVectorTy(Type::getInt1Ty(I->getContext()), VF),
                             CmpInst::BAD_ICMP_PREDICATE, Config.CostKind);

  SmallVector<const Value *, 4> Operands(I->operand_values());
  SafeDivisorCost += TTI.getArithmeticInstrCost(
      I->getOpcode(), VecTy, Config.CostKind,
      {TargetTransformInfo::OK_AnyValue, TargetTransformInfo::OP_None},
      {TargetTransformInfo::OK_AnyValue, TargetTransformInfo::OP_None},
      Operands, I);
  return {ScalarizationCost, SafeDivisorCost};
}

bool LoopVectorizationCostModel::interleavedAccessCanBeWidened(
    Instruction *I, ElementCount VF) const {
  assert(isAccessInterleaved(I) && "Expecting interleaved access.");
  assert(getWideningDecision(I, VF) == CM_Unknown &&
         "Decision should not be set yet.");
  auto *Group = getInterleavedAccessGroup(I);
  assert(Group && "Must have a group.");
  unsigned InterleaveFactor = Group->getFactor();

  // If the instruction's allocated size doesn't equal its type size, it
  // requires padding and will be scalarized.
  auto &DL = I->getDataLayout();
  auto *ScalarTy = getLoadStoreType(I);
  if (hasIrregularType(ScalarTy, DL))
    return false;

  // For scalable vectors, the interleave factors must be <= 8 since we require
  // the (de)interleaveN intrinsics instead of shufflevectors.
  if (VF.isScalable() && InterleaveFactor > 8)
    return false;

  // If the group involves a non-integral pointer, we may not be able to
  // losslessly cast all values to a common type.
  bool ScalarNI = DL.isNonIntegralPointerType(ScalarTy);
  for (Instruction *Member : Group->members()) {
    auto *MemberTy = getLoadStoreType(Member);
    bool MemberNI = DL.isNonIntegralPointerType(MemberTy);
    // Don't coerce non-integral pointers to integers or vice versa.
    if (MemberNI != ScalarNI)
      // TODO: Consider adding special nullptr value case here
      return false;
    if (MemberNI && ScalarNI &&
        ScalarTy->getPointerAddressSpace() !=
            MemberTy->getPointerAddressSpace())
      return false;
  }

  // Check if masking is required.
  // A Group may need masking for one of two reasons: it resides in a block that
  // needs predication, or it was decided to use masking to deal with gaps
  // (either a gap at the end of a load-access that may result in a speculative
  // load, or any gaps in a store-access).
  bool PredicatedAccessRequiresMasking =
      blockNeedsPredicationForAnyReason(I->getParent()) && isMaskRequired(I);
  bool LoadAccessWithGapsRequiresEpilogMasking =
      isa<LoadInst>(I) && Group->requiresScalarEpilogue() &&
      !isEpilogueAllowed();
  bool StoreAccessWithGapsRequiresMasking =
      isa<StoreInst>(I) && !Group->isFull();
  if (!PredicatedAccessRequiresMasking &&
      !LoadAccessWithGapsRequiresEpilogMasking &&
      !StoreAccessWithGapsRequiresMasking)
    return true;

  // If masked interleaving is required, we expect that the user/target had
  // enabled it, because otherwise it either wouldn't have been created or
  // it should have been invalidated by the CostModel.
  assert(useMaskedInterleavedAccesses(TTI) &&
         "Masked interleave-groups for predicated accesses are not enabled.");

  if (Group->isReverse())
    return false;

  // TODO: Support interleaved access that requires a gap mask for scalable VFs.
  bool NeedsMaskForGaps = LoadAccessWithGapsRequiresEpilogMasking ||
                          StoreAccessWithGapsRequiresMasking;
  if (VF.isScalable() && NeedsMaskForGaps)
    return false;

  auto *Ty = getLoadStoreType(I);
  const Align Alignment = getLoadStoreAlignment(I);
  unsigned AS = getLoadStoreAddressSpace(I);
  return isa<LoadInst>(I) ? TTI.isLegalMaskedLoad(Ty, Alignment, AS)
                          : TTI.isLegalMaskedStore(Ty, Alignment, AS);
}

bool LoopVectorizationCostModel::memoryInstructionCanBeWidened(
    Instruction *I, ElementCount VF) {
  // Get and ensure we have a valid memory instruction.
  assert((isa<LoadInst, StoreInst>(I)) && "Invalid memory instruction");

  auto *Ptr = getLoadStorePointerOperand(I);
  auto *ScalarTy = getLoadStoreType(I);

  // In order to be widened, the pointer should be consecutive, first of all.
  if (!Legal->isConsecutivePtr(ScalarTy, Ptr))
    return false;

  // If the instruction is a store located in a predicated block, it will be
  // scalarized.
  if (isScalarWithPredication(I, VF))
    return false;

  // If the instruction's allocated size doesn't equal it's type size, it
  // requires padding and will be scalarized.
  auto &DL = I->getDataLayout();
  if (hasIrregularType(ScalarTy, DL))
    return false;

  return true;
}

void LoopVectorizationCostModel::collectLoopUniforms(ElementCount VF) {
  // We should not collect Uniforms more than once per VF. Right now,
  // this function is called from collectUniformsAndScalars(), which
  // already does this check. Collecting Uniforms for VF=1 does not make any
  // sense.

  assert(VF.isVector() && !Uniforms.contains(VF) &&
         "This function should not be visited twice for the same VF");

  // Visit the list of Uniforms. If we find no uniform value, we won't
  // analyze again.  Uniforms.count(VF) will return 1.
  Uniforms[VF].clear();

  // Now we know that the loop is vectorizable!
  // Collect instructions inside the loop that will remain uniform after
  // vectorization.

  // Global values, params and instructions outside of current loop are out of
  // scope.
  auto IsOutOfScope = [&](Value *V) -> bool {
    Instruction *I = dyn_cast<Instruction>(V);
    return (!I || !TheLoop->contains(I));
  };

  // Worklist containing uniform instructions demanding lane 0.
  SetVector<Instruction *> Worklist;

  // Add uniform instructions demanding lane 0 to the worklist. Instructions
  // that require predication must not be considered uniform after
  // vectorization, because that would create an erroneous replicating region
  // where only a single instance out of VF should be formed.
  auto AddToWorklistIfAllowed = [&](Instruction *I) -> void {
    if (IsOutOfScope(I)) {
      LLVM_DEBUG(dbgs() << "LV: Found not uniform due to scope: "
                        << *I << "\n");
      return;
    }
    if (isPredicatedInst(I)) {
      LLVM_DEBUG(
          dbgs() << "LV: Found not uniform due to requiring predication: " << *I
                 << "\n");
      return;
    }
    LLVM_DEBUG(dbgs() << "LV: Found uniform instruction: " << *I << "\n");
    Worklist.insert(I);
  };

  // Start with the conditional branches exiting the loop. If the branch
  // condition is an instruction contained in the loop that is only used by the
  // branch, it is uniform. Note conditions from uncountable early exits are not
  // uniform.
  SmallVector<BasicBlock *> Exiting;
  TheLoop->getExitingBlocks(Exiting);
  for (BasicBlock *E : Exiting) {
    if (Legal->hasUncountableEarlyExit() && TheLoop->getLoopLatch() != E)
      continue;
    auto *Cmp = dyn_cast<Instruction>(E->getTerminator()->getOperand(0));
    if (Cmp && TheLoop->contains(Cmp) && Cmp->hasOneUse())
      AddToWorklistIfAllowed(Cmp);
  }

  auto PrevVF = VF.divideCoefficientBy(2);
  // Return true if all lanes perform the same memory operation, and we can
  // thus choose to execute only one.
  auto IsUniformMemOpUse = [&](Instruction *I) {
    // If the value was already known to not be uniform for the previous
    // (smaller VF), it cannot be uniform for the larger VF.
    if (PrevVF.isVector()) {
      auto Iter = Uniforms.find(PrevVF);
      if (Iter != Uniforms.end() && !Iter->second.contains(I))
        return false;
    }
    if (!Legal->isUniformMemOp(*I, VF))
      return false;
    if (isa<LoadInst>(I))
      // Loading the same address always produces the same result - at least
      // assuming aliasing and ordering which have already been checked.
      return true;
    // Storing the same value on every iteration.
    return TheLoop->isLoopInvariant(cast<StoreInst>(I)->getValueOperand());
  };

  auto IsUniformDecision = [&](Instruction *I, ElementCount VF) {
    InstWidening WideningDecision = getWideningDecision(I, VF);
    assert(WideningDecision != CM_Unknown &&
           "Widening decision should be ready at this moment");

    if (IsUniformMemOpUse(I))
      return true;

    return (WideningDecision == CM_Widen ||
            WideningDecision == CM_Widen_Reverse ||
            WideningDecision == CM_Interleave);
  };

  // Returns true if Ptr is the pointer operand of a memory access instruction
  // I, I is known to not require scalarization, and the pointer is not also
  // stored.
  auto IsVectorizedMemAccessUse = [&](Instruction *I, Value *Ptr) -> bool {
    if (isa<StoreInst>(I) && I->getOperand(0) == Ptr)
      return false;
    return getLoadStorePointerOperand(I) == Ptr &&
           (IsUniformDecision(I, VF) || Legal->isInvariant(Ptr));
  };

  // Holds a list of values which are known to have at least one uniform use.
  // Note that there may be other uses which aren't uniform.  A "uniform use"
  // here is something which only demands lane 0 of the unrolled iterations;
  // it does not imply that all lanes produce the same value (e.g. this is not
  // the usual meaning of uniform)
  SetVector<Value *> HasUniformUse;

  // Scan the loop for instructions which are either a) known to have only
  // lane 0 demanded or b) are uses which demand only lane 0 of their operand.
  for (auto *BB : TheLoop->blocks())
    for (auto &I : *BB) {
      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I)) {
        switch (II->getIntrinsicID()) {
        case Intrinsic::sideeffect:
        case Intrinsic::experimental_noalias_scope_decl:
        case Intrinsic::assume:
        case Intrinsic::lifetime_start:
        case Intrinsic::lifetime_end:
          if (TheLoop->hasLoopInvariantOperands(&I))
            AddToWorklistIfAllowed(&I);
          break;
        default:
          break;
        }
      }

      if (auto *EVI = dyn_cast<ExtractValueInst>(&I)) {
        if (IsOutOfScope(EVI->getAggregateOperand())) {
          AddToWorklistIfAllowed(EVI);
          continue;
        }
        // Only ExtractValue instructions where the aggregate value comes from a
        // call are allowed to be non-uniform.
        assert(isa<CallInst>(EVI->getAggregateOperand()) &&
               "Expected aggregate value to be call return value");
      }

      // If there's no pointer operand, there's nothing to do.
      auto *Ptr = getLoadStorePointerOperand(&I);
      if (!Ptr)
        continue;

      // If the pointer can be proven to be uniform, always add it to the
      // worklist.
      if (isa<Instruction>(Ptr) && Legal->isUniform(Ptr, VF))
        AddToWorklistIfAllowed(cast<Instruction>(Ptr));

      if (IsUniformMemOpUse(&I))
        AddToWorklistIfAllowed(&I);

      if (IsVectorizedMemAccessUse(&I, Ptr))
        HasUniformUse.insert(Ptr);
    }

  // Add to the worklist any operands which have *only* uniform (e.g. lane 0
  // demanding) users.  Since loops are assumed to be in LCSSA form, this
  // disallows uses outside the loop as well.
  for (auto *V : HasUniformUse) {
    if (IsOutOfScope(V))
      continue;
    auto *I = cast<Instruction>(V);
    bool UsersAreMemAccesses = all_of(I->users(), [&](User *U) -> bool {
      auto *UI = cast<Instruction>(U);
      return TheLoop->contains(UI) && IsVectorizedMemAccessUse(UI, V);
    });
    if (UsersAreMemAccesses)
      AddToWorklistIfAllowed(I);
  }

  // Expand Worklist in topological order: whenever a new instruction
  // is added , its users should be already inside Worklist.  It ensures
  // a uniform instruction will only be used by uniform instructions.
  unsigned Idx = 0;
  while (Idx != Worklist.size()) {
    Instruction *I = Worklist[Idx++];

    for (auto *OV : I->operand_values()) {
      // isOutOfScope operands cannot be uniform instructions.
      if (IsOutOfScope(OV))
        continue;
      // First order recurrence Phi's should typically be considered
      // non-uniform.
      auto *OP = dyn_cast<PHINode>(OV);
      if (OP && Legal->isFixedOrderRecurrence(OP))
        continue;
      // If all the users of the operand are uniform, then add the
      // operand into the uniform worklist.
      auto *OI = cast<Instruction>(OV);
      if (llvm::all_of(OI->users(), [&](User *U) -> bool {
            auto *J = cast<Instruction>(U);
            return Worklist.count(J) || IsVectorizedMemAccessUse(J, OI);
          }))
        AddToWorklistIfAllowed(OI);
    }
  }

  // For an instruction to be added into Worklist above, all its users inside
  // the loop should also be in Worklist. However, this condition cannot be
  // true for phi nodes that form a cyclic dependence. We must process phi
  // nodes separately. An induction variable will remain uniform if all users
  // of the induction variable and induction variable update remain uniform.
  // The code below handles both pointer and non-pointer induction variables.
  BasicBlock *Latch = TheLoop->getLoopLatch();
  for (const auto &Induction : Legal->getInductionVars()) {
    auto *Ind = Induction.first;
    auto *IndUpdate = cast<Instruction>(Ind->getIncomingValueForBlock(Latch));

    // Determine if all users of the induction variable are uniform after
    // vectorization.
    bool UniformInd = all_of(Ind->users(), [&](User *U) -> bool {
      auto *I = cast<Instruction>(U);
      return I == IndUpdate || !TheLoop->contains(I) || Worklist.count(I) ||
             IsVectorizedMemAccessUse(I, Ind);
    });
    if (!UniformInd)
      continue;

    // Determine if all users of the induction variable update instruction are
    // uniform after vectorization.
    bool UniformIndUpdate = all_of(IndUpdate->users(), [&](User *U) -> bool {
      auto *I = cast<Instruction>(U);
      return I == Ind || Worklist.count(I) ||
             IsVectorizedMemAccessUse(I, IndUpdate);
    });
    if (!UniformIndUpdate)
      continue;

    // The induction variable and its update instruction will remain uniform.
    AddToWorklistIfAllowed(Ind);
    AddToWorklistIfAllowed(IndUpdate);
  }

  Uniforms[VF].insert_range(Worklist);
}

FixedScalableVFPair
LoopVectorizationCostModel::computeMaxVF(ElementCount UserVF, unsigned UserIC) {
  if (Legal->getRuntimePointerChecking()->Need && TTI.hasBranchDivergence()) {
    // TODO: It may be useful to do since it's still likely to be dynamically
    // uniform if the target can skip.
    reportVectorizationFailure(
        "Not inserting runtime ptr check for divergent target",
        "runtime pointer checks needed. Not enabled for divergent target",
        "CantVersionLoopWithDivergentTarget", ORE, TheLoop);
    return FixedScalableVFPair::getNone();
  }

  ScalarEvolution *SE = PSE.getSE();
  ElementCount TC = getSmallConstantTripCount(SE, TheLoop);
  unsigned MaxTC = PSE.getSmallConstantMaxTripCount();
  if (!MaxTC && EpilogueLoweringStatus == CM_EpilogueAllowed)
    MaxTC = getMaxTCFromNonZeroRange(PSE, TheLoop);
  LLVM_DEBUG(dbgs() << "LV: Found trip count: " << TC << '\n');
  if (TC != ElementCount::getFixed(MaxTC))
    LLVM_DEBUG(dbgs() << "LV: Found maximum trip count: " << MaxTC << '\n');
  if (TC.isScalar()) {
    reportVectorizationFailure("Single iteration (non) loop",
        "loop trip count is one, irrelevant for vectorization",
        "SingleIterationLoop", ORE, TheLoop);
    return FixedScalableVFPair::getNone();
  }

  // If BTC matches the widest induction type and is -1 then the trip count
  // computation will wrap to 0 and the vector trip count will be 0. Do not try
  // to vectorize.
  const SCEV *BTC = SE->getBackedgeTakenCount(TheLoop);
  if (!isa<SCEVCouldNotCompute>(BTC) &&
      BTC->getType()->getScalarSizeInBits() >=
          Legal->getWidestInductionType()->getScalarSizeInBits() &&
      SE->isKnownPredicate(CmpInst::ICMP_EQ, BTC,
                           SE->getMinusOne(BTC->getType()))) {
    reportVectorizationFailure(
        "Trip count computation wrapped",
        "backedge-taken count is -1, loop trip count wrapped to 0",
        "TripCountWrapped", ORE, TheLoop);
    return FixedScalableVFPair::getNone();
  }

  assert(WideningDecisions.empty() && CallWideningDecisions.empty() &&
         Uniforms.empty() && Scalars.empty() &&
         "No cost-modeling decisions should have been taken at this point");

  switch (EpilogueLoweringStatus) {
  case CM_EpilogueAllowed:
    return Config.computeFeasibleMaxVF(MaxTC, UserVF, UserIC, false,
                                       requiresScalarEpilogue(true));
  case CM_EpilogueNotAllowedFoldTail:
    [[fallthrough]];
  case CM_EpilogueNotNeededFoldTail:
    LLVM_DEBUG(dbgs() << "LV: tail-folding hint/switch found.\n"
                      << "LV: Not allowing epilogue, creating tail-folded "
                      << "vector loop.\n");
    break;
  case CM_EpilogueNotAllowedLowTripLoop:
    // fallthrough as a special case of OptForSize
  case CM_EpilogueNotAllowedOptSize:
    if (EpilogueLoweringStatus == CM_EpilogueNotAllowedOptSize)
      LLVM_DEBUG(dbgs() << "LV: Not allowing epilogue due to -Os/-Oz.\n");
    else
      LLVM_DEBUG(dbgs() << "LV: Not allowing epilogue due to low trip "
                        << "count.\n");

    // Bail if runtime checks are required, which are not good when optimising
    // for size.
    if (Config.runtimeChecksRequired())
      return FixedScalableVFPair::getNone();

    break;
  }

  // Now try the tail folding

  // Invalidate interleave groups that require an epilogue if we can't mask
  // the interleave-group.
  if (!useMaskedInterleavedAccesses(TTI)) {
    // Note: There is no need to invalidate any cost modeling decisions here, as
    // none were taken so far (see assertion above).
    InterleaveInfo.invalidateGroupsRequiringScalarEpilogue();
  }

  FixedScalableVFPair MaxFactors = Config.computeFeasibleMaxVF(
      MaxTC, UserVF, UserIC, true, requiresScalarEpilogue(true));

  // Avoid tail folding if the trip count is known to be a multiple of any VF
  // we choose.
  std::optional<unsigned> MaxPowerOf2RuntimeVF =
      MaxFactors.FixedVF.getFixedValue();
  if (MaxFactors.ScalableVF) {
    std::optional<unsigned> MaxVScale = getMaxVScale(*TheFunction, TTI);
    if (MaxVScale) {
      MaxPowerOf2RuntimeVF = std::max<unsigned>(
          *MaxPowerOf2RuntimeVF,
          *MaxVScale * MaxFactors.ScalableVF.getKnownMinValue());
    } else
      MaxPowerOf2RuntimeVF = std::nullopt; // Stick with tail-folding for now.
  }

  auto NoScalarEpilogueNeeded = [this, &UserIC](unsigned MaxVF) {
    // Return false if the loop is neither a single-latch-exit loop nor an
    // early-exit loop as tail-folding is not supported in that case.
    if (TheLoop->getExitingBlock() != TheLoop->getLoopLatch() &&
        !Legal->hasUncountableEarlyExit())
      return false;
    unsigned MaxVFtimesIC = UserIC ? MaxVF * UserIC : MaxVF;
    ScalarEvolution *SE = PSE.getSE();
    // Calling getSymbolicMaxBackedgeTakenCount enables support for loops
    // with uncountable exits. For countable loops, the symbolic maximum must
    // remain identical to the known back-edge taken count.
    const SCEV *BackedgeTakenCount = PSE.getSymbolicMaxBackedgeTakenCount();
    assert((Legal->hasUncountableEarlyExit() ||
            BackedgeTakenCount == PSE.getBackedgeTakenCount()) &&
           "Invalid loop count");
    const SCEV *ExitCount = SE->getAddExpr(
        BackedgeTakenCount, SE->getOne(BackedgeTakenCount->getType()));
    const SCEV *Rem = SE->getURemExpr(
        SE->applyLoopGuards(ExitCount, TheLoop),
        SE->getConstant(BackedgeTakenCount->getType(), MaxVFtimesIC));
    return Rem->isZero();
  };

  if (MaxPowerOf2RuntimeVF > 0u) {
    assert((UserVF.isNonZero() || isPowerOf2_32(*MaxPowerOf2RuntimeVF)) &&
           "MaxFixedVF must be a power of 2");
    if (NoScalarEpilogueNeeded(*MaxPowerOf2RuntimeVF)) {
      // Accept MaxFixedVF if we do not have a tail.
      LLVM_DEBUG(dbgs() << "LV: No tail will remain for any chosen VF.\n");
      return MaxFactors;
    }
  }

  auto ExpectedTC = getSmallBestKnownTC(PSE, TheLoop);
  if (ExpectedTC && ExpectedTC->isFixed() &&
      ExpectedTC->getFixedValue() <=
          TTI.getMinTripCountTailFoldingThreshold()) {
    if (MaxPowerOf2RuntimeVF > 0u) {
      // If we have a low-trip-count, and the fixed-width VF is known to divide
      // the trip count but the scalable factor does not, use the fixed-width
      // factor in preference to allow the generation of a non-predicated loop.
      if (EpilogueLoweringStatus == CM_EpilogueNotAllowedLowTripLoop &&
          NoScalarEpilogueNeeded(MaxFactors.FixedVF.getFixedValue())) {
        LLVM_DEBUG(dbgs() << "LV: Picking a fixed-width so that no tail will "
                             "remain for any chosen VF.\n");
        MaxFactors.ScalableVF = ElementCount::getScalable(0);
        return MaxFactors;
      }
    }

    reportVectorizationFailure(
        "The trip count is below the minial threshold value.",
        "loop trip count is too low, avoiding vectorization", "LowTripCount",
        ORE, TheLoop);
    return FixedScalableVFPair::getNone();
  }

  // If we don't know the precise trip count, or if the trip count that we
  // found modulo the vectorization factor is not zero, try to fold the tail
  // by masking.
  // FIXME: look for a smaller MaxVF that does divide TC rather than masking.
  bool ContainsScalableVF = MaxFactors.ScalableVF.isNonZero();
  setTailFoldingStyle(ContainsScalableVF, UserIC);
  if (foldTailByMasking()) {
    if (foldTailWithEVL()) {
      LLVM_DEBUG(
          dbgs()
          << "LV: tail is folded with EVL, forcing unroll factor to be 1. Will "
             "try to generate VP Intrinsics with scalable vector "
             "factors only.\n");
      // Tail folded loop using VP intrinsics restricts the VF to be scalable
      // for now.
      // TODO: extend it for fixed vectors, if required.
      assert(ContainsScalableVF && "Expected scalable vector factor.");

      MaxFactors.FixedVF = ElementCount::getFixed(1);
    }
    return MaxFactors;
  }

  // If there was a tail-folding hint/switch, but we can't fold the tail by
  // masking, fallback to a vectorization with an epilogue.
  if (EpilogueLoweringStatus == CM_EpilogueNotNeededFoldTail) {
    LLVM_DEBUG(dbgs() << "LV: Cannot fold tail by masking: vectorize with an "
                         "epilogue instead.\n");
    EpilogueLoweringStatus = CM_EpilogueAllowed;
    return MaxFactors;
  }

  if (EpilogueLoweringStatus == CM_EpilogueNotAllowedFoldTail) {
    LLVM_DEBUG(dbgs() << "LV: Can't fold tail by masking: don't vectorize\n");
    return FixedScalableVFPair::getNone();
  }

  if (TC.isZero()) {
    reportVectorizationFailure(
        "unable to calculate the loop count due to complex control flow",
        "UnknownLoopCountComplexCFG", ORE, TheLoop);
    return FixedScalableVFPair::getNone();
  }

  reportVectorizationFailure(
      "Cannot optimize for size and vectorize at the same time.",
      "cannot optimize for size and vectorize at the same time. "
      "Enable vectorization of this loop with '#pragma clang loop "
      "vectorize(enable)' when compiling with -Os/-Oz",
      "NoTailLoopWithOptForSize", ORE, TheLoop);
  return FixedScalableVFPair::getNone();
}

bool LoopVectorizationPlanner::isMoreProfitable(const VectorizationFactor &A,
                                                const VectorizationFactor &B,
                                                const unsigned MaxTripCount,
                                                bool HasTail,
                                                bool IsEpilogue) const {
  InstructionCost CostA = A.Cost;
  InstructionCost CostB = B.Cost;

  // When there is a hint to always prefer scalable vectors, honour that hint.
  if (Hints.isScalableVectorizationAlwaysPreferred())
    if (A.Width.isScalable() && CostA.isValid() && !B.Width.isScalable() &&
        !B.Width.isScalar())
      return true;

  // Improve estimate for the vector width if it is scalable.
  unsigned EstimatedWidthA = A.Width.getKnownMinValue();
  unsigned EstimatedWidthB = B.Width.getKnownMinValue();
  if (std::optional<unsigned> VScale = Config.getVScaleForTuning()) {
    if (A.Width.isScalable())
      EstimatedWidthA *= *VScale;
    if (B.Width.isScalable())
      EstimatedWidthB *= *VScale;
  }

  // When optimizing for size choose whichever is smallest, which will be the
  // one with the smallest cost for the whole loop. On a tie pick the larger
  // vector width, on the assumption that throughput will be greater.
  if (Config.CostKind == TTI::TCK_CodeSize)
    return CostA < CostB ||
           (CostA == CostB && EstimatedWidthA > EstimatedWidthB);

  // Assume vscale may be larger than 1 (or the value being tuned for),
  // so that scalable vectorization is slightly favorable over fixed-width
  // vectorization.
  bool PreferScalable = !TTI.preferFixedOverScalableIfEqualCost(IsEpilogue) &&
                        A.Width.isScalable() && !B.Width.isScalable();

  auto CmpFn = [PreferScalable](const InstructionCost &LHS,
                                const InstructionCost &RHS) {
    return PreferScalable ? LHS <= RHS : LHS < RHS;
  };

  // To avoid the need for FP division:
  //      (CostA / EstimatedWidthA) < (CostB / EstimatedWidthB)
  // <=>  (CostA * EstimatedWidthB) < (CostB * EstimatedWidthA)
  bool LowerCostWithoutTC =
      CmpFn(CostA * EstimatedWidthB, CostB * EstimatedWidthA);
  if (!MaxTripCount)
    return LowerCostWithoutTC;

  auto GetCostForTC = [MaxTripCount, HasTail](unsigned VF,
                                              InstructionCost VectorCost,
                                              InstructionCost ScalarCost) {
    // If the trip count is a known (possibly small) constant, the trip count
    // will be rounded up to an integer number of iterations under
    // FoldTailByMasking. The total cost in that case will be
    // VecCost*ceil(TripCount/VF). When not folding the tail, the total
    // cost will be VecCost*floor(TC/VF) + ScalarCost*(TC%VF). There will be
    // some extra overheads, but for the purpose of comparing the costs of
    // different VFs we can use this to compare the total loop-body cost
    // expected after vectorization.
    if (HasTail)
      return VectorCost * (MaxTripCount / VF) +
             ScalarCost * (MaxTripCount % VF);
    return VectorCost * divideCeil(MaxTripCount, VF);
  };

  auto RTCostA = GetCostForTC(EstimatedWidthA, CostA, A.ScalarCost);
  auto RTCostB = GetCostForTC(EstimatedWidthB, CostB, B.ScalarCost);
  bool LowerCostWithTC = CmpFn(RTCostA, RTCostB);
  LLVM_DEBUG(if (LowerCostWithTC != LowerCostWithoutTC) {
    dbgs() << "LV: VF " << (LowerCostWithTC ? A.Width : B.Width)
           << " has lower cost than VF "
           << (LowerCostWithTC ? B.Width : A.Width)
           << " when taking the cost of the remaining scalar loop iterations "
              "into consideration for a maximum trip count of "
           << MaxTripCount << ".\n";
  });
  return LowerCostWithTC;
}

bool LoopVectorizationPlanner::isMoreProfitable(const VectorizationFactor &A,
                                                const VectorizationFactor &B,
                                                bool HasTail,
                                                bool IsEpilogue) const {
  const unsigned MaxTripCount = PSE.getSmallConstantMaxTripCount();
  return LoopVectorizationPlanner::isMoreProfitable(A, B, MaxTripCount, HasTail,
                                                    IsEpilogue);
}

void LoopVectorizationPlanner::emitInvalidCostRemarks(
    OptimizationRemarkEmitter *ORE) {
  using RecipeVFPair = std::pair<VPRecipeBase *, ElementCount>;
  SmallVector<RecipeVFPair> InvalidCosts;
  for (const auto &Plan : VPlans) {
    for (ElementCount VF : Plan->vectorFactors()) {
      // The VPlan-based cost model is designed for computing vector cost.
      // Querying VPlan-based cost model with a scarlar VF will cause some
      // errors because we expect the VF is vector for most of the widen
      // recipes.
      if (VF.isScalar())
        continue;

      VPCostContext CostCtx(CM.TTI, *CM.TLI, *Plan, CM, Config.CostKind, CM.PSE,
                            OrigLoop);
      precomputeCosts(*Plan, VF, CostCtx);
      auto Iter = vp_depth_first_deep(Plan->getVectorLoopRegion()->getEntry());
      for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(Iter)) {
        for (auto &R : *VPBB) {
          if (!R.cost(VF, CostCtx).isValid())
            InvalidCosts.emplace_back(&R, VF);
        }
      }
    }
  }
  if (InvalidCosts.empty())
    return;

  // Emit a report of VFs with invalid costs in the loop.

  // Group the remarks per recipe, keeping the recipe order from InvalidCosts.
  DenseMap<VPRecipeBase *, unsigned> Numbering;
  unsigned I = 0;
  for (auto &Pair : InvalidCosts)
    if (Numbering.try_emplace(Pair.first, I).second)
      ++I;

  // Sort the list, first on recipe(number) then on VF.
  sort(InvalidCosts, [&Numbering](RecipeVFPair &A, RecipeVFPair &B) {
    unsigned NA = Numbering[A.first];
    unsigned NB = Numbering[B.first];
    if (NA != NB)
      return NA < NB;
    return ElementCount::isKnownLT(A.second, B.second);
  });

  // For a list of ordered recipe-VF pairs:
  //   [(load, VF1), (load, VF2), (store, VF1)]
  // group the recipes together to emit separate remarks for:
  //   load  (VF1, VF2)
  //   store (VF1)
  auto Tail = ArrayRef<RecipeVFPair>(InvalidCosts);
  auto Subset = ArrayRef<RecipeVFPair>();
  do {
    if (Subset.empty())
      Subset = Tail.take_front(1);

    VPRecipeBase *R = Subset.front().first;

    unsigned Opcode =
        TypeSwitch<const VPRecipeBase *, unsigned>(R)
            .Case([](const VPHeaderPHIRecipe *R) { return Instruction::PHI; })
            .Case(
                [](const VPWidenStoreRecipe *R) { return Instruction::Store; })
            .Case([](const VPWidenLoadRecipe *R) { return Instruction::Load; })
            .Case<VPWidenCallRecipe, VPWidenIntrinsicRecipe>(
                [](const auto *R) { return Instruction::Call; })
            .Case<VPInstruction, VPWidenRecipe, VPReplicateRecipe,
                  VPWidenCastRecipe>(
                [](const auto *R) { return R->getOpcode(); })
            .Case([](const VPInterleaveRecipe *R) {
              return R->getStoredValues().empty() ? Instruction::Load
                                                  : Instruction::Store;
            })
            .Case([](const VPReductionRecipe *R) {
              return RecurrenceDescriptor::getOpcode(R->getRecurrenceKind());
            });

    // If the next recipe is different, or if there are no other pairs,
    // emit a remark for the collated subset. e.g.
    //   [(load, VF1), (load, VF2))]
    // to emit:
    //  remark: invalid costs for 'load' at VF=(VF1, VF2)
    if (Subset == Tail || Tail[Subset.size()].first != R) {
      std::string OutString;
      raw_string_ostream OS(OutString);
      assert(!Subset.empty() && "Unexpected empty range");
      OS << "Recipe with invalid costs prevented vectorization at VF=(";
      for (const auto &Pair : Subset)
        OS << (Pair.second == Subset.front().second ? "" : ", ") << Pair.second;
      OS << "):";
      if (Opcode == Instruction::Call) {
        StringRef Name = "";
        if (auto *Int = dyn_cast<VPWidenIntrinsicRecipe>(R)) {
          Name = Int->getIntrinsicName();
        } else {
          auto *WidenCall = dyn_cast<VPWidenCallRecipe>(R);
          Function *CalledFn =
              WidenCall ? WidenCall->getCalledScalarFunction()
                        : cast<Function>(R->getOperand(R->getNumOperands() - 1)
                                             ->getLiveInIRValue());
          Name = CalledFn->getName();
        }
        OS << " call to " << Name;
      } else
        OS << " " << Instruction::getOpcodeName(Opcode);
      reportVectorizationInfo(OutString, "InvalidCost", ORE, OrigLoop, nullptr,
                              R->getDebugLoc());
      Tail = Tail.drop_front(Subset.size());
      Subset = {};
    } else
      // Grow the subset by one element
      Subset = Tail.take_front(Subset.size() + 1);
  } while (!Tail.empty());
}

/// Check if any recipe of \p Plan will generate a vector value, which will be
/// assigned a vector register.
static bool willGenerateVectors(VPlan &Plan, ElementCount VF,
                                const TargetTransformInfo &TTI) {
  assert(VF.isVector() && "Checking a scalar VF?");
  VPTypeAnalysis TypeInfo(Plan);
  DenseSet<VPRecipeBase *> EphemeralRecipes;
  collectEphemeralRecipesForVPlan(Plan, EphemeralRecipes);
  // Set of already visited types.
  DenseSet<Type *> Visited;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
           vp_depth_first_shallow(Plan.getVectorLoopRegion()->getEntry()))) {
    for (VPRecipeBase &R : *VPBB) {
      if (EphemeralRecipes.contains(&R))
        continue;
      // Continue early if the recipe is considered to not produce a vector
      // result. Note that this includes VPInstruction where some opcodes may
      // produce a vector, to preserve existing behavior as VPInstructions model
      // aspects not directly mapped to existing IR instructions.
      switch (R.getVPRecipeID()) {
      case VPRecipeBase::VPDerivedIVSC:
      case VPRecipeBase::VPScalarIVStepsSC:
      case VPRecipeBase::VPReplicateSC:
      case VPRecipeBase::VPInstructionSC:
      case VPRecipeBase::VPCurrentIterationPHISC:
      case VPRecipeBase::VPVectorPointerSC:
      case VPRecipeBase::VPVectorEndPointerSC:
      case VPRecipeBase::VPExpandSCEVSC:
      case VPRecipeBase::VPPredInstPHISC:
      case VPRecipeBase::VPBranchOnMaskSC:
        continue;
      case VPRecipeBase::VPReductionSC:
      case VPRecipeBase::VPActiveLaneMaskPHISC:
      case VPRecipeBase::VPWidenCallSC:
      case VPRecipeBase::VPWidenCanonicalIVSC:
      case VPRecipeBase::VPWidenCastSC:
      case VPRecipeBase::VPWidenGEPSC:
      case VPRecipeBase::VPWidenIntrinsicSC:
      case VPRecipeBase::VPWidenSC:
      case VPRecipeBase::VPBlendSC:
      case VPRecipeBase::VPFirstOrderRecurrencePHISC:
      case VPRecipeBase::VPHistogramSC:
      case VPRecipeBase::VPWidenPHISC:
      case VPRecipeBase::VPWidenIntOrFpInductionSC:
      case VPRecipeBase::VPWidenPointerInductionSC:
      case VPRecipeBase::VPReductionPHISC:
      case VPRecipeBase::VPInterleaveEVLSC:
      case VPRecipeBase::VPInterleaveSC:
      case VPRecipeBase::VPWidenLoadEVLSC:
      case VPRecipeBase::VPWidenLoadSC:
      case VPRecipeBase::VPWidenStoreEVLSC:
      case VPRecipeBase::VPWidenStoreSC:
        break;
      default:
        llvm_unreachable("unhandled recipe");
      }

      auto WillGenerateTargetVectors = [&TTI, VF](Type *VectorTy) {
        unsigned NumLegalParts = TTI.getNumberOfParts(VectorTy);
        if (!NumLegalParts)
          return false;
        if (VF.isScalable()) {
          // <vscale x 1 x iN> is assumed to be profitable over iN because
          // scalable registers are a distinct register class from scalar
          // ones. If we ever find a target which wants to lower scalable
          // vectors back to scalars, we'll need to update this code to
          // explicitly ask TTI about the register class uses for each part.
          return NumLegalParts <= VF.getKnownMinValue();
        }
        // Two or more elements that share a register - are vectorized.
        return NumLegalParts < VF.getFixedValue();
      };

      // If no def nor is a store, e.g., branches, continue - no value to check.
      if (R.getNumDefinedValues() == 0 &&
          !isa<VPWidenStoreRecipe, VPWidenStoreEVLRecipe, VPInterleaveBase>(&R))
        continue;
      // For multi-def recipes, currently only interleaved loads, suffice to
      // check first def only.
      // For stores check their stored value; for interleaved stores suffice
      // the check first stored value only. In all cases this is the second
      // operand.
      VPValue *ToCheck =
          R.getNumDefinedValues() >= 1 ? R.getVPValue(0) : R.getOperand(1);
      Type *ScalarTy = TypeInfo.inferScalarType(ToCheck);
      if (!Visited.insert({ScalarTy}).second)
        continue;
      Type *WideTy = toVectorizedTy(ScalarTy, VF);
      if (any_of(getContainedTypes(WideTy), WillGenerateTargetVectors))
        return true;
    }
  }

  return false;
}

static bool hasReplicatorRegion(VPlan &Plan) {
  return any_of(VPBlockUtils::blocksOnly<VPRegionBlock>(vp_depth_first_shallow(
                    Plan.getVectorLoopRegion()->getEntry())),
                [](auto *VPRB) { return VPRB->isReplicator(); });
}

/// Returns true if the VPlan contains a VPReductionPHIRecipe with
/// FindLast recurrence kind.
static bool hasFindLastReductionPhi(VPlan &Plan) {
  return any_of(Plan.getVectorLoopRegion()->getEntryBasicBlock()->phis(),
                [](VPRecipeBase &R) {
                  auto *RedPhi = dyn_cast<VPReductionPHIRecipe>(&R);
                  return RedPhi &&
                         RecurrenceDescriptor::isFindLastRecurrenceKind(
                             RedPhi->getRecurrenceKind());
                });
}

/// Returns true if the VPlan contains header phi recipes that are not currently
/// supported for epilogue vectorization.
static bool hasUnsupportedHeaderPhiRecipe(VPlan &Plan) {
  return any_of(
      Plan.getVectorLoopRegion()->getEntryBasicBlock()->phis(),
      [](VPRecipeBase &R) {
        switch (R.getVPRecipeID()) {
        case VPRecipeBase::VPFirstOrderRecurrencePHISC:
          // TODO: Add support for fixed-order recurrences.
          return true;
        case VPRecipeBase::VPWidenIntOrFpInductionSC:
          return !cast<VPWidenIntOrFpInductionRecipe>(&R)->getPHINode();
        case VPRecipeBase::VPReductionPHISC: {
          auto *RedPhi = cast<VPReductionPHIRecipe>(&R);
          // TODO: Support FMinNum/FMaxNum, FindLast reductions, and reductions
          // without underlying values.
          RecurKind Kind = RedPhi->getRecurrenceKind();
          if (RecurrenceDescriptor::isFPMinMaxNumRecurrenceKind(Kind) ||
              RecurrenceDescriptor::isFindLastRecurrenceKind(Kind) ||
              !RedPhi->getUnderlyingValue())
            return true;
          // TODO: Add support for FindIV reductions with sunk expressions: the
          // resume value from the main loop is in expression domain (e.g.,
          // mul(ReducedIV, 3)), but the epilogue tracks raw IV values. A sunk
          // expression is identified by a non-VPInstruction user of
          // ComputeReductionResult.
          if (RecurrenceDescriptor::isFindIVRecurrenceKind(Kind)) {
            auto *RdxResult = vputils::findComputeReductionResult(RedPhi);
            assert(RdxResult &&
                   "FindIV reduction must have ComputeReductionResult");
            return any_of(RdxResult->users(),
                          std::not_fn(IsaPred<VPInstruction>));
          }
          return false;
        }
        default:
          return false;
        };
      });
}

bool LoopVectorizationPlanner::isCandidateForEpilogueVectorization(
    VPlan &MainPlan) const {
  // Bail out if the plan contains header phi recipes not yet supported
  // for epilogue vectorization.
  if (hasUnsupportedHeaderPhiRecipe(MainPlan))
    return false;

  // Epilogue vectorization code has not been auditted to ensure it handles
  // non-latch exits properly.  It may be fine, but it needs auditted and
  // tested.
  // TODO: Add support for loops with an early exit.
  if (OrigLoop->getExitingBlock() != OrigLoop->getLoopLatch())
    return false;

  return true;
}

bool LoopVectorizationCostModel::isEpilogueVectorizationProfitable(
    const ElementCount VF, const unsigned IC) const {
  // FIXME: We need a much better cost-model to take different parameters such
  // as register pressure, code size increase and cost of extra branches into
  // account. For now we apply a very crude heuristic and only consider loops
  // with vectorization factors larger than a certain value.

  // Allow the target to opt out.
  if (!TTI.preferEpilogueVectorization(VF * IC))
    return false;

  unsigned MinVFThreshold = EpilogueVectorizationMinVF.getNumOccurrences() > 0
                                ? EpilogueVectorizationMinVF
                                : TTI.getEpilogueVectorizationMinVF();
  return estimateElementCount(VF * IC, Config.getVScaleForTuning()) >=
         MinVFThreshold;
}

std::unique_ptr<VPlan> LoopVectorizationPlanner::selectBestEpiloguePlan(
    VPlan &MainPlan, ElementCount MainLoopVF, unsigned IC) {
  if (!EnableEpilogueVectorization) {
    LLVM_DEBUG(dbgs() << "LEV: Epilogue vectorization is disabled.\n");
    return nullptr;
  }

  if (!CM.isEpilogueAllowed()) {
    LLVM_DEBUG(dbgs() << "LEV: Unable to vectorize epilogue because no "
                         "epilogue is allowed.\n");
    return nullptr;
  }

  // Not really a cost consideration, but check for unsupported cases here to
  // simplify the logic.
  if (!isCandidateForEpilogueVectorization(MainPlan)) {
    LLVM_DEBUG(dbgs() << "LEV: Unable to vectorize epilogue because the loop "
                         "is not a supported candidate.\n");
    return nullptr;
  }

  if (EpilogueVectorizationForceVF > 1) {
    if (EpilogueVectorizationForceVF >=
        IC * estimateElementCount(MainLoopVF, Config.getVScaleForTuning())) {
      // Note that the main loop leaves IC * MainLoopVF iterations iff a scalar
      // epilogue is required, but then the epilogue loop also requires a scalar
      // epilogue.
      LLVM_DEBUG(dbgs() << "LEV: Forced epilogue VF results in dead epilogue "
                           "vector loop, skipping vectorizing epilogue.\n");
      return nullptr;
    }

    LLVM_DEBUG(dbgs() << "LEV: Epilogue vectorization factor is forced.\n");
    ElementCount ForcedEC = ElementCount::getFixed(EpilogueVectorizationForceVF);
    if (hasPlanWithVF(ForcedEC)) {
      std::unique_ptr<VPlan> Clone(getPlanFor(ForcedEC).duplicate());
      Clone->setVF(ForcedEC);
      return Clone;
    }

    LLVM_DEBUG(dbgs() << "LEV: Epilogue vectorization forced factor is not "
                         "viable.\n");
    return nullptr;
  }

  if (OrigLoop->getHeader()->getParent()->hasOptSize()) {
    LLVM_DEBUG(
        dbgs() << "LEV: Epilogue vectorization skipped due to opt for size.\n");
    return nullptr;
  }

  if (!CM.isEpilogueVectorizationProfitable(MainLoopVF, IC)) {
    LLVM_DEBUG(dbgs() << "LEV: Epilogue vectorization is not profitable for "
                         "this loop\n");
    return nullptr;
  }

  // Check if a plan's vector loop processes fewer iterations than VF (e.g. when
  // interleave groups have been narrowed) narrowInterleaveGroups)  and return
  // the adjusted, effective VF.
  using namespace VPlanPatternMatch;
  auto GetEffectiveVF = [](VPlan &Plan, ElementCount VF) -> ElementCount {
    auto *Exiting = Plan.getVectorLoopRegion()->getExitingBasicBlock();
    if (match(&Exiting->back(),
              m_BranchOnCount(m_Add(m_CanonicalIV(), m_Specific(&Plan.getUF())),
                              m_VPValue())))
      return ElementCount::get(1, VF.isScalable());
    return VF;
  };

  // Check if the main loop processes fewer than MainLoopVF elements per
  // iteration (e.g. due to narrowing interleave groups). Adjust MainLoopVF
  // as needed.
  MainLoopVF = GetEffectiveVF(MainPlan, MainLoopVF);

  // If MainLoopVF = vscale x 2, and vscale is expected to be 4, then we know
  // the main loop handles 8 lanes per iteration. We could still benefit from
  // vectorizing the epilogue loop with VF=4.
  ElementCount EstimatedRuntimeVF = ElementCount::getFixed(
      estimateElementCount(MainLoopVF, Config.getVScaleForTuning()));

  Type *TCType = Legal->getWidestInductionType();
  const SCEV *RemainingIterations = nullptr;
  unsigned MaxTripCount = 0;
  const SCEV *TC = vputils::getSCEVExprForVPValue(MainPlan.getTripCount(), PSE);
  assert(!isa<SCEVCouldNotCompute>(TC) && "Trip count SCEV must be computable");
  const SCEV *KnownMinTC;
  bool ScalableTC = match(TC, m_scev_c_Mul(m_SCEV(KnownMinTC), m_SCEVVScale()));
  bool ScalableRemIter = false;
  ScalarEvolution &SE = *PSE.getSE();
  // Use versions of TC and VF in which both are either scalable or fixed.
  if (ScalableTC == MainLoopVF.isScalable()) {
    ScalableRemIter = ScalableTC;
    RemainingIterations =
        SE.getURemExpr(TC, SE.getElementCount(TCType, MainLoopVF * IC));
  } else if (ScalableTC) {
    const SCEV *EstimatedTC = SE.getMulExpr(
        KnownMinTC,
        SE.getConstant(TCType, Config.getVScaleForTuning().value_or(1)));
    RemainingIterations = SE.getURemExpr(
        EstimatedTC, SE.getElementCount(TCType, MainLoopVF * IC));
  } else
    RemainingIterations =
        SE.getURemExpr(TC, SE.getElementCount(TCType, EstimatedRuntimeVF * IC));

  // No iterations left to process in the epilogue.
  if (RemainingIterations->isZero())
    return nullptr;

  if (MainLoopVF.isFixed()) {
    MaxTripCount = MainLoopVF.getFixedValue() * IC - 1;
    if (SE.isKnownPredicate(CmpInst::ICMP_ULT, RemainingIterations,
                            SE.getConstant(TCType, MaxTripCount))) {
      MaxTripCount = SE.getUnsignedRangeMax(RemainingIterations).getZExtValue();
    }
    LLVM_DEBUG(dbgs() << "LEV: Maximum Trip Count for Epilogue: "
                      << MaxTripCount << "\n");
  }

  auto SkipVF = [&](const SCEV *VF, const SCEV *RemIter) -> bool {
    return SE.isKnownPredicate(CmpInst::ICMP_UGT, VF, RemIter);
  };
  VectorizationFactor Result = VectorizationFactor::Disabled();
  VPlan *BestPlan = nullptr;
  for (auto &NextVF : ProfitableVFs) {
    // Skip candidate VFs without a corresponding VPlan.
    if (!hasPlanWithVF(NextVF.Width))
      continue;

    VPlan &CurrentPlan = getPlanFor(NextVF.Width);
    ElementCount EffectiveVF = GetEffectiveVF(CurrentPlan, NextVF.Width);
    // Skip candidate VFs with widths >= the (estimated) runtime VF (scalable
    // vectors) or > the VF of the main loop (fixed vectors).
    if ((!EffectiveVF.isScalable() && MainLoopVF.isScalable() &&
         ElementCount::isKnownGE(EffectiveVF, EstimatedRuntimeVF)) ||
        (EffectiveVF.isScalable() &&
         ElementCount::isKnownGE(EffectiveVF, MainLoopVF)) ||
        (!EffectiveVF.isScalable() && !MainLoopVF.isScalable() &&
         ElementCount::isKnownGT(EffectiveVF, MainLoopVF)))
      continue;

    // If EffectiveVF is greater than the number of remaining iterations, the
    // epilogue loop would be dead. Skip such factors. If the epilogue plan
    // also has narrowed interleave groups, use the effective VF since
    // the epilogue step will be reduced to its IC.
    // TODO: We should also consider comparing against a scalable
    // RemainingIterations when SCEV be able to evaluate non-canonical
    // vscale-based expressions.
    if (!ScalableRemIter) {
      // Handle the case where EffectiveVF and RemainingIterations are in
      // different numerical spaces.
      if (EffectiveVF.isScalable())
        EffectiveVF = ElementCount::getFixed(
            estimateElementCount(EffectiveVF, Config.getVScaleForTuning()));
      if (SkipVF(SE.getElementCount(TCType, EffectiveVF), RemainingIterations))
        continue;
    }

    if (Result.Width.isScalar() ||
        isMoreProfitable(NextVF, Result, MaxTripCount, !CM.foldTailByMasking(),
                         /*IsEpilogue*/ true)) {
      Result = NextVF;
      BestPlan = &CurrentPlan;
    }
  }

  if (!BestPlan)
    return nullptr;

  LLVM_DEBUG(dbgs() << "LEV: Vectorizing epilogue loop with VF = "
                    << Result.Width << "\n");
  std::unique_ptr<VPlan> Clone(BestPlan->duplicate());
  Clone->setVF(Result.Width);
  return Clone;
}

unsigned
LoopVectorizationPlanner::selectInterleaveCount(VPlan &Plan, ElementCount VF,
                                                InstructionCost LoopCost) {
  // -- The interleave heuristics --
  // We interleave the loop in order to expose ILP and reduce the loop overhead.
  // There are many micro-architectural considerations that we can't predict
  // at this level. For example, frontend pressure (on decode or fetch) due to
  // code size, or the number and capabilities of the execution ports.
  //
  // We use the following heuristics to select the interleave count:
  // 1. If the code has reductions, then we interleave to break the cross
  // iteration dependency.
  // 2. If the loop is really small, then we interleave to reduce the loop
  // overhead.
  // 3. We don't interleave if we think that we will spill registers to memory
  // due to the increased register pressure.

  // Only interleave tail-folded loops if wide lane masks are requested, as the
  // overhead of multiple instructions to calculate the predicate is likely
  // not beneficial. If an epilogue is not allowed for any other reason,
  // do not interleave.
  if (!CM.isEpilogueAllowed() &&
      !(CM.preferTailFoldedLoop() && CM.useWideActiveLaneMask()))
    return 1;

  if (any_of(Plan.getVectorLoopRegion()->getEntryBasicBlock()->phis(),
             IsaPred<VPCurrentIterationPHIRecipe>)) {
    LLVM_DEBUG(dbgs() << "LV: Loop requires variable-length step. "
                         "Unroll factor forced to be 1.\n");
    return 1;
  }

  // We used the distance for the interleave count.
  if (!Legal->isSafeForAnyVectorWidth())
    return 1;

  // We don't attempt to perform interleaving for loops with uncountable early
  // exits because the VPInstruction::AnyOf code cannot currently handle
  // multiple parts.
  if (Plan.hasEarlyExit())
    return 1;

  const bool HasReductions =
      any_of(Plan.getVectorLoopRegion()->getEntryBasicBlock()->phis(),
             IsaPred<VPReductionPHIRecipe>);

  // FIXME: implement interleaving for FindLast transform correctly.
  if (hasFindLastReductionPhi(Plan))
    return 1;

  VPRegisterUsage R =
      calculateRegisterUsageForPlan(Plan, {VF}, TTI, CM.ValuesToIgnore)[0];

  // If we did not calculate the cost for VF (because the user selected the VF)
  // then we calculate the cost of VF here.
  if (LoopCost == 0) {
    if (VF.isScalar())
      LoopCost = CM.expectedCost(VF);
    else
      LoopCost = cost(Plan, VF, &R);
    assert(LoopCost.isValid() && "Expected to have chosen a VF with valid cost");

    // Loop body is free and there is no need for interleaving.
    if (LoopCost == 0)
      return 1;
  }

  // We divide by these constants so assume that we have at least one
  // instruction that uses at least one register.
  for (auto &Pair : R.MaxLocalUsers) {
    Pair.second = std::max(Pair.second, 1U);
  }

  // We calculate the interleave count using the following formula.
  // Subtract the number of loop invariants from the number of available
  // registers. These registers are used by all of the interleaved instances.
  // Next, divide the remaining registers by the number of registers that is
  // required by the loop, in order to estimate how many parallel instances
  // fit without causing spills. All of this is rounded down if necessary to be
  // a power of two. We want power of two interleave count to simplify any
  // addressing operations or alignment considerations.
  // We also want power of two interleave counts to ensure that the induction
  // variable of the vector loop wraps to zero, when tail is folded by masking;
  // this currently happens when OptForSize, in which case IC is set to 1 above.
  unsigned IC = UINT_MAX;

  for (const auto &Pair : R.MaxLocalUsers) {
    unsigned TargetNumRegisters = TTI.getNumberOfRegisters(Pair.first);
    LLVM_DEBUG(dbgs() << "LV: The target has " << TargetNumRegisters
                      << " registers of "
                      << TTI.getRegisterClassName(Pair.first)
                      << " register class\n");
    if (VF.isScalar()) {
      if (ForceTargetNumScalarRegs.getNumOccurrences() > 0)
        TargetNumRegisters = ForceTargetNumScalarRegs;
    } else {
      if (ForceTargetNumVectorRegs.getNumOccurrences() > 0)
        TargetNumRegisters = ForceTargetNumVectorRegs;
    }
    unsigned MaxLocalUsers = Pair.second;
    unsigned LoopInvariantRegs = 0;
    if (R.LoopInvariantRegs.contains(Pair.first))
      LoopInvariantRegs = R.LoopInvariantRegs[Pair.first];

    unsigned TmpIC = llvm::bit_floor((TargetNumRegisters - LoopInvariantRegs) /
                                     MaxLocalUsers);
    // Don't count the induction variable as interleaved.
    if (EnableIndVarRegisterHeur) {
      TmpIC = llvm::bit_floor((TargetNumRegisters - LoopInvariantRegs - 1) /
                              std::max(1U, (MaxLocalUsers - 1)));
    }

    IC = std::min(IC, TmpIC);
  }

  // Clamp the interleave ranges to reasonable counts.
  unsigned MaxInterleaveCount = TTI.getMaxInterleaveFactor(VF);
  LLVM_DEBUG(dbgs() << "LV: MaxInterleaveFactor for the target is "
                    << MaxInterleaveCount << "\n");

  // Check if the user has overridden the max.
  if (VF.isScalar()) {
    if (ForceTargetMaxScalarInterleaveFactor.getNumOccurrences() > 0)
      MaxInterleaveCount = ForceTargetMaxScalarInterleaveFactor;
  } else {
    if (ForceTargetMaxVectorInterleaveFactor.getNumOccurrences() > 0)
      MaxInterleaveCount = ForceTargetMaxVectorInterleaveFactor;
  }

  // Try to get the exact trip count, or an estimate based on profiling data or
  // ConstantMax from PSE, failing that.
  auto BestKnownTC =
      getSmallBestKnownTC(PSE, OrigLoop,
                          /*CanUseConstantMax=*/true,
                          /*CanExcludeZeroTrips=*/CM.isEpilogueAllowed());

  // For fixed length VFs treat a scalable trip count as unknown.
  if (BestKnownTC && (BestKnownTC->isFixed() || VF.isScalable())) {
    // Re-evaluate trip counts and VFs to be in the same numerical space.
    unsigned AvailableTC =
        estimateElementCount(*BestKnownTC, Config.getVScaleForTuning());
    unsigned EstimatedVF =
        estimateElementCount(VF, Config.getVScaleForTuning());

    // At least one iteration must be scalar when this constraint holds. So the
    // maximum available iterations for interleaving is one less.
    if (CM.requiresScalarEpilogue(VF.isVector()))
      --AvailableTC;

    unsigned InterleaveCountLB = bit_floor(std::max(
        1u, std::min(AvailableTC / (EstimatedVF * 2), MaxInterleaveCount)));

    if (getSmallConstantTripCount(PSE.getSE(), OrigLoop).isNonZero()) {
      // If the best known trip count is exact, we select between two
      // prospective ICs, where
      //
      // 1) the aggressive IC is capped by the trip count divided by VF
      // 2) the conservative IC is capped by the trip count divided by (VF * 2)
      //
      // The final IC is selected in a way that the epilogue loop trip count is
      // minimized while maximizing the IC itself, so that we either run the
      // vector loop at least once if it generates a small epilogue loop, or
      // else we run the vector loop at least twice.

      unsigned InterleaveCountUB = bit_floor(std::max(
          1u, std::min(AvailableTC / EstimatedVF, MaxInterleaveCount)));
      MaxInterleaveCount = InterleaveCountLB;

      if (InterleaveCountUB != InterleaveCountLB) {
        unsigned TailTripCountUB =
            (AvailableTC % (EstimatedVF * InterleaveCountUB));
        unsigned TailTripCountLB =
            (AvailableTC % (EstimatedVF * InterleaveCountLB));
        // If both produce same scalar tail, maximize the IC to do the same work
        // in fewer vector loop iterations
        if (TailTripCountUB == TailTripCountLB)
          MaxInterleaveCount = InterleaveCountUB;
      }
    } else {
      // If trip count is an estimated compile time constant, limit the
      // IC to be capped by the trip count divided by VF * 2, such that the
      // vector loop runs at least twice to make interleaving seem profitable
      // when there is an epilogue loop present. Since exact Trip count is not
      // known we choose to be conservative in our IC estimate.
      MaxInterleaveCount = InterleaveCountLB;
    }
  }

  assert(MaxInterleaveCount > 0 &&
         "Maximum interleave count must be greater than 0");

  // Clamp the calculated IC to be between the 1 and the max interleave count
  // that the target and trip count allows.
  if (IC > MaxInterleaveCount)
    IC = MaxInterleaveCount;
  else
    // Make sure IC is greater than 0.
    IC = std::max(1u, IC);

  assert(IC > 0 && "Interleave count must be greater than 0.");

  // Interleave if we vectorized this loop and there is a reduction that could
  // benefit from interleaving.
  if (VF.isVector() && HasReductions) {
    LLVM_DEBUG(dbgs() << "LV: Interleaving because of reductions.\n");
    return IC;
  }

  // For any scalar loop that either requires runtime checks or tail-folding we
  // are better off leaving this to the unroller. Note that if we've already
  // vectorized the loop we will have done the runtime check and so interleaving
  // won't require further checks.
  bool ScalarInterleavingRequiresPredication =
      (VF.isScalar() && any_of(OrigLoop->blocks(), [this](BasicBlock *BB) {
         return Legal->blockNeedsPredication(BB);
       }));
  bool ScalarInterleavingRequiresRuntimePointerCheck =
      (VF.isScalar() && Legal->getRuntimePointerChecking()->Need);

  // We want to interleave small loops in order to reduce the loop overhead and
  // potentially expose ILP opportunities.
  LLVM_DEBUG(dbgs() << "LV: Loop cost is " << LoopCost << '\n'
                    << "LV: IC is " << IC << '\n'
                    << "LV: VF is " << VF << '\n');
  const bool AggressivelyInterleave =
      TTI.enableAggressiveInterleaving(HasReductions);
  if (!ScalarInterleavingRequiresRuntimePointerCheck &&
      !ScalarInterleavingRequiresPredication && LoopCost < SmallLoopCost) {
    // We assume that the cost overhead is 1 and we use the cost model
    // to estimate the cost of the loop and interleave until the cost of the
    // loop overhead is about 5% of the cost of the loop.
    unsigned SmallIC = std::min(IC, (unsigned)llvm::bit_floor<uint64_t>(
                                        SmallLoopCost / LoopCost.getValue()));

    // Interleave until store/load ports (estimated by max interleave count) are
    // saturated.
    unsigned NumStores = 0;
    unsigned NumLoads = 0;
    for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
             vp_depth_first_deep(Plan.getVectorLoopRegion()->getEntry()))) {
      for (VPRecipeBase &R : *VPBB) {
        if (isa<VPWidenLoadRecipe, VPWidenLoadEVLRecipe>(&R)) {
          NumLoads++;
          continue;
        }
        if (isa<VPWidenStoreRecipe, VPWidenStoreEVLRecipe>(&R)) {
          NumStores++;
          continue;
        }

        if (auto *InterleaveR = dyn_cast<VPInterleaveRecipe>(&R)) {
          if (unsigned StoreOps = InterleaveR->getNumStoreOperands())
            NumStores += StoreOps;
          else
            NumLoads += InterleaveR->getNumDefinedValues();
          continue;
        }
        if (auto *RepR = dyn_cast<VPReplicateRecipe>(&R)) {
          NumLoads += isa<LoadInst>(RepR->getUnderlyingInstr());
          NumStores += isa<StoreInst>(RepR->getUnderlyingInstr());
          continue;
        }
        if (isa<VPHistogramRecipe>(&R)) {
          NumLoads++;
          NumStores++;
          continue;
        }
      }
    }
    unsigned StoresIC = IC / (NumStores ? NumStores : 1);
    unsigned LoadsIC = IC / (NumLoads ? NumLoads : 1);

    // There is little point in interleaving for reductions containing selects
    // and compares when VF=1 since it may just create more overhead than it's
    // worth for loops with small trip counts. This is because we still have to
    // do the final reduction after the loop.
    bool HasSelectCmpReductions =
        HasReductions &&
        any_of(Plan.getVectorLoopRegion()->getEntryBasicBlock()->phis(),
               [](VPRecipeBase &R) {
                 auto *RedR = dyn_cast<VPReductionPHIRecipe>(&R);
                 return RedR && (RecurrenceDescriptor::isAnyOfRecurrenceKind(
                                     RedR->getRecurrenceKind()) ||
                                 RecurrenceDescriptor::isFindIVRecurrenceKind(
                                     RedR->getRecurrenceKind()));
               });
    if (HasSelectCmpReductions) {
      LLVM_DEBUG(dbgs() << "LV: Not interleaving select-cmp reductions.\n");
      return 1;
    }

    // If we have a scalar reduction (vector reductions are already dealt with
    // by this point), we can increase the critical path length if the loop
    // we're interleaving is inside another loop. For tree-wise reductions
    // set the limit to 2, and for ordered reductions it's best to disable
    // interleaving entirely.
    if (HasReductions && OrigLoop->getLoopDepth() > 1) {
      bool HasOrderedReductions =
          any_of(Plan.getVectorLoopRegion()->getEntryBasicBlock()->phis(),
                 [](VPRecipeBase &R) {
                   auto *RedR = dyn_cast<VPReductionPHIRecipe>(&R);

                   return RedR && RedR->isOrdered();
                 });
      if (HasOrderedReductions) {
        LLVM_DEBUG(
            dbgs() << "LV: Not interleaving scalar ordered reductions.\n");
        return 1;
      }

      unsigned F = MaxNestedScalarReductionIC;
      SmallIC = std::min(SmallIC, F);
      StoresIC = std::min(StoresIC, F);
      LoadsIC = std::min(LoadsIC, F);
    }

    if (EnableLoadStoreRuntimeInterleave &&
        std::max(StoresIC, LoadsIC) > SmallIC) {
      LLVM_DEBUG(
          dbgs() << "LV: Interleaving to saturate store or load ports.\n");
      return std::max(StoresIC, LoadsIC);
    }

    // If there are scalar reductions and TTI has enabled aggressive
    // interleaving for reductions, we will interleave to expose ILP.
    if (VF.isScalar() && AggressivelyInterleave) {
      LLVM_DEBUG(dbgs() << "LV: Interleaving to expose ILP.\n");
      // Interleave no less than SmallIC but not as aggressive as the normal IC
      // to satisfy the rare situation when resources are too limited.
      return std::max(IC / 2, SmallIC);
    }

    LLVM_DEBUG(dbgs() << "LV: Interleaving to reduce branch cost.\n");
    return SmallIC;
  }

  // Interleave if this is a large loop (small loops are already dealt with by
  // this point) that could benefit from interleaving.
  if (AggressivelyInterleave) {
    LLVM_DEBUG(dbgs() << "LV: Interleaving to expose ILP.\n");
    return IC;
  }

  LLVM_DEBUG(dbgs() << "LV: Not Interleaving.\n");
  return 1;
}

bool LoopVectorizationCostModel::useEmulatedMaskMemRefHack(Instruction *I,
                                                           ElementCount VF) {
  // TODO: Cost model for emulated masked load/store is completely
  // broken. This hack guides the cost model to use an artificially
  // high enough value to practically disable vectorization with such
  // operations, except where previously deployed legality hack allowed
  // using very low cost values. This is to avoid regressions coming simply
  // from moving "masked load/store" check from legality to cost model.
  // Masked Load/Gather emulation was previously never allowed.
  // Limited number of Masked Store/Scatter emulation was allowed.
  assert((isPredicatedInst(I)) &&
         "Expecting a scalar emulated instruction");
  return isa<LoadInst>(I) ||
         (isa<StoreInst>(I) &&
          NumPredStores > NumberOfStoresToPredicate);
}

void LoopVectorizationCostModel::collectInstsToScalarize(ElementCount VF) {
  assert(VF.isVector() && "Expected VF >= 2");

  // If we've already collected the instructions to scalarize or the predicated
  // BBs after vectorization, there's nothing to do. Collection may already have
  // occurred if we have a user-selected VF and are now computing the expected
  // cost for interleaving.
  if (InstsToScalarize.contains(VF) ||
      PredicatedBBsAfterVectorization.contains(VF))
    return;

  // Initialize a mapping for VF in InstsToScalalarize. If we find that it's
  // not profitable to scalarize any instructions, the presence of VF in the
  // map will indicate that we've analyzed it already.
  ScalarCostsTy &ScalarCostsVF = InstsToScalarize[VF];

  // Find all the instructions that are scalar with predication in the loop and
  // determine if it would be better to not if-convert the blocks they are in.
  // If so, we also record the instructions to scalarize.
  for (BasicBlock *BB : TheLoop->blocks()) {
    if (!blockNeedsPredicationForAnyReason(BB))
      continue;
    for (Instruction &I : *BB)
      if (isScalarWithPredication(&I, VF)) {
        ScalarCostsTy ScalarCosts;
        // Do not apply discount logic for:
        // 1. Scalars after vectorization, as there will only be a single copy
        // of the instruction.
        // 2. Scalable VF, as that would lead to invalid scalarization costs.
        // 3. Emulated masked memrefs, if a hacked cost is needed.
        if (!isScalarAfterVectorization(&I, VF) && !VF.isScalable() &&
            !useEmulatedMaskMemRefHack(&I, VF) &&
            computePredInstDiscount(&I, ScalarCosts, VF) >= 0) {
          for (const auto &[I, IC] : ScalarCosts)
            ScalarCostsVF.insert({I, IC});
          // Check if we decided to scalarize a call. If so, update the widening
          // decision of the call to CM_Scalarize with the computed scalar cost.
          for (const auto &[I, Cost] : ScalarCosts) {
            auto *CI = dyn_cast<CallInst>(I);
            if (!CI || !CallWideningDecisions.contains({CI, VF}))
              continue;
            CallWideningDecisions[{CI, VF}].Kind = CM_Scalarize;
            CallWideningDecisions[{CI, VF}].Cost = Cost;
          }
        }
        // Remember that BB will remain after vectorization.
        PredicatedBBsAfterVectorization[VF].insert(BB);
        for (auto *Pred : predecessors(BB)) {
          if (Pred->getSingleSuccessor() == BB)
            PredicatedBBsAfterVectorization[VF].insert(Pred);
        }
      }
  }
}

InstructionCost LoopVectorizationCostModel::computePredInstDiscount(
    Instruction *PredInst, ScalarCostsTy &ScalarCosts, ElementCount VF) {
  assert(!isUniformAfterVectorization(PredInst, VF) &&
         "Instruction marked uniform-after-vectorization will be predicated");

  // Initialize the discount to zero, meaning that the scalar version and the
  // vector version cost the same.
  InstructionCost Discount = 0;

  // Holds instructions to analyze. The instructions we visit are mapped in
  // ScalarCosts. Those instructions are the ones that would be scalarized if
  // we find that the scalar version costs less.
  SmallVector<Instruction *, 8> Worklist;

  // Returns true if the given instruction can be scalarized.
  auto CanBeScalarized = [&](Instruction *I) -> bool {
    // We only attempt to scalarize instructions forming a single-use chain
    // from the original predicated block that would otherwise be vectorized.
    // Although not strictly necessary, we give up on instructions we know will
    // already be scalar to avoid traversing chains that are unlikely to be
    // beneficial.
    if (!I->hasOneUse() || PredInst->getParent() != I->getParent() ||
        isScalarAfterVectorization(I, VF))
      return false;

    // If the instruction is scalar with predication, it will be analyzed
    // separately. We ignore it within the context of PredInst.
    if (isScalarWithPredication(I, VF))
      return false;

    // If any of the instruction's operands are uniform after vectorization,
    // the instruction cannot be scalarized. This prevents, for example, a
    // masked load from being scalarized.
    //
    // We assume we will only emit a value for lane zero of an instruction
    // marked uniform after vectorization, rather than VF identical values.
    // Thus, if we scalarize an instruction that uses a uniform, we would
    // create uses of values corresponding to the lanes we aren't emitting code
    // for. This behavior can be changed by allowing getScalarValue to clone
    // the lane zero values for uniforms rather than asserting.
    for (Use &U : I->operands())
      if (auto *J = dyn_cast<Instruction>(U.get()))
        if (isUniformAfterVectorization(J, VF))
          return false;

    // Otherwise, we can scalarize the instruction.
    return true;
  };

  // Compute the expected cost discount from scalarizing the entire expression
  // feeding the predicated instruction. We currently only consider expressions
  // that are single-use instruction chains.
  Worklist.push_back(PredInst);
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();

    // If we've already analyzed the instruction, there's nothing to do.
    if (ScalarCosts.contains(I))
      continue;

    // Cannot scalarize fixed-order recurrence phis at the moment.
    if (isa<PHINode>(I) && Legal->isFixedOrderRecurrence(cast<PHINode>(I)))
      continue;

    // Compute the cost of the vector instruction. Note that this cost already
    // includes the scalarization overhead of the predicated instruction.
    InstructionCost VectorCost = getInstructionCost(I, VF);

    // Compute the cost of the scalarized instruction. This cost is the cost of
    // the instruction as if it wasn't if-converted and instead remained in the
    // predicated block. We will scale this cost by block probability after
    // computing the scalarization overhead.
    InstructionCost ScalarCost =
        VF.getFixedValue() * getInstructionCost(I, ElementCount::getFixed(1));

    // Compute the scalarization overhead of needed insertelement instructions
    // and phi nodes.
    if (isScalarWithPredication(I, VF) && !I->getType()->isVoidTy()) {
      Type *WideTy = toVectorizedTy(I->getType(), VF);
      for (Type *VectorTy : getContainedTypes(WideTy)) {
        ScalarCost += TTI.getScalarizationOverhead(
            cast<VectorType>(VectorTy), APInt::getAllOnes(VF.getFixedValue()),
            /*Insert=*/true,
            /*Extract=*/false, Config.CostKind);
      }
      ScalarCost += VF.getFixedValue() *
                    TTI.getCFInstrCost(Instruction::PHI, Config.CostKind);
    }

    // Compute the scalarization overhead of needed extractelement
    // instructions. For each of the instruction's operands, if the operand can
    // be scalarized, add it to the worklist; otherwise, account for the
    // overhead.
    for (Use &U : I->operands())
      if (auto *J = dyn_cast<Instruction>(U.get())) {
        assert(canVectorizeTy(J->getType()) &&
               "Instruction has non-scalar type");
        if (CanBeScalarized(J))
          Worklist.push_back(J);
        else if (needsExtract(J, VF)) {
          Type *WideTy = toVectorizedTy(J->getType(), VF);
          for (Type *VectorTy : getContainedTypes(WideTy)) {
            ScalarCost += TTI.getScalarizationOverhead(
                cast<VectorType>(VectorTy),
                APInt::getAllOnes(VF.getFixedValue()), /*Insert*/ false,
                /*Extract*/ true, Config.CostKind);
          }
        }
      }

    // Scale the total scalar cost by block probability.
    ScalarCost /= getPredBlockCostDivisor(Config.CostKind, I->getParent());

    // Compute the discount. A non-negative discount means the vector version
    // of the instruction costs more, and scalarizing would be beneficial.
    Discount += VectorCost - ScalarCost;
    ScalarCosts[I] = ScalarCost;
  }

  return Discount;
}

InstructionCost LoopVectorizationCostModel::expectedCost(ElementCount VF) {
  InstructionCost Cost;
  assert(VF.isScalar() && "must only be called for scalar VFs");

  // For each block.
  for (BasicBlock *BB : TheLoop->blocks()) {
    InstructionCost BlockCost;

    // For each instruction in the old loop.
    for (Instruction &I : *BB) {
      // Skip ignored values.
      if (ValuesToIgnore.count(&I) ||
          (VF.isVector() && VecValuesToIgnore.count(&I)))
        continue;

      InstructionCost C = getInstructionCost(&I, VF);

      // Check if we should override the cost.
      if (C.isValid() && ForceTargetInstructionCost.getNumOccurrences() > 0)
        C = InstructionCost(ForceTargetInstructionCost);

      BlockCost += C;
      LLVM_DEBUG(dbgs() << "LV: Found an estimated cost of " << C << " for VF "
                        << VF << " For instruction: " << I << '\n');
    }

    // In the scalar loop, we may not always execute the predicated block, if it
    // is an if-else block. Thus, scale the block's cost by the probability of
    // executing it. getPredBlockCostDivisor will return 1 for blocks that are
    // only predicated by the header mask when folding the tail.
    Cost += BlockCost / getPredBlockCostDivisor(Config.CostKind, BB);
  }

  return Cost;
}

/// Gets the address access SCEV for Ptr, if it should be used for cost modeling
/// according to isAddressSCEVForCost.
///
/// This SCEV can be sent to the Target in order to estimate the address
/// calculation cost.
static const SCEV *getAddressAccessSCEV(
              Value *Ptr,
              PredicatedScalarEvolution &PSE,
              const Loop *TheLoop) {
  const SCEV *Addr = PSE.getSCEV(Ptr);
  return vputils::isAddressSCEVForCost(Addr, *PSE.getSE(), TheLoop) ? Addr
                                                                    : nullptr;
}

InstructionCost
LoopVectorizationCostModel::getMemInstScalarizationCost(Instruction *I,
                                                        ElementCount VF) {
  assert(VF.isVector() &&
         "Scalarization cost of instruction implies vectorization.");
  if (VF.isScalable())
    return InstructionCost::getInvalid();

  Type *ValTy = getLoadStoreType(I);
  auto *SE = PSE.getSE();

  unsigned AS = getLoadStoreAddressSpace(I);
  Value *Ptr = getLoadStorePointerOperand(I);
  Type *PtrTy = toVectorTy(Ptr->getType(), VF);
  // NOTE: PtrTy is a vector to signal `TTI::getAddressComputationCost`
  //       that it is being called from this specific place.

  // Figure out whether the access is strided and get the stride value
  // if it's known in compile time
  const SCEV *PtrSCEV = getAddressAccessSCEV(Ptr, PSE, TheLoop);

  // Get the cost of the scalar memory instruction and address computation.
  InstructionCost Cost =
      VF.getFixedValue() *
      TTI.getAddressComputationCost(PtrTy, SE, PtrSCEV, Config.CostKind);

  // Don't pass *I here, since it is scalar but will actually be part of a
  // vectorized loop where the user of it is a vectorized instruction.
  const Align Alignment = getLoadStoreAlignment(I);
  TTI::OperandValueInfo OpInfo = TTI::getOperandInfo(I->getOperand(0));
  Cost += VF.getFixedValue() *
          TTI.getMemoryOpCost(I->getOpcode(), ValTy->getScalarType(), Alignment,
                              AS, Config.CostKind, OpInfo);

  // Get the overhead of the extractelement and insertelement instructions
  // we might create due to scalarization.
  Cost += getScalarizationOverhead(I, VF);

  // If we have a predicated load/store, it will need extra i1 extracts and
  // conditional branches, but may not be executed for each vector lane. Scale
  // the cost by the probability of executing the predicated block.
  if (isPredicatedInst(I)) {
    Cost /= getPredBlockCostDivisor(Config.CostKind, I->getParent());

    // Add the cost of an i1 extract and a branch
    auto *VecI1Ty =
        VectorType::get(IntegerType::getInt1Ty(ValTy->getContext()), VF);
    Cost += TTI.getScalarizationOverhead(
        VecI1Ty, APInt::getAllOnes(VF.getFixedValue()),
        /*Insert=*/false, /*Extract=*/true, Config.CostKind);
    Cost += TTI.getCFInstrCost(Instruction::CondBr, Config.CostKind);

    if (useEmulatedMaskMemRefHack(I, VF))
      // Artificially setting to a high enough value to practically disable
      // vectorization with such operations.
      Cost = 3000000;
  }

  return Cost;
}

InstructionCost
LoopVectorizationCostModel::getConsecutiveMemOpCost(Instruction *I,
                                                    ElementCount VF) {
  Type *ValTy = getLoadStoreType(I);
  auto *VectorTy = cast<VectorType>(toVectorTy(ValTy, VF));
  Value *Ptr = getLoadStorePointerOperand(I);
  unsigned AS = getLoadStoreAddressSpace(I);
  int ConsecutiveStride = Legal->isConsecutivePtr(ValTy, Ptr);

  assert((ConsecutiveStride == 1 || ConsecutiveStride == -1) &&
         "Stride should be 1 or -1 for consecutive memory access");
  const Align Alignment = getLoadStoreAlignment(I);
  InstructionCost Cost = 0;
  if (isMaskRequired(I)) {
    unsigned IID = I->getOpcode() == Instruction::Load
                       ? Intrinsic::masked_load
                       : Intrinsic::masked_store;
    Cost += TTI.getMemIntrinsicInstrCost(
        MemIntrinsicCostAttributes(IID, VectorTy, Alignment, AS),
        Config.CostKind);
  } else {
    TTI::OperandValueInfo OpInfo = TTI::getOperandInfo(I->getOperand(0));
    Cost += TTI.getMemoryOpCost(I->getOpcode(), VectorTy, Alignment, AS,
                                Config.CostKind, OpInfo, I);
  }

  bool Reverse = ConsecutiveStride < 0;
  if (Reverse)
    Cost += TTI.getShuffleCost(TargetTransformInfo::SK_Reverse, VectorTy,
                               VectorTy, {}, Config.CostKind, 0);
  return Cost;
}

InstructionCost
LoopVectorizationCostModel::getUniformMemOpCost(Instruction *I,
                                                ElementCount VF) {
  assert(Legal->isUniformMemOp(*I, VF));

  Type *ValTy = getLoadStoreType(I);
  Type *PtrTy = getLoadStorePointerOperand(I)->getType();
  auto *VectorTy = cast<VectorType>(toVectorTy(ValTy, VF));
  const Align Alignment = getLoadStoreAlignment(I);
  unsigned AS = getLoadStoreAddressSpace(I);
  if (isa<LoadInst>(I)) {
    return TTI.getAddressComputationCost(PtrTy, nullptr, nullptr,
                                         Config.CostKind) +
           TTI.getMemoryOpCost(Instruction::Load, ValTy, Alignment, AS,
                               Config.CostKind) +
           TTI.getShuffleCost(TargetTransformInfo::SK_Broadcast, VectorTy,
                              VectorTy, {}, Config.CostKind);
  }
  StoreInst *SI = cast<StoreInst>(I);

  bool IsLoopInvariantStoreValue = Legal->isInvariant(SI->getValueOperand());
  // TODO: We have existing tests that request the cost of extracting element
  // VF.getKnownMinValue() - 1 from a scalable vector. This does not represent
  // the actual generated code, which involves extracting the last element of
  // a scalable vector where the lane to extract is unknown at compile time.
  InstructionCost Cost =
      TTI.getAddressComputationCost(PtrTy, nullptr, nullptr, Config.CostKind) +
      TTI.getMemoryOpCost(Instruction::Store, ValTy, Alignment, AS,
                          Config.CostKind);
  if (!IsLoopInvariantStoreValue)
    Cost += TTI.getIndexedVectorInstrCostFromEnd(Instruction::ExtractElement,
                                                 VectorTy, Config.CostKind, 0);
  return Cost;
}

InstructionCost
LoopVectorizationCostModel::getGatherScatterCost(Instruction *I,
                                                 ElementCount VF) {
  Type *ValTy = getLoadStoreType(I);
  auto *VectorTy = cast<VectorType>(toVectorTy(ValTy, VF));
  const Align Alignment = getLoadStoreAlignment(I);
  Value *Ptr = getLoadStorePointerOperand(I);
  Type *PtrTy = Ptr->getType();

  if (!Legal->isUniform(Ptr, VF))
    PtrTy = toVectorTy(PtrTy, VF);

  unsigned IID = I->getOpcode() == Instruction::Load
                     ? Intrinsic::masked_gather
                     : Intrinsic::masked_scatter;
  return TTI.getAddressComputationCost(PtrTy, nullptr, nullptr,
                                       Config.CostKind) +
         TTI.getMemIntrinsicInstrCost(
             MemIntrinsicCostAttributes(IID, VectorTy, Ptr, isMaskRequired(I),
                                        Alignment, I),
             Config.CostKind);
}

InstructionCost
LoopVectorizationCostModel::getInterleaveGroupCost(Instruction *I,
                                                   ElementCount VF) {
  const auto *Group = getInterleavedAccessGroup(I);
  assert(Group && "Fail to get an interleaved access group.");

  Instruction *InsertPos = Group->getInsertPos();
  Type *ValTy = getLoadStoreType(InsertPos);
  auto *VectorTy = cast<VectorType>(toVectorTy(ValTy, VF));
  unsigned AS = getLoadStoreAddressSpace(InsertPos);

  unsigned InterleaveFactor = Group->getFactor();
  auto *WideVecTy = VectorType::get(ValTy, VF * InterleaveFactor);

  // Holds the indices of existing members in the interleaved group.
  SmallVector<unsigned, 4> Indices;
  for (unsigned IF = 0; IF < InterleaveFactor; IF++)
    if (Group->getMember(IF))
      Indices.push_back(IF);

  // Calculate the cost of the whole interleaved group.
  bool UseMaskForGaps =
      (Group->requiresScalarEpilogue() && !isEpilogueAllowed()) ||
      (isa<StoreInst>(I) && !Group->isFull());
  InstructionCost Cost = TTI.getInterleavedMemoryOpCost(
      InsertPos->getOpcode(), WideVecTy, Group->getFactor(), Indices,
      Group->getAlign(), AS, Config.CostKind, isMaskRequired(I),
      UseMaskForGaps);

  if (Group->isReverse()) {
    // TODO: Add support for reversed masked interleaved access.
    assert(!isMaskRequired(I) &&
           "Reverse masked interleaved access not supported.");
    Cost += Group->getNumMembers() *
            TTI.getShuffleCost(TargetTransformInfo::SK_Reverse, VectorTy,
                               VectorTy, {}, Config.CostKind, 0);
  }
  return Cost;
}

std::optional<InstructionCost>
LoopVectorizationCostModel::getReductionPatternCost(Instruction *I,
                                                    ElementCount VF,
                                                    Type *Ty) const {
  using namespace llvm::PatternMatch;
  // Early exit for no inloop reductions
  if (Config.getInLoopReductions().empty() || VF.isScalar() ||
      !isa<VectorType>(Ty))
    return std::nullopt;
  auto *VectorTy = cast<VectorType>(Ty);

  // We are looking for a pattern of, and finding the minimal acceptable cost:
  //  reduce(mul(ext(A), ext(B))) or
  //  reduce(mul(A, B)) or
  //  reduce(ext(A)) or
  //  reduce(A).
  // The basic idea is that we walk down the tree to do that, finding the root
  // reduction instruction in InLoopReductionImmediateChains. From there we find
  // the pattern of mul/ext and test the cost of the entire pattern vs the cost
  // of the components. If the reduction cost is lower then we return it for the
  // reduction instruction and 0 for the other instructions in the pattern. If
  // it is not we return an invalid cost specifying the orignal cost method
  // should be used.
  Instruction *RetI = I;
  if (match(RetI, m_ZExtOrSExt(m_Value()))) {
    if (!RetI->hasOneUser())
      return std::nullopt;
    RetI = RetI->user_back();
  }

  if (match(RetI, m_OneUse(m_Mul(m_Value(), m_Value()))) &&
      RetI->user_back()->getOpcode() == Instruction::Add) {
    RetI = RetI->user_back();
  }

  // Test if the found instruction is a reduction, and if not return an invalid
  // cost specifying the parent to use the original cost modelling.
  Instruction *LastChain = Config.getInLoopReductionImmediateChain(RetI);
  if (!LastChain)
    return std::nullopt;

  // Find the reduction this chain is a part of and calculate the basic cost of
  // the reduction on its own.
  Instruction *ReductionPhi = LastChain;
  while (!isa<PHINode>(ReductionPhi))
    ReductionPhi = Config.getInLoopReductionImmediateChain(ReductionPhi);

  const RecurrenceDescriptor &RdxDesc =
      Legal->getRecurrenceDescriptor(cast<PHINode>(ReductionPhi));

  InstructionCost BaseCost;
  RecurKind RK = RdxDesc.getRecurrenceKind();
  if (RecurrenceDescriptor::isMinMaxRecurrenceKind(RK)) {
    Intrinsic::ID MinMaxID = getMinMaxReductionIntrinsicOp(RK);
    BaseCost = TTI.getMinMaxReductionCost(
        MinMaxID, VectorTy, RdxDesc.getFastMathFlags(), Config.CostKind);
  } else {
    BaseCost = TTI.getArithmeticReductionCost(RdxDesc.getOpcode(), VectorTy,
                                              RdxDesc.getFastMathFlags(),
                                              Config.CostKind);
  }

  // For a call to the llvm.fmuladd intrinsic we need to add the cost of a
  // normal fmul instruction to the cost of the fadd reduction.
  if (RK == RecurKind::FMulAdd)
    BaseCost += TTI.getArithmeticInstrCost(Instruction::FMul, VectorTy,
                                           Config.CostKind);

  // If we're using ordered reductions then we can just return the base cost
  // here, since getArithmeticReductionCost calculates the full ordered
  // reduction cost when FP reassociation is not allowed.
  if (Config.useOrderedReductions(RdxDesc))
    return BaseCost;

  // Get the operand that was not the reduction chain and match it to one of the
  // patterns, returning the better cost if it is found.
  Instruction *RedOp = RetI->getOperand(1) == LastChain
                           ? dyn_cast<Instruction>(RetI->getOperand(0))
                           : dyn_cast<Instruction>(RetI->getOperand(1));

  VectorTy = VectorType::get(I->getOperand(0)->getType(), VectorTy);

  Instruction *Op0, *Op1;
  if (RedOp && RdxDesc.getOpcode() == Instruction::Add &&
      match(RedOp,
            m_ZExtOrSExt(m_Mul(m_Instruction(Op0), m_Instruction(Op1)))) &&
      match(Op0, m_ZExtOrSExt(m_Value())) &&
      Op0->getOpcode() == Op1->getOpcode() &&
      Op0->getOperand(0)->getType() == Op1->getOperand(0)->getType() &&
      !TheLoop->isLoopInvariant(Op0) && !TheLoop->isLoopInvariant(Op1) &&
      (Op0->getOpcode() == RedOp->getOpcode() || Op0 == Op1)) {

    // Matched reduce.add(ext(mul(ext(A), ext(B)))
    // Note that the extend opcodes need to all match, or if A==B they will have
    // been converted to zext(mul(sext(A), sext(A))) as it is known positive,
    // which is equally fine.
    bool IsUnsigned = isa<ZExtInst>(Op0);
    auto *ExtType = VectorType::get(Op0->getOperand(0)->getType(), VectorTy);
    auto *MulType = VectorType::get(Op0->getType(), VectorTy);

    InstructionCost ExtCost =
        TTI.getCastInstrCost(Op0->getOpcode(), MulType, ExtType,
                             TTI::CastContextHint::None, Config.CostKind, Op0);
    InstructionCost MulCost =
        TTI.getArithmeticInstrCost(Instruction::Mul, MulType, Config.CostKind);
    InstructionCost Ext2Cost = TTI.getCastInstrCost(
        RedOp->getOpcode(), VectorTy, MulType, TTI::CastContextHint::None,
        Config.CostKind, RedOp);

    InstructionCost RedCost = TTI.getMulAccReductionCost(
        IsUnsigned, RdxDesc.getOpcode(), RdxDesc.getRecurrenceType(), ExtType,
        Config.CostKind);

    if (RedCost.isValid() &&
        RedCost < ExtCost * 2 + MulCost + Ext2Cost + BaseCost)
      return I == RetI ? RedCost : 0;
  } else if (RedOp && match(RedOp, m_ZExtOrSExt(m_Value())) &&
             !TheLoop->isLoopInvariant(RedOp)) {
    // Matched reduce(ext(A))
    bool IsUnsigned = isa<ZExtInst>(RedOp);
    auto *ExtType = VectorType::get(RedOp->getOperand(0)->getType(), VectorTy);
    InstructionCost RedCost = TTI.getExtendedReductionCost(
        RdxDesc.getOpcode(), IsUnsigned, RdxDesc.getRecurrenceType(), ExtType,
        RdxDesc.getFastMathFlags(), Config.CostKind);

    InstructionCost ExtCost = TTI.getCastInstrCost(
        RedOp->getOpcode(), VectorTy, ExtType, TTI::CastContextHint::None,
        Config.CostKind, RedOp);
    if (RedCost.isValid() && RedCost < BaseCost + ExtCost)
      return I == RetI ? RedCost : 0;
  } else if (RedOp && RdxDesc.getOpcode() == Instruction::Add &&
             match(RedOp, m_Mul(m_Instruction(Op0), m_Instruction(Op1)))) {
    if (match(Op0, m_ZExtOrSExt(m_Value())) &&
        Op0->getOpcode() == Op1->getOpcode() &&
        !TheLoop->isLoopInvariant(Op0) && !TheLoop->isLoopInvariant(Op1)) {
      bool IsUnsigned = isa<ZExtInst>(Op0);
      Type *Op0Ty = Op0->getOperand(0)->getType();
      Type *Op1Ty = Op1->getOperand(0)->getType();
      Type *LargestOpTy =
          Op0Ty->getIntegerBitWidth() < Op1Ty->getIntegerBitWidth() ? Op1Ty
                                                                    : Op0Ty;
      auto *ExtType = VectorType::get(LargestOpTy, VectorTy);

      // Matched reduce.add(mul(ext(A), ext(B))), where the two ext may be of
      // different sizes. We take the largest type as the ext to reduce, and add
      // the remaining cost as, for example reduce(mul(ext(ext(A)), ext(B))).
      InstructionCost ExtCost0 = TTI.getCastInstrCost(
          Op0->getOpcode(), VectorTy, VectorType::get(Op0Ty, VectorTy),
          TTI::CastContextHint::None, Config.CostKind, Op0);
      InstructionCost ExtCost1 = TTI.getCastInstrCost(
          Op1->getOpcode(), VectorTy, VectorType::get(Op1Ty, VectorTy),
          TTI::CastContextHint::None, Config.CostKind, Op1);
      InstructionCost MulCost = TTI.getArithmeticInstrCost(
          Instruction::Mul, VectorTy, Config.CostKind);

      InstructionCost RedCost = TTI.getMulAccReductionCost(
          IsUnsigned, RdxDesc.getOpcode(), RdxDesc.getRecurrenceType(), ExtType,
          Config.CostKind);
      InstructionCost ExtraExtCost = 0;
      if (Op0Ty != LargestOpTy || Op1Ty != LargestOpTy) {
        Instruction *ExtraExtOp = (Op0Ty != LargestOpTy) ? Op0 : Op1;
        ExtraExtCost = TTI.getCastInstrCost(
            ExtraExtOp->getOpcode(), ExtType,
            VectorType::get(ExtraExtOp->getOperand(0)->getType(), VectorTy),
            TTI::CastContextHint::None, Config.CostKind, ExtraExtOp);
      }

      if (RedCost.isValid() &&
          (RedCost + ExtraExtCost) < (ExtCost0 + ExtCost1 + MulCost + BaseCost))
        return I == RetI ? RedCost : 0;
    } else if (!match(I, m_ZExtOrSExt(m_Value()))) {
      // Matched reduce.add(mul())
      InstructionCost MulCost = TTI.getArithmeticInstrCost(
          Instruction::Mul, VectorTy, Config.CostKind);

      InstructionCost RedCost = TTI.getMulAccReductionCost(
          true, RdxDesc.getOpcode(), RdxDesc.getRecurrenceType(), VectorTy,
          Config.CostKind);

      if (RedCost.isValid() && RedCost < MulCost + BaseCost)
        return I == RetI ? RedCost : 0;
    }
  }

  return I == RetI ? std::optional<InstructionCost>(BaseCost) : std::nullopt;
}

InstructionCost
LoopVectorizationCostModel::getMemoryInstructionCost(Instruction *I,
                                                     ElementCount VF) {
  // Calculate scalar cost only. Vectorization cost should be ready at this
  // moment.
  if (VF.isScalar()) {
    Type *ValTy = getLoadStoreType(I);
    Type *PtrTy = getLoadStorePointerOperand(I)->getType();
    const Align Alignment = getLoadStoreAlignment(I);
    unsigned AS = getLoadStoreAddressSpace(I);

    TTI::OperandValueInfo OpInfo = TTI::getOperandInfo(I->getOperand(0));
    return TTI.getAddressComputationCost(PtrTy, nullptr, nullptr,
                                         Config.CostKind) +
           TTI.getMemoryOpCost(I->getOpcode(), ValTy, Alignment, AS,
                               Config.CostKind, OpInfo, I);
  }
  return getWideningCost(I, VF);
}

InstructionCost
LoopVectorizationCostModel::getScalarizationOverhead(Instruction *I,
                                                     ElementCount VF) const {

  // There is no mechanism yet to create a scalable scalarization loop,
  // so this is currently Invalid.
  if (VF.isScalable())
    return InstructionCost::getInvalid();

  if (VF.isScalar())
    return 0;

  InstructionCost Cost = 0;
  Type *RetTy = toVectorizedTy(I->getType(), VF);
  if (!RetTy->isVoidTy() &&
      (!isa<LoadInst>(I) || !TTI.supportsEfficientVectorElementLoadStore())) {

    TTI::VectorInstrContext VIC = TTI::VectorInstrContext::None;
    if (isa<LoadInst>(I))
      VIC = TTI::VectorInstrContext::Load;
    else if (isa<StoreInst>(I))
      VIC = TTI::VectorInstrContext::Store;

    for (Type *VectorTy : getContainedTypes(RetTy)) {
      Cost += TTI.getScalarizationOverhead(
          cast<VectorType>(VectorTy), APInt::getAllOnes(VF.getFixedValue()),
          /*Insert=*/true, /*Extract=*/false, Config.CostKind,
          /*ForPoisonSrc=*/true, {}, VIC);
    }
  }

  // Some targets keep addresses scalar.
  if (isa<LoadInst>(I) && !TTI.prefersVectorizedAddressing())
    return Cost;

  // Some targets support efficient element stores.
  if (isa<StoreInst>(I) && TTI.supportsEfficientVectorElementLoadStore())
    return Cost;

  // Collect operands to consider.
  CallInst *CI = dyn_cast<CallInst>(I);
  Instruction::op_range Ops = CI ? CI->args() : I->operands();

  // Skip operands that do not require extraction/scalarization and do not incur
  // any overhead.
  SmallVector<Type *> Tys;
  for (auto *V : filterExtractingOperands(Ops, VF))
    Tys.push_back(maybeVectorizeType(V->getType(), VF));

  TTI::VectorInstrContext OperandVIC = isa<StoreInst>(I)
                                           ? TTI::VectorInstrContext::Store
                                           : TTI::VectorInstrContext::None;
  return Cost +
         TTI.getOperandsScalarizationOverhead(Tys, Config.CostKind, OperandVIC);
}

void LoopVectorizationCostModel::setCostBasedWideningDecision(ElementCount VF) {
  if (VF.isScalar())
    return;
  NumPredStores = 0;
  for (BasicBlock *BB : TheLoop->blocks()) {
    // For each instruction in the old loop.
    for (Instruction &I : *BB) {
      Value *Ptr =  getLoadStorePointerOperand(&I);
      if (!Ptr)
        continue;

      // TODO: We should generate better code and update the cost model for
      // predicated uniform stores. Today they are treated as any other
      // predicated store (see added test cases in
      // invariant-store-vectorization.ll).
      if (isa<StoreInst>(&I) && isScalarWithPredication(&I, VF))
        NumPredStores++;

      if (Legal->isUniformMemOp(I, VF)) {
        auto IsLegalToScalarize = [&]() {
          if (!VF.isScalable())
            // Scalarization of fixed length vectors "just works".
            return true;

          // We have dedicated lowering for unpredicated uniform loads and
          // stores.  Note that even with tail folding we know that at least
          // one lane is active (i.e. generalized predication is not possible
          // here), and the logic below depends on this fact.
          if (!foldTailByMasking())
            return true;

          // For scalable vectors, a uniform memop load is always
          // uniform-by-parts  and we know how to scalarize that.
          if (isa<LoadInst>(I))
            return true;

          // A uniform store isn't neccessarily uniform-by-part
          // and we can't assume scalarization.
          auto &SI = cast<StoreInst>(I);
          return TheLoop->isLoopInvariant(SI.getValueOperand());
        };

        const InstructionCost GatherScatterCost =
            Config.isLegalGatherOrScatter(&I, VF)
                ? getGatherScatterCost(&I, VF)
                : InstructionCost::getInvalid();

        // Load: Scalar load + broadcast
        // Store: Scalar store + isLoopInvariantStoreValue ? 0 : extract
        // FIXME: This cost is a significant under-estimate for tail folded
        // memory ops.
        const InstructionCost ScalarizationCost =
            IsLegalToScalarize() ? getUniformMemOpCost(&I, VF)
                                 : InstructionCost::getInvalid();

        // Choose better solution for the current VF,  Note that Invalid
        // costs compare as maximumal large.  If both are invalid, we get
        // scalable invalid which signals a failure and a vectorization abort.
        if (GatherScatterCost < ScalarizationCost)
          setWideningDecision(&I, VF, CM_GatherScatter, GatherScatterCost);
        else
          setWideningDecision(&I, VF, CM_Scalarize, ScalarizationCost);
        continue;
      }

      // We assume that widening is the best solution when possible.
      if (memoryInstructionCanBeWidened(&I, VF)) {
        InstructionCost Cost = getConsecutiveMemOpCost(&I, VF);
        int ConsecutiveStride = Legal->isConsecutivePtr(
            getLoadStoreType(&I), getLoadStorePointerOperand(&I));
        assert((ConsecutiveStride == 1 || ConsecutiveStride == -1) &&
               "Expected consecutive stride.");
        InstWidening Decision =
            ConsecutiveStride == 1 ? CM_Widen : CM_Widen_Reverse;
        setWideningDecision(&I, VF, Decision, Cost);
        continue;
      }

      // Choose between Interleaving, Gather/Scatter or Scalarization.
      InstructionCost InterleaveCost = InstructionCost::getInvalid();
      unsigned NumAccesses = 1;
      if (isAccessInterleaved(&I)) {
        const auto *Group = getInterleavedAccessGroup(&I);
        assert(Group && "Fail to get an interleaved access group.");

        // Make one decision for the whole group.
        if (getWideningDecision(&I, VF) != CM_Unknown)
          continue;

        NumAccesses = Group->getNumMembers();
        if (interleavedAccessCanBeWidened(&I, VF))
          InterleaveCost = getInterleaveGroupCost(&I, VF);
      }

      InstructionCost GatherScatterCost =
          Config.isLegalGatherOrScatter(&I, VF)
              ? getGatherScatterCost(&I, VF) * NumAccesses
              : InstructionCost::getInvalid();

      InstructionCost ScalarizationCost =
          getMemInstScalarizationCost(&I, VF) * NumAccesses;

      // Choose better solution for the current VF,
      // write down this decision and use it during vectorization.
      InstructionCost Cost;
      InstWidening Decision;
      if (InterleaveCost <= GatherScatterCost &&
          InterleaveCost < ScalarizationCost) {
        Decision = CM_Interleave;
        Cost = InterleaveCost;
      } else if (GatherScatterCost < ScalarizationCost) {
        Decision = CM_GatherScatter;
        Cost = GatherScatterCost;
      } else {
        Decision = CM_Scalarize;
        Cost = ScalarizationCost;
      }
      // If the instructions belongs to an interleave group, the whole group
      // receives the same decision. The whole group receives the cost, but
      // the cost will actually be assigned to one instruction.
      if (const auto *Group = getInterleavedAccessGroup(&I)) {
        if (Decision == CM_Scalarize) {
          for (Instruction *I : Group->members())
            setWideningDecision(I, VF, Decision,
                                getMemInstScalarizationCost(I, VF));
        } else {
          setWideningDecision(Group, VF, Decision, Cost);
        }
      } else
        setWideningDecision(&I, VF, Decision, Cost);
    }
  }

  // Make sure that any load of address and any other address computation
  // remains scalar unless there is gather/scatter support. This avoids
  // inevitable extracts into address registers, and also has the benefit of
  // activating LSR more, since that pass can't optimize vectorized
  // addresses.
  if (TTI.prefersVectorizedAddressing())
    return;

  // Start with all scalar pointer uses.
  SmallPtrSet<Instruction *, 8> AddrDefs;
  for (BasicBlock *BB : TheLoop->blocks())
    for (Instruction &I : *BB) {
      Instruction *PtrDef =
        dyn_cast_or_null<Instruction>(getLoadStorePointerOperand(&I));
      if (PtrDef && TheLoop->contains(PtrDef) &&
          getWideningDecision(&I, VF) != CM_GatherScatter)
        AddrDefs.insert(PtrDef);
    }

  // Add all instructions used to generate the addresses.
  SmallVector<Instruction *, 4> Worklist;
  append_range(Worklist, AddrDefs);
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    for (auto &Op : I->operands())
      if (auto *InstOp = dyn_cast<Instruction>(Op))
        if (TheLoop->contains(InstOp) && !isa<PHINode>(InstOp) &&
            AddrDefs.insert(InstOp).second)
          Worklist.push_back(InstOp);
  }

  auto UpdateMemOpUserCost = [this, VF](LoadInst *LI) {
    // If there are direct memory op users of the newly scalarized load,
    // their cost may have changed because there's no scalarization
    // overhead for the operand. Update it.
    for (User *U : LI->users()) {
      if (!isa<LoadInst, StoreInst>(U))
        continue;
      if (getWideningDecision(cast<Instruction>(U), VF) != CM_Scalarize)
        continue;
      setWideningDecision(
          cast<Instruction>(U), VF, CM_Scalarize,
          getMemInstScalarizationCost(cast<Instruction>(U), VF));
    }
  };
  for (auto *I : AddrDefs) {
    if (isa<LoadInst>(I)) {
      // Setting the desired widening decision should ideally be handled in
      // by cost functions, but since this involves the task of finding out
      // if the loaded register is involved in an address computation, it is
      // instead changed here when we know this is the case.
      InstWidening Decision = getWideningDecision(I, VF);
      if (!isPredicatedInst(I) &&
          (Decision == CM_Widen || Decision == CM_Widen_Reverse ||
           (!Legal->isUniformMemOp(*I, VF) && Decision == CM_Scalarize))) {
        // Scalarize a widened load of address or update the cost of a scalar
        // load of an address.
        setWideningDecision(
            I, VF, CM_Scalarize,
            (VF.getKnownMinValue() *
             getMemoryInstructionCost(I, ElementCount::getFixed(1))));
        UpdateMemOpUserCost(cast<LoadInst>(I));
      } else if (const auto *Group = getInterleavedAccessGroup(I)) {
        // Scalarize all members of this interleaved group when any member
        // is used as an address. The address-used load skips scalarization
        // overhead, other members include it.
        for (Instruction *Member : Group->members()) {
          InstructionCost Cost = AddrDefs.contains(Member)
                                     ? (VF.getKnownMinValue() *
                                        getMemoryInstructionCost(
                                            Member, ElementCount::getFixed(1)))
                                     : getMemInstScalarizationCost(Member, VF);
          setWideningDecision(Member, VF, CM_Scalarize, Cost);
          UpdateMemOpUserCost(cast<LoadInst>(Member));
        }
      }
    } else {
      // Cannot scalarize fixed-order recurrence phis at the moment.
      if (isa<PHINode>(I) && Legal->isFixedOrderRecurrence(cast<PHINode>(I)))
        continue;

      // Make sure I gets scalarized and a cost estimate without
      // scalarization overhead.
      ForcedScalars[VF].insert(I);
    }
  }
}

void LoopVectorizationCostModel::setVectorizedCallDecision(ElementCount VF) {
  assert(!VF.isScalar() &&
         "Trying to set a vectorization decision for a scalar VF");

  auto ForcedScalar = ForcedScalars.find(VF);
  for (BasicBlock *BB : TheLoop->blocks()) {
    // For each instruction in the old loop.
    for (Instruction &I : *BB) {
      CallInst *CI = dyn_cast<CallInst>(&I);

      if (!CI)
        continue;

      InstructionCost ScalarCost = InstructionCost::getInvalid();
      InstructionCost VectorCost = InstructionCost::getInvalid();
      InstructionCost IntrinsicCost = InstructionCost::getInvalid();
      Function *ScalarFunc = CI->getCalledFunction();
      Type *ScalarRetTy = CI->getType();
      SmallVector<Type *, 4> Tys, ScalarTys;
      for (auto &ArgOp : CI->args())
        ScalarTys.push_back(ArgOp->getType());

      // Estimate cost of scalarized vector call. The source operands are
      // assumed to be vectors, so we need to extract individual elements from
      // there, execute VF scalar calls, and then gather the result into the
      // vector return value.
      if (VF.isFixed()) {
        InstructionCost ScalarCallCost = TTI.getCallInstrCost(
            ScalarFunc, ScalarRetTy, ScalarTys, Config.CostKind);

        // Compute costs of unpacking argument values for the scalar calls and
        // packing the return values to a vector.
        InstructionCost ScalarizationCost = getScalarizationOverhead(CI, VF);
        ScalarCost = ScalarCallCost * VF.getKnownMinValue() + ScalarizationCost;
      } else {
        // There is no point attempting to calculate the scalar cost for a
        // scalable VF as we know it will be Invalid.
        assert(!getScalarizationOverhead(CI, VF).isValid() &&
               "Unexpected valid cost for scalarizing scalable vectors");
        ScalarCost = InstructionCost::getInvalid();
      }

      // Honor ForcedScalars and UniformAfterVectorization decisions.
      // TODO: For calls, it might still be more profitable to widen. Use
      // VPlan-based cost model to compare different options.
      if (VF.isVector() && ((ForcedScalar != ForcedScalars.end() &&
                             ForcedScalar->second.contains(CI)) ||
                            isUniformAfterVectorization(CI, VF))) {
        setCallWideningDecision(CI, VF, CM_Scalarize, nullptr,
                                Intrinsic::not_intrinsic, std::nullopt,
                                ScalarCost);
        continue;
      }

      bool MaskRequired = isMaskRequired(CI);
      // Compute corresponding vector type for return value and arguments.
      Type *RetTy = toVectorizedTy(ScalarRetTy, VF);
      for (Type *ScalarTy : ScalarTys)
        Tys.push_back(toVectorizedTy(ScalarTy, VF));

      // An in-loop reduction using an fmuladd intrinsic is a special case;
      // we don't want the normal cost for that intrinsic.
      if (RecurrenceDescriptor::isFMulAddIntrinsic(CI))
        if (auto RedCost = getReductionPatternCost(CI, VF, RetTy)) {
          setCallWideningDecision(CI, VF, CM_IntrinsicCall, nullptr,
                                  getVectorIntrinsicIDForCall(CI, TLI),
                                  std::nullopt, *RedCost);
          continue;
        }

      // Find the cost of vectorizing the call, if we can find a suitable
      // vector variant of the function.
      VFInfo FuncInfo;
      Function *VecFunc = nullptr;
      // Search through any available variants for one we can use at this VF.
      for (VFInfo &Info : VFDatabase::getMappings(*CI)) {
        // Must match requested VF.
        if (Info.Shape.VF != VF)
          continue;

        // Must take a mask argument if one is required
        if (MaskRequired && !Info.isMasked())
          continue;

        // Check that all parameter kinds are supported
        bool ParamsOk = true;
        for (VFParameter Param : Info.Shape.Parameters) {
          switch (Param.ParamKind) {
          case VFParamKind::Vector:
            break;
          case VFParamKind::OMP_Uniform: {
            Value *ScalarParam = CI->getArgOperand(Param.ParamPos);
            // Make sure the scalar parameter in the loop is invariant.
            if (!PSE.getSE()->isLoopInvariant(PSE.getSCEV(ScalarParam),
                                              TheLoop))
              ParamsOk = false;
            break;
          }
          case VFParamKind::OMP_Linear: {
            Value *ScalarParam = CI->getArgOperand(Param.ParamPos);
            // Find the stride for the scalar parameter in this loop and see if
            // it matches the stride for the variant.
            // TODO: do we need to figure out the cost of an extract to get the
            // first lane? Or do we hope that it will be folded away?
            ScalarEvolution *SE = PSE.getSE();
            if (!match(SE->getSCEV(ScalarParam),
                       m_scev_AffineAddRec(
                           m_SCEV(), m_scev_SpecificSInt(Param.LinearStepOrPos),
                           m_SpecificLoop(TheLoop))))
              ParamsOk = false;
            break;
          }
          case VFParamKind::GlobalPredicate:
            break;
          default:
            ParamsOk = false;
            break;
          }
        }

        if (!ParamsOk)
          continue;

        // Found a suitable candidate, stop here.
        VecFunc = CI->getModule()->getFunction(Info.VectorName);
        FuncInfo = Info;
        break;
      }

      if (TLI && VecFunc && !CI->isNoBuiltin())
        VectorCost = TTI.getCallInstrCost(nullptr, RetTy, Tys, Config.CostKind);

      // Find the cost of an intrinsic; some targets may have instructions that
      // perform the operation without needing an actual call.
      Intrinsic::ID IID = getVectorIntrinsicIDForCall(CI, TLI);
      if (IID != Intrinsic::not_intrinsic)
        IntrinsicCost = getVectorIntrinsicCost(CI, VF);

      InstructionCost Cost = ScalarCost;
      InstWidening Decision = CM_Scalarize;

      if (VectorCost.isValid() && VectorCost <= Cost) {
        Cost = VectorCost;
        Decision = CM_VectorCall;
      }

      if (IntrinsicCost.isValid() && IntrinsicCost <= Cost) {
        Cost = IntrinsicCost;
        Decision = CM_IntrinsicCall;
      }

      setCallWideningDecision(CI, VF, Decision, VecFunc, IID,
                              FuncInfo.getParamIndexForOptionalMask(), Cost);
    }
  }
}

bool LoopVectorizationCostModel::shouldConsiderInvariant(Value *Op) {
  if (!Legal->isInvariant(Op))
    return false;
  // Consider Op invariant, if it or its operands aren't predicated
  // instruction in the loop. In that case, it is not trivially hoistable.
  auto *OpI = dyn_cast<Instruction>(Op);
  return !OpI || !TheLoop->contains(OpI) ||
         (!isPredicatedInst(OpI) &&
          (!isa<PHINode>(OpI) || OpI->getParent() != TheLoop->getHeader()) &&
          all_of(OpI->operands(),
                 [this](Value *Op) { return shouldConsiderInvariant(Op); }));
}

InstructionCost
LoopVectorizationCostModel::getInstructionCost(Instruction *I,
                                               ElementCount VF) {
  // If we know that this instruction will remain uniform, check the cost of
  // the scalar version.
  if (isUniformAfterVectorization(I, VF))
    VF = ElementCount::getFixed(1);

  if (VF.isVector() && isProfitableToScalarize(I, VF))
    return InstsToScalarize[VF][I];

  // Forced scalars do not have any scalarization overhead.
  auto ForcedScalar = ForcedScalars.find(VF);
  if (VF.isVector() && ForcedScalar != ForcedScalars.end()) {
    auto InstSet = ForcedScalar->second;
    if (InstSet.count(I))
      return getInstructionCost(I, ElementCount::getFixed(1)) *
             VF.getKnownMinValue();
  }

  const auto &MinBWs = Config.getMinimalBitwidths();
  uint64_t InstrMinBWs = MinBWs.lookup(I);
  Type *RetTy = I->getType();
  if (canTruncateToMinimalBitwidth(I, VF))
    RetTy = IntegerType::get(RetTy->getContext(), InstrMinBWs);
  auto *SE = PSE.getSE();

  Type *VectorTy;
  if (isScalarAfterVectorization(I, VF)) {
    [[maybe_unused]] auto HasSingleCopyAfterVectorization =
        [this](Instruction *I, ElementCount VF) -> bool {
      if (VF.isScalar())
        return true;

      auto Scalarized = InstsToScalarize.find(VF);
      assert(Scalarized != InstsToScalarize.end() &&
             "VF not yet analyzed for scalarization profitability");
      return !Scalarized->second.count(I) &&
             llvm::all_of(I->users(), [&](User *U) {
               auto *UI = cast<Instruction>(U);
               return !Scalarized->second.count(UI);
             });
    };

    // With the exception of GEPs and PHIs, after scalarization there should
    // only be one copy of the instruction generated in the loop. This is
    // because the VF is either 1, or any instructions that need scalarizing
    // have already been dealt with by the time we get here. As a result,
    // it means we don't have to multiply the instruction cost by VF.
    assert(I->getOpcode() == Instruction::GetElementPtr ||
           I->getOpcode() == Instruction::PHI ||
           (I->getOpcode() == Instruction::BitCast &&
            I->getType()->isPointerTy()) ||
           HasSingleCopyAfterVectorization(I, VF));
    VectorTy = RetTy;
  } else
    VectorTy = toVectorizedTy(RetTy, VF);

  if (VF.isVector() && VectorTy->isVectorTy() &&
      !TTI.getNumberOfParts(VectorTy))
    return InstructionCost::getInvalid();

  // TODO: We need to estimate the cost of intrinsic calls.
  switch (I->getOpcode()) {
  case Instruction::GetElementPtr:
    // We mark this instruction as zero-cost because the cost of GEPs in
    // vectorized code depends on whether the corresponding memory instruction
    // is scalarized or not. Therefore, we handle GEPs with the memory
    // instruction cost.
    return 0;
  case Instruction::UncondBr:
  case Instruction::CondBr: {
    // In cases of scalarized and predicated instructions, there will be VF
    // predicated blocks in the vectorized loop. Each branch around these
    // blocks requires also an extract of its vector compare i1 element.
    // Note that the conditional branch from the loop latch will be replaced by
    // a single branch controlling the loop, so there is no extra overhead from
    // scalarization.
    bool ScalarPredicatedBB = false;
    CondBrInst *BI = dyn_cast<CondBrInst>(I);
    if (VF.isVector() && BI &&
        (PredicatedBBsAfterVectorization[VF].count(BI->getSuccessor(0)) ||
         PredicatedBBsAfterVectorization[VF].count(BI->getSuccessor(1))) &&
        BI->getParent() != TheLoop->getLoopLatch())
      ScalarPredicatedBB = true;

    if (ScalarPredicatedBB) {
      // Not possible to scalarize scalable vector with predicated instructions.
      if (VF.isScalable())
        return InstructionCost::getInvalid();
      // Return cost for branches around scalarized and predicated blocks.
      auto *VecI1Ty =
          VectorType::get(IntegerType::getInt1Ty(RetTy->getContext()), VF);
      return (TTI.getScalarizationOverhead(
                  VecI1Ty, APInt::getAllOnes(VF.getFixedValue()),
                  /*Insert*/ false, /*Extract*/ true, Config.CostKind) +
              (TTI.getCFInstrCost(Instruction::CondBr, Config.CostKind) *
               VF.getFixedValue()));
    }

    if (I->getParent() == TheLoop->getLoopLatch() || VF.isScalar())
      // The back-edge branch will remain, as will all scalar branches.
      return TTI.getCFInstrCost(Instruction::UncondBr, Config.CostKind);

    // This branch will be eliminated by if-conversion.
    return 0;
    // Note: We currently assume zero cost for an unconditional branch inside
    // a predicated block since it will become a fall-through, although we
    // may decide in the future to call TTI for all branches.
  }
  case Instruction::Switch: {
    if (VF.isScalar())
      return TTI.getCFInstrCost(Instruction::Switch, Config.CostKind);
    auto *Switch = cast<SwitchInst>(I);
    return Switch->getNumCases() *
           TTI.getCmpSelInstrCost(
               Instruction::ICmp,
               toVectorTy(Switch->getCondition()->getType(), VF),
               toVectorTy(Type::getInt1Ty(I->getContext()), VF),
               CmpInst::ICMP_EQ, Config.CostKind);
  }
  case Instruction::PHI: {
    auto *Phi = cast<PHINode>(I);

    // First-order recurrences are replaced by vector shuffles inside the loop.
    if (VF.isVector() && Legal->isFixedOrderRecurrence(Phi)) {
      return TTI.getShuffleCost(
          TargetTransformInfo::SK_Splice, cast<VectorType>(VectorTy),
          cast<VectorType>(VectorTy), {}, Config.CostKind, -1);
    }

    // Phi nodes in non-header blocks (not inductions, reductions, etc.) are
    // converted into select instructions. We require N - 1 selects per phi
    // node, where N is the number of incoming values.
    if (VF.isVector() && Phi->getParent() != TheLoop->getHeader()) {
      Type *ResultTy = Phi->getType();

      // All instructions in an Any-of reduction chain are narrowed to bool.
      // Check if that is the case for this phi node.
      auto *HeaderUser = cast_if_present<PHINode>(
          find_singleton<User>(Phi->users(), [this](User *U, bool) -> User * {
            auto *Phi = dyn_cast<PHINode>(U);
            if (Phi && Phi->getParent() == TheLoop->getHeader())
              return Phi;
            return nullptr;
          }));
      if (HeaderUser) {
        auto &ReductionVars = Legal->getReductionVars();
        auto Iter = ReductionVars.find(HeaderUser);
        if (Iter != ReductionVars.end() &&
            RecurrenceDescriptor::isAnyOfRecurrenceKind(
                Iter->second.getRecurrenceKind()))
          ResultTy = Type::getInt1Ty(Phi->getContext());
      }
      return (Phi->getNumIncomingValues() - 1) *
             TTI.getCmpSelInstrCost(
                 Instruction::Select, toVectorTy(ResultTy, VF),
                 toVectorTy(Type::getInt1Ty(Phi->getContext()), VF),
                 CmpInst::BAD_ICMP_PREDICATE, Config.CostKind);
    }

    // When tail folding with EVL, if the phi is part of an out of loop
    // reduction then it will be transformed into a wide vp_merge.
    if (VF.isVector() && foldTailWithEVL() &&
        Legal->getReductionVars().contains(Phi) &&
        !Config.isInLoopReduction(Phi)) {
      IntrinsicCostAttributes ICA(
          Intrinsic::vp_merge, toVectorTy(Phi->getType(), VF),
          {toVectorTy(Type::getInt1Ty(Phi->getContext()), VF)});
      return TTI.getIntrinsicInstrCost(ICA, Config.CostKind);
    }

    return TTI.getCFInstrCost(Instruction::PHI, Config.CostKind);
  }
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
    if (VF.isVector() && isPredicatedInst(I)) {
      const auto [ScalarCost, SafeDivisorCost] = getDivRemSpeculationCost(I, VF);
      return isDivRemScalarWithPredication(ScalarCost, SafeDivisorCost) ?
        ScalarCost : SafeDivisorCost;
    }
    // We've proven all lanes safe to speculate, fall through.
    [[fallthrough]];
  case Instruction::Add:
  case Instruction::Sub: {
    auto Info = Legal->getHistogramInfo(I);
    if (Info && VF.isVector()) {
      const HistogramInfo *HGram = Info.value();
      // Assume that a non-constant update value (or a constant != 1) requires
      // a multiply, and add that into the cost.
      InstructionCost MulCost = TTI::TCC_Free;
      ConstantInt *RHS = dyn_cast<ConstantInt>(I->getOperand(1));
      if (!RHS || RHS->getZExtValue() != 1)
        MulCost = TTI.getArithmeticInstrCost(Instruction::Mul, VectorTy,
                                             Config.CostKind);

      // Find the cost of the histogram operation itself.
      Type *PtrTy = VectorType::get(HGram->Load->getPointerOperandType(), VF);
      Type *ScalarTy = I->getType();
      Type *MaskTy = VectorType::get(Type::getInt1Ty(I->getContext()), VF);
      IntrinsicCostAttributes ICA(Intrinsic::experimental_vector_histogram_add,
                                  Type::getVoidTy(I->getContext()),
                                  {PtrTy, ScalarTy, MaskTy});

      // Add the costs together with the add/sub operation.
      return TTI.getIntrinsicInstrCost(ICA, Config.CostKind) + MulCost +
             TTI.getArithmeticInstrCost(I->getOpcode(), VectorTy,
                                        Config.CostKind);
    }
    [[fallthrough]];
  }
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    // If we're speculating on the stride being 1, the multiplication may
    // fold away.  We can generalize this for all operations using the notion
    // of neutral elements.  (TODO)
    if (I->getOpcode() == Instruction::Mul &&
        ((TheLoop->isLoopInvariant(I->getOperand(0)) &&
          PSE.getSCEV(I->getOperand(0))->isOne()) ||
         (TheLoop->isLoopInvariant(I->getOperand(1)) &&
          PSE.getSCEV(I->getOperand(1))->isOne())))
      return 0;

    // Detect reduction patterns
    if (auto RedCost = getReductionPatternCost(I, VF, VectorTy))
      return *RedCost;

    // Certain instructions can be cheaper to vectorize if they have a constant
    // second vector operand. One example of this are shifts on x86.
    Value *Op2 = I->getOperand(1);
    if (!isa<Constant>(Op2) && TheLoop->isLoopInvariant(Op2) &&
        PSE.getSE()->isSCEVable(Op2->getType()) &&
        isa<SCEVConstant>(PSE.getSCEV(Op2))) {
      Op2 = cast<SCEVConstant>(PSE.getSCEV(Op2))->getValue();
    }
    auto Op2Info = TTI.getOperandInfo(Op2);
    if (Op2Info.Kind == TargetTransformInfo::OK_AnyValue &&
        shouldConsiderInvariant(Op2))
      Op2Info.Kind = TargetTransformInfo::OK_UniformValue;

    SmallVector<const Value *, 4> Operands(I->operand_values());
    return TTI.getArithmeticInstrCost(
        I->getOpcode(), VectorTy, Config.CostKind,
        {TargetTransformInfo::OK_AnyValue, TargetTransformInfo::OP_None},
        Op2Info, Operands, I, TLI);
  }
  case Instruction::FNeg: {
    return TTI.getArithmeticInstrCost(
        I->getOpcode(), VectorTy, Config.CostKind,
        {TargetTransformInfo::OK_AnyValue, TargetTransformInfo::OP_None},
        {TargetTransformInfo::OK_AnyValue, TargetTransformInfo::OP_None},
        I->getOperand(0), I);
  }
  case Instruction::Select: {
    SelectInst *SI = cast<SelectInst>(I);
    const SCEV *CondSCEV = SE->getSCEV(SI->getCondition());
    bool ScalarCond = (SE->isLoopInvariant(CondSCEV, TheLoop));

    const Value *Op0, *Op1;
    using namespace llvm::PatternMatch;
    if (!ScalarCond && (match(I, m_LogicalAnd(m_Value(Op0), m_Value(Op1))) ||
                        match(I, m_LogicalOr(m_Value(Op0), m_Value(Op1))))) {
      // select x, y, false --> x & y
      // select x, true, y --> x | y
      const auto [Op1VK, Op1VP] = TTI::getOperandInfo(Op0);
      const auto [Op2VK, Op2VP] = TTI::getOperandInfo(Op1);
      assert(Op0->getType()->getScalarSizeInBits() == 1 &&
              Op1->getType()->getScalarSizeInBits() == 1);

      return TTI.getArithmeticInstrCost(
          match(I, m_LogicalOr()) ? Instruction::Or : Instruction::And,
          VectorTy, Config.CostKind, {Op1VK, Op1VP}, {Op2VK, Op2VP}, {Op0, Op1},
          I);
    }

    Type *CondTy = SI->getCondition()->getType();
    if (!ScalarCond)
      CondTy = VectorType::get(CondTy, VF);

    CmpInst::Predicate Pred = CmpInst::BAD_ICMP_PREDICATE;
    if (auto *Cmp = dyn_cast<CmpInst>(SI->getCondition()))
      Pred = Cmp->getPredicate();
    return TTI.getCmpSelInstrCost(
        I->getOpcode(), VectorTy, CondTy, Pred, Config.CostKind,
        {TTI::OK_AnyValue, TTI::OP_None}, {TTI::OK_AnyValue, TTI::OP_None}, I);
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    Type *ValTy = I->getOperand(0)->getType();

    if (canTruncateToMinimalBitwidth(I, VF)) {
      [[maybe_unused]] Instruction *Op0AsInstruction =
          dyn_cast<Instruction>(I->getOperand(0));
      assert((!canTruncateToMinimalBitwidth(Op0AsInstruction, VF) ||
              InstrMinBWs == MinBWs.lookup(Op0AsInstruction)) &&
             "if both the operand and the compare are marked for "
             "truncation, they must have the same bitwidth");
      ValTy = IntegerType::get(ValTy->getContext(), InstrMinBWs);
    }

    VectorTy = toVectorTy(ValTy, VF);
    return TTI.getCmpSelInstrCost(
        I->getOpcode(), VectorTy, CmpInst::makeCmpResultType(VectorTy),
        cast<CmpInst>(I)->getPredicate(), Config.CostKind,
        {TTI::OK_AnyValue, TTI::OP_None}, {TTI::OK_AnyValue, TTI::OP_None}, I);
  }
  case Instruction::Store:
  case Instruction::Load: {
    ElementCount Width = VF;
    if (Width.isVector()) {
      InstWidening Decision = getWideningDecision(I, Width);
      assert(Decision != CM_Unknown &&
             "CM decision should be taken at this point");
      if (getWideningCost(I, VF) == InstructionCost::getInvalid())
        return InstructionCost::getInvalid();
      if (Decision == CM_Scalarize)
        Width = ElementCount::getFixed(1);
    }
    VectorTy = toVectorTy(getLoadStoreType(I), Width);
    return getMemoryInstructionCost(I, VF);
  }
  case Instruction::BitCast:
    if (I->getType()->isPointerTy())
      return 0;
    [[fallthrough]];
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::FPExt:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
  case Instruction::Trunc:
  case Instruction::FPTrunc: {
    // Computes the CastContextHint from a Load/Store instruction.
    auto ComputeCCH = [&](Instruction *I) -> TTI::CastContextHint {
      assert((isa<LoadInst>(I) || isa<StoreInst>(I)) &&
             "Expected a load or a store!");

      if (VF.isScalar() || !TheLoop->contains(I))
        return TTI::CastContextHint::Normal;

      switch (getWideningDecision(I, VF)) {
      case LoopVectorizationCostModel::CM_GatherScatter:
        return TTI::CastContextHint::GatherScatter;
      case LoopVectorizationCostModel::CM_Interleave:
        return TTI::CastContextHint::Interleave;
      case LoopVectorizationCostModel::CM_Scalarize:
      case LoopVectorizationCostModel::CM_Widen:
        return isPredicatedInst(I) ? TTI::CastContextHint::Masked
                                   : TTI::CastContextHint::Normal;
      case LoopVectorizationCostModel::CM_Widen_Reverse:
        return TTI::CastContextHint::Reversed;
      case LoopVectorizationCostModel::CM_Unknown:
        llvm_unreachable("Instr did not go through cost modelling?");
      case LoopVectorizationCostModel::CM_VectorCall:
      case LoopVectorizationCostModel::CM_IntrinsicCall:
        llvm_unreachable_internal("Instr has invalid widening decision");
      }

      llvm_unreachable("Unhandled case!");
    };

    unsigned Opcode = I->getOpcode();
    TTI::CastContextHint CCH = TTI::CastContextHint::None;
    // For Trunc, the context is the only user, which must be a StoreInst.
    if (Opcode == Instruction::Trunc || Opcode == Instruction::FPTrunc) {
      if (I->hasOneUse())
        if (StoreInst *Store = dyn_cast<StoreInst>(*I->user_begin()))
          CCH = ComputeCCH(Store);
    }
    // For Z/Sext, the context is the operand, which must be a LoadInst.
    else if (Opcode == Instruction::ZExt || Opcode == Instruction::SExt ||
             Opcode == Instruction::FPExt) {
      if (LoadInst *Load = dyn_cast<LoadInst>(I->getOperand(0)))
        CCH = ComputeCCH(Load);
    }

    // We optimize the truncation of induction variables having constant
    // integer steps. The cost of these truncations is the same as the scalar
    // operation.
    if (isOptimizableIVTruncate(I, VF)) {
      auto *Trunc = cast<TruncInst>(I);
      return TTI.getCastInstrCost(Instruction::Trunc, Trunc->getDestTy(),
                                  Trunc->getSrcTy(), CCH, Config.CostKind,
                                  Trunc);
    }

    // Detect reduction patterns
    if (auto RedCost = getReductionPatternCost(I, VF, VectorTy))
      return *RedCost;

    Type *SrcScalarTy = I->getOperand(0)->getType();
    Instruction *Op0AsInstruction = dyn_cast<Instruction>(I->getOperand(0));
    if (canTruncateToMinimalBitwidth(Op0AsInstruction, VF))
      SrcScalarTy = IntegerType::get(SrcScalarTy->getContext(),
                                     MinBWs.lookup(Op0AsInstruction));
    Type *SrcVecTy =
        VectorTy->isVectorTy() ? toVectorTy(SrcScalarTy, VF) : SrcScalarTy;

    if (canTruncateToMinimalBitwidth(I, VF)) {
      // If the result type is <= the source type, there will be no extend
      // after truncating the users to the minimal required bitwidth.
      if (VectorTy->getScalarSizeInBits() <= SrcVecTy->getScalarSizeInBits() &&
          (I->getOpcode() == Instruction::ZExt ||
           I->getOpcode() == Instruction::SExt))
        return 0;
    }

    return TTI.getCastInstrCost(Opcode, VectorTy, SrcVecTy, CCH,
                                Config.CostKind, I);
  }
  case Instruction::Call:
    return getVectorCallCost(cast<CallInst>(I), VF);
  case Instruction::ExtractValue:
    return TTI.getInstructionCost(I, Config.CostKind);
  case Instruction::Alloca:
    // We cannot easily widen alloca to a scalable alloca, as
    // the result would need to be a vector of pointers.
    if (VF.isScalable())
      return InstructionCost::getInvalid();
    return TTI.getArithmeticInstrCost(Instruction::Mul, RetTy, Config.CostKind);
  default:
    // This opcode is unknown. Assume that it is the same as 'mul'.
    return TTI.getArithmeticInstrCost(Instruction::Mul, VectorTy,
                                      Config.CostKind);
  } // end of switch.
}

void LoopVectorizationCostModel::collectValuesToIgnore() {
  // Ignore ephemeral values.
  CodeMetrics::collectEphemeralValues(TheLoop, AC, ValuesToIgnore);

  SmallVector<Value *, 4> DeadInterleavePointerOps;
  SmallVector<Value *, 4> DeadOps;

  // If a scalar epilogue is required, users outside the loop won't use
  // live-outs from the vector loop but from the scalar epilogue. Ignore them if
  // that is the case.
  bool RequiresScalarEpilogue = requiresScalarEpilogue(true);
  auto IsLiveOutDead = [this, RequiresScalarEpilogue](User *U) {
    return RequiresScalarEpilogue &&
           !TheLoop->contains(cast<Instruction>(U)->getParent());
  };

  LoopBlocksDFS DFS(TheLoop);
  DFS.perform(LI);
  for (BasicBlock *BB : reverse(make_range(DFS.beginRPO(), DFS.endRPO())))
    for (Instruction &I : reverse(*BB)) {
      if (VecValuesToIgnore.contains(&I) || ValuesToIgnore.contains(&I))
        continue;

      // Add instructions that would be trivially dead and are only used by
      // values already ignored to DeadOps to seed worklist.
      if (wouldInstructionBeTriviallyDead(&I, TLI) &&
          all_of(I.users(), [this, IsLiveOutDead](User *U) {
            return VecValuesToIgnore.contains(U) ||
                   ValuesToIgnore.contains(U) || IsLiveOutDead(U);
          }))
        DeadOps.push_back(&I);

      // For interleave groups, we only create a pointer for the start of the
      // interleave group. Queue up addresses of group members except the insert
      // position for further processing.
      if (isAccessInterleaved(&I)) {
        auto *Group = getInterleavedAccessGroup(&I);
        if (Group->getInsertPos() == &I)
          continue;
        Value *PointerOp = getLoadStorePointerOperand(&I);
        DeadInterleavePointerOps.push_back(PointerOp);
      }

      // Queue branches for analysis. They are dead, if their successors only
      // contain dead instructions.
      if (isa<CondBrInst>(&I))
        DeadOps.push_back(&I);
    }

  // Mark ops feeding interleave group members as free, if they are only used
  // by other dead computations.
  for (unsigned I = 0; I != DeadInterleavePointerOps.size(); ++I) {
    auto *Op = dyn_cast<Instruction>(DeadInterleavePointerOps[I]);
    if (!Op || !TheLoop->contains(Op) || any_of(Op->users(), [this](User *U) {
          Instruction *UI = cast<Instruction>(U);
          return !VecValuesToIgnore.contains(U) &&
                 (!isAccessInterleaved(UI) ||
                  getInterleavedAccessGroup(UI)->getInsertPos() == UI);
        }))
      continue;
    VecValuesToIgnore.insert(Op);
    append_range(DeadInterleavePointerOps, Op->operands());
  }

  // Mark ops that would be trivially dead and are only used by ignored
  // instructions as free.
  BasicBlock *Header = TheLoop->getHeader();

  // Returns true if the block contains only dead instructions. Such blocks will
  // be removed by VPlan-to-VPlan transforms and won't be considered by the
  // VPlan-based cost model, so skip them in the legacy cost-model as well.
  auto IsEmptyBlock = [this](BasicBlock *BB) {
    return all_of(*BB, [this](Instruction &I) {
      return ValuesToIgnore.contains(&I) || VecValuesToIgnore.contains(&I) ||
             isa<UncondBrInst>(&I);
    });
  };
  for (unsigned I = 0; I != DeadOps.size(); ++I) {
    auto *Op = dyn_cast<Instruction>(DeadOps[I]);

    // Check if the branch should be considered dead.
    if (auto *Br = dyn_cast_or_null<CondBrInst>(Op)) {
      BasicBlock *ThenBB = Br->getSuccessor(0);
      BasicBlock *ElseBB = Br->getSuccessor(1);
      // Don't considers branches leaving the loop for simplification.
      if (!TheLoop->contains(ThenBB) || !TheLoop->contains(ElseBB))
        continue;
      bool ThenEmpty = IsEmptyBlock(ThenBB);
      bool ElseEmpty = IsEmptyBlock(ElseBB);
      if ((ThenEmpty && ElseEmpty) ||
          (ThenEmpty && ThenBB->getSingleSuccessor() == ElseBB &&
           ElseBB->phis().empty()) ||
          (ElseEmpty && ElseBB->getSingleSuccessor() == ThenBB &&
           ThenBB->phis().empty())) {
        VecValuesToIgnore.insert(Br);
        DeadOps.push_back(Br->getCondition());
      }
      continue;
    }

    // Skip any op that shouldn't be considered dead.
    if (!Op || !TheLoop->contains(Op) ||
        (isa<PHINode>(Op) && Op->getParent() == Header) ||
        !wouldInstructionBeTriviallyDead(Op, TLI) ||
        any_of(Op->users(), [this, IsLiveOutDead](User *U) {
          return !VecValuesToIgnore.contains(U) &&
                 !ValuesToIgnore.contains(U) && !IsLiveOutDead(U);
        }))
      continue;

    // If all of Op's users are in ValuesToIgnore, add it to ValuesToIgnore
    // which applies for both scalar and vector versions. Otherwise it is only
    // dead in vector versions, so only add it to VecValuesToIgnore.
    if (all_of(Op->users(),
               [this](User *U) { return ValuesToIgnore.contains(U); }))
      ValuesToIgnore.insert(Op);

    VecValuesToIgnore.insert(Op);
    append_range(DeadOps, Op->operands());
  }

  // Ignore type-promoting instructions we identified during reduction
  // detection.
  for (const auto &Reduction : Legal->getReductionVars()) {
    const RecurrenceDescriptor &RedDes = Reduction.second;
    const SmallPtrSetImpl<Instruction *> &Casts = RedDes.getCastInsts();
    VecValuesToIgnore.insert_range(Casts);
  }
  // Ignore type-casting instructions we identified during induction
  // detection.
  for (const auto &Induction : Legal->getInductionVars()) {
    const InductionDescriptor &IndDes = Induction.second;
    VecValuesToIgnore.insert_range(IndDes.getCastInsts());
  }
}

// This function will select a scalable VF if the target supports scalable
// vectors and a fixed one otherwise.
// TODO: we could return a pair of values that specify the max VF and
// min VF, to be used in `buildVPlans(MinVF, MaxVF)` instead of
// `buildVPlans(VF, VF)`. We cannot do it because VPLAN at the moment
// doesn't have a cost model that can choose which plan to execute if
// more than one is generated.
static ElementCount determineVPlanVF(const TargetTransformInfo &TTI,
                                     VFSelectionContext &Config) {
  unsigned WidestType = Config.getSmallestAndWidestTypes().second;

  TargetTransformInfo::RegisterKind RegKind =
      TTI.enableScalableVectorization()
          ? TargetTransformInfo::RGK_ScalableVector
          : TargetTransformInfo::RGK_FixedWidthVector;

  TypeSize RegSize = TTI.getRegisterBitWidth(RegKind);
  unsigned N = RegSize.getKnownMinValue() / WidestType;
  return ElementCount::get(N, RegSize.isScalable());
}

VectorizationFactor
LoopVectorizationPlanner::planInVPlanNativePath(ElementCount UserVF) {
  ElementCount VF = UserVF;
  // Outer loop handling: They may require CFG and instruction level
  // transformations before even evaluating whether vectorization is profitable.
  // Since we cannot modify the incoming IR, we need to build VPlan upfront in
  // the vectorization pipeline.
  if (!OrigLoop->isInnermost()) {
    // If the user doesn't provide a vectorization factor, determine a
    // reasonable one.
    if (UserVF.isZero()) {
      VF = determineVPlanVF(TTI, Config);
      LLVM_DEBUG(dbgs() << "LV: VPlan computed VF " << VF << ".\n");

      // Make sure we have a VF > 1 for stress testing.
      if (VPlanBuildStressTest && (VF.isScalar() || VF.isZero())) {
        LLVM_DEBUG(dbgs() << "LV: VPlan stress testing: "
                          << "overriding computed VF.\n");
        VF = ElementCount::getFixed(4);
      }
    } else if (UserVF.isScalable() && !Config.supportsScalableVectors()) {
      LLVM_DEBUG(dbgs() << "LV: Not vectorizing. Scalable VF requested, but "
                        << "not supported by the target.\n");
      reportVectorizationFailure(
          "Scalable vectorization requested but not supported by the target",
          "the scalable user-specified vectorization width for outer-loop "
          "vectorization cannot be used because the target does not support "
          "scalable vectors.",
          "ScalableVFUnfeasible", ORE, OrigLoop);
      return VectorizationFactor::Disabled();
    }
    assert(EnableVPlanNativePath && "VPlan-native path is not enabled.");
    assert(isPowerOf2_32(VF.getKnownMinValue()) &&
           "VF needs to be a power of two");
    LLVM_DEBUG(dbgs() << "LV: Using " << (!UserVF.isZero() ? "user " : "")
                      << "VF " << VF << " to build VPlans.\n");
    buildVPlans(VF, VF);

    if (VPlans.empty())
      return VectorizationFactor::Disabled();

    // For VPlan build stress testing, we bail out after VPlan construction.
    if (VPlanBuildStressTest)
      return VectorizationFactor::Disabled();

    return {VF, 0 /*Cost*/, 0 /* ScalarCost */};
  }

  LLVM_DEBUG(
      dbgs() << "LV: Not vectorizing. Inner loops aren't supported in the "
                "VPlan-native path.\n");
  return VectorizationFactor::Disabled();
}

void LoopVectorizationPlanner::plan(ElementCount UserVF, unsigned UserIC) {
  assert(OrigLoop->isInnermost() && "Inner loop expected.");
  CM.collectValuesToIgnore();
  Config.collectElementTypesForWidening(&CM.ValuesToIgnore);

  FixedScalableVFPair MaxFactors = CM.computeMaxVF(UserVF, UserIC);
  if (!MaxFactors) // Cases that should not to be vectorized nor interleaved.
    return;

  // Compute the minimal bitwidths required for integer operations in the loop
  // for later use by the cost model.
  Config.computeMinimalBitwidths();

  // Invalidate interleave groups if all blocks of loop will be predicated.
  if (CM.blockNeedsPredicationForAnyReason(OrigLoop->getHeader()) &&
      !useMaskedInterleavedAccesses(TTI)) {
    LLVM_DEBUG(
        dbgs()
        << "LV: Invalidate all interleaved groups due to fold-tail by masking "
           "which requires masked-interleaved support.\n");
    if (CM.InterleaveInfo.invalidateGroups())
      // Invalidating interleave groups also requires invalidating all decisions
      // based on them, which includes widening decisions and uniform and scalar
      // values.
      CM.invalidateCostModelingDecisions();
  }

  if (CM.foldTailByMasking())
    Legal->prepareToFoldTailByMasking();

  ElementCount MaxUserVF =
      UserVF.isScalable() ? MaxFactors.ScalableVF : MaxFactors.FixedVF;
  if (UserVF) {
    if (!ElementCount::isKnownLE(UserVF, MaxUserVF)) {
      reportVectorizationInfo(
          "UserVF ignored because it may be larger than the maximal safe VF",
          "InvalidUserVF", ORE, OrigLoop);
    } else {
      assert(isPowerOf2_32(UserVF.getKnownMinValue()) &&
             "VF needs to be a power of two");
      // Collect the instructions (and their associated costs) that will be more
      // profitable to scalarize.
      Config.collectInLoopReductions();
      CM.collectNonVectorizedAndSetWideningDecisions(UserVF);
      ElementCount EpilogueUserVF =
          ElementCount::getFixed(EpilogueVectorizationForceVF);
      if (EpilogueUserVF.isVector() &&
          ElementCount::isKnownLT(EpilogueUserVF, UserVF)) {
        CM.collectNonVectorizedAndSetWideningDecisions(EpilogueUserVF);
        buildVPlansWithVPRecipes(EpilogueUserVF, EpilogueUserVF);
      }
      buildVPlansWithVPRecipes(UserVF, UserVF);
      if (!VPlans.empty() && VPlans.back()->getSingleVF() == UserVF) {
        // For scalar VF, skip VPlan cost check as VPlan cost is designed for
        // vector VFs only.
        if (UserVF.isScalar() ||
            cost(*VPlans.back(), UserVF, /*RU=*/nullptr).isValid()) {
          LLVM_DEBUG(dbgs() << "LV: Using user VF " << UserVF << ".\n");
          LLVM_DEBUG(printPlans(dbgs()));
          return;
        }
      }
      VPlans.clear();
      reportVectorizationInfo("UserVF ignored because of invalid costs.",
                              "InvalidCost", ORE, OrigLoop);
    }
  }

  // Collect the Vectorization Factor Candidates.
  SmallVector<ElementCount> VFCandidates;
  for (auto VF = ElementCount::getFixed(1);
       ElementCount::isKnownLE(VF, MaxFactors.FixedVF); VF *= 2)
    VFCandidates.push_back(VF);
  for (auto VF = ElementCount::getScalable(1);
       ElementCount::isKnownLE(VF, MaxFactors.ScalableVF); VF *= 2)
    VFCandidates.push_back(VF);

  Config.collectInLoopReductions();
  for (const auto &VF : VFCandidates) {
    // Collect Uniform and Scalar instructions after vectorization with VF.
    CM.collectNonVectorizedAndSetWideningDecisions(VF);
  }

  buildVPlansWithVPRecipes(ElementCount::getFixed(1), MaxFactors.FixedVF);
  buildVPlansWithVPRecipes(ElementCount::getScalable(1), MaxFactors.ScalableVF);

  LLVM_DEBUG(printPlans(dbgs()));
}

InstructionCost VPCostContext::getLegacyCost(Instruction *UI,
                                             ElementCount VF) const {
  InstructionCost Cost = CM.getInstructionCost(UI, VF);
  if (Cost.isValid() && ForceTargetInstructionCost.getNumOccurrences())
    return InstructionCost(ForceTargetInstructionCost);
  return Cost;
}

bool VPCostContext::skipCostComputation(Instruction *UI, bool IsVector) const {
  return CM.ValuesToIgnore.contains(UI) ||
         (IsVector && CM.VecValuesToIgnore.contains(UI)) ||
         SkipCostComputation.contains(UI);
}

uint64_t VPCostContext::getPredBlockCostDivisor(BasicBlock *BB) const {
  return CM.getPredBlockCostDivisor(CostKind, BB);
}

InstructionCost
LoopVectorizationPlanner::precomputeCosts(VPlan &Plan, ElementCount VF,
                                          VPCostContext &CostCtx) const {
  InstructionCost Cost;
  // Cost modeling for inductions is inaccurate in the legacy cost model
  // compared to the recipes that are generated. To match here initially during
  // VPlan cost model bring up directly use the induction costs from the legacy
  // cost model. Note that we do this as pre-processing; the VPlan may not have
  // any recipes associated with the original induction increment instruction
  // and may replace truncates with VPWidenIntOrFpInductionRecipe. We precompute
  // the cost of induction phis and increments (both that are represented by
  // recipes and those that are not), to avoid distinguishing between them here,
  // and skip all recipes that represent induction phis and increments (the
  // former case) later on, if they exist, to avoid counting them twice.
  // Similarly we pre-compute the cost of any optimized truncates.
  // TODO: Switch to more accurate costing based on VPlan.
  for (const auto &[IV, IndDesc] : Legal->getInductionVars()) {
    Instruction *IVInc = cast<Instruction>(
        IV->getIncomingValueForBlock(OrigLoop->getLoopLatch()));
    SmallVector<Instruction *> IVInsts = {IVInc};
    for (unsigned I = 0; I != IVInsts.size(); I++) {
      for (Value *Op : IVInsts[I]->operands()) {
        auto *OpI = dyn_cast<Instruction>(Op);
        if (Op == IV || !OpI || !OrigLoop->contains(OpI) || !Op->hasOneUse())
          continue;
        IVInsts.push_back(OpI);
      }
    }
    IVInsts.push_back(IV);
    for (User *U : IV->users()) {
      auto *CI = cast<Instruction>(U);
      if (!CostCtx.CM.isOptimizableIVTruncate(CI, VF))
        continue;
      IVInsts.push_back(CI);
    }

    // If the vector loop gets executed exactly once with the given VF, ignore
    // the costs of comparison and induction instructions, as they'll get
    // simplified away.
    // TODO: Remove this code after stepping away from the legacy cost model and
    // adding code to simplify VPlans before calculating their costs.
    auto TC = getSmallConstantTripCount(PSE.getSE(), OrigLoop);
    if (TC == VF && !CM.foldTailByMasking())
      addFullyUnrolledInstructionsToIgnore(OrigLoop, Legal->getInductionVars(),
                                           CostCtx.SkipCostComputation);

    for (Instruction *IVInst : IVInsts) {
      if (CostCtx.skipCostComputation(IVInst, VF.isVector()))
        continue;
      InstructionCost InductionCost = CostCtx.getLegacyCost(IVInst, VF);
      LLVM_DEBUG({
        dbgs() << "Cost of " << InductionCost << " for VF " << VF
               << ": induction instruction " << *IVInst << "\n";
      });
      Cost += InductionCost;
      CostCtx.SkipCostComputation.insert(IVInst);
    }
  }

  /// Compute the cost of all exiting conditions of the loop using the legacy
  /// cost model. This is to match the legacy behavior, which adds the cost of
  /// all exit conditions. Note that this over-estimates the cost, as there will
  /// be a single condition to control the vector loop.
  SmallVector<BasicBlock *> Exiting;
  CM.TheLoop->getExitingBlocks(Exiting);
  SetVector<Instruction *> ExitInstrs;
  // Collect all exit conditions.
  for (BasicBlock *EB : Exiting) {
    auto *Term = dyn_cast<CondBrInst>(EB->getTerminator());
    if (!Term || CostCtx.skipCostComputation(Term, VF.isVector()))
      continue;
    if (auto *CondI = dyn_cast<Instruction>(Term->getOperand(0))) {
      ExitInstrs.insert(CondI);
    }
  }
  // Compute the cost of all instructions only feeding the exit conditions.
  for (unsigned I = 0; I != ExitInstrs.size(); ++I) {
    Instruction *CondI = ExitInstrs[I];
    if (!OrigLoop->contains(CondI) ||
        !CostCtx.SkipCostComputation.insert(CondI).second)
      continue;
    InstructionCost CondICost = CostCtx.getLegacyCost(CondI, VF);
    LLVM_DEBUG({
      dbgs() << "Cost of " << CondICost << " for VF " << VF
             << ": exit condition instruction " << *CondI << "\n";
    });
    Cost += CondICost;
    for (Value *Op : CondI->operands()) {
      auto *OpI = dyn_cast<Instruction>(Op);
      if (!OpI || CostCtx.skipCostComputation(OpI, VF.isVector()) ||
          any_of(OpI->users(), [&ExitInstrs](User *U) {
            return !ExitInstrs.contains(cast<Instruction>(U));
          }))
        continue;
      ExitInstrs.insert(OpI);
    }
  }

  // Pre-compute the costs for branches except for the backedge, as the number
  // of replicate regions in a VPlan may not directly match the number of
  // branches, which would lead to different decisions.
  // TODO: Compute cost of branches for each replicate region in the VPlan,
  // which is more accurate than the legacy cost model.
  for (BasicBlock *BB : OrigLoop->blocks()) {
    if (CostCtx.skipCostComputation(BB->getTerminator(), VF.isVector()))
      continue;
    CostCtx.SkipCostComputation.insert(BB->getTerminator());
    if (BB == OrigLoop->getLoopLatch())
      continue;
    auto BranchCost = CostCtx.getLegacyCost(BB->getTerminator(), VF);
    Cost += BranchCost;
  }

  // Don't apply special costs when instruction cost is forced to make sure the
  // forced cost is used for each recipe.
  if (ForceTargetInstructionCost.getNumOccurrences())
    return Cost;

  // Pre-compute costs for instructions that are forced-scalar or profitable to
  // scalarize. For most such instructions, their scalarization costs are
  // accounted for here using the legacy cost model. However, some opcodes
  // are excluded from these precomputed scalarization costs and are instead
  // modeled later by the VPlan cost model (see UseVPlanCostModel below).
  for (Instruction *ForcedScalar : CM.ForcedScalars[VF]) {
    if (CostCtx.skipCostComputation(ForcedScalar, VF.isVector()))
      continue;
    CostCtx.SkipCostComputation.insert(ForcedScalar);
    InstructionCost ForcedCost = CostCtx.getLegacyCost(ForcedScalar, VF);
    LLVM_DEBUG({
      dbgs() << "Cost of " << ForcedCost << " for VF " << VF
             << ": forced scalar " << *ForcedScalar << "\n";
    });
    Cost += ForcedCost;
  }

  auto UseVPlanCostModel = [](Instruction *I) -> bool {
    switch (I->getOpcode()) {
    case Instruction::SDiv:
    case Instruction::UDiv:
    case Instruction::SRem:
    case Instruction::URem:
      return true;
    default:
      return false;
    }
  };
  for (const auto &[Scalarized, ScalarCost] : CM.InstsToScalarize[VF]) {
    if (UseVPlanCostModel(Scalarized) ||
        CostCtx.skipCostComputation(Scalarized, VF.isVector()))
      continue;
    CostCtx.SkipCostComputation.insert(Scalarized);
    LLVM_DEBUG({
      dbgs() << "Cost of " << ScalarCost << " for VF " << VF
             << ": profitable to scalarize " << *Scalarized << "\n";
    });
    Cost += ScalarCost;
  }

  return Cost;
}

InstructionCost LoopVectorizationPlanner::cost(VPlan &Plan, ElementCount VF,
                                               VPRegisterUsage *RU) const {
  VPCostContext CostCtx(CM.TTI, *CM.TLI, Plan, CM, Config.CostKind, PSE,
                        OrigLoop);
  InstructionCost Cost = precomputeCosts(Plan, VF, CostCtx);

  // Now compute and add the VPlan-based cost.
  Cost += Plan.cost(VF, CostCtx);

  // Add the cost of spills due to excess register usage
  if (RU && Config.shouldConsiderRegPressureForVF(VF))
    Cost += RU->spillCost(CM.TTI, Config.CostKind, ForceTargetNumVectorRegs);

#ifndef NDEBUG
  unsigned EstimatedWidth =
      estimateElementCount(VF, Config.getVScaleForTuning());
  LLVM_DEBUG(dbgs() << "Cost for VF " << VF << ": " << Cost
                    << " (Estimated cost per lane: ");
  if (Cost.isValid()) {
    double CostPerLane = double(Cost.getValue()) / EstimatedWidth;
    LLVM_DEBUG(dbgs() << format("%.1f", CostPerLane));
  } else /* No point dividing an invalid cost - it will still be invalid */
    LLVM_DEBUG(dbgs() << "Invalid");
  LLVM_DEBUG(dbgs() << ")\n");
#endif
  return Cost;
}

std::pair<VectorizationFactor, VPlan *>
LoopVectorizationPlanner::computeBestVF() {
  if (VPlans.empty())
    return {VectorizationFactor::Disabled(), nullptr};
  // If there is a single VPlan with a single VF, return it directly.
  VPlan &FirstPlan = *VPlans[0];
  ElementCount UserVF = Hints.getWidth();
  if (hasPlanWithVF(UserVF)) {
    if (VPlans.size() == 1) {
      assert(FirstPlan.getSingleVF() == UserVF &&
             "UserVF must match single VF");
      return {VectorizationFactor(FirstPlan.getSingleVF(), 0, 0), &FirstPlan};
    }
    if (EpilogueVectorizationForceVF > 1) {
      assert(VPlans.size() == 2 && "Must have exactly 2 VPlans built");
      assert(VPlans[0]->getSingleVF() ==
                 ElementCount::getFixed(EpilogueVectorizationForceVF) &&
             "expected first plan to be for the forced epilogue VF");
      assert(VPlans[1]->getSingleVF() == UserVF &&
             "expected second plan to be for the forced UserVF");
      return {VectorizationFactor(UserVF, 0, 0), VPlans[1].get()};
    }
  }

  LLVM_DEBUG(dbgs() << "LV: Computing best VF using cost kind: "
                    << (Config.CostKind == TTI::TCK_RecipThroughput
                            ? "Reciprocal Throughput\n"
                        : Config.CostKind == TTI::TCK_Latency
                            ? "Instruction Latency\n"
                        : Config.CostKind == TTI::TCK_CodeSize ? "Code Size\n"
                        : Config.CostKind == TTI::TCK_SizeAndLatency
                            ? "Code Size and Latency\n"
                            : "Unknown\n"));

  ElementCount ScalarVF = ElementCount::getFixed(1);
  assert(FirstPlan.hasVF(ScalarVF) &&
         "More than a single plan/VF w/o any plan having scalar VF");

  // TODO: Compute scalar cost using VPlan-based cost model.
  InstructionCost ScalarCost = CM.expectedCost(ScalarVF);
  LLVM_DEBUG(dbgs() << "LV: Scalar loop costs: " << ScalarCost << ".\n");
  VectorizationFactor ScalarFactor(ScalarVF, ScalarCost, ScalarCost);
  VectorizationFactor BestFactor = ScalarFactor;

  bool ForceVectorization = Hints.getForce() == LoopVectorizeHints::FK_Enabled;
  if (ForceVectorization) {
    // Ignore scalar width, because the user explicitly wants vectorization.
    // Initialize cost to max so that VF = 2 is, at least, chosen during cost
    // evaluation.
    BestFactor.Cost = InstructionCost::getMax();
  }

  VPlan *PlanForBestVF = &FirstPlan;

  for (auto &P : VPlans) {
    ArrayRef<ElementCount> VFs(P->vectorFactors().begin(),
                               P->vectorFactors().end());

    SmallVector<VPRegisterUsage, 8> RUs;
    bool ConsiderRegPressure = any_of(VFs, [this](ElementCount VF) {
      return Config.shouldConsiderRegPressureForVF(VF);
    });
    if (ConsiderRegPressure)
      RUs = calculateRegisterUsageForPlan(*P, VFs, TTI, CM.ValuesToIgnore);

    for (unsigned I = 0; I < VFs.size(); I++) {
      ElementCount VF = VFs[I];
      if (VF.isScalar())
        continue;
      if (!ForceVectorization && !willGenerateVectors(*P, VF, TTI)) {
        LLVM_DEBUG(
            dbgs()
            << "LV: Not considering vector loop of width " << VF
            << " because it will not generate any vector instructions.\n");
        continue;
      }
      if (Config.OptForSize && !ForceVectorization && hasReplicatorRegion(*P)) {
        LLVM_DEBUG(
            dbgs()
            << "LV: Not considering vector loop of width " << VF
            << " because it would cause replicated blocks to be generated,"
            << " which isn't allowed when optimizing for size.\n");
        continue;
      }

      InstructionCost Cost =
          cost(*P, VF, ConsiderRegPressure ? &RUs[I] : nullptr);
      VectorizationFactor CurrentFactor(VF, Cost, ScalarCost);

      if (isMoreProfitable(CurrentFactor, BestFactor, P->hasScalarTail())) {
        BestFactor = CurrentFactor;
        PlanForBestVF = P.get();
      }

      // If profitable add it to ProfitableVF list.
      if (isMoreProfitable(CurrentFactor, ScalarFactor, P->hasScalarTail()))
        ProfitableVFs.push_back(CurrentFactor);
    }
  }

  VPlan &BestPlan = *PlanForBestVF;

  assert((BestFactor.Width.isScalar() || BestFactor.ScalarCost > 0) &&
         "when vectorizing, the scalar cost must be computed.");

  LLVM_DEBUG(dbgs() << "LV: Selecting VF: " << BestFactor.Width << ".\n");
  return {BestFactor, &BestPlan};
}

DenseMap<const SCEV *, Value *> LoopVectorizationPlanner::executePlan(
    ElementCount BestVF, unsigned BestUF, VPlan &BestVPlan,
    InnerLoopVectorizer &ILV, DominatorTree *DT,
    EpilogueVectorizationKind EpilogueVecKind) {
  assert(BestVPlan.hasVF(BestVF) &&
         "Trying to execute plan with unsupported VF");
  assert(BestVPlan.hasUF(BestUF) &&
         "Trying to execute plan with unsupported UF");
  if (BestVPlan.hasEarlyExit())
    ++LoopsEarlyExitVectorized;

  VPlanTransforms::replaceWideCanonicalIVWithWideIV(
      BestVPlan, *PSE.getSE(), CM.TTI, Config.CostKind, BestVF, BestUF,
      CM.ValuesToIgnore);
  // TODO: Move to VPlan transform stage once the transition to the VPlan-based
  // cost model is complete for better cost estimates.
  RUN_VPLAN_PASS(VPlanTransforms::unrollByUF, BestVPlan, BestUF);
  RUN_VPLAN_PASS(VPlanTransforms::materializePacksAndUnpacks, BestVPlan);
  RUN_VPLAN_PASS(VPlanTransforms::materializeBroadcasts, BestVPlan);
  RUN_VPLAN_PASS(VPlanTransforms::replicateByVF, BestVPlan, BestVF);
  bool HasBranchWeights =
      hasBranchWeightMD(*OrigLoop->getLoopLatch()->getTerminator());
  if (HasBranchWeights) {
    std::optional<unsigned> VScale = Config.getVScaleForTuning();
    RUN_VPLAN_PASS(VPlanTransforms::addBranchWeightToMiddleTerminator,
                   BestVPlan, BestVF, VScale);
  }

  // Retrieving VectorPH now when it's easier while VPlan still has Regions.
  VPBasicBlock *VectorPH = cast<VPBasicBlock>(BestVPlan.getVectorPreheader());

  RUN_VPLAN_PASS(VPlanTransforms::materializeConstantVectorTripCount, BestVPlan,
                 BestVF, BestUF, PSE);
  RUN_VPLAN_PASS(VPlanTransforms::optimizeForVFAndUF, BestVPlan, BestVF, BestUF,
                 PSE);
  RUN_VPLAN_PASS(VPlanTransforms::simplifyRecipes, BestVPlan);
  if (EpilogueVecKind == EpilogueVectorizationKind::None)
    RUN_VPLAN_PASS(VPlanTransforms::removeBranchOnConst, BestVPlan,
                   /*OnlyLatches=*/false);
  if (BestVPlan.getEntry()->getSingleSuccessor() ==
      BestVPlan.getScalarPreheader()) {
    // TODO: The vector loop would be dead, should not even try to vectorize.
    ORE->emit([&]() {
      return OptimizationRemarkAnalysis(DEBUG_TYPE, "VectorizationDead",
                                        OrigLoop->getStartLoc(),
                                        OrigLoop->getHeader())
             << "Created vector loop never executes due to insufficient trip "
                "count.";
    });
    return DenseMap<const SCEV *, Value *>();
  }

  RUN_VPLAN_PASS(VPlanTransforms::removeDeadRecipes, BestVPlan);

  RUN_VPLAN_PASS(VPlanTransforms::convertToConcreteRecipes, BestVPlan);
  // Convert the exit condition to AVLNext == 0 for EVL tail folded loops.
  RUN_VPLAN_PASS(VPlanTransforms::convertEVLExitCond, BestVPlan);
  // Regions are dissolved after optimizing for VF and UF, which completely
  // removes unneeded loop regions first.
  VPlanTransforms::dissolveLoopRegions(BestVPlan);
  // Expand BranchOnTwoConds after dissolution, when latch has direct access to
  // its successors.
  VPlanTransforms::expandBranchOnTwoConds(BestVPlan);
  // Convert loops with variable-length stepping after regions are dissolved.
  VPlanTransforms::convertToVariableLengthStep(BestVPlan);
  // Remove dead back-edges for single-iteration loops with BranchOnCond(true).
  // Only process loop latches to avoid removing edges from the middle block,
  // which may be needed for epilogue vectorization.
  VPlanTransforms::removeBranchOnConst(BestVPlan, /*OnlyLatches=*/true);
  VPlanTransforms::materializeBackedgeTakenCount(BestVPlan, VectorPH);
  std::optional<uint64_t> MaxRuntimeStep;
  if (auto MaxVScale = getMaxVScale(*CM.TheFunction, CM.TTI))
    MaxRuntimeStep = uint64_t(*MaxVScale) * BestVF.getKnownMinValue() * BestUF;
  VPlanTransforms::materializeVectorTripCount(
      BestVPlan, VectorPH, CM.foldTailByMasking(),
      CM.requiresScalarEpilogue(BestVF.isVector()), &BestVPlan.getVFxUF(),
      MaxRuntimeStep);
  VPlanTransforms::materializeFactors(BestVPlan, VectorPH, BestVF);
  VPlanTransforms::cse(BestVPlan);
  VPlanTransforms::simplifyRecipes(BestVPlan);
  VPlanTransforms::simplifyKnownEVL(BestVPlan, BestVF, PSE);

  // 0. Generate SCEV-dependent code in the entry, including TripCount, before
  // making any changes to the CFG.
  DenseMap<const SCEV *, Value *> ExpandedSCEVs =
      VPlanTransforms::expandSCEVs(BestVPlan, *PSE.getSE());

  // Perform the actual loop transformation.
  VPTransformState State(&TTI, BestVF, LI, DT, ILV.AC, ILV.Builder, &BestVPlan,
                         OrigLoop->getParentLoop(),
                         Legal->getWidestInductionType());

#ifdef EXPENSIVE_CHECKS
  assert(DT->verify(DominatorTree::VerificationLevel::Fast));
#endif

  // 1. Set up the skeleton for vectorization, including vector pre-header and
  // middle block. The vector loop is created during VPlan execution.
  State.CFG.PrevBB = ILV.createVectorizedLoopSkeleton();
  if (VPBasicBlock *ScalarPH = BestVPlan.getScalarPreheader())
    replaceVPBBWithIRVPBB(ScalarPH, State.CFG.PrevBB->getSingleSuccessor(),
                          &BestVPlan);
  VPlanTransforms::removeDeadRecipes(BestVPlan);

  assert(verifyVPlanIsValid(BestVPlan) && "final VPlan is invalid");

  // After vectorization, the exit blocks of the original loop will have
  // additional predecessors. Invalidate SCEVs for the exit phis in case SE
  // looked through single-entry phis.
  ScalarEvolution &SE = *PSE.getSE();
  for (VPIRBasicBlock *Exit : BestVPlan.getExitBlocks()) {
    if (!Exit->hasPredecessors())
      continue;
    for (VPRecipeBase &PhiR : Exit->phis())
      SE.forgetLcssaPhiWithNewPredecessor(OrigLoop,
                                          &cast<VPIRPhi>(PhiR).getIRPhi());
  }
  // Forget the original loop and block dispositions.
  SE.forgetLoop(OrigLoop);
  SE.forgetBlockAndLoopDispositions();

  ILV.printDebugTracesAtStart();

  //===------------------------------------------------===//
  //
  // Notice: any optimization or new instruction that go
  // into the code below should also be implemented in
  // the cost-model.
  //
  //===------------------------------------------------===//

  // Retrieve loop information before executing the plan, which may remove the
  // original loop, if it becomes unreachable.
  MDNode *LID = OrigLoop->getLoopID();
  unsigned OrigLoopInvocationWeight = 0;
  std::optional<unsigned> OrigAverageTripCount =
      getLoopEstimatedTripCount(OrigLoop, &OrigLoopInvocationWeight);

  BestVPlan.execute(&State);

  // 2.6. Maintain Loop Hints
  // Keep all loop hints from the original loop on the vector loop (we'll
  // replace the vectorizer-specific hints below).
  VPBasicBlock *HeaderVPBB = vputils::getFirstLoopHeader(BestVPlan, State.VPDT);
  // Add metadata to disable runtime unrolling a scalar loop when there
  // are no runtime checks about strides and memory. A scalar loop that is
  // rarely used is not worth unrolling.
  bool DisableRuntimeUnroll = !ILV.RTChecks.hasChecks() && !BestVF.isScalar();
  updateLoopMetadataAndProfileInfo(
      HeaderVPBB ? LI->getLoopFor(State.CFG.VPBB2IRBB.lookup(HeaderVPBB))
                 : nullptr,
      HeaderVPBB, BestVPlan,
      EpilogueVecKind == EpilogueVectorizationKind::Epilogue, LID,
      OrigAverageTripCount, OrigLoopInvocationWeight,
      estimateElementCount(BestVF * BestUF, Config.getVScaleForTuning()),
      DisableRuntimeUnroll);

  // 3. Fix the vectorized code: take care of header phi's, live-outs,
  //    predication, updating analyses.
  ILV.fixVectorizedLoop(State);

  ILV.printDebugTracesAtEnd();

  return ExpandedSCEVs;
}

//===--------------------------------------------------------------------===//
// EpilogueVectorizerMainLoop
//===--------------------------------------------------------------------===//

void EpilogueVectorizerMainLoop::printDebugTracesAtStart() {
  LLVM_DEBUG({
    dbgs() << "Create Skeleton for epilogue vectorized loop (first pass)\n"
           << "Main Loop VF:" << EPI.MainLoopVF
           << ", Main Loop UF:" << EPI.MainLoopUF
           << ", Epilogue Loop VF:" << EPI.EpilogueVF
           << ", Epilogue Loop UF:" << EPI.EpilogueUF << "\n";
  });
}

void EpilogueVectorizerMainLoop::printDebugTracesAtEnd() {
  DEBUG_WITH_TYPE(VerboseDebug, {
    dbgs() << "intermediate fn:\n"
           << *OrigLoop->getHeader()->getParent() << "\n";
  });
}

//===--------------------------------------------------------------------===//
// EpilogueVectorizerEpilogueLoop
//===--------------------------------------------------------------------===//

/// This function creates a new scalar preheader, using the previous one as
/// entry block to the epilogue VPlan. The minimum iteration check is being
/// represented in VPlan.
BasicBlock *EpilogueVectorizerEpilogueLoop::createVectorizedLoopSkeleton() {
  BasicBlock *NewScalarPH = createScalarPreheader("vec.epilog.");
  BasicBlock *OriginalScalarPH = NewScalarPH->getSinglePredecessor();
  OriginalScalarPH->setName("vec.epilog.iter.check");
  VPIRBasicBlock *NewEntry = Plan.createVPIRBasicBlock(OriginalScalarPH);
  VPBasicBlock *OldEntry = Plan.getEntry();
  for (auto &R : make_early_inc_range(*OldEntry)) {
    // Skip moving VPIRInstructions (including VPIRPhis), which are unmovable by
    // defining.
    if (isa<VPIRInstruction>(&R))
      continue;
    R.moveBefore(*NewEntry, NewEntry->end());
  }

  VPBlockUtils::reassociateBlocks(OldEntry, NewEntry);
  Plan.setEntry(NewEntry);
  // OldEntry is now dead and will be cleaned up when the plan gets destroyed.

  return OriginalScalarPH;
}

void EpilogueVectorizerEpilogueLoop::printDebugTracesAtStart() {
  LLVM_DEBUG({
    dbgs() << "Create Skeleton for epilogue vectorized loop (second pass)\n"
           << "Epilogue Loop VF:" << EPI.EpilogueVF
           << ", Epilogue Loop UF:" << EPI.EpilogueUF << "\n";
  });
}

void EpilogueVectorizerEpilogueLoop::printDebugTracesAtEnd() {
  DEBUG_WITH_TYPE(VerboseDebug, {
    dbgs() << "final fn:\n" << *OrigLoop->getHeader()->getParent() << "\n";
  });
}

VPRecipeBase *VPRecipeBuilder::tryToWidenMemory(VPInstruction *VPI,
                                                VFRange &Range) {
  assert((VPI->getOpcode() == Instruction::Load ||
          VPI->getOpcode() == Instruction::Store) &&
         "Must be called with either a load or store");
  Instruction *I = VPI->getUnderlyingInstr();

  auto WillWiden = [&](ElementCount VF) -> bool {
    LoopVectorizationCostModel::InstWidening Decision =
        CM.getWideningDecision(I, VF);
    assert(Decision != LoopVectorizationCostModel::CM_Unknown &&
           "CM decision should be taken at this point.");
    if (Decision == LoopVectorizationCostModel::CM_Interleave)
      return true;
    if (CM.isScalarAfterVectorization(I, VF) ||
        CM.isProfitableToScalarize(I, VF))
      return false;
    return Decision != LoopVectorizationCostModel::CM_Scalarize;
  };

  if (!LoopVectorizationPlanner::getDecisionAndClampRange(WillWiden, Range))
    return nullptr;

  // If a mask is not required, drop it - use unmasked version for safe loads.
  // TODO: Determine if mask is needed in VPlan.
  VPValue *Mask = CM.isMaskRequired(I) ? VPI->getMask() : nullptr;

  // Determine if the pointer operand of the access is either consecutive or
  // reverse consecutive.
  LoopVectorizationCostModel::InstWidening Decision =
      CM.getWideningDecision(I, Range.Start);
  bool Reverse = Decision == LoopVectorizationCostModel::CM_Widen_Reverse;
  bool Consecutive =
      Reverse || Decision == LoopVectorizationCostModel::CM_Widen;

  VPValue *Ptr = VPI->getOpcode() == Instruction::Load ? VPI->getOperand(0)
                                                       : VPI->getOperand(1);
  if (Consecutive) {
    auto *GEP = dyn_cast<GetElementPtrInst>(
        Ptr->getUnderlyingValue()->stripPointerCasts());
    VPSingleDefRecipe *VectorPtr;
    if (Reverse) {
      // When folding the tail, we may compute an address that we don't in the
      // original scalar loop: drop the GEP no-wrap flags in this case.
      // Otherwise preserve existing flags without no-unsigned-wrap, as we will
      // emit negative indices.
      GEPNoWrapFlags Flags =
          CM.foldTailByMasking() || !GEP
              ? GEPNoWrapFlags::none()
              : GEP->getNoWrapFlags().withoutNoUnsignedWrap();
      VectorPtr = new VPVectorEndPointerRecipe(
          Ptr, &Plan.getVF(), getLoadStoreType(I),
          /*Stride*/ -1, Flags, VPI->getDebugLoc());
    } else {
      VectorPtr = new VPVectorPointerRecipe(Ptr, getLoadStoreType(I),
                                            GEP ? GEP->getNoWrapFlags()
                                                : GEPNoWrapFlags::none(),
                                            VPI->getDebugLoc());
    }
    Builder.setInsertPoint(VPI);
    Builder.insert(VectorPtr);
    Ptr = VectorPtr;
  }

  if (Reverse && Mask)
    Mask = Builder.createNaryOp(VPInstruction::Reverse, Mask, I->getDebugLoc());

  if (VPI->getOpcode() == Instruction::Load) {
    auto *Load = cast<LoadInst>(I);
    auto *LoadR = new VPWidenLoadRecipe(*Load, Ptr, Mask, Consecutive, *VPI,
                                        Load->getDebugLoc());
    if (Reverse) {
      Builder.insert(LoadR);
      return new VPInstruction(VPInstruction::Reverse, LoadR, {}, {},
                               LoadR->getDebugLoc());
    }
    return LoadR;
  }

  StoreInst *Store = cast<StoreInst>(I);
  VPValue *StoredVal = VPI->getOperand(0);
  if (Reverse)
    StoredVal = Builder.createNaryOp(VPInstruction::Reverse, StoredVal,
                                     Store->getDebugLoc());
  return new VPWidenStoreRecipe(*Store, Ptr, StoredVal, Mask, Consecutive, *VPI,
                                Store->getDebugLoc());
}

VPWidenIntOrFpInductionRecipe *
VPRecipeBuilder::tryToOptimizeInductionTruncate(VPInstruction *VPI,
                                                VFRange &Range) {
  auto *I = cast<TruncInst>(VPI->getUnderlyingInstr());
  // Optimize the special case where the source is a constant integer
  // induction variable. Notice that we can only optimize the 'trunc' case
  // because (a) FP conversions lose precision, (b) sext/zext may wrap, and
  // (c) other casts depend on pointer size.

  // Determine whether \p K is a truncation based on an induction variable that
  // can be optimized.
  if (!LoopVectorizationPlanner::getDecisionAndClampRange(
          bind_front(&LoopVectorizationCostModel::isOptimizableIVTruncate, CM,
                     I),
          Range))
    return nullptr;

  auto *WidenIV = cast<VPWidenIntOrFpInductionRecipe>(
      VPI->getOperand(0)->getDefiningRecipe());
  PHINode *Phi = WidenIV->getPHINode();
  VPIRValue *Start = WidenIV->getStartValue();
  const InductionDescriptor &IndDesc = WidenIV->getInductionDescriptor();

  // Wrap flags from the original induction do not apply to the truncated type,
  // so do not propagate them.
  VPIRFlags Flags = VPIRFlags::WrapFlagsTy(false, false);
  VPValue *Step =
      vputils::getOrCreateVPValueForSCEVExpr(Plan, IndDesc.getStep());
  return new VPWidenIntOrFpInductionRecipe(
      Phi, Start, Step, &Plan.getVF(), IndDesc, I, Flags, VPI->getDebugLoc());
}

VPSingleDefRecipe *VPRecipeBuilder::tryToWidenCall(VPInstruction *VPI,
                                                   VFRange &Range) {
  CallInst *CI = cast<CallInst>(VPI->getUnderlyingInstr());
  bool IsPredicated = LoopVectorizationPlanner::getDecisionAndClampRange(
      [this, CI](ElementCount VF) {
        return CM.isScalarWithPredication(CI, VF);
      },
      Range);

  if (IsPredicated)
    return nullptr;

  Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);
  if (ID && (ID == Intrinsic::assume || ID == Intrinsic::lifetime_end ||
             ID == Intrinsic::lifetime_start || ID == Intrinsic::sideeffect ||
             ID == Intrinsic::pseudoprobe ||
             ID == Intrinsic::experimental_noalias_scope_decl))
    return nullptr;

  SmallVector<VPValue *, 4> Ops(VPI->op_begin(),
                                VPI->op_begin() + CI->arg_size());

  // Is it beneficial to perform intrinsic call compared to lib call?
  bool ShouldUseVectorIntrinsic =
      ID && LoopVectorizationPlanner::getDecisionAndClampRange(
                [&](ElementCount VF) -> bool {
                  return CM.getCallWideningDecision(CI, VF).Kind ==
                         LoopVectorizationCostModel::CM_IntrinsicCall;
                },
                Range);
  if (ShouldUseVectorIntrinsic)
    return new VPWidenIntrinsicRecipe(*CI, ID, Ops, CI->getType(), *VPI, *VPI,
                                      VPI->getDebugLoc());

  Function *Variant = nullptr;
  std::optional<unsigned> MaskPos;
  // Is better to call a vectorized version of the function than to to scalarize
  // the call?
  auto ShouldUseVectorCall = LoopVectorizationPlanner::getDecisionAndClampRange(
      [&](ElementCount VF) -> bool {
        // The following case may be scalarized depending on the VF.
        // The flag shows whether we can use a usual Call for vectorized
        // version of the instruction.

        // If we've found a variant at a previous VF, then stop looking. A
        // vectorized variant of a function expects input in a certain shape
        // -- basically the number of input registers, the number of lanes
        // per register, and whether there's a mask required.
        // We store a pointer to the variant in the VPWidenCallRecipe, so
        // once we have an appropriate variant it's only valid for that VF.
        // This will force a different vplan to be generated for each VF that
        // finds a valid variant.
        if (Variant)
          return false;
        LoopVectorizationCostModel::CallWideningDecision Decision =
            CM.getCallWideningDecision(CI, VF);
        if (Decision.Kind == LoopVectorizationCostModel::CM_VectorCall) {
          Variant = Decision.Variant;
          MaskPos = Decision.MaskPos;
          return true;
        }

        return false;
      },
      Range);
  if (ShouldUseVectorCall) {
    if (MaskPos.has_value()) {
      // We have 2 cases that would require a mask:
      //   1) The call needs to be predicated, either due to a conditional
      //      in the scalar loop or use of an active lane mask with
      //      tail-folding, and we use the appropriate mask for the block.
      //   2) No mask is required for the call instruction, but the only
      //      available vector variant at this VF requires a mask, so we
      //      synthesize an all-true mask.
      VPValue *Mask = VPI->isMasked() ? VPI->getMask() : Plan.getTrue();

      Ops.insert(Ops.begin() + *MaskPos, Mask);
    }

    Ops.push_back(VPI->getOperand(VPI->getNumOperandsWithoutMask() - 1));
    return new VPWidenCallRecipe(CI, Variant, Ops, *VPI, *VPI,
                                 VPI->getDebugLoc());
  }

  return nullptr;
}

bool VPRecipeBuilder::shouldWiden(Instruction *I, VFRange &Range) const {
  assert((!isa<UncondBrInst, CondBrInst, PHINode, LoadInst, StoreInst>(I)) &&
         "Instruction should have been handled earlier");
  // Instruction should be widened, unless it is scalar after vectorization,
  // scalarization is profitable or it is predicated.
  auto WillScalarize = [this, I](ElementCount VF) -> bool {
    return CM.isScalarAfterVectorization(I, VF) ||
           CM.isProfitableToScalarize(I, VF) ||
           CM.isScalarWithPredication(I, VF);
  };
  return !LoopVectorizationPlanner::getDecisionAndClampRange(WillScalarize,
                                                             Range);
}

VPWidenRecipe *VPRecipeBuilder::tryToWiden(VPInstruction *VPI) {
  auto *I = VPI->getUnderlyingInstr();
  switch (VPI->getOpcode()) {
  default:
    return nullptr;
  case Instruction::SDiv:
  case Instruction::UDiv:
  case Instruction::SRem:
  case Instruction::URem: {
    // If not provably safe, use a select to form a safe divisor before widening the
    // div/rem operation itself.  Otherwise fall through to general handling below.
    if (CM.isPredicatedInst(I)) {
      SmallVector<VPValue *> Ops(VPI->operandsWithoutMask());
      VPValue *Mask = VPI->getMask();
      VPValue *One = Plan.getConstantInt(I->getType(), 1u);
      auto *SafeRHS =
          Builder.createSelect(Mask, Ops[1], One, VPI->getDebugLoc());
      Ops[1] = SafeRHS;
      return new VPWidenRecipe(*I, Ops, *VPI, *VPI, VPI->getDebugLoc());
    }
    [[fallthrough]];
  }
  case Instruction::Add:
  case Instruction::And:
  case Instruction::AShr:
  case Instruction::FAdd:
  case Instruction::FCmp:
  case Instruction::FDiv:
  case Instruction::FMul:
  case Instruction::FNeg:
  case Instruction::FRem:
  case Instruction::FSub:
  case Instruction::ICmp:
  case Instruction::LShr:
  case Instruction::Mul:
  case Instruction::Or:
  case Instruction::Select:
  case Instruction::Shl:
  case Instruction::Sub:
  case Instruction::Xor:
  case Instruction::Freeze:
    return new VPWidenRecipe(*I, VPI->operandsWithoutMask(), *VPI, *VPI,
                             VPI->getDebugLoc());
  case Instruction::ExtractValue: {
    SmallVector<VPValue *> NewOps(VPI->operandsWithoutMask());
    auto *EVI = cast<ExtractValueInst>(I);
    assert(EVI->getNumIndices() == 1 && "Expected one extractvalue index");
    unsigned Idx = EVI->getIndices()[0];
    NewOps.push_back(Plan.getConstantInt(32, Idx));
    return new VPWidenRecipe(*I, NewOps, *VPI, *VPI, VPI->getDebugLoc());
  }
  };
}

VPHistogramRecipe *VPRecipeBuilder::widenIfHistogram(VPInstruction *VPI) {
  if (VPI->getOpcode() != Instruction::Store)
    return nullptr;

  auto HistInfo =
      Legal->getHistogramInfo(cast<StoreInst>(VPI->getUnderlyingInstr()));
  if (!HistInfo)
    return nullptr;

  const HistogramInfo *HI = *HistInfo;
  // FIXME: Support other operations.
  unsigned Opcode = HI->Update->getOpcode();
  assert((Opcode == Instruction::Add || Opcode == Instruction::Sub) &&
         "Histogram update operation must be an Add or Sub");

  SmallVector<VPValue *, 3> HGramOps;
  // Bucket address.
  HGramOps.push_back(VPI->getOperand(1));
  // Increment value.
  HGramOps.push_back(Plan.getOrAddLiveIn(HI->Update->getOperand(1)));

  // In case of predicated execution (due to tail-folding, or conditional
  // execution, or both), pass the relevant mask.
  if (CM.isMaskRequired(HI->Store))
    HGramOps.push_back(VPI->getMask());

  return new VPHistogramRecipe(Opcode, HGramOps, VPI->getDebugLoc());
}

bool VPRecipeBuilder::replaceWithFinalIfReductionStore(
    VPInstruction *VPI, VPBuilder &FinalRedStoresBuilder) {
  StoreInst *SI;
  if ((SI = dyn_cast<StoreInst>(VPI->getUnderlyingInstr())) &&
      Legal->isInvariantAddressOfReduction(SI->getPointerOperand())) {
    // Only create recipe for the final invariant store of the reduction.
    if (Legal->isInvariantStoreOfReduction(SI)) {
      auto *Recipe = new VPReplicateRecipe(
          SI, VPI->operandsWithoutMask(), true /* IsUniform */,
          nullptr /*Mask*/, *VPI, *VPI, VPI->getDebugLoc());
      FinalRedStoresBuilder.insert(Recipe);
    }
    VPI->eraseFromParent();
    return true;
  }

  return false;
}

VPReplicateRecipe *VPRecipeBuilder::handleReplication(VPInstruction *VPI,
                                                      VFRange &Range) {
  auto *I = VPI->getUnderlyingInstr();
  bool IsUniform = LoopVectorizationPlanner::getDecisionAndClampRange(
      [&](ElementCount VF) { return CM.isUniformAfterVectorization(I, VF); },
      Range);

  bool IsPredicated = CM.isPredicatedInst(I);

  // Even if the instruction is not marked as uniform, there are certain
  // intrinsic calls that can be effectively treated as such, so we check for
  // them here. Conservatively, we only do this for scalable vectors, since
  // for fixed-width VFs we can always fall back on full scalarization.
  if (!IsUniform && Range.Start.isScalable() && isa<IntrinsicInst>(I)) {
    switch (cast<IntrinsicInst>(I)->getIntrinsicID()) {
    case Intrinsic::assume:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
      // For scalable vectors if one of the operands is variant then we still
      // want to mark as uniform, which will generate one instruction for just
      // the first lane of the vector. We can't scalarize the call in the same
      // way as for fixed-width vectors because we don't know how many lanes
      // there are.
      //
      // The reasons for doing it this way for scalable vectors are:
      //   1. For the assume intrinsic generating the instruction for the first
      //      lane is still be better than not generating any at all. For
      //      example, the input may be a splat across all lanes.
      //   2. For the lifetime start/end intrinsics the pointer operand only
      //      does anything useful when the input comes from a stack object,
      //      which suggests it should always be uniform. For non-stack objects
      //      the effect is to poison the object, which still allows us to
      //      remove the call.
      IsUniform = true;
      break;
    default:
      break;
    }
  }
  VPValue *BlockInMask = nullptr;
  if (!IsPredicated) {
    // Finalize the recipe for Instr, first if it is not predicated.
    LLVM_DEBUG(dbgs() << "LV: Scalarizing:" << *I << "\n");
  } else {
    LLVM_DEBUG(dbgs() << "LV: Scalarizing and predicating:" << *I << "\n");
    // Instructions marked for predication are replicated and a mask operand is
    // added initially. Masked replicate recipes will later be placed under an
    // if-then construct to prevent side-effects. Generate recipes to compute
    // the block mask for this region.
    BlockInMask = VPI->getMask();
  }

  // Note that there is some custom logic to mark some intrinsics as uniform
  // manually above for scalable vectors, which this assert needs to account for
  // as well.
  assert((Range.Start.isScalar() || !IsUniform || !IsPredicated ||
          (Range.Start.isScalable() && isa<IntrinsicInst>(I))) &&
         "Should not predicate a uniform recipe");
  auto *Recipe =
      new VPReplicateRecipe(I, VPI->operandsWithoutMask(), IsUniform,
                            BlockInMask, *VPI, *VPI, VPI->getDebugLoc());
  return Recipe;
}

VPRecipeBase *
VPRecipeBuilder::tryToCreateWidenNonPhiRecipe(VPSingleDefRecipe *R,
                                              VFRange &Range) {
  assert(!R->isPhi() && "phis must be handled earlier");
  // First, check for specific widening recipes that deal with optimizing
  // truncates, calls and memory operations.

  VPRecipeBase *Recipe;
  auto *VPI = cast<VPInstruction>(R);
  if (VPI->getOpcode() == Instruction::Trunc &&
      (Recipe = tryToOptimizeInductionTruncate(VPI, Range)))
    return Recipe;

  // All widen recipes below deal only with VF > 1.
  if (LoopVectorizationPlanner::getDecisionAndClampRange(
          [&](ElementCount VF) { return VF.isScalar(); }, Range))
    return nullptr;

  if (VPI->getOpcode() == Instruction::Call)
    return tryToWidenCall(VPI, Range);

  Instruction *Instr = R->getUnderlyingInstr();
  assert(!is_contained({Instruction::Load, Instruction::Store},
                       VPI->getOpcode()) &&
         "Should have been handled prior to this!");

  if (!shouldWiden(Instr, Range))
    return nullptr;

  if (VPI->getOpcode() == Instruction::GetElementPtr)
    return new VPWidenGEPRecipe(cast<GetElementPtrInst>(Instr),
                                VPI->operandsWithoutMask(), *VPI,
                                VPI->getDebugLoc());

  if (Instruction::isCast(VPI->getOpcode())) {
    auto *CI = cast<CastInst>(Instr);
    auto *CastR = cast<VPInstructionWithType>(VPI);
    return new VPWidenCastRecipe(CI->getOpcode(), VPI->getOperand(0),
                                 CastR->getResultType(), CI, *VPI, *VPI,
                                 VPI->getDebugLoc());
  }

  return tryToWiden(VPI);
}

// To allow RUN_VPLAN_PASS to print the VPlan after VF/UF independent
// optimizations.
static void printOptimizedVPlan(VPlan &) {}

void LoopVectorizationPlanner::buildVPlansWithVPRecipes(ElementCount MinVF,
                                                        ElementCount MaxVF) {
  if (ElementCount::isKnownGT(MinVF, MaxVF))
    return;

  assert(OrigLoop->isInnermost() && "Inner loop expected.");

  const LoopAccessInfo *LAI = Legal->getLAI();
  LoopVersioning LVer(*LAI, LAI->getRuntimePointerChecking()->getChecks(),
                      OrigLoop, LI, DT, PSE.getSE());
  if (!LAI->getRuntimePointerChecking()->getChecks().empty() &&
      !LAI->getRuntimePointerChecking()->getDiffChecks()) {
    // Only use noalias metadata when using memory checks guaranteeing no
    // overlap across all iterations.
    LVer.prepareNoAliasMetadata();
  }

  // Create initial base VPlan0, to serve as common starting point for all
  // candidates built later for specific VF ranges.
  auto VPlan0 = VPlanTransforms::buildVPlan0(
      OrigLoop, *LI, Legal->getWidestInductionType(),
      getDebugLocFromInstOrOperands(Legal->getPrimaryInduction()), PSE, &LVer);

  // Create recipes for header phis.
  if (!RUN_VPLAN_PASS(VPlanTransforms::createHeaderPhiRecipes, *VPlan0, PSE,
                      *OrigLoop, Legal->getInductionVars(),
                      Legal->getReductionVars(),
                      Legal->getFixedOrderRecurrences(),
                      Config.getInLoopReductions(), Hints.allowReordering()))
    return;

  RUN_VPLAN_PASS(VPlanTransforms::simplifyRecipes, *VPlan0);
  RUN_VPLAN_PASS(VPlanTransforms::removeDeadRecipes, *VPlan0);
  // If we're vectorizing a loop with an uncountable exit, make sure that the
  // recipes are safe to handle.
  // TODO: Remove this once we can properly check the VPlan itself for both
  //       the presence of an uncountable exit and the presence of stores in
  //       the loop inside handleEarlyExits itself.
  UncountableExitStyle EEStyle = UncountableExitStyle::NoUncountableExit;
  if (Legal->hasUncountableEarlyExit())
    EEStyle = Legal->hasUncountableExitWithSideEffects()
                  ? UncountableExitStyle::MaskedHandleExitInScalarLoop
                  : UncountableExitStyle::ReadOnly;

  if (!RUN_VPLAN_PASS(VPlanTransforms::handleEarlyExits, *VPlan0, EEStyle,
                      OrigLoop, PSE, *DT, Legal->getAssumptionCache()))
    return;

  RUN_VPLAN_PASS(VPlanTransforms::addMiddleCheck, *VPlan0,
                 CM.foldTailByMasking());
  RUN_VPLAN_PASS(VPlanTransforms::createLoopRegions, *VPlan0);
  if (CM.foldTailByMasking())
    RUN_VPLAN_PASS(VPlanTransforms::foldTailByMasking, *VPlan0);
  RUN_VPLAN_PASS(VPlanTransforms::introduceMasksAndLinearize, *VPlan0);

  auto MaxVFTimes2 = MaxVF * 2;
  for (ElementCount VF = MinVF; ElementCount::isKnownLT(VF, MaxVFTimes2);) {
    VFRange SubRange = {VF, MaxVFTimes2};
    auto Plan = tryToBuildVPlanWithVPRecipes(
        std::unique_ptr<VPlan>(VPlan0->duplicate()), SubRange);
    VF = SubRange.End;

    if (!Plan)
      continue;

    // Now optimize the initial VPlan.
    RUN_VPLAN_PASS(VPlanTransforms::hoistPredicatedLoads, *Plan, PSE, OrigLoop);
    RUN_VPLAN_PASS(VPlanTransforms::sinkPredicatedStores, *Plan, PSE, OrigLoop);
    RUN_VPLAN_PASS(VPlanTransforms::truncateToMinimalBitwidths, *Plan,
                   Config.getMinimalBitwidths());
    RUN_VPLAN_PASS(VPlanTransforms::optimize, *Plan);
    // TODO: try to put addExplicitVectorLength close to addActiveLaneMask
    if (CM.foldTailWithEVL()) {
      RUN_VPLAN_PASS(VPlanTransforms::addExplicitVectorLength, *Plan,
                     Config.getMaxSafeElements());
      RUN_VPLAN_PASS(VPlanTransforms::optimizeEVLMasks, *Plan);
    }

    if (auto P = VPlanTransforms::narrowInterleaveGroups(*Plan, TTI))
      VPlans.push_back(std::move(P));

    RUN_VPLAN_PASS_NO_VERIFY(printOptimizedVPlan, *Plan);
    assert(verifyVPlanIsValid(*Plan) && "VPlan is invalid");
    VPlans.push_back(std::move(Plan));
  }
}

VPlanPtr
LoopVectorizationPlanner::tryToBuildVPlanWithVPRecipes(VPlanPtr Plan,
                                                       VFRange &Range) {

  using namespace llvm::VPlanPatternMatch;
  SmallPtrSet<const InterleaveGroup<Instruction> *, 1> InterleaveGroups;

  // ---------------------------------------------------------------------------
  // Build initial VPlan: Scan the body of the loop in a topological order to
  // visit each basic block after having visited its predecessor basic blocks.
  // ---------------------------------------------------------------------------

  bool RequiresScalarEpilogueCheck =
      LoopVectorizationPlanner::getDecisionAndClampRange(
          [this](ElementCount VF) {
            return !CM.requiresScalarEpilogue(VF.isVector());
          },
          Range);
  // Update the branch in the middle block if a scalar epilogue is required.
  VPBasicBlock *MiddleVPBB = Plan->getMiddleBlock();
  if (!RequiresScalarEpilogueCheck && MiddleVPBB->getNumSuccessors() == 2) {
    auto *BranchOnCond = cast<VPInstruction>(MiddleVPBB->getTerminator());
    assert(MiddleVPBB->getSuccessors()[1] == Plan->getScalarPreheader() &&
           "second successor must be scalar preheader");
    BranchOnCond->setOperand(0, Plan->getFalse());
  }

  // Don't use getDecisionAndClampRange here, because we don't know the UF
  // so this function is better to be conservative, rather than to split
  // it up into different VPlans.
  // TODO: Consider using getDecisionAndClampRange here to split up VPlans.
  bool IVUpdateMayOverflow = false;
  for (ElementCount VF : Range)
    IVUpdateMayOverflow |= !isIndvarOverflowCheckKnownFalse(&CM, VF);

  TailFoldingStyle Style = CM.getTailFoldingStyle();
  // Use NUW for the induction increment if we proved that it won't overflow in
  // the vector loop or when not folding the tail. In the later case, we know
  // that the canonical induction increment will not overflow as the vector trip
  // count is >= increment and a multiple of the increment.
  VPRegionBlock *LoopRegion = Plan->getVectorLoopRegion();
  bool HasNUW = !IVUpdateMayOverflow || Style == TailFoldingStyle::None;
  if (!HasNUW) {
    auto *IVInc =
        LoopRegion->getExitingBasicBlock()->getTerminator()->getOperand(0);
    assert(match(IVInc,
                 m_VPInstruction<Instruction::Add>(
                     m_Specific(LoopRegion->getCanonicalIV()), m_VPValue())) &&
           "Did not find the canonical IV increment");
    LoopRegion->clearCanonicalIVNUW(cast<VPInstruction>(IVInc));
  }

  // ---------------------------------------------------------------------------
  // Pre-construction: record ingredients whose recipes we'll need to further
  // process after constructing the initial VPlan.
  // ---------------------------------------------------------------------------

  // For each interleave group which is relevant for this (possibly trimmed)
  // Range, add it to the set of groups to be later applied to the VPlan and add
  // placeholders for its members' Recipes which we'll be replacing with a
  // single VPInterleaveRecipe.
  for (InterleaveGroup<Instruction> *IG : IAI.getInterleaveGroups()) {
    auto ApplyIG = [IG, this](ElementCount VF) -> bool {
      bool Result = (VF.isVector() && // Query is illegal for VF == 1
                     CM.getWideningDecision(IG->getInsertPos(), VF) ==
                         LoopVectorizationCostModel::CM_Interleave);
      // For scalable vectors, the interleave factors must be <= 8 since we
      // require the (de)interleaveN intrinsics instead of shufflevectors.
      assert((!Result || !VF.isScalable() || IG->getFactor() <= 8) &&
             "Unsupported interleave factor for scalable vectors");
      return Result;
    };
    if (!getDecisionAndClampRange(ApplyIG, Range))
      continue;
    InterleaveGroups.insert(IG);
  }

  // ---------------------------------------------------------------------------
  // Construct wide recipes and apply predication for original scalar
  // VPInstructions in the loop.
  // ---------------------------------------------------------------------------
  VPRecipeBuilder RecipeBuilder(*Plan, TLI, Legal, CM, Builder);

  // Scan the body of the loop in a topological order to visit each basic block
  // after having visited its predecessor basic blocks.
  VPBasicBlock *HeaderVPBB = LoopRegion->getEntryBasicBlock();
  ReversePostOrderTraversal<VPBlockShallowTraversalWrapper<VPBlockBase *>> RPOT(
      HeaderVPBB);

  RUN_VPLAN_PASS(VPlanTransforms::createInLoopReductionRecipes, *Plan,
                 Range.Start);

  VPCostContext CostCtx(CM.TTI, *CM.TLI, *Plan, CM, Config.CostKind, CM.PSE,
                        OrigLoop);

  RUN_VPLAN_PASS_NO_VERIFY(VPlanTransforms::makeMemOpWideningDecisions, *Plan,
                           Range, RecipeBuilder);

  // Now process all other blocks and instructions.
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    // Convert input VPInstructions to widened recipes.
    for (VPRecipeBase &R : make_early_inc_range(
             make_range(VPBB->getFirstNonPhi(), VPBB->end()))) {
      // Skip recipes that do not need transforming or have already been
      // transformed.
      if (isa<VPWidenCanonicalIVRecipe, VPBlendRecipe, VPReductionRecipe,
              VPReplicateRecipe, VPWidenLoadRecipe, VPWidenStoreRecipe,
              VPVectorPointerRecipe, VPVectorEndPointerRecipe,
              VPHistogramRecipe>(&R))
        continue;
      auto *VPI = cast<VPInstruction>(&R);
      if (!VPI->getUnderlyingValue())
        continue;

      // TODO: Gradually replace uses of underlying instruction by analyses on
      // VPlan. Migrate code relying on the underlying instruction from VPlan0
      // to construct recipes below to not use the underlying instruction.
      Instruction *Instr = cast<Instruction>(VPI->getUnderlyingValue());
      Builder.setInsertPoint(VPI);

      VPRecipeBase *Recipe =
          RecipeBuilder.tryToCreateWidenNonPhiRecipe(VPI, Range);
      if (!Recipe)
        Recipe =
            RecipeBuilder.handleReplication(cast<VPInstruction>(VPI), Range);

      if (isa<VPWidenIntOrFpInductionRecipe>(Recipe) && isa<TruncInst>(Instr)) {
        // Optimized a truncate to VPWidenIntOrFpInductionRecipe. It needs to be
        // moved to the phi section in the header.
        Recipe->insertBefore(*HeaderVPBB, HeaderVPBB->getFirstNonPhi());
      } else {
        Builder.insert(Recipe);
      }
      if (Recipe->getNumDefinedValues() == 1) {
        VPI->replaceAllUsesWith(Recipe->getVPSingleValue());
      } else {
        assert(Recipe->getNumDefinedValues() == 0 &&
               "Unexpected multidef recipe");
      }
      R.eraseFromParent();
    }
  }

  assert(isa<VPRegionBlock>(LoopRegion) &&
         !LoopRegion->getEntryBasicBlock()->empty() &&
         "entry block must be set to a VPRegionBlock having a non-empty entry "
         "VPBasicBlock");

  RUN_VPLAN_PASS(VPlanTransforms::adjustFirstOrderRecurrenceMiddleUsers, *Plan,
                 Range);

  // ---------------------------------------------------------------------------
  // Transform initial VPlan: Apply previously taken decisions, in order, to
  // bring the VPlan to its final state.
  // ---------------------------------------------------------------------------

  addReductionResultComputation(Plan, RecipeBuilder, Range.Start);

  // Optimize FindIV reductions to use sentinel-based approach when possible.
  RUN_VPLAN_PASS(VPlanTransforms::optimizeFindIVReductions, *Plan, PSE,
                 *OrigLoop);
  RUN_VPLAN_PASS(VPlanTransforms::optimizeInductionLiveOutUsers, *Plan, PSE,
                 CM.foldTailByMasking());

  // Apply mandatory transformation to handle reductions with multiple in-loop
  // uses if possible, bail out otherwise.
  if (!RUN_VPLAN_PASS(VPlanTransforms::handleMultiUseReductions, *Plan, ORE,
                      OrigLoop))
    return nullptr;
  // Apply mandatory transformation to handle FP maxnum/minnum reduction with
  // NaNs if possible, bail out otherwise.
  if (!RUN_VPLAN_PASS(VPlanTransforms::handleMaxMinNumReductions, *Plan))
    return nullptr;

  // Create whole-vector selects for find-last recurrences.
  if (!RUN_VPLAN_PASS(VPlanTransforms::handleFindLastReductions, *Plan))
    return nullptr;

  RUN_VPLAN_PASS(VPlanTransforms::removeBranchOnConst, *Plan, false);

  // Create partial reduction recipes for scaled reductions and transform
  // recipes to abstract recipes if it is legal and beneficial and clamp the
  // range for better cost estimation.
  // TODO: Enable following transform when the EVL-version of extended-reduction
  // and mulacc-reduction are implemented.
  if (!CM.foldTailWithEVL()) {
    RUN_VPLAN_PASS(VPlanTransforms::createPartialReductions, *Plan, CostCtx,
                   Range);
    RUN_VPLAN_PASS(VPlanTransforms::convertToAbstractRecipes, *Plan, CostCtx,
                   Range);
  }

  // Ensure scalar VF plans only contain VF=1, as required by hasScalarVFOnly.
  if (Range.Start.isScalar())
    Range.End = Range.Start * 2;

  for (ElementCount VF : Range)
    Plan->addVF(VF);
  Plan->setName("Initial VPlan");

  // Interleave memory: for each Interleave Group we marked earlier as relevant
  // for this VPlan, replace the Recipes widening its memory instructions with a
  // single VPInterleaveRecipe at its insertion point.
  RUN_VPLAN_PASS(VPlanTransforms::createInterleaveGroups, *Plan,
                 InterleaveGroups, CM.isEpilogueAllowed());

  // Replace VPValues for known constant strides.
  RUN_VPLAN_PASS(VPlanTransforms::replaceSymbolicStrides, *Plan, PSE,
                 Legal->getLAI()->getSymbolicStrides());

  auto BlockNeedsPredication = [this](BasicBlock *BB) {
    return Legal->blockNeedsPredication(BB);
  };
  RUN_VPLAN_PASS(VPlanTransforms::dropPoisonGeneratingRecipes, *Plan,
                 BlockNeedsPredication);

  if (useActiveLaneMask(Style)) {
    // TODO: Move checks to VPlanTransforms::addActiveLaneMask once
    // TailFoldingStyle is visible there.
    bool ForControlFlow = useActiveLaneMaskForControlFlow(Style);
    RUN_VPLAN_PASS(VPlanTransforms::addActiveLaneMask, *Plan, ForControlFlow);
  }

  assert(verifyVPlanIsValid(*Plan) && "VPlan is invalid");
  return Plan;
}

VPlanPtr LoopVectorizationPlanner::tryToBuildVPlan(VFRange &Range) {
  // Outer loop handling: They may require CFG and instruction level
  // transformations before even evaluating whether vectorization is profitable.
  // Since we cannot modify the incoming IR, we need to build VPlan upfront in
  // the vectorization pipeline.
  assert(!OrigLoop->isInnermost());
  assert(EnableVPlanNativePath && "VPlan-native path is not enabled.");

  auto Plan = VPlanTransforms::buildVPlan0(
      OrigLoop, *LI, Legal->getWidestInductionType(),
      getDebugLocFromInstOrOperands(Legal->getPrimaryInduction()), PSE);

  if (!VPlanTransforms::createHeaderPhiRecipes(
          *Plan, PSE, *OrigLoop, Legal->getInductionVars(),
          MapVector<PHINode *, RecurrenceDescriptor>(),
          SmallPtrSet<const PHINode *, 1>(), SmallPtrSet<PHINode *, 1>(),
          /*AllowReordering=*/false))
    return nullptr;
  [[maybe_unused]] bool CanHandleExits = VPlanTransforms::handleEarlyExits(
      *Plan, UncountableExitStyle::NoUncountableExit, OrigLoop, PSE, *DT,
      Legal->getAssumptionCache());
  assert(CanHandleExits &&
         "early-exits are not supported in VPlan-native path");
  VPlanTransforms::addMiddleCheck(*Plan, /*TailFolded*/ false);

  VPlanTransforms::createLoopRegions(*Plan);

  for (ElementCount VF : Range)
    Plan->addVF(VF);

  if (!VPlanTransforms::tryToConvertVPInstructionsToVPRecipes(*Plan, *TLI))
    return nullptr;

  // Optimize induction live-out users to use precomputed end values.
  VPlanTransforms::optimizeInductionLiveOutUsers(*Plan, PSE,
                                                 /*FoldTail=*/false);

  assert(verifyVPlanIsValid(*Plan) && "VPlan is invalid");
  return Plan;
}

void LoopVectorizationPlanner::addReductionResultComputation(
    VPlanPtr &Plan, VPRecipeBuilder &RecipeBuilder, ElementCount MinVF) {
  using namespace VPlanPatternMatch;
  VPTypeAnalysis TypeInfo(*Plan);
  VPRegionBlock *VectorLoopRegion = Plan->getVectorLoopRegion();
  VPBasicBlock *MiddleVPBB = Plan->getMiddleBlock();
  SmallVector<VPRecipeBase *> ToDelete;
  VPBasicBlock *LatchVPBB = VectorLoopRegion->getExitingBasicBlock();
  Builder.setInsertPoint(&*std::prev(std::prev(LatchVPBB->end())));
  VPBasicBlock::iterator IP = MiddleVPBB->getFirstNonPhi();
  for (VPRecipeBase &R :
       Plan->getVectorLoopRegion()->getEntryBasicBlock()->phis()) {
    VPReductionPHIRecipe *PhiR = dyn_cast<VPReductionPHIRecipe>(&R);
    if (!PhiR)
      continue;

    RecurKind RecurrenceKind = PhiR->getRecurrenceKind();
    const RecurrenceDescriptor &RdxDesc = Legal->getRecurrenceDescriptor(
        cast<PHINode>(PhiR->getUnderlyingInstr()));
    Type *PhiTy = TypeInfo.inferScalarType(PhiR);
    // If tail is folded by masking, introduce selects between the phi
    // and the users outside the vector region of each reduction, at the
    // beginning of the dedicated latch block.
    auto *OrigExitingVPV = PhiR->getBackedgeValue();
    auto *NewExitingVPV = PhiR->getBackedgeValue();
    if (!PhiR->isInLoop() && CM.foldTailByMasking()) {
      VPValue *Cond = vputils::findHeaderMask(*Plan);
      NewExitingVPV =
          Builder.createSelect(Cond, OrigExitingVPV, PhiR, {}, "", *PhiR);
      OrigExitingVPV->replaceUsesWithIf(NewExitingVPV, [](VPUser &U, unsigned) {
        return match(&U,
                     m_VPInstruction<VPInstruction::ComputeReductionResult>());
      });

      if (CM.usePredicatedReductionSelect(RecurrenceKind))
        PhiR->setOperand(1, NewExitingVPV);
    }

    // We want code in the middle block to appear to execute on the location of
    // the scalar loop's latch terminator because: (a) it is all compiler
    // generated, (b) these instructions are always executed after evaluating
    // the latch conditional branch, and (c) other passes may add new
    // predecessors which terminate on this line. This is the easiest way to
    // ensure we don't accidentally cause an extra step back into the loop while
    // debugging.
    DebugLoc ExitDL = OrigLoop->getLoopLatch()->getTerminator()->getDebugLoc();

    // TODO: At the moment ComputeReductionResult also drives creation of the
    // bc.merge.rdx phi nodes, hence it needs to be created unconditionally here
    // even for in-loop reductions, until the reduction resume value handling is
    // also modeled in VPlan.
    VPInstruction *FinalReductionResult;
    VPBuilder::InsertPointGuard Guard(Builder);
    Builder.setInsertPoint(MiddleVPBB, IP);
    // For AnyOf reductions, find the select among PhiR's users and convert
    // the reduction phi to operate on bools before creating the final
    // reduction result.
    if (RecurrenceDescriptor::isAnyOfRecurrenceKind(RecurrenceKind)) {
      auto *AnyOfSelect =
          cast<VPSingleDefRecipe>(*find_if(PhiR->users(), [](VPUser *U) {
            return match(U, m_Select(m_VPValue(), m_VPValue(), m_VPValue()));
          }));
      VPValue *Start = PhiR->getStartValue();
      bool TrueValIsPhi = AnyOfSelect->getOperand(1) == PhiR;
      // NewVal is the non-phi operand of the select.
      VPValue *NewVal = TrueValIsPhi ? AnyOfSelect->getOperand(2)
                                     : AnyOfSelect->getOperand(1);

      // Adjust AnyOf reductions; replace the reduction phi for the selected
      // value with a boolean reduction phi node to check if the condition is
      // true in any iteration. The final value is selected by the final
      // ComputeReductionResult.
      VPValue *Cmp = AnyOfSelect->getOperand(0);
      // If the compare is checking the reduction PHI node, adjust it to check
      // the start value.
      if (VPRecipeBase *CmpR = Cmp->getDefiningRecipe())
        CmpR->replaceUsesOfWith(PhiR, PhiR->getStartValue());
      Builder.setInsertPoint(AnyOfSelect);

      // If the true value of the select is the reduction phi, the new value
      // is selected if the negated condition is true in any iteration.
      if (TrueValIsPhi)
        Cmp = Builder.createNot(Cmp);
      VPValue *Or = Builder.createOr(PhiR, Cmp);
      // Only replace uses inside the vector region with Or. External uses
      // (e.g. scalar preheader resume phis) must be replaced by the user
      // update loop below with FinalReductionResult.
      AnyOfSelect->replaceUsesWithIf(Or, [](VPUser &U, unsigned) {
        return cast<VPRecipeBase>(&U)->getRegion();
      });
      ToDelete.push_back(AnyOfSelect);

      // Convert the reduction phi to operate on bools.
      PhiR->setOperand(0, Plan->getFalse());

      // Update NewExitingVPV if it was pointing to the now-replaced select.
      if (NewExitingVPV == AnyOfSelect)
        NewExitingVPV = Or;

      Builder.setInsertPoint(MiddleVPBB, IP);

      FinalReductionResult =
          Builder.createAnyOfReduction(NewExitingVPV, NewVal, Start, ExitDL);
    } else {
      VPIRFlags Flags(RecurrenceKind, PhiR->isOrdered(), PhiR->isInLoop(),
                      PhiR->getFastMathFlags());
      FinalReductionResult =
          Builder.createNaryOp(VPInstruction::ComputeReductionResult,
                               {NewExitingVPV}, Flags, ExitDL);
    }
    // If the vector reduction can be performed in a smaller type, we truncate
    // then extend the loop exit value to enable InstCombine to evaluate the
    // entire expression in the smaller type.
    if (MinVF.isVector() && PhiTy != RdxDesc.getRecurrenceType() &&
        !RecurrenceDescriptor::isAnyOfRecurrenceKind(RecurrenceKind)) {
      assert(!PhiR->isInLoop() && "Unexpected truncated inloop reduction!");
      assert(!RecurrenceDescriptor::isMinMaxRecurrenceKind(RecurrenceKind) &&
             "Unexpected truncated min-max recurrence!");
      Type *RdxTy = RdxDesc.getRecurrenceType();
      VPWidenCastRecipe *Trunc;
      Instruction::CastOps ExtendOpc =
          RdxDesc.isSigned() ? Instruction::SExt : Instruction::ZExt;
      VPWidenCastRecipe *Extnd;
      {
        VPBuilder::InsertPointGuard Guard(Builder);
        Builder.setInsertPoint(
            NewExitingVPV->getDefiningRecipe()->getParent(),
            std::next(NewExitingVPV->getDefiningRecipe()->getIterator()));
        Trunc =
            Builder.createWidenCast(Instruction::Trunc, NewExitingVPV, RdxTy);
        Extnd = Builder.createWidenCast(ExtendOpc, Trunc, PhiTy);
      }
      if (PhiR->getOperand(1) == NewExitingVPV)
        PhiR->setOperand(1, Extnd->getVPSingleValue());

      // Update ComputeReductionResult with the truncated exiting value and
      // extend its result. Operand 0 provides the values to be reduced.
      FinalReductionResult->setOperand(0, Trunc);
      FinalReductionResult =
          Builder.createScalarCast(ExtendOpc, FinalReductionResult, PhiTy, {});
    }

    // Update all users outside the vector region. Also replace redundant
    // extracts.
    for (auto *U : to_vector(OrigExitingVPV->users())) {
      auto *Parent = cast<VPRecipeBase>(U)->getParent();
      if (FinalReductionResult == U || Parent->getParent())
        continue;
      // Skip ComputeReductionResult and FindIV reductions when they are not the
      // final result.
      if (match(U, m_VPInstruction<VPInstruction::ComputeReductionResult>()) ||
          (RecurrenceDescriptor::isFindIVRecurrenceKind(RecurrenceKind) &&
           match(U, m_VPInstruction<Instruction::ICmp>())))
        continue;
      U->replaceUsesOfWith(OrigExitingVPV, FinalReductionResult);

      // Look through ExtractLastPart.
      if (match(U, m_ExtractLastPart(m_VPValue())))
        U = cast<VPInstruction>(U)->getSingleUser();

      if (match(U, m_CombineOr(m_ExtractLane(m_VPValue(), m_VPValue()),
                               m_ExtractLastLane(m_VPValue()))))
        cast<VPInstruction>(U)->replaceAllUsesWith(FinalReductionResult);
    }

    RecurKind RK = PhiR->getRecurrenceKind();
    if ((!RecurrenceDescriptor::isAnyOfRecurrenceKind(RK) &&
         !RecurrenceDescriptor::isFindIVRecurrenceKind(RK) &&
         !RecurrenceDescriptor::isMinMaxRecurrenceKind(RK) &&
         !RecurrenceDescriptor::isFindLastRecurrenceKind(RK))) {
      VPBuilder PHBuilder(Plan->getVectorPreheader());
      VPValue *Iden = Plan->getOrAddLiveIn(
          getRecurrenceIdentity(RK, PhiTy, PhiR->getFastMathFlags()));
      auto *ScaleFactorVPV = Plan->getConstantInt(32, 1);
      VPValue *StartV = PHBuilder.createNaryOp(
          VPInstruction::ReductionStartVector,
          {PhiR->getStartValue(), Iden, ScaleFactorVPV}, *PhiR);
      PhiR->setOperand(0, StartV);
    }
  }
  for (VPRecipeBase *R : ToDelete)
    R->eraseFromParent();

  RUN_VPLAN_PASS(VPlanTransforms::clearReductionWrapFlags, *Plan);
}

void LoopVectorizationPlanner::attachRuntimeChecks(
    VPlan &Plan, GeneratedRTChecks &RTChecks, bool HasBranchWeights) const {
  const auto &[SCEVCheckCond, SCEVCheckBlock] = RTChecks.getSCEVChecks();
  if (SCEVCheckBlock && SCEVCheckBlock->hasNPredecessors(0)) {
    assert((!Config.OptForSize ||
            CM.Hints->getForce() == LoopVectorizeHints::FK_Enabled) &&
           "Cannot SCEV check stride or overflow when optimizing for size");
    RUN_VPLAN_PASS(VPlanTransforms::attachCheckBlock, Plan, SCEVCheckCond,
                   SCEVCheckBlock, HasBranchWeights);
  }
  const auto &[MemCheckCond, MemCheckBlock] = RTChecks.getMemRuntimeChecks();
  if (MemCheckBlock && MemCheckBlock->hasNPredecessors(0)) {
    // VPlan-native path does not do any analysis for runtime checks
    // currently.
    assert((!EnableVPlanNativePath || OrigLoop->isInnermost()) &&
           "Runtime checks are not supported for outer loops yet");

    if (Config.OptForSize) {
      assert(
          CM.Hints->getForce() == LoopVectorizeHints::FK_Enabled &&
          "Cannot emit memory checks when optimizing for size, unless forced "
          "to vectorize.");
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(DEBUG_TYPE, "VectorizationCodeSize",
                                          OrigLoop->getStartLoc(),
                                          OrigLoop->getHeader())
               << "Code-size may be reduced by not forcing "
                  "vectorization, or by source-code modifications "
                  "eliminating the need for runtime checks "
                  "(e.g., adding 'restrict').";
      });
    }
    RUN_VPLAN_PASS(VPlanTransforms::attachCheckBlock, Plan, MemCheckCond,
                   MemCheckBlock, HasBranchWeights);
  }
}

void LoopVectorizationPlanner::addMinimumIterationCheck(
    VPlan &Plan, ElementCount VF, unsigned UF,
    ElementCount MinProfitableTripCount) const {
  const uint32_t *BranchWeights =
      hasBranchWeightMD(*OrigLoop->getLoopLatch()->getTerminator())
          ? &MinItersBypassWeights[0]
          : nullptr;
  RUN_VPLAN_PASS(VPlanTransforms::addMinimumIterationCheck, Plan, VF, UF,
                 MinProfitableTripCount,
                 CM.requiresScalarEpilogue(VF.isVector()),
                 CM.foldTailByMasking(), OrigLoop, BranchWeights,
                 OrigLoop->getLoopPredecessor()->getTerminator()->getDebugLoc(),
                 PSE, /*CheckBlock=*/nullptr);
}

// Determine how to lower the epilogue, which depends on 1) optimising
// for minimum code-size, 2) tail-folding compiler options, 3) loop
// hints forcing tail-folding, and 4) a TTI hook that analyses whether the loop
// is suitable for tail-folding.
static EpilogueLowering
getEpilogueLowering(Function *F, Loop *L, LoopVectorizeHints &Hints,
                    bool OptForSize, TargetTransformInfo *TTI,
                    TargetLibraryInfo *TLI, LoopVectorizationLegality &LVL,
                    InterleavedAccessInfo *IAI) {
  // 1) OptSize takes precedence over all other options, i.e. if this is set,
  // don't look at hints or options, and don't request an epilogue.
  if (F->hasOptSize() ||
      (OptForSize && Hints.getForce() != LoopVectorizeHints::FK_Enabled))
    return CM_EpilogueNotAllowedOptSize;

  // 2) If set, obey the directives
  if (TailFoldingPolicy.getNumOccurrences()) {
    switch (TailFoldingPolicy) {
    case TailFoldingPolicyTy::None:
      return CM_EpilogueAllowed;
    case TailFoldingPolicyTy::PreferFoldTail:
      return CM_EpilogueNotNeededFoldTail;
    case TailFoldingPolicyTy::MustFoldTail:
      return CM_EpilogueNotAllowedFoldTail;
    };
  }

  // 3) If set, obey the hints
  switch (Hints.getPredicate()) {
  case LoopVectorizeHints::FK_Enabled:
    return CM_EpilogueNotNeededFoldTail;
  case LoopVectorizeHints::FK_Disabled:
    return CM_EpilogueAllowed;
  };

  // 4) if the TTI hook indicates this is profitable, request tail-folding.
  TailFoldingInfo TFI(TLI, &LVL, IAI);
  if (TTI->preferTailFoldingOverEpilogue(&TFI))
    return CM_EpilogueNotNeededFoldTail;

  return CM_EpilogueAllowed;
}

// Process the loop in the VPlan-native vectorization path. This path builds
// VPlan upfront in the vectorization pipeline, which allows to apply
// VPlan-to-VPlan transformations from the very beginning without modifying the
// input LLVM IR.
static bool processLoopInVPlanNativePath(
    Loop *L, PredicatedScalarEvolution &PSE, LoopInfo *LI, DominatorTree *DT,
    LoopVectorizationLegality *LVL, TargetTransformInfo *TTI,
    TargetLibraryInfo *TLI, DemandedBits *DB, AssumptionCache *AC,
    OptimizationRemarkEmitter *ORE,
    std::function<BlockFrequencyInfo &()> GetBFI, bool OptForSize,
    LoopVectorizeHints &Hints, LoopVectorizationRequirements &Requirements) {

  if (isa<SCEVCouldNotCompute>(PSE.getBackedgeTakenCount())) {
    LLVM_DEBUG(dbgs() << "LV: cannot compute the outer-loop trip count\n");
    return false;
  }
  assert(EnableVPlanNativePath && "VPlan-native path is disabled.");
  Function *F = L->getHeader()->getParent();
  InterleavedAccessInfo IAI(PSE, L, DT, LI, LVL->getLAI());

  EpilogueLowering SEL =
      getEpilogueLowering(F, L, Hints, OptForSize, TTI, TLI, *LVL, &IAI);

  VFSelectionContext Config(*TTI, LVL, L, *F, PSE, DB, ORE, &Hints, OptForSize);
  LoopVectorizationCostModel CM(SEL, L, PSE, LI, LVL, *TTI, TLI, AC, ORE,
                                GetBFI, F, &Hints, IAI, Config);
  // Use the planner for outer loop vectorization.
  // TODO: CM is not used at this point inside the planner. Turn CM into an
  // optional argument if we don't need it in the future.
  LoopVectorizationPlanner LVP(L, LI, DT, TLI, *TTI, LVL, CM, Config, IAI, PSE,
                               Hints, ORE);

  // Get user vectorization factor.
  ElementCount UserVF = Hints.getWidth();

  Config.collectElementTypesForWidening();

  // Plan how to best vectorize, return the best VF and its cost.
  const VectorizationFactor VF = LVP.planInVPlanNativePath(UserVF);

  // If we are stress testing VPlan builds, do not attempt to generate vector
  // code. Masked vector code generation support will follow soon.
  // Also, do not attempt to vectorize if no vector code will be produced.
  if (VPlanBuildStressTest || VectorizationFactor::Disabled() == VF)
    return false;

  VPlan &BestPlan = LVP.getPlanFor(VF.Width);

  {
    GeneratedRTChecks Checks(PSE, DT, LI, TTI, Config.CostKind);
    InnerLoopVectorizer LB(L, PSE, LI, DT, TTI, AC, VF.Width, /*UF=*/1, &CM,
                           Checks, BestPlan);
    LLVM_DEBUG(dbgs() << "Vectorizing outer loop in \"" << F->getName()
                      << "\"\n");
    LVP.addMinimumIterationCheck(BestPlan, VF.Width, /*UF=*/1,
                                 VF.MinProfitableTripCount);
    bool HasBranchWeights =
        hasBranchWeightMD(*L->getLoopLatch()->getTerminator());
    LVP.attachRuntimeChecks(BestPlan, Checks, HasBranchWeights);

    reportVectorization(ORE, L, VF, 1);

    LVP.executePlan(VF.Width, /*UF=*/1, BestPlan, LB, DT);
  }

  assert(!verifyFunction(*F, &dbgs()));
  return true;
}

// Emit a remark if there are stores to floats that required a floating point
// extension. If the vectorized loop was generated with floating point there
// will be a performance penalty from the conversion overhead and the change in
// the vector width.
static void checkMixedPrecision(Loop *L, OptimizationRemarkEmitter *ORE) {
  SmallVector<Instruction *, 4> Worklist;
  for (BasicBlock *BB : L->getBlocks()) {
    for (Instruction &Inst : *BB) {
      if (auto *S = dyn_cast<StoreInst>(&Inst)) {
        if (S->getValueOperand()->getType()->isFloatTy())
          Worklist.push_back(S);
      }
    }
  }

  // Traverse the floating point stores upwards searching, for floating point
  // conversions.
  SmallPtrSet<const Instruction *, 4> Visited;
  SmallPtrSet<const Instruction *, 4> EmittedRemark;
  while (!Worklist.empty()) {
    auto *I = Worklist.pop_back_val();
    if (!L->contains(I))
      continue;
    if (!Visited.insert(I).second)
      continue;

    // Emit a remark if the floating point store required a floating
    // point conversion.
    // TODO: More work could be done to identify the root cause such as a
    // constant or a function return type and point the user to it.
    if (isa<FPExtInst>(I) && EmittedRemark.insert(I).second)
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(LV_NAME, "VectorMixedPrecision",
                                          I->getDebugLoc(), L->getHeader())
               << "floating point conversion changes vector width. "
               << "Mixed floating point precision requires an up/down "
               << "cast that will negatively impact performance.";
      });

    for (Use &Op : I->operands())
      if (auto *OpI = dyn_cast<Instruction>(Op))
        Worklist.push_back(OpI);
  }
}

/// For loops with uncountable early exits, find the cost of doing work when
/// exiting the loop early, such as calculating the final exit values of
/// variables used outside the loop.
/// TODO: This is currently overly pessimistic because the loop may not take
/// the early exit, but better to keep this conservative for now. In future,
/// it might be possible to relax this by using branch probabilities.
static InstructionCost calculateEarlyExitCost(VPCostContext &CostCtx,
                                              VPlan &Plan, ElementCount VF) {
  InstructionCost Cost = 0;
  for (auto *ExitVPBB : Plan.getExitBlocks()) {
    for (auto *PredVPBB : ExitVPBB->getPredecessors()) {
      // If the predecessor is not the middle.block, then it must be the
      // vector.early.exit block, which may contain work to calculate the exit
      // values of variables used outside the loop.
      if (PredVPBB != Plan.getMiddleBlock()) {
        LLVM_DEBUG(dbgs() << "Calculating cost of work in exit block "
                          << PredVPBB->getName() << ":\n");
        Cost += PredVPBB->cost(VF, CostCtx);
      }
    }
  }
  return Cost;
}

/// This function determines whether or not it's still profitable to vectorize
/// the loop given the extra work we have to do outside of the loop:
///  1. Perform the runtime checks before entering the loop to ensure it's safe
///     to vectorize.
///  2. In the case of loops with uncountable early exits, we may have to do
///     extra work when exiting the loop early, such as calculating the final
///     exit values of variables used outside the loop.
///  3. The middle block.
static bool isOutsideLoopWorkProfitable(GeneratedRTChecks &Checks,
                                        VectorizationFactor &VF, Loop *L,
                                        PredicatedScalarEvolution &PSE,
                                        VPCostContext &CostCtx, VPlan &Plan,
                                        EpilogueLowering SEL,
                                        std::optional<unsigned> VScale) {
  InstructionCost RtC = Checks.getCost();
  if (!RtC.isValid())
    return false;

  // When interleaving only scalar and vector cost will be equal, which in turn
  // would lead to a divide by 0. Fall back to hard threshold.
  if (VF.Width.isScalar()) {
    // TODO: Should we rename VectorizeMemoryCheckThreshold?
    if (RtC > VectorizeMemoryCheckThreshold) {
      LLVM_DEBUG(
          dbgs()
          << "LV: Interleaving only is not profitable due to runtime checks\n");
      return false;
    }
    return true;
  }

  // The scalar cost should only be 0 when vectorizing with a user specified
  // VF/IC. In those cases, runtime checks should always be generated.
  uint64_t ScalarC = VF.ScalarCost.getValue();
  if (ScalarC == 0)
    return true;

  InstructionCost TotalCost = RtC;
  // Add on the cost of any work required in the vector early exit block, if
  // one exists.
  TotalCost += calculateEarlyExitCost(CostCtx, Plan, VF.Width);
  TotalCost += Plan.getMiddleBlock()->cost(VF.Width, CostCtx);

  // First, compute the minimum iteration count required so that the vector
  // loop outperforms the scalar loop.
  //  The total cost of the scalar loop is
  //   ScalarC * TC
  //  where
  //  * TC is the actual trip count of the loop.
  //  * ScalarC is the cost of a single scalar iteration.
  //
  //  The total cost of the vector loop is
  //    TotalCost + VecC * (TC / VF) + EpiC
  //  where
  //  * TotalCost is the sum of the costs cost of
  //    - the generated runtime checks, i.e. RtC
  //    - performing any additional work in the vector.early.exit block for
  //      loops with uncountable early exits.
  //    - the middle block, if ExpectedTC <=  VF.Width.
  //  * VecC is the cost of a single vector iteration.
  //  * TC is the actual trip count of the loop
  //  * VF is the vectorization factor
  //  * EpiCost is the cost of the generated epilogue, including the cost
  //    of the remaining scalar operations.
  //
  // Vectorization is profitable once the total vector cost is less than the
  // total scalar cost:
  //   TotalCost + VecC * (TC / VF) + EpiC <  ScalarC * TC
  //
  // Now we can compute the minimum required trip count TC as
  //   VF * (TotalCost + EpiC) / (ScalarC * VF - VecC) < TC
  //
  // For now we assume the epilogue cost EpiC = 0 for simplicity. Note that
  // the computations are performed on doubles, not integers and the result
  // is rounded up, hence we get an upper estimate of the TC.
  unsigned IntVF = estimateElementCount(VF.Width, VScale);
  uint64_t Div = ScalarC * IntVF - VF.Cost.getValue();
  uint64_t MinTC1 =
      Div == 0 ? 0 : divideCeil(TotalCost.getValue() * IntVF, Div);

  // Second, compute a minimum iteration count so that the cost of the
  // runtime checks is only a fraction of the total scalar loop cost. This
  // adds a loop-dependent bound on the overhead incurred if the runtime
  // checks fail. In case the runtime checks fail, the cost is RtC + ScalarC
  // * TC. To bound the runtime check to be a fraction 1/X of the scalar
  // cost, compute
  //   RtC < ScalarC * TC * (1 / X)  ==>  RtC * X / ScalarC < TC
  uint64_t MinTC2 = divideCeil(RtC.getValue() * 10, ScalarC);

  // Now pick the larger minimum. If it is not a multiple of VF and an epilogue
  // is allowed, choose the next closest multiple of VF. This should partly
  // compensate for ignoring the epilogue cost.
  uint64_t MinTC = std::max(MinTC1, MinTC2);
  if (SEL == CM_EpilogueAllowed)
    MinTC = alignTo(MinTC, IntVF);
  VF.MinProfitableTripCount = ElementCount::getFixed(MinTC);

  LLVM_DEBUG(
      dbgs() << "LV: Minimum required TC for runtime checks to be profitable:"
             << VF.MinProfitableTripCount << "\n");

  // Skip vectorization if the expected trip count is less than the minimum
  // required trip count.
  if (auto ExpectedTC = getSmallBestKnownTC(PSE, L)) {
    if (ElementCount::isKnownLT(*ExpectedTC, VF.MinProfitableTripCount)) {
      LLVM_DEBUG(dbgs() << "LV: Vectorization is not beneficial: expected "
                           "trip count < minimum profitable VF ("
                        << *ExpectedTC << " < " << VF.MinProfitableTripCount
                        << ")\n");

      return false;
    }
  }
  return true;
}

LoopVectorizePass::LoopVectorizePass(LoopVectorizeOptions Opts)
    : InterleaveOnlyWhenForced(Opts.InterleaveOnlyWhenForced ||
                               !EnableLoopInterleaving),
      VectorizeOnlyWhenForced(Opts.VectorizeOnlyWhenForced ||
                              !EnableLoopVectorization) {}

/// Prepare \p MainPlan for vectorizing the main vector loop during epilogue
/// vectorization.
static SmallVector<VPInstruction *>
preparePlanForMainVectorLoop(VPlan &MainPlan, VPlan &EpiPlan) {
  using namespace VPlanPatternMatch;
  // When vectorizing the epilogue, FindFirstIV & FindLastIV reductions can
  // introduce multiple uses of undef/poison. If the reduction start value may
  // be undef or poison it needs to be frozen and the frozen start has to be
  // used when computing the reduction result. We also need to use the frozen
  // value in the resume phi generated by the main vector loop, as this is also
  // used to compute the reduction result after the epilogue vector loop.
  auto AddFreezeForFindLastIVReductions = [](VPlan &Plan,
                                             bool UpdateResumePhis) {
    VPBuilder Builder(Plan.getEntry());
    for (VPRecipeBase &R : *Plan.getMiddleBlock()) {
      auto *VPI = dyn_cast<VPInstruction>(&R);
      if (!VPI)
        continue;
      VPValue *OrigStart;
      if (!matchFindIVResult(VPI, m_VPValue(), m_VPValue(OrigStart)))
        continue;
      if (isGuaranteedNotToBeUndefOrPoison(OrigStart->getLiveInIRValue()))
        continue;
      VPInstruction *Freeze =
          Builder.createNaryOp(Instruction::Freeze, {OrigStart}, {}, "fr");
      VPI->setOperand(2, Freeze);
      if (UpdateResumePhis)
        OrigStart->replaceUsesWithIf(Freeze, [Freeze](VPUser &U, unsigned) {
          return Freeze != &U && isa<VPPhi>(&U);
        });
    }
  };
  AddFreezeForFindLastIVReductions(MainPlan, true);
  AddFreezeForFindLastIVReductions(EpiPlan, false);

  VPValue *VectorTC = nullptr;
  auto *Term =
      MainPlan.getVectorLoopRegion()->getExitingBasicBlock()->getTerminator();
  [[maybe_unused]] bool MatchedTC =
      match(Term, m_BranchOnCount(m_VPValue(), m_VPValue(VectorTC)));
  assert(MatchedTC && "must match vector trip count");

  // If there is a suitable resume value for the canonical induction in the
  // scalar (which will become vector) epilogue loop, use it and move it to the
  // beginning of the scalar preheader. Otherwise create it below.
  VPBasicBlock *MainScalarPH = MainPlan.getScalarPreheader();
  auto ResumePhiIter =
      find_if(MainScalarPH->phis(), [VectorTC](VPRecipeBase &R) {
        return match(&R, m_VPInstruction<Instruction::PHI>(m_Specific(VectorTC),
                                                           m_ZeroInt()));
      });
  VPPhi *ResumePhi = nullptr;
  if (ResumePhiIter == MainScalarPH->phis().end()) {
    Type *Ty = VPTypeAnalysis(MainPlan).inferScalarType(VectorTC);
    VPBuilder ScalarPHBuilder(MainScalarPH, MainScalarPH->begin());
    ResumePhi = ScalarPHBuilder.createScalarPhi(
        {VectorTC, MainPlan.getZero(Ty)}, {}, "vec.epilog.resume.val");
  } else {
    ResumePhi = cast<VPPhi>(&*ResumePhiIter);
    ResumePhi->setName("vec.epilog.resume.val");
    if (&MainScalarPH->front() != ResumePhi)
      ResumePhi->moveBefore(*MainScalarPH, MainScalarPH->begin());
  }

  // Create a ResumeForEpilogue for the canonical IV resume as the
  // first non-phi, to keep it alive for the epilogue.
  VPBuilder ResumeBuilder(MainScalarPH);
  ResumeBuilder.createNaryOp(VPInstruction::ResumeForEpilogue, ResumePhi);

  // Create ResumeForEpilogue instructions for the resume phis of the
  // VPIRPhis in the scalar header of the main plan and return them so they can
  // be used as resume values when vectorizing the epilogue.
  return to_vector(
      map_range(MainPlan.getScalarHeader()->phis(), [&](VPRecipeBase &R) {
        assert(isa<VPIRPhi>(R) &&
               "only VPIRPhis expected in the scalar header");
        return ResumeBuilder.createNaryOp(VPInstruction::ResumeForEpilogue,
                                          R.getOperand(0));
      }));
}

/// Prepare \p Plan for vectorizing the epilogue loop. That is, re-use expanded
/// SCEVs from \p ExpandedSCEVs and set resume values for header recipes. Some
/// reductions require creating new instructions to compute the resume values.
/// They are collected in a vector and returned. They must be moved to the
/// preheader of the vector epilogue loop, after created by the execution of \p
/// Plan.
static SmallVector<Instruction *> preparePlanForEpilogueVectorLoop(
    VPlan &Plan, Loop *L, const SCEV2ValueTy &ExpandedSCEVs,
    EpilogueLoopVectorizationInfo &EPI, LoopVectorizationCostModel &CM,
    VFSelectionContext &Config, ScalarEvolution &SE) {
  VPRegionBlock *VectorLoop = Plan.getVectorLoopRegion();
  VPBasicBlock *Header = VectorLoop->getEntryBasicBlock();
  Header->setName("vec.epilog.vector.body");

  VPValue *IV = VectorLoop->getCanonicalIV();
  // When vectorizing the epilogue loop, the canonical induction needs to start
  // at the resume value from the main vector loop. Find the resume value
  // created during execution of the main VPlan. It must be the first phi in the
  // loop preheader. Add this resume value as an offset to the canonical IV of
  // the epilogue loop.
  using namespace llvm::PatternMatch;
  PHINode *EPResumeVal = &*L->getLoopPreheader()->phis().begin();
  for (Value *Inc : EPResumeVal->incoming_values()) {
    if (match(Inc, m_SpecificInt(0)))
      continue;
    assert(!EPI.VectorTripCount &&
           "Must only have a single non-zero incoming value");
    EPI.VectorTripCount = Inc;
  }
  // If we didn't find a non-zero vector trip count, all incoming values
  // must be zero, which also means the vector trip count is zero. Pick the
  // first zero as vector trip count.
  // TODO: We should not choose VF * UF so the main vector loop is known to
  // be dead.
  if (!EPI.VectorTripCount) {
    assert(EPResumeVal->getNumIncomingValues() > 0 &&
           all_of(EPResumeVal->incoming_values(), match_fn(m_SpecificInt(0))) &&
           "all incoming values must be 0");
    EPI.VectorTripCount = EPResumeVal->getOperand(0);
  }
  VPValue *VPV = Plan.getOrAddLiveIn(EPResumeVal);
  assert(all_of(IV->users(),
                [](const VPUser *U) {
                  return isa<VPScalarIVStepsRecipe>(U) ||
                         isa<VPDerivedIVRecipe>(U) ||
                         cast<VPRecipeBase>(U)->isScalarCast() ||
                         cast<VPInstruction>(U)->getOpcode() ==
                             Instruction::Add;
                }) &&
         "the canonical IV should only be used by its increment or "
         "ScalarIVSteps when resetting the start value");
  VPBuilder Builder(Header, Header->getFirstNonPhi());
  VPInstruction *Add = Builder.createAdd(IV, VPV);
  // Replace all users of the canonical IV and its increment with the offset
  // version, except for the Add itself and the canonical IV increment.
  auto *Increment = vputils::findCanonicalIVIncrement(Plan);
  assert(Increment && "Must have a canonical IV increment at this point");
  IV->replaceUsesWithIf(Add, [Add, Increment](VPUser &U, unsigned) {
    return &U != Add && &U != Increment;
  });
  VPInstruction *OffsetIVInc =
      VPBuilder::getToInsertAfter(Increment).createAdd(Increment, VPV);
  Increment->replaceAllUsesWith(OffsetIVInc);
  OffsetIVInc->setOperand(0, Increment);

  DenseMap<Value *, Value *> ToFrozen;
  SmallVector<Instruction *> InstsToMove;
  // Ensure that the start values for all header phi recipes are updated before
  // vectorizing the epilogue loop.
  for (VPRecipeBase &R : Header->phis()) {
    Value *ResumeV = nullptr;
    // TODO: Move setting of resume values to prepareToExecute.
    if (auto *ReductionPhi = dyn_cast<VPReductionPHIRecipe>(&R)) {
      // Find the reduction result by searching users of the phi or its backedge
      // value.
      auto IsReductionResult = [](VPRecipeBase *R) {
        auto *VPI = dyn_cast<VPInstruction>(R);
        return VPI && VPI->getOpcode() == VPInstruction::ComputeReductionResult;
      };
      auto *RdxResult = cast<VPInstruction>(
          vputils::findRecipe(ReductionPhi->getBackedgeValue(), IsReductionResult));
      assert(RdxResult && "expected to find reduction result");

      ResumeV = cast<PHINode>(ReductionPhi->getUnderlyingInstr())
                    ->getIncomingValueForBlock(L->getLoopPreheader());

      // Check for FindIV pattern by looking for icmp user of RdxResult.
      // The pattern is: select(icmp ne RdxResult, Sentinel), RdxResult, Start
      using namespace VPlanPatternMatch;
      VPValue *SentinelVPV = nullptr;
      bool IsFindIV = any_of(RdxResult->users(), [&](VPUser *U) {
        return match(U, VPlanPatternMatch::m_SpecificICmp(
                            ICmpInst::ICMP_NE, m_Specific(RdxResult),
                            m_VPValue(SentinelVPV)));
      });

      RecurKind RK = ReductionPhi->getRecurrenceKind();
      if (RecurrenceDescriptor::isAnyOfRecurrenceKind(RK) || IsFindIV) {
        auto *ResumePhi = cast<PHINode>(ResumeV);
        Value *StartV = ResumePhi->getIncomingValueForBlock(
            EPI.MainLoopIterationCountCheck);
        IRBuilder<> Builder(ResumePhi->getParent(),
                            ResumePhi->getParent()->getFirstNonPHIIt());

        if (RecurrenceDescriptor::isAnyOfRecurrenceKind(RK)) {
          // VPReductionPHIRecipes for AnyOf reductions expect a boolean as
          // start value; compare the final value from the main vector loop
          // to the start value.
          ResumeV = Builder.CreateICmpNE(ResumeV, StartV);
          if (auto *I = dyn_cast<Instruction>(ResumeV))
            InstsToMove.push_back(I);
        } else {
          assert(SentinelVPV && "expected to find icmp using RdxResult");
          if (auto *FreezeI = dyn_cast<FreezeInst>(StartV))
            ToFrozen[FreezeI->getOperand(0)] = StartV;

          // Adjust resume: select(icmp eq ResumeV, StartV), Sentinel, ResumeV
          Value *Cmp = Builder.CreateICmpEQ(ResumeV, StartV);
          if (auto *I = dyn_cast<Instruction>(Cmp))
            InstsToMove.push_back(I);
          ResumeV = Builder.CreateSelect(Cmp, SentinelVPV->getLiveInIRValue(),
                                         ResumeV);
          if (auto *I = dyn_cast<Instruction>(ResumeV))
            InstsToMove.push_back(I);
        }
      } else {
        VPValue *StartVal = Plan.getOrAddLiveIn(ResumeV);
        auto *PhiR = dyn_cast<VPReductionPHIRecipe>(&R);
        if (auto *VPI = dyn_cast<VPInstruction>(PhiR->getStartValue())) {
          assert(VPI->getOpcode() == VPInstruction::ReductionStartVector &&
                 "unexpected start value");
          // Partial sub-reductions always start at 0 and account for the
          // reduction start value in a final subtraction. Update it to use the
          // resume value from the main vector loop.
          if (PhiR->getVFScaleFactor() > 1 &&
              PhiR->getRecurrenceKind() == RecurKind::Sub) {
            auto *Sub = cast<VPInstruction>(RdxResult->getSingleUser());
            assert(Sub->getOpcode() == Instruction::Sub && "Unexpected opcode");
            assert(isa<VPIRValue>(Sub->getOperand(0)) &&
                   "Expected operand to match the original start value of the "
                   "reduction");
            assert(VPlanPatternMatch::match(VPI->getOperand(0),
                                            VPlanPatternMatch::m_ZeroInt()) &&
                   "Expected start value for partial sub-reduction to start at "
                   "zero");
            Sub->setOperand(0, StartVal);
          } else
            VPI->setOperand(0, StartVal);
          continue;
        }
      }
    } else {
      // Retrieve the induction resume values for wide inductions from
      // their original phi nodes in the scalar loop.
      PHINode *IndPhi = cast<VPWidenInductionRecipe>(&R)->getPHINode();
      // Hook up to the PHINode generated by a ResumePhi recipe of main
      // loop VPlan, which feeds the scalar loop.
      ResumeV = IndPhi->getIncomingValueForBlock(L->getLoopPreheader());
    }
    assert(ResumeV && "Must have a resume value");
    VPValue *StartVal = Plan.getOrAddLiveIn(ResumeV);
    cast<VPHeaderPHIRecipe>(&R)->setStartValue(StartVal);
  }

  // For some VPValues in the epilogue plan we must re-use the generated IR
  // values from the main plan. Replace them with live-in VPValues.
  // TODO: This is a workaround needed for epilogue vectorization and it
  // should be removed once induction resume value creation is done
  // directly in VPlan.
  for (auto &R : make_early_inc_range(*Plan.getEntry())) {
    // Re-use frozen values from the main plan for Freeze VPInstructions in the
    // epilogue plan. This ensures all users use the same frozen value.
    auto *VPI = dyn_cast<VPInstruction>(&R);
    if (VPI && VPI->getOpcode() == Instruction::Freeze) {
      VPI->replaceAllUsesWith(Plan.getOrAddLiveIn(
          ToFrozen.lookup(VPI->getOperand(0)->getLiveInIRValue())));
      continue;
    }

    // Re-use the trip count and steps expanded for the main loop, as
    // skeleton creation needs it as a value that dominates both the scalar
    // and vector epilogue loops
    auto *ExpandR = dyn_cast<VPExpandSCEVRecipe>(&R);
    if (!ExpandR)
      continue;
    VPValue *ExpandedVal =
        Plan.getOrAddLiveIn(ExpandedSCEVs.lookup(ExpandR->getSCEV()));
    ExpandR->replaceAllUsesWith(ExpandedVal);
    if (Plan.getTripCount() == ExpandR)
      Plan.resetTripCount(ExpandedVal);
    ExpandR->eraseFromParent();
  }

  auto VScale = Config.getVScaleForTuning();
  unsigned MainLoopStep =
      estimateElementCount(EPI.MainLoopVF * EPI.MainLoopUF, VScale);
  unsigned EpilogueLoopStep =
      estimateElementCount(EPI.EpilogueVF * EPI.EpilogueUF, VScale);
  RUN_VPLAN_PASS(
      VPlanTransforms::addMinimumVectorEpilogueIterationCheck, Plan,
      EPI.VectorTripCount, CM.requiresScalarEpilogue(EPI.EpilogueVF.isVector()),
      EPI.EpilogueVF, EPI.EpilogueUF, MainLoopStep, EpilogueLoopStep, SE);

  return InstsToMove;
}

static void
fixScalarResumeValuesFromBypass(BasicBlock *BypassBlock, Loop *L,
                                VPlan &BestEpiPlan,
                                ArrayRef<VPInstruction *> ResumeValues) {
  // Fix resume values from the additional bypass block.
  BasicBlock *PH = L->getLoopPreheader();
  for (auto *Pred : predecessors(PH)) {
    for (PHINode &Phi : PH->phis()) {
      if (Phi.getBasicBlockIndex(Pred) != -1)
        continue;
      Phi.addIncoming(Phi.getIncomingValueForBlock(BypassBlock), Pred);
    }
  }
  auto *ScalarPH = cast<VPIRBasicBlock>(BestEpiPlan.getScalarPreheader());
  if (ScalarPH->hasPredecessors()) {
    // Fix resume values for inductions and reductions from the additional
    // bypass block using the incoming values from the main loop's resume phis.
    // ResumeValues correspond 1:1 with the scalar loop header phis.
    for (auto [ResumeV, HeaderPhi] :
         zip(ResumeValues, BestEpiPlan.getScalarHeader()->phis())) {
      auto *HeaderPhiR = cast<VPIRPhi>(&HeaderPhi);
      auto *EpiResumePhi =
          cast<PHINode>(HeaderPhiR->getIRPhi().getIncomingValueForBlock(PH));
      if (EpiResumePhi->getBasicBlockIndex(BypassBlock) == -1)
        continue;
      auto *MainResumePhi = cast<PHINode>(ResumeV->getUnderlyingValue());
      EpiResumePhi->setIncomingValueForBlock(
          BypassBlock, MainResumePhi->getIncomingValueForBlock(BypassBlock));
    }
  }
}

/// Connect the epilogue vector loop generated for \p EpiPlan to the main vector
/// loop, after both plans have executed, updating branches from the iteration
/// and runtime checks of the main loop, as well as updating various phis. \p
/// InstsToMove contains instructions that need to be moved to the preheader of
/// the epilogue vector loop.
static void connectEpilogueVectorLoop(VPlan &EpiPlan, Loop *L,
                                      EpilogueLoopVectorizationInfo &EPI,
                                      DominatorTree *DT,
                                      GeneratedRTChecks &Checks,
                                      ArrayRef<Instruction *> InstsToMove,
                                      ArrayRef<VPInstruction *> ResumeValues) {
  BasicBlock *VecEpilogueIterationCountCheck =
      cast<VPIRBasicBlock>(EpiPlan.getEntry())->getIRBasicBlock();

  BasicBlock *VecEpiloguePreHeader =
      cast<CondBrInst>(VecEpilogueIterationCountCheck->getTerminator())
          ->getSuccessor(1);
  // Adjust the control flow taking the state info from the main loop
  // vectorization into account.
  assert(EPI.MainLoopIterationCountCheck && EPI.EpilogueIterationCountCheck &&
         "expected this to be saved from the previous pass.");
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);
  EPI.MainLoopIterationCountCheck->getTerminator()->replaceUsesOfWith(
      VecEpilogueIterationCountCheck, VecEpiloguePreHeader);

  DTU.applyUpdates({{DominatorTree::Delete, EPI.MainLoopIterationCountCheck,
                     VecEpilogueIterationCountCheck},
                    {DominatorTree::Insert, EPI.MainLoopIterationCountCheck,
                     VecEpiloguePreHeader}});

  BasicBlock *ScalarPH =
      cast<VPIRBasicBlock>(EpiPlan.getScalarPreheader())->getIRBasicBlock();
  EPI.EpilogueIterationCountCheck->getTerminator()->replaceUsesOfWith(
      VecEpilogueIterationCountCheck, ScalarPH);
  DTU.applyUpdates(
      {{DominatorTree::Delete, EPI.EpilogueIterationCountCheck,
        VecEpilogueIterationCountCheck},
       {DominatorTree::Insert, EPI.EpilogueIterationCountCheck, ScalarPH}});

  // Adjust the terminators of runtime check blocks and phis using them.
  BasicBlock *SCEVCheckBlock = Checks.getSCEVChecks().second;
  BasicBlock *MemCheckBlock = Checks.getMemRuntimeChecks().second;
  if (SCEVCheckBlock) {
    SCEVCheckBlock->getTerminator()->replaceUsesOfWith(
        VecEpilogueIterationCountCheck, ScalarPH);
    DTU.applyUpdates({{DominatorTree::Delete, SCEVCheckBlock,
                       VecEpilogueIterationCountCheck},
                      {DominatorTree::Insert, SCEVCheckBlock, ScalarPH}});
  }
  if (MemCheckBlock) {
    MemCheckBlock->getTerminator()->replaceUsesOfWith(
        VecEpilogueIterationCountCheck, ScalarPH);
    DTU.applyUpdates(
        {{DominatorTree::Delete, MemCheckBlock, VecEpilogueIterationCountCheck},
         {DominatorTree::Insert, MemCheckBlock, ScalarPH}});
  }

  // The vec.epilog.iter.check block may contain Phi nodes from inductions
  // or reductions which merge control-flow from the latch block and the
  // middle block. Update the incoming values here and move the Phi into the
  // preheader.
  SmallVector<PHINode *, 4> PhisInBlock(
      llvm::make_pointer_range(VecEpilogueIterationCountCheck->phis()));

  for (PHINode *Phi : PhisInBlock) {
    Phi->moveBefore(VecEpiloguePreHeader->getFirstNonPHIIt());
    Phi->replaceIncomingBlockWith(
        VecEpilogueIterationCountCheck->getSinglePredecessor(),
        VecEpilogueIterationCountCheck);

    // If the phi doesn't have an incoming value from the
    // EpilogueIterationCountCheck, we are done. Otherwise remove the
    // incoming value and also those from other check blocks. This is needed
    // for reduction phis only.
    if (none_of(Phi->blocks(), [&](BasicBlock *IncB) {
          return EPI.EpilogueIterationCountCheck == IncB;
        }))
      continue;
    Phi->removeIncomingValue(EPI.EpilogueIterationCountCheck);
    if (SCEVCheckBlock)
      Phi->removeIncomingValue(SCEVCheckBlock);
    if (MemCheckBlock)
      Phi->removeIncomingValue(MemCheckBlock);
  }

  auto IP = VecEpiloguePreHeader->getFirstNonPHIIt();
  for (auto *I : InstsToMove)
    I->moveBefore(IP);

  // VecEpilogueIterationCountCheck conditionally skips over the epilogue loop
  // after executing the main loop. We need to update the resume values of
  // inductions and reductions during epilogue vectorization.
  fixScalarResumeValuesFromBypass(VecEpilogueIterationCountCheck, L, EpiPlan,
                                  ResumeValues);

  // Remove dead phis that were moved to the epilogue preheader but are unused
  // (e.g., resume phis for inductions not widened in the epilogue vector loop).
  for (PHINode &Phi : make_early_inc_range(VecEpiloguePreHeader->phis()))
    if (Phi.use_empty())
      Phi.eraseFromParent();
}

bool LoopVectorizePass::processLoop(Loop *L) {
  assert((EnableVPlanNativePath || L->isInnermost()) &&
         "VPlan-native path is not enabled. Only process inner loops.");

  LLVM_DEBUG(dbgs() << "\nLV: Checking a loop in '"
                    << L->getHeader()->getParent()->getName() << "' from "
                    << L->getLocStr() << "\n");

  LoopVectorizeHints Hints(L, InterleaveOnlyWhenForced, *ORE, TTI);

  LLVM_DEBUG(
      dbgs() << "LV: Loop hints:"
             << " force="
             << (Hints.getForce() == LoopVectorizeHints::FK_Disabled
                     ? "disabled"
                     : (Hints.getForce() == LoopVectorizeHints::FK_Enabled
                            ? "enabled"
                            : "?"))
             << " width=" << Hints.getWidth()
             << " interleave=" << Hints.getInterleave() << "\n");

  // Function containing loop
  Function *F = L->getHeader()->getParent();

  // Looking at the diagnostic output is the only way to determine if a loop
  // was vectorized (other than looking at the IR or machine code), so it
  // is important to generate an optimization remark for each loop. Most of
  // these messages are generated as OptimizationRemarkAnalysis. Remarks
  // generated as OptimizationRemark and OptimizationRemarkMissed are
  // less verbose reporting vectorized loops and unvectorized loops that may
  // benefit from vectorization, respectively.

  if (!Hints.allowVectorization(F, L, VectorizeOnlyWhenForced)) {
    LLVM_DEBUG(dbgs() << "LV: Loop hints prevent vectorization.\n");
    return false;
  }

  PredicatedScalarEvolution PSE(*SE, *L);

  // Query this against the original loop and save it here because the profile
  // of the original loop header may change as the transformation happens.
  bool OptForSize = llvm::shouldOptimizeForSize(
      L->getHeader(), PSI,
      PSI && PSI->hasProfileSummary() ? &GetBFI() : nullptr,
      PGSOQueryType::IRPass);

  // Check if it is legal to vectorize the loop.
  LoopVectorizationRequirements Requirements;
  LoopVectorizationLegality LVL(L, PSE, DT, TTI, TLI, F, *LAIs, LI, ORE,
                                &Requirements, &Hints, DB, AC,
                                /*AllowRuntimeSCEVChecks=*/!OptForSize, AA);
  if (!LVL.canVectorize(EnableVPlanNativePath)) {
    LLVM_DEBUG(dbgs() << "LV: Not vectorizing: Cannot prove legality.\n");
    Hints.emitRemarkWithHints();
    return false;
  }

  if (LVL.hasUncountableEarlyExit()) {
    if (!EnableEarlyExitVectorization) {
      reportVectorizationFailure("Auto-vectorization of loops with uncountable "
                                 "early exit is not enabled",
                                 "UncountableEarlyExitLoopsDisabled", ORE, L);
      return false;
    }
  }

  // Entrance to the VPlan-native vectorization path. Outer loops are processed
  // here. They may require CFG and instruction level transformations before
  // even evaluating whether vectorization is profitable. Since we cannot modify
  // the incoming IR, we need to build VPlan upfront in the vectorization
  // pipeline.
  if (!L->isInnermost())
    return processLoopInVPlanNativePath(L, PSE, LI, DT, &LVL, TTI, TLI, DB, AC,
                                        ORE, GetBFI, OptForSize, Hints,
                                        Requirements);

  assert(L->isInnermost() && "Inner loop expected.");

  InterleavedAccessInfo IAI(PSE, L, DT, LI, LVL.getLAI());
  bool UseInterleaved = TTI->enableInterleavedAccessVectorization();

  // If an override option has been passed in for interleaved accesses, use it.
  if (EnableInterleavedMemAccesses.getNumOccurrences() > 0)
    UseInterleaved = EnableInterleavedMemAccesses;

  // Analyze interleaved memory accesses.
  if (UseInterleaved)
    IAI.analyzeInterleaving(useMaskedInterleavedAccesses(*TTI));

  if (LVL.hasUncountableEarlyExit()) {
    BasicBlock *LoopLatch = L->getLoopLatch();
    if (IAI.requiresScalarEpilogue() ||
        any_of(LVL.getCountableExitingBlocks(), not_equal_to(LoopLatch))) {
      reportVectorizationFailure("Auto-vectorization of early exit loops "
                                 "requiring a scalar epilogue is unsupported",
                                 "UncountableEarlyExitUnsupported", ORE, L);
      return false;
    }
  }

  // Check the function attributes and profiles to find out if this function
  // should be optimized for size.
  EpilogueLowering SEL =
      getEpilogueLowering(F, L, Hints, OptForSize, TTI, TLI, LVL, &IAI);

  // Check the loop for a trip count threshold: vectorize loops with a tiny trip
  // count by optimizing for size, to minimize overheads.
  auto ExpectedTC = getSmallBestKnownTC(PSE, L);
  if (ExpectedTC && ExpectedTC->isFixed() &&
      ExpectedTC->getFixedValue() < TinyTripCountVectorThreshold) {
    LLVM_DEBUG(dbgs() << "LV: Found a loop with a very small trip count. "
                      << "This loop is worth vectorizing only if no scalar "
                      << "iteration overheads are incurred.");
    if (Hints.getForce() == LoopVectorizeHints::FK_Enabled)
      LLVM_DEBUG(dbgs() << " But vectorizing was explicitly forced.\n");
    else {
      LLVM_DEBUG(dbgs() << "\n");
      // Tail-folded loops are efficient even when the loop
      // iteration count is low. However, setting the epilogue policy to
      // `CM_EpilogueNotAllowedLowTripLoop` prevents vectorizing loops
      // with runtime checks. It's more effective to let
      // `isOutsideLoopWorkProfitable` determine if vectorization is
      // beneficial for the loop.
      if (SEL != CM_EpilogueNotNeededFoldTail)
        SEL = CM_EpilogueNotAllowedLowTripLoop;
    }
  }

  // Check the function attributes to see if implicit floats or vectors are
  // allowed.
  if (F->hasFnAttribute(Attribute::NoImplicitFloat)) {
    reportVectorizationFailure(
        "Can't vectorize when the NoImplicitFloat attribute is used",
        "loop not vectorized due to NoImplicitFloat attribute",
        "NoImplicitFloat", ORE, L);
    Hints.emitRemarkWithHints();
    return false;
  }

  // Check if the target supports potentially unsafe FP vectorization.
  // FIXME: Add a check for the type of safety issue (denormal, signaling)
  // for the target we're vectorizing for, to make sure none of the
  // additional fp-math flags can help.
  if (Hints.isPotentiallyUnsafe() &&
      TTI->isFPVectorizationPotentiallyUnsafe()) {
    reportVectorizationFailure(
        "Potentially unsafe FP op prevents vectorization",
        "loop not vectorized due to unsafe FP support.",
        "UnsafeFP", ORE, L);
    Hints.emitRemarkWithHints();
    return false;
  }

  bool AllowOrderedReductions;
  // If the flag is set, use that instead and override the TTI behaviour.
  if (ForceOrderedReductions.getNumOccurrences() > 0)
    AllowOrderedReductions = ForceOrderedReductions;
  else
    AllowOrderedReductions = TTI->enableOrderedReductions();
  if (!LVL.canVectorizeFPMath(AllowOrderedReductions)) {
    ORE->emit([&]() {
      auto *ExactFPMathInst = Requirements.getExactFPInst();
      return OptimizationRemarkAnalysisFPCommute(DEBUG_TYPE, "CantReorderFPOps",
                                                 ExactFPMathInst->getDebugLoc(),
                                                 ExactFPMathInst->getParent())
             << "loop not vectorized: cannot prove it is safe to reorder "
                "floating-point operations";
    });
    LLVM_DEBUG(dbgs() << "LV: loop not vectorized: cannot prove it is safe to "
                         "reorder floating-point operations\n");
    Hints.emitRemarkWithHints();
    return false;
  }

  // Use the cost model.
  VFSelectionContext Config(*TTI, &LVL, L, *F, PSE, DB, ORE, &Hints,
                            OptForSize);
  LoopVectorizationCostModel CM(SEL, L, PSE, LI, &LVL, *TTI, TLI, AC, ORE,
                                GetBFI, F, &Hints, IAI, Config);
  // Use the planner for vectorization.
  LoopVectorizationPlanner LVP(L, LI, DT, TLI, *TTI, &LVL, CM, Config, IAI, PSE,
                               Hints, ORE);

  // Get user vectorization factor and interleave count.
  ElementCount UserVF = Hints.getWidth();
  unsigned UserIC = Hints.getInterleave();
  if (UserIC > 1 && !LVL.isSafeForAnyVectorWidth())
    UserIC = 1;

  // Plan how to best vectorize.
  LVP.plan(UserVF, UserIC);
  auto [VF, BestPlanPtr] = LVP.computeBestVF();
  unsigned IC = 1;

  if (ORE->allowExtraAnalysis(LV_NAME))
    LVP.emitInvalidCostRemarks(ORE);

  GeneratedRTChecks Checks(PSE, DT, LI, TTI, Config.CostKind);
  if (LVP.hasPlanWithVF(VF.Width)) {
    // Select the interleave count.
    IC = LVP.selectInterleaveCount(*BestPlanPtr, VF.Width, VF.Cost);

    unsigned SelectedIC = std::max(IC, UserIC);
    //  Optimistically generate runtime checks if they are needed. Drop them if
    //  they turn out to not be profitable.
    if (VF.Width.isVector() || SelectedIC > 1) {
      Checks.create(L, *LVL.getLAI(), PSE.getPredicate(), VF.Width, SelectedIC,
                    *ORE);

      // Bail out early if either the SCEV or memory runtime checks are known to
      // fail. In that case, the vector loop would never execute.
      using namespace llvm::PatternMatch;
      if (Checks.getSCEVChecks().first &&
          match(Checks.getSCEVChecks().first, m_One()))
        return false;
      if (Checks.getMemRuntimeChecks().first &&
          match(Checks.getMemRuntimeChecks().first, m_One()))
        return false;
    }

    // Check if it is profitable to vectorize with runtime checks.
    bool ForceVectorization =
        Hints.getForce() == LoopVectorizeHints::FK_Enabled;
    VPCostContext CostCtx(CM.TTI, *CM.TLI, *BestPlanPtr, CM, Config.CostKind,
                          CM.PSE, L);
    if (!ForceVectorization &&
        !isOutsideLoopWorkProfitable(Checks, VF, L, PSE, CostCtx, *BestPlanPtr,
                                     SEL, Config.getVScaleForTuning())) {
      ORE->emit([&]() {
        return OptimizationRemarkAnalysisAliasing(
                   DEBUG_TYPE, "CantReorderMemOps", L->getStartLoc(),
                   L->getHeader())
               << "loop not vectorized: cannot prove it is safe to reorder "
                  "memory operations";
      });
      LLVM_DEBUG(dbgs() << "LV: Too many memory checks needed.\n");
      Hints.emitRemarkWithHints();
      return false;
    }
  }

  // Identify the diagnostic messages that should be produced.
  std::pair<StringRef, std::string> VecDiagMsg, IntDiagMsg;
  bool VectorizeLoop = true, InterleaveLoop = true;
  if (VF.Width.isScalar()) {
    LLVM_DEBUG(dbgs() << "LV: Vectorization is possible but not beneficial.\n");
    VecDiagMsg = {
        "VectorizationNotBeneficial",
        "the cost-model indicates that vectorization is not beneficial"};
    VectorizeLoop = false;
  }

  if (UserIC == 1 && Hints.getInterleave() > 1) {
    assert(!LVL.isSafeForAnyVectorWidth() &&
           "UserIC should only be ignored due to unsafe dependencies");
    LLVM_DEBUG(dbgs() << "LV: Ignoring user-specified interleave count.\n");
    IntDiagMsg = {"InterleavingUnsafe",
                  "Ignoring user-specified interleave count due to possibly "
                  "unsafe dependencies in the loop."};
    InterleaveLoop = false;
  } else if (!LVP.hasPlanWithVF(VF.Width) && UserIC > 1) {
    // Tell the user interleaving was avoided up-front, despite being explicitly
    // requested.
    LLVM_DEBUG(dbgs() << "LV: Ignoring UserIC, because vectorization and "
                         "interleaving should be avoided up front\n");
    IntDiagMsg = {"InterleavingAvoided",
                  "Ignoring UserIC, because interleaving was avoided up front"};
    InterleaveLoop = false;
  } else if (IC == 1 && UserIC <= 1) {
    // Tell the user interleaving is not beneficial.
    LLVM_DEBUG(dbgs() << "LV: Interleaving is not beneficial.\n");
    IntDiagMsg = {
        "InterleavingNotBeneficial",
        "the cost-model indicates that interleaving is not beneficial"};
    InterleaveLoop = false;
    if (UserIC == 1) {
      IntDiagMsg.first = "InterleavingNotBeneficialAndDisabled";
      IntDiagMsg.second +=
          " and is explicitly disabled or interleave count is set to 1";
    }
  } else if (IC > 1 && UserIC == 1) {
    // Tell the user interleaving is beneficial, but it explicitly disabled.
    LLVM_DEBUG(dbgs() << "LV: Interleaving is beneficial but is explicitly "
                         "disabled.\n");
    IntDiagMsg = {"InterleavingBeneficialButDisabled",
                  "the cost-model indicates that interleaving is beneficial "
                  "but is explicitly disabled or interleave count is set to 1"};
    InterleaveLoop = false;
  }

  // If there is a histogram in the loop, do not just interleave without
  // vectorizing. The order of operations will be incorrect without the
  // histogram intrinsics, which are only used for recipes with VF > 1.
  if (!VectorizeLoop && InterleaveLoop && LVL.hasHistograms()) {
    LLVM_DEBUG(dbgs() << "LV: Not interleaving without vectorization due "
                      << "to histogram operations.\n");
    IntDiagMsg = {
        "HistogramPreventsScalarInterleaving",
        "Unable to interleave without vectorization due to constraints on "
        "the order of histogram operations"};
    InterleaveLoop = false;
  }

  // Override IC if user provided an interleave count.
  IC = UserIC > 0 ? UserIC : IC;

  // Emit diagnostic messages, if any.
  const char *VAPassName = Hints.vectorizeAnalysisPassName();
  if (!VectorizeLoop && !InterleaveLoop) {
    // Do not vectorize or interleaving the loop.
    ORE->emit([&]() {
      return OptimizationRemarkMissed(VAPassName, VecDiagMsg.first,
                                      L->getStartLoc(), L->getHeader())
             << VecDiagMsg.second;
    });
    ORE->emit([&]() {
      return OptimizationRemarkMissed(LV_NAME, IntDiagMsg.first,
                                      L->getStartLoc(), L->getHeader())
             << IntDiagMsg.second;
    });
    return false;
  }

  if (!VectorizeLoop && InterleaveLoop) {
    LLVM_DEBUG(dbgs() << "LV: Interleave Count is " << IC << '\n');
    ORE->emit([&]() {
      return OptimizationRemarkAnalysis(VAPassName, VecDiagMsg.first,
                                        L->getStartLoc(), L->getHeader())
             << VecDiagMsg.second;
    });
  } else if (VectorizeLoop && !InterleaveLoop) {
    LLVM_DEBUG(dbgs() << "LV: Found a vectorizable loop (" << VF.Width
                      << ") in " << L->getLocStr() << '\n');
    ORE->emit([&]() {
      return OptimizationRemarkAnalysis(LV_NAME, IntDiagMsg.first,
                                        L->getStartLoc(), L->getHeader())
             << IntDiagMsg.second;
    });
  } else if (VectorizeLoop && InterleaveLoop) {
    LLVM_DEBUG(dbgs() << "LV: Found a vectorizable loop (" << VF.Width
                      << ") in " << L->getLocStr() << '\n');
    LLVM_DEBUG(dbgs() << "LV: Interleave Count is " << IC << '\n');
  }

  // Report the vectorization decision.
  if (VF.Width.isScalar()) {
    using namespace ore;
    assert(IC > 1);
    ORE->emit([&]() {
      return OptimizationRemark(LV_NAME, "Interleaved", L->getStartLoc(),
                                L->getHeader())
             << "interleaved loop (interleaved count: "
             << NV("InterleaveCount", IC) << ")";
    });
  } else {
    // Report the vectorization decision.
    reportVectorization(ORE, L, VF, IC);
  }
  if (ORE->allowExtraAnalysis(LV_NAME))
    checkMixedPrecision(L, ORE);

  // If we decided that it is *legal* to interleave or vectorize the loop, then
  // do it.

  VPlan &BestPlan = *BestPlanPtr;
  // Consider vectorizing the epilogue too if it's profitable.
  std::unique_ptr<VPlan> EpiPlan =
      LVP.selectBestEpiloguePlan(BestPlan, VF.Width, IC);
  bool HasBranchWeights =
      hasBranchWeightMD(*L->getLoopLatch()->getTerminator());
  if (EpiPlan) {
    VPlan &BestEpiPlan = *EpiPlan;
    VPlan &BestMainPlan = BestPlan;
    ElementCount EpilogueVF = BestEpiPlan.getSingleVF();

    // The first pass vectorizes the main loop and creates a scalar epilogue
    // to be vectorized by executing the plan (potentially with a different
    // factor) again shortly afterwards.
    BestEpiPlan.getMiddleBlock()->setName("vec.epilog.middle.block");
    BestEpiPlan.getVectorPreheader()->setName("vec.epilog.ph");
    SmallVector<VPInstruction *> ResumeValues =
        preparePlanForMainVectorLoop(BestMainPlan, BestEpiPlan);
    EpilogueLoopVectorizationInfo EPI(VF.Width, IC, EpilogueVF, 1, BestEpiPlan);

    // Add minimum iteration check for the epilogue plan, followed by runtime
    // checks for the main plan.
    LVP.addMinimumIterationCheck(BestMainPlan, EPI.EpilogueVF, EPI.EpilogueUF,
                                 ElementCount::getFixed(0));
    LVP.attachRuntimeChecks(BestMainPlan, Checks, HasBranchWeights);
    RUN_VPLAN_PASS(VPlanTransforms::addIterationCountCheckBlock, BestMainPlan,
                   EPI.MainLoopVF, EPI.MainLoopUF,
                   CM.requiresScalarEpilogue(EPI.MainLoopVF.isVector()), L,
                   HasBranchWeights ? MinItersBypassWeights : nullptr,
                   L->getLoopPredecessor()->getTerminator()->getDebugLoc(),
                   PSE);

    EpilogueVectorizerMainLoop MainILV(L, PSE, LI, DT, TTI, AC, EPI, &CM,
                                       Checks, BestMainPlan);
    auto ExpandedSCEVs = LVP.executePlan(
        EPI.MainLoopVF, EPI.MainLoopUF, BestMainPlan, MainILV, DT,
        LoopVectorizationPlanner::EpilogueVectorizationKind::MainLoop);
    ++LoopsVectorized;

    // Derive EPI fields from VPlan-generated IR.
    BasicBlock *EntryBB =
        cast<VPIRBasicBlock>(BestMainPlan.getEntry())->getIRBasicBlock();
    EntryBB->setName("iter.check");
    EPI.EpilogueIterationCountCheck = EntryBB;
    // The check chain is: Entry -> [SCEV] -> [Mem] -> MainCheck -> VecPH.
    // MainCheck is the non-bypass successor of the last runtime check block
    // (or Entry if there are no runtime checks).
    BasicBlock *LastCheck = EntryBB;
    if (BasicBlock *MemBB = Checks.getMemRuntimeChecks().second)
      LastCheck = MemBB;
    else if (BasicBlock *SCEVBB = Checks.getSCEVChecks().second)
      LastCheck = SCEVBB;
    BasicBlock *ScalarPH = L->getLoopPreheader();
    auto *BI = cast<CondBrInst>(LastCheck->getTerminator());
    EPI.MainLoopIterationCountCheck =
        BI->getSuccessor(BI->getSuccessor(0) == ScalarPH);

    // Second pass vectorizes the epilogue and adjusts the control flow
    // edges from the first pass.
    EpilogueVectorizerEpilogueLoop EpilogILV(L, PSE, LI, DT, TTI, AC, EPI, &CM,
                                             Checks, BestEpiPlan);
    SmallVector<Instruction *> InstsToMove = preparePlanForEpilogueVectorLoop(
        BestEpiPlan, L, ExpandedSCEVs, EPI, CM, Config, *PSE.getSE());
    LVP.attachRuntimeChecks(BestEpiPlan, Checks, HasBranchWeights);
    LVP.executePlan(
        EPI.EpilogueVF, EPI.EpilogueUF, BestEpiPlan, EpilogILV, DT,
        LoopVectorizationPlanner::EpilogueVectorizationKind::Epilogue);
    connectEpilogueVectorLoop(BestEpiPlan, L, EPI, DT, Checks, InstsToMove,
                              ResumeValues);
    ++LoopsEpilogueVectorized;
  } else {
    InnerLoopVectorizer LB(L, PSE, LI, DT, TTI, AC, VF.Width, IC, &CM, Checks,
                           BestPlan);
    LVP.addMinimumIterationCheck(BestPlan, VF.Width, IC,
                                 VF.MinProfitableTripCount);
    LVP.attachRuntimeChecks(BestPlan, Checks, HasBranchWeights);

    LVP.executePlan(VF.Width, IC, BestPlan, LB, DT);
    ++LoopsVectorized;
  }

  assert(DT->verify(DominatorTree::VerificationLevel::Fast) &&
         "DT not preserved correctly");
  assert(!verifyFunction(*F, &dbgs()));

  return true;
}

LoopVectorizeResult LoopVectorizePass::runImpl(Function &F) {

  // Don't attempt if
  // 1. the target claims to have no vector registers, and
  // 2. interleaving won't help ILP.
  //
  // The second condition is necessary because, even if the target has no
  // vector registers, loop vectorization may still enable scalar
  // interleaving.
  if (!TTI->getNumberOfRegisters(TTI->getRegisterClassForType(true)) &&
      TTI->getMaxInterleaveFactor(ElementCount::getFixed(1)) < 2)
    return LoopVectorizeResult(false, false);

  bool Changed = false, CFGChanged = false;

  // The vectorizer requires loops to be in simplified form.
  // Since simplification may add new inner loops, it has to run before the
  // legality and profitability checks. This means running the loop vectorizer
  // will simplify all loops, regardless of whether anything end up being
  // vectorized.
  for (const auto &L : *LI)
    Changed |= CFGChanged |=
        simplifyLoop(L, DT, LI, SE, AC, nullptr, false /* PreserveLCSSA */);

  // Build up a worklist of inner-loops to vectorize. This is necessary as
  // the act of vectorizing or partially unrolling a loop creates new loops
  // and can invalidate iterators across the loops.
  SmallVector<Loop *, 8> Worklist;

  for (Loop *L : *LI)
    collectSupportedLoops(*L, LI, ORE, Worklist);

  LoopsAnalyzed += Worklist.size();

  // Now walk the identified inner loops.
  while (!Worklist.empty()) {
    Loop *L = Worklist.pop_back_val();

    // For the inner loops we actually process, form LCSSA to simplify the
    // transform.
    Changed |= formLCSSARecursively(*L, *DT, LI, SE);

    Changed |= CFGChanged |= processLoop(L);

    if (Changed) {
      LAIs->clear();

#ifndef NDEBUG
      if (VerifySCEV)
        SE->verify();
#endif
    }
  }

  // Process each loop nest in the function.
  return LoopVectorizeResult(Changed, CFGChanged);
}

PreservedAnalyses LoopVectorizePass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  LI = &AM.getResult<LoopAnalysis>(F);
  // There are no loops in the function. Return before computing other
  // expensive analyses.
  if (LI->empty())
    return PreservedAnalyses::all();
  SE = &AM.getResult<ScalarEvolutionAnalysis>(F);
  TTI = &AM.getResult<TargetIRAnalysis>(F);
  DT = &AM.getResult<DominatorTreeAnalysis>(F);
  TLI = &AM.getResult<TargetLibraryAnalysis>(F);
  AC = &AM.getResult<AssumptionAnalysis>(F);
  DB = &AM.getResult<DemandedBitsAnalysis>(F);
  ORE = &AM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  LAIs = &AM.getResult<LoopAccessAnalysis>(F);
  AA = &AM.getResult<AAManager>(F);

  auto &MAMProxy = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
  PSI = MAMProxy.getCachedResult<ProfileSummaryAnalysis>(*F.getParent());
  GetBFI = [&AM, &F]() -> BlockFrequencyInfo & {
    return AM.getResult<BlockFrequencyAnalysis>(F);
  };
  LoopVectorizeResult Result = runImpl(F);
  if (!Result.MadeAnyChange)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;

  if (isAssignmentTrackingEnabled(*F.getParent())) {
    for (auto &BB : F)
      RemoveRedundantDbgInstrs(&BB);
  }

  PA.preserve<LoopAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<ScalarEvolutionAnalysis>();
  PA.preserve<LoopAccessAnalysis>();

  if (Result.MadeCFGChange) {
    // Making CFG changes likely means a loop got vectorized. Indicate that
    // extra simplification passes should be run.
    // TODO: MadeCFGChanges is not a prefect proxy. Extra passes should only
    // be run if runtime checks have been added.
    AM.getResult<ShouldRunExtraVectorPasses>(F);
    PA.preserve<ShouldRunExtraVectorPasses>();
  } else {
    PA.preserveSet<CFGAnalyses>();
  }
  return PA;
}

void LoopVectorizePass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<LoopVectorizePass> *>(this)->printPipeline(
      OS, MapClassName2PassName);

  OS << '<';
  OS << (InterleaveOnlyWhenForced ? "" : "no-") << "interleave-forced-only;";
  OS << (VectorizeOnlyWhenForced ? "" : "no-") << "vectorize-forced-only;";
  OS << '>';
}
