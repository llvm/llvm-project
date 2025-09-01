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

/// @{
/// Metadata attribute names
const char LLVMLoopVectorizeFollowupAll[] = "llvm.loop.vectorize.followup_all";
const char LLVMLoopVectorizeFollowupVectorized[] =
    "llvm.loop.vectorize.followup_vectorized";
const char LLVMLoopVectorizeFollowupEpilogue[] =
    "llvm.loop.vectorize.followup_epilogue";
/// @}

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

// Option prefer-predicate-over-epilogue indicates that an epilogue is undesired,
// that predication is preferred, and this lists all options. I.e., the
// vectorizer will try to fold the tail-loop (epilogue) into the vector body
// and predicate the instructions accordingly. If tail-folding fails, there are
// different fallback strategies depending on these values:
namespace PreferPredicateTy {
  enum Option {
    ScalarEpilogue = 0,
    PredicateElseScalarEpilogue,
    PredicateOrDontVectorize
  };
} // namespace PreferPredicateTy

static cl::opt<PreferPredicateTy::Option> PreferPredicateOverEpilogue(
    "prefer-predicate-over-epilogue",
    cl::init(PreferPredicateTy::ScalarEpilogue),
    cl::Hidden,
    cl::desc("Tail-folding and predication preferences over creating a scalar "
             "epilogue loop."),
    cl::values(clEnumValN(PreferPredicateTy::ScalarEpilogue,
                         "scalar-epilogue",
                         "Don't tail-predicate loops, create scalar epilogue"),
              clEnumValN(PreferPredicateTy::PredicateElseScalarEpilogue,
                         "predicate-else-scalar-epilogue",
                         "prefer tail-folding, create scalar epilogue if tail "
                         "folding fails."),
              clEnumValN(PreferPredicateTy::PredicateOrDontVectorize,
                         "predicate-dont-vectorize",
                         "prefers tail-folding, don't attempt vectorization if "
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
        clEnumValN(TailFoldingStyle::DataAndControlFlowWithoutRuntimeCheck,
                   "data-and-control-without-rt-check",
                   "Similar to data-and-control, but remove the runtime check"),
        clEnumValN(TailFoldingStyle::DataWithEVL, "data-with-evl",
                   "Use predicated EVL instructions for tail folding. If EVL "
                   "is unsupported, fallback to data-without-lane-mask.")));

static cl::opt<bool> MaximizeBandwidth(
    "vectorizer-maximize-bandwidth", cl::init(false), cl::Hidden,
    cl::desc("Maximize bandwidth when selecting vectorization factor which "
             "will be determined by the smallest type in loop."));

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

static cl::opt<bool> ForceTargetSupportsScalableVectors(
    "force-target-supports-scalable-vectors", cl::init(false), cl::Hidden,
    cl::desc(
        "Pretend that scalable vectors are supported, even if the target does "
        "not support them. This flag should only be used for testing."));

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
static cl::opt<unsigned> NumberOfStoresToPredicate(
    "vectorize-num-stores-pred", cl::init(1), cl::Hidden,
    cl::desc("Max number of stores to be predicated behind an if."));

static cl::opt<bool> EnableIndVarRegisterHeur(
    "enable-ind-var-reg-heur", cl::init(true), cl::Hidden,
    cl::desc("Count the induction variable only once when interleaving"));

static cl::opt<bool> EnableCondStoresVectorization(
    "enable-cond-stores-vec", cl::init(true), cl::Hidden,
    cl::desc("Enable if predication of stores during vectorization."));

static cl::opt<unsigned> MaxNestedScalarReductionIC(
    "max-nested-scalar-reduction-interleave", cl::init(2), cl::Hidden,
    cl::desc("The maximum interleave count to use when interleaving a scalar "
             "reduction in a nested loop."));

static cl::opt<bool>
    PreferInLoopReductions("prefer-inloop-reductions", cl::init(false),
                           cl::Hidden,
                           cl::desc("Prefer in-loop vector reductions, "
                                    "overriding the targets preference."));

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
                          cl::desc("Verfiy VPlans after VPlan transforms."));

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

static cl::opt<bool> UseWiderVFIfCallVariantsPresent(
    "vectorizer-maximize-bandwidth-for-vector-calls", cl::init(true),
    cl::Hidden,
    cl::desc("Try wider VFs if they enable the use of vector variants"));

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

/// Returns "best known" trip count, which is either a valid positive trip count
/// or std::nullopt when an estimate cannot be made (including when the trip
/// count would overflow), for the specified loop \p L as defined by the
/// following procedure:
///   1) Returns exact trip count if it is known.
///   2) Returns expected trip count according to profile data if any.
///   3) Returns upper bound estimate if known, and if \p CanUseConstantMax.
///   4) Returns std::nullopt if all of the above failed.
static std::optional<ElementCount>
getSmallBestKnownTC(PredicatedScalarEvolution &PSE, Loop *L,
                    bool CanUseConstantMax = true) {
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
                      LoopVectorizationCostModel *CM, BlockFrequencyInfo *BFI,
                      ProfileSummaryInfo *PSI, GeneratedRTChecks &RTChecks,
                      VPlan &Plan)
      : OrigLoop(OrigLoop), PSE(PSE), LI(LI), DT(DT), TTI(TTI), AC(AC),
        VF(VecWidth), UF(UnrollFactor), Builder(PSE.getSE()->getContext()),
        Cost(CM), BFI(BFI), PSI(PSI), RTChecks(RTChecks), Plan(Plan),
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

  /// Returns the original loop trip count.
  Value *getTripCount() const { return TripCount; }

  /// Used to set the trip count after ILV's construction and after the
  /// preheader block has been executed. Note that this always holds the trip
  /// count of the original loop for both main loop and epilogue vectorization.
  void setTripCount(Value *TC) { TripCount = TC; }

  /// Return the additional bypass block which targets the scalar loop by
  /// skipping the epilogue loop after completing the main loop.
  BasicBlock *getAdditionalBypassBlock() const {
    assert(AdditionalBypassBlock &&
           "Trying to access AdditionalBypassBlock but it has not been set");
    return AdditionalBypassBlock;
  }

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

  /// The vector-loop preheader.
  BasicBlock *LoopVectorPreHeader = nullptr;

  /// Trip count of the original loop.
  Value *TripCount = nullptr;

  /// The profitablity analysis.
  LoopVectorizationCostModel *Cost;

  /// BFI and PSI are used to check for profile guided size optimizations.
  BlockFrequencyInfo *BFI;
  ProfileSummaryInfo *PSI;

  /// Structure to hold information about generated runtime checks, responsible
  /// for cleaning the checks, if vectorization turns out unprofitable.
  GeneratedRTChecks &RTChecks;

  /// The additional bypass block which conditionally skips over the epilogue
  /// loop after executing the main loop. Needed to resume inductions and
  /// reductions during epilogue vectorization.
  BasicBlock *AdditionalBypassBlock = nullptr;

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
  Value *TripCount = nullptr;
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
      BlockFrequencyInfo *BFI, ProfileSummaryInfo *PSI,
      GeneratedRTChecks &Checks, VPlan &Plan, ElementCount VecWidth,
      ElementCount MinProfitableTripCount, unsigned UnrollFactor)
      : InnerLoopVectorizer(OrigLoop, PSE, LI, DT, TTI, AC, VecWidth,
                            UnrollFactor, CM, BFI, PSI, Checks, Plan),
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
                             BlockFrequencyInfo *BFI, ProfileSummaryInfo *PSI,
                             GeneratedRTChecks &Check, VPlan &Plan)
      : InnerLoopAndEpilogueVectorizer(OrigLoop, PSE, LI, DT, TTI, AC, EPI, CM,
                                       BFI, PSI, Check, Plan, EPI.MainLoopVF,
                                       EPI.MainLoopVF, EPI.MainLoopUF) {}
  /// Implements the interface for creating a vectorized skeleton using the
  /// *main loop* strategy (i.e., the first pass of VPlan execution).
  BasicBlock *createVectorizedLoopSkeleton() final;

protected:
  /// Introduces a new VPIRBasicBlock for \p CheckIRBB to Plan between the
  /// vector preheader and its predecessor, also connecting the new block to the
  /// scalar preheader.
  void introduceCheckBlockInVPlan(BasicBlock *CheckIRBB);

  // Create a check to see if the main vector loop should be executed
  Value *createIterationCountCheck(ElementCount VF, unsigned UF) const;

  /// Emits an iteration count bypass check once for the main loop (when \p
  /// ForEpilogue is false) and once for the epilogue loop (when \p
  /// ForEpilogue is true).
  BasicBlock *emitIterationCountCheck(BasicBlock *Bypass, bool ForEpilogue);
  void printDebugTracesAtStart() override;
  void printDebugTracesAtEnd() override;
};

// A specialized derived class of inner loop vectorizer that performs
// vectorization of *epilogue* loops in the process of vectorizing loops and
// their epilogues.
class EpilogueVectorizerEpilogueLoop : public InnerLoopAndEpilogueVectorizer {
public:
  EpilogueVectorizerEpilogueLoop(
      Loop *OrigLoop, PredicatedScalarEvolution &PSE, LoopInfo *LI,
      DominatorTree *DT, const TargetTransformInfo *TTI, AssumptionCache *AC,
      EpilogueLoopVectorizationInfo &EPI, LoopVectorizationCostModel *CM,
      BlockFrequencyInfo *BFI, ProfileSummaryInfo *PSI,
      GeneratedRTChecks &Checks, VPlan &Plan)
      : InnerLoopAndEpilogueVectorizer(OrigLoop, PSE, LI, DT, TTI, AC, EPI, CM,
                                       BFI, PSI, Checks, Plan, EPI.EpilogueVF,
                                       EPI.EpilogueVF, EPI.EpilogueUF) {
    TripCount = EPI.TripCount;
  }
  /// Implements the interface for creating a vectorized skeleton using the
  /// *epilogue loop* strategy (i.e., the second pass of VPlan execution).
  BasicBlock *createVectorizedLoopSkeleton() final;

protected:
  /// Emits an iteration count bypass check after the main vector loop has
  /// finished to see if there are any iterations left to execute by either
  /// the vector epilogue or the scalar epilogue.
  BasicBlock *emitMinimumVectorEpilogueIterCountCheck(
                                                      BasicBlock *Bypass,
                                                      BasicBlock *Insert);
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
createLVAnalysis(const char *PassName, StringRef RemarkName, Loop *TheLoop,
                 Instruction *I, DebugLoc DL = {}) {
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

/// Return a value for Step multiplied by VF.
Value *createStepForVF(IRBuilderBase &B, Type *Ty, ElementCount VF,
                       int64_t Step) {
  assert(Ty->isIntegerTy() && "Expected an integer step");
  ElementCount VFxStep = VF.multiplyCoefficientBy(Step);
  assert(isPowerOf2_64(VF.getKnownMinValue()) && "must pass power-of-2 VF");
  if (VF.isScalable() && isPowerOf2_64(Step)) {
    return B.CreateShl(
        B.CreateVScale(Ty),
        ConstantInt::get(Ty, Log2_64(VFxStep.getKnownMinValue())), "", true);
  }
  return B.CreateElementCount(Ty, VFxStep);
}

/// Return the runtime value for VF.
Value *getRuntimeVF(IRBuilderBase &B, Type *Ty, ElementCount VF) {
  return B.CreateElementCount(Ty, VF);
}

void reportVectorizationFailure(const StringRef DebugMsg,
                                const StringRef OREMsg, const StringRef ORETag,
                                OptimizationRemarkEmitter *ORE, Loop *TheLoop,
                                Instruction *I) {
  LLVM_DEBUG(debugVectorizationMessage("Not vectorizing: ", DebugMsg, I));
  LoopVectorizeHints Hints(TheLoop, true /* doesn't matter */, *ORE);
  ORE->emit(
      createLVAnalysis(Hints.vectorizeAnalysisPassName(), ORETag, TheLoop, I)
      << "loop not vectorized: " << OREMsg);
}

/// Reports an informative message: print \p Msg for debugging purposes as well
/// as an optimization remark. Uses either \p I as location of the remark, or
/// otherwise \p TheLoop. If \p DL is passed, use it as debug location for the
/// remark. If \p DL is passed, use it as debug location for the remark.
static void reportVectorizationInfo(const StringRef Msg, const StringRef ORETag,
                                    OptimizationRemarkEmitter *ORE,
                                    Loop *TheLoop, Instruction *I = nullptr,
                                    DebugLoc DL = {}) {
  LLVM_DEBUG(debugVectorizationMessage("", Msg, I));
  LoopVectorizeHints Hints(TheLoop, true /* doesn't matter */, *ORE);
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

// Loop vectorization cost-model hints how the scalar epilogue loop should be
// lowered.
enum ScalarEpilogueLowering {

  // The default: allowing scalar epilogues.
  CM_ScalarEpilogueAllowed,

  // Vectorization with OptForSize: don't allow epilogues.
  CM_ScalarEpilogueNotAllowedOptSize,

  // A special case of vectorisation with OptForSize: loops with a very small
  // trip count are considered for vectorization under OptForSize, thereby
  // making sure the cost of their loop body is dominant, free of runtime
  // guards and scalar iteration overheads.
  CM_ScalarEpilogueNotAllowedLowTripLoop,

  // Loop hint predicate indicating an epilogue is undesired.
  CM_ScalarEpilogueNotNeededUsePredicate,

  // Directive indicating we must either tail fold or not vectorize
  CM_ScalarEpilogueNotAllowedUsePredicate
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
  LoopVectorizationCostModel(ScalarEpilogueLowering SEL, Loop *L,
                             PredicatedScalarEvolution &PSE, LoopInfo *LI,
                             LoopVectorizationLegality *Legal,
                             const TargetTransformInfo &TTI,
                             const TargetLibraryInfo *TLI, DemandedBits *DB,
                             AssumptionCache *AC,
                             OptimizationRemarkEmitter *ORE, const Function *F,
                             const LoopVectorizeHints *Hints,
                             InterleavedAccessInfo &IAI,
                             ProfileSummaryInfo *PSI, BlockFrequencyInfo *BFI)
      : ScalarEpilogueStatus(SEL), TheLoop(L), PSE(PSE), LI(LI), Legal(Legal),
        TTI(TTI), TLI(TLI), DB(DB), AC(AC), ORE(ORE), TheFunction(F),
        Hints(Hints), InterleaveInfo(IAI) {
    if (TTI.supportsScalableVectors() || ForceTargetSupportsScalableVectors)
      initializeVScaleForTuning();
    CostKind = F->hasMinSize() ? TTI::TCK_CodeSize : TTI::TCK_RecipThroughput;
    // Query this against the original loop and save it here because the profile
    // of the original loop header may change as the transformation happens.
    OptForSize = llvm::shouldOptimizeForSize(L->getHeader(), PSI, BFI,
                                             PGSOQueryType::IRPass);
  }

  /// \return An upper bound for the vectorization factors (both fixed and
  /// scalable). If the factors are 0, vectorization and interleaving should be
  /// avoided up front.
  FixedScalableVFPair computeMaxVF(ElementCount UserVF, unsigned UserIC);

  /// \return True if runtime checks are required for vectorization, and false
  /// otherwise.
  bool runtimeChecksRequired();

  /// Setup cost-based decisions for user vectorization factor.
  /// \return true if the UserVF is a feasible VF to be chosen.
  bool selectUserVectorizationFactor(ElementCount UserVF) {
    collectNonVectorizedAndSetWideningDecisions(UserVF);
    return expectedCost(UserVF).isValid();
  }

  /// \return True if maximizing vector bandwidth is enabled by the target or
  /// user options, for the given register kind.
  bool useMaxBandwidth(TargetTransformInfo::RegisterKind RegKind);

  /// \return True if register pressure should be calculated for the given VF.
  bool shouldCalculateRegPressureForVF(ElementCount VF);

  /// \return The size (in bits) of the smallest and widest types in the code
  /// that needs to be vectorized. We ignore values that remain scalar such as
  /// 64 bit loop indices.
  std::pair<unsigned, unsigned> getSmallestAndWidestTypes();

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

  /// Collect all element types in the loop for which widening is needed.
  void collectElementTypesForWidening();

  /// Split reductions into those that happen in the loop, and those that happen
  /// outside. In loop reductions are collected into InLoopReductions.
  void collectInLoopReductions();

  /// Returns true if we should use strict in-order reductions for the given
  /// RdxDesc. This is true if the -enable-strict-reductions flag is passed,
  /// the IsOrdered flag of RdxDesc is set and we do not allow reordering
  /// of FP operations.
  bool useOrderedReductions(const RecurrenceDescriptor &RdxDesc) const {
    return !Hints->allowReordering() && RdxDesc.isOrdered();
  }

  /// \returns The smallest bitwidth each instruction can be represented with.
  /// The vector equivalents of these instructions should be truncated to this
  /// type.
  const MapVector<Instruction *, uint64_t> &getMinimalBitwidths() const {
    return MinBWs;
  }

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
    // Pseudo probe needs to be duplicated for each unrolled iteration and
    // vector lane so that profiled loop trip count can be accurately
    // accumulated instead of being under counted.
    if (isa<PseudoProbeInst>(I))
      return false;

    if (VF.isScalar())
      return true;

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
    for (unsigned Idx = 0; Idx < Grp->getFactor(); ++Idx) {
      if (auto *I = Grp->getMember(Idx)) {
        if (Grp->getInsertPos() == I)
          WideningDecisions[{I, VF}] = {W, InsertPosCost};
        else
          WideningDecisions[{I, VF}] = {W, OtherMemberCost};
      }
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

  /// Returns true if the target machine supports masked store operation
  /// for the given \p DataType and kind of access to \p Ptr.
  bool isLegalMaskedStore(Type *DataType, Value *Ptr, Align Alignment,
                          unsigned AddressSpace) const {
    return Legal->isConsecutivePtr(DataType, Ptr) &&
           TTI.isLegalMaskedStore(DataType, Alignment, AddressSpace);
  }

  /// Returns true if the target machine supports masked load operation
  /// for the given \p DataType and kind of access to \p Ptr.
  bool isLegalMaskedLoad(Type *DataType, Value *Ptr, Align Alignment,
                         unsigned AddressSpace) const {
    return Legal->isConsecutivePtr(DataType, Ptr) &&
           TTI.isLegalMaskedLoad(DataType, Alignment, AddressSpace);
  }

  /// Returns true if the target machine can represent \p V as a masked gather
  /// or scatter operation.
  bool isLegalGatherOrScatter(Value *V, ElementCount VF) {
    bool LI = isa<LoadInst>(V);
    bool SI = isa<StoreInst>(V);
    if (!LI && !SI)
      return false;
    auto *Ty = getLoadStoreType(V);
    Align Align = getLoadStoreAlignment(V);
    if (VF.isVector())
      Ty = VectorType::get(Ty, VF);
    return (LI && TTI.isLegalMaskedGather(Ty, Align)) ||
           (SI && TTI.isLegalMaskedScatter(Ty, Align));
  }

  /// Returns true if the target machine supports all of the reduction
  /// variables found for the given VF.
  bool canVectorizeReductions(ElementCount VF) const {
    return (all_of(Legal->getReductionVars(), [&](auto &Reduction) -> bool {
      const RecurrenceDescriptor &RdxDesc = Reduction.second;
      return TTI.isLegalToVectorizeReduction(RdxDesc, VF);
    }));
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
  bool isScalarWithPredication(Instruction *I, ElementCount VF) const;

  /// Returns true if \p I is an instruction that needs to be predicated
  /// at runtime.  The result is independent of the predication mechanism.
  /// Superset of instructions that return true for isScalarWithPredication.
  bool isPredicatedInst(Instruction *I) const;

  /// Return the costs for our two available strategies for lowering a
  /// div/rem operation which requires speculating at least one lane.
  /// First result is for scalarization (will be invalid for scalable
  /// vectors); second is for the safe-divisor strategy.
  std::pair<InstructionCost, InstructionCost>
  getDivRemSpeculationCost(Instruction *I,
                           ElementCount VF) const;

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
    if (!isScalarEpilogueAllowed()) {
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

  /// Returns true if a scalar epilogue is not allowed due to optsize or a
  /// loop hint annotation.
  bool isScalarEpilogueAllowed() const {
    return ScalarEpilogueStatus == CM_ScalarEpilogueAllowed;
  }

  /// Returns the TailFoldingStyle that is best for the current loop.
  TailFoldingStyle getTailFoldingStyle(bool IVUpdateMayOverflow = true) const {
    if (!ChosenTailFoldingStyle)
      return TailFoldingStyle::None;
    return IVUpdateMayOverflow ? ChosenTailFoldingStyle->first
                               : ChosenTailFoldingStyle->second;
  }

  /// Selects and saves TailFoldingStyle for 2 options - if IV update may
  /// overflow or not.
  /// \param IsScalableVF true if scalable vector factors enabled.
  /// \param UserIC User specific interleave count.
  void setTailFoldingStyles(bool IsScalableVF, unsigned UserIC) {
    assert(!ChosenTailFoldingStyle && "Tail folding must not be selected yet.");
    if (!Legal->canFoldTailByMasking()) {
      ChosenTailFoldingStyle = {TailFoldingStyle::None, TailFoldingStyle::None};
      return;
    }

    // Default to TTI preference, but allow command line override.
    ChosenTailFoldingStyle = {
        TTI.getPreferredTailFoldingStyle(/*IVUpdateMayOverflow=*/true),
        TTI.getPreferredTailFoldingStyle(/*IVUpdateMayOverflow=*/false)};
    if (ForceTailFoldingStyle.getNumOccurrences())
      ChosenTailFoldingStyle = {ForceTailFoldingStyle.getValue(),
                                ForceTailFoldingStyle.getValue()};

    if (ChosenTailFoldingStyle->first != TailFoldingStyle::DataWithEVL &&
        ChosenTailFoldingStyle->second != TailFoldingStyle::DataWithEVL)
      return;
    // Override EVL styles if needed.
    // FIXME: Investigate opportunity for fixed vector factor.
    bool EVLIsLegal = UserIC <= 1 && IsScalableVF &&
                      TTI.hasActiveVectorLength() && !EnableVPlanNativePath;
    if (EVLIsLegal)
      return;
    // If for some reason EVL mode is unsupported, fallback to a scalar epilogue
    // if it's allowed, or DataWithoutLaneMask otherwise.
    if (ScalarEpilogueStatus == CM_ScalarEpilogueAllowed ||
        ScalarEpilogueStatus == CM_ScalarEpilogueNotNeededUsePredicate)
      ChosenTailFoldingStyle = {TailFoldingStyle::None, TailFoldingStyle::None};
    else
      ChosenTailFoldingStyle = {TailFoldingStyle::DataWithoutLaneMask,
                                TailFoldingStyle::DataWithoutLaneMask};

    LLVM_DEBUG(
        dbgs() << "LV: Preference for VP intrinsics indicated. Will "
                  "not try to generate VP Intrinsics "
               << (UserIC > 1
                       ? "since interleave count specified is greater than 1.\n"
                       : "due to non-interleaving reasons.\n"));
  }

  /// Returns true if all loop blocks should be masked to fold tail loop.
  bool foldTailByMasking() const {
    // TODO: check if it is possible to check for None style independent of
    // IVUpdateMayOverflow flag in getTailFoldingStyle.
    return getTailFoldingStyle() != TailFoldingStyle::None;
  }

  /// Return maximum safe number of elements to be processed per vector
  /// iteration, which do not prevent store-load forwarding and are safe with
  /// regard to the memory dependencies. Required for EVL-based VPlans to
  /// correctly calculate AVL (application vector length) as min(remaining AVL,
  /// MaxSafeElements).
  /// TODO: need to consider adjusting cost model to use this value as a
  /// vectorization factor for EVL-based vectorization.
  std::optional<unsigned> getMaxSafeElements() const { return MaxSafeElements; }

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

  /// Returns true if the Phi is part of an inloop reduction.
  bool isInLoopReduction(PHINode *Phi) const {
    return InLoopReductions.contains(Phi);
  }

  /// Returns true if the predicated reduction select should be used to set the
  /// incoming value for the reduction phi.
  bool usePredicatedReductionSelect() const {
    // Force to use predicated reduction select since the EVL of the
    // second-to-last iteration might not be VF*UF.
    if (foldTailWithEVL())
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

  bool hasPredStores() const { return NumPredStores > 0; }

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

  /// Return the value of vscale used for tuning the cost model.
  std::optional<unsigned> getVScaleForTuning() const { return VScaleForTuning; }

private:
  unsigned NumPredStores = 0;

  /// Used to store the value of vscale used for tuning the cost model. It is
  /// initialized during object construction.
  std::optional<unsigned> VScaleForTuning;

  /// Initializes the value of vscale used for tuning the cost model. If
  /// vscale_range.min == vscale_range.max then return vscale_range.max, else
  /// return the value returned by the corresponding TTI method.
  void initializeVScaleForTuning() {
    const Function *Fn = TheLoop->getHeader()->getParent();
    if (Fn->hasFnAttribute(Attribute::VScaleRange)) {
      auto Attr = Fn->getFnAttribute(Attribute::VScaleRange);
      auto Min = Attr.getVScaleRangeMin();
      auto Max = Attr.getVScaleRangeMax();
      if (Max && Min == Max) {
        VScaleForTuning = Max;
        return;
      }
    }

    VScaleForTuning = TTI.getVScaleForTuning();
  }

  /// \return An upper bound for the vectorization factors for both
  /// fixed and scalable vectorization, where the minimum-known number of
  /// elements is a power-of-2 larger than zero. If scalable vectorization is
  /// disabled or unsupported, then the scalable part will be equal to
  /// ElementCount::getScalable(0).
  FixedScalableVFPair computeFeasibleMaxVF(unsigned MaxTripCount,
                                           ElementCount UserVF,
                                           bool FoldTailByMasking);

  /// If \p VF > MaxTripcount, clamps it to the next lower VF that is <=
  /// MaxTripCount.
  ElementCount clampVFByMaxTripCount(ElementCount VF, unsigned MaxTripCount,
                                     bool FoldTailByMasking) const;

  /// \return the maximized element count based on the targets vector
  /// registers and the loop trip-count, but limited to a maximum safe VF.
  /// This is a helper function of computeFeasibleMaxVF.
  ElementCount getMaximizedVFForTarget(unsigned MaxTripCount,
                                       unsigned SmallestType,
                                       unsigned WidestType,
                                       ElementCount MaxSafeVF,
                                       bool FoldTailByMasking);

  /// Checks if scalable vectorization is supported and enabled. Caches the
  /// result to avoid repeated debug dumps for repeated queries.
  bool isScalableVectorizationAllowed();

  /// \return the maximum legal scalable VF, based on the safe max number
  /// of elements.
  ElementCount getMaxLegalScalableVF(unsigned MaxSafeElements);

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

  /// Returns true if an artificially high cost for emulated masked memrefs
  /// should be used.
  bool useEmulatedMaskMemRefHack(Instruction *I, ElementCount VF);

  /// Map of scalar integer values to the smallest bitwidth they can be legally
  /// represented as. The vector equivalents of these values should be truncated
  /// to this type.
  MapVector<Instruction *, uint64_t> MinBWs;

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
  ScalarEpilogueLowering ScalarEpilogueStatus = CM_ScalarEpilogueAllowed;

  /// Control finally chosen tail folding style. The first element is used if
  /// the IV update may overflow, the second element - if it does not.
  std::optional<std::pair<TailFoldingStyle, TailFoldingStyle>>
      ChosenTailFoldingStyle;

  /// true if scalable vectorization is supported and enabled.
  std::optional<bool> IsScalableVectorizationAllowed;

  /// Maximum safe number of elements to be processed per vector iteration,
  /// which do not prevent store-load forwarding and are safe with regard to the
  /// memory dependencies. Required for EVL-based veectorization, where this
  /// value is used as the upper bound of the safe AVL.
  std::optional<unsigned> MaxSafeElements;

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

  /// PHINodes of the reductions that should be expanded in-loop.
  SmallPtrSet<PHINode *, 4> InLoopReductions;

  /// A Map of inloop reduction operations and their immediate chain operand.
  /// FIXME: This can be removed once reductions can be costed correctly in
  /// VPlan. This was added to allow quick lookup of the inloop operations.
  DenseMap<Instruction *, Instruction *> InLoopReductionImmediateChains;

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

  /// Demanded bits analysis.
  DemandedBits *DB;

  /// Assumption cache.
  AssumptionCache *AC;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;

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

  /// All element types found in the loop.
  SmallPtrSet<Type *, 16> ElementTypesInLoop;

  /// The kind of cost that we are calculating
  TTI::TargetCostKind CostKind;

  /// Whether this loop should be optimized for size based on function attribute
  /// or profile information.
  bool OptForSize;

  /// The highest VF possible for this loop, without using MaxBandwidth.
  FixedScalableVFPair MaxPermissibleVFWithoutMaxBW;
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
                    const DataLayout &DL, TTI::TargetCostKind CostKind)
      : DT(DT), LI(LI), TTI(TTI), SCEVExp(*PSE.getSE(), DL, "scev.check"),
        MemCheckExp(*PSE.getSE(), DL, "scev.check"), PSE(PSE),
        CostKind(CostKind) {}

  /// Generate runtime checks in SCEVCheckBlock and MemCheckBlock, so we can
  /// accurately estimate the cost of the runtime checks. The blocks are
  /// un-linked from the IR and are added back during vector code generation. If
  /// there is no vector code generation, the check blocks are removed
  /// completely.
  void create(Loop *L, const LoopAccessInfo &LAI,
              const SCEVPredicate &UnionPred, ElementCount VF, unsigned IC) {

    // Hard cutoff to limit compile-time increase in case a very large number of
    // runtime checks needs to be generated.
    // TODO: Skip cutoff if the loop is guaranteed to execute, e.g. due to
    // profile info.
    CostTooHigh =
        LAI.getNumRuntimePointerChecks() > VectorizeMemoryCheckThreshold;
    if (CostTooHigh)
      return;

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
  std::pair<Value *, BasicBlock *> getSCEVChecks() {
    using namespace llvm::PatternMatch;
    if (!SCEVCheckCond || match(SCEVCheckCond, m_ZeroInt()))
      return {nullptr, nullptr};

    return {SCEVCheckCond, SCEVCheckBlock};
  }

  /// Retrieves the MemCheckCond and MemCheckBlock that were generated as IR
  /// outside VPlan.
  std::pair<Value *, BasicBlock *> getMemRuntimeChecks() {
    using namespace llvm::PatternMatch;
    if (MemRuntimeCheckCond && match(MemRuntimeCheckCond, m_ZeroInt()))
      return {nullptr, nullptr};
    return {MemRuntimeCheckCond, MemCheckBlock};
  }

  /// Return true if any runtime checks have been added
  bool hasChecks() const {
    using namespace llvm::PatternMatch;
    return (SCEVCheckCond && !match(SCEVCheckCond, m_ZeroInt())) ||
           MemRuntimeCheckCond;
  }
};
} // namespace

static bool useActiveLaneMask(TailFoldingStyle Style) {
  return Style == TailFoldingStyle::Data ||
         Style == TailFoldingStyle::DataAndControlFlow ||
         Style == TailFoldingStyle::DataAndControlFlowWithoutRuntimeCheck;
}

static bool useActiveLaneMaskForControlFlow(TailFoldingStyle Style) {
  return Style == TailFoldingStyle::DataAndControlFlow ||
         Style == TailFoldingStyle::DataAndControlFlowWithoutRuntimeCheck;
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

/// Compute the transformed value of Index at offset StartValue using step
/// StepValue.
/// For integer induction, returns StartValue + Index * StepValue.
/// For pointer induction, returns StartValue[Index * StepValue].
/// FIXME: The newly created binary instructions should contain nsw/nuw
/// flags, which can be found from the original scalar operations.
static Value *
emitTransformedIndex(IRBuilderBase &B, Value *Index, Value *StartValue,
                     Value *Step,
                     InductionDescriptor::InductionKind InductionKind,
                     const BinaryOperator *InductionBinOp) {
  using namespace llvm::PatternMatch;
  Type *StepTy = Step->getType();
  Value *CastedIndex = StepTy->isIntegerTy()
                           ? B.CreateSExtOrTrunc(Index, StepTy)
                           : B.CreateCast(Instruction::SIToFP, Index, StepTy);
  if (CastedIndex != Index) {
    CastedIndex->setName(CastedIndex->getName() + ".cast");
    Index = CastedIndex;
  }

  // Note: the IR at this point is broken. We cannot use SE to create any new
  // SCEV and then expand it, hoping that SCEV's simplification will give us
  // a more optimal code. Unfortunately, attempt of doing so on invalid IR may
  // lead to various SCEV crashes. So all we can do is to use builder and rely
  // on InstCombine for future simplifications. Here we handle some trivial
  // cases only.
  auto CreateAdd = [&B](Value *X, Value *Y) {
    assert(X->getType() == Y->getType() && "Types don't match!");
    if (match(X, m_ZeroInt()))
      return Y;
    if (match(Y, m_ZeroInt()))
      return X;
    return B.CreateAdd(X, Y);
  };

  // We allow X to be a vector type, in which case Y will potentially be
  // splatted into a vector with the same element count.
  auto CreateMul = [&B](Value *X, Value *Y) {
    assert(X->getType()->getScalarType() == Y->getType() &&
           "Types don't match!");
    if (match(X, m_One()))
      return Y;
    if (match(Y, m_One()))
      return X;
    VectorType *XVTy = dyn_cast<VectorType>(X->getType());
    if (XVTy && !isa<VectorType>(Y->getType()))
      Y = B.CreateVectorSplat(XVTy->getElementCount(), Y);
    return B.CreateMul(X, Y);
  };

  switch (InductionKind) {
  case InductionDescriptor::IK_IntInduction: {
    assert(!isa<VectorType>(Index->getType()) &&
           "Vector indices not supported for integer inductions yet");
    assert(Index->getType() == StartValue->getType() &&
           "Index type does not match StartValue type");
    if (isa<ConstantInt>(Step) && cast<ConstantInt>(Step)->isMinusOne())
      return B.CreateSub(StartValue, Index);
    auto *Offset = CreateMul(Index, Step);
    return CreateAdd(StartValue, Offset);
  }
  case InductionDescriptor::IK_PtrInduction:
    return B.CreatePtrAdd(StartValue, CreateMul(Index, Step));
  case InductionDescriptor::IK_FpInduction: {
    assert(!isa<VectorType>(Index->getType()) &&
           "Vector indices not supported for FP inductions yet");
    assert(Step->getType()->isFloatingPointTy() && "Expected FP Step value");
    assert(InductionBinOp &&
           (InductionBinOp->getOpcode() == Instruction::FAdd ||
            InductionBinOp->getOpcode() == Instruction::FSub) &&
           "Original bin op should be defined for FP induction");

    Value *MulExp = B.CreateFMul(Step, Index);
    return B.CreateBinOp(InductionBinOp->getOpcode(), StartValue, MulExp,
                         "induction");
  }
  case InductionDescriptor::IK_NoInduction:
    return nullptr;
  }
  llvm_unreachable("invalid enum");
}

static std::optional<unsigned> getMaxVScale(const Function &F,
                                            const TargetTransformInfo &TTI) {
  if (std::optional<unsigned> MaxVScale = TTI.getMaxVScale())
    return MaxVScale;

  if (F.hasFnAttribute(Attribute::VScaleRange))
    return F.getFnAttribute(Attribute::VScaleRange).getVScaleRangeMax();

  return std::nullopt;
}

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

void EpilogueVectorizerMainLoop::introduceCheckBlockInVPlan(
    BasicBlock *CheckIRBB) {
  // Note: The block with the minimum trip-count check is already connected
  // during earlier VPlan construction.
  VPBlockBase *ScalarPH = Plan.getScalarPreheader();
  VPBlockBase *PreVectorPH = VectorPHVPBB->getSinglePredecessor();
  assert(PreVectorPH->getNumSuccessors() == 2 && "Expected 2 successors");
  assert(PreVectorPH->getSuccessors()[0] == ScalarPH && "Unexpected successor");
  VPIRBasicBlock *CheckVPIRBB = Plan.createVPIRBasicBlock(CheckIRBB);
  VPBlockUtils::insertOnEdge(PreVectorPH, VectorPHVPBB, CheckVPIRBB);
  PreVectorPH = CheckVPIRBB;
  VPBlockUtils::connectBlocks(PreVectorPH, ScalarPH);
  PreVectorPH->swapSuccessors();

  // We just connected a new block to the scalar preheader. Update all
  // VPPhis by adding an incoming value for it, replicating the last value.
  unsigned NumPredecessors = ScalarPH->getNumPredecessors();
  for (VPRecipeBase &R : cast<VPBasicBlock>(ScalarPH)->phis()) {
    assert(isa<VPPhi>(&R) && "Phi expected to be VPPhi");
    assert(cast<VPPhi>(&R)->getNumIncoming() == NumPredecessors - 1 &&
           "must have incoming values for all operands");
    R.addOperand(R.getOperand(NumPredecessors - 2));
  }
}

Value *
EpilogueVectorizerMainLoop::createIterationCountCheck(ElementCount VF,
                                                      unsigned UF) const {
  // Generate code to check if the loop's trip count is less than VF * UF, or
  // equal to it in case a scalar epilogue is required; this implies that the
  // vector trip count is zero. This check also covers the case where adding one
  // to the backedge-taken count overflowed leading to an incorrect trip count
  // of zero. In this case we will also jump to the scalar loop.
  auto P = Cost->requiresScalarEpilogue(VF.isVector()) ? ICmpInst::ICMP_ULE
                                                       : ICmpInst::ICMP_ULT;

  // Reuse existing vector loop preheader for TC checks.
  // Note that new preheader block is generated for vector loop.
  BasicBlock *const TCCheckBlock = LoopVectorPreHeader;
  IRBuilder<InstSimplifyFolder> Builder(
      TCCheckBlock->getContext(),
      InstSimplifyFolder(TCCheckBlock->getDataLayout()));
  Builder.SetInsertPoint(TCCheckBlock->getTerminator());

  // If tail is to be folded, vector loop takes care of all iterations.
  Value *Count = getTripCount();
  Type *CountTy = Count->getType();
  Value *CheckMinIters = Builder.getFalse();
  auto CreateStep = [&]() -> Value * {
    // Create step with max(MinProTripCount, UF * VF).
    if (UF * VF.getKnownMinValue() >= MinProfitableTripCount.getKnownMinValue())
      return createStepForVF(Builder, CountTy, VF, UF);

    Value *MinProfTC =
        Builder.CreateElementCount(CountTy, MinProfitableTripCount);
    if (!VF.isScalable())
      return MinProfTC;
    return Builder.CreateBinaryIntrinsic(
        Intrinsic::umax, MinProfTC, createStepForVF(Builder, CountTy, VF, UF));
  };

  TailFoldingStyle Style = Cost->getTailFoldingStyle();
  if (Style == TailFoldingStyle::None) {
    Value *Step = CreateStep();
    ScalarEvolution &SE = *PSE.getSE();
    // TODO: Emit unconditional branch to vector preheader instead of
    // conditional branch with known condition.
    const SCEV *TripCountSCEV = SE.applyLoopGuards(SE.getSCEV(Count), OrigLoop);
    // Check if the trip count is < the step.
    if (SE.isKnownPredicate(P, TripCountSCEV, SE.getSCEV(Step))) {
      // TODO: Ensure step is at most the trip count when determining max VF and
      // UF, w/o tail folding.
      CheckMinIters = Builder.getTrue();
    } else if (!SE.isKnownPredicate(CmpInst::getInversePredicate(P),
                                    TripCountSCEV, SE.getSCEV(Step))) {
      // Generate the minimum iteration check only if we cannot prove the
      // check is known to be true, or known to be false.
      CheckMinIters = Builder.CreateICmp(P, Count, Step, "min.iters.check");
    } // else step known to be < trip count, use CheckMinIters preset to false.
  } else if (VF.isScalable() && !TTI->isVScaleKnownToBeAPowerOfTwo() &&
             !isIndvarOverflowCheckKnownFalse(Cost, VF, UF) &&
             Style != TailFoldingStyle::DataAndControlFlowWithoutRuntimeCheck) {
    // vscale is not necessarily a power-of-2, which means we cannot guarantee
    // an overflow to zero when updating induction variables and so an
    // additional overflow check is required before entering the vector loop.

    // Get the maximum unsigned value for the type.
    Value *MaxUIntTripCount =
        ConstantInt::get(CountTy, cast<IntegerType>(CountTy)->getMask());
    Value *LHS = Builder.CreateSub(MaxUIntTripCount, Count);

    // Don't execute the vector loop if (UMax - n) < (VF * UF).
    CheckMinIters = Builder.CreateICmp(ICmpInst::ICMP_ULT, LHS, CreateStep());
  }
  return CheckMinIters;
}

/// Replace \p VPBB with a VPIRBasicBlock wrapping \p IRBB. All recipes from \p
/// VPBB are moved to the end of the newly created VPIRBasicBlock. VPBB must
/// have a single predecessor, which is rewired to the new VPIRBasicBlock. All
/// successors of VPBB, if any, are rewired to the new VPIRBasicBlock.
static VPIRBasicBlock *replaceVPBBWithIRVPBB(VPBasicBlock *VPBB,
                                             BasicBlock *IRBB) {
  VPIRBasicBlock *IRVPBB = VPBB->getPlan()->createVPIRBasicBlock(IRBB);
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
  LoopVectorPreHeader = OrigLoop->getLoopPreheader();
  assert(LoopVectorPreHeader && "Invalid loop structure");
  assert((OrigLoop->getUniqueLatchExitBlock() ||
          Cost->requiresScalarEpilogue(VF.isVector())) &&
         "loops not exiting via the latch without required epilogue?");

  // NOTE: The Plan's scalar preheader VPBB isn't replaced with a VPIRBasicBlock
  // wrapping the newly created scalar preheader here at the moment, because the
  // Plan's scalar preheader may be unreachable at this point. Instead it is
  // replaced in executePlan.
  return SplitBlock(LoopVectorPreHeader, LoopVectorPreHeader->getTerminator(),
                    DT, LI, nullptr, Twine(Prefix) + "scalar.ph");
}

/// Return the expanded step for \p ID using \p ExpandedSCEVs to look up SCEV
/// expansion results.
static Value *getExpandedStep(const InductionDescriptor &ID,
                              const SCEV2ValueTy &ExpandedSCEVs) {
  const SCEV *Step = ID.getStep();
  if (auto *C = dyn_cast<SCEVConstant>(Step))
    return C->getValue();
  if (auto *U = dyn_cast<SCEVUnknown>(Step))
    return U->getValue();
  Value *V = ExpandedSCEVs.lookup(Step);
  assert(V && "SCEV must be expanded at this point");
  return V;
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

///Perform cse of induction variable instructions.
static void cse(BasicBlock *BB) {
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

  InstructionCost ScalarCallCost =
      TTI.getCallInstrCost(CI->getCalledFunction(), RetTy, Tys, CostKind);

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
                                    InstructionCost::getInvalid(), TLI);
  return TTI.getIntrinsicInstrCost(CostAttrs, CostKind);
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
  cse(HeaderBB);

  // Set/update profile weights for the vector and remainder loops as original
  // loop iterations are now distributed among them. Note that original loop
  // becomes the scalar remainder loop after vectorization.
  //
  // For cases like foldTailByMasking() and requiresScalarEpiloque() we may
  // end up getting slightly roughened result but that should be OK since
  // profile is not inherently precise anyway. Note also possible bypass of
  // vector code caused by legality checks is ignored, assigning all the weight
  // to the vector loop, optimistically.
  //
  // For scalable vectorization we can't know at compile time how many
  // iterations of the loop are handled in one vector iteration, so instead
  // use the value of vscale used for tuning.
  Loop *VectorLoop = LI->getLoopFor(HeaderBB);
  unsigned EstimatedVFxUF =
      estimateElementCount(VF * UF, Cost->getVScaleForTuning());
  setProfileInfoAfterUnrolling(OrigLoop, VectorLoop, OrigLoop, EstimatedVFxUF);
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

bool LoopVectorizationCostModel::isScalarWithPredication(
    Instruction *I, ElementCount VF) const {
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
    return isa<LoadInst>(I) ? !(isLegalMaskedLoad(Ty, Ptr, Alignment, AS) ||
                                TTI.isLegalMaskedGather(VTy, Alignment))
                            : !(isLegalMaskedStore(Ty, Ptr, Alignment, AS) ||
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

// TODO: Fold into LoopVectorizationLegality::isMaskRequired.
bool LoopVectorizationCostModel::isPredicatedInst(Instruction *I) const {
  // TODO: We can use the loop-preheader as context point here and get
  // context sensitive reasoning for isSafeToSpeculativelyExecute.
  if (isSafeToSpeculativelyExecute(I) ||
      (isa<LoadInst, StoreInst, CallInst>(I) && !Legal->isMaskRequired(I)) ||
      isa<BranchInst, SwitchInst, PHINode, AllocaInst>(I))
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
    assert(Legal->isMaskRequired(I) &&
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
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::URem:
    // If the divisor is loop-invariant no predication is needed.
    return !Legal->isInvariant(I->getOperand(1));
  }
}

std::pair<InstructionCost, InstructionCost>
LoopVectorizationCostModel::getDivRemSpeculationCost(Instruction *I,
                                                    ElementCount VF) const {
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
    ScalarizationCost +=
        VF.getFixedValue() * TTI.getCFInstrCost(Instruction::PHI, CostKind);

    // The cost of the non-predicated instruction.
    ScalarizationCost +=
        VF.getFixedValue() *
        TTI.getArithmeticInstrCost(I->getOpcode(), I->getType(), CostKind);

    // The cost of insertelement and extractelement instructions needed for
    // scalarization.
    ScalarizationCost += getScalarizationOverhead(I, VF);

    // Scale the cost by the probability of executing the predicated blocks.
    // This assumes the predicated block for each vector lane is equally
    // likely.
    ScalarizationCost = ScalarizationCost / getPredBlockCostDivisor(CostKind);
  }
  InstructionCost SafeDivisorCost = 0;

  auto *VecTy = toVectorTy(I->getType(), VF);

  // The cost of the select guard to ensure all lanes are well defined
  // after we speculate above any internal control flow.
  SafeDivisorCost +=
      TTI.getCmpSelInstrCost(Instruction::Select, VecTy,
                             toVectorTy(Type::getInt1Ty(I->getContext()), VF),
                             CmpInst::BAD_ICMP_PREDICATE, CostKind);

  // Certain instructions can be cheaper to vectorize if they have a constant
  // second vector operand. One example of this are shifts on x86.
  Value *Op2 = I->getOperand(1);
  auto Op2Info = TTI.getOperandInfo(Op2);
  if (Op2Info.Kind == TargetTransformInfo::OK_AnyValue &&
      Legal->isInvariant(Op2))
    Op2Info.Kind = TargetTransformInfo::OK_UniformValue;

  SmallVector<const Value *, 4> Operands(I->operand_values());
  SafeDivisorCost += TTI.getArithmeticInstrCost(
    I->getOpcode(), VecTy, CostKind,
    {TargetTransformInfo::OK_AnyValue, TargetTransformInfo::OP_None},
    Op2Info, Operands, I);
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
  for (unsigned Idx = 0; Idx < InterleaveFactor; Idx++) {
    Instruction *Member = Group->getMember(Idx);
    if (!Member)
      continue;
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
      blockNeedsPredicationForAnyReason(I->getParent()) &&
      Legal->isMaskRequired(I);
  bool LoadAccessWithGapsRequiresEpilogMasking =
      isa<LoadInst>(I) && Group->requiresScalarEpilogue() &&
      !isScalarEpilogueAllowed();
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

bool LoopVectorizationCostModel::runtimeChecksRequired() {
  LLVM_DEBUG(dbgs() << "LV: Performing code size checks.\n");

  if (Legal->getRuntimePointerChecking()->Need) {
    reportVectorizationFailure("Runtime ptr check is required with -Os/-Oz",
        "runtime pointer checks needed. Enable vectorization of this "
        "loop with '#pragma clang loop vectorize(enable)' when "
        "compiling with -Os/-Oz",
        "CantVersionLoopWithOptForSize", ORE, TheLoop);
    return true;
  }

  if (!PSE.getPredicate().isAlwaysTrue()) {
    reportVectorizationFailure("Runtime SCEV check is required with -Os/-Oz",
        "runtime SCEV checks needed. Enable vectorization of this "
        "loop with '#pragma clang loop vectorize(enable)' when "
        "compiling with -Os/-Oz",
        "CantVersionLoopWithOptForSize", ORE, TheLoop);
    return true;
  }

  // FIXME: Avoid specializing for stride==1 instead of bailing out.
  if (!Legal->getLAI()->getSymbolicStrides().empty()) {
    reportVectorizationFailure("Runtime stride check for small trip count",
        "runtime stride == 1 checks needed. Enable vectorization of "
        "this loop without such check by compiling with -Os/-Oz",
        "CantVersionLoopWithOptForSize", ORE, TheLoop);
    return true;
  }

  return false;
}

bool LoopVectorizationCostModel::isScalableVectorizationAllowed() {
  if (IsScalableVectorizationAllowed)
    return *IsScalableVectorizationAllowed;

  IsScalableVectorizationAllowed = false;
  if (!TTI.supportsScalableVectors() && !ForceTargetSupportsScalableVectors)
    return false;

  if (Hints->isScalableVectorizationDisabled()) {
    reportVectorizationInfo("Scalable vectorization is explicitly disabled",
                            "ScalableVectorizationDisabled", ORE, TheLoop);
    return false;
  }

  LLVM_DEBUG(dbgs() << "LV: Scalable vectorization is available\n");

  auto MaxScalableVF = ElementCount::getScalable(
      std::numeric_limits<ElementCount::ScalarTy>::max());

  // Test that the loop-vectorizer can legalize all operations for this MaxVF.
  // FIXME: While for scalable vectors this is currently sufficient, this should
  // be replaced by a more detailed mechanism that filters out specific VFs,
  // instead of invalidating vectorization for a whole set of VFs based on the
  // MaxVF.

  // Disable scalable vectorization if the loop contains unsupported reductions.
  if (!canVectorizeReductions(MaxScalableVF)) {
    reportVectorizationInfo(
        "Scalable vectorization not supported for the reduction "
        "operations found in this loop.",
        "ScalableVFUnfeasible", ORE, TheLoop);
    return false;
  }

  // Disable scalable vectorization if the loop contains any instructions
  // with element types not supported for scalable vectors.
  if (any_of(ElementTypesInLoop, [&](Type *Ty) {
        return !Ty->isVoidTy() &&
               !this->TTI.isElementTypeLegalForScalableVector(Ty);
      })) {
    reportVectorizationInfo("Scalable vectorization is not supported "
                            "for all element types found in this loop.",
                            "ScalableVFUnfeasible", ORE, TheLoop);
    return false;
  }

  if (!Legal->isSafeForAnyVectorWidth() && !getMaxVScale(*TheFunction, TTI)) {
    reportVectorizationInfo("The target does not provide maximum vscale value "
                            "for safe distance analysis.",
                            "ScalableVFUnfeasible", ORE, TheLoop);
    return false;
  }

  IsScalableVectorizationAllowed = true;
  return true;
}

ElementCount
LoopVectorizationCostModel::getMaxLegalScalableVF(unsigned MaxSafeElements) {
  if (!isScalableVectorizationAllowed())
    return ElementCount::getScalable(0);

  auto MaxScalableVF = ElementCount::getScalable(
      std::numeric_limits<ElementCount::ScalarTy>::max());
  if (Legal->isSafeForAnyVectorWidth())
    return MaxScalableVF;

  std::optional<unsigned> MaxVScale = getMaxVScale(*TheFunction, TTI);
  // Limit MaxScalableVF by the maximum safe dependence distance.
  MaxScalableVF = ElementCount::getScalable(MaxSafeElements / *MaxVScale);

  if (!MaxScalableVF)
    reportVectorizationInfo(
        "Max legal vector width too small, scalable vectorization "
        "unfeasible.",
        "ScalableVFUnfeasible", ORE, TheLoop);

  return MaxScalableVF;
}

FixedScalableVFPair LoopVectorizationCostModel::computeFeasibleMaxVF(
    unsigned MaxTripCount, ElementCount UserVF, bool FoldTailByMasking) {
  MinBWs = computeMinimumValueSizes(TheLoop->getBlocks(), *DB, &TTI);
  unsigned SmallestType, WidestType;
  std::tie(SmallestType, WidestType) = getSmallestAndWidestTypes();

  // Get the maximum safe dependence distance in bits computed by LAA.
  // It is computed by MaxVF * sizeOf(type) * 8, where type is taken from
  // the memory accesses that is most restrictive (involved in the smallest
  // dependence distance).
  unsigned MaxSafeElementsPowerOf2 =
      bit_floor(Legal->getMaxSafeVectorWidthInBits() / WidestType);
  if (!Legal->isSafeForAnyStoreLoadForwardDistances()) {
    unsigned SLDist = Legal->getMaxStoreLoadForwardSafeDistanceInBits();
    MaxSafeElementsPowerOf2 =
        std::min(MaxSafeElementsPowerOf2, SLDist / WidestType);
  }
  auto MaxSafeFixedVF = ElementCount::getFixed(MaxSafeElementsPowerOf2);
  auto MaxSafeScalableVF = getMaxLegalScalableVF(MaxSafeElementsPowerOf2);

  if (!Legal->isSafeForAnyVectorWidth())
    this->MaxSafeElements = MaxSafeElementsPowerOf2;

  LLVM_DEBUG(dbgs() << "LV: The max safe fixed VF is: " << MaxSafeFixedVF
                    << ".\n");
  LLVM_DEBUG(dbgs() << "LV: The max safe scalable VF is: " << MaxSafeScalableVF
                    << ".\n");

  // First analyze the UserVF, fall back if the UserVF should be ignored.
  if (UserVF) {
    auto MaxSafeUserVF =
        UserVF.isScalable() ? MaxSafeScalableVF : MaxSafeFixedVF;

    if (ElementCount::isKnownLE(UserVF, MaxSafeUserVF)) {
      // If `VF=vscale x N` is safe, then so is `VF=N`
      if (UserVF.isScalable())
        return FixedScalableVFPair(
            ElementCount::getFixed(UserVF.getKnownMinValue()), UserVF);

      return UserVF;
    }

    assert(ElementCount::isKnownGT(UserVF, MaxSafeUserVF));

    // Only clamp if the UserVF is not scalable. If the UserVF is scalable, it
    // is better to ignore the hint and let the compiler choose a suitable VF.
    if (!UserVF.isScalable()) {
      LLVM_DEBUG(dbgs() << "LV: User VF=" << UserVF
                        << " is unsafe, clamping to max safe VF="
                        << MaxSafeFixedVF << ".\n");
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(DEBUG_TYPE, "VectorizationFactor",
                                          TheLoop->getStartLoc(),
                                          TheLoop->getHeader())
               << "User-specified vectorization factor "
               << ore::NV("UserVectorizationFactor", UserVF)
               << " is unsafe, clamping to maximum safe vectorization factor "
               << ore::NV("VectorizationFactor", MaxSafeFixedVF);
      });
      return MaxSafeFixedVF;
    }

    if (!TTI.supportsScalableVectors() && !ForceTargetSupportsScalableVectors) {
      LLVM_DEBUG(dbgs() << "LV: User VF=" << UserVF
                        << " is ignored because scalable vectors are not "
                           "available.\n");
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(DEBUG_TYPE, "VectorizationFactor",
                                          TheLoop->getStartLoc(),
                                          TheLoop->getHeader())
               << "User-specified vectorization factor "
               << ore::NV("UserVectorizationFactor", UserVF)
               << " is ignored because the target does not support scalable "
                  "vectors. The compiler will pick a more suitable value.";
      });
    } else {
      LLVM_DEBUG(dbgs() << "LV: User VF=" << UserVF
                        << " is unsafe. Ignoring scalable UserVF.\n");
      ORE->emit([&]() {
        return OptimizationRemarkAnalysis(DEBUG_TYPE, "VectorizationFactor",
                                          TheLoop->getStartLoc(),
                                          TheLoop->getHeader())
               << "User-specified vectorization factor "
               << ore::NV("UserVectorizationFactor", UserVF)
               << " is unsafe. Ignoring the hint to let the compiler pick a "
                  "more suitable value.";
      });
    }
  }

  LLVM_DEBUG(dbgs() << "LV: The Smallest and Widest types: " << SmallestType
                    << " / " << WidestType << " bits.\n");

  FixedScalableVFPair Result(ElementCount::getFixed(1),
                             ElementCount::getScalable(0));
  if (auto MaxVF =
          getMaximizedVFForTarget(MaxTripCount, SmallestType, WidestType,
                                  MaxSafeFixedVF, FoldTailByMasking))
    Result.FixedVF = MaxVF;

  if (auto MaxVF =
          getMaximizedVFForTarget(MaxTripCount, SmallestType, WidestType,
                                  MaxSafeScalableVF, FoldTailByMasking))
    if (MaxVF.isScalable()) {
      Result.ScalableVF = MaxVF;
      LLVM_DEBUG(dbgs() << "LV: Found feasible scalable VF = " << MaxVF
                        << "\n");
    }

  return Result;
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

  switch (ScalarEpilogueStatus) {
  case CM_ScalarEpilogueAllowed:
    return computeFeasibleMaxVF(MaxTC, UserVF, false);
  case CM_ScalarEpilogueNotAllowedUsePredicate:
    [[fallthrough]];
  case CM_ScalarEpilogueNotNeededUsePredicate:
    LLVM_DEBUG(
        dbgs() << "LV: vector predicate hint/switch found.\n"
               << "LV: Not allowing scalar epilogue, creating predicated "
               << "vector loop.\n");
    break;
  case CM_ScalarEpilogueNotAllowedLowTripLoop:
    // fallthrough as a special case of OptForSize
  case CM_ScalarEpilogueNotAllowedOptSize:
    if (ScalarEpilogueStatus == CM_ScalarEpilogueNotAllowedOptSize)
      LLVM_DEBUG(
          dbgs() << "LV: Not allowing scalar epilogue due to -Os/-Oz.\n");
    else
      LLVM_DEBUG(dbgs() << "LV: Not allowing scalar epilogue due to low trip "
                        << "count.\n");

    // Bail if runtime checks are required, which are not good when optimising
    // for size.
    if (runtimeChecksRequired())
      return FixedScalableVFPair::getNone();

    break;
  }

  // Now try the tail folding

  // Invalidate interleave groups that require an epilogue if we can't mask
  // the interleave-group.
  if (!useMaskedInterleavedAccesses(TTI)) {
    assert(WideningDecisions.empty() && Uniforms.empty() && Scalars.empty() &&
           "No decisions should have been taken at this point");
    // Note: There is no need to invalidate any cost modeling decisions here, as
    // none were taken so far.
    InterleaveInfo.invalidateGroupsRequiringScalarEpilogue();
  }

  FixedScalableVFPair MaxFactors = computeFeasibleMaxVF(MaxTC, UserVF, true);

  // Avoid tail folding if the trip count is known to be a multiple of any VF
  // we choose.
  std::optional<unsigned> MaxPowerOf2RuntimeVF =
      MaxFactors.FixedVF.getFixedValue();
  if (MaxFactors.ScalableVF) {
    std::optional<unsigned> MaxVScale = getMaxVScale(*TheFunction, TTI);
    if (MaxVScale && TTI.isVScaleKnownToBeAPowerOfTwo()) {
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
      if (ScalarEpilogueStatus == CM_ScalarEpilogueNotAllowedLowTripLoop &&
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
  setTailFoldingStyles(ContainsScalableVF, UserIC);
  if (foldTailByMasking()) {
    if (getTailFoldingStyle() == TailFoldingStyle::DataWithEVL) {
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
  // masking, fallback to a vectorization with a scalar epilogue.
  if (ScalarEpilogueStatus == CM_ScalarEpilogueNotNeededUsePredicate) {
    LLVM_DEBUG(dbgs() << "LV: Cannot fold tail by masking: vectorize with a "
                         "scalar epilogue instead.\n");
    ScalarEpilogueStatus = CM_ScalarEpilogueAllowed;
    return MaxFactors;
  }

  if (ScalarEpilogueStatus == CM_ScalarEpilogueNotAllowedUsePredicate) {
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

bool LoopVectorizationCostModel::shouldCalculateRegPressureForVF(
    ElementCount VF) {
  if (!useMaxBandwidth(VF.isScalable()
                           ? TargetTransformInfo::RGK_ScalableVector
                           : TargetTransformInfo::RGK_FixedWidthVector))
    return false;
  // Only calculate register pressure for VFs enabled by MaxBandwidth.
  return ElementCount::isKnownGT(
      VF, VF.isScalable() ? MaxPermissibleVFWithoutMaxBW.ScalableVF
                          : MaxPermissibleVFWithoutMaxBW.FixedVF);
}

bool LoopVectorizationCostModel::useMaxBandwidth(
    TargetTransformInfo::RegisterKind RegKind) {
  return MaximizeBandwidth || (MaximizeBandwidth.getNumOccurrences() == 0 &&
                               (TTI.shouldMaximizeVectorBandwidth(RegKind) ||
                                (UseWiderVFIfCallVariantsPresent &&
                                 Legal->hasVectorCallVariants())));
}

ElementCount LoopVectorizationCostModel::clampVFByMaxTripCount(
    ElementCount VF, unsigned MaxTripCount, bool FoldTailByMasking) const {
  unsigned EstimatedVF = VF.getKnownMinValue();
  if (VF.isScalable() && TheFunction->hasFnAttribute(Attribute::VScaleRange)) {
    auto Attr = TheFunction->getFnAttribute(Attribute::VScaleRange);
    auto Min = Attr.getVScaleRangeMin();
    EstimatedVF *= Min;
  }

  // When a scalar epilogue is required, at least one iteration of the scalar
  // loop has to execute. Adjust MaxTripCount accordingly to avoid picking a
  // max VF that results in a dead vector loop.
  if (MaxTripCount > 0 && requiresScalarEpilogue(true))
    MaxTripCount -= 1;

  if (MaxTripCount && MaxTripCount <= EstimatedVF &&
      (!FoldTailByMasking || isPowerOf2_32(MaxTripCount))) {
    // If upper bound loop trip count (TC) is known at compile time there is no
    // point in choosing VF greater than TC (as done in the loop below). Select
    // maximum power of two which doesn't exceed TC. If VF is
    // scalable, we only fall back on a fixed VF when the TC is less than or
    // equal to the known number of lanes.
    auto ClampedUpperTripCount = llvm::bit_floor(MaxTripCount);
    LLVM_DEBUG(dbgs() << "LV: Clamping the MaxVF to maximum power of two not "
                         "exceeding the constant trip count: "
                      << ClampedUpperTripCount << "\n");
    return ElementCount::get(ClampedUpperTripCount,
                             FoldTailByMasking ? VF.isScalable() : false);
  }
  return VF;
}

ElementCount LoopVectorizationCostModel::getMaximizedVFForTarget(
    unsigned MaxTripCount, unsigned SmallestType, unsigned WidestType,
    ElementCount MaxSafeVF, bool FoldTailByMasking) {
  bool ComputeScalableMaxVF = MaxSafeVF.isScalable();
  const TypeSize WidestRegister = TTI.getRegisterBitWidth(
      ComputeScalableMaxVF ? TargetTransformInfo::RGK_ScalableVector
                           : TargetTransformInfo::RGK_FixedWidthVector);

  // Convenience function to return the minimum of two ElementCounts.
  auto MinVF = [](const ElementCount &LHS, const ElementCount &RHS) {
    assert((LHS.isScalable() == RHS.isScalable()) &&
           "Scalable flags must match");
    return ElementCount::isKnownLT(LHS, RHS) ? LHS : RHS;
  };

  // Ensure MaxVF is a power of 2; the dependence distance bound may not be.
  // Note that both WidestRegister and WidestType may not be a powers of 2.
  auto MaxVectorElementCount = ElementCount::get(
      llvm::bit_floor(WidestRegister.getKnownMinValue() / WidestType),
      ComputeScalableMaxVF);
  MaxVectorElementCount = MinVF(MaxVectorElementCount, MaxSafeVF);
  LLVM_DEBUG(dbgs() << "LV: The Widest register safe to use is: "
                    << (MaxVectorElementCount * WidestType) << " bits.\n");

  if (!MaxVectorElementCount) {
    LLVM_DEBUG(dbgs() << "LV: The target has no "
                      << (ComputeScalableMaxVF ? "scalable" : "fixed")
                      << " vector registers.\n");
    return ElementCount::getFixed(1);
  }

  ElementCount MaxVF = clampVFByMaxTripCount(MaxVectorElementCount,
                                             MaxTripCount, FoldTailByMasking);
  // If the MaxVF was already clamped, there's no point in trying to pick a
  // larger one.
  if (MaxVF != MaxVectorElementCount)
    return MaxVF;

  TargetTransformInfo::RegisterKind RegKind =
      ComputeScalableMaxVF ? TargetTransformInfo::RGK_ScalableVector
                           : TargetTransformInfo::RGK_FixedWidthVector;

  if (MaxVF.isScalable())
    MaxPermissibleVFWithoutMaxBW.ScalableVF = MaxVF;
  else
    MaxPermissibleVFWithoutMaxBW.FixedVF = MaxVF;

  if (useMaxBandwidth(RegKind)) {
    auto MaxVectorElementCountMaxBW = ElementCount::get(
        llvm::bit_floor(WidestRegister.getKnownMinValue() / SmallestType),
        ComputeScalableMaxVF);
    MaxVF = MinVF(MaxVectorElementCountMaxBW, MaxSafeVF);

    if (ElementCount MinVF =
            TTI.getMinimumVF(SmallestType, ComputeScalableMaxVF)) {
      if (ElementCount::isKnownLT(MaxVF, MinVF)) {
        LLVM_DEBUG(dbgs() << "LV: Overriding calculated MaxVF(" << MaxVF
                          << ") with target's minimum: " << MinVF << '\n');
        MaxVF = MinVF;
      }
    }

    MaxVF = clampVFByMaxTripCount(MaxVF, MaxTripCount, FoldTailByMasking);

    if (MaxVectorElementCount != MaxVF) {
      // Invalidate any widening decisions we might have made, in case the loop
      // requires prediction (decided later), but we have already made some
      // load/store widening decisions.
      invalidateCostModelingDecisions();
    }
  }
  return MaxVF;
}

bool LoopVectorizationPlanner::isMoreProfitable(const VectorizationFactor &A,
                                                const VectorizationFactor &B,
                                                const unsigned MaxTripCount,
                                                bool HasTail) const {
  InstructionCost CostA = A.Cost;
  InstructionCost CostB = B.Cost;

  // Improve estimate for the vector width if it is scalable.
  unsigned EstimatedWidthA = A.Width.getKnownMinValue();
  unsigned EstimatedWidthB = B.Width.getKnownMinValue();
  if (std::optional<unsigned> VScale = CM.getVScaleForTuning()) {
    if (A.Width.isScalable())
      EstimatedWidthA *= *VScale;
    if (B.Width.isScalable())
      EstimatedWidthB *= *VScale;
  }

  // When optimizing for size choose whichever is smallest, which will be the
  // one with the smallest cost for the whole loop. On a tie pick the larger
  // vector width, on the assumption that throughput will be greater.
  if (CM.CostKind == TTI::TCK_CodeSize)
    return CostA < CostB ||
           (CostA == CostB && EstimatedWidthA > EstimatedWidthB);

  // Assume vscale may be larger than 1 (or the value being tuned for),
  // so that scalable vectorization is slightly favorable over fixed-width
  // vectorization.
  bool PreferScalable = !TTI.preferFixedOverScalableIfEqualCost() &&
                        A.Width.isScalable() && !B.Width.isScalable();

  auto CmpFn = [PreferScalable](const InstructionCost &LHS,
                                const InstructionCost &RHS) {
    return PreferScalable ? LHS <= RHS : LHS < RHS;
  };

  // To avoid the need for FP division:
  //      (CostA / EstimatedWidthA) < (CostB / EstimatedWidthB)
  // <=>  (CostA * EstimatedWidthB) < (CostB * EstimatedWidthA)
  if (!MaxTripCount)
    return CmpFn(CostA * EstimatedWidthB, CostB * EstimatedWidthA);

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
  return CmpFn(RTCostA, RTCostB);
}

bool LoopVectorizationPlanner::isMoreProfitable(const VectorizationFactor &A,
                                                const VectorizationFactor &B,
                                                bool HasTail) const {
  const unsigned MaxTripCount = PSE.getSmallConstantMaxTripCount();
  return LoopVectorizationPlanner::isMoreProfitable(A, B, MaxTripCount,
                                                    HasTail);
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

      VPCostContext CostCtx(CM.TTI, *CM.TLI, *Plan, CM, CM.CostKind);
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
            .Case<VPHeaderPHIRecipe>(
                [](const auto *R) { return Instruction::PHI; })
            .Case<VPWidenSelectRecipe>(
                [](const auto *R) { return Instruction::Select; })
            .Case<VPWidenStoreRecipe>(
                [](const auto *R) { return Instruction::Store; })
            .Case<VPWidenLoadRecipe>(
                [](const auto *R) { return Instruction::Load; })
            .Case<VPWidenCallRecipe, VPWidenIntrinsicRecipe>(
                [](const auto *R) { return Instruction::Call; })
            .Case<VPInstruction, VPWidenRecipe, VPReplicateRecipe,
                  VPWidenCastRecipe>(
                [](const auto *R) { return R->getOpcode(); })
            .Case<VPInterleaveRecipe>([](const VPInterleaveRecipe *R) {
              return R->getStoredValues().empty() ? Instruction::Load
                                                  : Instruction::Store;
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
      switch (R.getVPDefID()) {
      case VPDef::VPDerivedIVSC:
      case VPDef::VPScalarIVStepsSC:
      case VPDef::VPReplicateSC:
      case VPDef::VPInstructionSC:
      case VPDef::VPCanonicalIVPHISC:
      case VPDef::VPVectorPointerSC:
      case VPDef::VPVectorEndPointerSC:
      case VPDef::VPExpandSCEVSC:
      case VPDef::VPEVLBasedIVPHISC:
      case VPDef::VPPredInstPHISC:
      case VPDef::VPBranchOnMaskSC:
        continue;
      case VPDef::VPReductionSC:
      case VPDef::VPActiveLaneMaskPHISC:
      case VPDef::VPWidenCallSC:
      case VPDef::VPWidenCanonicalIVSC:
      case VPDef::VPWidenCastSC:
      case VPDef::VPWidenGEPSC:
      case VPDef::VPWidenIntrinsicSC:
      case VPDef::VPWidenSC:
      case VPDef::VPWidenSelectSC:
      case VPDef::VPBlendSC:
      case VPDef::VPFirstOrderRecurrencePHISC:
      case VPDef::VPHistogramSC:
      case VPDef::VPWidenPHISC:
      case VPDef::VPWidenIntOrFpInductionSC:
      case VPDef::VPWidenPointerInductionSC:
      case VPDef::VPReductionPHISC:
      case VPDef::VPInterleaveSC:
      case VPDef::VPWidenLoadEVLSC:
      case VPDef::VPWidenLoadSC:
      case VPDef::VPWidenStoreEVLSC:
      case VPDef::VPWidenStoreSC:
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
          !isa<VPWidenStoreRecipe, VPWidenStoreEVLRecipe, VPInterleaveRecipe>(
              &R))
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

#ifndef NDEBUG
VectorizationFactor LoopVectorizationPlanner::selectVectorizationFactor() {
  InstructionCost ExpectedCost = CM.expectedCost(ElementCount::getFixed(1));
  LLVM_DEBUG(dbgs() << "LV: Scalar loop costs: " << ExpectedCost << ".\n");
  assert(ExpectedCost.isValid() && "Unexpected invalid cost for scalar loop");
  assert(
      any_of(VPlans,
             [](std::unique_ptr<VPlan> &P) { return P->hasScalarVFOnly(); }) &&
      "Expected Scalar VF to be a candidate");

  const VectorizationFactor ScalarCost(ElementCount::getFixed(1), ExpectedCost,
                                       ExpectedCost);
  VectorizationFactor ChosenFactor = ScalarCost;

  bool ForceVectorization = Hints.getForce() == LoopVectorizeHints::FK_Enabled;
  if (ForceVectorization &&
      (VPlans.size() > 1 || !VPlans[0]->hasScalarVFOnly())) {
    // Ignore scalar width, because the user explicitly wants vectorization.
    // Initialize cost to max so that VF = 2 is, at least, chosen during cost
    // evaluation.
    ChosenFactor.Cost = InstructionCost::getMax();
  }

  for (auto &P : VPlans) {
    ArrayRef<ElementCount> VFs(P->vectorFactors().begin(),
                               P->vectorFactors().end());

    SmallVector<VPRegisterUsage, 8> RUs;
    if (CM.useMaxBandwidth(TargetTransformInfo::RGK_ScalableVector) ||
        CM.useMaxBandwidth(TargetTransformInfo::RGK_FixedWidthVector))
      RUs = calculateRegisterUsageForPlan(*P, VFs, TTI, CM.ValuesToIgnore);

    for (unsigned I = 0; I < VFs.size(); I++) {
      ElementCount VF = VFs[I];
      // The cost for scalar VF=1 is already calculated, so ignore it.
      if (VF.isScalar())
        continue;

      /// If the register pressure needs to be considered for VF,
      /// don't consider the VF as valid if it exceeds the number
      /// of registers for the target.
      if (CM.shouldCalculateRegPressureForVF(VF) &&
          RUs[I].exceedsMaxNumRegs(TTI, ForceTargetNumVectorRegs))
        continue;

      InstructionCost C = CM.expectedCost(VF);

      // Add on other costs that are modelled in VPlan, but not in the legacy
      // cost model.
      VPCostContext CostCtx(CM.TTI, *CM.TLI, *P, CM, CM.CostKind);
      VPRegionBlock *VectorRegion = P->getVectorLoopRegion();
      assert(VectorRegion && "Expected to have a vector region!");
      for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(
               vp_depth_first_shallow(VectorRegion->getEntry()))) {
        for (VPRecipeBase &R : *VPBB) {
          auto *VPI = dyn_cast<VPInstruction>(&R);
          if (!VPI)
            continue;
          switch (VPI->getOpcode()) {
          // Selects are only modelled in the legacy cost model for safe
          // divisors.
          case Instruction::Select: {
            VPValue *VPV = VPI->getVPSingleValue();
            if (VPV->getNumUsers() == 1) {
              if (auto *WR = dyn_cast<VPWidenRecipe>(*VPV->user_begin())) {
                switch (WR->getOpcode()) {
                case Instruction::UDiv:
                case Instruction::SDiv:
                case Instruction::URem:
                case Instruction::SRem:
                  continue;
                default:
                  break;
                }
              }
            }
            [[fallthrough]];
          }
          case VPInstruction::ActiveLaneMask:
          case VPInstruction::ExplicitVectorLength:
            C += VPI->cost(VF, CostCtx);
            break;
          default:
            break;
          }
        }
      }

      VectorizationFactor Candidate(VF, C, ScalarCost.ScalarCost);
      unsigned Width =
          estimateElementCount(Candidate.Width, CM.getVScaleForTuning());
      LLVM_DEBUG(dbgs() << "LV: Vector loop of width " << VF
                        << " costs: " << (Candidate.Cost / Width));
      if (VF.isScalable())
        LLVM_DEBUG(dbgs() << " (assuming a minimum vscale of "
                          << CM.getVScaleForTuning().value_or(1) << ")");
      LLVM_DEBUG(dbgs() << ".\n");

      if (!ForceVectorization && !willGenerateVectors(*P, VF, TTI)) {
        LLVM_DEBUG(
            dbgs()
            << "LV: Not considering vector loop of width " << VF
            << " because it will not generate any vector instructions.\n");
        continue;
      }

      if (CM.OptForSize && !ForceVectorization && hasReplicatorRegion(*P)) {
        LLVM_DEBUG(
            dbgs()
            << "LV: Not considering vector loop of width " << VF
            << " because it would cause replicated blocks to be generated,"
            << " which isn't allowed when optimizing for size.\n");
        continue;
      }

      if (isMoreProfitable(Candidate, ChosenFactor, P->hasScalarTail()))
        ChosenFactor = Candidate;
    }
  }

  if (!EnableCondStoresVectorization && CM.hasPredStores()) {
    reportVectorizationFailure(
        "There are conditional stores.",
        "store that is conditionally executed prevents vectorization",
        "ConditionalStore", ORE, OrigLoop);
    ChosenFactor = ScalarCost;
  }

  LLVM_DEBUG(if (ForceVectorization && !ChosenFactor.Width.isScalar() &&
                 !isMoreProfitable(ChosenFactor, ScalarCost,
                                   !CM.foldTailByMasking())) dbgs()
             << "LV: Vectorization seems to be not beneficial, "
             << "but was forced by a user.\n");
  return ChosenFactor;
}
#endif

bool LoopVectorizationPlanner::isCandidateForEpilogueVectorization(
    ElementCount VF) const {
  // Cross iteration phis such as fixed-order recurrences and FMaxNum/FMinNum
  // reductions need special handling and are currently unsupported.
  if (any_of(OrigLoop->getHeader()->phis(), [&](PHINode &Phi) {
        if (!Legal->isReductionVariable(&Phi))
          return Legal->isFixedOrderRecurrence(&Phi);
        RecurKind RK = Legal->getRecurrenceDescriptor(&Phi).getRecurrenceKind();
        return RK == RecurKind::FMinNum || RK == RecurKind::FMaxNum;
      }))
    return false;

  // Phis with uses outside of the loop require special handling and are
  // currently unsupported.
  for (const auto &Entry : Legal->getInductionVars()) {
    // Look for uses of the value of the induction at the last iteration.
    Value *PostInc =
        Entry.first->getIncomingValueForBlock(OrigLoop->getLoopLatch());
    for (User *U : PostInc->users())
      if (!OrigLoop->contains(cast<Instruction>(U)))
        return false;
    // Look for uses of penultimate value of the induction.
    for (User *U : Entry.first->users())
      if (!OrigLoop->contains(cast<Instruction>(U)))
        return false;
  }

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

  // Allow the target to opt out entirely.
  if (!TTI.preferEpilogueVectorization())
    return false;

  // We also consider epilogue vectorization unprofitable for targets that don't
  // consider interleaving beneficial (eg. MVE).
  if (TTI.getMaxInterleaveFactor(VF) <= 1)
    return false;

  // TODO: PR #108190 introduced a discrepancy between fixed-width and scalable
  // VFs when deciding profitability.
  // See related "TODO: extend to support scalable VFs." in
  // selectEpilogueVectorizationFactor.
  unsigned Multiplier = VF.isFixed() ? IC : 1;
  unsigned MinVFThreshold = EpilogueVectorizationMinVF.getNumOccurrences() > 0
                                ? EpilogueVectorizationMinVF
                                : TTI.getEpilogueVectorizationMinVF();
  return estimateElementCount(VF * Multiplier, VScaleForTuning) >=
         MinVFThreshold;
}

VectorizationFactor LoopVectorizationPlanner::selectEpilogueVectorizationFactor(
    const ElementCount MainLoopVF, unsigned IC) {
  VectorizationFactor Result = VectorizationFactor::Disabled();
  if (!EnableEpilogueVectorization) {
    LLVM_DEBUG(dbgs() << "LEV: Epilogue vectorization is disabled.\n");
    return Result;
  }

  if (!CM.isScalarEpilogueAllowed()) {
    LLVM_DEBUG(dbgs() << "LEV: Unable to vectorize epilogue because no "
                         "epilogue is allowed.\n");
    return Result;
  }

  // Not really a cost consideration, but check for unsupported cases here to
  // simplify the logic.
  if (!isCandidateForEpilogueVectorization(MainLoopVF)) {
    LLVM_DEBUG(dbgs() << "LEV: Unable to vectorize epilogue because the loop "
                         "is not a supported candidate.\n");
    return Result;
  }

  if (EpilogueVectorizationForceVF > 1) {
    LLVM_DEBUG(dbgs() << "LEV: Epilogue vectorization factor is forced.\n");
    ElementCount ForcedEC = ElementCount::getFixed(EpilogueVectorizationForceVF);
    if (hasPlanWithVF(ForcedEC))
      return {ForcedEC, 0, 0};

    LLVM_DEBUG(dbgs() << "LEV: Epilogue vectorization forced factor is not "
                         "viable.\n");
    return Result;
  }

  if (OrigLoop->getHeader()->getParent()->hasOptSize()) {
    LLVM_DEBUG(
        dbgs() << "LEV: Epilogue vectorization skipped due to opt for size.\n");
    return Result;
  }

  if (!CM.isEpilogueVectorizationProfitable(MainLoopVF, IC)) {
    LLVM_DEBUG(dbgs() << "LEV: Epilogue vectorization is not profitable for "
                         "this loop\n");
    return Result;
  }

  // If MainLoopVF = vscale x 2, and vscale is expected to be 4, then we know
  // the main loop handles 8 lanes per iteration. We could still benefit from
  // vectorizing the epilogue loop with VF=4.
  ElementCount EstimatedRuntimeVF = ElementCount::getFixed(
      estimateElementCount(MainLoopVF, CM.getVScaleForTuning()));

  ScalarEvolution &SE = *PSE.getSE();
  Type *TCType = Legal->getWidestInductionType();
  const SCEV *RemainingIterations = nullptr;
  unsigned MaxTripCount = 0;
  const SCEV *TC =
      vputils::getSCEVExprForVPValue(getPlanFor(MainLoopVF).getTripCount(), SE);
  assert(!isa<SCEVCouldNotCompute>(TC) && "Trip count SCEV must be computable");
  RemainingIterations =
      SE.getURemExpr(TC, SE.getElementCount(TCType, MainLoopVF * IC));

  // No iterations left to process in the epilogue.
  if (RemainingIterations->isZero())
    return Result;

  if (MainLoopVF.isFixed()) {
    MaxTripCount = MainLoopVF.getFixedValue() * IC - 1;
    if (SE.isKnownPredicate(CmpInst::ICMP_ULT, RemainingIterations,
                            SE.getConstant(TCType, MaxTripCount))) {
      MaxTripCount = SE.getUnsignedRangeMax(RemainingIterations).getZExtValue();
    }
    LLVM_DEBUG(dbgs() << "LEV: Maximum Trip Count for Epilogue: "
                      << MaxTripCount << "\n");
  }

  for (auto &NextVF : ProfitableVFs) {
    // Skip candidate VFs without a corresponding VPlan.
    if (!hasPlanWithVF(NextVF.Width))
      continue;

    // Skip candidate VFs with widths >= the (estimated) runtime VF (scalable
    // vectors) or > the VF of the main loop (fixed vectors).
    if ((!NextVF.Width.isScalable() && MainLoopVF.isScalable() &&
         ElementCount::isKnownGE(NextVF.Width, EstimatedRuntimeVF)) ||
        (NextVF.Width.isScalable() &&
         ElementCount::isKnownGE(NextVF.Width, MainLoopVF)) ||
        (!NextVF.Width.isScalable() && !MainLoopVF.isScalable() &&
         ElementCount::isKnownGT(NextVF.Width, MainLoopVF)))
      continue;

    // If NextVF is greater than the number of remaining iterations, the
    // epilogue loop would be dead. Skip such factors.
    if (RemainingIterations && !NextVF.Width.isScalable()) {
      if (SE.isKnownPredicate(
              CmpInst::ICMP_UGT,
              SE.getConstant(TCType, NextVF.Width.getFixedValue()),
              RemainingIterations))
        continue;
    }

    if (Result.Width.isScalar() ||
        isMoreProfitable(NextVF, Result, MaxTripCount, !CM.foldTailByMasking()))
      Result = NextVF;
  }

  if (Result != VectorizationFactor::Disabled())
    LLVM_DEBUG(dbgs() << "LEV: Vectorizing epilogue loop with VF = "
                      << Result.Width << "\n");
  return Result;
}

std::pair<unsigned, unsigned>
LoopVectorizationCostModel::getSmallestAndWidestTypes() {
  unsigned MinWidth = -1U;
  unsigned MaxWidth = 8;
  const DataLayout &DL = TheFunction->getDataLayout();
  // For in-loop reductions, no element types are added to ElementTypesInLoop
  // if there are no loads/stores in the loop. In this case, check through the
  // reduction variables to determine the maximum width.
  if (ElementTypesInLoop.empty() && !Legal->getReductionVars().empty()) {
    for (const auto &PhiDescriptorPair : Legal->getReductionVars()) {
      const RecurrenceDescriptor &RdxDesc = PhiDescriptorPair.second;
      // When finding the min width used by the recurrence we need to account
      // for casts on the input operands of the recurrence.
      MinWidth = std::min(
          MinWidth,
          std::min(RdxDesc.getMinWidthCastToRecurrenceTypeInBits(),
                   RdxDesc.getRecurrenceType()->getScalarSizeInBits()));
      MaxWidth = std::max(MaxWidth,
                          RdxDesc.getRecurrenceType()->getScalarSizeInBits());
    }
  } else {
    for (Type *T : ElementTypesInLoop) {
      MinWidth = std::min<unsigned>(
          MinWidth, DL.getTypeSizeInBits(T->getScalarType()).getFixedValue());
      MaxWidth = std::max<unsigned>(
          MaxWidth, DL.getTypeSizeInBits(T->getScalarType()).getFixedValue());
    }
  }
  return {MinWidth, MaxWidth};
}

void LoopVectorizationCostModel::collectElementTypesForWidening() {
  ElementTypesInLoop.clear();
  // For each block.
  for (BasicBlock *BB : TheLoop->blocks()) {
    // For each instruction in the loop.
    for (Instruction &I : BB->instructionsWithoutDebug()) {
      Type *T = I.getType();

      // Skip ignored values.
      if (ValuesToIgnore.count(&I))
        continue;

      // Only examine Loads, Stores and PHINodes.
      if (!isa<LoadInst>(I) && !isa<StoreInst>(I) && !isa<PHINode>(I))
        continue;

      // Examine PHI nodes that are reduction variables. Update the type to
      // account for the recurrence type.
      if (auto *PN = dyn_cast<PHINode>(&I)) {
        if (!Legal->isReductionVariable(PN))
          continue;
        const RecurrenceDescriptor &RdxDesc =
            Legal->getRecurrenceDescriptor(PN);
        if (PreferInLoopReductions || useOrderedReductions(RdxDesc) ||
            TTI.preferInLoopReduction(RdxDesc.getRecurrenceKind(),
                                      RdxDesc.getRecurrenceType()))
          continue;
        T = RdxDesc.getRecurrenceType();
      }

      // Examine the stored values.
      if (auto *ST = dyn_cast<StoreInst>(&I))
        T = ST->getValueOperand()->getType();

      assert(T->isSized() &&
             "Expected the load/store/recurrence type to be sized");

      ElementTypesInLoop.insert(T);
    }
  }
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

  if (!CM.isScalarEpilogueAllowed())
    return 1;

  if (any_of(Plan.getVectorLoopRegion()->getEntryBasicBlock()->phis(),
             IsaPred<VPEVLBasedIVPHIRecipe>)) {
    LLVM_DEBUG(dbgs() << "LV: Preference for VP intrinsics indicated. "
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

  // If we did not calculate the cost for VF (because the user selected the VF)
  // then we calculate the cost of VF here.
  if (LoopCost == 0) {
    if (VF.isScalar())
      LoopCost = CM.expectedCost(VF);
    else
      LoopCost = cost(Plan, VF);
    assert(LoopCost.isValid() && "Expected to have chosen a VF with valid cost");

    // Loop body is free and there is no need for interleaving.
    if (LoopCost == 0)
      return 1;
  }

  VPRegisterUsage R =
      calculateRegisterUsageForPlan(Plan, {VF}, TTI, CM.ValuesToIgnore)[0];
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
  auto BestKnownTC = getSmallBestKnownTC(PSE, OrigLoop);

  // For fixed length VFs treat a scalable trip count as unknown.
  if (BestKnownTC && (BestKnownTC->isFixed() || VF.isScalable())) {
    // Re-evaluate trip counts and VFs to be in the same numerical space.
    unsigned AvailableTC =
        estimateElementCount(*BestKnownTC, CM.getVScaleForTuning());
    unsigned EstimatedVF = estimateElementCount(VF, CM.getVScaleForTuning());

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

  // For any scalar loop that either requires runtime checks or predication we
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
  const bool AggressivelyInterleaveReductions =
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
    if (VF.isScalar() && AggressivelyInterleaveReductions) {
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
  if (AggressivelyInterleaveReductions) {
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
            /*Extract=*/false, CostKind);
      }
      ScalarCost +=
          VF.getFixedValue() * TTI.getCFInstrCost(Instruction::PHI, CostKind);
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
                /*Extract*/ true, CostKind);
          }
        }
      }

    // Scale the total scalar cost by block probability.
    ScalarCost /= getPredBlockCostDivisor(CostKind);

    // Compute the discount. A non-negative discount means the vector version
    // of the instruction costs more, and scalarizing would be beneficial.
    Discount += VectorCost - ScalarCost;
    ScalarCosts[I] = ScalarCost;
  }

  return Discount;
}

InstructionCost LoopVectorizationCostModel::expectedCost(ElementCount VF) {
  InstructionCost Cost;

  // If the vector loop gets executed exactly once with the given VF, ignore the
  // costs of comparison and induction instructions, as they'll get simplified
  // away.
  SmallPtrSet<Instruction *, 2> ValuesToIgnoreForVF;
  auto TC = getSmallConstantTripCount(PSE.getSE(), TheLoop);
  if (TC == VF && !foldTailByMasking())
    addFullyUnrolledInstructionsToIgnore(TheLoop, Legal->getInductionVars(),
                                         ValuesToIgnoreForVF);

  // For each block.
  for (BasicBlock *BB : TheLoop->blocks()) {
    InstructionCost BlockCost;

    // For each instruction in the old loop.
    for (Instruction &I : BB->instructionsWithoutDebug()) {
      // Skip ignored values.
      if (ValuesToIgnore.count(&I) || ValuesToIgnoreForVF.count(&I) ||
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

    // If we are vectorizing a predicated block, it will have been
    // if-converted. This means that the block's instructions (aside from
    // stores and instructions that may divide by zero) will now be
    // unconditionally executed. For the scalar case, we may not always execute
    // the predicated block, if it is an if-else block. Thus, scale the block's
    // cost by the probability of executing it. blockNeedsPredication from
    // Legal is used so as to not include all blocks in tail folded loops.
    if (VF.isScalar() && Legal->blockNeedsPredication(BB))
      BlockCost /= getPredBlockCostDivisor(CostKind);

    Cost += BlockCost;
  }

  return Cost;
}

/// Gets Address Access SCEV after verifying that the access pattern
/// is loop invariant except the induction variable dependence.
///
/// This SCEV can be sent to the Target in order to estimate the address
/// calculation cost.
static const SCEV *getAddressAccessSCEV(
              Value *Ptr,
              LoopVectorizationLegality *Legal,
              PredicatedScalarEvolution &PSE,
              const Loop *TheLoop) {

  auto *Gep = dyn_cast<GetElementPtrInst>(Ptr);
  if (!Gep)
    return nullptr;

  // We are looking for a gep with all loop invariant indices except for one
  // which should be an induction variable.
  auto *SE = PSE.getSE();
  unsigned NumOperands = Gep->getNumOperands();
  for (unsigned Idx = 1; Idx < NumOperands; ++Idx) {
    Value *Opd = Gep->getOperand(Idx);
    if (!SE->isLoopInvariant(SE->getSCEV(Opd), TheLoop) &&
        !Legal->isInductionVariable(Opd))
      return nullptr;
  }

  // Now we know we have a GEP ptr, %inv, %ind, %inv. return the Ptr SCEV.
  return PSE.getSCEV(Ptr);
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
  const SCEV *PtrSCEV = getAddressAccessSCEV(Ptr, Legal, PSE, TheLoop);

  // Get the cost of the scalar memory instruction and address computation.
  InstructionCost Cost = VF.getFixedValue() * TTI.getAddressComputationCost(
                                                  PtrTy, SE, PtrSCEV, CostKind);

  // Don't pass *I here, since it is scalar but will actually be part of a
  // vectorized loop where the user of it is a vectorized instruction.
  const Align Alignment = getLoadStoreAlignment(I);
  Cost += VF.getFixedValue() * TTI.getMemoryOpCost(I->getOpcode(),
                                                   ValTy->getScalarType(),
                                                   Alignment, AS, CostKind);

  // Get the overhead of the extractelement and insertelement instructions
  // we might create due to scalarization.
  Cost += getScalarizationOverhead(I, VF);

  // If we have a predicated load/store, it will need extra i1 extracts and
  // conditional branches, but may not be executed for each vector lane. Scale
  // the cost by the probability of executing the predicated block.
  if (isPredicatedInst(I)) {
    Cost /= getPredBlockCostDivisor(CostKind);

    // Add the cost of an i1 extract and a branch
    auto *VecI1Ty =
        VectorType::get(IntegerType::getInt1Ty(ValTy->getContext()), VF);
    Cost += TTI.getScalarizationOverhead(
        VecI1Ty, APInt::getAllOnes(VF.getFixedValue()),
        /*Insert=*/false, /*Extract=*/true, CostKind);
    Cost += TTI.getCFInstrCost(Instruction::Br, CostKind);

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
  if (Legal->isMaskRequired(I)) {
    Cost += TTI.getMaskedMemoryOpCost(I->getOpcode(), VectorTy, Alignment, AS,
                                      CostKind);
  } else {
    TTI::OperandValueInfo OpInfo = TTI::getOperandInfo(I->getOperand(0));
    Cost += TTI.getMemoryOpCost(I->getOpcode(), VectorTy, Alignment, AS,
                                CostKind, OpInfo, I);
  }

  bool Reverse = ConsecutiveStride < 0;
  if (Reverse)
    Cost += TTI.getShuffleCost(TargetTransformInfo::SK_Reverse, VectorTy,
                               VectorTy, {}, CostKind, 0);
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
    return TTI.getAddressComputationCost(PtrTy, nullptr, nullptr, CostKind) +
           TTI.getMemoryOpCost(Instruction::Load, ValTy, Alignment, AS,
                               CostKind) +
           TTI.getShuffleCost(TargetTransformInfo::SK_Broadcast, VectorTy,
                              VectorTy, {}, CostKind);
  }
  StoreInst *SI = cast<StoreInst>(I);

  bool IsLoopInvariantStoreValue = Legal->isInvariant(SI->getValueOperand());
  // TODO: We have existing tests that request the cost of extracting element
  // VF.getKnownMinValue() - 1 from a scalable vector. This does not represent
  // the actual generated code, which involves extracting the last element of
  // a scalable vector where the lane to extract is unknown at compile time.
  InstructionCost Cost =
      TTI.getAddressComputationCost(PtrTy, nullptr, nullptr, CostKind) +
      TTI.getMemoryOpCost(Instruction::Store, ValTy, Alignment, AS, CostKind);
  if (!IsLoopInvariantStoreValue)
    Cost += TTI.getIndexedVectorInstrCostFromEnd(Instruction::ExtractElement,
                                                 VectorTy, CostKind, 0);
  return Cost;
}

InstructionCost
LoopVectorizationCostModel::getGatherScatterCost(Instruction *I,
                                                 ElementCount VF) {
  Type *ValTy = getLoadStoreType(I);
  auto *VectorTy = cast<VectorType>(toVectorTy(ValTy, VF));
  const Align Alignment = getLoadStoreAlignment(I);
  const Value *Ptr = getLoadStorePointerOperand(I);
  Type *PtrTy = toVectorTy(Ptr->getType(), VF);

  return TTI.getAddressComputationCost(PtrTy, nullptr, nullptr, CostKind) +
         TTI.getGatherScatterOpCost(I->getOpcode(), VectorTy, Ptr,
                                    Legal->isMaskRequired(I), Alignment,
                                    CostKind, I);
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
      (Group->requiresScalarEpilogue() && !isScalarEpilogueAllowed()) ||
      (isa<StoreInst>(I) && !Group->isFull());
  InstructionCost Cost = TTI.getInterleavedMemoryOpCost(
      InsertPos->getOpcode(), WideVecTy, Group->getFactor(), Indices,
      Group->getAlign(), AS, CostKind, Legal->isMaskRequired(I),
      UseMaskForGaps);

  if (Group->isReverse()) {
    // TODO: Add support for reversed masked interleaved access.
    assert(!Legal->isMaskRequired(I) &&
           "Reverse masked interleaved access not supported.");
    Cost += Group->getNumMembers() *
            TTI.getShuffleCost(TargetTransformInfo::SK_Reverse, VectorTy,
                               VectorTy, {}, CostKind, 0);
  }
  return Cost;
}

std::optional<InstructionCost>
LoopVectorizationCostModel::getReductionPatternCost(Instruction *I,
                                                    ElementCount VF,
                                                    Type *Ty) const {
  using namespace llvm::PatternMatch;
  // Early exit for no inloop reductions
  if (InLoopReductions.empty() || VF.isScalar() || !isa<VectorType>(Ty))
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
  Instruction *LastChain = InLoopReductionImmediateChains.lookup(RetI);
  if (!LastChain)
    return std::nullopt;

  // Find the reduction this chain is a part of and calculate the basic cost of
  // the reduction on its own.
  Instruction *ReductionPhi = LastChain;
  while (!isa<PHINode>(ReductionPhi))
    ReductionPhi = InLoopReductionImmediateChains.at(ReductionPhi);

  const RecurrenceDescriptor &RdxDesc =
      Legal->getRecurrenceDescriptor(cast<PHINode>(ReductionPhi));

  InstructionCost BaseCost;
  RecurKind RK = RdxDesc.getRecurrenceKind();
  if (RecurrenceDescriptor::isMinMaxRecurrenceKind(RK)) {
    Intrinsic::ID MinMaxID = getMinMaxReductionIntrinsicOp(RK);
    BaseCost = TTI.getMinMaxReductionCost(MinMaxID, VectorTy,
                                          RdxDesc.getFastMathFlags(), CostKind);
  } else {
    BaseCost = TTI.getArithmeticReductionCost(
        RdxDesc.getOpcode(), VectorTy, RdxDesc.getFastMathFlags(), CostKind);
  }

  // For a call to the llvm.fmuladd intrinsic we need to add the cost of a
  // normal fmul instruction to the cost of the fadd reduction.
  if (RK == RecurKind::FMulAdd)
    BaseCost +=
        TTI.getArithmeticInstrCost(Instruction::FMul, VectorTy, CostKind);

  // If we're using ordered reductions then we can just return the base cost
  // here, since getArithmeticReductionCost calculates the full ordered
  // reduction cost when FP reassociation is not allowed.
  if (useOrderedReductions(RdxDesc))
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
                             TTI::CastContextHint::None, CostKind, Op0);
    InstructionCost MulCost =
        TTI.getArithmeticInstrCost(Instruction::Mul, MulType, CostKind);
    InstructionCost Ext2Cost =
        TTI.getCastInstrCost(RedOp->getOpcode(), VectorTy, MulType,
                             TTI::CastContextHint::None, CostKind, RedOp);

    InstructionCost RedCost = TTI.getMulAccReductionCost(
        IsUnsigned, RdxDesc.getRecurrenceType(), ExtType, CostKind);

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
        RdxDesc.getFastMathFlags(), CostKind);

    InstructionCost ExtCost =
        TTI.getCastInstrCost(RedOp->getOpcode(), VectorTy, ExtType,
                             TTI::CastContextHint::None, CostKind, RedOp);
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
          TTI::CastContextHint::None, CostKind, Op0);
      InstructionCost ExtCost1 = TTI.getCastInstrCost(
          Op1->getOpcode(), VectorTy, VectorType::get(Op1Ty, VectorTy),
          TTI::CastContextHint::None, CostKind, Op1);
      InstructionCost MulCost =
          TTI.getArithmeticInstrCost(Instruction::Mul, VectorTy, CostKind);

      InstructionCost RedCost = TTI.getMulAccReductionCost(
          IsUnsigned, RdxDesc.getRecurrenceType(), ExtType, CostKind);
      InstructionCost ExtraExtCost = 0;
      if (Op0Ty != LargestOpTy || Op1Ty != LargestOpTy) {
        Instruction *ExtraExtOp = (Op0Ty != LargestOpTy) ? Op0 : Op1;
        ExtraExtCost = TTI.getCastInstrCost(
            ExtraExtOp->getOpcode(), ExtType,
            VectorType::get(ExtraExtOp->getOperand(0)->getType(), VectorTy),
            TTI::CastContextHint::None, CostKind, ExtraExtOp);
      }

      if (RedCost.isValid() &&
          (RedCost + ExtraExtCost) < (ExtCost0 + ExtCost1 + MulCost + BaseCost))
        return I == RetI ? RedCost : 0;
    } else if (!match(I, m_ZExtOrSExt(m_Value()))) {
      // Matched reduce.add(mul())
      InstructionCost MulCost =
          TTI.getArithmeticInstrCost(Instruction::Mul, VectorTy, CostKind);

      InstructionCost RedCost = TTI.getMulAccReductionCost(
          true, RdxDesc.getRecurrenceType(), VectorTy, CostKind);

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
    return TTI.getAddressComputationCost(PtrTy, nullptr, nullptr, CostKind) +
           TTI.getMemoryOpCost(I->getOpcode(), ValTy, Alignment, AS, CostKind,
                               OpInfo, I);
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

    for (Type *VectorTy : getContainedTypes(RetTy)) {
      Cost += TTI.getScalarizationOverhead(
          cast<VectorType>(VectorTy), APInt::getAllOnes(VF.getFixedValue()),
          /*Insert=*/true,
          /*Extract=*/false, CostKind);
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
  return Cost + TTI.getOperandsScalarizationOverhead(Tys, CostKind);
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
          isLegalGatherOrScatter(&I, VF) ?
          getGatherScatterCost(&I, VF) : InstructionCost::getInvalid();

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
          isLegalGatherOrScatter(&I, VF)
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
      if (const auto *Group = getInterleavedAccessGroup(&I))
        setWideningDecision(Group, VF, Decision, Cost);
      else
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
        if ((InstOp->getParent() == I->getParent()) && !isa<PHINode>(InstOp) &&
            AddrDefs.insert(InstOp).second)
          Worklist.push_back(InstOp);
  }

  for (auto *I : AddrDefs) {
    if (isa<LoadInst>(I)) {
      // Setting the desired widening decision should ideally be handled in
      // by cost functions, but since this involves the task of finding out
      // if the loaded register is involved in an address computation, it is
      // instead changed here when we know this is the case.
      InstWidening Decision = getWideningDecision(I, VF);
      if (Decision == CM_Widen || Decision == CM_Widen_Reverse)
        // Scalarize a widened load of address.
        setWideningDecision(
            I, VF, CM_Scalarize,
            (VF.getKnownMinValue() *
             getMemoryInstructionCost(I, ElementCount::getFixed(1))));
      else if (const auto *Group = getInterleavedAccessGroup(I)) {
        // Scalarize an interleave group of address loads.
        for (unsigned I = 0; I < Group->getFactor(); ++I) {
          if (Instruction *Member = Group->getMember(I))
            setWideningDecision(
                Member, VF, CM_Scalarize,
                (VF.getKnownMinValue() *
                 getMemoryInstructionCost(Member, ElementCount::getFixed(1))));
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
        InstructionCost ScalarCallCost =
            TTI.getCallInstrCost(ScalarFunc, ScalarRetTy, ScalarTys, CostKind);

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

      bool MaskRequired = Legal->isMaskRequired(CI);
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
        VectorCost = TTI.getCallInstrCost(nullptr, RetTy, Tys, CostKind);

      // Find the cost of an intrinsic; some targets may have instructions that
      // perform the operation without needing an actual call.
      Intrinsic::ID IID = getVectorIntrinsicIDForCall(CI, TLI);
      if (IID != Intrinsic::not_intrinsic)
        IntrinsicCost = getVectorIntrinsicCost(CI, VF);

      InstructionCost Cost = ScalarCost;
      InstWidening Decision = CM_Scalarize;

      if (VectorCost <= Cost) {
        Cost = VectorCost;
        Decision = CM_VectorCall;
      }

      if (IntrinsicCost <= Cost) {
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

  Type *RetTy = I->getType();
  if (canTruncateToMinimalBitwidth(I, VF))
    RetTy = IntegerType::get(RetTy->getContext(), MinBWs[I]);
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
  case Instruction::Br: {
    // In cases of scalarized and predicated instructions, there will be VF
    // predicated blocks in the vectorized loop. Each branch around these
    // blocks requires also an extract of its vector compare i1 element.
    // Note that the conditional branch from the loop latch will be replaced by
    // a single branch controlling the loop, so there is no extra overhead from
    // scalarization.
    bool ScalarPredicatedBB = false;
    BranchInst *BI = cast<BranchInst>(I);
    if (VF.isVector() && BI->isConditional() &&
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
      return (
          TTI.getScalarizationOverhead(
              VecI1Ty, APInt::getAllOnes(VF.getFixedValue()),
              /*Insert*/ false, /*Extract*/ true, CostKind) +
          (TTI.getCFInstrCost(Instruction::Br, CostKind) * VF.getFixedValue()));
    }

    if (I->getParent() == TheLoop->getLoopLatch() || VF.isScalar())
      // The back-edge branch will remain, as will all scalar branches.
      return TTI.getCFInstrCost(Instruction::Br, CostKind);

    // This branch will be eliminated by if-conversion.
    return 0;
    // Note: We currently assume zero cost for an unconditional branch inside
    // a predicated block since it will become a fall-through, although we
    // may decide in the future to call TTI for all branches.
  }
  case Instruction::Switch: {
    if (VF.isScalar())
      return TTI.getCFInstrCost(Instruction::Switch, CostKind);
    auto *Switch = cast<SwitchInst>(I);
    return Switch->getNumCases() *
           TTI.getCmpSelInstrCost(
               Instruction::ICmp,
               toVectorTy(Switch->getCondition()->getType(), VF),
               toVectorTy(Type::getInt1Ty(I->getContext()), VF),
               CmpInst::ICMP_EQ, CostKind);
  }
  case Instruction::PHI: {
    auto *Phi = cast<PHINode>(I);

    // First-order recurrences are replaced by vector shuffles inside the loop.
    if (VF.isVector() && Legal->isFixedOrderRecurrence(Phi)) {
      SmallVector<int> Mask(VF.getKnownMinValue());
      std::iota(Mask.begin(), Mask.end(), VF.getKnownMinValue() - 1);
      return TTI.getShuffleCost(TargetTransformInfo::SK_Splice,
                                cast<VectorType>(VectorTy),
                                cast<VectorType>(VectorTy), Mask, CostKind,
                                VF.getKnownMinValue() - 1);
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
                 CmpInst::BAD_ICMP_PREDICATE, CostKind);
    }

    // When tail folding with EVL, if the phi is part of an out of loop
    // reduction then it will be transformed into a wide vp_merge.
    if (VF.isVector() && foldTailWithEVL() &&
        Legal->getReductionVars().contains(Phi) && !isInLoopReduction(Phi)) {
      IntrinsicCostAttributes ICA(
          Intrinsic::vp_merge, toVectorTy(Phi->getType(), VF),
          {toVectorTy(Type::getInt1Ty(Phi->getContext()), VF)});
      return TTI.getIntrinsicInstrCost(ICA, CostKind);
    }

    return TTI.getCFInstrCost(Instruction::PHI, CostKind);
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
        MulCost =
            TTI.getArithmeticInstrCost(Instruction::Mul, VectorTy, CostKind);

      // Find the cost of the histogram operation itself.
      Type *PtrTy = VectorType::get(HGram->Load->getPointerOperandType(), VF);
      Type *ScalarTy = I->getType();
      Type *MaskTy = VectorType::get(Type::getInt1Ty(I->getContext()), VF);
      IntrinsicCostAttributes ICA(Intrinsic::experimental_vector_histogram_add,
                                  Type::getVoidTy(I->getContext()),
                                  {PtrTy, ScalarTy, MaskTy});

      // Add the costs together with the add/sub operation.
      return TTI.getIntrinsicInstrCost(ICA, CostKind) + MulCost +
             TTI.getArithmeticInstrCost(I->getOpcode(), VectorTy, CostKind);
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
        I->getOpcode(), VectorTy, CostKind,
        {TargetTransformInfo::OK_AnyValue, TargetTransformInfo::OP_None},
        Op2Info, Operands, I, TLI);
  }
  case Instruction::FNeg: {
    return TTI.getArithmeticInstrCost(
        I->getOpcode(), VectorTy, CostKind,
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

      SmallVector<const Value *, 2> Operands{Op0, Op1};
      return TTI.getArithmeticInstrCost(
          match(I, m_LogicalOr()) ? Instruction::Or : Instruction::And, VectorTy,
          CostKind, {Op1VK, Op1VP}, {Op2VK, Op2VP}, Operands, I);
    }

    Type *CondTy = SI->getCondition()->getType();
    if (!ScalarCond)
      CondTy = VectorType::get(CondTy, VF);

    CmpInst::Predicate Pred = CmpInst::BAD_ICMP_PREDICATE;
    if (auto *Cmp = dyn_cast<CmpInst>(SI->getCondition()))
      Pred = Cmp->getPredicate();
    return TTI.getCmpSelInstrCost(I->getOpcode(), VectorTy, CondTy, Pred,
                                  CostKind, {TTI::OK_AnyValue, TTI::OP_None},
                                  {TTI::OK_AnyValue, TTI::OP_None}, I);
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    Type *ValTy = I->getOperand(0)->getType();

    if (canTruncateToMinimalBitwidth(I, VF)) {
      [[maybe_unused]] Instruction *Op0AsInstruction =
          dyn_cast<Instruction>(I->getOperand(0));
      assert((!canTruncateToMinimalBitwidth(Op0AsInstruction, VF) ||
              MinBWs[I] == MinBWs[Op0AsInstruction]) &&
             "if both the operand and the compare are marked for "
             "truncation, they must have the same bitwidth");
      ValTy = IntegerType::get(ValTy->getContext(), MinBWs[I]);
    }

    VectorTy = toVectorTy(ValTy, VF);
    return TTI.getCmpSelInstrCost(
        I->getOpcode(), VectorTy, CmpInst::makeCmpResultType(VectorTy),
        cast<CmpInst>(I)->getPredicate(), CostKind,
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
                                  Trunc->getSrcTy(), CCH, CostKind, Trunc);
    }

    // Detect reduction patterns
    if (auto RedCost = getReductionPatternCost(I, VF, VectorTy))
      return *RedCost;

    Type *SrcScalarTy = I->getOperand(0)->getType();
    Instruction *Op0AsInstruction = dyn_cast<Instruction>(I->getOperand(0));
    if (canTruncateToMinimalBitwidth(Op0AsInstruction, VF))
      SrcScalarTy =
          IntegerType::get(SrcScalarTy->getContext(), MinBWs[Op0AsInstruction]);
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

    return TTI.getCastInstrCost(Opcode, VectorTy, SrcVecTy, CCH, CostKind, I);
  }
  case Instruction::Call:
    return getVectorCallCost(cast<CallInst>(I), VF);
  case Instruction::ExtractValue:
    return TTI.getInstructionCost(I, CostKind);
  case Instruction::Alloca:
    // We cannot easily widen alloca to a scalable alloca, as
    // the result would need to be a vector of pointers.
    if (VF.isScalable())
      return InstructionCost::getInvalid();
    [[fallthrough]];
  default:
    // This opcode is unknown. Assume that it is the same as 'mul'.
    return TTI.getArithmeticInstrCost(Instruction::Mul, VectorTy, CostKind);
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
  MapVector<Value *, SmallVector<Value *>> DeadInvariantStoreOps;
  for (BasicBlock *BB : reverse(make_range(DFS.beginRPO(), DFS.endRPO())))
    for (Instruction &I : reverse(*BB)) {
      // Find all stores to invariant variables. Since they are going to sink
      // outside the loop we do not need calculate cost for them.
      StoreInst *SI;
      if ((SI = dyn_cast<StoreInst>(&I)) &&
          Legal->isInvariantAddressOfReduction(SI->getPointerOperand())) {
        ValuesToIgnore.insert(&I);
        DeadInvariantStoreOps[SI->getPointerOperand()].push_back(
            SI->getValueOperand());
      }

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
      if (auto *Br = dyn_cast<BranchInst>(&I)) {
        if (Br->isConditional())
          DeadOps.push_back(&I);
      }
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
    DeadInterleavePointerOps.append(Op->op_begin(), Op->op_end());
  }

  for (const auto &[_, Ops] : DeadInvariantStoreOps)
    llvm::append_range(DeadOps, drop_end(Ops));

  // Mark ops that would be trivially dead and are only used by ignored
  // instructions as free.
  BasicBlock *Header = TheLoop->getHeader();

  // Returns true if the block contains only dead instructions. Such blocks will
  // be removed by VPlan-to-VPlan transforms and won't be considered by the
  // VPlan-based cost model, so skip them in the legacy cost-model as well.
  auto IsEmptyBlock = [this](BasicBlock *BB) {
    return all_of(*BB, [this](Instruction &I) {
      return ValuesToIgnore.contains(&I) || VecValuesToIgnore.contains(&I) ||
             (isa<BranchInst>(&I) && !cast<BranchInst>(&I)->isConditional());
    });
  };
  for (unsigned I = 0; I != DeadOps.size(); ++I) {
    auto *Op = dyn_cast<Instruction>(DeadOps[I]);

    // Check if the branch should be considered dead.
    if (auto *Br = dyn_cast_or_null<BranchInst>(Op)) {
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
    DeadOps.append(Op->op_begin(), Op->op_end());
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
    const SmallVectorImpl<Instruction *> &Casts = IndDes.getCastInsts();
    VecValuesToIgnore.insert_range(Casts);
  }
}

void LoopVectorizationCostModel::collectInLoopReductions() {
  // Avoid duplicating work finding in-loop reductions.
  if (!InLoopReductions.empty())
    return;

  for (const auto &Reduction : Legal->getReductionVars()) {
    PHINode *Phi = Reduction.first;
    const RecurrenceDescriptor &RdxDesc = Reduction.second;

    // We don't collect reductions that are type promoted (yet).
    if (RdxDesc.getRecurrenceType() != Phi->getType())
      continue;

    // If the target would prefer this reduction to happen "in-loop", then we
    // want to record it as such.
    RecurKind Kind = RdxDesc.getRecurrenceKind();
    if (!PreferInLoopReductions && !useOrderedReductions(RdxDesc) &&
        !TTI.preferInLoopReduction(Kind, Phi->getType()))
      continue;

    // Check that we can correctly put the reductions into the loop, by
    // finding the chain of operations that leads from the phi to the loop
    // exit value.
    SmallVector<Instruction *, 4> ReductionOperations =
        RdxDesc.getReductionOpChain(Phi, TheLoop);
    bool InLoop = !ReductionOperations.empty();

    if (InLoop) {
      InLoopReductions.insert(Phi);
      // Add the elements to InLoopReductionImmediateChains for cost modelling.
      Instruction *LastChain = Phi;
      for (auto *I : ReductionOperations) {
        InLoopReductionImmediateChains[I] = LastChain;
        LastChain = I;
      }
    }
    LLVM_DEBUG(dbgs() << "LV: Using " << (InLoop ? "inloop" : "out of loop")
                      << " reduction for phi: " << *Phi << "\n");
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
                                     LoopVectorizationCostModel &CM) {
  unsigned WidestType;
  std::tie(std::ignore, WidestType) = CM.getSmallestAndWidestTypes();

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
      VF = determineVPlanVF(TTI, CM);
      LLVM_DEBUG(dbgs() << "LV: VPlan computed VF " << VF << ".\n");

      // Make sure we have a VF > 1 for stress testing.
      if (VPlanBuildStressTest && (VF.isScalar() || VF.isZero())) {
        LLVM_DEBUG(dbgs() << "LV: VPlan stress testing: "
                          << "overriding computed VF.\n");
        VF = ElementCount::getFixed(4);
      }
    } else if (UserVF.isScalable() && !TTI.supportsScalableVectors() &&
               !ForceTargetSupportsScalableVectors) {
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
  CM.collectElementTypesForWidening();

  FixedScalableVFPair MaxFactors = CM.computeMaxVF(UserVF, UserIC);
  if (!MaxFactors) // Cases that should not to be vectorized nor interleaved.
    return;

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
      CM.collectInLoopReductions();
      if (CM.selectUserVectorizationFactor(UserVF)) {
        LLVM_DEBUG(dbgs() << "LV: Using user VF " << UserVF << ".\n");
        buildVPlansWithVPRecipes(UserVF, UserVF);
        LLVM_DEBUG(printPlans(dbgs()));
        return;
      }
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

  CM.collectInLoopReductions();
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

bool VPCostContext::isLegacyUniformAfterVectorization(Instruction *I,
                                                      ElementCount VF) const {
  return CM.isUniformAfterVectorization(I, VF);
}

bool VPCostContext::skipCostComputation(Instruction *UI, bool IsVector) const {
  return CM.ValuesToIgnore.contains(UI) ||
         (IsVector && CM.VecValuesToIgnore.contains(UI)) ||
         SkipCostComputation.contains(UI);
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
    auto *Term = dyn_cast<BranchInst>(EB->getTerminator());
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
          any_of(OpI->users(), [&ExitInstrs, this](User *U) {
            return OrigLoop->contains(cast<Instruction>(U)->getParent()) &&
                   !ExitInstrs.contains(cast<Instruction>(U));
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

  // Pre-compute costs for instructions that are forced-scalar or profitable to
  // scalarize. Their costs will be computed separately in the legacy cost
  // model.
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
  for (const auto &[Scalarized, ScalarCost] : CM.InstsToScalarize[VF]) {
    if (CostCtx.skipCostComputation(Scalarized, VF.isVector()))
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

InstructionCost LoopVectorizationPlanner::cost(VPlan &Plan,
                                               ElementCount VF) const {
  VPCostContext CostCtx(CM.TTI, *CM.TLI, Plan, CM, CM.CostKind);
  InstructionCost Cost = precomputeCosts(Plan, VF, CostCtx);

  // Now compute and add the VPlan-based cost.
  Cost += Plan.cost(VF, CostCtx);
#ifndef NDEBUG
  unsigned EstimatedWidth = estimateElementCount(VF, CM.getVScaleForTuning());
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

#ifndef NDEBUG
/// Return true if the original loop \ TheLoop contains any instructions that do
/// not have corresponding recipes in \p Plan and are not marked to be ignored
/// in \p CostCtx. This means the VPlan contains simplification that the legacy
/// cost-model did not account for.
static bool planContainsAdditionalSimplifications(VPlan &Plan,
                                                  VPCostContext &CostCtx,
                                                  Loop *TheLoop,
                                                  ElementCount VF) {
  // First collect all instructions for the recipes in Plan.
  auto GetInstructionForCost = [](const VPRecipeBase *R) -> Instruction * {
    if (auto *S = dyn_cast<VPSingleDefRecipe>(R))
      return dyn_cast_or_null<Instruction>(S->getUnderlyingValue());
    if (auto *WidenMem = dyn_cast<VPWidenMemoryRecipe>(R))
      return &WidenMem->getIngredient();
    return nullptr;
  };

  DenseSet<Instruction *> SeenInstrs;
  auto Iter = vp_depth_first_deep(Plan.getVectorLoopRegion()->getEntry());
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(Iter)) {
    for (VPRecipeBase &R : *VPBB) {
      if (auto *IR = dyn_cast<VPInterleaveRecipe>(&R)) {
        auto *IG = IR->getInterleaveGroup();
        unsigned NumMembers = IG->getNumMembers();
        for (unsigned I = 0; I != NumMembers; ++I) {
          if (Instruction *M = IG->getMember(I))
            SeenInstrs.insert(M);
        }
        continue;
      }
      // Unused FOR splices are removed by VPlan transforms, so the VPlan-based
      // cost model won't cost it whilst the legacy will.
      if (auto *FOR = dyn_cast<VPFirstOrderRecurrencePHIRecipe>(&R)) {
        if (none_of(FOR->users(), [](VPUser *U) {
              auto *VPI = dyn_cast<VPInstruction>(U);
              return VPI && VPI->getOpcode() ==
                                VPInstruction::FirstOrderRecurrenceSplice;
            }))
          return true;
      }
      // The VPlan-based cost model is more accurate for partial reduction and
      // comparing against the legacy cost isn't desirable.
      if (isa<VPPartialReductionRecipe>(&R))
        return true;

      /// If a VPlan transform folded a recipe to one producing a single-scalar,
      /// but the original instruction wasn't uniform-after-vectorization in the
      /// legacy cost model, the legacy cost overestimates the actual cost.
      if (auto *RepR = dyn_cast<VPReplicateRecipe>(&R)) {
        if (RepR->isSingleScalar() &&
            !CostCtx.isLegacyUniformAfterVectorization(
                RepR->getUnderlyingInstr(), VF))
          return true;
      }
      if (Instruction *UI = GetInstructionForCost(&R)) {
        // If we adjusted the predicate of the recipe, the cost in the legacy
        // cost model may be different.
        using namespace VPlanPatternMatch;
        CmpPredicate Pred;
        if (match(&R, m_Cmp(Pred, m_VPValue(), m_VPValue())) &&
            cast<VPRecipeWithIRFlags>(R).getPredicate() !=
                cast<CmpInst>(UI)->getPredicate())
          return true;
        SeenInstrs.insert(UI);
      }
    }
  }

  // Return true if the loop contains any instructions that are not also part of
  // the VPlan or are skipped for VPlan-based cost computations. This indicates
  // that the VPlan contains extra simplifications.
  return any_of(TheLoop->blocks(), [&SeenInstrs, &CostCtx,
                                    TheLoop](BasicBlock *BB) {
    return any_of(*BB, [&SeenInstrs, &CostCtx, TheLoop, BB](Instruction &I) {
      // Skip induction phis when checking for simplifications, as they may not
      // be lowered directly be lowered to a corresponding PHI recipe.
      if (isa<PHINode>(&I) && BB == TheLoop->getHeader() &&
          CostCtx.CM.Legal->isInductionPhi(cast<PHINode>(&I)))
        return false;
      return !SeenInstrs.contains(&I) && !CostCtx.skipCostComputation(&I, true);
    });
  });
}
#endif

VectorizationFactor LoopVectorizationPlanner::computeBestVF() {
  if (VPlans.empty())
    return VectorizationFactor::Disabled();
  // If there is a single VPlan with a single VF, return it directly.
  VPlan &FirstPlan = *VPlans[0];
  if (VPlans.size() == 1 && size(FirstPlan.vectorFactors()) == 1)
    return {*FirstPlan.vectorFactors().begin(), 0, 0};

  LLVM_DEBUG(dbgs() << "LV: Computing best VF using cost kind: "
                    << (CM.CostKind == TTI::TCK_RecipThroughput
                            ? "Reciprocal Throughput\n"
                        : CM.CostKind == TTI::TCK_Latency
                            ? "Instruction Latency\n"
                        : CM.CostKind == TTI::TCK_CodeSize ? "Code Size\n"
                        : CM.CostKind == TTI::TCK_SizeAndLatency
                            ? "Code Size and Latency\n"
                            : "Unknown\n"));

  ElementCount ScalarVF = ElementCount::getFixed(1);
  assert(hasPlanWithVF(ScalarVF) &&
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

  for (auto &P : VPlans) {
    ArrayRef<ElementCount> VFs(P->vectorFactors().begin(),
                               P->vectorFactors().end());

    SmallVector<VPRegisterUsage, 8> RUs;
    if (CM.useMaxBandwidth(TargetTransformInfo::RGK_ScalableVector) ||
        CM.useMaxBandwidth(TargetTransformInfo::RGK_FixedWidthVector))
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
      if (CM.OptForSize && !ForceVectorization && hasReplicatorRegion(*P)) {
        LLVM_DEBUG(
            dbgs()
            << "LV: Not considering vector loop of width " << VF
            << " because it would cause replicated blocks to be generated,"
            << " which isn't allowed when optimizing for size.\n");
        continue;
      }

      InstructionCost Cost = cost(*P, VF);
      VectorizationFactor CurrentFactor(VF, Cost, ScalarCost);

      if (CM.shouldCalculateRegPressureForVF(VF) &&
          RUs[I].exceedsMaxNumRegs(TTI, ForceTargetNumVectorRegs)) {
        LLVM_DEBUG(dbgs() << "LV(REG): Not considering vector loop of width "
                          << VF << " because it uses too many registers\n");
        continue;
      }

      if (isMoreProfitable(CurrentFactor, BestFactor, P->hasScalarTail()))
        BestFactor = CurrentFactor;

      // If profitable add it to ProfitableVF list.
      if (isMoreProfitable(CurrentFactor, ScalarFactor, P->hasScalarTail()))
        ProfitableVFs.push_back(CurrentFactor);
    }
  }

#ifndef NDEBUG
  // Select the optimal vectorization factor according to the legacy cost-model.
  // This is now only used to verify the decisions by the new VPlan-based
  // cost-model and will be retired once the VPlan-based cost-model is
  // stabilized.
  VectorizationFactor LegacyVF = selectVectorizationFactor();
  VPlan &BestPlan = getPlanFor(BestFactor.Width);

  // Pre-compute the cost and use it to check if BestPlan contains any
  // simplifications not accounted for in the legacy cost model. If that's the
  // case, don't trigger the assertion, as the extra simplifications may cause a
  // different VF to be picked by the VPlan-based cost model.
  VPCostContext CostCtx(CM.TTI, *CM.TLI, BestPlan, CM, CM.CostKind);
  precomputeCosts(BestPlan, BestFactor.Width, CostCtx);
  // Verify that the VPlan-based and legacy cost models agree, except for VPlans
  // with early exits and plans with additional VPlan simplifications. The
  // legacy cost model doesn't properly model costs for such loops.
  assert((BestFactor.Width == LegacyVF.Width || BestPlan.hasEarlyExit() ||
          planContainsAdditionalSimplifications(getPlanFor(BestFactor.Width),
                                                CostCtx, OrigLoop,
                                                BestFactor.Width) ||
          planContainsAdditionalSimplifications(
              getPlanFor(LegacyVF.Width), CostCtx, OrigLoop, LegacyVF.Width)) &&
         " VPlan cost model and legacy cost model disagreed");
  assert((BestFactor.Width.isScalar() || BestFactor.ScalarCost > 0) &&
         "when vectorizing, the scalar cost must be computed.");
#endif

  LLVM_DEBUG(dbgs() << "LV: Selecting VF: " << BestFactor.Width << ".\n");
  return BestFactor;
}

static void addRuntimeUnrollDisableMetaData(Loop *L) {
  SmallVector<Metadata *, 4> MDs;
  // Reserve first location for self reference to the LoopID metadata node.
  MDs.push_back(nullptr);
  bool IsUnrollMetadata = false;
  MDNode *LoopID = L->getLoopID();
  if (LoopID) {
    // First find existing loop unrolling disable metadata.
    for (unsigned I = 1, IE = LoopID->getNumOperands(); I < IE; ++I) {
      auto *MD = dyn_cast<MDNode>(LoopID->getOperand(I));
      if (MD) {
        const auto *S = dyn_cast<MDString>(MD->getOperand(0));
        IsUnrollMetadata =
            S && S->getString().starts_with("llvm.loop.unroll.disable");
      }
      MDs.push_back(LoopID->getOperand(I));
    }
  }

  if (!IsUnrollMetadata) {
    // Add runtime unroll disable metadata.
    LLVMContext &Context = L->getHeader()->getContext();
    SmallVector<Metadata *, 1> DisableOperands;
    DisableOperands.push_back(
        MDString::get(Context, "llvm.loop.unroll.runtime.disable"));
    MDNode *DisableNode = MDNode::get(Context, DisableOperands);
    MDs.push_back(DisableNode);
    MDNode *NewLoopID = MDNode::get(Context, MDs);
    // Set operand 0 to refer to the loop id itself.
    NewLoopID->replaceOperandWith(0, NewLoopID);
    L->setLoopID(NewLoopID);
  }
}

static Value *getStartValueFromReductionResult(VPInstruction *RdxResult) {
  using namespace VPlanPatternMatch;
  assert(RdxResult->getOpcode() == VPInstruction::ComputeFindIVResult &&
         "RdxResult must be ComputeFindIVResult");
  VPValue *StartVPV = RdxResult->getOperand(1);
  match(StartVPV, m_Freeze(m_VPValue(StartVPV)));
  return StartVPV->getLiveInIRValue();
}

// If \p EpiResumePhiR is resume VPPhi for a reduction when vectorizing the
// epilog loop, fix the reduction's scalar PHI node by adding the incoming value
// from the main vector loop.
static void fixReductionScalarResumeWhenVectorizingEpilog(
    VPPhi *EpiResumePhiR, VPTransformState &State, BasicBlock *BypassBlock) {
  // Get the VPInstruction computing the reduction result in the middle block.
  // The first operand may not be from the middle block if it is not connected
  // to the scalar preheader. In that case, there's nothing to fix.
  VPValue *Incoming = EpiResumePhiR->getOperand(0);
  match(Incoming, VPlanPatternMatch::m_ZExtOrSExt(
                      VPlanPatternMatch::m_VPValue(Incoming)));
  auto *EpiRedResult = dyn_cast<VPInstruction>(Incoming);
  if (!EpiRedResult ||
      (EpiRedResult->getOpcode() != VPInstruction::ComputeAnyOfResult &&
       EpiRedResult->getOpcode() != VPInstruction::ComputeReductionResult &&
       EpiRedResult->getOpcode() != VPInstruction::ComputeFindIVResult))
    return;

  auto *EpiRedHeaderPhi =
      cast<VPReductionPHIRecipe>(EpiRedResult->getOperand(0));
  RecurKind Kind = EpiRedHeaderPhi->getRecurrenceKind();
  Value *MainResumeValue;
  if (auto *VPI = dyn_cast<VPInstruction>(EpiRedHeaderPhi->getStartValue())) {
    assert((VPI->getOpcode() == VPInstruction::Broadcast ||
            VPI->getOpcode() == VPInstruction::ReductionStartVector) &&
           "unexpected start recipe");
    MainResumeValue = VPI->getOperand(0)->getUnderlyingValue();
  } else
    MainResumeValue = EpiRedHeaderPhi->getStartValue()->getUnderlyingValue();
  if (RecurrenceDescriptor::isAnyOfRecurrenceKind(Kind)) {
    [[maybe_unused]] Value *StartV =
        EpiRedResult->getOperand(1)->getLiveInIRValue();
    auto *Cmp = cast<ICmpInst>(MainResumeValue);
    assert(Cmp->getPredicate() == CmpInst::ICMP_NE &&
           "AnyOf expected to start with ICMP_NE");
    assert(Cmp->getOperand(1) == StartV &&
           "AnyOf expected to start by comparing main resume value to original "
           "start value");
    MainResumeValue = Cmp->getOperand(0);
  } else if (RecurrenceDescriptor::isFindIVRecurrenceKind(Kind)) {
    Value *StartV = getStartValueFromReductionResult(EpiRedResult);
    Value *SentinelV = EpiRedResult->getOperand(2)->getLiveInIRValue();
    using namespace llvm::PatternMatch;
    Value *Cmp, *OrigResumeV, *CmpOp;
    [[maybe_unused]] bool IsExpectedPattern =
        match(MainResumeValue,
              m_Select(m_OneUse(m_Value(Cmp)), m_Specific(SentinelV),
                       m_Value(OrigResumeV))) &&
        (match(Cmp, m_SpecificICmp(ICmpInst::ICMP_EQ, m_Specific(OrigResumeV),
                                   m_Value(CmpOp))) &&
         ((CmpOp == StartV && isGuaranteedNotToBeUndefOrPoison(CmpOp))));
    assert(IsExpectedPattern && "Unexpected reduction resume pattern");
    MainResumeValue = OrigResumeV;
  }
  PHINode *MainResumePhi = cast<PHINode>(MainResumeValue);

  // When fixing reductions in the epilogue loop we should already have
  // created a bc.merge.rdx Phi after the main vector body. Ensure that we carry
  // over the incoming values correctly.
  auto *EpiResumePhi = cast<PHINode>(State.get(EpiResumePhiR, true));
  EpiResumePhi->setIncomingValueForBlock(
      BypassBlock, MainResumePhi->getIncomingValueForBlock(BypassBlock));
}

DenseMap<const SCEV *, Value *> LoopVectorizationPlanner::executePlan(
    ElementCount BestVF, unsigned BestUF, VPlan &BestVPlan,
    InnerLoopVectorizer &ILV, DominatorTree *DT, bool VectorizingEpilogue) {
  assert(BestVPlan.hasVF(BestVF) &&
         "Trying to execute plan with unsupported VF");
  assert(BestVPlan.hasUF(BestUF) &&
         "Trying to execute plan with unsupported UF");
  if (BestVPlan.hasEarlyExit())
    ++LoopsEarlyExitVectorized;
  // TODO: Move to VPlan transform stage once the transition to the VPlan-based
  // cost model is complete for better cost estimates.
  VPlanTransforms::runPass(VPlanTransforms::unrollByUF, BestVPlan, BestUF);
  VPlanTransforms::runPass(VPlanTransforms::materializeBuildVectors, BestVPlan);
  VPlanTransforms::runPass(VPlanTransforms::materializeBroadcasts, BestVPlan);
  VPlanTransforms::runPass(VPlanTransforms::replicateByVF, BestVPlan, BestVF);
  bool HasBranchWeights =
      hasBranchWeightMD(*OrigLoop->getLoopLatch()->getTerminator());
  if (HasBranchWeights) {
    std::optional<unsigned> VScale = CM.getVScaleForTuning();
    VPlanTransforms::runPass(VPlanTransforms::addBranchWeightToMiddleTerminator,
                             BestVPlan, BestVF, VScale);
  }

  // Checks are the same for all VPlans, added to BestVPlan only for
  // compactness.
  attachRuntimeChecks(BestVPlan, ILV.RTChecks, HasBranchWeights);

  // Retrieving VectorPH now when it's easier while VPlan still has Regions.
  VPBasicBlock *VectorPH = cast<VPBasicBlock>(BestVPlan.getVectorPreheader());

  VPlanTransforms::optimizeForVFAndUF(BestVPlan, BestVF, BestUF, PSE);
  VPlanTransforms::simplifyRecipes(BestVPlan);
  VPlanTransforms::removeBranchOnConst(BestVPlan);
  VPlanTransforms::narrowInterleaveGroups(
      BestVPlan, BestVF,
      TTI.getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector));
  VPlanTransforms::removeDeadRecipes(BestVPlan);

  VPlanTransforms::convertToConcreteRecipes(BestVPlan);
  // Regions are dissolved after optimizing for VF and UF, which completely
  // removes unneeded loop regions first.
  VPlanTransforms::dissolveLoopRegions(BestVPlan);
  // Canonicalize EVL loops after regions are dissolved.
  VPlanTransforms::canonicalizeEVLLoops(BestVPlan);
  VPlanTransforms::materializeBackedgeTakenCount(BestVPlan, VectorPH);
  VPlanTransforms::materializeVectorTripCount(
      BestVPlan, VectorPH, CM.foldTailByMasking(),
      CM.requiresScalarEpilogue(BestVF.isVector()));
  VPlanTransforms::materializeVFAndVFxUF(BestVPlan, VectorPH, BestVF);
  VPlanTransforms::simplifyRecipes(BestVPlan);

  // 0. Generate SCEV-dependent code in the entry, including TripCount, before
  // making any changes to the CFG.
  DenseMap<const SCEV *, Value *> ExpandedSCEVs =
      VPlanTransforms::expandSCEVs(BestVPlan, *PSE.getSE());
  if (!ILV.getTripCount())
    ILV.setTripCount(BestVPlan.getTripCount()->getLiveInIRValue());
  else
    assert(VectorizingEpilogue && "should only re-use the existing trip "
                                  "count during epilogue vectorization");

  // Perform the actual loop transformation.
  VPTransformState State(&TTI, BestVF, LI, DT, ILV.AC, ILV.Builder, &BestVPlan,
                         OrigLoop->getParentLoop(),
                         Legal->getWidestInductionType());

#ifdef EXPENSIVE_CHECKS
  assert(DT->verify(DominatorTree::VerificationLevel::Fast));
#endif

  // 1. Set up the skeleton for vectorization, including vector pre-header and
  // middle block. The vector loop is created during VPlan execution.
  BasicBlock *EntryBB =
      cast<VPIRBasicBlock>(BestVPlan.getEntry())->getIRBasicBlock();
  State.CFG.PrevBB = ILV.createVectorizedLoopSkeleton();
  replaceVPBBWithIRVPBB(BestVPlan.getScalarPreheader(),
                        State.CFG.PrevBB->getSingleSuccessor());
  VPlanTransforms::removeDeadRecipes(BestVPlan);

  assert(verifyVPlanIsValid(BestVPlan, true /*VerifyLate*/) &&
         "final VPlan is invalid");

  // After vectorization, the exit blocks of the original loop will have
  // additional predecessors. Invalidate SCEVs for the exit phis in case SE
  // looked through single-entry phis.
  ScalarEvolution &SE = *PSE.getSE();
  for (VPIRBasicBlock *Exit : BestVPlan.getExitBlocks()) {
    if (Exit->getNumPredecessors() == 0)
      continue;
    for (VPRecipeBase &PhiR : Exit->phis())
      SE.forgetLcssaPhiWithNewPredecessor(
          OrigLoop, cast<PHINode>(&cast<VPIRPhi>(PhiR).getInstruction()));
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

  // Move check blocks to their final position.
  // TODO: Move as part of VPIRBB execute and update impacted tests.
  if (BasicBlock *MemCheckBlock = ILV.RTChecks.getMemRuntimeChecks().second)
    MemCheckBlock->moveAfter(EntryBB);
  if (BasicBlock *SCEVCheckBlock = ILV.RTChecks.getSCEVChecks().second)
    SCEVCheckBlock->moveAfter(EntryBB);

  BestVPlan.execute(&State);

  // 2.5 When vectorizing the epilogue, fix reduction resume values from the
  // additional bypass block.
  if (VectorizingEpilogue) {
    assert(!BestVPlan.hasEarlyExit() &&
           "Epilogue vectorisation not yet supported with early exits");
    BasicBlock *PH = OrigLoop->getLoopPreheader();
    BasicBlock *BypassBlock = ILV.getAdditionalBypassBlock();
    for (auto *Pred : predecessors(PH)) {
      for (PHINode &Phi : PH->phis()) {
        if (Phi.getBasicBlockIndex(Pred) != -1)
          continue;
        Phi.addIncoming(Phi.getIncomingValueForBlock(BypassBlock), Pred);
      }
    }
    VPBasicBlock *ScalarPH = BestVPlan.getScalarPreheader();
    if (ScalarPH->getNumPredecessors() > 0) {
      // If ScalarPH has predecessors, we may need to update its reduction
      // resume values.
      for (VPRecipeBase &R : ScalarPH->phis()) {
        fixReductionScalarResumeWhenVectorizingEpilog(cast<VPPhi>(&R), State,
                                                      BypassBlock);
      }
    }
  }

  // 2.6. Maintain Loop Hints
  // Keep all loop hints from the original loop on the vector loop (we'll
  // replace the vectorizer-specific hints below).
  VPBasicBlock *HeaderVPBB = vputils::getFirstLoopHeader(BestVPlan, State.VPDT);
  if (HeaderVPBB) {
    MDNode *OrigLoopID = OrigLoop->getLoopID();

    std::optional<MDNode *> VectorizedLoopID =
        makeFollowupLoopID(OrigLoopID, {LLVMLoopVectorizeFollowupAll,
                                        LLVMLoopVectorizeFollowupVectorized});

    Loop *L = LI->getLoopFor(State.CFG.VPBB2IRBB[HeaderVPBB]);
    if (VectorizedLoopID) {
      L->setLoopID(*VectorizedLoopID);
    } else {
      // Keep all loop hints from the original loop on the vector loop (we'll
      // replace the vectorizer-specific hints below).
      if (MDNode *LID = OrigLoop->getLoopID())
        L->setLoopID(LID);

      LoopVectorizeHints Hints(L, true, *ORE);
      Hints.setAlreadyVectorized();

      // Check if it's EVL-vectorized and mark the corresponding metadata.
      bool IsEVLVectorized =
          llvm::any_of(*HeaderVPBB, [](const VPRecipeBase &Recipe) {
            // Looking for the ExplictVectorLength VPInstruction.
            if (const auto *VI = dyn_cast<VPInstruction>(&Recipe))
              return VI->getOpcode() == VPInstruction::ExplicitVectorLength;
            return false;
          });
      if (IsEVLVectorized) {
        LLVMContext &Context = L->getHeader()->getContext();
        MDNode *LoopID = L->getLoopID();
        auto *IsEVLVectorizedMD = MDNode::get(
            Context,
            {MDString::get(Context, "llvm.loop.isvectorized.tailfoldingstyle"),
             MDString::get(Context, "evl")});
        MDNode *NewLoopID = makePostTransformationMetadata(Context, LoopID, {},
                                                           {IsEVLVectorizedMD});
        L->setLoopID(NewLoopID);
      }
    }
    TargetTransformInfo::UnrollingPreferences UP;
    TTI.getUnrollingPreferences(L, *PSE.getSE(), UP, ORE);
    if (!UP.UnrollVectorizedLoop || VectorizingEpilogue)
      addRuntimeUnrollDisableMetaData(L);
  }

  // 3. Fix the vectorized code: take care of header phi's, live-outs,
  //    predication, updating analyses.
  ILV.fixVectorizedLoop(State);

  ILV.printDebugTracesAtEnd();

  return ExpandedSCEVs;
}

//===--------------------------------------------------------------------===//
// EpilogueVectorizerMainLoop
//===--------------------------------------------------------------------===//

/// This function is partially responsible for generating the control flow
/// depicted in https://llvm.org/docs/Vectorizers.html#epilogue-vectorization.
BasicBlock *EpilogueVectorizerMainLoop::createVectorizedLoopSkeleton() {
  BasicBlock *ScalarPH = createScalarPreheader("");

  // Generate the code to check the minimum iteration count of the vector
  // epilogue (see below).
  EPI.EpilogueIterationCountCheck = emitIterationCountCheck(ScalarPH, true);
  EPI.EpilogueIterationCountCheck->setName("iter.check");

  // Generate the iteration count check for the main loop, *after* the check
  // for the epilogue loop, so that the path-length is shorter for the case
  // that goes directly through the vector epilogue. The longer-path length for
  // the main loop is compensated for, by the gain from vectorizing the larger
  // trip count. Note: the branch will get updated later on when we vectorize
  // the epilogue.
  EPI.MainLoopIterationCountCheck = emitIterationCountCheck(ScalarPH, false);

  return LoopVectorPreHeader;
}

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

BasicBlock *
EpilogueVectorizerMainLoop::emitIterationCountCheck(BasicBlock *Bypass,
                                                    bool ForEpilogue) {
  assert(Bypass && "Expected valid bypass basic block.");
  Value *Count = getTripCount();
  MinProfitableTripCount = ElementCount::getFixed(0);
  Value *CheckMinIters =
      createIterationCountCheck(ForEpilogue ? EPI.EpilogueVF : EPI.MainLoopVF,
                                ForEpilogue ? EPI.EpilogueUF : EPI.MainLoopUF);

  BasicBlock *const TCCheckBlock = LoopVectorPreHeader;
  if (!ForEpilogue)
    TCCheckBlock->setName("vector.main.loop.iter.check");

  // Create new preheader for vector loop.
  LoopVectorPreHeader = SplitBlock(TCCheckBlock, TCCheckBlock->getTerminator(),
                                   static_cast<DominatorTree *>(nullptr), LI,
                                   nullptr, "vector.ph");
  if (ForEpilogue) {
    // Save the trip count so we don't have to regenerate it in the
    // vec.epilog.iter.check. This is safe to do because the trip count
    // generated here dominates the vector epilog iter check.
    EPI.TripCount = Count;
  } else {
    VectorPHVPBB = replaceVPBBWithIRVPBB(VectorPHVPBB, LoopVectorPreHeader);
  }

  BranchInst &BI =
      *BranchInst::Create(Bypass, LoopVectorPreHeader, CheckMinIters);
  if (hasBranchWeightMD(*OrigLoop->getLoopLatch()->getTerminator()))
    setBranchWeights(BI, MinItersBypassWeights, /*IsExpected=*/false);
  ReplaceInstWithInst(TCCheckBlock->getTerminator(), &BI);

  // When vectorizing the main loop, its trip-count check is placed in a new
  // block, whereas the overall trip-count check is placed in the VPlan entry
  // block. When vectorizing the epilogue loop, its trip-count check is placed
  // in the VPlan entry block.
  if (!ForEpilogue)
    introduceCheckBlockInVPlan(TCCheckBlock);
  return TCCheckBlock;
}

//===--------------------------------------------------------------------===//
// EpilogueVectorizerEpilogueLoop
//===--------------------------------------------------------------------===//

/// This function is partially responsible for generating the control flow
/// depicted in https://llvm.org/docs/Vectorizers.html#epilogue-vectorization.
BasicBlock *EpilogueVectorizerEpilogueLoop::createVectorizedLoopSkeleton() {
  BasicBlock *ScalarPH = createScalarPreheader("vec.epilog.");

  // Now, compare the remaining count and if there aren't enough iterations to
  // execute the vectorized epilogue skip to the scalar part.
  LoopVectorPreHeader->setName("vec.epilog.ph");
  BasicBlock *VecEpilogueIterationCountCheck =
      SplitBlock(LoopVectorPreHeader, LoopVectorPreHeader->begin(), DT, LI,
                 nullptr, "vec.epilog.iter.check", true);
  VectorPHVPBB = replaceVPBBWithIRVPBB(VectorPHVPBB, LoopVectorPreHeader);

  emitMinimumVectorEpilogueIterCountCheck(ScalarPH,
                                          VecEpilogueIterationCountCheck);
  AdditionalBypassBlock = VecEpilogueIterationCountCheck;

  // Adjust the control flow taking the state info from the main loop
  // vectorization into account.
  assert(EPI.MainLoopIterationCountCheck && EPI.EpilogueIterationCountCheck &&
         "expected this to be saved from the previous pass.");
  EPI.MainLoopIterationCountCheck->getTerminator()->replaceUsesOfWith(
      VecEpilogueIterationCountCheck, LoopVectorPreHeader);

  EPI.EpilogueIterationCountCheck->getTerminator()->replaceUsesOfWith(
      VecEpilogueIterationCountCheck, ScalarPH);

  // Adjust the terminators of runtime check blocks and phis using them.
  BasicBlock *SCEVCheckBlock = RTChecks.getSCEVChecks().second;
  BasicBlock *MemCheckBlock = RTChecks.getMemRuntimeChecks().second;
  if (SCEVCheckBlock)
    SCEVCheckBlock->getTerminator()->replaceUsesOfWith(
        VecEpilogueIterationCountCheck, ScalarPH);
  if (MemCheckBlock)
    MemCheckBlock->getTerminator()->replaceUsesOfWith(
        VecEpilogueIterationCountCheck, ScalarPH);

  DT->changeImmediateDominator(ScalarPH, EPI.EpilogueIterationCountCheck);

  // The vec.epilog.iter.check block may contain Phi nodes from inductions or
  // reductions which merge control-flow from the latch block and the middle
  // block. Update the incoming values here and move the Phi into the preheader.
  SmallVector<PHINode *, 4> PhisInBlock(
      llvm::make_pointer_range(VecEpilogueIterationCountCheck->phis()));

  for (PHINode *Phi : PhisInBlock) {
    Phi->moveBefore(LoopVectorPreHeader->getFirstNonPHIIt());
    Phi->replaceIncomingBlockWith(
        VecEpilogueIterationCountCheck->getSinglePredecessor(),
        VecEpilogueIterationCountCheck);

    // If the phi doesn't have an incoming value from the
    // EpilogueIterationCountCheck, we are done. Otherwise remove the incoming
    // value and also those from other check blocks. This is needed for
    // reduction phis only.
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

  return LoopVectorPreHeader;
}

BasicBlock *
EpilogueVectorizerEpilogueLoop::emitMinimumVectorEpilogueIterCountCheck(
    BasicBlock *Bypass, BasicBlock *Insert) {

  assert(EPI.TripCount &&
         "Expected trip count to have been saved in the first pass.");
  Value *TC = EPI.TripCount;
  IRBuilder<> Builder(Insert->getTerminator());
  Value *Count = Builder.CreateSub(TC, EPI.VectorTripCount, "n.vec.remaining");

  // Generate code to check if the loop's trip count is less than VF * UF of the
  // vector epilogue loop.
  auto P = Cost->requiresScalarEpilogue(EPI.EpilogueVF.isVector())
               ? ICmpInst::ICMP_ULE
               : ICmpInst::ICMP_ULT;

  Value *CheckMinIters =
      Builder.CreateICmp(P, Count,
                         createStepForVF(Builder, Count->getType(),
                                         EPI.EpilogueVF, EPI.EpilogueUF),
                         "min.epilog.iters.check");

  BranchInst &BI =
      *BranchInst::Create(Bypass, LoopVectorPreHeader, CheckMinIters);
  if (hasBranchWeightMD(*OrigLoop->getLoopLatch()->getTerminator())) {
    auto VScale = Cost->getVScaleForTuning();
    unsigned MainLoopStep =
        estimateElementCount(EPI.MainLoopVF * EPI.MainLoopUF, VScale);
    unsigned EpilogueLoopStep =
        estimateElementCount(EPI.EpilogueVF * EPI.EpilogueUF, VScale);
    // We assume the remaining `Count` is equally distributed in
    // [0, MainLoopStep)
    // So the probability for `Count < EpilogueLoopStep` should be
    // min(MainLoopStep, EpilogueLoopStep) / MainLoopStep
    unsigned EstimatedSkipCount = std::min(MainLoopStep, EpilogueLoopStep);
    const uint32_t Weights[] = {EstimatedSkipCount,
                                MainLoopStep - EstimatedSkipCount};
    setBranchWeights(BI, Weights, /*IsExpected=*/false);
  }
  ReplaceInstWithInst(Insert->getTerminator(), &BI);

  // A new entry block has been created for the epilogue VPlan. Hook it in, as
  // otherwise we would try to modify the entry to the main vector loop.
  VPIRBasicBlock *NewEntry = Plan.createVPIRBasicBlock(Insert);
  VPBasicBlock *OldEntry = Plan.getEntry();
  VPBlockUtils::reassociateBlocks(OldEntry, NewEntry);
  Plan.setEntry(NewEntry);
  // OldEntry is now dead and will be cleaned up when the plan gets destroyed.

  return Insert;
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

VPWidenMemoryRecipe *
VPRecipeBuilder::tryToWidenMemory(Instruction *I, ArrayRef<VPValue *> Operands,
                                  VFRange &Range) {
  assert((isa<LoadInst>(I) || isa<StoreInst>(I)) &&
         "Must be called with either a load or store");

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

  VPValue *Mask = nullptr;
  if (Legal->isMaskRequired(I))
    Mask = getBlockInMask(Builder.getInsertBlock());

  // Determine if the pointer operand of the access is either consecutive or
  // reverse consecutive.
  LoopVectorizationCostModel::InstWidening Decision =
      CM.getWideningDecision(I, Range.Start);
  bool Reverse = Decision == LoopVectorizationCostModel::CM_Widen_Reverse;
  bool Consecutive =
      Reverse || Decision == LoopVectorizationCostModel::CM_Widen;

  VPValue *Ptr = isa<LoadInst>(I) ? Operands[0] : Operands[1];
  if (Consecutive) {
    auto *GEP = dyn_cast<GetElementPtrInst>(
        Ptr->getUnderlyingValue()->stripPointerCasts());
    VPSingleDefRecipe *VectorPtr;
    if (Reverse) {
      // When folding the tail, we may compute an address that we don't in the
      // original scalar loop and it may not be inbounds. Drop Inbounds in that
      // case.
      GEPNoWrapFlags Flags =
          (CM.foldTailByMasking() || !GEP || !GEP->isInBounds())
              ? GEPNoWrapFlags::none()
              : GEPNoWrapFlags::inBounds();
      VectorPtr =
          new VPVectorEndPointerRecipe(Ptr, &Plan.getVF(), getLoadStoreType(I),
                                       /*Stride*/ -1, Flags, I->getDebugLoc());
    } else {
      VectorPtr = new VPVectorPointerRecipe(Ptr, getLoadStoreType(I),
                                            GEP ? GEP->getNoWrapFlags()
                                                : GEPNoWrapFlags::none(),
                                            I->getDebugLoc());
    }
    Builder.insert(VectorPtr);
    Ptr = VectorPtr;
  }
  if (LoadInst *Load = dyn_cast<LoadInst>(I))
    return new VPWidenLoadRecipe(*Load, Ptr, Mask, Consecutive, Reverse,
                                 VPIRMetadata(*Load, LVer), I->getDebugLoc());

  StoreInst *Store = cast<StoreInst>(I);
  return new VPWidenStoreRecipe(*Store, Ptr, Operands[0], Mask, Consecutive,
                                Reverse, VPIRMetadata(*Store, LVer),
                                I->getDebugLoc());
}

/// Creates a VPWidenIntOrFpInductionRecpipe for \p Phi. If needed, it will also
/// insert a recipe to expand the step for the induction recipe.
static VPWidenIntOrFpInductionRecipe *
createWidenInductionRecipes(PHINode *Phi, Instruction *PhiOrTrunc,
                            VPValue *Start, const InductionDescriptor &IndDesc,
                            VPlan &Plan, ScalarEvolution &SE, Loop &OrigLoop) {
  assert(IndDesc.getStartValue() ==
         Phi->getIncomingValueForBlock(OrigLoop.getLoopPreheader()));
  assert(SE.isLoopInvariant(IndDesc.getStep(), &OrigLoop) &&
         "step must be loop invariant");

  VPValue *Step =
      vputils::getOrCreateVPValueForSCEVExpr(Plan, IndDesc.getStep());
  if (auto *TruncI = dyn_cast<TruncInst>(PhiOrTrunc)) {
    return new VPWidenIntOrFpInductionRecipe(Phi, Start, Step, &Plan.getVF(),
                                             IndDesc, TruncI,
                                             TruncI->getDebugLoc());
  }
  assert(isa<PHINode>(PhiOrTrunc) && "must be a phi node here");
  return new VPWidenIntOrFpInductionRecipe(Phi, Start, Step, &Plan.getVF(),
                                           IndDesc, Phi->getDebugLoc());
}

VPHeaderPHIRecipe *VPRecipeBuilder::tryToOptimizeInductionPHI(
    PHINode *Phi, ArrayRef<VPValue *> Operands, VFRange &Range) {

  // Check if this is an integer or fp induction. If so, build the recipe that
  // produces its scalar and vector values.
  if (auto *II = Legal->getIntOrFpInductionDescriptor(Phi))
    return createWidenInductionRecipes(Phi, Phi, Operands[0], *II, Plan,
                                       *PSE.getSE(), *OrigLoop);

  // Check if this is pointer induction. If so, build the recipe for it.
  if (auto *II = Legal->getPointerInductionDescriptor(Phi)) {
    VPValue *Step = vputils::getOrCreateVPValueForSCEVExpr(Plan, II->getStep());
    return new VPWidenPointerInductionRecipe(
        Phi, Operands[0], Step, &Plan.getVFxUF(), *II,
        LoopVectorizationPlanner::getDecisionAndClampRange(
            [&](ElementCount VF) {
              return CM.isScalarAfterVectorization(Phi, VF);
            },
            Range),
        Phi->getDebugLoc());
  }
  return nullptr;
}

VPWidenIntOrFpInductionRecipe *VPRecipeBuilder::tryToOptimizeInductionTruncate(
    TruncInst *I, ArrayRef<VPValue *> Operands, VFRange &Range) {
  // Optimize the special case where the source is a constant integer
  // induction variable. Notice that we can only optimize the 'trunc' case
  // because (a) FP conversions lose precision, (b) sext/zext may wrap, and
  // (c) other casts depend on pointer size.

  // Determine whether \p K is a truncation based on an induction variable that
  // can be optimized.
  auto IsOptimizableIVTruncate =
      [&](Instruction *K) -> std::function<bool(ElementCount)> {
    return [=](ElementCount VF) -> bool {
      return CM.isOptimizableIVTruncate(K, VF);
    };
  };

  if (LoopVectorizationPlanner::getDecisionAndClampRange(
          IsOptimizableIVTruncate(I), Range)) {

    auto *Phi = cast<PHINode>(I->getOperand(0));
    const InductionDescriptor &II = *Legal->getIntOrFpInductionDescriptor(Phi);
    VPValue *Start = Plan.getOrAddLiveIn(II.getStartValue());
    return createWidenInductionRecipes(Phi, I, Start, II, Plan, *PSE.getSE(),
                                       *OrigLoop);
  }
  return nullptr;
}

VPSingleDefRecipe *VPRecipeBuilder::tryToWidenCall(CallInst *CI,
                                                   ArrayRef<VPValue *> Operands,
                                                   VFRange &Range) {
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

  SmallVector<VPValue *, 4> Ops(Operands.take_front(CI->arg_size()));

  // Is it beneficial to perform intrinsic call compared to lib call?
  bool ShouldUseVectorIntrinsic =
      ID && LoopVectorizationPlanner::getDecisionAndClampRange(
                [&](ElementCount VF) -> bool {
                  return CM.getCallWideningDecision(CI, VF).Kind ==
                         LoopVectorizationCostModel::CM_IntrinsicCall;
                },
                Range);
  if (ShouldUseVectorIntrinsic)
    return new VPWidenIntrinsicRecipe(*CI, ID, Ops, CI->getType(),
                                      CI->getDebugLoc());

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
      //   1) The block needs to be predicated, either due to a conditional
      //      in the scalar loop or use of an active lane mask with
      //      tail-folding, and we use the appropriate mask for the block.
      //   2) No mask is required for the block, but the only available
      //      vector variant at this VF requires a mask, so we synthesize an
      //      all-true mask.
      VPValue *Mask = nullptr;
      if (Legal->isMaskRequired(CI))
        Mask = getBlockInMask(Builder.getInsertBlock());
      else
        Mask = Plan.getOrAddLiveIn(
            ConstantInt::getTrue(IntegerType::getInt1Ty(CI->getContext())));

      Ops.insert(Ops.begin() + *MaskPos, Mask);
    }

    Ops.push_back(Operands.back());
    return new VPWidenCallRecipe(CI, Variant, Ops, CI->getDebugLoc());
  }

  return nullptr;
}

bool VPRecipeBuilder::shouldWiden(Instruction *I, VFRange &Range) const {
  assert(!isa<BranchInst>(I) && !isa<PHINode>(I) && !isa<LoadInst>(I) &&
         !isa<StoreInst>(I) && "Instruction should have been handled earlier");
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

VPWidenRecipe *VPRecipeBuilder::tryToWiden(Instruction *I,
                                           ArrayRef<VPValue *> Operands) {
  switch (I->getOpcode()) {
  default:
    return nullptr;
  case Instruction::SDiv:
  case Instruction::UDiv:
  case Instruction::SRem:
  case Instruction::URem: {
    // If not provably safe, use a select to form a safe divisor before widening the
    // div/rem operation itself.  Otherwise fall through to general handling below.
    if (CM.isPredicatedInst(I)) {
      SmallVector<VPValue *> Ops(Operands);
      VPValue *Mask = getBlockInMask(Builder.getInsertBlock());
      VPValue *One =
          Plan.getOrAddLiveIn(ConstantInt::get(I->getType(), 1u, false));
      auto *SafeRHS = Builder.createSelect(Mask, Ops[1], One, I->getDebugLoc());
      Ops[1] = SafeRHS;
      return new VPWidenRecipe(*I, Ops);
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
  case Instruction::Freeze: {
    SmallVector<VPValue *> NewOps(Operands);
    if (Instruction::isBinaryOp(I->getOpcode())) {
      // The legacy cost model uses SCEV to check if some of the operands are
      // constants. To match the legacy cost model's behavior, use SCEV to try
      // to replace operands with constants.
      ScalarEvolution &SE = *PSE.getSE();
      auto GetConstantViaSCEV = [this, &SE](VPValue *Op) {
        if (!Op->isLiveIn())
          return Op;
        Value *V = Op->getUnderlyingValue();
        if (isa<Constant>(V) || !SE.isSCEVable(V->getType()))
          return Op;
        auto *C = dyn_cast<SCEVConstant>(SE.getSCEV(V));
        if (!C)
          return Op;
        return Plan.getOrAddLiveIn(C->getValue());
      };
      // For Mul, the legacy cost model checks both operands.
      if (I->getOpcode() == Instruction::Mul)
        NewOps[0] = GetConstantViaSCEV(NewOps[0]);
      // For other binops, the legacy cost model only checks the second operand.
      NewOps[1] = GetConstantViaSCEV(NewOps[1]);
    }
    return new VPWidenRecipe(*I, NewOps);
  }
  case Instruction::ExtractValue: {
    SmallVector<VPValue *> NewOps(Operands);
    Type *I32Ty = IntegerType::getInt32Ty(I->getContext());
    auto *EVI = cast<ExtractValueInst>(I);
    assert(EVI->getNumIndices() == 1 && "Expected one extractvalue index");
    unsigned Idx = EVI->getIndices()[0];
    NewOps.push_back(Plan.getOrAddLiveIn(ConstantInt::get(I32Ty, Idx, false)));
    return new VPWidenRecipe(*I, NewOps);
  }
  };
}

VPHistogramRecipe *
VPRecipeBuilder::tryToWidenHistogram(const HistogramInfo *HI,
                                     ArrayRef<VPValue *> Operands) {
  // FIXME: Support other operations.
  unsigned Opcode = HI->Update->getOpcode();
  assert((Opcode == Instruction::Add || Opcode == Instruction::Sub) &&
         "Histogram update operation must be an Add or Sub");

  SmallVector<VPValue *, 3> HGramOps;
  // Bucket address.
  HGramOps.push_back(Operands[1]);
  // Increment value.
  HGramOps.push_back(getVPValueOrAddLiveIn(HI->Update->getOperand(1)));

  // In case of predicated execution (due to tail-folding, or conditional
  // execution, or both), pass the relevant mask.
  if (Legal->isMaskRequired(HI->Store))
    HGramOps.push_back(getBlockInMask(Builder.getInsertBlock()));

  return new VPHistogramRecipe(Opcode, HGramOps, HI->Store->getDebugLoc());
}

VPReplicateRecipe *
VPRecipeBuilder::handleReplication(Instruction *I, ArrayRef<VPValue *> Operands,
                                   VFRange &Range) {
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
    BlockInMask = getBlockInMask(Builder.getInsertBlock());
  }

  // Note that there is some custom logic to mark some intrinsics as uniform
  // manually above for scalable vectors, which this assert needs to account for
  // as well.
  assert((Range.Start.isScalar() || !IsUniform || !IsPredicated ||
          (Range.Start.isScalable() && isa<IntrinsicInst>(I))) &&
         "Should not predicate a uniform recipe");
  auto *Recipe = new VPReplicateRecipe(I, Operands, IsUniform, BlockInMask,
                                       VPIRMetadata(*I, LVer));
  return Recipe;
}

/// Find all possible partial reductions in the loop and track all of those that
/// are valid so recipes can be formed later.
void VPRecipeBuilder::collectScaledReductions(VFRange &Range) {
  // Find all possible partial reductions.
  SmallVector<std::pair<PartialReductionChain, unsigned>>
      PartialReductionChains;
  for (const auto &[Phi, RdxDesc] : Legal->getReductionVars()) {
    getScaledReductions(Phi, RdxDesc.getLoopExitInstr(), Range,
                        PartialReductionChains);
  }

  // A partial reduction is invalid if any of its extends are used by
  // something that isn't another partial reduction. This is because the
  // extends are intended to be lowered along with the reduction itself.

  // Build up a set of partial reduction ops for efficient use checking.
  SmallPtrSet<User *, 4> PartialReductionOps;
  for (const auto &[PartialRdx, _] : PartialReductionChains)
    PartialReductionOps.insert(PartialRdx.ExtendUser);

  auto ExtendIsOnlyUsedByPartialReductions =
      [&PartialReductionOps](Instruction *Extend) {
        return all_of(Extend->users(), [&](const User *U) {
          return PartialReductionOps.contains(U);
        });
      };

  // Check if each use of a chain's two extends is a partial reduction
  // and only add those that don't have non-partial reduction users.
  for (auto Pair : PartialReductionChains) {
    PartialReductionChain Chain = Pair.first;
    if (ExtendIsOnlyUsedByPartialReductions(Chain.ExtendA) &&
        (!Chain.ExtendB || ExtendIsOnlyUsedByPartialReductions(Chain.ExtendB)))
      ScaledReductionMap.try_emplace(Chain.Reduction, Pair.second);
  }
}

bool VPRecipeBuilder::getScaledReductions(
    Instruction *PHI, Instruction *RdxExitInstr, VFRange &Range,
    SmallVectorImpl<std::pair<PartialReductionChain, unsigned>> &Chains) {
  if (!CM.TheLoop->contains(RdxExitInstr))
    return false;

  auto *Update = dyn_cast<BinaryOperator>(RdxExitInstr);
  if (!Update)
    return false;

  Value *Op = Update->getOperand(0);
  Value *PhiOp = Update->getOperand(1);
  if (Op == PHI)
    std::swap(Op, PhiOp);

  // Try and get a scaled reduction from the first non-phi operand.
  // If one is found, we use the discovered reduction instruction in
  // place of the accumulator for costing.
  if (auto *OpInst = dyn_cast<Instruction>(Op)) {
    if (getScaledReductions(PHI, OpInst, Range, Chains)) {
      PHI = Chains.rbegin()->first.Reduction;

      Op = Update->getOperand(0);
      PhiOp = Update->getOperand(1);
      if (Op == PHI)
        std::swap(Op, PhiOp);
    }
  }
  if (PhiOp != PHI)
    return false;

  using namespace llvm::PatternMatch;

  // If the update is a binary operator, check both of its operands to see if
  // they are extends. Otherwise, see if the update comes directly from an
  // extend.
  Instruction *Exts[2] = {nullptr};
  BinaryOperator *ExtendUser = dyn_cast<BinaryOperator>(Op);
  std::optional<unsigned> BinOpc;
  Type *ExtOpTypes[2] = {nullptr};

  auto CollectExtInfo = [this, &Exts,
                         &ExtOpTypes](SmallVectorImpl<Value *> &Ops) -> bool {
    unsigned I = 0;
    for (Value *OpI : Ops) {
      Value *ExtOp;
      if (!match(OpI, m_ZExtOrSExt(m_Value(ExtOp))))
        return false;
      Exts[I] = cast<Instruction>(OpI);

      // TODO: We should be able to support live-ins.
      if (!CM.TheLoop->contains(Exts[I]))
        return false;

      ExtOpTypes[I] = ExtOp->getType();
      I++;
    }
    return true;
  };

  if (ExtendUser) {
    if (!ExtendUser->hasOneUse())
      return false;

    // Use the side-effect of match to replace BinOp only if the pattern is
    // matched, we don't care at this point whether it actually matched.
    match(ExtendUser, m_Neg(m_BinOp(ExtendUser)));

    SmallVector<Value *> Ops(ExtendUser->operands());
    if (!CollectExtInfo(Ops))
      return false;

    BinOpc = std::make_optional(ExtendUser->getOpcode());
  } else if (match(Update, m_Add(m_Value(), m_Value()))) {
    // We already know the operands for Update are Op and PhiOp.
    SmallVector<Value *> Ops({Op});
    if (!CollectExtInfo(Ops))
      return false;

    ExtendUser = Update;
    BinOpc = std::nullopt;
  } else
    return false;

  TTI::PartialReductionExtendKind OpAExtend =
      TTI::getPartialReductionExtendKind(Exts[0]);
  TTI::PartialReductionExtendKind OpBExtend =
      Exts[1] ? TTI::getPartialReductionExtendKind(Exts[1]) : TTI::PR_None;
  PartialReductionChain Chain(RdxExitInstr, Exts[0], Exts[1], ExtendUser);

  TypeSize PHISize = PHI->getType()->getPrimitiveSizeInBits();
  TypeSize ASize = ExtOpTypes[0]->getPrimitiveSizeInBits();
  if (!PHISize.hasKnownScalarFactor(ASize))
    return false;
  unsigned TargetScaleFactor = PHISize.getKnownScalarFactor(ASize);

  if (LoopVectorizationPlanner::getDecisionAndClampRange(
          [&](ElementCount VF) {
            InstructionCost Cost = TTI->getPartialReductionCost(
                Update->getOpcode(), ExtOpTypes[0], ExtOpTypes[1],
                PHI->getType(), VF, OpAExtend, OpBExtend, BinOpc, CM.CostKind);
            return Cost.isValid();
          },
          Range)) {
    Chains.emplace_back(Chain, TargetScaleFactor);
    return true;
  }

  return false;
}

VPRecipeBase *VPRecipeBuilder::tryToCreateWidenRecipe(VPSingleDefRecipe *R,
                                                      VFRange &Range) {
  // First, check for specific widening recipes that deal with inductions, Phi
  // nodes, calls and memory operations.
  VPRecipeBase *Recipe;
  Instruction *Instr = R->getUnderlyingInstr();
  SmallVector<VPValue *, 4> Operands(R->operands());
  if (auto *PhiR = dyn_cast<VPPhi>(R)) {
    VPBasicBlock *Parent = PhiR->getParent();
    [[maybe_unused]] VPRegionBlock *LoopRegionOf =
        Parent->getEnclosingLoopRegion();
    assert(LoopRegionOf && LoopRegionOf->getEntry() == Parent &&
           "Non-header phis should have been handled during predication");
    auto *Phi = cast<PHINode>(R->getUnderlyingInstr());
    assert(Operands.size() == 2 && "Must have 2 operands for header phis");
    if ((Recipe = tryToOptimizeInductionPHI(Phi, Operands, Range)))
      return Recipe;

    VPHeaderPHIRecipe *PhiRecipe = nullptr;
    assert((Legal->isReductionVariable(Phi) ||
            Legal->isFixedOrderRecurrence(Phi)) &&
           "can only widen reductions and fixed-order recurrences here");
    VPValue *StartV = Operands[0];
    if (Legal->isReductionVariable(Phi)) {
      const RecurrenceDescriptor &RdxDesc = Legal->getRecurrenceDescriptor(Phi);
      assert(RdxDesc.getRecurrenceStartValue() ==
             Phi->getIncomingValueForBlock(OrigLoop->getLoopPreheader()));

      // If the PHI is used by a partial reduction, set the scale factor.
      unsigned ScaleFactor =
          getScalingForReduction(RdxDesc.getLoopExitInstr()).value_or(1);
      PhiRecipe = new VPReductionPHIRecipe(
          Phi, RdxDesc.getRecurrenceKind(), *StartV, CM.isInLoopReduction(Phi),
          CM.useOrderedReductions(RdxDesc), ScaleFactor);
    } else {
      // TODO: Currently fixed-order recurrences are modeled as chains of
      // first-order recurrences. If there are no users of the intermediate
      // recurrences in the chain, the fixed order recurrence should be modeled
      // directly, enabling more efficient codegen.
      PhiRecipe = new VPFirstOrderRecurrencePHIRecipe(Phi, *StartV);
    }
    // Add backedge value.
    PhiRecipe->addOperand(Operands[1]);
    return PhiRecipe;
  }
  assert(!R->isPhi() && "only VPPhi nodes expected at this point");

  if (isa<TruncInst>(Instr) && (Recipe = tryToOptimizeInductionTruncate(
                                    cast<TruncInst>(Instr), Operands, Range)))
    return Recipe;

  // All widen recipes below deal only with VF > 1.
  if (LoopVectorizationPlanner::getDecisionAndClampRange(
          [&](ElementCount VF) { return VF.isScalar(); }, Range))
    return nullptr;

  if (auto *CI = dyn_cast<CallInst>(Instr))
    return tryToWidenCall(CI, Operands, Range);

  if (StoreInst *SI = dyn_cast<StoreInst>(Instr))
    if (auto HistInfo = Legal->getHistogramInfo(SI))
      return tryToWidenHistogram(*HistInfo, Operands);

  if (isa<LoadInst>(Instr) || isa<StoreInst>(Instr))
    return tryToWidenMemory(Instr, Operands, Range);

  if (std::optional<unsigned> ScaleFactor = getScalingForReduction(Instr))
    return tryToCreatePartialReduction(Instr, Operands, ScaleFactor.value());

  if (!shouldWiden(Instr, Range))
    return nullptr;

  if (auto *GEP = dyn_cast<GetElementPtrInst>(Instr))
    return new VPWidenGEPRecipe(GEP, Operands);

  if (auto *SI = dyn_cast<SelectInst>(Instr)) {
    return new VPWidenSelectRecipe(*SI, Operands);
  }

  if (auto *CI = dyn_cast<CastInst>(Instr)) {
    return new VPWidenCastRecipe(CI->getOpcode(), Operands[0], CI->getType(),
                                 *CI);
  }

  return tryToWiden(Instr, Operands);
}

VPRecipeBase *
VPRecipeBuilder::tryToCreatePartialReduction(Instruction *Reduction,
                                             ArrayRef<VPValue *> Operands,
                                             unsigned ScaleFactor) {
  assert(Operands.size() == 2 &&
         "Unexpected number of operands for partial reduction");

  VPValue *BinOp = Operands[0];
  VPValue *Accumulator = Operands[1];
  VPRecipeBase *BinOpRecipe = BinOp->getDefiningRecipe();
  if (isa<VPReductionPHIRecipe>(BinOpRecipe) ||
      isa<VPPartialReductionRecipe>(BinOpRecipe))
    std::swap(BinOp, Accumulator);

  unsigned ReductionOpcode = Reduction->getOpcode();
  if (ReductionOpcode == Instruction::Sub) {
    auto *const Zero = ConstantInt::get(Reduction->getType(), 0);
    SmallVector<VPValue *, 2> Ops;
    Ops.push_back(Plan.getOrAddLiveIn(Zero));
    Ops.push_back(BinOp);
    BinOp = new VPWidenRecipe(*Reduction, Ops);
    Builder.insert(BinOp->getDefiningRecipe());
    ReductionOpcode = Instruction::Add;
  }

  VPValue *Cond = nullptr;
  if (CM.blockNeedsPredicationForAnyReason(Reduction->getParent())) {
    assert((ReductionOpcode == Instruction::Add ||
            ReductionOpcode == Instruction::Sub) &&
           "Expected an ADD or SUB operation for predicated partial "
           "reductions (because the neutral element in the mask is zero)!");
    Cond = getBlockInMask(Builder.getInsertBlock());
    VPValue *Zero =
        Plan.getOrAddLiveIn(ConstantInt::get(Reduction->getType(), 0));
    BinOp = Builder.createSelect(Cond, BinOp, Zero, Reduction->getDebugLoc());
  }
  return new VPPartialReductionRecipe(ReductionOpcode, Accumulator, BinOp, Cond,
                                      ScaleFactor, Reduction);
}

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
      getDebugLocFromInstOrOperands(Legal->getPrimaryInduction()), PSE);

  auto MaxVFTimes2 = MaxVF * 2;
  for (ElementCount VF = MinVF; ElementCount::isKnownLT(VF, MaxVFTimes2);) {
    VFRange SubRange = {VF, MaxVFTimes2};
    if (auto Plan = tryToBuildVPlanWithVPRecipes(
            std::unique_ptr<VPlan>(VPlan0->duplicate()), SubRange, &LVer)) {
      bool HasScalarVF = Plan->hasScalarVFOnly();
      // Now optimize the initial VPlan.
      if (!HasScalarVF)
        VPlanTransforms::runPass(VPlanTransforms::truncateToMinimalBitwidths,
                                 *Plan, CM.getMinimalBitwidths());
      VPlanTransforms::runPass(VPlanTransforms::optimize, *Plan);
      // TODO: try to put it close to addActiveLaneMask().
      if (CM.foldTailWithEVL() && !HasScalarVF)
        VPlanTransforms::runPass(VPlanTransforms::addExplicitVectorLength,
                                 *Plan, CM.getMaxSafeElements());
      assert(verifyVPlanIsValid(*Plan) && "VPlan is invalid");
      VPlans.push_back(std::move(Plan));
    }
    VF = SubRange.End;
  }
}

/// Create and return a ResumePhi for \p WideIV, unless it is truncated. If the
/// induction recipe is not canonical, creates a VPDerivedIVRecipe to compute
/// the end value of the induction.
static VPInstruction *addResumePhiRecipeForInduction(
    VPWidenInductionRecipe *WideIV, VPBuilder &VectorPHBuilder,
    VPBuilder &ScalarPHBuilder, VPTypeAnalysis &TypeInfo, VPValue *VectorTC) {
  auto *WideIntOrFp = dyn_cast<VPWidenIntOrFpInductionRecipe>(WideIV);
  // Truncated wide inductions resume from the last lane of their vector value
  // in the last vector iteration which is handled elsewhere.
  if (WideIntOrFp && WideIntOrFp->getTruncInst())
    return nullptr;

  VPValue *Start = WideIV->getStartValue();
  VPValue *Step = WideIV->getStepValue();
  const InductionDescriptor &ID = WideIV->getInductionDescriptor();
  VPValue *EndValue = VectorTC;
  if (!WideIntOrFp || !WideIntOrFp->isCanonical()) {
    EndValue = VectorPHBuilder.createDerivedIV(
        ID.getKind(), dyn_cast_or_null<FPMathOperator>(ID.getInductionBinOp()),
        Start, VectorTC, Step);
  }

  // EndValue is derived from the vector trip count (which has the same type as
  // the widest induction) and thus may be wider than the induction here.
  Type *ScalarTypeOfWideIV = TypeInfo.inferScalarType(WideIV);
  if (ScalarTypeOfWideIV != TypeInfo.inferScalarType(EndValue)) {
    EndValue = VectorPHBuilder.createScalarCast(Instruction::Trunc, EndValue,
                                                ScalarTypeOfWideIV,
                                                WideIV->getDebugLoc());
  }

  auto *ResumePhiRecipe = ScalarPHBuilder.createScalarPhi(
      {EndValue, Start}, WideIV->getDebugLoc(), "bc.resume.val");
  return ResumePhiRecipe;
}

/// Create resume phis in the scalar preheader for first-order recurrences,
/// reductions and inductions, and update the VPIRInstructions wrapping the
/// original phis in the scalar header. End values for inductions are added to
/// \p IVEndValues.
static void addScalarResumePhis(VPRecipeBuilder &Builder, VPlan &Plan,
                                DenseMap<VPValue *, VPValue *> &IVEndValues) {
  VPTypeAnalysis TypeInfo(Plan);
  auto *ScalarPH = Plan.getScalarPreheader();
  auto *MiddleVPBB = cast<VPBasicBlock>(ScalarPH->getPredecessors()[0]);
  VPRegionBlock *VectorRegion = Plan.getVectorLoopRegion();
  VPBuilder VectorPHBuilder(
      cast<VPBasicBlock>(VectorRegion->getSinglePredecessor()));
  VPBuilder MiddleBuilder(MiddleVPBB, MiddleVPBB->getFirstNonPhi());
  VPBuilder ScalarPHBuilder(ScalarPH);
  for (VPRecipeBase &ScalarPhiR : Plan.getScalarHeader()->phis()) {
    auto *ScalarPhiIRI = cast<VPIRPhi>(&ScalarPhiR);

    // TODO: Extract final value from induction recipe initially, optimize to
    // pre-computed end value together in optimizeInductionExitUsers.
    auto *VectorPhiR =
        cast<VPHeaderPHIRecipe>(Builder.getRecipe(&ScalarPhiIRI->getIRPhi()));
    if (auto *WideIVR = dyn_cast<VPWidenInductionRecipe>(VectorPhiR)) {
      if (VPInstruction *ResumePhi = addResumePhiRecipeForInduction(
              WideIVR, VectorPHBuilder, ScalarPHBuilder, TypeInfo,
              &Plan.getVectorTripCount())) {
        assert(isa<VPPhi>(ResumePhi) && "Expected a phi");
        IVEndValues[WideIVR] = ResumePhi->getOperand(0);
        ScalarPhiIRI->addOperand(ResumePhi);
        continue;
      }
      // TODO: Also handle truncated inductions here. Computing end-values
      // separately should be done as VPlan-to-VPlan optimization, after
      // legalizing all resume values to use the last lane from the loop.
      assert(cast<VPWidenIntOrFpInductionRecipe>(VectorPhiR)->getTruncInst() &&
             "should only skip truncated wide inductions");
      continue;
    }

    // The backedge value provides the value to resume coming out of a loop,
    // which for FORs is a vector whose last element needs to be extracted. The
    // start value provides the value if the loop is bypassed.
    bool IsFOR = isa<VPFirstOrderRecurrencePHIRecipe>(VectorPhiR);
    auto *ResumeFromVectorLoop = VectorPhiR->getBackedgeValue();
    assert(VectorRegion->getSingleSuccessor() == Plan.getMiddleBlock() &&
           "Cannot handle loops with uncountable early exits");
    if (IsFOR)
      ResumeFromVectorLoop = MiddleBuilder.createNaryOp(
          VPInstruction::ExtractLastElement, {ResumeFromVectorLoop}, {},
          "vector.recur.extract");
    StringRef Name = IsFOR ? "scalar.recur.init" : "bc.merge.rdx";
    auto *ResumePhiR = ScalarPHBuilder.createScalarPhi(
        {ResumeFromVectorLoop, VectorPhiR->getStartValue()}, {}, Name);
    ScalarPhiIRI->addOperand(ResumePhiR);
  }
}

/// Handle users in the exit block for first order reductions in the original
/// exit block. The penultimate value of recurrences is fed to their LCSSA phi
/// users in the original exit block using the VPIRInstruction wrapping to the
/// LCSSA phi.
static void addExitUsersForFirstOrderRecurrences(VPlan &Plan, VFRange &Range) {
  VPRegionBlock *VectorRegion = Plan.getVectorLoopRegion();
  auto *ScalarPHVPBB = Plan.getScalarPreheader();
  auto *MiddleVPBB = Plan.getMiddleBlock();
  VPBuilder ScalarPHBuilder(ScalarPHVPBB);
  VPBuilder MiddleBuilder(MiddleVPBB, MiddleVPBB->getFirstNonPhi());

  auto IsScalableOne = [](ElementCount VF) -> bool {
    return VF == ElementCount::getScalable(1);
  };

  for (auto &HeaderPhi : VectorRegion->getEntryBasicBlock()->phis()) {
    auto *FOR = dyn_cast<VPFirstOrderRecurrencePHIRecipe>(&HeaderPhi);
    if (!FOR)
      continue;

    assert(VectorRegion->getSingleSuccessor() == Plan.getMiddleBlock() &&
           "Cannot handle loops with uncountable early exits");

    // This is the second phase of vectorizing first-order recurrences, creating
    // extract for users outside the loop. An overview of the transformation is
    // described below. Suppose we have the following loop with some use after
    // the loop of the last a[i-1],
    //
    //   for (int i = 0; i < n; ++i) {
    //     t = a[i - 1];
    //     b[i] = a[i] - t;
    //   }
    //   use t;
    //
    // There is a first-order recurrence on "a". For this loop, the shorthand
    // scalar IR looks like:
    //
    //   scalar.ph:
    //     s.init = a[-1]
    //     br scalar.body
    //
    //   scalar.body:
    //     i = phi [0, scalar.ph], [i+1, scalar.body]
    //     s1 = phi [s.init, scalar.ph], [s2, scalar.body]
    //     s2 = a[i]
    //     b[i] = s2 - s1
    //     br cond, scalar.body, exit.block
    //
    //   exit.block:
    //     use = lcssa.phi [s1, scalar.body]
    //
    // In this example, s1 is a recurrence because it's value depends on the
    // previous iteration. In the first phase of vectorization, we created a
    // VPFirstOrderRecurrencePHIRecipe v1 for s1. Now we create the extracts
    // for users in the scalar preheader and exit block.
    //
    //   vector.ph:
    //     v_init = vector(..., ..., ..., a[-1])
    //     br vector.body
    //
    //   vector.body
    //     i = phi [0, vector.ph], [i+4, vector.body]
    //     v1 = phi [v_init, vector.ph], [v2, vector.body]
    //     v2 = a[i, i+1, i+2, i+3]
    //     b[i] = v2 - v1
    //     // Next, third phase will introduce v1' = splice(v1(3), v2(0, 1, 2))
    //     b[i, i+1, i+2, i+3] = v2 - v1
    //     br cond, vector.body, middle.block
    //
    //   middle.block:
    //     vector.recur.extract.for.phi = v2(2)
    //     vector.recur.extract = v2(3)
    //     br cond, scalar.ph, exit.block
    //
    //   scalar.ph:
    //     scalar.recur.init = phi [vector.recur.extract, middle.block],
    //                             [s.init, otherwise]
    //     br scalar.body
    //
    //   scalar.body:
    //     i = phi [0, scalar.ph], [i+1, scalar.body]
    //     s1 = phi [scalar.recur.init, scalar.ph], [s2, scalar.body]
    //     s2 = a[i]
    //     b[i] = s2 - s1
    //     br cond, scalar.body, exit.block
    //
    //   exit.block:
    //     lo = lcssa.phi [s1, scalar.body],
    //                    [vector.recur.extract.for.phi, middle.block]
    //
    // Now update VPIRInstructions modeling LCSSA phis in the exit block.
    // Extract the penultimate value of the recurrence and use it as operand for
    // the VPIRInstruction modeling the phi.
    for (VPUser *U : FOR->users()) {
      using namespace llvm::VPlanPatternMatch;
      if (!match(U, m_ExtractLastElement(m_Specific(FOR))))
        continue;
      // For VF vscale x 1, if vscale = 1, we are unable to extract the
      // penultimate value of the recurrence. Instead we rely on the existing
      // extract of the last element from the result of
      // VPInstruction::FirstOrderRecurrenceSplice.
      // TODO: Consider vscale_range info and UF.
      if (LoopVectorizationPlanner::getDecisionAndClampRange(IsScalableOne,
                                                             Range))
        return;
      VPValue *PenultimateElement = MiddleBuilder.createNaryOp(
          VPInstruction::ExtractPenultimateElement, {FOR->getBackedgeValue()},
          {}, "vector.recur.extract.for.phi");
      cast<VPInstruction>(U)->replaceAllUsesWith(PenultimateElement);
    }
  }
}

VPlanPtr LoopVectorizationPlanner::tryToBuildVPlanWithVPRecipes(
    VPlanPtr Plan, VFRange &Range, LoopVersioning *LVer) {

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
  VPlanTransforms::handleEarlyExits(*Plan, Legal->hasUncountableEarlyExit());
  VPlanTransforms::addMiddleCheck(*Plan, RequiresScalarEpilogueCheck,
                                  CM.foldTailByMasking());

  VPlanTransforms::createLoopRegions(*Plan);

  // Don't use getDecisionAndClampRange here, because we don't know the UF
  // so this function is better to be conservative, rather than to split
  // it up into different VPlans.
  // TODO: Consider using getDecisionAndClampRange here to split up VPlans.
  bool IVUpdateMayOverflow = false;
  for (ElementCount VF : Range)
    IVUpdateMayOverflow |= !isIndvarOverflowCheckKnownFalse(&CM, VF);

  TailFoldingStyle Style = CM.getTailFoldingStyle(IVUpdateMayOverflow);
  // Use NUW for the induction increment if we proved that it won't overflow in
  // the vector loop or when not folding the tail. In the later case, we know
  // that the canonical induction increment will not overflow as the vector trip
  // count is >= increment and a multiple of the increment.
  bool HasNUW = !IVUpdateMayOverflow || Style == TailFoldingStyle::None;
  if (!HasNUW) {
    auto *IVInc = Plan->getVectorLoopRegion()
                      ->getExitingBasicBlock()
                      ->getTerminator()
                      ->getOperand(0);
    assert(match(IVInc, m_VPInstruction<Instruction::Add>(
                            m_Specific(Plan->getCanonicalIV()), m_VPValue())) &&
           "Did not find the canonical IV increment");
    cast<VPRecipeWithIRFlags>(IVInc)->dropPoisonGeneratingFlags();
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
  // Predicate and linearize the top-level loop region.
  // ---------------------------------------------------------------------------
  auto BlockMaskCache = VPlanTransforms::introduceMasksAndLinearize(
      *Plan, CM.foldTailByMasking());

  // ---------------------------------------------------------------------------
  // Construct wide recipes and apply predication for original scalar
  // VPInstructions in the loop.
  // ---------------------------------------------------------------------------
  VPRecipeBuilder RecipeBuilder(*Plan, OrigLoop, TLI, &TTI, Legal, CM, PSE,
                                Builder, BlockMaskCache, LVer);
  RecipeBuilder.collectScaledReductions(Range);

  // Scan the body of the loop in a topological order to visit each basic block
  // after having visited its predecessor basic blocks.
  VPRegionBlock *LoopRegion = Plan->getVectorLoopRegion();
  VPBasicBlock *HeaderVPBB = LoopRegion->getEntryBasicBlock();
  ReversePostOrderTraversal<VPBlockShallowTraversalWrapper<VPBlockBase *>> RPOT(
      HeaderVPBB);

  auto *MiddleVPBB = Plan->getMiddleBlock();
  VPBasicBlock::iterator MBIP = MiddleVPBB->getFirstNonPhi();
  // Mapping from VPValues in the initial plan to their widened VPValues. Needed
  // temporarily to update created block masks.
  DenseMap<VPValue *, VPValue *> Old2New;
  for (VPBasicBlock *VPBB : VPBlockUtils::blocksOnly<VPBasicBlock>(RPOT)) {
    // Convert input VPInstructions to widened recipes.
    for (VPRecipeBase &R : make_early_inc_range(*VPBB)) {
      auto *SingleDef = cast<VPSingleDefRecipe>(&R);
      auto *UnderlyingValue = SingleDef->getUnderlyingValue();
      // Skip recipes that do not need transforming, including canonical IV,
      // wide canonical IV and VPInstructions without underlying values. The
      // latter are added above for masking.
      // FIXME: Migrate code relying on the underlying instruction from VPlan0
      // to construct recipes below to not use the underlying instruction.
      if (isa<VPCanonicalIVPHIRecipe, VPWidenCanonicalIVRecipe, VPBlendRecipe>(
              &R) ||
          (isa<VPInstruction>(&R) && !UnderlyingValue))
        continue;

      // FIXME: VPlan0, which models a copy of the original scalar loop, should
      // not use VPWidenPHIRecipe to model the phis.
      assert((isa<VPWidenPHIRecipe>(&R) || isa<VPInstruction>(&R)) &&
             UnderlyingValue && "unsupported recipe");

      // TODO: Gradually replace uses of underlying instruction by analyses on
      // VPlan.
      Instruction *Instr = cast<Instruction>(UnderlyingValue);
      Builder.setInsertPoint(SingleDef);

      // The stores with invariant address inside the loop will be deleted, and
      // in the exit block, a uniform store recipe will be created for the final
      // invariant store of the reduction.
      StoreInst *SI;
      if ((SI = dyn_cast<StoreInst>(Instr)) &&
          Legal->isInvariantAddressOfReduction(SI->getPointerOperand())) {
        // Only create recipe for the final invariant store of the reduction.
        if (Legal->isInvariantStoreOfReduction(SI)) {
          auto *Recipe =
              new VPReplicateRecipe(SI, R.operands(), true /* IsUniform */,
                                    nullptr /*Mask*/, VPIRMetadata(*SI, LVer));
          Recipe->insertBefore(*MiddleVPBB, MBIP);
        }
        R.eraseFromParent();
        continue;
      }

      VPRecipeBase *Recipe =
          RecipeBuilder.tryToCreateWidenRecipe(SingleDef, Range);
      if (!Recipe) {
        SmallVector<VPValue *, 4> Operands(R.operands());
        Recipe = RecipeBuilder.handleReplication(Instr, Operands, Range);
      }

      RecipeBuilder.setRecipe(Instr, Recipe);
      if (isa<VPWidenIntOrFpInductionRecipe>(Recipe) && isa<TruncInst>(Instr)) {
        // Optimized a truncate to VPWidenIntOrFpInductionRecipe. It needs to be
        // moved to the phi section in the header.
        Recipe->insertBefore(*HeaderVPBB, HeaderVPBB->getFirstNonPhi());
      } else {
        Builder.insert(Recipe);
      }
      if (Recipe->getNumDefinedValues() == 1) {
        SingleDef->replaceAllUsesWith(Recipe->getVPSingleValue());
        Old2New[SingleDef] = Recipe->getVPSingleValue();
      } else {
        assert(Recipe->getNumDefinedValues() == 0 &&
               "Unexpected multidef recipe");
        R.eraseFromParent();
      }
    }
  }

  // replaceAllUsesWith above may invalidate the block masks. Update them here.
  // TODO: Include the masks as operands in the predicated VPlan directly
  // to remove the need to keep a map of masks beyond the predication
  // transform.
  RecipeBuilder.updateBlockMaskCache(Old2New);
  for (VPValue *Old : Old2New.keys())
    Old->getDefiningRecipe()->eraseFromParent();

  assert(isa<VPRegionBlock>(Plan->getVectorLoopRegion()) &&
         !Plan->getVectorLoopRegion()->getEntryBasicBlock()->empty() &&
         "entry block must be set to a VPRegionBlock having a non-empty entry "
         "VPBasicBlock");

  // Update wide induction increments to use the same step as the corresponding
  // wide induction. This enables detecting induction increments directly in
  // VPlan and removes redundant splats.
  for (const auto &[Phi, ID] : Legal->getInductionVars()) {
    auto *IVInc = cast<Instruction>(
        Phi->getIncomingValueForBlock(OrigLoop->getLoopLatch()));
    if (IVInc->getOperand(0) != Phi || IVInc->getOpcode() != Instruction::Add)
      continue;
    VPWidenInductionRecipe *WideIV =
        cast<VPWidenInductionRecipe>(RecipeBuilder.getRecipe(Phi));
    VPRecipeBase *R = RecipeBuilder.getRecipe(IVInc);
    R->setOperand(1, WideIV->getStepValue());
  }

  addExitUsersForFirstOrderRecurrences(*Plan, Range);
  DenseMap<VPValue *, VPValue *> IVEndValues;
  addScalarResumePhis(RecipeBuilder, *Plan, IVEndValues);

  // ---------------------------------------------------------------------------
  // Transform initial VPlan: Apply previously taken decisions, in order, to
  // bring the VPlan to its final state.
  // ---------------------------------------------------------------------------

  // Adjust the recipes for any inloop reductions.
  adjustRecipesForReductions(Plan, RecipeBuilder, Range.Start);

  // Apply mandatory transformation to handle FP maxnum/minnum reduction with
  // NaNs if possible, bail out otherwise.
  if (!VPlanTransforms::runPass(VPlanTransforms::handleMaxMinNumReductions,
                                *Plan))
    return nullptr;

  // Transform recipes to abstract recipes if it is legal and beneficial and
  // clamp the range for better cost estimation.
  // TODO: Enable following transform when the EVL-version of extended-reduction
  // and mulacc-reduction are implemented.
  if (!CM.foldTailWithEVL()) {
    VPCostContext CostCtx(CM.TTI, *CM.TLI, *Plan, CM, CM.CostKind);
    VPlanTransforms::runPass(VPlanTransforms::convertToAbstractRecipes, *Plan,
                             CostCtx, Range);
  }

  for (ElementCount VF : Range)
    Plan->addVF(VF);
  Plan->setName("Initial VPlan");

  // Interleave memory: for each Interleave Group we marked earlier as relevant
  // for this VPlan, replace the Recipes widening its memory instructions with a
  // single VPInterleaveRecipe at its insertion point.
  VPlanTransforms::runPass(VPlanTransforms::createInterleaveGroups, *Plan,
                           InterleaveGroups, RecipeBuilder,
                           CM.isScalarEpilogueAllowed());

  // Replace VPValues for known constant strides guaranteed by predicate scalar
  // evolution.
  auto CanUseVersionedStride = [&Plan](VPUser &U, unsigned) {
    auto *R = cast<VPRecipeBase>(&U);
    return R->getParent()->getParent() ||
           R->getParent() ==
               Plan->getVectorLoopRegion()->getSinglePredecessor();
  };
  for (auto [_, Stride] : Legal->getLAI()->getSymbolicStrides()) {
    auto *StrideV = cast<SCEVUnknown>(Stride)->getValue();
    auto *ScevStride = dyn_cast<SCEVConstant>(PSE.getSCEV(StrideV));
    // Only handle constant strides for now.
    if (!ScevStride)
      continue;

    auto *CI = Plan->getOrAddLiveIn(
        ConstantInt::get(Stride->getType(), ScevStride->getAPInt()));
    if (VPValue *StrideVPV = Plan->getLiveIn(StrideV))
      StrideVPV->replaceUsesWithIf(CI, CanUseVersionedStride);

    // The versioned value may not be used in the loop directly but through a
    // sext/zext. Add new live-ins in those cases.
    for (Value *U : StrideV->users()) {
      if (!isa<SExtInst, ZExtInst>(U))
        continue;
      VPValue *StrideVPV = Plan->getLiveIn(U);
      if (!StrideVPV)
        continue;
      unsigned BW = U->getType()->getScalarSizeInBits();
      APInt C = isa<SExtInst>(U) ? ScevStride->getAPInt().sext(BW)
                                 : ScevStride->getAPInt().zext(BW);
      VPValue *CI = Plan->getOrAddLiveIn(ConstantInt::get(U->getType(), C));
      StrideVPV->replaceUsesWithIf(CI, CanUseVersionedStride);
    }
  }

  auto BlockNeedsPredication = [this](BasicBlock *BB) {
    return Legal->blockNeedsPredication(BB);
  };
  VPlanTransforms::runPass(VPlanTransforms::dropPoisonGeneratingRecipes, *Plan,
                           BlockNeedsPredication);

  // Sink users of fixed-order recurrence past the recipe defining the previous
  // value and introduce FirstOrderRecurrenceSplice VPInstructions.
  if (!VPlanTransforms::runPass(VPlanTransforms::adjustFixedOrderRecurrences,
                                *Plan, Builder))
    return nullptr;

  if (useActiveLaneMask(Style)) {
    // TODO: Move checks to VPlanTransforms::addActiveLaneMask once
    // TailFoldingStyle is visible there.
    bool ForControlFlow = useActiveLaneMaskForControlFlow(Style);
    bool WithoutRuntimeCheck =
        Style == TailFoldingStyle::DataAndControlFlowWithoutRuntimeCheck;
    VPlanTransforms::addActiveLaneMask(*Plan, ForControlFlow,
                                       WithoutRuntimeCheck);
  }
  VPlanTransforms::optimizeInductionExitUsers(*Plan, IVEndValues, *PSE.getSE());

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
  VPlanTransforms::handleEarlyExits(*Plan,
                                    /*HasUncountableExit*/ false);
  VPlanTransforms::addMiddleCheck(*Plan, /*RequiresScalarEpilogue*/ true,
                                  /*TailFolded*/ false);

  VPlanTransforms::createLoopRegions(*Plan);

  for (ElementCount VF : Range)
    Plan->addVF(VF);

  if (!VPlanTransforms::tryToConvertVPInstructionsToVPRecipes(
          Plan,
          [this](PHINode *P) {
            return Legal->getIntOrFpInductionDescriptor(P);
          },
          *TLI))
    return nullptr;

  // Collect mapping of IR header phis to header phi recipes, to be used in
  // addScalarResumePhis.
  DenseMap<VPBasicBlock *, VPValue *> BlockMaskCache;
  VPRecipeBuilder RecipeBuilder(*Plan, OrigLoop, TLI, &TTI, Legal, CM, PSE,
                                Builder, BlockMaskCache, nullptr /*LVer*/);
  for (auto &R : Plan->getVectorLoopRegion()->getEntryBasicBlock()->phis()) {
    if (isa<VPCanonicalIVPHIRecipe>(&R))
      continue;
    auto *HeaderR = cast<VPHeaderPHIRecipe>(&R);
    RecipeBuilder.setRecipe(HeaderR->getUnderlyingInstr(), HeaderR);
  }
  DenseMap<VPValue *, VPValue *> IVEndValues;
  // TODO: IVEndValues are not used yet in the native path, to optimize exit
  // values.
  addScalarResumePhis(RecipeBuilder, *Plan, IVEndValues);

  assert(verifyVPlanIsValid(*Plan) && "VPlan is invalid");
  return Plan;
}

// Adjust the recipes for reductions. For in-loop reductions the chain of
// instructions leading from the loop exit instr to the phi need to be converted
// to reductions, with one operand being vector and the other being the scalar
// reduction chain. For other reductions, a select is introduced between the phi
// and users outside the vector region when folding the tail.
//
// A ComputeReductionResult recipe is added to the middle block, also for
// in-loop reductions which compute their result in-loop, because generating
// the subsequent bc.merge.rdx phi is driven by ComputeReductionResult recipes.
//
// Adjust AnyOf reductions; replace the reduction phi for the selected value
// with a boolean reduction phi node to check if the condition is true in any
// iteration. The final value is selected by the final ComputeReductionResult.
void LoopVectorizationPlanner::adjustRecipesForReductions(
    VPlanPtr &Plan, VPRecipeBuilder &RecipeBuilder, ElementCount MinVF) {
  using namespace VPlanPatternMatch;
  VPRegionBlock *VectorLoopRegion = Plan->getVectorLoopRegion();
  VPBasicBlock *Header = VectorLoopRegion->getEntryBasicBlock();
  VPBasicBlock *MiddleVPBB = Plan->getMiddleBlock();
  SmallVector<VPRecipeBase *> ToDelete;

  for (VPRecipeBase &R : Header->phis()) {
    auto *PhiR = dyn_cast<VPReductionPHIRecipe>(&R);
    if (!PhiR || !PhiR->isInLoop() || (MinVF.isScalar() && !PhiR->isOrdered()))
      continue;

    RecurKind Kind = PhiR->getRecurrenceKind();
    assert(
        !RecurrenceDescriptor::isAnyOfRecurrenceKind(Kind) &&
        !RecurrenceDescriptor::isFindIVRecurrenceKind(Kind) &&
        "AnyOf and FindIV reductions are not allowed for in-loop reductions");

    // Collect the chain of "link" recipes for the reduction starting at PhiR.
    SetVector<VPSingleDefRecipe *> Worklist;
    Worklist.insert(PhiR);
    for (unsigned I = 0; I != Worklist.size(); ++I) {
      VPSingleDefRecipe *Cur = Worklist[I];
      for (VPUser *U : Cur->users()) {
        auto *UserRecipe = cast<VPSingleDefRecipe>(U);
        if (!UserRecipe->getParent()->getEnclosingLoopRegion()) {
          assert((UserRecipe->getParent() == MiddleVPBB ||
                  UserRecipe->getParent() == Plan->getScalarPreheader()) &&
                 "U must be either in the loop region, the middle block or the "
                 "scalar preheader.");
          continue;
        }
        Worklist.insert(UserRecipe);
      }
    }

    // Visit operation "Links" along the reduction chain top-down starting from
    // the phi until LoopExitValue. We keep track of the previous item
    // (PreviousLink) to tell which of the two operands of a Link will remain
    // scalar and which will be reduced. For minmax by select(cmp), Link will be
    // the select instructions. Blend recipes of in-loop reduction phi's  will
    // get folded to their non-phi operand, as the reduction recipe handles the
    // condition directly.
    VPSingleDefRecipe *PreviousLink = PhiR; // Aka Worklist[0].
    for (VPSingleDefRecipe *CurrentLink : drop_begin(Worklist)) {
      if (auto *Blend = dyn_cast<VPBlendRecipe>(CurrentLink)) {
        assert(Blend->getNumIncomingValues() == 2 &&
               "Blend must have 2 incoming values");
        if (Blend->getIncomingValue(0) == PhiR) {
          Blend->replaceAllUsesWith(Blend->getIncomingValue(1));
        } else {
          assert(Blend->getIncomingValue(1) == PhiR &&
                 "PhiR must be an operand of the blend");
          Blend->replaceAllUsesWith(Blend->getIncomingValue(0));
        }
        continue;
      }

      Instruction *CurrentLinkI = CurrentLink->getUnderlyingInstr();

      // Index of the first operand which holds a non-mask vector operand.
      unsigned IndexOfFirstOperand;
      // Recognize a call to the llvm.fmuladd intrinsic.
      bool IsFMulAdd = (Kind == RecurKind::FMulAdd);
      VPValue *VecOp;
      VPBasicBlock *LinkVPBB = CurrentLink->getParent();
      if (IsFMulAdd) {
        assert(
            RecurrenceDescriptor::isFMulAddIntrinsic(CurrentLinkI) &&
            "Expected instruction to be a call to the llvm.fmuladd intrinsic");
        assert(((MinVF.isScalar() && isa<VPReplicateRecipe>(CurrentLink)) ||
                isa<VPWidenIntrinsicRecipe>(CurrentLink)) &&
               CurrentLink->getOperand(2) == PreviousLink &&
               "expected a call where the previous link is the added operand");

        // If the instruction is a call to the llvm.fmuladd intrinsic then we
        // need to create an fmul recipe (multiplying the first two operands of
        // the fmuladd together) to use as the vector operand for the fadd
        // reduction.
        VPInstruction *FMulRecipe = new VPInstruction(
            Instruction::FMul,
            {CurrentLink->getOperand(0), CurrentLink->getOperand(1)},
            CurrentLinkI->getFastMathFlags());
        LinkVPBB->insert(FMulRecipe, CurrentLink->getIterator());
        VecOp = FMulRecipe;
      } else if (PhiR->isInLoop() && Kind == RecurKind::AddChainWithSubs &&
                 CurrentLinkI->getOpcode() == Instruction::Sub) {
        Type *PhiTy = PhiR->getUnderlyingValue()->getType();
        auto *Zero = Plan->getOrAddLiveIn(ConstantInt::get(PhiTy, 0));
        VPWidenRecipe *Sub = new VPWidenRecipe(
            Instruction::Sub, {Zero, CurrentLink->getOperand(1)}, {},
            VPIRMetadata(), CurrentLinkI->getDebugLoc());
        Sub->setUnderlyingValue(CurrentLinkI);
        LinkVPBB->insert(Sub, CurrentLink->getIterator());
        VecOp = Sub;
      } else {
        if (RecurrenceDescriptor::isMinMaxRecurrenceKind(Kind)) {
          if (isa<VPWidenRecipe>(CurrentLink)) {
            assert(isa<CmpInst>(CurrentLinkI) &&
                   "need to have the compare of the select");
            continue;
          }
          assert(isa<VPWidenSelectRecipe>(CurrentLink) &&
                 "must be a select recipe");
          IndexOfFirstOperand = 1;
        } else {
          assert((MinVF.isScalar() || isa<VPWidenRecipe>(CurrentLink)) &&
                 "Expected to replace a VPWidenSC");
          IndexOfFirstOperand = 0;
        }
        // Note that for non-commutable operands (cmp-selects), the semantics of
        // the cmp-select are captured in the recurrence kind.
        unsigned VecOpId =
            CurrentLink->getOperand(IndexOfFirstOperand) == PreviousLink
                ? IndexOfFirstOperand + 1
                : IndexOfFirstOperand;
        VecOp = CurrentLink->getOperand(VecOpId);
        assert(VecOp != PreviousLink &&
               CurrentLink->getOperand(CurrentLink->getNumOperands() - 1 -
                                       (VecOpId - IndexOfFirstOperand)) ==
                   PreviousLink &&
               "PreviousLink must be the operand other than VecOp");
      }

      VPValue *CondOp = nullptr;
      if (CM.blockNeedsPredicationForAnyReason(CurrentLinkI->getParent()))
        CondOp = RecipeBuilder.getBlockInMask(CurrentLink->getParent());

      // TODO: Retrieve FMFs from recipes directly.
      RecurrenceDescriptor RdxDesc = Legal->getRecurrenceDescriptor(
          cast<PHINode>(PhiR->getUnderlyingInstr()));
      // Non-FP RdxDescs will have all fast math flags set, so clear them.
      FastMathFlags FMFs = isa<FPMathOperator>(CurrentLinkI)
                               ? RdxDesc.getFastMathFlags()
                               : FastMathFlags();
      auto *RedRecipe = new VPReductionRecipe(
          Kind, FMFs, CurrentLinkI, PreviousLink, VecOp, CondOp,
          PhiR->isOrdered(), CurrentLinkI->getDebugLoc());
      // Append the recipe to the end of the VPBasicBlock because we need to
      // ensure that it comes after all of it's inputs, including CondOp.
      // Delete CurrentLink as it will be invalid if its operand is replaced
      // with a reduction defined at the bottom of the block in the next link.
      if (LinkVPBB->getNumSuccessors() == 0)
        RedRecipe->insertBefore(&*std::prev(std::prev(LinkVPBB->end())));
      else
        LinkVPBB->appendRecipe(RedRecipe);

      CurrentLink->replaceAllUsesWith(RedRecipe);
      ToDelete.push_back(CurrentLink);
      PreviousLink = RedRecipe;
    }
  }
  VPBasicBlock *LatchVPBB = VectorLoopRegion->getExitingBasicBlock();
  Builder.setInsertPoint(&*std::prev(std::prev(LatchVPBB->end())));
  VPBasicBlock::iterator IP = MiddleVPBB->getFirstNonPhi();
  for (VPRecipeBase &R :
       Plan->getVectorLoopRegion()->getEntryBasicBlock()->phis()) {
    VPReductionPHIRecipe *PhiR = dyn_cast<VPReductionPHIRecipe>(&R);
    if (!PhiR)
      continue;

    const RecurrenceDescriptor &RdxDesc = Legal->getRecurrenceDescriptor(
        cast<PHINode>(PhiR->getUnderlyingInstr()));
    Type *PhiTy = PhiR->getUnderlyingValue()->getType();
    // If tail is folded by masking, introduce selects between the phi
    // and the users outside the vector region of each reduction, at the
    // beginning of the dedicated latch block.
    auto *OrigExitingVPV = PhiR->getBackedgeValue();
    auto *NewExitingVPV = PhiR->getBackedgeValue();
    // Don't output selects for partial reductions because they have an output
    // with fewer lanes than the VF. So the operands of the select would have
    // different numbers of lanes. Partial reductions mask the input instead.
    if (!PhiR->isInLoop() && CM.foldTailByMasking() &&
        !isa<VPPartialReductionRecipe>(OrigExitingVPV->getDefiningRecipe())) {
      VPValue *Cond = RecipeBuilder.getBlockInMask(PhiR->getParent());
      std::optional<FastMathFlags> FMFs =
          PhiTy->isFloatingPointTy()
              ? std::make_optional(RdxDesc.getFastMathFlags())
              : std::nullopt;
      NewExitingVPV =
          Builder.createSelect(Cond, OrigExitingVPV, PhiR, {}, "", FMFs);
      OrigExitingVPV->replaceUsesWithIf(NewExitingVPV, [](VPUser &U, unsigned) {
        return isa<VPInstruction>(&U) &&
               (cast<VPInstruction>(&U)->getOpcode() ==
                    VPInstruction::ComputeAnyOfResult ||
                cast<VPInstruction>(&U)->getOpcode() ==
                    VPInstruction::ComputeReductionResult ||
                cast<VPInstruction>(&U)->getOpcode() ==
                    VPInstruction::ComputeFindIVResult);
      });
      if (CM.usePredicatedReductionSelect())
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
    RecurKind RecurrenceKind = PhiR->getRecurrenceKind();
    if (RecurrenceDescriptor::isFindIVRecurrenceKind(RecurrenceKind)) {
      VPValue *Start = PhiR->getStartValue();
      VPValue *Sentinel = Plan->getOrAddLiveIn(RdxDesc.getSentinelValue());
      FinalReductionResult =
          Builder.createNaryOp(VPInstruction::ComputeFindIVResult,
                               {PhiR, Start, Sentinel, NewExitingVPV}, ExitDL);
    } else if (RecurrenceDescriptor::isAnyOfRecurrenceKind(RecurrenceKind)) {
      VPValue *Start = PhiR->getStartValue();
      FinalReductionResult =
          Builder.createNaryOp(VPInstruction::ComputeAnyOfResult,
                               {PhiR, Start, NewExitingVPV}, ExitDL);
    } else {
      VPIRFlags Flags =
          RecurrenceDescriptor::isFloatingPointRecurrenceKind(RecurrenceKind)
              ? VPIRFlags(RdxDesc.getFastMathFlags())
              : VPIRFlags();
      FinalReductionResult =
          Builder.createNaryOp(VPInstruction::ComputeReductionResult,
                               {PhiR, NewExitingVPV}, Flags, ExitDL);
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
      auto *Trunc =
          new VPWidenCastRecipe(Instruction::Trunc, NewExitingVPV, RdxTy);
      Instruction::CastOps ExtendOpc =
          RdxDesc.isSigned() ? Instruction::SExt : Instruction::ZExt;
      auto *Extnd = new VPWidenCastRecipe(ExtendOpc, Trunc, PhiTy);
      Trunc->insertAfter(NewExitingVPV->getDefiningRecipe());
      Extnd->insertAfter(Trunc);
      if (PhiR->getOperand(1) == NewExitingVPV)
        PhiR->setOperand(1, Extnd->getVPSingleValue());

      // Update ComputeReductionResult with the truncated exiting value and
      // extend its result.
      FinalReductionResult->setOperand(1, Trunc);
      FinalReductionResult =
          Builder.createScalarCast(ExtendOpc, FinalReductionResult, PhiTy, {});
    }

    // Update all users outside the vector region. Also replace redundant
    // ExtractLastElement.
    for (auto *U : to_vector(OrigExitingVPV->users())) {
      auto *Parent = cast<VPRecipeBase>(U)->getParent();
      if (FinalReductionResult == U || Parent->getParent())
        continue;
      U->replaceUsesOfWith(OrigExitingVPV, FinalReductionResult);
      if (match(U, m_ExtractLastElement(m_VPValue())))
        cast<VPInstruction>(U)->replaceAllUsesWith(FinalReductionResult);
    }

    // Adjust AnyOf reductions; replace the reduction phi for the selected value
    // with a boolean reduction phi node to check if the condition is true in
    // any iteration. The final value is selected by the final
    // ComputeReductionResult.
    if (RecurrenceDescriptor::isAnyOfRecurrenceKind(RecurrenceKind)) {
      auto *Select = cast<VPRecipeBase>(*find_if(PhiR->users(), [](VPUser *U) {
        return isa<VPWidenSelectRecipe>(U) ||
               (isa<VPReplicateRecipe>(U) &&
                cast<VPReplicateRecipe>(U)->getUnderlyingInstr()->getOpcode() ==
                    Instruction::Select);
      }));
      VPValue *Cmp = Select->getOperand(0);
      // If the compare is checking the reduction PHI node, adjust it to check
      // the start value.
      if (VPRecipeBase *CmpR = Cmp->getDefiningRecipe())
        CmpR->replaceUsesOfWith(PhiR, PhiR->getStartValue());
      Builder.setInsertPoint(Select);

      // If the true value of the select is the reduction phi, the new value is
      // selected if the negated condition is true in any iteration.
      if (Select->getOperand(1) == PhiR)
        Cmp = Builder.createNot(Cmp);
      VPValue *Or = Builder.createOr(PhiR, Cmp);
      Select->getVPSingleValue()->replaceAllUsesWith(Or);
      // Delete Select now that it has invalid types.
      ToDelete.push_back(Select);

      // Convert the reduction phi to operate on bools.
      PhiR->setOperand(0, Plan->getOrAddLiveIn(ConstantInt::getFalse(
                              OrigLoop->getHeader()->getContext())));
      continue;
    }

    if (RecurrenceDescriptor::isFindIVRecurrenceKind(
            RdxDesc.getRecurrenceKind())) {
      // Adjust the start value for FindFirstIV/FindLastIV recurrences to use
      // the sentinel value after generating the ResumePhi recipe, which uses
      // the original start value.
      PhiR->setOperand(0, Plan->getOrAddLiveIn(RdxDesc.getSentinelValue()));
    }
    RecurKind RK = RdxDesc.getRecurrenceKind();
    if ((!RecurrenceDescriptor::isAnyOfRecurrenceKind(RK) &&
         !RecurrenceDescriptor::isFindIVRecurrenceKind(RK) &&
         !RecurrenceDescriptor::isMinMaxRecurrenceKind(RK))) {
      VPBuilder PHBuilder(Plan->getVectorPreheader());
      VPValue *Iden = Plan->getOrAddLiveIn(
          getRecurrenceIdentity(RK, PhiTy, RdxDesc.getFastMathFlags()));
      // If the PHI is used by a partial reduction, set the scale factor.
      unsigned ScaleFactor =
          RecipeBuilder.getScalingForReduction(RdxDesc.getLoopExitInstr())
              .value_or(1);
      Type *I32Ty = IntegerType::getInt32Ty(PhiTy->getContext());
      auto *ScaleFactorVPV =
          Plan->getOrAddLiveIn(ConstantInt::get(I32Ty, ScaleFactor));
      VPValue *StartV = PHBuilder.createNaryOp(
          VPInstruction::ReductionStartVector,
          {PhiR->getStartValue(), Iden, ScaleFactorVPV},
          PhiTy->isFloatingPointTy() ? RdxDesc.getFastMathFlags()
                                     : FastMathFlags());
      PhiR->setOperand(0, StartV);
    }
  }
  for (VPRecipeBase *R : ToDelete)
    R->eraseFromParent();

  VPlanTransforms::runPass(VPlanTransforms::clearReductionWrapFlags, *Plan);
}

void LoopVectorizationPlanner::attachRuntimeChecks(
    VPlan &Plan, GeneratedRTChecks &RTChecks, bool HasBranchWeights) const {
  const auto &[SCEVCheckCond, SCEVCheckBlock] = RTChecks.getSCEVChecks();
  if (SCEVCheckBlock && SCEVCheckBlock->hasNPredecessors(0)) {
    assert((!CM.OptForSize ||
            CM.Hints->getForce() == LoopVectorizeHints::FK_Enabled) &&
           "Cannot SCEV check stride or overflow when optimizing for size");
    VPlanTransforms::attachCheckBlock(Plan, SCEVCheckCond, SCEVCheckBlock,
                                      HasBranchWeights);
  }
  const auto &[MemCheckCond, MemCheckBlock] = RTChecks.getMemRuntimeChecks();
  if (MemCheckBlock && MemCheckBlock->hasNPredecessors(0)) {
    // VPlan-native path does not do any analysis for runtime checks
    // currently.
    assert((!EnableVPlanNativePath || OrigLoop->isInnermost()) &&
           "Runtime checks are not supported for outer loops yet");

    if (CM.OptForSize) {
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
    VPlanTransforms::attachCheckBlock(Plan, MemCheckCond, MemCheckBlock,
                                      HasBranchWeights);
  }
}

void LoopVectorizationPlanner::addMinimumIterationCheck(
    VPlan &Plan, ElementCount VF, unsigned UF,
    ElementCount MinProfitableTripCount) const {
  // vscale is not necessarily a power-of-2, which means we cannot guarantee
  // an overflow to zero when updating induction variables and so an
  // additional overflow check is required before entering the vector loop.
  bool IsIndvarOverflowCheckNeededForVF =
      VF.isScalable() && !TTI.isVScaleKnownToBeAPowerOfTwo() &&
      !isIndvarOverflowCheckKnownFalse(&CM, VF, UF) &&
      CM.getTailFoldingStyle() !=
          TailFoldingStyle::DataAndControlFlowWithoutRuntimeCheck;
  const uint32_t *BranchWeigths =
      hasBranchWeightMD(*OrigLoop->getLoopLatch()->getTerminator())
          ? &MinItersBypassWeights[0]
          : nullptr;
  VPlanTransforms::addMinimumIterationCheck(
      Plan, VF, UF, MinProfitableTripCount,
      CM.requiresScalarEpilogue(VF.isVector()), CM.foldTailByMasking(),
      IsIndvarOverflowCheckNeededForVF, OrigLoop, BranchWeigths,
      OrigLoop->getLoopPredecessor()->getTerminator()->getDebugLoc(),
      *PSE.getSE());
}

void VPDerivedIVRecipe::execute(VPTransformState &State) {
  assert(!State.Lane && "VPDerivedIVRecipe being replicated.");

  // Fast-math-flags propagate from the original induction instruction.
  IRBuilder<>::FastMathFlagGuard FMFG(State.Builder);
  if (FPBinOp)
    State.Builder.setFastMathFlags(FPBinOp->getFastMathFlags());

  Value *Step = State.get(getStepValue(), VPLane(0));
  Value *Index = State.get(getOperand(1), VPLane(0));
  Value *DerivedIV = emitTransformedIndex(
      State.Builder, Index, getStartValue()->getLiveInIRValue(), Step, Kind,
      cast_if_present<BinaryOperator>(FPBinOp));
  DerivedIV->setName(Name);
  State.set(this, DerivedIV, VPLane(0));
}

// Determine how to lower the scalar epilogue, which depends on 1) optimising
// for minimum code-size, 2) predicate compiler options, 3) loop hints forcing
// predication, and 4) a TTI hook that analyses whether the loop is suitable
// for predication.
static ScalarEpilogueLowering getScalarEpilogueLowering(
    Function *F, Loop *L, LoopVectorizeHints &Hints, ProfileSummaryInfo *PSI,
    BlockFrequencyInfo *BFI, TargetTransformInfo *TTI, TargetLibraryInfo *TLI,
    LoopVectorizationLegality &LVL, InterleavedAccessInfo *IAI) {
  // 1) OptSize takes precedence over all other options, i.e. if this is set,
  // don't look at hints or options, and don't request a scalar epilogue.
  // (For PGSO, as shouldOptimizeForSize isn't currently accessible from
  // LoopAccessInfo (due to code dependency and not being able to reliably get
  // PSI/BFI from a loop analysis under NPM), we cannot suppress the collection
  // of strides in LoopAccessInfo::analyzeLoop() and vectorize without
  // versioning when the vectorization is forced, unlike hasOptSize. So revert
  // back to the old way and vectorize with versioning when forced. See D81345.)
  if (F->hasOptSize() || (llvm::shouldOptimizeForSize(L->getHeader(), PSI, BFI,
                                                      PGSOQueryType::IRPass) &&
                          Hints.getForce() != LoopVectorizeHints::FK_Enabled))
    return CM_ScalarEpilogueNotAllowedOptSize;

  // 2) If set, obey the directives
  if (PreferPredicateOverEpilogue.getNumOccurrences()) {
    switch (PreferPredicateOverEpilogue) {
    case PreferPredicateTy::ScalarEpilogue:
      return CM_ScalarEpilogueAllowed;
    case PreferPredicateTy::PredicateElseScalarEpilogue:
      return CM_ScalarEpilogueNotNeededUsePredicate;
    case PreferPredicateTy::PredicateOrDontVectorize:
      return CM_ScalarEpilogueNotAllowedUsePredicate;
    };
  }

  // 3) If set, obey the hints
  switch (Hints.getPredicate()) {
  case LoopVectorizeHints::FK_Enabled:
    return CM_ScalarEpilogueNotNeededUsePredicate;
  case LoopVectorizeHints::FK_Disabled:
    return CM_ScalarEpilogueAllowed;
  };

  // 4) if the TTI hook indicates this is profitable, request predication.
  TailFoldingInfo TFI(TLI, &LVL, IAI);
  if (TTI->preferPredicateOverEpilogue(&TFI))
    return CM_ScalarEpilogueNotNeededUsePredicate;

  return CM_ScalarEpilogueAllowed;
}

// Process the loop in the VPlan-native vectorization path. This path builds
// VPlan upfront in the vectorization pipeline, which allows to apply
// VPlan-to-VPlan transformations from the very beginning without modifying the
// input LLVM IR.
static bool processLoopInVPlanNativePath(
    Loop *L, PredicatedScalarEvolution &PSE, LoopInfo *LI, DominatorTree *DT,
    LoopVectorizationLegality *LVL, TargetTransformInfo *TTI,
    TargetLibraryInfo *TLI, DemandedBits *DB, AssumptionCache *AC,
    OptimizationRemarkEmitter *ORE, BlockFrequencyInfo *BFI,
    ProfileSummaryInfo *PSI, LoopVectorizeHints &Hints,
    LoopVectorizationRequirements &Requirements) {

  if (isa<SCEVCouldNotCompute>(PSE.getBackedgeTakenCount())) {
    LLVM_DEBUG(dbgs() << "LV: cannot compute the outer-loop trip count\n");
    return false;
  }
  assert(EnableVPlanNativePath && "VPlan-native path is disabled.");
  Function *F = L->getHeader()->getParent();
  InterleavedAccessInfo IAI(PSE, L, DT, LI, LVL->getLAI());

  ScalarEpilogueLowering SEL =
      getScalarEpilogueLowering(F, L, Hints, PSI, BFI, TTI, TLI, *LVL, &IAI);

  LoopVectorizationCostModel CM(SEL, L, PSE, LI, LVL, *TTI, TLI, DB, AC, ORE, F,
                                &Hints, IAI, PSI, BFI);
  // Use the planner for outer loop vectorization.
  // TODO: CM is not used at this point inside the planner. Turn CM into an
  // optional argument if we don't need it in the future.
  LoopVectorizationPlanner LVP(L, LI, DT, TLI, *TTI, LVL, CM, IAI, PSE, Hints,
                               ORE);

  // Get user vectorization factor.
  ElementCount UserVF = Hints.getWidth();

  CM.collectElementTypesForWidening();

  // Plan how to best vectorize, return the best VF and its cost.
  const VectorizationFactor VF = LVP.planInVPlanNativePath(UserVF);

  // If we are stress testing VPlan builds, do not attempt to generate vector
  // code. Masked vector code generation support will follow soon.
  // Also, do not attempt to vectorize if no vector code will be produced.
  if (VPlanBuildStressTest || VectorizationFactor::Disabled() == VF)
    return false;

  VPlan &BestPlan = LVP.getPlanFor(VF.Width);

  {
    GeneratedRTChecks Checks(PSE, DT, LI, TTI, F->getDataLayout(), CM.CostKind);
    InnerLoopVectorizer LB(L, PSE, LI, DT, TTI, AC, VF.Width, /*UF=*/1, &CM,
                           BFI, PSI, Checks, BestPlan);
    LLVM_DEBUG(dbgs() << "Vectorizing outer loop in \""
                      << L->getHeader()->getParent()->getName() << "\"\n");
    LVP.addMinimumIterationCheck(BestPlan, VF.Width, /*UF=*/1,
                                 VF.MinProfitableTripCount);

    LVP.executePlan(VF.Width, /*UF=*/1, BestPlan, LB, DT, false);
  }

  reportVectorization(ORE, L, VF, 1);

  // Mark the loop as already vectorized to avoid vectorizing again.
  Hints.setAlreadyVectorized();
  assert(!verifyFunction(*L->getHeader()->getParent(), &dbgs()));
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
static bool isOutsideLoopWorkProfitable(GeneratedRTChecks &Checks,
                                        VectorizationFactor &VF, Loop *L,
                                        PredicatedScalarEvolution &PSE,
                                        VPCostContext &CostCtx, VPlan &Plan,
                                        ScalarEpilogueLowering SEL,
                                        std::optional<unsigned> VScale) {
  InstructionCost TotalCost = Checks.getCost();
  if (!TotalCost.isValid())
    return false;

  // Add on the cost of any work required in the vector early exit block, if
  // one exists.
  TotalCost += calculateEarlyExitCost(CostCtx, Plan, VF.Width);

  // When interleaving only scalar and vector cost will be equal, which in turn
  // would lead to a divide by 0. Fall back to hard threshold.
  if (VF.Width.isScalar()) {
    // TODO: Should we rename VectorizeMemoryCheckThreshold?
    if (TotalCost > VectorizeMemoryCheckThreshold) {
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

  // First, compute the minimum iteration count required so that the vector
  // loop outperforms the scalar loop.
  //  The total cost of the scalar loop is
  //   ScalarC * TC
  //  where
  //  * TC is the actual trip count of the loop.
  //  * ScalarC is the cost of a single scalar iteration.
  //
  //  The total cost of the vector loop is
  //    RtC + VecC * (TC / VF) + EpiC
  //  where
  //  * RtC is the cost of the generated runtime checks plus the cost of
  //    performing any additional work in the vector.early.exit block for loops
  //    with uncountable early exits.
  //  * VecC is the cost of a single vector iteration.
  //  * TC is the actual trip count of the loop
  //  * VF is the vectorization factor
  //  * EpiCost is the cost of the generated epilogue, including the cost
  //    of the remaining scalar operations.
  //
  // Vectorization is profitable once the total vector cost is less than the
  // total scalar cost:
  //   RtC + VecC * (TC / VF) + EpiC <  ScalarC * TC
  //
  // Now we can compute the minimum required trip count TC as
  //   VF * (RtC + EpiC) / (ScalarC * VF - VecC) < TC
  //
  // For now we assume the epilogue cost EpiC = 0 for simplicity. Note that
  // the computations are performed on doubles, not integers and the result
  // is rounded up, hence we get an upper estimate of the TC.
  unsigned IntVF = estimateElementCount(VF.Width, VScale);
  uint64_t RtC = TotalCost.getValue();
  uint64_t Div = ScalarC * IntVF - VF.Cost.getValue();
  uint64_t MinTC1 = Div == 0 ? 0 : divideCeil(RtC * IntVF, Div);

  // Second, compute a minimum iteration count so that the cost of the
  // runtime checks is only a fraction of the total scalar loop cost. This
  // adds a loop-dependent bound on the overhead incurred if the runtime
  // checks fail. In case the runtime checks fail, the cost is RtC + ScalarC
  // * TC. To bound the runtime check to be a fraction 1/X of the scalar
  // cost, compute
  //   RtC < ScalarC * TC * (1 / X)  ==>  RtC * X / ScalarC < TC
  uint64_t MinTC2 = divideCeil(RtC * 10, ScalarC);

  // Now pick the larger minimum. If it is not a multiple of VF and a scalar
  // epilogue is allowed, choose the next closest multiple of VF. This should
  // partly compensate for ignoring the epilogue cost.
  uint64_t MinTC = std::max(MinTC1, MinTC2);
  if (SEL == CM_ScalarEpilogueAllowed)
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
/// vectorization. Remove ResumePhis from \p MainPlan for inductions that
/// don't have a corresponding wide induction in \p EpiPlan.
static void preparePlanForMainVectorLoop(VPlan &MainPlan, VPlan &EpiPlan) {
  // Collect PHI nodes of widened phis in the VPlan for the epilogue. Those
  // will need their resume-values computed in the main vector loop. Others
  // can be removed from the main VPlan.
  SmallPtrSet<PHINode *, 2> EpiWidenedPhis;
  for (VPRecipeBase &R :
       EpiPlan.getVectorLoopRegion()->getEntryBasicBlock()->phis()) {
    if (isa<VPCanonicalIVPHIRecipe>(&R))
      continue;
    EpiWidenedPhis.insert(
        cast<PHINode>(R.getVPSingleValue()->getUnderlyingValue()));
  }
  for (VPRecipeBase &R :
       make_early_inc_range(MainPlan.getScalarHeader()->phis())) {
    auto *VPIRInst = cast<VPIRPhi>(&R);
    if (EpiWidenedPhis.contains(&VPIRInst->getIRPhi()))
      continue;
    // There is no corresponding wide induction in the epilogue plan that would
    // need a resume value. Remove the VPIRInst wrapping the scalar header phi
    // together with the corresponding ResumePhi. The resume values for the
    // scalar loop will be created during execution of EpiPlan.
    VPRecipeBase *ResumePhi = VPIRInst->getOperand(0)->getDefiningRecipe();
    VPIRInst->eraseFromParent();
    ResumePhi->eraseFromParent();
  }
  VPlanTransforms::runPass(VPlanTransforms::removeDeadRecipes, MainPlan);

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
      if (!VPI || VPI->getOpcode() != VPInstruction::ComputeFindIVResult)
        continue;
      VPValue *OrigStart = VPI->getOperand(1);
      if (isGuaranteedNotToBeUndefOrPoison(OrigStart->getLiveInIRValue()))
        continue;
      VPInstruction *Freeze =
          Builder.createNaryOp(Instruction::Freeze, {OrigStart}, {}, "fr");
      VPI->setOperand(1, Freeze);
      if (UpdateResumePhis)
        OrigStart->replaceUsesWithIf(Freeze, [Freeze](VPUser &U, unsigned) {
          return Freeze != &U && isa<VPPhi>(&U);
        });
    }
  };
  AddFreezeForFindLastIVReductions(MainPlan, true);
  AddFreezeForFindLastIVReductions(EpiPlan, false);

  VPBasicBlock *MainScalarPH = MainPlan.getScalarPreheader();
  VPValue *VectorTC = &MainPlan.getVectorTripCount();
  // If there is a suitable resume value for the canonical induction in the
  // scalar (which will become vector) epilogue loop, use it and move it to the
  // beginning of the scalar preheader. Otherwise create it below.
  auto ResumePhiIter =
      find_if(MainScalarPH->phis(), [VectorTC](VPRecipeBase &R) {
        return match(&R, m_VPInstruction<Instruction::PHI>(m_Specific(VectorTC),
                                                           m_SpecificInt(0)));
      });
  VPPhi *ResumePhi = nullptr;
  if (ResumePhiIter == MainScalarPH->phis().end()) {
    VPBuilder ScalarPHBuilder(MainScalarPH, MainScalarPH->begin());
    ResumePhi = ScalarPHBuilder.createScalarPhi(
        {VectorTC, MainPlan.getCanonicalIV()->getStartValue()}, {},
        "vec.epilog.resume.val");
  } else {
    ResumePhi = cast<VPPhi>(&*ResumePhiIter);
    if (MainScalarPH->begin() == MainScalarPH->end())
      ResumePhi->moveBefore(*MainScalarPH, MainScalarPH->end());
    else if (&*MainScalarPH->begin() != ResumePhi)
      ResumePhi->moveBefore(*MainScalarPH, MainScalarPH->begin());
  }
  // Add a user to to make sure the resume phi won't get removed.
  VPBuilder(MainScalarPH)
      .createNaryOp(VPInstruction::ResumeForEpilogue, ResumePhi);
}

/// Prepare \p Plan for vectorizing the epilogue loop. That is, re-use expanded
/// SCEVs from \p ExpandedSCEVs and set resume values for header recipes.
static void
preparePlanForEpilogueVectorLoop(VPlan &Plan, Loop *L,
                                 const SCEV2ValueTy &ExpandedSCEVs,
                                 EpilogueLoopVectorizationInfo &EPI) {
  VPRegionBlock *VectorLoop = Plan.getVectorLoopRegion();
  VPBasicBlock *Header = VectorLoop->getEntryBasicBlock();
  Header->setName("vec.epilog.vector.body");

  DenseMap<Value *, Value *> ToFrozen;
  // Ensure that the start values for all header phi recipes are updated before
  // vectorizing the epilogue loop.
  for (VPRecipeBase &R : Header->phis()) {
    if (auto *IV = dyn_cast<VPCanonicalIVPHIRecipe>(&R)) {
      // When vectorizing the epilogue loop, the canonical induction start
      // value needs to be changed from zero to the value after the main
      // vector loop. Find the resume value created during execution of the main
      // VPlan. It must be the first phi in the loop preheader.
      // FIXME: Improve modeling for canonical IV start values in the epilogue
      // loop.
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
        assert(
            EPResumeVal->getNumIncomingValues() > 0 &&
            all_of(EPResumeVal->incoming_values(),
                   [](Value *Inc) { return match(Inc, m_SpecificInt(0)); }) &&
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
      IV->setOperand(0, VPV);
      continue;
    }

    Value *ResumeV = nullptr;
    // TODO: Move setting of resume values to prepareToExecute.
    if (auto *ReductionPhi = dyn_cast<VPReductionPHIRecipe>(&R)) {
      auto *RdxResult =
          cast<VPInstruction>(*find_if(ReductionPhi->users(), [](VPUser *U) {
            auto *VPI = dyn_cast<VPInstruction>(U);
            return VPI &&
                   (VPI->getOpcode() == VPInstruction::ComputeAnyOfResult ||
                    VPI->getOpcode() == VPInstruction::ComputeReductionResult ||
                    VPI->getOpcode() == VPInstruction::ComputeFindIVResult);
          }));
      ResumeV = cast<PHINode>(ReductionPhi->getUnderlyingInstr())
                    ->getIncomingValueForBlock(L->getLoopPreheader());
      RecurKind RK = ReductionPhi->getRecurrenceKind();
      if (RecurrenceDescriptor::isAnyOfRecurrenceKind(RK)) {
        Value *StartV = RdxResult->getOperand(1)->getLiveInIRValue();
        // VPReductionPHIRecipes for AnyOf reductions expect a boolean as
        // start value; compare the final value from the main vector loop
        // to the start value.
        BasicBlock *PBB = cast<Instruction>(ResumeV)->getParent();
        IRBuilder<> Builder(PBB, PBB->getFirstNonPHIIt());
        ResumeV = Builder.CreateICmpNE(ResumeV, StartV);
      } else if (RecurrenceDescriptor::isFindIVRecurrenceKind(RK)) {
        Value *StartV = getStartValueFromReductionResult(RdxResult);
        ToFrozen[StartV] = cast<PHINode>(ResumeV)->getIncomingValueForBlock(
            EPI.MainLoopIterationCountCheck);

        // VPReductionPHIRecipe for FindFirstIV/FindLastIV reductions requires
        // an adjustment to the resume value. The resume value is adjusted to
        // the sentinel value when the final value from the main vector loop
        // equals the start value. This ensures correctness when the start value
        // might not be less than the minimum value of a monotonically
        // increasing induction variable.
        BasicBlock *ResumeBB = cast<Instruction>(ResumeV)->getParent();
        IRBuilder<> Builder(ResumeBB, ResumeBB->getFirstNonPHIIt());
        Value *Cmp = Builder.CreateICmpEQ(ResumeV, ToFrozen[StartV]);
        Value *Sentinel = RdxResult->getOperand(2)->getLiveInIRValue();
        ResumeV = Builder.CreateSelect(Cmp, Sentinel, ResumeV);
      } else {
        VPValue *StartVal = Plan.getOrAddLiveIn(ResumeV);
        auto *PhiR = dyn_cast<VPReductionPHIRecipe>(&R);
        if (auto *VPI = dyn_cast<VPInstruction>(PhiR->getStartValue())) {
          assert(VPI->getOpcode() == VPInstruction::ReductionStartVector &&
                 "unexpected start value");
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
}

// Generate bypass values from the additional bypass block. Note that when the
// vectorized epilogue is skipped due to iteration count check, then the
// resume value for the induction variable comes from the trip count of the
// main vector loop, passed as the second argument.
static Value *createInductionAdditionalBypassValues(
    PHINode *OrigPhi, const InductionDescriptor &II, IRBuilder<> &BypassBuilder,
    const SCEV2ValueTy &ExpandedSCEVs, Value *MainVectorTripCount,
    Instruction *OldInduction) {
  Value *Step = getExpandedStep(II, ExpandedSCEVs);
  // For the primary induction the additional bypass end value is known.
  // Otherwise it is computed.
  Value *EndValueFromAdditionalBypass = MainVectorTripCount;
  if (OrigPhi != OldInduction) {
    auto *BinOp = II.getInductionBinOp();
    // Fast-math-flags propagate from the original induction instruction.
    if (isa_and_nonnull<FPMathOperator>(BinOp))
      BypassBuilder.setFastMathFlags(BinOp->getFastMathFlags());

    // Compute the end value for the additional bypass.
    EndValueFromAdditionalBypass =
        emitTransformedIndex(BypassBuilder, MainVectorTripCount,
                             II.getStartValue(), Step, II.getKind(), BinOp);
    EndValueFromAdditionalBypass->setName("ind.end");
  }
  return EndValueFromAdditionalBypass;
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

  // Check if it is legal to vectorize the loop.
  LoopVectorizationRequirements Requirements;
  LoopVectorizationLegality LVL(L, PSE, DT, TTI, TLI, F, *LAIs, LI, ORE,
                                &Requirements, &Hints, DB, AC, BFI, PSI);
  if (!LVL.canVectorize(EnableVPlanNativePath)) {
    LLVM_DEBUG(dbgs() << "LV: Not vectorizing: Cannot prove legality.\n");
    Hints.emitRemarkWithHints();
    return false;
  }

  if (LVL.hasUncountableEarlyExit() && !EnableEarlyExitVectorization) {
    reportVectorizationFailure("Auto-vectorization of loops with uncountable "
                               "early exit is not enabled",
                               "UncountableEarlyExitLoopsDisabled", ORE, L);
    return false;
  }

  // Entrance to the VPlan-native vectorization path. Outer loops are processed
  // here. They may require CFG and instruction level transformations before
  // even evaluating whether vectorization is profitable. Since we cannot modify
  // the incoming IR, we need to build VPlan upfront in the vectorization
  // pipeline.
  if (!L->isInnermost())
    return processLoopInVPlanNativePath(L, PSE, LI, DT, &LVL, TTI, TLI, DB, AC,
                                        ORE, BFI, PSI, Hints, Requirements);

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
        any_of(LVL.getCountableExitingBlocks(),
               [LoopLatch](BasicBlock *BB) { return BB != LoopLatch; })) {
      reportVectorizationFailure("Auto-vectorization of early exit loops "
                                 "requiring a scalar epilogue is unsupported",
                                 "UncountableEarlyExitUnsupported", ORE, L);
      return false;
    }
  }

  // Check the function attributes and profiles to find out if this function
  // should be optimized for size.
  ScalarEpilogueLowering SEL =
      getScalarEpilogueLowering(F, L, Hints, PSI, BFI, TTI, TLI, LVL, &IAI);

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
      // Predicate tail-folded loops are efficient even when the loop
      // iteration count is low. However, setting the epilogue policy to
      // `CM_ScalarEpilogueNotAllowedLowTripLoop` prevents vectorizing loops
      // with runtime checks. It's more effective to let
      // `isOutsideLoopWorkProfitable` determine if vectorization is
      // beneficial for the loop.
      if (SEL != CM_ScalarEpilogueNotNeededUsePredicate)
        SEL = CM_ScalarEpilogueNotAllowedLowTripLoop;
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
  LoopVectorizationCostModel CM(SEL, L, PSE, LI, &LVL, *TTI, TLI, DB, AC, ORE,
                                F, &Hints, IAI, PSI, BFI);
  // Use the planner for vectorization.
  LoopVectorizationPlanner LVP(L, LI, DT, TLI, *TTI, &LVL, CM, IAI, PSE, Hints,
                               ORE);

  // Get user vectorization factor and interleave count.
  ElementCount UserVF = Hints.getWidth();
  unsigned UserIC = Hints.getInterleave();

  // Plan how to best vectorize.
  LVP.plan(UserVF, UserIC);
  VectorizationFactor VF = LVP.computeBestVF();
  unsigned IC = 1;

  if (ORE->allowExtraAnalysis(LV_NAME))
    LVP.emitInvalidCostRemarks(ORE);

  GeneratedRTChecks Checks(PSE, DT, LI, TTI, F->getDataLayout(), CM.CostKind);
  if (LVP.hasPlanWithVF(VF.Width)) {
    // Select the interleave count.
    IC = LVP.selectInterleaveCount(LVP.getPlanFor(VF.Width), VF.Width, VF.Cost);

    unsigned SelectedIC = std::max(IC, UserIC);
    //  Optimistically generate runtime checks if they are needed. Drop them if
    //  they turn out to not be profitable.
    if (VF.Width.isVector() || SelectedIC > 1) {
      Checks.create(L, *LVL.getLAI(), PSE.getPredicate(), VF.Width, SelectedIC);

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
    VPCostContext CostCtx(CM.TTI, *CM.TLI, LVP.getPlanFor(VF.Width), CM,
                          CM.CostKind);
    if (!ForceVectorization &&
        !isOutsideLoopWorkProfitable(Checks, VF, L, PSE, CostCtx,
                                     LVP.getPlanFor(VF.Width), SEL,
                                     CM.getVScaleForTuning())) {
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

  if (!LVP.hasPlanWithVF(VF.Width) && UserIC > 1) {
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

  bool DisableRuntimeUnroll = false;
  MDNode *OrigLoopID = L->getLoopID();

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

  VPlan &BestPlan = LVP.getPlanFor(VF.Width);
  // Consider vectorizing the epilogue too if it's profitable.
  VectorizationFactor EpilogueVF =
      LVP.selectEpilogueVectorizationFactor(VF.Width, IC);
  if (EpilogueVF.Width.isVector()) {
    std::unique_ptr<VPlan> BestMainPlan(BestPlan.duplicate());

    // The first pass vectorizes the main loop and creates a scalar epilogue
    // to be vectorized by executing the plan (potentially with a different
    // factor) again shortly afterwards.
    VPlan &BestEpiPlan = LVP.getPlanFor(EpilogueVF.Width);
    BestEpiPlan.getMiddleBlock()->setName("vec.epilog.middle.block");
    preparePlanForMainVectorLoop(*BestMainPlan, BestEpiPlan);
    EpilogueLoopVectorizationInfo EPI(VF.Width, IC, EpilogueVF.Width, 1,
                                      BestEpiPlan);
    EpilogueVectorizerMainLoop MainILV(L, PSE, LI, DT, TTI, AC, EPI, &CM, BFI,
                                       PSI, Checks, *BestMainPlan);
    auto ExpandedSCEVs = LVP.executePlan(EPI.MainLoopVF, EPI.MainLoopUF,
                                         *BestMainPlan, MainILV, DT, false);
    ++LoopsVectorized;

    // Second pass vectorizes the epilogue and adjusts the control flow
    // edges from the first pass.
    EpilogueVectorizerEpilogueLoop EpilogILV(L, PSE, LI, DT, TTI, AC, EPI, &CM,
                                             BFI, PSI, Checks, BestEpiPlan);
    EpilogILV.setTripCount(MainILV.getTripCount());
    preparePlanForEpilogueVectorLoop(BestEpiPlan, L, ExpandedSCEVs, EPI);

    LVP.executePlan(EPI.EpilogueVF, EPI.EpilogueUF, BestEpiPlan, EpilogILV, DT,
                    true);

    // Fix induction resume values from the additional bypass block.
    BasicBlock *BypassBlock = EpilogILV.getAdditionalBypassBlock();
    IRBuilder<> BypassBuilder(BypassBlock, BypassBlock->getFirstInsertionPt());
    BasicBlock *PH = L->getLoopPreheader();
    for (const auto &[IVPhi, II] : LVL.getInductionVars()) {
      auto *Inc = cast<PHINode>(IVPhi->getIncomingValueForBlock(PH));
      Value *V = createInductionAdditionalBypassValues(
          IVPhi, II, BypassBuilder, ExpandedSCEVs, EPI.VectorTripCount,
          LVL.getPrimaryInduction());
      // TODO: Directly add as extra operand to the VPResumePHI recipe.
      Inc->setIncomingValueForBlock(BypassBlock, V);
    }
    ++LoopsEpilogueVectorized;

    if (!Checks.hasChecks())
      DisableRuntimeUnroll = true;
  } else {
    InnerLoopVectorizer LB(L, PSE, LI, DT, TTI, AC, VF.Width, IC, &CM, BFI, PSI,
                           Checks, BestPlan);
    // TODO: Move to general VPlan pipeline once epilogue loops are also
    // supported.
    VPlanTransforms::runPass(
        VPlanTransforms::materializeConstantVectorTripCount, BestPlan, VF.Width,
        IC, PSE);
    LVP.addMinimumIterationCheck(BestPlan, VF.Width, IC,
                                 VF.MinProfitableTripCount);

    LVP.executePlan(VF.Width, IC, BestPlan, LB, DT, false);
    ++LoopsVectorized;

    // Add metadata to disable runtime unrolling a scalar loop when there
    // are no runtime checks about strides and memory. A scalar loop that is
    // rarely used is not worth unrolling.
    if (!Checks.hasChecks() && !VF.Width.isScalar())
      DisableRuntimeUnroll = true;
  }

  assert(DT->verify(DominatorTree::VerificationLevel::Fast) &&
         "DT not preserved correctly");

  std::optional<MDNode *> RemainderLoopID =
      makeFollowupLoopID(OrigLoopID, {LLVMLoopVectorizeFollowupAll,
                                      LLVMLoopVectorizeFollowupEpilogue});
  if (RemainderLoopID) {
    L->setLoopID(*RemainderLoopID);
  } else {
    if (DisableRuntimeUnroll)
      addRuntimeUnrollDisableMetaData(L);

    // Mark the loop as already vectorized to avoid vectorizing again.
    Hints.setAlreadyVectorized();
  }

  assert(!verifyFunction(*L->getHeader()->getParent(), &dbgs()));
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

  auto &MAMProxy = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
  PSI = MAMProxy.getCachedResult<ProfileSummaryAnalysis>(*F.getParent());
  BFI = nullptr;
  if (PSI && PSI->hasProfileSummary())
    BFI = &AM.getResult<BlockFrequencyAnalysis>(F);
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
