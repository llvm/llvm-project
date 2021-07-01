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
// docs/Proposal/VectorizationPlan.rst and
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
#include "VPlanHCFGBuilder.h"
#include "VPlanPredicator.h"
#include "VPlanTransforms.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
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
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
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
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/InjectTLIMappings.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
#include "llvm/Transforms/Vectorize/LoopVectorizationLegality.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

using namespace llvm;

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

static cl::opt<bool> EnableEpilogueVectorization(
    "enable-epilogue-vectorization", cl::init(true), cl::Hidden,
    cl::desc("Enable vectorization of epilogue loops."));

static cl::opt<unsigned> EpilogueVectorizationForceVF(
    "epilogue-vectorization-force-VF", cl::init(1), cl::Hidden,
    cl::desc("When epilogue vectorization is enabled, and a value greater than "
             "1 is specified, forces the given VF for all applicable epilogue "
             "loops."));

static cl::opt<unsigned> EpilogueVectorizationMinVF(
    "epilogue-vectorization-minimum-VF", cl::init(16), cl::Hidden,
    cl::desc("Only loops with vectorization factor equal to or larger than "
             "the specified value are considered for epilogue vectorization."));

/// Loops with a known constant trip count below this number are vectorized only
/// if no scalar iteration overheads are incurred.
static cl::opt<unsigned> TinyTripCountVectorThreshold(
    "vectorizer-min-trip-count", cl::init(16), cl::Hidden,
    cl::desc("Loops with a constant trip count that is smaller than this "
             "value are vectorized only if no scalar iteration overheads "
             "are incurred."));

static cl::opt<unsigned> PragmaVectorizeMemoryCheckThreshold(
    "pragma-vectorize-memory-check-threshold", cl::init(128), cl::Hidden,
    cl::desc("The maximum allowed number of runtime memory checks with a "
             "vectorize(enable) pragma."));

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

static cl::opt<unsigned> TinyTripCountInterleaveThreshold(
    "tiny-trip-count-interleave-threshold", cl::init(128), cl::Hidden,
    cl::desc("We don't interleave loops with a estimated constant trip count "
             "below this number"));

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

static cl::opt<unsigned> ForceTargetInstructionCost(
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

/// Interleave small loops with scalar reductions.
static cl::opt<bool> InterleaveSmallLoopScalarReduction(
    "interleave-small-loop-scalar-reduction", cl::init(false), cl::Hidden,
    cl::desc("Enable interleaving for loops with small iteration counts that "
             "contain scalar reductions to expose ILP."));

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

cl::opt<bool> EnableStrictReductions(
    "enable-strict-reductions", cl::init(false), cl::Hidden,
    cl::desc("Enable the vectorisation of loops with in-order (strict) "
             "FP reductions"));

static cl::opt<bool> PreferPredicatedReductionSelect(
    "prefer-predicated-reduction-select", cl::init(false), cl::Hidden,
    cl::desc(
        "Prefer predicating a reduction operation over an after loop select."));

cl::opt<bool> EnableVPlanNativePath(
    "enable-vplan-native-path", cl::init(false), cl::Hidden,
    cl::desc("Enable VPlan-native vectorization path with "
             "support for outer loop vectorization."));

// FIXME: Remove this switch once we have divergence analysis. Currently we
// assume divergent non-backedge branches when this switch is true.
cl::opt<bool> EnableVPlanPredication(
    "enable-vplan-predication", cl::init(false), cl::Hidden,
    cl::desc("Enable VPlan-native vectorization path predicator with "
             "support for outer loop vectorization."));

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

cl::opt<bool> PrintVPlansInDotFormat(
    "vplan-print-in-dot-format", cl::init(false), cl::Hidden,
    cl::desc("Use dot format instead of plain text when dumping VPlans"));

/// A helper function that returns true if the given type is irregular. The
/// type is irregular if its allocated size doesn't equal the store size of an
/// element of the corresponding vector type.
static bool hasIrregularType(Type *Ty, const DataLayout &DL) {
  // Determine if an array of N elements of type Ty is "bitcast compatible"
  // with a <N x Ty> vector.
  // This is only true if there is no padding between the array elements.
  return DL.getTypeAllocSizeInBits(Ty) != DL.getTypeSizeInBits(Ty);
}

/// A helper function that returns the reciprocal of the block probability of
/// predicated blocks. If we return X, we are assuming the predicated block
/// will execute once for every X iterations of the loop header.
///
/// TODO: We should use actual block probability here, if available. Currently,
///       we always assume predicated blocks have a 50% chance of executing.
static unsigned getReciprocalPredBlockProb() { return 2; }

/// A helper function that returns an integer or floating-point constant with
/// value C.
static Constant *getSignedIntOrFpConstant(Type *Ty, int64_t C) {
  return Ty->isIntegerTy() ? ConstantInt::getSigned(Ty, C)
                           : ConstantFP::get(Ty, C);
}

/// Returns "best known" trip count for the specified loop \p L as defined by
/// the following procedure:
///   1) Returns exact trip count if it is known.
///   2) Returns expected trip count according to profile data if any.
///   3) Returns upper bound estimate if it is known.
///   4) Returns None if all of the above failed.
static Optional<unsigned> getSmallBestKnownTC(ScalarEvolution &SE, Loop *L) {
  // Check if exact trip count is known.
  if (unsigned ExpectedTC = SE.getSmallConstantTripCount(L))
    return ExpectedTC;

  // Check if there is an expected trip count available from profile data.
  if (LoopVectorizeWithBlockFrequency)
    if (auto EstimatedTC = getLoopEstimatedTripCount(L))
      return EstimatedTC;

  // Check if upper bound estimate is known.
  if (unsigned ExpectedTC = SE.getSmallConstantMaxTripCount(L))
    return ExpectedTC;

  return None;
}

// Forward declare GeneratedRTChecks.
class GeneratedRTChecks;

namespace llvm {

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
                      const TargetLibraryInfo *TLI,
                      const TargetTransformInfo *TTI, AssumptionCache *AC,
                      OptimizationRemarkEmitter *ORE, ElementCount VecWidth,
                      unsigned UnrollFactor, LoopVectorizationLegality *LVL,
                      LoopVectorizationCostModel *CM, BlockFrequencyInfo *BFI,
                      ProfileSummaryInfo *PSI, GeneratedRTChecks &RTChecks)
      : OrigLoop(OrigLoop), PSE(PSE), LI(LI), DT(DT), TLI(TLI), TTI(TTI),
        AC(AC), ORE(ORE), VF(VecWidth), UF(UnrollFactor),
        Builder(PSE.getSE()->getContext()), Legal(LVL), Cost(CM), BFI(BFI),
        PSI(PSI), RTChecks(RTChecks) {
    // Query this against the original loop and save it here because the profile
    // of the original loop header may change as the transformation happens.
    OptForSizeBasedOnProfile = llvm::shouldOptimizeForSize(
        OrigLoop->getHeader(), PSI, BFI, PGSOQueryType::IRPass);
  }

  virtual ~InnerLoopVectorizer() = default;

  /// Create a new empty loop that will contain vectorized instructions later
  /// on, while the old loop will be used as the scalar remainder. Control flow
  /// is generated around the vectorized (and scalar epilogue) loops consisting
  /// of various checks and bypasses. Return the pre-header block of the new
  /// loop.
  /// In the case of epilogue vectorization, this function is overriden to
  /// handle the more complex control flow around the loops.
  virtual BasicBlock *createVectorizedLoopSkeleton();

  /// Widen a single instruction within the innermost loop.
  void widenInstruction(Instruction &I, VPValue *Def, VPUser &Operands,
                        VPTransformState &State);

  /// Widen a single call instruction within the innermost loop.
  void widenCallInstruction(CallInst &I, VPValue *Def, VPUser &ArgOperands,
                            VPTransformState &State);

  /// Widen a single select instruction within the innermost loop.
  void widenSelectInstruction(SelectInst &I, VPValue *VPDef, VPUser &Operands,
                              bool InvariantCond, VPTransformState &State);

  /// Fix the vectorized code, taking care of header phi's, live-outs, and more.
  void fixVectorizedLoop(VPTransformState &State);

  // Return true if any runtime check is added.
  bool areSafetyChecksAdded() { return AddedSafetyChecks; }

  /// A type for vectorized values in the new loop. Each value from the
  /// original loop, when vectorized, is represented by UF vector values in the
  /// new unrolled loop, where UF is the unroll factor.
  using VectorParts = SmallVector<Value *, 2>;

  /// Vectorize a single GetElementPtrInst based on information gathered and
  /// decisions taken during planning.
  void widenGEP(GetElementPtrInst *GEP, VPValue *VPDef, VPUser &Indices,
                unsigned UF, ElementCount VF, bool IsPtrLoopInvariant,
                SmallBitVector &IsIndexLoopInvariant, VPTransformState &State);

  /// Vectorize a single PHINode in a block. This method handles the induction
  /// variable canonicalization. It supports both VF = 1 for unrolled loops and
  /// arbitrary length vectors.
  void widenPHIInstruction(Instruction *PN, RecurrenceDescriptor *RdxDesc,
                           VPWidenPHIRecipe *PhiR, VPTransformState &State);

  /// A helper function to scalarize a single Instruction in the innermost loop.
  /// Generates a sequence of scalar instances for each lane between \p MinLane
  /// and \p MaxLane, times each part between \p MinPart and \p MaxPart,
  /// inclusive. Uses the VPValue operands from \p Operands instead of \p
  /// Instr's operands.
  void scalarizeInstruction(Instruction *Instr, VPValue *Def, VPUser &Operands,
                            const VPIteration &Instance, bool IfPredicateInstr,
                            VPTransformState &State);

  /// Widen an integer or floating-point induction variable \p IV. If \p Trunc
  /// is provided, the integer induction variable will first be truncated to
  /// the corresponding type.
  void widenIntOrFpInduction(PHINode *IV, Value *Start, TruncInst *Trunc,
                             VPValue *Def, VPValue *CastDef,
                             VPTransformState &State);

  /// Construct the vector value of a scalarized value \p V one lane at a time.
  void packScalarIntoVectorValue(VPValue *Def, const VPIteration &Instance,
                                 VPTransformState &State);

  /// Try to vectorize interleaved access group \p Group with the base address
  /// given in \p Addr, optionally masking the vector operations if \p
  /// BlockInMask is non-null. Use \p State to translate given VPValues to IR
  /// values in the vectorized loop.
  void vectorizeInterleaveGroup(const InterleaveGroup<Instruction> *Group,
                                ArrayRef<VPValue *> VPDefs,
                                VPTransformState &State, VPValue *Addr,
                                ArrayRef<VPValue *> StoredValues,
                                VPValue *BlockInMask = nullptr);

  /// Vectorize Load and Store instructions with the base address given in \p
  /// Addr, optionally masking the vector operations if \p BlockInMask is
  /// non-null. Use \p State to translate given VPValues to IR values in the
  /// vectorized loop.
  void vectorizeMemoryInstruction(Instruction *Instr, VPTransformState &State,
                                  VPValue *Def, VPValue *Addr,
                                  VPValue *StoredValue, VPValue *BlockInMask);

  /// Set the debug location in the builder \p Ptr using the debug location in
  /// \p V. If \p Ptr is None then it uses the class member's Builder.
  void setDebugLocFromInst(const Value *V,
                           Optional<IRBuilder<> *> CustomBuilder = None);

  /// Fix the non-induction PHIs in the OrigPHIsToFix vector.
  void fixNonInductionPHIs(VPTransformState &State);

  /// Returns true if the reordering of FP operations is not allowed, but we are
  /// able to vectorize with strict in-order reductions for the given RdxDesc.
  bool useOrderedReductions(RecurrenceDescriptor &RdxDesc);

  /// Create a broadcast instruction. This method generates a broadcast
  /// instruction (shuffle) for loop invariant values and for the induction
  /// value. If this is the induction variable then we extend it to N, N+1, ...
  /// this is needed because each iteration in the loop corresponds to a SIMD
  /// element.
  virtual Value *getBroadcastInstrs(Value *V);

protected:
  friend class LoopVectorizationPlanner;

  /// A small list of PHINodes.
  using PhiVector = SmallVector<PHINode *, 4>;

  /// A type for scalarized values in the new loop. Each value from the
  /// original loop, when scalarized, is represented by UF x VF scalar values
  /// in the new unrolled loop, where UF is the unroll factor and VF is the
  /// vectorization factor.
  using ScalarParts = SmallVector<SmallVector<Value *, 4>, 2>;

  /// Set up the values of the IVs correctly when exiting the vector loop.
  void fixupIVUsers(PHINode *OrigPhi, const InductionDescriptor &II,
                    Value *CountRoundDown, Value *EndValue,
                    BasicBlock *MiddleBlock);

  /// Create a new induction variable inside L.
  PHINode *createInductionVariable(Loop *L, Value *Start, Value *End,
                                   Value *Step, Instruction *DL);

  /// Handle all cross-iteration phis in the header.
  void fixCrossIterationPHIs(VPTransformState &State);

  /// Fix a first-order recurrence. This is the second phase of vectorizing
  /// this phi node.
  void fixFirstOrderRecurrence(VPWidenPHIRecipe *PhiR, VPTransformState &State);

  /// Fix a reduction cross-iteration phi. This is the second phase of
  /// vectorizing this phi node.
  void fixReduction(VPWidenPHIRecipe *Phi, VPTransformState &State);

  /// Clear NSW/NUW flags from reduction instructions if necessary.
  void clearReductionWrapFlags(const RecurrenceDescriptor &RdxDesc,
                               VPTransformState &State);

  /// Fixup the LCSSA phi nodes in the unique exit block.  This simply
  /// means we need to add the appropriate incoming value from the middle
  /// block as exiting edges from the scalar epilogue loop (if present) are
  /// already in place, and we exit the vector loop exclusively to the middle
  /// block.
  void fixLCSSAPHIs(VPTransformState &State);

  /// Iteratively sink the scalarized operands of a predicated instruction into
  /// the block that was created for it.
  void sinkScalarOperands(Instruction *PredInst);

  /// Shrinks vector element sizes to the smallest bitwidth they can be legally
  /// represented as.
  void truncateToMinimalBitwidths(VPTransformState &State);

  /// This function adds
  /// (StartIdx * Step, (StartIdx + 1) * Step, (StartIdx + 2) * Step, ...)
  /// to each vector element of Val. The sequence starts at StartIndex.
  /// \p Opcode is relevant for FP induction variable.
  virtual Value *getStepVector(Value *Val, int StartIdx, Value *Step,
                               Instruction::BinaryOps Opcode =
                               Instruction::BinaryOpsEnd);

  /// Compute scalar induction steps. \p ScalarIV is the scalar induction
  /// variable on which to base the steps, \p Step is the size of the step, and
  /// \p EntryVal is the value from the original loop that maps to the steps.
  /// Note that \p EntryVal doesn't have to be an induction variable - it
  /// can also be a truncate instruction.
  void buildScalarSteps(Value *ScalarIV, Value *Step, Instruction *EntryVal,
                        const InductionDescriptor &ID, VPValue *Def,
                        VPValue *CastDef, VPTransformState &State);

  /// Create a vector induction phi node based on an existing scalar one. \p
  /// EntryVal is the value from the original loop that maps to the vector phi
  /// node, and \p Step is the loop-invariant step. If \p EntryVal is a
  /// truncate instruction, instead of widening the original IV, we widen a
  /// version of the IV truncated to \p EntryVal's type.
  void createVectorIntOrFpInductionPHI(const InductionDescriptor &II,
                                       Value *Step, Value *Start,
                                       Instruction *EntryVal, VPValue *Def,
                                       VPValue *CastDef,
                                       VPTransformState &State);

  /// Returns true if an instruction \p I should be scalarized instead of
  /// vectorized for the chosen vectorization factor.
  bool shouldScalarizeInstruction(Instruction *I) const;

  /// Returns true if we should generate a scalar version of \p IV.
  bool needsScalarInduction(Instruction *IV) const;

  /// If there is a cast involved in the induction variable \p ID, which should
  /// be ignored in the vectorized loop body, this function records the
  /// VectorLoopValue of the respective Phi also as the VectorLoopValue of the
  /// cast. We had already proved that the casted Phi is equal to the uncasted
  /// Phi in the vectorized loop (under a runtime guard), and therefore
  /// there is no need to vectorize the cast - the same value can be used in the
  /// vector loop for both the Phi and the cast.
  /// If \p VectorLoopValue is a scalarized value, \p Lane is also specified,
  /// Otherwise, \p VectorLoopValue is a widened/vectorized value.
  ///
  /// \p EntryVal is the value from the original loop that maps to the vector
  /// phi node and is used to distinguish what is the IV currently being
  /// processed - original one (if \p EntryVal is a phi corresponding to the
  /// original IV) or the "newly-created" one based on the proof mentioned above
  /// (see also buildScalarSteps() and createVectorIntOrFPInductionPHI()). In the
  /// latter case \p EntryVal is a TruncInst and we must not record anything for
  /// that IV, but it's error-prone to expect callers of this routine to care
  /// about that, hence this explicit parameter.
  void recordVectorLoopValueForInductionCast(
      const InductionDescriptor &ID, const Instruction *EntryVal,
      Value *VectorLoopValue, VPValue *CastDef, VPTransformState &State,
      unsigned Part, unsigned Lane = UINT_MAX);

  /// Generate a shuffle sequence that will reverse the vector Vec.
  virtual Value *reverseVector(Value *Vec);

  /// Returns (and creates if needed) the original loop trip count.
  Value *getOrCreateTripCount(Loop *NewLoop);

  /// Returns (and creates if needed) the trip count of the widened loop.
  Value *getOrCreateVectorTripCount(Loop *NewLoop);

  /// Returns a bitcasted value to the requested vector type.
  /// Also handles bitcasts of vector<float> <-> vector<pointer> types.
  Value *createBitOrPointerCast(Value *V, VectorType *DstVTy,
                                const DataLayout &DL);

  /// Emit a bypass check to see if the vector trip count is zero, including if
  /// it overflows.
  void emitMinimumIterationCountCheck(Loop *L, BasicBlock *Bypass);

  /// Emit a bypass check to see if all of the SCEV assumptions we've
  /// had to make are correct. Returns the block containing the checks or
  /// nullptr if no checks have been added.
  BasicBlock *emitSCEVChecks(Loop *L, BasicBlock *Bypass);

  /// Emit bypass checks to check any memory assumptions we may have made.
  /// Returns the block containing the checks or nullptr if no checks have been
  /// added.
  BasicBlock *emitMemRuntimeChecks(Loop *L, BasicBlock *Bypass);

  /// Compute the transformed value of Index at offset StartValue using step
  /// StepValue.
  /// For integer induction, returns StartValue + Index * StepValue.
  /// For pointer induction, returns StartValue[Index * StepValue].
  /// FIXME: The newly created binary instructions should contain nsw/nuw
  /// flags, which can be found from the original scalar operations.
  Value *emitTransformedIndex(IRBuilder<> &B, Value *Index, ScalarEvolution *SE,
                              const DataLayout &DL,
                              const InductionDescriptor &ID) const;

  /// Emit basic blocks (prefixed with \p Prefix) for the iteration check,
  /// vector loop preheader, middle block and scalar preheader. Also
  /// allocate a loop object for the new vector loop and return it.
  Loop *createVectorLoopSkeleton(StringRef Prefix);

  /// Create new phi nodes for the induction variables to resume iteration count
  /// in the scalar epilogue, from where the vectorized loop left off (given by
  /// \p VectorTripCount).
  /// In cases where the loop skeleton is more complicated (eg. epilogue
  /// vectorization) and the resume values can come from an additional bypass
  /// block, the \p AdditionalBypass pair provides information about the bypass
  /// block and the end value on the edge from bypass to this loop.
  void createInductionResumeValues(
      Loop *L, Value *VectorTripCount,
      std::pair<BasicBlock *, Value *> AdditionalBypass = {nullptr, nullptr});

  /// Complete the loop skeleton by adding debug MDs, creating appropriate
  /// conditional branches in the middle block, preparing the builder and
  /// running the verifier. Take in the vector loop \p L as argument, and return
  /// the preheader of the completed vector loop.
  BasicBlock *completeLoopSkeleton(Loop *L, MDNode *OrigLoopID);

  /// Add additional metadata to \p To that was not present on \p Orig.
  ///
  /// Currently this is used to add the noalias annotations based on the
  /// inserted memchecks.  Use this for instructions that are *cloned* into the
  /// vector loop.
  void addNewMetadata(Instruction *To, const Instruction *Orig);

  /// Add metadata from one instruction to another.
  ///
  /// This includes both the original MDs from \p From and additional ones (\see
  /// addNewMetadata).  Use this for *newly created* instructions in the vector
  /// loop.
  void addMetadata(Instruction *To, Instruction *From);

  /// Similar to the previous function but it adds the metadata to a
  /// vector of instructions.
  void addMetadata(ArrayRef<Value *> To, Instruction *From);

  /// Allow subclasses to override and print debug traces before/after vplan
  /// execution, when trace information is requested.
  virtual void printDebugTracesAtStart(){};
  virtual void printDebugTracesAtEnd(){};

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

  /// Alias Analysis.
  AAResults *AA;

  /// Target Library Info.
  const TargetLibraryInfo *TLI;

  /// Target Transform Info.
  const TargetTransformInfo *TTI;

  /// Assumption Cache.
  AssumptionCache *AC;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;

  /// LoopVersioning.  It's only set up (non-null) if memchecks were
  /// used.
  ///
  /// This is currently only used to add no-alias metadata based on the
  /// memchecks.  The actually versioning is performed manually.
  std::unique_ptr<LoopVersioning> LVer;

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
  BasicBlock *LoopVectorPreHeader;

  /// The scalar-loop preheader.
  BasicBlock *LoopScalarPreHeader;

  /// Middle Block between the vector and the scalar.
  BasicBlock *LoopMiddleBlock;

  /// The (unique) ExitBlock of the scalar loop.  Note that
  /// there can be multiple exiting edges reaching this block.
  BasicBlock *LoopExitBlock;

  /// The vector loop body.
  BasicBlock *LoopVectorBody;

  /// The scalar loop body.
  BasicBlock *LoopScalarBody;

  /// A list of all bypass blocks. The first block is the entry of the loop.
  SmallVector<BasicBlock *, 4> LoopBypassBlocks;

  /// The new Induction variable which was added to the new block.
  PHINode *Induction = nullptr;

  /// The induction variable of the old basic block.
  PHINode *OldInduction = nullptr;

  /// Store instructions that were predicated.
  SmallVector<Instruction *, 4> PredicatedInstructions;

  /// Trip count of the original loop.
  Value *TripCount = nullptr;

  /// Trip count of the widened loop (TripCount - TripCount % (VF*UF))
  Value *VectorTripCount = nullptr;

  /// The legality analysis.
  LoopVectorizationLegality *Legal;

  /// The profitablity analysis.
  LoopVectorizationCostModel *Cost;

  // Record whether runtime checks are added.
  bool AddedSafetyChecks = false;

  // Holds the end values for each induction variable. We save the end values
  // so we can later fix-up the external users of the induction variables.
  DenseMap<PHINode *, Value *> IVEndValues;

  // Vector of original scalar PHIs whose corresponding widened PHIs need to be
  // fixed up at the end of vector code generation.
  SmallVector<PHINode *, 8> OrigPHIsToFix;

  /// BFI and PSI are used to check for profile guided size optimizations.
  BlockFrequencyInfo *BFI;
  ProfileSummaryInfo *PSI;

  // Whether this loop should be optimized for size based on profile guided size
  // optimizatios.
  bool OptForSizeBasedOnProfile;

  /// Structure to hold information about generated runtime checks, responsible
  /// for cleaning the checks, if vectorization turns out unprofitable.
  GeneratedRTChecks &RTChecks;
};

class InnerLoopUnroller : public InnerLoopVectorizer {
public:
  InnerLoopUnroller(Loop *OrigLoop, PredicatedScalarEvolution &PSE,
                    LoopInfo *LI, DominatorTree *DT,
                    const TargetLibraryInfo *TLI,
                    const TargetTransformInfo *TTI, AssumptionCache *AC,
                    OptimizationRemarkEmitter *ORE, unsigned UnrollFactor,
                    LoopVectorizationLegality *LVL,
                    LoopVectorizationCostModel *CM, BlockFrequencyInfo *BFI,
                    ProfileSummaryInfo *PSI, GeneratedRTChecks &Check)
      : InnerLoopVectorizer(OrigLoop, PSE, LI, DT, TLI, TTI, AC, ORE,
                            ElementCount::getFixed(1), UnrollFactor, LVL, CM,
                            BFI, PSI, Check) {}

private:
  Value *getBroadcastInstrs(Value *V) override;
  Value *getStepVector(Value *Val, int StartIdx, Value *Step,
                       Instruction::BinaryOps Opcode =
                       Instruction::BinaryOpsEnd) override;
  Value *reverseVector(Value *Vec) override;
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
  BasicBlock *SCEVSafetyCheck = nullptr;
  BasicBlock *MemSafetyCheck = nullptr;
  Value *TripCount = nullptr;
  Value *VectorTripCount = nullptr;

  EpilogueLoopVectorizationInfo(unsigned MVF, unsigned MUF, unsigned EVF,
                                unsigned EUF)
      : MainLoopVF(ElementCount::getFixed(MVF)), MainLoopUF(MUF),
        EpilogueVF(ElementCount::getFixed(EVF)), EpilogueUF(EUF) {
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
      DominatorTree *DT, const TargetLibraryInfo *TLI,
      const TargetTransformInfo *TTI, AssumptionCache *AC,
      OptimizationRemarkEmitter *ORE, EpilogueLoopVectorizationInfo &EPI,
      LoopVectorizationLegality *LVL, llvm::LoopVectorizationCostModel *CM,
      BlockFrequencyInfo *BFI, ProfileSummaryInfo *PSI,
      GeneratedRTChecks &Checks)
      : InnerLoopVectorizer(OrigLoop, PSE, LI, DT, TLI, TTI, AC, ORE,
                            EPI.MainLoopVF, EPI.MainLoopUF, LVL, CM, BFI, PSI,
                            Checks),
        EPI(EPI) {}

  // Override this function to handle the more complex control flow around the
  // three loops.
  BasicBlock *createVectorizedLoopSkeleton() final override {
    return createEpilogueVectorizedLoopSkeleton();
  }

  /// The interface for creating a vectorized skeleton using one of two
  /// different strategies, each corresponding to one execution of the vplan
  /// as described above.
  virtual BasicBlock *createEpilogueVectorizedLoopSkeleton() = 0;

  /// Holds and updates state information required to vectorize the main loop
  /// and its epilogue in two separate passes. This setup helps us avoid
  /// regenerating and recomputing runtime safety checks. It also helps us to
  /// shorten the iteration-count-check path length for the cases where the
  /// iteration count of the loop is so small that the main vector loop is
  /// completely skipped.
  EpilogueLoopVectorizationInfo &EPI;
};

/// A specialized derived class of inner loop vectorizer that performs
/// vectorization of *main* loops in the process of vectorizing loops and their
/// epilogues.
class EpilogueVectorizerMainLoop : public InnerLoopAndEpilogueVectorizer {
public:
  EpilogueVectorizerMainLoop(
      Loop *OrigLoop, PredicatedScalarEvolution &PSE, LoopInfo *LI,
      DominatorTree *DT, const TargetLibraryInfo *TLI,
      const TargetTransformInfo *TTI, AssumptionCache *AC,
      OptimizationRemarkEmitter *ORE, EpilogueLoopVectorizationInfo &EPI,
      LoopVectorizationLegality *LVL, llvm::LoopVectorizationCostModel *CM,
      BlockFrequencyInfo *BFI, ProfileSummaryInfo *PSI,
      GeneratedRTChecks &Check)
      : InnerLoopAndEpilogueVectorizer(OrigLoop, PSE, LI, DT, TLI, TTI, AC, ORE,
                                       EPI, LVL, CM, BFI, PSI, Check) {}
  /// Implements the interface for creating a vectorized skeleton using the
  /// *main loop* strategy (ie the first pass of vplan execution).
  BasicBlock *createEpilogueVectorizedLoopSkeleton() final override;

protected:
  /// Emits an iteration count bypass check once for the main loop (when \p
  /// ForEpilogue is false) and once for the epilogue loop (when \p
  /// ForEpilogue is true).
  BasicBlock *emitMinimumIterationCountCheck(Loop *L, BasicBlock *Bypass,
                                             bool ForEpilogue);
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
      DominatorTree *DT, const TargetLibraryInfo *TLI,
      const TargetTransformInfo *TTI, AssumptionCache *AC,
      OptimizationRemarkEmitter *ORE, EpilogueLoopVectorizationInfo &EPI,
      LoopVectorizationLegality *LVL, llvm::LoopVectorizationCostModel *CM,
      BlockFrequencyInfo *BFI, ProfileSummaryInfo *PSI,
      GeneratedRTChecks &Checks)
      : InnerLoopAndEpilogueVectorizer(OrigLoop, PSE, LI, DT, TLI, TTI, AC, ORE,
                                       EPI, LVL, CM, BFI, PSI, Checks) {}
  /// Implements the interface for creating a vectorized skeleton using the
  /// *epilogue loop* strategy (ie the second pass of vplan execution).
  BasicBlock *createEpilogueVectorizedLoopSkeleton() final override;

protected:
  /// Emits an iteration count bypass check after the main vector loop has
  /// finished to see if there are any iterations left to execute by either
  /// the vector epilogue or the scalar epilogue.
  BasicBlock *emitMinimumVectorEpilogueIterCountCheck(Loop *L,
                                                      BasicBlock *Bypass,
                                                      BasicBlock *Insert);
  void printDebugTracesAtStart() override;
  void printDebugTracesAtEnd() override;
};
} // end namespace llvm

/// Look for a meaningful debug location on the instruction or it's
/// operands.
static Instruction *getDebugLocFromInstOrOperands(Instruction *I) {
  if (!I)
    return I;

  DebugLoc Empty;
  if (I->getDebugLoc() != Empty)
    return I;

  for (Use &Op : I->operands()) {
    if (Instruction *OpInst = dyn_cast<Instruction>(Op))
      if (OpInst->getDebugLoc() != Empty)
        return OpInst;
  }

  return I;
}

void InnerLoopVectorizer::setDebugLocFromInst(
    const Value *V, Optional<IRBuilder<> *> CustomBuilder) {
  IRBuilder<> *B = (CustomBuilder == None) ? &Builder : *CustomBuilder;
  if (const Instruction *Inst = dyn_cast_or_null<Instruction>(V)) {
    const DILocation *DIL = Inst->getDebugLoc();

    // When a FSDiscriminator is enabled, we don't need to add the multiply
    // factors to the discriminators.
    if (DIL && Inst->getFunction()->isDebugInfoForProfiling() &&
        !isa<DbgInfoIntrinsic>(Inst) && !EnableFSDiscriminator) {
      // FIXME: For scalable vectors, assume vscale=1.
      auto NewDIL =
          DIL->cloneByMultiplyingDuplicationFactor(UF * VF.getKnownMinValue());
      if (NewDIL)
        B->SetCurrentDebugLocation(NewDIL.getValue());
      else
        LLVM_DEBUG(dbgs()
                   << "Failed to create new discriminator: "
                   << DIL->getFilename() << " Line: " << DIL->getLine());
    } else
      B->SetCurrentDebugLocation(DIL);
  } else
    B->SetCurrentDebugLocation(DebugLoc());
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
/// the location of the remark.  \return the remark object that can be
/// streamed to.
static OptimizationRemarkAnalysis createLVAnalysis(const char *PassName,
    StringRef RemarkName, Loop *TheLoop, Instruction *I) {
  Value *CodeRegion = TheLoop->getHeader();
  DebugLoc DL = TheLoop->getStartLoc();

  if (I) {
    CodeRegion = I->getParent();
    // If there is no debug location attached to the instruction, revert back to
    // using the loop's.
    if (I->getDebugLoc())
      DL = I->getDebugLoc();
  }

  return OptimizationRemarkAnalysis(PassName, RemarkName, DL, CodeRegion);
}

/// Return a value for Step multiplied by VF.
static Value *createStepForVF(IRBuilder<> &B, Constant *Step, ElementCount VF) {
  assert(isa<ConstantInt>(Step) && "Expected an integer step");
  Constant *StepVal = ConstantInt::get(
      Step->getType(),
      cast<ConstantInt>(Step)->getSExtValue() * VF.getKnownMinValue());
  return VF.isScalable() ? B.CreateVScale(StepVal) : StepVal;
}

namespace llvm {

/// Return the runtime value for VF.
Value *getRuntimeVF(IRBuilder<> &B, Type *Ty, ElementCount VF) {
  Constant *EC = ConstantInt::get(Ty, VF.getKnownMinValue());
  return VF.isScalable() ? B.CreateVScale(EC) : EC;
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

void reportVectorizationInfo(const StringRef Msg, const StringRef ORETag,
                             OptimizationRemarkEmitter *ORE, Loop *TheLoop,
                             Instruction *I) {
  LLVM_DEBUG(debugVectorizationMessage("", Msg, I));
  LoopVectorizeHints Hints(TheLoop, true /* doesn't matter */, *ORE);
  ORE->emit(
      createLVAnalysis(Hints.vectorizeAnalysisPassName(), ORETag, TheLoop, I)
      << Msg);
}

} // end namespace llvm

#ifndef NDEBUG
/// \return string containing a file name and a line # for the given loop.
static std::string getDebugLocString(const Loop *L) {
  std::string Result;
  if (L) {
    raw_string_ostream OS(Result);
    if (const DebugLoc LoopDbgLoc = L->getStartLoc())
      LoopDbgLoc.print(OS);
    else
      // Just print the module name.
      OS << L->getHeader()->getParent()->getParent()->getModuleIdentifier();
    OS.flush();
  }
  return Result;
}
#endif

void InnerLoopVectorizer::addNewMetadata(Instruction *To,
                                         const Instruction *Orig) {
  // If the loop was versioned with memchecks, add the corresponding no-alias
  // metadata.
  if (LVer && (isa<LoadInst>(Orig) || isa<StoreInst>(Orig)))
    LVer->annotateInstWithNoAlias(To, Orig);
}

void InnerLoopVectorizer::addMetadata(Instruction *To,
                                      Instruction *From) {
  propagateMetadata(To, From);
  addNewMetadata(To, From);
}

void InnerLoopVectorizer::addMetadata(ArrayRef<Value *> To,
                                      Instruction *From) {
  for (Value *V : To) {
    if (Instruction *I = dyn_cast<Instruction>(V))
      addMetadata(I, From);
  }
}

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

/// ElementCountComparator creates a total ordering for ElementCount
/// for the purposes of using it in a set structure.
struct ElementCountComparator {
  bool operator()(const ElementCount &LHS, const ElementCount &RHS) const {
    return std::make_tuple(LHS.isScalable(), LHS.getKnownMinValue()) <
           std::make_tuple(RHS.isScalable(), RHS.getKnownMinValue());
  }
};
using ElementCountSet = SmallSet<ElementCount, 16, ElementCountComparator>;

/// LoopVectorizationCostModel - estimates the expected speedups due to
/// vectorization.
/// In many cases vectorization is not profitable. This can happen because of
/// a number of reasons. In this class we mainly attempt to predict the
/// expected speedup/slowdowns due to the supported instruction set. We use the
/// TargetTransformInfo to query the different backends for the cost of
/// different operations.
class LoopVectorizationCostModel {
public:
  LoopVectorizationCostModel(ScalarEpilogueLowering SEL, Loop *L,
                             PredicatedScalarEvolution &PSE, LoopInfo *LI,
                             LoopVectorizationLegality *Legal,
                             const TargetTransformInfo &TTI,
                             const TargetLibraryInfo *TLI, DemandedBits *DB,
                             AssumptionCache *AC,
                             OptimizationRemarkEmitter *ORE, const Function *F,
                             const LoopVectorizeHints *Hints,
                             InterleavedAccessInfo &IAI)
      : ScalarEpilogueStatus(SEL), TheLoop(L), PSE(PSE), LI(LI), Legal(Legal),
        TTI(TTI), TLI(TLI), DB(DB), AC(AC), ORE(ORE), TheFunction(F),
        Hints(Hints), InterleaveInfo(IAI) {}

  /// \return An upper bound for the vectorization factors (both fixed and
  /// scalable). If the factors are 0, vectorization and interleaving should be
  /// avoided up front.
  FixedScalableVFPair computeMaxVF(ElementCount UserVF, unsigned UserIC);

  /// \return True if runtime checks are required for vectorization, and false
  /// otherwise.
  bool runtimeChecksRequired();

  /// \return The most profitable vectorization factor and the cost of that VF.
  /// This method checks every VF in \p CandidateVFs. If UserVF is not ZERO
  /// then this vectorization factor will be selected if vectorization is
  /// possible.
  VectorizationFactor
  selectVectorizationFactor(const ElementCountSet &CandidateVFs);

  VectorizationFactor
  selectEpilogueVectorizationFactor(const ElementCount MaxVF,
                                    const LoopVectorizationPlanner &LVP);

  /// Setup cost-based decisions for user vectorization factor.
  void selectUserVectorizationFactor(ElementCount UserVF) {
    collectUniformsAndScalars(UserVF);
    collectInstsToScalarize(UserVF);
  }

  /// \return The size (in bits) of the smallest and widest types in the code
  /// that needs to be vectorized. We ignore values that remain scalar such as
  /// 64 bit loop indices.
  std::pair<unsigned, unsigned> getSmallestAndWidestTypes();

  /// \return The desired interleave count.
  /// If interleave count has been specified by metadata it will be returned.
  /// Otherwise, the interleave count is computed and returned. VF and LoopCost
  /// are the selected vectorization factor and the cost of the selected VF.
  unsigned selectInterleaveCount(ElementCount VF, unsigned LoopCost);

  /// Memory access instruction may be vectorized in more than one way.
  /// Form of instruction after vectorization depends on cost.
  /// This function takes cost-based decisions for Load/Store instructions
  /// and collects them in a map. This decisions map is used for building
  /// the lists of loop-uniform and loop-scalar instructions.
  /// The calculated cost is saved with widening decision in order to
  /// avoid redundant calculations.
  void setCostBasedWideningDecision(ElementCount VF);

  /// A struct that represents some properties of the register usage
  /// of a loop.
  struct RegisterUsage {
    /// Holds the number of loop invariant values that are used in the loop.
    /// The key is ClassID of target-provided register class.
    SmallMapVector<unsigned, unsigned, 4> LoopInvariantRegs;
    /// Holds the maximum number of concurrent live intervals in the loop.
    /// The key is ClassID of target-provided register class.
    SmallMapVector<unsigned, unsigned, 4> MaxLocalUsers;
  };

  /// \return Returns information about the register usages of the loop for the
  /// given vectorization factors.
  SmallVector<RegisterUsage, 8>
  calculateRegisterUsage(ArrayRef<ElementCount> VFs);

  /// Collect values we want to ignore in the cost model.
  void collectValuesToIgnore();

  /// Split reductions into those that happen in the loop, and those that happen
  /// outside. In loop reductions are collected into InLoopReductionChains.
  void collectInLoopReductions();

  /// Returns true if we should use strict in-order reductions for the given
  /// RdxDesc. This is true if the -enable-strict-reductions flag is passed,
  /// the IsOrdered flag of RdxDesc is set and we do not allow reordering
  /// of FP operations.
  bool useOrderedReductions(const RecurrenceDescriptor &RdxDesc) {
    return EnableStrictReductions && !Hints->allowReordering() &&
           RdxDesc.isOrdered();
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

    // Cost model is not run in the VPlan-native path - return conservative
    // result until this changes.
    if (EnableVPlanNativePath)
      return false;

    auto Scalars = InstsToScalarize.find(VF);
    assert(Scalars != InstsToScalarize.end() &&
           "VF not yet analyzed for scalarization profitability");
    return Scalars->second.find(I) != Scalars->second.end();
  }

  /// Returns true if \p I is known to be uniform after vectorization.
  bool isUniformAfterVectorization(Instruction *I, ElementCount VF) const {
    if (VF.isScalar())
      return true;

    // Cost model is not run in the VPlan-native path - return conservative
    // result until this changes.
    if (EnableVPlanNativePath)
      return false;

    auto UniformsPerVF = Uniforms.find(VF);
    assert(UniformsPerVF != Uniforms.end() &&
           "VF not yet analyzed for uniformity");
    return UniformsPerVF->second.count(I);
  }

  /// Returns true if \p I is known to be scalar after vectorization.
  bool isScalarAfterVectorization(Instruction *I, ElementCount VF) const {
    if (VF.isScalar())
      return true;

    // Cost model is not run in the VPlan-native path - return conservative
    // result until this changes.
    if (EnableVPlanNativePath)
      return false;

    auto ScalarsPerVF = Scalars.find(VF);
    assert(ScalarsPerVF != Scalars.end() &&
           "Scalar values are not calculated for VF");
    return ScalarsPerVF->second.count(I);
  }

  /// \returns True if instruction \p I can be truncated to a smaller bitwidth
  /// for vectorization factor \p VF.
  bool canTruncateToMinimalBitwidth(Instruction *I, ElementCount VF) const {
    return VF.isVector() && MinBWs.find(I) != MinBWs.end() &&
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
    CM_Scalarize
  };

  /// Save vectorization decision \p W and \p Cost taken by the cost model for
  /// instruction \p I and vector width \p VF.
  void setWideningDecision(Instruction *I, ElementCount VF, InstWidening W,
                           InstructionCost Cost) {
    assert(VF.isVector() && "Expected VF >=2");
    WideningDecisions[std::make_pair(I, VF)] = std::make_pair(W, Cost);
  }

  /// Save vectorization decision \p W and \p Cost taken by the cost model for
  /// interleaving group \p Grp and vector width \p VF.
  void setWideningDecision(const InterleaveGroup<Instruction> *Grp,
                           ElementCount VF, InstWidening W,
                           InstructionCost Cost) {
    assert(VF.isVector() && "Expected VF >=2");
    /// Broadcast this decicion to all instructions inside the group.
    /// But the cost will be assigned to one instruction only.
    for (unsigned i = 0; i < Grp->getFactor(); ++i) {
      if (auto *I = Grp->getMember(i)) {
        if (Grp->getInsertPos() == I)
          WideningDecisions[std::make_pair(I, VF)] = std::make_pair(W, Cost);
        else
          WideningDecisions[std::make_pair(I, VF)] = std::make_pair(W, 0);
      }
    }
  }

  /// Return the cost model decision for the given instruction \p I and vector
  /// width \p VF. Return CM_Unknown if this instruction did not pass
  /// through the cost modeling.
  InstWidening getWideningDecision(Instruction *I, ElementCount VF) const {
    assert(VF.isVector() && "Expected VF to be a vector VF");
    // Cost model is not run in the VPlan-native path - return conservative
    // result until this changes.
    if (EnableVPlanNativePath)
      return CM_GatherScatter;

    std::pair<Instruction *, ElementCount> InstOnVF = std::make_pair(I, VF);
    auto Itr = WideningDecisions.find(InstOnVF);
    if (Itr == WideningDecisions.end())
      return CM_Unknown;
    return Itr->second.first;
  }

  /// Return the vectorization cost for the given instruction \p I and vector
  /// width \p VF.
  InstructionCost getWideningCost(Instruction *I, ElementCount VF) {
    assert(VF.isVector() && "Expected VF >=2");
    std::pair<Instruction *, ElementCount> InstOnVF = std::make_pair(I, VF);
    assert(WideningDecisions.find(InstOnVF) != WideningDecisions.end() &&
           "The cost is not calculated");
    return WideningDecisions[InstOnVF].second;
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
    Type *SrcTy = ToVectorTy(cast<CastInst>(I)->getSrcTy(), VF);
    Type *DestTy = ToVectorTy(cast<CastInst>(I)->getDestTy(), VF);

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

  /// Collect Uniform and Scalar values for the given \p VF.
  /// The sets depend on CM decision for Load/Store instructions
  /// that may be vectorized as interleave, gather-scatter or scalarized.
  void collectUniformsAndScalars(ElementCount VF) {
    // Do the analysis once.
    if (VF.isScalar() || Uniforms.find(VF) != Uniforms.end())
      return;
    setCostBasedWideningDecision(VF);
    collectLoopUniforms(VF);
    collectLoopScalars(VF);
  }

  /// Returns true if the target machine supports masked store operation
  /// for the given \p DataType and kind of access to \p Ptr.
  bool isLegalMaskedStore(Type *DataType, Value *Ptr, Align Alignment) const {
    return Legal->isConsecutivePtr(Ptr) &&
           TTI.isLegalMaskedStore(DataType, Alignment);
  }

  /// Returns true if the target machine supports masked load operation
  /// for the given \p DataType and kind of access to \p Ptr.
  bool isLegalMaskedLoad(Type *DataType, Value *Ptr, Align Alignment) const {
    return Legal->isConsecutivePtr(Ptr) &&
           TTI.isLegalMaskedLoad(DataType, Alignment);
  }

  /// Returns true if the target machine can represent \p V as a masked gather
  /// or scatter operation.
  bool isLegalGatherOrScatter(Value *V) {
    bool LI = isa<LoadInst>(V);
    bool SI = isa<StoreInst>(V);
    if (!LI && !SI)
      return false;
    auto *Ty = getLoadStoreType(V);
    Align Align = getLoadStoreAlignment(V);
    return (LI && TTI.isLegalMaskedGather(Ty, Align)) ||
           (SI && TTI.isLegalMaskedScatter(Ty, Align));
  }

  /// Returns true if the target machine supports all of the reduction
  /// variables found for the given VF.
  bool canVectorizeReductions(ElementCount VF) {
    return (all_of(Legal->getReductionVars(), [&](auto &Reduction) -> bool {
      const RecurrenceDescriptor &RdxDesc = Reduction.second;
      return TTI.isLegalToVectorizeReduction(RdxDesc, VF);
    }));
  }

  /// Returns true if \p I is an instruction that will be scalarized with
  /// predication. Such instructions include conditional stores and
  /// instructions that may divide by zero.
  /// If a non-zero VF has been calculated, we check if I will be scalarized
  /// predication for that VF.
  bool isScalarWithPredication(Instruction *I) const;

  // Returns true if \p I is an instruction that will be predicated either
  // through scalar predication or masked load/store or masked gather/scatter.
  // Superset of instructions that return true for isScalarWithPredication.
  bool isPredicatedInst(Instruction *I) {
    if (!blockNeedsPredication(I->getParent()))
      return false;
    // Loads and stores that need some form of masked operation are predicated
    // instructions.
    if (isa<LoadInst>(I) || isa<StoreInst>(I))
      return Legal->isMaskRequired(I);
    return isScalarWithPredication(I);
  }

  /// Returns true if \p I is a memory instruction with consecutive memory
  /// access that can be widened.
  bool
  memoryInstructionCanBeWidened(Instruction *I,
                                ElementCount VF = ElementCount::getFixed(1));

  /// Returns true if \p I is a memory instruction in an interleaved-group
  /// of memory accesses that can be vectorized with wide vector loads/stores
  /// and shuffles.
  bool
  interleavedAccessCanBeWidened(Instruction *I,
                                ElementCount VF = ElementCount::getFixed(1));

  /// Check if \p Instr belongs to any interleaved access group.
  bool isAccessInterleaved(Instruction *Instr) {
    return InterleaveInfo.isInterleaved(Instr);
  }

  /// Get the interleaved access group that \p Instr belongs to.
  const InterleaveGroup<Instruction> *
  getInterleavedAccessGroup(Instruction *Instr) {
    return InterleaveInfo.getInterleaveGroup(Instr);
  }

  /// Returns true if we're required to use a scalar epilogue for at least
  /// the final iteration of the original loop.
  bool requiresScalarEpilogue(ElementCount VF) const {
    if (!isScalarEpilogueAllowed())
      return false;
    // If we might exit from anywhere but the latch, must run the exiting
    // iteration in scalar form.
    if (TheLoop->getExitingBlock() != TheLoop->getLoopLatch())
      return true;
    return VF.isVector() && InterleaveInfo.requiresScalarEpilogue();
  }

  /// Returns true if a scalar epilogue is not allowed due to optsize or a
  /// loop hint annotation.
  bool isScalarEpilogueAllowed() const {
    return ScalarEpilogueStatus == CM_ScalarEpilogueAllowed;
  }

  /// Returns true if all loop blocks should be masked to fold tail loop.
  bool foldTailByMasking() const { return FoldTailByMasking; }

  bool blockNeedsPredication(BasicBlock *BB) const {
    return foldTailByMasking() || Legal->blockNeedsPredication(BB);
  }

  /// A SmallMapVector to store the InLoop reduction op chains, mapping phi
  /// nodes to the chain of instructions representing the reductions. Uses a
  /// MapVector to ensure deterministic iteration order.
  using ReductionChainMap =
      SmallMapVector<PHINode *, SmallVector<Instruction *, 4>, 4>;

  /// Return the chain of instructions representing an inloop reduction.
  const ReductionChainMap &getInLoopReductionChains() const {
    return InLoopReductionChains;
  }

  /// Returns true if the Phi is part of an inloop reduction.
  bool isInLoopReduction(PHINode *Phi) const {
    return InLoopReductionChains.count(Phi);
  }

  /// Estimate cost of an intrinsic call instruction CI if it were vectorized
  /// with factor VF.  Return the cost of the instruction, including
  /// scalarization overhead if it's needed.
  InstructionCost getVectorIntrinsicCost(CallInst *CI, ElementCount VF) const;

  /// Estimate cost of a call instruction CI if it were vectorized with factor
  /// VF. Return the cost of the instruction, including scalarization overhead
  /// if it's needed. The flag NeedToScalarize shows if the call needs to be
  /// scalarized -
  /// i.e. either vector version isn't available, or is too expensive.
  InstructionCost getVectorCallCost(CallInst *CI, ElementCount VF,
                                    bool &NeedToScalarize) const;

  /// Returns true if the per-lane cost of VectorizationFactor A is lower than
  /// that of B.
  bool isMoreProfitable(const VectorizationFactor &A,
                        const VectorizationFactor &B) const;

  /// Invalidates decisions already taken by the cost model.
  void invalidateCostModelingDecisions() {
    WideningDecisions.clear();
    Uniforms.clear();
    Scalars.clear();
  }

private:
  unsigned NumPredStores = 0;

  /// \return An upper bound for the vectorization factors for both
  /// fixed and scalable vectorization, where the minimum-known number of
  /// elements is a power-of-2 larger than zero. If scalable vectorization is
  /// disabled or unsupported, then the scalable part will be equal to
  /// ElementCount::getScalable(0).
  FixedScalableVFPair computeFeasibleMaxVF(unsigned ConstTripCount,
                                           ElementCount UserVF);

  /// \return the maximized element count based on the targets vector
  /// registers and the loop trip-count, but limited to a maximum safe VF.
  /// This is a helper function of computeFeasibleMaxVF.
  /// FIXME: MaxSafeVF is currently passed by reference to avoid some obscure
  /// issue that occurred on one of the buildbots which cannot be reproduced
  /// without having access to the properietary compiler (see comments on
  /// D98509). The issue is currently under investigation and this workaround
  /// will be removed as soon as possible.
  ElementCount getMaximizedVFForTarget(unsigned ConstTripCount,
                                       unsigned SmallestType,
                                       unsigned WidestType,
                                       const ElementCount &MaxSafeVF);

  /// \return the maximum legal scalable VF, based on the safe max number
  /// of elements.
  ElementCount getMaxLegalScalableVF(unsigned MaxSafeElements);

  /// The vectorization cost is a combination of the cost itself and a boolean
  /// indicating whether any of the contributing operations will actually
  /// operate on vector values after type legalization in the backend. If this
  /// latter value is false, then all operations will be scalarized (i.e. no
  /// vectorization has actually taken place).
  using VectorizationCostTy = std::pair<InstructionCost, bool>;

  /// Returns the expected execution cost. The unit of the cost does
  /// not matter because we use the 'cost' units to compare different
  /// vector widths. The cost that is returned is *not* normalized by
  /// the factor width.
  VectorizationCostTy expectedCost(ElementCount VF);

  /// Returns the execution time cost of an instruction for a given vector
  /// width. Vector width of one means scalar.
  VectorizationCostTy getInstructionCost(Instruction *I, ElementCount VF);

  /// The cost-computation logic from getInstructionCost which provides
  /// the vector type as an output parameter.
  InstructionCost getInstructionCost(Instruction *I, ElementCount VF,
                                     Type *&VectorTy);

  /// Return the cost of instructions in an inloop reduction pattern, if I is
  /// part of that pattern.
  InstructionCost getReductionPatternCost(Instruction *I, ElementCount VF,
                                          Type *VectorTy,
                                          TTI::TargetCostKind CostKind);

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

  /// Returns whether the instruction is a load or store and will be a emitted
  /// as a vector operation.
  bool isConsecutiveLoadOrStore(Instruction *I);

  /// Returns true if an artificially high cost for emulated masked memrefs
  /// should be used.
  bool useEmulatedMaskMemRefHack(Instruction *I);

  /// Map of scalar integer values to the smallest bitwidth they can be legally
  /// represented as. The vector equivalents of these values should be truncated
  /// to this type.
  MapVector<Instruction *, uint64_t> MinBWs;

  /// A type representing the costs for instructions if they were to be
  /// scalarized rather than vectorized. The entries are Instruction-Cost
  /// pairs.
  using ScalarCostsTy = DenseMap<Instruction *, InstructionCost>;

  /// A set containing all BasicBlocks that are known to present after
  /// vectorization as a predicated block.
  SmallPtrSet<BasicBlock *, 4> PredicatedBBsAfterVectorization;

  /// Records whether it is allowed to have the original scalar loop execute at
  /// least once. This may be needed as a fallback loop in case runtime
  /// aliasing/dependence checks fail, or to handle the tail/remainder
  /// iterations when the trip count is unknown or doesn't divide by the VF,
  /// or as a peel-loop to handle gaps in interleave-groups.
  /// Under optsize and when the trip count is very small we don't allow any
  /// iterations to execute in the scalar loop.
  ScalarEpilogueLowering ScalarEpilogueStatus = CM_ScalarEpilogueAllowed;

  /// All blocks of loop are to be masked to fold tail of scalar iterations.
  bool FoldTailByMasking = false;

  /// A map holding scalar costs for different vectorization factors. The
  /// presence of a cost for an instruction in the mapping indicates that the
  /// instruction will be scalarized when vectorizing with the associated
  /// vectorization factor. The entries are VF-ScalarCostTy pairs.
  DenseMap<ElementCount, ScalarCostsTy> InstsToScalarize;

  /// Holds the instructions known to be uniform after vectorization.
  /// The data is collected per VF.
  DenseMap<ElementCount, SmallPtrSet<Instruction *, 4>> Uniforms;

  /// Holds the instructions known to be scalar after vectorization.
  /// The data is collected per VF.
  DenseMap<ElementCount, SmallPtrSet<Instruction *, 4>> Scalars;

  /// Holds the instructions (address computations) that are forced to be
  /// scalarized.
  DenseMap<ElementCount, SmallPtrSet<Instruction *, 4>> ForcedScalars;

  /// PHINodes of the reductions that should be expanded in-loop along with
  /// their associated chains of reduction operations, in program order from top
  /// (PHI) to bottom
  ReductionChainMap InLoopReductionChains;

  /// A Map of inloop reduction operations and their immediate chain operand.
  /// FIXME: This can be removed once reductions can be costed correctly in
  /// vplan. This was added to allow quick lookup to the inloop operations,
  /// without having to loop through InLoopReductionChains.
  DenseMap<Instruction *, Instruction *> InLoopReductionImmediateChains;

  /// Returns the expected difference in cost from scalarizing the expression
  /// feeding a predicated instruction \p PredInst. The instructions to
  /// scalarize and their scalar costs are collected in \p ScalarCosts. A
  /// non-negative return value implies the expression will be scalarized.
  /// Currently, only single-use chains are considered for scalarization.
  int computePredInstDiscount(Instruction *PredInst, ScalarCostsTy &ScalarCosts,
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
  /// during vectorization. Non-uniform scalarized instructions will be
  /// represented by VF values in the vectorized loop, each corresponding to an
  /// iteration of the original scalar loop.
  void collectLoopScalars(ElementCount VF);

  /// Keeps cost model vectorization decision and cost for instructions.
  /// Right now it is used for memory instructions only.
  using DecisionList = DenseMap<std::pair<Instruction *, ElementCount>,
                                std::pair<InstWidening, InstructionCost>>;

  DecisionList WideningDecisions;

  /// Returns true if \p V is expected to be vectorized and it needs to be
  /// extracted.
  bool needsExtract(Value *V, ElementCount VF) const {
    Instruction *I = dyn_cast<Instruction>(V);
    if (VF.isScalar() || !I || !TheLoop->contains(I) ||
        TheLoop->isLoopInvariant(I))
      return false;

    // Assume we can vectorize V (and hence we need extraction) if the
    // scalars are not computed yet. This can happen, because it is called
    // via getScalarizationOverhead from setCostBasedWideningDecision, before
    // the scalars are collected. That should be a safe assumption in most
    // cases, because we check if the operands have vectorizable types
    // beforehand in LoopVectorizationLegality.
    return Scalars.find(VF) == Scalars.end() ||
           !isScalarAfterVectorization(I, VF);
  };

  /// Returns a range containing only operands needing to be extracted.
  SmallVector<Value *, 4> filterExtractingOperands(Instruction::op_range Ops,
                                                   ElementCount VF) const {
    return SmallVector<Value *, 4>(make_filter_range(
        Ops, [this, VF](Value *V) { return this->needsExtract(V, VF); }));
  }

  /// Determines if we have the infrastructure to vectorize loop \p L and its
  /// epilogue, assuming the main loop is vectorized by \p VF.
  bool isCandidateForEpilogueVectorization(const Loop &L,
                                           const ElementCount VF) const;

  /// Returns true if epilogue vectorization is considered profitable, and
  /// false otherwise.
  /// \p VF is the vectorization factor chosen for the original loop.
  bool isEpilogueVectorizationProfitable(const ElementCount VF) const;

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

  /// Profitable vector factors.
  SmallVector<VectorizationFactor, 8> ProfitableVFs;
};
} // end namespace llvm

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
  /// nullptr, either no SCEV checks have been generated or they have been used.
  Value *SCEVCheckCond = nullptr;

  /// Basic block which contains the generated memory runtime checks, if any.
  BasicBlock *MemCheckBlock = nullptr;

  /// The value representing the result of the generated memory runtime checks.
  /// If it is nullptr, either no memory runtime checks have been generated or
  /// they have been used.
  Instruction *MemRuntimeCheckCond = nullptr;

  DominatorTree *DT;
  LoopInfo *LI;

  SCEVExpander SCEVExp;
  SCEVExpander MemCheckExp;

public:
  GeneratedRTChecks(ScalarEvolution &SE, DominatorTree *DT, LoopInfo *LI,
                    const DataLayout &DL)
      : DT(DT), LI(LI), SCEVExp(SE, DL, "scev.check"),
        MemCheckExp(SE, DL, "scev.check") {}

  /// Generate runtime checks in SCEVCheckBlock and MemCheckBlock, so we can
  /// accurately estimate the cost of the runtime checks. The blocks are
  /// un-linked from the IR and is added back during vector code generation. If
  /// there is no vector code generation, the check blocks are removed
  /// completely.
  void Create(Loop *L, const LoopAccessInfo &LAI,
              const SCEVUnionPredicate &UnionPred) {

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
    }

    const auto &RtPtrChecking = *LAI.getRuntimePointerChecking();
    if (RtPtrChecking.Need) {
      auto *Pred = SCEVCheckBlock ? SCEVCheckBlock : Preheader;
      MemCheckBlock = SplitBlock(Pred, Pred->getTerminator(), DT, LI, nullptr,
                                 "vector.memcheck");

      std::tie(std::ignore, MemRuntimeCheckCond) =
          addRuntimeChecks(MemCheckBlock->getTerminator(), L,
                           RtPtrChecking.getChecks(), MemCheckExp);
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
      SCEVCheckBlock->getTerminator()->moveBefore(Preheader->getTerminator());
      new UnreachableInst(Preheader->getContext(), SCEVCheckBlock);
      Preheader->getTerminator()->eraseFromParent();
    }
    if (MemCheckBlock) {
      MemCheckBlock->getTerminator()->moveBefore(Preheader->getTerminator());
      new UnreachableInst(Preheader->getContext(), MemCheckBlock);
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
  }

  /// Remove the created SCEV & memory runtime check blocks & instructions, if
  /// unused.
  ~GeneratedRTChecks() {
    SCEVExpanderCleaner SCEVCleaner(SCEVExp, *DT);
    SCEVExpanderCleaner MemCheckCleaner(MemCheckExp, *DT);
    if (!SCEVCheckCond)
      SCEVCleaner.markResultUsed();

    if (!MemRuntimeCheckCond)
      MemCheckCleaner.markResultUsed();

    if (MemRuntimeCheckCond) {
      auto &SE = *MemCheckExp.getSE();
      // Memory runtime check generation creates compares that use expanded
      // values. Remove them before running the SCEVExpanderCleaners.
      for (auto &I : make_early_inc_range(reverse(*MemCheckBlock))) {
        if (MemCheckExp.isInsertedInstruction(&I))
          continue;
        SE.forgetValue(&I);
        SE.eraseValueFromMap(&I);
        I.eraseFromParent();
      }
    }
    MemCheckCleaner.cleanup();
    SCEVCleaner.cleanup();

    if (SCEVCheckCond)
      SCEVCheckBlock->eraseFromParent();
    if (MemRuntimeCheckCond)
      MemCheckBlock->eraseFromParent();
  }

  /// Adds the generated SCEVCheckBlock before \p LoopVectorPreHeader and
  /// adjusts the branches to branch to the vector preheader or \p Bypass,
  /// depending on the generated condition.
  BasicBlock *emitSCEVChecks(Loop *L, BasicBlock *Bypass,
                             BasicBlock *LoopVectorPreHeader,
                             BasicBlock *LoopExitBlock) {
    if (!SCEVCheckCond)
      return nullptr;
    if (auto *C = dyn_cast<ConstantInt>(SCEVCheckCond))
      if (C->isZero())
        return nullptr;

    auto *Pred = LoopVectorPreHeader->getSinglePredecessor();

    BranchInst::Create(LoopVectorPreHeader, SCEVCheckBlock);
    // Create new preheader for vector loop.
    if (auto *PL = LI->getLoopFor(LoopVectorPreHeader))
      PL->addBasicBlockToLoop(SCEVCheckBlock, *LI);

    SCEVCheckBlock->getTerminator()->eraseFromParent();
    SCEVCheckBlock->moveBefore(LoopVectorPreHeader);
    Pred->getTerminator()->replaceSuccessorWith(LoopVectorPreHeader,
                                                SCEVCheckBlock);

    DT->addNewBlock(SCEVCheckBlock, Pred);
    DT->changeImmediateDominator(LoopVectorPreHeader, SCEVCheckBlock);

    ReplaceInstWithInst(
        SCEVCheckBlock->getTerminator(),
        BranchInst::Create(Bypass, LoopVectorPreHeader, SCEVCheckCond));
    // Mark the check as used, to prevent it from being removed during cleanup.
    SCEVCheckCond = nullptr;
    return SCEVCheckBlock;
  }

  /// Adds the generated MemCheckBlock before \p LoopVectorPreHeader and adjusts
  /// the branches to branch to the vector preheader or \p Bypass, depending on
  /// the generated condition.
  BasicBlock *emitMemRuntimeChecks(Loop *L, BasicBlock *Bypass,
                                   BasicBlock *LoopVectorPreHeader) {
    // Check if we generated code that checks in runtime if arrays overlap.
    if (!MemRuntimeCheckCond)
      return nullptr;

    auto *Pred = LoopVectorPreHeader->getSinglePredecessor();
    Pred->getTerminator()->replaceSuccessorWith(LoopVectorPreHeader,
                                                MemCheckBlock);

    DT->addNewBlock(MemCheckBlock, Pred);
    DT->changeImmediateDominator(LoopVectorPreHeader, MemCheckBlock);
    MemCheckBlock->moveBefore(LoopVectorPreHeader);

    if (auto *PL = LI->getLoopFor(LoopVectorPreHeader))
      PL->addBasicBlockToLoop(MemCheckBlock, *LI);

    ReplaceInstWithInst(
        MemCheckBlock->getTerminator(),
        BranchInst::Create(Bypass, LoopVectorPreHeader, MemRuntimeCheckCond));
    MemCheckBlock->getTerminator()->setDebugLoc(
        Pred->getTerminator()->getDebugLoc());

    // Mark the check as used, to prevent it from being removed during cleanup.
    MemRuntimeCheckCond = nullptr;
    return MemCheckBlock;
  }
};

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

namespace {

/// The LoopVectorize Pass.
struct LoopVectorize : public FunctionPass {
  /// Pass identification, replacement for typeid
  static char ID;

  LoopVectorizePass Impl;

  explicit LoopVectorize(bool InterleaveOnlyWhenForced = false,
                         bool VectorizeOnlyWhenForced = false)
      : FunctionPass(ID),
        Impl({InterleaveOnlyWhenForced, VectorizeOnlyWhenForced}) {
    initializeLoopVectorizePass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    auto *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto *BFI = &getAnalysis<BlockFrequencyInfoWrapperPass>().getBFI();
    auto *TLIP = getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
    auto *TLI = TLIP ? &TLIP->getTLI(F) : nullptr;
    auto *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto *AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    auto *LAA = &getAnalysis<LoopAccessLegacyAnalysis>();
    auto *DB = &getAnalysis<DemandedBitsWrapperPass>().getDemandedBits();
    auto *ORE = &getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();
    auto *PSI = &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();

    std::function<const LoopAccessInfo &(Loop &)> GetLAA =
        [&](Loop &L) -> const LoopAccessInfo & { return LAA->getInfo(&L); };

    return Impl.runImpl(F, *SE, *LI, *TTI, *DT, *BFI, TLI, *DB, *AA, *AC,
                        GetLAA, *ORE, PSI).MadeAnyChange;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<LoopAccessLegacyAnalysis>();
    AU.addRequired<DemandedBitsWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
    AU.addRequired<InjectTLIMappingsLegacy>();

    // We currently do not preserve loopinfo/dominator analyses with outer loop
    // vectorization. Until this is addressed, mark these analyses as preserved
    // only for non-VPlan-native path.
    // TODO: Preserve Loop and Dominator analyses for VPlan-native path.
    if (!EnableVPlanNativePath) {
      AU.addPreserved<LoopInfoWrapperPass>();
      AU.addPreserved<DominatorTreeWrapperPass>();
    }

    AU.addPreserved<BasicAAWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Implementation of LoopVectorizationLegality, InnerLoopVectorizer and
// LoopVectorizationCostModel and LoopVectorizationPlanner.
//===----------------------------------------------------------------------===//

Value *InnerLoopVectorizer::getBroadcastInstrs(Value *V) {
  // We need to place the broadcast of invariant variables outside the loop,
  // but only if it's proven safe to do so. Else, broadcast will be inside
  // vector loop body.
  Instruction *Instr = dyn_cast<Instruction>(V);
  bool SafeToHoist = OrigLoop->isLoopInvariant(V) &&
                     (!Instr ||
                      DT->dominates(Instr->getParent(), LoopVectorPreHeader));
  // Place the code for broadcasting invariant variables in the new preheader.
  IRBuilder<>::InsertPointGuard Guard(Builder);
  if (SafeToHoist)
    Builder.SetInsertPoint(LoopVectorPreHeader->getTerminator());

  // Broadcast the scalar into all locations in the vector.
  Value *Shuf = Builder.CreateVectorSplat(VF, V, "broadcast");

  return Shuf;
}

void InnerLoopVectorizer::createVectorIntOrFpInductionPHI(
    const InductionDescriptor &II, Value *Step, Value *Start,
    Instruction *EntryVal, VPValue *Def, VPValue *CastDef,
    VPTransformState &State) {
  assert((isa<PHINode>(EntryVal) || isa<TruncInst>(EntryVal)) &&
         "Expected either an induction phi-node or a truncate of it!");

  // Construct the initial value of the vector IV in the vector loop preheader
  auto CurrIP = Builder.saveIP();
  Builder.SetInsertPoint(LoopVectorPreHeader->getTerminator());
  if (isa<TruncInst>(EntryVal)) {
    assert(Start->getType()->isIntegerTy() &&
           "Truncation requires an integer type");
    auto *TruncType = cast<IntegerType>(EntryVal->getType());
    Step = Builder.CreateTrunc(Step, TruncType);
    Start = Builder.CreateCast(Instruction::Trunc, Start, TruncType);
  }
  Value *SplatStart = Builder.CreateVectorSplat(VF, Start);
  Value *SteppedStart =
      getStepVector(SplatStart, 0, Step, II.getInductionOpcode());

  // We create vector phi nodes for both integer and floating-point induction
  // variables. Here, we determine the kind of arithmetic we will perform.
  Instruction::BinaryOps AddOp;
  Instruction::BinaryOps MulOp;
  if (Step->getType()->isIntegerTy()) {
    AddOp = Instruction::Add;
    MulOp = Instruction::Mul;
  } else {
    AddOp = II.getInductionOpcode();
    MulOp = Instruction::FMul;
  }

  // Multiply the vectorization factor by the step using integer or
  // floating-point arithmetic as appropriate.
  Type *StepType = Step->getType();
  if (Step->getType()->isFloatingPointTy())
    StepType = IntegerType::get(StepType->getContext(),
                                StepType->getScalarSizeInBits());
  Value *RuntimeVF = getRuntimeVF(Builder, StepType, VF);
  if (Step->getType()->isFloatingPointTy())
    RuntimeVF = Builder.CreateSIToFP(RuntimeVF, Step->getType());
  Value *Mul = Builder.CreateBinOp(MulOp, Step, RuntimeVF);

  // Create a vector splat to use in the induction update.
  //
  // FIXME: If the step is non-constant, we create the vector splat with
  //        IRBuilder. IRBuilder can constant-fold the multiply, but it doesn't
  //        handle a constant vector splat.
  Value *SplatVF = isa<Constant>(Mul)
                       ? ConstantVector::getSplat(VF, cast<Constant>(Mul))
                       : Builder.CreateVectorSplat(VF, Mul);
  Builder.restoreIP(CurrIP);

  // We may need to add the step a number of times, depending on the unroll
  // factor. The last of those goes into the PHI.
  PHINode *VecInd = PHINode::Create(SteppedStart->getType(), 2, "vec.ind",
                                    &*LoopVectorBody->getFirstInsertionPt());
  VecInd->setDebugLoc(EntryVal->getDebugLoc());
  Instruction *LastInduction = VecInd;
  for (unsigned Part = 0; Part < UF; ++Part) {
    State.set(Def, LastInduction, Part);

    if (isa<TruncInst>(EntryVal))
      addMetadata(LastInduction, EntryVal);
    recordVectorLoopValueForInductionCast(II, EntryVal, LastInduction, CastDef,
                                          State, Part);

    LastInduction = cast<Instruction>(
        Builder.CreateBinOp(AddOp, LastInduction, SplatVF, "step.add"));
    LastInduction->setDebugLoc(EntryVal->getDebugLoc());
  }

  // Move the last step to the end of the latch block. This ensures consistent
  // placement of all induction updates.
  auto *LoopVectorLatch = LI->getLoopFor(LoopVectorBody)->getLoopLatch();
  auto *Br = cast<BranchInst>(LoopVectorLatch->getTerminator());
  auto *ICmp = cast<Instruction>(Br->getCondition());
  LastInduction->moveBefore(ICmp);
  LastInduction->setName("vec.ind.next");

  VecInd->addIncoming(SteppedStart, LoopVectorPreHeader);
  VecInd->addIncoming(LastInduction, LoopVectorLatch);
}

bool InnerLoopVectorizer::shouldScalarizeInstruction(Instruction *I) const {
  return Cost->isScalarAfterVectorization(I, VF) ||
         Cost->isProfitableToScalarize(I, VF);
}

bool InnerLoopVectorizer::needsScalarInduction(Instruction *IV) const {
  if (shouldScalarizeInstruction(IV))
    return true;
  auto isScalarInst = [&](User *U) -> bool {
    auto *I = cast<Instruction>(U);
    return (OrigLoop->contains(I) && shouldScalarizeInstruction(I));
  };
  return llvm::any_of(IV->users(), isScalarInst);
}

void InnerLoopVectorizer::recordVectorLoopValueForInductionCast(
    const InductionDescriptor &ID, const Instruction *EntryVal,
    Value *VectorLoopVal, VPValue *CastDef, VPTransformState &State,
    unsigned Part, unsigned Lane) {
  assert((isa<PHINode>(EntryVal) || isa<TruncInst>(EntryVal)) &&
         "Expected either an induction phi-node or a truncate of it!");

  // This induction variable is not the phi from the original loop but the
  // newly-created IV based on the proof that casted Phi is equal to the
  // uncasted Phi in the vectorized loop (under a runtime guard possibly). It
  // re-uses the same InductionDescriptor that original IV uses but we don't
  // have to do any recording in this case - that is done when original IV is
  // processed.
  if (isa<TruncInst>(EntryVal))
    return;

  const SmallVectorImpl<Instruction *> &Casts = ID.getCastInsts();
  if (Casts.empty())
    return;
  // Only the first Cast instruction in the Casts vector is of interest.
  // The rest of the Casts (if exist) have no uses outside the
  // induction update chain itself.
  if (Lane < UINT_MAX)
    State.set(CastDef, VectorLoopVal, VPIteration(Part, Lane));
  else
    State.set(CastDef, VectorLoopVal, Part);
}

void InnerLoopVectorizer::widenIntOrFpInduction(PHINode *IV, Value *Start,
                                                TruncInst *Trunc, VPValue *Def,
                                                VPValue *CastDef,
                                                VPTransformState &State) {
  assert((IV->getType()->isIntegerTy() || IV != OldInduction) &&
         "Primary induction variable must have an integer type");

  auto II = Legal->getInductionVars().find(IV);
  assert(II != Legal->getInductionVars().end() && "IV is not an induction");

  auto ID = II->second;
  assert(IV->getType() == ID.getStartValue()->getType() && "Types must match");

  // The value from the original loop to which we are mapping the new induction
  // variable.
  Instruction *EntryVal = Trunc ? cast<Instruction>(Trunc) : IV;

  auto &DL = OrigLoop->getHeader()->getModule()->getDataLayout();

  // Generate code for the induction step. Note that induction steps are
  // required to be loop-invariant
  auto CreateStepValue = [&](const SCEV *Step) -> Value * {
    assert(PSE.getSE()->isLoopInvariant(Step, OrigLoop) &&
           "Induction step should be loop invariant");
    if (PSE.getSE()->isSCEVable(IV->getType())) {
      SCEVExpander Exp(*PSE.getSE(), DL, "induction");
      return Exp.expandCodeFor(Step, Step->getType(),
                               LoopVectorPreHeader->getTerminator());
    }
    return cast<SCEVUnknown>(Step)->getValue();
  };

  // The scalar value to broadcast. This is derived from the canonical
  // induction variable. If a truncation type is given, truncate the canonical
  // induction variable and step. Otherwise, derive these values from the
  // induction descriptor.
  auto CreateScalarIV = [&](Value *&Step) -> Value * {
    Value *ScalarIV = Induction;
    if (IV != OldInduction) {
      ScalarIV = IV->getType()->isIntegerTy()
                     ? Builder.CreateSExtOrTrunc(Induction, IV->getType())
                     : Builder.CreateCast(Instruction::SIToFP, Induction,
                                          IV->getType());
      ScalarIV = emitTransformedIndex(Builder, ScalarIV, PSE.getSE(), DL, ID);
      ScalarIV->setName("offset.idx");
    }
    if (Trunc) {
      auto *TruncType = cast<IntegerType>(Trunc->getType());
      assert(Step->getType()->isIntegerTy() &&
             "Truncation requires an integer step");
      ScalarIV = Builder.CreateTrunc(ScalarIV, TruncType);
      Step = Builder.CreateTrunc(Step, TruncType);
    }
    return ScalarIV;
  };

  // Create the vector values from the scalar IV, in the absence of creating a
  // vector IV.
  auto CreateSplatIV = [&](Value *ScalarIV, Value *Step) {
    Value *Broadcasted = getBroadcastInstrs(ScalarIV);
    for (unsigned Part = 0; Part < UF; ++Part) {
      assert(!VF.isScalable() && "scalable vectors not yet supported.");
      Value *EntryPart =
          getStepVector(Broadcasted, VF.getKnownMinValue() * Part, Step,
                        ID.getInductionOpcode());
      State.set(Def, EntryPart, Part);
      if (Trunc)
        addMetadata(EntryPart, Trunc);
      recordVectorLoopValueForInductionCast(ID, EntryVal, EntryPart, CastDef,
                                            State, Part);
    }
  };

  // Fast-math-flags propagate from the original induction instruction.
  IRBuilder<>::FastMathFlagGuard FMFG(Builder);
  if (ID.getInductionBinOp() && isa<FPMathOperator>(ID.getInductionBinOp()))
    Builder.setFastMathFlags(ID.getInductionBinOp()->getFastMathFlags());

  // Now do the actual transformations, and start with creating the step value.
  Value *Step = CreateStepValue(ID.getStep());
  if (VF.isZero() || VF.isScalar()) {
    Value *ScalarIV = CreateScalarIV(Step);
    CreateSplatIV(ScalarIV, Step);
    return;
  }

  // Determine if we want a scalar version of the induction variable. This is
  // true if the induction variable itself is not widened, or if it has at
  // least one user in the loop that is not widened.
  auto NeedsScalarIV = needsScalarInduction(EntryVal);
  if (!NeedsScalarIV) {
    createVectorIntOrFpInductionPHI(ID, Step, Start, EntryVal, Def, CastDef,
                                    State);
    return;
  }

  // Try to create a new independent vector induction variable. If we can't
  // create the phi node, we will splat the scalar induction variable in each
  // loop iteration.
  if (!shouldScalarizeInstruction(EntryVal)) {
    createVectorIntOrFpInductionPHI(ID, Step, Start, EntryVal, Def, CastDef,
                                    State);
    Value *ScalarIV = CreateScalarIV(Step);
    // Create scalar steps that can be used by instructions we will later
    // scalarize. Note that the addition of the scalar steps will not increase
    // the number of instructions in the loop in the common case prior to
    // InstCombine. We will be trading one vector extract for each scalar step.
    buildScalarSteps(ScalarIV, Step, EntryVal, ID, Def, CastDef, State);
    return;
  }

  // All IV users are scalar instructions, so only emit a scalar IV, not a
  // vectorised IV. Except when we tail-fold, then the splat IV feeds the
  // predicate used by the masked loads/stores.
  Value *ScalarIV = CreateScalarIV(Step);
  if (!Cost->isScalarEpilogueAllowed())
    CreateSplatIV(ScalarIV, Step);
  buildScalarSteps(ScalarIV, Step, EntryVal, ID, Def, CastDef, State);
}

Value *InnerLoopVectorizer::getStepVector(Value *Val, int StartIdx, Value *Step,
                                          Instruction::BinaryOps BinOp) {
  // Create and check the types.
  auto *ValVTy = cast<VectorType>(Val->getType());
  ElementCount VLen = ValVTy->getElementCount();

  Type *STy = Val->getType()->getScalarType();
  assert((STy->isIntegerTy() || STy->isFloatingPointTy()) &&
         "Induction Step must be an integer or FP");
  assert(Step->getType() == STy && "Step has wrong type");

  SmallVector<Constant *, 8> Indices;

  // Create a vector of consecutive numbers from zero to VF.
  VectorType *InitVecValVTy = ValVTy;
  Type *InitVecValSTy = STy;
  if (STy->isFloatingPointTy()) {
    InitVecValSTy =
        IntegerType::get(STy->getContext(), STy->getScalarSizeInBits());
    InitVecValVTy = VectorType::get(InitVecValSTy, VLen);
  }
  Value *InitVec = Builder.CreateStepVector(InitVecValVTy);

  // Add on StartIdx
  Value *StartIdxSplat = Builder.CreateVectorSplat(
      VLen, ConstantInt::get(InitVecValSTy, StartIdx));
  InitVec = Builder.CreateAdd(InitVec, StartIdxSplat);

  if (STy->isIntegerTy()) {
    Step = Builder.CreateVectorSplat(VLen, Step);
    assert(Step->getType() == Val->getType() && "Invalid step vec");
    // FIXME: The newly created binary instructions should contain nsw/nuw flags,
    // which can be found from the original scalar operations.
    Step = Builder.CreateMul(InitVec, Step);
    return Builder.CreateAdd(Val, Step, "induction");
  }

  // Floating point induction.
  assert((BinOp == Instruction::FAdd || BinOp == Instruction::FSub) &&
         "Binary Opcode should be specified for FP induction");
  InitVec = Builder.CreateUIToFP(InitVec, ValVTy);
  Step = Builder.CreateVectorSplat(VLen, Step);
  Value *MulOp = Builder.CreateFMul(InitVec, Step);
  return Builder.CreateBinOp(BinOp, Val, MulOp, "induction");
}

void InnerLoopVectorizer::buildScalarSteps(Value *ScalarIV, Value *Step,
                                           Instruction *EntryVal,
                                           const InductionDescriptor &ID,
                                           VPValue *Def, VPValue *CastDef,
                                           VPTransformState &State) {
  // We shouldn't have to build scalar steps if we aren't vectorizing.
  assert(VF.isVector() && "VF should be greater than one");
  // Get the value type and ensure it and the step have the same integer type.
  Type *ScalarIVTy = ScalarIV->getType()->getScalarType();
  assert(ScalarIVTy == Step->getType() &&
         "Val and Step should have the same type");

  // We build scalar steps for both integer and floating-point induction
  // variables. Here, we determine the kind of arithmetic we will perform.
  Instruction::BinaryOps AddOp;
  Instruction::BinaryOps MulOp;
  if (ScalarIVTy->isIntegerTy()) {
    AddOp = Instruction::Add;
    MulOp = Instruction::Mul;
  } else {
    AddOp = ID.getInductionOpcode();
    MulOp = Instruction::FMul;
  }

  // Determine the number of scalars we need to generate for each unroll
  // iteration. If EntryVal is uniform, we only need to generate the first
  // lane. Otherwise, we generate all VF values.
  bool IsUniform =
      Cost->isUniformAfterVectorization(cast<Instruction>(EntryVal), VF);
  unsigned Lanes = IsUniform ? 1 : VF.getKnownMinValue();
  // Compute the scalar steps and save the results in State.
  Type *IntStepTy = IntegerType::get(ScalarIVTy->getContext(),
                                     ScalarIVTy->getScalarSizeInBits());
  Type *VecIVTy = nullptr;
  Value *UnitStepVec = nullptr, *SplatStep = nullptr, *SplatIV = nullptr;
  if (!IsUniform && VF.isScalable()) {
    VecIVTy = VectorType::get(ScalarIVTy, VF);
    UnitStepVec = Builder.CreateStepVector(VectorType::get(IntStepTy, VF));
    SplatStep = Builder.CreateVectorSplat(VF, Step);
    SplatIV = Builder.CreateVectorSplat(VF, ScalarIV);
  }

  for (unsigned Part = 0; Part < UF; ++Part) {
    Value *StartIdx0 =
        createStepForVF(Builder, ConstantInt::get(IntStepTy, Part), VF);

    if (!IsUniform && VF.isScalable()) {
      auto *SplatStartIdx = Builder.CreateVectorSplat(VF, StartIdx0);
      auto *InitVec = Builder.CreateAdd(SplatStartIdx, UnitStepVec);
      if (ScalarIVTy->isFloatingPointTy())
        InitVec = Builder.CreateSIToFP(InitVec, VecIVTy);
      auto *Mul = Builder.CreateBinOp(MulOp, InitVec, SplatStep);
      auto *Add = Builder.CreateBinOp(AddOp, SplatIV, Mul);
      State.set(Def, Add, Part);
      recordVectorLoopValueForInductionCast(ID, EntryVal, Add, CastDef, State,
                                            Part);
      // It's useful to record the lane values too for the known minimum number
      // of elements so we do those below. This improves the code quality when
      // trying to extract the first element, for example.
    }

    if (ScalarIVTy->isFloatingPointTy())
      StartIdx0 = Builder.CreateSIToFP(StartIdx0, ScalarIVTy);

    for (unsigned Lane = 0; Lane < Lanes; ++Lane) {
      Value *StartIdx = Builder.CreateBinOp(
          AddOp, StartIdx0, getSignedIntOrFpConstant(ScalarIVTy, Lane));
      // The step returned by `createStepForVF` is a runtime-evaluated value
      // when VF is scalable. Otherwise, it should be folded into a Constant.
      assert((VF.isScalable() || isa<Constant>(StartIdx)) &&
             "Expected StartIdx to be folded to a constant when VF is not "
             "scalable");
      auto *Mul = Builder.CreateBinOp(MulOp, StartIdx, Step);
      auto *Add = Builder.CreateBinOp(AddOp, ScalarIV, Mul);
      State.set(Def, Add, VPIteration(Part, Lane));
      recordVectorLoopValueForInductionCast(ID, EntryVal, Add, CastDef, State,
                                            Part, Lane);
    }
  }
}

void InnerLoopVectorizer::packScalarIntoVectorValue(VPValue *Def,
                                                    const VPIteration &Instance,
                                                    VPTransformState &State) {
  Value *ScalarInst = State.get(Def, Instance);
  Value *VectorValue = State.get(Def, Instance.Part);
  VectorValue = Builder.CreateInsertElement(
      VectorValue, ScalarInst,
      Instance.Lane.getAsRuntimeExpr(State.Builder, VF));
  State.set(Def, VectorValue, Instance.Part);
}

Value *InnerLoopVectorizer::reverseVector(Value *Vec) {
  assert(Vec->getType()->isVectorTy() && "Invalid type");
  return Builder.CreateVectorReverse(Vec, "reverse");
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

// Try to vectorize the interleave group that \p Instr belongs to.
//
// E.g. Translate following interleaved load group (factor = 3):
//   for (i = 0; i < N; i+=3) {
//     R = Pic[i];             // Member of index 0
//     G = Pic[i+1];           // Member of index 1
//     B = Pic[i+2];           // Member of index 2
//     ... // do something to R, G, B
//   }
// To:
//   %wide.vec = load <12 x i32>                       ; Read 4 tuples of R,G,B
//   %R.vec = shuffle %wide.vec, poison, <0, 3, 6, 9>   ; R elements
//   %G.vec = shuffle %wide.vec, poison, <1, 4, 7, 10>  ; G elements
//   %B.vec = shuffle %wide.vec, poison, <2, 5, 8, 11>  ; B elements
//
// Or translate following interleaved store group (factor = 3):
//   for (i = 0; i < N; i+=3) {
//     ... do something to R, G, B
//     Pic[i]   = R;           // Member of index 0
//     Pic[i+1] = G;           // Member of index 1
//     Pic[i+2] = B;           // Member of index 2
//   }
// To:
//   %R_G.vec = shuffle %R.vec, %G.vec, <0, 1, 2, ..., 7>
//   %B_U.vec = shuffle %B.vec, poison, <0, 1, 2, 3, u, u, u, u>
//   %interleaved.vec = shuffle %R_G.vec, %B_U.vec,
//        <0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11>    ; Interleave R,G,B elements
//   store <12 x i32> %interleaved.vec              ; Write 4 tuples of R,G,B
void InnerLoopVectorizer::vectorizeInterleaveGroup(
    const InterleaveGroup<Instruction> *Group, ArrayRef<VPValue *> VPDefs,
    VPTransformState &State, VPValue *Addr, ArrayRef<VPValue *> StoredValues,
    VPValue *BlockInMask) {
  Instruction *Instr = Group->getInsertPos();
  const DataLayout &DL = Instr->getModule()->getDataLayout();

  // Prepare for the vector type of the interleaved load/store.
  Type *ScalarTy = getLoadStoreType(Instr);
  unsigned InterleaveFactor = Group->getFactor();
  assert(!VF.isScalable() && "scalable vectors not yet supported.");
  auto *VecTy = VectorType::get(ScalarTy, VF * InterleaveFactor);

  // Prepare for the new pointers.
  SmallVector<Value *, 2> AddrParts;
  unsigned Index = Group->getIndex(Instr);

  // TODO: extend the masked interleaved-group support to reversed access.
  assert((!BlockInMask || !Group->isReverse()) &&
         "Reversed masked interleave-group not supported.");

  // If the group is reverse, adjust the index to refer to the last vector lane
  // instead of the first. We adjust the index from the first vector lane,
  // rather than directly getting the pointer for lane VF - 1, because the
  // pointer operand of the interleaved access is supposed to be uniform. For
  // uniform instructions, we're only required to generate a value for the
  // first vector lane in each unroll iteration.
  if (Group->isReverse())
    Index += (VF.getKnownMinValue() - 1) * Group->getFactor();

  for (unsigned Part = 0; Part < UF; Part++) {
    Value *AddrPart = State.get(Addr, VPIteration(Part, 0));
    setDebugLocFromInst(AddrPart);

    // Notice current instruction could be any index. Need to adjust the address
    // to the member of index 0.
    //
    // E.g.  a = A[i+1];     // Member of index 1 (Current instruction)
    //       b = A[i];       // Member of index 0
    // Current pointer is pointed to A[i+1], adjust it to A[i].
    //
    // E.g.  A[i+1] = a;     // Member of index 1
    //       A[i]   = b;     // Member of index 0
    //       A[i+2] = c;     // Member of index 2 (Current instruction)
    // Current pointer is pointed to A[i+2], adjust it to A[i].

    bool InBounds = false;
    if (auto *gep = dyn_cast<GetElementPtrInst>(AddrPart->stripPointerCasts()))
      InBounds = gep->isInBounds();
    AddrPart = Builder.CreateGEP(ScalarTy, AddrPart, Builder.getInt32(-Index));
    cast<GetElementPtrInst>(AddrPart)->setIsInBounds(InBounds);

    // Cast to the vector pointer type.
    unsigned AddressSpace = AddrPart->getType()->getPointerAddressSpace();
    Type *PtrTy = VecTy->getPointerTo(AddressSpace);
    AddrParts.push_back(Builder.CreateBitCast(AddrPart, PtrTy));
  }

  setDebugLocFromInst(Instr);
  Value *PoisonVec = PoisonValue::get(VecTy);

  Value *MaskForGaps = nullptr;
  if (Group->requiresScalarEpilogue() && !Cost->isScalarEpilogueAllowed()) {
    MaskForGaps = createBitMaskForGaps(Builder, VF.getKnownMinValue(), *Group);
    assert(MaskForGaps && "Mask for Gaps is required but it is null");
  }

  // Vectorize the interleaved load group.
  if (isa<LoadInst>(Instr)) {
    // For each unroll part, create a wide load for the group.
    SmallVector<Value *, 2> NewLoads;
    for (unsigned Part = 0; Part < UF; Part++) {
      Instruction *NewLoad;
      if (BlockInMask || MaskForGaps) {
        assert(useMaskedInterleavedAccesses(*TTI) &&
               "masked interleaved groups are not allowed.");
        Value *GroupMask = MaskForGaps;
        if (BlockInMask) {
          Value *BlockInMaskPart = State.get(BlockInMask, Part);
          Value *ShuffledMask = Builder.CreateShuffleVector(
              BlockInMaskPart,
              createReplicatedMask(InterleaveFactor, VF.getKnownMinValue()),
              "interleaved.mask");
          GroupMask = MaskForGaps
                          ? Builder.CreateBinOp(Instruction::And, ShuffledMask,
                                                MaskForGaps)
                          : ShuffledMask;
        }
        NewLoad =
            Builder.CreateMaskedLoad(AddrParts[Part], Group->getAlign(),
                                     GroupMask, PoisonVec, "wide.masked.vec");
      }
      else
        NewLoad = Builder.CreateAlignedLoad(VecTy, AddrParts[Part],
                                            Group->getAlign(), "wide.vec");
      Group->addMetadata(NewLoad);
      NewLoads.push_back(NewLoad);
    }

    // For each member in the group, shuffle out the appropriate data from the
    // wide loads.
    unsigned J = 0;
    for (unsigned I = 0; I < InterleaveFactor; ++I) {
      Instruction *Member = Group->getMember(I);

      // Skip the gaps in the group.
      if (!Member)
        continue;

      auto StrideMask =
          createStrideMask(I, InterleaveFactor, VF.getKnownMinValue());
      for (unsigned Part = 0; Part < UF; Part++) {
        Value *StridedVec = Builder.CreateShuffleVector(
            NewLoads[Part], StrideMask, "strided.vec");

        // If this member has different type, cast the result type.
        if (Member->getType() != ScalarTy) {
          assert(!VF.isScalable() && "VF is assumed to be non scalable.");
          VectorType *OtherVTy = VectorType::get(Member->getType(), VF);
          StridedVec = createBitOrPointerCast(StridedVec, OtherVTy, DL);
        }

        if (Group->isReverse())
          StridedVec = reverseVector(StridedVec);

        State.set(VPDefs[J], StridedVec, Part);
      }
      ++J;
    }
    return;
  }

  // The sub vector type for current instruction.
  auto *SubVT = VectorType::get(ScalarTy, VF);

  // Vectorize the interleaved store group.
  for (unsigned Part = 0; Part < UF; Part++) {
    // Collect the stored vector from each member.
    SmallVector<Value *, 4> StoredVecs;
    for (unsigned i = 0; i < InterleaveFactor; i++) {
      // Interleaved store group doesn't allow a gap, so each index has a member
      assert(Group->getMember(i) && "Fail to get a member from an interleaved store group");

      Value *StoredVec = State.get(StoredValues[i], Part);

      if (Group->isReverse())
        StoredVec = reverseVector(StoredVec);

      // If this member has different type, cast it to a unified type.

      if (StoredVec->getType() != SubVT)
        StoredVec = createBitOrPointerCast(StoredVec, SubVT, DL);

      StoredVecs.push_back(StoredVec);
    }

    // Concatenate all vectors into a wide vector.
    Value *WideVec = concatenateVectors(Builder, StoredVecs);

    // Interleave the elements in the wide vector.
    Value *IVec = Builder.CreateShuffleVector(
        WideVec, createInterleaveMask(VF.getKnownMinValue(), InterleaveFactor),
        "interleaved.vec");

    Instruction *NewStoreInstr;
    if (BlockInMask) {
      Value *BlockInMaskPart = State.get(BlockInMask, Part);
      Value *ShuffledMask = Builder.CreateShuffleVector(
          BlockInMaskPart,
          createReplicatedMask(InterleaveFactor, VF.getKnownMinValue()),
          "interleaved.mask");
      NewStoreInstr = Builder.CreateMaskedStore(
          IVec, AddrParts[Part], Group->getAlign(), ShuffledMask);
    }
    else
      NewStoreInstr =
          Builder.CreateAlignedStore(IVec, AddrParts[Part], Group->getAlign());

    Group->addMetadata(NewStoreInstr);
  }
}

void InnerLoopVectorizer::vectorizeMemoryInstruction(
    Instruction *Instr, VPTransformState &State, VPValue *Def, VPValue *Addr,
    VPValue *StoredValue, VPValue *BlockInMask) {
  // Attempt to issue a wide load.
  LoadInst *LI = dyn_cast<LoadInst>(Instr);
  StoreInst *SI = dyn_cast<StoreInst>(Instr);

  assert((LI || SI) && "Invalid Load/Store instruction");
  assert((!SI || StoredValue) && "No stored value provided for widened store");
  assert((!LI || !StoredValue) && "Stored value provided for widened load");

  LoopVectorizationCostModel::InstWidening Decision =
      Cost->getWideningDecision(Instr, VF);
  assert((Decision == LoopVectorizationCostModel::CM_Widen ||
          Decision == LoopVectorizationCostModel::CM_Widen_Reverse ||
          Decision == LoopVectorizationCostModel::CM_GatherScatter) &&
         "CM decision is not to widen the memory instruction");

  Type *ScalarDataTy = getLoadStoreType(Instr);

  auto *DataTy = VectorType::get(ScalarDataTy, VF);
  const Align Alignment = getLoadStoreAlignment(Instr);

  // Determine if the pointer operand of the access is either consecutive or
  // reverse consecutive.
  bool Reverse = (Decision == LoopVectorizationCostModel::CM_Widen_Reverse);
  bool ConsecutiveStride =
      Reverse || (Decision == LoopVectorizationCostModel::CM_Widen);
  bool CreateGatherScatter =
      (Decision == LoopVectorizationCostModel::CM_GatherScatter);

  // Either Ptr feeds a vector load/store, or a vector GEP should feed a vector
  // gather/scatter. Otherwise Decision should have been to Scalarize.
  assert((ConsecutiveStride || CreateGatherScatter) &&
         "The instruction should be scalarized");
  (void)ConsecutiveStride;

  VectorParts BlockInMaskParts(UF);
  bool isMaskRequired = BlockInMask;
  if (isMaskRequired)
    for (unsigned Part = 0; Part < UF; ++Part)
      BlockInMaskParts[Part] = State.get(BlockInMask, Part);

  const auto CreateVecPtr = [&](unsigned Part, Value *Ptr) -> Value * {
    // Calculate the pointer for the specific unroll-part.
    GetElementPtrInst *PartPtr = nullptr;

    bool InBounds = false;
    if (auto *gep = dyn_cast<GetElementPtrInst>(Ptr->stripPointerCasts()))
      InBounds = gep->isInBounds();
    if (Reverse) {
      // If the address is consecutive but reversed, then the
      // wide store needs to start at the last vector element.
      // RunTimeVF =  VScale * VF.getKnownMinValue()
      // For fixed-width VScale is 1, then RunTimeVF = VF.getKnownMinValue()
      Value *RunTimeVF = getRuntimeVF(Builder, Builder.getInt32Ty(), VF);
      // NumElt = -Part * RunTimeVF
      Value *NumElt = Builder.CreateMul(Builder.getInt32(-Part), RunTimeVF);
      // LastLane = 1 - RunTimeVF
      Value *LastLane = Builder.CreateSub(Builder.getInt32(1), RunTimeVF);
      PartPtr =
          cast<GetElementPtrInst>(Builder.CreateGEP(ScalarDataTy, Ptr, NumElt));
      PartPtr->setIsInBounds(InBounds);
      PartPtr = cast<GetElementPtrInst>(
          Builder.CreateGEP(ScalarDataTy, PartPtr, LastLane));
      PartPtr->setIsInBounds(InBounds);
      if (isMaskRequired) // Reverse of a null all-one mask is a null mask.
        BlockInMaskParts[Part] = reverseVector(BlockInMaskParts[Part]);
    } else {
      Value *Increment = createStepForVF(Builder, Builder.getInt32(Part), VF);
      PartPtr = cast<GetElementPtrInst>(
          Builder.CreateGEP(ScalarDataTy, Ptr, Increment));
      PartPtr->setIsInBounds(InBounds);
    }

    unsigned AddressSpace = Ptr->getType()->getPointerAddressSpace();
    return Builder.CreateBitCast(PartPtr, DataTy->getPointerTo(AddressSpace));
  };

  // Handle Stores:
  if (SI) {
    setDebugLocFromInst(SI);

    for (unsigned Part = 0; Part < UF; ++Part) {
      Instruction *NewSI = nullptr;
      Value *StoredVal = State.get(StoredValue, Part);
      if (CreateGatherScatter) {
        Value *MaskPart = isMaskRequired ? BlockInMaskParts[Part] : nullptr;
        Value *VectorGep = State.get(Addr, Part);
        NewSI = Builder.CreateMaskedScatter(StoredVal, VectorGep, Alignment,
                                            MaskPart);
      } else {
        if (Reverse) {
          // If we store to reverse consecutive memory locations, then we need
          // to reverse the order of elements in the stored value.
          StoredVal = reverseVector(StoredVal);
          // We don't want to update the value in the map as it might be used in
          // another expression. So don't call resetVectorValue(StoredVal).
        }
        auto *VecPtr = CreateVecPtr(Part, State.get(Addr, VPIteration(0, 0)));
        if (isMaskRequired)
          NewSI = Builder.CreateMaskedStore(StoredVal, VecPtr, Alignment,
                                            BlockInMaskParts[Part]);
        else
          NewSI = Builder.CreateAlignedStore(StoredVal, VecPtr, Alignment);
      }
      addMetadata(NewSI, SI);
    }
    return;
  }

  // Handle loads.
  assert(LI && "Must have a load instruction");
  setDebugLocFromInst(LI);
  for (unsigned Part = 0; Part < UF; ++Part) {
    Value *NewLI;
    if (CreateGatherScatter) {
      Value *MaskPart = isMaskRequired ? BlockInMaskParts[Part] : nullptr;
      Value *VectorGep = State.get(Addr, Part);
      NewLI = Builder.CreateMaskedGather(VectorGep, Alignment, MaskPart,
                                         nullptr, "wide.masked.gather");
      addMetadata(NewLI, LI);
    } else {
      auto *VecPtr = CreateVecPtr(Part, State.get(Addr, VPIteration(0, 0)));
      if (isMaskRequired)
        NewLI = Builder.CreateMaskedLoad(
            VecPtr, Alignment, BlockInMaskParts[Part], PoisonValue::get(DataTy),
            "wide.masked.load");
      else
        NewLI =
            Builder.CreateAlignedLoad(DataTy, VecPtr, Alignment, "wide.load");

      // Add metadata to the load, but setVectorValue to the reverse shuffle.
      addMetadata(NewLI, LI);
      if (Reverse)
        NewLI = reverseVector(NewLI);
    }

    State.set(Def, NewLI, Part);
  }
}

void InnerLoopVectorizer::scalarizeInstruction(Instruction *Instr, VPValue *Def,
                                               VPUser &User,
                                               const VPIteration &Instance,
                                               bool IfPredicateInstr,
                                               VPTransformState &State) {
  assert(!Instr->getType()->isAggregateType() && "Can't handle vectors");

  // llvm.experimental.noalias.scope.decl intrinsics must only be duplicated for
  // the first lane and part.
  if (isa<NoAliasScopeDeclInst>(Instr))
    if (!Instance.isFirstIteration())
      return;

  setDebugLocFromInst(Instr);

  // Does this instruction return a value ?
  bool IsVoidRetTy = Instr->getType()->isVoidTy();

  Instruction *Cloned = Instr->clone();
  if (!IsVoidRetTy)
    Cloned->setName(Instr->getName() + ".cloned");

  State.Builder.SetInsertPoint(Builder.GetInsertBlock(),
                               Builder.GetInsertPoint());
  // Replace the operands of the cloned instructions with their scalar
  // equivalents in the new loop.
  for (unsigned op = 0, e = User.getNumOperands(); op != e; ++op) {
    auto *Operand = dyn_cast<Instruction>(Instr->getOperand(op));
    auto InputInstance = Instance;
    if (!Operand || !OrigLoop->contains(Operand) ||
        (Cost->isUniformAfterVectorization(Operand, State.VF)))
      InputInstance.Lane = VPLane::getFirstLane();
    auto *NewOp = State.get(User.getOperand(op), InputInstance);
    Cloned->setOperand(op, NewOp);
  }
  addNewMetadata(Cloned, Instr);

  // Place the cloned scalar in the new loop.
  Builder.Insert(Cloned);

  State.set(Def, Cloned, Instance);

  // If we just cloned a new assumption, add it the assumption cache.
  if (auto *II = dyn_cast<AssumeInst>(Cloned))
    AC->registerAssumption(II);

  // End if-block.
  if (IfPredicateInstr)
    PredicatedInstructions.push_back(Cloned);
}

PHINode *InnerLoopVectorizer::createInductionVariable(Loop *L, Value *Start,
                                                      Value *End, Value *Step,
                                                      Instruction *DL) {
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  // As we're just creating this loop, it's possible no latch exists
  // yet. If so, use the header as this will be a single block loop.
  if (!Latch)
    Latch = Header;

  IRBuilder<> B(&*Header->getFirstInsertionPt());
  Instruction *OldInst = getDebugLocFromInstOrOperands(OldInduction);
  setDebugLocFromInst(OldInst, &B);
  auto *Induction = B.CreatePHI(Start->getType(), 2, "index");

  B.SetInsertPoint(Latch->getTerminator());
  setDebugLocFromInst(OldInst, &B);

  // Create i+1 and fill the PHINode.
  //
  // If the tail is not folded, we know that End - Start >= Step (either
  // statically or through the minimum iteration checks). We also know that both
  // Start % Step == 0 and End % Step == 0. We exit the vector loop if %IV +
  // %Step == %End. Hence we must exit the loop before %IV + %Step unsigned
  // overflows and we can mark the induction increment as NUW.
  Value *Next = B.CreateAdd(Induction, Step, "index.next",
                            /*NUW=*/!Cost->foldTailByMasking(), /*NSW=*/false);
  Induction->addIncoming(Start, L->getLoopPreheader());
  Induction->addIncoming(Next, Latch);
  // Create the compare.
  Value *ICmp = B.CreateICmpEQ(Next, End);
  B.CreateCondBr(ICmp, L->getUniqueExitBlock(), Header);

  // Now we have two terminators. Remove the old one from the block.
  Latch->getTerminator()->eraseFromParent();

  return Induction;
}

Value *InnerLoopVectorizer::getOrCreateTripCount(Loop *L) {
  if (TripCount)
    return TripCount;

  assert(L && "Create Trip Count for null loop.");
  IRBuilder<> Builder(L->getLoopPreheader()->getTerminator());
  // Find the loop boundaries.
  ScalarEvolution *SE = PSE.getSE();
  const SCEV *BackedgeTakenCount = PSE.getBackedgeTakenCount();
  assert(!isa<SCEVCouldNotCompute>(BackedgeTakenCount) &&
         "Invalid loop count");

  Type *IdxTy = Legal->getWidestInductionType();
  assert(IdxTy && "No type for induction");

  // The exit count might have the type of i64 while the phi is i32. This can
  // happen if we have an induction variable that is sign extended before the
  // compare. The only way that we get a backedge taken count is that the
  // induction variable was signed and as such will not overflow. In such a case
  // truncation is legal.
  if (SE->getTypeSizeInBits(BackedgeTakenCount->getType()) >
      IdxTy->getPrimitiveSizeInBits())
    BackedgeTakenCount = SE->getTruncateOrNoop(BackedgeTakenCount, IdxTy);
  BackedgeTakenCount = SE->getNoopOrZeroExtend(BackedgeTakenCount, IdxTy);

  // Get the total trip count from the count by adding 1.
  const SCEV *ExitCount = SE->getAddExpr(
      BackedgeTakenCount, SE->getOne(BackedgeTakenCount->getType()));

  const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();

  // Expand the trip count and place the new instructions in the preheader.
  // Notice that the pre-header does not change, only the loop body.
  SCEVExpander Exp(*SE, DL, "induction");

  // Count holds the overall loop count (N).
  TripCount = Exp.expandCodeFor(ExitCount, ExitCount->getType(),
                                L->getLoopPreheader()->getTerminator());

  if (TripCount->getType()->isPointerTy())
    TripCount =
        CastInst::CreatePointerCast(TripCount, IdxTy, "exitcount.ptrcnt.to.int",
                                    L->getLoopPreheader()->getTerminator());

  return TripCount;
}

Value *InnerLoopVectorizer::getOrCreateVectorTripCount(Loop *L) {
  if (VectorTripCount)
    return VectorTripCount;

  Value *TC = getOrCreateTripCount(L);
  IRBuilder<> Builder(L->getLoopPreheader()->getTerminator());

  Type *Ty = TC->getType();
  // This is where we can make the step a runtime constant.
  Value *Step = createStepForVF(Builder, ConstantInt::get(Ty, UF), VF);

  // If the tail is to be folded by masking, round the number of iterations N
  // up to a multiple of Step instead of rounding down. This is done by first
  // adding Step-1 and then rounding down. Note that it's ok if this addition
  // overflows: the vector induction variable will eventually wrap to zero given
  // that it starts at zero and its Step is a power of two; the loop will then
  // exit, with the last early-exit vector comparison also producing all-true.
  if (Cost->foldTailByMasking()) {
    assert(isPowerOf2_32(VF.getKnownMinValue() * UF) &&
           "VF*UF must be a power of 2 when folding tail by masking");
    assert(!VF.isScalable() &&
           "Tail folding not yet supported for scalable vectors");
    TC = Builder.CreateAdd(
        TC, ConstantInt::get(Ty, VF.getKnownMinValue() * UF - 1), "n.rnd.up");
  }

  // Now we need to generate the expression for the part of the loop that the
  // vectorized body will execute. This is equal to N - (N % Step) if scalar
  // iterations are not required for correctness, or N - Step, otherwise. Step
  // is equal to the vectorization factor (number of SIMD elements) times the
  // unroll factor (number of SIMD instructions).
  Value *R = Builder.CreateURem(TC, Step, "n.mod.vf");

  // There are cases where we *must* run at least one iteration in the remainder
  // loop.  See the cost model for when this can happen.  If the step evenly
  // divides the trip count, we set the remainder to be equal to the step. If
  // the step does not evenly divide the trip count, no adjustment is necessary
  // since there will already be scalar iterations. Note that the minimum
  // iterations check ensures that N >= Step.
  if (Cost->requiresScalarEpilogue(VF)) {
    auto *IsZero = Builder.CreateICmpEQ(R, ConstantInt::get(R->getType(), 0));
    R = Builder.CreateSelect(IsZero, Step, R);
  }

  VectorTripCount = Builder.CreateSub(TC, R, "n.vec");

  return VectorTripCount;
}

Value *InnerLoopVectorizer::createBitOrPointerCast(Value *V, VectorType *DstVTy,
                                                   const DataLayout &DL) {
  // Verify that V is a vector type with same number of elements as DstVTy.
  auto *DstFVTy = cast<FixedVectorType>(DstVTy);
  unsigned VF = DstFVTy->getNumElements();
  auto *SrcVecTy = cast<FixedVectorType>(V->getType());
  assert((VF == SrcVecTy->getNumElements()) && "Vector dimensions do not match");
  Type *SrcElemTy = SrcVecTy->getElementType();
  Type *DstElemTy = DstFVTy->getElementType();
  assert((DL.getTypeSizeInBits(SrcElemTy) == DL.getTypeSizeInBits(DstElemTy)) &&
         "Vector elements must have same size");

  // Do a direct cast if element types are castable.
  if (CastInst::isBitOrNoopPointerCastable(SrcElemTy, DstElemTy, DL)) {
    return Builder.CreateBitOrPointerCast(V, DstFVTy);
  }
  // V cannot be directly casted to desired vector type.
  // May happen when V is a floating point vector but DstVTy is a vector of
  // pointers or vice-versa. Handle this using a two-step bitcast using an
  // intermediate Integer type for the bitcast i.e. Ptr <-> Int <-> Float.
  assert((DstElemTy->isPointerTy() != SrcElemTy->isPointerTy()) &&
         "Only one type should be a pointer type");
  assert((DstElemTy->isFloatingPointTy() != SrcElemTy->isFloatingPointTy()) &&
         "Only one type should be a floating point type");
  Type *IntTy =
      IntegerType::getIntNTy(V->getContext(), DL.getTypeSizeInBits(SrcElemTy));
  auto *VecIntTy = FixedVectorType::get(IntTy, VF);
  Value *CastVal = Builder.CreateBitOrPointerCast(V, VecIntTy);
  return Builder.CreateBitOrPointerCast(CastVal, DstFVTy);
}

void InnerLoopVectorizer::emitMinimumIterationCountCheck(Loop *L,
                                                         BasicBlock *Bypass) {
  Value *Count = getOrCreateTripCount(L);
  // Reuse existing vector loop preheader for TC checks.
  // Note that new preheader block is generated for vector loop.
  BasicBlock *const TCCheckBlock = LoopVectorPreHeader;
  IRBuilder<> Builder(TCCheckBlock->getTerminator());

  // Generate code to check if the loop's trip count is less than VF * UF, or
  // equal to it in case a scalar epilogue is required; this implies that the
  // vector trip count is zero. This check also covers the case where adding one
  // to the backedge-taken count overflowed leading to an incorrect trip count
  // of zero. In this case we will also jump to the scalar loop.
  auto P = Cost->requiresScalarEpilogue(VF) ? ICmpInst::ICMP_ULE
                                            : ICmpInst::ICMP_ULT;

  // If tail is to be folded, vector loop takes care of all iterations.
  Value *CheckMinIters = Builder.getFalse();
  if (!Cost->foldTailByMasking()) {
    Value *Step =
        createStepForVF(Builder, ConstantInt::get(Count->getType(), UF), VF);
    CheckMinIters = Builder.CreateICmp(P, Count, Step, "min.iters.check");
  }
  // Create new preheader for vector loop.
  LoopVectorPreHeader =
      SplitBlock(TCCheckBlock, TCCheckBlock->getTerminator(), DT, LI, nullptr,
                 "vector.ph");

  assert(DT->properlyDominates(DT->getNode(TCCheckBlock),
                               DT->getNode(Bypass)->getIDom()) &&
         "TC check is expected to dominate Bypass");

  // Update dominator for Bypass & LoopExit.
  DT->changeImmediateDominator(Bypass, TCCheckBlock);
  DT->changeImmediateDominator(LoopExitBlock, TCCheckBlock);

  ReplaceInstWithInst(
      TCCheckBlock->getTerminator(),
      BranchInst::Create(Bypass, LoopVectorPreHeader, CheckMinIters));
  LoopBypassBlocks.push_back(TCCheckBlock);
}

BasicBlock *InnerLoopVectorizer::emitSCEVChecks(Loop *L, BasicBlock *Bypass) {

  BasicBlock *const SCEVCheckBlock =
      RTChecks.emitSCEVChecks(L, Bypass, LoopVectorPreHeader, LoopExitBlock);
  if (!SCEVCheckBlock)
    return nullptr;

  assert(!(SCEVCheckBlock->getParent()->hasOptSize() ||
           (OptForSizeBasedOnProfile &&
            Cost->Hints->getForce() != LoopVectorizeHints::FK_Enabled)) &&
         "Cannot SCEV check stride or overflow when optimizing for size");


  // Update dominator only if this is first RT check.
  if (LoopBypassBlocks.empty()) {
    DT->changeImmediateDominator(Bypass, SCEVCheckBlock);
    DT->changeImmediateDominator(LoopExitBlock, SCEVCheckBlock);
  }

  LoopBypassBlocks.push_back(SCEVCheckBlock);
  AddedSafetyChecks = true;
  return SCEVCheckBlock;
}

BasicBlock *InnerLoopVectorizer::emitMemRuntimeChecks(Loop *L,
                                                      BasicBlock *Bypass) {
  // VPlan-native path does not do any analysis for runtime checks currently.
  if (EnableVPlanNativePath)
    return nullptr;

  BasicBlock *const MemCheckBlock =
      RTChecks.emitMemRuntimeChecks(L, Bypass, LoopVectorPreHeader);

  // Check if we generated code that checks in runtime if arrays overlap. We put
  // the checks into a separate block to make the more common case of few
  // elements faster.
  if (!MemCheckBlock)
    return nullptr;

  if (MemCheckBlock->getParent()->hasOptSize() || OptForSizeBasedOnProfile) {
    assert(Cost->Hints->getForce() == LoopVectorizeHints::FK_Enabled &&
           "Cannot emit memory checks when optimizing for size, unless forced "
           "to vectorize.");
    ORE->emit([&]() {
      return OptimizationRemarkAnalysis(DEBUG_TYPE, "VectorizationCodeSize",
                                        L->getStartLoc(), L->getHeader())
             << "Code-size may be reduced by not forcing "
                "vectorization, or by source-code modifications "
                "eliminating the need for runtime checks "
                "(e.g., adding 'restrict').";
    });
  }

  LoopBypassBlocks.push_back(MemCheckBlock);

  AddedSafetyChecks = true;

  // We currently don't use LoopVersioning for the actual loop cloning but we
  // still use it to add the noalias metadata.
  LVer = std::make_unique<LoopVersioning>(
      *Legal->getLAI(),
      Legal->getLAI()->getRuntimePointerChecking()->getChecks(), OrigLoop, LI,
      DT, PSE.getSE());
  LVer->prepareNoAliasMetadata();
  return MemCheckBlock;
}

Value *InnerLoopVectorizer::emitTransformedIndex(
    IRBuilder<> &B, Value *Index, ScalarEvolution *SE, const DataLayout &DL,
    const InductionDescriptor &ID) const {

  SCEVExpander Exp(*SE, DL, "induction");
  auto Step = ID.getStep();
  auto StartValue = ID.getStartValue();
  assert(Index->getType()->getScalarType() == Step->getType() &&
         "Index scalar type does not match StepValue type");

  // Note: the IR at this point is broken. We cannot use SE to create any new
  // SCEV and then expand it, hoping that SCEV's simplification will give us
  // a more optimal code. Unfortunately, attempt of doing so on invalid IR may
  // lead to various SCEV crashes. So all we can do is to use builder and rely
  // on InstCombine for future simplifications. Here we handle some trivial
  // cases only.
  auto CreateAdd = [&B](Value *X, Value *Y) {
    assert(X->getType() == Y->getType() && "Types don't match!");
    if (auto *CX = dyn_cast<ConstantInt>(X))
      if (CX->isZero())
        return Y;
    if (auto *CY = dyn_cast<ConstantInt>(Y))
      if (CY->isZero())
        return X;
    return B.CreateAdd(X, Y);
  };

  // We allow X to be a vector type, in which case Y will potentially be
  // splatted into a vector with the same element count.
  auto CreateMul = [&B](Value *X, Value *Y) {
    assert(X->getType()->getScalarType() == Y->getType() &&
           "Types don't match!");
    if (auto *CX = dyn_cast<ConstantInt>(X))
      if (CX->isOne())
        return Y;
    if (auto *CY = dyn_cast<ConstantInt>(Y))
      if (CY->isOne())
        return X;
    VectorType *XVTy = dyn_cast<VectorType>(X->getType());
    if (XVTy && !isa<VectorType>(Y->getType()))
      Y = B.CreateVectorSplat(XVTy->getElementCount(), Y);
    return B.CreateMul(X, Y);
  };

  // Get a suitable insert point for SCEV expansion. For blocks in the vector
  // loop, choose the end of the vector loop header (=LoopVectorBody), because
  // the DomTree is not kept up-to-date for additional blocks generated in the
  // vector loop. By using the header as insertion point, we guarantee that the
  // expanded instructions dominate all their uses.
  auto GetInsertPoint = [this, &B]() {
    BasicBlock *InsertBB = B.GetInsertPoint()->getParent();
    if (InsertBB != LoopVectorBody &&
        LI->getLoopFor(LoopVectorBody) == LI->getLoopFor(InsertBB))
      return LoopVectorBody->getTerminator();
    return &*B.GetInsertPoint();
  };

  switch (ID.getKind()) {
  case InductionDescriptor::IK_IntInduction: {
    assert(!isa<VectorType>(Index->getType()) &&
           "Vector indices not supported for integer inductions yet");
    assert(Index->getType() == StartValue->getType() &&
           "Index type does not match StartValue type");
    if (ID.getConstIntStepValue() && ID.getConstIntStepValue()->isMinusOne())
      return B.CreateSub(StartValue, Index);
    auto *Offset = CreateMul(
        Index, Exp.expandCodeFor(Step, Index->getType(), GetInsertPoint()));
    return CreateAdd(StartValue, Offset);
  }
  case InductionDescriptor::IK_PtrInduction: {
    assert(isa<SCEVConstant>(Step) &&
           "Expected constant step for pointer induction");
    return B.CreateGEP(
        StartValue->getType()->getPointerElementType(), StartValue,
        CreateMul(Index,
                  Exp.expandCodeFor(Step, Index->getType()->getScalarType(),
                                    GetInsertPoint())));
  }
  case InductionDescriptor::IK_FpInduction: {
    assert(!isa<VectorType>(Index->getType()) &&
           "Vector indices not supported for FP inductions yet");
    assert(Step->getType()->isFloatingPointTy() && "Expected FP Step value");
    auto InductionBinOp = ID.getInductionBinOp();
    assert(InductionBinOp &&
           (InductionBinOp->getOpcode() == Instruction::FAdd ||
            InductionBinOp->getOpcode() == Instruction::FSub) &&
           "Original bin op should be defined for FP induction");

    Value *StepValue = cast<SCEVUnknown>(Step)->getValue();
    Value *MulExp = B.CreateFMul(StepValue, Index);
    return B.CreateBinOp(InductionBinOp->getOpcode(), StartValue, MulExp,
                         "induction");
  }
  case InductionDescriptor::IK_NoInduction:
    return nullptr;
  }
  llvm_unreachable("invalid enum");
}

Loop *InnerLoopVectorizer::createVectorLoopSkeleton(StringRef Prefix) {
  LoopScalarBody = OrigLoop->getHeader();
  LoopVectorPreHeader = OrigLoop->getLoopPreheader();
  LoopExitBlock = OrigLoop->getUniqueExitBlock();
  assert(LoopExitBlock && "Must have an exit block");
  assert(LoopVectorPreHeader && "Invalid loop structure");

  LoopMiddleBlock =
      SplitBlock(LoopVectorPreHeader, LoopVectorPreHeader->getTerminator(), DT,
                 LI, nullptr, Twine(Prefix) + "middle.block");
  LoopScalarPreHeader =
      SplitBlock(LoopMiddleBlock, LoopMiddleBlock->getTerminator(), DT, LI,
                 nullptr, Twine(Prefix) + "scalar.ph");

  // Set up branch from middle block to the exit and scalar preheader blocks.
  // completeLoopSkeleton will update the condition to use an iteration check,
  // if required to decide whether to execute the remainder.
  BranchInst *BrInst =
      BranchInst::Create(LoopExitBlock, LoopScalarPreHeader, Builder.getTrue());
  auto *ScalarLatchTerm = OrigLoop->getLoopLatch()->getTerminator();
  BrInst->setDebugLoc(ScalarLatchTerm->getDebugLoc());
  ReplaceInstWithInst(LoopMiddleBlock->getTerminator(), BrInst);

  // We intentionally don't let SplitBlock to update LoopInfo since
  // LoopVectorBody should belong to another loop than LoopVectorPreHeader.
  // LoopVectorBody is explicitly added to the correct place few lines later.
  LoopVectorBody =
      SplitBlock(LoopVectorPreHeader, LoopVectorPreHeader->getTerminator(), DT,
                 nullptr, nullptr, Twine(Prefix) + "vector.body");

  // Update dominator for loop exit.
  DT->changeImmediateDominator(LoopExitBlock, LoopMiddleBlock);

  // Create and register the new vector loop.
  Loop *Lp = LI->AllocateLoop();
  Loop *ParentLoop = OrigLoop->getParentLoop();

  // Insert the new loop into the loop nest and register the new basic blocks
  // before calling any utilities such as SCEV that require valid LoopInfo.
  if (ParentLoop) {
    ParentLoop->addChildLoop(Lp);
  } else {
    LI->addTopLevelLoop(Lp);
  }
  Lp->addBasicBlockToLoop(LoopVectorBody, *LI);
  return Lp;
}

void InnerLoopVectorizer::createInductionResumeValues(
    Loop *L, Value *VectorTripCount,
    std::pair<BasicBlock *, Value *> AdditionalBypass) {
  assert(VectorTripCount && L && "Expected valid arguments");
  assert(((AdditionalBypass.first && AdditionalBypass.second) ||
          (!AdditionalBypass.first && !AdditionalBypass.second)) &&
         "Inconsistent information about additional bypass.");
  // We are going to resume the execution of the scalar loop.
  // Go over all of the induction variables that we found and fix the
  // PHIs that are left in the scalar version of the loop.
  // The starting values of PHI nodes depend on the counter of the last
  // iteration in the vectorized loop.
  // If we come from a bypass edge then we need to start from the original
  // start value.
  for (auto &InductionEntry : Legal->getInductionVars()) {
    PHINode *OrigPhi = InductionEntry.first;
    InductionDescriptor II = InductionEntry.second;

    // Create phi nodes to merge from the  backedge-taken check block.
    PHINode *BCResumeVal =
        PHINode::Create(OrigPhi->getType(), 3, "bc.resume.val",
                        LoopScalarPreHeader->getTerminator());
    // Copy original phi DL over to the new one.
    BCResumeVal->setDebugLoc(OrigPhi->getDebugLoc());
    Value *&EndValue = IVEndValues[OrigPhi];
    Value *EndValueFromAdditionalBypass = AdditionalBypass.second;
    if (OrigPhi == OldInduction) {
      // We know what the end value is.
      EndValue = VectorTripCount;
    } else {
      IRBuilder<> B(L->getLoopPreheader()->getTerminator());

      // Fast-math-flags propagate from the original induction instruction.
      if (II.getInductionBinOp() && isa<FPMathOperator>(II.getInductionBinOp()))
        B.setFastMathFlags(II.getInductionBinOp()->getFastMathFlags());

      Type *StepType = II.getStep()->getType();
      Instruction::CastOps CastOp =
          CastInst::getCastOpcode(VectorTripCount, true, StepType, true);
      Value *CRD = B.CreateCast(CastOp, VectorTripCount, StepType, "cast.crd");
      const DataLayout &DL = LoopScalarBody->getModule()->getDataLayout();
      EndValue = emitTransformedIndex(B, CRD, PSE.getSE(), DL, II);
      EndValue->setName("ind.end");

      // Compute the end value for the additional bypass (if applicable).
      if (AdditionalBypass.first) {
        B.SetInsertPoint(&(*AdditionalBypass.first->getFirstInsertionPt()));
        CastOp = CastInst::getCastOpcode(AdditionalBypass.second, true,
                                         StepType, true);
        CRD =
            B.CreateCast(CastOp, AdditionalBypass.second, StepType, "cast.crd");
        EndValueFromAdditionalBypass =
            emitTransformedIndex(B, CRD, PSE.getSE(), DL, II);
        EndValueFromAdditionalBypass->setName("ind.end");
      }
    }
    // The new PHI merges the original incoming value, in case of a bypass,
    // or the value at the end of the vectorized loop.
    BCResumeVal->addIncoming(EndValue, LoopMiddleBlock);

    // Fix the scalar body counter (PHI node).
    // The old induction's phi node in the scalar body needs the truncated
    // value.
    for (BasicBlock *BB : LoopBypassBlocks)
      BCResumeVal->addIncoming(II.getStartValue(), BB);

    if (AdditionalBypass.first)
      BCResumeVal->setIncomingValueForBlock(AdditionalBypass.first,
                                            EndValueFromAdditionalBypass);

    OrigPhi->setIncomingValueForBlock(LoopScalarPreHeader, BCResumeVal);
  }
}

BasicBlock *InnerLoopVectorizer::completeLoopSkeleton(Loop *L,
                                                      MDNode *OrigLoopID) {
  assert(L && "Expected valid loop.");

  // The trip counts should be cached by now.
  Value *Count = getOrCreateTripCount(L);
  Value *VectorTripCount = getOrCreateVectorTripCount(L);

  auto *ScalarLatchTerm = OrigLoop->getLoopLatch()->getTerminator();

  // Add a check in the middle block to see if we have completed
  // all of the iterations in the first vector loop.
  // If (N - N%VF) == N, then we *don't* need to run the remainder.
  // If tail is to be folded, we know we don't need to run the remainder.
  if (!Cost->foldTailByMasking()) {
    Instruction *CmpN = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_EQ,
                                        Count, VectorTripCount, "cmp.n",
                                        LoopMiddleBlock->getTerminator());

    // Here we use the same DebugLoc as the scalar loop latch terminator instead
    // of the corresponding compare because they may have ended up with
    // different line numbers and we want to avoid awkward line stepping while
    // debugging. Eg. if the compare has got a line number inside the loop.
    CmpN->setDebugLoc(ScalarLatchTerm->getDebugLoc());
    cast<BranchInst>(LoopMiddleBlock->getTerminator())->setCondition(CmpN);
  }

  // Get ready to start creating new instructions into the vectorized body.
  assert(LoopVectorPreHeader == L->getLoopPreheader() &&
         "Inconsistent vector loop preheader");
  Builder.SetInsertPoint(&*LoopVectorBody->getFirstInsertionPt());

  Optional<MDNode *> VectorizedLoopID =
      makeFollowupLoopID(OrigLoopID, {LLVMLoopVectorizeFollowupAll,
                                      LLVMLoopVectorizeFollowupVectorized});
  if (VectorizedLoopID.hasValue()) {
    L->setLoopID(VectorizedLoopID.getValue());

    // Do not setAlreadyVectorized if loop attributes have been defined
    // explicitly.
    return LoopVectorPreHeader;
  }

  // Keep all loop hints from the original loop on the vector loop (we'll
  // replace the vectorizer-specific hints below).
  if (MDNode *LID = OrigLoop->getLoopID())
    L->setLoopID(LID);

  LoopVectorizeHints Hints(L, true, *ORE);
  Hints.setAlreadyVectorized();

#ifdef EXPENSIVE_CHECKS
  assert(DT->verify(DominatorTree::VerificationLevel::Fast));
  LI->verify(*DT);
#endif

  return LoopVectorPreHeader;
}

BasicBlock *InnerLoopVectorizer::createVectorizedLoopSkeleton() {
  /*
   In this function we generate a new loop. The new loop will contain
   the vectorized instructions while the old loop will continue to run the
   scalar remainder.

       [ ] <-- loop iteration number check.
    /   |
   /    v
  |    [ ] <-- vector loop bypass (may consist of multiple blocks).
  |  /  |
  | /   v
  ||   [ ]     <-- vector pre header.
  |/    |
  |     v
  |    [  ] \
  |    [  ]_|   <-- vector loop.
  |     |
  |     v
  |   -[ ]   <--- middle-block.
  |  /  |
  | /   v
  -|- >[ ]     <--- new preheader.
   |    |
   |    v
   |   [ ] \
   |   [ ]_|   <-- old scalar loop to handle remainder.
    \   |
     \  v
      >[ ]     <-- exit block.
   ...
   */

  // Get the metadata of the original loop before it gets modified.
  MDNode *OrigLoopID = OrigLoop->getLoopID();

  // Workaround!  Compute the trip count of the original loop and cache it
  // before we start modifying the CFG.  This code has a systemic problem
  // wherein it tries to run analysis over partially constructed IR; this is
  // wrong, and not simply for SCEV.  The trip count of the original loop
  // simply happens to be prone to hitting this in practice.  In theory, we
  // can hit the same issue for any SCEV, or ValueTracking query done during
  // mutation.  See PR49900.
  getOrCreateTripCount(OrigLoop);

  // Create an empty vector loop, and prepare basic blocks for the runtime
  // checks.
  Loop *Lp = createVectorLoopSkeleton("");

  // Now, compare the new count to zero. If it is zero skip the vector loop and
  // jump to the scalar loop. This check also covers the case where the
  // backedge-taken count is uint##_max: adding one to it will overflow leading
  // to an incorrect trip count of zero. In this (rare) case we will also jump
  // to the scalar loop.
  emitMinimumIterationCountCheck(Lp, LoopScalarPreHeader);

  // Generate the code to check any assumptions that we've made for SCEV
  // expressions.
  emitSCEVChecks(Lp, LoopScalarPreHeader);

  // Generate the code that checks in runtime if arrays overlap. We put the
  // checks into a separate block to make the more common case of few elements
  // faster.
  emitMemRuntimeChecks(Lp, LoopScalarPreHeader);

  // Some loops have a single integer induction variable, while other loops
  // don't. One example is c++ iterators that often have multiple pointer
  // induction variables. In the code below we also support a case where we
  // don't have a single induction variable.
  //
  // We try to obtain an induction variable from the original loop as hard
  // as possible. However if we don't find one that:
  //   - is an integer
  //   - counts from zero, stepping by one
  //   - is the size of the widest induction variable type
  // then we create a new one.
  OldInduction = Legal->getPrimaryInduction();
  Type *IdxTy = Legal->getWidestInductionType();
  Value *StartIdx = ConstantInt::get(IdxTy, 0);
  // The loop step is equal to the vectorization factor (num of SIMD elements)
  // times the unroll factor (num of SIMD instructions).
  Builder.SetInsertPoint(&*Lp->getHeader()->getFirstInsertionPt());
  Value *Step = createStepForVF(Builder, ConstantInt::get(IdxTy, UF), VF);
  Value *CountRoundDown = getOrCreateVectorTripCount(Lp);
  Induction =
      createInductionVariable(Lp, StartIdx, CountRoundDown, Step,
                              getDebugLocFromInstOrOperands(OldInduction));

  // Emit phis for the new starting index of the scalar loop.
  createInductionResumeValues(Lp, CountRoundDown);

  return completeLoopSkeleton(Lp, OrigLoopID);
}

// Fix up external users of the induction variable. At this point, we are
// in LCSSA form, with all external PHIs that use the IV having one input value,
// coming from the remainder loop. We need those PHIs to also have a correct
// value for the IV when arriving directly from the middle block.
void InnerLoopVectorizer::fixupIVUsers(PHINode *OrigPhi,
                                       const InductionDescriptor &II,
                                       Value *CountRoundDown, Value *EndValue,
                                       BasicBlock *MiddleBlock) {
  // There are two kinds of external IV usages - those that use the value
  // computed in the last iteration (the PHI) and those that use the penultimate
  // value (the value that feeds into the phi from the loop latch).
  // We allow both, but they, obviously, have different values.

  assert(OrigLoop->getUniqueExitBlock() && "Expected a single exit block");

  DenseMap<Value *, Value *> MissingVals;

  // An external user of the last iteration's value should see the value that
  // the remainder loop uses to initialize its own IV.
  Value *PostInc = OrigPhi->getIncomingValueForBlock(OrigLoop->getLoopLatch());
  for (User *U : PostInc->users()) {
    Instruction *UI = cast<Instruction>(U);
    if (!OrigLoop->contains(UI)) {
      assert(isa<PHINode>(UI) && "Expected LCSSA form");
      MissingVals[UI] = EndValue;
    }
  }

  // An external user of the penultimate value need to see EndValue - Step.
  // The simplest way to get this is to recompute it from the constituent SCEVs,
  // that is Start + (Step * (CRD - 1)).
  for (User *U : OrigPhi->users()) {
    auto *UI = cast<Instruction>(U);
    if (!OrigLoop->contains(UI)) {
      const DataLayout &DL =
          OrigLoop->getHeader()->getModule()->getDataLayout();
      assert(isa<PHINode>(UI) && "Expected LCSSA form");

      IRBuilder<> B(MiddleBlock->getTerminator());

      // Fast-math-flags propagate from the original induction instruction.
      if (II.getInductionBinOp() && isa<FPMathOperator>(II.getInductionBinOp()))
        B.setFastMathFlags(II.getInductionBinOp()->getFastMathFlags());

      Value *CountMinusOne = B.CreateSub(
          CountRoundDown, ConstantInt::get(CountRoundDown->getType(), 1));
      Value *CMO =
          !II.getStep()->getType()->isIntegerTy()
              ? B.CreateCast(Instruction::SIToFP, CountMinusOne,
                             II.getStep()->getType())
              : B.CreateSExtOrTrunc(CountMinusOne, II.getStep()->getType());
      CMO->setName("cast.cmo");
      Value *Escape = emitTransformedIndex(B, CMO, PSE.getSE(), DL, II);
      Escape->setName("ind.escape");
      MissingVals[UI] = Escape;
    }
  }

  for (auto &I : MissingVals) {
    PHINode *PHI = cast<PHINode>(I.first);
    // One corner case we have to handle is two IVs "chasing" each-other,
    // that is %IV2 = phi [...], [ %IV1, %latch ]
    // In this case, if IV1 has an external use, we need to avoid adding both
    // "last value of IV1" and "penultimate value of IV2". So, verify that we
    // don't already have an incoming value for the middle block.
    if (PHI->getBasicBlockIndex(MiddleBlock) == -1)
      PHI->addIncoming(I.second, MiddleBlock);
  }
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
    return hash_combine(I->getOpcode(), hash_combine_range(I->value_op_begin(),
                                                           I->value_op_end()));
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
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;) {
    Instruction *In = &*I++;

    if (!CSEDenseMapInfo::canHandle(In))
      continue;

    // Check if we can replace this instruction with any of the
    // visited instructions.
    if (Instruction *V = CSEMap.lookup(In)) {
      In->replaceAllUsesWith(V);
      In->eraseFromParent();
      continue;
    }

    CSEMap[In] = In;
  }
}

InstructionCost
LoopVectorizationCostModel::getVectorCallCost(CallInst *CI, ElementCount VF,
                                              bool &NeedToScalarize) const {
  Function *F = CI->getCalledFunction();
  Type *ScalarRetTy = CI->getType();
  SmallVector<Type *, 4> Tys, ScalarTys;
  for (auto &ArgOp : CI->arg_operands())
    ScalarTys.push_back(ArgOp->getType());

  // Estimate cost of scalarized vector call. The source operands are assumed
  // to be vectors, so we need to extract individual elements from there,
  // execute VF scalar calls, and then gather the result into the vector return
  // value.
  InstructionCost ScalarCallCost =
      TTI.getCallInstrCost(F, ScalarRetTy, ScalarTys, TTI::TCK_RecipThroughput);
  if (VF.isScalar())
    return ScalarCallCost;

  // Compute corresponding vector type for return value and arguments.
  Type *RetTy = ToVectorTy(ScalarRetTy, VF);
  for (Type *ScalarTy : ScalarTys)
    Tys.push_back(ToVectorTy(ScalarTy, VF));

  // Compute costs of unpacking argument values for the scalar calls and
  // packing the return values to a vector.
  InstructionCost ScalarizationCost = getScalarizationOverhead(CI, VF);

  InstructionCost Cost =
      ScalarCallCost * VF.getKnownMinValue() + ScalarizationCost;

  // If we can't emit a vector call for this function, then the currently found
  // cost is the cost we need to return.
  NeedToScalarize = true;
  VFShape Shape = VFShape::get(*CI, VF, false /*HasGlobalPred*/);
  Function *VecFunc = VFDatabase(*CI).getVectorizedFunction(Shape);

  if (!TLI || CI->isNoBuiltin() || !VecFunc)
    return Cost;

  // If the corresponding vector cost is cheaper, return its cost.
  InstructionCost VectorCallCost =
      TTI.getCallInstrCost(nullptr, RetTy, Tys, TTI::TCK_RecipThroughput);
  if (VectorCallCost < Cost) {
    NeedToScalarize = false;
    Cost = VectorCallCost;
  }
  return Cost;
}

static Type *MaybeVectorizeType(Type *Elt, ElementCount VF) {
  if (VF.isScalar() || (!Elt->isIntOrPtrTy() && !Elt->isFloatingPointTy()))
    return Elt;
  return VectorType::get(Elt, VF);
}

InstructionCost
LoopVectorizationCostModel::getVectorIntrinsicCost(CallInst *CI,
                                                   ElementCount VF) const {
  Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);
  assert(ID && "Expected intrinsic call!");
  Type *RetTy = MaybeVectorizeType(CI->getType(), VF);
  FastMathFlags FMF;
  if (auto *FPMO = dyn_cast<FPMathOperator>(CI))
    FMF = FPMO->getFastMathFlags();

  SmallVector<const Value *> Arguments(CI->arg_begin(), CI->arg_end());
  FunctionType *FTy = CI->getCalledFunction()->getFunctionType();
  SmallVector<Type *> ParamTys;
  std::transform(FTy->param_begin(), FTy->param_end(),
                 std::back_inserter(ParamTys),
                 [&](Type *Ty) { return MaybeVectorizeType(Ty, VF); });

  IntrinsicCostAttributes CostAttrs(ID, RetTy, Arguments, ParamTys, FMF,
                                    dyn_cast<IntrinsicInst>(CI));
  return TTI.getIntrinsicInstrCost(CostAttrs,
                                   TargetTransformInfo::TCK_RecipThroughput);
}

static Type *smallestIntegerVectorType(Type *T1, Type *T2) {
  auto *I1 = cast<IntegerType>(cast<VectorType>(T1)->getElementType());
  auto *I2 = cast<IntegerType>(cast<VectorType>(T2)->getElementType());
  return I1->getBitWidth() < I2->getBitWidth() ? T1 : T2;
}

static Type *largestIntegerVectorType(Type *T1, Type *T2) {
  auto *I1 = cast<IntegerType>(cast<VectorType>(T1)->getElementType());
  auto *I2 = cast<IntegerType>(cast<VectorType>(T2)->getElementType());
  return I1->getBitWidth() > I2->getBitWidth() ? T1 : T2;
}

void InnerLoopVectorizer::truncateToMinimalBitwidths(VPTransformState &State) {
  // For every instruction `I` in MinBWs, truncate the operands, create a
  // truncated version of `I` and reextend its result. InstCombine runs
  // later and will remove any ext/trunc pairs.
  SmallPtrSet<Value *, 4> Erased;
  for (const auto &KV : Cost->getMinimalBitwidths()) {
    // If the value wasn't vectorized, we must maintain the original scalar
    // type. The absence of the value from State indicates that it
    // wasn't vectorized.
    VPValue *Def = State.Plan->getVPValue(KV.first);
    if (!State.hasAnyVectorValue(Def))
      continue;
    for (unsigned Part = 0; Part < UF; ++Part) {
      Value *I = State.get(Def, Part);
      if (Erased.count(I) || I->use_empty() || !isa<Instruction>(I))
        continue;
      Type *OriginalTy = I->getType();
      Type *ScalarTruncatedTy =
          IntegerType::get(OriginalTy->getContext(), KV.second);
      auto *TruncatedTy = FixedVectorType::get(
          ScalarTruncatedTy,
          cast<FixedVectorType>(OriginalTy)->getNumElements());
      if (TruncatedTy == OriginalTy)
        continue;

      IRBuilder<> B(cast<Instruction>(I));
      auto ShrinkOperand = [&](Value *V) -> Value * {
        if (auto *ZI = dyn_cast<ZExtInst>(V))
          if (ZI->getSrcTy() == TruncatedTy)
            return ZI->getOperand(0);
        return B.CreateZExtOrTrunc(V, TruncatedTy);
      };

      // The actual instruction modification depends on the instruction type,
      // unfortunately.
      Value *NewI = nullptr;
      if (auto *BO = dyn_cast<BinaryOperator>(I)) {
        NewI = B.CreateBinOp(BO->getOpcode(), ShrinkOperand(BO->getOperand(0)),
                             ShrinkOperand(BO->getOperand(1)));

        // Any wrapping introduced by shrinking this operation shouldn't be
        // considered undefined behavior. So, we can't unconditionally copy
        // arithmetic wrapping flags to NewI.
        cast<BinaryOperator>(NewI)->copyIRFlags(I, /*IncludeWrapFlags=*/false);
      } else if (auto *CI = dyn_cast<ICmpInst>(I)) {
        NewI =
            B.CreateICmp(CI->getPredicate(), ShrinkOperand(CI->getOperand(0)),
                         ShrinkOperand(CI->getOperand(1)));
      } else if (auto *SI = dyn_cast<SelectInst>(I)) {
        NewI = B.CreateSelect(SI->getCondition(),
                              ShrinkOperand(SI->getTrueValue()),
                              ShrinkOperand(SI->getFalseValue()));
      } else if (auto *CI = dyn_cast<CastInst>(I)) {
        switch (CI->getOpcode()) {
        default:
          llvm_unreachable("Unhandled cast!");
        case Instruction::Trunc:
          NewI = ShrinkOperand(CI->getOperand(0));
          break;
        case Instruction::SExt:
          NewI = B.CreateSExtOrTrunc(
              CI->getOperand(0),
              smallestIntegerVectorType(OriginalTy, TruncatedTy));
          break;
        case Instruction::ZExt:
          NewI = B.CreateZExtOrTrunc(
              CI->getOperand(0),
              smallestIntegerVectorType(OriginalTy, TruncatedTy));
          break;
        }
      } else if (auto *SI = dyn_cast<ShuffleVectorInst>(I)) {
        auto Elements0 = cast<FixedVectorType>(SI->getOperand(0)->getType())
                             ->getNumElements();
        auto *O0 = B.CreateZExtOrTrunc(
            SI->getOperand(0),
            FixedVectorType::get(ScalarTruncatedTy, Elements0));
        auto Elements1 = cast<FixedVectorType>(SI->getOperand(1)->getType())
                             ->getNumElements();
        auto *O1 = B.CreateZExtOrTrunc(
            SI->getOperand(1),
            FixedVectorType::get(ScalarTruncatedTy, Elements1));

        NewI = B.CreateShuffleVector(O0, O1, SI->getShuffleMask());
      } else if (isa<LoadInst>(I) || isa<PHINode>(I)) {
        // Don't do anything with the operands, just extend the result.
        continue;
      } else if (auto *IE = dyn_cast<InsertElementInst>(I)) {
        auto Elements = cast<FixedVectorType>(IE->getOperand(0)->getType())
                            ->getNumElements();
        auto *O0 = B.CreateZExtOrTrunc(
            IE->getOperand(0),
            FixedVectorType::get(ScalarTruncatedTy, Elements));
        auto *O1 = B.CreateZExtOrTrunc(IE->getOperand(1), ScalarTruncatedTy);
        NewI = B.CreateInsertElement(O0, O1, IE->getOperand(2));
      } else if (auto *EE = dyn_cast<ExtractElementInst>(I)) {
        auto Elements = cast<FixedVectorType>(EE->getOperand(0)->getType())
                            ->getNumElements();
        auto *O0 = B.CreateZExtOrTrunc(
            EE->getOperand(0),
            FixedVectorType::get(ScalarTruncatedTy, Elements));
        NewI = B.CreateExtractElement(O0, EE->getOperand(2));
      } else {
        // If we don't know what to do, be conservative and don't do anything.
        continue;
      }

      // Lastly, extend the result.
      NewI->takeName(cast<Instruction>(I));
      Value *Res = B.CreateZExtOrTrunc(NewI, OriginalTy);
      I->replaceAllUsesWith(Res);
      cast<Instruction>(I)->eraseFromParent();
      Erased.insert(I);
      State.reset(Def, Res, Part);
    }
  }

  // We'll have created a bunch of ZExts that are now parentless. Clean up.
  for (const auto &KV : Cost->getMinimalBitwidths()) {
    // If the value wasn't vectorized, we must maintain the original scalar
    // type. The absence of the value from State indicates that it
    // wasn't vectorized.
    VPValue *Def = State.Plan->getVPValue(KV.first);
    if (!State.hasAnyVectorValue(Def))
      continue;
    for (unsigned Part = 0; Part < UF; ++Part) {
      Value *I = State.get(Def, Part);
      ZExtInst *Inst = dyn_cast<ZExtInst>(I);
      if (Inst && Inst->use_empty()) {
        Value *NewI = Inst->getOperand(0);
        Inst->eraseFromParent();
        State.reset(Def, NewI, Part);
      }
    }
  }
}

void InnerLoopVectorizer::fixVectorizedLoop(VPTransformState &State) {
  // Insert truncates and extends for any truncated instructions as hints to
  // InstCombine.
  if (VF.isVector())
    truncateToMinimalBitwidths(State);

  // Fix widened non-induction PHIs by setting up the PHI operands.
  if (OrigPHIsToFix.size()) {
    assert(EnableVPlanNativePath &&
           "Unexpected non-induction PHIs for fixup in non VPlan-native path");
    fixNonInductionPHIs(State);
  }

  // At this point every instruction in the original loop is widened to a
  // vector form. Now we need to fix the recurrences in the loop. These PHI
  // nodes are currently empty because we did not want to introduce cycles.
  // This is the second stage of vectorizing recurrences.
  fixCrossIterationPHIs(State);

  // Forget the original basic block.
  PSE.getSE()->forgetLoop(OrigLoop);

  // Fix-up external users of the induction variables.
  for (auto &Entry : Legal->getInductionVars())
    fixupIVUsers(Entry.first, Entry.second,
                 getOrCreateVectorTripCount(LI->getLoopFor(LoopVectorBody)),
                 IVEndValues[Entry.first], LoopMiddleBlock);

  fixLCSSAPHIs(State);
  for (Instruction *PI : PredicatedInstructions)
    sinkScalarOperands(&*PI);

  // Remove redundant induction instructions.
  cse(LoopVectorBody);

  // Set/update profile weights for the vector and remainder loops as original
  // loop iterations are now distributed among them. Note that original loop
  // represented by LoopScalarBody becomes remainder loop after vectorization.
  //
  // For cases like foldTailByMasking() and requiresScalarEpiloque() we may
  // end up getting slightly roughened result but that should be OK since
  // profile is not inherently precise anyway. Note also possible bypass of
  // vector code caused by legality checks is ignored, assigning all the weight
  // to the vector loop, optimistically.
  //
  // For scalable vectorization we can't know at compile time how many iterations
  // of the loop are handled in one vector iteration, so instead assume a pessimistic
  // vscale of '1'.
  setProfileInfoAfterUnrolling(
      LI->getLoopFor(LoopScalarBody), LI->getLoopFor(LoopVectorBody),
      LI->getLoopFor(LoopScalarBody), VF.getKnownMinValue() * UF);
}

void InnerLoopVectorizer::fixCrossIterationPHIs(VPTransformState &State) {
  // In order to support recurrences we need to be able to vectorize Phi nodes.
  // Phi nodes have cycles, so we need to vectorize them in two stages. This is
  // stage #2: We now need to fix the recurrences by adding incoming edges to
  // the currently empty PHI nodes. At this point every instruction in the
  // original loop is widened to a vector form so we can use them to construct
  // the incoming edges.
  VPBasicBlock *Header = State.Plan->getEntry()->getEntryBasicBlock();
  for (VPRecipeBase &R : Header->phis()) {
    auto *PhiR = dyn_cast<VPWidenPHIRecipe>(&R);
    if (!PhiR)
      continue;
    auto *OrigPhi = cast<PHINode>(PhiR->getUnderlyingValue());
    if (PhiR->getRecurrenceDescriptor()) {
      fixReduction(PhiR, State);
    } else if (Legal->isFirstOrderRecurrence(OrigPhi))
      fixFirstOrderRecurrence(PhiR, State);
  }
}

void InnerLoopVectorizer::fixFirstOrderRecurrence(VPWidenPHIRecipe *PhiR,
                                                  VPTransformState &State) {
  // This is the second phase of vectorizing first-order recurrences. An
  // overview of the transformation is described below. Suppose we have the
  // following loop.
  //
  //   for (int i = 0; i < n; ++i)
  //     b[i] = a[i] - a[i - 1];
  //
  // There is a first-order recurrence on "a". For this loop, the shorthand
  // scalar IR looks like:
  //
  //   scalar.ph:
  //     s_init = a[-1]
  //     br scalar.body
  //
  //   scalar.body:
  //     i = phi [0, scalar.ph], [i+1, scalar.body]
  //     s1 = phi [s_init, scalar.ph], [s2, scalar.body]
  //     s2 = a[i]
  //     b[i] = s2 - s1
  //     br cond, scalar.body, ...
  //
  // In this example, s1 is a recurrence because it's value depends on the
  // previous iteration. In the first phase of vectorization, we created a
  // temporary value for s1. We now complete the vectorization and produce the
  // shorthand vector IR shown below (for VF = 4, UF = 1).
  //
  //   vector.ph:
  //     v_init = vector(..., ..., ..., a[-1])
  //     br vector.body
  //
  //   vector.body
  //     i = phi [0, vector.ph], [i+4, vector.body]
  //     v1 = phi [v_init, vector.ph], [v2, vector.body]
  //     v2 = a[i, i+1, i+2, i+3];
  //     v3 = vector(v1(3), v2(0, 1, 2))
  //     b[i, i+1, i+2, i+3] = v2 - v3
  //     br cond, vector.body, middle.block
  //
  //   middle.block:
  //     x = v2(3)
  //     br scalar.ph
  //
  //   scalar.ph:
  //     s_init = phi [x, middle.block], [a[-1], otherwise]
  //     br scalar.body
  //
  // After execution completes the vector loop, we extract the next value of
  // the recurrence (x) to use as the initial value in the scalar loop.

  auto *ScalarInit = PhiR->getStartValue()->getLiveInIRValue();

  auto *IdxTy = Builder.getInt32Ty();
  auto *One = ConstantInt::get(IdxTy, 1);

  // Create a vector from the initial value.
  auto *VectorInit = ScalarInit;
  if (VF.isVector()) {
    Builder.SetInsertPoint(LoopVectorPreHeader->getTerminator());
    auto *RuntimeVF = getRuntimeVF(Builder, IdxTy, VF);
    auto *LastIdx = Builder.CreateSub(RuntimeVF, One);
    VectorInit = Builder.CreateInsertElement(
        PoisonValue::get(VectorType::get(VectorInit->getType(), VF)),
        VectorInit, LastIdx, "vector.recur.init");
  }

  VPValue *PreviousDef = PhiR->getBackedgeValue();
  // We constructed a temporary phi node in the first phase of vectorization.
  // This phi node will eventually be deleted.
  Builder.SetInsertPoint(cast<Instruction>(State.get(PhiR, 0)));

  // Create a phi node for the new recurrence. The current value will either be
  // the initial value inserted into a vector or loop-varying vector value.
  auto *VecPhi = Builder.CreatePHI(VectorInit->getType(), 2, "vector.recur");
  VecPhi->addIncoming(VectorInit, LoopVectorPreHeader);

  // Get the vectorized previous value of the last part UF - 1. It appears last
  // among all unrolled iterations, due to the order of their construction.
  Value *PreviousLastPart = State.get(PreviousDef, UF - 1);

  // Find and set the insertion point after the previous value if it is an
  // instruction.
  BasicBlock::iterator InsertPt;
  // Note that the previous value may have been constant-folded so it is not
  // guaranteed to be an instruction in the vector loop.
  // FIXME: Loop invariant values do not form recurrences. We should deal with
  //        them earlier.
  if (LI->getLoopFor(LoopVectorBody)->isLoopInvariant(PreviousLastPart))
    InsertPt = LoopVectorBody->getFirstInsertionPt();
  else {
    Instruction *PreviousInst = cast<Instruction>(PreviousLastPart);
    if (isa<PHINode>(PreviousLastPart))
      // If the previous value is a phi node, we should insert after all the phi
      // nodes in the block containing the PHI to avoid breaking basic block
      // verification. Note that the basic block may be different to
      // LoopVectorBody, in case we predicate the loop.
      InsertPt = PreviousInst->getParent()->getFirstInsertionPt();
    else
      InsertPt = ++PreviousInst->getIterator();
  }
  Builder.SetInsertPoint(&*InsertPt);

  // The vector from which to take the initial value for the current iteration
  // (actual or unrolled). Initially, this is the vector phi node.
  Value *Incoming = VecPhi;

  // Shuffle the current and previous vector and update the vector parts.
  for (unsigned Part = 0; Part < UF; ++Part) {
    Value *PreviousPart = State.get(PreviousDef, Part);
    Value *PhiPart = State.get(PhiR, Part);
    auto *Shuffle = VF.isVector()
                        ? Builder.CreateVectorSplice(Incoming, PreviousPart, -1)
                        : Incoming;
    PhiPart->replaceAllUsesWith(Shuffle);
    cast<Instruction>(PhiPart)->eraseFromParent();
    State.reset(PhiR, Shuffle, Part);
    Incoming = PreviousPart;
  }

  // Fix the latch value of the new recurrence in the vector loop.
  VecPhi->addIncoming(Incoming, LI->getLoopFor(LoopVectorBody)->getLoopLatch());

  // Extract the last vector element in the middle block. This will be the
  // initial value for the recurrence when jumping to the scalar loop.
  auto *ExtractForScalar = Incoming;
  if (VF.isVector()) {
    Builder.SetInsertPoint(LoopMiddleBlock->getTerminator());
    auto *RuntimeVF = getRuntimeVF(Builder, IdxTy, VF);
    auto *LastIdx = Builder.CreateSub(RuntimeVF, One);
    ExtractForScalar = Builder.CreateExtractElement(ExtractForScalar, LastIdx,
                                                    "vector.recur.extract");
  }
  // Extract the second last element in the middle block if the
  // Phi is used outside the loop. We need to extract the phi itself
  // and not the last element (the phi update in the current iteration). This
  // will be the value when jumping to the exit block from the LoopMiddleBlock,
  // when the scalar loop is not run at all.
  Value *ExtractForPhiUsedOutsideLoop = nullptr;
  if (VF.isVector()) {
    auto *RuntimeVF = getRuntimeVF(Builder, IdxTy, VF);
    auto *Idx = Builder.CreateSub(RuntimeVF, ConstantInt::get(IdxTy, 2));
    ExtractForPhiUsedOutsideLoop = Builder.CreateExtractElement(
        Incoming, Idx, "vector.recur.extract.for.phi");
  } else if (UF > 1)
    // When loop is unrolled without vectorizing, initialize
    // ExtractForPhiUsedOutsideLoop with the value just prior to unrolled value
    // of `Incoming`. This is analogous to the vectorized case above: extracting
    // the second last element when VF > 1.
    ExtractForPhiUsedOutsideLoop = State.get(PreviousDef, UF - 2);

  // Fix the initial value of the original recurrence in the scalar loop.
  Builder.SetInsertPoint(&*LoopScalarPreHeader->begin());
  PHINode *Phi = cast<PHINode>(PhiR->getUnderlyingValue());
  auto *Start = Builder.CreatePHI(Phi->getType(), 2, "scalar.recur.init");
  for (auto *BB : predecessors(LoopScalarPreHeader)) {
    auto *Incoming = BB == LoopMiddleBlock ? ExtractForScalar : ScalarInit;
    Start->addIncoming(Incoming, BB);
  }

  Phi->setIncomingValueForBlock(LoopScalarPreHeader, Start);
  Phi->setName("scalar.recur");

  // Finally, fix users of the recurrence outside the loop. The users will need
  // either the last value of the scalar recurrence or the last value of the
  // vector recurrence we extracted in the middle block. Since the loop is in
  // LCSSA form, we just need to find all the phi nodes for the original scalar
  // recurrence in the exit block, and then add an edge for the middle block.
  // Note that LCSSA does not imply single entry when the original scalar loop
  // had multiple exiting edges (as we always run the last iteration in the
  // scalar epilogue); in that case, the exiting path through middle will be
  // dynamically dead and the value picked for the phi doesn't matter.
  for (PHINode &LCSSAPhi : LoopExitBlock->phis())
    if (any_of(LCSSAPhi.incoming_values(),
               [Phi](Value *V) { return V == Phi; }))
      LCSSAPhi.addIncoming(ExtractForPhiUsedOutsideLoop, LoopMiddleBlock);
}

void InnerLoopVectorizer::fixReduction(VPWidenPHIRecipe *PhiR,
                                       VPTransformState &State) {
  PHINode *OrigPhi = cast<PHINode>(PhiR->getUnderlyingValue());
  // Get it's reduction variable descriptor.
  assert(Legal->isReductionVariable(OrigPhi) &&
         "Unable to find the reduction variable");
  const RecurrenceDescriptor &RdxDesc = *PhiR->getRecurrenceDescriptor();

  RecurKind RK = RdxDesc.getRecurrenceKind();
  TrackingVH<Value> ReductionStartValue = RdxDesc.getRecurrenceStartValue();
  Instruction *LoopExitInst = RdxDesc.getLoopExitInstr();
  setDebugLocFromInst(ReductionStartValue);
  bool IsInLoopReductionPhi = Cost->isInLoopReduction(OrigPhi);

  VPValue *LoopExitInstDef = State.Plan->getVPValue(LoopExitInst);
  // This is the vector-clone of the value that leaves the loop.
  Type *VecTy = State.get(LoopExitInstDef, 0)->getType();

  // Wrap flags are in general invalid after vectorization, clear them.
  clearReductionWrapFlags(RdxDesc, State);

  // Fix the vector-loop phi.

  // Reductions do not have to start at zero. They can start with
  // any loop invariant values.
  BasicBlock *VectorLoopLatch = LI->getLoopFor(LoopVectorBody)->getLoopLatch();

  bool IsOrdered = IsInLoopReductionPhi && Cost->useOrderedReductions(RdxDesc);

  for (unsigned Part = 0; Part < UF; ++Part) {
    if (IsOrdered && Part > 0)
      break;
    Value *VecRdxPhi = State.get(PhiR->getVPSingleValue(), Part);
    Value *Val = State.get(PhiR->getBackedgeValue(), Part);
    if (IsOrdered)
      Val = State.get(PhiR->getBackedgeValue(), UF - 1);

    cast<PHINode>(VecRdxPhi)->addIncoming(Val, VectorLoopLatch);
  }

  // Before each round, move the insertion point right between
  // the PHIs and the values we are going to write.
  // This allows us to write both PHINodes and the extractelement
  // instructions.
  Builder.SetInsertPoint(&*LoopMiddleBlock->getFirstInsertionPt());

  setDebugLocFromInst(LoopExitInst);

  Type *PhiTy = OrigPhi->getType();
  // If tail is folded by masking, the vector value to leave the loop should be
  // a Select choosing between the vectorized LoopExitInst and vectorized Phi,
  // instead of the former. For an inloop reduction the reduction will already
  // be predicated, and does not need to be handled here.
  if (Cost->foldTailByMasking() && !IsInLoopReductionPhi) {
    for (unsigned Part = 0; Part < UF; ++Part) {
      Value *VecLoopExitInst = State.get(LoopExitInstDef, Part);
      Value *Sel = nullptr;
      for (User *U : VecLoopExitInst->users()) {
        if (isa<SelectInst>(U)) {
          assert(!Sel && "Reduction exit feeding two selects");
          Sel = U;
        } else
          assert(isa<PHINode>(U) && "Reduction exit must feed Phi's or select");
      }
      assert(Sel && "Reduction exit feeds no select");
      State.reset(LoopExitInstDef, Sel, Part);

      // If the target can create a predicated operator for the reduction at no
      // extra cost in the loop (for example a predicated vadd), it can be
      // cheaper for the select to remain in the loop than be sunk out of it,
      // and so use the select value for the phi instead of the old
      // LoopExitValue.
      if (PreferPredicatedReductionSelect ||
          TTI->preferPredicatedReductionSelect(
              RdxDesc.getOpcode(), PhiTy,
              TargetTransformInfo::ReductionFlags())) {
        auto *VecRdxPhi =
            cast<PHINode>(State.get(PhiR->getVPSingleValue(), Part));
        VecRdxPhi->setIncomingValueForBlock(
            LI->getLoopFor(LoopVectorBody)->getLoopLatch(), Sel);
      }
    }
  }

  // If the vector reduction can be performed in a smaller type, we truncate
  // then extend the loop exit value to enable InstCombine to evaluate the
  // entire expression in the smaller type.
  if (VF.isVector() && PhiTy != RdxDesc.getRecurrenceType()) {
    assert(!IsInLoopReductionPhi && "Unexpected truncated inloop reduction!");
    Type *RdxVecTy = VectorType::get(RdxDesc.getRecurrenceType(), VF);
    Builder.SetInsertPoint(
        LI->getLoopFor(LoopVectorBody)->getLoopLatch()->getTerminator());
    VectorParts RdxParts(UF);
    for (unsigned Part = 0; Part < UF; ++Part) {
      RdxParts[Part] = State.get(LoopExitInstDef, Part);
      Value *Trunc = Builder.CreateTrunc(RdxParts[Part], RdxVecTy);
      Value *Extnd = RdxDesc.isSigned() ? Builder.CreateSExt(Trunc, VecTy)
                                        : Builder.CreateZExt(Trunc, VecTy);
      for (Value::user_iterator UI = RdxParts[Part]->user_begin();
           UI != RdxParts[Part]->user_end();)
        if (*UI != Trunc) {
          (*UI++)->replaceUsesOfWith(RdxParts[Part], Extnd);
          RdxParts[Part] = Extnd;
        } else {
          ++UI;
        }
    }
    Builder.SetInsertPoint(&*LoopMiddleBlock->getFirstInsertionPt());
    for (unsigned Part = 0; Part < UF; ++Part) {
      RdxParts[Part] = Builder.CreateTrunc(RdxParts[Part], RdxVecTy);
      State.reset(LoopExitInstDef, RdxParts[Part], Part);
    }
  }

  // Reduce all of the unrolled parts into a single vector.
  Value *ReducedPartRdx = State.get(LoopExitInstDef, 0);
  unsigned Op = RecurrenceDescriptor::getOpcode(RK);

  // The middle block terminator has already been assigned a DebugLoc here (the
  // OrigLoop's single latch terminator). We want the whole middle block to
  // appear to execute on this line because: (a) it is all compiler generated,
  // (b) these instructions are always executed after evaluating the latch
  // conditional branch, and (c) other passes may add new predecessors which
  // terminate on this line. This is the easiest way to ensure we don't
  // accidentally cause an extra step back into the loop while debugging.
  setDebugLocFromInst(LoopMiddleBlock->getTerminator());
  if (IsOrdered)
    ReducedPartRdx = State.get(LoopExitInstDef, UF - 1);
  else {
    // Floating-point operations should have some FMF to enable the reduction.
    IRBuilderBase::FastMathFlagGuard FMFG(Builder);
    Builder.setFastMathFlags(RdxDesc.getFastMathFlags());
    for (unsigned Part = 1; Part < UF; ++Part) {
      Value *RdxPart = State.get(LoopExitInstDef, Part);
      if (Op != Instruction::ICmp && Op != Instruction::FCmp) {
        ReducedPartRdx = Builder.CreateBinOp(
            (Instruction::BinaryOps)Op, RdxPart, ReducedPartRdx, "bin.rdx");
      } else {
        ReducedPartRdx = createMinMaxOp(Builder, RK, ReducedPartRdx, RdxPart);
      }
    }
  }

  // Create the reduction after the loop. Note that inloop reductions create the
  // target reduction in the loop using a Reduction recipe.
  if (VF.isVector() && !IsInLoopReductionPhi) {
    ReducedPartRdx =
        createTargetReduction(Builder, TTI, RdxDesc, ReducedPartRdx);
    // If the reduction can be performed in a smaller type, we need to extend
    // the reduction to the wider type before we branch to the original loop.
    if (PhiTy != RdxDesc.getRecurrenceType())
      ReducedPartRdx = RdxDesc.isSigned()
                           ? Builder.CreateSExt(ReducedPartRdx, PhiTy)
                           : Builder.CreateZExt(ReducedPartRdx, PhiTy);
  }

  // Create a phi node that merges control-flow from the backedge-taken check
  // block and the middle block.
  PHINode *BCBlockPhi = PHINode::Create(PhiTy, 2, "bc.merge.rdx",
                                        LoopScalarPreHeader->getTerminator());
  for (unsigned I = 0, E = LoopBypassBlocks.size(); I != E; ++I)
    BCBlockPhi->addIncoming(ReductionStartValue, LoopBypassBlocks[I]);
  BCBlockPhi->addIncoming(ReducedPartRdx, LoopMiddleBlock);

  // Now, we need to fix the users of the reduction variable
  // inside and outside of the scalar remainder loop.

  // We know that the loop is in LCSSA form. We need to update the PHI nodes
  // in the exit blocks.  See comment on analogous loop in
  // fixFirstOrderRecurrence for a more complete explaination of the logic.
  for (PHINode &LCSSAPhi : LoopExitBlock->phis())
    if (any_of(LCSSAPhi.incoming_values(),
               [LoopExitInst](Value *V) { return V == LoopExitInst; }))
      LCSSAPhi.addIncoming(ReducedPartRdx, LoopMiddleBlock);

  // Fix the scalar loop reduction variable with the incoming reduction sum
  // from the vector body and from the backedge value.
  int IncomingEdgeBlockIdx =
      OrigPhi->getBasicBlockIndex(OrigLoop->getLoopLatch());
  assert(IncomingEdgeBlockIdx >= 0 && "Invalid block index");
  // Pick the other block.
  int SelfEdgeBlockIdx = (IncomingEdgeBlockIdx ? 0 : 1);
  OrigPhi->setIncomingValue(SelfEdgeBlockIdx, BCBlockPhi);
  OrigPhi->setIncomingValue(IncomingEdgeBlockIdx, LoopExitInst);
}

void InnerLoopVectorizer::clearReductionWrapFlags(const RecurrenceDescriptor &RdxDesc,
                                                  VPTransformState &State) {
  RecurKind RK = RdxDesc.getRecurrenceKind();
  if (RK != RecurKind::Add && RK != RecurKind::Mul)
    return;

  Instruction *LoopExitInstr = RdxDesc.getLoopExitInstr();
  assert(LoopExitInstr && "null loop exit instruction");
  SmallVector<Instruction *, 8> Worklist;
  SmallPtrSet<Instruction *, 8> Visited;
  Worklist.push_back(LoopExitInstr);
  Visited.insert(LoopExitInstr);

  while (!Worklist.empty()) {
    Instruction *Cur = Worklist.pop_back_val();
    if (isa<OverflowingBinaryOperator>(Cur))
      for (unsigned Part = 0; Part < UF; ++Part) {
        Value *V = State.get(State.Plan->getVPValue(Cur), Part);
        cast<Instruction>(V)->dropPoisonGeneratingFlags();
      }

    for (User *U : Cur->users()) {
      Instruction *UI = cast<Instruction>(U);
      if ((Cur != LoopExitInstr || OrigLoop->contains(UI->getParent())) &&
          Visited.insert(UI).second)
        Worklist.push_back(UI);
    }
  }
}

void InnerLoopVectorizer::fixLCSSAPHIs(VPTransformState &State) {
  for (PHINode &LCSSAPhi : LoopExitBlock->phis()) {
    if (LCSSAPhi.getBasicBlockIndex(LoopMiddleBlock) != -1)
      // Some phis were already hand updated by the reduction and recurrence
      // code above, leave them alone.
      continue;

    auto *IncomingValue = LCSSAPhi.getIncomingValue(0);
    // Non-instruction incoming values will have only one value.

    VPLane Lane = VPLane::getFirstLane();
    if (isa<Instruction>(IncomingValue) &&
        !Cost->isUniformAfterVectorization(cast<Instruction>(IncomingValue),
                                           VF))
      Lane = VPLane::getLastLaneForVF(VF);

    // Can be a loop invariant incoming value or the last scalar value to be
    // extracted from the vectorized loop.
    Builder.SetInsertPoint(LoopMiddleBlock->getTerminator());
    Value *lastIncomingValue =
        OrigLoop->isLoopInvariant(IncomingValue)
            ? IncomingValue
            : State.get(State.Plan->getVPValue(IncomingValue),
                        VPIteration(UF - 1, Lane));
    LCSSAPhi.addIncoming(lastIncomingValue, LoopMiddleBlock);
  }
}

void InnerLoopVectorizer::sinkScalarOperands(Instruction *PredInst) {
  // The basic block and loop containing the predicated instruction.
  auto *PredBB = PredInst->getParent();
  auto *VectorLoop = LI->getLoopFor(PredBB);

  // Initialize a worklist with the operands of the predicated instruction.
  SetVector<Value *> Worklist(PredInst->op_begin(), PredInst->op_end());

  // Holds instructions that we need to analyze again. An instruction may be
  // reanalyzed if we don't yet know if we can sink it or not.
  SmallVector<Instruction *, 8> InstsToReanalyze;

  // Returns true if a given use occurs in the predicated block. Phi nodes use
  // their operands in their corresponding predecessor blocks.
  auto isBlockOfUsePredicated = [&](Use &U) -> bool {
    auto *I = cast<Instruction>(U.getUser());
    BasicBlock *BB = I->getParent();
    if (auto *Phi = dyn_cast<PHINode>(I))
      BB = Phi->getIncomingBlock(
          PHINode::getIncomingValueNumForOperand(U.getOperandNo()));
    return BB == PredBB;
  };

  // Iteratively sink the scalarized operands of the predicated instruction
  // into the block we created for it. When an instruction is sunk, it's
  // operands are then added to the worklist. The algorithm ends after one pass
  // through the worklist doesn't sink a single instruction.
  bool Changed;
  do {
    // Add the instructions that need to be reanalyzed to the worklist, and
    // reset the changed indicator.
    Worklist.insert(InstsToReanalyze.begin(), InstsToReanalyze.end());
    InstsToReanalyze.clear();
    Changed = false;

    while (!Worklist.empty()) {
      auto *I = dyn_cast<Instruction>(Worklist.pop_back_val());

      // We can't sink an instruction if it is a phi node, is not in the loop,
      // or may have side effects.
      if (!I || isa<PHINode>(I) || !VectorLoop->contains(I) ||
          I->mayHaveSideEffects())
        continue;

      // If the instruction is already in PredBB, check if we can sink its
      // operands. In that case, VPlan's sinkScalarOperands() succeeded in
      // sinking the scalar instruction I, hence it appears in PredBB; but it
      // may have failed to sink I's operands (recursively), which we try
      // (again) here.
      if (I->getParent() == PredBB) {
        Worklist.insert(I->op_begin(), I->op_end());
        continue;
      }

      // It's legal to sink the instruction if all its uses occur in the
      // predicated block. Otherwise, there's nothing to do yet, and we may
      // need to reanalyze the instruction.
      if (!llvm::all_of(I->uses(), isBlockOfUsePredicated)) {
        InstsToReanalyze.push_back(I);
        continue;
      }

      // Move the instruction to the beginning of the predicated block, and add
      // it's operands to the worklist.
      I->moveBefore(&*PredBB->getFirstInsertionPt());
      Worklist.insert(I->op_begin(), I->op_end());

      // The sinking may have enabled other instructions to be sunk, so we will
      // need to iterate.
      Changed = true;
    }
  } while (Changed);
}

void InnerLoopVectorizer::fixNonInductionPHIs(VPTransformState &State) {
  for (PHINode *OrigPhi : OrigPHIsToFix) {
    VPWidenPHIRecipe *VPPhi =
        cast<VPWidenPHIRecipe>(State.Plan->getVPValue(OrigPhi));
    PHINode *NewPhi = cast<PHINode>(State.get(VPPhi, 0));
    // Make sure the builder has a valid insert point.
    Builder.SetInsertPoint(NewPhi);
    for (unsigned i = 0; i < VPPhi->getNumOperands(); ++i) {
      VPValue *Inc = VPPhi->getIncomingValue(i);
      VPBasicBlock *VPBB = VPPhi->getIncomingBlock(i);
      NewPhi->addIncoming(State.get(Inc, 0), State.CFG.VPBB2IRBB[VPBB]);
    }
  }
}

bool InnerLoopVectorizer::useOrderedReductions(RecurrenceDescriptor &RdxDesc) {
  return Cost->useOrderedReductions(RdxDesc);
}

void InnerLoopVectorizer::widenGEP(GetElementPtrInst *GEP, VPValue *VPDef,
                                   VPUser &Operands, unsigned UF,
                                   ElementCount VF, bool IsPtrLoopInvariant,
                                   SmallBitVector &IsIndexLoopInvariant,
                                   VPTransformState &State) {
  // Construct a vector GEP by widening the operands of the scalar GEP as
  // necessary. We mark the vector GEP 'inbounds' if appropriate. A GEP
  // results in a vector of pointers when at least one operand of the GEP
  // is vector-typed. Thus, to keep the representation compact, we only use
  // vector-typed operands for loop-varying values.

  if (VF.isVector() && IsPtrLoopInvariant && IsIndexLoopInvariant.all()) {
    // If we are vectorizing, but the GEP has only loop-invariant operands,
    // the GEP we build (by only using vector-typed operands for
    // loop-varying values) would be a scalar pointer. Thus, to ensure we
    // produce a vector of pointers, we need to either arbitrarily pick an
    // operand to broadcast, or broadcast a clone of the original GEP.
    // Here, we broadcast a clone of the original.
    //
    // TODO: If at some point we decide to scalarize instructions having
    //       loop-invariant operands, this special case will no longer be
    //       required. We would add the scalarization decision to
    //       collectLoopScalars() and teach getVectorValue() to broadcast
    //       the lane-zero scalar value.
    auto *Clone = Builder.Insert(GEP->clone());
    for (unsigned Part = 0; Part < UF; ++Part) {
      Value *EntryPart = Builder.CreateVectorSplat(VF, Clone);
      State.set(VPDef, EntryPart, Part);
      addMetadata(EntryPart, GEP);
    }
  } else {
    // If the GEP has at least one loop-varying operand, we are sure to
    // produce a vector of pointers. But if we are only unrolling, we want
    // to produce a scalar GEP for each unroll part. Thus, the GEP we
    // produce with the code below will be scalar (if VF == 1) or vector
    // (otherwise). Note that for the unroll-only case, we still maintain
    // values in the vector mapping with initVector, as we do for other
    // instructions.
    for (unsigned Part = 0; Part < UF; ++Part) {
      // The pointer operand of the new GEP. If it's loop-invariant, we
      // won't broadcast it.
      auto *Ptr = IsPtrLoopInvariant
                      ? State.get(Operands.getOperand(0), VPIteration(0, 0))
                      : State.get(Operands.getOperand(0), Part);

      // Collect all the indices for the new GEP. If any index is
      // loop-invariant, we won't broadcast it.
      SmallVector<Value *, 4> Indices;
      for (unsigned I = 1, E = Operands.getNumOperands(); I < E; I++) {
        VPValue *Operand = Operands.getOperand(I);
        if (IsIndexLoopInvariant[I - 1])
          Indices.push_back(State.get(Operand, VPIteration(0, 0)));
        else
          Indices.push_back(State.get(Operand, Part));
      }

      // Create the new GEP. Note that this GEP may be a scalar if VF == 1,
      // but it should be a vector, otherwise.
      auto *NewGEP =
          GEP->isInBounds()
              ? Builder.CreateInBoundsGEP(GEP->getSourceElementType(), Ptr,
                                          Indices)
              : Builder.CreateGEP(GEP->getSourceElementType(), Ptr, Indices);
      assert((VF.isScalar() || NewGEP->getType()->isVectorTy()) &&
             "NewGEP is not a pointer vector");
      State.set(VPDef, NewGEP, Part);
      addMetadata(NewGEP, GEP);
    }
  }
}

void InnerLoopVectorizer::widenPHIInstruction(Instruction *PN,
                                              RecurrenceDescriptor *RdxDesc,
                                              VPWidenPHIRecipe *PhiR,
                                              VPTransformState &State) {
  PHINode *P = cast<PHINode>(PN);
  if (EnableVPlanNativePath) {
    // Currently we enter here in the VPlan-native path for non-induction
    // PHIs where all control flow is uniform. We simply widen these PHIs.
    // Create a vector phi with no operands - the vector phi operands will be
    // set at the end of vector code generation.
    Type *VecTy = (State.VF.isScalar())
                      ? PN->getType()
                      : VectorType::get(PN->getType(), State.VF);
    Value *VecPhi = Builder.CreatePHI(VecTy, PN->getNumOperands(), "vec.phi");
    State.set(PhiR, VecPhi, 0);
    OrigPHIsToFix.push_back(P);

    return;
  }

  assert(PN->getParent() == OrigLoop->getHeader() &&
         "Non-header phis should have been handled elsewhere");

  // In order to support recurrences we need to be able to vectorize Phi nodes.
  // Phi nodes have cycles, so we need to vectorize them in two stages. This is
  // stage #1: We create a new vector PHI node with no incoming edges. We'll use
  // this value when we vectorize all of the instructions that use the PHI.
  if (RdxDesc || Legal->isFirstOrderRecurrence(P)) {
    bool ScalarPHI =
        (State.VF.isScalar()) || Cost->isInLoopReduction(cast<PHINode>(PN));
    Type *VecTy =
        ScalarPHI ? PN->getType() : VectorType::get(PN->getType(), State.VF);

    bool IsOrdered = Cost->isInLoopReduction(cast<PHINode>(PN)) &&
                     Cost->useOrderedReductions(*RdxDesc);
    unsigned LastPartForNewPhi = IsOrdered ? 1 : State.UF;
    for (unsigned Part = 0; Part < LastPartForNewPhi; ++Part) {
      Value *EntryPart = PHINode::Create(
          VecTy, 2, "vec.phi", &*LoopVectorBody->getFirstInsertionPt());
      State.set(PhiR, EntryPart, Part);
    }
    if (Legal->isFirstOrderRecurrence(P))
      return;
    VPValue *StartVPV = PhiR->getStartValue();
    Value *StartV = StartVPV->getLiveInIRValue();

    Value *Iden = nullptr;

    assert(Legal->isReductionVariable(P) && StartV &&
           "RdxDesc should only be set for reduction variables; in that case "
           "a StartV is also required");
    RecurKind RK = RdxDesc->getRecurrenceKind();
    if (RecurrenceDescriptor::isMinMaxRecurrenceKind(RK)) {
      // MinMax reduction have the start value as their identify.
      if (ScalarPHI) {
        Iden = StartV;
      } else {
        IRBuilderBase::InsertPointGuard IPBuilder(Builder);
        Builder.SetInsertPoint(LoopVectorPreHeader->getTerminator());
        StartV = Iden =
            Builder.CreateVectorSplat(State.VF, StartV, "minmax.ident");
      }
    } else {
      Constant *IdenC = RecurrenceDescriptor::getRecurrenceIdentity(
          RK, VecTy->getScalarType(), RdxDesc->getFastMathFlags());
      Iden = IdenC;

      if (!ScalarPHI) {
        Iden = ConstantVector::getSplat(State.VF, IdenC);
        IRBuilderBase::InsertPointGuard IPBuilder(Builder);
        Builder.SetInsertPoint(LoopVectorPreHeader->getTerminator());
        Constant *Zero = Builder.getInt32(0);
        StartV = Builder.CreateInsertElement(Iden, StartV, Zero);
      }
    }

    for (unsigned Part = 0; Part < LastPartForNewPhi; ++Part) {
      Value *EntryPart = State.get(PhiR, Part);
      // Make sure to add the reduction start value only to the
      // first unroll part.
      Value *StartVal = (Part == 0) ? StartV : Iden;
      cast<PHINode>(EntryPart)->addIncoming(StartVal, LoopVectorPreHeader);
    }

    return;
  }

  assert(!Legal->isReductionVariable(P) &&
         "reductions should be handled above");

  setDebugLocFromInst(P);

  // This PHINode must be an induction variable.
  // Make sure that we know about it.
  assert(Legal->getInductionVars().count(P) && "Not an induction variable");

  InductionDescriptor II = Legal->getInductionVars().lookup(P);
  const DataLayout &DL = OrigLoop->getHeader()->getModule()->getDataLayout();

  // FIXME: The newly created binary instructions should contain nsw/nuw flags,
  // which can be found from the original scalar operations.
  switch (II.getKind()) {
  case InductionDescriptor::IK_NoInduction:
    llvm_unreachable("Unknown induction");
  case InductionDescriptor::IK_IntInduction:
  case InductionDescriptor::IK_FpInduction:
    llvm_unreachable("Integer/fp induction is handled elsewhere.");
  case InductionDescriptor::IK_PtrInduction: {
    // Handle the pointer induction variable case.
    assert(P->getType()->isPointerTy() && "Unexpected type.");

    if (Cost->isScalarAfterVectorization(P, State.VF)) {
      // This is the normalized GEP that starts counting at zero.
      Value *PtrInd =
          Builder.CreateSExtOrTrunc(Induction, II.getStep()->getType());
      // Determine the number of scalars we need to generate for each unroll
      // iteration. If the instruction is uniform, we only need to generate the
      // first lane. Otherwise, we generate all VF values.
      bool IsUniform = Cost->isUniformAfterVectorization(P, State.VF);
      unsigned Lanes = IsUniform ? 1 : State.VF.getKnownMinValue();

      bool NeedsVectorIndex = !IsUniform && VF.isScalable();
      Value *UnitStepVec = nullptr, *PtrIndSplat = nullptr;
      if (NeedsVectorIndex) {
        Type *VecIVTy = VectorType::get(PtrInd->getType(), VF);
        UnitStepVec = Builder.CreateStepVector(VecIVTy);
        PtrIndSplat = Builder.CreateVectorSplat(VF, PtrInd);
      }

      for (unsigned Part = 0; Part < UF; ++Part) {
        Value *PartStart = createStepForVF(
            Builder, ConstantInt::get(PtrInd->getType(), Part), VF);

        if (NeedsVectorIndex) {
          Value *PartStartSplat = Builder.CreateVectorSplat(VF, PartStart);
          Value *Indices = Builder.CreateAdd(PartStartSplat, UnitStepVec);
          Value *GlobalIndices = Builder.CreateAdd(PtrIndSplat, Indices);
          Value *SclrGep =
              emitTransformedIndex(Builder, GlobalIndices, PSE.getSE(), DL, II);
          SclrGep->setName("next.gep");
          State.set(PhiR, SclrGep, Part);
          // We've cached the whole vector, which means we can support the
          // extraction of any lane.
          continue;
        }

        for (unsigned Lane = 0; Lane < Lanes; ++Lane) {
          Value *Idx = Builder.CreateAdd(
              PartStart, ConstantInt::get(PtrInd->getType(), Lane));
          Value *GlobalIdx = Builder.CreateAdd(PtrInd, Idx);
          Value *SclrGep =
              emitTransformedIndex(Builder, GlobalIdx, PSE.getSE(), DL, II);
          SclrGep->setName("next.gep");
          State.set(PhiR, SclrGep, VPIteration(Part, Lane));
        }
      }
      return;
    }
    assert(isa<SCEVConstant>(II.getStep()) &&
           "Induction step not a SCEV constant!");
    Type *PhiType = II.getStep()->getType();

    // Build a pointer phi
    Value *ScalarStartValue = II.getStartValue();
    Type *ScStValueType = ScalarStartValue->getType();
    PHINode *NewPointerPhi =
        PHINode::Create(ScStValueType, 2, "pointer.phi", Induction);
    NewPointerPhi->addIncoming(ScalarStartValue, LoopVectorPreHeader);

    // A pointer induction, performed by using a gep
    BasicBlock *LoopLatch = LI->getLoopFor(LoopVectorBody)->getLoopLatch();
    Instruction *InductionLoc = LoopLatch->getTerminator();
    const SCEV *ScalarStep = II.getStep();
    SCEVExpander Exp(*PSE.getSE(), DL, "induction");
    Value *ScalarStepValue =
        Exp.expandCodeFor(ScalarStep, PhiType, InductionLoc);
    Value *RuntimeVF = getRuntimeVF(Builder, PhiType, VF);
    Value *NumUnrolledElems =
        Builder.CreateMul(RuntimeVF, ConstantInt::get(PhiType, State.UF));
    Value *InductionGEP = GetElementPtrInst::Create(
        ScStValueType->getPointerElementType(), NewPointerPhi,
        Builder.CreateMul(ScalarStepValue, NumUnrolledElems), "ptr.ind",
        InductionLoc);
    NewPointerPhi->addIncoming(InductionGEP, LoopLatch);

    // Create UF many actual address geps that use the pointer
    // phi as base and a vectorized version of the step value
    // (<step*0, ..., step*N>) as offset.
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      Type *VecPhiType = VectorType::get(PhiType, State.VF);
      Value *StartOffsetScalar =
          Builder.CreateMul(RuntimeVF, ConstantInt::get(PhiType, Part));
      Value *StartOffset =
          Builder.CreateVectorSplat(State.VF, StartOffsetScalar);
      // Create a vector of consecutive numbers from zero to VF.
      StartOffset =
          Builder.CreateAdd(StartOffset, Builder.CreateStepVector(VecPhiType));

      Value *GEP = Builder.CreateGEP(
          ScStValueType->getPointerElementType(), NewPointerPhi,
          Builder.CreateMul(
              StartOffset, Builder.CreateVectorSplat(State.VF, ScalarStepValue),
              "vector.gep"));
      State.set(PhiR, GEP, Part);
    }
  }
  }
}

/// A helper function for checking whether an integer division-related
/// instruction may divide by zero (in which case it must be predicated if
/// executed conditionally in the scalar code).
/// TODO: It may be worthwhile to generalize and check isKnownNonZero().
/// Non-zero divisors that are non compile-time constants will not be
/// converted into multiplication, so we will still end up scalarizing
/// the division, but can do so w/o predication.
static bool mayDivideByZero(Instruction &I) {
  assert((I.getOpcode() == Instruction::UDiv ||
          I.getOpcode() == Instruction::SDiv ||
          I.getOpcode() == Instruction::URem ||
          I.getOpcode() == Instruction::SRem) &&
         "Unexpected instruction");
  Value *Divisor = I.getOperand(1);
  auto *CInt = dyn_cast<ConstantInt>(Divisor);
  return !CInt || CInt->isZero();
}

void InnerLoopVectorizer::widenInstruction(Instruction &I, VPValue *Def,
                                           VPUser &User,
                                           VPTransformState &State) {
  switch (I.getOpcode()) {
  case Instruction::Call:
  case Instruction::Br:
  case Instruction::PHI:
  case Instruction::GetElementPtr:
  case Instruction::Select:
    llvm_unreachable("This instruction is handled by a different recipe.");
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::URem:
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::FNeg:
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
    // Just widen unops and binops.
    setDebugLocFromInst(&I);

    for (unsigned Part = 0; Part < UF; ++Part) {
      SmallVector<Value *, 2> Ops;
      for (VPValue *VPOp : User.operands())
        Ops.push_back(State.get(VPOp, Part));

      Value *V = Builder.CreateNAryOp(I.getOpcode(), Ops);

      if (auto *VecOp = dyn_cast<Instruction>(V))
        VecOp->copyIRFlags(&I);

      // Use this vector value for all users of the original instruction.
      State.set(Def, V, Part);
      addMetadata(V, &I);
    }

    break;
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    // Widen compares. Generate vector compares.
    bool FCmp = (I.getOpcode() == Instruction::FCmp);
    auto *Cmp = cast<CmpInst>(&I);
    setDebugLocFromInst(Cmp);
    for (unsigned Part = 0; Part < UF; ++Part) {
      Value *A = State.get(User.getOperand(0), Part);
      Value *B = State.get(User.getOperand(1), Part);
      Value *C = nullptr;
      if (FCmp) {
        // Propagate fast math flags.
        IRBuilder<>::FastMathFlagGuard FMFG(Builder);
        Builder.setFastMathFlags(Cmp->getFastMathFlags());
        C = Builder.CreateFCmp(Cmp->getPredicate(), A, B);
      } else {
        C = Builder.CreateICmp(Cmp->getPredicate(), A, B);
      }
      State.set(Def, C, Part);
      addMetadata(C, &I);
    }

    break;
  }

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
  case Instruction::FPTrunc:
  case Instruction::BitCast: {
    auto *CI = cast<CastInst>(&I);
    setDebugLocFromInst(CI);

    /// Vectorize casts.
    Type *DestTy =
        (VF.isScalar()) ? CI->getType() : VectorType::get(CI->getType(), VF);

    for (unsigned Part = 0; Part < UF; ++Part) {
      Value *A = State.get(User.getOperand(0), Part);
      Value *Cast = Builder.CreateCast(CI->getOpcode(), A, DestTy);
      State.set(Def, Cast, Part);
      addMetadata(Cast, &I);
    }
    break;
  }
  default:
    // This instruction is not vectorized by simple widening.
    LLVM_DEBUG(dbgs() << "LV: Found an unhandled instruction: " << I);
    llvm_unreachable("Unhandled instruction!");
  } // end of switch.
}

void InnerLoopVectorizer::widenCallInstruction(CallInst &I, VPValue *Def,
                                               VPUser &ArgOperands,
                                               VPTransformState &State) {
  assert(!isa<DbgInfoIntrinsic>(I) &&
         "DbgInfoIntrinsic should have been dropped during VPlan construction");
  setDebugLocFromInst(&I);

  Module *M = I.getParent()->getParent()->getParent();
  auto *CI = cast<CallInst>(&I);

  SmallVector<Type *, 4> Tys;
  for (Value *ArgOperand : CI->arg_operands())
    Tys.push_back(ToVectorTy(ArgOperand->getType(), VF.getKnownMinValue()));

  Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);

  // The flag shows whether we use Intrinsic or a usual Call for vectorized
  // version of the instruction.
  // Is it beneficial to perform intrinsic call compared to lib call?
  bool NeedToScalarize = false;
  InstructionCost CallCost = Cost->getVectorCallCost(CI, VF, NeedToScalarize);
  InstructionCost IntrinsicCost = ID ? Cost->getVectorIntrinsicCost(CI, VF) : 0;
  bool UseVectorIntrinsic = ID && IntrinsicCost <= CallCost;
  assert((UseVectorIntrinsic || !NeedToScalarize) &&
         "Instruction should be scalarized elsewhere.");
  assert((IntrinsicCost.isValid() || CallCost.isValid()) &&
         "Either the intrinsic cost or vector call cost must be valid");

  for (unsigned Part = 0; Part < UF; ++Part) {
    SmallVector<Type *, 2> TysForDecl = {CI->getType()};
    SmallVector<Value *, 4> Args;
    for (auto &I : enumerate(ArgOperands.operands())) {
      // Some intrinsics have a scalar argument - don't replace it with a
      // vector.
      Value *Arg;
      if (!UseVectorIntrinsic || !hasVectorInstrinsicScalarOpd(ID, I.index()))
        Arg = State.get(I.value(), Part);
      else {
        Arg = State.get(I.value(), VPIteration(0, 0));
        if (hasVectorInstrinsicOverloadedScalarOpd(ID, I.index()))
          TysForDecl.push_back(Arg->getType());
      }
      Args.push_back(Arg);
    }

    Function *VectorF;
    if (UseVectorIntrinsic) {
      // Use vector version of the intrinsic.
      if (VF.isVector())
        TysForDecl[0] = VectorType::get(CI->getType()->getScalarType(), VF);
      VectorF = Intrinsic::getDeclaration(M, ID, TysForDecl);
      assert(VectorF && "Can't retrieve vector intrinsic.");
    } else {
      // Use vector version of the function call.
      const VFShape Shape = VFShape::get(*CI, VF, false /*HasGlobalPred*/);
#ifndef NDEBUG
      assert(VFDatabase(*CI).getVectorizedFunction(Shape) != nullptr &&
             "Can't create vector function.");
#endif
        VectorF = VFDatabase(*CI).getVectorizedFunction(Shape);
    }
      SmallVector<OperandBundleDef, 1> OpBundles;
      CI->getOperandBundlesAsDefs(OpBundles);
      CallInst *V = Builder.CreateCall(VectorF, Args, OpBundles);

      if (isa<FPMathOperator>(V))
        V->copyFastMathFlags(CI);

      State.set(Def, V, Part);
      addMetadata(V, &I);
  }
}

void InnerLoopVectorizer::widenSelectInstruction(SelectInst &I, VPValue *VPDef,
                                                 VPUser &Operands,
                                                 bool InvariantCond,
                                                 VPTransformState &State) {
  setDebugLocFromInst(&I);

  // The condition can be loop invariant  but still defined inside the
  // loop. This means that we can't just use the original 'cond' value.
  // We have to take the 'vectorized' value and pick the first lane.
  // Instcombine will make this a no-op.
  auto *InvarCond = InvariantCond
                        ? State.get(Operands.getOperand(0), VPIteration(0, 0))
                        : nullptr;

  for (unsigned Part = 0; Part < UF; ++Part) {
    Value *Cond =
        InvarCond ? InvarCond : State.get(Operands.getOperand(0), Part);
    Value *Op0 = State.get(Operands.getOperand(1), Part);
    Value *Op1 = State.get(Operands.getOperand(2), Part);
    Value *Sel = Builder.CreateSelect(Cond, Op0, Op1);
    State.set(VPDef, Sel, Part);
    addMetadata(Sel, &I);
  }
}

void LoopVectorizationCostModel::collectLoopScalars(ElementCount VF) {
  // We should not collect Scalars more than once per VF. Right now, this
  // function is called from collectUniformsAndScalars(), which already does
  // this check. Collecting Scalars for VF=1 does not make any sense.
  assert(VF.isVector() && Scalars.find(VF) == Scalars.end() &&
         "This function should not be visited twice for the same VF");

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
  auto isScalarUse = [&](Instruction *MemAccess, Value *Ptr) {
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

  // A helper that returns true if the given value is a bitcast or
  // getelementptr instruction contained in the loop.
  auto isLoopVaryingBitCastOrGEP = [&](Value *V) {
    return ((isa<BitCastInst>(V) && V->getType()->isPointerTy()) ||
            isa<GetElementPtrInst>(V)) &&
           !TheLoop->isLoopInvariant(V);
  };

  auto isScalarPtrInduction = [&](Instruction *MemAccess, Value *Ptr) {
    if (!isa<PHINode>(Ptr) ||
        !Legal->getInductionVars().count(cast<PHINode>(Ptr)))
      return false;
    auto &Induction = Legal->getInductionVars()[cast<PHINode>(Ptr)];
    if (Induction.getKind() != InductionDescriptor::IK_PtrInduction)
      return false;
    return isScalarUse(MemAccess, Ptr);
  };

  // A helper that evaluates a memory access's use of a pointer. If the
  // pointer is actually the pointer induction of a loop, it is being
  // inserted into Worklist. If the use will be a scalar use, and the
  // pointer is only used by memory accesses, we place the pointer in
  // ScalarPtrs. Otherwise, the pointer is placed in PossibleNonScalarPtrs.
  auto evaluatePtrUse = [&](Instruction *MemAccess, Value *Ptr) {
    if (isScalarPtrInduction(MemAccess, Ptr)) {
      Worklist.insert(cast<Instruction>(Ptr));
      Instruction *Update = cast<Instruction>(
          cast<PHINode>(Ptr)->getIncomingValueForBlock(Latch));
      Worklist.insert(Update);
      LLVM_DEBUG(dbgs() << "LV: Found new scalar instruction: " << *Ptr
                        << "\n");
      LLVM_DEBUG(dbgs() << "LV: Found new scalar instruction: " << *Update
                        << "\n");
      return;
    }
    // We only care about bitcast and getelementptr instructions contained in
    // the loop.
    if (!isLoopVaryingBitCastOrGEP(Ptr))
      return;

    // If the pointer has already been identified as scalar (e.g., if it was
    // also identified as uniform), there's nothing to do.
    auto *I = cast<Instruction>(Ptr);
    if (Worklist.count(I))
      return;

    // If the use of the pointer will be a scalar use, and all users of the
    // pointer are memory accesses, place the pointer in ScalarPtrs. Otherwise,
    // place the pointer in PossibleNonScalarPtrs.
    if (isScalarUse(MemAccess, Ptr) && llvm::all_of(I->users(), [&](User *U) {
          return isa<LoadInst>(U) || isa<StoreInst>(U);
        }))
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
  Worklist.insert(Uniforms[VF].begin(), Uniforms[VF].end());

  // (2) Add to the worklist all bitcast and getelementptr instructions used by
  // memory accesses requiring a scalar use. The pointer operands of loads and
  // stores will be scalar as long as the memory accesses is not a gather or
  // scatter operation. The value operand of a store will remain scalar if the
  // store is scalarized.
  for (auto *BB : TheLoop->blocks())
    for (auto &I : *BB) {
      if (auto *Load = dyn_cast<LoadInst>(&I)) {
        evaluatePtrUse(Load, Load->getPointerOperand());
      } else if (auto *Store = dyn_cast<StoreInst>(&I)) {
        evaluatePtrUse(Store, Store->getPointerOperand());
        evaluatePtrUse(Store, Store->getValueOperand());
      }
    }
  for (auto *I : ScalarPtrs)
    if (!PossibleNonScalarPtrs.count(I)) {
      LLVM_DEBUG(dbgs() << "LV: Found scalar instruction: " << *I << "\n");
      Worklist.insert(I);
    }

  // Insert the forced scalars.
  // FIXME: Currently widenPHIInstruction() often creates a dead vector
  // induction variable when the PHI user is scalarized.
  auto ForcedScalar = ForcedScalars.find(VF);
  if (ForcedScalar != ForcedScalars.end())
    for (auto *I : ForcedScalar->second)
      Worklist.insert(I);

  // Expand the worklist by looking through any bitcasts and getelementptr
  // instructions we've already identified as scalar. This is similar to the
  // expansion step in collectLoopUniforms(); however, here we're only
  // expanding to include additional bitcasts and getelementptr instructions.
  unsigned Idx = 0;
  while (Idx != Worklist.size()) {
    Instruction *Dst = Worklist[Idx++];
    if (!isLoopVaryingBitCastOrGEP(Dst->getOperand(0)))
      continue;
    auto *Src = cast<Instruction>(Dst->getOperand(0));
    if (llvm::all_of(Src->users(), [&](User *U) -> bool {
          auto *J = cast<Instruction>(U);
          return !TheLoop->contains(J) || Worklist.count(J) ||
                 ((isa<LoadInst>(J) || isa<StoreInst>(J)) &&
                  isScalarUse(J, Src));
        })) {
      Worklist.insert(Src);
      LLVM_DEBUG(dbgs() << "LV: Found scalar instruction: " << *Src << "\n");
    }
  }

  // An induction variable will remain scalar if all users of the induction
  // variable and induction variable update remain scalar.
  for (auto &Induction : Legal->getInductionVars()) {
    auto *Ind = Induction.first;
    auto *IndUpdate = cast<Instruction>(Ind->getIncomingValueForBlock(Latch));

    // If tail-folding is applied, the primary induction variable will be used
    // to feed a vector compare.
    if (Ind == Legal->getPrimaryInduction() && foldTailByMasking())
      continue;

    // Determine if all users of the induction variable are scalar after
    // vectorization.
    auto ScalarInd = llvm::all_of(Ind->users(), [&](User *U) -> bool {
      auto *I = cast<Instruction>(U);
      return I == IndUpdate || !TheLoop->contains(I) || Worklist.count(I);
    });
    if (!ScalarInd)
      continue;

    // Determine if all users of the induction variable update instruction are
    // scalar after vectorization.
    auto ScalarIndUpdate =
        llvm::all_of(IndUpdate->users(), [&](User *U) -> bool {
          auto *I = cast<Instruction>(U);
          return I == Ind || !TheLoop->contains(I) || Worklist.count(I);
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

  Scalars[VF].insert(Worklist.begin(), Worklist.end());
}

bool LoopVectorizationCostModel::isScalarWithPredication(Instruction *I) const {
  if (!blockNeedsPredication(I->getParent()))
    return false;
  switch(I->getOpcode()) {
  default:
    break;
  case Instruction::Load:
  case Instruction::Store: {
    if (!Legal->isMaskRequired(I))
      return false;
    auto *Ptr = getLoadStorePointerOperand(I);
    auto *Ty = getLoadStoreType(I);
    const Align Alignment = getLoadStoreAlignment(I);
    return isa<LoadInst>(I) ? !(isLegalMaskedLoad(Ty, Ptr, Alignment) ||
                                TTI.isLegalMaskedGather(Ty, Alignment))
                            : !(isLegalMaskedStore(Ty, Ptr, Alignment) ||
                                TTI.isLegalMaskedScatter(Ty, Alignment));
  }
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::URem:
    return mayDivideByZero(*I);
  }
  return false;
}

bool LoopVectorizationCostModel::interleavedAccessCanBeWidened(
    Instruction *I, ElementCount VF) {
  assert(isAccessInterleaved(I) && "Expecting interleaved access.");
  assert(getWideningDecision(I, VF) == CM_Unknown &&
         "Decision should not be set yet.");
  auto *Group = getInterleavedAccessGroup(I);
  assert(Group && "Must have a group.");

  // If the instruction's allocated size doesn't equal it's type size, it
  // requires padding and will be scalarized.
  auto &DL = I->getModule()->getDataLayout();
  auto *ScalarTy = getLoadStoreType(I);
  if (hasIrregularType(ScalarTy, DL))
    return false;

  // Check if masking is required.
  // A Group may need masking for one of two reasons: it resides in a block that
  // needs predication, or it was decided to use masking to deal with gaps.
  bool PredicatedAccessRequiresMasking =
      Legal->blockNeedsPredication(I->getParent()) && Legal->isMaskRequired(I);
  bool AccessWithGapsRequiresMasking =
      Group->requiresScalarEpilogue() && !isScalarEpilogueAllowed();
  if (!PredicatedAccessRequiresMasking && !AccessWithGapsRequiresMasking)
    return true;

  // If masked interleaving is required, we expect that the user/target had
  // enabled it, because otherwise it either wouldn't have been created or
  // it should have been invalidated by the CostModel.
  assert(useMaskedInterleavedAccesses(TTI) &&
         "Masked interleave-groups for predicated accesses are not enabled.");

  auto *Ty = getLoadStoreType(I);
  const Align Alignment = getLoadStoreAlignment(I);
  return isa<LoadInst>(I) ? TTI.isLegalMaskedLoad(Ty, Alignment)
                          : TTI.isLegalMaskedStore(Ty, Alignment);
}

bool LoopVectorizationCostModel::memoryInstructionCanBeWidened(
    Instruction *I, ElementCount VF) {
  // Get and ensure we have a valid memory instruction.
  LoadInst *LI = dyn_cast<LoadInst>(I);
  StoreInst *SI = dyn_cast<StoreInst>(I);
  assert((LI || SI) && "Invalid memory instruction");

  auto *Ptr = getLoadStorePointerOperand(I);

  // In order to be widened, the pointer should be consecutive, first of all.
  if (!Legal->isConsecutivePtr(Ptr))
    return false;

  // If the instruction is a store located in a predicated block, it will be
  // scalarized.
  if (isScalarWithPredication(I))
    return false;

  // If the instruction's allocated size doesn't equal it's type size, it
  // requires padding and will be scalarized.
  auto &DL = I->getModule()->getDataLayout();
  auto *ScalarTy = LI ? LI->getType() : SI->getValueOperand()->getType();
  if (hasIrregularType(ScalarTy, DL))
    return false;

  return true;
}

void LoopVectorizationCostModel::collectLoopUniforms(ElementCount VF) {
  // We should not collect Uniforms more than once per VF. Right now,
  // this function is called from collectUniformsAndScalars(), which
  // already does this check. Collecting Uniforms for VF=1 does not make any
  // sense.

  assert(VF.isVector() && Uniforms.find(VF) == Uniforms.end() &&
         "This function should not be visited twice for the same VF");

  // Visit the list of Uniforms. If we'll not find any uniform value, we'll
  // not analyze again.  Uniforms.count(VF) will return 1.
  Uniforms[VF].clear();

  // We now know that the loop is vectorizable!
  // Collect instructions inside the loop that will remain uniform after
  // vectorization.

  // Global values, params and instructions outside of current loop are out of
  // scope.
  auto isOutOfScope = [&](Value *V) -> bool {
    Instruction *I = dyn_cast<Instruction>(V);
    return (!I || !TheLoop->contains(I));
  };

  SetVector<Instruction *> Worklist;
  BasicBlock *Latch = TheLoop->getLoopLatch();

  // Instructions that are scalar with predication must not be considered
  // uniform after vectorization, because that would create an erroneous
  // replicating region where only a single instance out of VF should be formed.
  // TODO: optimize such seldom cases if found important, see PR40816.
  auto addToWorklistIfAllowed = [&](Instruction *I) -> void {
    if (isOutOfScope(I)) {
      LLVM_DEBUG(dbgs() << "LV: Found not uniform due to scope: "
                        << *I << "\n");
      return;
    }
    if (isScalarWithPredication(I)) {
      LLVM_DEBUG(dbgs() << "LV: Found not uniform being ScalarWithPredication: "
                        << *I << "\n");
      return;
    }
    LLVM_DEBUG(dbgs() << "LV: Found uniform instruction: " << *I << "\n");
    Worklist.insert(I);
  };

  // Start with the conditional branch. If the branch condition is an
  // instruction contained in the loop that is only used by the branch, it is
  // uniform.
  auto *Cmp = dyn_cast<Instruction>(Latch->getTerminator()->getOperand(0));
  if (Cmp && TheLoop->contains(Cmp) && Cmp->hasOneUse())
    addToWorklistIfAllowed(Cmp);

  auto isUniformDecision = [&](Instruction *I, ElementCount VF) {
    InstWidening WideningDecision = getWideningDecision(I, VF);
    assert(WideningDecision != CM_Unknown &&
           "Widening decision should be ready at this moment");

    // A uniform memory op is itself uniform.  We exclude uniform stores
    // here as they demand the last lane, not the first one.
    if (isa<LoadInst>(I) && Legal->isUniformMemOp(*I)) {
      assert(WideningDecision == CM_Scalarize);
      return true;
    }

    return (WideningDecision == CM_Widen ||
            WideningDecision == CM_Widen_Reverse ||
            WideningDecision == CM_Interleave);
  };


  // Returns true if Ptr is the pointer operand of a memory access instruction
  // I, and I is known to not require scalarization.
  auto isVectorizedMemAccessUse = [&](Instruction *I, Value *Ptr) -> bool {
    return getLoadStorePointerOperand(I) == Ptr && isUniformDecision(I, VF);
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
      // If there's no pointer operand, there's nothing to do.
      auto *Ptr = getLoadStorePointerOperand(&I);
      if (!Ptr)
        continue;

      // A uniform memory op is itself uniform.  We exclude uniform stores
      // here as they demand the last lane, not the first one.
      if (isa<LoadInst>(I) && Legal->isUniformMemOp(I))
        addToWorklistIfAllowed(&I);

      if (isUniformDecision(&I, VF)) {
        assert(isVectorizedMemAccessUse(&I, Ptr) && "consistency check");
        HasUniformUse.insert(Ptr);
      }
    }

  // Add to the worklist any operands which have *only* uniform (e.g. lane 0
  // demanding) users.  Since loops are assumed to be in LCSSA form, this
  // disallows uses outside the loop as well.
  for (auto *V : HasUniformUse) {
    if (isOutOfScope(V))
      continue;
    auto *I = cast<Instruction>(V);
    auto UsersAreMemAccesses =
      llvm::all_of(I->users(), [&](User *U) -> bool {
        return isVectorizedMemAccessUse(cast<Instruction>(U), V);
      });
    if (UsersAreMemAccesses)
      addToWorklistIfAllowed(I);
  }

  // Expand Worklist in topological order: whenever a new instruction
  // is added , its users should be already inside Worklist.  It ensures
  // a uniform instruction will only be used by uniform instructions.
  unsigned idx = 0;
  while (idx != Worklist.size()) {
    Instruction *I = Worklist[idx++];

    for (auto OV : I->operand_values()) {
      // isOutOfScope operands cannot be uniform instructions.
      if (isOutOfScope(OV))
        continue;
      // First order recurrence Phi's should typically be considered
      // non-uniform.
      auto *OP = dyn_cast<PHINode>(OV);
      if (OP && Legal->isFirstOrderRecurrence(OP))
        continue;
      // If all the users of the operand are uniform, then add the
      // operand into the uniform worklist.
      auto *OI = cast<Instruction>(OV);
      if (llvm::all_of(OI->users(), [&](User *U) -> bool {
            auto *J = cast<Instruction>(U);
            return Worklist.count(J) || isVectorizedMemAccessUse(J, OI);
          }))
        addToWorklistIfAllowed(OI);
    }
  }

  // For an instruction to be added into Worklist above, all its users inside
  // the loop should also be in Worklist. However, this condition cannot be
  // true for phi nodes that form a cyclic dependence. We must process phi
  // nodes separately. An induction variable will remain uniform if all users
  // of the induction variable and induction variable update remain uniform.
  // The code below handles both pointer and non-pointer induction variables.
  for (auto &Induction : Legal->getInductionVars()) {
    auto *Ind = Induction.first;
    auto *IndUpdate = cast<Instruction>(Ind->getIncomingValueForBlock(Latch));

    // Determine if all users of the induction variable are uniform after
    // vectorization.
    auto UniformInd = llvm::all_of(Ind->users(), [&](User *U) -> bool {
      auto *I = cast<Instruction>(U);
      return I == IndUpdate || !TheLoop->contains(I) || Worklist.count(I) ||
             isVectorizedMemAccessUse(I, Ind);
    });
    if (!UniformInd)
      continue;

    // Determine if all users of the induction variable update instruction are
    // uniform after vectorization.
    auto UniformIndUpdate =
        llvm::all_of(IndUpdate->users(), [&](User *U) -> bool {
          auto *I = cast<Instruction>(U);
          return I == Ind || !TheLoop->contains(I) || Worklist.count(I) ||
                 isVectorizedMemAccessUse(I, IndUpdate);
        });
    if (!UniformIndUpdate)
      continue;

    // The induction variable and its update instruction will remain uniform.
    addToWorklistIfAllowed(Ind);
    addToWorklistIfAllowed(IndUpdate);
  }

  Uniforms[VF].insert(Worklist.begin(), Worklist.end());
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

  if (!PSE.getUnionPredicate().getPredicates().empty()) {
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

ElementCount
LoopVectorizationCostModel::getMaxLegalScalableVF(unsigned MaxSafeElements) {
  if (!TTI.supportsScalableVectors() && !ForceTargetSupportsScalableVectors) {
    reportVectorizationInfo(
        "Disabling scalable vectorization, because target does not "
        "support scalable vectors.",
        "ScalableVectorsUnsupported", ORE, TheLoop);
    return ElementCount::getScalable(0);
  }

  if (Hints->isScalableVectorizationDisabled()) {
    reportVectorizationInfo("Scalable vectorization is explicitly disabled",
                            "ScalableVectorizationDisabled", ORE, TheLoop);
    return ElementCount::getScalable(0);
  }

  auto MaxScalableVF = ElementCount::getScalable(
      std::numeric_limits<ElementCount::ScalarTy>::max());

  // Disable scalable vectorization if the loop contains unsupported reductions.
  // Test that the loop-vectorizer can legalize all operations for this MaxVF.
  // FIXME: While for scalable vectors this is currently sufficient, this should
  // be replaced by a more detailed mechanism that filters out specific VFs,
  // instead of invalidating vectorization for a whole set of VFs based on the
  // MaxVF.
  if (!canVectorizeReductions(MaxScalableVF)) {
    reportVectorizationInfo(
        "Scalable vectorization not supported for the reduction "
        "operations found in this loop.",
        "ScalableVFUnfeasible", ORE, TheLoop);
    return ElementCount::getScalable(0);
  }

  if (Legal->isSafeForAnyVectorWidth())
    return MaxScalableVF;

  // Limit MaxScalableVF by the maximum safe dependence distance.
  Optional<unsigned> MaxVScale = TTI.getMaxVScale();
  MaxScalableVF = ElementCount::getScalable(
      MaxVScale ? (MaxSafeElements / MaxVScale.getValue()) : 0);
  if (!MaxScalableVF)
    reportVectorizationInfo(
        "Max legal vector width too small, scalable vectorization "
        "unfeasible.",
        "ScalableVFUnfeasible", ORE, TheLoop);

  return MaxScalableVF;
}

FixedScalableVFPair
LoopVectorizationCostModel::computeFeasibleMaxVF(unsigned ConstTripCount,
                                                 ElementCount UserVF) {
  MinBWs = computeMinimumValueSizes(TheLoop->getBlocks(), *DB, &TTI);
  unsigned SmallestType, WidestType;
  std::tie(SmallestType, WidestType) = getSmallestAndWidestTypes();

  // Get the maximum safe dependence distance in bits computed by LAA.
  // It is computed by MaxVF * sizeOf(type) * 8, where type is taken from
  // the memory accesses that is most restrictive (involved in the smallest
  // dependence distance).
  unsigned MaxSafeElements =
      PowerOf2Floor(Legal->getMaxSafeVectorWidthInBits() / WidestType);

  auto MaxSafeFixedVF = ElementCount::getFixed(MaxSafeElements);
  auto MaxSafeScalableVF = getMaxLegalScalableVF(MaxSafeElements);

  LLVM_DEBUG(dbgs() << "LV: The max safe fixed VF is: " << MaxSafeFixedVF
                    << ".\n");
  LLVM_DEBUG(dbgs() << "LV: The max safe scalable VF is: " << MaxSafeScalableVF
                    << ".\n");

  // First analyze the UserVF, fall back if the UserVF should be ignored.
  if (UserVF) {
    auto MaxSafeUserVF =
        UserVF.isScalable() ? MaxSafeScalableVF : MaxSafeFixedVF;

    if (ElementCount::isKnownLE(UserVF, MaxSafeUserVF))
      return UserVF;

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

    LLVM_DEBUG(dbgs() << "LV: User VF=" << UserVF
                      << " is unsafe. Ignoring scalable UserVF.\n");
    ORE->emit([&]() {
      return OptimizationRemarkAnalysis(DEBUG_TYPE, "VectorizationFactor",
                                        TheLoop->getStartLoc(),
                                        TheLoop->getHeader())
             << "User-specified vectorization factor "
             << ore::NV("UserVectorizationFactor", UserVF)
             << " is unsafe. Ignoring the hint to let the compiler pick a "
                "suitable VF.";
    });
  }

  LLVM_DEBUG(dbgs() << "LV: The Smallest and Widest types: " << SmallestType
                    << " / " << WidestType << " bits.\n");

  FixedScalableVFPair Result(ElementCount::getFixed(1),
                             ElementCount::getScalable(0));
  if (auto MaxVF = getMaximizedVFForTarget(ConstTripCount, SmallestType,
                                           WidestType, MaxSafeFixedVF))
    Result.FixedVF = MaxVF;

  if (auto MaxVF = getMaximizedVFForTarget(ConstTripCount, SmallestType,
                                           WidestType, MaxSafeScalableVF))
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
    // TODO: It may by useful to do since it's still likely to be dynamically
    // uniform if the target can skip.
    reportVectorizationFailure(
        "Not inserting runtime ptr check for divergent target",
        "runtime pointer checks needed. Not enabled for divergent target",
        "CantVersionLoopWithDivergentTarget", ORE, TheLoop);
    return FixedScalableVFPair::getNone();
  }

  unsigned TC = PSE.getSE()->getSmallConstantTripCount(TheLoop);
  LLVM_DEBUG(dbgs() << "LV: Found trip count: " << TC << '\n');
  if (TC == 1) {
    reportVectorizationFailure("Single iteration (non) loop",
        "loop trip count is one, irrelevant for vectorization",
        "SingleIterationLoop", ORE, TheLoop);
    return FixedScalableVFPair::getNone();
  }

  switch (ScalarEpilogueStatus) {
  case CM_ScalarEpilogueAllowed:
    return computeFeasibleMaxVF(TC, UserVF);
  case CM_ScalarEpilogueNotAllowedUsePredicate:
    LLVM_FALLTHROUGH;
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

  // The only loops we can vectorize without a scalar epilogue, are loops with
  // a bottom-test and a single exiting block. We'd have to handle the fact
  // that not every instruction executes on the last iteration.  This will
  // require a lane mask which varies through the vector loop body.  (TODO)
  if (TheLoop->getExitingBlock() != TheLoop->getLoopLatch()) {
    // If there was a tail-folding hint/switch, but we can't fold the tail by
    // masking, fallback to a vectorization with a scalar epilogue.
    if (ScalarEpilogueStatus == CM_ScalarEpilogueNotNeededUsePredicate) {
      LLVM_DEBUG(dbgs() << "LV: Cannot fold tail by masking: vectorize with a "
                           "scalar epilogue instead.\n");
      ScalarEpilogueStatus = CM_ScalarEpilogueAllowed;
      return computeFeasibleMaxVF(TC, UserVF);
    }
    return FixedScalableVFPair::getNone();
  }

  // Now try the tail folding

  // Invalidate interleave groups that require an epilogue if we can't mask
  // the interleave-group.
  if (!useMaskedInterleavedAccesses(TTI)) {
    assert(WideningDecisions.empty() && Uniforms.empty() && Scalars.empty() &&
           "No decisions should have been taken at this point");
    // Note: There is no need to invalidate any cost modeling decisions here, as
    // non where taken so far.
    InterleaveInfo.invalidateGroupsRequiringScalarEpilogue();
  }

  FixedScalableVFPair MaxFactors = computeFeasibleMaxVF(TC, UserVF);
  // Avoid tail folding if the trip count is known to be a multiple of any VF
  // we chose.
  // FIXME: The condition below pessimises the case for fixed-width vectors,
  // when scalable VFs are also candidates for vectorization.
  if (MaxFactors.FixedVF.isVector() && !MaxFactors.ScalableVF) {
    ElementCount MaxFixedVF = MaxFactors.FixedVF;
    assert((UserVF.isNonZero() || isPowerOf2_32(MaxFixedVF.getFixedValue())) &&
           "MaxFixedVF must be a power of 2");
    unsigned MaxVFtimesIC = UserIC ? MaxFixedVF.getFixedValue() * UserIC
                                   : MaxFixedVF.getFixedValue();
    ScalarEvolution *SE = PSE.getSE();
    const SCEV *BackedgeTakenCount = PSE.getBackedgeTakenCount();
    const SCEV *ExitCount = SE->getAddExpr(
        BackedgeTakenCount, SE->getOne(BackedgeTakenCount->getType()));
    const SCEV *Rem = SE->getURemExpr(
        SE->applyLoopGuards(ExitCount, TheLoop),
        SE->getConstant(BackedgeTakenCount->getType(), MaxVFtimesIC));
    if (Rem->isZero()) {
      // Accept MaxFixedVF if we do not have a tail.
      LLVM_DEBUG(dbgs() << "LV: No tail will remain for any chosen VF.\n");
      return MaxFactors;
    }
  }

  // If we don't know the precise trip count, or if the trip count that we
  // found modulo the vectorization factor is not zero, try to fold the tail
  // by masking.
  // FIXME: look for a smaller MaxVF that does divide TC rather than masking.
  if (Legal->prepareToFoldTailByMasking()) {
    FoldTailByMasking = true;
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

  if (TC == 0) {
    reportVectorizationFailure(
        "Unable to calculate the loop count due to complex control flow",
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

ElementCount LoopVectorizationCostModel::getMaximizedVFForTarget(
    unsigned ConstTripCount, unsigned SmallestType, unsigned WidestType,
    const ElementCount &MaxSafeVF) {
  bool ComputeScalableMaxVF = MaxSafeVF.isScalable();
  TypeSize WidestRegister = TTI.getRegisterBitWidth(
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
      PowerOf2Floor(WidestRegister.getKnownMinSize() / WidestType),
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

  const auto TripCountEC = ElementCount::getFixed(ConstTripCount);
  if (ConstTripCount &&
      ElementCount::isKnownLE(TripCountEC, MaxVectorElementCount) &&
      isPowerOf2_32(ConstTripCount)) {
    // We need to clamp the VF to be the ConstTripCount. There is no point in
    // choosing a higher viable VF as done in the loop below. If
    // MaxVectorElementCount is scalable, we only fall back on a fixed VF when
    // the TC is less than or equal to the known number of lanes.
    LLVM_DEBUG(dbgs() << "LV: Clamping the MaxVF to the constant trip count: "
                      << ConstTripCount << "\n");
    return TripCountEC;
  }

  ElementCount MaxVF = MaxVectorElementCount;
  if (TTI.shouldMaximizeVectorBandwidth() ||
      (MaximizeBandwidth && isScalarEpilogueAllowed())) {
    auto MaxVectorElementCountMaxBW = ElementCount::get(
        PowerOf2Floor(WidestRegister.getKnownMinSize() / SmallestType),
        ComputeScalableMaxVF);
    MaxVectorElementCountMaxBW = MinVF(MaxVectorElementCountMaxBW, MaxSafeVF);

    // Collect all viable vectorization factors larger than the default MaxVF
    // (i.e. MaxVectorElementCount).
    SmallVector<ElementCount, 8> VFs;
    for (ElementCount VS = MaxVectorElementCount * 2;
         ElementCount::isKnownLE(VS, MaxVectorElementCountMaxBW); VS *= 2)
      VFs.push_back(VS);

    // For each VF calculate its register usage.
    auto RUs = calculateRegisterUsage(VFs);

    // Select the largest VF which doesn't require more registers than existing
    // ones.
    for (int i = RUs.size() - 1; i >= 0; --i) {
      bool Selected = true;
      for (auto &pair : RUs[i].MaxLocalUsers) {
        unsigned TargetNumRegisters = TTI.getNumberOfRegisters(pair.first);
        if (pair.second > TargetNumRegisters)
          Selected = false;
      }
      if (Selected) {
        MaxVF = VFs[i];
        break;
      }
    }
    if (ElementCount MinVF =
            TTI.getMinimumVF(SmallestType, ComputeScalableMaxVF)) {
      if (ElementCount::isKnownLT(MaxVF, MinVF)) {
        LLVM_DEBUG(dbgs() << "LV: Overriding calculated MaxVF(" << MaxVF
                          << ") with target's minimum: " << MinVF << '\n');
        MaxVF = MinVF;
      }
    }
  }
  return MaxVF;
}

bool LoopVectorizationCostModel::isMoreProfitable(
    const VectorizationFactor &A, const VectorizationFactor &B) const {
  InstructionCost::CostType CostA = *A.Cost.getValue();
  InstructionCost::CostType CostB = *B.Cost.getValue();

  unsigned MaxTripCount = PSE.getSE()->getSmallConstantMaxTripCount(TheLoop);

  if (!A.Width.isScalable() && !B.Width.isScalable() && FoldTailByMasking &&
      MaxTripCount) {
    // If we are folding the tail and the trip count is a known (possibly small)
    // constant, the trip count will be rounded up to an integer number of
    // iterations. The total cost will be PerIterationCost*ceil(TripCount/VF),
    // which we compare directly. When not folding the tail, the total cost will
    // be PerIterationCost*floor(TC/VF) + Scalar remainder cost, and so is
    // approximated with the per-lane cost below instead of using the tripcount
    // as here.
    int64_t RTCostA = CostA * divideCeil(MaxTripCount, A.Width.getFixedValue());
    int64_t RTCostB = CostB * divideCeil(MaxTripCount, B.Width.getFixedValue());
    return RTCostA < RTCostB;
  }

  // When set to preferred, for now assume vscale may be larger than 1, so
  // that scalable vectorization is slightly favorable over fixed-width
  // vectorization.
  if (Hints->isScalableVectorizationPreferred())
    if (A.Width.isScalable() && !B.Width.isScalable())
      return (CostA * B.Width.getKnownMinValue()) <=
             (CostB * A.Width.getKnownMinValue());

  // To avoid the need for FP division:
  //      (CostA / A.Width) < (CostB / B.Width)
  // <=>  (CostA * B.Width) < (CostB * A.Width)
  return (CostA * B.Width.getKnownMinValue()) <
         (CostB * A.Width.getKnownMinValue());
}

VectorizationFactor LoopVectorizationCostModel::selectVectorizationFactor(
    const ElementCountSet &VFCandidates) {
  InstructionCost ExpectedCost = expectedCost(ElementCount::getFixed(1)).first;
  LLVM_DEBUG(dbgs() << "LV: Scalar loop costs: " << ExpectedCost << ".\n");
  assert(ExpectedCost.isValid() && "Unexpected invalid cost for scalar loop");
  assert(VFCandidates.count(ElementCount::getFixed(1)) &&
         "Expected Scalar VF to be a candidate");

  const VectorizationFactor ScalarCost(ElementCount::getFixed(1), ExpectedCost);
  VectorizationFactor ChosenFactor = ScalarCost;

  bool ForceVectorization = Hints->getForce() == LoopVectorizeHints::FK_Enabled;
  if (ForceVectorization && VFCandidates.size() > 1) {
    // Ignore scalar width, because the user explicitly wants vectorization.
    // Initialize cost to max so that VF = 2 is, at least, chosen during cost
    // evaluation.
    ChosenFactor.Cost = std::numeric_limits<InstructionCost::CostType>::max();
  }

  for (const auto &i : VFCandidates) {
    // The cost for scalar VF=1 is already calculated, so ignore it.
    if (i.isScalar())
      continue;

    // Notice that the vector loop needs to be executed less times, so
    // we need to divide the cost of the vector loops by the width of
    // the vector elements.
    VectorizationCostTy C = expectedCost(i);

    assert(C.first.isValid() && "Unexpected invalid cost for vector loop");
    VectorizationFactor Candidate(i, C.first);
    LLVM_DEBUG(
        dbgs() << "LV: Vector loop of width " << i << " costs: "
               << (*Candidate.Cost.getValue() /
                   Candidate.Width.getKnownMinValue())
               << (i.isScalable() ? " (assuming a minimum vscale of 1)" : "")
               << ".\n");

    if (!C.second && !ForceVectorization) {
      LLVM_DEBUG(
          dbgs() << "LV: Not considering vector loop of width " << i
                 << " because it will not generate any vector instructions.\n");
      continue;
    }

    // If profitable add it to ProfitableVF list.
    if (isMoreProfitable(Candidate, ScalarCost))
      ProfitableVFs.push_back(Candidate);

    if (isMoreProfitable(Candidate, ChosenFactor))
      ChosenFactor = Candidate;
  }

  if (!EnableCondStoresVectorization && NumPredStores) {
    reportVectorizationFailure("There are conditional stores.",
        "store that is conditionally executed prevents vectorization",
        "ConditionalStore", ORE, TheLoop);
    ChosenFactor = ScalarCost;
  }

  LLVM_DEBUG(if (ForceVectorization && !ChosenFactor.Width.isScalar() &&
                 *ChosenFactor.Cost.getValue() >= *ScalarCost.Cost.getValue())
                 dbgs()
             << "LV: Vectorization seems to be not beneficial, "
             << "but was forced by a user.\n");
  LLVM_DEBUG(dbgs() << "LV: Selecting VF: " << ChosenFactor.Width << ".\n");
  return ChosenFactor;
}

bool LoopVectorizationCostModel::isCandidateForEpilogueVectorization(
    const Loop &L, ElementCount VF) const {
  // Cross iteration phis such as reductions need special handling and are
  // currently unsupported.
  if (any_of(L.getHeader()->phis(), [&](PHINode &Phi) {
        return Legal->isFirstOrderRecurrence(&Phi) ||
               Legal->isReductionVariable(&Phi);
      }))
    return false;

  // Phis with uses outside of the loop require special handling and are
  // currently unsupported.
  for (auto &Entry : Legal->getInductionVars()) {
    // Look for uses of the value of the induction at the last iteration.
    Value *PostInc = Entry.first->getIncomingValueForBlock(L.getLoopLatch());
    for (User *U : PostInc->users())
      if (!L.contains(cast<Instruction>(U)))
        return false;
    // Look for uses of penultimate value of the induction.
    for (User *U : Entry.first->users())
      if (!L.contains(cast<Instruction>(U)))
        return false;
  }

  // Induction variables that are widened require special handling that is
  // currently not supported.
  if (any_of(Legal->getInductionVars(), [&](auto &Entry) {
        return !(this->isScalarAfterVectorization(Entry.first, VF) ||
                 this->isProfitableToScalarize(Entry.first, VF));
      }))
    return false;

  return true;
}

bool LoopVectorizationCostModel::isEpilogueVectorizationProfitable(
    const ElementCount VF) const {
  // FIXME: We need a much better cost-model to take different parameters such
  // as register pressure, code size increase and cost of extra branches into
  // account. For now we apply a very crude heuristic and only consider loops
  // with vectorization factors larger than a certain value.
  // We also consider epilogue vectorization unprofitable for targets that don't
  // consider interleaving beneficial (eg. MVE).
  if (TTI.getMaxInterleaveFactor(VF.getKnownMinValue()) <= 1)
    return false;
  if (VF.getFixedValue() >= EpilogueVectorizationMinVF)
    return true;
  return false;
}

VectorizationFactor
LoopVectorizationCostModel::selectEpilogueVectorizationFactor(
    const ElementCount MainLoopVF, const LoopVectorizationPlanner &LVP) {
  VectorizationFactor Result = VectorizationFactor::Disabled();
  if (!EnableEpilogueVectorization) {
    LLVM_DEBUG(dbgs() << "LEV: Epilogue vectorization is disabled.\n";);
    return Result;
  }

  if (!isScalarEpilogueAllowed()) {
    LLVM_DEBUG(
        dbgs() << "LEV: Unable to vectorize epilogue because no epilogue is "
                  "allowed.\n";);
    return Result;
  }

  // FIXME: This can be fixed for scalable vectors later, because at this stage
  // the LoopVectorizer will only consider vectorizing a loop with scalable
  // vectors when the loop has a hint to enable vectorization for a given VF.
  if (MainLoopVF.isScalable()) {
    LLVM_DEBUG(dbgs() << "LEV: Epilogue vectorization for scalable vectors not "
                         "yet supported.\n");
    return Result;
  }

  // Not really a cost consideration, but check for unsupported cases here to
  // simplify the logic.
  if (!isCandidateForEpilogueVectorization(*TheLoop, MainLoopVF)) {
    LLVM_DEBUG(
        dbgs() << "LEV: Unable to vectorize epilogue because the loop is "
                  "not a supported candidate.\n";);
    return Result;
  }

  if (EpilogueVectorizationForceVF > 1) {
    LLVM_DEBUG(dbgs() << "LEV: Epilogue vectorization factor is forced.\n";);
    if (LVP.hasPlanWithVFs(
            {MainLoopVF, ElementCount::getFixed(EpilogueVectorizationForceVF)}))
      return {ElementCount::getFixed(EpilogueVectorizationForceVF), 0};
    else {
      LLVM_DEBUG(
          dbgs()
              << "LEV: Epilogue vectorization forced factor is not viable.\n";);
      return Result;
    }
  }

  if (TheLoop->getHeader()->getParent()->hasOptSize() ||
      TheLoop->getHeader()->getParent()->hasMinSize()) {
    LLVM_DEBUG(
        dbgs()
            << "LEV: Epilogue vectorization skipped due to opt for size.\n";);
    return Result;
  }

  if (!isEpilogueVectorizationProfitable(MainLoopVF))
    return Result;

  for (auto &NextVF : ProfitableVFs)
    if (ElementCount::isKnownLT(NextVF.Width, MainLoopVF) &&
        (Result.Width.getFixedValue() == 1 ||
         isMoreProfitable(NextVF, Result)) &&
        LVP.hasPlanWithVFs({MainLoopVF, NextVF.Width}))
      Result = NextVF;

  if (Result != VectorizationFactor::Disabled())
    LLVM_DEBUG(dbgs() << "LEV: Vectorizing epilogue loop with VF = "
                      << Result.Width.getFixedValue() << "\n";);
  return Result;
}

std::pair<unsigned, unsigned>
LoopVectorizationCostModel::getSmallestAndWidestTypes() {
  unsigned MinWidth = -1U;
  unsigned MaxWidth = 8;
  const DataLayout &DL = TheFunction->getParent()->getDataLayout();

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
        const RecurrenceDescriptor &RdxDesc = Legal->getReductionVars()[PN];
        if (PreferInLoopReductions || useOrderedReductions(RdxDesc) ||
            TTI.preferInLoopReduction(RdxDesc.getOpcode(),
                                      RdxDesc.getRecurrenceType(),
                                      TargetTransformInfo::ReductionFlags()))
          continue;
        T = RdxDesc.getRecurrenceType();
      }

      // Examine the stored values.
      if (auto *ST = dyn_cast<StoreInst>(&I))
        T = ST->getValueOperand()->getType();

      // Ignore loaded pointer types and stored pointer types that are not
      // vectorizable.
      //
      // FIXME: The check here attempts to predict whether a load or store will
      //        be vectorized. We only know this for certain after a VF has
      //        been selected. Here, we assume that if an access can be
      //        vectorized, it will be. We should also look at extending this
      //        optimization to non-pointer types.
      //
      if (T->isPointerTy() && !isConsecutiveLoadOrStore(&I) &&
          !isAccessInterleaved(&I) && !isLegalGatherOrScatter(&I))
        continue;

      MinWidth = std::min(MinWidth,
                          (unsigned)DL.getTypeSizeInBits(T->getScalarType()));
      MaxWidth = std::max(MaxWidth,
                          (unsigned)DL.getTypeSizeInBits(T->getScalarType()));
    }
  }

  return {MinWidth, MaxWidth};
}

unsigned LoopVectorizationCostModel::selectInterleaveCount(ElementCount VF,
                                                           unsigned LoopCost) {
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

  if (!isScalarEpilogueAllowed())
    return 1;

  // We used the distance for the interleave count.
  if (Legal->getMaxSafeDepDistBytes() != -1U)
    return 1;

  auto BestKnownTC = getSmallBestKnownTC(*PSE.getSE(), TheLoop);
  const bool HasReductions = !Legal->getReductionVars().empty();
  // Do not interleave loops with a relatively small known or estimated trip
  // count. But we will interleave when InterleaveSmallLoopScalarReduction is
  // enabled, and the code has scalar reductions(HasReductions && VF = 1),
  // because with the above conditions interleaving can expose ILP and break
  // cross iteration dependences for reductions.
  if (BestKnownTC && (*BestKnownTC < TinyTripCountInterleaveThreshold) &&
      !(InterleaveSmallLoopScalarReduction && HasReductions && VF.isScalar()))
    return 1;

  RegisterUsage R = calculateRegisterUsage({VF})[0];
  // We divide by these constants so assume that we have at least one
  // instruction that uses at least one register.
  for (auto& pair : R.MaxLocalUsers) {
    pair.second = std::max(pair.second, 1U);
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

  for (auto& pair : R.MaxLocalUsers) {
    unsigned TargetNumRegisters = TTI.getNumberOfRegisters(pair.first);
    LLVM_DEBUG(dbgs() << "LV: The target has " << TargetNumRegisters
                      << " registers of "
                      << TTI.getRegisterClassName(pair.first) << " register class\n");
    if (VF.isScalar()) {
      if (ForceTargetNumScalarRegs.getNumOccurrences() > 0)
        TargetNumRegisters = ForceTargetNumScalarRegs;
    } else {
      if (ForceTargetNumVectorRegs.getNumOccurrences() > 0)
        TargetNumRegisters = ForceTargetNumVectorRegs;
    }
    unsigned MaxLocalUsers = pair.second;
    unsigned LoopInvariantRegs = 0;
    if (R.LoopInvariantRegs.find(pair.first) != R.LoopInvariantRegs.end())
      LoopInvariantRegs = R.LoopInvariantRegs[pair.first];

    unsigned TmpIC = PowerOf2Floor((TargetNumRegisters - LoopInvariantRegs) / MaxLocalUsers);
    // Don't count the induction variable as interleaved.
    if (EnableIndVarRegisterHeur) {
      TmpIC =
          PowerOf2Floor((TargetNumRegisters - LoopInvariantRegs - 1) /
                        std::max(1U, (MaxLocalUsers - 1)));
    }

    IC = std::min(IC, TmpIC);
  }

  // Clamp the interleave ranges to reasonable counts.
  unsigned MaxInterleaveCount =
      TTI.getMaxInterleaveFactor(VF.getKnownMinValue());

  // Check if the user has overridden the max.
  if (VF.isScalar()) {
    if (ForceTargetMaxScalarInterleaveFactor.getNumOccurrences() > 0)
      MaxInterleaveCount = ForceTargetMaxScalarInterleaveFactor;
  } else {
    if (ForceTargetMaxVectorInterleaveFactor.getNumOccurrences() > 0)
      MaxInterleaveCount = ForceTargetMaxVectorInterleaveFactor;
  }

  // If trip count is known or estimated compile time constant, limit the
  // interleave count to be less than the trip count divided by VF, provided it
  // is at least 1.
  //
  // For scalable vectors we can't know if interleaving is beneficial. It may
  // not be beneficial for small loops if none of the lanes in the second vector
  // iterations is enabled. However, for larger loops, there is likely to be a
  // similar benefit as for fixed-width vectors. For now, we choose to leave
  // the InterleaveCount as if vscale is '1', although if some information about
  // the vector is known (e.g. min vector size), we can make a better decision.
  if (BestKnownTC) {
    MaxInterleaveCount =
        std::min(*BestKnownTC / VF.getKnownMinValue(), MaxInterleaveCount);
    // Make sure MaxInterleaveCount is greater than 0.
    MaxInterleaveCount = std::max(1u, MaxInterleaveCount);
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

  // If we did not calculate the cost for VF (because the user selected the VF)
  // then we calculate the cost of VF here.
  if (LoopCost == 0) {
    assert(expectedCost(VF).first.isValid() && "Expected a valid cost");
    LoopCost = *expectedCost(VF).first.getValue();
  }

  assert(LoopCost && "Non-zero loop cost expected");

  // Interleave if we vectorized this loop and there is a reduction that could
  // benefit from interleaving.
  if (VF.isVector() && HasReductions) {
    LLVM_DEBUG(dbgs() << "LV: Interleaving because of reductions.\n");
    return IC;
  }

  // Note that if we've already vectorized the loop we will have done the
  // runtime check and so interleaving won't require further checks.
  bool InterleavingRequiresRuntimePointerCheck =
      (VF.isScalar() && Legal->getRuntimePointerChecking()->Need);

  // We want to interleave small loops in order to reduce the loop overhead and
  // potentially expose ILP opportunities.
  LLVM_DEBUG(dbgs() << "LV: Loop cost is " << LoopCost << '\n'
                    << "LV: IC is " << IC << '\n'
                    << "LV: VF is " << VF << '\n');
  const bool AggressivelyInterleaveReductions =
      TTI.enableAggressiveInterleaving(HasReductions);
  if (!InterleavingRequiresRuntimePointerCheck && LoopCost < SmallLoopCost) {
    // We assume that the cost overhead is 1 and we use the cost model
    // to estimate the cost of the loop and interleave until the cost of the
    // loop overhead is about 5% of the cost of the loop.
    unsigned SmallIC =
        std::min(IC, (unsigned)PowerOf2Floor(SmallLoopCost / LoopCost));

    // Interleave until store/load ports (estimated by max interleave count) are
    // saturated.
    unsigned NumStores = Legal->getNumStores();
    unsigned NumLoads = Legal->getNumLoads();
    unsigned StoresIC = IC / (NumStores ? NumStores : 1);
    unsigned LoadsIC = IC / (NumLoads ? NumLoads : 1);

    // If we have a scalar reduction (vector reductions are already dealt with
    // by this point), we can increase the critical path length if the loop
    // we're interleaving is inside another loop. Limit, by default to 2, so the
    // critical path only gets increased by one reduction operation.
    if (HasReductions && TheLoop->getLoopDepth() > 1) {
      unsigned F = static_cast<unsigned>(MaxNestedScalarReductionIC);
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
    if (InterleaveSmallLoopScalarReduction && VF.isScalar() &&
        AggressivelyInterleaveReductions) {
      LLVM_DEBUG(dbgs() << "LV: Interleaving to expose ILP.\n");
      // Interleave no less than SmallIC but not as aggressive as the normal IC
      // to satisfy the rare situation when resources are too limited.
      return std::max(IC / 2, SmallIC);
    } else {
      LLVM_DEBUG(dbgs() << "LV: Interleaving to reduce branch cost.\n");
      return SmallIC;
    }
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

SmallVector<LoopVectorizationCostModel::RegisterUsage, 8>
LoopVectorizationCostModel::calculateRegisterUsage(ArrayRef<ElementCount> VFs) {
  // This function calculates the register usage by measuring the highest number
  // of values that are alive at a single location. Obviously, this is a very
  // rough estimation. We scan the loop in a topological order in order and
  // assign a number to each instruction. We use RPO to ensure that defs are
  // met before their users. We assume that each instruction that has in-loop
  // users starts an interval. We record every time that an in-loop value is
  // used, so we have a list of the first and last occurrences of each
  // instruction. Next, we transpose this data structure into a multi map that
  // holds the list of intervals that *end* at a specific location. This multi
  // map allows us to perform a linear search. We scan the instructions linearly
  // and record each time that a new interval starts, by placing it in a set.
  // If we find this value in the multi-map then we remove it from the set.
  // The max register usage is the maximum size of the set.
  // We also search for instructions that are defined outside the loop, but are
  // used inside the loop. We need this number separately from the max-interval
  // usage number because when we unroll, loop-invariant values do not take
  // more register.
  LoopBlocksDFS DFS(TheLoop);
  DFS.perform(LI);

  RegisterUsage RU;

  // Each 'key' in the map opens a new interval. The values
  // of the map are the index of the 'last seen' usage of the
  // instruction that is the key.
  using IntervalMap = DenseMap<Instruction *, unsigned>;

  // Maps instruction to its index.
  SmallVector<Instruction *, 64> IdxToInstr;
  // Marks the end of each interval.
  IntervalMap EndPoint;
  // Saves the list of instruction indices that are used in the loop.
  SmallPtrSet<Instruction *, 8> Ends;
  // Saves the list of values that are used in the loop but are
  // defined outside the loop, such as arguments and constants.
  SmallPtrSet<Value *, 8> LoopInvariants;

  for (BasicBlock *BB : make_range(DFS.beginRPO(), DFS.endRPO())) {
    for (Instruction &I : BB->instructionsWithoutDebug()) {
      IdxToInstr.push_back(&I);

      // Save the end location of each USE.
      for (Value *U : I.operands()) {
        auto *Instr = dyn_cast<Instruction>(U);

        // Ignore non-instruction values such as arguments, constants, etc.
        if (!Instr)
          continue;

        // If this instruction is outside the loop then record it and continue.
        if (!TheLoop->contains(Instr)) {
          LoopInvariants.insert(Instr);
          continue;
        }

        // Overwrite previous end points.
        EndPoint[Instr] = IdxToInstr.size();
        Ends.insert(Instr);
      }
    }
  }

  // Saves the list of intervals that end with the index in 'key'.
  using InstrList = SmallVector<Instruction *, 2>;
  DenseMap<unsigned, InstrList> TransposeEnds;

  // Transpose the EndPoints to a list of values that end at each index.
  for (auto &Interval : EndPoint)
    TransposeEnds[Interval.second].push_back(Interval.first);

  SmallPtrSet<Instruction *, 8> OpenIntervals;
  SmallVector<RegisterUsage, 8> RUs(VFs.size());
  SmallVector<SmallMapVector<unsigned, unsigned, 4>, 8> MaxUsages(VFs.size());

  LLVM_DEBUG(dbgs() << "LV(REG): Calculating max register usage:\n");

  // A lambda that gets the register usage for the given type and VF.
  const auto &TTICapture = TTI;
  auto GetRegUsage = [&TTICapture](Type *Ty, ElementCount VF) {
    if (Ty->isTokenTy() || !VectorType::isValidElementType(Ty))
      return 0;
    return *TTICapture.getRegUsageForType(VectorType::get(Ty, VF)).getValue();
  };

  for (unsigned int i = 0, s = IdxToInstr.size(); i < s; ++i) {
    Instruction *I = IdxToInstr[i];

    // Remove all of the instructions that end at this location.
    InstrList &List = TransposeEnds[i];
    for (Instruction *ToRemove : List)
      OpenIntervals.erase(ToRemove);

    // Ignore instructions that are never used within the loop.
    if (!Ends.count(I))
      continue;

    // Skip ignored values.
    if (ValuesToIgnore.count(I))
      continue;

    // For each VF find the maximum usage of registers.
    for (unsigned j = 0, e = VFs.size(); j < e; ++j) {
      // Count the number of live intervals.
      SmallMapVector<unsigned, unsigned, 4> RegUsage;

      if (VFs[j].isScalar()) {
        for (auto Inst : OpenIntervals) {
          unsigned ClassID = TTI.getRegisterClassForType(false, Inst->getType());
          if (RegUsage.find(ClassID) == RegUsage.end())
            RegUsage[ClassID] = 1;
          else
            RegUsage[ClassID] += 1;
        }
      } else {
        collectUniformsAndScalars(VFs[j]);
        for (auto Inst : OpenIntervals) {
          // Skip ignored values for VF > 1.
          if (VecValuesToIgnore.count(Inst))
            continue;
          if (isScalarAfterVectorization(Inst, VFs[j])) {
            unsigned ClassID = TTI.getRegisterClassForType(false, Inst->getType());
            if (RegUsage.find(ClassID) == RegUsage.end())
              RegUsage[ClassID] = 1;
            else
              RegUsage[ClassID] += 1;
          } else {
            unsigned ClassID = TTI.getRegisterClassForType(true, Inst->getType());
            if (RegUsage.find(ClassID) == RegUsage.end())
              RegUsage[ClassID] = GetRegUsage(Inst->getType(), VFs[j]);
            else
              RegUsage[ClassID] += GetRegUsage(Inst->getType(), VFs[j]);
          }
        }
      }

      for (auto& pair : RegUsage) {
        if (MaxUsages[j].find(pair.first) != MaxUsages[j].end())
          MaxUsages[j][pair.first] = std::max(MaxUsages[j][pair.first], pair.second);
        else
          MaxUsages[j][pair.first] = pair.second;
      }
    }

    LLVM_DEBUG(dbgs() << "LV(REG): At #" << i << " Interval # "
                      << OpenIntervals.size() << '\n');

    // Add the current instruction to the list of open intervals.
    OpenIntervals.insert(I);
  }

  for (unsigned i = 0, e = VFs.size(); i < e; ++i) {
    SmallMapVector<unsigned, unsigned, 4> Invariant;

    for (auto Inst : LoopInvariants) {
      unsigned Usage =
          VFs[i].isScalar() ? 1 : GetRegUsage(Inst->getType(), VFs[i]);
      unsigned ClassID =
          TTI.getRegisterClassForType(VFs[i].isVector(), Inst->getType());
      if (Invariant.find(ClassID) == Invariant.end())
        Invariant[ClassID] = Usage;
      else
        Invariant[ClassID] += Usage;
    }

    LLVM_DEBUG({
      dbgs() << "LV(REG): VF = " << VFs[i] << '\n';
      dbgs() << "LV(REG): Found max usage: " << MaxUsages[i].size()
             << " item\n";
      for (const auto &pair : MaxUsages[i]) {
        dbgs() << "LV(REG): RegisterClass: "
               << TTI.getRegisterClassName(pair.first) << ", " << pair.second
               << " registers\n";
      }
      dbgs() << "LV(REG): Found invariant usage: " << Invariant.size()
             << " item\n";
      for (const auto &pair : Invariant) {
        dbgs() << "LV(REG): RegisterClass: "
               << TTI.getRegisterClassName(pair.first) << ", " << pair.second
               << " registers\n";
      }
    });

    RU.LoopInvariantRegs = Invariant;
    RU.MaxLocalUsers = MaxUsages[i];
    RUs[i] = RU;
  }

  return RUs;
}

bool LoopVectorizationCostModel::useEmulatedMaskMemRefHack(Instruction *I){
  // TODO: Cost model for emulated masked load/store is completely
  // broken. This hack guides the cost model to use an artificially
  // high enough value to practically disable vectorization with such
  // operations, except where previously deployed legality hack allowed
  // using very low cost values. This is to avoid regressions coming simply
  // from moving "masked load/store" check from legality to cost model.
  // Masked Load/Gather emulation was previously never allowed.
  // Limited number of Masked Store/Scatter emulation was allowed.
  assert(isPredicatedInst(I) &&
         "Expecting a scalar emulated instruction");
  return isa<LoadInst>(I) ||
         (isa<StoreInst>(I) &&
          NumPredStores > NumberOfStoresToPredicate);
}

void LoopVectorizationCostModel::collectInstsToScalarize(ElementCount VF) {
  // If we aren't vectorizing the loop, or if we've already collected the
  // instructions to scalarize, there's nothing to do. Collection may already
  // have occurred if we have a user-selected VF and are now computing the
  // expected cost for interleaving.
  if (VF.isScalar() || VF.isZero() ||
      InstsToScalarize.find(VF) != InstsToScalarize.end())
    return;

  // Initialize a mapping for VF in InstsToScalalarize. If we find that it's
  // not profitable to scalarize any instructions, the presence of VF in the
  // map will indicate that we've analyzed it already.
  ScalarCostsTy &ScalarCostsVF = InstsToScalarize[VF];

  // Find all the instructions that are scalar with predication in the loop and
  // determine if it would be better to not if-convert the blocks they are in.
  // If so, we also record the instructions to scalarize.
  for (BasicBlock *BB : TheLoop->blocks()) {
    if (!blockNeedsPredication(BB))
      continue;
    for (Instruction &I : *BB)
      if (isScalarWithPredication(&I)) {
        ScalarCostsTy ScalarCosts;
        // Do not apply discount logic if hacked cost is needed
        // for emulated masked memrefs.
        if (!useEmulatedMaskMemRefHack(&I) &&
            computePredInstDiscount(&I, ScalarCosts, VF) >= 0)
          ScalarCostsVF.insert(ScalarCosts.begin(), ScalarCosts.end());
        // Remember that BB will remain after vectorization.
        PredicatedBBsAfterVectorization.insert(BB);
      }
  }
}

int LoopVectorizationCostModel::computePredInstDiscount(
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
  auto canBeScalarized = [&](Instruction *I) -> bool {
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
    if (isScalarWithPredication(I))
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
    if (ScalarCosts.find(I) != ScalarCosts.end())
      continue;

    // Compute the cost of the vector instruction. Note that this cost already
    // includes the scalarization overhead of the predicated instruction.
    InstructionCost VectorCost = getInstructionCost(I, VF).first;

    // Compute the cost of the scalarized instruction. This cost is the cost of
    // the instruction as if it wasn't if-converted and instead remained in the
    // predicated block. We will scale this cost by block probability after
    // computing the scalarization overhead.
    assert(!VF.isScalable() && "scalable vectors not yet supported.");
    InstructionCost ScalarCost =
        VF.getKnownMinValue() *
        getInstructionCost(I, ElementCount::getFixed(1)).first;

    // Compute the scalarization overhead of needed insertelement instructions
    // and phi nodes.
    if (isScalarWithPredication(I) && !I->getType()->isVoidTy()) {
      ScalarCost += TTI.getScalarizationOverhead(
          cast<VectorType>(ToVectorTy(I->getType(), VF)),
          APInt::getAllOnesValue(VF.getKnownMinValue()), true, false);
      assert(!VF.isScalable() && "scalable vectors not yet supported.");
      ScalarCost +=
          VF.getKnownMinValue() *
          TTI.getCFInstrCost(Instruction::PHI, TTI::TCK_RecipThroughput);
    }

    // Compute the scalarization overhead of needed extractelement
    // instructions. For each of the instruction's operands, if the operand can
    // be scalarized, add it to the worklist; otherwise, account for the
    // overhead.
    for (Use &U : I->operands())
      if (auto *J = dyn_cast<Instruction>(U.get())) {
        assert(VectorType::isValidElementType(J->getType()) &&
               "Instruction has non-scalar type");
        if (canBeScalarized(J))
          Worklist.push_back(J);
        else if (needsExtract(J, VF)) {
          assert(!VF.isScalable() && "scalable vectors not yet supported.");
          ScalarCost += TTI.getScalarizationOverhead(
              cast<VectorType>(ToVectorTy(J->getType(), VF)),
              APInt::getAllOnesValue(VF.getKnownMinValue()), false, true);
        }
      }

    // Scale the total scalar cost by block probability.
    ScalarCost /= getReciprocalPredBlockProb();

    // Compute the discount. A non-negative discount means the vector version
    // of the instruction costs more, and scalarizing would be beneficial.
    Discount += VectorCost - ScalarCost;
    ScalarCosts[I] = ScalarCost;
  }

  return *Discount.getValue();
}

LoopVectorizationCostModel::VectorizationCostTy
LoopVectorizationCostModel::expectedCost(ElementCount VF) {
  VectorizationCostTy Cost;

  // For each block.
  for (BasicBlock *BB : TheLoop->blocks()) {
    VectorizationCostTy BlockCost;

    // For each instruction in the old loop.
    for (Instruction &I : BB->instructionsWithoutDebug()) {
      // Skip ignored values.
      if (ValuesToIgnore.count(&I) ||
          (VF.isVector() && VecValuesToIgnore.count(&I)))
        continue;

      VectorizationCostTy C = getInstructionCost(&I, VF);

      // Check if we should override the cost.
      if (ForceTargetInstructionCost.getNumOccurrences() > 0)
        C.first = InstructionCost(ForceTargetInstructionCost);

      BlockCost.first += C.first;
      BlockCost.second |= C.second;
      LLVM_DEBUG(dbgs() << "LV: Found an estimated cost of " << C.first
                        << " for VF " << VF << " For instruction: " << I
                        << '\n');
    }

    // If we are vectorizing a predicated block, it will have been
    // if-converted. This means that the block's instructions (aside from
    // stores and instructions that may divide by zero) will now be
    // unconditionally executed. For the scalar case, we may not always execute
    // the predicated block, if it is an if-else block. Thus, scale the block's
    // cost by the probability of executing it. blockNeedsPredication from
    // Legal is used so as to not include all blocks in tail folded loops.
    if (VF.isScalar() && Legal->blockNeedsPredication(BB))
      BlockCost.first /= getReciprocalPredBlockProb();

    Cost.first += BlockCost.first;
    Cost.second |= BlockCost.second;
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
  auto SE = PSE.getSE();
  unsigned NumOperands = Gep->getNumOperands();
  for (unsigned i = 1; i < NumOperands; ++i) {
    Value *Opd = Gep->getOperand(i);
    if (!SE->isLoopInvariant(SE->getSCEV(Opd), TheLoop) &&
        !Legal->isInductionVariable(Opd))
      return nullptr;
  }

  // Now we know we have a GEP ptr, %inv, %ind, %inv. return the Ptr SCEV.
  return PSE.getSCEV(Ptr);
}

static bool isStrideMul(Instruction *I, LoopVectorizationLegality *Legal) {
  return Legal->hasStride(I->getOperand(0)) ||
         Legal->hasStride(I->getOperand(1));
}

InstructionCost
LoopVectorizationCostModel::getMemInstScalarizationCost(Instruction *I,
                                                        ElementCount VF) {
  assert(VF.isVector() &&
         "Scalarization cost of instruction implies vectorization.");
  if (VF.isScalable())
    return InstructionCost::getInvalid();

  Type *ValTy = getLoadStoreType(I);
  auto SE = PSE.getSE();

  unsigned AS = getLoadStoreAddressSpace(I);
  Value *Ptr = getLoadStorePointerOperand(I);
  Type *PtrTy = ToVectorTy(Ptr->getType(), VF);

  // Figure out whether the access is strided and get the stride value
  // if it's known in compile time
  const SCEV *PtrSCEV = getAddressAccessSCEV(Ptr, Legal, PSE, TheLoop);

  // Get the cost of the scalar memory instruction and address computation.
  InstructionCost Cost =
      VF.getKnownMinValue() * TTI.getAddressComputationCost(PtrTy, SE, PtrSCEV);

  // Don't pass *I here, since it is scalar but will actually be part of a
  // vectorized loop where the user of it is a vectorized instruction.
  const Align Alignment = getLoadStoreAlignment(I);
  Cost += VF.getKnownMinValue() *
          TTI.getMemoryOpCost(I->getOpcode(), ValTy->getScalarType(), Alignment,
                              AS, TTI::TCK_RecipThroughput);

  // Get the overhead of the extractelement and insertelement instructions
  // we might create due to scalarization.
  Cost += getScalarizationOverhead(I, VF);

  // If we have a predicated load/store, it will need extra i1 extracts and
  // conditional branches, but may not be executed for each vector lane. Scale
  // the cost by the probability of executing the predicated block.
  if (isPredicatedInst(I)) {
    Cost /= getReciprocalPredBlockProb();

    // Add the cost of an i1 extract and a branch
    auto *Vec_i1Ty =
        VectorType::get(IntegerType::getInt1Ty(ValTy->getContext()), VF);
    Cost += TTI.getScalarizationOverhead(
        Vec_i1Ty, APInt::getAllOnesValue(VF.getKnownMinValue()),
        /*Insert=*/false, /*Extract=*/true);
    Cost += TTI.getCFInstrCost(Instruction::Br, TTI::TCK_RecipThroughput);

    if (useEmulatedMaskMemRefHack(I))
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
  auto *VectorTy = cast<VectorType>(ToVectorTy(ValTy, VF));
  Value *Ptr = getLoadStorePointerOperand(I);
  unsigned AS = getLoadStoreAddressSpace(I);
  int ConsecutiveStride = Legal->isConsecutivePtr(Ptr);
  enum TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;

  assert((ConsecutiveStride == 1 || ConsecutiveStride == -1) &&
         "Stride should be 1 or -1 for consecutive memory access");
  const Align Alignment = getLoadStoreAlignment(I);
  InstructionCost Cost = 0;
  if (Legal->isMaskRequired(I))
    Cost += TTI.getMaskedMemoryOpCost(I->getOpcode(), VectorTy, Alignment, AS,
                                      CostKind);
  else
    Cost += TTI.getMemoryOpCost(I->getOpcode(), VectorTy, Alignment, AS,
                                CostKind, I);

  bool Reverse = ConsecutiveStride < 0;
  if (Reverse)
    Cost +=
        TTI.getShuffleCost(TargetTransformInfo::SK_Reverse, VectorTy, None, 0);
  return Cost;
}

InstructionCost
LoopVectorizationCostModel::getUniformMemOpCost(Instruction *I,
                                                ElementCount VF) {
  assert(Legal->isUniformMemOp(*I));

  Type *ValTy = getLoadStoreType(I);
  auto *VectorTy = cast<VectorType>(ToVectorTy(ValTy, VF));
  const Align Alignment = getLoadStoreAlignment(I);
  unsigned AS = getLoadStoreAddressSpace(I);
  enum TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
  if (isa<LoadInst>(I)) {
    return TTI.getAddressComputationCost(ValTy) +
           TTI.getMemoryOpCost(Instruction::Load, ValTy, Alignment, AS,
                               CostKind) +
           TTI.getShuffleCost(TargetTransformInfo::SK_Broadcast, VectorTy);
  }
  StoreInst *SI = cast<StoreInst>(I);

  bool isLoopInvariantStoreValue = Legal->isUniform(SI->getValueOperand());
  return TTI.getAddressComputationCost(ValTy) +
         TTI.getMemoryOpCost(Instruction::Store, ValTy, Alignment, AS,
                             CostKind) +
         (isLoopInvariantStoreValue
              ? 0
              : TTI.getVectorInstrCost(Instruction::ExtractElement, VectorTy,
                                       VF.getKnownMinValue() - 1));
}

InstructionCost
LoopVectorizationCostModel::getGatherScatterCost(Instruction *I,
                                                 ElementCount VF) {
  Type *ValTy = getLoadStoreType(I);
  auto *VectorTy = cast<VectorType>(ToVectorTy(ValTy, VF));
  const Align Alignment = getLoadStoreAlignment(I);
  const Value *Ptr = getLoadStorePointerOperand(I);

  return TTI.getAddressComputationCost(VectorTy) +
         TTI.getGatherScatterOpCost(
             I->getOpcode(), VectorTy, Ptr, Legal->isMaskRequired(I), Alignment,
             TargetTransformInfo::TCK_RecipThroughput, I);
}

InstructionCost
LoopVectorizationCostModel::getInterleaveGroupCost(Instruction *I,
                                                   ElementCount VF) {
  // TODO: Once we have support for interleaving with scalable vectors
  // we can calculate the cost properly here.
  if (VF.isScalable())
    return InstructionCost::getInvalid();

  Type *ValTy = getLoadStoreType(I);
  auto *VectorTy = cast<VectorType>(ToVectorTy(ValTy, VF));
  unsigned AS = getLoadStoreAddressSpace(I);

  auto Group = getInterleavedAccessGroup(I);
  assert(Group && "Fail to get an interleaved access group.");

  unsigned InterleaveFactor = Group->getFactor();
  auto *WideVecTy = VectorType::get(ValTy, VF * InterleaveFactor);

  // Holds the indices of existing members in an interleaved load group.
  // An interleaved store group doesn't need this as it doesn't allow gaps.
  SmallVector<unsigned, 4> Indices;
  if (isa<LoadInst>(I)) {
    for (unsigned i = 0; i < InterleaveFactor; i++)
      if (Group->getMember(i))
        Indices.push_back(i);
  }

  // Calculate the cost of the whole interleaved group.
  bool UseMaskForGaps =
      Group->requiresScalarEpilogue() && !isScalarEpilogueAllowed();
  InstructionCost Cost = TTI.getInterleavedMemoryOpCost(
      I->getOpcode(), WideVecTy, Group->getFactor(), Indices, Group->getAlign(),
      AS, TTI::TCK_RecipThroughput, Legal->isMaskRequired(I), UseMaskForGaps);

  if (Group->isReverse()) {
    // TODO: Add support for reversed masked interleaved access.
    assert(!Legal->isMaskRequired(I) &&
           "Reverse masked interleaved access not supported.");
    Cost +=
        Group->getNumMembers() *
        TTI.getShuffleCost(TargetTransformInfo::SK_Reverse, VectorTy, None, 0);
  }
  return Cost;
}

InstructionCost LoopVectorizationCostModel::getReductionPatternCost(
    Instruction *I, ElementCount VF, Type *Ty, TTI::TargetCostKind CostKind) {
  // Early exit for no inloop reductions
  if (InLoopReductionChains.empty() || VF.isScalar() || !isa<VectorType>(Ty))
    return InstructionCost::getInvalid();
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
  if ((RetI->getOpcode() == Instruction::SExt ||
       RetI->getOpcode() == Instruction::ZExt)) {
    if (!RetI->hasOneUser())
      return InstructionCost::getInvalid();
    RetI = RetI->user_back();
  }
  if (RetI->getOpcode() == Instruction::Mul &&
      RetI->user_back()->getOpcode() == Instruction::Add) {
    if (!RetI->hasOneUser())
      return InstructionCost::getInvalid();
    RetI = RetI->user_back();
  }

  // Test if the found instruction is a reduction, and if not return an invalid
  // cost specifying the parent to use the original cost modelling.
  if (!InLoopReductionImmediateChains.count(RetI))
    return InstructionCost::getInvalid();

  // Find the reduction this chain is a part of and calculate the basic cost of
  // the reduction on its own.
  Instruction *LastChain = InLoopReductionImmediateChains[RetI];
  Instruction *ReductionPhi = LastChain;
  while (!isa<PHINode>(ReductionPhi))
    ReductionPhi = InLoopReductionImmediateChains[ReductionPhi];

  const RecurrenceDescriptor &RdxDesc =
      Legal->getReductionVars()[cast<PHINode>(ReductionPhi)];
  InstructionCost BaseCost = TTI.getArithmeticReductionCost(
      RdxDesc.getOpcode(), VectorTy, false, CostKind);

  // Get the operand that was not the reduction chain and match it to one of the
  // patterns, returning the better cost if it is found.
  Instruction *RedOp = RetI->getOperand(1) == LastChain
                           ? dyn_cast<Instruction>(RetI->getOperand(0))
                           : dyn_cast<Instruction>(RetI->getOperand(1));

  VectorTy = VectorType::get(I->getOperand(0)->getType(), VectorTy);

  if (RedOp && (isa<SExtInst>(RedOp) || isa<ZExtInst>(RedOp)) &&
      !TheLoop->isLoopInvariant(RedOp)) {
    bool IsUnsigned = isa<ZExtInst>(RedOp);
    auto *ExtType = VectorType::get(RedOp->getOperand(0)->getType(), VectorTy);
    InstructionCost RedCost = TTI.getExtendedAddReductionCost(
        /*IsMLA=*/false, IsUnsigned, RdxDesc.getRecurrenceType(), ExtType,
        CostKind);

    InstructionCost ExtCost =
        TTI.getCastInstrCost(RedOp->getOpcode(), VectorTy, ExtType,
                             TTI::CastContextHint::None, CostKind, RedOp);
    if (RedCost.isValid() && RedCost < BaseCost + ExtCost)
      return I == RetI ? *RedCost.getValue() : 0;
  } else if (RedOp && RedOp->getOpcode() == Instruction::Mul) {
    Instruction *Mul = RedOp;
    Instruction *Op0 = dyn_cast<Instruction>(Mul->getOperand(0));
    Instruction *Op1 = dyn_cast<Instruction>(Mul->getOperand(1));
    if (Op0 && Op1 && (isa<SExtInst>(Op0) || isa<ZExtInst>(Op0)) &&
        Op0->getOpcode() == Op1->getOpcode() &&
        Op0->getOperand(0)->getType() == Op1->getOperand(0)->getType() &&
        !TheLoop->isLoopInvariant(Op0) && !TheLoop->isLoopInvariant(Op1)) {
      bool IsUnsigned = isa<ZExtInst>(Op0);
      auto *ExtType = VectorType::get(Op0->getOperand(0)->getType(), VectorTy);
      // reduce(mul(ext, ext))
      InstructionCost ExtCost =
          TTI.getCastInstrCost(Op0->getOpcode(), VectorTy, ExtType,
                               TTI::CastContextHint::None, CostKind, Op0);
      InstructionCost MulCost =
          TTI.getArithmeticInstrCost(Mul->getOpcode(), VectorTy, CostKind);

      InstructionCost RedCost = TTI.getExtendedAddReductionCost(
          /*IsMLA=*/true, IsUnsigned, RdxDesc.getRecurrenceType(), ExtType,
          CostKind);

      if (RedCost.isValid() && RedCost < ExtCost * 2 + MulCost + BaseCost)
        return I == RetI ? *RedCost.getValue() : 0;
    } else {
      InstructionCost MulCost =
          TTI.getArithmeticInstrCost(Mul->getOpcode(), VectorTy, CostKind);

      InstructionCost RedCost = TTI.getExtendedAddReductionCost(
          /*IsMLA=*/true, true, RdxDesc.getRecurrenceType(), VectorTy,
          CostKind);

      if (RedCost.isValid() && RedCost < MulCost + BaseCost)
        return I == RetI ? *RedCost.getValue() : 0;
    }
  }

  return I == RetI ? BaseCost : InstructionCost::getInvalid();
}

InstructionCost
LoopVectorizationCostModel::getMemoryInstructionCost(Instruction *I,
                                                     ElementCount VF) {
  // Calculate scalar cost only. Vectorization cost should be ready at this
  // moment.
  if (VF.isScalar()) {
    Type *ValTy = getLoadStoreType(I);
    const Align Alignment = getLoadStoreAlignment(I);
    unsigned AS = getLoadStoreAddressSpace(I);

    return TTI.getAddressComputationCost(ValTy) +
           TTI.getMemoryOpCost(I->getOpcode(), ValTy, Alignment, AS,
                               TTI::TCK_RecipThroughput, I);
  }
  return getWideningCost(I, VF);
}

LoopVectorizationCostModel::VectorizationCostTy
LoopVectorizationCostModel::getInstructionCost(Instruction *I,
                                               ElementCount VF) {
  // If we know that this instruction will remain uniform, check the cost of
  // the scalar version.
  if (isUniformAfterVectorization(I, VF))
    VF = ElementCount::getFixed(1);

  if (VF.isVector() && isProfitableToScalarize(I, VF))
    return VectorizationCostTy(InstsToScalarize[VF][I], false);

  // Forced scalars do not have any scalarization overhead.
  auto ForcedScalar = ForcedScalars.find(VF);
  if (VF.isVector() && ForcedScalar != ForcedScalars.end()) {
    auto InstSet = ForcedScalar->second;
    if (InstSet.count(I))
      return VectorizationCostTy(
          (getInstructionCost(I, ElementCount::getFixed(1)).first *
           VF.getKnownMinValue()),
          false);
  }

  Type *VectorTy;
  InstructionCost C = getInstructionCost(I, VF, VectorTy);

  bool TypeNotScalarized =
      VF.isVector() && VectorTy->isVectorTy() &&
      TTI.getNumberOfParts(VectorTy) < VF.getKnownMinValue();
  return VectorizationCostTy(C, TypeNotScalarized);
}

InstructionCost
LoopVectorizationCostModel::getScalarizationOverhead(Instruction *I,
                                                     ElementCount VF) const {

  if (VF.isScalable())
    return InstructionCost::getInvalid();

  if (VF.isScalar())
    return 0;

  InstructionCost Cost = 0;
  Type *RetTy = ToVectorTy(I->getType(), VF);
  if (!RetTy->isVoidTy() &&
      (!isa<LoadInst>(I) || !TTI.supportsEfficientVectorElementLoadStore()))
    Cost += TTI.getScalarizationOverhead(
        cast<VectorType>(RetTy), APInt::getAllOnesValue(VF.getKnownMinValue()),
        true, false);

  // Some targets keep addresses scalar.
  if (isa<LoadInst>(I) && !TTI.prefersVectorizedAddressing())
    return Cost;

  // Some targets support efficient element stores.
  if (isa<StoreInst>(I) && TTI.supportsEfficientVectorElementLoadStore())
    return Cost;

  // Collect operands to consider.
  CallInst *CI = dyn_cast<CallInst>(I);
  Instruction::op_range Ops = CI ? CI->arg_operands() : I->operands();

  // Skip operands that do not require extraction/scalarization and do not incur
  // any overhead.
  SmallVector<Type *> Tys;
  for (auto *V : filterExtractingOperands(Ops, VF))
    Tys.push_back(MaybeVectorizeType(V->getType(), VF));
  return Cost + TTI.getOperandsScalarizationOverhead(
                    filterExtractingOperands(Ops, VF), Tys);
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
      if (isa<StoreInst>(&I) && isScalarWithPredication(&I))
        NumPredStores++;

      if (Legal->isUniformMemOp(I)) {
        // TODO: Avoid replicating loads and stores instead of
        // relying on instcombine to remove them.
        // Load: Scalar load + broadcast
        // Store: Scalar store + isLoopInvariantStoreValue ? 0 : extract
        InstructionCost Cost;
        if (isa<StoreInst>(&I) && VF.isScalable() &&
            isLegalGatherOrScatter(&I)) {
          Cost = getGatherScatterCost(&I, VF);
          setWideningDecision(&I, VF, CM_GatherScatter, Cost);
        } else {
          assert((isa<LoadInst>(&I) || !VF.isScalable()) &&
                 "Cannot yet scalarize uniform stores");
          Cost = getUniformMemOpCost(&I, VF);
          setWideningDecision(&I, VF, CM_Scalarize, Cost);
        }
        continue;
      }

      // We assume that widening is the best solution when possible.
      if (memoryInstructionCanBeWidened(&I, VF)) {
        InstructionCost Cost = getConsecutiveMemOpCost(&I, VF);
        int ConsecutiveStride =
               Legal->isConsecutivePtr(getLoadStorePointerOperand(&I));
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
        auto Group = getInterleavedAccessGroup(&I);
        assert(Group && "Fail to get an interleaved access group.");

        // Make one decision for the whole group.
        if (getWideningDecision(&I, VF) != CM_Unknown)
          continue;

        NumAccesses = Group->getNumMembers();
        if (interleavedAccessCanBeWidened(&I, VF))
          InterleaveCost = getInterleaveGroupCost(&I, VF);
      }

      InstructionCost GatherScatterCost =
          isLegalGatherOrScatter(&I)
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
        assert(!VF.isScalable() &&
               "We cannot yet scalarise for scalable vectors");
        Decision = CM_Scalarize;
        Cost = ScalarizationCost;
      }
      // If the instructions belongs to an interleave group, the whole group
      // receives the same decision. The whole group receives the cost, but
      // the cost will actually be assigned to one instruction.
      if (auto Group = getInterleavedAccessGroup(&I))
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
      else if (auto Group = getInterleavedAccessGroup(I)) {
        // Scalarize an interleave group of address loads.
        for (unsigned I = 0; I < Group->getFactor(); ++I) {
          if (Instruction *Member = Group->getMember(I))
            setWideningDecision(
                Member, VF, CM_Scalarize,
                (VF.getKnownMinValue() *
                 getMemoryInstructionCost(Member, ElementCount::getFixed(1))));
        }
      }
    } else
      // Make sure I gets scalarized and a cost estimate without
      // scalarization overhead.
      ForcedScalars[VF].insert(I);
  }
}

InstructionCost
LoopVectorizationCostModel::getInstructionCost(Instruction *I, ElementCount VF,
                                               Type *&VectorTy) {
  Type *RetTy = I->getType();
  if (canTruncateToMinimalBitwidth(I, VF))
    RetTy = IntegerType::get(RetTy->getContext(), MinBWs[I]);
  auto SE = PSE.getSE();
  TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;

  auto hasSingleCopyAfterVectorization = [this](Instruction *I,
                                                ElementCount VF) -> bool {
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
  (void) hasSingleCopyAfterVectorization;

  if (isScalarAfterVectorization(I, VF)) {
    // With the exception of GEPs and PHIs, after scalarization there should
    // only be one copy of the instruction generated in the loop. This is
    // because the VF is either 1, or any instructions that need scalarizing
    // have already been dealt with by the the time we get here. As a result,
    // it means we don't have to multiply the instruction cost by VF.
    assert(I->getOpcode() == Instruction::GetElementPtr ||
           I->getOpcode() == Instruction::PHI ||
           (I->getOpcode() == Instruction::BitCast &&
            I->getType()->isPointerTy()) ||
           hasSingleCopyAfterVectorization(I, VF));
    VectorTy = RetTy;
  } else
    VectorTy = ToVectorTy(RetTy, VF);

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
    bool ScalarPredicatedBB = false;
    BranchInst *BI = cast<BranchInst>(I);
    if (VF.isVector() && BI->isConditional() &&
        (PredicatedBBsAfterVectorization.count(BI->getSuccessor(0)) ||
         PredicatedBBsAfterVectorization.count(BI->getSuccessor(1))))
      ScalarPredicatedBB = true;

    if (ScalarPredicatedBB) {
      // Return cost for branches around scalarized and predicated blocks.
      assert(!VF.isScalable() && "scalable vectors not yet supported.");
      auto *Vec_i1Ty =
          VectorType::get(IntegerType::getInt1Ty(RetTy->getContext()), VF);
      return (TTI.getScalarizationOverhead(
                  Vec_i1Ty, APInt::getAllOnesValue(VF.getKnownMinValue()),
                  false, true) +
              (TTI.getCFInstrCost(Instruction::Br, CostKind) *
               VF.getKnownMinValue()));
    } else if (I->getParent() == TheLoop->getLoopLatch() || VF.isScalar())
      // The back-edge branch will remain, as will all scalar branches.
      return TTI.getCFInstrCost(Instruction::Br, CostKind);
    else
      // This branch will be eliminated by if-conversion.
      return 0;
    // Note: We currently assume zero cost for an unconditional branch inside
    // a predicated block since it will become a fall-through, although we
    // may decide in the future to call TTI for all branches.
  }
  case Instruction::PHI: {
    auto *Phi = cast<PHINode>(I);

    // First-order recurrences are replaced by vector shuffles inside the loop.
    // NOTE: Don't use ToVectorTy as SK_ExtractSubvector expects a vector type.
    if (VF.isVector() && Legal->isFirstOrderRecurrence(Phi))
      return TTI.getShuffleCost(
          TargetTransformInfo::SK_ExtractSubvector, cast<VectorType>(VectorTy),
          None, VF.getKnownMinValue() - 1, FixedVectorType::get(RetTy, 1));

    // Phi nodes in non-header blocks (not inductions, reductions, etc.) are
    // converted into select instructions. We require N - 1 selects per phi
    // node, where N is the number of incoming values.
    if (VF.isVector() && Phi->getParent() != TheLoop->getHeader())
      return (Phi->getNumIncomingValues() - 1) *
             TTI.getCmpSelInstrCost(
                 Instruction::Select, ToVectorTy(Phi->getType(), VF),
                 ToVectorTy(Type::getInt1Ty(Phi->getContext()), VF),
                 CmpInst::BAD_ICMP_PREDICATE, CostKind);

    return TTI.getCFInstrCost(Instruction::PHI, CostKind);
  }
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
    // If we have a predicated instruction, it may not be executed for each
    // vector lane. Get the scalarization cost and scale this amount by the
    // probability of executing the predicated block. If the instruction is not
    // predicated, we fall through to the next case.
    if (VF.isVector() && isScalarWithPredication(I)) {
      InstructionCost Cost = 0;

      // These instructions have a non-void type, so account for the phi nodes
      // that we will create. This cost is likely to be zero. The phi node
      // cost, if any, should be scaled by the block probability because it
      // models a copy at the end of each predicated block.
      Cost += VF.getKnownMinValue() *
              TTI.getCFInstrCost(Instruction::PHI, CostKind);

      // The cost of the non-predicated instruction.
      Cost += VF.getKnownMinValue() *
              TTI.getArithmeticInstrCost(I->getOpcode(), RetTy, CostKind);

      // The cost of insertelement and extractelement instructions needed for
      // scalarization.
      Cost += getScalarizationOverhead(I, VF);

      // Scale the cost by the probability of executing the predicated blocks.
      // This assumes the predicated block for each vector lane is equally
      // likely.
      return Cost / getReciprocalPredBlockProb();
    }
    LLVM_FALLTHROUGH;
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
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
    // Since we will replace the stride by 1 the multiplication should go away.
    if (I->getOpcode() == Instruction::Mul && isStrideMul(I, Legal))
      return 0;

    // Detect reduction patterns
    InstructionCost RedCost;
    if ((RedCost = getReductionPatternCost(I, VF, VectorTy, CostKind))
            .isValid())
      return RedCost;

    // Certain instructions can be cheaper to vectorize if they have a constant
    // second vector operand. One example of this are shifts on x86.
    Value *Op2 = I->getOperand(1);
    TargetTransformInfo::OperandValueProperties Op2VP;
    TargetTransformInfo::OperandValueKind Op2VK =
        TTI.getOperandInfo(Op2, Op2VP);
    if (Op2VK == TargetTransformInfo::OK_AnyValue && Legal->isUniform(Op2))
      Op2VK = TargetTransformInfo::OK_UniformValue;

    SmallVector<const Value *, 4> Operands(I->operand_values());
    return TTI.getArithmeticInstrCost(
        I->getOpcode(), VectorTy, CostKind, TargetTransformInfo::OK_AnyValue,
        Op2VK, TargetTransformInfo::OP_None, Op2VP, Operands, I);
  }
  case Instruction::FNeg: {
    return TTI.getArithmeticInstrCost(
        I->getOpcode(), VectorTy, CostKind, TargetTransformInfo::OK_AnyValue,
        TargetTransformInfo::OK_AnyValue, TargetTransformInfo::OP_None,
        TargetTransformInfo::OP_None, I->getOperand(0), I);
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
      TTI::OperandValueProperties Op1VP = TTI::OP_None;
      TTI::OperandValueProperties Op2VP = TTI::OP_None;
      TTI::OperandValueKind Op1VK = TTI::getOperandInfo(Op0, Op1VP);
      TTI::OperandValueKind Op2VK = TTI::getOperandInfo(Op1, Op2VP);
      assert(Op0->getType()->getScalarSizeInBits() == 1 &&
              Op1->getType()->getScalarSizeInBits() == 1);

      SmallVector<const Value *, 2> Operands{Op0, Op1};
      return TTI.getArithmeticInstrCost(
          match(I, m_LogicalOr()) ? Instruction::Or : Instruction::And, VectorTy,
          CostKind, Op1VK, Op2VK, Op1VP, Op2VP, Operands, I);
    }

    Type *CondTy = SI->getCondition()->getType();
    if (!ScalarCond)
      CondTy = VectorType::get(CondTy, VF);
    return TTI.getCmpSelInstrCost(I->getOpcode(), VectorTy, CondTy,
                                  CmpInst::BAD_ICMP_PREDICATE, CostKind, I);
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    Type *ValTy = I->getOperand(0)->getType();
    Instruction *Op0AsInstruction = dyn_cast<Instruction>(I->getOperand(0));
    if (canTruncateToMinimalBitwidth(Op0AsInstruction, VF))
      ValTy = IntegerType::get(ValTy->getContext(), MinBWs[Op0AsInstruction]);
    VectorTy = ToVectorTy(ValTy, VF);
    return TTI.getCmpSelInstrCost(I->getOpcode(), VectorTy, nullptr,
                                  CmpInst::BAD_ICMP_PREDICATE, CostKind, I);
  }
  case Instruction::Store:
  case Instruction::Load: {
    ElementCount Width = VF;
    if (Width.isVector()) {
      InstWidening Decision = getWideningDecision(I, Width);
      assert(Decision != CM_Unknown &&
             "CM decision should be taken at this point");
      if (Decision == CM_Scalarize)
        Width = ElementCount::getFixed(1);
    }
    VectorTy = ToVectorTy(getLoadStoreType(I), Width);
    return getMemoryInstructionCost(I, VF);
  }
  case Instruction::BitCast:
    if (I->getType()->isPointerTy())
      return 0;
    LLVM_FALLTHROUGH;
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
        return Legal->isMaskRequired(I) ? TTI::CastContextHint::Masked
                                        : TTI::CastContextHint::Normal;
      case LoopVectorizationCostModel::CM_Widen_Reverse:
        return TTI::CastContextHint::Reversed;
      case LoopVectorizationCostModel::CM_Unknown:
        llvm_unreachable("Instr did not go through cost modelling?");
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
    InstructionCost RedCost;
    if ((RedCost = getReductionPatternCost(I, VF, VectorTy, CostKind))
            .isValid())
      return RedCost;

    Type *SrcScalarTy = I->getOperand(0)->getType();
    Type *SrcVecTy =
        VectorTy->isVectorTy() ? ToVectorTy(SrcScalarTy, VF) : SrcScalarTy;
    if (canTruncateToMinimalBitwidth(I, VF)) {
      // This cast is going to be shrunk. This may remove the cast or it might
      // turn it into slightly different cast. For example, if MinBW == 16,
      // "zext i8 %1 to i32" becomes "zext i8 %1 to i16".
      //
      // Calculate the modified src and dest types.
      Type *MinVecTy = VectorTy;
      if (Opcode == Instruction::Trunc) {
        SrcVecTy = smallestIntegerVectorType(SrcVecTy, MinVecTy);
        VectorTy =
            largestIntegerVectorType(ToVectorTy(I->getType(), VF), MinVecTy);
      } else if (Opcode == Instruction::ZExt || Opcode == Instruction::SExt) {
        SrcVecTy = largestIntegerVectorType(SrcVecTy, MinVecTy);
        VectorTy =
            smallestIntegerVectorType(ToVectorTy(I->getType(), VF), MinVecTy);
      }
    }

    return TTI.getCastInstrCost(Opcode, VectorTy, SrcVecTy, CCH, CostKind, I);
  }
  case Instruction::Call: {
    bool NeedToScalarize;
    CallInst *CI = cast<CallInst>(I);
    InstructionCost CallCost = getVectorCallCost(CI, VF, NeedToScalarize);
    if (getVectorIntrinsicIDForCall(CI, TLI)) {
      InstructionCost IntrinsicCost = getVectorIntrinsicCost(CI, VF);
      return std::min(CallCost, IntrinsicCost);
    }
    return CallCost;
  }
  case Instruction::ExtractValue:
    return TTI.getInstructionCost(I, TTI::TCK_RecipThroughput);
  default:
    // This opcode is unknown. Assume that it is the same as 'mul'.
    return TTI.getArithmeticInstrCost(Instruction::Mul, VectorTy, CostKind);
  } // end of switch.
}

char LoopVectorize::ID = 0;

static const char lv_name[] = "Loop Vectorization";

INITIALIZE_PASS_BEGIN(LoopVectorize, LV_NAME, lv_name, false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(BasicAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(BlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopAccessLegacyAnalysis)
INITIALIZE_PASS_DEPENDENCY(DemandedBitsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(InjectTLIMappingsLegacy)
INITIALIZE_PASS_END(LoopVectorize, LV_NAME, lv_name, false, false)

namespace llvm {

Pass *createLoopVectorizePass() { return new LoopVectorize(); }

Pass *createLoopVectorizePass(bool InterleaveOnlyWhenForced,
                              bool VectorizeOnlyWhenForced) {
  return new LoopVectorize(InterleaveOnlyWhenForced, VectorizeOnlyWhenForced);
}

} // end namespace llvm

bool LoopVectorizationCostModel::isConsecutiveLoadOrStore(Instruction *Inst) {
  // Check if the pointer operand of a load or store instruction is
  // consecutive.
  if (auto *Ptr = getLoadStorePointerOperand(Inst))
    return Legal->isConsecutivePtr(Ptr);
  return false;
}

void LoopVectorizationCostModel::collectValuesToIgnore() {
  // Ignore ephemeral values.
  CodeMetrics::collectEphemeralValues(TheLoop, AC, ValuesToIgnore);

  // Ignore type-promoting instructions we identified during reduction
  // detection.
  for (auto &Reduction : Legal->getReductionVars()) {
    RecurrenceDescriptor &RedDes = Reduction.second;
    const SmallPtrSetImpl<Instruction *> &Casts = RedDes.getCastInsts();
    VecValuesToIgnore.insert(Casts.begin(), Casts.end());
  }
  // Ignore type-casting instructions we identified during induction
  // detection.
  for (auto &Induction : Legal->getInductionVars()) {
    InductionDescriptor &IndDes = Induction.second;
    const SmallVectorImpl<Instruction *> &Casts = IndDes.getCastInsts();
    VecValuesToIgnore.insert(Casts.begin(), Casts.end());
  }
}

void LoopVectorizationCostModel::collectInLoopReductions() {
  for (auto &Reduction : Legal->getReductionVars()) {
    PHINode *Phi = Reduction.first;
    RecurrenceDescriptor &RdxDesc = Reduction.second;

    // We don't collect reductions that are type promoted (yet).
    if (RdxDesc.getRecurrenceType() != Phi->getType())
      continue;

    // If the target would prefer this reduction to happen "in-loop", then we
    // want to record it as such.
    unsigned Opcode = RdxDesc.getOpcode();
    if (!PreferInLoopReductions && !useOrderedReductions(RdxDesc) &&
        !TTI.preferInLoopReduction(Opcode, Phi->getType(),
                                   TargetTransformInfo::ReductionFlags()))
      continue;

    // Check that we can correctly put the reductions into the loop, by
    // finding the chain of operations that leads from the phi to the loop
    // exit value.
    SmallVector<Instruction *, 4> ReductionOperations =
        RdxDesc.getReductionOpChain(Phi, TheLoop);
    bool InLoop = !ReductionOperations.empty();
    if (InLoop) {
      InLoopReductionChains[Phi] = ReductionOperations;
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

// TODO: we could return a pair of values that specify the max VF and
// min VF, to be used in `buildVPlans(MinVF, MaxVF)` instead of
// `buildVPlans(VF, VF)`. We cannot do it because VPLAN at the moment
// doesn't have a cost model that can choose which plan to execute if
// more than one is generated.
static unsigned determineVPlanVF(const unsigned WidestVectorRegBits,
                                 LoopVectorizationCostModel &CM) {
  unsigned WidestType;
  std::tie(std::ignore, WidestType) = CM.getSmallestAndWidestTypes();
  return WidestVectorRegBits / WidestType;
}

VectorizationFactor
LoopVectorizationPlanner::planInVPlanNativePath(ElementCount UserVF) {
  assert(!UserVF.isScalable() && "scalable vectors not yet supported");
  ElementCount VF = UserVF;
  // Outer loop handling: They may require CFG and instruction level
  // transformations before even evaluating whether vectorization is profitable.
  // Since we cannot modify the incoming IR, we need to build VPlan upfront in
  // the vectorization pipeline.
  if (!OrigLoop->isInnermost()) {
    // If the user doesn't provide a vectorization factor, determine a
    // reasonable one.
    if (UserVF.isZero()) {
      VF = ElementCount::getFixed(determineVPlanVF(
          TTI->getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector)
              .getFixedSize(),
          CM));
      LLVM_DEBUG(dbgs() << "LV: VPlan computed VF " << VF << ".\n");

      // Make sure we have a VF > 1 for stress testing.
      if (VPlanBuildStressTest && (VF.isScalar() || VF.isZero())) {
        LLVM_DEBUG(dbgs() << "LV: VPlan stress testing: "
                          << "overriding computed VF.\n");
        VF = ElementCount::getFixed(4);
      }
    }
    assert(EnableVPlanNativePath && "VPlan-native path is not enabled.");
    assert(isPowerOf2_32(VF.getKnownMinValue()) &&
           "VF needs to be a power of two");
    LLVM_DEBUG(dbgs() << "LV: Using " << (!UserVF.isZero() ? "user " : "")
                      << "VF " << VF << " to build VPlans.\n");
    buildVPlans(VF, VF);

    // For VPlan build stress testing, we bail out after VPlan construction.
    if (VPlanBuildStressTest)
      return VectorizationFactor::Disabled();

    return {VF, 0 /*Cost*/};
  }

  LLVM_DEBUG(
      dbgs() << "LV: Not vectorizing. Inner loops aren't supported in the "
                "VPlan-native path.\n");
  return VectorizationFactor::Disabled();
}

Optional<VectorizationFactor>
LoopVectorizationPlanner::plan(ElementCount UserVF, unsigned UserIC) {
  assert(OrigLoop->isInnermost() && "Inner loop expected.");
  FixedScalableVFPair MaxFactors = CM.computeMaxVF(UserVF, UserIC);
  if (!MaxFactors) // Cases that should not to be vectorized nor interleaved.
    return None;

  // Invalidate interleave groups if all blocks of loop will be predicated.
  if (CM.blockNeedsPredication(OrigLoop->getHeader()) &&
      !useMaskedInterleavedAccesses(*TTI)) {
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

  ElementCount MaxUserVF =
      UserVF.isScalable() ? MaxFactors.ScalableVF : MaxFactors.FixedVF;
  bool UserVFIsLegal = ElementCount::isKnownLE(UserVF, MaxUserVF);
  if (!UserVF.isZero() && UserVFIsLegal) {
    LLVM_DEBUG(dbgs() << "LV: Using " << (UserVFIsLegal ? "user" : "max")
                      << " VF " << UserVF << ".\n");
    assert(isPowerOf2_32(UserVF.getKnownMinValue()) &&
           "VF needs to be a power of two");
    // Collect the instructions (and their associated costs) that will be more
    // profitable to scalarize.
    CM.selectUserVectorizationFactor(UserVF);
    CM.collectInLoopReductions();
    buildVPlansWithVPRecipes(UserVF, UserVF);
    LLVM_DEBUG(printPlans(dbgs()));
    return {{UserVF, 0}};
  }

  // Populate the set of Vectorization Factor Candidates.
  ElementCountSet VFCandidates;
  for (auto VF = ElementCount::getFixed(1);
       ElementCount::isKnownLE(VF, MaxFactors.FixedVF); VF *= 2)
    VFCandidates.insert(VF);
  for (auto VF = ElementCount::getScalable(1);
       ElementCount::isKnownLE(VF, MaxFactors.ScalableVF); VF *= 2)
    VFCandidates.insert(VF);

  for (const auto &VF : VFCandidates) {
    // Collect Uniform and Scalar instructions after vectorization with VF.
    CM.collectUniformsAndScalars(VF);

    // Collect the instructions (and their associated costs) that will be more
    // profitable to scalarize.
    if (VF.isVector())
      CM.collectInstsToScalarize(VF);
  }

  CM.collectInLoopReductions();
  buildVPlansWithVPRecipes(ElementCount::getFixed(1), MaxFactors.FixedVF);
  buildVPlansWithVPRecipes(ElementCount::getScalable(1), MaxFactors.ScalableVF);

  LLVM_DEBUG(printPlans(dbgs()));
  if (!MaxFactors.hasVector())
    return VectorizationFactor::Disabled();

  // Select the optimal vectorization factor.
  auto SelectedVF = CM.selectVectorizationFactor(VFCandidates);

  // Check if it is profitable to vectorize with runtime checks.
  unsigned NumRuntimePointerChecks = Requirements.getNumRuntimePointerChecks();
  if (SelectedVF.Width.getKnownMinValue() > 1 && NumRuntimePointerChecks) {
    bool PragmaThresholdReached =
        NumRuntimePointerChecks > PragmaVectorizeMemoryCheckThreshold;
    bool ThresholdReached =
        NumRuntimePointerChecks > VectorizerParams::RuntimeMemoryCheckThreshold;
    if ((ThresholdReached && !Hints.allowReordering()) ||
        PragmaThresholdReached) {
      ORE->emit([&]() {
        return OptimizationRemarkAnalysisAliasing(
                   DEBUG_TYPE, "CantReorderMemOps", OrigLoop->getStartLoc(),
                   OrigLoop->getHeader())
               << "loop not vectorized: cannot prove it is safe to reorder "
                  "memory operations";
      });
      LLVM_DEBUG(dbgs() << "LV: Too many memory checks needed.\n");
      Hints.emitRemarkWithHints();
      return VectorizationFactor::Disabled();
    }
  }
  return SelectedVF;
}

void LoopVectorizationPlanner::setBestPlan(ElementCount VF, unsigned UF) {
  LLVM_DEBUG(dbgs() << "Setting best plan to VF=" << VF << ", UF=" << UF
                    << '\n');
  BestVF = VF;
  BestUF = UF;

  erase_if(VPlans, [VF](const VPlanPtr &Plan) {
    return !Plan->hasVF(VF);
  });
  assert(VPlans.size() == 1 && "Best VF has not a single VPlan.");
}

void LoopVectorizationPlanner::executePlan(InnerLoopVectorizer &ILV,
                                           DominatorTree *DT) {
  // Perform the actual loop transformation.

  // 1. Create a new empty loop. Unlink the old loop and connect the new one.
  assert(BestVF.hasValue() && "Vectorization Factor is missing");
  assert(VPlans.size() == 1 && "Not a single VPlan to execute.");

  VPTransformState State{
      *BestVF, BestUF, LI, DT, ILV.Builder, &ILV, VPlans.front().get()};
  State.CFG.PrevBB = ILV.createVectorizedLoopSkeleton();
  State.TripCount = ILV.getOrCreateTripCount(nullptr);
  State.CanonicalIV = ILV.Induction;

  ILV.printDebugTracesAtStart();

  //===------------------------------------------------===//
  //
  // Notice: any optimization or new instruction that go
  // into the code below should also be implemented in
  // the cost-model.
  //
  //===------------------------------------------------===//

  // 2. Copy and widen instructions from the old loop into the new loop.
  VPlans.front()->execute(&State);

  // 3. Fix the vectorized code: take care of header phi's, live-outs,
  //    predication, updating analyses.
  ILV.fixVectorizedLoop(State);

  ILV.printDebugTracesAtEnd();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void LoopVectorizationPlanner::printPlans(raw_ostream &O) {
  for (const auto &Plan : VPlans)
    if (PrintVPlansInDotFormat)
      Plan->printDOT(O);
    else
      Plan->print(O);
}
#endif

void LoopVectorizationPlanner::collectTriviallyDeadInstructions(
    SmallPtrSetImpl<Instruction *> &DeadInstructions) {

  // We create new control-flow for the vectorized loop, so the original exit
  // conditions will be dead after vectorization if it's only used by the
  // terminator
  SmallVector<BasicBlock*> ExitingBlocks;
  OrigLoop->getExitingBlocks(ExitingBlocks);
  for (auto *BB : ExitingBlocks) {
    auto *Cmp = dyn_cast<Instruction>(BB->getTerminator()->getOperand(0));
    if (!Cmp || !Cmp->hasOneUse())
      continue;

    // TODO: we should introduce a getUniqueExitingBlocks on Loop
    if (!DeadInstructions.insert(Cmp).second)
      continue;

    // The operands of the icmp is often a dead trunc, used by IndUpdate.
    // TODO: can recurse through operands in general
    for (Value *Op : Cmp->operands()) {
      if (isa<TruncInst>(Op) && Op->hasOneUse())
          DeadInstructions.insert(cast<Instruction>(Op));
    }
  }

  // We create new "steps" for induction variable updates to which the original
  // induction variables map. An original update instruction will be dead if
  // all its users except the induction variable are dead.
  auto *Latch = OrigLoop->getLoopLatch();
  for (auto &Induction : Legal->getInductionVars()) {
    PHINode *Ind = Induction.first;
    auto *IndUpdate = cast<Instruction>(Ind->getIncomingValueForBlock(Latch));

    // If the tail is to be folded by masking, the primary induction variable,
    // if exists, isn't dead: it will be used for masking. Don't kill it.
    if (CM.foldTailByMasking() && IndUpdate == Legal->getPrimaryInduction())
      continue;

    if (llvm::all_of(IndUpdate->users(), [&](User *U) -> bool {
          return U == Ind || DeadInstructions.count(cast<Instruction>(U));
        }))
      DeadInstructions.insert(IndUpdate);

    // We record as "Dead" also the type-casting instructions we had identified
    // during induction analysis. We don't need any handling for them in the
    // vectorized loop because we have proven that, under a proper runtime
    // test guarding the vectorized loop, the value of the phi, and the casted
    // value of the phi, are the same. The last instruction in this casting chain
    // will get its scalar/vector/widened def from the scalar/vector/widened def
    // of the respective phi node. Any other casts in the induction def-use chain
    // have no other uses outside the phi update chain, and will be ignored.
    InductionDescriptor &IndDes = Induction.second;
    const SmallVectorImpl<Instruction *> &Casts = IndDes.getCastInsts();
    DeadInstructions.insert(Casts.begin(), Casts.end());
  }
}

Value *InnerLoopUnroller::reverseVector(Value *Vec) { return Vec; }

Value *InnerLoopUnroller::getBroadcastInstrs(Value *V) { return V; }

Value *InnerLoopUnroller::getStepVector(Value *Val, int StartIdx, Value *Step,
                                        Instruction::BinaryOps BinOp) {
  // When unrolling and the VF is 1, we only need to add a simple scalar.
  Type *Ty = Val->getType();
  assert(!Ty->isVectorTy() && "Val must be a scalar");

  if (Ty->isFloatingPointTy()) {
    Constant *C = ConstantFP::get(Ty, (double)StartIdx);

    // Floating-point operations inherit FMF via the builder's flags.
    Value *MulOp = Builder.CreateFMul(C, Step);
    return Builder.CreateBinOp(BinOp, Val, MulOp);
  }
  Constant *C = ConstantInt::get(Ty, StartIdx);
  return Builder.CreateAdd(Val, Builder.CreateMul(C, Step), "induction");
}

static void AddRuntimeUnrollDisableMetaData(Loop *L) {
  SmallVector<Metadata *, 4> MDs;
  // Reserve first location for self reference to the LoopID metadata node.
  MDs.push_back(nullptr);
  bool IsUnrollMetadata = false;
  MDNode *LoopID = L->getLoopID();
  if (LoopID) {
    // First find existing loop unrolling disable metadata.
    for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
      auto *MD = dyn_cast<MDNode>(LoopID->getOperand(i));
      if (MD) {
        const auto *S = dyn_cast<MDString>(MD->getOperand(0));
        IsUnrollMetadata =
            S && S->getString().startswith("llvm.loop.unroll.disable");
      }
      MDs.push_back(LoopID->getOperand(i));
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

//===--------------------------------------------------------------------===//
// EpilogueVectorizerMainLoop
//===--------------------------------------------------------------------===//

/// This function is partially responsible for generating the control flow
/// depicted in https://llvm.org/docs/Vectorizers.html#epilogue-vectorization.
BasicBlock *EpilogueVectorizerMainLoop::createEpilogueVectorizedLoopSkeleton() {
  MDNode *OrigLoopID = OrigLoop->getLoopID();
  Loop *Lp = createVectorLoopSkeleton("");

  // Generate the code to check the minimum iteration count of the vector
  // epilogue (see below).
  EPI.EpilogueIterationCountCheck =
      emitMinimumIterationCountCheck(Lp, LoopScalarPreHeader, true);
  EPI.EpilogueIterationCountCheck->setName("iter.check");

  // Generate the code to check any assumptions that we've made for SCEV
  // expressions.
  EPI.SCEVSafetyCheck = emitSCEVChecks(Lp, LoopScalarPreHeader);

  // Generate the code that checks at runtime if arrays overlap. We put the
  // checks into a separate block to make the more common case of few elements
  // faster.
  EPI.MemSafetyCheck = emitMemRuntimeChecks(Lp, LoopScalarPreHeader);

  // Generate the iteration count check for the main loop, *after* the check
  // for the epilogue loop, so that the path-length is shorter for the case
  // that goes directly through the vector epilogue. The longer-path length for
  // the main loop is compensated for, by the gain from vectorizing the larger
  // trip count. Note: the branch will get updated later on when we vectorize
  // the epilogue.
  EPI.MainLoopIterationCountCheck =
      emitMinimumIterationCountCheck(Lp, LoopScalarPreHeader, false);

  // Generate the induction variable.
  OldInduction = Legal->getPrimaryInduction();
  Type *IdxTy = Legal->getWidestInductionType();
  Value *StartIdx = ConstantInt::get(IdxTy, 0);
  Constant *Step = ConstantInt::get(IdxTy, VF.getKnownMinValue() * UF);
  Value *CountRoundDown = getOrCreateVectorTripCount(Lp);
  EPI.VectorTripCount = CountRoundDown;
  Induction =
      createInductionVariable(Lp, StartIdx, CountRoundDown, Step,
                              getDebugLocFromInstOrOperands(OldInduction));

  // Skip induction resume value creation here because they will be created in
  // the second pass. If we created them here, they wouldn't be used anyway,
  // because the vplan in the second pass still contains the inductions from the
  // original loop.

  return completeLoopSkeleton(Lp, OrigLoopID);
}

void EpilogueVectorizerMainLoop::printDebugTracesAtStart() {
  LLVM_DEBUG({
    dbgs() << "Create Skeleton for epilogue vectorized loop (first pass)\n"
           << "Main Loop VF:" << EPI.MainLoopVF.getKnownMinValue()
           << ", Main Loop UF:" << EPI.MainLoopUF
           << ", Epilogue Loop VF:" << EPI.EpilogueVF.getKnownMinValue()
           << ", Epilogue Loop UF:" << EPI.EpilogueUF << "\n";
  });
}

void EpilogueVectorizerMainLoop::printDebugTracesAtEnd() {
  DEBUG_WITH_TYPE(VerboseDebug, {
    dbgs() << "intermediate fn:\n" << *Induction->getFunction() << "\n";
  });
}

BasicBlock *EpilogueVectorizerMainLoop::emitMinimumIterationCountCheck(
    Loop *L, BasicBlock *Bypass, bool ForEpilogue) {
  assert(L && "Expected valid Loop.");
  assert(Bypass && "Expected valid bypass basic block.");
  unsigned VFactor =
      ForEpilogue ? EPI.EpilogueVF.getKnownMinValue() : VF.getKnownMinValue();
  unsigned UFactor = ForEpilogue ? EPI.EpilogueUF : UF;
  Value *Count = getOrCreateTripCount(L);
  // Reuse existing vector loop preheader for TC checks.
  // Note that new preheader block is generated for vector loop.
  BasicBlock *const TCCheckBlock = LoopVectorPreHeader;
  IRBuilder<> Builder(TCCheckBlock->getTerminator());

  // Generate code to check if the loop's trip count is less than VF * UF of the
  // main vector loop.
  auto P = Cost->requiresScalarEpilogue(ForEpilogue ? EPI.EpilogueVF : VF) ?
      ICmpInst::ICMP_ULE : ICmpInst::ICMP_ULT;

  Value *CheckMinIters = Builder.CreateICmp(
      P, Count, ConstantInt::get(Count->getType(), VFactor * UFactor),
      "min.iters.check");

  if (!ForEpilogue)
    TCCheckBlock->setName("vector.main.loop.iter.check");

  // Create new preheader for vector loop.
  LoopVectorPreHeader = SplitBlock(TCCheckBlock, TCCheckBlock->getTerminator(),
                                   DT, LI, nullptr, "vector.ph");

  if (ForEpilogue) {
    assert(DT->properlyDominates(DT->getNode(TCCheckBlock),
                                 DT->getNode(Bypass)->getIDom()) &&
           "TC check is expected to dominate Bypass");

    // Update dominator for Bypass & LoopExit.
    DT->changeImmediateDominator(Bypass, TCCheckBlock);
    DT->changeImmediateDominator(LoopExitBlock, TCCheckBlock);

    LoopBypassBlocks.push_back(TCCheckBlock);

    // Save the trip count so we don't have to regenerate it in the
    // vec.epilog.iter.check. This is safe to do because the trip count
    // generated here dominates the vector epilog iter check.
    EPI.TripCount = Count;
  }

  ReplaceInstWithInst(
      TCCheckBlock->getTerminator(),
      BranchInst::Create(Bypass, LoopVectorPreHeader, CheckMinIters));

  return TCCheckBlock;
}

//===--------------------------------------------------------------------===//
// EpilogueVectorizerEpilogueLoop
//===--------------------------------------------------------------------===//

/// This function is partially responsible for generating the control flow
/// depicted in https://llvm.org/docs/Vectorizers.html#epilogue-vectorization.
BasicBlock *
EpilogueVectorizerEpilogueLoop::createEpilogueVectorizedLoopSkeleton() {
  MDNode *OrigLoopID = OrigLoop->getLoopID();
  Loop *Lp = createVectorLoopSkeleton("vec.epilog.");

  // Now, compare the remaining count and if there aren't enough iterations to
  // execute the vectorized epilogue skip to the scalar part.
  BasicBlock *VecEpilogueIterationCountCheck = LoopVectorPreHeader;
  VecEpilogueIterationCountCheck->setName("vec.epilog.iter.check");
  LoopVectorPreHeader =
      SplitBlock(LoopVectorPreHeader, LoopVectorPreHeader->getTerminator(), DT,
                 LI, nullptr, "vec.epilog.ph");
  emitMinimumVectorEpilogueIterCountCheck(Lp, LoopScalarPreHeader,
                                          VecEpilogueIterationCountCheck);

  // Adjust the control flow taking the state info from the main loop
  // vectorization into account.
  assert(EPI.MainLoopIterationCountCheck && EPI.EpilogueIterationCountCheck &&
         "expected this to be saved from the previous pass.");
  EPI.MainLoopIterationCountCheck->getTerminator()->replaceUsesOfWith(
      VecEpilogueIterationCountCheck, LoopVectorPreHeader);

  DT->changeImmediateDominator(LoopVectorPreHeader,
                               EPI.MainLoopIterationCountCheck);

  EPI.EpilogueIterationCountCheck->getTerminator()->replaceUsesOfWith(
      VecEpilogueIterationCountCheck, LoopScalarPreHeader);

  if (EPI.SCEVSafetyCheck)
    EPI.SCEVSafetyCheck->getTerminator()->replaceUsesOfWith(
        VecEpilogueIterationCountCheck, LoopScalarPreHeader);
  if (EPI.MemSafetyCheck)
    EPI.MemSafetyCheck->getTerminator()->replaceUsesOfWith(
        VecEpilogueIterationCountCheck, LoopScalarPreHeader);

  DT->changeImmediateDominator(
      VecEpilogueIterationCountCheck,
      VecEpilogueIterationCountCheck->getSinglePredecessor());

  DT->changeImmediateDominator(LoopScalarPreHeader,
                               EPI.EpilogueIterationCountCheck);
  DT->changeImmediateDominator(LoopExitBlock, EPI.EpilogueIterationCountCheck);

  // Keep track of bypass blocks, as they feed start values to the induction
  // phis in the scalar loop preheader.
  if (EPI.SCEVSafetyCheck)
    LoopBypassBlocks.push_back(EPI.SCEVSafetyCheck);
  if (EPI.MemSafetyCheck)
    LoopBypassBlocks.push_back(EPI.MemSafetyCheck);
  LoopBypassBlocks.push_back(EPI.EpilogueIterationCountCheck);

  // Generate a resume induction for the vector epilogue and put it in the
  // vector epilogue preheader
  Type *IdxTy = Legal->getWidestInductionType();
  PHINode *EPResumeVal = PHINode::Create(IdxTy, 2, "vec.epilog.resume.val",
                                         LoopVectorPreHeader->getFirstNonPHI());
  EPResumeVal->addIncoming(EPI.VectorTripCount, VecEpilogueIterationCountCheck);
  EPResumeVal->addIncoming(ConstantInt::get(IdxTy, 0),
                           EPI.MainLoopIterationCountCheck);

  // Generate the induction variable.
  OldInduction = Legal->getPrimaryInduction();
  Value *CountRoundDown = getOrCreateVectorTripCount(Lp);
  Constant *Step = ConstantInt::get(IdxTy, VF.getKnownMinValue() * UF);
  Value *StartIdx = EPResumeVal;
  Induction =
      createInductionVariable(Lp, StartIdx, CountRoundDown, Step,
                              getDebugLocFromInstOrOperands(OldInduction));

  // Generate induction resume values. These variables save the new starting
  // indexes for the scalar loop. They are used to test if there are any tail
  // iterations left once the vector loop has completed.
  // Note that when the vectorized epilogue is skipped due to iteration count
  // check, then the resume value for the induction variable comes from
  // the trip count of the main vector loop, hence passing the AdditionalBypass
  // argument.
  createInductionResumeValues(Lp, CountRoundDown,
                              {VecEpilogueIterationCountCheck,
                               EPI.VectorTripCount} /* AdditionalBypass */);

  AddRuntimeUnrollDisableMetaData(Lp);
  return completeLoopSkeleton(Lp, OrigLoopID);
}

BasicBlock *
EpilogueVectorizerEpilogueLoop::emitMinimumVectorEpilogueIterCountCheck(
    Loop *L, BasicBlock *Bypass, BasicBlock *Insert) {

  assert(EPI.TripCount &&
         "Expected trip count to have been safed in the first pass.");
  assert(
      (!isa<Instruction>(EPI.TripCount) ||
       DT->dominates(cast<Instruction>(EPI.TripCount)->getParent(), Insert)) &&
      "saved trip count does not dominate insertion point.");
  Value *TC = EPI.TripCount;
  IRBuilder<> Builder(Insert->getTerminator());
  Value *Count = Builder.CreateSub(TC, EPI.VectorTripCount, "n.vec.remaining");

  // Generate code to check if the loop's trip count is less than VF * UF of the
  // vector epilogue loop.
  auto P = Cost->requiresScalarEpilogue(EPI.EpilogueVF) ?
      ICmpInst::ICMP_ULE : ICmpInst::ICMP_ULT;

  Value *CheckMinIters = Builder.CreateICmp(
      P, Count,
      ConstantInt::get(Count->getType(),
                       EPI.EpilogueVF.getKnownMinValue() * EPI.EpilogueUF),
      "min.epilog.iters.check");

  ReplaceInstWithInst(
      Insert->getTerminator(),
      BranchInst::Create(Bypass, LoopVectorPreHeader, CheckMinIters));

  LoopBypassBlocks.push_back(Insert);
  return Insert;
}

void EpilogueVectorizerEpilogueLoop::printDebugTracesAtStart() {
  LLVM_DEBUG({
    dbgs() << "Create Skeleton for epilogue vectorized loop (second pass)\n"
           << "Epilogue Loop VF:" << EPI.EpilogueVF.getKnownMinValue()
           << ", Epilogue Loop UF:" << EPI.EpilogueUF << "\n";
  });
}

void EpilogueVectorizerEpilogueLoop::printDebugTracesAtEnd() {
  DEBUG_WITH_TYPE(VerboseDebug, {
    dbgs() << "final fn:\n" << *Induction->getFunction() << "\n";
  });
}

bool LoopVectorizationPlanner::getDecisionAndClampRange(
    const std::function<bool(ElementCount)> &Predicate, VFRange &Range) {
  assert(!Range.isEmpty() && "Trying to test an empty VF range.");
  bool PredicateAtRangeStart = Predicate(Range.Start);

  for (ElementCount TmpVF = Range.Start * 2;
       ElementCount::isKnownLT(TmpVF, Range.End); TmpVF *= 2)
    if (Predicate(TmpVF) != PredicateAtRangeStart) {
      Range.End = TmpVF;
      break;
    }

  return PredicateAtRangeStart;
}

/// Build VPlans for the full range of feasible VF's = {\p MinVF, 2 * \p MinVF,
/// 4 * \p MinVF, ..., \p MaxVF} by repeatedly building a VPlan for a sub-range
/// of VF's starting at a given VF and extending it as much as possible. Each
/// vectorization decision can potentially shorten this sub-range during
/// buildVPlan().
void LoopVectorizationPlanner::buildVPlans(ElementCount MinVF,
                                           ElementCount MaxVF) {
  auto MaxVFPlusOne = MaxVF.getWithIncrement(1);
  for (ElementCount VF = MinVF; ElementCount::isKnownLT(VF, MaxVFPlusOne);) {
    VFRange SubRange = {VF, MaxVFPlusOne};
    VPlans.push_back(buildVPlan(SubRange));
    VF = SubRange.End;
  }
}

VPValue *VPRecipeBuilder::createEdgeMask(BasicBlock *Src, BasicBlock *Dst,
                                         VPlanPtr &Plan) {
  assert(is_contained(predecessors(Dst), Src) && "Invalid edge");

  // Look for cached value.
  std::pair<BasicBlock *, BasicBlock *> Edge(Src, Dst);
  EdgeMaskCacheTy::iterator ECEntryIt = EdgeMaskCache.find(Edge);
  if (ECEntryIt != EdgeMaskCache.end())
    return ECEntryIt->second;

  VPValue *SrcMask = createBlockInMask(Src, Plan);

  // The terminator has to be a branch inst!
  BranchInst *BI = dyn_cast<BranchInst>(Src->getTerminator());
  assert(BI && "Unexpected terminator found");

  if (!BI->isConditional() || BI->getSuccessor(0) == BI->getSuccessor(1))
    return EdgeMaskCache[Edge] = SrcMask;

  // If source is an exiting block, we know the exit edge is dynamically dead
  // in the vector loop, and thus we don't need to restrict the mask.  Avoid
  // adding uses of an otherwise potentially dead instruction.
  if (OrigLoop->isLoopExiting(Src))
    return EdgeMaskCache[Edge] = SrcMask;

  VPValue *EdgeMask = Plan->getOrAddVPValue(BI->getCondition());
  assert(EdgeMask && "No Edge Mask found for condition");

  if (BI->getSuccessor(0) != Dst)
    EdgeMask = Builder.createNot(EdgeMask);

  if (SrcMask) { // Otherwise block in-mask is all-one, no need to AND.
    // The condition is 'SrcMask && EdgeMask', which is equivalent to
    // 'select i1 SrcMask, i1 EdgeMask, i1 false'.
    // The select version does not introduce new UB if SrcMask is false and
    // EdgeMask is poison. Using 'and' here introduces undefined behavior.
    VPValue *False = Plan->getOrAddVPValue(
        ConstantInt::getFalse(BI->getCondition()->getType()));
    EdgeMask = Builder.createSelect(SrcMask, EdgeMask, False);
  }

  return EdgeMaskCache[Edge] = EdgeMask;
}

VPValue *VPRecipeBuilder::createBlockInMask(BasicBlock *BB, VPlanPtr &Plan) {
  assert(OrigLoop->contains(BB) && "Block is not a part of a loop");

  // Look for cached value.
  BlockMaskCacheTy::iterator BCEntryIt = BlockMaskCache.find(BB);
  if (BCEntryIt != BlockMaskCache.end())
    return BCEntryIt->second;

  // All-one mask is modelled as no-mask following the convention for masked
  // load/store/gather/scatter. Initialize BlockMask to no-mask.
  VPValue *BlockMask = nullptr;

  if (OrigLoop->getHeader() == BB) {
    if (!CM.blockNeedsPredication(BB))
      return BlockMaskCache[BB] = BlockMask; // Loop incoming mask is all-one.

    // Create the block in mask as the first non-phi instruction in the block.
    VPBuilder::InsertPointGuard Guard(Builder);
    auto NewInsertionPoint = Builder.getInsertBlock()->getFirstNonPhi();
    Builder.setInsertPoint(Builder.getInsertBlock(), NewInsertionPoint);

    // Introduce the early-exit compare IV <= BTC to form header block mask.
    // This is used instead of IV < TC because TC may wrap, unlike BTC.
    // Start by constructing the desired canonical IV.
    VPValue *IV = nullptr;
    if (Legal->getPrimaryInduction())
      IV = Plan->getOrAddVPValue(Legal->getPrimaryInduction());
    else {
      auto IVRecipe = new VPWidenCanonicalIVRecipe();
      Builder.getInsertBlock()->insert(IVRecipe, NewInsertionPoint);
      IV = IVRecipe->getVPSingleValue();
    }
    VPValue *BTC = Plan->getOrCreateBackedgeTakenCount();
    bool TailFolded = !CM.isScalarEpilogueAllowed();

    if (TailFolded && CM.TTI.emitGetActiveLaneMask()) {
      // While ActiveLaneMask is a binary op that consumes the loop tripcount
      // as a second argument, we only pass the IV here and extract the
      // tripcount from the transform state where codegen of the VP instructions
      // happen.
      BlockMask = Builder.createNaryOp(VPInstruction::ActiveLaneMask, {IV});
    } else {
      BlockMask = Builder.createNaryOp(VPInstruction::ICmpULE, {IV, BTC});
    }
    return BlockMaskCache[BB] = BlockMask;
  }

  // This is the block mask. We OR all incoming edges.
  for (auto *Predecessor : predecessors(BB)) {
    VPValue *EdgeMask = createEdgeMask(Predecessor, BB, Plan);
    if (!EdgeMask) // Mask of predecessor is all-one so mask of block is too.
      return BlockMaskCache[BB] = EdgeMask;

    if (!BlockMask) { // BlockMask has its initialized nullptr value.
      BlockMask = EdgeMask;
      continue;
    }

    BlockMask = Builder.createOr(BlockMask, EdgeMask);
  }

  return BlockMaskCache[BB] = BlockMask;
}

VPRecipeBase *VPRecipeBuilder::tryToWidenMemory(Instruction *I,
                                                ArrayRef<VPValue *> Operands,
                                                VFRange &Range,
                                                VPlanPtr &Plan) {
  assert((isa<LoadInst>(I) || isa<StoreInst>(I)) &&
         "Must be called with either a load or store");

  auto willWiden = [&](ElementCount VF) -> bool {
    if (VF.isScalar())
      return false;
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

  if (!LoopVectorizationPlanner::getDecisionAndClampRange(willWiden, Range))
    return nullptr;

  VPValue *Mask = nullptr;
  if (Legal->isMaskRequired(I))
    Mask = createBlockInMask(I->getParent(), Plan);

  if (LoadInst *Load = dyn_cast<LoadInst>(I))
    return new VPWidenMemoryInstructionRecipe(*Load, Operands[0], Mask);

  StoreInst *Store = cast<StoreInst>(I);
  return new VPWidenMemoryInstructionRecipe(*Store, Operands[1], Operands[0],
                                            Mask);
}

VPWidenIntOrFpInductionRecipe *
VPRecipeBuilder::tryToOptimizeInductionPHI(PHINode *Phi,
                                           ArrayRef<VPValue *> Operands) const {
  // Check if this is an integer or fp induction. If so, build the recipe that
  // produces its scalar and vector values.
  InductionDescriptor II = Legal->getInductionVars().lookup(Phi);
  if (II.getKind() == InductionDescriptor::IK_IntInduction ||
      II.getKind() == InductionDescriptor::IK_FpInduction) {
    assert(II.getStartValue() ==
           Phi->getIncomingValueForBlock(OrigLoop->getLoopPreheader()));
    const SmallVectorImpl<Instruction *> &Casts = II.getCastInsts();
    return new VPWidenIntOrFpInductionRecipe(
        Phi, Operands[0], Casts.empty() ? nullptr : Casts.front());
  }

  return nullptr;
}

VPWidenIntOrFpInductionRecipe *VPRecipeBuilder::tryToOptimizeInductionTruncate(
    TruncInst *I, ArrayRef<VPValue *> Operands, VFRange &Range,
    VPlan &Plan) const {
  // Optimize the special case where the source is a constant integer
  // induction variable. Notice that we can only optimize the 'trunc' case
  // because (a) FP conversions lose precision, (b) sext/zext may wrap, and
  // (c) other casts depend on pointer size.

  // Determine whether \p K is a truncation based on an induction variable that
  // can be optimized.
  auto isOptimizableIVTruncate =
      [&](Instruction *K) -> std::function<bool(ElementCount)> {
    return [=](ElementCount VF) -> bool {
      return CM.isOptimizableIVTruncate(K, VF);
    };
  };

  if (LoopVectorizationPlanner::getDecisionAndClampRange(
          isOptimizableIVTruncate(I), Range)) {

    InductionDescriptor II =
        Legal->getInductionVars().lookup(cast<PHINode>(I->getOperand(0)));
    VPValue *Start = Plan.getOrAddVPValue(II.getStartValue());
    return new VPWidenIntOrFpInductionRecipe(cast<PHINode>(I->getOperand(0)),
                                             Start, nullptr, I);
  }
  return nullptr;
}

VPRecipeOrVPValueTy VPRecipeBuilder::tryToBlend(PHINode *Phi,
                                                ArrayRef<VPValue *> Operands,
                                                VPlanPtr &Plan) {
  // If all incoming values are equal, the incoming VPValue can be used directly
  // instead of creating a new VPBlendRecipe.
  VPValue *FirstIncoming = Operands[0];
  if (all_of(Operands, [FirstIncoming](const VPValue *Inc) {
        return FirstIncoming == Inc;
      })) {
    return Operands[0];
  }

  // We know that all PHIs in non-header blocks are converted into selects, so
  // we don't have to worry about the insertion order and we can just use the
  // builder. At this point we generate the predication tree. There may be
  // duplications since this is a simple recursive scan, but future
  // optimizations will clean it up.
  SmallVector<VPValue *, 2> OperandsWithMask;
  unsigned NumIncoming = Phi->getNumIncomingValues();

  for (unsigned In = 0; In < NumIncoming; In++) {
    VPValue *EdgeMask =
      createEdgeMask(Phi->getIncomingBlock(In), Phi->getParent(), Plan);
    assert((EdgeMask || NumIncoming == 1) &&
           "Multiple predecessors with one having a full mask");
    OperandsWithMask.push_back(Operands[In]);
    if (EdgeMask)
      OperandsWithMask.push_back(EdgeMask);
  }
  return toVPRecipeResult(new VPBlendRecipe(Phi, OperandsWithMask));
}

VPWidenCallRecipe *VPRecipeBuilder::tryToWidenCall(CallInst *CI,
                                                   ArrayRef<VPValue *> Operands,
                                                   VFRange &Range) const {

  bool IsPredicated = LoopVectorizationPlanner::getDecisionAndClampRange(
      [this, CI](ElementCount VF) { return CM.isScalarWithPredication(CI); },
      Range);

  if (IsPredicated)
    return nullptr;

  Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);
  if (ID && (ID == Intrinsic::assume || ID == Intrinsic::lifetime_end ||
             ID == Intrinsic::lifetime_start || ID == Intrinsic::sideeffect ||
             ID == Intrinsic::pseudoprobe ||
             ID == Intrinsic::experimental_noalias_scope_decl))
    return nullptr;

  auto willWiden = [&](ElementCount VF) -> bool {
    Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);
    // The following case may be scalarized depending on the VF.
    // The flag shows whether we use Intrinsic or a usual Call for vectorized
    // version of the instruction.
    // Is it beneficial to perform intrinsic call compared to lib call?
    bool NeedToScalarize = false;
    InstructionCost CallCost = CM.getVectorCallCost(CI, VF, NeedToScalarize);
    InstructionCost IntrinsicCost = ID ? CM.getVectorIntrinsicCost(CI, VF) : 0;
    bool UseVectorIntrinsic = ID && IntrinsicCost <= CallCost;
    assert((IntrinsicCost.isValid() || CallCost.isValid()) &&
           "Either the intrinsic cost or vector call cost must be valid");
    return UseVectorIntrinsic || !NeedToScalarize;
  };

  if (!LoopVectorizationPlanner::getDecisionAndClampRange(willWiden, Range))
    return nullptr;

  ArrayRef<VPValue *> Ops = Operands.take_front(CI->getNumArgOperands());
  return new VPWidenCallRecipe(*CI, make_range(Ops.begin(), Ops.end()));
}

bool VPRecipeBuilder::shouldWiden(Instruction *I, VFRange &Range) const {
  assert(!isa<BranchInst>(I) && !isa<PHINode>(I) && !isa<LoadInst>(I) &&
         !isa<StoreInst>(I) && "Instruction should have been handled earlier");
  // Instruction should be widened, unless it is scalar after vectorization,
  // scalarization is profitable or it is predicated.
  auto WillScalarize = [this, I](ElementCount VF) -> bool {
    return CM.isScalarAfterVectorization(I, VF) ||
           CM.isProfitableToScalarize(I, VF) || CM.isScalarWithPredication(I);
  };
  return !LoopVectorizationPlanner::getDecisionAndClampRange(WillScalarize,
                                                             Range);
}

VPWidenRecipe *VPRecipeBuilder::tryToWiden(Instruction *I,
                                           ArrayRef<VPValue *> Operands) const {
  auto IsVectorizableOpcode = [](unsigned Opcode) {
    switch (Opcode) {
    case Instruction::Add:
    case Instruction::And:
    case Instruction::AShr:
    case Instruction::BitCast:
    case Instruction::FAdd:
    case Instruction::FCmp:
    case Instruction::FDiv:
    case Instruction::FMul:
    case Instruction::FNeg:
    case Instruction::FPExt:
    case Instruction::FPToSI:
    case Instruction::FPToUI:
    case Instruction::FPTrunc:
    case Instruction::FRem:
    case Instruction::FSub:
    case Instruction::ICmp:
    case Instruction::IntToPtr:
    case Instruction::LShr:
    case Instruction::Mul:
    case Instruction::Or:
    case Instruction::PtrToInt:
    case Instruction::SDiv:
    case Instruction::Select:
    case Instruction::SExt:
    case Instruction::Shl:
    case Instruction::SIToFP:
    case Instruction::SRem:
    case Instruction::Sub:
    case Instruction::Trunc:
    case Instruction::UDiv:
    case Instruction::UIToFP:
    case Instruction::URem:
    case Instruction::Xor:
    case Instruction::ZExt:
      return true;
    }
    return false;
  };

  if (!IsVectorizableOpcode(I->getOpcode()))
    return nullptr;

  // Success: widen this instruction.
  return new VPWidenRecipe(*I, make_range(Operands.begin(), Operands.end()));
}

void VPRecipeBuilder::fixHeaderPhis() {
  BasicBlock *OrigLatch = OrigLoop->getLoopLatch();
  for (VPWidenPHIRecipe *R : PhisToFix) {
    auto *PN = cast<PHINode>(R->getUnderlyingValue());
    VPRecipeBase *IncR =
        getRecipe(cast<Instruction>(PN->getIncomingValueForBlock(OrigLatch)));
    R->addOperand(IncR->getVPSingleValue());
  }
}

VPBasicBlock *VPRecipeBuilder::handleReplication(
    Instruction *I, VFRange &Range, VPBasicBlock *VPBB,
    VPlanPtr &Plan) {
  bool IsUniform = LoopVectorizationPlanner::getDecisionAndClampRange(
      [&](ElementCount VF) { return CM.isUniformAfterVectorization(I, VF); },
      Range);

  bool IsPredicated = LoopVectorizationPlanner::getDecisionAndClampRange(
      [&](ElementCount VF) { return CM.isPredicatedInst(I); }, Range);

  auto *Recipe = new VPReplicateRecipe(I, Plan->mapToVPValues(I->operands()),
                                       IsUniform, IsPredicated);
  setRecipe(I, Recipe);
  Plan->addVPValue(I, Recipe);

  // Find if I uses a predicated instruction. If so, it will use its scalar
  // value. Avoid hoisting the insert-element which packs the scalar value into
  // a vector value, as that happens iff all users use the vector value.
  for (VPValue *Op : Recipe->operands()) {
    auto *PredR = dyn_cast_or_null<VPPredInstPHIRecipe>(Op->getDef());
    if (!PredR)
      continue;
    auto *RepR =
        cast_or_null<VPReplicateRecipe>(PredR->getOperand(0)->getDef());
    assert(RepR->isPredicated() &&
           "expected Replicate recipe to be predicated");
    RepR->setAlsoPack(false);
  }

  // Finalize the recipe for Instr, first if it is not predicated.
  if (!IsPredicated) {
    LLVM_DEBUG(dbgs() << "LV: Scalarizing:" << *I << "\n");
    VPBB->appendRecipe(Recipe);
    return VPBB;
  }
  LLVM_DEBUG(dbgs() << "LV: Scalarizing and predicating:" << *I << "\n");
  assert(VPBB->getSuccessors().empty() &&
         "VPBB has successors when handling predicated replication.");
  // Record predicated instructions for above packing optimizations.
  VPBlockBase *Region = createReplicateRegion(I, Recipe, Plan);
  VPBlockUtils::insertBlockAfter(Region, VPBB);
  auto *RegSucc = new VPBasicBlock();
  VPBlockUtils::insertBlockAfter(RegSucc, Region);
  return RegSucc;
}

VPRegionBlock *VPRecipeBuilder::createReplicateRegion(Instruction *Instr,
                                                      VPRecipeBase *PredRecipe,
                                                      VPlanPtr &Plan) {
  // Instructions marked for predication are replicated and placed under an
  // if-then construct to prevent side-effects.

  // Generate recipes to compute the block mask for this region.
  VPValue *BlockInMask = createBlockInMask(Instr->getParent(), Plan);

  // Build the triangular if-then region.
  std::string RegionName = (Twine("pred.") + Instr->getOpcodeName()).str();
  assert(Instr->getParent() && "Predicated instruction not in any basic block");
  auto *BOMRecipe = new VPBranchOnMaskRecipe(BlockInMask);
  auto *Entry = new VPBasicBlock(Twine(RegionName) + ".entry", BOMRecipe);
  auto *PHIRecipe = Instr->getType()->isVoidTy()
                        ? nullptr
                        : new VPPredInstPHIRecipe(Plan->getOrAddVPValue(Instr));
  if (PHIRecipe) {
    Plan->removeVPValueFor(Instr);
    Plan->addVPValue(Instr, PHIRecipe);
  }
  auto *Exit = new VPBasicBlock(Twine(RegionName) + ".continue", PHIRecipe);
  auto *Pred = new VPBasicBlock(Twine(RegionName) + ".if", PredRecipe);
  VPRegionBlock *Region = new VPRegionBlock(Entry, Exit, RegionName, true);

  // Note: first set Entry as region entry and then connect successors starting
  // from it in order, to propagate the "parent" of each VPBasicBlock.
  VPBlockUtils::insertTwoBlocksAfter(Pred, Exit, BlockInMask, Entry);
  VPBlockUtils::connectBlocks(Pred, Exit);

  return Region;
}

VPRecipeOrVPValueTy
VPRecipeBuilder::tryToCreateWidenRecipe(Instruction *Instr,
                                        ArrayRef<VPValue *> Operands,
                                        VFRange &Range, VPlanPtr &Plan) {
  // First, check for specific widening recipes that deal with calls, memory
  // operations, inductions and Phi nodes.
  if (auto *CI = dyn_cast<CallInst>(Instr))
    return toVPRecipeResult(tryToWidenCall(CI, Operands, Range));

  if (isa<LoadInst>(Instr) || isa<StoreInst>(Instr))
    return toVPRecipeResult(tryToWidenMemory(Instr, Operands, Range, Plan));

  VPRecipeBase *Recipe;
  if (auto Phi = dyn_cast<PHINode>(Instr)) {
    if (Phi->getParent() != OrigLoop->getHeader())
      return tryToBlend(Phi, Operands, Plan);
    if ((Recipe = tryToOptimizeInductionPHI(Phi, Operands)))
      return toVPRecipeResult(Recipe);

    VPWidenPHIRecipe *PhiRecipe = nullptr;
    if (Legal->isReductionVariable(Phi) || Legal->isFirstOrderRecurrence(Phi)) {
      VPValue *StartV = Operands[0];
      if (Legal->isReductionVariable(Phi)) {
        RecurrenceDescriptor &RdxDesc = Legal->getReductionVars()[Phi];
        assert(RdxDesc.getRecurrenceStartValue() ==
               Phi->getIncomingValueForBlock(OrigLoop->getLoopPreheader()));
        PhiRecipe = new VPWidenPHIRecipe(Phi, RdxDesc, *StartV);
      } else {
        PhiRecipe = new VPWidenPHIRecipe(Phi, *StartV);
      }

      // Record the incoming value from the backedge, so we can add the incoming
      // value from the backedge after all recipes have been created.
      recordRecipeOf(cast<Instruction>(
          Phi->getIncomingValueForBlock(OrigLoop->getLoopLatch())));
      PhisToFix.push_back(PhiRecipe);
    } else {
      // TODO: record start and backedge value for remaining pointer induction
      // phis.
      assert(Phi->getType()->isPointerTy() &&
             "only pointer phis should be handled here");
      PhiRecipe = new VPWidenPHIRecipe(Phi);
    }

    return toVPRecipeResult(PhiRecipe);
  }

  if (isa<TruncInst>(Instr) &&
      (Recipe = tryToOptimizeInductionTruncate(cast<TruncInst>(Instr), Operands,
                                               Range, *Plan)))
    return toVPRecipeResult(Recipe);

  if (!shouldWiden(Instr, Range))
    return nullptr;

  if (auto GEP = dyn_cast<GetElementPtrInst>(Instr))
    return toVPRecipeResult(new VPWidenGEPRecipe(
        GEP, make_range(Operands.begin(), Operands.end()), OrigLoop));

  if (auto *SI = dyn_cast<SelectInst>(Instr)) {
    bool InvariantCond =
        PSE.getSE()->isLoopInvariant(PSE.getSCEV(SI->getOperand(0)), OrigLoop);
    return toVPRecipeResult(new VPWidenSelectRecipe(
        *SI, make_range(Operands.begin(), Operands.end()), InvariantCond));
  }

  return toVPRecipeResult(tryToWiden(Instr, Operands));
}

void LoopVectorizationPlanner::buildVPlansWithVPRecipes(ElementCount MinVF,
                                                        ElementCount MaxVF) {
  assert(OrigLoop->isInnermost() && "Inner loop expected.");

  // Collect instructions from the original loop that will become trivially dead
  // in the vectorized loop. We don't need to vectorize these instructions. For
  // example, original induction update instructions can become dead because we
  // separately emit induction "steps" when generating code for the new loop.
  // Similarly, we create a new latch condition when setting up the structure
  // of the new loop, so the old one can become dead.
  SmallPtrSet<Instruction *, 4> DeadInstructions;
  collectTriviallyDeadInstructions(DeadInstructions);

  // Add assume instructions we need to drop to DeadInstructions, to prevent
  // them from being added to the VPlan.
  // TODO: We only need to drop assumes in blocks that get flattend. If the
  // control flow is preserved, we should keep them.
  auto &ConditionalAssumes = Legal->getConditionalAssumes();
  DeadInstructions.insert(ConditionalAssumes.begin(), ConditionalAssumes.end());

  MapVector<Instruction *, Instruction *> &SinkAfter = Legal->getSinkAfter();
  // Dead instructions do not need sinking. Remove them from SinkAfter.
  for (Instruction *I : DeadInstructions)
    SinkAfter.erase(I);

  // Cannot sink instructions after dead instructions (there won't be any
  // recipes for them). Instead, find the first non-dead previous instruction.
  for (auto &P : Legal->getSinkAfter()) {
    Instruction *SinkTarget = P.second;
    Instruction *FirstInst = &*SinkTarget->getParent()->begin();
    (void)FirstInst;
    while (DeadInstructions.contains(SinkTarget)) {
      assert(
          SinkTarget != FirstInst &&
          "Must find a live instruction (at least the one feeding the "
          "first-order recurrence PHI) before reaching beginning of the block");
      SinkTarget = SinkTarget->getPrevNode();
      assert(SinkTarget != P.first &&
             "sink source equals target, no sinking required");
    }
    P.second = SinkTarget;
  }

  auto MaxVFPlusOne = MaxVF.getWithIncrement(1);
  for (ElementCount VF = MinVF; ElementCount::isKnownLT(VF, MaxVFPlusOne);) {
    VFRange SubRange = {VF, MaxVFPlusOne};
    VPlans.push_back(
        buildVPlanWithVPRecipes(SubRange, DeadInstructions, SinkAfter));
    VF = SubRange.End;
  }
}

VPlanPtr LoopVectorizationPlanner::buildVPlanWithVPRecipes(
    VFRange &Range, SmallPtrSetImpl<Instruction *> &DeadInstructions,
    const MapVector<Instruction *, Instruction *> &SinkAfter) {

  SmallPtrSet<const InterleaveGroup<Instruction> *, 1> InterleaveGroups;

  VPRecipeBuilder RecipeBuilder(OrigLoop, TLI, Legal, CM, PSE, Builder);

  // ---------------------------------------------------------------------------
  // Pre-construction: record ingredients whose recipes we'll need to further
  // process after constructing the initial VPlan.
  // ---------------------------------------------------------------------------

  // Mark instructions we'll need to sink later and their targets as
  // ingredients whose recipe we'll need to record.
  for (auto &Entry : SinkAfter) {
    RecipeBuilder.recordRecipeOf(Entry.first);
    RecipeBuilder.recordRecipeOf(Entry.second);
  }
  for (auto &Reduction : CM.getInLoopReductionChains()) {
    PHINode *Phi = Reduction.first;
    RecurKind Kind = Legal->getReductionVars()[Phi].getRecurrenceKind();
    const SmallVector<Instruction *, 4> &ReductionOperations = Reduction.second;

    RecipeBuilder.recordRecipeOf(Phi);
    for (auto &R : ReductionOperations) {
      RecipeBuilder.recordRecipeOf(R);
      // For min/max reducitons, where we have a pair of icmp/select, we also
      // need to record the ICmp recipe, so it can be removed later.
      if (RecurrenceDescriptor::isMinMaxRecurrenceKind(Kind))
        RecipeBuilder.recordRecipeOf(cast<Instruction>(R->getOperand(0)));
    }
  }

  // For each interleave group which is relevant for this (possibly trimmed)
  // Range, add it to the set of groups to be later applied to the VPlan and add
  // placeholders for its members' Recipes which we'll be replacing with a
  // single VPInterleaveRecipe.
  for (InterleaveGroup<Instruction> *IG : IAI.getInterleaveGroups()) {
    auto applyIG = [IG, this](ElementCount VF) -> bool {
      return (VF.isVector() && // Query is illegal for VF == 1
              CM.getWideningDecision(IG->getInsertPos(), VF) ==
                  LoopVectorizationCostModel::CM_Interleave);
    };
    if (!getDecisionAndClampRange(applyIG, Range))
      continue;
    InterleaveGroups.insert(IG);
    for (unsigned i = 0; i < IG->getFactor(); i++)
      if (Instruction *Member = IG->getMember(i))
        RecipeBuilder.recordRecipeOf(Member);
  };

  // ---------------------------------------------------------------------------
  // Build initial VPlan: Scan the body of the loop in a topological order to
  // visit each basic block after having visited its predecessor basic blocks.
  // ---------------------------------------------------------------------------

  // Create a dummy pre-entry VPBasicBlock to start building the VPlan.
  auto Plan = std::make_unique<VPlan>();
  VPBasicBlock *VPBB = new VPBasicBlock("Pre-Entry");
  Plan->setEntry(VPBB);

  // Scan the body of the loop in a topological order to visit each basic block
  // after having visited its predecessor basic blocks.
  LoopBlocksDFS DFS(OrigLoop);
  DFS.perform(LI);

  for (BasicBlock *BB : make_range(DFS.beginRPO(), DFS.endRPO())) {
    // Relevant instructions from basic block BB will be grouped into VPRecipe
    // ingredients and fill a new VPBasicBlock.
    unsigned VPBBsForBB = 0;
    auto *FirstVPBBForBB = new VPBasicBlock(BB->getName());
    VPBlockUtils::insertBlockAfter(FirstVPBBForBB, VPBB);
    VPBB = FirstVPBBForBB;
    Builder.setInsertPoint(VPBB);

    // Introduce each ingredient into VPlan.
    // TODO: Model and preserve debug instrinsics in VPlan.
    for (Instruction &I : BB->instructionsWithoutDebug()) {
      Instruction *Instr = &I;

      // First filter out irrelevant instructions, to ensure no recipes are
      // built for them.
      if (isa<BranchInst>(Instr) || DeadInstructions.count(Instr))
        continue;

      SmallVector<VPValue *, 4> Operands;
      auto *Phi = dyn_cast<PHINode>(Instr);
      if (Phi && Phi->getParent() == OrigLoop->getHeader()) {
        Operands.push_back(Plan->getOrAddVPValue(
            Phi->getIncomingValueForBlock(OrigLoop->getLoopPreheader())));
      } else {
        auto OpRange = Plan->mapToVPValues(Instr->operands());
        Operands = {OpRange.begin(), OpRange.end()};
      }
      if (auto RecipeOrValue = RecipeBuilder.tryToCreateWidenRecipe(
              Instr, Operands, Range, Plan)) {
        // If Instr can be simplified to an existing VPValue, use it.
        if (RecipeOrValue.is<VPValue *>()) {
          auto *VPV = RecipeOrValue.get<VPValue *>();
          Plan->addVPValue(Instr, VPV);
          // If the re-used value is a recipe, register the recipe for the
          // instruction, in case the recipe for Instr needs to be recorded.
          if (auto *R = dyn_cast_or_null<VPRecipeBase>(VPV->getDef()))
            RecipeBuilder.setRecipe(Instr, R);
          continue;
        }
        // Otherwise, add the new recipe.
        VPRecipeBase *Recipe = RecipeOrValue.get<VPRecipeBase *>();
        for (auto *Def : Recipe->definedValues()) {
          auto *UV = Def->getUnderlyingValue();
          Plan->addVPValue(UV, Def);
        }

        RecipeBuilder.setRecipe(Instr, Recipe);
        VPBB->appendRecipe(Recipe);
        continue;
      }

      // Otherwise, if all widening options failed, Instruction is to be
      // replicated. This may create a successor for VPBB.
      VPBasicBlock *NextVPBB =
          RecipeBuilder.handleReplication(Instr, Range, VPBB, Plan);
      if (NextVPBB != VPBB) {
        VPBB = NextVPBB;
        VPBB->setName(BB->hasName() ? BB->getName() + "." + Twine(VPBBsForBB++)
                                    : "");
      }
    }
  }

  RecipeBuilder.fixHeaderPhis();

  // Discard empty dummy pre-entry VPBasicBlock. Note that other VPBasicBlocks
  // may also be empty, such as the last one VPBB, reflecting original
  // basic-blocks with no recipes.
  VPBasicBlock *PreEntry = cast<VPBasicBlock>(Plan->getEntry());
  assert(PreEntry->empty() && "Expecting empty pre-entry block.");
  VPBlockBase *Entry = Plan->setEntry(PreEntry->getSingleSuccessor());
  VPBlockUtils::disconnectBlocks(PreEntry, Entry);
  delete PreEntry;

  // ---------------------------------------------------------------------------
  // Transform initial VPlan: Apply previously taken decisions, in order, to
  // bring the VPlan to its final state.
  // ---------------------------------------------------------------------------

  // Apply Sink-After legal constraints.
  for (auto &Entry : SinkAfter) {
    VPRecipeBase *Sink = RecipeBuilder.getRecipe(Entry.first);
    VPRecipeBase *Target = RecipeBuilder.getRecipe(Entry.second);

    auto GetReplicateRegion = [](VPRecipeBase *R) -> VPRegionBlock * {
      auto *Region =
          dyn_cast_or_null<VPRegionBlock>(R->getParent()->getParent());
      if (Region && Region->isReplicator()) {
        assert(Region->getNumSuccessors() == 1 &&
               Region->getNumPredecessors() == 1 && "Expected SESE region!");
        assert(R->getParent()->size() == 1 &&
               "A recipe in an original replicator region must be the only "
               "recipe in its block");
        return Region;
      }
      return nullptr;
    };
    auto *TargetRegion = GetReplicateRegion(Target);
    auto *SinkRegion = GetReplicateRegion(Sink);
    if (!SinkRegion) {
      // If the sink source is not a replicate region, sink the recipe directly.
      if (TargetRegion) {
        // The target is in a replication region, make sure to move Sink to
        // the block after it, not into the replication region itself.
        VPBasicBlock *NextBlock =
            cast<VPBasicBlock>(TargetRegion->getSuccessors().front());
        Sink->moveBefore(*NextBlock, NextBlock->getFirstNonPhi());
      } else
        Sink->moveAfter(Target);
      continue;
    }

    // The sink source is in a replicate region. Unhook the region from the CFG.
    auto *SinkPred = SinkRegion->getSinglePredecessor();
    auto *SinkSucc = SinkRegion->getSingleSuccessor();
    VPBlockUtils::disconnectBlocks(SinkPred, SinkRegion);
    VPBlockUtils::disconnectBlocks(SinkRegion, SinkSucc);
    VPBlockUtils::connectBlocks(SinkPred, SinkSucc);

    if (TargetRegion) {
      // The target recipe is also in a replicate region, move the sink region
      // after the target region.
      auto *TargetSucc = TargetRegion->getSingleSuccessor();
      VPBlockUtils::disconnectBlocks(TargetRegion, TargetSucc);
      VPBlockUtils::connectBlocks(TargetRegion, SinkRegion);
      VPBlockUtils::connectBlocks(SinkRegion, TargetSucc);
    } else {
      // The sink source is in a replicate region, we need to move the whole
      // replicate region, which should only contain a single recipe in the main
      // block.
      auto *SplitBlock =
          Target->getParent()->splitAt(std::next(Target->getIterator()));

      auto *SplitPred = SplitBlock->getSinglePredecessor();

      VPBlockUtils::disconnectBlocks(SplitPred, SplitBlock);
      VPBlockUtils::connectBlocks(SplitPred, SinkRegion);
      VPBlockUtils::connectBlocks(SinkRegion, SplitBlock);
      if (VPBB == SplitPred)
        VPBB = SplitBlock;
    }
  }

  // Interleave memory: for each Interleave Group we marked earlier as relevant
  // for this VPlan, replace the Recipes widening its memory instructions with a
  // single VPInterleaveRecipe at its insertion point.
  for (auto IG : InterleaveGroups) {
    auto *Recipe = cast<VPWidenMemoryInstructionRecipe>(
        RecipeBuilder.getRecipe(IG->getInsertPos()));
    SmallVector<VPValue *, 4> StoredValues;
    for (unsigned i = 0; i < IG->getFactor(); ++i)
      if (auto *SI = dyn_cast_or_null<StoreInst>(IG->getMember(i)))
        StoredValues.push_back(Plan->getOrAddVPValue(SI->getOperand(0)));

    auto *VPIG = new VPInterleaveRecipe(IG, Recipe->getAddr(), StoredValues,
                                        Recipe->getMask());
    VPIG->insertBefore(Recipe);
    unsigned J = 0;
    for (unsigned i = 0; i < IG->getFactor(); ++i)
      if (Instruction *Member = IG->getMember(i)) {
        if (!Member->getType()->isVoidTy()) {
          VPValue *OriginalV = Plan->getVPValue(Member);
          Plan->removeVPValueFor(Member);
          Plan->addVPValue(Member, VPIG->getVPValue(J));
          OriginalV->replaceAllUsesWith(VPIG->getVPValue(J));
          J++;
        }
        RecipeBuilder.getRecipe(Member)->eraseFromParent();
      }
  }

  // Adjust the recipes for any inloop reductions.
  adjustRecipesForInLoopReductions(Plan, RecipeBuilder, Range.Start);

  // Finally, if tail is folded by masking, introduce selects between the phi
  // and the live-out instruction of each reduction, at the end of the latch.
  if (CM.foldTailByMasking() && !Legal->getReductionVars().empty()) {
    Builder.setInsertPoint(VPBB);
    auto *Cond = RecipeBuilder.createBlockInMask(OrigLoop->getHeader(), Plan);
    for (auto &Reduction : Legal->getReductionVars()) {
      if (CM.isInLoopReduction(Reduction.first))
        continue;
      VPValue *Phi = Plan->getOrAddVPValue(Reduction.first);
      VPValue *Red = Plan->getOrAddVPValue(Reduction.second.getLoopExitInstr());
      Builder.createNaryOp(Instruction::Select, {Cond, Red, Phi});
    }
  }

  VPlanTransforms::sinkScalarOperands(*Plan);
  VPlanTransforms::mergeReplicateRegions(*Plan);

  std::string PlanName;
  raw_string_ostream RSO(PlanName);
  ElementCount VF = Range.Start;
  Plan->addVF(VF);
  RSO << "Initial VPlan for VF={" << VF;
  for (VF *= 2; ElementCount::isKnownLT(VF, Range.End); VF *= 2) {
    Plan->addVF(VF);
    RSO << "," << VF;
  }
  RSO << "},UF>=1";
  RSO.flush();
  Plan->setName(PlanName);

  return Plan;
}

VPlanPtr LoopVectorizationPlanner::buildVPlan(VFRange &Range) {
  // Outer loop handling: They may require CFG and instruction level
  // transformations before even evaluating whether vectorization is profitable.
  // Since we cannot modify the incoming IR, we need to build VPlan upfront in
  // the vectorization pipeline.
  assert(!OrigLoop->isInnermost());
  assert(EnableVPlanNativePath && "VPlan-native path is not enabled.");

  // Create new empty VPlan
  auto Plan = std::make_unique<VPlan>();

  // Build hierarchical CFG
  VPlanHCFGBuilder HCFGBuilder(OrigLoop, LI, *Plan);
  HCFGBuilder.buildHierarchicalCFG();

  for (ElementCount VF = Range.Start; ElementCount::isKnownLT(VF, Range.End);
       VF *= 2)
    Plan->addVF(VF);

  if (EnableVPlanPredication) {
    VPlanPredicator VPP(*Plan);
    VPP.predicate();

    // Avoid running transformation to recipes until masked code generation in
    // VPlan-native path is in place.
    return Plan;
  }

  SmallPtrSet<Instruction *, 1> DeadInstructions;
  VPlanTransforms::VPInstructionsToVPRecipes(OrigLoop, Plan,
                                             Legal->getInductionVars(),
                                             DeadInstructions, *PSE.getSE());
  return Plan;
}

// Adjust the recipes for any inloop reductions. The chain of instructions
// leading from the loop exit instr to the phi need to be converted to
// reductions, with one operand being vector and the other being the scalar
// reduction chain.
void LoopVectorizationPlanner::adjustRecipesForInLoopReductions(
    VPlanPtr &Plan, VPRecipeBuilder &RecipeBuilder, ElementCount MinVF) {
  for (auto &Reduction : CM.getInLoopReductionChains()) {
    PHINode *Phi = Reduction.first;
    RecurrenceDescriptor &RdxDesc = Legal->getReductionVars()[Phi];
    const SmallVector<Instruction *, 4> &ReductionOperations = Reduction.second;

    if (MinVF.isScalar() && !CM.useOrderedReductions(RdxDesc))
      continue;

    // ReductionOperations are orders top-down from the phi's use to the
    // LoopExitValue. We keep a track of the previous item (the Chain) to tell
    // which of the two operands will remain scalar and which will be reduced.
    // For minmax the chain will be the select instructions.
    Instruction *Chain = Phi;
    for (Instruction *R : ReductionOperations) {
      VPRecipeBase *WidenRecipe = RecipeBuilder.getRecipe(R);
      RecurKind Kind = RdxDesc.getRecurrenceKind();

      VPValue *ChainOp = Plan->getVPValue(Chain);
      unsigned FirstOpId;
      if (RecurrenceDescriptor::isMinMaxRecurrenceKind(Kind)) {
        assert(isa<VPWidenSelectRecipe>(WidenRecipe) &&
               "Expected to replace a VPWidenSelectSC");
        FirstOpId = 1;
      } else {
        assert((MinVF.isScalar() || isa<VPWidenRecipe>(WidenRecipe)) &&
               "Expected to replace a VPWidenSC");
        FirstOpId = 0;
      }
      unsigned VecOpId =
          R->getOperand(FirstOpId) == Chain ? FirstOpId + 1 : FirstOpId;
      VPValue *VecOp = Plan->getVPValue(R->getOperand(VecOpId));

      auto *CondOp = CM.foldTailByMasking()
                         ? RecipeBuilder.createBlockInMask(R->getParent(), Plan)
                         : nullptr;
      VPReductionRecipe *RedRecipe = new VPReductionRecipe(
          &RdxDesc, R, ChainOp, VecOp, CondOp, TTI);
      WidenRecipe->getVPSingleValue()->replaceAllUsesWith(RedRecipe);
      Plan->removeVPValueFor(R);
      Plan->addVPValue(R, RedRecipe);
      WidenRecipe->getParent()->insert(RedRecipe, WidenRecipe->getIterator());
      WidenRecipe->getVPSingleValue()->replaceAllUsesWith(RedRecipe);
      WidenRecipe->eraseFromParent();

      if (RecurrenceDescriptor::isMinMaxRecurrenceKind(Kind)) {
        VPRecipeBase *CompareRecipe =
            RecipeBuilder.getRecipe(cast<Instruction>(R->getOperand(0)));
        assert(isa<VPWidenRecipe>(CompareRecipe) &&
               "Expected to replace a VPWidenSC");
        assert(cast<VPWidenRecipe>(CompareRecipe)->getNumUsers() == 0 &&
               "Expected no remaining users");
        CompareRecipe->eraseFromParent();
      }
      Chain = R;
    }
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VPInterleaveRecipe::print(raw_ostream &O, const Twine &Indent,
                               VPSlotTracker &SlotTracker) const {
  O << Indent << "INTERLEAVE-GROUP with factor " << IG->getFactor() << " at ";
  IG->getInsertPos()->printAsOperand(O, false);
  O << ", ";
  getAddr()->printAsOperand(O, SlotTracker);
  VPValue *Mask = getMask();
  if (Mask) {
    O << ", ";
    Mask->printAsOperand(O, SlotTracker);
  }
  for (unsigned i = 0; i < IG->getFactor(); ++i)
    if (Instruction *I = IG->getMember(i))
      O << "\n" << Indent << "  " << VPlanIngredient(I) << " " << i;
}
#endif

void VPWidenCallRecipe::execute(VPTransformState &State) {
  State.ILV->widenCallInstruction(*cast<CallInst>(getUnderlyingInstr()), this,
                                  *this, State);
}

void VPWidenSelectRecipe::execute(VPTransformState &State) {
  State.ILV->widenSelectInstruction(*cast<SelectInst>(getUnderlyingInstr()),
                                    this, *this, InvariantCond, State);
}

void VPWidenRecipe::execute(VPTransformState &State) {
  State.ILV->widenInstruction(*getUnderlyingInstr(), this, *this, State);
}

void VPWidenGEPRecipe::execute(VPTransformState &State) {
  State.ILV->widenGEP(cast<GetElementPtrInst>(getUnderlyingInstr()), this,
                      *this, State.UF, State.VF, IsPtrLoopInvariant,
                      IsIndexLoopInvariant, State);
}

void VPWidenIntOrFpInductionRecipe::execute(VPTransformState &State) {
  assert(!State.Instance && "Int or FP induction being replicated.");
  State.ILV->widenIntOrFpInduction(IV, getStartValue()->getLiveInIRValue(),
                                   getTruncInst(), getVPValue(0),
                                   getCastValue(), State);
}

void VPWidenPHIRecipe::execute(VPTransformState &State) {
  State.ILV->widenPHIInstruction(cast<PHINode>(getUnderlyingValue()), RdxDesc,
                                 this, State);
}

void VPBlendRecipe::execute(VPTransformState &State) {
  State.ILV->setDebugLocFromInst(Phi, &State.Builder);
  // We know that all PHIs in non-header blocks are converted into
  // selects, so we don't have to worry about the insertion order and we
  // can just use the builder.
  // At this point we generate the predication tree. There may be
  // duplications since this is a simple recursive scan, but future
  // optimizations will clean it up.

  unsigned NumIncoming = getNumIncomingValues();

  // Generate a sequence of selects of the form:
  // SELECT(Mask3, In3,
  //        SELECT(Mask2, In2,
  //               SELECT(Mask1, In1,
  //                      In0)))
  // Note that Mask0 is never used: lanes for which no path reaches this phi and
  // are essentially undef are taken from In0.
  InnerLoopVectorizer::VectorParts Entry(State.UF);
  for (unsigned In = 0; In < NumIncoming; ++In) {
    for (unsigned Part = 0; Part < State.UF; ++Part) {
      // We might have single edge PHIs (blocks) - use an identity
      // 'select' for the first PHI operand.
      Value *In0 = State.get(getIncomingValue(In), Part);
      if (In == 0)
        Entry[Part] = In0; // Initialize with the first incoming value.
      else {
        // Select between the current value and the previous incoming edge
        // based on the incoming mask.
        Value *Cond = State.get(getMask(In), Part);
        Entry[Part] =
            State.Builder.CreateSelect(Cond, In0, Entry[Part], "predphi");
      }
    }
  }
  for (unsigned Part = 0; Part < State.UF; ++Part)
    State.set(this, Entry[Part], Part);
}

void VPInterleaveRecipe::execute(VPTransformState &State) {
  assert(!State.Instance && "Interleave group being replicated.");
  State.ILV->vectorizeInterleaveGroup(IG, definedValues(), State, getAddr(),
                                      getStoredValues(), getMask());
}

void VPReductionRecipe::execute(VPTransformState &State) {
  assert(!State.Instance && "Reduction being replicated.");
  Value *PrevInChain = State.get(getChainOp(), 0);
  for (unsigned Part = 0; Part < State.UF; ++Part) {
    RecurKind Kind = RdxDesc->getRecurrenceKind();
    bool IsOrdered = State.ILV->useOrderedReductions(*RdxDesc);
    Value *NewVecOp = State.get(getVecOp(), Part);
    if (VPValue *Cond = getCondOp()) {
      Value *NewCond = State.get(Cond, Part);
      VectorType *VecTy = cast<VectorType>(NewVecOp->getType());
      Constant *Iden = RecurrenceDescriptor::getRecurrenceIdentity(
          Kind, VecTy->getElementType(), RdxDesc->getFastMathFlags());
      Constant *IdenVec =
          ConstantVector::getSplat(VecTy->getElementCount(), Iden);
      Value *Select = State.Builder.CreateSelect(NewCond, NewVecOp, IdenVec);
      NewVecOp = Select;
    }
    Value *NewRed;
    Value *NextInChain;
    if (IsOrdered) {
      if (State.VF.isVector())
        NewRed = createOrderedReduction(State.Builder, *RdxDesc, NewVecOp,
                                        PrevInChain);
      else
        NewRed = State.Builder.CreateBinOp(
            (Instruction::BinaryOps)getUnderlyingInstr()->getOpcode(),
            PrevInChain, NewVecOp);
      PrevInChain = NewRed;
    } else {
      PrevInChain = State.get(getChainOp(), Part);
      NewRed = createTargetReduction(State.Builder, TTI, *RdxDesc, NewVecOp);
    }
    if (RecurrenceDescriptor::isMinMaxRecurrenceKind(Kind)) {
      NextInChain =
          createMinMaxOp(State.Builder, RdxDesc->getRecurrenceKind(),
                         NewRed, PrevInChain);
    } else if (IsOrdered)
      NextInChain = NewRed;
    else {
      NextInChain = State.Builder.CreateBinOp(
          (Instruction::BinaryOps)getUnderlyingInstr()->getOpcode(), NewRed,
          PrevInChain);
    }
    State.set(this, NextInChain, Part);
  }
}

void VPReplicateRecipe::execute(VPTransformState &State) {
  if (State.Instance) { // Generate a single instance.
    assert(!State.VF.isScalable() && "Can't scalarize a scalable vector");
    State.ILV->scalarizeInstruction(getUnderlyingInstr(), this, *this,
                                    *State.Instance, IsPredicated, State);
    // Insert scalar instance packing it into a vector.
    if (AlsoPack && State.VF.isVector()) {
      // If we're constructing lane 0, initialize to start from poison.
      if (State.Instance->Lane.isFirstLane()) {
        assert(!State.VF.isScalable() && "VF is assumed to be non scalable.");
        Value *Poison = PoisonValue::get(
            VectorType::get(getUnderlyingValue()->getType(), State.VF));
        State.set(this, Poison, State.Instance->Part);
      }
      State.ILV->packScalarIntoVectorValue(this, *State.Instance, State);
    }
    return;
  }

  // Generate scalar instances for all VF lanes of all UF parts, unless the
  // instruction is uniform inwhich case generate only the first lane for each
  // of the UF parts.
  unsigned EndLane = IsUniform ? 1 : State.VF.getKnownMinValue();
  assert((!State.VF.isScalable() || IsUniform) &&
         "Can't scalarize a scalable vector");
  for (unsigned Part = 0; Part < State.UF; ++Part)
    for (unsigned Lane = 0; Lane < EndLane; ++Lane)
      State.ILV->scalarizeInstruction(getUnderlyingInstr(), this, *this,
                                      VPIteration(Part, Lane), IsPredicated,
                                      State);
}

void VPBranchOnMaskRecipe::execute(VPTransformState &State) {
  assert(State.Instance && "Branch on Mask works only on single instance.");

  unsigned Part = State.Instance->Part;
  unsigned Lane = State.Instance->Lane.getKnownLane();

  Value *ConditionBit = nullptr;
  VPValue *BlockInMask = getMask();
  if (BlockInMask) {
    ConditionBit = State.get(BlockInMask, Part);
    if (ConditionBit->getType()->isVectorTy())
      ConditionBit = State.Builder.CreateExtractElement(
          ConditionBit, State.Builder.getInt32(Lane));
  } else // Block in mask is all-one.
    ConditionBit = State.Builder.getTrue();

  // Replace the temporary unreachable terminator with a new conditional branch,
  // whose two destinations will be set later when they are created.
  auto *CurrentTerminator = State.CFG.PrevBB->getTerminator();
  assert(isa<UnreachableInst>(CurrentTerminator) &&
         "Expected to replace unreachable terminator with conditional branch.");
  auto *CondBr = BranchInst::Create(State.CFG.PrevBB, nullptr, ConditionBit);
  CondBr->setSuccessor(0, nullptr);
  ReplaceInstWithInst(CurrentTerminator, CondBr);
}

void VPPredInstPHIRecipe::execute(VPTransformState &State) {
  assert(State.Instance && "Predicated instruction PHI works per instance.");
  Instruction *ScalarPredInst =
      cast<Instruction>(State.get(getOperand(0), *State.Instance));
  BasicBlock *PredicatedBB = ScalarPredInst->getParent();
  BasicBlock *PredicatingBB = PredicatedBB->getSinglePredecessor();
  assert(PredicatingBB && "Predicated block has no single predecessor.");
  assert(isa<VPReplicateRecipe>(getOperand(0)) &&
         "operand must be VPReplicateRecipe");

  // By current pack/unpack logic we need to generate only a single phi node: if
  // a vector value for the predicated instruction exists at this point it means
  // the instruction has vector users only, and a phi for the vector value is
  // needed. In this case the recipe of the predicated instruction is marked to
  // also do that packing, thereby "hoisting" the insert-element sequence.
  // Otherwise, a phi node for the scalar value is needed.
  unsigned Part = State.Instance->Part;
  if (State.hasVectorValue(getOperand(0), Part)) {
    Value *VectorValue = State.get(getOperand(0), Part);
    InsertElementInst *IEI = cast<InsertElementInst>(VectorValue);
    PHINode *VPhi = State.Builder.CreatePHI(IEI->getType(), 2);
    VPhi->addIncoming(IEI->getOperand(0), PredicatingBB); // Unmodified vector.
    VPhi->addIncoming(IEI, PredicatedBB); // New vector with inserted element.
    if (State.hasVectorValue(this, Part))
      State.reset(this, VPhi, Part);
    else
      State.set(this, VPhi, Part);
    // NOTE: Currently we need to update the value of the operand, so the next
    // predicated iteration inserts its generated value in the correct vector.
    State.reset(getOperand(0), VPhi, Part);
  } else {
    Type *PredInstType = getOperand(0)->getUnderlyingValue()->getType();
    PHINode *Phi = State.Builder.CreatePHI(PredInstType, 2);
    Phi->addIncoming(PoisonValue::get(ScalarPredInst->getType()),
                     PredicatingBB);
    Phi->addIncoming(ScalarPredInst, PredicatedBB);
    if (State.hasScalarValue(this, *State.Instance))
      State.reset(this, Phi, *State.Instance);
    else
      State.set(this, Phi, *State.Instance);
    // NOTE: Currently we need to update the value of the operand, so the next
    // predicated iteration inserts its generated value in the correct vector.
    State.reset(getOperand(0), Phi, *State.Instance);
  }
}

void VPWidenMemoryInstructionRecipe::execute(VPTransformState &State) {
  VPValue *StoredValue = isStore() ? getStoredValue() : nullptr;
  State.ILV->vectorizeMemoryInstruction(
      &Ingredient, State, StoredValue ? nullptr : getVPSingleValue(), getAddr(),
      StoredValue, getMask());
}

// Determine how to lower the scalar epilogue, which depends on 1) optimising
// for minimum code-size, 2) predicate compiler options, 3) loop hints forcing
// predication, and 4) a TTI hook that analyses whether the loop is suitable
// for predication.
static ScalarEpilogueLowering getScalarEpilogueLowering(
    Function *F, Loop *L, LoopVectorizeHints &Hints, ProfileSummaryInfo *PSI,
    BlockFrequencyInfo *BFI, TargetTransformInfo *TTI, TargetLibraryInfo *TLI,
    AssumptionCache *AC, LoopInfo *LI, ScalarEvolution *SE, DominatorTree *DT,
    LoopVectorizationLegality &LVL) {
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
  if (TTI->preferPredicateOverEpilogue(L, LI, *SE, *AC, TLI, DT,
                                       LVL.getLAI()))
    return CM_ScalarEpilogueNotNeededUsePredicate;

  return CM_ScalarEpilogueAllowed;
}

Value *VPTransformState::get(VPValue *Def, unsigned Part) {
  // If Values have been set for this Def return the one relevant for \p Part.
  if (hasVectorValue(Def, Part))
    return Data.PerPartOutput[Def][Part];

  if (!hasScalarValue(Def, {Part, 0})) {
    Value *IRV = Def->getLiveInIRValue();
    Value *B = ILV->getBroadcastInstrs(IRV);
    set(Def, B, Part);
    return B;
  }

  Value *ScalarValue = get(Def, {Part, 0});
  // If we aren't vectorizing, we can just copy the scalar map values over
  // to the vector map.
  if (VF.isScalar()) {
    set(Def, ScalarValue, Part);
    return ScalarValue;
  }

  auto *RepR = dyn_cast<VPReplicateRecipe>(Def);
  bool IsUniform = RepR && RepR->isUniform();

  unsigned LastLane = IsUniform ? 0 : VF.getKnownMinValue() - 1;
  // Check if there is a scalar value for the selected lane.
  if (!hasScalarValue(Def, {Part, LastLane})) {
    // At the moment, VPWidenIntOrFpInductionRecipes can also be uniform.
    assert(isa<VPWidenIntOrFpInductionRecipe>(Def->getDef()) &&
           "unexpected recipe found to be invariant");
    IsUniform = true;
    LastLane = 0;
  }

  auto *LastInst = cast<Instruction>(get(Def, {Part, LastLane}));
  // Set the insert point after the last scalarized instruction or after the
  // last PHI, if LastInst is a PHI. This ensures the insertelement sequence
  // will directly follow the scalar definitions.
  auto OldIP = Builder.saveIP();
  auto NewIP =
      isa<PHINode>(LastInst)
          ? BasicBlock::iterator(LastInst->getParent()->getFirstNonPHI())
          : std::next(BasicBlock::iterator(LastInst));
  Builder.SetInsertPoint(&*NewIP);

  // However, if we are vectorizing, we need to construct the vector values.
  // If the value is known to be uniform after vectorization, we can just
  // broadcast the scalar value corresponding to lane zero for each unroll
  // iteration. Otherwise, we construct the vector values using
  // insertelement instructions. Since the resulting vectors are stored in
  // State, we will only generate the insertelements once.
  Value *VectorValue = nullptr;
  if (IsUniform) {
    VectorValue = ILV->getBroadcastInstrs(ScalarValue);
    set(Def, VectorValue, Part);
  } else {
    // Initialize packing with insertelements to start from undef.
    assert(!VF.isScalable() && "VF is assumed to be non scalable.");
    Value *Undef = PoisonValue::get(VectorType::get(LastInst->getType(), VF));
    set(Def, Undef, Part);
    for (unsigned Lane = 0; Lane < VF.getKnownMinValue(); ++Lane)
      ILV->packScalarIntoVectorValue(Def, {Part, Lane}, *this);
    VectorValue = get(Def, Part);
  }
  Builder.restoreIP(OldIP);
  return VectorValue;
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

  ScalarEpilogueLowering SEL = getScalarEpilogueLowering(
      F, L, Hints, PSI, BFI, TTI, TLI, AC, LI, PSE.getSE(), DT, *LVL);

  LoopVectorizationCostModel CM(SEL, L, PSE, LI, LVL, *TTI, TLI, DB, AC, ORE, F,
                                &Hints, IAI);
  // Use the planner for outer loop vectorization.
  // TODO: CM is not used at this point inside the planner. Turn CM into an
  // optional argument if we don't need it in the future.
  LoopVectorizationPlanner LVP(L, LI, TLI, TTI, LVL, CM, IAI, PSE, Hints,
                               Requirements, ORE);

  // Get user vectorization factor.
  ElementCount UserVF = Hints.getWidth();

  // Plan how to best vectorize, return the best VF and its cost.
  const VectorizationFactor VF = LVP.planInVPlanNativePath(UserVF);

  // If we are stress testing VPlan builds, do not attempt to generate vector
  // code. Masked vector code generation support will follow soon.
  // Also, do not attempt to vectorize if no vector code will be produced.
  if (VPlanBuildStressTest || EnableVPlanPredication ||
      VectorizationFactor::Disabled() == VF)
    return false;

  LVP.setBestPlan(VF.Width, 1);

  {
    GeneratedRTChecks Checks(*PSE.getSE(), DT, LI,
                             F->getParent()->getDataLayout());
    InnerLoopVectorizer LB(L, PSE, LI, DT, TLI, TTI, AC, ORE, VF.Width, 1, LVL,
                           &CM, BFI, PSI, Checks);
    LLVM_DEBUG(dbgs() << "Vectorizing outer loop in \""
                      << L->getHeader()->getParent()->getName() << "\"\n");
    LVP.executePlan(LB, DT);
  }

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

LoopVectorizePass::LoopVectorizePass(LoopVectorizeOptions Opts)
    : InterleaveOnlyWhenForced(Opts.InterleaveOnlyWhenForced ||
                               !EnableLoopInterleaving),
      VectorizeOnlyWhenForced(Opts.VectorizeOnlyWhenForced ||
                              !EnableLoopVectorization) {}

bool LoopVectorizePass::processLoop(Loop *L) {
  assert((EnableVPlanNativePath || L->isInnermost()) &&
         "VPlan-native path is not enabled. Only process inner loops.");

#ifndef NDEBUG
  const std::string DebugLocStr = getDebugLocString(L);
#endif /* NDEBUG */

  LLVM_DEBUG(dbgs() << "\nLV: Checking a loop in \""
                    << L->getHeader()->getParent()->getName() << "\" from "
                    << DebugLocStr << "\n");

  LoopVectorizeHints Hints(L, InterleaveOnlyWhenForced, *ORE);

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
  LoopVectorizationLegality LVL(L, PSE, DT, TTI, TLI, AA, F, GetLAA, LI, ORE,
                                &Requirements, &Hints, DB, AC, BFI, PSI);
  if (!LVL.canVectorize(EnableVPlanNativePath)) {
    LLVM_DEBUG(dbgs() << "LV: Not vectorizing: Cannot prove legality.\n");
    Hints.emitRemarkWithHints();
    return false;
  }

  // Check the function attributes and profiles to find out if this function
  // should be optimized for size.
  ScalarEpilogueLowering SEL = getScalarEpilogueLowering(
      F, L, Hints, PSI, BFI, TTI, TLI, AC, LI, PSE.getSE(), DT, LVL);

  // Entrance to the VPlan-native vectorization path. Outer loops are processed
  // here. They may require CFG and instruction level transformations before
  // even evaluating whether vectorization is profitable. Since we cannot modify
  // the incoming IR, we need to build VPlan upfront in the vectorization
  // pipeline.
  if (!L->isInnermost())
    return processLoopInVPlanNativePath(L, PSE, LI, DT, &LVL, TTI, TLI, DB, AC,
                                        ORE, BFI, PSI, Hints, Requirements);

  assert(L->isInnermost() && "Inner loop expected.");

  // Check the loop for a trip count threshold: vectorize loops with a tiny trip
  // count by optimizing for size, to minimize overheads.
  auto ExpectedTC = getSmallBestKnownTC(*SE, L);
  if (ExpectedTC && *ExpectedTC < TinyTripCountVectorThreshold) {
    LLVM_DEBUG(dbgs() << "LV: Found a loop with a very small trip count. "
                      << "This loop is worth vectorizing only if no scalar "
                      << "iteration overheads are incurred.");
    if (Hints.getForce() == LoopVectorizeHints::FK_Enabled)
      LLVM_DEBUG(dbgs() << " But vectorizing was explicitly forced.\n");
    else {
      LLVM_DEBUG(dbgs() << "\n");
      SEL = CM_ScalarEpilogueNotAllowedLowTripLoop;
    }
  }

  // Check the function attributes to see if implicit floats are allowed.
  // FIXME: This check doesn't seem possibly correct -- what if the loop is
  // an integer loop and the vector instructions selected are purely integer
  // vector instructions?
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

  if (!LVL.canVectorizeFPMath(EnableStrictReductions)) {
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

  bool UseInterleaved = TTI->enableInterleavedAccessVectorization();
  InterleavedAccessInfo IAI(PSE, L, DT, LI, LVL.getLAI());

  // If an override option has been passed in for interleaved accesses, use it.
  if (EnableInterleavedMemAccesses.getNumOccurrences() > 0)
    UseInterleaved = EnableInterleavedMemAccesses;

  // Analyze interleaved memory accesses.
  if (UseInterleaved) {
    IAI.analyzeInterleaving(useMaskedInterleavedAccesses(*TTI));
  }

  // Use the cost model.
  LoopVectorizationCostModel CM(SEL, L, PSE, LI, &LVL, *TTI, TLI, DB, AC, ORE,
                                F, &Hints, IAI);
  CM.collectValuesToIgnore();

  // Use the planner for vectorization.
  LoopVectorizationPlanner LVP(L, LI, TLI, TTI, &LVL, CM, IAI, PSE, Hints,
                               Requirements, ORE);

  // Get user vectorization factor and interleave count.
  ElementCount UserVF = Hints.getWidth();
  unsigned UserIC = Hints.getInterleave();

  // Plan how to best vectorize, return the best VF and its cost.
  Optional<VectorizationFactor> MaybeVF = LVP.plan(UserVF, UserIC);

  VectorizationFactor VF = VectorizationFactor::Disabled();
  unsigned IC = 1;

  if (MaybeVF) {
    VF = *MaybeVF;
    // Select the interleave count.
    IC = CM.selectInterleaveCount(VF.Width, *VF.Cost.getValue());
  }

  // Identify the diagnostic messages that should be produced.
  std::pair<StringRef, std::string> VecDiagMsg, IntDiagMsg;
  bool VectorizeLoop = true, InterleaveLoop = true;
  if (VF.Width.isScalar()) {
    LLVM_DEBUG(dbgs() << "LV: Vectorization is possible but not beneficial.\n");
    VecDiagMsg = std::make_pair(
        "VectorizationNotBeneficial",
        "the cost-model indicates that vectorization is not beneficial");
    VectorizeLoop = false;
  }

  if (!MaybeVF && UserIC > 1) {
    // Tell the user interleaving was avoided up-front, despite being explicitly
    // requested.
    LLVM_DEBUG(dbgs() << "LV: Ignoring UserIC, because vectorization and "
                         "interleaving should be avoided up front\n");
    IntDiagMsg = std::make_pair(
        "InterleavingAvoided",
        "Ignoring UserIC, because interleaving was avoided up front");
    InterleaveLoop = false;
  } else if (IC == 1 && UserIC <= 1) {
    // Tell the user interleaving is not beneficial.
    LLVM_DEBUG(dbgs() << "LV: Interleaving is not beneficial.\n");
    IntDiagMsg = std::make_pair(
        "InterleavingNotBeneficial",
        "the cost-model indicates that interleaving is not beneficial");
    InterleaveLoop = false;
    if (UserIC == 1) {
      IntDiagMsg.first = "InterleavingNotBeneficialAndDisabled";
      IntDiagMsg.second +=
          " and is explicitly disabled or interleave count is set to 1";
    }
  } else if (IC > 1 && UserIC == 1) {
    // Tell the user interleaving is beneficial, but it explicitly disabled.
    LLVM_DEBUG(
        dbgs() << "LV: Interleaving is beneficial but is explicitly disabled.");
    IntDiagMsg = std::make_pair(
        "InterleavingBeneficialButDisabled",
        "the cost-model indicates that interleaving is beneficial "
        "but is explicitly disabled or interleave count is set to 1");
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
  } else if (!VectorizeLoop && InterleaveLoop) {
    LLVM_DEBUG(dbgs() << "LV: Interleave Count is " << IC << '\n');
    ORE->emit([&]() {
      return OptimizationRemarkAnalysis(VAPassName, VecDiagMsg.first,
                                        L->getStartLoc(), L->getHeader())
             << VecDiagMsg.second;
    });
  } else if (VectorizeLoop && !InterleaveLoop) {
    LLVM_DEBUG(dbgs() << "LV: Found a vectorizable loop (" << VF.Width
                      << ") in " << DebugLocStr << '\n');
    ORE->emit([&]() {
      return OptimizationRemarkAnalysis(LV_NAME, IntDiagMsg.first,
                                        L->getStartLoc(), L->getHeader())
             << IntDiagMsg.second;
    });
  } else if (VectorizeLoop && InterleaveLoop) {
    LLVM_DEBUG(dbgs() << "LV: Found a vectorizable loop (" << VF.Width
                      << ") in " << DebugLocStr << '\n');
    LLVM_DEBUG(dbgs() << "LV: Interleave Count is " << IC << '\n');
  }

  bool DisableRuntimeUnroll = false;
  MDNode *OrigLoopID = L->getLoopID();
  {
    // Optimistically generate runtime checks. Drop them if they turn out to not
    // be profitable. Limit the scope of Checks, so the cleanup happens
    // immediately after vector codegeneration is done.
    GeneratedRTChecks Checks(*PSE.getSE(), DT, LI,
                             F->getParent()->getDataLayout());
    if (!VF.Width.isScalar() || IC > 1)
      Checks.Create(L, *LVL.getLAI(), PSE.getUnionPredicate());
    LVP.setBestPlan(VF.Width, IC);

    using namespace ore;
    if (!VectorizeLoop) {
      assert(IC > 1 && "interleave count should not be 1 or 0");
      // If we decided that it is not legal to vectorize the loop, then
      // interleave it.
      InnerLoopUnroller Unroller(L, PSE, LI, DT, TLI, TTI, AC, ORE, IC, &LVL,
                                 &CM, BFI, PSI, Checks);
      LVP.executePlan(Unroller, DT);

      ORE->emit([&]() {
        return OptimizationRemark(LV_NAME, "Interleaved", L->getStartLoc(),
                                  L->getHeader())
               << "interleaved loop (interleaved count: "
               << NV("InterleaveCount", IC) << ")";
      });
    } else {
      // If we decided that it is *legal* to vectorize the loop, then do it.

      // Consider vectorizing the epilogue too if it's profitable.
      VectorizationFactor EpilogueVF =
          CM.selectEpilogueVectorizationFactor(VF.Width, LVP);
      if (EpilogueVF.Width.isVector()) {

        // The first pass vectorizes the main loop and creates a scalar epilogue
        // to be vectorized by executing the plan (potentially with a different
        // factor) again shortly afterwards.
        EpilogueLoopVectorizationInfo EPI(VF.Width.getKnownMinValue(), IC,
                                          EpilogueVF.Width.getKnownMinValue(),
                                          1);
        EpilogueVectorizerMainLoop MainILV(L, PSE, LI, DT, TLI, TTI, AC, ORE,
                                           EPI, &LVL, &CM, BFI, PSI, Checks);

        LVP.setBestPlan(EPI.MainLoopVF, EPI.MainLoopUF);
        LVP.executePlan(MainILV, DT);
        ++LoopsVectorized;

        simplifyLoop(L, DT, LI, SE, AC, nullptr, false /* PreserveLCSSA */);
        formLCSSARecursively(*L, *DT, LI, SE);

        // Second pass vectorizes the epilogue and adjusts the control flow
        // edges from the first pass.
        LVP.setBestPlan(EPI.EpilogueVF, EPI.EpilogueUF);
        EPI.MainLoopVF = EPI.EpilogueVF;
        EPI.MainLoopUF = EPI.EpilogueUF;
        EpilogueVectorizerEpilogueLoop EpilogILV(L, PSE, LI, DT, TLI, TTI, AC,
                                                 ORE, EPI, &LVL, &CM, BFI, PSI,
                                                 Checks);
        LVP.executePlan(EpilogILV, DT);
        ++LoopsEpilogueVectorized;

        if (!MainILV.areSafetyChecksAdded())
          DisableRuntimeUnroll = true;
      } else {
        InnerLoopVectorizer LB(L, PSE, LI, DT, TLI, TTI, AC, ORE, VF.Width, IC,
                               &LVL, &CM, BFI, PSI, Checks);
        LVP.executePlan(LB, DT);
        ++LoopsVectorized;

        // Add metadata to disable runtime unrolling a scalar loop when there
        // are no runtime checks about strides and memory. A scalar loop that is
        // rarely used is not worth unrolling.
        if (!LB.areSafetyChecksAdded())
          DisableRuntimeUnroll = true;
      }
      // Report the vectorization decision.
      ORE->emit([&]() {
        return OptimizationRemark(LV_NAME, "Vectorized", L->getStartLoc(),
                                  L->getHeader())
               << "vectorized loop (vectorization width: "
               << NV("VectorizationFactor", VF.Width)
               << ", interleaved count: " << NV("InterleaveCount", IC) << ")";
      });
    }

    if (ORE->allowExtraAnalysis(LV_NAME))
      checkMixedPrecision(L, ORE);
  }

  Optional<MDNode *> RemainderLoopID =
      makeFollowupLoopID(OrigLoopID, {LLVMLoopVectorizeFollowupAll,
                                      LLVMLoopVectorizeFollowupEpilogue});
  if (RemainderLoopID.hasValue()) {
    L->setLoopID(RemainderLoopID.getValue());
  } else {
    if (DisableRuntimeUnroll)
      AddRuntimeUnrollDisableMetaData(L);

    // Mark the loop as already vectorized to avoid vectorizing again.
    Hints.setAlreadyVectorized();
  }

  assert(!verifyFunction(*L->getHeader()->getParent(), &dbgs()));
  return true;
}

LoopVectorizeResult LoopVectorizePass::runImpl(
    Function &F, ScalarEvolution &SE_, LoopInfo &LI_, TargetTransformInfo &TTI_,
    DominatorTree &DT_, BlockFrequencyInfo &BFI_, TargetLibraryInfo *TLI_,
    DemandedBits &DB_, AAResults &AA_, AssumptionCache &AC_,
    std::function<const LoopAccessInfo &(Loop &)> &GetLAA_,
    OptimizationRemarkEmitter &ORE_, ProfileSummaryInfo *PSI_) {
  SE = &SE_;
  LI = &LI_;
  TTI = &TTI_;
  DT = &DT_;
  BFI = &BFI_;
  TLI = TLI_;
  AA = &AA_;
  AC = &AC_;
  GetLAA = &GetLAA_;
  DB = &DB_;
  ORE = &ORE_;
  PSI = PSI_;

  // Don't attempt if
  // 1. the target claims to have no vector registers, and
  // 2. interleaving won't help ILP.
  //
  // The second condition is necessary because, even if the target has no
  // vector registers, loop vectorization may still enable scalar
  // interleaving.
  if (!TTI->getNumberOfRegisters(TTI->getRegisterClassForType(true)) &&
      TTI->getMaxInterleaveFactor(1) < 2)
    return LoopVectorizeResult(false, false);

  bool Changed = false, CFGChanged = false;

  // The vectorizer requires loops to be in simplified form.
  // Since simplification may add new inner loops, it has to run before the
  // legality and profitability checks. This means running the loop vectorizer
  // will simplify all loops, regardless of whether anything end up being
  // vectorized.
  for (auto &L : *LI)
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
  }

  // Process each loop nest in the function.
  return LoopVectorizeResult(Changed, CFGChanged);
}

PreservedAnalyses LoopVectorizePass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
    auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
    auto &LI = AM.getResult<LoopAnalysis>(F);
    auto &TTI = AM.getResult<TargetIRAnalysis>(F);
    auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
    auto &BFI = AM.getResult<BlockFrequencyAnalysis>(F);
    auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
    auto &AA = AM.getResult<AAManager>(F);
    auto &AC = AM.getResult<AssumptionAnalysis>(F);
    auto &DB = AM.getResult<DemandedBitsAnalysis>(F);
    auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);
    MemorySSA *MSSA = EnableMSSALoopDependency
                          ? &AM.getResult<MemorySSAAnalysis>(F).getMSSA()
                          : nullptr;

    auto &LAM = AM.getResult<LoopAnalysisManagerFunctionProxy>(F).getManager();
    std::function<const LoopAccessInfo &(Loop &)> GetLAA =
        [&](Loop &L) -> const LoopAccessInfo & {
      LoopStandardAnalysisResults AR = {AA,  AC,  DT,      LI,  SE,
                                        TLI, TTI, nullptr, MSSA};
      return LAM.getResult<LoopAccessAnalysis>(L, AR);
    };
    auto &MAMProxy = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
    ProfileSummaryInfo *PSI =
        MAMProxy.getCachedResult<ProfileSummaryAnalysis>(*F.getParent());
    LoopVectorizeResult Result =
        runImpl(F, SE, LI, TTI, DT, BFI, &TLI, DB, AA, AC, GetLAA, ORE, PSI);
    if (!Result.MadeAnyChange)
      return PreservedAnalyses::all();
    PreservedAnalyses PA;

    // We currently do not preserve loopinfo/dominator analyses with outer loop
    // vectorization. Until this is addressed, mark these analyses as preserved
    // only for non-VPlan-native path.
    // TODO: Preserve Loop and Dominator analyses for VPlan-native path.
    if (!EnableVPlanNativePath) {
      PA.preserve<LoopAnalysis>();
      PA.preserve<DominatorTreeAnalysis>();
    }
    if (!Result.MadeCFGChange)
      PA.preserveSet<CFGAnalyses>();
    return PA;
}
