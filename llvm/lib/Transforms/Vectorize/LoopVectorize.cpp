//===- LoopVectorize.cpp - A Loop Vectorizer ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Transforms/Vectorize.h"
#include <algorithm>
#include <map>
#include <tuple>

using namespace llvm;
using namespace llvm::PatternMatch;

#define LV_NAME "loop-vectorize"
#define DEBUG_TYPE LV_NAME

STATISTIC(LoopsVectorized, "Number of loops vectorized");
STATISTIC(LoopsAnalyzed, "Number of loops analyzed for vectorization");

static cl::opt<bool>
    EnableIfConversion("enable-if-conversion", cl::init(true), cl::Hidden,
                       cl::desc("Enable if-conversion during vectorization."));

/// We don't vectorize loops with a known constant trip count below this number.
static cl::opt<unsigned> TinyTripCountVectorThreshold(
    "vectorizer-min-trip-count", cl::init(16), cl::Hidden,
    cl::desc("Don't vectorize loops with a constant "
             "trip count that is smaller than this "
             "value."));

static cl::opt<bool> MaximizeBandwidth(
    "vectorizer-maximize-bandwidth", cl::init(false), cl::Hidden,
    cl::desc("Maximize bandwidth when selecting vectorization factor which "
             "will be determined by the smallest type in loop."));

static cl::opt<bool> EnableInterleavedMemAccesses(
    "enable-interleaved-mem-accesses", cl::init(false), cl::Hidden,
    cl::desc("Enable vectorization on interleaved memory accesses in a loop"));

/// Maximum factor for an interleaved memory access.
static cl::opt<unsigned> MaxInterleaveGroupFactor(
    "max-interleave-group-factor", cl::Hidden,
    cl::desc("Maximum factor for an interleaved access group (default = 8)"),
    cl::init(8));

/// We don't interleave loops with a known constant trip count below this
/// number.
static const unsigned TinyTripCountInterleaveThreshold = 128;

static cl::opt<unsigned> ForceTargetNumScalarRegs(
    "force-target-num-scalar-regs", cl::init(0), cl::Hidden,
    cl::desc("A flag that overrides the target's number of scalar registers."));

static cl::opt<unsigned> ForceTargetNumVectorRegs(
    "force-target-num-vector-regs", cl::init(0), cl::Hidden,
    cl::desc("A flag that overrides the target's number of vector registers."));

/// Maximum vectorization interleave count.
static const unsigned MaxInterleaveFactor = 16;

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

static cl::opt<unsigned> SmallLoopCost(
    "small-loop-cost", cl::init(20), cl::Hidden,
    cl::desc(
        "The cost of a loop that is considered 'small' by the interleaver."));

static cl::opt<bool> LoopVectorizeWithBlockFrequency(
    "loop-vectorize-with-block-frequency", cl::init(false), cl::Hidden,
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

static cl::opt<unsigned> PragmaVectorizeMemoryCheckThreshold(
    "pragma-vectorize-memory-check-threshold", cl::init(128), cl::Hidden,
    cl::desc("The maximum allowed number of runtime memory checks with a "
             "vectorize(enable) pragma."));

static cl::opt<unsigned> VectorizeSCEVCheckThreshold(
    "vectorize-scev-check-threshold", cl::init(16), cl::Hidden,
    cl::desc("The maximum number of SCEV checks allowed."));

static cl::opt<unsigned> PragmaVectorizeSCEVCheckThreshold(
    "pragma-vectorize-scev-check-threshold", cl::init(128), cl::Hidden,
    cl::desc("The maximum number of SCEV checks allowed with a "
             "vectorize(enable) pragma"));

/// Create an analysis remark that explains why vectorization failed
///
/// \p PassName is the name of the pass (e.g. can be AlwaysPrint).  \p
/// RemarkName is the identifier for the remark.  If \p I is passed it is an
/// instruction that prevents vectorization.  Otherwise \p TheLoop is used for
/// the location of the remark.  \return the remark object that can be
/// streamed to.
static OptimizationRemarkAnalysis
createMissedAnalysis(const char *PassName, StringRef RemarkName, Loop *TheLoop,
                     Instruction *I = nullptr) {
  Value *CodeRegion = TheLoop->getHeader();
  DebugLoc DL = TheLoop->getStartLoc();

  if (I) {
    CodeRegion = I->getParent();
    // If there is no debug location attached to the instruction, revert back to
    // using the loop's.
    if (I->getDebugLoc())
      DL = I->getDebugLoc();
  }

  OptimizationRemarkAnalysis R(PassName, RemarkName, DL, CodeRegion);
  R << "loop not vectorized: ";
  return R;
}

namespace {

// Forward declarations.
class LoopVectorizeHints;
class LoopVectorizationLegality;
class LoopVectorizationCostModel;
class LoopVectorizationRequirements;

/// Returns true if the given loop body has a cycle, excluding the loop
/// itself.
static bool hasCyclesInLoopBody(const Loop &L) {
  if (!L.empty())
    return true;

  for (const auto &SCC :
       make_range(scc_iterator<Loop, LoopBodyTraits>::begin(L),
                  scc_iterator<Loop, LoopBodyTraits>::end(L))) {
    if (SCC.size() > 1) {
      DEBUG(dbgs() << "LVL: Detected a cycle in the loop body:\n");
      DEBUG(L.dump());
      return true;
    }
  }
  return false;
}

/// A helper function for converting Scalar types to vector types.
/// If the incoming type is void, we return void. If the VF is 1, we return
/// the scalar type.
static Type *ToVectorTy(Type *Scalar, unsigned VF) {
  if (Scalar->isVoidTy() || VF == 1)
    return Scalar;
  return VectorType::get(Scalar, VF);
}

/// A helper function that returns GEP instruction and knows to skip a
/// 'bitcast'. The 'bitcast' may be skipped if the source and the destination
/// pointee types of the 'bitcast' have the same size.
/// For example:
///   bitcast double** %var to i64* - can be skipped
///   bitcast double** %var to i8*  - can not
static GetElementPtrInst *getGEPInstruction(Value *Ptr) {

  if (isa<GetElementPtrInst>(Ptr))
    return cast<GetElementPtrInst>(Ptr);

  if (isa<BitCastInst>(Ptr) &&
      isa<GetElementPtrInst>(cast<BitCastInst>(Ptr)->getOperand(0))) {
    Type *BitcastTy = Ptr->getType();
    Type *GEPTy = cast<BitCastInst>(Ptr)->getSrcTy();
    if (!isa<PointerType>(BitcastTy) || !isa<PointerType>(GEPTy))
      return nullptr;
    Type *Pointee1Ty = cast<PointerType>(BitcastTy)->getPointerElementType();
    Type *Pointee2Ty = cast<PointerType>(GEPTy)->getPointerElementType();
    const DataLayout &DL = cast<BitCastInst>(Ptr)->getModule()->getDataLayout();
    if (DL.getTypeSizeInBits(Pointee1Ty) == DL.getTypeSizeInBits(Pointee2Ty))
      return cast<GetElementPtrInst>(cast<BitCastInst>(Ptr)->getOperand(0));
  }
  return nullptr;
}

// FIXME: The following helper functions have multiple implementations
// in the project. They can be effectively organized in a common Load/Store
// utilities unit.

/// A helper function that returns the pointer operand of a load or store
/// instruction.
static Value *getPointerOperand(Value *I) {
  if (auto *LI = dyn_cast<LoadInst>(I))
    return LI->getPointerOperand();
  if (auto *SI = dyn_cast<StoreInst>(I))
    return SI->getPointerOperand();
  return nullptr;
}

/// A helper function that returns the type of loaded or stored value.
static Type *getMemInstValueType(Value *I) {
  assert((isa<LoadInst>(I) || isa<StoreInst>(I)) &&
         "Expected Load or Store instruction");
  if (auto *LI = dyn_cast<LoadInst>(I))
    return LI->getType();
  return cast<StoreInst>(I)->getValueOperand()->getType();
}

/// A helper function that returns the alignment of load or store instruction.
static unsigned getMemInstAlignment(Value *I) {
  assert((isa<LoadInst>(I) || isa<StoreInst>(I)) &&
         "Expected Load or Store instruction");
  if (auto *LI = dyn_cast<LoadInst>(I))
    return LI->getAlignment();
  return cast<StoreInst>(I)->getAlignment();
}

/// A helper function that returns the address space of the pointer operand of
/// load or store instruction.
static unsigned getMemInstAddressSpace(Value *I) {
  assert((isa<LoadInst>(I) || isa<StoreInst>(I)) &&
         "Expected Load or Store instruction");
  if (auto *LI = dyn_cast<LoadInst>(I))
    return LI->getPointerAddressSpace();
  return cast<StoreInst>(I)->getPointerAddressSpace();
}

/// A helper function that returns true if the given type is irregular. The
/// type is irregular if its allocated size doesn't equal the store size of an
/// element of the corresponding vector type at the given vectorization factor.
static bool hasIrregularType(Type *Ty, const DataLayout &DL, unsigned VF) {

  // Determine if an array of VF elements of type Ty is "bitcast compatible"
  // with a <VF x Ty> vector.
  if (VF > 1) {
    auto *VectorTy = VectorType::get(Ty, VF);
    return VF * DL.getTypeAllocSize(Ty) != DL.getTypeStoreSize(VectorTy);
  }

  // If the vectorization factor is one, we just check if an array of type Ty
  // requires padding between elements.
  return DL.getTypeAllocSizeInBits(Ty) != DL.getTypeSizeInBits(Ty);
}

/// A helper function that returns the reciprocal of the block probability of
/// predicated blocks. If we return X, we are assuming the predicated block
/// will execute once for for every X iterations of the loop header.
///
/// TODO: We should use actual block probability here, if available. Currently,
///       we always assume predicated blocks have a 50% chance of executing.
static unsigned getReciprocalPredBlockProb() { return 2; }

/// A helper function that adds a 'fast' flag to floating-point operations.
static Value *addFastMathFlag(Value *V) {
  if (isa<FPMathOperator>(V)) {
    FastMathFlags Flags;
    Flags.setUnsafeAlgebra();
    cast<Instruction>(V)->setFastMathFlags(Flags);
  }
  return V;
}

/// A helper function that returns an integer or floating-point constant with
/// value C.
static Constant *getSignedIntOrFpConstant(Type *Ty, int64_t C) {
  return Ty->isIntegerTy() ? ConstantInt::getSigned(Ty, C)
                           : ConstantFP::get(Ty, C);
}

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
                      OptimizationRemarkEmitter *ORE, unsigned VecWidth,
                      unsigned UnrollFactor, LoopVectorizationLegality *LVL,
                      LoopVectorizationCostModel *CM)
      : OrigLoop(OrigLoop), PSE(PSE), LI(LI), DT(DT), TLI(TLI), TTI(TTI),
        AC(AC), ORE(ORE), VF(VecWidth), UF(UnrollFactor),
        Builder(PSE.getSE()->getContext()), Induction(nullptr),
        OldInduction(nullptr), VectorLoopValueMap(UnrollFactor, VecWidth),
        TripCount(nullptr), VectorTripCount(nullptr), Legal(LVL), Cost(CM),
        AddedSafetyChecks(false) {}

  // Perform the actual loop widening (vectorization).
  void vectorize() {
    // Create a new empty loop. Unlink the old loop and connect the new one.
    createEmptyLoop();
    // Widen each instruction in the old loop to a new one in the new loop.
    vectorizeLoop();
  }

  // Return true if any runtime check is added.
  bool areSafetyChecksAdded() { return AddedSafetyChecks; }

  virtual ~InnerLoopVectorizer() {}

protected:
  /// A small list of PHINodes.
  typedef SmallVector<PHINode *, 4> PhiVector;

  /// A type for vectorized values in the new loop. Each value from the
  /// original loop, when vectorized, is represented by UF vector values in the
  /// new unrolled loop, where UF is the unroll factor.
  typedef SmallVector<Value *, 2> VectorParts;

  /// A type for scalarized values in the new loop. Each value from the
  /// original loop, when scalarized, is represented by UF x VF scalar values
  /// in the new unrolled loop, where UF is the unroll factor and VF is the
  /// vectorization factor.
  typedef SmallVector<SmallVector<Value *, 4>, 2> ScalarParts;

  // When we if-convert we need to create edge masks. We have to cache values
  // so that we don't end up with exponential recursion/IR.
  typedef DenseMap<std::pair<BasicBlock *, BasicBlock *>, VectorParts>
      EdgeMaskCache;

  /// Create an empty loop, based on the loop ranges of the old loop.
  void createEmptyLoop();

  /// Set up the values of the IVs correctly when exiting the vector loop.
  void fixupIVUsers(PHINode *OrigPhi, const InductionDescriptor &II,
                    Value *CountRoundDown, Value *EndValue,
                    BasicBlock *MiddleBlock);

  /// Create a new induction variable inside L.
  PHINode *createInductionVariable(Loop *L, Value *Start, Value *End,
                                   Value *Step, Instruction *DL);
  /// Copy and widen the instructions from the old loop.
  virtual void vectorizeLoop();

  /// Fix a first-order recurrence. This is the second phase of vectorizing
  /// this phi node.
  void fixFirstOrderRecurrence(PHINode *Phi);

  /// \brief The Loop exit block may have single value PHI nodes where the
  /// incoming value is 'Undef'. While vectorizing we only handled real values
  /// that were defined inside the loop. Here we fix the 'undef case'.
  /// See PR14725.
  void fixLCSSAPHIs();

  /// Iteratively sink the scalarized operands of a predicated instruction into
  /// the block that was created for it.
  void sinkScalarOperands(Instruction *PredInst);

  /// Predicate conditional instructions that require predication on their
  /// respective conditions.
  void predicateInstructions();

  /// Collect the instructions from the original loop that would be trivially
  /// dead in the vectorized loop if generated.
  void collectTriviallyDeadInstructions();

  /// Shrinks vector element sizes to the smallest bitwidth they can be legally
  /// represented as.
  void truncateToMinimalBitwidths();

  /// A helper function that computes the predicate of the block BB, assuming
  /// that the header block of the loop is set to True. It returns the *entry*
  /// mask for the block BB.
  VectorParts createBlockInMask(BasicBlock *BB);
  /// A helper function that computes the predicate of the edge between SRC
  /// and DST.
  VectorParts createEdgeMask(BasicBlock *Src, BasicBlock *Dst);

  /// A helper function to vectorize a single BB within the innermost loop.
  void vectorizeBlockInLoop(BasicBlock *BB, PhiVector *PV);

  /// Vectorize a single PHINode in a block. This method handles the induction
  /// variable canonicalization. It supports both VF = 1 for unrolled loops and
  /// arbitrary length vectors.
  void widenPHIInstruction(Instruction *PN, unsigned UF, unsigned VF,
                           PhiVector *PV);

  /// Insert the new loop to the loop hierarchy and pass manager
  /// and update the analysis passes.
  void updateAnalysis();

  /// This instruction is un-vectorizable. Implement it as a sequence
  /// of scalars. If \p IfPredicateInstr is true we need to 'hide' each
  /// scalarized instruction behind an if block predicated on the control
  /// dependence of the instruction.
  virtual void scalarizeInstruction(Instruction *Instr,
                                    bool IfPredicateInstr = false);

  /// Vectorize Load and Store instructions,
  virtual void vectorizeMemoryInstruction(Instruction *Instr);

  /// Create a broadcast instruction. This method generates a broadcast
  /// instruction (shuffle) for loop invariant values and for the induction
  /// value. If this is the induction variable then we extend it to N, N+1, ...
  /// this is needed because each iteration in the loop corresponds to a SIMD
  /// element.
  virtual Value *getBroadcastInstrs(Value *V);

  /// This function adds (StartIdx, StartIdx + Step, StartIdx + 2*Step, ...)
  /// to each vector element of Val. The sequence starts at StartIndex.
  /// \p Opcode is relevant for FP induction variable.
  virtual Value *getStepVector(Value *Val, int StartIdx, Value *Step,
                               Instruction::BinaryOps Opcode =
                               Instruction::BinaryOpsEnd);

  /// Compute scalar induction steps. \p ScalarIV is the scalar induction
  /// variable on which to base the steps, \p Step is the size of the step, and
  /// \p EntryVal is the value from the original loop that maps to the steps.
  /// Note that \p EntryVal doesn't have to be an induction variable (e.g., it
  /// can be a truncate instruction).
  void buildScalarSteps(Value *ScalarIV, Value *Step, Value *EntryVal,
                        const InductionDescriptor &ID);

  /// Create a vector induction phi node based on an existing scalar one. \p
  /// EntryVal is the value from the original loop that maps to the vector phi
  /// node, and \p Step is the loop-invariant step. If \p EntryVal is a
  /// truncate instruction, instead of widening the original IV, we widen a
  /// version of the IV truncated to \p EntryVal's type.
  void createVectorIntOrFpInductionPHI(const InductionDescriptor &II,
                                       Value *Step, Instruction *EntryVal);

  /// Widen an integer or floating-point induction variable \p IV. If \p Trunc
  /// is provided, the integer induction variable will first be truncated to
  /// the corresponding type.
  void widenIntOrFpInduction(PHINode *IV, TruncInst *Trunc = nullptr);

  /// Returns true if an instruction \p I should be scalarized instead of
  /// vectorized for the chosen vectorization factor.
  bool shouldScalarizeInstruction(Instruction *I) const;

  /// Returns true if we should generate a scalar version of \p IV.
  bool needsScalarInduction(Instruction *IV) const;

  /// Return a constant reference to the VectorParts corresponding to \p V from
  /// the original loop. If the value has already been vectorized, the
  /// corresponding vector entry in VectorLoopValueMap is returned. If,
  /// however, the value has a scalar entry in VectorLoopValueMap, we construct
  /// new vector values on-demand by inserting the scalar values into vectors
  /// with an insertelement sequence. If the value has been neither vectorized
  /// nor scalarized, it must be loop invariant, so we simply broadcast the
  /// value into vectors.
  const VectorParts &getVectorValue(Value *V);

  /// Return a value in the new loop corresponding to \p V from the original
  /// loop at unroll index \p Part and vector index \p Lane. If the value has
  /// been vectorized but not scalarized, the necessary extractelement
  /// instruction will be generated.
  Value *getScalarValue(Value *V, unsigned Part, unsigned Lane);

  /// Try to vectorize the interleaved access group that \p Instr belongs to.
  void vectorizeInterleaveGroup(Instruction *Instr);

  /// Generate a shuffle sequence that will reverse the vector Vec.
  virtual Value *reverseVector(Value *Vec);

  /// Returns (and creates if needed) the original loop trip count.
  Value *getOrCreateTripCount(Loop *NewLoop);

  /// Returns (and creates if needed) the trip count of the widened loop.
  Value *getOrCreateVectorTripCount(Loop *NewLoop);

  /// Emit a bypass check to see if the trip count would overflow, or we
  /// wouldn't have enough iterations to execute one vector loop.
  void emitMinimumIterationCountCheck(Loop *L, BasicBlock *Bypass);
  /// Emit a bypass check to see if the vector trip count is nonzero.
  void emitVectorLoopEnteredCheck(Loop *L, BasicBlock *Bypass);
  /// Emit a bypass check to see if all of the SCEV assumptions we've
  /// had to make are correct.
  void emitSCEVChecks(Loop *L, BasicBlock *Bypass);
  /// Emit bypass checks to check any memory assumptions we may have made.
  void emitMemRuntimeChecks(Loop *L, BasicBlock *Bypass);

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

  /// \brief Similar to the previous function but it adds the metadata to a
  /// vector of instructions.
  void addMetadata(ArrayRef<Value *> To, Instruction *From);

  /// \brief Set the debug location in the builder using the debug location in
  /// the instruction.
  void setDebugLocFromInst(IRBuilder<> &B, const Value *Ptr);

  /// This is a helper class for maintaining vectorization state. It's used for
  /// mapping values from the original loop to their corresponding values in
  /// the new loop. Two mappings are maintained: one for vectorized values and
  /// one for scalarized values. Vectorized values are represented with UF
  /// vector values in the new loop, and scalarized values are represented with
  /// UF x VF scalar values in the new loop. UF and VF are the unroll and
  /// vectorization factors, respectively.
  ///
  /// Entries can be added to either map with initVector and initScalar, which
  /// initialize and return a constant reference to the new entry. If a
  /// non-constant reference to a vector entry is required, getVector can be
  /// used to retrieve a mutable entry. We currently directly modify the mapped
  /// values during "fix-up" operations that occur once the first phase of
  /// widening is complete. These operations include type truncation and the
  /// second phase of recurrence widening.
  ///
  /// Otherwise, entries from either map should be accessed using the
  /// getVectorValue or getScalarValue functions from InnerLoopVectorizer.
  /// getVectorValue and getScalarValue coordinate to generate a vector or
  /// scalar value on-demand if one is not yet available. When vectorizing a
  /// loop, we visit the definition of an instruction before its uses. When
  /// visiting the definition, we either vectorize or scalarize the
  /// instruction, creating an entry for it in the corresponding map. (In some
  /// cases, such as induction variables, we will create both vector and scalar
  /// entries.) Then, as we encounter uses of the definition, we derive values
  /// for each scalar or vector use unless such a value is already available.
  /// For example, if we scalarize a definition and one of its uses is vector,
  /// we build the required vector on-demand with an insertelement sequence
  /// when visiting the use. Otherwise, if the use is scalar, we can use the
  /// existing scalar definition.
  struct ValueMap {

    /// Construct an empty map with the given unroll and vectorization factors.
    ValueMap(unsigned UnrollFactor, unsigned VecWidth)
        : UF(UnrollFactor), VF(VecWidth) {
      // The unroll and vectorization factors are only used in asserts builds
      // to verify map entries are sized appropriately.
      (void)UF;
      (void)VF;
    }

    /// \return True if the map has a vector entry for \p Key.
    bool hasVector(Value *Key) const { return VectorMapStorage.count(Key); }

    /// \return True if the map has a scalar entry for \p Key.
    bool hasScalar(Value *Key) const { return ScalarMapStorage.count(Key); }

    /// \brief Map \p Key to the given VectorParts \p Entry, and return a
    /// constant reference to the new vector map entry. The given key should
    /// not already be in the map, and the given VectorParts should be
    /// correctly sized for the current unroll factor.
    const VectorParts &initVector(Value *Key, const VectorParts &Entry) {
      assert(!hasVector(Key) && "Vector entry already initialized");
      assert(Entry.size() == UF && "VectorParts has wrong dimensions");
      VectorMapStorage[Key] = Entry;
      return VectorMapStorage[Key];
    }

    /// \brief Map \p Key to the given ScalarParts \p Entry, and return a
    /// constant reference to the new scalar map entry. The given key should
    /// not already be in the map, and the given ScalarParts should be
    /// correctly sized for the current unroll and vectorization factors.
    const ScalarParts &initScalar(Value *Key, const ScalarParts &Entry) {
      assert(!hasScalar(Key) && "Scalar entry already initialized");
      assert(Entry.size() == UF &&
             all_of(make_range(Entry.begin(), Entry.end()),
                    [&](const SmallVectorImpl<Value *> &Values) -> bool {
                      return Values.size() == VF;
                    }) &&
             "ScalarParts has wrong dimensions");
      ScalarMapStorage[Key] = Entry;
      return ScalarMapStorage[Key];
    }

    /// \return A reference to the vector map entry corresponding to \p Key.
    /// The key should already be in the map. This function should only be used
    /// when it's necessary to update values that have already been vectorized.
    /// This is the case for "fix-up" operations including type truncation and
    /// the second phase of recurrence vectorization. If a non-const reference
    /// isn't required, getVectorValue should be used instead.
    VectorParts &getVector(Value *Key) {
      assert(hasVector(Key) && "Vector entry not initialized");
      return VectorMapStorage.find(Key)->second;
    }

    /// Retrieve an entry from the vector or scalar maps. The preferred way to
    /// access an existing mapped entry is with getVectorValue or
    /// getScalarValue from InnerLoopVectorizer. Until those functions can be
    /// moved inside ValueMap, we have to declare them as friends.
    friend const VectorParts &InnerLoopVectorizer::getVectorValue(Value *V);
    friend Value *InnerLoopVectorizer::getScalarValue(Value *V, unsigned Part,
                                                      unsigned Lane);

  private:
    /// The unroll factor. Each entry in the vector map contains UF vector
    /// values.
    unsigned UF;

    /// The vectorization factor. Each entry in the scalar map contains UF x VF
    /// scalar values.
    unsigned VF;

    /// The vector and scalar map storage. We use std::map and not DenseMap
    /// because insertions to DenseMap invalidate its iterators.
    std::map<Value *, VectorParts> VectorMapStorage;
    std::map<Value *, ScalarParts> ScalarMapStorage;
  };

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
  AliasAnalysis *AA;
  /// Target Library Info.
  const TargetLibraryInfo *TLI;
  /// Target Transform Info.
  const TargetTransformInfo *TTI;
  /// Assumption Cache.
  AssumptionCache *AC;
  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;

  /// \brief LoopVersioning.  It's only set up (non-null) if memchecks were
  /// used.
  ///
  /// This is currently only used to add no-alias metadata based on the
  /// memchecks.  The actually versioning is performed manually.
  std::unique_ptr<LoopVersioning> LVer;

  /// The vectorization SIMD factor to use. Each vector will have this many
  /// vector elements.
  unsigned VF;

protected:
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
  /// The ExitBlock of the scalar loop.
  BasicBlock *LoopExitBlock;
  /// The vector loop body.
  BasicBlock *LoopVectorBody;
  /// The scalar loop body.
  BasicBlock *LoopScalarBody;
  /// A list of all bypass blocks. The first block is the entry of the loop.
  SmallVector<BasicBlock *, 4> LoopBypassBlocks;

  /// The new Induction variable which was added to the new block.
  PHINode *Induction;
  /// The induction variable of the old basic block.
  PHINode *OldInduction;

  /// Maps values from the original loop to their corresponding values in the
  /// vectorized loop. A key value can map to either vector values, scalar
  /// values or both kinds of values, depending on whether the key was
  /// vectorized and scalarized.
  ValueMap VectorLoopValueMap;

  /// Store instructions that should be predicated, as a pair
  ///   <StoreInst, Predicate>
  SmallVector<std::pair<Instruction *, Value *>, 4> PredicatedInstructions;
  EdgeMaskCache MaskCache;
  /// Trip count of the original loop.
  Value *TripCount;
  /// Trip count of the widened loop (TripCount - TripCount % (VF*UF))
  Value *VectorTripCount;

  /// The legality analysis.
  LoopVectorizationLegality *Legal;

  /// The profitablity analysis.
  LoopVectorizationCostModel *Cost;

  // Record whether runtime checks are added.
  bool AddedSafetyChecks;

  // Holds instructions from the original loop whose counterparts in the
  // vectorized loop would be trivially dead if generated. For example,
  // original induction update instructions can become dead because we
  // separately emit induction "steps" when generating code for the new loop.
  // Similarly, we create a new latch condition when setting up the structure
  // of the new loop, so the old one can become dead.
  SmallPtrSet<Instruction *, 4> DeadInstructions;

  // Holds the end values for each induction variable. We save the end values
  // so we can later fix-up the external users of the induction variables.
  DenseMap<PHINode *, Value *> IVEndValues;
};

class InnerLoopUnroller : public InnerLoopVectorizer {
public:
  InnerLoopUnroller(Loop *OrigLoop, PredicatedScalarEvolution &PSE,
                    LoopInfo *LI, DominatorTree *DT,
                    const TargetLibraryInfo *TLI,
                    const TargetTransformInfo *TTI, AssumptionCache *AC,
                    OptimizationRemarkEmitter *ORE, unsigned UnrollFactor,
                    LoopVectorizationLegality *LVL,
                    LoopVectorizationCostModel *CM)
      : InnerLoopVectorizer(OrigLoop, PSE, LI, DT, TLI, TTI, AC, ORE, 1,
                            UnrollFactor, LVL, CM) {}

private:
  void scalarizeInstruction(Instruction *Instr,
                            bool IfPredicateInstr = false) override;
  void vectorizeMemoryInstruction(Instruction *Instr) override;
  Value *getBroadcastInstrs(Value *V) override;
  Value *getStepVector(Value *Val, int StartIdx, Value *Step,
                       Instruction::BinaryOps Opcode =
                       Instruction::BinaryOpsEnd) override;
  Value *reverseVector(Value *Vec) override;
};

/// \brief Look for a meaningful debug location on the instruction or it's
/// operands.
static Instruction *getDebugLocFromInstOrOperands(Instruction *I) {
  if (!I)
    return I;

  DebugLoc Empty;
  if (I->getDebugLoc() != Empty)
    return I;

  for (User::op_iterator OI = I->op_begin(), OE = I->op_end(); OI != OE; ++OI) {
    if (Instruction *OpInst = dyn_cast<Instruction>(*OI))
      if (OpInst->getDebugLoc() != Empty)
        return OpInst;
  }

  return I;
}

void InnerLoopVectorizer::setDebugLocFromInst(IRBuilder<> &B, const Value *Ptr) {
  if (const Instruction *Inst = dyn_cast_or_null<Instruction>(Ptr)) {
    const DILocation *DIL = Inst->getDebugLoc();
    if (DIL && Inst->getFunction()->isDebugInfoForProfiling())
      B.SetCurrentDebugLocation(DIL->cloneWithDuplicationFactor(UF * VF));
    else
      B.SetCurrentDebugLocation(DIL);
  } else
    B.SetCurrentDebugLocation(DebugLoc());
}

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

/// \brief The group of interleaved loads/stores sharing the same stride and
/// close to each other.
///
/// Each member in this group has an index starting from 0, and the largest
/// index should be less than interleaved factor, which is equal to the absolute
/// value of the access's stride.
///
/// E.g. An interleaved load group of factor 4:
///        for (unsigned i = 0; i < 1024; i+=4) {
///          a = A[i];                           // Member of index 0
///          b = A[i+1];                         // Member of index 1
///          d = A[i+3];                         // Member of index 3
///          ...
///        }
///
///      An interleaved store group of factor 4:
///        for (unsigned i = 0; i < 1024; i+=4) {
///          ...
///          A[i]   = a;                         // Member of index 0
///          A[i+1] = b;                         // Member of index 1
///          A[i+2] = c;                         // Member of index 2
///          A[i+3] = d;                         // Member of index 3
///        }
///
/// Note: the interleaved load group could have gaps (missing members), but
/// the interleaved store group doesn't allow gaps.
class InterleaveGroup {
public:
  InterleaveGroup(Instruction *Instr, int Stride, unsigned Align)
      : Align(Align), SmallestKey(0), LargestKey(0), InsertPos(Instr) {
    assert(Align && "The alignment should be non-zero");

    Factor = std::abs(Stride);
    assert(Factor > 1 && "Invalid interleave factor");

    Reverse = Stride < 0;
    Members[0] = Instr;
  }

  bool isReverse() const { return Reverse; }
  unsigned getFactor() const { return Factor; }
  unsigned getAlignment() const { return Align; }
  unsigned getNumMembers() const { return Members.size(); }

  /// \brief Try to insert a new member \p Instr with index \p Index and
  /// alignment \p NewAlign. The index is related to the leader and it could be
  /// negative if it is the new leader.
  ///
  /// \returns false if the instruction doesn't belong to the group.
  bool insertMember(Instruction *Instr, int Index, unsigned NewAlign) {
    assert(NewAlign && "The new member's alignment should be non-zero");

    int Key = Index + SmallestKey;

    // Skip if there is already a member with the same index.
    if (Members.count(Key))
      return false;

    if (Key > LargestKey) {
      // The largest index is always less than the interleave factor.
      if (Index >= static_cast<int>(Factor))
        return false;

      LargestKey = Key;
    } else if (Key < SmallestKey) {
      // The largest index is always less than the interleave factor.
      if (LargestKey - Key >= static_cast<int>(Factor))
        return false;

      SmallestKey = Key;
    }

    // It's always safe to select the minimum alignment.
    Align = std::min(Align, NewAlign);
    Members[Key] = Instr;
    return true;
  }

  /// \brief Get the member with the given index \p Index
  ///
  /// \returns nullptr if contains no such member.
  Instruction *getMember(unsigned Index) const {
    int Key = SmallestKey + Index;
    if (!Members.count(Key))
      return nullptr;

    return Members.find(Key)->second;
  }

  /// \brief Get the index for the given member. Unlike the key in the member
  /// map, the index starts from 0.
  unsigned getIndex(Instruction *Instr) const {
    for (auto I : Members)
      if (I.second == Instr)
        return I.first - SmallestKey;

    llvm_unreachable("InterleaveGroup contains no such member");
  }

  Instruction *getInsertPos() const { return InsertPos; }
  void setInsertPos(Instruction *Inst) { InsertPos = Inst; }

private:
  unsigned Factor; // Interleave Factor.
  bool Reverse;
  unsigned Align;
  DenseMap<int, Instruction *> Members;
  int SmallestKey;
  int LargestKey;

  // To avoid breaking dependences, vectorized instructions of an interleave
  // group should be inserted at either the first load or the last store in
  // program order.
  //
  // E.g. %even = load i32             // Insert Position
  //      %add = add i32 %even         // Use of %even
  //      %odd = load i32
  //
  //      store i32 %even
  //      %odd = add i32               // Def of %odd
  //      store i32 %odd               // Insert Position
  Instruction *InsertPos;
};

/// \brief Drive the analysis of interleaved memory accesses in the loop.
///
/// Use this class to analyze interleaved accesses only when we can vectorize
/// a loop. Otherwise it's meaningless to do analysis as the vectorization
/// on interleaved accesses is unsafe.
///
/// The analysis collects interleave groups and records the relationships
/// between the member and the group in a map.
class InterleavedAccessInfo {
public:
  InterleavedAccessInfo(PredicatedScalarEvolution &PSE, Loop *L,
                        DominatorTree *DT, LoopInfo *LI)
      : PSE(PSE), TheLoop(L), DT(DT), LI(LI), LAI(nullptr),
        RequiresScalarEpilogue(false) {}

  ~InterleavedAccessInfo() {
    SmallSet<InterleaveGroup *, 4> DelSet;
    // Avoid releasing a pointer twice.
    for (auto &I : InterleaveGroupMap)
      DelSet.insert(I.second);
    for (auto *Ptr : DelSet)
      delete Ptr;
  }

  /// \brief Analyze the interleaved accesses and collect them in interleave
  /// groups. Substitute symbolic strides using \p Strides.
  void analyzeInterleaving(const ValueToValueMap &Strides);

  /// \brief Check if \p Instr belongs to any interleave group.
  bool isInterleaved(Instruction *Instr) const {
    return InterleaveGroupMap.count(Instr);
  }

  /// \brief Return the maximum interleave factor of all interleaved groups.
  unsigned getMaxInterleaveFactor() const {
    unsigned MaxFactor = 1;
    for (auto &Entry : InterleaveGroupMap)
      MaxFactor = std::max(MaxFactor, Entry.second->getFactor());
    return MaxFactor;
  }

  /// \brief Get the interleave group that \p Instr belongs to.
  ///
  /// \returns nullptr if doesn't have such group.
  InterleaveGroup *getInterleaveGroup(Instruction *Instr) const {
    if (InterleaveGroupMap.count(Instr))
      return InterleaveGroupMap.find(Instr)->second;
    return nullptr;
  }

  /// \brief Returns true if an interleaved group that may access memory
  /// out-of-bounds requires a scalar epilogue iteration for correctness.
  bool requiresScalarEpilogue() const { return RequiresScalarEpilogue; }

  /// \brief Initialize the LoopAccessInfo used for dependence checking.
  void setLAI(const LoopAccessInfo *Info) { LAI = Info; }

private:
  /// A wrapper around ScalarEvolution, used to add runtime SCEV checks.
  /// Simplifies SCEV expressions in the context of existing SCEV assumptions.
  /// The interleaved access analysis can also add new predicates (for example
  /// by versioning strides of pointers).
  PredicatedScalarEvolution &PSE;
  Loop *TheLoop;
  DominatorTree *DT;
  LoopInfo *LI;
  const LoopAccessInfo *LAI;

  /// True if the loop may contain non-reversed interleaved groups with
  /// out-of-bounds accesses. We ensure we don't speculatively access memory
  /// out-of-bounds by executing at least one scalar epilogue iteration.
  bool RequiresScalarEpilogue;

  /// Holds the relationships between the members and the interleave group.
  DenseMap<Instruction *, InterleaveGroup *> InterleaveGroupMap;

  /// Holds dependences among the memory accesses in the loop. It maps a source
  /// access to a set of dependent sink accesses.
  DenseMap<Instruction *, SmallPtrSet<Instruction *, 2>> Dependences;

  /// \brief The descriptor for a strided memory access.
  struct StrideDescriptor {
    StrideDescriptor(int64_t Stride, const SCEV *Scev, uint64_t Size,
                     unsigned Align)
        : Stride(Stride), Scev(Scev), Size(Size), Align(Align) {}

    StrideDescriptor() = default;

    // The access's stride. It is negative for a reverse access.
    int64_t Stride = 0;
    const SCEV *Scev = nullptr; // The scalar expression of this access
    uint64_t Size = 0;          // The size of the memory object.
    unsigned Align = 0;         // The alignment of this access.
  };

  /// \brief A type for holding instructions and their stride descriptors.
  typedef std::pair<Instruction *, StrideDescriptor> StrideEntry;

  /// \brief Create a new interleave group with the given instruction \p Instr,
  /// stride \p Stride and alignment \p Align.
  ///
  /// \returns the newly created interleave group.
  InterleaveGroup *createInterleaveGroup(Instruction *Instr, int Stride,
                                         unsigned Align) {
    assert(!InterleaveGroupMap.count(Instr) &&
           "Already in an interleaved access group");
    InterleaveGroupMap[Instr] = new InterleaveGroup(Instr, Stride, Align);
    return InterleaveGroupMap[Instr];
  }

  /// \brief Release the group and remove all the relationships.
  void releaseGroup(InterleaveGroup *Group) {
    for (unsigned i = 0; i < Group->getFactor(); i++)
      if (Instruction *Member = Group->getMember(i))
        InterleaveGroupMap.erase(Member);

    delete Group;
  }

  /// \brief Collect all the accesses with a constant stride in program order.
  void collectConstStrideAccesses(
      MapVector<Instruction *, StrideDescriptor> &AccessStrideInfo,
      const ValueToValueMap &Strides);

  /// \brief Returns true if \p Stride is allowed in an interleaved group.
  static bool isStrided(int Stride) {
    unsigned Factor = std::abs(Stride);
    return Factor >= 2 && Factor <= MaxInterleaveGroupFactor;
  }

  /// \brief Returns true if \p BB is a predicated block.
  bool isPredicated(BasicBlock *BB) const {
    return LoopAccessInfo::blockNeedsPredication(BB, TheLoop, DT);
  }

  /// \brief Returns true if LoopAccessInfo can be used for dependence queries.
  bool areDependencesValid() const {
    return LAI && LAI->getDepChecker().getDependences();
  }

  /// \brief Returns true if memory accesses \p A and \p B can be reordered, if
  /// necessary, when constructing interleaved groups.
  ///
  /// \p A must precede \p B in program order. We return false if reordering is
  /// not necessary or is prevented because \p A and \p B may be dependent.
  bool canReorderMemAccessesForInterleavedGroups(StrideEntry *A,
                                                 StrideEntry *B) const {

    // Code motion for interleaved accesses can potentially hoist strided loads
    // and sink strided stores. The code below checks the legality of the
    // following two conditions:
    //
    // 1. Potentially moving a strided load (B) before any store (A) that
    //    precedes B, or
    //
    // 2. Potentially moving a strided store (A) after any load or store (B)
    //    that A precedes.
    //
    // It's legal to reorder A and B if we know there isn't a dependence from A
    // to B. Note that this determination is conservative since some
    // dependences could potentially be reordered safely.

    // A is potentially the source of a dependence.
    auto *Src = A->first;
    auto SrcDes = A->second;

    // B is potentially the sink of a dependence.
    auto *Sink = B->first;
    auto SinkDes = B->second;

    // Code motion for interleaved accesses can't violate WAR dependences.
    // Thus, reordering is legal if the source isn't a write.
    if (!Src->mayWriteToMemory())
      return true;

    // At least one of the accesses must be strided.
    if (!isStrided(SrcDes.Stride) && !isStrided(SinkDes.Stride))
      return true;

    // If dependence information is not available from LoopAccessInfo,
    // conservatively assume the instructions can't be reordered.
    if (!areDependencesValid())
      return false;

    // If we know there is a dependence from source to sink, assume the
    // instructions can't be reordered. Otherwise, reordering is legal.
    return !Dependences.count(Src) || !Dependences.lookup(Src).count(Sink);
  }

  /// \brief Collect the dependences from LoopAccessInfo.
  ///
  /// We process the dependences once during the interleaved access analysis to
  /// enable constant-time dependence queries.
  void collectDependences() {
    if (!areDependencesValid())
      return;
    auto *Deps = LAI->getDepChecker().getDependences();
    for (auto Dep : *Deps)
      Dependences[Dep.getSource(*LAI)].insert(Dep.getDestination(*LAI));
  }
};

/// Utility class for getting and setting loop vectorizer hints in the form
/// of loop metadata.
/// This class keeps a number of loop annotations locally (as member variables)
/// and can, upon request, write them back as metadata on the loop. It will
/// initially scan the loop for existing metadata, and will update the local
/// values based on information in the loop.
/// We cannot write all values to metadata, as the mere presence of some info,
/// for example 'force', means a decision has been made. So, we need to be
/// careful NOT to add them if the user hasn't specifically asked so.
class LoopVectorizeHints {
  enum HintKind { HK_WIDTH, HK_UNROLL, HK_FORCE };

  /// Hint - associates name and validation with the hint value.
  struct Hint {
    const char *Name;
    unsigned Value; // This may have to change for non-numeric values.
    HintKind Kind;

    Hint(const char *Name, unsigned Value, HintKind Kind)
        : Name(Name), Value(Value), Kind(Kind) {}

    bool validate(unsigned Val) {
      switch (Kind) {
      case HK_WIDTH:
        return isPowerOf2_32(Val) && Val <= VectorizerParams::MaxVectorWidth;
      case HK_UNROLL:
        return isPowerOf2_32(Val) && Val <= MaxInterleaveFactor;
      case HK_FORCE:
        return (Val <= 1);
      }
      return false;
    }
  };

  /// Vectorization width.
  Hint Width;
  /// Vectorization interleave factor.
  Hint Interleave;
  /// Vectorization forced
  Hint Force;

  /// Return the loop metadata prefix.
  static StringRef Prefix() { return "llvm.loop."; }

  /// True if there is any unsafe math in the loop.
  bool PotentiallyUnsafe;

public:
  enum ForceKind {
    FK_Undefined = -1, ///< Not selected.
    FK_Disabled = 0,   ///< Forcing disabled.
    FK_Enabled = 1,    ///< Forcing enabled.
  };

  LoopVectorizeHints(const Loop *L, bool DisableInterleaving,
                     OptimizationRemarkEmitter &ORE)
      : Width("vectorize.width", VectorizerParams::VectorizationFactor,
              HK_WIDTH),
        Interleave("interleave.count", DisableInterleaving, HK_UNROLL),
        Force("vectorize.enable", FK_Undefined, HK_FORCE),
        PotentiallyUnsafe(false), TheLoop(L), ORE(ORE) {
    // Populate values with existing loop metadata.
    getHintsFromMetadata();

    // force-vector-interleave overrides DisableInterleaving.
    if (VectorizerParams::isInterleaveForced())
      Interleave.Value = VectorizerParams::VectorizationInterleave;

    DEBUG(if (DisableInterleaving && Interleave.Value == 1) dbgs()
          << "LV: Interleaving disabled by the pass manager\n");
  }

  /// Mark the loop L as already vectorized by setting the width to 1.
  void setAlreadyVectorized() {
    Width.Value = Interleave.Value = 1;
    Hint Hints[] = {Width, Interleave};
    writeHintsToMetadata(Hints);
  }

  bool allowVectorization(Function *F, Loop *L, bool AlwaysVectorize) const {
    if (getForce() == LoopVectorizeHints::FK_Disabled) {
      DEBUG(dbgs() << "LV: Not vectorizing: #pragma vectorize disable.\n");
      emitRemarkWithHints();
      return false;
    }

    if (!AlwaysVectorize && getForce() != LoopVectorizeHints::FK_Enabled) {
      DEBUG(dbgs() << "LV: Not vectorizing: No #pragma vectorize enable.\n");
      emitRemarkWithHints();
      return false;
    }

    if (getWidth() == 1 && getInterleave() == 1) {
      // FIXME: Add a separate metadata to indicate when the loop has already
      // been vectorized instead of setting width and count to 1.
      DEBUG(dbgs() << "LV: Not vectorizing: Disabled/already vectorized.\n");
      // FIXME: Add interleave.disable metadata. This will allow
      // vectorize.disable to be used without disabling the pass and errors
      // to differentiate between disabled vectorization and a width of 1.
      ORE.emit(OptimizationRemarkAnalysis(vectorizeAnalysisPassName(),
                                          "AllDisabled", L->getStartLoc(),
                                          L->getHeader())
               << "loop not vectorized: vectorization and interleaving are "
                  "explicitly disabled, or vectorize width and interleave "
                  "count are both set to 1");
      return false;
    }

    return true;
  }

  /// Dumps all the hint information.
  void emitRemarkWithHints() const {
    using namespace ore;
    if (Force.Value == LoopVectorizeHints::FK_Disabled)
      ORE.emit(OptimizationRemarkMissed(LV_NAME, "MissedExplicitlyDisabled",
                                        TheLoop->getStartLoc(),
                                        TheLoop->getHeader())
               << "loop not vectorized: vectorization is explicitly disabled");
    else {
      OptimizationRemarkMissed R(LV_NAME, "MissedDetails",
                                 TheLoop->getStartLoc(), TheLoop->getHeader());
      R << "loop not vectorized";
      if (Force.Value == LoopVectorizeHints::FK_Enabled) {
        R << " (Force=" << NV("Force", true);
        if (Width.Value != 0)
          R << ", Vector Width=" << NV("VectorWidth", Width.Value);
        if (Interleave.Value != 0)
          R << ", Interleave Count=" << NV("InterleaveCount", Interleave.Value);
        R << ")";
      }
      ORE.emit(R);
    }
  }

  unsigned getWidth() const { return Width.Value; }
  unsigned getInterleave() const { return Interleave.Value; }
  enum ForceKind getForce() const { return (ForceKind)Force.Value; }

  /// \brief If hints are provided that force vectorization, use the AlwaysPrint
  /// pass name to force the frontend to print the diagnostic.
  const char *vectorizeAnalysisPassName() const {
    if (getWidth() == 1)
      return LV_NAME;
    if (getForce() == LoopVectorizeHints::FK_Disabled)
      return LV_NAME;
    if (getForce() == LoopVectorizeHints::FK_Undefined && getWidth() == 0)
      return LV_NAME;
    return OptimizationRemarkAnalysis::AlwaysPrint;
  }

  bool allowReordering() const {
    // When enabling loop hints are provided we allow the vectorizer to change
    // the order of operations that is given by the scalar loop. This is not
    // enabled by default because can be unsafe or inefficient. For example,
    // reordering floating-point operations will change the way round-off
    // error accumulates in the loop.
    return getForce() == LoopVectorizeHints::FK_Enabled || getWidth() > 1;
  }

  bool isPotentiallyUnsafe() const {
    // Avoid FP vectorization if the target is unsure about proper support.
    // This may be related to the SIMD unit in the target not handling
    // IEEE 754 FP ops properly, or bad single-to-double promotions.
    // Otherwise, a sequence of vectorized loops, even without reduction,
    // could lead to different end results on the destination vectors.
    return getForce() != LoopVectorizeHints::FK_Enabled && PotentiallyUnsafe;
  }

  void setPotentiallyUnsafe() { PotentiallyUnsafe = true; }

private:
  /// Find hints specified in the loop metadata and update local values.
  void getHintsFromMetadata() {
    MDNode *LoopID = TheLoop->getLoopID();
    if (!LoopID)
      return;

    // First operand should refer to the loop id itself.
    assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
    assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

    for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
      const MDString *S = nullptr;
      SmallVector<Metadata *, 4> Args;

      // The expected hint is either a MDString or a MDNode with the first
      // operand a MDString.
      if (const MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i))) {
        if (!MD || MD->getNumOperands() == 0)
          continue;
        S = dyn_cast<MDString>(MD->getOperand(0));
        for (unsigned i = 1, ie = MD->getNumOperands(); i < ie; ++i)
          Args.push_back(MD->getOperand(i));
      } else {
        S = dyn_cast<MDString>(LoopID->getOperand(i));
        assert(Args.size() == 0 && "too many arguments for MDString");
      }

      if (!S)
        continue;

      // Check if the hint starts with the loop metadata prefix.
      StringRef Name = S->getString();
      if (Args.size() == 1)
        setHint(Name, Args[0]);
    }
  }

  /// Checks string hint with one operand and set value if valid.
  void setHint(StringRef Name, Metadata *Arg) {
    if (!Name.startswith(Prefix()))
      return;
    Name = Name.substr(Prefix().size(), StringRef::npos);

    const ConstantInt *C = mdconst::dyn_extract<ConstantInt>(Arg);
    if (!C)
      return;
    unsigned Val = C->getZExtValue();

    Hint *Hints[] = {&Width, &Interleave, &Force};
    for (auto H : Hints) {
      if (Name == H->Name) {
        if (H->validate(Val))
          H->Value = Val;
        else
          DEBUG(dbgs() << "LV: ignoring invalid hint '" << Name << "'\n");
        break;
      }
    }
  }

  /// Create a new hint from name / value pair.
  MDNode *createHintMetadata(StringRef Name, unsigned V) const {
    LLVMContext &Context = TheLoop->getHeader()->getContext();
    Metadata *MDs[] = {MDString::get(Context, Name),
                       ConstantAsMetadata::get(
                           ConstantInt::get(Type::getInt32Ty(Context), V))};
    return MDNode::get(Context, MDs);
  }

  /// Matches metadata with hint name.
  bool matchesHintMetadataName(MDNode *Node, ArrayRef<Hint> HintTypes) {
    MDString *Name = dyn_cast<MDString>(Node->getOperand(0));
    if (!Name)
      return false;

    for (auto H : HintTypes)
      if (Name->getString().endswith(H.Name))
        return true;
    return false;
  }

  /// Sets current hints into loop metadata, keeping other values intact.
  void writeHintsToMetadata(ArrayRef<Hint> HintTypes) {
    if (HintTypes.size() == 0)
      return;

    // Reserve the first element to LoopID (see below).
    SmallVector<Metadata *, 4> MDs(1);
    // If the loop already has metadata, then ignore the existing operands.
    MDNode *LoopID = TheLoop->getLoopID();
    if (LoopID) {
      for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
        MDNode *Node = cast<MDNode>(LoopID->getOperand(i));
        // If node in update list, ignore old value.
        if (!matchesHintMetadataName(Node, HintTypes))
          MDs.push_back(Node);
      }
    }

    // Now, add the missing hints.
    for (auto H : HintTypes)
      MDs.push_back(createHintMetadata(Twine(Prefix(), H.Name).str(), H.Value));

    // Replace current metadata node with new one.
    LLVMContext &Context = TheLoop->getHeader()->getContext();
    MDNode *NewLoopID = MDNode::get(Context, MDs);
    // Set operand 0 to refer to the loop id itself.
    NewLoopID->replaceOperandWith(0, NewLoopID);

    TheLoop->setLoopID(NewLoopID);
  }

  /// The loop these hints belong to.
  const Loop *TheLoop;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter &ORE;
};

static void emitMissedWarning(Function *F, Loop *L,
                              const LoopVectorizeHints &LH,
                              OptimizationRemarkEmitter *ORE) {
  LH.emitRemarkWithHints();

  if (LH.getForce() == LoopVectorizeHints::FK_Enabled) {
    if (LH.getWidth() != 1)
      ORE->emit(DiagnosticInfoOptimizationFailure(
                    DEBUG_TYPE, "FailedRequestedVectorization",
                    L->getStartLoc(), L->getHeader())
                << "loop not vectorized: "
                << "failed explicitly specified loop vectorization");
    else if (LH.getInterleave() != 1)
      ORE->emit(DiagnosticInfoOptimizationFailure(
                    DEBUG_TYPE, "FailedRequestedInterleaving", L->getStartLoc(),
                    L->getHeader())
                << "loop not interleaved: "
                << "failed explicitly specified loop interleaving");
  }
}

/// LoopVectorizationLegality checks if it is legal to vectorize a loop, and
/// to what vectorization factor.
/// This class does not look at the profitability of vectorization, only the
/// legality. This class has two main kinds of checks:
/// * Memory checks - The code in canVectorizeMemory checks if vectorization
///   will change the order of memory accesses in a way that will change the
///   correctness of the program.
/// * Scalars checks - The code in canVectorizeInstrs and canVectorizeMemory
/// checks for a number of different conditions, such as the availability of a
/// single induction variable, that all types are supported and vectorize-able,
/// etc. This code reflects the capabilities of InnerLoopVectorizer.
/// This class is also used by InnerLoopVectorizer for identifying
/// induction variable and the different reduction variables.
class LoopVectorizationLegality {
public:
  LoopVectorizationLegality(
      Loop *L, PredicatedScalarEvolution &PSE, DominatorTree *DT,
      TargetLibraryInfo *TLI, AliasAnalysis *AA, Function *F,
      const TargetTransformInfo *TTI,
      std::function<const LoopAccessInfo &(Loop &)> *GetLAA, LoopInfo *LI,
      OptimizationRemarkEmitter *ORE, LoopVectorizationRequirements *R,
      LoopVectorizeHints *H)
      : NumPredStores(0), TheLoop(L), PSE(PSE), TLI(TLI), TTI(TTI), DT(DT),
        GetLAA(GetLAA), LAI(nullptr), ORE(ORE), InterleaveInfo(PSE, L, DT, LI),
        PrimaryInduction(nullptr), WidestIndTy(nullptr), HasFunNoNaNAttr(false),
        Requirements(R), Hints(H) {}

  /// ReductionList contains the reduction descriptors for all
  /// of the reductions that were found in the loop.
  typedef DenseMap<PHINode *, RecurrenceDescriptor> ReductionList;

  /// InductionList saves induction variables and maps them to the
  /// induction descriptor.
  typedef MapVector<PHINode *, InductionDescriptor> InductionList;

  /// RecurrenceSet contains the phi nodes that are recurrences other than
  /// inductions and reductions.
  typedef SmallPtrSet<const PHINode *, 8> RecurrenceSet;

  /// Returns true if it is legal to vectorize this loop.
  /// This does not mean that it is profitable to vectorize this
  /// loop, only that it is legal to do so.
  bool canVectorize();

  /// Returns the primary induction variable.
  PHINode *getPrimaryInduction() { return PrimaryInduction; }

  /// Returns the reduction variables found in the loop.
  ReductionList *getReductionVars() { return &Reductions; }

  /// Returns the induction variables found in the loop.
  InductionList *getInductionVars() { return &Inductions; }

  /// Return the first-order recurrences found in the loop.
  RecurrenceSet *getFirstOrderRecurrences() { return &FirstOrderRecurrences; }

  /// Returns the widest induction type.
  Type *getWidestInductionType() { return WidestIndTy; }

  /// Returns True if V is an induction variable in this loop.
  bool isInductionVariable(const Value *V);

  /// Returns True if PN is a reduction variable in this loop.
  bool isReductionVariable(PHINode *PN) { return Reductions.count(PN); }

  /// Returns True if Phi is a first-order recurrence in this loop.
  bool isFirstOrderRecurrence(const PHINode *Phi);

  /// Return true if the block BB needs to be predicated in order for the loop
  /// to be vectorized.
  bool blockNeedsPredication(BasicBlock *BB);

  /// Check if this pointer is consecutive when vectorizing. This happens
  /// when the last index of the GEP is the induction variable, or that the
  /// pointer itself is an induction variable.
  /// This check allows us to vectorize A[idx] into a wide load/store.
  /// Returns:
  /// 0 - Stride is unknown or non-consecutive.
  /// 1 - Address is consecutive.
  /// -1 - Address is consecutive, and decreasing.
  int isConsecutivePtr(Value *Ptr);

  /// Returns true if the value V is uniform within the loop.
  bool isUniform(Value *V);

  /// Returns the information that we collected about runtime memory check.
  const RuntimePointerChecking *getRuntimePointerChecking() const {
    return LAI->getRuntimePointerChecking();
  }

  const LoopAccessInfo *getLAI() const { return LAI; }

  /// \brief Check if \p Instr belongs to any interleaved access group.
  bool isAccessInterleaved(Instruction *Instr) {
    return InterleaveInfo.isInterleaved(Instr);
  }

  /// \brief Return the maximum interleave factor of all interleaved groups.
  unsigned getMaxInterleaveFactor() const {
    return InterleaveInfo.getMaxInterleaveFactor();
  }

  /// \brief Get the interleaved access group that \p Instr belongs to.
  const InterleaveGroup *getInterleavedAccessGroup(Instruction *Instr) {
    return InterleaveInfo.getInterleaveGroup(Instr);
  }

  /// \brief Returns true if an interleaved group requires a scalar iteration
  /// to handle accesses with gaps.
  bool requiresScalarEpilogue() const {
    return InterleaveInfo.requiresScalarEpilogue();
  }

  unsigned getMaxSafeDepDistBytes() { return LAI->getMaxSafeDepDistBytes(); }

  bool hasStride(Value *V) { return LAI->hasStride(V); }

  /// Returns true if the target machine supports masked store operation
  /// for the given \p DataType and kind of access to \p Ptr.
  bool isLegalMaskedStore(Type *DataType, Value *Ptr) {
    return isConsecutivePtr(Ptr) && TTI->isLegalMaskedStore(DataType);
  }
  /// Returns true if the target machine supports masked load operation
  /// for the given \p DataType and kind of access to \p Ptr.
  bool isLegalMaskedLoad(Type *DataType, Value *Ptr) {
    return isConsecutivePtr(Ptr) && TTI->isLegalMaskedLoad(DataType);
  }
  /// Returns true if the target machine supports masked scatter operation
  /// for the given \p DataType.
  bool isLegalMaskedScatter(Type *DataType) {
    return TTI->isLegalMaskedScatter(DataType);
  }
  /// Returns true if the target machine supports masked gather operation
  /// for the given \p DataType.
  bool isLegalMaskedGather(Type *DataType) {
    return TTI->isLegalMaskedGather(DataType);
  }
  /// Returns true if the target machine can represent \p V as a masked gather
  /// or scatter operation.
  bool isLegalGatherOrScatter(Value *V) {
    auto *LI = dyn_cast<LoadInst>(V);
    auto *SI = dyn_cast<StoreInst>(V);
    if (!LI && !SI)
      return false;
    auto *Ptr = getPointerOperand(V);
    auto *Ty = cast<PointerType>(Ptr->getType())->getElementType();
    return (LI && isLegalMaskedGather(Ty)) || (SI && isLegalMaskedScatter(Ty));
  }

  /// Returns true if vector representation of the instruction \p I
  /// requires mask.
  bool isMaskRequired(const Instruction *I) { return (MaskedOp.count(I) != 0); }
  unsigned getNumStores() const { return LAI->getNumStores(); }
  unsigned getNumLoads() const { return LAI->getNumLoads(); }
  unsigned getNumPredStores() const { return NumPredStores; }

  /// Returns true if \p I is an instruction that will be scalarized with
  /// predication. Such instructions include conditional stores and
  /// instructions that may divide by zero.
  bool isScalarWithPredication(Instruction *I);

  /// Returns true if \p I is a memory instruction with consecutive memory
  /// access that can be widened.
  bool memoryInstructionCanBeWidened(Instruction *I, unsigned VF = 1);

private:
  /// Check if a single basic block loop is vectorizable.
  /// At this point we know that this is a loop with a constant trip count
  /// and we only need to check individual instructions.
  bool canVectorizeInstrs();

  /// When we vectorize loops we may change the order in which
  /// we read and write from memory. This method checks if it is
  /// legal to vectorize the code, considering only memory constrains.
  /// Returns true if the loop is vectorizable
  bool canVectorizeMemory();

  /// Return true if we can vectorize this loop using the IF-conversion
  /// transformation.
  bool canVectorizeWithIfConvert();

  /// Return true if all of the instructions in the block can be speculatively
  /// executed. \p SafePtrs is a list of addresses that are known to be legal
  /// and we know that we can read from them without segfault.
  bool blockCanBePredicated(BasicBlock *BB, SmallPtrSetImpl<Value *> &SafePtrs);

  /// Updates the vectorization state by adding \p Phi to the inductions list.
  /// This can set \p Phi as the main induction of the loop if \p Phi is a
  /// better choice for the main induction than the existing one.
  void addInductionPhi(PHINode *Phi, const InductionDescriptor &ID,
                       SmallPtrSetImpl<Value *> &AllowedExit);

  /// Create an analysis remark that explains why vectorization failed
  ///
  /// \p RemarkName is the identifier for the remark.  If \p I is passed it is
  /// an instruction that prevents vectorization.  Otherwise the loop is used
  /// for the location of the remark.  \return the remark object that can be
  /// streamed to.
  OptimizationRemarkAnalysis
  createMissedAnalysis(StringRef RemarkName, Instruction *I = nullptr) const {
    return ::createMissedAnalysis(Hints->vectorizeAnalysisPassName(),
                                  RemarkName, TheLoop, I);
  }

  /// \brief If an access has a symbolic strides, this maps the pointer value to
  /// the stride symbol.
  const ValueToValueMap *getSymbolicStrides() {
    // FIXME: Currently, the set of symbolic strides is sometimes queried before
    // it's collected.  This happens from canVectorizeWithIfConvert, when the
    // pointer is checked to reference consecutive elements suitable for a
    // masked access.
    return LAI ? &LAI->getSymbolicStrides() : nullptr;
  }

  unsigned NumPredStores;

  /// The loop that we evaluate.
  Loop *TheLoop;
  /// A wrapper around ScalarEvolution used to add runtime SCEV checks.
  /// Applies dynamic knowledge to simplify SCEV expressions in the context
  /// of existing SCEV assumptions. The analysis will also add a minimal set
  /// of new predicates if this is required to enable vectorization and
  /// unrolling.
  PredicatedScalarEvolution &PSE;
  /// Target Library Info.
  TargetLibraryInfo *TLI;
  /// Target Transform Info
  const TargetTransformInfo *TTI;
  /// Dominator Tree.
  DominatorTree *DT;
  // LoopAccess analysis.
  std::function<const LoopAccessInfo &(Loop &)> *GetLAA;
  // And the loop-accesses info corresponding to this loop.  This pointer is
  // null until canVectorizeMemory sets it up.
  const LoopAccessInfo *LAI;
  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter *ORE;

  /// The interleave access information contains groups of interleaved accesses
  /// with the same stride and close to each other.
  InterleavedAccessInfo InterleaveInfo;

  //  ---  vectorization state --- //

  /// Holds the primary induction variable. This is the counter of the
  /// loop.
  PHINode *PrimaryInduction;
  /// Holds the reduction variables.
  ReductionList Reductions;
  /// Holds all of the induction variables that we found in the loop.
  /// Notice that inductions don't need to start at zero and that induction
  /// variables can be pointers.
  InductionList Inductions;
  /// Holds the phi nodes that are first-order recurrences.
  RecurrenceSet FirstOrderRecurrences;
  /// Holds the widest induction type encountered.
  Type *WidestIndTy;

  /// Allowed outside users. This holds the induction and reduction
  /// vars which can be accessed from outside the loop.
  SmallPtrSet<Value *, 4> AllowedExit;

  /// Can we assume the absence of NaNs.
  bool HasFunNoNaNAttr;

  /// Vectorization requirements that will go through late-evaluation.
  LoopVectorizationRequirements *Requirements;

  /// Used to emit an analysis of any legality issues.
  LoopVectorizeHints *Hints;

  /// While vectorizing these instructions we have to generate a
  /// call to the appropriate masked intrinsic
  SmallPtrSet<const Instruction *, 8> MaskedOp;
};

/// LoopVectorizationCostModel - estimates the expected speedups due to
/// vectorization.
/// In many cases vectorization is not profitable. This can happen because of
/// a number of reasons. In this class we mainly attempt to predict the
/// expected speedup/slowdowns due to the supported instruction set. We use the
/// TargetTransformInfo to query the different backends for the cost of
/// different operations.
class LoopVectorizationCostModel {
public:
  LoopVectorizationCostModel(Loop *L, PredicatedScalarEvolution &PSE,
                             LoopInfo *LI, LoopVectorizationLegality *Legal,
                             const TargetTransformInfo &TTI,
                             const TargetLibraryInfo *TLI, DemandedBits *DB,
                             AssumptionCache *AC,
                             OptimizationRemarkEmitter *ORE, const Function *F,
                             const LoopVectorizeHints *Hints)
      : TheLoop(L), PSE(PSE), LI(LI), Legal(Legal), TTI(TTI), TLI(TLI), DB(DB),
        AC(AC), ORE(ORE), TheFunction(F), Hints(Hints) {}

  /// Information about vectorization costs
  struct VectorizationFactor {
    unsigned Width; // Vector width with best cost
    unsigned Cost;  // Cost of the loop with that width
  };
  /// \return The most profitable vectorization factor and the cost of that VF.
  /// This method checks every power of two up to VF. If UserVF is not ZERO
  /// then this vectorization factor will be selected if vectorization is
  /// possible.
  VectorizationFactor selectVectorizationFactor(bool OptForSize);

  /// \return The size (in bits) of the smallest and widest types in the code
  /// that needs to be vectorized. We ignore values that remain scalar such as
  /// 64 bit loop indices.
  std::pair<unsigned, unsigned> getSmallestAndWidestTypes();

  /// \return The desired interleave count.
  /// If interleave count has been specified by metadata it will be returned.
  /// Otherwise, the interleave count is computed and returned. VF and LoopCost
  /// are the selected vectorization factor and the cost of the selected VF.
  unsigned selectInterleaveCount(bool OptForSize, unsigned VF,
                                 unsigned LoopCost);

  /// Memory access instruction may be vectorized in more than one way.
  /// Form of instruction after vectorization depends on cost.
  /// This function takes cost-based decisions for Load/Store instructions
  /// and collects them in a map. This decisions map is used for building
  /// the lists of loop-uniform and loop-scalar instructions.
  /// The calculated cost is saved with widening decision in order to
  /// avoid redundant calculations.
  void setCostBasedWideningDecision(unsigned VF);

  /// \brief A struct that represents some properties of the register usage
  /// of a loop.
  struct RegisterUsage {
    /// Holds the number of loop invariant values that are used in the loop.
    unsigned LoopInvariantRegs;
    /// Holds the maximum number of concurrent live intervals in the loop.
    unsigned MaxLocalUsers;
    /// Holds the number of instructions in the loop.
    unsigned NumInstructions;
  };

  /// \return Returns information about the register usages of the loop for the
  /// given vectorization factors.
  SmallVector<RegisterUsage, 8> calculateRegisterUsage(ArrayRef<unsigned> VFs);

  /// Collect values we want to ignore in the cost model.
  void collectValuesToIgnore();

  /// \returns The smallest bitwidth each instruction can be represented with.
  /// The vector equivalents of these instructions should be truncated to this
  /// type.
  const MapVector<Instruction *, uint64_t> &getMinimalBitwidths() const {
    return MinBWs;
  }

  /// \returns True if it is more profitable to scalarize instruction \p I for
  /// vectorization factor \p VF.
  bool isProfitableToScalarize(Instruction *I, unsigned VF) const {
    auto Scalars = InstsToScalarize.find(VF);
    assert(Scalars != InstsToScalarize.end() &&
           "VF not yet analyzed for scalarization profitability");
    return Scalars->second.count(I);
  }

  /// Returns true if \p I is known to be uniform after vectorization.
  bool isUniformAfterVectorization(Instruction *I, unsigned VF) const {
    if (VF == 1)
      return true;
    assert(Uniforms.count(VF) && "VF not yet analyzed for uniformity");
    auto UniformsPerVF = Uniforms.find(VF);
    return UniformsPerVF->second.count(I);
  }

  /// Returns true if \p I is known to be scalar after vectorization.
  bool isScalarAfterVectorization(Instruction *I, unsigned VF) const {
    if (VF == 1)
      return true;
    assert(Scalars.count(VF) && "Scalar values are not calculated for VF");
    auto ScalarsPerVF = Scalars.find(VF);
    return ScalarsPerVF->second.count(I);
  }

  /// \returns True if instruction \p I can be truncated to a smaller bitwidth
  /// for vectorization factor \p VF.
  bool canTruncateToMinimalBitwidth(Instruction *I, unsigned VF) const {
    return VF > 1 && MinBWs.count(I) && !isProfitableToScalarize(I, VF) &&
           !isScalarAfterVectorization(I, VF);
  }

  /// Decision that was taken during cost calculation for memory instruction.
  enum InstWidening {
    CM_Unknown,
    CM_Widen,
    CM_Interleave,
    CM_GatherScatter,
    CM_Scalarize
  };

  /// Save vectorization decision \p W and \p Cost taken by the cost model for
  /// instruction \p I and vector width \p VF.
  void setWideningDecision(Instruction *I, unsigned VF, InstWidening W,
                           unsigned Cost) {
    assert(VF >= 2 && "Expected VF >=2");
    WideningDecisions[std::make_pair(I, VF)] = std::make_pair(W, Cost);
  }

  /// Save vectorization decision \p W and \p Cost taken by the cost model for
  /// interleaving group \p Grp and vector width \p VF.
  void setWideningDecision(const InterleaveGroup *Grp, unsigned VF,
                           InstWidening W, unsigned Cost) {
    assert(VF >= 2 && "Expected VF >=2");
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
  InstWidening getWideningDecision(Instruction *I, unsigned VF) {
    assert(VF >= 2 && "Expected VF >=2");
    std::pair<Instruction *, unsigned> InstOnVF = std::make_pair(I, VF);
    auto Itr = WideningDecisions.find(InstOnVF);
    if (Itr == WideningDecisions.end())
      return CM_Unknown;
    return Itr->second.first;
  }

  /// Return the vectorization cost for the given instruction \p I and vector
  /// width \p VF.
  unsigned getWideningCost(Instruction *I, unsigned VF) {
    assert(VF >= 2 && "Expected VF >=2");
    std::pair<Instruction *, unsigned> InstOnVF = std::make_pair(I, VF);
    assert(WideningDecisions.count(InstOnVF) && "The cost is not calculated");
    return WideningDecisions[InstOnVF].second;
  }

  /// Return True if instruction \p I is an optimizable truncate whose operand
  /// is an induction variable. Such a truncate will be removed by adding a new
  /// induction variable with the destination type.
  bool isOptimizableIVTruncate(Instruction *I, unsigned VF) {

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
    return Legal->isInductionVariable(Op);
  }

private:
  /// The vectorization cost is a combination of the cost itself and a boolean
  /// indicating whether any of the contributing operations will actually
  /// operate on
  /// vector values after type legalization in the backend. If this latter value
  /// is
  /// false, then all operations will be scalarized (i.e. no vectorization has
  /// actually taken place).
  typedef std::pair<unsigned, bool> VectorizationCostTy;

  /// Returns the expected execution cost. The unit of the cost does
  /// not matter because we use the 'cost' units to compare different
  /// vector widths. The cost that is returned is *not* normalized by
  /// the factor width.
  VectorizationCostTy expectedCost(unsigned VF);

  /// Returns the execution time cost of an instruction for a given vector
  /// width. Vector width of one means scalar.
  VectorizationCostTy getInstructionCost(Instruction *I, unsigned VF);

  /// The cost-computation logic from getInstructionCost which provides
  /// the vector type as an output parameter.
  unsigned getInstructionCost(Instruction *I, unsigned VF, Type *&VectorTy);

  /// Calculate vectorization cost of memory instruction \p I.
  unsigned getMemoryInstructionCost(Instruction *I, unsigned VF);

  /// The cost computation for scalarized memory instruction.
  unsigned getMemInstScalarizationCost(Instruction *I, unsigned VF);

  /// The cost computation for interleaving group of memory instructions.
  unsigned getInterleaveGroupCost(Instruction *I, unsigned VF);

  /// The cost computation for Gather/Scatter instruction.
  unsigned getGatherScatterCost(Instruction *I, unsigned VF);

  /// The cost computation for widening instruction \p I with consecutive
  /// memory access.
  unsigned getConsecutiveMemOpCost(Instruction *I, unsigned VF);

  /// The cost calculation for Load instruction \p I with uniform pointer -
  /// scalar load + broadcast.
  unsigned getUniformMemOpCost(Instruction *I, unsigned VF);

  /// Returns whether the instruction is a load or store and will be a emitted
  /// as a vector operation.
  bool isConsecutiveLoadOrStore(Instruction *I);

  /// Create an analysis remark that explains why vectorization failed
  ///
  /// \p RemarkName is the identifier for the remark.  \return the remark object
  /// that can be streamed to.
  OptimizationRemarkAnalysis createMissedAnalysis(StringRef RemarkName) {
    return ::createMissedAnalysis(Hints->vectorizeAnalysisPassName(),
                                  RemarkName, TheLoop);
  }

  /// Map of scalar integer values to the smallest bitwidth they can be legally
  /// represented as. The vector equivalents of these values should be truncated
  /// to this type.
  MapVector<Instruction *, uint64_t> MinBWs;

  /// A type representing the costs for instructions if they were to be
  /// scalarized rather than vectorized. The entries are Instruction-Cost
  /// pairs.
  typedef DenseMap<Instruction *, unsigned> ScalarCostsTy;

  /// A map holding scalar costs for different vectorization factors. The
  /// presence of a cost for an instruction in the mapping indicates that the
  /// instruction will be scalarized when vectorizing with the associated
  /// vectorization factor. The entries are VF-ScalarCostTy pairs.
  DenseMap<unsigned, ScalarCostsTy> InstsToScalarize;

  /// Holds the instructions known to be uniform after vectorization.
  /// The data is collected per VF.
  DenseMap<unsigned, SmallPtrSet<Instruction *, 4>> Uniforms;

  /// Holds the instructions known to be scalar after vectorization.
  /// The data is collected per VF.
  DenseMap<unsigned, SmallPtrSet<Instruction *, 4>> Scalars;

  /// Returns the expected difference in cost from scalarizing the expression
  /// feeding a predicated instruction \p PredInst. The instructions to
  /// scalarize and their scalar costs are collected in \p ScalarCosts. A
  /// non-negative return value implies the expression will be scalarized.
  /// Currently, only single-use chains are considered for scalarization.
  int computePredInstDiscount(Instruction *PredInst, ScalarCostsTy &ScalarCosts,
                              unsigned VF);

  /// Collects the instructions to scalarize for each predicated instruction in
  /// the loop.
  void collectInstsToScalarize(unsigned VF);

  /// Collect the instructions that are uniform after vectorization. An
  /// instruction is uniform if we represent it with a single scalar value in
  /// the vectorized loop corresponding to each vector iteration. Examples of
  /// uniform instructions include pointer operands of consecutive or
  /// interleaved memory accesses. Note that although uniformity implies an
  /// instruction will be scalar, the reverse is not true. In general, a
  /// scalarized instruction will be represented by VF scalar values in the
  /// vectorized loop, each corresponding to an iteration of the original
  /// scalar loop.
  void collectLoopUniforms(unsigned VF);

  /// Collect the instructions that are scalar after vectorization. An
  /// instruction is scalar if it is known to be uniform or will be scalarized
  /// during vectorization. Non-uniform scalarized instructions will be
  /// represented by VF values in the vectorized loop, each corresponding to an
  /// iteration of the original scalar loop.
  void collectLoopScalars(unsigned VF);

  /// Collect Uniform and Scalar values for the given \p VF.
  /// The sets depend on CM decision for Load/Store instructions
  /// that may be vectorized as interleave, gather-scatter or scalarized.
  void collectUniformsAndScalars(unsigned VF) {
    // Do the analysis once.
    if (VF == 1 || Uniforms.count(VF))
      return;
    setCostBasedWideningDecision(VF);
    collectLoopUniforms(VF);
    collectLoopScalars(VF);
  }

  /// Keeps cost model vectorization decision and cost for instructions.
  /// Right now it is used for memory instructions only.
  typedef DenseMap<std::pair<Instruction *, unsigned>,
                   std::pair<InstWidening, unsigned>>
      DecisionList;

  DecisionList WideningDecisions;

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
  /// Values to ignore in the cost model.
  SmallPtrSet<const Value *, 16> ValuesToIgnore;
  /// Values to ignore in the cost model when VF > 1.
  SmallPtrSet<const Value *, 16> VecValuesToIgnore;
};

/// \brief This holds vectorization requirements that must be verified late in
/// the process. The requirements are set by legalize and costmodel. Once
/// vectorization has been determined to be possible and profitable the
/// requirements can be verified by looking for metadata or compiler options.
/// For example, some loops require FP commutativity which is only allowed if
/// vectorization is explicitly specified or if the fast-math compiler option
/// has been provided.
/// Late evaluation of these requirements allows helpful diagnostics to be
/// composed that tells the user what need to be done to vectorize the loop. For
/// example, by specifying #pragma clang loop vectorize or -ffast-math. Late
/// evaluation should be used only when diagnostics can generated that can be
/// followed by a non-expert user.
class LoopVectorizationRequirements {
public:
  LoopVectorizationRequirements(OptimizationRemarkEmitter &ORE)
      : NumRuntimePointerChecks(0), UnsafeAlgebraInst(nullptr), ORE(ORE) {}

  void addUnsafeAlgebraInst(Instruction *I) {
    // First unsafe algebra instruction.
    if (!UnsafeAlgebraInst)
      UnsafeAlgebraInst = I;
  }

  void addRuntimePointerChecks(unsigned Num) { NumRuntimePointerChecks = Num; }

  bool doesNotMeet(Function *F, Loop *L, const LoopVectorizeHints &Hints) {
    const char *PassName = Hints.vectorizeAnalysisPassName();
    bool Failed = false;
    if (UnsafeAlgebraInst && !Hints.allowReordering()) {
      ORE.emit(
          OptimizationRemarkAnalysisFPCommute(PassName, "CantReorderFPOps",
                                              UnsafeAlgebraInst->getDebugLoc(),
                                              UnsafeAlgebraInst->getParent())
          << "loop not vectorized: cannot prove it is safe to reorder "
             "floating-point operations");
      Failed = true;
    }

    // Test if runtime memcheck thresholds are exceeded.
    bool PragmaThresholdReached =
        NumRuntimePointerChecks > PragmaVectorizeMemoryCheckThreshold;
    bool ThresholdReached =
        NumRuntimePointerChecks > VectorizerParams::RuntimeMemoryCheckThreshold;
    if ((ThresholdReached && !Hints.allowReordering()) ||
        PragmaThresholdReached) {
      ORE.emit(OptimizationRemarkAnalysisAliasing(PassName, "CantReorderMemOps",
                                                  L->getStartLoc(),
                                                  L->getHeader())
               << "loop not vectorized: cannot prove it is safe to reorder "
                  "memory operations");
      DEBUG(dbgs() << "LV: Too many memory checks needed.\n");
      Failed = true;
    }

    return Failed;
  }

private:
  unsigned NumRuntimePointerChecks;
  Instruction *UnsafeAlgebraInst;

  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter &ORE;
};

static void addAcyclicInnerLoop(Loop &L, SmallVectorImpl<Loop *> &V) {
  if (L.empty()) {
    if (!hasCyclesInLoopBody(L))
      V.push_back(&L);
    return;
  }
  for (Loop *InnerL : L)
    addAcyclicInnerLoop(*InnerL, V);
}

/// The LoopVectorize Pass.
struct LoopVectorize : public FunctionPass {
  /// Pass identification, replacement for typeid
  static char ID;

  explicit LoopVectorize(bool NoUnrolling = false, bool AlwaysVectorize = true)
      : FunctionPass(ID) {
    Impl.DisableUnrolling = NoUnrolling;
    Impl.AlwaysVectorize = AlwaysVectorize;
    initializeLoopVectorizePass(*PassRegistry::getPassRegistry());
  }

  LoopVectorizePass Impl;

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    auto *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto *BFI = &getAnalysis<BlockFrequencyInfoWrapperPass>().getBFI();
    auto *TLIP = getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
    auto *TLI = TLIP ? &TLIP->getTLI() : nullptr;
    auto *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto *AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    auto *LAA = &getAnalysis<LoopAccessLegacyAnalysis>();
    auto *DB = &getAnalysis<DemandedBitsWrapperPass>().getDemandedBits();
    auto *ORE = &getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

    std::function<const LoopAccessInfo &(Loop &)> GetLAA =
        [&](Loop &L) -> const LoopAccessInfo & { return LAA->getInfo(&L); };

    return Impl.runImpl(F, *SE, *LI, *TTI, *DT, *BFI, TLI, *DB, *AA, *AC,
                        GetLAA, *ORE);
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
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<BasicAAWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Implementation of LoopVectorizationLegality, InnerLoopVectorizer and
// LoopVectorizationCostModel.
//===----------------------------------------------------------------------===//

Value *InnerLoopVectorizer::getBroadcastInstrs(Value *V) {
  // We need to place the broadcast of invariant variables outside the loop.
  Instruction *Instr = dyn_cast<Instruction>(V);
  bool NewInstr = (Instr && Instr->getParent() == LoopVectorBody);
  bool Invariant = OrigLoop->isLoopInvariant(V) && !NewInstr;

  // Place the code for broadcasting invariant variables in the new preheader.
  IRBuilder<>::InsertPointGuard Guard(Builder);
  if (Invariant)
    Builder.SetInsertPoint(LoopVectorPreHeader->getTerminator());

  // Broadcast the scalar into all locations in the vector.
  Value *Shuf = Builder.CreateVectorSplat(VF, V, "broadcast");

  return Shuf;
}

void InnerLoopVectorizer::createVectorIntOrFpInductionPHI(
    const InductionDescriptor &II, Value *Step, Instruction *EntryVal) {
  Value *Start = II.getStartValue();

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
  Value *ConstVF = getSignedIntOrFpConstant(Step->getType(), VF);
  Value *Mul = addFastMathFlag(Builder.CreateBinOp(MulOp, Step, ConstVF));

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
  Instruction *LastInduction = VecInd;
  VectorParts Entry(UF);
  for (unsigned Part = 0; Part < UF; ++Part) {
    Entry[Part] = LastInduction;
    LastInduction = cast<Instruction>(addFastMathFlag(
        Builder.CreateBinOp(AddOp, LastInduction, SplatVF, "step.add")));
  }
  VectorLoopValueMap.initVector(EntryVal, Entry);
  if (isa<TruncInst>(EntryVal))
    addMetadata(Entry, EntryVal);

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
  return any_of(IV->users(), isScalarInst);
}

void InnerLoopVectorizer::widenIntOrFpInduction(PHINode *IV, TruncInst *Trunc) {

  assert((IV->getType()->isIntegerTy() || IV != OldInduction) &&
         "Primary induction variable must have an integer type");

  auto II = Legal->getInductionVars()->find(IV);
  assert(II != Legal->getInductionVars()->end() && "IV is not an induction");

  auto ID = II->second;
  assert(IV->getType() == ID.getStartValue()->getType() && "Types must match");

  // The scalar value to broadcast. This will be derived from the canonical
  // induction variable.
  Value *ScalarIV = nullptr;

  // The value from the original loop to which we are mapping the new induction
  // variable.
  Instruction *EntryVal = Trunc ? cast<Instruction>(Trunc) : IV;

  // True if we have vectorized the induction variable.
  auto VectorizedIV = false;

  // Determine if we want a scalar version of the induction variable. This is
  // true if the induction variable itself is not widened, or if it has at
  // least one user in the loop that is not widened.
  auto NeedsScalarIV = VF > 1 && needsScalarInduction(EntryVal);

  // Generate code for the induction step. Note that induction steps are
  // required to be loop-invariant
  assert(PSE.getSE()->isLoopInvariant(ID.getStep(), OrigLoop) &&
         "Induction step should be loop invariant");
  auto &DL = OrigLoop->getHeader()->getModule()->getDataLayout();
  Value *Step = nullptr;
  if (PSE.getSE()->isSCEVable(IV->getType())) {
    SCEVExpander Exp(*PSE.getSE(), DL, "induction");
    Step = Exp.expandCodeFor(ID.getStep(), ID.getStep()->getType(),
                             LoopVectorPreHeader->getTerminator());
  } else {
    Step = cast<SCEVUnknown>(ID.getStep())->getValue();
  }

  // Try to create a new independent vector induction variable. If we can't
  // create the phi node, we will splat the scalar induction variable in each
  // loop iteration.
  if (VF > 1 && !shouldScalarizeInstruction(EntryVal)) {
    createVectorIntOrFpInductionPHI(ID, Step, EntryVal);
    VectorizedIV = true;
  }

  // If we haven't yet vectorized the induction variable, or if we will create
  // a scalar one, we need to define the scalar induction variable and step
  // values. If we were given a truncation type, truncate the canonical
  // induction variable and step. Otherwise, derive these values from the
  // induction descriptor.
  if (!VectorizedIV || NeedsScalarIV) {
    if (Trunc) {
      auto *TruncType = cast<IntegerType>(Trunc->getType());
      assert(Step->getType()->isIntegerTy() &&
             "Truncation requires an integer step");
      ScalarIV = Builder.CreateCast(Instruction::Trunc, Induction, TruncType);
      Step = Builder.CreateTrunc(Step, TruncType);
    } else {
      ScalarIV = Induction;
      if (IV != OldInduction) {
        ScalarIV = IV->getType()->isIntegerTy()
                       ? Builder.CreateSExtOrTrunc(ScalarIV, IV->getType())
                       : Builder.CreateCast(Instruction::SIToFP, Induction,
                                            IV->getType());
        ScalarIV = ID.transform(Builder, ScalarIV, PSE.getSE(), DL);
        ScalarIV->setName("offset.idx");
      }
    }
  }

  // If we haven't yet vectorized the induction variable, splat the scalar
  // induction variable, and build the necessary step vectors.
  if (!VectorizedIV) {
    Value *Broadcasted = getBroadcastInstrs(ScalarIV);
    VectorParts Entry(UF);
    for (unsigned Part = 0; Part < UF; ++Part)
      Entry[Part] =
          getStepVector(Broadcasted, VF * Part, Step, ID.getInductionOpcode());
    VectorLoopValueMap.initVector(EntryVal, Entry);
    if (Trunc)
      addMetadata(Entry, Trunc);
  }

  // If an induction variable is only used for counting loop iterations or
  // calculating addresses, it doesn't need to be widened. Create scalar steps
  // that can be used by instructions we will later scalarize. Note that the
  // addition of the scalar steps will not increase the number of instructions
  // in the loop in the common case prior to InstCombine. We will be trading
  // one vector extract for each scalar step.
  if (NeedsScalarIV)
    buildScalarSteps(ScalarIV, Step, EntryVal, ID);
}

Value *InnerLoopVectorizer::getStepVector(Value *Val, int StartIdx, Value *Step,
                                          Instruction::BinaryOps BinOp) {
  // Create and check the types.
  assert(Val->getType()->isVectorTy() && "Must be a vector");
  int VLen = Val->getType()->getVectorNumElements();

  Type *STy = Val->getType()->getScalarType();
  assert((STy->isIntegerTy() || STy->isFloatingPointTy()) &&
         "Induction Step must be an integer or FP");
  assert(Step->getType() == STy && "Step has wrong type");

  SmallVector<Constant *, 8> Indices;

  if (STy->isIntegerTy()) {
    // Create a vector of consecutive numbers from zero to VF.
    for (int i = 0; i < VLen; ++i)
      Indices.push_back(ConstantInt::get(STy, StartIdx + i));

    // Add the consecutive indices to the vector value.
    Constant *Cv = ConstantVector::get(Indices);
    assert(Cv->getType() == Val->getType() && "Invalid consecutive vec");
    Step = Builder.CreateVectorSplat(VLen, Step);
    assert(Step->getType() == Val->getType() && "Invalid step vec");
    // FIXME: The newly created binary instructions should contain nsw/nuw flags,
    // which can be found from the original scalar operations.
    Step = Builder.CreateMul(Cv, Step);
    return Builder.CreateAdd(Val, Step, "induction");
  }

  // Floating point induction.
  assert((BinOp == Instruction::FAdd || BinOp == Instruction::FSub) &&
         "Binary Opcode should be specified for FP induction");
  // Create a vector of consecutive numbers from zero to VF.
  for (int i = 0; i < VLen; ++i)
    Indices.push_back(ConstantFP::get(STy, (double)(StartIdx + i)));

  // Add the consecutive indices to the vector value.
  Constant *Cv = ConstantVector::get(Indices);

  Step = Builder.CreateVectorSplat(VLen, Step);

  // Floating point operations had to be 'fast' to enable the induction.
  FastMathFlags Flags;
  Flags.setUnsafeAlgebra();

  Value *MulOp = Builder.CreateFMul(Cv, Step);
  if (isa<Instruction>(MulOp))
    // Have to check, MulOp may be a constant
    cast<Instruction>(MulOp)->setFastMathFlags(Flags);

  Value *BOp = Builder.CreateBinOp(BinOp, Val, MulOp, "induction");
  if (isa<Instruction>(BOp))
    cast<Instruction>(BOp)->setFastMathFlags(Flags);
  return BOp;
}

void InnerLoopVectorizer::buildScalarSteps(Value *ScalarIV, Value *Step,
                                           Value *EntryVal,
                                           const InductionDescriptor &ID) {

  // We shouldn't have to build scalar steps if we aren't vectorizing.
  assert(VF > 1 && "VF should be greater than one");

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
  unsigned Lanes =
    Cost->isUniformAfterVectorization(cast<Instruction>(EntryVal), VF) ? 1 : VF;

  // Compute the scalar steps and save the results in VectorLoopValueMap.
  ScalarParts Entry(UF);
  for (unsigned Part = 0; Part < UF; ++Part) {
    Entry[Part].resize(VF);
    for (unsigned Lane = 0; Lane < Lanes; ++Lane) {
      auto *StartIdx = getSignedIntOrFpConstant(ScalarIVTy, VF * Part + Lane);
      auto *Mul = addFastMathFlag(Builder.CreateBinOp(MulOp, StartIdx, Step));
      auto *Add = addFastMathFlag(Builder.CreateBinOp(AddOp, ScalarIV, Mul));
      Entry[Part][Lane] = Add;
    }
  }
  VectorLoopValueMap.initScalar(EntryVal, Entry);
}

int LoopVectorizationLegality::isConsecutivePtr(Value *Ptr) {

  const ValueToValueMap &Strides = getSymbolicStrides() ? *getSymbolicStrides() :
    ValueToValueMap();

  int Stride = getPtrStride(PSE, Ptr, TheLoop, Strides, true, false);
  if (Stride == 1 || Stride == -1)
    return Stride;
  return 0;
}

bool LoopVectorizationLegality::isUniform(Value *V) {
  return LAI->isUniform(V);
}

const InnerLoopVectorizer::VectorParts &
InnerLoopVectorizer::getVectorValue(Value *V) {
  assert(V != Induction && "The new induction variable should not be used.");
  assert(!V->getType()->isVectorTy() && "Can't widen a vector");
  assert(!V->getType()->isVoidTy() && "Type does not produce a value");

  // If we have a stride that is replaced by one, do it here.
  if (Legal->hasStride(V))
    V = ConstantInt::get(V->getType(), 1);

  // If we have this scalar in the map, return it.
  if (VectorLoopValueMap.hasVector(V))
    return VectorLoopValueMap.VectorMapStorage[V];

  // If the value has not been vectorized, check if it has been scalarized
  // instead. If it has been scalarized, and we actually need the value in
  // vector form, we will construct the vector values on demand.
  if (VectorLoopValueMap.hasScalar(V)) {

    // Initialize a new vector map entry.
    VectorParts Entry(UF);

    // If we've scalarized a value, that value should be an instruction.
    auto *I = cast<Instruction>(V);

    // If we aren't vectorizing, we can just copy the scalar map values over to
    // the vector map.
    if (VF == 1) {
      for (unsigned Part = 0; Part < UF; ++Part)
        Entry[Part] = getScalarValue(V, Part, 0);
      return VectorLoopValueMap.initVector(V, Entry);
    }

    // Get the last scalar instruction we generated for V. If the value is
    // known to be uniform after vectorization, this corresponds to lane zero
    // of the last unroll iteration. Otherwise, the last instruction is the one
    // we created for the last vector lane of the last unroll iteration.
    unsigned LastLane = Cost->isUniformAfterVectorization(I, VF) ? 0 : VF - 1;
    auto *LastInst = cast<Instruction>(getScalarValue(V, UF - 1, LastLane));

    // Set the insert point after the last scalarized instruction. This ensures
    // the insertelement sequence will directly follow the scalar definitions.
    auto OldIP = Builder.saveIP();
    auto NewIP = std::next(BasicBlock::iterator(LastInst));
    Builder.SetInsertPoint(&*NewIP);

    // However, if we are vectorizing, we need to construct the vector values.
    // If the value is known to be uniform after vectorization, we can just
    // broadcast the scalar value corresponding to lane zero for each unroll
    // iteration. Otherwise, we construct the vector values using insertelement
    // instructions. Since the resulting vectors are stored in
    // VectorLoopValueMap, we will only generate the insertelements once.
    for (unsigned Part = 0; Part < UF; ++Part) {
      Value *VectorValue = nullptr;
      if (Cost->isUniformAfterVectorization(I, VF)) {
        VectorValue = getBroadcastInstrs(getScalarValue(V, Part, 0));
      } else {
        VectorValue = UndefValue::get(VectorType::get(V->getType(), VF));
        for (unsigned Lane = 0; Lane < VF; ++Lane)
          VectorValue = Builder.CreateInsertElement(
              VectorValue, getScalarValue(V, Part, Lane),
              Builder.getInt32(Lane));
      }
      Entry[Part] = VectorValue;
    }
    Builder.restoreIP(OldIP);
    return VectorLoopValueMap.initVector(V, Entry);
  }

  // If this scalar is unknown, assume that it is a constant or that it is
  // loop invariant. Broadcast V and save the value for future uses.
  Value *B = getBroadcastInstrs(V);
  return VectorLoopValueMap.initVector(V, VectorParts(UF, B));
}

Value *InnerLoopVectorizer::getScalarValue(Value *V, unsigned Part,
                                           unsigned Lane) {

  // If the value is not an instruction contained in the loop, it should
  // already be scalar.
  if (OrigLoop->isLoopInvariant(V))
    return V;

  assert(Lane > 0 ?
         !Cost->isUniformAfterVectorization(cast<Instruction>(V), VF)
         : true && "Uniform values only have lane zero");

  // If the value from the original loop has not been vectorized, it is
  // represented by UF x VF scalar values in the new loop. Return the requested
  // scalar value.
  if (VectorLoopValueMap.hasScalar(V))
    return VectorLoopValueMap.ScalarMapStorage[V][Part][Lane];

  // If the value has not been scalarized, get its entry in VectorLoopValueMap
  // for the given unroll part. If this entry is not a vector type (i.e., the
  // vectorization factor is one), there is no need to generate an
  // extractelement instruction.
  auto *U = getVectorValue(V)[Part];
  if (!U->getType()->isVectorTy()) {
    assert(VF == 1 && "Value not scalarized has non-vector type");
    return U;
  }

  // Otherwise, the value from the original loop has been vectorized and is
  // represented by UF vector values. Extract and return the requested scalar
  // value from the appropriate vector lane.
  return Builder.CreateExtractElement(U, Builder.getInt32(Lane));
}

Value *InnerLoopVectorizer::reverseVector(Value *Vec) {
  assert(Vec->getType()->isVectorTy() && "Invalid type");
  SmallVector<Constant *, 8> ShuffleMask;
  for (unsigned i = 0; i < VF; ++i)
    ShuffleMask.push_back(Builder.getInt32(VF - i - 1));

  return Builder.CreateShuffleVector(Vec, UndefValue::get(Vec->getType()),
                                     ConstantVector::get(ShuffleMask),
                                     "reverse");
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
//   %R.vec = shuffle %wide.vec, undef, <0, 3, 6, 9>   ; R elements
//   %G.vec = shuffle %wide.vec, undef, <1, 4, 7, 10>  ; G elements
//   %B.vec = shuffle %wide.vec, undef, <2, 5, 8, 11>  ; B elements
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
//   %B_U.vec = shuffle %B.vec, undef, <0, 1, 2, 3, u, u, u, u>
//   %interleaved.vec = shuffle %R_G.vec, %B_U.vec,
//        <0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11>    ; Interleave R,G,B elements
//   store <12 x i32> %interleaved.vec              ; Write 4 tuples of R,G,B
void InnerLoopVectorizer::vectorizeInterleaveGroup(Instruction *Instr) {
  const InterleaveGroup *Group = Legal->getInterleavedAccessGroup(Instr);
  assert(Group && "Fail to get an interleaved access group.");

  // Skip if current instruction is not the insert position.
  if (Instr != Group->getInsertPos())
    return;

  Value *Ptr = getPointerOperand(Instr);

  // Prepare for the vector type of the interleaved load/store.
  Type *ScalarTy = getMemInstValueType(Instr);
  unsigned InterleaveFactor = Group->getFactor();
  Type *VecTy = VectorType::get(ScalarTy, InterleaveFactor * VF);
  Type *PtrTy = VecTy->getPointerTo(getMemInstAddressSpace(Instr));

  // Prepare for the new pointers.
  setDebugLocFromInst(Builder, Ptr);
  SmallVector<Value *, 2> NewPtrs;
  unsigned Index = Group->getIndex(Instr);

  // If the group is reverse, adjust the index to refer to the last vector lane
  // instead of the first. We adjust the index from the first vector lane,
  // rather than directly getting the pointer for lane VF - 1, because the
  // pointer operand of the interleaved access is supposed to be uniform. For
  // uniform instructions, we're only required to generate a value for the
  // first vector lane in each unroll iteration.
  if (Group->isReverse())
    Index += (VF - 1) * Group->getFactor();

  for (unsigned Part = 0; Part < UF; Part++) {
    Value *NewPtr = getScalarValue(Ptr, Part, 0);

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
    NewPtr = Builder.CreateGEP(NewPtr, Builder.getInt32(-Index));

    // Cast to the vector pointer type.
    NewPtrs.push_back(Builder.CreateBitCast(NewPtr, PtrTy));
  }

  setDebugLocFromInst(Builder, Instr);
  Value *UndefVec = UndefValue::get(VecTy);

  // Vectorize the interleaved load group.
  if (isa<LoadInst>(Instr)) {

    // For each unroll part, create a wide load for the group.
    SmallVector<Value *, 2> NewLoads;
    for (unsigned Part = 0; Part < UF; Part++) {
      auto *NewLoad = Builder.CreateAlignedLoad(
          NewPtrs[Part], Group->getAlignment(), "wide.vec");
      addMetadata(NewLoad, Instr);
      NewLoads.push_back(NewLoad);
    }

    // For each member in the group, shuffle out the appropriate data from the
    // wide loads.
    for (unsigned I = 0; I < InterleaveFactor; ++I) {
      Instruction *Member = Group->getMember(I);

      // Skip the gaps in the group.
      if (!Member)
        continue;

      VectorParts Entry(UF);
      Constant *StrideMask = createStrideMask(Builder, I, InterleaveFactor, VF);
      for (unsigned Part = 0; Part < UF; Part++) {
        Value *StridedVec = Builder.CreateShuffleVector(
            NewLoads[Part], UndefVec, StrideMask, "strided.vec");

        // If this member has different type, cast the result type.
        if (Member->getType() != ScalarTy) {
          VectorType *OtherVTy = VectorType::get(Member->getType(), VF);
          StridedVec = Builder.CreateBitOrPointerCast(StridedVec, OtherVTy);
        }

        Entry[Part] =
            Group->isReverse() ? reverseVector(StridedVec) : StridedVec;
      }
      VectorLoopValueMap.initVector(Member, Entry);
    }
    return;
  }

  // The sub vector type for current instruction.
  VectorType *SubVT = VectorType::get(ScalarTy, VF);

  // Vectorize the interleaved store group.
  for (unsigned Part = 0; Part < UF; Part++) {
    // Collect the stored vector from each member.
    SmallVector<Value *, 4> StoredVecs;
    for (unsigned i = 0; i < InterleaveFactor; i++) {
      // Interleaved store group doesn't allow a gap, so each index has a member
      Instruction *Member = Group->getMember(i);
      assert(Member && "Fail to get a member from an interleaved store group");

      Value *StoredVec =
          getVectorValue(cast<StoreInst>(Member)->getValueOperand())[Part];
      if (Group->isReverse())
        StoredVec = reverseVector(StoredVec);

      // If this member has different type, cast it to an unified type.
      if (StoredVec->getType() != SubVT)
        StoredVec = Builder.CreateBitOrPointerCast(StoredVec, SubVT);

      StoredVecs.push_back(StoredVec);
    }

    // Concatenate all vectors into a wide vector.
    Value *WideVec = concatenateVectors(Builder, StoredVecs);

    // Interleave the elements in the wide vector.
    Constant *IMask = createInterleaveMask(Builder, VF, InterleaveFactor);
    Value *IVec = Builder.CreateShuffleVector(WideVec, UndefVec, IMask,
                                              "interleaved.vec");

    Instruction *NewStoreInstr =
        Builder.CreateAlignedStore(IVec, NewPtrs[Part], Group->getAlignment());
    addMetadata(NewStoreInstr, Instr);
  }
}

void InnerLoopVectorizer::vectorizeMemoryInstruction(Instruction *Instr) {
  // Attempt to issue a wide load.
  LoadInst *LI = dyn_cast<LoadInst>(Instr);
  StoreInst *SI = dyn_cast<StoreInst>(Instr);

  assert((LI || SI) && "Invalid Load/Store instruction");

  LoopVectorizationCostModel::InstWidening Decision =
      Cost->getWideningDecision(Instr, VF);
  assert(Decision != LoopVectorizationCostModel::CM_Unknown &&
         "CM decision should be taken at this point");
  if (Decision == LoopVectorizationCostModel::CM_Interleave)
    return vectorizeInterleaveGroup(Instr);

  Type *ScalarDataTy = getMemInstValueType(Instr);
  Type *DataTy = VectorType::get(ScalarDataTy, VF);
  Value *Ptr = getPointerOperand(Instr);
  unsigned Alignment = getMemInstAlignment(Instr);
  // An alignment of 0 means target abi alignment. We need to use the scalar's
  // target abi alignment in such a case.
  const DataLayout &DL = Instr->getModule()->getDataLayout();
  if (!Alignment)
    Alignment = DL.getABITypeAlignment(ScalarDataTy);
  unsigned AddressSpace = getMemInstAddressSpace(Instr);

  // Scalarize the memory instruction if necessary.
  if (Decision == LoopVectorizationCostModel::CM_Scalarize)
    return scalarizeInstruction(Instr, Legal->isScalarWithPredication(Instr));

  // Determine if the pointer operand of the access is either consecutive or
  // reverse consecutive.
  int ConsecutiveStride = Legal->isConsecutivePtr(Ptr);
  bool Reverse = ConsecutiveStride < 0;
  bool CreateGatherScatter =
      (Decision == LoopVectorizationCostModel::CM_GatherScatter);

  VectorParts VectorGep;

  // Handle consecutive loads/stores.
  GetElementPtrInst *Gep = getGEPInstruction(Ptr);
  if (ConsecutiveStride) {
    if (Gep) {
      unsigned NumOperands = Gep->getNumOperands();
#ifndef NDEBUG
      // The original GEP that identified as a consecutive memory access
      // should have only one loop-variant operand.
      unsigned NumOfLoopVariantOps = 0;
      for (unsigned i = 0; i < NumOperands; ++i)
        if (!PSE.getSE()->isLoopInvariant(PSE.getSCEV(Gep->getOperand(i)),
                                          OrigLoop))
          NumOfLoopVariantOps++;
      assert(NumOfLoopVariantOps == 1 &&
             "Consecutive GEP should have only one loop-variant operand");
#endif
      GetElementPtrInst *Gep2 = cast<GetElementPtrInst>(Gep->clone());
      Gep2->setName("gep.indvar");

      // A new GEP is created for a 0-lane value of the first unroll iteration.
      // The GEPs for the rest of the unroll iterations are computed below as an
      // offset from this GEP.
      for (unsigned i = 0; i < NumOperands; ++i)
        // We can apply getScalarValue() for all GEP indices. It returns an
        // original value for loop-invariant operand and 0-lane for consecutive
        // operand.
        Gep2->setOperand(i, getScalarValue(Gep->getOperand(i),
                                           0, /* First unroll iteration */
                                           0  /* 0-lane of the vector */ ));
      setDebugLocFromInst(Builder, Gep);
      Ptr = Builder.Insert(Gep2);

    } else { // No GEP
      setDebugLocFromInst(Builder, Ptr);
      Ptr = getScalarValue(Ptr, 0, 0);
    }
  } else {
    // At this point we should vector version of GEP for Gather or Scatter
    assert(CreateGatherScatter && "The instruction should be scalarized");
    if (Gep) {
      // Vectorizing GEP, across UF parts. We want to get a vector value for base
      // and each index that's defined inside the loop, even if it is
      // loop-invariant but wasn't hoisted out. Otherwise we want to keep them
      // scalar.
      SmallVector<VectorParts, 4> OpsV;
      for (Value *Op : Gep->operands()) {
        Instruction *SrcInst = dyn_cast<Instruction>(Op);
        if (SrcInst && OrigLoop->contains(SrcInst))
          OpsV.push_back(getVectorValue(Op));
        else
          OpsV.push_back(VectorParts(UF, Op));
      }
      for (unsigned Part = 0; Part < UF; ++Part) {
        SmallVector<Value *, 4> Ops;
        Value *GEPBasePtr = OpsV[0][Part];
        for (unsigned i = 1; i < Gep->getNumOperands(); i++)
          Ops.push_back(OpsV[i][Part]);
        Value *NewGep =  Builder.CreateGEP(GEPBasePtr, Ops, "VectorGep");
        cast<GetElementPtrInst>(NewGep)->setIsInBounds(Gep->isInBounds());
        assert(NewGep->getType()->isVectorTy() && "Expected vector GEP");

        NewGep =
            Builder.CreateBitCast(NewGep, VectorType::get(Ptr->getType(), VF));
        VectorGep.push_back(NewGep);
      }
    } else
      VectorGep = getVectorValue(Ptr);
  }

  VectorParts Mask = createBlockInMask(Instr->getParent());
  // Handle Stores:
  if (SI) {
    assert(!Legal->isUniform(SI->getPointerOperand()) &&
           "We do not allow storing to uniform addresses");
    setDebugLocFromInst(Builder, SI);
    // We don't want to update the value in the map as it might be used in
    // another expression. So don't use a reference type for "StoredVal".
    VectorParts StoredVal = getVectorValue(SI->getValueOperand());

    for (unsigned Part = 0; Part < UF; ++Part) {
      Instruction *NewSI = nullptr;
      if (CreateGatherScatter) {
        Value *MaskPart = Legal->isMaskRequired(SI) ? Mask[Part] : nullptr;
        NewSI = Builder.CreateMaskedScatter(StoredVal[Part], VectorGep[Part],
                                            Alignment, MaskPart);
      } else {
        // Calculate the pointer for the specific unroll-part.
        Value *PartPtr =
            Builder.CreateGEP(nullptr, Ptr, Builder.getInt32(Part * VF));

        if (Reverse) {
          // If we store to reverse consecutive memory locations, then we need
          // to reverse the order of elements in the stored value.
          StoredVal[Part] = reverseVector(StoredVal[Part]);
          // If the address is consecutive but reversed, then the
          // wide store needs to start at the last vector element.
          PartPtr =
              Builder.CreateGEP(nullptr, Ptr, Builder.getInt32(-Part * VF));
          PartPtr =
              Builder.CreateGEP(nullptr, PartPtr, Builder.getInt32(1 - VF));
          Mask[Part] = reverseVector(Mask[Part]);
        }

        Value *VecPtr =
            Builder.CreateBitCast(PartPtr, DataTy->getPointerTo(AddressSpace));

        if (Legal->isMaskRequired(SI))
          NewSI = Builder.CreateMaskedStore(StoredVal[Part], VecPtr, Alignment,
                                            Mask[Part]);
        else
          NewSI =
              Builder.CreateAlignedStore(StoredVal[Part], VecPtr, Alignment);
      }
      addMetadata(NewSI, SI);
    }
    return;
  }

  // Handle loads.
  assert(LI && "Must have a load instruction");
  setDebugLocFromInst(Builder, LI);
  VectorParts Entry(UF);
  for (unsigned Part = 0; Part < UF; ++Part) {
    Instruction *NewLI;
    if (CreateGatherScatter) {
      Value *MaskPart = Legal->isMaskRequired(LI) ? Mask[Part] : nullptr;
      NewLI = Builder.CreateMaskedGather(VectorGep[Part], Alignment, MaskPart,
                                         0, "wide.masked.gather");
      Entry[Part] = NewLI;
    } else {
      // Calculate the pointer for the specific unroll-part.
      Value *PartPtr =
          Builder.CreateGEP(nullptr, Ptr, Builder.getInt32(Part * VF));

      if (Reverse) {
        // If the address is consecutive but reversed, then the
        // wide load needs to start at the last vector element.
        PartPtr = Builder.CreateGEP(nullptr, Ptr, Builder.getInt32(-Part * VF));
        PartPtr = Builder.CreateGEP(nullptr, PartPtr, Builder.getInt32(1 - VF));
        Mask[Part] = reverseVector(Mask[Part]);
      }

      Value *VecPtr =
          Builder.CreateBitCast(PartPtr, DataTy->getPointerTo(AddressSpace));
      if (Legal->isMaskRequired(LI))
        NewLI = Builder.CreateMaskedLoad(VecPtr, Alignment, Mask[Part],
                                         UndefValue::get(DataTy),
                                         "wide.masked.load");
      else
        NewLI = Builder.CreateAlignedLoad(VecPtr, Alignment, "wide.load");
      Entry[Part] = Reverse ? reverseVector(NewLI) : NewLI;
    }
    addMetadata(NewLI, LI);
  }
  VectorLoopValueMap.initVector(Instr, Entry);
}

void InnerLoopVectorizer::scalarizeInstruction(Instruction *Instr,
                                               bool IfPredicateInstr) {
  assert(!Instr->getType()->isAggregateType() && "Can't handle vectors");
  DEBUG(dbgs() << "LV: Scalarizing"
               << (IfPredicateInstr ? " and predicating:" : ":") << *Instr
               << '\n');
  // Holds vector parameters or scalars, in case of uniform vals.
  SmallVector<VectorParts, 4> Params;

  setDebugLocFromInst(Builder, Instr);

  // Does this instruction return a value ?
  bool IsVoidRetTy = Instr->getType()->isVoidTy();

  // Initialize a new scalar map entry.
  ScalarParts Entry(UF);

  VectorParts Cond;
  if (IfPredicateInstr)
    Cond = createBlockInMask(Instr->getParent());

  // Determine the number of scalars we need to generate for each unroll
  // iteration. If the instruction is uniform, we only need to generate the
  // first lane. Otherwise, we generate all VF values.
  unsigned Lanes = Cost->isUniformAfterVectorization(Instr, VF) ? 1 : VF;

  // For each vector unroll 'part':
  for (unsigned Part = 0; Part < UF; ++Part) {
    Entry[Part].resize(VF);
    // For each scalar that we create:
    for (unsigned Lane = 0; Lane < Lanes; ++Lane) {

      // Start if-block.
      Value *Cmp = nullptr;
      if (IfPredicateInstr) {
        Cmp = Builder.CreateExtractElement(Cond[Part], Builder.getInt32(Lane));
        Cmp = Builder.CreateICmp(ICmpInst::ICMP_EQ, Cmp,
                                 ConstantInt::get(Cmp->getType(), 1));
      }

      Instruction *Cloned = Instr->clone();
      if (!IsVoidRetTy)
        Cloned->setName(Instr->getName() + ".cloned");

      // Replace the operands of the cloned instructions with their scalar
      // equivalents in the new loop.
      for (unsigned op = 0, e = Instr->getNumOperands(); op != e; ++op) {
        auto *NewOp = getScalarValue(Instr->getOperand(op), Part, Lane);
        Cloned->setOperand(op, NewOp);
      }
      addNewMetadata(Cloned, Instr);

      // Place the cloned scalar in the new loop.
      Builder.Insert(Cloned);

      // Add the cloned scalar to the scalar map entry.
      Entry[Part][Lane] = Cloned;

      // If we just cloned a new assumption, add it the assumption cache.
      if (auto *II = dyn_cast<IntrinsicInst>(Cloned))
        if (II->getIntrinsicID() == Intrinsic::assume)
          AC->registerAssumption(II);

      // End if-block.
      if (IfPredicateInstr)
        PredicatedInstructions.push_back(std::make_pair(Cloned, Cmp));
    }
  }
  VectorLoopValueMap.initScalar(Instr, Entry);
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

  IRBuilder<> Builder(&*Header->getFirstInsertionPt());
  Instruction *OldInst = getDebugLocFromInstOrOperands(OldInduction);
  setDebugLocFromInst(Builder, OldInst);
  auto *Induction = Builder.CreatePHI(Start->getType(), 2, "index");

  Builder.SetInsertPoint(Latch->getTerminator());
  setDebugLocFromInst(Builder, OldInst);

  // Create i+1 and fill the PHINode.
  Value *Next = Builder.CreateAdd(Induction, Step, "index.next");
  Induction->addIncoming(Start, L->getLoopPreheader());
  Induction->addIncoming(Next, Latch);
  // Create the compare.
  Value *ICmp = Builder.CreateICmpEQ(Next, End);
  Builder.CreateCondBr(ICmp, L->getExitBlock(), Header);

  // Now we have two terminators. Remove the old one from the block.
  Latch->getTerminator()->eraseFromParent();

  return Induction;
}

Value *InnerLoopVectorizer::getOrCreateTripCount(Loop *L) {
  if (TripCount)
    return TripCount;

  IRBuilder<> Builder(L->getLoopPreheader()->getTerminator());
  // Find the loop boundaries.
  ScalarEvolution *SE = PSE.getSE();
  const SCEV *BackedgeTakenCount = PSE.getBackedgeTakenCount();
  assert(BackedgeTakenCount != SE->getCouldNotCompute() &&
         "Invalid loop count");

  Type *IdxTy = Legal->getWidestInductionType();

  // The exit count might have the type of i64 while the phi is i32. This can
  // happen if we have an induction variable that is sign extended before the
  // compare. The only way that we get a backedge taken count is that the
  // induction variable was signed and as such will not overflow. In such a case
  // truncation is legal.
  if (BackedgeTakenCount->getType()->getPrimitiveSizeInBits() >
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

  // Now we need to generate the expression for the part of the loop that the
  // vectorized body will execute. This is equal to N - (N % Step) if scalar
  // iterations are not required for correctness, or N - Step, otherwise. Step
  // is equal to the vectorization factor (number of SIMD elements) times the
  // unroll factor (number of SIMD instructions).
  Constant *Step = ConstantInt::get(TC->getType(), VF * UF);
  Value *R = Builder.CreateURem(TC, Step, "n.mod.vf");

  // If there is a non-reversed interleaved group that may speculatively access
  // memory out-of-bounds, we need to ensure that there will be at least one
  // iteration of the scalar epilogue loop. Thus, if the step evenly divides
  // the trip count, we set the remainder to be equal to the step. If the step
  // does not evenly divide the trip count, no adjustment is necessary since
  // there will already be scalar iterations. Note that the minimum iterations
  // check ensures that N >= Step.
  if (VF > 1 && Legal->requiresScalarEpilogue()) {
    auto *IsZero = Builder.CreateICmpEQ(R, ConstantInt::get(R->getType(), 0));
    R = Builder.CreateSelect(IsZero, Step, R);
  }

  VectorTripCount = Builder.CreateSub(TC, R, "n.vec");

  return VectorTripCount;
}

void InnerLoopVectorizer::emitMinimumIterationCountCheck(Loop *L,
                                                         BasicBlock *Bypass) {
  Value *Count = getOrCreateTripCount(L);
  BasicBlock *BB = L->getLoopPreheader();
  IRBuilder<> Builder(BB->getTerminator());

  // Generate code to check that the loop's trip count that we computed by
  // adding one to the backedge-taken count will not overflow.
  Value *CheckMinIters = Builder.CreateICmpULT(
      Count, ConstantInt::get(Count->getType(), VF * UF), "min.iters.check");

  BasicBlock *NewBB =
      BB->splitBasicBlock(BB->getTerminator(), "min.iters.checked");
  // Update dominator tree immediately if the generated block is a
  // LoopBypassBlock because SCEV expansions to generate loop bypass
  // checks may query it before the current function is finished.
  DT->addNewBlock(NewBB, BB);
  if (L->getParentLoop())
    L->getParentLoop()->addBasicBlockToLoop(NewBB, *LI);
  ReplaceInstWithInst(BB->getTerminator(),
                      BranchInst::Create(Bypass, NewBB, CheckMinIters));
  LoopBypassBlocks.push_back(BB);
}

void InnerLoopVectorizer::emitVectorLoopEnteredCheck(Loop *L,
                                                     BasicBlock *Bypass) {
  Value *TC = getOrCreateVectorTripCount(L);
  BasicBlock *BB = L->getLoopPreheader();
  IRBuilder<> Builder(BB->getTerminator());

  // Now, compare the new count to zero. If it is zero skip the vector loop and
  // jump to the scalar loop.
  Value *Cmp = Builder.CreateICmpEQ(TC, Constant::getNullValue(TC->getType()),
                                    "cmp.zero");

  // Generate code to check that the loop's trip count that we computed by
  // adding one to the backedge-taken count will not overflow.
  BasicBlock *NewBB = BB->splitBasicBlock(BB->getTerminator(), "vector.ph");
  // Update dominator tree immediately if the generated block is a
  // LoopBypassBlock because SCEV expansions to generate loop bypass
  // checks may query it before the current function is finished.
  DT->addNewBlock(NewBB, BB);
  if (L->getParentLoop())
    L->getParentLoop()->addBasicBlockToLoop(NewBB, *LI);
  ReplaceInstWithInst(BB->getTerminator(),
                      BranchInst::Create(Bypass, NewBB, Cmp));
  LoopBypassBlocks.push_back(BB);
}

void InnerLoopVectorizer::emitSCEVChecks(Loop *L, BasicBlock *Bypass) {
  BasicBlock *BB = L->getLoopPreheader();

  // Generate the code to check that the SCEV assumptions that we made.
  // We want the new basic block to start at the first instruction in a
  // sequence of instructions that form a check.
  SCEVExpander Exp(*PSE.getSE(), Bypass->getModule()->getDataLayout(),
                   "scev.check");
  Value *SCEVCheck =
      Exp.expandCodeForPredicate(&PSE.getUnionPredicate(), BB->getTerminator());

  if (auto *C = dyn_cast<ConstantInt>(SCEVCheck))
    if (C->isZero())
      return;

  // Create a new block containing the stride check.
  BB->setName("vector.scevcheck");
  auto *NewBB = BB->splitBasicBlock(BB->getTerminator(), "vector.ph");
  // Update dominator tree immediately if the generated block is a
  // LoopBypassBlock because SCEV expansions to generate loop bypass
  // checks may query it before the current function is finished.
  DT->addNewBlock(NewBB, BB);
  if (L->getParentLoop())
    L->getParentLoop()->addBasicBlockToLoop(NewBB, *LI);
  ReplaceInstWithInst(BB->getTerminator(),
                      BranchInst::Create(Bypass, NewBB, SCEVCheck));
  LoopBypassBlocks.push_back(BB);
  AddedSafetyChecks = true;
}

void InnerLoopVectorizer::emitMemRuntimeChecks(Loop *L, BasicBlock *Bypass) {
  BasicBlock *BB = L->getLoopPreheader();

  // Generate the code that checks in runtime if arrays overlap. We put the
  // checks into a separate block to make the more common case of few elements
  // faster.
  Instruction *FirstCheckInst;
  Instruction *MemRuntimeCheck;
  std::tie(FirstCheckInst, MemRuntimeCheck) =
      Legal->getLAI()->addRuntimeChecks(BB->getTerminator());
  if (!MemRuntimeCheck)
    return;

  // Create a new block containing the memory check.
  BB->setName("vector.memcheck");
  auto *NewBB = BB->splitBasicBlock(BB->getTerminator(), "vector.ph");
  // Update dominator tree immediately if the generated block is a
  // LoopBypassBlock because SCEV expansions to generate loop bypass
  // checks may query it before the current function is finished.
  DT->addNewBlock(NewBB, BB);
  if (L->getParentLoop())
    L->getParentLoop()->addBasicBlockToLoop(NewBB, *LI);
  ReplaceInstWithInst(BB->getTerminator(),
                      BranchInst::Create(Bypass, NewBB, MemRuntimeCheck));
  LoopBypassBlocks.push_back(BB);
  AddedSafetyChecks = true;

  // We currently don't use LoopVersioning for the actual loop cloning but we
  // still use it to add the noalias metadata.
  LVer = llvm::make_unique<LoopVersioning>(*Legal->getLAI(), OrigLoop, LI, DT,
                                           PSE.getSE());
  LVer->prepareNoAliasMetadata();
}

void InnerLoopVectorizer::createEmptyLoop() {
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

  BasicBlock *OldBasicBlock = OrigLoop->getHeader();
  BasicBlock *VectorPH = OrigLoop->getLoopPreheader();
  BasicBlock *ExitBlock = OrigLoop->getExitBlock();
  assert(VectorPH && "Invalid loop structure");
  assert(ExitBlock && "Must have an exit block");

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

  // Split the single block loop into the two loop structure described above.
  BasicBlock *VecBody =
      VectorPH->splitBasicBlock(VectorPH->getTerminator(), "vector.body");
  BasicBlock *MiddleBlock =
      VecBody->splitBasicBlock(VecBody->getTerminator(), "middle.block");
  BasicBlock *ScalarPH =
      MiddleBlock->splitBasicBlock(MiddleBlock->getTerminator(), "scalar.ph");

  // Create and register the new vector loop.
  Loop *Lp = new Loop();
  Loop *ParentLoop = OrigLoop->getParentLoop();

  // Insert the new loop into the loop nest and register the new basic blocks
  // before calling any utilities such as SCEV that require valid LoopInfo.
  if (ParentLoop) {
    ParentLoop->addChildLoop(Lp);
    ParentLoop->addBasicBlockToLoop(ScalarPH, *LI);
    ParentLoop->addBasicBlockToLoop(MiddleBlock, *LI);
  } else {
    LI->addTopLevelLoop(Lp);
  }
  Lp->addBasicBlockToLoop(VecBody, *LI);

  // Find the loop boundaries.
  Value *Count = getOrCreateTripCount(Lp);

  Value *StartIdx = ConstantInt::get(IdxTy, 0);

  // We need to test whether the backedge-taken count is uint##_max. Adding one
  // to it will cause overflow and an incorrect loop trip count in the vector
  // body. In case of overflow we want to directly jump to the scalar remainder
  // loop.
  emitMinimumIterationCountCheck(Lp, ScalarPH);
  // Now, compare the new count to zero. If it is zero skip the vector loop and
  // jump to the scalar loop.
  emitVectorLoopEnteredCheck(Lp, ScalarPH);
  // Generate the code to check any assumptions that we've made for SCEV
  // expressions.
  emitSCEVChecks(Lp, ScalarPH);

  // Generate the code that checks in runtime if arrays overlap. We put the
  // checks into a separate block to make the more common case of few elements
  // faster.
  emitMemRuntimeChecks(Lp, ScalarPH);

  // Generate the induction variable.
  // The loop step is equal to the vectorization factor (num of SIMD elements)
  // times the unroll factor (num of SIMD instructions).
  Value *CountRoundDown = getOrCreateVectorTripCount(Lp);
  Constant *Step = ConstantInt::get(IdxTy, VF * UF);
  Induction =
      createInductionVariable(Lp, StartIdx, CountRoundDown, Step,
                              getDebugLocFromInstOrOperands(OldInduction));

  // We are going to resume the execution of the scalar loop.
  // Go over all of the induction variables that we found and fix the
  // PHIs that are left in the scalar version of the loop.
  // The starting values of PHI nodes depend on the counter of the last
  // iteration in the vectorized loop.
  // If we come from a bypass edge then we need to start from the original
  // start value.

  // This variable saves the new starting index for the scalar loop. It is used
  // to test if there are any tail iterations left once the vector loop has
  // completed.
  LoopVectorizationLegality::InductionList *List = Legal->getInductionVars();
  for (auto &InductionEntry : *List) {
    PHINode *OrigPhi = InductionEntry.first;
    InductionDescriptor II = InductionEntry.second;

    // Create phi nodes to merge from the  backedge-taken check block.
    PHINode *BCResumeVal = PHINode::Create(
        OrigPhi->getType(), 3, "bc.resume.val", ScalarPH->getTerminator());
    Value *&EndValue = IVEndValues[OrigPhi];
    if (OrigPhi == OldInduction) {
      // We know what the end value is.
      EndValue = CountRoundDown;
    } else {
      IRBuilder<> B(LoopBypassBlocks.back()->getTerminator());
      Type *StepType = II.getStep()->getType();
      Instruction::CastOps CastOp =
        CastInst::getCastOpcode(CountRoundDown, true, StepType, true);
      Value *CRD = B.CreateCast(CastOp, CountRoundDown, StepType, "cast.crd");
      const DataLayout &DL = OrigLoop->getHeader()->getModule()->getDataLayout();
      EndValue = II.transform(B, CRD, PSE.getSE(), DL);
      EndValue->setName("ind.end");
    }

    // The new PHI merges the original incoming value, in case of a bypass,
    // or the value at the end of the vectorized loop.
    BCResumeVal->addIncoming(EndValue, MiddleBlock);

    // Fix the scalar body counter (PHI node).
    unsigned BlockIdx = OrigPhi->getBasicBlockIndex(ScalarPH);

    // The old induction's phi node in the scalar body needs the truncated
    // value.
    for (BasicBlock *BB : LoopBypassBlocks)
      BCResumeVal->addIncoming(II.getStartValue(), BB);
    OrigPhi->setIncomingValue(BlockIdx, BCResumeVal);
  }

  // Add a check in the middle block to see if we have completed
  // all of the iterations in the first vector loop.
  // If (N - N%VF) == N, then we *don't* need to run the remainder.
  Value *CmpN =
      CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_EQ, Count,
                      CountRoundDown, "cmp.n", MiddleBlock->getTerminator());
  ReplaceInstWithInst(MiddleBlock->getTerminator(),
                      BranchInst::Create(ExitBlock, ScalarPH, CmpN));

  // Get ready to start creating new instructions into the vectorized body.
  Builder.SetInsertPoint(&*VecBody->getFirstInsertionPt());

  // Save the state.
  LoopVectorPreHeader = Lp->getLoopPreheader();
  LoopScalarPreHeader = ScalarPH;
  LoopMiddleBlock = MiddleBlock;
  LoopExitBlock = ExitBlock;
  LoopVectorBody = VecBody;
  LoopScalarBody = OldBasicBlock;

  // Keep all loop hints from the original loop on the vector loop (we'll
  // replace the vectorizer-specific hints below).
  if (MDNode *LID = OrigLoop->getLoopID())
    Lp->setLoopID(LID);

  LoopVectorizeHints Hints(Lp, true, *ORE);
  Hints.setAlreadyVectorized();
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

  assert(OrigLoop->getExitBlock() && "Expected a single exit block");

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
      Value *CountMinusOne = B.CreateSub(
          CountRoundDown, ConstantInt::get(CountRoundDown->getType(), 1));
      Value *CMO = B.CreateSExtOrTrunc(CountMinusOne, II.getStep()->getType(),
                                       "cast.cmo");
      Value *Escape = II.transform(B, CMO, PSE.getSE(), DL);
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
  static bool canHandle(Instruction *I) {
    return isa<InsertElementInst>(I) || isa<ExtractElementInst>(I) ||
           isa<ShuffleVectorInst>(I) || isa<GetElementPtrInst>(I);
  }
  static inline Instruction *getEmptyKey() {
    return DenseMapInfo<Instruction *>::getEmptyKey();
  }
  static inline Instruction *getTombstoneKey() {
    return DenseMapInfo<Instruction *>::getTombstoneKey();
  }
  static unsigned getHashValue(Instruction *I) {
    assert(canHandle(I) && "Unknown instruction!");
    return hash_combine(I->getOpcode(), hash_combine_range(I->value_op_begin(),
                                                           I->value_op_end()));
  }
  static bool isEqual(Instruction *LHS, Instruction *RHS) {
    if (LHS == getEmptyKey() || RHS == getEmptyKey() ||
        LHS == getTombstoneKey() || RHS == getTombstoneKey())
      return LHS == RHS;
    return LHS->isIdenticalTo(RHS);
  }
};
}

///\brief Perform cse of induction variable instructions.
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

/// \brief Estimate the overhead of scalarizing an instruction. This is a
/// convenience wrapper for the type-based getScalarizationOverhead API.
static unsigned getScalarizationOverhead(Instruction *I, unsigned VF,
                                         const TargetTransformInfo &TTI) {
  if (VF == 1)
    return 0;

  unsigned Cost = 0;
  Type *RetTy = ToVectorTy(I->getType(), VF);
  if (!RetTy->isVoidTy())
    Cost += TTI.getScalarizationOverhead(RetTy, true, false);

  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    SmallVector<const Value *, 4> Operands(CI->arg_operands());
    Cost += TTI.getOperandsScalarizationOverhead(Operands, VF);
  } else {
    SmallVector<const Value *, 4> Operands(I->operand_values());
    Cost += TTI.getOperandsScalarizationOverhead(Operands, VF);
  }

  return Cost;
}

// Estimate cost of a call instruction CI if it were vectorized with factor VF.
// Return the cost of the instruction, including scalarization overhead if it's
// needed. The flag NeedToScalarize shows if the call needs to be scalarized -
// i.e. either vector version isn't available, or is too expensive.
static unsigned getVectorCallCost(CallInst *CI, unsigned VF,
                                  const TargetTransformInfo &TTI,
                                  const TargetLibraryInfo *TLI,
                                  bool &NeedToScalarize) {
  Function *F = CI->getCalledFunction();
  StringRef FnName = CI->getCalledFunction()->getName();
  Type *ScalarRetTy = CI->getType();
  SmallVector<Type *, 4> Tys, ScalarTys;
  for (auto &ArgOp : CI->arg_operands())
    ScalarTys.push_back(ArgOp->getType());

  // Estimate cost of scalarized vector call. The source operands are assumed
  // to be vectors, so we need to extract individual elements from there,
  // execute VF scalar calls, and then gather the result into the vector return
  // value.
  unsigned ScalarCallCost = TTI.getCallInstrCost(F, ScalarRetTy, ScalarTys);
  if (VF == 1)
    return ScalarCallCost;

  // Compute corresponding vector type for return value and arguments.
  Type *RetTy = ToVectorTy(ScalarRetTy, VF);
  for (Type *ScalarTy : ScalarTys)
    Tys.push_back(ToVectorTy(ScalarTy, VF));

  // Compute costs of unpacking argument values for the scalar calls and
  // packing the return values to a vector.
  unsigned ScalarizationCost = getScalarizationOverhead(CI, VF, TTI);

  unsigned Cost = ScalarCallCost * VF + ScalarizationCost;

  // If we can't emit a vector call for this function, then the currently found
  // cost is the cost we need to return.
  NeedToScalarize = true;
  if (!TLI || !TLI->isFunctionVectorizable(FnName, VF) || CI->isNoBuiltin())
    return Cost;

  // If the corresponding vector cost is cheaper, return its cost.
  unsigned VectorCallCost = TTI.getCallInstrCost(nullptr, RetTy, Tys);
  if (VectorCallCost < Cost) {
    NeedToScalarize = false;
    return VectorCallCost;
  }
  return Cost;
}

// Estimate cost of an intrinsic call instruction CI if it were vectorized with
// factor VF.  Return the cost of the instruction, including scalarization
// overhead if it's needed.
static unsigned getVectorIntrinsicCost(CallInst *CI, unsigned VF,
                                       const TargetTransformInfo &TTI,
                                       const TargetLibraryInfo *TLI) {
  Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);
  assert(ID && "Expected intrinsic call!");

  Type *RetTy = ToVectorTy(CI->getType(), VF);
  SmallVector<Type *, 4> Tys;
  for (Value *ArgOperand : CI->arg_operands())
    Tys.push_back(ToVectorTy(ArgOperand->getType(), VF));

  FastMathFlags FMF;
  if (auto *FPMO = dyn_cast<FPMathOperator>(CI))
    FMF = FPMO->getFastMathFlags();

  return TTI.getIntrinsicInstrCost(ID, RetTy, Tys, FMF);
}

static Type *smallestIntegerVectorType(Type *T1, Type *T2) {
  auto *I1 = cast<IntegerType>(T1->getVectorElementType());
  auto *I2 = cast<IntegerType>(T2->getVectorElementType());
  return I1->getBitWidth() < I2->getBitWidth() ? T1 : T2;
}
static Type *largestIntegerVectorType(Type *T1, Type *T2) {
  auto *I1 = cast<IntegerType>(T1->getVectorElementType());
  auto *I2 = cast<IntegerType>(T2->getVectorElementType());
  return I1->getBitWidth() > I2->getBitWidth() ? T1 : T2;
}

void InnerLoopVectorizer::truncateToMinimalBitwidths() {
  // For every instruction `I` in MinBWs, truncate the operands, create a
  // truncated version of `I` and reextend its result. InstCombine runs
  // later and will remove any ext/trunc pairs.
  //
  SmallPtrSet<Value *, 4> Erased;
  for (const auto &KV : Cost->getMinimalBitwidths()) {
    // If the value wasn't vectorized, we must maintain the original scalar
    // type. The absence of the value from VectorLoopValueMap indicates that it
    // wasn't vectorized.
    if (!VectorLoopValueMap.hasVector(KV.first))
      continue;
    VectorParts &Parts = VectorLoopValueMap.getVector(KV.first);
    for (Value *&I : Parts) {
      if (Erased.count(I) || I->use_empty() || !isa<Instruction>(I))
        continue;
      Type *OriginalTy = I->getType();
      Type *ScalarTruncatedTy =
          IntegerType::get(OriginalTy->getContext(), KV.second);
      Type *TruncatedTy = VectorType::get(ScalarTruncatedTy,
                                          OriginalTy->getVectorNumElements());
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
        cast<BinaryOperator>(NewI)->copyIRFlags(I);
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
        auto Elements0 = SI->getOperand(0)->getType()->getVectorNumElements();
        auto *O0 = B.CreateZExtOrTrunc(
            SI->getOperand(0), VectorType::get(ScalarTruncatedTy, Elements0));
        auto Elements1 = SI->getOperand(1)->getType()->getVectorNumElements();
        auto *O1 = B.CreateZExtOrTrunc(
            SI->getOperand(1), VectorType::get(ScalarTruncatedTy, Elements1));

        NewI = B.CreateShuffleVector(O0, O1, SI->getMask());
      } else if (isa<LoadInst>(I)) {
        // Don't do anything with the operands, just extend the result.
        continue;
      } else if (auto *IE = dyn_cast<InsertElementInst>(I)) {
        auto Elements = IE->getOperand(0)->getType()->getVectorNumElements();
        auto *O0 = B.CreateZExtOrTrunc(
            IE->getOperand(0), VectorType::get(ScalarTruncatedTy, Elements));
        auto *O1 = B.CreateZExtOrTrunc(IE->getOperand(1), ScalarTruncatedTy);
        NewI = B.CreateInsertElement(O0, O1, IE->getOperand(2));
      } else if (auto *EE = dyn_cast<ExtractElementInst>(I)) {
        auto Elements = EE->getOperand(0)->getType()->getVectorNumElements();
        auto *O0 = B.CreateZExtOrTrunc(
            EE->getOperand(0), VectorType::get(ScalarTruncatedTy, Elements));
        NewI = B.CreateExtractElement(O0, EE->getOperand(2));
      } else {
        llvm_unreachable("Unhandled instruction type!");
      }

      // Lastly, extend the result.
      NewI->takeName(cast<Instruction>(I));
      Value *Res = B.CreateZExtOrTrunc(NewI, OriginalTy);
      I->replaceAllUsesWith(Res);
      cast<Instruction>(I)->eraseFromParent();
      Erased.insert(I);
      I = Res;
    }
  }

  // We'll have created a bunch of ZExts that are now parentless. Clean up.
  for (const auto &KV : Cost->getMinimalBitwidths()) {
    // If the value wasn't vectorized, we must maintain the original scalar
    // type. The absence of the value from VectorLoopValueMap indicates that it
    // wasn't vectorized.
    if (!VectorLoopValueMap.hasVector(KV.first))
      continue;
    VectorParts &Parts = VectorLoopValueMap.getVector(KV.first);
    for (Value *&I : Parts) {
      ZExtInst *Inst = dyn_cast<ZExtInst>(I);
      if (Inst && Inst->use_empty()) {
        Value *NewI = Inst->getOperand(0);
        Inst->eraseFromParent();
        I = NewI;
      }
    }
  }
}

void InnerLoopVectorizer::vectorizeLoop() {
  //===------------------------------------------------===//
  //
  // Notice: any optimization or new instruction that go
  // into the code below should be also be implemented in
  // the cost-model.
  //
  //===------------------------------------------------===//
  Constant *Zero = Builder.getInt32(0);

  // In order to support recurrences we need to be able to vectorize Phi nodes.
  // Phi nodes have cycles, so we need to vectorize them in two stages. First,
  // we create a new vector PHI node with no incoming edges. We use this value
  // when we vectorize all of the instructions that use the PHI. Next, after
  // all of the instructions in the block are complete we add the new incoming
  // edges to the PHI. At this point all of the instructions in the basic block
  // are vectorized, so we can use them to construct the PHI.
  PhiVector PHIsToFix;

  // Collect instructions from the original loop that will become trivially
  // dead in the vectorized loop. We don't need to vectorize these
  // instructions.
  collectTriviallyDeadInstructions();

  // Scan the loop in a topological order to ensure that defs are vectorized
  // before users.
  LoopBlocksDFS DFS(OrigLoop);
  DFS.perform(LI);

  // Vectorize all of the blocks in the original loop.
  for (BasicBlock *BB : make_range(DFS.beginRPO(), DFS.endRPO()))
    vectorizeBlockInLoop(BB, &PHIsToFix);

  // Insert truncates and extends for any truncated instructions as hints to
  // InstCombine.
  if (VF > 1)
    truncateToMinimalBitwidths();

  // At this point every instruction in the original loop is widened to a
  // vector form. Now we need to fix the recurrences in PHIsToFix. These PHI
  // nodes are currently empty because we did not want to introduce cycles.
  // This is the second stage of vectorizing recurrences.
  for (PHINode *Phi : PHIsToFix) {
    assert(Phi && "Unable to recover vectorized PHI");

    // Handle first-order recurrences that need to be fixed.
    if (Legal->isFirstOrderRecurrence(Phi)) {
      fixFirstOrderRecurrence(Phi);
      continue;
    }

    // If the phi node is not a first-order recurrence, it must be a reduction.
    // Get it's reduction variable descriptor.
    assert(Legal->isReductionVariable(Phi) &&
           "Unable to find the reduction variable");
    RecurrenceDescriptor RdxDesc = (*Legal->getReductionVars())[Phi];

    RecurrenceDescriptor::RecurrenceKind RK = RdxDesc.getRecurrenceKind();
    TrackingVH<Value> ReductionStartValue = RdxDesc.getRecurrenceStartValue();
    Instruction *LoopExitInst = RdxDesc.getLoopExitInstr();
    RecurrenceDescriptor::MinMaxRecurrenceKind MinMaxKind =
        RdxDesc.getMinMaxRecurrenceKind();
    setDebugLocFromInst(Builder, ReductionStartValue);

    // We need to generate a reduction vector from the incoming scalar.
    // To do so, we need to generate the 'identity' vector and override
    // one of the elements with the incoming scalar reduction. We need
    // to do it in the vector-loop preheader.
    Builder.SetInsertPoint(LoopBypassBlocks[1]->getTerminator());

    // This is the vector-clone of the value that leaves the loop.
    const VectorParts &VectorExit = getVectorValue(LoopExitInst);
    Type *VecTy = VectorExit[0]->getType();

    // Find the reduction identity variable. Zero for addition, or, xor,
    // one for multiplication, -1 for And.
    Value *Identity;
    Value *VectorStart;
    if (RK == RecurrenceDescriptor::RK_IntegerMinMax ||
        RK == RecurrenceDescriptor::RK_FloatMinMax) {
      // MinMax reduction have the start value as their identify.
      if (VF == 1) {
        VectorStart = Identity = ReductionStartValue;
      } else {
        VectorStart = Identity =
            Builder.CreateVectorSplat(VF, ReductionStartValue, "minmax.ident");
      }
    } else {
      // Handle other reduction kinds:
      Constant *Iden = RecurrenceDescriptor::getRecurrenceIdentity(
          RK, VecTy->getScalarType());
      if (VF == 1) {
        Identity = Iden;
        // This vector is the Identity vector where the first element is the
        // incoming scalar reduction.
        VectorStart = ReductionStartValue;
      } else {
        Identity = ConstantVector::getSplat(VF, Iden);

        // This vector is the Identity vector where the first element is the
        // incoming scalar reduction.
        VectorStart =
            Builder.CreateInsertElement(Identity, ReductionStartValue, Zero);
      }
    }

    // Fix the vector-loop phi.

    // Reductions do not have to start at zero. They can start with
    // any loop invariant values.
    const VectorParts &VecRdxPhi = getVectorValue(Phi);
    BasicBlock *Latch = OrigLoop->getLoopLatch();
    Value *LoopVal = Phi->getIncomingValueForBlock(Latch);
    const VectorParts &Val = getVectorValue(LoopVal);
    for (unsigned part = 0; part < UF; ++part) {
      // Make sure to add the reduction stat value only to the
      // first unroll part.
      Value *StartVal = (part == 0) ? VectorStart : Identity;
      cast<PHINode>(VecRdxPhi[part])
          ->addIncoming(StartVal, LoopVectorPreHeader);
      cast<PHINode>(VecRdxPhi[part])
          ->addIncoming(Val[part], LoopVectorBody);
    }

    // Before each round, move the insertion point right between
    // the PHIs and the values we are going to write.
    // This allows us to write both PHINodes and the extractelement
    // instructions.
    Builder.SetInsertPoint(&*LoopMiddleBlock->getFirstInsertionPt());

    VectorParts &RdxParts = VectorLoopValueMap.getVector(LoopExitInst);
    setDebugLocFromInst(Builder, LoopExitInst);

    // If the vector reduction can be performed in a smaller type, we truncate
    // then extend the loop exit value to enable InstCombine to evaluate the
    // entire expression in the smaller type.
    if (VF > 1 && Phi->getType() != RdxDesc.getRecurrenceType()) {
      Type *RdxVecTy = VectorType::get(RdxDesc.getRecurrenceType(), VF);
      Builder.SetInsertPoint(LoopVectorBody->getTerminator());
      for (unsigned part = 0; part < UF; ++part) {
        Value *Trunc = Builder.CreateTrunc(RdxParts[part], RdxVecTy);
        Value *Extnd = RdxDesc.isSigned() ? Builder.CreateSExt(Trunc, VecTy)
                                          : Builder.CreateZExt(Trunc, VecTy);
        for (Value::user_iterator UI = RdxParts[part]->user_begin();
             UI != RdxParts[part]->user_end();)
          if (*UI != Trunc) {
            (*UI++)->replaceUsesOfWith(RdxParts[part], Extnd);
            RdxParts[part] = Extnd;
          } else {
            ++UI;
          }
      }
      Builder.SetInsertPoint(&*LoopMiddleBlock->getFirstInsertionPt());
      for (unsigned part = 0; part < UF; ++part)
        RdxParts[part] = Builder.CreateTrunc(RdxParts[part], RdxVecTy);
    }

    // Reduce all of the unrolled parts into a single vector.
    Value *ReducedPartRdx = RdxParts[0];
    unsigned Op = RecurrenceDescriptor::getRecurrenceBinOp(RK);
    setDebugLocFromInst(Builder, ReducedPartRdx);
    for (unsigned part = 1; part < UF; ++part) {
      if (Op != Instruction::ICmp && Op != Instruction::FCmp)
        // Floating point operations had to be 'fast' to enable the reduction.
        ReducedPartRdx = addFastMathFlag(
            Builder.CreateBinOp((Instruction::BinaryOps)Op, RdxParts[part],
                                ReducedPartRdx, "bin.rdx"));
      else
        ReducedPartRdx = RecurrenceDescriptor::createMinMaxOp(
            Builder, MinMaxKind, ReducedPartRdx, RdxParts[part]);
    }

    if (VF > 1) {
      // VF is a power of 2 so we can emit the reduction using log2(VF) shuffles
      // and vector ops, reducing the set of values being computed by half each
      // round.
      assert(isPowerOf2_32(VF) &&
             "Reduction emission only supported for pow2 vectors!");
      Value *TmpVec = ReducedPartRdx;
      SmallVector<Constant *, 32> ShuffleMask(VF, nullptr);
      for (unsigned i = VF; i != 1; i >>= 1) {
        // Move the upper half of the vector to the lower half.
        for (unsigned j = 0; j != i / 2; ++j)
          ShuffleMask[j] = Builder.getInt32(i / 2 + j);

        // Fill the rest of the mask with undef.
        std::fill(&ShuffleMask[i / 2], ShuffleMask.end(),
                  UndefValue::get(Builder.getInt32Ty()));

        Value *Shuf = Builder.CreateShuffleVector(
            TmpVec, UndefValue::get(TmpVec->getType()),
            ConstantVector::get(ShuffleMask), "rdx.shuf");

        if (Op != Instruction::ICmp && Op != Instruction::FCmp)
          // Floating point operations had to be 'fast' to enable the reduction.
          TmpVec = addFastMathFlag(Builder.CreateBinOp(
              (Instruction::BinaryOps)Op, TmpVec, Shuf, "bin.rdx"));
        else
          TmpVec = RecurrenceDescriptor::createMinMaxOp(Builder, MinMaxKind,
                                                        TmpVec, Shuf);
      }

      // The result is in the first element of the vector.
      ReducedPartRdx =
          Builder.CreateExtractElement(TmpVec, Builder.getInt32(0));

      // If the reduction can be performed in a smaller type, we need to extend
      // the reduction to the wider type before we branch to the original loop.
      if (Phi->getType() != RdxDesc.getRecurrenceType())
        ReducedPartRdx =
            RdxDesc.isSigned()
                ? Builder.CreateSExt(ReducedPartRdx, Phi->getType())
                : Builder.CreateZExt(ReducedPartRdx, Phi->getType());
    }

    // Create a phi node that merges control-flow from the backedge-taken check
    // block and the middle block.
    PHINode *BCBlockPhi = PHINode::Create(Phi->getType(), 2, "bc.merge.rdx",
                                          LoopScalarPreHeader->getTerminator());
    for (unsigned I = 0, E = LoopBypassBlocks.size(); I != E; ++I)
      BCBlockPhi->addIncoming(ReductionStartValue, LoopBypassBlocks[I]);
    BCBlockPhi->addIncoming(ReducedPartRdx, LoopMiddleBlock);

    // Now, we need to fix the users of the reduction variable
    // inside and outside of the scalar remainder loop.
    // We know that the loop is in LCSSA form. We need to update the
    // PHI nodes in the exit blocks.
    for (BasicBlock::iterator LEI = LoopExitBlock->begin(),
                              LEE = LoopExitBlock->end();
         LEI != LEE; ++LEI) {
      PHINode *LCSSAPhi = dyn_cast<PHINode>(LEI);
      if (!LCSSAPhi)
        break;

      // All PHINodes need to have a single entry edge, or two if
      // we already fixed them.
      assert(LCSSAPhi->getNumIncomingValues() < 3 && "Invalid LCSSA PHI");

      // We found a reduction value exit-PHI. Update it with the
      // incoming bypass edge.
      if (LCSSAPhi->getIncomingValue(0) == LoopExitInst)
        LCSSAPhi->addIncoming(ReducedPartRdx, LoopMiddleBlock);
    } // end of the LCSSA phi scan.

    // Fix the scalar loop reduction variable with the incoming reduction sum
    // from the vector body and from the backedge value.
    int IncomingEdgeBlockIdx =
        Phi->getBasicBlockIndex(OrigLoop->getLoopLatch());
    assert(IncomingEdgeBlockIdx >= 0 && "Invalid block index");
    // Pick the other block.
    int SelfEdgeBlockIdx = (IncomingEdgeBlockIdx ? 0 : 1);
    Phi->setIncomingValue(SelfEdgeBlockIdx, BCBlockPhi);
    Phi->setIncomingValue(IncomingEdgeBlockIdx, LoopExitInst);
  } // end of for each Phi in PHIsToFix.

  // Update the dominator tree.
  //
  // FIXME: After creating the structure of the new loop, the dominator tree is
  //        no longer up-to-date, and it remains that way until we update it
  //        here. An out-of-date dominator tree is problematic for SCEV,
  //        because SCEVExpander uses it to guide code generation. The
  //        vectorizer use SCEVExpanders in several places. Instead, we should
  //        keep the dominator tree up-to-date as we go.
  updateAnalysis();

  // Fix-up external users of the induction variables.
  for (auto &Entry : *Legal->getInductionVars())
    fixupIVUsers(Entry.first, Entry.second,
                 getOrCreateVectorTripCount(LI->getLoopFor(LoopVectorBody)),
                 IVEndValues[Entry.first], LoopMiddleBlock);

  fixLCSSAPHIs();
  predicateInstructions();

  // Remove redundant induction instructions.
  cse(LoopVectorBody);
}

void InnerLoopVectorizer::fixFirstOrderRecurrence(PHINode *Phi) {

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

  // Get the original loop preheader and single loop latch.
  auto *Preheader = OrigLoop->getLoopPreheader();
  auto *Latch = OrigLoop->getLoopLatch();

  // Get the initial and previous values of the scalar recurrence.
  auto *ScalarInit = Phi->getIncomingValueForBlock(Preheader);
  auto *Previous = Phi->getIncomingValueForBlock(Latch);

  // Create a vector from the initial value.
  auto *VectorInit = ScalarInit;
  if (VF > 1) {
    Builder.SetInsertPoint(LoopVectorPreHeader->getTerminator());
    VectorInit = Builder.CreateInsertElement(
        UndefValue::get(VectorType::get(VectorInit->getType(), VF)), VectorInit,
        Builder.getInt32(VF - 1), "vector.recur.init");
  }

  // We constructed a temporary phi node in the first phase of vectorization.
  // This phi node will eventually be deleted.
  VectorParts &PhiParts = VectorLoopValueMap.getVector(Phi);
  Builder.SetInsertPoint(cast<Instruction>(PhiParts[0]));

  // Create a phi node for the new recurrence. The current value will either be
  // the initial value inserted into a vector or loop-varying vector value.
  auto *VecPhi = Builder.CreatePHI(VectorInit->getType(), 2, "vector.recur");
  VecPhi->addIncoming(VectorInit, LoopVectorPreHeader);

  // Get the vectorized previous value. We ensured the previous values was an
  // instruction when detecting the recurrence.
  auto &PreviousParts = getVectorValue(Previous);

  // Set the insertion point to be after this instruction. We ensured the
  // previous value dominated all uses of the phi when detecting the
  // recurrence.
  Builder.SetInsertPoint(
      &*++BasicBlock::iterator(cast<Instruction>(PreviousParts[UF - 1])));

  // We will construct a vector for the recurrence by combining the values for
  // the current and previous iterations. This is the required shuffle mask.
  SmallVector<Constant *, 8> ShuffleMask(VF);
  ShuffleMask[0] = Builder.getInt32(VF - 1);
  for (unsigned I = 1; I < VF; ++I)
    ShuffleMask[I] = Builder.getInt32(I + VF - 1);

  // The vector from which to take the initial value for the current iteration
  // (actual or unrolled). Initially, this is the vector phi node.
  Value *Incoming = VecPhi;

  // Shuffle the current and previous vector and update the vector parts.
  for (unsigned Part = 0; Part < UF; ++Part) {
    auto *Shuffle =
        VF > 1
            ? Builder.CreateShuffleVector(Incoming, PreviousParts[Part],
                                          ConstantVector::get(ShuffleMask))
            : Incoming;
    PhiParts[Part]->replaceAllUsesWith(Shuffle);
    cast<Instruction>(PhiParts[Part])->eraseFromParent();
    PhiParts[Part] = Shuffle;
    Incoming = PreviousParts[Part];
  }

  // Fix the latch value of the new recurrence in the vector loop.
  VecPhi->addIncoming(Incoming, LI->getLoopFor(LoopVectorBody)->getLoopLatch());

  // Extract the last vector element in the middle block. This will be the
  // initial value for the recurrence when jumping to the scalar loop.
  auto *Extract = Incoming;
  if (VF > 1) {
    Builder.SetInsertPoint(LoopMiddleBlock->getTerminator());
    Extract = Builder.CreateExtractElement(Extract, Builder.getInt32(VF - 1),
                                           "vector.recur.extract");
  }

  // Fix the initial value of the original recurrence in the scalar loop.
  Builder.SetInsertPoint(&*LoopScalarPreHeader->begin());
  auto *Start = Builder.CreatePHI(Phi->getType(), 2, "scalar.recur.init");
  for (auto *BB : predecessors(LoopScalarPreHeader)) {
    auto *Incoming = BB == LoopMiddleBlock ? Extract : ScalarInit;
    Start->addIncoming(Incoming, BB);
  }

  Phi->setIncomingValue(Phi->getBasicBlockIndex(LoopScalarPreHeader), Start);
  Phi->setName("scalar.recur");

  // Finally, fix users of the recurrence outside the loop. The users will need
  // either the last value of the scalar recurrence or the last value of the
  // vector recurrence we extracted in the middle block. Since the loop is in
  // LCSSA form, we just need to find the phi node for the original scalar
  // recurrence in the exit block, and then add an edge for the middle block.
  for (auto &I : *LoopExitBlock) {
    auto *LCSSAPhi = dyn_cast<PHINode>(&I);
    if (!LCSSAPhi)
      break;
    if (LCSSAPhi->getIncomingValue(0) == Phi) {
      LCSSAPhi->addIncoming(Extract, LoopMiddleBlock);
      break;
    }
  }
}

void InnerLoopVectorizer::fixLCSSAPHIs() {
  for (Instruction &LEI : *LoopExitBlock) {
    auto *LCSSAPhi = dyn_cast<PHINode>(&LEI);
    if (!LCSSAPhi)
      break;
    if (LCSSAPhi->getNumIncomingValues() == 1)
      LCSSAPhi->addIncoming(UndefValue::get(LCSSAPhi->getType()),
                            LoopMiddleBlock);
  }
}

void InnerLoopVectorizer::collectTriviallyDeadInstructions() {
  BasicBlock *Latch = OrigLoop->getLoopLatch();

  // We create new control-flow for the vectorized loop, so the original
  // condition will be dead after vectorization if it's only used by the
  // branch.
  auto *Cmp = dyn_cast<Instruction>(Latch->getTerminator()->getOperand(0));
  if (Cmp && Cmp->hasOneUse())
    DeadInstructions.insert(Cmp);

  // We create new "steps" for induction variable updates to which the original
  // induction variables map. An original update instruction will be dead if
  // all its users except the induction variable are dead.
  for (auto &Induction : *Legal->getInductionVars()) {
    PHINode *Ind = Induction.first;
    auto *IndUpdate = cast<Instruction>(Ind->getIncomingValueForBlock(Latch));
    if (all_of(IndUpdate->users(), [&](User *U) -> bool {
          return U == Ind || DeadInstructions.count(cast<Instruction>(U));
        }))
      DeadInstructions.insert(IndUpdate);
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

      // We can't sink an instruction if it is a phi node, is already in the
      // predicated block, is not in the loop, or may have side effects.
      if (!I || isa<PHINode>(I) || I->getParent() == PredBB ||
          !VectorLoop->contains(I) || I->mayHaveSideEffects())
        continue;

      // It's legal to sink the instruction if all its uses occur in the
      // predicated block. Otherwise, there's nothing to do yet, and we may
      // need to reanalyze the instruction.
      if (!all_of(I->uses(), isBlockOfUsePredicated)) {
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

void InnerLoopVectorizer::predicateInstructions() {

  // For each instruction I marked for predication on value C, split I into its
  // own basic block to form an if-then construct over C. Since I may be fed by
  // an extractelement instruction or other scalar operand, we try to
  // iteratively sink its scalar operands into the predicated block. If I feeds
  // an insertelement instruction, we try to move this instruction into the
  // predicated block as well. For non-void types, a phi node will be created
  // for the resulting value (either vector or scalar).
  //
  // So for some predicated instruction, e.g. the conditional sdiv in:
  //
  // for.body:
  //  ...
  //  %add = add nsw i32 %mul, %0
  //  %cmp5 = icmp sgt i32 %2, 7
  //  br i1 %cmp5, label %if.then, label %if.end
  //
  // if.then:
  //  %div = sdiv i32 %0, %1
  //  br label %if.end
  //
  // if.end:
  //  %x.0 = phi i32 [ %div, %if.then ], [ %add, %for.body ]
  //
  // the sdiv at this point is scalarized and if-converted using a select.
  // The inactive elements in the vector are not used, but the predicated
  // instruction is still executed for all vector elements, essentially:
  //
  // vector.body:
  //  ...
  //  %17 = add nsw <2 x i32> %16, %wide.load
  //  %29 = extractelement <2 x i32> %wide.load, i32 0
  //  %30 = extractelement <2 x i32> %wide.load51, i32 0
  //  %31 = sdiv i32 %29, %30
  //  %32 = insertelement <2 x i32> undef, i32 %31, i32 0
  //  %35 = extractelement <2 x i32> %wide.load, i32 1
  //  %36 = extractelement <2 x i32> %wide.load51, i32 1
  //  %37 = sdiv i32 %35, %36
  //  %38 = insertelement <2 x i32> %32, i32 %37, i32 1
  //  %predphi = select <2 x i1> %26, <2 x i32> %38, <2 x i32> %17
  //
  // Predication will now re-introduce the original control flow to avoid false
  // side-effects by the sdiv instructions on the inactive elements, yielding
  // (after cleanup):
  //
  // vector.body:
  //  ...
  //  %5 = add nsw <2 x i32> %4, %wide.load
  //  %8 = icmp sgt <2 x i32> %wide.load52, <i32 7, i32 7>
  //  %9 = extractelement <2 x i1> %8, i32 0
  //  br i1 %9, label %pred.sdiv.if, label %pred.sdiv.continue
  //
  // pred.sdiv.if:
  //  %10 = extractelement <2 x i32> %wide.load, i32 0
  //  %11 = extractelement <2 x i32> %wide.load51, i32 0
  //  %12 = sdiv i32 %10, %11
  //  %13 = insertelement <2 x i32> undef, i32 %12, i32 0
  //  br label %pred.sdiv.continue
  //
  // pred.sdiv.continue:
  //  %14 = phi <2 x i32> [ undef, %vector.body ], [ %13, %pred.sdiv.if ]
  //  %15 = extractelement <2 x i1> %8, i32 1
  //  br i1 %15, label %pred.sdiv.if54, label %pred.sdiv.continue55
  //
  // pred.sdiv.if54:
  //  %16 = extractelement <2 x i32> %wide.load, i32 1
  //  %17 = extractelement <2 x i32> %wide.load51, i32 1
  //  %18 = sdiv i32 %16, %17
  //  %19 = insertelement <2 x i32> %14, i32 %18, i32 1
  //  br label %pred.sdiv.continue55
  //
  // pred.sdiv.continue55:
  //  %20 = phi <2 x i32> [ %14, %pred.sdiv.continue ], [ %19, %pred.sdiv.if54 ]
  //  %predphi = select <2 x i1> %8, <2 x i32> %20, <2 x i32> %5

  for (auto KV : PredicatedInstructions) {
    BasicBlock::iterator I(KV.first);
    BasicBlock *Head = I->getParent();
    auto *BB = SplitBlock(Head, &*std::next(I), DT, LI);
    auto *T = SplitBlockAndInsertIfThen(KV.second, &*I, /*Unreachable=*/false,
                                        /*BranchWeights=*/nullptr, DT, LI);
    I->moveBefore(T);
    sinkScalarOperands(&*I);

    I->getParent()->setName(Twine("pred.") + I->getOpcodeName() + ".if");
    BB->setName(Twine("pred.") + I->getOpcodeName() + ".continue");

    // If the instruction is non-void create a Phi node at reconvergence point.
    if (!I->getType()->isVoidTy()) {
      Value *IncomingTrue = nullptr;
      Value *IncomingFalse = nullptr;

      if (I->hasOneUse() && isa<InsertElementInst>(*I->user_begin())) {
        // If the predicated instruction is feeding an insert-element, move it
        // into the Then block; Phi node will be created for the vector.
        InsertElementInst *IEI = cast<InsertElementInst>(*I->user_begin());
        IEI->moveBefore(T);
        IncomingTrue = IEI; // the new vector with the inserted element.
        IncomingFalse = IEI->getOperand(0); // the unmodified vector
      } else {
        // Phi node will be created for the scalar predicated instruction.
        IncomingTrue = &*I;
        IncomingFalse = UndefValue::get(I->getType());
      }

      BasicBlock *PostDom = I->getParent()->getSingleSuccessor();
      assert(PostDom && "Then block has multiple successors");
      PHINode *Phi =
          PHINode::Create(IncomingTrue->getType(), 2, "", &PostDom->front());
      IncomingTrue->replaceAllUsesWith(Phi);
      Phi->addIncoming(IncomingFalse, Head);
      Phi->addIncoming(IncomingTrue, I->getParent());
    }
  }

  DEBUG(DT->verifyDomTree());
}

InnerLoopVectorizer::VectorParts
InnerLoopVectorizer::createEdgeMask(BasicBlock *Src, BasicBlock *Dst) {
  assert(is_contained(predecessors(Dst), Src) && "Invalid edge");

  // Look for cached value.
  std::pair<BasicBlock *, BasicBlock *> Edge(Src, Dst);
  EdgeMaskCache::iterator ECEntryIt = MaskCache.find(Edge);
  if (ECEntryIt != MaskCache.end())
    return ECEntryIt->second;

  VectorParts SrcMask = createBlockInMask(Src);

  // The terminator has to be a branch inst!
  BranchInst *BI = dyn_cast<BranchInst>(Src->getTerminator());
  assert(BI && "Unexpected terminator found");

  if (BI->isConditional()) {
    VectorParts EdgeMask = getVectorValue(BI->getCondition());

    if (BI->getSuccessor(0) != Dst)
      for (unsigned part = 0; part < UF; ++part)
        EdgeMask[part] = Builder.CreateNot(EdgeMask[part]);

    for (unsigned part = 0; part < UF; ++part)
      EdgeMask[part] = Builder.CreateAnd(EdgeMask[part], SrcMask[part]);

    MaskCache[Edge] = EdgeMask;
    return EdgeMask;
  }

  MaskCache[Edge] = SrcMask;
  return SrcMask;
}

InnerLoopVectorizer::VectorParts
InnerLoopVectorizer::createBlockInMask(BasicBlock *BB) {
  assert(OrigLoop->contains(BB) && "Block is not a part of a loop");

  // Loop incoming mask is all-one.
  if (OrigLoop->getHeader() == BB) {
    Value *C = ConstantInt::get(IntegerType::getInt1Ty(BB->getContext()), 1);
    return getVectorValue(C);
  }

  // This is the block mask. We OR all incoming edges, and with zero.
  Value *Zero = ConstantInt::get(IntegerType::getInt1Ty(BB->getContext()), 0);
  VectorParts BlockMask = getVectorValue(Zero);

  // For each pred:
  for (pred_iterator it = pred_begin(BB), e = pred_end(BB); it != e; ++it) {
    VectorParts EM = createEdgeMask(*it, BB);
    for (unsigned part = 0; part < UF; ++part)
      BlockMask[part] = Builder.CreateOr(BlockMask[part], EM[part]);
  }

  return BlockMask;
}

void InnerLoopVectorizer::widenPHIInstruction(Instruction *PN, unsigned UF,
                                              unsigned VF, PhiVector *PV) {
  PHINode *P = cast<PHINode>(PN);
  // Handle recurrences.
  if (Legal->isReductionVariable(P) || Legal->isFirstOrderRecurrence(P)) {
    VectorParts Entry(UF);
    for (unsigned part = 0; part < UF; ++part) {
      // This is phase one of vectorizing PHIs.
      Type *VecTy =
          (VF == 1) ? PN->getType() : VectorType::get(PN->getType(), VF);
      Entry[part] = PHINode::Create(
          VecTy, 2, "vec.phi", &*LoopVectorBody->getFirstInsertionPt());
    }
    VectorLoopValueMap.initVector(P, Entry);
    PV->push_back(P);
    return;
  }

  setDebugLocFromInst(Builder, P);
  // Check for PHI nodes that are lowered to vector selects.
  if (P->getParent() != OrigLoop->getHeader()) {
    // We know that all PHIs in non-header blocks are converted into
    // selects, so we don't have to worry about the insertion order and we
    // can just use the builder.
    // At this point we generate the predication tree. There may be
    // duplications since this is a simple recursive scan, but future
    // optimizations will clean it up.

    unsigned NumIncoming = P->getNumIncomingValues();

    // Generate a sequence of selects of the form:
    // SELECT(Mask3, In3,
    //      SELECT(Mask2, In2,
    //                   ( ...)))
    VectorParts Entry(UF);
    for (unsigned In = 0; In < NumIncoming; In++) {
      VectorParts Cond =
          createEdgeMask(P->getIncomingBlock(In), P->getParent());
      const VectorParts &In0 = getVectorValue(P->getIncomingValue(In));

      for (unsigned part = 0; part < UF; ++part) {
        // We might have single edge PHIs (blocks) - use an identity
        // 'select' for the first PHI operand.
        if (In == 0)
          Entry[part] = Builder.CreateSelect(Cond[part], In0[part], In0[part]);
        else
          // Select between the current value and the previous incoming edge
          // based on the incoming mask.
          Entry[part] = Builder.CreateSelect(Cond[part], In0[part], Entry[part],
                                             "predphi");
      }
    }
    VectorLoopValueMap.initVector(P, Entry);
    return;
  }

  // This PHINode must be an induction variable.
  // Make sure that we know about it.
  assert(Legal->getInductionVars()->count(P) && "Not an induction variable");

  InductionDescriptor II = Legal->getInductionVars()->lookup(P);
  const DataLayout &DL = OrigLoop->getHeader()->getModule()->getDataLayout();

  // FIXME: The newly created binary instructions should contain nsw/nuw flags,
  // which can be found from the original scalar operations.
  switch (II.getKind()) {
  case InductionDescriptor::IK_NoInduction:
    llvm_unreachable("Unknown induction");
  case InductionDescriptor::IK_IntInduction:
  case InductionDescriptor::IK_FpInduction:
    return widenIntOrFpInduction(P);
  case InductionDescriptor::IK_PtrInduction: {
    // Handle the pointer induction variable case.
    assert(P->getType()->isPointerTy() && "Unexpected type.");
    // This is the normalized GEP that starts counting at zero.
    Value *PtrInd = Induction;
    PtrInd = Builder.CreateSExtOrTrunc(PtrInd, II.getStep()->getType());
    // Determine the number of scalars we need to generate for each unroll
    // iteration. If the instruction is uniform, we only need to generate the
    // first lane. Otherwise, we generate all VF values.
    unsigned Lanes = Cost->isUniformAfterVectorization(P, VF) ? 1 : VF;
    // These are the scalar results. Notice that we don't generate vector GEPs
    // because scalar GEPs result in better code.
    ScalarParts Entry(UF);
    for (unsigned Part = 0; Part < UF; ++Part) {
      Entry[Part].resize(VF);
      for (unsigned Lane = 0; Lane < Lanes; ++Lane) {
        Constant *Idx = ConstantInt::get(PtrInd->getType(), Lane + Part * VF);
        Value *GlobalIdx = Builder.CreateAdd(PtrInd, Idx);
        Value *SclrGep = II.transform(Builder, GlobalIdx, PSE.getSE(), DL);
        SclrGep->setName("next.gep");
        Entry[Part][Lane] = SclrGep;
      }
    }
    VectorLoopValueMap.initScalar(P, Entry);
    return;
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

void InnerLoopVectorizer::vectorizeBlockInLoop(BasicBlock *BB, PhiVector *PV) {
  // For each instruction in the old loop.
  for (Instruction &I : *BB) {

    // If the instruction will become trivially dead when vectorized, we don't
    // need to generate it.
    if (DeadInstructions.count(&I))
      continue;

    // Scalarize instructions that should remain scalar after vectorization.
    if (VF > 1 &&
        !(isa<BranchInst>(&I) || isa<PHINode>(&I) ||
          isa<DbgInfoIntrinsic>(&I)) &&
        shouldScalarizeInstruction(&I)) {
      scalarizeInstruction(&I, Legal->isScalarWithPredication(&I));
      continue;
    }

    switch (I.getOpcode()) {
    case Instruction::Br:
      // Nothing to do for PHIs and BR, since we already took care of the
      // loop control flow instructions.
      continue;
    case Instruction::PHI: {
      // Vectorize PHINodes.
      widenPHIInstruction(&I, UF, VF, PV);
      continue;
    } // End of PHI.

    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::SRem:
    case Instruction::URem:
      // Scalarize with predication if this instruction may divide by zero and
      // block execution is conditional, otherwise fallthrough.
      if (Legal->isScalarWithPredication(&I)) {
        scalarizeInstruction(&I, true);
        continue;
      }
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
      // Just widen binops.
      auto *BinOp = cast<BinaryOperator>(&I);
      setDebugLocFromInst(Builder, BinOp);
      const VectorParts &A = getVectorValue(BinOp->getOperand(0));
      const VectorParts &B = getVectorValue(BinOp->getOperand(1));

      // Use this vector value for all users of the original instruction.
      VectorParts Entry(UF);
      for (unsigned Part = 0; Part < UF; ++Part) {
        Value *V = Builder.CreateBinOp(BinOp->getOpcode(), A[Part], B[Part]);

        if (BinaryOperator *VecOp = dyn_cast<BinaryOperator>(V))
          VecOp->copyIRFlags(BinOp);

        Entry[Part] = V;
      }

      VectorLoopValueMap.initVector(&I, Entry);
      addMetadata(Entry, BinOp);
      break;
    }
    case Instruction::Select: {
      // Widen selects.
      // If the selector is loop invariant we can create a select
      // instruction with a scalar condition. Otherwise, use vector-select.
      auto *SE = PSE.getSE();
      bool InvariantCond =
          SE->isLoopInvariant(PSE.getSCEV(I.getOperand(0)), OrigLoop);
      setDebugLocFromInst(Builder, &I);

      // The condition can be loop invariant  but still defined inside the
      // loop. This means that we can't just use the original 'cond' value.
      // We have to take the 'vectorized' value and pick the first lane.
      // Instcombine will make this a no-op.
      const VectorParts &Cond = getVectorValue(I.getOperand(0));
      const VectorParts &Op0 = getVectorValue(I.getOperand(1));
      const VectorParts &Op1 = getVectorValue(I.getOperand(2));

      auto *ScalarCond = getScalarValue(I.getOperand(0), 0, 0);

      VectorParts Entry(UF);
      for (unsigned Part = 0; Part < UF; ++Part) {
        Entry[Part] = Builder.CreateSelect(
            InvariantCond ? ScalarCond : Cond[Part], Op0[Part], Op1[Part]);
      }

      VectorLoopValueMap.initVector(&I, Entry);
      addMetadata(Entry, &I);
      break;
    }

    case Instruction::ICmp:
    case Instruction::FCmp: {
      // Widen compares. Generate vector compares.
      bool FCmp = (I.getOpcode() == Instruction::FCmp);
      auto *Cmp = dyn_cast<CmpInst>(&I);
      setDebugLocFromInst(Builder, Cmp);
      const VectorParts &A = getVectorValue(Cmp->getOperand(0));
      const VectorParts &B = getVectorValue(Cmp->getOperand(1));
      VectorParts Entry(UF);
      for (unsigned Part = 0; Part < UF; ++Part) {
        Value *C = nullptr;
        if (FCmp) {
          C = Builder.CreateFCmp(Cmp->getPredicate(), A[Part], B[Part]);
          cast<FCmpInst>(C)->copyFastMathFlags(Cmp);
        } else {
          C = Builder.CreateICmp(Cmp->getPredicate(), A[Part], B[Part]);
        }
        Entry[Part] = C;
      }

      VectorLoopValueMap.initVector(&I, Entry);
      addMetadata(Entry, &I);
      break;
    }

    case Instruction::Store:
    case Instruction::Load:
      vectorizeMemoryInstruction(&I);
      break;
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
      auto *CI = dyn_cast<CastInst>(&I);
      setDebugLocFromInst(Builder, CI);

      // Optimize the special case where the source is a constant integer
      // induction variable. Notice that we can only optimize the 'trunc' case
      // because (a) FP conversions lose precision, (b) sext/zext may wrap, and
      // (c) other casts depend on pointer size.
      if (Cost->isOptimizableIVTruncate(CI, VF)) {
        widenIntOrFpInduction(cast<PHINode>(CI->getOperand(0)),
                              cast<TruncInst>(CI));
        break;
      }

      /// Vectorize casts.
      Type *DestTy =
          (VF == 1) ? CI->getType() : VectorType::get(CI->getType(), VF);

      const VectorParts &A = getVectorValue(CI->getOperand(0));
      VectorParts Entry(UF);
      for (unsigned Part = 0; Part < UF; ++Part)
        Entry[Part] = Builder.CreateCast(CI->getOpcode(), A[Part], DestTy);
      VectorLoopValueMap.initVector(&I, Entry);
      addMetadata(Entry, &I);
      break;
    }

    case Instruction::Call: {
      // Ignore dbg intrinsics.
      if (isa<DbgInfoIntrinsic>(I))
        break;
      setDebugLocFromInst(Builder, &I);

      Module *M = BB->getParent()->getParent();
      auto *CI = cast<CallInst>(&I);

      StringRef FnName = CI->getCalledFunction()->getName();
      Function *F = CI->getCalledFunction();
      Type *RetTy = ToVectorTy(CI->getType(), VF);
      SmallVector<Type *, 4> Tys;
      for (Value *ArgOperand : CI->arg_operands())
        Tys.push_back(ToVectorTy(ArgOperand->getType(), VF));

      Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);
      if (ID && (ID == Intrinsic::assume || ID == Intrinsic::lifetime_end ||
                 ID == Intrinsic::lifetime_start)) {
        scalarizeInstruction(&I);
        break;
      }
      // The flag shows whether we use Intrinsic or a usual Call for vectorized
      // version of the instruction.
      // Is it beneficial to perform intrinsic call compared to lib call?
      bool NeedToScalarize;
      unsigned CallCost = getVectorCallCost(CI, VF, *TTI, TLI, NeedToScalarize);
      bool UseVectorIntrinsic =
          ID && getVectorIntrinsicCost(CI, VF, *TTI, TLI) <= CallCost;
      if (!UseVectorIntrinsic && NeedToScalarize) {
        scalarizeInstruction(&I);
        break;
      }

      VectorParts Entry(UF);
      for (unsigned Part = 0; Part < UF; ++Part) {
        SmallVector<Value *, 4> Args;
        for (unsigned i = 0, ie = CI->getNumArgOperands(); i != ie; ++i) {
          Value *Arg = CI->getArgOperand(i);
          // Some intrinsics have a scalar argument - don't replace it with a
          // vector.
          if (!UseVectorIntrinsic || !hasVectorInstrinsicScalarOpd(ID, i)) {
            const VectorParts &VectorArg = getVectorValue(CI->getArgOperand(i));
            Arg = VectorArg[Part];
          }
          Args.push_back(Arg);
        }

        Function *VectorF;
        if (UseVectorIntrinsic) {
          // Use vector version of the intrinsic.
          Type *TysForDecl[] = {CI->getType()};
          if (VF > 1)
            TysForDecl[0] = VectorType::get(CI->getType()->getScalarType(), VF);
          VectorF = Intrinsic::getDeclaration(M, ID, TysForDecl);
        } else {
          // Use vector version of the library call.
          StringRef VFnName = TLI->getVectorizedFunction(FnName, VF);
          assert(!VFnName.empty() && "Vector function name is empty.");
          VectorF = M->getFunction(VFnName);
          if (!VectorF) {
            // Generate a declaration
            FunctionType *FTy = FunctionType::get(RetTy, Tys, false);
            VectorF =
                Function::Create(FTy, Function::ExternalLinkage, VFnName, M);
            VectorF->copyAttributesFrom(F);
          }
        }
        assert(VectorF && "Can't create vector function.");

        SmallVector<OperandBundleDef, 1> OpBundles;
        CI->getOperandBundlesAsDefs(OpBundles);
        CallInst *V = Builder.CreateCall(VectorF, Args, OpBundles);

        if (isa<FPMathOperator>(V))
          V->copyFastMathFlags(CI);

        Entry[Part] = V;
      }

      VectorLoopValueMap.initVector(&I, Entry);
      addMetadata(Entry, &I);
      break;
    }

    default:
      // All other instructions are unsupported. Scalarize them.
      scalarizeInstruction(&I);
      break;
    } // end of switch.
  }   // end of for_each instr.
}

void InnerLoopVectorizer::updateAnalysis() {
  // Forget the original basic block.
  PSE.getSE()->forgetLoop(OrigLoop);

  // Update the dominator tree information.
  assert(DT->properlyDominates(LoopBypassBlocks.front(), LoopExitBlock) &&
         "Entry does not dominate exit.");

  // We don't predicate stores by this point, so the vector body should be a
  // single loop.
  DT->addNewBlock(LoopVectorBody, LoopVectorPreHeader);

  DT->addNewBlock(LoopMiddleBlock, LoopVectorBody);
  DT->addNewBlock(LoopScalarPreHeader, LoopBypassBlocks[0]);
  DT->changeImmediateDominator(LoopScalarBody, LoopScalarPreHeader);
  DT->changeImmediateDominator(LoopExitBlock, LoopBypassBlocks[0]);

  DEBUG(DT->verifyDomTree());
}

/// \brief Check whether it is safe to if-convert this phi node.
///
/// Phi nodes with constant expressions that can trap are not safe to if
/// convert.
static bool canIfConvertPHINodes(BasicBlock *BB) {
  for (Instruction &I : *BB) {
    auto *Phi = dyn_cast<PHINode>(&I);
    if (!Phi)
      return true;
    for (Value *V : Phi->incoming_values())
      if (auto *C = dyn_cast<Constant>(V))
        if (C->canTrap())
          return false;
  }
  return true;
}

bool LoopVectorizationLegality::canVectorizeWithIfConvert() {
  if (!EnableIfConversion) {
    ORE->emit(createMissedAnalysis("IfConversionDisabled")
              << "if-conversion is disabled");
    return false;
  }

  assert(TheLoop->getNumBlocks() > 1 && "Single block loops are vectorizable");

  // A list of pointers that we can safely read and write to.
  SmallPtrSet<Value *, 8> SafePointes;

  // Collect safe addresses.
  for (BasicBlock *BB : TheLoop->blocks()) {
    if (blockNeedsPredication(BB))
      continue;

    for (Instruction &I : *BB)
      if (auto *Ptr = getPointerOperand(&I))
        SafePointes.insert(Ptr);
  }

  // Collect the blocks that need predication.
  BasicBlock *Header = TheLoop->getHeader();
  for (BasicBlock *BB : TheLoop->blocks()) {
    // We don't support switch statements inside loops.
    if (!isa<BranchInst>(BB->getTerminator())) {
      ORE->emit(createMissedAnalysis("LoopContainsSwitch", BB->getTerminator())
                << "loop contains a switch statement");
      return false;
    }

    // We must be able to predicate all blocks that need to be predicated.
    if (blockNeedsPredication(BB)) {
      if (!blockCanBePredicated(BB, SafePointes)) {
        ORE->emit(createMissedAnalysis("NoCFGForSelect", BB->getTerminator())
                  << "control flow cannot be substituted for a select");
        return false;
      }
    } else if (BB != Header && !canIfConvertPHINodes(BB)) {
      ORE->emit(createMissedAnalysis("NoCFGForSelect", BB->getTerminator())
                << "control flow cannot be substituted for a select");
      return false;
    }
  }

  // We can if-convert this loop.
  return true;
}

bool LoopVectorizationLegality::canVectorize() {
  // We must have a loop in canonical form. Loops with indirectbr in them cannot
  // be canonicalized.
  if (!TheLoop->getLoopPreheader()) {
    ORE->emit(createMissedAnalysis("CFGNotUnderstood")
              << "loop control flow is not understood by vectorizer");
    return false;
  }

  // FIXME: The code is currently dead, since the loop gets sent to
  // LoopVectorizationLegality is already an innermost loop.
  //
  // We can only vectorize innermost loops.
  if (!TheLoop->empty()) {
    ORE->emit(createMissedAnalysis("NotInnermostLoop")
              << "loop is not the innermost loop");
    return false;
  }

  // We must have a single backedge.
  if (TheLoop->getNumBackEdges() != 1) {
    ORE->emit(createMissedAnalysis("CFGNotUnderstood")
              << "loop control flow is not understood by vectorizer");
    return false;
  }

  // We must have a single exiting block.
  if (!TheLoop->getExitingBlock()) {
    ORE->emit(createMissedAnalysis("CFGNotUnderstood")
              << "loop control flow is not understood by vectorizer");
    return false;
  }

  // We only handle bottom-tested loops, i.e. loop in which the condition is
  // checked at the end of each iteration. With that we can assume that all
  // instructions in the loop are executed the same number of times.
  if (TheLoop->getExitingBlock() != TheLoop->getLoopLatch()) {
    ORE->emit(createMissedAnalysis("CFGNotUnderstood")
              << "loop control flow is not understood by vectorizer");
    return false;
  }

  // We need to have a loop header.
  DEBUG(dbgs() << "LV: Found a loop: " << TheLoop->getHeader()->getName()
               << '\n');

  // Check if we can if-convert non-single-bb loops.
  unsigned NumBlocks = TheLoop->getNumBlocks();
  if (NumBlocks != 1 && !canVectorizeWithIfConvert()) {
    DEBUG(dbgs() << "LV: Can't if-convert the loop.\n");
    return false;
  }

  // ScalarEvolution needs to be able to find the exit count.
  const SCEV *ExitCount = PSE.getBackedgeTakenCount();
  if (ExitCount == PSE.getSE()->getCouldNotCompute()) {
    ORE->emit(createMissedAnalysis("CantComputeNumberOfIterations")
              << "could not determine number of loop iterations");
    DEBUG(dbgs() << "LV: SCEV could not compute the loop exit count.\n");
    return false;
  }

  // Check if we can vectorize the instructions and CFG in this loop.
  if (!canVectorizeInstrs()) {
    DEBUG(dbgs() << "LV: Can't vectorize the instructions or CFG\n");
    return false;
  }

  // Go over each instruction and look at memory deps.
  if (!canVectorizeMemory()) {
    DEBUG(dbgs() << "LV: Can't vectorize due to memory conflicts\n");
    return false;
  }

  DEBUG(dbgs() << "LV: We can vectorize this loop"
               << (LAI->getRuntimePointerChecking()->Need
                       ? " (with a runtime bound check)"
                       : "")
               << "!\n");

  bool UseInterleaved = TTI->enableInterleavedAccessVectorization();

  // If an override option has been passed in for interleaved accesses, use it.
  if (EnableInterleavedMemAccesses.getNumOccurrences() > 0)
    UseInterleaved = EnableInterleavedMemAccesses;

  // Analyze interleaved memory accesses.
  if (UseInterleaved)
    InterleaveInfo.analyzeInterleaving(*getSymbolicStrides());

  unsigned SCEVThreshold = VectorizeSCEVCheckThreshold;
  if (Hints->getForce() == LoopVectorizeHints::FK_Enabled)
    SCEVThreshold = PragmaVectorizeSCEVCheckThreshold;

  if (PSE.getUnionPredicate().getComplexity() > SCEVThreshold) {
    ORE->emit(createMissedAnalysis("TooManySCEVRunTimeChecks")
              << "Too many SCEV assumptions need to be made and checked "
              << "at runtime");
    DEBUG(dbgs() << "LV: Too many SCEV checks needed.\n");
    return false;
  }

  // Okay! We can vectorize. At this point we don't have any other mem analysis
  // which may limit our maximum vectorization factor, so just return true with
  // no restrictions.
  return true;
}

static Type *convertPointerToIntegerType(const DataLayout &DL, Type *Ty) {
  if (Ty->isPointerTy())
    return DL.getIntPtrType(Ty);

  // It is possible that char's or short's overflow when we ask for the loop's
  // trip count, work around this by changing the type size.
  if (Ty->getScalarSizeInBits() < 32)
    return Type::getInt32Ty(Ty->getContext());

  return Ty;
}

static Type *getWiderType(const DataLayout &DL, Type *Ty0, Type *Ty1) {
  Ty0 = convertPointerToIntegerType(DL, Ty0);
  Ty1 = convertPointerToIntegerType(DL, Ty1);
  if (Ty0->getScalarSizeInBits() > Ty1->getScalarSizeInBits())
    return Ty0;
  return Ty1;
}

/// \brief Check that the instruction has outside loop users and is not an
/// identified reduction variable.
static bool hasOutsideLoopUser(const Loop *TheLoop, Instruction *Inst,
                               SmallPtrSetImpl<Value *> &AllowedExit) {
  // Reduction and Induction instructions are allowed to have exit users. All
  // other instructions must not have external users.
  if (!AllowedExit.count(Inst))
    // Check that all of the users of the loop are inside the BB.
    for (User *U : Inst->users()) {
      Instruction *UI = cast<Instruction>(U);
      // This user may be a reduction exit value.
      if (!TheLoop->contains(UI)) {
        DEBUG(dbgs() << "LV: Found an outside user for : " << *UI << '\n');
        return true;
      }
    }
  return false;
}

void LoopVectorizationLegality::addInductionPhi(
    PHINode *Phi, const InductionDescriptor &ID,
    SmallPtrSetImpl<Value *> &AllowedExit) {
  Inductions[Phi] = ID;
  Type *PhiTy = Phi->getType();
  const DataLayout &DL = Phi->getModule()->getDataLayout();

  // Get the widest type.
  if (!PhiTy->isFloatingPointTy()) {
    if (!WidestIndTy)
      WidestIndTy = convertPointerToIntegerType(DL, PhiTy);
    else
      WidestIndTy = getWiderType(DL, PhiTy, WidestIndTy);
  }

  // Int inductions are special because we only allow one IV.
  if (ID.getKind() == InductionDescriptor::IK_IntInduction &&
      ID.getConstIntStepValue() &&
      ID.getConstIntStepValue()->isOne() &&
      isa<Constant>(ID.getStartValue()) &&
      cast<Constant>(ID.getStartValue())->isNullValue()) {

    // Use the phi node with the widest type as induction. Use the last
    // one if there are multiple (no good reason for doing this other
    // than it is expedient). We've checked that it begins at zero and
    // steps by one, so this is a canonical induction variable.
    if (!PrimaryInduction || PhiTy == WidestIndTy)
      PrimaryInduction = Phi;
  }

  // Both the PHI node itself, and the "post-increment" value feeding
  // back into the PHI node may have external users.
  AllowedExit.insert(Phi);
  AllowedExit.insert(Phi->getIncomingValueForBlock(TheLoop->getLoopLatch()));

  DEBUG(dbgs() << "LV: Found an induction variable.\n");
  return;
}

bool LoopVectorizationLegality::canVectorizeInstrs() {
  BasicBlock *Header = TheLoop->getHeader();

  // Look for the attribute signaling the absence of NaNs.
  Function &F = *Header->getParent();
  HasFunNoNaNAttr =
      F.getFnAttribute("no-nans-fp-math").getValueAsString() == "true";

  // For each block in the loop.
  for (BasicBlock *BB : TheLoop->blocks()) {
    // Scan the instructions in the block and look for hazards.
    for (Instruction &I : *BB) {
      if (auto *Phi = dyn_cast<PHINode>(&I)) {
        Type *PhiTy = Phi->getType();
        // Check that this PHI type is allowed.
        if (!PhiTy->isIntegerTy() && !PhiTy->isFloatingPointTy() &&
            !PhiTy->isPointerTy()) {
          ORE->emit(createMissedAnalysis("CFGNotUnderstood", Phi)
                    << "loop control flow is not understood by vectorizer");
          DEBUG(dbgs() << "LV: Found an non-int non-pointer PHI.\n");
          return false;
        }

        // If this PHINode is not in the header block, then we know that we
        // can convert it to select during if-conversion. No need to check if
        // the PHIs in this block are induction or reduction variables.
        if (BB != Header) {
          // Check that this instruction has no outside users or is an
          // identified reduction value with an outside user.
          if (!hasOutsideLoopUser(TheLoop, Phi, AllowedExit))
            continue;
          ORE->emit(createMissedAnalysis("NeitherInductionNorReduction", Phi)
                    << "value could not be identified as "
                       "an induction or reduction variable");
          return false;
        }

        // We only allow if-converted PHIs with exactly two incoming values.
        if (Phi->getNumIncomingValues() != 2) {
          ORE->emit(createMissedAnalysis("CFGNotUnderstood", Phi)
                    << "control flow not understood by vectorizer");
          DEBUG(dbgs() << "LV: Found an invalid PHI.\n");
          return false;
        }

        RecurrenceDescriptor RedDes;
        if (RecurrenceDescriptor::isReductionPHI(Phi, TheLoop, RedDes)) {
          if (RedDes.hasUnsafeAlgebra())
            Requirements->addUnsafeAlgebraInst(RedDes.getUnsafeAlgebraInst());
          AllowedExit.insert(RedDes.getLoopExitInstr());
          Reductions[Phi] = RedDes;
          continue;
        }

        InductionDescriptor ID;
        if (InductionDescriptor::isInductionPHI(Phi, TheLoop, PSE, ID)) {
          addInductionPhi(Phi, ID, AllowedExit);
          if (ID.hasUnsafeAlgebra() && !HasFunNoNaNAttr)
            Requirements->addUnsafeAlgebraInst(ID.getUnsafeAlgebraInst());
          continue;
        }

        if (RecurrenceDescriptor::isFirstOrderRecurrence(Phi, TheLoop, DT)) {
          FirstOrderRecurrences.insert(Phi);
          continue;
        }

        // As a last resort, coerce the PHI to a AddRec expression
        // and re-try classifying it a an induction PHI.
        if (InductionDescriptor::isInductionPHI(Phi, TheLoop, PSE, ID, true)) {
          addInductionPhi(Phi, ID, AllowedExit);
          continue;
        }

        ORE->emit(createMissedAnalysis("NonReductionValueUsedOutsideLoop", Phi)
                  << "value that could not be identified as "
                     "reduction is used outside the loop");
        DEBUG(dbgs() << "LV: Found an unidentified PHI." << *Phi << "\n");
        return false;
      } // end of PHI handling

      // We handle calls that:
      //   * Are debug info intrinsics.
      //   * Have a mapping to an IR intrinsic.
      //   * Have a vector version available.
      auto *CI = dyn_cast<CallInst>(&I);
      if (CI && !getVectorIntrinsicIDForCall(CI, TLI) &&
          !isa<DbgInfoIntrinsic>(CI) &&
          !(CI->getCalledFunction() && TLI &&
            TLI->isFunctionVectorizable(CI->getCalledFunction()->getName()))) {
        ORE->emit(createMissedAnalysis("CantVectorizeCall", CI)
                  << "call instruction cannot be vectorized");
        DEBUG(dbgs() << "LV: Found a non-intrinsic, non-libfunc callsite.\n");
        return false;
      }

      // Intrinsics such as powi,cttz and ctlz are legal to vectorize if the
      // second argument is the same (i.e. loop invariant)
      if (CI && hasVectorInstrinsicScalarOpd(
                    getVectorIntrinsicIDForCall(CI, TLI), 1)) {
        auto *SE = PSE.getSE();
        if (!SE->isLoopInvariant(PSE.getSCEV(CI->getOperand(1)), TheLoop)) {
          ORE->emit(createMissedAnalysis("CantVectorizeIntrinsic", CI)
                    << "intrinsic instruction cannot be vectorized");
          DEBUG(dbgs() << "LV: Found unvectorizable intrinsic " << *CI << "\n");
          return false;
        }
      }

      // Check that the instruction return type is vectorizable.
      // Also, we can't vectorize extractelement instructions.
      if ((!VectorType::isValidElementType(I.getType()) &&
           !I.getType()->isVoidTy()) ||
          isa<ExtractElementInst>(I)) {
        ORE->emit(createMissedAnalysis("CantVectorizeInstructionReturnType", &I)
                  << "instruction return type cannot be vectorized");
        DEBUG(dbgs() << "LV: Found unvectorizable type.\n");
        return false;
      }

      // Check that the stored type is vectorizable.
      if (auto *ST = dyn_cast<StoreInst>(&I)) {
        Type *T = ST->getValueOperand()->getType();
        if (!VectorType::isValidElementType(T)) {
          ORE->emit(createMissedAnalysis("CantVectorizeStore", ST)
                    << "store instruction cannot be vectorized");
          return false;
        }

        // FP instructions can allow unsafe algebra, thus vectorizable by
        // non-IEEE-754 compliant SIMD units.
        // This applies to floating-point math operations and calls, not memory
        // operations, shuffles, or casts, as they don't change precision or
        // semantics.
      } else if (I.getType()->isFloatingPointTy() && (CI || I.isBinaryOp()) &&
                 !I.hasUnsafeAlgebra()) {
        DEBUG(dbgs() << "LV: Found FP op with unsafe algebra.\n");
        Hints->setPotentiallyUnsafe();
      }

      // Reduction instructions are allowed to have exit users.
      // All other instructions must not have external users.
      if (hasOutsideLoopUser(TheLoop, &I, AllowedExit)) {
        ORE->emit(createMissedAnalysis("ValueUsedOutsideLoop", &I)
                  << "value cannot be used outside the loop");
        return false;
      }

    } // next instr.
  }

  if (!PrimaryInduction) {
    DEBUG(dbgs() << "LV: Did not find one integer induction var.\n");
    if (Inductions.empty()) {
      ORE->emit(createMissedAnalysis("NoInductionVariable")
                << "loop induction variable could not be identified");
      return false;
    }
  }

  // Now we know the widest induction type, check if our found induction
  // is the same size. If it's not, unset it here and InnerLoopVectorizer
  // will create another.
  if (PrimaryInduction && WidestIndTy != PrimaryInduction->getType())
    PrimaryInduction = nullptr;

  return true;
}

void LoopVectorizationCostModel::collectLoopScalars(unsigned VF) {

  // We should not collect Scalars more than once per VF. Right now,
  // this function is called from collectUniformsAndScalars(), which 
  // already does this check. Collecting Scalars for VF=1 does not make any
  // sense.

  assert(VF >= 2 && !Scalars.count(VF) &&
         "This function should not be visited twice for the same VF");

  // If an instruction is uniform after vectorization, it will remain scalar.
  Scalars[VF].insert(Uniforms[VF].begin(), Uniforms[VF].end());

  // Collect the getelementptr instructions that will not be vectorized. A
  // getelementptr instruction is only vectorized if it is used for a legal
  // gather or scatter operation.
  for (auto *BB : TheLoop->blocks())
    for (auto &I : *BB) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
        Scalars[VF].insert(GEP);
        continue;
      }
      auto *Ptr = getPointerOperand(&I);
      if (!Ptr)
        continue;
      auto *GEP = getGEPInstruction(Ptr);
      if (GEP && getWideningDecision(&I, VF) == CM_GatherScatter)
        Scalars[VF].erase(GEP);
    }

  // An induction variable will remain scalar if all users of the induction
  // variable and induction variable update remain scalar.
  auto *Latch = TheLoop->getLoopLatch();
  for (auto &Induction : *Legal->getInductionVars()) {
    auto *Ind = Induction.first;
    auto *IndUpdate = cast<Instruction>(Ind->getIncomingValueForBlock(Latch));

    // Determine if all users of the induction variable are scalar after
    // vectorization.
    auto ScalarInd = all_of(Ind->users(), [&](User *U) -> bool {
      auto *I = cast<Instruction>(U);
      return I == IndUpdate || !TheLoop->contains(I) || Scalars[VF].count(I);
    });
    if (!ScalarInd)
      continue;

    // Determine if all users of the induction variable update instruction are
    // scalar after vectorization.
    auto ScalarIndUpdate = all_of(IndUpdate->users(), [&](User *U) -> bool {
      auto *I = cast<Instruction>(U);
      return I == Ind || !TheLoop->contains(I) || Scalars[VF].count(I);
    });
    if (!ScalarIndUpdate)
      continue;

    // The induction variable and its update instruction will remain scalar.
    Scalars[VF].insert(Ind);
    Scalars[VF].insert(IndUpdate);
  }
}

bool LoopVectorizationLegality::isScalarWithPredication(Instruction *I) {
  if (!blockNeedsPredication(I->getParent()))
    return false;
  switch(I->getOpcode()) {
  default:
    break;
  case Instruction::Store:
    return !isMaskRequired(I);
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::URem:
    return mayDivideByZero(*I);
  }
  return false;
}

bool LoopVectorizationLegality::memoryInstructionCanBeWidened(Instruction *I,
                                                              unsigned VF) {
  // Get and ensure we have a valid memory instruction.
  LoadInst *LI = dyn_cast<LoadInst>(I);
  StoreInst *SI = dyn_cast<StoreInst>(I);
  assert((LI || SI) && "Invalid memory instruction");

  auto *Ptr = getPointerOperand(I);

  // In order to be widened, the pointer should be consecutive, first of all.
  if (!isConsecutivePtr(Ptr))
    return false;

  // If the instruction is a store located in a predicated block, it will be
  // scalarized.
  if (isScalarWithPredication(I))
    return false;

  // If the instruction's allocated size doesn't equal it's type size, it
  // requires padding and will be scalarized.
  auto &DL = I->getModule()->getDataLayout();
  auto *ScalarTy = LI ? LI->getType() : SI->getValueOperand()->getType();
  if (hasIrregularType(ScalarTy, DL, VF))
    return false;

  return true;
}

void LoopVectorizationCostModel::collectLoopUniforms(unsigned VF) {

  // We should not collect Uniforms more than once per VF. Right now,
  // this function is called from collectUniformsAndScalars(), which 
  // already does this check. Collecting Uniforms for VF=1 does not make any
  // sense.

  assert(VF >= 2 && !Uniforms.count(VF) &&
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

  // Start with the conditional branch. If the branch condition is an
  // instruction contained in the loop that is only used by the branch, it is
  // uniform.
  auto *Cmp = dyn_cast<Instruction>(Latch->getTerminator()->getOperand(0));
  if (Cmp && TheLoop->contains(Cmp) && Cmp->hasOneUse()) {
    Worklist.insert(Cmp);
    DEBUG(dbgs() << "LV: Found uniform instruction: " << *Cmp << "\n");
  }

  // Holds consecutive and consecutive-like pointers. Consecutive-like pointers
  // are pointers that are treated like consecutive pointers during
  // vectorization. The pointer operands of interleaved accesses are an
  // example.
  SmallSetVector<Instruction *, 8> ConsecutiveLikePtrs;

  // Holds pointer operands of instructions that are possibly non-uniform.
  SmallPtrSet<Instruction *, 8> PossibleNonUniformPtrs;

  auto isUniformDecision = [&](Instruction *I, unsigned VF) {
    InstWidening WideningDecision = getWideningDecision(I, VF);
    assert(WideningDecision != CM_Unknown &&
           "Widening decision should be ready at this moment");

    return (WideningDecision == CM_Widen ||
            WideningDecision == CM_Interleave);
  };
  // Iterate over the instructions in the loop, and collect all
  // consecutive-like pointer operands in ConsecutiveLikePtrs. If it's possible
  // that a consecutive-like pointer operand will be scalarized, we collect it
  // in PossibleNonUniformPtrs instead. We use two sets here because a single
  // getelementptr instruction can be used by both vectorized and scalarized
  // memory instructions. For example, if a loop loads and stores from the same
  // location, but the store is conditional, the store will be scalarized, and
  // the getelementptr won't remain uniform.
  for (auto *BB : TheLoop->blocks())
    for (auto &I : *BB) {

      // If there's no pointer operand, there's nothing to do.
      auto *Ptr = dyn_cast_or_null<Instruction>(getPointerOperand(&I));
      if (!Ptr)
        continue;

      // True if all users of Ptr are memory accesses that have Ptr as their
      // pointer operand.
      auto UsersAreMemAccesses = all_of(Ptr->users(), [&](User *U) -> bool {
        return getPointerOperand(U) == Ptr;
      });

      // Ensure the memory instruction will not be scalarized or used by
      // gather/scatter, making its pointer operand non-uniform. If the pointer
      // operand is used by any instruction other than a memory access, we
      // conservatively assume the pointer operand may be non-uniform.
      if (!UsersAreMemAccesses || !isUniformDecision(&I, VF))
        PossibleNonUniformPtrs.insert(Ptr);

      // If the memory instruction will be vectorized and its pointer operand
      // is consecutive-like, or interleaving - the pointer operand should
      // remain uniform.
      else
        ConsecutiveLikePtrs.insert(Ptr);
    }

  // Add to the Worklist all consecutive and consecutive-like pointers that
  // aren't also identified as possibly non-uniform.
  for (auto *V : ConsecutiveLikePtrs)
    if (!PossibleNonUniformPtrs.count(V)) {
      DEBUG(dbgs() << "LV: Found uniform instruction: " << *V << "\n");
      Worklist.insert(V);
    }

  // Expand Worklist in topological order: whenever a new instruction
  // is added , its users should be either already inside Worklist, or
  // out of scope. It ensures a uniform instruction will only be used
  // by uniform instructions or out of scope instructions.
  unsigned idx = 0;
  while (idx != Worklist.size()) {
    Instruction *I = Worklist[idx++];

    for (auto OV : I->operand_values()) {
      if (isOutOfScope(OV))
        continue;
      auto *OI = cast<Instruction>(OV);
      if (all_of(OI->users(), [&](User *U) -> bool {
            auto *J = cast<Instruction>(U);
            return !TheLoop->contains(J) || Worklist.count(J) ||
                   (OI == getPointerOperand(J) && isUniformDecision(J, VF));
          })) {
        Worklist.insert(OI);
        DEBUG(dbgs() << "LV: Found uniform instruction: " << *OI << "\n");
      }
    }
  }

  // Returns true if Ptr is the pointer operand of a memory access instruction
  // I, and I is known to not require scalarization.
  auto isVectorizedMemAccessUse = [&](Instruction *I, Value *Ptr) -> bool {
    return getPointerOperand(I) == Ptr && isUniformDecision(I, VF);
  };

  // For an instruction to be added into Worklist above, all its users inside
  // the loop should also be in Worklist. However, this condition cannot be
  // true for phi nodes that form a cyclic dependence. We must process phi
  // nodes separately. An induction variable will remain uniform if all users
  // of the induction variable and induction variable update remain uniform.
  // The code below handles both pointer and non-pointer induction variables.
  for (auto &Induction : *Legal->getInductionVars()) {
    auto *Ind = Induction.first;
    auto *IndUpdate = cast<Instruction>(Ind->getIncomingValueForBlock(Latch));

    // Determine if all users of the induction variable are uniform after
    // vectorization.
    auto UniformInd = all_of(Ind->users(), [&](User *U) -> bool {
      auto *I = cast<Instruction>(U);
      return I == IndUpdate || !TheLoop->contains(I) || Worklist.count(I) ||
             isVectorizedMemAccessUse(I, Ind);
    });
    if (!UniformInd)
      continue;

    // Determine if all users of the induction variable update instruction are
    // uniform after vectorization.
    auto UniformIndUpdate = all_of(IndUpdate->users(), [&](User *U) -> bool {
      auto *I = cast<Instruction>(U);
      return I == Ind || !TheLoop->contains(I) || Worklist.count(I) ||
             isVectorizedMemAccessUse(I, IndUpdate);
    });
    if (!UniformIndUpdate)
      continue;

    // The induction variable and its update instruction will remain uniform.
    Worklist.insert(Ind);
    Worklist.insert(IndUpdate);
    DEBUG(dbgs() << "LV: Found uniform instruction: " << *Ind << "\n");
    DEBUG(dbgs() << "LV: Found uniform instruction: " << *IndUpdate << "\n");
  }

  Uniforms[VF].insert(Worklist.begin(), Worklist.end());
}

bool LoopVectorizationLegality::canVectorizeMemory() {
  LAI = &(*GetLAA)(*TheLoop);
  InterleaveInfo.setLAI(LAI);
  const OptimizationRemarkAnalysis *LAR = LAI->getReport();
  if (LAR) {
    OptimizationRemarkAnalysis VR(Hints->vectorizeAnalysisPassName(),
                                  "loop not vectorized: ", *LAR);
    ORE->emit(VR);
  }
  if (!LAI->canVectorizeMemory())
    return false;

  if (LAI->hasStoreToLoopInvariantAddress()) {
    ORE->emit(createMissedAnalysis("CantVectorizeStoreToLoopInvariantAddress")
              << "write to a loop invariant address could not be vectorized");
    DEBUG(dbgs() << "LV: We don't allow storing to uniform addresses\n");
    return false;
  }

  Requirements->addRuntimePointerChecks(LAI->getNumRuntimePointerChecks());
  PSE.addPredicate(LAI->getPSE().getUnionPredicate());

  return true;
}

bool LoopVectorizationLegality::isInductionVariable(const Value *V) {
  Value *In0 = const_cast<Value *>(V);
  PHINode *PN = dyn_cast_or_null<PHINode>(In0);
  if (!PN)
    return false;

  return Inductions.count(PN);
}

bool LoopVectorizationLegality::isFirstOrderRecurrence(const PHINode *Phi) {
  return FirstOrderRecurrences.count(Phi);
}

bool LoopVectorizationLegality::blockNeedsPredication(BasicBlock *BB) {
  return LoopAccessInfo::blockNeedsPredication(BB, TheLoop, DT);
}

bool LoopVectorizationLegality::blockCanBePredicated(
    BasicBlock *BB, SmallPtrSetImpl<Value *> &SafePtrs) {
  const bool IsAnnotatedParallel = TheLoop->isAnnotatedParallel();

  for (Instruction &I : *BB) {
    // Check that we don't have a constant expression that can trap as operand.
    for (Value *Operand : I.operands()) {
      if (auto *C = dyn_cast<Constant>(Operand))
        if (C->canTrap())
          return false;
    }
    // We might be able to hoist the load.
    if (I.mayReadFromMemory()) {
      auto *LI = dyn_cast<LoadInst>(&I);
      if (!LI)
        return false;
      if (!SafePtrs.count(LI->getPointerOperand())) {
        if (isLegalMaskedLoad(LI->getType(), LI->getPointerOperand()) ||
            isLegalMaskedGather(LI->getType())) {
          MaskedOp.insert(LI);
          continue;
        }
        // !llvm.mem.parallel_loop_access implies if-conversion safety.
        if (IsAnnotatedParallel)
          continue;
        return false;
      }
    }

    if (I.mayWriteToMemory()) {
      auto *SI = dyn_cast<StoreInst>(&I);
      // We only support predication of stores in basic blocks with one
      // predecessor.
      if (!SI)
        return false;

      // Build a masked store if it is legal for the target.
      if (isLegalMaskedStore(SI->getValueOperand()->getType(),
                             SI->getPointerOperand()) ||
          isLegalMaskedScatter(SI->getValueOperand()->getType())) {
        MaskedOp.insert(SI);
        continue;
      }

      bool isSafePtr = (SafePtrs.count(SI->getPointerOperand()) != 0);
      bool isSinglePredecessor = SI->getParent()->getSinglePredecessor();

      if (++NumPredStores > NumberOfStoresToPredicate || !isSafePtr ||
          !isSinglePredecessor)
        return false;
    }
    if (I.mayThrow())
      return false;
  }

  return true;
}

void InterleavedAccessInfo::collectConstStrideAccesses(
    MapVector<Instruction *, StrideDescriptor> &AccessStrideInfo,
    const ValueToValueMap &Strides) {

  auto &DL = TheLoop->getHeader()->getModule()->getDataLayout();

  // Since it's desired that the load/store instructions be maintained in
  // "program order" for the interleaved access analysis, we have to visit the
  // blocks in the loop in reverse postorder (i.e., in a topological order).
  // Such an ordering will ensure that any load/store that may be executed
  // before a second load/store will precede the second load/store in
  // AccessStrideInfo.
  LoopBlocksDFS DFS(TheLoop);
  DFS.perform(LI);
  for (BasicBlock *BB : make_range(DFS.beginRPO(), DFS.endRPO()))
    for (auto &I : *BB) {
      auto *LI = dyn_cast<LoadInst>(&I);
      auto *SI = dyn_cast<StoreInst>(&I);
      if (!LI && !SI)
        continue;

      Value *Ptr = getPointerOperand(&I);
      // We don't check wrapping here because we don't know yet if Ptr will be 
      // part of a full group or a group with gaps. Checking wrapping for all 
      // pointers (even those that end up in groups with no gaps) will be overly
      // conservative. For full groups, wrapping should be ok since if we would 
      // wrap around the address space we would do a memory access at nullptr
      // even without the transformation. The wrapping checks are therefore
      // deferred until after we've formed the interleaved groups.
      int64_t Stride = getPtrStride(PSE, Ptr, TheLoop, Strides,
                                    /*Assume=*/true, /*ShouldCheckWrap=*/false);

      const SCEV *Scev = replaceSymbolicStrideSCEV(PSE, Strides, Ptr);
      PointerType *PtrTy = dyn_cast<PointerType>(Ptr->getType());
      uint64_t Size = DL.getTypeAllocSize(PtrTy->getElementType());

      // An alignment of 0 means target ABI alignment.
      unsigned Align = getMemInstAlignment(&I);
      if (!Align)
        Align = DL.getABITypeAlignment(PtrTy->getElementType());

      AccessStrideInfo[&I] = StrideDescriptor(Stride, Scev, Size, Align);
    }
}

// Analyze interleaved accesses and collect them into interleaved load and
// store groups.
//
// When generating code for an interleaved load group, we effectively hoist all
// loads in the group to the location of the first load in program order. When
// generating code for an interleaved store group, we sink all stores to the
// location of the last store. This code motion can change the order of load
// and store instructions and may break dependences.
//
// The code generation strategy mentioned above ensures that we won't violate
// any write-after-read (WAR) dependences.
//
// E.g., for the WAR dependence:  a = A[i];      // (1)
//                                A[i] = b;      // (2)
//
// The store group of (2) is always inserted at or below (2), and the load
// group of (1) is always inserted at or above (1). Thus, the instructions will
// never be reordered. All other dependences are checked to ensure the
// correctness of the instruction reordering.
//
// The algorithm visits all memory accesses in the loop in bottom-up program
// order. Program order is established by traversing the blocks in the loop in
// reverse postorder when collecting the accesses.
//
// We visit the memory accesses in bottom-up order because it can simplify the
// construction of store groups in the presence of write-after-write (WAW)
// dependences.
//
// E.g., for the WAW dependence:  A[i] = a;      // (1)
//                                A[i] = b;      // (2)
//                                A[i + 1] = c;  // (3)
//
// We will first create a store group with (3) and (2). (1) can't be added to
// this group because it and (2) are dependent. However, (1) can be grouped
// with other accesses that may precede it in program order. Note that a
// bottom-up order does not imply that WAW dependences should not be checked.
void InterleavedAccessInfo::analyzeInterleaving(
    const ValueToValueMap &Strides) {
  DEBUG(dbgs() << "LV: Analyzing interleaved accesses...\n");

  // Holds all accesses with a constant stride.
  MapVector<Instruction *, StrideDescriptor> AccessStrideInfo;
  collectConstStrideAccesses(AccessStrideInfo, Strides);

  if (AccessStrideInfo.empty())
    return;

  // Collect the dependences in the loop.
  collectDependences();

  // Holds all interleaved store groups temporarily.
  SmallSetVector<InterleaveGroup *, 4> StoreGroups;
  // Holds all interleaved load groups temporarily.
  SmallSetVector<InterleaveGroup *, 4> LoadGroups;

  // Search in bottom-up program order for pairs of accesses (A and B) that can
  // form interleaved load or store groups. In the algorithm below, access A
  // precedes access B in program order. We initialize a group for B in the
  // outer loop of the algorithm, and then in the inner loop, we attempt to
  // insert each A into B's group if:
  //
  //  1. A and B have the same stride,
  //  2. A and B have the same memory object size, and
  //  3. A belongs in B's group according to its distance from B.
  //
  // Special care is taken to ensure group formation will not break any
  // dependences.
  for (auto BI = AccessStrideInfo.rbegin(), E = AccessStrideInfo.rend();
       BI != E; ++BI) {
    Instruction *B = BI->first;
    StrideDescriptor DesB = BI->second;

    // Initialize a group for B if it has an allowable stride. Even if we don't
    // create a group for B, we continue with the bottom-up algorithm to ensure
    // we don't break any of B's dependences.
    InterleaveGroup *Group = nullptr;
    if (isStrided(DesB.Stride)) {
      Group = getInterleaveGroup(B);
      if (!Group) {
        DEBUG(dbgs() << "LV: Creating an interleave group with:" << *B << '\n');
        Group = createInterleaveGroup(B, DesB.Stride, DesB.Align);
      }
      if (B->mayWriteToMemory())
        StoreGroups.insert(Group);
      else
        LoadGroups.insert(Group);
    }

    for (auto AI = std::next(BI); AI != E; ++AI) {
      Instruction *A = AI->first;
      StrideDescriptor DesA = AI->second;

      // Our code motion strategy implies that we can't have dependences
      // between accesses in an interleaved group and other accesses located
      // between the first and last member of the group. Note that this also
      // means that a group can't have more than one member at a given offset.
      // The accesses in a group can have dependences with other accesses, but
      // we must ensure we don't extend the boundaries of the group such that
      // we encompass those dependent accesses.
      //
      // For example, assume we have the sequence of accesses shown below in a
      // stride-2 loop:
      //
      //  (1, 2) is a group | A[i]   = a;  // (1)
      //                    | A[i-1] = b;  // (2) |
      //                      A[i-3] = c;  // (3)
      //                      A[i]   = d;  // (4) | (2, 4) is not a group
      //
      // Because accesses (2) and (3) are dependent, we can group (2) with (1)
      // but not with (4). If we did, the dependent access (3) would be within
      // the boundaries of the (2, 4) group.
      if (!canReorderMemAccessesForInterleavedGroups(&*AI, &*BI)) {

        // If a dependence exists and A is already in a group, we know that A
        // must be a store since A precedes B and WAR dependences are allowed.
        // Thus, A would be sunk below B. We release A's group to prevent this
        // illegal code motion. A will then be free to form another group with
        // instructions that precede it.
        if (isInterleaved(A)) {
          InterleaveGroup *StoreGroup = getInterleaveGroup(A);
          StoreGroups.remove(StoreGroup);
          releaseGroup(StoreGroup);
        }

        // If a dependence exists and A is not already in a group (or it was
        // and we just released it), B might be hoisted above A (if B is a
        // load) or another store might be sunk below A (if B is a store). In
        // either case, we can't add additional instructions to B's group. B
        // will only form a group with instructions that it precedes.
        break;
      }

      // At this point, we've checked for illegal code motion. If either A or B
      // isn't strided, there's nothing left to do.
      if (!isStrided(DesA.Stride) || !isStrided(DesB.Stride))
        continue;

      // Ignore A if it's already in a group or isn't the same kind of memory
      // operation as B.
      if (isInterleaved(A) || A->mayReadFromMemory() != B->mayReadFromMemory())
        continue;

      // Check rules 1 and 2. Ignore A if its stride or size is different from
      // that of B.
      if (DesA.Stride != DesB.Stride || DesA.Size != DesB.Size)
        continue;

      // Ignore A if the memory object of A and B don't belong to the same
      // address space
      if (getMemInstAddressSpace(A) != getMemInstAddressSpace(B))
        continue;

      // Calculate the distance from A to B.
      const SCEVConstant *DistToB = dyn_cast<SCEVConstant>(
          PSE.getSE()->getMinusSCEV(DesA.Scev, DesB.Scev));
      if (!DistToB)
        continue;
      int64_t DistanceToB = DistToB->getAPInt().getSExtValue();

      // Check rule 3. Ignore A if its distance to B is not a multiple of the
      // size.
      if (DistanceToB % static_cast<int64_t>(DesB.Size))
        continue;

      // Ignore A if either A or B is in a predicated block. Although we
      // currently prevent group formation for predicated accesses, we may be
      // able to relax this limitation in the future once we handle more
      // complicated blocks.
      if (isPredicated(A->getParent()) || isPredicated(B->getParent()))
        continue;

      // The index of A is the index of B plus A's distance to B in multiples
      // of the size.
      int IndexA =
          Group->getIndex(B) + DistanceToB / static_cast<int64_t>(DesB.Size);

      // Try to insert A into B's group.
      if (Group->insertMember(A, IndexA, DesA.Align)) {
        DEBUG(dbgs() << "LV: Inserted:" << *A << '\n'
                     << "    into the interleave group with" << *B << '\n');
        InterleaveGroupMap[A] = Group;

        // Set the first load in program order as the insert position.
        if (A->mayReadFromMemory())
          Group->setInsertPos(A);
      }
    } // Iteration over A accesses.
  } // Iteration over B accesses.

  // Remove interleaved store groups with gaps.
  for (InterleaveGroup *Group : StoreGroups)
    if (Group->getNumMembers() != Group->getFactor())
      releaseGroup(Group);

  // Remove interleaved groups with gaps (currently only loads) whose memory
  // accesses may wrap around. We have to revisit the getPtrStride analysis,
  // this time with ShouldCheckWrap=true, since collectConstStrideAccesses does
  // not check wrapping (see documentation there).
  // FORNOW we use Assume=false;
  // TODO: Change to Assume=true but making sure we don't exceed the threshold
  // of runtime SCEV assumptions checks (thereby potentially failing to
  // vectorize altogether).
  // Additional optional optimizations:
  // TODO: If we are peeling the loop and we know that the first pointer doesn't
  // wrap then we can deduce that all pointers in the group don't wrap.
  // This means that we can forcefully peel the loop in order to only have to
  // check the first pointer for no-wrap. When we'll change to use Assume=true
  // we'll only need at most one runtime check per interleaved group.
  //
  for (InterleaveGroup *Group : LoadGroups) {

    // Case 1: A full group. Can Skip the checks; For full groups, if the wide
    // load would wrap around the address space we would do a memory access at
    // nullptr even without the transformation.
    if (Group->getNumMembers() == Group->getFactor())
      continue;

    // Case 2: If first and last members of the group don't wrap this implies
    // that all the pointers in the group don't wrap.
    // So we check only group member 0 (which is always guaranteed to exist),
    // and group member Factor - 1; If the latter doesn't exist we rely on
    // peeling (if it is a non-reveresed accsess -- see Case 3).
    Value *FirstMemberPtr = getPointerOperand(Group->getMember(0));
    if (!getPtrStride(PSE, FirstMemberPtr, TheLoop, Strides, /*Assume=*/false,
                      /*ShouldCheckWrap=*/true)) {
      DEBUG(dbgs() << "LV: Invalidate candidate interleaved group due to "
                      "first group member potentially pointer-wrapping.\n");
      releaseGroup(Group);
      continue;
    }
    Instruction *LastMember = Group->getMember(Group->getFactor() - 1);
    if (LastMember) {
      Value *LastMemberPtr = getPointerOperand(LastMember);
      if (!getPtrStride(PSE, LastMemberPtr, TheLoop, Strides, /*Assume=*/false, 
                        /*ShouldCheckWrap=*/true)) {
        DEBUG(dbgs() << "LV: Invalidate candidate interleaved group due to "
                        "last group member potentially pointer-wrapping.\n");
        releaseGroup(Group);
      }
    } else {
      // Case 3: A non-reversed interleaved load group with gaps: We need
      // to execute at least one scalar epilogue iteration. This will ensure 
      // we don't speculatively access memory out-of-bounds. We only need
      // to look for a member at index factor - 1, since every group must have 
      // a member at index zero.
      if (Group->isReverse()) {
        releaseGroup(Group);
        continue;
      }
      DEBUG(dbgs() << "LV: Interleaved group requires epilogue iteration.\n");
      RequiresScalarEpilogue = true;
    }
  }
}

LoopVectorizationCostModel::VectorizationFactor
LoopVectorizationCostModel::selectVectorizationFactor(bool OptForSize) {
  // Width 1 means no vectorize
  VectorizationFactor Factor = {1U, 0U};
  if (OptForSize && Legal->getRuntimePointerChecking()->Need) {
    ORE->emit(createMissedAnalysis("CantVersionLoopWithOptForSize")
              << "runtime pointer checks needed. Enable vectorization of this "
                 "loop with '#pragma clang loop vectorize(enable)' when "
                 "compiling with -Os/-Oz");
    DEBUG(dbgs()
          << "LV: Aborting. Runtime ptr check is required with -Os/-Oz.\n");
    return Factor;
  }

  if (!EnableCondStoresVectorization && Legal->getNumPredStores()) {
    ORE->emit(createMissedAnalysis("ConditionalStore")
              << "store that is conditionally executed prevents vectorization");
    DEBUG(dbgs() << "LV: No vectorization. There are conditional stores.\n");
    return Factor;
  }

  MinBWs = computeMinimumValueSizes(TheLoop->getBlocks(), *DB, &TTI);
  unsigned SmallestType, WidestType;
  std::tie(SmallestType, WidestType) = getSmallestAndWidestTypes();
  unsigned WidestRegister = TTI.getRegisterBitWidth(true);
  unsigned MaxSafeDepDist = -1U;

  // Get the maximum safe dependence distance in bits computed by LAA. If the
  // loop contains any interleaved accesses, we divide the dependence distance
  // by the maximum interleave factor of all interleaved groups. Note that
  // although the division ensures correctness, this is a fairly conservative
  // computation because the maximum distance computed by LAA may not involve
  // any of the interleaved accesses.
  if (Legal->getMaxSafeDepDistBytes() != -1U)
    MaxSafeDepDist =
        Legal->getMaxSafeDepDistBytes() * 8 / Legal->getMaxInterleaveFactor();

  WidestRegister =
      ((WidestRegister < MaxSafeDepDist) ? WidestRegister : MaxSafeDepDist);
  unsigned MaxVectorSize = WidestRegister / WidestType;

  DEBUG(dbgs() << "LV: The Smallest and Widest types: " << SmallestType << " / "
               << WidestType << " bits.\n");
  DEBUG(dbgs() << "LV: The Widest register is: " << WidestRegister
               << " bits.\n");

  if (MaxVectorSize == 0) {
    DEBUG(dbgs() << "LV: The target has no vector registers.\n");
    MaxVectorSize = 1;
  }

  assert(MaxVectorSize <= 64 && "Did not expect to pack so many elements"
                                " into one vector!");

  unsigned VF = MaxVectorSize;
  if (MaximizeBandwidth && !OptForSize) {
    // Collect all viable vectorization factors.
    SmallVector<unsigned, 8> VFs;
    unsigned NewMaxVectorSize = WidestRegister / SmallestType;
    for (unsigned VS = MaxVectorSize; VS <= NewMaxVectorSize; VS *= 2)
      VFs.push_back(VS);

    // For each VF calculate its register usage.
    auto RUs = calculateRegisterUsage(VFs);

    // Select the largest VF which doesn't require more registers than existing
    // ones.
    unsigned TargetNumRegisters = TTI.getNumberOfRegisters(true);
    for (int i = RUs.size() - 1; i >= 0; --i) {
      if (RUs[i].MaxLocalUsers <= TargetNumRegisters) {
        VF = VFs[i];
        break;
      }
    }
  }

  // If we optimize the program for size, avoid creating the tail loop.
  if (OptForSize) {
    unsigned TC = PSE.getSE()->getSmallConstantTripCount(TheLoop);
    DEBUG(dbgs() << "LV: Found trip count: " << TC << '\n');

    // If we don't know the precise trip count, don't try to vectorize.
    if (TC < 2) {
      ORE->emit(
          createMissedAnalysis("UnknownLoopCountComplexCFG")
          << "unable to calculate the loop count due to complex control flow");
      DEBUG(dbgs() << "LV: Aborting. A tail loop is required with -Os/-Oz.\n");
      return Factor;
    }

    // Find the maximum SIMD width that can fit within the trip count.
    VF = TC % MaxVectorSize;

    if (VF == 0)
      VF = MaxVectorSize;
    else {
      // If the trip count that we found modulo the vectorization factor is not
      // zero then we require a tail.
      ORE->emit(createMissedAnalysis("NoTailLoopWithOptForSize")
                << "cannot optimize for size and vectorize at the "
                   "same time. Enable vectorization of this loop "
                   "with '#pragma clang loop vectorize(enable)' "
                   "when compiling with -Os/-Oz");
      DEBUG(dbgs() << "LV: Aborting. A tail loop is required with -Os/-Oz.\n");
      return Factor;
    }
  }

  int UserVF = Hints->getWidth();
  if (UserVF != 0) {
    assert(isPowerOf2_32(UserVF) && "VF needs to be a power of two");
    DEBUG(dbgs() << "LV: Using user VF " << UserVF << ".\n");

    Factor.Width = UserVF;

    collectUniformsAndScalars(UserVF);
    collectInstsToScalarize(UserVF);
    return Factor;
  }

  float Cost = expectedCost(1).first;
#ifndef NDEBUG
  const float ScalarCost = Cost;
#endif /* NDEBUG */
  unsigned Width = 1;
  DEBUG(dbgs() << "LV: Scalar loop costs: " << (int)ScalarCost << ".\n");

  bool ForceVectorization = Hints->getForce() == LoopVectorizeHints::FK_Enabled;
  // Ignore scalar width, because the user explicitly wants vectorization.
  if (ForceVectorization && VF > 1) {
    Width = 2;
    Cost = expectedCost(Width).first / (float)Width;
  }

  for (unsigned i = 2; i <= VF; i *= 2) {
    // Notice that the vector loop needs to be executed less times, so
    // we need to divide the cost of the vector loops by the width of
    // the vector elements.
    VectorizationCostTy C = expectedCost(i);
    float VectorCost = C.first / (float)i;
    DEBUG(dbgs() << "LV: Vector loop of width " << i
                 << " costs: " << (int)VectorCost << ".\n");
    if (!C.second && !ForceVectorization) {
      DEBUG(
          dbgs() << "LV: Not considering vector loop of width " << i
                 << " because it will not generate any vector instructions.\n");
      continue;
    }
    if (VectorCost < Cost) {
      Cost = VectorCost;
      Width = i;
    }
  }

  DEBUG(if (ForceVectorization && Width > 1 && Cost >= ScalarCost) dbgs()
        << "LV: Vectorization seems to be not beneficial, "
        << "but was forced by a user.\n");
  DEBUG(dbgs() << "LV: Selecting VF: " << Width << ".\n");
  Factor.Width = Width;
  Factor.Cost = Width * Cost;
  return Factor;
}

std::pair<unsigned, unsigned>
LoopVectorizationCostModel::getSmallestAndWidestTypes() {
  unsigned MinWidth = -1U;
  unsigned MaxWidth = 8;
  const DataLayout &DL = TheFunction->getParent()->getDataLayout();

  // For each block.
  for (BasicBlock *BB : TheLoop->blocks()) {
    // For each instruction in the loop.
    for (Instruction &I : *BB) {
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
        RecurrenceDescriptor RdxDesc = (*Legal->getReductionVars())[PN];
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
          !Legal->isAccessInterleaved(&I) && !Legal->isLegalGatherOrScatter(&I))
        continue;

      MinWidth = std::min(MinWidth,
                          (unsigned)DL.getTypeSizeInBits(T->getScalarType()));
      MaxWidth = std::max(MaxWidth,
                          (unsigned)DL.getTypeSizeInBits(T->getScalarType()));
    }
  }

  return {MinWidth, MaxWidth};
}

unsigned LoopVectorizationCostModel::selectInterleaveCount(bool OptForSize,
                                                           unsigned VF,
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

  // When we optimize for size, we don't interleave.
  if (OptForSize)
    return 1;

  // We used the distance for the interleave count.
  if (Legal->getMaxSafeDepDistBytes() != -1U)
    return 1;

  // Do not interleave loops with a relatively small trip count.
  unsigned TC = PSE.getSE()->getSmallConstantTripCount(TheLoop);
  if (TC > 1 && TC < TinyTripCountInterleaveThreshold)
    return 1;

  unsigned TargetNumRegisters = TTI.getNumberOfRegisters(VF > 1);
  DEBUG(dbgs() << "LV: The target has " << TargetNumRegisters
               << " registers\n");

  if (VF == 1) {
    if (ForceTargetNumScalarRegs.getNumOccurrences() > 0)
      TargetNumRegisters = ForceTargetNumScalarRegs;
  } else {
    if (ForceTargetNumVectorRegs.getNumOccurrences() > 0)
      TargetNumRegisters = ForceTargetNumVectorRegs;
  }

  RegisterUsage R = calculateRegisterUsage({VF})[0];
  // We divide by these constants so assume that we have at least one
  // instruction that uses at least one register.
  R.MaxLocalUsers = std::max(R.MaxLocalUsers, 1U);
  R.NumInstructions = std::max(R.NumInstructions, 1U);

  // We calculate the interleave count using the following formula.
  // Subtract the number of loop invariants from the number of available
  // registers. These registers are used by all of the interleaved instances.
  // Next, divide the remaining registers by the number of registers that is
  // required by the loop, in order to estimate how many parallel instances
  // fit without causing spills. All of this is rounded down if necessary to be
  // a power of two. We want power of two interleave count to simplify any
  // addressing operations or alignment considerations.
  unsigned IC = PowerOf2Floor((TargetNumRegisters - R.LoopInvariantRegs) /
                              R.MaxLocalUsers);

  // Don't count the induction variable as interleaved.
  if (EnableIndVarRegisterHeur)
    IC = PowerOf2Floor((TargetNumRegisters - R.LoopInvariantRegs - 1) /
                       std::max(1U, (R.MaxLocalUsers - 1)));

  // Clamp the interleave ranges to reasonable counts.
  unsigned MaxInterleaveCount = TTI.getMaxInterleaveFactor(VF);

  // Check if the user has overridden the max.
  if (VF == 1) {
    if (ForceTargetMaxScalarInterleaveFactor.getNumOccurrences() > 0)
      MaxInterleaveCount = ForceTargetMaxScalarInterleaveFactor;
  } else {
    if (ForceTargetMaxVectorInterleaveFactor.getNumOccurrences() > 0)
      MaxInterleaveCount = ForceTargetMaxVectorInterleaveFactor;
  }

  // If we did not calculate the cost for VF (because the user selected the VF)
  // then we calculate the cost of VF here.
  if (LoopCost == 0)
    LoopCost = expectedCost(VF).first;

  // Clamp the calculated IC to be between the 1 and the max interleave count
  // that the target allows.
  if (IC > MaxInterleaveCount)
    IC = MaxInterleaveCount;
  else if (IC < 1)
    IC = 1;

  // Interleave if we vectorized this loop and there is a reduction that could
  // benefit from interleaving.
  if (VF > 1 && Legal->getReductionVars()->size()) {
    DEBUG(dbgs() << "LV: Interleaving because of reductions.\n");
    return IC;
  }

  // Note that if we've already vectorized the loop we will have done the
  // runtime check and so interleaving won't require further checks.
  bool InterleavingRequiresRuntimePointerCheck =
      (VF == 1 && Legal->getRuntimePointerChecking()->Need);

  // We want to interleave small loops in order to reduce the loop overhead and
  // potentially expose ILP opportunities.
  DEBUG(dbgs() << "LV: Loop cost is " << LoopCost << '\n');
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
    if (Legal->getReductionVars()->size() && TheLoop->getLoopDepth() > 1) {
      unsigned F = static_cast<unsigned>(MaxNestedScalarReductionIC);
      SmallIC = std::min(SmallIC, F);
      StoresIC = std::min(StoresIC, F);
      LoadsIC = std::min(LoadsIC, F);
    }

    if (EnableLoadStoreRuntimeInterleave &&
        std::max(StoresIC, LoadsIC) > SmallIC) {
      DEBUG(dbgs() << "LV: Interleaving to saturate store or load ports.\n");
      return std::max(StoresIC, LoadsIC);
    }

    DEBUG(dbgs() << "LV: Interleaving to reduce branch cost.\n");
    return SmallIC;
  }

  // Interleave if this is a large loop (small loops are already dealt with by
  // this point) that could benefit from interleaving.
  bool HasReductions = (Legal->getReductionVars()->size() > 0);
  if (TTI.enableAggressiveInterleaving(HasReductions)) {
    DEBUG(dbgs() << "LV: Interleaving to expose ILP.\n");
    return IC;
  }

  DEBUG(dbgs() << "LV: Not Interleaving.\n");
  return 1;
}

SmallVector<LoopVectorizationCostModel::RegisterUsage, 8>
LoopVectorizationCostModel::calculateRegisterUsage(ArrayRef<unsigned> VFs) {
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
  RU.NumInstructions = 0;

  // Each 'key' in the map opens a new interval. The values
  // of the map are the index of the 'last seen' usage of the
  // instruction that is the key.
  typedef DenseMap<Instruction *, unsigned> IntervalMap;
  // Maps instruction to its index.
  DenseMap<unsigned, Instruction *> IdxToInstr;
  // Marks the end of each interval.
  IntervalMap EndPoint;
  // Saves the list of instruction indices that are used in the loop.
  SmallSet<Instruction *, 8> Ends;
  // Saves the list of values that are used in the loop but are
  // defined outside the loop, such as arguments and constants.
  SmallPtrSet<Value *, 8> LoopInvariants;

  unsigned Index = 0;
  for (BasicBlock *BB : make_range(DFS.beginRPO(), DFS.endRPO())) {
    RU.NumInstructions += BB->size();
    for (Instruction &I : *BB) {
      IdxToInstr[Index++] = &I;

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
        EndPoint[Instr] = Index;
        Ends.insert(Instr);
      }
    }
  }

  // Saves the list of intervals that end with the index in 'key'.
  typedef SmallVector<Instruction *, 2> InstrList;
  DenseMap<unsigned, InstrList> TransposeEnds;

  // Transpose the EndPoints to a list of values that end at each index.
  for (auto &Interval : EndPoint)
    TransposeEnds[Interval.second].push_back(Interval.first);

  SmallSet<Instruction *, 8> OpenIntervals;

  // Get the size of the widest register.
  unsigned MaxSafeDepDist = -1U;
  if (Legal->getMaxSafeDepDistBytes() != -1U)
    MaxSafeDepDist = Legal->getMaxSafeDepDistBytes() * 8;
  unsigned WidestRegister =
      std::min(TTI.getRegisterBitWidth(true), MaxSafeDepDist);
  const DataLayout &DL = TheFunction->getParent()->getDataLayout();

  SmallVector<RegisterUsage, 8> RUs(VFs.size());
  SmallVector<unsigned, 8> MaxUsages(VFs.size(), 0);

  DEBUG(dbgs() << "LV(REG): Calculating max register usage:\n");

  // A lambda that gets the register usage for the given type and VF.
  auto GetRegUsage = [&DL, WidestRegister](Type *Ty, unsigned VF) {
    if (Ty->isTokenTy())
      return 0U;
    unsigned TypeSize = DL.getTypeSizeInBits(Ty->getScalarType());
    return std::max<unsigned>(1, VF * TypeSize / WidestRegister);
  };

  for (unsigned int i = 0; i < Index; ++i) {
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
      if (VFs[j] == 1) {
        MaxUsages[j] = std::max(MaxUsages[j], OpenIntervals.size());
        continue;
      }
      collectUniformsAndScalars(VFs[j]);
      // Count the number of live intervals.
      unsigned RegUsage = 0;
      for (auto Inst : OpenIntervals) {
        // Skip ignored values for VF > 1.
        if (VecValuesToIgnore.count(Inst) ||
            isScalarAfterVectorization(Inst, VFs[j]))
          continue;
        RegUsage += GetRegUsage(Inst->getType(), VFs[j]);
      }
      MaxUsages[j] = std::max(MaxUsages[j], RegUsage);
    }

    DEBUG(dbgs() << "LV(REG): At #" << i << " Interval # "
                 << OpenIntervals.size() << '\n');

    // Add the current instruction to the list of open intervals.
    OpenIntervals.insert(I);
  }

  for (unsigned i = 0, e = VFs.size(); i < e; ++i) {
    unsigned Invariant = 0;
    if (VFs[i] == 1)
      Invariant = LoopInvariants.size();
    else {
      for (auto Inst : LoopInvariants)
        Invariant += GetRegUsage(Inst->getType(), VFs[i]);
    }

    DEBUG(dbgs() << "LV(REG): VF = " << VFs[i] << '\n');
    DEBUG(dbgs() << "LV(REG): Found max usage: " << MaxUsages[i] << '\n');
    DEBUG(dbgs() << "LV(REG): Found invariant usage: " << Invariant << '\n');
    DEBUG(dbgs() << "LV(REG): LoopSize: " << RU.NumInstructions << '\n');

    RU.LoopInvariantRegs = Invariant;
    RU.MaxLocalUsers = MaxUsages[i];
    RUs[i] = RU;
  }

  return RUs;
}

void LoopVectorizationCostModel::collectInstsToScalarize(unsigned VF) {

  // If we aren't vectorizing the loop, or if we've already collected the
  // instructions to scalarize, there's nothing to do. Collection may already
  // have occurred if we have a user-selected VF and are now computing the
  // expected cost for interleaving.
  if (VF < 2 || InstsToScalarize.count(VF))
    return;

  // Initialize a mapping for VF in InstsToScalalarize. If we find that it's
  // not profitable to scalarize any instructions, the presence of VF in the
  // map will indicate that we've analyzed it already.
  ScalarCostsTy &ScalarCostsVF = InstsToScalarize[VF];

  // Find all the instructions that are scalar with predication in the loop and
  // determine if it would be better to not if-convert the blocks they are in.
  // If so, we also record the instructions to scalarize.
  for (BasicBlock *BB : TheLoop->blocks()) {
    if (!Legal->blockNeedsPredication(BB))
      continue;
    for (Instruction &I : *BB)
      if (Legal->isScalarWithPredication(&I)) {
        ScalarCostsTy ScalarCosts;
        if (computePredInstDiscount(&I, ScalarCosts, VF) >= 0)
          ScalarCostsVF.insert(ScalarCosts.begin(), ScalarCosts.end());
      }
  }
}

int LoopVectorizationCostModel::computePredInstDiscount(
    Instruction *PredInst, DenseMap<Instruction *, unsigned> &ScalarCosts,
    unsigned VF) {

  assert(!isUniformAfterVectorization(PredInst, VF) &&
         "Instruction marked uniform-after-vectorization will be predicated");

  // Initialize the discount to zero, meaning that the scalar version and the
  // vector version cost the same.
  int Discount = 0;

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
    if (Legal->isScalarWithPredication(I))
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

  // Returns true if an operand that cannot be scalarized must be extracted
  // from a vector. We will account for this scalarization overhead below. Note
  // that the non-void predicated instructions are placed in their own blocks,
  // and their return values are inserted into vectors. Thus, an extract would
  // still be required.
  auto needsExtract = [&](Instruction *I) -> bool {
    return TheLoop->contains(I) && !isScalarAfterVectorization(I, VF);
  };

  // Compute the expected cost discount from scalarizing the entire expression
  // feeding the predicated instruction. We currently only consider expressions
  // that are single-use instruction chains.
  Worklist.push_back(PredInst);
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();

    // If we've already analyzed the instruction, there's nothing to do.
    if (ScalarCosts.count(I))
      continue;

    // Compute the cost of the vector instruction. Note that this cost already
    // includes the scalarization overhead of the predicated instruction.
    unsigned VectorCost = getInstructionCost(I, VF).first;

    // Compute the cost of the scalarized instruction. This cost is the cost of
    // the instruction as if it wasn't if-converted and instead remained in the
    // predicated block. We will scale this cost by block probability after
    // computing the scalarization overhead.
    unsigned ScalarCost = VF * getInstructionCost(I, 1).first;

    // Compute the scalarization overhead of needed insertelement instructions
    // and phi nodes.
    if (Legal->isScalarWithPredication(I) && !I->getType()->isVoidTy()) {
      ScalarCost += TTI.getScalarizationOverhead(ToVectorTy(I->getType(), VF),
                                                 true, false);
      ScalarCost += VF * TTI.getCFInstrCost(Instruction::PHI);
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
        else if (needsExtract(J))
          ScalarCost += TTI.getScalarizationOverhead(
                              ToVectorTy(J->getType(),VF), false, true);
      }

    // Scale the total scalar cost by block probability.
    ScalarCost /= getReciprocalPredBlockProb();

    // Compute the discount. A non-negative discount means the vector version
    // of the instruction costs more, and scalarizing would be beneficial.
    Discount += VectorCost - ScalarCost;
    ScalarCosts[I] = ScalarCost;
  }

  return Discount;
}

LoopVectorizationCostModel::VectorizationCostTy
LoopVectorizationCostModel::expectedCost(unsigned VF) {
  VectorizationCostTy Cost;

  // Collect Uniform and Scalar instructions after vectorization with VF.
  collectUniformsAndScalars(VF);

  // Collect the instructions (and their associated costs) that will be more
  // profitable to scalarize.
  collectInstsToScalarize(VF);

  // For each block.
  for (BasicBlock *BB : TheLoop->blocks()) {
    VectorizationCostTy BlockCost;

    // For each instruction in the old loop.
    for (Instruction &I : *BB) {
      // Skip dbg intrinsics.
      if (isa<DbgInfoIntrinsic>(I))
        continue;

      // Skip ignored values.
      if (ValuesToIgnore.count(&I))
        continue;

      VectorizationCostTy C = getInstructionCost(&I, VF);

      // Check if we should override the cost.
      if (ForceTargetInstructionCost.getNumOccurrences() > 0)
        C.first = ForceTargetInstructionCost;

      BlockCost.first += C.first;
      BlockCost.second |= C.second;
      DEBUG(dbgs() << "LV: Found an estimated cost of " << C.first << " for VF "
                   << VF << " For instruction: " << I << '\n');
    }

    // If we are vectorizing a predicated block, it will have been
    // if-converted. This means that the block's instructions (aside from
    // stores and instructions that may divide by zero) will now be
    // unconditionally executed. For the scalar case, we may not always execute
    // the predicated block. Thus, scale the block's cost by the probability of
    // executing it.
    if (VF == 1 && Legal->blockNeedsPredication(BB))
      BlockCost.first /= getReciprocalPredBlockProb();

    Cost.first += BlockCost.first;
    Cost.second |= BlockCost.second;
  }

  return Cost;
}

/// \brief Gets Address Access SCEV after verifying that the access pattern
/// is loop invariant except the induction variable dependence.
///
/// This SCEV can be sent to the Target in order to estimate the address
/// calculation cost.
static const SCEV *getAddressAccessSCEV(
              Value *Ptr,
              LoopVectorizationLegality *Legal,
              ScalarEvolution *SE,
              const Loop *TheLoop) {
  auto *Gep = dyn_cast<GetElementPtrInst>(Ptr);
  if (!Gep)
    return nullptr;

  // We are looking for a gep with all loop invariant indices except for one
  // which should be an induction variable.
  unsigned NumOperands = Gep->getNumOperands();
  for (unsigned i = 1; i < NumOperands; ++i) {
    Value *Opd = Gep->getOperand(i);
    if (!SE->isLoopInvariant(SE->getSCEV(Opd), TheLoop) &&
        !Legal->isInductionVariable(Opd))
      return nullptr;
  }

  // Now we know we have a GEP ptr, %inv, %ind, %inv. return the Ptr SCEV.
  return SE->getSCEV(Ptr);
}

static bool isStrideMul(Instruction *I, LoopVectorizationLegality *Legal) {
  return Legal->hasStride(I->getOperand(0)) ||
         Legal->hasStride(I->getOperand(1));
}

unsigned LoopVectorizationCostModel::getMemInstScalarizationCost(Instruction *I,
                                                                 unsigned VF) {
  Type *ValTy = getMemInstValueType(I);
  auto SE = PSE.getSE();

  unsigned Alignment = getMemInstAlignment(I);
  unsigned AS = getMemInstAddressSpace(I);
  Value *Ptr = getPointerOperand(I);
  Type *PtrTy = ToVectorTy(Ptr->getType(), VF);

  // Figure out whether the access is strided and get the stride value
  // if it's known in compile time
  const SCEV *PtrSCEV = getAddressAccessSCEV(Ptr, Legal, SE, TheLoop);

  // Get the cost of the scalar memory instruction and address computation.
  unsigned Cost = VF * TTI.getAddressComputationCost(PtrTy, SE, PtrSCEV);

  Cost += VF *
          TTI.getMemoryOpCost(I->getOpcode(), ValTy->getScalarType(), Alignment,
                              AS);

  // Get the overhead of the extractelement and insertelement instructions
  // we might create due to scalarization.
  Cost += getScalarizationOverhead(I, VF, TTI);

  // If we have a predicated store, it may not be executed for each vector
  // lane. Scale the cost by the probability of executing the predicated
  // block.
  if (Legal->isScalarWithPredication(I))
    Cost /= getReciprocalPredBlockProb();

  return Cost;
}

unsigned LoopVectorizationCostModel::getConsecutiveMemOpCost(Instruction *I,
                                                             unsigned VF) {
  Type *ValTy = getMemInstValueType(I);
  Type *VectorTy = ToVectorTy(ValTy, VF);
  unsigned Alignment = getMemInstAlignment(I);
  Value *Ptr = getPointerOperand(I);
  unsigned AS = getMemInstAddressSpace(I);
  int ConsecutiveStride = Legal->isConsecutivePtr(Ptr);

  assert((ConsecutiveStride == 1 || ConsecutiveStride == -1) &&
         "Stride should be 1 or -1 for consecutive memory access");
  unsigned Cost = 0;
  if (Legal->isMaskRequired(I))
    Cost += TTI.getMaskedMemoryOpCost(I->getOpcode(), VectorTy, Alignment, AS);
  else
    Cost += TTI.getMemoryOpCost(I->getOpcode(), VectorTy, Alignment, AS);

  bool Reverse = ConsecutiveStride < 0;
  if (Reverse)
    Cost += TTI.getShuffleCost(TargetTransformInfo::SK_Reverse, VectorTy, 0);
  return Cost;
}

unsigned LoopVectorizationCostModel::getUniformMemOpCost(Instruction *I,
                                                         unsigned VF) {
  LoadInst *LI = cast<LoadInst>(I);
  Type *ValTy = LI->getType();
  Type *VectorTy = ToVectorTy(ValTy, VF);
  unsigned Alignment = LI->getAlignment();
  unsigned AS = LI->getPointerAddressSpace();

  return TTI.getAddressComputationCost(ValTy) +
         TTI.getMemoryOpCost(Instruction::Load, ValTy, Alignment, AS) +
         TTI.getShuffleCost(TargetTransformInfo::SK_Broadcast, VectorTy);
}

unsigned LoopVectorizationCostModel::getGatherScatterCost(Instruction *I,
                                                          unsigned VF) {
  Type *ValTy = getMemInstValueType(I);
  Type *VectorTy = ToVectorTy(ValTy, VF);
  unsigned Alignment = getMemInstAlignment(I);
  Value *Ptr = getPointerOperand(I);

  return TTI.getAddressComputationCost(VectorTy) +
         TTI.getGatherScatterOpCost(I->getOpcode(), VectorTy, Ptr,
                                    Legal->isMaskRequired(I), Alignment);
}

unsigned LoopVectorizationCostModel::getInterleaveGroupCost(Instruction *I,
                                                            unsigned VF) {
  Type *ValTy = getMemInstValueType(I);
  Type *VectorTy = ToVectorTy(ValTy, VF);
  unsigned AS = getMemInstAddressSpace(I);

  auto Group = Legal->getInterleavedAccessGroup(I);
  assert(Group && "Fail to get an interleaved access group.");

  unsigned InterleaveFactor = Group->getFactor();
  Type *WideVecTy = VectorType::get(ValTy, VF * InterleaveFactor);

  // Holds the indices of existing members in an interleaved load group.
  // An interleaved store group doesn't need this as it doesn't allow gaps.
  SmallVector<unsigned, 4> Indices;
  if (isa<LoadInst>(I)) {
    for (unsigned i = 0; i < InterleaveFactor; i++)
      if (Group->getMember(i))
        Indices.push_back(i);
  }

  // Calculate the cost of the whole interleaved group.
  unsigned Cost = TTI.getInterleavedMemoryOpCost(I->getOpcode(), WideVecTy,
                                                 Group->getFactor(), Indices,
                                                 Group->getAlignment(), AS);

  if (Group->isReverse())
    Cost += Group->getNumMembers() *
            TTI.getShuffleCost(TargetTransformInfo::SK_Reverse, VectorTy, 0);
  return Cost;
}

unsigned LoopVectorizationCostModel::getMemoryInstructionCost(Instruction *I,
                                                              unsigned VF) {

  // Calculate scalar cost only. Vectorization cost should be ready at this
  // moment.
  if (VF == 1) {
    Type *ValTy = getMemInstValueType(I);
    unsigned Alignment = getMemInstAlignment(I);
    unsigned AS = getMemInstAlignment(I);

    return TTI.getAddressComputationCost(ValTy) +
           TTI.getMemoryOpCost(I->getOpcode(), ValTy, Alignment, AS);
  }
  return getWideningCost(I, VF);
}

LoopVectorizationCostModel::VectorizationCostTy
LoopVectorizationCostModel::getInstructionCost(Instruction *I, unsigned VF) {
  // If we know that this instruction will remain uniform, check the cost of
  // the scalar version.
  if (isUniformAfterVectorization(I, VF))
    VF = 1;

  if (VF > 1 && isProfitableToScalarize(I, VF))
    return VectorizationCostTy(InstsToScalarize[VF][I], false);

  Type *VectorTy;
  unsigned C = getInstructionCost(I, VF, VectorTy);

  bool TypeNotScalarized =
      VF > 1 && !VectorTy->isVoidTy() && TTI.getNumberOfParts(VectorTy) < VF;
  return VectorizationCostTy(C, TypeNotScalarized);
}

void LoopVectorizationCostModel::setCostBasedWideningDecision(unsigned VF) {
  if (VF == 1)
    return;
  for (BasicBlock *BB : TheLoop->blocks()) {
    // For each instruction in the old loop.
    for (Instruction &I : *BB) {
      Value *Ptr = getPointerOperand(&I);
      if (!Ptr)
        continue;

      if (isa<LoadInst>(&I) && Legal->isUniform(Ptr)) {
        // Scalar load + broadcast
        unsigned Cost = getUniformMemOpCost(&I, VF);
        setWideningDecision(&I, VF, CM_Scalarize, Cost);
        continue;
      }

      // We assume that widening is the best solution when possible.
      if (Legal->memoryInstructionCanBeWidened(&I, VF)) {
        unsigned Cost = getConsecutiveMemOpCost(&I, VF);
        setWideningDecision(&I, VF, CM_Widen, Cost);
        continue;
      }

      // Choose between Interleaving, Gather/Scatter or Scalarization.
      unsigned InterleaveCost = UINT_MAX;
      unsigned NumAccesses = 1;
      if (Legal->isAccessInterleaved(&I)) {
        auto Group = Legal->getInterleavedAccessGroup(&I);
        assert(Group && "Fail to get an interleaved access group.");

        // Make one decision for the whole group.
        if (getWideningDecision(&I, VF) != CM_Unknown)
          continue;

        NumAccesses = Group->getNumMembers();
        InterleaveCost = getInterleaveGroupCost(&I, VF);
      }

      unsigned GatherScatterCost =
          Legal->isLegalGatherOrScatter(&I)
              ? getGatherScatterCost(&I, VF) * NumAccesses
              : UINT_MAX;

      unsigned ScalarizationCost =
          getMemInstScalarizationCost(&I, VF) * NumAccesses;

      // Choose better solution for the current VF,
      // write down this decision and use it during vectorization.
      unsigned Cost;
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
      if (auto Group = Legal->getInterleavedAccessGroup(&I))
        setWideningDecision(Group, VF, Decision, Cost);
      else
        setWideningDecision(&I, VF, Decision, Cost);
    }
  }
}

unsigned LoopVectorizationCostModel::getInstructionCost(Instruction *I,
                                                        unsigned VF,
                                                        Type *&VectorTy) {
  Type *RetTy = I->getType();
  if (canTruncateToMinimalBitwidth(I, VF))
    RetTy = IntegerType::get(RetTy->getContext(), MinBWs[I]);
  VectorTy = ToVectorTy(RetTy, VF);
  auto SE = PSE.getSE();

  // TODO: We need to estimate the cost of intrinsic calls.
  switch (I->getOpcode()) {
  case Instruction::GetElementPtr:
    // We mark this instruction as zero-cost because the cost of GEPs in
    // vectorized code depends on whether the corresponding memory instruction
    // is scalarized or not. Therefore, we handle GEPs with the memory
    // instruction cost.
    return 0;
  case Instruction::Br: {
    return TTI.getCFInstrCost(I->getOpcode());
  }
  case Instruction::PHI: {
    auto *Phi = cast<PHINode>(I);

    // First-order recurrences are replaced by vector shuffles inside the loop.
    if (VF > 1 && Legal->isFirstOrderRecurrence(Phi))
      return TTI.getShuffleCost(TargetTransformInfo::SK_ExtractSubvector,
                                VectorTy, VF - 1, VectorTy);

    // TODO: IF-converted IFs become selects.
    return 0;
  }
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
    // If we have a predicated instruction, it may not be executed for each
    // vector lane. Get the scalarization cost and scale this amount by the
    // probability of executing the predicated block. If the instruction is not
    // predicated, we fall through to the next case.
    if (VF > 1 && Legal->isScalarWithPredication(I)) {
      unsigned Cost = 0;

      // These instructions have a non-void type, so account for the phi nodes
      // that we will create. This cost is likely to be zero. The phi node
      // cost, if any, should be scaled by the block probability because it
      // models a copy at the end of each predicated block.
      Cost += VF * TTI.getCFInstrCost(Instruction::PHI);

      // The cost of the non-predicated instruction.
      Cost += VF * TTI.getArithmeticInstrCost(I->getOpcode(), RetTy);

      // The cost of insertelement and extractelement instructions needed for
      // scalarization.
      Cost += getScalarizationOverhead(I, VF, TTI);

      // Scale the cost by the probability of executing the predicated blocks.
      // This assumes the predicated block for each vector lane is equally
      // likely.
      return Cost / getReciprocalPredBlockProb();
    }
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
    // Certain instructions can be cheaper to vectorize if they have a constant
    // second vector operand. One example of this are shifts on x86.
    TargetTransformInfo::OperandValueKind Op1VK =
        TargetTransformInfo::OK_AnyValue;
    TargetTransformInfo::OperandValueKind Op2VK =
        TargetTransformInfo::OK_AnyValue;
    TargetTransformInfo::OperandValueProperties Op1VP =
        TargetTransformInfo::OP_None;
    TargetTransformInfo::OperandValueProperties Op2VP =
        TargetTransformInfo::OP_None;
    Value *Op2 = I->getOperand(1);

    // Check for a splat or for a non uniform vector of constants.
    if (isa<ConstantInt>(Op2)) {
      ConstantInt *CInt = cast<ConstantInt>(Op2);
      if (CInt && CInt->getValue().isPowerOf2())
        Op2VP = TargetTransformInfo::OP_PowerOf2;
      Op2VK = TargetTransformInfo::OK_UniformConstantValue;
    } else if (isa<ConstantVector>(Op2) || isa<ConstantDataVector>(Op2)) {
      Op2VK = TargetTransformInfo::OK_NonUniformConstantValue;
      Constant *SplatValue = cast<Constant>(Op2)->getSplatValue();
      if (SplatValue) {
        ConstantInt *CInt = dyn_cast<ConstantInt>(SplatValue);
        if (CInt && CInt->getValue().isPowerOf2())
          Op2VP = TargetTransformInfo::OP_PowerOf2;
        Op2VK = TargetTransformInfo::OK_UniformConstantValue;
      }
    } else if (Legal->isUniform(Op2)) {
      Op2VK = TargetTransformInfo::OK_UniformValue;
    }
    SmallVector<const Value *, 4> Operands(I->operand_values()); 
    return TTI.getArithmeticInstrCost(I->getOpcode(), VectorTy, Op1VK,
                                      Op2VK, Op1VP, Op2VP, Operands);
  }
  case Instruction::Select: {
    SelectInst *SI = cast<SelectInst>(I);
    const SCEV *CondSCEV = SE->getSCEV(SI->getCondition());
    bool ScalarCond = (SE->isLoopInvariant(CondSCEV, TheLoop));
    Type *CondTy = SI->getCondition()->getType();
    if (!ScalarCond)
      CondTy = VectorType::get(CondTy, VF);

    return TTI.getCmpSelInstrCost(I->getOpcode(), VectorTy, CondTy);
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    Type *ValTy = I->getOperand(0)->getType();
    Instruction *Op0AsInstruction = dyn_cast<Instruction>(I->getOperand(0));
    if (canTruncateToMinimalBitwidth(Op0AsInstruction, VF))
      ValTy = IntegerType::get(ValTy->getContext(), MinBWs[Op0AsInstruction]);
    VectorTy = ToVectorTy(ValTy, VF);
    return TTI.getCmpSelInstrCost(I->getOpcode(), VectorTy);
  }
  case Instruction::Store:
  case Instruction::Load: {
    VectorTy = ToVectorTy(getMemInstValueType(I), VF);
    return getMemoryInstructionCost(I, VF);
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
    // We optimize the truncation of induction variables having constant
    // integer steps. The cost of these truncations is the same as the scalar
    // operation.
    if (isOptimizableIVTruncate(I, VF)) {
      auto *Trunc = cast<TruncInst>(I);
      return TTI.getCastInstrCost(Instruction::Trunc, Trunc->getDestTy(),
                                  Trunc->getSrcTy());
    }

    Type *SrcScalarTy = I->getOperand(0)->getType();
    Type *SrcVecTy = ToVectorTy(SrcScalarTy, VF);
    if (canTruncateToMinimalBitwidth(I, VF)) {
      // This cast is going to be shrunk. This may remove the cast or it might
      // turn it into slightly different cast. For example, if MinBW == 16,
      // "zext i8 %1 to i32" becomes "zext i8 %1 to i16".
      //
      // Calculate the modified src and dest types.
      Type *MinVecTy = VectorTy;
      if (I->getOpcode() == Instruction::Trunc) {
        SrcVecTy = smallestIntegerVectorType(SrcVecTy, MinVecTy);
        VectorTy =
            largestIntegerVectorType(ToVectorTy(I->getType(), VF), MinVecTy);
      } else if (I->getOpcode() == Instruction::ZExt ||
                 I->getOpcode() == Instruction::SExt) {
        SrcVecTy = largestIntegerVectorType(SrcVecTy, MinVecTy);
        VectorTy =
            smallestIntegerVectorType(ToVectorTy(I->getType(), VF), MinVecTy);
      }
    }

    return TTI.getCastInstrCost(I->getOpcode(), VectorTy, SrcVecTy);
  }
  case Instruction::Call: {
    bool NeedToScalarize;
    CallInst *CI = cast<CallInst>(I);
    unsigned CallCost = getVectorCallCost(CI, VF, TTI, TLI, NeedToScalarize);
    if (getVectorIntrinsicIDForCall(CI, TLI))
      return std::min(CallCost, getVectorIntrinsicCost(CI, VF, TTI, TLI));
    return CallCost;
  }
  default:
    // The cost of executing VF copies of the scalar instruction. This opcode
    // is unknown. Assume that it is the same as 'mul'.
    return VF * TTI.getArithmeticInstrCost(Instruction::Mul, VectorTy) +
           getScalarizationOverhead(I, VF, TTI);
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
INITIALIZE_PASS_END(LoopVectorize, LV_NAME, lv_name, false, false)

namespace llvm {
Pass *createLoopVectorizePass(bool NoUnrolling, bool AlwaysVectorize) {
  return new LoopVectorize(NoUnrolling, AlwaysVectorize);
}
}

bool LoopVectorizationCostModel::isConsecutiveLoadOrStore(Instruction *Inst) {

  // Check if the pointer operand of a load or store instruction is
  // consecutive.
  if (auto *Ptr = getPointerOperand(Inst))
    return Legal->isConsecutivePtr(Ptr);
  return false;
}

void LoopVectorizationCostModel::collectValuesToIgnore() {
  // Ignore ephemeral values.
  CodeMetrics::collectEphemeralValues(TheLoop, AC, ValuesToIgnore);

  // Ignore type-promoting instructions we identified during reduction
  // detection.
  for (auto &Reduction : *Legal->getReductionVars()) {
    RecurrenceDescriptor &RedDes = Reduction.second;
    SmallPtrSetImpl<Instruction *> &Casts = RedDes.getCastInsts();
    VecValuesToIgnore.insert(Casts.begin(), Casts.end());
  }
}

void InnerLoopUnroller::scalarizeInstruction(Instruction *Instr,
                                             bool IfPredicateInstr) {
  assert(!Instr->getType()->isAggregateType() && "Can't handle vectors");
  // Holds vector parameters or scalars, in case of uniform vals.
  SmallVector<VectorParts, 4> Params;

  setDebugLocFromInst(Builder, Instr);

  // Does this instruction return a value ?
  bool IsVoidRetTy = Instr->getType()->isVoidTy();

  // Initialize a new scalar map entry.
  ScalarParts Entry(UF);

  VectorParts Cond;
  if (IfPredicateInstr)
    Cond = createBlockInMask(Instr->getParent());

  // For each vector unroll 'part':
  for (unsigned Part = 0; Part < UF; ++Part) {
    Entry[Part].resize(1);
    // For each scalar that we create:

    // Start an "if (pred) a[i] = ..." block.
    Value *Cmp = nullptr;
    if (IfPredicateInstr) {
      if (Cond[Part]->getType()->isVectorTy())
        Cond[Part] =
            Builder.CreateExtractElement(Cond[Part], Builder.getInt32(0));
      Cmp = Builder.CreateICmp(ICmpInst::ICMP_EQ, Cond[Part],
                               ConstantInt::get(Cond[Part]->getType(), 1));
    }

    Instruction *Cloned = Instr->clone();
    if (!IsVoidRetTy)
      Cloned->setName(Instr->getName() + ".cloned");

    // Replace the operands of the cloned instructions with their scalar
    // equivalents in the new loop.
    for (unsigned op = 0, e = Instr->getNumOperands(); op != e; ++op) {
      auto *NewOp = getScalarValue(Instr->getOperand(op), Part, 0);
      Cloned->setOperand(op, NewOp);
    }

    // Place the cloned scalar in the new loop.
    Builder.Insert(Cloned);

    // Add the cloned scalar to the scalar map entry.
    Entry[Part][0] = Cloned;

    // If we just cloned a new assumption, add it the assumption cache.
    if (auto *II = dyn_cast<IntrinsicInst>(Cloned))
      if (II->getIntrinsicID() == Intrinsic::assume)
        AC->registerAssumption(II);

    // End if-block.
    if (IfPredicateInstr)
      PredicatedInstructions.push_back(std::make_pair(Cloned, Cmp));
  }
  VectorLoopValueMap.initScalar(Instr, Entry);
}

void InnerLoopUnroller::vectorizeMemoryInstruction(Instruction *Instr) {
  auto *SI = dyn_cast<StoreInst>(Instr);
  bool IfPredicateInstr = (SI && Legal->blockNeedsPredication(SI->getParent()));

  return scalarizeInstruction(Instr, IfPredicateInstr);
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

    // Floating point operations had to be 'fast' to enable the unrolling.
    Value *MulOp = addFastMathFlag(Builder.CreateFMul(C, Step));
    return addFastMathFlag(Builder.CreateBinOp(BinOp, Val, MulOp));
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

bool LoopVectorizePass::processLoop(Loop *L) {
  assert(L->empty() && "Only process inner loops.");

#ifndef NDEBUG
  const std::string DebugLocStr = getDebugLocString(L);
#endif /* NDEBUG */

  DEBUG(dbgs() << "\nLV: Checking a loop in \""
               << L->getHeader()->getParent()->getName() << "\" from "
               << DebugLocStr << "\n");

  LoopVectorizeHints Hints(L, DisableUnrolling, *ORE);

  DEBUG(dbgs() << "LV: Loop hints:"
               << " force="
               << (Hints.getForce() == LoopVectorizeHints::FK_Disabled
                       ? "disabled"
                       : (Hints.getForce() == LoopVectorizeHints::FK_Enabled
                              ? "enabled"
                              : "?"))
               << " width=" << Hints.getWidth()
               << " unroll=" << Hints.getInterleave() << "\n");

  // Function containing loop
  Function *F = L->getHeader()->getParent();

  // Looking at the diagnostic output is the only way to determine if a loop
  // was vectorized (other than looking at the IR or machine code), so it
  // is important to generate an optimization remark for each loop. Most of
  // these messages are generated as OptimizationRemarkAnalysis. Remarks
  // generated as OptimizationRemark and OptimizationRemarkMissed are
  // less verbose reporting vectorized loops and unvectorized loops that may
  // benefit from vectorization, respectively.

  if (!Hints.allowVectorization(F, L, AlwaysVectorize)) {
    DEBUG(dbgs() << "LV: Loop hints prevent vectorization.\n");
    return false;
  }

  // Check the loop for a trip count threshold:
  // do not vectorize loops with a tiny trip count.
  const unsigned MaxTC = SE->getSmallConstantMaxTripCount(L);
  if (MaxTC > 0u && MaxTC < TinyTripCountVectorThreshold) {
    DEBUG(dbgs() << "LV: Found a loop with a very small trip count. "
                 << "This loop is not worth vectorizing.");
    if (Hints.getForce() == LoopVectorizeHints::FK_Enabled)
      DEBUG(dbgs() << " But vectorizing was explicitly forced.\n");
    else {
      DEBUG(dbgs() << "\n");
      ORE->emit(createMissedAnalysis(Hints.vectorizeAnalysisPassName(),
                                     "NotBeneficial", L)
                << "vectorization is not beneficial "
                   "and is not explicitly forced");
      return false;
    }
  }

  PredicatedScalarEvolution PSE(*SE, *L);

  // Check if it is legal to vectorize the loop.
  LoopVectorizationRequirements Requirements(*ORE);
  LoopVectorizationLegality LVL(L, PSE, DT, TLI, AA, F, TTI, GetLAA, LI, ORE,
                                &Requirements, &Hints);
  if (!LVL.canVectorize()) {
    DEBUG(dbgs() << "LV: Not vectorizing: Cannot prove legality.\n");
    emitMissedWarning(F, L, Hints, ORE);
    return false;
  }

  // Use the cost model.
  LoopVectorizationCostModel CM(L, PSE, LI, &LVL, *TTI, TLI, DB, AC, ORE, F,
                                &Hints);
  CM.collectValuesToIgnore();

  // Check the function attributes to find out if this function should be
  // optimized for size.
  bool OptForSize =
      Hints.getForce() != LoopVectorizeHints::FK_Enabled && F->optForSize();

  // Compute the weighted frequency of this loop being executed and see if it
  // is less than 20% of the function entry baseline frequency. Note that we
  // always have a canonical loop here because we think we *can* vectorize.
  // FIXME: This is hidden behind a flag due to pervasive problems with
  // exactly what block frequency models.
  if (LoopVectorizeWithBlockFrequency) {
    BlockFrequency LoopEntryFreq = BFI->getBlockFreq(L->getLoopPreheader());
    if (Hints.getForce() != LoopVectorizeHints::FK_Enabled &&
        LoopEntryFreq < ColdEntryFreq)
      OptForSize = true;
  }

  // Check the function attributes to see if implicit floats are allowed.
  // FIXME: This check doesn't seem possibly correct -- what if the loop is
  // an integer loop and the vector instructions selected are purely integer
  // vector instructions?
  if (F->hasFnAttribute(Attribute::NoImplicitFloat)) {
    DEBUG(dbgs() << "LV: Can't vectorize when the NoImplicitFloat"
                    "attribute is used.\n");
    ORE->emit(createMissedAnalysis(Hints.vectorizeAnalysisPassName(),
                                   "NoImplicitFloat", L)
              << "loop not vectorized due to NoImplicitFloat attribute");
    emitMissedWarning(F, L, Hints, ORE);
    return false;
  }

  // Check if the target supports potentially unsafe FP vectorization.
  // FIXME: Add a check for the type of safety issue (denormal, signaling)
  // for the target we're vectorizing for, to make sure none of the
  // additional fp-math flags can help.
  if (Hints.isPotentiallyUnsafe() &&
      TTI->isFPVectorizationPotentiallyUnsafe()) {
    DEBUG(dbgs() << "LV: Potentially unsafe FP op prevents vectorization.\n");
    ORE->emit(
        createMissedAnalysis(Hints.vectorizeAnalysisPassName(), "UnsafeFP", L)
        << "loop not vectorized due to unsafe FP support.");
    emitMissedWarning(F, L, Hints, ORE);
    return false;
  }

  // Select the optimal vectorization factor.
  const LoopVectorizationCostModel::VectorizationFactor VF =
      CM.selectVectorizationFactor(OptForSize);

  // Select the interleave count.
  unsigned IC = CM.selectInterleaveCount(OptForSize, VF.Width, VF.Cost);

  // Get user interleave count.
  unsigned UserIC = Hints.getInterleave();

  // Identify the diagnostic messages that should be produced.
  std::pair<StringRef, std::string> VecDiagMsg, IntDiagMsg;
  bool VectorizeLoop = true, InterleaveLoop = true;
  if (Requirements.doesNotMeet(F, L, Hints)) {
    DEBUG(dbgs() << "LV: Not vectorizing: loop did not meet vectorization "
                    "requirements.\n");
    emitMissedWarning(F, L, Hints, ORE);
    return false;
  }

  if (VF.Width == 1) {
    DEBUG(dbgs() << "LV: Vectorization is possible but not beneficial.\n");
    VecDiagMsg = std::make_pair(
        "VectorizationNotBeneficial",
        "the cost-model indicates that vectorization is not beneficial");
    VectorizeLoop = false;
  }

  if (IC == 1 && UserIC <= 1) {
    // Tell the user interleaving is not beneficial.
    DEBUG(dbgs() << "LV: Interleaving is not beneficial.\n");
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
    DEBUG(dbgs()
          << "LV: Interleaving is beneficial but is explicitly disabled.");
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
    ORE->emit(OptimizationRemarkMissed(VAPassName, VecDiagMsg.first,
                                         L->getStartLoc(), L->getHeader())
              << VecDiagMsg.second);
    ORE->emit(OptimizationRemarkMissed(LV_NAME, IntDiagMsg.first,
                                         L->getStartLoc(), L->getHeader())
              << IntDiagMsg.second);
    return false;
  } else if (!VectorizeLoop && InterleaveLoop) {
    DEBUG(dbgs() << "LV: Interleave Count is " << IC << '\n');
    ORE->emit(OptimizationRemarkAnalysis(VAPassName, VecDiagMsg.first,
                                         L->getStartLoc(), L->getHeader())
              << VecDiagMsg.second);
  } else if (VectorizeLoop && !InterleaveLoop) {
    DEBUG(dbgs() << "LV: Found a vectorizable loop (" << VF.Width << ") in "
                 << DebugLocStr << '\n');
    ORE->emit(OptimizationRemarkAnalysis(LV_NAME, IntDiagMsg.first,
                                         L->getStartLoc(), L->getHeader())
              << IntDiagMsg.second);
  } else if (VectorizeLoop && InterleaveLoop) {
    DEBUG(dbgs() << "LV: Found a vectorizable loop (" << VF.Width << ") in "
                 << DebugLocStr << '\n');
    DEBUG(dbgs() << "LV: Interleave Count is " << IC << '\n');
  }

  using namespace ore;
  if (!VectorizeLoop) {
    assert(IC > 1 && "interleave count should not be 1 or 0");
    // If we decided that it is not legal to vectorize the loop, then
    // interleave it.
    InnerLoopUnroller Unroller(L, PSE, LI, DT, TLI, TTI, AC, ORE, IC, &LVL,
                               &CM);
    Unroller.vectorize();

    ORE->emit(OptimizationRemark(LV_NAME, "Interleaved", L->getStartLoc(),
                                 L->getHeader())
              << "interleaved loop (interleaved count: "
              << NV("InterleaveCount", IC) << ")");
  } else {
    // If we decided that it is *legal* to vectorize the loop, then do it.
    InnerLoopVectorizer LB(L, PSE, LI, DT, TLI, TTI, AC, ORE, VF.Width, IC,
                           &LVL, &CM);
    LB.vectorize();
    ++LoopsVectorized;

    // Add metadata to disable runtime unrolling a scalar loop when there are
    // no runtime checks about strides and memory. A scalar loop that is
    // rarely used is not worth unrolling.
    if (!LB.areSafetyChecksAdded())
      AddRuntimeUnrollDisableMetaData(L);

    // Report the vectorization decision.
    ORE->emit(OptimizationRemark(LV_NAME, "Vectorized", L->getStartLoc(),
                                 L->getHeader())
              << "vectorized loop (vectorization width: "
              << NV("VectorizationFactor", VF.Width)
              << ", interleaved count: " << NV("InterleaveCount", IC) << ")");
  }

  // Mark the loop as already vectorized to avoid vectorizing again.
  Hints.setAlreadyVectorized();

  DEBUG(verifyFunction(*L->getHeader()->getParent()));
  return true;
}

bool LoopVectorizePass::runImpl(
    Function &F, ScalarEvolution &SE_, LoopInfo &LI_, TargetTransformInfo &TTI_,
    DominatorTree &DT_, BlockFrequencyInfo &BFI_, TargetLibraryInfo *TLI_,
    DemandedBits &DB_, AliasAnalysis &AA_, AssumptionCache &AC_,
    std::function<const LoopAccessInfo &(Loop &)> &GetLAA_,
    OptimizationRemarkEmitter &ORE_) {

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

  // Compute some weights outside of the loop over the loops. Compute this
  // using a BranchProbability to re-use its scaling math.
  const BranchProbability ColdProb(1, 5); // 20%
  ColdEntryFreq = BlockFrequency(BFI->getEntryFreq()) * ColdProb;

  // Don't attempt if
  // 1. the target claims to have no vector registers, and
  // 2. interleaving won't help ILP.
  //
  // The second condition is necessary because, even if the target has no
  // vector registers, loop vectorization may still enable scalar
  // interleaving.
  if (!TTI->getNumberOfRegisters(true) && TTI->getMaxInterleaveFactor(1) < 2)
    return false;

  bool Changed = false;

  // The vectorizer requires loops to be in simplified form.
  // Since simplification may add new inner loops, it has to run before the
  // legality and profitability checks. This means running the loop vectorizer
  // will simplify all loops, regardless of whether anything end up being
  // vectorized.
  for (auto &L : *LI)
    Changed |= simplifyLoop(L, DT, LI, SE, AC, false /* PreserveLCSSA */);

  // Build up a worklist of inner-loops to vectorize. This is necessary as
  // the act of vectorizing or partially unrolling a loop creates new loops
  // and can invalidate iterators across the loops.
  SmallVector<Loop *, 8> Worklist;

  for (Loop *L : *LI)
    addAcyclicInnerLoop(*L, Worklist);

  LoopsAnalyzed += Worklist.size();

  // Now walk the identified inner loops.
  while (!Worklist.empty()) {
    Loop *L = Worklist.pop_back_val();

    // For the inner loops we actually process, form LCSSA to simplify the
    // transform.
    Changed |= formLCSSARecursively(*L, *DT, LI, SE);

    Changed |= processLoop(L);
  }

  // Process each loop nest in the function.
  return Changed;

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

    auto &LAM = AM.getResult<LoopAnalysisManagerFunctionProxy>(F).getManager();
    std::function<const LoopAccessInfo &(Loop &)> GetLAA =
        [&](Loop &L) -> const LoopAccessInfo & {
      LoopStandardAnalysisResults AR = {AA, AC, DT, LI, SE, TLI, TTI};
      return LAM.getResult<LoopAccessAnalysis>(L, AR);
    };
    bool Changed =
        runImpl(F, SE, LI, TTI, DT, BFI, &TLI, DB, AA, AC, GetLAA, ORE);
    if (!Changed)
      return PreservedAnalyses::all();
    PreservedAnalyses PA;
    PA.preserve<LoopAnalysis>();
    PA.preserve<DominatorTreeAnalysis>();
    PA.preserve<BasicAA>();
    PA.preserve<GlobalsAA>();
    return PA;
}
