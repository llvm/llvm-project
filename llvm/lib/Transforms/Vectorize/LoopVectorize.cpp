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
// Other ideas/concepts are from:
//  A. Zaks and D. Nuzman. Autovectorization in GCC-two years later.
//
//  S. Maleki, Y. Gao, M. Garzaran, T. Wong and D. Padua.  An Evaluation of
//  Vectorizing Compilers.
//
//===----------------------------------------------------------------------===//

#define LV_NAME "loop-vectorize"
#define DEBUG_TYPE LV_NAME

#include "llvm/Transforms/Vectorize.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <map>

using namespace llvm;
using namespace llvm::PatternMatch;

static cl::opt<unsigned>
VectorizationFactor("force-vector-width", cl::init(0), cl::Hidden,
                    cl::desc("Sets the SIMD width. Zero is autoselect."));

static cl::opt<unsigned>
VectorizationUnroll("force-vector-unroll", cl::init(0), cl::Hidden,
                    cl::desc("Sets the vectorization unroll count. "
                             "Zero is autoselect."));

static cl::opt<bool>
EnableIfConversion("enable-if-conversion", cl::init(true), cl::Hidden,
                   cl::desc("Enable if-conversion during vectorization."));

/// We don't vectorize loops with a known constant trip count below this number.
static cl::opt<unsigned>
TinyTripCountVectorThreshold("vectorizer-min-trip-count", cl::init(16),
                             cl::Hidden,
                             cl::desc("Don't vectorize loops with a constant "
                                      "trip count that is smaller than this "
                                      "value."));

/// We don't unroll loops with a known constant trip count below this number.
static const unsigned TinyTripCountUnrollThreshold = 128;

/// When performing memory disambiguation checks at runtime do not make more
/// than this number of comparisons.
static const unsigned RuntimeMemoryCheckThreshold = 8;

/// Maximum simd width.
static const unsigned MaxVectorWidth = 64;

/// Maximum vectorization unroll count.
static const unsigned MaxUnrollFactor = 16;

/// The cost of a loop that is considered 'small' by the unroller.
static const unsigned SmallLoopCost = 20;

namespace {

// Forward declarations.
class LoopVectorizationLegality;
class LoopVectorizationCostModel;

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
  InnerLoopVectorizer(Loop *OrigLoop, ScalarEvolution *SE, LoopInfo *LI,
                      DominatorTree *DT, DataLayout *DL,
                      const TargetLibraryInfo *TLI, unsigned VecWidth,
                      unsigned UnrollFactor)
      : OrigLoop(OrigLoop), SE(SE), LI(LI), DT(DT), DL(DL), TLI(TLI),
        VF(VecWidth), UF(UnrollFactor), Builder(SE->getContext()), Induction(0),
        OldInduction(0), WidenMap(UnrollFactor) {}

  // Perform the actual loop widening (vectorization).
  void vectorize(LoopVectorizationLegality *Legal) {
    // Create a new empty loop. Unlink the old loop and connect the new one.
    createEmptyLoop(Legal);
    // Widen each instruction in the old loop to a new one in the new loop.
    // Use the Legality module to find the induction and reduction variables.
    vectorizeLoop(Legal);
    // Register the new loop and update the analysis passes.
    updateAnalysis();
  }

  virtual ~InnerLoopVectorizer() {}

protected:
  /// A small list of PHINodes.
  typedef SmallVector<PHINode*, 4> PhiVector;
  /// When we unroll loops we have multiple vector values for each scalar.
  /// This data structure holds the unrolled and vectorized values that
  /// originated from one scalar instruction.
  typedef SmallVector<Value*, 2> VectorParts;

  // When we if-convert we need create edge masks. We have to cache values so
  // that we don't end up with exponential recursion/IR.
  typedef DenseMap<std::pair<BasicBlock*, BasicBlock*>,
                   VectorParts> EdgeMaskCache;

  /// Add code that checks at runtime if the accessed arrays overlap.
  /// Returns the comparator value or NULL if no check is needed.
  Instruction *addRuntimeCheck(LoopVectorizationLegality *Legal,
                               Instruction *Loc);
  /// Create an empty loop, based on the loop ranges of the old loop.
  void createEmptyLoop(LoopVectorizationLegality *Legal);
  /// Copy and widen the instructions from the old loop.
  virtual void vectorizeLoop(LoopVectorizationLegality *Legal);

  /// \brief The Loop exit block may have single value PHI nodes where the
  /// incoming value is 'Undef'. While vectorizing we only handled real values
  /// that were defined inside the loop. Here we fix the 'undef case'.
  /// See PR14725.
  void fixLCSSAPHIs();

  /// A helper function that computes the predicate of the block BB, assuming
  /// that the header block of the loop is set to True. It returns the *entry*
  /// mask for the block BB.
  VectorParts createBlockInMask(BasicBlock *BB);
  /// A helper function that computes the predicate of the edge between SRC
  /// and DST.
  VectorParts createEdgeMask(BasicBlock *Src, BasicBlock *Dst);

  /// A helper function to vectorize a single BB within the innermost loop.
  void vectorizeBlockInLoop(LoopVectorizationLegality *Legal, BasicBlock *BB,
                            PhiVector *PV);

  /// Vectorize a single PHINode in a block. This method handles the induction
  /// variable canonicalization. It supports both VF = 1 for unrolled loops and
  /// arbitrary length vectors.
  void widenPHIInstruction(Instruction *PN, VectorParts &Entry,
                           LoopVectorizationLegality *Legal,
                           unsigned UF, unsigned VF, PhiVector *PV);

  /// Insert the new loop to the loop hierarchy and pass manager
  /// and update the analysis passes.
  void updateAnalysis();

  /// This instruction is un-vectorizable. Implement it as a sequence
  /// of scalars.
  virtual void scalarizeInstruction(Instruction *Instr);

  /// Vectorize Load and Store instructions,
  virtual void vectorizeMemoryInstruction(Instruction *Instr,
                                  LoopVectorizationLegality *Legal);

  /// Create a broadcast instruction. This method generates a broadcast
  /// instruction (shuffle) for loop invariant values and for the induction
  /// value. If this is the induction variable then we extend it to N, N+1, ...
  /// this is needed because each iteration in the loop corresponds to a SIMD
  /// element.
  virtual Value *getBroadcastInstrs(Value *V);

  /// This function adds 0, 1, 2 ... to each vector element, starting at zero.
  /// If Negate is set then negative numbers are added e.g. (0, -1, -2, ...).
  /// The sequence starts at StartIndex.
  virtual Value *getConsecutiveVector(Value* Val, int StartIdx, bool Negate);

  /// When we go over instructions in the basic block we rely on previous
  /// values within the current basic block or on loop invariant values.
  /// When we widen (vectorize) values we place them in the map. If the values
  /// are not within the map, they have to be loop invariant, so we simply
  /// broadcast them into a vector.
  VectorParts &getVectorValue(Value *V);

  /// Generate a shuffle sequence that will reverse the vector Vec.
  virtual Value *reverseVector(Value *Vec);

  /// This is a helper class that holds the vectorizer state. It maps scalar
  /// instructions to vector instructions. When the code is 'unrolled' then
  /// then a single scalar value is mapped to multiple vector parts. The parts
  /// are stored in the VectorPart type.
  struct ValueMap {
    /// C'tor.  UnrollFactor controls the number of vectors ('parts') that
    /// are mapped.
    ValueMap(unsigned UnrollFactor) : UF(UnrollFactor) {}

    /// \return True if 'Key' is saved in the Value Map.
    bool has(Value *Key) const { return MapStorage.count(Key); }

    /// Initializes a new entry in the map. Sets all of the vector parts to the
    /// save value in 'Val'.
    /// \return A reference to a vector with splat values.
    VectorParts &splat(Value *Key, Value *Val) {
      VectorParts &Entry = MapStorage[Key];
      Entry.assign(UF, Val);
      return Entry;
    }

    ///\return A reference to the value that is stored at 'Key'.
    VectorParts &get(Value *Key) {
      VectorParts &Entry = MapStorage[Key];
      if (Entry.empty())
        Entry.resize(UF);
      assert(Entry.size() == UF);
      return Entry;
    }

  private:
    /// The unroll factor. Each entry in the map stores this number of vector
    /// elements.
    unsigned UF;

    /// Map storage. We use std::map and not DenseMap because insertions to a
    /// dense map invalidates its iterators.
    std::map<Value *, VectorParts> MapStorage;
  };

  /// The original loop.
  Loop *OrigLoop;
  /// Scev analysis to use.
  ScalarEvolution *SE;
  /// Loop Info.
  LoopInfo *LI;
  /// Dominator Tree.
  DominatorTree *DT;
  /// Data Layout.
  DataLayout *DL;
  /// Target Library Info.
  const TargetLibraryInfo *TLI;

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
  ///The ExitBlock of the scalar loop.
  BasicBlock *LoopExitBlock;
  ///The vector loop body.
  BasicBlock *LoopVectorBody;
  ///The scalar loop body.
  BasicBlock *LoopScalarBody;
  /// A list of all bypass blocks. The first block is the entry of the loop.
  SmallVector<BasicBlock *, 4> LoopBypassBlocks;

  /// The new Induction variable which was added to the new block.
  PHINode *Induction;
  /// The induction variable of the old basic block.
  PHINode *OldInduction;
  /// Holds the extended (to the widest induction type) start index.
  Value *ExtendedIdx;
  /// Maps scalars to widened vectors.
  ValueMap WidenMap;
  EdgeMaskCache MaskCache;
};

class InnerLoopUnroller : public InnerLoopVectorizer {
public:
  InnerLoopUnroller(Loop *OrigLoop, ScalarEvolution *SE, LoopInfo *LI,
                    DominatorTree *DT, DataLayout *DL,
                    const TargetLibraryInfo *TLI, unsigned UnrollFactor) :
    InnerLoopVectorizer(OrigLoop, SE, LI, DT, DL, TLI, 1, UnrollFactor) { }

private:
  virtual void scalarizeInstruction(Instruction *Instr);
  virtual void vectorizeMemoryInstruction(Instruction *Instr,
                                          LoopVectorizationLegality *Legal);
  virtual Value *getBroadcastInstrs(Value *V);
  virtual Value *getConsecutiveVector(Value* Val, int StartIdx, bool Negate);
  virtual Value *reverseVector(Value *Vec);
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

/// \brief Set the debug location in the builder using the debug location in the
/// instruction.
static void setDebugLocFromInst(IRBuilder<> &B, const Value *Ptr) {
  if (const Instruction *Inst = dyn_cast_or_null<Instruction>(Ptr))
    B.SetCurrentDebugLocation(Inst->getDebugLoc());
  else
    B.SetCurrentDebugLocation(DebugLoc());
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
  LoopVectorizationLegality(Loop *L, ScalarEvolution *SE, DataLayout *DL,
                            DominatorTree *DT, TargetLibraryInfo *TLI)
      : TheLoop(L), SE(SE), DL(DL), DT(DT), TLI(TLI),
        Induction(0), WidestIndTy(0), HasFunNoNaNAttr(false),
        MaxSafeDepDistBytes(-1U) {}

  /// This enum represents the kinds of reductions that we support.
  enum ReductionKind {
    RK_NoReduction, ///< Not a reduction.
    RK_IntegerAdd,  ///< Sum of integers.
    RK_IntegerMult, ///< Product of integers.
    RK_IntegerOr,   ///< Bitwise or logical OR of numbers.
    RK_IntegerAnd,  ///< Bitwise or logical AND of numbers.
    RK_IntegerXor,  ///< Bitwise or logical XOR of numbers.
    RK_IntegerMinMax, ///< Min/max implemented in terms of select(cmp()).
    RK_FloatAdd,    ///< Sum of floats.
    RK_FloatMult,   ///< Product of floats.
    RK_FloatMinMax  ///< Min/max implemented in terms of select(cmp()).
  };

  /// This enum represents the kinds of inductions that we support.
  enum InductionKind {
    IK_NoInduction,         ///< Not an induction variable.
    IK_IntInduction,        ///< Integer induction variable. Step = 1.
    IK_ReverseIntInduction, ///< Reverse int induction variable. Step = -1.
    IK_PtrInduction,        ///< Pointer induction var. Step = sizeof(elem).
    IK_ReversePtrInduction  ///< Reverse ptr indvar. Step = - sizeof(elem).
  };

  // This enum represents the kind of minmax reduction.
  enum MinMaxReductionKind {
    MRK_Invalid,
    MRK_UIntMin,
    MRK_UIntMax,
    MRK_SIntMin,
    MRK_SIntMax,
    MRK_FloatMin,
    MRK_FloatMax
  };

  /// This struct holds information about reduction variables.
  struct ReductionDescriptor {
    ReductionDescriptor() : StartValue(0), LoopExitInstr(0),
      Kind(RK_NoReduction), MinMaxKind(MRK_Invalid) {}

    ReductionDescriptor(Value *Start, Instruction *Exit, ReductionKind K,
                        MinMaxReductionKind MK)
        : StartValue(Start), LoopExitInstr(Exit), Kind(K), MinMaxKind(MK) {}

    // The starting value of the reduction.
    // It does not have to be zero!
    TrackingVH<Value> StartValue;
    // The instruction who's value is used outside the loop.
    Instruction *LoopExitInstr;
    // The kind of the reduction.
    ReductionKind Kind;
    // If this a min/max reduction the kind of reduction.
    MinMaxReductionKind MinMaxKind;
  };

  /// This POD struct holds information about a potential reduction operation.
  struct ReductionInstDesc {
    ReductionInstDesc(bool IsRedux, Instruction *I) :
      IsReduction(IsRedux), PatternLastInst(I), MinMaxKind(MRK_Invalid) {}

    ReductionInstDesc(Instruction *I, MinMaxReductionKind K) :
      IsReduction(true), PatternLastInst(I), MinMaxKind(K) {}

    // Is this instruction a reduction candidate.
    bool IsReduction;
    // The last instruction in a min/max pattern (select of the select(icmp())
    // pattern), or the current reduction instruction otherwise.
    Instruction *PatternLastInst;
    // If this is a min/max pattern the comparison predicate.
    MinMaxReductionKind MinMaxKind;
  };

  /// This struct holds information about the memory runtime legality
  /// check that a group of pointers do not overlap.
  struct RuntimePointerCheck {
    RuntimePointerCheck() : Need(false) {}

    /// Reset the state of the pointer runtime information.
    void reset() {
      Need = false;
      Pointers.clear();
      Starts.clear();
      Ends.clear();
      IsWritePtr.clear();
      DependencySetId.clear();
    }

    /// Insert a pointer and calculate the start and end SCEVs.
    void insert(ScalarEvolution *SE, Loop *Lp, Value *Ptr, bool WritePtr,
                unsigned DepSetId);

    /// This flag indicates if we need to add the runtime check.
    bool Need;
    /// Holds the pointers that we need to check.
    SmallVector<TrackingVH<Value>, 2> Pointers;
    /// Holds the pointer value at the beginning of the loop.
    SmallVector<const SCEV*, 2> Starts;
    /// Holds the pointer value at the end of the loop.
    SmallVector<const SCEV*, 2> Ends;
    /// Holds the information if this pointer is used for writing to memory.
    SmallVector<bool, 2> IsWritePtr;
    /// Holds the id of the set of pointers that could be dependent because of a
    /// shared underlying object.
    SmallVector<unsigned, 2> DependencySetId;
  };

  /// A struct for saving information about induction variables.
  struct InductionInfo {
    InductionInfo(Value *Start, InductionKind K) : StartValue(Start), IK(K) {}
    InductionInfo() : StartValue(0), IK(IK_NoInduction) {}
    /// Start value.
    TrackingVH<Value> StartValue;
    /// Induction kind.
    InductionKind IK;
  };

  /// ReductionList contains the reduction descriptors for all
  /// of the reductions that were found in the loop.
  typedef DenseMap<PHINode*, ReductionDescriptor> ReductionList;

  /// InductionList saves induction variables and maps them to the
  /// induction descriptor.
  typedef MapVector<PHINode*, InductionInfo> InductionList;

  /// Returns true if it is legal to vectorize this loop.
  /// This does not mean that it is profitable to vectorize this
  /// loop, only that it is legal to do so.
  bool canVectorize();

  /// Returns the Induction variable.
  PHINode *getInduction() { return Induction; }

  /// Returns the reduction variables found in the loop.
  ReductionList *getReductionVars() { return &Reductions; }

  /// Returns the induction variables found in the loop.
  InductionList *getInductionVars() { return &Inductions; }

  /// Returns the widest induction type.
  Type *getWidestInductionType() { return WidestIndTy; }

  /// Returns True if V is an induction variable in this loop.
  bool isInductionVariable(const Value *V);

  /// Return true if the block BB needs to be predicated in order for the loop
  /// to be vectorized.
  bool blockNeedsPredication(BasicBlock *BB);

  /// Check if this  pointer is consecutive when vectorizing. This happens
  /// when the last index of the GEP is the induction variable, or that the
  /// pointer itself is an induction variable.
  /// This check allows us to vectorize A[idx] into a wide load/store.
  /// Returns:
  /// 0 - Stride is unknown or non consecutive.
  /// 1 - Address is consecutive.
  /// -1 - Address is consecutive, and decreasing.
  int isConsecutivePtr(Value *Ptr);

  /// Returns true if the value V is uniform within the loop.
  bool isUniform(Value *V);

  /// Returns true if this instruction will remain scalar after vectorization.
  bool isUniformAfterVectorization(Instruction* I) { return Uniforms.count(I); }

  /// Returns the information that we collected about runtime memory check.
  RuntimePointerCheck *getRuntimePointerCheck() { return &PtrRtCheck; }

  /// This function returns the identity element (or neutral element) for
  /// the operation K.
  static Constant *getReductionIdentity(ReductionKind K, Type *Tp);

  unsigned getMaxSafeDepDistBytes() { return MaxSafeDepDistBytes; }

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

  /// Collect the variables that need to stay uniform after vectorization.
  void collectLoopUniforms();

  /// Return true if all of the instructions in the block can be speculatively
  /// executed. \p SafePtrs is a list of addresses that are known to be legal
  /// and we know that we can read from them without segfault.
  bool blockCanBePredicated(BasicBlock *BB, SmallPtrSet<Value *, 8>& SafePtrs);

  /// Returns True, if 'Phi' is the kind of reduction variable for type
  /// 'Kind'. If this is a reduction variable, it adds it to ReductionList.
  bool AddReductionVar(PHINode *Phi, ReductionKind Kind);
  /// Returns a struct describing if the instruction 'I' can be a reduction
  /// variable of type 'Kind'. If the reduction is a min/max pattern of
  /// select(icmp()) this function advances the instruction pointer 'I' from the
  /// compare instruction to the select instruction and stores this pointer in
  /// 'PatternLastInst' member of the returned struct.
  ReductionInstDesc isReductionInstr(Instruction *I, ReductionKind Kind,
                                     ReductionInstDesc &Desc);
  /// Returns true if the instruction is a Select(ICmp(X, Y), X, Y) instruction
  /// pattern corresponding to a min(X, Y) or max(X, Y).
  static ReductionInstDesc isMinMaxSelectCmpPattern(Instruction *I,
                                                    ReductionInstDesc &Prev);
  /// Returns the induction kind of Phi. This function may return NoInduction
  /// if the PHI is not an induction variable.
  InductionKind isInductionVariable(PHINode *Phi);

  /// The loop that we evaluate.
  Loop *TheLoop;
  /// Scev analysis.
  ScalarEvolution *SE;
  /// DataLayout analysis.
  DataLayout *DL;
  /// Dominators.
  DominatorTree *DT;
  /// Target Library Info.
  TargetLibraryInfo *TLI;

  //  ---  vectorization state --- //

  /// Holds the integer induction variable. This is the counter of the
  /// loop.
  PHINode *Induction;
  /// Holds the reduction variables.
  ReductionList Reductions;
  /// Holds all of the induction variables that we found in the loop.
  /// Notice that inductions don't need to start at zero and that induction
  /// variables can be pointers.
  InductionList Inductions;
  /// Holds the widest induction type encountered.
  Type *WidestIndTy;

  /// Allowed outside users. This holds the reduction
  /// vars which can be accessed from outside the loop.
  SmallPtrSet<Value*, 4> AllowedExit;
  /// This set holds the variables which are known to be uniform after
  /// vectorization.
  SmallPtrSet<Instruction*, 4> Uniforms;
  /// We need to check that all of the pointers in this list are disjoint
  /// at runtime.
  RuntimePointerCheck PtrRtCheck;
  /// Can we assume the absence of NaNs.
  bool HasFunNoNaNAttr;

  unsigned MaxSafeDepDistBytes;
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
  LoopVectorizationCostModel(Loop *L, ScalarEvolution *SE, LoopInfo *LI,
                             LoopVectorizationLegality *Legal,
                             const TargetTransformInfo &TTI,
                             DataLayout *DL, const TargetLibraryInfo *TLI)
      : TheLoop(L), SE(SE), LI(LI), Legal(Legal), TTI(TTI), DL(DL), TLI(TLI) {}

  /// Information about vectorization costs
  struct VectorizationFactor {
    unsigned Width; // Vector width with best cost
    unsigned Cost; // Cost of the loop with that width
  };
  /// \return The most profitable vectorization factor and the cost of that VF.
  /// This method checks every power of two up to VF. If UserVF is not ZERO
  /// then this vectorization factor will be selected if vectorization is
  /// possible.
  VectorizationFactor selectVectorizationFactor(bool OptForSize,
                                                unsigned UserVF);

  /// \return The size (in bits) of the widest type in the code that
  /// needs to be vectorized. We ignore values that remain scalar such as
  /// 64 bit loop indices.
  unsigned getWidestType();

  /// \return The most profitable unroll factor.
  /// If UserUF is non-zero then this method finds the best unroll-factor
  /// based on register pressure and other parameters.
  /// VF and LoopCost are the selected vectorization factor and the cost of the
  /// selected VF.
  unsigned selectUnrollFactor(bool OptForSize, unsigned UserUF, unsigned VF,
                              unsigned LoopCost);

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

  /// \return  information about the register usage of the loop.
  RegisterUsage calculateRegisterUsage();

private:
  /// Returns the expected execution cost. The unit of the cost does
  /// not matter because we use the 'cost' units to compare different
  /// vector widths. The cost that is returned is *not* normalized by
  /// the factor width.
  unsigned expectedCost(unsigned VF);

  /// Returns the execution time cost of an instruction for a given vector
  /// width. Vector width of one means scalar.
  unsigned getInstructionCost(Instruction *I, unsigned VF);

  /// A helper function for converting Scalar types to vector types.
  /// If the incoming type is void, we return void. If the VF is 1, we return
  /// the scalar type.
  static Type* ToVectorTy(Type *Scalar, unsigned VF);

  /// Returns whether the instruction is a load or store and will be a emitted
  /// as a vector operation.
  bool isConsecutiveLoadOrStore(Instruction *I);

  /// The loop that we evaluate.
  Loop *TheLoop;
  /// Scev analysis.
  ScalarEvolution *SE;
  /// Loop Info analysis.
  LoopInfo *LI;
  /// Vectorization legality.
  LoopVectorizationLegality *Legal;
  /// Vector target information.
  const TargetTransformInfo &TTI;
  /// Target data layout information.
  DataLayout *DL;
  /// Target Library Info.
  const TargetLibraryInfo *TLI;
};

/// Utility class for getting and setting loop vectorizer hints in the form
/// of loop metadata.
struct LoopVectorizeHints {
  /// Vectorization width.
  unsigned Width;
  /// Vectorization unroll factor.
  unsigned Unroll;

  LoopVectorizeHints(const Loop *L, bool DisableUnrolling)
  : Width(VectorizationFactor)
  , Unroll(DisableUnrolling ? 1 : VectorizationUnroll)
  , LoopID(L->getLoopID()) {
    getHints(L);
    // The command line options override any loop metadata except for when
    // width == 1 which is used to indicate the loop is already vectorized.
    if (VectorizationFactor.getNumOccurrences() > 0 && Width != 1)
      Width = VectorizationFactor;
    if (VectorizationUnroll.getNumOccurrences() > 0)
      Unroll = VectorizationUnroll;

    DEBUG(if (DisableUnrolling && Unroll == 1)
            dbgs() << "LV: Unrolling disabled by the pass manager\n");
  }

  /// Return the loop vectorizer metadata prefix.
  static StringRef Prefix() { return "llvm.vectorizer."; }

  MDNode *createHint(LLVMContext &Context, StringRef Name, unsigned V) {
    SmallVector<Value*, 2> Vals;
    Vals.push_back(MDString::get(Context, Name));
    Vals.push_back(ConstantInt::get(Type::getInt32Ty(Context), V));
    return MDNode::get(Context, Vals);
  }

  /// Mark the loop L as already vectorized by setting the width to 1.
  void setAlreadyVectorized(Loop *L) {
    LLVMContext &Context = L->getHeader()->getContext();

    Width = 1;

    // Create a new loop id with one more operand for the already_vectorized
    // hint. If the loop already has a loop id then copy the existing operands.
    SmallVector<Value*, 4> Vals(1);
    if (LoopID)
      for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i)
        Vals.push_back(LoopID->getOperand(i));

    Vals.push_back(createHint(Context, Twine(Prefix(), "width").str(), Width));
    Vals.push_back(createHint(Context, Twine(Prefix(), "unroll").str(), 1));

    MDNode *NewLoopID = MDNode::get(Context, Vals);
    // Set operand 0 to refer to the loop id itself.
    NewLoopID->replaceOperandWith(0, NewLoopID);

    L->setLoopID(NewLoopID);
    if (LoopID)
      LoopID->replaceAllUsesWith(NewLoopID);

    LoopID = NewLoopID;
  }

private:
  MDNode *LoopID;

  /// Find hints specified in the loop metadata.
  void getHints(const Loop *L) {
    if (!LoopID)
      return;

    // First operand should refer to the loop id itself.
    assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
    assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

    for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
      const MDString *S = 0;
      SmallVector<Value*, 4> Args;

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

      // Check if the hint starts with the vectorizer prefix.
      StringRef Hint = S->getString();
      if (!Hint.startswith(Prefix()))
        continue;
      // Remove the prefix.
      Hint = Hint.substr(Prefix().size(), StringRef::npos);

      if (Args.size() == 1)
        getHint(Hint, Args[0]);
    }
  }

  // Check string hint with one operand.
  void getHint(StringRef Hint, Value *Arg) {
    const ConstantInt *C = dyn_cast<ConstantInt>(Arg);
    if (!C) return;
    unsigned Val = C->getZExtValue();

    if (Hint == "width") {
      if (isPowerOf2_32(Val) && Val <= MaxVectorWidth)
        Width = Val;
      else
        DEBUG(dbgs() << "LV: ignoring invalid width hint metadata\n");
    } else if (Hint == "unroll") {
      if (isPowerOf2_32(Val) && Val <= MaxUnrollFactor)
        Unroll = Val;
      else
        DEBUG(dbgs() << "LV: ignoring invalid unroll hint metadata\n");
    } else {
      DEBUG(dbgs() << "LV: ignoring unknown hint " << Hint << '\n');
    }
  }
};

/// The LoopVectorize Pass.
struct LoopVectorize : public LoopPass {
  /// Pass identification, replacement for typeid
  static char ID;

  explicit LoopVectorize(bool NoUnrolling = false)
    : LoopPass(ID), DisableUnrolling(NoUnrolling) {
    initializeLoopVectorizePass(*PassRegistry::getPassRegistry());
  }

  ScalarEvolution *SE;
  DataLayout *DL;
  LoopInfo *LI;
  TargetTransformInfo *TTI;
  DominatorTree *DT;
  TargetLibraryInfo *TLI;
  bool DisableUnrolling;

  virtual bool runOnLoop(Loop *L, LPPassManager &LPM) {
    // We only vectorize innermost loops.
    if (!L->empty())
      return false;

    SE = &getAnalysis<ScalarEvolution>();
    DL = getAnalysisIfAvailable<DataLayout>();
    LI = &getAnalysis<LoopInfo>();
    TTI = &getAnalysis<TargetTransformInfo>();
    DT = &getAnalysis<DominatorTree>();
    TLI = getAnalysisIfAvailable<TargetLibraryInfo>();

    // If the target claims to have no vector registers don't attempt
    // vectorization.
    if (!TTI->getNumberOfRegisters(true))
      return false;

    if (DL == NULL) {
      DEBUG(dbgs() << "LV: Not vectorizing because of missing data layout\n");
      return false;
    }

    DEBUG(dbgs() << "LV: Checking a loop in \"" <<
          L->getHeader()->getParent()->getName() << "\"\n");

    LoopVectorizeHints Hints(L, DisableUnrolling);

    if (Hints.Width == 1 && Hints.Unroll == 1) {
      DEBUG(dbgs() << "LV: Not vectorizing.\n");
      return false;
    }

    // Check if it is legal to vectorize the loop.
    LoopVectorizationLegality LVL(L, SE, DL, DT, TLI);
    if (!LVL.canVectorize()) {
      DEBUG(dbgs() << "LV: Not vectorizing.\n");
      return false;
    }

    // Use the cost model.
    LoopVectorizationCostModel CM(L, SE, LI, &LVL, *TTI, DL, TLI);

    // Check the function attributes to find out if this function should be
    // optimized for size.
    Function *F = L->getHeader()->getParent();
    Attribute::AttrKind SzAttr = Attribute::OptimizeForSize;
    Attribute::AttrKind FlAttr = Attribute::NoImplicitFloat;
    unsigned FnIndex = AttributeSet::FunctionIndex;
    bool OptForSize = F->getAttributes().hasAttribute(FnIndex, SzAttr);
    bool NoFloat = F->getAttributes().hasAttribute(FnIndex, FlAttr);

    if (NoFloat) {
      DEBUG(dbgs() << "LV: Can't vectorize when the NoImplicitFloat"
            "attribute is used.\n");
      return false;
    }

    // Select the optimal vectorization factor.
    LoopVectorizationCostModel::VectorizationFactor VF;
    VF = CM.selectVectorizationFactor(OptForSize, Hints.Width);
    // Select the unroll factor.
    unsigned UF = CM.selectUnrollFactor(OptForSize, Hints.Unroll, VF.Width,
                                        VF.Cost);

    DEBUG(dbgs() << "LV: Found a vectorizable loop ("<< VF.Width << ") in "<<
          F->getParent()->getModuleIdentifier() << '\n');
    DEBUG(dbgs() << "LV: Unroll Factor is " << UF << '\n');

    if (VF.Width == 1) {
      DEBUG(dbgs() << "LV: Vectorization is possible but not beneficial.\n");
      if (UF == 1)
        return false;
      // We decided not to vectorize, but we may want to unroll.
      InnerLoopUnroller Unroller(L, SE, LI, DT, DL, TLI, UF);
      Unroller.vectorize(&LVL);
    } else {
      // If we decided that it is *legal* to vectorize the loop then do it.
      InnerLoopVectorizer LB(L, SE, LI, DT, DL, TLI, VF.Width, UF);
      LB.vectorize(&LVL);
    }

    // Mark the loop as already vectorized to avoid vectorizing again.
    Hints.setAlreadyVectorized(L);

    DEBUG(verifyFunction(*L->getHeader()->getParent()));
    return true;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    LoopPass::getAnalysisUsage(AU);
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequiredID(LCSSAID);
    AU.addRequired<DominatorTree>();
    AU.addRequired<LoopInfo>();
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<TargetTransformInfo>();
    AU.addPreserved<LoopInfo>();
    AU.addPreserved<DominatorTree>();
  }

};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Implementation of LoopVectorizationLegality, InnerLoopVectorizer and
// LoopVectorizationCostModel.
//===----------------------------------------------------------------------===//

void
LoopVectorizationLegality::RuntimePointerCheck::insert(ScalarEvolution *SE,
                                                       Loop *Lp, Value *Ptr,
                                                       bool WritePtr,
                                                       unsigned DepSetId) {
  const SCEV *Sc = SE->getSCEV(Ptr);
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Sc);
  assert(AR && "Invalid addrec expression");
  const SCEV *Ex = SE->getBackedgeTakenCount(Lp);
  const SCEV *ScEnd = AR->evaluateAtIteration(Ex, *SE);
  Pointers.push_back(Ptr);
  Starts.push_back(AR->getStart());
  Ends.push_back(ScEnd);
  IsWritePtr.push_back(WritePtr);
  DependencySetId.push_back(DepSetId);
}

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

Value *InnerLoopVectorizer::getConsecutiveVector(Value* Val, int StartIdx,
                                                 bool Negate) {
  assert(Val->getType()->isVectorTy() && "Must be a vector");
  assert(Val->getType()->getScalarType()->isIntegerTy() &&
         "Elem must be an integer");
  // Create the types.
  Type *ITy = Val->getType()->getScalarType();
  VectorType *Ty = cast<VectorType>(Val->getType());
  int VLen = Ty->getNumElements();
  SmallVector<Constant*, 8> Indices;

  // Create a vector of consecutive numbers from zero to VF.
  for (int i = 0; i < VLen; ++i) {
    int64_t Idx = Negate ? (-i) : i;
    Indices.push_back(ConstantInt::get(ITy, StartIdx + Idx, Negate));
  }

  // Add the consecutive indices to the vector value.
  Constant *Cv = ConstantVector::get(Indices);
  assert(Cv->getType() == Val->getType() && "Invalid consecutive vec");
  return Builder.CreateAdd(Val, Cv, "induction");
}

/// \brief Find the operand of the GEP that should be checked for consecutive
/// stores. This ignores trailing indices that have no effect on the final
/// pointer.
static unsigned getGEPInductionOperand(DataLayout *DL,
                                       const GetElementPtrInst *Gep) {
  unsigned LastOperand = Gep->getNumOperands() - 1;
  unsigned GEPAllocSize = DL->getTypeAllocSize(
      cast<PointerType>(Gep->getType()->getScalarType())->getElementType());

  // Walk backwards and try to peel off zeros.
  while (LastOperand > 1 && match(Gep->getOperand(LastOperand), m_Zero())) {
    // Find the type we're currently indexing into.
    gep_type_iterator GEPTI = gep_type_begin(Gep);
    std::advance(GEPTI, LastOperand - 1);

    // If it's a type with the same allocation size as the result of the GEP we
    // can peel off the zero index.
    if (DL->getTypeAllocSize(*GEPTI) != GEPAllocSize)
      break;
    --LastOperand;
  }

  return LastOperand;
}

int LoopVectorizationLegality::isConsecutivePtr(Value *Ptr) {
  assert(Ptr->getType()->isPointerTy() && "Unexpected non ptr");
  // Make sure that the pointer does not point to structs.
  if (Ptr->getType()->getPointerElementType()->isAggregateType())
    return 0;

  // If this value is a pointer induction variable we know it is consecutive.
  PHINode *Phi = dyn_cast_or_null<PHINode>(Ptr);
  if (Phi && Inductions.count(Phi)) {
    InductionInfo II = Inductions[Phi];
    if (IK_PtrInduction == II.IK)
      return 1;
    else if (IK_ReversePtrInduction == II.IK)
      return -1;
  }

  GetElementPtrInst *Gep = dyn_cast_or_null<GetElementPtrInst>(Ptr);
  if (!Gep)
    return 0;

  unsigned NumOperands = Gep->getNumOperands();
  Value *GpPtr = Gep->getPointerOperand();
  // If this GEP value is a consecutive pointer induction variable and all of
  // the indices are constant then we know it is consecutive. We can
  Phi = dyn_cast<PHINode>(GpPtr);
  if (Phi && Inductions.count(Phi)) {

    // Make sure that the pointer does not point to structs.
    PointerType *GepPtrType = cast<PointerType>(GpPtr->getType());
    if (GepPtrType->getElementType()->isAggregateType())
      return 0;

    // Make sure that all of the index operands are loop invariant.
    for (unsigned i = 1; i < NumOperands; ++i)
      if (!SE->isLoopInvariant(SE->getSCEV(Gep->getOperand(i)), TheLoop))
        return 0;

    InductionInfo II = Inductions[Phi];
    if (IK_PtrInduction == II.IK)
      return 1;
    else if (IK_ReversePtrInduction == II.IK)
      return -1;
  }

  unsigned InductionOperand = getGEPInductionOperand(DL, Gep);

  // Check that all of the gep indices are uniform except for our induction
  // operand.
  for (unsigned i = 0; i != NumOperands; ++i)
    if (i != InductionOperand &&
        !SE->isLoopInvariant(SE->getSCEV(Gep->getOperand(i)), TheLoop))
      return 0;

  // We can emit wide load/stores only if the last non-zero index is the
  // induction variable.
  const SCEV *Last = SE->getSCEV(Gep->getOperand(InductionOperand));
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Last)) {
    const SCEV *Step = AR->getStepRecurrence(*SE);

    // The memory is consecutive because the last index is consecutive
    // and all other indices are loop invariant.
    if (Step->isOne())
      return 1;
    if (Step->isAllOnesValue())
      return -1;
  }

  return 0;
}

bool LoopVectorizationLegality::isUniform(Value *V) {
  return (SE->isLoopInvariant(SE->getSCEV(V), TheLoop));
}

InnerLoopVectorizer::VectorParts&
InnerLoopVectorizer::getVectorValue(Value *V) {
  assert(V != Induction && "The new induction variable should not be used.");
  assert(!V->getType()->isVectorTy() && "Can't widen a vector");

  // If we have this scalar in the map, return it.
  if (WidenMap.has(V))
    return WidenMap.get(V);

  // If this scalar is unknown, assume that it is a constant or that it is
  // loop invariant. Broadcast V and save the value for future uses.
  Value *B = getBroadcastInstrs(V);
  return WidenMap.splat(V, B);
}

Value *InnerLoopVectorizer::reverseVector(Value *Vec) {
  assert(Vec->getType()->isVectorTy() && "Invalid type");
  SmallVector<Constant*, 8> ShuffleMask;
  for (unsigned i = 0; i < VF; ++i)
    ShuffleMask.push_back(Builder.getInt32(VF - i - 1));

  return Builder.CreateShuffleVector(Vec, UndefValue::get(Vec->getType()),
                                     ConstantVector::get(ShuffleMask),
                                     "reverse");
}


void InnerLoopVectorizer::vectorizeMemoryInstruction(Instruction *Instr,
                                             LoopVectorizationLegality *Legal) {
  // Attempt to issue a wide load.
  LoadInst *LI = dyn_cast<LoadInst>(Instr);
  StoreInst *SI = dyn_cast<StoreInst>(Instr);

  assert((LI || SI) && "Invalid Load/Store instruction");

  Type *ScalarDataTy = LI ? LI->getType() : SI->getValueOperand()->getType();
  Type *DataTy = VectorType::get(ScalarDataTy, VF);
  Value *Ptr = LI ? LI->getPointerOperand() : SI->getPointerOperand();
  unsigned Alignment = LI ? LI->getAlignment() : SI->getAlignment();
  // An alignment of 0 means target abi alignment. We need to use the scalar's
  // target abi alignment in such a case.
  if (!Alignment)
    Alignment = DL->getABITypeAlignment(ScalarDataTy);
  unsigned AddressSpace = Ptr->getType()->getPointerAddressSpace();
  unsigned ScalarAllocatedSize = DL->getTypeAllocSize(ScalarDataTy);
  unsigned VectorElementSize = DL->getTypeStoreSize(DataTy)/VF;

  if (ScalarAllocatedSize != VectorElementSize)
    return scalarizeInstruction(Instr);

  // If the pointer is loop invariant or if it is non consecutive,
  // scalarize the load.
  int ConsecutiveStride = Legal->isConsecutivePtr(Ptr);
  bool Reverse = ConsecutiveStride < 0;
  bool UniformLoad = LI && Legal->isUniform(Ptr);
  if (!ConsecutiveStride || UniformLoad)
    return scalarizeInstruction(Instr);

  Constant *Zero = Builder.getInt32(0);
  VectorParts &Entry = WidenMap.get(Instr);

  // Handle consecutive loads/stores.
  GetElementPtrInst *Gep = dyn_cast<GetElementPtrInst>(Ptr);
  if (Gep && Legal->isInductionVariable(Gep->getPointerOperand())) {
    setDebugLocFromInst(Builder, Gep);
    Value *PtrOperand = Gep->getPointerOperand();
    Value *FirstBasePtr = getVectorValue(PtrOperand)[0];
    FirstBasePtr = Builder.CreateExtractElement(FirstBasePtr, Zero);

    // Create the new GEP with the new induction variable.
    GetElementPtrInst *Gep2 = cast<GetElementPtrInst>(Gep->clone());
    Gep2->setOperand(0, FirstBasePtr);
    Gep2->setName("gep.indvar.base");
    Ptr = Builder.Insert(Gep2);
  } else if (Gep) {
    setDebugLocFromInst(Builder, Gep);
    assert(SE->isLoopInvariant(SE->getSCEV(Gep->getPointerOperand()),
                               OrigLoop) && "Base ptr must be invariant");

    // The last index does not have to be the induction. It can be
    // consecutive and be a function of the index. For example A[I+1];
    unsigned NumOperands = Gep->getNumOperands();
    unsigned InductionOperand = getGEPInductionOperand(DL, Gep);
    // Create the new GEP with the new induction variable.
    GetElementPtrInst *Gep2 = cast<GetElementPtrInst>(Gep->clone());

    for (unsigned i = 0; i < NumOperands; ++i) {
      Value *GepOperand = Gep->getOperand(i);
      Instruction *GepOperandInst = dyn_cast<Instruction>(GepOperand);

      // Update last index or loop invariant instruction anchored in loop.
      if (i == InductionOperand ||
          (GepOperandInst && OrigLoop->contains(GepOperandInst))) {
        assert((i == InductionOperand ||
               SE->isLoopInvariant(SE->getSCEV(GepOperandInst), OrigLoop)) &&
               "Must be last index or loop invariant");

        VectorParts &GEPParts = getVectorValue(GepOperand);
        Value *Index = GEPParts[0];
        Index = Builder.CreateExtractElement(Index, Zero);
        Gep2->setOperand(i, Index);
        Gep2->setName("gep.indvar.idx");
      }
    }
    Ptr = Builder.Insert(Gep2);
  } else {
    // Use the induction element ptr.
    assert(isa<PHINode>(Ptr) && "Invalid induction ptr");
    setDebugLocFromInst(Builder, Ptr);
    VectorParts &PtrVal = getVectorValue(Ptr);
    Ptr = Builder.CreateExtractElement(PtrVal[0], Zero);
  }

  // Handle Stores:
  if (SI) {
    assert(!Legal->isUniform(SI->getPointerOperand()) &&
           "We do not allow storing to uniform addresses");
    setDebugLocFromInst(Builder, SI);
    // We don't want to update the value in the map as it might be used in
    // another expression. So don't use a reference type for "StoredVal".
    VectorParts StoredVal = getVectorValue(SI->getValueOperand());

    for (unsigned Part = 0; Part < UF; ++Part) {
      // Calculate the pointer for the specific unroll-part.
      Value *PartPtr = Builder.CreateGEP(Ptr, Builder.getInt32(Part * VF));

      if (Reverse) {
        // If we store to reverse consecutive memory locations then we need
        // to reverse the order of elements in the stored value.
        StoredVal[Part] = reverseVector(StoredVal[Part]);
        // If the address is consecutive but reversed, then the
        // wide store needs to start at the last vector element.
        PartPtr = Builder.CreateGEP(Ptr, Builder.getInt32(-Part * VF));
        PartPtr = Builder.CreateGEP(PartPtr, Builder.getInt32(1 - VF));
      }

      Value *VecPtr = Builder.CreateBitCast(PartPtr,
                                            DataTy->getPointerTo(AddressSpace));
      Builder.CreateStore(StoredVal[Part], VecPtr)->setAlignment(Alignment);
    }
    return;
  }

  // Handle loads.
  assert(LI && "Must have a load instruction");
  setDebugLocFromInst(Builder, LI);
  for (unsigned Part = 0; Part < UF; ++Part) {
    // Calculate the pointer for the specific unroll-part.
    Value *PartPtr = Builder.CreateGEP(Ptr, Builder.getInt32(Part * VF));

    if (Reverse) {
      // If the address is consecutive but reversed, then the
      // wide store needs to start at the last vector element.
      PartPtr = Builder.CreateGEP(Ptr, Builder.getInt32(-Part * VF));
      PartPtr = Builder.CreateGEP(PartPtr, Builder.getInt32(1 - VF));
    }

    Value *VecPtr = Builder.CreateBitCast(PartPtr,
                                          DataTy->getPointerTo(AddressSpace));
    Value *LI = Builder.CreateLoad(VecPtr, "wide.load");
    cast<LoadInst>(LI)->setAlignment(Alignment);
    Entry[Part] = Reverse ? reverseVector(LI) :  LI;
  }
}

void InnerLoopVectorizer::scalarizeInstruction(Instruction *Instr) {
  assert(!Instr->getType()->isAggregateType() && "Can't handle vectors");
  // Holds vector parameters or scalars, in case of uniform vals.
  SmallVector<VectorParts, 4> Params;

  setDebugLocFromInst(Builder, Instr);

  // Find all of the vectorized parameters.
  for (unsigned op = 0, e = Instr->getNumOperands(); op != e; ++op) {
    Value *SrcOp = Instr->getOperand(op);

    // If we are accessing the old induction variable, use the new one.
    if (SrcOp == OldInduction) {
      Params.push_back(getVectorValue(SrcOp));
      continue;
    }

    // Try using previously calculated values.
    Instruction *SrcInst = dyn_cast<Instruction>(SrcOp);

    // If the src is an instruction that appeared earlier in the basic block
    // then it should already be vectorized.
    if (SrcInst && OrigLoop->contains(SrcInst)) {
      assert(WidenMap.has(SrcInst) && "Source operand is unavailable");
      // The parameter is a vector value from earlier.
      Params.push_back(WidenMap.get(SrcInst));
    } else {
      // The parameter is a scalar from outside the loop. Maybe even a constant.
      VectorParts Scalars;
      Scalars.append(UF, SrcOp);
      Params.push_back(Scalars);
    }
  }

  assert(Params.size() == Instr->getNumOperands() &&
         "Invalid number of operands");

  // Does this instruction return a value ?
  bool IsVoidRetTy = Instr->getType()->isVoidTy();

  Value *UndefVec = IsVoidRetTy ? 0 :
    UndefValue::get(VectorType::get(Instr->getType(), VF));
  // Create a new entry in the WidenMap and initialize it to Undef or Null.
  VectorParts &VecResults = WidenMap.splat(Instr, UndefVec);

  // For each vector unroll 'part':
  for (unsigned Part = 0; Part < UF; ++Part) {
    // For each scalar that we create:
    for (unsigned Width = 0; Width < VF; ++Width) {
      Instruction *Cloned = Instr->clone();
      if (!IsVoidRetTy)
        Cloned->setName(Instr->getName() + ".cloned");
      // Replace the operands of the cloned instructions with extracted scalars.
      for (unsigned op = 0, e = Instr->getNumOperands(); op != e; ++op) {
        Value *Op = Params[op][Part];
        // Param is a vector. Need to extract the right lane.
        if (Op->getType()->isVectorTy())
          Op = Builder.CreateExtractElement(Op, Builder.getInt32(Width));
        Cloned->setOperand(op, Op);
      }

      // Place the cloned scalar in the new loop.
      Builder.Insert(Cloned);

      // If the original scalar returns a value we need to place it in a vector
      // so that future users will be able to use it.
      if (!IsVoidRetTy)
        VecResults[Part] = Builder.CreateInsertElement(VecResults[Part], Cloned,
                                                       Builder.getInt32(Width));
    }
  }
}

Instruction *
InnerLoopVectorizer::addRuntimeCheck(LoopVectorizationLegality *Legal,
                                     Instruction *Loc) {
  LoopVectorizationLegality::RuntimePointerCheck *PtrRtCheck =
  Legal->getRuntimePointerCheck();

  if (!PtrRtCheck->Need)
    return NULL;

  unsigned NumPointers = PtrRtCheck->Pointers.size();
  SmallVector<TrackingVH<Value> , 2> Starts;
  SmallVector<TrackingVH<Value> , 2> Ends;

  LLVMContext &Ctx = Loc->getContext();
  SCEVExpander Exp(*SE, "induction");

  for (unsigned i = 0; i < NumPointers; ++i) {
    Value *Ptr = PtrRtCheck->Pointers[i];
    const SCEV *Sc = SE->getSCEV(Ptr);

    if (SE->isLoopInvariant(Sc, OrigLoop)) {
      DEBUG(dbgs() << "LV: Adding RT check for a loop invariant ptr:" <<
            *Ptr <<"\n");
      Starts.push_back(Ptr);
      Ends.push_back(Ptr);
    } else {
      DEBUG(dbgs() << "LV: Adding RT check for range:" << *Ptr << '\n');
      unsigned AS = Ptr->getType()->getPointerAddressSpace();

      // Use this type for pointer arithmetic.
      Type *PtrArithTy = Type::getInt8PtrTy(Ctx, AS);

      Value *Start = Exp.expandCodeFor(PtrRtCheck->Starts[i], PtrArithTy, Loc);
      Value *End = Exp.expandCodeFor(PtrRtCheck->Ends[i], PtrArithTy, Loc);
      Starts.push_back(Start);
      Ends.push_back(End);
    }
  }

  IRBuilder<> ChkBuilder(Loc);
  // Our instructions might fold to a constant.
  Value *MemoryRuntimeCheck = 0;
  for (unsigned i = 0; i < NumPointers; ++i) {
    for (unsigned j = i+1; j < NumPointers; ++j) {
      // No need to check if two readonly pointers intersect.
      if (!PtrRtCheck->IsWritePtr[i] && !PtrRtCheck->IsWritePtr[j])
        continue;

      // Only need to check pointers between two different dependency sets.
      if (PtrRtCheck->DependencySetId[i] == PtrRtCheck->DependencySetId[j])
       continue;

      unsigned AS0 = Starts[i]->getType()->getPointerAddressSpace();
      unsigned AS1 = Starts[j]->getType()->getPointerAddressSpace();

      assert((AS0 == Ends[j]->getType()->getPointerAddressSpace()) &&
             (AS1 == Ends[i]->getType()->getPointerAddressSpace()) &&
             "Trying to bounds check pointers with different address spaces");

      Type *PtrArithTy0 = Type::getInt8PtrTy(Ctx, AS0);
      Type *PtrArithTy1 = Type::getInt8PtrTy(Ctx, AS1);

      Value *Start0 = ChkBuilder.CreateBitCast(Starts[i], PtrArithTy0, "bc");
      Value *Start1 = ChkBuilder.CreateBitCast(Starts[j], PtrArithTy1, "bc");
      Value *End0 =   ChkBuilder.CreateBitCast(Ends[i],   PtrArithTy1, "bc");
      Value *End1 =   ChkBuilder.CreateBitCast(Ends[j],   PtrArithTy0, "bc");

      Value *Cmp0 = ChkBuilder.CreateICmpULE(Start0, End1, "bound0");
      Value *Cmp1 = ChkBuilder.CreateICmpULE(Start1, End0, "bound1");
      Value *IsConflict = ChkBuilder.CreateAnd(Cmp0, Cmp1, "found.conflict");
      if (MemoryRuntimeCheck)
        IsConflict = ChkBuilder.CreateOr(MemoryRuntimeCheck, IsConflict,
                                         "conflict.rdx");
      MemoryRuntimeCheck = IsConflict;
    }
  }

  // We have to do this trickery because the IRBuilder might fold the check to a
  // constant expression in which case there is no Instruction anchored in a
  // the block.
  Instruction *Check = BinaryOperator::CreateAnd(MemoryRuntimeCheck,
                                                 ConstantInt::getTrue(Ctx));
  ChkBuilder.Insert(Check, "memcheck.conflict");
  return Check;
}

void
InnerLoopVectorizer::createEmptyLoop(LoopVectorizationLegality *Legal) {
  /*
   In this function we generate a new loop. The new loop will contain
   the vectorized instructions while the old loop will continue to run the
   scalar remainder.

       [ ] <-- vector loop bypass (may consist of multiple blocks).
     /  |
    /   v
   |   [ ]     <-- vector pre header.
   |    |
   |    v
   |   [  ] \
   |   [  ]_|   <-- vector loop.
   |    |
    \   v
      >[ ]   <--- middle-block.
     /  |
    /   v
   |   [ ]     <--- new preheader.
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
  BasicBlock *BypassBlock = OrigLoop->getLoopPreheader();
  BasicBlock *ExitBlock = OrigLoop->getExitBlock();
  assert(ExitBlock && "Must have an exit block");

  // Some loops have a single integer induction variable, while other loops
  // don't. One example is c++ iterators that often have multiple pointer
  // induction variables. In the code below we also support a case where we
  // don't have a single induction variable.
  OldInduction = Legal->getInduction();
  Type *IdxTy = Legal->getWidestInductionType();

  // Find the loop boundaries.
  const SCEV *ExitCount = SE->getBackedgeTakenCount(OrigLoop);
  assert(ExitCount != SE->getCouldNotCompute() && "Invalid loop count");

  // The exit count might have the type of i64 while the phi is i32. This can
  // happen if we have an induction variable that is sign extended before the
  // compare. The only way that we get a backedge taken count is that the
  // induction variable was signed and as such will not overflow. In such a case
  // truncation is legal.
  if (ExitCount->getType()->getPrimitiveSizeInBits() >
      IdxTy->getPrimitiveSizeInBits())
    ExitCount = SE->getTruncateOrNoop(ExitCount, IdxTy);

  ExitCount = SE->getNoopOrZeroExtend(ExitCount, IdxTy);
  // Get the total trip count from the count by adding 1.
  ExitCount = SE->getAddExpr(ExitCount,
                             SE->getConstant(ExitCount->getType(), 1));

  // Expand the trip count and place the new instructions in the preheader.
  // Notice that the pre-header does not change, only the loop body.
  SCEVExpander Exp(*SE, "induction");

  // Count holds the overall loop count (N).
  Value *Count = Exp.expandCodeFor(ExitCount, ExitCount->getType(),
                                   BypassBlock->getTerminator());

  // The loop index does not have to start at Zero. Find the original start
  // value from the induction PHI node. If we don't have an induction variable
  // then we know that it starts at zero.
  Builder.SetInsertPoint(BypassBlock->getTerminator());
  Value *StartIdx = ExtendedIdx = OldInduction ?
    Builder.CreateZExt(OldInduction->getIncomingValueForBlock(BypassBlock),
                       IdxTy):
    ConstantInt::get(IdxTy, 0);

  assert(BypassBlock && "Invalid loop structure");
  LoopBypassBlocks.push_back(BypassBlock);

  // Split the single block loop into the two loop structure described above.
  BasicBlock *VectorPH =
  BypassBlock->splitBasicBlock(BypassBlock->getTerminator(), "vector.ph");
  BasicBlock *VecBody =
  VectorPH->splitBasicBlock(VectorPH->getTerminator(), "vector.body");
  BasicBlock *MiddleBlock =
  VecBody->splitBasicBlock(VecBody->getTerminator(), "middle.block");
  BasicBlock *ScalarPH =
  MiddleBlock->splitBasicBlock(MiddleBlock->getTerminator(), "scalar.ph");

  // Create and register the new vector loop.
  Loop* Lp = new Loop();
  Loop *ParentLoop = OrigLoop->getParentLoop();

  // Insert the new loop into the loop nest and register the new basic blocks
  // before calling any utilities such as SCEV that require valid LoopInfo.
  if (ParentLoop) {
    ParentLoop->addChildLoop(Lp);
    ParentLoop->addBasicBlockToLoop(ScalarPH, LI->getBase());
    ParentLoop->addBasicBlockToLoop(VectorPH, LI->getBase());
    ParentLoop->addBasicBlockToLoop(MiddleBlock, LI->getBase());
  } else {
    LI->addTopLevelLoop(Lp);
  }
  Lp->addBasicBlockToLoop(VecBody, LI->getBase());

  // Use this IR builder to create the loop instructions (Phi, Br, Cmp)
  // inside the loop.
  Builder.SetInsertPoint(VecBody->getFirstNonPHI());

  // Generate the induction variable.
  setDebugLocFromInst(Builder, getDebugLocFromInstOrOperands(OldInduction));
  Induction = Builder.CreatePHI(IdxTy, 2, "index");
  // The loop step is equal to the vectorization factor (num of SIMD elements)
  // times the unroll factor (num of SIMD instructions).
  Constant *Step = ConstantInt::get(IdxTy, VF * UF);

  // This is the IR builder that we use to add all of the logic for bypassing
  // the new vector loop.
  IRBuilder<> BypassBuilder(BypassBlock->getTerminator());
  setDebugLocFromInst(BypassBuilder,
                      getDebugLocFromInstOrOperands(OldInduction));

  // We may need to extend the index in case there is a type mismatch.
  // We know that the count starts at zero and does not overflow.
  if (Count->getType() != IdxTy) {
    // The exit count can be of pointer type. Convert it to the correct
    // integer type.
    if (ExitCount->getType()->isPointerTy())
      Count = BypassBuilder.CreatePointerCast(Count, IdxTy, "ptrcnt.to.int");
    else
      Count = BypassBuilder.CreateZExtOrTrunc(Count, IdxTy, "cnt.cast");
  }

  // Add the start index to the loop count to get the new end index.
  Value *IdxEnd = BypassBuilder.CreateAdd(Count, StartIdx, "end.idx");

  // Now we need to generate the expression for N - (N % VF), which is
  // the part that the vectorized body will execute.
  Value *R = BypassBuilder.CreateURem(Count, Step, "n.mod.vf");
  Value *CountRoundDown = BypassBuilder.CreateSub(Count, R, "n.vec");
  Value *IdxEndRoundDown = BypassBuilder.CreateAdd(CountRoundDown, StartIdx,
                                                     "end.idx.rnd.down");

  // Now, compare the new count to zero. If it is zero skip the vector loop and
  // jump to the scalar loop.
  Value *Cmp = BypassBuilder.CreateICmpEQ(IdxEndRoundDown, StartIdx,
                                          "cmp.zero");

  BasicBlock *LastBypassBlock = BypassBlock;

  // Generate the code that checks in runtime if arrays overlap. We put the
  // checks into a separate block to make the more common case of few elements
  // faster.
  Instruction *MemRuntimeCheck = addRuntimeCheck(Legal,
                                                 BypassBlock->getTerminator());
  if (MemRuntimeCheck) {
    // Create a new block containing the memory check.
    BasicBlock *CheckBlock = BypassBlock->splitBasicBlock(MemRuntimeCheck,
                                                          "vector.memcheck");
    if (ParentLoop)
      ParentLoop->addBasicBlockToLoop(CheckBlock, LI->getBase());
    LoopBypassBlocks.push_back(CheckBlock);

    // Replace the branch into the memory check block with a conditional branch
    // for the "few elements case".
    Instruction *OldTerm = BypassBlock->getTerminator();
    BranchInst::Create(MiddleBlock, CheckBlock, Cmp, OldTerm);
    OldTerm->eraseFromParent();

    Cmp = MemRuntimeCheck;
    LastBypassBlock = CheckBlock;
  }

  LastBypassBlock->getTerminator()->eraseFromParent();
  BranchInst::Create(MiddleBlock, VectorPH, Cmp,
                     LastBypassBlock);

  // We are going to resume the execution of the scalar loop.
  // Go over all of the induction variables that we found and fix the
  // PHIs that are left in the scalar version of the loop.
  // The starting values of PHI nodes depend on the counter of the last
  // iteration in the vectorized loop.
  // If we come from a bypass edge then we need to start from the original
  // start value.

  // This variable saves the new starting index for the scalar loop.
  PHINode *ResumeIndex = 0;
  LoopVectorizationLegality::InductionList::iterator I, E;
  LoopVectorizationLegality::InductionList *List = Legal->getInductionVars();
  // Set builder to point to last bypass block.
  BypassBuilder.SetInsertPoint(LoopBypassBlocks.back()->getTerminator());
  for (I = List->begin(), E = List->end(); I != E; ++I) {
    PHINode *OrigPhi = I->first;
    LoopVectorizationLegality::InductionInfo II = I->second;

    Type *ResumeValTy = (OrigPhi == OldInduction) ? IdxTy : OrigPhi->getType();
    PHINode *ResumeVal = PHINode::Create(ResumeValTy, 2, "resume.val",
                                         MiddleBlock->getTerminator());
    // We might have extended the type of the induction variable but we need a
    // truncated version for the scalar loop.
    PHINode *TruncResumeVal = (OrigPhi == OldInduction) ?
      PHINode::Create(OrigPhi->getType(), 2, "trunc.resume.val",
                      MiddleBlock->getTerminator()) : 0;

    Value *EndValue = 0;
    switch (II.IK) {
    case LoopVectorizationLegality::IK_NoInduction:
      llvm_unreachable("Unknown induction");
    case LoopVectorizationLegality::IK_IntInduction: {
      // Handle the integer induction counter.
      assert(OrigPhi->getType()->isIntegerTy() && "Invalid type");

      // We have the canonical induction variable.
      if (OrigPhi == OldInduction) {
        // Create a truncated version of the resume value for the scalar loop,
        // we might have promoted the type to a larger width.
        EndValue =
          BypassBuilder.CreateTrunc(IdxEndRoundDown, OrigPhi->getType());
        // The new PHI merges the original incoming value, in case of a bypass,
        // or the value at the end of the vectorized loop.
        for (unsigned I = 0, E = LoopBypassBlocks.size(); I != E; ++I)
          TruncResumeVal->addIncoming(II.StartValue, LoopBypassBlocks[I]);
        TruncResumeVal->addIncoming(EndValue, VecBody);

        // We know what the end value is.
        EndValue = IdxEndRoundDown;
        // We also know which PHI node holds it.
        ResumeIndex = ResumeVal;
        break;
      }

      // Not the canonical induction variable - add the vector loop count to the
      // start value.
      Value *CRD = BypassBuilder.CreateSExtOrTrunc(CountRoundDown,
                                                   II.StartValue->getType(),
                                                   "cast.crd");
      EndValue = BypassBuilder.CreateAdd(CRD, II.StartValue , "ind.end");
      break;
    }
    case LoopVectorizationLegality::IK_ReverseIntInduction: {
      // Convert the CountRoundDown variable to the PHI size.
      Value *CRD = BypassBuilder.CreateSExtOrTrunc(CountRoundDown,
                                                   II.StartValue->getType(),
                                                   "cast.crd");
      // Handle reverse integer induction counter.
      EndValue = BypassBuilder.CreateSub(II.StartValue, CRD, "rev.ind.end");
      break;
    }
    case LoopVectorizationLegality::IK_PtrInduction: {
      // For pointer induction variables, calculate the offset using
      // the end index.
      EndValue = BypassBuilder.CreateGEP(II.StartValue, CountRoundDown,
                                         "ptr.ind.end");
      break;
    }
    case LoopVectorizationLegality::IK_ReversePtrInduction: {
      // The value at the end of the loop for the reverse pointer is calculated
      // by creating a GEP with a negative index starting from the start value.
      Value *Zero = ConstantInt::get(CountRoundDown->getType(), 0);
      Value *NegIdx = BypassBuilder.CreateSub(Zero, CountRoundDown,
                                              "rev.ind.end");
      EndValue = BypassBuilder.CreateGEP(II.StartValue, NegIdx,
                                         "rev.ptr.ind.end");
      break;
    }
    }// end of case

    // The new PHI merges the original incoming value, in case of a bypass,
    // or the value at the end of the vectorized loop.
    for (unsigned I = 0, E = LoopBypassBlocks.size(); I != E; ++I) {
      if (OrigPhi == OldInduction)
        ResumeVal->addIncoming(StartIdx, LoopBypassBlocks[I]);
      else
        ResumeVal->addIncoming(II.StartValue, LoopBypassBlocks[I]);
    }
    ResumeVal->addIncoming(EndValue, VecBody);

    // Fix the scalar body counter (PHI node).
    unsigned BlockIdx = OrigPhi->getBasicBlockIndex(ScalarPH);
    // The old inductions phi node in the scalar body needs the truncated value.
    if (OrigPhi == OldInduction)
      OrigPhi->setIncomingValue(BlockIdx, TruncResumeVal);
    else
      OrigPhi->setIncomingValue(BlockIdx, ResumeVal);
  }

  // If we are generating a new induction variable then we also need to
  // generate the code that calculates the exit value. This value is not
  // simply the end of the counter because we may skip the vectorized body
  // in case of a runtime check.
  if (!OldInduction){
    assert(!ResumeIndex && "Unexpected resume value found");
    ResumeIndex = PHINode::Create(IdxTy, 2, "new.indc.resume.val",
                                  MiddleBlock->getTerminator());
    for (unsigned I = 0, E = LoopBypassBlocks.size(); I != E; ++I)
      ResumeIndex->addIncoming(StartIdx, LoopBypassBlocks[I]);
    ResumeIndex->addIncoming(IdxEndRoundDown, VecBody);
  }

  // Make sure that we found the index where scalar loop needs to continue.
  assert(ResumeIndex && ResumeIndex->getType()->isIntegerTy() &&
         "Invalid resume Index");

  // Add a check in the middle block to see if we have completed
  // all of the iterations in the first vector loop.
  // If (N - N%VF) == N, then we *don't* need to run the remainder.
  Value *CmpN = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_EQ, IdxEnd,
                                ResumeIndex, "cmp.n",
                                MiddleBlock->getTerminator());

  BranchInst::Create(ExitBlock, ScalarPH, CmpN, MiddleBlock->getTerminator());
  // Remove the old terminator.
  MiddleBlock->getTerminator()->eraseFromParent();

  // Create i+1 and fill the PHINode.
  Value *NextIdx = Builder.CreateAdd(Induction, Step, "index.next");
  Induction->addIncoming(StartIdx, VectorPH);
  Induction->addIncoming(NextIdx, VecBody);
  // Create the compare.
  Value *ICmp = Builder.CreateICmpEQ(NextIdx, IdxEndRoundDown);
  Builder.CreateCondBr(ICmp, MiddleBlock, VecBody);

  // Now we have two terminators. Remove the old one from the block.
  VecBody->getTerminator()->eraseFromParent();

  // Get ready to start creating new instructions into the vectorized body.
  Builder.SetInsertPoint(VecBody->getFirstInsertionPt());

  // Save the state.
  LoopVectorPreHeader = VectorPH;
  LoopScalarPreHeader = ScalarPH;
  LoopMiddleBlock = MiddleBlock;
  LoopExitBlock = ExitBlock;
  LoopVectorBody = VecBody;
  LoopScalarBody = OldBasicBlock;

  LoopVectorizeHints Hints(Lp, true);
  Hints.setAlreadyVectorized(Lp);
}

/// This function returns the identity element (or neutral element) for
/// the operation K.
Constant*
LoopVectorizationLegality::getReductionIdentity(ReductionKind K, Type *Tp) {
  switch (K) {
  case RK_IntegerXor:
  case RK_IntegerAdd:
  case RK_IntegerOr:
    // Adding, Xoring, Oring zero to a number does not change it.
    return ConstantInt::get(Tp, 0);
  case RK_IntegerMult:
    // Multiplying a number by 1 does not change it.
    return ConstantInt::get(Tp, 1);
  case RK_IntegerAnd:
    // AND-ing a number with an all-1 value does not change it.
    return ConstantInt::get(Tp, -1, true);
  case  RK_FloatMult:
    // Multiplying a number by 1 does not change it.
    return ConstantFP::get(Tp, 1.0L);
  case  RK_FloatAdd:
    // Adding zero to a number does not change it.
    return ConstantFP::get(Tp, 0.0L);
  default:
    llvm_unreachable("Unknown reduction kind");
  }
}

static Intrinsic::ID checkUnaryFloatSignature(const CallInst &I,
                                              Intrinsic::ID ValidIntrinsicID) {
  if (I.getNumArgOperands() != 1 ||
      !I.getArgOperand(0)->getType()->isFloatingPointTy() ||
      I.getType() != I.getArgOperand(0)->getType() ||
      !I.onlyReadsMemory())
    return Intrinsic::not_intrinsic;

  return ValidIntrinsicID;
}

static Intrinsic::ID checkBinaryFloatSignature(const CallInst &I,
                                               Intrinsic::ID ValidIntrinsicID) {
  if (I.getNumArgOperands() != 2 ||
      !I.getArgOperand(0)->getType()->isFloatingPointTy() ||
      !I.getArgOperand(1)->getType()->isFloatingPointTy() ||
      I.getType() != I.getArgOperand(0)->getType() ||
      I.getType() != I.getArgOperand(1)->getType() ||
      !I.onlyReadsMemory())
    return Intrinsic::not_intrinsic;

  return ValidIntrinsicID;
}


static Intrinsic::ID
getIntrinsicIDForCall(CallInst *CI, const TargetLibraryInfo *TLI) {
  // If we have an intrinsic call, check if it is trivially vectorizable.
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::sqrt:
    case Intrinsic::sin:
    case Intrinsic::cos:
    case Intrinsic::exp:
    case Intrinsic::exp2:
    case Intrinsic::log:
    case Intrinsic::log10:
    case Intrinsic::log2:
    case Intrinsic::fabs:
    case Intrinsic::copysign:
    case Intrinsic::floor:
    case Intrinsic::ceil:
    case Intrinsic::trunc:
    case Intrinsic::rint:
    case Intrinsic::nearbyint:
    case Intrinsic::round:
    case Intrinsic::pow:
    case Intrinsic::fma:
    case Intrinsic::fmuladd:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
      return II->getIntrinsicID();
    default:
      return Intrinsic::not_intrinsic;
    }
  }

  if (!TLI)
    return Intrinsic::not_intrinsic;

  LibFunc::Func Func;
  Function *F = CI->getCalledFunction();
  // We're going to make assumptions on the semantics of the functions, check
  // that the target knows that it's available in this environment and it does
  // not have local linkage.
  if (!F || F->hasLocalLinkage() || !TLI->getLibFunc(F->getName(), Func))
    return Intrinsic::not_intrinsic;

  // Otherwise check if we have a call to a function that can be turned into a
  // vector intrinsic.
  switch (Func) {
  default:
    break;
  case LibFunc::sin:
  case LibFunc::sinf:
  case LibFunc::sinl:
    return checkUnaryFloatSignature(*CI, Intrinsic::sin);
  case LibFunc::cos:
  case LibFunc::cosf:
  case LibFunc::cosl:
    return checkUnaryFloatSignature(*CI, Intrinsic::cos);
  case LibFunc::exp:
  case LibFunc::expf:
  case LibFunc::expl:
    return checkUnaryFloatSignature(*CI, Intrinsic::exp);
  case LibFunc::exp2:
  case LibFunc::exp2f:
  case LibFunc::exp2l:
    return checkUnaryFloatSignature(*CI, Intrinsic::exp2);
  case LibFunc::log:
  case LibFunc::logf:
  case LibFunc::logl:
    return checkUnaryFloatSignature(*CI, Intrinsic::log);
  case LibFunc::log10:
  case LibFunc::log10f:
  case LibFunc::log10l:
    return checkUnaryFloatSignature(*CI, Intrinsic::log10);
  case LibFunc::log2:
  case LibFunc::log2f:
  case LibFunc::log2l:
    return checkUnaryFloatSignature(*CI, Intrinsic::log2);
  case LibFunc::fabs:
  case LibFunc::fabsf:
  case LibFunc::fabsl:
    return checkUnaryFloatSignature(*CI, Intrinsic::fabs);
  case LibFunc::copysign:
  case LibFunc::copysignf:
  case LibFunc::copysignl:
    return checkBinaryFloatSignature(*CI, Intrinsic::copysign);
  case LibFunc::floor:
  case LibFunc::floorf:
  case LibFunc::floorl:
    return checkUnaryFloatSignature(*CI, Intrinsic::floor);
  case LibFunc::ceil:
  case LibFunc::ceilf:
  case LibFunc::ceill:
    return checkUnaryFloatSignature(*CI, Intrinsic::ceil);
  case LibFunc::trunc:
  case LibFunc::truncf:
  case LibFunc::truncl:
    return checkUnaryFloatSignature(*CI, Intrinsic::trunc);
  case LibFunc::rint:
  case LibFunc::rintf:
  case LibFunc::rintl:
    return checkUnaryFloatSignature(*CI, Intrinsic::rint);
  case LibFunc::nearbyint:
  case LibFunc::nearbyintf:
  case LibFunc::nearbyintl:
    return checkUnaryFloatSignature(*CI, Intrinsic::nearbyint);
  case LibFunc::round:
  case LibFunc::roundf:
  case LibFunc::roundl:
    return checkUnaryFloatSignature(*CI, Intrinsic::round);
  case LibFunc::pow:
  case LibFunc::powf:
  case LibFunc::powl:
    return checkBinaryFloatSignature(*CI, Intrinsic::pow);
  }

  return Intrinsic::not_intrinsic;
}

/// This function translates the reduction kind to an LLVM binary operator.
static unsigned
getReductionBinOp(LoopVectorizationLegality::ReductionKind Kind) {
  switch (Kind) {
    case LoopVectorizationLegality::RK_IntegerAdd:
      return Instruction::Add;
    case LoopVectorizationLegality::RK_IntegerMult:
      return Instruction::Mul;
    case LoopVectorizationLegality::RK_IntegerOr:
      return Instruction::Or;
    case LoopVectorizationLegality::RK_IntegerAnd:
      return Instruction::And;
    case LoopVectorizationLegality::RK_IntegerXor:
      return Instruction::Xor;
    case LoopVectorizationLegality::RK_FloatMult:
      return Instruction::FMul;
    case LoopVectorizationLegality::RK_FloatAdd:
      return Instruction::FAdd;
    case LoopVectorizationLegality::RK_IntegerMinMax:
      return Instruction::ICmp;
    case LoopVectorizationLegality::RK_FloatMinMax:
      return Instruction::FCmp;
    default:
      llvm_unreachable("Unknown reduction operation");
  }
}

Value *createMinMaxOp(IRBuilder<> &Builder,
                      LoopVectorizationLegality::MinMaxReductionKind RK,
                      Value *Left,
                      Value *Right) {
  CmpInst::Predicate P = CmpInst::ICMP_NE;
  switch (RK) {
  default:
    llvm_unreachable("Unknown min/max reduction kind");
  case LoopVectorizationLegality::MRK_UIntMin:
    P = CmpInst::ICMP_ULT;
    break;
  case LoopVectorizationLegality::MRK_UIntMax:
    P = CmpInst::ICMP_UGT;
    break;
  case LoopVectorizationLegality::MRK_SIntMin:
    P = CmpInst::ICMP_SLT;
    break;
  case LoopVectorizationLegality::MRK_SIntMax:
    P = CmpInst::ICMP_SGT;
    break;
  case LoopVectorizationLegality::MRK_FloatMin:
    P = CmpInst::FCMP_OLT;
    break;
  case LoopVectorizationLegality::MRK_FloatMax:
    P = CmpInst::FCMP_OGT;
    break;
  }

  Value *Cmp;
  if (RK == LoopVectorizationLegality::MRK_FloatMin ||
      RK == LoopVectorizationLegality::MRK_FloatMax)
    Cmp = Builder.CreateFCmp(P, Left, Right, "rdx.minmax.cmp");
  else
    Cmp = Builder.CreateICmp(P, Left, Right, "rdx.minmax.cmp");

  Value *Select = Builder.CreateSelect(Cmp, Left, Right, "rdx.minmax.select");
  return Select;
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
    Instruction *In = I++;

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

void
InnerLoopVectorizer::vectorizeLoop(LoopVectorizationLegality *Legal) {
  //===------------------------------------------------===//
  //
  // Notice: any optimization or new instruction that go
  // into the code below should be also be implemented in
  // the cost-model.
  //
  //===------------------------------------------------===//
  Constant *Zero = Builder.getInt32(0);

  // In order to support reduction variables we need to be able to vectorize
  // Phi nodes. Phi nodes have cycles, so we need to vectorize them in two
  // stages. First, we create a new vector PHI node with no incoming edges.
  // We use this value when we vectorize all of the instructions that use the
  // PHI. Next, after all of the instructions in the block are complete we
  // add the new incoming edges to the PHI. At this point all of the
  // instructions in the basic block are vectorized, so we can use them to
  // construct the PHI.
  PhiVector RdxPHIsToFix;

  // Scan the loop in a topological order to ensure that defs are vectorized
  // before users.
  LoopBlocksDFS DFS(OrigLoop);
  DFS.perform(LI);

  // Vectorize all of the blocks in the original loop.
  for (LoopBlocksDFS::RPOIterator bb = DFS.beginRPO(),
       be = DFS.endRPO(); bb != be; ++bb)
    vectorizeBlockInLoop(Legal, *bb, &RdxPHIsToFix);

  // At this point every instruction in the original loop is widened to
  // a vector form. We are almost done. Now, we need to fix the PHI nodes
  // that we vectorized. The PHI nodes are currently empty because we did
  // not want to introduce cycles. Notice that the remaining PHI nodes
  // that we need to fix are reduction variables.

  // Create the 'reduced' values for each of the induction vars.
  // The reduced values are the vector values that we scalarize and combine
  // after the loop is finished.
  for (PhiVector::iterator it = RdxPHIsToFix.begin(), e = RdxPHIsToFix.end();
       it != e; ++it) {
    PHINode *RdxPhi = *it;
    assert(RdxPhi && "Unable to recover vectorized PHI");

    // Find the reduction variable descriptor.
    assert(Legal->getReductionVars()->count(RdxPhi) &&
           "Unable to find the reduction variable");
    LoopVectorizationLegality::ReductionDescriptor RdxDesc =
    (*Legal->getReductionVars())[RdxPhi];

    setDebugLocFromInst(Builder, RdxDesc.StartValue);

    // We need to generate a reduction vector from the incoming scalar.
    // To do so, we need to generate the 'identity' vector and overide
    // one of the elements with the incoming scalar reduction. We need
    // to do it in the vector-loop preheader.
    Builder.SetInsertPoint(LoopBypassBlocks.front()->getTerminator());

    // This is the vector-clone of the value that leaves the loop.
    VectorParts &VectorExit = getVectorValue(RdxDesc.LoopExitInstr);
    Type *VecTy = VectorExit[0]->getType();

    // Find the reduction identity variable. Zero for addition, or, xor,
    // one for multiplication, -1 for And.
    Value *Identity;
    Value *VectorStart;
    if (RdxDesc.Kind == LoopVectorizationLegality::RK_IntegerMinMax ||
        RdxDesc.Kind == LoopVectorizationLegality::RK_FloatMinMax) {
      // MinMax reduction have the start value as their identify.
      if (VF == 1) {
        VectorStart = Identity = RdxDesc.StartValue;
      } else {
        VectorStart = Identity = Builder.CreateVectorSplat(VF,
                                                           RdxDesc.StartValue,
                                                           "minmax.ident");
      }
    } else {
      // Handle other reduction kinds:
      Constant *Iden =
      LoopVectorizationLegality::getReductionIdentity(RdxDesc.Kind,
                                                      VecTy->getScalarType());
      if (VF == 1) {
        Identity = Iden;
        // This vector is the Identity vector where the first element is the
        // incoming scalar reduction.
        VectorStart = RdxDesc.StartValue;
      } else {
        Identity = ConstantVector::getSplat(VF, Iden);

        // This vector is the Identity vector where the first element is the
        // incoming scalar reduction.
        VectorStart = Builder.CreateInsertElement(Identity,
                                                  RdxDesc.StartValue, Zero);
      }
    }

    // Fix the vector-loop phi.
    // We created the induction variable so we know that the
    // preheader is the first entry.
    BasicBlock *VecPreheader = Induction->getIncomingBlock(0);

    // Reductions do not have to start at zero. They can start with
    // any loop invariant values.
    VectorParts &VecRdxPhi = WidenMap.get(RdxPhi);
    BasicBlock *Latch = OrigLoop->getLoopLatch();
    Value *LoopVal = RdxPhi->getIncomingValueForBlock(Latch);
    VectorParts &Val = getVectorValue(LoopVal);
    for (unsigned part = 0; part < UF; ++part) {
      // Make sure to add the reduction stat value only to the
      // first unroll part.
      Value *StartVal = (part == 0) ? VectorStart : Identity;
      cast<PHINode>(VecRdxPhi[part])->addIncoming(StartVal, VecPreheader);
      cast<PHINode>(VecRdxPhi[part])->addIncoming(Val[part], LoopVectorBody);
    }

    // Before each round, move the insertion point right between
    // the PHIs and the values we are going to write.
    // This allows us to write both PHINodes and the extractelement
    // instructions.
    Builder.SetInsertPoint(LoopMiddleBlock->getFirstInsertionPt());

    VectorParts RdxParts;
    setDebugLocFromInst(Builder, RdxDesc.LoopExitInstr);
    for (unsigned part = 0; part < UF; ++part) {
      // This PHINode contains the vectorized reduction variable, or
      // the initial value vector, if we bypass the vector loop.
      VectorParts &RdxExitVal = getVectorValue(RdxDesc.LoopExitInstr);
      PHINode *NewPhi = Builder.CreatePHI(VecTy, 2, "rdx.vec.exit.phi");
      Value *StartVal = (part == 0) ? VectorStart : Identity;
      for (unsigned I = 0, E = LoopBypassBlocks.size(); I != E; ++I)
        NewPhi->addIncoming(StartVal, LoopBypassBlocks[I]);
      NewPhi->addIncoming(RdxExitVal[part], LoopVectorBody);
      RdxParts.push_back(NewPhi);
    }

    // Reduce all of the unrolled parts into a single vector.
    Value *ReducedPartRdx = RdxParts[0];
    unsigned Op = getReductionBinOp(RdxDesc.Kind);
    setDebugLocFromInst(Builder, ReducedPartRdx);
    for (unsigned part = 1; part < UF; ++part) {
      if (Op != Instruction::ICmp && Op != Instruction::FCmp)
        ReducedPartRdx = Builder.CreateBinOp((Instruction::BinaryOps)Op,
                                             RdxParts[part], ReducedPartRdx,
                                             "bin.rdx");
      else
        ReducedPartRdx = createMinMaxOp(Builder, RdxDesc.MinMaxKind,
                                        ReducedPartRdx, RdxParts[part]);
    }

    if (VF > 1) {
      // VF is a power of 2 so we can emit the reduction using log2(VF) shuffles
      // and vector ops, reducing the set of values being computed by half each
      // round.
      assert(isPowerOf2_32(VF) &&
             "Reduction emission only supported for pow2 vectors!");
      Value *TmpVec = ReducedPartRdx;
      SmallVector<Constant*, 32> ShuffleMask(VF, 0);
      for (unsigned i = VF; i != 1; i >>= 1) {
        // Move the upper half of the vector to the lower half.
        for (unsigned j = 0; j != i/2; ++j)
          ShuffleMask[j] = Builder.getInt32(i/2 + j);

        // Fill the rest of the mask with undef.
        std::fill(&ShuffleMask[i/2], ShuffleMask.end(),
                  UndefValue::get(Builder.getInt32Ty()));

        Value *Shuf =
        Builder.CreateShuffleVector(TmpVec,
                                    UndefValue::get(TmpVec->getType()),
                                    ConstantVector::get(ShuffleMask),
                                    "rdx.shuf");

        if (Op != Instruction::ICmp && Op != Instruction::FCmp)
          TmpVec = Builder.CreateBinOp((Instruction::BinaryOps)Op, TmpVec, Shuf,
                                       "bin.rdx");
        else
          TmpVec = createMinMaxOp(Builder, RdxDesc.MinMaxKind, TmpVec, Shuf);
      }

      // The result is in the first element of the vector.
      ReducedPartRdx = Builder.CreateExtractElement(TmpVec,
                                                    Builder.getInt32(0));
    }

    // Now, we need to fix the users of the reduction variable
    // inside and outside of the scalar remainder loop.
    // We know that the loop is in LCSSA form. We need to update the
    // PHI nodes in the exit blocks.
    for (BasicBlock::iterator LEI = LoopExitBlock->begin(),
         LEE = LoopExitBlock->end(); LEI != LEE; ++LEI) {
      PHINode *LCSSAPhi = dyn_cast<PHINode>(LEI);
      if (!LCSSAPhi) break;

      // All PHINodes need to have a single entry edge, or two if
      // we already fixed them.
      assert(LCSSAPhi->getNumIncomingValues() < 3 && "Invalid LCSSA PHI");

      // We found our reduction value exit-PHI. Update it with the
      // incoming bypass edge.
      if (LCSSAPhi->getIncomingValue(0) == RdxDesc.LoopExitInstr) {
        // Add an edge coming from the bypass.
        LCSSAPhi->addIncoming(ReducedPartRdx, LoopMiddleBlock);
        break;
      }
    }// end of the LCSSA phi scan.

    // Fix the scalar loop reduction variable with the incoming reduction sum
    // from the vector body and from the backedge value.
    int IncomingEdgeBlockIdx =
    (RdxPhi)->getBasicBlockIndex(OrigLoop->getLoopLatch());
    assert(IncomingEdgeBlockIdx >= 0 && "Invalid block index");
    // Pick the other block.
    int SelfEdgeBlockIdx = (IncomingEdgeBlockIdx ? 0 : 1);
    (RdxPhi)->setIncomingValue(SelfEdgeBlockIdx, ReducedPartRdx);
    (RdxPhi)->setIncomingValue(IncomingEdgeBlockIdx, RdxDesc.LoopExitInstr);
  }// end of for each redux variable.

  fixLCSSAPHIs();

  // Remove redundant induction instructions.
  cse(LoopVectorBody);
}

void InnerLoopVectorizer::fixLCSSAPHIs() {
  for (BasicBlock::iterator LEI = LoopExitBlock->begin(),
       LEE = LoopExitBlock->end(); LEI != LEE; ++LEI) {
    PHINode *LCSSAPhi = dyn_cast<PHINode>(LEI);
    if (!LCSSAPhi) break;
    if (LCSSAPhi->getNumIncomingValues() == 1)
      LCSSAPhi->addIncoming(UndefValue::get(LCSSAPhi->getType()),
                            LoopMiddleBlock);
  }
} 

InnerLoopVectorizer::VectorParts
InnerLoopVectorizer::createEdgeMask(BasicBlock *Src, BasicBlock *Dst) {
  assert(std::find(pred_begin(Dst), pred_end(Dst), Src) != pred_end(Dst) &&
         "Invalid edge");

  // Look for cached value.
  std::pair<BasicBlock*, BasicBlock*> Edge(Src, Dst);
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

void InnerLoopVectorizer::widenPHIInstruction(Instruction *PN,
                                              InnerLoopVectorizer::VectorParts &Entry,
                                              LoopVectorizationLegality *Legal,
                                              unsigned UF, unsigned VF, PhiVector *PV) {
  PHINode* P = cast<PHINode>(PN);
  // Handle reduction variables:
  if (Legal->getReductionVars()->count(P)) {
    for (unsigned part = 0; part < UF; ++part) {
      // This is phase one of vectorizing PHIs.
      Type *VecTy = (VF == 1) ? PN->getType() :
      VectorType::get(PN->getType(), VF);
      Entry[part] = PHINode::Create(VecTy, 2, "vec.phi",
                                    LoopVectorBody-> getFirstInsertionPt());
    }
    PV->push_back(P);
    return;
  }

  setDebugLocFromInst(Builder, P);
  // Check for PHI nodes that are lowered to vector selects.
  if (P->getParent() != OrigLoop->getHeader()) {
    // We know that all PHIs in non header blocks are converted into
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
    for (unsigned In = 0; In < NumIncoming; In++) {
      VectorParts Cond = createEdgeMask(P->getIncomingBlock(In),
                                        P->getParent());
      VectorParts &In0 = getVectorValue(P->getIncomingValue(In));

      for (unsigned part = 0; part < UF; ++part) {
        // We might have single edge PHIs (blocks) - use an identity
        // 'select' for the first PHI operand.
        if (In == 0)
          Entry[part] = Builder.CreateSelect(Cond[part], In0[part],
                                             In0[part]);
        else
          // Select between the current value and the previous incoming edge
          // based on the incoming mask.
          Entry[part] = Builder.CreateSelect(Cond[part], In0[part],
                                             Entry[part], "predphi");
      }
    }
    return;
  }

  // This PHINode must be an induction variable.
  // Make sure that we know about it.
  assert(Legal->getInductionVars()->count(P) &&
         "Not an induction variable");

  LoopVectorizationLegality::InductionInfo II =
  Legal->getInductionVars()->lookup(P);

  switch (II.IK) {
    case LoopVectorizationLegality::IK_NoInduction:
      llvm_unreachable("Unknown induction");
    case LoopVectorizationLegality::IK_IntInduction: {
      assert(P->getType() == II.StartValue->getType() && "Types must match");
      Type *PhiTy = P->getType();
      Value *Broadcasted;
      if (P == OldInduction) {
        // Handle the canonical induction variable. We might have had to
        // extend the type.
        Broadcasted = Builder.CreateTrunc(Induction, PhiTy);
      } else {
        // Handle other induction variables that are now based on the
        // canonical one.
        Value *NormalizedIdx = Builder.CreateSub(Induction, ExtendedIdx,
                                                 "normalized.idx");
        NormalizedIdx = Builder.CreateSExtOrTrunc(NormalizedIdx, PhiTy);
        Broadcasted = Builder.CreateAdd(II.StartValue, NormalizedIdx,
                                        "offset.idx");
      }
      Broadcasted = getBroadcastInstrs(Broadcasted);
      // After broadcasting the induction variable we need to make the vector
      // consecutive by adding 0, 1, 2, etc.
      for (unsigned part = 0; part < UF; ++part)
        Entry[part] = getConsecutiveVector(Broadcasted, VF * part, false);
      return;
    }
    case LoopVectorizationLegality::IK_ReverseIntInduction:
    case LoopVectorizationLegality::IK_PtrInduction:
    case LoopVectorizationLegality::IK_ReversePtrInduction:
      // Handle reverse integer and pointer inductions.
      Value *StartIdx = ExtendedIdx;
      // This is the normalized GEP that starts counting at zero.
      Value *NormalizedIdx = Builder.CreateSub(Induction, StartIdx,
                                               "normalized.idx");

      // Handle the reverse integer induction variable case.
      if (LoopVectorizationLegality::IK_ReverseIntInduction == II.IK) {
        IntegerType *DstTy = cast<IntegerType>(II.StartValue->getType());
        Value *CNI = Builder.CreateSExtOrTrunc(NormalizedIdx, DstTy,
                                               "resize.norm.idx");
        Value *ReverseInd  = Builder.CreateSub(II.StartValue, CNI,
                                               "reverse.idx");

        // This is a new value so do not hoist it out.
        Value *Broadcasted = getBroadcastInstrs(ReverseInd);
        // After broadcasting the induction variable we need to make the
        // vector consecutive by adding  ... -3, -2, -1, 0.
        for (unsigned part = 0; part < UF; ++part)
          Entry[part] = getConsecutiveVector(Broadcasted, -(int)VF * part,
                                             true);
        return;
      }

      // Handle the pointer induction variable case.
      assert(P->getType()->isPointerTy() && "Unexpected type.");

      // Is this a reverse induction ptr or a consecutive induction ptr.
      bool Reverse = (LoopVectorizationLegality::IK_ReversePtrInduction ==
                      II.IK);

      // This is the vector of results. Notice that we don't generate
      // vector geps because scalar geps result in better code.
      for (unsigned part = 0; part < UF; ++part) {
        if (VF == 1) {
          int EltIndex = (part) * (Reverse ? -1 : 1);
          Constant *Idx = ConstantInt::get(Induction->getType(), EltIndex);
          Value *GlobalIdx;
          if (Reverse)
            GlobalIdx = Builder.CreateSub(Idx, NormalizedIdx, "gep.ridx");
          else
            GlobalIdx = Builder.CreateAdd(NormalizedIdx, Idx, "gep.idx");

          Value *SclrGep = Builder.CreateGEP(II.StartValue, GlobalIdx,
                                             "next.gep");
          Entry[part] = SclrGep;
          continue;
        }

        Value *VecVal = UndefValue::get(VectorType::get(P->getType(), VF));
        for (unsigned int i = 0; i < VF; ++i) {
          int EltIndex = (i + part * VF) * (Reverse ? -1 : 1);
          Constant *Idx = ConstantInt::get(Induction->getType(), EltIndex);
          Value *GlobalIdx;
          if (!Reverse)
            GlobalIdx = Builder.CreateAdd(NormalizedIdx, Idx, "gep.idx");
          else
            GlobalIdx = Builder.CreateSub(Idx, NormalizedIdx, "gep.ridx");

          Value *SclrGep = Builder.CreateGEP(II.StartValue, GlobalIdx,
                                             "next.gep");
          VecVal = Builder.CreateInsertElement(VecVal, SclrGep,
                                               Builder.getInt32(i),
                                               "insert.gep");
        }
        Entry[part] = VecVal;
      }
      return;
  }
}

void
InnerLoopVectorizer::vectorizeBlockInLoop(LoopVectorizationLegality *Legal,
                                          BasicBlock *BB, PhiVector *PV) {
  // For each instruction in the old loop.
  for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
    VectorParts &Entry = WidenMap.get(it);
    switch (it->getOpcode()) {
    case Instruction::Br:
      // Nothing to do for PHIs and BR, since we already took care of the
      // loop control flow instructions.
      continue;
    case Instruction::PHI:{
      // Vectorize PHINodes.
      widenPHIInstruction(it, Entry, Legal, UF, VF, PV);
      continue;
    }// End of PHI.

    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      // Just widen binops.
      BinaryOperator *BinOp = dyn_cast<BinaryOperator>(it);
      setDebugLocFromInst(Builder, BinOp);
      VectorParts &A = getVectorValue(it->getOperand(0));
      VectorParts &B = getVectorValue(it->getOperand(1));

      // Use this vector value for all users of the original instruction.
      for (unsigned Part = 0; Part < UF; ++Part) {
        Value *V = Builder.CreateBinOp(BinOp->getOpcode(), A[Part], B[Part]);

        // Update the NSW, NUW and Exact flags. Notice: V can be an Undef.
        BinaryOperator *VecOp = dyn_cast<BinaryOperator>(V);
        if (VecOp && isa<OverflowingBinaryOperator>(BinOp)) {
          VecOp->setHasNoSignedWrap(BinOp->hasNoSignedWrap());
          VecOp->setHasNoUnsignedWrap(BinOp->hasNoUnsignedWrap());
        }
        if (VecOp && isa<PossiblyExactOperator>(VecOp))
          VecOp->setIsExact(BinOp->isExact());

        Entry[Part] = V;
      }
      break;
    }
    case Instruction::Select: {
      // Widen selects.
      // If the selector is loop invariant we can create a select
      // instruction with a scalar condition. Otherwise, use vector-select.
      bool InvariantCond = SE->isLoopInvariant(SE->getSCEV(it->getOperand(0)),
                                               OrigLoop);
      setDebugLocFromInst(Builder, it);

      // The condition can be loop invariant  but still defined inside the
      // loop. This means that we can't just use the original 'cond' value.
      // We have to take the 'vectorized' value and pick the first lane.
      // Instcombine will make this a no-op.
      VectorParts &Cond = getVectorValue(it->getOperand(0));
      VectorParts &Op0  = getVectorValue(it->getOperand(1));
      VectorParts &Op1  = getVectorValue(it->getOperand(2));

      Value *ScalarCond = (VF == 1) ? Cond[0] :
        Builder.CreateExtractElement(Cond[0], Builder.getInt32(0));

      for (unsigned Part = 0; Part < UF; ++Part) {
        Entry[Part] = Builder.CreateSelect(
          InvariantCond ? ScalarCond : Cond[Part],
          Op0[Part],
          Op1[Part]);
      }
      break;
    }

    case Instruction::ICmp:
    case Instruction::FCmp: {
      // Widen compares. Generate vector compares.
      bool FCmp = (it->getOpcode() == Instruction::FCmp);
      CmpInst *Cmp = dyn_cast<CmpInst>(it);
      setDebugLocFromInst(Builder, it);
      VectorParts &A = getVectorValue(it->getOperand(0));
      VectorParts &B = getVectorValue(it->getOperand(1));
      for (unsigned Part = 0; Part < UF; ++Part) {
        Value *C = 0;
        if (FCmp)
          C = Builder.CreateFCmp(Cmp->getPredicate(), A[Part], B[Part]);
        else
          C = Builder.CreateICmp(Cmp->getPredicate(), A[Part], B[Part]);
        Entry[Part] = C;
      }
      break;
    }

    case Instruction::Store:
    case Instruction::Load:
        vectorizeMemoryInstruction(it, Legal);
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
      CastInst *CI = dyn_cast<CastInst>(it);
      setDebugLocFromInst(Builder, it);
      /// Optimize the special case where the source is the induction
      /// variable. Notice that we can only optimize the 'trunc' case
      /// because: a. FP conversions lose precision, b. sext/zext may wrap,
      /// c. other casts depend on pointer size.
      if (CI->getOperand(0) == OldInduction &&
          it->getOpcode() == Instruction::Trunc) {
        Value *ScalarCast = Builder.CreateCast(CI->getOpcode(), Induction,
                                               CI->getType());
        Value *Broadcasted = getBroadcastInstrs(ScalarCast);
        for (unsigned Part = 0; Part < UF; ++Part)
          Entry[Part] = getConsecutiveVector(Broadcasted, VF * Part, false);
        break;
      }
      /// Vectorize casts.
      Type *DestTy = (VF == 1) ? CI->getType() :
                                 VectorType::get(CI->getType(), VF);

      VectorParts &A = getVectorValue(it->getOperand(0));
      for (unsigned Part = 0; Part < UF; ++Part)
        Entry[Part] = Builder.CreateCast(CI->getOpcode(), A[Part], DestTy);
      break;
    }

    case Instruction::Call: {
      // Ignore dbg intrinsics.
      if (isa<DbgInfoIntrinsic>(it))
        break;
      setDebugLocFromInst(Builder, it);

      Module *M = BB->getParent()->getParent();
      CallInst *CI = cast<CallInst>(it);
      Intrinsic::ID ID = getIntrinsicIDForCall(CI, TLI);
      assert(ID && "Not an intrinsic call!");
      switch (ID) {
      case Intrinsic::lifetime_end:
      case Intrinsic::lifetime_start:
        scalarizeInstruction(it);
        break;
      default:
        for (unsigned Part = 0; Part < UF; ++Part) {
          SmallVector<Value *, 4> Args;
          for (unsigned i = 0, ie = CI->getNumArgOperands(); i != ie; ++i) {
            VectorParts &Arg = getVectorValue(CI->getArgOperand(i));
            Args.push_back(Arg[Part]);
          }
          Type *Tys[] = {CI->getType()};
          if (VF > 1)
            Tys[0] = VectorType::get(CI->getType()->getScalarType(), VF);

          Function *F = Intrinsic::getDeclaration(M, ID, Tys);
          Entry[Part] = Builder.CreateCall(F, Args);
        }
        break;
      }
      break;
    }

    default:
      // All other instructions are unsupported. Scalarize them.
      scalarizeInstruction(it);
      break;
    }// end of switch.
  }// end of for_each instr.
}

void InnerLoopVectorizer::updateAnalysis() {
  // Forget the original basic block.
  SE->forgetLoop(OrigLoop);

  // Update the dominator tree information.
  assert(DT->properlyDominates(LoopBypassBlocks.front(), LoopExitBlock) &&
         "Entry does not dominate exit.");

  for (unsigned I = 1, E = LoopBypassBlocks.size(); I != E; ++I)
    DT->addNewBlock(LoopBypassBlocks[I], LoopBypassBlocks[I-1]);
  DT->addNewBlock(LoopVectorPreHeader, LoopBypassBlocks.back());
  DT->addNewBlock(LoopVectorBody, LoopVectorPreHeader);
  DT->addNewBlock(LoopMiddleBlock, LoopBypassBlocks.front());
  DT->addNewBlock(LoopScalarPreHeader, LoopMiddleBlock);
  DT->changeImmediateDominator(LoopScalarBody, LoopScalarPreHeader);
  DT->changeImmediateDominator(LoopExitBlock, LoopMiddleBlock);

  DEBUG(DT->verifyAnalysis());
}

/// \brief Check whether it is safe to if-convert this phi node.
///
/// Phi nodes with constant expressions that can trap are not safe to if
/// convert.
static bool canIfConvertPHINodes(BasicBlock *BB) {
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
    PHINode *Phi = dyn_cast<PHINode>(I);
    if (!Phi)
      return true;
    for (unsigned p = 0, e = Phi->getNumIncomingValues(); p != e; ++p)
      if (Constant *C = dyn_cast<Constant>(Phi->getIncomingValue(p)))
        if (C->canTrap())
          return false;
  }
  return true;
}

bool LoopVectorizationLegality::canVectorizeWithIfConvert() {
  if (!EnableIfConversion)
    return false;

  assert(TheLoop->getNumBlocks() > 1 && "Single block loops are vectorizable");

  // A list of pointers that we can safely read and write to.
  SmallPtrSet<Value *, 8> SafePointes;

  // Collect safe addresses.
  for (Loop::block_iterator BI = TheLoop->block_begin(),
         BE = TheLoop->block_end(); BI != BE; ++BI) {
    BasicBlock *BB = *BI;

    if (blockNeedsPredication(BB))
      continue;

    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      if (LoadInst *LI = dyn_cast<LoadInst>(I))
        SafePointes.insert(LI->getPointerOperand());
      else if (StoreInst *SI = dyn_cast<StoreInst>(I))
        SafePointes.insert(SI->getPointerOperand());
    }
  }

  // Collect the blocks that need predication.
  BasicBlock *Header = TheLoop->getHeader();
  for (Loop::block_iterator BI = TheLoop->block_begin(),
         BE = TheLoop->block_end(); BI != BE; ++BI) {
    BasicBlock *BB = *BI;

    // We don't support switch statements inside loops.
    if (!isa<BranchInst>(BB->getTerminator()))
      return false;

    // We must be able to predicate all blocks that need to be predicated.
    if (blockNeedsPredication(BB)) {
      if (!blockCanBePredicated(BB, SafePointes))
        return false;
    } else if (BB != Header && !canIfConvertPHINodes(BB))
      return false;

  }

  // We can if-convert this loop.
  return true;
}

bool LoopVectorizationLegality::canVectorize() {
  // We must have a loop in canonical form. Loops with indirectbr in them cannot
  // be canonicalized.
  if (!TheLoop->getLoopPreheader())
    return false;

  // We can only vectorize innermost loops.
  if (TheLoop->getSubLoopsVector().size())
    return false;

  // We must have a single backedge.
  if (TheLoop->getNumBackEdges() != 1)
    return false;

  // We must have a single exiting block.
  if (!TheLoop->getExitingBlock())
    return false;

  // We need to have a loop header.
  DEBUG(dbgs() << "LV: Found a loop: " <<
        TheLoop->getHeader()->getName() << '\n');

  // Check if we can if-convert non single-bb loops.
  unsigned NumBlocks = TheLoop->getNumBlocks();
  if (NumBlocks != 1 && !canVectorizeWithIfConvert()) {
    DEBUG(dbgs() << "LV: Can't if-convert the loop.\n");
    return false;
  }

  // ScalarEvolution needs to be able to find the exit count.
  const SCEV *ExitCount = SE->getBackedgeTakenCount(TheLoop);
  if (ExitCount == SE->getCouldNotCompute()) {
    DEBUG(dbgs() << "LV: SCEV could not compute the loop exit count.\n");
    return false;
  }

  // Do not loop-vectorize loops with a tiny trip count.
  BasicBlock *Latch = TheLoop->getLoopLatch();
  unsigned TC = SE->getSmallConstantTripCount(TheLoop, Latch);
  if (TC > 0u && TC < TinyTripCountVectorThreshold) {
    DEBUG(dbgs() << "LV: Found a loop with a very small trip count. " <<
          "This loop is not worth vectorizing.\n");
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

  // Collect all of the variables that remain uniform after vectorization.
  collectLoopUniforms();

  DEBUG(dbgs() << "LV: We can vectorize this loop" <<
        (PtrRtCheck.Need ? " (with a runtime bound check)" : "")
        <<"!\n");

  // Okay! We can vectorize. At this point we don't have any other mem analysis
  // which may limit our maximum vectorization factor, so just return true with
  // no restrictions.
  return true;
}

static Type *convertPointerToIntegerType(DataLayout &DL, Type *Ty) {
  if (Ty->isPointerTy())
    return DL.getIntPtrType(Ty);

  // It is possible that char's or short's overflow when we ask for the loop's
  // trip count, work around this by changing the type size.
  if (Ty->getScalarSizeInBits() < 32)
    return Type::getInt32Ty(Ty->getContext());

  return Ty;
}

static Type* getWiderType(DataLayout &DL, Type *Ty0, Type *Ty1) {
  Ty0 = convertPointerToIntegerType(DL, Ty0);
  Ty1 = convertPointerToIntegerType(DL, Ty1);
  if (Ty0->getScalarSizeInBits() > Ty1->getScalarSizeInBits())
    return Ty0;
  return Ty1;
}

/// \brief Check that the instruction has outside loop users and is not an
/// identified reduction variable.
static bool hasOutsideLoopUser(const Loop *TheLoop, Instruction *Inst,
                               SmallPtrSet<Value *, 4> &Reductions) {
  // Reduction instructions are allowed to have exit users. All other
  // instructions must not have external users.
  if (!Reductions.count(Inst))
    //Check that all of the users of the loop are inside the BB.
    for (Value::use_iterator I = Inst->use_begin(), E = Inst->use_end();
         I != E; ++I) {
      Instruction *U = cast<Instruction>(*I);
      // This user may be a reduction exit value.
      if (!TheLoop->contains(U)) {
        DEBUG(dbgs() << "LV: Found an outside user for : " << *U << '\n');
        return true;
      }
    }
  return false;
}

bool LoopVectorizationLegality::canVectorizeInstrs() {
  BasicBlock *PreHeader = TheLoop->getLoopPreheader();
  BasicBlock *Header = TheLoop->getHeader();

  // Look for the attribute signaling the absence of NaNs.
  Function &F = *Header->getParent();
  if (F.hasFnAttribute("no-nans-fp-math"))
    HasFunNoNaNAttr = F.getAttributes().getAttribute(
      AttributeSet::FunctionIndex,
      "no-nans-fp-math").getValueAsString() == "true";

  // For each block in the loop.
  for (Loop::block_iterator bb = TheLoop->block_begin(),
       be = TheLoop->block_end(); bb != be; ++bb) {

    // Scan the instructions in the block and look for hazards.
    for (BasicBlock::iterator it = (*bb)->begin(), e = (*bb)->end(); it != e;
         ++it) {

      if (PHINode *Phi = dyn_cast<PHINode>(it)) {
        Type *PhiTy = Phi->getType();
        // Check that this PHI type is allowed.
        if (!PhiTy->isIntegerTy() &&
            !PhiTy->isFloatingPointTy() &&
            !PhiTy->isPointerTy()) {
          DEBUG(dbgs() << "LV: Found an non-int non-pointer PHI.\n");
          return false;
        }

        // If this PHINode is not in the header block, then we know that we
        // can convert it to select during if-conversion. No need to check if
        // the PHIs in this block are induction or reduction variables.
        if (*bb != Header) {
          // Check that this instruction has no outside users or is an
          // identified reduction value with an outside user.
          if(!hasOutsideLoopUser(TheLoop, it, AllowedExit))
            continue;
          return false;
        }

        // We only allow if-converted PHIs with more than two incoming values.
        if (Phi->getNumIncomingValues() != 2) {
          DEBUG(dbgs() << "LV: Found an invalid PHI.\n");
          return false;
        }

        // This is the value coming from the preheader.
        Value *StartValue = Phi->getIncomingValueForBlock(PreHeader);
        // Check if this is an induction variable.
        InductionKind IK = isInductionVariable(Phi);

        if (IK_NoInduction != IK) {
          // Get the widest type.
          if (!WidestIndTy)
            WidestIndTy = convertPointerToIntegerType(*DL, PhiTy);
          else
            WidestIndTy = getWiderType(*DL, PhiTy, WidestIndTy);

          // Int inductions are special because we only allow one IV.
          if (IK == IK_IntInduction) {
            // Use the phi node with the widest type as induction. Use the last
            // one if there are multiple (no good reason for doing this other
            // than it is expedient).
            if (!Induction || PhiTy == WidestIndTy)
              Induction = Phi;
          }

          DEBUG(dbgs() << "LV: Found an induction variable.\n");
          Inductions[Phi] = InductionInfo(StartValue, IK);

          // Until we explicitly handle the case of an induction variable with
          // an outside loop user we have to give up vectorizing this loop.
          if (hasOutsideLoopUser(TheLoop, it, AllowedExit))
            return false;

          continue;
        }

        if (AddReductionVar(Phi, RK_IntegerAdd)) {
          DEBUG(dbgs() << "LV: Found an ADD reduction PHI."<< *Phi <<"\n");
          continue;
        }
        if (AddReductionVar(Phi, RK_IntegerMult)) {
          DEBUG(dbgs() << "LV: Found a MUL reduction PHI."<< *Phi <<"\n");
          continue;
        }
        if (AddReductionVar(Phi, RK_IntegerOr)) {
          DEBUG(dbgs() << "LV: Found an OR reduction PHI."<< *Phi <<"\n");
          continue;
        }
        if (AddReductionVar(Phi, RK_IntegerAnd)) {
          DEBUG(dbgs() << "LV: Found an AND reduction PHI."<< *Phi <<"\n");
          continue;
        }
        if (AddReductionVar(Phi, RK_IntegerXor)) {
          DEBUG(dbgs() << "LV: Found a XOR reduction PHI."<< *Phi <<"\n");
          continue;
        }
        if (AddReductionVar(Phi, RK_IntegerMinMax)) {
          DEBUG(dbgs() << "LV: Found a MINMAX reduction PHI."<< *Phi <<"\n");
          continue;
        }
        if (AddReductionVar(Phi, RK_FloatMult)) {
          DEBUG(dbgs() << "LV: Found an FMult reduction PHI."<< *Phi <<"\n");
          continue;
        }
        if (AddReductionVar(Phi, RK_FloatAdd)) {
          DEBUG(dbgs() << "LV: Found an FAdd reduction PHI."<< *Phi <<"\n");
          continue;
        }
        if (AddReductionVar(Phi, RK_FloatMinMax)) {
          DEBUG(dbgs() << "LV: Found an float MINMAX reduction PHI."<< *Phi <<
                "\n");
          continue;
        }

        DEBUG(dbgs() << "LV: Found an unidentified PHI."<< *Phi <<"\n");
        return false;
      }// end of PHI handling

      // We still don't handle functions. However, we can ignore dbg intrinsic
      // calls and we do handle certain intrinsic and libm functions.
      CallInst *CI = dyn_cast<CallInst>(it);
      if (CI && !getIntrinsicIDForCall(CI, TLI) && !isa<DbgInfoIntrinsic>(CI)) {
        DEBUG(dbgs() << "LV: Found a call site.\n");
        return false;
      }

      // Check that the instruction return type is vectorizable.
      // Also, we can't vectorize extractelement instructions.
      if ((!VectorType::isValidElementType(it->getType()) &&
           !it->getType()->isVoidTy()) || isa<ExtractElementInst>(it)) {
        DEBUG(dbgs() << "LV: Found unvectorizable type.\n");
        return false;
      }

      // Check that the stored type is vectorizable.
      if (StoreInst *ST = dyn_cast<StoreInst>(it)) {
        Type *T = ST->getValueOperand()->getType();
        if (!VectorType::isValidElementType(T))
          return false;
      }

      // Reduction instructions are allowed to have exit users.
      // All other instructions must not have external users.
      if (hasOutsideLoopUser(TheLoop, it, AllowedExit))
        return false;

    } // next instr.

  }

  if (!Induction) {
    DEBUG(dbgs() << "LV: Did not find one integer induction var.\n");
    if (Inductions.empty())
      return false;
  }

  return true;
}

void LoopVectorizationLegality::collectLoopUniforms() {
  // We now know that the loop is vectorizable!
  // Collect variables that will remain uniform after vectorization.
  std::vector<Value*> Worklist;
  BasicBlock *Latch = TheLoop->getLoopLatch();

  // Start with the conditional branch and walk up the block.
  Worklist.push_back(Latch->getTerminator()->getOperand(0));

  while (Worklist.size()) {
    Instruction *I = dyn_cast<Instruction>(Worklist.back());
    Worklist.pop_back();

    // Look at instructions inside this loop.
    // Stop when reaching PHI nodes.
    // TODO: we need to follow values all over the loop, not only in this block.
    if (!I || !TheLoop->contains(I) || isa<PHINode>(I))
      continue;

    // This is a known uniform.
    Uniforms.insert(I);

    // Insert all operands.
    Worklist.insert(Worklist.end(), I->op_begin(), I->op_end());
  }
}

namespace {
/// \brief Analyses memory accesses in a loop.
///
/// Checks whether run time pointer checks are needed and builds sets for data
/// dependence checking.
class AccessAnalysis {
public:
  /// \brief Read or write access location.
  typedef PointerIntPair<Value *, 1, bool> MemAccessInfo;
  typedef SmallPtrSet<MemAccessInfo, 8> MemAccessInfoSet;

  /// \brief Set of potential dependent memory accesses.
  typedef EquivalenceClasses<MemAccessInfo> DepCandidates;

  AccessAnalysis(DataLayout *Dl, DepCandidates &DA) :
    DL(Dl), DepCands(DA), AreAllWritesIdentified(true),
    AreAllReadsIdentified(true), IsRTCheckNeeded(false) {}

  /// \brief Register a load  and whether it is only read from.
  void addLoad(Value *Ptr, bool IsReadOnly) {
    Accesses.insert(MemAccessInfo(Ptr, false));
    if (IsReadOnly)
      ReadOnlyPtr.insert(Ptr);
  }

  /// \brief Register a store.
  void addStore(Value *Ptr) {
    Accesses.insert(MemAccessInfo(Ptr, true));
  }

  /// \brief Check whether we can check the pointers at runtime for
  /// non-intersection.
  bool canCheckPtrAtRT(LoopVectorizationLegality::RuntimePointerCheck &RtCheck,
                       unsigned &NumComparisons, ScalarEvolution *SE,
                       Loop *TheLoop, bool ShouldCheckStride = false);

  /// \brief Goes over all memory accesses, checks whether a RT check is needed
  /// and builds sets of dependent accesses.
  void buildDependenceSets() {
    // Process read-write pointers first.
    processMemAccesses(false);
    // Next, process read pointers.
    processMemAccesses(true);
  }

  bool isRTCheckNeeded() { return IsRTCheckNeeded; }

  bool isDependencyCheckNeeded() { return !CheckDeps.empty(); }
  void resetDepChecks() { CheckDeps.clear(); }

  MemAccessInfoSet &getDependenciesToCheck() { return CheckDeps; }

private:
  typedef SetVector<MemAccessInfo> PtrAccessSet;
  typedef DenseMap<Value*, MemAccessInfo> UnderlyingObjToAccessMap;

  /// \brief Go over all memory access or only the deferred ones if
  /// \p UseDeferred is true and check whether runtime pointer checks are needed
  /// and build sets of dependency check candidates.
  void processMemAccesses(bool UseDeferred);

  /// Set of all accesses.
  PtrAccessSet Accesses;

  /// Set of access to check after all writes have been processed.
  PtrAccessSet DeferredAccesses;

  /// Map of pointers to last access encountered.
  UnderlyingObjToAccessMap ObjToLastAccess;

  /// Set of accesses that need a further dependence check.
  MemAccessInfoSet CheckDeps;

  /// Set of pointers that are read only.
  SmallPtrSet<Value*, 16> ReadOnlyPtr;

  /// Set of underlying objects already written to.
  SmallPtrSet<Value*, 16> WriteObjects;

  DataLayout *DL;

  /// Sets of potentially dependent accesses - members of one set share an
  /// underlying pointer. The set "CheckDeps" identfies which sets really need a
  /// dependence check.
  DepCandidates &DepCands;

  bool AreAllWritesIdentified;
  bool AreAllReadsIdentified;
  bool IsRTCheckNeeded;
};

} // end anonymous namespace

/// \brief Check whether a pointer can participate in a runtime bounds check.
static bool hasComputableBounds(ScalarEvolution *SE, Value *Ptr) {
  const SCEV *PtrScev = SE->getSCEV(Ptr);
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PtrScev);
  if (!AR)
    return false;

  return AR->isAffine();
}

/// \brief Check the stride of the pointer and ensure that it does not wrap in
/// the address space.
static int isStridedPtr(ScalarEvolution *SE, DataLayout *DL, Value *Ptr,
                        const Loop *Lp);

bool AccessAnalysis::canCheckPtrAtRT(
                       LoopVectorizationLegality::RuntimePointerCheck &RtCheck,
                        unsigned &NumComparisons, ScalarEvolution *SE,
                        Loop *TheLoop, bool ShouldCheckStride) {
  // Find pointers with computable bounds. We are going to use this information
  // to place a runtime bound check.
  unsigned NumReadPtrChecks = 0;
  unsigned NumWritePtrChecks = 0;
  bool CanDoRT = true;

  bool IsDepCheckNeeded = isDependencyCheckNeeded();
  // We assign consecutive id to access from different dependence sets.
  // Accesses within the same set don't need a runtime check.
  unsigned RunningDepId = 1;
  DenseMap<Value *, unsigned> DepSetId;

  for (PtrAccessSet::iterator AI = Accesses.begin(), AE = Accesses.end();
       AI != AE; ++AI) {
    const MemAccessInfo &Access = *AI;
    Value *Ptr = Access.getPointer();
    bool IsWrite = Access.getInt();

    // Just add write checks if we have both.
    if (!IsWrite && Accesses.count(MemAccessInfo(Ptr, true)))
      continue;

    if (IsWrite)
      ++NumWritePtrChecks;
    else
      ++NumReadPtrChecks;

    if (hasComputableBounds(SE, Ptr) &&
        // When we run after a failing dependency check we have to make sure we
        // don't have wrapping pointers.
        (!ShouldCheckStride || isStridedPtr(SE, DL, Ptr, TheLoop) == 1)) {
      // The id of the dependence set.
      unsigned DepId;

      if (IsDepCheckNeeded) {
        Value *Leader = DepCands.getLeaderValue(Access).getPointer();
        unsigned &LeaderId = DepSetId[Leader];
        if (!LeaderId)
          LeaderId = RunningDepId++;
        DepId = LeaderId;
      } else
        // Each access has its own dependence set.
        DepId = RunningDepId++;

      RtCheck.insert(SE, TheLoop, Ptr, IsWrite, DepId);

      DEBUG(dbgs() << "LV: Found a runtime check ptr:" << *Ptr << '\n');
    } else {
      CanDoRT = false;
    }
  }

  if (IsDepCheckNeeded && CanDoRT && RunningDepId == 2)
    NumComparisons = 0; // Only one dependence set.
  else {
    NumComparisons = (NumWritePtrChecks * (NumReadPtrChecks +
                                           NumWritePtrChecks - 1));
  }

  // If the pointers that we would use for the bounds comparison have different
  // address spaces, assume the values aren't directly comparable, so we can't
  // use them for the runtime check. We also have to assume they could
  // overlap. In the future there should be metadata for whether address spaces
  // are disjoint.
  unsigned NumPointers = RtCheck.Pointers.size();
  for (unsigned i = 0; i < NumPointers; ++i) {
    for (unsigned j = i + 1; j < NumPointers; ++j) {
      // Only need to check pointers between two different dependency sets.
      if (RtCheck.DependencySetId[i] == RtCheck.DependencySetId[j])
       continue;

      Value *PtrI = RtCheck.Pointers[i];
      Value *PtrJ = RtCheck.Pointers[j];

      unsigned ASi = PtrI->getType()->getPointerAddressSpace();
      unsigned ASj = PtrJ->getType()->getPointerAddressSpace();
      if (ASi != ASj) {
        DEBUG(dbgs() << "LV: Runtime check would require comparison between"
                       " different address spaces\n");
        return false;
      }
    }
  }

  return CanDoRT;
}

static bool isFunctionScopeIdentifiedObject(Value *Ptr) {
  return isNoAliasArgument(Ptr) || isNoAliasCall(Ptr) || isa<AllocaInst>(Ptr);
}

void AccessAnalysis::processMemAccesses(bool UseDeferred) {
  // We process the set twice: first we process read-write pointers, last we
  // process read-only pointers. This allows us to skip dependence tests for
  // read-only pointers.

  PtrAccessSet &S = UseDeferred ? DeferredAccesses : Accesses;
  for (PtrAccessSet::iterator AI = S.begin(), AE = S.end(); AI != AE; ++AI) {
    const MemAccessInfo &Access = *AI;
    Value *Ptr = Access.getPointer();
    bool IsWrite = Access.getInt();

    DepCands.insert(Access);

    // Memorize read-only pointers for later processing and skip them in the
    // first round (they need to be checked after we have seen all write
    // pointers). Note: we also mark pointer that are not consecutive as
    // "read-only" pointers (so that we check "a[b[i]] +="). Hence, we need the
    // second check for "!IsWrite".
    bool IsReadOnlyPtr = ReadOnlyPtr.count(Ptr) && !IsWrite;
    if (!UseDeferred && IsReadOnlyPtr) {
      DeferredAccesses.insert(Access);
      continue;
    }

    bool NeedDepCheck = false;
    // Check whether there is the possiblity of dependency because of underlying
    // objects being the same.
    typedef SmallVector<Value*, 16> ValueVector;
    ValueVector TempObjects;
    GetUnderlyingObjects(Ptr, TempObjects, DL);
    for (ValueVector::iterator UI = TempObjects.begin(), UE = TempObjects.end();
         UI != UE; ++UI) {
      Value *UnderlyingObj = *UI;

      // If this is a write then it needs to be an identified object.  If this a
      // read and all writes (so far) are identified function scope objects we
      // don't need an identified underlying object but only an Argument (the
      // next write is going to invalidate this assumption if it is
      // unidentified).
      // This is a micro-optimization for the case where all writes are
      // identified and we have one argument pointer.
      // Otherwise, we do need a runtime check.
      if ((IsWrite && !isFunctionScopeIdentifiedObject(UnderlyingObj)) ||
          (!IsWrite && (!AreAllWritesIdentified ||
                        !isa<Argument>(UnderlyingObj)) &&
           !isIdentifiedObject(UnderlyingObj))) {
        DEBUG(dbgs() << "LV: Found an unidentified " <<
              (IsWrite ?  "write" : "read" ) << " ptr: " << *UnderlyingObj <<
              "\n");
        IsRTCheckNeeded = (IsRTCheckNeeded ||
                           !isIdentifiedObject(UnderlyingObj) ||
                           !AreAllReadsIdentified);

        if (IsWrite)
          AreAllWritesIdentified = false;
        if (!IsWrite)
          AreAllReadsIdentified = false;
      }

      // If this is a write - check other reads and writes for conflicts.  If
      // this is a read only check other writes for conflicts (but only if there
      // is no other write to the ptr - this is an optimization to catch "a[i] =
      // a[i] + " without having to do a dependence check).
      if ((IsWrite || IsReadOnlyPtr) && WriteObjects.count(UnderlyingObj))
        NeedDepCheck = true;

      if (IsWrite)
        WriteObjects.insert(UnderlyingObj);

      // Create sets of pointers connected by shared underlying objects.
      UnderlyingObjToAccessMap::iterator Prev =
        ObjToLastAccess.find(UnderlyingObj);
      if (Prev != ObjToLastAccess.end())
        DepCands.unionSets(Access, Prev->second);

      ObjToLastAccess[UnderlyingObj] = Access;
    }

    if (NeedDepCheck)
      CheckDeps.insert(Access);
  }
}

namespace {
/// \brief Checks memory dependences among accesses to the same underlying
/// object to determine whether there vectorization is legal or not (and at
/// which vectorization factor).
///
/// This class works under the assumption that we already checked that memory
/// locations with different underlying pointers are "must-not alias".
/// We use the ScalarEvolution framework to symbolically evalutate access
/// functions pairs. Since we currently don't restructure the loop we can rely
/// on the program order of memory accesses to determine their safety.
/// At the moment we will only deem accesses as safe for:
///  * A negative constant distance assuming program order.
///
///      Safe: tmp = a[i + 1];     OR     a[i + 1] = x;
///            a[i] = tmp;                y = a[i];
///
///   The latter case is safe because later checks guarantuee that there can't
///   be a cycle through a phi node (that is, we check that "x" and "y" is not
///   the same variable: a header phi can only be an induction or a reduction, a
///   reduction can't have a memory sink, an induction can't have a memory
///   source). This is important and must not be violated (or we have to
///   resort to checking for cycles through memory).
///
///  * A positive constant distance assuming program order that is bigger
///    than the biggest memory access.
///
///     tmp = a[i]        OR              b[i] = x
///     a[i+2] = tmp                      y = b[i+2];
///
///     Safe distance: 2 x sizeof(a[0]), and 2 x sizeof(b[0]), respectively.
///
///  * Zero distances and all accesses have the same size.
///
class MemoryDepChecker {
public:
  typedef PointerIntPair<Value *, 1, bool> MemAccessInfo;
  typedef SmallPtrSet<MemAccessInfo, 8> MemAccessInfoSet;

  MemoryDepChecker(ScalarEvolution *Se, DataLayout *Dl, const Loop *L)
      : SE(Se), DL(Dl), InnermostLoop(L), AccessIdx(0),
        ShouldRetryWithRuntimeCheck(false) {}

  /// \brief Register the location (instructions are given increasing numbers)
  /// of a write access.
  void addAccess(StoreInst *SI) {
    Value *Ptr = SI->getPointerOperand();
    Accesses[MemAccessInfo(Ptr, true)].push_back(AccessIdx);
    InstMap.push_back(SI);
    ++AccessIdx;
  }

  /// \brief Register the location (instructions are given increasing numbers)
  /// of a write access.
  void addAccess(LoadInst *LI) {
    Value *Ptr = LI->getPointerOperand();
    Accesses[MemAccessInfo(Ptr, false)].push_back(AccessIdx);
    InstMap.push_back(LI);
    ++AccessIdx;
  }

  /// \brief Check whether the dependencies between the accesses are safe.
  ///
  /// Only checks sets with elements in \p CheckDeps.
  bool areDepsSafe(AccessAnalysis::DepCandidates &AccessSets,
                   MemAccessInfoSet &CheckDeps);

  /// \brief The maximum number of bytes of a vector register we can vectorize
  /// the accesses safely with.
  unsigned getMaxSafeDepDistBytes() { return MaxSafeDepDistBytes; }

  /// \brief In same cases when the dependency check fails we can still
  /// vectorize the loop with a dynamic array access check.
  bool shouldRetryWithRuntimeCheck() { return ShouldRetryWithRuntimeCheck; }

private:
  ScalarEvolution *SE;
  DataLayout *DL;
  const Loop *InnermostLoop;

  /// \brief Maps access locations (ptr, read/write) to program order.
  DenseMap<MemAccessInfo, std::vector<unsigned> > Accesses;

  /// \brief Memory access instructions in program order.
  SmallVector<Instruction *, 16> InstMap;

  /// \brief The program order index to be used for the next instruction.
  unsigned AccessIdx;

  // We can access this many bytes in parallel safely.
  unsigned MaxSafeDepDistBytes;

  /// \brief If we see a non constant dependence distance we can still try to
  /// vectorize this loop with runtime checks.
  bool ShouldRetryWithRuntimeCheck;

  /// \brief Check whether there is a plausible dependence between the two
  /// accesses.
  ///
  /// Access \p A must happen before \p B in program order. The two indices
  /// identify the index into the program order map.
  ///
  /// This function checks  whether there is a plausible dependence (or the
  /// absence of such can't be proved) between the two accesses. If there is a
  /// plausible dependence but the dependence distance is bigger than one
  /// element access it records this distance in \p MaxSafeDepDistBytes (if this
  /// distance is smaller than any other distance encountered so far).
  /// Otherwise, this function returns true signaling a possible dependence.
  bool isDependent(const MemAccessInfo &A, unsigned AIdx,
                   const MemAccessInfo &B, unsigned BIdx);

  /// \brief Check whether the data dependence could prevent store-load
  /// forwarding.
  bool couldPreventStoreLoadForward(unsigned Distance, unsigned TypeByteSize);
};

} // end anonymous namespace

static bool isInBoundsGep(Value *Ptr) {
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr))
    return GEP->isInBounds();
  return false;
}

/// \brief Check whether the access through \p Ptr has a constant stride.
static int isStridedPtr(ScalarEvolution *SE, DataLayout *DL, Value *Ptr,
                        const Loop *Lp) {
  const Type *Ty = Ptr->getType();
  assert(Ty->isPointerTy() && "Unexpected non ptr");

  // Make sure that the pointer does not point to aggregate types.
  const PointerType *PtrTy = cast<PointerType>(Ty);
  if (PtrTy->getElementType()->isAggregateType()) {
    DEBUG(dbgs() << "LV: Bad stride - Not a pointer to a scalar type" << *Ptr <<
          "\n");
    return 0;
  }

  const SCEV *PtrScev = SE->getSCEV(Ptr);
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PtrScev);
  if (!AR) {
    DEBUG(dbgs() << "LV: Bad stride - Not an AddRecExpr pointer "
          << *Ptr << " SCEV: " << *PtrScev << "\n");
    return 0;
  }

  // The accesss function must stride over the innermost loop.
  if (Lp != AR->getLoop()) {
    DEBUG(dbgs() << "LV: Bad stride - Not striding over innermost loop " <<
          *Ptr << " SCEV: " << *PtrScev << "\n");
  }

  // The address calculation must not wrap. Otherwise, a dependence could be
  // inverted.
  // An inbounds getelementptr that is a AddRec with a unit stride
  // cannot wrap per definition. The unit stride requirement is checked later.
  // An getelementptr without an inbounds attribute and unit stride would have
  // to access the pointer value "0" which is undefined behavior in address
  // space 0, therefore we can also vectorize this case.
  bool IsInBoundsGEP = isInBoundsGep(Ptr);
  bool IsNoWrapAddRec = AR->getNoWrapFlags(SCEV::NoWrapMask);
  bool IsInAddressSpaceZero = PtrTy->getAddressSpace() == 0;
  if (!IsNoWrapAddRec && !IsInBoundsGEP && !IsInAddressSpaceZero) {
    DEBUG(dbgs() << "LV: Bad stride - Pointer may wrap in the address space "
          << *Ptr << " SCEV: " << *PtrScev << "\n");
    return 0;
  }

  // Check the step is constant.
  const SCEV *Step = AR->getStepRecurrence(*SE);

  // Calculate the pointer stride and check if it is consecutive.
  const SCEVConstant *C = dyn_cast<SCEVConstant>(Step);
  if (!C) {
    DEBUG(dbgs() << "LV: Bad stride - Not a constant strided " << *Ptr <<
          " SCEV: " << *PtrScev << "\n");
    return 0;
  }

  int64_t Size = DL->getTypeAllocSize(PtrTy->getElementType());
  const APInt &APStepVal = C->getValue()->getValue();

  // Huge step value - give up.
  if (APStepVal.getBitWidth() > 64)
    return 0;

  int64_t StepVal = APStepVal.getSExtValue();

  // Strided access.
  int64_t Stride = StepVal / Size;
  int64_t Rem = StepVal % Size;
  if (Rem)
    return 0;

  // If the SCEV could wrap but we have an inbounds gep with a unit stride we
  // know we can't "wrap around the address space". In case of address space
  // zero we know that this won't happen without triggering undefined behavior.
  if (!IsNoWrapAddRec && (IsInBoundsGEP || IsInAddressSpaceZero) &&
      Stride != 1 && Stride != -1)
    return 0;

  return Stride;
}

bool MemoryDepChecker::couldPreventStoreLoadForward(unsigned Distance,
                                                    unsigned TypeByteSize) {
  // If loads occur at a distance that is not a multiple of a feasible vector
  // factor store-load forwarding does not take place.
  // Positive dependences might cause troubles because vectorizing them might
  // prevent store-load forwarding making vectorized code run a lot slower.
  //   a[i] = a[i-3] ^ a[i-8];
  //   The stores to a[i:i+1] don't align with the stores to a[i-3:i-2] and
  //   hence on your typical architecture store-load forwarding does not take
  //   place. Vectorizing in such cases does not make sense.
  // Store-load forwarding distance.
  const unsigned NumCyclesForStoreLoadThroughMemory = 8*TypeByteSize;
  // Maximum vector factor.
  unsigned MaxVFWithoutSLForwardIssues = MaxVectorWidth*TypeByteSize;
  if(MaxSafeDepDistBytes < MaxVFWithoutSLForwardIssues)
    MaxVFWithoutSLForwardIssues = MaxSafeDepDistBytes;

  for (unsigned vf = 2*TypeByteSize; vf <= MaxVFWithoutSLForwardIssues;
       vf *= 2) {
    if (Distance % vf && Distance / vf < NumCyclesForStoreLoadThroughMemory) {
      MaxVFWithoutSLForwardIssues = (vf >>=1);
      break;
    }
  }

  if (MaxVFWithoutSLForwardIssues< 2*TypeByteSize) {
    DEBUG(dbgs() << "LV: Distance " << Distance <<
          " that could cause a store-load forwarding conflict\n");
    return true;
  }

  if (MaxVFWithoutSLForwardIssues < MaxSafeDepDistBytes &&
      MaxVFWithoutSLForwardIssues != MaxVectorWidth*TypeByteSize)
    MaxSafeDepDistBytes = MaxVFWithoutSLForwardIssues;
  return false;
}

bool MemoryDepChecker::isDependent(const MemAccessInfo &A, unsigned AIdx,
                                   const MemAccessInfo &B, unsigned BIdx) {
  assert (AIdx < BIdx && "Must pass arguments in program order");

  Value *APtr = A.getPointer();
  Value *BPtr = B.getPointer();
  bool AIsWrite = A.getInt();
  bool BIsWrite = B.getInt();

  // Two reads are independent.
  if (!AIsWrite && !BIsWrite)
    return false;

  const SCEV *AScev = SE->getSCEV(APtr);
  const SCEV *BScev = SE->getSCEV(BPtr);

  int StrideAPtr = isStridedPtr(SE, DL, APtr, InnermostLoop);
  int StrideBPtr = isStridedPtr(SE, DL, BPtr, InnermostLoop);

  const SCEV *Src = AScev;
  const SCEV *Sink = BScev;

  // If the induction step is negative we have to invert source and sink of the
  // dependence.
  if (StrideAPtr < 0) {
    //Src = BScev;
    //Sink = AScev;
    std::swap(APtr, BPtr);
    std::swap(Src, Sink);
    std::swap(AIsWrite, BIsWrite);
    std::swap(AIdx, BIdx);
    std::swap(StrideAPtr, StrideBPtr);
  }

  const SCEV *Dist = SE->getMinusSCEV(Sink, Src);

  DEBUG(dbgs() << "LV: Src Scev: " << *Src << "Sink Scev: " << *Sink
        << "(Induction step: " << StrideAPtr <<  ")\n");
  DEBUG(dbgs() << "LV: Distance for " << *InstMap[AIdx] << " to "
        << *InstMap[BIdx] << ": " << *Dist << "\n");

  // Need consecutive accesses. We don't want to vectorize
  // "A[B[i]] += ..." and similar code or pointer arithmetic that could wrap in
  // the address space.
  if (!StrideAPtr || !StrideBPtr || StrideAPtr != StrideBPtr){
    DEBUG(dbgs() << "Non-consecutive pointer access\n");
    return true;
  }

  const SCEVConstant *C = dyn_cast<SCEVConstant>(Dist);
  if (!C) {
    DEBUG(dbgs() << "LV: Dependence because of non constant distance\n");
    ShouldRetryWithRuntimeCheck = true;
    return true;
  }

  Type *ATy = APtr->getType()->getPointerElementType();
  Type *BTy = BPtr->getType()->getPointerElementType();
  unsigned TypeByteSize = DL->getTypeAllocSize(ATy);

  // Negative distances are not plausible dependencies.
  const APInt &Val = C->getValue()->getValue();
  if (Val.isNegative()) {
    bool IsTrueDataDependence = (AIsWrite && !BIsWrite);
    if (IsTrueDataDependence &&
        (couldPreventStoreLoadForward(Val.abs().getZExtValue(), TypeByteSize) ||
         ATy != BTy))
      return true;

    DEBUG(dbgs() << "LV: Dependence is negative: NoDep\n");
    return false;
  }

  // Write to the same location with the same size.
  // Could be improved to assert type sizes are the same (i32 == float, etc).
  if (Val == 0) {
    if (ATy == BTy)
      return false;
    DEBUG(dbgs() << "LV: Zero dependence difference but different types\n");
    return true;
  }

  assert(Val.isStrictlyPositive() && "Expect a positive value");

  // Positive distance bigger than max vectorization factor.
  if (ATy != BTy) {
    DEBUG(dbgs() <<
          "LV: ReadWrite-Write positive dependency with different types\n");
    return false;
  }

  unsigned Distance = (unsigned) Val.getZExtValue();

  // Bail out early if passed-in parameters make vectorization not feasible.
  unsigned ForcedFactor = VectorizationFactor ? VectorizationFactor : 1;
  unsigned ForcedUnroll = VectorizationUnroll ? VectorizationUnroll : 1;

  // The distance must be bigger than the size needed for a vectorized version
  // of the operation and the size of the vectorized operation must not be
  // bigger than the currrent maximum size.
  if (Distance < 2*TypeByteSize ||
      2*TypeByteSize > MaxSafeDepDistBytes ||
      Distance < TypeByteSize * ForcedUnroll * ForcedFactor) {
    DEBUG(dbgs() << "LV: Failure because of Positive distance "
        << Val.getSExtValue() << '\n');
    return true;
  }

  MaxSafeDepDistBytes = Distance < MaxSafeDepDistBytes ?
    Distance : MaxSafeDepDistBytes;

  bool IsTrueDataDependence = (!AIsWrite && BIsWrite);
  if (IsTrueDataDependence &&
      couldPreventStoreLoadForward(Distance, TypeByteSize))
     return true;

  DEBUG(dbgs() << "LV: Positive distance " << Val.getSExtValue() <<
        " with max VF = " << MaxSafeDepDistBytes / TypeByteSize << '\n');

  return false;
}

bool
MemoryDepChecker::areDepsSafe(AccessAnalysis::DepCandidates &AccessSets,
                              MemAccessInfoSet &CheckDeps) {

  MaxSafeDepDistBytes = -1U;
  while (!CheckDeps.empty()) {
    MemAccessInfo CurAccess = *CheckDeps.begin();

    // Get the relevant memory access set.
    EquivalenceClasses<MemAccessInfo>::iterator I =
      AccessSets.findValue(AccessSets.getLeaderValue(CurAccess));

    // Check accesses within this set.
    EquivalenceClasses<MemAccessInfo>::member_iterator AI, AE;
    AI = AccessSets.member_begin(I), AE = AccessSets.member_end();

    // Check every access pair.
    while (AI != AE) {
      CheckDeps.erase(*AI);
      EquivalenceClasses<MemAccessInfo>::member_iterator OI = llvm::next(AI);
      while (OI != AE) {
        // Check every accessing instruction pair in program order.
        for (std::vector<unsigned>::iterator I1 = Accesses[*AI].begin(),
             I1E = Accesses[*AI].end(); I1 != I1E; ++I1)
          for (std::vector<unsigned>::iterator I2 = Accesses[*OI].begin(),
               I2E = Accesses[*OI].end(); I2 != I2E; ++I2) {
            if (*I1 < *I2 && isDependent(*AI, *I1, *OI, *I2))
              return false;
            if (*I2 < *I1 && isDependent(*OI, *I2, *AI, *I1))
              return false;
          }
        ++OI;
      }
      AI++;
    }
  }
  return true;
}

bool LoopVectorizationLegality::canVectorizeMemory() {

  typedef SmallVector<Value*, 16> ValueVector;
  typedef SmallPtrSet<Value*, 16> ValueSet;

  // Holds the Load and Store *instructions*.
  ValueVector Loads;
  ValueVector Stores;

  // Holds all the different accesses in the loop.
  unsigned NumReads = 0;
  unsigned NumReadWrites = 0;

  PtrRtCheck.Pointers.clear();
  PtrRtCheck.Need = false;

  const bool IsAnnotatedParallel = TheLoop->isAnnotatedParallel();
  MemoryDepChecker DepChecker(SE, DL, TheLoop);

  // For each block.
  for (Loop::block_iterator bb = TheLoop->block_begin(),
       be = TheLoop->block_end(); bb != be; ++bb) {

    // Scan the BB and collect legal loads and stores.
    for (BasicBlock::iterator it = (*bb)->begin(), e = (*bb)->end(); it != e;
         ++it) {

      // If this is a load, save it. If this instruction can read from memory
      // but is not a load, then we quit. Notice that we don't handle function
      // calls that read or write.
      if (it->mayReadFromMemory()) {
        // Many math library functions read the rounding mode. We will only
        // vectorize a loop if it contains known function calls that don't set
        // the flag. Therefore, it is safe to ignore this read from memory.
        CallInst *Call = dyn_cast<CallInst>(it);
        if (Call && getIntrinsicIDForCall(Call, TLI))
          continue;

        LoadInst *Ld = dyn_cast<LoadInst>(it);
        if (!Ld) return false;
        if (!Ld->isSimple() && !IsAnnotatedParallel) {
          DEBUG(dbgs() << "LV: Found a non-simple load.\n");
          return false;
        }
        Loads.push_back(Ld);
        DepChecker.addAccess(Ld);
        continue;
      }

      // Save 'store' instructions. Abort if other instructions write to memory.
      if (it->mayWriteToMemory()) {
        StoreInst *St = dyn_cast<StoreInst>(it);
        if (!St) return false;
        if (!St->isSimple() && !IsAnnotatedParallel) {
          DEBUG(dbgs() << "LV: Found a non-simple store.\n");
          return false;
        }
        Stores.push_back(St);
        DepChecker.addAccess(St);
      }
    } // Next instr.
  } // Next block.

  // Now we have two lists that hold the loads and the stores.
  // Next, we find the pointers that they use.

  // Check if we see any stores. If there are no stores, then we don't
  // care if the pointers are *restrict*.
  if (!Stores.size()) {
    DEBUG(dbgs() << "LV: Found a read-only loop!\n");
    return true;
  }

  AccessAnalysis::DepCandidates DependentAccesses;
  AccessAnalysis Accesses(DL, DependentAccesses);

  // Holds the analyzed pointers. We don't want to call GetUnderlyingObjects
  // multiple times on the same object. If the ptr is accessed twice, once
  // for read and once for write, it will only appear once (on the write
  // list). This is okay, since we are going to check for conflicts between
  // writes and between reads and writes, but not between reads and reads.
  ValueSet Seen;

  ValueVector::iterator I, IE;
  for (I = Stores.begin(), IE = Stores.end(); I != IE; ++I) {
    StoreInst *ST = cast<StoreInst>(*I);
    Value* Ptr = ST->getPointerOperand();

    if (isUniform(Ptr)) {
      DEBUG(dbgs() << "LV: We don't allow storing to uniform addresses\n");
      return false;
    }

    // If we did *not* see this pointer before, insert it to  the read-write
    // list. At this phase it is only a 'write' list.
    if (Seen.insert(Ptr)) {
      ++NumReadWrites;
      Accesses.addStore(Ptr);
    }
  }

  if (IsAnnotatedParallel) {
    DEBUG(dbgs()
          << "LV: A loop annotated parallel, ignore memory dependency "
          << "checks.\n");
    return true;
  }

  for (I = Loads.begin(), IE = Loads.end(); I != IE; ++I) {
    LoadInst *LD = cast<LoadInst>(*I);
    Value* Ptr = LD->getPointerOperand();
    // If we did *not* see this pointer before, insert it to the
    // read list. If we *did* see it before, then it is already in
    // the read-write list. This allows us to vectorize expressions
    // such as A[i] += x;  Because the address of A[i] is a read-write
    // pointer. This only works if the index of A[i] is consecutive.
    // If the address of i is unknown (for example A[B[i]]) then we may
    // read a few words, modify, and write a few words, and some of the
    // words may be written to the same address.
    bool IsReadOnlyPtr = false;
    if (Seen.insert(Ptr) || !isStridedPtr(SE, DL, Ptr, TheLoop)) {
      ++NumReads;
      IsReadOnlyPtr = true;
    }
    Accesses.addLoad(Ptr, IsReadOnlyPtr);
  }

  // If we write (or read-write) to a single destination and there are no
  // other reads in this loop then is it safe to vectorize.
  if (NumReadWrites == 1 && NumReads == 0) {
    DEBUG(dbgs() << "LV: Found a write-only loop!\n");
    return true;
  }

  // Build dependence sets and check whether we need a runtime pointer bounds
  // check.
  Accesses.buildDependenceSets();
  bool NeedRTCheck = Accesses.isRTCheckNeeded();

  // Find pointers with computable bounds. We are going to use this information
  // to place a runtime bound check.
  unsigned NumComparisons = 0;
  bool CanDoRT = false;
  if (NeedRTCheck)
    CanDoRT = Accesses.canCheckPtrAtRT(PtrRtCheck, NumComparisons, SE, TheLoop);


  DEBUG(dbgs() << "LV: We need to do " << NumComparisons <<
        " pointer comparisons.\n");

  // If we only have one set of dependences to check pointers among we don't
  // need a runtime check.
  if (NumComparisons == 0 && NeedRTCheck)
    NeedRTCheck = false;

  // Check that we did not collect too many pointers or found an unsizeable
  // pointer.
  if (!CanDoRT || NumComparisons > RuntimeMemoryCheckThreshold) {
    PtrRtCheck.reset();
    CanDoRT = false;
  }

  if (CanDoRT) {
    DEBUG(dbgs() << "LV: We can perform a memory runtime check if needed.\n");
  }

  if (NeedRTCheck && !CanDoRT) {
    DEBUG(dbgs() << "LV: We can't vectorize because we can't find " <<
          "the array bounds.\n");
    PtrRtCheck.reset();
    return false;
  }

  PtrRtCheck.Need = NeedRTCheck;

  bool CanVecMem = true;
  if (Accesses.isDependencyCheckNeeded()) {
    DEBUG(dbgs() << "LV: Checking memory dependencies\n");
    CanVecMem = DepChecker.areDepsSafe(DependentAccesses,
                                       Accesses.getDependenciesToCheck());
    MaxSafeDepDistBytes = DepChecker.getMaxSafeDepDistBytes();

    if (!CanVecMem && DepChecker.shouldRetryWithRuntimeCheck()) {
      DEBUG(dbgs() << "LV: Retrying with memory checks\n");
      NeedRTCheck = true;

      // Clear the dependency checks. We assume they are not needed.
      Accesses.resetDepChecks();

      PtrRtCheck.reset();
      PtrRtCheck.Need = true;

      CanDoRT = Accesses.canCheckPtrAtRT(PtrRtCheck, NumComparisons, SE,
                                         TheLoop, true);
      // Check that we did not collect too many pointers or found an unsizeable
      // pointer.
      if (!CanDoRT || NumComparisons > RuntimeMemoryCheckThreshold) {
        DEBUG(dbgs() << "LV: Can't vectorize with memory checks\n");
        PtrRtCheck.reset();
        return false;
      }

      CanVecMem = true;
    }
  }

  DEBUG(dbgs() << "LV: We" << (NeedRTCheck ? "" : " don't") <<
        " need a runtime memory check.\n");

  return CanVecMem;
}

static bool hasMultipleUsesOf(Instruction *I,
                              SmallPtrSet<Instruction *, 8> &Insts) {
  unsigned NumUses = 0;
  for(User::op_iterator Use = I->op_begin(), E = I->op_end(); Use != E; ++Use) {
    if (Insts.count(dyn_cast<Instruction>(*Use)))
      ++NumUses;
    if (NumUses > 1)
      return true;
  }

  return false;
}

static bool areAllUsesIn(Instruction *I, SmallPtrSet<Instruction *, 8> &Set) {
  for(User::op_iterator Use = I->op_begin(), E = I->op_end(); Use != E; ++Use)
    if (!Set.count(dyn_cast<Instruction>(*Use)))
      return false;
  return true;
}

bool LoopVectorizationLegality::AddReductionVar(PHINode *Phi,
                                                ReductionKind Kind) {
  if (Phi->getNumIncomingValues() != 2)
    return false;

  // Reduction variables are only found in the loop header block.
  if (Phi->getParent() != TheLoop->getHeader())
    return false;

  // Obtain the reduction start value from the value that comes from the loop
  // preheader.
  Value *RdxStart = Phi->getIncomingValueForBlock(TheLoop->getLoopPreheader());

  // ExitInstruction is the single value which is used outside the loop.
  // We only allow for a single reduction value to be used outside the loop.
  // This includes users of the reduction, variables (which form a cycle
  // which ends in the phi node).
  Instruction *ExitInstruction = 0;
  // Indicates that we found a reduction operation in our scan.
  bool FoundReduxOp = false;

  // We start with the PHI node and scan for all of the users of this
  // instruction. All users must be instructions that can be used as reduction
  // variables (such as ADD). We must have a single out-of-block user. The cycle
  // must include the original PHI.
  bool FoundStartPHI = false;

  // To recognize min/max patterns formed by a icmp select sequence, we store
  // the number of instruction we saw from the recognized min/max pattern,
  //  to make sure we only see exactly the two instructions.
  unsigned NumCmpSelectPatternInst = 0;
  ReductionInstDesc ReduxDesc(false, 0);

  SmallPtrSet<Instruction *, 8> VisitedInsts;
  SmallVector<Instruction *, 8> Worklist;
  Worklist.push_back(Phi);
  VisitedInsts.insert(Phi);

  // A value in the reduction can be used:
  //  - By the reduction:
  //      - Reduction operation:
  //        - One use of reduction value (safe).
  //        - Multiple use of reduction value (not safe).
  //      - PHI:
  //        - All uses of the PHI must be the reduction (safe).
  //        - Otherwise, not safe.
  //  - By one instruction outside of the loop (safe).
  //  - By further instructions outside of the loop (not safe).
  //  - By an instruction that is not part of the reduction (not safe).
  //    This is either:
  //      * An instruction type other than PHI or the reduction operation.
  //      * A PHI in the header other than the initial PHI.
  while (!Worklist.empty()) {
    Instruction *Cur = Worklist.back();
    Worklist.pop_back();

    // No Users.
    // If the instruction has no users then this is a broken chain and can't be
    // a reduction variable.
    if (Cur->use_empty())
      return false;

    bool IsAPhi = isa<PHINode>(Cur);

    // A header PHI use other than the original PHI.
    if (Cur != Phi && IsAPhi && Cur->getParent() == Phi->getParent())
      return false;

    // Reductions of instructions such as Div, and Sub is only possible if the
    // LHS is the reduction variable.
    if (!Cur->isCommutative() && !IsAPhi && !isa<SelectInst>(Cur) &&
        !isa<ICmpInst>(Cur) && !isa<FCmpInst>(Cur) &&
        !VisitedInsts.count(dyn_cast<Instruction>(Cur->getOperand(0))))
      return false;

    // Any reduction instruction must be of one of the allowed kinds.
    ReduxDesc = isReductionInstr(Cur, Kind, ReduxDesc);
    if (!ReduxDesc.IsReduction)
      return false;

    // A reduction operation must only have one use of the reduction value.
    if (!IsAPhi && Kind != RK_IntegerMinMax && Kind != RK_FloatMinMax &&
        hasMultipleUsesOf(Cur, VisitedInsts))
      return false;

    // All inputs to a PHI node must be a reduction value.
    if(IsAPhi && Cur != Phi && !areAllUsesIn(Cur, VisitedInsts))
      return false;

    if (Kind == RK_IntegerMinMax && (isa<ICmpInst>(Cur) ||
                                     isa<SelectInst>(Cur)))
      ++NumCmpSelectPatternInst;
    if (Kind == RK_FloatMinMax && (isa<FCmpInst>(Cur) ||
                                   isa<SelectInst>(Cur)))
      ++NumCmpSelectPatternInst;

    // Check  whether we found a reduction operator.
    FoundReduxOp |= !IsAPhi;

    // Process users of current instruction. Push non PHI nodes after PHI nodes
    // onto the stack. This way we are going to have seen all inputs to PHI
    // nodes once we get to them.
    SmallVector<Instruction *, 8> NonPHIs;
    SmallVector<Instruction *, 8> PHIs;
    for (Value::use_iterator UI = Cur->use_begin(), E = Cur->use_end(); UI != E;
         ++UI) {
      Instruction *Usr = cast<Instruction>(*UI);

      // Check if we found the exit user.
      BasicBlock *Parent = Usr->getParent();
      if (!TheLoop->contains(Parent)) {
        // Exit if you find multiple outside users or if the header phi node is
        // being used. In this case the user uses the value of the previous
        // iteration, in which case we would loose "VF-1" iterations of the
        // reduction operation if we vectorize.
        if (ExitInstruction != 0 || Cur == Phi)
          return false;

        // The instruction used by an outside user must be the last instruction
        // before we feed back to the reduction phi. Otherwise, we loose VF-1
        // operations on the value.
        if (std::find(Phi->op_begin(), Phi->op_end(), Cur) == Phi->op_end())
         return false;

        ExitInstruction = Cur;
        continue;
      }

      // Process instructions only once (termination).
      if (VisitedInsts.insert(Usr)) {
        if (isa<PHINode>(Usr))
          PHIs.push_back(Usr);
        else
          NonPHIs.push_back(Usr);
      }
      // Remember that we completed the cycle.
      if (Usr == Phi)
        FoundStartPHI = true;
    }
    Worklist.append(PHIs.begin(), PHIs.end());
    Worklist.append(NonPHIs.begin(), NonPHIs.end());
  }

  // This means we have seen one but not the other instruction of the
  // pattern or more than just a select and cmp.
  if ((Kind == RK_IntegerMinMax || Kind == RK_FloatMinMax) &&
      NumCmpSelectPatternInst != 2)
    return false;

  if (!FoundStartPHI || !FoundReduxOp || !ExitInstruction)
    return false;

  // We found a reduction var if we have reached the original phi node and we
  // only have a single instruction with out-of-loop users.

  // This instruction is allowed to have out-of-loop users.
  AllowedExit.insert(ExitInstruction);

  // Save the description of this reduction variable.
  ReductionDescriptor RD(RdxStart, ExitInstruction, Kind,
                         ReduxDesc.MinMaxKind);
  Reductions[Phi] = RD;
  // We've ended the cycle. This is a reduction variable if we have an
  // outside user and it has a binary op.

  return true;
}

/// Returns true if the instruction is a Select(ICmp(X, Y), X, Y) instruction
/// pattern corresponding to a min(X, Y) or max(X, Y).
LoopVectorizationLegality::ReductionInstDesc
LoopVectorizationLegality::isMinMaxSelectCmpPattern(Instruction *I,
                                                    ReductionInstDesc &Prev) {

  assert((isa<ICmpInst>(I) || isa<FCmpInst>(I) || isa<SelectInst>(I)) &&
         "Expect a select instruction");
  Instruction *Cmp = 0;
  SelectInst *Select = 0;

  // We must handle the select(cmp()) as a single instruction. Advance to the
  // select.
  if ((Cmp = dyn_cast<ICmpInst>(I)) || (Cmp = dyn_cast<FCmpInst>(I))) {
    if (!Cmp->hasOneUse() || !(Select = dyn_cast<SelectInst>(*I->use_begin())))
      return ReductionInstDesc(false, I);
    return ReductionInstDesc(Select, Prev.MinMaxKind);
  }

  // Only handle single use cases for now.
  if (!(Select = dyn_cast<SelectInst>(I)))
    return ReductionInstDesc(false, I);
  if (!(Cmp = dyn_cast<ICmpInst>(I->getOperand(0))) &&
      !(Cmp = dyn_cast<FCmpInst>(I->getOperand(0))))
    return ReductionInstDesc(false, I);
  if (!Cmp->hasOneUse())
    return ReductionInstDesc(false, I);

  Value *CmpLeft;
  Value *CmpRight;

  // Look for a min/max pattern.
  if (m_UMin(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return ReductionInstDesc(Select, MRK_UIntMin);
  else if (m_UMax(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return ReductionInstDesc(Select, MRK_UIntMax);
  else if (m_SMax(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return ReductionInstDesc(Select, MRK_SIntMax);
  else if (m_SMin(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return ReductionInstDesc(Select, MRK_SIntMin);
  else if (m_OrdFMin(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return ReductionInstDesc(Select, MRK_FloatMin);
  else if (m_OrdFMax(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return ReductionInstDesc(Select, MRK_FloatMax);
  else if (m_UnordFMin(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return ReductionInstDesc(Select, MRK_FloatMin);
  else if (m_UnordFMax(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return ReductionInstDesc(Select, MRK_FloatMax);

  return ReductionInstDesc(false, I);
}

LoopVectorizationLegality::ReductionInstDesc
LoopVectorizationLegality::isReductionInstr(Instruction *I,
                                            ReductionKind Kind,
                                            ReductionInstDesc &Prev) {
  bool FP = I->getType()->isFloatingPointTy();
  bool FastMath = (FP && I->isCommutative() && I->isAssociative());
  switch (I->getOpcode()) {
  default:
    return ReductionInstDesc(false, I);
  case Instruction::PHI:
      if (FP && (Kind != RK_FloatMult && Kind != RK_FloatAdd &&
                 Kind != RK_FloatMinMax))
        return ReductionInstDesc(false, I);
    return ReductionInstDesc(I, Prev.MinMaxKind);
  case Instruction::Sub:
  case Instruction::Add:
    return ReductionInstDesc(Kind == RK_IntegerAdd, I);
  case Instruction::Mul:
    return ReductionInstDesc(Kind == RK_IntegerMult, I);
  case Instruction::And:
    return ReductionInstDesc(Kind == RK_IntegerAnd, I);
  case Instruction::Or:
    return ReductionInstDesc(Kind == RK_IntegerOr, I);
  case Instruction::Xor:
    return ReductionInstDesc(Kind == RK_IntegerXor, I);
  case Instruction::FMul:
    return ReductionInstDesc(Kind == RK_FloatMult && FastMath, I);
  case Instruction::FAdd:
    return ReductionInstDesc(Kind == RK_FloatAdd && FastMath, I);
  case Instruction::FCmp:
  case Instruction::ICmp:
  case Instruction::Select:
    if (Kind != RK_IntegerMinMax &&
        (!HasFunNoNaNAttr || Kind != RK_FloatMinMax))
      return ReductionInstDesc(false, I);
    return isMinMaxSelectCmpPattern(I, Prev);
  }
}

LoopVectorizationLegality::InductionKind
LoopVectorizationLegality::isInductionVariable(PHINode *Phi) {
  Type *PhiTy = Phi->getType();
  // We only handle integer and pointer inductions variables.
  if (!PhiTy->isIntegerTy() && !PhiTy->isPointerTy())
    return IK_NoInduction;

  // Check that the PHI is consecutive.
  const SCEV *PhiScev = SE->getSCEV(Phi);
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PhiScev);
  if (!AR) {
    DEBUG(dbgs() << "LV: PHI is not a poly recurrence.\n");
    return IK_NoInduction;
  }
  const SCEV *Step = AR->getStepRecurrence(*SE);

  // Integer inductions need to have a stride of one.
  if (PhiTy->isIntegerTy()) {
    if (Step->isOne())
      return IK_IntInduction;
    if (Step->isAllOnesValue())
      return IK_ReverseIntInduction;
    return IK_NoInduction;
  }

  // Calculate the pointer stride and check if it is consecutive.
  const SCEVConstant *C = dyn_cast<SCEVConstant>(Step);
  if (!C)
    return IK_NoInduction;

  assert(PhiTy->isPointerTy() && "The PHI must be a pointer");
  uint64_t Size = DL->getTypeAllocSize(PhiTy->getPointerElementType());
  if (C->getValue()->equalsInt(Size))
    return IK_PtrInduction;
  else if (C->getValue()->equalsInt(0 - Size))
    return IK_ReversePtrInduction;

  return IK_NoInduction;
}

bool LoopVectorizationLegality::isInductionVariable(const Value *V) {
  Value *In0 = const_cast<Value*>(V);
  PHINode *PN = dyn_cast_or_null<PHINode>(In0);
  if (!PN)
    return false;

  return Inductions.count(PN);
}

bool LoopVectorizationLegality::blockNeedsPredication(BasicBlock *BB)  {
  assert(TheLoop->contains(BB) && "Unknown block used");

  // Blocks that do not dominate the latch need predication.
  BasicBlock* Latch = TheLoop->getLoopLatch();
  return !DT->dominates(BB, Latch);
}

bool LoopVectorizationLegality::blockCanBePredicated(BasicBlock *BB,
                                            SmallPtrSet<Value *, 8>& SafePtrs) {
  for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
    // We might be able to hoist the load.
    if (it->mayReadFromMemory()) {
      LoadInst *LI = dyn_cast<LoadInst>(it);
      if (!LI || !SafePtrs.count(LI->getPointerOperand()))
        return false;
    }

    // We don't predicate stores at the moment.
    if (it->mayWriteToMemory() || it->mayThrow())
      return false;

    // Check that we don't have a constant expression that can trap as operand.
    for (Instruction::op_iterator OI = it->op_begin(), OE = it->op_end();
         OI != OE; ++OI) {
      if (Constant *C = dyn_cast<Constant>(*OI))
        if (C->canTrap())
          return false;
    }

    // The instructions below can trap.
    switch (it->getOpcode()) {
    default: continue;
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
             return false;
    }
  }

  return true;
}

LoopVectorizationCostModel::VectorizationFactor
LoopVectorizationCostModel::selectVectorizationFactor(bool OptForSize,
                                                      unsigned UserVF) {
  // Width 1 means no vectorize
  VectorizationFactor Factor = { 1U, 0U };
  if (OptForSize && Legal->getRuntimePointerCheck()->Need) {
    DEBUG(dbgs() << "LV: Aborting. Runtime ptr check is required in Os.\n");
    return Factor;
  }

  // Find the trip count.
  unsigned TC = SE->getSmallConstantTripCount(TheLoop, TheLoop->getLoopLatch());
  DEBUG(dbgs() << "LV: Found trip count: " << TC << '\n');

  unsigned WidestType = getWidestType();
  unsigned WidestRegister = TTI.getRegisterBitWidth(true);
  unsigned MaxSafeDepDist = -1U;
  if (Legal->getMaxSafeDepDistBytes() != -1U)
    MaxSafeDepDist = Legal->getMaxSafeDepDistBytes() * 8;
  WidestRegister = ((WidestRegister < MaxSafeDepDist) ?
                    WidestRegister : MaxSafeDepDist);
  unsigned MaxVectorSize = WidestRegister / WidestType;
  DEBUG(dbgs() << "LV: The Widest type: " << WidestType << " bits.\n");
  DEBUG(dbgs() << "LV: The Widest register is: "
          << WidestRegister << " bits.\n");

  if (MaxVectorSize == 0) {
    DEBUG(dbgs() << "LV: The target has no vector registers.\n");
    MaxVectorSize = 1;
  }

  assert(MaxVectorSize <= 32 && "Did not expect to pack so many elements"
         " into one vector!");

  unsigned VF = MaxVectorSize;

  // If we optimize the program for size, avoid creating the tail loop.
  if (OptForSize) {
    // If we are unable to calculate the trip count then don't try to vectorize.
    if (TC < 2) {
      DEBUG(dbgs() << "LV: Aborting. A tail loop is required in Os.\n");
      return Factor;
    }

    // Find the maximum SIMD width that can fit within the trip count.
    VF = TC % MaxVectorSize;

    if (VF == 0)
      VF = MaxVectorSize;

    // If the trip count that we found modulo the vectorization factor is not
    // zero then we require a tail.
    if (VF < 2) {
      DEBUG(dbgs() << "LV: Aborting. A tail loop is required in Os.\n");
      return Factor;
    }
  }

  if (UserVF != 0) {
    assert(isPowerOf2_32(UserVF) && "VF needs to be a power of two");
    DEBUG(dbgs() << "LV: Using user VF " << UserVF << ".\n");

    Factor.Width = UserVF;
    return Factor;
  }

  float Cost = expectedCost(1);
  unsigned Width = 1;
  DEBUG(dbgs() << "LV: Scalar loop costs: " << (int)Cost << ".\n");
  for (unsigned i=2; i <= VF; i*=2) {
    // Notice that the vector loop needs to be executed less times, so
    // we need to divide the cost of the vector loops by the width of
    // the vector elements.
    float VectorCost = expectedCost(i) / (float)i;
    DEBUG(dbgs() << "LV: Vector loop of width " << i << " costs: " <<
          (int)VectorCost << ".\n");
    if (VectorCost < Cost) {
      Cost = VectorCost;
      Width = i;
    }
  }

  DEBUG(dbgs() << "LV: Selecting VF = : "<< Width << ".\n");
  Factor.Width = Width;
  Factor.Cost = Width * Cost;
  return Factor;
}

unsigned LoopVectorizationCostModel::getWidestType() {
  unsigned MaxWidth = 8;

  // For each block.
  for (Loop::block_iterator bb = TheLoop->block_begin(),
       be = TheLoop->block_end(); bb != be; ++bb) {
    BasicBlock *BB = *bb;

    // For each instruction in the loop.
    for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
      Type *T = it->getType();

      // Only examine Loads, Stores and PHINodes.
      if (!isa<LoadInst>(it) && !isa<StoreInst>(it) && !isa<PHINode>(it))
        continue;

      // Examine PHI nodes that are reduction variables.
      if (PHINode *PN = dyn_cast<PHINode>(it))
        if (!Legal->getReductionVars()->count(PN))
          continue;

      // Examine the stored values.
      if (StoreInst *ST = dyn_cast<StoreInst>(it))
        T = ST->getValueOperand()->getType();

      // Ignore loaded pointer types and stored pointer types that are not
      // consecutive. However, we do want to take consecutive stores/loads of
      // pointer vectors into account.
      if (T->isPointerTy() && !isConsecutiveLoadOrStore(it))
        continue;

      MaxWidth = std::max(MaxWidth,
                          (unsigned)DL->getTypeSizeInBits(T->getScalarType()));
    }
  }

  return MaxWidth;
}

unsigned
LoopVectorizationCostModel::selectUnrollFactor(bool OptForSize,
                                               unsigned UserUF,
                                               unsigned VF,
                                               unsigned LoopCost) {

  // -- The unroll heuristics --
  // We unroll the loop in order to expose ILP and reduce the loop overhead.
  // There are many micro-architectural considerations that we can't predict
  // at this level. For example frontend pressure (on decode or fetch) due to
  // code size, or the number and capabilities of the execution ports.
  //
  // We use the following heuristics to select the unroll factor:
  // 1. If the code has reductions the we unroll in order to break the cross
  // iteration dependency.
  // 2. If the loop is really small then we unroll in order to reduce the loop
  // overhead.
  // 3. We don't unroll if we think that we will spill registers to memory due
  // to the increased register pressure.

  // Use the user preference, unless 'auto' is selected.
  if (UserUF != 0)
    return UserUF;

  // When we optimize for size we don't unroll.
  if (OptForSize)
    return 1;

  // We used the distance for the unroll factor.
  if (Legal->getMaxSafeDepDistBytes() != -1U)
    return 1;

  // Do not unroll loops with a relatively small trip count.
  unsigned TC = SE->getSmallConstantTripCount(TheLoop,
                                              TheLoop->getLoopLatch());
  if (TC > 1 && TC < TinyTripCountUnrollThreshold)
    return 1;

  unsigned TargetVectorRegisters = TTI.getNumberOfRegisters(true);
  DEBUG(dbgs() << "LV: The target has " << TargetVectorRegisters <<
        " vector registers\n");

  LoopVectorizationCostModel::RegisterUsage R = calculateRegisterUsage();
  // We divide by these constants so assume that we have at least one
  // instruction that uses at least one register.
  R.MaxLocalUsers = std::max(R.MaxLocalUsers, 1U);
  R.NumInstructions = std::max(R.NumInstructions, 1U);

  // We calculate the unroll factor using the following formula.
  // Subtract the number of loop invariants from the number of available
  // registers. These registers are used by all of the unrolled instances.
  // Next, divide the remaining registers by the number of registers that is
  // required by the loop, in order to estimate how many parallel instances
  // fit without causing spills.
  unsigned UF = (TargetVectorRegisters - R.LoopInvariantRegs) / R.MaxLocalUsers;

  // Clamp the unroll factor ranges to reasonable factors.
  unsigned MaxUnrollSize = TTI.getMaximumUnrollFactor();

  // If we did not calculate the cost for VF (because the user selected the VF)
  // then we calculate the cost of VF here.
  if (LoopCost == 0)
    LoopCost = expectedCost(VF);

  // Clamp the calculated UF to be between the 1 and the max unroll factor
  // that the target allows.
  if (UF > MaxUnrollSize)
    UF = MaxUnrollSize;
  else if (UF < 1)
    UF = 1;

  bool HasReductions = Legal->getReductionVars()->size();

  // Decide if we want to unroll if we decided that it is legal to vectorize
  // but not profitable.
  if (VF == 1) {
    if (TheLoop->getNumBlocks() > 1 || !HasReductions ||
        LoopCost > SmallLoopCost)
      return 1;

    return UF;
  }

  if (HasReductions) {
    DEBUG(dbgs() << "LV: Unrolling because of reductions.\n");
    return UF;
  }

  // We want to unroll tiny loops in order to reduce the loop overhead.
  // We assume that the cost overhead is 1 and we use the cost model
  // to estimate the cost of the loop and unroll until the cost of the
  // loop overhead is about 5% of the cost of the loop.
  DEBUG(dbgs() << "LV: Loop cost is " << LoopCost << '\n');
  if (LoopCost < SmallLoopCost) {
    DEBUG(dbgs() << "LV: Unrolling to reduce branch cost.\n");
    unsigned NewUF = SmallLoopCost / (LoopCost + 1);
    return std::min(NewUF, UF);
  }

  DEBUG(dbgs() << "LV: Not Unrolling.\n");
  return 1;
}

LoopVectorizationCostModel::RegisterUsage
LoopVectorizationCostModel::calculateRegisterUsage() {
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

  RegisterUsage R;
  R.NumInstructions = 0;

  // Each 'key' in the map opens a new interval. The values
  // of the map are the index of the 'last seen' usage of the
  // instruction that is the key.
  typedef DenseMap<Instruction*, unsigned> IntervalMap;
  // Maps instruction to its index.
  DenseMap<unsigned, Instruction*> IdxToInstr;
  // Marks the end of each interval.
  IntervalMap EndPoint;
  // Saves the list of instruction indices that are used in the loop.
  SmallSet<Instruction*, 8> Ends;
  // Saves the list of values that are used in the loop but are
  // defined outside the loop, such as arguments and constants.
  SmallPtrSet<Value*, 8> LoopInvariants;

  unsigned Index = 0;
  for (LoopBlocksDFS::RPOIterator bb = DFS.beginRPO(),
       be = DFS.endRPO(); bb != be; ++bb) {
    R.NumInstructions += (*bb)->size();
    for (BasicBlock::iterator it = (*bb)->begin(), e = (*bb)->end(); it != e;
         ++it) {
      Instruction *I = it;
      IdxToInstr[Index++] = I;

      // Save the end location of each USE.
      for (unsigned i = 0; i < I->getNumOperands(); ++i) {
        Value *U = I->getOperand(i);
        Instruction *Instr = dyn_cast<Instruction>(U);

        // Ignore non-instruction values such as arguments, constants, etc.
        if (!Instr) continue;

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
  typedef SmallVector<Instruction*, 2> InstrList;
  DenseMap<unsigned, InstrList> TransposeEnds;

  // Transpose the EndPoints to a list of values that end at each index.
  for (IntervalMap::iterator it = EndPoint.begin(), e = EndPoint.end();
       it != e; ++it)
    TransposeEnds[it->second].push_back(it->first);

  SmallSet<Instruction*, 8> OpenIntervals;
  unsigned MaxUsage = 0;


  DEBUG(dbgs() << "LV(REG): Calculating max register usage:\n");
  for (unsigned int i = 0; i < Index; ++i) {
    Instruction *I = IdxToInstr[i];
    // Ignore instructions that are never used within the loop.
    if (!Ends.count(I)) continue;

    // Remove all of the instructions that end at this location.
    InstrList &List = TransposeEnds[i];
    for (unsigned int j=0, e = List.size(); j < e; ++j)
      OpenIntervals.erase(List[j]);

    // Count the number of live interals.
    MaxUsage = std::max(MaxUsage, OpenIntervals.size());

    DEBUG(dbgs() << "LV(REG): At #" << i << " Interval # " <<
          OpenIntervals.size() << '\n');

    // Add the current instruction to the list of open intervals.
    OpenIntervals.insert(I);
  }

  unsigned Invariant = LoopInvariants.size();
  DEBUG(dbgs() << "LV(REG): Found max usage: " << MaxUsage << '\n');
  DEBUG(dbgs() << "LV(REG): Found invariant usage: " << Invariant << '\n');
  DEBUG(dbgs() << "LV(REG): LoopSize: " << R.NumInstructions << '\n');

  R.LoopInvariantRegs = Invariant;
  R.MaxLocalUsers = MaxUsage;
  return R;
}

unsigned LoopVectorizationCostModel::expectedCost(unsigned VF) {
  unsigned Cost = 0;

  // For each block.
  for (Loop::block_iterator bb = TheLoop->block_begin(),
       be = TheLoop->block_end(); bb != be; ++bb) {
    unsigned BlockCost = 0;
    BasicBlock *BB = *bb;

    // For each instruction in the old loop.
    for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
      // Skip dbg intrinsics.
      if (isa<DbgInfoIntrinsic>(it))
        continue;

      unsigned C = getInstructionCost(it, VF);
      BlockCost += C;
      DEBUG(dbgs() << "LV: Found an estimated cost of " << C << " for VF " <<
            VF << " For instruction: " << *it << '\n');
    }

    // We assume that if-converted blocks have a 50% chance of being executed.
    // When the code is scalar then some of the blocks are avoided due to CF.
    // When the code is vectorized we execute all code paths.
    if (VF == 1 && Legal->blockNeedsPredication(*bb))
      BlockCost /= 2;

    Cost += BlockCost;
  }

  return Cost;
}

/// \brief Check whether the address computation for a non-consecutive memory
/// access looks like an unlikely candidate for being merged into the indexing
/// mode.
///
/// We look for a GEP which has one index that is an induction variable and all
/// other indices are loop invariant. If the stride of this access is also
/// within a small bound we decide that this address computation can likely be
/// merged into the addressing mode.
/// In all other cases, we identify the address computation as complex.
static bool isLikelyComplexAddressComputation(Value *Ptr,
                                              LoopVectorizationLegality *Legal,
                                              ScalarEvolution *SE,
                                              const Loop *TheLoop) {
  GetElementPtrInst *Gep = dyn_cast<GetElementPtrInst>(Ptr);
  if (!Gep)
    return true;

  // We are looking for a gep with all loop invariant indices except for one
  // which should be an induction variable.
  unsigned NumOperands = Gep->getNumOperands();
  for (unsigned i = 1; i < NumOperands; ++i) {
    Value *Opd = Gep->getOperand(i);
    if (!SE->isLoopInvariant(SE->getSCEV(Opd), TheLoop) &&
        !Legal->isInductionVariable(Opd))
      return true;
  }

  // Now we know we have a GEP ptr, %inv, %ind, %inv. Make sure that the step
  // can likely be merged into the address computation.
  unsigned MaxMergeDistance = 64;

  const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(SE->getSCEV(Ptr));
  if (!AddRec)
    return true;

  // Check the step is constant.
  const SCEV *Step = AddRec->getStepRecurrence(*SE);
  // Calculate the pointer stride and check if it is consecutive.
  const SCEVConstant *C = dyn_cast<SCEVConstant>(Step);
  if (!C)
    return true;

  const APInt &APStepVal = C->getValue()->getValue();

  // Huge step value - give up.
  if (APStepVal.getBitWidth() > 64)
    return true;

  int64_t StepVal = APStepVal.getSExtValue();

  return StepVal > MaxMergeDistance;
}

unsigned
LoopVectorizationCostModel::getInstructionCost(Instruction *I, unsigned VF) {
  // If we know that this instruction will remain uniform, check the cost of
  // the scalar version.
  if (Legal->isUniformAfterVectorization(I))
    VF = 1;

  Type *RetTy = I->getType();
  Type *VectorTy = ToVectorTy(RetTy, VF);

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
  case Instruction::PHI:
    //TODO: IF-converted IFs become selects.
    return 0;
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    // Certain instructions can be cheaper to vectorize if they have a constant
    // second vector operand. One example of this are shifts on x86.
    TargetTransformInfo::OperandValueKind Op1VK =
      TargetTransformInfo::OK_AnyValue;
    TargetTransformInfo::OperandValueKind Op2VK =
      TargetTransformInfo::OK_AnyValue;

    if (isa<ConstantInt>(I->getOperand(1)))
      Op2VK = TargetTransformInfo::OK_UniformConstantValue;

    return TTI.getArithmeticInstrCost(I->getOpcode(), VectorTy, Op1VK, Op2VK);
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
    VectorTy = ToVectorTy(ValTy, VF);
    return TTI.getCmpSelInstrCost(I->getOpcode(), VectorTy);
  }
  case Instruction::Store:
  case Instruction::Load: {
    StoreInst *SI = dyn_cast<StoreInst>(I);
    LoadInst *LI = dyn_cast<LoadInst>(I);
    Type *ValTy = (SI ? SI->getValueOperand()->getType() :
                   LI->getType());
    VectorTy = ToVectorTy(ValTy, VF);

    unsigned Alignment = SI ? SI->getAlignment() : LI->getAlignment();
    unsigned AS = SI ? SI->getPointerAddressSpace() :
      LI->getPointerAddressSpace();
    Value *Ptr = SI ? SI->getPointerOperand() : LI->getPointerOperand();
    // We add the cost of address computation here instead of with the gep
    // instruction because only here we know whether the operation is
    // scalarized.
    if (VF == 1)
      return TTI.getAddressComputationCost(VectorTy) +
        TTI.getMemoryOpCost(I->getOpcode(), VectorTy, Alignment, AS);

    // Scalarized loads/stores.
    int ConsecutiveStride = Legal->isConsecutivePtr(Ptr);
    bool Reverse = ConsecutiveStride < 0;
    unsigned ScalarAllocatedSize = DL->getTypeAllocSize(ValTy);
    unsigned VectorElementSize = DL->getTypeStoreSize(VectorTy)/VF;
    if (!ConsecutiveStride || ScalarAllocatedSize != VectorElementSize) {
      bool IsComplexComputation =
        isLikelyComplexAddressComputation(Ptr, Legal, SE, TheLoop);
      unsigned Cost = 0;
      // The cost of extracting from the value vector and pointer vector.
      Type *PtrTy = ToVectorTy(Ptr->getType(), VF);
      for (unsigned i = 0; i < VF; ++i) {
        //  The cost of extracting the pointer operand.
        Cost += TTI.getVectorInstrCost(Instruction::ExtractElement, PtrTy, i);
        // In case of STORE, the cost of ExtractElement from the vector.
        // In case of LOAD, the cost of InsertElement into the returned
        // vector.
        Cost += TTI.getVectorInstrCost(SI ? Instruction::ExtractElement :
                                            Instruction::InsertElement,
                                            VectorTy, i);
      }

      // The cost of the scalar loads/stores.
      Cost += VF * TTI.getAddressComputationCost(PtrTy, IsComplexComputation);
      Cost += VF * TTI.getMemoryOpCost(I->getOpcode(), ValTy->getScalarType(),
                                       Alignment, AS);
      return Cost;
    }

    // Wide load/stores.
    unsigned Cost = TTI.getAddressComputationCost(VectorTy);
    Cost += TTI.getMemoryOpCost(I->getOpcode(), VectorTy, Alignment, AS);

    if (Reverse)
      Cost += TTI.getShuffleCost(TargetTransformInfo::SK_Reverse,
                                  VectorTy, 0);
    return Cost;
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
    // We optimize the truncation of induction variable.
    // The cost of these is the same as the scalar operation.
    if (I->getOpcode() == Instruction::Trunc &&
        Legal->isInductionVariable(I->getOperand(0)))
      return TTI.getCastInstrCost(I->getOpcode(), I->getType(),
                                  I->getOperand(0)->getType());

    Type *SrcVecTy = ToVectorTy(I->getOperand(0)->getType(), VF);
    return TTI.getCastInstrCost(I->getOpcode(), VectorTy, SrcVecTy);
  }
  case Instruction::Call: {
    CallInst *CI = cast<CallInst>(I);
    Intrinsic::ID ID = getIntrinsicIDForCall(CI, TLI);
    assert(ID && "Not an intrinsic call!");
    Type *RetTy = ToVectorTy(CI->getType(), VF);
    SmallVector<Type*, 4> Tys;
    for (unsigned i = 0, ie = CI->getNumArgOperands(); i != ie; ++i)
      Tys.push_back(ToVectorTy(CI->getArgOperand(i)->getType(), VF));
    return TTI.getIntrinsicInstrCost(ID, RetTy, Tys);
  }
  default: {
    // We are scalarizing the instruction. Return the cost of the scalar
    // instruction, plus the cost of insert and extract into vector
    // elements, times the vector width.
    unsigned Cost = 0;

    if (!RetTy->isVoidTy() && VF != 1) {
      unsigned InsCost = TTI.getVectorInstrCost(Instruction::InsertElement,
                                                VectorTy);
      unsigned ExtCost = TTI.getVectorInstrCost(Instruction::ExtractElement,
                                                VectorTy);

      // The cost of inserting the results plus extracting each one of the
      // operands.
      Cost += VF * (InsCost + ExtCost * I->getNumOperands());
    }

    // The cost of executing VF copies of the scalar instruction. This opcode
    // is unknown. Assume that it is the same as 'mul'.
    Cost += VF * TTI.getArithmeticInstrCost(Instruction::Mul, VectorTy);
    return Cost;
  }
  }// end of switch.
}

Type* LoopVectorizationCostModel::ToVectorTy(Type *Scalar, unsigned VF) {
  if (Scalar->isVoidTy() || VF == 1)
    return Scalar;
  return VectorType::get(Scalar, VF);
}

char LoopVectorize::ID = 0;
static const char lv_name[] = "Loop Vectorization";
INITIALIZE_PASS_BEGIN(LoopVectorize, LV_NAME, lv_name, false, false)
INITIALIZE_AG_DEPENDENCY(TargetTransformInfo)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_END(LoopVectorize, LV_NAME, lv_name, false, false)

namespace llvm {
  Pass *createLoopVectorizePass(bool NoUnrolling) {
    return new LoopVectorize(NoUnrolling);
  }
}

bool LoopVectorizationCostModel::isConsecutiveLoadOrStore(Instruction *Inst) {
  // Check for a store.
  if (StoreInst *ST = dyn_cast<StoreInst>(Inst))
    return Legal->isConsecutivePtr(ST->getPointerOperand()) != 0;

  // Check for a load.
  if (LoadInst *LI = dyn_cast<LoadInst>(Inst))
    return Legal->isConsecutivePtr(LI->getPointerOperand()) != 0;

  return false;
}


void InnerLoopUnroller::scalarizeInstruction(Instruction *Instr) {
  assert(!Instr->getType()->isAggregateType() && "Can't handle vectors");
  // Holds vector parameters or scalars, in case of uniform vals.
  SmallVector<VectorParts, 4> Params;

  setDebugLocFromInst(Builder, Instr);

  // Find all of the vectorized parameters.
  for (unsigned op = 0, e = Instr->getNumOperands(); op != e; ++op) {
    Value *SrcOp = Instr->getOperand(op);

    // If we are accessing the old induction variable, use the new one.
    if (SrcOp == OldInduction) {
      Params.push_back(getVectorValue(SrcOp));
      continue;
    }

    // Try using previously calculated values.
    Instruction *SrcInst = dyn_cast<Instruction>(SrcOp);

    // If the src is an instruction that appeared earlier in the basic block
    // then it should already be vectorized.
    if (SrcInst && OrigLoop->contains(SrcInst)) {
      assert(WidenMap.has(SrcInst) && "Source operand is unavailable");
      // The parameter is a vector value from earlier.
      Params.push_back(WidenMap.get(SrcInst));
    } else {
      // The parameter is a scalar from outside the loop. Maybe even a constant.
      VectorParts Scalars;
      Scalars.append(UF, SrcOp);
      Params.push_back(Scalars);
    }
  }

  assert(Params.size() == Instr->getNumOperands() &&
         "Invalid number of operands");

  // Does this instruction return a value ?
  bool IsVoidRetTy = Instr->getType()->isVoidTy();

  Value *UndefVec = IsVoidRetTy ? 0 :
  UndefValue::get(Instr->getType());
  // Create a new entry in the WidenMap and initialize it to Undef or Null.
  VectorParts &VecResults = WidenMap.splat(Instr, UndefVec);

  // For each vector unroll 'part':
  for (unsigned Part = 0; Part < UF; ++Part) {
    // For each scalar that we create:

    Instruction *Cloned = Instr->clone();
      if (!IsVoidRetTy)
        Cloned->setName(Instr->getName() + ".cloned");
      // Replace the operands of the cloned instructions with extracted scalars.
      for (unsigned op = 0, e = Instr->getNumOperands(); op != e; ++op) {
        Value *Op = Params[op][Part];
        Cloned->setOperand(op, Op);
      }

      // Place the cloned scalar in the new loop.
      Builder.Insert(Cloned);

      // If the original scalar returns a value we need to place it in a vector
      // so that future users will be able to use it.
      if (!IsVoidRetTy)
        VecResults[Part] = Cloned;
  }
}

void
InnerLoopUnroller::vectorizeMemoryInstruction(Instruction *Instr,
                                              LoopVectorizationLegality*) {
  return scalarizeInstruction(Instr);
}

Value *InnerLoopUnroller::reverseVector(Value *Vec) {
  return Vec;
}

Value *InnerLoopUnroller::getBroadcastInstrs(Value *V) {
  return V;
}

Value *InnerLoopUnroller::getConsecutiveVector(Value* Val, int StartIdx,
                                               bool Negate) {
  // When unrolling and the VF is 1, we only need to add a simple scalar.
  Type *ITy = Val->getType();
  assert(!ITy->isVectorTy() && "Val must be a scalar");
  Constant *C = ConstantInt::get(ITy, StartIdx, Negate);
  return Builder.CreateAdd(Val, C, "induction");
}

