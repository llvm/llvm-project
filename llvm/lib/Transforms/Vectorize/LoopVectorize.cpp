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
// and generates target-independent LLVM-IR. Legalization of the IR is done
// in the codegen. However, the vectorizes uses (will use) the codegen
// interfaces to generate IR that is likely to result in an optimal binary.
//
// The loop vectorizer combines consecutive loop iteration into a single
// 'wide' iteration. After this transformation the index is incremented
// by the SIMD vector width, and not by one.
//
// This pass has three parts:
// 1. The main loop pass that drives the different parts.
// 2. LoopVectorizationLegality - A unit that checks for the legality
//    of the vectorization.
// 3. SingleBlockLoopVectorizer - A unit that performs the actual
//    widening of instructions.
// 4. LoopVectorizationCostModel - A unit that checks for the profitability
//    of vectorization. It decides on the optimal vector width, which
//    can be one, if vectorization is not profitable.
//===----------------------------------------------------------------------===//
//
// The reduction-variable vectorization is based on the paper:
//  D. Nuzman and R. Henderson. Multi-platform Auto-vectorization.
//
// Variable uniformity checks are inspired by:
// Karrenberg, R. and Hack, S. Whole Function Vectorization.
//
// Other ideas/concepts are from:
//  A. Zaks and D. Nuzman. Autovectorization in GCC-two years later.
//
//===----------------------------------------------------------------------===//
#define LV_NAME "loop-vectorize"
#define DEBUG_TYPE LV_NAME
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Value.h"
#include "llvm/Function.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/TargetTransformInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/DataLayout.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
using namespace llvm;

static cl::opt<unsigned>
VectorizationFactor("force-vector-width", cl::init(0), cl::Hidden,
          cl::desc("Set the default vectorization width. Zero is autoselect."));

/// We don't vectorize loops with a known constant trip count below this number.
const unsigned TinyTripCountThreshold = 16;

/// When performing a runtime memory check, do not check more than this
/// number of pointers. Notice that the check is quadratic!
const unsigned RuntimeMemoryCheckThreshold = 2;

namespace {

// Forward declarations.
class LoopVectorizationLegality;
class LoopVectorizationCostModel;

/// SingleBlockLoopVectorizer vectorizes loops which contain only one basic
/// block to a specified vectorization factor (VF).
/// This class performs the widening of scalars into vectors, or multiple
/// scalars. This class also implements the following features:
/// * It inserts an epilogue loop for handling loops that don't have iteration
///   counts that are known to be a multiple of the vectorization factor.
/// * It handles the code generation for reduction variables.
/// * Scalarization (implementation using scalars) of un-vectorizable
///   instructions.
/// SingleBlockLoopVectorizer does not perform any vectorization-legality
/// checks, and relies on the caller to check for the different legality
/// aspects. The SingleBlockLoopVectorizer relies on the
/// LoopVectorizationLegality class to provide information about the induction
/// and reduction variables that were found to a given vectorization factor.
class SingleBlockLoopVectorizer {
public:
  /// Ctor.
  SingleBlockLoopVectorizer(Loop *Orig, ScalarEvolution *Se, LoopInfo *Li,
                            DominatorTree *dt, LPPassManager *Lpm,
                            unsigned VecWidth):
  OrigLoop(Orig), SE(Se), LI(Li), DT(dt), LPM(Lpm), VF(VecWidth),
  Builder(Se->getContext()), Induction(0), OldInduction(0) { }

  // Perform the actual loop widening (vectorization).
  void vectorize(LoopVectorizationLegality *Legal) {
    ///Create a new empty loop. Unlink the old loop and connect the new one.
    createEmptyLoop(Legal);
    /// Widen each instruction in the old loop to a new one in the new loop.
    /// Use the Legality module to find the induction and reduction variables.
    vectorizeLoop(Legal);
    // Register the new loop and update the analysis passes.
    updateAnalysis();
 }

private:
  /// Create an empty loop, based on the loop ranges of the old loop.
  void createEmptyLoop(LoopVectorizationLegality *Legal);
  /// Copy and widen the instructions from the old loop.
  void vectorizeLoop(LoopVectorizationLegality *Legal);
  /// Insert the new loop to the loop hierarchy and pass manager
  /// and update the analysis passes.
  void updateAnalysis();

  /// This instruction is un-vectorizable. Implement it as a sequence
  /// of scalars.
  void scalarizeInstruction(Instruction *Instr);

  /// Create a broadcast instruction. This method generates a broadcast
  /// instruction (shuffle) for loop invariant values and for the induction
  /// value. If this is the induction variable then we extend it to N, N+1, ...
  /// this is needed because each iteration in the loop corresponds to a SIMD
  /// element.
  Value *getBroadcastInstrs(Value *V);

  /// This is a helper function used by getBroadcastInstrs. It adds 0, 1, 2 ..
  /// for each element in the vector. Starting from zero.
  Value *getConsecutiveVector(Value* Val);

  /// When we go over instructions in the basic block we rely on previous
  /// values within the current basic block or on loop invariant values.
  /// When we widen (vectorize) values we place them in the map. If the values
  /// are not within the map, they have to be loop invariant, so we simply
  /// broadcast them into a vector.
  Value *getVectorValue(Value *V);

  /// Get a uniform vector of constant integers. We use this to get
  /// vectors of ones and zeros for the reduction code.
  Constant* getUniformVector(unsigned Val, Type* ScalarTy);

  typedef DenseMap<Value*, Value*> ValueMap;

  /// The original loop.
  Loop *OrigLoop;
  // Scev analysis to use.
  ScalarEvolution *SE;
  // Loop Info.
  LoopInfo *LI;
  // Dominator Tree.
  DominatorTree *DT;
  // Loop Pass Manager;
  LPPassManager *LPM;
  // The vectorization factor to use.
  unsigned VF;

  // The builder that we use
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
  ///The first bypass block.
  BasicBlock *LoopBypassBlock;

  /// The new Induction variable which was added to the new block.
  PHINode *Induction;
  /// The induction variable of the old basic block.
  PHINode *OldInduction;
  // Maps scalars to widened vectors.
  ValueMap WidenMap;
};

/// LoopVectorizationLegality checks if it is legal to vectorize a loop, and
/// to what vectorization factor.
/// This class does not look at the profitability of vectorization, only the
/// legality. This class has two main kinds of checks:
/// * Memory checks - The code in canVectorizeMemory checks if vectorization
///   will change the order of memory accesses in a way that will change the
///   correctness of the program.
/// * Scalars checks - The code in canVectorizeBlock checks for a number
///   of different conditions, such as the availability of a single induction
///   variable, that all types are supported and vectorize-able, etc.
/// This code reflects the capabilities of SingleBlockLoopVectorizer.
/// This class is also used by SingleBlockLoopVectorizer for identifying
/// induction variable and the different reduction variables.
class LoopVectorizationLegality {
public:
  LoopVectorizationLegality(Loop *Lp, ScalarEvolution *Se, DataLayout *Dl):
  TheLoop(Lp), SE(Se), DL(Dl), Induction(0) { }

  /// This represents the kinds of reductions that we support.
  enum ReductionKind {
    NoReduction, /// Not a reduction.
    IntegerAdd,  /// Sum of numbers.
    IntegerMult, /// Product of numbers.
    IntegerOr,   /// Bitwise or logical OR of numbers.
    IntegerAnd,  /// Bitwise or logical AND of numbers.
    IntegerXor   /// Bitwise or logical XOR of numbers.
  };

  /// This POD struct holds information about reduction variables.
  struct ReductionDescriptor {
    // Default C'tor
    ReductionDescriptor():
    StartValue(0), LoopExitInstr(0), Kind(NoReduction) {}

    // C'tor.
    ReductionDescriptor(Value *Start, Instruction *Exit, ReductionKind K):
    StartValue(Start), LoopExitInstr(Exit), Kind(K) {}

    // The starting value of the reduction.
    // It does not have to be zero!
    Value *StartValue;
    // The instruction who's value is used outside the loop.
    Instruction *LoopExitInstr;
    // The kind of the reduction.
    ReductionKind Kind;
  };

  // This POD struct holds information about the memory runtime legality
  // check that a group of pointers do not overlap.
  struct RuntimePointerCheck {
    /// This flag indicates if we need to add the runtime check.
    bool Need;
    /// Holds the pointers that we need to check.
    SmallVector<Value*, 2> Pointers;
  };

  /// ReductionList contains the reduction descriptors for all
  /// of the reductions that were found in the loop.
  typedef DenseMap<PHINode*, ReductionDescriptor> ReductionList;

  /// Returns true if it is legal to vectorize this loop.
  /// This does not mean that it is profitable to vectorize this
  /// loop, only that it is legal to do so.
  bool canVectorize();

  /// Returns the Induction variable.
  PHINode *getInduction() {return Induction;}

  /// Returns the reduction variables found in the loop.
  ReductionList *getReductionVars() { return &Reductions; }

  /// Check if the pointer returned by this GEP is consecutive
  /// when the index is vectorized. This happens when the last
  /// index of the GEP is consecutive, like the induction variable.
  /// This check allows us to vectorize A[idx] into a wide load/store.
  bool isConsecutiveGep(Value *Ptr);

  /// Returns true if the value V is uniform within the loop.
  bool isUniform(Value *V);

  /// Returns true if this instruction will remain scalar after vectorization.
  bool isUniformAfterVectorization(Instruction* I) {return Uniforms.count(I);}

  /// Returns the information that we collected about runtime memory check.
  RuntimePointerCheck *getRuntimePointerCheck() {return &PtrRtCheck; }
private:
  /// Check if a single basic block loop is vectorizable.
  /// At this point we know that this is a loop with a constant trip count
  /// and we only need to check individual instructions.
  bool canVectorizeBlock(BasicBlock &BB);

  /// When we vectorize loops we may change the order in which
  /// we read and write from memory. This method checks if it is
  /// legal to vectorize the code, considering only memory constrains.
  /// Returns true if BB is vectorizable
  bool canVectorizeMemory(BasicBlock &BB);

  /// Returns True, if 'Phi' is the kind of reduction variable for type
  /// 'Kind'. If this is a reduction variable, it adds it to ReductionList.
  bool AddReductionVar(PHINode *Phi, ReductionKind Kind);
  /// Returns true if the instruction I can be a reduction variable of type
  /// 'Kind'.
  bool isReductionInstr(Instruction *I, ReductionKind Kind);
  /// Returns True, if 'Phi' is an induction variable.
  bool isInductionVariable(PHINode *Phi);
  /// Return true if can compute the address bounds of Ptr within the loop.
  bool hasComputableBounds(Value *Ptr);

  /// The loop that we evaluate.
  Loop *TheLoop;
  /// Scev analysis.
  ScalarEvolution *SE;
  /// DataLayout analysis.
  DataLayout *DL;

  //  ---  vectorization state --- //

  /// Holds the induction variable.
  PHINode *Induction;
  /// Holds the reduction variables.
  ReductionList Reductions;
  /// Allowed outside users. This holds the reduction
  /// vars which can be accessed from outside the loop.
  SmallPtrSet<Value*, 4> AllowedExit;
  /// This set holds the variables which are known to be uniform after
  /// vectorization.
  SmallPtrSet<Instruction*, 4> Uniforms;
  /// We need to check that all of the pointers in this list are disjoint
  /// at runtime.
  RuntimePointerCheck PtrRtCheck;
};

/// LoopVectorizationCostModel - estimates the expected speedups due to
/// vectorization.
/// In many cases vectorization is not profitable. This can happen because
/// of a number of reasons. In this class we mainly attempt to predict
/// the expected speedup/slowdowns due to the supported instruction set.
/// We use the VectorTargetTransformInfo to query the different backends
/// for the cost of different operations.
class LoopVectorizationCostModel {
public:
  /// C'tor.
  LoopVectorizationCostModel(Loop *Lp, ScalarEvolution *Se,
                             LoopVectorizationLegality *Leg,
                             const VectorTargetTransformInfo *Vtti):
  TheLoop(Lp), SE(Se), Legal(Leg), VTTI(Vtti) { }

  /// Returns the most profitable vectorization factor for the loop that is
  /// smaller or equal to the VF argument. This method checks every power
  /// of two up to VF.
  unsigned findBestVectorizationFactor(unsigned VF = 8);

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

  /// The loop that we evaluate.
  Loop *TheLoop;
  /// Scev analysis.
  ScalarEvolution *SE;

  /// Vectorization legality.
  LoopVectorizationLegality *Legal;
  /// Vector target information.
  const VectorTargetTransformInfo *VTTI;
};

struct LoopVectorize : public LoopPass {
  static char ID; // Pass identification, replacement for typeid

  LoopVectorize() : LoopPass(ID) {
    initializeLoopVectorizePass(*PassRegistry::getPassRegistry());
  }

  ScalarEvolution *SE;
  DataLayout *DL;
  LoopInfo *LI;
  TargetTransformInfo *TTI;
  DominatorTree *DT;

  virtual bool runOnLoop(Loop *L, LPPassManager &LPM) {
    // We only vectorize innermost loops.
    if (!L->empty())
      return false;

    SE = &getAnalysis<ScalarEvolution>();
    DL = getAnalysisIfAvailable<DataLayout>();
    LI = &getAnalysis<LoopInfo>();
    TTI = getAnalysisIfAvailable<TargetTransformInfo>();
    DT = &getAnalysis<DominatorTree>();

    DEBUG(dbgs() << "LV: Checking a loop in \"" <<
          L->getHeader()->getParent()->getName() << "\"\n");

    // Check if it is legal to vectorize the loop.
    LoopVectorizationLegality LVL(L, SE, DL);
    if (!LVL.canVectorize()) {
      DEBUG(dbgs() << "LV: Not vectorizing.\n");
      return false;
    }

    // Select the preffered vectorization factor.
    unsigned VF = 1;
    if (VectorizationFactor == 0) {
      const VectorTargetTransformInfo *VTTI = 0;
      if (TTI)
        VTTI = TTI->getVectorTargetTransformInfo();
      // Use the cost model.
      LoopVectorizationCostModel CM(L, SE, &LVL, VTTI);
      VF = CM.findBestVectorizationFactor();

      if (VF == 1) {
        DEBUG(dbgs() << "LV: Vectorization is possible but not beneficial.\n");
        return false;
      }

    } else {
      // Use the user command flag.
      VF = VectorizationFactor;
    }

    DEBUG(dbgs() << "LV: Found a vectorizable loop ("<< VF << ") in "<<
          L->getHeader()->getParent()->getParent()->getModuleIdentifier()<<
          "\n");

    // If we decided that it is *legal* to vectorizer the loop then do it.
    SingleBlockLoopVectorizer LB(L, SE, LI, DT, &LPM, VF);
    LB.vectorize(&LVL);

    DEBUG(verifyFunction(*L->getHeader()->getParent()));
    return true;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    LoopPass::getAnalysisUsage(AU);
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequiredID(LCSSAID);
    AU.addRequired<LoopInfo>();
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<DominatorTree>();
    AU.addPreserved<LoopInfo>();
    AU.addPreserved<DominatorTree>();
  }

};

Value *SingleBlockLoopVectorizer::getBroadcastInstrs(Value *V) {
  // Instructions that access the old induction variable
  // actually want to get the new one.
  if (V == OldInduction)
    V = Induction;
  // Create the types.
  LLVMContext &C = V->getContext();
  Type *VTy = VectorType::get(V->getType(), VF);
  Type *I32 = IntegerType::getInt32Ty(C);
  Constant *Zero = ConstantInt::get(I32, 0);
  Value *Zeros = ConstantAggregateZero::get(VectorType::get(I32, VF));
  Value *UndefVal = UndefValue::get(VTy);
  // Insert the value into a new vector.
  Value *SingleElem = Builder.CreateInsertElement(UndefVal, V, Zero);
  // Broadcast the scalar into all locations in the vector.
  Value *Shuf = Builder.CreateShuffleVector(SingleElem, UndefVal, Zeros,
                                             "broadcast");
  // We are accessing the induction variable. Make sure to promote the
  // index for each consecutive SIMD lane. This adds 0,1,2 ... to all lanes.
  if (V == Induction)
    return getConsecutiveVector(Shuf);
  return Shuf;
}

Value *SingleBlockLoopVectorizer::getConsecutiveVector(Value* Val) {
  assert(Val->getType()->isVectorTy() && "Must be a vector");
  assert(Val->getType()->getScalarType()->isIntegerTy() &&
         "Elem must be an integer");
  // Create the types.
  Type *ITy = Val->getType()->getScalarType();
  VectorType *Ty = cast<VectorType>(Val->getType());
  unsigned VLen = Ty->getNumElements();
  SmallVector<Constant*, 8> Indices;

  // Create a vector of consecutive numbers from zero to VF.
  for (unsigned i = 0; i < VLen; ++i)
    Indices.push_back(ConstantInt::get(ITy, i));

  // Add the consecutive indices to the vector value.
  Constant *Cv = ConstantVector::get(Indices);
  assert(Cv->getType() == Val->getType() && "Invalid consecutive vec");
  return Builder.CreateAdd(Val, Cv, "induction");
}

bool LoopVectorizationLegality::isConsecutiveGep(Value *Ptr) {
  GetElementPtrInst *Gep = dyn_cast_or_null<GetElementPtrInst>(Ptr);
  if (!Gep)
    return false;

  unsigned NumOperands = Gep->getNumOperands();
  Value *LastIndex = Gep->getOperand(NumOperands - 1);

  // Check that all of the gep indices are uniform except for the last.
  for (unsigned i = 0; i < NumOperands - 1; ++i)
    if (!SE->isLoopInvariant(SE->getSCEV(Gep->getOperand(i)), TheLoop))
      return false;

  // We can emit wide load/stores only of the last index is the induction
  // variable.
  const SCEV *Last = SE->getSCEV(LastIndex);
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Last)) {
    const SCEV *Step = AR->getStepRecurrence(*SE);

    // The memory is consecutive because the last index is consecutive
    // and all other indices are loop invariant.
    if (Step->isOne())
      return true;
  }

  return false;
}

bool LoopVectorizationLegality::isUniform(Value *V) {
  return (SE->isLoopInvariant(SE->getSCEV(V), TheLoop));
}

Value *SingleBlockLoopVectorizer::getVectorValue(Value *V) {
  assert(!V->getType()->isVectorTy() && "Can't widen a vector");
  // If we saved a vectorized copy of V, use it.
  Value *&MapEntry = WidenMap[V];
  if (MapEntry)
    return MapEntry;

  // Broadcast V and save the value for future uses.
  Value *B = getBroadcastInstrs(V);
  MapEntry = B;
  return B;
}

Constant*
SingleBlockLoopVectorizer::getUniformVector(unsigned Val, Type* ScalarTy) {
  SmallVector<Constant*, 8> Indices;
  // Create a vector of consecutive numbers from zero to VF.
  for (unsigned i = 0; i < VF; ++i)
    Indices.push_back(ConstantInt::get(ScalarTy, Val, true));

  // Add the consecutive indices to the vector value.
  return ConstantVector::get(Indices);
}

void SingleBlockLoopVectorizer::scalarizeInstruction(Instruction *Instr) {
  assert(!Instr->getType()->isAggregateType() && "Can't handle vectors");
  // Holds vector parameters or scalars, in case of uniform vals.
  SmallVector<Value*, 8> Params;

  // Find all of the vectorized parameters.
  for (unsigned op = 0, e = Instr->getNumOperands(); op != e; ++op) {
    Value *SrcOp = Instr->getOperand(op);

    // If we are accessing the old induction variable, use the new one.
    if (SrcOp == OldInduction) {
      Params.push_back(getBroadcastInstrs(Induction));
      continue;
    }

    // Try using previously calculated values.
    Instruction *SrcInst = dyn_cast<Instruction>(SrcOp);

    // If the src is an instruction that appeared earlier in the basic block
    // then it should already be vectorized.
    if (SrcInst && SrcInst->getParent() == Instr->getParent()) {
      assert(WidenMap.count(SrcInst) && "Source operand is unavailable");
      // The parameter is a vector value from earlier.
      Params.push_back(WidenMap[SrcInst]);
    } else {
      // The parameter is a scalar from outside the loop. Maybe even a constant.
      Params.push_back(SrcOp);
    }
  }

  assert(Params.size() == Instr->getNumOperands() &&
         "Invalid number of operands");

  // Does this instruction return a value ?
  bool IsVoidRetTy = Instr->getType()->isVoidTy();
  Value *VecResults = 0;

  // If we have a return value, create an empty vector. We place the scalarized
  // instructions in this vector.
  if (!IsVoidRetTy)
    VecResults = UndefValue::get(VectorType::get(Instr->getType(), VF));

  // For each scalar that we create:
  for (unsigned i = 0; i < VF; ++i) {
    Instruction *Cloned = Instr->clone();
    if (!IsVoidRetTy)
      Cloned->setName(Instr->getName() + ".cloned");
    // Replace the operands of the cloned instrucions with extracted scalars.
    for (unsigned op = 0, e = Instr->getNumOperands(); op != e; ++op) {
      Value *Op = Params[op];
      // Param is a vector. Need to extract the right lane.
      if (Op->getType()->isVectorTy())
        Op = Builder.CreateExtractElement(Op, Builder.getInt32(i));
      Cloned->setOperand(op, Op);
    }

    // Place the cloned scalar in the new loop.
    Builder.Insert(Cloned);

    // If the original scalar returns a value we need to place it in a vector
    // so that future users will be able to use it.
    if (!IsVoidRetTy)
      VecResults = Builder.CreateInsertElement(VecResults, Cloned,
                                               Builder.getInt32(i));
  }

  if (!IsVoidRetTy)
    WidenMap[Instr] = VecResults;
}

void
SingleBlockLoopVectorizer::createEmptyLoop(LoopVectorizationLegality *Legal) {
  /*
   In this function we generate a new loop. The new loop will contain
   the vectorized instructions while the old loop will continue to run the
   scalar remainder.

    [ ] <-- vector loop bypass.
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

  OldInduction = Legal->getInduction();
  assert(OldInduction && "We must have a single phi node.");
  Type *IdxTy = OldInduction->getType();

  // Find the loop boundaries.
  const SCEV *ExitCount = SE->getExitCount(OrigLoop, OrigLoop->getHeader());
  assert(ExitCount != SE->getCouldNotCompute() && "Invalid loop count");

  // Get the total trip count from the count by adding 1.
  ExitCount = SE->getAddExpr(ExitCount,
                             SE->getConstant(ExitCount->getType(), 1));
  // We may need to extend the index in case there is a type mismatch.
  // We know that the count starts at zero and does not overflow.
  // We are using Zext because it should be less expensive.
  if (ExitCount->getType() != IdxTy)
    ExitCount = SE->getZeroExtendExpr(ExitCount, IdxTy);

  // This is the original scalar-loop preheader.
  BasicBlock *BypassBlock = OrigLoop->getLoopPreheader();
  BasicBlock *ExitBlock = OrigLoop->getExitBlock();
  assert(ExitBlock && "Must have an exit block");

  // The loop index does not have to start at Zero. It starts with this value.
  Value *StartIdx = OldInduction->getIncomingValueForBlock(BypassBlock);

  assert(OrigLoop->getNumBlocks() == 1 && "Invalid loop");
  assert(BypassBlock && "Invalid loop structure");

  BasicBlock *VectorPH =
      BypassBlock->splitBasicBlock(BypassBlock->getTerminator(), "vector.ph");
  BasicBlock *VecBody = VectorPH->splitBasicBlock(VectorPH->getTerminator(),
                                                 "vector.body");

  BasicBlock *MiddleBlock = VecBody->splitBasicBlock(VecBody->getTerminator(),
                                                  "middle.block");
  BasicBlock *ScalarPH =
    MiddleBlock->splitBasicBlock(MiddleBlock->getTerminator(),
                                 "scalar.preheader");
  // Find the induction variable.
  BasicBlock *OldBasicBlock = OrigLoop->getHeader();

  // Use this IR builder to create the loop instructions (Phi, Br, Cmp)
  // inside the loop.
  Builder.SetInsertPoint(VecBody->getFirstInsertionPt());

  // Generate the induction variable.
  Induction = Builder.CreatePHI(IdxTy, 2, "index");
  Constant *Step = ConstantInt::get(IdxTy, VF);

  // Expand the trip count and place the new instructions in the preheader.
  // Notice that the pre-header does not change, only the loop body.
  SCEVExpander Exp(*SE, "induction");
  Instruction *Loc = BypassBlock->getTerminator();

  // Count holds the overall loop count (N).
  Value *Count = Exp.expandCodeFor(ExitCount, Induction->getType(), Loc);

  // Add the start index to the loop count to get the new end index.
  Value *IdxEnd = BinaryOperator::CreateAdd(Count, StartIdx, "end.idx", Loc);

  // Now we need to generate the expression for N - (N % VF), which is
  // the part that the vectorized body will execute.
  Constant *CIVF = ConstantInt::get(IdxTy, VF);
  Value *R = BinaryOperator::CreateURem(Count, CIVF, "n.mod.vf", Loc);
  Value *CountRoundDown = BinaryOperator::CreateSub(Count, R, "n.vec", Loc);
  Value *IdxEndRoundDown = BinaryOperator::CreateAdd(CountRoundDown, StartIdx,
                                                     "end.idx.rnd.down", Loc);

  // Now, compare the new count to zero. If it is zero, jump to the scalar part.
  Value *Cmp = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_EQ,
                               IdxEndRoundDown,
                               StartIdx,
                               "cmp.zero", Loc);

  LoopVectorizationLegality::RuntimePointerCheck *PtrRtCheck =
    Legal->getRuntimePointerCheck();
  Value *MemoryRuntimeCheck = 0;
  if (PtrRtCheck->Need) {
    unsigned NumPointers = PtrRtCheck->Pointers.size();
    SmallVector<Value* , 2> Starts;
    SmallVector<Value* , 2> Ends;

    // Use this type for pointer arithmetic.
    Type* PtrArithTy = PtrRtCheck->Pointers[0]->getType();

    for (unsigned i=0; i < NumPointers; ++i) {
      Value *Ptr = PtrRtCheck->Pointers[i];
      const SCEV *Sc = SE->getSCEV(Ptr);

      if (SE->isLoopInvariant(Sc, OrigLoop)) {
        DEBUG(dbgs() << "LV1: Adding RT check for a loop invariant ptr:" <<
              *Ptr <<"\n");
        Starts.push_back(Ptr);
        Ends.push_back(Ptr);
      } else {
        DEBUG(dbgs() << "LV: Adding RT check for range:" << *Ptr <<"\n");
        const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Sc);
        Value *Start = Exp.expandCodeFor(AR->getStart(), PtrArithTy, Loc);
        const SCEV *Ex = SE->getExitCount(OrigLoop, OrigLoop->getHeader());
        const SCEV *ScEnd = AR->evaluateAtIteration(Ex, *SE);
        assert(!isa<SCEVCouldNotCompute>(ScEnd) && "Invalid scev range.");
        Value *End = Exp.expandCodeFor(ScEnd, PtrArithTy, Loc);
        Starts.push_back(Start);
        Ends.push_back(End);
      }
    }

    for (unsigned i=0; i < NumPointers; ++i) {
      for (unsigned j=i+1; j < NumPointers; ++j) {
        Value *Cmp0 = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_ULE,
                                      Starts[0], Ends[1], "bound0", Loc);
        Value *Cmp1 = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_ULE,
                                      Starts[1], Ends[0], "bound1", Loc);
        Value *IsConflict = BinaryOperator::Create(Instruction::And, Cmp0, Cmp1,
                                                    "found.conflict", Loc);
        if (MemoryRuntimeCheck) {
          MemoryRuntimeCheck = BinaryOperator::Create(Instruction::Or,
                                                      MemoryRuntimeCheck,
                                                      IsConflict,
                                                      "conflict.rdx", Loc);
        } else {
          MemoryRuntimeCheck = IsConflict;
        }
      }
    }
  }// end of need-runtime-check code.

  // If we are using memory runtime checks, include them in.
  if (MemoryRuntimeCheck) {
    Cmp = BinaryOperator::Create(Instruction::Or, Cmp, MemoryRuntimeCheck,
                                 "CntOrMem", Loc);
  }

  BranchInst::Create(MiddleBlock, VectorPH, Cmp, Loc);
  // Remove the old terminator.
  Loc->eraseFromParent();

  // We are going to resume the execution of the scalar loop.
  // This PHI decides on what number to start. If we come from the
  // vector loop then we need to start with the end index minus the
  // index modulo VF. If we come from a bypass edge then we need to start
  // from the real start.
  PHINode* ResumeIndex = PHINode::Create(IdxTy, 2, "resume.idx",
                                         MiddleBlock->getTerminator());
  ResumeIndex->addIncoming(StartIdx, BypassBlock);
  ResumeIndex->addIncoming(IdxEndRoundDown, VecBody);

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

  // Fix the scalar body iteration count.
  unsigned BlockIdx = OldInduction->getBasicBlockIndex(ScalarPH);
  OldInduction->setIncomingValue(BlockIdx, ResumeIndex);

  // Get ready to start creating new instructions into the vectorized body.
  Builder.SetInsertPoint(VecBody->getFirstInsertionPt());

  // Register the new loop.
  Loop* Lp = new Loop();
  LPM->insertLoop(Lp, OrigLoop->getParentLoop());

  Lp->addBasicBlockToLoop(VecBody, LI->getBase());

  Loop *ParentLoop = OrigLoop->getParentLoop();
  if (ParentLoop) {
    ParentLoop->addBasicBlockToLoop(ScalarPH, LI->getBase());
    ParentLoop->addBasicBlockToLoop(VectorPH, LI->getBase());
    ParentLoop->addBasicBlockToLoop(MiddleBlock, LI->getBase());
  }

  // Save the state.
  LoopVectorPreHeader = VectorPH;
  LoopScalarPreHeader = ScalarPH;
  LoopMiddleBlock = MiddleBlock;
  LoopExitBlock = ExitBlock;
  LoopVectorBody = VecBody;
  LoopScalarBody = OldBasicBlock;
  LoopBypassBlock = BypassBlock;
}

/// This function returns the identity element (or neutral element) for
/// the operation K.
static unsigned
getReductionIdentity(LoopVectorizationLegality::ReductionKind K) {
  switch (K) {
  case LoopVectorizationLegality::IntegerXor:
  case LoopVectorizationLegality::IntegerAdd:
  case LoopVectorizationLegality::IntegerOr:
    // Adding, Xoring, Oring zero to a number does not change it.
    return 0;
  case LoopVectorizationLegality::IntegerMult:
    // Multiplying a number by 1 does not change it.
    return 1;
  case LoopVectorizationLegality::IntegerAnd:
    // AND-ing a number with an all-1 value does not change it.
    return -1;
  default:
    llvm_unreachable("Unknown reduction kind");
  }
}

void
SingleBlockLoopVectorizer::vectorizeLoop(LoopVectorizationLegality *Legal) {
  //===------------------------------------------------===//
  //
  // Notice: any optimization or new instruction that go
  // into the code below should be also be implemented in
  // the cost-model.
  //
  //===------------------------------------------------===//
  typedef SmallVector<PHINode*, 4> PhiVector;
  BasicBlock &BB = *OrigLoop->getHeader();
  Constant *Zero = ConstantInt::get(
    IntegerType::getInt32Ty(BB.getContext()), 0);

  // In order to support reduction variables we need to be able to vectorize
  // Phi nodes. Phi nodes have cycles, so we need to vectorize them in two
  // steages. First, we create a new vector PHI node with no incoming edges.
  // We use this value when we vectorize all of the instructions that use the
  // PHI. Next, after all of the instructions in the block are complete we
  // add the new incoming edges to the PHI. At this point all of the
  // instructions in the basic block are vectorized, so we can use them to
  // construct the PHI.
  PhiVector PHIsToFix;

  // For each instruction in the old loop.
  for (BasicBlock::iterator it = BB.begin(), e = BB.end(); it != e; ++it) {
    Instruction *Inst = it;

    switch (Inst->getOpcode()) {
      case Instruction::Br:
        // Nothing to do for PHIs and BR, since we already took care of the
        // loop control flow instructions.
        continue;
      case Instruction::PHI:{
        PHINode* P = cast<PHINode>(Inst);
        // Special handling for the induction var.
        if (OldInduction == Inst)
          continue;
        // This is phase one of vectorizing PHIs.
        // This has to be a reduction variable.
        assert(Legal->getReductionVars()->count(P) && "Not a Reduction");
        Type *VecTy = VectorType::get(Inst->getType(), VF);
        WidenMap[Inst] = Builder.CreatePHI(VecTy, 2, "vec.phi");
        PHIsToFix.push_back(P);
        continue;
      }
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
        BinaryOperator *BinOp = dyn_cast<BinaryOperator>(Inst);
        Value *A = getVectorValue(Inst->getOperand(0));
        Value *B = getVectorValue(Inst->getOperand(1));

        // Use this vector value for all users of the original instruction.
        Value *V = Builder.CreateBinOp(BinOp->getOpcode(), A, B);
        WidenMap[Inst] = V;

        // Update the NSW, NUW and Exact flags.
        BinaryOperator *VecOp = cast<BinaryOperator>(V);
        if (isa<OverflowingBinaryOperator>(BinOp)) {
          VecOp->setHasNoSignedWrap(BinOp->hasNoSignedWrap());
          VecOp->setHasNoUnsignedWrap(BinOp->hasNoUnsignedWrap());
        }
        if (isa<PossiblyExactOperator>(VecOp))
          VecOp->setIsExact(BinOp->isExact());
        break;
      }
      case Instruction::Select: {
        // Widen selects.
        // If the selector is loop invariant we can create a select
        // instruction with a scalar condition. Otherwise, use vector-select.
        Value *Cond = Inst->getOperand(0);
        bool InvariantCond = SE->isLoopInvariant(SE->getSCEV(Cond), OrigLoop);

        // The condition can be loop invariant  but still defined inside the
        // loop. This means that we can't just use the original 'cond' value.
        // We have to take the 'vectorized' value and pick the first lane.
        // Instcombine will make this a no-op.
        Cond = getVectorValue(Cond);
        if (InvariantCond)
          Cond = Builder.CreateExtractElement(Cond, Builder.getInt32(0));

        Value *Op0 = getVectorValue(Inst->getOperand(1));
        Value *Op1 = getVectorValue(Inst->getOperand(2));
        WidenMap[Inst] = Builder.CreateSelect(Cond, Op0, Op1);
        break;
      }

      case Instruction::ICmp:
      case Instruction::FCmp: {
        // Widen compares. Generate vector compares.
        bool FCmp = (Inst->getOpcode() == Instruction::FCmp);
        CmpInst *Cmp = dyn_cast<CmpInst>(Inst);
        Value *A = getVectorValue(Inst->getOperand(0));
        Value *B = getVectorValue(Inst->getOperand(1));
        if (FCmp)
          WidenMap[Inst] = Builder.CreateFCmp(Cmp->getPredicate(), A, B);
        else
          WidenMap[Inst] = Builder.CreateICmp(Cmp->getPredicate(), A, B);
        break;
      }

      case Instruction::Store: {
        // Attempt to issue a wide store.
        StoreInst *SI = dyn_cast<StoreInst>(Inst);
        Type *StTy = VectorType::get(SI->getValueOperand()->getType(), VF);
        Value *Ptr = SI->getPointerOperand();
        unsigned Alignment = SI->getAlignment();

        assert(!Legal->isUniform(Ptr) &&
               "We do not allow storing to uniform addresses");

        GetElementPtrInst *Gep = dyn_cast<GetElementPtrInst>(Ptr);

        // This store does not use GEPs.
        if (!Legal->isConsecutiveGep(Gep)) {
          scalarizeInstruction(Inst);
          break;
        }

        // The last index does not have to be the induction. It can be
        // consecutive and be a function of the index. For example A[I+1];
        unsigned NumOperands = Gep->getNumOperands();
        Value *LastIndex = getVectorValue(Gep->getOperand(NumOperands - 1));
        LastIndex = Builder.CreateExtractElement(LastIndex, Zero);

        // Create the new GEP with the new induction variable.
        GetElementPtrInst *Gep2 = cast<GetElementPtrInst>(Gep->clone());
        Gep2->setOperand(NumOperands - 1, LastIndex);
        Ptr = Builder.Insert(Gep2);
        Ptr = Builder.CreateBitCast(Ptr, StTy->getPointerTo());
        Value *Val = getVectorValue(SI->getValueOperand());
        Builder.CreateStore(Val, Ptr)->setAlignment(Alignment);
        break;
      }
      case Instruction::Load: {
        // Attempt to issue a wide load.
        LoadInst *LI = dyn_cast<LoadInst>(Inst);
        Type *RetTy = VectorType::get(LI->getType(), VF);
        Value *Ptr = LI->getPointerOperand();
        unsigned Alignment = LI->getAlignment();
        GetElementPtrInst *Gep = dyn_cast<GetElementPtrInst>(Ptr);

        // If we don't have a gep, or that the pointer is loop invariant,
        // scalarize the load.
        if (!Gep || Legal->isUniform(Gep) || !Legal->isConsecutiveGep(Gep)) {
          scalarizeInstruction(Inst);
          break;
        }

        // The last index does not have to be the induction. It can be
        // consecutive and be a function of the index. For example A[I+1];
        unsigned NumOperands = Gep->getNumOperands();
        Value *LastIndex = getVectorValue(Gep->getOperand(NumOperands -1));
        LastIndex = Builder.CreateExtractElement(LastIndex, Zero);

        // Create the new GEP with the new induction variable.
        GetElementPtrInst *Gep2 = cast<GetElementPtrInst>(Gep->clone());
        Gep2->setOperand(NumOperands - 1, LastIndex);
        Ptr = Builder.Insert(Gep2);
        Ptr = Builder.CreateBitCast(Ptr, RetTy->getPointerTo());
        LI = Builder.CreateLoad(Ptr);
        LI->setAlignment(Alignment);
        // Use this vector value for all users of the load.
        WidenMap[Inst] = LI;
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
        /// Vectorize bitcasts.
        CastInst *CI = dyn_cast<CastInst>(Inst);
        Value *A = getVectorValue(Inst->getOperand(0));
        Type *DestTy = VectorType::get(CI->getType()->getScalarType(), VF);
        WidenMap[Inst] = Builder.CreateCast(CI->getOpcode(), A, DestTy);
        break;
      }

      default:
        /// All other instructions are unsupported. Scalarize them.
        scalarizeInstruction(Inst);
        break;
    }// end of switch.
  }// end of for_each instr.

  // At this point every instruction in the original loop is widended to
  // a vector form. We are almost done. Now, we need to fix the PHI nodes
  // that we vectorized. The PHI nodes are currently empty because we did
  // not want to introduce cycles. Notice that the remaining PHI nodes
  // that we need to fix are reduction variables.

  // Create the 'reduced' values for each of the induction vars.
  // The reduced values are the vector values that we scalarize and combine
  // after the loop is finished.
  for (PhiVector::iterator it = PHIsToFix.begin(), e = PHIsToFix.end();
       it != e; ++it) {
    PHINode *RdxPhi = *it;
    PHINode *VecRdxPhi = dyn_cast<PHINode>(WidenMap[RdxPhi]);
    assert(RdxPhi && "Unable to recover vectorized PHI");

    // Find the reduction variable descriptor.
    assert(Legal->getReductionVars()->count(RdxPhi) &&
           "Unable to find the reduction variable");
    LoopVectorizationLegality::ReductionDescriptor RdxDesc =
      (*Legal->getReductionVars())[RdxPhi];

    // We need to generate a reduction vector from the incoming scalar.
    // To do so, we need to generate the 'identity' vector and overide
    // one of the elements with the incoming scalar reduction. We need
    // to do it in the vector-loop preheader.
    Builder.SetInsertPoint(LoopBypassBlock->getTerminator());

    // This is the vector-clone of the value that leaves the loop.
    Value *VectorExit = getVectorValue(RdxDesc.LoopExitInstr);
    Type *VecTy = VectorExit->getType();

    // Find the reduction identity variable. Zero for addition, or, xor,
    // one for multiplication, -1 for And.
    Constant *Identity = getUniformVector(getReductionIdentity(RdxDesc.Kind),
                                          VecTy->getScalarType());

    // This vector is the Identity vector where the first element is the
    // incoming scalar reduction.
    Value *VectorStart = Builder.CreateInsertElement(Identity,
                                                    RdxDesc.StartValue, Zero);


    // Fix the vector-loop phi.
    // We created the induction variable so we know that the
    // preheader is the first entry.
    BasicBlock *VecPreheader = Induction->getIncomingBlock(0);

    // Reductions do not have to start at zero. They can start with
    // any loop invariant values.
    VecRdxPhi->addIncoming(VectorStart, VecPreheader);
    unsigned SelfEdgeIdx = (RdxPhi)->getBasicBlockIndex(LoopScalarBody);
    Value *Val = getVectorValue(RdxPhi->getIncomingValue(SelfEdgeIdx));
    VecRdxPhi->addIncoming(Val, LoopVectorBody);

    // Before each round, move the insertion point right between
    // the PHIs and the values we are going to write.
    // This allows us to write both PHINodes and the extractelement
    // instructions.
    Builder.SetInsertPoint(LoopMiddleBlock->getFirstInsertionPt());

    // This PHINode contains the vectorized reduction variable, or
    // the initial value vector, if we bypass the vector loop.
    PHINode *NewPhi = Builder.CreatePHI(VecTy, 2, "rdx.vec.exit.phi");
    NewPhi->addIncoming(VectorStart, LoopBypassBlock);
    NewPhi->addIncoming(getVectorValue(RdxDesc.LoopExitInstr), LoopVectorBody);

    // Extract the first scalar.
    Value *Scalar0 =
      Builder.CreateExtractElement(NewPhi, Builder.getInt32(0));
    // Extract and reduce the remaining vector elements.
    for (unsigned i=1; i < VF; ++i) {
      Value *Scalar1 =
        Builder.CreateExtractElement(NewPhi, Builder.getInt32(i));
      switch (RdxDesc.Kind) {
        case LoopVectorizationLegality::IntegerAdd:
          Scalar0 = Builder.CreateAdd(Scalar0, Scalar1);
          break;
        case LoopVectorizationLegality::IntegerMult:
          Scalar0 = Builder.CreateMul(Scalar0, Scalar1);
          break;
        case LoopVectorizationLegality::IntegerOr:
          Scalar0 = Builder.CreateOr(Scalar0, Scalar1);
          break;
        case LoopVectorizationLegality::IntegerAnd:
          Scalar0 = Builder.CreateAnd(Scalar0, Scalar1);
          break;
        case LoopVectorizationLegality::IntegerXor:
          Scalar0 = Builder.CreateXor(Scalar0, Scalar1);
          break;
        default:
          llvm_unreachable("Unknown reduction operation");
      }
    }

    // Now, we need to fix the users of the reduction variable
    // inside and outside of the scalar remainder loop.
    // We know that the loop is in LCSSA form. We need to update the
    // PHI nodes in the exit blocks.
    for (BasicBlock::iterator LEI = LoopExitBlock->begin(),
         LEE = LoopExitBlock->end(); LEI != LEE; ++LEI) {
      PHINode *LCSSAPhi = dyn_cast<PHINode>(LEI);
      if (!LCSSAPhi) continue;

      // All PHINodes need to have a single entry edge, or two if
      // we already fixed them.
      assert(LCSSAPhi->getNumIncomingValues() < 3 && "Invalid LCSSA PHI");

      // We found our reduction value exit-PHI. Update it with the
      // incoming bypass edge.
      if (LCSSAPhi->getIncomingValue(0) == RdxDesc.LoopExitInstr) {
        // Add an edge coming from the bypass.
        LCSSAPhi->addIncoming(Scalar0, LoopMiddleBlock);
        break;
      }
    }// end of the LCSSA phi scan.

    // Fix the scalar loop reduction variable with the incoming reduction sum
    // from the vector body and from the backedge value.
    int IncomingEdgeBlockIdx = (RdxPhi)->getBasicBlockIndex(LoopScalarBody);
    int SelfEdgeBlockIdx = (IncomingEdgeBlockIdx ? 0 : 1); // The other block.
    (RdxPhi)->setIncomingValue(SelfEdgeBlockIdx, Scalar0);
    (RdxPhi)->setIncomingValue(IncomingEdgeBlockIdx, RdxDesc.LoopExitInstr);
  }// end of for each redux variable.
}

void SingleBlockLoopVectorizer::updateAnalysis() {
  // The original basic block.
  SE->forgetLoop(OrigLoop);

  // Update the dominator tree information.
  assert(DT->properlyDominates(LoopBypassBlock, LoopExitBlock) &&
         "Entry does not dominate exit.");

  DT->addNewBlock(LoopVectorPreHeader, LoopBypassBlock);
  DT->addNewBlock(LoopVectorBody, LoopVectorPreHeader);
  DT->addNewBlock(LoopMiddleBlock, LoopBypassBlock);
  DT->addNewBlock(LoopScalarPreHeader, LoopMiddleBlock);
  DT->changeImmediateDominator(LoopScalarBody, LoopScalarPreHeader);
  DT->changeImmediateDominator(LoopExitBlock, LoopMiddleBlock);

  DEBUG(DT->verifyAnalysis());
}

bool LoopVectorizationLegality::canVectorize() {
  if (!TheLoop->getLoopPreheader()) {
    assert(false && "No preheader!!");
    DEBUG(dbgs() << "LV: Loop not normalized." << "\n");
    return  false;
  }

  // We can only vectorize single basic block loops.
  unsigned NumBlocks = TheLoop->getNumBlocks();
  if (NumBlocks != 1) {
    DEBUG(dbgs() << "LV: Too many blocks:" << NumBlocks << "\n");
    return false;
  }

  // We need to have a loop header.
  BasicBlock *BB = TheLoop->getHeader();
  DEBUG(dbgs() << "LV: Found a loop: " << BB->getName() << "\n");

  // ScalarEvolution needs to be able to find the exit count.
  const SCEV *ExitCount = SE->getExitCount(TheLoop, BB);
  if (ExitCount == SE->getCouldNotCompute()) {
    DEBUG(dbgs() << "LV: SCEV could not compute the loop exit count.\n");
    return false;
  }

  // Do not loop-vectorize loops with a tiny trip count.
  unsigned TC = SE->getSmallConstantTripCount(TheLoop, BB);
  if (TC > 0u && TC < TinyTripCountThreshold) {
    DEBUG(dbgs() << "LV: Found a loop with a very small trip count. " <<
          "This loop is not worth vectorizing.\n");
    return false;
  }

  // Go over each instruction and look at memory deps.
  if (!canVectorizeBlock(*BB)) {
    DEBUG(dbgs() << "LV: Can't vectorize this loop header\n");
    return false;
  }

  DEBUG(dbgs() << "LV: We can vectorize this loop" <<
        (PtrRtCheck.Need ? " (with a runtime bound check)" : "")
        <<"!\n");

  // Okay! We can vectorize. At this point we don't have any other mem analysis
  // which may limit our maximum vectorization factor, so just return true with
  // no restrictions.
  return true;
}

bool LoopVectorizationLegality::canVectorizeBlock(BasicBlock &BB) {
  // Scan the instructions in the block and look for hazards.
  for (BasicBlock::iterator it = BB.begin(), e = BB.end(); it != e; ++it) {
    Instruction *I = it;

    PHINode *Phi = dyn_cast<PHINode>(I);
    if (Phi) {
      // This should not happen because the loop should be normalized.
      if (Phi->getNumIncomingValues() != 2) {
        DEBUG(dbgs() << "LV: Found an invalid PHI.\n");
        return false;
      }
      // We only look at integer phi nodes.
      if (!Phi->getType()->isIntegerTy()) {
        DEBUG(dbgs() << "LV: Found an non-int PHI.\n");
        return false;
      }

      if (isInductionVariable(Phi)) {
        if (Induction) {
          DEBUG(dbgs() << "LV: Found too many inductions."<< *Phi <<"\n");
          return false;
        }
        DEBUG(dbgs() << "LV: Found the induction PHI."<< *Phi <<"\n");
        Induction = Phi;
        continue;
      }
      if (AddReductionVar(Phi, IntegerAdd)) {
        DEBUG(dbgs() << "LV: Found an ADD reduction PHI."<< *Phi <<"\n");
        continue;
      }
      if (AddReductionVar(Phi, IntegerMult)) {
        DEBUG(dbgs() << "LV: Found a MUL reduction PHI."<< *Phi <<"\n");
        continue;
      }
      if (AddReductionVar(Phi, IntegerOr)) {
        DEBUG(dbgs() << "LV: Found an OR reduction PHI."<< *Phi <<"\n");
        continue;
      }
      if (AddReductionVar(Phi, IntegerAnd)) {
        DEBUG(dbgs() << "LV: Found an AND reduction PHI."<< *Phi <<"\n");
        continue;
      }
      if (AddReductionVar(Phi, IntegerXor)) {
        DEBUG(dbgs() << "LV: Found a XOR reduction PHI."<< *Phi <<"\n");
        continue;
      }

      DEBUG(dbgs() << "LV: Found an unidentified PHI."<< *Phi <<"\n");
      return false;
    }// end of PHI handling

    // We still don't handle functions.
    CallInst *CI = dyn_cast<CallInst>(I);
    if (CI) {
      DEBUG(dbgs() << "LV: Found a call site.\n");
      return false;
    }

    // We do not re-vectorize vectors.
    if (!VectorType::isValidElementType(I->getType()) &&
        !I->getType()->isVoidTy()) {
      DEBUG(dbgs() << "LV: Found unvectorizable type." << "\n");
      return false;
    }

    // Reduction instructions are allowed to have exit users.
    // All other instructions must not have external users.
    if (!AllowedExit.count(I))
      //Check that all of the users of the loop are inside the BB.
      for (Value::use_iterator it = I->use_begin(), e = I->use_end();
           it != e; ++it) {
        Instruction *U = cast<Instruction>(*it);
        // This user may be a reduction exit value.
        BasicBlock *Parent = U->getParent();
        if (Parent != &BB) {
          DEBUG(dbgs() << "LV: Found an outside user for : "<< *U << "\n");
          return false;
        }
    }
  } // next instr.

  if (!Induction) {
      DEBUG(dbgs() << "LV: Did not find an induction var.\n");
      return false;
  }

  // Don't vectorize if the memory dependencies do not allow vectorization.
  if (!canVectorizeMemory(BB))
    return false;

  // We now know that the loop is vectorizable!
  // Collect variables that will remain uniform after vectorization.
  std::vector<Value*> Worklist;

  // Start with the conditional branch and walk up the block.
  Worklist.push_back(BB.getTerminator()->getOperand(0));

  while (Worklist.size()) {
    Instruction *I = dyn_cast<Instruction>(Worklist.back());
    Worklist.pop_back();
    // Look at instructions inside this block.
    if (!I) continue;
    if (I->getParent() != &BB) continue;

    // Stop when reaching PHI nodes.
    if (isa<PHINode>(I)) {
      assert(I == Induction && "Found a uniform PHI that is not the induction");
      break;
    }

    // This is a known uniform.
    Uniforms.insert(I);

    // Insert all operands.
    for (int i=0, Op = I->getNumOperands(); i < Op; ++i) {
      Worklist.push_back(I->getOperand(i));
    }
  }

  return true;
}

bool LoopVectorizationLegality::canVectorizeMemory(BasicBlock &BB) {
  typedef SmallVector<Value*, 16> ValueVector;
  typedef SmallPtrSet<Value*, 16> ValueSet;
  // Holds the Load and Store *instructions*.
  ValueVector Loads;
  ValueVector Stores;
  PtrRtCheck.Pointers.clear();
  PtrRtCheck.Need = false;

  // Scan the BB and collect legal loads and stores.
  for (BasicBlock::iterator it = BB.begin(), e = BB.end(); it != e; ++it) {
    Instruction *I = it;

    // If this is a load, save it. If this instruction can read from memory
    // but is not a load, then we quit. Notice that we don't handle function
    // calls that read or write.
    if (I->mayReadFromMemory()) {
      LoadInst *Ld = dyn_cast<LoadInst>(I);
      if (!Ld) return false;
      if (!Ld->isSimple()) {
        DEBUG(dbgs() << "LV: Found a non-simple load.\n");
        return false;
      }
      Loads.push_back(Ld);
      continue;
    }

    // Save store instructions. Abort if other instructions write to memory.
    if (I->mayWriteToMemory()) {
      StoreInst *St = dyn_cast<StoreInst>(I);
      if (!St) return false;
      if (!St->isSimple()) {
        DEBUG(dbgs() << "LV: Found a non-simple store.\n");
        return false;
      }
      Stores.push_back(St);
    }
  } // next instr.

  // Now we have two lists that hold the loads and the stores.
  // Next, we find the pointers that they use.

  // Check if we see any stores. If there are no stores, then we don't
  // care if the pointers are *restrict*.
  if (!Stores.size()) {
        DEBUG(dbgs() << "LV: Found a read-only loop!\n");
        return true;
  }

  // Holds the read and read-write *pointers* that we find.
  ValueVector Reads;
  ValueVector ReadWrites;

  // Holds the analyzed pointers. We don't want to call GetUnderlyingObjects
  // multiple times on the same object. If the ptr is accessed twice, once
  // for read and once for write, it will only appear once (on the write
  // list). This is okay, since we are going to check for conflicts between
  // writes and between reads and writes, but not between reads and reads.
  ValueSet Seen;

  ValueVector::iterator I, IE;
  for (I = Stores.begin(), IE = Stores.end(); I != IE; ++I) {
    StoreInst *ST = dyn_cast<StoreInst>(*I);
    assert(ST && "Bad StoreInst");
    Value* Ptr = ST->getPointerOperand();

    if (isUniform(Ptr)) {
      DEBUG(dbgs() << "LV: We don't allow storing to uniform addresses\n");
      return false;
    }

    // If we did *not* see this pointer before, insert it to
    // the read-write list. At this phase it is only a 'write' list.
    if (Seen.insert(Ptr))
      ReadWrites.push_back(Ptr);
  }

  for (I = Loads.begin(), IE = Loads.end(); I != IE; ++I) {
    LoadInst *LD = dyn_cast<LoadInst>(*I);
    assert(LD && "Bad LoadInst");
    Value* Ptr = LD->getPointerOperand();
    // If we did *not* see this pointer before, insert it to the
    // read list. If we *did* see it before, then it is already in
    // the read-write list. This allows us to vectorize expressions
    // such as A[i] += x;  Because the address of A[i] is a read-write
    // pointer. This only works if the index of A[i] is consecutive.
    // If the address of i is unknown (for example A[B[i]]) then we may
    // read a few words, modify, and write a few words, and some of the
    // words may be written to the same address.
    if (Seen.insert(Ptr) || !isConsecutiveGep(Ptr))
      Reads.push_back(Ptr);
  }

  // If we write (or read-write) to a single destination and there are no
  // other reads in this loop then is it safe to vectorize.
  if (ReadWrites.size() == 1 && Reads.size() == 0) {
    DEBUG(dbgs() << "LV: Found a write-only loop!\n");
    return true;
  }

  // Find pointers with computable bounds. We are going to use this information
  // to place a runtime bound check.
  bool RT = true;
  for (I = ReadWrites.begin(), IE = ReadWrites.end(); I != IE; ++I)
    if (hasComputableBounds(*I)) {
      PtrRtCheck.Pointers.push_back(*I);
      DEBUG(dbgs() << "LV: Found a runtime check ptr:" << **I <<"\n");
    } else {
      RT = false;
      break;
    }
  for (I = Reads.begin(), IE = Reads.end(); I != IE; ++I)
    if (hasComputableBounds(*I)) {
      PtrRtCheck.Pointers.push_back(*I);
      DEBUG(dbgs() << "LV: Found a runtime check ptr:" << **I <<"\n");
    } else {
      RT = false;
      break;
    }

  // Check that we did not collect too many pointers or found a
  // unsizeable pointer.
  if (!RT || PtrRtCheck.Pointers.size() > RuntimeMemoryCheckThreshold) {
    PtrRtCheck.Pointers.clear();
    RT = false;
  }

  PtrRtCheck.Need = RT;

  if (RT) {
    DEBUG(dbgs() << "LV: We can perform a memory runtime check if needed.\n");
  }

  // Now that the pointers are in two lists (Reads and ReadWrites), we
  // can check that there are no conflicts between each of the writes and
  // between the writes to the reads.
  ValueSet WriteObjects;
  ValueVector TempObjects;

  // Check that the read-writes do not conflict with other read-write
  // pointers.
  for (I = ReadWrites.begin(), IE = ReadWrites.end(); I != IE; ++I) {
    GetUnderlyingObjects(*I, TempObjects, DL);
    for (ValueVector::iterator it=TempObjects.begin(), e=TempObjects.end();
         it != e; ++it) {
      if (!isIdentifiedObject(*it)) {
        DEBUG(dbgs() << "LV: Found an unidentified write ptr:"<< **it <<"\n");
        return RT;
      }
      if (!WriteObjects.insert(*it)) {
        DEBUG(dbgs() << "LV: Found a possible write-write reorder:"
              << **it <<"\n");
        return RT;
      }
    }
    TempObjects.clear();
  }

  /// Check that the reads don't conflict with the read-writes.
  for (I = Reads.begin(), IE = Reads.end(); I != IE; ++I) {
    GetUnderlyingObjects(*I, TempObjects, DL);
    for (ValueVector::iterator it=TempObjects.begin(), e=TempObjects.end();
         it != e; ++it) {
      if (!isIdentifiedObject(*it)) {
        DEBUG(dbgs() << "LV: Found an unidentified read ptr:"<< **it <<"\n");
        return RT;
      }
      if (WriteObjects.count(*it)) {
        DEBUG(dbgs() << "LV: Found a possible read/write reorder:"
              << **it <<"\n");
        return RT;
      }
    }
    TempObjects.clear();
  }

  // It is safe to vectorize and we don't need any runtime checks.
  DEBUG(dbgs() << "LV: We don't need a runtime memory check.\n");
  PtrRtCheck.Pointers.clear();
  PtrRtCheck.Need = false;
  return true;
}

bool LoopVectorizationLegality::AddReductionVar(PHINode *Phi,
                                                ReductionKind Kind) {
  if (Phi->getNumIncomingValues() != 2)
    return false;

  // Find the possible incoming reduction variable.
  BasicBlock *BB = Phi->getParent();
  int SelfEdgeIdx = Phi->getBasicBlockIndex(BB);
  int InEdgeBlockIdx = (SelfEdgeIdx ? 0 : 1); // The other entry.
  Value *RdxStart = Phi->getIncomingValue(InEdgeBlockIdx);

  // ExitInstruction is the single value which is used outside the loop.
  // We only allow for a single reduction value to be used outside the loop.
  // This includes users of the reduction, variables (which form a cycle
  // which ends in the phi node).
  Instruction *ExitInstruction = 0;

  // Iter is our iterator. We start with the PHI node and scan for all of the
  // users of this instruction. All users must be instructions which can be
  // used as reduction variables (such as ADD). We may have a single
  // out-of-block user. They cycle must end with the original PHI.
  // Also, we can't have multiple block-local users.
  Instruction *Iter = Phi;
  while (true) {
    // Any reduction instr must be of one of the allowed kinds.
    if (!isReductionInstr(Iter, Kind))
      return false;

    // Did we found a user inside this block ?
    bool FoundInBlockUser = false;
    // Did we reach the initial PHI node ?
    bool FoundStartPHI = false;

    // If the instruction has no users then this is a broken
    // chain and can't be a reduction variable.
    if (Iter->use_empty())
      return false;

    // For each of the *users* of iter.
    for (Value::use_iterator it = Iter->use_begin(), e = Iter->use_end();
         it != e; ++it) {
      Instruction *U = cast<Instruction>(*it);
      // We already know that the PHI is a user.
      if (U == Phi) {
        FoundStartPHI = true;
        continue;
      }
      // Check if we found the exit user.
      BasicBlock *Parent = U->getParent();
      if (Parent != BB) {
        // We must have a single exit instruction.
        if (ExitInstruction != 0)
          return false;
        ExitInstruction = Iter;
      }
      // We can't have multiple inside users.
      if (FoundInBlockUser)
        return false;
      FoundInBlockUser = true;
      Iter = U;
    }

    // We found a reduction var if we have reached the original
    // phi node and we only have a single instruction with out-of-loop
    // users.
   if (FoundStartPHI && ExitInstruction) {
     // This instruction is allowed to have out-of-loop users.
     AllowedExit.insert(ExitInstruction);

     // Save the description of this reduction variable.
     ReductionDescriptor RD(RdxStart, ExitInstruction, Kind);
     Reductions[Phi] = RD;
     return true;
   }
  }
}

bool
LoopVectorizationLegality::isReductionInstr(Instruction *I,
                                            ReductionKind Kind) {
    switch (I->getOpcode()) {
    default:
      return false;
    case Instruction::PHI:
      // possibly.
      return true;
    case Instruction::Add:
    case Instruction::Sub:
      return Kind == IntegerAdd;
    case Instruction::Mul:
    case Instruction::UDiv:
    case Instruction::SDiv:
      return Kind == IntegerMult;
    case Instruction::And:
      return Kind == IntegerAnd;
    case Instruction::Or:
      return Kind == IntegerOr;
    case Instruction::Xor:
      return Kind == IntegerXor;
    }
}

bool LoopVectorizationLegality::isInductionVariable(PHINode *Phi) {
  // Check that the PHI is consecutive and starts at zero.
  const SCEV *PhiScev = SE->getSCEV(Phi);
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PhiScev);
  if (!AR) {
    DEBUG(dbgs() << "LV: PHI is not a poly recurrence.\n");
    return false;
  }
  const SCEV *Step = AR->getStepRecurrence(*SE);

  if (!Step->isOne()) {
    DEBUG(dbgs() << "LV: PHI stride does not equal one.\n");
    return false;
  }
  return true;
}

bool LoopVectorizationLegality::hasComputableBounds(Value *Ptr) {
  const SCEV *PhiScev = SE->getSCEV(Ptr);
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PhiScev);
  if (!AR)
    return false;

  return AR->isAffine();
}

unsigned
LoopVectorizationCostModel::findBestVectorizationFactor(unsigned VF) {
  if (!VTTI) {
    DEBUG(dbgs() << "LV: No vector target information. Not vectorizing. \n");
    return 1;
  }

  float Cost = expectedCost(1);
  unsigned Width = 1;
  DEBUG(dbgs() << "LV: Scalar loop costs: "<< (int)Cost << ".\n");
  for (unsigned i=2; i <= VF; i*=2) {
    // Notice that the vector loop needs to be executed less times, so
    // we need to divide the cost of the vector loops by the width of
    // the vector elements.
    float VectorCost = expectedCost(i) / (float)i;
    DEBUG(dbgs() << "LV: Vector loop of width "<< i << " costs: " <<
          (int)VectorCost << ".\n");
    if (VectorCost < Cost) {
      Cost = VectorCost;
      Width = i;
    }
  }

  DEBUG(dbgs() << "LV: Selecting VF = : "<< Width << ".\n");
  return Width;
}

unsigned LoopVectorizationCostModel::expectedCost(unsigned VF) {
  // We can only estimate the cost of single basic block loops.
  assert(1 == TheLoop->getNumBlocks() && "Too many blocks in loop");

  BasicBlock *BB = TheLoop->getHeader();
  unsigned Cost = 0;

  // For each instruction in the old loop.
  for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
    Instruction *Inst = it;
    unsigned C = getInstructionCost(Inst, VF);
    Cost += C;
    DEBUG(dbgs() << "LV: Found an estimated cost of "<< C <<" for VF "<< VF <<
          " For instruction: "<< *Inst << "\n");
  }

  return Cost;
}

unsigned
LoopVectorizationCostModel::getInstructionCost(Instruction *I, unsigned VF) {
  assert(VTTI && "Invalid vector target transformation info");

  // If we know that this instruction will remain uniform, check the cost of
  // the scalar version.
  if (Legal->isUniformAfterVectorization(I))
    VF = 1;

  Type *RetTy = I->getType();
  Type *VectorTy = ToVectorTy(RetTy, VF);


  // TODO: We need to estimate the cost of intrinsic calls.
  switch (I->getOpcode()) {
    case Instruction::GetElementPtr:
      // We mark this instruction as zero-cost because scalar GEPs are usually
      // lowered to the intruction addressing mode. At the moment we don't
      // generate vector geps.
      return 0;
    case Instruction::Br: {
      return VTTI->getCFInstrCost(I->getOpcode());
    }
    case Instruction::PHI:
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
      return VTTI->getArithmeticInstrCost(I->getOpcode(), VectorTy);
    }
    case Instruction::Select: {
      SelectInst *SI = cast<SelectInst>(I);
      const SCEV *CondSCEV = SE->getSCEV(SI->getCondition());
      bool ScalarCond = (SE->isLoopInvariant(CondSCEV, TheLoop));
      Type *CondTy = SI->getCondition()->getType();
      if (ScalarCond)
        CondTy = VectorType::get(CondTy, VF);

      return VTTI->getCmpSelInstrCost(I->getOpcode(), VectorTy, CondTy);
    }
    case Instruction::ICmp:
    case Instruction::FCmp: {
      Type *ValTy = I->getOperand(0)->getType();
      VectorTy = ToVectorTy(ValTy, VF);
      return VTTI->getCmpSelInstrCost(I->getOpcode(), VectorTy);
    }
    case Instruction::Store: {
      StoreInst *SI = cast<StoreInst>(I);
      Type *ValTy = SI->getValueOperand()->getType();
      VectorTy = ToVectorTy(ValTy, VF);

      if (VF == 1)
        return VTTI->getMemoryOpCost(I->getOpcode(), ValTy,
                              SI->getAlignment(), SI->getPointerAddressSpace());

      // Scalarized stores.
      if (!Legal->isConsecutiveGep(SI->getPointerOperand())) {
        unsigned Cost = 0;
        unsigned ExtCost = VTTI->getInstrCost(Instruction::ExtractElement,
                                              ValTy);
        // The cost of extracting from the value vector.
        Cost += VF * (ExtCost);
        // The cost of the scalar stores.
        Cost += VF * VTTI->getMemoryOpCost(I->getOpcode(),
                                           ValTy->getScalarType(),
                                           SI->getAlignment(),
                                           SI->getPointerAddressSpace());
        return Cost;
      }

      // Wide stores.
      return VTTI->getMemoryOpCost(I->getOpcode(), VectorTy, SI->getAlignment(),
                                   SI->getPointerAddressSpace());
    }
    case Instruction::Load: {
      LoadInst *LI = cast<LoadInst>(I);

      if (VF == 1)
        return VTTI->getMemoryOpCost(I->getOpcode(), RetTy,
                                     LI->getAlignment(),
                                     LI->getPointerAddressSpace());

      // Scalarized loads.
      if (!Legal->isConsecutiveGep(LI->getPointerOperand())) {
        unsigned Cost = 0;
        unsigned InCost = VTTI->getInstrCost(Instruction::InsertElement, RetTy);
        // The cost of inserting the loaded value into the result vector.
        Cost += VF * (InCost);
        // The cost of the scalar stores.
        Cost += VF * VTTI->getMemoryOpCost(I->getOpcode(),
                                           RetTy->getScalarType(),
                                           LI->getAlignment(),
                                           LI->getPointerAddressSpace());
        return Cost;
      }

      // Wide loads.
      return VTTI->getMemoryOpCost(I->getOpcode(), VectorTy, LI->getAlignment(),
                                   LI->getPointerAddressSpace());
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
      Type *SrcVecTy = ToVectorTy(I->getOperand(0)->getType(), VF);
      return VTTI->getCastInstrCost(I->getOpcode(), VectorTy, SrcVecTy);
    }
    default: {
      // We are scalarizing the instruction. Return the cost of the scalar
      // instruction, plus the cost of insert and extract into vector
      // elements, times the vector width.
      unsigned Cost = 0;

      bool IsVoid = RetTy->isVoidTy();

      unsigned InsCost = (IsVoid ? 0 :
                          VTTI->getInstrCost(Instruction::InsertElement,
                                             VectorTy));

      unsigned ExtCost = VTTI->getInstrCost(Instruction::ExtractElement,
                                            VectorTy);

      // The cost of inserting the results plus extracting each one of the
      // operands.
      Cost += VF * (InsCost + ExtCost * I->getNumOperands());

      // The cost of executing VF copies of the scalar instruction.
      Cost += VF * VTTI->getInstrCost(I->getOpcode(), RetTy);
      return Cost;
    }
  }// end of switch.
}

Type* LoopVectorizationCostModel::ToVectorTy(Type *Scalar, unsigned VF) {
  if (Scalar->isVoidTy() || VF == 1)
    return Scalar;
  return VectorType::get(Scalar, VF);
}

} // namespace

char LoopVectorize::ID = 0;
static const char lv_name[] = "Loop Vectorization";
INITIALIZE_PASS_BEGIN(LoopVectorize, LV_NAME, lv_name, false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_END(LoopVectorize, LV_NAME, lv_name, false, false)

namespace llvm {
  Pass *createLoopVectorizePass() {
    return new LoopVectorize();
  }
}

