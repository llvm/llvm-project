//===- SLPVectorizer.cpp - A bottom up SLP Vectorizer ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass implements the Bottom Up SLP vectorizer. It detects consecutive
// stores that can be put together into vector-stores. Next, it attempts to
// construct vectorizable tree using the use-def chains. If a profitable tree
// was found, the SLP vectorizer performs vectorization on the tree.
//
// The pass is inspired by the work described in the paper:
//  "Loop-Aware SLP in GCC" by Ira Rosen, Dorit Nuzman, Ayal Zaks.
//
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/Vectorize/SLPVectorizer.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize.h"
#include <algorithm>
#include <memory>

using namespace llvm;
using namespace slpvectorizer;

#define SV_NAME "slp-vectorizer"
#define DEBUG_TYPE "SLP"

STATISTIC(NumVectorInstructions, "Number of vector instructions generated");

static cl::opt<int>
    SLPCostThreshold("slp-threshold", cl::init(0), cl::Hidden,
                     cl::desc("Only vectorize if you gain more than this "
                              "number "));

static cl::opt<bool>
ShouldVectorizeHor("slp-vectorize-hor", cl::init(true), cl::Hidden,
                   cl::desc("Attempt to vectorize horizontal reductions"));

static cl::opt<bool> ShouldStartVectorizeHorAtStore(
    "slp-vectorize-hor-store", cl::init(false), cl::Hidden,
    cl::desc(
        "Attempt to vectorize horizontal reductions feeding into a store"));

static cl::opt<int>
MaxVectorRegSizeOption("slp-max-reg-size", cl::init(128), cl::Hidden,
    cl::desc("Attempt to vectorize for this register size in bits"));

/// Limits the size of scheduling regions in a block.
/// It avoid long compile times for _very_ large blocks where vector
/// instructions are spread over a wide range.
/// This limit is way higher than needed by real-world functions.
static cl::opt<int>
ScheduleRegionSizeBudget("slp-schedule-budget", cl::init(100000), cl::Hidden,
    cl::desc("Limit the size of the SLP scheduling region per block"));

static cl::opt<int> MinVectorRegSizeOption(
    "slp-min-reg-size", cl::init(128), cl::Hidden,
    cl::desc("Attempt to vectorize for this register size in bits"));

static cl::opt<unsigned> RecursionMaxDepth(
    "slp-recursion-max-depth", cl::init(12), cl::Hidden,
    cl::desc("Limit the recursion depth when building a vectorizable tree"));

static cl::opt<unsigned> MinTreeSize(
    "slp-min-tree-size", cl::init(3), cl::Hidden,
    cl::desc("Only vectorize small trees if they are fully vectorizable"));

// Limit the number of alias checks. The limit is chosen so that
// it has no negative effect on the llvm benchmarks.
static const unsigned AliasedCheckLimit = 10;

// Another limit for the alias checks: The maximum distance between load/store
// instructions where alias checks are done.
// This limit is useful for very large basic blocks.
static const unsigned MaxMemDepDistance = 160;

/// If the ScheduleRegionSizeBudget is exhausted, we allow small scheduling
/// regions to be handled.
static const int MinScheduleRegionSize = 16;

/// \brief Predicate for the element types that the SLP vectorizer supports.
///
/// The most important thing to filter here are types which are invalid in LLVM
/// vectors. We also filter target specific types which have absolutely no
/// meaningful vectorization path such as x86_fp80 and ppc_f128. This just
/// avoids spending time checking the cost model and realizing that they will
/// be inevitably scalarized.
static bool isValidElementType(Type *Ty) {
  return VectorType::isValidElementType(Ty) && !Ty->isX86_FP80Ty() &&
         !Ty->isPPC_FP128Ty();
}

/// \returns true if all of the instructions in \p VL are in the same block or
/// false otherwise.
static bool allSameBlock(ArrayRef<Value *> VL) {
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  if (!I0)
    return false;
  BasicBlock *BB = I0->getParent();
  for (int i = 1, e = VL.size(); i < e; i++) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    if (!I)
      return false;

    if (BB != I->getParent())
      return false;
  }
  return true;
}

/// \returns True if all of the values in \p VL are constants.
static bool allConstant(ArrayRef<Value *> VL) {
  for (Value *i : VL)
    if (!isa<Constant>(i))
      return false;
  return true;
}

/// \returns True if all of the values in \p VL are identical.
static bool isSplat(ArrayRef<Value *> VL) {
  for (unsigned i = 1, e = VL.size(); i < e; ++i)
    if (VL[i] != VL[0])
      return false;
  return true;
}

///\returns Opcode that can be clubbed with \p Op to create an alternate
/// sequence which can later be merged as a ShuffleVector instruction.
static unsigned getAltOpcode(unsigned Op) {
  switch (Op) {
  case Instruction::FAdd:
    return Instruction::FSub;
  case Instruction::FSub:
    return Instruction::FAdd;
  case Instruction::Add:
    return Instruction::Sub;
  case Instruction::Sub:
    return Instruction::Add;
  default:
    return 0;
  }
}

///\returns bool representing if Opcode \p Op can be part
/// of an alternate sequence which can later be merged as
/// a ShuffleVector instruction.
static bool canCombineAsAltInst(unsigned Op) {
  return Op == Instruction::FAdd || Op == Instruction::FSub ||
         Op == Instruction::Sub || Op == Instruction::Add;
}

/// \returns ShuffleVector instruction if instructions in \p VL have
///  alternate fadd,fsub / fsub,fadd/add,sub/sub,add sequence.
/// (i.e. e.g. opcodes of fadd,fsub,fadd,fsub...)
static unsigned isAltInst(ArrayRef<Value *> VL) {
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  unsigned Opcode = I0->getOpcode();
  unsigned AltOpcode = getAltOpcode(Opcode);
  for (int i = 1, e = VL.size(); i < e; i++) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    if (!I || I->getOpcode() != ((i & 1) ? AltOpcode : Opcode))
      return 0;
  }
  return Instruction::ShuffleVector;
}

/// \returns The opcode if all of the Instructions in \p VL have the same
/// opcode, or zero.
static unsigned getSameOpcode(ArrayRef<Value *> VL) {
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  if (!I0)
    return 0;
  unsigned Opcode = I0->getOpcode();
  for (int i = 1, e = VL.size(); i < e; i++) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    if (!I || Opcode != I->getOpcode()) {
      if (canCombineAsAltInst(Opcode) && i == 1)
        return isAltInst(VL);
      return 0;
    }
  }
  return Opcode;
}

/// Get the intersection (logical and) of all of the potential IR flags
/// of each scalar operation (VL) that will be converted into a vector (I).
/// Flag set: NSW, NUW, exact, and all of fast-math.
static void propagateIRFlags(Value *I, ArrayRef<Value *> VL) {
  if (auto *VecOp = dyn_cast<Instruction>(I)) {
    if (auto *I0 = dyn_cast<Instruction>(VL[0])) {
      // VecOVp is initialized to the 0th scalar, so start counting from index
      // '1'.
      VecOp->copyIRFlags(I0);
      for (int i = 1, e = VL.size(); i < e; ++i) {
        if (auto *Scalar = dyn_cast<Instruction>(VL[i]))
          VecOp->andIRFlags(Scalar);
      }
    }
  }
}

/// \returns true if all of the values in \p VL have the same type or false
/// otherwise.
static bool allSameType(ArrayRef<Value *> VL) {
  Type *Ty = VL[0]->getType();
  for (int i = 1, e = VL.size(); i < e; i++)
    if (VL[i]->getType() != Ty)
      return false;

  return true;
}

/// \returns True if Extract{Value,Element} instruction extracts element Idx.
static bool matchExtractIndex(Instruction *E, unsigned Idx, unsigned Opcode) {
  assert(Opcode == Instruction::ExtractElement ||
         Opcode == Instruction::ExtractValue);
  if (Opcode == Instruction::ExtractElement) {
    ConstantInt *CI = dyn_cast<ConstantInt>(E->getOperand(1));
    return CI && CI->getZExtValue() == Idx;
  } else {
    ExtractValueInst *EI = cast<ExtractValueInst>(E);
    return EI->getNumIndices() == 1 && *EI->idx_begin() == Idx;
  }
}

/// \returns True if in-tree use also needs extract. This refers to
/// possible scalar operand in vectorized instruction.
static bool InTreeUserNeedToExtract(Value *Scalar, Instruction *UserInst,
                                    TargetLibraryInfo *TLI) {

  unsigned Opcode = UserInst->getOpcode();
  switch (Opcode) {
  case Instruction::Load: {
    LoadInst *LI = cast<LoadInst>(UserInst);
    return (LI->getPointerOperand() == Scalar);
  }
  case Instruction::Store: {
    StoreInst *SI = cast<StoreInst>(UserInst);
    return (SI->getPointerOperand() == Scalar);
  }
  case Instruction::Call: {
    CallInst *CI = cast<CallInst>(UserInst);
    Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);
    if (hasVectorInstrinsicScalarOpd(ID, 1)) {
      return (CI->getArgOperand(1) == Scalar);
    }
  }
  default:
    return false;
  }
}

/// \returns the AA location that is being access by the instruction.
static MemoryLocation getLocation(Instruction *I, AliasAnalysis *AA) {
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return MemoryLocation::get(SI);
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return MemoryLocation::get(LI);
  return MemoryLocation();
}

/// \returns True if the instruction is not a volatile or atomic load/store.
static bool isSimple(Instruction *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->isSimple();
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->isSimple();
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(I))
    return !MI->isVolatile();
  return true;
}

namespace llvm {
namespace slpvectorizer {
/// Bottom Up SLP Vectorizer.
class BoUpSLP {
public:
  typedef SmallVector<Value *, 8> ValueList;
  typedef SmallVector<Instruction *, 16> InstrList;
  typedef SmallPtrSet<Value *, 16> ValueSet;
  typedef SmallVector<StoreInst *, 8> StoreList;
  typedef MapVector<Value *, SmallVector<Instruction *, 2>>
      ExtraValueToDebugLocsMap;

  BoUpSLP(Function *Func, ScalarEvolution *Se, TargetTransformInfo *Tti,
          TargetLibraryInfo *TLi, AliasAnalysis *Aa, LoopInfo *Li,
          DominatorTree *Dt, AssumptionCache *AC, DemandedBits *DB,
          const DataLayout *DL)
      : NumLoadsWantToKeepOrder(0), NumLoadsWantToChangeOrder(0), F(Func),
        SE(Se), TTI(Tti), TLI(TLi), AA(Aa), LI(Li), DT(Dt), AC(AC), DB(DB),
        DL(DL), Builder(Se->getContext()) {
    CodeMetrics::collectEphemeralValues(F, AC, EphValues);
    // Use the vector register size specified by the target unless overridden
    // by a command-line option.
    // TODO: It would be better to limit the vectorization factor based on
    //       data type rather than just register size. For example, x86 AVX has
    //       256-bit registers, but it does not support integer operations
    //       at that width (that requires AVX2).
    if (MaxVectorRegSizeOption.getNumOccurrences())
      MaxVecRegSize = MaxVectorRegSizeOption;
    else
      MaxVecRegSize = TTI->getRegisterBitWidth(true);

    MinVecRegSize = MinVectorRegSizeOption;
  }

  /// \brief Vectorize the tree that starts with the elements in \p VL.
  /// Returns the vectorized root.
  Value *vectorizeTree();
  /// Vectorize the tree but with the list of externally used values \p
  /// ExternallyUsedValues. Values in this MapVector can be replaced but the
  /// generated extractvalue instructions.
  Value *vectorizeTree(ExtraValueToDebugLocsMap &ExternallyUsedValues);

  /// \returns the cost incurred by unwanted spills and fills, caused by
  /// holding live values over call sites.
  int getSpillCost();

  /// \returns the vectorization cost of the subtree that starts at \p VL.
  /// A negative number means that this is profitable.
  int getTreeCost();

  /// Construct a vectorizable tree that starts at \p Roots, ignoring users for
  /// the purpose of scheduling and extraction in the \p UserIgnoreLst.
  void buildTree(ArrayRef<Value *> Roots,
                 ArrayRef<Value *> UserIgnoreLst = None);
  /// Construct a vectorizable tree that starts at \p Roots, ignoring users for
  /// the purpose of scheduling and extraction in the \p UserIgnoreLst taking
  /// into account (anf updating it, if required) list of externally used
  /// values stored in \p ExternallyUsedValues.
  void buildTree(ArrayRef<Value *> Roots,
                 ExtraValueToDebugLocsMap &ExternallyUsedValues,
                 ArrayRef<Value *> UserIgnoreLst = None);

  /// Clear the internal data structures that are created by 'buildTree'.
  void deleteTree() {
    VectorizableTree.clear();
    ScalarToTreeEntry.clear();
    MustGather.clear();
    ExternalUses.clear();
    NumLoadsWantToKeepOrder = 0;
    NumLoadsWantToChangeOrder = 0;
    for (auto &Iter : BlocksSchedules) {
      BlockScheduling *BS = Iter.second.get();
      BS->clear();
    }
    MinBWs.clear();
  }

  /// \brief Perform LICM and CSE on the newly generated gather sequences.
  void optimizeGatherSequence();

  /// \returns true if it is beneficial to reverse the vector order.
  bool shouldReorder() const {
    return NumLoadsWantToChangeOrder > NumLoadsWantToKeepOrder;
  }

  /// \return The vector element size in bits to use when vectorizing the
  /// expression tree ending at \p V. If V is a store, the size is the width of
  /// the stored value. Otherwise, the size is the width of the largest loaded
  /// value reaching V. This method is used by the vectorizer to calculate
  /// vectorization factors.
  unsigned getVectorElementSize(Value *V);

  /// Compute the minimum type sizes required to represent the entries in a
  /// vectorizable tree.
  void computeMinimumValueSizes();

  // \returns maximum vector register size as set by TTI or overridden by cl::opt.
  unsigned getMaxVecRegSize() const {
    return MaxVecRegSize;
  }

  // \returns minimum vector register size as set by cl::opt.
  unsigned getMinVecRegSize() const {
    return MinVecRegSize;
  }

  /// \brief Check if ArrayType or StructType is isomorphic to some VectorType.
  ///
  /// \returns number of elements in vector if isomorphism exists, 0 otherwise.
  unsigned canMapToVector(Type *T, const DataLayout &DL) const;

  /// \returns True if the VectorizableTree is both tiny and not fully
  /// vectorizable. We do not vectorize such trees.
  bool isTreeTinyAndNotFullyVectorizable();

private:
  struct TreeEntry;

  /// \returns the cost of the vectorizable entry.
  int getEntryCost(TreeEntry *E);

  /// This is the recursive part of buildTree.
  void buildTree_rec(ArrayRef<Value *> Roots, unsigned Depth);

  /// \returns True if the ExtractElement/ExtractValue instructions in VL can
  /// be vectorized to use the original vector (or aggregate "bitcast" to a vector).
  bool canReuseExtract(ArrayRef<Value *> VL, unsigned Opcode) const;

  /// Vectorize a single entry in the tree. VL icontains all isomorphic scalars
  /// in order of its usage in a user program, for example ADD1, ADD2 and so on
  /// or LOAD1 , LOAD2 etc.
  Value *vectorizeTree(ArrayRef<Value *> VL, TreeEntry *E);

  /// Vectorize a single entry in the tree, starting in \p VL.
  Value *vectorizeTree(ArrayRef<Value *> VL);

  /// \returns the pointer to the vectorized value if \p VL is already
  /// vectorized, or NULL. They may happen in cycles.
  Value *alreadyVectorized(ArrayRef<Value *> VL) const;

  /// \returns the scalarization cost for this type. Scalarization in this
  /// context means the creation of vectors from a group of scalars.
  int getGatherCost(Type *Ty);

  /// \returns the scalarization cost for this list of values. Assuming that
  /// this subtree gets vectorized, we may need to extract the values from the
  /// roots. This method calculates the cost of extracting the values.
  int getGatherCost(ArrayRef<Value *> VL);

  /// \brief Set the Builder insert point to one after the last instruction in
  /// the bundle
  void setInsertPointAfterBundle(ArrayRef<Value *> VL);

  /// \returns a vector from a collection of scalars in \p VL.
  Value *Gather(ArrayRef<Value *> VL, VectorType *Ty);

  /// \returns whether the VectorizableTree is fully vectorizable and will
  /// be beneficial even the tree height is tiny.
  bool isFullyVectorizableTinyTree();

  /// \reorder commutative operands in alt shuffle if they result in
  ///  vectorized code.
  void reorderAltShuffleOperands(ArrayRef<Value *> VL,
                                 SmallVectorImpl<Value *> &Left,
                                 SmallVectorImpl<Value *> &Right);
  /// \reorder commutative operands to get better probability of
  /// generating vectorized code.
  void reorderInputsAccordingToOpcode(ArrayRef<Value *> VL,
                                      SmallVectorImpl<Value *> &Left,
                                      SmallVectorImpl<Value *> &Right);
  struct TreeEntry {
    TreeEntry() : Scalars(), VectorizedValue(nullptr),
    NeedToGather(0), NeedToShuffle(0) {}

    /// \returns true if the scalars in VL are equal to this entry.
    bool isSame(ArrayRef<Value *> VL) const {
      assert(VL.size() == Scalars.size() && "Invalid size");
      return std::equal(VL.begin(), VL.end(), Scalars.begin());
    }

    /// \returns true if the scalars in VL are found in this tree entry.
    bool isFoundJumbled(ArrayRef<Value *> VL, const DataLayout &DL,
                        ScalarEvolution &SE) const {
      assert(VL.size() == Scalars.size() && "Invalid size");
      SmallVector<Value *, 8> List;
      if (!sortMemAccesses(VL, DL, SE, List))
        return false;

      return std::equal(List.begin(), List.end(), Scalars.begin());
    }

    /// A vector of scalars.
    ValueList Scalars;

    /// The Scalars are vectorized into this value. It is initialized to Null.
    Value *VectorizedValue;

    /// Do we need to gather this sequence ?
    bool NeedToGather;

    /// Do we need to shuffle the load ?
    bool NeedToShuffle;
  };

  /// Create a new VectorizableTree entry.
  TreeEntry *newTreeEntry(ArrayRef<Value *> VL, bool Vectorized,
                          bool NeedToShuffle) {
    VectorizableTree.emplace_back();
    int idx = VectorizableTree.size() - 1;
    TreeEntry *Last = &VectorizableTree[idx];
    Last->Scalars.insert(Last->Scalars.begin(), VL.begin(), VL.end());
    Last->NeedToGather = !Vectorized;
    Last->NeedToShuffle = NeedToShuffle;
    if (Vectorized) {
      for (int i = 0, e = VL.size(); i != e; ++i) {
        assert(!ScalarToTreeEntry.count(VL[i]) && "Scalar already in tree!");
        ScalarToTreeEntry[VL[i]] = idx;
      }
    } else {
      MustGather.insert(VL.begin(), VL.end());
    }
    return Last;
  }

  /// -- Vectorization State --
  /// Holds all of the tree entries.
  std::vector<TreeEntry> VectorizableTree;

  /// Maps a specific scalar to its tree entry.
  SmallDenseMap<Value*, int> ScalarToTreeEntry;

  /// A list of scalars that we found that we need to keep as scalars.
  ValueSet MustGather;

  /// This POD struct describes one external user in the vectorized tree.
  struct ExternalUser {
    ExternalUser (Value *S, llvm::User *U, int L) :
      Scalar(S), User(U), Lane(L){}
    // Which scalar in our function.
    Value *Scalar;
    // Which user that uses the scalar.
    llvm::User *User;
    // Which lane does the scalar belong to.
    int Lane;
  };
  typedef SmallVector<ExternalUser, 16> UserList;

  /// Checks if two instructions may access the same memory.
  ///
  /// \p Loc1 is the location of \p Inst1. It is passed explicitly because it
  /// is invariant in the calling loop.
  bool isAliased(const MemoryLocation &Loc1, Instruction *Inst1,
                 Instruction *Inst2) {

    // First check if the result is already in the cache.
    AliasCacheKey key = std::make_pair(Inst1, Inst2);
    Optional<bool> &result = AliasCache[key];
    if (result.hasValue()) {
      return result.getValue();
    }
    MemoryLocation Loc2 = getLocation(Inst2, AA);
    bool aliased = true;
    if (Loc1.Ptr && Loc2.Ptr && isSimple(Inst1) && isSimple(Inst2)) {
      // Do the alias check.
      aliased = AA->alias(Loc1, Loc2);
    }
    // Store the result in the cache.
    result = aliased;
    return aliased;
  }

  typedef std::pair<Instruction *, Instruction *> AliasCacheKey;

  /// Cache for alias results.
  /// TODO: consider moving this to the AliasAnalysis itself.
  DenseMap<AliasCacheKey, Optional<bool>> AliasCache;

  /// Removes an instruction from its block and eventually deletes it.
  /// It's like Instruction::eraseFromParent() except that the actual deletion
  /// is delayed until BoUpSLP is destructed.
  /// This is required to ensure that there are no incorrect collisions in the
  /// AliasCache, which can happen if a new instruction is allocated at the
  /// same address as a previously deleted instruction.
  void eraseInstruction(Instruction *I) {
    I->removeFromParent();
    I->dropAllReferences();
    DeletedInstructions.push_back(std::unique_ptr<Instruction>(I));
  }

  /// Temporary store for deleted instructions. Instructions will be deleted
  /// eventually when the BoUpSLP is destructed.
  SmallVector<std::unique_ptr<Instruction>, 8> DeletedInstructions;

  /// A list of values that need to extracted out of the tree.
  /// This list holds pairs of (Internal Scalar : External User). External User
  /// can be nullptr, it means that this Internal Scalar will be used later,
  /// after vectorization.
  UserList ExternalUses;

  /// Values used only by @llvm.assume calls.
  SmallPtrSet<const Value *, 32> EphValues;

  /// Holds all of the instructions that we gathered.
  SetVector<Instruction *> GatherSeq;
  /// A list of blocks that we are going to CSE.
  SetVector<BasicBlock *> CSEBlocks;

  /// Contains all scheduling relevant data for an instruction.
  /// A ScheduleData either represents a single instruction or a member of an
  /// instruction bundle (= a group of instructions which is combined into a
  /// vector instruction).
  struct ScheduleData {

    // The initial value for the dependency counters. It means that the
    // dependencies are not calculated yet.
    enum { InvalidDeps = -1 };

    ScheduleData()
        : Inst(nullptr), FirstInBundle(nullptr), NextInBundle(nullptr),
          NextLoadStore(nullptr), SchedulingRegionID(0), SchedulingPriority(0),
          Dependencies(InvalidDeps), UnscheduledDeps(InvalidDeps),
          UnscheduledDepsInBundle(InvalidDeps), IsScheduled(false) {}

    void init(int BlockSchedulingRegionID) {
      FirstInBundle = this;
      NextInBundle = nullptr;
      NextLoadStore = nullptr;
      IsScheduled = false;
      SchedulingRegionID = BlockSchedulingRegionID;
      UnscheduledDepsInBundle = UnscheduledDeps;
      clearDependencies();
    }

    /// Returns true if the dependency information has been calculated.
    bool hasValidDependencies() const { return Dependencies != InvalidDeps; }

    /// Returns true for single instructions and for bundle representatives
    /// (= the head of a bundle).
    bool isSchedulingEntity() const { return FirstInBundle == this; }

    /// Returns true if it represents an instruction bundle and not only a
    /// single instruction.
    bool isPartOfBundle() const {
      return NextInBundle != nullptr || FirstInBundle != this;
    }

    /// Returns true if it is ready for scheduling, i.e. it has no more
    /// unscheduled depending instructions/bundles.
    bool isReady() const {
      assert(isSchedulingEntity() &&
             "can't consider non-scheduling entity for ready list");
      return UnscheduledDepsInBundle == 0 && !IsScheduled;
    }

    /// Modifies the number of unscheduled dependencies, also updating it for
    /// the whole bundle.
    int incrementUnscheduledDeps(int Incr) {
      UnscheduledDeps += Incr;
      return FirstInBundle->UnscheduledDepsInBundle += Incr;
    }

    /// Sets the number of unscheduled dependencies to the number of
    /// dependencies.
    void resetUnscheduledDeps() {
      incrementUnscheduledDeps(Dependencies - UnscheduledDeps);
    }

    /// Clears all dependency information.
    void clearDependencies() {
      Dependencies = InvalidDeps;
      resetUnscheduledDeps();
      MemoryDependencies.clear();
    }

    void dump(raw_ostream &os) const {
      if (!isSchedulingEntity()) {
        os << "/ " << *Inst;
      } else if (NextInBundle) {
        os << '[' << *Inst;
        ScheduleData *SD = NextInBundle;
        while (SD) {
          os << ';' << *SD->Inst;
          SD = SD->NextInBundle;
        }
        os << ']';
      } else {
        os << *Inst;
      }
    }

    Instruction *Inst;

    /// Points to the head in an instruction bundle (and always to this for
    /// single instructions).
    ScheduleData *FirstInBundle;

    /// Single linked list of all instructions in a bundle. Null if it is a
    /// single instruction.
    ScheduleData *NextInBundle;

    /// Single linked list of all memory instructions (e.g. load, store, call)
    /// in the block - until the end of the scheduling region.
    ScheduleData *NextLoadStore;

    /// The dependent memory instructions.
    /// This list is derived on demand in calculateDependencies().
    SmallVector<ScheduleData *, 4> MemoryDependencies;

    /// This ScheduleData is in the current scheduling region if this matches
    /// the current SchedulingRegionID of BlockScheduling.
    int SchedulingRegionID;

    /// Used for getting a "good" final ordering of instructions.
    int SchedulingPriority;

    /// The number of dependencies. Constitutes of the number of users of the
    /// instruction plus the number of dependent memory instructions (if any).
    /// This value is calculated on demand.
    /// If InvalidDeps, the number of dependencies is not calculated yet.
    ///
    int Dependencies;

    /// The number of dependencies minus the number of dependencies of scheduled
    /// instructions. As soon as this is zero, the instruction/bundle gets ready
    /// for scheduling.
    /// Note that this is negative as long as Dependencies is not calculated.
    int UnscheduledDeps;

    /// The sum of UnscheduledDeps in a bundle. Equals to UnscheduledDeps for
    /// single instructions.
    int UnscheduledDepsInBundle;

    /// True if this instruction is scheduled (or considered as scheduled in the
    /// dry-run).
    bool IsScheduled;
  };

#ifndef NDEBUG
  friend inline raw_ostream &operator<<(raw_ostream &os,
                                        const BoUpSLP::ScheduleData &SD) {
    SD.dump(os);
    return os;
  }
#endif

  /// Contains all scheduling data for a basic block.
  ///
  struct BlockScheduling {

    BlockScheduling(BasicBlock *BB)
        : BB(BB), ChunkSize(BB->size()), ChunkPos(ChunkSize),
          ScheduleStart(nullptr), ScheduleEnd(nullptr),
          FirstLoadStoreInRegion(nullptr), LastLoadStoreInRegion(nullptr),
          ScheduleRegionSize(0),
          ScheduleRegionSizeLimit(ScheduleRegionSizeBudget),
          // Make sure that the initial SchedulingRegionID is greater than the
          // initial SchedulingRegionID in ScheduleData (which is 0).
          SchedulingRegionID(1) {}

    void clear() {
      ReadyInsts.clear();
      ScheduleStart = nullptr;
      ScheduleEnd = nullptr;
      FirstLoadStoreInRegion = nullptr;
      LastLoadStoreInRegion = nullptr;

      // Reduce the maximum schedule region size by the size of the
      // previous scheduling run.
      ScheduleRegionSizeLimit -= ScheduleRegionSize;
      if (ScheduleRegionSizeLimit < MinScheduleRegionSize)
        ScheduleRegionSizeLimit = MinScheduleRegionSize;
      ScheduleRegionSize = 0;

      // Make a new scheduling region, i.e. all existing ScheduleData is not
      // in the new region yet.
      ++SchedulingRegionID;
    }

    ScheduleData *getScheduleData(Value *V) {
      ScheduleData *SD = ScheduleDataMap[V];
      if (SD && SD->SchedulingRegionID == SchedulingRegionID)
        return SD;
      return nullptr;
    }

    bool isInSchedulingRegion(ScheduleData *SD) {
      return SD->SchedulingRegionID == SchedulingRegionID;
    }

    /// Marks an instruction as scheduled and puts all dependent ready
    /// instructions into the ready-list.
    template <typename ReadyListType>
    void schedule(ScheduleData *SD, ReadyListType &ReadyList) {
      SD->IsScheduled = true;
      DEBUG(dbgs() << "SLP:   schedule " << *SD << "\n");

      ScheduleData *BundleMember = SD;
      while (BundleMember) {
        // Handle the def-use chain dependencies.
        for (Use &U : BundleMember->Inst->operands()) {
          ScheduleData *OpDef = getScheduleData(U.get());
          if (OpDef && OpDef->hasValidDependencies() &&
              OpDef->incrementUnscheduledDeps(-1) == 0) {
            // There are no more unscheduled dependencies after decrementing,
            // so we can put the dependent instruction into the ready list.
            ScheduleData *DepBundle = OpDef->FirstInBundle;
            assert(!DepBundle->IsScheduled &&
                   "already scheduled bundle gets ready");
            ReadyList.insert(DepBundle);
            DEBUG(dbgs() << "SLP:    gets ready (def): " << *DepBundle << "\n");
          }
        }
        // Handle the memory dependencies.
        for (ScheduleData *MemoryDepSD : BundleMember->MemoryDependencies) {
          if (MemoryDepSD->incrementUnscheduledDeps(-1) == 0) {
            // There are no more unscheduled dependencies after decrementing,
            // so we can put the dependent instruction into the ready list.
            ScheduleData *DepBundle = MemoryDepSD->FirstInBundle;
            assert(!DepBundle->IsScheduled &&
                   "already scheduled bundle gets ready");
            ReadyList.insert(DepBundle);
            DEBUG(dbgs() << "SLP:    gets ready (mem): " << *DepBundle << "\n");
          }
        }
        BundleMember = BundleMember->NextInBundle;
      }
    }

    /// Put all instructions into the ReadyList which are ready for scheduling.
    template <typename ReadyListType>
    void initialFillReadyList(ReadyListType &ReadyList) {
      for (auto *I = ScheduleStart; I != ScheduleEnd; I = I->getNextNode()) {
        ScheduleData *SD = getScheduleData(I);
        if (SD->isSchedulingEntity() && SD->isReady()) {
          ReadyList.insert(SD);
          DEBUG(dbgs() << "SLP:    initially in ready list: " << *I << "\n");
        }
      }
    }

    /// Checks if a bundle of instructions can be scheduled, i.e. has no
    /// cyclic dependencies. This is only a dry-run, no instructions are
    /// actually moved at this stage.
    bool tryScheduleBundle(ArrayRef<Value *> VL, BoUpSLP *SLP);

    /// Un-bundles a group of instructions.
    void cancelScheduling(ArrayRef<Value *> VL);

    /// Extends the scheduling region so that V is inside the region.
    /// \returns true if the region size is within the limit.
    bool extendSchedulingRegion(Value *V);

    /// Initialize the ScheduleData structures for new instructions in the
    /// scheduling region.
    void initScheduleData(Instruction *FromI, Instruction *ToI,
                          ScheduleData *PrevLoadStore,
                          ScheduleData *NextLoadStore);

    /// Updates the dependency information of a bundle and of all instructions/
    /// bundles which depend on the original bundle.
    void calculateDependencies(ScheduleData *SD, bool InsertInReadyList,
                               BoUpSLP *SLP);

    /// Sets all instruction in the scheduling region to un-scheduled.
    void resetSchedule();

    BasicBlock *BB;

    /// Simple memory allocation for ScheduleData.
    std::vector<std::unique_ptr<ScheduleData[]>> ScheduleDataChunks;

    /// The size of a ScheduleData array in ScheduleDataChunks.
    int ChunkSize;

    /// The allocator position in the current chunk, which is the last entry
    /// of ScheduleDataChunks.
    int ChunkPos;

    /// Attaches ScheduleData to Instruction.
    /// Note that the mapping survives during all vectorization iterations, i.e.
    /// ScheduleData structures are recycled.
    DenseMap<Value *, ScheduleData *> ScheduleDataMap;

    struct ReadyList : SmallVector<ScheduleData *, 8> {
      void insert(ScheduleData *SD) { push_back(SD); }
    };

    /// The ready-list for scheduling (only used for the dry-run).
    ReadyList ReadyInsts;

    /// The first instruction of the scheduling region.
    Instruction *ScheduleStart;

    /// The first instruction _after_ the scheduling region.
    Instruction *ScheduleEnd;

    /// The first memory accessing instruction in the scheduling region
    /// (can be null).
    ScheduleData *FirstLoadStoreInRegion;

    /// The last memory accessing instruction in the scheduling region
    /// (can be null).
    ScheduleData *LastLoadStoreInRegion;

    /// The current size of the scheduling region.
    int ScheduleRegionSize;

    /// The maximum size allowed for the scheduling region.
    int ScheduleRegionSizeLimit;

    /// The ID of the scheduling region. For a new vectorization iteration this
    /// is incremented which "removes" all ScheduleData from the region.
    int SchedulingRegionID;
  };

  /// Attaches the BlockScheduling structures to basic blocks.
  MapVector<BasicBlock *, std::unique_ptr<BlockScheduling>> BlocksSchedules;

  /// Performs the "real" scheduling. Done before vectorization is actually
  /// performed in a basic block.
  void scheduleBlock(BlockScheduling *BS);

  /// List of users to ignore during scheduling and that don't need extracting.
  ArrayRef<Value *> UserIgnoreList;

  // Number of load bundles that contain consecutive loads.
  int NumLoadsWantToKeepOrder;

  // Number of load bundles that contain consecutive loads in reversed order.
  int NumLoadsWantToChangeOrder;

  // Analysis and block reference.
  Function *F;
  ScalarEvolution *SE;
  TargetTransformInfo *TTI;
  TargetLibraryInfo *TLI;
  AliasAnalysis *AA;
  LoopInfo *LI;
  DominatorTree *DT;
  AssumptionCache *AC;
  DemandedBits *DB;
  const DataLayout *DL;
  unsigned MaxVecRegSize; // This is set by TTI or overridden by cl::opt.
  unsigned MinVecRegSize; // Set by cl::opt (default: 128).
  /// Instruction builder to construct the vectorized tree.
  IRBuilder<> Builder;

  /// A map of scalar integer values to the smallest bit width with which they
  /// can legally be represented. The values map to (width, signed) pairs,
  /// where "width" indicates the minimum bit width and "signed" is True if the
  /// value must be signed-extended, rather than zero-extended, back to its
  /// original width.
  MapVector<Value *, std::pair<uint64_t, bool>> MinBWs;
};

} // end namespace llvm
} // end namespace slpvectorizer

void BoUpSLP::buildTree(ArrayRef<Value *> Roots,
                        ArrayRef<Value *> UserIgnoreLst) {
  ExtraValueToDebugLocsMap ExternallyUsedValues;
  buildTree(Roots, ExternallyUsedValues, UserIgnoreLst);
}
void BoUpSLP::buildTree(ArrayRef<Value *> Roots,
                        ExtraValueToDebugLocsMap &ExternallyUsedValues,
                        ArrayRef<Value *> UserIgnoreLst) {
  deleteTree();
  UserIgnoreList = UserIgnoreLst;
  if (!allSameType(Roots))
    return;
  buildTree_rec(Roots, 0);

  // Collect the values that we need to extract from the tree.
  for (TreeEntry &EIdx : VectorizableTree) {
    TreeEntry *Entry = &EIdx;

    // For each lane:
    for (int Lane = 0, LE = Entry->Scalars.size(); Lane != LE; ++Lane) {
      Value *Scalar = Entry->Scalars[Lane];

      // No need to handle users of gathered values.
      if (Entry->NeedToGather)
        continue;

      // Check if the scalar is externally used as an extra arg.
      auto ExtI = ExternallyUsedValues.find(Scalar);
      if (ExtI != ExternallyUsedValues.end()) {
        DEBUG(dbgs() << "SLP: Need to extract: Extra arg from lane " <<
              Lane << " from " << *Scalar << ".\n");
        ExternalUses.emplace_back(Scalar, nullptr, Lane);
        continue;
      }
      for (User *U : Scalar->users()) {
        DEBUG(dbgs() << "SLP: Checking user:" << *U << ".\n");

        Instruction *UserInst = dyn_cast<Instruction>(U);
        if (!UserInst)
          continue;

        // Skip in-tree scalars that become vectors
        if (ScalarToTreeEntry.count(U)) {
          int Idx = ScalarToTreeEntry[U];
          TreeEntry *UseEntry = &VectorizableTree[Idx];
          Value *UseScalar = UseEntry->Scalars[0];
          // Some in-tree scalars will remain as scalar in vectorized
          // instructions. If that is the case, the one in Lane 0 will
          // be used.
          if (UseScalar != U ||
              !InTreeUserNeedToExtract(Scalar, UserInst, TLI)) {
            DEBUG(dbgs() << "SLP: \tInternal user will be removed:" << *U
                         << ".\n");
            assert(!VectorizableTree[Idx].NeedToGather && "Bad state");
            continue;
          }
        }

        // Ignore users in the user ignore list.
        if (is_contained(UserIgnoreList, UserInst))
          continue;

        DEBUG(dbgs() << "SLP: Need to extract:" << *U << " from lane " <<
              Lane << " from " << *Scalar << ".\n");
        ExternalUses.push_back(ExternalUser(Scalar, U, Lane));
      }
    }
  }
}


void BoUpSLP::buildTree_rec(ArrayRef<Value *> VL, unsigned Depth) {
  bool isAltShuffle = false;
  assert((allConstant(VL) || allSameType(VL)) && "Invalid types!");

  if (Depth == RecursionMaxDepth) {
    DEBUG(dbgs() << "SLP: Gathering due to max recursion depth.\n");
    newTreeEntry(VL, false, false);
    return;
  }

  // Don't handle vectors.
  if (VL[0]->getType()->isVectorTy()) {
    DEBUG(dbgs() << "SLP: Gathering due to vector type.\n");
    newTreeEntry(VL, false, false);
    return;
  }

  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    if (SI->getValueOperand()->getType()->isVectorTy()) {
      DEBUG(dbgs() << "SLP: Gathering due to store vector type.\n");
      newTreeEntry(VL, false, false);
      return;
    }
  unsigned Opcode = getSameOpcode(VL);

  // Check that this shuffle vector refers to the alternate
  // sequence of opcodes.
  if (Opcode == Instruction::ShuffleVector) {
    Instruction *I0 = dyn_cast<Instruction>(VL[0]);
    unsigned Op = I0->getOpcode();
    if (Op != Instruction::ShuffleVector)
      isAltShuffle = true;
  }

  // If all of the operands are identical or constant we have a simple solution.
  if (allConstant(VL) || isSplat(VL) || !allSameBlock(VL) || !Opcode) {
    DEBUG(dbgs() << "SLP: Gathering due to C,S,B,O. \n");
    newTreeEntry(VL, false, false);
    return;
  }

  // We now know that this is a vector of instructions of the same type from
  // the same block.

  // Don't vectorize ephemeral values.
  for (unsigned i = 0, e = VL.size(); i != e; ++i) {
    if (EphValues.count(VL[i])) {
      DEBUG(dbgs() << "SLP: The instruction (" << *VL[i] <<
            ") is ephemeral.\n");
      newTreeEntry(VL, false, false);
      return;
    }
  }

  // Check if this is a duplicate of another entry.
  if (ScalarToTreeEntry.count(VL[0])) {
    int Idx = ScalarToTreeEntry[VL[0]];
    TreeEntry *E = &VectorizableTree[Idx];
    for (unsigned i = 0, e = VL.size(); i != e; ++i) {
      DEBUG(dbgs() << "SLP: \tChecking bundle: " << *VL[i] << ".\n");
      if (E->Scalars[i] != VL[i]) {
        DEBUG(dbgs() << "SLP: Gathering due to partial overlap.\n");
        newTreeEntry(VL, false, false);
        return;
      }
    }
    DEBUG(dbgs() << "SLP: Perfect diamond merge at " << *VL[0] << ".\n");
    return;
  }

  // Check that none of the instructions in the bundle are already in the tree.
  for (unsigned i = 0, e = VL.size(); i != e; ++i) {
    if (ScalarToTreeEntry.count(VL[i])) {
      DEBUG(dbgs() << "SLP: The instruction (" << *VL[i] <<
            ") is already in tree.\n");
      newTreeEntry(VL, false, false);
      return;
    }
  }

  // If any of the scalars is marked as a value that needs to stay scalar then
  // we need to gather the scalars.
  for (unsigned i = 0, e = VL.size(); i != e; ++i) {
    if (MustGather.count(VL[i])) {
      DEBUG(dbgs() << "SLP: Gathering due to gathered scalar.\n");
      newTreeEntry(VL, false, false);
      return;
    }
  }

  // Check that all of the users of the scalars that we want to vectorize are
  // schedulable.
  Instruction *VL0 = cast<Instruction>(VL[0]);
  BasicBlock *BB = cast<Instruction>(VL0)->getParent();

  if (!DT->isReachableFromEntry(BB)) {
    // Don't go into unreachable blocks. They may contain instructions with
    // dependency cycles which confuse the final scheduling.
    DEBUG(dbgs() << "SLP: bundle in unreachable block.\n");
    newTreeEntry(VL, false, false);
    return;
  }

  // Check that every instructions appears once in this bundle.
  for (unsigned i = 0, e = VL.size(); i < e; ++i)
    for (unsigned j = i+1; j < e; ++j)
      if (VL[i] == VL[j]) {
        DEBUG(dbgs() << "SLP: Scalar used twice in bundle.\n");
        newTreeEntry(VL, false, false);
        return;
      }

  auto &BSRef = BlocksSchedules[BB];
  if (!BSRef) {
    BSRef = llvm::make_unique<BlockScheduling>(BB);
  }
  BlockScheduling &BS = *BSRef.get();

  if (!BS.tryScheduleBundle(VL, this)) {
    DEBUG(dbgs() << "SLP: We are not able to schedule this bundle!\n");
    assert((!BS.getScheduleData(VL[0]) ||
            !BS.getScheduleData(VL[0])->isPartOfBundle()) &&
           "tryScheduleBundle should cancelScheduling on failure");
    newTreeEntry(VL, false, false);
    return;
  }
  DEBUG(dbgs() << "SLP: We are able to schedule this bundle.\n");

  switch (Opcode) {
    case Instruction::PHI: {
      PHINode *PH = dyn_cast<PHINode>(VL0);

      // Check for terminator values (e.g. invoke).
      for (unsigned j = 0; j < VL.size(); ++j)
        for (unsigned i = 0, e = PH->getNumIncomingValues(); i < e; ++i) {
          TerminatorInst *Term = dyn_cast<TerminatorInst>(
              cast<PHINode>(VL[j])->getIncomingValueForBlock(PH->getIncomingBlock(i)));
          if (Term) {
            DEBUG(dbgs() << "SLP: Need to swizzle PHINodes (TerminatorInst use).\n");
            BS.cancelScheduling(VL);
            newTreeEntry(VL, false, false);
            return;
          }
        }

      newTreeEntry(VL, true, false);
      DEBUG(dbgs() << "SLP: added a vector of PHINodes.\n");

      for (unsigned i = 0, e = PH->getNumIncomingValues(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (Value *j : VL)
          Operands.push_back(cast<PHINode>(j)->getIncomingValueForBlock(
              PH->getIncomingBlock(i)));

        buildTree_rec(Operands, Depth + 1);
      }
      return;
    }
    case Instruction::ExtractValue:
    case Instruction::ExtractElement: {
      bool Reuse = canReuseExtract(VL, Opcode);
      if (Reuse) {
        DEBUG(dbgs() << "SLP: Reusing extract sequence.\n");
      } else {
        BS.cancelScheduling(VL);
      }
      newTreeEntry(VL, Reuse, false);
      return;
    }
    case Instruction::Load: {
      // Check that a vectorized load would load the same memory as a scalar
      // load.
      // For example we don't want vectorize loads that are smaller than 8 bit.
      // Even though we have a packed struct {<i2, i2, i2, i2>} LLVM treats
      // loading/storing it as an i8 struct. If we vectorize loads/stores from
      // such a struct we read/write packed bits disagreeing with the
      // unvectorized version.
      Type *ScalarTy = VL[0]->getType();

      if (DL->getTypeSizeInBits(ScalarTy) !=
          DL->getTypeAllocSizeInBits(ScalarTy)) {
        BS.cancelScheduling(VL);
        newTreeEntry(VL, false, false);
        DEBUG(dbgs() << "SLP: Gathering loads of non-packed type.\n");
        return;
      }

      // Make sure all loads in the bundle are simple - we can't vectorize
      // atomic or volatile loads.
      for (unsigned i = 0, e = VL.size() - 1; i < e; ++i) {
        LoadInst *L = cast<LoadInst>(VL[i]);
        if (!L->isSimple()) {
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false, false);
          DEBUG(dbgs() << "SLP: Gathering non-simple loads.\n");
          return;
        }
      }

      // Check if the loads are consecutive, reversed, or neither.
      bool Consecutive = true;
      bool ReverseConsecutive = true;
      for (unsigned i = 0, e = VL.size() - 1; i < e; ++i) {
        if (!isConsecutiveAccess(VL[i], VL[i + 1], *DL, *SE)) {
          Consecutive = false;
          break;
        } else {
          ReverseConsecutive = false;
        }
      }

      if (Consecutive) {
        ++NumLoadsWantToKeepOrder;
        newTreeEntry(VL, true, false);
        DEBUG(dbgs() << "SLP: added a vector of loads.\n");
        return;
      }

      // If none of the load pairs were consecutive when checked in order,
      // check the reverse order.
      if (ReverseConsecutive)
        for (unsigned i = VL.size() - 1; i > 0; --i)
          if (!isConsecutiveAccess(VL[i], VL[i - 1], *DL, *SE)) {
            ReverseConsecutive = false;
            break;
          }

      if (VL.size() > 2 && !ReverseConsecutive) {
        bool ShuffledLoads = true;
        SmallVector<Value *, 8> Sorted;
        if (sortMemAccesses(VL, *DL, *SE, Sorted)) {
          auto NewVL = makeArrayRef(Sorted.begin(), Sorted.end());
          for (unsigned i = 0, e = NewVL.size() - 1; i < e; ++i) {
            if (!isConsecutiveAccess(NewVL[i], NewVL[i + 1], *DL, *SE)) {
              ShuffledLoads = false;
              break;
            }
          }
          if (ShuffledLoads) {
            newTreeEntry(NewVL, true, true);
            return;
          }
        }
      }

      BS.cancelScheduling(VL);
      newTreeEntry(VL, false, false);

      if (ReverseConsecutive) {
        ++NumLoadsWantToChangeOrder;
        DEBUG(dbgs() << "SLP: Gathering reversed loads.\n");
      } else {
        DEBUG(dbgs() << "SLP: Gathering non-consecutive loads.\n");
      }
      return;
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
      Type *SrcTy = VL0->getOperand(0)->getType();
      for (Value *Val : VL) {
        Type *Ty = cast<Instruction>(Val)->getOperand(0)->getType();
        if (Ty != SrcTy || !isValidElementType(Ty)) {
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false, false);
          DEBUG(dbgs() << "SLP: Gathering casts with different src types.\n");
          return;
        }
      }
      newTreeEntry(VL, true, false);
      DEBUG(dbgs() << "SLP: added a vector of casts.\n");

      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (Value *j : VL)
          Operands.push_back(cast<Instruction>(j)->getOperand(i));

        buildTree_rec(Operands, Depth+1);
      }
      return;
    }
    case Instruction::ICmp:
    case Instruction::FCmp: {
      // Check that all of the compares have the same predicate.
      CmpInst::Predicate P0 = cast<CmpInst>(VL0)->getPredicate();
      Type *ComparedTy = cast<Instruction>(VL[0])->getOperand(0)->getType();
      for (unsigned i = 1, e = VL.size(); i < e; ++i) {
        CmpInst *Cmp = cast<CmpInst>(VL[i]);
        if (Cmp->getPredicate() != P0 ||
            Cmp->getOperand(0)->getType() != ComparedTy) {
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false, false);
          DEBUG(dbgs() << "SLP: Gathering cmp with different predicate.\n");
          return;
        }
      }

      newTreeEntry(VL, true, false);
      DEBUG(dbgs() << "SLP: added a vector of compares.\n");

      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (Value *j : VL)
          Operands.push_back(cast<Instruction>(j)->getOperand(i));

        buildTree_rec(Operands, Depth+1);
      }
      return;
    }
    case Instruction::Select:
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
      newTreeEntry(VL, true, false);
      DEBUG(dbgs() << "SLP: added a vector of bin op.\n");

      // Sort operands of the instructions so that each side is more likely to
      // have the same opcode.
      if (isa<BinaryOperator>(VL0) && VL0->isCommutative()) {
        ValueList Left, Right;
        reorderInputsAccordingToOpcode(VL, Left, Right);
        buildTree_rec(Left, Depth + 1);
        buildTree_rec(Right, Depth + 1);
        return;
      }

      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (Value *j : VL)
          Operands.push_back(cast<Instruction>(j)->getOperand(i));

        buildTree_rec(Operands, Depth+1);
      }
      return;
    }
    case Instruction::GetElementPtr: {
      // We don't combine GEPs with complicated (nested) indexing.
      for (Value *Val : VL) {
        if (cast<Instruction>(Val)->getNumOperands() != 2) {
          DEBUG(dbgs() << "SLP: not-vectorizable GEP (nested indexes).\n");
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false, false);
          return;
        }
      }

      // We can't combine several GEPs into one vector if they operate on
      // different types.
      Type *Ty0 = cast<Instruction>(VL0)->getOperand(0)->getType();
      for (Value *Val : VL) {
        Type *CurTy = cast<Instruction>(Val)->getOperand(0)->getType();
        if (Ty0 != CurTy) {
          DEBUG(dbgs() << "SLP: not-vectorizable GEP (different types).\n");
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false, false);
          return;
        }
      }

      // We don't combine GEPs with non-constant indexes.
      for (Value *Val : VL) {
        auto Op = cast<Instruction>(Val)->getOperand(1);
        if (!isa<ConstantInt>(Op)) {
          DEBUG(
              dbgs() << "SLP: not-vectorizable GEP (non-constant indexes).\n");
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false, false);
          return;
        }
      }

      newTreeEntry(VL, true, false);
      DEBUG(dbgs() << "SLP: added a vector of GEPs.\n");
      for (unsigned i = 0, e = 2; i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (Value *j : VL)
          Operands.push_back(cast<Instruction>(j)->getOperand(i));

        buildTree_rec(Operands, Depth + 1);
      }
      return;
    }
    case Instruction::Store: {
      // Check if the stores are consecutive or of we need to swizzle them.
      for (unsigned i = 0, e = VL.size() - 1; i < e; ++i)
        if (!isConsecutiveAccess(VL[i], VL[i + 1], *DL, *SE)) {
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false, false);
          DEBUG(dbgs() << "SLP: Non-consecutive store.\n");
          return;
        }

      newTreeEntry(VL, true, false);
      DEBUG(dbgs() << "SLP: added a vector of stores.\n");

      ValueList Operands;
      for (Value *j : VL)
        Operands.push_back(cast<Instruction>(j)->getOperand(0));

      buildTree_rec(Operands, Depth + 1);
      return;
    }
    case Instruction::Call: {
      // Check if the calls are all to the same vectorizable intrinsic.
      CallInst *CI = cast<CallInst>(VL[0]);
      // Check if this is an Intrinsic call or something that can be
      // represented by an intrinsic call
      Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);
      if (!isTriviallyVectorizable(ID)) {
        BS.cancelScheduling(VL);
        newTreeEntry(VL, false, false);
        DEBUG(dbgs() << "SLP: Non-vectorizable call.\n");
        return;
      }
      Function *Int = CI->getCalledFunction();
      Value *A1I = nullptr;
      if (hasVectorInstrinsicScalarOpd(ID, 1))
        A1I = CI->getArgOperand(1);
      for (unsigned i = 1, e = VL.size(); i != e; ++i) {
        CallInst *CI2 = dyn_cast<CallInst>(VL[i]);
        if (!CI2 || CI2->getCalledFunction() != Int ||
            getVectorIntrinsicIDForCall(CI2, TLI) != ID ||
            !CI->hasIdenticalOperandBundleSchema(*CI2)) {
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false, false);
          DEBUG(dbgs() << "SLP: mismatched calls:" << *CI << "!=" << *VL[i]
                       << "\n");
          return;
        }
        // ctlz,cttz and powi are special intrinsics whose second argument
        // should be same in order for them to be vectorized.
        if (hasVectorInstrinsicScalarOpd(ID, 1)) {
          Value *A1J = CI2->getArgOperand(1);
          if (A1I != A1J) {
            BS.cancelScheduling(VL);
            newTreeEntry(VL, false, false);
            DEBUG(dbgs() << "SLP: mismatched arguments in call:" << *CI
                         << " argument "<< A1I<<"!=" << A1J
                         << "\n");
            return;
          }
        }
        // Verify that the bundle operands are identical between the two calls.
        if (CI->hasOperandBundles() &&
            !std::equal(CI->op_begin() + CI->getBundleOperandsStartIndex(),
                        CI->op_begin() + CI->getBundleOperandsEndIndex(),
                        CI2->op_begin() + CI2->getBundleOperandsStartIndex())) {
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false, false);
          DEBUG(dbgs() << "SLP: mismatched bundle operands in calls:" << *CI << "!="
                       << *VL[i] << '\n');
          return;
        }
      }

      newTreeEntry(VL, true, false);
      for (unsigned i = 0, e = CI->getNumArgOperands(); i != e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (Value *j : VL) {
          CallInst *CI2 = dyn_cast<CallInst>(j);
          Operands.push_back(CI2->getArgOperand(i));
        }
        buildTree_rec(Operands, Depth + 1);
      }
      return;
    }
    case Instruction::ShuffleVector: {
      // If this is not an alternate sequence of opcode like add-sub
      // then do not vectorize this instruction.
      if (!isAltShuffle) {
        BS.cancelScheduling(VL);
        newTreeEntry(VL, false, false);
        DEBUG(dbgs() << "SLP: ShuffleVector are not vectorized.\n");
        return;
      }
      newTreeEntry(VL, true, false);
      DEBUG(dbgs() << "SLP: added a ShuffleVector op.\n");

      // Reorder operands if reordering would enable vectorization.
      if (isa<BinaryOperator>(VL0)) {
        ValueList Left, Right;
        reorderAltShuffleOperands(VL, Left, Right);
        buildTree_rec(Left, Depth + 1);
        buildTree_rec(Right, Depth + 1);
        return;
      }

      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (Value *j : VL)
          Operands.push_back(cast<Instruction>(j)->getOperand(i));

        buildTree_rec(Operands, Depth + 1);
      }
      return;
    }
    default:
      BS.cancelScheduling(VL);
      newTreeEntry(VL, false, false);
      DEBUG(dbgs() << "SLP: Gathering unknown instruction.\n");
      return;
  }
}

unsigned BoUpSLP::canMapToVector(Type *T, const DataLayout &DL) const {
  unsigned N;
  Type *EltTy;
  auto *ST = dyn_cast<StructType>(T);
  if (ST) {
    N = ST->getNumElements();
    EltTy = *ST->element_begin();
  } else {
    N = cast<ArrayType>(T)->getNumElements();
    EltTy = cast<ArrayType>(T)->getElementType();
  }
  if (!isValidElementType(EltTy))
    return 0;
  uint64_t VTSize = DL.getTypeStoreSizeInBits(VectorType::get(EltTy, N));
  if (VTSize < MinVecRegSize || VTSize > MaxVecRegSize || VTSize != DL.getTypeStoreSizeInBits(T))
    return 0;
  if (ST) {
    // Check that struct is homogeneous.
    for (const auto *Ty : ST->elements())
      if (Ty != EltTy)
        return 0;
  }
  return N;
}

bool BoUpSLP::canReuseExtract(ArrayRef<Value *> VL, unsigned Opcode) const {
  assert(Opcode == Instruction::ExtractElement ||
         Opcode == Instruction::ExtractValue);
  assert(Opcode == getSameOpcode(VL) && "Invalid opcode");
  // Check if all of the extracts come from the same vector and from the
  // correct offset.
  Value *VL0 = VL[0];
  Instruction *E0 = cast<Instruction>(VL0);
  Value *Vec = E0->getOperand(0);

  // We have to extract from a vector/aggregate with the same number of elements.
  unsigned NElts;
  if (Opcode == Instruction::ExtractValue) {
    const DataLayout &DL = E0->getModule()->getDataLayout();
    NElts = canMapToVector(Vec->getType(), DL);
    if (!NElts)
      return false;
    // Check if load can be rewritten as load of vector.
    LoadInst *LI = dyn_cast<LoadInst>(Vec);
    if (!LI || !LI->isSimple() || !LI->hasNUses(VL.size()))
      return false;
  } else {
    NElts = Vec->getType()->getVectorNumElements();
  }

  if (NElts != VL.size())
    return false;

  // Check that all of the indices extract from the correct offset.
  if (!matchExtractIndex(E0, 0, Opcode))
    return false;

  for (unsigned i = 1, e = VL.size(); i < e; ++i) {
    Instruction *E = cast<Instruction>(VL[i]);
    if (!matchExtractIndex(E, i, Opcode))
      return false;
    if (E->getOperand(0) != Vec)
      return false;
  }

  return true;
}

int BoUpSLP::getEntryCost(TreeEntry *E) {
  ArrayRef<Value*> VL = E->Scalars;

  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());

  // If we have computed a smaller type for the expression, update VecTy so
  // that the costs will be accurate.
  if (MinBWs.count(VL[0]))
    VecTy = VectorType::get(
        IntegerType::get(F->getContext(), MinBWs[VL[0]].first), VL.size());

  if (E->NeedToGather) {
    if (allConstant(VL))
      return 0;
    if (isSplat(VL)) {
      return TTI->getShuffleCost(TargetTransformInfo::SK_Broadcast, VecTy, 0);
    }
    return getGatherCost(E->Scalars);
  }
  unsigned Opcode = getSameOpcode(VL);
  assert(Opcode && allSameType(VL) && allSameBlock(VL) && "Invalid VL");
  Instruction *VL0 = cast<Instruction>(VL[0]);
  switch (Opcode) {
    case Instruction::PHI: {
      return 0;
    }
    case Instruction::ExtractValue:
    case Instruction::ExtractElement: {
      if (canReuseExtract(VL, Opcode)) {
        int DeadCost = 0;
        for (unsigned i = 0, e = VL.size(); i < e; ++i) {
          Instruction *E = cast<Instruction>(VL[i]);
          // If all users are going to be vectorized, instruction can be
          // considered as dead.
          // The same, if have only one user, it will be vectorized for sure.
          if (E->hasOneUse() ||
              std::all_of(E->user_begin(), E->user_end(), [this](User *U) {
                return ScalarToTreeEntry.count(U) > 0;
              }))
            // Take credit for instruction that will become dead.
            DeadCost +=
                TTI->getVectorInstrCost(Instruction::ExtractElement, VecTy, i);
        }
        return -DeadCost;
      }
      return getGatherCost(VecTy);
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
      Type *SrcTy = VL0->getOperand(0)->getType();

      // Calculate the cost of this instruction.
      int ScalarCost = VL.size() * TTI->getCastInstrCost(VL0->getOpcode(),
                                                         VL0->getType(), SrcTy);

      VectorType *SrcVecTy = VectorType::get(SrcTy, VL.size());
      int VecCost = TTI->getCastInstrCost(VL0->getOpcode(), VecTy, SrcVecTy);
      return VecCost - ScalarCost;
    }
    case Instruction::FCmp:
    case Instruction::ICmp:
    case Instruction::Select: {
      // Calculate the cost of this instruction.
      VectorType *MaskTy = VectorType::get(Builder.getInt1Ty(), VL.size());
      int ScalarCost = VecTy->getNumElements() *
          TTI->getCmpSelInstrCost(Opcode, ScalarTy, Builder.getInt1Ty());
      int VecCost = TTI->getCmpSelInstrCost(Opcode, VecTy, MaskTy);
      return VecCost - ScalarCost;
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
      // Certain instructions can be cheaper to vectorize if they have a
      // constant second vector operand.
      TargetTransformInfo::OperandValueKind Op1VK =
          TargetTransformInfo::OK_AnyValue;
      TargetTransformInfo::OperandValueKind Op2VK =
          TargetTransformInfo::OK_UniformConstantValue;
      TargetTransformInfo::OperandValueProperties Op1VP =
          TargetTransformInfo::OP_None;
      TargetTransformInfo::OperandValueProperties Op2VP =
          TargetTransformInfo::OP_None;

      // If all operands are exactly the same ConstantInt then set the
      // operand kind to OK_UniformConstantValue.
      // If instead not all operands are constants, then set the operand kind
      // to OK_AnyValue. If all operands are constants but not the same,
      // then set the operand kind to OK_NonUniformConstantValue.
      ConstantInt *CInt = nullptr;
      for (unsigned i = 0; i < VL.size(); ++i) {
        const Instruction *I = cast<Instruction>(VL[i]);
        if (!isa<ConstantInt>(I->getOperand(1))) {
          Op2VK = TargetTransformInfo::OK_AnyValue;
          break;
        }
        if (i == 0) {
          CInt = cast<ConstantInt>(I->getOperand(1));
          continue;
        }
        if (Op2VK == TargetTransformInfo::OK_UniformConstantValue &&
            CInt != cast<ConstantInt>(I->getOperand(1)))
          Op2VK = TargetTransformInfo::OK_NonUniformConstantValue;
      }
      // FIXME: Currently cost of model modification for division by power of
      // 2 is handled for X86 and AArch64. Add support for other targets.
      if (Op2VK == TargetTransformInfo::OK_UniformConstantValue && CInt &&
          CInt->getValue().isPowerOf2())
        Op2VP = TargetTransformInfo::OP_PowerOf2;

      int ScalarCost = VecTy->getNumElements() *
                       TTI->getArithmeticInstrCost(Opcode, ScalarTy, Op1VK,
                                                   Op2VK, Op1VP, Op2VP);
      int VecCost = TTI->getArithmeticInstrCost(Opcode, VecTy, Op1VK, Op2VK,
                                                Op1VP, Op2VP);
      return VecCost - ScalarCost;
    }
    case Instruction::GetElementPtr: {
      TargetTransformInfo::OperandValueKind Op1VK =
          TargetTransformInfo::OK_AnyValue;
      TargetTransformInfo::OperandValueKind Op2VK =
          TargetTransformInfo::OK_UniformConstantValue;

      int ScalarCost =
          VecTy->getNumElements() *
          TTI->getArithmeticInstrCost(Instruction::Add, ScalarTy, Op1VK, Op2VK);
      int VecCost =
          TTI->getArithmeticInstrCost(Instruction::Add, VecTy, Op1VK, Op2VK);

      return VecCost - ScalarCost;
    }
    case Instruction::Load: {
      // Cost of wide load - cost of scalar loads.
      unsigned alignment = dyn_cast<LoadInst>(VL0)->getAlignment();
      int ScalarLdCost = VecTy->getNumElements() *
            TTI->getMemoryOpCost(Instruction::Load, ScalarTy, alignment, 0);
      int VecLdCost = TTI->getMemoryOpCost(Instruction::Load,
                                           VecTy, alignment, 0);
      if (E->NeedToShuffle) {
        VecLdCost += TTI->getShuffleCost(
            TargetTransformInfo::SK_PermuteSingleSrc, VecTy, 0);
      }
      return VecLdCost - ScalarLdCost;
    }
    case Instruction::Store: {
      // We know that we can merge the stores. Calculate the cost.
      unsigned alignment = dyn_cast<StoreInst>(VL0)->getAlignment();
      int ScalarStCost = VecTy->getNumElements() *
            TTI->getMemoryOpCost(Instruction::Store, ScalarTy, alignment, 0);
      int VecStCost = TTI->getMemoryOpCost(Instruction::Store,
                                           VecTy, alignment, 0);
      return VecStCost - ScalarStCost;
    }
    case Instruction::Call: {
      CallInst *CI = cast<CallInst>(VL0);
      Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);

      // Calculate the cost of the scalar and vector calls.
      SmallVector<Type*, 4> ScalarTys, VecTys;
      for (unsigned op = 0, opc = CI->getNumArgOperands(); op!= opc; ++op) {
        ScalarTys.push_back(CI->getArgOperand(op)->getType());
        VecTys.push_back(VectorType::get(CI->getArgOperand(op)->getType(),
                                         VecTy->getNumElements()));
      }

      FastMathFlags FMF;
      if (auto *FPMO = dyn_cast<FPMathOperator>(CI))
        FMF = FPMO->getFastMathFlags();

      int ScalarCallCost = VecTy->getNumElements() *
          TTI->getIntrinsicInstrCost(ID, ScalarTy, ScalarTys, FMF);

      int VecCallCost = TTI->getIntrinsicInstrCost(ID, VecTy, VecTys, FMF);

      DEBUG(dbgs() << "SLP: Call cost "<< VecCallCost - ScalarCallCost
            << " (" << VecCallCost  << "-" <<  ScalarCallCost << ")"
            << " for " << *CI << "\n");

      return VecCallCost - ScalarCallCost;
    }
    case Instruction::ShuffleVector: {
      TargetTransformInfo::OperandValueKind Op1VK =
          TargetTransformInfo::OK_AnyValue;
      TargetTransformInfo::OperandValueKind Op2VK =
          TargetTransformInfo::OK_AnyValue;
      int ScalarCost = 0;
      int VecCost = 0;
      for (Value *i : VL) {
        Instruction *I = cast<Instruction>(i);
        if (!I)
          break;
        ScalarCost +=
            TTI->getArithmeticInstrCost(I->getOpcode(), ScalarTy, Op1VK, Op2VK);
      }
      // VecCost is equal to sum of the cost of creating 2 vectors
      // and the cost of creating shuffle.
      Instruction *I0 = cast<Instruction>(VL[0]);
      VecCost =
          TTI->getArithmeticInstrCost(I0->getOpcode(), VecTy, Op1VK, Op2VK);
      Instruction *I1 = cast<Instruction>(VL[1]);
      VecCost +=
          TTI->getArithmeticInstrCost(I1->getOpcode(), VecTy, Op1VK, Op2VK);
      VecCost +=
          TTI->getShuffleCost(TargetTransformInfo::SK_Alternate, VecTy, 0);
      return VecCost - ScalarCost;
    }
    default:
      llvm_unreachable("Unknown instruction");
  }
}

bool BoUpSLP::isFullyVectorizableTinyTree() {
  DEBUG(dbgs() << "SLP: Check whether the tree with height " <<
        VectorizableTree.size() << " is fully vectorizable .\n");

  // We only handle trees of heights 1 and 2.
  if (VectorizableTree.size() == 1 && !VectorizableTree[0].NeedToGather)
    return true;

  if (VectorizableTree.size() != 2)
    return false;

  // Handle splat and all-constants stores.
  if (!VectorizableTree[0].NeedToGather &&
      (allConstant(VectorizableTree[1].Scalars) ||
       isSplat(VectorizableTree[1].Scalars)))
    return true;

  // Gathering cost would be too much for tiny trees.
  if (VectorizableTree[0].NeedToGather || VectorizableTree[1].NeedToGather)
    return false;

  return true;
}

bool BoUpSLP::isTreeTinyAndNotFullyVectorizable() {

  // We can vectorize the tree if its size is greater than or equal to the
  // minimum size specified by the MinTreeSize command line option.
  if (VectorizableTree.size() >= MinTreeSize)
    return false;

  // If we have a tiny tree (a tree whose size is less than MinTreeSize), we
  // can vectorize it if we can prove it fully vectorizable.
  if (isFullyVectorizableTinyTree())
    return false;

  assert(VectorizableTree.empty()
             ? ExternalUses.empty()
             : true && "We shouldn't have any external users");

  // Otherwise, we can't vectorize the tree. It is both tiny and not fully
  // vectorizable.
  return true;
}

int BoUpSLP::getSpillCost() {
  // Walk from the bottom of the tree to the top, tracking which values are
  // live. When we see a call instruction that is not part of our tree,
  // query TTI to see if there is a cost to keeping values live over it
  // (for example, if spills and fills are required).
  unsigned BundleWidth = VectorizableTree.front().Scalars.size();
  int Cost = 0;

  SmallPtrSet<Instruction*, 4> LiveValues;
  Instruction *PrevInst = nullptr;

  for (const auto &N : VectorizableTree) {
    Instruction *Inst = dyn_cast<Instruction>(N.Scalars[0]);
    if (!Inst)
      continue;

    if (!PrevInst) {
      PrevInst = Inst;
      continue;
    }

    // Update LiveValues.
    LiveValues.erase(PrevInst);
    for (auto &J : PrevInst->operands()) {
      if (isa<Instruction>(&*J) && ScalarToTreeEntry.count(&*J))
        LiveValues.insert(cast<Instruction>(&*J));
    }

    DEBUG(
      dbgs() << "SLP: #LV: " << LiveValues.size();
      for (auto *X : LiveValues)
        dbgs() << " " << X->getName();
      dbgs() << ", Looking at ";
      Inst->dump();
      );

    // Now find the sequence of instructions between PrevInst and Inst.
    BasicBlock::reverse_iterator InstIt = ++Inst->getIterator().getReverse(),
                                 PrevInstIt =
                                     PrevInst->getIterator().getReverse();
    while (InstIt != PrevInstIt) {
      if (PrevInstIt == PrevInst->getParent()->rend()) {
        PrevInstIt = Inst->getParent()->rbegin();
        continue;
      }

      if (isa<CallInst>(&*PrevInstIt) && &*PrevInstIt != PrevInst) {
        SmallVector<Type*, 4> V;
        for (auto *II : LiveValues)
          V.push_back(VectorType::get(II->getType(), BundleWidth));
        Cost += TTI->getCostOfKeepingLiveOverCall(V);
      }

      ++PrevInstIt;
    }

    PrevInst = Inst;
  }

  return Cost;
}

int BoUpSLP::getTreeCost() {
  int Cost = 0;
  DEBUG(dbgs() << "SLP: Calculating cost for tree of size " <<
        VectorizableTree.size() << ".\n");

  unsigned BundleWidth = VectorizableTree[0].Scalars.size();

  for (TreeEntry &TE : VectorizableTree) {
    int C = getEntryCost(&TE);
    DEBUG(dbgs() << "SLP: Adding cost " << C << " for bundle that starts with "
                 << *TE.Scalars[0] << ".\n");
    Cost += C;
  }

  SmallSet<Value *, 16> ExtractCostCalculated;
  int ExtractCost = 0;
  for (ExternalUser &EU : ExternalUses) {
    // We only add extract cost once for the same scalar.
    if (!ExtractCostCalculated.insert(EU.Scalar).second)
      continue;

    // Uses by ephemeral values are free (because the ephemeral value will be
    // removed prior to code generation, and so the extraction will be
    // removed as well).
    if (EphValues.count(EU.User))
      continue;

    // If we plan to rewrite the tree in a smaller type, we will need to sign
    // extend the extracted value back to the original type. Here, we account
    // for the extract and the added cost of the sign extend if needed.
    auto *VecTy = VectorType::get(EU.Scalar->getType(), BundleWidth);
    auto *ScalarRoot = VectorizableTree[0].Scalars[0];
    if (MinBWs.count(ScalarRoot)) {
      auto *MinTy = IntegerType::get(F->getContext(), MinBWs[ScalarRoot].first);
      auto Extend =
          MinBWs[ScalarRoot].second ? Instruction::SExt : Instruction::ZExt;
      VecTy = VectorType::get(MinTy, BundleWidth);
      ExtractCost += TTI->getExtractWithExtendCost(Extend, EU.Scalar->getType(),
                                                   VecTy, EU.Lane);
    } else {
      ExtractCost +=
          TTI->getVectorInstrCost(Instruction::ExtractElement, VecTy, EU.Lane);
    }
  }

  int SpillCost = getSpillCost();
  Cost += SpillCost + ExtractCost;

  DEBUG(dbgs() << "SLP: Spill Cost = " << SpillCost << ".\n"
               << "SLP: Extract Cost = " << ExtractCost << ".\n"
               << "SLP: Total Cost = " << Cost << ".\n");
  return Cost;
}

int BoUpSLP::getGatherCost(Type *Ty) {
  int Cost = 0;
  for (unsigned i = 0, e = cast<VectorType>(Ty)->getNumElements(); i < e; ++i)
    Cost += TTI->getVectorInstrCost(Instruction::InsertElement, Ty, i);
  return Cost;
}

int BoUpSLP::getGatherCost(ArrayRef<Value *> VL) {
  // Find the type of the operands in VL.
  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());
  // Find the cost of inserting/extracting values from the vector.
  return getGatherCost(VecTy);
}

// Reorder commutative operations in alternate shuffle if the resulting vectors
// are consecutive loads. This would allow us to vectorize the tree.
// If we have something like-
// load a[0] - load b[0]
// load b[1] + load a[1]
// load a[2] - load b[2]
// load a[3] + load b[3]
// Reordering the second load b[1]  load a[1] would allow us to vectorize this
// code.
void BoUpSLP::reorderAltShuffleOperands(ArrayRef<Value *> VL,
                                        SmallVectorImpl<Value *> &Left,
                                        SmallVectorImpl<Value *> &Right) {
  // Push left and right operands of binary operation into Left and Right
  for (Value *i : VL) {
    Left.push_back(cast<Instruction>(i)->getOperand(0));
    Right.push_back(cast<Instruction>(i)->getOperand(1));
  }

  // Reorder if we have a commutative operation and consecutive access
  // are on either side of the alternate instructions.
  for (unsigned j = 0; j < VL.size() - 1; ++j) {
    if (LoadInst *L = dyn_cast<LoadInst>(Left[j])) {
      if (LoadInst *L1 = dyn_cast<LoadInst>(Right[j + 1])) {
        Instruction *VL1 = cast<Instruction>(VL[j]);
        Instruction *VL2 = cast<Instruction>(VL[j + 1]);
        if (VL1->isCommutative() && isConsecutiveAccess(L, L1, *DL, *SE)) {
          std::swap(Left[j], Right[j]);
          continue;
        } else if (VL2->isCommutative() &&
                   isConsecutiveAccess(L, L1, *DL, *SE)) {
          std::swap(Left[j + 1], Right[j + 1]);
          continue;
        }
        // else unchanged
      }
    }
    if (LoadInst *L = dyn_cast<LoadInst>(Right[j])) {
      if (LoadInst *L1 = dyn_cast<LoadInst>(Left[j + 1])) {
        Instruction *VL1 = cast<Instruction>(VL[j]);
        Instruction *VL2 = cast<Instruction>(VL[j + 1]);
        if (VL1->isCommutative() && isConsecutiveAccess(L, L1, *DL, *SE)) {
          std::swap(Left[j], Right[j]);
          continue;
        } else if (VL2->isCommutative() &&
                   isConsecutiveAccess(L, L1, *DL, *SE)) {
          std::swap(Left[j + 1], Right[j + 1]);
          continue;
        }
        // else unchanged
      }
    }
  }
}

// Return true if I should be commuted before adding it's left and right
// operands to the arrays Left and Right.
//
// The vectorizer is trying to either have all elements one side being
// instruction with the same opcode to enable further vectorization, or having
// a splat to lower the vectorizing cost.
static bool shouldReorderOperands(int i, Instruction &I,
                                  SmallVectorImpl<Value *> &Left,
                                  SmallVectorImpl<Value *> &Right,
                                  bool AllSameOpcodeLeft,
                                  bool AllSameOpcodeRight, bool SplatLeft,
                                  bool SplatRight) {
  Value *VLeft = I.getOperand(0);
  Value *VRight = I.getOperand(1);
  // If we have "SplatRight", try to see if commuting is needed to preserve it.
  if (SplatRight) {
    if (VRight == Right[i - 1])
      // Preserve SplatRight
      return false;
    if (VLeft == Right[i - 1]) {
      // Commuting would preserve SplatRight, but we don't want to break
      // SplatLeft either, i.e. preserve the original order if possible.
      // (FIXME: why do we care?)
      if (SplatLeft && VLeft == Left[i - 1])
        return false;
      return true;
    }
  }
  // Symmetrically handle Right side.
  if (SplatLeft) {
    if (VLeft == Left[i - 1])
      // Preserve SplatLeft
      return false;
    if (VRight == Left[i - 1])
      return true;
  }

  Instruction *ILeft = dyn_cast<Instruction>(VLeft);
  Instruction *IRight = dyn_cast<Instruction>(VRight);

  // If we have "AllSameOpcodeRight", try to see if the left operands preserves
  // it and not the right, in this case we want to commute.
  if (AllSameOpcodeRight) {
    unsigned RightPrevOpcode = cast<Instruction>(Right[i - 1])->getOpcode();
    if (IRight && RightPrevOpcode == IRight->getOpcode())
      // Do not commute, a match on the right preserves AllSameOpcodeRight
      return false;
    if (ILeft && RightPrevOpcode == ILeft->getOpcode()) {
      // We have a match and may want to commute, but first check if there is
      // not also a match on the existing operands on the Left to preserve
      // AllSameOpcodeLeft, i.e. preserve the original order if possible.
      // (FIXME: why do we care?)
      if (AllSameOpcodeLeft && ILeft &&
          cast<Instruction>(Left[i - 1])->getOpcode() == ILeft->getOpcode())
        return false;
      return true;
    }
  }
  // Symmetrically handle Left side.
  if (AllSameOpcodeLeft) {
    unsigned LeftPrevOpcode = cast<Instruction>(Left[i - 1])->getOpcode();
    if (ILeft && LeftPrevOpcode == ILeft->getOpcode())
      return false;
    if (IRight && LeftPrevOpcode == IRight->getOpcode())
      return true;
  }
  return false;
}

void BoUpSLP::reorderInputsAccordingToOpcode(ArrayRef<Value *> VL,
                                             SmallVectorImpl<Value *> &Left,
                                             SmallVectorImpl<Value *> &Right) {

  if (VL.size()) {
    // Peel the first iteration out of the loop since there's nothing
    // interesting to do anyway and it simplifies the checks in the loop.
    auto VLeft = cast<Instruction>(VL[0])->getOperand(0);
    auto VRight = cast<Instruction>(VL[0])->getOperand(1);
    if (!isa<Instruction>(VRight) && isa<Instruction>(VLeft))
      // Favor having instruction to the right. FIXME: why?
      std::swap(VLeft, VRight);
    Left.push_back(VLeft);
    Right.push_back(VRight);
  }

  // Keep track if we have instructions with all the same opcode on one side.
  bool AllSameOpcodeLeft = isa<Instruction>(Left[0]);
  bool AllSameOpcodeRight = isa<Instruction>(Right[0]);
  // Keep track if we have one side with all the same value (broadcast).
  bool SplatLeft = true;
  bool SplatRight = true;

  for (unsigned i = 1, e = VL.size(); i != e; ++i) {
    Instruction *I = cast<Instruction>(VL[i]);
    assert(I->isCommutative() && "Can only process commutative instruction");
    // Commute to favor either a splat or maximizing having the same opcodes on
    // one side.
    if (shouldReorderOperands(i, *I, Left, Right, AllSameOpcodeLeft,
                              AllSameOpcodeRight, SplatLeft, SplatRight)) {
      Left.push_back(I->getOperand(1));
      Right.push_back(I->getOperand(0));
    } else {
      Left.push_back(I->getOperand(0));
      Right.push_back(I->getOperand(1));
    }
    // Update Splat* and AllSameOpcode* after the insertion.
    SplatRight = SplatRight && (Right[i - 1] == Right[i]);
    SplatLeft = SplatLeft && (Left[i - 1] == Left[i]);
    AllSameOpcodeLeft = AllSameOpcodeLeft && isa<Instruction>(Left[i]) &&
                        (cast<Instruction>(Left[i - 1])->getOpcode() ==
                         cast<Instruction>(Left[i])->getOpcode());
    AllSameOpcodeRight = AllSameOpcodeRight && isa<Instruction>(Right[i]) &&
                         (cast<Instruction>(Right[i - 1])->getOpcode() ==
                          cast<Instruction>(Right[i])->getOpcode());
  }

  // If one operand end up being broadcast, return this operand order.
  if (SplatRight || SplatLeft)
    return;

  // Finally check if we can get longer vectorizable chain by reordering
  // without breaking the good operand order detected above.
  // E.g. If we have something like-
  // load a[0]  load b[0]
  // load b[1]  load a[1]
  // load a[2]  load b[2]
  // load a[3]  load b[3]
  // Reordering the second load b[1]  load a[1] would allow us to vectorize
  // this code and we still retain AllSameOpcode property.
  // FIXME: This load reordering might break AllSameOpcode in some rare cases
  // such as-
  // add a[0],c[0]  load b[0]
  // add a[1],c[2]  load b[1]
  // b[2]           load b[2]
  // add a[3],c[3]  load b[3]
  for (unsigned j = 0; j < VL.size() - 1; ++j) {
    if (LoadInst *L = dyn_cast<LoadInst>(Left[j])) {
      if (LoadInst *L1 = dyn_cast<LoadInst>(Right[j + 1])) {
        if (isConsecutiveAccess(L, L1, *DL, *SE)) {
          std::swap(Left[j + 1], Right[j + 1]);
          continue;
        }
      }
    }
    if (LoadInst *L = dyn_cast<LoadInst>(Right[j])) {
      if (LoadInst *L1 = dyn_cast<LoadInst>(Left[j + 1])) {
        if (isConsecutiveAccess(L, L1, *DL, *SE)) {
          std::swap(Left[j + 1], Right[j + 1]);
          continue;
        }
      }
    }
    // else unchanged
  }
}

void BoUpSLP::setInsertPointAfterBundle(ArrayRef<Value *> VL) {

  // Get the basic block this bundle is in. All instructions in the bundle
  // should be in this block.
  auto *Front = cast<Instruction>(VL.front());
  auto *BB = Front->getParent();
  assert(all_of(make_range(VL.begin(), VL.end()), [&](Value *V) -> bool {
    return cast<Instruction>(V)->getParent() == BB;
  }));

  // The last instruction in the bundle in program order.
  Instruction *LastInst = nullptr;

  // Find the last instruction. The common case should be that BB has been
  // scheduled, and the last instruction is VL.back(). So we start with
  // VL.back() and iterate over schedule data until we reach the end of the
  // bundle. The end of the bundle is marked by null ScheduleData.
  if (BlocksSchedules.count(BB)) {
    auto *Bundle = BlocksSchedules[BB]->getScheduleData(VL.back());
    if (Bundle && Bundle->isPartOfBundle())
      for (; Bundle; Bundle = Bundle->NextInBundle)
        LastInst = Bundle->Inst;
  }

  // LastInst can still be null at this point if there's either not an entry
  // for BB in BlocksSchedules or there's no ScheduleData available for
  // VL.back(). This can be the case if buildTree_rec aborts for various
  // reasons (e.g., the maximum recursion depth is reached, the maximum region
  // size is reached, etc.). ScheduleData is initialized in the scheduling
  // "dry-run".
  //
  // If this happens, we can still find the last instruction by brute force. We
  // iterate forwards from Front (inclusive) until we either see all
  // instructions in the bundle or reach the end of the block. If Front is the
  // last instruction in program order, LastInst will be set to Front, and we
  // will visit all the remaining instructions in the block.
  //
  // One of the reasons we exit early from buildTree_rec is to place an upper
  // bound on compile-time. Thus, taking an additional compile-time hit here is
  // not ideal. However, this should be exceedingly rare since it requires that
  // we both exit early from buildTree_rec and that the bundle be out-of-order
  // (causing us to iterate all the way to the end of the block).
  if (!LastInst) {
    SmallPtrSet<Value *, 16> Bundle(VL.begin(), VL.end());
    for (auto &I : make_range(BasicBlock::iterator(Front), BB->end())) {
      if (Bundle.erase(&I))
        LastInst = &I;
      if (Bundle.empty())
        break;
    }
  }

  // Set the insertion point after the last instruction in the bundle. Set the
  // debug location to Front.
  Builder.SetInsertPoint(BB, ++LastInst->getIterator());
  Builder.SetCurrentDebugLocation(Front->getDebugLoc());
}

Value *BoUpSLP::Gather(ArrayRef<Value *> VL, VectorType *Ty) {
  Value *Vec = UndefValue::get(Ty);
  // Generate the 'InsertElement' instruction.
  for (unsigned i = 0; i < Ty->getNumElements(); ++i) {
    Vec = Builder.CreateInsertElement(Vec, VL[i], Builder.getInt32(i));
    if (Instruction *Insrt = dyn_cast<Instruction>(Vec)) {
      GatherSeq.insert(Insrt);
      CSEBlocks.insert(Insrt->getParent());

      // Add to our 'need-to-extract' list.
      if (ScalarToTreeEntry.count(VL[i])) {
        int Idx = ScalarToTreeEntry[VL[i]];
        TreeEntry *E = &VectorizableTree[Idx];
        // Find which lane we need to extract.
        int FoundLane = -1;
        for (unsigned Lane = 0, LE = VL.size(); Lane != LE; ++Lane) {
          // Is this the lane of the scalar that we are looking for ?
          if (E->Scalars[Lane] == VL[i]) {
            FoundLane = Lane;
            break;
          }
        }
        assert(FoundLane >= 0 && "Could not find the correct lane");
        ExternalUses.push_back(ExternalUser(VL[i], Insrt, FoundLane));
      }
    }
  }

  return Vec;
}

Value *BoUpSLP::alreadyVectorized(ArrayRef<Value *> VL) const {
  SmallDenseMap<Value*, int>::const_iterator Entry
    = ScalarToTreeEntry.find(VL[0]);
  if (Entry != ScalarToTreeEntry.end()) {
    int Idx = Entry->second;
    const TreeEntry *En = &VectorizableTree[Idx];
    if (En->isSame(VL) && En->VectorizedValue)
      return En->VectorizedValue;
  }
  return nullptr;
}

Value *BoUpSLP::vectorizeTree(ArrayRef<Value *> VL) {
  if (ScalarToTreeEntry.count(VL[0])) {
    int Idx = ScalarToTreeEntry[VL[0]];
    TreeEntry *E = &VectorizableTree[Idx];
    if (E->isSame(VL) || (E->NeedToShuffle && E->isFoundJumbled(VL, *DL, *SE)))
      return vectorizeTree(VL, E);
  }

  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());

  return Gather(VL, VecTy);
}

Value *BoUpSLP::vectorizeTree(ArrayRef<Value *> VL, TreeEntry *E) {
  IRBuilder<>::InsertPointGuard Guard(Builder);

  if (E->VectorizedValue && !E->NeedToShuffle) {
    DEBUG(dbgs() << "SLP: Diamond merged for " << *E->Scalars[0] << ".\n");
    return E->VectorizedValue;
  }

  Instruction *VL0 = cast<Instruction>(E->Scalars[0]);
  Type *ScalarTy = VL0->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL0))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, E->Scalars.size());

  if (E->NeedToGather) {
    setInsertPointAfterBundle(E->Scalars);
    auto *V = Gather(E->Scalars, VecTy);
    E->VectorizedValue = V;
    return V;
  }

  unsigned Opcode = getSameOpcode(E->Scalars);

  switch (Opcode) {
    case Instruction::PHI: {
      PHINode *PH = dyn_cast<PHINode>(VL0);
      Builder.SetInsertPoint(PH->getParent()->getFirstNonPHI());
      Builder.SetCurrentDebugLocation(PH->getDebugLoc());
      PHINode *NewPhi = Builder.CreatePHI(VecTy, PH->getNumIncomingValues());
      E->VectorizedValue = NewPhi;

      // PHINodes may have multiple entries from the same block. We want to
      // visit every block once.
      SmallSet<BasicBlock*, 4> VisitedBBs;

      for (unsigned i = 0, e = PH->getNumIncomingValues(); i < e; ++i) {
        ValueList Operands;
        BasicBlock *IBB = PH->getIncomingBlock(i);

        if (!VisitedBBs.insert(IBB).second) {
          NewPhi->addIncoming(NewPhi->getIncomingValueForBlock(IBB), IBB);
          continue;
        }

        // Prepare the operand vector.
        for (Value *V : E->Scalars)
          Operands.push_back(cast<PHINode>(V)->getIncomingValueForBlock(IBB));

        Builder.SetInsertPoint(IBB->getTerminator());
        Builder.SetCurrentDebugLocation(PH->getDebugLoc());
        Value *Vec = vectorizeTree(Operands);
        NewPhi->addIncoming(Vec, IBB);
      }

      assert(NewPhi->getNumIncomingValues() == PH->getNumIncomingValues() &&
             "Invalid number of incoming values");
      return NewPhi;
    }

    case Instruction::ExtractElement: {
      if (canReuseExtract(E->Scalars, Instruction::ExtractElement)) {
        Value *V = VL0->getOperand(0);
        E->VectorizedValue = V;
        return V;
      }
      setInsertPointAfterBundle(E->Scalars);
      auto *V = Gather(E->Scalars, VecTy);
      E->VectorizedValue = V;
      return V;
    }
    case Instruction::ExtractValue: {
      if (canReuseExtract(E->Scalars, Instruction::ExtractValue)) {
        LoadInst *LI = cast<LoadInst>(VL0->getOperand(0));
        Builder.SetInsertPoint(LI);
        PointerType *PtrTy = PointerType::get(VecTy, LI->getPointerAddressSpace());
        Value *Ptr = Builder.CreateBitCast(LI->getOperand(0), PtrTy);
        LoadInst *V = Builder.CreateAlignedLoad(Ptr, LI->getAlignment());
        E->VectorizedValue = V;
        return propagateMetadata(V, E->Scalars);
      }
      setInsertPointAfterBundle(E->Scalars);
      auto *V = Gather(E->Scalars, VecTy);
      E->VectorizedValue = V;
      return V;
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
      ValueList INVL;
      for (Value *V : E->Scalars)
        INVL.push_back(cast<Instruction>(V)->getOperand(0));

      setInsertPointAfterBundle(E->Scalars);

      Value *InVec = vectorizeTree(INVL);

      if (Value *V = alreadyVectorized(E->Scalars))
        return V;

      CastInst *CI = dyn_cast<CastInst>(VL0);
      Value *V = Builder.CreateCast(CI->getOpcode(), InVec, VecTy);
      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::FCmp:
    case Instruction::ICmp: {
      ValueList LHSV, RHSV;
      for (Value *V : E->Scalars) {
        LHSV.push_back(cast<Instruction>(V)->getOperand(0));
        RHSV.push_back(cast<Instruction>(V)->getOperand(1));
      }

      setInsertPointAfterBundle(E->Scalars);

      Value *L = vectorizeTree(LHSV);
      Value *R = vectorizeTree(RHSV);

      if (Value *V = alreadyVectorized(E->Scalars))
        return V;

      CmpInst::Predicate P0 = cast<CmpInst>(VL0)->getPredicate();
      Value *V;
      if (Opcode == Instruction::FCmp)
        V = Builder.CreateFCmp(P0, L, R);
      else
        V = Builder.CreateICmp(P0, L, R);

      E->VectorizedValue = V;
      propagateIRFlags(E->VectorizedValue, E->Scalars);
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::Select: {
      ValueList TrueVec, FalseVec, CondVec;
      for (Value *V : E->Scalars) {
        CondVec.push_back(cast<Instruction>(V)->getOperand(0));
        TrueVec.push_back(cast<Instruction>(V)->getOperand(1));
        FalseVec.push_back(cast<Instruction>(V)->getOperand(2));
      }

      setInsertPointAfterBundle(E->Scalars);

      Value *Cond = vectorizeTree(CondVec);
      Value *True = vectorizeTree(TrueVec);
      Value *False = vectorizeTree(FalseVec);

      if (Value *V = alreadyVectorized(E->Scalars))
        return V;

      Value *V = Builder.CreateSelect(Cond, True, False);
      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
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
      ValueList LHSVL, RHSVL;
      if (isa<BinaryOperator>(VL0) && VL0->isCommutative())
        reorderInputsAccordingToOpcode(E->Scalars, LHSVL, RHSVL);
      else
        for (Value *V : E->Scalars) {
          LHSVL.push_back(cast<Instruction>(V)->getOperand(0));
          RHSVL.push_back(cast<Instruction>(V)->getOperand(1));
        }

      setInsertPointAfterBundle(E->Scalars);

      Value *LHS = vectorizeTree(LHSVL);
      Value *RHS = vectorizeTree(RHSVL);

      if (Value *V = alreadyVectorized(E->Scalars))
        return V;

      BinaryOperator *BinOp = cast<BinaryOperator>(VL0);
      Value *V = Builder.CreateBinOp(BinOp->getOpcode(), LHS, RHS);
      E->VectorizedValue = V;
      propagateIRFlags(E->VectorizedValue, E->Scalars);
      ++NumVectorInstructions;

      if (Instruction *I = dyn_cast<Instruction>(V))
        return propagateMetadata(I, E->Scalars);

      return V;
    }
    case Instruction::Load: {
      // Loads are inserted at the head of the tree because we don't want to
      // sink them all the way down past store instructions.
      setInsertPointAfterBundle(E->Scalars);

      LoadInst *LI = cast<LoadInst>(VL0);
      Type *ScalarLoadTy = LI->getType();
      unsigned AS = LI->getPointerAddressSpace();

      Value *VecPtr = Builder.CreateBitCast(LI->getPointerOperand(),
                                            VecTy->getPointerTo(AS));

      // The pointer operand uses an in-tree scalar so we add the new BitCast to
      // ExternalUses list to make sure that an extract will be generated in the
      // future.
      if (ScalarToTreeEntry.count(LI->getPointerOperand()))
        ExternalUses.push_back(
            ExternalUser(LI->getPointerOperand(), cast<User>(VecPtr), 0));

      unsigned Alignment = LI->getAlignment();
      LI = Builder.CreateLoad(VecPtr);
      if (!Alignment) {
        Alignment = DL->getABITypeAlignment(ScalarLoadTy);
      }
      LI->setAlignment(Alignment);
      E->VectorizedValue = LI;
      ++NumVectorInstructions;
      propagateMetadata(LI, E->Scalars);

      // As program order of scalar loads are jumbled, the vectorized 'load'
      // must be followed by a 'shuffle' with the required jumbled mask.
      if (!VL.empty() && (E->NeedToShuffle)) {
        assert(VL.size() == E->Scalars.size() &&
               "Equal number of scalars expected");
        SmallVector<Constant *, 8> Mask;
        for (Value *Val : VL) {
          if (ScalarToTreeEntry.count(Val)) {
            int Idx = ScalarToTreeEntry[Val];
            TreeEntry *E = &VectorizableTree[Idx];
            for (unsigned Lane = 0, LE = VL.size(); Lane != LE; ++Lane) {
              if (E->Scalars[Lane] == Val) {
                Mask.push_back(Builder.getInt32(Lane));
                break;
              }
            }
          }
        }

        // Generate shuffle for jumbled memory access
        Value *Undef = UndefValue::get(VecTy);
        Value *Shuf = Builder.CreateShuffleVector((Value *)LI, Undef,
                                                  ConstantVector::get(Mask));
        return Shuf;
      }

      return LI;
    }
    case Instruction::Store: {
      StoreInst *SI = cast<StoreInst>(VL0);
      unsigned Alignment = SI->getAlignment();
      unsigned AS = SI->getPointerAddressSpace();

      ValueList ValueOp;
      for (Value *V : E->Scalars)
        ValueOp.push_back(cast<StoreInst>(V)->getValueOperand());

      setInsertPointAfterBundle(E->Scalars);

      Value *VecValue = vectorizeTree(ValueOp);
      Value *VecPtr = Builder.CreateBitCast(SI->getPointerOperand(),
                                            VecTy->getPointerTo(AS));
      StoreInst *S = Builder.CreateStore(VecValue, VecPtr);

      // The pointer operand uses an in-tree scalar so we add the new BitCast to
      // ExternalUses list to make sure that an extract will be generated in the
      // future.
      if (ScalarToTreeEntry.count(SI->getPointerOperand()))
        ExternalUses.push_back(
            ExternalUser(SI->getPointerOperand(), cast<User>(VecPtr), 0));

      if (!Alignment) {
        Alignment = DL->getABITypeAlignment(SI->getValueOperand()->getType());
      }
      S->setAlignment(Alignment);
      E->VectorizedValue = S;
      ++NumVectorInstructions;
      return propagateMetadata(S, E->Scalars);
    }
    case Instruction::GetElementPtr: {
      setInsertPointAfterBundle(E->Scalars);

      ValueList Op0VL;
      for (Value *V : E->Scalars)
        Op0VL.push_back(cast<GetElementPtrInst>(V)->getOperand(0));

      Value *Op0 = vectorizeTree(Op0VL);

      std::vector<Value *> OpVecs;
      for (int j = 1, e = cast<GetElementPtrInst>(VL0)->getNumOperands(); j < e;
           ++j) {
        ValueList OpVL;
        for (Value *V : E->Scalars)
          OpVL.push_back(cast<GetElementPtrInst>(V)->getOperand(j));

        Value *OpVec = vectorizeTree(OpVL);
        OpVecs.push_back(OpVec);
      }

      Value *V = Builder.CreateGEP(
          cast<GetElementPtrInst>(VL0)->getSourceElementType(), Op0, OpVecs);
      E->VectorizedValue = V;
      ++NumVectorInstructions;

      if (Instruction *I = dyn_cast<Instruction>(V))
        return propagateMetadata(I, E->Scalars);

      return V;
    }
    case Instruction::Call: {
      CallInst *CI = cast<CallInst>(VL0);
      setInsertPointAfterBundle(E->Scalars);
      Function *FI;
      Intrinsic::ID IID  = Intrinsic::not_intrinsic;
      Value *ScalarArg = nullptr;
      if (CI && (FI = CI->getCalledFunction())) {
        IID = FI->getIntrinsicID();
      }
      std::vector<Value *> OpVecs;
      for (int j = 0, e = CI->getNumArgOperands(); j < e; ++j) {
        ValueList OpVL;
        // ctlz,cttz and powi are special intrinsics whose second argument is
        // a scalar. This argument should not be vectorized.
        if (hasVectorInstrinsicScalarOpd(IID, 1) && j == 1) {
          CallInst *CEI = cast<CallInst>(E->Scalars[0]);
          ScalarArg = CEI->getArgOperand(j);
          OpVecs.push_back(CEI->getArgOperand(j));
          continue;
        }
        for (Value *V : E->Scalars) {
          CallInst *CEI = cast<CallInst>(V);
          OpVL.push_back(CEI->getArgOperand(j));
        }

        Value *OpVec = vectorizeTree(OpVL);
        DEBUG(dbgs() << "SLP: OpVec[" << j << "]: " << *OpVec << "\n");
        OpVecs.push_back(OpVec);
      }

      Module *M = F->getParent();
      Intrinsic::ID ID = getVectorIntrinsicIDForCall(CI, TLI);
      Type *Tys[] = { VectorType::get(CI->getType(), E->Scalars.size()) };
      Function *CF = Intrinsic::getDeclaration(M, ID, Tys);
      SmallVector<OperandBundleDef, 1> OpBundles;
      CI->getOperandBundlesAsDefs(OpBundles);
      Value *V = Builder.CreateCall(CF, OpVecs, OpBundles);

      // The scalar argument uses an in-tree scalar so we add the new vectorized
      // call to ExternalUses list to make sure that an extract will be
      // generated in the future.
      if (ScalarArg && ScalarToTreeEntry.count(ScalarArg))
        ExternalUses.push_back(ExternalUser(ScalarArg, cast<User>(V), 0));

      E->VectorizedValue = V;
      propagateIRFlags(E->VectorizedValue, E->Scalars);
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::ShuffleVector: {
      ValueList LHSVL, RHSVL;
      assert(isa<BinaryOperator>(VL0) && "Invalid Shuffle Vector Operand");
      reorderAltShuffleOperands(E->Scalars, LHSVL, RHSVL);
      setInsertPointAfterBundle(E->Scalars);

      Value *LHS = vectorizeTree(LHSVL);
      Value *RHS = vectorizeTree(RHSVL);

      if (Value *V = alreadyVectorized(E->Scalars))
        return V;

      // Create a vector of LHS op1 RHS
      BinaryOperator *BinOp0 = cast<BinaryOperator>(VL0);
      Value *V0 = Builder.CreateBinOp(BinOp0->getOpcode(), LHS, RHS);

      // Create a vector of LHS op2 RHS
      Instruction *VL1 = cast<Instruction>(E->Scalars[1]);
      BinaryOperator *BinOp1 = cast<BinaryOperator>(VL1);
      Value *V1 = Builder.CreateBinOp(BinOp1->getOpcode(), LHS, RHS);

      // Create shuffle to take alternate operations from the vector.
      // Also, gather up odd and even scalar ops to propagate IR flags to
      // each vector operation.
      ValueList OddScalars, EvenScalars;
      unsigned e = E->Scalars.size();
      SmallVector<Constant *, 8> Mask(e);
      for (unsigned i = 0; i < e; ++i) {
        if (i & 1) {
          Mask[i] = Builder.getInt32(e + i);
          OddScalars.push_back(E->Scalars[i]);
        } else {
          Mask[i] = Builder.getInt32(i);
          EvenScalars.push_back(E->Scalars[i]);
        }
      }

      Value *ShuffleMask = ConstantVector::get(Mask);
      propagateIRFlags(V0, EvenScalars);
      propagateIRFlags(V1, OddScalars);

      Value *V = Builder.CreateShuffleVector(V0, V1, ShuffleMask);
      E->VectorizedValue = V;
      ++NumVectorInstructions;
      if (Instruction *I = dyn_cast<Instruction>(V))
        return propagateMetadata(I, E->Scalars);

      return V;
    }
    default:
    llvm_unreachable("unknown inst");
  }
  return nullptr;
}

Value *BoUpSLP::vectorizeTree() {
  ExtraValueToDebugLocsMap ExternallyUsedValues;
  return vectorizeTree(ExternallyUsedValues);
}

Value *
BoUpSLP::vectorizeTree(ExtraValueToDebugLocsMap &ExternallyUsedValues) {

  // All blocks must be scheduled before any instructions are inserted.
  for (auto &BSIter : BlocksSchedules) {
    scheduleBlock(BSIter.second.get());
  }

  Builder.SetInsertPoint(&F->getEntryBlock().front());
  auto *VectorRoot = vectorizeTree(ArrayRef<Value *>(), &VectorizableTree[0]);

  // If the vectorized tree can be rewritten in a smaller type, we truncate the
  // vectorized root. InstCombine will then rewrite the entire expression. We
  // sign extend the extracted values below.
  auto *ScalarRoot = VectorizableTree[0].Scalars[0];
  if (MinBWs.count(ScalarRoot)) {
    if (auto *I = dyn_cast<Instruction>(VectorRoot))
      Builder.SetInsertPoint(&*++BasicBlock::iterator(I));
    auto BundleWidth = VectorizableTree[0].Scalars.size();
    auto *MinTy = IntegerType::get(F->getContext(), MinBWs[ScalarRoot].first);
    auto *VecTy = VectorType::get(MinTy, BundleWidth);
    auto *Trunc = Builder.CreateTrunc(VectorRoot, VecTy);
    VectorizableTree[0].VectorizedValue = Trunc;
  }

  DEBUG(dbgs() << "SLP: Extracting " << ExternalUses.size() << " values .\n");

  // If necessary, sign-extend or zero-extend ScalarRoot to the larger type
  // specified by ScalarType.
  auto extend = [&](Value *ScalarRoot, Value *Ex, Type *ScalarType) {
    if (!MinBWs.count(ScalarRoot))
      return Ex;
    if (MinBWs[ScalarRoot].second)
      return Builder.CreateSExt(Ex, ScalarType);
    return Builder.CreateZExt(Ex, ScalarType);
  };

  // Extract all of the elements with the external uses.
  for (const auto &ExternalUse : ExternalUses) {
    Value *Scalar = ExternalUse.Scalar;
    llvm::User *User = ExternalUse.User;

    // Skip users that we already RAUW. This happens when one instruction
    // has multiple uses of the same value.
    if (User && !is_contained(Scalar->users(), User))
      continue;
    assert(ScalarToTreeEntry.count(Scalar) && "Invalid scalar");

    int Idx = ScalarToTreeEntry[Scalar];
    TreeEntry *E = &VectorizableTree[Idx];
    assert(!E->NeedToGather && "Extracting from a gather list");

    Value *Vec = E->VectorizedValue;
    assert(Vec && "Can't find vectorizable value");

    Value *Lane = Builder.getInt32(ExternalUse.Lane);
    // If User == nullptr, the Scalar is used as extra arg. Generate
    // ExtractElement instruction and update the record for this scalar in
    // ExternallyUsedValues.
    if (!User) {
      assert(ExternallyUsedValues.count(Scalar) &&
             "Scalar with nullptr as an external user must be registered in "
             "ExternallyUsedValues map");
      if (auto *VecI = dyn_cast<Instruction>(Vec)) {
        Builder.SetInsertPoint(VecI->getParent(),
                               std::next(VecI->getIterator()));
      } else {
        Builder.SetInsertPoint(&F->getEntryBlock().front());
      }
      Value *Ex = Builder.CreateExtractElement(Vec, Lane);
      Ex = extend(ScalarRoot, Ex, Scalar->getType());
      CSEBlocks.insert(cast<Instruction>(Scalar)->getParent());
      auto &Locs = ExternallyUsedValues[Scalar];
      ExternallyUsedValues.insert({Ex, Locs});
      ExternallyUsedValues.erase(Scalar);
      continue;
    }

    // Generate extracts for out-of-tree users.
    // Find the insertion point for the extractelement lane.
    if (auto *VecI = dyn_cast<Instruction>(Vec)) {
      if (PHINode *PH = dyn_cast<PHINode>(User)) {
        for (int i = 0, e = PH->getNumIncomingValues(); i != e; ++i) {
          if (PH->getIncomingValue(i) == Scalar) {
            TerminatorInst *IncomingTerminator =
                PH->getIncomingBlock(i)->getTerminator();
            if (isa<CatchSwitchInst>(IncomingTerminator)) {
              Builder.SetInsertPoint(VecI->getParent(),
                                     std::next(VecI->getIterator()));
            } else {
              Builder.SetInsertPoint(PH->getIncomingBlock(i)->getTerminator());
            }
            Value *Ex = Builder.CreateExtractElement(Vec, Lane);
            Ex = extend(ScalarRoot, Ex, Scalar->getType());
            CSEBlocks.insert(PH->getIncomingBlock(i));
            PH->setOperand(i, Ex);
          }
        }
      } else {
        Builder.SetInsertPoint(cast<Instruction>(User));
        Value *Ex = Builder.CreateExtractElement(Vec, Lane);
        Ex = extend(ScalarRoot, Ex, Scalar->getType());
        CSEBlocks.insert(cast<Instruction>(User)->getParent());
        User->replaceUsesOfWith(Scalar, Ex);
     }
    } else {
      Builder.SetInsertPoint(&F->getEntryBlock().front());
      Value *Ex = Builder.CreateExtractElement(Vec, Lane);
      Ex = extend(ScalarRoot, Ex, Scalar->getType());
      CSEBlocks.insert(&F->getEntryBlock());
      User->replaceUsesOfWith(Scalar, Ex);
    }

    DEBUG(dbgs() << "SLP: Replaced:" << *User << ".\n");
  }

  // For each vectorized value:
  for (TreeEntry &EIdx : VectorizableTree) {
    TreeEntry *Entry = &EIdx;

    // For each lane:
    for (int Lane = 0, LE = Entry->Scalars.size(); Lane != LE; ++Lane) {
      Value *Scalar = Entry->Scalars[Lane];
      // No need to handle users of gathered values.
      if (Entry->NeedToGather)
        continue;

      assert(Entry->VectorizedValue && "Can't find vectorizable value");

      Type *Ty = Scalar->getType();
      if (!Ty->isVoidTy()) {
#ifndef NDEBUG
        for (User *U : Scalar->users()) {
          DEBUG(dbgs() << "SLP: \tvalidating user:" << *U << ".\n");

          assert((ScalarToTreeEntry.count(U) ||
                  // It is legal to replace users in the ignorelist by undef.
                  is_contained(UserIgnoreList, U)) &&
                 "Replacing out-of-tree value with undef");
        }
#endif
        Value *Undef = UndefValue::get(Ty);
        Scalar->replaceAllUsesWith(Undef);
      }
      DEBUG(dbgs() << "SLP: \tErasing scalar:" << *Scalar << ".\n");
      eraseInstruction(cast<Instruction>(Scalar));
    }
  }

  Builder.ClearInsertionPoint();

  return VectorizableTree[0].VectorizedValue;
}

void BoUpSLP::optimizeGatherSequence() {
  DEBUG(dbgs() << "SLP: Optimizing " << GatherSeq.size()
        << " gather sequences instructions.\n");
  // LICM InsertElementInst sequences.
  for (Instruction *it : GatherSeq) {
    InsertElementInst *Insert = dyn_cast<InsertElementInst>(it);

    if (!Insert)
      continue;

    // Check if this block is inside a loop.
    Loop *L = LI->getLoopFor(Insert->getParent());
    if (!L)
      continue;

    // Check if it has a preheader.
    BasicBlock *PreHeader = L->getLoopPreheader();
    if (!PreHeader)
      continue;

    // If the vector or the element that we insert into it are
    // instructions that are defined in this basic block then we can't
    // hoist this instruction.
    Instruction *CurrVec = dyn_cast<Instruction>(Insert->getOperand(0));
    Instruction *NewElem = dyn_cast<Instruction>(Insert->getOperand(1));
    if (CurrVec && L->contains(CurrVec))
      continue;
    if (NewElem && L->contains(NewElem))
      continue;

    // We can hoist this instruction. Move it to the pre-header.
    Insert->moveBefore(PreHeader->getTerminator());
  }

  // Make a list of all reachable blocks in our CSE queue.
  SmallVector<const DomTreeNode *, 8> CSEWorkList;
  CSEWorkList.reserve(CSEBlocks.size());
  for (BasicBlock *BB : CSEBlocks)
    if (DomTreeNode *N = DT->getNode(BB)) {
      assert(DT->isReachableFromEntry(N));
      CSEWorkList.push_back(N);
    }

  // Sort blocks by domination. This ensures we visit a block after all blocks
  // dominating it are visited.
  std::stable_sort(CSEWorkList.begin(), CSEWorkList.end(),
                   [this](const DomTreeNode *A, const DomTreeNode *B) {
    return DT->properlyDominates(A, B);
  });

  // Perform O(N^2) search over the gather sequences and merge identical
  // instructions. TODO: We can further optimize this scan if we split the
  // instructions into different buckets based on the insert lane.
  SmallVector<Instruction *, 16> Visited;
  for (auto I = CSEWorkList.begin(), E = CSEWorkList.end(); I != E; ++I) {
    assert((I == CSEWorkList.begin() || !DT->dominates(*I, *std::prev(I))) &&
           "Worklist not sorted properly!");
    BasicBlock *BB = (*I)->getBlock();
    // For all instructions in blocks containing gather sequences:
    for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e;) {
      Instruction *In = &*it++;
      if (!isa<InsertElementInst>(In) && !isa<ExtractElementInst>(In))
        continue;

      // Check if we can replace this instruction with any of the
      // visited instructions.
      for (Instruction *v : Visited) {
        if (In->isIdenticalTo(v) &&
            DT->dominates(v->getParent(), In->getParent())) {
          In->replaceAllUsesWith(v);
          eraseInstruction(In);
          In = nullptr;
          break;
        }
      }
      if (In) {
        assert(!is_contained(Visited, In));
        Visited.push_back(In);
      }
    }
  }
  CSEBlocks.clear();
  GatherSeq.clear();
}

// Groups the instructions to a bundle (which is then a single scheduling entity)
// and schedules instructions until the bundle gets ready.
bool BoUpSLP::BlockScheduling::tryScheduleBundle(ArrayRef<Value *> VL,
                                                 BoUpSLP *SLP) {
  if (isa<PHINode>(VL[0]))
    return true;

  // Initialize the instruction bundle.
  Instruction *OldScheduleEnd = ScheduleEnd;
  ScheduleData *PrevInBundle = nullptr;
  ScheduleData *Bundle = nullptr;
  bool ReSchedule = false;
  DEBUG(dbgs() << "SLP:  bundle: " << *VL[0] << "\n");

  // Make sure that the scheduling region contains all
  // instructions of the bundle.
  for (Value *V : VL) {
    if (!extendSchedulingRegion(V))
      return false;
  }

  for (Value *V : VL) {
    ScheduleData *BundleMember = getScheduleData(V);
    assert(BundleMember &&
           "no ScheduleData for bundle member (maybe not in same basic block)");
    if (BundleMember->IsScheduled) {
      // A bundle member was scheduled as single instruction before and now
      // needs to be scheduled as part of the bundle. We just get rid of the
      // existing schedule.
      DEBUG(dbgs() << "SLP:  reset schedule because " << *BundleMember
                   << " was already scheduled\n");
      ReSchedule = true;
    }
    assert(BundleMember->isSchedulingEntity() &&
           "bundle member already part of other bundle");
    if (PrevInBundle) {
      PrevInBundle->NextInBundle = BundleMember;
    } else {
      Bundle = BundleMember;
    }
    BundleMember->UnscheduledDepsInBundle = 0;
    Bundle->UnscheduledDepsInBundle += BundleMember->UnscheduledDeps;

    // Group the instructions to a bundle.
    BundleMember->FirstInBundle = Bundle;
    PrevInBundle = BundleMember;
  }
  if (ScheduleEnd != OldScheduleEnd) {
    // The scheduling region got new instructions at the lower end (or it is a
    // new region for the first bundle). This makes it necessary to
    // recalculate all dependencies.
    // It is seldom that this needs to be done a second time after adding the
    // initial bundle to the region.
    for (auto *I = ScheduleStart; I != ScheduleEnd; I = I->getNextNode()) {
      ScheduleData *SD = getScheduleData(I);
      SD->clearDependencies();
    }
    ReSchedule = true;
  }
  if (ReSchedule) {
    resetSchedule();
    initialFillReadyList(ReadyInsts);
  }

  DEBUG(dbgs() << "SLP: try schedule bundle " << *Bundle << " in block "
               << BB->getName() << "\n");

  calculateDependencies(Bundle, true, SLP);

  // Now try to schedule the new bundle. As soon as the bundle is "ready" it
  // means that there are no cyclic dependencies and we can schedule it.
  // Note that's important that we don't "schedule" the bundle yet (see
  // cancelScheduling).
  while (!Bundle->isReady() && !ReadyInsts.empty()) {

    ScheduleData *pickedSD = ReadyInsts.back();
    ReadyInsts.pop_back();

    if (pickedSD->isSchedulingEntity() && pickedSD->isReady()) {
      schedule(pickedSD, ReadyInsts);
    }
  }
  if (!Bundle->isReady()) {
    cancelScheduling(VL);
    return false;
  }
  return true;
}

void BoUpSLP::BlockScheduling::cancelScheduling(ArrayRef<Value *> VL) {
  if (isa<PHINode>(VL[0]))
    return;

  ScheduleData *Bundle = getScheduleData(VL[0]);
  DEBUG(dbgs() << "SLP:  cancel scheduling of " << *Bundle << "\n");
  assert(!Bundle->IsScheduled &&
         "Can't cancel bundle which is already scheduled");
  assert(Bundle->isSchedulingEntity() && Bundle->isPartOfBundle() &&
         "tried to unbundle something which is not a bundle");

  // Un-bundle: make single instructions out of the bundle.
  ScheduleData *BundleMember = Bundle;
  while (BundleMember) {
    assert(BundleMember->FirstInBundle == Bundle && "corrupt bundle links");
    BundleMember->FirstInBundle = BundleMember;
    ScheduleData *Next = BundleMember->NextInBundle;
    BundleMember->NextInBundle = nullptr;
    BundleMember->UnscheduledDepsInBundle = BundleMember->UnscheduledDeps;
    if (BundleMember->UnscheduledDepsInBundle == 0) {
      ReadyInsts.insert(BundleMember);
    }
    BundleMember = Next;
  }
}

bool BoUpSLP::BlockScheduling::extendSchedulingRegion(Value *V) {
  if (getScheduleData(V))
    return true;
  Instruction *I = dyn_cast<Instruction>(V);
  assert(I && "bundle member must be an instruction");
  assert(!isa<PHINode>(I) && "phi nodes don't need to be scheduled");
  if (!ScheduleStart) {
    // It's the first instruction in the new region.
    initScheduleData(I, I->getNextNode(), nullptr, nullptr);
    ScheduleStart = I;
    ScheduleEnd = I->getNextNode();
    assert(ScheduleEnd && "tried to vectorize a TerminatorInst?");
    DEBUG(dbgs() << "SLP:  initialize schedule region to " << *I << "\n");
    return true;
  }
  // Search up and down at the same time, because we don't know if the new
  // instruction is above or below the existing scheduling region.
  BasicBlock::reverse_iterator UpIter =
      ++ScheduleStart->getIterator().getReverse();
  BasicBlock::reverse_iterator UpperEnd = BB->rend();
  BasicBlock::iterator DownIter = ScheduleEnd->getIterator();
  BasicBlock::iterator LowerEnd = BB->end();
  for (;;) {
    if (++ScheduleRegionSize > ScheduleRegionSizeLimit) {
      DEBUG(dbgs() << "SLP:  exceeded schedule region size limit\n");
      return false;
    }

    if (UpIter != UpperEnd) {
      if (&*UpIter == I) {
        initScheduleData(I, ScheduleStart, nullptr, FirstLoadStoreInRegion);
        ScheduleStart = I;
        DEBUG(dbgs() << "SLP:  extend schedule region start to " << *I << "\n");
        return true;
      }
      UpIter++;
    }
    if (DownIter != LowerEnd) {
      if (&*DownIter == I) {
        initScheduleData(ScheduleEnd, I->getNextNode(), LastLoadStoreInRegion,
                         nullptr);
        ScheduleEnd = I->getNextNode();
        assert(ScheduleEnd && "tried to vectorize a TerminatorInst?");
        DEBUG(dbgs() << "SLP:  extend schedule region end to " << *I << "\n");
        return true;
      }
      DownIter++;
    }
    assert((UpIter != UpperEnd || DownIter != LowerEnd) &&
           "instruction not found in block");
  }
  return true;
}

void BoUpSLP::BlockScheduling::initScheduleData(Instruction *FromI,
                                                Instruction *ToI,
                                                ScheduleData *PrevLoadStore,
                                                ScheduleData *NextLoadStore) {
  ScheduleData *CurrentLoadStore = PrevLoadStore;
  for (Instruction *I = FromI; I != ToI; I = I->getNextNode()) {
    ScheduleData *SD = ScheduleDataMap[I];
    if (!SD) {
      // Allocate a new ScheduleData for the instruction.
      if (ChunkPos >= ChunkSize) {
        ScheduleDataChunks.push_back(
            llvm::make_unique<ScheduleData[]>(ChunkSize));
        ChunkPos = 0;
      }
      SD = &(ScheduleDataChunks.back()[ChunkPos++]);
      ScheduleDataMap[I] = SD;
      SD->Inst = I;
    }
    assert(!isInSchedulingRegion(SD) &&
           "new ScheduleData already in scheduling region");
    SD->init(SchedulingRegionID);

    if (I->mayReadOrWriteMemory()) {
      // Update the linked list of memory accessing instructions.
      if (CurrentLoadStore) {
        CurrentLoadStore->NextLoadStore = SD;
      } else {
        FirstLoadStoreInRegion = SD;
      }
      CurrentLoadStore = SD;
    }
  }
  if (NextLoadStore) {
    if (CurrentLoadStore)
      CurrentLoadStore->NextLoadStore = NextLoadStore;
  } else {
    LastLoadStoreInRegion = CurrentLoadStore;
  }
}

void BoUpSLP::BlockScheduling::calculateDependencies(ScheduleData *SD,
                                                     bool InsertInReadyList,
                                                     BoUpSLP *SLP) {
  assert(SD->isSchedulingEntity());

  SmallVector<ScheduleData *, 10> WorkList;
  WorkList.push_back(SD);

  while (!WorkList.empty()) {
    ScheduleData *SD = WorkList.back();
    WorkList.pop_back();

    ScheduleData *BundleMember = SD;
    while (BundleMember) {
      assert(isInSchedulingRegion(BundleMember));
      if (!BundleMember->hasValidDependencies()) {

        DEBUG(dbgs() << "SLP:       update deps of " << *BundleMember << "\n");
        BundleMember->Dependencies = 0;
        BundleMember->resetUnscheduledDeps();

        // Handle def-use chain dependencies.
        for (User *U : BundleMember->Inst->users()) {
          if (isa<Instruction>(U)) {
            ScheduleData *UseSD = getScheduleData(U);
            if (UseSD && isInSchedulingRegion(UseSD->FirstInBundle)) {
              BundleMember->Dependencies++;
              ScheduleData *DestBundle = UseSD->FirstInBundle;
              if (!DestBundle->IsScheduled) {
                BundleMember->incrementUnscheduledDeps(1);
              }
              if (!DestBundle->hasValidDependencies()) {
                WorkList.push_back(DestBundle);
              }
            }
          } else {
            // I'm not sure if this can ever happen. But we need to be safe.
            // This lets the instruction/bundle never be scheduled and
            // eventually disable vectorization.
            BundleMember->Dependencies++;
            BundleMember->incrementUnscheduledDeps(1);
          }
        }

        // Handle the memory dependencies.
        ScheduleData *DepDest = BundleMember->NextLoadStore;
        if (DepDest) {
          Instruction *SrcInst = BundleMember->Inst;
          MemoryLocation SrcLoc = getLocation(SrcInst, SLP->AA);
          bool SrcMayWrite = BundleMember->Inst->mayWriteToMemory();
          unsigned numAliased = 0;
          unsigned DistToSrc = 1;

          while (DepDest) {
            assert(isInSchedulingRegion(DepDest));

            // We have two limits to reduce the complexity:
            // 1) AliasedCheckLimit: It's a small limit to reduce calls to
            //    SLP->isAliased (which is the expensive part in this loop).
            // 2) MaxMemDepDistance: It's for very large blocks and it aborts
            //    the whole loop (even if the loop is fast, it's quadratic).
            //    It's important for the loop break condition (see below) to
            //    check this limit even between two read-only instructions.
            if (DistToSrc >= MaxMemDepDistance ||
                    ((SrcMayWrite || DepDest->Inst->mayWriteToMemory()) &&
                     (numAliased >= AliasedCheckLimit ||
                      SLP->isAliased(SrcLoc, SrcInst, DepDest->Inst)))) {

              // We increment the counter only if the locations are aliased
              // (instead of counting all alias checks). This gives a better
              // balance between reduced runtime and accurate dependencies.
              numAliased++;

              DepDest->MemoryDependencies.push_back(BundleMember);
              BundleMember->Dependencies++;
              ScheduleData *DestBundle = DepDest->FirstInBundle;
              if (!DestBundle->IsScheduled) {
                BundleMember->incrementUnscheduledDeps(1);
              }
              if (!DestBundle->hasValidDependencies()) {
                WorkList.push_back(DestBundle);
              }
            }
            DepDest = DepDest->NextLoadStore;

            // Example, explaining the loop break condition: Let's assume our
            // starting instruction is i0 and MaxMemDepDistance = 3.
            //
            //                      +--------v--v--v
            //             i0,i1,i2,i3,i4,i5,i6,i7,i8
            //             +--------^--^--^
            //
            // MaxMemDepDistance let us stop alias-checking at i3 and we add
            // dependencies from i0 to i3,i4,.. (even if they are not aliased).
            // Previously we already added dependencies from i3 to i6,i7,i8
            // (because of MaxMemDepDistance). As we added a dependency from
            // i0 to i3, we have transitive dependencies from i0 to i6,i7,i8
            // and we can abort this loop at i6.
            if (DistToSrc >= 2 * MaxMemDepDistance)
                break;
            DistToSrc++;
          }
        }
      }
      BundleMember = BundleMember->NextInBundle;
    }
    if (InsertInReadyList && SD->isReady()) {
      ReadyInsts.push_back(SD);
      DEBUG(dbgs() << "SLP:     gets ready on update: " << *SD->Inst << "\n");
    }
  }
}

void BoUpSLP::BlockScheduling::resetSchedule() {
  assert(ScheduleStart &&
         "tried to reset schedule on block which has not been scheduled");
  for (Instruction *I = ScheduleStart; I != ScheduleEnd; I = I->getNextNode()) {
    ScheduleData *SD = getScheduleData(I);
    assert(isInSchedulingRegion(SD));
    SD->IsScheduled = false;
    SD->resetUnscheduledDeps();
  }
  ReadyInsts.clear();
}

void BoUpSLP::scheduleBlock(BlockScheduling *BS) {

  if (!BS->ScheduleStart)
    return;

  DEBUG(dbgs() << "SLP: schedule block " << BS->BB->getName() << "\n");

  BS->resetSchedule();

  // For the real scheduling we use a more sophisticated ready-list: it is
  // sorted by the original instruction location. This lets the final schedule
  // be as  close as possible to the original instruction order.
  struct ScheduleDataCompare {
    bool operator()(ScheduleData *SD1, ScheduleData *SD2) const {
      return SD2->SchedulingPriority < SD1->SchedulingPriority;
    }
  };
  std::set<ScheduleData *, ScheduleDataCompare> ReadyInsts;

  // Ensure that all dependency data is updated and fill the ready-list with
  // initial instructions.
  int Idx = 0;
  int NumToSchedule = 0;
  for (auto *I = BS->ScheduleStart; I != BS->ScheduleEnd;
       I = I->getNextNode()) {
    ScheduleData *SD = BS->getScheduleData(I);
    assert(
        SD->isPartOfBundle() == (ScalarToTreeEntry.count(SD->Inst) != 0) &&
        "scheduler and vectorizer have different opinion on what is a bundle");
    SD->FirstInBundle->SchedulingPriority = Idx++;
    if (SD->isSchedulingEntity()) {
      BS->calculateDependencies(SD, false, this);
      NumToSchedule++;
    }
  }
  BS->initialFillReadyList(ReadyInsts);

  Instruction *LastScheduledInst = BS->ScheduleEnd;

  // Do the "real" scheduling.
  while (!ReadyInsts.empty()) {
    ScheduleData *picked = *ReadyInsts.begin();
    ReadyInsts.erase(ReadyInsts.begin());

    // Move the scheduled instruction(s) to their dedicated places, if not
    // there yet.
    ScheduleData *BundleMember = picked;
    while (BundleMember) {
      Instruction *pickedInst = BundleMember->Inst;
      if (LastScheduledInst->getNextNode() != pickedInst) {
        BS->BB->getInstList().remove(pickedInst);
        BS->BB->getInstList().insert(LastScheduledInst->getIterator(),
                                     pickedInst);
      }
      LastScheduledInst = pickedInst;
      BundleMember = BundleMember->NextInBundle;
    }

    BS->schedule(picked, ReadyInsts);
    NumToSchedule--;
  }
  assert(NumToSchedule == 0 && "could not schedule all instructions");

  // Avoid duplicate scheduling of the block.
  BS->ScheduleStart = nullptr;
}

unsigned BoUpSLP::getVectorElementSize(Value *V) {
  // If V is a store, just return the width of the stored value without
  // traversing the expression tree. This is the common case.
  if (auto *Store = dyn_cast<StoreInst>(V))
    return DL->getTypeSizeInBits(Store->getValueOperand()->getType());

  // If V is not a store, we can traverse the expression tree to find loads
  // that feed it. The type of the loaded value may indicate a more suitable
  // width than V's type. We want to base the vector element size on the width
  // of memory operations where possible.
  SmallVector<Instruction *, 16> Worklist;
  SmallPtrSet<Instruction *, 16> Visited;
  if (auto *I = dyn_cast<Instruction>(V))
    Worklist.push_back(I);

  // Traverse the expression tree in bottom-up order looking for loads. If we
  // encounter an instruciton we don't yet handle, we give up.
  auto MaxWidth = 0u;
  auto FoundUnknownInst = false;
  while (!Worklist.empty() && !FoundUnknownInst) {
    auto *I = Worklist.pop_back_val();
    Visited.insert(I);

    // We should only be looking at scalar instructions here. If the current
    // instruction has a vector type, give up.
    auto *Ty = I->getType();
    if (isa<VectorType>(Ty))
      FoundUnknownInst = true;

    // If the current instruction is a load, update MaxWidth to reflect the
    // width of the loaded value.
    else if (isa<LoadInst>(I))
      MaxWidth = std::max<unsigned>(MaxWidth, DL->getTypeSizeInBits(Ty));

    // Otherwise, we need to visit the operands of the instruction. We only
    // handle the interesting cases from buildTree here. If an operand is an
    // instruction we haven't yet visited, we add it to the worklist.
    else if (isa<PHINode>(I) || isa<CastInst>(I) || isa<GetElementPtrInst>(I) ||
             isa<CmpInst>(I) || isa<SelectInst>(I) || isa<BinaryOperator>(I)) {
      for (Use &U : I->operands())
        if (auto *J = dyn_cast<Instruction>(U.get()))
          if (!Visited.count(J))
            Worklist.push_back(J);
    }

    // If we don't yet handle the instruction, give up.
    else
      FoundUnknownInst = true;
  }

  // If we didn't encounter a memory access in the expression tree, or if we
  // gave up for some reason, just return the width of V.
  if (!MaxWidth || FoundUnknownInst)
    return DL->getTypeSizeInBits(V->getType());

  // Otherwise, return the maximum width we found.
  return MaxWidth;
}

// Determine if a value V in a vectorizable expression Expr can be demoted to a
// smaller type with a truncation. We collect the values that will be demoted
// in ToDemote and additional roots that require investigating in Roots.
static bool collectValuesToDemote(Value *V, SmallPtrSetImpl<Value *> &Expr,
                                  SmallVectorImpl<Value *> &ToDemote,
                                  SmallVectorImpl<Value *> &Roots) {

  // We can always demote constants.
  if (isa<Constant>(V)) {
    ToDemote.push_back(V);
    return true;
  }

  // If the value is not an instruction in the expression with only one use, it
  // cannot be demoted.
  auto *I = dyn_cast<Instruction>(V);
  if (!I || !I->hasOneUse() || !Expr.count(I))
    return false;

  switch (I->getOpcode()) {

  // We can always demote truncations and extensions. Since truncations can
  // seed additional demotion, we save the truncated value.
  case Instruction::Trunc:
    Roots.push_back(I->getOperand(0));
  case Instruction::ZExt:
  case Instruction::SExt:
    break;

  // We can demote certain binary operations if we can demote both of their
  // operands.
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    if (!collectValuesToDemote(I->getOperand(0), Expr, ToDemote, Roots) ||
        !collectValuesToDemote(I->getOperand(1), Expr, ToDemote, Roots))
      return false;
    break;

  // We can demote selects if we can demote their true and false values.
  case Instruction::Select: {
    SelectInst *SI = cast<SelectInst>(I);
    if (!collectValuesToDemote(SI->getTrueValue(), Expr, ToDemote, Roots) ||
        !collectValuesToDemote(SI->getFalseValue(), Expr, ToDemote, Roots))
      return false;
    break;
  }

  // We can demote phis if we can demote all their incoming operands. Note that
  // we don't need to worry about cycles since we ensure single use above.
  case Instruction::PHI: {
    PHINode *PN = cast<PHINode>(I);
    for (Value *IncValue : PN->incoming_values())
      if (!collectValuesToDemote(IncValue, Expr, ToDemote, Roots))
        return false;
    break;
  }

  // Otherwise, conservatively give up.
  default:
    return false;
  }

  // Record the value that we can demote.
  ToDemote.push_back(V);
  return true;
}

void BoUpSLP::computeMinimumValueSizes() {
  // If there are no external uses, the expression tree must be rooted by a
  // store. We can't demote in-memory values, so there is nothing to do here.
  if (ExternalUses.empty())
    return;

  // We only attempt to truncate integer expressions.
  auto &TreeRoot = VectorizableTree[0].Scalars;
  auto *TreeRootIT = dyn_cast<IntegerType>(TreeRoot[0]->getType());
  if (!TreeRootIT)
    return;

  // If the expression is not rooted by a store, these roots should have
  // external uses. We will rely on InstCombine to rewrite the expression in
  // the narrower type. However, InstCombine only rewrites single-use values.
  // This means that if a tree entry other than a root is used externally, it
  // must have multiple uses and InstCombine will not rewrite it. The code
  // below ensures that only the roots are used externally.
  SmallPtrSet<Value *, 32> Expr(TreeRoot.begin(), TreeRoot.end());
  for (auto &EU : ExternalUses)
    if (!Expr.erase(EU.Scalar))
      return;
  if (!Expr.empty())
    return;

  // Collect the scalar values of the vectorizable expression. We will use this
  // context to determine which values can be demoted. If we see a truncation,
  // we mark it as seeding another demotion.
  for (auto &Entry : VectorizableTree)
    Expr.insert(Entry.Scalars.begin(), Entry.Scalars.end());

  // Ensure the roots of the vectorizable tree don't form a cycle. They must
  // have a single external user that is not in the vectorizable tree.
  for (auto *Root : TreeRoot)
    if (!Root->hasOneUse() || Expr.count(*Root->user_begin()))
      return;

  // Conservatively determine if we can actually truncate the roots of the
  // expression. Collect the values that can be demoted in ToDemote and
  // additional roots that require investigating in Roots.
  SmallVector<Value *, 32> ToDemote;
  SmallVector<Value *, 4> Roots;
  for (auto *Root : TreeRoot)
    if (!collectValuesToDemote(Root, Expr, ToDemote, Roots))
      return;

  // The maximum bit width required to represent all the values that can be
  // demoted without loss of precision. It would be safe to truncate the roots
  // of the expression to this width.
  auto MaxBitWidth = 8u;

  // We first check if all the bits of the roots are demanded. If they're not,
  // we can truncate the roots to this narrower type.
  for (auto *Root : TreeRoot) {
    auto Mask = DB->getDemandedBits(cast<Instruction>(Root));
    MaxBitWidth = std::max<unsigned>(
        Mask.getBitWidth() - Mask.countLeadingZeros(), MaxBitWidth);
  }

  // True if the roots can be zero-extended back to their original type, rather
  // than sign-extended. We know that if the leading bits are not demanded, we
  // can safely zero-extend. So we initialize IsKnownPositive to True.
  bool IsKnownPositive = true;

  // If all the bits of the roots are demanded, we can try a little harder to
  // compute a narrower type. This can happen, for example, if the roots are
  // getelementptr indices. InstCombine promotes these indices to the pointer
  // width. Thus, all their bits are technically demanded even though the
  // address computation might be vectorized in a smaller type.
  //
  // We start by looking at each entry that can be demoted. We compute the
  // maximum bit width required to store the scalar by using ValueTracking to
  // compute the number of high-order bits we can truncate.
  if (MaxBitWidth == DL->getTypeSizeInBits(TreeRoot[0]->getType())) {
    MaxBitWidth = 8u;

    // Determine if the sign bit of all the roots is known to be zero. If not,
    // IsKnownPositive is set to False.
    IsKnownPositive = all_of(TreeRoot, [&](Value *R) {
      bool KnownZero = false;
      bool KnownOne = false;
      ComputeSignBit(R, KnownZero, KnownOne, *DL);
      return KnownZero;
    });

    // Determine the maximum number of bits required to store the scalar
    // values.
    for (auto *Scalar : ToDemote) {
      auto NumSignBits = ComputeNumSignBits(Scalar, *DL, 0, AC, 0, DT);
      auto NumTypeBits = DL->getTypeSizeInBits(Scalar->getType());
      MaxBitWidth = std::max<unsigned>(NumTypeBits - NumSignBits, MaxBitWidth);
    }

    // If we can't prove that the sign bit is zero, we must add one to the
    // maximum bit width to account for the unknown sign bit. This preserves
    // the existing sign bit so we can safely sign-extend the root back to the
    // original type. Otherwise, if we know the sign bit is zero, we will
    // zero-extend the root instead.
    //
    // FIXME: This is somewhat suboptimal, as there will be cases where adding
    //        one to the maximum bit width will yield a larger-than-necessary
    //        type. In general, we need to add an extra bit only if we can't
    //        prove that the upper bit of the original type is equal to the
    //        upper bit of the proposed smaller type. If these two bits are the
    //        same (either zero or one) we know that sign-extending from the
    //        smaller type will result in the same value. Here, since we can't
    //        yet prove this, we are just making the proposed smaller type
    //        larger to ensure correctness.
    if (!IsKnownPositive)
      ++MaxBitWidth;
  }

  // Round MaxBitWidth up to the next power-of-two.
  if (!isPowerOf2_64(MaxBitWidth))
    MaxBitWidth = NextPowerOf2(MaxBitWidth);

  // If the maximum bit width we compute is less than the with of the roots'
  // type, we can proceed with the narrowing. Otherwise, do nothing.
  if (MaxBitWidth >= TreeRootIT->getBitWidth())
    return;

  // If we can truncate the root, we must collect additional values that might
  // be demoted as a result. That is, those seeded by truncations we will
  // modify.
  while (!Roots.empty())
    collectValuesToDemote(Roots.pop_back_val(), Expr, ToDemote, Roots);

  // Finally, map the values we can demote to the maximum bit with we computed.
  for (auto *Scalar : ToDemote)
    MinBWs[Scalar] = std::make_pair(MaxBitWidth, !IsKnownPositive);
}

namespace {
/// The SLPVectorizer Pass.
struct SLPVectorizer : public FunctionPass {
  SLPVectorizerPass Impl;

  /// Pass identification, replacement for typeid
  static char ID;

  explicit SLPVectorizer() : FunctionPass(ID) {
    initializeSLPVectorizerPass(*PassRegistry::getPassRegistry());
  }


  bool doInitialization(Module &M) override {
    return false;
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    auto *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto *TLIP = getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
    auto *TLI = TLIP ? &TLIP->getTLI() : nullptr;
    auto *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto *AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    auto *DB = &getAnalysis<DemandedBitsWrapperPass>().getDemandedBits();

    return Impl.runImpl(F, SE, TTI, TLI, AA, LI, DT, AC, DB);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    FunctionPass::getAnalysisUsage(AU);
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<DemandedBitsWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<AAResultsWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.setPreservesCFG();
  }
};
} // end anonymous namespace

PreservedAnalyses SLPVectorizerPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto *SE = &AM.getResult<ScalarEvolutionAnalysis>(F);
  auto *TTI = &AM.getResult<TargetIRAnalysis>(F);
  auto *TLI = AM.getCachedResult<TargetLibraryAnalysis>(F);
  auto *AA = &AM.getResult<AAManager>(F);
  auto *LI = &AM.getResult<LoopAnalysis>(F);
  auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
  auto *AC = &AM.getResult<AssumptionAnalysis>(F);
  auto *DB = &AM.getResult<DemandedBitsAnalysis>(F);

  bool Changed = runImpl(F, SE, TTI, TLI, AA, LI, DT, AC, DB);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<AAManager>();
  PA.preserve<GlobalsAA>();
  return PA;
}

bool SLPVectorizerPass::runImpl(Function &F, ScalarEvolution *SE_,
                                TargetTransformInfo *TTI_,
                                TargetLibraryInfo *TLI_, AliasAnalysis *AA_,
                                LoopInfo *LI_, DominatorTree *DT_,
                                AssumptionCache *AC_, DemandedBits *DB_) {
  SE = SE_;
  TTI = TTI_;
  TLI = TLI_;
  AA = AA_;
  LI = LI_;
  DT = DT_;
  AC = AC_;
  DB = DB_;
  DL = &F.getParent()->getDataLayout();

  Stores.clear();
  GEPs.clear();
  bool Changed = false;

  // If the target claims to have no vector registers don't attempt
  // vectorization.
  if (!TTI->getNumberOfRegisters(true))
    return false;

  // Don't vectorize when the attribute NoImplicitFloat is used.
  if (F.hasFnAttribute(Attribute::NoImplicitFloat))
    return false;

  DEBUG(dbgs() << "SLP: Analyzing blocks in " << F.getName() << ".\n");

  // Use the bottom up slp vectorizer to construct chains that start with
  // store instructions.
  BoUpSLP R(&F, SE, TTI, TLI, AA, LI, DT, AC, DB, DL);

  // A general note: the vectorizer must use BoUpSLP::eraseInstruction() to
  // delete instructions.

  // Scan the blocks in the function in post order.
  for (auto BB : post_order(&F.getEntryBlock())) {
    collectSeedInstructions(BB);

    // Vectorize trees that end at stores.
    if (!Stores.empty()) {
      DEBUG(dbgs() << "SLP: Found stores for " << Stores.size()
                   << " underlying objects.\n");
      Changed |= vectorizeStoreChains(R);
    }

    // Vectorize trees that end at reductions.
    Changed |= vectorizeChainsInBlock(BB, R);

    // Vectorize the index computations of getelementptr instructions. This
    // is primarily intended to catch gather-like idioms ending at
    // non-consecutive loads.
    if (!GEPs.empty()) {
      DEBUG(dbgs() << "SLP: Found GEPs for " << GEPs.size()
                   << " underlying objects.\n");
      Changed |= vectorizeGEPIndices(BB, R);
    }
  }

  if (Changed) {
    R.optimizeGatherSequence();
    DEBUG(dbgs() << "SLP: vectorized \"" << F.getName() << "\"\n");
    DEBUG(verifyFunction(F));
  }
  return Changed;
}

/// \brief Check that the Values in the slice in VL array are still existent in
/// the WeakVH array.
/// Vectorization of part of the VL array may cause later values in the VL array
/// to become invalid. We track when this has happened in the WeakVH array.
static bool hasValueBeenRAUWed(ArrayRef<Value *> VL, ArrayRef<WeakVH> VH,
                               unsigned SliceBegin, unsigned SliceSize) {
  VL = VL.slice(SliceBegin, SliceSize);
  VH = VH.slice(SliceBegin, SliceSize);
  return !std::equal(VL.begin(), VL.end(), VH.begin());
}

bool SLPVectorizerPass::vectorizeStoreChain(ArrayRef<Value *> Chain, BoUpSLP &R,
                                            unsigned VecRegSize) {
  unsigned ChainLen = Chain.size();
  DEBUG(dbgs() << "SLP: Analyzing a store chain of length " << ChainLen
        << "\n");
  unsigned Sz = R.getVectorElementSize(Chain[0]);
  unsigned VF = VecRegSize / Sz;

  if (!isPowerOf2_32(Sz) || VF < 2)
    return false;

  // Keep track of values that were deleted by vectorizing in the loop below.
  SmallVector<WeakVH, 8> TrackValues(Chain.begin(), Chain.end());

  bool Changed = false;
  // Look for profitable vectorizable trees at all offsets, starting at zero.
  for (unsigned i = 0, e = ChainLen; i < e; ++i) {
    if (i + VF > e)
      break;

    // Check that a previous iteration of this loop did not delete the Value.
    if (hasValueBeenRAUWed(Chain, TrackValues, i, VF))
      continue;

    DEBUG(dbgs() << "SLP: Analyzing " << VF << " stores at offset " << i
          << "\n");
    ArrayRef<Value *> Operands = Chain.slice(i, VF);

    R.buildTree(Operands);
    if (R.isTreeTinyAndNotFullyVectorizable())
      continue;

    R.computeMinimumValueSizes();

    int Cost = R.getTreeCost();

    DEBUG(dbgs() << "SLP: Found cost=" << Cost << " for VF=" << VF << "\n");
    if (Cost < -SLPCostThreshold) {
      DEBUG(dbgs() << "SLP: Decided to vectorize cost=" << Cost << "\n");
      R.vectorizeTree();

      // Move to the next bundle.
      i += VF - 1;
      Changed = true;
    }
  }

  return Changed;
}

bool SLPVectorizerPass::vectorizeStores(ArrayRef<StoreInst *> Stores,
                                        BoUpSLP &R) {
  SetVector<StoreInst *> Heads, Tails;
  SmallDenseMap<StoreInst *, StoreInst *> ConsecutiveChain;

  // We may run into multiple chains that merge into a single chain. We mark the
  // stores that we vectorized so that we don't visit the same store twice.
  BoUpSLP::ValueSet VectorizedStores;
  bool Changed = false;

  // Do a quadratic search on all of the given stores and find
  // all of the pairs of stores that follow each other.
  SmallVector<unsigned, 16> IndexQueue;
  for (unsigned i = 0, e = Stores.size(); i < e; ++i) {
    IndexQueue.clear();
    // If a store has multiple consecutive store candidates, search Stores
    // array according to the sequence: from i+1 to e, then from i-1 to 0.
    // This is because usually pairing with immediate succeeding or preceding
    // candidate create the best chance to find slp vectorization opportunity.
    unsigned j = 0;
    for (j = i + 1; j < e; ++j)
      IndexQueue.push_back(j);
    for (j = i; j > 0; --j)
      IndexQueue.push_back(j - 1);

    for (auto &k : IndexQueue) {
      if (isConsecutiveAccess(Stores[i], Stores[k], *DL, *SE)) {
        Tails.insert(Stores[k]);
        Heads.insert(Stores[i]);
        ConsecutiveChain[Stores[i]] = Stores[k];
        break;
      }
    }
  }

  // For stores that start but don't end a link in the chain:
  for (SetVector<StoreInst *>::iterator it = Heads.begin(), e = Heads.end();
       it != e; ++it) {
    if (Tails.count(*it))
      continue;

    // We found a store instr that starts a chain. Now follow the chain and try
    // to vectorize it.
    BoUpSLP::ValueList Operands;
    StoreInst *I = *it;
    // Collect the chain into a list.
    while (Tails.count(I) || Heads.count(I)) {
      if (VectorizedStores.count(I))
        break;
      Operands.push_back(I);
      // Move to the next value in the chain.
      I = ConsecutiveChain[I];
    }

    // FIXME: Is division-by-2 the correct step? Should we assert that the
    // register size is a power-of-2?
    for (unsigned Size = R.getMaxVecRegSize(); Size >= R.getMinVecRegSize();
         Size /= 2) {
      if (vectorizeStoreChain(Operands, R, Size)) {
        // Mark the vectorized stores so that we don't vectorize them again.
        VectorizedStores.insert(Operands.begin(), Operands.end());
        Changed = true;
        break;
      }
    }
  }

  return Changed;
}

void SLPVectorizerPass::collectSeedInstructions(BasicBlock *BB) {

  // Initialize the collections. We will make a single pass over the block.
  Stores.clear();
  GEPs.clear();

  // Visit the store and getelementptr instructions in BB and organize them in
  // Stores and GEPs according to the underlying objects of their pointer
  // operands.
  for (Instruction &I : *BB) {

    // Ignore store instructions that are volatile or have a pointer operand
    // that doesn't point to a scalar type.
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (!SI->isSimple())
        continue;
      if (!isValidElementType(SI->getValueOperand()->getType()))
        continue;
      Stores[GetUnderlyingObject(SI->getPointerOperand(), *DL)].push_back(SI);
    }

    // Ignore getelementptr instructions that have more than one index, a
    // constant index, or a pointer operand that doesn't point to a scalar
    // type.
    else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      auto Idx = GEP->idx_begin()->get();
      if (GEP->getNumIndices() > 1 || isa<Constant>(Idx))
        continue;
      if (!isValidElementType(Idx->getType()))
        continue;
      if (GEP->getType()->isVectorTy())
        continue;
      GEPs[GetUnderlyingObject(GEP->getPointerOperand(), *DL)].push_back(GEP);
    }
  }
}

bool SLPVectorizerPass::tryToVectorizePair(Value *A, Value *B, BoUpSLP &R) {
  if (!A || !B)
    return false;
  Value *VL[] = { A, B };
  return tryToVectorizeList(VL, R, None, true);
}

bool SLPVectorizerPass::tryToVectorizeList(ArrayRef<Value *> VL, BoUpSLP &R,
                                           ArrayRef<Value *> BuildVector,
                                           bool AllowReorder) {
  if (VL.size() < 2)
    return false;

  DEBUG(dbgs() << "SLP: Trying to vectorize a list of length = " << VL.size()
               << ".\n");

  // Check that all of the parts are scalar instructions of the same type.
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  if (!I0)
    return false;

  unsigned Opcode0 = I0->getOpcode();

  unsigned Sz = R.getVectorElementSize(I0);
  unsigned MinVF = std::max(2U, R.getMinVecRegSize() / Sz);
  unsigned MaxVF = std::max<unsigned>(PowerOf2Floor(VL.size()), MinVF);
  if (MaxVF < 2)
    return false;

  for (Value *V : VL) {
    Type *Ty = V->getType();
    if (!isValidElementType(Ty))
      return false;
    Instruction *Inst = dyn_cast<Instruction>(V);
    if (!Inst || Inst->getOpcode() != Opcode0)
      return false;
  }

  bool Changed = false;

  // Keep track of values that were deleted by vectorizing in the loop below.
  SmallVector<WeakVH, 8> TrackValues(VL.begin(), VL.end());

  unsigned NextInst = 0, MaxInst = VL.size();
  for (unsigned VF = MaxVF; NextInst + 1 < MaxInst && VF >= MinVF;
       VF /= 2) {
    // No actual vectorization should happen, if number of parts is the same as
    // provided vectorization factor (i.e. the scalar type is used for vector
    // code during codegen).
    auto *VecTy = VectorType::get(VL[0]->getType(), VF);
    if (TTI->getNumberOfParts(VecTy) == VF)
      continue;
    for (unsigned I = NextInst; I < MaxInst; ++I) {
      unsigned OpsWidth = 0;

      if (I + VF > MaxInst)
        OpsWidth = MaxInst - I;
      else
        OpsWidth = VF;

      if (!isPowerOf2_32(OpsWidth) || OpsWidth < 2)
        break;

      // Check that a previous iteration of this loop did not delete the Value.
      if (hasValueBeenRAUWed(VL, TrackValues, I, OpsWidth))
        continue;

      DEBUG(dbgs() << "SLP: Analyzing " << OpsWidth << " operations "
                   << "\n");
      ArrayRef<Value *> Ops = VL.slice(I, OpsWidth);

      ArrayRef<Value *> BuildVectorSlice;
      if (!BuildVector.empty())
        BuildVectorSlice = BuildVector.slice(I, OpsWidth);

      R.buildTree(Ops, BuildVectorSlice);
      // TODO: check if we can allow reordering for more cases.
      if (AllowReorder && R.shouldReorder()) {
        // Conceptually, there is nothing actually preventing us from trying to
        // reorder a larger list. In fact, we do exactly this when vectorizing
        // reductions. However, at this point, we only expect to get here from
        // tryToVectorizePair().
        assert(Ops.size() == 2);
        assert(BuildVectorSlice.empty());
        Value *ReorderedOps[] = {Ops[1], Ops[0]};
        R.buildTree(ReorderedOps, None);
      }
      if (R.isTreeTinyAndNotFullyVectorizable())
        continue;

      R.computeMinimumValueSizes();
      int Cost = R.getTreeCost();

      if (Cost < -SLPCostThreshold) {
        DEBUG(dbgs() << "SLP: Vectorizing list at cost:" << Cost << ".\n");
        Value *VectorizedRoot = R.vectorizeTree();

        // Reconstruct the build vector by extracting the vectorized root. This
        // way we handle the case where some elements of the vector are
        // undefined.
        //  (return (inserelt <4 xi32> (insertelt undef (opd0) 0) (opd1) 2))
        if (!BuildVectorSlice.empty()) {
          // The insert point is the last build vector instruction. The
          // vectorized root will precede it. This guarantees that we get an
          // instruction. The vectorized tree could have been constant folded.
          Instruction *InsertAfter = cast<Instruction>(BuildVectorSlice.back());
          unsigned VecIdx = 0;
          for (auto &V : BuildVectorSlice) {
            IRBuilder<NoFolder> Builder(InsertAfter->getParent(),
                                        ++BasicBlock::iterator(InsertAfter));
            Instruction *I = cast<Instruction>(V);
            assert(isa<InsertElementInst>(I) || isa<InsertValueInst>(I));
            Instruction *Extract =
                cast<Instruction>(Builder.CreateExtractElement(
                    VectorizedRoot, Builder.getInt32(VecIdx++)));
            I->setOperand(1, Extract);
            I->removeFromParent();
            I->insertAfter(Extract);
            InsertAfter = I;
          }
        }
        // Move to the next bundle.
        I += VF - 1;
        NextInst = I + 1;
        Changed = true;
      }
    }
  }

  return Changed;
}

bool SLPVectorizerPass::tryToVectorize(BinaryOperator *V, BoUpSLP &R) {
  if (!V)
    return false;

  Value *P = V->getParent();

  // Vectorize in current basic block only.
  auto *Op0 = dyn_cast<Instruction>(V->getOperand(0));
  auto *Op1 = dyn_cast<Instruction>(V->getOperand(1));
  if (!Op0 || !Op1 || Op0->getParent() != P || Op1->getParent() != P)
    return false;

  // Try to vectorize V.
  if (tryToVectorizePair(Op0, Op1, R))
    return true;

  auto *A = dyn_cast<BinaryOperator>(Op0);
  auto *B = dyn_cast<BinaryOperator>(Op1);
  // Try to skip B.
  if (B && B->hasOneUse()) {
    auto *B0 = dyn_cast<BinaryOperator>(B->getOperand(0));
    auto *B1 = dyn_cast<BinaryOperator>(B->getOperand(1));
    if (B0 && B0->getParent() == P && tryToVectorizePair(A, B0, R))
      return true;
    if (B1 && B1->getParent() == P && tryToVectorizePair(A, B1, R))
      return true;
  }

  // Try to skip A.
  if (A && A->hasOneUse()) {
    auto *A0 = dyn_cast<BinaryOperator>(A->getOperand(0));
    auto *A1 = dyn_cast<BinaryOperator>(A->getOperand(1));
    if (A0 && A0->getParent() == P && tryToVectorizePair(A0, B, R))
      return true;
    if (A1 && A1->getParent() == P && tryToVectorizePair(A1, B, R))
      return true;
  }
  return false;
}

/// \brief Generate a shuffle mask to be used in a reduction tree.
///
/// \param VecLen The length of the vector to be reduced.
/// \param NumEltsToRdx The number of elements that should be reduced in the
///        vector.
/// \param IsPairwise Whether the reduction is a pairwise or splitting
///        reduction. A pairwise reduction will generate a mask of
///        <0,2,...> or <1,3,..> while a splitting reduction will generate
///        <2,3, undef,undef> for a vector of 4 and NumElts = 2.
/// \param IsLeft True will generate a mask of even elements, odd otherwise.
static Value *createRdxShuffleMask(unsigned VecLen, unsigned NumEltsToRdx,
                                   bool IsPairwise, bool IsLeft,
                                   IRBuilder<> &Builder) {
  assert((IsPairwise || !IsLeft) && "Don't support a <0,1,undef,...> mask");

  SmallVector<Constant *, 32> ShuffleMask(
      VecLen, UndefValue::get(Builder.getInt32Ty()));

  if (IsPairwise)
    // Build a mask of 0, 2, ... (left) or 1, 3, ... (right).
    for (unsigned i = 0; i != NumEltsToRdx; ++i)
      ShuffleMask[i] = Builder.getInt32(2 * i + !IsLeft);
  else
    // Move the upper half of the vector to the lower half.
    for (unsigned i = 0; i != NumEltsToRdx; ++i)
      ShuffleMask[i] = Builder.getInt32(NumEltsToRdx + i);

  return ConstantVector::get(ShuffleMask);
}

namespace {
/// Model horizontal reductions.
///
/// A horizontal reduction is a tree of reduction operations (currently add and
/// fadd) that has operations that can be put into a vector as its leaf.
/// For example, this tree:
///
/// mul mul mul mul
///  \  /    \  /
///   +       +
///    \     /
///       +
/// This tree has "mul" as its reduced values and "+" as its reduction
/// operations. A reduction might be feeding into a store or a binary operation
/// feeding a phi.
///    ...
///    \  /
///     +
///     |
///  phi +=
///
///  Or:
///    ...
///    \  /
///     +
///     |
///   *p =
///
class HorizontalReduction {
  SmallVector<Value *, 16> ReductionOps;
  SmallVector<Value *, 32> ReducedVals;
  // Use map vector to make stable output.
  MapVector<Instruction *, Value *> ExtraArgs;

  BinaryOperator *ReductionRoot = nullptr;

  /// The opcode of the reduction.
  Instruction::BinaryOps ReductionOpcode = Instruction::BinaryOpsEnd;
  /// The opcode of the values we perform a reduction on.
  unsigned ReducedValueOpcode = 0;
  /// Should we model this reduction as a pairwise reduction tree or a tree that
  /// splits the vector in halves and adds those halves.
  bool IsPairwiseReduction = false;

  /// Checks if the ParentStackElem.first should be marked as a reduction
  /// operation with an extra argument or as extra argument itself.
  void markExtraArg(std::pair<Instruction *, unsigned> &ParentStackElem,
                    Value *ExtraArg) {
    if (ExtraArgs.count(ParentStackElem.first)) {
      ExtraArgs[ParentStackElem.first] = nullptr;
      // We ran into something like:
      // ParentStackElem.first = ExtraArgs[ParentStackElem.first] + ExtraArg.
      // The whole ParentStackElem.first should be considered as an extra value
      // in this case.
      // Do not perform analysis of remaining operands of ParentStackElem.first
      // instruction, this whole instruction is an extra argument.
      ParentStackElem.second = ParentStackElem.first->getNumOperands();
    } else {
      // We ran into something like:
      // ParentStackElem.first += ... + ExtraArg + ...
      ExtraArgs[ParentStackElem.first] = ExtraArg;
    }
  }

public:
  HorizontalReduction() = default;

  /// \brief Try to find a reduction tree.
  bool matchAssociativeReduction(PHINode *Phi, BinaryOperator *B) {
    assert((!Phi || is_contained(Phi->operands(), B)) &&
           "Thi phi needs to use the binary operator");

    // We could have a initial reductions that is not an add.
    //  r *= v1 + v2 + v3 + v4
    // In such a case start looking for a tree rooted in the first '+'.
    if (Phi) {
      if (B->getOperand(0) == Phi) {
        Phi = nullptr;
        B = dyn_cast<BinaryOperator>(B->getOperand(1));
      } else if (B->getOperand(1) == Phi) {
        Phi = nullptr;
        B = dyn_cast<BinaryOperator>(B->getOperand(0));
      }
    }

    if (!B)
      return false;

    Type *Ty = B->getType();
    if (!isValidElementType(Ty))
      return false;

    ReductionOpcode = B->getOpcode();
    ReducedValueOpcode = 0;
    ReductionRoot = B;

    // We currently only support adds.
    if ((ReductionOpcode != Instruction::Add &&
         ReductionOpcode != Instruction::FAdd) ||
        !B->isAssociative())
      return false;

    // Post order traverse the reduction tree starting at B. We only handle true
    // trees containing only binary operators or selects.
    SmallVector<std::pair<Instruction *, unsigned>, 32> Stack;
    Stack.push_back(std::make_pair(B, 0));
    while (!Stack.empty()) {
      Instruction *TreeN = Stack.back().first;
      unsigned EdgeToVist = Stack.back().second++;
      bool IsReducedValue = TreeN->getOpcode() != ReductionOpcode;

      // Postorder vist.
      if (EdgeToVist == 2 || IsReducedValue) {
        if (IsReducedValue)
          ReducedVals.push_back(TreeN);
        else {
          auto I = ExtraArgs.find(TreeN);
          if (I != ExtraArgs.end() && !I->second) {
            // Check if TreeN is an extra argument of its parent operation.
            if (Stack.size() <= 1) {
              // TreeN can't be an extra argument as it is a root reduction
              // operation.
              return false;
            }
            // Yes, TreeN is an extra argument, do not add it to a list of
            // reduction operations.
            // Stack[Stack.size() - 2] always points to the parent operation.
            markExtraArg(Stack[Stack.size() - 2], TreeN);
            ExtraArgs.erase(TreeN);
          } else
            ReductionOps.push_back(TreeN);
        }
        // Retract.
        Stack.pop_back();
        continue;
      }

      // Visit left or right.
      Value *NextV = TreeN->getOperand(EdgeToVist);
      if (NextV != Phi) {
        auto *I = dyn_cast<Instruction>(NextV);
        // Continue analysis if the next operand is a reduction operation or
        // (possibly) a reduced value. If the reduced value opcode is not set,
        // the first met operation != reduction operation is considered as the
        // reduced value class.
        if (I && (!ReducedValueOpcode || I->getOpcode() == ReducedValueOpcode ||
                  I->getOpcode() == ReductionOpcode)) {
          // Only handle trees in the current basic block.
          if (I->getParent() != B->getParent()) {
            // I is an extra argument for TreeN (its parent operation).
            markExtraArg(Stack.back(), I);
            continue;
          }

          // Each tree node needs to have one user except for the ultimate
          // reduction.
          if (!I->hasOneUse() && I != B) {
            // I is an extra argument for TreeN (its parent operation).
            markExtraArg(Stack.back(), I);
            continue;
          }

          if (I->getOpcode() == ReductionOpcode) {
            // We need to be able to reassociate the reduction operations.
            if (!I->isAssociative()) {
              // I is an extra argument for TreeN (its parent operation).
              markExtraArg(Stack.back(), I);
              continue;
            }
          } else if (ReducedValueOpcode &&
                     ReducedValueOpcode != I->getOpcode()) {
            // Make sure that the opcodes of the operations that we are going to
            // reduce match.
            // I is an extra argument for TreeN (its parent operation).
            markExtraArg(Stack.back(), I);
            continue;
          } else if (!ReducedValueOpcode)
            ReducedValueOpcode = I->getOpcode();

          Stack.push_back(std::make_pair(I, 0));
          continue;
        }
      }
      // NextV is an extra argument for TreeN (its parent operation).
      markExtraArg(Stack.back(), NextV);
    }
    return true;
  }

  /// \brief Attempt to vectorize the tree found by
  /// matchAssociativeReduction.
  bool tryToReduce(BoUpSLP &V, TargetTransformInfo *TTI) {
    if (ReducedVals.empty())
      return false;

    // If there is a sufficient number of reduction values, reduce
    // to a nearby power-of-2. Can safely generate oversized
    // vectors and rely on the backend to split them to legal sizes.
    unsigned NumReducedVals = ReducedVals.size();
    if (NumReducedVals < 4)
      return false;

    unsigned ReduxWidth = PowerOf2Floor(NumReducedVals);

    Value *VectorizedTree = nullptr;
    IRBuilder<> Builder(ReductionRoot);
    FastMathFlags Unsafe;
    Unsafe.setUnsafeAlgebra();
    Builder.setFastMathFlags(Unsafe);
    unsigned i = 0;

    BoUpSLP::ExtraValueToDebugLocsMap ExternallyUsedValues;
    // The same extra argument may be used several time, so log each attempt
    // to use it.
    for (auto &Pair : ExtraArgs)
      ExternallyUsedValues[Pair.second].push_back(Pair.first);
    while (i < NumReducedVals - ReduxWidth + 1 && ReduxWidth > 2) {
      auto VL = makeArrayRef(&ReducedVals[i], ReduxWidth);
      V.buildTree(VL, ExternallyUsedValues, ReductionOps);
      if (V.shouldReorder()) {
        SmallVector<Value *, 8> Reversed(VL.rbegin(), VL.rend());
        V.buildTree(Reversed, ExternallyUsedValues, ReductionOps);
      }
      if (V.isTreeTinyAndNotFullyVectorizable())
        break;

      V.computeMinimumValueSizes();

      // Estimate cost.
      int Cost =
          V.getTreeCost() + getReductionCost(TTI, ReducedVals[i], ReduxWidth);
      if (Cost >= -SLPCostThreshold)
        break;

      DEBUG(dbgs() << "SLP: Vectorizing horizontal reduction at cost:" << Cost
                   << ". (HorRdx)\n");

      // Vectorize a tree.
      DebugLoc Loc = cast<Instruction>(ReducedVals[i])->getDebugLoc();
      Value *VectorizedRoot = V.vectorizeTree(ExternallyUsedValues);

      // Emit a reduction.
      Value *ReducedSubTree =
          emitReduction(VectorizedRoot, Builder, ReduxWidth, ReductionOps);
      if (VectorizedTree) {
        Builder.SetCurrentDebugLocation(Loc);
        VectorizedTree = Builder.CreateBinOp(ReductionOpcode, VectorizedTree,
                                             ReducedSubTree, "bin.rdx");
        propagateIRFlags(VectorizedTree, ReductionOps);
      } else
        VectorizedTree = ReducedSubTree;
      i += ReduxWidth;
      ReduxWidth = PowerOf2Floor(NumReducedVals - i);
    }

    if (VectorizedTree) {
      // Finish the reduction.
      for (; i < NumReducedVals; ++i) {
        auto *I = cast<Instruction>(ReducedVals[i]);
        Builder.SetCurrentDebugLocation(I->getDebugLoc());
        VectorizedTree =
            Builder.CreateBinOp(ReductionOpcode, VectorizedTree, I);
        propagateIRFlags(VectorizedTree, ReductionOps);
      }
      for (auto &Pair : ExternallyUsedValues) {
        assert(!Pair.second.empty() &&
               "At least one DebugLoc must be inserted");
        // Add each externally used value to the final reduction.
        for (auto *I : Pair.second) {
          Builder.SetCurrentDebugLocation(I->getDebugLoc());
          VectorizedTree = Builder.CreateBinOp(ReductionOpcode, VectorizedTree,
                                               Pair.first, "bin.extra");
          propagateIRFlags(VectorizedTree, I);
        }
      }
      // Update users.
      ReductionRoot->replaceAllUsesWith(VectorizedTree);
    }
    return VectorizedTree != nullptr;
  }

  unsigned numReductionValues() const {
    return ReducedVals.size();
  }

private:
  /// \brief Calculate the cost of a reduction.
  int getReductionCost(TargetTransformInfo *TTI, Value *FirstReducedVal,
                       unsigned ReduxWidth) {
    Type *ScalarTy = FirstReducedVal->getType();
    Type *VecTy = VectorType::get(ScalarTy, ReduxWidth);

    int PairwiseRdxCost = TTI->getReductionCost(ReductionOpcode, VecTy, true);
    int SplittingRdxCost = TTI->getReductionCost(ReductionOpcode, VecTy, false);

    IsPairwiseReduction = PairwiseRdxCost < SplittingRdxCost;
    int VecReduxCost = IsPairwiseReduction ? PairwiseRdxCost : SplittingRdxCost;

    int ScalarReduxCost =
        (ReduxWidth - 1) *
        TTI->getArithmeticInstrCost(ReductionOpcode, ScalarTy);

    DEBUG(dbgs() << "SLP: Adding cost " << VecReduxCost - ScalarReduxCost
                 << " for reduction that starts with " << *FirstReducedVal
                 << " (It is a "
                 << (IsPairwiseReduction ? "pairwise" : "splitting")
                 << " reduction)\n");

    return VecReduxCost - ScalarReduxCost;
  }

  /// \brief Emit a horizontal reduction of the vectorized value.
  Value *emitReduction(Value *VectorizedValue, IRBuilder<> &Builder,
                       unsigned ReduxWidth, ArrayRef<Value *> RedOps) {
    assert(VectorizedValue && "Need to have a vectorized tree node");
    assert(isPowerOf2_32(ReduxWidth) &&
           "We only handle power-of-two reductions for now");

    Value *TmpVec = VectorizedValue;
    for (unsigned i = ReduxWidth / 2; i != 0; i >>= 1) {
      if (IsPairwiseReduction) {
        Value *LeftMask =
          createRdxShuffleMask(ReduxWidth, i, true, true, Builder);
        Value *RightMask =
          createRdxShuffleMask(ReduxWidth, i, true, false, Builder);

        Value *LeftShuf = Builder.CreateShuffleVector(
          TmpVec, UndefValue::get(TmpVec->getType()), LeftMask, "rdx.shuf.l");
        Value *RightShuf = Builder.CreateShuffleVector(
          TmpVec, UndefValue::get(TmpVec->getType()), (RightMask),
          "rdx.shuf.r");
        TmpVec = Builder.CreateBinOp(ReductionOpcode, LeftShuf, RightShuf,
                                     "bin.rdx");
      } else {
        Value *UpperHalf =
          createRdxShuffleMask(ReduxWidth, i, false, false, Builder);
        Value *Shuf = Builder.CreateShuffleVector(
          TmpVec, UndefValue::get(TmpVec->getType()), UpperHalf, "rdx.shuf");
        TmpVec = Builder.CreateBinOp(ReductionOpcode, TmpVec, Shuf, "bin.rdx");
      }
      propagateIRFlags(TmpVec, RedOps);
    }

    // The result is in the first element of the vector.
    return Builder.CreateExtractElement(TmpVec, Builder.getInt32(0));
  }
};
} // end anonymous namespace

/// \brief Recognize construction of vectors like
///  %ra = insertelement <4 x float> undef, float %s0, i32 0
///  %rb = insertelement <4 x float> %ra, float %s1, i32 1
///  %rc = insertelement <4 x float> %rb, float %s2, i32 2
///  %rd = insertelement <4 x float> %rc, float %s3, i32 3
///
/// Returns true if it matches
///
static bool findBuildVector(InsertElementInst *FirstInsertElem,
                            SmallVectorImpl<Value *> &BuildVector,
                            SmallVectorImpl<Value *> &BuildVectorOpds) {
  if (!isa<UndefValue>(FirstInsertElem->getOperand(0)))
    return false;

  InsertElementInst *IE = FirstInsertElem;
  while (true) {
    BuildVector.push_back(IE);
    BuildVectorOpds.push_back(IE->getOperand(1));

    if (IE->use_empty())
      return false;

    InsertElementInst *NextUse = dyn_cast<InsertElementInst>(IE->user_back());
    if (!NextUse)
      return true;

    // If this isn't the final use, make sure the next insertelement is the only
    // use. It's OK if the final constructed vector is used multiple times
    if (!IE->hasOneUse())
      return false;

    IE = NextUse;
  }

  return false;
}

/// \brief Like findBuildVector, but looks backwards for construction of aggregate.
///
/// \return true if it matches.
static bool findBuildAggregate(InsertValueInst *IV,
                               SmallVectorImpl<Value *> &BuildVector,
                               SmallVectorImpl<Value *> &BuildVectorOpds) {
  Value *V;
  do {
    BuildVector.push_back(IV);
    BuildVectorOpds.push_back(IV->getInsertedValueOperand());
    V = IV->getAggregateOperand();
    if (isa<UndefValue>(V))
      break;
    IV = dyn_cast<InsertValueInst>(V);
    if (!IV || !IV->hasOneUse())
      return false;
  } while (true);
  std::reverse(BuildVector.begin(), BuildVector.end());
  std::reverse(BuildVectorOpds.begin(), BuildVectorOpds.end());
  return true;
}

static bool PhiTypeSorterFunc(Value *V, Value *V2) {
  return V->getType() < V2->getType();
}

/// \brief Try and get a reduction value from a phi node.
///
/// Given a phi node \p P in a block \p ParentBB, consider possible reductions
/// if they come from either \p ParentBB or a containing loop latch.
///
/// \returns A candidate reduction value if possible, or \code nullptr \endcode
/// if not possible.
static Value *getReductionValue(const DominatorTree *DT, PHINode *P,
                                BasicBlock *ParentBB, LoopInfo *LI) {
  // There are situations where the reduction value is not dominated by the
  // reduction phi. Vectorizing such cases has been reported to cause
  // miscompiles. See PR25787.
  auto DominatedReduxValue = [&](Value *R) {
    return (
        dyn_cast<Instruction>(R) &&
        DT->dominates(P->getParent(), dyn_cast<Instruction>(R)->getParent()));
  };

  Value *Rdx = nullptr;

  // Return the incoming value if it comes from the same BB as the phi node.
  if (P->getIncomingBlock(0) == ParentBB) {
    Rdx = P->getIncomingValue(0);
  } else if (P->getIncomingBlock(1) == ParentBB) {
    Rdx = P->getIncomingValue(1);
  }

  if (Rdx && DominatedReduxValue(Rdx))
    return Rdx;

  // Otherwise, check whether we have a loop latch to look at.
  Loop *BBL = LI->getLoopFor(ParentBB);
  if (!BBL)
    return nullptr;
  BasicBlock *BBLatch = BBL->getLoopLatch();
  if (!BBLatch)
    return nullptr;

  // There is a loop latch, return the incoming value if it comes from
  // that. This reduction pattern occasionally turns up.
  if (P->getIncomingBlock(0) == BBLatch) {
    Rdx = P->getIncomingValue(0);
  } else if (P->getIncomingBlock(1) == BBLatch) {
    Rdx = P->getIncomingValue(1);
  }

  if (Rdx && DominatedReduxValue(Rdx))
    return Rdx;

  return nullptr;
}

namespace {
/// Tracks instructons and its children.
class WeakVHWithLevel final : public CallbackVH {
  /// Operand index of the instruction currently beeing analized.
  unsigned Level = 0;
  /// Is this the instruction that should be vectorized, or are we now
  /// processing children (i.e. operands of this instruction) for potential
  /// vectorization?
  bool IsInitial = true;

public:
  explicit WeakVHWithLevel() = default;
  WeakVHWithLevel(Value *V) : CallbackVH(V){};
  /// Restart children analysis each time it is repaced by the new instruction.
  void allUsesReplacedWith(Value *New) override {
    setValPtr(New);
    Level = 0;
    IsInitial = true;
  }
  /// Check if the instruction was not deleted during vectorization.
  bool isValid() const { return !getValPtr(); }
  /// Is the istruction itself must be vectorized?
  bool isInitial() const { return IsInitial; }
  /// Try to vectorize children.
  void clearInitial() { IsInitial = false; }
  /// Are all children processed already?
  bool isFinal() const {
    assert(getValPtr() &&
           (isa<Instruction>(getValPtr()) &&
            cast<Instruction>(getValPtr())->getNumOperands() >= Level));
    return getValPtr() &&
           cast<Instruction>(getValPtr())->getNumOperands() == Level;
  }
  /// Get next child operation.
  Value *nextOperand() {
    assert(getValPtr() && isa<Instruction>(getValPtr()) &&
           cast<Instruction>(getValPtr())->getNumOperands() > Level);
    return cast<Instruction>(getValPtr())->getOperand(Level++);
  }
  virtual ~WeakVHWithLevel() = default;
};
} // namespace

/// \brief Attempt to reduce a horizontal reduction.
/// If it is legal to match a horizontal reduction feeding
/// the phi node P with reduction operators Root in a basic block BB, then check
/// if it can be done.
/// \returns true if a horizontal reduction was matched and reduced.
/// \returns false if a horizontal reduction was not matched.
static bool canBeVectorized(
    PHINode *P, Instruction *Root, BasicBlock *BB, BoUpSLP &R,
    TargetTransformInfo *TTI,
    const function_ref<bool(BinaryOperator *, BoUpSLP &)> Vectorize) {
  if (!ShouldVectorizeHor)
    return false;

  if (!Root)
    return false;

  if (Root->getParent() != BB)
    return false;
  SmallVector<WeakVHWithLevel, 8> Stack(1, Root);
  SmallSet<Value *, 8> VisitedInstrs;
  bool Res = false;
  while (!Stack.empty()) {
    Value *V = Stack.back();
    if (!V) {
      Stack.pop_back();
      continue;
    }
    auto *Inst = dyn_cast<Instruction>(V);
    if (!Inst || isa<PHINode>(Inst)) {
      Stack.pop_back();
      continue;
    }
    if (Stack.back().isInitial()) {
      Stack.back().clearInitial();
      if (auto *BI = dyn_cast<BinaryOperator>(Inst)) {
        HorizontalReduction HorRdx;
        if (HorRdx.matchAssociativeReduction(P, BI)) {
          if (HorRdx.tryToReduce(R, TTI)) {
            Res = true;
            P = nullptr;
            continue;
          }
        }
        if (P) {
          Inst = dyn_cast<Instruction>(BI->getOperand(0));
          if (Inst == P)
            Inst = dyn_cast<Instruction>(BI->getOperand(1));
          if (!Inst) {
            P = nullptr;
            continue;
          }
        }
      }
      P = nullptr;
      if (Vectorize(dyn_cast<BinaryOperator>(Inst), R)) {
        Res = true;
        continue;
      }
    }
    if (Stack.back().isFinal()) {
      Stack.pop_back();
      continue;
    }

    if (auto *NextV = dyn_cast<Instruction>(Stack.back().nextOperand()))
      if (NextV->getParent() == BB && VisitedInstrs.insert(NextV).second &&
          Stack.size() < RecursionMaxDepth)
        Stack.push_back(NextV);
  }
  return Res;
}

bool SLPVectorizerPass::vectorizeRootInstruction(PHINode *P, Value *V,
                                                 BasicBlock *BB, BoUpSLP &R,
                                                 TargetTransformInfo *TTI) {
  if (!V)
    return false;
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;

  if (!isa<BinaryOperator>(I))
    P = nullptr;
  // Try to match and vectorize a horizontal reduction.
  return canBeVectorized(P, I, BB, R, TTI,
                         [this](BinaryOperator *BI, BoUpSLP &R) -> bool {
                           return tryToVectorize(BI, R);
                         });
}

bool SLPVectorizerPass::vectorizeChainsInBlock(BasicBlock *BB, BoUpSLP &R) {
  bool Changed = false;
  SmallVector<Value *, 4> Incoming;
  SmallSet<Value *, 16> VisitedInstrs;

  bool HaveVectorizedPhiNodes = true;
  while (HaveVectorizedPhiNodes) {
    HaveVectorizedPhiNodes = false;

    // Collect the incoming values from the PHIs.
    Incoming.clear();
    for (Instruction &I : *BB) {
      PHINode *P = dyn_cast<PHINode>(&I);
      if (!P)
        break;

      if (!VisitedInstrs.count(P))
        Incoming.push_back(P);
    }

    // Sort by type.
    std::stable_sort(Incoming.begin(), Incoming.end(), PhiTypeSorterFunc);

    // Try to vectorize elements base on their type.
    for (SmallVector<Value *, 4>::iterator IncIt = Incoming.begin(),
                                           E = Incoming.end();
         IncIt != E;) {

      // Look for the next elements with the same type.
      SmallVector<Value *, 4>::iterator SameTypeIt = IncIt;
      while (SameTypeIt != E &&
             (*SameTypeIt)->getType() == (*IncIt)->getType()) {
        VisitedInstrs.insert(*SameTypeIt);
        ++SameTypeIt;
      }

      // Try to vectorize them.
      unsigned NumElts = (SameTypeIt - IncIt);
      DEBUG(errs() << "SLP: Trying to vectorize starting at PHIs (" << NumElts << ")\n");
      if (NumElts > 1 && tryToVectorizeList(makeArrayRef(IncIt, NumElts), R)) {
        // Success start over because instructions might have been changed.
        HaveVectorizedPhiNodes = true;
        Changed = true;
        break;
      }

      // Start over at the next instruction of a different type (or the end).
      IncIt = SameTypeIt;
    }
  }

  VisitedInstrs.clear();

  for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; it++) {
    // We may go through BB multiple times so skip the one we have checked.
    if (!VisitedInstrs.insert(&*it).second)
      continue;

    if (isa<DbgInfoIntrinsic>(it))
      continue;

    // Try to vectorize reductions that use PHINodes.
    if (PHINode *P = dyn_cast<PHINode>(it)) {
      // Check that the PHI is a reduction PHI.
      if (P->getNumIncomingValues() != 2)
        return Changed;

      // Try to match and vectorize a horizontal reduction.
      if (vectorizeRootInstruction(P, getReductionValue(DT, P, BB, LI), BB, R,
                                   TTI)) {
        Changed = true;
        it = BB->begin();
        e = BB->end();
        continue;
      }
      continue;
    }

    if (ShouldStartVectorizeHorAtStore) {
      if (StoreInst *SI = dyn_cast<StoreInst>(it)) {
        // Try to match and vectorize a horizontal reduction.
        if (vectorizeRootInstruction(nullptr, SI->getValueOperand(), BB, R,
                                     TTI)) {
          Changed = true;
          it = BB->begin();
          e = BB->end();
          continue;
        }
      }
    }

    // Try to vectorize horizontal reductions feeding into a return.
    if (ReturnInst *RI = dyn_cast<ReturnInst>(it)) {
      if (RI->getNumOperands() != 0) {
        // Try to match and vectorize a horizontal reduction.
        if (vectorizeRootInstruction(nullptr, RI->getOperand(0), BB, R, TTI)) {
          Changed = true;
          it = BB->begin();
          e = BB->end();
          continue;
        }
      }
    }

    // Try to vectorize trees that start at compare instructions.
    if (CmpInst *CI = dyn_cast<CmpInst>(it)) {
      if (tryToVectorizePair(CI->getOperand(0), CI->getOperand(1), R)) {
        Changed = true;
        // We would like to start over since some instructions are deleted
        // and the iterator may become invalid value.
        it = BB->begin();
        e = BB->end();
        continue;
      }

      for (int I = 0; I < 2; ++I) {
        if (vectorizeRootInstruction(nullptr, CI->getOperand(I), BB, R, TTI)) {
          Changed = true;
          // We would like to start over since some instructions are deleted
          // and the iterator may become invalid value.
          it = BB->begin();
          e = BB->end();
          break;
        }
      }
      continue;
    }

    // Try to vectorize trees that start at insertelement instructions.
    if (InsertElementInst *FirstInsertElem = dyn_cast<InsertElementInst>(it)) {
      SmallVector<Value *, 16> BuildVector;
      SmallVector<Value *, 16> BuildVectorOpds;
      if (!findBuildVector(FirstInsertElem, BuildVector, BuildVectorOpds))
        continue;

      // Vectorize starting with the build vector operands ignoring the
      // BuildVector instructions for the purpose of scheduling and user
      // extraction.
      if (tryToVectorizeList(BuildVectorOpds, R, BuildVector)) {
        Changed = true;
        it = BB->begin();
        e = BB->end();
      }

      continue;
    }

    // Try to vectorize trees that start at insertvalue instructions feeding into
    // a store.
    if (StoreInst *SI = dyn_cast<StoreInst>(it)) {
      if (InsertValueInst *LastInsertValue = dyn_cast<InsertValueInst>(SI->getValueOperand())) {
        const DataLayout &DL = BB->getModule()->getDataLayout();
        if (R.canMapToVector(SI->getValueOperand()->getType(), DL)) {
          SmallVector<Value *, 16> BuildVector;
          SmallVector<Value *, 16> BuildVectorOpds;
          if (!findBuildAggregate(LastInsertValue, BuildVector, BuildVectorOpds))
            continue;

          DEBUG(dbgs() << "SLP: store of array mappable to vector: " << *SI << "\n");
          if (tryToVectorizeList(BuildVectorOpds, R, BuildVector, false)) {
            Changed = true;
            it = BB->begin();
            e = BB->end();
          }
          continue;
        }
      }
    }
  }

  return Changed;
}

bool SLPVectorizerPass::vectorizeGEPIndices(BasicBlock *BB, BoUpSLP &R) {
  auto Changed = false;
  for (auto &Entry : GEPs) {

    // If the getelementptr list has fewer than two elements, there's nothing
    // to do.
    if (Entry.second.size() < 2)
      continue;

    DEBUG(dbgs() << "SLP: Analyzing a getelementptr list of length "
                 << Entry.second.size() << ".\n");

    // We process the getelementptr list in chunks of 16 (like we do for
    // stores) to minimize compile-time.
    for (unsigned BI = 0, BE = Entry.second.size(); BI < BE; BI += 16) {
      auto Len = std::min<unsigned>(BE - BI, 16);
      auto GEPList = makeArrayRef(&Entry.second[BI], Len);

      // Initialize a set a candidate getelementptrs. Note that we use a
      // SetVector here to preserve program order. If the index computations
      // are vectorizable and begin with loads, we want to minimize the chance
      // of having to reorder them later.
      SetVector<Value *> Candidates(GEPList.begin(), GEPList.end());

      // Some of the candidates may have already been vectorized after we
      // initially collected them. If so, the WeakVHs will have nullified the
      // values, so remove them from the set of candidates.
      Candidates.remove(nullptr);

      // Remove from the set of candidates all pairs of getelementptrs with
      // constant differences. Such getelementptrs are likely not good
      // candidates for vectorization in a bottom-up phase since one can be
      // computed from the other. We also ensure all candidate getelementptr
      // indices are unique.
      for (int I = 0, E = GEPList.size(); I < E && Candidates.size() > 1; ++I) {
        auto *GEPI = cast<GetElementPtrInst>(GEPList[I]);
        if (!Candidates.count(GEPI))
          continue;
        auto *SCEVI = SE->getSCEV(GEPList[I]);
        for (int J = I + 1; J < E && Candidates.size() > 1; ++J) {
          auto *GEPJ = cast<GetElementPtrInst>(GEPList[J]);
          auto *SCEVJ = SE->getSCEV(GEPList[J]);
          if (isa<SCEVConstant>(SE->getMinusSCEV(SCEVI, SCEVJ))) {
            Candidates.remove(GEPList[I]);
            Candidates.remove(GEPList[J]);
          } else if (GEPI->idx_begin()->get() == GEPJ->idx_begin()->get()) {
            Candidates.remove(GEPList[J]);
          }
        }
      }

      // We break out of the above computation as soon as we know there are
      // fewer than two candidates remaining.
      if (Candidates.size() < 2)
        continue;

      // Add the single, non-constant index of each candidate to the bundle. We
      // ensured the indices met these constraints when we originally collected
      // the getelementptrs.
      SmallVector<Value *, 16> Bundle(Candidates.size());
      auto BundleIndex = 0u;
      for (auto *V : Candidates) {
        auto *GEP = cast<GetElementPtrInst>(V);
        auto *GEPIdx = GEP->idx_begin()->get();
        assert(GEP->getNumIndices() == 1 || !isa<Constant>(GEPIdx));
        Bundle[BundleIndex++] = GEPIdx;
      }

      // Try and vectorize the indices. We are currently only interested in
      // gather-like cases of the form:
      //
      // ... = g[a[0] - b[0]] + g[a[1] - b[1]] + ...
      //
      // where the loads of "a", the loads of "b", and the subtractions can be
      // performed in parallel. It's likely that detecting this pattern in a
      // bottom-up phase will be simpler and less costly than building a
      // full-blown top-down phase beginning at the consecutive loads.
      Changed |= tryToVectorizeList(Bundle, R);
    }
  }
  return Changed;
}

bool SLPVectorizerPass::vectorizeStoreChains(BoUpSLP &R) {
  bool Changed = false;
  // Attempt to sort and vectorize each of the store-groups.
  for (StoreListMap::iterator it = Stores.begin(), e = Stores.end(); it != e;
       ++it) {
    if (it->second.size() < 2)
      continue;

    DEBUG(dbgs() << "SLP: Analyzing a store chain of length "
          << it->second.size() << ".\n");

    // Process the stores in chunks of 16.
    // TODO: The limit of 16 inhibits greater vectorization factors.
    //       For example, AVX2 supports v32i8. Increasing this limit, however,
    //       may cause a significant compile-time increase.
    for (unsigned CI = 0, CE = it->second.size(); CI < CE; CI+=16) {
      unsigned Len = std::min<unsigned>(CE - CI, 16);
      Changed |= vectorizeStores(makeArrayRef(&it->second[CI], Len), R);
    }
  }
  return Changed;
}

char SLPVectorizer::ID = 0;
static const char lv_name[] = "SLP Vectorizer";
INITIALIZE_PASS_BEGIN(SLPVectorizer, SV_NAME, lv_name, false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(DemandedBitsWrapperPass)
INITIALIZE_PASS_END(SLPVectorizer, SV_NAME, lv_name, false, false)

namespace llvm {
Pass *createSLPVectorizerPass() { return new SLPVectorizer(); }
}
