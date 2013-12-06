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
#define SV_NAME "slp-vectorizer"
#define DEBUG_TYPE "SLP"

#include "llvm/Transforms/Vectorize.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>

using namespace llvm;

static cl::opt<int>
    SLPCostThreshold("slp-threshold", cl::init(0), cl::Hidden,
                     cl::desc("Only vectorize if you gain more than this "
                              "number "));

static cl::opt<bool>
ShouldVectorizeHor("slp-vectorize-hor", cl::init(false), cl::Hidden,
                   cl::desc("Attempt to vectorize horizontal reductions"));

static cl::opt<bool> ShouldStartVectorizeHorAtStore(
    "slp-vectorize-hor-store", cl::init(false), cl::Hidden,
    cl::desc(
        "Attempt to vectorize horizontal reductions feeding into a store"));

namespace {

static const unsigned MinVecRegSize = 128;

static const unsigned RecursionMaxDepth = 12;

/// A helper class for numbering instructions in multiple blocks.
/// Numbers start at zero for each basic block.
struct BlockNumbering {

  BlockNumbering(BasicBlock *Bb) : BB(Bb), Valid(false) {}

  BlockNumbering() : BB(0), Valid(false) {}

  void numberInstructions() {
    unsigned Loc = 0;
    InstrIdx.clear();
    InstrVec.clear();
    // Number the instructions in the block.
    for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
      InstrIdx[it] = Loc++;
      InstrVec.push_back(it);
      assert(InstrVec[InstrIdx[it]] == it && "Invalid allocation");
    }
    Valid = true;
  }

  int getIndex(Instruction *I) {
    assert(I->getParent() == BB && "Invalid instruction");
    if (!Valid)
      numberInstructions();
    assert(InstrIdx.count(I) && "Unknown instruction");
    return InstrIdx[I];
  }

  Instruction *getInstruction(unsigned loc) {
    if (!Valid)
      numberInstructions();
    assert(InstrVec.size() > loc && "Invalid Index");
    return InstrVec[loc];
  }

  void forget() { Valid = false; }

private:
  /// The block we are numbering.
  BasicBlock *BB;
  /// Is the block numbered.
  bool Valid;
  /// Maps instructions to numbers and back.
  SmallDenseMap<Instruction *, int> InstrIdx;
  /// Maps integers to Instructions.
  SmallVector<Instruction *, 32> InstrVec;
};

/// \returns the parent basic block if all of the instructions in \p VL
/// are in the same block or null otherwise.
static BasicBlock *getSameBlock(ArrayRef<Value *> VL) {
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  if (!I0)
    return 0;
  BasicBlock *BB = I0->getParent();
  for (int i = 1, e = VL.size(); i < e; i++) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    if (!I)
      return 0;

    if (BB != I->getParent())
      return 0;
  }
  return BB;
}

/// \returns True if all of the values in \p VL are constants.
static bool allConstant(ArrayRef<Value *> VL) {
  for (unsigned i = 0, e = VL.size(); i < e; ++i)
    if (!isa<Constant>(VL[i]))
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

/// \returns The opcode if all of the Instructions in \p VL have the same
/// opcode, or zero.
static unsigned getSameOpcode(ArrayRef<Value *> VL) {
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  if (!I0)
    return 0;
  unsigned Opcode = I0->getOpcode();
  for (int i = 1, e = VL.size(); i < e; i++) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    if (!I || Opcode != I->getOpcode())
      return 0;
  }
  return Opcode;
}

/// \returns \p I after propagating metadata from \p VL.
static Instruction *propagateMetadata(Instruction *I, ArrayRef<Value *> VL) {
  Instruction *I0 = cast<Instruction>(VL[0]);
  SmallVector<std::pair<unsigned, MDNode *>, 4> Metadata;
  I0->getAllMetadataOtherThanDebugLoc(Metadata);

  for (unsigned i = 0, n = Metadata.size(); i != n; ++i) {
    unsigned Kind = Metadata[i].first;
    MDNode *MD = Metadata[i].second;

    for (int i = 1, e = VL.size(); MD && i != e; i++) {
      Instruction *I = cast<Instruction>(VL[i]);
      MDNode *IMD = I->getMetadata(Kind);

      switch (Kind) {
      default:
        MD = 0; // Remove unknown metadata
        break;
      case LLVMContext::MD_tbaa:
        MD = MDNode::getMostGenericTBAA(MD, IMD);
        break;
      case LLVMContext::MD_fpmath:
        MD = MDNode::getMostGenericFPMath(MD, IMD);
        break;
      }
    }
    I->setMetadata(Kind, MD);
  }
  return I;
}

/// \returns The type that all of the values in \p VL have or null if there
/// are different types.
static Type* getSameType(ArrayRef<Value *> VL) {
  Type *Ty = VL[0]->getType();
  for (int i = 1, e = VL.size(); i < e; i++)
    if (VL[i]->getType() != Ty)
      return 0;

  return Ty;
}

/// \returns True if the ExtractElement instructions in VL can be vectorized
/// to use the original vector.
static bool CanReuseExtract(ArrayRef<Value *> VL) {
  assert(Instruction::ExtractElement == getSameOpcode(VL) && "Invalid opcode");
  // Check if all of the extracts come from the same vector and from the
  // correct offset.
  Value *VL0 = VL[0];
  ExtractElementInst *E0 = cast<ExtractElementInst>(VL0);
  Value *Vec = E0->getOperand(0);

  // We have to extract from the same vector type.
  unsigned NElts = Vec->getType()->getVectorNumElements();

  if (NElts != VL.size())
    return false;

  // Check that all of the indices extract from the correct offset.
  ConstantInt *CI = dyn_cast<ConstantInt>(E0->getOperand(1));
  if (!CI || CI->getZExtValue())
    return false;

  for (unsigned i = 1, e = VL.size(); i < e; ++i) {
    ExtractElementInst *E = cast<ExtractElementInst>(VL[i]);
    ConstantInt *CI = dyn_cast<ConstantInt>(E->getOperand(1));

    if (!CI || CI->getZExtValue() != i || E->getOperand(0) != Vec)
      return false;
  }

  return true;
}

static void reorderInputsAccordingToOpcode(ArrayRef<Value *> VL,
                                           SmallVectorImpl<Value *> &Left,
                                           SmallVectorImpl<Value *> &Right) {

  SmallVector<Value *, 16> OrigLeft, OrigRight;

  bool AllSameOpcodeLeft = true;
  bool AllSameOpcodeRight = true;
  for (unsigned i = 0, e = VL.size(); i != e; ++i) {
    Instruction *I = cast<Instruction>(VL[i]);
    Value *V0 = I->getOperand(0);
    Value *V1 = I->getOperand(1);

    OrigLeft.push_back(V0);
    OrigRight.push_back(V1);

    Instruction *I0 = dyn_cast<Instruction>(V0);
    Instruction *I1 = dyn_cast<Instruction>(V1);

    // Check whether all operands on one side have the same opcode. In this case
    // we want to preserve the original order and not make things worse by
    // reordering.
    AllSameOpcodeLeft = I0;
    AllSameOpcodeRight = I1;

    if (i && AllSameOpcodeLeft) {
      if(Instruction *P0 = dyn_cast<Instruction>(OrigLeft[i-1])) {
        if(P0->getOpcode() != I0->getOpcode())
          AllSameOpcodeLeft = false;
      } else
        AllSameOpcodeLeft = false;
    }
    if (i && AllSameOpcodeRight) {
      if(Instruction *P1 = dyn_cast<Instruction>(OrigRight[i-1])) {
        if(P1->getOpcode() != I1->getOpcode())
          AllSameOpcodeRight = false;
      } else
        AllSameOpcodeRight = false;
    }

    // Sort two opcodes. In the code below we try to preserve the ability to use
    // broadcast of values instead of individual inserts.
    // vl1 = load
    // vl2 = phi
    // vr1 = load
    // vr2 = vr2
    //    = vl1 x vr1
    //    = vl2 x vr2
    // If we just sorted according to opcode we would leave the first line in
    // tact but we would swap vl2 with vr2 because opcode(phi) > opcode(load).
    //    = vl1 x vr1
    //    = vr2 x vl2
    // Because vr2 and vr1 are from the same load we loose the opportunity of a
    // broadcast for the packed right side in the backend: we have [vr1, vl2]
    // instead of [vr1, vr2=vr1].
    if (I0 && I1) {
       if(!i && I0->getOpcode() > I1->getOpcode()) {
         Left.push_back(I1);
         Right.push_back(I0);
       } else if (i && I0->getOpcode() > I1->getOpcode() && Right[i-1] != I1) {
         // Try not to destroy a broad cast for no apparent benefit.
         Left.push_back(I1);
         Right.push_back(I0);
       } else if (i && I0->getOpcode() == I1->getOpcode() && Right[i-1] ==  I0) {
         // Try preserve broadcasts.
         Left.push_back(I1);
         Right.push_back(I0);
       } else if (i && I0->getOpcode() == I1->getOpcode() && Left[i-1] == I1) {
         // Try preserve broadcasts.
         Left.push_back(I1);
         Right.push_back(I0);
       } else {
         Left.push_back(I0);
         Right.push_back(I1);
       }
       continue;
    }
    // One opcode, put the instruction on the right.
    if (I0) {
      Left.push_back(V1);
      Right.push_back(I0);
      continue;
    }
    Left.push_back(V0);
    Right.push_back(V1);
  }

  bool LeftBroadcast = isSplat(Left);
  bool RightBroadcast = isSplat(Right);

  // Don't reorder if the operands where good to begin with.
  if (!(LeftBroadcast || RightBroadcast) &&
      (AllSameOpcodeRight || AllSameOpcodeLeft)) {
    Left = OrigLeft;
    Right = OrigRight;
  }
}

/// Bottom Up SLP Vectorizer.
class BoUpSLP {
public:
  typedef SmallVector<Value *, 8> ValueList;
  typedef SmallVector<Instruction *, 16> InstrList;
  typedef SmallPtrSet<Value *, 16> ValueSet;
  typedef SmallVector<StoreInst *, 8> StoreList;

  BoUpSLP(Function *Func, ScalarEvolution *Se, DataLayout *Dl,
          TargetTransformInfo *Tti, AliasAnalysis *Aa, LoopInfo *Li,
          DominatorTree *Dt) :
    F(Func), SE(Se), DL(Dl), TTI(Tti), AA(Aa), LI(Li), DT(Dt),
    Builder(Se->getContext()) {
      // Setup the block numbering utility for all of the blocks in the
      // function.
      for (Function::iterator it = F->begin(), e = F->end(); it != e; ++it) {
        BasicBlock *BB = it;
        BlocksNumbers[BB] = BlockNumbering(BB);
      }
    }

  /// \brief Vectorize the tree that starts with the elements in \p VL.
  /// Returns the vectorized root.
  Value *vectorizeTree();

  /// \returns the vectorization cost of the subtree that starts at \p VL.
  /// A negative number means that this is profitable.
  int getTreeCost();

  /// Construct a vectorizable tree that starts at \p Roots and is possibly
  /// used by a reduction of \p RdxOps.
  void buildTree(ArrayRef<Value *> Roots, ValueSet *RdxOps = 0);

  /// Clear the internal data structures that are created by 'buildTree'.
  void deleteTree() {
    RdxOps = 0;
    VectorizableTree.clear();
    ScalarToTreeEntry.clear();
    MustGather.clear();
    ExternalUses.clear();
    MemBarrierIgnoreList.clear();
  }

  /// \returns true if the memory operations A and B are consecutive.
  bool isConsecutiveAccess(Value *A, Value *B);

  /// \brief Perform LICM and CSE on the newly generated gather sequences.
  void optimizeGatherSequence();
private:
  struct TreeEntry;

  /// \returns the cost of the vectorizable entry.
  int getEntryCost(TreeEntry *E);

  /// This is the recursive part of buildTree.
  void buildTree_rec(ArrayRef<Value *> Roots, unsigned Depth);

  /// Vectorize a single entry in the tree.
  Value *vectorizeTree(TreeEntry *E);

  /// Vectorize a single entry in the tree, starting in \p VL.
  Value *vectorizeTree(ArrayRef<Value *> VL);

  /// \returns the pointer to the vectorized value if \p VL is already
  /// vectorized, or NULL. They may happen in cycles.
  Value *alreadyVectorized(ArrayRef<Value *> VL) const;

  /// \brief Take the pointer operand from the Load/Store instruction.
  /// \returns NULL if this is not a valid Load/Store instruction.
  static Value *getPointerOperand(Value *I);

  /// \brief Take the address space operand from the Load/Store instruction.
  /// \returns -1 if this is not a valid Load/Store instruction.
  static unsigned getAddressSpaceOperand(Value *I);

  /// \returns the scalarization cost for this type. Scalarization in this
  /// context means the creation of vectors from a group of scalars.
  int getGatherCost(Type *Ty);

  /// \returns the scalarization cost for this list of values. Assuming that
  /// this subtree gets vectorized, we may need to extract the values from the
  /// roots. This method calculates the cost of extracting the values.
  int getGatherCost(ArrayRef<Value *> VL);

  /// \returns the AA location that is being access by the instruction.
  AliasAnalysis::Location getLocation(Instruction *I);

  /// \brief Checks if it is possible to sink an instruction from
  /// \p Src to \p Dst.
  /// \returns the pointer to the barrier instruction if we can't sink.
  Value *getSinkBarrier(Instruction *Src, Instruction *Dst);

  /// \returns the index of the last instruction in the BB from \p VL.
  int getLastIndex(ArrayRef<Value *> VL);

  /// \returns the Instruction in the bundle \p VL.
  Instruction *getLastInstruction(ArrayRef<Value *> VL);

  /// \brief Set the Builder insert point to one after the last instruction in
  /// the bundle
  void setInsertPointAfterBundle(ArrayRef<Value *> VL);

  /// \returns a vector from a collection of scalars in \p VL.
  Value *Gather(ArrayRef<Value *> VL, VectorType *Ty);

  /// \returns whether the VectorizableTree is fully vectoriable and will
  /// be beneficial even the tree height is tiny.
  bool isFullyVectorizableTinyTree(); 

  struct TreeEntry {
    TreeEntry() : Scalars(), VectorizedValue(0), LastScalarIndex(0),
    NeedToGather(0) {}

    /// \returns true if the scalars in VL are equal to this entry.
    bool isSame(ArrayRef<Value *> VL) const {
      assert(VL.size() == Scalars.size() && "Invalid size");
      return std::equal(VL.begin(), VL.end(), Scalars.begin());
    }

    /// A vector of scalars.
    ValueList Scalars;

    /// The Scalars are vectorized into this value. It is initialized to Null.
    Value *VectorizedValue;

    /// The index in the basic block of the last scalar.
    int LastScalarIndex;

    /// Do we need to gather this sequence ?
    bool NeedToGather;
  };

  /// Create a new VectorizableTree entry.
  TreeEntry *newTreeEntry(ArrayRef<Value *> VL, bool Vectorized) {
    VectorizableTree.push_back(TreeEntry());
    int idx = VectorizableTree.size() - 1;
    TreeEntry *Last = &VectorizableTree[idx];
    Last->Scalars.insert(Last->Scalars.begin(), VL.begin(), VL.end());
    Last->NeedToGather = !Vectorized;
    if (Vectorized) {
      Last->LastScalarIndex = getLastIndex(VL);
      for (int i = 0, e = VL.size(); i != e; ++i) {
        assert(!ScalarToTreeEntry.count(VL[i]) && "Scalar already in tree!");
        ScalarToTreeEntry[VL[i]] = idx;
      }
    } else {
      Last->LastScalarIndex = 0;
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
      Scalar(S), User(U), Lane(L){};
    // Which scalar in our function.
    Value *Scalar;
    // Which user that uses the scalar.
    llvm::User *User;
    // Which lane does the scalar belong to.
    int Lane;
  };
  typedef SmallVector<ExternalUser, 16> UserList;

  /// A list of values that need to extracted out of the tree.
  /// This list holds pairs of (Internal Scalar : External User).
  UserList ExternalUses;

  /// A list of instructions to ignore while sinking
  /// memory instructions. This map must be reset between runs of getCost.
  ValueSet MemBarrierIgnoreList;

  /// Holds all of the instructions that we gathered.
  SetVector<Instruction *> GatherSeq;
  /// A list of blocks that we are going to CSE.
  SmallSet<BasicBlock *, 8> CSEBlocks;

  /// Numbers instructions in different blocks.
  DenseMap<BasicBlock *, BlockNumbering> BlocksNumbers;

  /// Reduction operators.
  ValueSet *RdxOps;

  // Analysis and block reference.
  Function *F;
  ScalarEvolution *SE;
  DataLayout *DL;
  TargetTransformInfo *TTI;
  AliasAnalysis *AA;
  LoopInfo *LI;
  DominatorTree *DT;
  /// Instruction builder to construct the vectorized tree.
  IRBuilder<> Builder;
};

void BoUpSLP::buildTree(ArrayRef<Value *> Roots, ValueSet *Rdx) {
  deleteTree();
  RdxOps = Rdx;
  if (!getSameType(Roots))
    return;
  buildTree_rec(Roots, 0);

  // Collect the values that we need to extract from the tree.
  for (int EIdx = 0, EE = VectorizableTree.size(); EIdx < EE; ++EIdx) {
    TreeEntry *Entry = &VectorizableTree[EIdx];

    // For each lane:
    for (int Lane = 0, LE = Entry->Scalars.size(); Lane != LE; ++Lane) {
      Value *Scalar = Entry->Scalars[Lane];

      // No need to handle users of gathered values.
      if (Entry->NeedToGather)
        continue;

      for (Value::use_iterator User = Scalar->use_begin(),
           UE = Scalar->use_end(); User != UE; ++User) {
        DEBUG(dbgs() << "SLP: Checking user:" << **User << ".\n");

        // Skip in-tree scalars that become vectors.
        if (ScalarToTreeEntry.count(*User)) {
          DEBUG(dbgs() << "SLP: \tInternal user will be removed:" <<
                **User << ".\n");
          int Idx = ScalarToTreeEntry[*User]; (void) Idx;
          assert(!VectorizableTree[Idx].NeedToGather && "Bad state");
          continue;
        }
        Instruction *UserInst = dyn_cast<Instruction>(*User);
        if (!UserInst)
          continue;

        // Ignore uses that are part of the reduction.
        if (Rdx && std::find(Rdx->begin(), Rdx->end(), UserInst) != Rdx->end())
          continue;

        DEBUG(dbgs() << "SLP: Need to extract:" << **User << " from lane " <<
              Lane << " from " << *Scalar << ".\n");
        ExternalUses.push_back(ExternalUser(Scalar, *User, Lane));
      }
    }
  }
}


void BoUpSLP::buildTree_rec(ArrayRef<Value *> VL, unsigned Depth) {
  bool SameTy = getSameType(VL); (void)SameTy;
  assert(SameTy && "Invalid types!");

  if (Depth == RecursionMaxDepth) {
    DEBUG(dbgs() << "SLP: Gathering due to max recursion depth.\n");
    newTreeEntry(VL, false);
    return;
  }

  // Don't handle vectors.
  if (VL[0]->getType()->isVectorTy()) {
    DEBUG(dbgs() << "SLP: Gathering due to vector type.\n");
    newTreeEntry(VL, false);
    return;
  }

  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    if (SI->getValueOperand()->getType()->isVectorTy()) {
      DEBUG(dbgs() << "SLP: Gathering due to store vector type.\n");
      newTreeEntry(VL, false);
      return;
    }

  // If all of the operands are identical or constant we have a simple solution.
  if (allConstant(VL) || isSplat(VL) || !getSameBlock(VL) ||
      !getSameOpcode(VL)) {
    DEBUG(dbgs() << "SLP: Gathering due to C,S,B,O. \n");
    newTreeEntry(VL, false);
    return;
  }

  // We now know that this is a vector of instructions of the same type from
  // the same block.

  // Check if this is a duplicate of another entry.
  if (ScalarToTreeEntry.count(VL[0])) {
    int Idx = ScalarToTreeEntry[VL[0]];
    TreeEntry *E = &VectorizableTree[Idx];
    for (unsigned i = 0, e = VL.size(); i != e; ++i) {
      DEBUG(dbgs() << "SLP: \tChecking bundle: " << *VL[i] << ".\n");
      if (E->Scalars[i] != VL[i]) {
        DEBUG(dbgs() << "SLP: Gathering due to partial overlap.\n");
        newTreeEntry(VL, false);
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
      newTreeEntry(VL, false);
      return;
    }
  }

  // If any of the scalars appears in the table OR it is marked as a value that
  // needs to stat scalar then we need to gather the scalars.
  for (unsigned i = 0, e = VL.size(); i != e; ++i) {
    if (ScalarToTreeEntry.count(VL[i]) || MustGather.count(VL[i])) {
      DEBUG(dbgs() << "SLP: Gathering due to gathered scalar. \n");
      newTreeEntry(VL, false);
      return;
    }
  }

  // Check that all of the users of the scalars that we want to vectorize are
  // schedulable.
  Instruction *VL0 = cast<Instruction>(VL[0]);
  int MyLastIndex = getLastIndex(VL);
  BasicBlock *BB = cast<Instruction>(VL0)->getParent();

  for (unsigned i = 0, e = VL.size(); i != e; ++i) {
    Instruction *Scalar = cast<Instruction>(VL[i]);
    DEBUG(dbgs() << "SLP: Checking users of  " << *Scalar << ". \n");
    for (Value::use_iterator U = Scalar->use_begin(), UE = Scalar->use_end();
         U != UE; ++U) {
      DEBUG(dbgs() << "SLP: \tUser " << **U << ". \n");
      Instruction *User = dyn_cast<Instruction>(*U);
      if (!User) {
        DEBUG(dbgs() << "SLP: Gathering due unknown user. \n");
        newTreeEntry(VL, false);
        return;
      }

      // We don't care if the user is in a different basic block.
      BasicBlock *UserBlock = User->getParent();
      if (UserBlock != BB) {
        DEBUG(dbgs() << "SLP: User from a different basic block "
              << *User << ". \n");
        continue;
      }

      // If this is a PHINode within this basic block then we can place the
      // extract wherever we want.
      if (isa<PHINode>(*User)) {
        DEBUG(dbgs() << "SLP: \tWe can schedule PHIs:" << *User << ". \n");
        continue;
      }

      // Check if this is a safe in-tree user.
      if (ScalarToTreeEntry.count(User)) {
        int Idx = ScalarToTreeEntry[User];
        int VecLocation = VectorizableTree[Idx].LastScalarIndex;
        if (VecLocation <= MyLastIndex) {
          DEBUG(dbgs() << "SLP: Gathering due to unschedulable vector. \n");
          newTreeEntry(VL, false);
          return;
        }
        DEBUG(dbgs() << "SLP: In-tree user (" << *User << ") at #" <<
              VecLocation << " vector value (" << *Scalar << ") at #"
              << MyLastIndex << ".\n");
        continue;
      }

      // This user is part of the reduction.
      if (RdxOps && RdxOps->count(User))
        continue;

      // Make sure that we can schedule this unknown user.
      BlockNumbering &BN = BlocksNumbers[BB];
      int UserIndex = BN.getIndex(User);
      if (UserIndex < MyLastIndex) {

        DEBUG(dbgs() << "SLP: Can't schedule extractelement for "
              << *User << ". \n");
        newTreeEntry(VL, false);
        return;
      }
    }
  }

  // Check that every instructions appears once in this bundle.
  for (unsigned i = 0, e = VL.size(); i < e; ++i)
    for (unsigned j = i+1; j < e; ++j)
      if (VL[i] == VL[j]) {
        DEBUG(dbgs() << "SLP: Scalar used twice in bundle.\n");
        newTreeEntry(VL, false);
        return;
      }

  // Check that instructions in this bundle don't reference other instructions.
  // The runtime of this check is O(N * N-1 * uses(N)) and a typical N is 4.
  for (unsigned i = 0, e = VL.size(); i < e; ++i) {
    for (Value::use_iterator U = VL[i]->use_begin(), UE = VL[i]->use_end();
         U != UE; ++U) {
      for (unsigned j = 0; j < e; ++j) {
        if (i != j && *U == VL[j]) {
          DEBUG(dbgs() << "SLP: Intra-bundle dependencies!" << **U << ". \n");
          newTreeEntry(VL, false);
          return;
        }
      }
    }
  }

  DEBUG(dbgs() << "SLP: We are able to schedule this bundle.\n");

  unsigned Opcode = getSameOpcode(VL);

  // Check if it is safe to sink the loads or the stores.
  if (Opcode == Instruction::Load || Opcode == Instruction::Store) {
    Instruction *Last = getLastInstruction(VL);

    for (unsigned i = 0, e = VL.size(); i < e; ++i) {
      if (VL[i] == Last)
        continue;
      Value *Barrier = getSinkBarrier(cast<Instruction>(VL[i]), Last);
      if (Barrier) {
        DEBUG(dbgs() << "SLP: Can't sink " << *VL[i] << "\n down to " << *Last
              << "\n because of " << *Barrier << ".  Gathering.\n");
        newTreeEntry(VL, false);
        return;
      }
    }
  }

  switch (Opcode) {
    case Instruction::PHI: {
      PHINode *PH = dyn_cast<PHINode>(VL0);

      // Check for terminator values (e.g. invoke).
      for (unsigned j = 0; j < VL.size(); ++j)
        for (unsigned i = 0, e = PH->getNumIncomingValues(); i < e; ++i) {
          TerminatorInst *Term = dyn_cast<TerminatorInst>(cast<PHINode>(VL[j])->getIncomingValue(i));
          if (Term) {
            DEBUG(dbgs() << "SLP: Need to swizzle PHINodes (TerminatorInst use).\n");
            newTreeEntry(VL, false);
            return;
          }
        }

      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of PHINodes.\n");

      for (unsigned i = 0, e = PH->getNumIncomingValues(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (unsigned j = 0; j < VL.size(); ++j)
          Operands.push_back(cast<PHINode>(VL[j])->getIncomingValue(i));

        buildTree_rec(Operands, Depth + 1);
      }
      return;
    }
    case Instruction::ExtractElement: {
      bool Reuse = CanReuseExtract(VL);
      if (Reuse) {
        DEBUG(dbgs() << "SLP: Reusing extract sequence.\n");
      }
      newTreeEntry(VL, Reuse);
      return;
    }
    case Instruction::Load: {
      // Check if the loads are consecutive or of we need to swizzle them.
      for (unsigned i = 0, e = VL.size() - 1; i < e; ++i) {
        LoadInst *L = cast<LoadInst>(VL[i]);
        if (!L->isSimple() || !isConsecutiveAccess(VL[i], VL[i + 1])) {
          newTreeEntry(VL, false);
          DEBUG(dbgs() << "SLP: Need to swizzle loads.\n");
          return;
        }
      }
      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of loads.\n");
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
      for (unsigned i = 0; i < VL.size(); ++i) {
        Type *Ty = cast<Instruction>(VL[i])->getOperand(0)->getType();
        if (Ty != SrcTy || Ty->isAggregateType() || Ty->isVectorTy()) {
          newTreeEntry(VL, false);
          DEBUG(dbgs() << "SLP: Gathering casts with different src types.\n");
          return;
        }
      }
      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of casts.\n");

      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (unsigned j = 0; j < VL.size(); ++j)
          Operands.push_back(cast<Instruction>(VL[j])->getOperand(i));

        buildTree_rec(Operands, Depth+1);
      }
      return;
    }
    case Instruction::ICmp:
    case Instruction::FCmp: {
      // Check that all of the compares have the same predicate.
      CmpInst::Predicate P0 = dyn_cast<CmpInst>(VL0)->getPredicate();
      Type *ComparedTy = cast<Instruction>(VL[0])->getOperand(0)->getType();
      for (unsigned i = 1, e = VL.size(); i < e; ++i) {
        CmpInst *Cmp = cast<CmpInst>(VL[i]);
        if (Cmp->getPredicate() != P0 ||
            Cmp->getOperand(0)->getType() != ComparedTy) {
          newTreeEntry(VL, false);
          DEBUG(dbgs() << "SLP: Gathering cmp with different predicate.\n");
          return;
        }
      }

      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of compares.\n");

      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (unsigned j = 0; j < VL.size(); ++j)
          Operands.push_back(cast<Instruction>(VL[j])->getOperand(i));

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
      newTreeEntry(VL, true);
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
        for (unsigned j = 0; j < VL.size(); ++j)
          Operands.push_back(cast<Instruction>(VL[j])->getOperand(i));

        buildTree_rec(Operands, Depth+1);
      }
      return;
    }
    case Instruction::Store: {
      // Check if the stores are consecutive or of we need to swizzle them.
      for (unsigned i = 0, e = VL.size() - 1; i < e; ++i)
        if (!isConsecutiveAccess(VL[i], VL[i + 1])) {
          newTreeEntry(VL, false);
          DEBUG(dbgs() << "SLP: Non consecutive store.\n");
          return;
        }

      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of stores.\n");

      ValueList Operands;
      for (unsigned j = 0; j < VL.size(); ++j)
        Operands.push_back(cast<Instruction>(VL[j])->getOperand(0));

      // We can ignore these values because we are sinking them down.
      MemBarrierIgnoreList.insert(VL.begin(), VL.end());
      buildTree_rec(Operands, Depth + 1);
      return;
    }
    default:
      newTreeEntry(VL, false);
      DEBUG(dbgs() << "SLP: Gathering unknown instruction.\n");
      return;
  }
}

int BoUpSLP::getEntryCost(TreeEntry *E) {
  ArrayRef<Value*> VL = E->Scalars;

  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());

  if (E->NeedToGather) {
    if (allConstant(VL))
      return 0;
    if (isSplat(VL)) {
      return TTI->getShuffleCost(TargetTransformInfo::SK_Broadcast, VecTy, 0);
    }
    return getGatherCost(E->Scalars);
  }

  assert(getSameOpcode(VL) && getSameType(VL) && getSameBlock(VL) &&
         "Invalid VL");
  Instruction *VL0 = cast<Instruction>(VL[0]);
  unsigned Opcode = VL0->getOpcode();
  switch (Opcode) {
    case Instruction::PHI: {
      return 0;
    }
    case Instruction::ExtractElement: {
      if (CanReuseExtract(VL))
        return 0;
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
      // Calculate the cost of this instruction.
      int ScalarCost = 0;
      int VecCost = 0;
      if (Opcode == Instruction::FCmp || Opcode == Instruction::ICmp ||
          Opcode == Instruction::Select) {
        VectorType *MaskTy = VectorType::get(Builder.getInt1Ty(), VL.size());
        ScalarCost = VecTy->getNumElements() *
        TTI->getCmpSelInstrCost(Opcode, ScalarTy, Builder.getInt1Ty());
        VecCost = TTI->getCmpSelInstrCost(Opcode, VecTy, MaskTy);
      } else {
        // Certain instructions can be cheaper to vectorize if they have a
        // constant second vector operand.
        TargetTransformInfo::OperandValueKind Op1VK =
            TargetTransformInfo::OK_AnyValue;
        TargetTransformInfo::OperandValueKind Op2VK =
            TargetTransformInfo::OK_UniformConstantValue;

        // Check whether all second operands are constant.
        for (unsigned i = 0; i < VL.size(); ++i)
          if (!isa<ConstantInt>(cast<Instruction>(VL[i])->getOperand(1))) {
            Op2VK = TargetTransformInfo::OK_AnyValue;
            break;
          }

        ScalarCost =
            VecTy->getNumElements() *
            TTI->getArithmeticInstrCost(Opcode, ScalarTy, Op1VK, Op2VK);
        VecCost = TTI->getArithmeticInstrCost(Opcode, VecTy, Op1VK, Op2VK);
      }
      return VecCost - ScalarCost;
    }
    case Instruction::Load: {
      // Cost of wide load - cost of scalar loads.
      int ScalarLdCost = VecTy->getNumElements() *
      TTI->getMemoryOpCost(Instruction::Load, ScalarTy, 1, 0);
      int VecLdCost = TTI->getMemoryOpCost(Instruction::Load, VecTy, 1, 0);
      return VecLdCost - ScalarLdCost;
    }
    case Instruction::Store: {
      // We know that we can merge the stores. Calculate the cost.
      int ScalarStCost = VecTy->getNumElements() *
      TTI->getMemoryOpCost(Instruction::Store, ScalarTy, 1, 0);
      int VecStCost = TTI->getMemoryOpCost(Instruction::Store, VecTy, 1, 0);
      return VecStCost - ScalarStCost;
    }
    default:
      llvm_unreachable("Unknown instruction");
  }
}

bool BoUpSLP::isFullyVectorizableTinyTree() {
  DEBUG(dbgs() << "SLP: Check whether the tree with height " <<
        VectorizableTree.size() << " is fully vectorizable .\n");

  // We only handle trees of height 2.
  if (VectorizableTree.size() != 2)
    return false;

  // Gathering cost would be too much for tiny trees.
  if (VectorizableTree[0].NeedToGather || VectorizableTree[1].NeedToGather) 
    return false; 

  return true; 
}

int BoUpSLP::getTreeCost() {
  int Cost = 0;
  DEBUG(dbgs() << "SLP: Calculating cost for tree of size " <<
        VectorizableTree.size() << ".\n");

  // We only vectorize tiny trees if it is fully vectorizable.
  if (VectorizableTree.size() < 3 && !isFullyVectorizableTinyTree()) {
    if (!VectorizableTree.size()) {
      assert(!ExternalUses.size() && "We should not have any external users");
    }
    return INT_MAX;
  }

  unsigned BundleWidth = VectorizableTree[0].Scalars.size();

  for (unsigned i = 0, e = VectorizableTree.size(); i != e; ++i) {
    int C = getEntryCost(&VectorizableTree[i]);
    DEBUG(dbgs() << "SLP: Adding cost " << C << " for bundle that starts with "
          << *VectorizableTree[i].Scalars[0] << " .\n");
    Cost += C;
  }

  int ExtractCost = 0;
  for (UserList::iterator I = ExternalUses.begin(), E = ExternalUses.end();
       I != E; ++I) {

    VectorType *VecTy = VectorType::get(I->Scalar->getType(), BundleWidth);
    ExtractCost += TTI->getVectorInstrCost(Instruction::ExtractElement, VecTy,
                                           I->Lane);
  }


  DEBUG(dbgs() << "SLP: Total Cost " << Cost + ExtractCost<< ".\n");
  return  Cost + ExtractCost;
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

AliasAnalysis::Location BoUpSLP::getLocation(Instruction *I) {
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return AA->getLocation(SI);
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return AA->getLocation(LI);
  return AliasAnalysis::Location();
}

Value *BoUpSLP::getPointerOperand(Value *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->getPointerOperand();
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->getPointerOperand();
  return 0;
}

unsigned BoUpSLP::getAddressSpaceOperand(Value *I) {
  if (LoadInst *L = dyn_cast<LoadInst>(I))
    return L->getPointerAddressSpace();
  if (StoreInst *S = dyn_cast<StoreInst>(I))
    return S->getPointerAddressSpace();
  return -1;
}

bool BoUpSLP::isConsecutiveAccess(Value *A, Value *B) {
  Value *PtrA = getPointerOperand(A);
  Value *PtrB = getPointerOperand(B);
  unsigned ASA = getAddressSpaceOperand(A);
  unsigned ASB = getAddressSpaceOperand(B);

  // Check that the address spaces match and that the pointers are valid.
  if (!PtrA || !PtrB || (ASA != ASB))
    return false;

  // Make sure that A and B are different pointers of the same type.
  if (PtrA == PtrB || PtrA->getType() != PtrB->getType())
    return false;

  unsigned PtrBitWidth = DL->getPointerSizeInBits(ASA);
  Type *Ty = cast<PointerType>(PtrA->getType())->getElementType();
  APInt Size(PtrBitWidth, DL->getTypeStoreSize(Ty));

  APInt OffsetA(PtrBitWidth, 0), OffsetB(PtrBitWidth, 0);
  PtrA = PtrA->stripAndAccumulateInBoundsConstantOffsets(*DL, OffsetA);
  PtrB = PtrB->stripAndAccumulateInBoundsConstantOffsets(*DL, OffsetB);

  APInt OffsetDelta = OffsetB - OffsetA;

  // Check if they are based on the same pointer. That makes the offsets
  // sufficient.
  if (PtrA == PtrB)
    return OffsetDelta == Size;

  // Compute the necessary base pointer delta to have the necessary final delta
  // equal to the size.
  APInt BaseDelta = Size - OffsetDelta;

  // Otherwise compute the distance with SCEV between the base pointers.
  const SCEV *PtrSCEVA = SE->getSCEV(PtrA);
  const SCEV *PtrSCEVB = SE->getSCEV(PtrB);
  const SCEV *C = SE->getConstant(BaseDelta);
  const SCEV *X = SE->getAddExpr(PtrSCEVA, C);
  return X == PtrSCEVB;
}

Value *BoUpSLP::getSinkBarrier(Instruction *Src, Instruction *Dst) {
  assert(Src->getParent() == Dst->getParent() && "Not the same BB");
  BasicBlock::iterator I = Src, E = Dst;
  /// Scan all of the instruction from SRC to DST and check if
  /// the source may alias.
  for (++I; I != E; ++I) {
    // Ignore store instructions that are marked as 'ignore'.
    if (MemBarrierIgnoreList.count(I))
      continue;
    if (Src->mayWriteToMemory()) /* Write */ {
      if (!I->mayReadOrWriteMemory())
        continue;
    } else /* Read */ {
      if (!I->mayWriteToMemory())
        continue;
    }
    AliasAnalysis::Location A = getLocation(&*I);
    AliasAnalysis::Location B = getLocation(Src);

    if (!A.Ptr || !B.Ptr || AA->alias(A, B))
      return I;
  }
  return 0;
}

int BoUpSLP::getLastIndex(ArrayRef<Value *> VL) {
  BasicBlock *BB = cast<Instruction>(VL[0])->getParent();
  assert(BB == getSameBlock(VL) && BlocksNumbers.count(BB) && "Invalid block");
  BlockNumbering &BN = BlocksNumbers[BB];

  int MaxIdx = BN.getIndex(BB->getFirstNonPHI());
  for (unsigned i = 0, e = VL.size(); i < e; ++i)
    MaxIdx = std::max(MaxIdx, BN.getIndex(cast<Instruction>(VL[i])));
  return MaxIdx;
}

Instruction *BoUpSLP::getLastInstruction(ArrayRef<Value *> VL) {
  BasicBlock *BB = cast<Instruction>(VL[0])->getParent();
  assert(BB == getSameBlock(VL) && BlocksNumbers.count(BB) && "Invalid block");
  BlockNumbering &BN = BlocksNumbers[BB];

  int MaxIdx = BN.getIndex(cast<Instruction>(VL[0]));
  for (unsigned i = 1, e = VL.size(); i < e; ++i)
    MaxIdx = std::max(MaxIdx, BN.getIndex(cast<Instruction>(VL[i])));
  Instruction *I = BN.getInstruction(MaxIdx);
  assert(I && "bad location");
  return I;
}

void BoUpSLP::setInsertPointAfterBundle(ArrayRef<Value *> VL) {
  Instruction *VL0 = cast<Instruction>(VL[0]);
  Instruction *LastInst = getLastInstruction(VL);
  BasicBlock::iterator NextInst = LastInst;
  ++NextInst;
  Builder.SetInsertPoint(VL0->getParent(), NextInst);
  Builder.SetCurrentDebugLocation(VL0->getDebugLoc());
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
  return 0;
}

Value *BoUpSLP::vectorizeTree(ArrayRef<Value *> VL) {
  if (ScalarToTreeEntry.count(VL[0])) {
    int Idx = ScalarToTreeEntry[VL[0]];
    TreeEntry *E = &VectorizableTree[Idx];
    if (E->isSame(VL))
      return vectorizeTree(E);
  }

  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());

  return Gather(VL, VecTy);
}

Value *BoUpSLP::vectorizeTree(TreeEntry *E) {
  IRBuilder<>::InsertPointGuard Guard(Builder);

  if (E->VectorizedValue) {
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
    return Gather(E->Scalars, VecTy);
  }

  unsigned Opcode = VL0->getOpcode();
  assert(Opcode == getSameOpcode(E->Scalars) && "Invalid opcode");

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

        if (!VisitedBBs.insert(IBB)) {
          NewPhi->addIncoming(NewPhi->getIncomingValueForBlock(IBB), IBB);
          continue;
        }

        // Prepare the operand vector.
        for (unsigned j = 0; j < E->Scalars.size(); ++j)
          Operands.push_back(cast<PHINode>(E->Scalars[j])->
                             getIncomingValueForBlock(IBB));

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
      if (CanReuseExtract(E->Scalars)) {
        Value *V = VL0->getOperand(0);
        E->VectorizedValue = V;
        return V;
      }
      return Gather(E->Scalars, VecTy);
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
      for (int i = 0, e = E->Scalars.size(); i < e; ++i)
        INVL.push_back(cast<Instruction>(E->Scalars[i])->getOperand(0));

      setInsertPointAfterBundle(E->Scalars);

      Value *InVec = vectorizeTree(INVL);

      if (Value *V = alreadyVectorized(E->Scalars))
        return V;

      CastInst *CI = dyn_cast<CastInst>(VL0);
      Value *V = Builder.CreateCast(CI->getOpcode(), InVec, VecTy);
      E->VectorizedValue = V;
      return V;
    }
    case Instruction::FCmp:
    case Instruction::ICmp: {
      ValueList LHSV, RHSV;
      for (int i = 0, e = E->Scalars.size(); i < e; ++i) {
        LHSV.push_back(cast<Instruction>(E->Scalars[i])->getOperand(0));
        RHSV.push_back(cast<Instruction>(E->Scalars[i])->getOperand(1));
      }

      setInsertPointAfterBundle(E->Scalars);

      Value *L = vectorizeTree(LHSV);
      Value *R = vectorizeTree(RHSV);

      if (Value *V = alreadyVectorized(E->Scalars))
        return V;

      CmpInst::Predicate P0 = dyn_cast<CmpInst>(VL0)->getPredicate();
      Value *V;
      if (Opcode == Instruction::FCmp)
        V = Builder.CreateFCmp(P0, L, R);
      else
        V = Builder.CreateICmp(P0, L, R);

      E->VectorizedValue = V;
      return V;
    }
    case Instruction::Select: {
      ValueList TrueVec, FalseVec, CondVec;
      for (int i = 0, e = E->Scalars.size(); i < e; ++i) {
        CondVec.push_back(cast<Instruction>(E->Scalars[i])->getOperand(0));
        TrueVec.push_back(cast<Instruction>(E->Scalars[i])->getOperand(1));
        FalseVec.push_back(cast<Instruction>(E->Scalars[i])->getOperand(2));
      }

      setInsertPointAfterBundle(E->Scalars);

      Value *Cond = vectorizeTree(CondVec);
      Value *True = vectorizeTree(TrueVec);
      Value *False = vectorizeTree(FalseVec);

      if (Value *V = alreadyVectorized(E->Scalars))
        return V;

      Value *V = Builder.CreateSelect(Cond, True, False);
      E->VectorizedValue = V;
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
        for (int i = 0, e = E->Scalars.size(); i < e; ++i) {
          LHSVL.push_back(cast<Instruction>(E->Scalars[i])->getOperand(0));
          RHSVL.push_back(cast<Instruction>(E->Scalars[i])->getOperand(1));
        }

      setInsertPointAfterBundle(E->Scalars);

      Value *LHS = vectorizeTree(LHSVL);
      Value *RHS = vectorizeTree(RHSVL);

      if (LHS == RHS && isa<Instruction>(LHS)) {
        assert((VL0->getOperand(0) == VL0->getOperand(1)) && "Invalid order");
      }

      if (Value *V = alreadyVectorized(E->Scalars))
        return V;

      BinaryOperator *BinOp = cast<BinaryOperator>(VL0);
      Value *V = Builder.CreateBinOp(BinOp->getOpcode(), LHS, RHS);
      E->VectorizedValue = V;

      if (Instruction *I = dyn_cast<Instruction>(V))
        return propagateMetadata(I, E->Scalars);

      return V;
    }
    case Instruction::Load: {
      // Loads are inserted at the head of the tree because we don't want to
      // sink them all the way down past store instructions.
      setInsertPointAfterBundle(E->Scalars);

      LoadInst *LI = cast<LoadInst>(VL0);
      unsigned AS = LI->getPointerAddressSpace();

      Value *VecPtr = Builder.CreateBitCast(LI->getPointerOperand(),
                                            VecTy->getPointerTo(AS));
      unsigned Alignment = LI->getAlignment();
      LI = Builder.CreateLoad(VecPtr);
      LI->setAlignment(Alignment);
      E->VectorizedValue = LI;
      return propagateMetadata(LI, E->Scalars);
    }
    case Instruction::Store: {
      StoreInst *SI = cast<StoreInst>(VL0);
      unsigned Alignment = SI->getAlignment();
      unsigned AS = SI->getPointerAddressSpace();

      ValueList ValueOp;
      for (int i = 0, e = E->Scalars.size(); i < e; ++i)
        ValueOp.push_back(cast<StoreInst>(E->Scalars[i])->getValueOperand());

      setInsertPointAfterBundle(E->Scalars);

      Value *VecValue = vectorizeTree(ValueOp);
      Value *VecPtr = Builder.CreateBitCast(SI->getPointerOperand(),
                                            VecTy->getPointerTo(AS));
      StoreInst *S = Builder.CreateStore(VecValue, VecPtr);
      S->setAlignment(Alignment);
      E->VectorizedValue = S;
      return propagateMetadata(S, E->Scalars);
    }
    default:
    llvm_unreachable("unknown inst");
  }
  return 0;
}

Value *BoUpSLP::vectorizeTree() {
  Builder.SetInsertPoint(F->getEntryBlock().begin());
  vectorizeTree(&VectorizableTree[0]);

  DEBUG(dbgs() << "SLP: Extracting " << ExternalUses.size() << " values .\n");

  // Extract all of the elements with the external uses.
  for (UserList::iterator it = ExternalUses.begin(), e = ExternalUses.end();
       it != e; ++it) {
    Value *Scalar = it->Scalar;
    llvm::User *User = it->User;

    // Skip users that we already RAUW. This happens when one instruction
    // has multiple uses of the same value.
    if (std::find(Scalar->use_begin(), Scalar->use_end(), User) ==
        Scalar->use_end())
      continue;
    assert(ScalarToTreeEntry.count(Scalar) && "Invalid scalar");

    int Idx = ScalarToTreeEntry[Scalar];
    TreeEntry *E = &VectorizableTree[Idx];
    assert(!E->NeedToGather && "Extracting from a gather list");

    Value *Vec = E->VectorizedValue;
    assert(Vec && "Can't find vectorizable value");

    Value *Lane = Builder.getInt32(it->Lane);
    // Generate extracts for out-of-tree users.
    // Find the insertion point for the extractelement lane.
    if (PHINode *PN = dyn_cast<PHINode>(Vec)) {
      Builder.SetInsertPoint(PN->getParent()->getFirstInsertionPt());
      Value *Ex = Builder.CreateExtractElement(Vec, Lane);
      CSEBlocks.insert(PN->getParent());
      User->replaceUsesOfWith(Scalar, Ex);
    } else if (isa<Instruction>(Vec)){
      if (PHINode *PH = dyn_cast<PHINode>(User)) {
        for (int i = 0, e = PH->getNumIncomingValues(); i != e; ++i) {
          if (PH->getIncomingValue(i) == Scalar) {
            Builder.SetInsertPoint(PH->getIncomingBlock(i)->getTerminator());
            Value *Ex = Builder.CreateExtractElement(Vec, Lane);
            CSEBlocks.insert(PH->getIncomingBlock(i));
            PH->setOperand(i, Ex);
          }
        }
      } else {
        Builder.SetInsertPoint(cast<Instruction>(User));
        Value *Ex = Builder.CreateExtractElement(Vec, Lane);
        CSEBlocks.insert(cast<Instruction>(User)->getParent());
        User->replaceUsesOfWith(Scalar, Ex);
     }
    } else {
      Builder.SetInsertPoint(F->getEntryBlock().begin());
      Value *Ex = Builder.CreateExtractElement(Vec, Lane);
      CSEBlocks.insert(&F->getEntryBlock());
      User->replaceUsesOfWith(Scalar, Ex);
    }

    DEBUG(dbgs() << "SLP: Replaced:" << *User << ".\n");
  }

  // For each vectorized value:
  for (int EIdx = 0, EE = VectorizableTree.size(); EIdx < EE; ++EIdx) {
    TreeEntry *Entry = &VectorizableTree[EIdx];

    // For each lane:
    for (int Lane = 0, LE = Entry->Scalars.size(); Lane != LE; ++Lane) {
      Value *Scalar = Entry->Scalars[Lane];

      // No need to handle users of gathered values.
      if (Entry->NeedToGather)
        continue;

      assert(Entry->VectorizedValue && "Can't find vectorizable value");

      Type *Ty = Scalar->getType();
      if (!Ty->isVoidTy()) {
        for (Value::use_iterator User = Scalar->use_begin(),
             UE = Scalar->use_end(); User != UE; ++User) {
          DEBUG(dbgs() << "SLP: \tvalidating user:" << **User << ".\n");

          assert((ScalarToTreeEntry.count(*User) ||
                  // It is legal to replace the reduction users by undef.
                  (RdxOps && RdxOps->count(*User))) &&
                 "Replacing out-of-tree value with undef");
        }
        Value *Undef = UndefValue::get(Ty);
        Scalar->replaceAllUsesWith(Undef);
      }
      DEBUG(dbgs() << "SLP: \tErasing scalar:" << *Scalar << ".\n");
      cast<Instruction>(Scalar)->eraseFromParent();
    }
  }

  for (Function::iterator it = F->begin(), e = F->end(); it != e; ++it) {
    BlocksNumbers[it].forget();
  }
  Builder.ClearInsertionPoint();

  return VectorizableTree[0].VectorizedValue;
}

class DTCmp {
  const DominatorTree *DT;

public:
  DTCmp(const DominatorTree *DT) : DT(DT) {}
  bool operator()(const BasicBlock *A, const BasicBlock *B) const {
    return DT->properlyDominates(A, B);
  }
};

void BoUpSLP::optimizeGatherSequence() {
  DEBUG(dbgs() << "SLP: Optimizing " << GatherSeq.size()
        << " gather sequences instructions.\n");
  // LICM InsertElementInst sequences.
  for (SetVector<Instruction *>::iterator it = GatherSeq.begin(),
       e = GatherSeq.end(); it != e; ++it) {
    InsertElementInst *Insert = dyn_cast<InsertElementInst>(*it);

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

  // Sort blocks by domination. This ensures we visit a block after all blocks
  // dominating it are visited.
  SmallVector<BasicBlock *, 8> CSEWorkList(CSEBlocks.begin(), CSEBlocks.end());
  std::stable_sort(CSEWorkList.begin(), CSEWorkList.end(), DTCmp(DT));

  // Perform O(N^2) search over the gather sequences and merge identical
  // instructions. TODO: We can further optimize this scan if we split the
  // instructions into different buckets based on the insert lane.
  SmallVector<Instruction *, 16> Visited;
  for (SmallVectorImpl<BasicBlock *>::iterator I = CSEWorkList.begin(),
                                               E = CSEWorkList.end();
       I != E; ++I) {
    assert((I == CSEWorkList.begin() || !DT->dominates(*I, *llvm::prior(I))) &&
           "Worklist not sorted properly!");
    BasicBlock *BB = *I;
    // For all instructions in blocks containing gather sequences:
    for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e;) {
      Instruction *In = it++;
      if (!isa<InsertElementInst>(In) && !isa<ExtractElementInst>(In))
        continue;

      // Check if we can replace this instruction with any of the
      // visited instructions.
      for (SmallVectorImpl<Instruction *>::iterator v = Visited.begin(),
                                                    ve = Visited.end();
           v != ve; ++v) {
        if (In->isIdenticalTo(*v) &&
            DT->dominates((*v)->getParent(), In->getParent())) {
          In->replaceAllUsesWith(*v);
          In->eraseFromParent();
          In = 0;
          break;
        }
      }
      if (In) {
        assert(std::find(Visited.begin(), Visited.end(), In) == Visited.end());
        Visited.push_back(In);
      }
    }
  }
  CSEBlocks.clear();
  GatherSeq.clear();
}

/// The SLPVectorizer Pass.
struct SLPVectorizer : public FunctionPass {
  typedef SmallVector<StoreInst *, 8> StoreList;
  typedef MapVector<Value *, StoreList> StoreListMap;

  /// Pass identification, replacement for typeid
  static char ID;

  explicit SLPVectorizer() : FunctionPass(ID) {
    initializeSLPVectorizerPass(*PassRegistry::getPassRegistry());
  }

  ScalarEvolution *SE;
  DataLayout *DL;
  TargetTransformInfo *TTI;
  AliasAnalysis *AA;
  LoopInfo *LI;
  DominatorTree *DT;

  virtual bool runOnFunction(Function &F) {
    SE = &getAnalysis<ScalarEvolution>();
    DL = getAnalysisIfAvailable<DataLayout>();
    TTI = &getAnalysis<TargetTransformInfo>();
    AA = &getAnalysis<AliasAnalysis>();
    LI = &getAnalysis<LoopInfo>();
    DT = &getAnalysis<DominatorTree>();

    StoreRefs.clear();
    bool Changed = false;

    // If the target claims to have no vector registers don't attempt
    // vectorization.
    if (!TTI->getNumberOfRegisters(true))
      return false;

    // Must have DataLayout. We can't require it because some tests run w/o
    // triple.
    if (!DL)
      return false;

    // Don't vectorize when the attribute NoImplicitFloat is used.
    if (F.hasFnAttribute(Attribute::NoImplicitFloat))
      return false;

    DEBUG(dbgs() << "SLP: Analyzing blocks in " << F.getName() << ".\n");

    // Use the bollom up slp vectorizer to construct chains that start with
    // he store instructions.
    BoUpSLP R(&F, SE, DL, TTI, AA, LI, DT);

    // Scan the blocks in the function in post order.
    for (po_iterator<BasicBlock*> it = po_begin(&F.getEntryBlock()),
         e = po_end(&F.getEntryBlock()); it != e; ++it) {
      BasicBlock *BB = *it;

      // Vectorize trees that end at stores.
      if (unsigned count = collectStores(BB, R)) {
        (void)count;
        DEBUG(dbgs() << "SLP: Found " << count << " stores to vectorize.\n");
        Changed |= vectorizeStoreChains(R);
      }

      // Vectorize trees that end at reductions.
      Changed |= vectorizeChainsInBlock(BB, R);
    }

    if (Changed) {
      R.optimizeGatherSequence();
      DEBUG(dbgs() << "SLP: vectorized \"" << F.getName() << "\"\n");
      DEBUG(verifyFunction(F));
    }
    return Changed;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    FunctionPass::getAnalysisUsage(AU);
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<AliasAnalysis>();
    AU.addRequired<TargetTransformInfo>();
    AU.addRequired<LoopInfo>();
    AU.addRequired<DominatorTree>();
    AU.addPreserved<LoopInfo>();
    AU.addPreserved<DominatorTree>();
    AU.setPreservesCFG();
  }

private:

  /// \brief Collect memory references and sort them according to their base
  /// object. We sort the stores to their base objects to reduce the cost of the
  /// quadratic search on the stores. TODO: We can further reduce this cost
  /// if we flush the chain creation every time we run into a memory barrier.
  unsigned collectStores(BasicBlock *BB, BoUpSLP &R);

  /// \brief Try to vectorize a chain that starts at two arithmetic instrs.
  bool tryToVectorizePair(Value *A, Value *B, BoUpSLP &R);

  /// \brief Try to vectorize a list of operands.
  /// \returns true if a value was vectorized.
  bool tryToVectorizeList(ArrayRef<Value *> VL, BoUpSLP &R);

  /// \brief Try to vectorize a chain that may start at the operands of \V;
  bool tryToVectorize(BinaryOperator *V, BoUpSLP &R);

  /// \brief Vectorize the stores that were collected in StoreRefs.
  bool vectorizeStoreChains(BoUpSLP &R);

  /// \brief Scan the basic block and look for patterns that are likely to start
  /// a vectorization chain.
  bool vectorizeChainsInBlock(BasicBlock *BB, BoUpSLP &R);

  bool vectorizeStoreChain(ArrayRef<Value *> Chain, int CostThreshold,
                           BoUpSLP &R);

  bool vectorizeStores(ArrayRef<StoreInst *> Stores, int costThreshold,
                       BoUpSLP &R);
private:
  StoreListMap StoreRefs;
};

/// \brief Check that the Values in the slice in VL array are still existant in
/// the WeakVH array.
/// Vectorization of part of the VL array may cause later values in the VL array
/// to become invalid. We track when this has happened in the WeakVH array.
static bool hasValueBeenRAUWed(ArrayRef<Value *> &VL,
                               SmallVectorImpl<WeakVH> &VH,
                               unsigned SliceBegin,
                               unsigned SliceSize) {
  for (unsigned i = SliceBegin; i < SliceBegin + SliceSize; ++i)
    if (VH[i] != VL[i])
      return true;

  return false;
}

bool SLPVectorizer::vectorizeStoreChain(ArrayRef<Value *> Chain,
                                          int CostThreshold, BoUpSLP &R) {
  unsigned ChainLen = Chain.size();
  DEBUG(dbgs() << "SLP: Analyzing a store chain of length " << ChainLen
        << "\n");
  Type *StoreTy = cast<StoreInst>(Chain[0])->getValueOperand()->getType();
  unsigned Sz = DL->getTypeSizeInBits(StoreTy);
  unsigned VF = MinVecRegSize / Sz;

  if (!isPowerOf2_32(Sz) || VF < 2)
    return false;

  // Keep track of values that were delete by vectorizing in the loop below.
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

    int Cost = R.getTreeCost();

    DEBUG(dbgs() << "SLP: Found cost=" << Cost << " for VF=" << VF << "\n");
    if (Cost < CostThreshold) {
      DEBUG(dbgs() << "SLP: Decided to vectorize cost=" << Cost << "\n");
      R.vectorizeTree();

      // Move to the next bundle.
      i += VF - 1;
      Changed = true;
    }
  }

  return Changed;
}

bool SLPVectorizer::vectorizeStores(ArrayRef<StoreInst *> Stores,
                                    int costThreshold, BoUpSLP &R) {
  SetVector<Value *> Heads, Tails;
  SmallDenseMap<Value *, Value *> ConsecutiveChain;

  // We may run into multiple chains that merge into a single chain. We mark the
  // stores that we vectorized so that we don't visit the same store twice.
  BoUpSLP::ValueSet VectorizedStores;
  bool Changed = false;

  // Do a quadratic search on all of the given stores and find
  // all of the pairs of stores that follow each other.
  for (unsigned i = 0, e = Stores.size(); i < e; ++i) {
    for (unsigned j = 0; j < e; ++j) {
      if (i == j)
        continue;

      if (R.isConsecutiveAccess(Stores[i], Stores[j])) {
        Tails.insert(Stores[j]);
        Heads.insert(Stores[i]);
        ConsecutiveChain[Stores[i]] = Stores[j];
      }
    }
  }

  // For stores that start but don't end a link in the chain:
  for (SetVector<Value *>::iterator it = Heads.begin(), e = Heads.end();
       it != e; ++it) {
    if (Tails.count(*it))
      continue;

    // We found a store instr that starts a chain. Now follow the chain and try
    // to vectorize it.
    BoUpSLP::ValueList Operands;
    Value *I = *it;
    // Collect the chain into a list.
    while (Tails.count(I) || Heads.count(I)) {
      if (VectorizedStores.count(I))
        break;
      Operands.push_back(I);
      // Move to the next value in the chain.
      I = ConsecutiveChain[I];
    }

    bool Vectorized = vectorizeStoreChain(Operands, costThreshold, R);

    // Mark the vectorized stores so that we don't vectorize them again.
    if (Vectorized)
      VectorizedStores.insert(Operands.begin(), Operands.end());
    Changed |= Vectorized;
  }

  return Changed;
}


unsigned SLPVectorizer::collectStores(BasicBlock *BB, BoUpSLP &R) {
  unsigned count = 0;
  StoreRefs.clear();
  for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; ++it) {
    StoreInst *SI = dyn_cast<StoreInst>(it);
    if (!SI)
      continue;

    // Don't touch volatile stores.
    if (!SI->isSimple())
      continue;

    // Check that the pointer points to scalars.
    Type *Ty = SI->getValueOperand()->getType();
    if (Ty->isAggregateType() || Ty->isVectorTy())
      return 0;

    // Find the base pointer.
    Value *Ptr = GetUnderlyingObject(SI->getPointerOperand(), DL);

    // Save the store locations.
    StoreRefs[Ptr].push_back(SI);
    count++;
  }
  return count;
}

bool SLPVectorizer::tryToVectorizePair(Value *A, Value *B, BoUpSLP &R) {
  if (!A || !B)
    return false;
  Value *VL[] = { A, B };
  return tryToVectorizeList(VL, R);
}

bool SLPVectorizer::tryToVectorizeList(ArrayRef<Value *> VL, BoUpSLP &R) {
  if (VL.size() < 2)
    return false;

  DEBUG(dbgs() << "SLP: Vectorizing a list of length = " << VL.size() << ".\n");

  // Check that all of the parts are scalar instructions of the same type.
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  if (!I0)
    return false;

  unsigned Opcode0 = I0->getOpcode();

  Type *Ty0 = I0->getType();
  unsigned Sz = DL->getTypeSizeInBits(Ty0);
  unsigned VF = MinVecRegSize / Sz;

  for (int i = 0, e = VL.size(); i < e; ++i) {
    Type *Ty = VL[i]->getType();
    if (Ty->isAggregateType() || Ty->isVectorTy())
      return false;
    Instruction *Inst = dyn_cast<Instruction>(VL[i]);
    if (!Inst || Inst->getOpcode() != Opcode0)
      return false;
  }

  bool Changed = false;

  // Keep track of values that were delete by vectorizing in the loop below.
  SmallVector<WeakVH, 8> TrackValues(VL.begin(), VL.end());

  for (unsigned i = 0, e = VL.size(); i < e; ++i) {
    unsigned OpsWidth = 0;

    if (i + VF > e)
      OpsWidth = e - i;
    else
      OpsWidth = VF;

    if (!isPowerOf2_32(OpsWidth) || OpsWidth < 2)
      break;

    // Check that a previous iteration of this loop did not delete the Value.
    if (hasValueBeenRAUWed(VL, TrackValues, i, OpsWidth))
      continue;

    DEBUG(dbgs() << "SLP: Analyzing " << OpsWidth << " operations "
                 << "\n");
    ArrayRef<Value *> Ops = VL.slice(i, OpsWidth);

    R.buildTree(Ops);
    int Cost = R.getTreeCost();

    if (Cost < -SLPCostThreshold) {
      DEBUG(dbgs() << "SLP: Vectorizing pair at cost:" << Cost << ".\n");
      R.vectorizeTree();

      // Move to the next bundle.
      i += VF - 1;
      Changed = true;
    }
  }

  return Changed;
}

bool SLPVectorizer::tryToVectorize(BinaryOperator *V, BoUpSLP &R) {
  if (!V)
    return false;

  // Try to vectorize V.
  if (tryToVectorizePair(V->getOperand(0), V->getOperand(1), R))
    return true;

  BinaryOperator *A = dyn_cast<BinaryOperator>(V->getOperand(0));
  BinaryOperator *B = dyn_cast<BinaryOperator>(V->getOperand(1));
  // Try to skip B.
  if (B && B->hasOneUse()) {
    BinaryOperator *B0 = dyn_cast<BinaryOperator>(B->getOperand(0));
    BinaryOperator *B1 = dyn_cast<BinaryOperator>(B->getOperand(1));
    if (tryToVectorizePair(A, B0, R)) {
      B->moveBefore(V);
      return true;
    }
    if (tryToVectorizePair(A, B1, R)) {
      B->moveBefore(V);
      return true;
    }
  }

  // Try to skip A.
  if (A && A->hasOneUse()) {
    BinaryOperator *A0 = dyn_cast<BinaryOperator>(A->getOperand(0));
    BinaryOperator *A1 = dyn_cast<BinaryOperator>(A->getOperand(1));
    if (tryToVectorizePair(A0, B, R)) {
      A->moveBefore(V);
      return true;
    }
    if (tryToVectorizePair(A1, B, R)) {
      A->moveBefore(V);
      return true;
    }
  }
  return 0;
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
  SmallPtrSet<Value *, 16> ReductionOps;
  SmallVector<Value *, 32> ReducedVals;

  BinaryOperator *ReductionRoot;
  PHINode *ReductionPHI;

  /// The opcode of the reduction.
  unsigned ReductionOpcode;
  /// The opcode of the values we perform a reduction on.
  unsigned ReducedValueOpcode;
  /// The width of one full horizontal reduction operation.
  unsigned ReduxWidth;
  /// Should we model this reduction as a pairwise reduction tree or a tree that
  /// splits the vector in halves and adds those halves.
  bool IsPairwiseReduction;

public:
  HorizontalReduction()
    : ReductionRoot(0), ReductionPHI(0), ReductionOpcode(0),
    ReducedValueOpcode(0), ReduxWidth(0), IsPairwiseReduction(false) {}

  /// \brief Try to find a reduction tree.
  bool matchAssociativeReduction(PHINode *Phi, BinaryOperator *B,
                                 DataLayout *DL) {
    assert((!Phi ||
            std::find(Phi->op_begin(), Phi->op_end(), B) != Phi->op_end()) &&
           "Thi phi needs to use the binary operator");

    // We could have a initial reductions that is not an add.
    //  r *= v1 + v2 + v3 + v4
    // In such a case start looking for a tree rooted in the first '+'.
    if (Phi) {
      if (B->getOperand(0) == Phi) {
        Phi = 0;
        B = dyn_cast<BinaryOperator>(B->getOperand(1));
      } else if (B->getOperand(1) == Phi) {
        Phi = 0;
        B = dyn_cast<BinaryOperator>(B->getOperand(0));
      }
    }

    if (!B)
      return false;

    Type *Ty = B->getType();
    if (Ty->isVectorTy())
      return false;

    ReductionOpcode = B->getOpcode();
    ReducedValueOpcode = 0;
    ReduxWidth = MinVecRegSize / DL->getTypeSizeInBits(Ty);
    ReductionRoot = B;
    ReductionPHI = Phi;

    if (ReduxWidth < 4)
      return false;

    // We currently only support adds.
    if (ReductionOpcode != Instruction::Add &&
        ReductionOpcode != Instruction::FAdd)
      return false;

    // Post order traverse the reduction tree starting at B. We only handle true
    // trees containing only binary operators.
    SmallVector<std::pair<BinaryOperator *, unsigned>, 32> Stack;
    Stack.push_back(std::make_pair(B, 0));
    while (!Stack.empty()) {
      BinaryOperator *TreeN = Stack.back().first;
      unsigned EdgeToVist = Stack.back().second++;
      bool IsReducedValue = TreeN->getOpcode() != ReductionOpcode;

      // Only handle trees in the current basic block.
      if (TreeN->getParent() != B->getParent())
        return false;

      // Each tree node needs to have one user except for the ultimate
      // reduction.
      if (!TreeN->hasOneUse() && TreeN != B)
        return false;

      // Postorder vist.
      if (EdgeToVist == 2 || IsReducedValue) {
        if (IsReducedValue) {
          // Make sure that the opcodes of the operations that we are going to
          // reduce match.
          if (!ReducedValueOpcode)
            ReducedValueOpcode = TreeN->getOpcode();
          else if (ReducedValueOpcode != TreeN->getOpcode())
            return false;
          ReducedVals.push_back(TreeN);
        } else {
          // We need to be able to reassociate the adds.
          if (!TreeN->isAssociative())
            return false;
          ReductionOps.insert(TreeN);
        }
        // Retract.
        Stack.pop_back();
        continue;
      }

      // Visit left or right.
      Value *NextV = TreeN->getOperand(EdgeToVist);
      BinaryOperator *Next = dyn_cast<BinaryOperator>(NextV);
      if (Next)
        Stack.push_back(std::make_pair(Next, 0));
      else if (NextV != Phi)
        return false;
    }
    return true;
  }

  /// \brief Attempt to vectorize the tree found by
  /// matchAssociativeReduction.
  bool tryToReduce(BoUpSLP &V, TargetTransformInfo *TTI) {
    if (ReducedVals.empty())
      return false;

    unsigned NumReducedVals = ReducedVals.size();
    if (NumReducedVals < ReduxWidth)
      return false;

    Value *VectorizedTree = 0;
    IRBuilder<> Builder(ReductionRoot);
    FastMathFlags Unsafe;
    Unsafe.setUnsafeAlgebra();
    Builder.SetFastMathFlags(Unsafe);
    unsigned i = 0;

    for (; i < NumReducedVals - ReduxWidth + 1; i += ReduxWidth) {
      ArrayRef<Value *> ValsToReduce(&ReducedVals[i], ReduxWidth);
      V.buildTree(ValsToReduce, &ReductionOps);

      // Estimate cost.
      int Cost = V.getTreeCost() + getReductionCost(TTI, ReducedVals[i]);
      if (Cost >= -SLPCostThreshold)
        break;

      DEBUG(dbgs() << "SLP: Vectorizing horizontal reduction at cost:" << Cost
                   << ". (HorRdx)\n");

      // Vectorize a tree.
      DebugLoc Loc = cast<Instruction>(ReducedVals[i])->getDebugLoc();
      Value *VectorizedRoot = V.vectorizeTree();

      // Emit a reduction.
      Value *ReducedSubTree = emitReduction(VectorizedRoot, Builder);
      if (VectorizedTree) {
        Builder.SetCurrentDebugLocation(Loc);
        VectorizedTree = createBinOp(Builder, ReductionOpcode, VectorizedTree,
                                     ReducedSubTree, "bin.rdx");
      } else
        VectorizedTree = ReducedSubTree;
    }

    if (VectorizedTree) {
      // Finish the reduction.
      for (; i < NumReducedVals; ++i) {
        Builder.SetCurrentDebugLocation(
          cast<Instruction>(ReducedVals[i])->getDebugLoc());
        VectorizedTree = createBinOp(Builder, ReductionOpcode, VectorizedTree,
                                     ReducedVals[i]);
      }
      // Update users.
      if (ReductionPHI) {
        assert(ReductionRoot != NULL && "Need a reduction operation");
        ReductionRoot->setOperand(0, VectorizedTree);
        ReductionRoot->setOperand(1, ReductionPHI);
      } else
        ReductionRoot->replaceAllUsesWith(VectorizedTree);
    }
    return VectorizedTree != 0;
  }

private:

  /// \brief Calcuate the cost of a reduction.
  int getReductionCost(TargetTransformInfo *TTI, Value *FirstReducedVal) {
    Type *ScalarTy = FirstReducedVal->getType();
    Type *VecTy = VectorType::get(ScalarTy, ReduxWidth);

    int PairwiseRdxCost = TTI->getReductionCost(ReductionOpcode, VecTy, true);
    int SplittingRdxCost = TTI->getReductionCost(ReductionOpcode, VecTy, false);

    IsPairwiseReduction = PairwiseRdxCost < SplittingRdxCost;
    int VecReduxCost = IsPairwiseReduction ? PairwiseRdxCost : SplittingRdxCost;

    int ScalarReduxCost =
        ReduxWidth * TTI->getArithmeticInstrCost(ReductionOpcode, VecTy);

    DEBUG(dbgs() << "SLP: Adding cost " << VecReduxCost - ScalarReduxCost
                 << " for reduction that starts with " << *FirstReducedVal
                 << " (It is a "
                 << (IsPairwiseReduction ? "pairwise" : "splitting")
                 << " reduction)\n");

    return VecReduxCost - ScalarReduxCost;
  }

  static Value *createBinOp(IRBuilder<> &Builder, unsigned Opcode, Value *L,
                            Value *R, const Twine &Name = "") {
    if (Opcode == Instruction::FAdd)
      return Builder.CreateFAdd(L, R, Name);
    return Builder.CreateBinOp((Instruction::BinaryOps)Opcode, L, R, Name);
  }

  /// \brief Emit a horizontal reduction of the vectorized value.
  Value *emitReduction(Value *VectorizedValue, IRBuilder<> &Builder) {
    assert(VectorizedValue && "Need to have a vectorized tree node");
    Instruction *ValToReduce = dyn_cast<Instruction>(VectorizedValue);
    assert(isPowerOf2_32(ReduxWidth) &&
           "We only handle power-of-two reductions for now");

    Value *TmpVec = ValToReduce;
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
        TmpVec = createBinOp(Builder, ReductionOpcode, LeftShuf, RightShuf,
                             "bin.rdx");
      } else {
        Value *UpperHalf =
          createRdxShuffleMask(ReduxWidth, i, false, false, Builder);
        Value *Shuf = Builder.CreateShuffleVector(
          TmpVec, UndefValue::get(TmpVec->getType()), UpperHalf, "rdx.shuf");
        TmpVec = createBinOp(Builder, ReductionOpcode, TmpVec, Shuf, "bin.rdx");
      }
    }

    // The result is in the first element of the vector.
    return Builder.CreateExtractElement(TmpVec, Builder.getInt32(0));
  }
};

/// \brief Recognize construction of vectors like
///  %ra = insertelement <4 x float> undef, float %s0, i32 0
///  %rb = insertelement <4 x float> %ra, float %s1, i32 1
///  %rc = insertelement <4 x float> %rb, float %s2, i32 2
///  %rd = insertelement <4 x float> %rc, float %s3, i32 3
///
/// Returns true if it matches
///
static bool findBuildVector(InsertElementInst *IE,
                            SmallVectorImpl<Value *> &Ops) {
  if (!isa<UndefValue>(IE->getOperand(0)))
    return false;

  while (true) {
    Ops.push_back(IE->getOperand(1));

    if (IE->use_empty())
      return false;

    InsertElementInst *NextUse = dyn_cast<InsertElementInst>(IE->use_back());
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

static bool PhiTypeSorterFunc(Value *V, Value *V2) {
  return V->getType() < V2->getType();
}

bool SLPVectorizer::vectorizeChainsInBlock(BasicBlock *BB, BoUpSLP &R) {
  bool Changed = false;
  SmallVector<Value *, 4> Incoming;
  SmallSet<Value *, 16> VisitedInstrs;

  bool HaveVectorizedPhiNodes = true;
  while (HaveVectorizedPhiNodes) {
    HaveVectorizedPhiNodes = false;

    // Collect the incoming values from the PHIs.
    Incoming.clear();
    for (BasicBlock::iterator instr = BB->begin(), ie = BB->end(); instr != ie;
         ++instr) {
      PHINode *P = dyn_cast<PHINode>(instr);
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
      if (NumElts > 1 &&
          tryToVectorizeList(ArrayRef<Value *>(IncIt, NumElts), R)) {
        // Success start over because instructions might have been changed.
        HaveVectorizedPhiNodes = true;
        Changed = true;
        break;
      }

      // Start over at the next instruction of a differnt type (or the end).
      IncIt = SameTypeIt;
    }
  }

  VisitedInstrs.clear();

  for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; it++) {
    // We may go through BB multiple times so skip the one we have checked.
    if (!VisitedInstrs.insert(it))
      continue;

    if (isa<DbgInfoIntrinsic>(it))
      continue;

    // Try to vectorize reductions that use PHINodes.
    if (PHINode *P = dyn_cast<PHINode>(it)) {
      // Check that the PHI is a reduction PHI.
      if (P->getNumIncomingValues() != 2)
        return Changed;
      Value *Rdx =
          (P->getIncomingBlock(0) == BB
               ? (P->getIncomingValue(0))
               : (P->getIncomingBlock(1) == BB ? P->getIncomingValue(1) : 0));
      // Check if this is a Binary Operator.
      BinaryOperator *BI = dyn_cast_or_null<BinaryOperator>(Rdx);
      if (!BI)
        continue;

      // Try to match and vectorize a horizontal reduction.
      HorizontalReduction HorRdx;
      if (ShouldVectorizeHor &&
          HorRdx.matchAssociativeReduction(P, BI, DL) &&
          HorRdx.tryToReduce(R, TTI)) {
        Changed = true;
        it = BB->begin();
        e = BB->end();
        continue;
      }

     Value *Inst = BI->getOperand(0);
      if (Inst == P)
        Inst = BI->getOperand(1);

      if (tryToVectorize(dyn_cast<BinaryOperator>(Inst), R)) {
        // We would like to start over since some instructions are deleted
        // and the iterator may become invalid value.
        Changed = true;
        it = BB->begin();
        e = BB->end();
        continue;
      }

      continue;
    }

    // Try to vectorize horizontal reductions feeding into a store.
    if (ShouldStartVectorizeHorAtStore)
      if (StoreInst *SI = dyn_cast<StoreInst>(it))
        if (BinaryOperator *BinOp =
                dyn_cast<BinaryOperator>(SI->getValueOperand())) {
          HorizontalReduction HorRdx;
          if (((HorRdx.matchAssociativeReduction(0, BinOp, DL) &&
                HorRdx.tryToReduce(R, TTI)) ||
               tryToVectorize(BinOp, R))) {
            Changed = true;
            it = BB->begin();
            e = BB->end();
            continue;
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

      for (int i = 0; i < 2; ++i) {
         if (BinaryOperator *BI = dyn_cast<BinaryOperator>(CI->getOperand(i))) {
            if (tryToVectorizePair(BI->getOperand(0), BI->getOperand(1), R)) {
              Changed = true;
              // We would like to start over since some instructions are deleted
              // and the iterator may become invalid value.
              it = BB->begin();
              e = BB->end();
            }
         }
      }
      continue;
    }

    // Try to vectorize trees that start at insertelement instructions.
    if (InsertElementInst *IE = dyn_cast<InsertElementInst>(it)) {
      SmallVector<Value *, 8> Ops;
      if (!findBuildVector(IE, Ops))
        continue;

      if (tryToVectorizeList(Ops, R)) {
        Changed = true;
        it = BB->begin();
        e = BB->end();
      }

      continue;
    }
  }

  return Changed;
}

bool SLPVectorizer::vectorizeStoreChains(BoUpSLP &R) {
  bool Changed = false;
  // Attempt to sort and vectorize each of the store-groups.
  for (StoreListMap::iterator it = StoreRefs.begin(), e = StoreRefs.end();
       it != e; ++it) {
    if (it->second.size() < 2)
      continue;

    DEBUG(dbgs() << "SLP: Analyzing a store chain of length "
          << it->second.size() << ".\n");

    // Process the stores in chunks of 16.
    for (unsigned CI = 0, CE = it->second.size(); CI < CE; CI+=16) {
      unsigned Len = std::min<unsigned>(CE - CI, 16);
      ArrayRef<StoreInst *> Chunk(&it->second[CI], Len);
      Changed |= vectorizeStores(Chunk, -SLPCostThreshold, R);
    }
  }
  return Changed;
}

} // end anonymous namespace

char SLPVectorizer::ID = 0;
static const char lv_name[] = "SLP Vectorizer";
INITIALIZE_PASS_BEGIN(SLPVectorizer, SV_NAME, lv_name, false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_AG_DEPENDENCY(TargetTransformInfo)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_END(SLPVectorizer, SV_NAME, lv_name, false, false)

namespace llvm {
Pass *createSLPVectorizerPass() { return new SLPVectorizer(); }
}
