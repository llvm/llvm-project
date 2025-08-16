//===- VPlan.h - Represent A Vectorizer Plan --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains the declarations of the Vectorization Plan base classes:
/// 1. VPBasicBlock and VPRegionBlock that inherit from a common pure virtual
///    VPBlockBase, together implementing a Hierarchical CFG;
/// 2. Pure virtual VPRecipeBase serving as the base class for recipes contained
///    within VPBasicBlocks;
/// 3. Pure virtual VPSingleDefRecipe serving as a base class for recipes that
///    also inherit from VPValue.
/// 4. VPInstruction, a concrete Recipe and VPUser modeling a single planned
///    instruction;
/// 5. The VPlan class holding a candidate for vectorization;
/// These are documented in docs/VectorizationPlan.rst.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLAN_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLAN_H

#include "VPlanAnalysis.h"
#include "VPlanValue.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InstructionCost.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <string>

namespace llvm {

class BasicBlock;
class DominatorTree;
class InnerLoopVectorizer;
class IRBuilderBase;
struct VPTransformState;
class raw_ostream;
class RecurrenceDescriptor;
class SCEV;
class Type;
class VPBasicBlock;
class VPBuilder;
class VPDominatorTree;
class VPRegionBlock;
class VPlan;
class VPLane;
class VPReplicateRecipe;
class VPlanSlp;
class Value;
class LoopVectorizationCostModel;
class LoopVersioning;

struct VPCostContext;

namespace Intrinsic {
typedef unsigned ID;
}

using VPlanPtr = std::unique_ptr<VPlan>;

/// VPBlockBase is the building block of the Hierarchical Control-Flow Graph.
/// A VPBlockBase can be either a VPBasicBlock or a VPRegionBlock.
class LLVM_ABI_FOR_TEST VPBlockBase {
  friend class VPBlockUtils;

  const unsigned char SubclassID; ///< Subclass identifier (for isa/dyn_cast).

  /// An optional name for the block.
  std::string Name;

  /// The immediate VPRegionBlock which this VPBlockBase belongs to, or null if
  /// it is a topmost VPBlockBase.
  VPRegionBlock *Parent = nullptr;

  /// List of predecessor blocks.
  SmallVector<VPBlockBase *, 1> Predecessors;

  /// List of successor blocks.
  SmallVector<VPBlockBase *, 1> Successors;

  /// VPlan containing the block. Can only be set on the entry block of the
  /// plan.
  VPlan *Plan = nullptr;

  /// Add \p Successor as the last successor to this block.
  void appendSuccessor(VPBlockBase *Successor) {
    assert(Successor && "Cannot add nullptr successor!");
    Successors.push_back(Successor);
  }

  /// Add \p Predecessor as the last predecessor to this block.
  void appendPredecessor(VPBlockBase *Predecessor) {
    assert(Predecessor && "Cannot add nullptr predecessor!");
    Predecessors.push_back(Predecessor);
  }

  /// Remove \p Predecessor from the predecessors of this block.
  void removePredecessor(VPBlockBase *Predecessor) {
    auto Pos = find(Predecessors, Predecessor);
    assert(Pos && "Predecessor does not exist");
    Predecessors.erase(Pos);
  }

  /// Remove \p Successor from the successors of this block.
  void removeSuccessor(VPBlockBase *Successor) {
    auto Pos = find(Successors, Successor);
    assert(Pos && "Successor does not exist");
    Successors.erase(Pos);
  }

  /// This function replaces one predecessor with another, useful when
  /// trying to replace an old block in the CFG with a new one.
  void replacePredecessor(VPBlockBase *Old, VPBlockBase *New) {
    auto I = find(Predecessors, Old);
    assert(I != Predecessors.end());
    assert(Old->getParent() == New->getParent() &&
           "replaced predecessor must have the same parent");
    *I = New;
  }

  /// This function replaces one successor with another, useful when
  /// trying to replace an old block in the CFG with a new one.
  void replaceSuccessor(VPBlockBase *Old, VPBlockBase *New) {
    auto I = find(Successors, Old);
    assert(I != Successors.end());
    assert(Old->getParent() == New->getParent() &&
           "replaced successor must have the same parent");
    *I = New;
  }

protected:
  VPBlockBase(const unsigned char SC, const std::string &N)
      : SubclassID(SC), Name(N) {}

public:
  /// An enumeration for keeping track of the concrete subclass of VPBlockBase
  /// that are actually instantiated. Values of this enumeration are kept in the
  /// SubclassID field of the VPBlockBase objects. They are used for concrete
  /// type identification.
  using VPBlockTy = enum { VPRegionBlockSC, VPBasicBlockSC, VPIRBasicBlockSC };

  using VPBlocksTy = SmallVectorImpl<VPBlockBase *>;

  virtual ~VPBlockBase() = default;

  const std::string &getName() const { return Name; }

  void setName(const Twine &newName) { Name = newName.str(); }

  /// \return an ID for the concrete type of this object.
  /// This is used to implement the classof checks. This should not be used
  /// for any other purpose, as the values may change as LLVM evolves.
  unsigned getVPBlockID() const { return SubclassID; }

  VPRegionBlock *getParent() { return Parent; }
  const VPRegionBlock *getParent() const { return Parent; }

  /// \return A pointer to the plan containing the current block.
  VPlan *getPlan();
  const VPlan *getPlan() const;

  /// Sets the pointer of the plan containing the block. The block must be the
  /// entry block into the VPlan.
  void setPlan(VPlan *ParentPlan);

  void setParent(VPRegionBlock *P) { Parent = P; }

  /// \return the VPBasicBlock that is the entry of this VPBlockBase,
  /// recursively, if the latter is a VPRegionBlock. Otherwise, if this
  /// VPBlockBase is a VPBasicBlock, it is returned.
  const VPBasicBlock *getEntryBasicBlock() const;
  VPBasicBlock *getEntryBasicBlock();

  /// \return the VPBasicBlock that is the exiting this VPBlockBase,
  /// recursively, if the latter is a VPRegionBlock. Otherwise, if this
  /// VPBlockBase is a VPBasicBlock, it is returned.
  const VPBasicBlock *getExitingBasicBlock() const;
  VPBasicBlock *getExitingBasicBlock();

  const VPBlocksTy &getSuccessors() const { return Successors; }
  VPBlocksTy &getSuccessors() { return Successors; }

  iterator_range<VPBlockBase **> successors() { return Successors; }
  iterator_range<VPBlockBase **> predecessors() { return Predecessors; }

  const VPBlocksTy &getPredecessors() const { return Predecessors; }
  VPBlocksTy &getPredecessors() { return Predecessors; }

  /// \return the successor of this VPBlockBase if it has a single successor.
  /// Otherwise return a null pointer.
  VPBlockBase *getSingleSuccessor() const {
    return (Successors.size() == 1 ? *Successors.begin() : nullptr);
  }

  /// \return the predecessor of this VPBlockBase if it has a single
  /// predecessor. Otherwise return a null pointer.
  VPBlockBase *getSinglePredecessor() const {
    return (Predecessors.size() == 1 ? *Predecessors.begin() : nullptr);
  }

  size_t getNumSuccessors() const { return Successors.size(); }
  size_t getNumPredecessors() const { return Predecessors.size(); }

  /// An Enclosing Block of a block B is any block containing B, including B
  /// itself. \return the closest enclosing block starting from "this", which
  /// has successors. \return the root enclosing block if all enclosing blocks
  /// have no successors.
  VPBlockBase *getEnclosingBlockWithSuccessors();

  /// \return the closest enclosing block starting from "this", which has
  /// predecessors. \return the root enclosing block if all enclosing blocks
  /// have no predecessors.
  VPBlockBase *getEnclosingBlockWithPredecessors();

  /// \return the successors either attached directly to this VPBlockBase or, if
  /// this VPBlockBase is the exit block of a VPRegionBlock and has no
  /// successors of its own, search recursively for the first enclosing
  /// VPRegionBlock that has successors and return them. If no such
  /// VPRegionBlock exists, return the (empty) successors of the topmost
  /// VPBlockBase reached.
  const VPBlocksTy &getHierarchicalSuccessors() {
    return getEnclosingBlockWithSuccessors()->getSuccessors();
  }

  /// \return the hierarchical successor of this VPBlockBase if it has a single
  /// hierarchical successor. Otherwise return a null pointer.
  VPBlockBase *getSingleHierarchicalSuccessor() {
    return getEnclosingBlockWithSuccessors()->getSingleSuccessor();
  }

  /// \return the predecessors either attached directly to this VPBlockBase or,
  /// if this VPBlockBase is the entry block of a VPRegionBlock and has no
  /// predecessors of its own, search recursively for the first enclosing
  /// VPRegionBlock that has predecessors and return them. If no such
  /// VPRegionBlock exists, return the (empty) predecessors of the topmost
  /// VPBlockBase reached.
  const VPBlocksTy &getHierarchicalPredecessors() {
    return getEnclosingBlockWithPredecessors()->getPredecessors();
  }

  /// \return the hierarchical predecessor of this VPBlockBase if it has a
  /// single hierarchical predecessor. Otherwise return a null pointer.
  VPBlockBase *getSingleHierarchicalPredecessor() {
    return getEnclosingBlockWithPredecessors()->getSinglePredecessor();
  }

  /// Set a given VPBlockBase \p Successor as the single successor of this
  /// VPBlockBase. This VPBlockBase is not added as predecessor of \p Successor.
  /// This VPBlockBase must have no successors.
  void setOneSuccessor(VPBlockBase *Successor) {
    assert(Successors.empty() && "Setting one successor when others exist.");
    assert(Successor->getParent() == getParent() &&
           "connected blocks must have the same parent");
    appendSuccessor(Successor);
  }

  /// Set two given VPBlockBases \p IfTrue and \p IfFalse to be the two
  /// successors of this VPBlockBase. This VPBlockBase is not added as
  /// predecessor of \p IfTrue or \p IfFalse. This VPBlockBase must have no
  /// successors.
  void setTwoSuccessors(VPBlockBase *IfTrue, VPBlockBase *IfFalse) {
    assert(Successors.empty() && "Setting two successors when others exist.");
    appendSuccessor(IfTrue);
    appendSuccessor(IfFalse);
  }

  /// Set each VPBasicBlock in \p NewPreds as predecessor of this VPBlockBase.
  /// This VPBlockBase must have no predecessors. This VPBlockBase is not added
  /// as successor of any VPBasicBlock in \p NewPreds.
  void setPredecessors(ArrayRef<VPBlockBase *> NewPreds) {
    assert(Predecessors.empty() && "Block predecessors already set.");
    for (auto *Pred : NewPreds)
      appendPredecessor(Pred);
  }

  /// Set each VPBasicBlock in \p NewSuccss as successor of this VPBlockBase.
  /// This VPBlockBase must have no successors. This VPBlockBase is not added
  /// as predecessor of any VPBasicBlock in \p NewSuccs.
  void setSuccessors(ArrayRef<VPBlockBase *> NewSuccs) {
    assert(Successors.empty() && "Block successors already set.");
    for (auto *Succ : NewSuccs)
      appendSuccessor(Succ);
  }

  /// Remove all the predecessor of this block.
  void clearPredecessors() { Predecessors.clear(); }

  /// Remove all the successors of this block.
  void clearSuccessors() { Successors.clear(); }

  /// Swap predecessors of the block. The block must have exactly 2
  /// predecessors.
  void swapPredecessors() {
    assert(Predecessors.size() == 2 && "must have 2 predecessors to swap");
    std::swap(Predecessors[0], Predecessors[1]);
  }

  /// Swap successors of the block. The block must have exactly 2 successors.
  // TODO: This should be part of introducing conditional branch recipes rather
  // than being independent.
  void swapSuccessors() {
    assert(Successors.size() == 2 && "must have 2 successors to swap");
    std::swap(Successors[0], Successors[1]);
  }

  /// Returns the index for \p Pred in the blocks predecessors list.
  unsigned getIndexForPredecessor(const VPBlockBase *Pred) const {
    assert(count(Predecessors, Pred) == 1 &&
           "must have Pred exactly once in Predecessors");
    return std::distance(Predecessors.begin(), find(Predecessors, Pred));
  }

  /// Returns the index for \p Succ in the blocks successor list.
  unsigned getIndexForSuccessor(const VPBlockBase *Succ) const {
    assert(count(Successors, Succ) == 1 &&
           "must have Succ exactly once in Successors");
    return std::distance(Successors.begin(), find(Successors, Succ));
  }

  /// The method which generates the output IR that correspond to this
  /// VPBlockBase, thereby "executing" the VPlan.
  virtual void execute(VPTransformState *State) = 0;

  /// Return the cost of the block.
  virtual InstructionCost cost(ElementCount VF, VPCostContext &Ctx) = 0;

  /// Return true if it is legal to hoist instructions into this block.
  bool isLegalToHoistInto() {
    // There are currently no constraints that prevent an instruction to be
    // hoisted into a VPBlockBase.
    return true;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void printAsOperand(raw_ostream &OS, bool PrintType = false) const {
    OS << getName();
  }

  /// Print plain-text dump of this VPBlockBase to \p O, prefixing all lines
  /// with \p Indent. \p SlotTracker is used to print unnamed VPValue's using
  /// consequtive numbers.
  ///
  /// Note that the numbering is applied to the whole VPlan, so printing
  /// individual blocks is consistent with the whole VPlan printing.
  virtual void print(raw_ostream &O, const Twine &Indent,
                     VPSlotTracker &SlotTracker) const = 0;

  /// Print plain-text dump of this VPlan to \p O.
  void print(raw_ostream &O) const;

  /// Print the successors of this block to \p O, prefixing all lines with \p
  /// Indent.
  void printSuccessors(raw_ostream &O, const Twine &Indent) const;

  /// Dump this VPBlockBase to dbgs().
  LLVM_DUMP_METHOD void dump() const { print(dbgs()); }
#endif

  /// Clone the current block and it's recipes without updating the operands of
  /// the cloned recipes, including all blocks in the single-entry single-exit
  /// region for VPRegionBlocks.
  virtual VPBlockBase *clone() = 0;
};

/// VPRecipeBase is a base class modeling a sequence of one or more output IR
/// instructions. VPRecipeBase owns the VPValues it defines through VPDef
/// and is responsible for deleting its defined values. Single-value
/// recipes must inherit from VPSingleDef instead of inheriting from both
/// VPRecipeBase and VPValue separately.
class LLVM_ABI_FOR_TEST VPRecipeBase
    : public ilist_node_with_parent<VPRecipeBase, VPBasicBlock>,
      public VPDef,
      public VPUser {
  friend VPBasicBlock;
  friend class VPBlockUtils;

  /// Each VPRecipe belongs to a single VPBasicBlock.
  VPBasicBlock *Parent = nullptr;

  /// The debug location for the recipe.
  DebugLoc DL;

public:
  VPRecipeBase(const unsigned char SC, ArrayRef<VPValue *> Operands,
               DebugLoc DL = {})
      : VPDef(SC), VPUser(Operands), DL(DL) {}

  virtual ~VPRecipeBase() = default;

  /// Clone the current recipe.
  virtual VPRecipeBase *clone() = 0;

  /// \return the VPBasicBlock which this VPRecipe belongs to.
  VPBasicBlock *getParent() { return Parent; }
  const VPBasicBlock *getParent() const { return Parent; }

  /// The method which generates the output IR instructions that correspond to
  /// this VPRecipe, thereby "executing" the VPlan.
  virtual void execute(VPTransformState &State) = 0;

  /// Return the cost of this recipe, taking into account if the cost
  /// computation should be skipped and the ForceTargetInstructionCost flag.
  /// Also takes care of printing the cost for debugging.
  InstructionCost cost(ElementCount VF, VPCostContext &Ctx);

  /// Insert an unlinked recipe into a basic block immediately before
  /// the specified recipe.
  void insertBefore(VPRecipeBase *InsertPos);
  /// Insert an unlinked recipe into \p BB immediately before the insertion
  /// point \p IP;
  void insertBefore(VPBasicBlock &BB, iplist<VPRecipeBase>::iterator IP);

  /// Insert an unlinked Recipe into a basic block immediately after
  /// the specified Recipe.
  void insertAfter(VPRecipeBase *InsertPos);

  /// Unlink this recipe from its current VPBasicBlock and insert it into
  /// the VPBasicBlock that MovePos lives in, right after MovePos.
  void moveAfter(VPRecipeBase *MovePos);

  /// Unlink this recipe and insert into BB before I.
  ///
  /// \pre I is a valid iterator into BB.
  void moveBefore(VPBasicBlock &BB, iplist<VPRecipeBase>::iterator I);

  /// This method unlinks 'this' from the containing basic block, but does not
  /// delete it.
  void removeFromParent();

  /// This method unlinks 'this' from the containing basic block and deletes it.
  ///
  /// \returns an iterator pointing to the element after the erased one
  iplist<VPRecipeBase>::iterator eraseFromParent();

  /// Method to support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const VPDef *D) {
    // All VPDefs are also VPRecipeBases.
    return true;
  }

  static inline bool classof(const VPUser *U) { return true; }

  /// Returns true if the recipe may have side-effects.
  bool mayHaveSideEffects() const;

  /// Returns true for PHI-like recipes.
  bool isPhi() const;

  /// Returns true if the recipe may read from memory.
  bool mayReadFromMemory() const;

  /// Returns true if the recipe may write to memory.
  bool mayWriteToMemory() const;

  /// Returns true if the recipe may read from or write to memory.
  bool mayReadOrWriteMemory() const {
    return mayReadFromMemory() || mayWriteToMemory();
  }

  /// Returns the debug location of the recipe.
  DebugLoc getDebugLoc() const { return DL; }

  /// Return true if the recipe is a scalar cast.
  bool isScalarCast() const;

  /// Set the recipe's debug location to \p NewDL.
  void setDebugLoc(DebugLoc NewDL) { DL = NewDL; }

protected:
  /// Compute the cost of this recipe either using a recipe's specialized
  /// implementation or using the legacy cost model and the underlying
  /// instructions.
  virtual InstructionCost computeCost(ElementCount VF,
                                      VPCostContext &Ctx) const;
};

// Helper macro to define common classof implementations for recipes.
#define VP_CLASSOF_IMPL(VPDefID)                                               \
  static inline bool classof(const VPDef *D) {                                 \
    return D->getVPDefID() == VPDefID;                                         \
  }                                                                            \
  static inline bool classof(const VPValue *V) {                               \
    auto *R = V->getDefiningRecipe();                                          \
    return R && R->getVPDefID() == VPDefID;                                    \
  }                                                                            \
  static inline bool classof(const VPUser *U) {                                \
    auto *R = dyn_cast<VPRecipeBase>(U);                                       \
    return R && R->getVPDefID() == VPDefID;                                    \
  }                                                                            \
  static inline bool classof(const VPRecipeBase *R) {                          \
    return R->getVPDefID() == VPDefID;                                         \
  }                                                                            \
  static inline bool classof(const VPSingleDefRecipe *R) {                     \
    return R->getVPDefID() == VPDefID;                                         \
  }

/// VPSingleDef is a base class for recipes for modeling a sequence of one or
/// more output IR that define a single result VPValue.
/// Note that VPRecipeBase must be inherited from before VPValue.
class VPSingleDefRecipe : public VPRecipeBase, public VPValue {
public:
  VPSingleDefRecipe(const unsigned char SC, ArrayRef<VPValue *> Operands,
                    DebugLoc DL = {})
      : VPRecipeBase(SC, Operands, DL), VPValue(this) {}

  VPSingleDefRecipe(const unsigned char SC, ArrayRef<VPValue *> Operands,
                    Value *UV, DebugLoc DL = {})
      : VPRecipeBase(SC, Operands, DL), VPValue(this, UV) {}

  static inline bool classof(const VPRecipeBase *R) {
    switch (R->getVPDefID()) {
    case VPRecipeBase::VPDerivedIVSC:
    case VPRecipeBase::VPEVLBasedIVPHISC:
    case VPRecipeBase::VPExpandSCEVSC:
    case VPRecipeBase::VPExpressionSC:
    case VPRecipeBase::VPInstructionSC:
    case VPRecipeBase::VPReductionEVLSC:
    case VPRecipeBase::VPReductionSC:
    case VPRecipeBase::VPReplicateSC:
    case VPRecipeBase::VPScalarIVStepsSC:
    case VPRecipeBase::VPVectorPointerSC:
    case VPRecipeBase::VPVectorEndPointerSC:
    case VPRecipeBase::VPWidenCallSC:
    case VPRecipeBase::VPWidenCanonicalIVSC:
    case VPRecipeBase::VPWidenCastSC:
    case VPRecipeBase::VPWidenGEPSC:
    case VPRecipeBase::VPWidenIntrinsicSC:
    case VPRecipeBase::VPWidenSC:
    case VPRecipeBase::VPWidenSelectSC:
    case VPRecipeBase::VPBlendSC:
    case VPRecipeBase::VPPredInstPHISC:
    case VPRecipeBase::VPCanonicalIVPHISC:
    case VPRecipeBase::VPActiveLaneMaskPHISC:
    case VPRecipeBase::VPFirstOrderRecurrencePHISC:
    case VPRecipeBase::VPWidenPHISC:
    case VPRecipeBase::VPWidenIntOrFpInductionSC:
    case VPRecipeBase::VPWidenPointerInductionSC:
    case VPRecipeBase::VPReductionPHISC:
    case VPRecipeBase::VPPartialReductionSC:
      return true;
    case VPRecipeBase::VPBranchOnMaskSC:
    case VPRecipeBase::VPInterleaveSC:
    case VPRecipeBase::VPIRInstructionSC:
    case VPRecipeBase::VPWidenLoadEVLSC:
    case VPRecipeBase::VPWidenLoadSC:
    case VPRecipeBase::VPWidenStoreEVLSC:
    case VPRecipeBase::VPWidenStoreSC:
    case VPRecipeBase::VPHistogramSC:
      // TODO: Widened stores don't define a value, but widened loads do. Split
      // the recipes to be able to make widened loads VPSingleDefRecipes.
      return false;
    }
    llvm_unreachable("Unhandled VPDefID");
  }

  static inline bool classof(const VPUser *U) {
    auto *R = dyn_cast<VPRecipeBase>(U);
    return R && classof(R);
  }

  virtual VPSingleDefRecipe *clone() override = 0;

  /// Returns the underlying instruction.
  Instruction *getUnderlyingInstr() {
    return cast<Instruction>(getUnderlyingValue());
  }
  const Instruction *getUnderlyingInstr() const {
    return cast<Instruction>(getUnderlyingValue());
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print this VPSingleDefRecipe to dbgs() (for debugging).
  LLVM_DUMP_METHOD void dump() const;
#endif
};

/// Class to record and manage LLVM IR flags.
class VPIRFlags {
  enum class OperationType : unsigned char {
    Cmp,
    OverflowingBinOp,
    Trunc,
    DisjointOp,
    PossiblyExactOp,
    GEPOp,
    FPMathOp,
    NonNegOp,
    Other
  };

public:
  struct WrapFlagsTy {
    char HasNUW : 1;
    char HasNSW : 1;

    WrapFlagsTy(bool HasNUW, bool HasNSW) : HasNUW(HasNUW), HasNSW(HasNSW) {}
  };

  struct TruncFlagsTy {
    char HasNUW : 1;
    char HasNSW : 1;

    TruncFlagsTy(bool HasNUW, bool HasNSW) : HasNUW(HasNUW), HasNSW(HasNSW) {}
  };

  struct DisjointFlagsTy {
    char IsDisjoint : 1;
    DisjointFlagsTy(bool IsDisjoint) : IsDisjoint(IsDisjoint) {}
  };

  struct NonNegFlagsTy {
    char NonNeg : 1;
    NonNegFlagsTy(bool IsNonNeg) : NonNeg(IsNonNeg) {}
  };

private:
  struct ExactFlagsTy {
    char IsExact : 1;
  };
  struct FastMathFlagsTy {
    char AllowReassoc : 1;
    char NoNaNs : 1;
    char NoInfs : 1;
    char NoSignedZeros : 1;
    char AllowReciprocal : 1;
    char AllowContract : 1;
    char ApproxFunc : 1;

    LLVM_ABI_FOR_TEST FastMathFlagsTy(const FastMathFlags &FMF);
  };

  OperationType OpType;

  union {
    CmpInst::Predicate CmpPredicate;
    WrapFlagsTy WrapFlags;
    TruncFlagsTy TruncFlags;
    DisjointFlagsTy DisjointFlags;
    ExactFlagsTy ExactFlags;
    GEPNoWrapFlags GEPFlags;
    NonNegFlagsTy NonNegFlags;
    FastMathFlagsTy FMFs;
    unsigned AllFlags;
  };

public:
  VPIRFlags() : OpType(OperationType::Other), AllFlags(0) {}

  VPIRFlags(Instruction &I) {
    if (auto *Op = dyn_cast<CmpInst>(&I)) {
      OpType = OperationType::Cmp;
      CmpPredicate = Op->getPredicate();
    } else if (auto *Op = dyn_cast<PossiblyDisjointInst>(&I)) {
      OpType = OperationType::DisjointOp;
      DisjointFlags.IsDisjoint = Op->isDisjoint();
    } else if (auto *Op = dyn_cast<OverflowingBinaryOperator>(&I)) {
      OpType = OperationType::OverflowingBinOp;
      WrapFlags = {Op->hasNoUnsignedWrap(), Op->hasNoSignedWrap()};
    } else if (auto *Op = dyn_cast<TruncInst>(&I)) {
      OpType = OperationType::Trunc;
      TruncFlags = {Op->hasNoUnsignedWrap(), Op->hasNoSignedWrap()};
    } else if (auto *Op = dyn_cast<PossiblyExactOperator>(&I)) {
      OpType = OperationType::PossiblyExactOp;
      ExactFlags.IsExact = Op->isExact();
    } else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      OpType = OperationType::GEPOp;
      GEPFlags = GEP->getNoWrapFlags();
    } else if (auto *PNNI = dyn_cast<PossiblyNonNegInst>(&I)) {
      OpType = OperationType::NonNegOp;
      NonNegFlags.NonNeg = PNNI->hasNonNeg();
    } else if (auto *Op = dyn_cast<FPMathOperator>(&I)) {
      OpType = OperationType::FPMathOp;
      FMFs = Op->getFastMathFlags();
    } else {
      OpType = OperationType::Other;
      AllFlags = 0;
    }
  }

  VPIRFlags(CmpInst::Predicate Pred)
      : OpType(OperationType::Cmp), CmpPredicate(Pred) {}

  VPIRFlags(WrapFlagsTy WrapFlags)
      : OpType(OperationType::OverflowingBinOp), WrapFlags(WrapFlags) {}

  VPIRFlags(FastMathFlags FMFs) : OpType(OperationType::FPMathOp), FMFs(FMFs) {}

  VPIRFlags(DisjointFlagsTy DisjointFlags)
      : OpType(OperationType::DisjointOp), DisjointFlags(DisjointFlags) {}

  VPIRFlags(NonNegFlagsTy NonNegFlags)
      : OpType(OperationType::NonNegOp), NonNegFlags(NonNegFlags) {}

  VPIRFlags(GEPNoWrapFlags GEPFlags)
      : OpType(OperationType::GEPOp), GEPFlags(GEPFlags) {}

public:
  void transferFlags(VPIRFlags &Other) {
    OpType = Other.OpType;
    AllFlags = Other.AllFlags;
  }

  /// Drop all poison-generating flags.
  void dropPoisonGeneratingFlags() {
    // NOTE: This needs to be kept in-sync with
    // Instruction::dropPoisonGeneratingFlags.
    switch (OpType) {
    case OperationType::OverflowingBinOp:
      WrapFlags.HasNUW = false;
      WrapFlags.HasNSW = false;
      break;
    case OperationType::Trunc:
      TruncFlags.HasNUW = false;
      TruncFlags.HasNSW = false;
      break;
    case OperationType::DisjointOp:
      DisjointFlags.IsDisjoint = false;
      break;
    case OperationType::PossiblyExactOp:
      ExactFlags.IsExact = false;
      break;
    case OperationType::GEPOp:
      GEPFlags = GEPNoWrapFlags::none();
      break;
    case OperationType::FPMathOp:
      FMFs.NoNaNs = false;
      FMFs.NoInfs = false;
      break;
    case OperationType::NonNegOp:
      NonNegFlags.NonNeg = false;
      break;
    case OperationType::Cmp:
    case OperationType::Other:
      break;
    }
  }

  /// Apply the IR flags to \p I.
  void applyFlags(Instruction &I) const {
    switch (OpType) {
    case OperationType::OverflowingBinOp:
      I.setHasNoUnsignedWrap(WrapFlags.HasNUW);
      I.setHasNoSignedWrap(WrapFlags.HasNSW);
      break;
    case OperationType::Trunc:
      I.setHasNoUnsignedWrap(TruncFlags.HasNUW);
      I.setHasNoSignedWrap(TruncFlags.HasNSW);
      break;
    case OperationType::DisjointOp:
      cast<PossiblyDisjointInst>(&I)->setIsDisjoint(DisjointFlags.IsDisjoint);
      break;
    case OperationType::PossiblyExactOp:
      I.setIsExact(ExactFlags.IsExact);
      break;
    case OperationType::GEPOp:
      cast<GetElementPtrInst>(&I)->setNoWrapFlags(GEPFlags);
      break;
    case OperationType::FPMathOp:
      I.setHasAllowReassoc(FMFs.AllowReassoc);
      I.setHasNoNaNs(FMFs.NoNaNs);
      I.setHasNoInfs(FMFs.NoInfs);
      I.setHasNoSignedZeros(FMFs.NoSignedZeros);
      I.setHasAllowReciprocal(FMFs.AllowReciprocal);
      I.setHasAllowContract(FMFs.AllowContract);
      I.setHasApproxFunc(FMFs.ApproxFunc);
      break;
    case OperationType::NonNegOp:
      I.setNonNeg(NonNegFlags.NonNeg);
      break;
    case OperationType::Cmp:
    case OperationType::Other:
      break;
    }
  }

  CmpInst::Predicate getPredicate() const {
    assert(OpType == OperationType::Cmp &&
           "recipe doesn't have a compare predicate");
    return CmpPredicate;
  }

  void setPredicate(CmpInst::Predicate Pred) {
    assert(OpType == OperationType::Cmp &&
           "recipe doesn't have a compare predicate");
    CmpPredicate = Pred;
  }

  GEPNoWrapFlags getGEPNoWrapFlags() const { return GEPFlags; }

  /// Returns true if the recipe has fast-math flags.
  bool hasFastMathFlags() const { return OpType == OperationType::FPMathOp; }

  LLVM_ABI_FOR_TEST FastMathFlags getFastMathFlags() const;

  /// Returns true if the recipe has non-negative flag.
  bool hasNonNegFlag() const { return OpType == OperationType::NonNegOp; }

  bool isNonNeg() const {
    assert(OpType == OperationType::NonNegOp &&
           "recipe doesn't have a NNEG flag");
    return NonNegFlags.NonNeg;
  }

  bool hasNoUnsignedWrap() const {
    switch (OpType) {
    case OperationType::OverflowingBinOp:
      return WrapFlags.HasNUW;
    case OperationType::Trunc:
      return TruncFlags.HasNUW;
    default:
      llvm_unreachable("recipe doesn't have a NUW flag");
    }
  }

  bool hasNoSignedWrap() const {
    switch (OpType) {
    case OperationType::OverflowingBinOp:
      return WrapFlags.HasNSW;
    case OperationType::Trunc:
      return TruncFlags.HasNSW;
    default:
      llvm_unreachable("recipe doesn't have a NSW flag");
    }
  }

  bool isDisjoint() const {
    assert(OpType == OperationType::DisjointOp &&
           "recipe cannot have a disjoing flag");
    return DisjointFlags.IsDisjoint;
  }

#if !defined(NDEBUG)
  /// Returns true if the set flags are valid for \p Opcode.
  bool flagsValidForOpcode(unsigned Opcode) const;
#endif

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void printFlags(raw_ostream &O) const;
#endif
};

/// A pure-virtual common base class for recipes defining a single VPValue and
/// using IR flags.
struct VPRecipeWithIRFlags : public VPSingleDefRecipe, public VPIRFlags {
  VPRecipeWithIRFlags(const unsigned char SC, ArrayRef<VPValue *> Operands,
                      DebugLoc DL = {})
      : VPSingleDefRecipe(SC, Operands, DL), VPIRFlags() {}

  VPRecipeWithIRFlags(const unsigned char SC, ArrayRef<VPValue *> Operands,
                      Instruction &I)
      : VPSingleDefRecipe(SC, Operands, &I, I.getDebugLoc()), VPIRFlags(I) {}

  VPRecipeWithIRFlags(const unsigned char SC, ArrayRef<VPValue *> Operands,
                      const VPIRFlags &Flags, DebugLoc DL = {})
      : VPSingleDefRecipe(SC, Operands, DL), VPIRFlags(Flags) {}

  static inline bool classof(const VPRecipeBase *R) {
    return R->getVPDefID() == VPRecipeBase::VPInstructionSC ||
           R->getVPDefID() == VPRecipeBase::VPWidenSC ||
           R->getVPDefID() == VPRecipeBase::VPWidenGEPSC ||
           R->getVPDefID() == VPRecipeBase::VPWidenCallSC ||
           R->getVPDefID() == VPRecipeBase::VPWidenCastSC ||
           R->getVPDefID() == VPRecipeBase::VPWidenIntrinsicSC ||
           R->getVPDefID() == VPRecipeBase::VPWidenSelectSC ||
           R->getVPDefID() == VPRecipeBase::VPReductionSC ||
           R->getVPDefID() == VPRecipeBase::VPReductionEVLSC ||
           R->getVPDefID() == VPRecipeBase::VPReplicateSC ||
           R->getVPDefID() == VPRecipeBase::VPVectorEndPointerSC ||
           R->getVPDefID() == VPRecipeBase::VPVectorPointerSC;
  }

  static inline bool classof(const VPUser *U) {
    auto *R = dyn_cast<VPRecipeBase>(U);
    return R && classof(R);
  }

  static inline bool classof(const VPValue *V) {
    auto *R = dyn_cast_or_null<VPRecipeBase>(V->getDefiningRecipe());
    return R && classof(R);
  }

  void execute(VPTransformState &State) override = 0;

  /// Compute the cost for this recipe for \p VF, using \p Opcode and \p Ctx.
  std::optional<InstructionCost>
  getCostForRecipeWithOpcode(unsigned Opcode, ElementCount VF,
                             VPCostContext &Ctx) const;
};

/// Helper to access the operand that contains the unroll part for this recipe
/// after unrolling.
template <unsigned PartOpIdx> class LLVM_ABI_FOR_TEST VPUnrollPartAccessor {
protected:
  /// Return the VPValue operand containing the unroll part or null if there is
  /// no such operand.
  VPValue *getUnrollPartOperand(const VPUser &U) const;

  /// Return the unroll part.
  unsigned getUnrollPart(const VPUser &U) const;
};

/// Helper to manage IR metadata for recipes. It filters out metadata that
/// cannot be propagated.
class VPIRMetadata {
  SmallVector<std::pair<unsigned, MDNode *>> Metadata;

public:
  VPIRMetadata() {}

  /// Adds metatadata that can be preserved from the original instruction
  /// \p I.
  VPIRMetadata(Instruction &I) { getMetadataToPropagate(&I, Metadata); }

  /// Adds metatadata that can be preserved from the original instruction
  /// \p I and noalias metadata guaranteed by runtime checks using \p LVer.
  VPIRMetadata(Instruction &I, LoopVersioning *LVer);

  /// Copy constructor for cloning.
  VPIRMetadata(const VPIRMetadata &Other) : Metadata(Other.Metadata) {}

  VPIRMetadata &operator=(const VPIRMetadata &Other) {
    Metadata = Other.Metadata;
    return *this;
  }

  /// Add all metadata to \p I.
  void applyMetadata(Instruction &I) const;

  /// Add metadata with kind \p Kind and \p Node.
  void addMetadata(unsigned Kind, MDNode *Node) {
    Metadata.emplace_back(Kind, Node);
  }

  /// Intersect this VPIRMetada object with \p MD, keeping only metadata
  /// nodes that are common to both.
  void intersect(const VPIRMetadata &MD);
};

/// This is a concrete Recipe that models a single VPlan-level instruction.
/// While as any Recipe it may generate a sequence of IR instructions when
/// executed, these instructions would always form a single-def expression as
/// the VPInstruction is also a single def-use vertex.
class LLVM_ABI_FOR_TEST VPInstruction : public VPRecipeWithIRFlags,
                                        public VPIRMetadata,
                                        public VPUnrollPartAccessor<1> {
  friend class VPlanSlp;

public:
  /// VPlan opcodes, extending LLVM IR with idiomatics instructions.
  enum {
    FirstOrderRecurrenceSplice =
        Instruction::OtherOpsEnd + 1, // Combines the incoming and previous
                                      // values of a first-order recurrence.
    Not,
    SLPLoad,
    SLPStore,
    ActiveLaneMask,
    ExplicitVectorLength,
    CalculateTripCountMinusVF,
    // Increment the canonical IV separately for each unrolled part.
    CanonicalIVIncrementForPart,
    BranchOnCount,
    BranchOnCond,
    Broadcast,
    /// Given operands of (the same) struct type, creates a struct of fixed-
    /// width vectors each containing a struct field of all operands. The
    /// number of operands matches the element count of every vector.
    BuildStructVector,
    /// Creates a fixed-width vector containing all operands. The number of
    /// operands matches the vector element count.
    BuildVector,
    /// Compute the final result of a AnyOf reduction with select(cmp(),x,y),
    /// where one of (x,y) is loop invariant, and both x and y are integer type.
    ComputeAnyOfResult,
    ComputeFindIVResult,
    ComputeReductionResult,
    // Extracts the last lane from its operand if it is a vector, or the last
    // part if scalar. In the latter case, the recipe will be removed during
    // unrolling.
    ExtractLastElement,
    // Extracts the second-to-last lane from its operand or the second-to-last
    // part if it is scalar. In the latter case, the recipe will be removed
    // during unrolling.
    ExtractPenultimateElement,
    LogicalAnd, // Non-poison propagating logical And.
    // Add an offset in bytes (second operand) to a base pointer (first
    // operand). Only generates scalar values (either for the first lane only or
    // for all lanes, depending on its uses).
    PtrAdd,
    // Add a vector offset in bytes (second operand) to a scalar base pointer
    // (first operand).
    WidePtrAdd,
    // Returns a scalar boolean value, which is true if any lane of its
    // (boolean) vector operands is true. It produces the reduced value across
    // all unrolled iterations. Unrolling will add all copies of its original
    // operand as additional operands.
    AnyOf,
    // Calculates the first active lane index of the vector predicate operands.
    // It produces the lane index across all unrolled iterations. Unrolling will
    // add all copies of its original operand as additional operands.
    FirstActiveLane,

    // The opcodes below are used for VPInstructionWithType.
    //
    /// Scale the first operand (vector step) by the second operand
    /// (scalar-step).  Casts both operands to the result type if needed.
    WideIVStep,
    /// Start vector for reductions with 3 operands: the original start value,
    /// the identity value for the reduction and an integer indicating the
    /// scaling factor.
    ReductionStartVector,
    // Creates a step vector starting from 0 to VF with a step of 1.
    StepVector,
    /// Extracts a single lane (first operand) from a set of vector operands.
    /// The lane specifies an index into a vector formed by combining all vector
    /// operands (all operands after the first one).
    ExtractLane,
    /// Explicit user for the resume phi of the canonical induction in the main
    /// VPlan, used by the epilogue vector loop.
    ResumeForEpilogue,
    /// Returns the value for vscale.
    VScale,
  };

private:
  typedef unsigned char OpcodeTy;
  OpcodeTy Opcode;

  /// An optional name that can be used for the generated IR instruction.
  const std::string Name;

  /// Returns true if this VPInstruction generates scalar values for all lanes.
  /// Most VPInstructions generate a single value per part, either vector or
  /// scalar. VPReplicateRecipe takes care of generating multiple (scalar)
  /// values per all lanes, stemming from an original ingredient. This method
  /// identifies the (rare) cases of VPInstructions that do so as well, w/o an
  /// underlying ingredient.
  bool doesGeneratePerAllLanes() const;

  /// Returns true if we can generate a scalar for the first lane only if
  /// needed.
  bool canGenerateScalarForFirstLane() const;

  /// Utility methods serving execute(): generates a single vector instance of
  /// the modeled instruction. \returns the generated value. . In some cases an
  /// existing value is returned rather than a generated one.
  Value *generate(VPTransformState &State);

  /// Utility methods serving execute(): generates a scalar single instance of
  /// the modeled instruction for a given lane. \returns the scalar generated
  /// value for lane \p Lane.
  Value *generatePerLane(VPTransformState &State, const VPLane &Lane);

#if !defined(NDEBUG)
  /// Return the number of operands determined by the opcode of the
  /// VPInstruction. Returns -1u if the number of operands cannot be determined
  /// directly by the opcode.
  static unsigned getNumOperandsForOpcode(unsigned Opcode);
#endif

public:
  VPInstruction(unsigned Opcode, ArrayRef<VPValue *> Operands, DebugLoc DL = {},
                const Twine &Name = "")
      : VPRecipeWithIRFlags(VPDef::VPInstructionSC, Operands, DL),
        VPIRMetadata(), Opcode(Opcode), Name(Name.str()) {}

  VPInstruction(unsigned Opcode, ArrayRef<VPValue *> Operands,
                const VPIRFlags &Flags, DebugLoc DL = {},
                const Twine &Name = "");

  VP_CLASSOF_IMPL(VPDef::VPInstructionSC)

  VPInstruction *clone() override {
    SmallVector<VPValue *, 2> Operands(operands());
    auto *New = new VPInstruction(Opcode, Operands, *this, getDebugLoc(), Name);
    if (getUnderlyingValue())
      New->setUnderlyingValue(getUnderlyingInstr());
    return New;
  }

  unsigned getOpcode() const { return Opcode; }

  /// Generate the instruction.
  /// TODO: We currently execute only per-part unless a specific instance is
  /// provided.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPInstruction.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the VPInstruction to \p O.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;

  /// Print the VPInstruction to dbgs() (for debugging).
  LLVM_DUMP_METHOD void dump() const;
#endif

  bool hasResult() const {
    // CallInst may or may not have a result, depending on the called function.
    // Conservatively return calls have results for now.
    switch (getOpcode()) {
    case Instruction::Ret:
    case Instruction::Br:
    case Instruction::Store:
    case Instruction::Switch:
    case Instruction::IndirectBr:
    case Instruction::Resume:
    case Instruction::CatchRet:
    case Instruction::Unreachable:
    case Instruction::Fence:
    case Instruction::AtomicRMW:
    case VPInstruction::BranchOnCond:
    case VPInstruction::BranchOnCount:
      return false;
    default:
      return true;
    }
  }

  /// Returns true if the underlying opcode may read from or write to memory.
  bool opcodeMayReadOrWriteFromMemory() const;

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override;

  /// Returns true if the recipe only uses the first part of operand \p Op.
  bool onlyFirstPartUsed(const VPValue *Op) const override;

  /// Returns true if this VPInstruction produces a scalar value from a vector,
  /// e.g. by performing a reduction or extracting a lane.
  bool isVectorToScalar() const;

  /// Returns true if this VPInstruction's operands are single scalars and the
  /// result is also a single scalar.
  bool isSingleScalar() const;

  /// Returns the symbolic name assigned to the VPInstruction.
  StringRef getName() const { return Name; }
};

/// A specialization of VPInstruction augmenting it with a dedicated result
/// type, to be used when the opcode and operands of the VPInstruction don't
/// directly determine the result type. Note that there is no separate VPDef ID
/// for VPInstructionWithType; it shares the same ID as VPInstruction and is
/// distinguished purely by the opcode.
class VPInstructionWithType : public VPInstruction {
  /// Scalar result type produced by the recipe.
  Type *ResultTy;

public:
  VPInstructionWithType(unsigned Opcode, ArrayRef<VPValue *> Operands,
                        Type *ResultTy, const VPIRFlags &Flags, DebugLoc DL,
                        const Twine &Name = "")
      : VPInstruction(Opcode, Operands, Flags, DL, Name), ResultTy(ResultTy) {}

  static inline bool classof(const VPRecipeBase *R) {
    // VPInstructionWithType are VPInstructions with specific opcodes requiring
    // type information.
    if (R->isScalarCast())
      return true;
    auto *VPI = dyn_cast<VPInstruction>(R);
    if (!VPI)
      return false;
    switch (VPI->getOpcode()) {
    case VPInstruction::WideIVStep:
    case VPInstruction::StepVector:
    case VPInstruction::VScale:
      return true;
    default:
      return false;
    }
  }

  static inline bool classof(const VPUser *R) {
    return isa<VPInstructionWithType>(cast<VPRecipeBase>(R));
  }

  VPInstruction *clone() override {
    SmallVector<VPValue *, 2> Operands(operands());
    auto *New =
        new VPInstructionWithType(getOpcode(), Operands, getResultType(), *this,
                                  getDebugLoc(), getName());
    New->setUnderlyingValue(getUnderlyingValue());
    return New;
  }

  void execute(VPTransformState &State) override;

  /// Return the cost of this VPInstruction.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override {
    // TODO: Compute accurate cost after retiring the legacy cost model.
    return 0;
  }

  Type *getResultType() const { return ResultTy; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif
};

/// Helper type to provide functions to access incoming values and blocks for
/// phi-like recipes.
class VPPhiAccessors {
protected:
  /// Return a VPRecipeBase* to the current object.
  virtual const VPRecipeBase *getAsRecipe() const = 0;

public:
  virtual ~VPPhiAccessors() = default;

  /// Returns the incoming VPValue with index \p Idx.
  VPValue *getIncomingValue(unsigned Idx) const {
    return getAsRecipe()->getOperand(Idx);
  }

  /// Returns the incoming block with index \p Idx.
  const VPBasicBlock *getIncomingBlock(unsigned Idx) const;

  /// Returns the number of incoming values, also number of incoming blocks.
  virtual unsigned getNumIncoming() const {
    return getAsRecipe()->getNumOperands();
  }

  /// Returns an interator range over the incoming values.
  VPUser::const_operand_range incoming_values() const {
    return make_range(getAsRecipe()->op_begin(),
                      getAsRecipe()->op_begin() + getNumIncoming());
  }

  using const_incoming_blocks_range = iterator_range<mapped_iterator<
      detail::index_iterator, std::function<const VPBasicBlock *(size_t)>>>;

  /// Returns an iterator range over the incoming blocks.
  const_incoming_blocks_range incoming_blocks() const {
    std::function<const VPBasicBlock *(size_t)> GetBlock = [this](size_t Idx) {
      return getIncomingBlock(Idx);
    };
    return map_range(index_range(0, getNumIncoming()), GetBlock);
  }

  /// Returns an iterator range over pairs of incoming values and corresponding
  /// incoming blocks.
  detail::zippy<llvm::detail::zip_first, VPUser::const_operand_range,
                const_incoming_blocks_range>
  incoming_values_and_blocks() const {
    return zip_equal(incoming_values(), incoming_blocks());
  }

  /// Removes the incoming value for \p IncomingBlock, which must be a
  /// predecessor.
  void removeIncomingValueFor(VPBlockBase *IncomingBlock) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void printPhiOperands(raw_ostream &O, VPSlotTracker &SlotTracker) const;
#endif
};

struct LLVM_ABI_FOR_TEST VPPhi : public VPInstruction, public VPPhiAccessors {
  VPPhi(ArrayRef<VPValue *> Operands, DebugLoc DL, const Twine &Name = "")
      : VPInstruction(Instruction::PHI, Operands, DL, Name) {}

  static inline bool classof(const VPUser *U) {
    auto *VPI = dyn_cast<VPInstruction>(U);
    return VPI && VPI->getOpcode() == Instruction::PHI;
  }

  static inline bool classof(const VPValue *V) {
    auto *VPI = dyn_cast<VPInstruction>(V);
    return VPI && VPI->getOpcode() == Instruction::PHI;
  }

  static inline bool classof(const VPSingleDefRecipe *SDR) {
    auto *VPI = dyn_cast<VPInstruction>(SDR);
    return VPI && VPI->getOpcode() == Instruction::PHI;
  }

  VPPhi *clone() override {
    auto *PhiR = new VPPhi(operands(), getDebugLoc(), getName());
    PhiR->setUnderlyingValue(getUnderlyingValue());
    return PhiR;
  }

  void execute(VPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

protected:
  const VPRecipeBase *getAsRecipe() const override { return this; }
};

/// A recipe to wrap on original IR instruction not to be modified during
/// execution, except for PHIs. PHIs are modeled via the VPIRPhi subclass.
/// Expect PHIs, VPIRInstructions cannot have any operands.
class VPIRInstruction : public VPRecipeBase {
  Instruction &I;

protected:
  /// VPIRInstruction::create() should be used to create VPIRInstructions, as
  /// subclasses may need to be created, e.g. VPIRPhi.
  VPIRInstruction(Instruction &I)
      : VPRecipeBase(VPDef::VPIRInstructionSC, ArrayRef<VPValue *>()), I(I) {}

public:
  ~VPIRInstruction() override = default;

  /// Create a new VPIRPhi for \p \I, if it is a PHINode, otherwise create a
  /// VPIRInstruction.
  LLVM_ABI_FOR_TEST static VPIRInstruction *create(Instruction &I);

  VP_CLASSOF_IMPL(VPDef::VPIRInstructionSC)

  VPIRInstruction *clone() override {
    auto *R = create(I);
    for (auto *Op : operands())
      R->addOperand(Op);
    return R;
  }

  void execute(VPTransformState &State) override;

  /// Return the cost of this VPIRInstruction.
  LLVM_ABI_FOR_TEST InstructionCost
  computeCost(ElementCount VF, VPCostContext &Ctx) const override;

  Instruction &getInstruction() const { return I; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  bool usesScalars(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }

  bool onlyFirstPartUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }

  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }

  /// Update the recipes first operand to the last lane of the operand using \p
  /// Builder. Must only be used for VPIRInstructions with at least one operand
  /// wrapping a PHINode.
  void extractLastLaneOfFirstOperand(VPBuilder &Builder);
};

/// An overlay for VPIRInstructions wrapping PHI nodes enabling convenient use
/// cast/dyn_cast/isa and execute() implementation. A single VPValue operand is
/// allowed, and it is used to add a new incoming value for the single
/// predecessor VPBB.
struct LLVM_ABI_FOR_TEST VPIRPhi : public VPIRInstruction,
                                   public VPPhiAccessors {
  VPIRPhi(PHINode &PN) : VPIRInstruction(PN) {}

  static inline bool classof(const VPRecipeBase *U) {
    auto *R = dyn_cast<VPIRInstruction>(U);
    return R && isa<PHINode>(R->getInstruction());
  }

  PHINode &getIRPhi() { return cast<PHINode>(getInstruction()); }

  void execute(VPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

protected:
  const VPRecipeBase *getAsRecipe() const override { return this; }
};

/// VPWidenRecipe is a recipe for producing a widened instruction using the
/// opcode and operands of the recipe. This recipe covers most of the
/// traditional vectorization cases where each recipe transforms into a
/// vectorized version of itself.
class LLVM_ABI_FOR_TEST VPWidenRecipe : public VPRecipeWithIRFlags,
                                        public VPIRMetadata {
  unsigned Opcode;

public:
  VPWidenRecipe(unsigned Opcode, ArrayRef<VPValue *> Operands,
                const VPIRFlags &Flags, const VPIRMetadata &Metadata,
                DebugLoc DL)
      : VPRecipeWithIRFlags(VPDef::VPWidenSC, Operands, Flags, DL),
        VPIRMetadata(Metadata), Opcode(Opcode) {}

  VPWidenRecipe(Instruction &I, ArrayRef<VPValue *> Operands)
      : VPRecipeWithIRFlags(VPDef::VPWidenSC, Operands, I), VPIRMetadata(I),
        Opcode(I.getOpcode()) {}

  ~VPWidenRecipe() override = default;

  VPWidenRecipe *clone() override {
    auto *R =
        new VPWidenRecipe(getOpcode(), operands(), *this, *this, getDebugLoc());
    R->setUnderlyingValue(getUnderlyingValue());
    return R;
  }

  VP_CLASSOF_IMPL(VPDef::VPWidenSC)

  /// Produce a widened instruction using the opcode and operands of the recipe,
  /// processing State.VF elements.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPWidenRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

  unsigned getOpcode() const { return Opcode; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif
};

/// VPWidenCastRecipe is a recipe to create vector cast instructions.
class VPWidenCastRecipe : public VPRecipeWithIRFlags, public VPIRMetadata {
  /// Cast instruction opcode.
  Instruction::CastOps Opcode;

  /// Result type for the cast.
  Type *ResultTy;

public:
  VPWidenCastRecipe(Instruction::CastOps Opcode, VPValue *Op, Type *ResultTy,
                    CastInst &UI)
      : VPRecipeWithIRFlags(VPDef::VPWidenCastSC, Op, UI), VPIRMetadata(UI),
        Opcode(Opcode), ResultTy(ResultTy) {
    assert(UI.getOpcode() == Opcode &&
           "opcode of underlying cast doesn't match");
  }

  VPWidenCastRecipe(Instruction::CastOps Opcode, VPValue *Op, Type *ResultTy,
                    const VPIRFlags &Flags = {}, DebugLoc DL = {})
      : VPRecipeWithIRFlags(VPDef::VPWidenCastSC, Op, Flags, DL),
        VPIRMetadata(), Opcode(Opcode), ResultTy(ResultTy) {
    assert(flagsValidForOpcode(Opcode) &&
           "Set flags not supported for the provided opcode");
  }

  ~VPWidenCastRecipe() override = default;

  VPWidenCastRecipe *clone() override {
    if (auto *UV = getUnderlyingValue())
      return new VPWidenCastRecipe(Opcode, getOperand(0), ResultTy,
                                   *cast<CastInst>(UV));

    return new VPWidenCastRecipe(Opcode, getOperand(0), ResultTy);
  }

  VP_CLASSOF_IMPL(VPDef::VPWidenCastSC)

  /// Produce widened copies of the cast.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPWidenCastRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  Instruction::CastOps getOpcode() const { return Opcode; }

  /// Returns the result type of the cast.
  Type *getResultType() const { return ResultTy; }
};

/// A recipe for widening vector intrinsics.
class VPWidenIntrinsicRecipe : public VPRecipeWithIRFlags, public VPIRMetadata {
  /// ID of the vector intrinsic to widen.
  Intrinsic::ID VectorIntrinsicID;

  /// Scalar return type of the intrinsic.
  Type *ResultTy;

  /// True if the intrinsic may read from memory.
  bool MayReadFromMemory;

  /// True if the intrinsic may read write to memory.
  bool MayWriteToMemory;

  /// True if the intrinsic may have side-effects.
  bool MayHaveSideEffects;

public:
  VPWidenIntrinsicRecipe(CallInst &CI, Intrinsic::ID VectorIntrinsicID,
                         ArrayRef<VPValue *> CallArguments, Type *Ty,
                         DebugLoc DL = {})
      : VPRecipeWithIRFlags(VPDef::VPWidenIntrinsicSC, CallArguments, CI),
        VPIRMetadata(CI), VectorIntrinsicID(VectorIntrinsicID), ResultTy(Ty),
        MayReadFromMemory(CI.mayReadFromMemory()),
        MayWriteToMemory(CI.mayWriteToMemory()),
        MayHaveSideEffects(CI.mayHaveSideEffects()) {}

  VPWidenIntrinsicRecipe(Intrinsic::ID VectorIntrinsicID,
                         ArrayRef<VPValue *> CallArguments, Type *Ty,
                         DebugLoc DL = {})
      : VPRecipeWithIRFlags(VPDef::VPWidenIntrinsicSC, CallArguments, DL),
        VPIRMetadata(), VectorIntrinsicID(VectorIntrinsicID), ResultTy(Ty) {
    LLVMContext &Ctx = Ty->getContext();
    AttributeSet Attrs = Intrinsic::getFnAttributes(Ctx, VectorIntrinsicID);
    MemoryEffects ME = Attrs.getMemoryEffects();
    MayReadFromMemory = !ME.onlyWritesMemory();
    MayWriteToMemory = !ME.onlyReadsMemory();
    MayHaveSideEffects = MayWriteToMemory ||
                         !Attrs.hasAttribute(Attribute::NoUnwind) ||
                         !Attrs.hasAttribute(Attribute::WillReturn);
  }

  ~VPWidenIntrinsicRecipe() override = default;

  VPWidenIntrinsicRecipe *clone() override {
    if (Value *CI = getUnderlyingValue())
      return new VPWidenIntrinsicRecipe(*cast<CallInst>(CI), VectorIntrinsicID,
                                        operands(), ResultTy, getDebugLoc());
    return new VPWidenIntrinsicRecipe(VectorIntrinsicID, operands(), ResultTy,
                                      getDebugLoc());
  }

  VP_CLASSOF_IMPL(VPDef::VPWidenIntrinsicSC)

  /// Produce a widened version of the vector intrinsic.
  void execute(VPTransformState &State) override;

  /// Return the cost of this vector intrinsic.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

  /// Return the ID of the intrinsic.
  Intrinsic::ID getVectorIntrinsicID() const { return VectorIntrinsicID; }

  /// Return the scalar return type of the intrinsic.
  Type *getResultType() const { return ResultTy; }

  /// Return to name of the intrinsic as string.
  StringRef getIntrinsicName() const;

  /// Returns true if the intrinsic may read from memory.
  bool mayReadFromMemory() const { return MayReadFromMemory; }

  /// Returns true if the intrinsic may write to memory.
  bool mayWriteToMemory() const { return MayWriteToMemory; }

  /// Returns true if the intrinsic may have side-effects.
  bool mayHaveSideEffects() const { return MayHaveSideEffects; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  bool onlyFirstLaneUsed(const VPValue *Op) const override;
};

/// A recipe for widening Call instructions using library calls.
class LLVM_ABI_FOR_TEST VPWidenCallRecipe : public VPRecipeWithIRFlags,
                                            public VPIRMetadata {
  /// Variant stores a pointer to the chosen function. There is a 1:1 mapping
  /// between a given VF and the chosen vectorized variant, so there will be a
  /// different VPlan for each VF with a valid variant.
  Function *Variant;

public:
  VPWidenCallRecipe(Value *UV, Function *Variant,
                    ArrayRef<VPValue *> CallArguments, DebugLoc DL = {})
      : VPRecipeWithIRFlags(VPDef::VPWidenCallSC, CallArguments,
                            *cast<Instruction>(UV)),
        VPIRMetadata(*cast<Instruction>(UV)), Variant(Variant) {
    assert(
        isa<Function>(getOperand(getNumOperands() - 1)->getLiveInIRValue()) &&
        "last operand must be the called function");
  }

  ~VPWidenCallRecipe() override = default;

  VPWidenCallRecipe *clone() override {
    return new VPWidenCallRecipe(getUnderlyingValue(), Variant, operands(),
                                 getDebugLoc());
  }

  VP_CLASSOF_IMPL(VPDef::VPWidenCallSC)

  /// Produce a widened version of the call instruction.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPWidenCallRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

  Function *getCalledScalarFunction() const {
    return cast<Function>(getOperand(getNumOperands() - 1)->getLiveInIRValue());
  }

  operand_range args() { return make_range(op_begin(), std::prev(op_end())); }
  const_operand_range args() const {
    return make_range(op_begin(), std::prev(op_end()));
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif
};

/// A recipe representing a sequence of load -> update -> store as part of
/// a histogram operation. This means there may be aliasing between vector
/// lanes, which is handled by the llvm.experimental.vector.histogram family
/// of intrinsics. The only update operations currently supported are
/// 'add' and 'sub' where the other term is loop-invariant.
class VPHistogramRecipe : public VPRecipeBase {
  /// Opcode of the update operation, currently either add or sub.
  unsigned Opcode;

public:
  VPHistogramRecipe(unsigned Opcode, ArrayRef<VPValue *> Operands,
                    DebugLoc DL = {})
      : VPRecipeBase(VPDef::VPHistogramSC, Operands, DL), Opcode(Opcode) {}

  ~VPHistogramRecipe() override = default;

  VPHistogramRecipe *clone() override {
    return new VPHistogramRecipe(Opcode, operands(), getDebugLoc());
  }

  VP_CLASSOF_IMPL(VPDef::VPHistogramSC);

  /// Produce a vectorized histogram operation.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPHistogramRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

  unsigned getOpcode() const { return Opcode; }

  /// Return the mask operand if one was provided, or a null pointer if all
  /// lanes should be executed unconditionally.
  VPValue *getMask() const {
    return getNumOperands() == 3 ? getOperand(2) : nullptr;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif
};

/// A recipe for widening select instructions.
struct LLVM_ABI_FOR_TEST VPWidenSelectRecipe : public VPRecipeWithIRFlags,
                                               public VPIRMetadata {
  VPWidenSelectRecipe(SelectInst &I, ArrayRef<VPValue *> Operands)
      : VPRecipeWithIRFlags(VPDef::VPWidenSelectSC, Operands, I),
        VPIRMetadata(I) {}

  ~VPWidenSelectRecipe() override = default;

  VPWidenSelectRecipe *clone() override {
    return new VPWidenSelectRecipe(*cast<SelectInst>(getUnderlyingInstr()),
                                   operands());
  }

  VP_CLASSOF_IMPL(VPDef::VPWidenSelectSC)

  /// Produce a widened version of the select instruction.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPWidenSelectRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  unsigned getOpcode() const { return Instruction::Select; }

  VPValue *getCond() const {
    return getOperand(0);
  }

  bool isInvariantCond() const {
    return getCond()->isDefinedOutsideLoopRegions();
  }

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return Op == getCond() && isInvariantCond();
  }
};

/// A recipe for handling GEP instructions.
class LLVM_ABI_FOR_TEST VPWidenGEPRecipe : public VPRecipeWithIRFlags {
  bool isPointerLoopInvariant() const {
    return getOperand(0)->isDefinedOutsideLoopRegions();
  }

  bool isIndexLoopInvariant(unsigned I) const {
    return getOperand(I + 1)->isDefinedOutsideLoopRegions();
  }

  bool areAllOperandsInvariant() const {
    return all_of(operands(), [](VPValue *Op) {
      return Op->isDefinedOutsideLoopRegions();
    });
  }

public:
  VPWidenGEPRecipe(GetElementPtrInst *GEP, ArrayRef<VPValue *> Operands)
      : VPRecipeWithIRFlags(VPDef::VPWidenGEPSC, Operands, *GEP) {
    SmallVector<std::pair<unsigned, MDNode *>> Metadata;
    (void)Metadata;
    getMetadataToPropagate(GEP, Metadata);
    assert(Metadata.empty() && "unexpected metadata on GEP");
  }

  ~VPWidenGEPRecipe() override = default;

  VPWidenGEPRecipe *clone() override {
    return new VPWidenGEPRecipe(cast<GetElementPtrInst>(getUnderlyingInstr()),
                                operands());
  }

  VP_CLASSOF_IMPL(VPDef::VPWidenGEPSC)

  /// Generate the gep nodes.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPWidenGEPRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override {
    // TODO: Compute accurate cost after retiring the legacy cost model.
    return 0;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    if (Op == getOperand(0))
      return isPointerLoopInvariant();
    else
      return !isPointerLoopInvariant() && Op->isDefinedOutsideLoopRegions();
  }
};

/// A recipe to compute a pointer to the last element of each part of a widened
/// memory access for widened memory accesses of IndexedTy. Used for
/// VPWidenMemoryRecipes or VPInterleaveRecipes that are reversed.
class VPVectorEndPointerRecipe : public VPRecipeWithIRFlags,
                                 public VPUnrollPartAccessor<2> {
  Type *IndexedTy;

  /// The constant stride of the pointer computed by this recipe, expressed in
  /// units of IndexedTy.
  int64_t Stride;

public:
  VPVectorEndPointerRecipe(VPValue *Ptr, VPValue *VF, Type *IndexedTy,
                           int64_t Stride, GEPNoWrapFlags GEPFlags, DebugLoc DL)
      : VPRecipeWithIRFlags(VPDef::VPVectorEndPointerSC,
                            ArrayRef<VPValue *>({Ptr, VF}), GEPFlags, DL),
        IndexedTy(IndexedTy), Stride(Stride) {
    assert(Stride < 0 && "Stride must be negative");
  }

  VP_CLASSOF_IMPL(VPDef::VPVectorEndPointerSC)

  VPValue *getVFValue() { return getOperand(1); }
  const VPValue *getVFValue() const { return getOperand(1); }

  void execute(VPTransformState &State) override;

  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }

  /// Return the cost of this VPVectorPointerRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override {
    // TODO: Compute accurate cost after retiring the legacy cost model.
    return 0;
  }

  /// Returns true if the recipe only uses the first part of operand \p Op.
  bool onlyFirstPartUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    assert(getNumOperands() <= 2 && "must have at most two operands");
    return true;
  }

  VPVectorEndPointerRecipe *clone() override {
    return new VPVectorEndPointerRecipe(getOperand(0), getVFValue(), IndexedTy,
                                        Stride, getGEPNoWrapFlags(),
                                        getDebugLoc());
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif
};

/// A recipe to compute the pointers for widened memory accesses of IndexTy.
class VPVectorPointerRecipe : public VPRecipeWithIRFlags,
                              public VPUnrollPartAccessor<1> {
  Type *IndexedTy;

public:
  VPVectorPointerRecipe(VPValue *Ptr, Type *IndexedTy, GEPNoWrapFlags GEPFlags,
                        DebugLoc DL)
      : VPRecipeWithIRFlags(VPDef::VPVectorPointerSC, ArrayRef<VPValue *>(Ptr),
                            GEPFlags, DL),
        IndexedTy(IndexedTy) {}

  VP_CLASSOF_IMPL(VPDef::VPVectorPointerSC)

  void execute(VPTransformState &State) override;

  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }

  /// Returns true if the recipe only uses the first part of operand \p Op.
  bool onlyFirstPartUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    assert(getNumOperands() <= 2 && "must have at most two operands");
    return true;
  }

  VPVectorPointerRecipe *clone() override {
    return new VPVectorPointerRecipe(getOperand(0), IndexedTy,
                                     getGEPNoWrapFlags(), getDebugLoc());
  }

  /// Return true if this VPVectorPointerRecipe corresponds to part 0. Note that
  /// this is only accurate after the VPlan has been unrolled.
  bool isFirstPart() const { return getUnrollPart(*this) == 0; }

  /// Return the cost of this VPHeaderPHIRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override {
    // TODO: Compute accurate cost after retiring the legacy cost model.
    return 0;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif
};

/// A pure virtual base class for all recipes modeling header phis, including
/// phis for first order recurrences, pointer inductions and reductions. The
/// start value is the first operand of the recipe and the incoming value from
/// the backedge is the second operand.
///
/// Inductions are modeled using the following sub-classes:
///  * VPCanonicalIVPHIRecipe: Canonical scalar induction of the vector loop,
///    starting at a specified value (zero for the main vector loop, the resume
///    value for the epilogue vector loop) and stepping by 1. The induction
///    controls exiting of the vector loop by comparing against the vector trip
///    count. Produces a single scalar PHI for the induction value per
///    iteration.
///  * VPWidenIntOrFpInductionRecipe: Generates vector values for integer and
///    floating point inductions with arbitrary start and step values. Produces
///    a vector PHI per-part.
///  * VPDerivedIVRecipe: Converts the canonical IV value to the corresponding
///    value of an IV with different start and step values. Produces a single
///    scalar value per iteration
///  * VPScalarIVStepsRecipe: Generates scalar values per-lane based on a
///    canonical or derived induction.
///  * VPWidenPointerInductionRecipe: Generate vector and scalar values for a
///    pointer induction. Produces either a vector PHI per-part or scalar values
///    per-lane based on the canonical induction.
class LLVM_ABI_FOR_TEST VPHeaderPHIRecipe : public VPSingleDefRecipe,
                                            public VPPhiAccessors {
protected:
  VPHeaderPHIRecipe(unsigned char VPDefID, Instruction *UnderlyingInstr,
                    VPValue *Start, DebugLoc DL = DebugLoc::getUnknown())
      : VPSingleDefRecipe(VPDefID, ArrayRef<VPValue *>({Start}),
                          UnderlyingInstr, DL) {}

  const VPRecipeBase *getAsRecipe() const override { return this; }

public:
  ~VPHeaderPHIRecipe() override = default;

  /// Method to support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const VPRecipeBase *B) {
    return B->getVPDefID() >= VPDef::VPFirstHeaderPHISC &&
           B->getVPDefID() <= VPDef::VPLastHeaderPHISC;
  }
  static inline bool classof(const VPValue *V) {
    auto *B = V->getDefiningRecipe();
    return B && B->getVPDefID() >= VPRecipeBase::VPFirstHeaderPHISC &&
           B->getVPDefID() <= VPRecipeBase::VPLastHeaderPHISC;
  }

  /// Generate the phi nodes.
  void execute(VPTransformState &State) override = 0;

  /// Return the cost of this header phi recipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override = 0;
#endif

  /// Returns the start value of the phi, if one is set.
  VPValue *getStartValue() {
    return getNumOperands() == 0 ? nullptr : getOperand(0);
  }
  VPValue *getStartValue() const {
    return getNumOperands() == 0 ? nullptr : getOperand(0);
  }

  /// Update the start value of the recipe.
  void setStartValue(VPValue *V) { setOperand(0, V); }

  /// Returns the incoming value from the loop backedge.
  virtual VPValue *getBackedgeValue() {
    return getOperand(1);
  }

  /// Returns the backedge value as a recipe. The backedge value is guaranteed
  /// to be a recipe.
  virtual VPRecipeBase &getBackedgeRecipe() {
    return *getBackedgeValue()->getDefiningRecipe();
  }
};

/// Base class for widened induction (VPWidenIntOrFpInductionRecipe and
/// VPWidenPointerInductionRecipe), providing shared functionality, including
/// retrieving the step value, induction descriptor and original phi node.
class VPWidenInductionRecipe : public VPHeaderPHIRecipe {
  const InductionDescriptor &IndDesc;

public:
  VPWidenInductionRecipe(unsigned char Kind, PHINode *IV, VPValue *Start,
                         VPValue *Step, const InductionDescriptor &IndDesc,
                         DebugLoc DL)
      : VPHeaderPHIRecipe(Kind, IV, Start, DL), IndDesc(IndDesc) {
    addOperand(Step);
  }

  static inline bool classof(const VPRecipeBase *R) {
    return R->getVPDefID() == VPDef::VPWidenIntOrFpInductionSC ||
           R->getVPDefID() == VPDef::VPWidenPointerInductionSC;
  }

  static inline bool classof(const VPValue *V) {
    auto *R = V->getDefiningRecipe();
    return R && classof(R);
  }

  static inline bool classof(const VPHeaderPHIRecipe *R) {
    return classof(static_cast<const VPRecipeBase *>(R));
  }

  virtual void execute(VPTransformState &State) override = 0;

  /// Returns the step value of the induction.
  VPValue *getStepValue() { return getOperand(1); }
  const VPValue *getStepValue() const { return getOperand(1); }

  /// Update the step value of the recipe.
  void setStepValue(VPValue *V) { setOperand(1, V); }

  VPValue *getVFValue() { return getOperand(2); }
  const VPValue *getVFValue() const { return getOperand(2); }

  /// Returns the number of incoming values, also number of incoming blocks.
  /// Note that at the moment, VPWidenPointerInductionRecipe only has a single
  /// incoming value, its start value.
  unsigned getNumIncoming() const override { return 1; }

  PHINode *getPHINode() const { return cast<PHINode>(getUnderlyingValue()); }

  /// Returns the induction descriptor for the recipe.
  const InductionDescriptor &getInductionDescriptor() const { return IndDesc; }

  VPValue *getBackedgeValue() override {
    // TODO: All operands of base recipe must exist and be at same index in
    // derived recipe.
    llvm_unreachable(
        "VPWidenIntOrFpInductionRecipe generates its own backedge value");
  }

  VPRecipeBase &getBackedgeRecipe() override {
    // TODO: All operands of base recipe must exist and be at same index in
    // derived recipe.
    llvm_unreachable(
        "VPWidenIntOrFpInductionRecipe generates its own backedge value");
  }

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    // The recipe creates its own wide start value, so it only requests the
    // first lane of the operand.
    // TODO: Remove once creating the start value is modeled separately.
    return Op == getStartValue() || Op == getStepValue();
  }
};

/// A recipe for handling phi nodes of integer and floating-point inductions,
/// producing their vector values. This is an abstract recipe and must be
/// converted to concrete recipes before executing.
class VPWidenIntOrFpInductionRecipe : public VPWidenInductionRecipe {
  TruncInst *Trunc;

  // If this recipe is unrolled it will have 2 additional operands.
  bool isUnrolled() const { return getNumOperands() == 5; }

public:
  VPWidenIntOrFpInductionRecipe(PHINode *IV, VPValue *Start, VPValue *Step,
                                VPValue *VF, const InductionDescriptor &IndDesc,
                                DebugLoc DL)
      : VPWidenInductionRecipe(VPDef::VPWidenIntOrFpInductionSC, IV, Start,
                               Step, IndDesc, DL),
        Trunc(nullptr) {
    addOperand(VF);
  }

  VPWidenIntOrFpInductionRecipe(PHINode *IV, VPValue *Start, VPValue *Step,
                                VPValue *VF, const InductionDescriptor &IndDesc,
                                TruncInst *Trunc, DebugLoc DL)
      : VPWidenInductionRecipe(VPDef::VPWidenIntOrFpInductionSC, IV, Start,
                               Step, IndDesc, DL),
        Trunc(Trunc) {
    addOperand(VF);
    SmallVector<std::pair<unsigned, MDNode *>> Metadata;
    (void)Metadata;
    if (Trunc)
      getMetadataToPropagate(Trunc, Metadata);
    assert(Metadata.empty() && "unexpected metadata on Trunc");
  }

  ~VPWidenIntOrFpInductionRecipe() override = default;

  VPWidenIntOrFpInductionRecipe *clone() override {
    return new VPWidenIntOrFpInductionRecipe(
        getPHINode(), getStartValue(), getStepValue(), getVFValue(),
        getInductionDescriptor(), Trunc, getDebugLoc());
  }

  VP_CLASSOF_IMPL(VPDef::VPWidenIntOrFpInductionSC)

  void execute(VPTransformState &State) override {
    llvm_unreachable("cannot execute this recipe, should be expanded via "
                     "expandVPWidenIntOrFpInductionRecipe");
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  VPValue *getSplatVFValue() {
    // If the recipe has been unrolled return the VPValue for the induction
    // increment.
    return isUnrolled() ? getOperand(getNumOperands() - 2) : nullptr;
  }

  /// Returns the number of incoming values, also number of incoming blocks.
  /// Note that at the moment, VPWidenIntOrFpInductionRecipes only have a single
  /// incoming value, its start value.
  unsigned getNumIncoming() const override { return 1; }

  /// Returns the first defined value as TruncInst, if it is one or nullptr
  /// otherwise.
  TruncInst *getTruncInst() { return Trunc; }
  const TruncInst *getTruncInst() const { return Trunc; }

  /// Returns true if the induction is canonical, i.e. starting at 0 and
  /// incremented by UF * VF (= the original IV is incremented by 1) and has the
  /// same type as the canonical induction.
  bool isCanonical() const;

  /// Returns the scalar type of the induction.
  Type *getScalarType() const {
    return Trunc ? Trunc->getType()
                 : getStartValue()->getLiveInIRValue()->getType();
  }

  /// Returns the VPValue representing the value of this induction at
  /// the last unrolled part, if it exists. Returns itself if unrolling did not
  /// take place.
  VPValue *getLastUnrolledPartOperand() {
    return isUnrolled() ? getOperand(getNumOperands() - 1) : this;
  }
};

class VPWidenPointerInductionRecipe : public VPWidenInductionRecipe {
  bool IsScalarAfterVectorization;

public:
  /// Create a new VPWidenPointerInductionRecipe for \p Phi with start value \p
  /// Start and the number of elements unrolled \p NumUnrolledElems, typically
  /// VF*UF.
  VPWidenPointerInductionRecipe(PHINode *Phi, VPValue *Start, VPValue *Step,
                                VPValue *NumUnrolledElems,
                                const InductionDescriptor &IndDesc,
                                bool IsScalarAfterVectorization, DebugLoc DL)
      : VPWidenInductionRecipe(VPDef::VPWidenPointerInductionSC, Phi, Start,
                               Step, IndDesc, DL),
        IsScalarAfterVectorization(IsScalarAfterVectorization) {
    addOperand(NumUnrolledElems);
  }

  ~VPWidenPointerInductionRecipe() override = default;

  VPWidenPointerInductionRecipe *clone() override {
    return new VPWidenPointerInductionRecipe(
        cast<PHINode>(getUnderlyingInstr()), getOperand(0), getOperand(1),
        getOperand(2), getInductionDescriptor(), IsScalarAfterVectorization,
        getDebugLoc());
  }

  VP_CLASSOF_IMPL(VPDef::VPWidenPointerInductionSC)

  /// Generate vector values for the pointer induction.
  void execute(VPTransformState &State) override {
    llvm_unreachable("cannot execute this recipe, should be expanded via "
                     "expandVPWidenPointerInduction");
  };

  /// Returns true if only scalar values will be generated.
  bool onlyScalarsGenerated(bool IsScalable);

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif
};

/// A recipe for widened phis. Incoming values are operands of the recipe and
/// their operand index corresponds to the incoming predecessor block. If the
/// recipe is placed in an entry block to a (non-replicate) region, it must have
/// exactly 2 incoming values, the first from the predecessor of the region and
/// the second from the exiting block of the region.
class LLVM_ABI_FOR_TEST VPWidenPHIRecipe : public VPSingleDefRecipe,
                                           public VPPhiAccessors {
  /// Name to use for the generated IR instruction for the widened phi.
  std::string Name;

protected:
  const VPRecipeBase *getAsRecipe() const override { return this; }

public:
  /// Create a new VPWidenPHIRecipe for \p Phi with start value \p Start and
  /// debug location \p DL.
  VPWidenPHIRecipe(PHINode *Phi, VPValue *Start = nullptr, DebugLoc DL = {},
                   const Twine &Name = "")
      : VPSingleDefRecipe(VPDef::VPWidenPHISC, ArrayRef<VPValue *>(), Phi, DL),
        Name(Name.str()) {
    if (Start)
      addOperand(Start);
  }

  VPWidenPHIRecipe *clone() override {
    auto *C = new VPWidenPHIRecipe(cast<PHINode>(getUnderlyingValue()),
                                   getOperand(0), getDebugLoc(), Name);
    for (VPValue *Op : llvm::drop_begin(operands()))
      C->addOperand(Op);
    return C;
  }

  ~VPWidenPHIRecipe() override = default;

  VP_CLASSOF_IMPL(VPDef::VPWidenPHISC)

  /// Generate the phi/select nodes.
  void execute(VPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif
};

/// A recipe for handling first-order recurrence phis. The start value is the
/// first operand of the recipe and the incoming value from the backedge is the
/// second operand.
struct VPFirstOrderRecurrencePHIRecipe : public VPHeaderPHIRecipe {
  VPFirstOrderRecurrencePHIRecipe(PHINode *Phi, VPValue &Start)
      : VPHeaderPHIRecipe(VPDef::VPFirstOrderRecurrencePHISC, Phi, &Start) {}

  VP_CLASSOF_IMPL(VPDef::VPFirstOrderRecurrencePHISC)

  VPFirstOrderRecurrencePHIRecipe *clone() override {
    return new VPFirstOrderRecurrencePHIRecipe(
        cast<PHINode>(getUnderlyingInstr()), *getOperand(0));
  }

  void execute(VPTransformState &State) override;

  /// Return the cost of this first-order recurrence phi recipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return Op == getStartValue();
  }
};

/// A recipe for handling reduction phis. The start value is the first operand
/// of the recipe and the incoming value from the backedge is the second
/// operand.
class VPReductionPHIRecipe : public VPHeaderPHIRecipe,
                             public VPUnrollPartAccessor<2> {
  /// The recurrence kind of the reduction.
  const RecurKind Kind;

  /// The phi is part of an in-loop reduction.
  bool IsInLoop;

  /// The phi is part of an ordered reduction. Requires IsInLoop to be true.
  bool IsOrdered;

  /// When expanding the reduction PHI, the plan's VF element count is divided
  /// by this factor to form the reduction phi's VF.
  unsigned VFScaleFactor = 1;

public:
  /// Create a new VPReductionPHIRecipe for the reduction \p Phi.
  VPReductionPHIRecipe(PHINode *Phi, RecurKind Kind, VPValue &Start,
                       bool IsInLoop = false, bool IsOrdered = false,
                       unsigned VFScaleFactor = 1)
      : VPHeaderPHIRecipe(VPDef::VPReductionPHISC, Phi, &Start), Kind(Kind),
        IsInLoop(IsInLoop), IsOrdered(IsOrdered), VFScaleFactor(VFScaleFactor) {
    assert((!IsOrdered || IsInLoop) && "IsOrdered requires IsInLoop");
  }

  ~VPReductionPHIRecipe() override = default;

  VPReductionPHIRecipe *clone() override {
    auto *R = new VPReductionPHIRecipe(
        dyn_cast_or_null<PHINode>(getUnderlyingValue()), getRecurrenceKind(),
        *getOperand(0), IsInLoop, IsOrdered, VFScaleFactor);
    R->addOperand(getBackedgeValue());
    return R;
  }

  VP_CLASSOF_IMPL(VPDef::VPReductionPHISC)

  /// Generate the phi/select nodes.
  void execute(VPTransformState &State) override;

  /// Get the factor that the VF of this recipe's output should be scaled by.
  unsigned getVFScaleFactor() const { return VFScaleFactor; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  /// Returns the number of incoming values, also number of incoming blocks.
  /// Note that at the moment, VPWidenPointerInductionRecipe only has a single
  /// incoming value, its start value.
  unsigned getNumIncoming() const override { return 2; }

  /// Returns the recurrence kind of the reduction.
  RecurKind getRecurrenceKind() const { return Kind; }

  /// Returns true, if the phi is part of an ordered reduction.
  bool isOrdered() const { return IsOrdered; }

  /// Returns true, if the phi is part of an in-loop reduction.
  bool isInLoop() const { return IsInLoop; }

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return isOrdered() || isInLoop();
  }
};

/// A recipe for vectorizing a phi-node as a sequence of mask-based select
/// instructions.
class LLVM_ABI_FOR_TEST VPBlendRecipe : public VPSingleDefRecipe {
public:
  /// The blend operation is a User of the incoming values and of their
  /// respective masks, ordered [I0, M0, I1, M1, I2, M2, ...]. Note that M0 can
  /// be omitted (implied by passing an odd number of operands) in which case
  /// all other incoming values are merged into it.
  VPBlendRecipe(PHINode *Phi, ArrayRef<VPValue *> Operands, DebugLoc DL)
      : VPSingleDefRecipe(VPDef::VPBlendSC, Operands, Phi, DL) {
    assert(Operands.size() > 0 && "Expected at least one operand!");
  }

  VPBlendRecipe *clone() override {
    SmallVector<VPValue *> Ops(operands());
    return new VPBlendRecipe(cast_or_null<PHINode>(getUnderlyingValue()), Ops,
                             getDebugLoc());
  }

  VP_CLASSOF_IMPL(VPDef::VPBlendSC)

  /// A normalized blend is one that has an odd number of operands, whereby the
  /// first operand does not have an associated mask.
  bool isNormalized() const { return getNumOperands() % 2; }

  /// Return the number of incoming values, taking into account when normalized
  /// the first incoming value will have no mask.
  unsigned getNumIncomingValues() const {
    return (getNumOperands() + isNormalized()) / 2;
  }

  /// Return incoming value number \p Idx.
  VPValue *getIncomingValue(unsigned Idx) const {
    return Idx == 0 ? getOperand(0) : getOperand(Idx * 2 - isNormalized());
  }

  /// Return mask number \p Idx.
  VPValue *getMask(unsigned Idx) const {
    assert((Idx > 0 || !isNormalized()) && "First index has no mask!");
    return Idx == 0 ? getOperand(1) : getOperand(Idx * 2 + !isNormalized());
  }

  void execute(VPTransformState &State) override {
    llvm_unreachable("VPBlendRecipe should be expanded by simplifyBlends");
  }

  /// Return the cost of this VPWidenMemoryRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    // Recursing through Blend recipes only, must terminate at header phi's the
    // latest.
    return all_of(users(),
                  [this](VPUser *U) { return U->onlyFirstLaneUsed(this); });
  }
};

/// VPInterleaveRecipe is a recipe for transforming an interleave group of load
/// or stores into one wide load/store and shuffles. The first operand of a
/// VPInterleave recipe is the address, followed by the stored values, followed
/// by an optional mask.
class LLVM_ABI_FOR_TEST VPInterleaveRecipe : public VPRecipeBase,
                                             public VPIRMetadata {
  const InterleaveGroup<Instruction> *IG;

  /// Indicates if the interleave group is in a conditional block and requires a
  /// mask.
  bool HasMask = false;

  /// Indicates if gaps between members of the group need to be masked out or if
  /// unusued gaps can be loaded speculatively.
  bool NeedsMaskForGaps = false;

public:
  VPInterleaveRecipe(const InterleaveGroup<Instruction> *IG, VPValue *Addr,
                     ArrayRef<VPValue *> StoredValues, VPValue *Mask,
                     bool NeedsMaskForGaps, const VPIRMetadata &MD, DebugLoc DL)
      : VPRecipeBase(VPDef::VPInterleaveSC, {Addr}, DL), VPIRMetadata(MD),
        IG(IG), NeedsMaskForGaps(NeedsMaskForGaps) {
    // TODO: extend the masked interleaved-group support to reversed access.
    assert((!Mask || !IG->isReverse()) &&
           "Reversed masked interleave-group not supported.");
    for (unsigned I = 0; I < IG->getFactor(); ++I)
      if (Instruction *Inst = IG->getMember(I)) {
        if (Inst->getType()->isVoidTy())
          continue;
        new VPValue(Inst, this);
      }

    for (auto *SV : StoredValues)
      addOperand(SV);
    if (Mask) {
      HasMask = true;
      addOperand(Mask);
    }
  }
  ~VPInterleaveRecipe() override = default;

  VPInterleaveRecipe *clone() override {
    return new VPInterleaveRecipe(IG, getAddr(), getStoredValues(), getMask(),
                                  NeedsMaskForGaps, *this, getDebugLoc());
  }

  VP_CLASSOF_IMPL(VPDef::VPInterleaveSC)

  /// Return the address accessed by this recipe.
  VPValue *getAddr() const {
    return getOperand(0); // Address is the 1st, mandatory operand.
  }

  /// Return the mask used by this recipe. Note that a full mask is represented
  /// by a nullptr.
  VPValue *getMask() const {
    // Mask is optional and therefore the last, currently 2nd operand.
    return HasMask ? getOperand(getNumOperands() - 1) : nullptr;
  }

  /// Return the VPValues stored by this interleave group. If it is a load
  /// interleave group, return an empty ArrayRef.
  ArrayRef<VPValue *> getStoredValues() const {
    // The first operand is the address, followed by the stored values, followed
    // by an optional mask.
    return ArrayRef<VPValue *>(op_begin(), getNumOperands())
        .slice(1, getNumStoreOperands());
  }

  /// Generate the wide load or store, and shuffles.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPInterleaveRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  const InterleaveGroup<Instruction> *getInterleaveGroup() { return IG; }

  /// Returns the number of stored operands of this interleave group. Returns 0
  /// for load interleave groups.
  unsigned getNumStoreOperands() const {
    return getNumOperands() - (HasMask ? 2 : 1);
  }

  /// The recipe only uses the first lane of the address.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return Op == getAddr() && !llvm::is_contained(getStoredValues(), Op);
  }

  Instruction *getInsertPos() const { return IG->getInsertPos(); }
};

/// A recipe to represent inloop reduction operations, performing a reduction on
/// a vector operand into a scalar value, and adding the result to a chain.
/// The Operands are {ChainOp, VecOp, [Condition]}.
class LLVM_ABI_FOR_TEST VPReductionRecipe : public VPRecipeWithIRFlags {
  /// The recurrence kind for the reduction in question.
  RecurKind RdxKind;
  bool IsOrdered;
  /// Whether the reduction is conditional.
  bool IsConditional = false;

protected:
  VPReductionRecipe(const unsigned char SC, RecurKind RdxKind,
                    FastMathFlags FMFs, Instruction *I,
                    ArrayRef<VPValue *> Operands, VPValue *CondOp,
                    bool IsOrdered, DebugLoc DL)
      : VPRecipeWithIRFlags(SC, Operands, FMFs, DL), RdxKind(RdxKind),
        IsOrdered(IsOrdered) {
    if (CondOp) {
      IsConditional = true;
      addOperand(CondOp);
    }
    setUnderlyingValue(I);
  }

public:
  VPReductionRecipe(RecurKind RdxKind, FastMathFlags FMFs, Instruction *I,
                    VPValue *ChainOp, VPValue *VecOp, VPValue *CondOp,
                    bool IsOrdered, DebugLoc DL = {})
      : VPReductionRecipe(VPDef::VPReductionSC, RdxKind, FMFs, I,
                          ArrayRef<VPValue *>({ChainOp, VecOp}), CondOp,
                          IsOrdered, DL) {}

  VPReductionRecipe(const RecurKind RdxKind, FastMathFlags FMFs,
                    VPValue *ChainOp, VPValue *VecOp, VPValue *CondOp,
                    bool IsOrdered, DebugLoc DL = {})
      : VPReductionRecipe(VPDef::VPReductionSC, RdxKind, FMFs, nullptr,
                          ArrayRef<VPValue *>({ChainOp, VecOp}), CondOp,
                          IsOrdered, DL) {}

  ~VPReductionRecipe() override = default;

  VPReductionRecipe *clone() override {
    return new VPReductionRecipe(RdxKind, getFastMathFlags(),
                                 getUnderlyingInstr(), getChainOp(), getVecOp(),
                                 getCondOp(), IsOrdered, getDebugLoc());
  }

  static inline bool classof(const VPRecipeBase *R) {
    return R->getVPDefID() == VPRecipeBase::VPReductionSC ||
           R->getVPDefID() == VPRecipeBase::VPReductionEVLSC;
  }

  static inline bool classof(const VPUser *U) {
    auto *R = dyn_cast<VPRecipeBase>(U);
    return R && classof(R);
  }

  /// Generate the reduction in the loop.
  void execute(VPTransformState &State) override;

  /// Return the cost of VPReductionRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  /// Return the recurrence kind for the in-loop reduction.
  RecurKind getRecurrenceKind() const { return RdxKind; }
  /// Return true if the in-loop reduction is ordered.
  bool isOrdered() const { return IsOrdered; };
  /// Return true if the in-loop reduction is conditional.
  bool isConditional() const { return IsConditional; };
  /// The VPValue of the scalar Chain being accumulated.
  VPValue *getChainOp() const { return getOperand(0); }
  /// The VPValue of the vector value to be reduced.
  VPValue *getVecOp() const { return getOperand(1); }
  /// The VPValue of the condition for the block.
  VPValue *getCondOp() const {
    return isConditional() ? getOperand(getNumOperands() - 1) : nullptr;
  }
};

/// A recipe for forming partial reductions. In the loop, an accumulator and
/// vector operand are added together and passed to the next iteration as the
/// next accumulator. After the loop body, the accumulator is reduced to a
/// scalar value.
class VPPartialReductionRecipe : public VPReductionRecipe {
  unsigned Opcode;

  /// The divisor by which the VF of this recipe's output should be divided
  /// during execution.
  unsigned VFScaleFactor;

public:
  VPPartialReductionRecipe(Instruction *ReductionInst, VPValue *Op0,
                           VPValue *Op1, VPValue *Cond, unsigned VFScaleFactor)
      : VPPartialReductionRecipe(ReductionInst->getOpcode(), Op0, Op1, Cond,
                                 VFScaleFactor, ReductionInst) {}
  VPPartialReductionRecipe(unsigned Opcode, VPValue *Op0, VPValue *Op1,
                           VPValue *Cond, unsigned ScaleFactor,
                           Instruction *ReductionInst = nullptr)
      : VPReductionRecipe(VPDef::VPPartialReductionSC, RecurKind::Add,
                          FastMathFlags(), ReductionInst,
                          ArrayRef<VPValue *>({Op0, Op1}), Cond, false, {}),
        Opcode(Opcode), VFScaleFactor(ScaleFactor) {
    [[maybe_unused]] auto *AccumulatorRecipe =
        getChainOp()->getDefiningRecipe();
    assert((isa<VPReductionPHIRecipe>(AccumulatorRecipe) ||
            isa<VPPartialReductionRecipe>(AccumulatorRecipe)) &&
           "Unexpected operand order for partial reduction recipe");
  }
  ~VPPartialReductionRecipe() override = default;

  VPPartialReductionRecipe *clone() override {
    return new VPPartialReductionRecipe(Opcode, getOperand(0), getOperand(1),
                                        getCondOp(), VFScaleFactor,
                                        getUnderlyingInstr());
  }

  VP_CLASSOF_IMPL(VPDef::VPPartialReductionSC)

  /// Generate the reduction in the loop.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPPartialReductionRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

  /// Get the binary op's opcode.
  unsigned getOpcode() const { return Opcode; }

  /// Get the factor that the VF of this recipe's output should be scaled by.
  unsigned getVFScaleFactor() const { return VFScaleFactor; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif
};

/// A recipe to represent inloop reduction operations with vector-predication
/// intrinsics, performing a reduction on a vector operand with the explicit
/// vector length (EVL) into a scalar value, and adding the result to a chain.
/// The Operands are {ChainOp, VecOp, EVL, [Condition]}.
class LLVM_ABI_FOR_TEST VPReductionEVLRecipe : public VPReductionRecipe {
public:
  VPReductionEVLRecipe(VPReductionRecipe &R, VPValue &EVL, VPValue *CondOp,
                       DebugLoc DL = {})
      : VPReductionRecipe(
            VPDef::VPReductionEVLSC, R.getRecurrenceKind(),
            R.getFastMathFlags(),
            cast_or_null<Instruction>(R.getUnderlyingValue()),
            ArrayRef<VPValue *>({R.getChainOp(), R.getVecOp(), &EVL}), CondOp,
            R.isOrdered(), DL) {}

  ~VPReductionEVLRecipe() override = default;

  VPReductionEVLRecipe *clone() override {
    llvm_unreachable("cloning not implemented yet");
  }

  VP_CLASSOF_IMPL(VPDef::VPReductionEVLSC)

  /// Generate the reduction in the loop
  void execute(VPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  /// The VPValue of the explicit vector length.
  VPValue *getEVL() const { return getOperand(2); }

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return Op == getEVL();
  }
};

/// VPReplicateRecipe replicates a given instruction producing multiple scalar
/// copies of the original scalar type, one per lane, instead of producing a
/// single copy of widened type for all lanes. If the instruction is known to be
/// a single scalar, only one copy, per lane zero, will be generated.
class LLVM_ABI_FOR_TEST VPReplicateRecipe : public VPRecipeWithIRFlags,
                                            public VPIRMetadata {
  /// Indicator if only a single replica per lane is needed.
  bool IsSingleScalar;

  /// Indicator if the replicas are also predicated.
  bool IsPredicated;

public:
  VPReplicateRecipe(Instruction *I, ArrayRef<VPValue *> Operands,
                    bool IsSingleScalar, VPValue *Mask = nullptr,
                    VPIRMetadata Metadata = {})
      : VPRecipeWithIRFlags(VPDef::VPReplicateSC, Operands, *I),
        VPIRMetadata(Metadata), IsSingleScalar(IsSingleScalar),
        IsPredicated(Mask) {
    if (Mask)
      addOperand(Mask);
  }

  ~VPReplicateRecipe() override = default;

  VPReplicateRecipe *clone() override {
    auto *Copy =
        new VPReplicateRecipe(getUnderlyingInstr(), operands(), IsSingleScalar,
                              isPredicated() ? getMask() : nullptr, *this);
    Copy->transferFlags(*this);
    return Copy;
  }

  VP_CLASSOF_IMPL(VPDef::VPReplicateSC)

  /// Generate replicas of the desired Ingredient. Replicas will be generated
  /// for all parts and lanes unless a specific part and lane are specified in
  /// the \p State.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPReplicateRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  bool isSingleScalar() const { return IsSingleScalar; }

  bool isPredicated() const { return IsPredicated; }

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return isSingleScalar();
  }

  /// Returns true if the recipe uses scalars of operand \p Op.
  bool usesScalars(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }

  /// Returns true if the recipe is used by a widened recipe via an intervening
  /// VPPredInstPHIRecipe. In this case, the scalar values should also be packed
  /// in a vector.
  bool shouldPack() const;

  /// Return the mask of a predicated VPReplicateRecipe.
  VPValue *getMask() {
    assert(isPredicated() && "Trying to get the mask of a unpredicated recipe");
    return getOperand(getNumOperands() - 1);
  }

  unsigned getOpcode() const { return getUnderlyingInstr()->getOpcode(); }
};

/// A recipe for generating conditional branches on the bits of a mask.
class LLVM_ABI_FOR_TEST VPBranchOnMaskRecipe : public VPRecipeBase {
public:
  VPBranchOnMaskRecipe(VPValue *BlockInMask, DebugLoc DL)
      : VPRecipeBase(VPDef::VPBranchOnMaskSC, {BlockInMask}, DL) {}

  VPBranchOnMaskRecipe *clone() override {
    return new VPBranchOnMaskRecipe(getOperand(0), getDebugLoc());
  }

  VP_CLASSOF_IMPL(VPDef::VPBranchOnMaskSC)

  /// Generate the extraction of the appropriate bit from the block mask and the
  /// conditional branch.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPBranchOnMaskRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override {
    O << Indent << "BRANCH-ON-MASK ";
    printOperands(O, SlotTracker);
  }
#endif

  /// Returns true if the recipe uses scalars of operand \p Op.
  bool usesScalars(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }
};

/// A recipe to combine multiple recipes into a single 'expression' recipe,
/// which should be considered a single entity for cost-modeling and transforms.
/// The recipe needs to be 'decomposed', i.e. replaced by its individual
/// expression recipes, before execute. The individual expression recipes are
/// completely disconnected from the def-use graph of other recipes not part of
/// the expression. Def-use edges between pairs of expression recipes remain
/// intact, whereas every edge between an expression recipe and a recipe outside
/// the expression is elevated to connect the non-expression recipe with the
/// VPExpressionRecipe itself.
class VPExpressionRecipe : public VPSingleDefRecipe {
  /// Recipes included in this VPExpressionRecipe.
  SmallVector<VPSingleDefRecipe *> ExpressionRecipes;

  /// Temporary VPValues used for external operands of the expression, i.e.
  /// operands not defined by recipes in the expression.
  SmallVector<VPValue *> LiveInPlaceholders;

  enum class ExpressionTypes {
    /// Represents an inloop extended reduction operation, performing a
    /// reduction on an extended vector operand into a scalar value, and adding
    /// the result to a chain.
    ExtendedReduction,
    /// Represent an inloop multiply-accumulate reduction, multiplying the
    /// extended vector operands, performing a reduction.add on the result, and
    /// adding the scalar result to a chain.
    ExtMulAccReduction,
    /// Represent an inloop multiply-accumulate reduction, multiplying the
    /// vector operands, performing a reduction.add on the result, and adding
    /// the scalar result to a chain.
    MulAccReduction,
  };

  /// Type of the expression.
  ExpressionTypes ExpressionType;

  /// Construct a new VPExpressionRecipe by internalizing recipes in \p
  /// ExpressionRecipes. External operands (i.e. not defined by another recipe
  /// in the expression) are replaced by temporary VPValues and the original
  /// operands are transferred to the VPExpressionRecipe itself. Clone recipes
  /// as needed (excluding last) to ensure they are only used by other recipes
  /// in the expression.
  VPExpressionRecipe(ExpressionTypes ExpressionType,
                     ArrayRef<VPSingleDefRecipe *> ExpressionRecipes);

public:
  VPExpressionRecipe(VPWidenCastRecipe *Ext, VPReductionRecipe *Red)
      : VPExpressionRecipe(ExpressionTypes::ExtendedReduction, {Ext, Red}) {}
  VPExpressionRecipe(VPWidenRecipe *Mul, VPReductionRecipe *Red)
      : VPExpressionRecipe(ExpressionTypes::MulAccReduction, {Mul, Red}) {}
  VPExpressionRecipe(VPWidenCastRecipe *Ext0, VPWidenCastRecipe *Ext1,
                     VPWidenRecipe *Mul, VPReductionRecipe *Red)
      : VPExpressionRecipe(ExpressionTypes::ExtMulAccReduction,
                           {Ext0, Ext1, Mul, Red}) {}

  ~VPExpressionRecipe() override {
    for (auto *R : reverse(ExpressionRecipes))
      delete R;
    for (VPValue *T : LiveInPlaceholders)
      delete T;
  }

  VP_CLASSOF_IMPL(VPDef::VPExpressionSC)

  VPExpressionRecipe *clone() override {
    assert(!ExpressionRecipes.empty() && "empty expressions should be removed");
    SmallVector<VPSingleDefRecipe *> NewExpressiondRecipes;
    for (auto *R : ExpressionRecipes)
      NewExpressiondRecipes.push_back(R->clone());
    for (auto *New : NewExpressiondRecipes) {
      for (const auto &[Idx, Old] : enumerate(ExpressionRecipes))
        New->replaceUsesOfWith(Old, NewExpressiondRecipes[Idx]);
      // Update placeholder operands in the cloned recipe to use the external
      // operands, to be internalized when the cloned expression is constructed.
      for (const auto &[Placeholder, OutsideOp] :
           zip(LiveInPlaceholders, operands()))
        New->replaceUsesOfWith(Placeholder, OutsideOp);
    }
    return new VPExpressionRecipe(ExpressionType, NewExpressiondRecipes);
  }

  /// Return the VPValue to use to infer the result type of the recipe.
  VPValue *getOperandOfResultType() const {
    unsigned OpIdx =
        cast<VPReductionRecipe>(ExpressionRecipes.back())->isConditional() ? 2
                                                                           : 1;
    return getOperand(getNumOperands() - OpIdx);
  }

  /// Insert the recipes of the expression back into the VPlan, directly before
  /// the current recipe. Leaves the expression recipe empty, which must be
  /// removed before codegen.
  void decompose();

  /// Method for generating code, must not be called as this recipe is abstract.
  void execute(VPTransformState &State) override {
    llvm_unreachable("recipe must be removed before execute");
  }

  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  /// Returns true if this expression contains recipes that may read from or
  /// write to memory.
  bool mayReadOrWriteMemory() const;

  /// Returns true if this expression contains recipes that may have side
  /// effects.
  bool mayHaveSideEffects() const;
};

/// VPPredInstPHIRecipe is a recipe for generating the phi nodes needed when
/// control converges back from a Branch-on-Mask. The phi nodes are needed in
/// order to merge values that are set under such a branch and feed their uses.
/// The phi nodes can be scalar or vector depending on the users of the value.
/// This recipe works in concert with VPBranchOnMaskRecipe.
class LLVM_ABI_FOR_TEST VPPredInstPHIRecipe : public VPSingleDefRecipe {
public:
  /// Construct a VPPredInstPHIRecipe given \p PredInst whose value needs a phi
  /// nodes after merging back from a Branch-on-Mask.
  VPPredInstPHIRecipe(VPValue *PredV, DebugLoc DL)
      : VPSingleDefRecipe(VPDef::VPPredInstPHISC, PredV, DL) {}
  ~VPPredInstPHIRecipe() override = default;

  VPPredInstPHIRecipe *clone() override {
    return new VPPredInstPHIRecipe(getOperand(0), getDebugLoc());
  }

  VP_CLASSOF_IMPL(VPDef::VPPredInstPHISC)

  /// Generates phi nodes for live-outs (from a replicate region) as needed to
  /// retain SSA form.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPPredInstPHIRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override {
    // TODO: Compute accurate cost after retiring the legacy cost model.
    return 0;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  /// Returns true if the recipe uses scalars of operand \p Op.
  bool usesScalars(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }
};

/// A common base class for widening memory operations. An optional mask can be
/// provided as the last operand.
class LLVM_ABI_FOR_TEST VPWidenMemoryRecipe : public VPRecipeBase,
                                              public VPIRMetadata {
protected:
  Instruction &Ingredient;

  /// Whether the accessed addresses are consecutive.
  bool Consecutive;

  /// Whether the consecutive accessed addresses are in reverse order.
  bool Reverse;

  /// Whether the memory access is masked.
  bool IsMasked = false;

  void setMask(VPValue *Mask) {
    assert(!IsMasked && "cannot re-set mask");
    if (!Mask)
      return;
    addOperand(Mask);
    IsMasked = true;
  }

  VPWidenMemoryRecipe(const char unsigned SC, Instruction &I,
                      std::initializer_list<VPValue *> Operands,
                      bool Consecutive, bool Reverse,
                      const VPIRMetadata &Metadata, DebugLoc DL)
      : VPRecipeBase(SC, Operands, DL), VPIRMetadata(Metadata), Ingredient(I),
        Consecutive(Consecutive), Reverse(Reverse) {
    assert((Consecutive || !Reverse) && "Reverse implies consecutive");
  }

public:
  VPWidenMemoryRecipe *clone() override {
    llvm_unreachable("cloning not supported");
  }

  static inline bool classof(const VPRecipeBase *R) {
    return R->getVPDefID() == VPRecipeBase::VPWidenLoadSC ||
           R->getVPDefID() == VPRecipeBase::VPWidenStoreSC ||
           R->getVPDefID() == VPRecipeBase::VPWidenLoadEVLSC ||
           R->getVPDefID() == VPRecipeBase::VPWidenStoreEVLSC;
  }

  static inline bool classof(const VPUser *U) {
    auto *R = dyn_cast<VPRecipeBase>(U);
    return R && classof(R);
  }

  /// Return whether the loaded-from / stored-to addresses are consecutive.
  bool isConsecutive() const { return Consecutive; }

  /// Return whether the consecutive loaded/stored addresses are in reverse
  /// order.
  bool isReverse() const { return Reverse; }

  /// Return the address accessed by this recipe.
  VPValue *getAddr() const { return getOperand(0); }

  /// Returns true if the recipe is masked.
  bool isMasked() const { return IsMasked; }

  /// Return the mask used by this recipe. Note that a full mask is represented
  /// by a nullptr.
  VPValue *getMask() const {
    // Mask is optional and therefore the last operand.
    return isMasked() ? getOperand(getNumOperands() - 1) : nullptr;
  }

  /// Generate the wide load/store.
  void execute(VPTransformState &State) override {
    llvm_unreachable("VPWidenMemoryRecipe should not be instantiated.");
  }

  /// Return the cost of this VPWidenMemoryRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

  Instruction &getIngredient() const { return Ingredient; }
};

/// A recipe for widening load operations, using the address to load from and an
/// optional mask.
struct LLVM_ABI_FOR_TEST VPWidenLoadRecipe final : public VPWidenMemoryRecipe,
                                                   public VPValue {
  VPWidenLoadRecipe(LoadInst &Load, VPValue *Addr, VPValue *Mask,
                    bool Consecutive, bool Reverse,
                    const VPIRMetadata &Metadata, DebugLoc DL)
      : VPWidenMemoryRecipe(VPDef::VPWidenLoadSC, Load, {Addr}, Consecutive,
                            Reverse, Metadata, DL),
        VPValue(this, &Load) {
    setMask(Mask);
  }

  VPWidenLoadRecipe *clone() override {
    return new VPWidenLoadRecipe(cast<LoadInst>(Ingredient), getAddr(),
                                 getMask(), Consecutive, Reverse, *this,
                                 getDebugLoc());
  }

  VP_CLASSOF_IMPL(VPDef::VPWidenLoadSC);

  /// Generate a wide load or gather.
  void execute(VPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    // Widened, consecutive loads operations only demand the first lane of
    // their address.
    return Op == getAddr() && isConsecutive();
  }
};

/// A recipe for widening load operations with vector-predication intrinsics,
/// using the address to load from, the explicit vector length and an optional
/// mask.
struct VPWidenLoadEVLRecipe final : public VPWidenMemoryRecipe, public VPValue {
  VPWidenLoadEVLRecipe(VPWidenLoadRecipe &L, VPValue *Addr, VPValue &EVL,
                       VPValue *Mask)
      : VPWidenMemoryRecipe(VPDef::VPWidenLoadEVLSC, L.getIngredient(),
                            {Addr, &EVL}, L.isConsecutive(), L.isReverse(), L,
                            L.getDebugLoc()),
        VPValue(this, &getIngredient()) {
    setMask(Mask);
  }

  VP_CLASSOF_IMPL(VPDef::VPWidenLoadEVLSC)

  /// Return the EVL operand.
  VPValue *getEVL() const { return getOperand(1); }

  /// Generate the wide load or gather.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPWidenLoadEVLRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    // Widened loads only demand the first lane of EVL and consecutive loads
    // only demand the first lane of their address.
    return Op == getEVL() || (Op == getAddr() && isConsecutive());
  }
};

/// A recipe for widening store operations, using the stored value, the address
/// to store to and an optional mask.
struct LLVM_ABI_FOR_TEST VPWidenStoreRecipe final : public VPWidenMemoryRecipe {
  VPWidenStoreRecipe(StoreInst &Store, VPValue *Addr, VPValue *StoredVal,
                     VPValue *Mask, bool Consecutive, bool Reverse,
                     const VPIRMetadata &Metadata, DebugLoc DL)
      : VPWidenMemoryRecipe(VPDef::VPWidenStoreSC, Store, {Addr, StoredVal},
                            Consecutive, Reverse, Metadata, DL) {
    setMask(Mask);
  }

  VPWidenStoreRecipe *clone() override {
    return new VPWidenStoreRecipe(cast<StoreInst>(Ingredient), getAddr(),
                                  getStoredValue(), getMask(), Consecutive,
                                  Reverse, *this, getDebugLoc());
  }

  VP_CLASSOF_IMPL(VPDef::VPWidenStoreSC);

  /// Return the value stored by this recipe.
  VPValue *getStoredValue() const { return getOperand(1); }

  /// Generate a wide store or scatter.
  void execute(VPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    // Widened, consecutive stores only demand the first lane of their address,
    // unless the same operand is also stored.
    return Op == getAddr() && isConsecutive() && Op != getStoredValue();
  }
};

/// A recipe for widening store operations with vector-predication intrinsics,
/// using the value to store, the address to store to, the explicit vector
/// length and an optional mask.
struct VPWidenStoreEVLRecipe final : public VPWidenMemoryRecipe {
  VPWidenStoreEVLRecipe(VPWidenStoreRecipe &S, VPValue *Addr, VPValue &EVL,
                        VPValue *Mask)
      : VPWidenMemoryRecipe(VPDef::VPWidenStoreEVLSC, S.getIngredient(),
                            {Addr, S.getStoredValue(), &EVL}, S.isConsecutive(),
                            S.isReverse(), S, S.getDebugLoc()) {
    setMask(Mask);
  }

  VP_CLASSOF_IMPL(VPDef::VPWidenStoreEVLSC)

  /// Return the address accessed by this recipe.
  VPValue *getStoredValue() const { return getOperand(1); }

  /// Return the EVL operand.
  VPValue *getEVL() const { return getOperand(2); }

  /// Generate the wide store or scatter.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPWidenStoreEVLRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    if (Op == getEVL()) {
      assert(getStoredValue() != Op && "unexpected store of EVL");
      return true;
    }
    // Widened, consecutive memory operations only demand the first lane of
    // their address, unless the same operand is also stored. That latter can
    // happen with opaque pointers.
    return Op == getAddr() && isConsecutive() && Op != getStoredValue();
  }
};

/// Recipe to expand a SCEV expression.
class VPExpandSCEVRecipe : public VPSingleDefRecipe {
  const SCEV *Expr;
  ScalarEvolution &SE;

public:
  VPExpandSCEVRecipe(const SCEV *Expr, ScalarEvolution &SE)
      : VPSingleDefRecipe(VPDef::VPExpandSCEVSC, {}), Expr(Expr), SE(SE) {}

  ~VPExpandSCEVRecipe() override = default;

  VPExpandSCEVRecipe *clone() override {
    return new VPExpandSCEVRecipe(Expr, SE);
  }

  VP_CLASSOF_IMPL(VPDef::VPExpandSCEVSC)

  /// Generate a canonical vector induction variable of the vector loop, with
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPExpandSCEVRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override {
    // TODO: Compute accurate cost after retiring the legacy cost model.
    return 0;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  const SCEV *getSCEV() const { return Expr; }
};

/// Canonical scalar induction phi of the vector loop. Starting at the specified
/// start value (either 0 or the resume value when vectorizing the epilogue
/// loop). VPWidenCanonicalIVRecipe represents the vector version of the
/// canonical induction variable.
class VPCanonicalIVPHIRecipe : public VPHeaderPHIRecipe {
public:
  VPCanonicalIVPHIRecipe(VPValue *StartV, DebugLoc DL)
      : VPHeaderPHIRecipe(VPDef::VPCanonicalIVPHISC, nullptr, StartV, DL) {}

  ~VPCanonicalIVPHIRecipe() override = default;

  VPCanonicalIVPHIRecipe *clone() override {
    auto *R = new VPCanonicalIVPHIRecipe(getOperand(0), getDebugLoc());
    R->addOperand(getBackedgeValue());
    return R;
  }

  VP_CLASSOF_IMPL(VPDef::VPCanonicalIVPHISC)

  void execute(VPTransformState &State) override {
    llvm_unreachable("cannot execute this recipe, should be replaced by a "
                     "scalar phi recipe");
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  /// Returns the scalar type of the induction.
  Type *getScalarType() const {
    return getStartValue()->getLiveInIRValue()->getType();
  }

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }

  /// Returns true if the recipe only uses the first part of operand \p Op.
  bool onlyFirstPartUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }

  /// Return the cost of this VPCanonicalIVPHIRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override {
    // For now, match the behavior of the legacy cost model.
    return 0;
  }
};

/// A recipe for generating the active lane mask for the vector loop that is
/// used to predicate the vector operations.
/// TODO: It would be good to use the existing VPWidenPHIRecipe instead and
/// remove VPActiveLaneMaskPHIRecipe.
class VPActiveLaneMaskPHIRecipe : public VPHeaderPHIRecipe {
public:
  VPActiveLaneMaskPHIRecipe(VPValue *StartMask, DebugLoc DL)
      : VPHeaderPHIRecipe(VPDef::VPActiveLaneMaskPHISC, nullptr, StartMask,
                          DL) {}

  ~VPActiveLaneMaskPHIRecipe() override = default;

  VPActiveLaneMaskPHIRecipe *clone() override {
    auto *R = new VPActiveLaneMaskPHIRecipe(getOperand(0), getDebugLoc());
    if (getNumOperands() == 2)
      R->addOperand(getOperand(1));
    return R;
  }

  VP_CLASSOF_IMPL(VPDef::VPActiveLaneMaskPHISC)

  /// Generate the active lane mask phi of the vector loop.
  void execute(VPTransformState &State) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif
};

/// A recipe for generating the phi node for the current index of elements,
/// adjusted in accordance with EVL value. It starts at the start value of the
/// canonical induction and gets incremented by EVL in each iteration of the
/// vector loop.
class VPEVLBasedIVPHIRecipe : public VPHeaderPHIRecipe {
public:
  VPEVLBasedIVPHIRecipe(VPValue *StartIV, DebugLoc DL)
      : VPHeaderPHIRecipe(VPDef::VPEVLBasedIVPHISC, nullptr, StartIV, DL) {}

  ~VPEVLBasedIVPHIRecipe() override = default;

  VPEVLBasedIVPHIRecipe *clone() override {
    llvm_unreachable("cloning not implemented yet");
  }

  VP_CLASSOF_IMPL(VPDef::VPEVLBasedIVPHISC)

  void execute(VPTransformState &State) override {
    llvm_unreachable("cannot execute this recipe, should be replaced by a "
                     "scalar phi recipe");
  }

  /// Return the cost of this VPEVLBasedIVPHIRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override {
    // For now, match the behavior of the legacy cost model.
    return 0;
  }

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif
};

/// A Recipe for widening the canonical induction variable of the vector loop.
class VPWidenCanonicalIVRecipe : public VPSingleDefRecipe,
                                 public VPUnrollPartAccessor<1> {
public:
  VPWidenCanonicalIVRecipe(VPCanonicalIVPHIRecipe *CanonicalIV)
      : VPSingleDefRecipe(VPDef::VPWidenCanonicalIVSC, {CanonicalIV}) {}

  ~VPWidenCanonicalIVRecipe() override = default;

  VPWidenCanonicalIVRecipe *clone() override {
    return new VPWidenCanonicalIVRecipe(
        cast<VPCanonicalIVPHIRecipe>(getOperand(0)));
  }

  VP_CLASSOF_IMPL(VPDef::VPWidenCanonicalIVSC)

  /// Generate a canonical vector induction variable of the vector loop, with
  /// start = {<Part*VF, Part*VF+1, ..., Part*VF+VF-1> for 0 <= Part < UF}, and
  /// step = <VF*UF, VF*UF, ..., VF*UF>.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPWidenCanonicalIVPHIRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override {
    // TODO: Compute accurate cost after retiring the legacy cost model.
    return 0;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif
};

/// A recipe for converting the input value \p IV value to the corresponding
/// value of an IV with different start and step values, using Start + IV *
/// Step.
class VPDerivedIVRecipe : public VPSingleDefRecipe {
  /// Kind of the induction.
  const InductionDescriptor::InductionKind Kind;
  /// If not nullptr, the floating point induction binary operator. Must be set
  /// for floating point inductions.
  const FPMathOperator *FPBinOp;

  /// Name to use for the generated IR instruction for the derived IV.
  std::string Name;

public:
  VPDerivedIVRecipe(const InductionDescriptor &IndDesc, VPValue *Start,
                    VPCanonicalIVPHIRecipe *CanonicalIV, VPValue *Step,
                    const Twine &Name = "")
      : VPDerivedIVRecipe(
            IndDesc.getKind(),
            dyn_cast_or_null<FPMathOperator>(IndDesc.getInductionBinOp()),
            Start, CanonicalIV, Step, Name) {}

  VPDerivedIVRecipe(InductionDescriptor::InductionKind Kind,
                    const FPMathOperator *FPBinOp, VPValue *Start, VPValue *IV,
                    VPValue *Step, const Twine &Name = "")
      : VPSingleDefRecipe(VPDef::VPDerivedIVSC, {Start, IV, Step}), Kind(Kind),
        FPBinOp(FPBinOp), Name(Name.str()) {}

  ~VPDerivedIVRecipe() override = default;

  VPDerivedIVRecipe *clone() override {
    return new VPDerivedIVRecipe(Kind, FPBinOp, getStartValue(), getOperand(1),
                                 getStepValue());
  }

  VP_CLASSOF_IMPL(VPDef::VPDerivedIVSC)

  /// Generate the transformed value of the induction at offset StartValue (1.
  /// operand) + IV (2. operand) * StepValue (3, operand).
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPDerivedIVRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override {
    // TODO: Compute accurate cost after retiring the legacy cost model.
    return 0;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  Type *getScalarType() const {
    return getStartValue()->getLiveInIRValue()->getType();
  }

  VPValue *getStartValue() const { return getOperand(0); }
  VPValue *getStepValue() const { return getOperand(2); }

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }
};

/// A recipe for handling phi nodes of integer and floating-point inductions,
/// producing their scalar values.
class LLVM_ABI_FOR_TEST VPScalarIVStepsRecipe : public VPRecipeWithIRFlags,
                                                public VPUnrollPartAccessor<3> {
  Instruction::BinaryOps InductionOpcode;

public:
  VPScalarIVStepsRecipe(VPValue *IV, VPValue *Step, VPValue *VF,
                        Instruction::BinaryOps Opcode, FastMathFlags FMFs,
                        DebugLoc DL)
      : VPRecipeWithIRFlags(VPDef::VPScalarIVStepsSC,
                            ArrayRef<VPValue *>({IV, Step, VF}), FMFs, DL),
        InductionOpcode(Opcode) {}

  VPScalarIVStepsRecipe(const InductionDescriptor &IndDesc, VPValue *IV,
                        VPValue *Step, VPValue *VF, DebugLoc DL = {})
      : VPScalarIVStepsRecipe(
            IV, Step, VF, IndDesc.getInductionOpcode(),
            dyn_cast_or_null<FPMathOperator>(IndDesc.getInductionBinOp())
                ? IndDesc.getInductionBinOp()->getFastMathFlags()
                : FastMathFlags(),
            DL) {}

  ~VPScalarIVStepsRecipe() override = default;

  VPScalarIVStepsRecipe *clone() override {
    return new VPScalarIVStepsRecipe(
        getOperand(0), getOperand(1), getOperand(2), InductionOpcode,
        hasFastMathFlags() ? getFastMathFlags() : FastMathFlags(),
        getDebugLoc());
  }

  /// Return true if this VPScalarIVStepsRecipe corresponds to part 0. Note that
  /// this is only accurate after the VPlan has been unrolled.
  bool isPart0() const { return getUnrollPart(*this) == 0; }

  VP_CLASSOF_IMPL(VPDef::VPScalarIVStepsSC)

  /// Generate the scalarized versions of the phi node as needed by their users.
  void execute(VPTransformState &State) override;

  /// Return the cost of this VPScalarIVStepsRecipe.
  InstructionCost computeCost(ElementCount VF,
                              VPCostContext &Ctx) const override {
    // TODO: Compute accurate cost after retiring the legacy cost model.
    return 0;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the recipe.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
#endif

  VPValue *getStepValue() const { return getOperand(1); }

  /// Returns true if the recipe only uses the first lane of operand \p Op.
  bool onlyFirstLaneUsed(const VPValue *Op) const override {
    assert(is_contained(operands(), Op) &&
           "Op must be an operand of the recipe");
    return true;
  }
};

/// Casting from VPRecipeBase -> VPPhiAccessors is supported for all recipe
/// types implementing VPPhiAccessors. Used by isa<> & co.
template <> struct CastIsPossible<VPPhiAccessors, const VPRecipeBase *> {
  static inline bool isPossible(const VPRecipeBase *f) {
    // TODO: include VPPredInstPHIRecipe too, once it implements VPPhiAccessors.
    return isa<VPIRPhi, VPHeaderPHIRecipe, VPWidenPHIRecipe, VPPhi>(f);
  }
};
/// Support casting from VPRecipeBase -> VPPhiAccessors, by down-casting to the
/// recipe types implementing VPPhiAccessors. Used by cast<>, dyn_cast<> & co.
template <typename SrcTy>
struct CastInfoVPPhiAccessors : public CastIsPossible<VPPhiAccessors, SrcTy> {

  using Self = CastInfo<VPPhiAccessors, SrcTy>;

  /// doCast is used by cast<>.
  static inline VPPhiAccessors *doCast(SrcTy R) {
    return const_cast<VPPhiAccessors *>([R]() -> const VPPhiAccessors * {
      switch (R->getVPDefID()) {
      case VPDef::VPInstructionSC:
        return cast<VPPhi>(R);
      case VPDef::VPIRInstructionSC:
        return cast<VPIRPhi>(R);
      case VPDef::VPWidenPHISC:
        return cast<VPWidenPHIRecipe>(R);
      default:
        return cast<VPHeaderPHIRecipe>(R);
      }
    }());
  }

  /// doCastIfPossible is used by dyn_cast<>.
  static inline VPPhiAccessors *doCastIfPossible(SrcTy f) {
    if (!Self::isPossible(f))
      return nullptr;
    return doCast(f);
  }
};
template <>
struct CastInfo<VPPhiAccessors, VPRecipeBase *>
    : CastInfoVPPhiAccessors<VPRecipeBase *> {};
template <>
struct CastInfo<VPPhiAccessors, const VPRecipeBase *>
    : CastInfoVPPhiAccessors<const VPRecipeBase *> {};

/// VPBasicBlock serves as the leaf of the Hierarchical Control-Flow Graph. It
/// holds a sequence of zero or more VPRecipe's each representing a sequence of
/// output IR instructions. All PHI-like recipes must come before any non-PHI recipes.
class LLVM_ABI_FOR_TEST VPBasicBlock : public VPBlockBase {
  friend class VPlan;

  /// Use VPlan::createVPBasicBlock to create VPBasicBlocks.
  VPBasicBlock(const Twine &Name = "", VPRecipeBase *Recipe = nullptr)
      : VPBlockBase(VPBasicBlockSC, Name.str()) {
    if (Recipe)
      appendRecipe(Recipe);
  }

public:
  using RecipeListTy = iplist<VPRecipeBase>;

protected:
  /// The VPRecipes held in the order of output instructions to generate.
  RecipeListTy Recipes;

  VPBasicBlock(const unsigned char BlockSC, const Twine &Name = "")
      : VPBlockBase(BlockSC, Name.str()) {}

public:
  ~VPBasicBlock() override {
    while (!Recipes.empty())
      Recipes.pop_back();
  }

  /// Instruction iterators...
  using iterator = RecipeListTy::iterator;
  using const_iterator = RecipeListTy::const_iterator;
  using reverse_iterator = RecipeListTy::reverse_iterator;
  using const_reverse_iterator = RecipeListTy::const_reverse_iterator;

  //===--------------------------------------------------------------------===//
  /// Recipe iterator methods
  ///
  inline iterator begin() { return Recipes.begin(); }
  inline const_iterator begin() const { return Recipes.begin(); }
  inline iterator end() { return Recipes.end(); }
  inline const_iterator end() const { return Recipes.end(); }

  inline reverse_iterator rbegin() { return Recipes.rbegin(); }
  inline const_reverse_iterator rbegin() const { return Recipes.rbegin(); }
  inline reverse_iterator rend() { return Recipes.rend(); }
  inline const_reverse_iterator rend() const { return Recipes.rend(); }

  inline size_t size() const { return Recipes.size(); }
  inline bool empty() const { return Recipes.empty(); }
  inline const VPRecipeBase &front() const { return Recipes.front(); }
  inline VPRecipeBase &front() { return Recipes.front(); }
  inline const VPRecipeBase &back() const { return Recipes.back(); }
  inline VPRecipeBase &back() { return Recipes.back(); }

  /// Returns a reference to the list of recipes.
  RecipeListTy &getRecipeList() { return Recipes; }

  /// Returns a pointer to a member of the recipe list.
  static RecipeListTy VPBasicBlock::*getSublistAccess(VPRecipeBase *) {
    return &VPBasicBlock::Recipes;
  }

  /// Method to support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const VPBlockBase *V) {
    return V->getVPBlockID() == VPBlockBase::VPBasicBlockSC ||
           V->getVPBlockID() == VPBlockBase::VPIRBasicBlockSC;
  }

  void insert(VPRecipeBase *Recipe, iterator InsertPt) {
    assert(Recipe && "No recipe to append.");
    assert(!Recipe->Parent && "Recipe already in VPlan");
    Recipe->Parent = this;
    Recipes.insert(InsertPt, Recipe);
  }

  /// Augment the existing recipes of a VPBasicBlock with an additional
  /// \p Recipe as the last recipe.
  void appendRecipe(VPRecipeBase *Recipe) { insert(Recipe, end()); }

  /// The method which generates the output IR instructions that correspond to
  /// this VPBasicBlock, thereby "executing" the VPlan.
  void execute(VPTransformState *State) override;

  /// Return the cost of this VPBasicBlock.
  InstructionCost cost(ElementCount VF, VPCostContext &Ctx) override;

  /// Return the position of the first non-phi node recipe in the block.
  iterator getFirstNonPhi();

  /// Returns an iterator range over the PHI-like recipes in the block.
  iterator_range<iterator> phis() {
    return make_range(begin(), getFirstNonPhi());
  }

  /// Split current block at \p SplitAt by inserting a new block between the
  /// current block and its successors and moving all recipes starting at
  /// SplitAt to the new block. Returns the new block.
  VPBasicBlock *splitAt(iterator SplitAt);

  VPRegionBlock *getEnclosingLoopRegion();
  const VPRegionBlock *getEnclosingLoopRegion() const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print this VPBsicBlock to \p O, prefixing all lines with \p Indent. \p
  /// SlotTracker is used to print unnamed VPValue's using consequtive numbers.
  ///
  /// Note that the numbering is applied to the whole VPlan, so printing
  /// individual blocks is consistent with the whole VPlan printing.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
  using VPBlockBase::print; // Get the print(raw_stream &O) version.
#endif

  /// If the block has multiple successors, return the branch recipe terminating
  /// the block. If there are no or only a single successor, return nullptr;
  VPRecipeBase *getTerminator();
  const VPRecipeBase *getTerminator() const;

  /// Returns true if the block is exiting it's parent region.
  bool isExiting() const;

  /// Clone the current block and it's recipes, without updating the operands of
  /// the cloned recipes.
  VPBasicBlock *clone() override;

  /// Returns the predecessor block at index \p Idx with the predecessors as per
  /// the corresponding plain CFG. If the block is an entry block to a region,
  /// the first predecessor is the single predecessor of a region, and the
  /// second predecessor is the exiting block of the region.
  const VPBasicBlock *getCFGPredecessor(unsigned Idx) const;

protected:
  /// Execute the recipes in the IR basic block \p BB.
  void executeRecipes(VPTransformState *State, BasicBlock *BB);

  /// Connect the VPBBs predecessors' in the VPlan CFG to the IR basic block
  /// generated for this VPBB.
  void connectToPredecessors(VPTransformState &State);

private:
  /// Create an IR BasicBlock to hold the output instructions generated by this
  /// VPBasicBlock, and return it. Update the CFGState accordingly.
  BasicBlock *createEmptyBasicBlock(VPTransformState &State);
};

inline const VPBasicBlock *
VPPhiAccessors::getIncomingBlock(unsigned Idx) const {
  return getAsRecipe()->getParent()->getCFGPredecessor(Idx);
}

/// A special type of VPBasicBlock that wraps an existing IR basic block.
/// Recipes of the block get added before the first non-phi instruction in the
/// wrapped block.
/// Note: At the moment, VPIRBasicBlock can only be used to wrap VPlan's
/// preheader block.
class VPIRBasicBlock : public VPBasicBlock {
  friend class VPlan;

  BasicBlock *IRBB;

  /// Use VPlan::createVPIRBasicBlock to create VPIRBasicBlocks.
  VPIRBasicBlock(BasicBlock *IRBB)
      : VPBasicBlock(VPIRBasicBlockSC,
                     (Twine("ir-bb<") + IRBB->getName() + Twine(">")).str()),
        IRBB(IRBB) {}

public:
  ~VPIRBasicBlock() override {}

  static inline bool classof(const VPBlockBase *V) {
    return V->getVPBlockID() == VPBlockBase::VPIRBasicBlockSC;
  }

  /// The method which generates the output IR instructions that correspond to
  /// this VPBasicBlock, thereby "executing" the VPlan.
  void execute(VPTransformState *State) override;

  VPIRBasicBlock *clone() override;

  BasicBlock *getIRBasicBlock() const { return IRBB; }
};

/// VPRegionBlock represents a collection of VPBasicBlocks and VPRegionBlocks
/// which form a Single-Entry-Single-Exiting subgraph of the output IR CFG.
/// A VPRegionBlock may indicate that its contents are to be replicated several
/// times. This is designed to support predicated scalarization, in which a
/// scalar if-then code structure needs to be generated VF * UF times. Having
/// this replication indicator helps to keep a single model for multiple
/// candidate VF's. The actual replication takes place only once the desired VF
/// and UF have been determined.
class LLVM_ABI_FOR_TEST VPRegionBlock : public VPBlockBase {
  friend class VPlan;

  /// Hold the Single Entry of the SESE region modelled by the VPRegionBlock.
  VPBlockBase *Entry;

  /// Hold the Single Exiting block of the SESE region modelled by the
  /// VPRegionBlock.
  VPBlockBase *Exiting;

  /// An indicator whether this region is to generate multiple replicated
  /// instances of output IR corresponding to its VPBlockBases.
  bool IsReplicator;

  /// Use VPlan::createVPRegionBlock to create VPRegionBlocks.
  VPRegionBlock(VPBlockBase *Entry, VPBlockBase *Exiting,
                const std::string &Name = "", bool IsReplicator = false)
      : VPBlockBase(VPRegionBlockSC, Name), Entry(Entry), Exiting(Exiting),
        IsReplicator(IsReplicator) {
    assert(Entry->getPredecessors().empty() && "Entry block has predecessors.");
    assert(Exiting->getSuccessors().empty() && "Exit block has successors.");
    Entry->setParent(this);
    Exiting->setParent(this);
  }
  VPRegionBlock(const std::string &Name = "", bool IsReplicator = false)
      : VPBlockBase(VPRegionBlockSC, Name), Entry(nullptr), Exiting(nullptr),
        IsReplicator(IsReplicator) {}

public:
  ~VPRegionBlock() override {}

  /// Method to support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const VPBlockBase *V) {
    return V->getVPBlockID() == VPBlockBase::VPRegionBlockSC;
  }

  const VPBlockBase *getEntry() const { return Entry; }
  VPBlockBase *getEntry() { return Entry; }

  /// Set \p EntryBlock as the entry VPBlockBase of this VPRegionBlock. \p
  /// EntryBlock must have no predecessors.
  void setEntry(VPBlockBase *EntryBlock) {
    assert(EntryBlock->getPredecessors().empty() &&
           "Entry block cannot have predecessors.");
    Entry = EntryBlock;
    EntryBlock->setParent(this);
  }

  const VPBlockBase *getExiting() const { return Exiting; }
  VPBlockBase *getExiting() { return Exiting; }

  /// Set \p ExitingBlock as the exiting VPBlockBase of this VPRegionBlock. \p
  /// ExitingBlock must have no successors.
  void setExiting(VPBlockBase *ExitingBlock) {
    assert(ExitingBlock->getSuccessors().empty() &&
           "Exit block cannot have successors.");
    Exiting = ExitingBlock;
    ExitingBlock->setParent(this);
  }

  /// Returns the pre-header VPBasicBlock of the loop region.
  VPBasicBlock *getPreheaderVPBB() {
    assert(!isReplicator() && "should only get pre-header of loop regions");
    return getSinglePredecessor()->getExitingBasicBlock();
  }

  /// An indicator whether this region is to generate multiple replicated
  /// instances of output IR corresponding to its VPBlockBases.
  bool isReplicator() const { return IsReplicator; }

  /// The method which generates the output IR instructions that correspond to
  /// this VPRegionBlock, thereby "executing" the VPlan.
  void execute(VPTransformState *State) override;

  // Return the cost of this region.
  InstructionCost cost(ElementCount VF, VPCostContext &Ctx) override;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print this VPRegionBlock to \p O (recursively), prefixing all lines with
  /// \p Indent. \p SlotTracker is used to print unnamed VPValue's using
  /// consequtive numbers.
  ///
  /// Note that the numbering is applied to the whole VPlan, so printing
  /// individual regions is consistent with the whole VPlan printing.
  void print(raw_ostream &O, const Twine &Indent,
             VPSlotTracker &SlotTracker) const override;
  using VPBlockBase::print; // Get the print(raw_stream &O) version.
#endif

  /// Clone all blocks in the single-entry single-exit region of the block and
  /// their recipes without updating the operands of the cloned recipes.
  VPRegionBlock *clone() override;

  /// Remove the current region from its VPlan, connecting its predecessor to
  /// its entry, and its exiting block to its successor.
  void dissolveToCFGLoop();
};

/// VPlan models a candidate for vectorization, encoding various decisions take
/// to produce efficient output IR, including which branches, basic-blocks and
/// output IR instructions to generate, and their cost. VPlan holds a
/// Hierarchical-CFG of VPBasicBlocks and VPRegionBlocks rooted at an Entry
/// VPBasicBlock.
class VPlan {
  friend class VPlanPrinter;
  friend class VPSlotTracker;

  /// VPBasicBlock corresponding to the original preheader. Used to place
  /// VPExpandSCEV recipes for expressions used during skeleton creation and the
  /// rest of VPlan execution.
  /// When this VPlan is used for the epilogue vector loop, the entry will be
  /// replaced by a new entry block created during skeleton creation.
  VPBasicBlock *Entry;

  /// VPIRBasicBlock wrapping the header of the original scalar loop.
  VPIRBasicBlock *ScalarHeader;

  /// Immutable list of VPIRBasicBlocks wrapping the exit blocks of the original
  /// scalar loop. Note that some exit blocks may be unreachable at the moment,
  /// e.g. if the scalar epilogue always executes.
  SmallVector<VPIRBasicBlock *, 2> ExitBlocks;

  /// Holds the VFs applicable to this VPlan.
  SmallSetVector<ElementCount, 2> VFs;

  /// Holds the UFs applicable to this VPlan. If empty, the VPlan is valid for
  /// any UF.
  SmallSetVector<unsigned, 2> UFs;

  /// Holds the name of the VPlan, for printing.
  std::string Name;

  /// Represents the trip count of the original loop, for folding
  /// the tail.
  VPValue *TripCount = nullptr;

  /// Represents the backedge taken count of the original loop, for folding
  /// the tail. It equals TripCount - 1.
  VPValue *BackedgeTakenCount = nullptr;

  /// Represents the vector trip count.
  VPValue VectorTripCount;

  /// Represents the vectorization factor of the loop.
  VPValue VF;

  /// Represents the loop-invariant VF * UF of the vector loop region.
  VPValue VFxUF;

  /// Holds a mapping between Values and their corresponding VPValue inside
  /// VPlan.
  Value2VPValueTy Value2VPValue;

  /// Contains all the external definitions created for this VPlan. External
  /// definitions are VPValues that hold a pointer to their underlying IR.
  SmallVector<VPValue *, 16> VPLiveIns;

  /// Mapping from SCEVs to the VPValues representing their expansions.
  /// NOTE: This mapping is temporary and will be removed once all users have
  /// been modeled in VPlan directly.
  DenseMap<const SCEV *, VPValue *> SCEVToExpansion;

  /// Blocks allocated and owned by the VPlan. They will be deleted once the
  /// VPlan is destroyed.
  SmallVector<VPBlockBase *> CreatedBlocks;

  /// Construct a VPlan with \p Entry to the plan and with \p ScalarHeader
  /// wrapping the original header of the scalar loop.
  VPlan(VPBasicBlock *Entry, VPIRBasicBlock *ScalarHeader)
      : Entry(Entry), ScalarHeader(ScalarHeader) {
    Entry->setPlan(this);
    assert(ScalarHeader->getNumSuccessors() == 0 &&
           "scalar header must be a leaf node");
  }

public:
  /// Construct a VPlan for \p L. This will create VPIRBasicBlocks wrapping the
  /// original preheader and scalar header of \p L, to be used as entry and
  /// scalar header blocks of the new VPlan.
  VPlan(Loop *L);

  /// Construct a VPlan with a new VPBasicBlock as entry, a VPIRBasicBlock
  /// wrapping \p ScalarHeaderBB and a trip count of \p TC.
  VPlan(BasicBlock *ScalarHeaderBB, VPValue *TC) {
    setEntry(createVPBasicBlock("preheader"));
    ScalarHeader = createVPIRBasicBlock(ScalarHeaderBB);
    TripCount = TC;
  }

  LLVM_ABI_FOR_TEST ~VPlan();

  void setEntry(VPBasicBlock *VPBB) {
    Entry = VPBB;
    VPBB->setPlan(this);
  }

  /// Generate the IR code for this VPlan.
  void execute(VPTransformState *State);

  /// Return the cost of this plan.
  InstructionCost cost(ElementCount VF, VPCostContext &Ctx);

  VPBasicBlock *getEntry() { return Entry; }
  const VPBasicBlock *getEntry() const { return Entry; }

  /// Returns the preheader of the vector loop region, if one exists, or null
  /// otherwise.
  VPBasicBlock *getVectorPreheader() {
    VPRegionBlock *VectorRegion = getVectorLoopRegion();
    return VectorRegion
               ? cast<VPBasicBlock>(VectorRegion->getSinglePredecessor())
               : nullptr;
  }

  /// Returns the VPRegionBlock of the vector loop.
  LLVM_ABI_FOR_TEST VPRegionBlock *getVectorLoopRegion();
  LLVM_ABI_FOR_TEST const VPRegionBlock *getVectorLoopRegion() const;

  /// Returns the 'middle' block of the plan, that is the block that selects
  /// whether to execute the scalar tail loop or the exit block from the loop
  /// latch. If there is an early exit from the vector loop, the middle block
  /// conceptully has the early exit block as third successor, split accross 2
  /// VPBBs. In that case, the second VPBB selects whether to execute the scalar
  /// tail loop or the exit bock. If the scalar tail loop or exit block are
  /// known to always execute, the middle block may branch directly to that
  /// block. This function cannot be called once the vector loop region has been
  /// removed.
  VPBasicBlock *getMiddleBlock() {
    VPRegionBlock *LoopRegion = getVectorLoopRegion();
    assert(
        LoopRegion &&
        "cannot call the function after vector loop region has been removed");
    auto *RegionSucc = cast<VPBasicBlock>(LoopRegion->getSingleSuccessor());
    if (RegionSucc->getSingleSuccessor() ||
        is_contained(RegionSucc->getSuccessors(), getScalarPreheader()))
      return RegionSucc;
    // There is an early exit. The successor of RegionSucc is the middle block.
    return cast<VPBasicBlock>(RegionSucc->getSuccessors()[1]);
  }

  const VPBasicBlock *getMiddleBlock() const {
    return const_cast<VPlan *>(this)->getMiddleBlock();
  }

  /// Return the VPBasicBlock for the preheader of the scalar loop.
  VPBasicBlock *getScalarPreheader() const {
    return cast<VPBasicBlock>(getScalarHeader()->getSinglePredecessor());
  }

  /// Return the VPIRBasicBlock wrapping the header of the scalar loop.
  VPIRBasicBlock *getScalarHeader() const { return ScalarHeader; }

  /// Return an ArrayRef containing VPIRBasicBlocks wrapping the exit blocks of
  /// the original scalar loop.
  ArrayRef<VPIRBasicBlock *> getExitBlocks() const { return ExitBlocks; }

  /// Return the VPIRBasicBlock corresponding to \p IRBB. \p IRBB must be an
  /// exit block.
  VPIRBasicBlock *getExitBlock(BasicBlock *IRBB) const;

  /// Returns true if \p VPBB is an exit block.
  bool isExitBlock(VPBlockBase *VPBB);

  /// The trip count of the original loop.
  VPValue *getTripCount() const {
    assert(TripCount && "trip count needs to be set before accessing it");
    return TripCount;
  }

  /// Set the trip count assuming it is currently null; if it is not - use
  /// resetTripCount().
  void setTripCount(VPValue *NewTripCount) {
    assert(!TripCount && NewTripCount && "TripCount should not be set yet.");
    TripCount = NewTripCount;
  }

  /// Resets the trip count for the VPlan. The caller must make sure all uses of
  /// the original trip count have been replaced.
  void resetTripCount(VPValue *NewTripCount) {
    assert(TripCount && NewTripCount && TripCount->getNumUsers() == 0 &&
           "TripCount must be set when resetting");
    TripCount = NewTripCount;
  }

  /// The backedge taken count of the original loop.
  VPValue *getOrCreateBackedgeTakenCount() {
    if (!BackedgeTakenCount)
      BackedgeTakenCount = new VPValue();
    return BackedgeTakenCount;
  }

  /// The vector trip count.
  VPValue &getVectorTripCount() { return VectorTripCount; }

  /// Returns the VF of the vector loop region.
  VPValue &getVF() { return VF; };

  /// Returns VF * UF of the vector loop region.
  VPValue &getVFxUF() { return VFxUF; }

  LLVMContext &getContext() const {
    return getScalarHeader()->getIRBasicBlock()->getContext();
  }

  void addVF(ElementCount VF) { VFs.insert(VF); }

  void setVF(ElementCount VF) {
    assert(hasVF(VF) && "Cannot set VF not already in plan");
    VFs.clear();
    VFs.insert(VF);
  }

  bool hasVF(ElementCount VF) const { return VFs.count(VF); }
  bool hasScalableVF() const {
    return any_of(VFs, [](ElementCount VF) { return VF.isScalable(); });
  }

  /// Returns an iterator range over all VFs of the plan.
  iterator_range<SmallSetVector<ElementCount, 2>::iterator>
  vectorFactors() const {
    return {VFs.begin(), VFs.end()};
  }

  bool hasScalarVFOnly() const {
    bool HasScalarVFOnly = VFs.size() == 1 && VFs[0].isScalar();
    assert(HasScalarVFOnly == hasVF(ElementCount::getFixed(1)) &&
           "Plan with scalar VF should only have a single VF");
    return HasScalarVFOnly;
  }

  bool hasUF(unsigned UF) const { return UFs.empty() || UFs.contains(UF); }

  unsigned getUF() const {
    assert(UFs.size() == 1 && "Expected a single UF");
    return UFs[0];
  }

  void setUF(unsigned UF) {
    assert(hasUF(UF) && "Cannot set the UF not already in plan");
    UFs.clear();
    UFs.insert(UF);
  }

  /// Returns true if the VPlan already has been unrolled, i.e. it has a single
  /// concrete UF.
  bool isUnrolled() const { return UFs.size() == 1; }

  /// Return a string with the name of the plan and the applicable VFs and UFs.
  std::string getName() const;

  void setName(const Twine &newName) { Name = newName.str(); }

  /// Gets the live-in VPValue for \p V or adds a new live-in (if none exists
  ///  yet) for \p V.
  VPValue *getOrAddLiveIn(Value *V) {
    assert(V && "Trying to get or add the VPValue of a null Value");
    auto [It, Inserted] = Value2VPValue.try_emplace(V);
    if (Inserted) {
      VPValue *VPV = new VPValue(V);
      VPLiveIns.push_back(VPV);
      assert(VPV->isLiveIn() && "VPV must be a live-in.");
      It->second = VPV;
    }

    assert(It->second->isLiveIn() && "Only live-ins should be in mapping");
    return It->second;
  }

  /// Return a VPValue wrapping i1 true.
  VPValue *getTrue() {
    LLVMContext &Ctx = getContext();
    return getOrAddLiveIn(ConstantInt::getTrue(Ctx));
  }

  /// Return a VPValue wrapping i1 false.
  VPValue *getFalse() {
    LLVMContext &Ctx = getContext();
    return getOrAddLiveIn(ConstantInt::getFalse(Ctx));
  }

  /// Return the live-in VPValue for \p V, if there is one or nullptr otherwise.
  VPValue *getLiveIn(Value *V) const { return Value2VPValue.lookup(V); }

  /// Return the list of live-in VPValues available in the VPlan.
  ArrayRef<VPValue *> getLiveIns() const {
    assert(all_of(Value2VPValue,
                  [this](const auto &P) {
                    return is_contained(VPLiveIns, P.second);
                  }) &&
           "all VPValues in Value2VPValue must also be in VPLiveIns");
    return VPLiveIns;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Print the live-ins of this VPlan to \p O.
  void printLiveIns(raw_ostream &O) const;

  /// Print this VPlan to \p O.
  void print(raw_ostream &O) const;

  /// Print this VPlan in DOT format to \p O.
  void printDOT(raw_ostream &O) const;

  /// Dump the plan to stderr (for debugging).
  LLVM_DUMP_METHOD void dump() const;
#endif

  /// Returns the canonical induction recipe of the vector loop.
  VPCanonicalIVPHIRecipe *getCanonicalIV() {
    VPBasicBlock *EntryVPBB = getVectorLoopRegion()->getEntryBasicBlock();
    if (EntryVPBB->empty()) {
      // VPlan native path.
      EntryVPBB = cast<VPBasicBlock>(EntryVPBB->getSingleSuccessor());
    }
    return cast<VPCanonicalIVPHIRecipe>(&*EntryVPBB->begin());
  }

  VPValue *getSCEVExpansion(const SCEV *S) const {
    return SCEVToExpansion.lookup(S);
  }

  void addSCEVExpansion(const SCEV *S, VPValue *V) {
    assert(!SCEVToExpansion.contains(S) && "SCEV already expanded");
    SCEVToExpansion[S] = V;
  }

  /// Clone the current VPlan, update all VPValues of the new VPlan and cloned
  /// recipes to refer to the clones, and return it.
  VPlan *duplicate();

  /// Create a new VPBasicBlock with \p Name and containing \p Recipe if
  /// present. The returned block is owned by the VPlan and deleted once the
  /// VPlan is destroyed.
  VPBasicBlock *createVPBasicBlock(const Twine &Name,
                                   VPRecipeBase *Recipe = nullptr) {
    auto *VPB = new VPBasicBlock(Name, Recipe);
    CreatedBlocks.push_back(VPB);
    return VPB;
  }

  /// Create a new VPRegionBlock with \p Entry, \p Exiting and \p Name. If \p
  /// IsReplicator is true, the region is a replicate region. The returned block
  /// is owned by the VPlan and deleted once the VPlan is destroyed.
  VPRegionBlock *createVPRegionBlock(VPBlockBase *Entry, VPBlockBase *Exiting,
                                     const std::string &Name = "",
                                     bool IsReplicator = false) {
    auto *VPB = new VPRegionBlock(Entry, Exiting, Name, IsReplicator);
    CreatedBlocks.push_back(VPB);
    return VPB;
  }

  /// Create a new loop VPRegionBlock with \p Name and entry and exiting blocks set
  /// to nullptr. The returned block is owned by the VPlan and deleted once the
  /// VPlan is destroyed.
  VPRegionBlock *createVPRegionBlock(const std::string &Name = "") {
    auto *VPB = new VPRegionBlock(Name);
    CreatedBlocks.push_back(VPB);
    return VPB;
  }

  /// Create a VPIRBasicBlock wrapping \p IRBB, but do not create
  /// VPIRInstructions wrapping the instructions in t\p IRBB.  The returned
  /// block is owned by the VPlan and deleted once the VPlan is destroyed.
  VPIRBasicBlock *createEmptyVPIRBasicBlock(BasicBlock *IRBB);

  /// Create a VPIRBasicBlock from \p IRBB containing VPIRInstructions for all
  /// instructions in \p IRBB, except its terminator which is managed by the
  /// successors of the block in VPlan. The returned block is owned by the VPlan
  /// and deleted once the VPlan is destroyed.
  LLVM_ABI_FOR_TEST VPIRBasicBlock *createVPIRBasicBlock(BasicBlock *IRBB);

  /// Returns true if the VPlan is based on a loop with an early exit. That is
  /// the case if the VPlan has either more than one exit block or a single exit
  /// block with multiple predecessors (one for the exit via the latch and one
  /// via the other early exit).
  bool hasEarlyExit() const {
    return count_if(ExitBlocks,
                    [](VPIRBasicBlock *EB) {
                      return EB->getNumPredecessors() != 0;
                    }) > 1 ||
           (ExitBlocks.size() == 1 && ExitBlocks[0]->getNumPredecessors() > 1);
  }

  /// Returns true if the scalar tail may execute after the vector loop. Note
  /// that this relies on unneeded branches to the scalar tail loop being
  /// removed.
  bool hasScalarTail() const {
    return !(getScalarPreheader()->getNumPredecessors() == 0 ||
             getScalarPreheader()->getSinglePredecessor() == getEntry());
  }
};

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
inline raw_ostream &operator<<(raw_ostream &OS, const VPlan &Plan) {
  Plan.print(OS);
  return OS;
}
#endif

} // end namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLAN_H
