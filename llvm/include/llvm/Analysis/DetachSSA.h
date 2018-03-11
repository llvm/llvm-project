//===- DetachSSA.h - Build Detach SSA ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file exposes an interface to building/using detach SSA to
/// walk detach instructions using a use/def graph.
///
/// This analysis is heavily based on MemorySSA.
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DETACHSSA_H
#define LLVM_ANALYSIS_DETACHSSA_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedUser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/OperandTraits.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

class Function;
class Instruction;
class DetachAccess;
class LLVMContext;
class raw_ostream;

namespace DSSAHelpers {

struct AllAccessTag {};
struct DefsOnlyTag {};

} // end namespace DSSAHelpers

enum {
  // Used to signify what the default invalid ID is for DetachAccess's
  // getID()
  INVALID_DETACHACCESS_ID = 0
};

template <class T> class detachaccess_def_iterator_base;
using detachaccess_def_iterator = detachaccess_def_iterator_base<DetachAccess>;
using const_detachaccess_def_iterator =
    detachaccess_def_iterator_base<const DetachAccess>;

// \brief The base for all detach accesses, i.e., detaches (defs) and syncs
// (uses).
class DetachAccess
  : public DerivedUser,
    public ilist_node<DetachAccess, ilist_tag<DSSAHelpers::AllAccessTag>>,
    public ilist_node<DetachAccess, ilist_tag<DSSAHelpers::DefsOnlyTag>> {
public:
  using AllAccessType =
      ilist_node<DetachAccess, ilist_tag<DSSAHelpers::AllAccessTag>>;
  using DefsOnlyType =
      ilist_node<DetachAccess, ilist_tag<DSSAHelpers::DefsOnlyTag>>;

  // Methods for support type inquiry through isa, cast, and
  // dyn_cast
  static inline bool classof(const Value *V) {
    unsigned ID = V->getValueID();
    return ID == DetachUseVal || ID == DetachPhiVal || ID == DetachDefVal;
  }

  DetachAccess(const DetachAccess &) = delete;
  DetachAccess &operator=(const DetachAccess &) = delete;

  void *operator new(size_t, unsigned) = delete;
  void *operator new(size_t) = delete;

  BasicBlock *getBlock() const { return Block; }

  void print(raw_ostream &OS) const;
  void dump() const;

  /// \brief The user iterators for a detach access
  using iterator = user_iterator;
  using const_iterator = const_user_iterator;

  /// \brief This iterator walks over all of the defs in a given
  /// DetachAccess. For DetachPhi nodes, this walks arguments. For
  /// DetachUse/DetachDef, this walks the defining access.
  detachaccess_def_iterator defs_begin();
  const_detachaccess_def_iterator defs_begin() const;
  detachaccess_def_iterator defs_end();
  const_detachaccess_def_iterator defs_end() const;

  /// \brief Get the iterators for the all access list and the defs only list
  /// We default to the all access list.
  AllAccessType::self_iterator getIterator() {
    return this->AllAccessType::getIterator();
  }
  AllAccessType::const_self_iterator getIterator() const {
    return this->AllAccessType::getIterator();
  }
  AllAccessType::reverse_self_iterator getReverseIterator() {
    return this->AllAccessType::getReverseIterator();
  }
  AllAccessType::const_reverse_self_iterator getReverseIterator() const {
    return this->AllAccessType::getReverseIterator();
  }
  DefsOnlyType::self_iterator getDefsIterator() {
    return this->DefsOnlyType::getIterator();
  }
  DefsOnlyType::const_self_iterator getDefsIterator() const {
    return this->DefsOnlyType::getIterator();
  }
  DefsOnlyType::reverse_self_iterator getReverseDefsIterator() {
    return this->DefsOnlyType::getReverseIterator();
  }
  DefsOnlyType::const_reverse_self_iterator getReverseDefsIterator() const {
    return this->DefsOnlyType::getReverseIterator();
  }

protected:
  friend class DetachDef;
  friend class DetachPhi;
  friend class DetachSSA;
  friend class DetachUse;
  friend class DetachUseOrDef;

  /// \brief Used by DetachSSA to change the block of a DetachAccess when it is
  /// moved.
  void setBlock(BasicBlock *BB) { Block = BB; }

  /// \brief Used for debugging and tracking things about DetachAccesses.
  /// Guaranteed unique among DetachAccesses, no guarantees otherwise.
  inline unsigned getID() const;

  DetachAccess(LLVMContext &C, unsigned Vty, DeleteValueTy DeleteValue,
               BasicBlock *BB, unsigned NumOperands)
      : DerivedUser(Type::getVoidTy(C), Vty, nullptr, NumOperands, DeleteValue),
        Block(BB) {}

private:
  BasicBlock *Block;
};

inline raw_ostream &operator<<(raw_ostream &OS, const DetachAccess &DA) {
  DA.print(OS);
  return OS;
}

/// \brief Class that has the common methods + fields of detach uses/defs. It's
/// a little awkward to have, but there are many cases where we want either a
/// use or def, and there are many cases where uses are needed (defs aren't
/// acceptable), and vice-versa.
///
/// This class should never be instantiated directly; make a DetachUse or
/// DetachDef instead.
class DetachUseOrDef : public DetachAccess {
public:
  void *operator new(size_t) = delete;

  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(DetachAccess);

  /// \brief Get the instruction that this DetachAccess represents.
  Instruction *getDAInst() const { return DAInst; }

  /// \brief Get the access that produces the detach state used by this Use.
  DetachAccess *getDefiningAccess() const { return getOperand(0); }

  static inline bool classof(const Value *DA) {
    return DA->getValueID() == DetachUseVal || DA->getValueID() == DetachDefVal;
  }

  // Sadly, these have to be public because they are needed in some of the
  // iterators.
  inline bool isOptimized() const;
  inline DetachAccess *getOptimized() const;
  inline void setOptimized(DetachAccess *);

  /// \brief Reset the ID of what this DetachUse was optimized to, causing it to
  /// be rewalked by the walker if necessary.
  /// This really should only be called by tests.
  inline void resetOptimized();

protected:
  friend class DetachSSA;

  DetachUseOrDef(LLVMContext &C, DetachAccess *DDA, unsigned Vty,
                 DeleteValueTy DeleteValue, Instruction *TI, BasicBlock *BB)
      : DetachAccess(C, Vty, DeleteValue, BB, 1), DAInst(TI) {
    setDefiningAccess(DDA);
  }

  void setDefiningAccess(DetachAccess *DDA, bool Optimized = false) {
    if (!Optimized) {
      setOperand(0, DDA);
      return;
    }
    setOptimized(DDA);
  }

private:
  Instruction *DAInst;
};

template <>
struct OperandTraits<DetachUseOrDef>
    : public FixedNumOperandTraits<DetachUseOrDef, 1> {};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(DetachUseOrDef, DetachAccess)

/// \brief Represents a detach use, i.e., a sync instruction.
class DetachUse final : public DetachUseOrDef {
public:
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(DetachAccess);

  DetachUse(LLVMContext &C, DetachAccess *DDA, Instruction *SI, BasicBlock *BB)
      : DetachUseOrDef(C, DDA, DetachUseVal, deleteMe, SI, BB),
        OptimizedID(0) {}

  // allocate space for exactly one operand
  void *operator new(size_t s) { return User::operator new(s, 1); }

  static inline bool classof(const Value *DA) {
    return DA->getValueID() == DetachUseVal;
  }

  void print(raw_ostream &OS) const;

  void setOptimized(DetachAccess *DDA) {
    OptimizedID = DDA->getID();
    setOperand(0, DDA);
  }

  bool isOptimized() const {
    return getDefiningAccess() && OptimizedID == getDefiningAccess()->getID();
  }

  DetachAccess *getOptimized() const {
    return getDefiningAccess();
  }
  void resetOptimized() {
    OptimizedID = INVALID_DETACHACCESS_ID;
  }

protected:
  friend class DetachSSA;

private:
  static void deleteMe(DerivedUser *Self);

  unsigned int OptimizedID;
};

template <>
struct OperandTraits<DetachUse> : public FixedNumOperandTraits<DetachUse, 1> {};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(DetachUse, DetachAccess)

/// \brief Represents a detach definition, i.e., a detach.
class DetachDef final : public DetachUseOrDef {
public:
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(DetachAccess);

  DetachDef(LLVMContext &C, DetachAccess *DDA, Instruction *DI, BasicBlock *BB,
            unsigned Ver)
      : DetachUseOrDef(C, DDA, DetachDefVal, deleteMe, DI, BB),
        ID(Ver), Optimized(nullptr), OptimizedID(INVALID_DETACHACCESS_ID) {}

  // allocate space for exactly one operand
  void *operator new(size_t s) { return User::operator new(s, 1); }
  void *operator new(size_t, unsigned) = delete;

  static inline bool classof(const Value *DA) {
    return DA->getValueID() == DetachDefVal;
  }

  void setOptimized(DetachAccess *DA) {
    Optimized = DA;
    OptimizedID = getDefiningAccess()->getID();
  }
  DetachAccess *getOptimized() const { return Optimized; }
  bool isOptimized() const {
    return getOptimized() && getDefiningAccess() &&
           OptimizedID == getDefiningAccess()->getID();
  }
  void resetOptimized() {
    OptimizedID = INVALID_DETACHACCESS_ID;
  }

  void print(raw_ostream &OS) const;

  friend class DetachSSA;

  unsigned getID() const { return ID; }

private:
  static void deleteMe(DerivedUser *Self);

  const unsigned ID;
  DetachAccess *Optimized;
  unsigned int OptimizedID;
};

template <>
struct OperandTraits<DetachDef> : public FixedNumOperandTraits<DetachDef, 1> {};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(DetachDef, DetachAccess)

/// \brief Represents phi nodes for detach accesses.
///
/// These have the same semantics as regular phi nodes, with the exception that
/// only one phi will ever exist in a given basic block.
/// Guaranteeing one phi per block means guaranteeing there is only ever one
/// valid reaching DetachDef/DetachPHI along each path to the phi node.
/// This is ensured by not allowing disambiguation of the RHS of a DetachDef or
/// a DetachPhi's operands.
class DetachPhi final : public DetachAccess {
  // allocate space for exactly zero operands
  void *operator new(size_t s) { return User::operator new(s); }

public:
  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(DetachAccess);

  DetachPhi(LLVMContext &C, BasicBlock *BB, unsigned Ver, unsigned NumPreds = 0)
      : DetachAccess(C, DetachPhiVal, deleteMe, BB, 0), ID(Ver),
        ReservedSpace(NumPreds) {
    allocHungoffUses(ReservedSpace);
  }

  // Block iterator interface. This provides access to the list of incoming
  // basic blocks, which parallels the list of incoming values.
  using block_iterator = BasicBlock **;
  using const_block_iterator = BasicBlock *const *;

  block_iterator block_begin() {
    auto *Ref = reinterpret_cast<Use::UserRef *>(op_begin() + ReservedSpace);
    return reinterpret_cast<block_iterator>(Ref + 1);
  }

  const_block_iterator block_begin() const {
    const auto *Ref =
        reinterpret_cast<const Use::UserRef *>(op_begin() + ReservedSpace);
    return reinterpret_cast<const_block_iterator>(Ref + 1);
  }

  block_iterator block_end() { return block_begin() + getNumOperands(); }

  const_block_iterator block_end() const {
    return block_begin() + getNumOperands();
  }

  iterator_range<block_iterator> blocks() {
    return make_range(block_begin(), block_end());
  }

  iterator_range<const_block_iterator> blocks() const {
    return make_range(block_begin(), block_end());
  }

  op_range incoming_values() { return operands(); }

  const_op_range incoming_values() const { return operands(); }

  /// \brief Return the number of incoming edges
  unsigned getNumIncomingValues() const { return getNumOperands(); }

  /// \brief Return incoming value number x
  DetachAccess *getIncomingValue(unsigned I) const { return getOperand(I); }
  void setIncomingValue(unsigned I, DetachAccess *V) {
    assert(V && "PHI node got a null value!");
    setOperand(I, V);
  }
  static unsigned getOperandNumForIncomingValue(unsigned I) { return I; }
  static unsigned getIncomingValueNumForOperand(unsigned I) { return I; }

  /// \brief Return incoming basic block number @p i.
  BasicBlock *getIncomingBlock(unsigned I) const { return block_begin()[I]; }

  /// \brief Return incoming basic block corresponding
  /// to an operand of the PHI.
  BasicBlock *getIncomingBlock(const Use &U) const {
    assert(this == U.getUser() && "Iterator doesn't point to PHI's Uses?");
    return getIncomingBlock(unsigned(&U - op_begin()));
  }

  /// \brief Return incoming basic block corresponding
  /// to value use iterator.
  BasicBlock *getIncomingBlock(DetachAccess::const_user_iterator I) const {
    return getIncomingBlock(I.getUse());
  }

  void setIncomingBlock(unsigned I, BasicBlock *BB) {
    assert(BB && "PHI node got a null basic block!");
    block_begin()[I] = BB;
  }

  /// \brief Add an incoming value to the end of the PHI list
  void addIncoming(DetachAccess *V, BasicBlock *BB) {
    if (getNumOperands() == ReservedSpace)
      growOperands(); // Get more space!
    // Initialize some new operands.
    setNumHungOffUseOperands(getNumOperands() + 1);
    setIncomingValue(getNumOperands() - 1, V);
    setIncomingBlock(getNumOperands() - 1, BB);
  }

  /// \brief Return the first index of the specified basic
  /// block in the value list for this PHI.  Returns -1 if no instance.
  int getBasicBlockIndex(const BasicBlock *BB) const {
    for (unsigned I = 0, E = getNumOperands(); I != E; ++I)
      if (block_begin()[I] == BB)
        return I;
    return -1;
  }

  Value *getIncomingValueForBlock(const BasicBlock *BB) const {
    int Idx = getBasicBlockIndex(BB);
    assert(Idx >= 0 && "Invalid basic block argument!");
    return getIncomingValue(Idx);
  }

  static inline bool classof(const Value *V) {
    return V->getValueID() == DetachPhiVal;
  }

  void print(raw_ostream &OS) const;

  unsigned getID() const { return ID; }

protected:
  friend class DetachSSA;

  /// \brief this is more complicated than the generic
  /// User::allocHungoffUses, because we have to allocate Uses for the incoming
  /// values and pointers to the incoming blocks, all in one allocation.
  void allocHungoffUses(unsigned N) {
    User::allocHungoffUses(N, /* IsPhi */ true);
  }

private:
  // For debugging only
  const unsigned ID;
  unsigned ReservedSpace;

  /// \brief This grows the operand list in response to a push_back style of
  /// operation.  This grows the number of ops by 1.5 times.
  void growOperands() {
    unsigned E = getNumOperands();
    // 2 op PHI nodes are VERY common, so reserve at least enough for that.
    ReservedSpace = std::max(E + E / 2, 2u);
    growHungoffUses(ReservedSpace, /* IsPhi */ true);
  }

  static void deleteMe(DerivedUser *Self);
};

inline unsigned DetachAccess::getID() const {
  assert((isa<DetachDef>(this) || isa<DetachPhi>(this)) &&
         "only detach defs and phis have ids");
  if (const auto *DD = dyn_cast<DetachDef>(this))
    return DD->getID();
  return cast<DetachPhi>(this)->getID();
}

inline bool DetachUseOrDef::isOptimized() const {
  if (const auto *DD = dyn_cast<DetachDef>(this))
    return DD->isOptimized();
  return cast<DetachUse>(this)->isOptimized();
}

inline DetachAccess *DetachUseOrDef::getOptimized() const {
  if (const auto *DD = dyn_cast<DetachDef>(this))
    return DD->getOptimized();
  return cast<DetachUse>(this)->getOptimized();
}

inline void DetachUseOrDef::setOptimized(DetachAccess *DA) {
  if (auto *DD = dyn_cast<DetachDef>(this))
    DD->setOptimized(DA);
  else
    cast<DetachUse>(this)->setOptimized(DA);
}

inline void DetachUseOrDef::resetOptimized() {
  if (auto *DD = dyn_cast<DetachDef>(this))
    DD->resetOptimized();
  else
    cast<DetachUse>(this)->resetOptimized();
}


template <> struct OperandTraits<DetachPhi> : public HungoffOperandTraits<2> {};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(DetachPhi, DetachAccess)


/// \brief Encapsulates DetachSSA, including all data associated with detach
/// accesses.
class DetachSSA {
public:
  DetachSSA(Function &, DominatorTree *);
  ~DetachSSA();

  /// \brief Given a detach Mod/Ref'ing instruction, get the DetachSSA
  /// access associated with it. If passed a basic block gets the detach phi
  /// node that exists for that block, if there is one. Otherwise, this will get
  /// a DetachUseOrDef.
  DetachUseOrDef *getDetachAccess(const Instruction *) const;
  DetachPhi *getDetachAccess(const BasicBlock *BB) const;

  void dump() const;
  void print(raw_ostream &) const;

  /// \brief Return true if \p MA represents the live on entry value
  inline bool isLiveOnEntryDef(const DetachAccess *DA) const {
    return DA == LiveOnEntryDef.get();
  }

  inline DetachAccess *getLiveOnEntryDef() const {
    return LiveOnEntryDef.get();
  }

  // Sadly, iplists, by default, owns and deletes pointers added to the
  // list. It's not currently possible to have two iplists for the same type,
  // where one owns the pointers, and one does not. This is because the traits
  // are per-type, not per-tag.  If this ever changes, we should make the
  // DefList an iplist.
  using AccessList = iplist<DetachAccess, ilist_tag<DSSAHelpers::AllAccessTag>>;
  using DefsList =
      simple_ilist<DetachAccess, ilist_tag<DSSAHelpers::DefsOnlyTag>>;

  /// \brief Return the list of MemoryAccess's for a given basic block.
  ///
  /// This list is not modifiable by the user.
  const AccessList *getBlockAccesses(const BasicBlock *BB) const {
    return getWritableBlockAccesses(BB);
  }

  /// \brief Return the list of MemoryDef's and MemoryPhi's for a given basic
  /// block.
  ///
  /// This list is not modifiable by the user.
  const DefsList *getBlockDefs(const BasicBlock *BB) const {
    return getWritableBlockDefs(BB);
  }

  /// \brief Given two detach accesses in the same basic block, determine
  /// whether DetachAccess \p A dominates DetachAccess \p B.
  bool locallyDominates(const DetachAccess *A, const DetachAccess *B) const;

  /// \brief Given two detach accesses in potentially different blocks,
  /// determine whether DetachAccess \p A dominates DetachAccess \p B.
  bool dominates(const DetachAccess *A, const DetachAccess *B) const;

  /// \brief Given a DetachAccess and a Use, determine whether DetachAccess \p A
  /// dominates Use \p B.
  bool dominates(const DetachAccess *A, const Use &B) const;

  /// \brief Verify that DetachSSA is self consistent (IE definitions dominate
  /// all uses, uses appear in the right places).  This is used by unit tests.
  void verifyDetachSSA() const;

  /// Used in various insertion functions to specify whether we are talking
  /// about the beginning or end of a block.
  enum InsertionPlace { Beginning, End };

protected:
  // Used by Detach SSA annotater, dumpers, and wrapper pass
  friend class DetachSSAAnnotatedWriter;
  friend class DetachSSAPrinterLegacyPass;

  void verifyDefUses(Function &F) const;
  void verifyDomination(Function &F) const;
  void verifyOrdering(Function &F) const;

  AccessList *getWritableBlockAccesses(const BasicBlock *BB) const {
    auto It = PerBlockAccesses.find(BB);
    return It == PerBlockAccesses.end() ? nullptr : It->second.get();
  }

  DefsList *getWritableBlockDefs(const BasicBlock *BB) const {
    auto It = PerBlockDefs.find(BB);
    return It == PerBlockDefs.end() ? nullptr : It->second.get();
  }

  void moveTo(DetachUseOrDef *What, BasicBlock *BB, AccessList::iterator Where);
  void moveTo(DetachUseOrDef *What, BasicBlock *BB, InsertionPlace Point);
  // Rename the dominator tree branch rooted at BB.
  void renamePass(BasicBlock *BB, DetachAccess *IncomingVal,
                  SmallPtrSetImpl<BasicBlock *> &Visited) {
    renamePass(DT->getNode(BB), IncomingVal, Visited, true, true);
  }
  void removeFromLookups(DetachAccess *);
  void removeFromLists(DetachAccess *, bool ShouldDelete = true);
  void insertIntoListsForBlock(DetachAccess *, const BasicBlock *,
                               InsertionPlace);
  void insertIntoListsBefore(DetachAccess *, const BasicBlock *,
                             AccessList::iterator);
  // DetachUseOrDef *createDefinedAccess(Instruction *, DetachAccess *);

private:
  // class CachingWalker;

  // CachingWalker *getWalkerImpl();
  void buildDetachSSA();

  void verifyUseInDefs(DetachAccess *, DetachAccess *) const;
  using AccessMap = DenseMap<const BasicBlock *, std::unique_ptr<AccessList>>;
  using DefsMap = DenseMap<const BasicBlock *, std::unique_ptr<DefsList>>;

  void
  determineInsertionPoint(const SmallPtrSetImpl<BasicBlock *> &DefiningBlocks);
  void markUnreachableAsLiveOnEntry(BasicBlock *BB);
  bool dominatesUse(const DetachAccess *, const DetachAccess *) const;
  DetachPhi *createDetachPhi(BasicBlock *BB);
  // DetachUseOrDef *createNewAccess(Instruction *);
  DetachAccess *findDominatingDef(BasicBlock *, enum InsertionPlace);
  void placePHINodes(const SmallPtrSetImpl<BasicBlock *> &,
                     const DenseMap<const BasicBlock *, unsigned int> &);
  DetachAccess *renameBlock(BasicBlock *, DetachAccess *, bool);
  void renameSuccessorPhis(BasicBlock *, DetachAccess *, bool);
  void renamePass(DomTreeNode *, DetachAccess *IncomingVal,
                  SmallPtrSetImpl<BasicBlock *> &Visited,
                  bool SkipVisited = false, bool RenameAllUses = false);
  AccessList *getOrCreateAccessList(const BasicBlock *);
  DefsList *getOrCreateDefsList(const BasicBlock *);
  void renumberBlock(const BasicBlock *) const;
  DominatorTree *DT;
  Function &F;

  // Detach SSA mappings
  DenseMap<const Value *, DetachAccess *> ValueToDetachAccess;
  // These two mappings contain the main block to access/def mappings for
  // DetachSSA. The list contained in PerBlockAccesses really owns all the
  // DetachAccesses.
  // Both maps maintain the invariant that if a block is found in them, the
  // corresponding list is not empty, and if a block is not found in them, the
  // corresponding list is empty.
  AccessMap PerBlockAccesses;
  DefsMap PerBlockDefs;
  std::unique_ptr<DetachAccess> LiveOnEntryDef;

  // Domination mappings
  // Note that the numbering is local to a block, even though the map is
  // global.
  mutable SmallPtrSet<const BasicBlock *, 16> BlockNumberingValid;
  mutable DenseMap<const DetachAccess *, unsigned long> BlockNumbering;

  // Memory SSA building info
  // std::unique_ptr<CachingWalker> Walker;
  unsigned NextID;
};

// This pass does eager building and then printing of DetachSSA. It is used by
// the tests to be able to build, dump, and verify Detach SSA.
class DetachSSAPrinterLegacyPass : public FunctionPass {
public:
  DetachSSAPrinterLegacyPass();

  bool runOnFunction(Function &) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  static char ID;
};

/// An analysis that produces \c DetachSSA for a function.
///
class DetachSSAAnalysis : public AnalysisInfoMixin<DetachSSAAnalysis> {
  friend AnalysisInfoMixin<DetachSSAAnalysis>;

  static AnalysisKey Key;

public:
  // Wrap DetachSSA result to ensure address stability of internal DetachSSA
  // pointers after construction.  Use a wrapper class instead of plain
  // unique_ptr<DetachSSA> to avoid build breakage on MSVC.
  struct Result {
    Result(std::unique_ptr<DetachSSA> &&DSSA) : DSSA(std::move(DSSA)) {}
    DetachSSA &getDSSA() { return *DSSA.get(); }

    std::unique_ptr<DetachSSA> DSSA;
  };

  Result run(Function &F, FunctionAnalysisManager &AM);
};

/// \brief Printer pass for \c DetachSSA.
class DetachSSAPrinterPass : public PassInfoMixin<DetachSSAPrinterPass> {
  raw_ostream &OS;

public:
  explicit DetachSSAPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// \brief Verifier pass for \c DetachSSA.
struct DetachSSAVerifierPass : PassInfoMixin<DetachSSAVerifierPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// \brief Legacy analysis pass which computes \c DetachSSA.
class DetachSSAWrapperPass : public FunctionPass {
public:
  DetachSSAWrapperPass();

  static char ID;

  bool runOnFunction(Function &) override;
  void releaseMemory() override;
  DetachSSA &getDSSA() { return *DSSA; }
  const DetachSSA &getDSSA() const { return *DSSA; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  void verifyAnalysis() const override;
  void print(raw_ostream &OS, const Module *M = nullptr) const override;

private:
  std::unique_ptr<DetachSSA> DSSA;
};

/// \brief Iterator base class used to implement const and non-const iterators
/// over the defining accesses of a DetachAccess.
template <class T>
class detachaccess_def_iterator_base
    : public iterator_facade_base<detachaccess_def_iterator_base<T>,
                                  std::forward_iterator_tag, T, ptrdiff_t, T *,
                                  T *> {
  using BaseT = typename detachaccess_def_iterator_base::iterator_facade_base;

public:
  detachaccess_def_iterator_base(T *Start) : Access(Start) {}
  detachaccess_def_iterator_base() = default;

  bool operator==(const detachaccess_def_iterator_base &Other) const {
    return Access == Other.Access && (!Access || ArgNo == Other.ArgNo);
  }

  // This is a bit ugly, but for DetachPHI's, unlike PHINodes, you can't get the
  // block from the operand in constant time (In a PHINode, the uselist has
  // both, so it's just subtraction). We provide it as part of the
  // iterator to avoid callers having to linear walk to get the block.
  // If the operation becomes constant time on DetachPHI's, this bit of
  // abstraction breaking should be removed.
  BasicBlock *getPhiArgBlock() const {
    DetachPhi *DP = dyn_cast<DetachPhi>(Access);
    assert(DP && "Tried to get phi arg block when not iterating over a PHI");
    return DP->getIncomingBlock(ArgNo);
  }
  typename BaseT::iterator::pointer operator*() const {
    assert(Access && "Tried to access past the end of our iterator");
    // Go to the first argument for phis, and the defining access for everything
    // else.
    if (DetachPhi *DP = dyn_cast<DetachPhi>(Access))
      return DP->getIncomingValue(ArgNo);
    return cast<DetachUseOrDef>(Access)->getDefiningAccess();
  }
  using BaseT::operator++;
  detachaccess_def_iterator &operator++() {
    assert(Access && "Hit end of iterator");
    if (DetachPhi *DP = dyn_cast<DetachPhi>(Access)) {
      if (++ArgNo >= DP->getNumIncomingValues()) {
        ArgNo = 0;
        Access = nullptr;
      }
    } else {
      Access = nullptr;
    }
    return *this;
  }

private:
  T *Access = nullptr;
  unsigned ArgNo = 0;
};

inline detachaccess_def_iterator DetachAccess::defs_begin() {
  return detachaccess_def_iterator(this);
}

inline const_detachaccess_def_iterator DetachAccess::defs_begin() const {
  return const_detachaccess_def_iterator(this);
}

inline detachaccess_def_iterator DetachAccess::defs_end() {
  return detachaccess_def_iterator();
}

inline const_detachaccess_def_iterator DetachAccess::defs_end() const {
  return const_detachaccess_def_iterator();
}

/// \brief GraphTraits for a DetachAccess, which walks defs in the normal case,
/// and uses in the inverse case.
template <> struct GraphTraits<DetachAccess *> {
  using NodeRef = DetachAccess *;
  using ChildIteratorType = detachaccess_def_iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) { return N->defs_begin(); }
  static ChildIteratorType child_end(NodeRef N) { return N->defs_end(); }
};

template <> struct GraphTraits<Inverse<DetachAccess *>> {
  using NodeRef = DetachAccess *;
  using ChildIteratorType = DetachAccess::iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }
  static ChildIteratorType child_begin(NodeRef N) { return N->user_begin(); }
  static ChildIteratorType child_end(NodeRef N) { return N->user_end(); }
};

} // End namespace llvm

#endif // LLVM_ANALYSIS_DETACHSSA_H
