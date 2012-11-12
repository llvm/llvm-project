//===-- llvm/BasicBlock.h - Represent a basic block in the VM ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the BasicBlock class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BASICBLOCK_H
#define LLVM_BASICBLOCK_H

#include "llvm/Instruction.h"
#include "llvm/SymbolTableListTraits.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class LandingPadInst;
class TerminatorInst;
class LLVMContext;
class BlockAddress;

template<> struct ilist_traits<Instruction>
  : public SymbolTableListTraits<Instruction, BasicBlock> {
  // createSentinel is used to get hold of a node that marks the end of
  // the list...
  // The sentinel is relative to this instance, so we use a non-static
  // method.
  Instruction *createSentinel() const {
    // since i(p)lists always publicly derive from the corresponding
    // traits, placing a data member in this class will augment i(p)list.
    // But since the NodeTy is expected to publicly derive from
    // ilist_node<NodeTy>, there is a legal viable downcast from it
    // to NodeTy. We use this trick to superpose i(p)list with a "ghostly"
    // NodeTy, which becomes the sentinel. Dereferencing the sentinel is
    // forbidden (save the ilist_node<NodeTy>) so no one will ever notice
    // the superposition.
    return static_cast<Instruction*>(&Sentinel);
  }
  static void destroySentinel(Instruction*) {}

  Instruction *provideInitialHead() const { return createSentinel(); }
  Instruction *ensureHead(Instruction*) const { return createSentinel(); }
  static void noteHead(Instruction*, Instruction*) {}
private:
  mutable ilist_half_node<Instruction> Sentinel;
};

/// This represents a single basic block in LLVM. A basic block is simply a
/// container of instructions that execute sequentially. Basic blocks are Values
/// because they are referenced by instructions such as branches and switch
/// tables. The type of a BasicBlock is "Type::LabelTy" because the basic block
/// represents a label to which a branch can jump.
///
/// A well formed basic block is formed of a list of non-terminating
/// instructions followed by a single TerminatorInst instruction.
/// TerminatorInst's may not occur in the middle of basic blocks, and must
/// terminate the blocks. The BasicBlock class allows malformed basic blocks to
/// occur because it may be useful in the intermediate stage of constructing or
/// modifying a program. However, the verifier will ensure that basic blocks
/// are "well formed".
/// @brief LLVM Basic Block Representation
class BasicBlock : public Value, // Basic blocks are data objects also
                   public ilist_node<BasicBlock> {
  friend class BlockAddress;
public:
  typedef iplist<Instruction> InstListType;
private:
  InstListType InstList;
  Function *Parent;

  void setParent(Function *parent);
  friend class SymbolTableListTraits<BasicBlock, Function>;

  BasicBlock(const BasicBlock &) LLVM_DELETED_FUNCTION;
  void operator=(const BasicBlock &) LLVM_DELETED_FUNCTION;

  /// BasicBlock ctor - If the function parameter is specified, the basic block
  /// is automatically inserted at either the end of the function (if
  /// InsertBefore is null), or before the specified basic block.
  ///
  explicit BasicBlock(LLVMContext &C, const Twine &Name = "",
                      Function *Parent = 0, BasicBlock *InsertBefore = 0);
public:
  /// getContext - Get the context in which this basic block lives.
  LLVMContext &getContext() const;

  /// Instruction iterators...
  typedef InstListType::iterator                              iterator;
  typedef InstListType::const_iterator                  const_iterator;

  /// Create - Creates a new BasicBlock. If the Parent parameter is specified,
  /// the basic block is automatically inserted at either the end of the
  /// function (if InsertBefore is 0), or before the specified basic block.
  static BasicBlock *Create(LLVMContext &Context, const Twine &Name = "",
                            Function *Parent = 0,BasicBlock *InsertBefore = 0) {
    return new BasicBlock(Context, Name, Parent, InsertBefore);
  }
  ~BasicBlock();

  /// getParent - Return the enclosing method, or null if none
  ///
  const Function *getParent() const { return Parent; }
        Function *getParent()       { return Parent; }

  /// getTerminator() - If this is a well formed basic block, then this returns
  /// a pointer to the terminator instruction.  If it is not, then you get a
  /// null pointer back.
  ///
  TerminatorInst *getTerminator();
  const TerminatorInst *getTerminator() const;

  /// Returns a pointer to the first instructon in this block that is not a
  /// PHINode instruction. When adding instruction to the beginning of the
  /// basic block, they should be added before the returned value, not before
  /// the first instruction, which might be PHI.
  /// Returns 0 is there's no non-PHI instruction.
  Instruction* getFirstNonPHI();
  const Instruction* getFirstNonPHI() const {
    return const_cast<BasicBlock*>(this)->getFirstNonPHI();
  }

  // Same as above, but also skip debug intrinsics.
  Instruction* getFirstNonPHIOrDbg();
  const Instruction* getFirstNonPHIOrDbg() const {
    return const_cast<BasicBlock*>(this)->getFirstNonPHIOrDbg();
  }

  // Same as above, but also skip lifetime intrinsics.
  Instruction* getFirstNonPHIOrDbgOrLifetime();
  const Instruction* getFirstNonPHIOrDbgOrLifetime() const {
    return const_cast<BasicBlock*>(this)->getFirstNonPHIOrDbgOrLifetime();
  }

  /// getFirstInsertionPt - Returns an iterator to the first instruction in this
  /// block that is suitable for inserting a non-PHI instruction. In particular,
  /// it skips all PHIs and LandingPad instructions.
  iterator getFirstInsertionPt();
  const_iterator getFirstInsertionPt() const {
    return const_cast<BasicBlock*>(this)->getFirstInsertionPt();
  }

  /// removeFromParent - This method unlinks 'this' from the containing
  /// function, but does not delete it.
  ///
  void removeFromParent();

  /// eraseFromParent - This method unlinks 'this' from the containing function
  /// and deletes it.
  ///
  void eraseFromParent();

  /// moveBefore - Unlink this basic block from its current function and
  /// insert it into the function that MovePos lives in, right before MovePos.
  void moveBefore(BasicBlock *MovePos);

  /// moveAfter - Unlink this basic block from its current function and
  /// insert it into the function that MovePos lives in, right after MovePos.
  void moveAfter(BasicBlock *MovePos);


  /// getSinglePredecessor - If this basic block has a single predecessor block,
  /// return the block, otherwise return a null pointer.
  BasicBlock *getSinglePredecessor();
  const BasicBlock *getSinglePredecessor() const {
    return const_cast<BasicBlock*>(this)->getSinglePredecessor();
  }

  /// getUniquePredecessor - If this basic block has a unique predecessor block,
  /// return the block, otherwise return a null pointer.
  /// Note that unique predecessor doesn't mean single edge, there can be
  /// multiple edges from the unique predecessor to this block (for example
  /// a switch statement with multiple cases having the same destination).
  BasicBlock *getUniquePredecessor();
  const BasicBlock *getUniquePredecessor() const {
    return const_cast<BasicBlock*>(this)->getUniquePredecessor();
  }

  //===--------------------------------------------------------------------===//
  /// Instruction iterator methods
  ///
  inline iterator                begin()       { return InstList.begin(); }
  inline const_iterator          begin() const { return InstList.begin(); }
  inline iterator                end  ()       { return InstList.end();   }
  inline const_iterator          end  () const { return InstList.end();   }

  inline size_t                   size() const { return InstList.size();  }
  inline bool                    empty() const { return InstList.empty(); }
  inline const Instruction      &front() const { return InstList.front(); }
  inline       Instruction      &front()       { return InstList.front(); }
  inline const Instruction       &back() const { return InstList.back();  }
  inline       Instruction       &back()       { return InstList.back();  }

  /// getInstList() - Return the underlying instruction list container.  You
  /// need to access it directly if you want to modify it currently.
  ///
  const InstListType &getInstList() const { return InstList; }
        InstListType &getInstList()       { return InstList; }

  /// getSublistAccess() - returns pointer to member of instruction list
  static iplist<Instruction> BasicBlock::*getSublistAccess(Instruction*) {
    return &BasicBlock::InstList;
  }

  /// getValueSymbolTable() - returns pointer to symbol table (if any)
  ValueSymbolTable *getValueSymbolTable();

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Value *V) {
    return V->getValueID() == Value::BasicBlockVal;
  }

  /// dropAllReferences() - This function causes all the subinstructions to "let
  /// go" of all references that they are maintaining.  This allows one to
  /// 'delete' a whole class at a time, even though there may be circular
  /// references... first all references are dropped, and all use counts go to
  /// zero.  Then everything is delete'd for real.  Note that no operations are
  /// valid on an object that has "dropped all references", except operator
  /// delete.
  ///
  void dropAllReferences();

  /// removePredecessor - This method is used to notify a BasicBlock that the
  /// specified Predecessor of the block is no longer able to reach it.  This is
  /// actually not used to update the Predecessor list, but is actually used to
  /// update the PHI nodes that reside in the block.  Note that this should be
  /// called while the predecessor still refers to this block.
  ///
  void removePredecessor(BasicBlock *Pred, bool DontDeleteUselessPHIs = false);

  /// splitBasicBlock - This splits a basic block into two at the specified
  /// instruction.  Note that all instructions BEFORE the specified iterator
  /// stay as part of the original basic block, an unconditional branch is added
  /// to the original BB, and the rest of the instructions in the BB are moved
  /// to the new BB, including the old terminator.  The newly formed BasicBlock
  /// is returned.  This function invalidates the specified iterator.
  ///
  /// Note that this only works on well formed basic blocks (must have a
  /// terminator), and 'I' must not be the end of instruction list (which would
  /// cause a degenerate basic block to be formed, having a terminator inside of
  /// the basic block).
  ///
  /// Also note that this doesn't preserve any passes. To split blocks while
  /// keeping loop information consistent, use the SplitBlock utility function.
  ///
  BasicBlock *splitBasicBlock(iterator I, const Twine &BBName = "");

  /// hasAddressTaken - returns true if there are any uses of this basic block
  /// other than direct branches, switches, etc. to it.
  bool hasAddressTaken() const { return getSubclassDataFromValue() != 0; }

  /// replaceSuccessorsPhiUsesWith - Update all phi nodes in all our successors
  /// to refer to basic block New instead of to us.
  void replaceSuccessorsPhiUsesWith(BasicBlock *New);

  /// isLandingPad - Return true if this basic block is a landing pad. I.e.,
  /// it's the destination of the 'unwind' edge of an invoke instruction.
  bool isLandingPad() const;

  /// getLandingPadInst() - Return the landingpad instruction associated with
  /// the landing pad.
  LandingPadInst *getLandingPadInst();
  const LandingPadInst *getLandingPadInst() const;

private:
  /// AdjustBlockAddressRefCount - BasicBlock stores the number of BlockAddress
  /// objects using it.  This is almost always 0, sometimes one, possibly but
  /// almost never 2, and inconceivably 3 or more.
  void AdjustBlockAddressRefCount(int Amt) {
    setValueSubclassData(getSubclassDataFromValue()+Amt);
    assert((int)(signed char)getSubclassDataFromValue() >= 0 &&
           "Refcount wrap-around");
  }
  // Shadow Value::setValueSubclassData with a private forwarding method so that
  // any future subclasses cannot accidentally use it.
  void setValueSubclassData(unsigned short D) {
    Value::setValueSubclassData(D);
  }
};

} // End llvm namespace

#endif
