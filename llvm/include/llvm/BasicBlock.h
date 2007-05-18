//===-- llvm/BasicBlock.h - Represent a basic block in the VM ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
// This file contains the declaration of the BasicBlock class.
//===----------------------------------------------------------------------===//

#ifndef LLVM_BASICBLOCK_H
#define LLVM_BASICBLOCK_H

#include "llvm/Instruction.h"
#include "llvm/SymbolTableListTraits.h"
#include "llvm/ADT/ilist"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class TerminatorInst;
template <class Term, class BB> class SuccIterator;  // Successor Iterator
template <class Ptr, class USE_iterator> class PredIterator;

template<> struct ilist_traits<Instruction>
  : public SymbolTableListTraits<Instruction, BasicBlock> {
  // createSentinel is used to create a node that marks the end of the list...
  static Instruction *createSentinel();
  static void destroySentinel(Instruction *I) { delete I; }
  static iplist<Instruction> &getList(BasicBlock *BB);
  static ValueSymbolTable *getSymTab(BasicBlock *ItemParent);
  static int getListOffset();
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
class BasicBlock : public Value {       // Basic blocks are data objects also
public:
  typedef iplist<Instruction> InstListType;
private :
  InstListType InstList;
  BasicBlock *Prev, *Next; // Next and Prev links for our intrusive linked list
  Function *Parent;

  void setParent(Function *parent);
  void setNext(BasicBlock *N) { Next = N; }
  void setPrev(BasicBlock *N) { Prev = N; }
  friend class SymbolTableListTraits<BasicBlock, Function>;

  BasicBlock(const BasicBlock &);     // Do not implement
  void operator=(const BasicBlock &); // Do not implement

public:
  /// Instruction iterators...
  typedef InstListType::iterator                              iterator;
  typedef InstListType::const_iterator                  const_iterator;

  /// BasicBlock ctor - If the function parameter is specified, the basic block
  /// is automatically inserted at either the end of the function (if
  /// InsertBefore is null), or before the specified basic block.
  ///
  explicit BasicBlock(const std::string &Name = "", Function *Parent = 0,
                      BasicBlock *InsertBefore = 0);
  ~BasicBlock();

  /// getParent - Return the enclosing method, or null if none
  ///
  const Function *getParent() const { return Parent; }
        Function *getParent()       { return Parent; }

  /// use_back - Specialize the methods defined in Value, as we know that an
  /// BasicBlock can only be used by Instructions (specifically PHI and terms).
  Instruction       *use_back()       { return cast<Instruction>(*use_begin());}
  const Instruction *use_back() const { return cast<Instruction>(*use_begin());}
  
  /// getTerminator() - If this is a well formed basic block, then this returns
  /// a pointer to the terminator instruction.  If it is not, then you get a
  /// null pointer back.
  ///
  TerminatorInst *getTerminator();
  const TerminatorInst *const getTerminator() const;
  
  /// Returns a pointer to the first instructon in this block that is not a 
  /// PHINode instruction. When adding instruction to the beginning of the
  /// basic block, they should be added before the returned value, not before
  /// the first instruction, which might be PHI.
  /// Returns 0 is there's no non-PHI instruction.
  Instruction* getFirstNonPHI();
  
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

  virtual void print(std::ostream &OS) const { print(OS, 0); }
  void print(std::ostream *OS) const { if (OS) print(*OS); }
  void print(std::ostream &OS, AssemblyAnnotationWriter *AAW) const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const BasicBlock *) { return true; }
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
  BasicBlock *splitBasicBlock(iterator I, const std::string &BBName = "");
  
  
  static unsigned getInstListOffset() {
    BasicBlock *Obj = 0;
    return unsigned(reinterpret_cast<uintptr_t>(&Obj->InstList));
  }

private:
  // getNext/Prev - Return the next or previous basic block in the list.  Access
  // these with Function::iterator.
  BasicBlock *getNext()       { return Next; }
  const BasicBlock *getNext() const { return Next; }
  BasicBlock *getPrev()       { return Prev; }
  const BasicBlock *getPrev() const { return Prev; }
};

inline int 
ilist_traits<Instruction>::getListOffset() {
  return BasicBlock::getInstListOffset();
}

} // End llvm namespace

#endif
