//===-- llvm/Instructions.h - Instruction subclass definitions --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes the class definitions of all of the subclasses of the
// Instruction class.  This is meant to be an easy way to get access to all
// instruction subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_INSTRUCTIONS_H
#define LLVM_IR_INSTRUCTIONS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/OperandTraits.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace llvm {

class APInt;
class ConstantInt;
class DataLayout;
class LLVMContext;

enum SynchronizationScope {
  SingleThread = 0,
  CrossThread = 1
};

//===----------------------------------------------------------------------===//
//                                AllocaInst Class
//===----------------------------------------------------------------------===//

/// an instruction to allocate memory on the stack
class AllocaInst : public UnaryInstruction {
  Type *AllocatedType;

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  AllocaInst *cloneImpl() const;

public:
  explicit AllocaInst(Type *Ty, unsigned AddrSpace,
                      Value *ArraySize = nullptr,
                      const Twine &Name = "",
                      Instruction *InsertBefore = nullptr);
  AllocaInst(Type *Ty, unsigned AddrSpace, Value *ArraySize,
             const Twine &Name, BasicBlock *InsertAtEnd);

  AllocaInst(Type *Ty, unsigned AddrSpace,
             const Twine &Name, Instruction *InsertBefore = nullptr);
  AllocaInst(Type *Ty, unsigned AddrSpace,
             const Twine &Name, BasicBlock *InsertAtEnd);

  AllocaInst(Type *Ty, unsigned AddrSpace, Value *ArraySize, unsigned Align,
             const Twine &Name = "", Instruction *InsertBefore = nullptr);
  AllocaInst(Type *Ty, unsigned AddrSpace, Value *ArraySize, unsigned Align,
             const Twine &Name, BasicBlock *InsertAtEnd);

  // Out of line virtual method, so the vtable, etc. has a home.
  ~AllocaInst() override;

  /// Return true if there is an allocation size parameter to the allocation
  /// instruction that is not 1.
  bool isArrayAllocation() const;

  /// Get the number of elements allocated. For a simple allocation of a single
  /// element, this will return a constant 1 value.
  const Value *getArraySize() const { return getOperand(0); }
  Value *getArraySize() { return getOperand(0); }

  /// Overload to return most specific pointer type.
  PointerType *getType() const {
    return cast<PointerType>(Instruction::getType());
  }

  /// Return the type that is being allocated by the instruction.
  Type *getAllocatedType() const { return AllocatedType; }
  /// for use only in special circumstances that need to generically
  /// transform a whole instruction (eg: IR linking and vectorization).
  void setAllocatedType(Type *Ty) { AllocatedType = Ty; }

  /// Return the alignment of the memory that is being allocated by the
  /// instruction.
  unsigned getAlignment() const {
    return (1u << (getSubclassDataFromInstruction() & 31)) >> 1;
  }
  void setAlignment(unsigned Align);

  /// Return true if this alloca is in the entry block of the function and is a
  /// constant size. If so, the code generator will fold it into the
  /// prolog/epilog code, so it is basically free.
  bool isStaticAlloca() const;

  /// Return true if this alloca is used as an inalloca argument to a call. Such
  /// allocas are never considered static even if they are in the entry block.
  bool isUsedWithInAlloca() const {
    return getSubclassDataFromInstruction() & 32;
  }

  /// Specify whether this alloca is used to represent the arguments to a call.
  void setUsedWithInAlloca(bool V) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~32) |
                               (V ? 32 : 0));
  }

  /// Return true if this alloca is used as a swifterror argument to a call.
  bool isSwiftError() const {
    return getSubclassDataFromInstruction() & 64;
  }

  /// Specify whether this alloca is used to represent a swifterror.
  void setSwiftError(bool V) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~64) |
                               (V ? 64 : 0));
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Alloca);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  // Shadow Instruction::setInstructionSubclassData with a private forwarding
  // method so that subclasses cannot accidentally use it.
  void setInstructionSubclassData(unsigned short D) {
    Instruction::setInstructionSubclassData(D);
  }
};

//===----------------------------------------------------------------------===//
//                                LoadInst Class
//===----------------------------------------------------------------------===//

/// An instruction for reading from memory. This uses the SubclassData field in
/// Value to store whether or not the load is volatile.
class LoadInst : public UnaryInstruction {
  void AssertOK();

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  LoadInst *cloneImpl() const;

public:
  LoadInst(Value *Ptr, const Twine &NameStr, Instruction *InsertBefore);
  LoadInst(Value *Ptr, const Twine &NameStr, BasicBlock *InsertAtEnd);
  LoadInst(Type *Ty, Value *Ptr, const Twine &NameStr, bool isVolatile = false,
           Instruction *InsertBefore = nullptr);
  LoadInst(Value *Ptr, const Twine &NameStr, bool isVolatile = false,
           Instruction *InsertBefore = nullptr)
      : LoadInst(cast<PointerType>(Ptr->getType())->getElementType(), Ptr,
                 NameStr, isVolatile, InsertBefore) {}
  LoadInst(Value *Ptr, const Twine &NameStr, bool isVolatile,
           BasicBlock *InsertAtEnd);
  LoadInst(Value *Ptr, const Twine &NameStr, bool isVolatile, unsigned Align,
           Instruction *InsertBefore = nullptr)
      : LoadInst(cast<PointerType>(Ptr->getType())->getElementType(), Ptr,
                 NameStr, isVolatile, Align, InsertBefore) {}
  LoadInst(Type *Ty, Value *Ptr, const Twine &NameStr, bool isVolatile,
           unsigned Align, Instruction *InsertBefore = nullptr);
  LoadInst(Value *Ptr, const Twine &NameStr, bool isVolatile,
           unsigned Align, BasicBlock *InsertAtEnd);
  LoadInst(Value *Ptr, const Twine &NameStr, bool isVolatile, unsigned Align,
           AtomicOrdering Order, SynchronizationScope SynchScope = CrossThread,
           Instruction *InsertBefore = nullptr)
      : LoadInst(cast<PointerType>(Ptr->getType())->getElementType(), Ptr,
                 NameStr, isVolatile, Align, Order, SynchScope, InsertBefore) {}
  LoadInst(Type *Ty, Value *Ptr, const Twine &NameStr, bool isVolatile,
           unsigned Align, AtomicOrdering Order,
           SynchronizationScope SynchScope = CrossThread,
           Instruction *InsertBefore = nullptr);
  LoadInst(Value *Ptr, const Twine &NameStr, bool isVolatile,
           unsigned Align, AtomicOrdering Order,
           SynchronizationScope SynchScope,
           BasicBlock *InsertAtEnd);
  LoadInst(Value *Ptr, const char *NameStr, Instruction *InsertBefore);
  LoadInst(Value *Ptr, const char *NameStr, BasicBlock *InsertAtEnd);
  LoadInst(Type *Ty, Value *Ptr, const char *NameStr = nullptr,
           bool isVolatile = false, Instruction *InsertBefore = nullptr);
  explicit LoadInst(Value *Ptr, const char *NameStr = nullptr,
                    bool isVolatile = false,
                    Instruction *InsertBefore = nullptr)
      : LoadInst(cast<PointerType>(Ptr->getType())->getElementType(), Ptr,
                 NameStr, isVolatile, InsertBefore) {}
  LoadInst(Value *Ptr, const char *NameStr, bool isVolatile,
           BasicBlock *InsertAtEnd);

  /// Return true if this is a load from a volatile memory location.
  bool isVolatile() const { return getSubclassDataFromInstruction() & 1; }

  /// Specify whether this is a volatile load or not.
  void setVolatile(bool V) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~1) |
                               (V ? 1 : 0));
  }

  /// Return the alignment of the access that is being performed.
  unsigned getAlignment() const {
    return (1 << ((getSubclassDataFromInstruction() >> 1) & 31)) >> 1;
  }

  void setAlignment(unsigned Align);

  /// Returns the ordering effect of this fence.
  AtomicOrdering getOrdering() const {
    return AtomicOrdering((getSubclassDataFromInstruction() >> 7) & 7);
  }

  /// Set the ordering constraint on this load. May not be Release or
  /// AcquireRelease.
  void setOrdering(AtomicOrdering Ordering) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~(7 << 7)) |
                               ((unsigned)Ordering << 7));
  }

  SynchronizationScope getSynchScope() const {
    return SynchronizationScope((getSubclassDataFromInstruction() >> 6) & 1);
  }

  /// Specify whether this load is ordered with respect to all
  /// concurrently executing threads, or only with respect to signal handlers
  /// executing in the same thread.
  void setSynchScope(SynchronizationScope xthread) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~(1 << 6)) |
                               (xthread << 6));
  }

  void setAtomic(AtomicOrdering Ordering,
                 SynchronizationScope SynchScope = CrossThread) {
    setOrdering(Ordering);
    setSynchScope(SynchScope);
  }

  bool isSimple() const { return !isAtomic() && !isVolatile(); }
  bool isUnordered() const {
    return (getOrdering() == AtomicOrdering::NotAtomic ||
            getOrdering() == AtomicOrdering::Unordered) &&
           !isVolatile();
  }

  Value *getPointerOperand() { return getOperand(0); }
  const Value *getPointerOperand() const { return getOperand(0); }
  static unsigned getPointerOperandIndex() { return 0U; }

  /// Returns the address space of the pointer operand.
  unsigned getPointerAddressSpace() const {
    return getPointerOperand()->getType()->getPointerAddressSpace();
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Load;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  // Shadow Instruction::setInstructionSubclassData with a private forwarding
  // method so that subclasses cannot accidentally use it.
  void setInstructionSubclassData(unsigned short D) {
    Instruction::setInstructionSubclassData(D);
  }
};

//===----------------------------------------------------------------------===//
//                                StoreInst Class
//===----------------------------------------------------------------------===//

/// An instruction for storing to memory.
class StoreInst : public Instruction {
  void AssertOK();

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  StoreInst *cloneImpl() const;

public:
  StoreInst(Value *Val, Value *Ptr, Instruction *InsertBefore);
  StoreInst(Value *Val, Value *Ptr, BasicBlock *InsertAtEnd);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile = false,
            Instruction *InsertBefore = nullptr);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile, BasicBlock *InsertAtEnd);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile,
            unsigned Align, Instruction *InsertBefore = nullptr);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile,
            unsigned Align, BasicBlock *InsertAtEnd);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile,
            unsigned Align, AtomicOrdering Order,
            SynchronizationScope SynchScope = CrossThread,
            Instruction *InsertBefore = nullptr);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile,
            unsigned Align, AtomicOrdering Order,
            SynchronizationScope SynchScope,
            BasicBlock *InsertAtEnd);

  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }

  void *operator new(size_t, unsigned) = delete;

  /// Return true if this is a store to a volatile memory location.
  bool isVolatile() const { return getSubclassDataFromInstruction() & 1; }

  /// Specify whether this is a volatile store or not.
  void setVolatile(bool V) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~1) |
                               (V ? 1 : 0));
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// Return the alignment of the access that is being performed
  unsigned getAlignment() const {
    return (1 << ((getSubclassDataFromInstruction() >> 1) & 31)) >> 1;
  }

  void setAlignment(unsigned Align);

  /// Returns the ordering effect of this store.
  AtomicOrdering getOrdering() const {
    return AtomicOrdering((getSubclassDataFromInstruction() >> 7) & 7);
  }

  /// Set the ordering constraint on this store.  May not be Acquire or
  /// AcquireRelease.
  void setOrdering(AtomicOrdering Ordering) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~(7 << 7)) |
                               ((unsigned)Ordering << 7));
  }

  SynchronizationScope getSynchScope() const {
    return SynchronizationScope((getSubclassDataFromInstruction() >> 6) & 1);
  }

  /// Specify whether this store instruction is ordered with respect to all
  /// concurrently executing threads, or only with respect to signal handlers
  /// executing in the same thread.
  void setSynchScope(SynchronizationScope xthread) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~(1 << 6)) |
                               (xthread << 6));
  }

  void setAtomic(AtomicOrdering Ordering,
                 SynchronizationScope SynchScope = CrossThread) {
    setOrdering(Ordering);
    setSynchScope(SynchScope);
  }

  bool isSimple() const { return !isAtomic() && !isVolatile(); }
  bool isUnordered() const {
    return (getOrdering() == AtomicOrdering::NotAtomic ||
            getOrdering() == AtomicOrdering::Unordered) &&
           !isVolatile();
  }

  Value *getValueOperand() { return getOperand(0); }
  const Value *getValueOperand() const { return getOperand(0); }

  Value *getPointerOperand() { return getOperand(1); }
  const Value *getPointerOperand() const { return getOperand(1); }
  static unsigned getPointerOperandIndex() { return 1U; }

  /// Returns the address space of the pointer operand.
  unsigned getPointerAddressSpace() const {
    return getPointerOperand()->getType()->getPointerAddressSpace();
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Store;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  // Shadow Instruction::setInstructionSubclassData with a private forwarding
  // method so that subclasses cannot accidentally use it.
  void setInstructionSubclassData(unsigned short D) {
    Instruction::setInstructionSubclassData(D);
  }
};

template <>
struct OperandTraits<StoreInst> : public FixedNumOperandTraits<StoreInst, 2> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(StoreInst, Value)

//===----------------------------------------------------------------------===//
//                                FenceInst Class
//===----------------------------------------------------------------------===//

/// An instruction for ordering other memory operations.
class FenceInst : public Instruction {
  void Init(AtomicOrdering Ordering, SynchronizationScope SynchScope);

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  FenceInst *cloneImpl() const;

public:
  // Ordering may only be Acquire, Release, AcquireRelease, or
  // SequentiallyConsistent.
  FenceInst(LLVMContext &C, AtomicOrdering Ordering,
            SynchronizationScope SynchScope = CrossThread,
            Instruction *InsertBefore = nullptr);
  FenceInst(LLVMContext &C, AtomicOrdering Ordering,
            SynchronizationScope SynchScope,
            BasicBlock *InsertAtEnd);

  // allocate space for exactly zero operands
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }

  void *operator new(size_t, unsigned) = delete;

  /// Returns the ordering effect of this fence.
  AtomicOrdering getOrdering() const {
    return AtomicOrdering(getSubclassDataFromInstruction() >> 1);
  }

  /// Set the ordering constraint on this fence.  May only be Acquire, Release,
  /// AcquireRelease, or SequentiallyConsistent.
  void setOrdering(AtomicOrdering Ordering) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & 1) |
                               ((unsigned)Ordering << 1));
  }

  SynchronizationScope getSynchScope() const {
    return SynchronizationScope(getSubclassDataFromInstruction() & 1);
  }

  /// Specify whether this fence orders other operations with respect to all
  /// concurrently executing threads, or only with respect to signal handlers
  /// executing in the same thread.
  void setSynchScope(SynchronizationScope xthread) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~1) |
                               xthread);
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Fence;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  // Shadow Instruction::setInstructionSubclassData with a private forwarding
  // method so that subclasses cannot accidentally use it.
  void setInstructionSubclassData(unsigned short D) {
    Instruction::setInstructionSubclassData(D);
  }
};

//===----------------------------------------------------------------------===//
//                                AtomicCmpXchgInst Class
//===----------------------------------------------------------------------===//

/// an instruction that atomically checks whether a
/// specified value is in a memory location, and, if it is, stores a new value
/// there.  Returns the value that was loaded.
///
class AtomicCmpXchgInst : public Instruction {
  void Init(Value *Ptr, Value *Cmp, Value *NewVal,
            AtomicOrdering SuccessOrdering, AtomicOrdering FailureOrdering,
            SynchronizationScope SynchScope);

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  AtomicCmpXchgInst *cloneImpl() const;

public:
  AtomicCmpXchgInst(Value *Ptr, Value *Cmp, Value *NewVal,
                    AtomicOrdering SuccessOrdering,
                    AtomicOrdering FailureOrdering,
                    SynchronizationScope SynchScope,
                    Instruction *InsertBefore = nullptr);
  AtomicCmpXchgInst(Value *Ptr, Value *Cmp, Value *NewVal,
                    AtomicOrdering SuccessOrdering,
                    AtomicOrdering FailureOrdering,
                    SynchronizationScope SynchScope,
                    BasicBlock *InsertAtEnd);

  // allocate space for exactly three operands
  void *operator new(size_t s) {
    return User::operator new(s, 3);
  }

  void *operator new(size_t, unsigned) = delete;

  /// Return true if this is a cmpxchg from a volatile memory
  /// location.
  ///
  bool isVolatile() const {
    return getSubclassDataFromInstruction() & 1;
  }

  /// Specify whether this is a volatile cmpxchg.
  ///
  void setVolatile(bool V) {
     setInstructionSubclassData((getSubclassDataFromInstruction() & ~1) |
                                (unsigned)V);
  }

  /// Return true if this cmpxchg may spuriously fail.
  bool isWeak() const {
    return getSubclassDataFromInstruction() & 0x100;
  }

  void setWeak(bool IsWeak) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~0x100) |
                               (IsWeak << 8));
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// Set the ordering constraint on this cmpxchg.
  void setSuccessOrdering(AtomicOrdering Ordering) {
    assert(Ordering != AtomicOrdering::NotAtomic &&
           "CmpXchg instructions can only be atomic.");
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~0x1c) |
                               ((unsigned)Ordering << 2));
  }

  void setFailureOrdering(AtomicOrdering Ordering) {
    assert(Ordering != AtomicOrdering::NotAtomic &&
           "CmpXchg instructions can only be atomic.");
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~0xe0) |
                               ((unsigned)Ordering << 5));
  }

  /// Specify whether this cmpxchg is atomic and orders other operations with
  /// respect to all concurrently executing threads, or only with respect to
  /// signal handlers executing in the same thread.
  void setSynchScope(SynchronizationScope SynchScope) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~2) |
                               (SynchScope << 1));
  }

  /// Returns the ordering constraint on this cmpxchg.
  AtomicOrdering getSuccessOrdering() const {
    return AtomicOrdering((getSubclassDataFromInstruction() >> 2) & 7);
  }

  /// Returns the ordering constraint on this cmpxchg.
  AtomicOrdering getFailureOrdering() const {
    return AtomicOrdering((getSubclassDataFromInstruction() >> 5) & 7);
  }

  /// Returns whether this cmpxchg is atomic between threads or only within a
  /// single thread.
  SynchronizationScope getSynchScope() const {
    return SynchronizationScope((getSubclassDataFromInstruction() & 2) >> 1);
  }

  Value *getPointerOperand() { return getOperand(0); }
  const Value *getPointerOperand() const { return getOperand(0); }
  static unsigned getPointerOperandIndex() { return 0U; }

  Value *getCompareOperand() { return getOperand(1); }
  const Value *getCompareOperand() const { return getOperand(1); }

  Value *getNewValOperand() { return getOperand(2); }
  const Value *getNewValOperand() const { return getOperand(2); }

  /// Returns the address space of the pointer operand.
  unsigned getPointerAddressSpace() const {
    return getPointerOperand()->getType()->getPointerAddressSpace();
  }

  /// Returns the strongest permitted ordering on failure, given the
  /// desired ordering on success.
  ///
  /// If the comparison in a cmpxchg operation fails, there is no atomic store
  /// so release semantics cannot be provided. So this function drops explicit
  /// Release requests from the AtomicOrdering. A SequentiallyConsistent
  /// operation would remain SequentiallyConsistent.
  static AtomicOrdering
  getStrongestFailureOrdering(AtomicOrdering SuccessOrdering) {
    switch (SuccessOrdering) {
    default:
      llvm_unreachable("invalid cmpxchg success ordering");
    case AtomicOrdering::Release:
    case AtomicOrdering::Monotonic:
      return AtomicOrdering::Monotonic;
    case AtomicOrdering::AcquireRelease:
    case AtomicOrdering::Acquire:
      return AtomicOrdering::Acquire;
    case AtomicOrdering::SequentiallyConsistent:
      return AtomicOrdering::SequentiallyConsistent;
    }
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::AtomicCmpXchg;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  // Shadow Instruction::setInstructionSubclassData with a private forwarding
  // method so that subclasses cannot accidentally use it.
  void setInstructionSubclassData(unsigned short D) {
    Instruction::setInstructionSubclassData(D);
  }
};

template <>
struct OperandTraits<AtomicCmpXchgInst> :
    public FixedNumOperandTraits<AtomicCmpXchgInst, 3> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(AtomicCmpXchgInst, Value)

//===----------------------------------------------------------------------===//
//                                AtomicRMWInst Class
//===----------------------------------------------------------------------===//

/// an instruction that atomically reads a memory location,
/// combines it with another value, and then stores the result back.  Returns
/// the old value.
///
class AtomicRMWInst : public Instruction {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  AtomicRMWInst *cloneImpl() const;

public:
  /// This enumeration lists the possible modifications atomicrmw can make.  In
  /// the descriptions, 'p' is the pointer to the instruction's memory location,
  /// 'old' is the initial value of *p, and 'v' is the other value passed to the
  /// instruction.  These instructions always return 'old'.
  enum BinOp {
    /// *p = v
    Xchg,
    /// *p = old + v
    Add,
    /// *p = old - v
    Sub,
    /// *p = old & v
    And,
    /// *p = ~(old & v)
    Nand,
    /// *p = old | v
    Or,
    /// *p = old ^ v
    Xor,
    /// *p = old >signed v ? old : v
    Max,
    /// *p = old <signed v ? old : v
    Min,
    /// *p = old >unsigned v ? old : v
    UMax,
    /// *p = old <unsigned v ? old : v
    UMin,

    FIRST_BINOP = Xchg,
    LAST_BINOP = UMin,
    BAD_BINOP
  };

  AtomicRMWInst(BinOp Operation, Value *Ptr, Value *Val,
                AtomicOrdering Ordering, SynchronizationScope SynchScope,
                Instruction *InsertBefore = nullptr);
  AtomicRMWInst(BinOp Operation, Value *Ptr, Value *Val,
                AtomicOrdering Ordering, SynchronizationScope SynchScope,
                BasicBlock *InsertAtEnd);

  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }

  void *operator new(size_t, unsigned) = delete;

  BinOp getOperation() const {
    return static_cast<BinOp>(getSubclassDataFromInstruction() >> 5);
  }

  void setOperation(BinOp Operation) {
    unsigned short SubclassData = getSubclassDataFromInstruction();
    setInstructionSubclassData((SubclassData & 31) |
                               (Operation << 5));
  }

  /// Return true if this is a RMW on a volatile memory location.
  ///
  bool isVolatile() const {
    return getSubclassDataFromInstruction() & 1;
  }

  /// Specify whether this is a volatile RMW or not.
  ///
  void setVolatile(bool V) {
     setInstructionSubclassData((getSubclassDataFromInstruction() & ~1) |
                                (unsigned)V);
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// Set the ordering constraint on this RMW.
  void setOrdering(AtomicOrdering Ordering) {
    assert(Ordering != AtomicOrdering::NotAtomic &&
           "atomicrmw instructions can only be atomic.");
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~(7 << 2)) |
                               ((unsigned)Ordering << 2));
  }

  /// Specify whether this RMW orders other operations with respect to all
  /// concurrently executing threads, or only with respect to signal handlers
  /// executing in the same thread.
  void setSynchScope(SynchronizationScope SynchScope) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~2) |
                               (SynchScope << 1));
  }

  /// Returns the ordering constraint on this RMW.
  AtomicOrdering getOrdering() const {
    return AtomicOrdering((getSubclassDataFromInstruction() >> 2) & 7);
  }

  /// Returns whether this RMW is atomic between threads or only within a
  /// single thread.
  SynchronizationScope getSynchScope() const {
    return SynchronizationScope((getSubclassDataFromInstruction() & 2) >> 1);
  }

  Value *getPointerOperand() { return getOperand(0); }
  const Value *getPointerOperand() const { return getOperand(0); }
  static unsigned getPointerOperandIndex() { return 0U; }

  Value *getValOperand() { return getOperand(1); }
  const Value *getValOperand() const { return getOperand(1); }

  /// Returns the address space of the pointer operand.
  unsigned getPointerAddressSpace() const {
    return getPointerOperand()->getType()->getPointerAddressSpace();
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::AtomicRMW;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  void Init(BinOp Operation, Value *Ptr, Value *Val,
            AtomicOrdering Ordering, SynchronizationScope SynchScope);

  // Shadow Instruction::setInstructionSubclassData with a private forwarding
  // method so that subclasses cannot accidentally use it.
  void setInstructionSubclassData(unsigned short D) {
    Instruction::setInstructionSubclassData(D);
  }
};

template <>
struct OperandTraits<AtomicRMWInst>
    : public FixedNumOperandTraits<AtomicRMWInst,2> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(AtomicRMWInst, Value)

//===----------------------------------------------------------------------===//
//                             GetElementPtrInst Class
//===----------------------------------------------------------------------===//

// checkGEPType - Simple wrapper function to give a better assertion failure
// message on bad indexes for a gep instruction.
//
inline Type *checkGEPType(Type *Ty) {
  assert(Ty && "Invalid GetElementPtrInst indices for type!");
  return Ty;
}

/// an instruction for type-safe pointer arithmetic to
/// access elements of arrays and structs
///
class GetElementPtrInst : public Instruction {
  Type *SourceElementType;
  Type *ResultElementType;

  void anchor() override;

  GetElementPtrInst(const GetElementPtrInst &GEPI);
  void init(Value *Ptr, ArrayRef<Value *> IdxList, const Twine &NameStr);

  /// Constructors - Create a getelementptr instruction with a base pointer an
  /// list of indices. The first ctor can optionally insert before an existing
  /// instruction, the second appends the new instruction to the specified
  /// BasicBlock.
  inline GetElementPtrInst(Type *PointeeType, Value *Ptr,
                           ArrayRef<Value *> IdxList, unsigned Values,
                           const Twine &NameStr, Instruction *InsertBefore);
  inline GetElementPtrInst(Type *PointeeType, Value *Ptr,
                           ArrayRef<Value *> IdxList, unsigned Values,
                           const Twine &NameStr, BasicBlock *InsertAtEnd);

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  GetElementPtrInst *cloneImpl() const;

public:
  static GetElementPtrInst *Create(Type *PointeeType, Value *Ptr,
                                   ArrayRef<Value *> IdxList,
                                   const Twine &NameStr = "",
                                   Instruction *InsertBefore = nullptr) {
    unsigned Values = 1 + unsigned(IdxList.size());
    if (!PointeeType)
      PointeeType =
          cast<PointerType>(Ptr->getType()->getScalarType())->getElementType();
    else
      assert(
          PointeeType ==
          cast<PointerType>(Ptr->getType()->getScalarType())->getElementType());
    return new (Values) GetElementPtrInst(PointeeType, Ptr, IdxList, Values,
                                          NameStr, InsertBefore);
  }

  static GetElementPtrInst *Create(Type *PointeeType, Value *Ptr,
                                   ArrayRef<Value *> IdxList,
                                   const Twine &NameStr,
                                   BasicBlock *InsertAtEnd) {
    unsigned Values = 1 + unsigned(IdxList.size());
    if (!PointeeType)
      PointeeType =
          cast<PointerType>(Ptr->getType()->getScalarType())->getElementType();
    else
      assert(
          PointeeType ==
          cast<PointerType>(Ptr->getType()->getScalarType())->getElementType());
    return new (Values) GetElementPtrInst(PointeeType, Ptr, IdxList, Values,
                                          NameStr, InsertAtEnd);
  }

  /// Create an "inbounds" getelementptr. See the documentation for the
  /// "inbounds" flag in LangRef.html for details.
  static GetElementPtrInst *CreateInBounds(Value *Ptr,
                                           ArrayRef<Value *> IdxList,
                                           const Twine &NameStr = "",
                                           Instruction *InsertBefore = nullptr){
    return CreateInBounds(nullptr, Ptr, IdxList, NameStr, InsertBefore);
  }

  static GetElementPtrInst *
  CreateInBounds(Type *PointeeType, Value *Ptr, ArrayRef<Value *> IdxList,
                 const Twine &NameStr = "",
                 Instruction *InsertBefore = nullptr) {
    GetElementPtrInst *GEP =
        Create(PointeeType, Ptr, IdxList, NameStr, InsertBefore);
    GEP->setIsInBounds(true);
    return GEP;
  }

  static GetElementPtrInst *CreateInBounds(Value *Ptr,
                                           ArrayRef<Value *> IdxList,
                                           const Twine &NameStr,
                                           BasicBlock *InsertAtEnd) {
    return CreateInBounds(nullptr, Ptr, IdxList, NameStr, InsertAtEnd);
  }

  static GetElementPtrInst *CreateInBounds(Type *PointeeType, Value *Ptr,
                                           ArrayRef<Value *> IdxList,
                                           const Twine &NameStr,
                                           BasicBlock *InsertAtEnd) {
    GetElementPtrInst *GEP =
        Create(PointeeType, Ptr, IdxList, NameStr, InsertAtEnd);
    GEP->setIsInBounds(true);
    return GEP;
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  Type *getSourceElementType() const { return SourceElementType; }

  void setSourceElementType(Type *Ty) { SourceElementType = Ty; }
  void setResultElementType(Type *Ty) { ResultElementType = Ty; }

  Type *getResultElementType() const {
    assert(ResultElementType ==
           cast<PointerType>(getType()->getScalarType())->getElementType());
    return ResultElementType;
  }

  /// Returns the address space of this instruction's pointer type.
  unsigned getAddressSpace() const {
    // Note that this is always the same as the pointer operand's address space
    // and that is cheaper to compute, so cheat here.
    return getPointerAddressSpace();
  }

  /// Returns the type of the element that would be loaded with
  /// a load instruction with the specified parameters.
  ///
  /// Null is returned if the indices are invalid for the specified
  /// pointer type.
  ///
  static Type *getIndexedType(Type *Ty, ArrayRef<Value *> IdxList);
  static Type *getIndexedType(Type *Ty, ArrayRef<Constant *> IdxList);
  static Type *getIndexedType(Type *Ty, ArrayRef<uint64_t> IdxList);

  inline op_iterator       idx_begin()       { return op_begin()+1; }
  inline const_op_iterator idx_begin() const { return op_begin()+1; }
  inline op_iterator       idx_end()         { return op_end(); }
  inline const_op_iterator idx_end()   const { return op_end(); }

  inline iterator_range<op_iterator> indices() {
    return make_range(idx_begin(), idx_end());
  }

  inline iterator_range<const_op_iterator> indices() const {
    return make_range(idx_begin(), idx_end());
  }

  Value *getPointerOperand() {
    return getOperand(0);
  }
  const Value *getPointerOperand() const {
    return getOperand(0);
  }
  static unsigned getPointerOperandIndex() {
    return 0U;    // get index for modifying correct operand.
  }

  /// Method to return the pointer operand as a
  /// PointerType.
  Type *getPointerOperandType() const {
    return getPointerOperand()->getType();
  }

  /// Returns the address space of the pointer operand.
  unsigned getPointerAddressSpace() const {
    return getPointerOperandType()->getPointerAddressSpace();
  }

  /// Returns the pointer type returned by the GEP
  /// instruction, which may be a vector of pointers.
  static Type *getGEPReturnType(Value *Ptr, ArrayRef<Value *> IdxList) {
    return getGEPReturnType(
      cast<PointerType>(Ptr->getType()->getScalarType())->getElementType(),
      Ptr, IdxList);
  }
  static Type *getGEPReturnType(Type *ElTy, Value *Ptr,
                                ArrayRef<Value *> IdxList) {
    Type *PtrTy = PointerType::get(checkGEPType(getIndexedType(ElTy, IdxList)),
                                   Ptr->getType()->getPointerAddressSpace());
    // Vector GEP
    if (Ptr->getType()->isVectorTy()) {
      unsigned NumElem = Ptr->getType()->getVectorNumElements();
      return VectorType::get(PtrTy, NumElem);
    }
    for (Value *Index : IdxList)
      if (Index->getType()->isVectorTy()) {
        unsigned NumElem = Index->getType()->getVectorNumElements();
        return VectorType::get(PtrTy, NumElem);
      }
    // Scalar GEP
    return PtrTy;
  }

  unsigned getNumIndices() const {  // Note: always non-negative
    return getNumOperands() - 1;
  }

  bool hasIndices() const {
    return getNumOperands() > 1;
  }

  /// Return true if all of the indices of this GEP are
  /// zeros.  If so, the result pointer and the first operand have the same
  /// value, just potentially different types.
  bool hasAllZeroIndices() const;

  /// Return true if all of the indices of this GEP are
  /// constant integers.  If so, the result pointer and the first operand have
  /// a constant offset between them.
  bool hasAllConstantIndices() const;

  /// Set or clear the inbounds flag on this GEP instruction.
  /// See LangRef.html for the meaning of inbounds on a getelementptr.
  void setIsInBounds(bool b = true);

  /// Determine whether the GEP has the inbounds flag.
  bool isInBounds() const;

  /// Accumulate the constant address offset of this GEP if possible.
  ///
  /// This routine accepts an APInt into which it will accumulate the constant
  /// offset of this GEP if the GEP is in fact constant. If the GEP is not
  /// all-constant, it returns false and the value of the offset APInt is
  /// undefined (it is *not* preserved!). The APInt passed into this routine
  /// must be at least as wide as the IntPtr type for the address space of
  /// the base GEP pointer.
  bool accumulateConstantOffset(const DataLayout &DL, APInt &Offset) const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::GetElementPtr);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<GetElementPtrInst> :
  public VariadicOperandTraits<GetElementPtrInst, 1> {
};

GetElementPtrInst::GetElementPtrInst(Type *PointeeType, Value *Ptr,
                                     ArrayRef<Value *> IdxList, unsigned Values,
                                     const Twine &NameStr,
                                     Instruction *InsertBefore)
    : Instruction(getGEPReturnType(PointeeType, Ptr, IdxList), GetElementPtr,
                  OperandTraits<GetElementPtrInst>::op_end(this) - Values,
                  Values, InsertBefore),
      SourceElementType(PointeeType),
      ResultElementType(getIndexedType(PointeeType, IdxList)) {
  assert(ResultElementType ==
         cast<PointerType>(getType()->getScalarType())->getElementType());
  init(Ptr, IdxList, NameStr);
}

GetElementPtrInst::GetElementPtrInst(Type *PointeeType, Value *Ptr,
                                     ArrayRef<Value *> IdxList, unsigned Values,
                                     const Twine &NameStr,
                                     BasicBlock *InsertAtEnd)
    : Instruction(getGEPReturnType(PointeeType, Ptr, IdxList), GetElementPtr,
                  OperandTraits<GetElementPtrInst>::op_end(this) - Values,
                  Values, InsertAtEnd),
      SourceElementType(PointeeType),
      ResultElementType(getIndexedType(PointeeType, IdxList)) {
  assert(ResultElementType ==
         cast<PointerType>(getType()->getScalarType())->getElementType());
  init(Ptr, IdxList, NameStr);
}

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(GetElementPtrInst, Value)

//===----------------------------------------------------------------------===//
//                               ICmpInst Class
//===----------------------------------------------------------------------===//

/// This instruction compares its operands according to the predicate given
/// to the constructor. It only operates on integers or pointers. The operands
/// must be identical types.
/// Represent an integer comparison operator.
class ICmpInst: public CmpInst {
  void anchor() override;

  void AssertOK() {
    assert(getPredicate() >= CmpInst::FIRST_ICMP_PREDICATE &&
           getPredicate() <= CmpInst::LAST_ICMP_PREDICATE &&
           "Invalid ICmp predicate value");
    assert(getOperand(0)->getType() == getOperand(1)->getType() &&
          "Both operands to ICmp instruction are not of the same type!");
    // Check that the operands are the right type
    assert((getOperand(0)->getType()->isIntOrIntVectorTy() ||
            getOperand(0)->getType()->isPtrOrPtrVectorTy()) &&
           "Invalid operand types for ICmp instruction");
  }

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  /// Clone an identical ICmpInst
  ICmpInst *cloneImpl() const;

public:
  /// Constructor with insert-before-instruction semantics.
  ICmpInst(
    Instruction *InsertBefore,  ///< Where to insert
    Predicate pred,  ///< The predicate to use for the comparison
    Value *LHS,      ///< The left-hand-side of the expression
    Value *RHS,      ///< The right-hand-side of the expression
    const Twine &NameStr = ""  ///< Name of the instruction
  ) : CmpInst(makeCmpResultType(LHS->getType()),
              Instruction::ICmp, pred, LHS, RHS, NameStr,
              InsertBefore) {
#ifndef NDEBUG
  AssertOK();
#endif
  }

  /// Constructor with insert-at-end semantics.
  ICmpInst(
    BasicBlock &InsertAtEnd, ///< Block to insert into.
    Predicate pred,  ///< The predicate to use for the comparison
    Value *LHS,      ///< The left-hand-side of the expression
    Value *RHS,      ///< The right-hand-side of the expression
    const Twine &NameStr = ""  ///< Name of the instruction
  ) : CmpInst(makeCmpResultType(LHS->getType()),
              Instruction::ICmp, pred, LHS, RHS, NameStr,
              &InsertAtEnd) {
#ifndef NDEBUG
  AssertOK();
#endif
  }

  /// Constructor with no-insertion semantics
  ICmpInst(
    Predicate pred, ///< The predicate to use for the comparison
    Value *LHS,     ///< The left-hand-side of the expression
    Value *RHS,     ///< The right-hand-side of the expression
    const Twine &NameStr = "" ///< Name of the instruction
  ) : CmpInst(makeCmpResultType(LHS->getType()),
              Instruction::ICmp, pred, LHS, RHS, NameStr) {
#ifndef NDEBUG
  AssertOK();
#endif
  }

  /// For example, EQ->EQ, SLE->SLE, UGT->SGT, etc.
  /// @returns the predicate that would be the result if the operand were
  /// regarded as signed.
  /// Return the signed version of the predicate
  Predicate getSignedPredicate() const {
    return getSignedPredicate(getPredicate());
  }

  /// This is a static version that you can use without an instruction.
  /// Return the signed version of the predicate.
  static Predicate getSignedPredicate(Predicate pred);

  /// For example, EQ->EQ, SLE->ULE, UGT->UGT, etc.
  /// @returns the predicate that would be the result if the operand were
  /// regarded as unsigned.
  /// Return the unsigned version of the predicate
  Predicate getUnsignedPredicate() const {
    return getUnsignedPredicate(getPredicate());
  }

  /// This is a static version that you can use without an instruction.
  /// Return the unsigned version of the predicate.
  static Predicate getUnsignedPredicate(Predicate pred);

  /// Return true if this predicate is either EQ or NE.  This also
  /// tests for commutativity.
  static bool isEquality(Predicate P) {
    return P == ICMP_EQ || P == ICMP_NE;
  }

  /// Return true if this predicate is either EQ or NE.  This also
  /// tests for commutativity.
  bool isEquality() const {
    return isEquality(getPredicate());
  }

  /// @returns true if the predicate of this ICmpInst is commutative
  /// Determine if this relation is commutative.
  bool isCommutative() const { return isEquality(); }

  /// Return true if the predicate is relational (not EQ or NE).
  ///
  bool isRelational() const {
    return !isEquality();
  }

  /// Return true if the predicate is relational (not EQ or NE).
  ///
  static bool isRelational(Predicate P) {
    return !isEquality(P);
  }

  /// Exchange the two operands to this instruction in such a way that it does
  /// not modify the semantics of the instruction. The predicate value may be
  /// changed to retain the same result if the predicate is order dependent
  /// (e.g. ult).
  /// Swap operands and adjust predicate.
  void swapOperands() {
    setPredicate(getSwappedPredicate());
    Op<0>().swap(Op<1>());
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::ICmp;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                               FCmpInst Class
//===----------------------------------------------------------------------===//

/// This instruction compares its operands according to the predicate given
/// to the constructor. It only operates on floating point values or packed
/// vectors of floating point values. The operands must be identical types.
/// Represents a floating point comparison operator.
class FCmpInst: public CmpInst {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  /// Clone an identical FCmpInst
  FCmpInst *cloneImpl() const;

public:
  /// Constructor with insert-before-instruction semantics.
  FCmpInst(
    Instruction *InsertBefore, ///< Where to insert
    Predicate pred,  ///< The predicate to use for the comparison
    Value *LHS,      ///< The left-hand-side of the expression
    Value *RHS,      ///< The right-hand-side of the expression
    const Twine &NameStr = ""  ///< Name of the instruction
  ) : CmpInst(makeCmpResultType(LHS->getType()),
              Instruction::FCmp, pred, LHS, RHS, NameStr,
              InsertBefore) {
    assert(pred <= FCmpInst::LAST_FCMP_PREDICATE &&
           "Invalid FCmp predicate value");
    assert(getOperand(0)->getType() == getOperand(1)->getType() &&
           "Both operands to FCmp instruction are not of the same type!");
    // Check that the operands are the right type
    assert(getOperand(0)->getType()->isFPOrFPVectorTy() &&
           "Invalid operand types for FCmp instruction");
  }

  /// Constructor with insert-at-end semantics.
  FCmpInst(
    BasicBlock &InsertAtEnd, ///< Block to insert into.
    Predicate pred,  ///< The predicate to use for the comparison
    Value *LHS,      ///< The left-hand-side of the expression
    Value *RHS,      ///< The right-hand-side of the expression
    const Twine &NameStr = ""  ///< Name of the instruction
  ) : CmpInst(makeCmpResultType(LHS->getType()),
              Instruction::FCmp, pred, LHS, RHS, NameStr,
              &InsertAtEnd) {
    assert(pred <= FCmpInst::LAST_FCMP_PREDICATE &&
           "Invalid FCmp predicate value");
    assert(getOperand(0)->getType() == getOperand(1)->getType() &&
           "Both operands to FCmp instruction are not of the same type!");
    // Check that the operands are the right type
    assert(getOperand(0)->getType()->isFPOrFPVectorTy() &&
           "Invalid operand types for FCmp instruction");
  }

  /// Constructor with no-insertion semantics
  FCmpInst(
    Predicate pred, ///< The predicate to use for the comparison
    Value *LHS,     ///< The left-hand-side of the expression
    Value *RHS,     ///< The right-hand-side of the expression
    const Twine &NameStr = "" ///< Name of the instruction
  ) : CmpInst(makeCmpResultType(LHS->getType()),
              Instruction::FCmp, pred, LHS, RHS, NameStr) {
    assert(pred <= FCmpInst::LAST_FCMP_PREDICATE &&
           "Invalid FCmp predicate value");
    assert(getOperand(0)->getType() == getOperand(1)->getType() &&
           "Both operands to FCmp instruction are not of the same type!");
    // Check that the operands are the right type
    assert(getOperand(0)->getType()->isFPOrFPVectorTy() &&
           "Invalid operand types for FCmp instruction");
  }

  /// @returns true if the predicate of this instruction is EQ or NE.
  /// Determine if this is an equality predicate.
  static bool isEquality(Predicate Pred) {
    return Pred == FCMP_OEQ || Pred == FCMP_ONE || Pred == FCMP_UEQ ||
           Pred == FCMP_UNE;
  }

  /// @returns true if the predicate of this instruction is EQ or NE.
  /// Determine if this is an equality predicate.
  bool isEquality() const { return isEquality(getPredicate()); }

  /// @returns true if the predicate of this instruction is commutative.
  /// Determine if this is a commutative predicate.
  bool isCommutative() const {
    return isEquality() ||
           getPredicate() == FCMP_FALSE ||
           getPredicate() == FCMP_TRUE ||
           getPredicate() == FCMP_ORD ||
           getPredicate() == FCMP_UNO;
  }

  /// @returns true if the predicate is relational (not EQ or NE).
  /// Determine if this a relational predicate.
  bool isRelational() const { return !isEquality(); }

  /// Exchange the two operands to this instruction in such a way that it does
  /// not modify the semantics of the instruction. The predicate value may be
  /// changed to retain the same result if the predicate is order dependent
  /// (e.g. ult).
  /// Swap operands and adjust predicate.
  void swapOperands() {
    setPredicate(getSwappedPredicate());
    Op<0>().swap(Op<1>());
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::FCmp;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
/// This class represents a function call, abstracting a target
/// machine's calling convention.  This class uses low bit of the SubClassData
/// field to indicate whether or not this is a tail call.  The rest of the bits
/// hold the calling convention of the call.
///
class CallInst : public Instruction,
                 public OperandBundleUser<CallInst, User::op_iterator> {
  friend class OperandBundleUser<CallInst, User::op_iterator>;

  AttributeList Attrs; ///< parameter attributes for call
  FunctionType *FTy;

  CallInst(const CallInst &CI);

  /// Construct a CallInst given a range of arguments.
  /// Construct a CallInst from a range of arguments
  inline CallInst(FunctionType *Ty, Value *Func, ArrayRef<Value *> Args,
                  ArrayRef<OperandBundleDef> Bundles, const Twine &NameStr,
                  Instruction *InsertBefore);

  inline CallInst(Value *Func, ArrayRef<Value *> Args,
                  ArrayRef<OperandBundleDef> Bundles, const Twine &NameStr,
                  Instruction *InsertBefore)
      : CallInst(cast<FunctionType>(
                     cast<PointerType>(Func->getType())->getElementType()),
                 Func, Args, Bundles, NameStr, InsertBefore) {}

  inline CallInst(Value *Func, ArrayRef<Value *> Args, const Twine &NameStr,
                  Instruction *InsertBefore)
      : CallInst(Func, Args, None, NameStr, InsertBefore) {}

  /// Construct a CallInst given a range of arguments.
  /// Construct a CallInst from a range of arguments
  inline CallInst(Value *Func, ArrayRef<Value *> Args,
                  ArrayRef<OperandBundleDef> Bundles, const Twine &NameStr,
                  BasicBlock *InsertAtEnd);

  explicit CallInst(Value *F, const Twine &NameStr,
                    Instruction *InsertBefore);

  CallInst(Value *F, const Twine &NameStr, BasicBlock *InsertAtEnd);

  void init(Value *Func, ArrayRef<Value *> Args,
            ArrayRef<OperandBundleDef> Bundles, const Twine &NameStr) {
    init(cast<FunctionType>(
             cast<PointerType>(Func->getType())->getElementType()),
         Func, Args, Bundles, NameStr);
  }
  void init(FunctionType *FTy, Value *Func, ArrayRef<Value *> Args,
            ArrayRef<OperandBundleDef> Bundles, const Twine &NameStr);
  void init(Value *Func, const Twine &NameStr);

  bool hasDescriptor() const { return HasDescriptor; }

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  CallInst *cloneImpl() const;

public:
  ~CallInst() override;

  static CallInst *Create(Value *Func, ArrayRef<Value *> Args,
                          ArrayRef<OperandBundleDef> Bundles = None,
                          const Twine &NameStr = "",
                          Instruction *InsertBefore = nullptr) {
    return Create(cast<FunctionType>(
                      cast<PointerType>(Func->getType())->getElementType()),
                  Func, Args, Bundles, NameStr, InsertBefore);
  }

  static CallInst *Create(Value *Func, ArrayRef<Value *> Args,
                          const Twine &NameStr,
                          Instruction *InsertBefore = nullptr) {
    return Create(cast<FunctionType>(
                      cast<PointerType>(Func->getType())->getElementType()),
                  Func, Args, None, NameStr, InsertBefore);
  }

  static CallInst *Create(FunctionType *Ty, Value *Func, ArrayRef<Value *> Args,
                          const Twine &NameStr,
                          Instruction *InsertBefore = nullptr) {
    return new (unsigned(Args.size() + 1))
        CallInst(Ty, Func, Args, None, NameStr, InsertBefore);
  }

  static CallInst *Create(FunctionType *Ty, Value *Func, ArrayRef<Value *> Args,
                          ArrayRef<OperandBundleDef> Bundles = None,
                          const Twine &NameStr = "",
                          Instruction *InsertBefore = nullptr) {
    const unsigned TotalOps =
        unsigned(Args.size()) + CountBundleInputs(Bundles) + 1;
    const unsigned DescriptorBytes = Bundles.size() * sizeof(BundleOpInfo);

    return new (TotalOps, DescriptorBytes)
        CallInst(Ty, Func, Args, Bundles, NameStr, InsertBefore);
  }

  static CallInst *Create(Value *Func, ArrayRef<Value *> Args,
                          ArrayRef<OperandBundleDef> Bundles,
                          const Twine &NameStr, BasicBlock *InsertAtEnd) {
    const unsigned TotalOps =
        unsigned(Args.size()) + CountBundleInputs(Bundles) + 1;
    const unsigned DescriptorBytes = Bundles.size() * sizeof(BundleOpInfo);

    return new (TotalOps, DescriptorBytes)
        CallInst(Func, Args, Bundles, NameStr, InsertAtEnd);
  }

  static CallInst *Create(Value *Func, ArrayRef<Value *> Args,
                          const Twine &NameStr, BasicBlock *InsertAtEnd) {
    return new (unsigned(Args.size() + 1))
        CallInst(Func, Args, None, NameStr, InsertAtEnd);
  }

  static CallInst *Create(Value *F, const Twine &NameStr = "",
                          Instruction *InsertBefore = nullptr) {
    return new(1) CallInst(F, NameStr, InsertBefore);
  }

  static CallInst *Create(Value *F, const Twine &NameStr,
                          BasicBlock *InsertAtEnd) {
    return new(1) CallInst(F, NameStr, InsertAtEnd);
  }

  /// Create a clone of \p CI with a different set of operand bundles and
  /// insert it before \p InsertPt.
  ///
  /// The returned call instruction is identical \p CI in every way except that
  /// the operand bundles for the new instruction are set to the operand bundles
  /// in \p Bundles.
  static CallInst *Create(CallInst *CI, ArrayRef<OperandBundleDef> Bundles,
                          Instruction *InsertPt = nullptr);

  /// Generate the IR for a call to malloc:
  /// 1. Compute the malloc call's argument as the specified type's size,
  ///    possibly multiplied by the array size if the array size is not
  ///    constant 1.
  /// 2. Call malloc with that argument.
  /// 3. Bitcast the result of the malloc call to the specified type.
  static Instruction *CreateMalloc(Instruction *InsertBefore,
                                   Type *IntPtrTy, Type *AllocTy,
                                   Value *AllocSize, Value *ArraySize = nullptr,
                                   Function* MallocF = nullptr,
                                   const Twine &Name = "");
  static Instruction *CreateMalloc(BasicBlock *InsertAtEnd,
                                   Type *IntPtrTy, Type *AllocTy,
                                   Value *AllocSize, Value *ArraySize = nullptr,
                                   Function* MallocF = nullptr,
                                   const Twine &Name = "");
  static Instruction *CreateMalloc(Instruction *InsertBefore,
                                   Type *IntPtrTy, Type *AllocTy,
                                   Value *AllocSize, Value *ArraySize = nullptr,
                                   ArrayRef<OperandBundleDef> Bundles = None,
                                   Function* MallocF = nullptr,
                                   const Twine &Name = "");
  static Instruction *CreateMalloc(BasicBlock *InsertAtEnd,
                                   Type *IntPtrTy, Type *AllocTy,
                                   Value *AllocSize, Value *ArraySize = nullptr,
                                   ArrayRef<OperandBundleDef> Bundles = None,
                                   Function* MallocF = nullptr,
                                   const Twine &Name = "");
  /// Generate the IR for a call to the builtin free function.
  static Instruction *CreateFree(Value *Source,
                                 Instruction *InsertBefore);
  static Instruction *CreateFree(Value *Source,
                                 BasicBlock *InsertAtEnd);
  static Instruction *CreateFree(Value *Source,
                                 ArrayRef<OperandBundleDef> Bundles,
                                 Instruction *InsertBefore);
  static Instruction *CreateFree(Value *Source,
                                 ArrayRef<OperandBundleDef> Bundles,
                                 BasicBlock *InsertAtEnd);

  FunctionType *getFunctionType() const { return FTy; }

  void mutateFunctionType(FunctionType *FTy) {
    mutateType(FTy->getReturnType());
    this->FTy = FTy;
  }

  // Note that 'musttail' implies 'tail'.
  enum TailCallKind { TCK_None = 0, TCK_Tail = 1, TCK_MustTail = 2,
                      TCK_NoTail = 3 };
  TailCallKind getTailCallKind() const {
    return TailCallKind(getSubclassDataFromInstruction() & 3);
  }

  bool isTailCall() const {
    unsigned Kind = getSubclassDataFromInstruction() & 3;
    return Kind == TCK_Tail || Kind == TCK_MustTail;
  }

  bool isMustTailCall() const {
    return (getSubclassDataFromInstruction() & 3) == TCK_MustTail;
  }

  bool isNoTailCall() const {
    return (getSubclassDataFromInstruction() & 3) == TCK_NoTail;
  }

  void setTailCall(bool isTC = true) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~3) |
                               unsigned(isTC ? TCK_Tail : TCK_None));
  }

  void setTailCallKind(TailCallKind TCK) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~3) |
                               unsigned(TCK));
  }

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// Return the number of call arguments.
  ///
  unsigned getNumArgOperands() const {
    return getNumOperands() - getNumTotalBundleOperands() - 1;
  }

  /// getArgOperand/setArgOperand - Return/set the i-th call argument.
  ///
  Value *getArgOperand(unsigned i) const {
    assert(i < getNumArgOperands() && "Out of bounds!");
    return getOperand(i);
  }
  void setArgOperand(unsigned i, Value *v) {
    assert(i < getNumArgOperands() && "Out of bounds!");
    setOperand(i, v);
  }

  /// Return the iterator pointing to the beginning of the argument list.
  op_iterator arg_begin() { return op_begin(); }

  /// Return the iterator pointing to the end of the argument list.
  op_iterator arg_end() {
    // [ call args ], [ operand bundles ], callee
    return op_end() - getNumTotalBundleOperands() - 1;
  }

  /// Iteration adapter for range-for loops.
  iterator_range<op_iterator> arg_operands() {
    return make_range(arg_begin(), arg_end());
  }

  /// Return the iterator pointing to the beginning of the argument list.
  const_op_iterator arg_begin() const { return op_begin(); }

  /// Return the iterator pointing to the end of the argument list.
  const_op_iterator arg_end() const {
    // [ call args ], [ operand bundles ], callee
    return op_end() - getNumTotalBundleOperands() - 1;
  }

  /// Iteration adapter for range-for loops.
  iterator_range<const_op_iterator> arg_operands() const {
    return make_range(arg_begin(), arg_end());
  }

  /// Wrappers for getting the \c Use of a call argument.
  const Use &getArgOperandUse(unsigned i) const {
    assert(i < getNumArgOperands() && "Out of bounds!");
    return getOperandUse(i);
  }
  Use &getArgOperandUse(unsigned i) {
    assert(i < getNumArgOperands() && "Out of bounds!");
    return getOperandUse(i);
  }

  /// If one of the arguments has the 'returned' attribute, return its
  /// operand value. Otherwise, return nullptr.
  Value *getReturnedArgOperand() const;

  /// getCallingConv/setCallingConv - Get or set the calling convention of this
  /// function call.
  CallingConv::ID getCallingConv() const {
    return static_cast<CallingConv::ID>(getSubclassDataFromInstruction() >> 2);
  }
  void setCallingConv(CallingConv::ID CC) {
    auto ID = static_cast<unsigned>(CC);
    assert(!(ID & ~CallingConv::MaxID) && "Unsupported calling convention");
    setInstructionSubclassData((getSubclassDataFromInstruction() & 3) |
                               (ID << 2));
  }

  /// Return the parameter attributes for this call.
  ///
  AttributeList getAttributes() const { return Attrs; }

  /// Set the parameter attributes for this call.
  ///
  void setAttributes(AttributeList A) { Attrs = A; }

  /// adds the attribute to the list of attributes.
  void addAttribute(unsigned i, Attribute::AttrKind Kind);

  /// adds the attribute to the list of attributes.
  void addAttribute(unsigned i, Attribute Attr);

  /// removes the attribute from the list of attributes.
  void removeAttribute(unsigned i, Attribute::AttrKind Kind);

  /// removes the attribute from the list of attributes.
  void removeAttribute(unsigned i, StringRef Kind);

  /// adds the dereferenceable attribute to the list of attributes.
  void addDereferenceableAttr(unsigned i, uint64_t Bytes);

  /// adds the dereferenceable_or_null attribute to the list of
  /// attributes.
  void addDereferenceableOrNullAttr(unsigned i, uint64_t Bytes);

  /// Determine whether this call has the given attribute.
  bool hasFnAttr(Attribute::AttrKind Kind) const {
    assert(Kind != Attribute::NoBuiltin &&
           "Use CallInst::isNoBuiltin() to check for Attribute::NoBuiltin");
    return hasFnAttrImpl(Kind);
  }

  /// Determine whether this call has the given attribute.
  bool hasFnAttr(StringRef Kind) const {
    return hasFnAttrImpl(Kind);
  }

  /// Determine whether the call or the callee has the given attributes.
  bool paramHasAttr(unsigned i, Attribute::AttrKind Kind) const;

  /// Get the attribute of a given kind at a position.
  Attribute getAttribute(unsigned i, Attribute::AttrKind Kind) const {
    return getAttributes().getAttribute(i, Kind);
  }

  /// Get the attribute of a given kind at a position.
  Attribute getAttribute(unsigned i, StringRef Kind) const {
    return getAttributes().getAttribute(i, Kind);
  }

  /// Return true if the data operand at index \p i has the attribute \p
  /// A.
  ///
  /// Data operands include call arguments and values used in operand bundles,
  /// but does not include the callee operand.  This routine dispatches to the
  /// underlying AttributeList or the OperandBundleUser as appropriate.
  ///
  /// The index \p i is interpreted as
  ///
  ///  \p i == Attribute::ReturnIndex  -> the return value
  ///  \p i in [1, arg_size + 1)  -> argument number (\p i - 1)
  ///  \p i in [arg_size + 1, data_operand_size + 1) -> bundle operand at index
  ///     (\p i - 1) in the operand list.
  bool dataOperandHasImpliedAttr(unsigned i, Attribute::AttrKind Kind) const;

  /// Extract the alignment for a call or parameter (0=unknown).
  unsigned getParamAlignment(unsigned i) const {
    return Attrs.getParamAlignment(i);
  }

  /// Extract the number of dereferenceable bytes for a call or
  /// parameter (0=unknown).
  uint64_t getDereferenceableBytes(unsigned i) const {
    return Attrs.getDereferenceableBytes(i);
  }

  /// Extract the number of dereferenceable_or_null bytes for a call or
  /// parameter (0=unknown).
  uint64_t getDereferenceableOrNullBytes(unsigned i) const {
    return Attrs.getDereferenceableOrNullBytes(i);
  }

  /// @brief Determine if the parameter or return value is marked with NoAlias
  /// attribute.
  /// @param n The parameter to check. 1 is the first parameter, 0 is the return
  bool doesNotAlias(unsigned n) const {
    return Attrs.hasAttribute(n, Attribute::NoAlias);
  }

  /// Return true if the call should not be treated as a call to a
  /// builtin.
  bool isNoBuiltin() const {
    return hasFnAttrImpl(Attribute::NoBuiltin) &&
      !hasFnAttrImpl(Attribute::Builtin);
  }

  /// Return true if the call should not be inlined.
  bool isNoInline() const { return hasFnAttr(Attribute::NoInline); }
  void setIsNoInline() {
    addAttribute(AttributeList::FunctionIndex, Attribute::NoInline);
  }

  /// Return true if the call can return twice
  bool canReturnTwice() const {
    return hasFnAttr(Attribute::ReturnsTwice);
  }
  void setCanReturnTwice() {
    addAttribute(AttributeList::FunctionIndex, Attribute::ReturnsTwice);
  }

  /// Determine if the call does not access memory.
  bool doesNotAccessMemory() const {
    return hasFnAttr(Attribute::ReadNone);
  }
  void setDoesNotAccessMemory() {
    addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
  }

  /// Determine if the call does not access or only reads memory.
  bool onlyReadsMemory() const {
    return doesNotAccessMemory() || hasFnAttr(Attribute::ReadOnly);
  }
  void setOnlyReadsMemory() {
    addAttribute(AttributeList::FunctionIndex, Attribute::ReadOnly);
  }

  /// Determine if the call does not access or only writes memory.
  bool doesNotReadMemory() const {
    return doesNotAccessMemory() || hasFnAttr(Attribute::WriteOnly);
  }
  void setDoesNotReadMemory() {
    addAttribute(AttributeList::FunctionIndex, Attribute::WriteOnly);
  }

  /// @brief Determine if the call can access memmory only using pointers based
  /// on its arguments.
  bool onlyAccessesArgMemory() const {
    return hasFnAttr(Attribute::ArgMemOnly);
  }
  void setOnlyAccessesArgMemory() {
    addAttribute(AttributeList::FunctionIndex, Attribute::ArgMemOnly);
  }

  /// Determine if the call cannot return.
  bool doesNotReturn() const { return hasFnAttr(Attribute::NoReturn); }
  void setDoesNotReturn() {
    addAttribute(AttributeList::FunctionIndex, Attribute::NoReturn);
  }

  /// Determine if the call cannot unwind.
  bool doesNotThrow() const { return hasFnAttr(Attribute::NoUnwind); }
  void setDoesNotThrow() {
    addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
  }

  /// Determine if the call cannot be duplicated.
  bool cannotDuplicate() const {return hasFnAttr(Attribute::NoDuplicate); }
  void setCannotDuplicate() {
    addAttribute(AttributeList::FunctionIndex, Attribute::NoDuplicate);
  }

  /// Determine if the call is convergent
  bool isConvergent() const { return hasFnAttr(Attribute::Convergent); }
  void setConvergent() {
    addAttribute(AttributeList::FunctionIndex, Attribute::Convergent);
  }
  void setNotConvergent() {
    removeAttribute(AttributeList::FunctionIndex, Attribute::Convergent);
  }

  /// Determine if the call returns a structure through first
  /// pointer argument.
  bool hasStructRetAttr() const {
    if (getNumArgOperands() == 0)
      return false;

    // Be friendly and also check the callee.
    return paramHasAttr(1, Attribute::StructRet);
  }

  /// Determine if any call argument is an aggregate passed by value.
  bool hasByValArgument() const {
    return Attrs.hasAttrSomewhere(Attribute::ByVal);
  }

  /// Return the function called, or null if this is an
  /// indirect function invocation.
  ///
  Function *getCalledFunction() const {
    return dyn_cast<Function>(Op<-1>());
  }

  /// Get a pointer to the function that is invoked by this
  /// instruction.
  const Value *getCalledValue() const { return Op<-1>(); }
        Value *getCalledValue()       { return Op<-1>(); }

  /// Set the function called.
  void setCalledFunction(Value* Fn) {
    setCalledFunction(
        cast<FunctionType>(cast<PointerType>(Fn->getType())->getElementType()),
        Fn);
  }
  void setCalledFunction(FunctionType *FTy, Value *Fn) {
    this->FTy = FTy;
    assert(FTy == cast<FunctionType>(
                      cast<PointerType>(Fn->getType())->getElementType()));
    Op<-1>() = Fn;
  }

  /// Check if this call is an inline asm statement.
  bool isInlineAsm() const {
    return isa<InlineAsm>(Op<-1>());
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Call;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  template <typename AttrKind> bool hasFnAttrImpl(AttrKind Kind) const {
    if (Attrs.hasAttribute(AttributeList::FunctionIndex, Kind))
      return true;

    // Operand bundles override attributes on the called function, but don't
    // override attributes directly present on the call instruction.
    if (isFnAttrDisallowedByOpBundle(Kind))
      return false;

    if (const Function *F = getCalledFunction())
      return F->getAttributes().hasAttribute(AttributeList::FunctionIndex,
                                             Kind);
    return false;
  }

  // Shadow Instruction::setInstructionSubclassData with a private forwarding
  // method so that subclasses cannot accidentally use it.
  void setInstructionSubclassData(unsigned short D) {
    Instruction::setInstructionSubclassData(D);
  }
};

template <>
struct OperandTraits<CallInst> : public VariadicOperandTraits<CallInst, 1> {
};

CallInst::CallInst(Value *Func, ArrayRef<Value *> Args,
                   ArrayRef<OperandBundleDef> Bundles, const Twine &NameStr,
                   BasicBlock *InsertAtEnd)
    : Instruction(
          cast<FunctionType>(cast<PointerType>(Func->getType())
                                 ->getElementType())->getReturnType(),
          Instruction::Call, OperandTraits<CallInst>::op_end(this) -
                                 (Args.size() + CountBundleInputs(Bundles) + 1),
          unsigned(Args.size() + CountBundleInputs(Bundles) + 1), InsertAtEnd) {
  init(Func, Args, Bundles, NameStr);
}

CallInst::CallInst(FunctionType *Ty, Value *Func, ArrayRef<Value *> Args,
                   ArrayRef<OperandBundleDef> Bundles, const Twine &NameStr,
                   Instruction *InsertBefore)
    : Instruction(Ty->getReturnType(), Instruction::Call,
                  OperandTraits<CallInst>::op_end(this) -
                      (Args.size() + CountBundleInputs(Bundles) + 1),
                  unsigned(Args.size() + CountBundleInputs(Bundles) + 1),
                  InsertBefore) {
  init(Ty, Func, Args, Bundles, NameStr);
}

// Note: if you get compile errors about private methods then
//       please update your code to use the high-level operand
//       interfaces. See line 943 above.
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(CallInst, Value)

//===----------------------------------------------------------------------===//
//                               SelectInst Class
//===----------------------------------------------------------------------===//

/// This class represents the LLVM 'select' instruction.
///
class SelectInst : public Instruction {
  SelectInst(Value *C, Value *S1, Value *S2, const Twine &NameStr,
             Instruction *InsertBefore)
    : Instruction(S1->getType(), Instruction::Select,
                  &Op<0>(), 3, InsertBefore) {
    init(C, S1, S2);
    setName(NameStr);
  }

  SelectInst(Value *C, Value *S1, Value *S2, const Twine &NameStr,
             BasicBlock *InsertAtEnd)
    : Instruction(S1->getType(), Instruction::Select,
                  &Op<0>(), 3, InsertAtEnd) {
    init(C, S1, S2);
    setName(NameStr);
  }

  void init(Value *C, Value *S1, Value *S2) {
    assert(!areInvalidOperands(C, S1, S2) && "Invalid operands for select");
    Op<0>() = C;
    Op<1>() = S1;
    Op<2>() = S2;
  }

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  SelectInst *cloneImpl() const;

public:
  static SelectInst *Create(Value *C, Value *S1, Value *S2,
                            const Twine &NameStr = "",
                            Instruction *InsertBefore = nullptr,
                            Instruction *MDFrom = nullptr) {
    SelectInst *Sel = new(3) SelectInst(C, S1, S2, NameStr, InsertBefore);
    if (MDFrom)
      Sel->copyMetadata(*MDFrom);
    return Sel;
  }

  static SelectInst *Create(Value *C, Value *S1, Value *S2,
                            const Twine &NameStr,
                            BasicBlock *InsertAtEnd) {
    return new(3) SelectInst(C, S1, S2, NameStr, InsertAtEnd);
  }

  const Value *getCondition() const { return Op<0>(); }
  const Value *getTrueValue() const { return Op<1>(); }
  const Value *getFalseValue() const { return Op<2>(); }
  Value *getCondition() { return Op<0>(); }
  Value *getTrueValue() { return Op<1>(); }
  Value *getFalseValue() { return Op<2>(); }

  void setCondition(Value *V) { Op<0>() = V; }
  void setTrueValue(Value *V) { Op<1>() = V; }
  void setFalseValue(Value *V) { Op<2>() = V; }

  /// Return a string if the specified operands are invalid
  /// for a select operation, otherwise return null.
  static const char *areInvalidOperands(Value *Cond, Value *True, Value *False);

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  OtherOps getOpcode() const {
    return static_cast<OtherOps>(Instruction::getOpcode());
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Select;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<SelectInst> : public FixedNumOperandTraits<SelectInst, 3> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(SelectInst, Value)

//===----------------------------------------------------------------------===//
//                                VAArgInst Class
//===----------------------------------------------------------------------===//

/// This class represents the va_arg llvm instruction, which returns
/// an argument of the specified type given a va_list and increments that list
///
class VAArgInst : public UnaryInstruction {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  VAArgInst *cloneImpl() const;

public:
  VAArgInst(Value *List, Type *Ty, const Twine &NameStr = "",
             Instruction *InsertBefore = nullptr)
    : UnaryInstruction(Ty, VAArg, List, InsertBefore) {
    setName(NameStr);
  }

  VAArgInst(Value *List, Type *Ty, const Twine &NameStr,
            BasicBlock *InsertAtEnd)
    : UnaryInstruction(Ty, VAArg, List, InsertAtEnd) {
    setName(NameStr);
  }

  Value *getPointerOperand() { return getOperand(0); }
  const Value *getPointerOperand() const { return getOperand(0); }
  static unsigned getPointerOperandIndex() { return 0U; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == VAArg;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                ExtractElementInst Class
//===----------------------------------------------------------------------===//

/// This instruction extracts a single (scalar)
/// element from a VectorType value
///
class ExtractElementInst : public Instruction {
  ExtractElementInst(Value *Vec, Value *Idx, const Twine &NameStr = "",
                     Instruction *InsertBefore = nullptr);
  ExtractElementInst(Value *Vec, Value *Idx, const Twine &NameStr,
                     BasicBlock *InsertAtEnd);

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  ExtractElementInst *cloneImpl() const;

public:
  static ExtractElementInst *Create(Value *Vec, Value *Idx,
                                   const Twine &NameStr = "",
                                   Instruction *InsertBefore = nullptr) {
    return new(2) ExtractElementInst(Vec, Idx, NameStr, InsertBefore);
  }

  static ExtractElementInst *Create(Value *Vec, Value *Idx,
                                   const Twine &NameStr,
                                   BasicBlock *InsertAtEnd) {
    return new(2) ExtractElementInst(Vec, Idx, NameStr, InsertAtEnd);
  }

  /// Return true if an extractelement instruction can be
  /// formed with the specified operands.
  static bool isValidOperands(const Value *Vec, const Value *Idx);

  Value *getVectorOperand() { return Op<0>(); }
  Value *getIndexOperand() { return Op<1>(); }
  const Value *getVectorOperand() const { return Op<0>(); }
  const Value *getIndexOperand() const { return Op<1>(); }

  VectorType *getVectorOperandType() const {
    return cast<VectorType>(getVectorOperand()->getType());
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::ExtractElement;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<ExtractElementInst> :
  public FixedNumOperandTraits<ExtractElementInst, 2> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ExtractElementInst, Value)

//===----------------------------------------------------------------------===//
//                                InsertElementInst Class
//===----------------------------------------------------------------------===//

/// This instruction inserts a single (scalar)
/// element into a VectorType value
///
class InsertElementInst : public Instruction {
  InsertElementInst(Value *Vec, Value *NewElt, Value *Idx,
                    const Twine &NameStr = "",
                    Instruction *InsertBefore = nullptr);
  InsertElementInst(Value *Vec, Value *NewElt, Value *Idx, const Twine &NameStr,
                    BasicBlock *InsertAtEnd);

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  InsertElementInst *cloneImpl() const;

public:
  static InsertElementInst *Create(Value *Vec, Value *NewElt, Value *Idx,
                                   const Twine &NameStr = "",
                                   Instruction *InsertBefore = nullptr) {
    return new(3) InsertElementInst(Vec, NewElt, Idx, NameStr, InsertBefore);
  }

  static InsertElementInst *Create(Value *Vec, Value *NewElt, Value *Idx,
                                   const Twine &NameStr,
                                   BasicBlock *InsertAtEnd) {
    return new(3) InsertElementInst(Vec, NewElt, Idx, NameStr, InsertAtEnd);
  }

  /// Return true if an insertelement instruction can be
  /// formed with the specified operands.
  static bool isValidOperands(const Value *Vec, const Value *NewElt,
                              const Value *Idx);

  /// Overload to return most specific vector type.
  ///
  VectorType *getType() const {
    return cast<VectorType>(Instruction::getType());
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::InsertElement;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<InsertElementInst> :
  public FixedNumOperandTraits<InsertElementInst, 3> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InsertElementInst, Value)

//===----------------------------------------------------------------------===//
//                           ShuffleVectorInst Class
//===----------------------------------------------------------------------===//

/// This instruction constructs a fixed permutation of two
/// input vectors.
///
class ShuffleVectorInst : public Instruction {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  ShuffleVectorInst *cloneImpl() const;

public:
  ShuffleVectorInst(Value *V1, Value *V2, Value *Mask,
                    const Twine &NameStr = "",
                    Instruction *InsertBefor = nullptr);
  ShuffleVectorInst(Value *V1, Value *V2, Value *Mask,
                    const Twine &NameStr, BasicBlock *InsertAtEnd);

  // allocate space for exactly three operands
  void *operator new(size_t s) {
    return User::operator new(s, 3);
  }

  /// Return true if a shufflevector instruction can be
  /// formed with the specified operands.
  static bool isValidOperands(const Value *V1, const Value *V2,
                              const Value *Mask);

  /// Overload to return most specific vector type.
  ///
  VectorType *getType() const {
    return cast<VectorType>(Instruction::getType());
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  Constant *getMask() const {
    return cast<Constant>(getOperand(2));
  }

  /// Return the shuffle mask value for the specified element of the mask.
  /// Return -1 if the element is undef.
  static int getMaskValue(Constant *Mask, unsigned Elt);

  /// Return the shuffle mask value of this instruction for the given element
  /// index. Return -1 if the element is undef.
  int getMaskValue(unsigned Elt) const {
    return getMaskValue(getMask(), Elt);
  }

  /// Convert the input shuffle mask operand to a vector of integers. Undefined
  /// elements of the mask are returned as -1.
  static void getShuffleMask(Constant *Mask, SmallVectorImpl<int> &Result);

  /// Return the mask for this instruction as a vector of integers. Undefined
  /// elements of the mask are returned as -1.
  void getShuffleMask(SmallVectorImpl<int> &Result) const {
    return getShuffleMask(getMask(), Result);
  }

  SmallVector<int, 16> getShuffleMask() const {
    SmallVector<int, 16> Mask;
    getShuffleMask(Mask);
    return Mask;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::ShuffleVector;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<ShuffleVectorInst> :
  public FixedNumOperandTraits<ShuffleVectorInst, 3> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ShuffleVectorInst, Value)

//===----------------------------------------------------------------------===//
//                                ExtractValueInst Class
//===----------------------------------------------------------------------===//

/// This instruction extracts a struct member or array
/// element value from an aggregate value.
///
class ExtractValueInst : public UnaryInstruction {
  SmallVector<unsigned, 4> Indices;

  ExtractValueInst(const ExtractValueInst &EVI);
  /// Constructors - Create a extractvalue instruction with a base aggregate
  /// value and a list of indices.  The first ctor can optionally insert before
  /// an existing instruction, the second appends the new instruction to the
  /// specified BasicBlock.
  inline ExtractValueInst(Value *Agg,
                          ArrayRef<unsigned> Idxs,
                          const Twine &NameStr,
                          Instruction *InsertBefore);
  inline ExtractValueInst(Value *Agg,
                          ArrayRef<unsigned> Idxs,
                          const Twine &NameStr, BasicBlock *InsertAtEnd);

  // allocate space for exactly one operand
  void *operator new(size_t s) { return User::operator new(s, 1); }

  void init(ArrayRef<unsigned> Idxs, const Twine &NameStr);

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  ExtractValueInst *cloneImpl() const;

public:
  static ExtractValueInst *Create(Value *Agg,
                                  ArrayRef<unsigned> Idxs,
                                  const Twine &NameStr = "",
                                  Instruction *InsertBefore = nullptr) {
    return new
      ExtractValueInst(Agg, Idxs, NameStr, InsertBefore);
  }

  static ExtractValueInst *Create(Value *Agg,
                                  ArrayRef<unsigned> Idxs,
                                  const Twine &NameStr,
                                  BasicBlock *InsertAtEnd) {
    return new ExtractValueInst(Agg, Idxs, NameStr, InsertAtEnd);
  }

  /// Returns the type of the element that would be extracted
  /// with an extractvalue instruction with the specified parameters.
  ///
  /// Null is returned if the indices are invalid for the specified type.
  static Type *getIndexedType(Type *Agg, ArrayRef<unsigned> Idxs);

  typedef const unsigned* idx_iterator;
  inline idx_iterator idx_begin() const { return Indices.begin(); }
  inline idx_iterator idx_end()   const { return Indices.end(); }
  inline iterator_range<idx_iterator> indices() const {
    return make_range(idx_begin(), idx_end());
  }

  Value *getAggregateOperand() {
    return getOperand(0);
  }
  const Value *getAggregateOperand() const {
    return getOperand(0);
  }
  static unsigned getAggregateOperandIndex() {
    return 0U;                      // get index for modifying correct operand
  }

  ArrayRef<unsigned> getIndices() const {
    return Indices;
  }

  unsigned getNumIndices() const {
    return (unsigned)Indices.size();
  }

  bool hasIndices() const {
    return true;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::ExtractValue;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

ExtractValueInst::ExtractValueInst(Value *Agg,
                                   ArrayRef<unsigned> Idxs,
                                   const Twine &NameStr,
                                   Instruction *InsertBefore)
  : UnaryInstruction(checkGEPType(getIndexedType(Agg->getType(), Idxs)),
                     ExtractValue, Agg, InsertBefore) {
  init(Idxs, NameStr);
}

ExtractValueInst::ExtractValueInst(Value *Agg,
                                   ArrayRef<unsigned> Idxs,
                                   const Twine &NameStr,
                                   BasicBlock *InsertAtEnd)
  : UnaryInstruction(checkGEPType(getIndexedType(Agg->getType(), Idxs)),
                     ExtractValue, Agg, InsertAtEnd) {
  init(Idxs, NameStr);
}

//===----------------------------------------------------------------------===//
//                                InsertValueInst Class
//===----------------------------------------------------------------------===//

/// This instruction inserts a struct field of array element
/// value into an aggregate value.
///
class InsertValueInst : public Instruction {
  SmallVector<unsigned, 4> Indices;

  InsertValueInst(const InsertValueInst &IVI);

  /// Constructors - Create a insertvalue instruction with a base aggregate
  /// value, a value to insert, and a list of indices.  The first ctor can
  /// optionally insert before an existing instruction, the second appends
  /// the new instruction to the specified BasicBlock.
  inline InsertValueInst(Value *Agg, Value *Val,
                         ArrayRef<unsigned> Idxs,
                         const Twine &NameStr,
                         Instruction *InsertBefore);
  inline InsertValueInst(Value *Agg, Value *Val,
                         ArrayRef<unsigned> Idxs,
                         const Twine &NameStr, BasicBlock *InsertAtEnd);

  /// Constructors - These two constructors are convenience methods because one
  /// and two index insertvalue instructions are so common.
  InsertValueInst(Value *Agg, Value *Val, unsigned Idx,
                  const Twine &NameStr = "",
                  Instruction *InsertBefore = nullptr);
  InsertValueInst(Value *Agg, Value *Val, unsigned Idx, const Twine &NameStr,
                  BasicBlock *InsertAtEnd);

  void init(Value *Agg, Value *Val, ArrayRef<unsigned> Idxs,
            const Twine &NameStr);

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  InsertValueInst *cloneImpl() const;

public:
  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }

  void *operator new(size_t, unsigned) = delete;

  static InsertValueInst *Create(Value *Agg, Value *Val,
                                 ArrayRef<unsigned> Idxs,
                                 const Twine &NameStr = "",
                                 Instruction *InsertBefore = nullptr) {
    return new InsertValueInst(Agg, Val, Idxs, NameStr, InsertBefore);
  }

  static InsertValueInst *Create(Value *Agg, Value *Val,
                                 ArrayRef<unsigned> Idxs,
                                 const Twine &NameStr,
                                 BasicBlock *InsertAtEnd) {
    return new InsertValueInst(Agg, Val, Idxs, NameStr, InsertAtEnd);
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  typedef const unsigned* idx_iterator;
  inline idx_iterator idx_begin() const { return Indices.begin(); }
  inline idx_iterator idx_end()   const { return Indices.end(); }
  inline iterator_range<idx_iterator> indices() const {
    return make_range(idx_begin(), idx_end());
  }

  Value *getAggregateOperand() {
    return getOperand(0);
  }
  const Value *getAggregateOperand() const {
    return getOperand(0);
  }
  static unsigned getAggregateOperandIndex() {
    return 0U;                      // get index for modifying correct operand
  }

  Value *getInsertedValueOperand() {
    return getOperand(1);
  }
  const Value *getInsertedValueOperand() const {
    return getOperand(1);
  }
  static unsigned getInsertedValueOperandIndex() {
    return 1U;                      // get index for modifying correct operand
  }

  ArrayRef<unsigned> getIndices() const {
    return Indices;
  }

  unsigned getNumIndices() const {
    return (unsigned)Indices.size();
  }

  bool hasIndices() const {
    return true;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::InsertValue;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<InsertValueInst> :
  public FixedNumOperandTraits<InsertValueInst, 2> {
};

InsertValueInst::InsertValueInst(Value *Agg,
                                 Value *Val,
                                 ArrayRef<unsigned> Idxs,
                                 const Twine &NameStr,
                                 Instruction *InsertBefore)
  : Instruction(Agg->getType(), InsertValue,
                OperandTraits<InsertValueInst>::op_begin(this),
                2, InsertBefore) {
  init(Agg, Val, Idxs, NameStr);
}

InsertValueInst::InsertValueInst(Value *Agg,
                                 Value *Val,
                                 ArrayRef<unsigned> Idxs,
                                 const Twine &NameStr,
                                 BasicBlock *InsertAtEnd)
  : Instruction(Agg->getType(), InsertValue,
                OperandTraits<InsertValueInst>::op_begin(this),
                2, InsertAtEnd) {
  init(Agg, Val, Idxs, NameStr);
}

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InsertValueInst, Value)

//===----------------------------------------------------------------------===//
//                               PHINode Class
//===----------------------------------------------------------------------===//

// PHINode - The PHINode class is used to represent the magical mystical PHI
// node, that can not exist in nature, but can be synthesized in a computer
// scientist's overactive imagination.
//
class PHINode : public Instruction {
  /// The number of operands actually allocated.  NumOperands is
  /// the number actually in use.
  unsigned ReservedSpace;

  PHINode(const PHINode &PN);
  // allocate space for exactly zero operands

  explicit PHINode(Type *Ty, unsigned NumReservedValues,
                   const Twine &NameStr = "",
                   Instruction *InsertBefore = nullptr)
    : Instruction(Ty, Instruction::PHI, nullptr, 0, InsertBefore),
      ReservedSpace(NumReservedValues) {
    setName(NameStr);
    allocHungoffUses(ReservedSpace);
  }

  PHINode(Type *Ty, unsigned NumReservedValues, const Twine &NameStr,
          BasicBlock *InsertAtEnd)
    : Instruction(Ty, Instruction::PHI, nullptr, 0, InsertAtEnd),
      ReservedSpace(NumReservedValues) {
    setName(NameStr);
    allocHungoffUses(ReservedSpace);
  }

  void *operator new(size_t s) {
    return User::operator new(s);
  }

  void anchor() override;

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  PHINode *cloneImpl() const;

  // allocHungoffUses - this is more complicated than the generic
  // User::allocHungoffUses, because we have to allocate Uses for the incoming
  // values and pointers to the incoming blocks, all in one allocation.
  void allocHungoffUses(unsigned N) {
    User::allocHungoffUses(N, /* IsPhi */ true);
  }

public:
  void *operator new(size_t, unsigned) = delete;

  /// Constructors - NumReservedValues is a hint for the number of incoming
  /// edges that this phi node will have (use 0 if you really have no idea).
  static PHINode *Create(Type *Ty, unsigned NumReservedValues,
                         const Twine &NameStr = "",
                         Instruction *InsertBefore = nullptr) {
    return new PHINode(Ty, NumReservedValues, NameStr, InsertBefore);
  }

  static PHINode *Create(Type *Ty, unsigned NumReservedValues,
                         const Twine &NameStr, BasicBlock *InsertAtEnd) {
    return new PHINode(Ty, NumReservedValues, NameStr, InsertAtEnd);
  }

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  // Block iterator interface. This provides access to the list of incoming
  // basic blocks, which parallels the list of incoming values.

  typedef BasicBlock **block_iterator;
  typedef BasicBlock * const *const_block_iterator;

  block_iterator block_begin() {
    Use::UserRef *ref =
      reinterpret_cast<Use::UserRef*>(op_begin() + ReservedSpace);
    return reinterpret_cast<block_iterator>(ref + 1);
  }

  const_block_iterator block_begin() const {
    const Use::UserRef *ref =
      reinterpret_cast<const Use::UserRef*>(op_begin() + ReservedSpace);
    return reinterpret_cast<const_block_iterator>(ref + 1);
  }

  block_iterator block_end() {
    return block_begin() + getNumOperands();
  }

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

  /// Return the number of incoming edges
  ///
  unsigned getNumIncomingValues() const { return getNumOperands(); }

  /// Return incoming value number x
  ///
  Value *getIncomingValue(unsigned i) const {
    return getOperand(i);
  }
  void setIncomingValue(unsigned i, Value *V) {
    assert(V && "PHI node got a null value!");
    assert(getType() == V->getType() &&
           "All operands to PHI node must be the same type as the PHI node!");
    setOperand(i, V);
  }
  static unsigned getOperandNumForIncomingValue(unsigned i) {
    return i;
  }
  static unsigned getIncomingValueNumForOperand(unsigned i) {
    return i;
  }

  /// Return incoming basic block number @p i.
  ///
  BasicBlock *getIncomingBlock(unsigned i) const {
    return block_begin()[i];
  }

  /// Return incoming basic block corresponding
  /// to an operand of the PHI.
  ///
  BasicBlock *getIncomingBlock(const Use &U) const {
    assert(this == U.getUser() && "Iterator doesn't point to PHI's Uses?");
    return getIncomingBlock(unsigned(&U - op_begin()));
  }

  /// Return incoming basic block corresponding
  /// to value use iterator.
  ///
  BasicBlock *getIncomingBlock(Value::const_user_iterator I) const {
    return getIncomingBlock(I.getUse());
  }

  void setIncomingBlock(unsigned i, BasicBlock *BB) {
    assert(BB && "PHI node got a null basic block!");
    block_begin()[i] = BB;
  }

  /// Add an incoming value to the end of the PHI list
  ///
  void addIncoming(Value *V, BasicBlock *BB) {
    if (getNumOperands() == ReservedSpace)
      growOperands();  // Get more space!
    // Initialize some new operands.
    setNumHungOffUseOperands(getNumOperands() + 1);
    setIncomingValue(getNumOperands() - 1, V);
    setIncomingBlock(getNumOperands() - 1, BB);
  }

  /// Remove an incoming value.  This is useful if a
  /// predecessor basic block is deleted.  The value removed is returned.
  ///
  /// If the last incoming value for a PHI node is removed (and DeletePHIIfEmpty
  /// is true), the PHI node is destroyed and any uses of it are replaced with
  /// dummy values.  The only time there should be zero incoming values to a PHI
  /// node is when the block is dead, so this strategy is sound.
  ///
  Value *removeIncomingValue(unsigned Idx, bool DeletePHIIfEmpty = true);

  Value *removeIncomingValue(const BasicBlock *BB, bool DeletePHIIfEmpty=true) {
    int Idx = getBasicBlockIndex(BB);
    assert(Idx >= 0 && "Invalid basic block argument to remove!");
    return removeIncomingValue(Idx, DeletePHIIfEmpty);
  }

  /// Return the first index of the specified basic
  /// block in the value list for this PHI.  Returns -1 if no instance.
  ///
  int getBasicBlockIndex(const BasicBlock *BB) const {
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
      if (block_begin()[i] == BB)
        return i;
    return -1;
  }

  Value *getIncomingValueForBlock(const BasicBlock *BB) const {
    int Idx = getBasicBlockIndex(BB);
    assert(Idx >= 0 && "Invalid basic block argument!");
    return getIncomingValue(Idx);
  }

  /// If the specified PHI node always merges together the
  /// same value, return the value, otherwise return null.
  Value *hasConstantValue() const;

  /// Whether the specified PHI node always merges
  /// together the same value, assuming undefs are equal to a unique
  /// non-undef value.
  bool hasConstantOrUndefValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::PHI;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  void growOperands();
};

template <>
struct OperandTraits<PHINode> : public HungoffOperandTraits<2> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(PHINode, Value)

//===----------------------------------------------------------------------===//
//                           LandingPadInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// The landingpad instruction holds all of the information
/// necessary to generate correct exception handling. The landingpad instruction
/// cannot be moved from the top of a landing pad block, which itself is
/// accessible only from the 'unwind' edge of an invoke. This uses the
/// SubclassData field in Value to store whether or not the landingpad is a
/// cleanup.
///
class LandingPadInst : public Instruction {
  /// The number of operands actually allocated.  NumOperands is
  /// the number actually in use.
  unsigned ReservedSpace;

  LandingPadInst(const LandingPadInst &LP);

public:
  enum ClauseType { Catch, Filter };

private:
  explicit LandingPadInst(Type *RetTy, unsigned NumReservedValues,
                          const Twine &NameStr, Instruction *InsertBefore);
  explicit LandingPadInst(Type *RetTy, unsigned NumReservedValues,
                          const Twine &NameStr, BasicBlock *InsertAtEnd);

  // Allocate space for exactly zero operands.
  void *operator new(size_t s) {
    return User::operator new(s);
  }

  void growOperands(unsigned Size);
  void init(unsigned NumReservedValues, const Twine &NameStr);

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  LandingPadInst *cloneImpl() const;

public:
  void *operator new(size_t, unsigned) = delete;

  /// Constructors - NumReservedClauses is a hint for the number of incoming
  /// clauses that this landingpad will have (use 0 if you really have no idea).
  static LandingPadInst *Create(Type *RetTy, unsigned NumReservedClauses,
                                const Twine &NameStr = "",
                                Instruction *InsertBefore = nullptr);
  static LandingPadInst *Create(Type *RetTy, unsigned NumReservedClauses,
                                const Twine &NameStr, BasicBlock *InsertAtEnd);

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// Return 'true' if this landingpad instruction is a
  /// cleanup. I.e., it should be run when unwinding even if its landing pad
  /// doesn't catch the exception.
  bool isCleanup() const { return getSubclassDataFromInstruction() & 1; }

  /// Indicate that this landingpad instruction is a cleanup.
  void setCleanup(bool V) {
    setInstructionSubclassData((getSubclassDataFromInstruction() & ~1) |
                               (V ? 1 : 0));
  }

  /// Add a catch or filter clause to the landing pad.
  void addClause(Constant *ClauseVal);

  /// Get the value of the clause at index Idx. Use isCatch/isFilter to
  /// determine what type of clause this is.
  Constant *getClause(unsigned Idx) const {
    return cast<Constant>(getOperandList()[Idx]);
  }

  /// Return 'true' if the clause and index Idx is a catch clause.
  bool isCatch(unsigned Idx) const {
    return !isa<ArrayType>(getOperandList()[Idx]->getType());
  }

  /// Return 'true' if the clause and index Idx is a filter clause.
  bool isFilter(unsigned Idx) const {
    return isa<ArrayType>(getOperandList()[Idx]->getType());
  }

  /// Get the number of clauses for this landing pad.
  unsigned getNumClauses() const { return getNumOperands(); }

  /// Grow the size of the operand list to accommodate the new
  /// number of clauses.
  void reserveClauses(unsigned Size) { growOperands(Size); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::LandingPad;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<LandingPadInst> : public HungoffOperandTraits<1> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(LandingPadInst, Value)

//===----------------------------------------------------------------------===//
//                               ReturnInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// Return a value (possibly void), from a function.  Execution
/// does not continue in this function any longer.
///
class ReturnInst : public TerminatorInst {
  ReturnInst(const ReturnInst &RI);

private:
  // ReturnInst constructors:
  // ReturnInst()                  - 'ret void' instruction
  // ReturnInst(    null)          - 'ret void' instruction
  // ReturnInst(Value* X)          - 'ret X'    instruction
  // ReturnInst(    null, Inst *I) - 'ret void' instruction, insert before I
  // ReturnInst(Value* X, Inst *I) - 'ret X'    instruction, insert before I
  // ReturnInst(    null, BB *B)   - 'ret void' instruction, insert @ end of B
  // ReturnInst(Value* X, BB *B)   - 'ret X'    instruction, insert @ end of B
  //
  // NOTE: If the Value* passed is of type void then the constructor behaves as
  // if it was passed NULL.
  explicit ReturnInst(LLVMContext &C, Value *retVal = nullptr,
                      Instruction *InsertBefore = nullptr);
  ReturnInst(LLVMContext &C, Value *retVal, BasicBlock *InsertAtEnd);
  explicit ReturnInst(LLVMContext &C, BasicBlock *InsertAtEnd);

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  ReturnInst *cloneImpl() const;

public:
  ~ReturnInst() override;

  static ReturnInst* Create(LLVMContext &C, Value *retVal = nullptr,
                            Instruction *InsertBefore = nullptr) {
    return new(!!retVal) ReturnInst(C, retVal, InsertBefore);
  }

  static ReturnInst* Create(LLVMContext &C, Value *retVal,
                            BasicBlock *InsertAtEnd) {
    return new(!!retVal) ReturnInst(C, retVal, InsertAtEnd);
  }

  static ReturnInst* Create(LLVMContext &C, BasicBlock *InsertAtEnd) {
    return new(0) ReturnInst(C, InsertAtEnd);
  }

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// Convenience accessor. Returns null if there is no return value.
  Value *getReturnValue() const {
    return getNumOperands() != 0 ? getOperand(0) : nullptr;
  }

  unsigned getNumSuccessors() const { return 0; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Ret);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  BasicBlock *getSuccessorV(unsigned idx) const override;
  unsigned getNumSuccessorsV() const override;
  void setSuccessorV(unsigned idx, BasicBlock *B) override;
};

template <>
struct OperandTraits<ReturnInst> : public VariadicOperandTraits<ReturnInst> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ReturnInst, Value)

//===----------------------------------------------------------------------===//
//                               BranchInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// Conditional or Unconditional Branch instruction.
///
class BranchInst : public TerminatorInst {
  /// Ops list - Branches are strange.  The operands are ordered:
  ///  [Cond, FalseDest,] TrueDest.  This makes some accessors faster because
  /// they don't have to check for cond/uncond branchness. These are mostly
  /// accessed relative from op_end().
  BranchInst(const BranchInst &BI);
  // BranchInst constructors (where {B, T, F} are blocks, and C is a condition):
  // BranchInst(BB *B)                           - 'br B'
  // BranchInst(BB* T, BB *F, Value *C)          - 'br C, T, F'
  // BranchInst(BB* B, Inst *I)                  - 'br B'        insert before I
  // BranchInst(BB* T, BB *F, Value *C, Inst *I) - 'br C, T, F', insert before I
  // BranchInst(BB* B, BB *I)                    - 'br B'        insert at end
  // BranchInst(BB* T, BB *F, Value *C, BB *I)   - 'br C, T, F', insert at end
  explicit BranchInst(BasicBlock *IfTrue, Instruction *InsertBefore = nullptr);
  BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *Cond,
             Instruction *InsertBefore = nullptr);
  BranchInst(BasicBlock *IfTrue, BasicBlock *InsertAtEnd);
  BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *Cond,
             BasicBlock *InsertAtEnd);

  void AssertOK();

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  BranchInst *cloneImpl() const;

public:
  static BranchInst *Create(BasicBlock *IfTrue,
                            Instruction *InsertBefore = nullptr) {
    return new(1) BranchInst(IfTrue, InsertBefore);
  }

  static BranchInst *Create(BasicBlock *IfTrue, BasicBlock *IfFalse,
                            Value *Cond, Instruction *InsertBefore = nullptr) {
    return new(3) BranchInst(IfTrue, IfFalse, Cond, InsertBefore);
  }

  static BranchInst *Create(BasicBlock *IfTrue, BasicBlock *InsertAtEnd) {
    return new(1) BranchInst(IfTrue, InsertAtEnd);
  }

  static BranchInst *Create(BasicBlock *IfTrue, BasicBlock *IfFalse,
                            Value *Cond, BasicBlock *InsertAtEnd) {
    return new(3) BranchInst(IfTrue, IfFalse, Cond, InsertAtEnd);
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  bool isUnconditional() const { return getNumOperands() == 1; }
  bool isConditional()   const { return getNumOperands() == 3; }

  Value *getCondition() const {
    assert(isConditional() && "Cannot get condition of an uncond branch!");
    return Op<-3>();
  }

  void setCondition(Value *V) {
    assert(isConditional() && "Cannot set condition of unconditional branch!");
    Op<-3>() = V;
  }

  unsigned getNumSuccessors() const { return 1+isConditional(); }

  BasicBlock *getSuccessor(unsigned i) const {
    assert(i < getNumSuccessors() && "Successor # out of range for Branch!");
    return cast_or_null<BasicBlock>((&Op<-1>() - i)->get());
  }

  void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(idx < getNumSuccessors() && "Successor # out of range for Branch!");
    *(&Op<-1>() - idx) = NewSucc;
  }

  /// Swap the successors of this branch instruction.
  ///
  /// Swaps the successors of the branch instruction. This also swaps any
  /// branch weight metadata associated with the instruction so that it
  /// continues to map correctly to each operand.
  void swapSuccessors();

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Br);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  BasicBlock *getSuccessorV(unsigned idx) const override;
  unsigned getNumSuccessorsV() const override;
  void setSuccessorV(unsigned idx, BasicBlock *B) override;
};

template <>
struct OperandTraits<BranchInst> : public VariadicOperandTraits<BranchInst, 1> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(BranchInst, Value)

//===----------------------------------------------------------------------===//
//                               SwitchInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// Multiway switch
///
class SwitchInst : public TerminatorInst {
  unsigned ReservedSpace;

  // Operand[0]    = Value to switch on
  // Operand[1]    = Default basic block destination
  // Operand[2n  ] = Value to match
  // Operand[2n+1] = BasicBlock to go to on match
  SwitchInst(const SwitchInst &SI);

  /// Create a new switch instruction, specifying a value to switch on and a
  /// default destination. The number of additional cases can be specified here
  /// to make memory allocation more efficient. This constructor can also
  /// auto-insert before another instruction.
  SwitchInst(Value *Value, BasicBlock *Default, unsigned NumCases,
             Instruction *InsertBefore);

  /// Create a new switch instruction, specifying a value to switch on and a
  /// default destination. The number of additional cases can be specified here
  /// to make memory allocation more efficient. This constructor also
  /// auto-inserts at the end of the specified BasicBlock.
  SwitchInst(Value *Value, BasicBlock *Default, unsigned NumCases,
             BasicBlock *InsertAtEnd);

  // allocate space for exactly zero operands
  void *operator new(size_t s) {
    return User::operator new(s);
  }

  void init(Value *Value, BasicBlock *Default, unsigned NumReserved);
  void growOperands();

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  SwitchInst *cloneImpl() const;

public:
  void *operator new(size_t, unsigned) = delete;

  // -2
  static const unsigned DefaultPseudoIndex = static_cast<unsigned>(~0L-1);

  template <typename CaseHandleT> class CaseIteratorImpl;

  /// A handle to a particular switch case. It exposes a convenient interface
  /// to both the case value and the successor block.
  ///
  /// We define this as a template and instantiate it to form both a const and
  /// non-const handle.
  template <typename SwitchInstT, typename ConstantIntT, typename BasicBlockT>
  class CaseHandleImpl {
    // Directly befriend both const and non-const iterators.
    friend class SwitchInst::CaseIteratorImpl<
        CaseHandleImpl<SwitchInstT, ConstantIntT, BasicBlockT>>;

  protected:
    // Expose the switch type we're parameterized with to the iterator.
    typedef SwitchInstT SwitchInstType;

    SwitchInstT *SI;
    ptrdiff_t Index;

    CaseHandleImpl() = default;
    CaseHandleImpl(SwitchInstT *SI, ptrdiff_t Index) : SI(SI), Index(Index) {}

  public:
    /// Resolves case value for current case.
    ConstantIntT *getCaseValue() const {
      assert((unsigned)Index < SI->getNumCases() &&
             "Index out the number of cases.");
      return reinterpret_cast<ConstantIntT *>(SI->getOperand(2 + Index * 2));
    }

    /// Resolves successor for current case.
    BasicBlockT *getCaseSuccessor() const {
      assert(((unsigned)Index < SI->getNumCases() ||
              (unsigned)Index == DefaultPseudoIndex) &&
             "Index out the number of cases.");
      return SI->getSuccessor(getSuccessorIndex());
    }

    /// Returns number of current case.
    unsigned getCaseIndex() const { return Index; }

    /// Returns TerminatorInst's successor index for current case successor.
    unsigned getSuccessorIndex() const {
      assert(((unsigned)Index == DefaultPseudoIndex ||
              (unsigned)Index < SI->getNumCases()) &&
             "Index out the number of cases.");
      return (unsigned)Index != DefaultPseudoIndex ? Index + 1 : 0;
    }

    bool operator==(const CaseHandleImpl &RHS) const {
      assert(SI == RHS.SI && "Incompatible operators.");
      return Index == RHS.Index;
    }
  };

  typedef CaseHandleImpl<const SwitchInst, const ConstantInt, const BasicBlock>
      ConstCaseHandle;

  class CaseHandle
      : public CaseHandleImpl<SwitchInst, ConstantInt, BasicBlock> {
    friend class SwitchInst::CaseIteratorImpl<CaseHandle>;

  public:
    CaseHandle(SwitchInst *SI, ptrdiff_t Index) : CaseHandleImpl(SI, Index) {}

    /// Sets the new value for current case.
    void setValue(ConstantInt *V) {
      assert((unsigned)Index < SI->getNumCases() &&
             "Index out the number of cases.");
      SI->setOperand(2 + Index*2, reinterpret_cast<Value*>(V));
    }

    /// Sets the new successor for current case.
    void setSuccessor(BasicBlock *S) {
      SI->setSuccessor(getSuccessorIndex(), S);
    }
  };

  template <typename CaseHandleT>
  class CaseIteratorImpl
      : public iterator_facade_base<CaseIteratorImpl<CaseHandleT>,
                                    std::random_access_iterator_tag,
                                    CaseHandleT> {
    typedef typename CaseHandleT::SwitchInstType SwitchInstT;

    CaseHandleT Case;

  public:
    /// Default constructed iterator is in an invalid state until assigned to
    /// a case for a particular switch.
    CaseIteratorImpl() = default;

    /// Initializes case iterator for given SwitchInst and for given
    /// case number.
    CaseIteratorImpl(SwitchInstT *SI, unsigned CaseNum) : Case(SI, CaseNum) {}

    /// Initializes case iterator for given SwitchInst and for given
    /// TerminatorInst's successor index.
    static CaseIteratorImpl fromSuccessorIndex(SwitchInstT *SI,
                                               unsigned SuccessorIndex) {
      assert(SuccessorIndex < SI->getNumSuccessors() &&
             "Successor index # out of range!");
      return SuccessorIndex != 0 ? CaseIteratorImpl(SI, SuccessorIndex - 1)
                                 : CaseIteratorImpl(SI, DefaultPseudoIndex);
    }

    /// Support converting to the const variant. This will be a no-op for const
    /// variant.
    operator CaseIteratorImpl<ConstCaseHandle>() const {
      return CaseIteratorImpl<ConstCaseHandle>(Case.SI, Case.Index);
    }

    CaseIteratorImpl &operator+=(ptrdiff_t N) {
      // Check index correctness after addition.
      // Note: Index == getNumCases() means end().
      assert(Case.Index + N >= 0 &&
             (unsigned)(Case.Index + N) <= Case.SI->getNumCases() &&
             "Case.Index out the number of cases.");
      Case.Index += N;
      return *this;
    }
    CaseIteratorImpl &operator-=(ptrdiff_t N) {
      // Check index correctness after subtraction.
      // Note: Case.Index == getNumCases() means end().
      assert(Case.Index - N >= 0 &&
             (unsigned)(Case.Index - N) <= Case.SI->getNumCases() &&
             "Case.Index out the number of cases.");
      Case.Index -= N;
      return *this;
    }
    ptrdiff_t operator-(const CaseIteratorImpl &RHS) const {
      assert(Case.SI == RHS.Case.SI && "Incompatible operators.");
      return Case.Index - RHS.Case.Index;
    }
    bool operator==(const CaseIteratorImpl &RHS) const {
      return Case == RHS.Case;
    }
    bool operator<(const CaseIteratorImpl &RHS) const {
      assert(Case.SI == RHS.Case.SI && "Incompatible operators.");
      return Case.Index < RHS.Case.Index;
    }
    CaseHandleT &operator*() { return Case; }
    const CaseHandleT &operator*() const { return Case; }
  };

  typedef CaseIteratorImpl<CaseHandle> CaseIt;
  typedef CaseIteratorImpl<ConstCaseHandle> ConstCaseIt;

  static SwitchInst *Create(Value *Value, BasicBlock *Default,
                            unsigned NumCases,
                            Instruction *InsertBefore = nullptr) {
    return new SwitchInst(Value, Default, NumCases, InsertBefore);
  }

  static SwitchInst *Create(Value *Value, BasicBlock *Default,
                            unsigned NumCases, BasicBlock *InsertAtEnd) {
    return new SwitchInst(Value, Default, NumCases, InsertAtEnd);
  }

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  // Accessor Methods for Switch stmt
  Value *getCondition() const { return getOperand(0); }
  void setCondition(Value *V) { setOperand(0, V); }

  BasicBlock *getDefaultDest() const {
    return cast<BasicBlock>(getOperand(1));
  }

  void setDefaultDest(BasicBlock *DefaultCase) {
    setOperand(1, reinterpret_cast<Value*>(DefaultCase));
  }

  /// Return the number of 'cases' in this switch instruction, excluding the
  /// default case.
  unsigned getNumCases() const {
    return getNumOperands()/2 - 1;
  }

  /// Returns a read/write iterator that points to the first case in the
  /// SwitchInst.
  CaseIt case_begin() {
    return CaseIt(this, 0);
  }

  /// Returns a read-only iterator that points to the first case in the
  /// SwitchInst.
  ConstCaseIt case_begin() const {
    return ConstCaseIt(this, 0);
  }

  /// Returns a read/write iterator that points one past the last in the
  /// SwitchInst.
  CaseIt case_end() {
    return CaseIt(this, getNumCases());
  }

  /// Returns a read-only iterator that points one past the last in the
  /// SwitchInst.
  ConstCaseIt case_end() const {
    return ConstCaseIt(this, getNumCases());
  }

  /// Iteration adapter for range-for loops.
  iterator_range<CaseIt> cases() {
    return make_range(case_begin(), case_end());
  }

  /// Constant iteration adapter for range-for loops.
  iterator_range<ConstCaseIt> cases() const {
    return make_range(case_begin(), case_end());
  }

  /// Returns an iterator that points to the default case.
  /// Note: this iterator allows to resolve successor only. Attempt
  /// to resolve case value causes an assertion.
  /// Also note, that increment and decrement also causes an assertion and
  /// makes iterator invalid.
  CaseIt case_default() {
    return CaseIt(this, DefaultPseudoIndex);
  }
  ConstCaseIt case_default() const {
    return ConstCaseIt(this, DefaultPseudoIndex);
  }

  /// Search all of the case values for the specified constant. If it is
  /// explicitly handled, return the case iterator of it, otherwise return
  /// default case iterator to indicate that it is handled by the default
  /// handler.
  CaseIt findCaseValue(const ConstantInt *C) {
    CaseIt I = llvm::find_if(
        cases(), [C](CaseHandle &Case) { return Case.getCaseValue() == C; });
    if (I != case_end())
      return I;

    return case_default();
  }
  ConstCaseIt findCaseValue(const ConstantInt *C) const {
    ConstCaseIt I = llvm::find_if(cases(), [C](ConstCaseHandle &Case) {
      return Case.getCaseValue() == C;
    });
    if (I != case_end())
      return I;

    return case_default();
  }

  /// Finds the unique case value for a given successor. Returns null if the
  /// successor is not found, not unique, or is the default case.
  ConstantInt *findCaseDest(BasicBlock *BB) {
    if (BB == getDefaultDest())
      return nullptr;

    ConstantInt *CI = nullptr;
    for (auto Case : cases()) {
      if (Case.getCaseSuccessor() != BB)
        continue;

      if (CI)
        return nullptr; // Multiple cases lead to BB.

      CI = Case.getCaseValue();
    }

    return CI;
  }

  /// Add an entry to the switch instruction.
  /// Note:
  /// This action invalidates case_end(). Old case_end() iterator will
  /// point to the added case.
  void addCase(ConstantInt *OnVal, BasicBlock *Dest);

  /// This method removes the specified case and its successor from the switch
  /// instruction. Note that this operation may reorder the remaining cases at
  /// index idx and above.
  /// Note:
  /// This action invalidates iterators for all cases following the one removed,
  /// including the case_end() iterator. It returns an iterator for the next
  /// case.
  CaseIt removeCase(CaseIt I);

  unsigned getNumSuccessors() const { return getNumOperands()/2; }
  BasicBlock *getSuccessor(unsigned idx) const {
    assert(idx < getNumSuccessors() &&"Successor idx out of range for switch!");
    return cast<BasicBlock>(getOperand(idx*2+1));
  }
  void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(idx < getNumSuccessors() && "Successor # out of range for switch!");
    setOperand(idx * 2 + 1, NewSucc);
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Switch;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  BasicBlock *getSuccessorV(unsigned idx) const override;
  unsigned getNumSuccessorsV() const override;
  void setSuccessorV(unsigned idx, BasicBlock *B) override;
};

template <>
struct OperandTraits<SwitchInst> : public HungoffOperandTraits<2> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(SwitchInst, Value)

//===----------------------------------------------------------------------===//
//                             IndirectBrInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// Indirect Branch Instruction.
///
class IndirectBrInst : public TerminatorInst {
  unsigned ReservedSpace;

  // Operand[0]   = Address to jump to
  // Operand[n+1] = n-th destination
  IndirectBrInst(const IndirectBrInst &IBI);

  /// Create a new indirectbr instruction, specifying an
  /// Address to jump to.  The number of expected destinations can be specified
  /// here to make memory allocation more efficient.  This constructor can also
  /// autoinsert before another instruction.
  IndirectBrInst(Value *Address, unsigned NumDests, Instruction *InsertBefore);

  /// Create a new indirectbr instruction, specifying an
  /// Address to jump to.  The number of expected destinations can be specified
  /// here to make memory allocation more efficient.  This constructor also
  /// autoinserts at the end of the specified BasicBlock.
  IndirectBrInst(Value *Address, unsigned NumDests, BasicBlock *InsertAtEnd);

  // allocate space for exactly zero operands
  void *operator new(size_t s) {
    return User::operator new(s);
  }

  void init(Value *Address, unsigned NumDests);
  void growOperands();

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  IndirectBrInst *cloneImpl() const;

public:
  void *operator new(size_t, unsigned) = delete;

  static IndirectBrInst *Create(Value *Address, unsigned NumDests,
                                Instruction *InsertBefore = nullptr) {
    return new IndirectBrInst(Address, NumDests, InsertBefore);
  }

  static IndirectBrInst *Create(Value *Address, unsigned NumDests,
                                BasicBlock *InsertAtEnd) {
    return new IndirectBrInst(Address, NumDests, InsertAtEnd);
  }

  /// Provide fast operand accessors.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  // Accessor Methods for IndirectBrInst instruction.
  Value *getAddress() { return getOperand(0); }
  const Value *getAddress() const { return getOperand(0); }
  void setAddress(Value *V) { setOperand(0, V); }

  /// return the number of possible destinations in this
  /// indirectbr instruction.
  unsigned getNumDestinations() const { return getNumOperands()-1; }

  /// Return the specified destination.
  BasicBlock *getDestination(unsigned i) { return getSuccessor(i); }
  const BasicBlock *getDestination(unsigned i) const { return getSuccessor(i); }

  /// Add a destination.
  ///
  void addDestination(BasicBlock *Dest);

  /// This method removes the specified successor from the
  /// indirectbr instruction.
  void removeDestination(unsigned i);

  unsigned getNumSuccessors() const { return getNumOperands()-1; }
  BasicBlock *getSuccessor(unsigned i) const {
    return cast<BasicBlock>(getOperand(i+1));
  }
  void setSuccessor(unsigned i, BasicBlock *NewSucc) {
    setOperand(i + 1, NewSucc);
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::IndirectBr;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  BasicBlock *getSuccessorV(unsigned idx) const override;
  unsigned getNumSuccessorsV() const override;
  void setSuccessorV(unsigned idx, BasicBlock *B) override;
};

template <>
struct OperandTraits<IndirectBrInst> : public HungoffOperandTraits<1> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(IndirectBrInst, Value)

//===----------------------------------------------------------------------===//
//                               InvokeInst Class
//===----------------------------------------------------------------------===//

/// Invoke instruction.  The SubclassData field is used to hold the
/// calling convention of the call.
///
class InvokeInst : public TerminatorInst,
                   public OperandBundleUser<InvokeInst, User::op_iterator> {
  friend class OperandBundleUser<InvokeInst, User::op_iterator>;

  AttributeList Attrs;
  FunctionType *FTy;

  InvokeInst(const InvokeInst &BI);

  /// Construct an InvokeInst given a range of arguments.
  ///
  /// Construct an InvokeInst from a range of arguments
  inline InvokeInst(Value *Func, BasicBlock *IfNormal, BasicBlock *IfException,
                    ArrayRef<Value *> Args, ArrayRef<OperandBundleDef> Bundles,
                    unsigned Values, const Twine &NameStr,
                    Instruction *InsertBefore)
      : InvokeInst(cast<FunctionType>(
                       cast<PointerType>(Func->getType())->getElementType()),
                   Func, IfNormal, IfException, Args, Bundles, Values, NameStr,
                   InsertBefore) {}

  inline InvokeInst(FunctionType *Ty, Value *Func, BasicBlock *IfNormal,
                    BasicBlock *IfException, ArrayRef<Value *> Args,
                    ArrayRef<OperandBundleDef> Bundles, unsigned Values,
                    const Twine &NameStr, Instruction *InsertBefore);
  /// Construct an InvokeInst given a range of arguments.
  ///
  /// Construct an InvokeInst from a range of arguments
  inline InvokeInst(Value *Func, BasicBlock *IfNormal, BasicBlock *IfException,
                    ArrayRef<Value *> Args, ArrayRef<OperandBundleDef> Bundles,
                    unsigned Values, const Twine &NameStr,
                    BasicBlock *InsertAtEnd);

  bool hasDescriptor() const { return HasDescriptor; }

  void init(Value *Func, BasicBlock *IfNormal, BasicBlock *IfException,
            ArrayRef<Value *> Args, ArrayRef<OperandBundleDef> Bundles,
            const Twine &NameStr) {
    init(cast<FunctionType>(
             cast<PointerType>(Func->getType())->getElementType()),
         Func, IfNormal, IfException, Args, Bundles, NameStr);
  }

  void init(FunctionType *FTy, Value *Func, BasicBlock *IfNormal,
            BasicBlock *IfException, ArrayRef<Value *> Args,
            ArrayRef<OperandBundleDef> Bundles, const Twine &NameStr);

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  InvokeInst *cloneImpl() const;

public:
  static InvokeInst *Create(Value *Func, BasicBlock *IfNormal,
                            BasicBlock *IfException, ArrayRef<Value *> Args,
                            const Twine &NameStr,
                            Instruction *InsertBefore = nullptr) {
    return Create(cast<FunctionType>(
                      cast<PointerType>(Func->getType())->getElementType()),
                  Func, IfNormal, IfException, Args, None, NameStr,
                  InsertBefore);
  }

  static InvokeInst *Create(Value *Func, BasicBlock *IfNormal,
                            BasicBlock *IfException, ArrayRef<Value *> Args,
                            ArrayRef<OperandBundleDef> Bundles = None,
                            const Twine &NameStr = "",
                            Instruction *InsertBefore = nullptr) {
    return Create(cast<FunctionType>(
                      cast<PointerType>(Func->getType())->getElementType()),
                  Func, IfNormal, IfException, Args, Bundles, NameStr,
                  InsertBefore);
  }

  static InvokeInst *Create(FunctionType *Ty, Value *Func, BasicBlock *IfNormal,
                            BasicBlock *IfException, ArrayRef<Value *> Args,
                            const Twine &NameStr,
                            Instruction *InsertBefore = nullptr) {
    unsigned Values = unsigned(Args.size()) + 3;
    return new (Values) InvokeInst(Ty, Func, IfNormal, IfException, Args, None,
                                   Values, NameStr, InsertBefore);
  }

  static InvokeInst *Create(FunctionType *Ty, Value *Func, BasicBlock *IfNormal,
                            BasicBlock *IfException, ArrayRef<Value *> Args,
                            ArrayRef<OperandBundleDef> Bundles = None,
                            const Twine &NameStr = "",
                            Instruction *InsertBefore = nullptr) {
    unsigned Values = unsigned(Args.size()) + CountBundleInputs(Bundles) + 3;
    unsigned DescriptorBytes = Bundles.size() * sizeof(BundleOpInfo);

    return new (Values, DescriptorBytes)
        InvokeInst(Ty, Func, IfNormal, IfException, Args, Bundles, Values,
                   NameStr, InsertBefore);
  }

  static InvokeInst *Create(Value *Func,
                            BasicBlock *IfNormal, BasicBlock *IfException,
                            ArrayRef<Value *> Args, const Twine &NameStr,
                            BasicBlock *InsertAtEnd) {
    unsigned Values = unsigned(Args.size()) + 3;
    return new (Values) InvokeInst(Func, IfNormal, IfException, Args, None,
                                   Values, NameStr, InsertAtEnd);
  }
  static InvokeInst *Create(Value *Func, BasicBlock *IfNormal,
                            BasicBlock *IfException, ArrayRef<Value *> Args,
                            ArrayRef<OperandBundleDef> Bundles,
                            const Twine &NameStr, BasicBlock *InsertAtEnd) {
    unsigned Values = unsigned(Args.size()) + CountBundleInputs(Bundles) + 3;
    unsigned DescriptorBytes = Bundles.size() * sizeof(BundleOpInfo);

    return new (Values, DescriptorBytes)
        InvokeInst(Func, IfNormal, IfException, Args, Bundles, Values, NameStr,
                   InsertAtEnd);
  }

  /// Create a clone of \p II with a different set of operand bundles and
  /// insert it before \p InsertPt.
  ///
  /// The returned invoke instruction is identical to \p II in every way except
  /// that the operand bundles for the new instruction are set to the operand
  /// bundles in \p Bundles.
  static InvokeInst *Create(InvokeInst *II, ArrayRef<OperandBundleDef> Bundles,
                            Instruction *InsertPt = nullptr);

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  FunctionType *getFunctionType() const { return FTy; }

  void mutateFunctionType(FunctionType *FTy) {
    mutateType(FTy->getReturnType());
    this->FTy = FTy;
  }

  /// Return the number of invoke arguments.
  ///
  unsigned getNumArgOperands() const {
    return getNumOperands() - getNumTotalBundleOperands() - 3;
  }

  /// getArgOperand/setArgOperand - Return/set the i-th invoke argument.
  ///
  Value *getArgOperand(unsigned i) const {
    assert(i < getNumArgOperands() && "Out of bounds!");
    return getOperand(i);
  }
  void setArgOperand(unsigned i, Value *v) {
    assert(i < getNumArgOperands() && "Out of bounds!");
    setOperand(i, v);
  }

  /// Return the iterator pointing to the beginning of the argument list.
  op_iterator arg_begin() { return op_begin(); }

  /// Return the iterator pointing to the end of the argument list.
  op_iterator arg_end() {
    // [ invoke args ], [ operand bundles ], normal dest, unwind dest, callee
    return op_end() - getNumTotalBundleOperands() - 3;
  }

  /// Iteration adapter for range-for loops.
  iterator_range<op_iterator> arg_operands() {
    return make_range(arg_begin(), arg_end());
  }

  /// Return the iterator pointing to the beginning of the argument list.
  const_op_iterator arg_begin() const { return op_begin(); }

  /// Return the iterator pointing to the end of the argument list.
  const_op_iterator arg_end() const {
    // [ invoke args ], [ operand bundles ], normal dest, unwind dest, callee
    return op_end() - getNumTotalBundleOperands() - 3;
  }

  /// Iteration adapter for range-for loops.
  iterator_range<const_op_iterator> arg_operands() const {
    return make_range(arg_begin(), arg_end());
  }

  /// Wrappers for getting the \c Use of a invoke argument.
  const Use &getArgOperandUse(unsigned i) const {
    assert(i < getNumArgOperands() && "Out of bounds!");
    return getOperandUse(i);
  }
  Use &getArgOperandUse(unsigned i) {
    assert(i < getNumArgOperands() && "Out of bounds!");
    return getOperandUse(i);
  }

  /// If one of the arguments has the 'returned' attribute, return its
  /// operand value. Otherwise, return nullptr.
  Value *getReturnedArgOperand() const;

  /// getCallingConv/setCallingConv - Get or set the calling convention of this
  /// function call.
  CallingConv::ID getCallingConv() const {
    return static_cast<CallingConv::ID>(getSubclassDataFromInstruction());
  }
  void setCallingConv(CallingConv::ID CC) {
    auto ID = static_cast<unsigned>(CC);
    assert(!(ID & ~CallingConv::MaxID) && "Unsupported calling convention");
    setInstructionSubclassData(ID);
  }

  /// Return the parameter attributes for this invoke.
  ///
  AttributeList getAttributes() const { return Attrs; }

  /// Set the parameter attributes for this invoke.
  ///
  void setAttributes(AttributeList A) { Attrs = A; }

  /// adds the attribute to the list of attributes.
  void addAttribute(unsigned i, Attribute::AttrKind Kind);

  /// adds the attribute to the list of attributes.
  void addAttribute(unsigned i, Attribute Attr);

  /// removes the attribute from the list of attributes.
  void removeAttribute(unsigned i, Attribute::AttrKind Kind);

  /// removes the attribute from the list of attributes.
  void removeAttribute(unsigned i, StringRef Kind);

  /// adds the dereferenceable attribute to the list of attributes.
  void addDereferenceableAttr(unsigned i, uint64_t Bytes);

  /// adds the dereferenceable_or_null attribute to the list of
  /// attributes.
  void addDereferenceableOrNullAttr(unsigned i, uint64_t Bytes);

  /// Determine whether this call has the given attribute.
  bool hasFnAttr(Attribute::AttrKind Kind) const {
    assert(Kind != Attribute::NoBuiltin &&
           "Use CallInst::isNoBuiltin() to check for Attribute::NoBuiltin");
    return hasFnAttrImpl(Kind);
  }

  /// Determine whether this call has the given attribute.
  bool hasFnAttr(StringRef Kind) const {
    return hasFnAttrImpl(Kind);
  }

  /// Determine whether the call or the callee has the given attributes.
  bool paramHasAttr(unsigned i, Attribute::AttrKind Kind) const;

  /// Get the attribute of a given kind at a position.
  Attribute getAttribute(unsigned i, Attribute::AttrKind Kind) const {
    return getAttributes().getAttribute(i, Kind);
  }

  /// Get the attribute of a given kind at a position.
  Attribute getAttribute(unsigned i, StringRef Kind) const {
    return getAttributes().getAttribute(i, Kind);
  }

  /// Return true if the data operand at index \p i has the attribute \p
  /// A.
  ///
  /// Data operands include invoke arguments and values used in operand bundles,
  /// but does not include the invokee operand, or the two successor blocks.
  /// This routine dispatches to the underlying AttributeList or the
  /// OperandBundleUser as appropriate.
  ///
  /// The index \p i is interpreted as
  ///
  ///  \p i == Attribute::ReturnIndex  -> the return value
  ///  \p i in [1, arg_size + 1)  -> argument number (\p i - 1)
  ///  \p i in [arg_size + 1, data_operand_size + 1) -> bundle operand at index
  ///     (\p i - 1) in the operand list.
  bool dataOperandHasImpliedAttr(unsigned i, Attribute::AttrKind Kind) const;

  /// Extract the alignment for a call or parameter (0=unknown).
  unsigned getParamAlignment(unsigned i) const {
    return Attrs.getParamAlignment(i);
  }

  /// Extract the number of dereferenceable bytes for a call or
  /// parameter (0=unknown).
  uint64_t getDereferenceableBytes(unsigned i) const {
    return Attrs.getDereferenceableBytes(i);
  }

  /// Extract the number of dereferenceable_or_null bytes for a call or
  /// parameter (0=unknown).
  uint64_t getDereferenceableOrNullBytes(unsigned i) const {
    return Attrs.getDereferenceableOrNullBytes(i);
  }

  /// @brief Determine if the parameter or return value is marked with NoAlias
  /// attribute.
  /// @param n The parameter to check. 1 is the first parameter, 0 is the return
  bool doesNotAlias(unsigned n) const {
    return Attrs.hasAttribute(n, Attribute::NoAlias);
  }

  /// Return true if the call should not be treated as a call to a
  /// builtin.
  bool isNoBuiltin() const {
    // We assert in hasFnAttr if one passes in Attribute::NoBuiltin, so we have
    // to check it by hand.
    return hasFnAttrImpl(Attribute::NoBuiltin) &&
      !hasFnAttrImpl(Attribute::Builtin);
  }

  /// Return true if the call should not be inlined.
  bool isNoInline() const { return hasFnAttr(Attribute::NoInline); }
  void setIsNoInline() {
    addAttribute(AttributeList::FunctionIndex, Attribute::NoInline);
  }

  /// Determine if the call does not access memory.
  bool doesNotAccessMemory() const {
    return hasFnAttr(Attribute::ReadNone);
  }
  void setDoesNotAccessMemory() {
    addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
  }

  /// Determine if the call does not access or only reads memory.
  bool onlyReadsMemory() const {
    return doesNotAccessMemory() || hasFnAttr(Attribute::ReadOnly);
  }
  void setOnlyReadsMemory() {
    addAttribute(AttributeList::FunctionIndex, Attribute::ReadOnly);
  }

  /// Determine if the call does not access or only writes memory.
  bool doesNotReadMemory() const {
    return doesNotAccessMemory() || hasFnAttr(Attribute::WriteOnly);
  }
  void setDoesNotReadMemory() {
    addAttribute(AttributeList::FunctionIndex, Attribute::WriteOnly);
  }

  /// @brief Determine if the call access memmory only using it's pointer
  /// arguments.
  bool onlyAccessesArgMemory() const {
    return hasFnAttr(Attribute::ArgMemOnly);
  }
  void setOnlyAccessesArgMemory() {
    addAttribute(AttributeList::FunctionIndex, Attribute::ArgMemOnly);
  }

  /// Determine if the call cannot return.
  bool doesNotReturn() const { return hasFnAttr(Attribute::NoReturn); }
  void setDoesNotReturn() {
    addAttribute(AttributeList::FunctionIndex, Attribute::NoReturn);
  }

  /// Determine if the call cannot unwind.
  bool doesNotThrow() const { return hasFnAttr(Attribute::NoUnwind); }
  void setDoesNotThrow() {
    addAttribute(AttributeList::FunctionIndex, Attribute::NoUnwind);
  }

  /// Determine if the invoke cannot be duplicated.
  bool cannotDuplicate() const {return hasFnAttr(Attribute::NoDuplicate); }
  void setCannotDuplicate() {
    addAttribute(AttributeList::FunctionIndex, Attribute::NoDuplicate);
  }

  /// Determine if the invoke is convergent
  bool isConvergent() const { return hasFnAttr(Attribute::Convergent); }
  void setConvergent() {
    addAttribute(AttributeList::FunctionIndex, Attribute::Convergent);
  }
  void setNotConvergent() {
    removeAttribute(AttributeList::FunctionIndex, Attribute::Convergent);
  }

  /// Determine if the call returns a structure through first
  /// pointer argument.
  bool hasStructRetAttr() const {
    if (getNumArgOperands() == 0)
      return false;

    // Be friendly and also check the callee.
    return paramHasAttr(1, Attribute::StructRet);
  }

  /// Determine if any call argument is an aggregate passed by value.
  bool hasByValArgument() const {
    return Attrs.hasAttrSomewhere(Attribute::ByVal);
  }

  /// Return the function called, or null if this is an
  /// indirect function invocation.
  ///
  Function *getCalledFunction() const {
    return dyn_cast<Function>(Op<-3>());
  }

  /// Get a pointer to the function that is invoked by this
  /// instruction
  const Value *getCalledValue() const { return Op<-3>(); }
        Value *getCalledValue()       { return Op<-3>(); }

  /// Set the function called.
  void setCalledFunction(Value* Fn) {
    setCalledFunction(
        cast<FunctionType>(cast<PointerType>(Fn->getType())->getElementType()),
        Fn);
  }
  void setCalledFunction(FunctionType *FTy, Value *Fn) {
    this->FTy = FTy;
    assert(FTy == cast<FunctionType>(
                      cast<PointerType>(Fn->getType())->getElementType()));
    Op<-3>() = Fn;
  }

  // get*Dest - Return the destination basic blocks...
  BasicBlock *getNormalDest() const {
    return cast<BasicBlock>(Op<-2>());
  }
  BasicBlock *getUnwindDest() const {
    return cast<BasicBlock>(Op<-1>());
  }
  void setNormalDest(BasicBlock *B) {
    Op<-2>() = reinterpret_cast<Value*>(B);
  }
  void setUnwindDest(BasicBlock *B) {
    Op<-1>() = reinterpret_cast<Value*>(B);
  }

  /// Get the landingpad instruction from the landing pad
  /// block (the unwind destination).
  LandingPadInst *getLandingPadInst() const;

  BasicBlock *getSuccessor(unsigned i) const {
    assert(i < 2 && "Successor # out of range for invoke!");
    return i == 0 ? getNormalDest() : getUnwindDest();
  }

  void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(idx < 2 && "Successor # out of range for invoke!");
    *(&Op<-2>() + idx) = reinterpret_cast<Value*>(NewSucc);
  }

  unsigned getNumSuccessors() const { return 2; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Invoke);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  BasicBlock *getSuccessorV(unsigned idx) const override;
  unsigned getNumSuccessorsV() const override;
  void setSuccessorV(unsigned idx, BasicBlock *B) override;

  template <typename AttrKind> bool hasFnAttrImpl(AttrKind Kind) const {
    if (Attrs.hasAttribute(AttributeList::FunctionIndex, Kind))
      return true;

    // Operand bundles override attributes on the called function, but don't
    // override attributes directly present on the invoke instruction.
    if (isFnAttrDisallowedByOpBundle(Kind))
      return false;

    if (const Function *F = getCalledFunction())
      return F->getAttributes().hasAttribute(AttributeList::FunctionIndex,
                                             Kind);
    return false;
  }

  // Shadow Instruction::setInstructionSubclassData with a private forwarding
  // method so that subclasses cannot accidentally use it.
  void setInstructionSubclassData(unsigned short D) {
    Instruction::setInstructionSubclassData(D);
  }
};

template <>
struct OperandTraits<InvokeInst> : public VariadicOperandTraits<InvokeInst, 3> {
};

InvokeInst::InvokeInst(FunctionType *Ty, Value *Func, BasicBlock *IfNormal,
                       BasicBlock *IfException, ArrayRef<Value *> Args,
                       ArrayRef<OperandBundleDef> Bundles, unsigned Values,
                       const Twine &NameStr, Instruction *InsertBefore)
    : TerminatorInst(Ty->getReturnType(), Instruction::Invoke,
                     OperandTraits<InvokeInst>::op_end(this) - Values, Values,
                     InsertBefore) {
  init(Ty, Func, IfNormal, IfException, Args, Bundles, NameStr);
}

InvokeInst::InvokeInst(Value *Func, BasicBlock *IfNormal,
                       BasicBlock *IfException, ArrayRef<Value *> Args,
                       ArrayRef<OperandBundleDef> Bundles, unsigned Values,
                       const Twine &NameStr, BasicBlock *InsertAtEnd)
    : TerminatorInst(
          cast<FunctionType>(cast<PointerType>(Func->getType())
                                 ->getElementType())->getReturnType(),
          Instruction::Invoke, OperandTraits<InvokeInst>::op_end(this) - Values,
          Values, InsertAtEnd) {
  init(Func, IfNormal, IfException, Args, Bundles, NameStr);
}

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(InvokeInst, Value)

//===----------------------------------------------------------------------===//
//                              ResumeInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// Resume the propagation of an exception.
///
class ResumeInst : public TerminatorInst {
  ResumeInst(const ResumeInst &RI);

  explicit ResumeInst(Value *Exn, Instruction *InsertBefore=nullptr);
  ResumeInst(Value *Exn, BasicBlock *InsertAtEnd);

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  ResumeInst *cloneImpl() const;

public:
  static ResumeInst *Create(Value *Exn, Instruction *InsertBefore = nullptr) {
    return new(1) ResumeInst(Exn, InsertBefore);
  }

  static ResumeInst *Create(Value *Exn, BasicBlock *InsertAtEnd) {
    return new(1) ResumeInst(Exn, InsertAtEnd);
  }

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// Convenience accessor.
  Value *getValue() const { return Op<0>(); }

  unsigned getNumSuccessors() const { return 0; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Resume;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  BasicBlock *getSuccessorV(unsigned idx) const override;
  unsigned getNumSuccessorsV() const override;
  void setSuccessorV(unsigned idx, BasicBlock *B) override;
};

template <>
struct OperandTraits<ResumeInst> :
    public FixedNumOperandTraits<ResumeInst, 1> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ResumeInst, Value)

//===----------------------------------------------------------------------===//
//                         CatchSwitchInst Class
//===----------------------------------------------------------------------===//
class CatchSwitchInst : public TerminatorInst {
  /// The number of operands actually allocated.  NumOperands is
  /// the number actually in use.
  unsigned ReservedSpace;

  // Operand[0] = Outer scope
  // Operand[1] = Unwind block destination
  // Operand[n] = BasicBlock to go to on match
  CatchSwitchInst(const CatchSwitchInst &CSI);

  /// Create a new switch instruction, specifying a
  /// default destination.  The number of additional handlers can be specified
  /// here to make memory allocation more efficient.
  /// This constructor can also autoinsert before another instruction.
  CatchSwitchInst(Value *ParentPad, BasicBlock *UnwindDest,
                  unsigned NumHandlers, const Twine &NameStr,
                  Instruction *InsertBefore);

  /// Create a new switch instruction, specifying a
  /// default destination.  The number of additional handlers can be specified
  /// here to make memory allocation more efficient.
  /// This constructor also autoinserts at the end of the specified BasicBlock.
  CatchSwitchInst(Value *ParentPad, BasicBlock *UnwindDest,
                  unsigned NumHandlers, const Twine &NameStr,
                  BasicBlock *InsertAtEnd);

  // allocate space for exactly zero operands
  void *operator new(size_t s) { return User::operator new(s); }

  void init(Value *ParentPad, BasicBlock *UnwindDest, unsigned NumReserved);
  void growOperands(unsigned Size);

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  CatchSwitchInst *cloneImpl() const;

public:
  void *operator new(size_t, unsigned) = delete;

  static CatchSwitchInst *Create(Value *ParentPad, BasicBlock *UnwindDest,
                                 unsigned NumHandlers,
                                 const Twine &NameStr = "",
                                 Instruction *InsertBefore = nullptr) {
    return new CatchSwitchInst(ParentPad, UnwindDest, NumHandlers, NameStr,
                               InsertBefore);
  }

  static CatchSwitchInst *Create(Value *ParentPad, BasicBlock *UnwindDest,
                                 unsigned NumHandlers, const Twine &NameStr,
                                 BasicBlock *InsertAtEnd) {
    return new CatchSwitchInst(ParentPad, UnwindDest, NumHandlers, NameStr,
                               InsertAtEnd);
  }

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  // Accessor Methods for CatchSwitch stmt
  Value *getParentPad() const { return getOperand(0); }
  void setParentPad(Value *ParentPad) { setOperand(0, ParentPad); }

  // Accessor Methods for CatchSwitch stmt
  bool hasUnwindDest() const { return getSubclassDataFromInstruction() & 1; }
  bool unwindsToCaller() const { return !hasUnwindDest(); }
  BasicBlock *getUnwindDest() const {
    if (hasUnwindDest())
      return cast<BasicBlock>(getOperand(1));
    return nullptr;
  }
  void setUnwindDest(BasicBlock *UnwindDest) {
    assert(UnwindDest);
    assert(hasUnwindDest());
    setOperand(1, UnwindDest);
  }

  /// return the number of 'handlers' in this catchswitch
  /// instruction, except the default handler
  unsigned getNumHandlers() const {
    if (hasUnwindDest())
      return getNumOperands() - 2;
    return getNumOperands() - 1;
  }

private:
  static BasicBlock *handler_helper(Value *V) { return cast<BasicBlock>(V); }
  static const BasicBlock *handler_helper(const Value *V) {
    return cast<BasicBlock>(V);
  }

public:
  typedef std::pointer_to_unary_function<Value *, BasicBlock *> DerefFnTy;
  typedef mapped_iterator<op_iterator, DerefFnTy> handler_iterator;
  typedef iterator_range<handler_iterator> handler_range;
  typedef std::pointer_to_unary_function<const Value *, const BasicBlock *>
      ConstDerefFnTy;
  typedef mapped_iterator<const_op_iterator, ConstDerefFnTy> const_handler_iterator;
  typedef iterator_range<const_handler_iterator> const_handler_range;

  /// Returns an iterator that points to the first handler in CatchSwitchInst.
  handler_iterator handler_begin() {
    op_iterator It = op_begin() + 1;
    if (hasUnwindDest())
      ++It;
    return handler_iterator(It, DerefFnTy(handler_helper));
  }

  /// Returns an iterator that points to the first handler in the
  /// CatchSwitchInst.
  const_handler_iterator handler_begin() const {
    const_op_iterator It = op_begin() + 1;
    if (hasUnwindDest())
      ++It;
    return const_handler_iterator(It, ConstDerefFnTy(handler_helper));
  }

  /// Returns a read-only iterator that points one past the last
  /// handler in the CatchSwitchInst.
  handler_iterator handler_end() {
    return handler_iterator(op_end(), DerefFnTy(handler_helper));
  }

  /// Returns an iterator that points one past the last handler in the
  /// CatchSwitchInst.
  const_handler_iterator handler_end() const {
    return const_handler_iterator(op_end(), ConstDerefFnTy(handler_helper));
  }

  /// iteration adapter for range-for loops.
  handler_range handlers() {
    return make_range(handler_begin(), handler_end());
  }

  /// iteration adapter for range-for loops.
  const_handler_range handlers() const {
    return make_range(handler_begin(), handler_end());
  }

  /// Add an entry to the switch instruction...
  /// Note:
  /// This action invalidates handler_end(). Old handler_end() iterator will
  /// point to the added handler.
  void addHandler(BasicBlock *Dest);

  void removeHandler(handler_iterator HI);

  unsigned getNumSuccessors() const { return getNumOperands() - 1; }
  BasicBlock *getSuccessor(unsigned Idx) const {
    assert(Idx < getNumSuccessors() &&
           "Successor # out of range for catchswitch!");
    return cast<BasicBlock>(getOperand(Idx + 1));
  }
  void setSuccessor(unsigned Idx, BasicBlock *NewSucc) {
    assert(Idx < getNumSuccessors() &&
           "Successor # out of range for catchswitch!");
    setOperand(Idx + 1, NewSucc);
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::CatchSwitch;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  BasicBlock *getSuccessorV(unsigned Idx) const override;
  unsigned getNumSuccessorsV() const override;
  void setSuccessorV(unsigned Idx, BasicBlock *B) override;
};

template <>
struct OperandTraits<CatchSwitchInst> : public HungoffOperandTraits<2> {};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(CatchSwitchInst, Value)

//===----------------------------------------------------------------------===//
//                               CleanupPadInst Class
//===----------------------------------------------------------------------===//
class CleanupPadInst : public FuncletPadInst {
private:
  explicit CleanupPadInst(Value *ParentPad, ArrayRef<Value *> Args,
                          unsigned Values, const Twine &NameStr,
                          Instruction *InsertBefore)
      : FuncletPadInst(Instruction::CleanupPad, ParentPad, Args, Values,
                       NameStr, InsertBefore) {}
  explicit CleanupPadInst(Value *ParentPad, ArrayRef<Value *> Args,
                          unsigned Values, const Twine &NameStr,
                          BasicBlock *InsertAtEnd)
      : FuncletPadInst(Instruction::CleanupPad, ParentPad, Args, Values,
                       NameStr, InsertAtEnd) {}

public:
  static CleanupPadInst *Create(Value *ParentPad, ArrayRef<Value *> Args = None,
                                const Twine &NameStr = "",
                                Instruction *InsertBefore = nullptr) {
    unsigned Values = 1 + Args.size();
    return new (Values)
        CleanupPadInst(ParentPad, Args, Values, NameStr, InsertBefore);
  }

  static CleanupPadInst *Create(Value *ParentPad, ArrayRef<Value *> Args,
                                const Twine &NameStr, BasicBlock *InsertAtEnd) {
    unsigned Values = 1 + Args.size();
    return new (Values)
        CleanupPadInst(ParentPad, Args, Values, NameStr, InsertAtEnd);
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::CleanupPad;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                               CatchPadInst Class
//===----------------------------------------------------------------------===//
class CatchPadInst : public FuncletPadInst {
private:
  explicit CatchPadInst(Value *CatchSwitch, ArrayRef<Value *> Args,
                        unsigned Values, const Twine &NameStr,
                        Instruction *InsertBefore)
      : FuncletPadInst(Instruction::CatchPad, CatchSwitch, Args, Values,
                       NameStr, InsertBefore) {}
  explicit CatchPadInst(Value *CatchSwitch, ArrayRef<Value *> Args,
                        unsigned Values, const Twine &NameStr,
                        BasicBlock *InsertAtEnd)
      : FuncletPadInst(Instruction::CatchPad, CatchSwitch, Args, Values,
                       NameStr, InsertAtEnd) {}

public:
  static CatchPadInst *Create(Value *CatchSwitch, ArrayRef<Value *> Args,
                              const Twine &NameStr = "",
                              Instruction *InsertBefore = nullptr) {
    unsigned Values = 1 + Args.size();
    return new (Values)
        CatchPadInst(CatchSwitch, Args, Values, NameStr, InsertBefore);
  }

  static CatchPadInst *Create(Value *CatchSwitch, ArrayRef<Value *> Args,
                              const Twine &NameStr, BasicBlock *InsertAtEnd) {
    unsigned Values = 1 + Args.size();
    return new (Values)
        CatchPadInst(CatchSwitch, Args, Values, NameStr, InsertAtEnd);
  }

  /// Convenience accessors
  CatchSwitchInst *getCatchSwitch() const {
    return cast<CatchSwitchInst>(Op<-1>());
  }
  void setCatchSwitch(Value *CatchSwitch) {
    assert(CatchSwitch);
    Op<-1>() = CatchSwitch;
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::CatchPad;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                               CatchReturnInst Class
//===----------------------------------------------------------------------===//

class CatchReturnInst : public TerminatorInst {
  CatchReturnInst(const CatchReturnInst &RI);
  CatchReturnInst(Value *CatchPad, BasicBlock *BB, Instruction *InsertBefore);
  CatchReturnInst(Value *CatchPad, BasicBlock *BB, BasicBlock *InsertAtEnd);

  void init(Value *CatchPad, BasicBlock *BB);

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  CatchReturnInst *cloneImpl() const;

public:
  static CatchReturnInst *Create(Value *CatchPad, BasicBlock *BB,
                                 Instruction *InsertBefore = nullptr) {
    assert(CatchPad);
    assert(BB);
    return new (2) CatchReturnInst(CatchPad, BB, InsertBefore);
  }

  static CatchReturnInst *Create(Value *CatchPad, BasicBlock *BB,
                                 BasicBlock *InsertAtEnd) {
    assert(CatchPad);
    assert(BB);
    return new (2) CatchReturnInst(CatchPad, BB, InsertAtEnd);
  }

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// Convenience accessors.
  CatchPadInst *getCatchPad() const { return cast<CatchPadInst>(Op<0>()); }
  void setCatchPad(CatchPadInst *CatchPad) {
    assert(CatchPad);
    Op<0>() = CatchPad;
  }

  BasicBlock *getSuccessor() const { return cast<BasicBlock>(Op<1>()); }
  void setSuccessor(BasicBlock *NewSucc) {
    assert(NewSucc);
    Op<1>() = NewSucc;
  }
  unsigned getNumSuccessors() const { return 1; }

  /// Get the parentPad of this catchret's catchpad's catchswitch.
  /// The successor block is implicitly a member of this funclet.
  Value *getCatchSwitchParentPad() const {
    return getCatchPad()->getCatchSwitch()->getParentPad();
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::CatchRet);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  BasicBlock *getSuccessorV(unsigned Idx) const override;
  unsigned getNumSuccessorsV() const override;
  void setSuccessorV(unsigned Idx, BasicBlock *B) override;
};

template <>
struct OperandTraits<CatchReturnInst>
    : public FixedNumOperandTraits<CatchReturnInst, 2> {};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(CatchReturnInst, Value)

//===----------------------------------------------------------------------===//
//                               CleanupReturnInst Class
//===----------------------------------------------------------------------===//

class CleanupReturnInst : public TerminatorInst {
private:
  CleanupReturnInst(const CleanupReturnInst &RI);
  CleanupReturnInst(Value *CleanupPad, BasicBlock *UnwindBB, unsigned Values,
                    Instruction *InsertBefore = nullptr);
  CleanupReturnInst(Value *CleanupPad, BasicBlock *UnwindBB, unsigned Values,
                    BasicBlock *InsertAtEnd);

  void init(Value *CleanupPad, BasicBlock *UnwindBB);

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  CleanupReturnInst *cloneImpl() const;

public:
  static CleanupReturnInst *Create(Value *CleanupPad,
                                   BasicBlock *UnwindBB = nullptr,
                                   Instruction *InsertBefore = nullptr) {
    assert(CleanupPad);
    unsigned Values = 1;
    if (UnwindBB)
      ++Values;
    return new (Values)
        CleanupReturnInst(CleanupPad, UnwindBB, Values, InsertBefore);
  }

  static CleanupReturnInst *Create(Value *CleanupPad, BasicBlock *UnwindBB,
                                   BasicBlock *InsertAtEnd) {
    assert(CleanupPad);
    unsigned Values = 1;
    if (UnwindBB)
      ++Values;
    return new (Values)
        CleanupReturnInst(CleanupPad, UnwindBB, Values, InsertAtEnd);
  }

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  bool hasUnwindDest() const { return getSubclassDataFromInstruction() & 1; }
  bool unwindsToCaller() const { return !hasUnwindDest(); }

  /// Convenience accessor.
  CleanupPadInst *getCleanupPad() const {
    return cast<CleanupPadInst>(Op<0>());
  }
  void setCleanupPad(CleanupPadInst *CleanupPad) {
    assert(CleanupPad);
    Op<0>() = CleanupPad;
  }

  unsigned getNumSuccessors() const { return hasUnwindDest() ? 1 : 0; }

  BasicBlock *getUnwindDest() const {
    return hasUnwindDest() ? cast<BasicBlock>(Op<1>()) : nullptr;
  }
  void setUnwindDest(BasicBlock *NewDest) {
    assert(NewDest);
    assert(hasUnwindDest());
    Op<1>() = NewDest;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::CleanupRet);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  BasicBlock *getSuccessorV(unsigned Idx) const override;
  unsigned getNumSuccessorsV() const override;
  void setSuccessorV(unsigned Idx, BasicBlock *B) override;

  // Shadow Instruction::setInstructionSubclassData with a private forwarding
  // method so that subclasses cannot accidentally use it.
  void setInstructionSubclassData(unsigned short D) {
    Instruction::setInstructionSubclassData(D);
  }
};

template <>
struct OperandTraits<CleanupReturnInst>
    : public VariadicOperandTraits<CleanupReturnInst, /*MINARITY=*/1> {};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(CleanupReturnInst, Value)

//===----------------------------------------------------------------------===//
//                           UnreachableInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// This function has undefined behavior.  In particular, the
/// presence of this instruction indicates some higher level knowledge that the
/// end of the block cannot be reached.
///
class UnreachableInst : public TerminatorInst {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  UnreachableInst *cloneImpl() const;

public:
  explicit UnreachableInst(LLVMContext &C, Instruction *InsertBefore = nullptr);
  explicit UnreachableInst(LLVMContext &C, BasicBlock *InsertAtEnd);

  // allocate space for exactly zero operands
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }

  void *operator new(size_t, unsigned) = delete;

  unsigned getNumSuccessors() const { return 0; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Unreachable;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

private:
  BasicBlock *getSuccessorV(unsigned idx) const override;
  unsigned getNumSuccessorsV() const override;
  void setSuccessorV(unsigned idx, BasicBlock *B) override;
};

//===----------------------------------------------------------------------===//
//                                 TruncInst Class
//===----------------------------------------------------------------------===//

/// This class represents a truncation of integer types.
class TruncInst : public CastInst {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  /// Clone an identical TruncInst
  TruncInst *cloneImpl() const;

public:
  /// Constructor with insert-before-instruction semantics
  TruncInst(
    Value *S,                           ///< The value to be truncated
    Type *Ty,                           ///< The (smaller) type to truncate to
    const Twine &NameStr = "",          ///< A name for the new instruction
    Instruction *InsertBefore = nullptr ///< Where to insert the new instruction
  );

  /// Constructor with insert-at-end-of-block semantics
  TruncInst(
    Value *S,                     ///< The value to be truncated
    Type *Ty,                     ///< The (smaller) type to truncate to
    const Twine &NameStr,         ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Trunc;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 ZExtInst Class
//===----------------------------------------------------------------------===//

/// This class represents zero extension of integer types.
class ZExtInst : public CastInst {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  /// Clone an identical ZExtInst
  ZExtInst *cloneImpl() const;

public:
  /// Constructor with insert-before-instruction semantics
  ZExtInst(
    Value *S,                           ///< The value to be zero extended
    Type *Ty,                           ///< The type to zero extend to
    const Twine &NameStr = "",          ///< A name for the new instruction
    Instruction *InsertBefore = nullptr ///< Where to insert the new instruction
  );

  /// Constructor with insert-at-end semantics.
  ZExtInst(
    Value *S,                     ///< The value to be zero extended
    Type *Ty,                     ///< The type to zero extend to
    const Twine &NameStr,         ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == ZExt;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 SExtInst Class
//===----------------------------------------------------------------------===//

/// This class represents a sign extension of integer types.
class SExtInst : public CastInst {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  /// Clone an identical SExtInst
  SExtInst *cloneImpl() const;

public:
  /// Constructor with insert-before-instruction semantics
  SExtInst(
    Value *S,                           ///< The value to be sign extended
    Type *Ty,                           ///< The type to sign extend to
    const Twine &NameStr = "",          ///< A name for the new instruction
    Instruction *InsertBefore = nullptr ///< Where to insert the new instruction
  );

  /// Constructor with insert-at-end-of-block semantics
  SExtInst(
    Value *S,                     ///< The value to be sign extended
    Type *Ty,                     ///< The type to sign extend to
    const Twine &NameStr,         ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == SExt;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 FPTruncInst Class
//===----------------------------------------------------------------------===//

/// This class represents a truncation of floating point types.
class FPTruncInst : public CastInst {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  /// Clone an identical FPTruncInst
  FPTruncInst *cloneImpl() const;

public:
  /// Constructor with insert-before-instruction semantics
  FPTruncInst(
    Value *S,                           ///< The value to be truncated
    Type *Ty,                           ///< The type to truncate to
    const Twine &NameStr = "",          ///< A name for the new instruction
    Instruction *InsertBefore = nullptr ///< Where to insert the new instruction
  );

  /// Constructor with insert-before-instruction semantics
  FPTruncInst(
    Value *S,                     ///< The value to be truncated
    Type *Ty,                     ///< The type to truncate to
    const Twine &NameStr,         ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == FPTrunc;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 FPExtInst Class
//===----------------------------------------------------------------------===//

/// This class represents an extension of floating point types.
class FPExtInst : public CastInst {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  /// Clone an identical FPExtInst
  FPExtInst *cloneImpl() const;

public:
  /// Constructor with insert-before-instruction semantics
  FPExtInst(
    Value *S,                           ///< The value to be extended
    Type *Ty,                           ///< The type to extend to
    const Twine &NameStr = "",          ///< A name for the new instruction
    Instruction *InsertBefore = nullptr ///< Where to insert the new instruction
  );

  /// Constructor with insert-at-end-of-block semantics
  FPExtInst(
    Value *S,                     ///< The value to be extended
    Type *Ty,                     ///< The type to extend to
    const Twine &NameStr,         ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == FPExt;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 UIToFPInst Class
//===----------------------------------------------------------------------===//

/// This class represents a cast unsigned integer to floating point.
class UIToFPInst : public CastInst {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  /// Clone an identical UIToFPInst
  UIToFPInst *cloneImpl() const;

public:
  /// Constructor with insert-before-instruction semantics
  UIToFPInst(
    Value *S,                           ///< The value to be converted
    Type *Ty,                           ///< The type to convert to
    const Twine &NameStr = "",          ///< A name for the new instruction
    Instruction *InsertBefore = nullptr ///< Where to insert the new instruction
  );

  /// Constructor with insert-at-end-of-block semantics
  UIToFPInst(
    Value *S,                     ///< The value to be converted
    Type *Ty,                     ///< The type to convert to
    const Twine &NameStr,         ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == UIToFP;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 SIToFPInst Class
//===----------------------------------------------------------------------===//

/// This class represents a cast from signed integer to floating point.
class SIToFPInst : public CastInst {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  /// Clone an identical SIToFPInst
  SIToFPInst *cloneImpl() const;

public:
  /// Constructor with insert-before-instruction semantics
  SIToFPInst(
    Value *S,                           ///< The value to be converted
    Type *Ty,                           ///< The type to convert to
    const Twine &NameStr = "",          ///< A name for the new instruction
    Instruction *InsertBefore = nullptr ///< Where to insert the new instruction
  );

  /// Constructor with insert-at-end-of-block semantics
  SIToFPInst(
    Value *S,                     ///< The value to be converted
    Type *Ty,                     ///< The type to convert to
    const Twine &NameStr,         ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == SIToFP;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 FPToUIInst Class
//===----------------------------------------------------------------------===//

/// This class represents a cast from floating point to unsigned integer
class FPToUIInst  : public CastInst {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  /// Clone an identical FPToUIInst
  FPToUIInst *cloneImpl() const;

public:
  /// Constructor with insert-before-instruction semantics
  FPToUIInst(
    Value *S,                           ///< The value to be converted
    Type *Ty,                           ///< The type to convert to
    const Twine &NameStr = "",          ///< A name for the new instruction
    Instruction *InsertBefore = nullptr ///< Where to insert the new instruction
  );

  /// Constructor with insert-at-end-of-block semantics
  FPToUIInst(
    Value *S,                     ///< The value to be converted
    Type *Ty,                     ///< The type to convert to
    const Twine &NameStr,         ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< Where to insert the new instruction
  );

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == FPToUI;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 FPToSIInst Class
//===----------------------------------------------------------------------===//

/// This class represents a cast from floating point to signed integer.
class FPToSIInst  : public CastInst {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  /// Clone an identical FPToSIInst
  FPToSIInst *cloneImpl() const;

public:
  /// Constructor with insert-before-instruction semantics
  FPToSIInst(
    Value *S,                           ///< The value to be converted
    Type *Ty,                           ///< The type to convert to
    const Twine &NameStr = "",          ///< A name for the new instruction
    Instruction *InsertBefore = nullptr ///< Where to insert the new instruction
  );

  /// Constructor with insert-at-end-of-block semantics
  FPToSIInst(
    Value *S,                     ///< The value to be converted
    Type *Ty,                     ///< The type to convert to
    const Twine &NameStr,         ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == FPToSI;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 IntToPtrInst Class
//===----------------------------------------------------------------------===//

/// This class represents a cast from an integer to a pointer.
class IntToPtrInst : public CastInst {
public:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  /// Constructor with insert-before-instruction semantics
  IntToPtrInst(
    Value *S,                           ///< The value to be converted
    Type *Ty,                           ///< The type to convert to
    const Twine &NameStr = "",          ///< A name for the new instruction
    Instruction *InsertBefore = nullptr ///< Where to insert the new instruction
  );

  /// Constructor with insert-at-end-of-block semantics
  IntToPtrInst(
    Value *S,                     ///< The value to be converted
    Type *Ty,                     ///< The type to convert to
    const Twine &NameStr,         ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// Clone an identical IntToPtrInst.
  IntToPtrInst *cloneImpl() const;

  /// Returns the address space of this instruction's pointer type.
  unsigned getAddressSpace() const {
    return getType()->getPointerAddressSpace();
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == IntToPtr;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 PtrToIntInst Class
//===----------------------------------------------------------------------===//

/// This class represents a cast from a pointer to an integer.
class PtrToIntInst : public CastInst {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  /// Clone an identical PtrToIntInst.
  PtrToIntInst *cloneImpl() const;

public:
  /// Constructor with insert-before-instruction semantics
  PtrToIntInst(
    Value *S,                           ///< The value to be converted
    Type *Ty,                           ///< The type to convert to
    const Twine &NameStr = "",          ///< A name for the new instruction
    Instruction *InsertBefore = nullptr ///< Where to insert the new instruction
  );

  /// Constructor with insert-at-end-of-block semantics
  PtrToIntInst(
    Value *S,                     ///< The value to be converted
    Type *Ty,                     ///< The type to convert to
    const Twine &NameStr,         ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// Gets the pointer operand.
  Value *getPointerOperand() { return getOperand(0); }
  /// Gets the pointer operand.
  const Value *getPointerOperand() const { return getOperand(0); }
  /// Gets the operand index of the pointer operand.
  static unsigned getPointerOperandIndex() { return 0U; }

  /// Returns the address space of the pointer operand.
  unsigned getPointerAddressSpace() const {
    return getPointerOperand()->getType()->getPointerAddressSpace();
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == PtrToInt;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                             BitCastInst Class
//===----------------------------------------------------------------------===//

/// This class represents a no-op cast from one type to another.
class BitCastInst : public CastInst {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  /// Clone an identical BitCastInst.
  BitCastInst *cloneImpl() const;

public:
  /// Constructor with insert-before-instruction semantics
  BitCastInst(
    Value *S,                           ///< The value to be casted
    Type *Ty,                           ///< The type to casted to
    const Twine &NameStr = "",          ///< A name for the new instruction
    Instruction *InsertBefore = nullptr ///< Where to insert the new instruction
  );

  /// Constructor with insert-at-end-of-block semantics
  BitCastInst(
    Value *S,                     ///< The value to be casted
    Type *Ty,                     ///< The type to casted to
    const Twine &NameStr,         ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == BitCast;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                          AddrSpaceCastInst Class
//===----------------------------------------------------------------------===//

/// This class represents a conversion between pointers from one address space
/// to another.
class AddrSpaceCastInst : public CastInst {
protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  /// Clone an identical AddrSpaceCastInst.
  AddrSpaceCastInst *cloneImpl() const;

public:
  /// Constructor with insert-before-instruction semantics
  AddrSpaceCastInst(
    Value *S,                           ///< The value to be casted
    Type *Ty,                           ///< The type to casted to
    const Twine &NameStr = "",          ///< A name for the new instruction
    Instruction *InsertBefore = nullptr ///< Where to insert the new instruction
  );

  /// Constructor with insert-at-end-of-block semantics
  AddrSpaceCastInst(
    Value *S,                     ///< The value to be casted
    Type *Ty,                     ///< The type to casted to
    const Twine &NameStr,         ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == AddrSpaceCast;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

  /// Gets the pointer operand.
  Value *getPointerOperand() {
    return getOperand(0);
  }

  /// Gets the pointer operand.
  const Value *getPointerOperand() const {
    return getOperand(0);
  }

  /// Gets the operand index of the pointer operand.
  static unsigned getPointerOperandIndex() {
    return 0U;
  }

  /// Returns the address space of the pointer operand.
  unsigned getSrcAddressSpace() const {
    return getPointerOperand()->getType()->getPointerAddressSpace();
  }

  /// Returns the address space of the result.
  unsigned getDestAddressSpace() const {
    return getType()->getPointerAddressSpace();
  }
};

} // end namespace llvm

#endif // LLVM_IR_INSTRUCTIONS_H
