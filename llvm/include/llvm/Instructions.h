//===-- llvm/Instructions.h - Instruction subclass definitions --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes the class definitions of all of the subclasses of the
// Instruction class.  This is meant to be an easy way to get access to all
// instruction subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INSTRUCTIONS_H
#define LLVM_INSTRUCTIONS_H

#include <iterator>

#include "llvm/InstrTypes.h"
#include "llvm/DerivedTypes.h"

namespace llvm {

class BasicBlock;
class ConstantInt;
class PointerType;
class VectorType;
class ConstantRange;
class APInt;
class ParamAttrsList;

//===----------------------------------------------------------------------===//
//                             AllocationInst Class
//===----------------------------------------------------------------------===//

/// AllocationInst - This class is the common base class of MallocInst and
/// AllocaInst.
///
class AllocationInst : public UnaryInstruction {
  unsigned Alignment;
protected:
  AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy, unsigned Align,
                 const std::string &Name = "", Instruction *InsertBefore = 0);
  AllocationInst(const Type *Ty, Value *ArraySize, unsigned iTy, unsigned Align,
                 const std::string &Name, BasicBlock *InsertAtEnd);
public:
  // Out of line virtual method, so the vtable, etc has a home.
  virtual ~AllocationInst();

  /// isArrayAllocation - Return true if there is an allocation size parameter
  /// to the allocation instruction that is not 1.
  ///
  bool isArrayAllocation() const;

  /// getArraySize - Get the number of element allocated, for a simple
  /// allocation of a single element, this will return a constant 1 value.
  ///
  inline const Value *getArraySize() const { return getOperand(0); }
  inline Value *getArraySize() { return getOperand(0); }

  /// getType - Overload to return most specific pointer type
  ///
  inline const PointerType *getType() const {
    return reinterpret_cast<const PointerType*>(Instruction::getType());
  }

  /// getAllocatedType - Return the type that is being allocated by the
  /// instruction.
  ///
  const Type *getAllocatedType() const;

  /// getAlignment - Return the alignment of the memory that is being allocated
  /// by the instruction.
  ///
  unsigned getAlignment() const { return Alignment; }
  void setAlignment(unsigned Align) {
    assert((Align & (Align-1)) == 0 && "Alignment is not a power of 2!");
    Alignment = Align;
  }

  virtual Instruction *clone() const = 0;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const AllocationInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Alloca ||
           I->getOpcode() == Instruction::Malloc;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                MallocInst Class
//===----------------------------------------------------------------------===//

/// MallocInst - an instruction to allocated memory on the heap
///
class MallocInst : public AllocationInst {
  MallocInst(const MallocInst &MI);
public:
  explicit MallocInst(const Type *Ty, Value *ArraySize = 0,
                      const std::string &Name = "",
                      Instruction *InsertBefore = 0)
    : AllocationInst(Ty, ArraySize, Malloc, 0, Name, InsertBefore) {}
  MallocInst(const Type *Ty, Value *ArraySize, const std::string &Name,
             BasicBlock *InsertAtEnd)
    : AllocationInst(Ty, ArraySize, Malloc, 0, Name, InsertAtEnd) {}

  MallocInst(const Type *Ty, const std::string &Name,
             Instruction *InsertBefore = 0)
    : AllocationInst(Ty, 0, Malloc, 0, Name, InsertBefore) {}
  MallocInst(const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd)
    : AllocationInst(Ty, 0, Malloc, 0, Name, InsertAtEnd) {}

  MallocInst(const Type *Ty, Value *ArraySize, unsigned Align,
             const std::string &Name, BasicBlock *InsertAtEnd)
    : AllocationInst(Ty, ArraySize, Malloc, Align, Name, InsertAtEnd) {}
  MallocInst(const Type *Ty, Value *ArraySize, unsigned Align,
                      const std::string &Name = "",
                      Instruction *InsertBefore = 0)
    : AllocationInst(Ty, ArraySize, Malloc, Align, Name, InsertBefore) {}

  virtual MallocInst *clone() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const MallocInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Malloc);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                AllocaInst Class
//===----------------------------------------------------------------------===//

/// AllocaInst - an instruction to allocate memory on the stack
///
class AllocaInst : public AllocationInst {
  AllocaInst(const AllocaInst &);
public:
  explicit AllocaInst(const Type *Ty, Value *ArraySize = 0,
                      const std::string &Name = "",
                      Instruction *InsertBefore = 0)
    : AllocationInst(Ty, ArraySize, Alloca, 0, Name, InsertBefore) {}
  AllocaInst(const Type *Ty, Value *ArraySize, const std::string &Name,
             BasicBlock *InsertAtEnd)
    : AllocationInst(Ty, ArraySize, Alloca, 0, Name, InsertAtEnd) {}

  AllocaInst(const Type *Ty, const std::string &Name,
             Instruction *InsertBefore = 0)
    : AllocationInst(Ty, 0, Alloca, 0, Name, InsertBefore) {}
  AllocaInst(const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd)
    : AllocationInst(Ty, 0, Alloca, 0, Name, InsertAtEnd) {}

  AllocaInst(const Type *Ty, Value *ArraySize, unsigned Align,
             const std::string &Name = "", Instruction *InsertBefore = 0)
    : AllocationInst(Ty, ArraySize, Alloca, Align, Name, InsertBefore) {}
  AllocaInst(const Type *Ty, Value *ArraySize, unsigned Align,
             const std::string &Name, BasicBlock *InsertAtEnd)
    : AllocationInst(Ty, ArraySize, Alloca, Align, Name, InsertAtEnd) {}

  virtual AllocaInst *clone() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const AllocaInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Alloca);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                 FreeInst Class
//===----------------------------------------------------------------------===//

/// FreeInst - an instruction to deallocate memory
///
class FreeInst : public UnaryInstruction {
  void AssertOK();
public:
  explicit FreeInst(Value *Ptr, Instruction *InsertBefore = 0);
  FreeInst(Value *Ptr, BasicBlock *InsertAfter);

  virtual FreeInst *clone() const;
  
  // Accessor methods for consistency with other memory operations
  Value *getPointerOperand() { return getOperand(0); }
  const Value *getPointerOperand() const { return getOperand(0); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const FreeInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Free);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                LoadInst Class
//===----------------------------------------------------------------------===//

/// LoadInst - an instruction for reading from memory.  This uses the
/// SubclassData field in Value to store whether or not the load is volatile.
///
class LoadInst : public UnaryInstruction {

  LoadInst(const LoadInst &LI)
    : UnaryInstruction(LI.getType(), Load, LI.getOperand(0)) {
    setVolatile(LI.isVolatile());
    setAlignment(LI.getAlignment());

#ifndef NDEBUG
    AssertOK();
#endif
  }
  void AssertOK();
public:
  LoadInst(Value *Ptr, const std::string &Name, Instruction *InsertBefore);
  LoadInst(Value *Ptr, const std::string &Name, BasicBlock *InsertAtEnd);
  LoadInst(Value *Ptr, const std::string &Name, bool isVolatile = false, 
           Instruction *InsertBefore = 0);
  LoadInst(Value *Ptr, const std::string &Name, bool isVolatile, unsigned Align,
           Instruction *InsertBefore = 0);
  LoadInst(Value *Ptr, const std::string &Name, bool isVolatile,
           BasicBlock *InsertAtEnd);
  LoadInst(Value *Ptr, const std::string &Name, bool isVolatile, unsigned Align,
           BasicBlock *InsertAtEnd);

  LoadInst(Value *Ptr, const char *Name, Instruction *InsertBefore);
  LoadInst(Value *Ptr, const char *Name, BasicBlock *InsertAtEnd);
  explicit LoadInst(Value *Ptr, const char *Name = 0, bool isVolatile = false, 
                    Instruction *InsertBefore = 0);
  LoadInst(Value *Ptr, const char *Name, bool isVolatile,
           BasicBlock *InsertAtEnd);
  
  /// isVolatile - Return true if this is a load from a volatile memory
  /// location.
  ///
  bool isVolatile() const { return SubclassData & 1; }

  /// setVolatile - Specify whether this is a volatile load or not.
  ///
  void setVolatile(bool V) { 
    SubclassData = (SubclassData & ~1) | V; 
  }

  virtual LoadInst *clone() const;

  /// getAlignment - Return the alignment of the access that is being performed
  ///
  unsigned getAlignment() const {
    return (1 << (SubclassData>>1)) >> 1;
  }
  
  void setAlignment(unsigned Align);

  Value *getPointerOperand() { return getOperand(0); }
  const Value *getPointerOperand() const { return getOperand(0); }
  static unsigned getPointerOperandIndex() { return 0U; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const LoadInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Load;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                                StoreInst Class
//===----------------------------------------------------------------------===//

/// StoreInst - an instruction for storing to memory
///
class StoreInst : public Instruction {
  Use Ops[2];
  
  StoreInst(const StoreInst &SI) : Instruction(SI.getType(), Store, Ops, 2) {
    Ops[0].init(SI.Ops[0], this);
    Ops[1].init(SI.Ops[1], this);
    setVolatile(SI.isVolatile());
    setAlignment(SI.getAlignment());
    
#ifndef NDEBUG
    AssertOK();
#endif
  }
  void AssertOK();
public:
  StoreInst(Value *Val, Value *Ptr, Instruction *InsertBefore);
  StoreInst(Value *Val, Value *Ptr, BasicBlock *InsertAtEnd);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile = false,
            Instruction *InsertBefore = 0);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile,
            unsigned Align, Instruction *InsertBefore = 0);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile, BasicBlock *InsertAtEnd);
  StoreInst(Value *Val, Value *Ptr, bool isVolatile,
            unsigned Align, BasicBlock *InsertAtEnd);


  /// isVolatile - Return true if this is a load from a volatile memory
  /// location.
  ///
  bool isVolatile() const { return SubclassData & 1; }

  /// setVolatile - Specify whether this is a volatile load or not.
  ///
  void setVolatile(bool V) { 
    SubclassData = (SubclassData & ~1) | V; 
  }

  /// Transparently provide more efficient getOperand methods.
  Value *getOperand(unsigned i) const {
    assert(i < 2 && "getOperand() out of range!");
    return Ops[i];
  }
  void setOperand(unsigned i, Value *Val) {
    assert(i < 2 && "setOperand() out of range!");
    Ops[i] = Val;
  }
  unsigned getNumOperands() const { return 2; }

  /// getAlignment - Return the alignment of the access that is being performed
  ///
  unsigned getAlignment() const {
    return (1 << (SubclassData>>1)) >> 1;
  }
  
  void setAlignment(unsigned Align);
  
  virtual StoreInst *clone() const;

  Value *getPointerOperand() { return getOperand(1); }
  const Value *getPointerOperand() const { return getOperand(1); }
  static unsigned getPointerOperandIndex() { return 1U; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const StoreInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Store;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                             GetElementPtrInst Class
//===----------------------------------------------------------------------===//

// checkType - Simple wrapper function to give a better assertion failure
// message on bad indexes for a gep instruction.
//
static inline const Type *checkType(const Type *Ty) {
  assert(Ty && "Invalid GetElementPtrInst indices for type!");
  return Ty;
}

/// GetElementPtrInst - an instruction for type-safe pointer arithmetic to
/// access elements of arrays and structs
///
class GetElementPtrInst : public Instruction {
  GetElementPtrInst(const GetElementPtrInst &GEPI)
    : Instruction(reinterpret_cast<const Type*>(GEPI.getType()), GetElementPtr,
                  0, GEPI.getNumOperands()) {
    Use *OL = OperandList = new Use[NumOperands];
    Use *GEPIOL = GEPI.OperandList;
    for (unsigned i = 0, E = NumOperands; i != E; ++i)
      OL[i].init(GEPIOL[i], this);
  }
  void init(Value *Ptr, Value* const *Idx, unsigned NumIdx);
  void init(Value *Ptr, Value *Idx);

  template<typename InputIterator>
  void init(Value *Ptr, InputIterator IdxBegin, InputIterator IdxEnd,
            const std::string &Name,
            // This argument ensures that we have an iterator we can
            // do arithmetic on in constant time
            std::random_access_iterator_tag) {
    typename std::iterator_traits<InputIterator>::difference_type NumIdx = 
      std::distance(IdxBegin, IdxEnd);
    
    if (NumIdx > 0) {
      // This requires that the itoerator points to contiguous memory.
      init(Ptr, &*IdxBegin, NumIdx);
    }
    else {
      init(Ptr, 0, NumIdx);
    }

    setName(Name);
  }

  /// getIndexedType - Returns the type of the element that would be loaded with
  /// a load instruction with the specified parameters.
  ///
  /// A null type is returned if the indices are invalid for the specified
  /// pointer type.
  ///
  static const Type *getIndexedType(const Type *Ptr,
                                    Value* const *Idx, unsigned NumIdx,
                                    bool AllowStructLeaf = false);

  template<typename InputIterator>
  static const Type *getIndexedType(const Type *Ptr,
                                    InputIterator IdxBegin, 
                                    InputIterator IdxEnd,
                                    bool AllowStructLeaf,
                                    // This argument ensures that we
                                    // have an iterator we can do
                                    // arithmetic on in constant time
                                    std::random_access_iterator_tag) {
    typename std::iterator_traits<InputIterator>::difference_type NumIdx = 
      std::distance(IdxBegin, IdxEnd);

    if (NumIdx > 0) {
      // This requires that the iterator points to contiguous memory.
      return(getIndexedType(Ptr, (Value *const *)&*IdxBegin, NumIdx,
                            AllowStructLeaf));
    }
    else {
      return(getIndexedType(Ptr, (Value *const*)0, NumIdx, AllowStructLeaf));
    }
  }

public:
  /// Constructors - Create a getelementptr instruction with a base pointer an
  /// list of indices.  The first ctor can optionally insert before an existing
  /// instruction, the second appends the new instruction to the specified
  /// BasicBlock.
  template<typename InputIterator>
  GetElementPtrInst(Value *Ptr, InputIterator IdxBegin, 
                    InputIterator IdxEnd,
                    const std::string &Name = "",
                    Instruction *InsertBefore =0)
      : Instruction(PointerType::get(
                      checkType(getIndexedType(Ptr->getType(),
                                               IdxBegin, IdxEnd, true))),
                    GetElementPtr, 0, 0, InsertBefore) {
    init(Ptr, IdxBegin, IdxEnd, Name,
         typename std::iterator_traits<InputIterator>::iterator_category());
  }
  template<typename InputIterator>
  GetElementPtrInst(Value *Ptr, InputIterator IdxBegin, InputIterator IdxEnd,
                    const std::string &Name, BasicBlock *InsertAtEnd)
      : Instruction(PointerType::get(
                      checkType(getIndexedType(Ptr->getType(),
                                               IdxBegin, IdxEnd, true))),
                    GetElementPtr, 0, 0, InsertAtEnd) {
    init(Ptr, IdxBegin, IdxEnd, Name,
         typename std::iterator_traits<InputIterator>::iterator_category());
  }

  /// Constructors - These two constructors are convenience methods because one
  /// and two index getelementptr instructions are so common.
  GetElementPtrInst(Value *Ptr, Value *Idx,
                    const std::string &Name = "", Instruction *InsertBefore =0);
  GetElementPtrInst(Value *Ptr, Value *Idx,
                    const std::string &Name, BasicBlock *InsertAtEnd);
  ~GetElementPtrInst();

  virtual GetElementPtrInst *clone() const;

  // getType - Overload to return most specific pointer type...
  inline const PointerType *getType() const {
    return reinterpret_cast<const PointerType*>(Instruction::getType());
  }

  /// getIndexedType - Returns the type of the element that would be loaded with
  /// a load instruction with the specified parameters.
  ///
  /// A null type is returned if the indices are invalid for the specified
  /// pointer type.
  ///
  template<typename InputIterator>
  static const Type *getIndexedType(const Type *Ptr,
                                    InputIterator IdxBegin,
                                    InputIterator IdxEnd,
                                    bool AllowStructLeaf = false) {
    return(getIndexedType(Ptr, IdxBegin, IdxEnd, AllowStructLeaf, 
                          typename std::iterator_traits<InputIterator>::
                          iterator_category()));
  }  
  static const Type *getIndexedType(const Type *Ptr, Value *Idx);

  inline op_iterator       idx_begin()       { return op_begin()+1; }
  inline const_op_iterator idx_begin() const { return op_begin()+1; }
  inline op_iterator       idx_end()         { return op_end(); }
  inline const_op_iterator idx_end()   const { return op_end(); }

  Value *getPointerOperand() {
    return getOperand(0);
  }
  const Value *getPointerOperand() const {
    return getOperand(0);
  }
  static unsigned getPointerOperandIndex() {
    return 0U;                      // get index for modifying correct operand
  }

  inline unsigned getNumIndices() const {  // Note: always non-negative
    return getNumOperands() - 1;
  }

  inline bool hasIndices() const {
    return getNumOperands() > 1;
  }
  
  /// hasAllZeroIndices - Return true if all of the indices of this GEP are
  /// zeros.  If so, the result pointer and the first operand have the same
  /// value, just potentially different types.
  bool hasAllZeroIndices() const;
  
  /// hasAllConstantIndices - Return true if all of the indices of this GEP are
  /// constant integers.  If so, the result pointer and the first operand have
  /// a constant offset between them.
  bool hasAllConstantIndices() const;
  

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const GetElementPtrInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::GetElementPtr);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                               ICmpInst Class
//===----------------------------------------------------------------------===//

/// This instruction compares its operands according to the predicate given
/// to the constructor. It only operates on integers, pointers, or packed 
/// vectors of integrals. The two operands must be the same type.
/// @brief Represent an integer comparison operator.
class ICmpInst: public CmpInst {
public:
  /// This enumeration lists the possible predicates for the ICmpInst. The
  /// values in the range 0-31 are reserved for FCmpInst while values in the
  /// range 32-64 are reserved for ICmpInst. This is necessary to ensure the
  /// predicate values are not overlapping between the classes.
  enum Predicate {
    ICMP_EQ  = 32,    ///< equal
    ICMP_NE  = 33,    ///< not equal
    ICMP_UGT = 34,    ///< unsigned greater than
    ICMP_UGE = 35,    ///< unsigned greater or equal
    ICMP_ULT = 36,    ///< unsigned less than
    ICMP_ULE = 37,    ///< unsigned less or equal
    ICMP_SGT = 38,    ///< signed greater than
    ICMP_SGE = 39,    ///< signed greater or equal
    ICMP_SLT = 40,    ///< signed less than
    ICMP_SLE = 41,    ///< signed less or equal
    FIRST_ICMP_PREDICATE = ICMP_EQ,
    LAST_ICMP_PREDICATE = ICMP_SLE,
    BAD_ICMP_PREDICATE = ICMP_SLE + 1
  };

  /// @brief Constructor with insert-before-instruction semantics.
  ICmpInst(
    Predicate pred,  ///< The predicate to use for the comparison
    Value *LHS,      ///< The left-hand-side of the expression
    Value *RHS,      ///< The right-hand-side of the expression
    const std::string &Name = "",  ///< Name of the instruction
    Instruction *InsertBefore = 0  ///< Where to insert
  ) : CmpInst(Instruction::ICmp, pred, LHS, RHS, Name, InsertBefore) {
  }

  /// @brief Constructor with insert-at-block-end semantics.
  ICmpInst(
    Predicate pred, ///< The predicate to use for the comparison
    Value *LHS,     ///< The left-hand-side of the expression
    Value *RHS,     ///< The right-hand-side of the expression
    const std::string &Name,  ///< Name of the instruction
    BasicBlock *InsertAtEnd   ///< Block to insert into.
  ) : CmpInst(Instruction::ICmp, pred, LHS, RHS, Name, InsertAtEnd) {
  }

  /// @brief Return the predicate for this instruction.
  Predicate getPredicate() const { return Predicate(SubclassData); }

  /// @brief Set the predicate for this instruction to the specified value.
  void setPredicate(Predicate P) { SubclassData = P; }
  
  /// For example, EQ -> NE, UGT -> ULE, SLT -> SGE, etc.
  /// @returns the inverse predicate for the instruction's current predicate. 
  /// @brief Return the inverse of the instruction's predicate.
  Predicate getInversePredicate() const {
    return getInversePredicate(getPredicate());
  }

  /// For example, EQ -> NE, UGT -> ULE, SLT -> SGE, etc.
  /// @returns the inverse predicate for predicate provided in \p pred. 
  /// @brief Return the inverse of a given predicate
  static Predicate getInversePredicate(Predicate pred);

  /// For example, EQ->EQ, SLE->SGE, ULT->UGT, etc.
  /// @returns the predicate that would be the result of exchanging the two 
  /// operands of the ICmpInst instruction without changing the result 
  /// produced.  
  /// @brief Return the predicate as if the operands were swapped
  Predicate getSwappedPredicate() const {
    return getSwappedPredicate(getPredicate());
  }

  /// This is a static version that you can use without an instruction 
  /// available.
  /// @brief Return the predicate as if the operands were swapped.
  static Predicate getSwappedPredicate(Predicate pred);

  /// For example, EQ->EQ, SLE->SLE, UGT->SGT, etc.
  /// @returns the predicate that would be the result if the operand were
  /// regarded as signed.
  /// @brief Return the signed version of the predicate
  Predicate getSignedPredicate() const {
    return getSignedPredicate(getPredicate());
  }

  /// This is a static version that you can use without an instruction.
  /// @brief Return the signed version of the predicate.
  static Predicate getSignedPredicate(Predicate pred);

  /// This also tests for commutativity. If isEquality() returns true then
  /// the predicate is also commutative. 
  /// @returns true if the predicate of this instruction is EQ or NE.
  /// @brief Determine if this is an equality predicate.
  bool isEquality() const {
    return SubclassData == ICMP_EQ || SubclassData == ICMP_NE;
  }

  /// @returns true if the predicate of this ICmpInst is commutative
  /// @brief Determine if this relation is commutative.
  bool isCommutative() const { return isEquality(); }

  /// @returns true if the predicate is relational (not EQ or NE). 
  /// @brief Determine if this a relational predicate.
  bool isRelational() const {
    return !isEquality();
  }

  /// @returns true if the predicate of this ICmpInst is signed, false otherwise
  /// @brief Determine if this instruction's predicate is signed.
  bool isSignedPredicate() const { return isSignedPredicate(getPredicate()); }

  /// @returns true if the predicate provided is signed, false otherwise
  /// @brief Determine if the predicate is signed.
  static bool isSignedPredicate(Predicate pred);

  /// Initialize a set of values that all satisfy the predicate with C. 
  /// @brief Make a ConstantRange for a relation with a constant value.
  static ConstantRange makeConstantRange(Predicate pred, const APInt &C);

  /// Exchange the two operands to this instruction in such a way that it does
  /// not modify the semantics of the instruction. The predicate value may be
  /// changed to retain the same result if the predicate is order dependent
  /// (e.g. ult). 
  /// @brief Swap operands and adjust predicate.
  void swapOperands() {
    SubclassData = getSwappedPredicate();
    std::swap(Ops[0], Ops[1]);
  }

  virtual ICmpInst *clone() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ICmpInst *) { return true; }
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
/// @brief Represents a floating point comparison operator.
class FCmpInst: public CmpInst {
public:
  /// This enumeration lists the possible predicates for the FCmpInst. Values
  /// in the range 0-31 are reserved for FCmpInst.
  enum Predicate {
    // Opcode        U L G E    Intuitive operation
    FCMP_FALSE = 0, ///<  0 0 0 0    Always false (always folded)
    FCMP_OEQ   = 1, ///<  0 0 0 1    True if ordered and equal
    FCMP_OGT   = 2, ///<  0 0 1 0    True if ordered and greater than
    FCMP_OGE   = 3, ///<  0 0 1 1    True if ordered and greater than or equal
    FCMP_OLT   = 4, ///<  0 1 0 0    True if ordered and less than
    FCMP_OLE   = 5, ///<  0 1 0 1    True if ordered and less than or equal
    FCMP_ONE   = 6, ///<  0 1 1 0    True if ordered and operands are unequal
    FCMP_ORD   = 7, ///<  0 1 1 1    True if ordered (no nans)
    FCMP_UNO   = 8, ///<  1 0 0 0    True if unordered: isnan(X) | isnan(Y)
    FCMP_UEQ   = 9, ///<  1 0 0 1    True if unordered or equal
    FCMP_UGT   =10, ///<  1 0 1 0    True if unordered or greater than
    FCMP_UGE   =11, ///<  1 0 1 1    True if unordered, greater than, or equal
    FCMP_ULT   =12, ///<  1 1 0 0    True if unordered or less than
    FCMP_ULE   =13, ///<  1 1 0 1    True if unordered, less than, or equal
    FCMP_UNE   =14, ///<  1 1 1 0    True if unordered or not equal
    FCMP_TRUE  =15, ///<  1 1 1 1    Always true (always folded)
    FIRST_FCMP_PREDICATE = FCMP_FALSE,
    LAST_FCMP_PREDICATE = FCMP_TRUE,
    BAD_FCMP_PREDICATE = FCMP_TRUE + 1
  };

  /// @brief Constructor with insert-before-instruction semantics.
  FCmpInst(
    Predicate pred,  ///< The predicate to use for the comparison
    Value *LHS,      ///< The left-hand-side of the expression
    Value *RHS,      ///< The right-hand-side of the expression
    const std::string &Name = "",  ///< Name of the instruction
    Instruction *InsertBefore = 0  ///< Where to insert
  ) : CmpInst(Instruction::FCmp, pred, LHS, RHS, Name, InsertBefore) {
  }

  /// @brief Constructor with insert-at-block-end semantics.
  FCmpInst(
    Predicate pred, ///< The predicate to use for the comparison
    Value *LHS,     ///< The left-hand-side of the expression
    Value *RHS,     ///< The right-hand-side of the expression
    const std::string &Name,  ///< Name of the instruction
    BasicBlock *InsertAtEnd   ///< Block to insert into.
  ) : CmpInst(Instruction::FCmp, pred, LHS, RHS, Name, InsertAtEnd) {
  }

  /// @brief Return the predicate for this instruction.
  Predicate getPredicate() const { return Predicate(SubclassData); }

  /// @brief Set the predicate for this instruction to the specified value.
  void setPredicate(Predicate P) { SubclassData = P; }

  /// For example, OEQ -> UNE, UGT -> OLE, OLT -> UGE, etc.
  /// @returns the inverse predicate for the instructions current predicate. 
  /// @brief Return the inverse of the predicate
  Predicate getInversePredicate() const {
    return getInversePredicate(getPredicate());
  }

  /// For example, OEQ -> UNE, UGT -> OLE, OLT -> UGE, etc.
  /// @returns the inverse predicate for \p pred.
  /// @brief Return the inverse of a given predicate
  static Predicate getInversePredicate(Predicate pred);

  /// For example, OEQ->OEQ, ULE->UGE, OLT->OGT, etc.
  /// @returns the predicate that would be the result of exchanging the two 
  /// operands of the ICmpInst instruction without changing the result 
  /// produced.  
  /// @brief Return the predicate as if the operands were swapped
  Predicate getSwappedPredicate() const {
    return getSwappedPredicate(getPredicate());
  }

  /// This is a static version that you can use without an instruction 
  /// available.
  /// @brief Return the predicate as if the operands were swapped.
  static Predicate getSwappedPredicate(Predicate Opcode);

  /// This also tests for commutativity. If isEquality() returns true then
  /// the predicate is also commutative. Only the equality predicates are
  /// commutative.
  /// @returns true if the predicate of this instruction is EQ or NE.
  /// @brief Determine if this is an equality predicate.
  bool isEquality() const {
    return SubclassData == FCMP_OEQ || SubclassData == FCMP_ONE ||
           SubclassData == FCMP_UEQ || SubclassData == FCMP_UNE;
  }
  bool isCommutative() const { return isEquality(); }

  /// @returns true if the predicate is relational (not EQ or NE). 
  /// @brief Determine if this a relational predicate.
  bool isRelational() const { return !isEquality(); }

  /// Exchange the two operands to this instruction in such a way that it does
  /// not modify the semantics of the instruction. The predicate value may be
  /// changed to retain the same result if the predicate is order dependent
  /// (e.g. ult). 
  /// @brief Swap operands and adjust predicate.
  void swapOperands() {
    SubclassData = getSwappedPredicate();
    std::swap(Ops[0], Ops[1]);
  }

  virtual FCmpInst *clone() const;

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const FCmpInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::FCmp;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                 CallInst Class
//===----------------------------------------------------------------------===//
/// CallInst - This class represents a function call, abstracting a target
/// machine's calling convention.  This class uses low bit of the SubClassData
/// field to indicate whether or not this is a tail call.  The rest of the bits
/// hold the calling convention of the call.
///

class CallInst : public Instruction {
  ParamAttrsList *ParamAttrs; ///< parameter attributes for call
  CallInst(const CallInst &CI);
  void init(Value *Func, Value* const *Params, unsigned NumParams);
  void init(Value *Func, Value *Actual1, Value *Actual2);
  void init(Value *Func, Value *Actual);
  void init(Value *Func);

  template<typename InputIterator>
  void init(Value *Func, InputIterator ArgBegin, InputIterator ArgEnd,
            const std::string &Name,
            // This argument ensures that we have an iterator we can
            // do arithmetic on in constant time
            std::random_access_iterator_tag) {
    unsigned NumArgs = (unsigned)std::distance(ArgBegin, ArgEnd);
    
    // This requires that the iterator points to contiguous memory.
    init(Func, NumArgs ? &*ArgBegin : 0, NumArgs);
    setName(Name);
  }

public:
  /// Construct a CallInst given a range of arguments.  InputIterator
  /// must be a random-access iterator pointing to contiguous storage
  /// (e.g. a std::vector<>::iterator).  Checks are made for
  /// random-accessness but not for contiguous storage as that would
  /// incur runtime overhead.
  /// @brief Construct a CallInst from a range of arguments
  template<typename InputIterator>
  CallInst(Value *Func, InputIterator ArgBegin, InputIterator ArgEnd,
           const std::string &Name = "", Instruction *InsertBefore = 0)
      : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                       ->getElementType())->getReturnType(),
                    Instruction::Call, 0, 0, InsertBefore) {
    init(Func, ArgBegin, ArgEnd, Name, 
         typename std::iterator_traits<InputIterator>::iterator_category());
  }

  /// Construct a CallInst given a range of arguments.  InputIterator
  /// must be a random-access iterator pointing to contiguous storage
  /// (e.g. a std::vector<>::iterator).  Checks are made for
  /// random-accessness but not for contiguous storage as that would
  /// incur runtime overhead.
  /// @brief Construct a CallInst from a range of arguments
  template<typename InputIterator>
  CallInst(Value *Func, InputIterator ArgBegin, InputIterator ArgEnd,
           const std::string &Name, BasicBlock *InsertAtEnd)
      : Instruction(cast<FunctionType>(cast<PointerType>(Func->getType())
                                       ->getElementType())->getReturnType(),
                    Instruction::Call, 0, 0, InsertAtEnd) {
    init(Func, ArgBegin, ArgEnd, Name,
         typename std::iterator_traits<InputIterator>::iterator_category());
  }

  CallInst(Value *F, Value *Actual, const std::string& Name = "",
           Instruction *InsertBefore = 0);
  CallInst(Value *F, Value *Actual, const std::string& Name,
           BasicBlock *InsertAtEnd);
  explicit CallInst(Value *F, const std::string &Name = "",
                    Instruction *InsertBefore = 0);
  CallInst(Value *F, const std::string &Name, BasicBlock *InsertAtEnd);
  ~CallInst();

  virtual CallInst *clone() const;
  
  bool isTailCall() const           { return SubclassData & 1; }
  void setTailCall(bool isTailCall = true) {
    SubclassData = (SubclassData & ~1) | unsigned(isTailCall);
  }

  /// getCallingConv/setCallingConv - Get or set the calling convention of this
  /// function call.
  unsigned getCallingConv() const { return SubclassData >> 1; }
  void setCallingConv(unsigned CC) {
    SubclassData = (SubclassData & 1) | (CC << 1);
  }

  /// Obtains a pointer to the ParamAttrsList object which holds the
  /// parameter attributes information, if any.
  /// @returns 0 if no attributes have been set.
  /// @brief Get the parameter attributes.
  ParamAttrsList *getParamAttrs() const { return ParamAttrs; }

  /// Sets the parameter attributes for this CallInst. To construct a 
  /// ParamAttrsList, see ParameterAttributes.h
  /// @brief Set the parameter attributes.
  void setParamAttrs(ParamAttrsList *attrs);

  /// getCalledFunction - Return the function being called by this instruction
  /// if it is a direct call.  If it is a call through a function pointer,
  /// return null.
  Function *getCalledFunction() const {
    return static_cast<Function*>(dyn_cast<Function>(getOperand(0)));
  }

  /// getCalledValue - Get a pointer to the function that is invoked by this 
  /// instruction
  inline const Value *getCalledValue() const { return getOperand(0); }
  inline       Value *getCalledValue()       { return getOperand(0); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const CallInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Call;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                               SelectInst Class
//===----------------------------------------------------------------------===//

/// SelectInst - This class represents the LLVM 'select' instruction.
///
class SelectInst : public Instruction {
  Use Ops[3];

  void init(Value *C, Value *S1, Value *S2) {
    Ops[0].init(C, this);
    Ops[1].init(S1, this);
    Ops[2].init(S2, this);
  }

  SelectInst(const SelectInst &SI)
    : Instruction(SI.getType(), SI.getOpcode(), Ops, 3) {
    init(SI.Ops[0], SI.Ops[1], SI.Ops[2]);
  }
public:
  SelectInst(Value *C, Value *S1, Value *S2, const std::string &Name = "",
             Instruction *InsertBefore = 0)
    : Instruction(S1->getType(), Instruction::Select, Ops, 3, InsertBefore) {
    init(C, S1, S2);
    setName(Name);
  }
  SelectInst(Value *C, Value *S1, Value *S2, const std::string &Name,
             BasicBlock *InsertAtEnd)
    : Instruction(S1->getType(), Instruction::Select, Ops, 3, InsertAtEnd) {
    init(C, S1, S2);
    setName(Name);
  }

  Value *getCondition() const { return Ops[0]; }
  Value *getTrueValue() const { return Ops[1]; }
  Value *getFalseValue() const { return Ops[2]; }

  /// Transparently provide more efficient getOperand methods.
  Value *getOperand(unsigned i) const {
    assert(i < 3 && "getOperand() out of range!");
    return Ops[i];
  }
  void setOperand(unsigned i, Value *Val) {
    assert(i < 3 && "setOperand() out of range!");
    Ops[i] = Val;
  }
  unsigned getNumOperands() const { return 3; }

  OtherOps getOpcode() const {
    return static_cast<OtherOps>(Instruction::getOpcode());
  }

  virtual SelectInst *clone() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SelectInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Select;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                VAArgInst Class
//===----------------------------------------------------------------------===//

/// VAArgInst - This class represents the va_arg llvm instruction, which returns
/// an argument of the specified type given a va_list and increments that list
///
class VAArgInst : public UnaryInstruction {
  VAArgInst(const VAArgInst &VAA)
    : UnaryInstruction(VAA.getType(), VAArg, VAA.getOperand(0)) {}
public:
  VAArgInst(Value *List, const Type *Ty, const std::string &Name = "",
             Instruction *InsertBefore = 0)
    : UnaryInstruction(Ty, VAArg, List, InsertBefore) {
    setName(Name);
  }
  VAArgInst(Value *List, const Type *Ty, const std::string &Name,
            BasicBlock *InsertAtEnd)
    : UnaryInstruction(Ty, VAArg, List, InsertAtEnd) {
    setName(Name);
  }

  virtual VAArgInst *clone() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const VAArgInst *) { return true; }
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

/// ExtractElementInst - This instruction extracts a single (scalar)
/// element from a VectorType value
///
class ExtractElementInst : public Instruction {
  Use Ops[2];
  ExtractElementInst(const ExtractElementInst &EE) :
    Instruction(EE.getType(), ExtractElement, Ops, 2) {
    Ops[0].init(EE.Ops[0], this);
    Ops[1].init(EE.Ops[1], this);
  }

public:
  ExtractElementInst(Value *Vec, Value *Idx, const std::string &Name = "",
                     Instruction *InsertBefore = 0);
  ExtractElementInst(Value *Vec, unsigned Idx, const std::string &Name = "",
                     Instruction *InsertBefore = 0);
  ExtractElementInst(Value *Vec, Value *Idx, const std::string &Name,
                     BasicBlock *InsertAtEnd);
  ExtractElementInst(Value *Vec, unsigned Idx, const std::string &Name,
                     BasicBlock *InsertAtEnd);

  /// isValidOperands - Return true if an extractelement instruction can be
  /// formed with the specified operands.
  static bool isValidOperands(const Value *Vec, const Value *Idx);

  virtual ExtractElementInst *clone() const;

  /// Transparently provide more efficient getOperand methods.
  Value *getOperand(unsigned i) const {
    assert(i < 2 && "getOperand() out of range!");
    return Ops[i];
  }
  void setOperand(unsigned i, Value *Val) {
    assert(i < 2 && "setOperand() out of range!");
    Ops[i] = Val;
  }
  unsigned getNumOperands() const { return 2; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ExtractElementInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::ExtractElement;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                                InsertElementInst Class
//===----------------------------------------------------------------------===//

/// InsertElementInst - This instruction inserts a single (scalar)
/// element into a VectorType value
///
class InsertElementInst : public Instruction {
  Use Ops[3];
  InsertElementInst(const InsertElementInst &IE);
public:
  InsertElementInst(Value *Vec, Value *NewElt, Value *Idx,
                    const std::string &Name = "",Instruction *InsertBefore = 0);
  InsertElementInst(Value *Vec, Value *NewElt, unsigned Idx,
                    const std::string &Name = "",Instruction *InsertBefore = 0);
  InsertElementInst(Value *Vec, Value *NewElt, Value *Idx,
                    const std::string &Name, BasicBlock *InsertAtEnd);
  InsertElementInst(Value *Vec, Value *NewElt, unsigned Idx,
                    const std::string &Name, BasicBlock *InsertAtEnd);

  /// isValidOperands - Return true if an insertelement instruction can be
  /// formed with the specified operands.
  static bool isValidOperands(const Value *Vec, const Value *NewElt,
                              const Value *Idx);

  virtual InsertElementInst *clone() const;

  /// getType - Overload to return most specific vector type.
  ///
  inline const VectorType *getType() const {
    return reinterpret_cast<const VectorType*>(Instruction::getType());
  }

  /// Transparently provide more efficient getOperand methods.
  Value *getOperand(unsigned i) const {
    assert(i < 3 && "getOperand() out of range!");
    return Ops[i];
  }
  void setOperand(unsigned i, Value *Val) {
    assert(i < 3 && "setOperand() out of range!");
    Ops[i] = Val;
  }
  unsigned getNumOperands() const { return 3; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const InsertElementInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::InsertElement;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                           ShuffleVectorInst Class
//===----------------------------------------------------------------------===//

/// ShuffleVectorInst - This instruction constructs a fixed permutation of two
/// input vectors.
///
class ShuffleVectorInst : public Instruction {
  Use Ops[3];
  ShuffleVectorInst(const ShuffleVectorInst &IE);
public:
  ShuffleVectorInst(Value *V1, Value *V2, Value *Mask,
                    const std::string &Name = "", Instruction *InsertBefor = 0);
  ShuffleVectorInst(Value *V1, Value *V2, Value *Mask,
                    const std::string &Name, BasicBlock *InsertAtEnd);

  /// isValidOperands - Return true if a shufflevector instruction can be
  /// formed with the specified operands.
  static bool isValidOperands(const Value *V1, const Value *V2,
                              const Value *Mask);

  virtual ShuffleVectorInst *clone() const;

  /// getType - Overload to return most specific vector type.
  ///
  inline const VectorType *getType() const {
    return reinterpret_cast<const VectorType*>(Instruction::getType());
  }

  /// Transparently provide more efficient getOperand methods.
  Value *getOperand(unsigned i) const {
    assert(i < 3 && "getOperand() out of range!");
    return Ops[i];
  }
  void setOperand(unsigned i, Value *Val) {
    assert(i < 3 && "setOperand() out of range!");
    Ops[i] = Val;
  }
  unsigned getNumOperands() const { return 3; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ShuffleVectorInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::ShuffleVector;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                               PHINode Class
//===----------------------------------------------------------------------===//

// PHINode - The PHINode class is used to represent the magical mystical PHI
// node, that can not exist in nature, but can be synthesized in a computer
// scientist's overactive imagination.
//
class PHINode : public Instruction {
  /// ReservedSpace - The number of operands actually allocated.  NumOperands is
  /// the number actually in use.
  unsigned ReservedSpace;
  PHINode(const PHINode &PN);
public:
  explicit PHINode(const Type *Ty, const std::string &Name = "",
                   Instruction *InsertBefore = 0)
    : Instruction(Ty, Instruction::PHI, 0, 0, InsertBefore),
      ReservedSpace(0) {
    setName(Name);
  }

  PHINode(const Type *Ty, const std::string &Name, BasicBlock *InsertAtEnd)
    : Instruction(Ty, Instruction::PHI, 0, 0, InsertAtEnd),
      ReservedSpace(0) {
    setName(Name);
  }

  ~PHINode();

  /// reserveOperandSpace - This method can be used to avoid repeated
  /// reallocation of PHI operand lists by reserving space for the correct
  /// number of operands before adding them.  Unlike normal vector reserves,
  /// this method can also be used to trim the operand space.
  void reserveOperandSpace(unsigned NumValues) {
    resizeOperands(NumValues*2);
  }

  virtual PHINode *clone() const;

  /// getNumIncomingValues - Return the number of incoming edges
  ///
  unsigned getNumIncomingValues() const { return getNumOperands()/2; }

  /// getIncomingValue - Return incoming value number x
  ///
  Value *getIncomingValue(unsigned i) const {
    assert(i*2 < getNumOperands() && "Invalid value number!");
    return getOperand(i*2);
  }
  void setIncomingValue(unsigned i, Value *V) {
    assert(i*2 < getNumOperands() && "Invalid value number!");
    setOperand(i*2, V);
  }
  unsigned getOperandNumForIncomingValue(unsigned i) {
    return i*2;
  }

  /// getIncomingBlock - Return incoming basic block number x
  ///
  BasicBlock *getIncomingBlock(unsigned i) const {
    return reinterpret_cast<BasicBlock*>(getOperand(i*2+1));
  }
  void setIncomingBlock(unsigned i, BasicBlock *BB) {
    setOperand(i*2+1, reinterpret_cast<Value*>(BB));
  }
  unsigned getOperandNumForIncomingBlock(unsigned i) {
    return i*2+1;
  }

  /// addIncoming - Add an incoming value to the end of the PHI list
  ///
  void addIncoming(Value *V, BasicBlock *BB) {
    assert(getType() == V->getType() &&
           "All operands to PHI node must be the same type as the PHI node!");
    unsigned OpNo = NumOperands;
    if (OpNo+2 > ReservedSpace)
      resizeOperands(0);  // Get more space!
    // Initialize some new operands.
    NumOperands = OpNo+2;
    OperandList[OpNo].init(V, this);
    OperandList[OpNo+1].init(reinterpret_cast<Value*>(BB), this);
  }

  /// removeIncomingValue - Remove an incoming value.  This is useful if a
  /// predecessor basic block is deleted.  The value removed is returned.
  ///
  /// If the last incoming value for a PHI node is removed (and DeletePHIIfEmpty
  /// is true), the PHI node is destroyed and any uses of it are replaced with
  /// dummy values.  The only time there should be zero incoming values to a PHI
  /// node is when the block is dead, so this strategy is sound.
  ///
  Value *removeIncomingValue(unsigned Idx, bool DeletePHIIfEmpty = true);

  Value *removeIncomingValue(const BasicBlock *BB, bool DeletePHIIfEmpty =true){
    int Idx = getBasicBlockIndex(BB);
    assert(Idx >= 0 && "Invalid basic block argument to remove!");
    return removeIncomingValue(Idx, DeletePHIIfEmpty);
  }

  /// getBasicBlockIndex - Return the first index of the specified basic
  /// block in the value list for this PHI.  Returns -1 if no instance.
  ///
  int getBasicBlockIndex(const BasicBlock *BB) const {
    Use *OL = OperandList;
    for (unsigned i = 0, e = getNumOperands(); i != e; i += 2)
      if (OL[i+1] == reinterpret_cast<const Value*>(BB)) return i/2;
    return -1;
  }

  Value *getIncomingValueForBlock(const BasicBlock *BB) const {
    return getIncomingValue(getBasicBlockIndex(BB));
  }

  /// hasConstantValue - If the specified PHI node always merges together the
  /// same value, return the value, otherwise return null.
  ///
  Value *hasConstantValue(bool AllowNonDominatingInstruction = false) const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const PHINode *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::PHI;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
 private:
  void resizeOperands(unsigned NumOperands);
};

//===----------------------------------------------------------------------===//
//                               ReturnInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// ReturnInst - Return a value (possibly void), from a function.  Execution
/// does not continue in this function any longer.
///
class ReturnInst : public TerminatorInst {
  Use RetVal;  // Return Value: null if 'void'.
  ReturnInst(const ReturnInst &RI);
  void init(Value *RetVal);

public:
  // ReturnInst constructors:
  // ReturnInst()                  - 'ret void' instruction
  // ReturnInst(    null)          - 'ret void' instruction
  // ReturnInst(Value* X)          - 'ret X'    instruction
  // ReturnInst(    null, Inst *)  - 'ret void' instruction, insert before I
  // ReturnInst(Value* X, Inst *I) - 'ret X'    instruction, insert before I
  // ReturnInst(    null, BB *B)   - 'ret void' instruction, insert @ end of BB
  // ReturnInst(Value* X, BB *B)   - 'ret X'    instruction, insert @ end of BB
  //
  // NOTE: If the Value* passed is of type void then the constructor behaves as
  // if it was passed NULL.
  explicit ReturnInst(Value *retVal = 0, Instruction *InsertBefore = 0);
  ReturnInst(Value *retVal, BasicBlock *InsertAtEnd);
  explicit ReturnInst(BasicBlock *InsertAtEnd);

  virtual ReturnInst *clone() const;

  // Transparently provide more efficient getOperand methods.
  Value *getOperand(unsigned i) const {
    assert(i < getNumOperands() && "getOperand() out of range!");
    return RetVal;
  }
  void setOperand(unsigned i, Value *Val) {
    assert(i < getNumOperands() && "setOperand() out of range!");
    RetVal = Val;
  }

  Value *getReturnValue() const { return RetVal; }

  unsigned getNumSuccessors() const { return 0; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ReturnInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Ret);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
 private:
  virtual BasicBlock *getSuccessorV(unsigned idx) const;
  virtual unsigned getNumSuccessorsV() const;
  virtual void setSuccessorV(unsigned idx, BasicBlock *B);
};

//===----------------------------------------------------------------------===//
//                               BranchInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// BranchInst - Conditional or Unconditional Branch instruction.
///
class BranchInst : public TerminatorInst {
  /// Ops list - Branches are strange.  The operands are ordered:
  ///  TrueDest, FalseDest, Cond.  This makes some accessors faster because
  /// they don't have to check for cond/uncond branchness.
  Use Ops[3];
  BranchInst(const BranchInst &BI);
  void AssertOK();
public:
  // BranchInst constructors (where {B, T, F} are blocks, and C is a condition):
  // BranchInst(BB *B)                           - 'br B'
  // BranchInst(BB* T, BB *F, Value *C)          - 'br C, T, F'
  // BranchInst(BB* B, Inst *I)                  - 'br B'        insert before I
  // BranchInst(BB* T, BB *F, Value *C, Inst *I) - 'br C, T, F', insert before I
  // BranchInst(BB* B, BB *I)                    - 'br B'        insert at end
  // BranchInst(BB* T, BB *F, Value *C, BB *I)   - 'br C, T, F', insert at end
  explicit BranchInst(BasicBlock *IfTrue, Instruction *InsertBefore = 0);
  BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *Cond,
             Instruction *InsertBefore = 0);
  BranchInst(BasicBlock *IfTrue, BasicBlock *InsertAtEnd);
  BranchInst(BasicBlock *IfTrue, BasicBlock *IfFalse, Value *Cond,
             BasicBlock *InsertAtEnd);

  /// Transparently provide more efficient getOperand methods.
  Value *getOperand(unsigned i) const {
    assert(i < getNumOperands() && "getOperand() out of range!");
    return Ops[i];
  }
  void setOperand(unsigned i, Value *Val) {
    assert(i < getNumOperands() && "setOperand() out of range!");
    Ops[i] = Val;
  }

  virtual BranchInst *clone() const;

  inline bool isUnconditional() const { return getNumOperands() == 1; }
  inline bool isConditional()   const { return getNumOperands() == 3; }

  inline Value *getCondition() const {
    assert(isConditional() && "Cannot get condition of an uncond branch!");
    return getOperand(2);
  }

  void setCondition(Value *V) {
    assert(isConditional() && "Cannot set condition of unconditional branch!");
    setOperand(2, V);
  }

  // setUnconditionalDest - Change the current branch to an unconditional branch
  // targeting the specified block.
  // FIXME: Eliminate this ugly method.
  void setUnconditionalDest(BasicBlock *Dest) {
    if (isConditional()) {  // Convert this to an uncond branch.
      NumOperands = 1;
      Ops[1].set(0);
      Ops[2].set(0);
    }
    setOperand(0, reinterpret_cast<Value*>(Dest));
  }

  unsigned getNumSuccessors() const { return 1+isConditional(); }

  BasicBlock *getSuccessor(unsigned i) const {
    assert(i < getNumSuccessors() && "Successor # out of range for Branch!");
    return cast<BasicBlock>(getOperand(i));
  }

  void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(idx < getNumSuccessors() && "Successor # out of range for Branch!");
    setOperand(idx, reinterpret_cast<Value*>(NewSucc));
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const BranchInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Br);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
private:
  virtual BasicBlock *getSuccessorV(unsigned idx) const;
  virtual unsigned getNumSuccessorsV() const;
  virtual void setSuccessorV(unsigned idx, BasicBlock *B);
};

//===----------------------------------------------------------------------===//
//                               SwitchInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// SwitchInst - Multiway switch
///
class SwitchInst : public TerminatorInst {
  unsigned ReservedSpace;
  // Operand[0]    = Value to switch on
  // Operand[1]    = Default basic block destination
  // Operand[2n  ] = Value to match
  // Operand[2n+1] = BasicBlock to go to on match
  SwitchInst(const SwitchInst &RI);
  void init(Value *Value, BasicBlock *Default, unsigned NumCases);
  void resizeOperands(unsigned No);
public:
  /// SwitchInst ctor - Create a new switch instruction, specifying a value to
  /// switch on and a default destination.  The number of additional cases can
  /// be specified here to make memory allocation more efficient.  This
  /// constructor can also autoinsert before another instruction.
  SwitchInst(Value *Value, BasicBlock *Default, unsigned NumCases,
             Instruction *InsertBefore = 0);
  
  /// SwitchInst ctor - Create a new switch instruction, specifying a value to
  /// switch on and a default destination.  The number of additional cases can
  /// be specified here to make memory allocation more efficient.  This
  /// constructor also autoinserts at the end of the specified BasicBlock.
  SwitchInst(Value *Value, BasicBlock *Default, unsigned NumCases,
             BasicBlock *InsertAtEnd);
  ~SwitchInst();


  // Accessor Methods for Switch stmt
  inline Value *getCondition() const { return getOperand(0); }
  void setCondition(Value *V) { setOperand(0, V); }

  inline BasicBlock *getDefaultDest() const {
    return cast<BasicBlock>(getOperand(1));
  }

  /// getNumCases - return the number of 'cases' in this switch instruction.
  /// Note that case #0 is always the default case.
  unsigned getNumCases() const {
    return getNumOperands()/2;
  }

  /// getCaseValue - Return the specified case value.  Note that case #0, the
  /// default destination, does not have a case value.
  ConstantInt *getCaseValue(unsigned i) {
    assert(i && i < getNumCases() && "Illegal case value to get!");
    return getSuccessorValue(i);
  }

  /// getCaseValue - Return the specified case value.  Note that case #0, the
  /// default destination, does not have a case value.
  const ConstantInt *getCaseValue(unsigned i) const {
    assert(i && i < getNumCases() && "Illegal case value to get!");
    return getSuccessorValue(i);
  }

  /// findCaseValue - Search all of the case values for the specified constant.
  /// If it is explicitly handled, return the case number of it, otherwise
  /// return 0 to indicate that it is handled by the default handler.
  unsigned findCaseValue(const ConstantInt *C) const {
    for (unsigned i = 1, e = getNumCases(); i != e; ++i)
      if (getCaseValue(i) == C)
        return i;
    return 0;
  }

  /// findCaseDest - Finds the unique case value for a given successor. Returns
  /// null if the successor is not found, not unique, or is the default case.
  ConstantInt *findCaseDest(BasicBlock *BB) {
    if (BB == getDefaultDest()) return NULL;

    ConstantInt *CI = NULL;
    for (unsigned i = 1, e = getNumCases(); i != e; ++i) {
      if (getSuccessor(i) == BB) {
        if (CI) return NULL;   // Multiple cases lead to BB.
        else CI = getCaseValue(i);
      }
    }
    return CI;
  }

  /// addCase - Add an entry to the switch instruction...
  ///
  void addCase(ConstantInt *OnVal, BasicBlock *Dest);

  /// removeCase - This method removes the specified successor from the switch
  /// instruction.  Note that this cannot be used to remove the default
  /// destination (successor #0).
  ///
  void removeCase(unsigned idx);

  virtual SwitchInst *clone() const;

  unsigned getNumSuccessors() const { return getNumOperands()/2; }
  BasicBlock *getSuccessor(unsigned idx) const {
    assert(idx < getNumSuccessors() &&"Successor idx out of range for switch!");
    return cast<BasicBlock>(getOperand(idx*2+1));
  }
  void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(idx < getNumSuccessors() && "Successor # out of range for switch!");
    setOperand(idx*2+1, reinterpret_cast<Value*>(NewSucc));
  }

  // getSuccessorValue - Return the value associated with the specified
  // successor.
  inline ConstantInt *getSuccessorValue(unsigned idx) const {
    assert(idx < getNumSuccessors() && "Successor # out of range!");
    return reinterpret_cast<ConstantInt*>(getOperand(idx*2));
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SwitchInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Switch;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
private:
  virtual BasicBlock *getSuccessorV(unsigned idx) const;
  virtual unsigned getNumSuccessorsV() const;
  virtual void setSuccessorV(unsigned idx, BasicBlock *B);
};

//===----------------------------------------------------------------------===//
//                               InvokeInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------

/// InvokeInst - Invoke instruction.  The SubclassData field is used to hold the
/// calling convention of the call.
///
class InvokeInst : public TerminatorInst {
  ParamAttrsList *ParamAttrs;
  InvokeInst(const InvokeInst &BI);
  void init(Value *Fn, BasicBlock *IfNormal, BasicBlock *IfException,
            Value* const *Args, unsigned NumArgs);

  template<typename InputIterator>
  void init(Value *Func, BasicBlock *IfNormal, BasicBlock *IfException,
            InputIterator ArgBegin, InputIterator ArgEnd,
            const std::string &Name,
            // This argument ensures that we have an iterator we can
            // do arithmetic on in constant time
            std::random_access_iterator_tag) {
    unsigned NumArgs = (unsigned)std::distance(ArgBegin, ArgEnd);
    
    // This requires that the iterator points to contiguous memory.
    init(Func, IfNormal, IfException, NumArgs ? &*ArgBegin : 0, NumArgs);
    setName(Name);
  }

public:
  /// Construct an InvokeInst given a range of arguments.
  /// InputIterator must be a random-access iterator pointing to
  /// contiguous storage (e.g. a std::vector<>::iterator).  Checks are
  /// made for random-accessness but not for contiguous storage as
  /// that would incur runtime overhead.
  ///
  /// @brief Construct an InvokeInst from a range of arguments
  template<typename InputIterator>
  InvokeInst(Value *Func, BasicBlock *IfNormal, BasicBlock *IfException,
             InputIterator ArgBegin, InputIterator ArgEnd,
             const std::string &Name = "", Instruction *InsertBefore = 0)
      : TerminatorInst(cast<FunctionType>(cast<PointerType>(Func->getType())
                                          ->getElementType())->getReturnType(),
                       Instruction::Invoke, 0, 0, InsertBefore) {
    init(Func, IfNormal, IfException, ArgBegin, ArgEnd, Name,
         typename std::iterator_traits<InputIterator>::iterator_category());
  }

  /// Construct an InvokeInst given a range of arguments.
  /// InputIterator must be a random-access iterator pointing to
  /// contiguous storage (e.g. a std::vector<>::iterator).  Checks are
  /// made for random-accessness but not for contiguous storage as
  /// that would incur runtime overhead.
  ///
  /// @brief Construct an InvokeInst from a range of arguments
  template<typename InputIterator>
  InvokeInst(Value *Func, BasicBlock *IfNormal, BasicBlock *IfException,
             InputIterator ArgBegin, InputIterator ArgEnd,
             const std::string &Name, BasicBlock *InsertAtEnd)
      : TerminatorInst(cast<FunctionType>(cast<PointerType>(Func->getType())
                                          ->getElementType())->getReturnType(),
                       Instruction::Invoke, 0, 0, InsertAtEnd) {
    init(Func, IfNormal, IfException, ArgBegin, ArgEnd, Name,
         typename std::iterator_traits<InputIterator>::iterator_category());
  }

  ~InvokeInst();

  virtual InvokeInst *clone() const;

  /// getCallingConv/setCallingConv - Get or set the calling convention of this
  /// function call.
  unsigned getCallingConv() const { return SubclassData; }
  void setCallingConv(unsigned CC) {
    SubclassData = CC;
  }

  /// Obtains a pointer to the ParamAttrsList object which holds the
  /// parameter attributes information, if any.
  /// @returns 0 if no attributes have been set.
  /// @brief Get the parameter attributes.
  ParamAttrsList *getParamAttrs() const { return ParamAttrs; }

  /// Sets the parameter attributes for this InvokeInst. To construct a 
  /// ParamAttrsList, see ParameterAttributes.h
  /// @brief Set the parameter attributes.
  void setParamAttrs(ParamAttrsList *attrs);

  /// getCalledFunction - Return the function called, or null if this is an
  /// indirect function invocation.
  ///
  Function *getCalledFunction() const {
    return dyn_cast<Function>(getOperand(0));
  }

  // getCalledValue - Get a pointer to a function that is invoked by this inst.
  inline Value *getCalledValue() const { return getOperand(0); }

  // get*Dest - Return the destination basic blocks...
  BasicBlock *getNormalDest() const {
    return cast<BasicBlock>(getOperand(1));
  }
  BasicBlock *getUnwindDest() const {
    return cast<BasicBlock>(getOperand(2));
  }
  void setNormalDest(BasicBlock *B) {
    setOperand(1, reinterpret_cast<Value*>(B));
  }

  void setUnwindDest(BasicBlock *B) {
    setOperand(2, reinterpret_cast<Value*>(B));
  }

  inline BasicBlock *getSuccessor(unsigned i) const {
    assert(i < 2 && "Successor # out of range for invoke!");
    return i == 0 ? getNormalDest() : getUnwindDest();
  }

  void setSuccessor(unsigned idx, BasicBlock *NewSucc) {
    assert(idx < 2 && "Successor # out of range for invoke!");
    setOperand(idx+1, reinterpret_cast<Value*>(NewSucc));
  }

  unsigned getNumSuccessors() const { return 2; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const InvokeInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::Invoke);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
private:
  virtual BasicBlock *getSuccessorV(unsigned idx) const;
  virtual unsigned getNumSuccessorsV() const;
  virtual void setSuccessorV(unsigned idx, BasicBlock *B);
};


//===----------------------------------------------------------------------===//
//                              UnwindInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// UnwindInst - Immediately exit the current function, unwinding the stack
/// until an invoke instruction is found.
///
class UnwindInst : public TerminatorInst {
public:
  explicit UnwindInst(Instruction *InsertBefore = 0);
  explicit UnwindInst(BasicBlock *InsertAtEnd);

  virtual UnwindInst *clone() const;

  unsigned getNumSuccessors() const { return 0; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const UnwindInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Unwind;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
private:
  virtual BasicBlock *getSuccessorV(unsigned idx) const;
  virtual unsigned getNumSuccessorsV() const;
  virtual void setSuccessorV(unsigned idx, BasicBlock *B);
};

//===----------------------------------------------------------------------===//
//                           UnreachableInst Class
//===----------------------------------------------------------------------===//

//===---------------------------------------------------------------------------
/// UnreachableInst - This function has undefined behavior.  In particular, the
/// presence of this instruction indicates some higher level knowledge that the
/// end of the block cannot be reached.
///
class UnreachableInst : public TerminatorInst {
public:
  explicit UnreachableInst(Instruction *InsertBefore = 0);
  explicit UnreachableInst(BasicBlock *InsertAtEnd);

  virtual UnreachableInst *clone() const;

  unsigned getNumSuccessors() const { return 0; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const UnreachableInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Unreachable;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
private:
  virtual BasicBlock *getSuccessorV(unsigned idx) const;
  virtual unsigned getNumSuccessorsV() const;
  virtual void setSuccessorV(unsigned idx, BasicBlock *B);
};

//===----------------------------------------------------------------------===//
//                                 TruncInst Class
//===----------------------------------------------------------------------===//

/// @brief This class represents a truncation of integer types.
class TruncInst : public CastInst {
  /// Private copy constructor
  TruncInst(const TruncInst &CI)
    : CastInst(CI.getType(), Trunc, CI.getOperand(0)) {
  }
public:
  /// @brief Constructor with insert-before-instruction semantics
  TruncInst(
    Value *S,                     ///< The value to be truncated
    const Type *Ty,               ///< The (smaller) type to truncate to
    const std::string &Name = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  TruncInst(
    Value *S,                     ///< The value to be truncated
    const Type *Ty,               ///< The (smaller) type to truncate to
    const std::string &Name,      ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Clone an identical TruncInst
  virtual CastInst *clone() const;

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const TruncInst *) { return true; }
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

/// @brief This class represents zero extension of integer types.
class ZExtInst : public CastInst {
  /// @brief Private copy constructor
  ZExtInst(const ZExtInst &CI)
    : CastInst(CI.getType(), ZExt, CI.getOperand(0)) {
  }
public:
  /// @brief Constructor with insert-before-instruction semantics
  ZExtInst(
    Value *S,                     ///< The value to be zero extended
    const Type *Ty,               ///< The type to zero extend to
    const std::string &Name = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end semantics.
  ZExtInst(
    Value *S,                     ///< The value to be zero extended
    const Type *Ty,               ///< The type to zero extend to
    const std::string &Name,      ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Clone an identical ZExtInst
  virtual CastInst *clone() const;

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ZExtInst *) { return true; }
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

/// @brief This class represents a sign extension of integer types.
class SExtInst : public CastInst {
  /// @brief Private copy constructor
  SExtInst(const SExtInst &CI)
    : CastInst(CI.getType(), SExt, CI.getOperand(0)) {
  }
public:
  /// @brief Constructor with insert-before-instruction semantics
  SExtInst(
    Value *S,                     ///< The value to be sign extended
    const Type *Ty,               ///< The type to sign extend to
    const std::string &Name = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  SExtInst(
    Value *S,                     ///< The value to be sign extended
    const Type *Ty,               ///< The type to sign extend to
    const std::string &Name,      ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Clone an identical SExtInst
  virtual CastInst *clone() const;

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SExtInst *) { return true; }
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

/// @brief This class represents a truncation of floating point types.
class FPTruncInst : public CastInst {
  FPTruncInst(const FPTruncInst &CI)
    : CastInst(CI.getType(), FPTrunc, CI.getOperand(0)) {
  }
public:
  /// @brief Constructor with insert-before-instruction semantics
  FPTruncInst(
    Value *S,                     ///< The value to be truncated
    const Type *Ty,               ///< The type to truncate to
    const std::string &Name = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-before-instruction semantics
  FPTruncInst(
    Value *S,                     ///< The value to be truncated
    const Type *Ty,               ///< The type to truncate to
    const std::string &Name,      ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Clone an identical FPTruncInst
  virtual CastInst *clone() const;

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const FPTruncInst *) { return true; }
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

/// @brief This class represents an extension of floating point types.
class FPExtInst : public CastInst {
  FPExtInst(const FPExtInst &CI)
    : CastInst(CI.getType(), FPExt, CI.getOperand(0)) {
  }
public:
  /// @brief Constructor with insert-before-instruction semantics
  FPExtInst(
    Value *S,                     ///< The value to be extended
    const Type *Ty,               ///< The type to extend to
    const std::string &Name = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  FPExtInst(
    Value *S,                     ///< The value to be extended
    const Type *Ty,               ///< The type to extend to
    const std::string &Name,      ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Clone an identical FPExtInst
  virtual CastInst *clone() const;

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const FPExtInst *) { return true; }
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

/// @brief This class represents a cast unsigned integer to floating point.
class UIToFPInst : public CastInst {
  UIToFPInst(const UIToFPInst &CI)
    : CastInst(CI.getType(), UIToFP, CI.getOperand(0)) {
  }
public:
  /// @brief Constructor with insert-before-instruction semantics
  UIToFPInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const std::string &Name = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  UIToFPInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const std::string &Name,      ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Clone an identical UIToFPInst
  virtual CastInst *clone() const;

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const UIToFPInst *) { return true; }
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

/// @brief This class represents a cast from signed integer to floating point.
class SIToFPInst : public CastInst {
  SIToFPInst(const SIToFPInst &CI)
    : CastInst(CI.getType(), SIToFP, CI.getOperand(0)) {
  }
public:
  /// @brief Constructor with insert-before-instruction semantics
  SIToFPInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const std::string &Name = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  SIToFPInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const std::string &Name,      ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Clone an identical SIToFPInst
  virtual CastInst *clone() const;

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SIToFPInst *) { return true; }
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

/// @brief This class represents a cast from floating point to unsigned integer
class FPToUIInst  : public CastInst {
  FPToUIInst(const FPToUIInst &CI)
    : CastInst(CI.getType(), FPToUI, CI.getOperand(0)) {
  }
public:
  /// @brief Constructor with insert-before-instruction semantics
  FPToUIInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const std::string &Name = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  FPToUIInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const std::string &Name,      ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< Where to insert the new instruction
  );

  /// @brief Clone an identical FPToUIInst
  virtual CastInst *clone() const;

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const FPToUIInst *) { return true; }
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

/// @brief This class represents a cast from floating point to signed integer.
class FPToSIInst  : public CastInst {
  FPToSIInst(const FPToSIInst &CI)
    : CastInst(CI.getType(), FPToSI, CI.getOperand(0)) {
  }
public:
  /// @brief Constructor with insert-before-instruction semantics
  FPToSIInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const std::string &Name = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  FPToSIInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const std::string &Name,      ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Clone an identical FPToSIInst
  virtual CastInst *clone() const;

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const FPToSIInst *) { return true; }
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

/// @brief This class represents a cast from an integer to a pointer.
class IntToPtrInst : public CastInst {
  IntToPtrInst(const IntToPtrInst &CI)
    : CastInst(CI.getType(), IntToPtr, CI.getOperand(0)) {
  }
public:
  /// @brief Constructor with insert-before-instruction semantics
  IntToPtrInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const std::string &Name = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  IntToPtrInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const std::string &Name,      ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Clone an identical IntToPtrInst
  virtual CastInst *clone() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const IntToPtrInst *) { return true; }
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

/// @brief This class represents a cast from a pointer to an integer
class PtrToIntInst : public CastInst {
  PtrToIntInst(const PtrToIntInst &CI)
    : CastInst(CI.getType(), PtrToInt, CI.getOperand(0)) {
  }
public:
  /// @brief Constructor with insert-before-instruction semantics
  PtrToIntInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const std::string &Name = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  PtrToIntInst(
    Value *S,                     ///< The value to be converted
    const Type *Ty,               ///< The type to convert to
    const std::string &Name,      ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Clone an identical PtrToIntInst
  virtual CastInst *clone() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const PtrToIntInst *) { return true; }
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

/// @brief This class represents a no-op cast from one type to another.
class BitCastInst : public CastInst {
  BitCastInst(const BitCastInst &CI)
    : CastInst(CI.getType(), BitCast, CI.getOperand(0)) {
  }
public:
  /// @brief Constructor with insert-before-instruction semantics
  BitCastInst(
    Value *S,                     ///< The value to be casted
    const Type *Ty,               ///< The type to casted to
    const std::string &Name = "", ///< A name for the new instruction
    Instruction *InsertBefore = 0 ///< Where to insert the new instruction
  );

  /// @brief Constructor with insert-at-end-of-block semantics
  BitCastInst(
    Value *S,                     ///< The value to be casted
    const Type *Ty,               ///< The type to casted to
    const std::string &Name,      ///< A name for the new instruction
    BasicBlock *InsertAtEnd       ///< The block to insert the instruction into
  );

  /// @brief Clone an identical BitCastInst
  virtual CastInst *clone() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const BitCastInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == BitCast;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

} // End llvm namespace

#endif
