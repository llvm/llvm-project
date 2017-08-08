//===- llvm/InstrTypes.h - Important Instruction subclasses -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines various meta classes of instructions that exist in the VM
// representation.  Specific concrete subclasses of these may be found in the
// i*.h files...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_INSTRTYPES_H
#define LLVM_IR_INSTRTYPES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/OperandTraits.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

namespace llvm {

//===----------------------------------------------------------------------===//
//                            TerminatorInst Class
//===----------------------------------------------------------------------===//

/// Subclasses of this class are all able to terminate a basic
/// block. Thus, these are all the flow control type of operations.
///
class TerminatorInst : public Instruction {
protected:
  TerminatorInst(Type *Ty, Instruction::TermOps iType,
                 Use *Ops, unsigned NumOps,
                 Instruction *InsertBefore = nullptr)
    : Instruction(Ty, iType, Ops, NumOps, InsertBefore) {}

  TerminatorInst(Type *Ty, Instruction::TermOps iType,
                 Use *Ops, unsigned NumOps, BasicBlock *InsertAtEnd)
    : Instruction(Ty, iType, Ops, NumOps, InsertAtEnd) {}

public:
  /// Return the number of successors that this terminator has.
  unsigned getNumSuccessors() const;

  /// Return the specified successor.
  BasicBlock *getSuccessor(unsigned idx) const;

  /// Update the specified successor to point at the provided block.
  void setSuccessor(unsigned idx, BasicBlock *B);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const Instruction *I) {
    return I->isTerminator();
  }
  static bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

  // \brief Returns true if this terminator relates to exception handling.
  bool isExceptional() const {
    switch (getOpcode()) {
    case Instruction::CatchSwitch:
    case Instruction::CatchRet:
    case Instruction::CleanupRet:
    case Instruction::Invoke:
    case Instruction::Resume:
      return true;
    default:
      return false;
    }
  }

  //===--------------------------------------------------------------------===//
  // succ_iterator definition
  //===--------------------------------------------------------------------===//

  template <class Term, class BB> // Successor Iterator
  class SuccIterator : public std::iterator<std::random_access_iterator_tag, BB,
                                            int, BB *, BB *> {
    using super =
        std::iterator<std::random_access_iterator_tag, BB, int, BB *, BB *>;

  public:
    using pointer = typename super::pointer;
    using reference = typename super::reference;

  private:
    Term TermInst;
    unsigned idx;
    using Self = SuccIterator<Term, BB>;

    inline bool index_is_valid(unsigned idx) {
      return idx < TermInst->getNumSuccessors();
    }

    /// \brief Proxy object to allow write access in operator[]
    class SuccessorProxy {
      Self it;

    public:
      explicit SuccessorProxy(const Self &it) : it(it) {}

      SuccessorProxy(const SuccessorProxy &) = default;

      SuccessorProxy &operator=(SuccessorProxy r) {
        *this = reference(r);
        return *this;
      }

      SuccessorProxy &operator=(reference r) {
        it.TermInst->setSuccessor(it.idx, r);
        return *this;
      }

      operator reference() const { return *it; }
    };

  public:
    // begin iterator
    explicit inline SuccIterator(Term T) : TermInst(T), idx(0) {}
    // end iterator
    inline SuccIterator(Term T, bool) : TermInst(T) {
      if (TermInst)
        idx = TermInst->getNumSuccessors();
      else
        // Term == NULL happens, if a basic block is not fully constructed and
        // consequently getTerminator() returns NULL. In this case we construct
        // a SuccIterator which describes a basic block that has zero
        // successors.
        // Defining SuccIterator for incomplete and malformed CFGs is especially
        // useful for debugging.
        idx = 0;
    }

    /// This is used to interface between code that wants to
    /// operate on terminator instructions directly.
    unsigned getSuccessorIndex() const { return idx; }

    inline bool operator==(const Self &x) const { return idx == x.idx; }
    inline bool operator!=(const Self &x) const { return !operator==(x); }

    inline reference operator*() const { return TermInst->getSuccessor(idx); }
    inline pointer operator->() const { return operator*(); }

    inline Self &operator++() {
      ++idx;
      return *this;
    } // Preincrement

    inline Self operator++(int) { // Postincrement
      Self tmp = *this;
      ++*this;
      return tmp;
    }

    inline Self &operator--() {
      --idx;
      return *this;
    }                             // Predecrement
    inline Self operator--(int) { // Postdecrement
      Self tmp = *this;
      --*this;
      return tmp;
    }

    inline bool operator<(const Self &x) const {
      assert(TermInst == x.TermInst &&
             "Cannot compare iterators of different blocks!");
      return idx < x.idx;
    }

    inline bool operator<=(const Self &x) const {
      assert(TermInst == x.TermInst &&
             "Cannot compare iterators of different blocks!");
      return idx <= x.idx;
    }
    inline bool operator>=(const Self &x) const {
      assert(TermInst == x.TermInst &&
             "Cannot compare iterators of different blocks!");
      return idx >= x.idx;
    }

    inline bool operator>(const Self &x) const {
      assert(TermInst == x.TermInst &&
             "Cannot compare iterators of different blocks!");
      return idx > x.idx;
    }

    inline Self &operator+=(int Right) {
      unsigned new_idx = idx + Right;
      assert(index_is_valid(new_idx) && "Iterator index out of bound");
      idx = new_idx;
      return *this;
    }

    inline Self operator+(int Right) const {
      Self tmp = *this;
      tmp += Right;
      return tmp;
    }

    inline Self &operator-=(int Right) { return operator+=(-Right); }

    inline Self operator-(int Right) const { return operator+(-Right); }

    inline int operator-(const Self &x) const {
      assert(TermInst == x.TermInst &&
             "Cannot work on iterators of different blocks!");
      int distance = idx - x.idx;
      return distance;
    }

    inline SuccessorProxy operator[](int offset) {
      Self tmp = *this;
      tmp += offset;
      return SuccessorProxy(tmp);
    }

    /// Get the source BB of this iterator.
    inline BB *getSource() {
      assert(TermInst && "Source not available, if basic block was malformed");
      return TermInst->getParent();
    }
  };

  using succ_iterator = SuccIterator<TerminatorInst *, BasicBlock>;
  using succ_const_iterator =
      SuccIterator<const TerminatorInst *, const BasicBlock>;
  using succ_range = iterator_range<succ_iterator>;
  using succ_const_range = iterator_range<succ_const_iterator>;

private:
  inline succ_iterator succ_begin() { return succ_iterator(this); }
  inline succ_const_iterator succ_begin() const {
    return succ_const_iterator(this);
  }
  inline succ_iterator succ_end() { return succ_iterator(this, true); }
  inline succ_const_iterator succ_end() const {
    return succ_const_iterator(this, true);
  }

public:
  inline succ_range successors() {
    return succ_range(succ_begin(), succ_end());
  }
  inline succ_const_range successors() const {
    return succ_const_range(succ_begin(), succ_end());
  }
};

//===----------------------------------------------------------------------===//
//                          UnaryInstruction Class
//===----------------------------------------------------------------------===//

class UnaryInstruction : public Instruction {
protected:
  UnaryInstruction(Type *Ty, unsigned iType, Value *V,
                   Instruction *IB = nullptr)
    : Instruction(Ty, iType, &Op<0>(), 1, IB) {
    Op<0>() = V;
  }
  UnaryInstruction(Type *Ty, unsigned iType, Value *V, BasicBlock *IAE)
    : Instruction(Ty, iType, &Op<0>(), 1, IAE) {
    Op<0>() = V;
  }

public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 1);
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Alloca ||
           I->getOpcode() == Instruction::Load ||
           I->getOpcode() == Instruction::VAArg ||
           I->getOpcode() == Instruction::ExtractValue ||
           (I->getOpcode() >= CastOpsBegin && I->getOpcode() < CastOpsEnd);
  }
  static bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<UnaryInstruction> :
  public FixedNumOperandTraits<UnaryInstruction, 1> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(UnaryInstruction, Value)

//===----------------------------------------------------------------------===//
//                           BinaryOperator Class
//===----------------------------------------------------------------------===//

class BinaryOperator : public Instruction {
  void AssertOK();

protected:
  BinaryOperator(BinaryOps iType, Value *S1, Value *S2, Type *Ty,
                 const Twine &Name, Instruction *InsertBefore);
  BinaryOperator(BinaryOps iType, Value *S1, Value *S2, Type *Ty,
                 const Twine &Name, BasicBlock *InsertAtEnd);

  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;

  BinaryOperator *cloneImpl() const;

public:
  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// Construct a binary instruction, given the opcode and the two
  /// operands.  Optionally (if InstBefore is specified) insert the instruction
  /// into a BasicBlock right before the specified instruction.  The specified
  /// Instruction is allowed to be a dereferenced end iterator.
  ///
  static BinaryOperator *Create(BinaryOps Op, Value *S1, Value *S2,
                                const Twine &Name = Twine(),
                                Instruction *InsertBefore = nullptr);

  /// Construct a binary instruction, given the opcode and the two
  /// operands.  Also automatically insert this instruction to the end of the
  /// BasicBlock specified.
  ///
  static BinaryOperator *Create(BinaryOps Op, Value *S1, Value *S2,
                                const Twine &Name, BasicBlock *InsertAtEnd);

  /// These methods just forward to Create, and are useful when you
  /// statically know what type of instruction you're going to create.  These
  /// helpers just save some typing.
#define HANDLE_BINARY_INST(N, OPC, CLASS) \
  static BinaryOperator *Create##OPC(Value *V1, Value *V2, \
                                     const Twine &Name = "") {\
    return Create(Instruction::OPC, V1, V2, Name);\
  }
#include "llvm/IR/Instruction.def"
#define HANDLE_BINARY_INST(N, OPC, CLASS) \
  static BinaryOperator *Create##OPC(Value *V1, Value *V2, \
                                     const Twine &Name, BasicBlock *BB) {\
    return Create(Instruction::OPC, V1, V2, Name, BB);\
  }
#include "llvm/IR/Instruction.def"
#define HANDLE_BINARY_INST(N, OPC, CLASS) \
  static BinaryOperator *Create##OPC(Value *V1, Value *V2, \
                                     const Twine &Name, Instruction *I) {\
    return Create(Instruction::OPC, V1, V2, Name, I);\
  }
#include "llvm/IR/Instruction.def"

  static BinaryOperator *CreateWithCopiedFlags(BinaryOps Opc,
                                               Value *V1, Value *V2,
                                               BinaryOperator *CopyBO,
                                               const Twine &Name = "") {
    BinaryOperator *BO = Create(Opc, V1, V2, Name);
    BO->copyIRFlags(CopyBO);
    return BO;
  }

  static BinaryOperator *CreateNSW(BinaryOps Opc, Value *V1, Value *V2,
                                   const Twine &Name = "") {
    BinaryOperator *BO = Create(Opc, V1, V2, Name);
    BO->setHasNoSignedWrap(true);
    return BO;
  }
  static BinaryOperator *CreateNSW(BinaryOps Opc, Value *V1, Value *V2,
                                   const Twine &Name, BasicBlock *BB) {
    BinaryOperator *BO = Create(Opc, V1, V2, Name, BB);
    BO->setHasNoSignedWrap(true);
    return BO;
  }
  static BinaryOperator *CreateNSW(BinaryOps Opc, Value *V1, Value *V2,
                                   const Twine &Name, Instruction *I) {
    BinaryOperator *BO = Create(Opc, V1, V2, Name, I);
    BO->setHasNoSignedWrap(true);
    return BO;
  }

  static BinaryOperator *CreateNUW(BinaryOps Opc, Value *V1, Value *V2,
                                   const Twine &Name = "") {
    BinaryOperator *BO = Create(Opc, V1, V2, Name);
    BO->setHasNoUnsignedWrap(true);
    return BO;
  }
  static BinaryOperator *CreateNUW(BinaryOps Opc, Value *V1, Value *V2,
                                   const Twine &Name, BasicBlock *BB) {
    BinaryOperator *BO = Create(Opc, V1, V2, Name, BB);
    BO->setHasNoUnsignedWrap(true);
    return BO;
  }
  static BinaryOperator *CreateNUW(BinaryOps Opc, Value *V1, Value *V2,
                                   const Twine &Name, Instruction *I) {
    BinaryOperator *BO = Create(Opc, V1, V2, Name, I);
    BO->setHasNoUnsignedWrap(true);
    return BO;
  }

  static BinaryOperator *CreateExact(BinaryOps Opc, Value *V1, Value *V2,
                                     const Twine &Name = "") {
    BinaryOperator *BO = Create(Opc, V1, V2, Name);
    BO->setIsExact(true);
    return BO;
  }
  static BinaryOperator *CreateExact(BinaryOps Opc, Value *V1, Value *V2,
                                     const Twine &Name, BasicBlock *BB) {
    BinaryOperator *BO = Create(Opc, V1, V2, Name, BB);
    BO->setIsExact(true);
    return BO;
  }
  static BinaryOperator *CreateExact(BinaryOps Opc, Value *V1, Value *V2,
                                     const Twine &Name, Instruction *I) {
    BinaryOperator *BO = Create(Opc, V1, V2, Name, I);
    BO->setIsExact(true);
    return BO;
  }

#define DEFINE_HELPERS(OPC, NUWNSWEXACT)                                       \
  static BinaryOperator *Create##NUWNSWEXACT##OPC(Value *V1, Value *V2,        \
                                                  const Twine &Name = "") {    \
    return Create##NUWNSWEXACT(Instruction::OPC, V1, V2, Name);                \
  }                                                                            \
  static BinaryOperator *Create##NUWNSWEXACT##OPC(                             \
      Value *V1, Value *V2, const Twine &Name, BasicBlock *BB) {               \
    return Create##NUWNSWEXACT(Instruction::OPC, V1, V2, Name, BB);            \
  }                                                                            \
  static BinaryOperator *Create##NUWNSWEXACT##OPC(                             \
      Value *V1, Value *V2, const Twine &Name, Instruction *I) {               \
    return Create##NUWNSWEXACT(Instruction::OPC, V1, V2, Name, I);             \
  }

  DEFINE_HELPERS(Add, NSW) // CreateNSWAdd
  DEFINE_HELPERS(Add, NUW) // CreateNUWAdd
  DEFINE_HELPERS(Sub, NSW) // CreateNSWSub
  DEFINE_HELPERS(Sub, NUW) // CreateNUWSub
  DEFINE_HELPERS(Mul, NSW) // CreateNSWMul
  DEFINE_HELPERS(Mul, NUW) // CreateNUWMul
  DEFINE_HELPERS(Shl, NSW) // CreateNSWShl
  DEFINE_HELPERS(Shl, NUW) // CreateNUWShl

  DEFINE_HELPERS(SDiv, Exact)  // CreateExactSDiv
  DEFINE_HELPERS(UDiv, Exact)  // CreateExactUDiv
  DEFINE_HELPERS(AShr, Exact)  // CreateExactAShr
  DEFINE_HELPERS(LShr, Exact)  // CreateExactLShr

#undef DEFINE_HELPERS

  /// Helper functions to construct and inspect unary operations (NEG and NOT)
  /// via binary operators SUB and XOR:
  ///
  /// Create the NEG and NOT instructions out of SUB and XOR instructions.
  ///
  static BinaryOperator *CreateNeg(Value *Op, const Twine &Name = "",
                                   Instruction *InsertBefore = nullptr);
  static BinaryOperator *CreateNeg(Value *Op, const Twine &Name,
                                   BasicBlock *InsertAtEnd);
  static BinaryOperator *CreateNSWNeg(Value *Op, const Twine &Name = "",
                                      Instruction *InsertBefore = nullptr);
  static BinaryOperator *CreateNSWNeg(Value *Op, const Twine &Name,
                                      BasicBlock *InsertAtEnd);
  static BinaryOperator *CreateNUWNeg(Value *Op, const Twine &Name = "",
                                      Instruction *InsertBefore = nullptr);
  static BinaryOperator *CreateNUWNeg(Value *Op, const Twine &Name,
                                      BasicBlock *InsertAtEnd);
  static BinaryOperator *CreateFNeg(Value *Op, const Twine &Name = "",
                                    Instruction *InsertBefore = nullptr);
  static BinaryOperator *CreateFNeg(Value *Op, const Twine &Name,
                                    BasicBlock *InsertAtEnd);
  static BinaryOperator *CreateNot(Value *Op, const Twine &Name = "",
                                   Instruction *InsertBefore = nullptr);
  static BinaryOperator *CreateNot(Value *Op, const Twine &Name,
                                   BasicBlock *InsertAtEnd);

  /// Check if the given Value is a NEG, FNeg, or NOT instruction.
  ///
  static bool isNeg(const Value *V);
  static bool isFNeg(const Value *V, bool IgnoreZeroSign=false);
  static bool isNot(const Value *V);

  /// Helper functions to extract the unary argument of a NEG, FNEG or NOT
  /// operation implemented via Sub, FSub, or Xor.
  ///
  static const Value *getNegArgument(const Value *BinOp);
  static       Value *getNegArgument(      Value *BinOp);
  static const Value *getFNegArgument(const Value *BinOp);
  static       Value *getFNegArgument(      Value *BinOp);
  static const Value *getNotArgument(const Value *BinOp);
  static       Value *getNotArgument(      Value *BinOp);

  BinaryOps getOpcode() const {
    return static_cast<BinaryOps>(Instruction::getOpcode());
  }

  /// Exchange the two operands to this instruction.
  /// This instruction is safe to use on any binary instruction and
  /// does not modify the semantics of the instruction.  If the instruction
  /// cannot be reversed (ie, it's a Div), then return true.
  ///
  bool swapOperands();

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const Instruction *I) {
    return I->isBinaryOp();
  }
  static bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<BinaryOperator> :
  public FixedNumOperandTraits<BinaryOperator, 2> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(BinaryOperator, Value)

//===----------------------------------------------------------------------===//
//                               CastInst Class
//===----------------------------------------------------------------------===//

/// This is the base class for all instructions that perform data
/// casts. It is simply provided so that instruction category testing
/// can be performed with code like:
///
/// if (isa<CastInst>(Instr)) { ... }
/// @brief Base class of casting instructions.
class CastInst : public UnaryInstruction {
protected:
  /// @brief Constructor with insert-before-instruction semantics for subclasses
  CastInst(Type *Ty, unsigned iType, Value *S,
           const Twine &NameStr = "", Instruction *InsertBefore = nullptr)
    : UnaryInstruction(Ty, iType, S, InsertBefore) {
    setName(NameStr);
  }
  /// @brief Constructor with insert-at-end-of-block semantics for subclasses
  CastInst(Type *Ty, unsigned iType, Value *S,
           const Twine &NameStr, BasicBlock *InsertAtEnd)
    : UnaryInstruction(Ty, iType, S, InsertAtEnd) {
    setName(NameStr);
  }

public:
  /// Provides a way to construct any of the CastInst subclasses using an
  /// opcode instead of the subclass's constructor. The opcode must be in the
  /// CastOps category (Instruction::isCast(opcode) returns true). This
  /// constructor has insert-before-instruction semantics to automatically
  /// insert the new CastInst before InsertBefore (if it is non-null).
  /// @brief Construct any of the CastInst subclasses
  static CastInst *Create(
    Instruction::CastOps,    ///< The opcode of the cast instruction
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );
  /// Provides a way to construct any of the CastInst subclasses using an
  /// opcode instead of the subclass's constructor. The opcode must be in the
  /// CastOps category. This constructor has insert-at-end-of-block semantics
  /// to automatically insert the new CastInst at the end of InsertAtEnd (if
  /// its non-null).
  /// @brief Construct any of the CastInst subclasses
  static CastInst *Create(
    Instruction::CastOps,    ///< The opcode for the cast instruction
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which operand is casted
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a ZExt or BitCast cast instruction
  static CastInst *CreateZExtOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );

  /// @brief Create a ZExt or BitCast cast instruction
  static CastInst *CreateZExtOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which operand is casted
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a SExt or BitCast cast instruction
  static CastInst *CreateSExtOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );

  /// @brief Create a SExt or BitCast cast instruction
  static CastInst *CreateSExtOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which operand is casted
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a BitCast AddrSpaceCast, or a PtrToInt cast instruction.
  static CastInst *CreatePointerCast(
    Value *S,                ///< The pointer value to be casted (operand 0)
    Type *Ty,          ///< The type to which operand is casted
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a BitCast, AddrSpaceCast or a PtrToInt cast instruction.
  static CastInst *CreatePointerCast(
    Value *S,                ///< The pointer value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );

  /// @brief Create a BitCast or an AddrSpaceCast cast instruction.
  static CastInst *CreatePointerBitCastOrAddrSpaceCast(
    Value *S,                ///< The pointer value to be casted (operand 0)
    Type *Ty,          ///< The type to which operand is casted
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a BitCast or an AddrSpaceCast cast instruction.
  static CastInst *CreatePointerBitCastOrAddrSpaceCast(
    Value *S,                ///< The pointer value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );

  /// @brief Create a BitCast, a PtrToInt, or an IntToPTr cast instruction.
  ///
  /// If the value is a pointer type and the destination an integer type,
  /// creates a PtrToInt cast. If the value is an integer type and the
  /// destination a pointer type, creates an IntToPtr cast. Otherwise, creates
  /// a bitcast.
  static CastInst *CreateBitOrPointerCast(
    Value *S,                ///< The pointer value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );

  /// @brief Create a ZExt, BitCast, or Trunc for int -> int casts.
  static CastInst *CreateIntegerCast(
    Value *S,                ///< The pointer value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    bool isSigned,           ///< Whether to regard S as signed or not
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );

  /// @brief Create a ZExt, BitCast, or Trunc for int -> int casts.
  static CastInst *CreateIntegerCast(
    Value *S,                ///< The integer value to be casted (operand 0)
    Type *Ty,          ///< The integer type to which operand is casted
    bool isSigned,           ///< Whether to regard S as signed or not
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create an FPExt, BitCast, or FPTrunc for fp -> fp casts
  static CastInst *CreateFPCast(
    Value *S,                ///< The floating point value to be casted
    Type *Ty,          ///< The floating point type to cast to
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );

  /// @brief Create an FPExt, BitCast, or FPTrunc for fp -> fp casts
  static CastInst *CreateFPCast(
    Value *S,                ///< The floating point value to be casted
    Type *Ty,          ///< The floating point type to cast to
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a Trunc or BitCast cast instruction
  static CastInst *CreateTruncOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );

  /// @brief Create a Trunc or BitCast cast instruction
  static CastInst *CreateTruncOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which operand is casted
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Check whether it is valid to call getCastOpcode for these types.
  static bool isCastable(
    Type *SrcTy, ///< The Type from which the value should be cast.
    Type *DestTy ///< The Type to which the value should be cast.
  );

  /// @brief Check whether a bitcast between these types is valid
  static bool isBitCastable(
    Type *SrcTy, ///< The Type from which the value should be cast.
    Type *DestTy ///< The Type to which the value should be cast.
  );

  /// @brief Check whether a bitcast, inttoptr, or ptrtoint cast between these
  /// types is valid and a no-op.
  ///
  /// This ensures that any pointer<->integer cast has enough bits in the
  /// integer and any other cast is a bitcast.
  static bool isBitOrNoopPointerCastable(
      Type *SrcTy,  ///< The Type from which the value should be cast.
      Type *DestTy, ///< The Type to which the value should be cast.
      const DataLayout &DL);

  /// Returns the opcode necessary to cast Val into Ty using usual casting
  /// rules.
  /// @brief Infer the opcode for cast operand and type
  static Instruction::CastOps getCastOpcode(
    const Value *Val, ///< The value to cast
    bool SrcIsSigned, ///< Whether to treat the source as signed
    Type *Ty,   ///< The Type to which the value should be casted
    bool DstIsSigned  ///< Whether to treate the dest. as signed
  );

  /// There are several places where we need to know if a cast instruction
  /// only deals with integer source and destination types. To simplify that
  /// logic, this method is provided.
  /// @returns true iff the cast has only integral typed operand and dest type.
  /// @brief Determine if this is an integer-only cast.
  bool isIntegerCast() const;

  /// A lossless cast is one that does not alter the basic value. It implies
  /// a no-op cast but is more stringent, preventing things like int->float,
  /// long->double, or int->ptr.
  /// @returns true iff the cast is lossless.
  /// @brief Determine if this is a lossless cast.
  bool isLosslessCast() const;

  /// A no-op cast is one that can be effected without changing any bits.
  /// It implies that the source and destination types are the same size. The
  /// IntPtrTy argument is used to make accurate determinations for casts
  /// involving Integer and Pointer types. They are no-op casts if the integer
  /// is the same size as the pointer. However, pointer size varies with
  /// platform. Generally, the result of DataLayout::getIntPtrType() should be
  /// passed in. If that's not available, use Type::Int64Ty, which will make
  /// the isNoopCast call conservative.
  /// @brief Determine if the described cast is a no-op cast.
  static bool isNoopCast(
    Instruction::CastOps Opcode,  ///< Opcode of cast
    Type *SrcTy,   ///< SrcTy of cast
    Type *DstTy,   ///< DstTy of cast
    Type *IntPtrTy ///< Integer type corresponding to Ptr types
  );

  /// @brief Determine if this cast is a no-op cast.
  bool isNoopCast(
    Type *IntPtrTy ///< Integer type corresponding to pointer
  ) const;

  /// @brief Determine if this cast is a no-op cast.
  ///
  /// \param DL is the DataLayout to get the Int Ptr type from.
  bool isNoopCast(const DataLayout &DL) const;

  /// Determine how a pair of casts can be eliminated, if they can be at all.
  /// This is a helper function for both CastInst and ConstantExpr.
  /// @returns 0 if the CastInst pair can't be eliminated, otherwise
  /// returns Instruction::CastOps value for a cast that can replace
  /// the pair, casting SrcTy to DstTy.
  /// @brief Determine if a cast pair is eliminable
  static unsigned isEliminableCastPair(
    Instruction::CastOps firstOpcode,  ///< Opcode of first cast
    Instruction::CastOps secondOpcode, ///< Opcode of second cast
    Type *SrcTy, ///< SrcTy of 1st cast
    Type *MidTy, ///< DstTy of 1st cast & SrcTy of 2nd cast
    Type *DstTy, ///< DstTy of 2nd cast
    Type *SrcIntPtrTy, ///< Integer type corresponding to Ptr SrcTy, or null
    Type *MidIntPtrTy, ///< Integer type corresponding to Ptr MidTy, or null
    Type *DstIntPtrTy  ///< Integer type corresponding to Ptr DstTy, or null
  );

  /// @brief Return the opcode of this CastInst
  Instruction::CastOps getOpcode() const {
    return Instruction::CastOps(Instruction::getOpcode());
  }

  /// @brief Return the source type, as a convenience
  Type* getSrcTy() const { return getOperand(0)->getType(); }
  /// @brief Return the destination type, as a convenience
  Type* getDestTy() const { return getType(); }

  /// This method can be used to determine if a cast from S to DstTy using
  /// Opcode op is valid or not.
  /// @returns true iff the proposed cast is valid.
  /// @brief Determine if a cast is valid without creating one.
  static bool castIsValid(Instruction::CastOps op, Value *S, Type *DstTy);

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const Instruction *I) {
    return I->isCast();
  }
  static bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                               CmpInst Class
//===----------------------------------------------------------------------===//

/// This class is the base class for the comparison instructions.
/// @brief Abstract base class of comparison instructions.
class CmpInst : public Instruction {
public:
  /// This enumeration lists the possible predicates for CmpInst subclasses.
  /// Values in the range 0-31 are reserved for FCmpInst, while values in the
  /// range 32-64 are reserved for ICmpInst. This is necessary to ensure the
  /// predicate values are not overlapping between the classes.
  ///
  /// Some passes (e.g. InstCombine) depend on the bit-wise characteristics of
  /// FCMP_* values. Changing the bit patterns requires a potential change to
  /// those passes.
  enum Predicate {
    // Opcode              U L G E    Intuitive operation
    FCMP_FALSE =  0,  ///< 0 0 0 0    Always false (always folded)
    FCMP_OEQ   =  1,  ///< 0 0 0 1    True if ordered and equal
    FCMP_OGT   =  2,  ///< 0 0 1 0    True if ordered and greater than
    FCMP_OGE   =  3,  ///< 0 0 1 1    True if ordered and greater than or equal
    FCMP_OLT   =  4,  ///< 0 1 0 0    True if ordered and less than
    FCMP_OLE   =  5,  ///< 0 1 0 1    True if ordered and less than or equal
    FCMP_ONE   =  6,  ///< 0 1 1 0    True if ordered and operands are unequal
    FCMP_ORD   =  7,  ///< 0 1 1 1    True if ordered (no nans)
    FCMP_UNO   =  8,  ///< 1 0 0 0    True if unordered: isnan(X) | isnan(Y)
    FCMP_UEQ   =  9,  ///< 1 0 0 1    True if unordered or equal
    FCMP_UGT   = 10,  ///< 1 0 1 0    True if unordered or greater than
    FCMP_UGE   = 11,  ///< 1 0 1 1    True if unordered, greater than, or equal
    FCMP_ULT   = 12,  ///< 1 1 0 0    True if unordered or less than
    FCMP_ULE   = 13,  ///< 1 1 0 1    True if unordered, less than, or equal
    FCMP_UNE   = 14,  ///< 1 1 1 0    True if unordered or not equal
    FCMP_TRUE  = 15,  ///< 1 1 1 1    Always true (always folded)
    FIRST_FCMP_PREDICATE = FCMP_FALSE,
    LAST_FCMP_PREDICATE = FCMP_TRUE,
    BAD_FCMP_PREDICATE = FCMP_TRUE + 1,
    ICMP_EQ    = 32,  ///< equal
    ICMP_NE    = 33,  ///< not equal
    ICMP_UGT   = 34,  ///< unsigned greater than
    ICMP_UGE   = 35,  ///< unsigned greater or equal
    ICMP_ULT   = 36,  ///< unsigned less than
    ICMP_ULE   = 37,  ///< unsigned less or equal
    ICMP_SGT   = 38,  ///< signed greater than
    ICMP_SGE   = 39,  ///< signed greater or equal
    ICMP_SLT   = 40,  ///< signed less than
    ICMP_SLE   = 41,  ///< signed less or equal
    FIRST_ICMP_PREDICATE = ICMP_EQ,
    LAST_ICMP_PREDICATE = ICMP_SLE,
    BAD_ICMP_PREDICATE = ICMP_SLE + 1
  };

protected:
  CmpInst(Type *ty, Instruction::OtherOps op, Predicate pred,
          Value *LHS, Value *RHS, const Twine &Name = "",
          Instruction *InsertBefore = nullptr);

  CmpInst(Type *ty, Instruction::OtherOps op, Predicate pred,
          Value *LHS, Value *RHS, const Twine &Name,
          BasicBlock *InsertAtEnd);

public:
  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }

  /// Construct a compare instruction, given the opcode, the predicate and
  /// the two operands.  Optionally (if InstBefore is specified) insert the
  /// instruction into a BasicBlock right before the specified instruction.
  /// The specified Instruction is allowed to be a dereferenced end iterator.
  /// @brief Create a CmpInst
  static CmpInst *Create(OtherOps Op,
                         Predicate predicate, Value *S1,
                         Value *S2, const Twine &Name = "",
                         Instruction *InsertBefore = nullptr);

  /// Construct a compare instruction, given the opcode, the predicate and the
  /// two operands.  Also automatically insert this instruction to the end of
  /// the BasicBlock specified.
  /// @brief Create a CmpInst
  static CmpInst *Create(OtherOps Op, Predicate predicate, Value *S1,
                         Value *S2, const Twine &Name, BasicBlock *InsertAtEnd);

  /// @brief Get the opcode casted to the right type
  OtherOps getOpcode() const {
    return static_cast<OtherOps>(Instruction::getOpcode());
  }

  /// @brief Return the predicate for this instruction.
  Predicate getPredicate() const {
    return Predicate(getSubclassDataFromInstruction());
  }

  /// @brief Set the predicate for this instruction to the specified value.
  void setPredicate(Predicate P) { setInstructionSubclassData(P); }

  static bool isFPPredicate(Predicate P) {
    return P >= FIRST_FCMP_PREDICATE && P <= LAST_FCMP_PREDICATE;
  }

  static bool isIntPredicate(Predicate P) {
    return P >= FIRST_ICMP_PREDICATE && P <= LAST_ICMP_PREDICATE;
  }

  static StringRef getPredicateName(Predicate P);

  bool isFPPredicate() const { return isFPPredicate(getPredicate()); }
  bool isIntPredicate() const { return isIntPredicate(getPredicate()); }

  /// For example, EQ -> NE, UGT -> ULE, SLT -> SGE,
  ///              OEQ -> UNE, UGT -> OLE, OLT -> UGE, etc.
  /// @returns the inverse predicate for the instruction's current predicate.
  /// @brief Return the inverse of the instruction's predicate.
  Predicate getInversePredicate() const {
    return getInversePredicate(getPredicate());
  }

  /// For example, EQ -> NE, UGT -> ULE, SLT -> SGE,
  ///              OEQ -> UNE, UGT -> OLE, OLT -> UGE, etc.
  /// @returns the inverse predicate for predicate provided in \p pred.
  /// @brief Return the inverse of a given predicate
  static Predicate getInversePredicate(Predicate pred);

  /// For example, EQ->EQ, SLE->SGE, ULT->UGT,
  ///              OEQ->OEQ, ULE->UGE, OLT->OGT, etc.
  /// @returns the predicate that would be the result of exchanging the two
  /// operands of the CmpInst instruction without changing the result
  /// produced.
  /// @brief Return the predicate as if the operands were swapped
  Predicate getSwappedPredicate() const {
    return getSwappedPredicate(getPredicate());
  }

  /// This is a static version that you can use without an instruction
  /// available.
  /// @brief Return the predicate as if the operands were swapped.
  static Predicate getSwappedPredicate(Predicate pred);

  /// @brief Provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// This is just a convenience that dispatches to the subclasses.
  /// @brief Swap the operands and adjust predicate accordingly to retain
  /// the same comparison.
  void swapOperands();

  /// This is just a convenience that dispatches to the subclasses.
  /// @brief Determine if this CmpInst is commutative.
  bool isCommutative() const;

  /// This is just a convenience that dispatches to the subclasses.
  /// @brief Determine if this is an equals/not equals predicate.
  bool isEquality() const;

  /// @returns true if the comparison is signed, false otherwise.
  /// @brief Determine if this instruction is using a signed comparison.
  bool isSigned() const {
    return isSigned(getPredicate());
  }

  /// @returns true if the comparison is unsigned, false otherwise.
  /// @brief Determine if this instruction is using an unsigned comparison.
  bool isUnsigned() const {
    return isUnsigned(getPredicate());
  }

  /// For example, ULT->SLT, ULE->SLE, UGT->SGT, UGE->SGE, SLT->Failed assert
  /// @returns the signed version of the unsigned predicate pred.
  /// @brief return the signed version of a predicate
  static Predicate getSignedPredicate(Predicate pred);

  /// For example, ULT->SLT, ULE->SLE, UGT->SGT, UGE->SGE, SLT->Failed assert
  /// @returns the signed version of the predicate for this instruction (which
  /// has to be an unsigned predicate).
  /// @brief return the signed version of a predicate
  Predicate getSignedPredicate() {
    return getSignedPredicate(getPredicate());
  }

  /// This is just a convenience.
  /// @brief Determine if this is true when both operands are the same.
  bool isTrueWhenEqual() const {
    return isTrueWhenEqual(getPredicate());
  }

  /// This is just a convenience.
  /// @brief Determine if this is false when both operands are the same.
  bool isFalseWhenEqual() const {
    return isFalseWhenEqual(getPredicate());
  }

  /// @returns true if the predicate is unsigned, false otherwise.
  /// @brief Determine if the predicate is an unsigned operation.
  static bool isUnsigned(Predicate predicate);

  /// @returns true if the predicate is signed, false otherwise.
  /// @brief Determine if the predicate is an signed operation.
  static bool isSigned(Predicate predicate);

  /// @brief Determine if the predicate is an ordered operation.
  static bool isOrdered(Predicate predicate);

  /// @brief Determine if the predicate is an unordered operation.
  static bool isUnordered(Predicate predicate);

  /// Determine if the predicate is true when comparing a value with itself.
  static bool isTrueWhenEqual(Predicate predicate);

  /// Determine if the predicate is false when comparing a value with itself.
  static bool isFalseWhenEqual(Predicate predicate);

  /// Determine if Pred1 implies Pred2 is true when two compares have matching
  /// operands.
  static bool isImpliedTrueByMatchingCmp(Predicate Pred1, Predicate Pred2);

  /// Determine if Pred1 implies Pred2 is false when two compares have matching
  /// operands.
  static bool isImpliedFalseByMatchingCmp(Predicate Pred1, Predicate Pred2);

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::ICmp ||
           I->getOpcode() == Instruction::FCmp;
  }
  static bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

  /// @brief Create a result type for fcmp/icmp
  static Type* makeCmpResultType(Type* opnd_type) {
    if (VectorType* vt = dyn_cast<VectorType>(opnd_type)) {
      return VectorType::get(Type::getInt1Ty(opnd_type->getContext()),
                             vt->getNumElements());
    }
    return Type::getInt1Ty(opnd_type->getContext());
  }

private:
  // Shadow Value::setValueSubclassData with a private forwarding method so that
  // subclasses cannot accidentally use it.
  void setValueSubclassData(unsigned short D) {
    Value::setValueSubclassData(D);
  }
};

// FIXME: these are redundant if CmpInst < BinaryOperator
template <>
struct OperandTraits<CmpInst> : public FixedNumOperandTraits<CmpInst, 2> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(CmpInst, Value)

//===----------------------------------------------------------------------===//
//                           FuncletPadInst Class
//===----------------------------------------------------------------------===//
class FuncletPadInst : public Instruction {
private:
  FuncletPadInst(const FuncletPadInst &CPI);

  explicit FuncletPadInst(Instruction::FuncletPadOps Op, Value *ParentPad,
                          ArrayRef<Value *> Args, unsigned Values,
                          const Twine &NameStr, Instruction *InsertBefore);
  explicit FuncletPadInst(Instruction::FuncletPadOps Op, Value *ParentPad,
                          ArrayRef<Value *> Args, unsigned Values,
                          const Twine &NameStr, BasicBlock *InsertAtEnd);

  void init(Value *ParentPad, ArrayRef<Value *> Args, const Twine &NameStr);

protected:
  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;
  friend class CatchPadInst;
  friend class CleanupPadInst;

  FuncletPadInst *cloneImpl() const;

public:
  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// getNumArgOperands - Return the number of funcletpad arguments.
  ///
  unsigned getNumArgOperands() const { return getNumOperands() - 1; }

  /// Convenience accessors

  /// \brief Return the outer EH-pad this funclet is nested within.
  ///
  /// Note: This returns the associated CatchSwitchInst if this FuncletPadInst
  /// is a CatchPadInst.
  Value *getParentPad() const { return Op<-1>(); }
  void setParentPad(Value *ParentPad) {
    assert(ParentPad);
    Op<-1>() = ParentPad;
  }

  /// getArgOperand/setArgOperand - Return/set the i-th funcletpad argument.
  ///
  Value *getArgOperand(unsigned i) const { return getOperand(i); }
  void setArgOperand(unsigned i, Value *v) { setOperand(i, v); }

  /// arg_operands - iteration adapter for range-for loops.
  op_range arg_operands() { return op_range(op_begin(), op_end() - 1); }

  /// arg_operands - iteration adapter for range-for loops.
  const_op_range arg_operands() const {
    return const_op_range(op_begin(), op_end() - 1);
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const Instruction *I) { return I->isFuncletPad(); }
  static bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<FuncletPadInst>
    : public VariadicOperandTraits<FuncletPadInst, /*MINARITY=*/1> {};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(FuncletPadInst, Value)

/// \brief A lightweight accessor for an operand bundle meant to be passed
/// around by value.
struct OperandBundleUse {
  ArrayRef<Use> Inputs;

  OperandBundleUse() = default;
  explicit OperandBundleUse(StringMapEntry<uint32_t> *Tag, ArrayRef<Use> Inputs)
      : Inputs(Inputs), Tag(Tag) {}

  /// \brief Return true if the operand at index \p Idx in this operand bundle
  /// has the attribute A.
  bool operandHasAttr(unsigned Idx, Attribute::AttrKind A) const {
    if (isDeoptOperandBundle())
      if (A == Attribute::ReadOnly || A == Attribute::NoCapture)
        return Inputs[Idx]->getType()->isPointerTy();

    // Conservative answer:  no operands have any attributes.
    return false;
  }

  /// \brief Return the tag of this operand bundle as a string.
  StringRef getTagName() const {
    return Tag->getKey();
  }

  /// \brief Return the tag of this operand bundle as an integer.
  ///
  /// Operand bundle tags are interned by LLVMContextImpl::getOrInsertBundleTag,
  /// and this function returns the unique integer getOrInsertBundleTag
  /// associated the tag of this operand bundle to.
  uint32_t getTagID() const {
    return Tag->getValue();
  }

  /// \brief Return true if this is a "deopt" operand bundle.
  bool isDeoptOperandBundle() const {
    return getTagID() == LLVMContext::OB_deopt;
  }

  /// \brief Return true if this is a "funclet" operand bundle.
  bool isFuncletOperandBundle() const {
    return getTagID() == LLVMContext::OB_funclet;
  }

private:
  /// \brief Pointer to an entry in LLVMContextImpl::getOrInsertBundleTag.
  StringMapEntry<uint32_t> *Tag;
};

/// \brief A container for an operand bundle being viewed as a set of values
/// rather than a set of uses.
///
/// Unlike OperandBundleUse, OperandBundleDefT owns the memory it carries, and
/// so it is possible to create and pass around "self-contained" instances of
/// OperandBundleDef and ConstOperandBundleDef.
template <typename InputTy> class OperandBundleDefT {
  std::string Tag;
  std::vector<InputTy> Inputs;

public:
  explicit OperandBundleDefT(std::string Tag, std::vector<InputTy> Inputs)
      : Tag(std::move(Tag)), Inputs(std::move(Inputs)) {}
  explicit OperandBundleDefT(std::string Tag, ArrayRef<InputTy> Inputs)
      : Tag(std::move(Tag)), Inputs(Inputs) {}

  explicit OperandBundleDefT(const OperandBundleUse &OBU) {
    Tag = OBU.getTagName();
    Inputs.insert(Inputs.end(), OBU.Inputs.begin(), OBU.Inputs.end());
  }

  ArrayRef<InputTy> inputs() const { return Inputs; }

  using input_iterator = typename std::vector<InputTy>::const_iterator;

  size_t input_size() const { return Inputs.size(); }
  input_iterator input_begin() const { return Inputs.begin(); }
  input_iterator input_end() const { return Inputs.end(); }

  StringRef getTag() const { return Tag; }
};

using OperandBundleDef = OperandBundleDefT<Value *>;
using ConstOperandBundleDef = OperandBundleDefT<const Value *>;

/// \brief A mixin to add operand bundle functionality to llvm instruction
/// classes.
///
/// OperandBundleUser uses the descriptor area co-allocated with the host User
/// to store some meta information about which operands are "normal" operands,
/// and which ones belong to some operand bundle.
///
/// The layout of an operand bundle user is
///
///          +-----------uint32_t End-------------------------------------+
///          |                                                            |
///          |  +--------uint32_t Begin--------------------+              |
///          |  |                                          |              |
///          ^  ^                                          v              v
///  |------|------|----|----|----|----|----|---------|----|---------|----|-----
///  | BOI0 | BOI1 | .. | DU | U0 | U1 | .. | BOI0_U0 | .. | BOI1_U0 | .. | Un
///  |------|------|----|----|----|----|----|---------|----|---------|----|-----
///   v  v                                  ^              ^
///   |  |                                  |              |
///   |  +--------uint32_t Begin------------+              |
///   |                                                    |
///   +-----------uint32_t End-----------------------------+
///
///
/// BOI0, BOI1 ... are descriptions of operand bundles in this User's use list.
/// These descriptions are installed and managed by this class, and they're all
/// instances of OperandBundleUser<T>::BundleOpInfo.
///
/// DU is an additional descriptor installed by User's 'operator new' to keep
/// track of the 'BOI0 ... BOIN' co-allocation.  OperandBundleUser does not
/// access or modify DU in any way, it's an implementation detail private to
/// User.
///
/// The regular Use& vector for the User starts at U0.  The operand bundle uses
/// are part of the Use& vector, just like normal uses.  In the diagram above,
/// the operand bundle uses start at BOI0_U0.  Each instance of BundleOpInfo has
/// information about a contiguous set of uses constituting an operand bundle,
/// and the total set of operand bundle uses themselves form a contiguous set of
/// uses (i.e. there are no gaps between uses corresponding to individual
/// operand bundles).
///
/// This class does not know the location of the set of operand bundle uses
/// within the use list -- that is decided by the User using this class via the
/// BeginIdx argument in populateBundleOperandInfos.
///
/// Currently operand bundle users with hung-off operands are not supported.
template <typename InstrTy, typename OpIteratorTy> class OperandBundleUser {
public:
  /// \brief Return the number of operand bundles associated with this User.
  unsigned getNumOperandBundles() const {
    return std::distance(bundle_op_info_begin(), bundle_op_info_end());
  }

  /// \brief Return true if this User has any operand bundles.
  bool hasOperandBundles() const { return getNumOperandBundles() != 0; }

  /// \brief Return the index of the first bundle operand in the Use array.
  unsigned getBundleOperandsStartIndex() const {
    assert(hasOperandBundles() && "Don't call otherwise!");
    return bundle_op_info_begin()->Begin;
  }

  /// \brief Return the index of the last bundle operand in the Use array.
  unsigned getBundleOperandsEndIndex() const {
    assert(hasOperandBundles() && "Don't call otherwise!");
    return bundle_op_info_end()[-1].End;
  }

  /// Return true if the operand at index \p Idx is a bundle operand.
  bool isBundleOperand(unsigned Idx) const {
    return hasOperandBundles() && Idx >= getBundleOperandsStartIndex() &&
           Idx < getBundleOperandsEndIndex();
  }

  /// \brief Return the total number operands (not operand bundles) used by
  /// every operand bundle in this OperandBundleUser.
  unsigned getNumTotalBundleOperands() const {
    if (!hasOperandBundles())
      return 0;

    unsigned Begin = getBundleOperandsStartIndex();
    unsigned End = getBundleOperandsEndIndex();

    assert(Begin <= End && "Should be!");
    return End - Begin;
  }

  /// \brief Return the operand bundle at a specific index.
  OperandBundleUse getOperandBundleAt(unsigned Index) const {
    assert(Index < getNumOperandBundles() && "Index out of bounds!");
    return operandBundleFromBundleOpInfo(*(bundle_op_info_begin() + Index));
  }

  /// \brief Return the number of operand bundles with the tag Name attached to
  /// this instruction.
  unsigned countOperandBundlesOfType(StringRef Name) const {
    unsigned Count = 0;
    for (unsigned i = 0, e = getNumOperandBundles(); i != e; ++i)
      if (getOperandBundleAt(i).getTagName() == Name)
        Count++;

    return Count;
  }

  /// \brief Return the number of operand bundles with the tag ID attached to
  /// this instruction.
  unsigned countOperandBundlesOfType(uint32_t ID) const {
    unsigned Count = 0;
    for (unsigned i = 0, e = getNumOperandBundles(); i != e; ++i)
      if (getOperandBundleAt(i).getTagID() == ID)
        Count++;

    return Count;
  }

  /// \brief Return an operand bundle by name, if present.
  ///
  /// It is an error to call this for operand bundle types that may have
  /// multiple instances of them on the same instruction.
  Optional<OperandBundleUse> getOperandBundle(StringRef Name) const {
    assert(countOperandBundlesOfType(Name) < 2 && "Precondition violated!");

    for (unsigned i = 0, e = getNumOperandBundles(); i != e; ++i) {
      OperandBundleUse U = getOperandBundleAt(i);
      if (U.getTagName() == Name)
        return U;
    }

    return None;
  }

  /// \brief Return an operand bundle by tag ID, if present.
  ///
  /// It is an error to call this for operand bundle types that may have
  /// multiple instances of them on the same instruction.
  Optional<OperandBundleUse> getOperandBundle(uint32_t ID) const {
    assert(countOperandBundlesOfType(ID) < 2 && "Precondition violated!");

    for (unsigned i = 0, e = getNumOperandBundles(); i != e; ++i) {
      OperandBundleUse U = getOperandBundleAt(i);
      if (U.getTagID() == ID)
        return U;
    }

    return None;
  }

  /// \brief Return the list of operand bundles attached to this instruction as
  /// a vector of OperandBundleDefs.
  ///
  /// This function copies the OperandBundeUse instances associated with this
  /// OperandBundleUser to a vector of OperandBundleDefs.  Note:
  /// OperandBundeUses and OperandBundleDefs are non-trivially *different*
  /// representations of operand bundles (see documentation above).
  void getOperandBundlesAsDefs(SmallVectorImpl<OperandBundleDef> &Defs) const {
    for (unsigned i = 0, e = getNumOperandBundles(); i != e; ++i)
      Defs.emplace_back(getOperandBundleAt(i));
  }

  /// \brief Return the operand bundle for the operand at index OpIdx.
  ///
  /// It is an error to call this with an OpIdx that does not correspond to an
  /// bundle operand.
  OperandBundleUse getOperandBundleForOperand(unsigned OpIdx) const {
    return operandBundleFromBundleOpInfo(getBundleOpInfoForOperand(OpIdx));
  }

  /// \brief Return true if this operand bundle user has operand bundles that
  /// may read from the heap.
  bool hasReadingOperandBundles() const {
    // Implementation note: this is a conservative implementation of operand
    // bundle semantics, where *any* operand bundle forces a callsite to be at
    // least readonly.
    return hasOperandBundles();
  }

  /// \brief Return true if this operand bundle user has operand bundles that
  /// may write to the heap.
  bool hasClobberingOperandBundles() const {
    for (auto &BOI : bundle_op_infos()) {
      if (BOI.Tag->second == LLVMContext::OB_deopt ||
          BOI.Tag->second == LLVMContext::OB_funclet)
        continue;

      // This instruction has an operand bundle that is not known to us.
      // Assume the worst.
      return true;
    }

    return false;
  }

  /// \brief Return true if the bundle operand at index \p OpIdx has the
  /// attribute \p A.
  bool bundleOperandHasAttr(unsigned OpIdx,  Attribute::AttrKind A) const {
    auto &BOI = getBundleOpInfoForOperand(OpIdx);
    auto OBU = operandBundleFromBundleOpInfo(BOI);
    return OBU.operandHasAttr(OpIdx - BOI.Begin, A);
  }

  /// \brief Return true if \p Other has the same sequence of operand bundle
  /// tags with the same number of operands on each one of them as this
  /// OperandBundleUser.
  bool hasIdenticalOperandBundleSchema(
      const OperandBundleUser<InstrTy, OpIteratorTy> &Other) const {
    if (getNumOperandBundles() != Other.getNumOperandBundles())
      return false;

    return std::equal(bundle_op_info_begin(), bundle_op_info_end(),
                      Other.bundle_op_info_begin());
  }

  /// \brief Return true if this operand bundle user contains operand bundles
  /// with tags other than those specified in \p IDs.
  bool hasOperandBundlesOtherThan(ArrayRef<uint32_t> IDs) const {
    for (unsigned i = 0, e = getNumOperandBundles(); i != e; ++i) {
      uint32_t ID = getOperandBundleAt(i).getTagID();
      if (!is_contained(IDs, ID))
        return true;
    }
    return false;
  }

protected:
  /// \brief Is the function attribute S disallowed by some operand bundle on
  /// this operand bundle user?
  bool isFnAttrDisallowedByOpBundle(StringRef S) const {
    // Operand bundles only possibly disallow readnone, readonly and argmenonly
    // attributes.  All String attributes are fine.
    return false;
  }

  /// \brief Is the function attribute A disallowed by some operand bundle on
  /// this operand bundle user?
  bool isFnAttrDisallowedByOpBundle(Attribute::AttrKind A) const {
    switch (A) {
    default:
      return false;

    case Attribute::ArgMemOnly:
      return hasReadingOperandBundles();

    case Attribute::ReadNone:
      return hasReadingOperandBundles();

    case Attribute::ReadOnly:
      return hasClobberingOperandBundles();
    }

    llvm_unreachable("switch has a default case!");
  }

  /// \brief Used to keep track of an operand bundle.  See the main comment on
  /// OperandBundleUser above.
  struct BundleOpInfo {
    /// \brief The operand bundle tag, interned by
    /// LLVMContextImpl::getOrInsertBundleTag.
    StringMapEntry<uint32_t> *Tag;

    /// \brief The index in the Use& vector where operands for this operand
    /// bundle starts.
    uint32_t Begin;

    /// \brief The index in the Use& vector where operands for this operand
    /// bundle ends.
    uint32_t End;

    bool operator==(const BundleOpInfo &Other) const {
      return Tag == Other.Tag && Begin == Other.Begin && End == Other.End;
    }
  };

  /// \brief Simple helper function to map a BundleOpInfo to an
  /// OperandBundleUse.
  OperandBundleUse
  operandBundleFromBundleOpInfo(const BundleOpInfo &BOI) const {
    auto op_begin = static_cast<const InstrTy *>(this)->op_begin();
    ArrayRef<Use> Inputs(op_begin + BOI.Begin, op_begin + BOI.End);
    return OperandBundleUse(BOI.Tag, Inputs);
  }

  using bundle_op_iterator = BundleOpInfo *;
  using const_bundle_op_iterator = const BundleOpInfo *;

  /// \brief Return the start of the list of BundleOpInfo instances associated
  /// with this OperandBundleUser.
  bundle_op_iterator bundle_op_info_begin() {
    if (!static_cast<InstrTy *>(this)->hasDescriptor())
      return nullptr;

    uint8_t *BytesBegin = static_cast<InstrTy *>(this)->getDescriptor().begin();
    return reinterpret_cast<bundle_op_iterator>(BytesBegin);
  }

  /// \brief Return the start of the list of BundleOpInfo instances associated
  /// with this OperandBundleUser.
  const_bundle_op_iterator bundle_op_info_begin() const {
    auto *NonConstThis =
        const_cast<OperandBundleUser<InstrTy, OpIteratorTy> *>(this);
    return NonConstThis->bundle_op_info_begin();
  }

  /// \brief Return the end of the list of BundleOpInfo instances associated
  /// with this OperandBundleUser.
  bundle_op_iterator bundle_op_info_end() {
    if (!static_cast<InstrTy *>(this)->hasDescriptor())
      return nullptr;

    uint8_t *BytesEnd = static_cast<InstrTy *>(this)->getDescriptor().end();
    return reinterpret_cast<bundle_op_iterator>(BytesEnd);
  }

  /// \brief Return the end of the list of BundleOpInfo instances associated
  /// with this OperandBundleUser.
  const_bundle_op_iterator bundle_op_info_end() const {
    auto *NonConstThis =
        const_cast<OperandBundleUser<InstrTy, OpIteratorTy> *>(this);
    return NonConstThis->bundle_op_info_end();
  }

  /// \brief Return the range [\p bundle_op_info_begin, \p bundle_op_info_end).
  iterator_range<bundle_op_iterator> bundle_op_infos() {
    return make_range(bundle_op_info_begin(), bundle_op_info_end());
  }

  /// \brief Return the range [\p bundle_op_info_begin, \p bundle_op_info_end).
  iterator_range<const_bundle_op_iterator> bundle_op_infos() const {
    return make_range(bundle_op_info_begin(), bundle_op_info_end());
  }

  /// \brief Populate the BundleOpInfo instances and the Use& vector from \p
  /// Bundles.  Return the op_iterator pointing to the Use& one past the last
  /// last bundle operand use.
  ///
  /// Each \p OperandBundleDef instance is tracked by a OperandBundleInfo
  /// instance allocated in this User's descriptor.
  OpIteratorTy populateBundleOperandInfos(ArrayRef<OperandBundleDef> Bundles,
                                          const unsigned BeginIndex) {
    auto It = static_cast<InstrTy *>(this)->op_begin() + BeginIndex;
    for (auto &B : Bundles)
      It = std::copy(B.input_begin(), B.input_end(), It);

    auto *ContextImpl = static_cast<InstrTy *>(this)->getContext().pImpl;
    auto BI = Bundles.begin();
    unsigned CurrentIndex = BeginIndex;

    for (auto &BOI : bundle_op_infos()) {
      assert(BI != Bundles.end() && "Incorrect allocation?");

      BOI.Tag = ContextImpl->getOrInsertBundleTag(BI->getTag());
      BOI.Begin = CurrentIndex;
      BOI.End = CurrentIndex + BI->input_size();
      CurrentIndex = BOI.End;
      BI++;
    }

    assert(BI == Bundles.end() && "Incorrect allocation?");

    return It;
  }

  /// \brief Return the BundleOpInfo for the operand at index OpIdx.
  ///
  /// It is an error to call this with an OpIdx that does not correspond to an
  /// bundle operand.
  const BundleOpInfo &getBundleOpInfoForOperand(unsigned OpIdx) const {
    for (auto &BOI : bundle_op_infos())
      if (BOI.Begin <= OpIdx && OpIdx < BOI.End)
        return BOI;

    llvm_unreachable("Did not find operand bundle for operand!");
  }

  /// \brief Return the total number of values used in \p Bundles.
  static unsigned CountBundleInputs(ArrayRef<OperandBundleDef> Bundles) {
    unsigned Total = 0;
    for (auto &B : Bundles)
      Total += B.input_size();
    return Total;
  }
};

} // end namespace llvm

#endif // LLVM_IR_INSTRTYPES_H
