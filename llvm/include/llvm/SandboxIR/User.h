//===- User.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_USER_H
#define LLVM_SANDBOXIR_USER_H

#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/SandboxIR/Use.h"
#include "llvm/SandboxIR/Value.h"

namespace llvm::sandboxir {

class Context;

/// Iterator for the `Use` edges of a User's operands.
/// \Returns the operand `Use` when dereferenced.
class OperandUseIterator {
  sandboxir::Use Use;
  /// Don't let the user create a non-empty OperandUseIterator.
  OperandUseIterator(const class Use &Use) : Use(Use) {}
  friend class User;                                  // For constructor
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS; // For constructor
#include "llvm/SandboxIR/SandboxIRValues.def"

public:
  using difference_type = std::ptrdiff_t;
  using value_type = sandboxir::Use;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::input_iterator_tag;

  OperandUseIterator() = default;
  value_type operator*() const;
  OperandUseIterator &operator++();
  OperandUseIterator operator++(int) {
    auto Copy = *this;
    this->operator++();
    return Copy;
  }
  bool operator==(const OperandUseIterator &Other) const {
    return Use == Other.Use;
  }
  bool operator!=(const OperandUseIterator &Other) const {
    return !(*this == Other);
  }
  OperandUseIterator operator+(unsigned Num) const;
  OperandUseIterator operator-(unsigned Num) const;
  int operator-(const OperandUseIterator &Other) const;
};

/// A sandboxir::User has operands.
class User : public Value {
protected:
  User(ClassID ID, llvm::Value *V, Context &Ctx) : Value(ID, V, Ctx) {}

  /// \Returns the Use edge that corresponds to \p OpIdx.
  /// Note: This is the default implementation that works for instructions that
  /// match the underlying LLVM instruction. All others should use a different
  /// implementation.
  Use getOperandUseDefault(unsigned OpIdx, bool Verify) const;
  /// \Returns the Use for the \p OpIdx'th operand. This is virtual to allow
  /// instructions to deviate from the LLVM IR operands, which is a requirement
  /// for sandboxir Instructions that consist of more than one LLVM Instruction.
  virtual Use getOperandUseInternal(unsigned OpIdx, bool Verify) const = 0;
  friend class OperandUseIterator; // for getOperandUseInternal()

  /// The default implementation works only for single-LLVMIR-instruction
  /// Users and only if they match exactly the LLVM instruction.
  unsigned getUseOperandNoDefault(const Use &Use) const {
    return Use.LLVMUse->getOperandNo();
  }
  /// \Returns the operand index of \p Use.
  virtual unsigned getUseOperandNo(const Use &Use) const = 0;
  friend unsigned Use::getOperandNo() const; // For getUseOperandNo()

  void swapOperandsInternal(unsigned OpIdxA, unsigned OpIdxB) {
    assert(OpIdxA < getNumOperands() && "OpIdxA out of bounds!");
    assert(OpIdxB < getNumOperands() && "OpIdxB out of bounds!");
    auto UseA = getOperandUse(OpIdxA);
    auto UseB = getOperandUse(OpIdxB);
    UseA.swap(UseB);
  }

#ifndef NDEBUG
  void verifyUserOfLLVMUse(const llvm::Use &Use) const;
#endif // NDEBUG

public:
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  using op_iterator = OperandUseIterator;
  using const_op_iterator = OperandUseIterator;
  using op_range = iterator_range<op_iterator>;
  using const_op_range = iterator_range<const_op_iterator>;

  virtual op_iterator op_begin() {
    assert(isa<llvm::User>(Val) && "Expect User value!");
    return op_iterator(getOperandUseInternal(0, /*Verify=*/false));
  }
  virtual op_iterator op_end() {
    assert(isa<llvm::User>(Val) && "Expect User value!");
    return op_iterator(
        getOperandUseInternal(getNumOperands(), /*Verify=*/false));
  }
  virtual const_op_iterator op_begin() const {
    return const_cast<User *>(this)->op_begin();
  }
  virtual const_op_iterator op_end() const {
    return const_cast<User *>(this)->op_end();
  }

  op_range operands() { return make_range<op_iterator>(op_begin(), op_end()); }
  const_op_range operands() const {
    return make_range<const_op_iterator>(op_begin(), op_end());
  }
  Value *getOperand(unsigned OpIdx) const { return getOperandUse(OpIdx).get(); }
  /// \Returns the operand edge for \p OpIdx. NOTE: This should also work for
  /// OpIdx == getNumOperands(), which is used for op_end().
  Use getOperandUse(unsigned OpIdx) const {
    return getOperandUseInternal(OpIdx, /*Verify=*/true);
  }
  virtual unsigned getNumOperands() const {
    return isa<llvm::User>(Val) ? cast<llvm::User>(Val)->getNumOperands() : 0;
  }

  virtual void setOperand(unsigned OperandIdx, Value *Operand);
  /// Replaces any operands that match \p FromV with \p ToV. Returns whether any
  /// operands were replaced.
  bool replaceUsesOfWith(Value *FromV, Value *ToV);

#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::User>(Val) && "Expected User!");
  }
  void dumpCommonHeader(raw_ostream &OS) const final;
  void dumpOS(raw_ostream &OS) const override {
    // TODO: Remove this tmp implementation once we get the Instruction classes.
  }
#endif
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_USER_H
