//===- DebugInfoCommon.h - Shared Debug Info Types --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines small types common to multiple DebugInfo translation units
// while transitioning to DebugInfoExprs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEBUGINFOCOMMON_H
#define LLVM_IR_DEBUGINFOCOMMON_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Compiler.h>

#include <cstdint>

namespace llvm {

enum class SignedOrUnsignedConstant { SignedConstant, UnsignedConstant };

/// A lightweight wrapper around an expression operand.
class ExprOperand {
  const uint64_t *Op = nullptr;

public:
  ExprOperand() = default;
  explicit ExprOperand(const uint64_t *Op) : Op(Op) {}

  const uint64_t *get() const { return Op; }

  /// Get the operand code.
  uint64_t getOp() const { return *Op; }

  /// Get an argument to the operand.
  ///
  /// Never returns the operand itself.
  uint64_t getArg(unsigned I) const { return Op[I + 1]; }

  unsigned getNumArgs() const { return getSize() - 1; }

  /// Return the size of the operand.
  ///
  /// Return the number of elements in the operand (1 + args).
  LLVM_ABI unsigned getSize() const;

  /// Append the elements of this operand to \p V.
  void appendToVector(SmallVectorImpl<uint64_t> &V) const {
    V.append(get(), get() + getSize());
  }
};

/// An iterator for expression operands.
class expr_op_iterator { // NOLINT(readability-identifier-naming)
  ExprOperand Op;

public:
  using iterator_category = std::input_iterator_tag;
  using value_type = ExprOperand;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type &;

  using element_iterator = ArrayRef<uint64_t>::iterator;

  expr_op_iterator() = default;
  explicit expr_op_iterator(element_iterator I) : Op(I) {}

  element_iterator getBase() const { return Op.get(); }
  const ExprOperand &operator*() const { return Op; }
  const ExprOperand *operator->() const { return &Op; }

  expr_op_iterator &operator++() {
    increment();
    return *this;
  }
  expr_op_iterator operator++(int) {
    expr_op_iterator T(*this);
    increment();
    return T;
  }

  /// Get the next iterator.
  ///
  /// \a std::next() doesn't work because this is technically an
  /// input_iterator, but it's a perfectly valid operation.  This is an
  /// accessor to provide the same functionality.
  expr_op_iterator getNext() const { return ++expr_op_iterator(*this); }

  bool operator==(const expr_op_iterator &X) const {
    return getBase() == X.getBase();
  }
  bool operator!=(const expr_op_iterator &X) const {
    return getBase() != X.getBase();
  }

private:
  void increment() { Op = ExprOperand(getBase() + Op.getSize()); }
};

// NOLINTNEXTLINE(readability-identifier-naming)
static inline expr_op_iterator expr_op_begin(ArrayRef<uint64_t> Elements) {
  return expr_op_iterator(Elements.begin());
}
// NOLINTNEXTLINE(readability-identifier-naming)
static inline expr_op_iterator expr_op_end(ArrayRef<uint64_t> Elements) {
  return expr_op_iterator(Elements.end());
}
// NOLINTNEXTLINE(readability-identifier-naming)
static inline iterator_range<expr_op_iterator>
expr_ops(ArrayRef<uint64_t> Elements) {
  return {expr_op_begin(Elements), expr_op_end(Elements)};
}

void appendOffsetImpl(SmallVectorImpl<uint64_t> &Ops, int64_t Offset);

} // end namespace llvm

#endif // LLVM_IR_DEBUGINFOCOMMON_H
