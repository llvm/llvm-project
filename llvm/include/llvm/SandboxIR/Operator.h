//===- Operator.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_OPERATOR_H
#define LLVM_SANDBOXIR_OPERATOR_H

#include "llvm/IR/Operator.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/User.h"

namespace llvm::sandboxir {

class Operator : public User {
public:
  // The Operator class is intended to be used as a utility, and is never itself
  // instantiated.
  Operator() = delete;
  void *operator new(size_t s) = delete;

  static bool classof(const Instruction *) { return true; }
  static bool classof(const ConstantExpr *) { return true; }
  static bool classof(const Value *From) {
    return llvm::Operator::classof(From->Val);
  }
  bool hasPoisonGeneratingFlags() const {
    return cast<llvm::Operator>(Val)->hasPoisonGeneratingFlags();
  }
};

class OverflowingBinaryOperator : public Operator {
public:
  bool hasNoUnsignedWrap() const {
    return cast<llvm::OverflowingBinaryOperator>(Val)->hasNoUnsignedWrap();
  }
  bool hasNoSignedWrap() const {
    return cast<llvm::OverflowingBinaryOperator>(Val)->hasNoSignedWrap();
  }
  unsigned getNoWrapKind() const {
    return cast<llvm::OverflowingBinaryOperator>(Val)->getNoWrapKind();
  }
  static bool classof(const Instruction *From) {
    return llvm::OverflowingBinaryOperator::classof(
        cast<llvm::Instruction>(From->Val));
  }
  static bool classof(const ConstantExpr *From) {
    return llvm::OverflowingBinaryOperator::classof(
        cast<llvm::ConstantExpr>(From->Val));
  }
  static bool classof(const Value *From) {
    return llvm::OverflowingBinaryOperator::classof(From->Val);
  }
};
} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_OPERATOR_H
