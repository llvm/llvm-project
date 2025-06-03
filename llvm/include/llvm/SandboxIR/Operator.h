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

class FPMathOperator : public Operator {
public:
  bool isFast() const { return cast<llvm::FPMathOperator>(Val)->isFast(); }
  bool hasAllowReassoc() const {
    return cast<llvm::FPMathOperator>(Val)->hasAllowReassoc();
  }
  bool hasNoNaNs() const {
    return cast<llvm::FPMathOperator>(Val)->hasNoNaNs();
  }
  bool hasNoInfs() const {
    return cast<llvm::FPMathOperator>(Val)->hasNoInfs();
  }
  bool hasNoSignedZeros() const {
    return cast<llvm::FPMathOperator>(Val)->hasNoSignedZeros();
  }
  bool hasAllowReciprocal() const {
    return cast<llvm::FPMathOperator>(Val)->hasAllowReciprocal();
  }
  bool hasAllowContract() const {
    return cast<llvm::FPMathOperator>(Val)->hasAllowContract();
  }
  bool hasApproxFunc() const {
    return cast<llvm::FPMathOperator>(Val)->hasApproxFunc();
  }
  FastMathFlags getFastMathFlags() const {
    return cast<llvm::FPMathOperator>(Val)->getFastMathFlags();
  }
  float getFPAccuracy() const {
    return cast<llvm::FPMathOperator>(Val)->getFPAccuracy();
  }
  static bool isSupportedFloatingPointType(Type *Ty) {
    return llvm::FPMathOperator::isSupportedFloatingPointType(Ty->LLVMTy);
  }
  static bool classof(const Value *V) {
    return llvm::FPMathOperator::classof(V->Val);
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_OPERATOR_H
