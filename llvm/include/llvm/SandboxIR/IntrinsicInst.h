//===- IntrinsicInst.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_INTRINSICINST_H
#define LLVM_SANDBOXIR_INTRINSICINST_H

#include "llvm/IR/IntrinsicInst.h"
#include "llvm/SandboxIR/Instruction.h"

namespace llvm::sandboxir {

class IntrinsicInst : public CallInst {
  IntrinsicInst(llvm::IntrinsicInst *I, Context &Ctx) : CallInst(I, Ctx) {}

public:
  Intrinsic::ID getIntrinsicID() const {
    return cast<llvm::IntrinsicInst>(Val)->getIntrinsicID();
  }
  bool isAssociative() const {
    return cast<llvm::IntrinsicInst>(Val)->isAssociative();
  }
  bool isCommutative() const {
    return cast<llvm::IntrinsicInst>(Val)->isCommutative();
  }
  bool isAssumeLikeIntrinsic() const {
    return cast<llvm::IntrinsicInst>(Val)->isAssumeLikeIntrinsic();
  }
  static bool mayLowerToFunctionCall(Intrinsic::ID IID) {
    return llvm::IntrinsicInst::mayLowerToFunctionCall(IID);
  }
  static bool classof(const Value *V) {
    auto *LLVMV = V->Val;
    return isa<llvm::IntrinsicInst>(LLVMV);
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_INTRINSICINST_H
