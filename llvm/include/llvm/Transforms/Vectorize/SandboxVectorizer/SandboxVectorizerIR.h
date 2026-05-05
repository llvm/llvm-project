//===- SandboxVectorizerIR.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines a SandboxIR specialization for the vectorizer.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SANDBOXVECTORIZERIR_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SANDBOXVECTORIZERIR_H

#include "llvm/IR/LLVMContext.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm::sandboxir {

class SBVecContext;

class PackInst final : public Instruction {
  Use getOperandUseInternal(unsigned OperandIdx, bool Verify) const final {
    llvm_unreachable("Unimplemented");
  }
  unsigned getUseOperandNo(const Use &Use) const final {
    llvm_unreachable("Unimplemented");
  }
  SmallVector<llvm::Instruction *, 1> getLLVMInstrs() const final {
    llvm_unreachable("Unimplemented");
  }
  unsigned getNumOfIRInstrs() const final { llvm_unreachable("Unimplemented"); }

  friend class SBVecContext;
  PackInst(ArrayRef<llvm::Instruction *> LLVMInstrs, Context &Ctx)
      : Instruction(ClassID::Pack, Opcode::Pack, LLVMInstrs[0], Ctx) {}

public:
  static Value *create(ArrayRef<Value *> PackOps, InsertPosition InsertBefore,
                       SBVecContext &Ctx);

  /// For isa/dyn_cast.
  static bool classof(const Value *From);
};

class SBVecContext : public Context {
  // Pack
  PackInst *createPackInst(ArrayRef<llvm::Instruction *> PackInstrs);
  friend class PackInst; // For createPackInst()

public:
  SBVecContext(llvm::LLVMContext &LLVMCtx) : Context(LLVMCtx) {}
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVEC_SANDBOXVECTORIZERIR_H
