//===- SandboxVectorizerIR.cpp - Sandbox IR Specialization ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizerIR.h"
#include "llvm/SandboxIR/Type.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

namespace llvm::sandboxir {

Value *PackInst::create(ArrayRef<Value *> PackOps, InsertPosition InsertBefore,
                        SBVecContext &Ctx) {
  auto &Builder = Instruction::setInsertPos(InsertBefore);
  // TODO: Replace with actual instruction sequence!
  auto *C = ConstantInt::get(Type::getInt32Ty(Ctx), 0);
  auto *LLVMC = cast<llvm::Constant>(C->Val);
  auto *LLVMVecTy = llvm::FixedVectorType::get(PackOps[0]->Val->getType(), 2);
  auto *LLVMPoison = llvm::PoisonValue::get(LLVMVecTy);
  llvm::Value *NewV =
      Builder.CreateInsertElement(LLVMPoison, PackOps[0]->Val, LLVMC, "TEST");

  if (auto *NewInsert = dyn_cast<llvm::InsertElementInst>(NewV))
    return Ctx.createPackInst({NewInsert});
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

bool PackInst::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Pack;
}

PackInst *
SBVecContext::createPackInst(ArrayRef<llvm::Instruction *> PackInstrs) {
  assert(all_of(PackInstrs,
                [](llvm::Value *V) {
                  return isa<llvm::InsertElementInst>(V) ||
                         isa<llvm::ExtractElementInst>(V);
                }) &&
         "Expected inserts or extracts!");
  auto NewPtr = std::unique_ptr<PackInst>(new PackInst(PackInstrs, *this));
  return cast<PackInst>(registerValue(std::move(NewPtr)));
}

} // namespace llvm::sandboxir
