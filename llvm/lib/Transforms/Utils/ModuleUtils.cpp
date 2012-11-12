//===-- ModuleUtils.cpp - Functions to manipulate Modules -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform manipulations on Modules.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/IRBuilder.h"
#include "llvm/Module.h"

using namespace llvm;

static void appendToGlobalArray(const char *Array, 
                                Module &M, Function *F, int Priority) {
  IRBuilder<> IRB(M.getContext());
  FunctionType *FnTy = FunctionType::get(IRB.getVoidTy(), false);
  StructType *Ty = StructType::get(
      IRB.getInt32Ty(), PointerType::getUnqual(FnTy), NULL);

  Constant *RuntimeCtorInit = ConstantStruct::get(
      Ty, IRB.getInt32(Priority), F, NULL);

  // Get the current set of static global constructors and add the new ctor
  // to the list.
  SmallVector<Constant *, 16> CurrentCtors;
  if (GlobalVariable * GVCtor = M.getNamedGlobal(Array)) {
    if (Constant *Init = GVCtor->getInitializer()) {
      unsigned n = Init->getNumOperands();
      CurrentCtors.reserve(n + 1);
      for (unsigned i = 0; i != n; ++i)
        CurrentCtors.push_back(cast<Constant>(Init->getOperand(i)));
    }
    GVCtor->eraseFromParent();
  }

  CurrentCtors.push_back(RuntimeCtorInit);

  // Create a new initializer.
  ArrayType *AT = ArrayType::get(RuntimeCtorInit->getType(),
                                 CurrentCtors.size());
  Constant *NewInit = ConstantArray::get(AT, CurrentCtors);

  // Create the new global variable and replace all uses of
  // the old global variable with the new one.
  (void)new GlobalVariable(M, NewInit->getType(), false,
                           GlobalValue::AppendingLinkage, NewInit, Array);
}

void llvm::appendToGlobalCtors(Module &M, Function *F, int Priority) {
  appendToGlobalArray("llvm.global_ctors", M, F, Priority);
}

void llvm::appendToGlobalDtors(Module &M, Function *F, int Priority) {
  appendToGlobalArray("llvm.global_dtors", M, F, Priority);
}
