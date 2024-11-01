//===- Function.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_FUNCTION_H
#define LLVM_SANDBOXIR_FUNCTION_H

#include "llvm/IR/Function.h"
#include "llvm/SandboxIR/Constant.h"

namespace llvm::sandboxir {

class Function : public GlobalWithNodeAPI<Function, llvm::Function,
                                          GlobalObject, llvm::GlobalObject> {
  /// Helper for mapped_iterator.
  struct LLVMBBToBB {
    Context &Ctx;
    LLVMBBToBB(Context &Ctx) : Ctx(Ctx) {}
    BasicBlock &operator()(llvm::BasicBlock &LLVMBB) const {
      return *cast<BasicBlock>(Ctx.getValue(&LLVMBB));
    }
  };
  /// Use Context::createFunction() instead.
  Function(llvm::Function *F, sandboxir::Context &Ctx)
      : GlobalWithNodeAPI(ClassID::Function, F, Ctx) {}
  friend class Context; // For constructor.

public:
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Function;
  }

  Module *getParent() {
    return Ctx.getModule(cast<llvm::Function>(Val)->getParent());
  }

  Argument *getArg(unsigned Idx) const {
    llvm::Argument *Arg = cast<llvm::Function>(Val)->getArg(Idx);
    return cast<Argument>(Ctx.getValue(Arg));
  }

  size_t arg_size() const { return cast<llvm::Function>(Val)->arg_size(); }
  bool arg_empty() const { return cast<llvm::Function>(Val)->arg_empty(); }

  using iterator = mapped_iterator<llvm::Function::iterator, LLVMBBToBB>;
  iterator begin() const {
    LLVMBBToBB BBGetter(Ctx);
    return iterator(cast<llvm::Function>(Val)->begin(), BBGetter);
  }
  iterator end() const {
    LLVMBBToBB BBGetter(Ctx);
    return iterator(cast<llvm::Function>(Val)->end(), BBGetter);
  }
  FunctionType *getFunctionType() const;

#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::Function>(Val) && "Expected Function!");
  }
  void dumpNameAndArgs(raw_ostream &OS) const;
  void dumpOS(raw_ostream &OS) const final;
#endif
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_FUNCTION_H
