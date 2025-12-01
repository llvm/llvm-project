#ifndef LLVM_TRANSFORMS_UTILS_POINTERTYPEPRINTER_H
#define LLVM_TRANSFORMS_UTILS_POINTERTYPEPRINTER_H

#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/MyTy.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/PointerTypeHelpers.h"

namespace llvm {
  class PointerTypePrinter {
  private:
    raw_ostream &OS;
    PointerTypeHelpers helper;
    void printFunction(Function &F);
    void printBasicBlock(BasicBlock &B);
    void printInstruction(Instruction &I);
    void printValue(Value *V);
    void printType(Value *V, Type *T);

  public:
    PointerTypePrinter(raw_ostream &Out) : OS(Out) {}
    void load(PointerTypeHelpers helper);
    void printModule(Module &M);
  };
}

#endif