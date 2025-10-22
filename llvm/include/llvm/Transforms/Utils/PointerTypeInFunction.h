#ifndef LLVM_TRANSFORMS_UTILS_POINTERTYPEINFUNCTION_H
#define LLVM_TRANSFORMS_UTILS_POINTERTYPEINFUNCTION_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Transforms/Utils/MyTy.h"

namespace llvm {

    class PointerTypeInFunctionPass : public PassInfoMixin<PointerTypeInFunctionPass>,
                                      public InstVisitor<PointerTypeInFunctionPass> {
    private:
      DenseMap<Value *, MyTy *> pointerTypeMap;
      void addPointerTypeMap(Value *opr, Type *type);
      void addPointerTypeMap(Value *opr, Value *val);
      void testPrint(Function &F);
    public:
      PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
      void visitAllocaInst(AllocaInst &AI);
      void visitLoadInst(LoadInst &LI);
      void visitStoreInst(StoreInst &LI);
      void visitGetElementPtrInst(GetElementPtrInst &GI);
    };
}

#endif
