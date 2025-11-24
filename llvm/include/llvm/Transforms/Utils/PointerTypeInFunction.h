#ifndef LLVM_TRANSFORMS_UTILS_POINTERTYPEINFUNCTION_H
#define LLVM_TRANSFORMS_UTILS_POINTERTYPEINFUNCTION_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Transforms/Utils/MyTy.h"
#include "llvm/Transforms/Utils/GlobalTypeInfo.h"

namespace llvm {

    class PointerTypeInFunctionPass : public AnalysisInfoMixin<PointerTypeInFunctionPass> {
    public:
      struct Result {
        DenseMap<Value *, std::shared_ptr<MyTy>> pointerTypeMap;
      };
      static llvm::AnalysisKey Key;
      Result run(Function &F, FunctionAnalysisManager &AM);
      void visitAllocaInst(Result *result, AllocaInst &AI);
      void visitLoadInst(Result *result, LoadInst &LI);
      void visitStoreInst(Result *result, StoreInst &LI);
      void visitGetElementPtrInst(Result *result, GetElementPtrInst &GI);
    
    private:
      void visitFunction(Result *result, Function &F);
      void recordPtmForNext(Result *, DenseMap<Value *, std::string> *);
      bool hasChanged(Result, DenseMap<Value *, std::string>);
      void addPointerTypeMap(Result *result, Value *opr, Type *type);
      void addPointerTypeMap(Result *result, Value *opr, Value *val);
      void testPrint(Result *result, Function &F);
      std::shared_ptr<MyTy> view_gep_index(Result *, GetElementPtrInst &, Value *,
          Type *, unsigned int);
      friend struct AnalysisInfoMixin<PointerTypeInFunctionPass>;
    };
}

#endif
