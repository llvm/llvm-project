#ifndef LLVM_TRANSFORMS_UTILS_POINTERTYPEINFUNCTION_H
#define LLVM_TRANSFORMS_UTILS_POINTERTYPEINFUNCTION_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Transforms/Utils/MyTy.h"

namespace llvm {

    class PointerTypeInFunctionPass : public AnalysisInfoMixin<PointerTypeInFunctionPass>,
                                      public InstVisitor<PointerTypeInFunctionPass> {
    public:
      struct Result {
        DenseMap< Value *, std::shared_ptr<MyTy> > pointerTypeMap;
      };
      static llvm::AnalysisKey Key;
      Result run(Function &F, FunctionAnalysisManager &AM);
      void visitAllocaInst(AllocaInst &AI);
      void visitLoadInst(LoadInst &LI);
      void visitStoreInst(StoreInst &LI);
      void visitGetElementPtrInst(GetElementPtrInst &GI);
    
    private:
      Result result;
      DenseMap<Value *, std::string> ptmLastTurn;
      void recordPtmForNext();
      bool checkFixedPoint();
      void addPointerTypeMap(Value *opr, Type *type);
      void addPointerTypeMap(Value *opr, Value *val);
      void testPrint(Function &F);
      std::shared_ptr<MyTy> view_gep_index(Value *, Type *, int, int);
      void view_gep_index(Value *, std::shared_ptr<MyTy>, int, int);
      friend struct AnalysisInfoMixin<PointerTypeInFunctionPass>;
    };
}

#endif
