#ifndef LLVM_TRANSFORMS_UTILS_POINTERTYPEHELPERS_H
#define LLVM_TRANSFORMS_UTILS_POINTERTYPEHELPERS_H

#include "llvm/Transforms/Utils/MyTy.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include <memory>

namespace llvm {
  class PointerTypeHelpers {
    bool ptmChanged;
    DenseMap<Value *, std::shared_ptr<MyTy>> ptm;
    DenseMap<Value *, std::shared_ptr<MyTy>> ptmLast;
    DenseMap<Type *, std::shared_ptr<MyTy>> structInfo;
    void visitFunction(Function& F);
    void visitAllocaInst(AllocaInst& AI);
    void visitLoadInst(LoadInst& LI);
    void visitStoreInst(StoreInst& SI);
    void visitGetElementPtrInst(GetElementPtrInst& GI);
    void visitAtomicCmpXchgInst(AtomicCmpXchgInst &AI);
    void visitAtomicRMWInst(AtomicRMWInst& AI);
    void visitPHINode(PHINode &PI);
    void visitSelectInst(SelectInst &SI);
    void addPtmByPointee(Value *opr, Type *ty);
    void addPtmByPointee(Value *opr, std::shared_ptr<MyTy> ty);
    void addPtmByValue(Value *opr, Value *val);
    void addPtmByOther(Value *opr, Value *oth);
    void addPtm(Value *opr, std::shared_ptr<MyTy> ty);
    std::shared_ptr<MyTy> toMyTy(Type *typ);
    std::shared_ptr<MyTy> viewGepIndex(
        GetElementPtrInst&, 
        Value *,
        std::shared_ptr<MyTy>, 
        unsigned int);
    void initializeStructInfo(Module& M);
  public:
    void processInFunction(Function& F);
    DenseMap<Value *, std::shared_ptr<MyTy>> getPtm();
    void initializeGlobalInfo(Module& M);
    DenseMap<Type *, std::shared_ptr<MyTy>> getStructInfo();
  };
}

#endif
