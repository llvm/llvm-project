#ifndef LLVM_TRANSFORMS_UTILS_POINTERTYPETOOL_FLOWANALYZER_H
#define LLVM_TRANSFORMS_UTILS_POINTERTYPETOOL_FLOWANALYZER_H

#include "llvm/Transforms/Utils/PointerTypeTool/MyTy.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Constants.h"

namespace llvm {

class FlowAnalyzer {
  struct BlockInfo {
    DenseMap<Value *, shared_ptr<MyTy>> typeInfo;
    SetVector<BasicBlock *> inBlock;
    SetVector<BasicBlock *> outBlock;
    BasicBlock *block;
  };
  DenseMap<BasicBlock *, BlockInfo> blockInfo;
  DenseMap<Type *, shared_ptr<MyStructTy>> structInfo;
  SetVector<BasicBlock *> workList;

  shared_ptr<MyTy> toMyTy(Type *ty);
  void addBlockEdge(BasicBlock *from, BasicBlock *to);
  void meet(BasicBlock *B);
  void process(BasicBlock *B);
  bool addTypeInfo(DenseMap<Value *, shared_ptr<MyTy>> &typeInfo,
                   Value *val, Type *ty);
  bool addTypeInfo(DenseMap<Value *, shared_ptr<MyTy>> &typeInfo,
                   Value *val, shared_ptr<MyTy> ty);
  bool addTypeInfo(DenseMap<Value *, shared_ptr<MyTy>> &typeInfo,
                   Value *val1, Value *val2);
  bool addTypeInfoByPointee(DenseMap<Value *, shared_ptr<MyTy>> &typeInfo,
                            Value *ptr, Type *ty);
  bool addTypeInfoByPointee(DenseMap<Value *, shared_ptr<MyTy>> &typeInfo,
                            Value *ptr, shared_ptr<MyTy> ty);
  bool addTypeInfoByPointee(DenseMap<Value *, shared_ptr<MyTy>> &typeInfo,
                            Value *ptr, Value *val);
  shared_ptr<MyTy> visitAggType(DenseMap<Value *, shared_ptr<MyTy>> &typeInfo,
                                shared_ptr<MyTy> type, Value *val,
                                SmallVector<Value *> &indices,
                                unsigned int index, bool &changed);
  shared_ptr<MyTy> visitAggPtrType(DenseMap<Value *, shared_ptr<MyTy>> &typeInfo,
                                shared_ptr<MyTy> type, Value *valPtr,
                                SmallVector<Value *> &indices,
                                unsigned int index, bool &changed);

  void initWithInst(Instruction *I);
  void initWithReturnInst(ReturnInst *RI);
  void initWithBranchInst(BranchInst *BI);
  void initWithSwitchInst(SwitchInst *SI);
  void initWithIndirectBrInst(IndirectBrInst *II);
  void initWithInvolkeInst(InvokeInst *II);
  
  void visitFunction(Function &F);
  void visitBasicBlock(BasicBlock *B);
  bool visitInst(Instruction *I);
  bool visitNonExInfoInst(Instruction *I);
  bool visitReturnInst(ReturnInst *RI);
  bool visitInvokeInst(InvokeInst *II);
  bool visitExtractElementInst(ExtractElementInst *EI);
  bool visitInsertElementInst(InsertElementInst *II);
  bool visitShuffleVectorInst(ShuffleVectorInst *SI);
  bool visitExtractValueInst(ExtractValueInst *EI);
  bool visitInsertValueInst(InsertValueInst *II);
  bool visitAllocaInst(AllocaInst *AI);
  bool visitLoadInst(LoadInst *LI);
  bool visitStoreInst(StoreInst *SI);
  bool visitAtomicCmpXchgInst(AtomicCmpXchgInst *AI);
  bool visitAtomicRWMInst(AtomicRMWInst *AI);
  bool visitGetElementPtrInst(GetElementPtrInst *GI);
  bool visitICmpInst(ICmpInst *II);
  bool visitPHINode(PHINode *PN);
  bool visitSelectInst(SelectInst *SI);

  void printUpdateLog(Value *V, shared_ptr<MyTy> T);
  void printPtrUpdateLog(Value *V, shared_ptr<MyTy> T);
  void printUpdateLog(Value *V, Type *T);
  void printPtrUpdateLog(Value *V, Type *T);
  void printUpdateLog(Value *V1, Value *V2);
  void printPtrUpdateLog(Value *P, Value *V);

public:
  FlowAnalyzer(Module &M);
  void visitModule(Module &M);
  const DenseMap<Type *, shared_ptr<MyStructTy>> &getStructInfo();
  const DenseMap<Value *, shared_ptr<MyTy>> &getTypeInfo(BasicBlock *B);
};

}

#endif