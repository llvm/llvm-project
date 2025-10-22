#include "llvm/Transforms/Utils/PointerTypeInFunction.h"

using namespace llvm;

PreservedAnalyses PointerTypeInFunctionPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  visit(F);
  testPrint(F);
  return PreservedAnalyses::all();
}

void PointerTypeInFunctionPass::visitAllocaInst(AllocaInst& AI) {
  Value *val = cast<Value>(&AI);
  addPointerTypeMap(val, AI.getAllocatedType());
}

void PointerTypeInFunctionPass::visitLoadInst(LoadInst &LI) {
  if (LI.getType()->isPointerTy()) {
    Value *val = cast<Value>(&LI);
    addPointerTypeMap(LI.getPointerOperand(), val);
  }
  else {
    addPointerTypeMap(LI.getPointerOperand(), LI.getType());
  }
}

void PointerTypeInFunctionPass::visitStoreInst(StoreInst &SI) {
  if (SI.getValueOperand()->getType()->isPointerTy()) {
    addPointerTypeMap(SI.getPointerOperand(), SI.getValueOperand());
  }
  else {
    addPointerTypeMap(SI.getPointerOperand(), SI.getValueOperand()->getType());
  }
}

void PointerTypeInFunctionPass::visitGetElementPtrInst(GetElementPtrInst &GI) {
  auto type = GI.getSourceElementType();
  auto opr = GI.getPointerOperand();
  Value *val = cast<Value>(&GI);
  auto cnt = GI.getNumIndices();
  if (cnt > 1) {
    addPointerTypeMap(opr, type);
    auto tarTy = type;
    for (unsigned int i = 1; i < cnt; i++) {
      if (tarTy->getTypeID() == Type::ArrayTyID) {
        tarTy = cast<ArrayType>(tarTy);
        tarTy = tarTy->getArrayElementType();
      }
      // Todo: Get type from Struct.
    }
    addPointerTypeMap(val, tarTy);
  }
}

void PointerTypeInFunctionPass::addPointerTypeMap(Value *opr, Type *type) {
  if (pointerTypeMap.contains(opr)) {
    pointerTypeMap[opr]->update(MyTy::from(type));
  }
  else {
    pointerTypeMap[opr] = new MyPointerTy(MyTy::from(type));
  }
}

void PointerTypeInFunctionPass::addPointerTypeMap(Value *opr, Value *val) {
  if (!pointerTypeMap.contains(val)) {
    pointerTypeMap[val] = MyTy::from(val->getType());
  }
  if (pointerTypeMap.contains(opr)) {
    pointerTypeMap[opr]->update(pointerTypeMap[val]);
  }
  else {
    pointerTypeMap[opr] = new MyPointerTy(pointerTypeMap[val]);
  }
}

void PointerTypeInFunctionPass::testPrint(Function &F) {
  if (F.getName() == "main") {
    errs() << "Pointers in function: " << F.getName() << "\n";
    for (auto &entry: pointerTypeMap) {
      errs() << "  The pointer \" ";
      entry.getFirst()->printAsOperand(errs(), false);
      errs() << " \" has type \" " << entry.getSecond()->to_string() << " \" \n";
    }
  }
}

