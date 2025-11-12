#include "llvm/Transforms/Utils/PointerTypeInFunction.h"

using namespace llvm;

AnalysisKey PointerTypeInFunctionPass::Key;

PointerTypeInFunctionPass::Result PointerTypeInFunctionPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  if (F.getName() == "main") {
    // 每次遍历找到一些信息，直到达到一个不变点
    int cnt = 0;
    visit(F);
    while (!checkFixedPoint()) {
      errs() << "Found " << ++cnt << " times.\n";
      testPrint(F);   
      recordPtmForNext();
      visit(F);
    }
  }
  return result;
}

void PointerTypeInFunctionPass::recordPtmForNext() {
  for (auto &entry : result.pointerTypeMap) {
    ptmLastTurn[entry.first] = entry.second->to_string();  
  }
}

bool PointerTypeInFunctionPass::checkFixedPoint() {
  for (auto &entry : result.pointerTypeMap) {
    if (ptmLastTurn.contains(entry.first)) {
      if (ptmLastTurn[entry.first] != entry.second->to_string()) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

void PointerTypeInFunctionPass::visitAllocaInst(AllocaInst& AI) {
  Value *val = cast<Value>(&AI);
  addPointerTypeMap(val, AI.getAllocatedType());
}

void PointerTypeInFunctionPass::visitLoadInst(LoadInst &LI) {
  Value *opr = LI.getPointerOperand();
  if (LI.getType()->isPointerTy()) {
    Value *val = cast<Value>(&LI);
    addPointerTypeMap(opr, val);
  }
  else {
    addPointerTypeMap(opr, LI.getType());
  }
}

void PointerTypeInFunctionPass::visitStoreInst(StoreInst &SI) {
  Value *opr = SI.getPointerOperand();
  Value *val = SI.getValueOperand();
  if (val->getType()->isPointerTy()) {
    addPointerTypeMap(opr, val);
  }
  else {
    addPointerTypeMap(opr, val->getType());
  }
}

void PointerTypeInFunctionPass::visitGetElementPtrInst(GetElementPtrInst &GI) {
  auto type = GI.getSourceElementType();
  auto opr = GI.getPointerOperand();
  Value *val = cast<Value>(&GI);
  auto cnt = GI.getNumIndices();
  addPointerTypeMap(opr, type);
  result.pointerTypeMap[opr]->update(view_gep_index(val, type, 1, cnt)); 
  view_gep_index(
      val, MyTy::ptr_cast<MyPointerTy>(result.pointerTypeMap[opr])->getInner(),
      1, cnt);
}

std::shared_ptr<MyTy>
PointerTypeInFunctionPass::view_gep_index(Value *val, Type* ty, int now, int cnt) {
  if (now == cnt) {
    addPointerTypeMap(val, ty);
    auto iTy =
        MyTy::ptr_cast<MyPointerTy>(result.pointerTypeMap[val])->getInner();
    return iTy;
  } else {
    if (ty->isArrayTy()) {
      auto nxTy = cast<ArrayType>(ty);
      return std::make_shared<MyArrayTy>(
          view_gep_index(val, nxTy->getElementType(), now + 1, cnt),
          nxTy->getNumElements());
    } else {
      // 结构体
      return 0;
    }
  }
}

void PointerTypeInFunctionPass::view_gep_index(Value *val, std::shared_ptr<MyTy> ty, 
                                          int now, int cnt) {
  if (now == cnt) {
    result.pointerTypeMap[val]->update(ty);
    return;
  } else {
    if (ty->isArray()) {
      auto nxTy = MyTy::ptr_cast<MyArrayTy>(ty);
      view_gep_index(val, nxTy->getElementTy(), now + 1, cnt);
    }
  }
}

void PointerTypeInFunctionPass::addPointerTypeMap(Value *opr, Type *type) {
  auto mTy = MyTy::from(type);
  errs() << " Update ";
  opr->printAsOperand(errs());
  errs() << " with " << mTy->to_string() << "\n";
  if (result.pointerTypeMap.contains(opr)) {
    result.pointerTypeMap[opr]->update(mTy);
  }
  else {
    result.pointerTypeMap[opr] = std::make_shared<MyPointerTy>(mTy);
  }
}

void PointerTypeInFunctionPass::addPointerTypeMap(Value *opr, Value *val) {
  if (!result.pointerTypeMap.contains(val)) {    
    result.pointerTypeMap[val] = MyTy::from(val->getType());
  }
  if (result.pointerTypeMap.contains(opr)) {  
    auto pTy =
        MyTy::ptr_cast<MyPointerTy>(result.pointerTypeMap[opr])->getInner();
    if (pTy->compatibleWith(result.pointerTypeMap[val])) {
      result.pointerTypeMap[val] =
          MyTy::leastCompatibleType(result.pointerTypeMap[val], pTy);
    } 
  }
  auto mTy = result.pointerTypeMap[val];
  errs() << " Update ";
  opr->printAsOperand(errs());
  errs() << " with " << mTy->to_string() << "\n";
  if (result.pointerTypeMap.contains(opr)) {
    result.pointerTypeMap[opr]->update(result.pointerTypeMap[val]);
  }
  else {
    result.pointerTypeMap[opr] = std::make_shared<MyPointerTy>(result.pointerTypeMap[val]);
  }
}

void PointerTypeInFunctionPass::testPrint(Function &F) {
  errs() << "Pointers in function: " << F.getName() << "\n";
  for (auto &entry: result.pointerTypeMap) {
    errs() << "  The pointer \" ";
    entry.getFirst()->printAsOperand(errs(), false);
    errs() << " \" has type \" " << entry.getSecond()->to_string() << " \" \n";
  }  
}


//wllvm