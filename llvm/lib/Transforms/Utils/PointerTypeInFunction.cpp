#include "llvm/Transforms/Utils/PointerTypeInFunction.h"

using namespace llvm;

AnalysisKey PointerTypeInFunctionPass::Key;

PointerTypeInFunctionPass::Result PointerTypeInFunctionPass::run(Function &F,
                                      FunctionAnalysisManager &FAM) {
  Result result;
  DenseMap<Value *, std::string> ptmWithInst;
  
  Module &M = *F.getParent();
  const auto &MAMProxy = FAM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
  auto &structInfo = MAMProxy.getCachedResult<GlobalTypeInfoPass>(M)->structInfo;

  while (hasChanged(result, ptmWithInst)) {
    visitFunction(&result, F);
    testPrint(&result, F);
    recordPtmForNext(&result, &ptmWithInst);
  }
  return result;
}

void PointerTypeInFunctionPass::visitFunction(Result *result, Function& F) {
  for (auto &B : F) {
    for (auto &I : B) {
      if (I.getOpcode() == Instruction::Alloca) {
        auto &AI = cast<AllocaInst>(I);
        visitAllocaInst(result, AI);
      } else if (I.getOpcode() == Instruction::Load) {
        auto &AI = cast<LoadInst>(I);
        visitLoadInst(result, AI);
      } else if (I.getOpcode() == Instruction::Store) {
        auto &AI = cast<StoreInst>(I);
        visitStoreInst(result, AI);
      } else if (I.getOpcode() == Instruction::GetElementPtr) {
        auto &AI = cast<GetElementPtrInst>(I);
        visitGetElementPtrInst(result, AI);
      }
    }
  }
}

void PointerTypeInFunctionPass::recordPtmForNext(
    Result *result, DenseMap<Value *, std::string> *ptmWithInst) {
  for (auto &entry : result->pointerTypeMap) {
    (*ptmWithInst)[entry.first] = entry.second->to_string();  
  }
}

bool PointerTypeInFunctionPass::hasChanged(
    Result result, DenseMap<Value *, std::string> ptmWithInst) {
  for (auto &entry : result.pointerTypeMap) {
    if (ptmWithInst.contains(entry.first)) {
      if (ptmWithInst[entry.first] != entry.second->to_string()) {
        return true;
      }
    } else {
      return true;
    }
  }
  return false;
}

void PointerTypeInFunctionPass::visitAllocaInst(Result *result, AllocaInst &AI) {
  Value *val = cast<Value>(&AI);
  addPointerTypeMap(result, val, AI.getAllocatedType());
}

void PointerTypeInFunctionPass::visitLoadInst(Result *result, LoadInst &LI) {
  Value *opr = LI.getPointerOperand();
  if (LI.getType()->isPointerTy()) {
    Value *val = cast<Value>(&LI);
    addPointerTypeMap(result, opr, val);
  }
  else {
    addPointerTypeMap(result, opr, LI.getType());
  }
}

void PointerTypeInFunctionPass::visitStoreInst(Result *result, StoreInst &SI) {
  Value *opr = SI.getPointerOperand();
  Value *val = SI.getValueOperand();
  if (val->getType()->isPointerTy()) {
    addPointerTypeMap(result, opr, val);
  }
  else {
    addPointerTypeMap(result, opr, val->getType());
  }
}

void PointerTypeInFunctionPass::visitGetElementPtrInst(Result *result,
                                                       GetElementPtrInst &GI) {
  auto type = GI.getSourceElementType();
  auto opr = GI.getPointerOperand();
  Value *val = cast<Value>(&GI);
  addPointerTypeMap(result, opr, type);
  result->pointerTypeMap[opr]->update(view_gep_index(result, GI, val, type, 1)); 
}

std::shared_ptr<MyTy>
PointerTypeInFunctionPass::view_gep_index(
    Result *result,
    GetElementPtrInst &GI, 
    Value *val, 
    Type* ty, 
    unsigned int now) {
  if (now == GI.getNumIndices()) {
    addPointerTypeMap(result, val, ty);
    auto iTy =
        MyTy::ptr_cast<MyPointerTy>(result->pointerTypeMap[val])->getInner();
    return iTy;
  } else {
    if (ty->isArrayTy()) {
      auto nxTy = cast<ArrayType>(ty);
      return std::make_shared<MyArrayTy>(
          view_gep_index(result, GI, val, nxTy->getElementType(), now + 1),
          nxTy->getNumElements());
    } else {
      auto nxTy = cast<StructType>(ty);
      auto index = cast<ConstantInt>(GI.getOperand(now + 1))->getZExtValue();
      view_gep_index(result, GI, val, nxTy->getElementType(index), now + 1);
      return std::make_shared<MyStructTy>(nxTy);
    }
  }
}

void PointerTypeInFunctionPass::addPointerTypeMap(Result *result, Value *opr,
                                                  Type *type) {
  auto mTy = MyTy::from(type);
  errs() << " Update ";
  opr->printAsOperand(errs());
  errs() << " with " << mTy->to_string() << "\n";
  if (result->pointerTypeMap.contains(opr)) {
    result->pointerTypeMap[opr]->update(mTy);
  }
  else {
    result->pointerTypeMap[opr] = std::make_shared<MyPointerTy>(mTy);
  }
}

void PointerTypeInFunctionPass::addPointerTypeMap(Result *result, Value *opr,
                                                  Value *val) {
  if (!result->pointerTypeMap.contains(val)) {    
    result->pointerTypeMap[val] = MyTy::from(val->getType());
  }
  if (!result->pointerTypeMap.contains(opr)) {
    result->pointerTypeMap[val] = MyTy::from(opr->getType());
  } 
  auto pTy = MyTy::ptr_cast<MyPointerTy>(result->pointerTypeMap[opr])->getInner();
  if (pTy->compatibleWith(result->pointerTypeMap[val])) {
    result->pointerTypeMap[val] =
        MyTy::leastCompatibleType(result->pointerTypeMap[val], pTy);
  } 
  auto &mTy = result->pointerTypeMap[val];
  errs() << " Update ";
  opr->printAsOperand(errs());
  errs() << " with " << mTy->to_string() << "\n";
  if (result->pointerTypeMap.contains(opr)) {
    result->pointerTypeMap[opr]->update(result->pointerTypeMap[val]);
  }
  else {
    result->pointerTypeMap[opr] = std::make_shared<MyPointerTy>(result->pointerTypeMap[val]);
  }
}

void PointerTypeInFunctionPass::testPrint(Result *result, Function &F) {
  errs() << "Pointers in function: " << F.getName() << "\n";
  for (auto &entry: result->pointerTypeMap) {
    errs() << "  The pointer \" ";
    entry.getFirst()->printAsOperand(errs(), false);
    errs() << " \" has type \" " << entry.getSecond()->to_string() << " \" \n";
  }  
}


//wllvm