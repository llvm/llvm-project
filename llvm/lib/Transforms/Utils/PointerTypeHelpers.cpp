#include "llvm/Transforms/Utils/PointerTypeHelpers.h"

using namespace llvm;

void PointerTypeHelpers::processInFunction(Function& F) {
  visitFunction(F);
  while (ptmChanged) {
    visitFunction(F);
  }
}

DenseMap<Value*, std::shared_ptr<MyTy>> PointerTypeHelpers::getPtm() {
  return ptm;
}

void PointerTypeHelpers::initializeGlobalInfo(Module& M) {
  initializeStructInfo(M);
  for (auto &gl : M.globals()) {
    if (gl.getType()->isPointerTy() || gl.getType()->isArrayTy() ||
        gl.getType()->isStructTy()) {
      auto val = static_cast<Value *>(&gl);
      addPtm(val, toMyTy(val->getType()));
    }
  }
}

void PointerTypeHelpers::visitFunction(Function& F) {
  ptmChanged = false;
  for (auto& B : F) {
    for (auto& I : B) {
      if (I.getOpcode() == Instruction::Alloca) {
        auto AI = static_cast<AllocaInst *>(&I);
        visitAllocaInst(*AI);
      } else if (I.getOpcode() == Instruction::Load) {
        auto AI = static_cast<LoadInst *>(&I);
        visitLoadInst(*AI);
      } else if (I.getOpcode() == Instruction::Store) {
        auto AI = static_cast<StoreInst *>(&I);
        visitStoreInst(*AI);
      } else if (I.getOpcode() == Instruction::GetElementPtr) {
        auto AI = static_cast<GetElementPtrInst *>(&I);
        visitGetElementPtrInst(*AI);
      }
    }
  }
}

void PointerTypeHelpers::visitAllocaInst(AllocaInst& AI) {
  Value *val = static_cast<Value *>(&AI);
  addPtmByPointee(val, AI.getAllocatedType());
}

void PointerTypeHelpers::visitLoadInst(LoadInst& LI) {
  Value *opr = LI.getPointerOperand();
  auto typ = LI.getType();
  if (typ->isPointerTy() || typ->isArrayTy() || typ->isStructTy()) {
    Value *val = static_cast<Value *>(&LI);
    addPtmByValue(opr, val);
  } else {
    addPtmByPointee(opr, typ);
  }
}

void PointerTypeHelpers::visitStoreInst(StoreInst &SI) {
  Value *opr = SI.getPointerOperand();
  Value *val = SI.getValueOperand();
  auto typ = val->getType();
  if (typ->isPointerTy() || typ->isArrayTy() || typ->isStructTy()) {
    addPtmByValue(opr, val);
  } else {
    addPtmByPointee(opr, typ);
  }
}

void PointerTypeHelpers::visitGetElementPtrInst(GetElementPtrInst& GI) {
  auto opr = GI.getPointerOperand();
  Value *val = static_cast<Value *>(&GI);
  addPtmByPointee(opr, GI.getSourceElementType());
  auto iTy = MyTy::ptr_cast<MyPointerTy>(ptm[opr])->getInner();
  addPtmByPointee(opr, viewGepIndex(GI, val, iTy, 1));
}

void PointerTypeHelpers::visitAtomicCmpXchgInst(AtomicCmpXchgInst& AI) {
  auto opr = AI.getPointerOperand();
  auto cmp_val = AI.getCompareOperand();
  auto new_val = AI.getNewValOperand();
  auto typ = cmp_val->getType();
  if (typ->isPointerTy()) {
    addPtmByOther(cmp_val, new_val);
    addPtmByValue(opr, cmp_val);
  } else {
    addPtmByPointee(opr, typ);
  }
}

void PointerTypeHelpers::visitAtomicRMWInst(AtomicRMWInst& AI) {
  auto opr = AI.getPointerOperand();
  auto val = AI.getValOperand();
  if (val->getType()->isPointerTy()) {
    addPtmByValue(opr, val);
  } else {
    addPtmByPointee(opr, val->getType());
  }
}

void PointerTypeHelpers::visitPHINode(PHINode &PI) {
  auto cnt = PI.getNumIncomingValues();
  auto opr = PI.getIncomingValue(0);
  auto typ = PI.getType();
  if (typ->isPointerTy() || typ->isArrayTy() || typ->isStructTy()) {
    for (auto i = 1u; i < cnt; i++) {
      auto val = PI.getIncomingValue(i);
      addPtmByOther(opr, val);
    }
    auto res = static_cast<Value *>(&PI);
    addPtmByOther(res, opr);
  }
}

void PointerTypeHelpers::visitSelectInst(SelectInst &SI) {
  auto tVal = SI.getTrueValue();
  auto fVal = SI.getFalseValue();
  auto typ = tVal->getType();
  auto res = static_cast<Value *>(&SI);
  if (typ->isPointerTy() || typ->isArrayTy() || typ->isStructTy()) {
    addPtmByOther(tVal, fVal);
    addPtmByOther(res, tVal);
  }
}

std::shared_ptr<MyTy> PointerTypeHelpers::viewGepIndex(
    GetElementPtrInst& GI,
    Value* val,
    std::shared_ptr<MyTy> ty,
    unsigned int now) {
  if (now == GI.getNumIndices()) {
    addPtmByPointee(val, ty);
    errs() << ptm[val]->to_string() << "\n";
    return MyTy::ptr_cast<MyPointerTy>(ptm[val])->getInner();
  } else {
    if (ty->isArray()) {
      auto nxTy = MyTy::ptr_cast<MyArrayTy>(ty);
      nxTy->update(viewGepIndex(GI, val, nxTy->getElementTy(), now + 1));
      return nxTy;
    } else {
      auto nxTy = MyTy::ptr_cast<MyStructTy>(ty);
      auto index = static_cast<ConstantInt *>(GI.getOperand(now + 1))->getZExtValue();
      errs() << "View Struct: " << nxTy->to_string() << " with index " << index << "\n";
      nxTy->updateElement(viewGepIndex(GI, val, nxTy->getElementTy(index), now + 1), index);
      return nxTy;
    }
  }
}

std::shared_ptr<MyTy> PointerTypeHelpers::toMyTy(Type *typ) {
  if (typ->isStructTy()) {
    return structInfo[typ];
  } else {
    return MyTy::from(typ);
  }
}

void PointerTypeHelpers::addPtmByPointee(Value* opr, Type* ty) {
  addPtmByPointee(opr, toMyTy(ty));
}

void PointerTypeHelpers::addPtmByPointee(Value *opr, std::shared_ptr<MyTy> ty) {
  errs() << "Add " << ty->to_string() << " for ";
  opr->printAsOperand(errs());
  errs() << "\n";
  if (ptm.contains(opr)) {
    std::string lastTy = ptm[opr]->to_string();
    ptm[opr]->update(ty);
    if (lastTy != ptm[opr]->to_string()) {
      ptmChanged = true;
    }
  } else {
    ptm[opr] = std::make_shared<MyPointerTy>(ty);
    ptmChanged = true;
  }
}

void PointerTypeHelpers::addPtmByValue(Value* opr, Value* val) {
  auto typ = val->getType();
  std::shared_ptr<MyTy> mTy;
  if (ptm.contains(val)) {
    mTy = ptm[val];
  } else {
    mTy = toMyTy(typ);
    if (typ->isPointerTy() || typ->isArrayTy() || typ->isStructTy()) {
      addPtm(val, mTy);
    }
  }
  addPtmByPointee(opr, mTy);
  if (ptm.contains(val)) {
    if (ptm[opr]->isPointer()) {
      auto iTy = MyTy::ptr_cast<MyPointerTy>(ptm[opr])->getInner(); 
      if (iTy->isPointer()) {
        addPtm(val, iTy);
      } else if (iTy->isArray()) {
        auto iiTy = MyTy::ptr_cast<MyArrayTy>(iTy)->getElementTy();
        if (iiTy->compatibleWith(mTy)) {
          addPtm(val, iiTy);
        } else {
          addPtm(val, iTy);
        }
      } else if (iTy->isStruct()) {
        auto iiTy = MyTy::ptr_cast<MyStructTy>(iTy)->getElementTy();
        if (iiTy->compatibleWith(mTy)) {
          addPtm(val, iiTy);
        } else {
          addPtm(val, iTy);
        }
      }
    } else if (ptm[opr]->isArray()) {
      auto iTy = MyTy::ptr_cast<MyArrayTy>(ptm[opr])->getElementTy();
      addPtm(val, iTy);
    } else if (ptm[opr]->isStruct()) {
      auto iTy = MyTy::ptr_cast<MyStructTy>(ptm[opr])->getElementTy();
      addPtm(val, iTy);
    }
  }
}

void PointerTypeHelpers::addPtmByOther(Value* opr, Value* oth) {
  std::shared_ptr<MyTy> mTy;
  if (ptm.contains(opr)) {
    mTy = ptm[opr];
  } else {
    mTy = toMyTy(opr->getType());
  }
  addPtm(oth, mTy);
  addPtm(opr, ptm[oth]);
}

void PointerTypeHelpers::addPtm(Value *val, std::shared_ptr<MyTy> ty) {
  if (ptm.contains(val)) {
    std::string lastTy = ptm[val]->to_string();
    ptm[val] = MyTy::leastCompatibleType(ptm[val], ty);
    if (lastTy != ptm[val]->to_string()) {
      ptmChanged = true;
    }
  } else {
    ptm[val] = ty;
    ptmChanged = true;
  }
}

void PointerTypeHelpers::initializeStructInfo(Module& M) {
  for (auto st : M.getIdentifiedStructTypes()) {
    structInfo[st] = std::make_shared<MyStructTy>(st, structInfo);
  }
}

DenseMap<Type*, std::shared_ptr<MyTy>> PointerTypeHelpers::getStructInfo() {
  return structInfo;
}