#include "llvm/Transforms/Utils/PointerTypeTool/FlowAnalyzer.h"

using namespace llvm;

FlowAnalyzer::FlowAnalyzer(Module& M) {
  for (auto st : M.getIdentifiedStructTypes()) {
    toMyTy(st);
  }

  for (auto &F : M) {
    for (auto &B : F) {
      BlockInfo info;
      info.block = &B;
    }
    
    for (auto &B : F) {
      for (auto &I : B) {
        initWithInst(&I);
      }
    }
  }
}

void FlowAnalyzer::initWithInst(Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::Ret:
    initWithReturnInst(static_cast<ReturnInst *>(I));
    break;
  case Instruction::Br:
    initWithBranchInst(static_cast<BranchInst *>(I));
    break;
  case Instruction::Switch:
    initWithSwitchInst(static_cast<SwitchInst *>(I));
    break;
  case Instruction::IndirectBr:
    initWithIndirectBrInst(static_cast<IndirectBrInst *>(I));
    break;
  case Instruction::Invoke:
    initWithInvolkeInst(static_cast<InvokeInst *>(I));
    break;
  default:
    break;
  }
}

void FlowAnalyzer::initWithReturnInst(ReturnInst* RI) {
  // Todo: Deal with function.
}

void FlowAnalyzer::initWithBranchInst(BranchInst *BI) {
  auto B = BI->getParent();
  auto cnt = BI->getNumSuccessors();
  for (auto i = 0u; i < cnt; i++) {
    auto to = BI->getSuccessor(i);
    addBlockEdge(B, to);
  }
}

void FlowAnalyzer::initWithSwitchInst(SwitchInst *SI) {
  auto B = SI->getParent();
  auto cnt = SI->getNumSuccessors();
  for (auto i = 0u; i < cnt; i++) {
    auto to = SI->getSuccessor(i);
    addBlockEdge(B, to);
  }
}

void FlowAnalyzer::initWithIndirectBrInst(IndirectBrInst *II) {
  auto B = II->getParent();
  auto cnt = II->getNumDestinations();
  for (auto i = 0u; i < cnt; i++) {
    auto to = II->getDestination(i);
    addBlockEdge(B, to);
  }
}

void FlowAnalyzer::initWithInvolkeInst(InvokeInst *II) {
  auto B = II->getParent();
  auto ND = II->getNormalDest();
  auto ED = II->getUnwindDest();
  addBlockEdge(B, ND);
  addBlockEdge(B, ED);
  // Todo: Deal with function.
}

void FlowAnalyzer::addBlockEdge(BasicBlock *from, BasicBlock *to) {
  blockInfo[from].outBlock.insert(to);
  blockInfo[to].inBlock.insert(from);
}

shared_ptr<MyTy> FlowAnalyzer::toMyTy(Type* ty) {
  switch (ty->getTypeID()) {
  case Type::IntegerTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID:
  case Type::BFloatTyID:
  case Type::FP128TyID:
  case Type::HalfTyID:
  case Type::LabelTyID:
  case Type::PPC_FP128TyID:
    return make_shared<MyBasicTy>(ty);
  case Type::ArrayTyID: {
    auto arrayTy = static_cast<ArrayType *>(ty);
    auto cnt = arrayTy->getNumElements();
    auto eTy = toMyTy(arrayTy->getElementType());
    return make_shared<MyArrayTy>(eTy, cnt);
  }
  case Type::FixedVectorTyID: {
    auto vecTy = static_cast<FixedVectorType *>(ty);
    auto cnt = vecTy->getNumElements();
    auto eTy = toMyTy(vecTy->getElementType());
    return make_shared<MyVectorTy>(eTy, cnt, true);
  }
  case Type::ScalableVectorTyID: {
    auto vecTy = static_cast<ScalableVectorType *>(ty);
    auto cnt = vecTy->getElementCount().getFixedValue();
    auto eTy = toMyTy(vecTy->getElementType());
    return make_shared<MyVectorTy>(eTy, cnt, false);
  }
  case Type::PointerTyID:
    return make_shared<MyPointerTy>(make_shared<MyTy>());
  case Type::StructTyID:
    if (!structInfo.contains(ty) || structInfo[ty]->isOpaque()) {
      auto structTy = static_cast<StructType *>(ty);
      auto name = structTy->hasName() ? structTy->getName().str() : "";
      SmallVector<shared_ptr<MyTy>> vec;
      int cnt = structTy->getNumElements();
      for (int i = 0; i < cnt; i++) {
        auto eTy = structTy->getElementType(i);
        vec.push_back(toMyTy(eTy));
      }
      auto sTy = make_shared<MyStructTy>(name, vec, structTy->isOpaque());
      structInfo[ty] = sTy;
    }
    return structInfo[ty];
  default:
    assert(false);
  }
}

void FlowAnalyzer::visitModule(Module &M) {
  for (auto &F : M) {
    visitFunction(F);
  }
}

void FlowAnalyzer::visitFunction(Function& F) {
  workList.clear();

  for (auto &B : F) {
    workList.insert(&B);
  }

  while (!workList.empty()) {
    auto B = workList.front();
    workList.remove(B);
    visitBasicBlock(B);
  }
}

void FlowAnalyzer::meet(BasicBlock *B) {
  auto &data = blockInfo[B].typeInfo;
  for (auto IB : blockInfo[B].inBlock) {
    for (auto &entry : blockInfo[IB].typeInfo) {
      auto key = entry.first;
      auto &val = entry.second;
      if (data.contains(key)) {
        data[key] = MyTy::leastCompatibleType(data[key], val);
      } else {
        data[key] = val;
      }
    }
  }
}

void FlowAnalyzer::process(BasicBlock *B) {
  bool changed = false;
  for (auto it = B->begin(); it != B->end(); it++) {
    auto &I = *it;
    changed |= visitInst(&I);
  }
  if (!changed) {
    return;
  }
  for (auto OB : blockInfo[B].outBlock) {
    workList.insert(OB);
  }
  while (changed) {
    changed = false;
    for (auto it = B->begin(); it != B->end(); it++) {
      auto &I = *it;
      changed |= visitInst(&I);
    }
  }
}

bool FlowAnalyzer::addTypeInfo(
    DenseMap<Value *, shared_ptr<MyTy>> &typeInfo, Value *val, Type *ty) {
  printUpdateLog(val, ty);
  return addTypeInfo(typeInfo, val, toMyTy(ty));
}

bool FlowAnalyzer::addTypeInfo(
    DenseMap<Value *, shared_ptr<MyTy>> &typeInfo, Value *val,
    shared_ptr<MyTy> ty) {
  printUpdateLog(val, ty);
  if (typeInfo.contains(val)) {
    string lastTy = typeInfo[val]->toString();
    typeInfo[val] = MyTy::leastCompatibleType(typeInfo[val], ty);
    return lastTy != typeInfo[val]->toString();
  } else {
    typeInfo[val] = ty;
    return false;
  }
}

bool FlowAnalyzer::addTypeInfo(
    DenseMap<Value *, shared_ptr<MyTy>> &typeInfo, Value *val1,
    Value *val2) {
  printUpdateLog(val1, val2);
  shared_ptr<MyTy> mTy;
  if (typeInfo.contains(val1)) {
    mTy = typeInfo[val1];
  } else {
    mTy = toMyTy(val1->getType());
  }
  bool changed = false;
  changed |= addTypeInfo(typeInfo, val2, mTy);
  changed |= addTypeInfo(typeInfo, val1, typeInfo[val2]);
  return changed;
}

bool FlowAnalyzer::addTypeInfoByPointee(
    DenseMap<Value *, shared_ptr<MyTy>> &typeInfo, Value *ptr, Type *ty) {
  printPtrUpdateLog(ptr, ty);
  return addTypeInfoByPointee(typeInfo, ptr, toMyTy(ty));
}

bool FlowAnalyzer::addTypeInfoByPointee(
    DenseMap<Value *, shared_ptr<MyTy>> &typeInfo, Value *ptr,
    shared_ptr<MyTy> ty) {
  printPtrUpdateLog(ptr, ty);
  if (typeInfo.contains(ptr)) {
    return typeInfo[ptr]->update(ty);
  } else {
    typeInfo[ptr] = make_shared<MyPointerTy>(ty);
    return false;
  }
}

bool FlowAnalyzer::addTypeInfoByPointee(
    DenseMap<Value *, shared_ptr<MyTy>> &typeInfo, Value *ptr,
    Value *val) {
  printPtrUpdateLog(ptr, val);
  auto ty = val->getType();
  bool ret = false;
  shared_ptr<MyTy> mTy;
  if (typeInfo.contains(val)) {
    mTy = typeInfo[val];
  } else {
    mTy = toMyTy(ty);
    ret = addTypeInfo(typeInfo, val, mTy) || ret;
  }
  ret = addTypeInfoByPointee(typeInfo, ptr, mTy) || ret;
  if (typeInfo[ptr]->isPointer()) {
    auto iTy = typeInfo[ptr]->getPointeeTyAsPtr();
    if (iTy->isArray() || iTy->isStruct()) {
      auto iiTy = iTy->getPointeeTyAsPtr();
      if (iiTy->compatibleWith(mTy)) {
        return addTypeInfo(typeInfo, val, iiTy) || ret;
      } else {
        return addTypeInfo(typeInfo, val, iTy) || ret;
      }
    } else {
      return addTypeInfo(typeInfo, val, iTy) || ret;
    }
  } else if (typeInfo[ptr]->isArray() || typeInfo[ptr]->isStruct()) {
    return addTypeInfo(typeInfo, val, typeInfo[ptr]->getPointeeTyAsPtr()) || ret;
  } else {
    assert(0);
  }
}

shared_ptr<MyTy>
FlowAnalyzer::visitAggType(DenseMap<Value *, shared_ptr<MyTy>> &typeInfo,
                           shared_ptr<MyTy> type, Value *val,
                           SmallVector<Value *> &indices, unsigned int index,
                           bool &changed) {
  if (index == indices.size() - 1) {
    changed |= addTypeInfo(typeInfo, val, type);
    return typeInfo[val];
  } else {
    if (type->isStruct()) {
      auto structIdx =
          static_cast<ConstantInt *>(indices[index])->getZExtValue();
      auto structTy = MyTy::ptr_cast<MyStructTy>(type);
      auto nxTy = structTy->getElementTy(structIdx);
      changed |= structTy->updateElement(
          visitAggType(typeInfo, nxTy, val, indices, index + 1, changed),
          structIdx);
      return structTy;
    } else {
      assert(type->isArray() ||
             (type->isPointer() &&
                 static_cast<ConstantInt *>(indices[index])->getZExtValue() == 0));
      changed |= type->update(visitAggPtrType(typeInfo, type->getPointeeTyAsPtr(),
                                       val, indices, index + 1, changed));
      return type;
    }
  }
}

shared_ptr<MyTy>
FlowAnalyzer::visitAggPtrType(DenseMap<Value *, shared_ptr<MyTy>> &typeInfo,
                           shared_ptr<MyTy> type, Value *valPtr,
                           SmallVector<Value *> &indices, unsigned int index,
                           bool &changed) {
  errs() << "visitAggPtrType: " << index << "/" << indices.size()
         << " " << type->toString() << "\n";
  if (index >= indices.size()) {
    changed |= addTypeInfoByPointee(typeInfo, valPtr, type);
    return typeInfo[valPtr]->getPointeeTyAsPtr();
  } else {
    errs() << type->toString() << " " << type->isStruct() << "\n";
    if (type->isStruct()) {
      auto structIdx =
          static_cast<ConstantInt *>(indices[index])->getZExtValue();
      auto structTy = MyTy::ptr_cast<MyStructTy>(type);
      auto nxTy = structTy->getElementTy(structIdx);
      changed |= structTy->updateElement(
          visitAggPtrType(typeInfo, nxTy, valPtr, indices, index + 1, changed),
          structIdx);
      return structTy;
    } else {
      assert(type->isArray() ||
             (type->isPointer() &&
              static_cast<ConstantInt *>(indices[index])->getZExtValue() == 0));
      changed |= type->update(visitAggPtrType(typeInfo, type->getPointeeTyAsPtr(), 
          valPtr, indices, index + 1, changed));
      return type;
    }
  }
}

void FlowAnalyzer::visitBasicBlock(BasicBlock *B) {
  meet(B);
  process(B);
}

bool FlowAnalyzer::visitInst(Instruction* I) {
  errs() << "\nNow visit " << I->getOpcodeName() << ":\n";
  switch (I->getOpcode()) {
  case Instruction::Ret:
    return visitReturnInst(static_cast<ReturnInst *>(I));
  case Instruction::Invoke:
    return visitInvokeInst(static_cast<InvokeInst *>(I));
  case Instruction::ExtractElement:
    return visitExtractElementInst(static_cast<ExtractElementInst *>(I));
  case Instruction::InsertElement:
    return visitInsertElementInst(static_cast<InsertElementInst *>(I));
  case Instruction::ShuffleVector:
    return visitShuffleVectorInst(static_cast<ShuffleVectorInst *>(I));
  case Instruction::ExtractValue:
    return visitExtractValueInst(static_cast<ExtractValueInst *>(I));
  case Instruction::InsertValue:
    return visitInsertValueInst(static_cast<InsertValueInst *>(I));
  case Instruction::Alloca:
    return visitAllocaInst(static_cast<AllocaInst *>(I));
  case Instruction::Load:
    return visitLoadInst(static_cast<LoadInst *>(I));
  case Instruction::Store:
    return visitStoreInst(static_cast<StoreInst *>(I));
  case Instruction::AtomicCmpXchg:
    return visitAtomicCmpXchgInst(static_cast<AtomicCmpXchgInst *>(I));
  case Instruction::AtomicRMW:
    return visitAtomicRWMInst(static_cast<AtomicRMWInst *>(I));
  case Instruction::GetElementPtr:
    return visitGetElementPtrInst(static_cast<GetElementPtrInst *>(I));
  case Instruction::ICmp:
    return visitICmpInst(static_cast<ICmpInst *>(I));
  case Instruction::PHI:
    return visitPHINode(static_cast<PHINode *>(I));
  case Instruction::Select:
    return visitSelectInst(static_cast<SelectInst *>(I));
  default:
    return visitNonExInfoInst(I);
  }
}

bool FlowAnalyzer::visitNonExInfoInst(Instruction* I) {
  auto B = I->getParent();
  auto &typeInfo = blockInfo[B].typeInfo;
  auto val = static_cast<Value *>(I);
  bool ret = val->getType()->isVoidTy() ? false : addTypeInfo(typeInfo, val, val->getType());
  for (auto i = 0u; i < I->getNumOperands(); i++) {
    auto op = I->getOperand(i);
    ret |= addTypeInfo(typeInfo, op, op->getType());
  }
  return ret;
}

bool FlowAnalyzer::visitReturnInst(ReturnInst *RI) {
  bool ret = visitNonExInfoInst(RI);
  // Todo: Deal with function.
  return ret;
}

bool FlowAnalyzer::visitInvokeInst(InvokeInst* II) { 
  bool ret = visitNonExInfoInst(II);
  // Todo: Deal with function.
  return ret;
}

bool FlowAnalyzer::visitExtractElementInst(ExtractElementInst *EI) {
  bool ret = visitNonExInfoInst(EI);

  auto B = EI->getParent();
  auto &typeInfo = blockInfo[B].typeInfo;
  auto res = static_cast<Value *>(EI);
  auto val = EI->getVectorOperand();

  ret |= typeInfo[val]->update(typeInfo[res]);
  auto eTy = MyTy::ptr_cast<MyVectorTy>(typeInfo[val])->getElementTy();
  ret |= addTypeInfo(typeInfo, res, eTy);
  return ret;
}

bool FlowAnalyzer::visitInsertElementInst(InsertElementInst *II) {
  bool ret = visitNonExInfoInst(II);

  auto B = II->getParent();
  auto &typeInfo = blockInfo[B].typeInfo;
  auto res = static_cast<Value *>(II);
  auto val = II->getOperand(0);
  auto elt = II->getOperand(1);

  ret |= typeInfo[val]->update(typeInfo[elt]);
  ret |= addTypeInfo(typeInfo, res, val);
  auto eTy = MyTy::ptr_cast<MyVectorTy>(typeInfo[val])->getElementTy();
  ret |= addTypeInfo(typeInfo, elt, eTy);
  return ret;
}

bool FlowAnalyzer::visitShuffleVectorInst(ShuffleVectorInst* SI) {
  bool ret = visitNonExInfoInst(SI);

  auto B = SI->getParent();
  auto &typeInfo = blockInfo[B].typeInfo;
  auto res = static_cast<Value *>(SI);
  auto v1 = SI->getOperand(0);
  auto v2 = SI->getOperand(1);

  ret |= addTypeInfo(typeInfo, v1, v2);
  ret |= addTypeInfo(typeInfo, res, v1);
  ret |= addTypeInfo(typeInfo, res, v2);

  return ret;
}

bool FlowAnalyzer::visitInsertValueInst(InsertValueInst *II) {
  bool ret = visitNonExInfoInst(II);

  auto B = II->getParent();
  auto &typeInfo = blockInfo[B].typeInfo;
  auto res = static_cast<Value *>(II);
  auto val = II->getAggregateOperand();
  auto elt = II->getInsertedValueOperand();
  SmallVector<Value *> indices;
  for (auto i = 0u; i < II->getNumIndices(); i++) {
    indices.push_back(II->getOperand(i + 2));
  }

  visitAggType(typeInfo, typeInfo[val], elt, indices, 0, ret);
  ret |= addTypeInfo(typeInfo, res, val);
  return ret;
}

bool FlowAnalyzer::visitExtractValueInst(ExtractValueInst* EI) {
  bool ret = visitNonExInfoInst(EI);

  auto B = EI->getParent();
  auto &typeInfo = blockInfo[B].typeInfo;
  auto res = static_cast<Value *>(EI);
  auto val = EI->getAggregateOperand();
  SmallVector<Value *> indices;
  for (auto i = 0u; i < EI->getNumIndices(); i++) {
    indices.push_back(EI->getOperand(i + 1));
  }

  visitAggType(typeInfo, typeInfo[val], res, indices, 0, ret);
  return ret;
}

bool FlowAnalyzer::visitAllocaInst(AllocaInst* AI) {
  auto B = AI->getParent();
  auto &typeInfo = blockInfo[B].typeInfo;
  auto res = static_cast<Value *>(AI);
  auto type = AI->getAllocatedType();

  return addTypeInfoByPointee(typeInfo, res, type);
}

bool FlowAnalyzer::visitLoadInst(LoadInst* LI) {
  auto B = LI->getParent();
  auto &typeInfo = blockInfo[B].typeInfo;
  auto res = static_cast<Value *>(LI);
  auto ptr = LI->getPointerOperand();

  return addTypeInfoByPointee(typeInfo, ptr, res);
}

bool FlowAnalyzer::visitStoreInst(StoreInst* SI) {
  auto B = SI->getParent();
  auto &typeInfo = blockInfo[B].typeInfo;
  auto val = SI->getValueOperand();
  auto ptr = SI->getPointerOperand();

  return addTypeInfoByPointee(typeInfo, ptr, val);
}

bool FlowAnalyzer::visitAtomicCmpXchgInst(AtomicCmpXchgInst* AI) {
  auto B = AI->getParent();
  auto &typeInfo = blockInfo[B].typeInfo;
  auto ptr = AI->getPointerOperand();
  auto cmpVal = AI->getCompareOperand();
  auto newVal = AI->getNewValOperand();

  bool ret = addTypeInfoByPointee(typeInfo, ptr, cmpVal);
  ret |= addTypeInfoByPointee(typeInfo, ptr, newVal);
  return addTypeInfo(typeInfo, cmpVal, newVal) || ret;
}

bool FlowAnalyzer::visitAtomicRWMInst(AtomicRMWInst* AI) {
  auto B = AI->getParent();
  auto &typeInfo = blockInfo[B].typeInfo;
  auto val = AI->getValOperand();
  auto ptr = AI->getPointerOperand();

  return addTypeInfoByPointee(typeInfo, ptr, val);
}

bool FlowAnalyzer::visitGetElementPtrInst(GetElementPtrInst* GI) {
  bool ret = visitNonExInfoInst(GI);

  auto B = GI->getParent();
  auto &typeInfo = blockInfo[B].typeInfo;
  auto res = static_cast<Value *>(GI);
  auto ptr = GI->getPointerOperand();

  ret |= addTypeInfoByPointee(typeInfo, ptr, GI->getSourceElementType());

  if (GI->getNumIndices() == 1) {
    ret |= addTypeInfo(typeInfo, ptr, res);
  } else {
    SmallVector<Value *> indices;
    for (auto i = 0u; i < GI->getNumIndices(); i++) {
      indices.push_back(GI->getOperand(i + 1));
    }
  
    visitAggPtrType(typeInfo, typeInfo[ptr], res, indices, 0, ret);
  }

  
  return ret;
}

bool FlowAnalyzer::visitICmpInst(ICmpInst *II) { 
  bool ret = visitNonExInfoInst(II);
  auto B = II->getParent();
  auto &typeInfo = blockInfo[B].typeInfo;
  auto op1 = II->getOperand(0);
  auto op2 = II->getOperand(1);

  return addTypeInfo(typeInfo, op1, op2) || ret;
}

bool FlowAnalyzer::visitPHINode(PHINode* PN) {
  bool ret = visitNonExInfoInst(PN);

  auto B = PN->getParent();
  auto &typeInfo = blockInfo[B].typeInfo;
  auto res = static_cast<Value *>(PN);

  for (auto i = 0u; i < PN->getNumIncomingValues(); i++) {
    auto val = PN->getIncomingValue(i);
    ret |= addTypeInfo(typeInfo, res, typeInfo[val]);
  }

  return ret;
}

bool FlowAnalyzer::visitSelectInst(SelectInst* SI) {
  bool ret = visitNonExInfoInst(SI);

  auto B = SI->getParent();
  auto &typeInfo = blockInfo[B].typeInfo;
  auto res = static_cast<Value *>(SI);
  auto val1 = SI->getTrueValue();
  auto val2 = SI->getFalseValue();

  ret |= addTypeInfo(typeInfo, res, typeInfo[val1]);
  ret |= addTypeInfo(typeInfo, res, typeInfo[val2]);
  return ret;
}

const DenseMap<Type*, shared_ptr<MyStructTy>>& FlowAnalyzer::getStructInfo() {
  return structInfo;
}

const DenseMap<Value*, shared_ptr<MyTy>>& FlowAnalyzer::getTypeInfo(BasicBlock *B) {
  return blockInfo[B].typeInfo;
}

void FlowAnalyzer::printUpdateLog(Value *V, shared_ptr<MyTy> T) {
  errs() << "Update ";
  V->printAsOperand(errs(), false);
  errs() << " with " << T->toString() << "\n";
}

void FlowAnalyzer::printPtrUpdateLog(Value *V, shared_ptr<MyTy> T) {
  errs() << "Update ptr ";
  V->printAsOperand(errs(), false);
  errs() << " with pointee type " << T->toString() << "\n";
}

void FlowAnalyzer::printUpdateLog(Value *V, Type *T) {
  errs() << "Update ";
  V->printAsOperand(errs(), false);
  errs() << " with ";
  T->print(errs(), false, true);
  errs() << "\n";
}

void FlowAnalyzer::printPtrUpdateLog(Value *V, Type *T) {
  errs() << "Update ptr ";
  V->printAsOperand(errs(), false);
  errs() << " with pointee type ";
  T->print(errs(), false, true);
  errs() << "\n";
}

void FlowAnalyzer::printUpdateLog(Value *V1, Value *V2) {
  errs() << "Update ";
  V1->printAsOperand(errs(), false);
  errs() << " with ";
  V2->printAsOperand(errs(), false);
  errs() << "\n";
}

void FlowAnalyzer::printPtrUpdateLog(Value *P, Value *V) {
  errs() << "Update ptr ";
  P->printAsOperand(errs(), false);
  errs() << " with pointee ";
  V->printAsOperand(errs(), false);
  errs() << "\n";
}
