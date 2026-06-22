//===- ValueEnumerator.cpp - AIR value/type/metadata enumerator -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Assigns the contiguous indices the AIR bitcode uses for types, global
// values, and module-level constants. Distinct from LLVM's in-tree
// ValueEnumerator because AIR uses typed POINTER records: the type table
// stores (Type *, pointee) pairs so the same opaque pointer type can take
// several slots, one per inferred pointee.
//
//===----------------------------------------------------------------------===//

#include "ValueEnumerator.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace llvm {
namespace metal {

ValueEnumerator::ValueEnumerator(Module &M, const PointeeTypeMap &PTM)
    : PTM(PTM) {
  auto &Ctx = M.getContext();

  Type *PtrAs0 = PointerType::get(Ctx, 0);
  for (auto &F : M)
    if (!F.isDeclaration()) {
      inferredPointee[PtrAs0] = F.getFunctionType();
      break;
    }

  for (auto &F : M)
    for (auto &Arg : F.args())
      if (Arg.getType()->isPointerTy() && !inferredPointee.count(Arg.getType()))
        if (auto *Ty =
                PointeeTypeMap::inferFromUsage(const_cast<Argument *>(&Arg)))
          inferredPointee[Arg.getType()] = Ty;

  auto applyPTMOverride = [&](Value *V) {
    Type *T = PTM.get(V);
    if (!T)
      return;
    if (!V->getType()->isPointerTy() || isa<GlobalVariable>(V) ||
        inferredPointee.count(V->getType()))
      return;
    if (auto *ST = dyn_cast<StructType>(T))
      if (ST->isOpaque())
        return;
    if (T->isPointerTy())
      return;
    inferredPointee[V->getType()] = T;
  };
  for (auto &GV : M.globals())
    applyPTMOverride(&GV);
  for (auto &F : M) {
    for (auto &Arg : F.args())
      applyPTMOverride(&Arg);
    for (auto &BB : F)
      for (auto &I : BB)
        applyPTMOverride(&I);
  }

  addType(Type::getVoidTy(Ctx));
  addType(Type::getFloatTy(Ctx));

  {
    StructType *EventTy = StructType::getTypeByName(Ctx, kEventTypeName);
    if (EventTy) {
      addType(EventTy);
      auto *PtrAs3 = PointerType::get(Ctx, 3);
      inferredPointee[PtrAs3] = EventTy;
      ptrTypeIdx(PtrAs3, EventTy);
    }
  }

  for (auto &F : M)
    if (!F.isDeclaration())
      addFunctionType(F.getFunctionType(), &F);
  for (auto &F : M)
    if (F.isDeclaration())
      addFunctionType(F.getFunctionType(), &F);

  for (auto &F : M)
    if (!F.isDeclaration())
      ptrTypeIdx(PtrAs0, F.getFunctionType());

  addType(Type::getMetadataTy(Ctx));
  addType(Type::getLabelTy(Ctx));

  for (auto &GV : M.globals()) {
    addType(GV.getValueType());
    globalPtrTypeIdx(&GV);
  }

  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        addType(I.getType());

        for (auto &Op : I.operands())
          if (!isa<BasicBlock>(Op))
            addType(Op->getType());

        if (auto *SV = dyn_cast<ShuffleVectorInst>(&I))
          addType(SV->getShuffleMaskForBitcode()->getType());

        if (auto *AI = dyn_cast<AllocaInst>(&I)) {
          addType(AI->getAllocatedType());
          ptrTypeIdx(PointerType::get(M.getContext(), 0),
                     AI->getAllocatedType());
        }

        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          if (isa<UndefValue>(SI->getPointerOperand()))
            ptrTypeIdx(SI->getPointerOperand()->getType(),
                       SI->getValueOperand()->getType());
        }
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (isa<UndefValue>(LI->getPointerOperand()))
            ptrTypeIdx(LI->getPointerOperand()->getType(), LI->getType());
        }
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          if (isa<UndefValue>(GEP->getPointerOperand()))
            ptrTypeIdx(GEP->getPointerOperand()->getType(),
                       GEP->getSourceElementType());
        }

        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          if (GEP->getType()->isPointerTy()) {
            Type *ResultPointee = GEP->getResultElementType();
            unsigned AddrSpace = GEP->getType()->getPointerAddressSpace();
            if (AddrSpace != 3 || !ResultPointee->isIntegerTy(8)) {
              if (auto *PtmTy = PTM.get(GEP))
                ResultPointee = PtmTy;
            }
            ptrTypeIdx(PointerType::get(M.getContext(), AddrSpace),
                       ResultPointee);
          }
        }

        if (auto *PN = dyn_cast<PHINode>(&I)) {
          if (PN->getType()->isPointerTy()) {
            unsigned AddrSpace = PN->getType()->getPointerAddressSpace();
            Type *Pointee = PTM.get(PN);
            if (!Pointee)
              Pointee = Type::getFloatTy(M.getContext());
            ptrTypeIdx(PointerType::get(M.getContext(), AddrSpace), Pointee);
          }
        }

        if (!isa<AllocaInst>(&I) && !isa<GetElementPtrInst>(&I) &&
            I.getType()->isPointerTy())
          if (auto *PtmTy = PTM.get(&I))
            ptrTypeIdx(PointerType::get(M.getContext(),
                                        I.getType()->getPointerAddressSpace()),
                       PtmTy);
      }
    }
  }

  for (auto &GV : M.globals()) {
    globalValueMap[&GV] = globalValues.size();
    globalValues.push_back(&GV);
  }

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    globalValueMap[&F] = globalValues.size();
    globalValues.push_back(&F);
  }
  for (auto &F : M) {
    if (!F.isDeclaration())
      continue;
    globalValueMap[&F] = globalValues.size();
    globalValues.push_back(&F);
  }

  for (auto &NMD : M.named_metadata())
    for (unsigned I = 0; I < NMD.getNumOperands(); I++)
      collectMetadataConstants(NMD.getOperand(I));

  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB) {
        SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
        I.getAllMetadataOtherThanDebugLoc(MDs);
        for (auto &P : MDs)
          collectMetadataConstants(P.second);
      }

  for (auto &GV : M.globals())
    if (GV.hasInitializer())
      addModuleConstant(GV.getInitializer());

  auto addSubConstants = [&](const Value *Op) {
    auto *C = dyn_cast<Constant>(Op);
    if (!C || isa<GlobalValue>(C))
      return;
    if (isa<ConstantArray>(C) || isa<ConstantStruct>(C) ||
        isa<ConstantVector>(C)) {
      for (unsigned J = 0; J < C->getNumOperands(); J++)
        if (auto *OC = dyn_cast<Constant>(C->getOperand(J)))
          addModuleConstant(OC);
    }
    if (auto *CDA = dyn_cast<ConstantDataArray>(C)) {
      for (unsigned J = 0; J < CDA->getNumElements(); J++)
        addModuleConstant(CDA->getElementAsConstant(J));
    }
  };
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        for (auto &Op : I.operands())
          addSubConstants(Op);

        if (auto *SV = dyn_cast<ShuffleVectorInst>(&I))
          addSubConstants(SV->getShuffleMaskForBitcode());
      }
    }
  }

  frozen = true;
}

unsigned ValueEnumerator::typeIdx(Type *T) {
  if (isa<PointerType>(T))
    return ptrTypeIdx(T, pointeeType(T));
  TypeEntry E{T, nullptr};
  auto It = typeMap.find(E);
  if (It != typeMap.end())
    return It->second;
  return addType(T);
}

unsigned ValueEnumerator::ptrTypeIdxForValue(const Value *V) {
  Type *Pointee = nullptr;
  if (auto *Ty = PTM.get(const_cast<Value *>(V)))
    Pointee = Ty;
  if (!Pointee)
    Pointee = pointeeType(V->getType());
  if (frozen && !typeMap.count(TypeEntry{V->getType(), Pointee})) {
    std::string Msg;
    raw_string_ostream OS(Msg);
    OS << "AIRWriter: pointer value '" << V->getName()
       << "' resolves to an unregistered typed-pointer entry at emit time; "
          "its producing instruction was not enumerated (would desync the "
          "frozen type table). Add it to ValueEnumerator's registration loop.";
    report_fatal_error(StringRef(OS.str()));
  }
  return ptrTypeIdx(V->getType(), Pointee);
}

unsigned ValueEnumerator::typeIdxForValue(const Value *V) {
  if (auto *F = dyn_cast<Function>(V))
    return ptrTypeIdx(F->getType(), F->getFunctionType());
  if (auto *GV = dyn_cast<GlobalVariable>(V))
    return globalPtrTypeIdx(GV);
  if (V->getType()->isPointerTy())
    return ptrTypeIdxForValue(V);
  return typeIdx(V->getType());
}

unsigned ValueEnumerator::ptrTypeIdx(Type *PtrTy, Type *Pointee) {
  TypeEntry E{PtrTy, Pointee};
  auto It = typeMap.find(E);
  if (It != typeMap.end())
    return It->second;

  addType(Pointee);
  return addEntry(E);
}

unsigned ValueEnumerator::globalPtrTypeIdx(const GlobalVariable *GV) {
  return ptrTypeIdx(GV->getType(), GV->getValueType());
}

unsigned ValueEnumerator::globalIdx(const Value *V) const {
  auto It = globalValueMap.find(V);
  assert(It != globalValueMap.end());
  return It->second;
}

unsigned ValueEnumerator::moduleConstIdx(const Constant *C) const {
  auto It = moduleConstMap.find(C);
  assert(It != moduleConstMap.end());
  return globalValues.size() + It->second;
}

bool ValueEnumerator::hasModuleConst(const Constant *C) const {
  return moduleConstMap.count(C);
}

Type *ValueEnumerator::pointeeType(Type *PtrTy) const {
  auto It = inferredPointee.find(PtrTy);
  if (It != inferredPointee.end())
    return It->second;
  if (auto *PT = dyn_cast<PointerType>(PtrTy)) {
    unsigned AddrSpace = PT->getAddressSpace();
    if (AddrSpace == 1 || AddrSpace == 3)
      return Type::getFloatTy(PtrTy->getContext());
  }
  return Type::getFloatTy(PtrTy->getContext());
}

Type *ValueEnumerator::pointeeTypeForValue(const Value *V) const {
  if (auto *Ty = PTM.get(const_cast<Value *>(V)))
    return Ty;
  if (auto *Ty = PointeeTypeMap::inferFromUsage(const_cast<Value *>(V)))
    return Ty;
  return pointeeType(V->getType());
}

unsigned ValueEnumerator::addEntry(TypeEntry E) {
  auto It = typeMap.find(E);
  if (It != typeMap.end())
    return It->second;
  unsigned Idx = types.size();
  typeMap[E] = Idx;
  types.push_back(E);
  return Idx;
}

unsigned ValueEnumerator::addType(Type *T) {
  if (isa<PointerType>(T))
    return ptrTypeIdx(T, pointeeType(T));

  if (auto *FT = dyn_cast<FunctionType>(T)) {
    TypeEntry E{T, nullptr};
    auto It = typeMap.find(E);
    if (It != typeMap.end())
      return It->second;
    return addFunctionType(FT, nullptr);
  }

  TypeEntry E{T, nullptr};
  auto It = typeMap.find(E);
  if (It != typeMap.end())
    return It->second;

  if (auto *VT = dyn_cast<VectorType>(T)) {
    addType(VT->getElementType());
  } else if (auto *AT = dyn_cast<ArrayType>(T)) {
    addType(AT->getElementType());
  } else if (auto *ST = dyn_cast<StructType>(T)) {
    if (!ST->isOpaque())
      for (auto *ET : ST->elements())
        addType(ET);
  }

  It = typeMap.find(E);
  if (It != typeMap.end())
    return It->second;

  return addEntry(E);
}

unsigned ValueEnumerator::addFunctionType(FunctionType *FT, const Function *F) {
  TypeEntry E{FT, nullptr};
  auto It = typeMap.find(E);
  if (It != typeMap.end())
    return It->second;

  SmallVector<unsigned, 8> ParamIndices;

  if (FT->getReturnType()->isPointerTy()) {
    Type *RetPointee = nullptr;

    if (F && F->isDeclaration() &&
        F->getName().starts_with("air.simdgroup_async_copy")) {
      auto &Ctx = F->getContext();
      if (auto *EventTy = StructType::getTypeByName(Ctx, kEventTypeName))
        RetPointee = EventTy;
    }

    if (!RetPointee && F && F->isDeclaration()) {
      for (auto *U : F->users()) {
        if (auto *CI = dyn_cast<CallInst>(U)) {
          if (auto *Ty = PTM.get(const_cast<CallInst *>(CI))) {
            RetPointee = Ty;
            break;
          }
        }
      }
    }
    if (!RetPointee)
      RetPointee = pointeeType(FT->getReturnType());
    unsigned RetIdx = ptrTypeIdx(FT->getReturnType(), RetPointee);
    funcTypeReturnIndex[FT] = RetIdx;
  } else {
    addType(FT->getReturnType());
  }

  for (unsigned I = 0; I < FT->getNumParams(); I++) {
    Type *PT = FT->getParamType(I);
    if (!PT->isPointerTy()) {
      ParamIndices.push_back(addType(PT));
      continue;
    }

    Type *Pointee = nullptr;

    if (F && F->isDeclaration()) {
      StringRef Name = F->getName();
      if (Name.starts_with("air.atomic.")) {
        unsigned AddrSpace = cast<PointerType>(PT)->getAddressSpace();
        if (AddrSpace == 1 || AddrSpace == 3) {
          if (Name.ends_with(".i32"))
            Pointee = Type::getInt32Ty(F->getContext());
          else if (Name.ends_with(".f32"))
            Pointee = Type::getFloatTy(F->getContext());
        }
      }

      if (Name.starts_with("air.simdgroup_async_copy")) {
        unsigned AddrSpace = cast<PointerType>(PT)->getAddressSpace();
        if (AddrSpace == 1 || AddrSpace == 3)
          Pointee = Type::getInt8Ty(F->getContext());
      }

      if (Name == "air.wait_simdgroup_events" && I == 1) {
        auto *PtrAs3 = PointerType::get(F->getContext(), 3);
        Pointee = PtrAs3;
        ParamIndices.push_back(ptrTypeIdx(PT, Pointee));
        continue;
      }
    }

    if (!Pointee && F && !F->isDeclaration() && I < F->arg_size())
      Pointee = pointeeTypeForValue(F->getArg(I));

    if (!Pointee && F && F->isDeclaration()) {
      for (auto &Sib : *F->getParent()) {
        if (Sib.getFunctionType() != FT)
          continue;
        for (auto *U : Sib.users()) {
          if (auto *CI = dyn_cast<CallInst>(U)) {
            if (CI->getCalledFunction() == &Sib && I < CI->arg_size()) {
              if (isa<UndefValue>(CI->getArgOperand(I)))
                continue;
              Pointee = pointeeTypeForValue(CI->getArgOperand(I));
              if (Pointee)
                break;
            }
          }
        }
        if (Pointee)
          break;
      }
    }
    if (!Pointee)
      Pointee = pointeeType(PT);
    ParamIndices.push_back(ptrTypeIdx(PT, Pointee));
  }

  funcTypeParamIndices[FT] = ParamIndices;

  It = typeMap.find(E);
  if (It != typeMap.end())
    return It->second;

  return addEntry(E);
}

void ValueEnumerator::addModuleConstant(const Constant *C) {
  if (moduleConstMap.count(C) || globalValueMap.count(C))
    return;

  if (auto *CDA = dyn_cast<ConstantDataSequential>(C)) {
    for (unsigned I = 0; I < CDA->getNumElements(); I++)
      addModuleConstant(CDA->getElementAsConstant(I));
  }
  for (unsigned I = 0; I < C->getNumOperands(); I++)
    if (auto *OC = dyn_cast<Constant>(C->getOperand(I)))
      addModuleConstant(OC);
  addType(C->getType());
  moduleConstMap[C] = moduleConstants.size();
  moduleConstants.push_back(C);
}

void ValueEnumerator::collectMetadataConstants(const MDNode *N) {
  SmallPtrSet<const MDNode *, 16> Seen;
  collectMetadataConstants(N, Seen);
}

void ValueEnumerator::collectMetadataConstants(
    const MDNode *N, SmallPtrSetImpl<const MDNode *> &Seen) {

  if (!N || !Seen.insert(N).second)
    return;
  for (unsigned I = 0; I < N->getNumOperands(); I++) {
    if (auto *VAM = dyn_cast_or_null<ValueAsMetadata>(N->getOperand(I)))
      if (auto *C = dyn_cast<Constant>(VAM->getValue()))
        addModuleConstant(C);
    if (auto *Sub = dyn_cast_or_null<MDNode>(N->getOperand(I)))
      collectMetadataConstants(Sub, Seen);
  }
}

}
}
