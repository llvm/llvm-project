//===-- AMDGPUPromoteAlloca.cpp - Promote Allocas -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates allocas by either converting them into vectors or
// by migrating them to local address space.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "amdgpu-promote-alloca"

using namespace llvm;

namespace {

class AMDGPUPromoteAlloca : public FunctionPass,
                       public InstVisitor<AMDGPUPromoteAlloca> {

  static char ID;
  Module *Mod;
  const AMDGPUSubtarget &ST;
  int LocalMemAvailable;

public:
  AMDGPUPromoteAlloca(const AMDGPUSubtarget &st) : FunctionPass(ID), ST(st),
                                                   LocalMemAvailable(0) { }
  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;
  const char *getPassName() const override { return "AMDGPU Promote Alloca"; }
  void visitAlloca(AllocaInst &I);
};

} // End anonymous namespace

char AMDGPUPromoteAlloca::ID = 0;

bool AMDGPUPromoteAlloca::doInitialization(Module &M) {
  Mod = &M;
  return false;
}

bool AMDGPUPromoteAlloca::runOnFunction(Function &F) {

  FunctionType *FTy = F.getFunctionType();

  LocalMemAvailable = ST.getLocalMemorySize();


  // If the function has any arguments in the local address space, then it's
  // possible these arguments require the entire local memory space, so
  // we cannot use local memory in the pass.
  for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i) {
    Type *ParamTy = FTy->getParamType(i);
    if (ParamTy->isPointerTy() &&
        ParamTy->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
      LocalMemAvailable = 0;
      DEBUG(dbgs() << "Function has local memory argument.  Promoting to "
                      "local memory disabled.\n");
      break;
    }
  }

  if (LocalMemAvailable > 0) {
    // Check how much local memory is being used by global objects
    for (Module::global_iterator I = Mod->global_begin(),
                                 E = Mod->global_end(); I != E; ++I) {
      GlobalVariable *GV = &*I;
      PointerType *GVTy = GV->getType();
      if (GVTy->getAddressSpace() != AMDGPUAS::LOCAL_ADDRESS)
        continue;
      for (Value::use_iterator U = GV->use_begin(),
                               UE = GV->use_end(); U != UE; ++U) {
        Instruction *Use = dyn_cast<Instruction>(*U);
        if (!Use)
          continue;
        if (Use->getParent()->getParent() == &F)
          LocalMemAvailable -=
              Mod->getDataLayout().getTypeAllocSize(GVTy->getElementType());
      }
    }
  }

  LocalMemAvailable = std::max(0, LocalMemAvailable);
  DEBUG(dbgs() << LocalMemAvailable << "bytes free in local memory.\n");

  visit(F);

  return false;
}

static VectorType *arrayTypeToVecType(Type *ArrayTy) {
  return VectorType::get(ArrayTy->getArrayElementType(),
                         ArrayTy->getArrayNumElements());
}

static Value *
calculateVectorIndex(Value *Ptr,
                     const std::map<GetElementPtrInst *, Value *> &GEPIdx) {
  if (isa<AllocaInst>(Ptr))
    return Constant::getNullValue(Type::getInt32Ty(Ptr->getContext()));

  GetElementPtrInst *GEP = cast<GetElementPtrInst>(Ptr);

  auto I = GEPIdx.find(GEP);
  return I == GEPIdx.end() ? nullptr : I->second;
}

static Value* GEPToVectorIndex(GetElementPtrInst *GEP) {
  // FIXME we only support simple cases
  if (GEP->getNumOperands() != 3)
    return NULL;

  ConstantInt *I0 = dyn_cast<ConstantInt>(GEP->getOperand(1));
  if (!I0 || !I0->isZero())
    return NULL;

  return GEP->getOperand(2);
}

// Not an instruction handled below to turn into a vector.
//
// TODO: Check isTriviallyVectorizable for calls and handle other
// instructions.
static bool canVectorizeInst(Instruction *Inst, User *User) {
  switch (Inst->getOpcode()) {
  case Instruction::Load:
  case Instruction::BitCast:
  case Instruction::AddrSpaceCast:
    return true;
  case Instruction::Store: {
    // Must be the stored pointer operand, not a stored value.
    StoreInst *SI = cast<StoreInst>(Inst);
    return SI->getPointerOperand() == User;
  }
  default:
    return false;
  }
}

static bool tryPromoteAllocaToVector(AllocaInst *Alloca) {
  Type *AllocaTy = Alloca->getAllocatedType();

  DEBUG(dbgs() << "Alloca Candidate for vectorization \n");

  // FIXME: There is no reason why we can't support larger arrays, we
  // are just being conservative for now.
  if (!AllocaTy->isArrayTy() ||
      AllocaTy->getArrayElementType()->isVectorTy() ||
      AllocaTy->getArrayNumElements() > 4) {

    DEBUG(dbgs() << "  Cannot convert type to vector");
    return false;
  }

  std::map<GetElementPtrInst*, Value*> GEPVectorIdx;
  std::vector<Value*> WorkList;
  for (User *AllocaUser : Alloca->users()) {
    GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(AllocaUser);
    if (!GEP) {
      if (!canVectorizeInst(cast<Instruction>(AllocaUser), Alloca))
        return false;

      WorkList.push_back(AllocaUser);
      continue;
    }

    Value *Index = GEPToVectorIndex(GEP);

    // If we can't compute a vector index from this GEP, then we can't
    // promote this alloca to vector.
    if (!Index) {
      DEBUG(dbgs() << "  Cannot compute vector index for GEP " << *GEP << '\n');
      return false;
    }

    GEPVectorIdx[GEP] = Index;
    for (User *GEPUser : AllocaUser->users()) {
      if (!canVectorizeInst(cast<Instruction>(GEPUser), AllocaUser))
        return false;

      WorkList.push_back(GEPUser);
    }
  }

  VectorType *VectorTy = arrayTypeToVecType(AllocaTy);

  DEBUG(dbgs() << "  Converting alloca to vector "
        << *AllocaTy << " -> " << *VectorTy << '\n');

  for (std::vector<Value*>::iterator I = WorkList.begin(),
                                     E = WorkList.end(); I != E; ++I) {
    Instruction *Inst = cast<Instruction>(*I);
    IRBuilder<> Builder(Inst);
    switch (Inst->getOpcode()) {
    case Instruction::Load: {
      Value *Ptr = Inst->getOperand(0);
      Value *Index = calculateVectorIndex(Ptr, GEPVectorIdx);
      Value *BitCast = Builder.CreateBitCast(Alloca, VectorTy->getPointerTo(0));
      Value *VecValue = Builder.CreateLoad(BitCast);
      Value *ExtractElement = Builder.CreateExtractElement(VecValue, Index);
      Inst->replaceAllUsesWith(ExtractElement);
      Inst->eraseFromParent();
      break;
    }
    case Instruction::Store: {
      Value *Ptr = Inst->getOperand(1);
      Value *Index = calculateVectorIndex(Ptr, GEPVectorIdx);
      Value *BitCast = Builder.CreateBitCast(Alloca, VectorTy->getPointerTo(0));
      Value *VecValue = Builder.CreateLoad(BitCast);
      Value *NewVecValue = Builder.CreateInsertElement(VecValue,
                                                       Inst->getOperand(0),
                                                       Index);
      Builder.CreateStore(NewVecValue, BitCast);
      Inst->eraseFromParent();
      break;
    }
    case Instruction::BitCast:
    case Instruction::AddrSpaceCast:
      break;

    default:
      Inst->dump();
      llvm_unreachable("Inconsistency in instructions promotable to vector");
    }
  }
  return true;
}

static bool collectUsesWithPtrTypes(Value *Val, std::vector<Value*> &WorkList) {
  bool Success = true;
  for (User *User : Val->users()) {
    if(std::find(WorkList.begin(), WorkList.end(), User) != WorkList.end())
      continue;
    if (CallInst *CI = dyn_cast<CallInst>(User)) {
      // TODO: We might be able to handle some cases where the callee is a
      // constantexpr bitcast of a function.
      if (!CI->getCalledFunction())
        return false;

      WorkList.push_back(User);
      continue;
    }

    // FIXME: Correctly handle ptrtoint instructions.
    Instruction *UseInst = dyn_cast<Instruction>(User);
    if (UseInst && UseInst->getOpcode() == Instruction::PtrToInt)
      return false;

    if (StoreInst *SI = dyn_cast_or_null<StoreInst>(UseInst)) {
      // Reject if the stored value is not the pointer operand.
      if (SI->getPointerOperand() != Val)
        return false;
    }

    if (!User->getType()->isPointerTy())
      continue;

    WorkList.push_back(User);

    Success &= collectUsesWithPtrTypes(User, WorkList);
  }
  return Success;
}

void AMDGPUPromoteAlloca::visitAlloca(AllocaInst &I) {
  if (!I.isStaticAlloca())
    return;

  IRBuilder<> Builder(&I);

  // First try to replace the alloca with a vector
  Type *AllocaTy = I.getAllocatedType();

  DEBUG(dbgs() << "Trying to promote " << I << '\n');

  if (tryPromoteAllocaToVector(&I))
    return;

  DEBUG(dbgs() << " alloca is not a candidate for vectorization.\n");

  // FIXME: This is the maximum work group size.  We should try to get
  // value from the reqd_work_group_size function attribute if it is
  // available.
  unsigned WorkGroupSize = 256;
  int AllocaSize =
      WorkGroupSize * Mod->getDataLayout().getTypeAllocSize(AllocaTy);

  if (AllocaSize > LocalMemAvailable) {
    DEBUG(dbgs() << " Not enough local memory to promote alloca.\n");
    return;
  }

  std::vector<Value*> WorkList;

  if (!collectUsesWithPtrTypes(&I, WorkList)) {
    DEBUG(dbgs() << " Do not know how to convert all uses\n");
    return;
  }

  DEBUG(dbgs() << "Promoting alloca to local memory\n");
  LocalMemAvailable -= AllocaSize;

  Type *GVTy = ArrayType::get(I.getAllocatedType(), 256);
  GlobalVariable *GV = new GlobalVariable(
      *Mod, GVTy, false, GlobalValue::ExternalLinkage, 0, I.getName(), 0,
      GlobalVariable::NotThreadLocal, AMDGPUAS::LOCAL_ADDRESS);

  FunctionType *FTy = FunctionType::get(
      Type::getInt32Ty(Mod->getContext()), false);
  AttributeSet AttrSet;
  AttrSet.addAttribute(Mod->getContext(), 0, Attribute::ReadNone);

  Value *ReadLocalSizeY = Mod->getOrInsertFunction(
      "llvm.r600.read.local.size.y", FTy, AttrSet);
  Value *ReadLocalSizeZ = Mod->getOrInsertFunction(
      "llvm.r600.read.local.size.z", FTy, AttrSet);
  Value *ReadTIDIGX = Mod->getOrInsertFunction(
      "llvm.r600.read.tidig.x", FTy, AttrSet);
  Value *ReadTIDIGY = Mod->getOrInsertFunction(
      "llvm.r600.read.tidig.y", FTy, AttrSet);
  Value *ReadTIDIGZ = Mod->getOrInsertFunction(
      "llvm.r600.read.tidig.z", FTy, AttrSet);

  Value *TCntY = Builder.CreateCall(ReadLocalSizeY, {});
  Value *TCntZ = Builder.CreateCall(ReadLocalSizeZ, {});
  Value *TIdX = Builder.CreateCall(ReadTIDIGX, {});
  Value *TIdY = Builder.CreateCall(ReadTIDIGY, {});
  Value *TIdZ = Builder.CreateCall(ReadTIDIGZ, {});

  Value *Tmp0 = Builder.CreateMul(TCntY, TCntZ);
  Tmp0 = Builder.CreateMul(Tmp0, TIdX);
  Value *Tmp1 = Builder.CreateMul(TIdY, TCntZ);
  Value *TID = Builder.CreateAdd(Tmp0, Tmp1);
  TID = Builder.CreateAdd(TID, TIdZ);

  std::vector<Value*> Indices;
  Indices.push_back(Constant::getNullValue(Type::getInt32Ty(Mod->getContext())));
  Indices.push_back(TID);

  Value *Offset = Builder.CreateGEP(GVTy, GV, Indices);
  I.mutateType(Offset->getType());
  I.replaceAllUsesWith(Offset);
  I.eraseFromParent();

  for (std::vector<Value*>::iterator i = WorkList.begin(),
                                     e = WorkList.end(); i != e; ++i) {
    Value *V = *i;
    CallInst *Call = dyn_cast<CallInst>(V);
    if (!Call) {
      Type *EltTy = V->getType()->getPointerElementType();
      PointerType *NewTy = PointerType::get(EltTy, AMDGPUAS::LOCAL_ADDRESS);

      // The operand's value should be corrected on its own.
      if (isa<AddrSpaceCastInst>(V))
        continue;

      // FIXME: It doesn't really make sense to try to do this for all
      // instructions.
      V->mutateType(NewTy);
      continue;
    }

    IntrinsicInst *Intr = dyn_cast<IntrinsicInst>(Call);
    if (!Intr) {
      std::vector<Type*> ArgTypes;
      for (unsigned ArgIdx = 0, ArgEnd = Call->getNumArgOperands();
                                ArgIdx != ArgEnd; ++ArgIdx) {
        ArgTypes.push_back(Call->getArgOperand(ArgIdx)->getType());
      }
      Function *F = Call->getCalledFunction();
      FunctionType *NewType = FunctionType::get(Call->getType(), ArgTypes,
                                                F->isVarArg());
      Constant *C = Mod->getOrInsertFunction((F->getName() + ".local").str(),
                                             NewType, F->getAttributes());
      Function *NewF = cast<Function>(C);
      Call->setCalledFunction(NewF);
      continue;
    }

    Builder.SetInsertPoint(Intr);
    switch (Intr->getIntrinsicID()) {
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
      // These intrinsics are for address space 0 only
      Intr->eraseFromParent();
      continue;
    case Intrinsic::memcpy: {
      MemCpyInst *MemCpy = cast<MemCpyInst>(Intr);
      Builder.CreateMemCpy(MemCpy->getRawDest(), MemCpy->getRawSource(),
                           MemCpy->getLength(), MemCpy->getAlignment(),
                           MemCpy->isVolatile());
      Intr->eraseFromParent();
      continue;
    }
    case Intrinsic::memset: {
      MemSetInst *MemSet = cast<MemSetInst>(Intr);
      Builder.CreateMemSet(MemSet->getRawDest(), MemSet->getValue(),
                           MemSet->getLength(), MemSet->getAlignment(),
                           MemSet->isVolatile());
      Intr->eraseFromParent();
      continue;
    }
    default:
      Intr->dump();
      llvm_unreachable("Don't know how to promote alloca intrinsic use.");
    }
  }
}

FunctionPass *llvm::createAMDGPUPromoteAlloca(const AMDGPUSubtarget &ST) {
  return new AMDGPUPromoteAlloca(ST);
}
