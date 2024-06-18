//===-- GPUSan.cpp - GPU sanitizer ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/GPUSan.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "gpusan"

cl::opt<bool> UseTags(
    "gpusan-use-tags",
    cl::desc(
        "Use tags to detect use after if the number of allocations is large"),
    cl::init(false));

namespace {
class GPUSanImpl final {
public:
  GPUSanImpl(Module &M) : M(M), Ctx(M.getContext()) {}

  bool instrument();

private:
  bool instrumentGlobals();
  bool instrumentFunction(Function &Fn);
  void instrumentAllocation(Instruction &I, Value &Size);
  void instrumentAllocaInst(AllocaInst &AI);
  void instrumentAccess(Instruction &I, int PtrIdx, Type &AccessTy);
  void instrumentLoadInst(LoadInst &LI);
  void instrumentStoreInst(StoreInst &SI);
  void instrumentGEPInst(GetElementPtrInst &GEP);
  bool instrumentCallInst(CallInst &CI);

  void getOrCreateFn(FunctionCallee &FC, StringRef Name, Type *RetTy,
                     ArrayRef<Type *> ArgTys) {
    if (!FC) {
      auto *NewAllocationFnTy = FunctionType::get(RetTy, ArgTys, false);
      FC = M.getOrInsertFunction(Name, NewAllocationFnTy);
    }
  }

  FunctionCallee getNewAllocationFn() {
    getOrCreateFn(NewAllocationFn, "ompx_new_allocation", PtrTy,
                  {PtrTy, Int64Ty, Int64Ty});
    return NewAllocationFn;
  }
  FunctionCallee getAccessFn() {
    getOrCreateFn(AccessFn, "ompx_check_access", PtrTy,
                  {PtrTy, Int64Ty, Int64Ty});
    return AccessFn;
  }
  FunctionCallee getGEPFn() {
    getOrCreateFn(GEPFn, "ompx_gep", PtrTy, {PtrTy, Int64Ty});
    return GEPFn;
  }
  FunctionCallee getUnpackFn() {
    getOrCreateFn(UnpackFn, "ompx_unpack", PtrTy, {PtrTy});
    return UnpackFn;
  }

  Module &M;
  LLVMContext &Ctx;

  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *IntptrTy = M.getDataLayout().getIntPtrType(Ctx);
  PointerType *PtrTy = PointerType::getUnqual(Ctx);
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);

  const DataLayout &DL = M.getDataLayout();

  FunctionCallee GEPFn;
  FunctionCallee UnpackFn;
  FunctionCallee AccessFn;
  FunctionCallee NewAllocationFn;
};

} // end anonymous namespace

bool GPUSanImpl::instrumentGlobals() {
  return false;
  Function *CTorFn;
  std::tie(CTorFn, std::ignore) = getOrCreateSanitizerCtorAndInitFunctions(
      M, "ompx.ctor", "ompx.init",
      /*InitArgTypes=*/{},
      /*InitArgs=*/{},
      // This callback is invoked when the functions are created the first
      // time. Hook them into the global ctors list in that case:
      [&](Function *Ctor, FunctionCallee) {
        appendToGlobalCtors(M, Ctor, 0, Ctor);
      });
  return true;
}

void GPUSanImpl::instrumentAllocation(Instruction &I, Value &Size) {
  IRBuilder<> IRB(I.getNextNode());
  Value *PlainI = IRB.CreatePointerBitCastOrAddrSpaceCast(&I, PtrTy);
  static int X = 1;
  auto *CB = IRB.CreateCall(getNewAllocationFn(),
                            {PlainI, &Size, ConstantInt::get(Int64Ty, X++)},
                            I.getName() + ".san");
  I.replaceUsesWithIf(IRB.CreatePointerBitCastOrAddrSpaceCast(CB, I.getType()),
                      [=](Use &U) {
                        return U.getUser() != PlainI && U.getUser() != CB &&
                               !isa<LifetimeIntrinsic>(U.getUser());
                      });
}

void GPUSanImpl::instrumentAllocaInst(AllocaInst &AI) {
  auto SizeOrNone = AI.getAllocationSize(DL);
  if (!SizeOrNone)
    llvm_unreachable("TODO");
  Value *Size = ConstantInt::get(Int64Ty, *SizeOrNone);
  instrumentAllocation(AI, *Size);
}

void GPUSanImpl::instrumentAccess(Instruction &I, int PtrIdx, Type &AccessTy) {
  auto TySize = DL.getTypeStoreSize(&AccessTy);
  assert(!TySize.isScalable());
  Value *Size = ConstantInt::get(Int64Ty, TySize.getFixedValue());
  IRBuilder<> IRB(&I);
  Value *PtrOp = I.getOperand(PtrIdx);
  auto *UO = getUnderlyingObject(PtrOp);
  if (isa<GlobalObject>(UO))
    return;
  Value *PlainPtrOp = IRB.CreatePointerBitCastOrAddrSpaceCast(PtrOp, PtrTy);
  static int X = 1;
  auto *CB = IRB.CreateCall(getAccessFn(),
                            {PlainPtrOp, Size, ConstantInt::get(Int64Ty, X++)},
                            I.getName() + ".san");
  I.setOperand(PtrIdx,
               IRB.CreatePointerBitCastOrAddrSpaceCast(CB, PtrOp->getType()));
}

void GPUSanImpl::instrumentLoadInst(LoadInst &LI) {
  instrumentAccess(LI, LoadInst::getPointerOperandIndex(), *LI.getType());
}

void GPUSanImpl::instrumentStoreInst(StoreInst &SI) {
  instrumentAccess(SI, StoreInst::getPointerOperandIndex(),
                   *SI.getValueOperand()->getType());
}

void GPUSanImpl::instrumentGEPInst(GetElementPtrInst &GEP) {
  Value *PtrOp = GEP.getPointerOperand();
  auto *UO = getUnderlyingObject(PtrOp);
  if (isa<GlobalObject>(UO))
    return;
  GEP.setOperand(GetElementPtrInst::getPointerOperandIndex(),
                 Constant::getNullValue(PtrOp->getType()));

  IRBuilder<> IRB(GEP.getNextNode());
  Value *PlainPtrOp = IRB.CreatePointerBitCastOrAddrSpaceCast(PtrOp, PtrTy);
  auto *CB = IRB.CreateCall(getGEPFn(), {PlainPtrOp, UndefValue::get(Int64Ty)},
                            GEP.getName() + ".san");
  GEP.replaceAllUsesWith(
      IRB.CreatePointerBitCastOrAddrSpaceCast(CB, GEP.getType()));
  Value *Offset =
      new PtrToIntInst(&GEP, Int64Ty, GEP.getName() + ".san.offset", CB);
  CB->setArgOperand(1, Offset);
}

bool GPUSanImpl::instrumentCallInst(CallInst &CI) {
  bool Changed = false;
  if (isa<LifetimeIntrinsic>(CI))
    return Changed;
  if (auto *Fn = CI.getCalledFunction()) {
    if ((Fn->isDeclaration() || Fn->getName().starts_with("__kmpc")) &&
        !Fn->getName().starts_with("ompx")) {
      IRBuilder<> IRB(&CI);
      for (int I = 0, E = CI.arg_size(); I != E; ++I) {
        Value *Op = CI.getArgOperand(I);
        if (!Op->getType()->isPointerTy())
          continue;
        auto *UO = getUnderlyingObject(Op);
        if (isa<GlobalObject>(UO))
          continue;
        Value *PlainOp = IRB.CreatePointerBitCastOrAddrSpaceCast(Op, PtrTy);
        auto *CB =
            IRB.CreateCall(getUnpackFn(), {PlainOp}, Op->getName() + ".unpack");
        CI.setArgOperand(
            I, IRB.CreatePointerBitCastOrAddrSpaceCast(CB, Op->getType()));
        Changed = true;
      }
    }
  }
  return Changed;
}

bool GPUSanImpl::instrumentFunction(Function &Fn) {
  bool Changed = false;

  for (auto &I : instructions(Fn)) {
    switch (I.getOpcode()) {
    case Instruction::Alloca:
      instrumentAllocaInst(cast<AllocaInst>(I));
      Changed = true;
      break;
    case Instruction::Load:
      instrumentLoadInst(cast<LoadInst>(I));
      Changed = true;
      break;
    case Instruction::Store:
      instrumentStoreInst(cast<StoreInst>(I));
      Changed = true;
      break;
    case Instruction::GetElementPtr:
      instrumentGEPInst(cast<GetElementPtrInst>(I));
      Changed = true;
      break;
    case Instruction::Call:
      Changed = instrumentCallInst(cast<CallInst>(I));
      break;
    default:
      break;
    }
  }

  return Changed;
}

bool GPUSanImpl::instrument() {
  bool Changed = instrumentGlobals();
  for (Function &Fn : M)
    if (!Fn.getName().contains("ompx") && !Fn.getName().contains("__kmpc"))
      Changed |= instrumentFunction(Fn);
  M.dump();
  return Changed;
}

PreservedAnalyses GPUSanPass::run(Module &M, ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  GPUSanImpl Lowerer(M);
  if (!Lowerer.instrument())
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
