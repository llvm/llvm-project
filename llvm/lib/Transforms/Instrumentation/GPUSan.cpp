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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
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

enum PtrOrigin {
  UNKNOWN,
  LOCAL,
  GLOBAL,
  SYSTEM,
  NONE,
};

static std::string getSuffix(PtrOrigin PO) {
  switch (PO) {
  case UNKNOWN:
    return "";
  case LOCAL:
    return "_local";
  case GLOBAL:
    return "_global";
  default:
    break;
  }
  llvm_unreachable("Bad pointer origin!");
}

class GPUSanImpl final {
public:
  GPUSanImpl(Module &M, FunctionAnalysisManager &FAM)
      : M(M), FAM(FAM), Ctx(M.getContext()) {}

  bool instrument();

private:
  bool instrumentGlobals();
  bool instrumentFunction(Function &Fn);
  Value *instrumentAllocation(Instruction &I, Value &Size, FunctionCallee Fn);
  Value *instrumentAllocaInst(LoopInfo &LI, AllocaInst &AI);
  void instrumentAccess(LoopInfo &LI, Instruction &I, int PtrIdx,
                        Type &AccessTy, bool IsRead);
  void instrumentLoadInst(LoopInfo &LI, LoadInst &LoadI);
  void instrumentStoreInst(LoopInfo &LI, StoreInst &StoreI);
  void instrumentGEPInst(LoopInfo &LI, GetElementPtrInst &GEP);
  bool instrumentCallInst(LoopInfo &LI, CallInst &CI);
  void
  instrumentReturns(SmallVectorImpl<std::pair<AllocaInst *, Value *>> &Allocas,
                    SmallVectorImpl<ReturnInst *> &Returns);

  Value *getPC(IRBuilder<> &IRB);
  Value *getFunctionName(IRBuilder<> &IRB);
  Value *getFileName(IRBuilder<> &IRB);
  Value *getLineNo(IRBuilder<> &IRB);
  PtrOrigin getPtrOrigin(LoopInfo &LI, Value *Ptr);

  FunctionCallee getOrCreateFn(FunctionCallee &FC, StringRef Name, Type *RetTy,
                               ArrayRef<Type *> ArgTys) {
    if (!FC) {
      auto *NewAllocationFnTy = FunctionType::get(RetTy, ArgTys, false);
      FC = M.getOrInsertFunction(Name, NewAllocationFnTy);
    }
    return FC;
  }

  FunctionCallee getNewFn(PtrOrigin PO) {
    assert(PO <= GLOBAL && "Origin does not need handling.");
    return getOrCreateFn(NewFn[PO], "ompx_new" + getSuffix(PO), PtrTy,
                         {PtrTy, Int64Ty, Int64Ty, Int64Ty});
  }
  FunctionCallee getFreeFn(PtrOrigin PO) {
    assert(PO <= GLOBAL && "Origin does not need handling.");
    return getOrCreateFn(FreeFn[PO], "ompx_free" + getSuffix(PO), VoidTy,
                         {PtrTy, Int64Ty});
  }
  FunctionCallee getFreeNLocalFn() {
    return getOrCreateFn(FreeNLocal, "ompx_free_local_n", VoidTy, {Int32Ty});
  }
  FunctionCallee getCheckFn(PtrOrigin PO) {
    assert(PO <= GLOBAL && "Origin does not need handling.");
    return getOrCreateFn(
        CheckFn[PO], "ompx_check" + getSuffix(PO), PtrTy,
        {PtrTy, Int64Ty, Int64Ty, Int64Ty, PtrTy, PtrTy, Int64Ty});
  }
  FunctionCallee getGEPFn(PtrOrigin PO) {
    assert(PO <= GLOBAL && "Origin does not need handling.");
    return getOrCreateFn(GEPFn[PO], "ompx_gep" + getSuffix(PO), PtrTy,
                         {PtrTy, Int64Ty, Int64Ty});
  }
  FunctionCallee getUnpackFn(PtrOrigin PO) {
    assert(PO <= GLOBAL && "Origin does not need handling.");
    return getOrCreateFn(UnpackFn[PO], "ompx_unpack" + getSuffix(PO), PtrTy,
                         {PtrTy, Int64Ty});
  }
  FunctionCallee getLeakCheckFn() {
    FunctionCallee LeakCheckFn;
    return getOrCreateFn(LeakCheckFn, "ompx_leak_check", VoidTy, {});
  }

  Module &M;
  FunctionAnalysisManager &FAM;
  LLVMContext &Ctx;
  bool HasAllocas;

  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *IntptrTy = M.getDataLayout().getIntPtrType(Ctx);
  PointerType *PtrTy = PointerType::getUnqual(Ctx);
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);

  const DataLayout &DL = M.getDataLayout();

  FunctionCallee NewFn[3];
  FunctionCallee GEPFn[3];
  FunctionCallee FreeFn[3];
  FunctionCallee CheckFn[3];
  FunctionCallee UnpackFn[3];
  FunctionCallee FreeNLocal;

  StringMap<Value *> GlobalStringMap;
};

} // end anonymous namespace

Value *GPUSanImpl::getPC(IRBuilder<> &IRB) {
  return IRB.CreateIntrinsic(Int64Ty, Intrinsic::amdgcn_s_getpc, {}, nullptr,
                             "PC");
}
Value *GPUSanImpl::getFunctionName(IRBuilder<> &IRB) {
  const auto &DLoc = IRB.getCurrentDebugLocation();
  StringRef FnName = IRB.GetInsertPoint()->getFunction()->getName();
  if (DLoc && DLoc.get()) {
    StringRef SubprogramName = DLoc.get()->getSubprogramLinkageName();
    if (!SubprogramName.empty())
      FnName = SubprogramName;
  }
  StringRef Name = FnName.take_back(255);
  Value *&NameVal = GlobalStringMap[Name];
  if (!NameVal)
    NameVal = IRB.CreateAddrSpaceCast(
        IRB.CreateGlobalStringPtr(Name, "", DL.getDefaultGlobalsAddressSpace(),
                                  &M),
        PtrTy);
  return NameVal;
}
Value *GPUSanImpl::getFileName(IRBuilder<> &IRB) {
  const auto &DLoc = IRB.getCurrentDebugLocation();
  if (!DLoc || DLoc->getFilename().empty())
    return ConstantPointerNull::get(PtrTy);
  StringRef Name = DLoc->getFilename().take_back(255);
  Value *&NameVal = GlobalStringMap[Name];
  if (!NameVal)
    NameVal = IRB.CreateAddrSpaceCast(
        IRB.CreateGlobalStringPtr(Name, "", DL.getDefaultGlobalsAddressSpace(),
                                  &M),
        PtrTy);
  return NameVal;
}
Value *GPUSanImpl::getLineNo(IRBuilder<> &IRB) {
  const auto &DLoc = IRB.getCurrentDebugLocation();
  if (!DLoc)
    return Constant::getNullValue(Int64Ty);
  return ConstantInt::get(Int64Ty, DLoc.getLine());
}

PtrOrigin GPUSanImpl::getPtrOrigin(LoopInfo &LI, Value *Ptr) {
  SmallVector<const Value *> Objects;
  getUnderlyingObjects(Ptr, Objects, &LI);
  PtrOrigin PO = NONE;
  for (auto *Obj : Objects) {
    PtrOrigin ObjPO = HasAllocas ? UNKNOWN : GLOBAL;
    if (isa<AllocaInst>(Obj)) {
      ObjPO = LOCAL;
    } else if (isa<GlobalVariable>(Obj)) {
      ObjPO = GLOBAL;
    } else if (auto *II = dyn_cast<IntrinsicInst>(Obj)) {
      if (II->getIntrinsicID() == Intrinsic::amdgcn_implicitarg_ptr ||
          II->getIntrinsicID() == Intrinsic::amdgcn_dispatch_ptr)
        return SYSTEM;
    } else if (auto *CI = dyn_cast<CallInst>(Obj)) {
      if (auto *Callee = CI->getCalledFunction())
        if (Callee->getName().starts_with("ompx_")) {
          if (Callee->getName().ends_with("_global"))
            ObjPO = GLOBAL;
          else if (Callee->getName().ends_with("_local"))
            ObjPO = LOCAL;
        }
    } else if (auto *Arg = dyn_cast<Argument>(Obj)) {
      if (Arg->getParent()->hasFnAttribute("kernel"))
        ObjPO = GLOBAL;
    }
    if (PO == NONE || PO == ObjPO) {
      PO = ObjPO;
    } else {
      return UNKNOWN;
    }
  }
  return PO;
}

bool GPUSanImpl::instrumentGlobals() {
  Function *DtorFn =
      Function::Create(FunctionType::get(VoidTy, false),
                       GlobalValue::PrivateLinkage, "san.dtor", &M);
  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", DtorFn);
  IRBuilder<> IRB(Entry);
  IRB.CreateCall(getLeakCheckFn());
  IRB.CreateRetVoid();
  appendToGlobalDtors(M, DtorFn, 0, nullptr);
  return true;

  Function *DTorFn;
  std::tie(DTorFn, std::ignore) = getOrCreateSanitizerCtorAndInitFunctions(
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

Value *GPUSanImpl::instrumentAllocation(Instruction &I, Value &Size,
                                        FunctionCallee Fn) {
  IRBuilder<> IRB(&*I.getParent()->getFirstNonPHIOrDbgOrAlloca());
  Value *PlainI = IRB.CreatePointerBitCastOrAddrSpaceCast(&I, PtrTy);
  static int AllocationId = 1;
  auto *CB = IRB.CreateCall(
      Fn,
      {PlainI, &Size, ConstantInt::get(Int64Ty, AllocationId++), getPC(IRB)},
      I.getName() + ".san");
  I.replaceUsesWithIf(IRB.CreatePointerBitCastOrAddrSpaceCast(CB, I.getType()),
                      [=](Use &U) {
                        return U.getUser() != PlainI && U.getUser() != CB &&
                               !isa<LifetimeIntrinsic>(U.getUser());
                      });
  return CB;
}

Value *GPUSanImpl::instrumentAllocaInst(LoopInfo &LI, AllocaInst &AI) {
  auto SizeOrNone = AI.getAllocationSize(DL);
  if (!SizeOrNone)
    llvm_unreachable("TODO");
  Value *Size = ConstantInt::get(Int64Ty, *SizeOrNone);
  return instrumentAllocation(AI, *Size, getNewFn(LOCAL));
}

void GPUSanImpl::instrumentAccess(LoopInfo &LI, Instruction &I, int PtrIdx,
                                  Type &AccessTy, bool IsRead) {
  Value *PtrOp = I.getOperand(PtrIdx);
  PtrOrigin PO = getPtrOrigin(LI, PtrOp);
  if (PO > GLOBAL)
    return;

  static int32_t ReadAccessId = -1;
  static int32_t WriteAccessId = 1;
  const int32_t &AccessId = IsRead ? ReadAccessId-- : WriteAccessId++;

  auto TySize = DL.getTypeStoreSize(&AccessTy);
  assert(!TySize.isScalable());
  Value *Size = ConstantInt::get(Int64Ty, TySize.getFixedValue());
  IRBuilder<> IRB(&I);
  Value *PlainPtrOp = IRB.CreatePointerBitCastOrAddrSpaceCast(PtrOp, PtrTy);
  auto *CB = IRB.CreateCall(
      getCheckFn(PO),
      {PlainPtrOp, Size, ConstantInt::get(Int64Ty, AccessId), getPC(IRB),
       getFunctionName(IRB), getFileName(IRB), getLineNo(IRB)},
      I.getName() + ".san");
  I.setOperand(PtrIdx,
               IRB.CreatePointerBitCastOrAddrSpaceCast(CB, PtrOp->getType()));
}

void GPUSanImpl::instrumentLoadInst(LoopInfo &LI, LoadInst &LoadI) {
  instrumentAccess(LI, LoadI, LoadInst::getPointerOperandIndex(),
                   *LoadI.getType(),
                   /*IsRead=*/true);
}

void GPUSanImpl::instrumentStoreInst(LoopInfo &LI, StoreInst &StoreI) {
  instrumentAccess(LI, StoreI, StoreInst::getPointerOperandIndex(),
                   *StoreI.getValueOperand()->getType(), /*IsRead=*/false);
}

void GPUSanImpl::instrumentGEPInst(LoopInfo &LI, GetElementPtrInst &GEP) {
  Value *PtrOp = GEP.getPointerOperand();
  PtrOrigin PO = getPtrOrigin(LI, PtrOp);
  if (PO > GLOBAL)
    return;

  GEP.setOperand(GetElementPtrInst::getPointerOperandIndex(),
                 Constant::getNullValue(PtrOp->getType()));
  IRBuilder<> IRB(GEP.getNextNode());
  Value *PlainPtrOp = IRB.CreatePointerBitCastOrAddrSpaceCast(PtrOp, PtrTy);
  auto *CB = IRB.CreateCall(getGEPFn(PO),
                            {PlainPtrOp, UndefValue::get(Int64Ty), getPC(IRB)},
                            GEP.getName() + ".san");
  GEP.replaceAllUsesWith(
      IRB.CreatePointerBitCastOrAddrSpaceCast(CB, GEP.getType()));
  Value *Offset =
      new PtrToIntInst(&GEP, Int64Ty, GEP.getName() + ".san.offset", CB);
  CB->setArgOperand(1, Offset);
}

bool GPUSanImpl::instrumentCallInst(LoopInfo &LI, CallInst &CI) {
  bool Changed = false;
  if (isa<LifetimeIntrinsic>(CI))
    return Changed;
  if (auto *Fn = CI.getCalledFunction()) {
    if ((Fn->isDeclaration() || Fn->getName().starts_with("__kmpc") ||
         Fn->getName().starts_with("rpc_")) &&
        !Fn->getName().starts_with("ompx")) {
      IRBuilder<> IRB(&CI);
      for (int I = 0, E = CI.arg_size(); I != E; ++I) {
        Value *Op = CI.getArgOperand(I);
        if (!Op->getType()->isPointerTy())
          continue;
        PtrOrigin PO = getPtrOrigin(LI, Op);
        if (PO > GLOBAL)
          continue;
        Value *PlainOp = IRB.CreatePointerBitCastOrAddrSpaceCast(Op, PtrTy);
        auto *CB = IRB.CreateCall(getUnpackFn(PO), {PlainOp, getPC(IRB)},
                                  Op->getName() + ".unpack");
        CI.setArgOperand(
            I, IRB.CreatePointerBitCastOrAddrSpaceCast(CB, Op->getType()));
        Changed = true;
      }
    }
  }
  return Changed;
}

bool GPUSanImpl::instrumentFunction(Function &Fn) {
  if (Fn.isDeclaration())
    return false;
  bool Changed = false;
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(Fn);
  SmallVector<std::pair<AllocaInst *, Value *>> Allocas;
  SmallVector<ReturnInst *> Returns;
  for (auto &I : instructions(Fn)) {
    switch (I.getOpcode()) {
    case Instruction::Alloca: {
      AllocaInst &AI = cast<AllocaInst>(I);
      Value *FakePtr = instrumentAllocaInst(LI, AI);
      Allocas.push_back({&AI, FakePtr});
      Changed = true;
      break;
    }
    case Instruction::Load:
      instrumentLoadInst(LI, cast<LoadInst>(I));
      Changed = true;
      break;
    case Instruction::Store:
      instrumentStoreInst(LI, cast<StoreInst>(I));
      Changed = true;
      break;
    case Instruction::GetElementPtr:
      instrumentGEPInst(LI, cast<GetElementPtrInst>(I));
      Changed = true;
      break;
    case Instruction::Call:
      Changed = instrumentCallInst(LI, cast<CallInst>(I));
      break;
    case Instruction::Ret:
      Returns.push_back(&cast<ReturnInst>(I));
      break;
    default:
      break;
    }
  }

  instrumentReturns(Allocas, Returns);

  return Changed;
}

void GPUSanImpl::instrumentReturns(
    SmallVectorImpl<std::pair<AllocaInst *, Value *>> &Allocas,
    SmallVectorImpl<ReturnInst *> &Returns) {
  if (Allocas.empty())
    return;
  for (auto *RI : Returns) {
    IRBuilder<> IRB(RI);
    IRB.CreateCall(getFreeNLocalFn(),
                   {ConstantInt::get(Int32Ty, Allocas.size())}, ".free");
  }
}

bool GPUSanImpl::instrument() {
  bool Changed = instrumentGlobals();
  HasAllocas = [&]() {
    for (Function &Fn : M)
      for (auto &I : instructions(Fn))
        if (isa<AllocaInst>(I))
          return true;
    return false;
  }();

  for (Function &Fn : M)
    if (!Fn.getName().contains("ompx") && !Fn.getName().contains("__kmpc") &&
        !Fn.getName().starts_with("rpc_"))
      if (!Fn.hasFnAttribute(Attribute::DisableSanitizerInstrumentation))
        Changed |= instrumentFunction(Fn);
  return Changed;
}

PreservedAnalyses GPUSanPass::run(Module &M, ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  GPUSanImpl Lowerer(M, FAM);
  if (!Lowerer.instrument())
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
