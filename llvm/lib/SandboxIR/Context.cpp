//===- Context.cpp - The Context class of Sandbox IR ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/Module.h"

namespace llvm::sandboxir {

std::unique_ptr<Value> Context::detachLLVMValue(llvm::Value *V) {
  std::unique_ptr<Value> Erased;
  auto It = LLVMValueToValueMap.find(V);
  if (It != LLVMValueToValueMap.end()) {
    auto *Val = It->second.release();
    Erased = std::unique_ptr<Value>(Val);
    LLVMValueToValueMap.erase(It);
  }
  return Erased;
}

std::unique_ptr<Value> Context::detach(Value *V) {
  assert(V->getSubclassID() != Value::ClassID::Constant &&
         "Can't detach a constant!");
  assert(V->getSubclassID() != Value::ClassID::User && "Can't detach a user!");
  return detachLLVMValue(V->Val);
}

Value *Context::registerValue(std::unique_ptr<Value> &&VPtr) {
  assert(VPtr->getSubclassID() != Value::ClassID::User &&
         "Can't register a user!");

  Value *V = VPtr.get();
  [[maybe_unused]] auto Pair =
      LLVMValueToValueMap.insert({VPtr->Val, std::move(VPtr)});
  assert(Pair.second && "Already exists!");

  // Track creation of instructions.
  // Please note that we don't allow the creation of detached instructions,
  // meaning that the instructions need to be inserted into a block upon
  // creation. This is why the tracker class combines creation and insertion.
  if (auto *I = dyn_cast<Instruction>(V)) {
    getTracker().emplaceIfTracking<CreateAndInsertInst>(I);
    runCreateInstrCallbacks(I);
  }

  return V;
}

Value *Context::getOrCreateValueInternal(llvm::Value *LLVMV, llvm::User *U) {
  auto Pair = LLVMValueToValueMap.insert({LLVMV, nullptr});
  auto It = Pair.first;
  if (!Pair.second)
    return It->second.get();

  if (auto *C = dyn_cast<llvm::Constant>(LLVMV)) {
    switch (C->getValueID()) {
    case llvm::Value::ConstantIntVal:
      It->second = std::unique_ptr<ConstantInt>(
          new ConstantInt(cast<llvm::ConstantInt>(C), *this));
      return It->second.get();
    case llvm::Value::ConstantFPVal:
      It->second = std::unique_ptr<ConstantFP>(
          new ConstantFP(cast<llvm::ConstantFP>(C), *this));
      return It->second.get();
    case llvm::Value::BlockAddressVal:
      It->second = std::unique_ptr<BlockAddress>(
          new BlockAddress(cast<llvm::BlockAddress>(C), *this));
      return It->second.get();
    case llvm::Value::ConstantTokenNoneVal:
      It->second = std::unique_ptr<ConstantTokenNone>(
          new ConstantTokenNone(cast<llvm::ConstantTokenNone>(C), *this));
      return It->second.get();
    case llvm::Value::ConstantAggregateZeroVal: {
      auto *CAZ = cast<llvm::ConstantAggregateZero>(C);
      It->second = std::unique_ptr<ConstantAggregateZero>(
          new ConstantAggregateZero(CAZ, *this));
      auto *Ret = It->second.get();
      // Must create sandboxir for elements.
      auto EC = CAZ->getElementCount();
      if (EC.isFixed()) {
        for (auto ElmIdx : seq<unsigned>(0, EC.getFixedValue()))
          getOrCreateValueInternal(CAZ->getElementValue(ElmIdx), CAZ);
      }
      return Ret;
    }
    case llvm::Value::ConstantPointerNullVal:
      It->second = std::unique_ptr<ConstantPointerNull>(
          new ConstantPointerNull(cast<llvm::ConstantPointerNull>(C), *this));
      return It->second.get();
    case llvm::Value::PoisonValueVal:
      It->second = std::unique_ptr<PoisonValue>(
          new PoisonValue(cast<llvm::PoisonValue>(C), *this));
      return It->second.get();
    case llvm::Value::UndefValueVal:
      It->second = std::unique_ptr<UndefValue>(
          new UndefValue(cast<llvm::UndefValue>(C), *this));
      return It->second.get();
    case llvm::Value::DSOLocalEquivalentVal: {
      auto *DSOLE = cast<llvm::DSOLocalEquivalent>(C);
      It->second = std::unique_ptr<DSOLocalEquivalent>(
          new DSOLocalEquivalent(DSOLE, *this));
      auto *Ret = It->second.get();
      getOrCreateValueInternal(DSOLE->getGlobalValue(), DSOLE);
      return Ret;
    }
    case llvm::Value::ConstantArrayVal:
      It->second = std::unique_ptr<ConstantArray>(
          new ConstantArray(cast<llvm::ConstantArray>(C), *this));
      break;
    case llvm::Value::ConstantStructVal:
      It->second = std::unique_ptr<ConstantStruct>(
          new ConstantStruct(cast<llvm::ConstantStruct>(C), *this));
      break;
    case llvm::Value::ConstantVectorVal:
      It->second = std::unique_ptr<ConstantVector>(
          new ConstantVector(cast<llvm::ConstantVector>(C), *this));
      break;
    case llvm::Value::FunctionVal:
      It->second = std::unique_ptr<Function>(
          new Function(cast<llvm::Function>(C), *this));
      break;
    case llvm::Value::GlobalIFuncVal:
      It->second = std::unique_ptr<GlobalIFunc>(
          new GlobalIFunc(cast<llvm::GlobalIFunc>(C), *this));
      break;
    case llvm::Value::GlobalVariableVal:
      It->second = std::unique_ptr<GlobalVariable>(
          new GlobalVariable(cast<llvm::GlobalVariable>(C), *this));
      break;
    case llvm::Value::GlobalAliasVal:
      It->second = std::unique_ptr<GlobalAlias>(
          new GlobalAlias(cast<llvm::GlobalAlias>(C), *this));
      break;
    case llvm::Value::NoCFIValueVal:
      It->second = std::unique_ptr<NoCFIValue>(
          new NoCFIValue(cast<llvm::NoCFIValue>(C), *this));
      break;
    case llvm::Value::ConstantPtrAuthVal:
      It->second = std::unique_ptr<ConstantPtrAuth>(
          new ConstantPtrAuth(cast<llvm::ConstantPtrAuth>(C), *this));
      break;
    case llvm::Value::ConstantExprVal:
      It->second = std::unique_ptr<ConstantExpr>(
          new ConstantExpr(cast<llvm::ConstantExpr>(C), *this));
      break;
    default:
      It->second = std::unique_ptr<Constant>(new Constant(C, *this));
      break;
    }
    auto *NewC = It->second.get();
    for (llvm::Value *COp : C->operands())
      getOrCreateValueInternal(COp, C);
    return NewC;
  }
  if (auto *Arg = dyn_cast<llvm::Argument>(LLVMV)) {
    It->second = std::unique_ptr<Argument>(new Argument(Arg, *this));
    return It->second.get();
  }
  if (auto *BB = dyn_cast<llvm::BasicBlock>(LLVMV)) {
    assert(isa<llvm::BlockAddress>(U) &&
           "This won't create a SBBB, don't call this function directly!");
    if (auto *SBBB = getValue(BB))
      return SBBB;
    return nullptr;
  }
  assert(isa<llvm::Instruction>(LLVMV) && "Expected Instruction");

  switch (cast<llvm::Instruction>(LLVMV)->getOpcode()) {
  case llvm::Instruction::VAArg: {
    auto *LLVMVAArg = cast<llvm::VAArgInst>(LLVMV);
    It->second = std::unique_ptr<VAArgInst>(new VAArgInst(LLVMVAArg, *this));
    return It->second.get();
  }
  case llvm::Instruction::Freeze: {
    auto *LLVMFreeze = cast<llvm::FreezeInst>(LLVMV);
    It->second = std::unique_ptr<FreezeInst>(new FreezeInst(LLVMFreeze, *this));
    return It->second.get();
  }
  case llvm::Instruction::Fence: {
    auto *LLVMFence = cast<llvm::FenceInst>(LLVMV);
    It->second = std::unique_ptr<FenceInst>(new FenceInst(LLVMFence, *this));
    return It->second.get();
  }
  case llvm::Instruction::Select: {
    auto *LLVMSel = cast<llvm::SelectInst>(LLVMV);
    It->second = std::unique_ptr<SelectInst>(new SelectInst(LLVMSel, *this));
    return It->second.get();
  }
  case llvm::Instruction::ExtractElement: {
    auto *LLVMIns = cast<llvm::ExtractElementInst>(LLVMV);
    It->second = std::unique_ptr<ExtractElementInst>(
        new ExtractElementInst(LLVMIns, *this));
    return It->second.get();
  }
  case llvm::Instruction::InsertElement: {
    auto *LLVMIns = cast<llvm::InsertElementInst>(LLVMV);
    It->second = std::unique_ptr<InsertElementInst>(
        new InsertElementInst(LLVMIns, *this));
    return It->second.get();
  }
  case llvm::Instruction::ShuffleVector: {
    auto *LLVMIns = cast<llvm::ShuffleVectorInst>(LLVMV);
    It->second = std::unique_ptr<ShuffleVectorInst>(
        new ShuffleVectorInst(LLVMIns, *this));
    return It->second.get();
  }
  case llvm::Instruction::ExtractValue: {
    auto *LLVMIns = cast<llvm::ExtractValueInst>(LLVMV);
    It->second =
        std::unique_ptr<ExtractValueInst>(new ExtractValueInst(LLVMIns, *this));
    return It->second.get();
  }
  case llvm::Instruction::InsertValue: {
    auto *LLVMIns = cast<llvm::InsertValueInst>(LLVMV);
    It->second =
        std::unique_ptr<InsertValueInst>(new InsertValueInst(LLVMIns, *this));
    return It->second.get();
  }
  case llvm::Instruction::Br: {
    auto *LLVMBr = cast<llvm::BranchInst>(LLVMV);
    It->second = std::unique_ptr<BranchInst>(new BranchInst(LLVMBr, *this));
    return It->second.get();
  }
  case llvm::Instruction::Load: {
    auto *LLVMLd = cast<llvm::LoadInst>(LLVMV);
    It->second = std::unique_ptr<LoadInst>(new LoadInst(LLVMLd, *this));
    return It->second.get();
  }
  case llvm::Instruction::Store: {
    auto *LLVMSt = cast<llvm::StoreInst>(LLVMV);
    It->second = std::unique_ptr<StoreInst>(new StoreInst(LLVMSt, *this));
    return It->second.get();
  }
  case llvm::Instruction::Ret: {
    auto *LLVMRet = cast<llvm::ReturnInst>(LLVMV);
    It->second = std::unique_ptr<ReturnInst>(new ReturnInst(LLVMRet, *this));
    return It->second.get();
  }
  case llvm::Instruction::Call: {
    auto *LLVMCall = cast<llvm::CallInst>(LLVMV);
    It->second = std::unique_ptr<CallInst>(new CallInst(LLVMCall, *this));
    return It->second.get();
  }
  case llvm::Instruction::Invoke: {
    auto *LLVMInvoke = cast<llvm::InvokeInst>(LLVMV);
    It->second = std::unique_ptr<InvokeInst>(new InvokeInst(LLVMInvoke, *this));
    return It->second.get();
  }
  case llvm::Instruction::CallBr: {
    auto *LLVMCallBr = cast<llvm::CallBrInst>(LLVMV);
    It->second = std::unique_ptr<CallBrInst>(new CallBrInst(LLVMCallBr, *this));
    return It->second.get();
  }
  case llvm::Instruction::LandingPad: {
    auto *LLVMLPad = cast<llvm::LandingPadInst>(LLVMV);
    It->second =
        std::unique_ptr<LandingPadInst>(new LandingPadInst(LLVMLPad, *this));
    return It->second.get();
  }
  case llvm::Instruction::CatchPad: {
    auto *LLVMCPI = cast<llvm::CatchPadInst>(LLVMV);
    It->second =
        std::unique_ptr<CatchPadInst>(new CatchPadInst(LLVMCPI, *this));
    return It->second.get();
  }
  case llvm::Instruction::CleanupPad: {
    auto *LLVMCPI = cast<llvm::CleanupPadInst>(LLVMV);
    It->second =
        std::unique_ptr<CleanupPadInst>(new CleanupPadInst(LLVMCPI, *this));
    return It->second.get();
  }
  case llvm::Instruction::CatchRet: {
    auto *LLVMCRI = cast<llvm::CatchReturnInst>(LLVMV);
    It->second =
        std::unique_ptr<CatchReturnInst>(new CatchReturnInst(LLVMCRI, *this));
    return It->second.get();
  }
  case llvm::Instruction::CleanupRet: {
    auto *LLVMCRI = cast<llvm::CleanupReturnInst>(LLVMV);
    It->second = std::unique_ptr<CleanupReturnInst>(
        new CleanupReturnInst(LLVMCRI, *this));
    return It->second.get();
  }
  case llvm::Instruction::GetElementPtr: {
    auto *LLVMGEP = cast<llvm::GetElementPtrInst>(LLVMV);
    It->second = std::unique_ptr<GetElementPtrInst>(
        new GetElementPtrInst(LLVMGEP, *this));
    return It->second.get();
  }
  case llvm::Instruction::CatchSwitch: {
    auto *LLVMCatchSwitchInst = cast<llvm::CatchSwitchInst>(LLVMV);
    It->second = std::unique_ptr<CatchSwitchInst>(
        new CatchSwitchInst(LLVMCatchSwitchInst, *this));
    return It->second.get();
  }
  case llvm::Instruction::Resume: {
    auto *LLVMResumeInst = cast<llvm::ResumeInst>(LLVMV);
    It->second =
        std::unique_ptr<ResumeInst>(new ResumeInst(LLVMResumeInst, *this));
    return It->second.get();
  }
  case llvm::Instruction::Switch: {
    auto *LLVMSwitchInst = cast<llvm::SwitchInst>(LLVMV);
    It->second =
        std::unique_ptr<SwitchInst>(new SwitchInst(LLVMSwitchInst, *this));
    return It->second.get();
  }
  case llvm::Instruction::FNeg: {
    auto *LLVMUnaryOperator = cast<llvm::UnaryOperator>(LLVMV);
    It->second = std::unique_ptr<UnaryOperator>(
        new UnaryOperator(LLVMUnaryOperator, *this));
    return It->second.get();
  }
  case llvm::Instruction::Add:
  case llvm::Instruction::FAdd:
  case llvm::Instruction::Sub:
  case llvm::Instruction::FSub:
  case llvm::Instruction::Mul:
  case llvm::Instruction::FMul:
  case llvm::Instruction::UDiv:
  case llvm::Instruction::SDiv:
  case llvm::Instruction::FDiv:
  case llvm::Instruction::URem:
  case llvm::Instruction::SRem:
  case llvm::Instruction::FRem:
  case llvm::Instruction::Shl:
  case llvm::Instruction::LShr:
  case llvm::Instruction::AShr:
  case llvm::Instruction::And:
  case llvm::Instruction::Or:
  case llvm::Instruction::Xor: {
    auto *LLVMBinaryOperator = cast<llvm::BinaryOperator>(LLVMV);
    It->second = std::unique_ptr<BinaryOperator>(
        new BinaryOperator(LLVMBinaryOperator, *this));
    return It->second.get();
  }
  case llvm::Instruction::AtomicRMW: {
    auto *LLVMAtomicRMW = cast<llvm::AtomicRMWInst>(LLVMV);
    It->second =
        std::unique_ptr<AtomicRMWInst>(new AtomicRMWInst(LLVMAtomicRMW, *this));
    return It->second.get();
  }
  case llvm::Instruction::AtomicCmpXchg: {
    auto *LLVMAtomicCmpXchg = cast<llvm::AtomicCmpXchgInst>(LLVMV);
    It->second = std::unique_ptr<AtomicCmpXchgInst>(
        new AtomicCmpXchgInst(LLVMAtomicCmpXchg, *this));
    return It->second.get();
  }
  case llvm::Instruction::Alloca: {
    auto *LLVMAlloca = cast<llvm::AllocaInst>(LLVMV);
    It->second = std::unique_ptr<AllocaInst>(new AllocaInst(LLVMAlloca, *this));
    return It->second.get();
  }
  case llvm::Instruction::ZExt:
  case llvm::Instruction::SExt:
  case llvm::Instruction::FPToUI:
  case llvm::Instruction::FPToSI:
  case llvm::Instruction::FPExt:
  case llvm::Instruction::PtrToInt:
  case llvm::Instruction::IntToPtr:
  case llvm::Instruction::SIToFP:
  case llvm::Instruction::UIToFP:
  case llvm::Instruction::Trunc:
  case llvm::Instruction::FPTrunc:
  case llvm::Instruction::BitCast:
  case llvm::Instruction::AddrSpaceCast: {
    auto *LLVMCast = cast<llvm::CastInst>(LLVMV);
    It->second = std::unique_ptr<CastInst>(new CastInst(LLVMCast, *this));
    return It->second.get();
  }
  case llvm::Instruction::PHI: {
    auto *LLVMPhi = cast<llvm::PHINode>(LLVMV);
    It->second = std::unique_ptr<PHINode>(new PHINode(LLVMPhi, *this));
    return It->second.get();
  }
  case llvm::Instruction::ICmp: {
    auto *LLVMICmp = cast<llvm::ICmpInst>(LLVMV);
    It->second = std::unique_ptr<ICmpInst>(new ICmpInst(LLVMICmp, *this));
    return It->second.get();
  }
  case llvm::Instruction::FCmp: {
    auto *LLVMFCmp = cast<llvm::FCmpInst>(LLVMV);
    It->second = std::unique_ptr<FCmpInst>(new FCmpInst(LLVMFCmp, *this));
    return It->second.get();
  }
  case llvm::Instruction::Unreachable: {
    auto *LLVMUnreachable = cast<llvm::UnreachableInst>(LLVMV);
    It->second = std::unique_ptr<UnreachableInst>(
        new UnreachableInst(LLVMUnreachable, *this));
    return It->second.get();
  }
  default:
    break;
  }

  It->second = std::unique_ptr<OpaqueInst>(
      new OpaqueInst(cast<llvm::Instruction>(LLVMV), *this));
  return It->second.get();
}

Argument *Context::getOrCreateArgument(llvm::Argument *LLVMArg) {
  auto Pair = LLVMValueToValueMap.insert({LLVMArg, nullptr});
  auto It = Pair.first;
  if (Pair.second) {
    It->second = std::unique_ptr<Argument>(new Argument(LLVMArg, *this));
    return cast<Argument>(It->second.get());
  }
  return cast<Argument>(It->second.get());
}

Constant *Context::getOrCreateConstant(llvm::Constant *LLVMC) {
  return cast<Constant>(getOrCreateValueInternal(LLVMC, 0));
}

BasicBlock *Context::createBasicBlock(llvm::BasicBlock *LLVMBB) {
  assert(getValue(LLVMBB) == nullptr && "Already exists!");
  auto NewBBPtr = std::unique_ptr<BasicBlock>(new BasicBlock(LLVMBB, *this));
  auto *BB = cast<BasicBlock>(registerValue(std::move(NewBBPtr)));
  // Create SandboxIR for BB's body.
  BB->buildBasicBlockFromLLVMIR(LLVMBB);
  return BB;
}

VAArgInst *Context::createVAArgInst(llvm::VAArgInst *SI) {
  auto NewPtr = std::unique_ptr<VAArgInst>(new VAArgInst(SI, *this));
  return cast<VAArgInst>(registerValue(std::move(NewPtr)));
}

FreezeInst *Context::createFreezeInst(llvm::FreezeInst *SI) {
  auto NewPtr = std::unique_ptr<FreezeInst>(new FreezeInst(SI, *this));
  return cast<FreezeInst>(registerValue(std::move(NewPtr)));
}

FenceInst *Context::createFenceInst(llvm::FenceInst *SI) {
  auto NewPtr = std::unique_ptr<FenceInst>(new FenceInst(SI, *this));
  return cast<FenceInst>(registerValue(std::move(NewPtr)));
}

SelectInst *Context::createSelectInst(llvm::SelectInst *SI) {
  auto NewPtr = std::unique_ptr<SelectInst>(new SelectInst(SI, *this));
  return cast<SelectInst>(registerValue(std::move(NewPtr)));
}

ExtractElementInst *
Context::createExtractElementInst(llvm::ExtractElementInst *EEI) {
  auto NewPtr =
      std::unique_ptr<ExtractElementInst>(new ExtractElementInst(EEI, *this));
  return cast<ExtractElementInst>(registerValue(std::move(NewPtr)));
}

InsertElementInst *
Context::createInsertElementInst(llvm::InsertElementInst *IEI) {
  auto NewPtr =
      std::unique_ptr<InsertElementInst>(new InsertElementInst(IEI, *this));
  return cast<InsertElementInst>(registerValue(std::move(NewPtr)));
}

ShuffleVectorInst *
Context::createShuffleVectorInst(llvm::ShuffleVectorInst *SVI) {
  auto NewPtr =
      std::unique_ptr<ShuffleVectorInst>(new ShuffleVectorInst(SVI, *this));
  return cast<ShuffleVectorInst>(registerValue(std::move(NewPtr)));
}

ExtractValueInst *Context::createExtractValueInst(llvm::ExtractValueInst *EVI) {
  auto NewPtr =
      std::unique_ptr<ExtractValueInst>(new ExtractValueInst(EVI, *this));
  return cast<ExtractValueInst>(registerValue(std::move(NewPtr)));
}

InsertValueInst *Context::createInsertValueInst(llvm::InsertValueInst *IVI) {
  auto NewPtr =
      std::unique_ptr<InsertValueInst>(new InsertValueInst(IVI, *this));
  return cast<InsertValueInst>(registerValue(std::move(NewPtr)));
}

BranchInst *Context::createBranchInst(llvm::BranchInst *BI) {
  auto NewPtr = std::unique_ptr<BranchInst>(new BranchInst(BI, *this));
  return cast<BranchInst>(registerValue(std::move(NewPtr)));
}

LoadInst *Context::createLoadInst(llvm::LoadInst *LI) {
  auto NewPtr = std::unique_ptr<LoadInst>(new LoadInst(LI, *this));
  return cast<LoadInst>(registerValue(std::move(NewPtr)));
}

StoreInst *Context::createStoreInst(llvm::StoreInst *SI) {
  auto NewPtr = std::unique_ptr<StoreInst>(new StoreInst(SI, *this));
  return cast<StoreInst>(registerValue(std::move(NewPtr)));
}

ReturnInst *Context::createReturnInst(llvm::ReturnInst *I) {
  auto NewPtr = std::unique_ptr<ReturnInst>(new ReturnInst(I, *this));
  return cast<ReturnInst>(registerValue(std::move(NewPtr)));
}

CallInst *Context::createCallInst(llvm::CallInst *I) {
  auto NewPtr = std::unique_ptr<CallInst>(new CallInst(I, *this));
  return cast<CallInst>(registerValue(std::move(NewPtr)));
}

InvokeInst *Context::createInvokeInst(llvm::InvokeInst *I) {
  auto NewPtr = std::unique_ptr<InvokeInst>(new InvokeInst(I, *this));
  return cast<InvokeInst>(registerValue(std::move(NewPtr)));
}

CallBrInst *Context::createCallBrInst(llvm::CallBrInst *I) {
  auto NewPtr = std::unique_ptr<CallBrInst>(new CallBrInst(I, *this));
  return cast<CallBrInst>(registerValue(std::move(NewPtr)));
}

UnreachableInst *Context::createUnreachableInst(llvm::UnreachableInst *UI) {
  auto NewPtr =
      std::unique_ptr<UnreachableInst>(new UnreachableInst(UI, *this));
  return cast<UnreachableInst>(registerValue(std::move(NewPtr)));
}
LandingPadInst *Context::createLandingPadInst(llvm::LandingPadInst *I) {
  auto NewPtr = std::unique_ptr<LandingPadInst>(new LandingPadInst(I, *this));
  return cast<LandingPadInst>(registerValue(std::move(NewPtr)));
}
CatchPadInst *Context::createCatchPadInst(llvm::CatchPadInst *I) {
  auto NewPtr = std::unique_ptr<CatchPadInst>(new CatchPadInst(I, *this));
  return cast<CatchPadInst>(registerValue(std::move(NewPtr)));
}
CleanupPadInst *Context::createCleanupPadInst(llvm::CleanupPadInst *I) {
  auto NewPtr = std::unique_ptr<CleanupPadInst>(new CleanupPadInst(I, *this));
  return cast<CleanupPadInst>(registerValue(std::move(NewPtr)));
}
CatchReturnInst *Context::createCatchReturnInst(llvm::CatchReturnInst *I) {
  auto NewPtr = std::unique_ptr<CatchReturnInst>(new CatchReturnInst(I, *this));
  return cast<CatchReturnInst>(registerValue(std::move(NewPtr)));
}
CleanupReturnInst *
Context::createCleanupReturnInst(llvm::CleanupReturnInst *I) {
  auto NewPtr =
      std::unique_ptr<CleanupReturnInst>(new CleanupReturnInst(I, *this));
  return cast<CleanupReturnInst>(registerValue(std::move(NewPtr)));
}
GetElementPtrInst *
Context::createGetElementPtrInst(llvm::GetElementPtrInst *I) {
  auto NewPtr =
      std::unique_ptr<GetElementPtrInst>(new GetElementPtrInst(I, *this));
  return cast<GetElementPtrInst>(registerValue(std::move(NewPtr)));
}
CatchSwitchInst *Context::createCatchSwitchInst(llvm::CatchSwitchInst *I) {
  auto NewPtr = std::unique_ptr<CatchSwitchInst>(new CatchSwitchInst(I, *this));
  return cast<CatchSwitchInst>(registerValue(std::move(NewPtr)));
}
ResumeInst *Context::createResumeInst(llvm::ResumeInst *I) {
  auto NewPtr = std::unique_ptr<ResumeInst>(new ResumeInst(I, *this));
  return cast<ResumeInst>(registerValue(std::move(NewPtr)));
}
SwitchInst *Context::createSwitchInst(llvm::SwitchInst *I) {
  auto NewPtr = std::unique_ptr<SwitchInst>(new SwitchInst(I, *this));
  return cast<SwitchInst>(registerValue(std::move(NewPtr)));
}
UnaryOperator *Context::createUnaryOperator(llvm::UnaryOperator *I) {
  auto NewPtr = std::unique_ptr<UnaryOperator>(new UnaryOperator(I, *this));
  return cast<UnaryOperator>(registerValue(std::move(NewPtr)));
}
BinaryOperator *Context::createBinaryOperator(llvm::BinaryOperator *I) {
  auto NewPtr = std::unique_ptr<BinaryOperator>(new BinaryOperator(I, *this));
  return cast<BinaryOperator>(registerValue(std::move(NewPtr)));
}
AtomicRMWInst *Context::createAtomicRMWInst(llvm::AtomicRMWInst *I) {
  auto NewPtr = std::unique_ptr<AtomicRMWInst>(new AtomicRMWInst(I, *this));
  return cast<AtomicRMWInst>(registerValue(std::move(NewPtr)));
}
AtomicCmpXchgInst *
Context::createAtomicCmpXchgInst(llvm::AtomicCmpXchgInst *I) {
  auto NewPtr =
      std::unique_ptr<AtomicCmpXchgInst>(new AtomicCmpXchgInst(I, *this));
  return cast<AtomicCmpXchgInst>(registerValue(std::move(NewPtr)));
}
AllocaInst *Context::createAllocaInst(llvm::AllocaInst *I) {
  auto NewPtr = std::unique_ptr<AllocaInst>(new AllocaInst(I, *this));
  return cast<AllocaInst>(registerValue(std::move(NewPtr)));
}
CastInst *Context::createCastInst(llvm::CastInst *I) {
  auto NewPtr = std::unique_ptr<CastInst>(new CastInst(I, *this));
  return cast<CastInst>(registerValue(std::move(NewPtr)));
}
PHINode *Context::createPHINode(llvm::PHINode *I) {
  auto NewPtr = std::unique_ptr<PHINode>(new PHINode(I, *this));
  return cast<PHINode>(registerValue(std::move(NewPtr)));
}
ICmpInst *Context::createICmpInst(llvm::ICmpInst *I) {
  auto NewPtr = std::unique_ptr<ICmpInst>(new ICmpInst(I, *this));
  return cast<ICmpInst>(registerValue(std::move(NewPtr)));
}
FCmpInst *Context::createFCmpInst(llvm::FCmpInst *I) {
  auto NewPtr = std::unique_ptr<FCmpInst>(new FCmpInst(I, *this));
  return cast<FCmpInst>(registerValue(std::move(NewPtr)));
}
Value *Context::getValue(llvm::Value *V) const {
  auto It = LLVMValueToValueMap.find(V);
  if (It != LLVMValueToValueMap.end())
    return It->second.get();
  return nullptr;
}

Context::Context(LLVMContext &LLVMCtx)
    : LLVMCtx(LLVMCtx), IRTracker(*this),
      LLVMIRBuilder(LLVMCtx, ConstantFolder()) {}

Context::~Context() {}

Module *Context::getModule(llvm::Module *LLVMM) const {
  auto It = LLVMModuleToModuleMap.find(LLVMM);
  if (It != LLVMModuleToModuleMap.end())
    return It->second.get();
  return nullptr;
}

Module *Context::getOrCreateModule(llvm::Module *LLVMM) {
  auto Pair = LLVMModuleToModuleMap.insert({LLVMM, nullptr});
  auto It = Pair.first;
  if (!Pair.second)
    return It->second.get();
  It->second = std::unique_ptr<Module>(new Module(*LLVMM, *this));
  return It->second.get();
}

Function *Context::createFunction(llvm::Function *F) {
  assert(getValue(F) == nullptr && "Already exists!");
  // Create the module if needed before we create the new sandboxir::Function.
  // Note: this won't fully populate the module. The only globals that will be
  // available will be the ones being used within the function.
  getOrCreateModule(F->getParent());

  auto NewFPtr = std::unique_ptr<Function>(new Function(F, *this));
  auto *SBF = cast<Function>(registerValue(std::move(NewFPtr)));
  // Create arguments.
  for (auto &Arg : F->args())
    getOrCreateArgument(&Arg);
  // Create BBs.
  for (auto &BB : *F)
    createBasicBlock(&BB);
  return SBF;
}

Module *Context::createModule(llvm::Module *LLVMM) {
  auto *M = getOrCreateModule(LLVMM);
  // Create the functions.
  for (auto &LLVMF : *LLVMM)
    createFunction(&LLVMF);
  // Create globals.
  for (auto &Global : LLVMM->globals())
    getOrCreateValue(&Global);
  // Create aliases.
  for (auto &Alias : LLVMM->aliases())
    getOrCreateValue(&Alias);
  // Create ifuncs.
  for (auto &IFunc : LLVMM->ifuncs())
    getOrCreateValue(&IFunc);

  return M;
}

void Context::runEraseInstrCallbacks(Instruction *I) {
  for (const auto &CBEntry : EraseInstrCallbacks)
    CBEntry.second(I);
}

void Context::runCreateInstrCallbacks(Instruction *I) {
  for (auto &CBEntry : CreateInstrCallbacks)
    CBEntry.second(I);
}

void Context::runMoveInstrCallbacks(Instruction *I, const BBIterator &WhereIt) {
  for (auto &CBEntry : MoveInstrCallbacks)
    CBEntry.second(I, WhereIt);
}

// An arbitrary limit, to check for accidental misuse. We expect a small number
// of callbacks to be registered at a time, but we can increase this number if
// we discover we needed more.
[[maybe_unused]] static constexpr int MaxRegisteredCallbacks = 16;

Context::CallbackID Context::registerEraseInstrCallback(EraseInstrCallback CB) {
  assert(EraseInstrCallbacks.size() <= MaxRegisteredCallbacks &&
         "EraseInstrCallbacks size limit exceeded");
  CallbackID ID{NextCallbackID++};
  EraseInstrCallbacks[ID] = CB;
  return ID;
}
void Context::unregisterEraseInstrCallback(CallbackID ID) {
  [[maybe_unused]] bool Erased = EraseInstrCallbacks.erase(ID);
  assert(Erased &&
         "Callback ID not found in EraseInstrCallbacks during deregistration");
}

Context::CallbackID
Context::registerCreateInstrCallback(CreateInstrCallback CB) {
  assert(CreateInstrCallbacks.size() <= MaxRegisteredCallbacks &&
         "CreateInstrCallbacks size limit exceeded");
  CallbackID ID{NextCallbackID++};
  CreateInstrCallbacks[ID] = CB;
  return ID;
}
void Context::unregisterCreateInstrCallback(CallbackID ID) {
  [[maybe_unused]] bool Erased = CreateInstrCallbacks.erase(ID);
  assert(Erased &&
         "Callback ID not found in CreateInstrCallbacks during deregistration");
}

Context::CallbackID Context::registerMoveInstrCallback(MoveInstrCallback CB) {
  assert(MoveInstrCallbacks.size() <= MaxRegisteredCallbacks &&
         "MoveInstrCallbacks size limit exceeded");
  CallbackID ID{NextCallbackID++};
  MoveInstrCallbacks[ID] = CB;
  return ID;
}
void Context::unregisterMoveInstrCallback(CallbackID ID) {
  [[maybe_unused]] bool Erased = MoveInstrCallbacks.erase(ID);
  assert(Erased &&
         "Callback ID not found in MoveInstrCallbacks during deregistration");
}

} // namespace llvm::sandboxir
