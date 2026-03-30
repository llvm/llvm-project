//===-- SPIRVCtorDtorLowering.cpp - Handle global ctors and dtors --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass creates a unified init and fini kernel with the required metadata
// to call global constructors and destructors on SPIR-V targets.
//
//===----------------------------------------------------------------------===//

#include "SPIRVCtorDtorLowering.h"
#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "SPIRV.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MD5.h"
#include "llvm/Transforms/IPO/OpenMPOpt.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "spirv-lower-ctor-dtor"

static cl::opt<std::string>
    GlobalStr("spirv-lower-global-ctor-dtor-id",
              cl::desc("Override unique ID of ctor/dtor globals."),
              cl::init(""), cl::Hidden);

static cl::opt<bool>
    CreateKernels("spirv-emit-init-fini-kernel",
                  cl::desc("Emit kernels to call ctor/dtor globals."),
                  cl::init(true), cl::Hidden);

namespace {
constexpr int SPIRV_GLOBAL_AS = 1;

std::string getHash(StringRef Str) {
  llvm::MD5 Hasher;
  llvm::MD5::MD5Result Hash;
  Hasher.update(Str);
  Hasher.final(Hash);
  return llvm::utohexstr(Hash.low(), /*LowerCase=*/true);
}

void addKernelAttrs(Function *F) {
  F->setCallingConv(CallingConv::SPIR_KERNEL);
  F->addFnAttr("uniform-work-group-size", "true");
}

Function *createInitOrFiniKernelFunction(Module &M, bool IsCtor) {
  StringRef InitOrFiniKernelName =
      IsCtor ? "spirv$device$init" : "spirv$device$fini";
  if (M.getFunction(InitOrFiniKernelName))
    return nullptr;

  Function *InitOrFiniKernel = Function::createWithDefaultAttr(
      FunctionType::get(Type::getVoidTy(M.getContext()), false),
      GlobalValue::WeakODRLinkage, 0, InitOrFiniKernelName, &M);
  addKernelAttrs(InitOrFiniKernel);

  return InitOrFiniKernel;
}

// We create the IR required to call each callback in this section. This is
// equivalent to the following code. Normally, the linker would provide us with
// the definitions of the init and fini array sections. The 'spirv-link' linker
// does not do this so initializing these values is done by the offload runtime.
//
// extern "C" void **__init_array_start = nullptr;
// extern "C" void **__init_array_end = nullptr;
// extern "C" void **__fini_array_start = nullptr;
// extern "C" void **__fini_array_end = nullptr;
//
// using InitCallback = void();
// using FiniCallback = void();
//
// void call_init_array_callbacks() {
//   for (auto start = __init_array_start; start != __init_array_end; ++start)
//     reinterpret_cast<InitCallback *>(*start)();
// }
//
// void call_fini_array_callbacks() {
//   size_t fini_array_size = __fini_array_end - __fini_array_start;
//   for (size_t i = fini_array_size; i > 0; --i)
//     reinterpret_cast<FiniCallback *>(__fini_array_start[i - 1])();
// }
void createInitOrFiniCalls(Function &F, bool IsCtor) {
  Module &M = *F.getParent();
  LLVMContext &C = M.getContext();

  IRBuilder<> IRB(BasicBlock::Create(C, "entry", &F));
  auto *LoopBB = BasicBlock::Create(C, "while.entry", &F);
  auto *ExitBB = BasicBlock::Create(C, "while.end", &F);
  Type *PtrTy = IRB.getPtrTy(SPIRV_GLOBAL_AS);

  auto CreateGlobal = [&](const char *Name) -> GlobalVariable * {
    auto *GV = new GlobalVariable(
        M, PointerType::getUnqual(C),
        /*isConstant=*/false, GlobalValue::WeakAnyLinkage,
        Constant::getNullValue(PointerType::getUnqual(C)), Name,
        /*InsertBefore=*/nullptr, GlobalVariable::NotThreadLocal,
        /*AddressSpace=*/SPIRV_GLOBAL_AS);
    GV->setVisibility(GlobalVariable::ProtectedVisibility);
    return GV;
  };

  auto *Begin = M.getOrInsertGlobal(
      IsCtor ? "__init_array_start" : "__fini_array_start",
      PointerType::getUnqual(C), function_ref<GlobalVariable *()>([&]() {
        return CreateGlobal(IsCtor ? "__init_array_start"
                                   : "__fini_array_start");
      }));
  auto *End = M.getOrInsertGlobal(
      IsCtor ? "__init_array_end" : "__fini_array_end",
      PointerType::getUnqual(C), function_ref<GlobalVariable *()>([&]() {
        return CreateGlobal(IsCtor ? "__init_array_end" : "__fini_array_end");
      }));
  auto *CallBackTy = FunctionType::get(IRB.getVoidTy(), {});

  // The destructor array must be called in reverse order. Get an expression to
  // the end of the array and iterate backwards in that case.
  Value *BeginVal = IRB.CreateLoad(Begin->getType(), Begin, "begin");
  Value *EndVal = IRB.CreateLoad(Begin->getType(), End, "stop");
  if (!IsCtor) {
    Value *OldBeginVal = BeginVal;
    BeginVal =
        IRB.CreateInBoundsGEP(PointerType::getUnqual(C), EndVal,
                              ArrayRef<Value *>(ConstantInt::getAllOnesValue(
                                  IntegerType::getInt64Ty(C))),
                              "start");
    EndVal = OldBeginVal;
  }
  IRB.CreateCondBr(
      IRB.CreateCmp(IsCtor ? ICmpInst::ICMP_NE : ICmpInst::ICMP_UGE, BeginVal,
                    EndVal),
      LoopBB, ExitBB);
  IRB.SetInsertPoint(LoopBB);
  auto *CallBackPHI = IRB.CreatePHI(PtrTy, 2, "ptr");
  auto *CallBack = IRB.CreateLoad(IRB.getPtrTy(F.getAddressSpace()),
                                  CallBackPHI, "callback");
  IRB.CreateCall(CallBackTy, CallBack);
  auto *NewCallBack =
      IRB.CreateConstGEP1_64(PtrTy, CallBackPHI, IsCtor ? 1 : -1, "next");
  auto *EndCmp = IRB.CreateCmp(IsCtor ? ICmpInst::ICMP_EQ : ICmpInst::ICMP_ULT,
                               NewCallBack, EndVal, "end");
  CallBackPHI->addIncoming(BeginVal, &F.getEntryBlock());
  CallBackPHI->addIncoming(NewCallBack, LoopBB);
  IRB.CreateCondBr(EndCmp, ExitBB, LoopBB);
  IRB.SetInsertPoint(ExitBB);
  IRB.CreateRetVoid();
}

bool createInitOrFiniGlobals(Module &M, GlobalVariable *GV, bool IsCtor) {
  ConstantArray *GA = dyn_cast<ConstantArray>(GV->getInitializer());
  if (!GA || GA->getNumOperands() == 0)
    return false;

  // SPIR-V has no way to emit variables at specific sections or support for
  // the traditional constructor sections. Instead, we emit mangled global
  // names so the runtime can build the list manually.
  for (Value *V : GA->operands()) {
    auto *CS = cast<ConstantStruct>(V);
    auto *F = cast<Constant>(CS->getOperand(1));
    uint64_t Priority = cast<ConstantInt>(CS->getOperand(0))->getSExtValue();
    std::string PriorityStr = "." + std::to_string(Priority);
    // We append a semi-unique hash and the priority to the global name.
    std::string GlobalID =
        !GlobalStr.empty() ? GlobalStr : getHash(M.getSourceFileName());
    std::string NameStr =
        ((IsCtor ? "__init_array_object_" : "__fini_array_object_") +
         F->getName() + "_" + GlobalID + "_" + std::to_string(Priority))
            .str();
    llvm::transform(NameStr, NameStr.begin(),
                    [](char c) { return c == '.' ? '_' : c; });

    auto *GV = new GlobalVariable(M, F->getType(), /*IsConstant=*/true,
                                  GlobalValue::ExternalLinkage, F, NameStr,
                                  nullptr, GlobalValue::NotThreadLocal,
                                  /*AddressSpace=*/SPIRV_GLOBAL_AS);
    GV->setSection(IsCtor ? ".init_array" + PriorityStr
                          : ".fini_array" + PriorityStr);
    GV->setVisibility(GlobalVariable::ProtectedVisibility);
  }

  return true;
}

bool createInitOrFiniKernel(Module &M, StringRef GlobalName, bool IsCtor) {
  GlobalVariable *GV = M.getGlobalVariable(GlobalName);
  if (!GV || !GV->hasInitializer())
    return false;

  if (!createInitOrFiniGlobals(M, GV, IsCtor))
    return false;

  if (!CreateKernels)
    return true;

  Function *InitOrFiniKernel = createInitOrFiniKernelFunction(M, IsCtor);
  if (!InitOrFiniKernel)
    return false;

  createInitOrFiniCalls(*InitOrFiniKernel, IsCtor);

  GV->eraseFromParent();
  return true;
}

bool lowerCtorsAndDtors(Module &M) {
  // Only run this pass for OpenMP offload compilation
  if (!llvm::omp::isOpenMPDevice(M))
    return false;

  bool Modified = false;
  Modified |= createInitOrFiniKernel(M, "llvm.global_ctors", /*IsCtor =*/true);
  Modified |= createInitOrFiniKernel(M, "llvm.global_dtors", /*IsCtor =*/false);
  return Modified;
}

class SPIRVCtorDtorLoweringLegacy final : public ModulePass {
public:
  static char ID;
  SPIRVCtorDtorLoweringLegacy() : ModulePass(ID) {}
  bool runOnModule(Module &M) override { return lowerCtorsAndDtors(M); }
};

} // End anonymous namespace

PreservedAnalyses SPIRVCtorDtorLoweringPass::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  return lowerCtorsAndDtors(M) ? PreservedAnalyses::none()
                               : PreservedAnalyses::all();
}

char SPIRVCtorDtorLoweringLegacy::ID = 0;
INITIALIZE_PASS(SPIRVCtorDtorLoweringLegacy, DEBUG_TYPE,
                "SPIRV lower ctors and dtors", false, false)

ModulePass *llvm::createSPIRVCtorDtorLoweringLegacyPass() {
  return new SPIRVCtorDtorLoweringLegacy();
}
