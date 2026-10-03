//===-- InputGenGPU.cpp - InputGen GPU instrumentation pass ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/InputGenGPU.h"
#include "llvm/Transforms/IPO/Instrumentor.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Instrumentation.h"

using namespace llvm;
using namespace llvm::instrumentor;

#define DEBUG_TYPE "inputgen-gpu"

static cl::opt<std::string> InputGenGPUEntryFunction(
    "inputgen-gpu-entry-function",
    cl::desc("Device function wrapped by the InputGen GPU entry kernel"),
    cl::init(""));

static cl::list<std::string> InputGenGPURuntimeBitcodes(
    "inputgen-gpu-runtime-bitcode",
    cl::desc("InputGen GPU runtime bitcode file; may be repeated"),
    cl::ZeroOrMore);

namespace {

#define INPUTGEN_GPU_ENTRY_STATE(Variable, Constant, CType, Symbol)            \
  constexpr StringLiteral Constant(Symbol);
#include "llvm/Frontend/Offloading/InputGenGPUABI.def"

std::string getInputGenGPUEntryPointName(StringRef EntryFunctionName) {
  return (Twine("__ig_entry_") + EntryFunctionName).str();
}

bool createInputGenGPUEntryKernel(Module &M, InstrumentorIRBuilderTy &IIRB,
                                  StringRef EntryFunctionName) {
  if (EntryFunctionName.empty())
    return false;

  Function *EntryFn = M.getFunction(EntryFunctionName);
  if (!EntryFn || EntryFn->isDeclaration()) {
    IIRB.Ctx.diagnose(DiagnosticInfoInstrumentation(
        Twine("inputgen entry function '") + EntryFunctionName +
            "' was not found or is only a declaration",
        DS_Warning));
    return false;
  }

  CallingConv::ID KernelCC;
  const Triple &T = M.getTargetTriple();
  if (T.isAMDGPU())
    KernelCC = CallingConv::AMDGPU_KERNEL;
  else if (T.isNVPTX())
    KernelCC = CallingConv::PTX_Kernel;
  else {
    IIRB.Ctx.diagnose(DiagnosticInfoInstrumentation(
        Twine("inputgen entry kernels are not supported for target '") +
            T.str() + "'",
        DS_Warning));
    return false;
  }

  std::string EntryPointName = getInputGenGPUEntryPointName(EntryFunctionName);
  if (M.getNamedValue(EntryPointName)) {
    IIRB.Ctx.diagnose(DiagnosticInfoInstrumentation(
        Twine("inputgen entry point '") + EntryPointName + "' already exists",
        DS_Warning));
    return false;
  }

  const DataLayout &DL = M.getDataLayout();
  unsigned GlobalAS = DL.getDefaultGlobalsAddressSpace();
  auto GetOrInsertGlobalInDefaultAS = [&](StringRef Name, Type *Ty) {
    return M.getOrInsertGlobal(Name, Ty, [&] {
      return new GlobalVariable(
          M, Ty, /*isConstant=*/false, GlobalValue::ExternalLinkage,
          /*Initializer=*/nullptr, Name,
          /*InsertBefore=*/nullptr, GlobalVariable::NotThreadLocal, GlobalAS);
    });
  };

  GlobalVariable *BufferGV =
      GetOrInsertGlobalInDefaultAS(InputGenEntryBufferSymbol, IIRB.PtrTy);
  GlobalVariable *BufferSizeGV =
      GetOrInsertGlobalInDefaultAS(InputGenEntryBufferSizeSymbol, IIRB.Int64Ty);
  GlobalVariable *BufferOffsetGV = GetOrInsertGlobalInDefaultAS(
      InputGenEntryBufferOffsetSymbol, IIRB.Int64Ty);
  GlobalVariable *ModeGV =
      GetOrInsertGlobalInDefaultAS(InputGenEntryModeSymbol, IIRB.Int32Ty);

  FunctionType *EntryPointTy = FunctionType::get(
      IIRB.VoidTy, {IIRB.Int32Ty, IIRB.PtrTy, IIRB.Int64Ty, IIRB.PtrTy},
      /*isVarArg=*/false);
  Function *EntryPoint = Function::Create(
      EntryPointTy, GlobalValue::ExternalLinkage, EntryPointName, M);
  EntryPoint->setCallingConv(KernelCC);

  auto ArgIt = EntryPoint->arg_begin();
  Argument *Mode = &*ArgIt++;
  Argument *Buffer = &*ArgIt++;
  Argument *Size = &*ArgIt++;
  Argument *Result = &*ArgIt++;
  Mode->setName("mode");
  Buffer->setName("input_buffer");
  Size->setName("input_buffer_size");
  Result->setName("result");

  BasicBlock *EntryBB = BasicBlock::Create(IIRB.Ctx, "entry", EntryPoint);
  IIRB.IRB.SetInsertPoint(EntryBB);

  IIRB.IRB.CreateAlignedStore(Mode, ModeGV, DL.getABITypeAlign(IIRB.Int32Ty));
  IIRB.IRB.CreateAlignedStore(Buffer, BufferGV, DL.getABITypeAlign(IIRB.PtrTy));
  IIRB.IRB.CreateAlignedStore(Size, BufferSizeGV,
                              DL.getABITypeAlign(IIRB.Int64Ty));
  IIRB.IRB.CreateAlignedStore(ConstantInt::get(IIRB.Int64Ty, 0), BufferOffsetGV,
                              DL.getABITypeAlign(IIRB.Int64Ty));

  SmallVector<Value *> Args;
  Args.reserve(EntryFn->arg_size());
  for (Argument &Arg : EntryFn->args()) {
    Type *ArgTy = Arg.getType();
    if (!ArgTy->isPointerTy()) {
      Args.push_back(Constant::getNullValue(ArgTy));
      continue;
    }

    // Create placeholder storage for pointer arguments.
    AllocaInst *AI =
        IIRB.IRB.CreateAlloca(IIRB.Int64Ty, DL.getAllocaAddrSpace());
    AI->setAlignment(Align(8));
    Args.push_back(IIRB.IRB.CreatePointerBitCastOrAddrSpaceCast(AI, ArgTy));
  }

  CallInst *CI = IIRB.IRB.CreateCall(EntryFn->getFunctionType(), EntryFn, Args);
  if (!EntryFn->getReturnType()->isVoidTy())
    IIRB.IRB.CreateAlignedStore(CI, Result,
                                DL.getABITypeAlign(EntryFn->getReturnType()));
  IIRB.IRB.CreateRetVoid();

  return true;
}

class InputGenGPUConfig final : public InstrumentationConfig {
  void populate(InstrumentorIRBuilderTy &IIRB) override {
    InstrumentationConfig::populate(IIRB);

    RuntimePrefix->setString("__ig_");
    HostEnabled->setBool(false);
    GPUEnabled->setBool(true);
    SmallVector<StringRef> RuntimeBitcodeRefs;
    for (StringRef RuntimeBitcode : InputGenGPURuntimeBitcodes)
      RuntimeBitcodeRefs.push_back(RuntimeBitcode);
    RuntimeBitcodes->setStringList(RuntimeBitcodeRefs);
    InlineRuntimeEagerly->setBool(false);

    for (auto &ChoiceMap : IChoices) {
      for (auto &ChoiceIt : ChoiceMap) {
        auto *IO = ChoiceIt.second;
        IO->Enabled = false;
        IO->Filter = "";
        for (IRTArg &Arg : IO->IRTArgs)
          Arg.Enabled = false;
      }
    }

    auto *PostLoad =
        IChoices[InstrumentationLocation::INSTRUCTION_POST].lookup("load");
    if (!PostLoad)
      return;

    PostLoad->Enabled = true;
    for (IRTArg &Arg : PostLoad->IRTArgs) {
      Arg.Enabled = Arg.Name == "value" || Arg.Name == "value_size" ||
                    Arg.Name == "value_type_id" || Arg.Name == "id";
    }
  }

  bool instrumentBeforeRuntimeLink(Module &M,
                                   InstrumentorIRBuilderTy &IIRB) override {
    return createInputGenGPUEntryKernel(M, IIRB, InputGenGPUEntryFunction);
  }
};

} // end anonymous namespace

PreservedAnalyses InputGenGPUPass::run(Module &M, ModuleAnalysisManager &MAM) {
  const Triple &T = M.getTargetTriple();
  if (!InputGenGPUEntryFunction.empty() && !T.isAMDGPU() && !T.isNVPTX()) {
    M.getContext().diagnose(DiagnosticInfoInstrumentation(
        Twine("inputgen entry kernels are not supported for target '") +
            T.str() + "'",
        DS_Warning));
    return PreservedAnalyses::all();
  }

  InputGenGPUConfig IConf;
  InstrumentorIRBuilderTy IIRB(M);
  return InstrumentorPass(/*FS=*/nullptr, &IConf, &IIRB).run(M, MAM);
}
