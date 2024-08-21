//===- DXILOpLowering.cpp - Lowering to DXIL operations -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILOpLowering.h"
#include "DXILConstants.h"
#include "DXILIntrinsicExpansion.h"
#include "DXILOpBuilder.h"
#include "DirectX.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "dxil-op-lower"

using namespace llvm;
using namespace llvm::dxil;

static bool isVectorArgExpansion(Function &F) {
  switch (F.getIntrinsicID()) {
  case Intrinsic::dx_dot2:
  case Intrinsic::dx_dot3:
  case Intrinsic::dx_dot4:
    return true;
  }
  return false;
}

static SmallVector<Value *> populateOperands(Value *Arg, IRBuilder<> &Builder) {
  SmallVector<Value *> ExtractedElements;
  auto *VecArg = dyn_cast<FixedVectorType>(Arg->getType());
  for (unsigned I = 0; I < VecArg->getNumElements(); ++I) {
    Value *Index = ConstantInt::get(Type::getInt32Ty(Arg->getContext()), I);
    Value *ExtractedElement = Builder.CreateExtractElement(Arg, Index);
    ExtractedElements.push_back(ExtractedElement);
  }
  return ExtractedElements;
}

static SmallVector<Value *> argVectorFlatten(CallInst *Orig,
                                             IRBuilder<> &Builder) {
  // Note: arg[NumOperands-1] is a pointer and is not needed by our flattening.
  unsigned NumOperands = Orig->getNumOperands() - 1;
  assert(NumOperands > 0);
  Value *Arg0 = Orig->getOperand(0);
  [[maybe_unused]] auto *VecArg0 = dyn_cast<FixedVectorType>(Arg0->getType());
  assert(VecArg0);
  SmallVector<Value *> NewOperands = populateOperands(Arg0, Builder);
  for (unsigned I = 1; I < NumOperands; ++I) {
    Value *Arg = Orig->getOperand(I);
    [[maybe_unused]] auto *VecArg = dyn_cast<FixedVectorType>(Arg->getType());
    assert(VecArg);
    assert(VecArg0->getElementType() == VecArg->getElementType());
    assert(VecArg0->getNumElements() == VecArg->getNumElements());
    auto NextOperandList = populateOperands(Arg, Builder);
    NewOperands.append(NextOperandList.begin(), NextOperandList.end());
  }
  return NewOperands;
}

namespace {
class OpLowerer {
  Module &M;
  DXILOpBuilder OpBuilder;

public:
  OpLowerer(Module &M) : M(M), OpBuilder(M) {}

  void replaceFunction(Function &F,
                       llvm::function_ref<Error(CallInst *CI)> ReplaceCall) {
    for (User *U : make_early_inc_range(F.users())) {
      CallInst *CI = dyn_cast<CallInst>(U);
      if (!CI)
        continue;

      if (Error E = ReplaceCall(CI)) {
        std::string Message(toString(std::move(E)));
        DiagnosticInfoUnsupported Diag(*CI->getFunction(), Message,
                                       CI->getDebugLoc());
        M.getContext().diagnose(Diag);
        continue;
      }
    }
    if (F.user_empty())
      F.eraseFromParent();
  }

  void replaceFunctionWithOp(Function &F, dxil::OpCode DXILOp) {
    bool IsVectorArgExpansion = isVectorArgExpansion(F);
    replaceFunction(F, [&](CallInst *CI) -> Error {
      SmallVector<Value *> Args;
      OpBuilder.getIRB().SetInsertPoint(CI);
      if (IsVectorArgExpansion) {
        SmallVector<Value *> NewArgs = argVectorFlatten(CI, OpBuilder.getIRB());
        Args.append(NewArgs.begin(), NewArgs.end());
      } else
        Args.append(CI->arg_begin(), CI->arg_end());

      Expected<CallInst *> OpCall =
          OpBuilder.tryCreateOp(DXILOp, Args, F.getReturnType());
      if (Error E = OpCall.takeError())
        return E;

      CI->replaceAllUsesWith(*OpCall);
      CI->eraseFromParent();
      return Error::success();
    });
  }

  bool lowerIntrinsics() {
    bool Updated = false;

    for (Function &F : make_early_inc_range(M.functions())) {
      if (!F.isDeclaration())
        continue;
      Intrinsic::ID ID = F.getIntrinsicID();
      switch (ID) {
      default:
        continue;
#define DXIL_OP_INTRINSIC(OpCode, Intrin)                                      \
  case Intrin:                                                                 \
    replaceFunctionWithOp(F, OpCode);                                          \
    break;
#include "DXILOperation.inc"
      }
      Updated = true;
    }
    return Updated;
  }
};
} // namespace

PreservedAnalyses DXILOpLowering::run(Module &M, ModuleAnalysisManager &) {
  if (OpLowerer(M).lowerIntrinsics())
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

namespace {
class DXILOpLoweringLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override {
    return OpLowerer(M).lowerIntrinsics();
  }
  StringRef getPassName() const override { return "DXIL Op Lowering"; }
  DXILOpLoweringLegacy() : ModulePass(ID) {}

  static char ID; // Pass identification.
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    // Specify the passes that your pass depends on
    AU.addRequired<DXILIntrinsicExpansionLegacy>();
  }
};
char DXILOpLoweringLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DXILOpLoweringLegacy, DEBUG_TYPE, "DXIL Op Lowering",
                      false, false)
INITIALIZE_PASS_END(DXILOpLoweringLegacy, DEBUG_TYPE, "DXIL Op Lowering", false,
                    false)

ModulePass *llvm::createDXILOpLoweringLegacyPass() {
  return new DXILOpLoweringLegacy();
}
