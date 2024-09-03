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
#include "llvm/Analysis/DXILResource.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
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
  DXILResourceMap &DRM;
  SmallVector<CallInst *> CleanupCasts;

public:
  OpLowerer(Module &M, DXILResourceMap &DRM) : M(M), OpBuilder(M), DRM(DRM) {}

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

  /// Create a cast between a `target("dx")` type and `dx.types.Handle`, which
  /// is intended to be removed by the end of lowering. This is used to allow
  /// lowering of ops which need to change their return or argument types in a
  /// piecemeal way - we can add the casts in to avoid updating all of the uses
  /// or defs, and by the end all of the casts will be redundant.
  Value *createTmpHandleCast(Value *V, Type *Ty) {
    Function *CastFn = Intrinsic::getDeclaration(&M, Intrinsic::dx_cast_handle,
                                                 {Ty, V->getType()});
    CallInst *Cast = OpBuilder.getIRB().CreateCall(CastFn, {V});
    CleanupCasts.push_back(Cast);
    return Cast;
  }

  void cleanupHandleCasts() {
    SmallVector<CallInst *> ToRemove;
    SmallVector<Function *> CastFns;

    for (CallInst *Cast : CleanupCasts) {
      // These casts were only put in to ease the move from `target("dx")` types
      // to `dx.types.Handle in a piecemeal way. At this point, all of the
      // non-cast uses should now be `dx.types.Handle`, and remaining casts
      // should all form pairs to and from the now unused `target("dx")` type.
      CastFns.push_back(Cast->getCalledFunction());

      // If the cast is not to `dx.types.Handle`, it should be the first part of
      // the pair. Keep track so we can remove it once it has no more uses.
      if (Cast->getType() != OpBuilder.getHandleType()) {
        ToRemove.push_back(Cast);
        continue;
      }
      // Otherwise, we're the second handle in a pair. Forward the arguments and
      // remove the (second) cast.
      CallInst *Def = cast<CallInst>(Cast->getOperand(0));
      assert(Def->getIntrinsicID() == Intrinsic::dx_cast_handle &&
             "Unbalanced pair of temporary handle casts");
      Cast->replaceAllUsesWith(Def->getOperand(0));
      Cast->eraseFromParent();
    }
    for (CallInst *Cast : ToRemove) {
      assert(Cast->user_empty() && "Temporary handle cast still has users");
      Cast->eraseFromParent();
    }

    // Deduplicate the cast functions so that we only erase each one once.
    llvm::sort(CastFns);
    CastFns.erase(llvm::unique(CastFns), CastFns.end());
    for (Function *F : CastFns)
      F->eraseFromParent();

    CleanupCasts.clear();
  }

  void lowerToCreateHandle(Function &F) {
    IRBuilder<> &IRB = OpBuilder.getIRB();
    Type *Int8Ty = IRB.getInt8Ty();
    Type *Int32Ty = IRB.getInt32Ty();

    replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);

      auto *It = DRM.find(CI);
      assert(It != DRM.end() && "Resource not in map?");
      dxil::ResourceInfo &RI = *It;
      const auto &Binding = RI.getBinding();

      std::array<Value *, 4> Args{
          ConstantInt::get(Int8Ty, llvm::to_underlying(RI.getResourceClass())),
          ConstantInt::get(Int32Ty, Binding.RecordID), CI->getArgOperand(3),
          CI->getArgOperand(4)};
      Expected<CallInst *> OpCall =
          OpBuilder.tryCreateOp(OpCode::CreateHandle, Args);
      if (Error E = OpCall.takeError())
        return E;

      Value *Cast = createTmpHandleCast(*OpCall, CI->getType());

      CI->replaceAllUsesWith(Cast);
      CI->eraseFromParent();
      return Error::success();
    });
  }

  void lowerToBindAndAnnotateHandle(Function &F) {
    IRBuilder<> &IRB = OpBuilder.getIRB();

    replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);

      auto *It = DRM.find(CI);
      assert(It != DRM.end() && "Resource not in map?");
      dxil::ResourceInfo &RI = *It;

      const auto &Binding = RI.getBinding();
      std::pair<uint32_t, uint32_t> Props = RI.getAnnotateProps();

      // For `CreateHandleFromBinding` we need the upper bound rather than the
      // size, so we need to be careful about the difference for "unbounded".
      uint32_t Unbounded = std::numeric_limits<uint32_t>::max();
      uint32_t UpperBound = Binding.Size == Unbounded
                                ? Unbounded
                                : Binding.LowerBound + Binding.Size - 1;
      Constant *ResBind = OpBuilder.getResBind(
          Binding.LowerBound, UpperBound, Binding.Space, RI.getResourceClass());
      std::array<Value *, 3> BindArgs{ResBind, CI->getArgOperand(3),
                                      CI->getArgOperand(4)};
      Expected<CallInst *> OpBind =
          OpBuilder.tryCreateOp(OpCode::CreateHandleFromBinding, BindArgs);
      if (Error E = OpBind.takeError())
        return E;

      std::array<Value *, 2> AnnotateArgs{
          *OpBind, OpBuilder.getResProps(Props.first, Props.second)};
      Expected<CallInst *> OpAnnotate =
          OpBuilder.tryCreateOp(OpCode::AnnotateHandle, AnnotateArgs);
      if (Error E = OpAnnotate.takeError())
        return E;

      Value *Cast = createTmpHandleCast(*OpAnnotate, CI->getType());

      CI->replaceAllUsesWith(Cast);
      CI->eraseFromParent();

      return Error::success();
    });
  }

  /// Lower `dx.handle.fromBinding` intrinsics depending on the shader model and
  /// taking into account binding information from DXILResourceAnalysis.
  void lowerHandleFromBinding(Function &F) {
    Triple TT(Triple(M.getTargetTriple()));
    if (TT.getDXILVersion() < VersionTuple(1, 6))
      lowerToCreateHandle(F);
    else
      lowerToBindAndAnnotateHandle(F);
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
      case Intrinsic::dx_handle_fromBinding:
        lowerHandleFromBinding(F);
      }
      Updated = true;
    }
    if (Updated)
      cleanupHandleCasts();

    return Updated;
  }
};
} // namespace

PreservedAnalyses DXILOpLowering::run(Module &M, ModuleAnalysisManager &MAM) {
  DXILResourceMap &DRM = MAM.getResult<DXILResourceAnalysis>(M);

  bool MadeChanges = OpLowerer(M, DRM).lowerIntrinsics();
  if (!MadeChanges)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<DXILResourceAnalysis>();
  return PA;
}

namespace {
class DXILOpLoweringLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override {
    DXILResourceMap &DRM =
        getAnalysis<DXILResourceWrapperPass>().getResourceMap();

    return OpLowerer(M, DRM).lowerIntrinsics();
  }
  StringRef getPassName() const override { return "DXIL Op Lowering"; }
  DXILOpLoweringLegacy() : ModulePass(ID) {}

  static char ID; // Pass identification.
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<DXILIntrinsicExpansionLegacy>();
    AU.addRequired<DXILResourceWrapperPass>();
    AU.addPreserved<DXILResourceWrapperPass>();
  }
};
char DXILOpLoweringLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DXILOpLoweringLegacy, DEBUG_TYPE, "DXIL Op Lowering",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DXILResourceWrapperPass)
INITIALIZE_PASS_END(DXILOpLoweringLegacy, DEBUG_TYPE, "DXIL Op Lowering", false,
                    false)

ModulePass *llvm::createDXILOpLoweringLegacyPass() {
  return new DXILOpLoweringLegacy();
}
