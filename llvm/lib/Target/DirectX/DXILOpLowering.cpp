//===- DXILOpLowering.cpp - Lowering to DXIL operations -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILOpLowering.h"
#include "DXILConstants.h"
#include "DXILOpBuilder.h"
#include "DXILRootSignature.h"
#include "DXILShaderFlags.h"
#include "DirectX.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Use.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "dxil-op-lower"

using namespace llvm;
using namespace llvm::dxil;

namespace {
class OpLowerer {
  Module &M;
  DXILOpBuilder OpBuilder;
  DXILResourceMap &DRM;
  DXILResourceTypeMap &DRTM;
  const ModuleMetadataInfo &MMDI;
  SmallVector<CallInst *> CleanupCasts;
  Function *CleanupNURI = nullptr;

public:
  OpLowerer(Module &M, DXILResourceMap &DRM, DXILResourceTypeMap &DRTM,
            const ModuleMetadataInfo &MMDI)
      : M(M), OpBuilder(M), DRM(DRM), DRTM(DRTM), MMDI(MMDI) {}

  /// Replace every call to \c F using \c ReplaceCall, and then erase \c F. If
  /// there is an error replacing a call, we emit a diagnostic and return true.
  [[nodiscard]] bool
  replaceFunction(Function &F,
                  llvm::function_ref<Error(CallInst *CI)> ReplaceCall) {
    for (User *U : make_early_inc_range(F.users())) {
      CallInst *CI = dyn_cast<CallInst>(U);
      if (!CI)
        continue;

      if (Error E = ReplaceCall(CI)) {
        std::string Message(toString(std::move(E)));
        M.getContext().diagnose(DiagnosticInfoUnsupported(
            *CI->getFunction(), Message, CI->getDebugLoc()));

        return true;
      }
    }
    if (F.user_empty())
      F.eraseFromParent();
    return false;
  }

  struct IntrinArgSelect {
    enum class Type {
#define DXIL_OP_INTRINSIC_ARG_SELECT_TYPE(name) name,
#include "DXILOperation.inc"
    };
    Type Type;
    int Value;
  };

  /// Replaces uses of a struct with uses of an equivalent named struct.
  ///
  /// DXIL operations that return structs give them well known names, so we need
  /// to update uses when we switch from an LLVM intrinsic to an op.
  Error replaceNamedStructUses(CallInst *Intrin, CallInst *DXILOp) {
    auto *IntrinTy = cast<StructType>(Intrin->getType());
    auto *DXILOpTy = cast<StructType>(DXILOp->getType());
    if (!IntrinTy->isLayoutIdentical(DXILOpTy))
      return make_error<StringError>(
          "Type mismatch between intrinsic and DXIL op",
          inconvertibleErrorCode());

    for (Use &U : make_early_inc_range(Intrin->uses()))
      if (auto *EVI = dyn_cast<ExtractValueInst>(U.getUser()))
        EVI->setOperand(0, DXILOp);
      else if (auto *IVI = dyn_cast<InsertValueInst>(U.getUser()))
        IVI->setOperand(0, DXILOp);
      else
        return make_error<StringError>("DXIL ops that return structs may only "
                                       "be used by insert- and extractvalue",
                                       inconvertibleErrorCode());
    return Error::success();
  }

  [[nodiscard]] bool
  replaceFunctionWithOp(Function &F, dxil::OpCode DXILOp,
                        ArrayRef<IntrinArgSelect> ArgSelects) {
    return replaceFunction(F, [&](CallInst *CI) -> Error {
      OpBuilder.getIRB().SetInsertPoint(CI);
      SmallVector<Value *> Args;
      if (ArgSelects.size()) {
        for (const IntrinArgSelect &A : ArgSelects) {
          switch (A.Type) {
          case IntrinArgSelect::Type::Index:
            Args.push_back(CI->getArgOperand(A.Value));
            break;
          case IntrinArgSelect::Type::I8:
            Args.push_back(OpBuilder.getIRB().getInt8((uint8_t)A.Value));
            break;
          case IntrinArgSelect::Type::I32:
            Args.push_back(OpBuilder.getIRB().getInt32(A.Value));
            break;
          }
        }
      } else {
        Args.append(CI->arg_begin(), CI->arg_end());
      }

      Expected<CallInst *> OpCall =
          OpBuilder.tryCreateOp(DXILOp, Args, CI->getName(), F.getReturnType());
      if (Error E = OpCall.takeError())
        return E;

      if (isa<StructType>(CI->getType())) {
        if (Error E = replaceNamedStructUses(CI, *OpCall))
          return E;
      } else
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
    CallInst *Cast = OpBuilder.getIRB().CreateIntrinsic(
        Intrinsic::dx_resource_casthandle, {Ty, V->getType()}, {V});
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
      assert(Def->getIntrinsicID() == Intrinsic::dx_resource_casthandle &&
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

  void cleanupNonUniformResourceIndexCalls() {
    // Replace all NonUniformResourceIndex calls with their argument.
    if (!CleanupNURI)
      return;
    for (User *U : make_early_inc_range(CleanupNURI->users())) {
      CallInst *CI = dyn_cast<CallInst>(U);
      if (!CI)
        continue;
      CI->replaceAllUsesWith(CI->getArgOperand(0));
      CI->eraseFromParent();
    }
    CleanupNURI->eraseFromParent();
    CleanupNURI = nullptr;
  }

  // Remove the resource global associated with the handleFromBinding call
  // instruction and their uses as they aren't needed anymore.
  // TODO: We should verify that all the globals get removed.
  // It's expected we'll need a custom pass in the future that will eliminate
  // the need for this here.
  void removeResourceGlobals(CallInst *CI) {
    for (User *User : make_early_inc_range(CI->users())) {
      if (StoreInst *Store = dyn_cast<StoreInst>(User)) {
        Value *V = Store->getOperand(1);
        Store->eraseFromParent();
        if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
          if (GV->use_empty()) {
            GV->removeDeadConstantUsers();
            GV->eraseFromParent();
          }
      }
    }
  }

  void replaceHandleFromBindingCall(CallInst *CI, Value *Replacement) {
    assert(CI->getCalledFunction()->getIntrinsicID() ==
           Intrinsic::dx_resource_handlefrombinding);

    removeResourceGlobals(CI);

    auto *NameGlobal = dyn_cast<llvm::GlobalVariable>(CI->getArgOperand(4));

    CI->replaceAllUsesWith(Replacement);
    CI->eraseFromParent();

    if (NameGlobal && NameGlobal->use_empty())
      NameGlobal->removeFromParent();
  }

  bool hasNonUniformIndex(Value *IndexOp) {
    if (isa<llvm::Constant>(IndexOp))
      return false;

    SmallVector<Value *> WorkList;
    WorkList.push_back(IndexOp);

    while (!WorkList.empty()) {
      Value *V = WorkList.pop_back_val();
      if (auto *CI = dyn_cast<CallInst>(V)) {
        if (CI->getCalledFunction()->getIntrinsicID() ==
            Intrinsic::dx_resource_nonuniformindex)
          return true;
      }
      if (auto *U = llvm::dyn_cast<llvm::User>(V)) {
        for (llvm::Value *Op : U->operands()) {
          if (isa<llvm::Constant>(Op))
            continue;
          WorkList.push_back(Op);
        }
      }
    }
    return false;
  }

  [[nodiscard]] bool lowerToCreateHandle(Function &F) {
    IRBuilder<> &IRB = OpBuilder.getIRB();
    Type *Int8Ty = IRB.getInt8Ty();
    Type *Int32Ty = IRB.getInt32Ty();
    Type *Int1Ty = IRB.getInt1Ty();

    return replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);

      auto *It = DRM.find(CI);
      assert(It != DRM.end() && "Resource not in map?");
      dxil::ResourceInfo &RI = *It;

      const auto &Binding = RI.getBinding();
      dxil::ResourceClass RC = DRTM[RI.getHandleTy()].getResourceClass();

      Value *IndexOp = CI->getArgOperand(3);
      if (Binding.LowerBound != 0)
        IndexOp = IRB.CreateAdd(IndexOp,
                                ConstantInt::get(Int32Ty, Binding.LowerBound));

      bool HasNonUniformIndex =
          (Binding.Size == 1) ? false : hasNonUniformIndex(IndexOp);
      std::array<Value *, 4> Args{
          ConstantInt::get(Int8Ty, llvm::to_underlying(RC)),
          ConstantInt::get(Int32Ty, Binding.RecordID), IndexOp,
          ConstantInt::get(Int1Ty, HasNonUniformIndex)};
      Expected<CallInst *> OpCall =
          OpBuilder.tryCreateOp(OpCode::CreateHandle, Args, CI->getName());
      if (Error E = OpCall.takeError())
        return E;

      Value *Cast = createTmpHandleCast(*OpCall, CI->getType());
      replaceHandleFromBindingCall(CI, Cast);
      return Error::success();
    });
  }

  [[nodiscard]] bool lowerToBindAndAnnotateHandle(Function &F) {
    IRBuilder<> &IRB = OpBuilder.getIRB();
    Type *Int32Ty = IRB.getInt32Ty();
    Type *Int1Ty = IRB.getInt1Ty();

    return replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);

      auto *It = DRM.find(CI);
      assert(It != DRM.end() && "Resource not in map?");
      dxil::ResourceInfo &RI = *It;

      const auto &Binding = RI.getBinding();
      dxil::ResourceTypeInfo &RTI = DRTM[RI.getHandleTy()];
      dxil::ResourceClass RC = RTI.getResourceClass();

      Value *IndexOp = CI->getArgOperand(3);
      if (Binding.LowerBound != 0)
        IndexOp = IRB.CreateAdd(IndexOp,
                                ConstantInt::get(Int32Ty, Binding.LowerBound));

      std::pair<uint32_t, uint32_t> Props =
          RI.getAnnotateProps(*F.getParent(), RTI);

      // For `CreateHandleFromBinding` we need the upper bound rather than the
      // size, so we need to be careful about the difference for "unbounded".
      uint32_t Unbounded = std::numeric_limits<uint32_t>::max();
      uint32_t UpperBound = Binding.Size == Unbounded
                                ? Unbounded
                                : Binding.LowerBound + Binding.Size - 1;
      Constant *ResBind = OpBuilder.getResBind(Binding.LowerBound, UpperBound,
                                               Binding.Space, RC);
      bool NonUniformIndex =
          (Binding.Size == 1) ? false : hasNonUniformIndex(IndexOp);
      Constant *NonUniformOp = ConstantInt::get(Int1Ty, NonUniformIndex);
      std::array<Value *, 3> BindArgs{ResBind, IndexOp, NonUniformOp};
      Expected<CallInst *> OpBind = OpBuilder.tryCreateOp(
          OpCode::CreateHandleFromBinding, BindArgs, CI->getName());
      if (Error E = OpBind.takeError())
        return E;

      std::array<Value *, 2> AnnotateArgs{
          *OpBind, OpBuilder.getResProps(Props.first, Props.second)};
      Expected<CallInst *> OpAnnotate = OpBuilder.tryCreateOp(
          OpCode::AnnotateHandle, AnnotateArgs,
          CI->hasName() ? CI->getName() + "_annot" : Twine());
      if (Error E = OpAnnotate.takeError())
        return E;

      Value *Cast = createTmpHandleCast(*OpAnnotate, CI->getType());
      replaceHandleFromBindingCall(CI, Cast);
      return Error::success();
    });
  }

  /// Lower `dx.resource.handlefrombinding` intrinsics depending on the shader
  /// model and taking into account binding information from
  /// DXILResourceAnalysis.
  bool lowerHandleFromBinding(Function &F) {
    if (MMDI.DXILVersion < VersionTuple(1, 6))
      return lowerToCreateHandle(F);
    return lowerToBindAndAnnotateHandle(F);
  }

  /// Replace uses of \c Intrin with the values in the `dx.ResRet` of \c Op.
  /// Since we expect to be post-scalarization, make an effort to avoid vectors.
  Error replaceResRetUses(CallInst *Intrin, CallInst *Op, bool HasCheckBit) {
    IRBuilder<> &IRB = OpBuilder.getIRB();

    Instruction *OldResult = Intrin;
    Type *OldTy = Intrin->getType();

    if (HasCheckBit) {
      auto *ST = cast<StructType>(OldTy);

      Value *CheckOp = nullptr;
      Type *Int32Ty = IRB.getInt32Ty();
      for (Use &U : make_early_inc_range(OldResult->uses())) {
        if (auto *EVI = dyn_cast<ExtractValueInst>(U.getUser())) {
          ArrayRef<unsigned> Indices = EVI->getIndices();
          assert(Indices.size() == 1);
          // We're only interested in uses of the check bit for now.
          if (Indices[0] != 1)
            continue;
          if (!CheckOp) {
            Value *NewEVI = IRB.CreateExtractValue(Op, 4);
            Expected<CallInst *> OpCall = OpBuilder.tryCreateOp(
                OpCode::CheckAccessFullyMapped, {NewEVI},
                OldResult->hasName() ? OldResult->getName() + "_check"
                                     : Twine(),
                Int32Ty);
            if (Error E = OpCall.takeError())
              return E;
            CheckOp = *OpCall;
          }
          EVI->replaceAllUsesWith(CheckOp);
          EVI->eraseFromParent();
        }
      }

      if (OldResult->use_empty()) {
        // Only the check bit was used, so we're done here.
        OldResult->eraseFromParent();
        return Error::success();
      }

      assert(OldResult->hasOneUse() &&
             isa<ExtractValueInst>(*OldResult->user_begin()) &&
             "Expected only use to be extract of first element");
      OldResult = cast<Instruction>(*OldResult->user_begin());
      OldTy = ST->getElementType(0);
    }

    // For scalars, we just extract the first element.
    if (!isa<FixedVectorType>(OldTy)) {
      Value *EVI = IRB.CreateExtractValue(Op, 0);
      OldResult->replaceAllUsesWith(EVI);
      OldResult->eraseFromParent();
      if (OldResult != Intrin) {
        assert(Intrin->use_empty() && "Intrinsic still has uses?");
        Intrin->eraseFromParent();
      }
      return Error::success();
    }

    std::array<Value *, 4> Extracts = {};
    SmallVector<ExtractElementInst *> DynamicAccesses;

    // The users of the operation should all be scalarized, so we attempt to
    // replace the extractelements with extractvalues directly.
    for (Use &U : make_early_inc_range(OldResult->uses())) {
      if (auto *EEI = dyn_cast<ExtractElementInst>(U.getUser())) {
        if (auto *IndexOp = dyn_cast<ConstantInt>(EEI->getIndexOperand())) {
          size_t IndexVal = IndexOp->getZExtValue();
          assert(IndexVal < 4 && "Index into buffer load out of range");
          if (!Extracts[IndexVal])
            Extracts[IndexVal] = IRB.CreateExtractValue(Op, IndexVal);
          EEI->replaceAllUsesWith(Extracts[IndexVal]);
          EEI->eraseFromParent();
        } else {
          DynamicAccesses.push_back(EEI);
        }
      }
    }

    const auto *VecTy = cast<FixedVectorType>(OldTy);
    const unsigned N = VecTy->getNumElements();

    // If there's a dynamic access we need to round trip through stack memory so
    // that we don't leave vectors around.
    if (!DynamicAccesses.empty()) {
      Type *Int32Ty = IRB.getInt32Ty();
      Constant *Zero = ConstantInt::get(Int32Ty, 0);

      Type *ElTy = VecTy->getElementType();
      Type *ArrayTy = ArrayType::get(ElTy, N);
      Value *Alloca = IRB.CreateAlloca(ArrayTy);

      for (int I = 0, E = N; I != E; ++I) {
        if (!Extracts[I])
          Extracts[I] = IRB.CreateExtractValue(Op, I);
        Value *GEP = IRB.CreateInBoundsGEP(
            ArrayTy, Alloca, {Zero, ConstantInt::get(Int32Ty, I)});
        IRB.CreateStore(Extracts[I], GEP);
      }

      for (ExtractElementInst *EEI : DynamicAccesses) {
        Value *GEP = IRB.CreateInBoundsGEP(ArrayTy, Alloca,
                                           {Zero, EEI->getIndexOperand()});
        Value *Load = IRB.CreateLoad(ElTy, GEP);
        EEI->replaceAllUsesWith(Load);
        EEI->eraseFromParent();
      }
    }

    // If we still have uses, then we're not fully scalarized and need to
    // recreate the vector. This should only happen for things like exported
    // functions from libraries.
    if (!OldResult->use_empty()) {
      for (int I = 0, E = N; I != E; ++I)
        if (!Extracts[I])
          Extracts[I] = IRB.CreateExtractValue(Op, I);

      Value *Vec = PoisonValue::get(OldTy);
      for (int I = 0, E = N; I != E; ++I)
        Vec = IRB.CreateInsertElement(Vec, Extracts[I], I);
      OldResult->replaceAllUsesWith(Vec);
    }

    OldResult->eraseFromParent();
    if (OldResult != Intrin) {
      assert(Intrin->use_empty() && "Intrinsic still has uses?");
      Intrin->eraseFromParent();
    }

    return Error::success();
  }

  [[nodiscard]] bool lowerTypedBufferLoad(Function &F, bool HasCheckBit) {
    IRBuilder<> &IRB = OpBuilder.getIRB();
    Type *Int32Ty = IRB.getInt32Ty();

    return replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);

      Value *Handle =
          createTmpHandleCast(CI->getArgOperand(0), OpBuilder.getHandleType());
      Value *Index0 = CI->getArgOperand(1);
      Value *Index1 = UndefValue::get(Int32Ty);

      Type *OldTy = CI->getType();
      if (HasCheckBit)
        OldTy = cast<StructType>(OldTy)->getElementType(0);
      Type *NewRetTy = OpBuilder.getResRetType(OldTy->getScalarType());

      std::array<Value *, 3> Args{Handle, Index0, Index1};
      Expected<CallInst *> OpCall = OpBuilder.tryCreateOp(
          OpCode::BufferLoad, Args, CI->getName(), NewRetTy);
      if (Error E = OpCall.takeError())
        return E;
      if (Error E = replaceResRetUses(CI, *OpCall, HasCheckBit))
        return E;

      return Error::success();
    });
  }

  [[nodiscard]] bool lowerRawBufferLoad(Function &F) {
    const DataLayout &DL = F.getDataLayout();
    IRBuilder<> &IRB = OpBuilder.getIRB();
    Type *Int8Ty = IRB.getInt8Ty();
    Type *Int32Ty = IRB.getInt32Ty();

    return replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);

      Type *OldTy = cast<StructType>(CI->getType())->getElementType(0);
      Type *ScalarTy = OldTy->getScalarType();
      Type *NewRetTy = OpBuilder.getResRetType(ScalarTy);

      Value *Handle =
          createTmpHandleCast(CI->getArgOperand(0), OpBuilder.getHandleType());
      Value *Index0 = CI->getArgOperand(1);
      Value *Index1 = CI->getArgOperand(2);
      uint64_t NumElements =
          DL.getTypeSizeInBits(OldTy) / DL.getTypeSizeInBits(ScalarTy);
      Value *Mask = ConstantInt::get(Int8Ty, ~(~0U << NumElements));
      Value *Align =
          ConstantInt::get(Int32Ty, DL.getPrefTypeAlign(ScalarTy).value());

      Expected<CallInst *> OpCall =
          MMDI.DXILVersion >= VersionTuple(1, 2)
              ? OpBuilder.tryCreateOp(OpCode::RawBufferLoad,
                                      {Handle, Index0, Index1, Mask, Align},
                                      CI->getName(), NewRetTy)
              : OpBuilder.tryCreateOp(OpCode::BufferLoad,
                                      {Handle, Index0, Index1}, CI->getName(),
                                      NewRetTy);
      if (Error E = OpCall.takeError())
        return E;
      if (Error E = replaceResRetUses(CI, *OpCall, /*HasCheckBit=*/true))
        return E;

      return Error::success();
    });
  }

  [[nodiscard]] bool lowerCBufferLoad(Function &F) {
    IRBuilder<> &IRB = OpBuilder.getIRB();

    return replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);

      Type *OldTy = cast<StructType>(CI->getType())->getElementType(0);
      Type *ScalarTy = OldTy->getScalarType();
      Type *NewRetTy = OpBuilder.getCBufRetType(ScalarTy);

      Value *Handle =
          createTmpHandleCast(CI->getArgOperand(0), OpBuilder.getHandleType());
      Value *Index = CI->getArgOperand(1);

      Expected<CallInst *> OpCall = OpBuilder.tryCreateOp(
          OpCode::CBufferLoadLegacy, {Handle, Index}, CI->getName(), NewRetTy);
      if (Error E = OpCall.takeError())
        return E;
      if (Error E = replaceNamedStructUses(CI, *OpCall))
        return E;

      CI->eraseFromParent();
      return Error::success();
    });
  }

  [[nodiscard]] bool lowerUpdateCounter(Function &F) {
    IRBuilder<> &IRB = OpBuilder.getIRB();
    Type *Int32Ty = IRB.getInt32Ty();

    return replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);
      Value *Handle =
          createTmpHandleCast(CI->getArgOperand(0), OpBuilder.getHandleType());
      Value *Op1 = CI->getArgOperand(1);

      std::array<Value *, 2> Args{Handle, Op1};

      Expected<CallInst *> OpCall = OpBuilder.tryCreateOp(
          OpCode::UpdateCounter, Args, CI->getName(), Int32Ty);

      if (Error E = OpCall.takeError())
        return E;

      CI->replaceAllUsesWith(*OpCall);
      CI->eraseFromParent();
      return Error::success();
    });
  }

  [[nodiscard]] bool lowerGetDimensionsX(Function &F) {
    IRBuilder<> &IRB = OpBuilder.getIRB();
    Type *Int32Ty = IRB.getInt32Ty();

    return replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);
      Value *Handle =
          createTmpHandleCast(CI->getArgOperand(0), OpBuilder.getHandleType());
      Value *Undef = UndefValue::get(Int32Ty);

      Expected<CallInst *> OpCall = OpBuilder.tryCreateOp(
          OpCode::GetDimensions, {Handle, Undef}, CI->getName(), Int32Ty);
      if (Error E = OpCall.takeError())
        return E;
      Value *Dim = IRB.CreateExtractValue(*OpCall, 0);

      CI->replaceAllUsesWith(Dim);
      CI->eraseFromParent();
      return Error::success();
    });
  }

  [[nodiscard]] bool lowerGetPointer(Function &F) {
    // These should have already been handled in DXILResourceAccess, so we can
    // just clean up the dead prototype.
    assert(F.user_empty() && "getpointer operations should have been removed");
    F.eraseFromParent();
    return false;
  }

  [[nodiscard]] bool lowerBufferStore(Function &F, bool IsRaw) {
    const DataLayout &DL = F.getDataLayout();
    IRBuilder<> &IRB = OpBuilder.getIRB();
    Type *Int8Ty = IRB.getInt8Ty();
    Type *Int32Ty = IRB.getInt32Ty();

    return replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);

      Value *Handle =
          createTmpHandleCast(CI->getArgOperand(0), OpBuilder.getHandleType());
      Value *Index0 = CI->getArgOperand(1);
      Value *Index1 = IsRaw ? CI->getArgOperand(2) : UndefValue::get(Int32Ty);

      Value *Data = CI->getArgOperand(IsRaw ? 3 : 2);
      Type *DataTy = Data->getType();
      Type *ScalarTy = DataTy->getScalarType();

      uint64_t NumElements =
          DL.getTypeSizeInBits(DataTy) / DL.getTypeSizeInBits(ScalarTy);
      Value *Mask =
          ConstantInt::get(Int8Ty, IsRaw ? ~(~0U << NumElements) : 15U);

      // TODO: check that we only have vector or scalar...
      if (NumElements > 4)
        return make_error<StringError>(
            "Buffer store data must have at most 4 elements",
            inconvertibleErrorCode());

      std::array<Value *, 4> DataElements{nullptr, nullptr, nullptr, nullptr};
      if (DataTy == ScalarTy)
        DataElements[0] = Data;
      else {
        // Since we're post-scalarizer, if we see a vector here it's likely
        // constructed solely for the argument of the store. Just use the scalar
        // values from before they're inserted into the temporary.
        auto *IEI = dyn_cast<InsertElementInst>(Data);
        while (IEI) {
          auto *IndexOp = dyn_cast<ConstantInt>(IEI->getOperand(2));
          if (!IndexOp)
            break;
          size_t IndexVal = IndexOp->getZExtValue();
          assert(IndexVal < 4 && "Too many elements for buffer store");
          DataElements[IndexVal] = IEI->getOperand(1);
          IEI = dyn_cast<InsertElementInst>(IEI->getOperand(0));
        }
      }

      // If for some reason we weren't able to forward the arguments from the
      // scalarizer artifact, then we may need to actually extract elements from
      // the vector.
      for (int I = 0, E = NumElements; I < E; ++I)
        if (DataElements[I] == nullptr)
          DataElements[I] =
              IRB.CreateExtractElement(Data, ConstantInt::get(Int32Ty, I));

      // For any elements beyond the length of the vector, we should fill it up
      // with undef - however, for typed buffers we repeat the first element to
      // match DXC.
      for (int I = NumElements, E = 4; I < E; ++I)
        if (DataElements[I] == nullptr)
          DataElements[I] = IsRaw ? UndefValue::get(ScalarTy) : DataElements[0];

      dxil::OpCode Op = OpCode::BufferStore;
      SmallVector<Value *, 9> Args{
          Handle,          Index0,          Index1,          DataElements[0],
          DataElements[1], DataElements[2], DataElements[3], Mask};
      if (IsRaw && MMDI.DXILVersion >= VersionTuple(1, 2)) {
        Op = OpCode::RawBufferStore;
        // RawBufferStore requires the alignment
        Args.push_back(
            ConstantInt::get(Int32Ty, DL.getPrefTypeAlign(ScalarTy).value()));
      }
      Expected<CallInst *> OpCall =
          OpBuilder.tryCreateOp(Op, Args, CI->getName());
      if (Error E = OpCall.takeError())
        return E;

      CI->eraseFromParent();
      // Clean up any leftover `insertelement`s
      auto *IEI = dyn_cast<InsertElementInst>(Data);
      while (IEI && IEI->use_empty()) {
        InsertElementInst *Tmp = IEI;
        IEI = dyn_cast<InsertElementInst>(IEI->getOperand(0));
        Tmp->eraseFromParent();
      }

      return Error::success();
    });
  }

  [[nodiscard]] bool lowerCtpopToCountBits(Function &F) {
    IRBuilder<> &IRB = OpBuilder.getIRB();
    Type *Int32Ty = IRB.getInt32Ty();

    return replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);
      SmallVector<Value *> Args;
      Args.append(CI->arg_begin(), CI->arg_end());

      Type *RetTy = Int32Ty;
      Type *FRT = F.getReturnType();
      if (const auto *VT = dyn_cast<VectorType>(FRT))
        RetTy = VectorType::get(RetTy, VT);

      Expected<CallInst *> OpCall = OpBuilder.tryCreateOp(
          dxil::OpCode::CountBits, Args, CI->getName(), RetTy);
      if (Error E = OpCall.takeError())
        return E;

      // If the result type is 32 bits we can do a direct replacement.
      if (FRT->isIntOrIntVectorTy(32)) {
        CI->replaceAllUsesWith(*OpCall);
        CI->eraseFromParent();
        return Error::success();
      }

      unsigned CastOp;
      unsigned CastOp2;
      if (FRT->isIntOrIntVectorTy(16)) {
        CastOp = Instruction::ZExt;
        CastOp2 = Instruction::SExt;
      } else { // must be 64 bits
        assert(FRT->isIntOrIntVectorTy(64) &&
               "Currently only lowering 16, 32, or 64 bit ctpop to CountBits \
                is supported.");
        CastOp = Instruction::Trunc;
        CastOp2 = Instruction::Trunc;
      }

      // It is correct to replace the ctpop with the dxil op and
      // remove all casts to i32
      bool NeedsCast = false;
      for (User *User : make_early_inc_range(CI->users())) {
        Instruction *I = dyn_cast<Instruction>(User);
        if (I && (I->getOpcode() == CastOp || I->getOpcode() == CastOp2) &&
            I->getType() == RetTy) {
          I->replaceAllUsesWith(*OpCall);
          I->eraseFromParent();
        } else
          NeedsCast = true;
      }

      // It is correct to replace a ctpop with the dxil op and
      // a cast from i32 to the return type of the ctpop
      // the cast is emitted here if there is a non-cast to i32
      // instr which uses the ctpop
      if (NeedsCast) {
        Value *Cast =
            IRB.CreateZExtOrTrunc(*OpCall, F.getReturnType(), "ctpop.cast");
        CI->replaceAllUsesWith(Cast);
      }

      CI->eraseFromParent();
      return Error::success();
    });
  }

  [[nodiscard]] bool lowerLifetimeIntrinsic(Function &F) {
    IRBuilder<> &IRB = OpBuilder.getIRB();
    return replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);
      Value *Ptr = CI->getArgOperand(0);
      assert(Ptr->getType()->isPointerTy() &&
             "Expected operand of lifetime intrinsic to be a pointer");

      auto ZeroOrUndef = [&](Type *Ty) {
        return MMDI.ValidatorVersion < VersionTuple(1, 6)
                   ? Constant::getNullValue(Ty)
                   : UndefValue::get(Ty);
      };

      Value *Val = nullptr;
      if (auto *GV = dyn_cast<GlobalVariable>(Ptr)) {
        if (GV->hasInitializer() || GV->isExternallyInitialized())
          return Error::success();
        Val = ZeroOrUndef(GV->getValueType());
      } else if (auto *AI = dyn_cast<AllocaInst>(Ptr))
        Val = ZeroOrUndef(AI->getAllocatedType());

      assert(Val && "Expected operand of lifetime intrinsic to be a global "
                    "variable or alloca instruction");
      IRB.CreateStore(Val, Ptr, false);

      CI->eraseFromParent();
      return Error::success();
    });
  }

  [[nodiscard]] bool lowerIsFPClass(Function &F) {
    IRBuilder<> &IRB = OpBuilder.getIRB();
    Type *RetTy = IRB.getInt1Ty();

    return replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);
      SmallVector<Value *> Args;
      Value *Fl = CI->getArgOperand(0);
      Args.push_back(Fl);

      dxil::OpCode OpCode;
      Value *T = CI->getArgOperand(1);
      auto *TCI = dyn_cast<ConstantInt>(T);
      switch (TCI->getZExtValue()) {
      case FPClassTest::fcInf:
        OpCode = dxil::OpCode::IsInf;
        break;
      case FPClassTest::fcNan:
        OpCode = dxil::OpCode::IsNaN;
        break;
      case FPClassTest::fcNormal:
        OpCode = dxil::OpCode::IsNormal;
        break;
      case FPClassTest::fcFinite:
        OpCode = dxil::OpCode::IsFinite;
        break;
      default:
        SmallString<128> Msg =
            formatv("Unsupported FPClassTest {0} for DXIL Op Lowering",
                    TCI->getZExtValue());
        return make_error<StringError>(Msg, inconvertibleErrorCode());
      }

      Expected<CallInst *> OpCall =
          OpBuilder.tryCreateOp(OpCode, Args, CI->getName(), RetTy);
      if (Error E = OpCall.takeError())
        return E;

      CI->replaceAllUsesWith(*OpCall);
      CI->eraseFromParent();
      return Error::success();
    });
  }

  bool lowerIntrinsics() {
    bool Updated = false;
    bool HasErrors = false;

    for (Function &F : make_early_inc_range(M.functions())) {
      if (!F.isDeclaration())
        continue;
      Intrinsic::ID ID = F.getIntrinsicID();
      switch (ID) {
      // NOTE: Skip dx_resource_casthandle here. They are
      // resolved after this loop in cleanupHandleCasts.
      case Intrinsic::dx_resource_casthandle:
      // NOTE: llvm.dbg.value is supported as is in DXIL.
      case Intrinsic::dbg_value:
      case Intrinsic::not_intrinsic:
        if (F.use_empty())
          F.eraseFromParent();
        continue;
      default:
        if (F.use_empty())
          F.eraseFromParent();
        else {
          SmallString<128> Msg = formatv(
              "Unsupported intrinsic {0} for DXIL lowering", F.getName());
          M.getContext().emitError(Msg);
          HasErrors |= true;
        }
        break;

#define DXIL_OP_INTRINSIC(OpCode, Intrin, ...)                                 \
  case Intrin:                                                                 \
    HasErrors |= replaceFunctionWithOp(                                        \
        F, OpCode, ArrayRef<IntrinArgSelect>{__VA_ARGS__});                    \
    break;
#include "DXILOperation.inc"
      case Intrinsic::dx_resource_handlefrombinding:
        HasErrors |= lowerHandleFromBinding(F);
        break;
      case Intrinsic::dx_resource_getpointer:
        HasErrors |= lowerGetPointer(F);
        break;
      case Intrinsic::dx_resource_nonuniformindex:
        assert(!CleanupNURI &&
               "overloaded llvm.dx.resource.nonuniformindex intrinsics?");
        CleanupNURI = &F;
        break;
      case Intrinsic::dx_resource_load_typedbuffer:
        HasErrors |= lowerTypedBufferLoad(F, /*HasCheckBit=*/true);
        break;
      case Intrinsic::dx_resource_store_typedbuffer:
        HasErrors |= lowerBufferStore(F, /*IsRaw=*/false);
        break;
      case Intrinsic::dx_resource_load_rawbuffer:
        HasErrors |= lowerRawBufferLoad(F);
        break;
      case Intrinsic::dx_resource_store_rawbuffer:
        HasErrors |= lowerBufferStore(F, /*IsRaw=*/true);
        break;
      case Intrinsic::dx_resource_load_cbufferrow_2:
      case Intrinsic::dx_resource_load_cbufferrow_4:
      case Intrinsic::dx_resource_load_cbufferrow_8:
        HasErrors |= lowerCBufferLoad(F);
        break;
      case Intrinsic::dx_resource_updatecounter:
        HasErrors |= lowerUpdateCounter(F);
        break;
      case Intrinsic::dx_resource_getdimensions_x:
        HasErrors |= lowerGetDimensionsX(F);
        break;
      case Intrinsic::ctpop:
        HasErrors |= lowerCtpopToCountBits(F);
        break;
      case Intrinsic::lifetime_start:
      case Intrinsic::lifetime_end:
        if (F.use_empty())
          F.eraseFromParent();
        else {
          if (MMDI.DXILVersion < VersionTuple(1, 6))
            HasErrors |= lowerLifetimeIntrinsic(F);
          else
            continue;
        }
        break;
      case Intrinsic::is_fpclass:
        HasErrors |= lowerIsFPClass(F);
        break;
      }
      Updated = true;
    }
    if (Updated && !HasErrors) {
      cleanupHandleCasts();
      cleanupNonUniformResourceIndexCalls();
    }

    return Updated;
  }
};
} // namespace

PreservedAnalyses DXILOpLowering::run(Module &M, ModuleAnalysisManager &MAM) {
  DXILResourceMap &DRM = MAM.getResult<DXILResourceAnalysis>(M);
  DXILResourceTypeMap &DRTM = MAM.getResult<DXILResourceTypeAnalysis>(M);
  const ModuleMetadataInfo MMDI = MAM.getResult<DXILMetadataAnalysis>(M);

  const bool MadeChanges = OpLowerer(M, DRM, DRTM, MMDI).lowerIntrinsics();
  if (!MadeChanges)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<DXILResourceAnalysis>();
  PA.preserve<DXILMetadataAnalysis>();
  PA.preserve<ShaderFlagsAnalysis>();
  PA.preserve<RootSignatureAnalysis>();
  return PA;
}

namespace {
class DXILOpLoweringLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override {
    DXILResourceMap &DRM =
        getAnalysis<DXILResourceWrapperPass>().getResourceMap();
    DXILResourceTypeMap &DRTM =
        getAnalysis<DXILResourceTypeWrapperPass>().getResourceTypeMap();
    const ModuleMetadataInfo MMDI =
        getAnalysis<DXILMetadataAnalysisWrapperPass>().getModuleMetadata();

    return OpLowerer(M, DRM, DRTM, MMDI).lowerIntrinsics();
  }
  StringRef getPassName() const override { return "DXIL Op Lowering"; }
  DXILOpLoweringLegacy() : ModulePass(ID) {}

  static char ID; // Pass identification.
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<DXILResourceTypeWrapperPass>();
    AU.addRequired<DXILResourceWrapperPass>();
    AU.addRequired<DXILMetadataAnalysisWrapperPass>();
    AU.addPreserved<DXILResourceWrapperPass>();
    AU.addPreserved<DXILMetadataAnalysisWrapperPass>();
    AU.addPreserved<ShaderFlagsAnalysisWrapper>();
    AU.addPreserved<RootSignatureAnalysisWrapper>();
  }
};
char DXILOpLoweringLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DXILOpLoweringLegacy, DEBUG_TYPE, "DXIL Op Lowering",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DXILResourceTypeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DXILResourceWrapperPass)
INITIALIZE_PASS_END(DXILOpLoweringLegacy, DEBUG_TYPE, "DXIL Op Lowering", false,
                    false)

ModulePass *llvm::createDXILOpLoweringLegacyPass() {
  return new DXILOpLoweringLegacy();
}
