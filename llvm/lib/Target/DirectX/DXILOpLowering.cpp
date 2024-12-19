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
#include "DXILResourceAnalysis.h"
#include "DXILShaderFlags.h"
#include "DirectX.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
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
  DXILBindingMap &DBM;
  DXILResourceTypeMap &DRTM;
  SmallVector<CallInst *> CleanupCasts;

public:
  OpLowerer(Module &M, DXILBindingMap &DBM, DXILResourceTypeMap &DRTM)
      : M(M), OpBuilder(M), DBM(DBM), DRTM(DRTM) {}

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
        DiagnosticInfoUnsupported Diag(*CI->getFunction(), Message,
                                       CI->getDebugLoc());
        M.getContext().diagnose(Diag);
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

  [[nodiscard]] bool
  replaceFunctionWithOp(Function &F, dxil::OpCode DXILOp,
                        ArrayRef<IntrinArgSelect> ArgSelects) {
    bool IsVectorArgExpansion = isVectorArgExpansion(F);
    assert(!(IsVectorArgExpansion && ArgSelects.size()) &&
           "Cann't do vector arg expansion when using arg selects.");
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
      } else if (IsVectorArgExpansion) {
        Args = argVectorFlatten(CI, OpBuilder.getIRB());
      } else {
        Args.append(CI->arg_begin(), CI->arg_end());
      }

      Expected<CallInst *> OpCall =
          OpBuilder.tryCreateOp(DXILOp, Args, CI->getName(), F.getReturnType());
      if (Error E = OpCall.takeError())
        return E;

      CI->replaceAllUsesWith(*OpCall);
      CI->eraseFromParent();
      return Error::success();
    });
  }

  [[nodiscard]] bool replaceFunctionWithNamedStructOp(
      Function &F, dxil::OpCode DXILOp, Type *NewRetTy,
      llvm::function_ref<Error(CallInst *CI, CallInst *Op)> ReplaceUses) {
    bool IsVectorArgExpansion = isVectorArgExpansion(F);
    return replaceFunction(F, [&](CallInst *CI) -> Error {
      SmallVector<Value *> Args;
      OpBuilder.getIRB().SetInsertPoint(CI);
      if (IsVectorArgExpansion) {
        SmallVector<Value *> NewArgs = argVectorFlatten(CI, OpBuilder.getIRB());
        Args.append(NewArgs.begin(), NewArgs.end());
      } else
        Args.append(CI->arg_begin(), CI->arg_end());

      Expected<CallInst *> OpCall =
          OpBuilder.tryCreateOp(DXILOp, Args, CI->getName(), NewRetTy);
      if (Error E = OpCall.takeError())
        return E;
      if (Error E = ReplaceUses(CI, *OpCall))
        return E;

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

  [[nodiscard]] bool lowerToCreateHandle(Function &F) {
    IRBuilder<> &IRB = OpBuilder.getIRB();
    Type *Int8Ty = IRB.getInt8Ty();
    Type *Int32Ty = IRB.getInt32Ty();

    return replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);

      auto *It = DBM.find(CI);
      assert(It != DBM.end() && "Resource not in map?");
      dxil::ResourceBindingInfo &RI = *It;

      const auto &Binding = RI.getBinding();
      dxil::ResourceClass RC = DRTM[RI.getHandleTy()].getResourceClass();

      Value *IndexOp = CI->getArgOperand(3);
      if (Binding.LowerBound != 0)
        IndexOp = IRB.CreateAdd(IndexOp,
                                ConstantInt::get(Int32Ty, Binding.LowerBound));

      std::array<Value *, 4> Args{
          ConstantInt::get(Int8Ty, llvm::to_underlying(RC)),
          ConstantInt::get(Int32Ty, Binding.RecordID), IndexOp,
          CI->getArgOperand(4)};
      Expected<CallInst *> OpCall =
          OpBuilder.tryCreateOp(OpCode::CreateHandle, Args, CI->getName());
      if (Error E = OpCall.takeError())
        return E;

      Value *Cast = createTmpHandleCast(*OpCall, CI->getType());

      removeResourceGlobals(CI);

      CI->replaceAllUsesWith(Cast);
      CI->eraseFromParent();
      return Error::success();
    });
  }

  [[nodiscard]] bool lowerToBindAndAnnotateHandle(Function &F) {
    IRBuilder<> &IRB = OpBuilder.getIRB();
    Type *Int32Ty = IRB.getInt32Ty();

    return replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);

      auto *It = DBM.find(CI);
      assert(It != DBM.end() && "Resource not in map?");
      dxil::ResourceBindingInfo &RI = *It;

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
      std::array<Value *, 3> BindArgs{ResBind, IndexOp, CI->getArgOperand(4)};
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

      removeResourceGlobals(CI);

      CI->replaceAllUsesWith(Cast);
      CI->eraseFromParent();

      return Error::success();
    });
  }

  /// Lower `dx.resource.handlefrombinding` intrinsics depending on the shader
  /// model and taking into account binding information from
  /// DXILResourceBindingAnalysis.
  bool lowerHandleFromBinding(Function &F) {
    Triple TT(Triple(M.getTargetTriple()));
    if (TT.getDXILVersion() < VersionTuple(1, 6))
      return lowerToCreateHandle(F);
    return lowerToBindAndAnnotateHandle(F);
  }

  Error replaceSplitDoubleCallUsages(CallInst *Intrin, CallInst *Op) {
    for (Use &U : make_early_inc_range(Intrin->uses())) {
      if (auto *EVI = dyn_cast<ExtractValueInst>(U.getUser())) {

        if (EVI->getNumIndices() != 1)
          return createStringError(std::errc::invalid_argument,
                                   "Splitdouble has only 2 elements");
        EVI->setOperand(0, Op);
      } else {
        return make_error<StringError>(
            "Splitdouble use is not ExtractValueInst",
            inconvertibleErrorCode());
      }
    }

    Intrin->eraseFromParent();

    return Error::success();
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

      OldResult = cast<Instruction>(
          IRB.CreateExtractValue(Op, 0, OldResult->getName()));
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

      Value *Vec = UndefValue::get(OldTy);
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

  [[nodiscard]] bool lowerGetPointer(Function &F) {
    // These should have already been handled in DXILResourceAccess, so we can
    // just clean up the dead prototype.
    assert(F.user_empty() && "getpointer operations should have been removed");
    F.eraseFromParent();
    return false;
  }

  [[nodiscard]] bool lowerTypedBufferStore(Function &F) {
    IRBuilder<> &IRB = OpBuilder.getIRB();
    Type *Int8Ty = IRB.getInt8Ty();
    Type *Int32Ty = IRB.getInt32Ty();

    return replaceFunction(F, [&](CallInst *CI) -> Error {
      IRB.SetInsertPoint(CI);

      Value *Handle =
          createTmpHandleCast(CI->getArgOperand(0), OpBuilder.getHandleType());
      Value *Index0 = CI->getArgOperand(1);
      Value *Index1 = UndefValue::get(Int32Ty);
      // For typed stores, the mask must always cover all four elements.
      Constant *Mask = ConstantInt::get(Int8Ty, 0xF);

      Value *Data = CI->getArgOperand(2);
      auto *DataTy = dyn_cast<FixedVectorType>(Data->getType());
      if (!DataTy || DataTy->getNumElements() != 4)
        return make_error<StringError>(
            "typedBufferStore data must be a vector of 4 elements",
            inconvertibleErrorCode());

      // Since we're post-scalarizer, we likely have a vector that's constructed
      // solely for the argument of the store. If so, just use the scalar values
      // from before they're inserted into the temporary.
      std::array<Value *, 4> DataElements{nullptr, nullptr, nullptr, nullptr};
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

      // If for some reason we weren't able to forward the arguments from the
      // scalarizer artifact, then we need to actually extract elements from the
      // vector.
      for (int I = 0, E = 4; I != E; ++I)
        if (DataElements[I] == nullptr)
          DataElements[I] =
              IRB.CreateExtractElement(Data, ConstantInt::get(Int32Ty, I));

      std::array<Value *, 8> Args{
          Handle,          Index0,          Index1,          DataElements[0],
          DataElements[1], DataElements[2], DataElements[3], Mask};
      Expected<CallInst *> OpCall =
          OpBuilder.tryCreateOp(OpCode::BufferStore, Args, CI->getName());
      if (Error E = OpCall.takeError())
        return E;

      CI->eraseFromParent();
      // Clean up any leftover `insertelement`s
      IEI = dyn_cast<InsertElementInst>(Data);
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

  bool lowerIntrinsics() {
    bool Updated = false;
    bool HasErrors = false;

    for (Function &F : make_early_inc_range(M.functions())) {
      if (!F.isDeclaration())
        continue;
      Intrinsic::ID ID = F.getIntrinsicID();
      switch (ID) {
      default:
        continue;
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
      case Intrinsic::dx_resource_load_typedbuffer:
        HasErrors |= lowerTypedBufferLoad(F, /*HasCheckBit=*/false);
        break;
      case Intrinsic::dx_resource_loadchecked_typedbuffer:
        HasErrors |= lowerTypedBufferLoad(F, /*HasCheckBit=*/true);
        break;
      case Intrinsic::dx_resource_store_typedbuffer:
        HasErrors |= lowerTypedBufferStore(F);
        break;
      case Intrinsic::dx_resource_updatecounter:
        HasErrors |= lowerUpdateCounter(F);
        break;
      // TODO: this can be removed when
      // https://github.com/llvm/llvm-project/issues/113192 is fixed
      case Intrinsic::dx_splitdouble:
        HasErrors |= replaceFunctionWithNamedStructOp(
            F, OpCode::SplitDouble,
            OpBuilder.getSplitDoubleType(M.getContext()),
            [&](CallInst *CI, CallInst *Op) {
              return replaceSplitDoubleCallUsages(CI, Op);
            });
        break;
      case Intrinsic::ctpop:
        HasErrors |= lowerCtpopToCountBits(F);
        break;
      }
      Updated = true;
    }
    if (Updated && !HasErrors)
      cleanupHandleCasts();

    return Updated;
  }
};
} // namespace

PreservedAnalyses DXILOpLowering::run(Module &M, ModuleAnalysisManager &MAM) {
  DXILBindingMap &DBM = MAM.getResult<DXILResourceBindingAnalysis>(M);
  DXILResourceTypeMap &DRTM = MAM.getResult<DXILResourceTypeAnalysis>(M);

  bool MadeChanges = OpLowerer(M, DBM, DRTM).lowerIntrinsics();
  if (!MadeChanges)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<DXILResourceBindingAnalysis>();
  PA.preserve<DXILMetadataAnalysis>();
  PA.preserve<ShaderFlagsAnalysis>();
  return PA;
}

namespace {
class DXILOpLoweringLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override {
    DXILBindingMap &DBM =
        getAnalysis<DXILResourceBindingWrapperPass>().getBindingMap();
    DXILResourceTypeMap &DRTM =
        getAnalysis<DXILResourceTypeWrapperPass>().getResourceTypeMap();

    return OpLowerer(M, DBM, DRTM).lowerIntrinsics();
  }
  StringRef getPassName() const override { return "DXIL Op Lowering"; }
  DXILOpLoweringLegacy() : ModulePass(ID) {}

  static char ID; // Pass identification.
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<DXILResourceTypeWrapperPass>();
    AU.addRequired<DXILResourceBindingWrapperPass>();
    AU.addPreserved<DXILResourceBindingWrapperPass>();
    AU.addPreserved<DXILResourceMDWrapper>();
    AU.addPreserved<DXILMetadataAnalysisWrapperPass>();
    AU.addPreserved<ShaderFlagsAnalysisWrapper>();
  }
};
char DXILOpLoweringLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DXILOpLoweringLegacy, DEBUG_TYPE, "DXIL Op Lowering",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DXILResourceTypeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DXILResourceBindingWrapperPass)
INITIALIZE_PASS_END(DXILOpLoweringLegacy, DEBUG_TYPE, "DXIL Op Lowering", false,
                    false)

ModulePass *llvm::createDXILOpLoweringLegacyPass() {
  return new DXILOpLoweringLegacy();
}
