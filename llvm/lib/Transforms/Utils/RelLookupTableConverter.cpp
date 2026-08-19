//===- RelLookupTableConverterPass - Rel Table Conv -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements relative lookup table converter that converts
// lookup tables to relative lookup tables to make them PIC-friendly.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/RelLookupTableConverter.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace llvm;

struct LookupTableUseInfo {
  GetElementPtrInst *GEP;
  LoadInst *Load;
  Value *Index;
};

struct LookupTableInfo {
  SmallVector<LookupTableUseInfo, 4> Uses;
  SmallVector<Constant *> Ptrs;
};

static bool shouldConvertToRelLookupTable(LookupTableInfo &Info, Module &M,
                                          GlobalVariable &GV) {
  // If the original lookup table does not have local linkage and is
  // not dso_local, do not generate a relative lookup table.
  // This optimization creates a relative lookup table that consists of
  // offsets between the start of the lookup table and its elements.
  // To be able to generate these offsets, relative lookup table and
  // its elements should have internal linkage and be dso_local, which means
  // that they should resolve to symbols within the same linkage unit.
  if (!GV.hasInitializer() || !GV.isConstant() || !GV.hasLocalLinkage() ||
      !GV.isDSOLocal() || !GV.isImplicitDSOLocal())
    return false;

  const DataLayout &DL = M.getDataLayout();
  std::optional<APInt> CommonStride;
  Type *CommonElemType = nullptr;

  for (User *U : GV.users()) {
    auto *GEP = dyn_cast<GetElementPtrInst>(U);
    if (!GEP || !GEP->hasOneUse())
      return false;

    auto *Load = dyn_cast<LoadInst>(GEP->use_begin()->getUser());
    if (!Load || !Load->hasOneUse())
      return false;

    // If values are not 64-bit pointers, do not generate a relative lookup
    // table.
    Type *ElemType = Load->getType();
    if (!ElemType->isPointerTy() || DL.getPointerTypeSizeInBits(ElemType) != 64)
      return false;

    if (!CommonElemType)
      CommonElemType = ElemType;
    else if (CommonElemType != ElemType)
      return false;

    // Make sure this is a gep of the form GV + scale*var.
    unsigned IndexWidth =
        DL.getIndexTypeSizeInBits(Load->getPointerOperand()->getType());
    SmallMapVector<Value *, APInt, 4> VarOffsets;
    APInt ConstOffset(IndexWidth, 0);
    if (!GEP->collectOffset(DL, IndexWidth, VarOffsets, ConstOffset) ||
        !ConstOffset.isZero() || VarOffsets.size() != 1)
      return false;

    const APInt &Stride = VarOffsets.front().second;
    if (!CommonStride)
      CommonStride = Stride;
    else if (CommonStride != Stride)
      return false;

    // This can't be a pointer lookup table if the stride is smaller than a
    // pointer.
    if (Stride.ult(DL.getTypeStoreSize(ElemType)))
      return false;

    Value *Index = VarOffsets.front().first;
    Info.Uses.push_back({GEP, Load, Index});
  }

  if (Info.Uses.empty())
    return false;

  SmallVector<GlobalVariable *, 4> GVOps;
  Triple TT = M.getTargetTriple();
  // FIXME: This should be removed in the future.
  bool ShouldDropUnnamedAddr =
      // Drop unnamed_addr to avoid matching pattern in
      // `handleIndirectSymViaGOTPCRel`, which generates GOTPCREL relocations
      // not supported by the GNU linker and LLD versions below 18 on aarch64.
      TT.isAArch64()
      // Apple's ld64 (and ld-prime on Xcode 15.2) miscompile something on
      // x86_64-apple-darwin. See
      // https://github.com/rust-lang/rust/issues/140686 and
      // https://github.com/rust-lang/rust/issues/141306.
      || (TT.isX86() && TT.isOSDarwin());

  unsigned IndexWidth = DL.getIndexTypeSizeInBits(
      Info.Uses.front().Load->getPointerOperand()->getType());
  APInt Offset(IndexWidth, 0);
  uint64_t GVSize = GV.getGlobalSize(DL);
  for (; Offset.ult(GVSize); Offset += *CommonStride) {
    Constant *C = ConstantFoldLoadFromConst(GV.getInitializer(), CommonElemType,
                                            Offset, DL);
    if (!C)
      return false;

    GlobalValue *GVOp;
    APInt GVOffset;

    // If an operand is not a constant offset from a lookup table,
    // do not generate a relative lookup table.
    if (!IsConstantOffsetFromGlobal(C, GVOp, GVOffset, DL))
      return false;

    // If operand is mutable, do not generate a relative lookup table.
    auto *GlovalVarOp = dyn_cast<GlobalVariable>(GVOp);
    if (!GlovalVarOp || !GlovalVarOp->isConstant())
      return false;

    if (!GlovalVarOp->hasLocalLinkage() ||
        !GlovalVarOp->isDSOLocal() ||
        !GlovalVarOp->isImplicitDSOLocal())
      return false;

    if (ShouldDropUnnamedAddr)
      GVOps.push_back(GlovalVarOp);

    Info.Ptrs.push_back(C);
  }

  if (ShouldDropUnnamedAddr)
    for (auto *GVOp : GVOps)
      GVOp->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

  return true;
}

static GlobalVariable *createRelLookupTable(LookupTableInfo &Info, Module &M,
                                            GlobalVariable &LookupTable) {
  ArrayType *IntArrayTy =
      ArrayType::get(Type::getInt32Ty(M.getContext()), Info.Ptrs.size());

  GlobalVariable *RelLookupTable = new GlobalVariable(
      M, IntArrayTy, LookupTable.isConstant(), LookupTable.getLinkage(),
      nullptr, LookupTable.getName() + ".rel", &LookupTable,
      LookupTable.getThreadLocalMode(), LookupTable.getAddressSpace(),
      LookupTable.isExternallyInitialized());

  uint64_t Idx = 0;
  SmallVector<Constant *, 64> RelLookupTableContents(Info.Ptrs.size());

  for (Constant *Element : Info.Ptrs) {
    Type *IntPtrTy = M.getDataLayout().getIntPtrType(M.getContext());
    Constant *Base = llvm::ConstantExpr::getPtrToInt(RelLookupTable, IntPtrTy);
    Constant *Target = llvm::ConstantExpr::getPtrToInt(Element, IntPtrTy);
    Constant *Sub = llvm::ConstantExpr::getSub(Target, Base);
    Constant *RelOffset =
        llvm::ConstantExpr::getTrunc(Sub, Type::getInt32Ty(M.getContext()));
    RelLookupTableContents[Idx++] = RelOffset;
  }

  Constant *Initializer =
      ConstantArray::get(IntArrayTy, RelLookupTableContents);
  RelLookupTable->setInitializer(Initializer);
  RelLookupTable->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  RelLookupTable->setAlignment(llvm::Align(4));
  return RelLookupTable;
}

static void convertToRelLookupTable(LookupTableInfo &Info,
                                    GlobalVariable &LookupTable) {
  Module &M = *LookupTable.getParent();

  // Generate an array that consists of relative offsets.
  GlobalVariable *RelLookupTable = createRelLookupTable(Info, M, LookupTable);

  for (auto &U : Info.Uses) {
    GetElementPtrInst *GEP = U.GEP;
    LoadInst *Load = U.Load;
    Value *Index = U.Index;

    BasicBlock *BB = GEP->getParent();
    IRBuilder<> Builder(BB);

    // Place new instruction sequence before GEP.
    Builder.SetInsertPoint(GEP);
    IntegerType *IntTy = cast<IntegerType>(Index->getType());
    Value *Offset =
        Builder.CreateShl(Index, ConstantInt::get(IntTy, 2), "reltable.shift");

    // Insert the call to load.relative intrinsic before LOAD.
    // GEP might not be immediately followed by a LOAD, like it can be hoisted
    // outside the loop or another instruction might be inserted them in
    // between.
    Builder.SetInsertPoint(Load);
    Function *LoadRelIntrinsic = llvm::Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::load_relative, {Index->getType()});

    // Create a call to load.relative intrinsic that computes the target address
    // by adding base address (lookup table address) and relative offset.
    Value *Result = Builder.CreateCall(
        LoadRelIntrinsic, {RelLookupTable, Offset}, "reltable.intrinsic");

    // Replace load instruction with the new generated instruction sequence.
    Load->replaceAllUsesWith(Result);
    // Remove Load instruction.
    Load->eraseFromParent();
    GEP->eraseFromParent();
  }
}

// Convert lookup tables to relative lookup tables in the module.
static bool convertToRelativeLookupTables(
    Module &M, function_ref<TargetTransformInfo &(Function &)> GetTTI) {
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    // Check if we have a target that supports relative lookup tables.
    if (!GetTTI(F).shouldBuildRelLookupTables())
      return false;

    // We assume that the result is independent of the checked function.
    break;
  }

  bool Changed = false;

  for (GlobalVariable &GV : llvm::make_early_inc_range(M.globals())) {
    LookupTableInfo Info;
    if (!shouldConvertToRelLookupTable(Info, M, GV))
      continue;

    convertToRelLookupTable(Info, GV);

    // Remove the original lookup table.
    GV.eraseFromParent();

    Changed = true;
  }

  return Changed;
}

PreservedAnalyses RelLookupTableConverterPass::run(Module &M,
                                                   ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  auto GetTTI = [&](Function &F) -> TargetTransformInfo & {
    return FAM.getResult<TargetIRAnalysis>(F);
  };

  if (!convertToRelativeLookupTables(M, GetTTI))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
