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

#include <optional>

using namespace llvm;

struct LookupTableInfo {
  Value *Index;
  SmallVector<Constant *> Ptrs;
  SmallVector<GlobalVariable *> GVOps;
};

static std::optional<LookupTableInfo>
shouldConvertToRelLookupTable(const Module &M, GlobalVariable &GV) {
  // If lookup table has more than one user,
  // do not generate a relative lookup table.
  // This is to simplify the analysis that needs to be done for this pass.
  // TODO: Add support for lookup tables with multiple uses.
  // For ex, this can happen when a function that uses a lookup table gets
  // inlined into multiple call sites.
  //
  // If the original lookup table does not have local linkage and is
  // not dso_local, do not generate a relative lookup table.
  // This optimization creates a relative lookup table that consists of
  // offsets between the start of the lookup table and its elements.
  // To be able to generate these offsets, relative lookup table and
  // its elements should have internal linkage and be dso_local, which means
  // that they should resolve to symbols within the same linkage unit.
  if (!GV.hasInitializer() || !GV.isConstant() || !GV.hasOneUse() ||
      !GV.hasLocalLinkage() || !GV.isDSOLocal() || !GV.isImplicitDSOLocal())
    return std::nullopt;

  auto *GEP = dyn_cast<GetElementPtrInst>(GV.use_begin()->getUser());
  if (!GEP || !GEP->hasOneUse())
    return std::nullopt;

  auto *Load = dyn_cast<LoadInst>(GEP->use_begin()->getUser());
  if (!Load)
    return std::nullopt;

  // If values are not 64-bit pointers, do not generate a relative lookup table.
  const DataLayout &DL = M.getDataLayout();
  Type *ElemType = Load->getType();
  if (!ElemType->isPointerTy() || DL.getPointerTypeSizeInBits(ElemType) != 64)
    return std::nullopt;

  // Make sure this is a gep of the form GV + scale*var.
  unsigned IndexWidth =
      DL.getIndexTypeSizeInBits(Load->getPointerOperand()->getType());
  SmallMapVector<Value *, APInt, 4> VarOffsets;
  APInt ConstOffset(IndexWidth, 0);
  if (!GEP->collectOffset(DL, IndexWidth, VarOffsets, ConstOffset) ||
      !ConstOffset.isZero() || VarOffsets.size() != 1)
    return std::nullopt;

  // This can't be a pointer lookup table if the stride is smaller than a
  // pointer.
  LookupTableInfo Info;
  Info.Index = VarOffsets.front().first;
  const APInt &Stride = VarOffsets.front().second;
  if (Stride.ult(DL.getTypeStoreSize(ElemType)))
    return std::nullopt;

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

  APInt Offset(IndexWidth, 0);
  uint64_t GVSize = GV.getGlobalSize(DL);
  for (; Offset.ult(GVSize); Offset += Stride) {
    Constant *C =
        ConstantFoldLoadFromConst(GV.getInitializer(), ElemType, Offset, DL);
    if (!C)
      return std::nullopt;

    GlobalValue *GVOp;
    APInt GVOffset;

    // If an operand is not a constant offset from a lookup table,
    // do not generate a relative lookup table.
    if (!IsConstantOffsetFromGlobal(C, GVOp, GVOffset, DL))
      return std::nullopt;

    // If operand is mutable, do not generate a relative lookup table.
    auto *GlobalVarOp = dyn_cast<GlobalVariable>(GVOp);
    if (!GlobalVarOp || !GlobalVarOp->isConstant())
      return std::nullopt;

    if (!GlobalVarOp->hasLocalLinkage() || !GlobalVarOp->isDSOLocal() ||
        !GlobalVarOp->isImplicitDSOLocal())
      return std::nullopt;

    // On AArch64 small code model, the text-to-data span can be up to 4GB,
    // which exceeds 32-bit signed relative offsets. Avoid converting if the
    // target operand requires dynamic relocations (placing it in .data.rel.ro
    // in the data segment rather than .rodata in the text segment).
    if (TT.isAArch64() &&
        (!GlobalVarOp->hasInitializer() ||
         GlobalVarOp->getInitializer()->needsDynamicRelocation()))
      return std::nullopt;

    if (ShouldDropUnnamedAddr)
      Info.GVOps.push_back(GlobalVarOp);

    Info.Ptrs.push_back(C);
  }

  return Info;
}

static GlobalVariable *createRelLookupTable(LookupTableInfo &Info,
                                            GlobalVariable &LookupTable) {
  Module &M = *LookupTable.getParent();
  ArrayType *IntArrayTy =
      ArrayType::get(Type::getInt32Ty(M.getContext()), Info.Ptrs.size());

  GlobalVariable *RelLookupTable = new GlobalVariable(
      M, IntArrayTy, LookupTable.isConstant(), LookupTable.getLinkage(),
      nullptr, LookupTable.getName() + ".rel", &LookupTable,
      LookupTable.getThreadLocalMode(), LookupTable.getAddressSpace(),
      LookupTable.isExternallyInitialized());

  Type *IntPtrTy = M.getDataLayout().getIntPtrType(M.getContext());
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  Constant *Base = ConstantExpr::getPtrToInt(RelLookupTable, IntPtrTy);

  uint64_t Idx = 0;
  SmallVector<Constant *, 64> RelLookupTableContents(Info.Ptrs.size());

  for (Constant *Element : Info.Ptrs) {
    Constant *Target = ConstantExpr::getPtrToInt(Element, IntPtrTy);
    Constant *Sub = ConstantExpr::getSub(Target, Base);
    Constant *RelOffset = ConstantExpr::getTrunc(Sub, Int32Ty);
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
  for (GlobalVariable *GVOp : Info.GVOps)
    GVOp->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

  GetElementPtrInst *GEP =
      cast<GetElementPtrInst>(LookupTable.use_begin()->getUser());
  LoadInst *Load = cast<LoadInst>(GEP->use_begin()->getUser());

  Module &M = *LookupTable.getParent();
  BasicBlock *BB = GEP->getParent();
  IRBuilder<> Builder(BB);

  // Generate an array that consists of relative offsets.
  GlobalVariable *RelLookupTable =
      createRelLookupTable(Info, LookupTable);

  // Place new instruction sequence before GEP.
  Builder.SetInsertPoint(GEP);
  IntegerType *IntTy = cast<IntegerType>(Info.Index->getType());
  Value *Offset = Builder.CreateShl(Info.Index, ConstantInt::get(IntTy, 2),
                                    "reltable.shift");

  // Insert the call to load.relative intrinsic before LOAD.
  // GEP might not be immediately followed by a LOAD, like it can be hoisted
  // outside the loop or another instruction might be inserted them in between.
  Builder.SetInsertPoint(Load);
  Function *LoadRelIntrinsic = llvm::Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::load_relative, {Info.Index->getType()});

  // Create a call to load.relative intrinsic that computes the target address
  // by adding base address (lookup table address) and relative offset.
  Value *Result = Builder.CreateCall(LoadRelIntrinsic, {RelLookupTable, Offset},
                                     "reltable.intrinsic");

  // Replace load instruction with the new generated instruction sequence.
  Load->replaceAllUsesWith(Result);
  // Remove Load and GEP instructions.
  Load->eraseFromParent();
  GEP->eraseFromParent();
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
    std::optional<LookupTableInfo> Info = shouldConvertToRelLookupTable(M, GV);
    if (!Info)
      continue;

    convertToRelLookupTable(*Info, GV);

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
