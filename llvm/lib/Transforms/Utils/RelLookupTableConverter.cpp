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

struct LookupTableInfo {
  SmallVector<Constant *> Ptrs;
};

struct LookupTableUse {
  Value *Index;
  GetElementPtrInst *GEP;
  LoadInst *Load;
};

struct AccessPattern {
  APInt Stride;
  Type *ElemType;

  bool operator==(const AccessPattern &Other) const {
    return Stride == Other.Stride && ElemType == Other.ElemType;
  }
  bool operator!=(const AccessPattern &Other) const {
    return !(*this == Other);
  }
};

static bool analyzeLookupTableUse(AccessPattern &Pattern,
                                  LookupTableUse &UseInfo, User *Usr) {
  // TODO: Allow multiple load users of GEP.
  UseInfo.GEP = dyn_cast<GetElementPtrInst>(Usr);
  if (!UseInfo.GEP || !UseInfo.GEP->hasOneUse())
    return false;

  UseInfo.Load = dyn_cast<LoadInst>(UseInfo.GEP->use_begin()->getUser());
  if (!UseInfo.Load || UseInfo.Load->isVolatile())
    return false;

  // If values are not 64-bit pointers, do not generate a relative lookup table.
  const DataLayout &DL = UseInfo.Load->getDataLayout();
  Pattern.ElemType = UseInfo.Load->getType();
  if (!Pattern.ElemType->isPointerTy() ||
      DL.getPointerTypeSizeInBits(Pattern.ElemType) != 64)
    return false;

  // Make sure this is a gep of the form GV + scale*var.
  unsigned IndexWidth = DL.getIndexTypeSizeInBits(UseInfo.GEP->getType());
  SmallMapVector<Value *, APInt, 4> VarOffsets;
  APInt ConstOffset(IndexWidth, 0);
  if (!UseInfo.GEP->collectOffset(DL, IndexWidth, VarOffsets, ConstOffset) ||
      !ConstOffset.isZero() || VarOffsets.size() != 1)
    return false;

  // This can't be a pointer lookup table if the stride is smaller than a
  // pointer.
  UseInfo.Index = VarOffsets.front().first;
  Pattern.Stride = VarOffsets.front().second;
  if (Pattern.Stride.ult(DL.getTypeStoreSize(Pattern.ElemType)))
    return false;

  return true;
}

static bool shouldConvertToRelLookupTable(LookupTableInfo &Info,
                                          SmallVectorImpl<LookupTableUse> &Uses,
                                          Module &M, GlobalVariable &GV) {
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

  std::optional<AccessPattern> Pattern;
  for (User *Usr : GV.users()) {
    AccessPattern CurPattern;
    LookupTableUse CurUse;
    if (!analyzeLookupTableUse(CurPattern, CurUse, Usr))
      return false;

    // Make sure the access pattern is always the same
    if (!Pattern)
      Pattern = CurPattern;
    else if (*Pattern != CurPattern)
      return false;

    Uses.push_back(CurUse);
  }
  if (!Pattern)
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

  const DataLayout &DL = M.getDataLayout();
  APInt Offset(Pattern->Stride.getBitWidth(), 0);
  uint64_t GVSize = GV.getGlobalSize(DL);
  bool HasNonPoison = false;
  for (; Offset.ult(GVSize); Offset += Pattern->Stride) {
    Constant *C = ConstantFoldLoadFromConst(GV.getInitializer(),
                                            Pattern->ElemType, Offset, DL);
    if (!C)
      return false;

    if (isa<PoisonValue>(C)) {
      Info.Ptrs.push_back(C);
      continue;
    }

    GlobalValue *GVOp;
    APInt GVOffset;

    // If an operand is not a constant offset from a lookup table,
    // do not generate a relative lookup table.
    if (!IsConstantOffsetFromGlobal(C, GVOp, GVOffset, DL))
      return false;

    // If operand is mutable, do not generate a relative lookup table.
    auto *GlobalVarOp = dyn_cast<GlobalVariable>(GVOp);
    if (!GlobalVarOp || !GlobalVarOp->isConstant())
      return false;

    // Weak declarations have to be excluded because they may be null.
    // Weak definitions are excluded because the assembler currently incorrectly
    // resolves them against the local symbol, instead of emitting a relocation
    // (https://github.com/llvm/llvm-project/issues/244540).
    // Non-implicit dso_local is excluded because we currently do not convey to
    // the linker that the symbol is non-preemptible, which may result in a
    // relocation error.
    if (GlobalVarOp->isWeakForLinker() || !GlobalVarOp->isImplicitDSOLocal())
      return false;

    // On AArch64 small code model, the text-to-data span can be up to 4GB,
    // which exceeds 32-bit signed relative offsets. Avoid converting if the
    // target operand requires dynamic relocations (placing it in .data.rel.ro
    // in the data segment rather than .rodata in the text segment).
    if (TT.isAArch64() &&
        (!GlobalVarOp->hasInitializer() ||
         GlobalVarOp->getInitializer()->needsDynamicRelocation()))
      return false;

    if (ShouldDropUnnamedAddr)
      GVOps.push_back(GlobalVarOp);

    HasNonPoison = true;
    Info.Ptrs.push_back(C);
  }

  if (!HasNonPoison)
    return false;

  if (ShouldDropUnnamedAddr)
    for (auto *GVOp : GVOps)
      GVOp->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

  return true;
}

static GlobalVariable *createRelLookupTable(const LookupTableInfo &Info,
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
    if (isa<PoisonValue>(Element)) {
      RelLookupTableContents[Idx++] = PoisonValue::get(Int32Ty);
      continue;
    }

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

static void convertToRelLookupTable(const LookupTableInfo &Info,
                                    ArrayRef<LookupTableUse> Uses,
                                    GlobalVariable &LookupTable) {
  // Generate an array that consists of relative offsets.
  GlobalVariable *RelLookupTable =
      createRelLookupTable(Info, LookupTable);

  Module &M = *LookupTable.getParent();
  for (const LookupTableUse &UseInfo : Uses) {
    BasicBlock *BB = UseInfo.GEP->getParent();
    IRBuilder<> Builder(BB);

    // Place new instruction sequence before GEP.
    Builder.SetInsertPoint(UseInfo.GEP);
    IntegerType *IntTy = cast<IntegerType>(UseInfo.Index->getType());
    Value *Offset = Builder.CreateShl(UseInfo.Index, ConstantInt::get(IntTy, 2),
                                      "reltable.shift");

    // Insert the call to load.relative intrinsic before LOAD.
    // GEP might not be immediately followed by a LOAD, like it can be hoisted
    // outside the loop or another instruction might be inserted them in
    // between.
    Builder.SetInsertPoint(UseInfo.Load);
    Function *LoadRelIntrinsic = llvm::Intrinsic::getOrInsertDeclaration(
        &M, Intrinsic::load_relative, {UseInfo.Index->getType()});

    // Create a call to load.relative intrinsic that computes the target address
    // by adding base address (lookup table address) and relative offset.
    Value *Result = Builder.CreateCall(
        LoadRelIntrinsic, {RelLookupTable, Offset}, "reltable.intrinsic");

    // Replace load instruction with the new generated instruction sequence.
    UseInfo.Load->replaceAllUsesWith(Result);
    // Remove Load and GEP instructions.
    UseInfo.Load->eraseFromParent();
    UseInfo.GEP->eraseFromParent();
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
    SmallVector<LookupTableUse> Uses;
    if (!shouldConvertToRelLookupTable(Info, Uses, M, GV))
      continue;

    convertToRelLookupTable(Info, Uses, GV);

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
