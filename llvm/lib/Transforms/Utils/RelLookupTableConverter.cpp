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
#include "llvm/Analysis/SimplifyQuery.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "rellookuptable"

using namespace llvm;

static bool isValidGEP(const GlobalVariable *GV, const GetElementPtrInst *GEP) {
  if (GEP->getOperand(0) != GV)
    return false;

  if (GEP->getNumOperands() == 3) {
    // Match against a GEP with 3 operands representing
    //
    //   1. The global itself
    //   2. The first index (which should be zero)
    //   3. The actual offset from the start of the global.
    //
    // The GEP should look something like this:
    //
    //   getelementptr [4 x ptr], ptr @glob, i32 0, i32 %idx
    //
    const auto *Idx = dyn_cast<ConstantInt>(GEP->getOperand(1));
    if (!Idx || !Idx->isZero())
      return false;

    if (GV->getValueType() != GEP->getSourceElementType())
      return false;

  } else if (GEP->getNumOperands() == 2) {
    // Match against a GEP with an integral source element type and 2 operands
    // representing:
    //
    //   1. The global itself
    //   2. The first index
    //
    // The GEP should look something like this:
    //
    //   getelementptr i8, ptr @glob, i32 %idx
    //
    // Here we take the byte offset from the start of the global.
    if (!GEP->getSourceElementType()->isIntegerTy())
      return false;

  } else {
    // Don't accept any other GEPs.
    return false;
  }

  // Conservatively only accept GEPs that are used by loads. This is strict,
  // but ensures the global never escapes the module so we can see all uses
  // of it.
  for (const User *U : GEP->users()) {
    auto *Load = dyn_cast<LoadInst>(U);
    if (!Load)
      return false;

    if (!Load->getType()->isPointerTy())
      return false;
  }

  return true;
}

static bool shouldConvertToRelLookupTable(Module &M, GlobalVariable &GV) {
  // The global should look something like this:
  //
  //   @symbols = dso_local constant [3 x ptr] [ptr @.str, ptr @.str.1, ptr @.str.2]
  //
  // This definition must be the one we know will persist at link/runtime.
  if (!GV.hasExactDefinition())
    return false;

  // We must never be able to mutate this global.
  if (!GV.isConstant())
    return false;

  // The global must be local to the TU. We need this because it guarantees this
  // global can't be directly referenced outside the TU. It's important that we
  // see all uses of this global to ensure we can adjust every instances of how
  // it's accessed.
  if (!GV.hasLocalLinkage())
    return false;

  // Definitely don't operate on stuff like llvm.compiler.used.
  if (GV.getName().starts_with("llvm."))
    return false;

  // Operate only on struct or array types.
  //
  // TODO: Tecnically this can also work for a GlobalVariable whose initializer
  // is another GlobalVariable. This would save one reloc for that case.
  const Constant *Initializer = GV.getInitializer();
  if (!Initializer->getType()->isAggregateType())
    return false;

  // Ensure the only user of this global is valid GEPs.
  for (const User *U : GV.users()) {
    const auto *GEP = dyn_cast<GetElementPtrInst>(U);
    if (!GEP)
      return false;

    if (!isValidGEP(&GV, GEP))
      return false;
  }

  // If values are not 64-bit pointers, do not generate a relative lookup table.
  const DataLayout &DL = M.getDataLayout();

  for (const Use &Op : Initializer->operands()) {
    Constant *ConstOp = cast<Constant>(&Op);
    GlobalValue *GVOp;
    APInt Offset;

    // If an operand is not a constant offset from a lookup table,
    // do not generate a relative lookup table.
    if (!IsConstantOffsetFromGlobal(ConstOp, GVOp, Offset, DL))
      return false;

    // If operand is mutable, do not generate a relative lookup table.
    auto *GlovalVarOp = dyn_cast<GlobalVariable>(GVOp);
    if (!GlovalVarOp || !GlovalVarOp->isConstant())
      return false;

    bool DSOLocal = GVOp->isDSOLocal() || GVOp->isImplicitDSOLocal();
    if (!DSOLocal)
      return false;

    Type *ElemType = Op->getType();
    if (!ElemType->isPointerTy() || DL.getPointerTypeSizeInBits(ElemType) != 64)
      return false;
  }

  return true;
}

static GlobalVariable *createRelLookupTable(Module &M, GlobalVariable &GV) {
  const Constant *Initializer = GV.getInitializer();
  size_t NumOperands = Initializer->getNumOperands();
  Type *OffsetTy = Type::getInt32Ty(M.getContext());

  Type *ReplacementTy = [&]() -> Type *{
    if (Initializer->getType()->isStructTy()) {
      SmallVector<Type *, 8> ElemTys(NumOperands, OffsetTy);
      return llvm::StructType::create(ElemTys);
    } else if (Initializer->getType()->isArrayTy()) {
      return llvm::ArrayType::get(OffsetTy, NumOperands);
    }
    llvm_unreachable("An aggregate type should be one of a struct or array");
  }();

  GlobalVariable *Replacement = new GlobalVariable(M, ReplacementTy, /*isConstant=*/true,
      GV.getLinkage(), /*Initializer=*/nullptr);
  Replacement->takeName(&GV); // Take over the old global's name
  Replacement->setUnnamedAddr(GV.getUnnamedAddr());
  Replacement->setVisibility(GV.getVisibility());
  Replacement->setAlignment(llvm::Align(4));  // Unconditional 4-byte alignment

  SmallVector<Constant *, 8> members(NumOperands);
  for (size_t i = 0; i < NumOperands; ++i) {
    Constant *OriginalMember = cast<Constant>(Initializer->getOperand(i));

    // Take the offset.
    Type *IntPtrTy = M.getDataLayout().getIntPtrType(M.getContext());
    Constant *Base = llvm::ConstantExpr::getPtrToInt(Replacement, IntPtrTy);
    Constant *Target =
        llvm::ConstantExpr::getPtrToInt(OriginalMember, IntPtrTy);
    Constant *Sub = llvm::ConstantExpr::getSub(Target, Base);
    Constant *RelOffset = llvm::ConstantExpr::getTrunc(Sub, OffsetTy);

    members[i] = RelOffset;
  }

  Constant *ReplacementInit = [&]() -> Constant * {
    // TODO: Is there any value in keeping this as a struct still since all elements will be the same type?
    if (Initializer->getType()->isStructTy())
      return llvm::ConstantStruct::get(cast<StructType>(ReplacementTy), members);
    else if (Initializer->getType()->isArrayTy())
      return llvm::ConstantArray::get(cast<ArrayType>(ReplacementTy), members);
    llvm_unreachable("An aggregate type should be one of a struct or array");
  }();

  Replacement->setInitializer(ReplacementInit);
  return Replacement;
}

static void convertToRelLookupTable(GlobalVariable &GV) {
  Module &M = *GV.getParent();

  // Generate a global that consists of relative offsets.
  GlobalVariable *Replacement = createRelLookupTable(M, GV);

  // Save these loads and GEPs to erase from their parents after we iterate
  // through the users.
  SmallVector<Instruction *, 16> ToRemove;

  // Rn, we only account for geps, loads, and stores.
  for (User *user : GV.users()) {
    // We assert in an earlier check that all uses of this global must be GEPs.
    auto *GEP = cast<GetElementPtrInst>(user);
    for (User *user : GEP->users()) {
      // We assert in an earlier check that all uses of this GEP must be loads.
      auto *Load = cast<LoadInst>(user);
      assert(GEP->getOperand(0) == &GV &&
             "The first GEP operand should always be the global");

      // Place new instruction sequence before GEP.
      BasicBlock *BB = GEP->getParent();
      IRBuilder<> Builder(BB);
      Builder.SetInsertPoint(GEP);

      // 1. The global itself
      // 2. The first index (which should be zero)
      // 3. The actual offset from the start of the global.
      Value *Offset = [&]() -> Value * {
        if (GEP->getNumOperands() == 3) {
          // Convert to offset in bytes.
          Value *Offset = GEP->getOperand(2);
          return Builder.CreateShl(Offset,
                                   ConstantInt::get(Offset->getType(), 2));
        }
        if (GEP->getNumOperands() == 2) {
          assert(GEP->getSourceElementType()->isIntegerTy() &&
                 "Unhandled source element type");
          ;

          size_t BitWidth =
              cast<IntegerType>(GEP->getSourceElementType())->getBitWidth();
          assert(isPowerOf2_32(BitWidth) && BitWidth >= 8 &&
                 "Expected bitwidth to be multiple of byte size");

          Value *Offset = GEP->getOperand(1);
          if (BitWidth != 8)
            Offset = Builder.CreateShl(
                Offset,
                ConstantInt::get(Offset->getType(), Log2_32(BitWidth / 8)));

          return Offset;
        }
        llvm_unreachable("Unhandled GEP pattern");
      }();

      // Insert the call to load.relative intrinsic before LOAD.
      // GEP might not be immediately followed by a LOAD, like it can be hoisted
      // outside the loop or another instruction might be inserted them in
      // between.
      Builder.SetInsertPoint(Load);
      Function *LoadRelIntrinsic = llvm::Intrinsic::getDeclaration(
          &M, Intrinsic::load_relative, {Offset->getType()});

      // Create a call to load.relative intrinsic that computes the target
      // address by adding base address (lookup table address) and relative
      // offset.
      Value *RelLoad = Builder.CreateCall(LoadRelIntrinsic, {&GV, Offset},
                                          "reltable.intrinsic");

      // Replace load instruction with the new generated instruction sequence.
      Load->replaceAllUsesWith(RelLoad);

      // NOTE: We remove the loads later since we cannot remove them during Use
      // iteration.
      ToRemove.push_back(Load);
    }

    ToRemove.push_back(GEP);
  }

  for (auto *Instr : ToRemove)
    Instr->eraseFromParent();

  // Remove the original lookup table.
  GV.replaceAllUsesWith(Replacement);
  GV.eraseFromParent();
  LLVM_DEBUG(dbgs() << "Converted " << Replacement->getName());
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
    if (!shouldConvertToRelLookupTable(M, GV))
      continue;

    convertToRelLookupTable(GV);

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
