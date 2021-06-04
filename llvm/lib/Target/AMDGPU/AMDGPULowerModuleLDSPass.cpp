//===-- AMDGPULowerModuleLDSPass.cpp ------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates LDS uses from non-kernel functions.
//
// The strategy is to create a new struct with a field for each LDS variable
// and allocate that struct at the same address for every kernel. Uses of the
// original LDS variables are then replaced with compile time offsets from that
// known address. AMDGPUMachineFunction allocates the LDS global.
//
// Local variables with constant annotation or non-undef initializer are passed
// through unchanged for simplication or error diagnostics in later passes.
//
// To reduce the memory overhead variables that are only used by kernels are
// excluded from this transform. The analysis to determine whether a variable
// is only used by a kernel is cheap and conservative so this may allocate
// a variable in every kernel when it was not strictly necessary to do so.
//
// A possible future refinement is to specialise the structure per-kernel, so
// that fields can be elided based on more expensive analysis.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "Utils/AMDGPULDSUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <algorithm>
#include <vector>

#define DEBUG_TYPE "amdgpu-lower-module-lds"

using namespace llvm;

namespace {

class AMDGPULowerModuleLDS : public ModulePass {

  static void removeFromUsedList(Module &M, StringRef Name,
                                 SmallPtrSetImpl<Constant *> &ToRemove) {
    GlobalVariable *GV = M.getNamedGlobal(Name);
    if (!GV || ToRemove.empty()) {
      return;
    }

    SmallVector<Constant *, 16> Init;
    auto *CA = cast<ConstantArray>(GV->getInitializer());
    for (auto &Op : CA->operands()) {
      // ModuleUtils::appendToUsed only inserts Constants
      Constant *C = cast<Constant>(Op);
      if (!ToRemove.contains(C->stripPointerCasts())) {
        Init.push_back(C);
      }
    }

    if (Init.size() == CA->getNumOperands()) {
      return; // none to remove
    }

    GV->eraseFromParent();

    for (Constant *C : ToRemove) {
      C->removeDeadConstantUsers();
    }

    if (!Init.empty()) {
      ArrayType *ATy =
          ArrayType::get(Type::getInt8PtrTy(M.getContext()), Init.size());
      GV =
          new llvm::GlobalVariable(M, ATy, false, GlobalValue::AppendingLinkage,
                                   ConstantArray::get(ATy, Init), Name);
      GV->setSection("llvm.metadata");
    }
  }

  static void
  removeFromUsedLists(Module &M,
                      const std::vector<GlobalVariable *> &LocalVars) {
    SmallPtrSet<Constant *, 32> LocalVarsSet;
    for (size_t I = 0; I < LocalVars.size(); I++) {
      if (Constant *C = dyn_cast<Constant>(LocalVars[I]->stripPointerCasts())) {
        LocalVarsSet.insert(C);
      }
    }
    removeFromUsedList(M, "llvm.used", LocalVarsSet);
    removeFromUsedList(M, "llvm.compiler.used", LocalVarsSet);
  }

  static void markUsedByKernel(IRBuilder<> &Builder, Function *Func,
                               GlobalVariable *SGV) {
    // The llvm.amdgcn.module.lds instance is implicitly used by all kernels
    // that might call a function which accesses a field within it. This is
    // presently approximated to 'all kernels' if there are any such functions
    // in the module. This implicit use is reified as an explicit use here so
    // that later passes, specifically PromoteAlloca, account for the required
    // memory without any knowledge of this transform.

    // An operand bundle on llvm.donothing works because the call instruction
    // survives until after the last pass that needs to account for LDS. It is
    // better than inline asm as the latter survives until the end of codegen. A
    // totally robust solution would be a function with the same semantics as
    // llvm.donothing that takes a pointer to the instance and is lowered to a
    // no-op after LDS is allocated, but that is not presently necessary.

    LLVMContext &Ctx = Func->getContext();

    Builder.SetInsertPoint(Func->getEntryBlock().getFirstNonPHI());

    FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx), {});

    Function *Decl =
        Intrinsic::getDeclaration(Func->getParent(), Intrinsic::donothing, {});

    Value *UseInstance[1] = {Builder.CreateInBoundsGEP(
        SGV->getValueType(), SGV, ConstantInt::get(Type::getInt32Ty(Ctx), 0))};

    Builder.CreateCall(FTy, Decl, {},
                       {OperandBundleDefT<Value *>("ExplicitUse", UseInstance)},
                       "");
  }

private:
  SmallPtrSet<GlobalValue *, 32> UsedList;

public:
  static char ID;

  AMDGPULowerModuleLDS() : ModulePass(ID) {
    initializeAMDGPULowerModuleLDSPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    UsedList = AMDGPU::getUsedList(M);

    bool Changed = processUsedLDS(M);

    for (Function &F : M.functions()) {
      if (!AMDGPU::isKernelCC(&F))
        continue;
      Changed |= processUsedLDS(M, &F);
    }

    UsedList.clear();
    return Changed;
  }

private:
  bool processUsedLDS(Module &M, Function *F = nullptr) {
    LLVMContext &Ctx = M.getContext();
    const DataLayout &DL = M.getDataLayout();

    // Find variables to move into new struct instance
    std::vector<GlobalVariable *> FoundLocalVars =
        AMDGPU::findVariablesToLower(M, UsedList, F);

    if (FoundLocalVars.empty()) {
      // No variables to rewrite, no changes made.
      return false;
    }

    // Sort by alignment, descending, to minimise padding.
    // On ties, sort by size, descending, then by name, lexicographical.
    llvm::stable_sort(
        FoundLocalVars,
        [&](const GlobalVariable *LHS, const GlobalVariable *RHS) -> bool {
          Align ALHS = AMDGPU::getAlign(DL, LHS);
          Align ARHS = AMDGPU::getAlign(DL, RHS);
          if (ALHS != ARHS) {
            return ALHS > ARHS;
          }

          TypeSize SLHS = DL.getTypeAllocSize(LHS->getValueType());
          TypeSize SRHS = DL.getTypeAllocSize(RHS->getValueType());
          if (SLHS != SRHS) {
            return SLHS > SRHS;
          }

          // By variable name on tie for predictable order in test cases.
          return LHS->getName() < RHS->getName();
        });

    std::vector<GlobalVariable *> LocalVars;
    LocalVars.reserve(FoundLocalVars.size()); // will be at least this large
    {
      // This usually won't need to insert any padding, perhaps avoid the alloc
      uint64_t CurrentOffset = 0;
      for (size_t I = 0; I < FoundLocalVars.size(); I++) {
        GlobalVariable *FGV = FoundLocalVars[I];
        Align DataAlign = AMDGPU::getAlign(DL, FGV);

        uint64_t DataAlignV = DataAlign.value();
        if (uint64_t Rem = CurrentOffset % DataAlignV) {
          uint64_t Padding = DataAlignV - Rem;

          // Append an array of padding bytes to meet alignment requested
          // Note (o +      (a - (o % a)) ) % a == 0
          //      (offset + Padding       ) % align == 0

          Type *ATy = ArrayType::get(Type::getInt8Ty(Ctx), Padding);
          LocalVars.push_back(new GlobalVariable(
              M, ATy, false, GlobalValue::InternalLinkage, UndefValue::get(ATy),
              "", nullptr, GlobalValue::NotThreadLocal, AMDGPUAS::LOCAL_ADDRESS,
              false));
          CurrentOffset += Padding;
        }

        LocalVars.push_back(FGV);
        CurrentOffset += DL.getTypeAllocSize(FGV->getValueType());
      }
    }

    std::vector<Type *> LocalVarTypes;
    LocalVarTypes.reserve(LocalVars.size());
    std::transform(
        LocalVars.cbegin(), LocalVars.cend(), std::back_inserter(LocalVarTypes),
        [](const GlobalVariable *V) -> Type * { return V->getValueType(); });

    std::string VarName(
        F ? (Twine("llvm.amdgcn.kernel.") + F->getName() + ".lds").str()
          : "llvm.amdgcn.module.lds");
    StructType *LDSTy = StructType::create(Ctx, LocalVarTypes, VarName + ".t");

    Align MaxAlign =
        AMDGPU::getAlign(DL, LocalVars[0]); // was sorted on alignment

    GlobalVariable *SGV = new GlobalVariable(
        M, LDSTy, false, GlobalValue::InternalLinkage, UndefValue::get(LDSTy),
        VarName, nullptr, GlobalValue::NotThreadLocal, AMDGPUAS::LOCAL_ADDRESS,
        false);
    SGV->setAlignment(MaxAlign);
    if (!F) {
      appendToCompilerUsed(
          M, {static_cast<GlobalValue *>(
                 ConstantExpr::getPointerBitCastOrAddrSpaceCast(
                     cast<Constant>(SGV), Type::getInt8PtrTy(Ctx)))});
    }

    // The verifier rejects used lists containing an inttoptr of a constant
    // so remove the variables from these lists before replaceAllUsesWith
    removeFromUsedLists(M, LocalVars);

    // Replace uses of ith variable with a constantexpr to the ith field of the
    // instance that will be allocated by AMDGPUMachineFunction
    Type *I32 = Type::getInt32Ty(Ctx);
    for (size_t I = 0; I < LocalVars.size(); I++) {
      GlobalVariable *GV = LocalVars[I];
      Constant *GEPIdx[] = {ConstantInt::get(I32, 0), ConstantInt::get(I32, I)};
      Constant *GEP = ConstantExpr::getGetElementPtr(LDSTy, SGV, GEPIdx);
      if (F) {
        GV->replaceUsesWithIf(GEP, [F](Use &U) {
          return AMDGPU::isUsedOnlyFromFunction(U.getUser(), F);
        });
      } else {
        GV->replaceAllUsesWith(GEP);
      }
      if (GV->use_empty()) {
        UsedList.erase(GV);
        GV->eraseFromParent();
      }
    }

    // Mark kernels with asm that reads the address of the allocated structure
    // This is not necessary for lowering. This lets other passes, specifically
    // PromoteAlloca, accurately calculate how much LDS will be used by the
    // kernel after lowering.
    if (!F) {
      IRBuilder<> Builder(Ctx);
      SmallPtrSet<Function *, 32> Kernels;
      for (auto &I : M.functions()) {
        Function *Func = &I;
        if (AMDGPU::isKernelCC(Func) && !Kernels.contains(Func)) {
          markUsedByKernel(Builder, Func, SGV);
          Kernels.insert(Func);
        }
      }
    }
    return true;
  }
};

} // namespace
char AMDGPULowerModuleLDS::ID = 0;

char &llvm::AMDGPULowerModuleLDSID = AMDGPULowerModuleLDS::ID;

INITIALIZE_PASS(AMDGPULowerModuleLDS, DEBUG_TYPE,
                "Lower uses of LDS variables from non-kernel functions", false,
                false)

ModulePass *llvm::createAMDGPULowerModuleLDSPass() {
  return new AMDGPULowerModuleLDS();
}

PreservedAnalyses AMDGPULowerModuleLDSPass::run(Module &M,
                                                ModuleAnalysisManager &) {
  return AMDGPULowerModuleLDS().runOnModule(M) ? PreservedAnalyses::none()
                                               : PreservedAnalyses::all();
}
