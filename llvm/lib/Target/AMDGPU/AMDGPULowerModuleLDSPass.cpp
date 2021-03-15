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

  static bool isKernelCC(Function *Func) {
    return AMDGPU::isModuleEntryFunctionCC(Func->getCallingConv());
  }

  static Align getAlign(DataLayout const &DL, const GlobalVariable *GV) {
    return DL.getValueOrABITypeAlignment(GV->getPointerAlignment(DL),
                                         GV->getValueType());
  }

  static bool
  userRequiresLowering(const SmallPtrSetImpl<GlobalValue *> &UsedList,
                       User *InitialUser) {
    // Any LDS variable can be lowered by moving into the created struct
    // Each variable so lowered is allocated in every kernel, so variables
    // whose users are all known to be safe to lower without the transform
    // are left unchanged.
    SmallPtrSet<User *, 8> Visited;
    SmallVector<User *, 16> Stack;
    Stack.push_back(InitialUser);

    while (!Stack.empty()) {
      User *V = Stack.pop_back_val();
      Visited.insert(V);

      if (auto *G = dyn_cast<GlobalValue>(V->stripPointerCasts())) {
        if (UsedList.contains(G)) {
          continue;
        }
      }

      if (auto *I = dyn_cast<Instruction>(V)) {
        if (isKernelCC(I->getFunction())) {
          continue;
        }
      }

      if (auto *E = dyn_cast<ConstantExpr>(V)) {
        for (Value::user_iterator EU = E->user_begin(); EU != E->user_end();
             ++EU) {
          if (Visited.insert(*EU).second) {
            Stack.push_back(*EU);
          }
        }
        continue;
      }

      // Unknown user, conservatively lower the variable
      return true;
    }

    return false;
  }

  static std::vector<GlobalVariable *>
  findVariablesToLower(Module &M,
                       const SmallPtrSetImpl<GlobalValue *> &UsedList) {
    std::vector<llvm::GlobalVariable *> LocalVars;
    for (auto &GV : M.globals()) {
      if (GV.getType()->getPointerAddressSpace() != AMDGPUAS::LOCAL_ADDRESS) {
        continue;
      }
      if (!GV.hasInitializer()) {
        // addrspace(3) without initializer implies cuda/hip extern __shared__
        // the semantics for such a variable appears to be that all extern
        // __shared__ variables alias one another, in which case this transform
        // is not required
        continue;
      }
      if (!isa<UndefValue>(GV.getInitializer())) {
        // Initializers are unimplemented for local address space.
        // Leave such variables in place for consistent error reporting.
        continue;
      }
      if (GV.isConstant()) {
        // A constant undef variable can't be written to, and any load is
        // undef, so it should be eliminated by the optimizer. It could be
        // dropped by the back end if not. This pass skips over it.
        continue;
      }
      if (std::none_of(GV.user_begin(), GV.user_end(), [&](User *U) {
            return userRequiresLowering(UsedList, U);
          })) {
        continue;
      }
      LocalVars.push_back(&GV);
    }
    return LocalVars;
  }

  static void removeFromUsedList(Module &M, StringRef Name,
                                 SmallPtrSetImpl<Constant *> &ToRemove) {
    GlobalVariable *GV = M.getGlobalVariable(Name);
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

  static SmallPtrSet<GlobalValue *, 32> getUsedList(Module &M) {
    SmallPtrSet<GlobalValue *, 32> UsedList;

    SmallVector<GlobalValue *, 32> TmpVec;
    collectUsedGlobalVariables(M, TmpVec, true);
    UsedList.insert(TmpVec.begin(), TmpVec.end());

    TmpVec.clear();
    collectUsedGlobalVariables(M, TmpVec, false);
    UsedList.insert(TmpVec.begin(), TmpVec.end());

    return UsedList;
  }

public:
  static char ID;

  AMDGPULowerModuleLDS() : ModulePass(ID) {
    initializeAMDGPULowerModuleLDSPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    LLVMContext &Ctx = M.getContext();
    const DataLayout &DL = M.getDataLayout();
    SmallPtrSet<GlobalValue *, 32> UsedList = getUsedList(M);

    // Find variables to move into new struct instance
    std::vector<GlobalVariable *> FoundLocalVars =
        findVariablesToLower(M, UsedList);

    if (FoundLocalVars.empty()) {
      // No variables to rewrite, no changes made.
      return false;
    }

    // Sort by alignment, descending, to minimise padding.
    // On ties, sort by size, descending, then by name, lexicographical.
    llvm::stable_sort(
        FoundLocalVars,
        [&](const GlobalVariable *LHS, const GlobalVariable *RHS) -> bool {
          Align ALHS = getAlign(DL, LHS);
          Align ARHS = getAlign(DL, RHS);
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
        Align DataAlign = getAlign(DL, FGV);

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

    StructType *LDSTy = StructType::create(
        Ctx, LocalVarTypes, llvm::StringRef("llvm.amdgcn.module.lds.t"));

    Align MaxAlign = getAlign(DL, LocalVars[0]); // was sorted on alignment
    Constant *InstanceAddress = Constant::getIntegerValue(
        PointerType::get(LDSTy, AMDGPUAS::LOCAL_ADDRESS), APInt(32, 0));

    GlobalVariable *SGV = new GlobalVariable(
        M, LDSTy, false, GlobalValue::InternalLinkage, UndefValue::get(LDSTy),
        "llvm.amdgcn.module.lds", nullptr, GlobalValue::NotThreadLocal,
        AMDGPUAS::LOCAL_ADDRESS, false);
    SGV->setAlignment(MaxAlign);
    appendToCompilerUsed(
        M, {static_cast<GlobalValue *>(
               ConstantExpr::getPointerBitCastOrAddrSpaceCast(
                   cast<Constant>(SGV), Type::getInt8PtrTy(Ctx)))});

    // The verifier rejects used lists containing an inttoptr of a constant
    // so remove the variables from these lists before replaceAllUsesWith
    removeFromUsedLists(M, LocalVars);

    // Replace uses of ith variable with a constantexpr to the ith field of the
    // instance that will be allocated by AMDGPUMachineFunction
    Type *I32 = Type::getInt32Ty(Ctx);
    for (size_t I = 0; I < LocalVars.size(); I++) {
      GlobalVariable *GV = LocalVars[I];
      Constant *GEPIdx[] = {ConstantInt::get(I32, 0), ConstantInt::get(I32, I)};
      GV->replaceAllUsesWith(
          ConstantExpr::getGetElementPtr(LDSTy, InstanceAddress, GEPIdx));
      GV->eraseFromParent();
    }

    // Mark kernels with asm that reads the address of the allocated structure
    // This is not necessary for lowering. This lets other passes, specifically
    // PromoteAlloca, accurately calculate how much LDS will be used by the
    // kernel after lowering.
    {
      IRBuilder<> Builder(Ctx);
      SmallPtrSet<Function *, 32> Kernels;
      for (auto &I : M.functions()) {
        Function *Func = &I;
        if (isKernelCC(Func) && !Kernels.contains(Func)) {
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
