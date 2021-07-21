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
// NOTE: Since this pass will directly pack LDS (assume large LDS) into a struct
// type which would cause allocating huge memory for struct instance within
// every kernel. Hence, before running this pass, it is advisable to run the
// pass "amdgpu-replace-lds-use-with-pointer" which will replace LDS uses within
// non-kernel functions by pointers and thereby minimizes the unnecessary per
// kernel allocation of LDS memory.
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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/OptimizedStructLayout.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <vector>

#define DEBUG_TYPE "amdgpu-lower-module-lds"

using namespace llvm;

static cl::opt<bool> SuperAlignLDSGlobals(
    "amdgpu-super-align-lds-globals",
    cl::desc("Increase alignment of LDS if it is not on align boundary"),
    cl::init(true), cl::Hidden);

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
      // Only lower compute kernels' LDS.
      if (!AMDGPU::isKernel(F.getCallingConv()))
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
        AMDGPU::findVariablesToLower(M, F);

    if (FoundLocalVars.empty()) {
      // No variables to rewrite, no changes made.
      return false;
    }

    // Increase the alignment of LDS globals if necessary to maximise the chance
    // that we can use aligned LDS instructions to access them.
    if (SuperAlignLDSGlobals) {
      for (auto *GV : FoundLocalVars) {
        Align Alignment = AMDGPU::getAlign(DL, GV);
        TypeSize GVSize = DL.getTypeAllocSize(GV->getValueType());

        if (GVSize > 8) {
          // We might want to use a b96 or b128 load/store
          Alignment = std::max(Alignment, Align(16));
        } else if (GVSize > 4) {
          // We might want to use a b64 load/store
          Alignment = std::max(Alignment, Align(8));
        } else if (GVSize > 2) {
          // We might want to use a b32 load/store
          Alignment = std::max(Alignment, Align(4));
        } else if (GVSize > 1) {
          // We might want to use a b16 load/store
          Alignment = std::max(Alignment, Align(2));
        }

        GV->setAlignment(Alignment);
      }
    }

    SmallVector<OptimizedStructLayoutField, 8> LayoutFields;
    LayoutFields.reserve(FoundLocalVars.size());
    for (GlobalVariable *GV : FoundLocalVars) {
      OptimizedStructLayoutField F(GV, DL.getTypeAllocSize(GV->getValueType()),
                                   AMDGPU::getAlign(DL, GV));
      LayoutFields.emplace_back(F);
    }

    performOptimizedStructLayout(LayoutFields);

    std::vector<GlobalVariable *> LocalVars;
    LocalVars.reserve(FoundLocalVars.size()); // will be at least this large
    {
      // This usually won't need to insert any padding, perhaps avoid the alloc
      uint64_t CurrentOffset = 0;
      for (size_t I = 0; I < LayoutFields.size(); I++) {
        GlobalVariable *FGV = static_cast<GlobalVariable *>(
            const_cast<void *>(LayoutFields[I].Id));
        Align DataAlign = LayoutFields[I].Alignment;

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
        CurrentOffset += LayoutFields[I].Size;
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

    Align StructAlign =
        AMDGPU::getAlign(DL, LocalVars[0]);

    GlobalVariable *SGV = new GlobalVariable(
        M, LDSTy, false, GlobalValue::InternalLinkage, UndefValue::get(LDSTy),
        VarName, nullptr, GlobalValue::NotThreadLocal, AMDGPUAS::LOCAL_ADDRESS,
        false);
    SGV->setAlignment(StructAlign);
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
        // Replace all constant uses with instructions if they belong to the
        // current kernel.
        for (User *U : make_early_inc_range(GV->users())) {
          if (ConstantExpr *C = dyn_cast<ConstantExpr>(U))
            AMDGPU::replaceConstantUsesInFunction(C, F);
        }

        GV->removeDeadConstantUsers();

        GV->replaceUsesWithIf(GEP, [F](Use &U) {
          Instruction *I = dyn_cast<Instruction>(U.getUser());
          return I && I->getFunction() == F;
        });
      } else {
        GV->replaceAllUsesWith(GEP);
      }
      if (GV->use_empty()) {
        UsedList.erase(GV);
        GV->eraseFromParent();
      }

      uint64_t Off = DL.getStructLayout(LDSTy)->getElementOffset(I);
      Align A = commonAlignment(StructAlign, Off);
      refineUsesAlignment(GEP, A, DL);
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

  void refineUsesAlignment(Value *Ptr, Align A, const DataLayout &DL,
                           unsigned MaxDepth = 5) {
    if (!MaxDepth || A == 1)
      return;

    for (User *U : Ptr->users()) {
      if (auto *LI = dyn_cast<LoadInst>(U)) {
        LI->setAlignment(std::max(A, LI->getAlign()));
        continue;
      }
      if (auto *SI = dyn_cast<StoreInst>(U)) {
        if (SI->getPointerOperand() == Ptr)
          SI->setAlignment(std::max(A, SI->getAlign()));
        continue;
      }
      if (auto *AI = dyn_cast<AtomicRMWInst>(U)) {
        // None of atomicrmw operations can work on pointers, but let's
        // check it anyway in case it will or we will process ConstantExpr.
        if (AI->getPointerOperand() == Ptr)
          AI->setAlignment(std::max(A, AI->getAlign()));
        continue;
      }
      if (auto *AI = dyn_cast<AtomicCmpXchgInst>(U)) {
        if (AI->getPointerOperand() == Ptr)
          AI->setAlignment(std::max(A, AI->getAlign()));
        continue;
      }
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        unsigned BitWidth = DL.getIndexTypeSizeInBits(GEP->getType());
        APInt Off(BitWidth, 0);
        if (GEP->getPointerOperand() == Ptr &&
            GEP->accumulateConstantOffset(DL, Off)) {
          Align GA = commonAlignment(A, Off.getLimitedValue());
          refineUsesAlignment(GEP, GA, DL, MaxDepth - 1);
        }
        continue;
      }
      if (auto *I = dyn_cast<Instruction>(U)) {
        if (I->getOpcode() == Instruction::BitCast ||
            I->getOpcode() == Instruction::AddrSpaceCast)
          refineUsesAlignment(I, A, DL, MaxDepth - 1);
      }
    }
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
