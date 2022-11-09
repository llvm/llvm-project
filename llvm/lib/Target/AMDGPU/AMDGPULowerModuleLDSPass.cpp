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
// through unchanged for simplification or error diagnostics in later passes.
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
#include "Utils/AMDGPUMemoryUtils.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/OptimizedStructLayout.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <tuple>
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
    // The verifier rejects used lists containing an inttoptr of a constant
    // so remove the variables from these lists before replaceAllUsesWith

    SmallPtrSet<Constant *, 32> LocalVarsSet;
    for (GlobalVariable *LocalVar : LocalVars)
      if (Constant *C = dyn_cast<Constant>(LocalVar->stripPointerCasts()))
        LocalVarsSet.insert(C);
    removeFromUsedList(M, "llvm.used", LocalVarsSet);
    removeFromUsedList(M, "llvm.compiler.used", LocalVarsSet);
  }

  static void markUsedByKernel(IRBuilder<> &Builder, Function *Func,
                               GlobalVariable *SGV) {
    // The llvm.amdgcn.module.lds instance is implicitly used by all kernels
    // that might call a function which accesses a field within it. This is
    // presently approximated to 'all kernels' if there are any such functions
    // in the module. This implicit use is redefined as an explicit use here so
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

  static bool eliminateConstantExprUsesOfLDSFromAllInstructions(Module &M) {
    // Constants are uniqued within LLVM. A ConstantExpr referring to a LDS
    // global may have uses from multiple different functions as a result.
    // This pass specialises LDS variables with respect to the kernel that
    // allocates them.

    // This is semantically equivalent to:
    // for (auto &F : M.functions())
    //   for (auto &BB : F)
    //     for (auto &I : BB)
    //       for (Use &Op : I.operands())
    //         if (constantExprUsesLDS(Op))
    //           replaceConstantExprInFunction(I, Op);

    bool Changed = false;

    // Find all ConstantExpr that are direct users of an LDS global
    SmallVector<ConstantExpr *> Stack;
    for (auto &GV : M.globals())
      if (AMDGPU::isLDSVariableToLower(GV))
        for (User *U : GV.users())
          if (ConstantExpr *C = dyn_cast<ConstantExpr>(U))
            Stack.push_back(C);

    // Expand to include constexpr users of direct users
    SetVector<ConstantExpr *> ConstExprUsersOfLDS;
    while (!Stack.empty()) {
      ConstantExpr *V = Stack.pop_back_val();
      if (ConstExprUsersOfLDS.contains(V))
        continue;

      ConstExprUsersOfLDS.insert(V);

      for (auto *Nested : V->users())
        if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Nested))
          Stack.push_back(CE);
    }

    // Find all instructions that use any of the ConstExpr users of LDS
    SetVector<Instruction *> InstructionWorklist;
    for (ConstantExpr *CE : ConstExprUsersOfLDS)
      for (User *U : CE->users())
        if (auto *I = dyn_cast<Instruction>(U))
          InstructionWorklist.insert(I);

    // Replace those ConstExpr operands with instructions
    while (!InstructionWorklist.empty()) {
      Instruction *I = InstructionWorklist.pop_back_val();
      for (Use &U : I->operands()) {

        auto *BI = I;
        if (auto *Phi = dyn_cast<PHINode>(I)) {
          BasicBlock *BB = Phi->getIncomingBlock(U);
          BasicBlock::iterator It = BB->getFirstInsertionPt();
          assert(It != BB->end() && "Unexpected empty basic block");
          BI = &(*(It));
        }

        if (ConstantExpr *C = dyn_cast<ConstantExpr>(U.get())) {
          if (ConstExprUsersOfLDS.contains(C)) {
            Changed = true;
            Instruction *NI = C->getAsInstruction(BI);
            InstructionWorklist.insert(NI);
            U.set(NI);
            C->removeDeadConstantUsers();
          }
        }
      }
    }

    return Changed;
  }

public:
  static char ID;

  AMDGPULowerModuleLDS() : ModulePass(ID) {
    initializeAMDGPULowerModuleLDSPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    LLVMContext &Ctx = M.getContext();
    CallGraph CG = CallGraph(M);
    bool Changed = superAlignLDSGlobals(M);

    Changed |= eliminateConstantExprUsesOfLDSFromAllInstructions(M);

    // Move variables used by functions into amdgcn.module.lds
    std::vector<GlobalVariable *> ModuleScopeVariables =
        AMDGPU::findLDSVariablesToLower(M, nullptr);
    if (!ModuleScopeVariables.empty()) {
      std::string VarName = "llvm.amdgcn.module.lds";

      GlobalVariable *SGV;
      DenseMap<GlobalVariable *, Constant *> LDSVarsToConstantGEP;
      std::tie(SGV, LDSVarsToConstantGEP) =
          createLDSVariableReplacement(M, VarName, ModuleScopeVariables);

      appendToCompilerUsed(
          M, {static_cast<GlobalValue *>(
                 ConstantExpr::getPointerBitCastOrAddrSpaceCast(
                     cast<Constant>(SGV), Type::getInt8PtrTy(Ctx)))});

      removeFromUsedLists(M, ModuleScopeVariables);
      replaceLDSVariablesWithStruct(M, ModuleScopeVariables, SGV,
                                    LDSVarsToConstantGEP,
                                    [](Use &) { return true; });

      // This ensures the variable is allocated when called functions access it.
      // It also lets other passes, specifically PromoteAlloca, accurately
      // calculate how much LDS will be used by the kernel after lowering.

      IRBuilder<> Builder(Ctx);
      for (Function &Func : M.functions()) {
        if (!Func.isDeclaration() && AMDGPU::isKernel(Func.getCallingConv())) {
          const CallGraphNode *N = CG[&Func];
          const bool CalleesRequireModuleLDS = N->size() > 0;

          if (CalleesRequireModuleLDS) {
            // If a function this kernel might call requires module LDS,
            // annotate the kernel to let later passes know it will allocate
            // this structure, even if not apparent from the IR.
            markUsedByKernel(Builder, &Func, SGV);
          } else {
            // However if we are certain this kernel cannot call a function that
            // requires module LDS, annotate the kernel so the backend can elide
            // the allocation without repeating callgraph walks.
            Func.addFnAttr("amdgpu-elide-module-lds");
          }
        }
      }

      Changed = true;
    }

    // Move variables used by kernels into per-kernel instances
    for (Function &F : M.functions()) {
      if (F.isDeclaration())
        continue;

      // Only lower compute kernels' LDS.
      if (!AMDGPU::isKernel(F.getCallingConv()))
        continue;

      std::vector<GlobalVariable *> KernelUsedVariables =
          AMDGPU::findLDSVariablesToLower(M, &F);

      if (!KernelUsedVariables.empty()) {
        // The association between kernel function and LDS struct is done by
        // symbol name, which only works if the function in question has a name
        // This is not expected to be a problem in practice as kernels are
        // called by name making anonymous ones (which are named by the backend)
        // difficult to use. This does mean that llvm test cases need
        // to name the kernels.
        if (!F.hasName()) {
          report_fatal_error("Anonymous kernels cannot use LDS variables");
        }

        std::string VarName =
            (Twine("llvm.amdgcn.kernel.") + F.getName() + ".lds").str();
        GlobalVariable *SGV;
        DenseMap<GlobalVariable *, Constant *> LDSVarsToConstantGEP;
        std::tie(SGV, LDSVarsToConstantGEP) =
            createLDSVariableReplacement(M, VarName, KernelUsedVariables);

        removeFromUsedLists(M, KernelUsedVariables);
        replaceLDSVariablesWithStruct(
            M, KernelUsedVariables, SGV, LDSVarsToConstantGEP, [&F](Use &U) {
              Instruction *I = dyn_cast<Instruction>(U.getUser());
              return I && I->getFunction() == &F;
            });
        Changed = true;
      }
    }

    for (auto &GV : make_early_inc_range(M.globals()))
      if (AMDGPU::isLDSVariableToLower(GV)) {
        GV.removeDeadConstantUsers();
        if (GV.use_empty())
          GV.eraseFromParent();
      }

    return Changed;
  }

private:
  // Increase the alignment of LDS globals if necessary to maximise the chance
  // that we can use aligned LDS instructions to access them.
  static bool superAlignLDSGlobals(Module &M) {
    const DataLayout &DL = M.getDataLayout();
    bool Changed = false;
    if (!SuperAlignLDSGlobals) {
      return Changed;
    }

    for (auto &GV : M.globals()) {
      if (GV.getType()->getPointerAddressSpace() != AMDGPUAS::LOCAL_ADDRESS) {
        // Only changing alignment of LDS variables
        continue;
      }
      if (!GV.hasInitializer()) {
        // cuda/hip extern __shared__ variable, leave alignment alone
        continue;
      }

      Align Alignment = AMDGPU::getAlign(DL, &GV);
      TypeSize GVSize = DL.getTypeAllocSize(GV.getValueType());

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

      if (Alignment != AMDGPU::getAlign(DL, &GV)) {
        Changed = true;
        GV.setAlignment(Alignment);
      }
    }
    return Changed;
  }

  std::tuple<GlobalVariable *, DenseMap<GlobalVariable *, Constant *>>
  createLDSVariableReplacement(
      Module &M, std::string VarName,
      std::vector<GlobalVariable *> const &LDSVarsToTransform) {
    // Create a struct instance containing LDSVarsToTransform and map from those
    // variables to ConstantExprGEP
    // Variables may be introduced to meet alignment requirements. No aliasing
    // metadata is useful for these as they have no uses. Erased before return.

    LLVMContext &Ctx = M.getContext();
    const DataLayout &DL = M.getDataLayout();
    assert(!LDSVarsToTransform.empty());

    SmallVector<OptimizedStructLayoutField, 8> LayoutFields;
    LayoutFields.reserve(LDSVarsToTransform.size());
    {
      // The order of fields in this struct depends on the order of
      // varables in the argument which varies when changing how they
      // are identified, leading to spurious test breakage.
      std::vector<GlobalVariable *> Sorted(LDSVarsToTransform.begin(),
                                           LDSVarsToTransform.end());
      llvm::sort(Sorted.begin(), Sorted.end(),
                 [](const GlobalVariable *lhs, const GlobalVariable *rhs) {
                   return lhs->getName() < rhs->getName();
                 });
      for (GlobalVariable *GV : Sorted) {
        OptimizedStructLayoutField F(GV,
                                     DL.getTypeAllocSize(GV->getValueType()),
                                     AMDGPU::getAlign(DL, GV));
        LayoutFields.emplace_back(F);
      }
    }

    performOptimizedStructLayout(LayoutFields);

    std::vector<GlobalVariable *> LocalVars;
    BitVector IsPaddingField;
    LocalVars.reserve(LDSVarsToTransform.size()); // will be at least this large
    IsPaddingField.reserve(LDSVarsToTransform.size());
    {
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
          IsPaddingField.push_back(true);
          CurrentOffset += Padding;
        }

        LocalVars.push_back(FGV);
        IsPaddingField.push_back(false);
        CurrentOffset += LayoutFields[I].Size;
      }
    }

    std::vector<Type *> LocalVarTypes;
    LocalVarTypes.reserve(LocalVars.size());
    std::transform(
        LocalVars.cbegin(), LocalVars.cend(), std::back_inserter(LocalVarTypes),
        [](const GlobalVariable *V) -> Type * { return V->getValueType(); });

    StructType *LDSTy = StructType::create(Ctx, LocalVarTypes, VarName + ".t");

    Align StructAlign = AMDGPU::getAlign(DL, LocalVars[0]);

    GlobalVariable *SGV = new GlobalVariable(
        M, LDSTy, false, GlobalValue::InternalLinkage, UndefValue::get(LDSTy),
        VarName, nullptr, GlobalValue::NotThreadLocal, AMDGPUAS::LOCAL_ADDRESS,
        false);
    SGV->setAlignment(StructAlign);

    DenseMap<GlobalVariable *, Constant *> Map;
    Type *I32 = Type::getInt32Ty(Ctx);
    for (size_t I = 0; I < LocalVars.size(); I++) {
      GlobalVariable *GV = LocalVars[I];
      Constant *GEPIdx[] = {ConstantInt::get(I32, 0), ConstantInt::get(I32, I)};
      Constant *GEP = ConstantExpr::getGetElementPtr(LDSTy, SGV, GEPIdx, true);
      if (IsPaddingField[I]) {
        assert(GV->use_empty());
        GV->eraseFromParent();
      } else {
        Map[GV] = GEP;
      }
    }
    assert(Map.size() == LDSVarsToTransform.size());
    return std::make_tuple(SGV, std::move(Map));
  }

  template <typename PredicateTy>
  void replaceLDSVariablesWithStruct(
      Module &M, std::vector<GlobalVariable *> const &LDSVarsToTransform,
      GlobalVariable *SGV,
      DenseMap<GlobalVariable *, Constant *> &LDSVarsToConstantGEP,
      PredicateTy Predicate) {
    LLVMContext &Ctx = M.getContext();
    const DataLayout &DL = M.getDataLayout();

    // Create alias.scope and their lists. Each field in the new structure
    // does not alias with all other fields.
    SmallVector<MDNode *> AliasScopes;
    SmallVector<Metadata *> NoAliasList;
    const size_t NumberVars = LDSVarsToTransform.size();
    if (NumberVars > 1) {
      MDBuilder MDB(Ctx);
      AliasScopes.reserve(NumberVars);
      MDNode *Domain = MDB.createAnonymousAliasScopeDomain();
      for (size_t I = 0; I < NumberVars; I++) {
        MDNode *Scope = MDB.createAnonymousAliasScope(Domain);
        AliasScopes.push_back(Scope);
      }
      NoAliasList.append(&AliasScopes[1], AliasScopes.end());
    }

    // Replace uses of ith variable with a constantexpr to the corresponding
    // field of the instance that will be allocated by AMDGPUMachineFunction
    for (size_t I = 0; I < NumberVars; I++) {
      GlobalVariable *GV = LDSVarsToTransform[I];
      Constant *GEP = LDSVarsToConstantGEP[GV];

      GV->replaceUsesWithIf(GEP, Predicate);
      if (GV->use_empty()) {
        GV->eraseFromParent();
      }

      APInt APOff(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
      GEP->stripAndAccumulateInBoundsConstantOffsets(DL, APOff);
      uint64_t Offset = APOff.getZExtValue();

      Align A = commonAlignment(SGV->getAlign().valueOrOne(), Offset);

      if (I)
        NoAliasList[I - 1] = AliasScopes[I - 1];
      MDNode *NoAlias =
          NoAliasList.empty() ? nullptr : MDNode::get(Ctx, NoAliasList);
      MDNode *AliasScope =
          AliasScopes.empty() ? nullptr : MDNode::get(Ctx, {AliasScopes[I]});

      refineUsesAlignmentAndAA(GEP, A, DL, AliasScope, NoAlias);
    }
  }

  void refineUsesAlignmentAndAA(Value *Ptr, Align A, const DataLayout &DL,
                                MDNode *AliasScope, MDNode *NoAlias,
                                unsigned MaxDepth = 5) {
    if (!MaxDepth || (A == 1 && !AliasScope))
      return;

    for (User *U : Ptr->users()) {
      if (auto *I = dyn_cast<Instruction>(U)) {
        if (AliasScope && I->mayReadOrWriteMemory()) {
          MDNode *AS = I->getMetadata(LLVMContext::MD_alias_scope);
          AS = (AS ? MDNode::getMostGenericAliasScope(AS, AliasScope)
                   : AliasScope);
          I->setMetadata(LLVMContext::MD_alias_scope, AS);

          MDNode *NA = I->getMetadata(LLVMContext::MD_noalias);
          NA = (NA ? MDNode::intersect(NA, NoAlias) : NoAlias);
          I->setMetadata(LLVMContext::MD_noalias, NA);
        }
      }

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
        if (GEP->getPointerOperand() == Ptr) {
          Align GA;
          if (GEP->accumulateConstantOffset(DL, Off))
            GA = commonAlignment(A, Off.getLimitedValue());
          refineUsesAlignmentAndAA(GEP, GA, DL, AliasScope, NoAlias,
                                   MaxDepth - 1);
        }
        continue;
      }
      if (auto *I = dyn_cast<Instruction>(U)) {
        if (I->getOpcode() == Instruction::BitCast ||
            I->getOpcode() == Instruction::AddrSpaceCast)
          refineUsesAlignmentAndAA(I, A, DL, AliasScope, NoAlias, MaxDepth - 1);
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
