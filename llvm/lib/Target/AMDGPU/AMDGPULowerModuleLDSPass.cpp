//===-- AMDGPULowerModuleLDSPass.cpp ------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates local data store, LDS, uses from non-kernel functions.
// LDS is contiguous memory allocated per kernel execution.
//
// Background.
//
// The programming model is global variables, or equivalently function local
// static variables, accessible from kernels or other functions. For uses from
// kernels this is straightforward - assign an integer to the kernel for the
// memory required by all the variables combined, allocate them within that.
// For uses from functions there are performance tradeoffs to choose between.
//
// This model means the GPU runtime can specify the amount of memory allocated.
// If this is more than the kernel assumed, the excess can be made available
// using a language specific feature, which IR represents as a variable with
// no initializer. This feature is not yet implemented for non-kernel functions.
// This lowering could be extended to handle that use case, but would probably
// require closer integration with promoteAllocaToLDS.
//
// Consequences of this GPU feature:
// - memory is limited and exceeding it halts compilation
// - a global accessed by one kernel exists independent of other kernels
// - a global exists independent of simultaneous execution of the same kernel
// - the address of the global may be different from different kernels as they
//   do not alias, which permits only allocating variables they use
// - if the address is allowed to differ, functions need help to find it
//
// Uses from kernels are implemented here by grouping them in a per-kernel
// struct instance. This duplicates the variables, accurately modelling their
// aliasing properties relative to a single global representation. It also
// permits control over alignment via padding.
//
// Uses from functions are more complicated and the primary purpose of this
// IR pass. Several different lowering are chosen between to meet requirements
// to avoid allocating any LDS where it is not necessary, as that impacts
// occupancy and may fail the compilation, while not imposing overhead on a
// feature whose primary advantage over global memory is performance. The basic
// design goal is to avoid one kernel imposing overhead on another.
//
// Implementation.
//
// LDS variables with constant annotation or non-undef initializer are passed
// through unchanged for simplification or error diagnostics in later passes.
// Non-undef initializers are not yet implemented for LDS.
//
// LDS variables that are always allocated at the same address can be found
// by lookup at that address. Otherwise runtime information/cost is required.
//
// The simplest strategy possible is to group all LDS variables in a single
// struct and allocate that struct in every kernel such that the original
// variables are always at the same address. LDS is however a limited resource
// so this strategy is unusable in practice. It is not implemented here.
//
// Strategy | Precise allocation | Zero runtime cost | General purpose |
//  --------+--------------------+-------------------+-----------------+
//   Module |                 No |               Yes |             Yes |
//    Table |                Yes |                No |             Yes |
//   Kernel |                Yes |               Yes |              No |
//   Hybrid |                Yes |           Partial |             Yes |
//
// Module spends LDS memory to save cycles. Table spends cycles and global
// memory to save LDS. Kernel is as fast as kernel allocation but only works
// for variables that are known reachable from a single kernel. Hybrid picks
// between all three. When forced to choose between LDS and cycles it minimises
// LDS use.

// The "module" lowering implemented here finds LDS variables which are used by
// non-kernel functions and creates a new struct with a field for each of those
// LDS variables. Variables that are only used from kernels are excluded.
// Kernels that do not use this struct are annoteated with the attribute
// amdgpu-elide-module-lds which allows the back end to elide the allocation.
//
// The "table" lowering implemented here has three components.
// First kernels are assigned a unique integer identifier which is available in
// functions it calls through the intrinsic amdgcn_lds_kernel_id. The integer
// is passed through a specific SGPR, thus works with indirect calls.
// Second, each kernel allocates LDS variables independent of other kernels and
// writes the addresses it chose for each variable into an array in consistent
// order. If the kernel does not allocate a given variable, it writes undef to
// the corresponding array location. These arrays are written to a constant
// table in the order matching the kernel unique integer identifier.
// Third, uses from non-kernel functions are replaced with a table lookup using
// the intrinsic function to find the address of the variable.
//
// "Kernel" lowering is only applicable for variables that are unambiguously
// reachable from exactly one kernel. For those cases, accesses to the variable
// can be lowered to ConstantExpr address of a struct instance specific to that
// one kernel. This is zero cost in space and in compute. It will raise a fatal
// error on any variable that might be reachable from multiple kernels and is
// thus most easily used as part of the hybrid lowering strategy.
//
// Hybrid lowering is a mixture of the above. It uses the zero cost kernel
// lowering where it can. It lowers the variable accessed by the greatest
// number of kernels using the module strategy as that is free for the first
// variable. Any futher variables that can be lowered with the module strategy
// without incurring LDS memory overhead are. The remaining ones are lowered
// via table.
//
// Consequences
// - No heuristics or user controlled magic numbers, hybrid is the right choice
// - Kernels that don't use functions (or have had them all inlined) are not
//   affected by any lowering for kernels that do.
// - Kernels that don't make indirect function calls are not affected by those
//   that do.
// - Variables which are used by lots of kernels, e.g. those injected by a
//   language runtime in most kernels, are expected to have no overhead
// - Implementations that instantiate templates per-kernel where those templates
//   use LDS are expected to hit the "Kernel" lowering strategy
// - The runtime properties impose a cost in compiler implementation complexity
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "Utils/AMDGPUMemoryUtils.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/OptimizedStructLayout.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <tuple>
#include <vector>

#include <cstdio>

#define DEBUG_TYPE "amdgpu-lower-module-lds"

using namespace llvm;

namespace {

cl::opt<bool> SuperAlignLDSGlobals(
    "amdgpu-super-align-lds-globals",
    cl::desc("Increase alignment of LDS if it is not on align boundary"),
    cl::init(true), cl::Hidden);

enum class LoweringKind { module, table, kernel, hybrid };
cl::opt<LoweringKind> LoweringKindLoc(
    "amdgpu-lower-module-lds-strategy",
    cl::desc("Specify lowering strategy for function LDS access:"), cl::Hidden,
    cl::init(LoweringKind::module),
    cl::values(
        clEnumValN(LoweringKind::table, "table", "Lower via table lookup"),
        clEnumValN(LoweringKind::module, "module", "Lower via module struct"),
        clEnumValN(
            LoweringKind::kernel, "kernel",
            "Lower variables reachable from one kernel, otherwise abort"),
        clEnumValN(LoweringKind::hybrid, "hybrid",
                   "Lower via mixture of above strategies")));

bool isKernelLDS(const Function *F) {
  // Some weirdness here. AMDGPU::isKernelCC does not call into
  // AMDGPU::isKernel with the calling conv, it instead calls into
  // isModuleEntryFunction which returns true for more calling conventions
  // than AMDGPU::isKernel does. There's a FIXME on AMDGPU::isKernel.
  // There's also a test that checks that the LDS lowering does not hit on
  // a graphics shader, denoted amdgpu_ps, so stay with the limited case.
  // Putting LDS in the name of the function to draw attention to this.
  return AMDGPU::isKernel(F->getCallingConv());
}

class AMDGPULowerModuleLDS : public ModulePass {

  static void
  removeLocalVarsFromUsedLists(Module &M,
                               const DenseSet<GlobalVariable *> &LocalVars) {
    // The verifier rejects used lists containing an inttoptr of a constant
    // so remove the variables from these lists before replaceAllUsesWith
    SmallPtrSet<Constant *, 8> LocalVarsSet;
    for (GlobalVariable *LocalVar : LocalVars)
      LocalVarsSet.insert(cast<Constant>(LocalVar->stripPointerCasts()));

    removeFromUsedLists(
        M, [&LocalVarsSet](Constant *C) { return LocalVarsSet.count(C); });

    for (GlobalVariable *LocalVar : LocalVars)
      LocalVar->removeDeadConstantUsers();
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

  using FunctionVariableMap = DenseMap<Function *, DenseSet<GlobalVariable *>>;

  using VariableFunctionMap = DenseMap<GlobalVariable *, DenseSet<Function *>>;

  static void getUsesOfLDSByFunction(CallGraph const &CG, Module &M,
                                     FunctionVariableMap &kernels,
                                     FunctionVariableMap &functions) {

    // Get uses from the current function, excluding uses by called functions
    // Two output variables to avoid walking the globals list twice
    for (auto &GV : M.globals()) {
      if (!AMDGPU::isLDSVariableToLower(GV)) {
        continue;
      }

      SmallVector<User *, 16> Stack(GV.users());
      for (User *V : GV.users()) {
        if (auto *I = dyn_cast<Instruction>(V)) {
          Function *F = I->getFunction();
          if (isKernelLDS(F)) {
            kernels[F].insert(&GV);
          } else {
            functions[F].insert(&GV);
          }
        }
      }
    }
  }

  struct LDSUsesInfoTy {
    FunctionVariableMap direct_access;
    FunctionVariableMap indirect_access;
  };

  static LDSUsesInfoTy getTransitiveUsesOfLDS(CallGraph const &CG, Module &M) {

    FunctionVariableMap direct_map_kernel;
    FunctionVariableMap direct_map_function;
    getUsesOfLDSByFunction(CG, M, direct_map_kernel, direct_map_function);

    // Collect variables that are used by functions whose address has escaped
    DenseSet<GlobalVariable *> VariablesReachableThroughFunctionPointer;
    for (Function &F : M.functions()) {
      if (!isKernelLDS(&F))
          if (F.hasAddressTaken(nullptr,
                                /* IgnoreCallbackUses */ false,
                                /* IgnoreAssumeLikeCalls */ false,
                                /* IgnoreLLVMUsed */ true,
                                /* IgnoreArcAttachedCall */ false)) {
          set_union(VariablesReachableThroughFunctionPointer,
                    direct_map_function[&F]);
        }
    }

    auto functionMakesUnknownCall = [&](const Function *F) -> bool {
      assert(!F->isDeclaration());
      for (CallGraphNode::CallRecord R : *CG[F]) {
        if (!R.second->getFunction()) {
          return true;
        }
      }
      return false;
    };

    // Work out which variables are reachable through function calls
    FunctionVariableMap transitive_map_function = direct_map_function;

    // If the function makes any unknown call, assume the worst case that it can
    // access all variables accessed by functions whose address escaped
    for (Function &F : M.functions()) {
      if (!F.isDeclaration() && functionMakesUnknownCall(&F)) {
        if (!isKernelLDS(&F)) {
          set_union(transitive_map_function[&F],
                    VariablesReachableThroughFunctionPointer);
        }
      }
    }

    // Direct implementation of collecting all variables reachable from each
    // function
    for (Function &Func : M.functions()) {
      if (Func.isDeclaration() || isKernelLDS(&Func))
        continue;

      DenseSet<Function *> seen; // catches cycles
      SmallVector<Function *, 4> wip{&Func};

      while (!wip.empty()) {
        Function *F = wip.pop_back_val();

        // Can accelerate this by referring to transitive map for functions that
        // have already been computed, with more care than this
        set_union(transitive_map_function[&Func], direct_map_function[F]);

        for (CallGraphNode::CallRecord R : *CG[F]) {
          Function *ith = R.second->getFunction();
          if (ith) {
            if (!seen.contains(ith)) {
              seen.insert(ith);
              wip.push_back(ith);
            }
          }
        }
      }
    }

    // direct_map_kernel lists which variables are used by the kernel
    // find the variables which are used through a function call
    FunctionVariableMap indirect_map_kernel;

    for (Function &Func : M.functions()) {
      if (Func.isDeclaration() || !isKernelLDS(&Func))
        continue;

      for (CallGraphNode::CallRecord R : *CG[&Func]) {
        Function *ith = R.second->getFunction();
        if (ith) {
          set_union(indirect_map_kernel[&Func], transitive_map_function[ith]);
        } else {
          set_union(indirect_map_kernel[&Func],
                    VariablesReachableThroughFunctionPointer);
        }
      }
    }

    return {std::move(direct_map_kernel), std::move(indirect_map_kernel)};
  }

  struct LDSVariableReplacement {
    GlobalVariable *SGV = nullptr;
    DenseMap<GlobalVariable *, Constant *> LDSVarsToConstantGEP;
  };

  // remap from lds global to a constantexpr gep to where it has been moved to
  // for each kernel
  // an array with an element for each kernel containing where the corresponding
  // variable was remapped to

  static Constant *getAddressesOfVariablesInKernel(
      LLVMContext &Ctx, ArrayRef<GlobalVariable *> Variables,
      DenseMap<GlobalVariable *, Constant *> &LDSVarsToConstantGEP) {
    // Create a ConstantArray containing the address of each Variable within the
    // kernel corresponding to LDSVarsToConstantGEP, or poison if that kernel
    // does not allocate it
    // TODO: Drop the ptrtoint conversion

    Type *I32 = Type::getInt32Ty(Ctx);

    ArrayType *KernelOffsetsType = ArrayType::get(I32, Variables.size());

    SmallVector<Constant *> Elements;
    for (size_t i = 0; i < Variables.size(); i++) {
      GlobalVariable *GV = Variables[i];
      if (LDSVarsToConstantGEP.count(GV) != 0) {
        auto elt = ConstantExpr::getPtrToInt(LDSVarsToConstantGEP[GV], I32);
        Elements.push_back(elt);
      } else {
        Elements.push_back(PoisonValue::get(I32));
      }
    }
    return ConstantArray::get(KernelOffsetsType, Elements);
  }

  static GlobalVariable *buildLookupTable(
      Module &M, ArrayRef<GlobalVariable *> Variables,
      ArrayRef<Function *> kernels,
      DenseMap<Function *, LDSVariableReplacement> &KernelToReplacement) {
    if (Variables.empty()) {
      return nullptr;
    }
    LLVMContext &Ctx = M.getContext();

    const size_t NumberVariables = Variables.size();
    const size_t NumberKernels = kernels.size();

    ArrayType *KernelOffsetsType =
        ArrayType::get(Type::getInt32Ty(Ctx), NumberVariables);

    ArrayType *AllKernelsOffsetsType =
        ArrayType::get(KernelOffsetsType, NumberKernels);

    std::vector<Constant *> overallConstantExprElts(NumberKernels);
    for (size_t i = 0; i < NumberKernels; i++) {
      LDSVariableReplacement Replacement = KernelToReplacement[kernels[i]];
      overallConstantExprElts[i] = getAddressesOfVariablesInKernel(
          Ctx, Variables, Replacement.LDSVarsToConstantGEP);
    }

    Constant *init =
        ConstantArray::get(AllKernelsOffsetsType, overallConstantExprElts);

    return new GlobalVariable(
        M, AllKernelsOffsetsType, true, GlobalValue::InternalLinkage, init,
        "llvm.amdgcn.lds.offset.table", nullptr, GlobalValue::NotThreadLocal,
        AMDGPUAS::CONSTANT_ADDRESS);
  }

  void replaceUsesInInstructionsWithTableLookup(
      Module &M, ArrayRef<GlobalVariable *> ModuleScopeVariables,
      GlobalVariable *LookupTable) {

    LLVMContext &Ctx = M.getContext();
    IRBuilder<> Builder(Ctx);
    Type *I32 = Type::getInt32Ty(Ctx);

    // Accesses from a function use the amdgcn_lds_kernel_id intrinsic which
    // lowers to a read from a live in register. Emit it once in the entry
    // block to spare deduplicating it later.

    DenseMap<Function *, Value *> tableKernelIndexCache;
    auto getTableKernelIndex = [&](Function *F) -> Value * {
      if (tableKernelIndexCache.count(F) == 0) {
        LLVMContext &Ctx = M.getContext();
        FunctionType *FTy = FunctionType::get(Type::getInt32Ty(Ctx), {});
        Function *Decl =
            Intrinsic::getDeclaration(&M, Intrinsic::amdgcn_lds_kernel_id, {});

        BasicBlock::iterator it =
            F->getEntryBlock().getFirstNonPHIOrDbgOrAlloca();
        Instruction &i = *it;
        Builder.SetInsertPoint(&i);

        tableKernelIndexCache[F] = Builder.CreateCall(FTy, Decl, {});
      }

      return tableKernelIndexCache[F];
    };

    for (size_t Index = 0; Index < ModuleScopeVariables.size(); Index++) {
      auto *GV = ModuleScopeVariables[Index];

      for (Use &U : make_early_inc_range(GV->uses())) {
        auto *I = dyn_cast<Instruction>(U.getUser());
        if (!I)
          continue;

        Value *tableKernelIndex = getTableKernelIndex(I->getFunction());

        // So if the phi uses this value multiple times, what does this look
        // like?
        if (auto *Phi = dyn_cast<PHINode>(I)) {
          BasicBlock *BB = Phi->getIncomingBlock(U);
          Builder.SetInsertPoint(&(*(BB->getFirstInsertionPt())));
        } else {
          Builder.SetInsertPoint(I);
        }

        Value *GEPIdx[3] = {
            ConstantInt::get(I32, 0),
            tableKernelIndex,
            ConstantInt::get(I32, Index),
        };

        Value *Address = Builder.CreateInBoundsGEP(
            LookupTable->getValueType(), LookupTable, GEPIdx, GV->getName());

        Value *loaded = Builder.CreateLoad(I32, Address);

        Value *replacement =
            Builder.CreateIntToPtr(loaded, GV->getType(), GV->getName());

        U.set(replacement);
      }
    }
  }

  static DenseSet<Function *> kernelsThatIndirectlyAccessAnyOfPassedVariables(
      Module &M, LDSUsesInfoTy &LDSUsesInfo,
      DenseSet<GlobalVariable *> const &VariableSet) {

    DenseSet<Function *> KernelSet;

    if (VariableSet.empty()) return KernelSet;

    for (Function &Func : M.functions()) {
      if (Func.isDeclaration() || !isKernelLDS(&Func))
        continue;
      for (GlobalVariable *GV : LDSUsesInfo.indirect_access[&Func]) {
        if (VariableSet.contains(GV)) {
          KernelSet.insert(&Func);
          break;
        }
      }
    }

    return KernelSet;
  }

  static GlobalVariable *
  chooseBestVariableForModuleStrategy(const DataLayout &DL,
                                      VariableFunctionMap &LDSVars) {
    // Find the global variable with the most indirect uses from kernels

    struct CandidateTy {
      GlobalVariable *GV = nullptr;
      size_t UserCount = 0;
      size_t Size = 0;

      CandidateTy() = default;

      CandidateTy(GlobalVariable *GV, uint64_t UserCount, uint64_t AllocSize)
          : GV(GV), UserCount(UserCount), Size(AllocSize) {}

      bool operator<(const CandidateTy &Other) const {
        // Fewer users makes module scope variable less attractive
        if (UserCount < Other.UserCount) {
          return true;
        }
        if (UserCount > Other.UserCount) {
          return false;
        }

        // Bigger makes module scope variable less attractive
        if (Size < Other.Size) {
          return false;
        }

        if (Size > Other.Size) {
          return true;
        }

        // Arbitrary but consistent
        return GV->getName() < Other.GV->getName();
      }
    };

    CandidateTy MostUsed;

    for (auto &K : LDSVars) {
      GlobalVariable *GV = K.first;
      if (K.second.size() <= 1) {
        // A variable reachable by only one kernel is best lowered with kernel
        // strategy
        continue;
      }
      CandidateTy Candidate(GV, K.second.size(),
                      DL.getTypeAllocSize(GV->getValueType()).getFixedValue());
      if (MostUsed < Candidate)
        MostUsed = Candidate;
    }

    return MostUsed.GV;
  }

  bool runOnModule(Module &M) override {
    LLVMContext &Ctx = M.getContext();
    CallGraph CG = CallGraph(M);
    bool Changed = superAlignLDSGlobals(M);

    Changed |= eliminateConstantExprUsesOfLDSFromAllInstructions(M);

    Changed = true; // todo: narrow this down

    // For each kernel, what variables does it access directly or through
    // callees
    LDSUsesInfoTy LDSUsesInfo = getTransitiveUsesOfLDS(CG, M);

    // For each variable accessed through callees, which kernels access it
    VariableFunctionMap LDSToKernelsThatNeedToAccessItIndirectly;
    for (auto &K : LDSUsesInfo.indirect_access) {
      Function *F = K.first;
      assert(isKernelLDS(F));
      for (GlobalVariable *GV : K.second) {
        LDSToKernelsThatNeedToAccessItIndirectly[GV].insert(F);
      }
    }

    // Partition variables into the different strategies
    DenseSet<GlobalVariable *> ModuleScopeVariables;
    DenseSet<GlobalVariable *> TableLookupVariables;
    DenseSet<GlobalVariable *> KernelAccessVariables;

    {
      GlobalVariable *HybridModuleRoot =
          LoweringKindLoc != LoweringKind::hybrid
              ? nullptr
              : chooseBestVariableForModuleStrategy(
                    M.getDataLayout(),
                    LDSToKernelsThatNeedToAccessItIndirectly);

      DenseSet<Function *> const EmptySet;
      DenseSet<Function *> const &HybridModuleRootKernels =
          HybridModuleRoot
              ? LDSToKernelsThatNeedToAccessItIndirectly[HybridModuleRoot]
              : EmptySet;

      for (auto &K : LDSToKernelsThatNeedToAccessItIndirectly) {
        // Each iteration of this loop assigns exactly one global variable to
        // exactly one of the implementation strategies.

        GlobalVariable *GV = K.first;
        assert(AMDGPU::isLDSVariableToLower(*GV));
        assert(K.second.size() != 0);

        switch (LoweringKindLoc) {
        case LoweringKind::module:
          ModuleScopeVariables.insert(GV);
          break;

        case LoweringKind::table:
          TableLookupVariables.insert(GV);
          break;

        case LoweringKind::kernel:
          if (K.second.size() == 1) {
            KernelAccessVariables.insert(GV);
          } else {
            report_fatal_error(
                "cannot lower LDS '" + GV->getName() +
                "' to kernel access as it is reachable from multiple kernels");
          }
          break;

        case LoweringKind::hybrid: {
          if (GV == HybridModuleRoot) {
            assert(K.second.size() != 1);
            ModuleScopeVariables.insert(GV);
          } else if (K.second.size() == 1) {
            KernelAccessVariables.insert(GV);
          } else if (set_is_subset(K.second, HybridModuleRootKernels)) {
            ModuleScopeVariables.insert(GV);
          } else {
            TableLookupVariables.insert(GV);
          }
          break;
        }
        }
      }

      assert(ModuleScopeVariables.size() + TableLookupVariables.size() +
                 KernelAccessVariables.size() ==
             LDSToKernelsThatNeedToAccessItIndirectly.size());
    } // Variables have now been partitioned into the three lowering strategies.

    // If the kernel accesses a variable that is going to be stored in the
    // module instance through a call then that kernel needs to allocate the
    // module instance
    DenseSet<Function *> KernelsThatAllocateModuleLDS =
        kernelsThatIndirectlyAccessAnyOfPassedVariables(M, LDSUsesInfo,
                                                        ModuleScopeVariables);
    DenseSet<Function *> KernelsThatAllocateTableLDS =
        kernelsThatIndirectlyAccessAnyOfPassedVariables(M, LDSUsesInfo,
                                                        TableLookupVariables);

    if (!ModuleScopeVariables.empty()) {
      LDSVariableReplacement ModuleScopeReplacement =
          createLDSVariableReplacement(M, "llvm.amdgcn.module.lds",
                                       ModuleScopeVariables);

      appendToCompilerUsed(M,
                           {static_cast<GlobalValue *>(
                               ConstantExpr::getPointerBitCastOrAddrSpaceCast(
                                   cast<Constant>(ModuleScopeReplacement.SGV),
                                   Type::getInt8PtrTy(Ctx)))});

      // historic
      removeLocalVarsFromUsedLists(M, ModuleScopeVariables);

      // Replace all uses of module scope variable from non-kernel functions
      replaceLDSVariablesWithStruct(
          M, ModuleScopeVariables, ModuleScopeReplacement, [&](Use &U) {
            Instruction *I = dyn_cast<Instruction>(U.getUser());
            if (!I) {
              return false;
            }
            Function *F = I->getFunction();
            return !isKernelLDS(F);
          });

      // Replace uses of module scope variable from kernel functions that
      // allocate the module scope variable, otherwise leave them unchanged
      // Record on each kernel whether the module scope global is used by it

      LLVMContext &Ctx = M.getContext();
      IRBuilder<> Builder(Ctx);

      for (Function &Func : M.functions()) {
        if (Func.isDeclaration() || !isKernelLDS(&Func))
          continue;

        if (KernelsThatAllocateModuleLDS.contains(&Func)) {
          replaceLDSVariablesWithStruct(
              M, ModuleScopeVariables, ModuleScopeReplacement, [&](Use &U) {
                Instruction *I = dyn_cast<Instruction>(U.getUser());
                if (!I) {
                  return false;
                }
                Function *F = I->getFunction();
                return F == &Func;
              });

          markUsedByKernel(Builder, &Func, ModuleScopeReplacement.SGV);

        } else {
          Func.addFnAttr("amdgpu-elide-module-lds");
        }
      }
    }

    // Create a struct for each kernel for the non-module-scope variables
    DenseMap<Function *, LDSVariableReplacement> KernelToReplacement;
    for (Function &Func : M.functions()) {
      if (Func.isDeclaration() || !isKernelLDS(&Func))
        continue;

      DenseSet<GlobalVariable *> KernelUsedVariables;
      for (auto &v : LDSUsesInfo.direct_access[&Func]) {
        KernelUsedVariables.insert(v);
      }
      for (auto &v : LDSUsesInfo.indirect_access[&Func]) {
        KernelUsedVariables.insert(v);
      }

      // Variables allocated in module lds must all resolve to that struct,
      // not to the per-kernel instance.
      if (KernelsThatAllocateModuleLDS.contains(&Func)) {
        for (GlobalVariable *v : ModuleScopeVariables) {
          KernelUsedVariables.erase(v);
        }
      }

      if (KernelUsedVariables.empty()) {
        // Either used no LDS, or all the LDS it used was also in module
        continue;
      }

      // The association between kernel function and LDS struct is done by
      // symbol name, which only works if the function in question has a
      // name This is not expected to be a problem in practice as kernels
      // are called by name making anonymous ones (which are named by the
      // backend) difficult to use. This does mean that llvm test cases need
      // to name the kernels.
      if (!Func.hasName()) {
        report_fatal_error("Anonymous kernels cannot use LDS variables");
      }

      std::string VarName =
          (Twine("llvm.amdgcn.kernel.") + Func.getName() + ".lds").str();

      auto Replacement =
          createLDSVariableReplacement(M, VarName, KernelUsedVariables);

      // remove preserves existing codegen
      removeLocalVarsFromUsedLists(M, KernelUsedVariables);
      KernelToReplacement[&Func] = Replacement;

      // Rewrite uses within kernel to the new struct
      replaceLDSVariablesWithStruct(
          M, KernelUsedVariables, Replacement, [&Func](Use &U) {
            Instruction *I = dyn_cast<Instruction>(U.getUser());
            return I && I->getFunction() == &Func;
          });
    }

    // Lower zero cost accesses to the kernel instances just created
    for (auto &GV : KernelAccessVariables) {
      auto &funcs = LDSToKernelsThatNeedToAccessItIndirectly[GV];
      assert(funcs.size() == 1); // Only one kernel can access it
      LDSVariableReplacement Replacement =
          KernelToReplacement[*(funcs.begin())];

      DenseSet<GlobalVariable *> Vec;
      Vec.insert(GV);

      replaceLDSVariablesWithStruct(M, Vec, Replacement, [](Use &U) {
                                                           return isa<Instruction>(U.getUser());
      });
    }

    if (!KernelsThatAllocateTableLDS.empty()) {
      // Collect the kernels that allocate table lookup LDS
      std::vector<Function *> OrderedKernels;
      {
        for (Function &Func : M.functions()) {
          if (Func.isDeclaration())
            continue;
          if (!isKernelLDS(&Func))
            continue;

          if (KernelsThatAllocateTableLDS.contains(&Func)) {
            assert(Func.hasName()); // else fatal error earlier
            OrderedKernels.push_back(&Func);
          }
        }

        // Put them in an arbitrary but reproducible order
        llvm::sort(OrderedKernels.begin(), OrderedKernels.end(),
                   [](const Function *lhs, const Function *rhs) -> bool {
                     return lhs->getName() < rhs->getName();
                   });

        // Annotate the kernels with their order in this vector
        LLVMContext &Ctx = M.getContext();
        IRBuilder<> Builder(Ctx);

        if (OrderedKernels.size() > UINT32_MAX) {
          // 32 bit keeps it in one SGPR. > 2**32 kernels won't fit on the GPU
          report_fatal_error("Unimplemented LDS lowering for > 2**32 kernels");
        }

        for (size_t i = 0; i < OrderedKernels.size(); i++) {
          Metadata *AttrMDArgs[1] = {
              ConstantAsMetadata::get(Builder.getInt32(i)),
          };
          OrderedKernels[i]->setMetadata("llvm.amdgcn.lds.kernel.id",
                                         MDNode::get(Ctx, AttrMDArgs));

          markUsedByKernel(Builder, OrderedKernels[i],
                           KernelToReplacement[OrderedKernels[i]].SGV);
        }
      }

      // The order must be consistent between lookup table and accesses to
      // lookup table
      std::vector<GlobalVariable *> TableLookupVariablesOrdered(
          TableLookupVariables.begin(), TableLookupVariables.end());
      llvm::sort(TableLookupVariablesOrdered.begin(),
                 TableLookupVariablesOrdered.end(),
                 [](const GlobalVariable *lhs, const GlobalVariable *rhs) {
                   return lhs->getName() < rhs->getName();
                 });

      GlobalVariable *LookupTable = buildLookupTable(
          M, TableLookupVariablesOrdered, OrderedKernels, KernelToReplacement);
      replaceUsesInInstructionsWithTableLookup(M, TableLookupVariablesOrdered,
                                               LookupTable);
    }

    for (auto &GV : make_early_inc_range(M.globals()))
      if (AMDGPU::isLDSVariableToLower(GV)) {

        // probably want to remove from used lists
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

  static LDSVariableReplacement createLDSVariableReplacement(
      Module &M, std::string VarName,
      DenseSet<GlobalVariable *> const &LDSVarsToTransform) {
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
    return {SGV, std::move(Map)};
  }

  template <typename PredicateTy>
  void replaceLDSVariablesWithStruct(
      Module &M, DenseSet<GlobalVariable *> const &LDSVarsToTransformArg,
      LDSVariableReplacement Replacement, PredicateTy Predicate) {
    LLVMContext &Ctx = M.getContext();
    const DataLayout &DL = M.getDataLayout();

    // A hack... we need to insert the aliasing info in a predictable order for
    // lit tests. Would like to have them in a stable order already, ideally the
    // same order they get allocated, which might mean an ordered set container
    std::vector<GlobalVariable *> LDSVarsToTransform(
        LDSVarsToTransformArg.begin(), LDSVarsToTransformArg.end());
    llvm::sort(LDSVarsToTransform.begin(), LDSVarsToTransform.end(),
               [](const GlobalVariable *lhs, const GlobalVariable *rhs) {
                 return lhs->getName() < rhs->getName();
               });

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
      Constant *GEP = Replacement.LDSVarsToConstantGEP[GV];

      GV->replaceUsesWithIf(GEP, Predicate);

      APInt APOff(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
      GEP->stripAndAccumulateInBoundsConstantOffsets(DL, APOff);
      uint64_t Offset = APOff.getZExtValue();

      Align A =
          commonAlignment(Replacement.SGV->getAlign().valueOrOne(), Offset);

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
