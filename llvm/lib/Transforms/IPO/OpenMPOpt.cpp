//===-- IPO/OpenMPOpt.cpp - Collection of OpenMP specific optimizations ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpenMP specific optimizations:
//
// - Deduplication of runtime calls, e.g., omp_get_thread_num.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/OpenMPOpt.h"

#include "llvm/ADT/EnumeratedArray.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/CallSite.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/CallGraphUpdater.h"

using namespace llvm;
using namespace omp;
using namespace types;

#define DEBUG_TYPE "openmp-opt"

static cl::opt<bool> DisableOpenMPOptimizations(
    "openmp-opt-disable", cl::ZeroOrMore,
    cl::desc("Disable OpenMP specific optimizations."), cl::Hidden,
    cl::init(false));

STATISTIC(NumOpenMPRuntimeCallsDeduplicated,
          "Number of OpenMP runtime calls deduplicated");
STATISTIC(NumOpenMPRuntimeFunctionsIdentified,
          "Number of OpenMP runtime functions identified");
STATISTIC(NumOpenMPRuntimeFunctionUsesIdentified,
          "Number of OpenMP runtime function uses identified");

#if !defined(NDEBUG)
static constexpr auto TAG = "[" DEBUG_TYPE "]";
#endif

namespace {
struct OpenMPOpt {

  OpenMPOpt(SmallPtrSetImpl<Function *> &SCC,
            SmallPtrSetImpl<Function *> &ModuleSlice,
            CallGraphUpdater &CGUpdater)
      : M(*(*SCC.begin())->getParent()), SCC(SCC), ModuleSlice(ModuleSlice),
        OMPBuilder(M), CGUpdater(CGUpdater) {
    initializeTypes(M);
    initializeRuntimeFunctions();
    OMPBuilder.initialize();
  }

  /// Generic information that describes a runtime function
  struct RuntimeFunctionInfo {
    /// The kind, as described by the RuntimeFunction enum.
    RuntimeFunction Kind;

    /// The name of the function.
    StringRef Name;

    /// Flag to indicate a variadic function.
    bool IsVarArg;

    /// The return type of the function.
    Type *ReturnType;

    /// The argument types of the function.
    SmallVector<Type *, 8> ArgumentTypes;

    /// The declaration if available.
    Function *Declaration;

    /// Uses of this runtime function per function containing the use.
    DenseMap<Function *, SmallPtrSet<Use *, 16>> UsesMap;

    /// Return the number of arguments (or the minimal number for variadic
    /// functions).
    size_t getNumArgs() const { return ArgumentTypes.size(); }

    /// Run the callback \p CB on each use and forget the use if the result is
    /// true. The callback will be fed the function in which the use was
    /// encountered as second argument.
    void foreachUse(function_ref<bool(Use &, Function &)> CB) {
      SmallVector<Use *, 8> ToBeDeleted;
      for (auto &It : UsesMap) {
        ToBeDeleted.clear();
        for (Use *U : It.second)
          if (CB(*U, *It.first))
            ToBeDeleted.push_back(U);
        for (Use *U : ToBeDeleted)
          It.second.erase(U);
      }
    }
  };

  /// Run all OpenMP optimizations on the underlying SCC/ModuleSlice.
  bool run() {
    bool Changed = false;

    LLVM_DEBUG(dbgs() << TAG << "Run on SCC with " << SCC.size()
                      << " functions in a slice with " << ModuleSlice.size()
                      << " functions\n");

    Changed |= deduplicateRuntimeCalls();
    Changed |= deleteParallelRegions();

    return Changed;
  }

private:
  /// Try to delete parallel regions if possible
  bool deleteParallelRegions() {
    const unsigned CallbackCalleeOperand = 2;

    RuntimeFunctionInfo &RFI = RFIs[OMPRTL___kmpc_fork_call];
    if (!RFI.Declaration)
      return false;

    bool Changed = false;
    auto DeleteCallCB = [&](Use &U, Function &) {
      CallInst *CI = getCallIfRegularCall(U);
      if (!CI)
        return false;
      auto *Fn = dyn_cast<Function>(
          CI->getArgOperand(CallbackCalleeOperand)->stripPointerCasts());
      if (!Fn)
        return false;
      if (!Fn->onlyReadsMemory())
        return false;
      if (!Fn->hasFnAttribute(Attribute::WillReturn))
        return false;

      LLVM_DEBUG(dbgs() << TAG << "Delete read-only parallel region in "
                        << CI->getCaller()->getName() << "\n");
      CGUpdater.removeCallSite(*CI);
      CI->eraseFromParent();
      Changed = true;
      return true;
    };

    RFI.foreachUse(DeleteCallCB);

    return Changed;
  }

  /// Try to eliminiate runtime calls by reusing existing ones.
  bool deduplicateRuntimeCalls() {
    bool Changed = false;

    RuntimeFunction DeduplicableRuntimeCallIDs[] = {
        OMPRTL_omp_get_num_threads,
        OMPRTL_omp_in_parallel,
        OMPRTL_omp_get_cancellation,
        OMPRTL_omp_get_thread_limit,
        OMPRTL_omp_get_supported_active_levels,
        OMPRTL_omp_get_level,
        OMPRTL_omp_get_ancestor_thread_num,
        OMPRTL_omp_get_team_size,
        OMPRTL_omp_get_active_level,
        OMPRTL_omp_in_final,
        OMPRTL_omp_get_proc_bind,
        OMPRTL_omp_get_num_places,
        OMPRTL_omp_get_num_procs,
        OMPRTL_omp_get_place_num,
        OMPRTL_omp_get_partition_num_places,
        OMPRTL_omp_get_partition_place_nums};

    // Global-tid is handled separatly.
    SmallSetVector<Value *, 16> GTIdArgs;
    collectGlobalThreadIdArguments(GTIdArgs);
    LLVM_DEBUG(dbgs() << TAG << "Found " << GTIdArgs.size()
                      << " global thread ID arguments\n");

    for (Function *F : SCC) {
      for (auto DeduplicableRuntimeCallID : DeduplicableRuntimeCallIDs)
        deduplicateRuntimeCalls(*F, RFIs[DeduplicableRuntimeCallID]);

      // __kmpc_global_thread_num is special as we can replace it with an
      // argument in enough cases to make it worth trying.
      Value *GTIdArg = nullptr;
      for (Argument &Arg : F->args())
        if (GTIdArgs.count(&Arg)) {
          GTIdArg = &Arg;
          break;
        }
      Changed |= deduplicateRuntimeCalls(
          *F, RFIs[OMPRTL___kmpc_global_thread_num], GTIdArg);
    }

    return Changed;
  }

  static Value *combinedIdentStruct(Value *Ident0, Value *Ident1,
                                    bool GlobalOnly) {
    // TODO: Figure out how to actually combine multiple debug locations. For
    //       now we just keep the first we find.
    if (Ident0)
      return Ident0;
    if (!GlobalOnly || isa<GlobalValue>(Ident1))
      return Ident1;
    return nullptr;
  }

  /// Return an `struct ident_t*` value that represents the ones used in the
  /// calls of \p RFI inside of \p F. If \p GlobalOnly is true, we will not
  /// return a local `struct ident_t*`. For now, if we cannot find a suitable
  /// return value we create one from scratch. We also do not yet combine
  /// information, e.g., the source locations, see combinedIdentStruct.
  Value *getCombinedIdentFromCallUsesIn(RuntimeFunctionInfo &RFI, Function &F,
                                        bool GlobalOnly) {
    Value *Ident = nullptr;
    auto CombineIdentStruct = [&](Use &U, Function &Caller) {
      CallInst *CI = getCallIfRegularCall(U, &RFI);
      if (!CI || &F != &Caller)
        return false;
      Ident = combinedIdentStruct(Ident, CI->getArgOperand(0),
                                  /* GlobalOnly */ true);
      return false;
    };
    RFI.foreachUse(CombineIdentStruct);

    if (!Ident) {
      // The IRBuilder uses the insertion block to get to the module, this is
      // unfortunate but we work around it for now.
      if (!OMPBuilder.getInsertionPoint().getBlock())
        OMPBuilder.updateToLocation(OpenMPIRBuilder::InsertPointTy(
            &F.getEntryBlock(), F.getEntryBlock().begin()));
      // Create a fallback location if non was found.
      // TODO: Use the debug locations of the calls instead.
      Constant *Loc = OMPBuilder.getOrCreateDefaultSrcLocStr();
      Ident = OMPBuilder.getOrCreateIdent(Loc);
    }
    return Ident;
  }

  /// Try to eliminiate calls of \p RFI in \p F by reusing an existing one or
  /// \p ReplVal if given.
  bool deduplicateRuntimeCalls(Function &F, RuntimeFunctionInfo &RFI,
                               Value *ReplVal = nullptr) {
    auto &Uses = RFI.UsesMap[&F];
    if (Uses.size() + (ReplVal != nullptr) < 2)
      return false;

    LLVM_DEBUG(dbgs() << TAG << "Deduplicate " << Uses.size() << " uses of "
                      << RFI.Name
                      << (ReplVal ? " with an existing value\n" : "\n")
                      << "\n");
    assert((!ReplVal || (isa<Argument>(ReplVal) &&
                         cast<Argument>(ReplVal)->getParent() == &F)) &&
           "Unexpected replacement value!");

    // TODO: Use dominance to find a good position instead.
    auto CanBeMoved = [](CallBase &CB) {
      unsigned NumArgs = CB.getNumArgOperands();
      if (NumArgs == 0)
        return true;
      if (CB.getArgOperand(0)->getType() != IdentPtr)
        return false;
      for (unsigned u = 1; u < NumArgs; ++u)
        if (isa<Instruction>(CB.getArgOperand(u)))
          return false;
      return true;
    };

    if (!ReplVal) {
      for (Use *U : Uses)
        if (CallInst *CI = getCallIfRegularCall(*U, &RFI)) {
          if (!CanBeMoved(*CI))
            continue;
          CI->moveBefore(&*F.getEntryBlock().getFirstInsertionPt());
          ReplVal = CI;
          break;
        }
      if (!ReplVal)
        return false;
    }

    // If we use a call as a replacement value we need to make sure the ident is
    // valid at the new location. For now we just pick a global one, either
    // existing and used by one of the calls, or created from scratch.
    if (CallBase *CI = dyn_cast<CallBase>(ReplVal)) {
      if (CI->getNumArgOperands() > 0 &&
          CI->getArgOperand(0)->getType() == IdentPtr) {
        Value *Ident = getCombinedIdentFromCallUsesIn(RFI, F,
                                                      /* GlobalOnly */ true);
        CI->setArgOperand(0, Ident);
      }
    }

    bool Changed = false;
    auto ReplaceAndDeleteCB = [&](Use &U, Function &Caller) {
      CallInst *CI = getCallIfRegularCall(U, &RFI);
      if (!CI || CI == ReplVal || &F != &Caller)
        return false;
      assert(CI->getCaller() == &F && "Unexpected call!");
      CGUpdater.removeCallSite(*CI);
      CI->replaceAllUsesWith(ReplVal);
      CI->eraseFromParent();
      ++NumOpenMPRuntimeCallsDeduplicated;
      Changed = true;
      return true;
    };
    RFI.foreachUse(ReplaceAndDeleteCB);

    return Changed;
  }

  /// Collect arguments that represent the global thread id in \p GTIdArgs.
  void collectGlobalThreadIdArguments(SmallSetVector<Value *, 16> &GTIdArgs) {
    // TODO: Below we basically perform a fixpoint iteration with a pessimistic
    //       initialization. We could define an AbstractAttribute instead and
    //       run the Attributor here once it can be run as an SCC pass.

    // Helper to check the argument \p ArgNo at all call sites of \p F for
    // a GTId.
    auto CallArgOpIsGTId = [&](Function &F, unsigned ArgNo, CallInst &RefCI) {
      if (!F.hasLocalLinkage())
        return false;
      for (Use &U : F.uses()) {
        if (CallInst *CI = getCallIfRegularCall(U)) {
          Value *ArgOp = CI->getArgOperand(ArgNo);
          if (CI == &RefCI || GTIdArgs.count(ArgOp) ||
              getCallIfRegularCall(*ArgOp,
                                   &RFIs[OMPRTL___kmpc_global_thread_num]))
            continue;
        }
        return false;
      }
      return true;
    };

    // Helper to identify uses of a GTId as GTId arguments.
    auto AddUserArgs = [&](Value &GTId) {
      for (Use &U : GTId.uses())
        if (CallInst *CI = dyn_cast<CallInst>(U.getUser()))
          if (CI->isArgOperand(&U))
            if (Function *Callee = CI->getCalledFunction())
              if (CallArgOpIsGTId(*Callee, U.getOperandNo(), *CI))
                GTIdArgs.insert(Callee->getArg(U.getOperandNo()));
    };

    // The argument users of __kmpc_global_thread_num calls are GTIds.
    RuntimeFunctionInfo &GlobThreadNumRFI =
        RFIs[OMPRTL___kmpc_global_thread_num];
    for (auto &It : GlobThreadNumRFI.UsesMap)
      for (Use *U : It.second)
        if (CallInst *CI = getCallIfRegularCall(*U, &GlobThreadNumRFI))
          AddUserArgs(*CI);

    // Transitively search for more arguments by looking at the users of the
    // ones we know already. During the search the GTIdArgs vector is extended
    // so we cannot cache the size nor can we use a range based for.
    for (unsigned u = 0; u < GTIdArgs.size(); ++u)
      AddUserArgs(*GTIdArgs[u]);
  }

  /// Return the call if \p U is a callee use in a regular call. If \p RFI is
  /// given it has to be the callee or a nullptr is returned.
  CallInst *getCallIfRegularCall(Use &U, RuntimeFunctionInfo *RFI = nullptr) {
    CallInst *CI = dyn_cast<CallInst>(U.getUser());
    if (CI && CI->isCallee(&U) && !CI->hasOperandBundles() &&
        (!RFI || CI->getCalledFunction() == RFI->Declaration))
      return CI;
    return nullptr;
  }

  /// Return the call if \p V is a regular call. If \p RFI is given it has to be
  /// the callee or a nullptr is returned.
  CallInst *getCallIfRegularCall(Value &V, RuntimeFunctionInfo *RFI = nullptr) {
    CallInst *CI = dyn_cast<CallInst>(&V);
    if (CI && !CI->hasOperandBundles() &&
        (!RFI || CI->getCalledFunction() == RFI->Declaration))
      return CI;
    return nullptr;
  }

  /// Helper to initialize all runtime function information for those defined in
  /// OpenMPKinds.def.
  void initializeRuntimeFunctions() {
    // Helper to collect all uses of the decleration in the UsesMap.
    auto CollectUses = [&](RuntimeFunctionInfo &RFI) {
      unsigned NumUses = 0;
      if (!RFI.Declaration)
        return NumUses;
      OMPBuilder.addAttributes(RFI.Kind, *RFI.Declaration);

      NumOpenMPRuntimeFunctionsIdentified += 1;
      NumOpenMPRuntimeFunctionUsesIdentified += RFI.Declaration->getNumUses();

      // TODO: We directly convert uses into proper calls and unknown uses.
      for (Use &U : RFI.Declaration->uses()) {
        if (Instruction *UserI = dyn_cast<Instruction>(U.getUser())) {
          if (ModuleSlice.count(UserI->getFunction())) {
            RFI.UsesMap[UserI->getFunction()].insert(&U);
            ++NumUses;
          }
        } else {
          RFI.UsesMap[nullptr].insert(&U);
          ++NumUses;
        }
      }
      return NumUses;
    };

#define OMP_RTL(_Enum, _Name, _IsVarArg, _ReturnType, ...)                     \
  {                                                                            \
    auto &RFI = RFIs[_Enum];                                                   \
    RFI.Kind = _Enum;                                                          \
    RFI.Name = _Name;                                                          \
    RFI.IsVarArg = _IsVarArg;                                                  \
    RFI.ReturnType = _ReturnType;                                              \
    RFI.ArgumentTypes = SmallVector<Type *, 8>({__VA_ARGS__});                 \
    RFI.Declaration = M.getFunction(_Name);                                    \
    unsigned NumUses = CollectUses(RFI);                                       \
    (void)NumUses;                                                             \
    LLVM_DEBUG({                                                               \
      dbgs() << TAG << RFI.Name << (RFI.Declaration ? "" : " not")             \
             << " found\n";                                                    \
      if (RFI.Declaration)                                                     \
        dbgs() << TAG << "-> got " << NumUses << " uses in "                   \
               << RFI.UsesMap.size() << " different functions.\n";             \
    });                                                                        \
  }
#include "llvm/Frontend/OpenMP/OMPKinds.def"

    // TODO: We should validate the declaration agains the types we expect.
    // TODO: We should attach the attributes defined in OMPKinds.def.
  }

  /// The underyling module.
  Module &M;

  /// The SCC we are operating on.
  SmallPtrSetImpl<Function *> &SCC;

  /// The slice of the module we are allowed to look at.
  SmallPtrSetImpl<Function *> &ModuleSlice;

  /// An OpenMP-IR-Builder instance
  OpenMPIRBuilder OMPBuilder;

  /// Callback to update the call graph, the first argument is a removed call,
  /// the second an optional replacement call.
  CallGraphUpdater &CGUpdater;

  /// Map from runtime function kind to the runtime function description.
  EnumeratedArray<RuntimeFunctionInfo, RuntimeFunction,
                  RuntimeFunction::OMPRTL___last>
      RFIs;
};
} // namespace

PreservedAnalyses OpenMPOptPass::run(LazyCallGraph::SCC &C,
                                     CGSCCAnalysisManager &AM,
                                     LazyCallGraph &CG, CGSCCUpdateResult &UR) {
  if (!containsOpenMP(*C.begin()->getFunction().getParent(), OMPInModule))
    return PreservedAnalyses::all();

  if (DisableOpenMPOptimizations)
    return PreservedAnalyses::all();

  SmallPtrSet<Function *, 16> SCC;
  for (LazyCallGraph::Node &N : C)
    SCC.insert(&N.getFunction());

  if (SCC.empty())
    return PreservedAnalyses::all();

  CallGraphUpdater CGUpdater;
  CGUpdater.initialize(CG, C, AM, UR);
  // TODO: Compute the module slice we are allowed to look at.
  OpenMPOpt OMPOpt(SCC, SCC, CGUpdater);
  bool Changed = OMPOpt.run();
  (void)Changed;
  return PreservedAnalyses::all();
}

namespace {

struct OpenMPOptLegacyPass : public CallGraphSCCPass {
  CallGraphUpdater CGUpdater;
  OpenMPInModule OMPInModule;
  static char ID;

  OpenMPOptLegacyPass() : CallGraphSCCPass(ID) {
    initializeOpenMPOptLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    CallGraphSCCPass::getAnalysisUsage(AU);
  }

  bool doInitialization(CallGraph &CG) override {
    // Disable the pass if there is no OpenMP (runtime call) in the module.
    containsOpenMP(CG.getModule(), OMPInModule);
    return false;
  }

  bool runOnSCC(CallGraphSCC &CGSCC) override {
    if (!containsOpenMP(CGSCC.getCallGraph().getModule(), OMPInModule))
      return false;
    if (DisableOpenMPOptimizations || skipSCC(CGSCC))
      return false;

    SmallPtrSet<Function *, 16> SCC;
    for (CallGraphNode *CGN : CGSCC)
      if (Function *Fn = CGN->getFunction())
        if (!Fn->isDeclaration())
          SCC.insert(Fn);

    if (SCC.empty())
      return false;

    CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
    CGUpdater.initialize(CG, CGSCC);

    // TODO: Compute the module slice we are allowed to look at.
    OpenMPOpt OMPOpt(SCC, SCC, CGUpdater);
    return OMPOpt.run();
  }

  bool doFinalization(CallGraph &CG) override { return CGUpdater.finalize(); }
};

} // end anonymous namespace

bool llvm::omp::containsOpenMP(Module &M, OpenMPInModule &OMPInModule) {
  if (OMPInModule.isKnown())
    return OMPInModule;

#define OMP_RTL(_Enum, _Name, ...)                                             \
  if (M.getFunction(_Name))                                                    \
    return OMPInModule = true;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  return OMPInModule = false;
}

char OpenMPOptLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(OpenMPOptLegacyPass, "openmpopt",
                      "OpenMP specific optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_END(OpenMPOptLegacyPass, "openmpopt",
                    "OpenMP specific optimizations", false, false)

Pass *llvm::createOpenMPOptLegacyPass() { return new OpenMPOptLegacyPass(); }
