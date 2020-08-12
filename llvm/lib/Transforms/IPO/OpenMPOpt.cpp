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
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/Utils/CallGraphUpdater.h"

using namespace llvm;
using namespace omp;

#define DEBUG_TYPE "openmp-opt"

static cl::opt<bool> DisableOpenMPOptimizations(
    "openmp-opt-disable", cl::ZeroOrMore,
    cl::desc("Disable OpenMP specific optimizations."), cl::Hidden,
    cl::init(false));

static cl::opt<bool> PrintICVValues("openmp-print-icv-values", cl::init(false),
                                    cl::Hidden);
static cl::opt<bool> PrintOpenMPKernels("openmp-print-gpu-kernels",
                                        cl::init(false), cl::Hidden);

static cl::opt<bool> HideMemoryTransferLatency(
    "openmp-hide-memory-transfer-latency",
    cl::desc("[WIP] Tries to hide the latency of host to device memory"
             " transfers"),
    cl::Hidden, cl::init(false));


STATISTIC(NumOpenMPRuntimeCallsDeduplicated,
          "Number of OpenMP runtime calls deduplicated");
STATISTIC(NumOpenMPParallelRegionsDeleted,
          "Number of OpenMP parallel regions deleted");
STATISTIC(NumOpenMPRuntimeFunctionsIdentified,
          "Number of OpenMP runtime functions identified");
STATISTIC(NumOpenMPRuntimeFunctionUsesIdentified,
          "Number of OpenMP runtime function uses identified");
STATISTIC(NumOpenMPTargetRegionKernels,
          "Number of OpenMP target region entry points (=kernels) identified");
STATISTIC(
    NumOpenMPParallelRegionsReplacedInGPUStateMachine,
    "Number of OpenMP parallel regions replaced with ID in GPU state machines");

#if !defined(NDEBUG)
static constexpr auto TAG = "[" DEBUG_TYPE "]";
#endif

/// Apply \p CB to all uses of \p F. If \p LookThroughConstantExprUses is
/// true, constant expression users are not given to \p CB but their uses are
/// traversed transitively.
template <typename CBTy>
static void foreachUse(Function &F, CBTy CB,
                       bool LookThroughConstantExprUses = true) {
  SmallVector<Use *, 8> Worklist(make_pointer_range(F.uses()));

  for (unsigned idx = 0; idx < Worklist.size(); ++idx) {
    Use &U = *Worklist[idx];

    // Allow use in constant bitcasts and simply look through them.
    if (LookThroughConstantExprUses && isa<ConstantExpr>(U.getUser())) {
      for (Use &CEU : cast<ConstantExpr>(U.getUser())->uses())
        Worklist.push_back(&CEU);
      continue;
    }

    CB(U);
  }
}

namespace {

struct AAICVTracker;

/// OpenMP specific information. For now, stores RFIs and ICVs also needed for
/// Attributor runs.
struct OMPInformationCache : public InformationCache {
  OMPInformationCache(Module &M, AnalysisGetter &AG,
                      BumpPtrAllocator &Allocator, SetVector<Function *> &CGSCC,
                      SmallPtrSetImpl<Kernel> &Kernels)
      : InformationCache(M, AG, Allocator, &CGSCC), OMPBuilder(M),
        Kernels(Kernels) {
    initializeModuleSlice(CGSCC);

    OMPBuilder.initialize();
    initializeRuntimeFunctions();
    initializeInternalControlVars();
  }

  /// Generic information that describes an internal control variable.
  struct InternalControlVarInfo {
    /// The kind, as described by InternalControlVar enum.
    InternalControlVar Kind;

    /// The name of the ICV.
    StringRef Name;

    /// Environment variable associated with this ICV.
    StringRef EnvVarName;

    /// Initial value kind.
    ICVInitValue InitKind;

    /// Initial value.
    ConstantInt *InitValue;

    /// Setter RTL function associated with this ICV.
    RuntimeFunction Setter;

    /// Getter RTL function associated with this ICV.
    RuntimeFunction Getter;

    /// RTL Function corresponding to the override clause of this ICV
    RuntimeFunction Clause;
  };

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
    Function *Declaration = nullptr;

    /// Uses of this runtime function per function containing the use.
    using UseVector = SmallVector<Use *, 16>;

    /// Clear UsesMap for runtime function.
    void clearUsesMap() { UsesMap.clear(); }

    /// Boolean conversion that is true if the runtime function was found.
    operator bool() const { return Declaration; }

    /// Return the vector of uses in function \p F.
    UseVector &getOrCreateUseVector(Function *F) {
      std::shared_ptr<UseVector> &UV = UsesMap[F];
      if (!UV)
        UV = std::make_shared<UseVector>();
      return *UV;
    }

    /// Return the vector of uses in function \p F or `nullptr` if there are
    /// none.
    const UseVector *getUseVector(Function &F) const {
      auto I = UsesMap.find(&F);
      if (I != UsesMap.end())
        return I->second.get();
      return nullptr;
    }

    /// Return how many functions contain uses of this runtime function.
    size_t getNumFunctionsWithUses() const { return UsesMap.size(); }

    /// Return the number of arguments (or the minimal number for variadic
    /// functions).
    size_t getNumArgs() const { return ArgumentTypes.size(); }

    /// Run the callback \p CB on each use and forget the use if the result is
    /// true. The callback will be fed the function in which the use was
    /// encountered as second argument.
    void foreachUse(SmallVectorImpl<Function *> &SCC,
                    function_ref<bool(Use &, Function &)> CB) {
      for (Function *F : SCC)
        foreachUse(CB, F);
    }

    /// Run the callback \p CB on each use within the function \p F and forget
    /// the use if the result is true.
    void foreachUse(function_ref<bool(Use &, Function &)> CB, Function *F) {
      SmallVector<unsigned, 8> ToBeDeleted;
      ToBeDeleted.clear();

      unsigned Idx = 0;
      UseVector &UV = getOrCreateUseVector(F);

      for (Use *U : UV) {
        if (CB(*U, *F))
          ToBeDeleted.push_back(Idx);
        ++Idx;
      }

      // Remove the to-be-deleted indices in reverse order as prior
      // modifications will not modify the smaller indices.
      while (!ToBeDeleted.empty()) {
        unsigned Idx = ToBeDeleted.pop_back_val();
        UV[Idx] = UV.back();
        UV.pop_back();
      }
    }

  private:
    /// Map from functions to all uses of this runtime function contained in
    /// them.
    DenseMap<Function *, std::shared_ptr<UseVector>> UsesMap;
  };

  /// Initialize the ModuleSlice member based on \p SCC. ModuleSlices contains
  /// (a subset of) all functions that we can look at during this SCC traversal.
  /// This includes functions (transitively) called from the SCC and the
  /// (transitive) callers of SCC functions. We also can look at a function if
  /// there is a "reference edge", i.a., if the function somehow uses (!=calls)
  /// a function in the SCC or a caller of a function in the SCC.
  void initializeModuleSlice(SetVector<Function *> &SCC) {
    ModuleSlice.insert(SCC.begin(), SCC.end());

    SmallPtrSet<Function *, 16> Seen;
    SmallVector<Function *, 16> Worklist(SCC.begin(), SCC.end());
    while (!Worklist.empty()) {
      Function *F = Worklist.pop_back_val();
      ModuleSlice.insert(F);

      for (Instruction &I : instructions(*F))
        if (auto *CB = dyn_cast<CallBase>(&I))
          if (Function *Callee = CB->getCalledFunction())
            if (Seen.insert(Callee).second)
              Worklist.push_back(Callee);
    }

    Seen.clear();
    Worklist.append(SCC.begin(), SCC.end());
    while (!Worklist.empty()) {
      Function *F = Worklist.pop_back_val();
      ModuleSlice.insert(F);

      // Traverse all transitive uses.
      foreachUse(*F, [&](Use &U) {
        if (auto *UsrI = dyn_cast<Instruction>(U.getUser()))
          if (Seen.insert(UsrI->getFunction()).second)
            Worklist.push_back(UsrI->getFunction());
      });
    }
  }

  /// The slice of the module we are allowed to look at.
  SmallPtrSet<Function *, 8> ModuleSlice;

  /// An OpenMP-IR-Builder instance
  OpenMPIRBuilder OMPBuilder;

  /// Map from runtime function kind to the runtime function description.
  EnumeratedArray<RuntimeFunctionInfo, RuntimeFunction,
                  RuntimeFunction::OMPRTL___last>
      RFIs;

  /// Map from ICV kind to the ICV description.
  EnumeratedArray<InternalControlVarInfo, InternalControlVar,
                  InternalControlVar::ICV___last>
      ICVs;

  /// Helper to initialize all internal control variable information for those
  /// defined in OMPKinds.def.
  void initializeInternalControlVars() {
#define ICV_RT_SET(_Name, RTL)                                                 \
  {                                                                            \
    auto &ICV = ICVs[_Name];                                                   \
    ICV.Setter = RTL;                                                          \
  }
#define ICV_RT_GET(Name, RTL)                                                  \
  {                                                                            \
    auto &ICV = ICVs[Name];                                                    \
    ICV.Getter = RTL;                                                          \
  }
#define ICV_DATA_ENV(Enum, _Name, _EnvVarName, Init)                           \
  {                                                                            \
    auto &ICV = ICVs[Enum];                                                    \
    ICV.Name = _Name;                                                          \
    ICV.Kind = Enum;                                                           \
    ICV.InitKind = Init;                                                       \
    ICV.EnvVarName = _EnvVarName;                                              \
    switch (ICV.InitKind) {                                                    \
    case ICV_IMPLEMENTATION_DEFINED:                                           \
      ICV.InitValue = nullptr;                                                 \
      break;                                                                   \
    case ICV_ZERO:                                                             \
      ICV.InitValue = ConstantInt::get(                                        \
          Type::getInt32Ty(OMPBuilder.Int32->getContext()), 0);                \
      break;                                                                   \
    case ICV_FALSE:                                                            \
      ICV.InitValue = ConstantInt::getFalse(OMPBuilder.Int1->getContext());    \
      break;                                                                   \
    case ICV_LAST:                                                             \
      break;                                                                   \
    }                                                                          \
  }
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }

  /// Returns true if the function declaration \p F matches the runtime
  /// function types, that is, return type \p RTFRetType, and argument types
  /// \p RTFArgTypes.
  static bool declMatchesRTFTypes(Function *F, Type *RTFRetType,
                                  SmallVector<Type *, 8> &RTFArgTypes) {
    // TODO: We should output information to the user (under debug output
    //       and via remarks).

    if (!F)
      return false;
    if (F->getReturnType() != RTFRetType)
      return false;
    if (F->arg_size() != RTFArgTypes.size())
      return false;

    auto RTFTyIt = RTFArgTypes.begin();
    for (Argument &Arg : F->args()) {
      if (Arg.getType() != *RTFTyIt)
        return false;

      ++RTFTyIt;
    }

    return true;
  }

  // Helper to collect all uses of the declaration in the UsesMap.
  unsigned collectUses(RuntimeFunctionInfo &RFI, bool CollectStats = true) {
    unsigned NumUses = 0;
    if (!RFI.Declaration)
      return NumUses;
    OMPBuilder.addAttributes(RFI.Kind, *RFI.Declaration);

    if (CollectStats) {
      NumOpenMPRuntimeFunctionsIdentified += 1;
      NumOpenMPRuntimeFunctionUsesIdentified += RFI.Declaration->getNumUses();
    }

    // TODO: We directly convert uses into proper calls and unknown uses.
    for (Use &U : RFI.Declaration->uses()) {
      if (Instruction *UserI = dyn_cast<Instruction>(U.getUser())) {
        if (ModuleSlice.count(UserI->getFunction())) {
          RFI.getOrCreateUseVector(UserI->getFunction()).push_back(&U);
          ++NumUses;
        }
      } else {
        RFI.getOrCreateUseVector(nullptr).push_back(&U);
        ++NumUses;
      }
    }
    return NumUses;
  }

  // Helper function to recollect uses of all runtime functions.
  void recollectUses() {
    for (int Idx = 0; Idx < RFIs.size(); ++Idx) {
      auto &RFI = RFIs[static_cast<RuntimeFunction>(Idx)];
      RFI.clearUsesMap();
      collectUses(RFI, /*CollectStats*/ false);
    }
  }

  /// Helper to initialize all runtime function information for those defined
  /// in OpenMPKinds.def.
  void initializeRuntimeFunctions() {
    Module &M = *((*ModuleSlice.begin())->getParent());

    // Helper macros for handling __VA_ARGS__ in OMP_RTL
#define OMP_TYPE(VarName, ...)                                                 \
  Type *VarName = OMPBuilder.VarName;                                          \
  (void)VarName;

#define OMP_ARRAY_TYPE(VarName, ...)                                           \
  ArrayType *VarName##Ty = OMPBuilder.VarName##Ty;                             \
  (void)VarName##Ty;                                                           \
  PointerType *VarName##PtrTy = OMPBuilder.VarName##PtrTy;                     \
  (void)VarName##PtrTy;

#define OMP_FUNCTION_TYPE(VarName, ...)                                        \
  FunctionType *VarName = OMPBuilder.VarName;                                  \
  (void)VarName;                                                               \
  PointerType *VarName##Ptr = OMPBuilder.VarName##Ptr;                         \
  (void)VarName##Ptr;

#define OMP_STRUCT_TYPE(VarName, ...)                                          \
  StructType *VarName = OMPBuilder.VarName;                                    \
  (void)VarName;                                                               \
  PointerType *VarName##Ptr = OMPBuilder.VarName##Ptr;                         \
  (void)VarName##Ptr;

#define OMP_RTL(_Enum, _Name, _IsVarArg, _ReturnType, ...)                     \
  {                                                                            \
    SmallVector<Type *, 8> ArgsTypes({__VA_ARGS__});                           \
    Function *F = M.getFunction(_Name);                                        \
    if (declMatchesRTFTypes(F, OMPBuilder._ReturnType, ArgsTypes)) {           \
      auto &RFI = RFIs[_Enum];                                                 \
      RFI.Kind = _Enum;                                                        \
      RFI.Name = _Name;                                                        \
      RFI.IsVarArg = _IsVarArg;                                                \
      RFI.ReturnType = OMPBuilder._ReturnType;                                 \
      RFI.ArgumentTypes = std::move(ArgsTypes);                                \
      RFI.Declaration = F;                                                     \
      unsigned NumUses = collectUses(RFI);                                     \
      (void)NumUses;                                                           \
      LLVM_DEBUG({                                                             \
        dbgs() << TAG << RFI.Name << (RFI.Declaration ? "" : " not")           \
               << " found\n";                                                  \
        if (RFI.Declaration)                                                   \
          dbgs() << TAG << "-> got " << NumUses << " uses in "                 \
                 << RFI.getNumFunctionsWithUses()                              \
                 << " different functions.\n";                                 \
      });                                                                      \
    }                                                                          \
  }
#include "llvm/Frontend/OpenMP/OMPKinds.def"

    // TODO: We should attach the attributes defined in OMPKinds.def.
  }

  /// Collection of known kernels (\see Kernel) in the module.
  SmallPtrSetImpl<Kernel> &Kernels;
};

struct OpenMPOpt {

  using OptimizationRemarkGetter =
      function_ref<OptimizationRemarkEmitter &(Function *)>;

  OpenMPOpt(SmallVectorImpl<Function *> &SCC, CallGraphUpdater &CGUpdater,
            OptimizationRemarkGetter OREGetter,
            OMPInformationCache &OMPInfoCache, Attributor &A)
      : M(*(*SCC.begin())->getParent()), SCC(SCC), CGUpdater(CGUpdater),
        OREGetter(OREGetter), OMPInfoCache(OMPInfoCache), A(A) {}

  /// Run all OpenMP optimizations on the underlying SCC/ModuleSlice.
  bool run() {
    if (SCC.empty())
      return false;

    bool Changed = false;

    LLVM_DEBUG(dbgs() << TAG << "Run on SCC with " << SCC.size()
                      << " functions in a slice with "
                      << OMPInfoCache.ModuleSlice.size() << " functions\n");

    if (PrintICVValues)
      printICVs();
    if (PrintOpenMPKernels)
      printKernels();

    Changed |= rewriteDeviceCodeStateMachine();

    Changed |= runAttributor();

    // Recollect uses, in case Attributor deleted any.
    OMPInfoCache.recollectUses();

    Changed |= deduplicateRuntimeCalls();
    Changed |= deleteParallelRegions();
    if (HideMemoryTransferLatency)
      Changed |= hideMemTransfersLatency();

    return Changed;
  }

  /// Print initial ICV values for testing.
  /// FIXME: This should be done from the Attributor once it is added.
  void printICVs() const {
    InternalControlVar ICVs[] = {ICV_nthreads, ICV_active_levels, ICV_cancel};

    for (Function *F : OMPInfoCache.ModuleSlice) {
      for (auto ICV : ICVs) {
        auto ICVInfo = OMPInfoCache.ICVs[ICV];
        auto Remark = [&](OptimizationRemark OR) {
          return OR << "OpenMP ICV " << ore::NV("OpenMPICV", ICVInfo.Name)
                    << " Value: "
                    << (ICVInfo.InitValue
                            ? ICVInfo.InitValue->getValue().toString(10, true)
                            : "IMPLEMENTATION_DEFINED");
        };

        emitRemarkOnFunction(F, "OpenMPICVTracker", Remark);
      }
    }
  }

  /// Print OpenMP GPU kernels for testing.
  void printKernels() const {
    for (Function *F : SCC) {
      if (!OMPInfoCache.Kernels.count(F))
        continue;

      auto Remark = [&](OptimizationRemark OR) {
        return OR << "OpenMP GPU kernel "
                  << ore::NV("OpenMPGPUKernel", F->getName()) << "\n";
      };

      emitRemarkOnFunction(F, "OpenMPGPU", Remark);
    }
  }

  /// Return the call if \p U is a callee use in a regular call. If \p RFI is
  /// given it has to be the callee or a nullptr is returned.
  static CallInst *getCallIfRegularCall(
      Use &U, OMPInformationCache::RuntimeFunctionInfo *RFI = nullptr) {
    CallInst *CI = dyn_cast<CallInst>(U.getUser());
    if (CI && CI->isCallee(&U) && !CI->hasOperandBundles() &&
        (!RFI || CI->getCalledFunction() == RFI->Declaration))
      return CI;
    return nullptr;
  }

  /// Return the call if \p V is a regular call. If \p RFI is given it has to be
  /// the callee or a nullptr is returned.
  static CallInst *getCallIfRegularCall(
      Value &V, OMPInformationCache::RuntimeFunctionInfo *RFI = nullptr) {
    CallInst *CI = dyn_cast<CallInst>(&V);
    if (CI && !CI->hasOperandBundles() &&
        (!RFI || CI->getCalledFunction() == RFI->Declaration))
      return CI;
    return nullptr;
  }

private:
  /// Try to delete parallel regions if possible.
  bool deleteParallelRegions() {
    const unsigned CallbackCalleeOperand = 2;

    OMPInformationCache::RuntimeFunctionInfo &RFI =
        OMPInfoCache.RFIs[OMPRTL___kmpc_fork_call];

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

      auto Remark = [&](OptimizationRemark OR) {
        return OR << "Parallel region in "
                  << ore::NV("OpenMPParallelDelete", CI->getCaller()->getName())
                  << " deleted";
      };
      emitRemark<OptimizationRemark>(CI, "OpenMPParallelRegionDeletion",
                                     Remark);

      CGUpdater.removeCallSite(*CI);
      CI->eraseFromParent();
      Changed = true;
      ++NumOpenMPParallelRegionsDeleted;
      return true;
    };

    RFI.foreachUse(SCC, DeleteCallCB);

    return Changed;
  }

  /// Try to eliminate runtime calls by reusing existing ones.
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

    // Global-tid is handled separately.
    SmallSetVector<Value *, 16> GTIdArgs;
    collectGlobalThreadIdArguments(GTIdArgs);
    LLVM_DEBUG(dbgs() << TAG << "Found " << GTIdArgs.size()
                      << " global thread ID arguments\n");

    for (Function *F : SCC) {
      for (auto DeduplicableRuntimeCallID : DeduplicableRuntimeCallIDs)
        deduplicateRuntimeCalls(*F,
                                OMPInfoCache.RFIs[DeduplicableRuntimeCallID]);

      // __kmpc_global_thread_num is special as we can replace it with an
      // argument in enough cases to make it worth trying.
      Value *GTIdArg = nullptr;
      for (Argument &Arg : F->args())
        if (GTIdArgs.count(&Arg)) {
          GTIdArg = &Arg;
          break;
        }
      Changed |= deduplicateRuntimeCalls(
          *F, OMPInfoCache.RFIs[OMPRTL___kmpc_global_thread_num], GTIdArg);
    }

    return Changed;
  }

  /// Tries to hide the latency of runtime calls that involve host to
  /// device memory transfers by splitting them into their "issue" and "wait"
  /// versions. The "issue" is moved upwards as much as possible. The "wait" is
  /// moved downards as much as possible. The "issue" issues the memory transfer
  /// asynchronously, returning a handle. The "wait" waits in the returned
  /// handle for the memory transfer to finish.
  bool hideMemTransfersLatency() {
    auto &RFI = OMPInfoCache.RFIs[OMPRTL___tgt_target_data_begin_mapper];
    bool Changed = false;
    auto SplitMemTransfers = [&](Use &U, Function &Decl) {
      auto *RTCall = getCallIfRegularCall(U, &RFI);
      if (!RTCall)
        return false;

      bool WasSplit = splitTargetDataBeginRTC(RTCall);
      Changed |= WasSplit;
      return WasSplit;
    };
    RFI.foreachUse(SCC, SplitMemTransfers);

    return Changed;
  }

  /// Splits \p RuntimeCall into its "issue" and "wait" counterparts.
  bool splitTargetDataBeginRTC(CallInst *RuntimeCall) {
    auto &IRBuilder = OMPInfoCache.OMPBuilder;
    // Add "issue" runtime call declaration:
    // declare %struct.tgt_async_info @__tgt_target_data_begin_issue(i64, i32,
    //   i8**, i8**, i64*, i64*)
    FunctionCallee IssueDecl = IRBuilder.getOrCreateRuntimeFunction(
        M, OMPRTL___tgt_target_data_begin_mapper_issue);

    // Change RuntimeCall call site for its asynchronous version.
    SmallVector<Value *, 8> Args;
    for (auto &Arg : RuntimeCall->args())
      Args.push_back(Arg.get());

    CallInst *IssueCallsite =
        CallInst::Create(IssueDecl, Args, "handle", RuntimeCall);
    RuntimeCall->eraseFromParent();

    // Add "wait" runtime call declaration:
    // declare void @__tgt_target_data_begin_wait(i64, %struct.__tgt_async_info)
    FunctionCallee WaitDecl = IRBuilder.getOrCreateRuntimeFunction(
        M, OMPRTL___tgt_target_data_begin_mapper_wait);

    // Add call site to WaitDecl.
    Value *WaitParams[2] = {
        IssueCallsite->getArgOperand(0), // device_id.
        IssueCallsite // returned handle.
    };
    CallInst::Create(WaitDecl, WaitParams, /*NameStr=*/"",
                     IssueCallsite->getNextNode());

    return true;
  }

  static Value *combinedIdentStruct(Value *CurrentIdent, Value *NextIdent,
                                    bool GlobalOnly, bool &SingleChoice) {
    if (CurrentIdent == NextIdent)
      return CurrentIdent;

    // TODO: Figure out how to actually combine multiple debug locations. For
    //       now we just keep an existing one if there is a single choice.
    if (!GlobalOnly || isa<GlobalValue>(NextIdent)) {
      SingleChoice = !CurrentIdent;
      return NextIdent;
    }
    return nullptr;
  }

  /// Return an `struct ident_t*` value that represents the ones used in the
  /// calls of \p RFI inside of \p F. If \p GlobalOnly is true, we will not
  /// return a local `struct ident_t*`. For now, if we cannot find a suitable
  /// return value we create one from scratch. We also do not yet combine
  /// information, e.g., the source locations, see combinedIdentStruct.
  Value *
  getCombinedIdentFromCallUsesIn(OMPInformationCache::RuntimeFunctionInfo &RFI,
                                 Function &F, bool GlobalOnly) {
    bool SingleChoice = true;
    Value *Ident = nullptr;
    auto CombineIdentStruct = [&](Use &U, Function &Caller) {
      CallInst *CI = getCallIfRegularCall(U, &RFI);
      if (!CI || &F != &Caller)
        return false;
      Ident = combinedIdentStruct(Ident, CI->getArgOperand(0),
                                  /* GlobalOnly */ true, SingleChoice);
      return false;
    };
    RFI.foreachUse(SCC, CombineIdentStruct);

    if (!Ident || !SingleChoice) {
      // The IRBuilder uses the insertion block to get to the module, this is
      // unfortunate but we work around it for now.
      if (!OMPInfoCache.OMPBuilder.getInsertionPoint().getBlock())
        OMPInfoCache.OMPBuilder.updateToLocation(OpenMPIRBuilder::InsertPointTy(
            &F.getEntryBlock(), F.getEntryBlock().begin()));
      // Create a fallback location if non was found.
      // TODO: Use the debug locations of the calls instead.
      Constant *Loc = OMPInfoCache.OMPBuilder.getOrCreateDefaultSrcLocStr();
      Ident = OMPInfoCache.OMPBuilder.getOrCreateIdent(Loc);
    }
    return Ident;
  }

  /// Try to eliminate calls of \p RFI in \p F by reusing an existing one or
  /// \p ReplVal if given.
  bool deduplicateRuntimeCalls(Function &F,
                               OMPInformationCache::RuntimeFunctionInfo &RFI,
                               Value *ReplVal = nullptr) {
    auto *UV = RFI.getUseVector(F);
    if (!UV || UV->size() + (ReplVal != nullptr) < 2)
      return false;

    LLVM_DEBUG(
        dbgs() << TAG << "Deduplicate " << UV->size() << " uses of " << RFI.Name
               << (ReplVal ? " with an existing value\n" : "\n") << "\n");

    assert((!ReplVal || (isa<Argument>(ReplVal) &&
                         cast<Argument>(ReplVal)->getParent() == &F)) &&
           "Unexpected replacement value!");

    // TODO: Use dominance to find a good position instead.
    auto CanBeMoved = [this](CallBase &CB) {
      unsigned NumArgs = CB.getNumArgOperands();
      if (NumArgs == 0)
        return true;
      if (CB.getArgOperand(0)->getType() != OMPInfoCache.OMPBuilder.IdentPtr)
        return false;
      for (unsigned u = 1; u < NumArgs; ++u)
        if (isa<Instruction>(CB.getArgOperand(u)))
          return false;
      return true;
    };

    if (!ReplVal) {
      for (Use *U : *UV)
        if (CallInst *CI = getCallIfRegularCall(*U, &RFI)) {
          if (!CanBeMoved(*CI))
            continue;

          auto Remark = [&](OptimizationRemark OR) {
            auto newLoc = &*F.getEntryBlock().getFirstInsertionPt();
            return OR << "OpenMP runtime call "
                      << ore::NV("OpenMPOptRuntime", RFI.Name) << " moved to "
                      << ore::NV("OpenMPRuntimeMoves", newLoc->getDebugLoc());
          };
          emitRemark<OptimizationRemark>(CI, "OpenMPRuntimeCodeMotion", Remark);

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
          CI->getArgOperand(0)->getType() == OMPInfoCache.OMPBuilder.IdentPtr) {
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

      auto Remark = [&](OptimizationRemark OR) {
        return OR << "OpenMP runtime call "
                  << ore::NV("OpenMPOptRuntime", RFI.Name) << " deduplicated";
      };
      emitRemark<OptimizationRemark>(CI, "OpenMPRuntimeDeduplicated", Remark);

      CGUpdater.removeCallSite(*CI);
      CI->replaceAllUsesWith(ReplVal);
      CI->eraseFromParent();
      ++NumOpenMPRuntimeCallsDeduplicated;
      Changed = true;
      return true;
    };
    RFI.foreachUse(SCC, ReplaceAndDeleteCB);

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
              getCallIfRegularCall(
                  *ArgOp, &OMPInfoCache.RFIs[OMPRTL___kmpc_global_thread_num]))
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
    OMPInformationCache::RuntimeFunctionInfo &GlobThreadNumRFI =
        OMPInfoCache.RFIs[OMPRTL___kmpc_global_thread_num];

    GlobThreadNumRFI.foreachUse(SCC, [&](Use &U, Function &F) {
      if (CallInst *CI = getCallIfRegularCall(U, &GlobThreadNumRFI))
        AddUserArgs(*CI);
      return false;
    });

    // Transitively search for more arguments by looking at the users of the
    // ones we know already. During the search the GTIdArgs vector is extended
    // so we cannot cache the size nor can we use a range based for.
    for (unsigned u = 0; u < GTIdArgs.size(); ++u)
      AddUserArgs(*GTIdArgs[u]);
  }

  /// Kernel (=GPU) optimizations and utility functions
  ///
  ///{{

  /// Check if \p F is a kernel, hence entry point for target offloading.
  bool isKernel(Function &F) { return OMPInfoCache.Kernels.count(&F); }

  /// Cache to remember the unique kernel for a function.
  DenseMap<Function *, Optional<Kernel>> UniqueKernelMap;

  /// Find the unique kernel that will execute \p F, if any.
  Kernel getUniqueKernelFor(Function &F);

  /// Find the unique kernel that will execute \p I, if any.
  Kernel getUniqueKernelFor(Instruction &I) {
    return getUniqueKernelFor(*I.getFunction());
  }

  /// Rewrite the device (=GPU) code state machine create in non-SPMD mode in
  /// the cases we can avoid taking the address of a function.
  bool rewriteDeviceCodeStateMachine();

  ///
  ///}}

  /// Emit a remark generically
  ///
  /// This template function can be used to generically emit a remark. The
  /// RemarkKind should be one of the following:
  ///   - OptimizationRemark to indicate a successful optimization attempt
  ///   - OptimizationRemarkMissed to report a failed optimization attempt
  ///   - OptimizationRemarkAnalysis to provide additional information about an
  ///     optimization attempt
  ///
  /// The remark is built using a callback function provided by the caller that
  /// takes a RemarkKind as input and returns a RemarkKind.
  template <typename RemarkKind,
            typename RemarkCallBack = function_ref<RemarkKind(RemarkKind &&)>>
  void emitRemark(Instruction *Inst, StringRef RemarkName,
                  RemarkCallBack &&RemarkCB) const {
    Function *F = Inst->getParent()->getParent();
    auto &ORE = OREGetter(F);

    ORE.emit(
        [&]() { return RemarkCB(RemarkKind(DEBUG_TYPE, RemarkName, Inst)); });
  }

  /// Emit a remark on a function. Since only OptimizationRemark is supporting
  /// this, it can't be made generic.
  void
  emitRemarkOnFunction(Function *F, StringRef RemarkName,
                       function_ref<OptimizationRemark(OptimizationRemark &&)>
                           &&RemarkCB) const {
    auto &ORE = OREGetter(F);

    ORE.emit([&]() {
      return RemarkCB(OptimizationRemark(DEBUG_TYPE, RemarkName, F));
    });
  }

  /// The underlying module.
  Module &M;

  /// The SCC we are operating on.
  SmallVectorImpl<Function *> &SCC;

  /// Callback to update the call graph, the first argument is a removed call,
  /// the second an optional replacement call.
  CallGraphUpdater &CGUpdater;

  /// Callback to get an OptimizationRemarkEmitter from a Function *
  OptimizationRemarkGetter OREGetter;

  /// OpenMP-specific information cache. Also Used for Attributor runs.
  OMPInformationCache &OMPInfoCache;

  /// Attributor instance.
  Attributor &A;

  /// Helper function to run Attributor on SCC.
  bool runAttributor() {
    if (SCC.empty())
      return false;

    registerAAs();

    ChangeStatus Changed = A.run();

    LLVM_DEBUG(dbgs() << "[Attributor] Done with " << SCC.size()
                      << " functions, result: " << Changed << ".\n");

    return Changed == ChangeStatus::CHANGED;
  }

  /// Populate the Attributor with abstract attribute opportunities in the
  /// function.
  void registerAAs() {
    if (SCC.empty())
      return;

    // Create CallSite AA for all Getters.
    for (int Idx = 0; Idx < OMPInfoCache.ICVs.size() - 1; ++Idx) {
      auto ICVInfo = OMPInfoCache.ICVs[static_cast<InternalControlVar>(Idx)];

      auto &GetterRFI = OMPInfoCache.RFIs[ICVInfo.Getter];

      auto CreateAA = [&](Use &U, Function &Caller) {
        CallInst *CI = OpenMPOpt::getCallIfRegularCall(U, &GetterRFI);
        if (!CI)
          return false;

        auto &CB = cast<CallBase>(*CI);

        IRPosition CBPos = IRPosition::callsite_function(CB);
        A.getOrCreateAAFor<AAICVTracker>(CBPos);
        return false;
      };

      GetterRFI.foreachUse(SCC, CreateAA);
    }
  }
};

Kernel OpenMPOpt::getUniqueKernelFor(Function &F) {
  if (!OMPInfoCache.ModuleSlice.count(&F))
    return nullptr;

  // Use a scope to keep the lifetime of the CachedKernel short.
  {
    Optional<Kernel> &CachedKernel = UniqueKernelMap[&F];
    if (CachedKernel)
      return *CachedKernel;

    // TODO: We should use an AA to create an (optimistic and callback
    //       call-aware) call graph. For now we stick to simple patterns that
    //       are less powerful, basically the worst fixpoint.
    if (isKernel(F)) {
      CachedKernel = Kernel(&F);
      return *CachedKernel;
    }

    CachedKernel = nullptr;
    if (!F.hasLocalLinkage())
      return nullptr;
  }

  auto GetUniqueKernelForUse = [&](const Use &U) -> Kernel {
    if (auto *Cmp = dyn_cast<ICmpInst>(U.getUser())) {
      // Allow use in equality comparisons.
      if (Cmp->isEquality())
        return getUniqueKernelFor(*Cmp);
      return nullptr;
    }
    if (auto *CB = dyn_cast<CallBase>(U.getUser())) {
      // Allow direct calls.
      if (CB->isCallee(&U))
        return getUniqueKernelFor(*CB);
      // Allow the use in __kmpc_kernel_prepare_parallel calls.
      if (Function *Callee = CB->getCalledFunction())
        if (Callee->getName() == "__kmpc_kernel_prepare_parallel")
          return getUniqueKernelFor(*CB);
      return nullptr;
    }
    // Disallow every other use.
    return nullptr;
  };

  // TODO: In the future we want to track more than just a unique kernel.
  SmallPtrSet<Kernel, 2> PotentialKernels;
  foreachUse(F, [&](const Use &U) {
    PotentialKernels.insert(GetUniqueKernelForUse(U));
  });

  Kernel K = nullptr;
  if (PotentialKernels.size() == 1)
    K = *PotentialKernels.begin();

  // Cache the result.
  UniqueKernelMap[&F] = K;

  return K;
}

bool OpenMPOpt::rewriteDeviceCodeStateMachine() {
  OMPInformationCache::RuntimeFunctionInfo &KernelPrepareParallelRFI =
      OMPInfoCache.RFIs[OMPRTL___kmpc_kernel_prepare_parallel];

  bool Changed = false;
  if (!KernelPrepareParallelRFI)
    return Changed;

  for (Function *F : SCC) {

    // Check if the function is uses in a __kmpc_kernel_prepare_parallel call at
    // all.
    bool UnknownUse = false;
    bool KernelPrepareUse = false;
    unsigned NumDirectCalls = 0;

    SmallVector<Use *, 2> ToBeReplacedStateMachineUses;
    foreachUse(*F, [&](Use &U) {
      if (auto *CB = dyn_cast<CallBase>(U.getUser()))
        if (CB->isCallee(&U)) {
          ++NumDirectCalls;
          return;
        }

      if (isa<ICmpInst>(U.getUser())) {
        ToBeReplacedStateMachineUses.push_back(&U);
        return;
      }
      if (!KernelPrepareUse && OpenMPOpt::getCallIfRegularCall(
                                   *U.getUser(), &KernelPrepareParallelRFI)) {
        KernelPrepareUse = true;
        ToBeReplacedStateMachineUses.push_back(&U);
        return;
      }
      UnknownUse = true;
    });

    // Do not emit a remark if we haven't seen a __kmpc_kernel_prepare_parallel
    // use.
    if (!KernelPrepareUse)
      continue;

    {
      auto Remark = [&](OptimizationRemark OR) {
        return OR << "Found a parallel region that is called in a target "
                     "region but not part of a combined target construct nor "
                     "nesed inside a target construct without intermediate "
                     "code. This can lead to excessive register usage for "
                     "unrelated target regions in the same translation unit "
                     "due to spurious call edges assumed by ptxas.";
      };
      emitRemarkOnFunction(F, "OpenMPParallelRegionInNonSPMD", Remark);
    }

    // If this ever hits, we should investigate.
    // TODO: Checking the number of uses is not a necessary restriction and
    // should be lifted.
    if (UnknownUse || NumDirectCalls != 1 ||
        ToBeReplacedStateMachineUses.size() != 2) {
      {
        auto Remark = [&](OptimizationRemark OR) {
          return OR << "Parallel region is used in "
                    << (UnknownUse ? "unknown" : "unexpected")
                    << " ways; will not attempt to rewrite the state machine.";
        };
        emitRemarkOnFunction(F, "OpenMPParallelRegionInNonSPMD", Remark);
      }
      continue;
    }

    // Even if we have __kmpc_kernel_prepare_parallel calls, we (for now) give
    // up if the function is not called from a unique kernel.
    Kernel K = getUniqueKernelFor(*F);
    if (!K) {
      {
        auto Remark = [&](OptimizationRemark OR) {
          return OR << "Parallel region is not known to be called from a "
                       "unique single target region, maybe the surrounding "
                       "function has external linkage?; will not attempt to "
                       "rewrite the state machine use.";
        };
        emitRemarkOnFunction(F, "OpenMPParallelRegionInMultipleKernesl",
                             Remark);
      }
      continue;
    }

    // We now know F is a parallel body function called only from the kernel K.
    // We also identified the state machine uses in which we replace the
    // function pointer by a new global symbol for identification purposes. This
    // ensures only direct calls to the function are left.

    {
      auto RemarkParalleRegion = [&](OptimizationRemark OR) {
        return OR << "Specialize parallel region that is only reached from a "
                     "single target region to avoid spurious call edges and "
                     "excessive register usage in other target regions. "
                     "(parallel region ID: "
                  << ore::NV("OpenMPParallelRegion", F->getName())
                  << ", kernel ID: "
                  << ore::NV("OpenMPTargetRegion", K->getName()) << ")";
      };
      emitRemarkOnFunction(F, "OpenMPParallelRegionInNonSPMD",
                           RemarkParalleRegion);
      auto RemarkKernel = [&](OptimizationRemark OR) {
        return OR << "Target region containing the parallel region that is "
                     "specialized. (parallel region ID: "
                  << ore::NV("OpenMPParallelRegion", F->getName())
                  << ", kernel ID: "
                  << ore::NV("OpenMPTargetRegion", K->getName()) << ")";
      };
      emitRemarkOnFunction(K, "OpenMPParallelRegionInNonSPMD", RemarkKernel);
    }

    Module &M = *F->getParent();
    Type *Int8Ty = Type::getInt8Ty(M.getContext());

    auto *ID = new GlobalVariable(
        M, Int8Ty, /* isConstant */ true, GlobalValue::PrivateLinkage,
        UndefValue::get(Int8Ty), F->getName() + ".ID");

    for (Use *U : ToBeReplacedStateMachineUses)
      U->set(ConstantExpr::getBitCast(ID, U->get()->getType()));

    ++NumOpenMPParallelRegionsReplacedInGPUStateMachine;

    Changed = true;
  }

  return Changed;
}

/// Abstract Attribute for tracking ICV values.
struct AAICVTracker : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;
  AAICVTracker(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  void initialize(Attributor &A) override {
    Function *F = getAnchorScope();
    if (!F || !A.isFunctionIPOAmendable(*F))
      indicatePessimisticFixpoint();
  }

  /// Returns true if value is assumed to be tracked.
  bool isAssumedTracked() const { return getAssumed(); }

  /// Returns true if value is known to be tracked.
  bool isKnownTracked() const { return getAssumed(); }

  /// Create an abstract attribute biew for the position \p IRP.
  static AAICVTracker &createForPosition(const IRPosition &IRP, Attributor &A);

  /// Return the value with which \p I can be replaced for specific \p ICV.
  virtual Optional<Value *> getReplacementValue(InternalControlVar ICV,
                                                const Instruction *I,
                                                Attributor &A) const {
    return None;
  }

  /// Return an assumed unique ICV value if a single candidate is found. If
  /// there cannot be one, return a nullptr. If it is not clear yet, return the
  /// Optional::NoneType.
  virtual Optional<Value *>
  getUniqueReplacementValue(InternalControlVar ICV) const = 0;

  // Currently only nthreads is being tracked.
  // this array will only grow with time.
  InternalControlVar TrackableICVs[1] = {ICV_nthreads};

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAICVTracker"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is AAICVTracker
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  static const char ID;
};

struct AAICVTrackerFunction : public AAICVTracker {
  AAICVTrackerFunction(const IRPosition &IRP, Attributor &A)
      : AAICVTracker(IRP, A) {}

  // FIXME: come up with better string.
  const std::string getAsStr() const override { return "ICVTrackerFunction"; }

  // FIXME: come up with some stats.
  void trackStatistics() const override {}

  /// We don't manifest anything for this AA.
  ChangeStatus manifest(Attributor &A) override {
    return ChangeStatus::UNCHANGED;
  }

  // Map of ICV to their values at specific program point.
  EnumeratedArray<DenseMap<Instruction *, Value *>, InternalControlVar,
                  InternalControlVar::ICV___last>
      ICVReplacementValuesMap;

  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus HasChanged = ChangeStatus::UNCHANGED;

    Function *F = getAnchorScope();

    auto &OMPInfoCache = static_cast<OMPInformationCache &>(A.getInfoCache());

    for (InternalControlVar ICV : TrackableICVs) {
      auto &SetterRFI = OMPInfoCache.RFIs[OMPInfoCache.ICVs[ICV].Setter];

      auto &ValuesMap = ICVReplacementValuesMap[ICV];
      auto TrackValues = [&](Use &U, Function &) {
        CallInst *CI = OpenMPOpt::getCallIfRegularCall(U);
        if (!CI)
          return false;

        // FIXME: handle setters with more that 1 arguments.
        /// Track new value.
        if (ValuesMap.insert(std::make_pair(CI, CI->getArgOperand(0))).second)
          HasChanged = ChangeStatus::CHANGED;

        return false;
      };

      auto CallCheck = [&](Instruction &I) {
        Optional<Value *> ReplVal = getValueForCall(A, &I, ICV);
        if (ReplVal.hasValue() &&
            ValuesMap.insert(std::make_pair(&I, *ReplVal)).second)
          HasChanged = ChangeStatus::CHANGED;

        return true;
      };

      // Track all changes of an ICV.
      SetterRFI.foreachUse(TrackValues, F);

      A.checkForAllInstructions(CallCheck, *this, {Instruction::Call},
                                /* CheckBBLivenessOnly */ true);

      /// TODO: Figure out a way to avoid adding entry in
      /// ICVReplacementValuesMap
      Instruction *Entry = &F->getEntryBlock().front();
      if (HasChanged == ChangeStatus::CHANGED && !ValuesMap.count(Entry))
        ValuesMap.insert(std::make_pair(Entry, nullptr));
    }

    return HasChanged;
  }

  /// Hepler to check if \p I is a call and get the value for it if it is
  /// unique.
  Optional<Value *> getValueForCall(Attributor &A, const Instruction *I,
                                    InternalControlVar &ICV) const {

    const auto *CB = dyn_cast<CallBase>(I);
    if (!CB)
      return None;

    auto &OMPInfoCache = static_cast<OMPInformationCache &>(A.getInfoCache());
    auto &GetterRFI = OMPInfoCache.RFIs[OMPInfoCache.ICVs[ICV].Getter];
    auto &SetterRFI = OMPInfoCache.RFIs[OMPInfoCache.ICVs[ICV].Setter];
    Function *CalledFunction = CB->getCalledFunction();

    if (CalledFunction == GetterRFI.Declaration)
      return None;
    if (CalledFunction == SetterRFI.Declaration) {
      if (ICVReplacementValuesMap[ICV].count(I))
        return ICVReplacementValuesMap[ICV].lookup(I);

      return nullptr;
    }

    // Since we don't know, assume it changes the ICV.
    if (CalledFunction->isDeclaration())
      return nullptr;

    const auto &ICVTrackingAA =
        A.getAAFor<AAICVTracker>(*this, IRPosition::callsite_returned(*CB));

    if (ICVTrackingAA.isAssumedTracked())
      return ICVTrackingAA.getUniqueReplacementValue(ICV);

    // If we don't know, assume it changes.
    return nullptr;
  }

  // We don't check unique value for a function, so return None.
  Optional<Value *>
  getUniqueReplacementValue(InternalControlVar ICV) const override {
    return None;
  }

  /// Return the value with which \p I can be replaced for specific \p ICV.
  Optional<Value *> getReplacementValue(InternalControlVar ICV,
                                        const Instruction *I,
                                        Attributor &A) const override {
    const auto &ValuesMap = ICVReplacementValuesMap[ICV];
    if (ValuesMap.count(I))
      return ValuesMap.lookup(I);

    SmallVector<const Instruction *, 16> Worklist;
    SmallPtrSet<const Instruction *, 16> Visited;
    Worklist.push_back(I);

    Optional<Value *> ReplVal;

    while (!Worklist.empty()) {
      const Instruction *CurrInst = Worklist.pop_back_val();
      if (!Visited.insert(CurrInst).second)
        continue;

      const BasicBlock *CurrBB = CurrInst->getParent();

      // Go up and look for all potential setters/calls that might change the
      // ICV.
      while ((CurrInst = CurrInst->getPrevNode())) {
        if (ValuesMap.count(CurrInst)) {
          Optional<Value *> NewReplVal = ValuesMap.lookup(CurrInst);
          // Unknown value, track new.
          if (!ReplVal.hasValue()) {
            ReplVal = NewReplVal;
            break;
          }

          // If we found a new value, we can't know the icv value anymore.
          if (NewReplVal.hasValue())
            if (ReplVal != NewReplVal)
              return nullptr;

          break;
        }

        Optional<Value *> NewReplVal = getValueForCall(A, CurrInst, ICV);
        if (!NewReplVal.hasValue())
          continue;

        // Unknown value, track new.
        if (!ReplVal.hasValue()) {
          ReplVal = NewReplVal;
          break;
        }

        // if (NewReplVal.hasValue())
        // We found a new value, we can't know the icv value anymore.
        if (ReplVal != NewReplVal)
          return nullptr;
      }

      // If we are in the same BB and we have a value, we are done.
      if (CurrBB == I->getParent() && ReplVal.hasValue())
        return ReplVal;

      // Go through all predecessors and add terminators for analysis.
      for (const BasicBlock *Pred : predecessors(CurrBB))
        if (const Instruction *Terminator = Pred->getTerminator())
          Worklist.push_back(Terminator);
    }

    return ReplVal;
  }
};

struct AAICVTrackerFunctionReturned : AAICVTracker {
  AAICVTrackerFunctionReturned(const IRPosition &IRP, Attributor &A)
      : AAICVTracker(IRP, A) {}

  // FIXME: come up with better string.
  const std::string getAsStr() const override {
    return "ICVTrackerFunctionReturned";
  }

  // FIXME: come up with some stats.
  void trackStatistics() const override {}

  /// We don't manifest anything for this AA.
  ChangeStatus manifest(Attributor &A) override {
    return ChangeStatus::UNCHANGED;
  }

  // Map of ICV to their values at specific program point.
  EnumeratedArray<Optional<Value *>, InternalControlVar,
                  InternalControlVar::ICV___last>
      ICVReplacementValuesMap;

  /// Return the value with which \p I can be replaced for specific \p ICV.
  Optional<Value *>
  getUniqueReplacementValue(InternalControlVar ICV) const override {
    return ICVReplacementValuesMap[ICV];
  }

  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    const auto &ICVTrackingAA = A.getAAFor<AAICVTracker>(
        *this, IRPosition::function(*getAnchorScope()));

    if (!ICVTrackingAA.isAssumedTracked())
      return indicatePessimisticFixpoint();

    for (InternalControlVar ICV : TrackableICVs) {
      Optional<Value *> &ReplVal = ICVReplacementValuesMap[ICV];
      Optional<Value *> UniqueICVValue;

      auto CheckReturnInst = [&](Instruction &I) {
        Optional<Value *> NewReplVal =
            ICVTrackingAA.getReplacementValue(ICV, &I, A);

        // If we found a second ICV value there is no unique returned value.
        if (UniqueICVValue.hasValue() && UniqueICVValue != NewReplVal)
          return false;

        UniqueICVValue = NewReplVal;

        return true;
      };

      if (!A.checkForAllInstructions(CheckReturnInst, *this, {Instruction::Ret},
                                     /* CheckBBLivenessOnly */ true))
        UniqueICVValue = nullptr;

      if (UniqueICVValue == ReplVal)
        continue;

      ReplVal = UniqueICVValue;
      Changed = ChangeStatus::CHANGED;
    }

    return Changed;
  }
};

struct AAICVTrackerCallSite : AAICVTracker {
  AAICVTrackerCallSite(const IRPosition &IRP, Attributor &A)
      : AAICVTracker(IRP, A) {}

  void initialize(Attributor &A) override {
    Function *F = getAnchorScope();
    if (!F || !A.isFunctionIPOAmendable(*F))
      indicatePessimisticFixpoint();

    // We only initialize this AA for getters, so we need to know which ICV it
    // gets.
    auto &OMPInfoCache = static_cast<OMPInformationCache &>(A.getInfoCache());
    for (InternalControlVar ICV : TrackableICVs) {
      auto ICVInfo = OMPInfoCache.ICVs[ICV];
      auto &Getter = OMPInfoCache.RFIs[ICVInfo.Getter];
      if (Getter.Declaration == getAssociatedFunction()) {
        AssociatedICV = ICVInfo.Kind;
        return;
      }
    }

    /// Unknown ICV.
    indicatePessimisticFixpoint();
  }

  ChangeStatus manifest(Attributor &A) override {
    if (!ReplVal.hasValue() || !ReplVal.getValue())
      return ChangeStatus::UNCHANGED;

    A.changeValueAfterManifest(*getCtxI(), **ReplVal);
    A.deleteAfterManifest(*getCtxI());

    return ChangeStatus::CHANGED;
  }

  // FIXME: come up with better string.
  const std::string getAsStr() const override { return "ICVTrackerCallSite"; }

  // FIXME: come up with some stats.
  void trackStatistics() const override {}

  InternalControlVar AssociatedICV;
  Optional<Value *> ReplVal;

  ChangeStatus updateImpl(Attributor &A) override {
    const auto &ICVTrackingAA = A.getAAFor<AAICVTracker>(
        *this, IRPosition::function(*getAnchorScope()));

    // We don't have any information, so we assume it changes the ICV.
    if (!ICVTrackingAA.isAssumedTracked())
      return indicatePessimisticFixpoint();

    Optional<Value *> NewReplVal =
        ICVTrackingAA.getReplacementValue(AssociatedICV, getCtxI(), A);

    if (ReplVal == NewReplVal)
      return ChangeStatus::UNCHANGED;

    ReplVal = NewReplVal;
    return ChangeStatus::CHANGED;
  }

  // Return the value with which associated value can be replaced for specific
  // \p ICV.
  Optional<Value *>
  getUniqueReplacementValue(InternalControlVar ICV) const override {
    return ReplVal;
  }
};

struct AAICVTrackerCallSiteReturned : AAICVTracker {
  AAICVTrackerCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AAICVTracker(IRP, A) {}

  // FIXME: come up with better string.
  const std::string getAsStr() const override {
    return "ICVTrackerCallSiteReturned";
  }

  // FIXME: come up with some stats.
  void trackStatistics() const override {}

  /// We don't manifest anything for this AA.
  ChangeStatus manifest(Attributor &A) override {
    return ChangeStatus::UNCHANGED;
  }

  // Map of ICV to their values at specific program point.
  EnumeratedArray<Optional<Value *>, InternalControlVar,
                  InternalControlVar::ICV___last>
      ICVReplacementValuesMap;

  /// Return the value with which associated value can be replaced for specific
  /// \p ICV.
  Optional<Value *>
  getUniqueReplacementValue(InternalControlVar ICV) const override {
    return ICVReplacementValuesMap[ICV];
  }

  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    const auto &ICVTrackingAA = A.getAAFor<AAICVTracker>(
        *this, IRPosition::returned(*getAssociatedFunction()));

    // We don't have any information, so we assume it changes the ICV.
    if (!ICVTrackingAA.isAssumedTracked())
      return indicatePessimisticFixpoint();

    for (InternalControlVar ICV : TrackableICVs) {
      Optional<Value *> &ReplVal = ICVReplacementValuesMap[ICV];
      Optional<Value *> NewReplVal =
          ICVTrackingAA.getUniqueReplacementValue(ICV);

      if (ReplVal == NewReplVal)
        continue;

      ReplVal = NewReplVal;
      Changed = ChangeStatus::CHANGED;
    }
    return Changed;
  }
};
} // namespace

const char AAICVTracker::ID = 0;

AAICVTracker &AAICVTracker::createForPosition(const IRPosition &IRP,
                                              Attributor &A) {
  AAICVTracker *AA = nullptr;
  switch (IRP.getPositionKind()) {
  case IRPosition::IRP_INVALID:
  case IRPosition::IRP_FLOAT:
  case IRPosition::IRP_ARGUMENT:
  case IRPosition::IRP_CALL_SITE_ARGUMENT:
    llvm_unreachable("ICVTracker: invalid IRPosition!");
  case IRPosition::IRP_FUNCTION:
    AA = new (A.Allocator) AAICVTrackerFunction(IRP, A);
    break;
  case IRPosition::IRP_RETURNED:
    AA = new (A.Allocator) AAICVTrackerFunctionReturned(IRP, A);
    break;
  case IRPosition::IRP_CALL_SITE_RETURNED:
    AA = new (A.Allocator) AAICVTrackerCallSiteReturned(IRP, A);
    break;
  case IRPosition::IRP_CALL_SITE:
    AA = new (A.Allocator) AAICVTrackerCallSite(IRP, A);
    break;
  }

  return *AA;
}

PreservedAnalyses OpenMPOptPass::run(LazyCallGraph::SCC &C,
                                     CGSCCAnalysisManager &AM,
                                     LazyCallGraph &CG, CGSCCUpdateResult &UR) {
  if (!containsOpenMP(*C.begin()->getFunction().getParent(), OMPInModule))
    return PreservedAnalyses::all();

  if (DisableOpenMPOptimizations)
    return PreservedAnalyses::all();

  SmallVector<Function *, 16> SCC;
  // If there are kernels in the module, we have to run on all SCC's.
  bool SCCIsInteresting = !OMPInModule.getKernels().empty();
  for (LazyCallGraph::Node &N : C) {
    Function *Fn = &N.getFunction();
    SCC.push_back(Fn);

    // Do we already know that the SCC contains kernels,
    // or that OpenMP functions are called from this SCC?
    if (SCCIsInteresting)
      continue;
    // If not, let's check that.
    SCCIsInteresting |= OMPInModule.containsOMPRuntimeCalls(Fn);
  }

  if (!SCCIsInteresting || SCC.empty())
    return PreservedAnalyses::all();

  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();

  AnalysisGetter AG(FAM);

  auto OREGetter = [&FAM](Function *F) -> OptimizationRemarkEmitter & {
    return FAM.getResult<OptimizationRemarkEmitterAnalysis>(*F);
  };

  CallGraphUpdater CGUpdater;
  CGUpdater.initialize(CG, C, AM, UR);

  SetVector<Function *> Functions(SCC.begin(), SCC.end());
  BumpPtrAllocator Allocator;
  OMPInformationCache InfoCache(*(Functions.back()->getParent()), AG, Allocator,
                                /*CGSCC*/ Functions, OMPInModule.getKernels());

  SetVector<Function *> ModuleSlice(InfoCache.ModuleSlice.begin(),
                                    InfoCache.ModuleSlice.end());
  Attributor A(ModuleSlice, InfoCache, CGUpdater);

  // TODO: Compute the module slice we are allowed to look at.
  OpenMPOpt OMPOpt(SCC, CGUpdater, OREGetter, InfoCache, A);
  bool Changed = OMPOpt.run();
  if (Changed)
    return PreservedAnalyses::none();

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

    SmallVector<Function *, 16> SCC;
    // If there are kernels in the module, we have to run on all SCC's.
    bool SCCIsInteresting = !OMPInModule.getKernels().empty();
    for (CallGraphNode *CGN : CGSCC) {
      Function *Fn = CGN->getFunction();
      if (!Fn || Fn->isDeclaration())
        continue;
      SCC.push_back(Fn);

      // Do we already know that the SCC contains kernels,
      // or that OpenMP functions are called from this SCC?
      if (SCCIsInteresting)
        continue;
      // If not, let's check that.
      SCCIsInteresting |= OMPInModule.containsOMPRuntimeCalls(Fn);
    }

    if (!SCCIsInteresting || SCC.empty())
      return false;

    CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
    CGUpdater.initialize(CG, CGSCC);

    // Maintain a map of functions to avoid rebuilding the ORE
    DenseMap<Function *, std::unique_ptr<OptimizationRemarkEmitter>> OREMap;
    auto OREGetter = [&OREMap](Function *F) -> OptimizationRemarkEmitter & {
      std::unique_ptr<OptimizationRemarkEmitter> &ORE = OREMap[F];
      if (!ORE)
        ORE = std::make_unique<OptimizationRemarkEmitter>(F);
      return *ORE;
    };

    AnalysisGetter AG;
    SetVector<Function *> Functions(SCC.begin(), SCC.end());
    BumpPtrAllocator Allocator;
    OMPInformationCache InfoCache(
        *(Functions.back()->getParent()), AG, Allocator,
        /*CGSCC*/ Functions, OMPInModule.getKernels());

    SetVector<Function *> ModuleSlice(InfoCache.ModuleSlice.begin(),
                                      InfoCache.ModuleSlice.end());
    Attributor A(ModuleSlice, InfoCache, CGUpdater);

    // TODO: Compute the module slice we are allowed to look at.
    OpenMPOpt OMPOpt(SCC, CGUpdater, OREGetter, InfoCache, A);
    return OMPOpt.run();
  }

  bool doFinalization(CallGraph &CG) override { return CGUpdater.finalize(); }
};

} // end anonymous namespace

void OpenMPInModule::identifyKernels(Module &M) {

  NamedMDNode *MD = M.getOrInsertNamedMetadata("nvvm.annotations");
  if (!MD)
    return;

  for (auto *Op : MD->operands()) {
    if (Op->getNumOperands() < 2)
      continue;
    MDString *KindID = dyn_cast<MDString>(Op->getOperand(1));
    if (!KindID || KindID->getString() != "kernel")
      continue;

    Function *KernelFn =
        mdconst::dyn_extract_or_null<Function>(Op->getOperand(0));
    if (!KernelFn)
      continue;

    ++NumOpenMPTargetRegionKernels;

    Kernels.insert(KernelFn);
  }
}

bool llvm::omp::containsOpenMP(Module &M, OpenMPInModule &OMPInModule) {
  if (OMPInModule.isKnown())
    return OMPInModule;

  auto RecordFunctionsContainingUsesOf = [&](Function *F) {
    for (User *U : F->users())
      if (auto *I = dyn_cast<Instruction>(U))
        OMPInModule.FuncsWithOMPRuntimeCalls.insert(I->getFunction());
  };

  // MSVC doesn't like long if-else chains for some reason and instead just
  // issues an error. Work around it..
  do {
#define OMP_RTL(_Enum, _Name, ...)                                             \
  if (Function *F = M.getFunction(_Name)) {                                    \
    RecordFunctionsContainingUsesOf(F);                                        \
    OMPInModule = true;                                                        \
  }
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  } while (false);

  // Identify kernels once. TODO: We should split the OMPInformationCache into a
  // module and an SCC part. The kernel information, among other things, could
  // go into the module part.
  if (OMPInModule.isKnown() && OMPInModule) {
    OMPInModule.identifyKernels(M);
    return true;
  }

  return OMPInModule = false;
}

char OpenMPOptLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(OpenMPOptLegacyPass, "openmpopt",
                      "OpenMP specific optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_END(OpenMPOptLegacyPass, "openmpopt",
                    "OpenMP specific optimizations", false, false)

Pass *llvm::createOpenMPOptLegacyPass() { return new OpenMPOptLegacyPass(); }
