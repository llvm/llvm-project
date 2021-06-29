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
// - Replacing globalized device memory with stack memory.
// - Replacing globalized device memory with shared memory.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/OpenMPOpt.h"

#include "llvm/ADT/EnumeratedArray.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CallGraphUpdater.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

using namespace llvm::PatternMatch;
using namespace llvm;
using namespace omp;

#define DEBUG_TYPE "openmp-opt"

static cl::opt<bool> DisableOpenMPOptimizations(
    "openmp-opt-disable", cl::ZeroOrMore,
    cl::desc("Disable OpenMP specific optimizations."), cl::Hidden,
    cl::init(false));

static cl::opt<bool> EnableParallelRegionMerging(
    "openmp-opt-enable-merging", cl::ZeroOrMore,
    cl::desc("Enable the OpenMP region merging optimization."), cl::Hidden,
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
STATISTIC(NumOpenMPParallelRegionsMerged,
          "Number of OpenMP parallel regions merged");
STATISTIC(NumBytesMovedToSharedMemory,
          "Amount of memory pushed to shared memory");

#if !defined(NDEBUG)
static constexpr auto TAG = "[" DEBUG_TYPE "]";
#endif

namespace {

enum class AddressSpace : unsigned {
  Generic = 0,
  Global = 1,
  Shared = 3,
  Constant = 4,
  Local = 5,
};

struct AAHeapToShared;

struct AAICVTracker;

/// OpenMP specific information. For now, stores RFIs and ICVs also needed for
/// Attributor runs.
struct OMPInformationCache : public InformationCache {
  OMPInformationCache(Module &M, AnalysisGetter &AG,
                      BumpPtrAllocator &Allocator, SetVector<Function *> &CGSCC,
                      SmallPtrSetImpl<Kernel> &Kernels)
      : InformationCache(M, AG, Allocator, &CGSCC), OMPBuilder(M),
        Kernels(Kernels) {

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

  // Helper function to recollect uses of a runtime function.
  void recollectUsesForFunction(RuntimeFunction RTF) {
    auto &RFI = RFIs[RTF];
    RFI.clearUsesMap();
    collectUses(RFI, /*CollectStats*/ false);
  }

  // Helper function to recollect uses of all runtime functions.
  void recollectUses() {
    for (int Idx = 0; Idx < RFIs.size(); ++Idx)
      recollectUsesForFunction(static_cast<RuntimeFunction>(Idx));
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

/// Used to map the values physically (in the IR) stored in an offload
/// array, to a vector in memory.
struct OffloadArray {
  /// Physical array (in the IR).
  AllocaInst *Array = nullptr;
  /// Mapped values.
  SmallVector<Value *, 8> StoredValues;
  /// Last stores made in the offload array.
  SmallVector<StoreInst *, 8> LastAccesses;

  OffloadArray() = default;

  /// Initializes the OffloadArray with the values stored in \p Array before
  /// instruction \p Before is reached. Returns false if the initialization
  /// fails.
  /// This MUST be used immediately after the construction of the object.
  bool initialize(AllocaInst &Array, Instruction &Before) {
    if (!Array.getAllocatedType()->isArrayTy())
      return false;

    if (!getValues(Array, Before))
      return false;

    this->Array = &Array;
    return true;
  }

  static const unsigned DeviceIDArgNum = 1;
  static const unsigned BasePtrsArgNum = 3;
  static const unsigned PtrsArgNum = 4;
  static const unsigned SizesArgNum = 5;

private:
  /// Traverses the BasicBlock where \p Array is, collecting the stores made to
  /// \p Array, leaving StoredValues with the values stored before the
  /// instruction \p Before is reached.
  bool getValues(AllocaInst &Array, Instruction &Before) {
    // Initialize container.
    const uint64_t NumValues = Array.getAllocatedType()->getArrayNumElements();
    StoredValues.assign(NumValues, nullptr);
    LastAccesses.assign(NumValues, nullptr);

    // TODO: This assumes the instruction \p Before is in the same
    //  BasicBlock as Array. Make it general, for any control flow graph.
    BasicBlock *BB = Array.getParent();
    if (BB != Before.getParent())
      return false;

    const DataLayout &DL = Array.getModule()->getDataLayout();
    const unsigned int PointerSize = DL.getPointerSize();

    for (Instruction &I : *BB) {
      if (&I == &Before)
        break;

      if (!isa<StoreInst>(&I))
        continue;

      auto *S = cast<StoreInst>(&I);
      int64_t Offset = -1;
      auto *Dst =
          GetPointerBaseWithConstantOffset(S->getPointerOperand(), Offset, DL);
      if (Dst == &Array) {
        int64_t Idx = Offset / PointerSize;
        StoredValues[Idx] = getUnderlyingObject(S->getValueOperand());
        LastAccesses[Idx] = S;
      }
    }

    return isFilled();
  }

  /// Returns true if all values in StoredValues and
  /// LastAccesses are not nullptrs.
  bool isFilled() {
    const unsigned NumValues = StoredValues.size();
    for (unsigned I = 0; I < NumValues; ++I) {
      if (!StoredValues[I] || !LastAccesses[I])
        return false;
    }

    return true;
  }
};

struct OpenMPOpt {

  using OptimizationRemarkGetter =
      function_ref<OptimizationRemarkEmitter &(Function *)>;

  OpenMPOpt(SmallVectorImpl<Function *> &SCC, CallGraphUpdater &CGUpdater,
            OptimizationRemarkGetter OREGetter,
            OMPInformationCache &OMPInfoCache, Attributor &A)
      : M(*(*SCC.begin())->getParent()), SCC(SCC), CGUpdater(CGUpdater),
        OREGetter(OREGetter), OMPInfoCache(OMPInfoCache), A(A) {}

  /// Check if any remarks are enabled for openmp-opt
  bool remarksEnabled() {
    auto &Ctx = M.getContext();
    return Ctx.getDiagHandlerPtr()->isAnyRemarkEnabled(DEBUG_TYPE);
  }

  /// Run all OpenMP optimizations on the underlying SCC/ModuleSlice.
  bool run(bool IsModulePass) {
    if (SCC.empty())
      return false;

    bool Changed = false;

    LLVM_DEBUG(dbgs() << TAG << "Run on SCC with " << SCC.size()
                      << " functions in a slice with "
                      << OMPInfoCache.ModuleSlice.size() << " functions\n");

    if (IsModulePass) {
      Changed |= runAttributor();

      // Recollect uses, in case Attributor deleted any.
      OMPInfoCache.recollectUses();

      if (remarksEnabled())
        analysisGlobalization();
    } else {
      if (PrintICVValues)
        printICVs();
      if (PrintOpenMPKernels)
        printKernels();

      Changed |= rewriteDeviceCodeStateMachine();

      Changed |= runAttributor();

      // Recollect uses, in case Attributor deleted any.
      OMPInfoCache.recollectUses();

      Changed |= deleteParallelRegions();
      if (HideMemoryTransferLatency)
        Changed |= hideMemTransfersLatency();
      Changed |= deduplicateRuntimeCalls();
      if (EnableParallelRegionMerging) {
        if (mergeParallelRegions()) {
          deduplicateRuntimeCalls();
          Changed = true;
        }
      }
    }

    return Changed;
  }

  /// Print initial ICV values for testing.
  /// FIXME: This should be done from the Attributor once it is added.
  void printICVs() const {
    InternalControlVar ICVs[] = {ICV_nthreads, ICV_active_levels, ICV_cancel,
                                 ICV_proc_bind};

    for (Function *F : OMPInfoCache.ModuleSlice) {
      for (auto ICV : ICVs) {
        auto ICVInfo = OMPInfoCache.ICVs[ICV];
        auto Remark = [&](OptimizationRemarkAnalysis ORA) {
          return ORA << "OpenMP ICV " << ore::NV("OpenMPICV", ICVInfo.Name)
                     << " Value: "
                     << (ICVInfo.InitValue
                             ? toString(ICVInfo.InitValue->getValue(), 10, true)
                             : "IMPLEMENTATION_DEFINED");
        };

        emitRemark<OptimizationRemarkAnalysis>(F, "OpenMPICVTracker", Remark);
      }
    }
  }

  /// Print OpenMP GPU kernels for testing.
  void printKernels() const {
    for (Function *F : SCC) {
      if (!OMPInfoCache.Kernels.count(F))
        continue;

      auto Remark = [&](OptimizationRemarkAnalysis ORA) {
        return ORA << "OpenMP GPU kernel "
                   << ore::NV("OpenMPGPUKernel", F->getName()) << "\n";
      };

      emitRemark<OptimizationRemarkAnalysis>(F, "OpenMPGPU", Remark);
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
  /// Merge parallel regions when it is safe.
  bool mergeParallelRegions() {
    const unsigned CallbackCalleeOperand = 2;
    const unsigned CallbackFirstArgOperand = 3;
    using InsertPointTy = OpenMPIRBuilder::InsertPointTy;

    // Check if there are any __kmpc_fork_call calls to merge.
    OMPInformationCache::RuntimeFunctionInfo &RFI =
        OMPInfoCache.RFIs[OMPRTL___kmpc_fork_call];

    if (!RFI.Declaration)
      return false;

    // Unmergable calls that prevent merging a parallel region.
    OMPInformationCache::RuntimeFunctionInfo UnmergableCallsInfo[] = {
        OMPInfoCache.RFIs[OMPRTL___kmpc_push_proc_bind],
        OMPInfoCache.RFIs[OMPRTL___kmpc_push_num_threads],
    };

    bool Changed = false;
    LoopInfo *LI = nullptr;
    DominatorTree *DT = nullptr;

    SmallDenseMap<BasicBlock *, SmallPtrSet<Instruction *, 4>> BB2PRMap;

    BasicBlock *StartBB = nullptr, *EndBB = nullptr;
    auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                         BasicBlock &ContinuationIP) {
      BasicBlock *CGStartBB = CodeGenIP.getBlock();
      BasicBlock *CGEndBB =
          SplitBlock(CGStartBB, &*CodeGenIP.getPoint(), DT, LI);
      assert(StartBB != nullptr && "StartBB should not be null");
      CGStartBB->getTerminator()->setSuccessor(0, StartBB);
      assert(EndBB != nullptr && "EndBB should not be null");
      EndBB->getTerminator()->setSuccessor(0, CGEndBB);
    };

    auto PrivCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP, Value &,
                      Value &Inner, Value *&ReplacementValue) -> InsertPointTy {
      ReplacementValue = &Inner;
      return CodeGenIP;
    };

    auto FiniCB = [&](InsertPointTy CodeGenIP) {};

    /// Create a sequential execution region within a merged parallel region,
    /// encapsulated in a master construct with a barrier for synchronization.
    auto CreateSequentialRegion = [&](Function *OuterFn,
                                      BasicBlock *OuterPredBB,
                                      Instruction *SeqStartI,
                                      Instruction *SeqEndI) {
      // Isolate the instructions of the sequential region to a separate
      // block.
      BasicBlock *ParentBB = SeqStartI->getParent();
      BasicBlock *SeqEndBB =
          SplitBlock(ParentBB, SeqEndI->getNextNode(), DT, LI);
      BasicBlock *SeqAfterBB =
          SplitBlock(SeqEndBB, &*SeqEndBB->getFirstInsertionPt(), DT, LI);
      BasicBlock *SeqStartBB =
          SplitBlock(ParentBB, SeqStartI, DT, LI, nullptr, "seq.par.merged");

      assert(ParentBB->getUniqueSuccessor() == SeqStartBB &&
             "Expected a different CFG");
      const DebugLoc DL = ParentBB->getTerminator()->getDebugLoc();
      ParentBB->getTerminator()->eraseFromParent();

      auto BodyGenCB = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP,
                           BasicBlock &ContinuationIP) {
        BasicBlock *CGStartBB = CodeGenIP.getBlock();
        BasicBlock *CGEndBB =
            SplitBlock(CGStartBB, &*CodeGenIP.getPoint(), DT, LI);
        assert(SeqStartBB != nullptr && "SeqStartBB should not be null");
        CGStartBB->getTerminator()->setSuccessor(0, SeqStartBB);
        assert(SeqEndBB != nullptr && "SeqEndBB should not be null");
        SeqEndBB->getTerminator()->setSuccessor(0, CGEndBB);
      };
      auto FiniCB = [&](InsertPointTy CodeGenIP) {};

      // Find outputs from the sequential region to outside users and
      // broadcast their values to them.
      for (Instruction &I : *SeqStartBB) {
        SmallPtrSet<Instruction *, 4> OutsideUsers;
        for (User *Usr : I.users()) {
          Instruction &UsrI = *cast<Instruction>(Usr);
          // Ignore outputs to LT intrinsics, code extraction for the merged
          // parallel region will fix them.
          if (UsrI.isLifetimeStartOrEnd())
            continue;

          if (UsrI.getParent() != SeqStartBB)
            OutsideUsers.insert(&UsrI);
        }

        if (OutsideUsers.empty())
          continue;

        // Emit an alloca in the outer region to store the broadcasted
        // value.
        const DataLayout &DL = M.getDataLayout();
        AllocaInst *AllocaI = new AllocaInst(
            I.getType(), DL.getAllocaAddrSpace(), nullptr,
            I.getName() + ".seq.output.alloc", &OuterFn->front().front());

        // Emit a store instruction in the sequential BB to update the
        // value.
        new StoreInst(&I, AllocaI, SeqStartBB->getTerminator());

        // Emit a load instruction and replace the use of the output value
        // with it.
        for (Instruction *UsrI : OutsideUsers) {
          LoadInst *LoadI = new LoadInst(
              I.getType(), AllocaI, I.getName() + ".seq.output.load", UsrI);
          UsrI->replaceUsesOfWith(&I, LoadI);
        }
      }

      OpenMPIRBuilder::LocationDescription Loc(
          InsertPointTy(ParentBB, ParentBB->end()), DL);
      InsertPointTy SeqAfterIP =
          OMPInfoCache.OMPBuilder.createMaster(Loc, BodyGenCB, FiniCB);

      OMPInfoCache.OMPBuilder.createBarrier(SeqAfterIP, OMPD_parallel);

      BranchInst::Create(SeqAfterBB, SeqAfterIP.getBlock());

      LLVM_DEBUG(dbgs() << TAG << "After sequential inlining " << *OuterFn
                        << "\n");
    };

    // Helper to merge the __kmpc_fork_call calls in MergableCIs. They are all
    // contained in BB and only separated by instructions that can be
    // redundantly executed in parallel. The block BB is split before the first
    // call (in MergableCIs) and after the last so the entire region we merge
    // into a single parallel region is contained in a single basic block
    // without any other instructions. We use the OpenMPIRBuilder to outline
    // that block and call the resulting function via __kmpc_fork_call.
    auto Merge = [&](SmallVectorImpl<CallInst *> &MergableCIs, BasicBlock *BB) {
      // TODO: Change the interface to allow single CIs expanded, e.g, to
      // include an outer loop.
      assert(MergableCIs.size() > 1 && "Assumed multiple mergable CIs");

      auto Remark = [&](OptimizationRemark OR) {
        OR << "Parallel region at "
           << ore::NV("OpenMPParallelMergeFront",
                      MergableCIs.front()->getDebugLoc())
           << " merged with parallel regions at ";
        for (auto *CI : llvm::drop_begin(MergableCIs)) {
          OR << ore::NV("OpenMPParallelMerge", CI->getDebugLoc());
          if (CI != MergableCIs.back())
            OR << ", ";
        }
        return OR;
      };

      emitRemark<OptimizationRemark>(MergableCIs.front(),
                                     "OpenMPParallelRegionMerging", Remark);

      Function *OriginalFn = BB->getParent();
      LLVM_DEBUG(dbgs() << TAG << "Merge " << MergableCIs.size()
                        << " parallel regions in " << OriginalFn->getName()
                        << "\n");

      // Isolate the calls to merge in a separate block.
      EndBB = SplitBlock(BB, MergableCIs.back()->getNextNode(), DT, LI);
      BasicBlock *AfterBB =
          SplitBlock(EndBB, &*EndBB->getFirstInsertionPt(), DT, LI);
      StartBB = SplitBlock(BB, MergableCIs.front(), DT, LI, nullptr,
                           "omp.par.merged");

      assert(BB->getUniqueSuccessor() == StartBB && "Expected a different CFG");
      const DebugLoc DL = BB->getTerminator()->getDebugLoc();
      BB->getTerminator()->eraseFromParent();

      // Create sequential regions for sequential instructions that are
      // in-between mergable parallel regions.
      for (auto *It = MergableCIs.begin(), *End = MergableCIs.end() - 1;
           It != End; ++It) {
        Instruction *ForkCI = *It;
        Instruction *NextForkCI = *(It + 1);

        // Continue if there are not in-between instructions.
        if (ForkCI->getNextNode() == NextForkCI)
          continue;

        CreateSequentialRegion(OriginalFn, BB, ForkCI->getNextNode(),
                               NextForkCI->getPrevNode());
      }

      OpenMPIRBuilder::LocationDescription Loc(InsertPointTy(BB, BB->end()),
                                               DL);
      IRBuilder<>::InsertPoint AllocaIP(
          &OriginalFn->getEntryBlock(),
          OriginalFn->getEntryBlock().getFirstInsertionPt());
      // Create the merged parallel region with default proc binding, to
      // avoid overriding binding settings, and without explicit cancellation.
      InsertPointTy AfterIP = OMPInfoCache.OMPBuilder.createParallel(
          Loc, AllocaIP, BodyGenCB, PrivCB, FiniCB, nullptr, nullptr,
          OMP_PROC_BIND_default, /* IsCancellable */ false);
      BranchInst::Create(AfterBB, AfterIP.getBlock());

      // Perform the actual outlining.
      OMPInfoCache.OMPBuilder.finalize(OriginalFn,
                                       /* AllowExtractorSinking */ true);

      Function *OutlinedFn = MergableCIs.front()->getCaller();

      // Replace the __kmpc_fork_call calls with direct calls to the outlined
      // callbacks.
      SmallVector<Value *, 8> Args;
      for (auto *CI : MergableCIs) {
        Value *Callee =
            CI->getArgOperand(CallbackCalleeOperand)->stripPointerCasts();
        FunctionType *FT =
            cast<FunctionType>(Callee->getType()->getPointerElementType());
        Args.clear();
        Args.push_back(OutlinedFn->getArg(0));
        Args.push_back(OutlinedFn->getArg(1));
        for (unsigned U = CallbackFirstArgOperand, E = CI->getNumArgOperands();
             U < E; ++U)
          Args.push_back(CI->getArgOperand(U));

        CallInst *NewCI = CallInst::Create(FT, Callee, Args, "", CI);
        if (CI->getDebugLoc())
          NewCI->setDebugLoc(CI->getDebugLoc());

        // Forward parameter attributes from the callback to the callee.
        for (unsigned U = CallbackFirstArgOperand, E = CI->getNumArgOperands();
             U < E; ++U)
          for (const Attribute &A : CI->getAttributes().getParamAttributes(U))
            NewCI->addParamAttr(
                U - (CallbackFirstArgOperand - CallbackCalleeOperand), A);

        // Emit an explicit barrier to replace the implicit fork-join barrier.
        if (CI != MergableCIs.back()) {
          // TODO: Remove barrier if the merged parallel region includes the
          // 'nowait' clause.
          OMPInfoCache.OMPBuilder.createBarrier(
              InsertPointTy(NewCI->getParent(),
                            NewCI->getNextNode()->getIterator()),
              OMPD_parallel);
        }

        auto Remark = [&](OptimizationRemark OR) {
          return OR << "Parallel region at "
                    << ore::NV("OpenMPParallelMerge", CI->getDebugLoc())
                    << " merged with "
                    << ore::NV("OpenMPParallelMergeFront",
                               MergableCIs.front()->getDebugLoc());
        };
        if (CI != MergableCIs.front())
          emitRemark<OptimizationRemark>(CI, "OpenMPParallelRegionMerging",
                                         Remark);

        CI->eraseFromParent();
      }

      assert(OutlinedFn != OriginalFn && "Outlining failed");
      CGUpdater.registerOutlinedFunction(*OriginalFn, *OutlinedFn);
      CGUpdater.reanalyzeFunction(*OriginalFn);

      NumOpenMPParallelRegionsMerged += MergableCIs.size();

      return true;
    };

    // Helper function that identifes sequences of
    // __kmpc_fork_call uses in a basic block.
    auto DetectPRsCB = [&](Use &U, Function &F) {
      CallInst *CI = getCallIfRegularCall(U, &RFI);
      BB2PRMap[CI->getParent()].insert(CI);

      return false;
    };

    BB2PRMap.clear();
    RFI.foreachUse(SCC, DetectPRsCB);
    SmallVector<SmallVector<CallInst *, 4>, 4> MergableCIsVector;
    // Find mergable parallel regions within a basic block that are
    // safe to merge, that is any in-between instructions can safely
    // execute in parallel after merging.
    // TODO: support merging across basic-blocks.
    for (auto &It : BB2PRMap) {
      auto &CIs = It.getSecond();
      if (CIs.size() < 2)
        continue;

      BasicBlock *BB = It.getFirst();
      SmallVector<CallInst *, 4> MergableCIs;

      /// Returns true if the instruction is mergable, false otherwise.
      /// A terminator instruction is unmergable by definition since merging
      /// works within a BB. Instructions before the mergable region are
      /// mergable if they are not calls to OpenMP runtime functions that may
      /// set different execution parameters for subsequent parallel regions.
      /// Instructions in-between parallel regions are mergable if they are not
      /// calls to any non-intrinsic function since that may call a non-mergable
      /// OpenMP runtime function.
      auto IsMergable = [&](Instruction &I, bool IsBeforeMergableRegion) {
        // We do not merge across BBs, hence return false (unmergable) if the
        // instruction is a terminator.
        if (I.isTerminator())
          return false;

        if (!isa<CallInst>(&I))
          return true;

        CallInst *CI = cast<CallInst>(&I);
        if (IsBeforeMergableRegion) {
          Function *CalledFunction = CI->getCalledFunction();
          if (!CalledFunction)
            return false;
          // Return false (unmergable) if the call before the parallel
          // region calls an explicit affinity (proc_bind) or number of
          // threads (num_threads) compiler-generated function. Those settings
          // may be incompatible with following parallel regions.
          // TODO: ICV tracking to detect compatibility.
          for (const auto &RFI : UnmergableCallsInfo) {
            if (CalledFunction == RFI.Declaration)
              return false;
          }
        } else {
          // Return false (unmergable) if there is a call instruction
          // in-between parallel regions when it is not an intrinsic. It
          // may call an unmergable OpenMP runtime function in its callpath.
          // TODO: Keep track of possible OpenMP calls in the callpath.
          if (!isa<IntrinsicInst>(CI))
            return false;
        }

        return true;
      };
      // Find maximal number of parallel region CIs that are safe to merge.
      for (auto It = BB->begin(), End = BB->end(); It != End;) {
        Instruction &I = *It;
        ++It;

        if (CIs.count(&I)) {
          MergableCIs.push_back(cast<CallInst>(&I));
          continue;
        }

        // Continue expanding if the instruction is mergable.
        if (IsMergable(I, MergableCIs.empty()))
          continue;

        // Forward the instruction iterator to skip the next parallel region
        // since there is an unmergable instruction which can affect it.
        for (; It != End; ++It) {
          Instruction &SkipI = *It;
          if (CIs.count(&SkipI)) {
            LLVM_DEBUG(dbgs() << TAG << "Skip parallel region " << SkipI
                              << " due to " << I << "\n");
            ++It;
            break;
          }
        }

        // Store mergable regions found.
        if (MergableCIs.size() > 1) {
          MergableCIsVector.push_back(MergableCIs);
          LLVM_DEBUG(dbgs() << TAG << "Found " << MergableCIs.size()
                            << " parallel regions in block " << BB->getName()
                            << " of function " << BB->getParent()->getName()
                            << "\n";);
        }

        MergableCIs.clear();
      }

      if (!MergableCIsVector.empty()) {
        Changed = true;

        for (auto &MergableCIs : MergableCIsVector)
          Merge(MergableCIs, BB);
        MergableCIsVector.clear();
      }
    }

    if (Changed) {
      /// Re-collect use for fork calls, emitted barrier calls, and
      /// any emitted master/end_master calls.
      OMPInfoCache.recollectUsesForFunction(OMPRTL___kmpc_fork_call);
      OMPInfoCache.recollectUsesForFunction(OMPRTL___kmpc_barrier);
      OMPInfoCache.recollectUsesForFunction(OMPRTL___kmpc_master);
      OMPInfoCache.recollectUsesForFunction(OMPRTL___kmpc_end_master);
    }

    return Changed;
  }

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
        Changed |= deduplicateRuntimeCalls(
            *F, OMPInfoCache.RFIs[DeduplicableRuntimeCallID]);

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

      OffloadArray OffloadArrays[3];
      if (!getValuesInOffloadArrays(*RTCall, OffloadArrays))
        return false;

      LLVM_DEBUG(dumpValuesInOffloadArrays(OffloadArrays));

      // TODO: Check if can be moved upwards.
      bool WasSplit = false;
      Instruction *WaitMovementPoint = canBeMovedDownwards(*RTCall);
      if (WaitMovementPoint)
        WasSplit = splitTargetDataBeginRTC(*RTCall, *WaitMovementPoint);

      Changed |= WasSplit;
      return WasSplit;
    };
    RFI.foreachUse(SCC, SplitMemTransfers);

    return Changed;
  }

  void analysisGlobalization() {
    auto &RFI = OMPInfoCache.RFIs[OMPRTL___kmpc_alloc_shared];

    auto CheckGlobalization = [&](Use &U, Function &Decl) {
      if (CallInst *CI = getCallIfRegularCall(U, &RFI)) {
        auto Remark = [&](OptimizationRemarkMissed ORM) {
          return ORM
                 << "Found thread data sharing on the GPU. "
                 << "Expect degraded performance due to data globalization.";
        };
        emitRemark<OptimizationRemarkMissed>(CI, "OpenMPGlobalization", Remark);
      }

      return false;
    };

    RFI.foreachUse(SCC, CheckGlobalization);
  }

  /// Maps the values stored in the offload arrays passed as arguments to
  /// \p RuntimeCall into the offload arrays in \p OAs.
  bool getValuesInOffloadArrays(CallInst &RuntimeCall,
                                MutableArrayRef<OffloadArray> OAs) {
    assert(OAs.size() == 3 && "Need space for three offload arrays!");

    // A runtime call that involves memory offloading looks something like:
    // call void @__tgt_target_data_begin_mapper(arg0, arg1,
    //   i8** %offload_baseptrs, i8** %offload_ptrs, i64* %offload_sizes,
    // ...)
    // So, the idea is to access the allocas that allocate space for these
    // offload arrays, offload_baseptrs, offload_ptrs, offload_sizes.
    // Therefore:
    // i8** %offload_baseptrs.
    Value *BasePtrsArg =
        RuntimeCall.getArgOperand(OffloadArray::BasePtrsArgNum);
    // i8** %offload_ptrs.
    Value *PtrsArg = RuntimeCall.getArgOperand(OffloadArray::PtrsArgNum);
    // i8** %offload_sizes.
    Value *SizesArg = RuntimeCall.getArgOperand(OffloadArray::SizesArgNum);

    // Get values stored in **offload_baseptrs.
    auto *V = getUnderlyingObject(BasePtrsArg);
    if (!isa<AllocaInst>(V))
      return false;
    auto *BasePtrsArray = cast<AllocaInst>(V);
    if (!OAs[0].initialize(*BasePtrsArray, RuntimeCall))
      return false;

    // Get values stored in **offload_baseptrs.
    V = getUnderlyingObject(PtrsArg);
    if (!isa<AllocaInst>(V))
      return false;
    auto *PtrsArray = cast<AllocaInst>(V);
    if (!OAs[1].initialize(*PtrsArray, RuntimeCall))
      return false;

    // Get values stored in **offload_sizes.
    V = getUnderlyingObject(SizesArg);
    // If it's a [constant] global array don't analyze it.
    if (isa<GlobalValue>(V))
      return isa<Constant>(V);
    if (!isa<AllocaInst>(V))
      return false;

    auto *SizesArray = cast<AllocaInst>(V);
    if (!OAs[2].initialize(*SizesArray, RuntimeCall))
      return false;

    return true;
  }

  /// Prints the values in the OffloadArrays \p OAs using LLVM_DEBUG.
  /// For now this is a way to test that the function getValuesInOffloadArrays
  /// is working properly.
  /// TODO: Move this to a unittest when unittests are available for OpenMPOpt.
  void dumpValuesInOffloadArrays(ArrayRef<OffloadArray> OAs) {
    assert(OAs.size() == 3 && "There are three offload arrays to debug!");

    LLVM_DEBUG(dbgs() << TAG << " Successfully got offload values:\n");
    std::string ValuesStr;
    raw_string_ostream Printer(ValuesStr);
    std::string Separator = " --- ";

    for (auto *BP : OAs[0].StoredValues) {
      BP->print(Printer);
      Printer << Separator;
    }
    LLVM_DEBUG(dbgs() << "\t\toffload_baseptrs: " << Printer.str() << "\n");
    ValuesStr.clear();

    for (auto *P : OAs[1].StoredValues) {
      P->print(Printer);
      Printer << Separator;
    }
    LLVM_DEBUG(dbgs() << "\t\toffload_ptrs: " << Printer.str() << "\n");
    ValuesStr.clear();

    for (auto *S : OAs[2].StoredValues) {
      S->print(Printer);
      Printer << Separator;
    }
    LLVM_DEBUG(dbgs() << "\t\toffload_sizes: " << Printer.str() << "\n");
  }

  /// Returns the instruction where the "wait" counterpart \p RuntimeCall can be
  /// moved. Returns nullptr if the movement is not possible, or not worth it.
  Instruction *canBeMovedDownwards(CallInst &RuntimeCall) {
    // FIXME: This traverses only the BasicBlock where RuntimeCall is.
    //  Make it traverse the CFG.

    Instruction *CurrentI = &RuntimeCall;
    bool IsWorthIt = false;
    while ((CurrentI = CurrentI->getNextNode())) {

      // TODO: Once we detect the regions to be offloaded we should use the
      //  alias analysis manager to check if CurrentI may modify one of
      //  the offloaded regions.
      if (CurrentI->mayHaveSideEffects() || CurrentI->mayReadFromMemory()) {
        if (IsWorthIt)
          return CurrentI;

        return nullptr;
      }

      // FIXME: For now if we move it over anything without side effect
      //  is worth it.
      IsWorthIt = true;
    }

    // Return end of BasicBlock.
    return RuntimeCall.getParent()->getTerminator();
  }

  /// Splits \p RuntimeCall into its "issue" and "wait" counterparts.
  bool splitTargetDataBeginRTC(CallInst &RuntimeCall,
                               Instruction &WaitMovementPoint) {
    // Create stack allocated handle (__tgt_async_info) at the beginning of the
    // function. Used for storing information of the async transfer, allowing to
    // wait on it later.
    auto &IRBuilder = OMPInfoCache.OMPBuilder;
    auto *F = RuntimeCall.getCaller();
    Instruction *FirstInst = &(F->getEntryBlock().front());
    AllocaInst *Handle = new AllocaInst(
        IRBuilder.AsyncInfo, F->getAddressSpace(), "handle", FirstInst);

    // Add "issue" runtime call declaration:
    // declare %struct.tgt_async_info @__tgt_target_data_begin_issue(i64, i32,
    //   i8**, i8**, i64*, i64*)
    FunctionCallee IssueDecl = IRBuilder.getOrCreateRuntimeFunction(
        M, OMPRTL___tgt_target_data_begin_mapper_issue);

    // Change RuntimeCall call site for its asynchronous version.
    SmallVector<Value *, 16> Args;
    for (auto &Arg : RuntimeCall.args())
      Args.push_back(Arg.get());
    Args.push_back(Handle);

    CallInst *IssueCallsite =
        CallInst::Create(IssueDecl, Args, /*NameStr=*/"", &RuntimeCall);
    RuntimeCall.eraseFromParent();

    // Add "wait" runtime call declaration:
    // declare void @__tgt_target_data_begin_wait(i64, %struct.__tgt_async_info)
    FunctionCallee WaitDecl = IRBuilder.getOrCreateRuntimeFunction(
        M, OMPRTL___tgt_target_data_begin_mapper_wait);

    Value *WaitParams[2] = {
        IssueCallsite->getArgOperand(
            OffloadArray::DeviceIDArgNum), // device_id.
        Handle                             // handle to wait on.
    };
    CallInst::Create(WaitDecl, WaitParams, /*NameStr=*/"", &WaitMovementPoint);

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
            return OR << "OpenMP runtime call "
                      << ore::NV("OpenMPOptRuntime", RFI.Name)
                      << " moved to beginning of OpenMP region";
          };
          emitRemark<OptimizationRemark>(&F, "OpenMPRuntimeCodeMotion", Remark);

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
      emitRemark<OptimizationRemark>(&F, "OpenMPRuntimeDeduplicated", Remark);

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
  template <typename RemarkKind, typename RemarkCallBack>
  void emitRemark(Instruction *I, StringRef RemarkName,
                  RemarkCallBack &&RemarkCB) const {
    Function *F = I->getParent()->getParent();
    auto &ORE = OREGetter(F);

    ORE.emit([&]() { return RemarkCB(RemarkKind(DEBUG_TYPE, RemarkName, I)); });
  }

  /// Emit a remark on a function.
  template <typename RemarkKind, typename RemarkCallBack>
  void emitRemark(Function *F, StringRef RemarkName,
                  RemarkCallBack &&RemarkCB) const {
    auto &ORE = OREGetter(F);

    ORE.emit([&]() { return RemarkCB(RemarkKind(DEBUG_TYPE, RemarkName, F)); });
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
    auto &GlobalizationRFI = OMPInfoCache.RFIs[OMPRTL___kmpc_alloc_shared];
    auto CreateAA = [&](Use &U, Function &F) {
      A.getOrCreateAAFor<AAHeapToShared>(IRPosition::function(F));
      return false;
    };
    GlobalizationRFI.foreachUse(SCC, CreateAA);

    // Create an ExecutionDomain AA for every function and a HeapToStack AA for
    // every function if there is a device kernel.
    for (auto *F : SCC) {
      if (!F->isDeclaration())
        A.getOrCreateAAFor<AAExecutionDomain>(IRPosition::function(*F));
      if (isOpenMPDevice(M))
        A.getOrCreateAAFor<AAHeapToStack>(IRPosition::function(*F));
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
    if (!F.hasLocalLinkage()) {

      // See https://openmp.llvm.org/remarks/OptimizationRemarks.html
      auto Remark = [&](OptimizationRemarkAnalysis ORA) {
        return ORA
               << "[OMP100] Potentially unknown OpenMP target region caller";
      };
      emitRemark<OptimizationRemarkAnalysis>(&F, "OMP100", Remark);

      return nullptr;
    }
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

      OMPInformationCache::RuntimeFunctionInfo &KernelParallelRFI =
          OMPInfoCache.RFIs[OMPRTL___kmpc_parallel_51];
      // Allow the use in __kmpc_parallel_51 calls.
      if (OpenMPOpt::getCallIfRegularCall(*U.getUser(), &KernelParallelRFI))
        return getUniqueKernelFor(*CB);
      return nullptr;
    }
    // Disallow every other use.
    return nullptr;
  };

  // TODO: In the future we want to track more than just a unique kernel.
  SmallPtrSet<Kernel, 2> PotentialKernels;
  OMPInformationCache::foreachUse(F, [&](const Use &U) {
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
  OMPInformationCache::RuntimeFunctionInfo &KernelParallelRFI =
      OMPInfoCache.RFIs[OMPRTL___kmpc_parallel_51];

  bool Changed = false;
  if (!KernelParallelRFI)
    return Changed;

  for (Function *F : SCC) {

    // Check if the function is a use in a __kmpc_parallel_51 call at
    // all.
    bool UnknownUse = false;
    bool KernelParallelUse = false;
    unsigned NumDirectCalls = 0;

    SmallVector<Use *, 2> ToBeReplacedStateMachineUses;
    OMPInformationCache::foreachUse(*F, [&](Use &U) {
      if (auto *CB = dyn_cast<CallBase>(U.getUser()))
        if (CB->isCallee(&U)) {
          ++NumDirectCalls;
          return;
        }

      if (isa<ICmpInst>(U.getUser())) {
        ToBeReplacedStateMachineUses.push_back(&U);
        return;
      }

      // Find wrapper functions that represent parallel kernels.
      CallInst *CI =
          OpenMPOpt::getCallIfRegularCall(*U.getUser(), &KernelParallelRFI);
      const unsigned int WrapperFunctionArgNo = 6;
      if (!KernelParallelUse && CI &&
          CI->getArgOperandNo(&U) == WrapperFunctionArgNo) {
        KernelParallelUse = true;
        ToBeReplacedStateMachineUses.push_back(&U);
        return;
      }
      UnknownUse = true;
    });

    // Do not emit a remark if we haven't seen a __kmpc_parallel_51
    // use.
    if (!KernelParallelUse)
      continue;

    {
      auto Remark = [&](OptimizationRemarkAnalysis ORA) {
        return ORA << "Found a parallel region that is called in a target "
                      "region but not part of a combined target construct nor "
                      "nested inside a target construct without intermediate "
                      "code. This can lead to excessive register usage for "
                      "unrelated target regions in the same translation unit "
                      "due to spurious call edges assumed by ptxas.";
      };
      emitRemark<OptimizationRemarkAnalysis>(F, "OpenMPParallelRegionInNonSPMD",
                                             Remark);
    }

    // If this ever hits, we should investigate.
    // TODO: Checking the number of uses is not a necessary restriction and
    // should be lifted.
    if (UnknownUse || NumDirectCalls != 1 ||
        ToBeReplacedStateMachineUses.size() != 2) {
      {
        auto Remark = [&](OptimizationRemarkAnalysis ORA) {
          return ORA << "Parallel region is used in "
                     << (UnknownUse ? "unknown" : "unexpected")
                     << " ways; will not attempt to rewrite the state machine.";
        };
        emitRemark<OptimizationRemarkAnalysis>(
            F, "OpenMPParallelRegionInNonSPMD", Remark);
      }
      continue;
    }

    // Even if we have __kmpc_parallel_51 calls, we (for now) give
    // up if the function is not called from a unique kernel.
    Kernel K = getUniqueKernelFor(*F);
    if (!K) {
      {
        auto Remark = [&](OptimizationRemarkAnalysis ORA) {
          return ORA << "Parallel region is not known to be called from a "
                        "unique single target region, maybe the surrounding "
                        "function has external linkage?; will not attempt to "
                        "rewrite the state machine use.";
        };
        emitRemark<OptimizationRemarkAnalysis>(
            F, "OpenMPParallelRegionInMultipleKernesl", Remark);
      }
      continue;
    }

    // We now know F is a parallel body function called only from the kernel K.
    // We also identified the state machine uses in which we replace the
    // function pointer by a new global symbol for identification purposes. This
    // ensures only direct calls to the function are left.

    {
      auto RemarkParalleRegion = [&](OptimizationRemarkAnalysis ORA) {
        return ORA << "Specialize parallel region that is only reached from a "
                      "single target region to avoid spurious call edges and "
                      "excessive register usage in other target regions. "
                      "(parallel region ID: "
                   << ore::NV("OpenMPParallelRegion", F->getName())
                   << ", kernel ID: "
                   << ore::NV("OpenMPTargetRegion", K->getName()) << ")";
      };
      emitRemark<OptimizationRemarkAnalysis>(F, "OpenMPParallelRegionInNonSPMD",
                                             RemarkParalleRegion);
      auto RemarkKernel = [&](OptimizationRemarkAnalysis ORA) {
        return ORA << "Target region containing the parallel region that is "
                      "specialized. (parallel region ID: "
                   << ore::NV("OpenMPParallelRegion", F->getName())
                   << ", kernel ID: "
                   << ore::NV("OpenMPTargetRegion", K->getName()) << ")";
      };
      emitRemark<OptimizationRemarkAnalysis>(K, "OpenMPParallelRegionInNonSPMD",
                                             RemarkKernel);
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
    if (!CB || CB->hasFnAttr("no_openmp") ||
        CB->hasFnAttr("no_openmp_routines"))
      return None;

    auto &OMPInfoCache = static_cast<OMPInformationCache &>(A.getInfoCache());
    auto &GetterRFI = OMPInfoCache.RFIs[OMPInfoCache.ICVs[ICV].Getter];
    auto &SetterRFI = OMPInfoCache.RFIs[OMPInfoCache.ICVs[ICV].Setter];
    Function *CalledFunction = CB->getCalledFunction();

    // Indirect call, assume ICV changes.
    if (CalledFunction == nullptr)
      return nullptr;
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

    const auto &ICVTrackingAA = A.getAAFor<AAICVTracker>(
        *this, IRPosition::callsite_returned(*CB), DepClassTy::REQUIRED);

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
        *this, IRPosition::function(*getAnchorScope()), DepClassTy::REQUIRED);

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
        *this, IRPosition::function(*getAnchorScope()), DepClassTy::REQUIRED);

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
        *this, IRPosition::returned(*getAssociatedFunction()),
        DepClassTy::REQUIRED);

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

struct AAExecutionDomainFunction : public AAExecutionDomain {
  AAExecutionDomainFunction(const IRPosition &IRP, Attributor &A)
      : AAExecutionDomain(IRP, A) {}

  const std::string getAsStr() const override {
    return "[AAExecutionDomain] " + std::to_string(SingleThreadedBBs.size()) +
           "/" + std::to_string(NumBBs) + " BBs thread 0 only.";
  }

  /// See AbstractAttribute::trackStatistics().
  void trackStatistics() const override {}

  void initialize(Attributor &A) override {
    Function *F = getAnchorScope();
    for (const auto &BB : *F)
      SingleThreadedBBs.insert(&BB);
    NumBBs = SingleThreadedBBs.size();
  }

  ChangeStatus manifest(Attributor &A) override {
    LLVM_DEBUG({
      for (const BasicBlock *BB : SingleThreadedBBs)
        dbgs() << TAG << " Basic block @" << getAnchorScope()->getName() << " "
               << BB->getName() << " is executed by a single thread.\n";
    });
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus updateImpl(Attributor &A) override;

  /// Check if an instruction is executed by a single thread.
  bool isExecutedByInitialThreadOnly(const Instruction &I) const override {
    return isExecutedByInitialThreadOnly(*I.getParent());
  }

  bool isExecutedByInitialThreadOnly(const BasicBlock &BB) const override {
    return isValidState() && SingleThreadedBBs.contains(&BB);
  }

  /// Set of basic blocks that are executed by a single thread.
  DenseSet<const BasicBlock *> SingleThreadedBBs;

  /// Total number of basic blocks in this function.
  long unsigned NumBBs;
};

ChangeStatus AAExecutionDomainFunction::updateImpl(Attributor &A) {
  Function *F = getAnchorScope();
  ReversePostOrderTraversal<Function *> RPOT(F);
  auto NumSingleThreadedBBs = SingleThreadedBBs.size();

  bool AllCallSitesKnown;
  auto PredForCallSite = [&](AbstractCallSite ACS) {
    const auto &ExecutionDomainAA = A.getAAFor<AAExecutionDomain>(
        *this, IRPosition::function(*ACS.getInstruction()->getFunction()),
        DepClassTy::REQUIRED);
    return ACS.isDirectCall() &&
           ExecutionDomainAA.isExecutedByInitialThreadOnly(
               *ACS.getInstruction());
  };

  if (!A.checkForAllCallSites(PredForCallSite, *this,
                              /* RequiresAllCallSites */ true,
                              AllCallSitesKnown))
    SingleThreadedBBs.erase(&F->getEntryBlock());

  // Check if the edge into the successor block compares a thread-id function to
  // a constant zero.
  // TODO: Use AAValueSimplify to simplify and propogate constants.
  // TODO: Check more than a single use for thread ID's.
  auto IsInitialThreadOnly = [&](BranchInst *Edge, BasicBlock *SuccessorBB) {
    if (!Edge || !Edge->isConditional())
      return false;
    if (Edge->getSuccessor(0) != SuccessorBB)
      return false;

    auto *Cmp = dyn_cast<CmpInst>(Edge->getCondition());
    if (!Cmp || !Cmp->isTrueWhenEqual() || !Cmp->isEquality())
      return false;

    // Temporarily match the pattern generated by clang for teams regions.
    // TODO: Remove this once the new runtime is in place.
    ConstantInt *One, *NegOne;
    CmpInst::Predicate Pred;
    auto &&m_ThreadID = m_Intrinsic<Intrinsic::nvvm_read_ptx_sreg_tid_x>();
    auto &&m_WarpSize = m_Intrinsic<Intrinsic::nvvm_read_ptx_sreg_warpsize>();
    auto &&m_BlockSize = m_Intrinsic<Intrinsic::nvvm_read_ptx_sreg_ntid_x>();
    if (match(Cmp, m_Cmp(Pred, m_ThreadID,
                         m_And(m_Sub(m_BlockSize, m_ConstantInt(One)),
                               m_Xor(m_Sub(m_WarpSize, m_ConstantInt(One)),
                                     m_ConstantInt(NegOne))))))
      if (One->isOne() && NegOne->isMinusOne() &&
          Pred == CmpInst::Predicate::ICMP_EQ)
        return true;

    ConstantInt *C = dyn_cast<ConstantInt>(Cmp->getOperand(1));
    if (!C || !C->isZero())
      return false;

    if (auto *II = dyn_cast<IntrinsicInst>(Cmp->getOperand(0)))
      if (II->getIntrinsicID() == Intrinsic::nvvm_read_ptx_sreg_tid_x)
        return true;
    if (auto *II = dyn_cast<IntrinsicInst>(Cmp->getOperand(0)))
      if (II->getIntrinsicID() == Intrinsic::amdgcn_workitem_id_x)
        return true;

    return false;
  };

  // Merge all the predecessor states into the current basic block. A basic
  // block is executed by a single thread if all of its predecessors are.
  auto MergePredecessorStates = [&](BasicBlock *BB) {
    if (pred_begin(BB) == pred_end(BB))
      return SingleThreadedBBs.contains(BB);

    bool IsInitialThread = true;
    for (auto PredBB = pred_begin(BB), PredEndBB = pred_end(BB);
         PredBB != PredEndBB; ++PredBB) {
      if (!IsInitialThreadOnly(dyn_cast<BranchInst>((*PredBB)->getTerminator()),
                              BB))
        IsInitialThread &= SingleThreadedBBs.contains(*PredBB);
    }

    return IsInitialThread;
  };

  for (auto *BB : RPOT) {
    if (!MergePredecessorStates(BB))
      SingleThreadedBBs.erase(BB);
  }

  return (NumSingleThreadedBBs == SingleThreadedBBs.size())
             ? ChangeStatus::UNCHANGED
             : ChangeStatus::CHANGED;
}

/// Try to replace memory allocation calls called by a single thread with a
/// static buffer of shared memory.
struct AAHeapToShared : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;
  AAHeapToShared(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Create an abstract attribute view for the position \p IRP.
  static AAHeapToShared &createForPosition(const IRPosition &IRP,
                                           Attributor &A);

  /// See AbstractAttribute::getName().
  const std::string getName() const override { return "AAHeapToShared"; }

  /// See AbstractAttribute::getIdAddr().
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAHeapToShared.
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};

struct AAHeapToSharedFunction : public AAHeapToShared {
  AAHeapToSharedFunction(const IRPosition &IRP, Attributor &A)
      : AAHeapToShared(IRP, A) {}

  const std::string getAsStr() const override {
    return "[AAHeapToShared] " + std::to_string(MallocCalls.size()) +
           " malloc calls eligible.";
  }

  /// See AbstractAttribute::trackStatistics().
  void trackStatistics() const override {}

  void initialize(Attributor &A) override {
    auto &OMPInfoCache = static_cast<OMPInformationCache &>(A.getInfoCache());
    auto &RFI = OMPInfoCache.RFIs[OMPRTL___kmpc_alloc_shared];

    for (User *U : RFI.Declaration->users())
      if (CallBase *CB = dyn_cast<CallBase>(U))
        MallocCalls.insert(CB);
  }

  ChangeStatus manifest(Attributor &A) override {
    if (MallocCalls.empty())
      return ChangeStatus::UNCHANGED;

    auto &OMPInfoCache = static_cast<OMPInformationCache &>(A.getInfoCache());
    auto &FreeCall = OMPInfoCache.RFIs[OMPRTL___kmpc_free_shared];

    Function *F = getAnchorScope();
    auto *HS = A.lookupAAFor<AAHeapToStack>(IRPosition::function(*F), this,
                                            DepClassTy::OPTIONAL);

    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    for (CallBase *CB : MallocCalls) {
      // Skip replacing this if HeapToStack has already claimed it.
      if (HS && HS->isKnownHeapToStack(*CB))
        continue;

      // Find the unique free call to remove it.
      SmallVector<CallBase *, 4> FreeCalls;
      for (auto *U : CB->users()) {
        CallBase *C = dyn_cast<CallBase>(U);
        if (C && C->getCalledFunction() == FreeCall.Declaration)
          FreeCalls.push_back(C);
      }
      if (FreeCalls.size() != 1)
        continue;

      ConstantInt *AllocSize = dyn_cast<ConstantInt>(CB->getArgOperand(0));

      LLVM_DEBUG(dbgs() << TAG << "Replace globalization call in "
                        << CB->getCaller()->getName() << " with "
                        << AllocSize->getZExtValue()
                        << " bytes of shared memory\n");

      // Create a new shared memory buffer of the same size as the allocation
      // and replace all the uses of the original allocation with it.
      Module *M = CB->getModule();
      Type *Int8Ty = Type::getInt8Ty(M->getContext());
      Type *Int8ArrTy = ArrayType::get(Int8Ty, AllocSize->getZExtValue());
      auto *SharedMem = new GlobalVariable(
          *M, Int8ArrTy, /* IsConstant */ false, GlobalValue::InternalLinkage,
          UndefValue::get(Int8ArrTy), CB->getName(), nullptr,
          GlobalValue::NotThreadLocal,
          static_cast<unsigned>(AddressSpace::Shared));
      auto *NewBuffer =
          ConstantExpr::getPointerCast(SharedMem, Int8Ty->getPointerTo());

      auto Remark = [&](OptimizationRemark OR) {
        return OR << "Replaced globalized variable with "
                  << ore::NV("SharedMemory", AllocSize->getZExtValue())
                  << ((AllocSize->getZExtValue() != 1) ? " bytes " : " byte ")
                  << "of shared memory";
      };
      A.emitRemark<OptimizationRemark>(CB, "OpenMPReplaceGlobalization",
                                       Remark);

      SharedMem->setAlignment(MaybeAlign(32));

      A.changeValueAfterManifest(*CB, *NewBuffer);
      A.deleteAfterManifest(*CB);
      A.deleteAfterManifest(*FreeCalls.front());

      NumBytesMovedToSharedMemory += AllocSize->getZExtValue();
      Changed = ChangeStatus::CHANGED;
    }

    return Changed;
  }

  ChangeStatus updateImpl(Attributor &A) override {
    auto &OMPInfoCache = static_cast<OMPInformationCache &>(A.getInfoCache());
    auto &RFI = OMPInfoCache.RFIs[OMPRTL___kmpc_alloc_shared];
    Function *F = getAnchorScope();

    auto NumMallocCalls = MallocCalls.size();

    // Only consider malloc calls executed by a single thread with a constant.
    for (User *U : RFI.Declaration->users()) {
      const auto &ED = A.getAAFor<AAExecutionDomain>(
          *this, IRPosition::function(*F), DepClassTy::REQUIRED);
      if (CallBase *CB = dyn_cast<CallBase>(U))
        if (!dyn_cast<ConstantInt>(CB->getArgOperand(0)) ||
            !ED.isExecutedByInitialThreadOnly(*CB))
          MallocCalls.erase(CB);
    }

    if (NumMallocCalls != MallocCalls.size())
      return ChangeStatus::CHANGED;

    return ChangeStatus::UNCHANGED;
  }

  /// Collection of all malloc calls in a function.
  SmallPtrSet<CallBase *, 4> MallocCalls;
};

} // namespace

const char AAICVTracker::ID = 0;
const char AAExecutionDomain::ID = 0;
const char AAHeapToShared::ID = 0;

AAICVTracker &AAICVTracker::createForPosition(const IRPosition &IRP,
                                              Attributor &A) {
  AAICVTracker *AA = nullptr;
  switch (IRP.getPositionKind()) {
  case IRPosition::IRP_INVALID:
  case IRPosition::IRP_FLOAT:
  case IRPosition::IRP_ARGUMENT:
  case IRPosition::IRP_CALL_SITE_ARGUMENT:
    llvm_unreachable("ICVTracker can only be created for function position!");
  case IRPosition::IRP_RETURNED:
    AA = new (A.Allocator) AAICVTrackerFunctionReturned(IRP, A);
    break;
  case IRPosition::IRP_CALL_SITE_RETURNED:
    AA = new (A.Allocator) AAICVTrackerCallSiteReturned(IRP, A);
    break;
  case IRPosition::IRP_CALL_SITE:
    AA = new (A.Allocator) AAICVTrackerCallSite(IRP, A);
    break;
  case IRPosition::IRP_FUNCTION:
    AA = new (A.Allocator) AAICVTrackerFunction(IRP, A);
    break;
  }

  return *AA;
}

AAExecutionDomain &AAExecutionDomain::createForPosition(const IRPosition &IRP,
                                                        Attributor &A) {
  AAExecutionDomainFunction *AA = nullptr;
  switch (IRP.getPositionKind()) {
  case IRPosition::IRP_INVALID:
  case IRPosition::IRP_FLOAT:
  case IRPosition::IRP_ARGUMENT:
  case IRPosition::IRP_CALL_SITE_ARGUMENT:
  case IRPosition::IRP_RETURNED:
  case IRPosition::IRP_CALL_SITE_RETURNED:
  case IRPosition::IRP_CALL_SITE:
    llvm_unreachable(
        "AAExecutionDomain can only be created for function position!");
  case IRPosition::IRP_FUNCTION:
    AA = new (A.Allocator) AAExecutionDomainFunction(IRP, A);
    break;
  }

  return *AA;
}

AAHeapToShared &AAHeapToShared::createForPosition(const IRPosition &IRP,
                                                  Attributor &A) {
  AAHeapToSharedFunction *AA = nullptr;
  switch (IRP.getPositionKind()) {
  case IRPosition::IRP_INVALID:
  case IRPosition::IRP_FLOAT:
  case IRPosition::IRP_ARGUMENT:
  case IRPosition::IRP_CALL_SITE_ARGUMENT:
  case IRPosition::IRP_RETURNED:
  case IRPosition::IRP_CALL_SITE_RETURNED:
  case IRPosition::IRP_CALL_SITE:
    llvm_unreachable(
        "AAHeapToShared can only be created for function position!");
  case IRPosition::IRP_FUNCTION:
    AA = new (A.Allocator) AAHeapToSharedFunction(IRP, A);
    break;
  }

  return *AA;
}

PreservedAnalyses OpenMPOptPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (!containsOpenMP(M))
    return PreservedAnalyses::all();
  if (DisableOpenMPOptimizations)
    return PreservedAnalyses::all();

  KernelSet Kernels = getDeviceKernels(M);

  // Create internal copies of each function if this is a kernel Module.
  DenseSet<const Function *> InternalizedFuncs;
  if (isOpenMPDevice(M))
    for (Function &F : M)
      if (!F.isDeclaration() && !Kernels.contains(&F))
        if (Attributor::internalizeFunction(F, /* Force */ true))
          InternalizedFuncs.insert(&F);

  // Look at every function definition in the Module that wasn't internalized.
  SmallVector<Function *, 16> SCC;
  for (Function &F : M)
    if (!F.isDeclaration() && !InternalizedFuncs.contains(&F))
      SCC.push_back(&F);

  if (SCC.empty())
    return PreservedAnalyses::all();

  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  AnalysisGetter AG(FAM);

  auto OREGetter = [&FAM](Function *F) -> OptimizationRemarkEmitter & {
    return FAM.getResult<OptimizationRemarkEmitterAnalysis>(*F);
  };

  BumpPtrAllocator Allocator;
  CallGraphUpdater CGUpdater;

  SetVector<Function *> Functions(SCC.begin(), SCC.end());
  OMPInformationCache InfoCache(M, AG, Allocator, /*CGSCC*/ Functions, Kernels);

  unsigned MaxFixponitIterations = (Kernels.empty()) ? 64 : 32;
  Attributor A(Functions, InfoCache, CGUpdater, nullptr, true, false, MaxFixponitIterations, OREGetter,
               DEBUG_TYPE);

  OpenMPOpt OMPOpt(SCC, CGUpdater, OREGetter, InfoCache, A);
  bool Changed = OMPOpt.run(true);
  if (Changed)
    return PreservedAnalyses::none();

  return PreservedAnalyses::all();
}

PreservedAnalyses OpenMPOptCGSCCPass::run(LazyCallGraph::SCC &C,
                                          CGSCCAnalysisManager &AM,
                                          LazyCallGraph &CG,
                                          CGSCCUpdateResult &UR) {
  if (!containsOpenMP(*C.begin()->getFunction().getParent()))
    return PreservedAnalyses::all();
  if (DisableOpenMPOptimizations)
    return PreservedAnalyses::all();

  SmallVector<Function *, 16> SCC;
  // If there are kernels in the module, we have to run on all SCC's.
  for (LazyCallGraph::Node &N : C) {
    Function *Fn = &N.getFunction();
    SCC.push_back(Fn);
  }

  if (SCC.empty())
    return PreservedAnalyses::all();

  Module &M = *C.begin()->getFunction().getParent();

  KernelSet Kernels = getDeviceKernels(M);

  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();

  AnalysisGetter AG(FAM);

  auto OREGetter = [&FAM](Function *F) -> OptimizationRemarkEmitter & {
    return FAM.getResult<OptimizationRemarkEmitterAnalysis>(*F);
  };

  BumpPtrAllocator Allocator;
  CallGraphUpdater CGUpdater;
  CGUpdater.initialize(CG, C, AM, UR);

  SetVector<Function *> Functions(SCC.begin(), SCC.end());
  OMPInformationCache InfoCache(*(Functions.back()->getParent()), AG, Allocator,
                                /*CGSCC*/ Functions, Kernels);

  unsigned MaxFixponitIterations = (isOpenMPDevice(M)) ? 64 : 32;
  Attributor A(Functions, InfoCache, CGUpdater, nullptr, false, true, MaxFixponitIterations, OREGetter,
               DEBUG_TYPE);

  OpenMPOpt OMPOpt(SCC, CGUpdater, OREGetter, InfoCache, A);
  bool Changed = OMPOpt.run(false);
  if (Changed)
    return PreservedAnalyses::none();

  return PreservedAnalyses::all();
}

namespace {

struct OpenMPOptCGSCCLegacyPass : public CallGraphSCCPass {
  CallGraphUpdater CGUpdater;
  static char ID;

  OpenMPOptCGSCCLegacyPass() : CallGraphSCCPass(ID) {
    initializeOpenMPOptCGSCCLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    CallGraphSCCPass::getAnalysisUsage(AU);
  }

  bool runOnSCC(CallGraphSCC &CGSCC) override {
    if (!containsOpenMP(CGSCC.getCallGraph().getModule()))
      return false;
    if (DisableOpenMPOptimizations || skipSCC(CGSCC))
      return false;

    SmallVector<Function *, 16> SCC;
    // If there are kernels in the module, we have to run on all SCC's.
    for (CallGraphNode *CGN : CGSCC) {
      Function *Fn = CGN->getFunction();
      if (!Fn || Fn->isDeclaration())
        continue;
      SCC.push_back(Fn);
    }

    if (SCC.empty())
      return false;

    Module &M = CGSCC.getCallGraph().getModule();
    KernelSet Kernels = getDeviceKernels(M);

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
    OMPInformationCache InfoCache(*(Functions.back()->getParent()), AG,
                                  Allocator,
                                  /*CGSCC*/ Functions, Kernels);

    unsigned MaxFixponitIterations = (isOpenMPDevice(M)) ? 64 : 32;
    Attributor A(Functions, InfoCache, CGUpdater, nullptr, false, true,
                 MaxFixponitIterations, OREGetter, DEBUG_TYPE);

    OpenMPOpt OMPOpt(SCC, CGUpdater, OREGetter, InfoCache, A);
    return OMPOpt.run(false);
  }

  bool doFinalization(CallGraph &CG) override { return CGUpdater.finalize(); }
};

} // end anonymous namespace

KernelSet llvm::omp::getDeviceKernels(Module &M) {
  // TODO: Create a more cross-platform way of determining device kernels.
  NamedMDNode *MD = M.getOrInsertNamedMetadata("nvvm.annotations");
  KernelSet Kernels;

  if (!MD)
    return Kernels;

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

  return Kernels;
}

bool llvm::omp::containsOpenMP(Module &M) {
  Metadata *MD = M.getModuleFlag("openmp");
  if (!MD)
    return false;

  return true;
}

bool llvm::omp::isOpenMPDevice(Module &M) {
  Metadata *MD = M.getModuleFlag("openmp-device");
  if (!MD)
    return false;

  return true;
}

char OpenMPOptCGSCCLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(OpenMPOptCGSCCLegacyPass, "openmp-opt-cgscc",
                      "OpenMP specific optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_END(OpenMPOptCGSCCLegacyPass, "openmp-opt-cgscc",
                    "OpenMP specific optimizations", false, false)

Pass *llvm::createOpenMPOptCGSCCLegacyPass() {
  return new OpenMPOptCGSCCLegacyPass();
}
