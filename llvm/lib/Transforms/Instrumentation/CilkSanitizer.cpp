//===- CilkSanitizer.cpp - Nondeterminism detector for Cilk/Tapir ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of CilkSan, a determinacy-race detector for Cilk
// programs.
//
// This instrumentation pass inserts calls to the runtime library before
// appropriate memory accesses.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/CilkSanitizer.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/TapirRaceDetect.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Instrumentation/CSI.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "cilksan"

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumAccessesWithBadSize, "Number of accesses with bad size");
STATISTIC(NumOmittedReadsBeforeWrite,
          "Number of reads ignored due to following writes");
STATISTIC(NumOmittedReadsFromConstants,
          "Number of reads from constant data");
STATISTIC(NumOmittedNonCaptured, "Number of accesses ignored due to capturing");
STATISTIC(NumOmittedStaticNoRace,
          "Number of accesses proven statically to not race");
STATISTIC(NumInstrumentedMemIntrinsicReads,
          "Number of instrumented reads from memory intrinsics");
STATISTIC(NumInstrumentedMemIntrinsicWrites,
          "Number of instrumented writes from memory intrinsics");
STATISTIC(NumInstrumentedDetaches, "Number of instrumented detaches");
STATISTIC(NumInstrumentedDetachExits, "Number of instrumented detach exits");
STATISTIC(NumInstrumentedSyncs, "Number of instrumented syncs");
STATISTIC(NumInstrumentedAllocas, "Number of instrumented allocas");
STATISTIC(NumInstrumentedAllocFns,
          "Number of instrumented allocation functions");
STATISTIC(NumInstrumentedFrees, "Number of instrumented free calls");

static cl::opt<bool>
    EnableStaticRaceDetection(
        "enable-static-race-detection", cl::init(true), cl::Hidden,
        cl::desc("Enable static detection of determinacy races."));

static cl::opt<bool>
    AssumeRaceFreeLibraryFunctions(
        "assume-race-free-lib", cl::init(false), cl::Hidden,
        cl::desc("Assume library functions are race free."));

static cl::opt<bool>
    IgnoreInaccessibleMemory(
        "ignore-inaccessible-memory", cl::init(false), cl::Hidden,
        cl::desc("Ignore inaccessible memory when checking for races."));

static cl::opt<bool>
    AssumeNoExceptions(
        "cilksan-assume-no-exceptions", cl::init(false), cl::Hidden,
        cl::desc("Assume that ordinary calls cannot throw exceptions."));

static cl::opt<bool>
    AssumeLibCallsDontRecur(
        "cilksan-assume-lib-calls-dont-recur", cl::init(true), cl::Hidden,
        cl::desc("Assume that library calls do not recur."));

static cl::opt<bool>
    IgnoreSanitizeCilkAttr(
        "ignore-sanitize-cilk-attr", cl::init(false), cl::Hidden,
        cl::desc("Ignore the 'sanitize_cilk' attribute when choosing what to "
                 "instrument."));

static const char *const CsiUnitObjTableName = "__csi_unit_obj_table";
static const char *const CsiUnitObjTableArrayName = "__csi_unit_obj_tables";

/// Maintains a mapping from CSI ID of a load or store to the source information
/// of the object accessed by that load or store.
class ObjectTable : public ForensicTable {
public:
  ObjectTable() : ForensicTable() {}
  ObjectTable(Module &M, StringRef BaseIdName)
      : ForensicTable(M, BaseIdName) {}

  /// The number of entries in this table
  uint64_t size() const { return LocalIdToSourceLocationMap.size(); }

  /// Add the given instruction to this table.
  /// \returns The local ID of the Instruction.
  uint64_t add(Instruction &I, Value *Obj);

  /// Get the Type for a pointer to a table entry.
  ///
  /// A table entry is just a source location.
  static PointerType *getPointerType(LLVMContext &C);

  /// Insert this table into the given Module.
  ///
  /// The table is constructed as a ConstantArray indexed by local IDs.  The
  /// runtime is responsible for performing the mapping that allows the table to
  /// be indexed by global ID.
  Constant *insertIntoModule(Module &M) const;

private:
  struct SourceLocation {
    StringRef Name;
    int32_t Line;
    StringRef Filename;
    StringRef Directory;
  };

  /// Map of local ID to SourceLocation.
  DenseMap<uint64_t, SourceLocation> LocalIdToSourceLocationMap;

  /// Create a struct type to match the "struct SourceLocation" type.
  /// (and the source_loc_t type in csi.h).
  static StructType *getSourceLocStructType(LLVMContext &C);

  /// Append the line and file information to the table.
  void add(uint64_t ID, int32_t Line = -1,
           StringRef Filename = "", StringRef Directory = "",
           StringRef Name = "");
};

namespace {
struct CilkSanitizerImpl : public CSIImpl {
  class SimpleInstrumentor {
  public:
    SimpleInstrumentor(CilkSanitizerImpl &CilkSanImpl, TaskInfo &TI,
                       LoopInfo &LI, DominatorTree *DT)
        : CilkSanImpl(CilkSanImpl), TI(TI), LI(LI), DT(DT) {}

    bool InstrumentSimpleInstructions(
        SmallVectorImpl<Instruction *> &Instructions);
    bool InstrumentAnyMemIntrinsics(
        SmallVectorImpl<Instruction *> &MemIntrinsics);
    bool InstrumentCalls(SmallVectorImpl<Instruction *> &Calls);
    bool InstrumentAncillaryInstructions(
        SmallPtrSetImpl<Instruction *> &Allocas,
        SmallPtrSetImpl<Instruction *> &AllocationFnCalls,
        SmallPtrSetImpl<Instruction *> &FreeCalls,
        DenseMap<Value *, unsigned> &SyncRegNums,
        DenseMap<BasicBlock *, unsigned> &SRCounters, const DataLayout &DL,
        const TargetLibraryInfo *TLI);

  private:
    void getDetachesForInstruction(Instruction *I);

    CilkSanitizerImpl &CilkSanImpl;
    TaskInfo &TI;
    LoopInfo &LI;
    DominatorTree *DT;

    SmallPtrSet<DetachInst *, 8> Detaches;

    SmallVector<Instruction *, 8> DelayedSimpleInsts;
    SmallVector<std::pair<Instruction *, unsigned>, 8> DelayedMemIntrinsics;
    SmallVector<Instruction *, 8> DelayedCalls;
  };

  class Instrumentor {
  public:
    Instrumentor(CilkSanitizerImpl &CilkSanImpl, RaceInfo &RI, TaskInfo &TI,
                 LoopInfo &LI, DominatorTree *DT)
        : CilkSanImpl(CilkSanImpl), RI(RI), TI(TI), LI(LI), DT(DT) {}

    void InsertArgSuppressionFlags(Function &F, Value *FuncId);
    bool InstrumentSimpleInstructions(
        SmallVectorImpl<Instruction *> &Instructions);
    bool InstrumentAnyMemIntrinsics(
        SmallVectorImpl<Instruction *> &MemIntrinsics);
    bool InstrumentCalls(SmallVectorImpl<Instruction *> &Calls);
    bool InstrumentAncillaryInstructions(
        SmallPtrSetImpl<Instruction *> &Allocas,
        SmallPtrSetImpl<Instruction *> &AllocationFnCalls,
        SmallPtrSetImpl<Instruction *> &FreeCalls,
        DenseMap<Value *, unsigned> &SyncRegNums,
        DenseMap<BasicBlock *, unsigned> &SRCounters, const DataLayout &DL,
        const TargetLibraryInfo *TLI);
    bool PerformDelayedInstrumentation();

  private:
    void getDetachesForInstruction(Instruction *I);
    enum class SuppressionVal : uint8_t
      {
       NoAccess = 0,
       Mod = 1,
       Ref = 2,
       ModRef = Mod | Ref,
       NoAlias = 4,
      };
    static unsigned RaceTypeToFlagVal(RaceInfo::RaceType RT);
    Value *getSuppressionValue(Instruction *I, IRBuilder<> &IRB,
                               unsigned OperandNum = static_cast<unsigned>(-1),
                               SuppressionVal DefaultSV = SuppressionVal::ModRef,
                               bool CheckArgs = true);
    Value *getNoAliasSuppressionValue(Instruction *I, IRBuilder<> &IRB,
                                      unsigned OperandNum, MemoryLocation Loc,
                                      const RaceInfo::RaceData &RD,
                                      const Value *Obj, Value *SupprVal);
    Value *getSuppressionCheck(Instruction *I, IRBuilder<> &IRB,
                               unsigned OperandNum = static_cast<unsigned>(-1));
    Value *readSuppressionVal(Value *V, IRBuilder<> &IRB);

    CilkSanitizerImpl &CilkSanImpl;
    RaceInfo &RI;
    TaskInfo &TI;
    LoopInfo &LI;
    DominatorTree *DT;

    SmallPtrSet<DetachInst *, 8> Detaches;

    DenseMap<const Value *, Value *> LocalSuppressions;
    SmallPtrSet<const Value *, 8> ArgSuppressionFlags;

    SmallVector<Instruction *, 8> DelayedSimpleInsts;
    SmallVector<std::pair<Instruction *, unsigned>, 8> DelayedMemIntrinsics;
    SmallVector<Instruction *, 8> DelayedCalls;
  };

  CilkSanitizerImpl(Module &M, CallGraph *CG,
                    function_ref<DominatorTree &(Function &)> GetDomTree,
                    function_ref<TaskInfo &(Function &)> GetTaskInfo,
                    function_ref<LoopInfo &(Function &)> GetLoopInfo,
                    function_ref<DependenceInfo &(Function &)> GetDepInfo,
                    function_ref<RaceInfo &(Function &)> GetRaceInfo,
                    const TargetLibraryInfo *TLI, bool JitMode = false,
                    bool CallsMayThrow = !AssumeNoExceptions)
      : CSIImpl(M, CG, GetDomTree, GetLoopInfo, GetTaskInfo, TLI),
        GetDepInfo(GetDepInfo), GetRaceInfo(GetRaceInfo) {
    // Even though we're doing our own instrumentation, we want the CSI setup
    // for the instrumentation of function entry/exit, memory accesses (i.e.,
    // loads and stores), atomics, memory intrinsics.  We also want call sites,
    // for extracting debug information.
    Options.InstrumentBasicBlocks = false;
    Options.InstrumentLoops = true;
    // Cilksan defines its own hooks for instrumenting memory accesses, memory
    // intrinsics, and Tapir instructions, so we disable the default CSI
    // instrumentation hooks for these IR objects.
    Options.InstrumentMemoryAccesses = false;
    Options.InstrumentMemIntrinsics = false;
    Options.InstrumentTapir = false;
    Options.InstrumentCalls = false;
    Options.jitMode = JitMode;
    Options.CallsMayThrow = CallsMayThrow;
  }
  bool run();

  static StructType *getUnitObjTableType(LLVMContext &C,
                                         PointerType *EntryPointerType);
  static Constant *objTableToUnitObjTable(Module &M,
                                          StructType *UnitObjTableType,
                                          ObjectTable &ObjTable);
  static bool simpleCallCannotRace(const Instruction &I);
  static void getAllocFnArgs(
      const Instruction *I, SmallVectorImpl<Value*> &AllocFnArgs,
      Type *SizeTy, Type *AddrTy, const TargetLibraryInfo &TLI);

  void setupBlocks(Function &F, DominatorTree *DT = nullptr);

  // Methods for handling FED tables
  void initializeFEDTables() {}
  void collectUnitFEDTables() {}

  // Methods for handling object tables
  void initializeCsanObjectTables();
  void collectUnitObjectTables();

  // Create a call to the runtime unit initialization routine in a global
  // constructor.
  CallInst *createRTUnitInitCall(IRBuilder<> &IRB) override;

  // Initialize custom hooks for CilkSanitizer
  void initializeCsanHooks();

  Value *GetCalleeFuncID(const Function *Callee, IRBuilder<> &IRB);

  // Helper function for prepareToInstrumentFunction that chooses loads and
  // stores in a basic block to instrument.
  void chooseInstructionsToInstrument(
      SmallVectorImpl<Instruction *> &Local,
      SmallVectorImpl<Instruction *> &All,
      const TaskInfo &TI, LoopInfo &LI);

  // Helper methods for instrumenting different IR objects.
  bool instrumentLoadOrStore(Instruction *I, IRBuilder<> &IRB);
  bool instrumentLoadOrStore(Instruction *I) {
    IRBuilder<> IRB(I);
    return instrumentLoadOrStore(I, IRB);
  }
  bool instrumentAtomic(Instruction *I, IRBuilder<> &IRB);
  bool instrumentAtomic(Instruction *I) {
    IRBuilder<> IRB(I);
    return instrumentAtomic(I, IRB);
  }
  bool instrumentCallsite(Instruction *I,
                          SmallVectorImpl<Value *> *SupprVals = nullptr);
  bool suppressCallsite(Instruction *I);
  bool instrumentAllocationFn(Instruction *I, DominatorTree *DT);
  bool instrumentFree(Instruction *I);
  bool instrumentDetach(DetachInst *DI, unsigned SyncRegNum,
                        unsigned NumSyncRegs, DominatorTree *DT, TaskInfo &TI,
                        LoopInfo &LI);
  bool instrumentSync(SyncInst *SI, unsigned SyncRegNum);
  void instrumentLoop(Loop &L, TaskInfo &TI,
                      DenseMap<Value *, unsigned> &SyncRegNums,
                      ScalarEvolution *SE = nullptr);
  bool instrumentAlloca(Instruction *I);

  bool instrumentFunctionUsingRI(Function &F);
  // Helper method for RI-based race detection for instrumenting an access by a
  // memory intrinsic.
  bool instrumentAnyMemIntrinAcc(Instruction *I, unsigned OperandNum,
                                 IRBuilder<> &IRB);
  bool instrumentAnyMemIntrinAcc(Instruction *I, unsigned OperandNum) {
    IRBuilder<> IRB(I);
    return instrumentAnyMemIntrinAcc(I, OperandNum, IRB);
  }

private:
  // Analysis results
  function_ref<DependenceInfo &(Function &)> GetDepInfo;
  function_ref<RaceInfo &(Function &)> GetRaceInfo;

  // Instrumentation hooks
  FunctionCallee CsanFuncEntry = nullptr;
  FunctionCallee CsanFuncExit = nullptr;
  FunctionCallee CsanRead = nullptr;
  FunctionCallee CsanWrite = nullptr;
  FunctionCallee CsanLargeRead = nullptr;
  FunctionCallee CsanLargeWrite = nullptr;
  FunctionCallee CsanBeforeCallsite = nullptr;
  FunctionCallee CsanAfterCallsite = nullptr;
  FunctionCallee CsanDetach = nullptr;
  FunctionCallee CsanDetachContinue = nullptr;
  FunctionCallee CsanTaskEntry = nullptr;
  FunctionCallee CsanTaskExit = nullptr;
  FunctionCallee CsanSync = nullptr;
  FunctionCallee CsanBeforeLoop = nullptr;
  FunctionCallee CsanAfterLoop = nullptr;
  FunctionCallee CsanAfterAllocFn = nullptr;
  FunctionCallee CsanAfterFree = nullptr;

  // Hooks for suppressing instrumentation, e.g., around callsites that cannot
  // expose a race.
  FunctionCallee CsanDisableChecking = nullptr;
  FunctionCallee CsanEnableChecking = nullptr;

  FunctionCallee GetSuppressionFlag = nullptr;
  FunctionCallee SetSuppressionFlag = nullptr;

  // CilkSanitizer custom forensic tables
  ObjectTable LoadObj, StoreObj, AllocaObj, AllocFnObj;

  SmallVector<Constant *, 4> UnitObjTables;

  SmallVector<Instruction *, 8> AllocationFnCalls;
  SmallVector<Instruction *, 8> FreeCalls;
  SmallVector<Instruction *, 8> Allocas;
  SmallPtrSet<Instruction *, 8> ToInstrument;

  // Map of functions to updated race type, for interprocedural analysis of
  // races.
  DenseMap<const Function *, RaceInfo::RaceType> FunctionRaceType;
  // TODO: Record information about what function arguments each function ref's
  // and mod's.  When instrumenting norecurse callers, factor that information
  // into the suppression flags.
  // DenseMap<const Value *, ModRefInfo> FuncArgMR;
  DenseMap<const Value *, ModRefInfo> ObjectMRForRace;

  DenseMap<DetachInst *, SmallVector<SyncInst *, 2>> DetachToSync;

  SmallPtrSet<const Function *, 8> LocalNoRecurseFunctions;
  bool FunctionIsNoRecurse(const Function &F) const {
    return (F.doesNotRecurse() || LocalNoRecurseFunctions.count(&F));
  }
};

/// CilkSanitizer: instrument the code in module to find races.
struct CilkSanitizerLegacyPass : public ModulePass {
  static char ID;  // Pass identification, replacement for typeid.
  CilkSanitizerLegacyPass(bool JitMode = false,
                          bool CallsMayThrow = !AssumeNoExceptions)
      : ModulePass(ID), JitMode(JitMode), CallsMayThrow(CallsMayThrow) {
    initializeCilkSanitizerLegacyPassPass(*PassRegistry::getPassRegistry());
  }
  StringRef getPassName() const override { return "CilkSanitizer"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnModule(Module &M) override;

  bool JitMode = false;
  bool CallsMayThrow = true;
};
} // end anonymous namespace

char CilkSanitizerLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(
    CilkSanitizerLegacyPass, "csan",
    "CilkSanitizer: detects determinacy races in Cilk programs.",
    false, false)
INITIALIZE_PASS_DEPENDENCY(BasicAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DependenceAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TapirRaceDetectWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TaskInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(
    CilkSanitizerLegacyPass, "csan",
    "CilkSanitizer: detects determinacy races in Cilk programs.",
    false, false)

void CilkSanitizerLegacyPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<CallGraphWrapperPass>();
  AU.addRequired<DependenceAnalysisWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<TapirRaceDetectWrapperPass>();
  AU.addRequired<TaskInfoWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addPreserved<BasicAAWrapperPass>();
}

ModulePass *llvm::createCilkSanitizerLegacyPass(bool JitMode) {
  return new CilkSanitizerLegacyPass(JitMode);
}

ModulePass *llvm::createCilkSanitizerLegacyPass(bool JitMode,
                                                bool CallsMayThrow) {
  return new CilkSanitizerLegacyPass(JitMode, CallsMayThrow);
}

uint64_t ObjectTable::add(Instruction &I, Value *Obj) {
  uint64_t ID = getId(&I);
  // First, if the underlying object is a global variable, get that variable's
  // debug information.
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Obj)) {
    SmallVector<DIGlobalVariableExpression *, 1> DbgGVExprs;
    GV->getDebugInfo(DbgGVExprs);
    for (auto *GVE : DbgGVExprs) {
      auto *DGV = GVE->getVariable();
      if (DGV->getName() != "") {
        add(ID, DGV->getLine(), DGV->getFilename(), DGV->getDirectory(),
            DGV->getName());
        return ID;
      }
    }
    add(ID, -1, "", "", Obj->getName());
    return ID;
  }

  // Next, if this is an alloca instruction, look for a llvm.dbg.declare
  // intrinsic.
  if (AllocaInst *AI = dyn_cast<AllocaInst>(Obj)) {
    TinyPtrVector<DbgVariableIntrinsic *> DbgDeclares = FindDbgAddrUses(AI);
    if (!DbgDeclares.empty()) {
      auto *LV = DbgDeclares.front()->getVariable();
      add(ID, LV->getLine(), LV->getFilename(), LV->getDirectory(),
          LV->getName());
      return ID;
    }
  }

  // Otherwise just examine the llvm.dbg.value intrinsics for this object.
  SmallVector<DbgValueInst *, 1> DbgValues;
  findDbgValues(DbgValues, Obj);
  for (auto *DVI : DbgValues) {
    auto *LV = DVI->getVariable();
    if (LV->getName() != "") {
      add(ID, LV->getLine(), LV->getFilename(), LV->getDirectory(),
          LV->getName());
      return ID;
    }
  }

  add(ID, -1, "", "", Obj->getName());
  return ID;
}

PointerType *ObjectTable::getPointerType(LLVMContext &C) {
  return PointerType::get(getSourceLocStructType(C), 0);
}

StructType *ObjectTable::getSourceLocStructType(LLVMContext &C) {
  return StructType::get(
      /* Name */ PointerType::get(IntegerType::get(C, 8), 0),
      /* Line */ IntegerType::get(C, 32),
      /* File */ PointerType::get(IntegerType::get(C, 8), 0));
}

void ObjectTable::add(uint64_t ID, int32_t Line,
                      StringRef Filename, StringRef Directory,
                      StringRef Name) {
  assert(LocalIdToSourceLocationMap.find(ID) ==
             LocalIdToSourceLocationMap.end() &&
         "Id already exists in FED table.");
  LocalIdToSourceLocationMap[ID] = {Name, Line, Filename, Directory};
}

// The order of arguments to ConstantStruct::get() must match the
// obj_source_loc_t type in csan.h.
static void addObjTableEntries(SmallVectorImpl<Constant *> &TableEntries,
                               StructType *TableType, Constant *Name,
                               Constant *Line, Constant *File) {
  TableEntries.push_back(ConstantStruct::get(TableType, Name, Line, File));
}

Constant *ObjectTable::insertIntoModule(Module &M) const {
  LLVMContext &C = M.getContext();
  StructType *TableType = getSourceLocStructType(C);
  IntegerType *Int32Ty = IntegerType::get(C, 32);
  Constant *Zero = ConstantInt::get(Int32Ty, 0);
  Value *GepArgs[] = {Zero, Zero};
  SmallVector<Constant *, 6> TableEntries;

  // Get the object-table entries for each ID.
  for (uint64_t LocalID = 0; LocalID < IdCounter; ++LocalID) {
    const SourceLocation &E = LocalIdToSourceLocationMap.find(LocalID)->second;
    // Source line
    Constant *Line = ConstantInt::get(Int32Ty, E.Line);
    // Source file
    Constant *File;
    {
      std::string Filename = E.Filename.str();
      if (!E.Directory.empty())
        Filename = E.Directory.str() + "/" + Filename;
      File = getObjectStrGV(M, Filename, "__csi_unit_filename_");
    }
    // Variable name
    Constant *Name = getObjectStrGV(M, E.Name, "__csi_unit_object_name_");

    // Add entry to the table
    addObjTableEntries(TableEntries, TableType, Name, Line, File);
  }

  ArrayType *TableArrayType = ArrayType::get(TableType, TableEntries.size());
  Constant *Table = ConstantArray::get(TableArrayType, TableEntries);
  GlobalVariable *GV =
    new GlobalVariable(M, TableArrayType, false, GlobalValue::InternalLinkage,
                       Table, CsiUnitObjTableName);
  return ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
}

namespace {

using SCCNodeSet = SmallSetVector<Function *, 8>;

} // end anonymous namespace

static bool InstrBreaksNoRecurse(Instruction &I, const SCCNodeSet &SCCNodes,
                                 SmallPtrSetImpl<const Function *>
                                 &LocalNoRecurFns) {
  Function *F = *SCCNodes.begin();
  if (F->doesNotRecurse())
    return false;

  if (auto CS = CallSite(&I)) {
    if (isa<DbgInfoIntrinsic>(I))
      return false;

    if (isDetachedRethrow(&I))
      return false;

    const Function *Callee = CS.getCalledFunction();
    if (!Callee || Callee == F || (!Callee->doesNotRecurse() &&
                                   !LocalNoRecurFns.count(Callee))) {
      if (Callee && Callee != F) {
        switch (Callee->getIntrinsicID()) {
        default: return true;
        case Intrinsic::annotation:
        case Intrinsic::assume:
        case Intrinsic::sideeffect:
        case Intrinsic::invariant_start:
        case Intrinsic::invariant_end:
        case Intrinsic::launder_invariant_group:
        case Intrinsic::strip_invariant_group:
        case Intrinsic::is_constant:
        case Intrinsic::lifetime_start:
        case Intrinsic::lifetime_end:
        case Intrinsic::objectsize:
        case Intrinsic::ptr_annotation:
        case Intrinsic::var_annotation:
        case Intrinsic::experimental_gc_result:
        case Intrinsic::experimental_gc_relocate:
        case Intrinsic::coro_alloc:
        case Intrinsic::coro_begin:
        case Intrinsic::coro_free:
        case Intrinsic::coro_end:
        case Intrinsic::coro_frame:
        case Intrinsic::coro_size:
        case Intrinsic::coro_suspend:
        case Intrinsic::coro_param:
        case Intrinsic::coro_subfn_addr:
        case Intrinsic::syncregion_start:
          return false;
        }
      }
      return true;
    }
  }
  return false;
}

bool CilkSanitizerImpl::run() {
  // Initialize components of the CSI and Cilksan system.
  initializeCsi();
  initializeFEDTables();
  initializeCsanObjectTables();
  initializeCsanHooks();

  // Evaluate the SCC's in the callgraph in post order to support
  // interprocedural analysis of potential races in the module.
  SmallVector<Function *, 16> InstrumentedFunctions;

  // Fill SCCNodes with the elements of the SCC. Used for quickly looking up
  // whether a given CallGraphNode is in this SCC. Also track whether there are
  // any external or opt-none nodes that will prevent us from optimizing any
  // part of the SCC.
  for (scc_iterator<CallGraph *> I = scc_begin(CG); !I.isAtEnd(); ++I) {
    const std::vector<CallGraphNode *> &SCC = *I;
    SCCNodeSet SCCNodes;
    for (CallGraphNode *N : SCC) {
      Function *F = N->getFunction();
      if (F)
        SCCNodes.insert(F);
    }
    // Infer our own version of the norecurse attribute.  The norecurse
    // attribute requires an exact definition of the function, and therefore
    // does not get inferred on functions with weak or linkonce linkage.
    // However, CilkSanitizer only requires this attribute in propagating
    // analysis information across function boundaries.  Any alternative
    // implementation of said function can simply propagate such information
    // differently.  So CilkSanitizer infers the norecurse attribute itself,
    // without the requirement of an exact definition.
    AttributeInferer AI;
    AI.registerAttrInference(AttributeInferer::InferenceDescriptor{
        Attribute::NoRecurse,
            // Skip functions already marked norecurse.
            [](const Function &F) { return F.doesNotRecurse(); },
            // Instructions that break NoRecurse
            [this, SCCNodes](Instruction &I) {
              return InstrBreaksNoRecurse(I, SCCNodes, LocalNoRecurseFunctions);
            },
            [this](Function &F) {
              LLVM_DEBUG(dbgs() << "Setting function " << F.getName()
                                << " as locally norecurse.\n");
              LocalNoRecurseFunctions.insert(&F);
            },
            /* RequiresExactDefinition = */ false});
    // Derive any local function attributes we want.
    AI.run(SCCNodes);
  }

  // Instrument functions.
  for (scc_iterator<CallGraph *> I = scc_begin(CG); !I.isAtEnd(); ++I) {
    const std::vector<CallGraphNode *> &SCC = *I;
    for (CallGraphNode *N : SCC) {
      if (Function *F = N->getFunction())
        if (instrumentFunctionUsingRI(*F))
          InstrumentedFunctions.push_back(F);
    }
  }
  // After all functions have been analyzed and instrumented, update their
  // attributes.
  for (Function *F : InstrumentedFunctions) {
    updateInstrumentedFnAttrs(*F);
    F->removeFnAttr(Attribute::SanitizeCilk);
  }

  CSIImpl::collectUnitFEDTables();
  collectUnitFEDTables();
  collectUnitObjectTables();
  finalizeCsi();
  return true;
}

void CilkSanitizerImpl::initializeCsanObjectTables() {
  LoadObj = ObjectTable(M, CsiLoadBaseIdName);
  StoreObj = ObjectTable(M, CsiStoreBaseIdName);
  AllocaObj = ObjectTable(M, CsiAllocaBaseIdName);
  AllocFnObj = ObjectTable(M, CsiAllocFnBaseIdName);
}

// Create a struct type to match the unit_obj_entry_t type in csanrt.c.
StructType *CilkSanitizerImpl::getUnitObjTableType(
    LLVMContext &C, PointerType *EntryPointerType) {
  return StructType::get(IntegerType::get(C, 64), EntryPointerType);
}

Constant *CilkSanitizerImpl::objTableToUnitObjTable(
    Module &M, StructType *UnitObjTableType, ObjectTable &ObjTable) {
  Constant *NumEntries =
    ConstantInt::get(IntegerType::get(M.getContext(), 64), ObjTable.size());
  // Constant *BaseIdPtr =
  //   ConstantExpr::getPointerCast(FedTable.baseId(),
  //                                Type::getInt8PtrTy(M.getContext(), 0));
  Constant *InsertedTable = ObjTable.insertIntoModule(M);
  return ConstantStruct::get(UnitObjTableType, NumEntries,
                             InsertedTable);
}

void CilkSanitizerImpl::collectUnitObjectTables() {
  LLVMContext &C = M.getContext();
  StructType *UnitObjTableType =
      getUnitObjTableType(C, ObjectTable::getPointerType(C));

  UnitObjTables.push_back(
      objTableToUnitObjTable(M, UnitObjTableType, LoadObj));
  UnitObjTables.push_back(
      objTableToUnitObjTable(M, UnitObjTableType, StoreObj));
  UnitObjTables.push_back(
      objTableToUnitObjTable(M, UnitObjTableType, AllocaObj));
  UnitObjTables.push_back(
      objTableToUnitObjTable(M, UnitObjTableType, AllocFnObj));
}

CallInst *CilkSanitizerImpl::createRTUnitInitCall(IRBuilder<> &IRB) {
  LLVMContext &C = M.getContext();

  StructType *UnitFedTableType =
      getUnitFedTableType(C, FrontEndDataTable::getPointerType(C));
  StructType *UnitObjTableType =
      getUnitObjTableType(C, ObjectTable::getPointerType(C));

  // Lookup __csirt_unit_init
  SmallVector<Type *, 4> InitArgTypes({IRB.getInt8PtrTy(),
                                       PointerType::get(UnitFedTableType, 0),
                                       PointerType::get(UnitObjTableType, 0),
                                       InitCallsiteToFunction->getType()});
  FunctionType *InitFunctionTy =
      FunctionType::get(IRB.getVoidTy(), InitArgTypes, false);
  RTUnitInit = M.getOrInsertFunction(CsiRtUnitInitName, InitFunctionTy);
  assert(RTUnitInit);

  ArrayType *UnitFedTableArrayType =
      ArrayType::get(UnitFedTableType, UnitFedTables.size());
  Constant *FEDTable = ConstantArray::get(UnitFedTableArrayType, UnitFedTables);
  GlobalVariable *FEDGV = new GlobalVariable(M, UnitFedTableArrayType, false,
                                             GlobalValue::InternalLinkage, FEDTable,
                                             CsiUnitFedTableArrayName);

  ArrayType *UnitObjTableArrayType =
      ArrayType::get(UnitObjTableType, UnitObjTables.size());
  Constant *ObjTable = ConstantArray::get(UnitObjTableArrayType, UnitObjTables);
  GlobalVariable *ObjGV = new GlobalVariable(M, UnitObjTableArrayType, false,
                                             GlobalValue::InternalLinkage, ObjTable,
                                             CsiUnitObjTableArrayName);

  Constant *Zero = ConstantInt::get(IRB.getInt32Ty(), 0);
  Value *GepArgs[] = {Zero, Zero};

  // Insert call to __csirt_unit_init
  return IRB.CreateCall(
      RTUnitInit,
      {IRB.CreateGlobalStringPtr(M.getName()),
          ConstantExpr::getGetElementPtr(FEDGV->getValueType(), FEDGV, GepArgs),
          ConstantExpr::getGetElementPtr(ObjGV->getValueType(), ObjGV, GepArgs),
          InitCallsiteToFunction});
}

// Initialize all instrumentation hooks that are specific to CilkSanitizer.
void CilkSanitizerImpl::initializeCsanHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *FuncPropertyTy = CsiFuncProperty::getType(C);
  Type *FuncExitPropertyTy = CsiFuncExitProperty::getType(C);
  Type *TaskPropertyTy = CsiTaskProperty::getType(C);
  Type *TaskExitPropertyTy = CsiTaskExitProperty::getType(C);
  Type *LoadPropertyTy = CsiLoadStoreProperty::getType(C);
  Type *StorePropertyTy = CsiLoadStoreProperty::getType(C);
  Type *CallPropertyTy = CsiCallProperty::getType(C);
  Type *LoopPropertyTy = CsiLoopProperty::getType(C);
  Type *AllocFnPropertyTy = CsiAllocFnProperty::getType(C);
  Type *FreePropertyTy = CsiFreeProperty::getType(C);
  Type *RetType = IRB.getVoidTy();
  Type *AddrType = IRB.getInt8PtrTy();
  Type *NumBytesType = IRB.getInt32Ty();
  Type *LargeNumBytesType = IntptrTy;
  Type *IDType = IRB.getInt64Ty();

  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::NoCapture);
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::ReadNone);
    FnAttrs = FnAttrs.addParamAttribute(C, 2, Attribute::NoCapture);
    FnAttrs = FnAttrs.addParamAttribute(C, 2, Attribute::ReadNone);
    CsanFuncEntry = M.getOrInsertFunction("__csan_func_entry", FnAttrs, RetType,
                                          /* func_id */ IDType,
                                          /* frame_ptr */ AddrType,
                                          /* stack_ptr */ AddrType,
                                          FuncPropertyTy);
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    CsanFuncExit = M.getOrInsertFunction("__csan_func_exit", FnAttrs, RetType,
                                         /* func_exit_id */ IDType,
                                         /* func_id */ IDType,
                                         FuncExitPropertyTy);
  }

  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::NoCapture);
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::ReadNone);
    CsanRead = M.getOrInsertFunction("__csan_load", FnAttrs, RetType, IDType,
                                     AddrType, NumBytesType, LoadPropertyTy);
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::NoCapture);
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::ReadNone);
    CsanWrite = M.getOrInsertFunction("__csan_store", FnAttrs, RetType, IDType,
                                      AddrType, NumBytesType, StorePropertyTy);
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::NoCapture);
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::ReadNone);
    CsanLargeRead = M.getOrInsertFunction("__csan_large_load", FnAttrs, RetType,
                                          IDType, AddrType, LargeNumBytesType,
                                          LoadPropertyTy);
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::NoCapture);
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::ReadNone);
    CsanLargeWrite = M.getOrInsertFunction("__csan_large_store", FnAttrs,
                                           RetType, IDType, AddrType,
                                           LargeNumBytesType, StorePropertyTy);
  }

  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    CsanBeforeCallsite = M.getOrInsertFunction("__csan_before_call", FnAttrs,
                                               IRB.getVoidTy(), IDType,
                                               /*callee func_id*/ IDType,
                                               IRB.getInt8Ty(), CallPropertyTy);
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    CsanAfterCallsite = M.getOrInsertFunction("__csan_after_call", FnAttrs,
                                              IRB.getVoidTy(), IDType, IDType,
                                              IRB.getInt8Ty(), CallPropertyTy);
  }

  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    CsanDetach = M.getOrInsertFunction("__csan_detach", FnAttrs, RetType,
                                       /* detach_id */ IDType,
                                       /* sync_reg */ IRB.getInt8Ty());
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    FnAttrs = FnAttrs.addParamAttribute(C, 2, Attribute::NoCapture);
    FnAttrs = FnAttrs.addParamAttribute(C, 2, Attribute::ReadNone);
    FnAttrs = FnAttrs.addParamAttribute(C, 3, Attribute::NoCapture);
    FnAttrs = FnAttrs.addParamAttribute(C, 3, Attribute::ReadNone);
    CsanTaskEntry = M.getOrInsertFunction("__csan_task", FnAttrs, RetType,
                                          /* task_id */ IDType,
                                          /* detach_id */ IDType,
                                          /* frame_ptr */ AddrType,
                                          /* stack_ptr */ AddrType,
                                          TaskPropertyTy);
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                         Attribute::InaccessibleMemOnly);
    CsanTaskExit = M.getOrInsertFunction("__csan_task_exit", FnAttrs, RetType,
                                         /* task_exit_id */ IDType,
                                         /* task_id */ IDType,
                                         /* detach_id */ IDType,
                                         /* sync_reg */ IRB.getInt8Ty(),
                                         TaskExitPropertyTy);
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    CsanDetachContinue = M.getOrInsertFunction("__csan_detach_continue",
                                               FnAttrs, RetType,
                                               /* detach_continue_id */ IDType,
                                               /* detach_id */ IDType);
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    CsanSync = M.getOrInsertFunction("__csan_sync", FnAttrs, RetType, IDType,
                                     /* sync_reg */ IRB.getInt8Ty());
  }

  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::NoCapture);
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::ReadNone);
    FnAttrs = FnAttrs.addParamAttribute(C, 5, Attribute::NoCapture);
    FnAttrs = FnAttrs.addParamAttribute(C, 5, Attribute::ReadNone);
    CsanAfterAllocFn = M.getOrInsertFunction(
        "__csan_after_allocfn", FnAttrs, RetType, IDType,
        /* new ptr */ AddrType, /* size */ LargeNumBytesType,
        /* num elements */ LargeNumBytesType, /* alignment */ LargeNumBytesType,
        /* old ptr */ AddrType, /* property */ AllocFnPropertyTy);
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::NoCapture);
    FnAttrs = FnAttrs.addParamAttribute(C, 1, Attribute::ReadNone);
    CsanAfterFree = M.getOrInsertFunction("__csan_after_free", FnAttrs, RetType,
                                          IDType, AddrType,
                                          /* property */ FreePropertyTy);
  }

  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    CsanDisableChecking = M.getOrInsertFunction("__cilksan_disable_checking",
                                                FnAttrs, RetType);
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                         Attribute::InaccessibleMemOnly);
    CsanEnableChecking = M.getOrInsertFunction("__cilksan_enable_checking",
                                               FnAttrs, RetType);
  }

  Type *SuppressionFlagTy = IRB.getInt64Ty();
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOrArgMemOnly);
    FnAttrs = FnAttrs.addParamAttribute(C, 0, Attribute::NoCapture);
    GetSuppressionFlag = M.getOrInsertFunction(
        "__csan_get_suppression_flag", FnAttrs, RetType,
        PointerType::get(SuppressionFlagTy, 0), IDType, IRB.getInt8Ty());
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    SetSuppressionFlag = M.getOrInsertFunction("__csan_set_suppression_flag",
                                               FnAttrs, RetType,
                                               SuppressionFlagTy, IDType);
  }

  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    CsanBeforeLoop = M.getOrInsertFunction(
        "__csan_before_loop", FnAttrs, IRB.getVoidTy(), IDType,
        IRB.getInt64Ty(), LoopPropertyTy);
  }
  {
    AttributeList FnAttrs;
    FnAttrs = FnAttrs.addAttribute(C, AttributeList::FunctionIndex,
                                   Attribute::InaccessibleMemOnly);
    CsanAfterLoop = M.getOrInsertFunction("__csan_after_loop", FnAttrs,
                                          IRB.getVoidTy(), IDType,
                                          IRB.getInt8Ty(), LoopPropertyTy);
  }

  // Cilksan-specific attributes on CSI hooks
  Function *CsiAfterAllocaFn = cast<Function>(CsiAfterAlloca.getCallee());
  CsiAfterAllocaFn->addParamAttr(1, Attribute::NoCapture);
  CsiAfterAllocaFn->addParamAttr(1, Attribute::ReadNone);
  CsiAfterAllocaFn->addFnAttr(Attribute::InaccessibleMemOnly);
}

static BasicBlock *SplitOffPreds(
    BasicBlock *BB, SmallVectorImpl<BasicBlock *> &Preds, DominatorTree *DT) {
  if (BB->isLandingPad()) {
    SmallVector<BasicBlock *, 2> NewBBs;
    SplitLandingPadPredecessors(BB, Preds, ".csi-split-lp", ".csi-split",
                                NewBBs, DT);
    return NewBBs[1];
  }

  SplitBlockPredecessors(BB, Preds, ".csi-split", DT);
  return BB;
}

// Setup each block such that all of its predecessors belong to the same CSI ID
// space.
static void setupBlock(BasicBlock *BB, DominatorTree *DT,
                       const TargetLibraryInfo *TLI) {
  if (BB->getUniquePredecessor())
    return;

  SmallVector<BasicBlock *, 4> DetachPreds;
  SmallVector<BasicBlock *, 4> DetRethrowPreds;
  SmallVector<BasicBlock *, 4> SyncPreds;
  SmallVector<BasicBlock *, 4> AllocFnPreds;
  SmallVector<BasicBlock *, 4> InvokePreds;
  bool HasOtherPredTypes = false;
  unsigned NumPredTypes = 0;

  // Partition the predecessors of the landing pad.
  for (BasicBlock *Pred : predecessors(BB)) {
    if (isa<DetachInst>(Pred->getTerminator()))
      DetachPreds.push_back(Pred);
    else if (isDetachedRethrow(Pred->getTerminator()))
      DetRethrowPreds.push_back(Pred);
    else if (isa<SyncInst>(Pred->getTerminator()))
      SyncPreds.push_back(Pred);
    else if (isAllocationFn(Pred->getTerminator(), TLI, false, true))
      AllocFnPreds.push_back(Pred);
    else if (isa<InvokeInst>(Pred->getTerminator()))
      InvokePreds.push_back(Pred);
    else
      HasOtherPredTypes = true;
  }

  NumPredTypes = static_cast<unsigned>(!DetachPreds.empty()) +
    static_cast<unsigned>(!DetRethrowPreds.empty()) +
    static_cast<unsigned>(!SyncPreds.empty()) +
    static_cast<unsigned>(!AllocFnPreds.empty()) +
    static_cast<unsigned>(!InvokePreds.empty()) +
    static_cast<unsigned>(HasOtherPredTypes);

  BasicBlock *BBToSplit = BB;
  // Split off the predecessors of each type.
  if (!DetachPreds.empty() && NumPredTypes > 1) {
    BBToSplit = SplitOffPreds(BBToSplit, DetachPreds, DT);
    NumPredTypes--;
  }
  if (!DetRethrowPreds.empty() && NumPredTypes > 1) {
    BBToSplit = SplitOffPreds(BBToSplit, DetRethrowPreds, DT);
    NumPredTypes--;
  }
  if (!SyncPreds.empty() && NumPredTypes > 1) {
    BBToSplit = SplitOffPreds(BBToSplit, SyncPreds, DT);
    NumPredTypes--;
  }
  if (!AllocFnPreds.empty() && NumPredTypes > 1) {
    BBToSplit = SplitOffPreds(BBToSplit, AllocFnPreds, DT);
    NumPredTypes--;
  }
  if (!InvokePreds.empty() && NumPredTypes > 1) {
    BBToSplit = SplitOffPreds(BBToSplit, InvokePreds, DT);
    NumPredTypes--;
  }
}

// Setup all basic blocks such that each block's predecessors belong entirely to
// one CSI ID space.
void CilkSanitizerImpl::setupBlocks(Function &F, DominatorTree *DT) {
  SmallPtrSet<BasicBlock *, 8> BlocksToSetup;
  for (BasicBlock &BB : F) {
    if (BB.isLandingPad())
      BlocksToSetup.insert(&BB);

    if (InvokeInst *II = dyn_cast<InvokeInst>(BB.getTerminator()))
      BlocksToSetup.insert(II->getNormalDest());
    else if (SyncInst *SI = dyn_cast<SyncInst>(BB.getTerminator()))
      BlocksToSetup.insert(SI->getSuccessor(0));
  }

  for (BasicBlock *BB : BlocksToSetup)
    setupBlock(BB, DT, TLI);
}

// Do not instrument known races/"benign races" that come from compiler
// instrumentation. The user has no way of suppressing them.
static bool shouldInstrumentReadWriteFromAddress(const Module *M, Value *Addr) {
  // Peel off GEPs and BitCasts.
  Addr = Addr->stripInBoundsOffsets();

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Addr)) {
    if (GV->hasSection()) {
      StringRef SectionName = GV->getSection();
      // Check if the global is in the PGO counters section.
      auto OF = Triple(M->getTargetTriple()).getObjectFormat();
      if (SectionName.endswith(
              getInstrProfSectionName(IPSK_cnts, OF, /*AddSegmentInfo=*/false)))
        return false;
    }

    // Check if the global is private gcov data.
    if (GV->getName().startswith("__llvm_gcov") ||
        GV->getName().startswith("__llvm_gcda"))
      return false;
  }

  // Do not instrument acesses from different address spaces; we cannot deal
  // with them.
  if (Addr) {
    Type *PtrTy = cast<PointerType>(Addr->getType()->getScalarType());
    if (PtrTy->getPointerAddressSpace() != 0)
      return false;
  }

  return true;
}

/// Returns true if Addr can only refer to a locally allocated base object, that
/// is, an object created via an AllocaInst or an AllocationFn.
static bool LocalBaseObj(Value *Addr, const DataLayout &DL, LoopInfo *LI,
                         const TargetLibraryInfo *TLI) {
  // If we don't have an address, give up.
  if (!Addr)
    return false;

  // Get the base objects that this address might refer to.
  SmallVector<Value *, 1> BaseObjs;
  GetUnderlyingObjects(Addr, BaseObjs, DL, LI, 0);

  // If we could not determine the base objects, conservatively return false.
  if (BaseObjs.empty())
    return false;

  // If any base object is not an alloca or allocation function, then it's not
  // local.
  for (const Value *BaseObj : BaseObjs) {
    // if (!isa<AllocaInst>(BaseObj) && !isAllocationFn(BaseObj, TLI)) {
    if (isa<AllocaInst>(BaseObj) || isNoAliasCall(BaseObj))
      continue;

    if (const Argument *A = dyn_cast<Argument>(BaseObj))
      if (A->hasByValAttr())
        continue;

    LLVM_DEBUG(dbgs() << "Non-local base object " << *BaseObj << "\n");
    return false;
  }

  return true;
}

// /// Returns true if Addr can only refer to a locally allocated base object, that
// /// is, an object created via an AllocaInst or an AllocationFn.
// static bool LocalBaseObj(const CallBase *CS, const DataLayout &DL,
//                          LoopInfo *LI, const TargetLibraryInfo *TLI) {
//   // Check whether all pointer arguments point to local memory, and
//   // ignore calls that only access local memory.
//   for (auto CI = CS->arg_begin(), CE = CS->arg_end(); CI != CE; ++CI) {
//     Value *Arg = *CI;
//     if (!Arg->getType()->isPtrOrPtrVectorTy())
//       continue;

//     if (!LocalBaseObj(Arg, DL, LI, TLI))
//       return false;
//   }
//   return true;
// }

// Examine the uses of a Instruction AI to determine if it is used in a subtask.
// This method assumes that AI is an allocation instruction, i.e., either an
// AllocaInst or an AllocationFn.
static bool MightHaveDetachedUse(const Value *V, const TaskInfo &TI) {
  // Get the task for this allocation.
  const Task *AllocTask = nullptr;
  if (const Instruction *I = dyn_cast<Instruction>(V))
    AllocTask = TI.getTaskFor(I->getParent());
  else if (const Argument *A = dyn_cast<Argument>(V))
    AllocTask = TI.getTaskFor(&A->getParent()->getEntryBlock());

  // assert(AllocTask && "Null task for instruction.");
  if (!AllocTask) {
    LLVM_DEBUG(dbgs() << "MightHaveDetachedUse: No task found for given value "
               << *V << "\n");
    return false;
  }

  if (AllocTask->isSerial())
    // Alloc AI cannot be used in a subtask if its enclosing task is serial.
    return false;

  SmallVector<const Use *, 20> Worklist;
  SmallSet<const Use *, 20> Visited;

  // Add all uses of AI to the worklist.
  for (const Use &U : V->uses()) {
    Visited.insert(&U);
    Worklist.push_back(&U);
  }

  // Evaluate each use of AI.
  while (!Worklist.empty()) {
    const Use *U = Worklist.pop_back_val();

    // Check if this use of AI is in a different task from the allocation.
    Instruction *I = cast<Instruction>(U->getUser());
    LLVM_DEBUG(dbgs() << "\tExamining use: " << *I << "\n");
    if (AllocTask != TI.getTaskFor(I->getParent())) {
      assert(TI.getTaskFor(I->getParent()) != AllocTask->getParentTask() &&
             "Use of alloca appears in a parent task of that alloca");
      // Because the use of AI cannot appear in a parent task of AI, it must be
      // in a subtask.  In particular, the use cannot be in a shared-EH spindle.
      return true;
    }

    // If the pointer to AI is transformed using one of the following
    // operations, add uses of the transformed pointer to the worklist.
    switch (I->getOpcode()) {
    case Instruction::BitCast:
    case Instruction::GetElementPtr:
    case Instruction::PHI:
    case Instruction::Select:
    case Instruction::AddrSpaceCast:
      for (Use &UU : I->uses())
        if (Visited.insert(&UU).second)
          Worklist.push_back(&UU);
      break;
    default:
      break;
    }
  }
  return false;
}

/// Returns true if accesses on Addr could race due to pointer capture.
static bool PossibleRaceByCapture(Value *Addr, const DataLayout &DL,
                                  const TaskInfo &TI, LoopInfo *LI) {
  if (isa<GlobalValue>(Addr))
    // For this analysis, we consider all global values to be captured.
    return true;

  // Check for detached uses of the underlying base objects.
  SmallVector<Value *, 1> BaseObjs;
  GetUnderlyingObjects(Addr, BaseObjs, DL, LI, 0);

  // If we could not determine the base objects, conservatively return true.
  if (BaseObjs.empty())
    return true;

  for (const Value *BaseObj : BaseObjs) {
    // Skip any null objects
    if (const Constant *C = dyn_cast<Constant>(BaseObj)) {
      // if (C->isNullValue())
      //   continue;
      // Is this value a constant that cannot be derived from any pointer
      // value (we need to exclude constant expressions, for example, that
      // are formed from arithmetic on global symbols).
      bool IsNonPtrConst = isa<ConstantInt>(C) || isa<ConstantFP>(C) ||
                           isa<ConstantPointerNull>(C) ||
                           isa<ConstantDataVector>(C) || isa<UndefValue>(C);
      if (IsNonPtrConst)
        continue;
    }

    // If the base object is not an instruction, conservatively return true.
    if (!isa<Instruction>(BaseObj)) {
      // From BasicAliasAnalysis.cpp: If this is an argument that corresponds to
      // a byval or noalias argument, then it has not escaped before entering
      // the function.
      if (const Argument *A = dyn_cast<Argument>(BaseObj)) {
        if (!A->hasByValAttr() && !A->hasNoAliasAttr())
          return true;
      } else
        return true;
    }

    // If the base object might have a detached use, return true.
    if (MightHaveDetachedUse(BaseObj, TI))
      return true;
  }

  // Perform normal pointer-capture analysis.
  if (PointerMayBeCaptured(Addr, false, false))
    return true;

  return false;
}

// /// Returns true if any address referenced by the callsite could race due to
// /// pointer capture.
// static bool PossibleRaceByCapture(const CallBase *CS, const DataLayout &DL,
//                                   const TaskInfo &TI, LoopInfo *LI) {
//   // Check whether all pointer arguments point to local memory, and
//   // ignore calls that only access local memory.
//   for (auto CI = CS->arg_begin(), CE = CS->arg_end(); CI != CE; ++CI) {
//     Value *Arg = *CI;
//     if (!Arg->getType()->isPtrOrPtrVectorTy())
//       continue;

//     if (PossibleRaceByCapture(Arg, DL, TI, LI))
//       return true;
//   }
//   return false;
// }

static bool unknownObjectUses(Value *Addr, const DataLayout &DL, LoopInfo *LI,
                              const TargetLibraryInfo *TLI) {
  // Perform normal pointer-capture analysis.
  if (PointerMayBeCaptured(Addr, true, false))
    return true;

  // Check for detached uses of the underlying base objects.
  SmallVector<Value *, 1> BaseObjs;
  GetUnderlyingObjects(Addr, BaseObjs, DL, LI, 0);

  // If we could not determine the base objects, conservatively return true.
  if (BaseObjs.empty())
    return true;

  // If the base object is not an allocation function, return true.
  for (const Value *BaseObj : BaseObjs)
    if (!isAllocationFn(BaseObj, TLI))
      return true;

  return false;
}

void CilkSanitizerImpl::chooseInstructionsToInstrument(
    SmallVectorImpl<Instruction *> &Local, SmallVectorImpl<Instruction *> &All,
    const TaskInfo &TI, LoopInfo &LI) {
  SmallSet<Value*, 8> WriteTargets;
  // Iterate from the end.
  for (Instruction *I : reverse(Local)) {
    if (StoreInst *Store = dyn_cast<StoreInst>(I)) {
      Value *Addr = Store->getPointerOperand();
      if (!shouldInstrumentReadWriteFromAddress(I->getModule(), Addr))
        continue;
      WriteTargets.insert(Addr);
    } else {
      LoadInst *Load = cast<LoadInst>(I);
      Value *Addr = Load->getPointerOperand();
      if (!shouldInstrumentReadWriteFromAddress(I->getModule(), Addr))
        continue;
      if (WriteTargets.count(Addr)) {
        // We will write to this temp, so no reason to analyze the read.
        NumOmittedReadsBeforeWrite++;
        continue;
      }
      if (addrPointsToConstantData(Addr)) {
        // Addr points to some constant data -- it can not race with any writes.
        NumOmittedReadsFromConstants++;
        continue;
      }
    }
    Value *Addr = isa<StoreInst>(*I)
        ? cast<StoreInst>(I)->getPointerOperand()
        : cast<LoadInst>(I)->getPointerOperand();
    if (LocalBaseObj(Addr, DL, &LI, TLI) &&
        !PossibleRaceByCapture(Addr, DL, TI, &LI)) {
      // The variable is addressable but not captured, so it cannot be
      // referenced from a different thread and participate in a data race
      // (see llvm/Analysis/CaptureTracking.h for details).
      NumOmittedNonCaptured++;
      continue;
    }
    LLVM_DEBUG(dbgs() << "Pushing " << *I << "\n");
    All.push_back(I);
  }
  Local.clear();
}

// Helper function do determine if the call or invoke instruction Inst should be
// skipped when examining calls that affect race detection.
bool CilkSanitizerImpl::simpleCallCannotRace(const Instruction &I) {
  return callsPlaceholderFunction(I);
}

// Helper function to get the ID of a function being called.  These IDs are
// stored in separate global variables in the program.  This method will create
// a new global variable for the Callee's ID if necessary.
Value *CilkSanitizerImpl::GetCalleeFuncID(const Function *Callee,
                                          IRBuilder<> &IRB) {
  if (!Callee)
    // Unknown targets (i.e., indirect calls) are always unknown.
    return IRB.getInt64(CsiCallsiteUnknownTargetId);

  std::string GVName =
    CsiFuncIdVariablePrefix + Callee->getName().str();
  GlobalVariable *FuncIdGV = M.getNamedGlobal(GVName);
  if (!FuncIdGV) {
    FuncIdGV = dyn_cast<GlobalVariable>(M.getOrInsertGlobal(GVName,
                                                            IRB.getInt64Ty()));
    assert(FuncIdGV);
    FuncIdGV->setConstant(false);
    if (Options.jitMode && !Callee->empty())
      FuncIdGV->setLinkage(Callee->getLinkage());
    else
      FuncIdGV->setLinkage(GlobalValue::WeakAnyLinkage);
    FuncIdGV->setInitializer(IRB.getInt64(CsiCallsiteUnknownTargetId));
  }
  return IRB.CreateLoad(FuncIdGV);
}

//------------------------------------------------------------------------------
// SimpleInstrumentor methods, which do not do static race detection.
//------------------------------------------------------------------------------

bool CilkSanitizerImpl::SimpleInstrumentor::InstrumentSimpleInstructions(
    SmallVectorImpl<Instruction *> &Instructions) {
  bool Result = false;
  for (Instruction *I : Instructions) {
    bool LocalResult = false;
    if (isa<LoadInst>(I) || isa<StoreInst>(I))
      LocalResult |= CilkSanImpl.instrumentLoadOrStore(I);
    else if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I))
      LocalResult |= CilkSanImpl.instrumentAtomic(I);
    else
      dbgs() << "[Cilksan] Unknown simple instruction: " << *I << "\n";

    if (LocalResult) {
      Result |= LocalResult;
      // Record the detaches for the task containing this instruction.  These
      // detaches need to be instrumented.
      getDetachesForInstruction(I);
    }
  }
  return Result;
}

bool CilkSanitizerImpl::SimpleInstrumentor::InstrumentAnyMemIntrinsics(
    SmallVectorImpl<Instruction *> &MemIntrinsics) {
  bool Result = false;
  for (Instruction *I : MemIntrinsics) {
    bool LocalResult = false;
    if (auto *MT = dyn_cast<AnyMemTransferInst>(I)) {
      LocalResult |= CilkSanImpl.instrumentAnyMemIntrinAcc(I, /*Src*/1);
      LocalResult |= CilkSanImpl.instrumentAnyMemIntrinAcc(I, /*Dst*/0);
    } else {
      assert(isa<AnyMemIntrinsic>(I) &&
             "InstrumentAnyMemIntrinsics operating on not a memory intrinsic.");
      LocalResult |= CilkSanImpl.instrumentAnyMemIntrinAcc(I, unsigned(-1));
    }
    if (LocalResult) {
      Result |= LocalResult;
      // Record the detaches for the task containing this instruction.  These
      // detaches need to be instrumented.
      getDetachesForInstruction(I);
    }
  }
  return Result;
}

bool CilkSanitizerImpl::SimpleInstrumentor::InstrumentCalls(
    SmallVectorImpl<Instruction *> &Calls) {
  bool Result = false;
  for (Instruction *I : Calls) {
    // Allocation-function and free calls are handled separately.
    if (isAllocationFn(I, CilkSanImpl.TLI, false, true) ||
        isFreeCall(I, CilkSanImpl.TLI))
      continue;

    bool LocalResult = false;
    LocalResult |= CilkSanImpl.instrumentCallsite(I, /*SupprVals*/nullptr);
    if (LocalResult) {
      Result |= LocalResult;
      // Record the detaches for the task containing this instruction.  These
      // detaches need to be instrumented.
      getDetachesForInstruction(I);
    }
  }
  return Result;
}

bool CilkSanitizerImpl::SimpleInstrumentor::InstrumentAncillaryInstructions(
    SmallPtrSetImpl<Instruction *> &Allocas,
    SmallPtrSetImpl<Instruction *> &AllocationFnCalls,
    SmallPtrSetImpl<Instruction *> &FreeCalls,
    DenseMap<Value *, unsigned> &SyncRegNums,
    DenseMap<BasicBlock *, unsigned> &SRCounters, const DataLayout &DL,
    const TargetLibraryInfo *TLI) {
  bool Result = false;
  SmallPtrSet<SyncInst *, 4> Syncs;
  SmallPtrSet<Loop *, 4> Loops;

  // Instrument allocas and allocation-function calls that may be involved in a
  // race.
  for (Instruction *I : Allocas) {
    // The simple instrumentor just instruments everyting
    CilkSanImpl.instrumentAlloca(I);
    getDetachesForInstruction(I);
    Result = true;
  }
  for (Instruction *I : AllocationFnCalls) {
    // The simple instrumentor just instruments everyting
    CilkSanImpl.instrumentAllocationFn(I, DT);
    getDetachesForInstruction(I);
    Result = true;
  }
  for (Instruction *I : FreeCalls) {
    // The first argument of the free call is the pointer.
    Value *Ptr = I->getOperand(0);
    // If the pointer corresponds to an allocation function call in this
    // function, then instrument it.
    if (Instruction *PtrI = dyn_cast<Instruction>(Ptr)) {
      if (AllocationFnCalls.count(PtrI)) {
        CilkSanImpl.instrumentFree(I);
        getDetachesForInstruction(I);
        Result = true;
        continue;
      }
    }
    // The simple instrumentor just instruments everyting
    CilkSanImpl.instrumentFree(I);
    getDetachesForInstruction(I);
    Result = true;
  }

  // Instrument detaches
  for (DetachInst *DI : Detaches) {
    CilkSanImpl.instrumentDetach(DI, SyncRegNums[DI->getSyncRegion()],
                                 SRCounters[DI->getDetached()], DT, TI, LI);
    Result = true;
    // Get syncs associated with this detach
    for (SyncInst *SI : CilkSanImpl.DetachToSync[DI])
      Syncs.insert(SI);

    if (CilkSanImpl.Options.InstrumentLoops) {
      // Get any loop associated with this detach.
      Loop *L = LI.getLoopFor(DI->getParent());
      if (spawnsTapirLoopBody(DI, LI, TI))
        Loops.insert(L);
    }
  }

  // Instrument associated syncs
  for (SyncInst *SI : Syncs)
    CilkSanImpl.instrumentSync(SI, SyncRegNums[SI->getSyncRegion()]);

  if (CilkSanImpl.Options.InstrumentLoops) {
    // Recursively instrument all loops
    for (Loop *L : Loops)
      CilkSanImpl.instrumentLoop(*L, TI, SyncRegNums);
  }

  return Result;
}

// TODO: Combine this redundant logic with that in Instrumentor
void CilkSanitizerImpl::SimpleInstrumentor::getDetachesForInstruction(
    Instruction *I) {
  // Get the Task for I.
  Task *T = TI.getTaskFor(I->getParent());
  // Add the ancestors of T to the set of detaches to instrument.
  while (!T->isRootTask()) {
    // Once we encounter a detach we've previously added to the set, we know
    // that all its parents are also in the set.
    if (!Detaches.insert(T->getDetach()).second)
      return;
    T = T->getParentTask();
  }
}

//------------------------------------------------------------------------------
// Instrumentor methods
//------------------------------------------------------------------------------

void CilkSanitizerImpl::Instrumentor::getDetachesForInstruction(
    Instruction *I) {
  // Get the Task for I.
  Task *T = TI.getTaskFor(I->getParent());
  // Add the ancestors of T to the set of detaches to instrument.
  while (!T->isRootTask()) {
    // Once we encounter a detach we've previously added to the set, we know
    // that all its parents are also in the set.
    if (!Detaches.insert(T->getDetach()).second)
      return;
    T = T->getParentTask();
  }
}

unsigned CilkSanitizerImpl::Instrumentor::RaceTypeToFlagVal(
    RaceInfo::RaceType RT) {
  unsigned FlagVal = static_cast<unsigned>(SuppressionVal::NoAccess);
  if (RaceInfo::isLocalRace(RT) || RaceInfo::isOpaqueRace(RT))
    FlagVal = static_cast<unsigned>(SuppressionVal::ModRef);
  if (RaceInfo::isRaceViaAncestorMod(RT))
    FlagVal |= static_cast<unsigned>(SuppressionVal::Mod);
  if (RaceInfo::isRaceViaAncestorRef(RT))
    FlagVal |= static_cast<unsigned>(SuppressionVal::Ref);
  return FlagVal;
}

static Value *getSuppressionIRValue(IRBuilder<> &IRB, unsigned SV) {
  return IRB.getInt64(SV);
}

// Insert per-argument suppressions for this function
void CilkSanitizerImpl::Instrumentor::InsertArgSuppressionFlags(Function &F,
                                                                Value *FuncId) {
  LLVM_DEBUG(dbgs() << "InsertArgSuppressionFlags: " << F.getName() << "\n");
  IRBuilder<> IRB(&*(++(cast<Instruction>(FuncId)->getIterator())));
  unsigned ArgIdx = 0;
  for (Argument &Arg : F.args()) {
    if (!Arg.getType()->isPtrOrPtrVectorTy())
      continue;

    // Create a new flag for this argument suppression.
    Value *NewFlag = IRB.CreateAlloca(getSuppressionIRValue(IRB, 0)->getType(),
                                      Arg.getType()->getPointerAddressSpace());
    Value *FinalSV;
    // If this function is main, then it has no ancestors that can create races.
    if (F.getName() == "main") {
      FinalSV = getSuppressionIRValue(IRB, RaceTypeToFlagVal(RaceInfo::None));
      IRB.CreateStore(FinalSV, NewFlag);
    } else {
      // Call the runtime function to set the value of this flag.
      IRB.CreateCall(CilkSanImpl.GetSuppressionFlag, {NewFlag, FuncId,
                                                      IRB.getInt8(ArgIdx)});

      // Incorporate local information into this suppression value.
      unsigned LocalSV = static_cast<unsigned>(SuppressionVal::NoAccess);
      if (Arg.hasNoAliasAttr())
        LocalSV |= static_cast<unsigned>(SuppressionVal::NoAlias);

      // if (!F.hasFnAttribute(Attribute::NoRecurse) &&
      if (!CilkSanImpl.FunctionIsNoRecurse(F) &&
          RI.ObjectInvolvedInRace(&Arg)) {
        LLVM_DEBUG(dbgs() << "Setting local SV in may-recurse function " <<
                   F.getName() << " for arg " << Arg << "\n");
        // This function might recursively call itself, so incorporate
        // information we have about how this function reads or writes its own
        // arguments into these suppression flags.
        ModRefInfo ArgMR = RI.GetObjectMRForRace(&Arg);
        // TODO: Possibly make these checks more precise using information we
        // get from instrumenting functions previously.
        if (isRefSet(ArgMR)) {
          LLVM_DEBUG(dbgs() << "  Setting Mod\n");
          // If ref is set, then race detection found a local instruction that
          // might write arg, so we assume arg is modified.
          LocalSV |= static_cast<unsigned>(SuppressionVal::Mod);
        }
        if (isModSet(ArgMR)) {
          LLVM_DEBUG(dbgs() << "  Setting Ref\n");
          // If mod is set, then race detection found a local instruction that
          // might read or write  arg, so we assume arg is read.
          LocalSV |= static_cast<unsigned>(SuppressionVal::Ref);
        }
      }
      // Store this local suppression value.
      FinalSV = IRB.CreateOr(getSuppressionIRValue(IRB, LocalSV),
                             IRB.CreateLoad(NewFlag));
      IRB.CreateStore(FinalSV, NewFlag);
    }
    // Associate this flag with the argument for future lookups.
    LLVM_DEBUG(dbgs() << "Recording local suppression for arg " << Arg << ": "
               << *NewFlag << "\n");
    // LocalSuppressions[&Arg] = NewFlag;
    // ArgSuppressionFlags.insert(NewFlag);
    LocalSuppressions[&Arg] = FinalSV;
    ArgSuppressionFlags.insert(FinalSV);
    ++ArgIdx;
  }

  // Record other objects known to be involved in races.
  for (auto &ObjRD : RI.getObjectMRForRace()) {
    if (isa<Instruction>(ObjRD.first)) {
      unsigned SupprVal = static_cast<unsigned>(SuppressionVal::NoAccess);
      if (isModSet(ObjRD.second))
        SupprVal |= static_cast<unsigned>(SuppressionVal::Mod);
      if (isRefSet(ObjRD.second))
        SupprVal |= static_cast<unsigned>(SuppressionVal::Ref);
      // Determine if this object is no-alias.
      //
      // TODO: Figure out what "no-alias" information we can derive for allocas.
      if (const CallBase *CB = dyn_cast<CallBase>(ObjRD.first))
        if (CB->hasRetAttr(Attribute::NoAlias))
          SupprVal |= static_cast<unsigned>(SuppressionVal::NoAlias);

      LLVM_DEBUG(dbgs() << "Setting LocalSuppressions for " << *ObjRD.first
                 << " = " << SupprVal << "\n");
      LocalSuppressions[ObjRD.first] = getSuppressionIRValue(IRB, SupprVal);
      // CilkSanImpl.ObjectMRForRace[ObjRD.first] = ObjRD.second;
    }
  }
}

bool CilkSanitizerImpl::Instrumentor::InstrumentSimpleInstructions(
    SmallVectorImpl<Instruction *> &Instructions) {
  bool Result = false;
  for (Instruction *I : Instructions) {
    bool LocalResult = false;
    // Simple instructions, such as loads, stores, or atomics, have just one
    // pointer operand, and therefore should have at most one entry of RaceData.

    // If the instruction might participate in a local or opaque race,
    // instrument it unconditionally.
    if (RI.mightRaceOpaquely(I) || RI.mightRaceLocally(I)) {
      if (isa<LoadInst>(I) || isa<StoreInst>(I))
        LocalResult |= CilkSanImpl.instrumentLoadOrStore(I);
      else if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I))
        LocalResult |= CilkSanImpl.instrumentAtomic(I);
      else
        dbgs() << "[Cilksan] Unknown simple instruction: " << *I << "\n";
    } else if (RI.mightRaceViaAncestor(I)) {
      // Otherwise, if the instruction might participate in a race via an
      // ancestor function instantiation, instrument it conditionally, based on
      // the pointer.
      //
      // Delay handling this instruction.
      DelayedSimpleInsts.push_back(I);
      LocalResult |= true;
    }

    // If any instrumentation was inserted, collect associated instructions to
    // instrument.
    if (LocalResult) {
      Result |= LocalResult;
      // Record the detaches for the task containing this instruction.  These
      // detaches need to be instrumented.
      getDetachesForInstruction(I);
    }
  }
  return Result;
}

bool CilkSanitizerImpl::Instrumentor::InstrumentAnyMemIntrinsics(
    SmallVectorImpl<Instruction *> &MemIntrinsics) {
  bool Result = false;
  for (Instruction *I : MemIntrinsics) {
    bool LocalResult = false;
    // If this instruction cannot race, skip it.
    if (!RI.mightRace(I))
      continue;

    for (const RaceInfo::RaceData &RD : RI.getRaceData(I)) {
      assert(RD.getPtr() && "No pointer for race with memory intrinsic.");
      if (RaceInfo::isLocalRace(RD.Type) || RaceInfo::isOpaqueRace(RD.Type))
        LocalResult |= CilkSanImpl.instrumentAnyMemIntrinAcc(I, RD.OperandNum);
      else if (RaceInfo::isRaceViaAncestor(RD.Type)) {
        // Delay handling this instruction.
        DelayedMemIntrinsics.push_back(std::make_pair(I, RD.OperandNum));
        LocalResult |= true;
      }
    }

    // If any instrumentation was inserted, collect associated instructions to
    // instrument.
    if (LocalResult) {
      Result |= LocalResult;
      // Record the detaches for the task containing this instruction.  These
      // detaches need to be instrumented.
      getDetachesForInstruction(I);
    }
  }
  return Result;
}

bool CilkSanitizerImpl::Instrumentor::InstrumentCalls(
    SmallVectorImpl<Instruction *> &Calls) {
  bool Result = false;
  for (Instruction *I : Calls) {
    // Allocation-function and free calls are handled separately.
    if (isAllocationFn(I, CilkSanImpl.TLI, false, true) ||
        isFreeCall(I, CilkSanImpl.TLI))
      continue;

    bool LocalResult = false;
    bool GetDetaches = false;

    // Get current race data for this call.
    RaceInfo::RaceType CallRT = RI.getRaceType(I);
    LLVM_DEBUG({
        dbgs() << "Call " << *I << ":";
        RaceInfo::printRaceType(CallRT, dbgs());
        dbgs() << "\n";
      });

    // Get update race data, if it's available.
    RaceInfo::RaceType FuncRT = CallRT;
    CallBase *CB = dyn_cast<CallBase>(I);
    if (Function *CF = CB->getCalledFunction())
      if (CilkSanImpl.FunctionRaceType.count(CF))
        FuncRT = CilkSanImpl.FunctionRaceType[CF];

    LLVM_DEBUG({
        dbgs() << "  FuncRT:";
        RaceInfo::printRaceType(FuncRT, dbgs());
        dbgs() << "\n";
      });

    // Propagate information about opaque races from function to call.
    if (!RaceInfo::isOpaqueRace(FuncRT))
      CallRT = RaceInfo::clearOpaqueRace(CallRT);

    LLVM_DEBUG({
        dbgs() << "  New CallRT:";
        RaceInfo::printRaceType(CallRT, dbgs());
        dbgs() << "\n";
      });

    // If this instruction cannot race, see if we can suppress it
    if (!RaceInfo::isRace(CallRT)) {
      // We can only suppress calls whose functions don't have local races.
      if (!RaceInfo::isLocalRace(FuncRT)) {
        if (!CB->doesNotAccessMemory())
          LocalResult |= CilkSanImpl.suppressCallsite(I);
        continue;
      // } else {
      //   GetDetaches |= CilkSanImpl.instrumentCallsite(I);
      //   // SmallPtrSet<Value *, 1> Objects;
      //   // RI.getObjectsFor(I, Objects);
      //   // for (Value *Obj : Objects) {
      //   //   CilkSanImpl.ObjectMRForRace[Obj] = ModRefInfo::ModRef;
      //   // }
      }
      // continue;
    }

    // We're going to instrument this call for potential races.  First get
    // suppression information for its arguments, if any races depend on the
    // ancestor.
    SmallVector<Value *, 8> SupprVals;
    LLVM_DEBUG(dbgs() << "Getting suppression values for " << *CB << "\n");
    IRBuilder<> IRB(I);
    if (RaceInfo::isRaceViaAncestor(CallRT)) {
      // Otherwise, if the instruction might participate in a race via an
      // ancestor function instantiation, instrument it conditionally based on
      // the pointer.
      unsigned OpIdx = 0;
      for (const Value *Op : CB->args()) {
        if (!Op->getType()->isPtrOrPtrVectorTy()) {
          ++OpIdx;
          continue;
        }
        Value *SupprVal = getSuppressionValue(I, IRB, OpIdx);
        LLVM_DEBUG({
            dbgs() << "  Op: " << *CB->getArgOperand(OpIdx) << "\n";
            dbgs() << "  Suppression value: " << *SupprVal << "\n";
          });
        SupprVals.push_back(SupprVal);
        ++OpIdx;
      }
    } else {
      // We have either an opaque race or a local race, but _not_ a race via an
      // ancestor.  We want to propagate suppression information on pointer
      // arguments, but we don't need to be pessimistic when a value can't be
      // found.
      unsigned OpIdx = 0;
      for (const Value *Op : CB->args()) {
        if (!Op->getType()->isPtrOrPtrVectorTy()) {
          ++OpIdx;
          continue;
        }
        // Value *SupprVal = IRB.getInt8(SuppressionVal::NoAccess);
        Value *SupprVal = getSuppressionValue(I, IRB, OpIdx,
                                              SuppressionVal::NoAccess,
                                              /*CheckArgs=*/ false);
        LLVM_DEBUG({
            dbgs() << "  Op: " << *CB->getArgOperand(OpIdx) << "\n";
            dbgs() << "  Suppression value: " << *SupprVal << "\n";
          });
        SupprVals.push_back(SupprVal);
        ++OpIdx;
      }
    }
    Value *CalleeID = CilkSanImpl.GetCalleeFuncID(CB->getCalledFunction(), IRB);
    // We set the suppression flags in reverse order to support stack-like
    // accesses of the flags by in-order calls to GetSuppressionFlag in the
    // callee.
    for (Value *SupprVal : reverse(SupprVals))
      IRB.CreateCall(CilkSanImpl.SetSuppressionFlag, {SupprVal, CalleeID});

    GetDetaches |= CilkSanImpl.instrumentCallsite(I, &SupprVals);

    // If any instrumentation was inserted, collect associated instructions to
    // instrument.
    Result |= LocalResult;
    if (GetDetaches) {
      Result |= GetDetaches;
      // Record the detaches for the task containing this instruction.  These
      // detaches need to be instrumented.
      getDetachesForInstruction(I);
    }
  }
  return Result;
}

Value *CilkSanitizerImpl::Instrumentor::readSuppressionVal(Value *V,
                                                           IRBuilder<> &IRB) {
  if (!ArgSuppressionFlags.count(V))
    return V;
  // Marking the load as invariant is not technically correct, because the
  // __csan_get_suppression_flag call sets the value.  But this call happens
  // once, and all subsequent loads will return the same value.
  //
  // MDNode *MD = llvm::MDNode::get(IRB.getContext(), llvm::None);
  // cast<Instruction>(Load)->setMetadata(LLVMContext::MD_invariant_load, MD);

  // TODO: See if there's a better way to annotate this load for optimization.
  // LoadInst *I = IRB.CreateLoad(V);
  // if (auto *IMD = I->getMetadata(LLVMContext::MD_invariant_group))
  //   I->setMetadata(LLVMContext::MD_invariant_group, IMD);
  // else
  //   I->setMetadata(LLVMContext::MD_invariant_group,
  //                  MDNode::get(IRB.getContext(), {}));
  Value *SV;
  if (isa<AllocaInst>(V))
    SV = IRB.CreateLoad(V);
  else
    SV = V;
  return SV;
}

// Get the memory location for this instruction and operand.
static MemoryLocation getMemoryLocation(Instruction *I, unsigned OperandNum,
                                        const TargetLibraryInfo *TLI) {
  if (auto *MI = dyn_cast<AnyMemIntrinsic>(I)) {
    if (auto *MT = dyn_cast<AnyMemTransferInst>(I)) {
      if (OperandNum == 1)
        return MemoryLocation::getForSource(MT);
    }
    return MemoryLocation::getForDest(MI);
  } else if (OperandNum == static_cast<unsigned>(-1)) {
    return MemoryLocation::get(I);
  } else {
    assert(isa<CallBase>(I) &&
           "Unknown instruction and operand ID for getting MemoryLocation.");
    CallBase *CB = cast<CallBase>(I);
    return MemoryLocation::getForArgument(CB, OperandNum, TLI);
  }
}

Value *CilkSanitizerImpl::Instrumentor::getNoAliasSuppressionValue(
    Instruction *I, IRBuilder<> &IRB, unsigned OperandNum,
    MemoryLocation Loc, const RaceInfo::RaceData &RD, const Value *Obj,
    Value *ObjNoAliasFlag) {
  AliasAnalysis *AA = RI.getAA();
  for (const RaceInfo::RaceData &OtherRD : RI.getRaceData(I)) {
    // Skip checking other accesses that don't involve a pointer
    if (!OtherRD.Access.getPointer())
      continue;
    // Skip this operand when scanning for aliases
    if (OperandNum == OtherRD.OperandNum)
      continue;
    // If we can tell statically that these two memory locations don't alias,
    // move on.
    if (!AA->alias(Loc, getMemoryLocation(I, OtherRD.OperandNum,
                                          CilkSanImpl.TLI)))
      continue;

    // We trust that the suppression value in LocalSuppressions[] for this
    // object Obj, set by InsertArgSuppressionFlags, is correct.  We need to
    // check the underlying objects of the other arguments to see if they match
    // this object.

    // Otherwise we check the underlying objects.
    SmallPtrSet<const Value *, 1> OtherObjects;
    RI.getObjectsFor(OtherRD.Access, OtherObjects);
    for (const Value *OtherObj : OtherObjects) {
      // If we find another instance of this object in another argument,
      // then we don't have "no alias".
      if (Obj == OtherObj) {
        LLVM_DEBUG({
            dbgs() << "getNoAliasSuppressionValue: Matching objects found:\n";
            dbgs() << "  Obj: " << *Obj << "\n";
            dbgs() << "    I: " << *I << "\n";
            dbgs() << " Operands " << OperandNum << ", " << OtherRD.OperandNum
                   << "\n";
          });
        return getSuppressionIRValue(IRB, 0);
      }

      // We now know that Obj and OtherObj don't match.

      // If the other object is an argument, then we trust the noalias value in
      // the suppression for Obj.
      if (isa<Argument>(OtherObj))
        continue;

      // // If the other object is something we can't reason about locally, then we
      // // give up.
      // if (!isa<Instruction>(OtherObj))
      //   return getSuppressionIRValue(IRB, 0);

      // Otherwise, check if the other object might alias this one.
      if (AA->alias(Loc, MemoryLocation(OtherObj))) {
        LLVM_DEBUG({
            dbgs() << "getNoAliasSuppressionValue: Possible aliasing between:\n";
            dbgs() << "  Obj: " << *Obj << "\n";
            dbgs() << "  OtherObj: " << *OtherObj << "\n";
          });
        return getSuppressionIRValue(IRB, 0);
      }
    }
  }
  return ObjNoAliasFlag;
}

// TODO: Combine the logic of getSuppressionValue and getSuppressionCheck.
Value *CilkSanitizerImpl::Instrumentor::getSuppressionValue(
    Instruction *I, IRBuilder<> &IRB, unsigned OperandNum,
    SuppressionVal DefaultSV, bool CheckArgs) {
  Function *F = I->getFunction();
  AliasAnalysis *AA = RI.getAA();
  MemoryLocation Loc = getMemoryLocation(I, OperandNum, CilkSanImpl.TLI);
  Value *SV = getSuppressionIRValue(
      IRB, static_cast<unsigned>(SuppressionVal::NoAccess));
  Value *DefaultSuppression = getSuppressionIRValue(
      IRB, static_cast<unsigned>(DefaultSV));

  // // Check the other operands of this call to check for aliasing, e.g., because
  // // the same pointer is passed twice.
  // bool OperandMayAlias = false;
  // for (const RaceInfo::RaceData &OtherRD : RI.getRaceData(I)) {
  //   // Skip this operand when scanning for aliases
  //   if (OperandNum == OtherRD.OperandNum)
  //     continue;
  //   // Check for aliasing.
  //   if (OtherRD.OperandNum != static_cast<unsigned>(-1))
  //     if (AA->alias(Loc, getMemoryLocation(I, OtherRD.OperandNum,
  //                                          CilkSanImpl.TLI)))
  //       OperandMayAlias = true;
  // }

  // bool ObjectsDontAlias = true;
  Value *NoAliasFlag = getSuppressionIRValue(
      IRB, static_cast<unsigned>(SuppressionVal::NoAlias));
  // Check the recorded race data for I.
  for (const RaceInfo::RaceData &RD : RI.getRaceData(I)) {
    // Skip race data for different operands of the same instruction.
    if (OperandNum != RD.OperandNum)
      continue;

    SmallPtrSet<const Value *, 1> Objects;
    RI.getObjectsFor(RD.Access, Objects);
    // // Add objects to CilkSanImpl.ObjectMRForRace, to ensure ancillary
    // // instrumentation is added.
    // for (Value *Obj : Objects)
    //   if (!CilkSanImpl.ObjectMRForRace.count(Obj))
    //     CilkSanImpl.ObjectMRForRace[Obj] = ModRefInfo::ModRef;

    // Get suppressions from objects
    for (const Value *Obj : Objects) {
      // If we find an object with no suppression, give up.
      if (!LocalSuppressions.count(Obj)) {
        LLVM_DEBUG(dbgs() << "No local suppression found for obj " << *Obj
                   << "\n");
        return DefaultSuppression;
      }

      Value *FlagLoad = readSuppressionVal(LocalSuppressions[Obj], IRB);
      Value *FlagCheck = IRB.CreateAnd(
          FlagLoad, getSuppressionIRValue(IRB, RaceTypeToFlagVal(RD.Type)));
      SV = IRB.CreateOr(SV, FlagCheck);
      // SV = IRB.CreateOr(SV, FlagLoad);

      // Get the dynamic no-alias bit from the suppression value.
      Value *ObjNoAliasFlag = IRB.CreateAnd(
          FlagLoad, getSuppressionIRValue(
              IRB, static_cast<unsigned>(SuppressionVal::NoAlias)));
      Value *NoAliasCheck = IRB.CreateICmpNE(getSuppressionIRValue(IRB, 0),
                                             ObjNoAliasFlag);

      if (CheckArgs) {
        // Check the function arguments that might alias this object.
        for (Argument &Arg : F->args()) {
          // Ignore non-pointer arguments
          if (!Arg.getType()->isPtrOrPtrVectorTy())
            continue;
          // Ignore any arguments that match checked objects.
          if (&Arg == Obj)
            continue;
          // Check if Loc and Arg may alias.
          if (!AA->alias(Loc, MemoryLocation(&Arg)))
            continue;
          // If we have no local suppression information about the argument,
          // give up.
          if (!LocalSuppressions.count(&Arg)) {
            LLVM_DEBUG(dbgs() << "No local suppression found for arg " << Arg
                       << "\n");
            return DefaultSuppression;
          }

          Value *FlagLoad = readSuppressionVal(LocalSuppressions[&Arg], IRB);
          Value *FlagCheck = IRB.CreateSelect(
              NoAliasCheck, getSuppressionIRValue(IRB, 0),
              IRB.CreateAnd(
                  FlagLoad, getSuppressionIRValue(IRB,
                                                  RaceTypeToFlagVal(RD.Type))));
          SV = IRB.CreateOr(SV, FlagCheck);
          // Value *FlagCheck = IRB.CreateAnd(
          //     FlagLoad, getSuppressionIRValue(IRB, RaceTypeToFlagVal(RD.Type)));
          // SV = IRB.CreateOr(SV, FlagCheck);
        }
      }

      // // Check for noalias attributes to determine if we can set the noalias
      // // suppression bit in this value (at this call).
      // bool ObjNoAlias = false;
      // if (Argument *Arg = dyn_cast<Argument>(Obj))
      //   ObjNoAlias = Arg->hasNoAliasAttr();
      // else if (CallBase *CB = dyn_cast<CallBase>(Obj))
      //   ObjNoAlias = CB->hasRetAttr(Attribute::NoAlias);
      // ObjNoAliasFlag = IRB.CreateOr(
      //     ObjNoAliasFlag,
      //     getSuppressionIRValue(IRB, ObjNoAlias ? SuppressionVal::NoAlias : 0));

      // // Look for instances of the same object
      // //
      // // TODO: Possibly optimize this quadratic algorithm, if it proves to be a
      // // problem.
      // for (const RaceInfo::RaceData &OtherRD : RI.getRaceData(I)) {
      //   // Skip this operand when scanning for aliases
      //   if (OperandNum == OtherRD.OperandNum)
      //     continue;
      //   SmallPtrSet<Value *, 1> OtherObjects;
      //   RI.getObjectsFor(OtherRD.Access, OtherObjects);
      //   for (Value *OtherObj : OtherObjects) {
      //     // If we find another instance of this object in another argument,
      //     // then we don't have "no alias".
      //     if (Obj == OtherObj)
      //       ObjNoAliasFlag = getSuppressionIRValue(IRB, 0);
      //   }
      // }

      // Call getNoAliasSuppressionValue to evaluate the no-alias value in the
      // suppression for Obj, and intersect that result with the noalias
      // information for other objects.
      NoAliasFlag = IRB.CreateAnd(NoAliasFlag, getNoAliasSuppressionValue(
                                      I, IRB, OperandNum, Loc, RD, Obj,
                                      ObjNoAliasFlag));
    }
  }
  // Record the no-alias information.
  SV = IRB.CreateOr(SV, NoAliasFlag);
  return SV;
}

Value *CilkSanitizerImpl::Instrumentor::getSuppressionCheck(
    Instruction *I, IRBuilder<> &IRB, unsigned OperandNum) {
  Function *F = I->getFunction();
  AliasAnalysis *AA = RI.getAA();
  MemoryLocation Loc = getMemoryLocation(I, OperandNum, CilkSanImpl.TLI);
  Value *SuppressionChk = IRB.getTrue();
  // Check the recorded race data for I.
  for (const RaceInfo::RaceData &RD : RI.getRaceData(I)) {
    // Skip race data for different operands of the same instruction.
    if (OperandNum != RD.OperandNum)
      continue;

    SmallPtrSet<const Value *, 1> Objects;
    RI.getObjectsFor(RD.Access, Objects);
    for (const Value *Obj : Objects) {
      // Ignore objects that are not involved in races.
      if (!RI.ObjectInvolvedInRace(Obj))
        continue;

      // If we find an object with no suppression, give up.
      if (!LocalSuppressions.count(Obj)) {
        dbgs() << "No local suppression found for obj " << *Obj << "\n";
        dbgs() << "  I: " << *I << "\n";
        dbgs() << "  Ptr: " << *RD.Access.getPointer() << "\n";
        return IRB.getFalse();
      }

      Value *FlagLoad = readSuppressionVal(LocalSuppressions[Obj], IRB);
      Value *FlagCheck = IRB.CreateAnd(
          FlagLoad, getSuppressionIRValue(IRB, RaceTypeToFlagVal(RD.Type)));
      SuppressionChk = IRB.CreateAnd(
          SuppressionChk, IRB.CreateICmpEQ(getSuppressionIRValue(IRB, 0),
                                           FlagCheck));
      // Get the dynamic no-alias bit from the suppression value.
      Value *NoAliasCheck = IRB.CreateICmpNE(
          getSuppressionIRValue(IRB, 0), IRB.CreateAnd(
              FlagLoad, getSuppressionIRValue(
                  IRB, static_cast<unsigned>(SuppressionVal::NoAlias))));

      // Check the function arguments that might alias this object.
      for (Argument &Arg : F->args()) {
        // Ignore non-pointer arguments
        if (!Arg.getType()->isPtrOrPtrVectorTy())
          continue;
        // Ignore any arguments that match checked objects.
        if (&Arg == Obj)
          continue;
        // Check if Loc and Arg may alias.
        if (!AA->alias(Loc, MemoryLocation(&Arg)))
          continue;
        // If we have no local suppression information about the argument, give up.
        if (!LocalSuppressions.count(&Arg)) {
          dbgs() << "No local suppression found for arg " << Arg << "\n";
          return IRB.getFalse();
        }

        // Incorporate the suppression value for this argument if we don't have
        // a dynamic no-alias bit set.
        Value *FlagLoad = readSuppressionVal(LocalSuppressions[&Arg], IRB);
        Value *FlagCheck = IRB.CreateAnd(
            FlagLoad, getSuppressionIRValue(IRB, RaceTypeToFlagVal(RD.Type)));
        SuppressionChk = IRB.CreateAnd(
            SuppressionChk, IRB.CreateOr(
                NoAliasCheck, IRB.CreateICmpEQ(getSuppressionIRValue(IRB, 0),
                                               FlagCheck)));
      }
    }
  }
  return SuppressionChk;
}

bool CilkSanitizerImpl::Instrumentor::PerformDelayedInstrumentation() {
  bool Result = false;
  // Handle delayed simple instructions
  for (Instruction *I : DelayedSimpleInsts) {
    assert(RI.mightRaceViaAncestor(I) &&
           "Delayed instrumentation is not race via ancestor");
    IRBuilder<> IRB(I);

    Value *SupprChk = getSuppressionCheck(I, IRB);
    Instruction *CheckTerm = SplitBlockAndInsertIfThen(
        IRB.CreateICmpEQ(SupprChk, IRB.getFalse()), I, false, nullptr, DT,
        /*LI=*/nullptr);
    IRB.SetInsertPoint(CheckTerm);
    if (isa<LoadInst>(I) || isa<StoreInst>(I))
      Result |= CilkSanImpl.instrumentLoadOrStore(I, IRB);
    else if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I))
      Result |= CilkSanImpl.instrumentAtomic(I, IRB);
    else
      dbgs() << "[Cilksan] Unknown simple instruction: " << *I << "\n";
  }

  // Handle delayed memory intrinsics
  for (auto &MemIntrinOp : DelayedMemIntrinsics) {
    Instruction *I = MemIntrinOp.first;
    assert(RI.mightRaceViaAncestor(I) &&
           "Delayed instrumentation is not race via ancestor");
    unsigned OperandNum = MemIntrinOp.second;
    IRBuilder<> IRB(I);

    Value *SupprChk = getSuppressionCheck(I, IRB, OperandNum);
    Instruction *CheckTerm = SplitBlockAndInsertIfThen(
        IRB.CreateICmpEQ(SupprChk, IRB.getFalse()), I, false, nullptr, DT,
        /*LI=*/nullptr);
    IRB.SetInsertPoint(CheckTerm);
    Result |= CilkSanImpl.instrumentAnyMemIntrinAcc(I, OperandNum, IRB);
  }
  return Result;
}

bool CilkSanitizerImpl::Instrumentor::InstrumentAncillaryInstructions(
    SmallPtrSetImpl<Instruction *> &Allocas,
    SmallPtrSetImpl<Instruction *> &AllocationFnCalls,
    SmallPtrSetImpl<Instruction *> &FreeCalls,
    DenseMap<Value *, unsigned> &SyncRegNums,
    DenseMap<BasicBlock *, unsigned> &SRCounters, const DataLayout &DL,
    const TargetLibraryInfo *TLI) {
  bool Result = false;
  SmallPtrSet<SyncInst *, 4> Syncs;
  SmallPtrSet<Loop *, 4> Loops;

  // Instrument allocas and allocation-function calls that may be involved in a
  // race.
  for (Instruction *I : Allocas) {
    if (CilkSanImpl.ObjectMRForRace.count(I) ||
        PointerMayBeCaptured(I, true, false)) {
      CilkSanImpl.instrumentAlloca(I);
      getDetachesForInstruction(I);
      Result = true;
    }
  }
  for (Instruction *I : AllocationFnCalls) {
    if (CilkSanImpl.ObjectMRForRace.count(I) ||
        PointerMayBeCaptured(I, true, false)) {
      CilkSanImpl.instrumentAllocationFn(I, DT);
      getDetachesForInstruction(I);
      Result = true;
    }
  }
  for (Instruction *I : FreeCalls) {
    // The first argument of the free call is the pointer.
    Value *Ptr = I->getOperand(0);
    // If the pointer corresponds to an allocation function call in this
    // function, or if the pointer is involved in a race, then instrument it.
    if (Instruction *PtrI = dyn_cast<Instruction>(Ptr)) {
      if (AllocationFnCalls.count(PtrI)) {
        CilkSanImpl.instrumentFree(I);
        getDetachesForInstruction(I);
        Result = true;
        continue;
      }
    }
    if (RI.ObjectInvolvedInRace(Ptr) || unknownObjectUses(Ptr, DL, &LI, TLI)) {
      CilkSanImpl.instrumentFree(I);
      getDetachesForInstruction(I);
      Result = true;
    }
  }

  // Instrument detaches
  for (DetachInst *DI : Detaches) {
    CilkSanImpl.instrumentDetach(DI, SyncRegNums[DI->getSyncRegion()],
                                 SRCounters[DI->getDetached()], DT, TI, LI);
    Result = true;
    // Get syncs associated with this detach
    for (SyncInst *SI : CilkSanImpl.DetachToSync[DI])
      Syncs.insert(SI);

    if (CilkSanImpl.Options.InstrumentLoops) {
      // Get any loop associated with this detach.
      Loop *L = LI.getLoopFor(DI->getParent());
      if (spawnsTapirLoopBody(DI, LI, TI))
        Loops.insert(L);
    }
  }

  // Instrument associated syncs
  for (SyncInst *SI : Syncs)
    CilkSanImpl.instrumentSync(SI, SyncRegNums[SI->getSyncRegion()]);

  if (CilkSanImpl.Options.InstrumentLoops) {
    // Recursively instrument all loops
    for (Loop *L : Loops)
      CilkSanImpl.instrumentLoop(*L, TI, SyncRegNums);
  }

  return Result;
}

static bool CheckSanitizeCilkAttr(Function &F) {
  if (IgnoreSanitizeCilkAttr)
    return true;
  return F.hasFnAttribute(Attribute::SanitizeCilk);
}

bool CilkSanitizerImpl::instrumentFunctionUsingRI(Function &F) {
  if (F.empty() || shouldNotInstrumentFunction(F) ||
      !CheckSanitizeCilkAttr(F)) {
    LLVM_DEBUG({
        dbgs() << "Skipping " << F.getName() << "\n";
        if (F.empty())
          dbgs() << "  Empty function\n";
        else if (shouldNotInstrumentFunction(F))
          dbgs() << "  Function should not be instrumented\n";
        else if (!CheckSanitizeCilkAttr(F))
          dbgs() << "  Function lacks sanitize_cilk attribute\n";});
    return false;
  }

  LLVM_DEBUG(dbgs() << "Instrumenting " << F.getName() << "\n");

  if (Options.CallsMayThrow)
    setupCalls(F);
  setupBlocks(F);

  DominatorTree *DT = &GetDomTree(F);
  LoopInfo &LI = GetLoopInfo(F);
  if (Options.InstrumentLoops)
    for (Loop *L : LI)
      simplifyLoop(L, DT, &LI, nullptr, nullptr, nullptr,
                   /* PreserveLCSSA */false);

  SmallVector<Instruction *, 8> AllLoadsAndStores;
  SmallVector<Instruction *, 8> LocalLoadsAndStores;
  SmallVector<Instruction *, 8> AtomicAccesses;
  SmallVector<Instruction *, 8> MemIntrinCalls;
  SmallVector<Instruction *, 8> Callsites;
  // Ancillary instructions
  SmallPtrSet<Instruction *, 8> Allocas;
  SmallPtrSet<Instruction *, 8> AllocationFnCalls;
  SmallPtrSet<Instruction *, 8> FreeCalls;
  SmallVector<SyncInst *, 8> Syncs;
  DenseMap<BasicBlock *, unsigned> SRCounters;
  DenseMap<Value *, unsigned> SyncRegNums;

  TaskInfo &TI = GetTaskInfo(F);
  RaceInfo &RI = GetRaceInfo(F);
  // Evaluate the tasks that might be in parallel with each spindle.
  MaybeParallelTasks MPTasks;
  TI.evaluateParallelState<MaybeParallelTasks>(MPTasks);

  for (BasicBlock &BB : F) {
    // Record the Tapir sync instructions found
    if (SyncInst *SI = dyn_cast<SyncInst>(BB.getTerminator()))
      Syncs.push_back(SI);

    // Record the memory accesses in the basic block
    for (Instruction &Inst : BB) {
      // TODO: Handle VAArgInst
      if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst))
        LocalLoadsAndStores.push_back(&Inst);
      else if (isa<AtomicRMWInst>(Inst) || isa<AtomicCmpXchgInst>(Inst))
        AtomicAccesses.push_back(&Inst);
      else if (isa<CallInst>(Inst) || isa<InvokeInst>(Inst)) {
        // if (CallInst *CI = dyn_cast<CallInst>(&Inst))
        //   maybeMarkSanitizerLibraryCallNoBuiltin(CI, TLI);

        // If we find a sync region, record it.
        if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(&Inst))
          if (Intrinsic::syncregion_start == II->getIntrinsicID()) {
            // Identify this sync region with a counter value, where all sync
            // regions within a function or task are numbered from 0.
            BasicBlock *TEntry = TI.getTaskFor(&BB)->getEntry();
            // Create a new counter if need be.
            if (!SRCounters.count(TEntry))
              SRCounters[TEntry] = 0;
            SyncRegNums[&Inst] = SRCounters[TEntry]++;
          }

        // Record this function call as either an allocation function, a call to
        // free (or delete), a memory intrinsic, or an ordinary real function
        // call.
        if (isAllocationFn(&Inst, TLI, /*LookThroughBitCast=*/false,
                           /*IgnoreBuiltinAttr=*/true))
          AllocationFnCalls.insert(&Inst);
        else if (isFreeCall(&Inst, TLI))
          FreeCalls.insert(&Inst);
        else if (isa<AnyMemIntrinsic>(Inst))
          MemIntrinCalls.push_back(&Inst);
        else if (!simpleCallCannotRace(Inst))
          Callsites.push_back(&Inst);

        // Add the current set of local loads and stores to be considered for
        // instrumentation.
        if (!simpleCallCannotRace(Inst)) {
          chooseInstructionsToInstrument(LocalLoadsAndStores,
                                         AllLoadsAndStores, TI, LI);
        }
      } else if (isa<AllocaInst>(Inst)) {
        Allocas.insert(&Inst);
      }
    }
    chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores, TI,
                                     LI);
  }

  // Map each detach instruction with the sync instructions that could sync it.
  for (SyncInst *Sync : Syncs)
    for (const Task *MPT :
           MPTasks.TaskList[TI.getSpindleFor(Sync->getParent())])
      DetachToSync[MPT->getDetach()].push_back(Sync);

  // Record objects involved in races
  for (auto &ObjRD : RI.getObjectMRForRace())
    ObjectMRForRace[ObjRD.first] = ObjRD.second;

  uint64_t LocalId = getLocalFunctionID(F);
  IRBuilder<> IRB(&*F.getEntryBlock().getFirstInsertionPt());
  Value *FuncId = FunctionFED.localToGlobalId(LocalId, IRB);

  bool Result = false;
  if (!EnableStaticRaceDetection) {
    SimpleInstrumentor FuncI(*this, TI, LI, DT);
    Result |= FuncI.InstrumentSimpleInstructions(AllLoadsAndStores);
    Result |= FuncI.InstrumentSimpleInstructions(AtomicAccesses);
    Result |= FuncI.InstrumentAnyMemIntrinsics(MemIntrinCalls);
    Result |= FuncI.InstrumentCalls(Callsites);

    // Instrument ancillary instructions including allocas, allocation-function
    // calls, free calls, detaches, and syncs.
    Result |= FuncI.InstrumentAncillaryInstructions(Allocas, AllocationFnCalls,
                                                    FreeCalls, SyncRegNums,
                                                    SRCounters, DL, TLI);
  } else {
    Instrumentor FuncI(*this, RI, TI, LI, DT);
    // Insert suppression flags for each function argument.
    FuncI.InsertArgSuppressionFlags(F, FuncId);

    // TODO: Implement these instrumentation routines.
    //
    // -) Ancestor races: If pointer is a function argument, then make
    // instrumentation conditional on ModRef bit inserted for that argument.
    Result |= FuncI.InstrumentSimpleInstructions(AllLoadsAndStores);
    Result |= FuncI.InstrumentSimpleInstructions(AtomicAccesses);
    Result |= FuncI.InstrumentAnyMemIntrinsics(MemIntrinCalls);
    // -) Set the correct ModRef bit for each pointer argument for the call.
    Result |= FuncI.InstrumentCalls(Callsites);

    // Instrument ancillary instructions including allocas, allocation-function
    // calls, free calls, detaches, and syncs.
    Result |= FuncI.InstrumentAncillaryInstructions(Allocas, AllocationFnCalls,
                                                    FreeCalls, SyncRegNums,
                                                    SRCounters, DL, TLI);

    // Once we have handled ancillary instructions, we've done the necessary
    // analysis on this function.  We now perform delayed instrumentation, which
    // can involve changing the CFG and thereby violating some analyses.
    Result |= FuncI.PerformDelayedInstrumentation();
  }

  if (Result) {
    IRBuilder<> IRB(&*(++(cast<Instruction>(FuncId)->getIterator())));
    CsiFuncProperty FuncEntryProp;
    bool MaySpawn = !TI.isSerial();
    FuncEntryProp.setMaySpawn(MaySpawn);
    if (MaySpawn)
      FuncEntryProp.setNumSyncReg(SRCounters[TI.getRootTask()->getEntry()]);
    // TODO: Determine if we actually want the frame pointer, not the stack
    // pointer.
    Value *FrameAddr = IRB.CreateCall(
        Intrinsic::getDeclaration(&M, Intrinsic::frameaddress),
        {IRB.getInt32(0)});
    Value *StackSave = IRB.CreateCall(
        Intrinsic::getDeclaration(&M, Intrinsic::stacksave));
    IRB.CreateCall(CsanFuncEntry,
                   {FuncId, FrameAddr, StackSave, FuncEntryProp.getValue(IRB)});

    EscapeEnumerator EE(F, "csan_cleanup", false);
    while (IRBuilder<> *AtExit = EE.Next()) {
      uint64_t ExitLocalId = FunctionExitFED.add(*AtExit->GetInsertPoint());
      Value *ExitCsiId = FunctionExitFED.localToGlobalId(ExitLocalId, *AtExit);
      CsiFuncExitProperty FuncExitProp;
      FuncExitProp.setMaySpawn(MaySpawn);
      FuncExitProp.setEHReturn(isa<ResumeInst>(AtExit->GetInsertPoint()));
      AtExit->CreateCall(CsanFuncExit,
                         {ExitCsiId, FuncId, FuncExitProp.getValue(*AtExit)});
    }
  }

  // Record aggregate race information for the function and its arguments for
  // interprocedural analysis.
  //
  // TODO: Clean this up
  RaceInfo::RaceType FuncRT = RaceInfo::None;
  for (Instruction *I : AllLoadsAndStores)
    FuncRT = RaceInfo::unionRaceTypes(FuncRT, RI.getRaceType(I));
  for (Instruction *I : AtomicAccesses)
    FuncRT = RaceInfo::unionRaceTypes(FuncRT, RI.getRaceType(I));
  for (Instruction *I : MemIntrinCalls)
    FuncRT = RaceInfo::unionRaceTypes(FuncRT, RI.getRaceType(I));
  for (Instruction *I : Callsites) {
    if (const CallBase *CB = dyn_cast<CallBase>(I)) {
      // Use updated information about the race type of the call, if it's
      // available.
      const Function *CF = CB->getCalledFunction();
      if (FunctionRaceType.count(CF)) {
        FuncRT = RaceInfo::unionRaceTypes(FuncRT, FunctionRaceType[CF]);
        continue;
      }
    }
    FuncRT = RaceInfo::unionRaceTypes(FuncRT, RI.getRaceType(I));
  }
  FunctionRaceType[&F] = FuncRT;

  return Result;
}

bool CilkSanitizerImpl::instrumentLoadOrStore(Instruction *I,
                                              IRBuilder<> &IRB) {
  bool IsWrite = isa<StoreInst>(*I);
  Value *Addr = IsWrite
      ? cast<StoreInst>(I)->getPointerOperand()
      : cast<LoadInst>(I)->getPointerOperand();

  // swifterror memory addresses are mem2reg promoted by instruction selection.
  // As such they cannot have regular uses like an instrumentation function and
  // it makes no sense to track them as memory.
  if (Addr->isSwiftError())
    return false;

  int NumBytesAccessed = getNumBytesAccessed(Addr, DL);
  if (-1 == NumBytesAccessed) {
    // Ignore accesses with bad sizes.
    NumAccessesWithBadSize++;
    return false;
  }

  const unsigned Alignment = IsWrite
      ? cast<StoreInst>(I)->getAlignment()
      : cast<LoadInst>(I)->getAlignment();
  CsiLoadStoreProperty Prop;
  Prop.setAlignment(Alignment);
  if (IsWrite) {
    // Instrument store
    uint64_t LocalId = StoreFED.add(*I);
    uint64_t StoreObjId = StoreObj.add(*I, GetUnderlyingObject(Addr, DL));
    assert(LocalId == StoreObjId &&
           "Store received different ID's in FED and object tables.");
    Value *CsiId = StoreFED.localToGlobalId(LocalId, IRB);
    Value *Args[] = {CsiId,
                     IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                     IRB.getInt32(NumBytesAccessed),
                     Prop.getValue(IRB)};
    Instruction *Call = IRB.CreateCall(CsanWrite, Args);
    IRB.SetInstDebugLocation(Call);
    NumInstrumentedWrites++;
  } else {
    // Instrument load
    uint64_t LocalId = LoadFED.add(*I);
    uint64_t LoadObjId = LoadObj.add(*I, GetUnderlyingObject(Addr, DL));
    assert(LocalId == LoadObjId &&
           "Load received different ID's in FED and object tables.");
    Value *CsiId = LoadFED.localToGlobalId(LocalId, IRB);
    Value *Args[] = {CsiId,
                     IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                     IRB.getInt32(NumBytesAccessed),
                     Prop.getValue(IRB)};
    Instruction *Call = IRB.CreateCall(CsanRead, Args);
    IRB.SetInstDebugLocation(Call);
    NumInstrumentedReads++;
  }
  return true;
}

bool CilkSanitizerImpl::instrumentAtomic(Instruction *I, IRBuilder<> &IRB) {
  CsiLoadStoreProperty Prop;
  Value *Addr;
  if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I)) {
    Addr = RMWI->getPointerOperand();
  } else if (AtomicCmpXchgInst *CASI = dyn_cast<AtomicCmpXchgInst>(I)) {
    Addr = CASI->getPointerOperand();
  } else {
    return false;
  }

  int NumBytesAccessed = getNumBytesAccessed(Addr, DL);
  if (-1 == NumBytesAccessed) {
    // Ignore accesses with bad sizes.
    NumAccessesWithBadSize++;
    return false;
  }

  uint64_t LocalId = StoreFED.add(*I);
  uint64_t StoreObjId = StoreObj.add(*I, GetUnderlyingObject(Addr, DL));
  assert(LocalId == StoreObjId &&
         "Store received different ID's in FED and object tables.");
  Value *CsiId = StoreFED.localToGlobalId(LocalId, IRB);
  Value *Args[] = {CsiId,
                   IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                   IRB.getInt32(NumBytesAccessed),
                   Prop.getValue(IRB)};
  Instruction *Call = IRB.CreateCall(CsanWrite, Args);
  IRB.SetInstDebugLocation(Call);
  NumInstrumentedWrites++;
  return true;
}

bool CilkSanitizerImpl::instrumentCallsite(
    Instruction *I, SmallVectorImpl<Value *> *SupprVals) {
  if (callsPlaceholderFunction(*I))
    return false;

  bool IsInvoke = isa<InvokeInst>(I);
  CallBase *CB = dyn_cast<CallBase>(I);
  if (!CB)
    return false;
  Function *Called = CB->getCalledFunction();

  IRBuilder<> IRB(I);
  uint64_t LocalId = CallsiteFED.add(*I);
  Value *DefaultID = getDefaultID(IRB);
  Value *CallsiteId = CallsiteFED.localToGlobalId(LocalId, IRB);
  Value *FuncId = GetCalleeFuncID(Called, IRB);
  // GlobalVariable *FuncIdGV = NULL;
  // if (Called) {
  //   std::string GVName =
  //     CsiFuncIdVariablePrefix + Called->getName().str();
  //   FuncIdGV = dyn_cast<GlobalVariable>(M.getOrInsertGlobal(GVName,
  //                                                           IRB.getInt64Ty()));
  //   assert(FuncIdGV);
  //   FuncIdGV->setConstant(false);
  //   if (Options.jitMode && !Called->empty())
  //     FuncIdGV->setLinkage(Called->getLinkage());
  //   else
  //     FuncIdGV->setLinkage(GlobalValue::WeakAnyLinkage);
  //   FuncIdGV->setInitializer(IRB.getInt64(CsiCallsiteUnknownTargetId));
  //   FuncId = IRB.CreateLoad(FuncIdGV);
  // } else {
  //   // Unknown targets (i.e. indirect calls) are always unknown.
  //   FuncId = IRB.getInt64(CsiCallsiteUnknownTargetId);
  // }
  assert(FuncId != NULL);

  // Type *SVArrayElTy = IRB.getInt8Ty();
  // Value *SVArray = ConstantPointerNull::get(PointerType::get(SVArrayElTy, 0));
  // Value *StackAddr = nullptr;
  Value *NumSVVal = IRB.getInt8(0);
  // ConstantInt *SVArraySize = nullptr;
  if (SupprVals && !SupprVals->empty()) {
    unsigned NumSV = SupprVals->size();
    NumSVVal = IRB.getInt8(NumSV);
  //   // Save information about the stack before allocating the index array.
  //   StackAddr =
  //     IRB.CreateCall(Intrinsic::getDeclaration(&M, Intrinsic::stacksave));
  //   // Allocate the index array.
  //   SVArray = IRB.CreateAlloca(SVArrayElTy, NumSV);
  //   SVArraySize =
  //     IRB.getInt64(DL.getTypeAllocSize(
  //                      cast<AllocaInst>(SVArray)->getAllocatedType())
  //                  * NumSV);
  //   IRB.CreateLifetimeStart(SVArray, SVArraySize);
  //   // Populate the index array.
  //   unsigned SVIdx = 0;
  //   for (Value *SV : *SupprVals) {
  //     IRB.CreateStore(IRB.CreateZExt(SV, SVArrayElTy),
  //                     IRB.CreateInBoundsGEP(SVArray, { IRB.getInt32(SVIdx) }));
  //     SVIdx++;
  //   }
  }

  CsiCallProperty Prop;
  Value *DefaultPropVal = Prop.getValue(IRB);
  Prop.setIsIndirect(!Called);
  Value *PropVal = Prop.getValue(IRB);
  insertHookCall(I, CsanBeforeCallsite, {CallsiteId, FuncId,// SVArray, NumSVVal,
                                        NumSVVal, PropVal});
  // if (SupprVals && !SupprVals.empty()) {
  //   // Clean up the suppression-values array.
  //   IRB.CreateLifetimeEnd(SVArray, SVArraySize);
  //   IRB.CreateCall(Intrinsic::getDeclaration(&M, Intrinsic::stackrestore),
  //                  {StackAddr});
  // }

  // Don't bother adding after_call instrumentation for function calls that
  // don't return.
  if (CB->doesNotReturn())
    return true;

  BasicBlock::iterator Iter(I);
  if (IsInvoke) {
    // There are two "after" positions for invokes: the normal block and the
    // exception block.
    InvokeInst *II = cast<InvokeInst>(I);
    insertHookCallInSuccessorBB(
        II->getNormalDest(), II->getParent(), CsanAfterCallsite,
        {CallsiteId, FuncId, NumSVVal, PropVal},
        {DefaultID, DefaultID, IRB.getInt8(0), DefaultPropVal});
    insertHookCallInSuccessorBB(
        II->getUnwindDest(), II->getParent(), CsanAfterCallsite,
        {CallsiteId, FuncId, NumSVVal, PropVal},
        {DefaultID, DefaultID, IRB.getInt8(0), DefaultPropVal});
  } else {
    // Simple call instruction; there is only one "after" position.
    Iter++;
    IRB.SetInsertPoint(&*Iter);
    PropVal = Prop.getValue(IRB);
    insertHookCall(&*Iter, CsanAfterCallsite,
                   {CallsiteId, FuncId, NumSVVal, PropVal});
  }

  return true;
}

bool CilkSanitizerImpl::suppressCallsite(Instruction *I) {
  if (callsPlaceholderFunction(*I))
    return false;

  bool IsInvoke = isa<InvokeInst>(I);

  IRBuilder<> IRB(I);
  insertHookCall(I, CsanDisableChecking, {});

  BasicBlock::iterator Iter(I);
  if (IsInvoke) {
    // There are two "after" positions for invokes: the normal block and the
    // exception block.
    InvokeInst *II = cast<InvokeInst>(I);
    insertHookCallInSuccessorBB(
        II->getNormalDest(), II->getParent(), CsanEnableChecking, {}, {});
    insertHookCallInSuccessorBB(
        II->getUnwindDest(), II->getParent(), CsanEnableChecking, {}, {});
  } else {
    // Simple call instruction; there is only one "after" position.
    Iter++;
    IRB.SetInsertPoint(&*Iter);
    insertHookCall(&*Iter, CsanEnableChecking, {});
  }

  return true;
}

static bool IsMemTransferDstOperand(unsigned OperandNum) {
  // This check should be kept in sync with TapirRaceDetect::GetGeneralAccesses.
  return (OperandNum == 0);
}

static bool IsMemTransferSrcOperand(unsigned OperandNum) {
  // This check should be kept in sync with TapirRaceDetect::GetGeneralAccesses.
  return (OperandNum == 1);
}

bool CilkSanitizerImpl::instrumentAnyMemIntrinAcc(Instruction *I,
                                                  unsigned OperandNum,
                                                  IRBuilder<> &IRB) {
  CsiLoadStoreProperty Prop;
  if (AnyMemTransferInst *M = dyn_cast<AnyMemTransferInst>(I)) {
    // Only instrument the large load and the large store components as
    // necessary.
    bool Instrumented = false;

    if (IsMemTransferDstOperand(OperandNum)) {
      Value *Addr = M->getDest();
      Prop.setAlignment(M->getDestAlignment());
      // Instrument the store
      uint64_t StoreId = StoreFED.add(*I);

      // TODO: Don't recalculate underlying objects
      uint64_t StoreObjId = StoreObj.add(*I, GetUnderlyingObject(Addr, DL));
      assert(StoreId == StoreObjId &&
             "Store received different ID's in FED and object tables.");

      Value *CsiId = StoreFED.localToGlobalId(StoreId, IRB);
      Value *Args[] = {CsiId, IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                       IRB.CreateIntCast(M->getLength(), IntptrTy, false),
                       Prop.getValue(IRB)};
      Instruction *Call = IRB.CreateCall(CsanLargeWrite, Args);
      IRB.SetInstDebugLocation(Call);
      ++NumInstrumentedMemIntrinsicWrites;
      Instrumented = true;
    }

    if (IsMemTransferSrcOperand(OperandNum)) {
      Value *Addr = M->getSource();
      Prop.setAlignment(M->getSourceAlignment());
      // Instrument the load
      uint64_t LoadId = LoadFED.add(*I);

      // TODO: Don't recalculate underlying objects
      uint64_t LoadObjId = LoadObj.add(*I, GetUnderlyingObject(Addr, DL));
      assert(LoadId == LoadObjId &&
             "Load received different ID's in FED and object tables.");

      Value *CsiId = StoreFED.localToGlobalId(LoadId, IRB);
      Value *Args[] = {CsiId, IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                       IRB.CreateIntCast(M->getLength(), IntptrTy, false),
                       Prop.getValue(IRB)};
      Instruction *Call = IRB.CreateCall(CsanLargeRead, Args);
      IRB.SetInstDebugLocation(Call);
      ++NumInstrumentedMemIntrinsicReads;
      Instrumented = true;
    }
    return Instrumented;
  } else if (AnyMemIntrinsic *M = dyn_cast<AnyMemIntrinsic>(I)) {
    // assert(IsMemIntrinDstOperand(OperandNum) &&
    //        "Race on memset not on destination operand.");
    Value *Addr = M->getDest();
    Prop.setAlignment(M->getDestAlignment());
    uint64_t LocalId = StoreFED.add(*I);

    // TODO: Don't recalculate underlying objects
    uint64_t StoreObjId = StoreObj.add(*I, GetUnderlyingObject(Addr, DL));
    assert(LocalId == StoreObjId &&
           "Store received different ID's in FED and object tables.");

    Value *CsiId = StoreFED.localToGlobalId(LocalId, IRB);
    Value *Args[] = {CsiId, IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                     IRB.CreateIntCast(M->getLength(), IntptrTy, false),
                     Prop.getValue(IRB)};
    Instruction *Call = IRB.CreateCall(CsanLargeWrite, Args);
    IRB.SetInstDebugLocation(Call);
    ++NumInstrumentedMemIntrinsicWrites;
    return true;
  }
  return false;
}

static void getTaskExits(
    DetachInst *DI, SmallVectorImpl<BasicBlock *> &TaskReturns,
    SmallVectorImpl<BasicBlock *> &TaskResumes,
    SmallVectorImpl<Spindle *> &SharedEHExits,
    TaskInfo &TI) {
  BasicBlock *DetachedBlock = DI->getDetached();
  Task *T = TI.getTaskFor(DetachedBlock);
  BasicBlock *ContinueBlock = DI->getContinue();

  // Examine the predecessors of the continue block and save any predecessors in
  // the task as a task return.
  for (BasicBlock *Pred : predecessors(ContinueBlock)) {
    if (T->simplyEncloses(Pred)) {
      assert(isa<ReattachInst>(Pred->getTerminator()));
      TaskReturns.push_back(Pred);
    }
  }

  // If the detach cannot throw, we're done.
  if (!DI->hasUnwindDest())
    return;

  // Detached-rethrow exits can appear in strange places within a task-exiting
  // spindle.  Hence we loop over all blocks in the spindle to find
  // detached rethrows.
  for (Spindle *S : depth_first<InTask<Spindle *>>(T->getEntrySpindle())) {
    if (S->isSharedEH()) {
      if (llvm::any_of(predecessors(S),
                       [](const Spindle *Pred){ return !Pred->isSharedEH(); }))
        SharedEHExits.push_back(S);
      continue;
    }

    for (BasicBlock *B : S->blocks())
      if (isDetachedRethrow(B->getTerminator()))
        TaskResumes.push_back(B);
  }
}

bool CilkSanitizerImpl::instrumentDetach(DetachInst *DI, unsigned SyncRegNum,
                                         unsigned NumSyncRegs,
                                         DominatorTree *DT, TaskInfo &TI,
                                         LoopInfo &LI) {
  bool TapirLoopBody = spawnsTapirLoopBody(DI, LI, TI);
  // Instrument the detach instruction itself
  Value *DetachID;
  {
    IRBuilder<> IRB(DI);
    uint64_t LocalID = DetachFED.add(*DI);
    DetachID = DetachFED.localToGlobalId(LocalID, IRB);
    Instruction *Call = IRB.CreateCall(CsanDetach, {DetachID,
                                                    IRB.getInt8(SyncRegNum)});
    IRB.SetInstDebugLocation(Call);
  }
  NumInstrumentedDetaches++;

  // Find the detached block, continuation, and associated reattaches.
  BasicBlock *DetachedBlock = DI->getDetached();
  BasicBlock *ContinueBlock = DI->getContinue();
  SmallVector<BasicBlock *, 8> TaskExits, TaskResumes;
  SmallVector<Spindle *, 2> SharedEHExits;
  getTaskExits(DI, TaskExits, TaskResumes, SharedEHExits, TI);

  // Instrument the entry and exit points of the detached task.
  {
    // Instrument the entry point of the detached task.
    IRBuilder<> IRB(&*DetachedBlock->getFirstInsertionPt());
    uint64_t LocalID = TaskFED.add(*DetachedBlock);
    Value *TaskID = TaskFED.localToGlobalId(LocalID, IRB);
    CsiTaskProperty Prop;
    Prop.setIsTapirLoopBody(TapirLoopBody);
    Prop.setNumSyncReg(NumSyncRegs);
    // Get the frame and stack pointers.
    Value *FrameAddr = IRB.CreateCall(
        Intrinsic::getDeclaration(&M, Intrinsic::task_frameaddress),
        {IRB.getInt32(0)});
    Value *StackSave = IRB.CreateCall(
        Intrinsic::getDeclaration(&M, Intrinsic::stacksave));
    Instruction *Call = IRB.CreateCall(CsanTaskEntry,
                                       {TaskID, DetachID, FrameAddr,
                                        StackSave, Prop.getValue(IRB)});
    // Instruction *Call = IRB.CreateCall(CsanTaskEntry,
    //                                    {TaskID, DetachID, FrameAddr});
    IRB.SetInstDebugLocation(Call);

    // Instrument the exit points of the detached tasks.
    for (BasicBlock *TaskExit : TaskExits) {
      IRBuilder<> IRB(TaskExit->getTerminator());
      uint64_t LocalID = TaskExitFED.add(*TaskExit->getTerminator());
      Value *TaskExitID = TaskExitFED.localToGlobalId(LocalID, IRB);
      CsiTaskExitProperty ExitProp;
      ExitProp.setIsTapirLoopBody(TapirLoopBody);
      Instruction *Call = IRB.CreateCall(
          CsanTaskExit, {TaskExitID, TaskID, DetachID, IRB.getInt8(SyncRegNum),
                         ExitProp.getValue(IRB)});
      IRB.SetInstDebugLocation(Call);
      NumInstrumentedDetachExits++;
    }
    // Instrument the EH exits of the detached task.
    for (BasicBlock *TaskExit : TaskResumes) {
      IRBuilder<> IRB(TaskExit->getTerminator());
      uint64_t LocalID = TaskExitFED.add(*TaskExit->getTerminator());
      Value *TaskExitID = TaskExitFED.localToGlobalId(LocalID, IRB);
      CsiTaskExitProperty ExitProp;
      ExitProp.setIsTapirLoopBody(TapirLoopBody);
      Instruction *Call = IRB.CreateCall(
          CsanTaskExit, {TaskExitID, TaskID, DetachID, IRB.getInt8(SyncRegNum),
                         ExitProp.getValue(IRB)});
      IRB.SetInstDebugLocation(Call);
      NumInstrumentedDetachExits++;
    }

    Task *T = TI.getTaskFor(DetachedBlock);
    Value *DefaultID = getDefaultID(IRB);
    for (Spindle *SharedEH : SharedEHExits) {
      CsiTaskExitProperty ExitProp;
      ExitProp.setIsTapirLoopBody(TapirLoopBody);
      insertHookCallAtSharedEHSpindleExits(
          SharedEH, T, CsanTaskExit, TaskExitFED,
          {TaskID, DetachID, IRB.getInt8(SyncRegNum),
           ExitProp.getValueImpl(DI->getContext())},
          {DefaultID, DefaultID, IRB.getInt8(0),
           CsiTaskExitProperty::getDefaultValueImpl(DI->getContext())});
    }
  }

  // Instrument the continuation of the detach.
  {
    if (isCriticalContinueEdge(DI, 1))
      ContinueBlock = SplitCriticalEdge(
          DI, 1,
          CriticalEdgeSplittingOptions(DT, &LI).setSplitDetachContinue());

    IRBuilder<> IRB(&*ContinueBlock->getFirstInsertionPt());
    uint64_t LocalID = DetachContinueFED.add(*ContinueBlock);
    Value *ContinueID = DetachContinueFED.localToGlobalId(LocalID, IRB);
    Instruction *Call = IRB.CreateCall(CsanDetachContinue,
                                       {ContinueID, DetachID});
    IRB.SetInstDebugLocation(Call);
  }
  // Instrument the unwind of the detach, if it exists.
  if (DI->hasUnwindDest()) {
    BasicBlock *UnwindBlock = DI->getUnwindDest();
    IRBuilder<> IRB(DI);
    Value *DefaultID = getDefaultID(IRB);
    uint64_t LocalID = DetachContinueFED.add(*UnwindBlock);
    Value *ContinueID = DetachContinueFED.localToGlobalId(LocalID, IRB);
    insertHookCallInSuccessorBB(UnwindBlock, DI->getParent(),
                                CsanDetachContinue, {ContinueID, DetachID},
                                {DefaultID, DefaultID});
  }
  return true;
}

bool CilkSanitizerImpl::instrumentSync(SyncInst *SI, unsigned SyncRegNum) {
  IRBuilder<> IRB(SI);
  // Get the ID of this sync.
  uint64_t LocalID = SyncFED.add(*SI);
  Value *SyncID = SyncFED.localToGlobalId(LocalID, IRB);
  // Insert instrumentation before the sync.
  insertHookCall(SI, CsanSync, {SyncID, IRB.getInt8(SyncRegNum)});
  NumInstrumentedSyncs++;
  return true;
}

// Helper function to get a value for the runtime trip count of the given loop.
static const SCEV *getRuntimeTripCount(Loop &L, ScalarEvolution *SE) {
  BasicBlock *Latch = L.getLoopLatch();

  const SCEV *BECountSC = SE->getExitCount(&L, Latch);
  if (isa<SCEVCouldNotCompute>(BECountSC) ||
      !BECountSC->getType()->isIntegerTy()) {
    LLVM_DEBUG(dbgs() << "Could not compute exit block SCEV\n");
    return SE->getCouldNotCompute();
  }

  // Add 1 since the backedge count doesn't include the first loop iteration.
  const SCEV *TripCountSC =
      SE->getAddExpr(BECountSC, SE->getConstant(BECountSC->getType(), 1));
  if (isa<SCEVCouldNotCompute>(TripCountSC)) {
    LLVM_DEBUG(dbgs() << "Could not compute trip count SCEV.\n");
    return SE->getCouldNotCompute();
  }

  return TripCountSC;
}

void CilkSanitizerImpl::instrumentLoop(Loop &L,TaskInfo &TI,
                                       DenseMap<Value *, unsigned> &SyncRegNums,
                                       ScalarEvolution *SE) {
  assert(L.isLoopSimplifyForm() && "CSI assumes loops are in simplified form.");
  BasicBlock *Preheader = L.getLoopPreheader();
  Task *T = getTaskIfTapirLoop(&L, &TI);
  assert(T && "CilkSanitizer should only instrument Tapir loops.");
  unsigned SyncRegNum = SyncRegNums[T->getDetach()->getSyncRegion()];
  // We assign a local ID for this loop here, so that IDs for loops follow a
  // depth-first ordering.
  // csi_id_t LocalId = LoopFED.add(*Header);
  csi_id_t LocalId = LoopFED.add(*T->getDetach());

  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L.getExitingBlocks(ExitingBlocks);
  SmallVector<BasicBlock *, 4> ExitBlocks;
  L.getUniqueExitBlocks(ExitBlocks);

  // Record properties of this loop.
  CsiLoopProperty LoopProp;
  LoopProp.setIsTapirLoop(static_cast<bool>(getTaskIfTapirLoop(&L, &TI)));
  LoopProp.setHasUniqueExitingBlock((ExitingBlocks.size() == 1));

  IRBuilder<> IRB(Preheader->getTerminator());
  Value *LoopCsiId = LoopFED.localToGlobalId(LocalId, IRB);
  Value *LoopPropVal = LoopProp.getValue(IRB);

  // Try to evaluate the runtime trip count for this loop.  Default to a count
  // of -1 for unknown trip counts.
  Value *TripCount = IRB.getInt64(-1);
  if (SE) {
    const SCEV *TripCountSC = getRuntimeTripCount(L, SE);
    if (!isa<SCEVCouldNotCompute>(TripCountSC)) {
      // Extend the TripCount type if necessary.
      if (TripCountSC->getType() != IRB.getInt64Ty())
        TripCountSC = SE->getZeroExtendExpr(TripCountSC, IRB.getInt64Ty());
      // Compute the trip count to pass to the CSI hook.
      SCEVExpander Expander(*SE, DL, "csi");
      TripCount = Expander.expandCodeFor(TripCountSC, IRB.getInt64Ty(),
                                         &*IRB.GetInsertPoint());
    }
  }

  // Insert before-loop hook.
  insertHookCall(&*IRB.GetInsertPoint(), CsanBeforeLoop, {LoopCsiId, TripCount,
                                                          LoopPropVal});

  // // Insert loop-body-entry hook.
  // IRB.SetInsertPoint(&*Header->getFirstInsertionPt());
  // // TODO: Pass IVs to hook?
  // insertHookCall(&*IRB.GetInsertPoint(), CsiLoopBodyEntry, {LoopCsiId,
  //                                                           LoopPropVal});

  // // Insert hooks at the ends of the exiting blocks.
  // for (BasicBlock *BB : ExitingBlocks) {
  //   // Record properties of this loop exit
  //   CsiLoopExitProperty LoopExitProp;
  //   LoopExitProp.setIsLatch(L.isLoopLatch(BB));

  //   // Insert the loop-exit hook
  //   IRB.SetInsertPoint(BB->getTerminator());
  //   csi_id_t LocalExitId = LoopExitFED.add(*BB);
  //   Value *ExitCsiId = LoopFED.localToGlobalId(LocalExitId, IRB);
  //   Value *LoopExitPropVal = LoopExitProp.getValue(IRB);
  //   // TODO: For latches, record whether the loop will repeat.
  //   insertHookCall(&*IRB.GetInsertPoint(), CsiLoopBodyExit,
  //                  {ExitCsiId, LoopCsiId, LoopExitPropVal});
  // }
  // Insert after-loop hooks.
  for (BasicBlock *BB : ExitBlocks) {
    IRB.SetInsertPoint(&*BB->getFirstInsertionPt());
    insertHookCall(&*IRB.GetInsertPoint(), CsanAfterLoop,
                   {LoopCsiId, IRB.getInt8(SyncRegNum), LoopPropVal});
  }
}

bool CilkSanitizerImpl::instrumentAlloca(Instruction *I) {
  IRBuilder<> IRB(I);
  AllocaInst* AI = cast<AllocaInst>(I);

  uint64_t LocalId = AllocaFED.add(*I);
  Value *CsiId = AllocaFED.localToGlobalId(LocalId, IRB);
  uint64_t AllocaObjId = AllocaObj.add(*I, I);
  assert(LocalId == AllocaObjId &&
         "Alloca received different ID's in FED and object tables.");

  CsiAllocaProperty Prop;
  Prop.setIsStatic(AI->isStaticAlloca());
  Value *PropVal = Prop.getValue(IRB);

  // Get size of allocation.
  uint64_t Size = DL.getTypeAllocSize(AI->getAllocatedType());
  Value *SizeVal = IRB.getInt64(Size);
  if (AI->isArrayAllocation())
    SizeVal = IRB.CreateMul(SizeVal,
                            IRB.CreateZExtOrBitCast(AI->getArraySize(),
                                                    IRB.getInt64Ty()));

  BasicBlock::iterator Iter(I);
  Iter++;
  IRB.SetInsertPoint(&*Iter);

  Type *AddrType = IRB.getInt8PtrTy();
  Value *Addr = IRB.CreatePointerCast(I, AddrType);
  insertHookCall(&*Iter, CsiAfterAlloca, {CsiId, Addr, SizeVal, PropVal});

  NumInstrumentedAllocas++;
  return true;
}

static Value *getHeapObject(Instruction *I) {
  Value *Object = nullptr;
  unsigned NumOfBitCastUses = 0;

  // Determine if CallInst has a bitcast use.
  for (Value::user_iterator UI = I->user_begin(), E = I->user_end();
       UI != E;)
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(*UI++)) {
      // Look for a dbg.value intrinsic for this bitcast.
      SmallVector<DbgValueInst *, 1> DbgValues;
      findDbgValues(DbgValues, BCI);
      if (!DbgValues.empty()) {
        Object = BCI;
        NumOfBitCastUses++;
      }
    }

  // Heap-allocation call has 1 debug-bitcast use, so use that bitcast as the
  // object.
  if (NumOfBitCastUses == 1)
    return Object;

  // Otherwise just use the heap-allocation call directly.
  return I;
}

void CilkSanitizerImpl::getAllocFnArgs(
    const Instruction *I, SmallVectorImpl<Value*> &AllocFnArgs,
    Type *SizeTy, Type *AddrTy, const TargetLibraryInfo &TLI) {
  const Function *Called = nullptr;
  if (const CallInst *CI = dyn_cast<CallInst>(I))
    Called = CI->getCalledFunction();
  else if (const InvokeInst *II = dyn_cast<InvokeInst>(I))
    Called = II->getCalledFunction();

  LibFunc F;
  bool FoundLibFunc = TLI.getLibFunc(*Called, F);
  if (!FoundLibFunc)
    return;

  switch(F) {
  default: return;
    // TODO: Add aligned new's to this list after they're added to TLI.
  case LibFunc_malloc:
  case LibFunc_valloc:
  case LibFunc_Znwj:
  case LibFunc_ZnwjRKSt9nothrow_t:
  case LibFunc_Znwm:
  case LibFunc_ZnwmRKSt9nothrow_t:
  case LibFunc_Znaj:
  case LibFunc_ZnajRKSt9nothrow_t:
  case LibFunc_Znam:
  case LibFunc_ZnamRKSt9nothrow_t:
  case LibFunc_msvc_new_int:
  case LibFunc_msvc_new_int_nothrow:
  case LibFunc_msvc_new_longlong:
  case LibFunc_msvc_new_longlong_nothrow:
  case LibFunc_msvc_new_array_int:
  case LibFunc_msvc_new_array_int_nothrow:
  case LibFunc_msvc_new_array_longlong:
  case LibFunc_msvc_new_array_longlong_nothrow:
    {
      // Allocated size
      if (isa<CallInst>(I))
        AllocFnArgs.push_back(cast<CallInst>(I)->getArgOperand(0));
      else
        AllocFnArgs.push_back(cast<InvokeInst>(I)->getArgOperand(0));
      // Number of elements = 1
      AllocFnArgs.push_back(ConstantInt::get(SizeTy, 1));
      // Alignment = 0
      // TODO: Fix this for aligned new's, once they're added to TLI.
      AllocFnArgs.push_back(ConstantInt::get(SizeTy, 0));
      // Old pointer = NULL
      AllocFnArgs.push_back(Constant::getNullValue(AddrTy));
      return;
    }
  case LibFunc_calloc:
    {
      const CallInst *CI = cast<CallInst>(I);
      // Allocated size
      AllocFnArgs.push_back(CI->getArgOperand(1));
      // Number of elements
      AllocFnArgs.push_back(CI->getArgOperand(0));
      // Alignment = 0
      AllocFnArgs.push_back(ConstantInt::get(SizeTy, 0));
      // Old pointer = NULL
      AllocFnArgs.push_back(Constant::getNullValue(AddrTy));
      return;
    }
  case LibFunc_realloc:
  case LibFunc_reallocf:
    {
      const CallInst *CI = cast<CallInst>(I);
      // Allocated size
      AllocFnArgs.push_back(CI->getArgOperand(1));
      // Number of elements = 1
      AllocFnArgs.push_back(ConstantInt::get(SizeTy, 1));
      // Alignment = 0
      AllocFnArgs.push_back(ConstantInt::get(SizeTy, 0));
      // Old pointer
      AllocFnArgs.push_back(CI->getArgOperand(0));
      return;
    }
  case LibFunc_aligned_alloc:
    {
      const CallInst *CI = cast<CallInst>(I);
      // Allocated size
      AllocFnArgs.push_back(CI->getArgOperand(1));
      // Number of elements = 1
      AllocFnArgs.push_back(ConstantInt::get(SizeTy, 1));
      // Alignment
      AllocFnArgs.push_back(CI->getArgOperand(0));
      // Old pointer = NULL
      AllocFnArgs.push_back(Constant::getNullValue(AddrTy));
      return;
    }
  }
}

bool CilkSanitizerImpl::instrumentAllocationFn(Instruction *I,
                                               DominatorTree *DT) {
  bool IsInvoke = isa<InvokeInst>(I);
  Function *Called = nullptr;
  if (CallInst *CI = dyn_cast<CallInst>(I))
    Called = CI->getCalledFunction();
  else if (InvokeInst *II = dyn_cast<InvokeInst>(I))
    Called = II->getCalledFunction();

  assert(Called && "Could not get called function for allocation fn.");

  IRBuilder<> IRB(I);
  Value *DefaultID = getDefaultID(IRB);
  uint64_t LocalId = AllocFnFED.add(*I);
  Value *AllocFnId = AllocFnFED.localToGlobalId(LocalId, IRB);
  uint64_t AllocFnObjId = AllocFnObj.add(*I, getHeapObject(I));
  assert(LocalId == AllocFnObjId &&
         "Allocation fn received different ID's in FED and object tables.");

  SmallVector<Value *, 4> AllocFnArgs;
  getAllocFnArgs(I, AllocFnArgs, IntptrTy, IRB.getInt8PtrTy(), *TLI);
  SmallVector<Value *, 4> DefaultAllocFnArgs(
      {/* Allocated size */ Constant::getNullValue(IntptrTy),
       /* Number of elements */ Constant::getNullValue(IntptrTy),
       /* Alignment */ Constant::getNullValue(IntptrTy),
       /* Old pointer */ Constant::getNullValue(IRB.getInt8PtrTy()),});

  CsiAllocFnProperty Prop;
  Value *DefaultPropVal = Prop.getValue(IRB);
  LibFunc AllocLibF;
  TLI->getLibFunc(*Called, AllocLibF);
  Prop.setAllocFnTy(static_cast<unsigned>(getAllocFnTy(AllocLibF)));
  AllocFnArgs.push_back(Prop.getValue(IRB));
  DefaultAllocFnArgs.push_back(DefaultPropVal);

  BasicBlock::iterator Iter(I);
  if (IsInvoke) {
    // There are two "after" positions for invokes: the normal block and the
    // exception block.
    InvokeInst *II = cast<InvokeInst>(I);

    BasicBlock *NormalBB = II->getNormalDest();
    unsigned SuccNum = GetSuccessorNumber(II->getParent(), NormalBB);
    if (isCriticalEdge(II, SuccNum))
      NormalBB = SplitCriticalEdge(II, SuccNum,
                                   CriticalEdgeSplittingOptions(DT));
    // Insert hook into normal destination.
    {
      IRB.SetInsertPoint(&*NormalBB->getFirstInsertionPt());
      SmallVector<Value *, 4> AfterAllocFnArgs;
      AfterAllocFnArgs.push_back(AllocFnId);
      AfterAllocFnArgs.push_back(IRB.CreatePointerCast(I, IRB.getInt8PtrTy()));
      AfterAllocFnArgs.append(AllocFnArgs.begin(), AllocFnArgs.end());
      insertHookCall(&*IRB.GetInsertPoint(), CsanAfterAllocFn,
                     AfterAllocFnArgs);
    }
    // Insert hook into unwind destination.
    {
      // The return value of the allocation function is not valid in the unwind
      // destination.
      SmallVector<Value *, 4> AfterAllocFnArgs, DefaultAfterAllocFnArgs;
      AfterAllocFnArgs.push_back(AllocFnId);
      AfterAllocFnArgs.push_back(Constant::getNullValue(IRB.getInt8PtrTy()));
      AfterAllocFnArgs.append(AllocFnArgs.begin(), AllocFnArgs.end());
      DefaultAfterAllocFnArgs.push_back(DefaultID);
      DefaultAfterAllocFnArgs.push_back(
          Constant::getNullValue(IRB.getInt8PtrTy()));
      DefaultAfterAllocFnArgs.append(DefaultAllocFnArgs.begin(),
                                     DefaultAllocFnArgs.end());
      insertHookCallInSuccessorBB(
          II->getUnwindDest(), II->getParent(), CsanAfterAllocFn,
          AfterAllocFnArgs, DefaultAfterAllocFnArgs);
    }
  } else {
    // Simple call instruction; there is only one "after" position.
    Iter++;
    IRB.SetInsertPoint(&*Iter);
    SmallVector<Value *, 4> AfterAllocFnArgs;
    AfterAllocFnArgs.push_back(AllocFnId);
    AfterAllocFnArgs.push_back(IRB.CreatePointerCast(I, IRB.getInt8PtrTy()));
    AfterAllocFnArgs.append(AllocFnArgs.begin(), AllocFnArgs.end());
    insertHookCall(&*Iter, CsanAfterAllocFn, AfterAllocFnArgs);
  }

  NumInstrumentedAllocFns++;
  return true;
}

bool CilkSanitizerImpl::instrumentFree(Instruction *I) {
  // It appears that frees (and deletes) never throw.
  assert(isa<CallInst>(I) && "Free call is not a call instruction");

  CallInst *FC = cast<CallInst>(I);
  Function *Called = FC->getCalledFunction();
  assert(Called && "Could not get called function for free.");

  IRBuilder<> IRB(I);
  uint64_t LocalId = FreeFED.add(*I);
  Value *FreeId = FreeFED.localToGlobalId(LocalId, IRB);
  // uint64_t FreeObjId = FreeObj.add(*I, getHeapObject(I));
  // assert(LocalId == FreeObjId &&
  //        "Allocation fn received different ID's in FED and object tables.");

  Value *Addr = FC->getArgOperand(0);
  CsiFreeProperty Prop;
  LibFunc FreeLibF;
  TLI->getLibFunc(*Called, FreeLibF);
  Prop.setFreeTy(static_cast<unsigned>(getFreeTy(FreeLibF)));

  BasicBlock::iterator Iter(I);
  Iter++;
  IRB.SetInsertPoint(&*Iter);
  insertHookCall(&*Iter, CsanAfterFree, {FreeId, Addr, Prop.getValue(IRB)});

  NumInstrumentedFrees++;
  return true;
}

bool CilkSanitizerLegacyPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  CallGraph *CG = &getAnalysis<CallGraphWrapperPass>().getCallGraph();
  const TargetLibraryInfo *TLI =
      &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  auto GetDomTree = [this](Function &F) -> DominatorTree & {
    return this->getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
  };
  auto GetTaskInfo = [this](Function &F) -> TaskInfo & {
    return this->getAnalysis<TaskInfoWrapperPass>(F).getTaskInfo();
  };
  auto GetLoopInfo = [this](Function &F) -> LoopInfo & {
    return this->getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
  };
  auto GetDepInfo = [this](Function &F) -> DependenceInfo & {
    return this->getAnalysis<DependenceAnalysisWrapperPass>(F).getDI();
  };
  auto GetRaceInfo = [this](Function &F) -> RaceInfo & {
    return this->getAnalysis<TapirRaceDetectWrapperPass>(F).getRaceInfo();
  };

  return CilkSanitizerImpl(M, CG, GetDomTree, GetTaskInfo, GetLoopInfo,
                           GetDepInfo, GetRaceInfo, TLI, JitMode,
                           CallsMayThrow).run();
}

PreservedAnalyses CilkSanitizerPass::run(Module &M, ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto &CG = AM.getResult<CallGraphAnalysis>(M);
  auto GetDT =
    [&FAM](Function &F) -> DominatorTree & {
      return FAM.getResult<DominatorTreeAnalysis>(F);
    };
  auto GetTI =
    [&FAM](Function &F) -> TaskInfo & {
      return FAM.getResult<TaskAnalysis>(F);
    };
  auto GetLI =
    [&FAM](Function &F) -> LoopInfo & {
      return FAM.getResult<LoopAnalysis>(F);
    };
  auto GetDI =
    [&FAM](Function &F) -> DependenceInfo & {
      return FAM.getResult<DependenceAnalysis>(F);
    };
  auto GetRI =
    [&FAM](Function &F) -> RaceInfo & {
      return FAM.getResult<TapirRaceDetect>(F);
    };
  auto *TLI = &AM.getResult<TargetLibraryAnalysis>(M);

  if (!CilkSanitizerImpl(M, &CG, GetDT, GetTI, GetLI, GetDI, GetRI, TLI)
      .run())
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
