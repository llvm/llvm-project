//===- CilkSanitizer.cpp - Nondeterminism detector for Cilk/Tapir ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
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
// Simple enum to track different types of races.  Each race type is a 2-bit
// value, where a 1 bit indicates a write access.
enum RaceType {
  None = 0,
  RW = 1,
  WR = 2,
  WW = 3,
};

static RaceType getRaceType(bool Acc1IsWrite, bool Acc2IsWrite) {
  return RaceType((static_cast<unsigned>(Acc1IsWrite) << 1) |
                  static_cast<unsigned>(Acc2IsWrite));
}

// Simple enum to track the kinds of accesses an operation might perform that
// can result in a determinacy race.
enum AccessType {
  Read  = 1,
  Write = 2,
  ReadWrite = Read | Write,
};

static AccessType getAccessType(const RaceType RT, unsigned Arg) {
  assert(Arg < 2 && "Invalid arg for RaceType");
  return static_cast<AccessType>(
      static_cast<bool>(static_cast<int>(RT) & ((!Arg) << 1)) + 1);
}

static bool isReadSet(const AccessType AT) {
  return static_cast<int>(AT) & static_cast<int>(AccessType::Read);
}

static bool isWriteSet(const AccessType AT) {
  return static_cast<int>(AT) & static_cast<int>(AccessType::Write);
}

static AccessType setRead(const AccessType AT) {
  return AccessType(static_cast<int>(AT) | static_cast<int>(AccessType::Read));
}

static AccessType setWrite(const AccessType AT) {
  return AccessType(static_cast<int>(AT) | static_cast<int>(AccessType::Write));
}

static AccessType unionAccessType(const AccessType AT1, const AccessType AT2) {
  return AccessType(static_cast<int>(AT1) | static_cast<int>(AT2));
}

struct CilkSanitizerImpl : public CSIImpl {
  class SimpleInstrumentor {
  public:
    SimpleInstrumentor(CilkSanitizerImpl &CilkSanImpl, TaskInfo &TI,
                       DominatorTree *DT)
        : CilkSanImpl(CilkSanImpl), TI(TI), DT(DT) {}

    bool InstrumentSimpleInstructions(
        SmallVectorImpl<Instruction *> &Instructions);
    bool InstrumentAnyMemIntrinsics(
        SmallVectorImpl<Instruction *> &MemIntrinsics);
    bool InstrumentCalls(SmallVectorImpl<Instruction *> &Calls);
    bool InstrumentAncillaryInstructions(
        SmallPtrSetImpl<Instruction *> &Allocas,
        SmallPtrSetImpl<Instruction *> &AllocationFnCalls,
        SmallPtrSetImpl<Instruction *> &FreeCalls);

  private:
    void getDetachesForInstruction(Instruction *I);

    CilkSanitizerImpl &CilkSanImpl;
    TaskInfo &TI;
    DominatorTree *DT;

    SmallPtrSet<DetachInst *, 8> Detaches;

    SmallVector<Instruction *, 8> DelayedSimpleInsts;
    SmallVector<std::pair<Instruction *, unsigned>, 8> DelayedMemIntrinsics;
    SmallVector<Instruction *, 8> DelayedCalls;
  };

  class Instrumentor {
  public:
    Instrumentor(CilkSanitizerImpl &CilkSanImpl, RaceInfo &RI, TaskInfo &TI,
                 DominatorTree *DT)
        : CilkSanImpl(CilkSanImpl), RI(RI), TI(TI), DT(DT) {}

    void InsertArgSuppressionFlags(Function &F, Value *FuncId);
    bool InstrumentSimpleInstructions(
        SmallVectorImpl<Instruction *> &Instructions);
    bool InstrumentAnyMemIntrinsics(
        SmallVectorImpl<Instruction *> &MemIntrinsics);
    bool InstrumentCalls(SmallVectorImpl<Instruction *> &Calls);
    bool InstrumentAncillaryInstructions(
        SmallPtrSetImpl<Instruction *> &Allocas,
        SmallPtrSetImpl<Instruction *> &AllocationFnCalls,
        SmallPtrSetImpl<Instruction *> &FreeCalls);
    bool PerformDelayedInstrumentation();

  private:
    void getDetachesForInstruction(Instruction *I);
    enum SuppressionVal : uint8_t
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
                                      const RaceInfo::RaceData &RD, Value *Obj,
                                      Value *SupprVal);
    Value *getSuppressionCheck(Instruction *I, IRBuilder<> &IRB,
                               unsigned OperandNum = static_cast<unsigned>(-1));
    Value *readSuppressionVal(Value *V, IRBuilder<> &IRB);

    CilkSanitizerImpl &CilkSanImpl;
    RaceInfo &RI;
    TaskInfo &TI;
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
    Options.InstrumentLoops = false;
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

  Value *GetCalleeFuncID(Function *Callee, IRBuilder<> &IRB);

  // Analyze the instructions in F to determine which instructions need to be
  // isntrumented for race detection.
  bool prepareToInstrumentFunction(Function &F);
  // Helper function for prepareToInstrumentFunction that chooses loads and
  // stores in a basic block to instrument.
  void chooseInstructionsToInstrument(
      SmallVectorImpl<Instruction *> &Local,
      SmallVectorImpl<Instruction *> &All,
      const TaskInfo &TI, LoopInfo &LI);
  // Determine which instructions among the set of loads, stores, atomics,
  // memory intrinsics, and callsites could race with each other.
  bool GetMaybeRacingAccesses(
      SmallPtrSetImpl<Instruction *> &ToInstrument,
      DenseMap<MemTransferInst *, AccessType> &MemTransferAccTypes,
      SmallVectorImpl<Instruction *> &NoRaceCallsites,
      SmallVectorImpl<Instruction *> &LoadsAndStores,
      SmallVectorImpl<Instruction *> &Atomics,
      SmallVectorImpl<Instruction *> &MemIntrinCalls,
      SmallVectorImpl<Instruction *> &Callsites,
      DenseMap<const Task *, SmallVector<Instruction *, 8>> &TaskToMemAcc,
      MPTaskListTy &MPTasks, const DominatorTree &DT, const TaskInfo &TI,
      LoopInfo &LI, DependenceInfo &DI);

  // Determine which call sites need instrumentation, based on what's known
  // about which functions can access memory in a racing fashion.
  void determineCallSitesToInstrument();

  // Insert hooks at relevant program points
  bool instrumentFunction(Function &F,
                          SmallPtrSetImpl<Instruction *> &ToInstrument,
                          SmallPtrSetImpl<Instruction *> &NoRaceCallsites);
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
  bool instrumentMemIntrinsic(
      Instruction *I,
      DenseMap<MemTransferInst *, AccessType> &MemTransferAccTypes);
  bool instrumentCallsite(Instruction *I,
                          SmallVectorImpl<Value *> *SupprVals = nullptr);
  bool suppressCallsite(Instruction *I);
  bool instrumentAllocationFn(Instruction *I, DominatorTree *DT);
  bool instrumentFree(Instruction *I);
  bool instrumentDetach(DetachInst *DI, DominatorTree *DT, TaskInfo &TI);
  bool instrumentSync(SyncInst *SI);
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
  Function *CsanFuncEntry = nullptr;
  Function *CsanFuncExit = nullptr;
  Function *CsanRead = nullptr;
  Function *CsanWrite = nullptr;
  Function *CsanLargeRead = nullptr;
  Function *CsanLargeWrite = nullptr;
  Function *CsanBeforeCallsite = nullptr;
  Function *CsanAfterCallsite = nullptr;
  Function *CsanDetach = nullptr;
  Function *CsanDetachContinue = nullptr;
  Function *CsanTaskEntry = nullptr;
  Function *CsanTaskExit = nullptr;
  Function *CsanSync = nullptr;
  Function *CsanAfterAllocFn = nullptr;
  Function *CsanAfterFree = nullptr;

  // Hooks for suppressing instrumentation, e.g., around callsites that cannot
  // expose a race.
  Function *CsanDisableChecking = nullptr;
  Function *CsanEnableChecking = nullptr;

  Function *GetSuppressionFlag = nullptr;
  Function *SetSuppressionFlag = nullptr;

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

  SmallPtrSet<Function *, 16> NoRecurseFunctions;
  bool FnDoesNotRecur(Function &F) {
    if (F.hasFnAttribute(Attribute::NoRecurse))
      return true;
    // dbgs() << "Checking if function " << F.getName()
    //        << " is norecurse by custom analysis: "
    //        << NoRecurseFunctions.count(&F) << "\n";
    return NoRecurseFunctions.count(&F);
  }
  DenseMap<MemTransferInst *, AccessType> MemTransferAccTypes;
  SmallVector<Instruction *, 8> NoRaceCallsites;
  SmallPtrSet<Function *, 8> MayRaceFunctions;
  DenseMap<DetachInst *, SmallVector<SyncInst *, 2>> DetachToSync;
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
} // namespace

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

//------------------------------------------------------------------------------
// Code copied from lib/Transforms/IPO/FunctionAttrs.cpp and adapted to perform
// custom norecurse computation.

static bool setDoesNotRecurse(Function &F,
                              SmallPtrSetImpl<Function *> &NoRecurseFunctions) {
  if (NoRecurseFunctions.count(&F))
    return false;
  NoRecurseFunctions.insert(&F);
  // ++NumNoRecurse;
  return true;
}

static bool checkNoRecurse(Function *F,
                           SmallPtrSetImpl<Function *> &NoRecurseFunctions,
                           const TargetLibraryInfo *TLI) {
  if (F->doesNotRecurse() || NoRecurseFunctions.count(F))
    return true;
  if (AssumeLibCallsDontRecur) {
    LibFunc LibF;
    if (TLI->getLibFunc(*F, LibF))
      // TODO: Pick the library functions we care about designating as
      // norecurse.
      return true;
  }
  return false;
}

static bool considerForRecurse(const Instruction *I) {
  if (isDetachedRethrow(I))
    return false;

  if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I))
    switch (II->getIntrinsicID()) {
    default: return true;
    case Intrinsic::annotation:
    case Intrinsic::assume:
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
    case Intrinsic::syncregion_start:
    case Intrinsic::vastart:
    case Intrinsic::vacopy:
    case Intrinsic::vaend:
      return false;
    }
  return true;
}

static bool addNoRecurse(const SmallSetVector<Function *, 8> &SCCNodes,
                         SmallPtrSetImpl<Function *> &NoRecurseFunctions,
                         SmallPtrSetImpl<Function *> &CallsUnknown,
                         const TargetLibraryInfo *TLI) {
  // Try and identify functions that do not recurse.

  // If the SCC contains multiple nodes we know for sure there is recursion.
  if (SCCNodes.size() != 1)
    return false;

  Function *F = *SCCNodes.begin();
  if (!F || F->isDeclaration() ||
      F->doesNotRecurse() || NoRecurseFunctions.count(F))
    return false;

  // If all of the calls in F are identifiable and are to norecurse functions, F
  // is norecurse. This check also detects self-recursion as F is not currently
  // marked norecurse, so any called from F to F will not be marked norecurse.
  bool MightRecurse = false;
  for (auto &BB : *F)
    for (auto &I : BB.instructionsWithoutDebug()) {
      if (!considerForRecurse(&I))
        continue;
      if (auto CS = CallSite(&I)) {
        Function *Callee = CS.getCalledFunction();
        LibFunc LibF;
        if (!Callee ||
            (Callee->isDeclaration() &&
             !(AssumeLibCallsDontRecur && TLI->getLibFunc(*Callee, LibF)))) {
          // dbgs() << "Function " << F->getName()
          //        << " calls an unknown function "
          //        << Callee->getName() << "\n";
          CallsUnknown.insert(F);
        }
        if (!Callee || Callee == F ||
            (!checkNoRecurse(Callee, NoRecurseFunctions, TLI) &&
             CallsUnknown.count(Callee))) {
          // dbgs() << "Function " << F->getName()
          //        << " potentially recurs by calling function "
          //        << Callee->getName() << "\n";
          // Function calls a potentially recursive function.
          MightRecurse = true;
        }
      }
    }

  if (MightRecurse)
    return false;

  // Every call was to a non-recursive function other than this function, and
  // we have no indirect recursion as the SCC size is one. This function cannot
  // recurse.
  return setDoesNotRecurse(*F, NoRecurseFunctions);
}

static bool addNoRecurseAttrsTopDown(
    Function &F, SmallPtrSetImpl<Function *> &NoRecurseFunctions,
    const TargetLibraryInfo *TLI) {
  // We check the preconditions for the function prior to calling this to avoid
  // the cost of building up a reversible post-order list. We assert them here
  // to make sure none of the invariants this relies on were violated.
  assert(!F.isDeclaration() && "Cannot deduce norecurse without a definition!");
  assert(!F.doesNotRecurse() &&
         "This function has already been deduced as norecurs!");
  assert(F.hasInternalLinkage() &&
         "Can only do top-down deduction for internal linkage functions!");

  // If F is internal and all of its uses are calls from a non-recursive
  // functions, then none of its calls could in fact recurse without going
  // through a function marked norecurse, and so we can mark this function too
  // as norecurse. Note that the uses must actually be calls -- otherwise
  // a pointer to this function could be returned from a norecurse function but
  // this function could be recursively (indirectly) called. Note that this
  // also detects if F is directly recursive as F is not yet marked as
  // a norecurse function.
  for (auto *U : F.users()) {
    auto *I = dyn_cast<Instruction>(U);
    if (!I)
      return false;
    CallSite CS(I);
    if (!CS)
      return false;

    Function *Caller = CS.getParent()->getParent();
    if (!checkNoRecurse(Caller, NoRecurseFunctions, TLI))
      return false;
  }
  return setDoesNotRecurse(F, NoRecurseFunctions);
}

static bool deduceFunctionAttributeInRPO(
    Module &M, CallGraph &CG, SmallPtrSetImpl<Function *> &NoRecurseFunctions,
    const TargetLibraryInfo *TLI) {
  // We only have a post-order SCC traversal (because SCCs are inherently
  // discovered in post-order), so we accumulate them in a vector and then walk
  // it in reverse. This is simpler than using the RPO iterator infrastructure
  // because we need to combine SCC detection and the PO walk of the call
  // graph. We can also cheat egregiously because we're primarily interested in
  // synthesizing norecurse and so we can only save the singular SCCs as SCCs
  // with multiple functions in them will clearly be recursive.
  SmallVector<Function *, 16> Worklist;
  for (scc_iterator<CallGraph *> I = scc_begin(&CG); !I.isAtEnd(); ++I) {
    if (I->size() != 1)
      continue;

    Function *F = I->front()->getFunction();
    if (F && !F->isDeclaration() && !F->doesNotRecurse() &&
        F->hasInternalLinkage())
      Worklist.push_back(F);
  }

  bool Changed = false;
  for (auto *F : llvm::reverse(Worklist))
    Changed |= addNoRecurseAttrsTopDown(*F, NoRecurseFunctions, TLI);

  return Changed;
}

bool CilkSanitizerImpl::run() {
  // Initialize components of the CSI and Cilksan system.
  initializeCsi();
  initializeFEDTables();
  initializeCsanObjectTables();
  initializeCsanHooks();

  // // Traverse the callgraph SCC's in post order and deduce no-recurse
  // // information.
  // SmallPtrSet<Function *, 8> CallsUnknown;
  // for (scc_iterator<CallGraph *> I = scc_begin(CG); !I.isAtEnd(); ++I) {
  //   const std::vector<CallGraphNode *> &SCC = *I;
  //   assert(!SCC.empty() && "SCC with no functions!");
  //   SmallSetVector<Function *, 8> SCCNodes;
  //   for (auto *CGN : SCC)
  //     if (Function *F = CGN->getFunction())
  //       SCCNodes.insert(F);
  //   addNoRecurse(SCCNodes, NoRecurseFunctions, CallsUnknown, TLI);
  // }
  // // Further refine the no-recurse information top-down
  // deduceFunctionAttributeInRPO(M, *CG, NoRecurseFunctions, TLI);

  // Evaluate the SCC's in the callgraph in post order to support
  // interprocedural analysis of potential races in the module.
  SmallVector<Function *, 16> InstrumentedFunctions;
  for (scc_iterator<CallGraph *> I = scc_begin(CG); !I.isAtEnd(); ++I) {
    const std::vector<CallGraphNode *> &SCC = *I;
    assert(!SCC.empty() && "SCC with no functions!");
    for (auto *CGN : SCC) {
      if (Function *F = CGN->getFunction()) {
        if (instrumentFunctionUsingRI(*F))
          InstrumentedFunctions.push_back(F);
      }
    }
  }
  // After all functions have been analyzed and instrumented, update their
  // attributes.
  for (Function *F : InstrumentedFunctions) {
    updateInstrumentedFnAttrs(*F);
    F->removeFnAttr(Attribute::SanitizeCilk);
  }

  // // Examine all functions in the module to find IR objects within each function
  // // that require instrumentation.
  // for (Function &F : M) {
  //   LLVM_DEBUG(dbgs() << "Preparing to instrument " << F.getName() << "\n");
  //   prepareToInstrumentFunction(F);
  // }

  // // Based on information of which functions might expose a race, determine
  // // which callsites in the function need to be instrumented.
  // determineCallSitesToInstrument();

  // // Determine which callsites can have instrumentation suppressed.
  // DenseMap<Function *, SmallPtrSet<Instruction *, 32>> ToInstrumentByFunction;
  // DenseMap<Function *, SmallPtrSet<Instruction *, 8>> NoRaceCallsitesByFunction;
  // for (Instruction *I : ToInstrument)
  //   ToInstrumentByFunction[I->getParent()->getParent()].insert(I);
  // for (Instruction *I : NoRaceCallsites)
  //   if (!ToInstrument.count(I))
  //     NoRaceCallsitesByFunction[I->getParent()->getParent()].insert(I);

  // // Insert the necessary instrumentation throughout each function in the
  // // module.
  // for (Function &F : M) {
  //   LLVM_DEBUG(dbgs() << "Instrumenting " << F.getName() << "\n");
  //   instrumentFunction(F, ToInstrumentByFunction[&F],
  //                      NoRaceCallsitesByFunction[&F]);
  // }

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
  Type *LoadPropertyTy = CsiLoadStoreProperty::getType(C);
  Type *StorePropertyTy = CsiLoadStoreProperty::getType(C);
  Type *CallPropertyTy = CsiCallProperty::getType(C);
  Type *AllocFnPropertyTy = CsiAllocFnProperty::getType(C);
  Type *FreePropertyTy = CsiFreeProperty::getType(C);
  Type *RetType = IRB.getVoidTy();
  Type *AddrType = IRB.getInt8PtrTy();
  Type *NumBytesType = IRB.getInt32Ty();
  Type *LargeNumBytesType = IntptrTy;
  Type *IDType = IRB.getInt64Ty();

  CsanFuncEntry = M.getOrInsertFunction("__csan_func_entry", RetType,
                                        /* func_id */ IDType,
                                        /* frame_ptr */ AddrType,
                                        /* stack_ptr */ AddrType,
                                        FuncPropertyTy);
  CsanFuncEntry->addParamAttr(1, Attribute::NoCapture);
  CsanFuncEntry->addParamAttr(1, Attribute::ReadNone);
  CsanFuncEntry->addParamAttr(2, Attribute::NoCapture);
  CsanFuncEntry->addParamAttr(2, Attribute::ReadNone);
  CsanFuncEntry->addFnAttr(Attribute::InaccessibleMemOnly);
  CsanFuncExit = M.getOrInsertFunction("__csan_func_exit", RetType,
                                       /* func_exit_id */ IDType,
                                       /* func_id */ IDType,
                                       FuncExitPropertyTy);
  CsanFuncExit->addFnAttr(Attribute::InaccessibleMemOnly);

  CsanRead = M.getOrInsertFunction("__csan_load", RetType, IDType,
                                   AddrType, NumBytesType, LoadPropertyTy);
  CsanRead->addParamAttr(1, Attribute::NoCapture);
  CsanRead->addParamAttr(1, Attribute::ReadNone);
  CsanRead->addFnAttr(Attribute::InaccessibleMemOnly);
  CsanWrite = M.getOrInsertFunction("__csan_store", RetType, IDType,
                                    AddrType, NumBytesType, StorePropertyTy);
  CsanWrite->addParamAttr(1, Attribute::NoCapture);
  CsanWrite->addParamAttr(1, Attribute::ReadNone);
  CsanWrite->addFnAttr(Attribute::InaccessibleMemOnly);
  CsanLargeRead = M.getOrInsertFunction("__csan_large_load", RetType, IDType,
                                        AddrType, LargeNumBytesType,
                                        LoadPropertyTy);
  CsanLargeRead->addParamAttr(1, Attribute::NoCapture);
  CsanLargeRead->addParamAttr(1, Attribute::ReadNone);
  CsanLargeRead->addFnAttr(Attribute::InaccessibleMemOnly);
  CsanLargeWrite = M.getOrInsertFunction("__csan_large_store", RetType, IDType,
                                         AddrType, LargeNumBytesType,
                                         StorePropertyTy);
  CsanLargeWrite->addParamAttr(1, Attribute::NoCapture);
  CsanLargeWrite->addParamAttr(1, Attribute::ReadNone);
  CsanLargeWrite->addFnAttr(Attribute::InaccessibleMemOnly);
  // CsanWrite = M.getOrInsertFunction("__csan_atomic_exchange", RetType,
  //                                   IDType, AddrType, NumBytesType,
  //                                   StorePropertyTy);

  CsanBeforeCallsite = M.getOrInsertFunction("__csan_before_call",
                                             IRB.getVoidTy(), IDType,
                                             /*callee func_id*/ IDType,
                                             IRB.getInt8Ty(), CallPropertyTy);
  CsanBeforeCallsite->addFnAttr(Attribute::InaccessibleMemOnly);
  CsanAfterCallsite = M.getOrInsertFunction("__csan_after_call",
                                            IRB.getVoidTy(), IDType, IDType,
                                            IRB.getInt8Ty(), CallPropertyTy);
  CsanAfterCallsite->addFnAttr(Attribute::InaccessibleMemOnly);

  CsanDetach = M.getOrInsertFunction("__csan_detach", RetType,
                                     /* detach_id */ IDType);
  CsanDetach->addFnAttr(Attribute::InaccessibleMemOnly);
  CsanTaskEntry = M.getOrInsertFunction("__csan_task", RetType,
                                        /* task_id */ IDType,
                                        /* detach_id */ IDType,
                                        /* frame_ptr */ AddrType,
                                        /* stack_ptr */ AddrType);
  CsanTaskEntry->addParamAttr(2, Attribute::NoCapture);
  CsanTaskEntry->addParamAttr(2, Attribute::ReadNone);
  CsanTaskEntry->addParamAttr(3, Attribute::NoCapture);
  CsanTaskEntry->addParamAttr(3, Attribute::ReadNone);
  CsanTaskEntry->addFnAttr(Attribute::InaccessibleMemOnly);

  CsanTaskExit = M.getOrInsertFunction("__csan_task_exit", RetType,
                                       /* task_exit_id */ IDType,
                                       /* task_id */ IDType,
                                       /* detach_id */ IDType);
  CsanTaskExit->addFnAttr(Attribute::InaccessibleMemOnly);
  CsanDetachContinue = M.getOrInsertFunction("__csan_detach_continue", RetType,
                                             /* detach_continue_id */ IDType,
                                             /* detach_id */ IDType);
  CsanDetachContinue->addFnAttr(Attribute::InaccessibleMemOnly);
  CsanSync = M.getOrInsertFunction("__csan_sync", RetType, IDType);
  CsanSync->addFnAttr(Attribute::InaccessibleMemOnly);

  // CsanBeforeAllocFn = M.getOrInsertFunction("__csan_before_allocfn", RetType,
  //                                           IDType, LargeNumBytesType,
  //                                           LargeNumBytesType,
  //                                           LargeNumBytesType, AddrType);
  CsanAfterAllocFn = M.getOrInsertFunction("__csan_after_allocfn", RetType,
                                           IDType,
                                           /* new ptr */ AddrType,
                                           /* size */ LargeNumBytesType,
                                           /* num elements */ LargeNumBytesType,
                                           /* alignment */ LargeNumBytesType,
                                           /* old ptr */ AddrType,
                                           /* property */ AllocFnPropertyTy);
  CsanAfterAllocFn->addParamAttr(1, Attribute::NoCapture);
  CsanAfterAllocFn->addParamAttr(1, Attribute::ReadNone);
  CsanAfterAllocFn->addParamAttr(5, Attribute::NoCapture);
  CsanAfterAllocFn->addParamAttr(5, Attribute::ReadNone);
  CsanAfterAllocFn->addFnAttr(Attribute::InaccessibleMemOnly);
  // CsanBeforeFree = M.getOrInsertFunction("__csan_before_free", RetType, IDType,
  //                                        AddrType);
  CsanAfterFree = M.getOrInsertFunction("__csan_after_free", RetType, IDType,
                                        AddrType,
                                        /* property */ FreePropertyTy);
  CsanAfterFree->addParamAttr(1, Attribute::NoCapture);
  CsanAfterFree->addParamAttr(1, Attribute::ReadNone);
  CsanAfterFree->addFnAttr(Attribute::InaccessibleMemOnly);

  CsanDisableChecking = M.getOrInsertFunction("__cilksan_disable_checking",
                                              RetType);
  CsanDisableChecking->addFnAttr(Attribute::InaccessibleMemOnly);
  CsanEnableChecking = M.getOrInsertFunction("__cilksan_enable_checking",
                                             RetType);
  CsanEnableChecking->addFnAttr(Attribute::InaccessibleMemOnly);

  Type *SuppressionFlagTy = IRB.getInt64Ty();
  GetSuppressionFlag = M.getOrInsertFunction(
      "__csan_get_suppression_flag", RetType,
      PointerType::get(SuppressionFlagTy, 0), IDType, IRB.getInt8Ty());
  GetSuppressionFlag->addParamAttr(0, Attribute::NoCapture);
  GetSuppressionFlag->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);

  SetSuppressionFlag = M.getOrInsertFunction("__csan_set_suppression_flag",
                                             RetType, SuppressionFlagTy,
                                             IDType);
  SetSuppressionFlag->addFnAttr(Attribute::InaccessibleMemOnly);
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

// Get the general memory accesses for the instruction \p I, and stores those
// accesses into \p AccI.  Returns true if general memory accesses could be
// derived for I, false otherwise.
static bool GetGeneralAccesses(
    Instruction *I, SmallVectorImpl<GeneralAccess> &AccI, AliasAnalysis *AA,
    const TargetLibraryInfo *TLI) {
  GeneralAccess GA;
  GA.I = I;

  // Handle memory intrinsics.
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(I)) {
    GA.ModRef = setMod(GA.ModRef);
    GA.Loc = MemoryLocation::getForDest(MI);
    AccI.push_back(GA);
    if (AnyMemTransferInst *MTI = dyn_cast<AnyMemTransferInst>(I)) {
      GA.ModRef = setRef(GA.ModRef);
      GA.Loc = MemoryLocation::getForSource(MTI);
      AccI.push_back(GA);
    }
    return true;
  }

  // Handle more standard memory operations.
  if (Optional<MemoryLocation> MLoc = MemoryLocation::getOrNone(I)) {
    GA.Loc = *MLoc;
    GA.ModRef = isa<LoadInst>(I) ? setRef(GA.ModRef) : setMod(GA.ModRef);
    AccI.push_back(GA);
    return true;
  }

  // Handle arbitrary call sites by examining pointee arguments.
  if (const CallBase *CS = dyn_cast<CallBase>(I)) {
    for (auto AI = CS->arg_begin(), AE = CS->arg_end(); AI != AE; ++AI) {
      const Value *Arg = *AI;
      if (!Arg->getType()->isPtrOrPtrVectorTy())
        continue;

      unsigned ArgIdx = std::distance(CS->arg_begin(), AI);
      MemoryLocation Loc = MemoryLocation::getForArgument(CS, ArgIdx, *TLI);
      if (AA->pointsToConstantMemory(Loc))
        continue;

      GA.Loc = Loc;
      auto MRI = AA->getArgModRefInfo(CS, ArgIdx);
      if (isModSet(MRI))
        GA.ModRef = setMod(GA.ModRef);
      else if (isRefSet(MRI))
        GA.ModRef = setRef(GA.ModRef);

      AccI.push_back(GA);
    }
    return true;
  }
  return false;
}

static Spindle *GetRepSpindleInTask(Spindle *S, const Task *T,
                                    const TaskInfo &TI) {
  Spindle *CurrS = S;
  Task *CurrT = S->getParentTask();
  while (T != CurrT) {
    DetachInst *DI = CurrT->getDetach();
    if (!DI)
      return nullptr;
    CurrS = TI.getSpindleFor(DI->getContinue());
    CurrT = CurrS->getParentTask();
  }
  return CurrS;
}

// Structure to record the set of child tasks that might be in parallel with
// this spindle, ignoring back edges of loops.
struct MaybeParallelTasksInLoopBody : public MaybeParallelTasks {
  MPTaskListTy TaskList;
  LoopInfo &LI;

  MaybeParallelTasksInLoopBody(LoopInfo &LI) : LI(LI) {}

  // This method is called once per unevaluated spindle in an inverse-post-order
  // walk of the spindle graph.
  bool evaluate(const Spindle *S, unsigned EvalNum) {
    LLVM_DEBUG(dbgs() << "evaluate @ " << *S << "\n");
    if (!TaskList.count(S))
      TaskList.try_emplace(S);

    bool Complete = true;
    for (const Spindle::SpindleEdge &PredEdge : S->in_edges()) {
      const Spindle *Pred = PredEdge.first;
      const BasicBlock *Inc = PredEdge.second;

      // If the incoming edge is a sync edge, get the associated sync region.
      const Value *SyncRegSynced = nullptr;
      if (const SyncInst *SI = dyn_cast<SyncInst>(Inc->getTerminator()))
        SyncRegSynced = SI->getSyncRegion();

      // Skip back edges for this task list.
      if (Loop *L = LI.getLoopFor(S->getEntry()))
        if ((L->getHeader() == S->getEntry()) && L->contains(Inc))
          continue;

      // Iterate through the tasks in the task list for Pred.
      for (const Task *MP : TaskList[Pred]) {
        // Filter out any tasks that are synced by the sync region.
        if (const DetachInst *DI = MP->getDetach())
          if (SyncRegSynced == DI->getSyncRegion())
            continue;
        // Insert the task into this spindle's task list.  If this task is a new
        // addition, then we haven't yet reached the fixed point of this
        // analysis.
        if (TaskList[S].insert(MP).second)
          Complete = false;
      }
    }
    LLVM_DEBUG({
        dbgs() << "New MPT list for " << *S
               << (Complete ? "(complete)\n" : "(not complete)\n");
        for (const Task *MP : TaskList[S])
          dbgs() << "\t" << MP->getEntry()->getName() << "\n";
      });
    return Complete;
  }
};

static bool DependenceMightRace(
    std::unique_ptr<Dependence> D, Instruction *Src, Instruction *Dst,
    Value *SrcAddr, Value *DstAddr,
    MPTaskListTy &MPTasks, MPTaskListTy &MPTasksInLoop, const DominatorTree &DT,
    const TaskInfo &TI, LoopInfo &LI, const DataLayout &DL) {
  if (!D)
    // No dependence means no race.
    return false;

  LLVM_DEBUG({
      D->dump(dbgs());
      StringRef DepType =
        D->isFlow() ? "flow" : D->isAnti() ? "anti" : "output";
      dbgs() << "Found " << DepType
             << " dependency between Src and Dst\n";
      unsigned Levels = D->getLevels();
      for (unsigned II = 1; II <= Levels; ++II) {
        const SCEV *Distance = D->getDistance(II);
        if (Distance)
          dbgs() << "Level " << II << " distance " << *Distance << "\n";
      }
    });

  // Only dependencies that cross tasks can produce determinacy races.
  // Dependencies that cross loop iterations within the same task don't matter.
  // To search the relevant loops, start at the spindle entry that most closely
  // dominates both instructions, and check outwards, up to the topmost root for
  // which Dst is in a maybe-parallel task.  Dominating blocks within the
  // spindle are all guaranteed to execute in series with each other, so
  // dependencies between those instructions matter.

  // Use the base objects for the addresses to try to further refine the checks.
  SmallVector<Value *, 1> BaseObjs;
  GetUnderlyingObjects(SrcAddr, BaseObjs, DL, &LI, 0);
  GetUnderlyingObjects(DstAddr, BaseObjs, DL, &LI, 0);
  unsigned MinObjDepth = unsigned(-1);
  for (Value *Obj : BaseObjs) {
    if (Constant *C = dyn_cast<Constant>(Obj)) {
      if (C->isNullValue())
        continue;
    }
    if (!isa<Instruction>(Obj)) {
      MinObjDepth = 0;
      break;
    }
    unsigned ObjDepth = LI.getLoopDepth(cast<Instruction>(Obj)->getParent());
    if (ObjDepth < MinObjDepth)
      MinObjDepth = ObjDepth;
  }
  LLVM_DEBUG(dbgs() << "Min loop depth " << MinObjDepth <<
             " for underlying object.\n");

  // Find the spindle that dominates both instructions.
  BasicBlock *Dom =
    DT.findNearestCommonDominator(Src->getParent(), Dst->getParent());
  // Find the deepest loop that contains both Src and Dst.
  Loop *CommonLoop = LI.getLoopFor(Dom);
  unsigned MaxLoopDepthToCheck = CommonLoop ? CommonLoop->getLoopDepth() : 0;
  while (MaxLoopDepthToCheck &&
         (!CommonLoop->contains(Src->getParent()) ||
          !CommonLoop->contains(Dst->getParent()))) {
    CommonLoop = CommonLoop->getParentLoop();
    MaxLoopDepthToCheck--;
  }

  // Check if dependence does not depend on looping.
  if (0 == MaxLoopDepthToCheck)
    // If there's no loop to worry about, then the existence of the dependence
    // implies the potential for a race.
    return true;

  LLVM_DEBUG(
      if (MinObjDepth > MaxLoopDepthToCheck) {
        dbgs() << "\tSrc " << *Src << "\n\tDst " << *Dst;
        dbgs() << "\n\tMaxLoopDepthToCheck " << MaxLoopDepthToCheck;
        dbgs() << "\n\tMinObjDepthToCheck " << MinObjDepth << "\n";
        dbgs() << *Src->getFunction();
      });
  assert(MinObjDepth <= MaxLoopDepthToCheck &&
         "Minimum loop depth of underlying object cannot be greater "
         "than maximum loop depth of dependence.");

  Spindle *DomSpindle = TI.getSpindleFor(Dom);
  if (MaxLoopDepthToCheck == MinObjDepth) {
    if (TI.getTaskFor(Src->getParent()) == TI.getTaskFor(Dst->getParent()))
      return false;

    if (!(D->getDirection(MaxLoopDepthToCheck) & Dependence::DVEntry::EQ))
      // Apparent dependence does not occur within the same iteration.
      return false;

    Spindle *SrcSpindle =
      GetRepSpindleInTask(TI.getSpindleFor(Src->getParent()),
                          TI.getTaskFor(DomSpindle), TI);
    Spindle *DstSpindle =
      GetRepSpindleInTask(TI.getSpindleFor(Dst->getParent()),
                          TI.getTaskFor(DomSpindle), TI);
    for (const Task *MPT : MPTasksInLoop[SrcSpindle])
      if (TI.encloses(MPT, Dst))
        return true;
    for (const Task *MPT : MPTasksInLoop[DstSpindle])
      if (TI.encloses(MPT, Src))
        return true;

    return false;
  }

  // Get the loop stack up from the loop containing DomSpindle.
  SmallVector<Loop *, 4> LoopsToCheck;
  Loop *CurrLoop = LI.getLoopFor(DomSpindle->getEntry());
  while (CurrLoop) {
    LoopsToCheck.push_back(CurrLoop);
    CurrLoop = CurrLoop->getParentLoop();
  }

  // Check the loop stack from the top down until a loop is found
  unsigned MinLoopDepthToCheck = 1;
  while (!LoopsToCheck.empty()) {
    Loop *CurrLoop = LoopsToCheck.pop_back_val();
    // Check the maybe-parallel tasks for the spindle containing the loop
    // header.
    Spindle *CurrSpindle = TI.getSpindleFor(CurrLoop->getHeader());
    bool MPTEnclosesDst = false;
    for (const Task *MPT : MPTasks[CurrSpindle]) {
      if (TI.encloses(MPT, Dst->getParent())) {
        MPTEnclosesDst = true;
        break;
      }
    }
    // If Dst is found in a maybe-parallel task, then the minimum loop depth has
    // been found.
    if (MPTEnclosesDst)
      break;
    // Otherwise go deeper.
    MinLoopDepthToCheck++;
  }

  // Scan the loop nests in common from inside out.
  for (unsigned II = MaxLoopDepthToCheck; II >= MinLoopDepthToCheck; --II) {
    LLVM_DEBUG(dbgs() << "Checking loop level " << II << "\n");
    if (D->isScalar(II))
      return true;
    if (D->getDirection(II) & unsigned(~Dependence::DVEntry::EQ))
      return true;
  }

  LLVM_DEBUG(dbgs() << "Dependence does not cross parallel tasks.\n");
  return false;
}

static RaceType InstrsMightRace(Instruction *I, Instruction *MPI,
                                MPTaskListTy &MPTasks,
                                MPTaskListTy &MPTasksInLoop, DependenceInfo &DI,
                                const DominatorTree &DT, const TaskInfo &TI,
                                LoopInfo &LI, const TargetLibraryInfo *TLI,
                                const DataLayout &DL,
                                bool SkipRead = false, bool SkipWrite = false) {
  // Handle the simple case of a pair of load/store instructions.
  if ((isa<LoadInst>(I) || isa<StoreInst>(I)) &&
      (isa<LoadInst>(MPI) || isa<StoreInst>(MPI))) {
    // Two parallel loads cannot race.
    if (isa<LoadInst>(I) && isa<LoadInst>(MPI))
      return RaceType::None;
    if (DependenceMightRace(
            DI.depends(I, MPI, true), I, MPI,
            const_cast<Value *>(MemoryLocation::get(I).Ptr),
            const_cast<Value *>(MemoryLocation::get(MPI).Ptr),
            MPTasks, MPTasksInLoop, DT, TI, LI, DL))
      return getRaceType(!isa<LoadInst>(I), !isa<LoadInst>(MPI));
    return RaceType::None;
  }

  // Handle more general memory accesses.
  SmallVector<GeneralAccess, 2> AccI, AccMPI;
  bool Recognized = true;
  Recognized &= GetGeneralAccesses(I, AccI, DI.getAA(), TLI);
  Recognized &= GetGeneralAccesses(MPI, AccMPI, DI.getAA(), TLI);
  if (!Recognized)
    // If we couldn't figure out the generalized memory accesses for I and MPI,
    // assume a write-write race.
    return RaceType::WW;

  for (GeneralAccess GA1 : AccI) {
    // If processing a memory transfer intrinsic, check if we should skip
    // checking the read or write access.
    if (SkipRead && isa<MemTransferInst>(I) && GA1.isRef())
      continue;
    if (SkipWrite && isa<MemTransferInst>(I) && GA1.isMod())
      continue;

    for (GeneralAccess GA2 : AccMPI) {
      // Two parallel loads cannot race.
      if (!GA1.isMod() && !GA2.isMod())
        continue;
      LLVM_DEBUG(dbgs() << "Checking addresses " << *GA1.Loc->Ptr << " vs " <<
                 *GA2.Loc->Ptr << "\n");
      if (DependenceMightRace(
              DI.depends(&GA1, &GA2, true), GA1.I, GA2.I,
              const_cast<Value *>(GA1.getPtr()),
              const_cast<Value *>(GA2.getPtr()), MPTasks, MPTasksInLoop,
              DT, TI, LI, DL))
        return getRaceType(GA1.isMod(), GA2.isMod());
    }
  }
  return RaceType::None;
}

/// Returns true if Instruction I might race with some memory-access instruction
/// in a logically parallel task within its function.
static bool InstrMightRaceWithTask(
    Instruction *I, MPTaskListTy &MPTasks, MPTaskListTy &MPTasksInLoop,
    DenseMap<const Task *, SmallVector<Instruction *, 8>> &TaskToMemAcc,
    SmallPtrSetImpl<Instruction *> &ToInstrument,
    DenseMap<MemTransferInst *, AccessType> &MemTransferAccTypes,
    const DominatorTree &DT, const TaskInfo &TI, LoopInfo &LI,
    DependenceInfo &DI, const TargetLibraryInfo *TLI, const DataLayout &DL,
    bool SkipRead = false, bool SkipWrite = false) {
  if (!EnableStaticRaceDetection) {
    // Ensure that the potentially-racing instructions are marked for
    // instrumentation.
    ToInstrument.insert(I);
    if (MemTransferInst *M = dyn_cast<MemTransferInst>(I))
      MemTransferAccTypes[M] = AccessType::ReadWrite;
    return true;
  }

  LLVM_DEBUG(dbgs() << "Checking for races with " << *I << ", in "
             << I->getParent()->getName() << "\n");

  // Collect all logically parallel tasks.
  SmallPtrSet<const Task *, 2> AllMPTasks;
  Spindle *CurrSpindle = TI.getSpindleFor(I->getParent());
  while (CurrSpindle) {
    for (const Task *MPT : MPTasks[CurrSpindle])
      AllMPTasks.insert(MPT);
    const Task *ParentTask = CurrSpindle->getParentTask();
    if (ParentTask->isRootTask())
      CurrSpindle = nullptr;
    else
      CurrSpindle = TI.getSpindleFor(ParentTask->getDetach()->getParent());
  }

  // Check the instructions in each logically parallel task for potential races.
  for (const Task *MPT : AllMPTasks) {
    LLVM_DEBUG(dbgs() << "MaybeParallel Task @ " << MPT->getEntry()->getName()
               << "\n");
    for (Instruction *MPI : TaskToMemAcc[MPT]) {
      LLVM_DEBUG(dbgs() << "Checking instructions " << *I << " vs. " << *MPI
                 << "\n");
      RaceType RT =
        InstrsMightRace(I, MPI, MPTasks, MPTasksInLoop,
                        DI, DT, TI, LI, TLI, DL, SkipRead, SkipWrite);
      if (RaceType::None == RT)
        continue;

      LLVM_DEBUG(dbgs() << "Treating instructions as possibly racing\n");

      // Ensure that the potentially-racing instructions are marked for
      // instrumentation.
      ToInstrument.insert(I);
      ToInstrument.insert(MPI);

      // For memory transfer intrinsics, we want to remember whether the read
      // access or the write access caused the race.
      if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
        if (!MemTransferAccTypes.count(M))
          MemTransferAccTypes[M] = getAccessType(RT, 0);
        else
          MemTransferAccTypes[M] = unionAccessType(MemTransferAccTypes[M],
                                                   getAccessType(RT, 0));
      }
      if (MemTransferInst *M = dyn_cast<MemTransferInst>(MPI)) {
        if (!MemTransferAccTypes.count(M))
          MemTransferAccTypes[M] = getAccessType(RT, 1);
        else
          MemTransferAccTypes[M] = unionAccessType(MemTransferAccTypes[M],
                                                   getAccessType(RT, 1));
      }
      return true;
    }
  }
  return false;
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
      if (A->hasByValAttr() || A->hasNoAliasAttr())
        continue;

    LLVM_DEBUG(dbgs() << "Non-local base object " << *BaseObj << "\n");
    return false;
  }

  return true;
}

/// Returns true if Addr can only refer to a locally allocated base object, that
/// is, an object created via an AllocaInst or an AllocationFn.
static bool LocalBaseObj(const CallBase *CS, const DataLayout &DL,
                         LoopInfo *LI, const TargetLibraryInfo *TLI) {
  // Check whether all pointer arguments point to local memory, and
  // ignore calls that only access local memory.
  for (auto CI = CS->arg_begin(), CE = CS->arg_end(); CI != CE; ++CI) {
    Value *Arg = *CI;
    if (!Arg->getType()->isPtrOrPtrVectorTy())
      continue;

    if (!LocalBaseObj(Arg, DL, LI, TLI))
      return false;
  }
  return true;
}

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

/// Returns true if any address referenced by the callsite could race due to
/// pointer capture.
static bool PossibleRaceByCapture(const CallBase *CS, const DataLayout &DL,
                                  const TaskInfo &TI, LoopInfo *LI) {
  // Check whether all pointer arguments point to local memory, and
  // ignore calls that only access local memory.
  for (auto CI = CS->arg_begin(), CE = CS->arg_end(); CI != CE; ++CI) {
    Value *Arg = *CI;
    if (!Arg->getType()->isPtrOrPtrVectorTy())
      continue;

    if (PossibleRaceByCapture(Arg, DL, TI, LI))
      return true;
  }
  return false;
}

/// Examine the given vectors of loads, stores, memory-intrinsic calls, and
/// function calls to determine which ones might race.
bool CilkSanitizerImpl::GetMaybeRacingAccesses(
    SmallPtrSetImpl<Instruction *> &ToInstrument,
    DenseMap<MemTransferInst *, AccessType> &MemTransferAccTypes,
    SmallVectorImpl<Instruction *> &NoRaceCallsites,
    SmallVectorImpl<Instruction *> &LoadsAndStores,
    SmallVectorImpl<Instruction *> &Atomics,
    SmallVectorImpl<Instruction *> &MemIntrinCalls,
    SmallVectorImpl<Instruction *> &Callsites,
    DenseMap<const Task *, SmallVector<Instruction *, 8>> &TaskToMemAcc,
    MPTaskListTy &MPTasks, const DominatorTree &DT, const TaskInfo &TI,
    LoopInfo &LI, DependenceInfo &DI) {
  SmallVector<Instruction *, 8> MaybeNoRaceCallsites;
  MaybeParallelTasksInLoopBody MPTasksInLoop(LI);
  TI.evaluateParallelState<MaybeParallelTasksInLoopBody>(MPTasksInLoop);
  AliasAnalysis *AA =  DI.getAA();
  bool MayRaceInternally = false;

  // Look for accesses that might race against the loads and stores.
  for (Instruction *I : LoadsAndStores) {
    if (ToInstrument.count(I))
      continue;

    LLVM_DEBUG(dbgs() << "Looking for racing accesses with " << *I << "\n");
    MemoryLocation Loc = MemoryLocation::get(I);
    if (AA->pointsToConstantMemory(Loc)) {
      NumOmittedReadsFromConstants++;
      continue;
    }
    Value *Addr = const_cast<Value *>(Loc.Ptr);

    if (!PossibleRaceByCapture(Addr, DL, TI, &LI)) {
      // The variable is addressable but not captured, so it cannot be
      // referenced from a different strand and participate in a race (see
      // llvm/Analysis/CaptureTracking.h for details).
      NumOmittedNonCaptured++;
      continue;
    }

    // If the underlying object is not locally allocated, then it can race with
    // accesses in other functions.
    if (!LocalBaseObj(Addr, DL, &LI, TLI)) {
      LLVM_DEBUG(dbgs() << "Non-local base object of load/store\n");
      ToInstrument.insert(I);
      // continue;
    }

    if (InstrMightRaceWithTask(
            I, MPTasks, MPTasksInLoop.TaskList, TaskToMemAcc,
            ToInstrument, MemTransferAccTypes, DT, TI, LI, DI, TLI, DL)) {
      LLVM_DEBUG(dbgs() << "Possible internal race with load/store\n");
      MayRaceInternally = true;
    } else
      NumOmittedStaticNoRace++;
  }

  // Look for accesses that might race against the loads and stores.
  for (Instruction *I : Atomics) {
    if (ToInstrument.count(I))
      continue;

    LLVM_DEBUG(dbgs() << "Looking for racing accesses with " << *I << "\n");

    MemoryLocation Loc = MemoryLocation::get(I);
    if (AA->pointsToConstantMemory(Loc)) {
      NumOmittedReadsFromConstants++;
      continue;
    }
    Value *Addr = const_cast<Value *>(Loc.Ptr);

    if (!PossibleRaceByCapture(Addr, DL, TI, &LI)) {
      // The variable is addressable but not captured, so it cannot be
      // referenced from a different strand and participate in a race (see
      // llvm/Analysis/CaptureTracking.h for details).
      NumOmittedNonCaptured++;
      continue;
    }

    // If the underlying object is not locally allocated, then it can race with
    // accesses in other functions.
    if (!LocalBaseObj(Addr, DL, &LI, TLI)) {
      ToInstrument.insert(I);
      // continue;
    }

    if (InstrMightRaceWithTask(
            I, MPTasks, MPTasksInLoop.TaskList, TaskToMemAcc,
            ToInstrument, MemTransferAccTypes, DT, TI, LI, DI, TLI, DL))
      MayRaceInternally = true;
    else
      NumOmittedStaticNoRace++;
  }

  // Look for accesses that might race against the memory intrinsic calls.
  for (Instruction *I : MemIntrinCalls) {
    if (ToInstrument.count(I))
      continue;

    LLVM_DEBUG(dbgs() << "Looking for racing accesses with " << *I << "\n");
    if (MemSetInst *M = dyn_cast<MemSetInst>(I)) {
      Value *Addr = M->getArgOperand(0);

      if (!PossibleRaceByCapture(Addr, DL, TI, &LI)) {
        // The variable is addressable but not captured, so it cannot be
        // referenced from a different strand and participate in a race (see
        // llvm/Analysis/CaptureTracking.h for details).
        NumOmittedNonCaptured++;
        continue;
      }

      // If the underlying object is not locally allocated, then it can race
      // with accesses in other functions.
      if (!LocalBaseObj(Addr, DL, &LI, TLI)) {
        LLVM_DEBUG(dbgs() << "MemSetInst base object is not local\n");
        ToInstrument.insert(I);
        // continue;
      }

      if (InstrMightRaceWithTask(
              I, MPTasks, MPTasksInLoop.TaskList, TaskToMemAcc,
              ToInstrument, MemTransferAccTypes, DT, TI, LI, DI, TLI, DL))
        MayRaceInternally = true;
      else
        NumOmittedStaticNoRace++;

    } else if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
      bool SkipRead = false, SkipWrite = false;
      MemoryLocation ReadLoc = MemoryLocation::getForSource(M);
      MemoryLocation WriteLoc = MemoryLocation::getForDest(M);
      // Try to quickly determine if instrumentation is needed for the read
      // access.
      if (AA->pointsToConstantMemory(ReadLoc)) {
        NumOmittedReadsFromConstants++;
        SkipRead = true;
      } else if (!PossibleRaceByCapture(const_cast<Value *>(ReadLoc.Ptr),
                                        DL, TI, &LI)) {
        // The variable is addressable but not captured, so it cannot be
        // referenced from a different strand and participate in a race (see
        // llvm/Analysis/CaptureTracking.h for details).
        NumOmittedNonCaptured++;
        SkipRead = true;
      } else if (!LocalBaseObj(const_cast<Value *>(ReadLoc.Ptr), DL, &LI, TLI)) {
        LLVM_DEBUG(dbgs() << "MemTransferInst load base object is not local\n");
        ToInstrument.insert(I);
        if (!MemTransferAccTypes.count(M))
          MemTransferAccTypes[M] = AccessType::Read;
        else
          MemTransferAccTypes[M] = setRead(MemTransferAccTypes[M]);
        // SkipRead = true;
      }
      // Try to quickly determine if instrumentation is needed for the write
      // access.  Nothing can write to constant memory, so just check the local
      // base object.
      if (!PossibleRaceByCapture(const_cast<Value *>(WriteLoc.Ptr),
                                 DL, TI, &LI)) {
          // The variable is addressable but not captured, so it cannot be
          // referenced from a different strand and participate in a race (see
          // llvm/Analysis/CaptureTracking.h for details).
          NumOmittedNonCaptured++;
          SkipWrite = true;
      } else if (!LocalBaseObj(const_cast<Value *>(WriteLoc.Ptr), DL, &LI, TLI)) {
        LLVM_DEBUG(dbgs() << "MemTransferInst store base object is not local\n");
        ToInstrument.insert(I);
        if (!MemTransferAccTypes.count(M))
          MemTransferAccTypes[M] = AccessType::Write;
        else
          MemTransferAccTypes[M] = setWrite(MemTransferAccTypes[M]);
        // SkipWrite = true;
      }

      // If necessary, check for potential races against this memcpy, skipping
      // appropriate component accesses.
      if (!SkipRead || !SkipWrite) {
        if (InstrMightRaceWithTask(
                I, MPTasks, MPTasksInLoop.TaskList, TaskToMemAcc,
                ToInstrument, MemTransferAccTypes,
                DT, TI, LI, DI, TLI, DL, SkipRead, SkipWrite))
          MayRaceInternally = true;
        else
          NumOmittedStaticNoRace++;
      }
    }
  }

  // Look for accesses that might race against general function calls.
  for (Instruction *I : Callsites) {
    if (ToInstrument.count(I))
      continue;

    LLVM_DEBUG(dbgs() << "Looking for racing accesses with " << *I << "\n");
    const CallBase *CS = cast<CallBase>(I);
    if (AA->doesNotAccessMemory(CS)) {
      // The callsite I might invoke a function that contains an internal race,
      // so we can't exclude it just yet.
      MaybeNoRaceCallsites.push_back(I);
      continue;
    }

    auto CSB = AA->getModRefBehavior(CS);
    if (!ToInstrument.count(I) && AliasAnalysis::onlyAccessesArgPointees(CSB) &&
        !PossibleRaceByCapture(CS, DL, TI, &LI)) {
      // The variable is addressable but not captured, so it cannot be
      // referenced from a different strand and participate in a race (see
      // llvm/Analysis/CaptureTracking.h for details).
      MaybeNoRaceCallsites.push_back(I);
      continue;
    }

    // If we don't have insight into the called function, then we must
    // instrument this call incase the called function itself contains a race.
    const Function *Called = CS->getCalledFunction();
    if (!Called) {
      LLVM_DEBUG(dbgs() << "Missing called function for call.\n");
      ToInstrument.insert(I);
      continue;
    }

    if (AssumeRaceFreeLibraryFunctions) {
      LibFunc LF;
      if (!TLI->getLibFunc(*Called, LF))
        ToInstrument.insert(I);
      else
        LLVM_DEBUG(dbgs() << "Assuming race-free library function " <<
                   Called->getName() << ".\n");
      // } else {
      //   LLVM_DEBUG(dbgs() << "Missing exact definition of called function.\n");
      //   ToInstrument.insert(I);
      // }
      // // continue;
    }

    // If the function accesses more than just its function arguments, then
    // analyzing local accesses can't prove that this function call does not
    // participate in a race.
    if (!IgnoreInaccessibleMemory &&
        AliasAnalysis::doesAccessInaccessibleMem(CSB)) {
      LLVM_DEBUG(dbgs() << "Call access inaccessible memory\n");
      ToInstrument.insert(I);
      // continue;
    } else if (!((!IgnoreInaccessibleMemory &&
                  AliasAnalysis::onlyAccessesArgPointees(CSB)) ||
                 (IgnoreInaccessibleMemory &&
                  AliasAnalysis::onlyAccessesInaccessibleOrArgMem(CSB)))) {
      LLVM_DEBUG(dbgs() << "Call access memory other than pointees\n");
      ToInstrument.insert(I);
      // continue;
    } else if (!LocalBaseObj(CS, DL, &LI, TLI)) {
      // If the callsite accesses an underlying object that is not locally
      // allocated, then it can race with accesses in other functions.
      LLVM_DEBUG(dbgs() << "CS accesses non-local object.\n");
      ToInstrument.insert(I);
      // continue;
    }

    if (InstrMightRaceWithTask(
            I, MPTasks, MPTasksInLoop.TaskList, TaskToMemAcc,
            ToInstrument, MemTransferAccTypes, DT, TI, LI, DI, TLI, DL))
      MayRaceInternally = true;
    else if (!ToInstrument.count(I))
      MaybeNoRaceCallsites.push_back(I);
  }

  // Determine callsites that are guaranteed not to participate in a race.
  for (Instruction *I : MaybeNoRaceCallsites)
    if (!ToInstrument.count(I))
      NoRaceCallsites.push_back(I);

  return MayRaceInternally;
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
Value *CilkSanitizerImpl::GetCalleeFuncID(Function *Callee, IRBuilder<> &IRB) {
  if (!Callee)
    // Unknown targets (i.e. indirect calls) are always unknown.
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
    SmallPtrSetImpl<Instruction *> &FreeCalls) {
  bool Result = false;
  SmallPtrSet<SyncInst *, 4> Syncs;

  // Instrument allocas and allocation-function calls that may be involved in a
  // race.
  for (Instruction *I : Allocas) {
    // The simple instrumentor just instruments everyting
    CilkSanImpl.instrumentAlloca(I);
    getDetachesForInstruction(I);
  }
  for (Instruction *I : AllocationFnCalls) {
    // The simple instrumentor just instruments everyting
    CilkSanImpl.instrumentAllocationFn(I, DT);
    getDetachesForInstruction(I);
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
        continue;
      }
    }
    // The simple instrumentor just instruments everyting
    CilkSanImpl.instrumentFree(I);
    getDetachesForInstruction(I);
  }

  // Instrument detaches
  for (DetachInst *DI : Detaches) {
    CilkSanImpl.instrumentDetach(DI, DT, TI);
    // Get syncs associated with this detach
    for (SyncInst *SI : CilkSanImpl.DetachToSync[DI])
      Syncs.insert(SI);
  }

  // Instrument associated syncs
  for (SyncInst *SI : Syncs)
    CilkSanImpl.instrumentSync(SI);

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
void CilkSanitizerImpl::Instrumentor::InsertArgSuppressionFlags(
    Function &F, Value *FuncId) {
  LLVM_DEBUG(dbgs() << "InsertArgSuppressionFlags: " << F.getName() << "\n");
  IRBuilder<> IRB(&*(++(cast<Instruction>(FuncId)->getIterator())));
  unsigned ArgIdx = 0;
  for (Argument &Arg : F.args()) {
    if (!Arg.getType()->isPtrOrPtrVectorTy())
      continue;

    // Create a new flag for this argument suppression.
    Value *NewFlag = IRB.CreateAlloca(getSuppressionIRValue(IRB, 0)->getType(),
                                      Arg.getType()->getPointerAddressSpace());
    // If this function is main, then it has no ancestors that can create races.
    if (F.getName() == "main")
      IRB.CreateStore(
          getSuppressionIRValue(IRB, RaceTypeToFlagVal(RaceInfo::None)),
          NewFlag);
    else {
      // Call the runtime function to set the value of this flag.
      IRB.CreateCall(CilkSanImpl.GetSuppressionFlag, {NewFlag, FuncId,
                                                      IRB.getInt8(ArgIdx)});

      // Incorporate local information into this suppression value.
      unsigned LocalSV = static_cast<unsigned>(SuppressionVal::NoAccess);
      if (Arg.hasNoAliasAttr())
        LocalSV |= static_cast<unsigned>(SuppressionVal::NoAlias);

      // if (!CilkSanImpl.FnDoesNotRecur(F) &&
      if (!F.hasFnAttribute(Attribute::NoRecurse) &&
          RI.ObjectInvolvedInRace(&Arg)) {
        LLVM_DEBUG(dbgs() << "Setting local SV in may-recurse function " <<
                   F.getName() << " for arg " << Arg << "\n");
        // This function might recursively call itself, so incorporate
        // information we have about how this function reads or writes its own
        // arguments into these suppression flags.
        ModRefInfo ArgMR = RI.GetObjectMRForRace(&Arg);
        // TODO: Possibly make these checks more precise using information we
        // get from instrumenting functions previously.
        if (isRefSet(ArgMR))
          // If ref is set, then race detection found a local instruction that
          // might write arg, so we assume arg is modified.
          LocalSV |= static_cast<unsigned>(SuppressionVal::Mod);
        if (isModSet(ArgMR))
          // If mod is set, then race detection found a local instruction that
          // might read or write  arg, so we assume arg is read.
          LocalSV |= static_cast<unsigned>(SuppressionVal::Ref);
      }
      // Store this local suppression value.
      IRB.CreateStore(IRB.CreateOr(getSuppressionIRValue(IRB, LocalSV),
                                   IRB.CreateLoad(NewFlag)), NewFlag);

    }
    // Associate this flag with the argument for future lookups.
    LLVM_DEBUG(dbgs() << "Recording local suppression for arg " << Arg << ": "
               << *NewFlag << "\n");
    LocalSuppressions[&Arg] = NewFlag;
    ArgSuppressionFlags.insert(NewFlag);
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
  return IRB.CreateLoad(V);
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
    MemoryLocation Loc, const RaceInfo::RaceData &RD, Value *Obj,
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
    SmallPtrSet<Value *, 1> OtherObjects;
    RI.getObjectsFor(OtherRD.Access, OtherObjects);
    for (Value *OtherObj : OtherObjects) {
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
  Value *SuppressionVal = getSuppressionIRValue(IRB, SuppressionVal::NoAccess);
  Value *DefaultSuppression = getSuppressionIRValue(IRB, DefaultSV);

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
  Value *NoAliasFlag = getSuppressionIRValue(IRB, SuppressionVal::NoAlias);
  // Check the recorded race data for I.
  for (const RaceInfo::RaceData &RD : RI.getRaceData(I)) {
    if (OperandNum != RD.OperandNum)
      continue;

    SmallPtrSet<Value *, 1> Objects;
    RI.getObjectsFor(RD.Access, Objects);
    // // Add objects to CilkSanImpl.ObjectMRForRace, to ensure ancillary
    // // instrumentation is added.
    // for (Value *Obj : Objects)
    //   if (!CilkSanImpl.ObjectMRForRace.count(Obj))
    //     CilkSanImpl.ObjectMRForRace[Obj] = ModRefInfo::ModRef;

    // Get suppressions from objects
    for (Value *Obj : Objects) {
      // If we find an object with no suppression, give up.
      if (!LocalSuppressions.count(Obj)) {
        LLVM_DEBUG(dbgs() << "No local suppression found for obj " << *Obj
                   << "\n");
        return DefaultSuppression;
      }

      Value *FlagLoad = readSuppressionVal(LocalSuppressions[Obj], IRB);
      Value *FlagCheck = IRB.CreateAnd(
          FlagLoad, getSuppressionIRValue(IRB, RaceTypeToFlagVal(RD.Type)));
      SuppressionVal = IRB.CreateOr(SuppressionVal, FlagCheck);
      // SuppressionVal = IRB.CreateOr(SuppressionVal, FlagLoad);

      // Get the dynamic no-alias bit from the suppression value.
      Value *ObjNoAliasFlag = IRB.CreateAnd(
          FlagLoad, getSuppressionIRValue(IRB, SuppressionVal::NoAlias));
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
          SuppressionVal = IRB.CreateOr(SuppressionVal, FlagCheck);
          // Value *FlagCheck = IRB.CreateAnd(
          //     FlagLoad, getSuppressionIRValue(IRB, RaceTypeToFlagVal(RD.Type)));
          // SuppressionVal = IRB.CreateOr(SuppressionVal, FlagCheck);
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
  SuppressionVal = IRB.CreateOr(SuppressionVal, NoAliasFlag);
  return SuppressionVal;
}

Value *CilkSanitizerImpl::Instrumentor::getSuppressionCheck(
    Instruction *I, IRBuilder<> &IRB, unsigned OperandNum) {
  Function *F = I->getFunction();
  AliasAnalysis *AA = RI.getAA();
  MemoryLocation Loc = getMemoryLocation(I, OperandNum, CilkSanImpl.TLI);
  Value *SuppressionChk = IRB.getTrue();
  // Check the recorded race data for I.
  for (const RaceInfo::RaceData &RD : RI.getRaceData(I)) {
    if (OperandNum != RD.OperandNum)
      continue;

    SmallPtrSet<Value *, 1> Objects;
    RI.getObjectsFor(RD.Access, Objects);
    for (Value *Obj : Objects) {
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
              FlagLoad, getSuppressionIRValue(IRB, SuppressionVal::NoAlias)));

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
    SmallPtrSetImpl<Instruction *> &FreeCalls) {
  bool Result = false;
  SmallPtrSet<SyncInst *, 4> Syncs;

  // Instrument allocas and allocation-function calls that may be involved in a
  // race.
  for (Instruction *I : Allocas) {
    if (CilkSanImpl.ObjectMRForRace.count(I) ||
        PointerMayBeCaptured(I, true, false)) {
      CilkSanImpl.instrumentAlloca(I);
      getDetachesForInstruction(I);
    }
  }
  for (Instruction *I : AllocationFnCalls) {
    if (CilkSanImpl.ObjectMRForRace.count(I) ||
        PointerMayBeCaptured(I, true, false)) {
      CilkSanImpl.instrumentAllocationFn(I, DT);
      getDetachesForInstruction(I);
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
        continue;
      }
    }
    if (RI.ObjectInvolvedInRace(Ptr) ||
        PointerMayBeCaptured(Ptr, true, false)) {
      CilkSanImpl.instrumentFree(I);
      getDetachesForInstruction(I);
    }
  }

  // Instrument detaches
  for (DetachInst *DI : Detaches) {
    CilkSanImpl.instrumentDetach(DI, DT, TI);
    // Get syncs associated with this detach
    for (SyncInst *SI : CilkSanImpl.DetachToSync[DI])
      Syncs.insert(SI);
  }

  // Instrument associated syncs
  for (SyncInst *SI : Syncs)
    CilkSanImpl.instrumentSync(SI);

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
    LLVM_DEBUG(dbgs() << "Skipping " << F.getName() << "\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Instrumenting " << F.getName() << "\n");

  if (Options.CallsMayThrow)
    setupCalls(F);
  setupBlocks(F);

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

  DominatorTree *DT = &GetDomTree(F);
  TaskInfo &TI = GetTaskInfo(F);
  LoopInfo &LI = GetLoopInfo(F);
  RaceInfo &RI = GetRaceInfo(F);
  // Evaluate the tasks that might be in parallel with each spindle.
  MaybeParallelTasks MPTasks;
  TI.evaluateParallelState<MaybeParallelTasks>(MPTasks);

  for (Spindle *S : depth_first(TI.getRootTask()->getEntrySpindle())) {
    for (BasicBlock *BB : S->blocks()) {
      // Record the Tapir sync instructions found
      if (SyncInst *SI = dyn_cast<SyncInst>(BB->getTerminator()))
        Syncs.push_back(SI);

      // Record the memory accesses in the basic block
      for (Instruction &Inst : *BB) {
        // TODO: Handle VAArgInst
        if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst))
          LocalLoadsAndStores.push_back(&Inst);
        else if (isa<AtomicRMWInst>(Inst) || isa<AtomicCmpXchgInst>(Inst))
          AtomicAccesses.push_back(&Inst);
        else if (isa<CallInst>(Inst) || isa<InvokeInst>(Inst)) {
          // if (CallInst *CI = dyn_cast<CallInst>(&Inst))
          //   maybeMarkSanitizerLibraryCallNoBuiltin(CI, TLI);

          // Record this function call as either an allocation function, a call
          // to free (or delete), a memory intrinsic, or an ordinary real
          // function call.
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
    SimpleInstrumentor FuncI(*this, TI, DT);
    Result |= FuncI.InstrumentSimpleInstructions(AllLoadsAndStores);
    Result |= FuncI.InstrumentSimpleInstructions(AtomicAccesses);
    Result |= FuncI.InstrumentAnyMemIntrinsics(MemIntrinCalls);
    Result |= FuncI.InstrumentCalls(Callsites);

    // Instrument ancillary instructions including allocas, allocation-function
    // calls, free calls, detaches, and syncs.
    Result |= FuncI.InstrumentAncillaryInstructions(Allocas, AllocationFnCalls,
                                                    FreeCalls);
  } else {
    Instrumentor FuncI(*this, RI, TI, DT);
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
                                                    FreeCalls);

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
    // TODO: Determine if we actually want the frame pointer, not the stack
    // pointer.
    Value *FrameAddr = IRB.CreateCall(
        Intrinsic::getDeclaration(&M, Intrinsic::frameaddress),
        {IRB.getInt32(0)});
    Value *StackSave = IRB.CreateCall(
        Intrinsic::getDeclaration(&M, Intrinsic::stacksave));
    IRB.CreateCall(CsanFuncEntry,
                   {FuncId, FrameAddr, StackSave, FuncEntryProp.getValue(IRB)});
    // IRB.CreateCall(CsanFuncEntry,
    //                {FuncId, FrameAddr, FuncEntryProp.getValue(IRB)});

    EscapeEnumerator EE(F, "csan_cleanup", false);
    while (IRBuilder<> *AtExit = EE.Next()) {
      // uint64_t ExitLocalId = FunctionExitFED.add(F);
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

bool CilkSanitizerImpl::prepareToInstrumentFunction(Function &F) {
  if (F.empty() || shouldNotInstrumentFunction(F) ||
      !CheckSanitizeCilkAttr(F))
    return false;

  if (Options.CallsMayThrow)
    setupCalls(F);
  setupBlocks(F);

  SmallVector<Instruction *, 8> AllLoadsAndStores;
  SmallVector<Instruction *, 8> LocalLoadsAndStores;
  SmallVector<Instruction *, 8> AtomicAccesses;
  SmallVector<Instruction *, 8> MemIntrinCalls;
  SmallVector<Instruction *, 8> Callsites;
  SmallVector<SyncInst *, 8> Syncs;

  DominatorTree *DT = &GetDomTree(F);
  TaskInfo &TI = GetTaskInfo(F);
  LoopInfo &LI = GetLoopInfo(F);
  DependenceInfo &DI = GetDepInfo(F);

  // Evaluate the tasks that might be in parallel with each spindle.
  MaybeParallelTasks MPTasks;
  TI.evaluateParallelState<MaybeParallelTasks>(MPTasks);

  for (Spindle *S : depth_first(TI.getRootTask()->getEntrySpindle())) {
    for (BasicBlock *BB : S->blocks()) {
      // Record the Tapir sync instructions found
      if (SyncInst *SI = dyn_cast<SyncInst>(BB->getTerminator()))
        Syncs.push_back(SI);

      // Record the memory accesses in the basic block
      for (Instruction &Inst : *BB) {
        // TODO: Handle VAArgInst
        if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst))
          LocalLoadsAndStores.push_back(&Inst);
        else if (isa<AtomicRMWInst>(Inst) || isa<AtomicCmpXchgInst>(Inst))
          AtomicAccesses.push_back(&Inst);
        else if (isa<CallInst>(Inst) || isa<InvokeInst>(Inst)) {
          // if (CallInst *CI = dyn_cast<CallInst>(&Inst))
          //   maybeMarkSanitizerLibraryCallNoBuiltin(CI, TLI);

          // Record this function call as either an allocation function, a call
          // to free (or delete), a memory intrinsic, or an ordinary real
          // function call.
          if (isAllocationFn(&Inst, TLI, /*LookThroughBitCast=*/false,
                             /*IgnoreBuiltinAttr=*/true))
            AllocationFnCalls.push_back(&Inst);
          else if (isFreeCall(&Inst, TLI))
            FreeCalls.push_back(&Inst);
          else if (isa<MemIntrinsic>(Inst))
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
          Allocas.push_back(&Inst);
        }
      }
      chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores, TI,
                                     LI);
    }
  }

  // Record which memory-accessing instructions that might race are associated
  // with each task.
  DenseMap<const Task *, SmallVector<Instruction *, 8>> TaskToMemAcc;
  for (Instruction *I : AllLoadsAndStores)
    TaskToMemAcc[TI.getTaskFor(I->getParent())].push_back(I);
  for (Instruction *I : AtomicAccesses)
    TaskToMemAcc[TI.getTaskFor(I->getParent())].push_back(I);
  for (Instruction *I : MemIntrinCalls)
    TaskToMemAcc[TI.getTaskFor(I->getParent())].push_back(I);
  for (Instruction *I : Callsites)
    TaskToMemAcc[TI.getTaskFor(I->getParent())].push_back(I);
  for (Instruction *I : FreeCalls)
    TaskToMemAcc[TI.getTaskFor(I->getParent())].push_back(I);

  // Perform static race detection among the memory-accessing instructions to
  // determine which require instrumentation.
  bool MayRaceInternally =
    GetMaybeRacingAccesses(ToInstrument, MemTransferAccTypes,
                           NoRaceCallsites, AllLoadsAndStores,
                           AtomicAccesses, MemIntrinCalls, Callsites,
                           TaskToMemAcc, MPTasks.TaskList, *DT, TI, LI, DI);

  // Record this function if instrumentation is needed on a detach or a callsite
  // it contains.
  if (!EnableStaticRaceDetection)
    MayRaceFunctions.insert(&F);
  else if (MayRaceInternally)
    MayRaceFunctions.insert(&F);

  // While we have the maybe-parallel-task information handy, use it to map each
  // detach instruction with the sync instructions that could sync it.
  for (SyncInst *Sync : Syncs)
    for (const Task *MPT :
           MPTasks.TaskList[TI.getSpindleFor(Sync->getParent())])
      DetachToSync[MPT->getDetach()].push_back(Sync);

  return true;
}

void CilkSanitizerImpl::determineCallSitesToInstrument() {
  // The NoRaceCallsites set identifies callsites that cannot race, based on
  // function-local analysis.  If those called functions might themselves
  // contain races, however, then they must be instrumented.  Hence we use
  // information about which functions were instrumented for race detection to
  // determine which of these callsites must be instrumented and which can be
  // suppressed.

  // Organize the call sites that might not race by called function.
  DenseMap<const Function *, SmallVector<Instruction *, 8>> FunctionToCallSite;
  for (Instruction *I : NoRaceCallsites) {
    if (ToInstrument.count(I))
      continue;
    ImmutableCallSite CS(I);
    if (const Function *F = CS.getCalledFunction())
      FunctionToCallSite[F].push_back(I);
  }

  // Evaluate the SCC's in the callgraph in post order to determine callsites
  // that reach instrumented functions and therefore need to be instrumented.
  for (scc_iterator<CallGraph *> I = scc_begin(CG); !I.isAtEnd(); ++I) {
    const std::vector<CallGraphNode *> &SCC = *I;
    assert(!SCC.empty() && "SCC with no functions!");
    for (auto *CGN : SCC) {
      if (Function *F = CGN->getFunction()) {
        if (MayRaceFunctions.count(F)) {
          for (Instruction *Call : FunctionToCallSite[F]) {
            ToInstrument.insert(Call);
            MayRaceFunctions.insert(Call->getParent()->getParent());
          }
        }
      }
    }
  }
}

bool CilkSanitizerImpl::instrumentFunction(
    Function &F, SmallPtrSetImpl<Instruction *> &ToInstrument,
    SmallPtrSetImpl<Instruction *> &NoRaceCallsites) {
  if (F.empty() || shouldNotInstrumentFunction(F) ||
      !CheckSanitizeCilkAttr(F))
    return false;
  bool Res = false;

  // Get analyses
  DominatorTree *DT = &GetDomTree(F);
  TaskInfo &TI = GetTaskInfo(F);
  LoopInfo &LI = GetLoopInfo(F);
  DependenceInfo &DI = GetDepInfo(F);

  // Function-local sets of instructions to instrument.
  SmallPtrSet<Instruction *, 8> MayRaceLoadsAndStores;
  SmallPtrSet<Instruction *, 8> MayRaceAtomics;
  SmallPtrSet<Instruction *, 8> MayRaceMemIntrinCalls;
  SmallPtrSet<Instruction *, 8> MayRaceCallsites;
  SmallPtrSet<DetachInst *, 8> MayRaceDetaches;
  SmallPtrSet<Instruction *, 8> MayRaceAllocas;
  SmallPtrSet<Instruction *, 8> MayRaceAllocFns;
  SmallPtrSet<Instruction *, 8> MayRaceFreeCalls;
  SmallPtrSet<SyncInst *, 8> MayRaceSyncs;

  // Partition the set of memory-accessing instructions that might race by type:
  // load or store, memory intrinsic, callsite, etc.
  SmallPtrSet<const Value *, 8> LocalAllocToInstrument;
  for (Instruction *I : ToInstrument) {
    // TODO: Handle VAArgs
    if (isa<LoadInst>(I) || isa<StoreInst>(I))
      MayRaceLoadsAndStores.insert(I);
    else if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I))
      MayRaceAtomics.insert(I);
    else if (isa<MemIntrinsic>(I))
      MayRaceMemIntrinCalls.insert(I);
    else if (isa<CallInst>(I) || isa<InvokeInst>(I))
      MayRaceCallsites.insert(I);

    // Record detaches associated with instruction I as necessary to instrument.
    Task *T = TI.getTaskFor(I->getParent());
    while (DetachInst *DI = T->getDetach()) {
      MayRaceDetaches.insert(DI);
      for (SyncInst *Sync : DetachToSync[DI])
        MayRaceSyncs.insert(Sync);
      if (!T->getParentTask())
        break;
      T = T->getParentTask();
    }

    // Determine local allocations to instrument.
    SmallVector<Value *, 1> BaseObjs;
    if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
      if (isReadSet(MemTransferAccTypes[M]))
        // Get the base object read by this memcpy.
        GetUnderlyingObjects(M->getArgOperand(1), BaseObjs, DL, &LI, 0);
      if (isWriteSet(MemTransferAccTypes[M]))
        // Get the base object written by this memcpy.
        GetUnderlyingObjects(M->getArgOperand(0), BaseObjs, DL, &LI, 0);
    } else {
      SmallVector<GeneralAccess, 2> AccI;
      GetGeneralAccesses(I, AccI, DI.getAA(), TLI);
      for (GeneralAccess GA : AccI)
        // Get the base objects that this address might refer to.
        GetUnderlyingObjects(const_cast<Value *>(GA.Loc->Ptr), BaseObjs,
                             DL, &LI, 0);
    }
    // Record any local allocations among the base objects.
    for (const Value *BaseObj : BaseObjs)
      if (isa<AllocaInst>(BaseObj) || isAllocationFn(BaseObj, TLI, false, true))
        LocalAllocToInstrument.insert(BaseObj);
  }

  // Partition the local allocations to instrument into allocas and
  // allocation-function calls.
  for (Instruction *I : Allocas)
    if (LocalAllocToInstrument.count(I))
      MayRaceAllocas.insert(I);
  for (Instruction *I : AllocationFnCalls)
    if (LocalAllocToInstrument.count(I))
      MayRaceAllocFns.insert(I);

  uint64_t LocalId = getLocalFunctionID(F);

  // If the function does not access memory and we can statically prove it
  // contains no races, don't instrument it.
  if (F.doesNotAccessMemory() && MayRaceLoadsAndStores.empty() &&
      MayRaceMemIntrinCalls.empty() && MayRaceCallsites.empty() &&
      MayRaceAtomics.empty())
    return false;

  // Instrument free calls for instrumented local allocations as well as free
  // calls that might participate in a race.
  for (Instruction *FreeCall : FreeCalls) {
    // If the free call possibly participates in a race, instrument it.
    if (ToInstrument.count(FreeCall)) {
      MayRaceFreeCalls.insert(FreeCall);
      continue;
    }

    // Check the pointer argument of the free call to see if it corresponds to a
    // local allocation.
    ImmutableCallSite FreeCS(FreeCall);
    const Value *Arg = FreeCS.getArgument(0);
    if (const Instruction *AI = dyn_cast<Instruction>(Arg))
      if (MayRaceAllocFns.count(AI))
        MayRaceFreeCalls.insert(FreeCall);
  }

  // Instrument all instructions that might race and the associated memory
  // allocation and parallel control flow.
  bool MaySpawn = !TI.isSerial();

  for (auto Inst : MayRaceLoadsAndStores)
    Res |= instrumentLoadOrStore(Inst);

  for (auto Inst : MayRaceAtomics)
    Res |= instrumentAtomic(Inst);

  for (auto Inst : MayRaceMemIntrinCalls)
    Res |= instrumentMemIntrinsic(Inst, MemTransferAccTypes);

  for (auto Inst : MayRaceAllocFns)
    Res |= instrumentAllocationFn(Inst, DT);

  for (auto Inst : MayRaceFreeCalls)
    Res |= instrumentFree(Inst);

  for (auto Inst : MayRaceCallsites)
    Res |= instrumentCallsite(Inst);

  // The remaining callsites that were thought not to race can now be
  // suppressed.
  for (auto Inst : NoRaceCallsites) {
    if (!ToInstrument.count(Inst)) {
      suppressCallsite(Inst);
      NumOmittedStaticNoRace++;
    }
  }

  for (auto Inst : MayRaceDetaches)
    Res |= instrumentDetach(Inst, DT, TI);

  for (auto Inst : MayRaceSyncs)
    Res |= instrumentSync(Inst);

  for (auto Inst : MayRaceAllocas)
    Res |= instrumentAlloca(Inst);

  if (Res) {
    IRBuilder<> IRB(&*F.getEntryBlock().getFirstInsertionPt());
    CsiFuncProperty FuncEntryProp;
    FuncEntryProp.setMaySpawn(MaySpawn);
    Value *FuncId = FunctionFED.localToGlobalId(LocalId, IRB);
    // TODO: Determine if we actually want the frame pointer, not the stack
    // pointer.
    Value *StackSave = IRB.CreateCall(
        Intrinsic::getDeclaration(&M, Intrinsic::stacksave));
    IRB.CreateCall(CsanFuncEntry,
                   {FuncId, StackSave, FuncEntryProp.getValue(IRB)});
    // Value *FrameAddr = IRB.CreateCall(
    //     Intrinsic::getDeclaration(&M, Intrinsic::frameaddress),
    //     {IRB.getInt32(0)});
    // IRB.CreateCall(CsanFuncEntry,
    //                {FuncId, FrameAddr, FuncEntryProp.getValue(IRB)});

    EscapeEnumerator EE(F, "csan_cleanup", false);
    while (IRBuilder<> *AtExit = EE.Next()) {
      // uint64_t ExitLocalId = FunctionExitFED.add(F);
      uint64_t ExitLocalId = FunctionExitFED.add(*AtExit->GetInsertPoint());
      Value *ExitCsiId = FunctionExitFED.localToGlobalId(ExitLocalId, *AtExit);
      CsiFuncExitProperty FuncExitProp;
      FuncExitProp.setMaySpawn(MaySpawn);
      FuncExitProp.setEHReturn(isa<ResumeInst>(AtExit->GetInsertPoint()));
      AtExit->CreateCall(CsanFuncExit,
                         {ExitCsiId, FuncId, FuncExitProp.getValue(*AtExit)});
    }

    updateInstrumentedFnAttrs(F);
    F.removeFnAttr(Attribute::SanitizeCilk);
  }

  return Res;
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

bool CilkSanitizerImpl::instrumentMemIntrinsic(
    Instruction *I,
    DenseMap<MemTransferInst *, AccessType> &MemTransferAccTypes) {
  CsiLoadStoreProperty Prop;
  IRBuilder<> IRB(I);
  if (MemSetInst *M = dyn_cast<MemSetInst>(I)) {
    Value *Addr = M->getArgOperand(0);

    if (ConstantInt *CI = dyn_cast<ConstantInt>(M->getArgOperand(3)))
      Prop.setAlignment(CI->getZExtValue());
    uint64_t LocalId = StoreFED.add(*I);
    uint64_t StoreObjId = StoreObj.add(*I, GetUnderlyingObject(Addr, DL));
    assert(LocalId == StoreObjId &&
           "Store received different ID's in FED and object tables.");
    Value *CsiId = StoreFED.localToGlobalId(LocalId, IRB);
    Value *Args[] = {CsiId,
                     IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                     IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false),
                     Prop.getValue(IRB)};
    Instruction *Call = IRB.CreateCall(CsanLargeWrite, Args);
    IRB.SetInstDebugLocation(Call);
    return true;

  } else if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
    // Only instrument the large load and the large store components as
    // necessary.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(M->getArgOperand(3)))
      Prop.setAlignment(CI->getZExtValue());
    Value *StoreAddr = M->getArgOperand(0);
    Value *LoadAddr = M->getArgOperand(1);
    bool Instrumented = false;

    if (isWriteSet(MemTransferAccTypes[M])) {
      // Instrument the store
      uint64_t StoreId = StoreFED.add(*I);
      uint64_t StoreObjId =
        StoreObj.add(*I, GetUnderlyingObject(StoreAddr, DL));
      assert(StoreId == StoreObjId &&
             "Store received different ID's in FED and object tables.");
      Value *StoreCsiId = StoreFED.localToGlobalId(StoreId, IRB);
      Value *StoreArgs[] =
        {StoreCsiId, IRB.CreatePointerCast(StoreAddr, IRB.getInt8PtrTy()),
         IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false),
         Prop.getValue(IRB)};
      Instruction *WriteCall = IRB.CreateCall(CsanLargeWrite, StoreArgs);
      IRB.SetInstDebugLocation(WriteCall);
      Instrumented = true;
    }

    if (isReadSet(MemTransferAccTypes[M])) {
      // Instrument the load
      uint64_t LoadId = LoadFED.add(*I);
      uint64_t LoadObjId = LoadObj.add(*I, GetUnderlyingObject(LoadAddr, DL));
      assert(LoadId == LoadObjId &&
             "Load received different ID's in FED and object tables.");
      Value *LoadCsiId = StoreFED.localToGlobalId(LoadId, IRB);
      Value *LoadArgs[] =
        {LoadCsiId, IRB.CreatePointerCast(LoadAddr, IRB.getInt8PtrTy()),
         IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false),
         Prop.getValue(IRB)};
      Instruction *ReadCall = IRB.CreateCall(CsanLargeRead, LoadArgs);
      IRB.SetInstDebugLocation(ReadCall);
      Instrumented = true;
    }
    return Instrumented;
  }
  return false;
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

bool CilkSanitizerImpl::instrumentDetach(DetachInst *DI,
                                         DominatorTree *DT, TaskInfo &TI) {
  // Instrument the detach instruction itself
  Value *DetachID;
  {
    IRBuilder<> IRB(DI);
    uint64_t LocalID = DetachFED.add(*DI);
    DetachID = DetachFED.localToGlobalId(LocalID, IRB);
    Instruction *Call = IRB.CreateCall(CsanDetach, {DetachID});
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
    // TODO: Determine if we actually want the frame pointer, not the stack
    // pointer.
    Value *FrameAddr = IRB.CreateCall(
        Intrinsic::getDeclaration(&M, Intrinsic::task_frameaddress),
        {IRB.getInt32(0)});
    Value *StackSave = IRB.CreateCall(
        Intrinsic::getDeclaration(&M, Intrinsic::stacksave));
    Instruction *Call = IRB.CreateCall(CsanTaskEntry,
                                       {TaskID, DetachID, FrameAddr,
                                        StackSave});
    // Instruction *Call = IRB.CreateCall(CsanTaskEntry,
    //                                    {TaskID, DetachID, FrameAddr});
    IRB.SetInstDebugLocation(Call);

    // Instrument the exit points of the detached tasks.
    for (BasicBlock *TaskExit : TaskExits) {
      IRBuilder<> IRB(TaskExit->getTerminator());
      uint64_t LocalID = TaskExitFED.add(*TaskExit->getTerminator());
      Value *TaskExitID = TaskExitFED.localToGlobalId(LocalID, IRB);
      Instruction *Call = IRB.CreateCall(CsanTaskExit,
                                         {TaskExitID, TaskID, DetachID});
      IRB.SetInstDebugLocation(Call);
      NumInstrumentedDetachExits++;
    }
    // Instrument the EH exits of the detached task.
    for (BasicBlock *TaskExit : TaskResumes) {
      IRBuilder<> IRB(TaskExit->getTerminator());
      uint64_t LocalID = TaskExitFED.add(*TaskExit->getTerminator());
      Value *TaskExitID = TaskExitFED.localToGlobalId(LocalID, IRB);
      Instruction *Call = IRB.CreateCall(CsanTaskExit,
                                         {TaskExitID, TaskID, DetachID});
      IRB.SetInstDebugLocation(Call);
      NumInstrumentedDetachExits++;
    }

    Task *T = TI.getTaskFor(DetachedBlock);
    Value *DefaultID = getDefaultID(IRB);
    for (Spindle *SharedEH : SharedEHExits)
      insertHookCallAtSharedEHSpindleExits(
          SharedEH, T, CsanTaskExit, TaskExitFED, {TaskID, DetachID},
          {DefaultID, DefaultID});
  }

  // Instrument the continuation of the detach.
  {
    if (isCriticalContinueEdge(DI, 1))
      ContinueBlock = SplitCriticalEdge(
          DI, 1,
          CriticalEdgeSplittingOptions(DT).setSplitDetachContinue());

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

bool CilkSanitizerImpl::instrumentSync(SyncInst *SI) {
  IRBuilder<> IRB(SI);
  // Get the ID of this sync.
  uint64_t LocalID = SyncFED.add(*SI);
  Value *SyncID = SyncFED.localToGlobalId(LocalID, IRB);
  // Insert instrumentation before the sync.
  insertHookCall(SI, CsanSync, {SyncID});
  NumInstrumentedSyncs++;
  return true;
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
