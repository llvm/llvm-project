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
  CilkSanitizerImpl(Module &M, CallGraph *CG,
                    function_ref<DominatorTree &(Function &)> GetDomTree,
                    function_ref<TaskInfo &(Function &)> GetTaskInfo,
                    function_ref<LoopInfo &(Function &)> GetLoopInfo,
                    function_ref<DependenceInfo &(Function &)> GetDepInfo,
                    const TargetLibraryInfo *TLI)
      : CSIImpl(M, CG, GetDomTree, GetTaskInfo, TLI),
        GetLoopInfo(GetLoopInfo), GetDepInfo(GetDepInfo) {
    // Even though we're doing our own instrumentation, we want the CSI setup
    // for the instrumentation of function entry/exit, memory accesses (i.e.,
    // loads and stores), atomics, memory intrinsics.  We also want call sites,
    // for extracting debug information.
    Options.InstrumentBasicBlocks = false;
    // Cilksan defines its own hooks for instrumenting memory accesses, memory
    // intrinsics, and Tapir instructions, so we disable the default CSI
    // instrumentation hooks for these IR objects.
    Options.InstrumentMemoryAccesses = false;
    Options.InstrumentMemIntrinsics = false;
    Options.InstrumentTapir = false;
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
  bool instrumentLoadOrStore(Instruction *I);
  bool instrumentAtomic(Instruction *I);
  bool instrumentMemIntrinsic(
      Instruction *I,
      DenseMap<MemTransferInst *, AccessType> &MemTransferAccTypes);
  bool instrumentCallsite(Instruction *I);
  bool suppressCallsite(Instruction *I);
  bool instrumentAllocationFn(Instruction *I, DominatorTree *DT);
  bool instrumentFree(Instruction *I);
  bool instrumentDetach(DetachInst *DI, DominatorTree *DT, TaskInfo &TI);
  bool instrumentSync(SyncInst *SI);
  bool instrumentAlloca(Instruction *I);

private:
  // List of all allocation function types.  This list needs to remain
  // consistent with TargetLibraryInfo and with csan.h.
  enum class AllocFnTy
    {
     malloc = 0,
     valloc,
     calloc,
     realloc,
     reallocf,
     Znwj,
     ZnwjRKSt9nothrow_t,
     Znwm,
     ZnwmRKSt9nothrow_t,
     Znaj,
     ZnajRKSt9nothrow_t,
     Znam,
     ZnamRKSt9nothrow_t,
     msvc_new_int,
     msvc_new_int_nothrow,
     msvc_new_longlong,
     msvc_new_longlong_nothrow,
     msvc_new_array_int,
     msvc_new_array_int_nothrow,
     msvc_new_array_longlong,
     msvc_new_array_longlong_nothrow,
     LAST_ALLOCFNTY
    };

  static AllocFnTy getAllocFnTy(const LibFunc &F) {
    switch (F) {
    default: return AllocFnTy::LAST_ALLOCFNTY;
    case LibFunc_malloc: return AllocFnTy::malloc;
    case LibFunc_valloc: return AllocFnTy::valloc;
    case LibFunc_calloc: return AllocFnTy::calloc;
    case LibFunc_realloc: return AllocFnTy::realloc;
    case LibFunc_reallocf: return AllocFnTy::reallocf;
    case LibFunc_Znwj: return AllocFnTy::Znwj;
    case LibFunc_ZnwjRKSt9nothrow_t: return AllocFnTy::ZnwjRKSt9nothrow_t;
    case LibFunc_Znwm: return AllocFnTy::Znwm;
    case LibFunc_ZnwmRKSt9nothrow_t: return AllocFnTy::ZnwmRKSt9nothrow_t;
    case LibFunc_Znaj: return AllocFnTy::Znaj;
    case LibFunc_ZnajRKSt9nothrow_t: return AllocFnTy::ZnajRKSt9nothrow_t;
    case LibFunc_Znam: return AllocFnTy::Znam;
    case LibFunc_ZnamRKSt9nothrow_t: return AllocFnTy::ZnamRKSt9nothrow_t;
    case LibFunc_msvc_new_int: return AllocFnTy::msvc_new_int;
    case LibFunc_msvc_new_int_nothrow: return AllocFnTy::msvc_new_int_nothrow;
    case LibFunc_msvc_new_longlong: return AllocFnTy::msvc_new_longlong;
    case LibFunc_msvc_new_longlong_nothrow:
      return AllocFnTy::msvc_new_longlong_nothrow;
    case LibFunc_msvc_new_array_int: return AllocFnTy::msvc_new_array_int;
    case LibFunc_msvc_new_array_int_nothrow:
      return AllocFnTy::msvc_new_array_int_nothrow;
    case LibFunc_msvc_new_array_longlong:
      return AllocFnTy::msvc_new_array_longlong;
    case LibFunc_msvc_new_array_longlong_nothrow:
      return AllocFnTy::msvc_new_array_longlong_nothrow;
    }
  }

  // List of all free function types.  This list needs to remain consistent with
  // TargetLibraryInfo and with csan.h.
  enum class FreeTy
    {
     free = 0,
     ZdlPv,
     ZdlPvRKSt9nothrow_t,
     ZdlPvj,
     ZdlPvm,
     ZdaPv,
     ZdaPvRKSt9nothrow_t,
     ZdaPvj,
     ZdaPvm,
     msvc_delete_ptr32,
     msvc_delete_ptr32_nothrow,
     msvc_delete_ptr32_int,
     msvc_delete_ptr64,
     msvc_delete_ptr64_nothrow,
     msvc_delete_ptr64_longlong,
     msvc_delete_array_ptr32,
     msvc_delete_array_ptr32_nothrow,
     msvc_delete_array_ptr32_int,
     msvc_delete_array_ptr64,
     msvc_delete_array_ptr64_nothrow,
     msvc_delete_array_ptr64_longlong,
     LAST_FREETY
    };

  static FreeTy getFreeTy(const LibFunc &F) {
    switch (F) {
    default: return FreeTy::LAST_FREETY;
    case LibFunc_free: return FreeTy::free;
    case LibFunc_ZdlPv: return FreeTy::ZdlPv;
    case LibFunc_ZdlPvRKSt9nothrow_t: return FreeTy::ZdlPvRKSt9nothrow_t;
    case LibFunc_ZdlPvj: return FreeTy::ZdlPvj;
    case LibFunc_ZdlPvm: return FreeTy::ZdlPvm;
    case LibFunc_ZdaPv: return FreeTy::ZdaPv;
    case LibFunc_ZdaPvRKSt9nothrow_t: return FreeTy::ZdaPvRKSt9nothrow_t;
    case LibFunc_ZdaPvj: return FreeTy::ZdaPvj;
    case LibFunc_ZdaPvm: return FreeTy::ZdaPvm;
    case LibFunc_msvc_delete_ptr32: return FreeTy::msvc_delete_ptr32;
    case LibFunc_msvc_delete_ptr32_nothrow:
      return FreeTy::msvc_delete_ptr32_nothrow;
    case LibFunc_msvc_delete_ptr32_int: return FreeTy::msvc_delete_ptr32_int;
    case LibFunc_msvc_delete_ptr64: return FreeTy::msvc_delete_ptr64;
    case LibFunc_msvc_delete_ptr64_nothrow:
      return FreeTy::msvc_delete_ptr64_nothrow;
    case LibFunc_msvc_delete_ptr64_longlong:
      return FreeTy::msvc_delete_ptr64_longlong;
    case LibFunc_msvc_delete_array_ptr32:
      return FreeTy::msvc_delete_array_ptr32;
    case LibFunc_msvc_delete_array_ptr32_nothrow:
      return FreeTy::msvc_delete_array_ptr32_nothrow;
    case LibFunc_msvc_delete_array_ptr32_int:
      return FreeTy::msvc_delete_array_ptr32_int;
    case LibFunc_msvc_delete_array_ptr64:
      return FreeTy::msvc_delete_array_ptr64;
    case LibFunc_msvc_delete_array_ptr64_nothrow:
      return FreeTy::msvc_delete_array_ptr64_nothrow;
    case LibFunc_msvc_delete_array_ptr64_longlong:
      return FreeTy::msvc_delete_array_ptr64_longlong;
    }
  }

  // Analysis results
  function_ref<LoopInfo &(Function &)> GetLoopInfo;
  function_ref<DependenceInfo &(Function &)> GetDepInfo;

  // Instrumentation hooks
  Function *CsanFuncEntry = nullptr;
  Function *CsanFuncExit = nullptr;
  Function *CsanRead = nullptr;
  Function *CsanWrite = nullptr;
  Function *CsanLargeRead = nullptr;
  Function *CsanLargeWrite = nullptr;
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

  // CilkSanitizer custom forensic tables
  ObjectTable LoadObj, StoreObj, AllocaObj, AllocFnObj;

  SmallVector<Constant *, 4> UnitObjTables;

  SmallVector<Instruction *, 8> AllocationFnCalls;
  SmallVector<Instruction *, 8> FreeCalls;
  SmallVector<Instruction *, 8> Allocas;
  SmallPtrSet<Instruction *, 8> ToInstrument;

  DenseMap<MemTransferInst *, AccessType> MemTransferAccTypes;
  SmallVector<Instruction *, 8> NoRaceCallsites;
  SmallPtrSet<Function *, 8> MayRaceFunctions;
  DenseMap<DetachInst *, SmallVector<SyncInst *, 2>> DetachToSync;
};

/// CilkSanitizer: instrument the code in module to find races.
struct CilkSanitizerLegacyPass : public ModulePass {
  static char ID;  // Pass identification, replacement for typeid.
  CilkSanitizerLegacyPass() : ModulePass(ID) {
    initializeCilkSanitizerLegacyPassPass(*PassRegistry::getPassRegistry());
  }
  StringRef getPassName() const override {
    return "CilkSanitizer";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnModule(Module &M) override;
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
  AU.addRequired<TaskInfoWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addPreserved<BasicAAWrapperPass>();
}

ModulePass *llvm::createCilkSanitizerLegacyPass() {
  return new CilkSanitizerLegacyPass();
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
    add(ID);
    return ID;
  }

  // Next, if this is an alloca instruction, look for a llvm.dbg.declare
  // intrinsic.
  if (AllocaInst *AI = dyn_cast<AllocaInst>(Obj)) {
    TinyPtrVector<DbgInfoIntrinsic *> DbgDeclares = FindDbgAddrUses(AI);
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

  add(ID);
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

bool CilkSanitizerImpl::run() {
  // Initialize components of the CSI and Cilksan system.
  initializeCsi();
  initializeFEDTables();
  initializeCsanObjectTables();
  initializeCsanHooks();

  // Examine all functions in the module to find IR objects within each function
  // that require instrumentation.
  for (Function &F : M) {
    LLVM_DEBUG(dbgs() << "Preparing to instrument " << F.getName() << "\n");
    prepareToInstrumentFunction(F);
  }

  // Based on information of which functions might expose a race, determine
  // which callsites in the function need to be instrumented.
  determineCallSitesToInstrument();

  // Determine which callsites can have instrumentation suppressed.
  DenseMap<Function *, SmallPtrSet<Instruction *, 32>> ToInstrumentByFunction;
  DenseMap<Function *, SmallPtrSet<Instruction *, 8>> NoRaceCallsitesByFunction;
  for (Instruction *I : ToInstrument)
    ToInstrumentByFunction[I->getParent()->getParent()].insert(I);
  for (Instruction *I : NoRaceCallsites)
    if (!ToInstrument.count(I))
      NoRaceCallsitesByFunction[I->getParent()->getParent()].insert(I);

  // Insert the necessary instrumentation throughout each function in the
  // module.
  for (Function &F : M) {
    LLVM_DEBUG(dbgs() << "Instrumenting " << F.getName() << "\n");
    instrumentFunction(F, ToInstrumentByFunction[&F],
                       NoRaceCallsitesByFunction[&F]);
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
  Type *LoadPropertyTy = CsiLoadStoreProperty::getType(C);
  Type *StorePropertyTy = CsiLoadStoreProperty::getType(C);
  Type *AllocFnPropertyTy = CsiAllocFnProperty::getType(C);
  Type *FreePropertyTy = CsiFreeProperty::getType(C);
  Type *RetType = IRB.getVoidTy();
  Type *AddrType = IRB.getInt8PtrTy();
  Type *NumBytesType = IRB.getInt32Ty();
  Type *LargeNumBytesType = IntptrTy;
  Type *IDType = IRB.getInt64Ty();

  CsanFuncEntry = M.getOrInsertFunction("__csan_func_entry", RetType,
                                        /* func_id */ IDType,
                                        /* stack_ptr */ AddrType,
                                        FuncPropertyTy);
  CsanFuncExit = M.getOrInsertFunction("__csan_func_exit", RetType,
                                       /* func_exit_id */ IDType,
                                       /* func_id */ IDType,
                                       FuncExitPropertyTy);

  CsanRead = M.getOrInsertFunction("__csan_load", RetType, IDType,
                                   AddrType, NumBytesType, LoadPropertyTy);
  CsanWrite = M.getOrInsertFunction("__csan_store", RetType, IDType,
                                    AddrType, NumBytesType, StorePropertyTy);
  CsanLargeRead = M.getOrInsertFunction("__csan_large_load", RetType, IDType,
                                        AddrType, LargeNumBytesType,
                                        LoadPropertyTy);
  CsanLargeWrite = M.getOrInsertFunction("__csan_large_store", RetType, IDType,
                                         AddrType, LargeNumBytesType,
                                         StorePropertyTy);
  // CsanWrite = M.getOrInsertFunction("__csan_atomic_exchange", RetType,
  //                                   IDType, AddrType, NumBytesType,
  //                                   StorePropertyTy);

  CsanDetach = M.getOrInsertFunction("__csan_detach", RetType,
                                     /* detach_id */ IDType);
  CsanTaskEntry = M.getOrInsertFunction("__csan_task", RetType,
                                        /* task_id */ IDType,
                                        /* detach_id */ IDType,
                                        /* stack_ptr */ AddrType);
  CsanTaskExit = M.getOrInsertFunction("__csan_task_exit", RetType,
                                       /* task_exit_id */ IDType,
                                       /* task_id */ IDType,
                                       /* detach_id */ IDType);
  CsanDetachContinue = M.getOrInsertFunction("__csan_detach_continue", RetType,
                                             /* detach_continue_id */ IDType,
                                             /* detach_id */ IDType);
  CsanSync = M.getOrInsertFunction("__csan_sync", RetType, IDType);
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
  // CsanBeforeFree = M.getOrInsertFunction("__csan_before_free", RetType, IDType,
  //                                        AddrType);
  CsanAfterFree = M.getOrInsertFunction("__csan_after_free", RetType, IDType,
                                        AddrType,
                                        /* property */ FreePropertyTy);

  CsanDisableChecking = M.getOrInsertFunction("__cilksan_disable_checking",
                                              RetType);
  CsanEnableChecking = M.getOrInsertFunction("__cilksan_enable_checking",
                                             RetType);
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
    else if (isAllocationFn(Pred->getTerminator(), TLI))
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
    GA.Type = GeneralAccess::WRITE;
    GA.Loc = MemoryLocation::getForDest(MI);
    AccI.push_back(GA);
    if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(I)) {
      GA.Type = GeneralAccess::READ;
      GA.Loc = MemoryLocation::getForSource(MTI);
      AccI.push_back(GA);
    }
    return true;
  }

  // Handle more standard memory operations.
  if (Optional<MemoryLocation> MLoc = MemoryLocation::getOrNone(I)) {
    GA.Loc = *MLoc;
    GA.Type = isa<LoadInst>(I) ? GeneralAccess::READ : GeneralAccess::WRITE;
    AccI.push_back(GA);
    return true;
  }

  // Handle arbitrary call sites by examining pointee arguments.
  ImmutableCallSite CS(I);
  if (CS) {
    for (auto AI = CS.arg_begin(), AE = CS.arg_end(); AI != AE; ++AI) {
      const Value *Arg = *AI;
      if (!Arg->getType()->isPtrOrPtrVectorTy())
        continue;

      unsigned ArgIdx = std::distance(CS.arg_begin(), AI);
      MemoryLocation Loc = MemoryLocation::getForArgument(CS, ArgIdx, *TLI);
      if (AA->pointsToConstantMemory(Loc))
        continue;

      GA.Loc = Loc;
      auto MRI = AA->getArgModRefInfo(CS, ArgIdx);
      if (isModSet(MRI))
        GA.Type = GeneralAccess::WRITE;
      else if (isRefSet(MRI))
        GA.Type = GeneralAccess::READ;

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
// this spindle, ignoring back edges of loops..
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
  Spindle *DomSpindle = TI.getSpindleFor(
      DT.findNearestCommonDominator(Src->getParent(), Dst->getParent()));
  // Find the loop depth of that spindle.
  unsigned MaxLoopDepthToCheck = LI.getLoopDepth(DomSpindle->getEntry());
  assert(MinObjDepth <= MaxLoopDepthToCheck &&
         "Minimum loop depth of underlying object cannot be greater "
         "than maximum loop depth of dependence.");

  if (MaxLoopDepthToCheck == MinObjDepth) {
    // Dependence does not depend on looping.
    if (!MaxLoopDepthToCheck)
      // If there's no loop to worry about, then the existence of the dependence
      // implies the potential for a race.
      return true;

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
    if (SkipRead && isa<MemTransferInst>(I) &&
        (GeneralAccess::READ == GA1.Type))
      continue;
    if (SkipWrite && isa<MemTransferInst>(I) &&
        (GeneralAccess::WRITE == GA1.Type))
      continue;

    for (GeneralAccess GA2 : AccMPI) {
      // Two parallel loads cannot race.
      if (GeneralAccess::READ == GA1.Type && GeneralAccess::READ == GA2.Type)
        continue;
      LLVM_DEBUG(dbgs() << "Checking addresses " << *GA1.Loc->Ptr << " vs " <<
                 *GA2.Loc->Ptr << "\n");
      if (DependenceMightRace(
              DI.depends(&GA1, &GA2, true), GA1.I, GA2.I,
              const_cast<Value *>(GA1.Loc->Ptr),
              const_cast<Value *>(GA2.Loc->Ptr), MPTasks, MPTasksInLoop,
              DT, TI, LI, DL))
        return getRaceType((GeneralAccess::WRITE == GA1.Type),
                           (GeneralAccess::WRITE == GA2.Type));
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
  for (const Value *BaseObj : BaseObjs)
    if (!isa<AllocaInst>(BaseObj) && !isAllocationFn(BaseObj, TLI)) {
      LLVM_DEBUG(dbgs() << "Non-local base object " << *BaseObj << "\n");
      return false;
    }

  return true;
}

/// Returns true if Addr can only refer to a locally allocated base object, that
/// is, an object created via an AllocaInst or an AllocationFn.
static bool LocalBaseObj(ImmutableCallSite &CS, const DataLayout &DL,
                         LoopInfo *LI, const TargetLibraryInfo *TLI) {
  // Check whether all pointer arguments point to local memory, and
  // ignore calls that only access local memory.
  for (auto CI = CS.arg_begin(), CE = CS.arg_end(); CI != CE; ++CI) {
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
static bool MightHaveDetachedUse(const Instruction *AI, const TaskInfo &TI) {
  // Get the task for this allocation.
  const Task *AllocTask = TI.getTaskFor(AI->getParent());
  assert(AllocTask && "Null task for instruction.");

  if (AllocTask->isSerial())
    // Alloc AI cannot be used in a subtask if its enclosing task is serial.
    return false;

  SmallVector<const Use *, 20> Worklist;
  SmallSet<const Use *, 20> Visited;

  // Add all uses of AI to the worklist.
  for (const Use &U : AI->uses()) {
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

  if (PointerMayBeCaptured(Addr, false, false))
    return true;

  // Check for detached uses of the underlying base objects.
  SmallVector<Value *, 1> BaseObjs;
  GetUnderlyingObjects(Addr, BaseObjs, DL, LI, 0);

  // If we could not determine the base objects, conservatively return true.
  if (BaseObjs.empty())
    return true;

  for (const Value *BaseObj : BaseObjs) {
    // Skip any null objects
    if (const Constant *C = dyn_cast<Constant>(BaseObj))
      if (C->isNullValue())
        continue;

    // If the base object is not an instruction, conservatively return true.
    if (!isa<Instruction>(BaseObj))
      return true;

    // If the base object might have a detached use, return true.
    if (MightHaveDetachedUse(cast<Instruction>(BaseObj), TI))
      return true;
  }

  return false;
}

/// Returns true if any address referenced by the callsite could race due to
/// pointer capture.
static bool PossibleRaceByCapture(ImmutableCallSite &CS, const DataLayout &DL,
                                  const TaskInfo &TI, LoopInfo *LI) {
  // Check whether all pointer arguments point to local memory, and
  // ignore calls that only access local memory.
  for (auto CI = CS.arg_begin(), CE = CS.arg_end(); CI != CE; ++CI) {
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
    ImmutableCallSite CS(I);
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
    const Function *Called = CS.getCalledFunction();
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

bool CilkSanitizerImpl::prepareToInstrumentFunction(Function &F) {
  if (F.empty() || shouldNotInstrumentFunction(F))
    return false;

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
          if (CallInst *CI = dyn_cast<CallInst>(&Inst))
            maybeMarkSanitizerLibraryCallNoBuiltin(CI, TLI);

          // Record this function call as either an allocation function, a call
          // to free (or delete), a memory intrinsic, or an ordinary real
          // function call.
          if (isAllocationFn(&Inst, TLI))
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

  // While the maybe-parallel-task information is handy, use it to map each
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
  if (F.empty() || shouldNotInstrumentFunction(F))
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
      if (isa<AllocaInst>(BaseObj) || isAllocationFn(BaseObj, TLI))
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

  uint64_t LocalId = getLocalFunctionID(F);

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
  }

  return Res;
}

bool CilkSanitizerImpl::instrumentLoadOrStore(Instruction *I) {
  IRBuilder<> IRB(I);
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

bool CilkSanitizerImpl::instrumentAtomic(Instruction *I) {
  IRBuilder<> IRB(I);
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

bool CilkSanitizerImpl::instrumentCallsite(Instruction *I) {
  if (callsPlaceholderFunction(*I))
    return false;

  bool IsInvoke = isa<InvokeInst>(I);

  Function *Called = NULL;
  if (CallInst *CI = dyn_cast<CallInst>(I))
    Called = CI->getCalledFunction();
  else if (InvokeInst *II = dyn_cast<InvokeInst>(I))
    Called = II->getCalledFunction();

  IRBuilder<> IRB(I);
  uint64_t LocalId = CallsiteFED.add(*I);
  Value *DefaultID = getDefaultID(IRB);
  Value *CallsiteId = CallsiteFED.localToGlobalId(LocalId, IRB);
  Value *FuncId = NULL;
  GlobalVariable *FuncIdGV = NULL;
  if (Called) {
    Module *M = I->getParent()->getParent()->getParent();
    std::string GVName =
      CsiFuncIdVariablePrefix + Called->getName().str();
    FuncIdGV = dyn_cast<GlobalVariable>(M->getOrInsertGlobal(GVName,
                                                             IRB.getInt64Ty()));
    assert(FuncIdGV);
    FuncIdGV->setConstant(false);
    FuncIdGV->setLinkage(GlobalValue::WeakAnyLinkage);
    FuncIdGV->setInitializer(IRB.getInt64(CsiCallsiteUnknownTargetId));
    FuncId = IRB.CreateLoad(FuncIdGV);
  } else {
    // Unknown targets (i.e. indirect calls) are always unknown.
    FuncId = IRB.getInt64(CsiCallsiteUnknownTargetId);
  }
  assert(FuncId != NULL);
  CsiCallProperty Prop;
  Value *DefaultPropVal = Prop.getValue(IRB);
  Prop.setIsIndirect(!Called);
  Value *PropVal = Prop.getValue(IRB);
  insertHookCall(I, CsiBeforeCallsite, {CallsiteId, FuncId, PropVal});

  BasicBlock::iterator Iter(I);
  if (IsInvoke) {
    // There are two "after" positions for invokes: the normal block and the
    // exception block.
    InvokeInst *II = cast<InvokeInst>(I);
    insertHookCallInSuccessorBB(
        II->getNormalDest(), II->getParent(), CsiAfterCallsite,
        {CallsiteId, FuncId, PropVal}, {DefaultID, DefaultID, DefaultPropVal});
    insertHookCallInSuccessorBB(
        II->getUnwindDest(), II->getParent(), CsiAfterCallsite,
        {CallsiteId, FuncId, PropVal}, {DefaultID, DefaultID, DefaultPropVal});
  } else {
    // Simple call instruction; there is only one "after" position.
    Iter++;
    IRB.SetInsertPoint(&*Iter);
    PropVal = Prop.getValue(IRB);
    insertHookCall(&*Iter, CsiAfterCallsite, {CallsiteId, FuncId, PropVal});
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
    Value *StackSave = IRB.CreateCall(
        Intrinsic::getDeclaration(&M, Intrinsic::stacksave));
    Instruction *Call = IRB.CreateCall(CsanTaskEntry,
                                       {TaskID, DetachID, StackSave});
    // Value *FrameAddr = IRB.CreateCall(
    //     Intrinsic::getDeclaration(&M, Intrinsic::frameaddress),
    //     {IRB.getInt32(0)});
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
    SizeVal = IRB.CreateMul(SizeVal, AI->getArraySize());

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

  return CilkSanitizerImpl(M, CG, GetDomTree, GetTaskInfo, GetLoopInfo,
                           GetDepInfo, TLI).run();
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
  auto *TLI = &AM.getResult<TargetLibraryAnalysis>(M);

  if (!CilkSanitizerImpl(M, &CG, GetDT, GetTI, GetLI, GetDI, TLI)
      .run())
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
