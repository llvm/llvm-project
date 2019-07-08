//===-- ComprehensiveStaticInstrumentation.cpp - CSI compiler pass --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of CSI, a framework that provides comprehensive static
// instrumentation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/ComprehensiveStaticInstrumentation.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Instrumentation/CSI.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "csi"

static cl::opt<bool>
    ClInstrumentFuncEntryExit("csi-instrument-func-entry-exit", cl::init(true),
                              cl::desc("Instrument function entry and exit"),
                              cl::Hidden);
static cl::opt<bool>
    ClInstrumentBasicBlocks("csi-instrument-basic-blocks", cl::init(true),
                            cl::desc("Instrument basic blocks"), cl::Hidden);
static cl::opt<bool>
    ClInstrumentMemoryAccesses("csi-instrument-memory-accesses", cl::init(true),
                               cl::desc("Instrument memory accesses"),
                               cl::Hidden);
static cl::opt<bool> ClInstrumentCalls("csi-instrument-function-calls",
                                       cl::init(true),
                                       cl::desc("Instrument function calls"),
                                       cl::Hidden);
static cl::opt<bool> ClInstrumentAtomics("csi-instrument-atomics",
                                         cl::init(true),
                                         cl::desc("Instrument atomics"),
                                         cl::Hidden);
static cl::opt<bool> ClInstrumentMemIntrinsics(
    "csi-instrument-memintrinsics", cl::init(true),
    cl::desc("Instrument memintrinsics (memset/memcpy/memmove)"), cl::Hidden);
static cl::opt<bool> ClInstrumentTapir("csi-instrument-tapir", cl::init(true),
                                       cl::desc("Instrument tapir constructs"),
                                       cl::Hidden);
static cl::opt<bool> ClInstrumentAllocas("csi-instrument-alloca",
                                         cl::init(true),
                                         cl::desc("Instrument allocas"),
                                         cl::Hidden);
static cl::opt<bool>
    ClInstrumentAllocFns("csi-instrument-allocfn", cl::init(true),
                         cl::desc("Instrument allocation functions"),
                         cl::Hidden);

static cl::opt<bool> ClInterpose("csi-interpose", cl::init(true),
                                 cl::desc("Enable function interpositioning"),
                                 cl::Hidden);

static cl::opt<std::string> ClToolBitcode(
    "csi-tool-bitcode", cl::init(""),
    cl::desc("Path to the tool bitcode file for compile-time instrumentation"),
    cl::Hidden);

static cl::opt<std::string>
    ClRuntimeBitcode("csi-runtime-bitcode", cl::init(""),
                     cl::desc("Path to the CSI runtime bitcode file for "
                              "optimized compile-time instrumentation"),
                     cl::Hidden);

static cl::opt<std::string> ClToolLibrary(
    "csi-tool-library", cl::init(""),
    cl::desc("Path to the tool library file for compile-time instrumentation"),
    cl::Hidden);

static cl::opt<std::string> ClConfigurationFilename(
    "csi-config-filename", cl::init(""),
    cl::desc("Path to the configuration file for surgical instrumentation"),
    cl::Hidden);

static cl::opt<InstrumentationConfigMode> ClConfigurationMode(
    "csi-config-mode", cl::init(InstrumentationConfigMode::WHITELIST),
    cl::values(clEnumValN(InstrumentationConfigMode::WHITELIST, "whitelist",
                          "Use configuration file as a whitelist"),
               clEnumValN(InstrumentationConfigMode::BLACKLIST, "blacklist",
                          "Use configuration file as a blacklist")),
    cl::desc("Specifies how to interpret the configuration file"), cl::Hidden);

static cl::opt<bool>
    AssumeNoExceptions(
        "csi-assume-no-exceptions", cl::init(false), cl::Hidden,
        cl::desc("Assume that ordinary calls cannot throw exceptions."));

static size_t numPassRuns = 0;
bool IsFirstRun() { return numPassRuns == 0; }

namespace {

static CSIOptions OverrideFromCL(CSIOptions Options) {
  Options.InstrumentFuncEntryExit = ClInstrumentFuncEntryExit;
  Options.InstrumentBasicBlocks = ClInstrumentBasicBlocks;
  Options.InstrumentMemoryAccesses = ClInstrumentMemoryAccesses;
  Options.InstrumentCalls = ClInstrumentCalls;
  Options.InstrumentAtomics = ClInstrumentAtomics;
  Options.InstrumentMemIntrinsics = ClInstrumentMemIntrinsics;
  Options.InstrumentTapir = ClInstrumentTapir;
  Options.InstrumentAllocas = ClInstrumentAllocas;
  Options.InstrumentAllocFns = ClInstrumentAllocFns;
  Options.CallsMayThrow = !AssumeNoExceptions;
  return Options;
}

/// The Comprehensive Static Instrumentation pass.
/// Inserts calls to user-defined hooks at predefined points in the IR.
struct ComprehensiveStaticInstrumentationLegacyPass : public ModulePass {
  static char ID; // Pass identification, replacement for typeid.

  ComprehensiveStaticInstrumentationLegacyPass(
      const CSIOptions &Options = OverrideFromCL(CSIOptions()))
      : ModulePass(ID), Options(Options) {
    initializeComprehensiveStaticInstrumentationLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }
  StringRef getPassName() const override {
    return "ComprehensiveStaticInstrumentation";
  }
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  CSIOptions Options;
}; // struct ComprehensiveStaticInstrumentation
} // anonymous namespace

char ComprehensiveStaticInstrumentationLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(ComprehensiveStaticInstrumentationLegacyPass, "csi",
                      "ComprehensiveStaticInstrumentation pass", false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TaskInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(ComprehensiveStaticInstrumentationLegacyPass, "csi",
                    "ComprehensiveStaticInstrumentation pass", false, false)

ModulePass *llvm::createComprehensiveStaticInstrumentationLegacyPass() {
  return new ComprehensiveStaticInstrumentationLegacyPass();
}
ModulePass *llvm::createComprehensiveStaticInstrumentationLegacyPass(
    const CSIOptions &Options) {
  return new ComprehensiveStaticInstrumentationLegacyPass(Options);
}

/// Return the first DILocation in the given basic block, or nullptr
/// if none exists.
static const DILocation *getFirstDebugLoc(const BasicBlock &BB) {
  for (const Instruction &Inst : BB)
    if (const DILocation *Loc = Inst.getDebugLoc())
      return Loc;

  return nullptr;
}

/// Set DebugLoc on the call instruction to a CSI hook, based on the
/// debug information of the instrumented instruction.
static void setInstrumentationDebugLoc(Instruction *Instrumented,
                                       Instruction *Call) {
  DISubprogram *Subprog = Instrumented->getFunction()->getSubprogram();
  if (Subprog) {
    if (Instrumented->getDebugLoc()) {
      Call->setDebugLoc(Instrumented->getDebugLoc());
    } else {
      LLVMContext &C = Instrumented->getContext();
      Call->setDebugLoc(DILocation::get(C, 0, 0, Subprog));
    }
  }
}

/// Set DebugLoc on the call instruction to a CSI hook, based on the
/// debug information of the instrumented instruction.
static void setInstrumentationDebugLoc(BasicBlock &Instrumented,
                                       Instruction *Call) {
  DISubprogram *Subprog = Instrumented.getParent()->getSubprogram();
  if (Subprog) {
    if (const DILocation *FirstDebugLoc = getFirstDebugLoc(Instrumented))
      Call->setDebugLoc(FirstDebugLoc);
    else {
      LLVMContext &C = Instrumented.getContext();
      Call->setDebugLoc(DILocation::get(C, 0, 0, Subprog));
    }
  }
}

/// Set DebugLoc on the call instruction to a CSI hook, based on the
/// debug information of the instrumented instruction.
static void setInstrumentationDebugLoc(Function &Instrumented,
                                       Instruction *Call) {
  DISubprogram *Subprog = Instrumented.getSubprogram();
  if (Subprog) {
    LLVMContext &C = Instrumented.getParent()->getContext();
    Call->setDebugLoc(DILocation::get(C, 0, 0, Subprog));
  }
}

bool CSIImpl::callsPlaceholderFunction(const Instruction &I) {
  if (isa<DbgInfoIntrinsic>(I))
    return true;

  if (isDetachedRethrow(&I))
    return true;

  if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I))
    if (Intrinsic::syncregion_start == II->getIntrinsicID() ||
        Intrinsic::lifetime_start == II->getIntrinsicID() ||
        Intrinsic::lifetime_end == II->getIntrinsicID())
      return true;

  return false;
}

bool CSIImpl::run() {
  initializeCsi();

  for (Function &F : M)
    instrumentFunction(F);

  collectUnitFEDTables();
  collectUnitSizeTables();

  finalizeCsi();

  if (IsFirstRun() && Options.jitMode) {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(ClToolLibrary.c_str());
  }
  linkInToolFromBitcode(ClToolBitcode);
  linkInToolFromBitcode(ClRuntimeBitcode);

  return true; // We always insert the unit constructor.
}

Constant *ForensicTable::getObjectStrGV(Module &M, StringRef Str,
                                        const Twine GVName) {
  LLVMContext &C = M.getContext();
  IntegerType *Int32Ty = IntegerType::get(C, 32);
  Constant *Zero = ConstantInt::get(Int32Ty, 0);
  Value *GepArgs[] = {Zero, Zero};
  if (Str.empty())
    return ConstantPointerNull::get(
        PointerType::get(IntegerType::get(C, 8), 0));

  Constant *NameStrConstant = ConstantDataArray::getString(C, Str);
  GlobalVariable *GV = M.getGlobalVariable((GVName + Str).str(), true);
  if (GV == NULL) {
    GV = new GlobalVariable(M, NameStrConstant->getType(), true,
                            GlobalValue::PrivateLinkage, NameStrConstant,
                            GVName + Str, nullptr,
                            GlobalVariable::NotThreadLocal, 0);
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  }
  assert(GV);
  return ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
}

ForensicTable::ForensicTable(Module &M, StringRef BaseIdName) {
  LLVMContext &C = M.getContext();
  IntegerType *Int64Ty = IntegerType::get(C, 64);
  IdCounter = 0;

  BaseId = M.getGlobalVariable(BaseIdName, true);
  if (NULL == BaseId)
    BaseId = new GlobalVariable(M, Int64Ty, false, GlobalValue::InternalLinkage,
                                ConstantInt::get(Int64Ty, 0), BaseIdName);
  assert(BaseId);
}

uint64_t ForensicTable::getId(const Value *V) {
  if (!ValueToLocalIdMap.count(V))
    ValueToLocalIdMap[V] = IdCounter++;
  assert(ValueToLocalIdMap.count(V) && "Value not in ID map.");
  return ValueToLocalIdMap[V];
}

Value *ForensicTable::localToGlobalId(uint64_t LocalId,
                                      IRBuilder<> &IRB) const {
  assert(BaseId);
  LLVMContext &C = IRB.getContext();
  LoadInst *Base = IRB.CreateLoad(BaseId);
  MDNode *MD = llvm::MDNode::get(C, None);
  Base->setMetadata(LLVMContext::MD_invariant_load, MD);
  Value *Offset = IRB.getInt64(LocalId);
  return IRB.CreateAdd(Base, Offset);
}

uint64_t SizeTable::add(const BasicBlock &BB) {
  uint64_t ID = getId(&BB);
  // Count the LLVM IR instructions
  int32_t NonEmptyIRSize = 0;
  for (const Instruction &I : BB) {
    if (isa<PHINode>(I))
      continue;
    if (CSIImpl::callsPlaceholderFunction(I))
      continue;
    NonEmptyIRSize++;
  }
  add(ID, BB.size(), NonEmptyIRSize);
  return ID;
}

PointerType *SizeTable::getPointerType(LLVMContext &C) {
  return PointerType::get(getSizeStructType(C), 0);
}

StructType *SizeTable::getSizeStructType(LLVMContext &C) {
  return StructType::get(
      /* FullIRSize */ IntegerType::get(C, 32),
      /* NonEmptyIRSize */ IntegerType::get(C, 32));
}

void SizeTable::add(uint64_t ID, int32_t FullIRSize, int32_t NonEmptyIRSize) {
  assert(NonEmptyIRSize <= FullIRSize && "Broken basic block IR sizes");
  assert(LocalIdToSizeMap.find(ID) == LocalIdToSizeMap.end() &&
         "ID already exists in FED table.");
  LocalIdToSizeMap[ID] = {FullIRSize, NonEmptyIRSize};
}

Constant *SizeTable::insertIntoModule(Module &M) const {
  LLVMContext &C = M.getContext();
  StructType *TableType = getSizeStructType(C);
  IntegerType *Int32Ty = IntegerType::get(C, 32);
  Constant *Zero = ConstantInt::get(Int32Ty, 0);
  Value *GepArgs[] = {Zero, Zero};
  SmallVector<Constant *, 1> TableEntries;

  for (uint64_t LocalID = 0; LocalID < IdCounter; ++LocalID) {
    const SizeInformation &E = LocalIdToSizeMap.find(LocalID)->second;
    Constant *FullIRSize = ConstantInt::get(Int32Ty, E.FullIRSize);
    Constant *NonEmptyIRSize = ConstantInt::get(Int32Ty, E.NonEmptyIRSize);
    // The order of arguments to ConstantStruct::get() must match the
    // sizeinfo_t type in csi.h.
    TableEntries.push_back(
        ConstantStruct::get(TableType, FullIRSize, NonEmptyIRSize));
  }

  ArrayType *TableArrayType = ArrayType::get(TableType, TableEntries.size());
  Constant *Table = ConstantArray::get(TableArrayType, TableEntries);
  GlobalVariable *GV =
      new GlobalVariable(M, TableArrayType, false, GlobalValue::InternalLinkage,
                         Table, CsiUnitSizeTableName);
  return ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
}

uint64_t FrontEndDataTable::add(const Function &F) {
  uint64_t ID = getId(&F);
  if (F.getSubprogram())
    add(ID, F.getSubprogram());
  else
    add(ID, -1, -1, F.getName(), F.getParent()->getName(),
        F.getName());
  return ID;
}

uint64_t FrontEndDataTable::add(const BasicBlock &BB) {
  uint64_t ID = getId(&BB);
  add(ID, getFirstDebugLoc(BB));
  return ID;
}

uint64_t FrontEndDataTable::add(const Instruction &I,
                                const StringRef &RealName) {
  uint64_t ID = getId(&I);
  if (auto DL = I.getDebugLoc())
    add(ID, DL, RealName);
  else {
    add(ID, -1, -1, I.getFunction()->getName(), I.getModule()->getName(),
        I.getName());
  }
  return ID;
}

PointerType *FrontEndDataTable::getPointerType(LLVMContext &C) {
  return PointerType::get(getSourceLocStructType(C), 0);
}

StructType *FrontEndDataTable::getSourceLocStructType(LLVMContext &C) {
  return StructType::get(
      /* Name */ PointerType::get(IntegerType::get(C, 8), 0),
      /* Line */ IntegerType::get(C, 32),
      /* Column */ IntegerType::get(C, 32),
      /* File */ PointerType::get(IntegerType::get(C, 8), 0));
}

void FrontEndDataTable::add(uint64_t ID, const DILocation *Loc,
                            const StringRef &RealName) {
  if (Loc) {
    // TODO: Add location information for inlining
    const DISubprogram *Subprog = Loc->getScope()->getSubprogram();
    add(ID, (int32_t)Loc->getLine(), (int32_t)Loc->getColumn(),
        Loc->getFilename(), Loc->getDirectory(),
        RealName == "" ? Subprog->getName() : RealName);
  } else
    add(ID);
}

void FrontEndDataTable::add(uint64_t ID, const DISubprogram *Subprog) {
  if (Subprog)
    add(ID, (int32_t)Subprog->getLine(), -1, Subprog->getFilename(),
        Subprog->getDirectory(), Subprog->getName());
  else
    add(ID);
}

void FrontEndDataTable::add(uint64_t ID, int32_t Line, int32_t Column,
                            StringRef Filename, StringRef Directory,
                            StringRef Name) {
  // TODO: This assert is too strong for unwind basic blocks' FED.
  /*assert(LocalIdToSourceLocationMap.find(ID) ==
             LocalIdToSourceLocationMap.end() &&
         "Id already exists in FED table."); */
  LocalIdToSourceLocationMap[ID] = {Name, Line, Column, Filename, Directory};
}

// The order of arguments to ConstantStruct::get() must match the source_loc_t
// type in csi.h.
static void addFEDTableEntries(SmallVectorImpl<Constant *> &FEDEntries,
                               StructType *FedType, Constant *Name,
                               Constant *Line, Constant *Column,
                               Constant *File) {
  FEDEntries.push_back(ConstantStruct::get(FedType, Name, Line, Column, File));
}

Constant *FrontEndDataTable::insertIntoModule(Module &M) const {
  LLVMContext &C = M.getContext();
  StructType *FedType = getSourceLocStructType(C);
  IntegerType *Int32Ty = IntegerType::get(C, 32);
  Constant *Zero = ConstantInt::get(Int32Ty, 0);
  Value *GepArgs[] = {Zero, Zero};
  SmallVector<Constant *, 11> FEDEntries;

  for (uint64_t LocalID = 0; LocalID < IdCounter; ++LocalID) {
    const SourceLocation &E = LocalIdToSourceLocationMap.find(LocalID)->second;
    Constant *Line = ConstantInt::get(Int32Ty, E.Line);
    Constant *Column = ConstantInt::get(Int32Ty, E.Column);
    Constant *File;
    {
      std::string Filename = E.Filename.str();
      if (!E.Directory.empty())
        Filename = E.Directory.str() + "/" + Filename;
      File = getObjectStrGV(M, Filename, "__csi_unit_filename_");
    }
    Constant *Name = getObjectStrGV(M, E.Name, "__csi_unit_function_name_");
    addFEDTableEntries(FEDEntries, FedType, Name, Line, Column, File);
  }

  ArrayType *FedArrayType = ArrayType::get(FedType, FEDEntries.size());
  Constant *Table = ConstantArray::get(FedArrayType, FEDEntries);
  GlobalVariable *GV =
      new GlobalVariable(M, FedArrayType, false, GlobalValue::InternalLinkage,
                         Table, CsiUnitFedTableName + BaseId->getName());
  return ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
}

/// Function entry and exit hook initialization
void CSIImpl::initializeFuncHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  // Initialize function entry hook
  Type *FuncPropertyTy = CsiFuncProperty::getType(C);
  CsiFuncEntry = M.getOrInsertFunction("__csi_func_entry", IRB.getVoidTy(),
                                       IRB.getInt64Ty(), FuncPropertyTy);
  // Initialize function exit hook
  Type *FuncExitPropertyTy = CsiFuncExitProperty::getType(C);
  CsiFuncExit = M.getOrInsertFunction("__csi_func_exit", IRB.getVoidTy(),
                                      IRB.getInt64Ty(), IRB.getInt64Ty(),
                                      FuncExitPropertyTy);
}

/// Basic-block hook initialization
void CSIImpl::initializeBasicBlockHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *PropertyTy = CsiBBProperty::getType(C);
  CsiBBEntry = M.getOrInsertFunction("__csi_bb_entry", IRB.getVoidTy(),
                                     IRB.getInt64Ty(), PropertyTy);
  CsiBBExit = M.getOrInsertFunction("__csi_bb_exit", IRB.getVoidTy(),
                                    IRB.getInt64Ty(), PropertyTy);
}

// Call-site hook initialization
void CSIImpl::initializeCallsiteHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *PropertyTy = CsiCallProperty::getType(C);
  CsiBeforeCallsite = M.getOrInsertFunction("__csi_before_call",
                                            IRB.getVoidTy(), IRB.getInt64Ty(),
                                            IRB.getInt64Ty(), PropertyTy);
  CsiAfterCallsite = M.getOrInsertFunction("__csi_after_call", IRB.getVoidTy(),
                                           IRB.getInt64Ty(), IRB.getInt64Ty(),
                                           PropertyTy);
}

// Alloca (local variable) hook initialization
void CSIImpl::initializeAllocaHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *IDType = IRB.getInt64Ty();
  Type *AddrType = IRB.getInt8PtrTy();
  Type *PropType = CsiAllocaProperty::getType(C);

  CsiBeforeAlloca = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_before_alloca", IRB.getVoidTy(), IDType, IntptrTy, PropType));
  CsiAfterAlloca = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_alloca", IRB.getVoidTy(), IDType,
                            AddrType, IntptrTy, PropType));
}

// Non-local-variable allocation/free hook initialization
void CSIImpl::initializeAllocFnHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *RetType = IRB.getVoidTy();
  Type *IDType = IRB.getInt64Ty();
  Type *AddrType = IRB.getInt8PtrTy();
  Type *LargeNumBytesType = IntptrTy;
  Type *AllocFnPropType = CsiAllocFnProperty::getType(C);
  Type *FreePropType = CsiFreeProperty::getType(C);

  CsiBeforeAllocFn = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_before_allocfn", RetType, IDType, LargeNumBytesType,
      LargeNumBytesType, LargeNumBytesType, AddrType, AllocFnPropType));
  CsiAfterAllocFn = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_allocfn", RetType, IDType,
                            /* new ptr */ AddrType,
                            /* size */ LargeNumBytesType,
                            /* num elements */ LargeNumBytesType,
                            /* alignment */ LargeNumBytesType,
                            /* old ptr */ AddrType,
                            /* property */ AllocFnPropType));

  CsiBeforeFree = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_before_free", RetType, IDType, AddrType, FreePropType));
  CsiAfterFree = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_after_free", RetType, IDType, AddrType, FreePropType));
}

// Load and store hook initialization
void CSIImpl::initializeLoadStoreHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *LoadPropertyTy = CsiLoadStoreProperty::getType(C);
  Type *StorePropertyTy = CsiLoadStoreProperty::getType(C);
  Type *RetType = IRB.getVoidTy();
  Type *AddrType = IRB.getInt8PtrTy();
  Type *NumBytesType = IRB.getInt32Ty();

  CsiBeforeRead = M.getOrInsertFunction("__csi_before_load", RetType,
                                        IRB.getInt64Ty(), AddrType,
                                        NumBytesType, LoadPropertyTy);
  CsiAfterRead = M.getOrInsertFunction("__csi_after_load", RetType,
                                       IRB.getInt64Ty(), AddrType, NumBytesType,
                                       LoadPropertyTy);

  CsiBeforeWrite = M.getOrInsertFunction("__csi_before_store", RetType,
                                         IRB.getInt64Ty(), AddrType,
                                         NumBytesType, StorePropertyTy);
  CsiAfterWrite = M.getOrInsertFunction("__csi_after_store", RetType,
                                        IRB.getInt64Ty(), AddrType,
                                        NumBytesType, StorePropertyTy);
}

// Initialization of hooks for LLVM memory intrinsics
void CSIImpl::initializeMemIntrinsicsHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);

  MemmoveFn = M.getOrInsertFunction("memmove", IRB.getInt8PtrTy(),
                                    IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                                    IntptrTy);
  MemcpyFn = M.getOrInsertFunction("memcpy", IRB.getInt8PtrTy(),
                                   IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                                   IntptrTy);
  MemsetFn = M.getOrInsertFunction("memset", IRB.getInt8PtrTy(),
                                   IRB.getInt8PtrTy(), IRB.getInt32Ty(),
                                   IntptrTy);
}

// Initialization of Tapir hooks
void CSIImpl::initializeTapirHooks() {
  LLVMContext &C = M.getContext();
  IRBuilder<> IRB(C);
  Type *IDType = IRB.getInt64Ty();
  Type *RetType = IRB.getVoidTy();

  CsiDetach = checkCsiInterfaceFunction(M.getOrInsertFunction(
      "__csi_detach", RetType,
      /* detach_id */ IDType, IntegerType::getInt32Ty(C)->getPointerTo()));
  CsiTaskEntry =
      checkCsiInterfaceFunction(M.getOrInsertFunction("__csi_task", RetType,
                                                      /* task_id */ IDType,
                                                      /* detach_id */ IDType));
  CsiTaskExit = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_task_exit", RetType,
                            /* task_exit_id */ IDType,
                            /* task_id */ IDType,
                            /* detach_id */ IDType));
  CsiDetachContinue = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_detach_continue", RetType,
                            /* detach_continue_id */ IDType,
                            /* detach_id */ IDType));
  CsiBeforeSync = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_before_sync", RetType, IDType,
                            IntegerType::getInt32Ty(C)->getPointerTo()));
  CsiAfterSync = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_after_sync", RetType, IDType,
                            IntegerType::getInt32Ty(C)->getPointerTo()));
}

// Prepare any calls in the CFG for instrumentation, e.g., by making sure any
// call that can throw is modeled with an invoke.
void CSIImpl::setupCalls(Function &F) {
  // We use the EscapeEnumerator's built-in functionality to promote calls to
  // invokes.
  EscapeEnumerator EE(F, "csi.cleanup", true);
  while (EE.Next())
    ;

  // TODO: Split each basic block immediately after each call, to ensure that
  // calls act like terminators?
}

static BasicBlock *SplitOffPreds(BasicBlock *BB,
                                 SmallVectorImpl<BasicBlock *> &Preds,
                                 DominatorTree *DT) {
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
static void setupBlock(BasicBlock *BB, const TargetLibraryInfo *TLI,
                       DominatorTree *DT) {
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
void CSIImpl::setupBlocks(Function &F, const TargetLibraryInfo *TLI,
                          DominatorTree *DT) {
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
    setupBlock(BB, TLI, DT);
}

int CSIImpl::getNumBytesAccessed(Value *Addr, const DataLayout &DL) {
  Type *OrigPtrTy = Addr->getType();
  Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();
  assert(OrigTy->isSized());
  uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  if (TypeSize % 8 != 0)
    return -1;
  return TypeSize / 8;
}

void CSIImpl::addLoadStoreInstrumentation(Instruction *I, Function *BeforeFn,
                                          Function *AfterFn, Value *CsiId,
                                          Type *AddrType, Value *Addr,
                                          int NumBytes,
                                          CsiLoadStoreProperty &Prop) {
  IRBuilder<> IRB(I);
  Value *PropVal = Prop.getValue(IRB);
  insertHookCall(I, BeforeFn,
                 {CsiId, IRB.CreatePointerCast(Addr, AddrType),
                  IRB.getInt32(NumBytes), PropVal});

  BasicBlock::iterator Iter = ++I->getIterator();
  IRB.SetInsertPoint(&*Iter);
  insertHookCall(&*Iter, AfterFn,
                 {CsiId, IRB.CreatePointerCast(Addr, AddrType),
                  IRB.getInt32(NumBytes), PropVal});
}

void CSIImpl::instrumentLoadOrStore(Instruction *I, CsiLoadStoreProperty &Prop,
                                    const DataLayout &DL) {
  IRBuilder<> IRB(I);
  bool IsWrite = isa<StoreInst>(I);
  Value *Addr = IsWrite ? cast<StoreInst>(I)->getPointerOperand()
                        : cast<LoadInst>(I)->getPointerOperand();
  int NumBytes = getNumBytesAccessed(Addr, DL);
  Type *AddrType = IRB.getInt8PtrTy();

  if (NumBytes == -1)
    return; // size that we don't recognize

  if (IsWrite) {
    uint64_t LocalId = StoreFED.add(*I);
    Value *CsiId = StoreFED.localToGlobalId(LocalId, IRB);
    addLoadStoreInstrumentation(I, CsiBeforeWrite, CsiAfterWrite, CsiId,
                                AddrType, Addr, NumBytes, Prop);
  } else { // is read
    uint64_t LocalId = LoadFED.add(*I);
    Value *CsiId = LoadFED.localToGlobalId(LocalId, IRB);
    addLoadStoreInstrumentation(I, CsiBeforeRead, CsiAfterRead, CsiId, AddrType,
                                Addr, NumBytes, Prop);
  }
}

void CSIImpl::instrumentAtomic(Instruction *I, const DataLayout &DL) {
  // For now, print a message that this code contains atomics.
  dbgs()
      << "WARNING: Uninstrumented atomic operations in program-under-test!\n";
}

// TODO: This code for instrumenting memory intrinsics was borrowed
// from TSan.  Different tools might have better ways to handle these
// function calls.  Replace this logic with a more flexible solution,
// possibly one based on interpositioning.
//
// If a memset intrinsic gets inlined by the code gen, we will miss it.
// So, we either need to ensure the intrinsic is not inlined, or instrument it.
// We do not instrument memset/memmove/memcpy intrinsics (too complicated),
// instead we simply replace them with regular function calls, which are then
// intercepted by the run-time.
// Since our pass runs after everyone else, the calls should not be
// replaced back with intrinsics. If that becomes wrong at some point,
// we will need to call e.g. __csi_memset to avoid the intrinsics.
bool CSIImpl::instrumentMemIntrinsic(Instruction *I) {
  IRBuilder<> IRB(I);
  if (MemSetInst *M = dyn_cast<MemSetInst>(I)) {
    Instruction *Call = IRB.CreateCall(
        MemsetFn,
        {IRB.CreatePointerCast(M->getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(M->getArgOperand(1), IRB.getInt32Ty(), false),
         IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
    setInstrumentationDebugLoc(I, Call);
    I->eraseFromParent();
    return true;
  } else if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
    Instruction *Call = IRB.CreateCall(
        isa<MemCpyInst>(M) ? MemcpyFn : MemmoveFn,
        {IRB.CreatePointerCast(M->getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreatePointerCast(M->getArgOperand(1), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
    setInstrumentationDebugLoc(I, Call);
    I->eraseFromParent();
    return true;
  }
  return false;
}

void CSIImpl::instrumentBasicBlock(BasicBlock &BB) {
  IRBuilder<> IRB(&*BB.getFirstInsertionPt());
  uint64_t LocalId = BasicBlockFED.add(BB);
  uint64_t BBSizeId = BBSize.add(BB);
  assert(LocalId == BBSizeId &&
         "BB recieved different ID's in FED and sizeinfo tables.");
  Value *CsiId = BasicBlockFED.localToGlobalId(LocalId, IRB);
  CsiBBProperty Prop;
  Prop.setIsLandingPad(BB.isLandingPad());
  Prop.setIsEHPad(BB.isEHPad());
  Instruction *TI = BB.getTerminator();
  Value *PropVal = Prop.getValue(IRB);
  insertHookCall(&*IRB.GetInsertPoint(), CsiBBEntry, {CsiId, PropVal});
  IRB.SetInsertPoint(TI);
  insertHookCall(TI, CsiBBExit, {CsiId, PropVal});
}

void CSIImpl::instrumentCallsite(Instruction *I, DominatorTree *DT) {
  if (callsPlaceholderFunction(*I))
    return;

  bool IsInvoke = isa<InvokeInst>(I);
  Function *Called = nullptr;
  if (CallInst *CI = dyn_cast<CallInst>(I))
    Called = CI->getCalledFunction();
  else if (InvokeInst *II = dyn_cast<InvokeInst>(I))
    Called = II->getCalledFunction();

  bool shouldInstrumentBefore = true;
  bool shouldInstrumentAfter = true;

  // Does this call require instrumentation before or after?
  if (Called) {
    shouldInstrumentBefore = Config->DoesFunctionRequireInstrumentationForPoint(
        Called->getName(), InstrumentationPoint::INSTR_BEFORE_CALL);
    shouldInstrumentAfter = Config->DoesFunctionRequireInstrumentationForPoint(
        Called->getName(), InstrumentationPoint::INSTR_AFTER_CALL);
  }

  if (!shouldInstrumentAfter && !shouldInstrumentBefore)
    return;

  IRBuilder<> IRB(I);
  Value *DefaultID = getDefaultID(IRB);
  uint64_t LocalId = CallsiteFED.add(*I, Called ? Called->getName() : "");
  Value *CallsiteId = CallsiteFED.localToGlobalId(LocalId, IRB);
  Value *FuncId = nullptr;
  GlobalVariable *FuncIdGV = nullptr;
  if (Called) {
    std::string GVName = CsiFuncIdVariablePrefix + Called->getName().str();
    FuncIdGV = dyn_cast<GlobalVariable>(
        M.getOrInsertGlobal(GVName, IRB.getInt64Ty()));
    assert(FuncIdGV);
    FuncIdGV->setConstant(false);
    if (Options.jitMode && !Called->empty())
      FuncIdGV->setLinkage(Called->getLinkage());
    else
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
  if (shouldInstrumentBefore)
    insertHookCall(I, CsiBeforeCallsite, {CallsiteId, FuncId, PropVal});

  BasicBlock::iterator Iter(I);
  if (shouldInstrumentAfter) {
    if (IsInvoke) {
      // There are two "after" positions for invokes: the normal block and the
      // exception block.
      InvokeInst *II = cast<InvokeInst>(I);
      insertHookCallInSuccessorBB(II->getNormalDest(), II->getParent(),
                                  CsiAfterCallsite,
                                  {CallsiteId, FuncId, PropVal},
                                  {DefaultID, DefaultID, DefaultPropVal});
      insertHookCallInSuccessorBB(II->getUnwindDest(), II->getParent(),
                                  CsiAfterCallsite,
                                  {CallsiteId, FuncId, PropVal},
                                  {DefaultID, DefaultID, DefaultPropVal});
    } else {
      // Simple call instruction; there is only one "after" position.
      Iter++;
      IRB.SetInsertPoint(&*Iter);
      PropVal = Prop.getValue(IRB);
      insertHookCall(&*Iter, CsiAfterCallsite, {CallsiteId, FuncId, PropVal});
    }
  }
}

void CSIImpl::interposeCall(Instruction *I) {

  Function *Called = nullptr;
  if (CallInst *CI = dyn_cast<CallInst>(I))
    Called = CI->getCalledFunction();
  else if (InvokeInst *II = dyn_cast<InvokeInst>(I))
    Called = II->getCalledFunction();

  // Should we interpose this call?
  if (Called && Called->getName().size() > 0) {
    bool shouldInterpose =
        Config->DoesFunctionRequireInterposition(Called->getName());

    if (shouldInterpose) {
      Function *interpositionFunction = getInterpositionFunction(Called);
      assert(interpositionFunction != nullptr);
      if (CallInst *CI = dyn_cast<CallInst>(I)) {
        CI->setCalledFunction(interpositionFunction);
      } else if (InvokeInst *II = dyn_cast<InvokeInst>(I)) {
        II->setCalledFunction(interpositionFunction);
      }
    }
  }
}

static void getTaskExits(DetachInst *DI,
                         SmallVectorImpl<BasicBlock *> &TaskReturns,
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
                       [](const Spindle *Pred) { return !Pred->isSharedEH(); }))
        SharedEHExits.push_back(S);
      continue;
    }

    for (BasicBlock *B : S->blocks())
      if (isDetachedRethrow(B->getTerminator()))
        TaskResumes.push_back(B);
  }
}

void CSIImpl::instrumentDetach(DetachInst *DI, DominatorTree *DT, TaskInfo &TI,
                               const DenseMap<Value *, Value *> &TrackVars) {
  // Instrument the detach instruction itself
  Value *DetachID;
  {
    IRBuilder<> IRB(DI);
    uint64_t LocalID = DetachFED.add(*DI);
    DetachID = DetachFED.localToGlobalId(LocalID, IRB);
    Value *TrackVar = TrackVars.lookup(DI->getSyncRegion());
    IRB.CreateStore(
        Constant::getIntegerValue(IntegerType::getInt32Ty(DI->getContext()),
                                  APInt(32, 1)),
        TrackVar);
    insertHookCall(DI, CsiDetach, {DetachID, TrackVar});
  }

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
    // Value *StackSave = IRB.CreateCall(
    //     Intrinsic::getDeclaration(&M, Intrinsic::stacksave));
    Instruction *Call = IRB.CreateCall(CsiTaskEntry, {TaskID, DetachID});
    setInstrumentationDebugLoc(*DetachedBlock, Call);

    // Instrument the exit points of the detached tasks.
    for (BasicBlock *Exit : TaskExits) {
      IRBuilder<> IRB(Exit->getTerminator());
      uint64_t LocalID = TaskExitFED.add(*Exit->getTerminator());
      Value *ExitID = TaskExitFED.localToGlobalId(LocalID, IRB);
      insertHookCall(Exit->getTerminator(), CsiTaskExit,
                     {ExitID, TaskID, DetachID});
    }
    // Instrument the EH exits of the detached task.
    for (BasicBlock *Exit : TaskResumes) {
      IRBuilder<> IRB(Exit->getTerminator());
      uint64_t LocalID = TaskExitFED.add(*Exit->getTerminator());
      Value *ExitID = TaskExitFED.localToGlobalId(LocalID, IRB);
      insertHookCall(Exit->getTerminator(), CsiTaskExit,
                     {ExitID, TaskID, DetachID});
    }

    Task *T = TI.getTaskFor(DetachedBlock);
    Value *DefaultID = getDefaultID(IRB);
    for (Spindle *SharedEH : SharedEHExits)
      insertHookCallAtSharedEHSpindleExits(SharedEH, T, CsiTaskExit,
                                           TaskExitFED, {TaskID, DetachID},
                                           {DefaultID, DefaultID});
  }

  // Instrument the continuation of the detach.
  {
    if (isCriticalContinueEdge(DI, 1))
      ContinueBlock = SplitCriticalEdge(
          DI, 1, CriticalEdgeSplittingOptions(DT).setSplitDetachContinue());

    IRBuilder<> IRB(&*ContinueBlock->getFirstInsertionPt());
    uint64_t LocalID = DetachContinueFED.add(*ContinueBlock);
    Value *ContinueID = DetachContinueFED.localToGlobalId(LocalID, IRB);
    Instruction *Call =
        IRB.CreateCall(CsiDetachContinue, {ContinueID, DetachID});
    setInstrumentationDebugLoc(*ContinueBlock, Call);
  }
  // Instrument the unwind of the detach, if it exists.
  if (DI->hasUnwindDest()) {
    BasicBlock *UnwindBlock = DI->getUnwindDest();
    IRBuilder<> IRB(DI);
    Value *DefaultID = getDefaultID(IRB);
    uint64_t LocalID = DetachContinueFED.add(*UnwindBlock);
    Value *ContinueID = DetachContinueFED.localToGlobalId(LocalID, IRB);
    insertHookCallInSuccessorBB(UnwindBlock, DI->getParent(), CsiDetachContinue,
                                {ContinueID, DetachID}, {DefaultID, DefaultID});
  }
}

void CSIImpl::instrumentSync(SyncInst *SI,
                             const DenseMap<Value *, Value *> &TrackVars) {
  IRBuilder<> IRB(SI);
  Value *DefaultID = getDefaultID(IRB);
  // Get the ID of this sync.
  uint64_t LocalID = SyncFED.add(*SI);
  Value *SyncID = SyncFED.localToGlobalId(LocalID, IRB);

  Value *TrackVar = TrackVars.lookup(SI->getSyncRegion());

  // Insert instrumentation before the sync.
  insertHookCall(SI, CsiBeforeSync, {SyncID, TrackVar});
  CallInst *call = insertHookCallInSuccessorBB(
      SI->getSuccessor(0), SI->getParent(), CsiAfterSync, {SyncID, TrackVar},
      {DefaultID,
       ConstantPointerNull::get(
           IntegerType::getInt32Ty(SI->getContext())->getPointerTo())});

  // Reset the tracking variable to 0.
  if (call != nullptr) {
    callsAfterSync.insert({SI->getSuccessor(0), call});
    IRB.SetInsertPoint(call->getNextNode());
    IRB.CreateStore(
        Constant::getIntegerValue(IntegerType::getInt32Ty(SI->getContext()),
                                  APInt(32, 0)),
        TrackVar);
  } else {
    assert(callsAfterSync.find(SI->getSuccessor(0)) != callsAfterSync.end());
  }
}

void CSIImpl::instrumentAlloca(Instruction *I) {
  IRBuilder<> IRB(I);
  AllocaInst *AI = cast<AllocaInst>(I);

  uint64_t LocalId = AllocaFED.add(*I);
  Value *CsiId = AllocaFED.localToGlobalId(LocalId, IRB);

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

  insertHookCall(I, CsiBeforeAlloca, {CsiId, SizeVal, PropVal});
  BasicBlock::iterator Iter(I);
  Iter++;
  IRB.SetInsertPoint(&*Iter);

  Type *AddrType = IRB.getInt8PtrTy();
  Value *Addr = IRB.CreatePointerCast(I, AddrType);
  insertHookCall(&*Iter, CsiAfterAlloca, {CsiId, Addr, SizeVal, PropVal});
}

void CSIImpl::getAllocFnArgs(const Instruction *I,
                             SmallVectorImpl<Value *> &AllocFnArgs,
                             Type *SizeTy, Type *AddrTy,
                             const TargetLibraryInfo &TLI) {
  const Function *Called = nullptr;
  if (const CallInst *CI = dyn_cast<CallInst>(I))
    Called = CI->getCalledFunction();
  else if (const InvokeInst *II = dyn_cast<InvokeInst>(I))
    Called = II->getCalledFunction();

  LibFunc F;
  bool FoundLibFunc = TLI.getLibFunc(*Called, F);
  if (!FoundLibFunc)
    return;

  switch (F) {
  default:
    return;
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
  case LibFunc_msvc_new_array_longlong_nothrow: {
    // Allocated size
    if (isa<CallInst>(I))
      AllocFnArgs.push_back(cast<CallInst>(I)->getArgOperand(0));
    else
      AllocFnArgs.push_back(cast<InvokeInst>(I)->getArgOperand(0));
    // Number of elements = 1
    AllocFnArgs.push_back(ConstantInt::get(SizeTy, 1));
    // Alignment = 0
    AllocFnArgs.push_back(ConstantInt::get(SizeTy, 0));
    // Old pointer = NULL
    AllocFnArgs.push_back(Constant::getNullValue(AddrTy));
    return;
  }
  case LibFunc_ZnwjSt11align_val_t:
  case LibFunc_ZnwmSt11align_val_t:
  case LibFunc_ZnajSt11align_val_t:
  case LibFunc_ZnamSt11align_val_t:
  case LibFunc_ZnwjSt11align_val_tRKSt9nothrow_t:
  case LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t:
  case LibFunc_ZnajSt11align_val_tRKSt9nothrow_t:
  case LibFunc_ZnamSt11align_val_tRKSt9nothrow_t: {
    if (const CallInst *CI = dyn_cast<CallInst>(I)) {
      AllocFnArgs.push_back(CI->getArgOperand(0));
      // Number of elements = 1
      AllocFnArgs.push_back(ConstantInt::get(SizeTy, 1));
      // Alignment
      AllocFnArgs.push_back(CI->getArgOperand(1));
      // Old pointer = NULL
      AllocFnArgs.push_back(Constant::getNullValue(AddrTy));
    } else {
      const InvokeInst *II = cast<InvokeInst>(I);
      AllocFnArgs.push_back(II->getArgOperand(0));
      // Number of elements = 1
      AllocFnArgs.push_back(ConstantInt::get(SizeTy, 1));
      // Alignment
      AllocFnArgs.push_back(II->getArgOperand(1));
      // Old pointer = NULL
      AllocFnArgs.push_back(Constant::getNullValue(AddrTy));
    }
    return;
  }
  case LibFunc_calloc: {
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
  case LibFunc_reallocf: {
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

void CSIImpl::instrumentAllocFn(Instruction *I, DominatorTree *DT) {
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

  SmallVector<Value *, 4> AllocFnArgs;
  getAllocFnArgs(I, AllocFnArgs, IntptrTy, IRB.getInt8PtrTy(), *TLI);
  SmallVector<Value *, 4> DefaultAllocFnArgs({
      /* Allocated size */ Constant::getNullValue(IntptrTy),
      /* Number of elements */ Constant::getNullValue(IntptrTy),
      /* Alignment */ Constant::getNullValue(IntptrTy),
      /* Old pointer */ Constant::getNullValue(IRB.getInt8PtrTy()),
  });

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
      NormalBB =
          SplitCriticalEdge(II, SuccNum, CriticalEdgeSplittingOptions(DT));
    // Insert hook into normal destination.
    {
      IRB.SetInsertPoint(&*NormalBB->getFirstInsertionPt());
      SmallVector<Value *, 4> AfterAllocFnArgs;
      AfterAllocFnArgs.push_back(AllocFnId);
      AfterAllocFnArgs.push_back(IRB.CreatePointerCast(I, IRB.getInt8PtrTy()));
      AfterAllocFnArgs.append(AllocFnArgs.begin(), AllocFnArgs.end());
      insertHookCall(&*IRB.GetInsertPoint(), CsiAfterAllocFn, AfterAllocFnArgs);
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
      insertHookCallInSuccessorBB(II->getUnwindDest(), II->getParent(),
                                  CsiAfterAllocFn, AfterAllocFnArgs,
                                  DefaultAfterAllocFnArgs);
    }
  } else {
    // Simple call instruction; there is only one "after" position.
    Iter++;
    IRB.SetInsertPoint(&*Iter);
    SmallVector<Value *, 4> AfterAllocFnArgs;
    AfterAllocFnArgs.push_back(AllocFnId);
    AfterAllocFnArgs.push_back(IRB.CreatePointerCast(I, IRB.getInt8PtrTy()));
    AfterAllocFnArgs.append(AllocFnArgs.begin(), AllocFnArgs.end());
    insertHookCall(&*Iter, CsiAfterAllocFn, AfterAllocFnArgs);
  }
}

void CSIImpl::instrumentFree(Instruction *I) {
  // It appears that frees (and deletes) never throw.
  assert(isa<CallInst>(I) && "Free call is not a call instruction");

  CallInst *FC = cast<CallInst>(I);
  Function *Called = FC->getCalledFunction();
  assert(Called && "Could not get called function for free.");

  IRBuilder<> IRB(I);
  uint64_t LocalId = FreeFED.add(*I);
  Value *FreeId = FreeFED.localToGlobalId(LocalId, IRB);

  Value *Addr = FC->getArgOperand(0);
  CsiFreeProperty Prop;
  LibFunc FreeLibF;
  TLI->getLibFunc(*Called, FreeLibF);
  Prop.setFreeTy(static_cast<unsigned>(getFreeTy(FreeLibF)));

  insertHookCall(I, CsiBeforeFree, {FreeId, Addr, Prop.getValue(IRB)});
  BasicBlock::iterator Iter(I);
  Iter++;
  insertHookCall(&*Iter, CsiAfterFree, {FreeId, Addr, Prop.getValue(IRB)});
}

CallInst *CSIImpl::insertHookCall(Instruction *I, Function *HookFunction,
                                  ArrayRef<Value *> HookArgs) {
  IRBuilder<> IRB(I);
  CallInst *Call = IRB.CreateCall(HookFunction, HookArgs);
  setInstrumentationDebugLoc(I, (Instruction *)Call);
  return Call;
}

bool CSIImpl::updateArgPHIs(BasicBlock *Succ, BasicBlock *BB,
                            ArrayRef<Value *> HookArgs,
                            ArrayRef<Value *> DefaultArgs) {
  // If we've already created a PHI node in this block for the hook arguments,
  // just add the incoming arguments to the PHIs.
  if (ArgPHIs.count(Succ)) {
    unsigned HookArgNum = 0;
    for (PHINode *ArgPHI : ArgPHIs[Succ]) {
      ArgPHI->setIncomingValue(ArgPHI->getBasicBlockIndex(BB),
                               HookArgs[HookArgNum]);
      ++HookArgNum;
    }
    return true;
  }

  // Create PHI nodes in this block for each hook argument.
  IRBuilder<> IRB(&Succ->front());
  unsigned HookArgNum = 0;
  for (Value *Arg : HookArgs) {
    PHINode *ArgPHI = IRB.CreatePHI(Arg->getType(), 2);
    for (BasicBlock *Pred : predecessors(Succ)) {
      if (Pred == BB)
        ArgPHI->addIncoming(Arg, BB);
      else
        ArgPHI->addIncoming(DefaultArgs[HookArgNum], Pred);
    }
    ArgPHIs[Succ].push_back(ArgPHI);
    ++HookArgNum;
  }
  return false;
}

CallInst *CSIImpl::insertHookCallInSuccessorBB(BasicBlock *Succ, BasicBlock *BB,
                                               Function *HookFunction,
                                               ArrayRef<Value *> HookArgs,
                                               ArrayRef<Value *> DefaultArgs) {
  assert(HookFunction && "No hook function given.");
  // If this successor block has a unique predecessor, just insert the hook call
  // as normal.
  if (Succ->getUniquePredecessor()) {
    assert(Succ->getUniquePredecessor() == BB &&
           "BB is not unique predecessor of successor block");
    return insertHookCall(&*Succ->getFirstInsertionPt(), HookFunction,
                          HookArgs);
  }

  if (updateArgPHIs(Succ, BB, HookArgs, DefaultArgs))
    return nullptr;

  SmallVector<Value *, 2> SuccessorHookArgs;
  for (PHINode *ArgPHI : ArgPHIs[Succ])
    SuccessorHookArgs.push_back(ArgPHI);

  IRBuilder<> IRB(&*Succ->getFirstInsertionPt());
  // Insert the hook call, using the PHI as the CSI ID.
  CallInst *Call = IRB.CreateCall(HookFunction, SuccessorHookArgs);
  setInstrumentationDebugLoc(*Succ, (Instruction *)Call);

  return Call;
}

void CSIImpl::insertHookCallAtSharedEHSpindleExits(
    Spindle *SharedEHSpindle, Task *T, Function *HookFunction,
    FrontEndDataTable &FED, ArrayRef<Value *> HookArgs,
    ArrayRef<Value *> DefaultArgs) {
  // Get the set of shared EH spindles to examine.  Store them in post order, so
  // they can be evaluated in reverse post order.
  SmallVector<Spindle *, 2> WorkList;
  for (Spindle *S : post_order<InTask<Spindle *>>(SharedEHSpindle))
    WorkList.push_back(S);

  // Traverse the shared-EH spindles in reverse post order, updating the
  // hook-argument PHI's along the way.
  SmallPtrSet<Spindle *, 2> Visited;
  for (Spindle *S : llvm::reverse(WorkList)) {
    bool NewPHINode = false;
    // If this spindle is the first shared-EH spindle in the traversal, use the
    // given hook arguments to update the PHI node.
    if (S == SharedEHSpindle) {
      for (Spindle::SpindleEdge &InEdge : S->in_edges()) {
        Spindle *SPred = InEdge.first;
        BasicBlock *Pred = InEdge.second;
        if (T->contains(SPred))
          NewPHINode |=
              updateArgPHIs(S->getEntry(), Pred, HookArgs, DefaultArgs);
      }
    } else {
      // Otherwise update the PHI node based on the predecessor shared-eh
      // spindles in this RPO traversal.
      for (Spindle::SpindleEdge &InEdge : S->in_edges()) {
        Spindle *SPred = InEdge.first;
        BasicBlock *Pred = InEdge.second;
        if (Visited.count(SPred)) {
          SmallVector<Value *, 4> NewHookArgs(
              ArgPHIs[SPred->getEntry()].begin(),
              ArgPHIs[SPred->getEntry()].end());
          NewPHINode |=
              updateArgPHIs(S->getEntry(), Pred, NewHookArgs, DefaultArgs);
        }
      }
    }
    Visited.insert(S);

    if (!NewPHINode)
      continue;

    // Detached-rethrow exits can appear in strange places within a task-exiting
    // spindle.  Hence we loop over all blocks in the spindle to find detached
    // rethrows.
    for (BasicBlock *B : S->blocks()) {
      if (isDetachedRethrow(B->getTerminator())) {
        IRBuilder<> IRB(B->getTerminator());
        uint64_t LocalID = FED.add(*B->getTerminator());
        Value *HookID = FED.localToGlobalId(LocalID, IRB);
        SmallVector<Value *, 4> Args({HookID});
        Args.append(ArgPHIs[S->getEntry()].begin(),
                    ArgPHIs[S->getEntry()].end());
        Instruction *Call = IRB.CreateCall(HookFunction, Args);
        setInstrumentationDebugLoc(*B, Call);
      }
    }
  }
}

void CSIImpl::initializeFEDTables() {
  FunctionFED = FrontEndDataTable(M, CsiFunctionBaseIdName);
  FunctionExitFED = FrontEndDataTable(M, CsiFunctionExitBaseIdName);
  BasicBlockFED = FrontEndDataTable(M, CsiBasicBlockBaseIdName);
  CallsiteFED = FrontEndDataTable(M, CsiCallsiteBaseIdName);
  LoadFED = FrontEndDataTable(M, CsiLoadBaseIdName);
  StoreFED = FrontEndDataTable(M, CsiStoreBaseIdName);
  AllocaFED = FrontEndDataTable(M, CsiAllocaBaseIdName);
  DetachFED = FrontEndDataTable(M, CsiDetachBaseIdName);
  TaskFED = FrontEndDataTable(M, CsiTaskBaseIdName);
  TaskExitFED = FrontEndDataTable(M, CsiTaskExitBaseIdName);
  DetachContinueFED = FrontEndDataTable(M, CsiDetachContinueBaseIdName);
  SyncFED = FrontEndDataTable(M, CsiSyncBaseIdName);
  AllocFnFED = FrontEndDataTable(M, CsiAllocFnBaseIdName);
  FreeFED = FrontEndDataTable(M, CsiFreeBaseIdName);
}

void CSIImpl::initializeSizeTables() {
  BBSize = SizeTable(M, CsiBasicBlockBaseIdName);
}

uint64_t CSIImpl::getLocalFunctionID(Function &F) {
  uint64_t LocalId = FunctionFED.add(F);
  FuncOffsetMap[F.getName()] = LocalId;
  return LocalId;
}

void CSIImpl::generateInitCallsiteToFunction() {
  LLVMContext &C = M.getContext();
  BasicBlock *EntryBB = BasicBlock::Create(C, "", InitCallsiteToFunction);
  IRBuilder<> IRB(ReturnInst::Create(C, EntryBB));

  GlobalVariable *Base = FunctionFED.baseId();
  LoadInst *LI = IRB.CreateLoad(Base);
  // Traverse the map of function name -> function local id. Generate
  // a store of each function's global ID to the corresponding weak
  // global variable.
  for (const auto &it : FuncOffsetMap) {
    std::string GVName = CsiFuncIdVariablePrefix + it.first.str();
    GlobalVariable *GV = nullptr;
    if ((GV = M.getGlobalVariable(GVName)) == nullptr) {
      GV = new GlobalVariable(M, IRB.getInt64Ty(), false,
                              (Options.jitMode ? GlobalValue::ExternalLinkage :
                               GlobalValue::WeakAnyLinkage),
                              IRB.getInt64(CsiCallsiteUnknownTargetId), GVName);
    }
    assert(GV);
    IRB.CreateStore(IRB.CreateAdd(LI, IRB.getInt64(it.second)), GV);
  }
}

void CSIImpl::initializeCsi() {
  IntptrTy = DL.getIntPtrType(M.getContext());

  initializeFEDTables();
  initializeSizeTables();
  if (Options.InstrumentFuncEntryExit)
    initializeFuncHooks();
  if (Options.InstrumentMemoryAccesses)
    initializeLoadStoreHooks();
  if (Options.InstrumentBasicBlocks)
    initializeBasicBlockHooks();
  if (Options.InstrumentCalls)
    initializeCallsiteHooks();
  if (Options.InstrumentMemIntrinsics)
    initializeMemIntrinsicsHooks();
  if (Options.InstrumentTapir)
    initializeTapirHooks();
  if (Options.InstrumentAllocas)
    initializeAllocaHooks();
  if (Options.InstrumentAllocFns)
    initializeAllocFnHooks();

  FunctionType *FnType =
      FunctionType::get(Type::getVoidTy(M.getContext()), {}, false);
  InitCallsiteToFunction = M.getOrInsertFunction(CsiInitCallsiteToFunctionName,
                                                 FnType);
  assert(InitCallsiteToFunction);

  InitCallsiteToFunction->setLinkage(GlobalValue::InternalLinkage);

  /*
  The runtime declares this as a __thread var --- need to change this decl
  generation or the tool won't compile DisableInstrGV = new GlobalVariable(M,
  IntegerType::get(M.getContext(), 1), false, GlobalValue::ExternalLinkage,
  nullptr, CsiDisableInstrumentationName, nullptr,
                                      GlobalValue::GeneralDynamicTLSModel, 0,
  true);
  */
}

// Create a struct type to match the unit_fed_entry_t type in csirt.c.
StructType *CSIImpl::getUnitFedTableType(LLVMContext &C,
                                         PointerType *EntryPointerType) {
  return StructType::get(IntegerType::get(C, 64), Type::getInt8PtrTy(C, 0),
                         EntryPointerType);
}

Constant *CSIImpl::fedTableToUnitFedTable(Module &M,
                                          StructType *UnitFedTableType,
                                          FrontEndDataTable &FedTable) {
  Constant *NumEntries =
      ConstantInt::get(IntegerType::get(M.getContext(), 64), FedTable.size());
  Constant *BaseIdPtr = ConstantExpr::getPointerCast(
      FedTable.baseId(), Type::getInt8PtrTy(M.getContext(), 0));
  Constant *InsertedTable = FedTable.insertIntoModule(M);
  return ConstantStruct::get(UnitFedTableType, NumEntries, BaseIdPtr,
                             InsertedTable);
}

void CSIImpl::collectUnitFEDTables() {
  LLVMContext &C = M.getContext();
  StructType *UnitFedTableType =
      getUnitFedTableType(C, FrontEndDataTable::getPointerType(C));

  // The order of the FED tables here must match the enum in csirt.c and the
  // instrumentation_counts_t in csi.h.
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, FunctionFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, FunctionExitFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, BasicBlockFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, CallsiteFED));
  UnitFedTables.push_back(fedTableToUnitFedTable(M, UnitFedTableType, LoadFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, StoreFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, DetachFED));
  UnitFedTables.push_back(fedTableToUnitFedTable(M, UnitFedTableType, TaskFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, TaskExitFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, DetachContinueFED));
  UnitFedTables.push_back(fedTableToUnitFedTable(M, UnitFedTableType, SyncFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, AllocaFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, AllocFnFED));
  UnitFedTables.push_back(fedTableToUnitFedTable(M, UnitFedTableType, FreeFED));
}

// Create a struct type to match the unit_obj_entry_t type in csirt.c.
StructType *CSIImpl::getUnitSizeTableType(LLVMContext &C,
                                          PointerType *EntryPointerType) {
  return StructType::get(IntegerType::get(C, 64), EntryPointerType);
}

Constant *CSIImpl::sizeTableToUnitSizeTable(Module &M,
                                            StructType *UnitSizeTableType,
                                            SizeTable &SzTable) {
  Constant *NumEntries =
      ConstantInt::get(IntegerType::get(M.getContext(), 64), SzTable.size());
  // Constant *BaseIdPtr =
  //   ConstantExpr::getPointerCast(FedTable.baseId(),
  //                                Type::getInt8PtrTy(M.getContext(), 0));
  Constant *InsertedTable = SzTable.insertIntoModule(M);
  return ConstantStruct::get(UnitSizeTableType, NumEntries, InsertedTable);
}

void CSIImpl::collectUnitSizeTables() {
  LLVMContext &C = M.getContext();
  StructType *UnitSizeTableType =
      getUnitSizeTableType(C, SizeTable::getPointerType(C));

  UnitSizeTables.push_back(
      sizeTableToUnitSizeTable(M, UnitSizeTableType, BBSize));
}

CallInst *CSIImpl::createRTUnitInitCall(IRBuilder<> &IRB) {
  LLVMContext &C = M.getContext();

  StructType *UnitFedTableType =
      getUnitFedTableType(C, FrontEndDataTable::getPointerType(C));
  StructType *UnitSizeTableType =
      getUnitSizeTableType(C, SizeTable::getPointerType(C));

  // Lookup __csirt_unit_init
  SmallVector<Type *, 4> InitArgTypes({IRB.getInt8PtrTy(),
                                       PointerType::get(UnitFedTableType, 0),
                                       PointerType::get(UnitSizeTableType, 0),
                                       InitCallsiteToFunction->getType()});
  FunctionType *InitFunctionTy =
      FunctionType::get(IRB.getVoidTy(), InitArgTypes, false);
  RTUnitInit = M.getOrInsertFunction(CsiRtUnitInitName, InitFunctionTy);
  assert(RTUnitInit);

  ArrayType *UnitFedTableArrayType =
      ArrayType::get(UnitFedTableType, UnitFedTables.size());
  Constant *FEDTable = ConstantArray::get(UnitFedTableArrayType, UnitFedTables);
  GlobalVariable *FEDGV = new GlobalVariable(
      M, UnitFedTableArrayType, false, GlobalValue::InternalLinkage, FEDTable,
      CsiUnitFedTableArrayName);
  ArrayType *UnitSizeTableArrayType =
      ArrayType::get(UnitSizeTableType, UnitSizeTables.size());
  Constant *SzTable =
      ConstantArray::get(UnitSizeTableArrayType, UnitSizeTables);
  GlobalVariable *SizeGV = new GlobalVariable(
      M, UnitSizeTableArrayType, false, GlobalValue::InternalLinkage, SzTable,
      CsiUnitSizeTableArrayName);

  Constant *Zero = ConstantInt::get(IRB.getInt32Ty(), 0);
  Value *GepArgs[] = {Zero, Zero};

  // Insert call to __csirt_unit_init
  return IRB.CreateCall(
      RTUnitInit,
      {IRB.CreateGlobalStringPtr(M.getName()),
       ConstantExpr::getGetElementPtr(FEDGV->getValueType(), FEDGV, GepArgs),
       ConstantExpr::getGetElementPtr(SizeGV->getValueType(), SizeGV, GepArgs),
       InitCallsiteToFunction});
}

void CSIImpl::finalizeCsi() {
  LLVMContext &C = M.getContext();

  // Add CSI global constructor, which calls unit init.
  Function *Ctor = (Function *)M.getOrInsertFunction(
      CsiRtUnitCtorName, FunctionType::get(Type::getVoidTy(C), false));

  if (!Options.jitMode) {
    Ctor->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
  }

  BasicBlock *CtorBB = BasicBlock::Create(C, "", Ctor);
  IRBuilder<> IRB(ReturnInst::Create(C, CtorBB));

  // Insert __csi_func_id_<f> weak symbols for all defined functions and
  // generate the runtime code that stores to all of them.
  generateInitCallsiteToFunction();

  CallInst *Call = createRTUnitInitCall(IRB);

  // Add the constructor to the global list if we're doing AOT compilation.
  // In JIT mode, we rely on the JIT compiler to call the constructor as
  // a self-standing function.
  if (!Options.jitMode) {
    appendToGlobalCtors(M, Ctor, CsiUnitCtorPriority);

    CallGraphNode *CNCtor = CG->getOrInsertFunction(Ctor);
    CallGraphNode *CNFunc = CG->getOrInsertFunction(RTUnitInit);
    CNCtor->addCalledFunction(Call, CNFunc);
  }
}

void llvm::CSIImpl::linkInToolFromBitcode(const std::string &bitcodePath) {
  if (bitcodePath != "") {
    std::unique_ptr<Module> toolModule;

    SMDiagnostic error;
    auto m = parseIRFile(bitcodePath, error, M.getContext());
    if (m) {
      toolModule = std::move(m);
    } else {
      llvm::errs() << "Error loading bitcode (" << bitcodePath
                   << "): " << error.getMessage() << "\n";
      report_fatal_error(error.getMessage());
    }

    std::vector<std::string> functions;

    for (Function &F : *toolModule) {
      if (!F.isDeclaration() && F.hasName()) {
        functions.push_back(F.getName());
      }
    }

    std::vector<std::string> globalVariables;

    std::vector<GlobalValue *> toRemove;
    for (GlobalValue &val : toolModule->getGlobalList()) {
      if (!val.isDeclaration()) {
        if (val.hasName() && (val.getName() == "llvm.global_ctors" ||
                              val.getName() == "llvm.global_dtors")) {
          toRemove.push_back(&val);
          continue;
        }

        // We can't have globals with internal linkage due to how compile-time
        // instrumentation works. Treat "static" variables as non-static.
        if (val.getLinkage() == GlobalValue::InternalLinkage)
          val.setLinkage(llvm::GlobalValue::CommonLinkage);

        if (val.hasName())
          globalVariables.push_back(val.getName());
      }
    }

    // We remove global constructors and destructors because they'll be linked
    // in at link time when the tool is linked. We can't have duplicates for
    // each translation unit.
    for (auto &val : toRemove) {
      val->eraseFromParent();
    }

    llvm::Linker linker(M);
    linker.linkInModule(std::move(toolModule),
                        llvm::Linker::Flags::LinkOnlyNeeded);

    // Set all tool's globals and functions to be "available externally" so
    // the linker won't complain about multiple definitions.
    for (auto &globalVariableName : globalVariables) {
      auto var = M.getGlobalVariable(globalVariableName);

      if (var && !var->isDeclaration() && !var->hasComdat()) {
        var->setLinkage(
            llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage);
      }
    }
    for (auto &functionName : functions) {
      auto function = M.getFunction(functionName);

      if (function && !function->isDeclaration() && !function->hasComdat()) {
        function->setLinkage(
            GlobalValue::LinkageTypes::AvailableExternallyLinkage);
      }
    }
  }
}

void llvm::CSIImpl::loadConfiguration() {
  if (ClConfigurationFilename != "")
    Config = InstrumentationConfig::ReadFromConfigurationFile(
        ClConfigurationFilename);
  else
    Config = InstrumentationConfig::GetDefault();

  Config->SetConfigMode(ClConfigurationMode);
}

bool CSIImpl::shouldNotInstrumentFunction(Function &F) {
  // Don't instrument standard library calls.
#ifdef WIN32
  if (F.hasName() && F.getName().find("_") == 0) {
    return true;
  }
#endif

  if (F.hasName() && F.getName().find("__csi") != std::string::npos)
    return true;

  // Never instrument the CSI ctor.
  if (F.hasName() && F.getName() == CsiRtUnitCtorName)
    return true;

  // Don't instrument functions that will run before or
  // simultaneously with CSI ctors.
  GlobalVariable *GV = M.getGlobalVariable("llvm.global_ctors");
  if (GV == nullptr)
    return false;
  ConstantArray *CA = cast<ConstantArray>(GV->getInitializer());
  for (Use &OP : CA->operands()) {
    if (isa<ConstantAggregateZero>(OP))
      continue;
    ConstantStruct *CS = cast<ConstantStruct>(OP);

    if (Function *CF = dyn_cast<Function>(CS->getOperand(1))) {
      uint64_t Priority =
          dyn_cast<ConstantInt>(CS->getOperand(0))->getLimitedValue();
      if (Priority <= CsiUnitCtorPriority && CF->getName() == F.getName()) {
        // Do not instrument F.
        return true;
      }
    }
  }
  // false means do instrument it.
  return false;
}

bool CSIImpl::isVtableAccess(Instruction *I) {
  if (MDNode *Tag = I->getMetadata(LLVMContext::MD_tbaa))
    return Tag->isTBAAVtableAccess();
  return false;
}

bool CSIImpl::addrPointsToConstantData(Value *Addr) {
  // If this is a GEP, just analyze its pointer operand.
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Addr))
    Addr = GEP->getPointerOperand();

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Addr)) {
    if (GV->isConstant()) {
      return true;
    }
  } else if (LoadInst *L = dyn_cast<LoadInst>(Addr)) {
    if (isVtableAccess(L)) {
      return true;
    }
  }
  return false;
}

bool CSIImpl::isAtomic(Instruction *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->isAtomic() && LI->getSyncScopeID() != SyncScope::SingleThread;
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->isAtomic() && SI->getSyncScopeID() != SyncScope::SingleThread;
  if (isa<AtomicRMWInst>(I))
    return true;
  if (isa<AtomicCmpXchgInst>(I))
    return true;
  if (isa<FenceInst>(I))
    return true;
  return false;
}

void CSIImpl::computeLoadAndStoreProperties(
    SmallVectorImpl<std::pair<Instruction *, CsiLoadStoreProperty>>
        &LoadAndStoreProperties,
    SmallVectorImpl<Instruction *> &BBLoadsAndStores, const DataLayout &DL) {
  SmallSet<Value *, 8> WriteTargets;

  for (SmallVectorImpl<Instruction *>::reverse_iterator
           It = BBLoadsAndStores.rbegin(),
           E = BBLoadsAndStores.rend();
       It != E; ++It) {
    Instruction *I = *It;
    unsigned Alignment;
    if (StoreInst *Store = dyn_cast<StoreInst>(I)) {
      Value *Addr = Store->getPointerOperand();
      WriteTargets.insert(Addr);
      CsiLoadStoreProperty Prop;
      // Update alignment property data
      Alignment = Store->getAlignment();
      Prop.setAlignment(Alignment);
      // Set vtable-access property
      Prop.setIsVtableAccess(isVtableAccess(Store));
      // Set constant-data-access property
      Prop.setIsConstant(addrPointsToConstantData(Addr));
      Value *Obj = GetUnderlyingObject(Addr, DL);
      // Set is-on-stack property
      Prop.setIsOnStack(isa<AllocaInst>(Obj));
      // Set may-be-captured property
      Prop.setMayBeCaptured(isa<GlobalValue>(Obj) ||
                            PointerMayBeCaptured(Addr, true, true));
      LoadAndStoreProperties.push_back(std::make_pair(I, Prop));
    } else {
      LoadInst *Load = cast<LoadInst>(I);
      Value *Addr = Load->getPointerOperand();
      CsiLoadStoreProperty Prop;
      // Update alignment property data
      Alignment = Load->getAlignment();
      Prop.setAlignment(Alignment);
      // Set vtable-access property
      Prop.setIsVtableAccess(isVtableAccess(Load));
      // Set constant-data-access-property
      Prop.setIsConstant(addrPointsToConstantData(Addr));
      Value *Obj = GetUnderlyingObject(Addr, DL);
      // Set is-on-stack property
      Prop.setIsOnStack(isa<AllocaInst>(Obj));
      // Set may-be-captured property
      Prop.setMayBeCaptured(isa<GlobalValue>(Obj) ||
                            PointerMayBeCaptured(Addr, true, true));
      // Set load-read-before-write-in-bb property
      bool HasBeenSeen = WriteTargets.count(Addr) > 0;
      Prop.setLoadReadBeforeWriteInBB(HasBeenSeen);
      LoadAndStoreProperties.push_back(std::make_pair(I, Prop));
    }
  }
  BBLoadsAndStores.clear();
}

// Update the attributes on the instrumented function that might be invalidated
// by the inserted instrumentation.
void CSIImpl::updateInstrumentedFnAttrs(Function &F) {
  AttrBuilder B;
  B.addAttribute(Attribute::ReadOnly)
      .addAttribute(Attribute::ReadNone)
      .addAttribute(Attribute::ArgMemOnly)
      .addAttribute(Attribute::InaccessibleMemOnly)
      .addAttribute(Attribute::InaccessibleMemOrArgMemOnly);
  F.removeAttributes(AttributeList::FunctionIndex, B);
}

void CSIImpl::instrumentFunction(Function &F) {
  // This is required to prevent instrumenting the call to
  // __csi_module_init from within the module constructor.

  if (F.empty() || shouldNotInstrumentFunction(F))
    return;

  if (Options.CallsMayThrow)
    setupCalls(F);

  setupBlocks(F, TLI);

  SmallVector<std::pair<Instruction *, CsiLoadStoreProperty>, 8>
      LoadAndStoreProperties;
  SmallVector<Instruction *, 8> AllocationFnCalls;
  SmallVector<Instruction *, 8> FreeCalls;
  SmallVector<Instruction *, 8> MemIntrinsics;
  SmallVector<Instruction *, 8> Callsites;
  SmallVector<BasicBlock *, 8> BasicBlocks;
  SmallVector<Instruction *, 8> AtomicAccesses;
  SmallVector<DetachInst *, 8> Detaches;
  SmallVector<SyncInst *, 8> Syncs;
  SmallVector<Instruction *, 8> Allocas;
  SmallVector<Instruction *, 8> AllCalls;
  bool MaySpawn = false;

  DominatorTree *DT = &GetDomTree(F);
  TaskInfo &TI = GetTaskInfo(F);

  // Compile lists of all instrumentation points before anything is modified.
  for (BasicBlock &BB : F) {
    SmallVector<Instruction *, 8> BBLoadsAndStores;
    for (Instruction &I : BB) {
      if (isAtomic(&I))
        AtomicAccesses.push_back(&I);
      else if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
        BBLoadsAndStores.push_back(&I);
      } else if (DetachInst *DI = dyn_cast<DetachInst>(&I)) {
        MaySpawn = true;
        Detaches.push_back(DI);
      } else if (SyncInst *SI = dyn_cast<SyncInst>(&I)) {
        Syncs.push_back(SI);
      } else if (isa<CallInst>(I) || isa<InvokeInst>(I)) {

        // Record this function call as either an allocation function, a call to
        // free (or delete), a memory intrinsic, or an ordinary real function
        // call.
        if (isAllocationFn(&I, TLI))
          AllocationFnCalls.push_back(&I);
        else if (isFreeCall(&I, TLI))
          FreeCalls.push_back(&I);
        else if (isa<MemIntrinsic>(I))
          MemIntrinsics.push_back(&I);
        else
          Callsites.push_back(&I);

        AllCalls.push_back(&I);

        computeLoadAndStoreProperties(LoadAndStoreProperties, BBLoadsAndStores,
                                      DL);
      } else if (isa<AllocaInst>(I)) {
        Allocas.push_back(&I);
      }
    }
    computeLoadAndStoreProperties(LoadAndStoreProperties, BBLoadsAndStores, DL);
    BasicBlocks.push_back(&BB);
  }

  uint64_t LocalId = getLocalFunctionID(F);

  // Instrument basic blocks.  Note that we do this before other instrumentation
  // so that we put this at the beginning of the basic block, and then the
  // function entry call goes before the call to basic block entry.
  if (Options.InstrumentBasicBlocks)
    for (BasicBlock *BB : BasicBlocks)
      instrumentBasicBlock(*BB);

  // Instrument Tapir constructs.
  if (Options.InstrumentTapir) {
    // Allocate a local variable that will keep track of whether
    // a spawn has occurred before a sync. It will be set to 1 after
    // a spawn and reset to 0 after a sync.
    auto TrackVars = keepTrackOfSpawns(F, Detaches, Syncs);

    if (Config->DoesFunctionRequireInstrumentationForPoint(
            F.getName(), InstrumentationPoint::INSTR_TAPIR_DETACH)) {
      for (DetachInst *DI : Detaches)
        instrumentDetach(DI, DT, TI, TrackVars);
    }
    if (Config->DoesFunctionRequireInstrumentationForPoint(
            F.getName(), InstrumentationPoint::INSTR_TAPIR_SYNC)) {
      for (SyncInst *SI : Syncs)
        instrumentSync(SI, TrackVars);
    }
  }

  // Do this work in a separate loop after copying the iterators so that we
  // aren't modifying the list as we're iterating.
  if (Options.InstrumentMemoryAccesses)
    for (std::pair<Instruction *, CsiLoadStoreProperty> p :
         LoadAndStoreProperties)
      instrumentLoadOrStore(p.first, p.second, DL);

  // Instrument atomic memory accesses in any case (they can be used to
  // implement synchronization).
  if (Options.InstrumentAtomics)
    for (Instruction *I : AtomicAccesses)
      instrumentAtomic(I, DL);

  if (Options.InstrumentMemIntrinsics)
    for (Instruction *I : MemIntrinsics)
      instrumentMemIntrinsic(I);

  if (Options.InstrumentCalls)
    for (Instruction *I : Callsites)
      instrumentCallsite(I, DT);

  if (Options.InstrumentAllocas)
    for (Instruction *I : Allocas)
      instrumentAlloca(I);

  if (Options.InstrumentAllocFns) {
    for (Instruction *I : AllocationFnCalls)
      instrumentAllocFn(I, DT);
    for (Instruction *I : FreeCalls)
      instrumentFree(I);
  }

  if (Options.Interpose && Config->DoesAnyFunctionRequireInterposition()) {
    for (Instruction *I : AllCalls)
      interposeCall(I);
  }

  // Instrument function entry/exit points.
  if (Options.InstrumentFuncEntryExit) {
    IRBuilder<> IRB(&*F.getEntryBlock().getFirstInsertionPt());
    Value *FuncId = FunctionFED.localToGlobalId(LocalId, IRB);
    if (Config->DoesFunctionRequireInstrumentationForPoint(
            F.getName(), InstrumentationPoint::INSTR_FUNCTION_ENTRY)) {
      CsiFuncProperty FuncEntryProp;
      FuncEntryProp.setMaySpawn(MaySpawn);
      Value *PropVal = FuncEntryProp.getValue(IRB);
      insertHookCall(&*IRB.GetInsertPoint(), CsiFuncEntry, {FuncId, PropVal});
    }
    if (Config->DoesFunctionRequireInstrumentationForPoint(
            F.getName(), InstrumentationPoint::INSTR_FUNCTION_EXIT)) {
      EscapeEnumerator EE(F, "csi.cleanup", false);
      while (IRBuilder<> *AtExit = EE.Next()) {
        // uint64_t ExitLocalId = FunctionExitFED.add(F);
        uint64_t ExitLocalId = FunctionExitFED.add(*AtExit->GetInsertPoint());
        Value *ExitCsiId =
            FunctionExitFED.localToGlobalId(ExitLocalId, *AtExit);
        CsiFuncExitProperty FuncExitProp;
        FuncExitProp.setMaySpawn(MaySpawn);
        FuncExitProp.setEHReturn(isa<ResumeInst>(AtExit->GetInsertPoint()));
        Value *PropVal = FuncExitProp.getValue(*AtExit);
        insertHookCall(&*AtExit->GetInsertPoint(), CsiFuncExit,
                       {ExitCsiId, FuncId, PropVal});
      }
    }
  }

  updateInstrumentedFnAttrs(F);
}

DenseMap<Value *, Value *>
llvm::CSIImpl::keepTrackOfSpawns(Function &F,
                                 const SmallVectorImpl<DetachInst *> &Detaches,
                                 const SmallVectorImpl<SyncInst *> &Syncs) {

  DenseMap<Value *, Value *> TrackVars;

  SmallPtrSet<Value *, 8> Regions;
  for (auto &Detach : Detaches) {
    Regions.insert(Detach->getSyncRegion());
  }
  for (auto &Sync : Syncs) {
    Regions.insert(Sync->getSyncRegion());
  }

  LLVMContext &C = F.getContext();

  IRBuilder<> Builder{&F.getEntryBlock(),
                      F.getEntryBlock().getFirstInsertionPt()};

  size_t RegionIndex = 0;
  for (auto Region : Regions) {
    Value *TrackVar = Builder.CreateAlloca(IntegerType::getInt32Ty(C), nullptr,
                                           "has_spawned_region_" +
                                               std::to_string(RegionIndex));
    Builder.CreateStore(
        Constant::getIntegerValue(IntegerType::getInt32Ty(C), APInt(32, 0)),
        TrackVar);

    TrackVars.insert({Region, TrackVar});
    RegionIndex++;
  }

  return TrackVars;
}

Function *llvm::CSIImpl::getInterpositionFunction(Function *F) {
  if (InterpositionFunctions.find(F) != InterpositionFunctions.end()) {
    return InterpositionFunctions.lookup(F);
  }

  std::string InterposedName =
      (std::string) "__csi_interpose_" + F->getName().str();

  Function *InterpositionFunction =
      (Function *)M.getOrInsertFunction(InterposedName, F->getFunctionType());

  InterpositionFunctions.insert({F, InterpositionFunction});

  return InterpositionFunction;
}

void ComprehensiveStaticInstrumentationLegacyPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<CallGraphWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<TaskInfoWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
}

bool ComprehensiveStaticInstrumentationLegacyPass::runOnModule(Module &M) {
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

  bool res = CSIImpl(M, CG, GetDomTree, GetTaskInfo, TLI, Options).run();

  verifyModule(M, &llvm::errs());

  numPassRuns++;

  return res;
}

ComprehensiveStaticInstrumentationPass::ComprehensiveStaticInstrumentationPass(
    const CSIOptions &Options)
    : Options(OverrideFromCL(Options)) {}

PreservedAnalyses
ComprehensiveStaticInstrumentationPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  auto &CG = AM.getResult<CallGraphAnalysis>(M);
  auto GetDT = [&FAM](Function &F) -> DominatorTree & {
    return FAM.getResult<DominatorTreeAnalysis>(F);
  };
  auto GetTI = [&FAM](Function &F) -> TaskInfo & {
    return FAM.getResult<TaskAnalysis>(F);
  };
  auto *TLI = &AM.getResult<TargetLibraryAnalysis>(M);

  if (!CSIImpl(M, &CG, GetDT, GetTI, TLI, Options).run())
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
