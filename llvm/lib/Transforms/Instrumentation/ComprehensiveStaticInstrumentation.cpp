//===-- ComprehensiveStaticInstrumentation.cpp - instrumentation hooks ----===//
//
//                     The LLVM Compiler Infrastructure
//
// TODO: License
//===----------------------------------------------------------------------===//
//
// This file is part of CSI, a framework that provides comprehensive static
// instrumentation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/CSI.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "csi"

static cl::opt<bool>  ClInstrumentFuncEntryExit(
    "csi-instrument-func-entry-exit", cl::init(true),
    cl::desc("Instrument function entry and exit"), cl::Hidden);
static cl::opt<bool>  ClInstrumentBasicBlocks(
    "csi-instrument-basic-blocks", cl::init(true),
    cl::desc("Instrument basic blocks"), cl::Hidden);
static cl::opt<bool>  ClInstrumentMemoryAccesses(
    "csi-instrument-memory-accesses", cl::init(true),
    cl::desc("Instrument memory accesses"), cl::Hidden);
static cl::opt<bool>  ClInstrumentCalls(
    "csi-instrument-function-calls", cl::init(true),
    cl::desc("Instrument function calls"), cl::Hidden);
static cl::opt<bool>  ClInstrumentAtomics(
    "csi-instrument-atomics", cl::init(true),
    cl::desc("Instrument atomics"), cl::Hidden);
static cl::opt<bool>  ClInstrumentMemIntrinsics(
    "csi-instrument-memintrinsics", cl::init(true),
    cl::desc("Instrument memintrinsics (memset/memcpy/memmove)"), cl::Hidden);
static cl::opt<bool>  ClInstrumentTapir(
    "csi-instrument-tapir", cl::init(true),
    cl::desc("Instrument tapir constructs"), cl::Hidden);

namespace {

static CSIOptions OverrideFromCL(CSIOptions Options) {
  Options.InstrumentFuncEntryExit |= ClInstrumentFuncEntryExit;
  Options.InstrumentBasicBlocks |= ClInstrumentBasicBlocks;
  Options.InstrumentMemoryAccesses |= ClInstrumentMemoryAccesses;
  Options.InstrumentCalls |= ClInstrumentCalls;
  Options.InstrumentAtomics |= ClInstrumentAtomics;
  Options.InstrumentMemIntrinsics |= ClInstrumentMemIntrinsics;
  Options.InstrumentTapir |= ClInstrumentTapir;
  return Options;
}

/// The Comprehensive Static Instrumentation pass.
/// Inserts calls to user-defined hooks at predefined points in the IR.
struct ComprehensiveStaticInstrumentation : public ModulePass {
  static char ID; // Pass identification, replacement for typeid.

  ComprehensiveStaticInstrumentation(
      const CSIOptions &Options = CSIOptions())
      : ModulePass(ID), Options(OverrideFromCL(Options)) {
    initializeComprehensiveStaticInstrumentationPass(
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

char ComprehensiveStaticInstrumentation::ID = 0;

INITIALIZE_PASS_BEGIN(ComprehensiveStaticInstrumentation, "csi",
                      "ComprehensiveStaticInstrumentation pass",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(ComprehensiveStaticInstrumentation, "csi",
                    "ComprehensiveStaticInstrumentation pass",
                    false, false)

ModulePass *llvm::createComprehensiveStaticInstrumentationPass(
    const CSIOptions &Options) {
  return new ComprehensiveStaticInstrumentation(Options);
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
      LLVMContext &C = Instrumented->getFunction()->getParent()->getContext();
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
      LLVMContext &C = Instrumented.getParent()->getParent()->getContext();
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

bool CSIImpl::run() {
  initializeCsi();

  for (Function &F : M)
    instrumentFunction(F);

  collectUnitFEDTables();
  collectUnitSizeTables();
  finalizeCsi();
  return true; // We always insert the unit constructor.
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
    if (isa<PHINode>(I)) continue;
    if (isa<DbgInfoIntrinsic>(I)) continue;
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I))
      if (Intrinsic::syncregion_start == II->getIntrinsicID() ||
          Intrinsic::lifetime_start == II->getIntrinsicID() ||
          Intrinsic::lifetime_end == II->getIntrinsicID())
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
  LocalIdToSizeMap[ID] = { FullIRSize, NonEmptyIRSize };
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
    TableEntries.push_back(ConstantStruct::get(TableType, FullIRSize,
                                               NonEmptyIRSize));
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
  add(ID, F.getSubprogram());
  return ID;
}

uint64_t FrontEndDataTable::add(const BasicBlock &BB) {
  uint64_t ID = getId(&BB);
  add(ID, getFirstDebugLoc(BB));
  return ID;
}

uint64_t FrontEndDataTable::add(const Instruction &I) {
  uint64_t ID = getId(&I);
  add(ID, I.getDebugLoc());
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

void FrontEndDataTable::add(uint64_t ID, const DILocation *Loc) {
  if (Loc) {
    // TODO: Add location information for inlining
    const DISubprogram *Subprog = Loc->getScope()->getSubprogram();
    add(ID, (int32_t)Loc->getLine(), (int32_t)Loc->getColumn(),
        Loc->getFilename(), Loc->getDirectory(), Subprog->getName());
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
  assert(LocalIdToSourceLocationMap.find(ID) ==
             LocalIdToSourceLocationMap.end() &&
         "Id already exists in FED table.");
  LocalIdToSourceLocationMap[ID] = {Name, Line, Column, Filename, Directory};
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
      Constant *FileStrConstant = ConstantDataArray::getString(C, Filename);
      GlobalVariable *GV =
        M.getGlobalVariable("__csi_unit_filename_" + Filename, true);
      if (GV == NULL) {
        GV = new GlobalVariable(M, FileStrConstant->getType(),
                                true, GlobalValue::PrivateLinkage,
                                FileStrConstant,
                                "__csi_unit_filename_" + Filename,
                                nullptr,
                                GlobalVariable::NotThreadLocal, 0);
        GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      }
      assert(GV);
      File =
        ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
    }
    Constant *Name;
    if (E.Name.empty())
      Name = ConstantPointerNull::get(PointerType::get(
                                          IntegerType::get(C, 8), 0));
    else {
      Constant *NameStrConstant = ConstantDataArray::getString(C, E.Name);
      GlobalVariable *GV =
        M.getGlobalVariable(("__csi_unit_function_name_" + E.Name).str(), true);
      if (GV == NULL) {
        GV = new GlobalVariable(M, NameStrConstant->getType(),
                                true, GlobalValue::PrivateLinkage,
                                NameStrConstant,
                                "__csi_unit_function_name_" + E.Name,
                                nullptr,
                                GlobalVariable::NotThreadLocal, 0);
        GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      }
      assert(GV);
      Name =
        ConstantExpr::getGetElementPtr(GV->getValueType(), GV, GepArgs);
    }
    // The order of arguments to ConstantStruct::get() must match the
    // source_loc_t type in csi.h.
    FEDEntries.push_back(ConstantStruct::get(FedType, Name, Line, Column,
                                             File));
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
  CsiFuncExit =  M.getOrInsertFunction("__csi_func_exit", IRB.getVoidTy(),
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

  CsiDetach = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_detach", RetType,
                            /* detach_id */ IDType));
  CsiTaskEntry = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_task", RetType,
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
  CsiSync = checkCsiInterfaceFunction(
      M.getOrInsertFunction("__csi_sync", RetType, IDType));
}

int CSIImpl::getNumBytesAccessed(Value *Addr, const DataLayout &DL) {
  Type *OrigPtrTy = Addr->getType();
  Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();
  assert(OrigTy->isSized());
  uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  if (TypeSize % 8 != 0) {
  // if (TypeSize != 8 && TypeSize != 16 && TypeSize != 32 && TypeSize != 64 &&
  //     TypeSize != 128 && TypeSize != 256 && TypeSize != 512) {
    return -1;
  }
  return TypeSize / 8;
}

void CSIImpl::addLoadStoreInstrumentation(
    Instruction *I, Function *BeforeFn, Function *AfterFn, Value *CsiId,
    Type *AddrType, Value *Addr, int NumBytes, CsiLoadStoreProperty &Prop) {
  IRBuilder<> IRB(I);
  Value *PropVal = Prop.getValue(IRB);
  insertConditionalHookCall(I, BeforeFn,
                            {CsiId, IRB.CreatePointerCast(Addr, AddrType),
                                IRB.getInt32(NumBytes), PropVal});

  BasicBlock::iterator Iter(I);
  Iter++;
  IRB.SetInsertPoint(&*Iter);
  insertConditionalHookCall(&*Iter, AfterFn,
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
  dbgs() << "WARNING: Uninstrumented atomic operations in program-under-test!\n";
}

// If a memset intrinsic gets inlined by the code gen, we will miss races on it.
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
  TerminatorInst *TI = BB.getTerminator();
  Value *PropVal = Prop.getValue(IRB);
  insertConditionalHookCall(&*IRB.GetInsertPoint(), CsiBBEntry,
                            {CsiId, PropVal});
  insertConditionalHookCall(TI, CsiBBExit,
                            {CsiId, PropVal});
}

void CSIImpl::instrumentCallsite(Instruction *I) {
  // Ignore calls to debug intrinsics
  if (isa<DbgInfoIntrinsic>(I))
    return;

  // Ignore intrinsics that act as placeholders, i.e., the intrinsics don't
  // ultimately result in a function call.
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I))
    if (Intrinsic::syncregion_start == II->getIntrinsicID() ||
        Intrinsic::lifetime_start == II->getIntrinsicID() ||
        Intrinsic::lifetime_end == II->getIntrinsicID())
      return;

  bool IsInvoke = false;
  Function *Called = NULL;
  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    Called = CI->getCalledFunction();
  } else if (InvokeInst *II = dyn_cast<InvokeInst>(I)) {
    Called = II->getCalledFunction();
    IsInvoke = true;
  }

  // if (Called && Called->getName().startswith("llvm.dbg")) {
  //   return;
  // }

  IRBuilder<> IRB(I);
  uint64_t LocalId = CallsiteFED.add(*I);
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
  Prop.setIsIndirect(!Called);
  Value *PropVal = Prop.getValue(IRB);
  insertConditionalHookCall(I, CsiBeforeCallsite,
                            {CallsiteId, FuncId, PropVal});

  BasicBlock::iterator Iter(I);
  if (IsInvoke) {
    // There are two "after" positions for invokes: the normal block
    // and the exception block. This also means we have to recompute
    // the callsite and function IDs in each basic block so that we
    // can use it for the after hook.

    // TODO: Do we want the "after" hook for this callsite to come
    // before or after the BB entry hook? Currently it is inserted
    // before BB entry because instrumentCallsite is called after
    // instrumentBasicBlock.
    InvokeInst *II = dyn_cast<InvokeInst>(I);
    BasicBlock *NormalBB = II->getNormalDest();
    IRB.SetInsertPoint(&*NormalBB->getFirstInsertionPt());
    CallsiteId = CallsiteFED.localToGlobalId(LocalId, IRB);
    if (FuncIdGV != NULL) FuncId = IRB.CreateLoad(FuncIdGV);
    PropVal = Prop.getValue(IRB);
    insertConditionalHookCall(&*IRB.GetInsertPoint(), CsiAfterCallsite,
                              {CallsiteId, FuncId, PropVal});

    BasicBlock *UnwindBB = II->getUnwindDest();
    IRB.SetInsertPoint(&*UnwindBB->getFirstInsertionPt());
    CallsiteId = CallsiteFED.localToGlobalId(LocalId, IRB);
    if (FuncIdGV != NULL) FuncId = IRB.CreateLoad(FuncIdGV);
    PropVal = Prop.getValue(IRB);
    insertConditionalHookCall(&*IRB.GetInsertPoint(), CsiAfterCallsite,
                              {CallsiteId, FuncId, PropVal});
  } else {
    // Simple call instruction; there is only one "after" position.
    Iter++;
    IRB.SetInsertPoint(&*Iter);
    PropVal = Prop.getValue(IRB);
    insertConditionalHookCall(&*Iter, CsiAfterCallsite,
                              {CallsiteId, FuncId, PropVal});
  }
}

void CSIImpl::instrumentDetach(DetachInst *DI, DominatorTree *DT) {
  // Instrument the detach instruction itself
  Value *DetachID;
  {
    IRBuilder<> IRB(DI);
    uint64_t LocalID = DetachFED.add(*DI);
    DetachID = DetachFED.localToGlobalId(LocalID, IRB);
    Instruction *Call = IRB.CreateCall(CsiDetach, {DetachID});
    IRB.SetInstDebugLocation(Call);
  }

  // Find the detached block, continuation, and associated reattaches.
  BasicBlock *DetachedBlock = DI->getDetached();
  BasicBlock *ContinueBlock = DI->getContinue();
  SmallVector<BasicBlock *, 8> TaskExits;
  // TODO: Extend this loop to find EH exits of the detached task.
  for (BasicBlock *Pred : predecessors(ContinueBlock))
    if (isa<ReattachInst>(Pred->getTerminator()))
      TaskExits.push_back(Pred);

  // Instrument the entry and exit points of the detached task.
  {
    // Instrument the entry point of the detached task.
    IRBuilder<> IRB(&*DetachedBlock->getFirstInsertionPt());
    uint64_t LocalID = TaskFED.add(*DetachedBlock);
    Value *TaskID = TaskFED.localToGlobalId(LocalID, IRB);
    Instruction *Call = IRB.CreateCall(CsiTaskEntry,
                                       {TaskID, DetachID});
    IRB.SetInstDebugLocation(Call);

    // Instrument the exit points of the detached tasks.
    for (BasicBlock *TaskExit : TaskExits) {
      IRBuilder<> IRB(TaskExit->getTerminator());
      uint64_t LocalID = TaskExitFED.add(*TaskExit->getTerminator());
      Value *TaskExitID = TaskExitFED.localToGlobalId(LocalID, IRB);
      Instruction *Call = IRB.CreateCall(CsiTaskExit,
                                         {TaskExitID, TaskID, DetachID});
      IRB.SetInstDebugLocation(Call);
    }
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
    Instruction *Call = IRB.CreateCall(CsiDetachContinue,
                                       {ContinueID, DetachID});
    IRB.SetInstDebugLocation(Call);
  }
}

void CSIImpl::instrumentSync(SyncInst *SI) {
  IRBuilder<> IRB(SI);
  // Get the ID of this sync.
  uint64_t LocalID = SyncFED.add(*SI);
  Value *SyncID = SyncFED.localToGlobalId(LocalID, IRB);
  // Insert instrumentation before the sync.
  Instruction *Call = IRB.CreateCall(CsiSync, {SyncID});
  IRB.SetInstDebugLocation(Call);
}

void CSIImpl::insertConditionalHookCall(Instruction *I, Function *HookFunction,
                                        ArrayRef<Value *> HookArgs) {
  IRBuilder<> IRB(I);
  // Value *Cond = IRB.CreateICmpEQ(IRB.CreateLoad(DisableInstrGV), IRB.getInt1(false));
  // TerminatorInst *TI = SplitBlockAndInsertIfThen(Cond, I, false);
  // IRB.SetInsertPoint(TI);
  // IRB.CreateStore(IRB.getInt1(true), DisableInstrGV);
  Instruction *Call = IRB.CreateCall(HookFunction, HookArgs);
  setInstrumentationDebugLoc(I, Call);
  // IRB.CreateStore(IRB.getInt1(false), DisableInstrGV);
}


void CSIImpl::initializeFEDTables() {
  FunctionFED = FrontEndDataTable(M, CsiFunctionBaseIdName);
  FunctionExitFED = FrontEndDataTable(M, CsiFunctionExitBaseIdName);
  BasicBlockFED = FrontEndDataTable(M, CsiBasicBlockBaseIdName);
  CallsiteFED = FrontEndDataTable(M, CsiCallsiteBaseIdName);
  LoadFED = FrontEndDataTable(M, CsiLoadBaseIdName);
  StoreFED = FrontEndDataTable(M, CsiStoreBaseIdName);
  DetachFED = FrontEndDataTable(M, CsiDetachBaseIdName);
  TaskFED = FrontEndDataTable(M, CsiTaskBaseIdName);
  TaskExitFED = FrontEndDataTable(M, CsiTaskExitBaseIdName);
  DetachContinueFED = FrontEndDataTable(M, CsiDetachContinueBaseIdName);
  SyncFED = FrontEndDataTable(M, CsiSyncBaseIdName);
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
                              GlobalValue::WeakAnyLinkage,
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

  FunctionType *FnType =
    FunctionType::get(Type::getVoidTy(M.getContext()), {}, false);
  InitCallsiteToFunction = M.getOrInsertFunction(CsiInitCallsiteToFunctionName,
                                                 FnType);
  assert(InitCallsiteToFunction);
  InitCallsiteToFunction->setLinkage(GlobalValue::InternalLinkage);

  /*
  The runtime declares this as a __thread var --- need to change this decl generation
    or the tool won't compile
  DisableInstrGV = new GlobalVariable(M, IntegerType::get(M.getContext(), 1), false,
                                      GlobalValue::ExternalLinkage, nullptr,
                                      CsiDisableInstrumentationName, nullptr,
                                      GlobalValue::GeneralDynamicTLSModel, 0, true);
  */
}

// Create a struct type to match the unit_fed_entry_t type in csirt.c.
StructType *CSIImpl::getUnitFedTableType(LLVMContext &C,
                                         PointerType *EntryPointerType) {
  return StructType::get(IntegerType::get(C, 64),
                         Type::getInt8PtrTy(C, 0),
                         EntryPointerType);
}

Constant *CSIImpl::fedTableToUnitFedTable(Module &M,
                                          StructType *UnitFedTableType,
                                          FrontEndDataTable &FedTable) {
  Constant *NumEntries =
    ConstantInt::get(IntegerType::get(M.getContext(), 64), FedTable.size());
  Constant *BaseIdPtr =
    ConstantExpr::getPointerCast(FedTable.baseId(),
                                 Type::getInt8PtrTy(M.getContext(), 0));
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
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, LoadFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, StoreFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, DetachFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, TaskFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, TaskExitFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, DetachContinueFED));
  UnitFedTables.push_back(
      fedTableToUnitFedTable(M, UnitFedTableType, SyncFED));
}

// Create a struct type to match the unit_obj_entry_t type in csirt.c.
StructType *CSIImpl::getUnitSizeTableType(LLVMContext &C,
                                          PointerType *EntryPointerType) {
  return StructType::get(IntegerType::get(C, 64),
                         EntryPointerType);
}

Constant *CSIImpl::sizeTableToUnitSizeTable(
    Module &M, StructType *UnitSizeTableType, SizeTable &SzTable) {
  Constant *NumEntries =
    ConstantInt::get(IntegerType::get(M.getContext(), 64), SzTable.size());
  // Constant *BaseIdPtr =
  //   ConstantExpr::getPointerCast(FedTable.baseId(),
  //                                Type::getInt8PtrTy(M.getContext(), 0));
  Constant *InsertedTable = SzTable.insertIntoModule(M);
  return ConstantStruct::get(UnitSizeTableType, NumEntries,
                             InsertedTable);
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
  GlobalVariable *FEDGV = new GlobalVariable(M, UnitFedTableArrayType, false,
                                             GlobalValue::InternalLinkage,
                                             FEDTable,
                                             CsiUnitFedTableArrayName);
  ArrayType *UnitSizeTableArrayType =
      ArrayType::get(UnitSizeTableType, UnitSizeTables.size());
  Constant *SzTable = ConstantArray::get(UnitSizeTableArrayType, UnitSizeTables);
  GlobalVariable *SizeGV = new GlobalVariable(M, UnitSizeTableArrayType, false,
                                              GlobalValue::InternalLinkage,
                                              SzTable,
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
  Function *Ctor =
      Function::Create(FunctionType::get(Type::getVoidTy(C), false),
                       GlobalValue::InternalLinkage, CsiRtUnitCtorName, &M);
  BasicBlock *CtorBB = BasicBlock::Create(C, "", Ctor);
  IRBuilder<> IRB(ReturnInst::Create(C, CtorBB));

  // Insert __csi_func_id_<f> weak symbols for all defined functions and
  // generate the runtime code that stores to all of them.
  generateInitCallsiteToFunction();

  CallInst *Call = createRTUnitInitCall(IRB);

  // Add the constructor to the global list
  appendToGlobalCtors(M, Ctor, CsiUnitCtorPriority);

  CallGraphNode *CNCtor = CG->getOrInsertFunction(Ctor);
  CallGraphNode *CNFunc = CG->getOrInsertFunction(RTUnitInit);
  CNCtor->addCalledFunction(Call, CNFunc);
}

bool CSIImpl::shouldNotInstrumentFunction(Function &F) {
  Module &M = *F.getParent();
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
    SmallVectorImpl<std::pair<Instruction *, CsiLoadStoreProperty>> &LoadAndStoreProperties,
    SmallVectorImpl<Instruction *> &BBLoadsAndStores,
    const DataLayout &DL) {
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

void CSIImpl::instrumentFunction(Function &F) {
  // This is required to prevent instrumenting the call to
  // __csi_module_init from within the module constructor.
  if (F.empty() || shouldNotInstrumentFunction(F))
    return;

  DominatorTree *DT = &GetDomTree(F);

  SmallVector<std::pair<Instruction *, CsiLoadStoreProperty>, 8>
    LoadAndStoreProperties;
  SmallVector<Instruction *, 8> ReturnInstructions;
  SmallVector<Instruction *, 8> MemIntrinsics;
  SmallVector<Instruction *, 8> Callsites;
  SmallVector<BasicBlock *, 8> BasicBlocks;
  SmallVector<Instruction*, 8> AtomicAccesses;
  SmallVector<DetachInst*, 8> Detaches;
  SmallVector<SyncInst*, 8> Syncs;
  bool MaySpawn = false;

  // Compile lists of all instrumentation points before anything is modified.
  for (BasicBlock &BB : F) {
    SmallVector<Instruction *, 8> BBLoadsAndStores;
    for (Instruction &I : BB) {
      if (isAtomic(&I))
        AtomicAccesses.push_back(&I);
      else if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
        BBLoadsAndStores.push_back(&I);
      } else if (isa<ReturnInst>(I)) {
        ReturnInstructions.push_back(&I);
      } else if (DetachInst *DI = dyn_cast<DetachInst>(&I)) {
        MaySpawn = true;
        Detaches.push_back(DI);
      } else if (SyncInst *SI = dyn_cast<SyncInst>(&I)) {
        Syncs.push_back(SI);
      } else if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
        if (isa<MemIntrinsic>(I)) {
          MemIntrinsics.push_back(&I);
        } else {
          Callsites.push_back(&I);
        }
        computeLoadAndStoreProperties(LoadAndStoreProperties, BBLoadsAndStores,
                                      DL);
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
    for (DetachInst *DI : Detaches)
      instrumentDetach(DI, DT);
    for (SyncInst *SI : Syncs)
      instrumentSync(SI);
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
      instrumentCallsite(I);

  // Instrument function entry/exit points.
  if (Options.InstrumentFuncEntryExit) {
    IRBuilder<> IRB(&*F.getEntryBlock().getFirstInsertionPt());
    CsiFuncProperty FuncEntryProp;
    FuncEntryProp.setMaySpawn(MaySpawn);
    CsiFuncExitProperty FuncExitProp;
    Value *FuncId = FunctionFED.localToGlobalId(LocalId, IRB);
    Value *PropVal = FuncEntryProp.getValue(IRB);
    insertConditionalHookCall(&*IRB.GetInsertPoint(), CsiFuncEntry,
                              {FuncId, PropVal});

    for (Instruction *I : ReturnInstructions) {
      IRBuilder<> IRBRet(I);
      // uint64_t ExitLocalId = FunctionExitFED.add(F);
      uint64_t ExitLocalId = FunctionExitFED.add(*I);
      Value *ExitCsiId = FunctionExitFED.localToGlobalId(ExitLocalId, IRBRet);
      PropVal = FuncExitProp.getValue(IRBRet);
      insertConditionalHookCall(I, CsiFuncExit,
                                {ExitCsiId, FuncId, PropVal});
    }
  }
}

void ComprehensiveStaticInstrumentation::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<CallGraphWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
}

bool ComprehensiveStaticInstrumentation::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  CallGraph *CG = &getAnalysis<CallGraphWrapperPass>().getCallGraph();
  auto GetDomTree = [this](Function &F) -> DominatorTree & {
    return this->getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
  };

  return CSIImpl(M, CG, GetDomTree, Options).run();
}
