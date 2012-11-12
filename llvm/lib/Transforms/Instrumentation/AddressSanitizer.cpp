//===-- AddressSanitizer.cpp - memory error detector ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
// Details of the algorithm:
//  http://code.google.com/p/address-sanitizer/wiki/AddressSanitizerAlgorithm
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asan"

#include "BlackList.h"
#include "llvm/Function.h"
#include "llvm/IRBuilder.h"
#include "llvm/InlineAsm.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <string>
#include <algorithm>

using namespace llvm;

static const uint64_t kDefaultShadowScale = 3;
static const uint64_t kDefaultShadowOffset32 = 1ULL << 29;
static const uint64_t kDefaultShadowOffset64 = 1ULL << 44;
static const uint64_t kDefaultShadowOffsetAndroid = 0;

static const size_t kMaxStackMallocSize = 1 << 16;  // 64K
static const uintptr_t kCurrentStackFrameMagic = 0x41B58AB3;
static const uintptr_t kRetiredStackFrameMagic = 0x45E0360E;

static const char *kAsanModuleCtorName = "asan.module_ctor";
static const char *kAsanModuleDtorName = "asan.module_dtor";
static const int   kAsanCtorAndCtorPriority = 1;
static const char *kAsanReportErrorTemplate = "__asan_report_";
static const char *kAsanRegisterGlobalsName = "__asan_register_globals";
static const char *kAsanUnregisterGlobalsName = "__asan_unregister_globals";
static const char *kAsanPoisonGlobalsName = "__asan_before_dynamic_init";
static const char *kAsanUnpoisonGlobalsName = "__asan_after_dynamic_init";
static const char *kAsanInitName = "__asan_init";
static const char *kAsanHandleNoReturnName = "__asan_handle_no_return";
static const char *kAsanMappingOffsetName = "__asan_mapping_offset";
static const char *kAsanMappingScaleName = "__asan_mapping_scale";
static const char *kAsanStackMallocName = "__asan_stack_malloc";
static const char *kAsanStackFreeName = "__asan_stack_free";

static const int kAsanStackLeftRedzoneMagic = 0xf1;
static const int kAsanStackMidRedzoneMagic = 0xf2;
static const int kAsanStackRightRedzoneMagic = 0xf3;
static const int kAsanStackPartialRedzoneMagic = 0xf4;

// Accesses sizes are powers of two: 1, 2, 4, 8, 16.
static const size_t kNumberOfAccessSizes = 5;

// Command-line flags.

// This flag may need to be replaced with -f[no-]asan-reads.
static cl::opt<bool> ClInstrumentReads("asan-instrument-reads",
       cl::desc("instrument read instructions"), cl::Hidden, cl::init(true));
static cl::opt<bool> ClInstrumentWrites("asan-instrument-writes",
       cl::desc("instrument write instructions"), cl::Hidden, cl::init(true));
static cl::opt<bool> ClInstrumentAtomics("asan-instrument-atomics",
       cl::desc("instrument atomic instructions (rmw, cmpxchg)"),
       cl::Hidden, cl::init(true));
static cl::opt<bool> ClAlwaysSlowPath("asan-always-slow-path",
       cl::desc("use instrumentation with slow path for all accesses"),
       cl::Hidden, cl::init(false));
// This flag limits the number of instructions to be instrumented
// in any given BB. Normally, this should be set to unlimited (INT_MAX),
// but due to http://llvm.org/bugs/show_bug.cgi?id=12652 we temporary
// set it to 10000.
static cl::opt<int> ClMaxInsnsToInstrumentPerBB("asan-max-ins-per-bb",
       cl::init(10000),
       cl::desc("maximal number of instructions to instrument in any given BB"),
       cl::Hidden);
// This flag may need to be replaced with -f[no]asan-stack.
static cl::opt<bool> ClStack("asan-stack",
       cl::desc("Handle stack memory"), cl::Hidden, cl::init(true));
// This flag may need to be replaced with -f[no]asan-use-after-return.
static cl::opt<bool> ClUseAfterReturn("asan-use-after-return",
       cl::desc("Check return-after-free"), cl::Hidden, cl::init(false));
// This flag may need to be replaced with -f[no]asan-globals.
static cl::opt<bool> ClGlobals("asan-globals",
       cl::desc("Handle global objects"), cl::Hidden, cl::init(true));
static cl::opt<bool> ClInitializers("asan-initialization-order",
       cl::desc("Handle C++ initializer order"), cl::Hidden, cl::init(false));
static cl::opt<bool> ClMemIntrin("asan-memintrin",
       cl::desc("Handle memset/memcpy/memmove"), cl::Hidden, cl::init(true));
// This flag may need to be replaced with -fasan-blacklist.
static cl::opt<std::string>  ClBlackListFile("asan-blacklist",
       cl::desc("File containing the list of functions to ignore "
                "during instrumentation"), cl::Hidden);

// These flags allow to change the shadow mapping.
// The shadow mapping looks like
//    Shadow = (Mem >> scale) + (1 << offset_log)
static cl::opt<int> ClMappingScale("asan-mapping-scale",
       cl::desc("scale of asan shadow mapping"), cl::Hidden, cl::init(0));
static cl::opt<int> ClMappingOffsetLog("asan-mapping-offset-log",
       cl::desc("offset of asan shadow mapping"), cl::Hidden, cl::init(-1));

// Optimization flags. Not user visible, used mostly for testing
// and benchmarking the tool.
static cl::opt<bool> ClOpt("asan-opt",
       cl::desc("Optimize instrumentation"), cl::Hidden, cl::init(true));
static cl::opt<bool> ClOptSameTemp("asan-opt-same-temp",
       cl::desc("Instrument the same temp just once"), cl::Hidden,
       cl::init(true));
static cl::opt<bool> ClOptGlobals("asan-opt-globals",
       cl::desc("Don't instrument scalar globals"), cl::Hidden, cl::init(true));

// Debug flags.
static cl::opt<int> ClDebug("asan-debug", cl::desc("debug"), cl::Hidden,
                            cl::init(0));
static cl::opt<int> ClDebugStack("asan-debug-stack", cl::desc("debug stack"),
                                 cl::Hidden, cl::init(0));
static cl::opt<std::string> ClDebugFunc("asan-debug-func",
                                        cl::Hidden, cl::desc("Debug func"));
static cl::opt<int> ClDebugMin("asan-debug-min", cl::desc("Debug min inst"),
                               cl::Hidden, cl::init(-1));
static cl::opt<int> ClDebugMax("asan-debug-max", cl::desc("Debug man inst"),
                               cl::Hidden, cl::init(-1));

namespace {
/// AddressSanitizer: instrument the code in module to find memory bugs.
struct AddressSanitizer : public FunctionPass {
  AddressSanitizer();
  virtual const char *getPassName() const;
  void instrumentMop(Instruction *I);
  void instrumentAddress(Instruction *OrigIns, IRBuilder<> &IRB,
                         Value *Addr, uint32_t TypeSize, bool IsWrite);
  Value *createSlowPathCmp(IRBuilder<> &IRB, Value *AddrLong,
                           Value *ShadowValue, uint32_t TypeSize);
  Instruction *generateCrashCode(Instruction *InsertBefore, Value *Addr,
                                 bool IsWrite, size_t AccessSizeIndex);
  bool instrumentMemIntrinsic(MemIntrinsic *MI);
  void instrumentMemIntrinsicParam(Instruction *OrigIns, Value *Addr,
                                   Value *Size,
                                   Instruction *InsertBefore, bool IsWrite);
  Value *memToShadow(Value *Shadow, IRBuilder<> &IRB);
  bool runOnFunction(Function &F);
  void createInitializerPoisonCalls(Module &M,
                                    Value *FirstAddr, Value *LastAddr);
  bool maybeInsertAsanInitAtFunctionEntry(Function &F);
  bool poisonStackInFunction(Function &F);
  virtual bool doInitialization(Module &M);
  virtual bool doFinalization(Module &M);
  bool insertGlobalRedzones(Module &M);
  static char ID;  // Pass identification, replacement for typeid

 private:
  uint64_t getAllocaSizeInBytes(AllocaInst *AI) {
    Type *Ty = AI->getAllocatedType();
    uint64_t SizeInBytes = TD->getTypeAllocSize(Ty);
    return SizeInBytes;
  }
  uint64_t getAlignedSize(uint64_t SizeInBytes) {
    return ((SizeInBytes + RedzoneSize - 1)
            / RedzoneSize) * RedzoneSize;
  }
  uint64_t getAlignedAllocaSize(AllocaInst *AI) {
    uint64_t SizeInBytes = getAllocaSizeInBytes(AI);
    return getAlignedSize(SizeInBytes);
  }

  Function *checkInterfaceFunction(Constant *FuncOrBitcast);
  bool ShouldInstrumentGlobal(GlobalVariable *G);
  void PoisonStack(const ArrayRef<AllocaInst*> &AllocaVec, IRBuilder<> IRB,
                   Value *ShadowBase, bool DoPoison);
  bool LooksLikeCodeInBug11395(Instruction *I);
  void FindDynamicInitializers(Module &M);
  bool HasDynamicInitializer(GlobalVariable *G);

  LLVMContext *C;
  DataLayout *TD;
  uint64_t MappingOffset;
  int MappingScale;
  size_t RedzoneSize;
  int LongSize;
  Type *IntptrTy;
  Type *IntptrPtrTy;
  Function *AsanCtorFunction;
  Function *AsanInitFunction;
  Function *AsanStackMallocFunc, *AsanStackFreeFunc;
  Function *AsanHandleNoReturnFunc;
  Instruction *CtorInsertBefore;
  OwningPtr<BlackList> BL;
  // This array is indexed by AccessIsWrite and log2(AccessSize).
  Function *AsanErrorCallback[2][kNumberOfAccessSizes];
  InlineAsm *EmptyAsm;
  SmallSet<GlobalValue*, 32> DynamicallyInitializedGlobals;
  SmallSet<GlobalValue*, 32> GlobalsCreatedByAsan;
};

}  // namespace

char AddressSanitizer::ID = 0;
INITIALIZE_PASS(AddressSanitizer, "asan",
    "AddressSanitizer: detects use-after-free and out-of-bounds bugs.",
    false, false)
AddressSanitizer::AddressSanitizer() : FunctionPass(ID) { }
FunctionPass *llvm::createAddressSanitizerPass() {
  return new AddressSanitizer();
}

const char *AddressSanitizer::getPassName() const {
  return "AddressSanitizer";
}

static size_t TypeSizeToSizeIndex(uint32_t TypeSize) {
  size_t Res = CountTrailingZeros_32(TypeSize / 8);
  assert(Res < kNumberOfAccessSizes);
  return Res;
}

// Create a constant for Str so that we can pass it to the run-time lib.
static GlobalVariable *createPrivateGlobalForString(Module &M, StringRef Str) {
  Constant *StrConst = ConstantDataArray::getString(M.getContext(), Str);
  return new GlobalVariable(M, StrConst->getType(), true,
                            GlobalValue::PrivateLinkage, StrConst, "");
}

Value *AddressSanitizer::memToShadow(Value *Shadow, IRBuilder<> &IRB) {
  // Shadow >> scale
  Shadow = IRB.CreateLShr(Shadow, MappingScale);
  if (MappingOffset == 0)
    return Shadow;
  // (Shadow >> scale) | offset
  return IRB.CreateOr(Shadow, ConstantInt::get(IntptrTy,
                                               MappingOffset));
}

void AddressSanitizer::instrumentMemIntrinsicParam(
    Instruction *OrigIns,
    Value *Addr, Value *Size, Instruction *InsertBefore, bool IsWrite) {
  // Check the first byte.
  {
    IRBuilder<> IRB(InsertBefore);
    instrumentAddress(OrigIns, IRB, Addr, 8, IsWrite);
  }
  // Check the last byte.
  {
    IRBuilder<> IRB(InsertBefore);
    Value *SizeMinusOne = IRB.CreateSub(
        Size, ConstantInt::get(Size->getType(), 1));
    SizeMinusOne = IRB.CreateIntCast(SizeMinusOne, IntptrTy, false);
    Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);
    Value *AddrPlusSizeMinisOne = IRB.CreateAdd(AddrLong, SizeMinusOne);
    instrumentAddress(OrigIns, IRB, AddrPlusSizeMinisOne, 8, IsWrite);
  }
}

// Instrument memset/memmove/memcpy
bool AddressSanitizer::instrumentMemIntrinsic(MemIntrinsic *MI) {
  Value *Dst = MI->getDest();
  MemTransferInst *MemTran = dyn_cast<MemTransferInst>(MI);
  Value *Src = MemTran ? MemTran->getSource() : 0;
  Value *Length = MI->getLength();

  Constant *ConstLength = dyn_cast<Constant>(Length);
  Instruction *InsertBefore = MI;
  if (ConstLength) {
    if (ConstLength->isNullValue()) return false;
  } else {
    // The size is not a constant so it could be zero -- check at run-time.
    IRBuilder<> IRB(InsertBefore);

    Value *Cmp = IRB.CreateICmpNE(Length,
                                  Constant::getNullValue(Length->getType()));
    InsertBefore = SplitBlockAndInsertIfThen(cast<Instruction>(Cmp), false);
  }

  instrumentMemIntrinsicParam(MI, Dst, Length, InsertBefore, true);
  if (Src)
    instrumentMemIntrinsicParam(MI, Src, Length, InsertBefore, false);
  return true;
}

// If I is an interesting memory access, return the PointerOperand
// and set IsWrite. Otherwise return NULL.
static Value *isInterestingMemoryAccess(Instruction *I, bool *IsWrite) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    if (!ClInstrumentReads) return NULL;
    *IsWrite = false;
    return LI->getPointerOperand();
  }
  if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    if (!ClInstrumentWrites) return NULL;
    *IsWrite = true;
    return SI->getPointerOperand();
  }
  if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
    if (!ClInstrumentAtomics) return NULL;
    *IsWrite = true;
    return RMW->getPointerOperand();
  }
  if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(I)) {
    if (!ClInstrumentAtomics) return NULL;
    *IsWrite = true;
    return XCHG->getPointerOperand();
  }
  return NULL;
}

void AddressSanitizer::FindDynamicInitializers(Module& M) {
  // Clang generates metadata identifying all dynamically initialized globals.
  NamedMDNode *DynamicGlobals =
      M.getNamedMetadata("llvm.asan.dynamically_initialized_globals");
  if (!DynamicGlobals)
    return;
  for (int i = 0, n = DynamicGlobals->getNumOperands(); i < n; ++i) {
    MDNode *MDN = DynamicGlobals->getOperand(i);
    assert(MDN->getNumOperands() == 1);
    Value *VG = MDN->getOperand(0);
    // The optimizer may optimize away a global entirely, in which case we
    // cannot instrument access to it.
    if (!VG)
      continue;

    GlobalVariable *G = cast<GlobalVariable>(VG);
    DynamicallyInitializedGlobals.insert(G);
  }
}
// Returns true if a global variable is initialized dynamically in this TU.
bool AddressSanitizer::HasDynamicInitializer(GlobalVariable *G) {
  return DynamicallyInitializedGlobals.count(G);
}

void AddressSanitizer::instrumentMop(Instruction *I) {
  bool IsWrite = false;
  Value *Addr = isInterestingMemoryAccess(I, &IsWrite);
  assert(Addr);
  if (ClOpt && ClOptGlobals) {
    if (GlobalVariable *G = dyn_cast<GlobalVariable>(Addr)) {
      // If initialization order checking is disabled, a simple access to a
      // dynamically initialized global is always valid.
      if (!ClInitializers)
        return;
      // If a global variable does not have dynamic initialization we don't
      // have to instrument it.  However, if a global has external linkage, we
      // assume it has dynamic initialization, as it may have an initializer
      // in a different TU.
      if (G->getLinkage() != GlobalVariable::ExternalLinkage &&
          !HasDynamicInitializer(G))
        return;
    }
  }

  Type *OrigPtrTy = Addr->getType();
  Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();

  assert(OrigTy->isSized());
  uint32_t TypeSize = TD->getTypeStoreSizeInBits(OrigTy);

  if (TypeSize != 8  && TypeSize != 16 &&
      TypeSize != 32 && TypeSize != 64 && TypeSize != 128) {
    // Ignore all unusual sizes.
    return;
  }

  IRBuilder<> IRB(I);
  instrumentAddress(I, IRB, Addr, TypeSize, IsWrite);
}

// Validate the result of Module::getOrInsertFunction called for an interface
// function of AddressSanitizer. If the instrumented module defines a function
// with the same name, their prototypes must match, otherwise
// getOrInsertFunction returns a bitcast.
Function *AddressSanitizer::checkInterfaceFunction(Constant *FuncOrBitcast) {
  if (isa<Function>(FuncOrBitcast)) return cast<Function>(FuncOrBitcast);
  FuncOrBitcast->dump();
  report_fatal_error("trying to redefine an AddressSanitizer "
                     "interface function");
}

Instruction *AddressSanitizer::generateCrashCode(
    Instruction *InsertBefore, Value *Addr,
    bool IsWrite, size_t AccessSizeIndex) {
  IRBuilder<> IRB(InsertBefore);
  CallInst *Call = IRB.CreateCall(AsanErrorCallback[IsWrite][AccessSizeIndex],
                                  Addr);
  // We don't do Call->setDoesNotReturn() because the BB already has
  // UnreachableInst at the end.
  // This EmptyAsm is required to avoid callback merge.
  IRB.CreateCall(EmptyAsm);
  return Call;
}

Value *AddressSanitizer::createSlowPathCmp(IRBuilder<> &IRB, Value *AddrLong,
                                            Value *ShadowValue,
                                            uint32_t TypeSize) {
  size_t Granularity = 1 << MappingScale;
  // Addr & (Granularity - 1)
  Value *LastAccessedByte = IRB.CreateAnd(
      AddrLong, ConstantInt::get(IntptrTy, Granularity - 1));
  // (Addr & (Granularity - 1)) + size - 1
  if (TypeSize / 8 > 1)
    LastAccessedByte = IRB.CreateAdd(
        LastAccessedByte, ConstantInt::get(IntptrTy, TypeSize / 8 - 1));
  // (uint8_t) ((Addr & (Granularity-1)) + size - 1)
  LastAccessedByte = IRB.CreateIntCast(
      LastAccessedByte, ShadowValue->getType(), false);
  // ((uint8_t) ((Addr & (Granularity-1)) + size - 1)) >= ShadowValue
  return IRB.CreateICmpSGE(LastAccessedByte, ShadowValue);
}

void AddressSanitizer::instrumentAddress(Instruction *OrigIns,
                                         IRBuilder<> &IRB, Value *Addr,
                                         uint32_t TypeSize, bool IsWrite) {
  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);

  Type *ShadowTy  = IntegerType::get(
      *C, std::max(8U, TypeSize >> MappingScale));
  Type *ShadowPtrTy = PointerType::get(ShadowTy, 0);
  Value *ShadowPtr = memToShadow(AddrLong, IRB);
  Value *CmpVal = Constant::getNullValue(ShadowTy);
  Value *ShadowValue = IRB.CreateLoad(
      IRB.CreateIntToPtr(ShadowPtr, ShadowPtrTy));

  Value *Cmp = IRB.CreateICmpNE(ShadowValue, CmpVal);
  size_t AccessSizeIndex = TypeSizeToSizeIndex(TypeSize);
  size_t Granularity = 1 << MappingScale;
  TerminatorInst *CrashTerm = 0;

  if (ClAlwaysSlowPath || (TypeSize < 8 * Granularity)) {
    TerminatorInst *CheckTerm =
        SplitBlockAndInsertIfThen(cast<Instruction>(Cmp), false);
    assert(dyn_cast<BranchInst>(CheckTerm)->isUnconditional());
    BasicBlock *NextBB = CheckTerm->getSuccessor(0);
    IRB.SetInsertPoint(CheckTerm);
    Value *Cmp2 = createSlowPathCmp(IRB, AddrLong, ShadowValue, TypeSize);
    BasicBlock *CrashBlock =
        BasicBlock::Create(*C, "", NextBB->getParent(), NextBB);
    CrashTerm = new UnreachableInst(*C, CrashBlock);
    BranchInst *NewTerm = BranchInst::Create(CrashBlock, NextBB, Cmp2);
    ReplaceInstWithInst(CheckTerm, NewTerm);
  } else {
    CrashTerm = SplitBlockAndInsertIfThen(cast<Instruction>(Cmp), true);
  }

  Instruction *Crash =
      generateCrashCode(CrashTerm, AddrLong, IsWrite, AccessSizeIndex);
  Crash->setDebugLoc(OrigIns->getDebugLoc());
}

void AddressSanitizer::createInitializerPoisonCalls(Module &M,
                                                    Value *FirstAddr,
                                                    Value *LastAddr) {
  // We do all of our poisoning and unpoisoning within _GLOBAL__I_a.
  Function *GlobalInit = M.getFunction("_GLOBAL__I_a");
  // If that function is not present, this TU contains no globals, or they have
  // all been optimized away
  if (!GlobalInit)
    return;

  // Set up the arguments to our poison/unpoison functions.
  IRBuilder<> IRB(GlobalInit->begin()->getFirstInsertionPt());

  // Declare our poisoning and unpoisoning functions.
  Function *AsanPoisonGlobals = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanPoisonGlobalsName, IRB.getVoidTy(), IntptrTy, IntptrTy, NULL));
  AsanPoisonGlobals->setLinkage(Function::ExternalLinkage);
  Function *AsanUnpoisonGlobals = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanUnpoisonGlobalsName, IRB.getVoidTy(), NULL));
  AsanUnpoisonGlobals->setLinkage(Function::ExternalLinkage);

  // Add a call to poison all external globals before the given function starts.
  IRB.CreateCall2(AsanPoisonGlobals, FirstAddr, LastAddr);

  // Add calls to unpoison all globals before each return instruction.
  for (Function::iterator I = GlobalInit->begin(), E = GlobalInit->end();
      I != E; ++I) {
    if (ReturnInst *RI = dyn_cast<ReturnInst>(I->getTerminator())) {
      CallInst::Create(AsanUnpoisonGlobals, "", RI);
    }
  }
}

bool AddressSanitizer::ShouldInstrumentGlobal(GlobalVariable *G) {
  Type *Ty = cast<PointerType>(G->getType())->getElementType();
  DEBUG(dbgs() << "GLOBAL: " << *G << "\n");

  if (BL->isIn(*G)) return false;
  if (!Ty->isSized()) return false;
  if (!G->hasInitializer()) return false;
  if (GlobalsCreatedByAsan.count(G)) return false;  // Our own global.
  // Touch only those globals that will not be defined in other modules.
  // Don't handle ODR type linkages since other modules may be built w/o asan.
  if (G->getLinkage() != GlobalVariable::ExternalLinkage &&
      G->getLinkage() != GlobalVariable::PrivateLinkage &&
      G->getLinkage() != GlobalVariable::InternalLinkage)
    return false;
  // Two problems with thread-locals:
  //   - The address of the main thread's copy can't be computed at link-time.
  //   - Need to poison all copies, not just the main thread's one.
  if (G->isThreadLocal())
    return false;
  // For now, just ignore this Alloca if the alignment is large.
  if (G->getAlignment() > RedzoneSize) return false;

  // Ignore all the globals with the names starting with "\01L_OBJC_".
  // Many of those are put into the .cstring section. The linker compresses
  // that section by removing the spare \0s after the string terminator, so
  // our redzones get broken.
  if ((G->getName().find("\01L_OBJC_") == 0) ||
      (G->getName().find("\01l_OBJC_") == 0)) {
    DEBUG(dbgs() << "Ignoring \\01L_OBJC_* global: " << *G);
    return false;
  }

  if (G->hasSection()) {
    StringRef Section(G->getSection());
    // Ignore the globals from the __OBJC section. The ObjC runtime assumes
    // those conform to /usr/lib/objc/runtime.h, so we can't add redzones to
    // them.
    if ((Section.find("__OBJC,") == 0) ||
        (Section.find("__DATA, __objc_") == 0)) {
      DEBUG(dbgs() << "Ignoring ObjC runtime global: " << *G);
      return false;
    }
    // See http://code.google.com/p/address-sanitizer/issues/detail?id=32
    // Constant CFString instances are compiled in the following way:
    //  -- the string buffer is emitted into
    //     __TEXT,__cstring,cstring_literals
    //  -- the constant NSConstantString structure referencing that buffer
    //     is placed into __DATA,__cfstring
    // Therefore there's no point in placing redzones into __DATA,__cfstring.
    // Moreover, it causes the linker to crash on OS X 10.7
    if (Section.find("__DATA,__cfstring") == 0) {
      DEBUG(dbgs() << "Ignoring CFString: " << *G);
      return false;
    }
  }

  return true;
}

// This function replaces all global variables with new variables that have
// trailing redzones. It also creates a function that poisons
// redzones and inserts this function into llvm.global_ctors.
bool AddressSanitizer::insertGlobalRedzones(Module &M) {
  SmallVector<GlobalVariable *, 16> GlobalsToChange;

  for (Module::GlobalListType::iterator G = M.global_begin(),
       E = M.global_end(); G != E; ++G) {
    if (ShouldInstrumentGlobal(G))
      GlobalsToChange.push_back(G);
  }

  size_t n = GlobalsToChange.size();
  if (n == 0) return false;

  // A global is described by a structure
  //   size_t beg;
  //   size_t size;
  //   size_t size_with_redzone;
  //   const char *name;
  //   size_t has_dynamic_init;
  // We initialize an array of such structures and pass it to a run-time call.
  StructType *GlobalStructTy = StructType::get(IntptrTy, IntptrTy,
                                               IntptrTy, IntptrTy,
                                               IntptrTy, NULL);
  SmallVector<Constant *, 16> Initializers(n), DynamicInit;

  IRBuilder<> IRB(CtorInsertBefore);

  if (ClInitializers)
    FindDynamicInitializers(M);

  // The addresses of the first and last dynamically initialized globals in
  // this TU.  Used in initialization order checking.
  Value *FirstDynamic = 0, *LastDynamic = 0;

  for (size_t i = 0; i < n; i++) {
    GlobalVariable *G = GlobalsToChange[i];
    PointerType *PtrTy = cast<PointerType>(G->getType());
    Type *Ty = PtrTy->getElementType();
    uint64_t SizeInBytes = TD->getTypeAllocSize(Ty);
    uint64_t RightRedzoneSize = RedzoneSize +
        (RedzoneSize - (SizeInBytes % RedzoneSize));
    Type *RightRedZoneTy = ArrayType::get(IRB.getInt8Ty(), RightRedzoneSize);
    // Determine whether this global should be poisoned in initialization.
    bool GlobalHasDynamicInitializer = HasDynamicInitializer(G);
    // Don't check initialization order if this global is blacklisted.
    GlobalHasDynamicInitializer &= !BL->isInInit(*G);

    StructType *NewTy = StructType::get(Ty, RightRedZoneTy, NULL);
    Constant *NewInitializer = ConstantStruct::get(
        NewTy, G->getInitializer(),
        Constant::getNullValue(RightRedZoneTy), NULL);

    SmallString<2048> DescriptionOfGlobal = G->getName();
    DescriptionOfGlobal += " (";
    DescriptionOfGlobal += M.getModuleIdentifier();
    DescriptionOfGlobal += ")";
    GlobalVariable *Name = createPrivateGlobalForString(M, DescriptionOfGlobal);

    // Create a new global variable with enough space for a redzone.
    GlobalVariable *NewGlobal = new GlobalVariable(
        M, NewTy, G->isConstant(), G->getLinkage(),
        NewInitializer, "", G, G->getThreadLocalMode());
    NewGlobal->copyAttributesFrom(G);
    NewGlobal->setAlignment(RedzoneSize);

    Value *Indices2[2];
    Indices2[0] = IRB.getInt32(0);
    Indices2[1] = IRB.getInt32(0);

    G->replaceAllUsesWith(
        ConstantExpr::getGetElementPtr(NewGlobal, Indices2, true));
    NewGlobal->takeName(G);
    G->eraseFromParent();

    Initializers[i] = ConstantStruct::get(
        GlobalStructTy,
        ConstantExpr::getPointerCast(NewGlobal, IntptrTy),
        ConstantInt::get(IntptrTy, SizeInBytes),
        ConstantInt::get(IntptrTy, SizeInBytes + RightRedzoneSize),
        ConstantExpr::getPointerCast(Name, IntptrTy),
        ConstantInt::get(IntptrTy, GlobalHasDynamicInitializer),
        NULL);

    // Populate the first and last globals declared in this TU.
    if (ClInitializers && GlobalHasDynamicInitializer) {
      LastDynamic = ConstantExpr::getPointerCast(NewGlobal, IntptrTy);
      if (FirstDynamic == 0)
        FirstDynamic = LastDynamic;
    }

    DEBUG(dbgs() << "NEW GLOBAL: " << *NewGlobal << "\n");
  }

  ArrayType *ArrayOfGlobalStructTy = ArrayType::get(GlobalStructTy, n);
  GlobalVariable *AllGlobals = new GlobalVariable(
      M, ArrayOfGlobalStructTy, false, GlobalVariable::PrivateLinkage,
      ConstantArray::get(ArrayOfGlobalStructTy, Initializers), "");

  // Create calls for poisoning before initializers run and unpoisoning after.
  if (ClInitializers && FirstDynamic && LastDynamic)
    createInitializerPoisonCalls(M, FirstDynamic, LastDynamic);

  Function *AsanRegisterGlobals = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanRegisterGlobalsName, IRB.getVoidTy(),
      IntptrTy, IntptrTy, NULL));
  AsanRegisterGlobals->setLinkage(Function::ExternalLinkage);

  IRB.CreateCall2(AsanRegisterGlobals,
                  IRB.CreatePointerCast(AllGlobals, IntptrTy),
                  ConstantInt::get(IntptrTy, n));

  // We also need to unregister globals at the end, e.g. when a shared library
  // gets closed.
  Function *AsanDtorFunction = Function::Create(
      FunctionType::get(Type::getVoidTy(*C), false),
      GlobalValue::InternalLinkage, kAsanModuleDtorName, &M);
  BasicBlock *AsanDtorBB = BasicBlock::Create(*C, "", AsanDtorFunction);
  IRBuilder<> IRB_Dtor(ReturnInst::Create(*C, AsanDtorBB));
  Function *AsanUnregisterGlobals =
      checkInterfaceFunction(M.getOrInsertFunction(
          kAsanUnregisterGlobalsName,
          IRB.getVoidTy(), IntptrTy, IntptrTy, NULL));
  AsanUnregisterGlobals->setLinkage(Function::ExternalLinkage);

  IRB_Dtor.CreateCall2(AsanUnregisterGlobals,
                       IRB.CreatePointerCast(AllGlobals, IntptrTy),
                       ConstantInt::get(IntptrTy, n));
  appendToGlobalDtors(M, AsanDtorFunction, kAsanCtorAndCtorPriority);

  DEBUG(dbgs() << M);
  return true;
}

// virtual
bool AddressSanitizer::doInitialization(Module &M) {
  // Initialize the private fields. No one has accessed them before.
  TD = getAnalysisIfAvailable<DataLayout>();

  if (!TD)
    return false;
  BL.reset(new BlackList(ClBlackListFile));

  C = &(M.getContext());
  LongSize = TD->getPointerSizeInBits();
  IntptrTy = Type::getIntNTy(*C, LongSize);
  IntptrPtrTy = PointerType::get(IntptrTy, 0);

  AsanCtorFunction = Function::Create(
      FunctionType::get(Type::getVoidTy(*C), false),
      GlobalValue::InternalLinkage, kAsanModuleCtorName, &M);
  BasicBlock *AsanCtorBB = BasicBlock::Create(*C, "", AsanCtorFunction);
  CtorInsertBefore = ReturnInst::Create(*C, AsanCtorBB);

  // call __asan_init in the module ctor.
  IRBuilder<> IRB(CtorInsertBefore);
  AsanInitFunction = checkInterfaceFunction(
      M.getOrInsertFunction(kAsanInitName, IRB.getVoidTy(), NULL));
  AsanInitFunction->setLinkage(Function::ExternalLinkage);
  IRB.CreateCall(AsanInitFunction);

  // Create __asan_report* callbacks.
  for (size_t AccessIsWrite = 0; AccessIsWrite <= 1; AccessIsWrite++) {
    for (size_t AccessSizeIndex = 0; AccessSizeIndex < kNumberOfAccessSizes;
         AccessSizeIndex++) {
      // IsWrite and TypeSize are encoded in the function name.
      std::string FunctionName = std::string(kAsanReportErrorTemplate) +
          (AccessIsWrite ? "store" : "load") + itostr(1 << AccessSizeIndex);
      // If we are merging crash callbacks, they have two parameters.
      AsanErrorCallback[AccessIsWrite][AccessSizeIndex] =
          checkInterfaceFunction(M.getOrInsertFunction(
              FunctionName, IRB.getVoidTy(), IntptrTy, NULL));
    }
  }

  AsanStackMallocFunc = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanStackMallocName, IntptrTy, IntptrTy, IntptrTy, NULL));
  AsanStackFreeFunc = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanStackFreeName, IRB.getVoidTy(),
      IntptrTy, IntptrTy, IntptrTy, NULL));
  AsanHandleNoReturnFunc = checkInterfaceFunction(M.getOrInsertFunction(
      kAsanHandleNoReturnName, IRB.getVoidTy(), NULL));

  // We insert an empty inline asm after __asan_report* to avoid callback merge.
  EmptyAsm = InlineAsm::get(FunctionType::get(IRB.getVoidTy(), false),
                            StringRef(""), StringRef(""),
                            /*hasSideEffects=*/true);

  llvm::Triple targetTriple(M.getTargetTriple());
  bool isAndroid = targetTriple.getEnvironment() == llvm::Triple::Android;

  MappingOffset = isAndroid ? kDefaultShadowOffsetAndroid :
    (LongSize == 32 ? kDefaultShadowOffset32 : kDefaultShadowOffset64);
  if (ClMappingOffsetLog >= 0) {
    if (ClMappingOffsetLog == 0) {
      // special case
      MappingOffset = 0;
    } else {
      MappingOffset = 1ULL << ClMappingOffsetLog;
    }
  }
  MappingScale = kDefaultShadowScale;
  if (ClMappingScale) {
    MappingScale = ClMappingScale;
  }
  // Redzone used for stack and globals is at least 32 bytes.
  // For scales 6 and 7, the redzone has to be 64 and 128 bytes respectively.
  RedzoneSize = std::max(32, (int)(1 << MappingScale));


  if (ClMappingOffsetLog >= 0) {
    // Tell the run-time the current values of mapping offset and scale.
    GlobalValue *asan_mapping_offset =
        new GlobalVariable(M, IntptrTy, true, GlobalValue::LinkOnceODRLinkage,
                       ConstantInt::get(IntptrTy, MappingOffset),
                       kAsanMappingOffsetName);
    // Read the global, otherwise it may be optimized away.
    IRB.CreateLoad(asan_mapping_offset, true);
  }
  if (ClMappingScale) {
    GlobalValue *asan_mapping_scale =
        new GlobalVariable(M, IntptrTy, true, GlobalValue::LinkOnceODRLinkage,
                           ConstantInt::get(IntptrTy, MappingScale),
                           kAsanMappingScaleName);
    // Read the global, otherwise it may be optimized away.
    IRB.CreateLoad(asan_mapping_scale, true);
  }

  appendToGlobalCtors(M, AsanCtorFunction, kAsanCtorAndCtorPriority);

  return true;
}

bool AddressSanitizer::doFinalization(Module &M) {
  // We transform the globals at the very end so that the optimization analysis
  // works on the original globals.
  if (ClGlobals)
    return insertGlobalRedzones(M);
  return false;
}


bool AddressSanitizer::maybeInsertAsanInitAtFunctionEntry(Function &F) {
  // For each NSObject descendant having a +load method, this method is invoked
  // by the ObjC runtime before any of the static constructors is called.
  // Therefore we need to instrument such methods with a call to __asan_init
  // at the beginning in order to initialize our runtime before any access to
  // the shadow memory.
  // We cannot just ignore these methods, because they may call other
  // instrumented functions.
  if (F.getName().find(" load]") != std::string::npos) {
    IRBuilder<> IRB(F.begin()->begin());
    IRB.CreateCall(AsanInitFunction);
    return true;
  }
  return false;
}

bool AddressSanitizer::runOnFunction(Function &F) {
  if (BL->isIn(F)) return false;
  if (&F == AsanCtorFunction) return false;
  DEBUG(dbgs() << "ASAN instrumenting:\n" << F << "\n");

  // If needed, insert __asan_init before checking for AddressSafety attr.
  maybeInsertAsanInitAtFunctionEntry(F);

  if (!F.getFnAttributes().hasAttribute(Attributes::AddressSafety))
    return false;

  if (!ClDebugFunc.empty() && ClDebugFunc != F.getName())
    return false;

  // We want to instrument every address only once per basic block (unless there
  // are calls between uses).
  SmallSet<Value*, 16> TempsToInstrument;
  SmallVector<Instruction*, 16> ToInstrument;
  SmallVector<Instruction*, 8> NoReturnCalls;
  bool IsWrite;

  // Fill the set of memory operations to instrument.
  for (Function::iterator FI = F.begin(), FE = F.end();
       FI != FE; ++FI) {
    TempsToInstrument.clear();
    int NumInsnsPerBB = 0;
    for (BasicBlock::iterator BI = FI->begin(), BE = FI->end();
         BI != BE; ++BI) {
      if (LooksLikeCodeInBug11395(BI)) return false;
      if (Value *Addr = isInterestingMemoryAccess(BI, &IsWrite)) {
        if (ClOpt && ClOptSameTemp) {
          if (!TempsToInstrument.insert(Addr))
            continue;  // We've seen this temp in the current BB.
        }
      } else if (isa<MemIntrinsic>(BI) && ClMemIntrin) {
        // ok, take it.
      } else {
        if (CallInst *CI = dyn_cast<CallInst>(BI)) {
          // A call inside BB.
          TempsToInstrument.clear();
          if (CI->doesNotReturn()) {
            NoReturnCalls.push_back(CI);
          }
        }
        continue;
      }
      ToInstrument.push_back(BI);
      NumInsnsPerBB++;
      if (NumInsnsPerBB >= ClMaxInsnsToInstrumentPerBB)
        break;
    }
  }

  // Instrument.
  int NumInstrumented = 0;
  for (size_t i = 0, n = ToInstrument.size(); i != n; i++) {
    Instruction *Inst = ToInstrument[i];
    if (ClDebugMin < 0 || ClDebugMax < 0 ||
        (NumInstrumented >= ClDebugMin && NumInstrumented <= ClDebugMax)) {
      if (isInterestingMemoryAccess(Inst, &IsWrite))
        instrumentMop(Inst);
      else
        instrumentMemIntrinsic(cast<MemIntrinsic>(Inst));
    }
    NumInstrumented++;
  }

  bool ChangedStack = poisonStackInFunction(F);

  // We must unpoison the stack before every NoReturn call (throw, _exit, etc).
  // See e.g. http://code.google.com/p/address-sanitizer/issues/detail?id=37
  for (size_t i = 0, n = NoReturnCalls.size(); i != n; i++) {
    Instruction *CI = NoReturnCalls[i];
    IRBuilder<> IRB(CI);
    IRB.CreateCall(AsanHandleNoReturnFunc);
  }
  DEBUG(dbgs() << "ASAN done instrumenting:\n" << F << "\n");

  return NumInstrumented > 0 || ChangedStack || !NoReturnCalls.empty();
}

static uint64_t ValueForPoison(uint64_t PoisonByte, size_t ShadowRedzoneSize) {
  if (ShadowRedzoneSize == 1) return PoisonByte;
  if (ShadowRedzoneSize == 2) return (PoisonByte << 8) + PoisonByte;
  if (ShadowRedzoneSize == 4)
    return (PoisonByte << 24) + (PoisonByte << 16) +
        (PoisonByte << 8) + (PoisonByte);
  llvm_unreachable("ShadowRedzoneSize is either 1, 2 or 4");
}

static void PoisonShadowPartialRightRedzone(uint8_t *Shadow,
                                            size_t Size,
                                            size_t RedzoneSize,
                                            size_t ShadowGranularity,
                                            uint8_t Magic) {
  for (size_t i = 0; i < RedzoneSize;
       i+= ShadowGranularity, Shadow++) {
    if (i + ShadowGranularity <= Size) {
      *Shadow = 0;  // fully addressable
    } else if (i >= Size) {
      *Shadow = Magic;  // unaddressable
    } else {
      *Shadow = Size - i;  // first Size-i bytes are addressable
    }
  }
}

void AddressSanitizer::PoisonStack(const ArrayRef<AllocaInst*> &AllocaVec,
                                   IRBuilder<> IRB,
                                   Value *ShadowBase, bool DoPoison) {
  size_t ShadowRZSize = RedzoneSize >> MappingScale;
  assert(ShadowRZSize >= 1 && ShadowRZSize <= 4);
  Type *RZTy = Type::getIntNTy(*C, ShadowRZSize * 8);
  Type *RZPtrTy = PointerType::get(RZTy, 0);

  Value *PoisonLeft  = ConstantInt::get(RZTy,
    ValueForPoison(DoPoison ? kAsanStackLeftRedzoneMagic : 0LL, ShadowRZSize));
  Value *PoisonMid   = ConstantInt::get(RZTy,
    ValueForPoison(DoPoison ? kAsanStackMidRedzoneMagic : 0LL, ShadowRZSize));
  Value *PoisonRight = ConstantInt::get(RZTy,
    ValueForPoison(DoPoison ? kAsanStackRightRedzoneMagic : 0LL, ShadowRZSize));

  // poison the first red zone.
  IRB.CreateStore(PoisonLeft, IRB.CreateIntToPtr(ShadowBase, RZPtrTy));

  // poison all other red zones.
  uint64_t Pos = RedzoneSize;
  for (size_t i = 0, n = AllocaVec.size(); i < n; i++) {
    AllocaInst *AI = AllocaVec[i];
    uint64_t SizeInBytes = getAllocaSizeInBytes(AI);
    uint64_t AlignedSize = getAlignedAllocaSize(AI);
    assert(AlignedSize - SizeInBytes < RedzoneSize);
    Value *Ptr = NULL;

    Pos += AlignedSize;

    assert(ShadowBase->getType() == IntptrTy);
    if (SizeInBytes < AlignedSize) {
      // Poison the partial redzone at right
      Ptr = IRB.CreateAdd(
          ShadowBase, ConstantInt::get(IntptrTy,
                                       (Pos >> MappingScale) - ShadowRZSize));
      size_t AddressableBytes = RedzoneSize - (AlignedSize - SizeInBytes);
      uint32_t Poison = 0;
      if (DoPoison) {
        PoisonShadowPartialRightRedzone((uint8_t*)&Poison, AddressableBytes,
                                        RedzoneSize,
                                        1ULL << MappingScale,
                                        kAsanStackPartialRedzoneMagic);
      }
      Value *PartialPoison = ConstantInt::get(RZTy, Poison);
      IRB.CreateStore(PartialPoison, IRB.CreateIntToPtr(Ptr, RZPtrTy));
    }

    // Poison the full redzone at right.
    Ptr = IRB.CreateAdd(ShadowBase,
                        ConstantInt::get(IntptrTy, Pos >> MappingScale));
    Value *Poison = i == AllocaVec.size() - 1 ? PoisonRight : PoisonMid;
    IRB.CreateStore(Poison, IRB.CreateIntToPtr(Ptr, RZPtrTy));

    Pos += RedzoneSize;
  }
}

// Workaround for bug 11395: we don't want to instrument stack in functions
// with large assembly blobs (32-bit only), otherwise reg alloc may crash.
// FIXME: remove once the bug 11395 is fixed.
bool AddressSanitizer::LooksLikeCodeInBug11395(Instruction *I) {
  if (LongSize != 32) return false;
  CallInst *CI = dyn_cast<CallInst>(I);
  if (!CI || !CI->isInlineAsm()) return false;
  if (CI->getNumArgOperands() <= 5) return false;
  // We have inline assembly with quite a few arguments.
  return true;
}

// Find all static Alloca instructions and put
// poisoned red zones around all of them.
// Then unpoison everything back before the function returns.
//
// Stack poisoning does not play well with exception handling.
// When an exception is thrown, we essentially bypass the code
// that unpoisones the stack. This is why the run-time library has
// to intercept __cxa_throw (as well as longjmp, etc) and unpoison the entire
// stack in the interceptor. This however does not work inside the
// actual function which catches the exception. Most likely because the
// compiler hoists the load of the shadow value somewhere too high.
// This causes asan to report a non-existing bug on 453.povray.
// It sounds like an LLVM bug.
bool AddressSanitizer::poisonStackInFunction(Function &F) {
  if (!ClStack) return false;
  SmallVector<AllocaInst*, 16> AllocaVec;
  SmallVector<Instruction*, 8> RetVec;
  uint64_t TotalSize = 0;

  // Filter out Alloca instructions we want (and can) handle.
  // Collect Ret instructions.
  for (Function::iterator FI = F.begin(), FE = F.end();
       FI != FE; ++FI) {
    BasicBlock &BB = *FI;
    for (BasicBlock::iterator BI = BB.begin(), BE = BB.end();
         BI != BE; ++BI) {
      if (isa<ReturnInst>(BI)) {
          RetVec.push_back(BI);
          continue;
      }

      AllocaInst *AI = dyn_cast<AllocaInst>(BI);
      if (!AI) continue;
      if (AI->isArrayAllocation()) continue;
      if (!AI->isStaticAlloca()) continue;
      if (!AI->getAllocatedType()->isSized()) continue;
      if (AI->getAlignment() > RedzoneSize) continue;
      AllocaVec.push_back(AI);
      uint64_t AlignedSize =  getAlignedAllocaSize(AI);
      TotalSize += AlignedSize;
    }
  }

  if (AllocaVec.empty()) return false;

  uint64_t LocalStackSize = TotalSize + (AllocaVec.size() + 1) * RedzoneSize;

  bool DoStackMalloc = ClUseAfterReturn
      && LocalStackSize <= kMaxStackMallocSize;

  Instruction *InsBefore = AllocaVec[0];
  IRBuilder<> IRB(InsBefore);


  Type *ByteArrayTy = ArrayType::get(IRB.getInt8Ty(), LocalStackSize);
  AllocaInst *MyAlloca =
      new AllocaInst(ByteArrayTy, "MyAlloca", InsBefore);
  MyAlloca->setAlignment(RedzoneSize);
  assert(MyAlloca->isStaticAlloca());
  Value *OrigStackBase = IRB.CreatePointerCast(MyAlloca, IntptrTy);
  Value *LocalStackBase = OrigStackBase;

  if (DoStackMalloc) {
    LocalStackBase = IRB.CreateCall2(AsanStackMallocFunc,
        ConstantInt::get(IntptrTy, LocalStackSize), OrigStackBase);
  }

  // This string will be parsed by the run-time (DescribeStackAddress).
  SmallString<2048> StackDescriptionStorage;
  raw_svector_ostream StackDescription(StackDescriptionStorage);
  StackDescription << F.getName() << " " << AllocaVec.size() << " ";

  uint64_t Pos = RedzoneSize;
  // Replace Alloca instructions with base+offset.
  for (size_t i = 0, n = AllocaVec.size(); i < n; i++) {
    AllocaInst *AI = AllocaVec[i];
    uint64_t SizeInBytes = getAllocaSizeInBytes(AI);
    StringRef Name = AI->getName();
    StackDescription << Pos << " " << SizeInBytes << " "
                     << Name.size() << " " << Name << " ";
    uint64_t AlignedSize = getAlignedAllocaSize(AI);
    assert((AlignedSize % RedzoneSize) == 0);
    AI->replaceAllUsesWith(
        IRB.CreateIntToPtr(
            IRB.CreateAdd(LocalStackBase, ConstantInt::get(IntptrTy, Pos)),
            AI->getType()));
    Pos += AlignedSize + RedzoneSize;
  }
  assert(Pos == LocalStackSize);

  // Write the Magic value and the frame description constant to the redzone.
  Value *BasePlus0 = IRB.CreateIntToPtr(LocalStackBase, IntptrPtrTy);
  IRB.CreateStore(ConstantInt::get(IntptrTy, kCurrentStackFrameMagic),
                  BasePlus0);
  Value *BasePlus1 = IRB.CreateAdd(LocalStackBase,
                                   ConstantInt::get(IntptrTy, LongSize/8));
  BasePlus1 = IRB.CreateIntToPtr(BasePlus1, IntptrPtrTy);
  GlobalVariable *StackDescriptionGlobal =
      createPrivateGlobalForString(*F.getParent(), StackDescription.str());
  GlobalsCreatedByAsan.insert(StackDescriptionGlobal);
  Value *Description = IRB.CreatePointerCast(StackDescriptionGlobal, IntptrTy);
  IRB.CreateStore(Description, BasePlus1);

  // Poison the stack redzones at the entry.
  Value *ShadowBase = memToShadow(LocalStackBase, IRB);
  PoisonStack(ArrayRef<AllocaInst*>(AllocaVec), IRB, ShadowBase, true);

  // Unpoison the stack before all ret instructions.
  for (size_t i = 0, n = RetVec.size(); i < n; i++) {
    Instruction *Ret = RetVec[i];
    IRBuilder<> IRBRet(Ret);

    // Mark the current frame as retired.
    IRBRet.CreateStore(ConstantInt::get(IntptrTy, kRetiredStackFrameMagic),
                       BasePlus0);
    // Unpoison the stack.
    PoisonStack(ArrayRef<AllocaInst*>(AllocaVec), IRBRet, ShadowBase, false);

    if (DoStackMalloc) {
      IRBRet.CreateCall3(AsanStackFreeFunc, LocalStackBase,
                         ConstantInt::get(IntptrTy, LocalStackSize),
                         OrigStackBase);
    }
  }

  // We are done. Remove the old unused alloca instructions.
  for (size_t i = 0, n = AllocaVec.size(); i < n; i++)
    AllocaVec[i]->eraseFromParent();

  if (ClDebugStack) {
    DEBUG(dbgs() << F);
  }

  return true;
}
