//===- HeapProfiler.cpp - heap allocation and access profiler
//--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HeapProfiler. Memory accesses are instrumented
// to increment the access count held in a shadow memory location, or
// alternatively to call into the runtime. Memory intrinsic calls (memmove,
// memcpy, memset) are changed to call the heap profiling runtime version
// instead.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/HeapProfiler.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "heapprof"

constexpr int LLVM_HEAP_PROFILER_VERSION = 1;

// Size of memory mapped to a single shadow location.
constexpr uint64_t DefaultShadowGranularity = 64;

// Scale from granularity down to shadow size.
constexpr uint64_t DefaultShadowScale = 3;

constexpr char HeapProfModuleCtorName[] = "heapprof.module_ctor";
constexpr uint64_t HeapProfCtorAndDtorPriority = 1;
// On Emscripten, the system needs more than one priorities for constructors.
constexpr uint64_t HeapProfEmscriptenCtorAndDtorPriority = 50;
constexpr char HeapProfInitName[] = "__heapprof_init";
constexpr char HeapProfVersionCheckNamePrefix[] =
    "__heapprof_version_mismatch_check_v";

constexpr char HeapProfShadowMemoryDynamicAddress[] =
    "__heapprof_shadow_memory_dynamic_address";

// Command-line flags.

static cl::opt<bool> ClInsertVersionCheck(
    "heapprof-guard-against-version-mismatch",
    cl::desc("Guard against compiler/runtime version mismatch."), cl::Hidden,
    cl::init(true));

// This flag may need to be replaced with -f[no-]memprof-reads.
static cl::opt<bool> ClInstrumentReads("heapprof-instrument-reads",
                                       cl::desc("instrument read instructions"),
                                       cl::Hidden, cl::init(true));

static cl::opt<bool>
    ClInstrumentWrites("heapprof-instrument-writes",
                       cl::desc("instrument write instructions"), cl::Hidden,
                       cl::init(true));

static cl::opt<bool> ClInstrumentAtomics(
    "heapprof-instrument-atomics",
    cl::desc("instrument atomic instructions (rmw, cmpxchg)"), cl::Hidden,
    cl::init(true));

static cl::opt<bool> ClUseCalls(
    "heapprof-use-callbacks",
    cl::desc("Use callbacks instead of inline instrumentation sequences."),
    cl::Hidden, cl::init(false));

static cl::opt<std::string>
    ClMemoryAccessCallbackPrefix("heapprof-memory-access-callback-prefix",
                                 cl::desc("Prefix for memory access callbacks"),
                                 cl::Hidden, cl::init("__heapprof_"));

// These flags allow to change the shadow mapping.
// The shadow mapping looks like
//    Shadow = ((Mem & mask) >> scale) + offset

static cl::opt<int> ClMappingScale("heapprof-mapping-scale",
                                   cl::desc("scale of heapprof shadow mapping"),
                                   cl::Hidden, cl::init(DefaultShadowScale));

static cl::opt<int>
    ClMappingGranularity("heapprof-mapping-granularity",
                         cl::desc("granularity of heapprof shadow mapping"),
                         cl::Hidden, cl::init(DefaultShadowGranularity));

// Debug flags.

static cl::opt<int> ClDebug("heapprof-debug", cl::desc("debug"), cl::Hidden,
                            cl::init(0));

static cl::opt<std::string> ClDebugFunc("heapprof-debug-func", cl::Hidden,
                                        cl::desc("Debug func"));

static cl::opt<int> ClDebugMin("heapprof-debug-min", cl::desc("Debug min inst"),
                               cl::Hidden, cl::init(-1));

static cl::opt<int> ClDebugMax("heapprof-debug-max", cl::desc("Debug max inst"),
                               cl::Hidden, cl::init(-1));

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");

namespace {

/// This struct defines the shadow mapping using the rule:
///   shadow = ((mem & mask) >> Scale) ADD DynamicShadowOffset.
struct ShadowMapping {
  ShadowMapping() {
    Scale = ClMappingScale;
    Granularity = ClMappingGranularity;
    Mask = ~(Granularity - 1);
  }

  int Scale;
  int Granularity;
  uint64_t Mask; // Computed as ~(Granularity-1)
};

static uint64_t getCtorAndDtorPriority(Triple &TargetTriple) {
  return TargetTriple.isOSEmscripten() ? HeapProfEmscriptenCtorAndDtorPriority
                                       : HeapProfCtorAndDtorPriority;
}

struct InterestingMemoryAccess {
  Value *Addr = nullptr;
  bool IsWrite;
  unsigned Alignment;
  uint64_t TypeSize;
  Value *MaybeMask = nullptr;
};

/// Instrument the code in module to profile heap accesses.
class HeapProfiler {
public:
  HeapProfiler(Module &M) {
    C = &(M.getContext());
    LongSize = M.getDataLayout().getPointerSizeInBits();
    IntptrTy = Type::getIntNTy(*C, LongSize);
  }

  /// If it is an interesting memory access, populate information
  /// about the access and return a InterestingMemoryAccess struct.
  /// Otherwise return None.
  Optional<InterestingMemoryAccess> isInterestingMemoryAccess(Instruction *I);

  void instrumentMop(Instruction *I, const DataLayout &DL,
                     InterestingMemoryAccess &Access);
  void instrumentAddress(Instruction *OrigIns, Instruction *InsertBefore,
                         Value *Addr, uint32_t TypeSize, bool IsWrite);
  void instrumentMaskedLoadOrStore(const DataLayout &DL, Value *Mask,
                                   Instruction *I, Value *Addr,
                                   unsigned Alignment, uint32_t TypeSize,
                                   bool IsWrite);
  void instrumentMemIntrinsic(MemIntrinsic *MI);
  Value *memToShadow(Value *Shadow, IRBuilder<> &IRB);
  bool instrumentFunction(Function &F);
  bool maybeInsertHeapProfInitAtFunctionEntry(Function &F);
  bool insertDynamicShadowAtFunctionEntry(Function &F);

private:
  void initializeCallbacks(Module &M);

  LLVMContext *C;
  int LongSize;
  Type *IntptrTy;
  ShadowMapping Mapping;

  // These arrays is indexed by AccessIsWrite
  FunctionCallee HeapProfMemoryAccessCallback[2];
  FunctionCallee HeapProfMemoryAccessCallbackSized[2];

  FunctionCallee HeapProfMemmove, HeapProfMemcpy, HeapProfMemset;
  Value *DynamicShadowOffset = nullptr;
};

class HeapProfilerLegacyPass : public FunctionPass {
public:
  static char ID;

  explicit HeapProfilerLegacyPass() : FunctionPass(ID) {
    initializeHeapProfilerLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "HeapProfilerFunctionPass"; }

  bool runOnFunction(Function &F) override {
    HeapProfiler Profiler(*F.getParent());
    return Profiler.instrumentFunction(F);
  }
};

class ModuleHeapProfiler {
public:
  ModuleHeapProfiler(Module &M) { TargetTriple = Triple(M.getTargetTriple()); }

  bool instrumentModule(Module &);

private:
  Triple TargetTriple;
  ShadowMapping Mapping;
  Function *HeapProfCtorFunction = nullptr;
};

class ModuleHeapProfilerLegacyPass : public ModulePass {
public:
  static char ID;

  explicit ModuleHeapProfilerLegacyPass() : ModulePass(ID) {
    initializeModuleHeapProfilerLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "ModuleHeapProfiler"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {}

  bool runOnModule(Module &M) override {
    ModuleHeapProfiler HeapProfiler(M);
    return HeapProfiler.instrumentModule(M);
  }
};

} // end anonymous namespace

HeapProfilerPass::HeapProfilerPass() {}

PreservedAnalyses HeapProfilerPass::run(Function &F,
                                        AnalysisManager<Function> &AM) {
  Module &M = *F.getParent();
  HeapProfiler Profiler(M);
  if (Profiler.instrumentFunction(F))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();

  return PreservedAnalyses::all();
}

ModuleHeapProfilerPass::ModuleHeapProfilerPass() {}

PreservedAnalyses ModuleHeapProfilerPass::run(Module &M,
                                              AnalysisManager<Module> &AM) {
  ModuleHeapProfiler Profiler(M);
  if (Profiler.instrumentModule(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

char HeapProfilerLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(HeapProfilerLegacyPass, "heapprof",
                      "HeapProfiler: profile heap allocations and accesses.",
                      false, false)
INITIALIZE_PASS_END(HeapProfilerLegacyPass, "heapprof",
                    "HeapProfiler: profile heap allocations and accesses.",
                    false, false)

FunctionPass *llvm::createHeapProfilerFunctionPass() {
  return new HeapProfilerLegacyPass();
}

char ModuleHeapProfilerLegacyPass::ID = 0;

INITIALIZE_PASS(ModuleHeapProfilerLegacyPass, "heapprof-module",
                "HeapProfiler: profile heap allocations and accesses."
                "ModulePass",
                false, false)

ModulePass *llvm::createModuleHeapProfilerLegacyPassPass() {
  return new ModuleHeapProfilerLegacyPass();
}

Value *HeapProfiler::memToShadow(Value *Shadow, IRBuilder<> &IRB) {
  // (Shadow & mask) >> scale
  Shadow = IRB.CreateAnd(Shadow, Mapping.Mask);
  Shadow = IRB.CreateLShr(Shadow, Mapping.Scale);
  // (Shadow >> scale) | offset
  assert(DynamicShadowOffset);
  return IRB.CreateAdd(Shadow, DynamicShadowOffset);
}

// Instrument memset/memmove/memcpy
void HeapProfiler::instrumentMemIntrinsic(MemIntrinsic *MI) {
  IRBuilder<> IRB(MI);
  if (isa<MemTransferInst>(MI)) {
    IRB.CreateCall(
        isa<MemMoveInst>(MI) ? HeapProfMemmove : HeapProfMemcpy,
        {IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
         IRB.CreatePointerCast(MI->getOperand(1), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)});
  } else if (isa<MemSetInst>(MI)) {
    IRB.CreateCall(
        HeapProfMemset,
        {IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(MI->getOperand(1), IRB.getInt32Ty(), false),
         IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)});
  }
  MI->eraseFromParent();
}

Optional<InterestingMemoryAccess>
HeapProfiler::isInterestingMemoryAccess(Instruction *I) {
  // Do not instrument the load fetching the dynamic shadow address.
  if (DynamicShadowOffset == I)
    return None;

  InterestingMemoryAccess Access;

  const DataLayout &DL = I->getModule()->getDataLayout();
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    if (!ClInstrumentReads)
      return None;
    Access.IsWrite = false;
    Access.TypeSize = DL.getTypeStoreSizeInBits(LI->getType());
    Access.Alignment = LI->getAlignment();
    Access.Addr = LI->getPointerOperand();
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    if (!ClInstrumentWrites)
      return None;
    Access.IsWrite = true;
    Access.TypeSize =
        DL.getTypeStoreSizeInBits(SI->getValueOperand()->getType());
    Access.Alignment = SI->getAlignment();
    Access.Addr = SI->getPointerOperand();
  } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
    if (!ClInstrumentAtomics)
      return None;
    Access.IsWrite = true;
    Access.TypeSize =
        DL.getTypeStoreSizeInBits(RMW->getValOperand()->getType());
    Access.Alignment = 0;
    Access.Addr = RMW->getPointerOperand();
  } else if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(I)) {
    if (!ClInstrumentAtomics)
      return None;
    Access.IsWrite = true;
    Access.TypeSize =
        DL.getTypeStoreSizeInBits(XCHG->getCompareOperand()->getType());
    Access.Alignment = 0;
    Access.Addr = XCHG->getPointerOperand();
  } else if (auto *CI = dyn_cast<CallInst>(I)) {
    auto *F = CI->getCalledFunction();
    if (F && (F->getIntrinsicID() == Intrinsic::masked_load ||
              F->getIntrinsicID() == Intrinsic::masked_store)) {
      unsigned OpOffset = 0;
      if (F->getIntrinsicID() == Intrinsic::masked_store) {
        if (!ClInstrumentWrites)
          return None;
        // Masked store has an initial operand for the value.
        OpOffset = 1;
        Access.IsWrite = true;
      } else {
        if (!ClInstrumentReads)
          return None;
        Access.IsWrite = false;
      }

      auto *BasePtr = CI->getOperand(0 + OpOffset);
      auto *Ty = cast<PointerType>(BasePtr->getType())->getElementType();
      Access.TypeSize = DL.getTypeStoreSizeInBits(Ty);
      if (auto *AlignmentConstant =
              dyn_cast<ConstantInt>(CI->getOperand(1 + OpOffset)))
        Access.Alignment = (unsigned)AlignmentConstant->getZExtValue();
      else
        Access.Alignment = 1; // No alignment guarantees. We probably got Undef
      Access.MaybeMask = CI->getOperand(2 + OpOffset);
      Access.Addr = BasePtr;
    }
  }

  if (!Access.Addr)
    return None;

  // Do not instrument acesses from different address spaces; we cannot deal
  // with them.
  Type *PtrTy = cast<PointerType>(Access.Addr->getType()->getScalarType());
  if (PtrTy->getPointerAddressSpace() != 0)
    return None;

  // Ignore swifterror addresses.
  // swifterror memory addresses are mem2reg promoted by instruction
  // selection. As such they cannot have regular uses like an instrumentation
  // function and it makes no sense to track them as memory.
  if (Access.Addr->isSwiftError())
    return None;

  return Access;
}

void HeapProfiler::instrumentMaskedLoadOrStore(const DataLayout &DL,
                                               Value *Mask, Instruction *I,
                                               Value *Addr, unsigned Alignment,
                                               uint32_t TypeSize,
                                               bool IsWrite) {
  auto *VTy = cast<FixedVectorType>(
      cast<PointerType>(Addr->getType())->getElementType());
  uint64_t ElemTypeSize = DL.getTypeStoreSizeInBits(VTy->getScalarType());
  unsigned Num = VTy->getNumElements();
  auto *Zero = ConstantInt::get(IntptrTy, 0);
  for (unsigned Idx = 0; Idx < Num; ++Idx) {
    Value *InstrumentedAddress = nullptr;
    Instruction *InsertBefore = I;
    if (auto *Vector = dyn_cast<ConstantVector>(Mask)) {
      // dyn_cast as we might get UndefValue
      if (auto *Masked = dyn_cast<ConstantInt>(Vector->getOperand(Idx))) {
        if (Masked->isZero())
          // Mask is constant false, so no instrumentation needed.
          continue;
        // If we have a true or undef value, fall through to instrumentAddress.
        // with InsertBefore == I
      }
    } else {
      IRBuilder<> IRB(I);
      Value *MaskElem = IRB.CreateExtractElement(Mask, Idx);
      Instruction *ThenTerm = SplitBlockAndInsertIfThen(MaskElem, I, false);
      InsertBefore = ThenTerm;
    }

    IRBuilder<> IRB(InsertBefore);
    InstrumentedAddress =
        IRB.CreateGEP(VTy, Addr, {Zero, ConstantInt::get(IntptrTy, Idx)});
    instrumentAddress(I, InsertBefore, InstrumentedAddress, ElemTypeSize,
                      IsWrite);
  }
}

void HeapProfiler::instrumentMop(Instruction *I, const DataLayout &DL,
                                 InterestingMemoryAccess &Access) {
  if (Access.IsWrite)
    NumInstrumentedWrites++;
  else
    NumInstrumentedReads++;

  if (Access.MaybeMask) {
    instrumentMaskedLoadOrStore(DL, Access.MaybeMask, I, Access.Addr,
                                Access.Alignment, Access.TypeSize,
                                Access.IsWrite);
  } else {
    // Since the access counts will be accumulated across the entire allocation,
    // we only update the shadow access count for the first location and thus
    // don't need to worry about alignment and type size.
    instrumentAddress(I, I, Access.Addr, Access.TypeSize, Access.IsWrite);
  }
}

void HeapProfiler::instrumentAddress(Instruction *OrigIns,
                                     Instruction *InsertBefore, Value *Addr,
                                     uint32_t TypeSize, bool IsWrite) {
  IRBuilder<> IRB(InsertBefore);
  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);

  if (ClUseCalls) {
    IRB.CreateCall(HeapProfMemoryAccessCallback[IsWrite], AddrLong);
    return;
  }

  // Create an inline sequence to compute shadow location, and increment the
  // value by one.
  Type *ShadowTy = Type::getInt64Ty(*C);
  Type *ShadowPtrTy = PointerType::get(ShadowTy, 0);
  Value *ShadowPtr = memToShadow(AddrLong, IRB);
  Value *ShadowAddr = IRB.CreateIntToPtr(ShadowPtr, ShadowPtrTy);
  Value *ShadowValue = IRB.CreateLoad(ShadowTy, ShadowAddr);
  Value *Inc = ConstantInt::get(Type::getInt64Ty(*C), 1);
  ShadowValue = IRB.CreateAdd(ShadowValue, Inc);
  IRB.CreateStore(ShadowValue, ShadowAddr);
}

bool ModuleHeapProfiler::instrumentModule(Module &M) {
  // Create a module constructor.
  std::string HeapProfVersion = std::to_string(LLVM_HEAP_PROFILER_VERSION);
  std::string VersionCheckName =
      ClInsertVersionCheck ? (HeapProfVersionCheckNamePrefix + HeapProfVersion)
                           : "";
  std::tie(HeapProfCtorFunction, std::ignore) =
      createSanitizerCtorAndInitFunctions(M, HeapProfModuleCtorName,
                                          HeapProfInitName, /*InitArgTypes=*/{},
                                          /*InitArgs=*/{}, VersionCheckName);

  const uint64_t Priority = getCtorAndDtorPriority(TargetTriple);
  appendToGlobalCtors(M, HeapProfCtorFunction, Priority);

  return true;
}

void HeapProfiler::initializeCallbacks(Module &M) {
  IRBuilder<> IRB(*C);

  for (size_t AccessIsWrite = 0; AccessIsWrite <= 1; AccessIsWrite++) {
    const std::string TypeStr = AccessIsWrite ? "store" : "load";

    SmallVector<Type *, 3> Args2 = {IntptrTy, IntptrTy};
    SmallVector<Type *, 2> Args1{1, IntptrTy};
    HeapProfMemoryAccessCallbackSized[AccessIsWrite] =
        M.getOrInsertFunction(ClMemoryAccessCallbackPrefix + TypeStr + "N",
                              FunctionType::get(IRB.getVoidTy(), Args2, false));

    HeapProfMemoryAccessCallback[AccessIsWrite] =
        M.getOrInsertFunction(ClMemoryAccessCallbackPrefix + TypeStr,
                              FunctionType::get(IRB.getVoidTy(), Args1, false));
  }
  HeapProfMemmove = M.getOrInsertFunction(
      ClMemoryAccessCallbackPrefix + "memmove", IRB.getInt8PtrTy(),
      IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IntptrTy);
  HeapProfMemcpy = M.getOrInsertFunction(
      ClMemoryAccessCallbackPrefix + "memcpy", IRB.getInt8PtrTy(),
      IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IntptrTy);
  HeapProfMemset = M.getOrInsertFunction(
      ClMemoryAccessCallbackPrefix + "memset", IRB.getInt8PtrTy(),
      IRB.getInt8PtrTy(), IRB.getInt32Ty(), IntptrTy);
}

bool HeapProfiler::maybeInsertHeapProfInitAtFunctionEntry(Function &F) {
  // For each NSObject descendant having a +load method, this method is invoked
  // by the ObjC runtime before any of the static constructors is called.
  // Therefore we need to instrument such methods with a call to __heapprof_init
  // at the beginning in order to initialize our runtime before any access to
  // the shadow memory.
  // We cannot just ignore these methods, because they may call other
  // instrumented functions.
  if (F.getName().find(" load]") != std::string::npos) {
    FunctionCallee HeapProfInitFunction =
        declareSanitizerInitFunction(*F.getParent(), HeapProfInitName, {});
    IRBuilder<> IRB(&F.front(), F.front().begin());
    IRB.CreateCall(HeapProfInitFunction, {});
    return true;
  }
  return false;
}

bool HeapProfiler::insertDynamicShadowAtFunctionEntry(Function &F) {
  IRBuilder<> IRB(&F.front().front());
  Value *GlobalDynamicAddress = F.getParent()->getOrInsertGlobal(
      HeapProfShadowMemoryDynamicAddress, IntptrTy);
  DynamicShadowOffset = IRB.CreateLoad(IntptrTy, GlobalDynamicAddress);
  return true;
}

bool HeapProfiler::instrumentFunction(Function &F) {
  if (F.getLinkage() == GlobalValue::AvailableExternallyLinkage)
    return false;
  if (ClDebugFunc == F.getName())
    return false;
  if (F.getName().startswith("__heapprof_"))
    return false;

  bool FunctionModified = false;

  // If needed, insert __heapprof_init.
  // This function needs to be called even if the function body is not
  // instrumented.
  if (maybeInsertHeapProfInitAtFunctionEntry(F))
    FunctionModified = true;

  LLVM_DEBUG(dbgs() << "HEAPPROF instrumenting:\n" << F << "\n");

  initializeCallbacks(*F.getParent());

  FunctionModified |= insertDynamicShadowAtFunctionEntry(F);

  SmallVector<Instruction *, 16> ToInstrument;

  // Fill the set of memory operations to instrument.
  for (auto &BB : F) {
    for (auto &Inst : BB) {
      if (isInterestingMemoryAccess(&Inst) || isa<MemIntrinsic>(Inst))
        ToInstrument.push_back(&Inst);
    }
  }

  int NumInstrumented = 0;
  for (auto *Inst : ToInstrument) {
    if (ClDebugMin < 0 || ClDebugMax < 0 ||
        (NumInstrumented >= ClDebugMin && NumInstrumented <= ClDebugMax)) {
      Optional<InterestingMemoryAccess> Access =
          isInterestingMemoryAccess(Inst);
      if (Access)
        instrumentMop(Inst, F.getParent()->getDataLayout(), *Access);
      else
        instrumentMemIntrinsic(cast<MemIntrinsic>(Inst));
    }
    NumInstrumented++;
  }

  if (NumInstrumented > 0)
    FunctionModified = true;

  LLVM_DEBUG(dbgs() << "HEAPPROF done instrumenting: " << FunctionModified
                    << " " << F << "\n");

  return FunctionModified;
}
