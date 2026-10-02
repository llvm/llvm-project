//===- ConcurrencySanitizer.cpp - watchpoint race detector ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements instrumentation for ConcurrencySanitizer, a sampling
// data-race detector. Memory accesses are preceded by runtime probes while the
// original operations remain unchanged.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/ConcurrencySanitizer.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/bit.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/NVPTXAddrSpace.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Instrumentation.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "csan"

static cl::opt<bool> ClInstrumentMemoryAccesses(
    "csan-instrument-memory-accesses", cl::init(true),
    cl::desc("Instrument memory accesses"), cl::Hidden);
static cl::opt<bool>
    ClInstrumentFuncEntryExit("csan-instrument-func-entry-exit", cl::init(true),
                              cl::desc("Instrument function entry and exit"),
                              cl::Hidden);
static cl::opt<bool> ClHandleCxxExceptions(
    "csan-handle-cxx-exceptions", cl::init(true),
    cl::desc("Handle C++ exceptions (insert cleanup blocks for unwinding)"),
    cl::Hidden);
static cl::opt<bool> ClInstrumentAtomics("csan-instrument-atomics",
                                         cl::init(true),
                                         cl::desc("Instrument atomics"),
                                         cl::Hidden);
static cl::opt<bool> ClInstrumentMemIntrinsics(
    "csan-instrument-memintrinsics", cl::init(true),
    cl::desc("Instrument memintrinsics (memset/memcpy/memmove)"), cl::Hidden);
static cl::opt<bool> ClDistinguishVolatile(
    "csan-distinguish-volatile", cl::init(false),
    cl::desc("Treat aligned volatile accesses up to 8 bytes as atomic"),
    cl::Hidden);
STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumOmittedReadsBeforeWrite,
          "Number of reads ignored due to following writes");

static constexpr char kCsanModuleCtorName[] = "csan.module_ctor";
static constexpr char kCsanInitName[] = "__csan_init";

// Accesses sizes are powers of two: 1, 2, 4, 8, 16.
static constexpr size_t kNumAccessSizes = 5;
static constexpr StringLiteral kCsanReadNames[kNumAccessSizes] = {
    "__csan_read1", "__csan_read2", "__csan_read4", "__csan_read8",
    "__csan_read16"};
static constexpr StringLiteral kCsanWriteNames[kNumAccessSizes] = {
    "__csan_write1", "__csan_write2", "__csan_write4", "__csan_write8",
    "__csan_write16"};
static constexpr StringLiteral kCsanUnalignedReadNames[kNumAccessSizes] = {
    "__csan_unaligned_read1", "__csan_unaligned_read2",
    "__csan_unaligned_read4", "__csan_unaligned_read8",
    "__csan_unaligned_read16"};
static constexpr StringLiteral kCsanUnalignedWriteNames[kNumAccessSizes] = {
    "__csan_unaligned_write1", "__csan_unaligned_write2",
    "__csan_unaligned_write4", "__csan_unaligned_write8",
    "__csan_unaligned_write16"};
static constexpr StringLiteral kCsanCompoundRWNames[kNumAccessSizes] = {
    "__csan_read_write1", "__csan_read_write2", "__csan_read_write4",
    "__csan_read_write8", "__csan_read_write16"};
static constexpr StringLiteral kCsanUnalignedCompoundRWNames[kNumAccessSizes] =
    {"__csan_unaligned_read_write1", "__csan_unaligned_read_write2",
     "__csan_unaligned_read_write4", "__csan_unaligned_read_write8",
     "__csan_unaligned_read_write16"};

namespace {

/// Must match CSAN_ACCESS_* in compiler-rt/lib/csan/csan_defs.h.
enum class AccessFlags : unsigned {
  None = 0,
  Atomic = 1u << 0,
  Compound = 1u << 1,
  LLVM_MARK_AS_BITMASK_ENUM(Compound),
};

static bool isAtomicMemoryAccess(const Instruction *I) {
  auto SSID = getAtomicSyncScopeID(I);
  if (!SSID)
    return false;
  if (isa<LoadInst>(I) || isa<StoreInst>(I))
    return *SSID != SyncScope::SingleThread;
  return true;
}

static Value *getCallbackAddress(IRBuilderBase &IRB, Value *Addr) {
  return IRB.CreateAddrSpaceCast(Addr, IRB.getPtrTy());
}

static ConstantInt *createOrdering(IRBuilderBase &IRB, AtomicOrdering Ord) {
  uint32_t Value = 0;
  switch (Ord) {
  case AtomicOrdering::NotAtomic:
    llvm_unreachable("unexpected atomic ordering");
  case AtomicOrdering::Unordered:
  case AtomicOrdering::Monotonic:
    Value = 0;
    break;
  case AtomicOrdering::Acquire:
    Value = 2;
    break;
  case AtomicOrdering::Release:
    Value = 3;
    break;
  case AtomicOrdering::AcquireRelease:
    Value = 4;
    break;
  case AtomicOrdering::SequentiallyConsistent:
    Value = 5;
    break;
  }
  return IRB.getInt32(Value);
}

static int getAccessSizeIndex(Type *Ty, const DataLayout &DL) {
  assert(Ty->isSized());
  if (Ty->isScalableTy())
    return -1;
  uint32_t TypeSize = DL.getTypeStoreSizeInBits(Ty);
  if (TypeSize != 8 && TypeSize != 16 && TypeSize != 32 && TypeSize != 64 &&
      TypeSize != 128)
    return -1;
  unsigned Idx = llvm::countr_zero(TypeSize / 8);
  return static_cast<int>(Idx);
}

static bool addressSpaceMayRace(const Triple &T, unsigned AS) {
  if (T.isAMDGPU()) {
    // GDS and buffer fat pointers cannot form a generic watchpoint key.
    return AS == AMDGPUAS::FLAT_ADDRESS || AS == AMDGPUAS::GLOBAL_ADDRESS ||
           AS == AMDGPUAS::LOCAL_ADDRESS;
  }
  if (T.isNVPTX()) {
    return AS == NVPTXAS::ADDRESS_SPACE_GENERIC ||
           AS == NVPTXAS::ADDRESS_SPACE_GLOBAL ||
           AS == NVPTXAS::ADDRESS_SPACE_SHARED ||
           AS == NVPTXAS::ADDRESS_SPACE_SHARED_CLUSTER;
  }
  if (T.isSPIRV()) {
    // FIXME: No exposed address spaces for SPIR-V.
    return false;
  }
  return AS == 0;
}

struct ConcurrencySanitizer {
  bool sanitizeFunction(Function &F, const TargetLibraryInfo &TLI);

private:
  struct MemoryAccessLists {
    SmallVector<Instruction *, 8> LoadsAndStores;
    SmallVector<Instruction *, 8> AtomicAccesses;
    SmallVector<IntrinsicInst *, 8> MemIntrinCalls;
    bool HasCalls = false;
  };

  void initialize(Module &M, const TargetLibraryInfo &TLI);
  void collectMemoryAccesses(Function &F, MemoryAccessLists &Out);
  bool instrumentLoadOrStore(Instruction *I, const DataLayout &DL);
  bool instrumentAtomic(Instruction *I, const DataLayout &DL);
  bool instrumentMemIntrinsic(IntrinsicInst *II, const DataLayout &DL);
  bool insertAccessProbe(Instruction *I, Value *Addr, Type *AccessTy,
                         const DataLayout &DL, bool IsWrite, bool IsCompound,
                         bool IsAtomic);
  bool shouldInstrumentAddress(Value *Addr) const;
  bool shouldInstrumentAccess(Instruction *I) const;
  void insertFuncEntryExit(Function &F);
  void insertRuntimeIgnores(Function &F);
  Constant *getFlags(AccessFlags Flags) const {
    return ConstantInt::get(FlagsTy, to_underlying(Flags));
  }

  Module *Mod = nullptr;
  Type *IntptrTy = nullptr;
  IntegerType *FlagsTy = nullptr;

  // void __csan_readN(ptr, i32);
  FunctionCallee CsanRead[kNumAccessSizes];
  // void __csan_writeN(ptr, i32);
  FunctionCallee CsanWrite[kNumAccessSizes];
  // void __csan_unaligned_readN(ptr, i32);
  FunctionCallee CsanUnalignedRead[kNumAccessSizes];
  // void __csan_unaligned_writeN(ptr, i32);
  FunctionCallee CsanUnalignedWrite[kNumAccessSizes];
  // void __csan_read_writeN(ptr, i32);
  FunctionCallee CsanCompoundRW[kNumAccessSizes];
  // void __csan_unaligned_read_writeN(ptr, i32);
  FunctionCallee CsanUnalignedCompoundRW[kNumAccessSizes];
  // void __csan_func_entry(ptr);
  FunctionCallee CsanFuncEntry;
  // void __csan_func_exit();
  FunctionCallee CsanFuncExit;
  // void __csan_ignore_thread_begin();
  FunctionCallee CsanIgnoreBegin;
  // void __csan_ignore_thread_end();
  FunctionCallee CsanIgnoreEnd;
  // void __csan_read_range(ptr, intptr_t, i32);
  FunctionCallee CsanReadRange;
  // void __csan_write_range(ptr, intptr_t, i32);
  FunctionCallee CsanWriteRange;
  // void __csan_atomic_thread_fence(i32);
  FunctionCallee CsanAtomicThreadFence;
  // void __csan_atomic_signal_fence(i32);
  FunctionCallee CsanAtomicSignalFence;
};

void insertModuleCtor(Module &M) {
  getOrCreateSanitizerCtorAndInitFunctions(
      M, kCsanModuleCtorName, kCsanInitName, /*InitArgTypes=*/{},
      /*InitArgs=*/{},
      [&](Function *Ctor, FunctionCallee) { appendToGlobalCtors(M, Ctor, 0); });
}

} // namespace

PreservedAnalyses ConcurrencySanitizerPass::run(Function &F,
                                                FunctionAnalysisManager &FAM) {
  ConcurrencySanitizer CSan;
  if (CSan.sanitizeFunction(F, FAM.getResult<TargetLibraryAnalysis>(F)))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

PreservedAnalyses ModuleConcurrencySanitizerPass::run(Module &M,
                                                      ModuleAnalysisManager &) {
  if (checkIfAlreadyInstrumented(M, "nosanitize_concurrency"))
    return PreservedAnalyses::all();
  insertModuleCtor(M);
  return PreservedAnalyses::none();
}

void ConcurrencySanitizer::initialize(Module &M, const TargetLibraryInfo &TLI) {
  LLVMContext &Ctx = M.getContext();
  Mod = &M;
  IntptrTy = M.getDataLayout().getIntPtrType(Ctx);
  FlagsTy = Type::getInt32Ty(Ctx);

  AttributeList Attr = AttributeList().addFnAttribute(Ctx, Attribute::NoUnwind);
  IRBuilder<> IRB(Ctx);
  Type *VoidTy = IRB.getVoidTy();
  Type *PtrTy = IRB.getPtrTy();
  CsanFuncEntry =
      M.getOrInsertFunction("__csan_func_entry", Attr, VoidTy, PtrTy);
  CsanFuncExit = M.getOrInsertFunction("__csan_func_exit", Attr, VoidTy);
  CsanIgnoreBegin =
      M.getOrInsertFunction("__csan_ignore_thread_begin", Attr, VoidTy);
  CsanIgnoreEnd =
      M.getOrInsertFunction("__csan_ignore_thread_end", Attr, VoidTy);
  auto AccessFn = [&](StringRef Name) {
    return M.getOrInsertFunction(Name, Attr, VoidTy, PtrTy, FlagsTy);
  };
  for (unsigned I = 0; I < kNumAccessSizes; ++I) {
    CsanRead[I] = AccessFn(kCsanReadNames[I]);
    CsanWrite[I] = AccessFn(kCsanWriteNames[I]);
    CsanUnalignedRead[I] = AccessFn(kCsanUnalignedReadNames[I]);
    CsanUnalignedWrite[I] = AccessFn(kCsanUnalignedWriteNames[I]);
    CsanCompoundRW[I] = AccessFn(kCsanCompoundRWNames[I]);
    CsanUnalignedCompoundRW[I] = AccessFn(kCsanUnalignedCompoundRWNames[I]);
  }
  IntegerType *OrdTy = IRB.getInt32Ty();
  CsanReadRange = M.getOrInsertFunction("__csan_read_range", Attr, VoidTy,
                                        PtrTy, IntptrTy, FlagsTy);
  CsanWriteRange = M.getOrInsertFunction("__csan_write_range", Attr, VoidTy,
                                         PtrTy, IntptrTy, FlagsTy);
  CsanAtomicThreadFence = M.getOrInsertFunction(
      "__csan_atomic_thread_fence",
      TLI.getAttrList(&Ctx, {0}, /*Signed=*/true, /*Ret=*/false, Attr), VoidTy,
      OrdTy);
  CsanAtomicSignalFence = M.getOrInsertFunction(
      "__csan_atomic_signal_fence",
      TLI.getAttrList(&Ctx, {0}, /*Signed=*/true, /*Ret=*/false, Attr), VoidTy,
      OrdTy);
}

bool ConcurrencySanitizer::sanitizeFunction(Function &F,
                                            const TargetLibraryInfo &TLI) {
  // This is required to prevent instrumenting call to __csan_init from within
  // the module constructor.
  if (F.getName() == kCsanModuleCtorName)
    return false;
  // Naked functions can not have prologue/epilogue
  // (__csan_func_entry/__csan_func_exit) generated, so don't instrument them at
  // all.
  if (F.hasFnAttribute(Attribute::Naked))
    return false;

  // __attribute__(disable_sanitizer_instrumentation) prevents all kinds of
  // instrumentation.
  if (F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation))
    return false;

  Mod = F.getParent();
  MemoryAccessLists Acc;
  collectMemoryAccesses(F, Acc);

  const bool SuppressChecking =
      F.hasFnAttribute("sanitize_concurrency_no_checking_at_run_time");
  const bool SanitizeFunction =
      F.hasFnAttribute(Attribute::SanitizeConcurrency) && !SuppressChecking;
  const bool MayInstrument =
      SanitizeFunction &&
      ((ClInstrumentMemoryAccesses && !Acc.LoadsAndStores.empty()) ||
       (ClInstrumentAtomics && !Acc.AtomicAccesses.empty()) ||
       (ClInstrumentMemIntrinsics && !Acc.MemIntrinCalls.empty()));
  const bool NeedsRuntimeIgnores = SuppressChecking && Acc.HasCalls;
  const bool NeedsFuncEntryExit = ClInstrumentFuncEntryExit && Acc.HasCalls;
  if (!MayInstrument && !NeedsRuntimeIgnores && !NeedsFuncEntryExit)
    return false;

  initialize(*Mod, TLI);
  bool Res = false;
  const DataLayout &DL = F.getDataLayout();

  if (ClInstrumentMemoryAccesses && SanitizeFunction)
    for (Instruction *I : Acc.LoadsAndStores)
      Res |= instrumentLoadOrStore(I, DL);

  if (ClInstrumentAtomics && SanitizeFunction)
    for (Instruction *I : Acc.AtomicAccesses)
      Res |= instrumentAtomic(I, DL);

  if (ClInstrumentMemIntrinsics && SanitizeFunction)
    for (IntrinsicInst *II : Acc.MemIntrinCalls)
      Res |= instrumentMemIntrinsic(II, DL);

  if (NeedsRuntimeIgnores) {
    insertRuntimeIgnores(F);
    Res = true;
  }

  if ((Res || Acc.HasCalls) && ClInstrumentFuncEntryExit) {
    insertFuncEntryExit(F);
    Res = true;
  }
  return Res;
}

bool ConcurrencySanitizer::shouldInstrumentAddress(Value *Addr) const {
  Value *BaseAddr = Addr->stripInBoundsOffsets();
  if (auto *GV = dyn_cast<GlobalVariable>(BaseAddr)) {
    if (GV->hasSection()) {
      StringRef SectionName = GV->getSection();
      auto OF = Mod->getTargetTriple().getObjectFormat();
      if (SectionName.ends_with(
              getInstrProfSectionName(IPSK_cnts, OF, /*AddSegmentInfo=*/false)))
        return false;
    }
  }

  Type *PtrTy = cast<PointerType>(Addr->getType()->getScalarType());
  unsigned AS = PtrTy->getPointerAddressSpace();
  if (Mod->getDataLayout().getPointerSizeInBits(AS) > 64)
    return false;
  return addressSpaceMayRace(Mod->getTargetTriple(), AS);
}

bool ConcurrencySanitizer::shouldInstrumentAccess(Instruction *I) const {
  const bool IsWrite = isa<StoreInst>(I);
  Value *Addr = getLoadStorePointerOperand(I);
  if (!shouldInstrumentAddress(Addr))
    return false;

  if (!IsWrite)
    if (auto *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(Addr)))
      if (GV->isConstant())
        return false;

  const AllocaInst *AI = findAllocaForValue(Addr);
  if (!AI || PointerMayBeCaptured(AI, /*ReturnCaptures=*/true))
    return true;

  const Triple &T = AI->getModule()->getTargetTriple();
  return T.isGPU() && addressSpaceMayRace(T, AI->getAddressSpace());
}

void ConcurrencySanitizer::collectMemoryAccesses(Function &F,
                                                 MemoryAccessLists &Out) {
  SmallVector<Instruction *, 8> LocalLoadsAndStores;
  auto FlushLocalAccesses = [&] {
    DenseSet<Value *> WriteTargets;
    for (Instruction *Inst : reverse(LocalLoadsAndStores)) {
      const bool IsWrite = isa<StoreInst>(Inst);
      Value *Addr = getLoadStorePointerOperand(Inst);
      if (!IsWrite && WriteTargets.contains(Addr)) {
        ++NumOmittedReadsBeforeWrite;
        continue;
      }

      Out.LoadsAndStores.push_back(Inst);
      if (IsWrite)
        WriteTargets.insert(Addr);
    }
    LocalLoadsAndStores.clear();
  };

  for (BasicBlock &BB : F) {
    for (Instruction &Inst : BB) {
      // Skip instructions inserted by another instrumentation.
      if (Inst.hasMetadata(LLVMContext::MD_nosanitize))
        continue;
      if (isAtomicMemoryAccess(&Inst))
        Out.AtomicAccesses.push_back(&Inst);
      else if ((isa<LoadInst>(Inst) || isa<StoreInst>(Inst)) &&
               shouldInstrumentAccess(&Inst))
        LocalLoadsAndStores.push_back(&Inst);
      else if (isa<CallInst>(Inst) || isa<InvokeInst>(Inst)) {
        FlushLocalAccesses();
        if (isa<AnyMemIntrinsic, MemSetPatternInst>(Inst))
          Out.MemIntrinCalls.push_back(cast<IntrinsicInst>(&Inst));
        Out.HasCalls = true;
      }
    }
    FlushLocalAccesses();
  }
}

bool ConcurrencySanitizer::instrumentLoadOrStore(Instruction *I,
                                                 const DataLayout &DL) {
  const bool IsWrite = isa<StoreInst>(I);
  Value *Addr = getLoadStorePointerOperand(I);
  if (!insertAccessProbe(I, Addr, getLoadStoreType(I), DL, IsWrite,
                         /*IsCompound=*/false, /*IsAtomic=*/false))
    return false;
  if (IsWrite)
    ++NumInstrumentedWrites;
  else
    ++NumInstrumentedReads;
  return true;
}

bool ConcurrencySanitizer::instrumentAtomic(Instruction *I,
                                            const DataLayout &DL) {
  if (auto *FI = dyn_cast<FenceInst>(I)) {
    InstrumentationIRBuilder IRB(I);
    FunctionCallee Fn = FI->getSyncScopeID() == SyncScope::SingleThread
                            ? CsanAtomicSignalFence
                            : CsanAtomicThreadFence;
    IRB.CreateCall(Fn, createOrdering(IRB, FI->getOrdering()));
    return true;
  }

  Value *Addr = nullptr;
  Type *AccessTy = nullptr;
  bool IsWrite = true;
  bool IsCompound = false;
  if (auto *LI = dyn_cast<LoadInst>(I)) {
    Addr = LI->getPointerOperand();
    AccessTy = LI->getType();
    IsWrite = false;
  } else if (auto *SI = dyn_cast<StoreInst>(I)) {
    Addr = SI->getPointerOperand();
    AccessTy = SI->getValueOperand()->getType();
  } else if (auto *RMW = dyn_cast<AtomicRMWInst>(I)) {
    Addr = RMW->getPointerOperand();
    AccessTy = RMW->getValOperand()->getType();
    IsCompound = true;
  } else if (auto *CAS = dyn_cast<AtomicCmpXchgInst>(I)) {
    // A failed CAS is still considered a write as the output is unkonwn.
    Addr = CAS->getPointerOperand();
    AccessTy = CAS->getNewValOperand()->getType();
    IsCompound = true;
  } else {
    return false;
  }

  if (!shouldInstrumentAddress(Addr))
    return false;
  if (!insertAccessProbe(I, Addr, AccessTy, DL, IsWrite, IsCompound,
                         /*IsAtomic=*/true))
    return false;
  if (IsCompound || IsWrite)
    ++NumInstrumentedWrites;
  if (IsCompound || !IsWrite)
    ++NumInstrumentedReads;
  return true;
}

bool ConcurrencySanitizer::instrumentMemIntrinsic(IntrinsicInst *II,
                                                  const DataLayout &DL) {
  auto *MI = dyn_cast<AnyMemIntrinsic>(II);
  auto *MSP = dyn_cast<MemSetPatternInst>(II);
  Value *Dst = MI ? MI->getRawDest() : MSP->getRawDest();
  Value *Src = nullptr;
  if (auto *MT = dyn_cast<AnyMemTransferInst>(II))
    Src = MT->getRawSource();

  bool InstrumentRead = Src && shouldInstrumentAddress(Src);
  bool InstrumentWrite = shouldInstrumentAddress(Dst);
  if (!InstrumentRead && !InstrumentWrite)
    return false;

  InstrumentationIRBuilder IRB(II);
  Value *Len = IRB.CreateIntCast(MI ? MI->getLength() : MSP->getLength(),
                                 IntptrTy, false);
  // The length of a pattern memset counts patterns, not bytes.
  if (MSP) {
    TypeSize Size = DL.getTypeAllocSize(MSP->getValue()->getType());
    Len = IRB.CreateMul(Len, IRB.CreateTypeSize(IntptrTy, Size));
  }
  Constant *Flags =
      getFlags(MI && MI->isAtomic() ? AccessFlags::Atomic : AccessFlags::None);
  if (InstrumentRead) {
    IRB.CreateCall(CsanReadRange, {getCallbackAddress(IRB, Src), Len, Flags});
    ++NumInstrumentedReads;
  }
  if (InstrumentWrite) {
    IRB.CreateCall(CsanWriteRange, {getCallbackAddress(IRB, Dst), Len, Flags});
    ++NumInstrumentedWrites;
  }
  return true;
}

bool ConcurrencySanitizer::insertAccessProbe(Instruction *I, Value *Addr,
                                             Type *AccessTy,
                                             const DataLayout &DL, bool IsWrite,
                                             bool IsCompound, bool IsAtomic) {
  if (Addr->isSwiftError())
    return false;
  AccessFlags Flags = AccessFlags::None;
  if (IsAtomic)
    Flags |= AccessFlags::Atomic;
  if (IsCompound)
    Flags |= AccessFlags::Compound;
  int Idx = getAccessSizeIndex(AccessTy, DL);
  if (Idx < 0) {
    if (IsCompound)
      return false;
    InstrumentationIRBuilder IRB(I);
    Value *Len = IRB.CreateTypeSize(IntptrTy, DL.getTypeStoreSize(AccessTy));
    IRB.CreateCall(IsWrite ? CsanWriteRange : CsanReadRange,
                   {getCallbackAddress(IRB, Addr), Len, getFlags(Flags)});
    return true;
  }

  Align Alignment = Align(1);
  bool IsVolatile = false;
  if (auto *LI = dyn_cast<LoadInst>(I)) {
    Alignment = LI->getAlign();
    IsVolatile = LI->isVolatile();
  } else if (auto *SI = dyn_cast<StoreInst>(I)) {
    Alignment = SI->getAlign();
    IsVolatile = SI->isVolatile();
  } else if (auto *RMW = dyn_cast<AtomicRMWInst>(I)) {
    Alignment = RMW->getAlign();
    IsVolatile = RMW->isVolatile();
  } else if (auto *CAS = dyn_cast<AtomicCmpXchgInst>(I)) {
    Alignment = CAS->getAlign();
    IsVolatile = CAS->isVolatile();
  }

  uint32_t TypeSize = DL.getTypeStoreSizeInBits(AccessTy);
  bool Unaligned =
      Alignment < Align(8) && Alignment.value() % (TypeSize / 8) != 0;
  FunctionCallee Callback;
  if (IsCompound)
    Callback = Unaligned ? CsanUnalignedCompoundRW[Idx] : CsanCompoundRW[Idx];
  else if (Unaligned)
    Callback = IsWrite ? CsanUnalignedWrite[Idx] : CsanUnalignedRead[Idx];
  else
    Callback = IsWrite ? CsanWrite[Idx] : CsanRead[Idx];

  InstrumentationIRBuilder IRB(I);
  Value *CallbackAddr = getCallbackAddress(IRB, Addr);
  Value *FlagsVal = getFlags(Flags);
  if (ClDistinguishVolatile && IsVolatile && TypeSize <= 64) {
    // The declared alignment may understate the pointer's real alignment.
    Value *IsAligned = IRB.getTrue();
    if (Unaligned)
      IsAligned = IRB.CreateIsNull(IRB.CreateAnd(
          IRB.CreatePtrToInt(CallbackAddr, IntptrTy), TypeSize / 8 - 1));
    FlagsVal = IRB.CreateSelect(
        IsAligned, getFlags(Flags | AccessFlags::Atomic), FlagsVal);
  }
  IRB.CreateCall(Callback, {CallbackAddr, FlagsVal});
  return true;
}

void ConcurrencySanitizer::insertFuncEntryExit(Function &F) {
  const DataLayout &DL = F.getDataLayout();
  InstrumentationIRBuilder IRB(&F.getEntryBlock(),
                               F.getEntryBlock().getFirstNonPHIIt());
  Type *ProgramAsPtrTy =
      PointerType::get(F.getContext(), DL.getProgramAddressSpace());
  Value *ReturnAddress = IRB.CreateIntrinsic(Intrinsic::returnaddress,
                                             {ProgramAsPtrTy}, IRB.getInt32(0));
  IRB.CreateCall(CsanFuncEntry, ReturnAddress);

  EscapeEnumerator EE(F, "csan_cleanup", ClHandleCxxExceptions);
  while (IRBuilder<> *AtExit = EE.Next()) {
    InstrumentationIRBuilder::ensureDebugInfo(*AtExit, F);
    AtExit->CreateCall(CsanFuncExit);
  }
}

void ConcurrencySanitizer::insertRuntimeIgnores(Function &F) {
  InstrumentationIRBuilder IRB(&F.getEntryBlock(),
                               F.getEntryBlock().getFirstNonPHIIt());
  IRB.CreateCall(CsanIgnoreBegin);
  EscapeEnumerator EE(F, "csan_ignore_cleanup", ClHandleCxxExceptions);
  while (IRBuilder<> *AtExit = EE.Next()) {
    InstrumentationIRBuilder::ensureDebugInfo(*AtExit, F);
    AtExit->CreateCall(CsanIgnoreEnd);
  }
}
