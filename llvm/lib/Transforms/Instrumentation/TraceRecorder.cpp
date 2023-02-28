//===-- TraceRecorder.cpp - race detector -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder, a race detector.
//
// The tool is under development, for the details about previous versions see
// http://code.google.com/p/data-race-test
//
// The instrumentation phase is quite simple:
//   - Insert calls to run-time library before every memory access.
//      - Optimizations may apply to avoid instrumenting some of the accesses.
//   - Insert calls at function entry/exit.
// The rest is handled by the run-time library.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/TraceRecorder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "trec"

static cl::opt<bool> ClInstrumentMemoryAccesses(
    "trec-instrument-memory-accesses", cl::init(true),
    cl::desc("Instrument memory accesses"), cl::Hidden);
static cl::opt<bool>
    ClInstrumentMemoryReads("trec-instrument-memory-read", cl::init(true),
                            cl::desc("Instrument memory reads"), cl::Hidden);
static cl::opt<bool>
    ClInstrumentMemoryWrites("trec-instrument-memory-write", cl::init(true),
                             cl::desc("Instrument memory writes"), cl::Hidden);
static cl::opt<bool>
    ClInstrumentFuncEntryExit("trec-instrument-func-entry-exit", cl::init(true),
                              cl::desc("Instrument function entry and exit"),
                              cl::Hidden);
static cl::opt<bool> ClInstrumentAtomics("trec-instrument-atomics",
                                         cl::init(true),
                                         cl::desc("Instrument atomics"),
                                         cl::Hidden);
static cl::opt<bool> ClInstrumentMemIntrinsics(
    "trec-instrument-memintrinsics", cl::init(true),
    cl::desc("Instrument memintrinsics (memset/memcpy/memmove)"), cl::Hidden);
static cl::opt<bool> ClInstrumentBranch(
    "trec-instrument-branch", cl::init(true),
    cl::desc("Instrument branch points (indirectcalls/invoke calls/conditional "
             "branches/switches)"),
    cl::Hidden);
static cl::opt<bool>
    ClInstrumentFuncParam("trec-instrument-function-parameters", cl::init(true),
                          cl::desc("Instrument function parameters"),
                          cl::Hidden);
static cl::opt<bool>
    ClTrecAddDebugInfo("trec-add-debug-info", cl::init(true),
                       cl::desc("Instrument to record debug information"),
                       cl::Hidden);

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumOmittedReadsBeforeWrite,
          "Number of reads ignored due to following writes");
STATISTIC(NumAccessesWithBadSize, "Number of accesses with bad size");
STATISTIC(NumInstrumentedVtableWrites, "Number of vtable ptr writes");
STATISTIC(NumInstrumentedVtableReads, "Number of vtable ptr reads");
STATISTIC(NumOmittedReadsFromConstantGlobals,
          "Number of reads from constant globals");
STATISTIC(NumOmittedReadsFromVtable, "Number of vtable reads");
STATISTIC(NumOmittedNonCaptured, "Number of accesses ignored due to capturing");

const char kTrecModuleCtorName[] = "trec.module_ctor";
const char kTrecInitName[] = "__trec_init";

namespace {
std::map<std::string, Value *> TraceRecorderModuleVarNames;
static enum Mode { Eagle, Verification, Unknown } mode;
/// TraceRecorder: instrument the code in module to record traces.
///
/// Instantiating TraceRecorder inserts the trec runtime library API
/// function declarations into the module if they don't exist already.
/// Instantiating ensures the __trec_init function is in the list of global
/// constructors for the module.
struct TraceRecorder {
  TraceRecorder(std::map<std::string, Value *> &VN) : VarNames(VN) {
    // Sanity check options and warn user.
    if (getenv("TREC_COMPILE_MODE") == nullptr)
      mode = Mode::Unknown;
    else if (strcmp(getenv("TREC_COMPILE_MODE"), "eagle") == 0)
      mode = Mode::Eagle;
    else if (strcmp(getenv("TREC_COMPILE_MODE"), "verification") == 0)
      mode = Mode::Verification;
    else
      mode = Mode::Unknown;
    if (mode == Mode::Unknown) {
      printf("Error: Unknown TraceRecorder mode: ENV variable "
             "`TREC_COMPILE_MODE` has "
             "not been set!\n");
      exit(-1);
    }
  }

  bool sanitizeFunction(Function &F, const TargetLibraryInfo &TLI);
  std::map<std::string, Value *> &VarNames;

private:
  SmallDenseMap<Instruction *, unsigned int> FuncCallOrders;
  unsigned int FuncCallOrderCounter;

  // Internal Instruction wrapper that contains more information about the
  // Instruction from prior analysis.
  struct InstructionInfo {
    // Instrumentation emitted for this instruction is for a compounded set of
    // read and write operations in the same basic block.
    static constexpr unsigned kCompoundRW = (1U << 0);

    explicit InstructionInfo(Instruction *Inst) : Inst(Inst) {}

    Instruction *Inst;
    unsigned Flags = 0;
  };

  void initialize(Module &M);
  bool instrumentLoadStore(const InstructionInfo &II, const DataLayout &DL);
  bool instrumentAtomic(Instruction *I, const DataLayout &DL);
  bool instrumentBranch(Instruction *I, const DataLayout &DL);
  bool instrumentMemIntrinsic(Instruction *I);
  bool instrumentFunctionReturn(Instruction *I);
  bool instrumentFunctionParamCall(Instruction *I);
  int getMemoryAccessFuncIndex(Type *OrigTy, Value *Addr, const DataLayout &DL);
  struct ValSourceInfo {
    Value *Addr;  // null if not found in variables
    uint16_t Idx; // index in function call parameters, start from 1. null if
                  // not found in parameters
    uint16_t offset;
    Value *AddrInst, *IdxInst;
    ValSourceInfo(Value *a = nullptr, uint16_t i = 0)
        : Addr(a), Idx(i), offset(0) {}
    void Reform(IRBuilder<> &IRB) {
      if (mode == Mode::Eagle) {
        if (Addr) {
          // load from memory
          // set the highest bit of IdxInst to 0
          if (Addr->getType()->isIntegerTy())
            AddrInst = IRB.CreateIntToPtr(Addr, IRB.getInt8PtrTy());
          else if (Addr->getType()->isPointerTy())
            AddrInst = IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy());
          else
            AddrInst = IRB.CreateIntToPtr(IRB.getInt8(0), IRB.getInt8PtrTy());
          IdxInst = IRB.getInt16(offset & 0x7fff);
        } else if (Idx) {
          // get from parameters/local function call returns
          // set the hightest bit of IdxInst to 1
          AddrInst = IRB.CreateIntToPtr(IRB.getInt16(Idx), IRB.getInt8PtrTy());
          IdxInst = IRB.getInt16(offset | 0x8000);
        } else {
          AddrInst = IRB.CreateIntToPtr(IRB.getInt8(0), IRB.getInt8PtrTy());
          IdxInst = IRB.getInt16(0);
        }
      } else if (mode == Mode::Verification) {
        if (Addr && Idx == 0) {
          // load from memory
          // set the highest bit of IdxInst to 0
          if (Addr->getType()->isIntegerTy())
            AddrInst = IRB.CreateIntToPtr(Addr, IRB.getInt8PtrTy());
          else if (Addr->getType()->isPointerTy())
            AddrInst = IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy());
          else
            AddrInst = IRB.CreateIntToPtr(IRB.getInt8(0), IRB.getInt8PtrTy());
          IdxInst = IRB.getInt16(offset & 0x7fff);
        } else if (Addr && Idx) {
          // use local/static variable
          // set the hightest bit of IdxInst to 1
          AddrInst = IRB.CreateIntToPtr(IRB.getInt64((uint64_t)Addr),
                                        IRB.getInt8PtrTy());
          IdxInst = IRB.getInt16(offset | 0x8000);
        } else if (Idx) {
          // get from parameters/local function call returns
          // set the hightest bit of IdxInst to 1
          AddrInst = IRB.CreateIntToPtr(IRB.getInt16(Idx), IRB.getInt8PtrTy());
          IdxInst = IRB.getInt16(offset | 0x8000);
        } else {
          printf("Error,%p,%d,%d\n", (void *)Addr, Idx, offset);
          exit(-2);
        }
      }
    }
  };
  void getSource(Value *Val, Function *F, ValSourceInfo &VSI);
  inline std::string concatFileName(std::string dir, std::string file) {
    return dir + "/" + file;
  }

  Type *IntptrTy;
  FunctionCallee TrecFuncEntry;
  FunctionCallee TrecFuncExit;
  FunctionCallee TrecIgnoreBegin;
  // Accesses sizes are powers of two: 1, 2, 4, 8, 16.
  static const size_t kNumberOfAccessSizes = 5;
  FunctionCallee TrecRead[kNumberOfAccessSizes];
  FunctionCallee TrecWrite[kNumberOfAccessSizes];
  FunctionCallee TrecUnalignedRead[kNumberOfAccessSizes];
  FunctionCallee TrecUnalignedWrite[kNumberOfAccessSizes];
  FunctionCallee TrecAtomicLoad[kNumberOfAccessSizes];
  FunctionCallee TrecAtomicStore[kNumberOfAccessSizes];
  FunctionCallee TrecAtomicRMW[AtomicRMWInst::LAST_BINOP + 1]
                              [kNumberOfAccessSizes];
  FunctionCallee TrecAtomicCAS[kNumberOfAccessSizes];
  FunctionCallee TrecAtomicThreadFence;
  FunctionCallee TrecAtomicSignalFence;
  FunctionCallee MemmoveFn, MemcpyFn, MemsetFn;
  FunctionCallee TrecBranch;
  FunctionCallee TrecFuncParam;
  FunctionCallee TrecFuncExitParam;
  FunctionCallee TrecInstDebugInfo;
};

void insertModuleCtor(Module &M) {
  getOrCreateSanitizerCtorAndInitFunctions(
      M, kTrecModuleCtorName, kTrecInitName, /*InitArgTypes=*/{},
      /*InitArgs=*/{},
      // This callback is invoked when the functions are created the first
      // time. Hook them into the global ctors list in that case:
      [&](Function *Ctor, FunctionCallee) { appendToGlobalCtors(M, Ctor, 0); });
}

} // namespace

PreservedAnalyses TraceRecorderPass::run(Function &F,
                                         FunctionAnalysisManager &FAM) {
  TraceRecorder TRec(TraceRecorderModuleVarNames);
  if (TRec.sanitizeFunction(F, FAM.getResult<TargetLibraryAnalysis>(F)))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

PreservedAnalyses ModuleTraceRecorderPass::run(Module &M,
                                               ModuleAnalysisManager &MAM) {
  insertModuleCtor(M);
  TraceRecorderModuleVarNames.clear();
  return PreservedAnalyses::none();
}

void TraceRecorder::initialize(Module &M) {
  const DataLayout &DL = M.getDataLayout();
  IntptrTy = DL.getIntPtrType(M.getContext());
  IRBuilder<> IRB(M.getContext());
  AttributeList Attr;
  Attr = Attr.addFnAttribute(M.getContext(), Attribute::NoUnwind);
  // Initialize the callbacks.
  TrecFuncEntry = M.getOrInsertFunction("__trec_func_entry", Attr,
                                        IRB.getVoidTy(), IRB.getInt8PtrTy());
  TrecFuncExit =
      M.getOrInsertFunction("__trec_func_exit", Attr, IRB.getVoidTy());
  IntegerType *OrdTy = IRB.getInt32Ty();
  for (size_t i = 0; i < kNumberOfAccessSizes; ++i) {
    const unsigned ByteSize = 1U << i;
    const unsigned BitSize = ByteSize * 8;
    std::string ByteSizeStr = utostr(ByteSize);
    std::string BitSizeStr = utostr(BitSize);
    SmallString<32> ReadName("__trec_read" + ByteSizeStr);
    if (i < 4)
      TrecRead[i] = M.getOrInsertFunction(
          ReadName, Attr, IRB.getVoidTy(), IRB.getInt8PtrTy(), IRB.getInt1Ty(),
          IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IRB.getInt16Ty());
    else
      TrecRead[i] = M.getOrInsertFunction(
          ReadName, Attr, IRB.getVoidTy(), IRB.getInt8PtrTy(), IRB.getInt1Ty(),
          IRB.getInt128Ty(), IRB.getInt8PtrTy(), IRB.getInt16Ty());
    SmallString<32> WriteName("__trec_write" + ByteSizeStr);
    if (i < 4)
      TrecWrite[i] = M.getOrInsertFunction(
          WriteName, Attr, IRB.getVoidTy(), IRB.getInt8PtrTy(), IRB.getInt1Ty(),
          IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IRB.getInt16Ty(),
          IRB.getInt8PtrTy(), IRB.getInt16Ty());
    else
      TrecWrite[i] = M.getOrInsertFunction(
          WriteName, Attr, IRB.getVoidTy(), IRB.getInt8PtrTy(), IRB.getInt1Ty(),
          IRB.getInt128Ty(), IRB.getInt8PtrTy(), IRB.getInt16Ty(),
          IRB.getInt8PtrTy(), IRB.getInt16Ty());
    SmallString<64> UnalignedReadName("__trec_unaligned_read" + ByteSizeStr);
    if (i < 4)
      TrecUnalignedRead[i] = M.getOrInsertFunction(
          UnalignedReadName, Attr, IRB.getVoidTy(), IRB.getInt8PtrTy(),
          IRB.getInt1Ty(), IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
          IRB.getInt16Ty());
    else
      TrecUnalignedRead[i] = M.getOrInsertFunction(
          UnalignedReadName, Attr, IRB.getVoidTy(), IRB.getInt8PtrTy(),
          IRB.getInt1Ty(), IRB.getInt128Ty(), IRB.getInt8PtrTy(),
          IRB.getInt16Ty());

    SmallString<64> UnalignedWriteName("__trec_unaligned_write" + ByteSizeStr);
    if (i < 4)
      TrecUnalignedWrite[i] = M.getOrInsertFunction(
          UnalignedWriteName, Attr, IRB.getVoidTy(), IRB.getInt8PtrTy(),
          IRB.getInt1Ty(), IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
          IRB.getInt16Ty(), IRB.getInt8PtrTy(), IRB.getInt16Ty());
    else
      TrecUnalignedWrite[i] = M.getOrInsertFunction(
          UnalignedWriteName, Attr, IRB.getVoidTy(), IRB.getInt8PtrTy(),
          IRB.getInt1Ty(), IRB.getInt128Ty(), IRB.getInt8PtrTy(),
          IRB.getInt16Ty(), IRB.getInt8PtrTy(), IRB.getInt16Ty());

    Type *Ty = Type::getIntNTy(M.getContext(), BitSize);
    Type *PtrTy = Ty->getPointerTo();
    Type *BoolTy = Type::getInt1Ty(M.getContext());
    SmallString<32> AtomicLoadName("__trec_atomic" + BitSizeStr + "_load");
    TrecAtomicLoad[i] =
        M.getOrInsertFunction(AtomicLoadName, Attr, Ty, PtrTy, OrdTy, BoolTy);

    SmallString<32> AtomicStoreName("__trec_atomic" + BitSizeStr + "_store");
    TrecAtomicStore[i] = M.getOrInsertFunction(
        AtomicStoreName, Attr, IRB.getVoidTy(), PtrTy, Ty, OrdTy, BoolTy);

    for (unsigned Op = AtomicRMWInst::FIRST_BINOP;
         Op <= AtomicRMWInst::LAST_BINOP; ++Op) {
      TrecAtomicRMW[Op][i] = nullptr;
      const char *NamePart = nullptr;
      if (Op == AtomicRMWInst::Xchg)
        NamePart = "_exchange";
      else if (Op == AtomicRMWInst::Add)
        NamePart = "_fetch_add";
      else if (Op == AtomicRMWInst::Sub)
        NamePart = "_fetch_sub";
      else if (Op == AtomicRMWInst::And)
        NamePart = "_fetch_and";
      else if (Op == AtomicRMWInst::Or)
        NamePart = "_fetch_or";
      else if (Op == AtomicRMWInst::Xor)
        NamePart = "_fetch_xor";
      else if (Op == AtomicRMWInst::Nand)
        NamePart = "_fetch_nand";
      else
        continue;
      SmallString<32> RMWName("__trec_atomic" + itostr(BitSize) + NamePart);
      TrecAtomicRMW[Op][i] =
          M.getOrInsertFunction(RMWName, Attr, Ty, PtrTy, Ty, OrdTy, BoolTy);
    }

    SmallString<32> AtomicCASName("__trec_atomic" + BitSizeStr +
                                  "_compare_exchange_val");
    TrecAtomicCAS[i] = M.getOrInsertFunction(AtomicCASName, Attr, Ty, PtrTy, Ty,
                                             Ty, OrdTy, OrdTy, BoolTy);
  }
  TrecAtomicThreadFence = M.getOrInsertFunction("__trec_atomic_thread_fence",
                                                Attr, IRB.getVoidTy(), OrdTy);
  TrecAtomicSignalFence = M.getOrInsertFunction("__trec_atomic_signal_fence",
                                                Attr, IRB.getVoidTy(), OrdTy);

  MemmoveFn =
      M.getOrInsertFunction("memmove", Attr, IRB.getInt8PtrTy(),
                            IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IntptrTy);
  MemcpyFn =
      M.getOrInsertFunction("memcpy", Attr, IRB.getInt8PtrTy(),
                            IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IntptrTy);
  MemsetFn =
      M.getOrInsertFunction("memset", Attr, IRB.getInt8PtrTy(),
                            IRB.getInt8PtrTy(), IRB.getInt32Ty(), IntptrTy);
  TrecBranch = M.getOrInsertFunction("__trec_branch", Attr, IRB.getVoidTy(),
                                     IRB.getInt64Ty());
  TrecFuncParam = M.getOrInsertFunction(
      "__trec_func_param", Attr, IRB.getVoidTy(), IRB.getInt16Ty(),
      IRB.getInt8PtrTy(), IRB.getInt16Ty(), IRB.getInt8PtrTy());
  TrecFuncExitParam = M.getOrInsertFunction(
      "__trec_func_exit_param", Attr, IRB.getVoidTy(), IRB.getInt8PtrTy(),
      IRB.getInt16Ty(), IRB.getInt8PtrTy());
  TrecInstDebugInfo = M.getOrInsertFunction(
      "__trec_inst_debug_info", Attr, IRB.getVoidTy(), IRB.getInt16Ty(),
      IRB.getInt16Ty(), IRB.getInt8PtrTy(), IRB.getInt8PtrTy());
}

static bool isVtableAccess(Instruction *I) {
  if (MDNode *Tag = I->getMetadata(LLVMContext::MD_tbaa))
    return Tag->isTBAAVtableAccess();
  return false;
}

static bool isAtomic(Instruction *I) {
  // TODO: Ask TTI whether synchronization scope is between threads.
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

bool TraceRecorder::sanitizeFunction(Function &F,
                                     const TargetLibraryInfo &TLI) {
  // This is required to prevent instrumenting call to __trec_init from
  // within the module constructor.
  if (F.getName() == kTrecModuleCtorName)
    return false;
  // If we cannot find the source file, then this function must not be written
  // by user. Do not instrument it.
  if (F.getSubprogram() == nullptr || F.getSubprogram()->getFile() == nullptr)
    return false;

  // Some cases that we do not instrument

  // Naked functions can not have prologue/epilogue
  // (__trec_func_entry/__trec_func_exit) generated, so don't
  // instrument them at all.
  if (F.hasFnAttribute(Attribute::Naked))
    return false;

  initialize(*F.getParent());
  FuncCallOrders.clear();
  FuncCallOrderCounter = F.arg_size() + 1;
  SmallVector<InstructionInfo> AllStores;
  SmallVector<InstructionInfo> AllLoads;
  SmallVector<Instruction *> AtomicAccesses;
  SmallVector<Instruction *> MemIntrinCalls;
  SmallVector<Instruction *> Branches;
  SmallVector<Instruction *> ParamFuncCalls;
  SmallVector<Instruction *> NoVoidFuncCalls;
  SmallVector<Instruction *> Returns;
  bool Res = false;
  const DataLayout &DL = F.getParent()->getDataLayout();

  for (auto &BB : F) {
    for (auto &Inst : BB) {
      if (isAtomic(&Inst))
        AtomicAccesses.push_back(&Inst);
      if (isa<MemSetInst>(Inst) || isa<MemCpyInst>(Inst) ||
          isa<MemMoveInst>(Inst))
        MemIntrinCalls.push_back(&Inst);
    }
  }
  if (ClInstrumentAtomics)
    for (auto Inst : AtomicAccesses) {
      Res |= instrumentAtomic(Inst, DL);
    }
  if (ClInstrumentMemIntrinsics)
    for (auto Inst : MemIntrinCalls) {
      Res |= instrumentMemIntrinsic(Inst);
    }
  // Traverse all instructions, collect loads/stores/returns, check for calls.
  for (auto &BB : F) {
    for (auto &Inst : BB) {
      if (isa<LoadInst>(Inst)) {
        AllLoads.emplace_back(InstructionInfo(&Inst));
      } else if (isa<StoreInst>(Inst))
        AllStores.emplace_back(InstructionInfo(&Inst));
      else if (isa<CallInst>(Inst) || isa<InvokeInst>(Inst) ||
               isa<CallBrInst>(Inst)) {
        if (!dyn_cast<CallBase>(&Inst)
                 ->getFunctionType()
                 ->getReturnType()
                 ->isVoidTy() ||
            dyn_cast<CallBase>(&Inst)->arg_size())
          ParamFuncCalls.push_back(&Inst);

        if (dyn_cast<CallBase>(&Inst)->getCalledFunction() == nullptr) {
          Branches.push_back(&Inst);
        }

        if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
          maybeMarkSanitizerLibraryCallNoBuiltin(CI, &TLI);
        }
      } else if (isa<BranchInst>(Inst) &&
                 dyn_cast<BranchInst>(&Inst)->isConditional()) {
        Branches.push_back(&Inst); // conditional branch

      } else if (isa<SwitchInst>(Inst)) {
        Branches.push_back(&Inst); // switch
      } else if (isa<ReturnInst>(Inst)) {
        Returns.push_back(&Inst); // function return
      }
    }
  }

  // We have collected all loads and stores.
  // FIXME: many of these accesses do not need to be checked for races
  // (e.g. variables that do not escape, etc).

  // Instrument atomic memory accesses in any case (they can be used to
  // implement synchronization).

  if (ClInstrumentBranch)
    for (auto Inst : Branches) {
      Res |= instrumentBranch(Inst, DL);
    }
  if (ClInstrumentFuncParam) {
    for (auto Inst : ParamFuncCalls) {
      Res |= instrumentFunctionParamCall(Inst);
    }
    for (auto Inst : Returns) {
      Res |= instrumentFunctionReturn(Inst);
    }
  }

  // Instrument memory accesses only if we want to report bugs in the function.
  if (ClInstrumentMemoryAccesses) {
    if (ClInstrumentMemoryWrites)
      for (const auto &II : AllStores) {
        Res |= instrumentLoadStore(II, DL);
      }
    if (ClInstrumentMemoryReads)
      for (const auto &II : AllLoads) {
        Res |= instrumentLoadStore(II, DL);
      }
  }
  // Instrument function entry/exit points if there were instrumented
  // accesses.
  if (Res && ClInstrumentFuncEntryExit) {
    IRBuilder<> IRB(F.getEntryBlock().getFirstNonPHI());
    StringRef FuncName = F.getName();
    if (ClTrecAddDebugInfo && F.getSubprogram() &&
        F.getSubprogram()->getFile()) {
      std::string CurrentFileName =
          concatFileName(F.getSubprogram()->getFile()->getDirectory().str(),
                         F.getSubprogram()->getFile()->getFilename().str());
      FuncName = F.getSubprogram()->getName();
      if (FuncName == "main") {
        Value *func_name, *file_name;
        if (VarNames.count(FuncName.str())) {
          func_name = VarNames.find(FuncName.str())->second;
        } else {
          func_name = IRB.CreateGlobalStringPtr(FuncName);
          VarNames.insert(std::make_pair(FuncName.str(), func_name));
        }
        if (VarNames.count(CurrentFileName)) {
          file_name = VarNames.find(CurrentFileName)->second;
        } else {
          file_name = IRB.CreateGlobalStringPtr(CurrentFileName);
          VarNames.insert(std::make_pair(CurrentFileName, file_name));
        }
        IRB.CreateCall(TrecInstDebugInfo,
                       {IRB.getInt16(F.getSubprogram()->getLine()),
                        IRB.getInt16(0), func_name, file_name});
      }
    }

    Value *name_ptr;
    if (VarNames.count(FuncName.str())) {
      name_ptr = VarNames.find(FuncName.str())->second;
    } else {
      name_ptr = IRB.CreateGlobalStringPtr(FuncName.str());
      VarNames.insert(std::make_pair(FuncName.str(), name_ptr));
    }

    IRB.CreateCall(TrecFuncEntry, {name_ptr});

    EscapeEnumerator EE(F);
    while (IRBuilder<> *AtExit = EE.Next()) {
      AtExit->CreateCall(TrecFuncExit, {});
    }
    Res |= true;
  }
  return Res;
}

bool TraceRecorder::instrumentBranch(Instruction *I, const DataLayout &DL) {
  IRBuilder<> IRB(I);
  Value *var_name, *file_name;

  if (VarNames.count("")) {
    var_name = VarNames.find("")->second;
    file_name = VarNames.find("")->second;
  } else {
    var_name = IRB.CreateGlobalStringPtr("");
    VarNames.insert(std::make_pair("", var_name));
    file_name = var_name;
  }
  Function &F = *(I->getParent()->getParent());
  std::string FuncName = F.getSubprogram() ? F.getSubprogram()->getName().str()
                                           : F.getName().str();
  if (VarNames.count(FuncName))
    var_name = VarNames.find(FuncName)->second;
  else {
    var_name = IRB.CreateGlobalStringPtr(FuncName);
    VarNames.insert(std::make_pair(FuncName, var_name));
  }

  if (F.getSubprogram() && F.getSubprogram()->getFile()) {
    std::string CurrentFileName =
        concatFileName(F.getSubprogram()->getFile()->getDirectory().str(),
                       F.getSubprogram()->getFile()->getFilename().str());
    if (VarNames.count(CurrentFileName))
      file_name = VarNames.find(CurrentFileName)->second;
    else {
      file_name = IRB.CreateGlobalStringPtr(CurrentFileName);
      VarNames.insert(std::make_pair(CurrentFileName, file_name));
    }
  }
  if (isa<BranchInst>(I)) {
    BranchInst *Br = dyn_cast<BranchInst>(I);
    Value *cond = Br->getCondition();

    IRB.CreateCall(TrecInstDebugInfo,
                   {IRB.getInt16(!I->getDebugLoc().isImplicitCode()
                                     ? I->getDebugLoc().getLine()
                                     : 0),
                    IRB.getInt16(!I->getDebugLoc().isImplicitCode()
                                     ? I->getDebugLoc().getCol()
                                     : 0),
                    var_name, file_name});
    if (cond->getType()->isIntegerTy()) {
      IRB.CreateCall(TrecBranch,
                     {IRB.CreateIntCast(cond, IRB.getInt64Ty(), false)});
    } else {
      IRB.CreateCall(TrecBranch, {IRB.getInt64(0)});
    }
    return true;
  } else if (isa<SwitchInst>(I)) {
    SwitchInst *sw = dyn_cast<SwitchInst>(I);
    Value *cond = sw->getCondition();

    IRB.CreateCall(TrecInstDebugInfo,
                   {IRB.getInt16(!I->getDebugLoc().isImplicitCode()
                                     ? I->getDebugLoc().getLine()
                                     : 0),
                    IRB.getInt16(!I->getDebugLoc().isImplicitCode()
                                     ? I->getDebugLoc().getCol()
                                     : 0),
                    var_name, file_name});
    if (cond->getType()->isIntegerTy()) {
      IRB.CreateCall(TrecBranch,
                     {IRB.CreateIntCast(cond, IRB.getInt64Ty(), false)});
    } else {
      IRB.CreateCall(TrecBranch, {IRB.getInt64(0)});
    }
    return true;
  }
  return false;
}

bool TraceRecorder::instrumentFunctionReturn(Instruction *I) {
  IRBuilder<> IRB(I);
  ValSourceInfo VSI_val;
  Value *RetVal = dyn_cast<ReturnInst>(I)->getReturnValue();
  bool res = false;
  if (RetVal) {
    getSource(RetVal, I->getParent()->getParent(), VSI_val);
    VSI_val.Reform(IRB);
    Value *RetValInst = RetVal;
    if (RetValInst->getType()->isIntegerTy())
      RetValInst = IRB.CreateIntToPtr(RetValInst, IRB.getInt8PtrTy());
    else if (RetValInst->getType()->isPointerTy())
      RetValInst = IRB.CreatePointerCast(RetValInst, IRB.getInt8PtrTy());
    else
      RetValInst =
          IRB.CreateIntToPtr(IRB.getInt64(0x123456789), IRB.getInt8PtrTy());

    IRB.CreateCall(TrecFuncExitParam,
                   {VSI_val.AddrInst, VSI_val.IdxInst, RetValInst});
    res = true;
  }
  return res;
}

bool TraceRecorder::instrumentFunctionParamCall(Instruction *I) {
  IRBuilder<> IRB(I);
  CallBase *CI = dyn_cast<CallBase>(I);
  if (CI->getCalledFunction() &&
      (CI->getCalledFunction()->hasFnAttribute(Attribute::Naked) ||
       CI->getCalledFunction()->getName().startswith("llvm.dbg")))
    return false;
  unsigned int arg_size = CI->arg_size();
  IRB.CreateCall(TrecFuncParam,
                 {IRB.getInt16(0),
                  IRB.CreateIntToPtr(IRB.getInt32(FuncCallOrderCounter),
                                     IRB.getInt8PtrTy()),
                  IRB.getInt16(arg_size),
                  IRB.CreateIntToPtr(IRB.getInt64(0), IRB.getInt8PtrTy())});
  FuncCallOrders.insert(std::make_pair(I, FuncCallOrderCounter++));
  for (unsigned int i = 0; i < arg_size; i++) {
    ValSourceInfo VSI;
    getSource(CI->getArgOperand(i), I->getParent()->getParent(), VSI);
    if (VSI.Addr || VSI.Idx) {
      Value *ValInst;
      if (CI->getArgOperand(i)->getType()->isIntegerTy())
        ValInst = IRB.CreateIntToPtr(CI->getArgOperand(i), IRB.getInt8PtrTy());
      else if (CI->getArgOperand(i)->getType()->isPointerTy())
        ValInst =
            IRB.CreatePointerCast(CI->getArgOperand(i), IRB.getInt8PtrTy());
      else
        ValInst = IRB.CreateIntToPtr(IRB.getInt64(0), IRB.getInt8PtrTy());
      VSI.Reform(IRB);
      IRB.CreateCall(TrecFuncParam,
                     {IRB.getInt16(i + 1), VSI.AddrInst, VSI.IdxInst, ValInst});
    }
  }
  Function *F = CI->getCalledFunction();

  if (ClTrecAddDebugInfo) {
    std::string CurrentFileName = "";
    if (CI->getDebugLoc().get())
      CurrentFileName =
          concatFileName((CI->getDebugLoc()->getScope()->getDirectory().str()),
                         (CI->getDebugLoc()->getScope()->getFilename().str()));
    StringRef FuncName = "";
    if (F)
      FuncName =
          (F->getSubprogram()) ? F->getSubprogram()->getName() : F->getName();
    if (FuncName == "pthread_create") {
      Function *called = dyn_cast<Function>(CI->getArgOperand(2));
      FuncName = called ? called->getSubprogram()->getName()
                        : CI->getArgOperand(2)->getName();
    }
    Value *func_name, *file_name;
    if (VarNames.count(FuncName.str())) {
      func_name = VarNames.find(FuncName.str())->second;
    } else {
      func_name = IRB.CreateGlobalStringPtr(FuncName);
      VarNames.insert(std::make_pair(FuncName.str(), func_name));
    }
    if (VarNames.count(CurrentFileName)) {
      file_name = VarNames.find(CurrentFileName)->second;
    } else {
      file_name = IRB.CreateGlobalStringPtr(CurrentFileName);
      VarNames.insert(std::make_pair(CurrentFileName, file_name));
    }
    IRB.CreateCall(TrecInstDebugInfo,
                   {IRB.getInt16(!I->getDebugLoc().isImplicitCode()
                                     ? I->getDebugLoc().getLine()
                                     : 0),
                    IRB.getInt16(!I->getDebugLoc().isImplicitCode()
                                     ? I->getDebugLoc().getCol()
                                     : 0),
                    func_name, file_name});
  }
  return true;
}

bool TraceRecorder::instrumentLoadStore(const InstructionInfo &II,
                                        const DataLayout &DL) {
  IRBuilder<> IRB(II.Inst);
  const bool IsWrite = isa<StoreInst>(*II.Inst);
  Value *Addr = IsWrite ? cast<StoreInst>(II.Inst)->getPointerOperand()
                        : cast<LoadInst>(II.Inst)->getPointerOperand();
  // swifterror memory addresses are mem2reg promoted by instruction
  // selection. As such they cannot have regular uses like an instrumentation
  // function and it makes no sense to track them as memory.
  if (Addr->isSwiftError())
    return false;
  Type *OrigTy = getLoadStoreType(II.Inst);
  int Idx = getMemoryAccessFuncIndex(OrigTy, Addr, DL);
  if (Idx < 0 || Idx >= 4)
    return false;
  // never instrument vtable update/read operations
  if (isVtableAccess(II.Inst)) {
    return false;
  }

  const uint64_t Alignment = IsWrite
                                 ? cast<StoreInst>(II.Inst)->getAlign().value()
                                 : cast<LoadInst>(II.Inst)->getAlign().value();
  const bool isPtrTy = isa<PointerType>(OrigTy);
  const uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  Type *ValType = Type::getIntNTy(II.Inst->getContext(), (1 << Idx) * 8);
  FunctionCallee OnAccessFunc = nullptr;
  if (Alignment == 0 || Alignment >= 8 || (Alignment % (TypeSize / 8)) == 0) {
    OnAccessFunc = IsWrite ? TrecWrite[Idx] : TrecRead[Idx];
  } else {
    OnAccessFunc = IsWrite ? TrecUnalignedWrite[Idx] : TrecUnalignedRead[Idx];
  }

  ValSourceInfo VSI_addr;
  getSource(Addr, II.Inst->getParent()->getParent(), VSI_addr);

  std::string AddrName, ValName;
  AddrName = Addr->getName().str();

  if (IsWrite) {
    Value *StoredValue = cast<StoreInst>(II.Inst)->getValueOperand();
    ValName = StoredValue->getName().str();

    if (ClTrecAddDebugInfo && (ValName != "" || AddrName != "" ||
                               !II.Inst->getDebugLoc().isImplicitCode())) {
      Value *ptr_to_valname, *ptr_to_addrname;
      if (VarNames.count(ValName)) {
        ptr_to_valname = VarNames.find(ValName)->second;
      } else {
        ptr_to_valname = IRB.CreateGlobalStringPtr(ValName);
        VarNames.insert(std::make_pair(ValName, ptr_to_valname));
      }

      if (VarNames.count(AddrName)) {
        ptr_to_addrname = VarNames.find(AddrName)->second;
      } else {
        ptr_to_addrname = IRB.CreateGlobalStringPtr(AddrName);
        VarNames.insert(std::make_pair(AddrName, ptr_to_addrname));
      }
      IRB.CreateCall(TrecInstDebugInfo,
                     {IRB.getInt16(!II.Inst->getDebugLoc().isImplicitCode()
                                       ? II.Inst->getDebugLoc().getLine()
                                       : 0),
                      IRB.getInt16(!II.Inst->getDebugLoc().isImplicitCode()
                                       ? II.Inst->getDebugLoc().getCol()
                                       : 0),
                      ptr_to_valname, ptr_to_addrname});
    }
    ValSourceInfo VSI_val;
    getSource(StoredValue, II.Inst->getParent()->getParent(), VSI_val);
    if (isa<VectorType>(StoredValue->getType()))
      StoredValue = IRB.CreateBitCast(StoredValue, ValType);

    if (!StoredValue->getType()->isIntegerTy()) {
      StoredValue = IRB.CreateBitOrPointerCast(StoredValue, ValType);
    }
    StoredValue = IRB.CreateIntToPtr(StoredValue, IRB.getInt8PtrTy());
    VSI_addr.Reform(IRB);
    VSI_val.Reform(IRB);
    IRB.CreateCall(OnAccessFunc,
                   {IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                    IRB.getInt1(isPtrTy), StoredValue, VSI_addr.AddrInst,
                    VSI_addr.IdxInst, VSI_val.AddrInst, VSI_val.IdxInst});

    NumInstrumentedWrites++;
  } else {
    // just for recording the PC number
    IRB.CreateCall(OnAccessFunc,
                   {IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                    IRB.getInt1(isPtrTy),
                    IRB.CreateIntToPtr(IRB.getInt8(0), IRB.getInt8PtrTy()),
                    IRB.CreateIntToPtr(IRB.getInt8(0), IRB.getInt8PtrTy()),
                    IRB.getInt16(0)});

    // read inst should not be the last inst in a BB, thus no need to check
    // for nullptr
    IRBuilder<> IRB2(II.Inst->getNextNode());
    Value *LoadedValue = II.Inst;
    ValName = II.Inst->getName().str();
    if (ClTrecAddDebugInfo && (ValName != "" || AddrName != "" ||
                               !II.Inst->getDebugLoc().isImplicitCode())) {
      Value *ptr_to_valname, *ptr_to_addrname;
      if (VarNames.count(ValName)) {
        ptr_to_valname = VarNames.find(ValName)->second;
      } else {
        ptr_to_valname = IRB2.CreateGlobalStringPtr(ValName);
        VarNames.insert(std::make_pair(ValName, ptr_to_valname));
      }
      if (VarNames.count(AddrName)) {
        ptr_to_addrname = VarNames.find(AddrName)->second;
      } else {
        ptr_to_addrname = IRB2.CreateGlobalStringPtr(AddrName);
        VarNames.insert(std::make_pair(AddrName, ptr_to_addrname));
      }
      IRB2.CreateCall(TrecInstDebugInfo,
                      {IRB2.getInt16(!II.Inst->getDebugLoc().isImplicitCode()
                                         ? II.Inst->getDebugLoc().getLine()
                                         : 0),
                       IRB2.getInt16(!II.Inst->getDebugLoc().isImplicitCode()
                                         ? II.Inst->getDebugLoc().getCol()
                                         : 0),
                       ptr_to_valname, ptr_to_addrname});
    }
    if (isa<VectorType>(LoadedValue->getType())) {
      LoadedValue = IRB2.CreateBitCast(LoadedValue, ValType);
    }

    if (!LoadedValue->getType()->isIntegerTy()) {
      LoadedValue = IRB2.CreateBitOrPointerCast(LoadedValue, ValType);
    }
    LoadedValue = IRB2.CreateIntToPtr(LoadedValue, IRB2.getInt8PtrTy());
    VSI_addr.Reform(IRB2);
    IRB2.CreateCall(OnAccessFunc,
                    {IRB2.CreatePointerCast(Addr, IRB2.getInt8PtrTy()),
                     IRB2.getInt1(isPtrTy), LoadedValue, VSI_addr.AddrInst,
                     VSI_addr.IdxInst});

    NumInstrumentedReads++;
  }
  return true;
}

void TraceRecorder::getSource(Value *Val, Function *F,
                              TraceRecorder::ValSourceInfo &VSI) {
  VSI.Addr = nullptr;
  VSI.Idx = 0;
  VSI.offset = 0;
  size_t arg_num = F->arg_size();
  SmallVector<std::pair<Value *, uint16_t>> possibleValues;
  possibleValues.push_back(std::make_pair(Val, 0));
  while (!possibleValues.empty()) {
    Value *SrcValue = possibleValues.back().first;
    uint16_t offset = possibleValues.back().second;
    possibleValues.pop_back();

    while (isa<LoadInst>(SrcValue) || isa<CastInst>(SrcValue) ||
           isa<GetElementPtrInst>(SrcValue) || isa<BinaryOperator>(SrcValue)) {
      if (isa<LoadInst>(SrcValue)) {
        // get source address
        VSI.Addr = dyn_cast<LoadInst>(SrcValue)->getPointerOperand();
        VSI.offset = offset;
        return;
      } else if (isa<CastInst>(SrcValue)) {
        // cast inst, get its source value
        SrcValue = dyn_cast<CastInst>(SrcValue)->getOperand(0);
      } else if (isa<GetElementPtrInst>(SrcValue)) {
        // GEP inst, get its source value
        APInt Offset(64, 0, false);
        if (dyn_cast<GetElementPtrInst>(SrcValue)->accumulateConstantOffset(
                F->getParent()->getDataLayout(), Offset)) {
          offset += *Offset.getRawData();
        }
        SrcValue = dyn_cast<GetElementPtrInst>(SrcValue)->getPointerOperand();
      } else if (isa<BinaryOperator>(SrcValue)) {
        BinaryOperator *I = dyn_cast<BinaryOperator>(SrcValue);
        if (!isa<ConstantInt>(I->getOperand(0)) &&
            !isa<ConstantInt>(I->getOperand(1))) {
          possibleValues.push_back(std::make_pair(I->getOperand(1), offset));
          possibleValues.push_back(std::make_pair(I->getOperand(0), offset));
        } else if (isa<ConstantInt>(I->getOperand(0)) &&
                   isa<ConstantInt>(I->getOperand(1))) {
          break;
        } else {

          if (I->getOpcode() == Instruction::BinaryOps::Add) {
            ConstantInt *cons = isa<ConstantInt>(I->getOperand(1))
                                    ? dyn_cast<ConstantInt>(I->getOperand(1))
                                    : dyn_cast<ConstantInt>(I->getOperand(0));
            offset += *cons->getValue().getRawData();
            possibleValues.push_back(std::make_pair(
                I->getOperand(isa<ConstantInt>(I->getOperand(1)) ? 0 : 1),
                offset));
          } else if (I->getOpcode() == Instruction::BinaryOps::Sub &&
                     isa<ConstantInt>(I->getOperand(1))) {
            ConstantInt *cons = dyn_cast<ConstantInt>(I->getOperand(1));
            offset -= *cons->getValue().getRawData();
            possibleValues.push_back(std::make_pair(I->getOperand(0), offset));
          }
        }
        break;
      }
    }

    if (isa<CallInst>(SrcValue) || isa<InvokeInst>(SrcValue) ||
        isa<CallBrInst>(SrcValue)) {
      if (FuncCallOrders.count(dyn_cast<Instruction>(SrcValue))) {
        VSI.Idx = FuncCallOrders.lookup(dyn_cast<Instruction>(SrcValue));
        VSI.offset = offset;
        return;
      }
    } else if (!isa<Instruction>(SrcValue) && arg_num) {
      for (unsigned int i = 0; i < arg_num; i++) {
        Value *arg = F->getArg(i);
        if (SrcValue == arg) {
          VSI.Idx = i + 1;
          VSI.offset = offset;
          return;
        }
      }
    }
  }

  if (mode == Mode::Verification) {
    VSI.Addr = Val;
    VSI.Idx = 0xffff;
  }
  return;
}

static ConstantInt *createOrdering(IRBuilder<> *IRB, AtomicOrdering ord) {
  uint32_t v = 0;
  switch (ord) {
  case AtomicOrdering::NotAtomic:
    llvm_unreachable("unexpected atomic ordering!");
  case AtomicOrdering::Unordered:
    LLVM_FALLTHROUGH;
  case AtomicOrdering::Monotonic:
    v = 0;
    break;
  // Not specified yet:
  // case AtomicOrdering::Consume:                v = 1; break;
  case AtomicOrdering::Acquire:
    v = 2;
    break;
  case AtomicOrdering::Release:
    v = 3;
    break;
  case AtomicOrdering::AcquireRelease:
    v = 4;
    break;
  case AtomicOrdering::SequentiallyConsistent:
    v = 5;
    break;
  }
  return IRB->getInt32(v);
}

// If a memset intrinsic gets inlined by the code gen, we will miss races on
// it. So, we either need to ensure the intrinsic is not inlined, or
// instrument it. We do not instrument memset/memmove/memcpy intrinsics (too
// complicated), instead we simply replace them with regular function calls,
// which are then intercepted by the run-time. Since trec is running after
// everyone else, the calls should not be replaced back with intrinsics. If
// that becomes wrong at some point, we will need to call e.g. __trec_memset
// to avoid the intrinsics.
bool TraceRecorder::instrumentMemIntrinsic(Instruction *I) {
  IRBuilder<> IRB(I);
  if (MemSetInst *M = dyn_cast<MemSetInst>(I)) {
    CallInst *NewInst = IRB.CreateCall(
        MemsetFn,
        {IRB.CreatePointerCast(M->getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(M->getArgOperand(1), IRB.getInt32Ty(), false),
         IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
    NewInst->setDebugLoc(M->getDebugLoc());
    I->eraseFromParent();
  } else if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
    CallInst *NewInst = IRB.CreateCall(
        isa<MemCpyInst>(M) ? MemcpyFn : MemmoveFn,
        {IRB.CreatePointerCast(M->getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreatePointerCast(M->getArgOperand(1), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
    NewInst->setDebugLoc(M->getDebugLoc());
    I->eraseFromParent();
  }
  return false;
}

// Both llvm and TraceRecorder atomic operations are based on C++11/C1x
// standards.  For background see C++11 standard.  A slightly older, publicly
// available draft of the standard (not entirely up-to-date, but close enough
// for casual browsing) is available here:
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3242.pdf
// The following page contains more background information:
// http://www.hpl.hp.com/personal/Hans_Boehm/c++mm/

bool TraceRecorder::instrumentAtomic(Instruction *I, const DataLayout &DL) {
  IRBuilder<> IRB(I);
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    Value *Addr = LI->getPointerOperand();
    Type *OrigTy = LI->getType();
    int Idx = getMemoryAccessFuncIndex(OrigTy, Addr, DL);
    if (Idx < 0)
      return false;
    const unsigned ByteSize = 1U << Idx;
    const unsigned BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Type *PtrTy = Ty->getPointerTo();
    Value *Args[] = {IRB.CreatePointerCast(Addr, PtrTy),
                     createOrdering(&IRB, LI->getOrdering()),
                     IRB.getInt1(OrigTy->isPointerTy())};

    CallInst *C = IRB.CreateCall(TrecAtomicLoad[Idx], Args);
    C->setDebugLoc(LI->getDebugLoc());
    Value *Cast = IRB.CreateBitOrPointerCast(C, OrigTy);
    I->replaceAllUsesWith(Cast);
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    Value *Addr = SI->getPointerOperand();
    int Idx =
        getMemoryAccessFuncIndex(SI->getValueOperand()->getType(), Addr, DL);
    if (Idx < 0)
      return false;
    const unsigned ByteSize = 1U << Idx;
    const unsigned BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Type *PtrTy = Ty->getPointerTo();
    Type *OrigTy = cast<PointerType>(Addr->getType())->getPointerElementType();
    Value *Args[] = {IRB.CreatePointerCast(Addr, PtrTy),
                     IRB.CreateBitOrPointerCast(SI->getValueOperand(), Ty),
                     createOrdering(&IRB, SI->getOrdering()),
                     IRB.getInt1(OrigTy->isPointerTy())};
    CallInst *C = CallInst::Create(TrecAtomicStore[Idx], Args);
    C->setDebugLoc(SI->getDebugLoc());
    ReplaceInstWithInst(I, C);
  } else if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I)) {
    Value *Addr = RMWI->getPointerOperand();
    int Idx =
        getMemoryAccessFuncIndex(RMWI->getValOperand()->getType(), Addr, DL);
    if (Idx < 0)
      return false;
    FunctionCallee F = TrecAtomicRMW[RMWI->getOperation()][Idx];
    if (!F)
      return false;
    const unsigned ByteSize = 1U << Idx;
    const unsigned BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Type *PtrTy = Ty->getPointerTo();
    Type *OrigTy = cast<PointerType>(Addr->getType())->getPointerElementType();
    Value *Args[] = {IRB.CreatePointerCast(Addr, PtrTy),
                     IRB.CreateIntCast(RMWI->getValOperand(), Ty, false),
                     createOrdering(&IRB, RMWI->getOrdering()),
                     IRB.getInt1(OrigTy->isPointerTy())};
    CallInst *C = CallInst::Create(F, Args);
    C->setDebugLoc(RMWI->getDebugLoc());
    ReplaceInstWithInst(I, C);
  } else if (AtomicCmpXchgInst *CASI = dyn_cast<AtomicCmpXchgInst>(I)) {
    Value *Addr = CASI->getPointerOperand();
    int Idx =
        getMemoryAccessFuncIndex(CASI->getNewValOperand()->getType(), Addr, DL);
    if (Idx < 0)
      return false;
    const unsigned ByteSize = 1U << Idx;
    const unsigned BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Type *PtrTy = Ty->getPointerTo();
    Type *OrigTy = cast<PointerType>(Addr->getType())->getPointerElementType();
    Value *CmpOperand =
        IRB.CreateBitOrPointerCast(CASI->getCompareOperand(), Ty);
    Value *NewOperand =
        IRB.CreateBitOrPointerCast(CASI->getNewValOperand(), Ty);
    Value *Args[] = {IRB.CreatePointerCast(Addr, PtrTy),
                     CmpOperand,
                     NewOperand,
                     createOrdering(&IRB, CASI->getSuccessOrdering()),
                     createOrdering(&IRB, CASI->getFailureOrdering()),
                     IRB.getInt1(OrigTy->isPointerTy())};
    CallInst *C = IRB.CreateCall(TrecAtomicCAS[Idx], Args);
    C->setDebugLoc(CASI->getDebugLoc());
    Value *Success = IRB.CreateICmpEQ(C, CmpOperand);
    Value *OldVal = C;
    Type *OrigOldValTy = CASI->getNewValOperand()->getType();
    if (Ty != OrigOldValTy) {
      // The value is a pointer, so we need to cast the return value.
      OldVal = IRB.CreateIntToPtr(C, OrigOldValTy);
    }

    Value *Res =
        IRB.CreateInsertValue(UndefValue::get(CASI->getType()), OldVal, 0);
    Res = IRB.CreateInsertValue(Res, Success, 1);

    I->replaceAllUsesWith(Res);
    I->eraseFromParent();
  } else if (FenceInst *FI = dyn_cast<FenceInst>(I)) {
    Value *Args[] = {createOrdering(&IRB, FI->getOrdering())};
    FunctionCallee F = FI->getSyncScopeID() == SyncScope::SingleThread
                           ? TrecAtomicSignalFence
                           : TrecAtomicThreadFence;
    CallInst *C = CallInst::Create(F, Args);
    C->setDebugLoc(FI->getDebugLoc());
    ReplaceInstWithInst(I, C);
  }
  return true;
}

int TraceRecorder::getMemoryAccessFuncIndex(Type *OrigTy, Value *Addr,
                                            const DataLayout &DL) {
  assert(OrigTy->isSized());
  uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  if (TypeSize != 8 && TypeSize != 16 && TypeSize != 32 && TypeSize != 64 &&
      TypeSize != 128) {
    NumAccessesWithBadSize++;
    // Ignore all unusual sizes.
    return -1;
  }
  size_t Idx = countTrailingZeros(TypeSize / 8);
  assert(Idx < kNumberOfAccessSizes);
  return Idx;
}
