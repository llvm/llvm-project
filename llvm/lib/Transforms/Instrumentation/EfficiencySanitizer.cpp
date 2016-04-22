//===-- EfficiencySanitizer.cpp - performance tuner -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of EfficiencySanitizer, a family of performance tuners
// that detects multiple performance issues via separate sub-tools.
//
// The instrumentation phase is straightforward:
//   - Take action on every memory access: either inlined instrumentation,
//     or Inserted calls to our run-time library.
//   - Optimizations may apply to avoid instrumenting some of the accesses.
//   - Turn mem{set,cpy,move} instrinsics into library calls.
// The rest is handled by the run-time library.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "esan"

// The tool type must be just one of these ClTool* options, as the tools
// cannot be combined due to shadow memory constraints.
static cl::opt<bool>
    ClToolCacheFrag("esan-cache-frag", cl::init(false),
                    cl::desc("Detect data cache fragmentation"), cl::Hidden);
// Each new tool will get its own opt flag here.
// These are converted to EfficiencySanitizerOptions for use
// in the code.

static cl::opt<bool> ClInstrumentLoadsAndStores(
    "esan-instrument-loads-and-stores", cl::init(true),
    cl::desc("Instrument loads and stores"), cl::Hidden);
static cl::opt<bool> ClInstrumentMemIntrinsics(
    "esan-instrument-memintrinsics", cl::init(true),
    cl::desc("Instrument memintrinsics (memset/memcpy/memmove)"), cl::Hidden);

STATISTIC(NumInstrumentedLoads, "Number of instrumented loads");
STATISTIC(NumInstrumentedStores, "Number of instrumented stores");
STATISTIC(NumFastpaths, "Number of instrumented fastpaths");
STATISTIC(NumAccessesWithIrregularSize,
          "Number of accesses with a size outside our targeted callout sizes");

static const char *const EsanModuleCtorName = "esan.module_ctor";
static const char *const EsanInitName = "__esan_init";

namespace {

static EfficiencySanitizerOptions
OverrideOptionsFromCL(EfficiencySanitizerOptions Options) {
  if (ClToolCacheFrag)
    Options.ToolType = EfficiencySanitizerOptions::ESAN_CacheFrag;

  // Direct opt invocation with no params will have the default ESAN_None.
  // We run the default tool in that case.
  if (Options.ToolType == EfficiencySanitizerOptions::ESAN_None)
    Options.ToolType = EfficiencySanitizerOptions::ESAN_CacheFrag;

  return Options;
}

/// EfficiencySanitizer: instrument each module to find performance issues.
class EfficiencySanitizer : public FunctionPass {
public:
  EfficiencySanitizer(
      const EfficiencySanitizerOptions &Opts = EfficiencySanitizerOptions())
      : FunctionPass(ID), Options(OverrideOptionsFromCL(Opts)) {}
  const char *getPassName() const override;
  bool runOnFunction(Function &F) override;
  bool doInitialization(Module &M) override;
  static char ID;

private:
  void initializeCallbacks(Module &M);
  bool instrumentLoadOrStore(Instruction *I, const DataLayout &DL);
  bool instrumentMemIntrinsic(MemIntrinsic *MI);
  bool shouldIgnoreMemoryAccess(Instruction *I);
  int getMemoryAccessFuncIndex(Value *Addr, const DataLayout &DL);
  bool instrumentFastpath(Instruction *I, const DataLayout &DL, bool IsStore,
                          Value *Addr, unsigned Alignment);
  // Each tool has its own fastpath routine:
  bool instrumentFastpathCacheFrag(Instruction *I, const DataLayout &DL,
                                   Value *Addr, unsigned Alignment);

  EfficiencySanitizerOptions Options;
  LLVMContext *Ctx;
  Type *IntptrTy;
  // Our slowpath involves callouts to the runtime library.
  // Access sizes are powers of two: 1, 2, 4, 8, 16.
  static const size_t NumberOfAccessSizes = 5;
  Function *EsanAlignedLoad[NumberOfAccessSizes];
  Function *EsanAlignedStore[NumberOfAccessSizes];
  Function *EsanUnalignedLoad[NumberOfAccessSizes];
  Function *EsanUnalignedStore[NumberOfAccessSizes];
  // For irregular sizes of any alignment:
  Function *EsanUnalignedLoadN, *EsanUnalignedStoreN;
  Function *MemmoveFn, *MemcpyFn, *MemsetFn;
  Function *EsanCtorFunction;
};
} // namespace

char EfficiencySanitizer::ID = 0;
INITIALIZE_PASS(EfficiencySanitizer, "esan",
                "EfficiencySanitizer: finds performance issues.", false, false)

const char *EfficiencySanitizer::getPassName() const {
  return "EfficiencySanitizer";
}

FunctionPass *
llvm::createEfficiencySanitizerPass(const EfficiencySanitizerOptions &Options) {
  return new EfficiencySanitizer(Options);
}

void EfficiencySanitizer::initializeCallbacks(Module &M) {
  IRBuilder<> IRB(M.getContext());
  // Initialize the callbacks.
  for (size_t Idx = 0; Idx < NumberOfAccessSizes; ++Idx) {
    const unsigned ByteSize = 1U << Idx;
    std::string ByteSizeStr = utostr(ByteSize);
    // We'll inline the most common (i.e., aligned and frequent sizes)
    // load + store instrumentation: these callouts are for the slowpath.
    SmallString<32> AlignedLoadName("__esan_aligned_load" + ByteSizeStr);
    EsanAlignedLoad[Idx] =
        checkSanitizerInterfaceFunction(M.getOrInsertFunction(
            AlignedLoadName, IRB.getVoidTy(), IRB.getInt8PtrTy(), nullptr));
    SmallString<32> AlignedStoreName("__esan_aligned_store" + ByteSizeStr);
    EsanAlignedStore[Idx] =
        checkSanitizerInterfaceFunction(M.getOrInsertFunction(
            AlignedStoreName, IRB.getVoidTy(), IRB.getInt8PtrTy(), nullptr));
    SmallString<32> UnalignedLoadName("__esan_unaligned_load" + ByteSizeStr);
    EsanUnalignedLoad[Idx] =
        checkSanitizerInterfaceFunction(M.getOrInsertFunction(
            UnalignedLoadName, IRB.getVoidTy(), IRB.getInt8PtrTy(), nullptr));
    SmallString<32> UnalignedStoreName("__esan_unaligned_store" + ByteSizeStr);
    EsanUnalignedStore[Idx] =
        checkSanitizerInterfaceFunction(M.getOrInsertFunction(
            UnalignedStoreName, IRB.getVoidTy(), IRB.getInt8PtrTy(), nullptr));
  }
  EsanUnalignedLoadN = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction("__esan_unaligned_loadN", IRB.getVoidTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  EsanUnalignedStoreN = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction("__esan_unaligned_storeN", IRB.getVoidTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  MemmoveFn = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction("memmove", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  MemcpyFn = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction("memcpy", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr));
  MemsetFn = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction("memset", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                            IRB.getInt32Ty(), IntptrTy, nullptr));
}

bool EfficiencySanitizer::doInitialization(Module &M) {
  Ctx = &M.getContext();
  const DataLayout &DL = M.getDataLayout();
  IRBuilder<> IRB(M.getContext());
  IntegerType *OrdTy = IRB.getInt32Ty();
  IntptrTy = DL.getIntPtrType(M.getContext());
  std::tie(EsanCtorFunction, std::ignore) = createSanitizerCtorAndInitFunctions(
      M, EsanModuleCtorName, EsanInitName, /*InitArgTypes=*/{OrdTy},
      /*InitArgs=*/{
          ConstantInt::get(OrdTy, static_cast<int>(Options.ToolType))});

  appendToGlobalCtors(M, EsanCtorFunction, 0);

  return true;
}

bool EfficiencySanitizer::shouldIgnoreMemoryAccess(Instruction *I) {
  if (Options.ToolType == EfficiencySanitizerOptions::ESAN_CacheFrag) {
    // We'd like to know about cache fragmentation in vtable accesses and
    // constant data references, so we do not currently ignore anything.
    return false;
  }
  // TODO(bruening): future tools will be returning true for some cases.
  return false;
}

bool EfficiencySanitizer::runOnFunction(Function &F) {
  // This is required to prevent instrumenting the call to __esan_init from
  // within the module constructor.
  if (&F == EsanCtorFunction)
    return false;
  // As a function pass, we must re-initialize every time.
  initializeCallbacks(*F.getParent());
  SmallVector<Instruction *, 8> LoadsAndStores;
  SmallVector<Instruction *, 8> MemIntrinCalls;
  bool Res = false;
  const DataLayout &DL = F.getParent()->getDataLayout();

  for (auto &BB : F) {
    for (auto &Inst : BB) {
      if ((isa<LoadInst>(Inst) || isa<StoreInst>(Inst) ||
           isa<AtomicRMWInst>(Inst) || isa<AtomicCmpXchgInst>(Inst)) &&
          !shouldIgnoreMemoryAccess(&Inst))
        LoadsAndStores.push_back(&Inst);
      else if (isa<MemIntrinsic>(Inst))
        MemIntrinCalls.push_back(&Inst);
    }
  }

  if (ClInstrumentLoadsAndStores) {
    for (auto Inst : LoadsAndStores) {
      Res |= instrumentLoadOrStore(Inst, DL);
    }
  }

  if (ClInstrumentMemIntrinsics) {
    for (auto Inst : MemIntrinCalls) {
      Res |= instrumentMemIntrinsic(cast<MemIntrinsic>(Inst));
    }
  }

  return Res;
}

bool EfficiencySanitizer::instrumentLoadOrStore(Instruction *I,
                                                const DataLayout &DL) {
  IRBuilder<> IRB(I);
  bool IsStore;
  Value *Addr;
  unsigned Alignment;
  if (LoadInst *Load = dyn_cast<LoadInst>(I)) {
    IsStore = false;
    Alignment = Load->getAlignment();
    Addr = Load->getPointerOperand();
  } else if (StoreInst *Store = dyn_cast<StoreInst>(I)) {
    IsStore = true;
    Alignment = Store->getAlignment();
    Addr = Store->getPointerOperand();
  } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
    IsStore = true;
    Alignment = 0;
    Addr = RMW->getPointerOperand();
  } else if (AtomicCmpXchgInst *Xchg = dyn_cast<AtomicCmpXchgInst>(I)) {
    IsStore = true;
    Alignment = 0;
    Addr = Xchg->getPointerOperand();
  } else
    llvm_unreachable("Unsupported mem access type");

  Type *OrigTy = cast<PointerType>(Addr->getType())->getElementType();
  const uint32_t TypeSizeBytes = DL.getTypeStoreSizeInBits(OrigTy) / 8;
  Value *OnAccessFunc = nullptr;
  if (IsStore)
    NumInstrumentedStores++;
  else
    NumInstrumentedLoads++;
  int Idx = getMemoryAccessFuncIndex(Addr, DL);
  if (Idx < 0) {
    OnAccessFunc = IsStore ? EsanUnalignedStoreN : EsanUnalignedLoadN;
    IRB.CreateCall(OnAccessFunc,
                   {IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                    ConstantInt::get(IntptrTy, TypeSizeBytes)});
  } else {
    if (instrumentFastpath(I, DL, IsStore, Addr, Alignment)) {
      NumFastpaths++;
      return true;
    }
    if (Alignment == 0 || Alignment >= 8 || (Alignment % TypeSizeBytes) == 0)
      OnAccessFunc = IsStore ? EsanAlignedStore[Idx] : EsanAlignedLoad[Idx];
    else
      OnAccessFunc = IsStore ? EsanUnalignedStore[Idx] : EsanUnalignedLoad[Idx];
    IRB.CreateCall(OnAccessFunc,
                   IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()));
  }
  return true;
}

// It's simplest to replace the memset/memmove/memcpy intrinsics with
// calls that the runtime library intercepts.
// Our pass is late enough that calls should not turn back into intrinsics.
bool EfficiencySanitizer::instrumentMemIntrinsic(MemIntrinsic *MI) {
  IRBuilder<> IRB(MI);
  bool Res = false;
  if (isa<MemSetInst>(MI)) {
    IRB.CreateCall(
        MemsetFn,
        {IRB.CreatePointerCast(MI->getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(MI->getArgOperand(1), IRB.getInt32Ty(), false),
         IRB.CreateIntCast(MI->getArgOperand(2), IntptrTy, false)});
    MI->eraseFromParent();
    Res = true;
  } else if (isa<MemTransferInst>(MI)) {
    IRB.CreateCall(
        isa<MemCpyInst>(MI) ? MemcpyFn : MemmoveFn,
        {IRB.CreatePointerCast(MI->getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreatePointerCast(MI->getArgOperand(1), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(MI->getArgOperand(2), IntptrTy, false)});
    MI->eraseFromParent();
    Res = true;
  } else
    llvm_unreachable("Unsupported mem intrinsic type");
  return Res;
}

int EfficiencySanitizer::getMemoryAccessFuncIndex(Value *Addr,
                                                  const DataLayout &DL) {
  Type *OrigPtrTy = Addr->getType();
  Type *OrigTy = cast<PointerType>(OrigPtrTy)->getElementType();
  assert(OrigTy->isSized());
  // The size is always a multiple of 8.
  uint32_t TypeSizeBytes = DL.getTypeStoreSizeInBits(OrigTy) / 8;
  if (TypeSizeBytes != 1 && TypeSizeBytes != 2 && TypeSizeBytes != 4 &&
      TypeSizeBytes != 8 && TypeSizeBytes != 16) {
    // Irregular sizes do not have per-size call targets.
    NumAccessesWithIrregularSize++;
    return -1;
  }
  size_t Idx = countTrailingZeros(TypeSizeBytes);
  assert(Idx < NumberOfAccessSizes);
  return Idx;
}

bool EfficiencySanitizer::instrumentFastpath(Instruction *I,
                                             const DataLayout &DL, bool IsStore,
                                             Value *Addr, unsigned Alignment) {
  if (Options.ToolType == EfficiencySanitizerOptions::ESAN_CacheFrag) {
    return instrumentFastpathCacheFrag(I, DL, Addr, Alignment);
  }
  return false;
}

bool EfficiencySanitizer::instrumentFastpathCacheFrag(Instruction *I,
                                                      const DataLayout &DL,
                                                      Value *Addr,
                                                      unsigned Alignment) {
  // TODO(bruening): implement a fastpath for aligned accesses
  return false;
}
