//===- HWAddressSanitizer.cpp - detector of uninitialized reads -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file is a part of HWAddressSanitizer, an address sanity checker
/// based on tagged addressing.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/HWAddressSanitizer.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerCommon.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "hwasan"

static const char *const kHwasanModuleCtorName = "hwasan.module_ctor";
static const char *const kHwasanNoteName = "hwasan.note";
static const char *const kHwasanInitName = "__hwasan_init";
static const char *const kHwasanPersonalityThunkName =
    "__hwasan_personality_thunk";

static const char *const kHwasanShadowMemoryDynamicAddress =
    "__hwasan_shadow_memory_dynamic_address";

// Accesses sizes are powers of two: 1, 2, 4, 8, 16.
static const size_t kNumberOfAccessSizes = 5;

static const size_t kDefaultShadowScale = 4;
static const uint64_t kDynamicShadowSentinel =
    std::numeric_limits<uint64_t>::max();
static const unsigned kPointerTagShift = 56;

static const unsigned kShadowBaseAlignment = 32;

static cl::opt<std::string> ClMemoryAccessCallbackPrefix(
    "hwasan-memory-access-callback-prefix",
    cl::desc("Prefix for memory access callbacks"), cl::Hidden,
    cl::init("__hwasan_"));

static cl::opt<bool>
    ClInstrumentWithCalls("hwasan-instrument-with-calls",
                cl::desc("instrument reads and writes with callbacks"),
                cl::Hidden, cl::init(false));

static cl::opt<bool> ClInstrumentReads("hwasan-instrument-reads",
                                       cl::desc("instrument read instructions"),
                                       cl::Hidden, cl::init(true));

static cl::opt<bool> ClInstrumentWrites(
    "hwasan-instrument-writes", cl::desc("instrument write instructions"),
    cl::Hidden, cl::init(true));

static cl::opt<bool> ClInstrumentAtomics(
    "hwasan-instrument-atomics",
    cl::desc("instrument atomic instructions (rmw, cmpxchg)"), cl::Hidden,
    cl::init(true));

static cl::opt<bool> ClInstrumentByval("hwasan-instrument-byval",
                                       cl::desc("instrument byval arguments"),
                                       cl::Hidden, cl::init(true));

static cl::opt<bool> ClRecover(
    "hwasan-recover",
    cl::desc("Enable recovery mode (continue-after-error)."),
    cl::Hidden, cl::init(false));

static cl::opt<bool> ClInstrumentStack("hwasan-instrument-stack",
                                       cl::desc("instrument stack (allocas)"),
                                       cl::Hidden, cl::init(true));

static cl::opt<bool> ClUARRetagToZero(
    "hwasan-uar-retag-to-zero",
    cl::desc("Clear alloca tags before returning from the function to allow "
             "non-instrumented and instrumented function calls mix. When set "
             "to false, allocas are retagged before returning from the "
             "function to detect use after return."),
    cl::Hidden, cl::init(true));

static cl::opt<bool> ClGenerateTagsWithCalls(
    "hwasan-generate-tags-with-calls",
    cl::desc("generate new tags with runtime library calls"), cl::Hidden,
    cl::init(false));

static cl::opt<bool> ClGlobals("hwasan-globals", cl::desc("Instrument globals"),
                               cl::Hidden, cl::init(false), cl::ZeroOrMore);

static cl::opt<int> ClMatchAllTag(
    "hwasan-match-all-tag",
    cl::desc("don't report bad accesses via pointers with this tag"),
    cl::Hidden, cl::init(-1));

static cl::opt<bool> ClEnableKhwasan(
    "hwasan-kernel",
    cl::desc("Enable KernelHWAddressSanitizer instrumentation"),
    cl::Hidden, cl::init(false));

// These flags allow to change the shadow mapping and control how shadow memory
// is accessed. The shadow mapping looks like:
//    Shadow = (Mem >> scale) + offset

static cl::opt<uint64_t>
    ClMappingOffset("hwasan-mapping-offset",
                    cl::desc("HWASan shadow mapping offset [EXPERIMENTAL]"),
                    cl::Hidden, cl::init(0));

static cl::opt<bool>
    ClWithIfunc("hwasan-with-ifunc",
                cl::desc("Access dynamic shadow through an ifunc global on "
                         "platforms that support this"),
                cl::Hidden, cl::init(false));

static cl::opt<bool> ClWithTls(
    "hwasan-with-tls",
    cl::desc("Access dynamic shadow through an thread-local pointer on "
             "platforms that support this"),
    cl::Hidden, cl::init(true));

static cl::opt<bool>
    ClRecordStackHistory("hwasan-record-stack-history",
                         cl::desc("Record stack frames with tagged allocations "
                                  "in a thread-local ring buffer"),
                         cl::Hidden, cl::init(true));
static cl::opt<bool>
    ClInstrumentMemIntrinsics("hwasan-instrument-mem-intrinsics",
                              cl::desc("instrument memory intrinsics"),
                              cl::Hidden, cl::init(true));

static cl::opt<bool>
    ClInstrumentLandingPads("hwasan-instrument-landing-pads",
                            cl::desc("instrument landing pads"), cl::Hidden,
                            cl::init(false), cl::ZeroOrMore);

static cl::opt<bool> ClUseShortGranules(
    "hwasan-use-short-granules",
    cl::desc("use short granules in allocas and outlined checks"), cl::Hidden,
    cl::init(false), cl::ZeroOrMore);

static cl::opt<bool> ClInstrumentPersonalityFunctions(
    "hwasan-instrument-personality-functions",
    cl::desc("instrument personality functions"), cl::Hidden, cl::init(false),
    cl::ZeroOrMore);

static cl::opt<bool> ClInlineAllChecks("hwasan-inline-all-checks",
                                       cl::desc("inline all checks"),
                                       cl::Hidden, cl::init(false));

namespace {

/// An instrumentation pass implementing detection of addressability bugs
/// using tagged pointers.
class HWAddressSanitizer {
public:
  explicit HWAddressSanitizer(Module &M, bool CompileKernel = false,
                              bool Recover = false) : M(M) {
    this->Recover = ClRecover.getNumOccurrences() > 0 ? ClRecover : Recover;
    this->CompileKernel = ClEnableKhwasan.getNumOccurrences() > 0 ?
        ClEnableKhwasan : CompileKernel;

    initializeModule();
  }

  bool sanitizeFunction(Function &F);
  void initializeModule();

  void initializeCallbacks(Module &M);

  Value *getDynamicShadowIfunc(IRBuilder<> &IRB);
  Value *getDynamicShadowNonTls(IRBuilder<> &IRB);

  void untagPointerOperand(Instruction *I, Value *Addr);
  Value *shadowBase();
  Value *memToShadow(Value *Shadow, IRBuilder<> &IRB);
  void instrumentMemAccessInline(Value *Ptr, bool IsWrite,
                                 unsigned AccessSizeIndex,
                                 Instruction *InsertBefore);
  void instrumentMemIntrinsic(MemIntrinsic *MI);
  bool instrumentMemAccess(InterestingMemoryOperand &O);
  bool ignoreAccess(Value *Ptr);
  void getInterestingMemoryOperands(
      Instruction *I, SmallVectorImpl<InterestingMemoryOperand> &Interesting);

  bool isInterestingAlloca(const AllocaInst &AI);
  bool tagAlloca(IRBuilder<> &IRB, AllocaInst *AI, Value *Tag, size_t Size);
  Value *tagPointer(IRBuilder<> &IRB, Type *Ty, Value *PtrLong, Value *Tag);
  Value *untagPointer(IRBuilder<> &IRB, Value *PtrLong);
  bool instrumentStack(
      SmallVectorImpl<AllocaInst *> &Allocas,
      DenseMap<AllocaInst *, std::vector<DbgVariableIntrinsic *>> &AllocaDbgMap,
      SmallVectorImpl<Instruction *> &RetVec, Value *StackTag);
  Value *readRegister(IRBuilder<> &IRB, StringRef Name);
  bool instrumentLandingPads(SmallVectorImpl<Instruction *> &RetVec);
  Value *getNextTagWithCall(IRBuilder<> &IRB);
  Value *getStackBaseTag(IRBuilder<> &IRB);
  Value *getAllocaTag(IRBuilder<> &IRB, Value *StackTag, AllocaInst *AI,
                     unsigned AllocaNo);
  Value *getUARTag(IRBuilder<> &IRB, Value *StackTag);

  Value *getHwasanThreadSlotPtr(IRBuilder<> &IRB, Type *Ty);
  void emitPrologue(IRBuilder<> &IRB, bool WithFrameRecord);

  void instrumentGlobal(GlobalVariable *GV, uint8_t Tag);
  void instrumentGlobals();

  void instrumentPersonalityFunctions();

private:
  LLVMContext *C;
  Module &M;
  Triple TargetTriple;
  FunctionCallee HWAsanMemmove, HWAsanMemcpy, HWAsanMemset;
  FunctionCallee HWAsanHandleVfork;

  /// This struct defines the shadow mapping using the rule:
  ///   shadow = (mem >> Scale) + Offset.
  /// If InGlobal is true, then
  ///   extern char __hwasan_shadow[];
  ///   shadow = (mem >> Scale) + &__hwasan_shadow
  /// If InTls is true, then
  ///   extern char *__hwasan_tls;
  ///   shadow = (mem>>Scale) + align_up(__hwasan_shadow, kShadowBaseAlignment)
  struct ShadowMapping {
    int Scale;
    uint64_t Offset;
    bool InGlobal;
    bool InTls;

    void init(Triple &TargetTriple);
    unsigned getObjectAlignment() const { return 1U << Scale; }
  };
  ShadowMapping Mapping;

  Type *VoidTy = Type::getVoidTy(M.getContext());
  Type *IntptrTy;
  Type *Int8PtrTy;
  Type *Int8Ty;
  Type *Int32Ty;
  Type *Int64Ty = Type::getInt64Ty(M.getContext());

  bool CompileKernel;
  bool Recover;
  bool UseShortGranules;
  bool InstrumentLandingPads;

  Function *HwasanCtorFunction;

  FunctionCallee HwasanMemoryAccessCallback[2][kNumberOfAccessSizes];
  FunctionCallee HwasanMemoryAccessCallbackSized[2];

  FunctionCallee HwasanTagMemoryFunc;
  FunctionCallee HwasanGenerateTagFunc;

  Constant *ShadowGlobal;

  Value *LocalDynamicShadow = nullptr;
  Value *StackBaseTag = nullptr;
  GlobalValue *ThreadPtrGlobal = nullptr;
};

class HWAddressSanitizerLegacyPass : public FunctionPass {
public:
  // Pass identification, replacement for typeid.
  static char ID;

  explicit HWAddressSanitizerLegacyPass(bool CompileKernel = false,
                                        bool Recover = false)
      : FunctionPass(ID), CompileKernel(CompileKernel), Recover(Recover) {
    initializeHWAddressSanitizerLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "HWAddressSanitizer"; }

  bool doInitialization(Module &M) override {
    HWASan = std::make_unique<HWAddressSanitizer>(M, CompileKernel, Recover);
    return true;
  }

  bool runOnFunction(Function &F) override {
    return HWASan->sanitizeFunction(F);
  }

  bool doFinalization(Module &M) override {
    HWASan.reset();
    return false;
  }

private:
  std::unique_ptr<HWAddressSanitizer> HWASan;
  bool CompileKernel;
  bool Recover;
};

} // end anonymous namespace

char HWAddressSanitizerLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(
    HWAddressSanitizerLegacyPass, "hwasan",
    "HWAddressSanitizer: detect memory bugs using tagged addressing.", false,
    false)
INITIALIZE_PASS_END(
    HWAddressSanitizerLegacyPass, "hwasan",
    "HWAddressSanitizer: detect memory bugs using tagged addressing.", false,
    false)

FunctionPass *llvm::createHWAddressSanitizerLegacyPassPass(bool CompileKernel,
                                                           bool Recover) {
  assert(!CompileKernel || Recover);
  return new HWAddressSanitizerLegacyPass(CompileKernel, Recover);
}

HWAddressSanitizerPass::HWAddressSanitizerPass(bool CompileKernel, bool Recover)
    : CompileKernel(CompileKernel), Recover(Recover) {}

PreservedAnalyses HWAddressSanitizerPass::run(Module &M,
                                              ModuleAnalysisManager &MAM) {
  HWAddressSanitizer HWASan(M, CompileKernel, Recover);
  bool Modified = false;
  for (Function &F : M)
    Modified |= HWASan.sanitizeFunction(F);
  if (Modified)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

/// Module-level initialization.
///
/// inserts a call to __hwasan_init to the module's constructor list.
void HWAddressSanitizer::initializeModule() {
  LLVM_DEBUG(dbgs() << "Init " << M.getName() << "\n");
  auto &DL = M.getDataLayout();

  TargetTriple = Triple(M.getTargetTriple());

  Mapping.init(TargetTriple);

  C = &(M.getContext());
  IRBuilder<> IRB(*C);
  IntptrTy = IRB.getIntPtrTy(DL);
  Int8PtrTy = IRB.getInt8PtrTy();
  Int8Ty = IRB.getInt8Ty();
  Int32Ty = IRB.getInt32Ty();

  HwasanCtorFunction = nullptr;

  // Older versions of Android do not have the required runtime support for
  // short granules, global or personality function instrumentation. On other
  // platforms we currently require using the latest version of the runtime.
  bool NewRuntime =
      !TargetTriple.isAndroid() || !TargetTriple.isAndroidVersionLT(30);

  UseShortGranules =
      ClUseShortGranules.getNumOccurrences() ? ClUseShortGranules : NewRuntime;

  // If we don't have personality function support, fall back to landing pads.
  InstrumentLandingPads = ClInstrumentLandingPads.getNumOccurrences()
                              ? ClInstrumentLandingPads
                              : !NewRuntime;

  if (!CompileKernel) {
    std::tie(HwasanCtorFunction, std::ignore) =
        getOrCreateSanitizerCtorAndInitFunctions(
            M, kHwasanModuleCtorName, kHwasanInitName,
            /*InitArgTypes=*/{},
            /*InitArgs=*/{},
            // This callback is invoked when the functions are created the first
            // time. Hook them into the global ctors list in that case:
            [&](Function *Ctor, FunctionCallee) {
              Comdat *CtorComdat = M.getOrInsertComdat(kHwasanModuleCtorName);
              Ctor->setComdat(CtorComdat);
              appendToGlobalCtors(M, Ctor, 0, Ctor);
            });

    bool InstrumentGlobals =
        ClGlobals.getNumOccurrences() ? ClGlobals : NewRuntime;
    if (InstrumentGlobals)
      instrumentGlobals();

    bool InstrumentPersonalityFunctions =
        ClInstrumentPersonalityFunctions.getNumOccurrences()
            ? ClInstrumentPersonalityFunctions
            : NewRuntime;
    if (InstrumentPersonalityFunctions)
      instrumentPersonalityFunctions();
  }

  if (!TargetTriple.isAndroid()) {
    Constant *C = M.getOrInsertGlobal("__hwasan_tls", IntptrTy, [&] {
      auto *GV = new GlobalVariable(M, IntptrTy, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage, nullptr,
                                    "__hwasan_tls", nullptr,
                                    GlobalVariable::InitialExecTLSModel);
      appendToCompilerUsed(M, GV);
      return GV;
    });
    ThreadPtrGlobal = cast<GlobalVariable>(C);
  }
}

void HWAddressSanitizer::initializeCallbacks(Module &M) {
  IRBuilder<> IRB(*C);
  for (size_t AccessIsWrite = 0; AccessIsWrite <= 1; AccessIsWrite++) {
    const std::string TypeStr = AccessIsWrite ? "store" : "load";
    const std::string EndingStr = Recover ? "_noabort" : "";

    HwasanMemoryAccessCallbackSized[AccessIsWrite] = M.getOrInsertFunction(
        ClMemoryAccessCallbackPrefix + TypeStr + "N" + EndingStr,
        FunctionType::get(IRB.getVoidTy(), {IntptrTy, IntptrTy}, false));

    for (size_t AccessSizeIndex = 0; AccessSizeIndex < kNumberOfAccessSizes;
         AccessSizeIndex++) {
      HwasanMemoryAccessCallback[AccessIsWrite][AccessSizeIndex] =
          M.getOrInsertFunction(
              ClMemoryAccessCallbackPrefix + TypeStr +
                  itostr(1ULL << AccessSizeIndex) + EndingStr,
              FunctionType::get(IRB.getVoidTy(), {IntptrTy}, false));
    }
  }

  HwasanTagMemoryFunc = M.getOrInsertFunction(
      "__hwasan_tag_memory", IRB.getVoidTy(), Int8PtrTy, Int8Ty, IntptrTy);
  HwasanGenerateTagFunc =
      M.getOrInsertFunction("__hwasan_generate_tag", Int8Ty);

  ShadowGlobal = M.getOrInsertGlobal("__hwasan_shadow",
                                     ArrayType::get(IRB.getInt8Ty(), 0));

  const std::string MemIntrinCallbackPrefix =
      CompileKernel ? std::string("") : ClMemoryAccessCallbackPrefix;
  HWAsanMemmove = M.getOrInsertFunction(MemIntrinCallbackPrefix + "memmove",
                                        IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                                        IRB.getInt8PtrTy(), IntptrTy);
  HWAsanMemcpy = M.getOrInsertFunction(MemIntrinCallbackPrefix + "memcpy",
                                       IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                                       IRB.getInt8PtrTy(), IntptrTy);
  HWAsanMemset = M.getOrInsertFunction(MemIntrinCallbackPrefix + "memset",
                                       IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                                       IRB.getInt32Ty(), IntptrTy);

  HWAsanHandleVfork =
      M.getOrInsertFunction("__hwasan_handle_vfork", IRB.getVoidTy(), IntptrTy);
}

Value *HWAddressSanitizer::getDynamicShadowIfunc(IRBuilder<> &IRB) {
  // An empty inline asm with input reg == output reg.
  // An opaque no-op cast, basically.
  InlineAsm *Asm = InlineAsm::get(
      FunctionType::get(Int8PtrTy, {ShadowGlobal->getType()}, false),
      StringRef(""), StringRef("=r,0"),
      /*hasSideEffects=*/false);
  return IRB.CreateCall(Asm, {ShadowGlobal}, ".hwasan.shadow");
}

Value *HWAddressSanitizer::getDynamicShadowNonTls(IRBuilder<> &IRB) {
  // Generate code only when dynamic addressing is needed.
  if (Mapping.Offset != kDynamicShadowSentinel)
    return nullptr;

  if (Mapping.InGlobal) {
    return getDynamicShadowIfunc(IRB);
  } else {
    Value *GlobalDynamicAddress =
        IRB.GetInsertBlock()->getParent()->getParent()->getOrInsertGlobal(
            kHwasanShadowMemoryDynamicAddress, Int8PtrTy);
    return IRB.CreateLoad(Int8PtrTy, GlobalDynamicAddress);
  }
}

bool HWAddressSanitizer::ignoreAccess(Value *Ptr) {
  // Do not instrument acesses from different address spaces; we cannot deal
  // with them.
  Type *PtrTy = cast<PointerType>(Ptr->getType()->getScalarType());
  if (PtrTy->getPointerAddressSpace() != 0)
    return true;

  // Ignore swifterror addresses.
  // swifterror memory addresses are mem2reg promoted by instruction
  // selection. As such they cannot have regular uses like an instrumentation
  // function and it makes no sense to track them as memory.
  if (Ptr->isSwiftError())
    return true;

  return false;
}

void HWAddressSanitizer::getInterestingMemoryOperands(
    Instruction *I, SmallVectorImpl<InterestingMemoryOperand> &Interesting) {
  // Skip memory accesses inserted by another instrumentation.
  if (I->hasMetadata("nosanitize"))
    return;

  // Do not instrument the load fetching the dynamic shadow address.
  if (LocalDynamicShadow == I)
    return;

  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    if (!ClInstrumentReads || ignoreAccess(LI->getPointerOperand()))
      return;
    Interesting.emplace_back(I, LI->getPointerOperandIndex(), false,
                             LI->getType(), LI->getAlign());
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    if (!ClInstrumentWrites || ignoreAccess(SI->getPointerOperand()))
      return;
    Interesting.emplace_back(I, SI->getPointerOperandIndex(), true,
                             SI->getValueOperand()->getType(), SI->getAlign());
  } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
    if (!ClInstrumentAtomics || ignoreAccess(RMW->getPointerOperand()))
      return;
    Interesting.emplace_back(I, RMW->getPointerOperandIndex(), true,
                             RMW->getValOperand()->getType(), None);
  } else if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(I)) {
    if (!ClInstrumentAtomics || ignoreAccess(XCHG->getPointerOperand()))
      return;
    Interesting.emplace_back(I, XCHG->getPointerOperandIndex(), true,
                             XCHG->getCompareOperand()->getType(), None);
  } else if (auto CI = dyn_cast<CallInst>(I)) {
    for (unsigned ArgNo = 0; ArgNo < CI->getNumArgOperands(); ArgNo++) {
      if (!ClInstrumentByval || !CI->isByValArgument(ArgNo) ||
          ignoreAccess(CI->getArgOperand(ArgNo)))
        continue;
      Type *Ty = CI->getParamByValType(ArgNo);
      Interesting.emplace_back(I, ArgNo, false, Ty, Align(1));
    }
  }
}

static unsigned getPointerOperandIndex(Instruction *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->getPointerOperandIndex();
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->getPointerOperandIndex();
  if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I))
    return RMW->getPointerOperandIndex();
  if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(I))
    return XCHG->getPointerOperandIndex();
  report_fatal_error("Unexpected instruction");
  return -1;
}

static size_t TypeSizeToSizeIndex(uint32_t TypeSize) {
  size_t Res = countTrailingZeros(TypeSize / 8);
  assert(Res < kNumberOfAccessSizes);
  return Res;
}

void HWAddressSanitizer::untagPointerOperand(Instruction *I, Value *Addr) {
  if (TargetTriple.isAArch64())
    return;

  IRBuilder<> IRB(I);
  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);
  Value *UntaggedPtr =
      IRB.CreateIntToPtr(untagPointer(IRB, AddrLong), Addr->getType());
  I->setOperand(getPointerOperandIndex(I), UntaggedPtr);
}

Value *HWAddressSanitizer::shadowBase() {
  if (LocalDynamicShadow)
    return LocalDynamicShadow;
  return ConstantExpr::getIntToPtr(ConstantInt::get(IntptrTy, Mapping.Offset),
                                   Int8PtrTy);
}

Value *HWAddressSanitizer::memToShadow(Value *Mem, IRBuilder<> &IRB) {
  // Mem >> Scale
  Value *Shadow = IRB.CreateLShr(Mem, Mapping.Scale);
  if (Mapping.Offset == 0)
    return IRB.CreateIntToPtr(Shadow, Int8PtrTy);
  // (Mem >> Scale) + Offset
  return IRB.CreateGEP(Int8Ty, shadowBase(), Shadow);
}

void HWAddressSanitizer::instrumentMemAccessInline(Value *Ptr, bool IsWrite,
                                                   unsigned AccessSizeIndex,
                                                   Instruction *InsertBefore) {
  const int64_t AccessInfo = Recover * 0x20 + IsWrite * 0x10 + AccessSizeIndex;
  IRBuilder<> IRB(InsertBefore);

  if (!ClInlineAllChecks && TargetTriple.isAArch64() &&
      TargetTriple.isOSBinFormatELF() && !Recover) {
    Module *M = IRB.GetInsertBlock()->getParent()->getParent();
    Ptr = IRB.CreateBitCast(Ptr, Int8PtrTy);
    IRB.CreateCall(Intrinsic::getDeclaration(
                       M, UseShortGranules
                              ? Intrinsic::hwasan_check_memaccess_shortgranules
                              : Intrinsic::hwasan_check_memaccess),
                   {shadowBase(), Ptr, ConstantInt::get(Int32Ty, AccessInfo)});
    return;
  }

  Value *PtrLong = IRB.CreatePointerCast(Ptr, IntptrTy);
  Value *PtrTag = IRB.CreateTrunc(IRB.CreateLShr(PtrLong, kPointerTagShift),
                                  IRB.getInt8Ty());
  Value *AddrLong = untagPointer(IRB, PtrLong);
  Value *Shadow = memToShadow(AddrLong, IRB);
  Value *MemTag = IRB.CreateLoad(Int8Ty, Shadow);
  Value *TagMismatch = IRB.CreateICmpNE(PtrTag, MemTag);

  int matchAllTag = ClMatchAllTag.getNumOccurrences() > 0 ?
      ClMatchAllTag : (CompileKernel ? 0xFF : -1);
  if (matchAllTag != -1) {
    Value *TagNotIgnored = IRB.CreateICmpNE(PtrTag,
        ConstantInt::get(PtrTag->getType(), matchAllTag));
    TagMismatch = IRB.CreateAnd(TagMismatch, TagNotIgnored);
  }

  Instruction *CheckTerm =
      SplitBlockAndInsertIfThen(TagMismatch, InsertBefore, false,
                                MDBuilder(*C).createBranchWeights(1, 100000));

  IRB.SetInsertPoint(CheckTerm);
  Value *OutOfShortGranuleTagRange =
      IRB.CreateICmpUGT(MemTag, ConstantInt::get(Int8Ty, 15));
  Instruction *CheckFailTerm =
      SplitBlockAndInsertIfThen(OutOfShortGranuleTagRange, CheckTerm, !Recover,
                                MDBuilder(*C).createBranchWeights(1, 100000));

  IRB.SetInsertPoint(CheckTerm);
  Value *PtrLowBits = IRB.CreateTrunc(IRB.CreateAnd(PtrLong, 15), Int8Ty);
  PtrLowBits = IRB.CreateAdd(
      PtrLowBits, ConstantInt::get(Int8Ty, (1 << AccessSizeIndex) - 1));
  Value *PtrLowBitsOOB = IRB.CreateICmpUGE(PtrLowBits, MemTag);
  SplitBlockAndInsertIfThen(PtrLowBitsOOB, CheckTerm, false,
                            MDBuilder(*C).createBranchWeights(1, 100000),
                            nullptr, nullptr, CheckFailTerm->getParent());

  IRB.SetInsertPoint(CheckTerm);
  Value *InlineTagAddr = IRB.CreateOr(AddrLong, 15);
  InlineTagAddr = IRB.CreateIntToPtr(InlineTagAddr, Int8PtrTy);
  Value *InlineTag = IRB.CreateLoad(Int8Ty, InlineTagAddr);
  Value *InlineTagMismatch = IRB.CreateICmpNE(PtrTag, InlineTag);
  SplitBlockAndInsertIfThen(InlineTagMismatch, CheckTerm, false,
                            MDBuilder(*C).createBranchWeights(1, 100000),
                            nullptr, nullptr, CheckFailTerm->getParent());

  IRB.SetInsertPoint(CheckFailTerm);
  InlineAsm *Asm;
  switch (TargetTriple.getArch()) {
    case Triple::x86_64:
      // The signal handler will find the data address in rdi.
      Asm = InlineAsm::get(
          FunctionType::get(IRB.getVoidTy(), {PtrLong->getType()}, false),
          "int3\nnopl " + itostr(0x40 + AccessInfo) + "(%rax)",
          "{rdi}",
          /*hasSideEffects=*/true);
      break;
    case Triple::aarch64:
    case Triple::aarch64_be:
      // The signal handler will find the data address in x0.
      Asm = InlineAsm::get(
          FunctionType::get(IRB.getVoidTy(), {PtrLong->getType()}, false),
          "brk #" + itostr(0x900 + AccessInfo),
          "{x0}",
          /*hasSideEffects=*/true);
      break;
    default:
      report_fatal_error("unsupported architecture");
  }
  IRB.CreateCall(Asm, PtrLong);
  if (Recover)
    cast<BranchInst>(CheckFailTerm)->setSuccessor(0, CheckTerm->getParent());
}

void HWAddressSanitizer::instrumentMemIntrinsic(MemIntrinsic *MI) {
  IRBuilder<> IRB(MI);
  if (isa<MemTransferInst>(MI)) {
    IRB.CreateCall(
        isa<MemMoveInst>(MI) ? HWAsanMemmove : HWAsanMemcpy,
        {IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
         IRB.CreatePointerCast(MI->getOperand(1), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)});
  } else if (isa<MemSetInst>(MI)) {
    IRB.CreateCall(
        HWAsanMemset,
        {IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(MI->getOperand(1), IRB.getInt32Ty(), false),
         IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false)});
  }
  MI->eraseFromParent();
}

bool HWAddressSanitizer::instrumentMemAccess(InterestingMemoryOperand &O) {
  Value *Addr = O.getPtr();

  LLVM_DEBUG(dbgs() << "Instrumenting: " << O.getInsn() << "\n");

  if (O.MaybeMask)
    return false; //FIXME

  IRBuilder<> IRB(O.getInsn());
  if (isPowerOf2_64(O.TypeSize) &&
      (O.TypeSize / 8 <= (1ULL << (kNumberOfAccessSizes - 1))) &&
      (!O.Alignment || *O.Alignment >= (1ULL << Mapping.Scale) ||
       *O.Alignment >= O.TypeSize / 8)) {
    size_t AccessSizeIndex = TypeSizeToSizeIndex(O.TypeSize);
    if (ClInstrumentWithCalls) {
      IRB.CreateCall(HwasanMemoryAccessCallback[O.IsWrite][AccessSizeIndex],
                     IRB.CreatePointerCast(Addr, IntptrTy));
    } else {
      instrumentMemAccessInline(Addr, O.IsWrite, AccessSizeIndex, O.getInsn());
    }
  } else {
    IRB.CreateCall(HwasanMemoryAccessCallbackSized[O.IsWrite],
                   {IRB.CreatePointerCast(Addr, IntptrTy),
                    ConstantInt::get(IntptrTy, O.TypeSize / 8)});
  }
  untagPointerOperand(O.getInsn(), Addr);

  return true;
}

static uint64_t getAllocaSizeInBytes(const AllocaInst &AI) {
  uint64_t ArraySize = 1;
  if (AI.isArrayAllocation()) {
    const ConstantInt *CI = dyn_cast<ConstantInt>(AI.getArraySize());
    assert(CI && "non-constant array size");
    ArraySize = CI->getZExtValue();
  }
  Type *Ty = AI.getAllocatedType();
  uint64_t SizeInBytes = AI.getModule()->getDataLayout().getTypeAllocSize(Ty);
  return SizeInBytes * ArraySize;
}

bool HWAddressSanitizer::tagAlloca(IRBuilder<> &IRB, AllocaInst *AI,
                                   Value *Tag, size_t Size) {
  size_t AlignedSize = alignTo(Size, Mapping.getObjectAlignment());
  if (!UseShortGranules)
    Size = AlignedSize;

  Value *JustTag = IRB.CreateTrunc(Tag, IRB.getInt8Ty());
  if (ClInstrumentWithCalls) {
    IRB.CreateCall(HwasanTagMemoryFunc,
                   {IRB.CreatePointerCast(AI, Int8PtrTy), JustTag,
                    ConstantInt::get(IntptrTy, AlignedSize)});
  } else {
    size_t ShadowSize = Size >> Mapping.Scale;
    Value *ShadowPtr = memToShadow(IRB.CreatePointerCast(AI, IntptrTy), IRB);
    // If this memset is not inlined, it will be intercepted in the hwasan
    // runtime library. That's OK, because the interceptor skips the checks if
    // the address is in the shadow region.
    // FIXME: the interceptor is not as fast as real memset. Consider lowering
    // llvm.memset right here into either a sequence of stores, or a call to
    // hwasan_tag_memory.
    if (ShadowSize)
      IRB.CreateMemSet(ShadowPtr, JustTag, ShadowSize, Align(1));
    if (Size != AlignedSize) {
      IRB.CreateStore(
          ConstantInt::get(Int8Ty, Size % Mapping.getObjectAlignment()),
          IRB.CreateConstGEP1_32(Int8Ty, ShadowPtr, ShadowSize));
      IRB.CreateStore(JustTag, IRB.CreateConstGEP1_32(
                                   Int8Ty, IRB.CreateBitCast(AI, Int8PtrTy),
                                   AlignedSize - 1));
    }
  }
  return true;
}

static unsigned RetagMask(unsigned AllocaNo) {
  // A list of 8-bit numbers that have at most one run of non-zero bits.
  // x = x ^ (mask << 56) can be encoded as a single armv8 instruction for these
  // masks.
  // The list does not include the value 255, which is used for UAR.
  //
  // Because we are more likely to use earlier elements of this list than later
  // ones, it is sorted in increasing order of probability of collision with a
  // mask allocated (temporally) nearby. The program that generated this list
  // can be found at:
  // https://github.com/google/sanitizers/blob/master/hwaddress-sanitizer/sort_masks.py
  static unsigned FastMasks[] = {0,  128, 64,  192, 32,  96,  224, 112, 240,
                                 48, 16,  120, 248, 56,  24,  8,   124, 252,
                                 60, 28,  12,  4,   126, 254, 62,  30,  14,
                                 6,  2,   127, 63,  31,  15,  7,   3,   1};
  return FastMasks[AllocaNo % (sizeof(FastMasks) / sizeof(FastMasks[0]))];
}

Value *HWAddressSanitizer::getNextTagWithCall(IRBuilder<> &IRB) {
  return IRB.CreateZExt(IRB.CreateCall(HwasanGenerateTagFunc), IntptrTy);
}

Value *HWAddressSanitizer::getStackBaseTag(IRBuilder<> &IRB) {
  if (ClGenerateTagsWithCalls)
    return getNextTagWithCall(IRB);
  if (StackBaseTag)
    return StackBaseTag;
  // FIXME: use addressofreturnaddress (but implement it in aarch64 backend
  // first).
  Module *M = IRB.GetInsertBlock()->getParent()->getParent();
  auto GetStackPointerFn = Intrinsic::getDeclaration(
      M, Intrinsic::frameaddress,
      IRB.getInt8PtrTy(M->getDataLayout().getAllocaAddrSpace()));
  Value *StackPointer = IRB.CreateCall(
      GetStackPointerFn, {Constant::getNullValue(IRB.getInt32Ty())});

  // Extract some entropy from the stack pointer for the tags.
  // Take bits 20..28 (ASLR entropy) and xor with bits 0..8 (these differ
  // between functions).
  Value *StackPointerLong = IRB.CreatePointerCast(StackPointer, IntptrTy);
  Value *StackTag =
      IRB.CreateXor(StackPointerLong, IRB.CreateLShr(StackPointerLong, 20),
                    "hwasan.stack.base.tag");
  return StackTag;
}

Value *HWAddressSanitizer::getAllocaTag(IRBuilder<> &IRB, Value *StackTag,
                                        AllocaInst *AI, unsigned AllocaNo) {
  if (ClGenerateTagsWithCalls)
    return getNextTagWithCall(IRB);
  return IRB.CreateXor(StackTag,
                       ConstantInt::get(IntptrTy, RetagMask(AllocaNo)));
}

Value *HWAddressSanitizer::getUARTag(IRBuilder<> &IRB, Value *StackTag) {
  if (ClUARRetagToZero)
    return ConstantInt::get(IntptrTy, 0);
  if (ClGenerateTagsWithCalls)
    return getNextTagWithCall(IRB);
  return IRB.CreateXor(StackTag, ConstantInt::get(IntptrTy, 0xFFU));
}

// Add a tag to an address.
Value *HWAddressSanitizer::tagPointer(IRBuilder<> &IRB, Type *Ty,
                                      Value *PtrLong, Value *Tag) {
  Value *TaggedPtrLong;
  if (CompileKernel) {
    // Kernel addresses have 0xFF in the most significant byte.
    Value *ShiftedTag = IRB.CreateOr(
        IRB.CreateShl(Tag, kPointerTagShift),
        ConstantInt::get(IntptrTy, (1ULL << kPointerTagShift) - 1));
    TaggedPtrLong = IRB.CreateAnd(PtrLong, ShiftedTag);
  } else {
    // Userspace can simply do OR (tag << 56);
    Value *ShiftedTag = IRB.CreateShl(Tag, kPointerTagShift);
    TaggedPtrLong = IRB.CreateOr(PtrLong, ShiftedTag);
  }
  return IRB.CreateIntToPtr(TaggedPtrLong, Ty);
}

// Remove tag from an address.
Value *HWAddressSanitizer::untagPointer(IRBuilder<> &IRB, Value *PtrLong) {
  Value *UntaggedPtrLong;
  if (CompileKernel) {
    // Kernel addresses have 0xFF in the most significant byte.
    UntaggedPtrLong = IRB.CreateOr(PtrLong,
        ConstantInt::get(PtrLong->getType(), 0xFFULL << kPointerTagShift));
  } else {
    // Userspace addresses have 0x00.
    UntaggedPtrLong = IRB.CreateAnd(PtrLong,
        ConstantInt::get(PtrLong->getType(), ~(0xFFULL << kPointerTagShift)));
  }
  return UntaggedPtrLong;
}

Value *HWAddressSanitizer::getHwasanThreadSlotPtr(IRBuilder<> &IRB, Type *Ty) {
  Module *M = IRB.GetInsertBlock()->getParent()->getParent();
  if (TargetTriple.isAArch64() && TargetTriple.isAndroid()) {
    // Android provides a fixed TLS slot for sanitizers. See TLS_SLOT_SANITIZER
    // in Bionic's libc/private/bionic_tls.h.
    Function *ThreadPointerFunc =
        Intrinsic::getDeclaration(M, Intrinsic::thread_pointer);
    Value *SlotPtr = IRB.CreatePointerCast(
        IRB.CreateConstGEP1_32(IRB.getInt8Ty(),
                               IRB.CreateCall(ThreadPointerFunc), 0x30),
        Ty->getPointerTo(0));
    return SlotPtr;
  }
  if (ThreadPtrGlobal)
    return ThreadPtrGlobal;


  return nullptr;
}

void HWAddressSanitizer::emitPrologue(IRBuilder<> &IRB, bool WithFrameRecord) {
  if (!Mapping.InTls) {
    LocalDynamicShadow = getDynamicShadowNonTls(IRB);
    return;
  }

  if (!WithFrameRecord && TargetTriple.isAndroid()) {
    LocalDynamicShadow = getDynamicShadowIfunc(IRB);
    return;
  }

  Value *SlotPtr = getHwasanThreadSlotPtr(IRB, IntptrTy);
  assert(SlotPtr);

  Value *ThreadLong = IRB.CreateLoad(IntptrTy, SlotPtr);
  // Extract the address field from ThreadLong. Unnecessary on AArch64 with TBI.
  Value *ThreadLongMaybeUntagged =
      TargetTriple.isAArch64() ? ThreadLong : untagPointer(IRB, ThreadLong);

  if (WithFrameRecord) {
    Function *F = IRB.GetInsertBlock()->getParent();
    StackBaseTag = IRB.CreateAShr(ThreadLong, 3);

    // Prepare ring buffer data.
    Value *PC;
    if (TargetTriple.getArch() == Triple::aarch64)
      PC = readRegister(IRB, "pc");
    else
      PC = IRB.CreatePtrToInt(F, IntptrTy);
    Module *M = F->getParent();
    auto GetStackPointerFn = Intrinsic::getDeclaration(
        M, Intrinsic::frameaddress,
        IRB.getInt8PtrTy(M->getDataLayout().getAllocaAddrSpace()));
    Value *SP = IRB.CreatePtrToInt(
        IRB.CreateCall(GetStackPointerFn,
                       {Constant::getNullValue(IRB.getInt32Ty())}),
        IntptrTy);
    // Mix SP and PC.
    // Assumptions:
    // PC is 0x0000PPPPPPPPPPPP  (48 bits are meaningful, others are zero)
    // SP is 0xsssssssssssSSSS0  (4 lower bits are zero)
    // We only really need ~20 lower non-zero bits (SSSS), so we mix like this:
    //       0xSSSSPPPPPPPPPPPP
    SP = IRB.CreateShl(SP, 44);

    // Store data to ring buffer.
    Value *RecordPtr =
        IRB.CreateIntToPtr(ThreadLongMaybeUntagged, IntptrTy->getPointerTo(0));
    IRB.CreateStore(IRB.CreateOr(PC, SP), RecordPtr);

    // Update the ring buffer. Top byte of ThreadLong defines the size of the
    // buffer in pages, it must be a power of two, and the start of the buffer
    // must be aligned by twice that much. Therefore wrap around of the ring
    // buffer is simply Addr &= ~((ThreadLong >> 56) << 12).
    // The use of AShr instead of LShr is due to
    //   https://bugs.llvm.org/show_bug.cgi?id=39030
    // Runtime library makes sure not to use the highest bit.
    Value *WrapMask = IRB.CreateXor(
        IRB.CreateShl(IRB.CreateAShr(ThreadLong, 56), 12, "", true, true),
        ConstantInt::get(IntptrTy, (uint64_t)-1));
    Value *ThreadLongNew = IRB.CreateAnd(
        IRB.CreateAdd(ThreadLong, ConstantInt::get(IntptrTy, 8)), WrapMask);
    IRB.CreateStore(ThreadLongNew, SlotPtr);
  }

  // Get shadow base address by aligning RecordPtr up.
  // Note: this is not correct if the pointer is already aligned.
  // Runtime library will make sure this never happens.
  LocalDynamicShadow = IRB.CreateAdd(
      IRB.CreateOr(
          ThreadLongMaybeUntagged,
          ConstantInt::get(IntptrTy, (1ULL << kShadowBaseAlignment) - 1)),
      ConstantInt::get(IntptrTy, 1), "hwasan.shadow");
  LocalDynamicShadow = IRB.CreateIntToPtr(LocalDynamicShadow, Int8PtrTy);
}

Value *HWAddressSanitizer::readRegister(IRBuilder<> &IRB, StringRef Name) {
  Module *M = IRB.GetInsertBlock()->getParent()->getParent();
  Function *ReadRegister =
      Intrinsic::getDeclaration(M, Intrinsic::read_register, IntptrTy);
  MDNode *MD = MDNode::get(*C, {MDString::get(*C, Name)});
  Value *Args[] = {MetadataAsValue::get(*C, MD)};
  return IRB.CreateCall(ReadRegister, Args);
}

bool HWAddressSanitizer::instrumentLandingPads(
    SmallVectorImpl<Instruction *> &LandingPadVec) {
  for (auto *LP : LandingPadVec) {
    IRBuilder<> IRB(LP->getNextNode());
    IRB.CreateCall(
        HWAsanHandleVfork,
        {readRegister(IRB, (TargetTriple.getArch() == Triple::x86_64) ? "rsp"
                                                                      : "sp")});
  }
  return true;
}

bool HWAddressSanitizer::instrumentStack(
    SmallVectorImpl<AllocaInst *> &Allocas,
    DenseMap<AllocaInst *, std::vector<DbgVariableIntrinsic *>> &AllocaDbgMap,
    SmallVectorImpl<Instruction *> &RetVec, Value *StackTag) {
  // Ideally, we want to calculate tagged stack base pointer, and rewrite all
  // alloca addresses using that. Unfortunately, offsets are not known yet
  // (unless we use ASan-style mega-alloca). Instead we keep the base tag in a
  // temp, shift-OR it into each alloca address and xor with the retag mask.
  // This generates one extra instruction per alloca use.
  for (unsigned N = 0; N < Allocas.size(); ++N) {
    auto *AI = Allocas[N];
    IRBuilder<> IRB(AI->getNextNode());

    // Replace uses of the alloca with tagged address.
    Value *Tag = getAllocaTag(IRB, StackTag, AI, N);
    Value *AILong = IRB.CreatePointerCast(AI, IntptrTy);
    Value *Replacement = tagPointer(IRB, AI->getType(), AILong, Tag);
    std::string Name =
        AI->hasName() ? AI->getName().str() : "alloca." + itostr(N);
    Replacement->setName(Name + ".hwasan");

    AI->replaceUsesWithIf(Replacement,
                          [AILong](Use &U) { return U.getUser() != AILong; });

    for (auto *DDI : AllocaDbgMap.lookup(AI)) {
      // Prepend "tag_offset, N" to the dwarf expression.
      // Tag offset logically applies to the alloca pointer, and it makes sense
      // to put it at the beginning of the expression.
      SmallVector<uint64_t, 8> NewOps = {dwarf::DW_OP_LLVM_tag_offset,
                                         RetagMask(N)};
      DDI->setArgOperand(
          2, MetadataAsValue::get(*C, DIExpression::prependOpcodes(
                                          DDI->getExpression(), NewOps)));
    }

    size_t Size = getAllocaSizeInBytes(*AI);
    tagAlloca(IRB, AI, Tag, Size);

    for (auto RI : RetVec) {
      IRB.SetInsertPoint(RI);

      // Re-tag alloca memory with the special UAR tag.
      Value *Tag = getUARTag(IRB, StackTag);
      tagAlloca(IRB, AI, Tag, alignTo(Size, Mapping.getObjectAlignment()));
    }
  }

  return true;
}

bool HWAddressSanitizer::isInterestingAlloca(const AllocaInst &AI) {
  return (AI.getAllocatedType()->isSized() &&
          // FIXME: instrument dynamic allocas, too
          AI.isStaticAlloca() &&
          // alloca() may be called with 0 size, ignore it.
          getAllocaSizeInBytes(AI) > 0 &&
          // We are only interested in allocas not promotable to registers.
          // Promotable allocas are common under -O0.
          !isAllocaPromotable(&AI) &&
          // inalloca allocas are not treated as static, and we don't want
          // dynamic alloca instrumentation for them as well.
          !AI.isUsedWithInAlloca() &&
          // swifterror allocas are register promoted by ISel
          !AI.isSwiftError());
}

bool HWAddressSanitizer::sanitizeFunction(Function &F) {
  if (&F == HwasanCtorFunction)
    return false;

  if (!F.hasFnAttribute(Attribute::SanitizeHWAddress))
    return false;

  LLVM_DEBUG(dbgs() << "Function: " << F.getName() << "\n");

  SmallVector<InterestingMemoryOperand, 16> OperandsToInstrument;
  SmallVector<MemIntrinsic *, 16> IntrinToInstrument;
  SmallVector<AllocaInst*, 8> AllocasToInstrument;
  SmallVector<Instruction*, 8> RetVec;
  SmallVector<Instruction*, 8> LandingPadVec;
  DenseMap<AllocaInst *, std::vector<DbgVariableIntrinsic *>> AllocaDbgMap;
  for (auto &BB : F) {
    for (auto &Inst : BB) {
      if (ClInstrumentStack)
        if (AllocaInst *AI = dyn_cast<AllocaInst>(&Inst)) {
          if (isInterestingAlloca(*AI))
            AllocasToInstrument.push_back(AI);
          continue;
        }

      if (isa<ReturnInst>(Inst) || isa<ResumeInst>(Inst) ||
          isa<CleanupReturnInst>(Inst))
        RetVec.push_back(&Inst);

      if (auto *DDI = dyn_cast<DbgVariableIntrinsic>(&Inst))
        if (auto *Alloca =
                dyn_cast_or_null<AllocaInst>(DDI->getVariableLocation()))
          AllocaDbgMap[Alloca].push_back(DDI);

      if (InstrumentLandingPads && isa<LandingPadInst>(Inst))
        LandingPadVec.push_back(&Inst);

      getInterestingMemoryOperands(&Inst, OperandsToInstrument);

      if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(&Inst))
        IntrinToInstrument.push_back(MI);
    }
  }

  initializeCallbacks(*F.getParent());

  bool Changed = false;

  if (!LandingPadVec.empty())
    Changed |= instrumentLandingPads(LandingPadVec);

  if (AllocasToInstrument.empty() && F.hasPersonalityFn() &&
      F.getPersonalityFn()->getName() == kHwasanPersonalityThunkName) {
    // __hwasan_personality_thunk is a no-op for functions without an
    // instrumented stack, so we can drop it.
    F.setPersonalityFn(nullptr);
    Changed = true;
  }

  if (AllocasToInstrument.empty() && OperandsToInstrument.empty() &&
      IntrinToInstrument.empty())
    return Changed;

  assert(!LocalDynamicShadow);

  Instruction *InsertPt = &*F.getEntryBlock().begin();
  IRBuilder<> EntryIRB(InsertPt);
  emitPrologue(EntryIRB,
               /*WithFrameRecord*/ ClRecordStackHistory &&
                   !AllocasToInstrument.empty());

  if (!AllocasToInstrument.empty()) {
    Value *StackTag =
        ClGenerateTagsWithCalls ? nullptr : getStackBaseTag(EntryIRB);
    instrumentStack(AllocasToInstrument, AllocaDbgMap, RetVec, StackTag);
  }
  // Pad and align each of the allocas that we instrumented to stop small
  // uninteresting allocas from hiding in instrumented alloca's padding and so
  // that we have enough space to store real tags for short granules.
  DenseMap<AllocaInst *, AllocaInst *> AllocaToPaddedAllocaMap;
  for (AllocaInst *AI : AllocasToInstrument) {
    uint64_t Size = getAllocaSizeInBytes(*AI);
    uint64_t AlignedSize = alignTo(Size, Mapping.getObjectAlignment());
    AI->setAlignment(
        Align(std::max(AI->getAlignment(), Mapping.getObjectAlignment())));
    if (Size != AlignedSize) {
      Type *AllocatedType = AI->getAllocatedType();
      if (AI->isArrayAllocation()) {
        uint64_t ArraySize =
            cast<ConstantInt>(AI->getArraySize())->getZExtValue();
        AllocatedType = ArrayType::get(AllocatedType, ArraySize);
      }
      Type *TypeWithPadding = StructType::get(
          AllocatedType, ArrayType::get(Int8Ty, AlignedSize - Size));
      auto *NewAI = new AllocaInst(
          TypeWithPadding, AI->getType()->getAddressSpace(), nullptr, "", AI);
      NewAI->takeName(AI);
      NewAI->setAlignment(AI->getAlign());
      NewAI->setUsedWithInAlloca(AI->isUsedWithInAlloca());
      NewAI->setSwiftError(AI->isSwiftError());
      NewAI->copyMetadata(*AI);
      auto *Bitcast = new BitCastInst(NewAI, AI->getType(), "", AI);
      AI->replaceAllUsesWith(Bitcast);
      AllocaToPaddedAllocaMap[AI] = NewAI;
    }
  }

  if (!AllocaToPaddedAllocaMap.empty()) {
    for (auto &BB : F)
      for (auto &Inst : BB)
        if (auto *DVI = dyn_cast<DbgVariableIntrinsic>(&Inst))
          if (auto *AI =
                  dyn_cast_or_null<AllocaInst>(DVI->getVariableLocation()))
            if (auto *NewAI = AllocaToPaddedAllocaMap.lookup(AI))
              DVI->setArgOperand(
                  0, MetadataAsValue::get(*C, LocalAsMetadata::get(NewAI)));
    for (auto &P : AllocaToPaddedAllocaMap)
      P.first->eraseFromParent();
  }

  // If we split the entry block, move any allocas that were originally in the
  // entry block back into the entry block so that they aren't treated as
  // dynamic allocas.
  if (EntryIRB.GetInsertBlock() != &F.getEntryBlock()) {
    InsertPt = &*F.getEntryBlock().begin();
    for (auto II = EntryIRB.GetInsertBlock()->begin(),
              IE = EntryIRB.GetInsertBlock()->end();
         II != IE;) {
      Instruction *I = &*II++;
      if (auto *AI = dyn_cast<AllocaInst>(I))
        if (isa<ConstantInt>(AI->getArraySize()))
          I->moveBefore(InsertPt);
    }
  }

  for (auto &Operand : OperandsToInstrument)
    instrumentMemAccess(Operand);

  if (ClInstrumentMemIntrinsics && !IntrinToInstrument.empty()) {
    for (auto Inst : IntrinToInstrument)
      instrumentMemIntrinsic(cast<MemIntrinsic>(Inst));
  }

  LocalDynamicShadow = nullptr;
  StackBaseTag = nullptr;

  return true;
}

void HWAddressSanitizer::instrumentGlobal(GlobalVariable *GV, uint8_t Tag) {
  Constant *Initializer = GV->getInitializer();
  uint64_t SizeInBytes =
      M.getDataLayout().getTypeAllocSize(Initializer->getType());
  uint64_t NewSize = alignTo(SizeInBytes, Mapping.getObjectAlignment());
  if (SizeInBytes != NewSize) {
    // Pad the initializer out to the next multiple of 16 bytes and add the
    // required short granule tag.
    std::vector<uint8_t> Init(NewSize - SizeInBytes, 0);
    Init.back() = Tag;
    Constant *Padding = ConstantDataArray::get(*C, Init);
    Initializer = ConstantStruct::getAnon({Initializer, Padding});
  }

  auto *NewGV = new GlobalVariable(M, Initializer->getType(), GV->isConstant(),
                                   GlobalValue::ExternalLinkage, Initializer,
                                   GV->getName() + ".hwasan");
  NewGV->copyAttributesFrom(GV);
  NewGV->setLinkage(GlobalValue::PrivateLinkage);
  NewGV->copyMetadata(GV, 0);
  NewGV->setAlignment(
      MaybeAlign(std::max(GV->getAlignment(), Mapping.getObjectAlignment())));

  // It is invalid to ICF two globals that have different tags. In the case
  // where the size of the global is a multiple of the tag granularity the
  // contents of the globals may be the same but the tags (i.e. symbol values)
  // may be different, and the symbols are not considered during ICF. In the
  // case where the size is not a multiple of the granularity, the short granule
  // tags would discriminate two globals with different tags, but there would
  // otherwise be nothing stopping such a global from being incorrectly ICF'd
  // with an uninstrumented (i.e. tag 0) global that happened to have the short
  // granule tag in the last byte.
  NewGV->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

  // Descriptor format (assuming little-endian):
  // bytes 0-3: relative address of global
  // bytes 4-6: size of global (16MB ought to be enough for anyone, but in case
  // it isn't, we create multiple descriptors)
  // byte 7: tag
  auto *DescriptorTy = StructType::get(Int32Ty, Int32Ty);
  const uint64_t MaxDescriptorSize = 0xfffff0;
  for (uint64_t DescriptorPos = 0; DescriptorPos < SizeInBytes;
       DescriptorPos += MaxDescriptorSize) {
    auto *Descriptor =
        new GlobalVariable(M, DescriptorTy, true, GlobalValue::PrivateLinkage,
                           nullptr, GV->getName() + ".hwasan.descriptor");
    auto *GVRelPtr = ConstantExpr::getTrunc(
        ConstantExpr::getAdd(
            ConstantExpr::getSub(
                ConstantExpr::getPtrToInt(NewGV, Int64Ty),
                ConstantExpr::getPtrToInt(Descriptor, Int64Ty)),
            ConstantInt::get(Int64Ty, DescriptorPos)),
        Int32Ty);
    uint32_t Size = std::min(SizeInBytes - DescriptorPos, MaxDescriptorSize);
    auto *SizeAndTag = ConstantInt::get(Int32Ty, Size | (uint32_t(Tag) << 24));
    Descriptor->setComdat(NewGV->getComdat());
    Descriptor->setInitializer(ConstantStruct::getAnon({GVRelPtr, SizeAndTag}));
    Descriptor->setSection("hwasan_globals");
    Descriptor->setMetadata(LLVMContext::MD_associated,
                            MDNode::get(*C, ValueAsMetadata::get(NewGV)));
    appendToCompilerUsed(M, Descriptor);
  }

  Constant *Aliasee = ConstantExpr::getIntToPtr(
      ConstantExpr::getAdd(
          ConstantExpr::getPtrToInt(NewGV, Int64Ty),
          ConstantInt::get(Int64Ty, uint64_t(Tag) << kPointerTagShift)),
      GV->getType());
  auto *Alias = GlobalAlias::create(GV->getValueType(), GV->getAddressSpace(),
                                    GV->getLinkage(), "", Aliasee, &M);
  Alias->setVisibility(GV->getVisibility());
  Alias->takeName(GV);
  GV->replaceAllUsesWith(Alias);
  GV->eraseFromParent();
}

void HWAddressSanitizer::instrumentGlobals() {
  // Start by creating a note that contains pointers to the list of global
  // descriptors. Adding a note to the output file will cause the linker to
  // create a PT_NOTE program header pointing to the note that we can use to
  // find the descriptor list starting from the program headers. A function
  // provided by the runtime initializes the shadow memory for the globals by
  // accessing the descriptor list via the note. The dynamic loader needs to
  // call this function whenever a library is loaded.
  //
  // The reason why we use a note for this instead of a more conventional
  // approach of having a global constructor pass a descriptor list pointer to
  // the runtime is because of an order of initialization problem. With
  // constructors we can encounter the following problematic scenario:
  //
  // 1) library A depends on library B and also interposes one of B's symbols
  // 2) B's constructors are called before A's (as required for correctness)
  // 3) during construction, B accesses one of its "own" globals (actually
  //    interposed by A) and triggers a HWASAN failure due to the initialization
  //    for A not having happened yet
  //
  // Even without interposition it is possible to run into similar situations in
  // cases where two libraries mutually depend on each other.
  //
  // We only need one note per binary, so put everything for the note in a
  // comdat. This need to be a comdat with an .init_array section to prevent
  // newer versions of lld from discarding the note.
  Comdat *NoteComdat = M.getOrInsertComdat(kHwasanModuleCtorName);

  Type *Int8Arr0Ty = ArrayType::get(Int8Ty, 0);
  auto Start =
      new GlobalVariable(M, Int8Arr0Ty, true, GlobalVariable::ExternalLinkage,
                         nullptr, "__start_hwasan_globals");
  Start->setVisibility(GlobalValue::HiddenVisibility);
  Start->setDSOLocal(true);
  auto Stop =
      new GlobalVariable(M, Int8Arr0Ty, true, GlobalVariable::ExternalLinkage,
                         nullptr, "__stop_hwasan_globals");
  Stop->setVisibility(GlobalValue::HiddenVisibility);
  Stop->setDSOLocal(true);

  // Null-terminated so actually 8 bytes, which are required in order to align
  // the note properly.
  auto *Name = ConstantDataArray::get(*C, "LLVM\0\0\0");

  auto *NoteTy = StructType::get(Int32Ty, Int32Ty, Int32Ty, Name->getType(),
                                 Int32Ty, Int32Ty);
  auto *Note =
      new GlobalVariable(M, NoteTy, /*isConstantGlobal=*/true,
                         GlobalValue::PrivateLinkage, nullptr, kHwasanNoteName);
  Note->setSection(".note.hwasan.globals");
  Note->setComdat(NoteComdat);
  Note->setAlignment(Align(4));
  Note->setDSOLocal(true);

  // The pointers in the note need to be relative so that the note ends up being
  // placed in rodata, which is the standard location for notes.
  auto CreateRelPtr = [&](Constant *Ptr) {
    return ConstantExpr::getTrunc(
        ConstantExpr::getSub(ConstantExpr::getPtrToInt(Ptr, Int64Ty),
                             ConstantExpr::getPtrToInt(Note, Int64Ty)),
        Int32Ty);
  };
  Note->setInitializer(ConstantStruct::getAnon(
      {ConstantInt::get(Int32Ty, 8),                           // n_namesz
       ConstantInt::get(Int32Ty, 8),                           // n_descsz
       ConstantInt::get(Int32Ty, ELF::NT_LLVM_HWASAN_GLOBALS), // n_type
       Name, CreateRelPtr(Start), CreateRelPtr(Stop)}));
  appendToCompilerUsed(M, Note);

  // Create a zero-length global in hwasan_globals so that the linker will
  // always create start and stop symbols.
  auto Dummy = new GlobalVariable(
      M, Int8Arr0Ty, /*isConstantGlobal*/ true, GlobalVariable::PrivateLinkage,
      Constant::getNullValue(Int8Arr0Ty), "hwasan.dummy.global");
  Dummy->setSection("hwasan_globals");
  Dummy->setComdat(NoteComdat);
  Dummy->setMetadata(LLVMContext::MD_associated,
                     MDNode::get(*C, ValueAsMetadata::get(Note)));
  appendToCompilerUsed(M, Dummy);

  std::vector<GlobalVariable *> Globals;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.isDeclarationForLinker() || GV.getName().startswith("llvm.") ||
        GV.isThreadLocal())
      continue;

    // Common symbols can't have aliases point to them, so they can't be tagged.
    if (GV.hasCommonLinkage())
      continue;

    // Globals with custom sections may be used in __start_/__stop_ enumeration,
    // which would be broken both by adding tags and potentially by the extra
    // padding/alignment that we insert.
    if (GV.hasSection())
      continue;

    Globals.push_back(&GV);
  }

  MD5 Hasher;
  Hasher.update(M.getSourceFileName());
  MD5::MD5Result Hash;
  Hasher.final(Hash);
  uint8_t Tag = Hash[0];

  for (GlobalVariable *GV : Globals) {
    // Skip tag 0 in order to avoid collisions with untagged memory.
    if (Tag == 0)
      Tag = 1;
    instrumentGlobal(GV, Tag++);
  }
}

void HWAddressSanitizer::instrumentPersonalityFunctions() {
  // We need to untag stack frames as we unwind past them. That is the job of
  // the personality function wrapper, which either wraps an existing
  // personality function or acts as a personality function on its own. Each
  // function that has a personality function or that can be unwound past has
  // its personality function changed to a thunk that calls the personality
  // function wrapper in the runtime.
  MapVector<Constant *, std::vector<Function *>> PersonalityFns;
  for (Function &F : M) {
    if (F.isDeclaration() || !F.hasFnAttribute(Attribute::SanitizeHWAddress))
      continue;

    if (F.hasPersonalityFn()) {
      PersonalityFns[F.getPersonalityFn()->stripPointerCasts()].push_back(&F);
    } else if (!F.hasFnAttribute(Attribute::NoUnwind)) {
      PersonalityFns[nullptr].push_back(&F);
    }
  }

  if (PersonalityFns.empty())
    return;

  FunctionCallee HwasanPersonalityWrapper = M.getOrInsertFunction(
      "__hwasan_personality_wrapper", Int32Ty, Int32Ty, Int32Ty, Int64Ty,
      Int8PtrTy, Int8PtrTy, Int8PtrTy, Int8PtrTy, Int8PtrTy);
  FunctionCallee UnwindGetGR = M.getOrInsertFunction("_Unwind_GetGR", VoidTy);
  FunctionCallee UnwindGetCFA = M.getOrInsertFunction("_Unwind_GetCFA", VoidTy);

  for (auto &P : PersonalityFns) {
    std::string ThunkName = kHwasanPersonalityThunkName;
    if (P.first)
      ThunkName += ("." + P.first->getName()).str();
    FunctionType *ThunkFnTy = FunctionType::get(
        Int32Ty, {Int32Ty, Int32Ty, Int64Ty, Int8PtrTy, Int8PtrTy}, false);
    bool IsLocal = P.first && (!isa<GlobalValue>(P.first) ||
                               cast<GlobalValue>(P.first)->hasLocalLinkage());
    auto *ThunkFn = Function::Create(ThunkFnTy,
                                     IsLocal ? GlobalValue::InternalLinkage
                                             : GlobalValue::LinkOnceODRLinkage,
                                     ThunkName, &M);
    if (!IsLocal) {
      ThunkFn->setVisibility(GlobalValue::HiddenVisibility);
      ThunkFn->setComdat(M.getOrInsertComdat(ThunkName));
    }

    auto *BB = BasicBlock::Create(*C, "entry", ThunkFn);
    IRBuilder<> IRB(BB);
    CallInst *WrapperCall = IRB.CreateCall(
        HwasanPersonalityWrapper,
        {ThunkFn->getArg(0), ThunkFn->getArg(1), ThunkFn->getArg(2),
         ThunkFn->getArg(3), ThunkFn->getArg(4),
         P.first ? IRB.CreateBitCast(P.first, Int8PtrTy)
                 : Constant::getNullValue(Int8PtrTy),
         IRB.CreateBitCast(UnwindGetGR.getCallee(), Int8PtrTy),
         IRB.CreateBitCast(UnwindGetCFA.getCallee(), Int8PtrTy)});
    WrapperCall->setTailCall();
    IRB.CreateRet(WrapperCall);

    for (Function *F : P.second)
      F->setPersonalityFn(ThunkFn);
  }
}

void HWAddressSanitizer::ShadowMapping::init(Triple &TargetTriple) {
  Scale = kDefaultShadowScale;
  if (ClMappingOffset.getNumOccurrences() > 0) {
    InGlobal = false;
    InTls = false;
    Offset = ClMappingOffset;
  } else if (ClEnableKhwasan || ClInstrumentWithCalls) {
    InGlobal = false;
    InTls = false;
    Offset = 0;
  } else if (ClWithIfunc) {
    InGlobal = true;
    InTls = false;
    Offset = kDynamicShadowSentinel;
  } else if (ClWithTls) {
    InGlobal = false;
    InTls = true;
    Offset = kDynamicShadowSentinel;
  } else {
    InGlobal = false;
    InTls = false;
    Offset = kDynamicShadowSentinel;
  }
}
