//===- HWAddressSanitizer.cpp - detector of uninitialized reads -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file is a part of HWAddressSanitizer, an address sanity checker
/// based on tagged addressing.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
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
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

using namespace llvm;

#define DEBUG_TYPE "hwasan"

static const char *const kHwasanModuleCtorName = "hwasan.module_ctor";
static const char *const kHwasanInitName = "__hwasan_init";

// Accesses sizes are powers of two: 1, 2, 4, 8, 16.
static const size_t kNumberOfAccessSizes = 5;

static const size_t kShadowScale = 4;
static const unsigned kAllocaAlignment = 1U << kShadowScale;
static const unsigned kPointerTagShift = 56;

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

static cl::opt<bool> ClRecover(
    "hwasan-recover",
    cl::desc("Enable recovery mode (continue-after-error)."),
    cl::Hidden, cl::init(false));

static cl::opt<bool> ClInstrumentStack("hwasan-instrument-stack",
                                       cl::desc("instrument stack (allocas)"),
                                       cl::Hidden, cl::init(true));

static cl::opt<bool> ClGenerateTagsWithCalls(
    "hwasan-generate-tags-with-calls",
    cl::desc("generate new tags with runtime library calls"), cl::Hidden,
    cl::init(false));

static cl::opt<unsigned long long> ClMappingOffset(
    "hwasan-mapping-offset",
    cl::desc("offset of hwasan shadow mapping [EXPERIMENTAL]"), cl::Hidden,
    cl::init(0));

static cl::opt<int> ClMatchAllTag(
    "hwasan-match-all-tag",
    cl::desc("don't report bad accesses via pointers with this tag"), cl::Hidden,
    cl::init(-1));

static cl::opt<bool> ClEnableKhwasan(
    "hwasan-kernel", cl::desc("Enable KernelHWAddressSanitizer instrumentation"),
    cl::Hidden, cl::init(false));

namespace {

/// \brief An instrumentation pass implementing detection of addressability bugs
/// using tagged pointers.
class HWAddressSanitizer : public FunctionPass {
public:
  // Pass identification, replacement for typeid.
  static char ID;

  HWAddressSanitizer(bool Recover = false)
      : FunctionPass(ID), Recover(Recover || ClRecover) {}

  StringRef getPassName() const override { return "HWAddressSanitizer"; }

  bool runOnFunction(Function &F) override;
  bool doInitialization(Module &M) override;

  void initializeCallbacks(Module &M);
  void untagPointerOperand(Instruction *I, Value *Addr);
  void instrumentMemAccessInline(Value *PtrLong, bool IsWrite,
                                 unsigned AccessSizeIndex,
                                 Instruction *InsertBefore);
  bool instrumentMemAccess(Instruction *I);
  Value *isInterestingMemoryAccess(Instruction *I, bool *IsWrite,
                                   uint64_t *TypeSize, unsigned *Alignment,
                                   Value **MaybeMask);

  bool isInterestingAlloca(const AllocaInst &AI);
  bool tagAlloca(IRBuilder<> &IRB, AllocaInst *AI, Value *Tag);
  Value *tagPointer(IRBuilder<> &IRB, Type *Ty, Value *PtrLong, Value *Tag);
  Value *untagPointer(IRBuilder<> &IRB, Value *PtrLong);
  bool instrumentStack(SmallVectorImpl<AllocaInst *> &Allocas,
                       SmallVectorImpl<Instruction *> &RetVec);
  Value *getNextTagWithCall(IRBuilder<> &IRB);
  Value *getStackBaseTag(IRBuilder<> &IRB);
  Value *getAllocaTag(IRBuilder<> &IRB, Value *StackTag, AllocaInst *AI,
                     unsigned AllocaNo);
  Value *getUARTag(IRBuilder<> &IRB, Value *StackTag);

private:
  LLVMContext *C;
  Triple TargetTriple;

  Type *IntptrTy;
  Type *Int8Ty;

  bool Recover;

  Function *HwasanCtorFunction;

  Function *HwasanMemoryAccessCallback[2][kNumberOfAccessSizes];
  Function *HwasanMemoryAccessCallbackSized[2];

  Function *HwasanTagMemoryFunc;
  Function *HwasanGenerateTagFunc;
};

} // end anonymous namespace

char HWAddressSanitizer::ID = 0;

INITIALIZE_PASS_BEGIN(
    HWAddressSanitizer, "hwasan",
    "HWAddressSanitizer: detect memory bugs using tagged addressing.", false, false)
INITIALIZE_PASS_END(
    HWAddressSanitizer, "hwasan",
    "HWAddressSanitizer: detect memory bugs using tagged addressing.", false, false)

FunctionPass *llvm::createHWAddressSanitizerPass(bool Recover) {
  return new HWAddressSanitizer(Recover);
}

/// \brief Module-level initialization.
///
/// inserts a call to __hwasan_init to the module's constructor list.
bool HWAddressSanitizer::doInitialization(Module &M) {
  DEBUG(dbgs() << "Init " << M.getName() << "\n");
  auto &DL = M.getDataLayout();

  TargetTriple = Triple(M.getTargetTriple());

  C = &(M.getContext());
  IRBuilder<> IRB(*C);
  IntptrTy = IRB.getIntPtrTy(DL);
  Int8Ty = IRB.getInt8Ty();

  HwasanCtorFunction = nullptr;
  if (!ClEnableKhwasan) {
    std::tie(HwasanCtorFunction, std::ignore) =
        createSanitizerCtorAndInitFunctions(M, kHwasanModuleCtorName,
                                            kHwasanInitName,
                                            /*InitArgTypes=*/{},
                                            /*InitArgs=*/{});
    appendToGlobalCtors(M, HwasanCtorFunction, 0);
  }
  return true;
}

void HWAddressSanitizer::initializeCallbacks(Module &M) {
  IRBuilder<> IRB(*C);
  for (size_t AccessIsWrite = 0; AccessIsWrite <= 1; AccessIsWrite++) {
    const std::string TypeStr = AccessIsWrite ? "store" : "load";
    const std::string EndingStr = Recover ? "_noabort" : "";

    HwasanMemoryAccessCallbackSized[AccessIsWrite] =
        checkSanitizerInterfaceFunction(M.getOrInsertFunction(
            ClMemoryAccessCallbackPrefix + TypeStr + "N" + EndingStr,
            FunctionType::get(IRB.getVoidTy(), {IntptrTy, IntptrTy}, false)));

    for (size_t AccessSizeIndex = 0; AccessSizeIndex < kNumberOfAccessSizes;
         AccessSizeIndex++) {
      HwasanMemoryAccessCallback[AccessIsWrite][AccessSizeIndex] =
          checkSanitizerInterfaceFunction(M.getOrInsertFunction(
              ClMemoryAccessCallbackPrefix + TypeStr +
                  itostr(1ULL << AccessSizeIndex) + EndingStr,
              FunctionType::get(IRB.getVoidTy(), {IntptrTy}, false)));
    }
  }

  HwasanTagMemoryFunc = checkSanitizerInterfaceFunction(M.getOrInsertFunction(
      "__hwasan_tag_memory", IRB.getVoidTy(), IntptrTy, Int8Ty, IntptrTy));
  HwasanGenerateTagFunc = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction("__hwasan_generate_tag", Int8Ty));
}

Value *HWAddressSanitizer::isInterestingMemoryAccess(Instruction *I,
                                                     bool *IsWrite,
                                                     uint64_t *TypeSize,
                                                     unsigned *Alignment,
                                                     Value **MaybeMask) {
  // Skip memory accesses inserted by another instrumentation.
  if (I->getMetadata("nosanitize")) return nullptr;

  Value *PtrOperand = nullptr;
  const DataLayout &DL = I->getModule()->getDataLayout();
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    if (!ClInstrumentReads) return nullptr;
    *IsWrite = false;
    *TypeSize = DL.getTypeStoreSizeInBits(LI->getType());
    *Alignment = LI->getAlignment();
    PtrOperand = LI->getPointerOperand();
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    if (!ClInstrumentWrites) return nullptr;
    *IsWrite = true;
    *TypeSize = DL.getTypeStoreSizeInBits(SI->getValueOperand()->getType());
    *Alignment = SI->getAlignment();
    PtrOperand = SI->getPointerOperand();
  } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
    if (!ClInstrumentAtomics) return nullptr;
    *IsWrite = true;
    *TypeSize = DL.getTypeStoreSizeInBits(RMW->getValOperand()->getType());
    *Alignment = 0;
    PtrOperand = RMW->getPointerOperand();
  } else if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(I)) {
    if (!ClInstrumentAtomics) return nullptr;
    *IsWrite = true;
    *TypeSize = DL.getTypeStoreSizeInBits(XCHG->getCompareOperand()->getType());
    *Alignment = 0;
    PtrOperand = XCHG->getPointerOperand();
  }

  if (PtrOperand) {
    // Do not instrument acesses from different address spaces; we cannot deal
    // with them.
    Type *PtrTy = cast<PointerType>(PtrOperand->getType()->getScalarType());
    if (PtrTy->getPointerAddressSpace() != 0)
      return nullptr;

    // Ignore swifterror addresses.
    // swifterror memory addresses are mem2reg promoted by instruction
    // selection. As such they cannot have regular uses like an instrumentation
    // function and it makes no sense to track them as memory.
    if (PtrOperand->isSwiftError())
      return nullptr;
  }

  return PtrOperand;
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

void HWAddressSanitizer::instrumentMemAccessInline(Value *PtrLong, bool IsWrite,
                                                   unsigned AccessSizeIndex,
                                                   Instruction *InsertBefore) {
  IRBuilder<> IRB(InsertBefore);
  Value *PtrTag = IRB.CreateTrunc(IRB.CreateLShr(PtrLong, kPointerTagShift),
                                  IRB.getInt8Ty());
  Value *AddrLong = untagPointer(IRB, PtrLong);
  Value *ShadowLong = IRB.CreateLShr(AddrLong, kShadowScale);
  if (ClMappingOffset)
    ShadowLong = IRB.CreateAdd(
        ShadowLong, ConstantInt::get(PtrLong->getType(), ClMappingOffset,
                                     /*isSigned=*/false));
  Value *MemTag =
      IRB.CreateLoad(IRB.CreateIntToPtr(ShadowLong, IRB.getInt8PtrTy()));
  Value *TagMismatch = IRB.CreateICmpNE(PtrTag, MemTag);

  if (ClMatchAllTag != -1) {
    Value *TagNotIgnored = IRB.CreateICmpNE(PtrTag,
        ConstantInt::get(PtrTag->getType(), ClMatchAllTag));
    TagMismatch = IRB.CreateAnd(TagMismatch, TagNotIgnored);
  }

  TerminatorInst *CheckTerm =
      SplitBlockAndInsertIfThen(TagMismatch, InsertBefore, !Recover,
                                MDBuilder(*C).createBranchWeights(1, 100000));

  IRB.SetInsertPoint(CheckTerm);
  const int64_t AccessInfo = Recover * 0x20 + IsWrite * 0x10 + AccessSizeIndex;
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
}

bool HWAddressSanitizer::instrumentMemAccess(Instruction *I) {
  DEBUG(dbgs() << "Instrumenting: " << *I << "\n");
  bool IsWrite = false;
  unsigned Alignment = 0;
  uint64_t TypeSize = 0;
  Value *MaybeMask = nullptr;
  Value *Addr =
      isInterestingMemoryAccess(I, &IsWrite, &TypeSize, &Alignment, &MaybeMask);

  if (!Addr)
    return false;

  if (MaybeMask)
    return false; //FIXME

  IRBuilder<> IRB(I);
  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);
  if (isPowerOf2_64(TypeSize) &&
      (TypeSize / 8 <= (1UL << (kNumberOfAccessSizes - 1))) &&
      (Alignment >= (1UL << kShadowScale) || Alignment == 0 ||
       Alignment >= TypeSize / 8)) {
    size_t AccessSizeIndex = TypeSizeToSizeIndex(TypeSize);
    if (ClInstrumentWithCalls) {
      IRB.CreateCall(HwasanMemoryAccessCallback[IsWrite][AccessSizeIndex],
                     AddrLong);
    } else {
      instrumentMemAccessInline(AddrLong, IsWrite, AccessSizeIndex, I);
    }
  } else {
    IRB.CreateCall(HwasanMemoryAccessCallbackSized[IsWrite],
                   {AddrLong, ConstantInt::get(IntptrTy, TypeSize / 8)});
  }
  untagPointerOperand(I, Addr);

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
                                   Value *Tag) {
  size_t Size = (getAllocaSizeInBytes(*AI) + kAllocaAlignment - 1) &
                ~(kAllocaAlignment - 1);

  Value *JustTag = IRB.CreateTrunc(Tag, IRB.getInt8Ty());
  if (ClInstrumentWithCalls) {
    IRB.CreateCall(HwasanTagMemoryFunc,
                   {IRB.CreatePointerCast(AI, IntptrTy), JustTag,
                    ConstantInt::get(IntptrTy, Size)});
  } else {
    size_t ShadowSize = Size >> kShadowScale;
    Value *ShadowPtr = IRB.CreateIntToPtr(
        IRB.CreateLShr(IRB.CreatePointerCast(AI, IntptrTy), kShadowScale),
        IRB.getInt8PtrTy());
    // If this memset is not inlined, it will be intercepted in the hwasan
    // runtime library. That's OK, because the interceptor skips the checks if
    // the address is in the shadow region.
    // FIXME: the interceptor is not as fast as real memset. Consider lowering
    // llvm.memset right here into either a sequence of stores, or a call to
    // hwasan_tag_memory.
    IRB.CreateMemSet(ShadowPtr, JustTag, ShadowSize, /*Align=*/1);
  }
  return true;
}

static unsigned RetagMask(unsigned AllocaNo) {
  // A list of 8-bit numbers that have at most one run of non-zero bits.
  // x = x ^ (mask << 56) can be encoded as a single armv8 instruction for these
  // masks.
  // The list does not include the value 255, which is used for UAR.
  static unsigned FastMasks[] = {
      0,   1,   2,   3,   4,   6,   7,   8,   12,  14,  15, 16,  24,
      28,  30,  31,  32,  48,  56,  60,  62,  63,  64,  96, 112, 120,
      124, 126, 127, 128, 192, 224, 240, 248, 252, 254};
  return FastMasks[AllocaNo % (sizeof(FastMasks) / sizeof(FastMasks[0]))];
}

Value *HWAddressSanitizer::getNextTagWithCall(IRBuilder<> &IRB) {
  return IRB.CreateZExt(IRB.CreateCall(HwasanGenerateTagFunc), IntptrTy);
}

Value *HWAddressSanitizer::getStackBaseTag(IRBuilder<> &IRB) {
  if (ClGenerateTagsWithCalls)
    return nullptr;
  // FIXME: use addressofreturnaddress (but implement it in aarch64 backend
  // first).
  Module *M = IRB.GetInsertBlock()->getParent()->getParent();
  auto GetStackPointerFn =
      Intrinsic::getDeclaration(M, Intrinsic::frameaddress);
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
  if (ClGenerateTagsWithCalls)
    return getNextTagWithCall(IRB);
  return IRB.CreateXor(StackTag, ConstantInt::get(IntptrTy, 0xFFU));
}

// Add a tag to an address.
Value *HWAddressSanitizer::tagPointer(IRBuilder<> &IRB, Type *Ty, Value *PtrLong,
                                      Value *Tag) {
  Value *TaggedPtrLong;
  if (ClEnableKhwasan) {
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
  if (ClEnableKhwasan) {
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

bool HWAddressSanitizer::instrumentStack(
    SmallVectorImpl<AllocaInst *> &Allocas,
    SmallVectorImpl<Instruction *> &RetVec) {
  Function *F = Allocas[0]->getParent()->getParent();
  Instruction *InsertPt = &*F->getEntryBlock().begin();
  IRBuilder<> IRB(InsertPt);

  Value *StackTag = getStackBaseTag(IRB);

  // Ideally, we want to calculate tagged stack base pointer, and rewrite all
  // alloca addresses using that. Unfortunately, offsets are not known yet
  // (unless we use ASan-style mega-alloca). Instead we keep the base tag in a
  // temp, shift-OR it into each alloca address and xor with the retag mask.
  // This generates one extra instruction per alloca use.
  for (unsigned N = 0; N < Allocas.size(); ++N) {
    auto *AI = Allocas[N];
    IRB.SetInsertPoint(AI->getNextNode());

    // Replace uses of the alloca with tagged address.
    Value *Tag = getAllocaTag(IRB, StackTag, AI, N);
    Value *AILong = IRB.CreatePointerCast(AI, IntptrTy);
    Value *Replacement = tagPointer(IRB, AI->getType(), AILong, Tag);
    std::string Name =
        AI->hasName() ? AI->getName().str() : "alloca." + itostr(N);
    Replacement->setName(Name + ".hwasan");

    for (auto UI = AI->use_begin(), UE = AI->use_end(); UI != UE;) {
      Use &U = *UI++;
      if (U.getUser() != AILong)
        U.set(Replacement);
    }

    tagAlloca(IRB, AI, Tag);

    for (auto RI : RetVec) {
      IRB.SetInsertPoint(RI);

      // Re-tag alloca memory with the special UAR tag.
      Value *Tag = getUARTag(IRB, StackTag);
      tagAlloca(IRB, AI, Tag);
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

bool HWAddressSanitizer::runOnFunction(Function &F) {
  if (&F == HwasanCtorFunction)
    return false;

  if (!F.hasFnAttribute(Attribute::SanitizeHWAddress))
    return false;

  DEBUG(dbgs() << "Function: " << F.getName() << "\n");

  initializeCallbacks(*F.getParent());

  bool Changed = false;
  SmallVector<Instruction*, 16> ToInstrument;
  SmallVector<AllocaInst*, 8> AllocasToInstrument;
  SmallVector<Instruction*, 8> RetVec;
  for (auto &BB : F) {
    for (auto &Inst : BB) {
      if (ClInstrumentStack)
        if (AllocaInst *AI = dyn_cast<AllocaInst>(&Inst)) {
          // Realign all allocas. We don't want small uninteresting allocas to
          // hide in instrumented alloca's padding.
          if (AI->getAlignment() < kAllocaAlignment)
            AI->setAlignment(kAllocaAlignment);
          // Instrument some of them.
          if (isInterestingAlloca(*AI))
            AllocasToInstrument.push_back(AI);
          continue;
        }

      if (isa<ReturnInst>(Inst) || isa<ResumeInst>(Inst) || isa<CleanupReturnInst>(Inst))
        RetVec.push_back(&Inst);

      Value *MaybeMask = nullptr;
      bool IsWrite;
      unsigned Alignment;
      uint64_t TypeSize;
      Value *Addr = isInterestingMemoryAccess(&Inst, &IsWrite, &TypeSize,
                                              &Alignment, &MaybeMask);
      if (Addr || isa<MemIntrinsic>(Inst))
        ToInstrument.push_back(&Inst);
    }
  }

  if (!AllocasToInstrument.empty())
    Changed |= instrumentStack(AllocasToInstrument, RetVec);

  for (auto Inst : ToInstrument)
    Changed |= instrumentMemAccess(Inst);

  return Changed;
}
