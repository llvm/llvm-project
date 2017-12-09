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
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "hwasan"

static const char *const kHwasanModuleCtorName = "hwasan.module_ctor";
static const char *const kHwasanInitName = "__hwasan_init";

// Accesses sizes are powers of two: 1, 2, 4, 8, 16.
static const size_t kNumberOfAccessSizes = 5;

static cl::opt<std::string> ClMemoryAccessCallbackPrefix(
    "hwasan-memory-access-callback-prefix",
    cl::desc("Prefix for memory access callbacks"), cl::Hidden,
    cl::init("__hwasan_"));

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

namespace {

/// \brief An instrumentation pass implementing detection of addressability bugs
/// using tagged pointers.
class HWAddressSanitizer : public FunctionPass {
public:
  // Pass identification, replacement for typeid.
  static char ID;

  HWAddressSanitizer() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "HWAddressSanitizer"; }

  bool runOnFunction(Function &F) override;
  bool doInitialization(Module &M) override;

  void initializeCallbacks(Module &M);
  bool instrumentMemAccess(Instruction *I);
  Value *isInterestingMemoryAccess(Instruction *I, bool *IsWrite,
                                   uint64_t *TypeSize, unsigned *Alignment,
                                   Value **MaybeMask);

private:
  LLVMContext *C;
  Type *IntptrTy;

  Function *HwasanCtorFunction;

  Function *HwasanMemoryAccessCallback[2][kNumberOfAccessSizes];
  Function *HwasanMemoryAccessCallbackSized[2];
};

} // end anonymous namespace

char HWAddressSanitizer::ID = 0;

INITIALIZE_PASS_BEGIN(
    HWAddressSanitizer, "hwasan",
    "HWAddressSanitizer: detect memory bugs using tagged addressing.", false, false)
INITIALIZE_PASS_END(
    HWAddressSanitizer, "hwasan",
    "HWAddressSanitizer: detect memory bugs using tagged addressing.", false, false)

FunctionPass *llvm::createHWAddressSanitizerPass() {
  return new HWAddressSanitizer();
}

/// \brief Module-level initialization.
///
/// inserts a call to __hwasan_init to the module's constructor list.
bool HWAddressSanitizer::doInitialization(Module &M) {
  DEBUG(dbgs() << "Init " << M.getName() << "\n");
  auto &DL = M.getDataLayout();

  Triple TargetTriple(M.getTargetTriple());

  C = &(M.getContext());
  IRBuilder<> IRB(*C);
  IntptrTy = IRB.getIntPtrTy(DL);

  std::tie(HwasanCtorFunction, std::ignore) =
      createSanitizerCtorAndInitFunctions(M, kHwasanModuleCtorName,
                                          kHwasanInitName,
                                          /*InitArgTypes=*/{},
                                          /*InitArgs=*/{});
  appendToGlobalCtors(M, HwasanCtorFunction, 0);
  return true;
}

void HWAddressSanitizer::initializeCallbacks(Module &M) {
  IRBuilder<> IRB(*C);
  for (size_t AccessIsWrite = 0; AccessIsWrite <= 1; AccessIsWrite++) {
    const std::string TypeStr = AccessIsWrite ? "store" : "load";

    HwasanMemoryAccessCallbackSized[AccessIsWrite] =
        checkSanitizerInterfaceFunction(M.getOrInsertFunction(
            ClMemoryAccessCallbackPrefix + TypeStr,
            FunctionType::get(IRB.getVoidTy(), {IntptrTy, IntptrTy}, false)));

    for (size_t AccessSizeIndex = 0; AccessSizeIndex < kNumberOfAccessSizes;
         AccessSizeIndex++) {
      HwasanMemoryAccessCallback[AccessIsWrite][AccessSizeIndex] =
          checkSanitizerInterfaceFunction(M.getOrInsertFunction(
              ClMemoryAccessCallbackPrefix + TypeStr +
                  itostr(1ULL << AccessSizeIndex),
              FunctionType::get(IRB.getVoidTy(), {IntptrTy}, false)));
    }
  }
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

static size_t TypeSizeToSizeIndex(uint32_t TypeSize) {
  size_t Res = countTrailingZeros(TypeSize / 8);
  assert(Res < kNumberOfAccessSizes);
  return Res;
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
      (TypeSize / 8 <= (1UL << (kNumberOfAccessSizes - 1)))) {
    size_t AccessSizeIndex = TypeSizeToSizeIndex(TypeSize);
    IRB.CreateCall(HwasanMemoryAccessCallback[IsWrite][AccessSizeIndex],
                   AddrLong);
  } else {
    IRB.CreateCall(HwasanMemoryAccessCallbackSized[IsWrite],
                   {AddrLong, ConstantInt::get(IntptrTy, TypeSize / 8)});
  }

  return true;
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
  for (auto &BB : F) {
    for (auto &Inst : BB) {
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

  for (auto Inst : ToInstrument)
    Changed |= instrumentMemAccess(Inst);

  return Changed;
}
