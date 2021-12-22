//===- AArch64StackTagging.cpp - Stack tagging in IR --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/Metadata.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerCommon.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <iterator>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "aarch64-stack-tagging"

static cl::opt<bool> ClMergeInit(
    "stack-tagging-merge-init", cl::Hidden, cl::init(true), cl::ZeroOrMore,
    cl::desc("merge stack variable initializers with tagging when possible"));

static cl::opt<bool>
    ClUseStackSafety("stack-tagging-use-stack-safety", cl::Hidden,
                     cl::init(true), cl::ZeroOrMore,
                     cl::desc("Use Stack Safety analysis results"));

static cl::opt<unsigned> ClScanLimit("stack-tagging-merge-init-scan-limit",
                                     cl::init(40), cl::Hidden);

static cl::opt<unsigned>
    ClMergeInitSizeLimit("stack-tagging-merge-init-size-limit", cl::init(272),
                         cl::Hidden);

static const Align kTagGranuleSize = Align(16);

namespace {

class InitializerBuilder {
  uint64_t Size;
  const DataLayout *DL;
  Value *BasePtr;
  Function *SetTagFn;
  Function *SetTagZeroFn;
  Function *StgpFn;

  // List of initializers sorted by start offset.
  struct Range {
    uint64_t Start, End;
    Instruction *Inst;
  };
  SmallVector<Range, 4> Ranges;
  // 8-aligned offset => 8-byte initializer
  // Missing keys are zero initialized.
  std::map<uint64_t, Value *> Out;

public:
  InitializerBuilder(uint64_t Size, const DataLayout *DL, Value *BasePtr,
                     Function *SetTagFn, Function *SetTagZeroFn,
                     Function *StgpFn)
      : Size(Size), DL(DL), BasePtr(BasePtr), SetTagFn(SetTagFn),
        SetTagZeroFn(SetTagZeroFn), StgpFn(StgpFn) {}

  bool addRange(uint64_t Start, uint64_t End, Instruction *Inst) {
    auto I =
        llvm::lower_bound(Ranges, Start, [](const Range &LHS, uint64_t RHS) {
          return LHS.End <= RHS;
        });
    if (I != Ranges.end() && End > I->Start) {
      // Overlap - bail.
      return false;
    }
    Ranges.insert(I, {Start, End, Inst});
    return true;
  }

  bool addStore(uint64_t Offset, StoreInst *SI, const DataLayout *DL) {
    int64_t StoreSize = DL->getTypeStoreSize(SI->getOperand(0)->getType());
    if (!addRange(Offset, Offset + StoreSize, SI))
      return false;
    IRBuilder<> IRB(SI);
    applyStore(IRB, Offset, Offset + StoreSize, SI->getOperand(0));
    return true;
  }

  bool addMemSet(uint64_t Offset, MemSetInst *MSI) {
    uint64_t StoreSize = cast<ConstantInt>(MSI->getLength())->getZExtValue();
    if (!addRange(Offset, Offset + StoreSize, MSI))
      return false;
    IRBuilder<> IRB(MSI);
    applyMemSet(IRB, Offset, Offset + StoreSize,
                cast<ConstantInt>(MSI->getValue()));
    return true;
  }

  void applyMemSet(IRBuilder<> &IRB, int64_t Start, int64_t End,
                   ConstantInt *V) {
    // Out[] does not distinguish between zero and undef, and we already know
    // that this memset does not overlap with any other initializer. Nothing to
    // do for memset(0).
    if (V->isZero())
      return;
    for (int64_t Offset = Start - Start % 8; Offset < End; Offset += 8) {
      uint64_t Cst = 0x0101010101010101UL;
      int LowBits = Offset < Start ? (Start - Offset) * 8 : 0;
      if (LowBits)
        Cst = (Cst >> LowBits) << LowBits;
      int HighBits = End - Offset < 8 ? (8 - (End - Offset)) * 8 : 0;
      if (HighBits)
        Cst = (Cst << HighBits) >> HighBits;
      ConstantInt *C =
          ConstantInt::get(IRB.getInt64Ty(), Cst * V->getZExtValue());

      Value *&CurrentV = Out[Offset];
      if (!CurrentV) {
        CurrentV = C;
      } else {
        CurrentV = IRB.CreateOr(CurrentV, C);
      }
    }
  }

  // Take a 64-bit slice of the value starting at the given offset (in bytes).
  // Offset can be negative. Pad with zeroes on both sides when necessary.
  Value *sliceValue(IRBuilder<> &IRB, Value *V, int64_t Offset) {
    if (Offset > 0) {
      V = IRB.CreateLShr(V, Offset * 8);
      V = IRB.CreateZExtOrTrunc(V, IRB.getInt64Ty());
    } else if (Offset < 0) {
      V = IRB.CreateZExtOrTrunc(V, IRB.getInt64Ty());
      V = IRB.CreateShl(V, -Offset * 8);
    } else {
      V = IRB.CreateZExtOrTrunc(V, IRB.getInt64Ty());
    }
    return V;
  }

  void applyStore(IRBuilder<> &IRB, int64_t Start, int64_t End,
                  Value *StoredValue) {
    StoredValue = flatten(IRB, StoredValue);
    for (int64_t Offset = Start - Start % 8; Offset < End; Offset += 8) {
      Value *V = sliceValue(IRB, StoredValue, Offset - Start);
      Value *&CurrentV = Out[Offset];
      if (!CurrentV) {
        CurrentV = V;
      } else {
        CurrentV = IRB.CreateOr(CurrentV, V);
      }
    }
  }

  void generate(IRBuilder<> &IRB) {
    LLVM_DEBUG(dbgs() << "Combined initializer\n");
    // No initializers => the entire allocation is undef.
    if (Ranges.empty()) {
      emitUndef(IRB, 0, Size);
      return;
    }

    // Look through 8-byte initializer list 16 bytes at a time;
    // If one of the two 8-byte halfs is non-zero non-undef, emit STGP.
    // Otherwise, emit zeroes up to next available item.
    uint64_t LastOffset = 0;
    for (uint64_t Offset = 0; Offset < Size; Offset += 16) {
      auto I1 = Out.find(Offset);
      auto I2 = Out.find(Offset + 8);
      if (I1 == Out.end() && I2 == Out.end())
        continue;

      if (Offset > LastOffset)
        emitZeroes(IRB, LastOffset, Offset - LastOffset);

      Value *Store1 = I1 == Out.end() ? Constant::getNullValue(IRB.getInt64Ty())
                                      : I1->second;
      Value *Store2 = I2 == Out.end() ? Constant::getNullValue(IRB.getInt64Ty())
                                      : I2->second;
      emitPair(IRB, Offset, Store1, Store2);
      LastOffset = Offset + 16;
    }

    // memset(0) does not update Out[], therefore the tail can be either undef
    // or zero.
    if (LastOffset < Size)
      emitZeroes(IRB, LastOffset, Size - LastOffset);

    for (const auto &R : Ranges) {
      R.Inst->eraseFromParent();
    }
  }

  void emitZeroes(IRBuilder<> &IRB, uint64_t Offset, uint64_t Size) {
    LLVM_DEBUG(dbgs() << "  [" << Offset << ", " << Offset + Size
                      << ") zero\n");
    Value *Ptr = BasePtr;
    if (Offset)
      Ptr = IRB.CreateConstGEP1_32(IRB.getInt8Ty(), Ptr, Offset);
    IRB.CreateCall(SetTagZeroFn,
                   {Ptr, ConstantInt::get(IRB.getInt64Ty(), Size)});
  }

  void emitUndef(IRBuilder<> &IRB, uint64_t Offset, uint64_t Size) {
    LLVM_DEBUG(dbgs() << "  [" << Offset << ", " << Offset + Size
                      << ") undef\n");
    Value *Ptr = BasePtr;
    if (Offset)
      Ptr = IRB.CreateConstGEP1_32(IRB.getInt8Ty(), Ptr, Offset);
    IRB.CreateCall(SetTagFn, {Ptr, ConstantInt::get(IRB.getInt64Ty(), Size)});
  }

  void emitPair(IRBuilder<> &IRB, uint64_t Offset, Value *A, Value *B) {
    LLVM_DEBUG(dbgs() << "  [" << Offset << ", " << Offset + 16 << "):\n");
    LLVM_DEBUG(dbgs() << "    " << *A << "\n    " << *B << "\n");
    Value *Ptr = BasePtr;
    if (Offset)
      Ptr = IRB.CreateConstGEP1_32(IRB.getInt8Ty(), Ptr, Offset);
    IRB.CreateCall(StgpFn, {Ptr, A, B});
  }

  Value *flatten(IRBuilder<> &IRB, Value *V) {
    if (V->getType()->isIntegerTy())
      return V;
    // vector of pointers -> vector of ints
    if (VectorType *VecTy = dyn_cast<VectorType>(V->getType())) {
      LLVMContext &Ctx = IRB.getContext();
      Type *EltTy = VecTy->getElementType();
      if (EltTy->isPointerTy()) {
        uint32_t EltSize = DL->getTypeSizeInBits(EltTy);
        auto *NewTy = FixedVectorType::get(
            IntegerType::get(Ctx, EltSize),
            cast<FixedVectorType>(VecTy)->getNumElements());
        V = IRB.CreatePointerCast(V, NewTy);
      }
    }
    return IRB.CreateBitOrPointerCast(
        V, IRB.getIntNTy(DL->getTypeStoreSize(V->getType()) * 8));
  }
};

class AArch64StackTagging : public FunctionPass {
  struct AllocaInfo {
    AllocaInst *AI;
    TrackingVH<Instruction> OldAI; // Track through RAUW to replace debug uses.
    SmallVector<IntrinsicInst *, 2> LifetimeStart;
    SmallVector<IntrinsicInst *, 2> LifetimeEnd;
    SmallVector<DbgVariableIntrinsic *, 2> DbgVariableIntrinsics;
    int Tag; // -1 for non-tagged allocations
  };

  const bool MergeInit;
  const bool UseStackSafety;

public:
  static char ID; // Pass ID, replacement for typeid

  AArch64StackTagging(bool IsOptNone = false)
      : FunctionPass(ID),
        MergeInit(ClMergeInit.getNumOccurrences() ? ClMergeInit : !IsOptNone),
        UseStackSafety(ClUseStackSafety.getNumOccurrences() ? ClUseStackSafety
                                                            : !IsOptNone) {
    initializeAArch64StackTaggingPass(*PassRegistry::getPassRegistry());
  }

  bool isInterestingAlloca(const AllocaInst &AI);
  void alignAndPadAlloca(AllocaInfo &Info);

  void tagAlloca(AllocaInst *AI, Instruction *InsertBefore, Value *Ptr,
                 uint64_t Size);
  void untagAlloca(AllocaInst *AI, Instruction *InsertBefore, uint64_t Size);

  Instruction *collectInitializers(Instruction *StartInst, Value *StartPtr,
                                   uint64_t Size, InitializerBuilder &IB);

  Instruction *
  insertBaseTaggedPointer(const MapVector<AllocaInst *, AllocaInfo> &Allocas,
                          const DominatorTree *DT);
  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return "AArch64 Stack Tagging"; }

private:
  Function *F = nullptr;
  Function *SetTagFunc = nullptr;
  const DataLayout *DL = nullptr;
  AAResults *AA = nullptr;
  const StackSafetyGlobalInfo *SSI = nullptr;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    if (UseStackSafety)
      AU.addRequired<StackSafetyGlobalInfoWrapperPass>();
    if (MergeInit)
      AU.addRequired<AAResultsWrapperPass>();
  }
};

} // end anonymous namespace

char AArch64StackTagging::ID = 0;

INITIALIZE_PASS_BEGIN(AArch64StackTagging, DEBUG_TYPE, "AArch64 Stack Tagging",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(StackSafetyGlobalInfoWrapperPass)
INITIALIZE_PASS_END(AArch64StackTagging, DEBUG_TYPE, "AArch64 Stack Tagging",
                    false, false)

FunctionPass *llvm::createAArch64StackTaggingPass(bool IsOptNone) {
  return new AArch64StackTagging(IsOptNone);
}

Instruction *AArch64StackTagging::collectInitializers(Instruction *StartInst,
                                                      Value *StartPtr,
                                                      uint64_t Size,
                                                      InitializerBuilder &IB) {
  MemoryLocation AllocaLoc{StartPtr, Size};
  Instruction *LastInst = StartInst;
  BasicBlock::iterator BI(StartInst);

  unsigned Count = 0;
  for (; Count < ClScanLimit && !BI->isTerminator(); ++BI) {
    if (!isa<DbgInfoIntrinsic>(*BI))
      ++Count;

    if (isNoModRef(AA->getModRefInfo(&*BI, AllocaLoc)))
      continue;

    if (!isa<StoreInst>(BI) && !isa<MemSetInst>(BI)) {
      // If the instruction is readnone, ignore it, otherwise bail out.  We
      // don't even allow readonly here because we don't want something like:
      // A[1] = 2; strlen(A); A[2] = 2; -> memcpy(A, ...); strlen(A).
      if (BI->mayWriteToMemory() || BI->mayReadFromMemory())
        break;
      continue;
    }

    if (StoreInst *NextStore = dyn_cast<StoreInst>(BI)) {
      if (!NextStore->isSimple())
        break;

      // Check to see if this store is to a constant offset from the start ptr.
      Optional<int64_t> Offset =
          isPointerOffset(StartPtr, NextStore->getPointerOperand(), *DL);
      if (!Offset)
        break;

      if (!IB.addStore(*Offset, NextStore, DL))
        break;
      LastInst = NextStore;
    } else {
      MemSetInst *MSI = cast<MemSetInst>(BI);

      if (MSI->isVolatile() || !isa<ConstantInt>(MSI->getLength()))
        break;

      if (!isa<ConstantInt>(MSI->getValue()))
        break;

      // Check to see if this store is to a constant offset from the start ptr.
      Optional<int64_t> Offset = isPointerOffset(StartPtr, MSI->getDest(), *DL);
      if (!Offset)
        break;

      if (!IB.addMemSet(*Offset, MSI))
        break;
      LastInst = MSI;
    }
  }
  return LastInst;
}

bool AArch64StackTagging::isInterestingAlloca(const AllocaInst &AI) {
  // FIXME: support dynamic allocas
  bool IsInteresting =
      AI.getAllocatedType()->isSized() && AI.isStaticAlloca() &&
      // alloca() may be called with 0 size, ignore it.
      AI.getAllocationSizeInBits(*DL).getValue() > 0 &&
      // inalloca allocas are not treated as static, and we don't want
      // dynamic alloca instrumentation for them as well.
      !AI.isUsedWithInAlloca() &&
      // swifterror allocas are register promoted by ISel
      !AI.isSwiftError() &&
      // safe allocas are not interesting
      !(SSI && SSI->isSafe(AI));
  return IsInteresting;
}

void AArch64StackTagging::tagAlloca(AllocaInst *AI, Instruction *InsertBefore,
                                    Value *Ptr, uint64_t Size) {
  auto SetTagZeroFunc =
      Intrinsic::getDeclaration(F->getParent(), Intrinsic::aarch64_settag_zero);
  auto StgpFunc =
      Intrinsic::getDeclaration(F->getParent(), Intrinsic::aarch64_stgp);

  InitializerBuilder IB(Size, DL, Ptr, SetTagFunc, SetTagZeroFunc, StgpFunc);
  bool LittleEndian =
      Triple(AI->getModule()->getTargetTriple()).isLittleEndian();
  // Current implementation of initializer merging assumes little endianness.
  if (MergeInit && !F->hasOptNone() && LittleEndian &&
      Size < ClMergeInitSizeLimit) {
    LLVM_DEBUG(dbgs() << "collecting initializers for " << *AI
                      << ", size = " << Size << "\n");
    InsertBefore = collectInitializers(InsertBefore, Ptr, Size, IB);
  }

  IRBuilder<> IRB(InsertBefore);
  IB.generate(IRB);
}

void AArch64StackTagging::untagAlloca(AllocaInst *AI, Instruction *InsertBefore,
                                      uint64_t Size) {
  IRBuilder<> IRB(InsertBefore);
  IRB.CreateCall(SetTagFunc, {IRB.CreatePointerCast(AI, IRB.getInt8PtrTy()),
                              ConstantInt::get(IRB.getInt64Ty(), Size)});
}

Instruction *AArch64StackTagging::insertBaseTaggedPointer(
    const MapVector<AllocaInst *, AllocaInfo> &Allocas,
    const DominatorTree *DT) {
  BasicBlock *PrologueBB = nullptr;
  // Try sinking IRG as deep as possible to avoid hurting shrink wrap.
  for (auto &I : Allocas) {
    const AllocaInfo &Info = I.second;
    AllocaInst *AI = Info.AI;
    if (Info.Tag < 0)
      continue;
    if (!PrologueBB) {
      PrologueBB = AI->getParent();
      continue;
    }
    PrologueBB = DT->findNearestCommonDominator(PrologueBB, AI->getParent());
  }
  assert(PrologueBB);

  IRBuilder<> IRB(&PrologueBB->front());
  Function *IRG_SP =
      Intrinsic::getDeclaration(F->getParent(), Intrinsic::aarch64_irg_sp);
  Instruction *Base =
      IRB.CreateCall(IRG_SP, {Constant::getNullValue(IRB.getInt64Ty())});
  Base->setName("basetag");
  return Base;
}

void AArch64StackTagging::alignAndPadAlloca(AllocaInfo &Info) {
  const Align NewAlignment =
      max(MaybeAlign(Info.AI->getAlign()), kTagGranuleSize);
  Info.AI->setAlignment(NewAlignment);

  uint64_t Size = Info.AI->getAllocationSizeInBits(*DL).getValue() / 8;
  uint64_t AlignedSize = alignTo(Size, kTagGranuleSize);
  if (Size == AlignedSize)
    return;

  // Add padding to the alloca.
  Type *AllocatedType =
      Info.AI->isArrayAllocation()
          ? ArrayType::get(
                Info.AI->getAllocatedType(),
                cast<ConstantInt>(Info.AI->getArraySize())->getZExtValue())
          : Info.AI->getAllocatedType();
  Type *PaddingType =
      ArrayType::get(Type::getInt8Ty(F->getContext()), AlignedSize - Size);
  Type *TypeWithPadding = StructType::get(AllocatedType, PaddingType);
  auto *NewAI = new AllocaInst(
      TypeWithPadding, Info.AI->getType()->getAddressSpace(), nullptr, "", Info.AI);
  NewAI->takeName(Info.AI);
  NewAI->setAlignment(Info.AI->getAlign());
  NewAI->setUsedWithInAlloca(Info.AI->isUsedWithInAlloca());
  NewAI->setSwiftError(Info.AI->isSwiftError());
  NewAI->copyMetadata(*Info.AI);

  auto *NewPtr = new BitCastInst(NewAI, Info.AI->getType(), "", Info.AI);
  Info.AI->replaceAllUsesWith(NewPtr);
  Info.AI->eraseFromParent();
  Info.AI = NewAI;
}

// FIXME: check for MTE extension
bool AArch64StackTagging::runOnFunction(Function &Fn) {
  if (!Fn.hasFnAttribute(Attribute::SanitizeMemTag))
    return false;

  if (UseStackSafety)
    SSI = &getAnalysis<StackSafetyGlobalInfoWrapperPass>().getResult();
  F = &Fn;
  DL = &Fn.getParent()->getDataLayout();
  if (MergeInit)
    AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();

  MapVector<AllocaInst *, AllocaInfo> Allocas; // need stable iteration order
  SmallVector<Instruction *, 8> RetVec;
  SmallVector<Instruction *, 4> UnrecognizedLifetimes;

  for (auto &BB : *F) {
    for (Instruction &I : BB) {
      if (auto *AI = dyn_cast<AllocaInst>(&I)) {
        Allocas[AI].AI = AI;
        Allocas[AI].OldAI = AI;
        continue;
      }

      if (auto *DVI = dyn_cast<DbgVariableIntrinsic>(&I)) {
        for (Value *V : DVI->location_ops())
          if (auto *AI = dyn_cast_or_null<AllocaInst>(V))
            if (Allocas[AI].DbgVariableIntrinsics.empty() ||
                Allocas[AI].DbgVariableIntrinsics.back() != DVI)
              Allocas[AI].DbgVariableIntrinsics.push_back(DVI);
        continue;
      }

      auto *II = dyn_cast<IntrinsicInst>(&I);
      if (II && (II->getIntrinsicID() == Intrinsic::lifetime_start ||
                 II->getIntrinsicID() == Intrinsic::lifetime_end)) {
        AllocaInst *AI = findAllocaForValue(II->getArgOperand(1));
        if (!AI) {
          UnrecognizedLifetimes.push_back(&I);
          continue;
        }
        if (II->getIntrinsicID() == Intrinsic::lifetime_start)
          Allocas[AI].LifetimeStart.push_back(II);
        else
          Allocas[AI].LifetimeEnd.push_back(II);
      }

      if (isa<ReturnInst, ResumeInst, CleanupReturnInst>(&I))
        RetVec.push_back(&I);
    }
  }

  if (Allocas.empty())
    return false;

  int NextTag = 0;
  int NumInterestingAllocas = 0;
  for (auto &I : Allocas) {
    AllocaInfo &Info = I.second;
    assert(Info.AI);

    if (!isInterestingAlloca(*Info.AI)) {
      Info.Tag = -1;
      continue;
    }

    alignAndPadAlloca(Info);
    NumInterestingAllocas++;
    Info.Tag = NextTag;
    NextTag = (NextTag + 1) % 16;
  }

  if (NumInterestingAllocas == 0)
    return true;

  std::unique_ptr<DominatorTree> DeleteDT;
  DominatorTree *DT = nullptr;
  if (auto *P = getAnalysisIfAvailable<DominatorTreeWrapperPass>())
    DT = &P->getDomTree();

  if (DT == nullptr && (NumInterestingAllocas > 1 ||
                        !F->hasFnAttribute(Attribute::OptimizeNone))) {
    DeleteDT = std::make_unique<DominatorTree>(*F);
    DT = DeleteDT.get();
  }

  std::unique_ptr<PostDominatorTree> DeletePDT;
  PostDominatorTree *PDT = nullptr;
  if (auto *P = getAnalysisIfAvailable<PostDominatorTreeWrapperPass>())
    PDT = &P->getPostDomTree();

  if (PDT == nullptr && !F->hasFnAttribute(Attribute::OptimizeNone)) {
    DeletePDT = std::make_unique<PostDominatorTree>(*F);
    PDT = DeletePDT.get();
  }

  SetTagFunc =
      Intrinsic::getDeclaration(F->getParent(), Intrinsic::aarch64_settag);

  Instruction *Base = insertBaseTaggedPointer(Allocas, DT);

  for (auto &I : Allocas) {
    const AllocaInfo &Info = I.second;
    AllocaInst *AI = Info.AI;
    if (Info.Tag < 0)
      continue;

    // Replace alloca with tagp(alloca).
    IRBuilder<> IRB(Info.AI->getNextNode());
    Function *TagP = Intrinsic::getDeclaration(
        F->getParent(), Intrinsic::aarch64_tagp, {Info.AI->getType()});
    Instruction *TagPCall =
        IRB.CreateCall(TagP, {Constant::getNullValue(Info.AI->getType()), Base,
                              ConstantInt::get(IRB.getInt64Ty(), Info.Tag)});
    if (Info.AI->hasName())
      TagPCall->setName(Info.AI->getName() + ".tag");
    Info.AI->replaceAllUsesWith(TagPCall);
    TagPCall->setOperand(0, Info.AI);

    if (UnrecognizedLifetimes.empty() && Info.LifetimeStart.size() == 1 &&
        Info.LifetimeEnd.size() == 1) {
      IntrinsicInst *Start = Info.LifetimeStart[0];
      IntrinsicInst *End = Info.LifetimeEnd[0];
      uint64_t Size =
          cast<ConstantInt>(Start->getArgOperand(0))->getZExtValue();
      Size = alignTo(Size, kTagGranuleSize);
      tagAlloca(AI, Start->getNextNode(), Start->getArgOperand(1), Size);

      auto TagEnd = [&](Instruction *Node) { untagAlloca(AI, Node, Size); };
      if (!DT || !PDT ||
          !forAllReachableExits(*DT, *PDT, Start, Info.LifetimeEnd, RetVec,
                                TagEnd))
        End->eraseFromParent();
    } else {
      uint64_t Size = Info.AI->getAllocationSizeInBits(*DL).getValue() / 8;
      Value *Ptr = IRB.CreatePointerCast(TagPCall, IRB.getInt8PtrTy());
      tagAlloca(AI, &*IRB.GetInsertPoint(), Ptr, Size);
      for (auto &RI : RetVec) {
        untagAlloca(AI, RI, Size);
      }
      // We may have inserted tag/untag outside of any lifetime interval.
      // Remove all lifetime intrinsics for this alloca.
      for (auto &II : Info.LifetimeStart)
        II->eraseFromParent();
      for (auto &II : Info.LifetimeEnd)
        II->eraseFromParent();
    }

    // Fixup debug intrinsics to point to the new alloca.
    for (auto DVI : Info.DbgVariableIntrinsics)
      DVI->replaceVariableLocationOp(Info.OldAI, Info.AI);
  }

  // If we have instrumented at least one alloca, all unrecognized lifetime
  // instrinsics have to go.
  for (auto &I : UnrecognizedLifetimes)
    I->eraseFromParent();

  return true;
}
