//===- BoundsChecking.cpp - Instrumentation for run-time bounds checking --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/BoundsChecking.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "bounds-checking"

static cl::opt<bool> SingleTrapBB("bounds-checking-single-trap",
                                  cl::desc("Use one trap block per function"));

STATISTIC(ChecksAdded, "Bounds checks added");
STATISTIC(ChecksSkipped, "Bounds checks skipped");
STATISTIC(ChecksUnable, "Bounds checks unable to add");

class BuilderTy : public IRBuilder<TargetFolder> {
public:
  BuilderTy(BasicBlock *TheBB, BasicBlock::iterator IP, TargetFolder Folder)
      : IRBuilder<TargetFolder>(TheBB, IP, Folder) {
    SetNoSanitizeMetadata();
  }
};

/// Gets the conditions under which memory accessing instructions will overflow.
///
/// \p Ptr is the pointer that will be read/written, and \p InstVal is either
/// the result from the load or the value being stored. It is used to determine
/// the size of memory block that is touched.
///
/// Returns the condition under which the access will overflow.
static Value *getBoundsCheckCond(Value *Ptr, Value *InstVal,
                                 const DataLayout &DL, TargetLibraryInfo &TLI,
                                 ObjectSizeOffsetEvaluator &ObjSizeEval,
                                 BuilderTy &IRB, ScalarEvolution &SE) {
  TypeSize NeededSize = DL.getTypeStoreSize(InstVal->getType());
  LLVM_DEBUG(dbgs() << "Instrument " << *Ptr << " for " << Twine(NeededSize)
                    << " bytes\n");

  SizeOffsetValue SizeOffset = ObjSizeEval.compute(Ptr);

  if (!SizeOffset.bothKnown()) {
    ++ChecksUnable;
    return nullptr;
  }

  Value *Size = SizeOffset.Size;
  Value *Offset = SizeOffset.Offset;
  ConstantInt *SizeCI = dyn_cast<ConstantInt>(Size);

  Type *IndexTy = DL.getIndexType(Ptr->getType());
  Value *NeededSizeVal = IRB.CreateTypeSize(IndexTy, NeededSize);

  auto SizeRange = SE.getUnsignedRange(SE.getSCEV(Size));
  auto OffsetRange = SE.getUnsignedRange(SE.getSCEV(Offset));
  auto NeededSizeRange = SE.getUnsignedRange(SE.getSCEV(NeededSizeVal));

  // three checks are required to ensure safety:
  // . Offset >= 0  (since the offset is given from the base ptr)
  // . Size >= Offset  (unsigned)
  // . Size - Offset >= NeededSize  (unsigned)
  //
  // optimization: if Size >= 0 (signed), skip 1st check
  // FIXME: add NSW/NUW here?  -- we dont care if the subtraction overflows
  Value *ObjSize = IRB.CreateSub(Size, Offset);
  Value *Cmp2 = SizeRange.getUnsignedMin().uge(OffsetRange.getUnsignedMax())
                    ? ConstantInt::getFalse(Ptr->getContext())
                    : IRB.CreateICmpULT(Size, Offset);
  Value *Cmp3 = SizeRange.sub(OffsetRange)
                        .getUnsignedMin()
                        .uge(NeededSizeRange.getUnsignedMax())
                    ? ConstantInt::getFalse(Ptr->getContext())
                    : IRB.CreateICmpULT(ObjSize, NeededSizeVal);
  Value *Or = IRB.CreateOr(Cmp2, Cmp3);
  if ((!SizeCI || SizeCI->getValue().slt(0)) &&
      !SizeRange.getSignedMin().isNonNegative()) {
    Value *Cmp1 = IRB.CreateICmpSLT(Offset, ConstantInt::get(IndexTy, 0));
    Or = IRB.CreateOr(Cmp1, Or);
  }

  return Or;
}

static CallInst *InsertTrap(BuilderTy &IRB, bool DebugTrapBB) {
  if (!DebugTrapBB)
    return IRB.CreateIntrinsic(Intrinsic::trap, {}, {});
  // FIXME: Ideally we would use the SanitizerHandler::OutOfBounds constant.
  return IRB.CreateIntrinsic(
      Intrinsic::ubsantrap, {},
      ConstantInt::get(IRB.getInt8Ty(),
                       IRB.GetInsertBlock()->getParent()->size()));
}

static CallInst *InsertCall(BuilderTy &IRB, bool MayReturn, StringRef Name) {
  Function *Fn = IRB.GetInsertBlock()->getParent();
  LLVMContext &Ctx = Fn->getContext();
  llvm::AttrBuilder B(Ctx);
  B.addAttribute(llvm::Attribute::NoUnwind);
  if (!MayReturn)
    B.addAttribute(llvm::Attribute::NoReturn);
  FunctionCallee Callee = Fn->getParent()->getOrInsertFunction(
      Name,
      llvm::AttributeList::get(Ctx, llvm::AttributeList::FunctionIndex, B),
      Type::getVoidTy(Ctx));
  return IRB.CreateCall(Callee);
}

/// Adds run-time bounds checks to memory accessing instructions.
///
/// \p Or is the condition that should guard the trap.
///
/// \p GetTrapBB is a callable that returns the trap BB to use on failure.
template <typename GetTrapBBT>
static void insertBoundsCheck(Value *Or, BuilderTy &IRB, GetTrapBBT GetTrapBB) {
  // check if the comparison is always false
  ConstantInt *C = dyn_cast_or_null<ConstantInt>(Or);
  if (C) {
    ++ChecksSkipped;
    // If non-zero, nothing to do.
    if (!C->getZExtValue())
      return;
  }
  ++ChecksAdded;

  BasicBlock::iterator SplitI = IRB.GetInsertPoint();
  BasicBlock *OldBB = SplitI->getParent();
  BasicBlock *Cont = OldBB->splitBasicBlock(SplitI);
  OldBB->getTerminator()->eraseFromParent();

  BasicBlock *TrapBB = GetTrapBB(IRB, Cont);

  if (C) {
    // If we have a constant zero, unconditionally branch.
    // FIXME: We should really handle this differently to bypass the splitting
    // the block.
    BranchInst::Create(TrapBB, OldBB);
    return;
  }

  // Create the conditional branch.
  BranchInst::Create(TrapBB, Cont, Or, OldBB);
}

static std::string
getRuntimeCallName(const BoundsCheckingPass::Options::Runtime &Opts) {
  std::string Name = "__ubsan_handle_local_out_of_bounds";
  if (Opts.MinRuntime)
    Name += "_minimal";
  if (!Opts.MayReturn)
    Name += "_abort";
  return Name;
}

static bool addBoundsChecking(Function &F, TargetLibraryInfo &TLI,
                              ScalarEvolution &SE,
                              const BoundsCheckingPass::Options &Opts) {
  if (F.hasFnAttribute(Attribute::NoSanitizeBounds))
    return false;

  const DataLayout &DL = F.getDataLayout();
  ObjectSizeOpts EvalOpts;
  EvalOpts.RoundToAlign = true;
  EvalOpts.EvalMode = ObjectSizeOpts::Mode::ExactUnderlyingSizeAndOffset;
  ObjectSizeOffsetEvaluator ObjSizeEval(DL, &TLI, F.getContext(), EvalOpts);

  // check HANDLE_MEMORY_INST in include/llvm/Instruction.def for memory
  // touching instructions
  SmallVector<std::pair<Instruction *, Value *>, 4> TrapInfo;
  for (Instruction &I : instructions(F)) {
    Value *Or = nullptr;
    BuilderTy IRB(I.getParent(), BasicBlock::iterator(&I), TargetFolder(DL));
    if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
      if (!LI->isVolatile())
        Or = getBoundsCheckCond(LI->getPointerOperand(), LI, DL, TLI,
                                ObjSizeEval, IRB, SE);
    } else if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
      if (!SI->isVolatile())
        Or = getBoundsCheckCond(SI->getPointerOperand(), SI->getValueOperand(),
                                DL, TLI, ObjSizeEval, IRB, SE);
    } else if (AtomicCmpXchgInst *AI = dyn_cast<AtomicCmpXchgInst>(&I)) {
      if (!AI->isVolatile())
        Or =
            getBoundsCheckCond(AI->getPointerOperand(), AI->getCompareOperand(),
                               DL, TLI, ObjSizeEval, IRB, SE);
    } else if (AtomicRMWInst *AI = dyn_cast<AtomicRMWInst>(&I)) {
      if (!AI->isVolatile())
        Or = getBoundsCheckCond(AI->getPointerOperand(), AI->getValOperand(),
                                DL, TLI, ObjSizeEval, IRB, SE);
    }
    if (Or)
      TrapInfo.push_back(std::make_pair(&I, Or));
  }

  std::string Name;
  if (Opts.Rt)
    Name = getRuntimeCallName(*Opts.Rt);

  // Create a trapping basic block on demand using a callback. Depending on
  // flags, this will either create a single block for the entire function or
  // will create a fresh block every time it is called.
  BasicBlock *ReuseTrapBB = nullptr;
  auto GetTrapBB = [&ReuseTrapBB, &Opts, &Name](BuilderTy &IRB,
                                                BasicBlock *Cont) {
    Function *Fn = IRB.GetInsertBlock()->getParent();
    auto DebugLoc = IRB.getCurrentDebugLocation();
    IRBuilder<>::InsertPointGuard Guard(IRB);

    // Create a trapping basic block on demand using a callback. Depending on
    // flags, this will either create a single block for the entire function or
    // will create a fresh block every time it is called.
    if (ReuseTrapBB)
      return ReuseTrapBB;

    BasicBlock *TrapBB = BasicBlock::Create(Fn->getContext(), "trap", Fn);
    IRB.SetInsertPoint(TrapBB);

    bool DebugTrapBB = !Opts.Merge;
    CallInst *TrapCall = Opts.Rt ? InsertCall(IRB, Opts.Rt->MayReturn, Name)
                                 : InsertTrap(IRB, DebugTrapBB);
    if (DebugTrapBB)
      TrapCall->addFnAttr(llvm::Attribute::NoMerge);

    TrapCall->setDoesNotThrow();
    TrapCall->setDebugLoc(DebugLoc);

    bool MayReturn = Opts.Rt && Opts.Rt->MayReturn;
    if (MayReturn) {
      IRB.CreateBr(Cont);
    } else {
      TrapCall->setDoesNotReturn();
      IRB.CreateUnreachable();
    }

    if (!MayReturn && SingleTrapBB && !DebugTrapBB)
      ReuseTrapBB = TrapBB;

    return TrapBB;
  };

  for (const auto &Entry : TrapInfo) {
    Instruction *Inst = Entry.first;
    BuilderTy IRB(Inst->getParent(), BasicBlock::iterator(Inst), TargetFolder(DL));
    insertBoundsCheck(Entry.second, IRB, GetTrapBB);
  }

  return !TrapInfo.empty();
}

PreservedAnalyses BoundsCheckingPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);

  if (!addBoundsChecking(F, TLI, SE, Opts))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

void BoundsCheckingPass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<BoundsCheckingPass> *>(this)->printPipeline(
      OS, MapClassName2PassName);
  OS << "<";
  if (Opts.Rt) {
    if (Opts.Rt->MinRuntime)
      OS << "min-";
    OS << "rt";
    if (!Opts.Rt->MayReturn)
      OS << "-abort";
  } else {
    OS << "trap";
  }
  if (Opts.Merge)
    OS << ";merge";
  OS << ">";
}
