//===-- AMDGPUAtomicOptimizer.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass optimizes atomic operations by using a single lane of a wavefront
/// to perform the atomic operation, thus reducing contention on that memory
/// location.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "amdgpu-atomic-optimizer"

using namespace llvm;
using namespace llvm::AMDGPU;

namespace {

struct ReplacementInfo {
  Instruction *I;
  AtomicRMWInst::BinOp Op;
  unsigned ValIdx;
  bool ValDivergent;
};

class AMDGPUAtomicOptimizer : public FunctionPass,
                              public InstVisitor<AMDGPUAtomicOptimizer> {
private:
  SmallVector<ReplacementInfo, 8> ToReplace;
  const LegacyDivergenceAnalysis *DA;
  const DataLayout *DL;
  DominatorTree *DT;
  const GCNSubtarget *ST;
  bool IsPixelShader;

  Value *buildScan(IRBuilder<> &B, AtomicRMWInst::BinOp Op, Value *V,
                   Value *const Identity) const;
  Value *buildShiftRight(IRBuilder<> &B, Value *V, Value *const Identity) const;
  void optimizeAtomic(Instruction &I, AtomicRMWInst::BinOp Op, unsigned ValIdx,
                      bool ValDivergent) const;

public:
  static char ID;

  AMDGPUAtomicOptimizer() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addRequired<LegacyDivergenceAnalysis>();
    AU.addRequired<TargetPassConfig>();
  }

  void visitAtomicRMWInst(AtomicRMWInst &I);
  void visitIntrinsicInst(IntrinsicInst &I);
};

} // namespace

char AMDGPUAtomicOptimizer::ID = 0;

char &llvm::AMDGPUAtomicOptimizerID = AMDGPUAtomicOptimizer::ID;

bool AMDGPUAtomicOptimizer::runOnFunction(Function &F) {
  if (skipFunction(F)) {
    return false;
  }

  DA = &getAnalysis<LegacyDivergenceAnalysis>();
  DL = &F.getParent()->getDataLayout();
  DominatorTreeWrapperPass *const DTW =
      getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  DT = DTW ? &DTW->getDomTree() : nullptr;
  const TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  const TargetMachine &TM = TPC.getTM<TargetMachine>();
  ST = &TM.getSubtarget<GCNSubtarget>(F);
  IsPixelShader = F.getCallingConv() == CallingConv::AMDGPU_PS;

  visit(F);

  const bool Changed = !ToReplace.empty();

  for (ReplacementInfo &Info : ToReplace) {
    optimizeAtomic(*Info.I, Info.Op, Info.ValIdx, Info.ValDivergent);
  }

  ToReplace.clear();

  return Changed;
}

void AMDGPUAtomicOptimizer::visitAtomicRMWInst(AtomicRMWInst &I) {
  // Early exit for unhandled address space atomic instructions.
  switch (I.getPointerAddressSpace()) {
  default:
    return;
  case AMDGPUAS::GLOBAL_ADDRESS:
  case AMDGPUAS::LOCAL_ADDRESS:
    break;
  }

  AtomicRMWInst::BinOp Op = I.getOperation();

  switch (Op) {
  default:
    return;
  case AtomicRMWInst::Add:
  case AtomicRMWInst::Sub:
  case AtomicRMWInst::And:
  case AtomicRMWInst::Or:
  case AtomicRMWInst::Xor:
  case AtomicRMWInst::Max:
  case AtomicRMWInst::Min:
  case AtomicRMWInst::UMax:
  case AtomicRMWInst::UMin:
    break;
  }

  const unsigned PtrIdx = 0;
  const unsigned ValIdx = 1;

  // If the pointer operand is divergent, then each lane is doing an atomic
  // operation on a different address, and we cannot optimize that.
  if (DA->isDivergentUse(&I.getOperandUse(PtrIdx))) {
    return;
  }

  const bool ValDivergent = DA->isDivergentUse(&I.getOperandUse(ValIdx));

  // If the value operand is divergent, each lane is contributing a different
  // value to the atomic calculation. We can only optimize divergent values if
  // we have DPP available on our subtarget, and the atomic operation is 32
  // bits.
  if (ValDivergent &&
      (!ST->hasDPP() || DL->getTypeSizeInBits(I.getType()) != 32)) {
    return;
  }

  // If we get here, we can optimize the atomic using a single wavefront-wide
  // atomic operation to do the calculation for the entire wavefront, so
  // remember the instruction so we can come back to it.
  const ReplacementInfo Info = {&I, Op, ValIdx, ValDivergent};

  ToReplace.push_back(Info);
}

void AMDGPUAtomicOptimizer::visitIntrinsicInst(IntrinsicInst &I) {
  AtomicRMWInst::BinOp Op;

  switch (I.getIntrinsicID()) {
  default:
    return;
  case Intrinsic::amdgcn_buffer_atomic_add:
  case Intrinsic::amdgcn_struct_buffer_atomic_add:
  case Intrinsic::amdgcn_raw_buffer_atomic_add:
    Op = AtomicRMWInst::Add;
    break;
  case Intrinsic::amdgcn_buffer_atomic_sub:
  case Intrinsic::amdgcn_struct_buffer_atomic_sub:
  case Intrinsic::amdgcn_raw_buffer_atomic_sub:
    Op = AtomicRMWInst::Sub;
    break;
  case Intrinsic::amdgcn_buffer_atomic_and:
  case Intrinsic::amdgcn_struct_buffer_atomic_and:
  case Intrinsic::amdgcn_raw_buffer_atomic_and:
    Op = AtomicRMWInst::And;
    break;
  case Intrinsic::amdgcn_buffer_atomic_or:
  case Intrinsic::amdgcn_struct_buffer_atomic_or:
  case Intrinsic::amdgcn_raw_buffer_atomic_or:
    Op = AtomicRMWInst::Or;
    break;
  case Intrinsic::amdgcn_buffer_atomic_xor:
  case Intrinsic::amdgcn_struct_buffer_atomic_xor:
  case Intrinsic::amdgcn_raw_buffer_atomic_xor:
    Op = AtomicRMWInst::Xor;
    break;
  case Intrinsic::amdgcn_buffer_atomic_smin:
  case Intrinsic::amdgcn_struct_buffer_atomic_smin:
  case Intrinsic::amdgcn_raw_buffer_atomic_smin:
    Op = AtomicRMWInst::Min;
    break;
  case Intrinsic::amdgcn_buffer_atomic_umin:
  case Intrinsic::amdgcn_struct_buffer_atomic_umin:
  case Intrinsic::amdgcn_raw_buffer_atomic_umin:
    Op = AtomicRMWInst::UMin;
    break;
  case Intrinsic::amdgcn_buffer_atomic_smax:
  case Intrinsic::amdgcn_struct_buffer_atomic_smax:
  case Intrinsic::amdgcn_raw_buffer_atomic_smax:
    Op = AtomicRMWInst::Max;
    break;
  case Intrinsic::amdgcn_buffer_atomic_umax:
  case Intrinsic::amdgcn_struct_buffer_atomic_umax:
  case Intrinsic::amdgcn_raw_buffer_atomic_umax:
    Op = AtomicRMWInst::UMax;
    break;
  }

  const unsigned ValIdx = 0;

  const bool ValDivergent = DA->isDivergentUse(&I.getOperandUse(ValIdx));

  // If the value operand is divergent, each lane is contributing a different
  // value to the atomic calculation. We can only optimize divergent values if
  // we have DPP available on our subtarget, and the atomic operation is 32
  // bits.
  if (ValDivergent &&
      (!ST->hasDPP() || DL->getTypeSizeInBits(I.getType()) != 32)) {
    return;
  }

  // If any of the other arguments to the intrinsic are divergent, we can't
  // optimize the operation.
  for (unsigned Idx = 1; Idx < I.getNumOperands(); Idx++) {
    if (DA->isDivergentUse(&I.getOperandUse(Idx))) {
      return;
    }
  }

  // If we get here, we can optimize the atomic using a single wavefront-wide
  // atomic operation to do the calculation for the entire wavefront, so
  // remember the instruction so we can come back to it.
  const ReplacementInfo Info = {&I, Op, ValIdx, ValDivergent};

  ToReplace.push_back(Info);
}

// Use the builder to create the non-atomic counterpart of the specified
// atomicrmw binary op.
static Value *buildNonAtomicBinOp(IRBuilder<> &B, AtomicRMWInst::BinOp Op,
                                  Value *LHS, Value *RHS) {
  CmpInst::Predicate Pred;

  switch (Op) {
  default:
    llvm_unreachable("Unhandled atomic op");
  case AtomicRMWInst::Add:
    return B.CreateBinOp(Instruction::Add, LHS, RHS);
  case AtomicRMWInst::Sub:
    return B.CreateBinOp(Instruction::Sub, LHS, RHS);
  case AtomicRMWInst::And:
    return B.CreateBinOp(Instruction::And, LHS, RHS);
  case AtomicRMWInst::Or:
    return B.CreateBinOp(Instruction::Or, LHS, RHS);
  case AtomicRMWInst::Xor:
    return B.CreateBinOp(Instruction::Xor, LHS, RHS);

  case AtomicRMWInst::Max:
    Pred = CmpInst::ICMP_SGT;
    break;
  case AtomicRMWInst::Min:
    Pred = CmpInst::ICMP_SLT;
    break;
  case AtomicRMWInst::UMax:
    Pred = CmpInst::ICMP_UGT;
    break;
  case AtomicRMWInst::UMin:
    Pred = CmpInst::ICMP_ULT;
    break;
  }
  Value *Cond = B.CreateICmp(Pred, LHS, RHS);
  return B.CreateSelect(Cond, LHS, RHS);
}

// Use the builder to create an inclusive scan of V across the wavefront, with
// all lanes active.
Value *AMDGPUAtomicOptimizer::buildScan(IRBuilder<> &B, AtomicRMWInst::BinOp Op,
                                        Value *V, Value *const Identity) const {
  Type *const Ty = V->getType();
  Module *M = B.GetInsertBlock()->getModule();
  Function *UpdateDPP =
      Intrinsic::getDeclaration(M, Intrinsic::amdgcn_update_dpp, Ty);
  Function *PermLaneX16 =
      Intrinsic::getDeclaration(M, Intrinsic::amdgcn_permlanex16, {});
  Function *ReadLane =
      Intrinsic::getDeclaration(M, Intrinsic::amdgcn_readlane, {});

  for (unsigned Idx = 0; Idx < 4; Idx++) {
    V = buildNonAtomicBinOp(
        B, Op, V,
        B.CreateCall(UpdateDPP,
                     {Identity, V, B.getInt32(DPP::ROW_SHR0 | 1 << Idx),
                      B.getInt32(0xf), B.getInt32(0xf), B.getFalse()}));
  }
  if (ST->hasDPPBroadcasts()) {
    // GFX9 has DPP row broadcast operations.
    V = buildNonAtomicBinOp(
        B, Op, V,
        B.CreateCall(UpdateDPP,
                     {Identity, V, B.getInt32(DPP::BCAST15), B.getInt32(0xa),
                      B.getInt32(0xf), B.getFalse()}));
    V = buildNonAtomicBinOp(
        B, Op, V,
        B.CreateCall(UpdateDPP,
                     {Identity, V, B.getInt32(DPP::BCAST31), B.getInt32(0xc),
                      B.getInt32(0xf), B.getFalse()}));
  } else {
    // On GFX10 all DPP operations are confined to a single row. To get cross-
    // row operations we have to use permlane or readlane.

    // Combine lane 15 into lanes 16..31 (and, for wave 64, lane 47 into lanes
    // 48..63).
    Value *const PermX =
        B.CreateCall(PermLaneX16, {V, V, B.getInt32(-1), B.getInt32(-1),
                                   B.getFalse(), B.getFalse()});
    V = buildNonAtomicBinOp(
        B, Op, V,
        B.CreateCall(UpdateDPP,
                     {Identity, PermX, B.getInt32(DPP::QUAD_PERM_ID),
                      B.getInt32(0xa), B.getInt32(0xf), B.getFalse()}));
    if (!ST->isWave32()) {
      // Combine lane 31 into lanes 32..63.
      Value *const Lane31 = B.CreateCall(ReadLane, {V, B.getInt32(31)});
      V = buildNonAtomicBinOp(
          B, Op, V,
          B.CreateCall(UpdateDPP,
                       {Identity, Lane31, B.getInt32(DPP::QUAD_PERM_ID),
                        B.getInt32(0xc), B.getInt32(0xf), B.getFalse()}));
    }
  }
  return V;
}

// Use the builder to create a shift right of V across the wavefront, with all
// lanes active, to turn an inclusive scan into an exclusive scan.
Value *AMDGPUAtomicOptimizer::buildShiftRight(IRBuilder<> &B, Value *V,
                                              Value *const Identity) const {
  Type *const Ty = V->getType();
  Module *M = B.GetInsertBlock()->getModule();
  Function *UpdateDPP =
      Intrinsic::getDeclaration(M, Intrinsic::amdgcn_update_dpp, Ty);
  Function *ReadLane =
      Intrinsic::getDeclaration(M, Intrinsic::amdgcn_readlane, {});
  Function *WriteLane =
      Intrinsic::getDeclaration(M, Intrinsic::amdgcn_writelane, {});

  if (ST->hasDPPWavefrontShifts()) {
    // GFX9 has DPP wavefront shift operations.
    V = B.CreateCall(UpdateDPP,
                     {Identity, V, B.getInt32(DPP::WAVE_SHR1), B.getInt32(0xf),
                      B.getInt32(0xf), B.getFalse()});
  } else {
    // On GFX10 all DPP operations are confined to a single row. To get cross-
    // row operations we have to use permlane or readlane.
    Value *Old = V;
    V = B.CreateCall(UpdateDPP,
                     {Identity, V, B.getInt32(DPP::ROW_SHR0 + 1),
                      B.getInt32(0xf), B.getInt32(0xf), B.getFalse()});

    // Copy the old lane 15 to the new lane 16.
    V = B.CreateCall(WriteLane, {B.CreateCall(ReadLane, {Old, B.getInt32(15)}),
                                 B.getInt32(16), V});

    if (!ST->isWave32()) {
      // Copy the old lane 31 to the new lane 32.
      V = B.CreateCall(
          WriteLane,
          {B.CreateCall(ReadLane, {Old, B.getInt32(31)}), B.getInt32(32), V});

      // Copy the old lane 47 to the new lane 48.
      V = B.CreateCall(
          WriteLane,
          {B.CreateCall(ReadLane, {Old, B.getInt32(47)}), B.getInt32(48), V});
    }
  }

  return V;
}

static APInt getIdentityValueForAtomicOp(AtomicRMWInst::BinOp Op,
                                         unsigned BitWidth) {
  switch (Op) {
  default:
    llvm_unreachable("Unhandled atomic op");
  case AtomicRMWInst::Add:
  case AtomicRMWInst::Sub:
  case AtomicRMWInst::Or:
  case AtomicRMWInst::Xor:
  case AtomicRMWInst::UMax:
    return APInt::getMinValue(BitWidth);
  case AtomicRMWInst::And:
  case AtomicRMWInst::UMin:
    return APInt::getMaxValue(BitWidth);
  case AtomicRMWInst::Max:
    return APInt::getSignedMinValue(BitWidth);
  case AtomicRMWInst::Min:
    return APInt::getSignedMaxValue(BitWidth);
  }
}

static Value *buildMul(IRBuilder<> &B, Value *LHS, Value *RHS) {
  const ConstantInt *CI = dyn_cast<ConstantInt>(LHS);
  return (CI && CI->isOne()) ? RHS : B.CreateMul(LHS, RHS);
}

void AMDGPUAtomicOptimizer::optimizeAtomic(Instruction &I,
                                           AtomicRMWInst::BinOp Op,
                                           unsigned ValIdx,
                                           bool ValDivergent) const {
  // Start building just before the instruction.
  IRBuilder<> B(&I);

  // If we are in a pixel shader, because of how we have to mask out helper
  // lane invocations, we need to record the entry and exit BB's.
  BasicBlock *PixelEntryBB = nullptr;
  BasicBlock *PixelExitBB = nullptr;

  // If we're optimizing an atomic within a pixel shader, we need to wrap the
  // entire atomic operation in a helper-lane check. We do not want any helper
  // lanes that are around only for the purposes of derivatives to take part
  // in any cross-lane communication, and we use a branch on whether the lane is
  // live to do this.
  if (IsPixelShader) {
    // Record I's original position as the entry block.
    PixelEntryBB = I.getParent();

    Value *const Cond = B.CreateIntrinsic(Intrinsic::amdgcn_ps_live, {}, {});
    Instruction *const NonHelperTerminator =
        SplitBlockAndInsertIfThen(Cond, &I, false, nullptr, DT, nullptr);

    // Record I's new position as the exit block.
    PixelExitBB = I.getParent();

    I.moveBefore(NonHelperTerminator);
    B.SetInsertPoint(&I);
  }

  Type *const Ty = I.getType();
  const unsigned TyBitWidth = DL->getTypeSizeInBits(Ty);
  auto *const VecTy = FixedVectorType::get(B.getInt32Ty(), 2);

  // This is the value in the atomic operation we need to combine in order to
  // reduce the number of atomic operations.
  Value *const V = I.getOperand(ValIdx);

  // We need to know how many lanes are active within the wavefront, and we do
  // this by doing a ballot of active lanes.
  Type *const WaveTy = B.getIntNTy(ST->getWavefrontSize());
  CallInst *const Ballot =
      B.CreateIntrinsic(Intrinsic::amdgcn_ballot, WaveTy, B.getTrue());

  // We need to know how many lanes are active within the wavefront that are
  // below us. If we counted each lane linearly starting from 0, a lane is
  // below us only if its associated index was less than ours. We do this by
  // using the mbcnt intrinsic.
  Value *Mbcnt;
  if (ST->isWave32()) {
    Mbcnt = B.CreateIntrinsic(Intrinsic::amdgcn_mbcnt_lo, {},
                              {Ballot, B.getInt32(0)});
  } else {
    Value *const BitCast = B.CreateBitCast(Ballot, VecTy);
    Value *const ExtractLo = B.CreateExtractElement(BitCast, B.getInt32(0));
    Value *const ExtractHi = B.CreateExtractElement(BitCast, B.getInt32(1));
    Mbcnt = B.CreateIntrinsic(Intrinsic::amdgcn_mbcnt_lo, {},
                              {ExtractLo, B.getInt32(0)});
    Mbcnt =
        B.CreateIntrinsic(Intrinsic::amdgcn_mbcnt_hi, {}, {ExtractHi, Mbcnt});
  }
  Mbcnt = B.CreateIntCast(Mbcnt, Ty, false);

  Value *const Identity = B.getInt(getIdentityValueForAtomicOp(Op, TyBitWidth));

  Value *ExclScan = nullptr;
  Value *NewV = nullptr;

  // If we have a divergent value in each lane, we need to combine the value
  // using DPP.
  if (ValDivergent) {
    // First we need to set all inactive invocations to the identity value, so
    // that they can correctly contribute to the final result.
    NewV = B.CreateIntrinsic(Intrinsic::amdgcn_set_inactive, Ty, {V, Identity});

    const AtomicRMWInst::BinOp ScanOp =
        Op == AtomicRMWInst::Sub ? AtomicRMWInst::Add : Op;
    NewV = buildScan(B, ScanOp, NewV, Identity);
    ExclScan = buildShiftRight(B, NewV, Identity);

    // Read the value from the last lane, which has accumlated the values of
    // each active lane in the wavefront. This will be our new value which we
    // will provide to the atomic operation.
    Value *const LastLaneIdx = B.getInt32(ST->getWavefrontSize() - 1);
    if (TyBitWidth == 64) {
      Value *const ExtractLo = B.CreateTrunc(NewV, B.getInt32Ty());
      Value *const ExtractHi =
          B.CreateTrunc(B.CreateLShr(NewV, 32), B.getInt32Ty());
      CallInst *const ReadLaneLo = B.CreateIntrinsic(
          Intrinsic::amdgcn_readlane, {}, {ExtractLo, LastLaneIdx});
      CallInst *const ReadLaneHi = B.CreateIntrinsic(
          Intrinsic::amdgcn_readlane, {}, {ExtractHi, LastLaneIdx});
      Value *const PartialInsert = B.CreateInsertElement(
          UndefValue::get(VecTy), ReadLaneLo, B.getInt32(0));
      Value *const Insert =
          B.CreateInsertElement(PartialInsert, ReadLaneHi, B.getInt32(1));
      NewV = B.CreateBitCast(Insert, Ty);
    } else if (TyBitWidth == 32) {
      NewV = B.CreateIntrinsic(Intrinsic::amdgcn_readlane, {},
                               {NewV, LastLaneIdx});
    } else {
      llvm_unreachable("Unhandled atomic bit width");
    }

    // Finally mark the readlanes in the WWM section.
    NewV = B.CreateIntrinsic(Intrinsic::amdgcn_wwm, Ty, NewV);
  } else {
    switch (Op) {
    default:
      llvm_unreachable("Unhandled atomic op");

    case AtomicRMWInst::Add:
    case AtomicRMWInst::Sub: {
      // The new value we will be contributing to the atomic operation is the
      // old value times the number of active lanes.
      Value *const Ctpop = B.CreateIntCast(
          B.CreateUnaryIntrinsic(Intrinsic::ctpop, Ballot), Ty, false);
      NewV = buildMul(B, V, Ctpop);
      break;
    }

    case AtomicRMWInst::And:
    case AtomicRMWInst::Or:
    case AtomicRMWInst::Max:
    case AtomicRMWInst::Min:
    case AtomicRMWInst::UMax:
    case AtomicRMWInst::UMin:
      // These operations with a uniform value are idempotent: doing the atomic
      // operation multiple times has the same effect as doing it once.
      NewV = V;
      break;

    case AtomicRMWInst::Xor:
      // The new value we will be contributing to the atomic operation is the
      // old value times the parity of the number of active lanes.
      Value *const Ctpop = B.CreateIntCast(
          B.CreateUnaryIntrinsic(Intrinsic::ctpop, Ballot), Ty, false);
      NewV = buildMul(B, V, B.CreateAnd(Ctpop, 1));
      break;
    }
  }

  // We only want a single lane to enter our new control flow, and we do this
  // by checking if there are any active lanes below us. Only one lane will
  // have 0 active lanes below us, so that will be the only one to progress.
  Value *const Cond = B.CreateICmpEQ(Mbcnt, B.getIntN(TyBitWidth, 0));

  // Store I's original basic block before we split the block.
  BasicBlock *const EntryBB = I.getParent();

  // We need to introduce some new control flow to force a single lane to be
  // active. We do this by splitting I's basic block at I, and introducing the
  // new block such that:
  // entry --> single_lane -\
  //       \------------------> exit
  Instruction *const SingleLaneTerminator =
      SplitBlockAndInsertIfThen(Cond, &I, false, nullptr, DT, nullptr);

  // Move the IR builder into single_lane next.
  B.SetInsertPoint(SingleLaneTerminator);

  // Clone the original atomic operation into single lane, replacing the
  // original value with our newly created one.
  Instruction *const NewI = I.clone();
  B.Insert(NewI);
  NewI->setOperand(ValIdx, NewV);

  // Move the IR builder into exit next, and start inserting just before the
  // original instruction.
  B.SetInsertPoint(&I);

  const bool NeedResult = !I.use_empty();
  if (NeedResult) {
    // Create a PHI node to get our new atomic result into the exit block.
    PHINode *const PHI = B.CreatePHI(Ty, 2);
    PHI->addIncoming(UndefValue::get(Ty), EntryBB);
    PHI->addIncoming(NewI, SingleLaneTerminator->getParent());

    // We need to broadcast the value who was the lowest active lane (the first
    // lane) to all other lanes in the wavefront. We use an intrinsic for this,
    // but have to handle 64-bit broadcasts with two calls to this intrinsic.
    Value *BroadcastI = nullptr;

    if (TyBitWidth == 64) {
      Value *const ExtractLo = B.CreateTrunc(PHI, B.getInt32Ty());
      Value *const ExtractHi =
          B.CreateTrunc(B.CreateLShr(PHI, 32), B.getInt32Ty());
      CallInst *const ReadFirstLaneLo =
          B.CreateIntrinsic(Intrinsic::amdgcn_readfirstlane, {}, ExtractLo);
      CallInst *const ReadFirstLaneHi =
          B.CreateIntrinsic(Intrinsic::amdgcn_readfirstlane, {}, ExtractHi);
      Value *const PartialInsert = B.CreateInsertElement(
          UndefValue::get(VecTy), ReadFirstLaneLo, B.getInt32(0));
      Value *const Insert =
          B.CreateInsertElement(PartialInsert, ReadFirstLaneHi, B.getInt32(1));
      BroadcastI = B.CreateBitCast(Insert, Ty);
    } else if (TyBitWidth == 32) {

      BroadcastI = B.CreateIntrinsic(Intrinsic::amdgcn_readfirstlane, {}, PHI);
    } else {
      llvm_unreachable("Unhandled atomic bit width");
    }

    // Now that we have the result of our single atomic operation, we need to
    // get our individual lane's slice into the result. We use the lane offset
    // we previously calculated combined with the atomic result value we got
    // from the first lane, to get our lane's index into the atomic result.
    Value *LaneOffset = nullptr;
    if (ValDivergent) {
      LaneOffset = B.CreateIntrinsic(Intrinsic::amdgcn_wwm, Ty, ExclScan);
    } else {
      switch (Op) {
      default:
        llvm_unreachable("Unhandled atomic op");
      case AtomicRMWInst::Add:
      case AtomicRMWInst::Sub:
        LaneOffset = buildMul(B, V, Mbcnt);
        break;
      case AtomicRMWInst::And:
      case AtomicRMWInst::Or:
      case AtomicRMWInst::Max:
      case AtomicRMWInst::Min:
      case AtomicRMWInst::UMax:
      case AtomicRMWInst::UMin:
        LaneOffset = B.CreateSelect(Cond, Identity, V);
        break;
      case AtomicRMWInst::Xor:
        LaneOffset = buildMul(B, V, B.CreateAnd(Mbcnt, 1));
        break;
      }
    }
    Value *const Result = buildNonAtomicBinOp(B, Op, BroadcastI, LaneOffset);

    if (IsPixelShader) {
      // Need a final PHI to reconverge to above the helper lane branch mask.
      B.SetInsertPoint(PixelExitBB->getFirstNonPHI());

      PHINode *const PHI = B.CreatePHI(Ty, 2);
      PHI->addIncoming(UndefValue::get(Ty), PixelEntryBB);
      PHI->addIncoming(Result, I.getParent());
      I.replaceAllUsesWith(PHI);
    } else {
      // Replace the original atomic instruction with the new one.
      I.replaceAllUsesWith(Result);
    }
  }

  // And delete the original.
  I.eraseFromParent();
}

INITIALIZE_PASS_BEGIN(AMDGPUAtomicOptimizer, DEBUG_TYPE,
                      "AMDGPU atomic optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPUAtomicOptimizer, DEBUG_TYPE,
                    "AMDGPU atomic optimizations", false, false)

FunctionPass *llvm::createAMDGPUAtomicOptimizerPass() {
  return new AMDGPUAtomicOptimizer();
}
