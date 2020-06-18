//===- AMDGPUTargetTransformInfo.cpp - AMDGPU specific TTI pass -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements a TargetTransformInfo analysis pass specific to the
// AMDGPU target machine. It uses the target's detailed information to provide
// more precise answers to certain TTI queries, while letting the target
// independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetTransformInfo.h"
#include "AMDGPUSubtarget.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MachineValueType.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "AMDGPUtti"

static cl::opt<unsigned> UnrollThresholdPrivate(
  "amdgpu-unroll-threshold-private",
  cl::desc("Unroll threshold for AMDGPU if private memory used in a loop"),
  cl::init(2700), cl::Hidden);

static cl::opt<unsigned> UnrollThresholdLocal(
  "amdgpu-unroll-threshold-local",
  cl::desc("Unroll threshold for AMDGPU if local memory used in a loop"),
  cl::init(1000), cl::Hidden);

static cl::opt<unsigned> UnrollThresholdIf(
  "amdgpu-unroll-threshold-if",
  cl::desc("Unroll threshold increment for AMDGPU for each if statement inside loop"),
  cl::init(150), cl::Hidden);

static cl::opt<bool> UnrollRuntimeLocal(
  "amdgpu-unroll-runtime-local",
  cl::desc("Allow runtime unroll for AMDGPU if local memory used in a loop"),
  cl::init(true), cl::Hidden);

static cl::opt<bool> UseLegacyDA(
  "amdgpu-use-legacy-divergence-analysis",
  cl::desc("Enable legacy divergence analysis for AMDGPU"),
  cl::init(false), cl::Hidden);

static cl::opt<unsigned> UnrollMaxBlockToAnalyze(
    "amdgpu-unroll-max-block-to-analyze",
    cl::desc("Inner loop block size threshold to analyze in unroll for AMDGPU"),
    cl::init(20), cl::Hidden);

static bool dependsOnLocalPhi(const Loop *L, const Value *Cond,
                              unsigned Depth = 0) {
  const Instruction *I = dyn_cast<Instruction>(Cond);
  if (!I)
    return false;

  for (const Value *V : I->operand_values()) {
    if (!L->contains(I))
      continue;
    if (const PHINode *PHI = dyn_cast<PHINode>(V)) {
      if (llvm::none_of(L->getSubLoops(), [PHI](const Loop* SubLoop) {
                  return SubLoop->contains(PHI); }))
        return true;
    } else if (Depth < 10 && dependsOnLocalPhi(L, V, Depth+1))
      return true;
  }
  return false;
}

void AMDGPUTTIImpl::getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                                            TTI::UnrollingPreferences &UP) {
  const Function &F = *L->getHeader()->getParent();
  UP.Threshold = AMDGPU::getIntegerAttribute(F, "amdgpu-unroll-threshold", 300);
  UP.MaxCount = std::numeric_limits<unsigned>::max();
  UP.Partial = true;

  // TODO: Do we want runtime unrolling?

  // Maximum alloca size than can fit registers. Reserve 16 registers.
  const unsigned MaxAlloca = (256 - 16) * 4;
  unsigned ThresholdPrivate = UnrollThresholdPrivate;
  unsigned ThresholdLocal = UnrollThresholdLocal;
  unsigned MaxBoost = std::max(ThresholdPrivate, ThresholdLocal);
  for (const BasicBlock *BB : L->getBlocks()) {
    const DataLayout &DL = BB->getModule()->getDataLayout();
    unsigned LocalGEPsSeen = 0;

    if (llvm::any_of(L->getSubLoops(), [BB](const Loop* SubLoop) {
               return SubLoop->contains(BB); }))
        continue; // Block belongs to an inner loop.

    for (const Instruction &I : *BB) {
      // Unroll a loop which contains an "if" statement whose condition
      // defined by a PHI belonging to the loop. This may help to eliminate
      // if region and potentially even PHI itself, saving on both divergence
      // and registers used for the PHI.
      // Add a small bonus for each of such "if" statements.
      if (const BranchInst *Br = dyn_cast<BranchInst>(&I)) {
        if (UP.Threshold < MaxBoost && Br->isConditional()) {
          BasicBlock *Succ0 = Br->getSuccessor(0);
          BasicBlock *Succ1 = Br->getSuccessor(1);
          if ((L->contains(Succ0) && L->isLoopExiting(Succ0)) ||
              (L->contains(Succ1) && L->isLoopExiting(Succ1)))
            continue;
          if (dependsOnLocalPhi(L, Br->getCondition())) {
            UP.Threshold += UnrollThresholdIf;
            LLVM_DEBUG(dbgs() << "Set unroll threshold " << UP.Threshold
                              << " for loop:\n"
                              << *L << " due to " << *Br << '\n');
            if (UP.Threshold >= MaxBoost)
              return;
          }
        }
        continue;
      }

      const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&I);
      if (!GEP)
        continue;

      unsigned AS = GEP->getAddressSpace();
      unsigned Threshold = 0;
      if (AS == AMDGPUAS::PRIVATE_ADDRESS)
        Threshold = ThresholdPrivate;
      else if (AS == AMDGPUAS::LOCAL_ADDRESS || AS == AMDGPUAS::REGION_ADDRESS)
        Threshold = ThresholdLocal;
      else
        continue;

      if (UP.Threshold >= Threshold)
        continue;

      if (AS == AMDGPUAS::PRIVATE_ADDRESS) {
        const Value *Ptr = GEP->getPointerOperand();
        const AllocaInst *Alloca =
            dyn_cast<AllocaInst>(GetUnderlyingObject(Ptr, DL));
        if (!Alloca || !Alloca->isStaticAlloca())
          continue;
        Type *Ty = Alloca->getAllocatedType();
        unsigned AllocaSize = Ty->isSized() ? DL.getTypeAllocSize(Ty) : 0;
        if (AllocaSize > MaxAlloca)
          continue;
      } else if (AS == AMDGPUAS::LOCAL_ADDRESS ||
                 AS == AMDGPUAS::REGION_ADDRESS) {
        LocalGEPsSeen++;
        // Inhibit unroll for local memory if we have seen addressing not to
        // a variable, most likely we will be unable to combine it.
        // Do not unroll too deep inner loops for local memory to give a chance
        // to unroll an outer loop for a more important reason.
        if (LocalGEPsSeen > 1 || L->getLoopDepth() > 2 ||
            (!isa<GlobalVariable>(GEP->getPointerOperand()) &&
             !isa<Argument>(GEP->getPointerOperand())))
          continue;
        LLVM_DEBUG(dbgs() << "Allow unroll runtime for loop:\n"
                          << *L << " due to LDS use.\n");
        UP.Runtime = UnrollRuntimeLocal;
      }

      // Check if GEP depends on a value defined by this loop itself.
      bool HasLoopDef = false;
      for (const Value *Op : GEP->operands()) {
        const Instruction *Inst = dyn_cast<Instruction>(Op);
        if (!Inst || L->isLoopInvariant(Op))
          continue;

        if (llvm::any_of(L->getSubLoops(), [Inst](const Loop* SubLoop) {
             return SubLoop->contains(Inst); }))
          continue;
        HasLoopDef = true;
        break;
      }
      if (!HasLoopDef)
        continue;

      // We want to do whatever we can to limit the number of alloca
      // instructions that make it through to the code generator.  allocas
      // require us to use indirect addressing, which is slow and prone to
      // compiler bugs.  If this loop does an address calculation on an
      // alloca ptr, then we want to use a higher than normal loop unroll
      // threshold. This will give SROA a better chance to eliminate these
      // allocas.
      //
      // We also want to have more unrolling for local memory to let ds
      // instructions with different offsets combine.
      //
      // Don't use the maximum allowed value here as it will make some
      // programs way too big.
      UP.Threshold = Threshold;
      LLVM_DEBUG(dbgs() << "Set unroll threshold " << Threshold
                        << " for loop:\n"
                        << *L << " due to " << *GEP << '\n');
      if (UP.Threshold >= MaxBoost)
        return;
    }

    // If we got a GEP in a small BB from inner loop then increase max trip
    // count to analyze for better estimation cost in unroll
    if (L->empty() && BB->size() < UnrollMaxBlockToAnalyze)
      UP.MaxIterationsCountToAnalyze = 32;
  }
}

unsigned GCNTTIImpl::getHardwareNumberOfRegisters(bool Vec) const {
  // The concept of vector registers doesn't really exist. Some packed vector
  // operations operate on the normal 32-bit registers.
  return 256;
}

unsigned GCNTTIImpl::getNumberOfRegisters(bool Vec) const {
  // This is really the number of registers to fill when vectorizing /
  // interleaving loops, so we lie to avoid trying to use all registers.
  return getHardwareNumberOfRegisters(Vec) >> 3;
}

unsigned GCNTTIImpl::getRegisterBitWidth(bool Vector) const {
  return 32;
}

unsigned GCNTTIImpl::getMinVectorRegisterBitWidth() const {
  return 32;
}

unsigned GCNTTIImpl::getLoadVectorFactor(unsigned VF, unsigned LoadSize,
                                            unsigned ChainSizeInBytes,
                                            VectorType *VecTy) const {
  unsigned VecRegBitWidth = VF * LoadSize;
  if (VecRegBitWidth > 128 && VecTy->getScalarSizeInBits() < 32)
    // TODO: Support element-size less than 32bit?
    return 128 / LoadSize;

  return VF;
}

unsigned GCNTTIImpl::getStoreVectorFactor(unsigned VF, unsigned StoreSize,
                                             unsigned ChainSizeInBytes,
                                             VectorType *VecTy) const {
  unsigned VecRegBitWidth = VF * StoreSize;
  if (VecRegBitWidth > 128)
    return 128 / StoreSize;

  return VF;
}

unsigned GCNTTIImpl::getLoadStoreVecRegBitWidth(unsigned AddrSpace) const {
  if (AddrSpace == AMDGPUAS::GLOBAL_ADDRESS ||
      AddrSpace == AMDGPUAS::CONSTANT_ADDRESS ||
      AddrSpace == AMDGPUAS::CONSTANT_ADDRESS_32BIT ||
      AddrSpace == AMDGPUAS::BUFFER_FAT_POINTER) {
    return 512;
  }

  if (AddrSpace == AMDGPUAS::PRIVATE_ADDRESS)
    return 8 * ST->getMaxPrivateElementSize();

  // Common to flat, global, local and region. Assume for unknown addrspace.
  return 128;
}

bool GCNTTIImpl::isLegalToVectorizeMemChain(unsigned ChainSizeInBytes,
                                               unsigned Alignment,
                                               unsigned AddrSpace) const {
  // We allow vectorization of flat stores, even though we may need to decompose
  // them later if they may access private memory. We don't have enough context
  // here, and legalization can handle it.
  if (AddrSpace == AMDGPUAS::PRIVATE_ADDRESS) {
    return (Alignment >= 4 || ST->hasUnalignedScratchAccess()) &&
      ChainSizeInBytes <= ST->getMaxPrivateElementSize();
  }
  return true;
}

bool GCNTTIImpl::isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes,
                                                unsigned Alignment,
                                                unsigned AddrSpace) const {
  return isLegalToVectorizeMemChain(ChainSizeInBytes, Alignment, AddrSpace);
}

bool GCNTTIImpl::isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes,
                                                 unsigned Alignment,
                                                 unsigned AddrSpace) const {
  return isLegalToVectorizeMemChain(ChainSizeInBytes, Alignment, AddrSpace);
}

// FIXME: Really we would like to issue multiple 128-bit loads and stores per
// iteration. Should we report a larger size and let it legalize?
//
// FIXME: Should we use narrower types for local/region, or account for when
// unaligned access is legal?
//
// FIXME: This could use fine tuning and microbenchmarks.
Type *GCNTTIImpl::getMemcpyLoopLoweringType(LLVMContext &Context, Value *Length,
                                            unsigned SrcAddrSpace,
                                            unsigned DestAddrSpace,
                                            unsigned SrcAlign,
                                            unsigned DestAlign) const {
  unsigned MinAlign = std::min(SrcAlign, DestAlign);

  // A (multi-)dword access at an address == 2 (mod 4) will be decomposed by the
  // hardware into byte accesses. If you assume all alignments are equally
  // probable, it's more efficient on average to use short accesses for this
  // case.
  if (MinAlign == 2)
    return Type::getInt16Ty(Context);

  // Not all subtargets have 128-bit DS instructions, and we currently don't
  // form them by default.
  if (SrcAddrSpace == AMDGPUAS::LOCAL_ADDRESS ||
      SrcAddrSpace == AMDGPUAS::REGION_ADDRESS ||
      DestAddrSpace == AMDGPUAS::LOCAL_ADDRESS ||
      DestAddrSpace == AMDGPUAS::REGION_ADDRESS) {
    return FixedVectorType::get(Type::getInt32Ty(Context), 2);
  }

  // Global memory works best with 16-byte accesses. Private memory will also
  // hit this, although they'll be decomposed.
  return FixedVectorType::get(Type::getInt32Ty(Context), 4);
}

void GCNTTIImpl::getMemcpyLoopResidualLoweringType(
  SmallVectorImpl<Type *> &OpsOut, LLVMContext &Context,
  unsigned RemainingBytes, unsigned SrcAddrSpace, unsigned DestAddrSpace,
  unsigned SrcAlign, unsigned DestAlign) const {
  assert(RemainingBytes < 16);

  unsigned MinAlign = std::min(SrcAlign, DestAlign);

  if (MinAlign != 2) {
    Type *I64Ty = Type::getInt64Ty(Context);
    while (RemainingBytes >= 8) {
      OpsOut.push_back(I64Ty);
      RemainingBytes -= 8;
    }

    Type *I32Ty = Type::getInt32Ty(Context);
    while (RemainingBytes >= 4) {
      OpsOut.push_back(I32Ty);
      RemainingBytes -= 4;
    }
  }

  Type *I16Ty = Type::getInt16Ty(Context);
  while (RemainingBytes >= 2) {
    OpsOut.push_back(I16Ty);
    RemainingBytes -= 2;
  }

  Type *I8Ty = Type::getInt8Ty(Context);
  while (RemainingBytes) {
    OpsOut.push_back(I8Ty);
    --RemainingBytes;
  }
}

unsigned GCNTTIImpl::getMaxInterleaveFactor(unsigned VF) {
  // Disable unrolling if the loop is not vectorized.
  // TODO: Enable this again.
  if (VF == 1)
    return 1;

  return 8;
}

bool GCNTTIImpl::getTgtMemIntrinsic(IntrinsicInst *Inst,
                                       MemIntrinsicInfo &Info) const {
  switch (Inst->getIntrinsicID()) {
  case Intrinsic::amdgcn_atomic_inc:
  case Intrinsic::amdgcn_atomic_dec:
  case Intrinsic::amdgcn_ds_ordered_add:
  case Intrinsic::amdgcn_ds_ordered_swap:
  case Intrinsic::amdgcn_ds_fadd:
  case Intrinsic::amdgcn_ds_fmin:
  case Intrinsic::amdgcn_ds_fmax: {
    auto *Ordering = dyn_cast<ConstantInt>(Inst->getArgOperand(2));
    auto *Volatile = dyn_cast<ConstantInt>(Inst->getArgOperand(4));
    if (!Ordering || !Volatile)
      return false; // Invalid.

    unsigned OrderingVal = Ordering->getZExtValue();
    if (OrderingVal > static_cast<unsigned>(AtomicOrdering::SequentiallyConsistent))
      return false;

    Info.PtrVal = Inst->getArgOperand(0);
    Info.Ordering = static_cast<AtomicOrdering>(OrderingVal);
    Info.ReadMem = true;
    Info.WriteMem = true;
    Info.IsVolatile = !Volatile->isNullValue();
    return true;
  }
  default:
    return false;
  }
}

int GCNTTIImpl::getArithmeticInstrCost(unsigned Opcode, Type *Ty,
                                       TTI::TargetCostKind CostKind,
                                       TTI::OperandValueKind Opd1Info,
                                       TTI::OperandValueKind Opd2Info,
                                       TTI::OperandValueProperties Opd1PropInfo,
                                       TTI::OperandValueProperties Opd2PropInfo,
                                       ArrayRef<const Value *> Args,
                                       const Instruction *CxtI) {
  EVT OrigTy = TLI->getValueType(DL, Ty);
  if (!OrigTy.isSimple()) {
    // FIXME: We're having to query the throughput cost so that the basic
    // implementation tries to generate legalize and scalarization costs. Maybe
    // we could hoist the scalarization code here?
    return BaseT::getArithmeticInstrCost(Opcode, Ty, TTI::TCK_RecipThroughput,
                                         Opd1Info, Opd2Info,
                                         Opd1PropInfo, Opd2PropInfo);
  }

  // Legalize the type.
  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Ty);
  int ISD = TLI->InstructionOpcodeToISD(Opcode);

  // Because we don't have any legal vector operations, but the legal types, we
  // need to account for split vectors.
  unsigned NElts = LT.second.isVector() ?
    LT.second.getVectorNumElements() : 1;

  MVT::SimpleValueType SLT = LT.second.getScalarType().SimpleTy;

  switch (ISD) {
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
    if (SLT == MVT::i64)
      return get64BitInstrCost() * LT.first * NElts;

    if (ST->has16BitInsts() && SLT == MVT::i16)
      NElts = (NElts + 1) / 2;

    // i32
    return getFullRateInstrCost() * LT.first * NElts;
  case ISD::ADD:
  case ISD::SUB:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    if (SLT == MVT::i64) {
      // and, or and xor are typically split into 2 VALU instructions.
      return 2 * getFullRateInstrCost() * LT.first * NElts;
    }

    if (ST->has16BitInsts() && SLT == MVT::i16)
      NElts = (NElts + 1) / 2;

    return LT.first * NElts * getFullRateInstrCost();
  case ISD::MUL: {
    const int QuarterRateCost = getQuarterRateInstrCost();
    if (SLT == MVT::i64) {
      const int FullRateCost = getFullRateInstrCost();
      return (4 * QuarterRateCost + (2 * 2) * FullRateCost) * LT.first * NElts;
    }

    if (ST->has16BitInsts() && SLT == MVT::i16)
      NElts = (NElts + 1) / 2;

    // i32
    return QuarterRateCost * NElts * LT.first;
  }
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
    if (SLT == MVT::f64)
      return LT.first * NElts * get64BitInstrCost();

    if (ST->has16BitInsts() && SLT == MVT::f16)
      NElts = (NElts + 1) / 2;

    if (SLT == MVT::f32 || SLT == MVT::f16)
      return LT.first * NElts * getFullRateInstrCost();
    break;
  case ISD::FDIV:
  case ISD::FREM:
    // FIXME: frem should be handled separately. The fdiv in it is most of it,
    // but the current lowering is also not entirely correct.
    if (SLT == MVT::f64) {
      int Cost = 4 * get64BitInstrCost() + 7 * getQuarterRateInstrCost();
      // Add cost of workaround.
      if (!ST->hasUsableDivScaleConditionOutput())
        Cost += 3 * getFullRateInstrCost();

      return LT.first * Cost * NElts;
    }

    if (!Args.empty() && match(Args[0], PatternMatch::m_FPOne())) {
      // TODO: This is more complicated, unsafe flags etc.
      if ((SLT == MVT::f32 && !HasFP32Denormals) ||
          (SLT == MVT::f16 && ST->has16BitInsts())) {
        return LT.first * getQuarterRateInstrCost() * NElts;
      }
    }

    if (SLT == MVT::f16 && ST->has16BitInsts()) {
      // 2 x v_cvt_f32_f16
      // f32 rcp
      // f32 fmul
      // v_cvt_f16_f32
      // f16 div_fixup
      int Cost = 4 * getFullRateInstrCost() + 2 * getQuarterRateInstrCost();
      return LT.first * Cost * NElts;
    }

    if (SLT == MVT::f32 || SLT == MVT::f16) {
      int Cost = 7 * getFullRateInstrCost() + 1 * getQuarterRateInstrCost();

      if (!HasFP32Denormals) {
        // FP mode switches.
        Cost += 2 * getFullRateInstrCost();
      }

      return LT.first * NElts * Cost;
    }
    break;
  default:
    break;
  }

  return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Opd1Info,
                                       Opd2Info,
                                       Opd1PropInfo, Opd2PropInfo);
}

// Return true if there's a potential benefit from using v2f16 instructions for
// an intrinsic, even if it requires nontrivial legalization.
static bool intrinsicHasPackedVectorBenefit(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::fma: // TODO: fmuladd
  // There's a small benefit to using vector ops in the legalized code.
  case Intrinsic::round:
    return true;
  default:
    return false;
  }
}

int GCNTTIImpl::getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                      TTI::TargetCostKind CostKind) {
  if (ICA.getID() == Intrinsic::fabs)
    return 0;

  if (!intrinsicHasPackedVectorBenefit(ICA.getID()))
    return BaseT::getIntrinsicInstrCost(ICA, CostKind);

  Type *RetTy = ICA.getReturnType();
  EVT OrigTy = TLI->getValueType(DL, RetTy);
  if (!OrigTy.isSimple()) {
    return BaseT::getIntrinsicInstrCost(ICA, CostKind);
  }

  // Legalize the type.
  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, RetTy);

  unsigned NElts = LT.second.isVector() ?
    LT.second.getVectorNumElements() : 1;

  MVT::SimpleValueType SLT = LT.second.getScalarType().SimpleTy;

  if (SLT == MVT::f64)
    return LT.first * NElts * get64BitInstrCost();

  if (ST->has16BitInsts() && SLT == MVT::f16)
    NElts = (NElts + 1) / 2;

  // TODO: Get more refined intrinsic costs?
  unsigned InstRate = getQuarterRateInstrCost();
  if (ICA.getID() == Intrinsic::fma) {
    InstRate = ST->hasFastFMAF32() ? getHalfRateInstrCost()
                                   : getQuarterRateInstrCost();
  }

  return LT.first * NElts * InstRate;
}

unsigned GCNTTIImpl::getCFInstrCost(unsigned Opcode,
                                    TTI::TargetCostKind CostKind) {
  if (CostKind == TTI::TCK_CodeSize || CostKind == TTI::TCK_SizeAndLatency)
    return Opcode == Instruction::PHI ? 0 : 1;

  // XXX - For some reason this isn't called for switch.
  switch (Opcode) {
  case Instruction::Br:
  case Instruction::Ret:
    return 10;
  default:
    return BaseT::getCFInstrCost(Opcode, CostKind);
  }
}

int GCNTTIImpl::getArithmeticReductionCost(unsigned Opcode, VectorType *Ty,
                                           bool IsPairwise,
                                           TTI::TargetCostKind CostKind) {
  EVT OrigTy = TLI->getValueType(DL, Ty);

  // Computes cost on targets that have packed math instructions(which support
  // 16-bit types only).
  if (IsPairwise ||
      !ST->hasVOP3PInsts() ||
      OrigTy.getScalarSizeInBits() != 16)
    return BaseT::getArithmeticReductionCost(Opcode, Ty, IsPairwise, CostKind);

  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Ty);
  return LT.first * getFullRateInstrCost();
}

int GCNTTIImpl::getMinMaxReductionCost(VectorType *Ty, VectorType *CondTy,
                                       bool IsPairwise, bool IsUnsigned,
                                       TTI::TargetCostKind CostKind) {
  EVT OrigTy = TLI->getValueType(DL, Ty);

  // Computes cost on targets that have packed math instructions(which support
  // 16-bit types only).
  if (IsPairwise ||
      !ST->hasVOP3PInsts() ||
      OrigTy.getScalarSizeInBits() != 16)
    return BaseT::getMinMaxReductionCost(Ty, CondTy, IsPairwise, IsUnsigned,
                                         CostKind);

  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Ty);
  return LT.first * getHalfRateInstrCost();
}

int GCNTTIImpl::getVectorInstrCost(unsigned Opcode, Type *ValTy,
                                      unsigned Index) {
  switch (Opcode) {
  case Instruction::ExtractElement:
  case Instruction::InsertElement: {
    unsigned EltSize
      = DL.getTypeSizeInBits(cast<VectorType>(ValTy)->getElementType());
    if (EltSize < 32) {
      if (EltSize == 16 && Index == 0 && ST->has16BitInsts())
        return 0;
      return BaseT::getVectorInstrCost(Opcode, ValTy, Index);
    }

    // Extracts are just reads of a subregister, so are free. Inserts are
    // considered free because we don't want to have any cost for scalarizing
    // operations, and we don't have to copy into a different register class.

    // Dynamic indexing isn't free and is best avoided.
    return Index == ~0u ? 2 : 0;
  }
  default:
    return BaseT::getVectorInstrCost(Opcode, ValTy, Index);
  }
}

static bool isArgPassedInSGPR(const Argument *A) {
  const Function *F = A->getParent();

  // Arguments to compute shaders are never a source of divergence.
  CallingConv::ID CC = F->getCallingConv();
  switch (CC) {
  case CallingConv::AMDGPU_KERNEL:
  case CallingConv::SPIR_KERNEL:
    return true;
  case CallingConv::AMDGPU_VS:
  case CallingConv::AMDGPU_LS:
  case CallingConv::AMDGPU_HS:
  case CallingConv::AMDGPU_ES:
  case CallingConv::AMDGPU_GS:
  case CallingConv::AMDGPU_PS:
  case CallingConv::AMDGPU_CS:
    // For non-compute shaders, SGPR inputs are marked with either inreg or byval.
    // Everything else is in VGPRs.
    return F->getAttributes().hasParamAttribute(A->getArgNo(), Attribute::InReg) ||
           F->getAttributes().hasParamAttribute(A->getArgNo(), Attribute::ByVal);
  default:
    // TODO: Should calls support inreg for SGPR inputs?
    return false;
  }
}

/// Analyze if the results of inline asm are divergent. If \p Indices is empty,
/// this is analyzing the collective result of all output registers. Otherwise,
/// this is only querying a specific result index if this returns multiple
/// registers in a struct.
bool GCNTTIImpl::isInlineAsmSourceOfDivergence(
  const CallInst *CI, ArrayRef<unsigned> Indices) const {
  // TODO: Handle complex extract indices
  if (Indices.size() > 1)
    return true;

  const DataLayout &DL = CI->getModule()->getDataLayout();
  const SIRegisterInfo *TRI = ST->getRegisterInfo();
  TargetLowering::AsmOperandInfoVector TargetConstraints =
      TLI->ParseConstraints(DL, ST->getRegisterInfo(), *CI);

  const int TargetOutputIdx = Indices.empty() ? -1 : Indices[0];

  int OutputIdx = 0;
  for (auto &TC : TargetConstraints) {
    if (TC.Type != InlineAsm::isOutput)
      continue;

    // Skip outputs we don't care about.
    if (TargetOutputIdx != -1 && TargetOutputIdx != OutputIdx++)
      continue;

    TLI->ComputeConstraintToUse(TC, SDValue());

    Register AssignedReg;
    const TargetRegisterClass *RC;
    std::tie(AssignedReg, RC) = TLI->getRegForInlineAsmConstraint(
      TRI, TC.ConstraintCode, TC.ConstraintVT);
    if (AssignedReg) {
      // FIXME: This is a workaround for getRegForInlineAsmConstraint
      // returning VS_32
      RC = TRI->getPhysRegClass(AssignedReg);
    }

    // For AGPR constraints null is returned on subtargets without AGPRs, so
    // assume divergent for null.
    if (!RC || !TRI->isSGPRClass(RC))
      return true;
  }

  return false;
}

/// \returns true if the new GPU divergence analysis is enabled.
bool GCNTTIImpl::useGPUDivergenceAnalysis() const {
  return !UseLegacyDA;
}

/// \returns true if the result of the value could potentially be
/// different across workitems in a wavefront.
bool GCNTTIImpl::isSourceOfDivergence(const Value *V) const {
  if (const Argument *A = dyn_cast<Argument>(V))
    return !isArgPassedInSGPR(A);

  // Loads from the private and flat address spaces are divergent, because
  // threads can execute the load instruction with the same inputs and get
  // different results.
  //
  // All other loads are not divergent, because if threads issue loads with the
  // same arguments, they will always get the same result.
  if (const LoadInst *Load = dyn_cast<LoadInst>(V))
    return Load->getPointerAddressSpace() == AMDGPUAS::PRIVATE_ADDRESS ||
           Load->getPointerAddressSpace() == AMDGPUAS::FLAT_ADDRESS;

  // Atomics are divergent because they are executed sequentially: when an
  // atomic operation refers to the same address in each thread, then each
  // thread after the first sees the value written by the previous thread as
  // original value.
  if (isa<AtomicRMWInst>(V) || isa<AtomicCmpXchgInst>(V))
    return true;

  if (const IntrinsicInst *Intrinsic = dyn_cast<IntrinsicInst>(V))
    return AMDGPU::isIntrinsicSourceOfDivergence(Intrinsic->getIntrinsicID());

  // Assume all function calls are a source of divergence.
  if (const CallInst *CI = dyn_cast<CallInst>(V)) {
    if (CI->isInlineAsm())
      return isInlineAsmSourceOfDivergence(CI);
    return true;
  }

  // Assume all function calls are a source of divergence.
  if (isa<InvokeInst>(V))
    return true;

  return false;
}

bool GCNTTIImpl::isAlwaysUniform(const Value *V) const {
  if (const IntrinsicInst *Intrinsic = dyn_cast<IntrinsicInst>(V)) {
    switch (Intrinsic->getIntrinsicID()) {
    default:
      return false;
    case Intrinsic::amdgcn_readfirstlane:
    case Intrinsic::amdgcn_readlane:
    case Intrinsic::amdgcn_icmp:
    case Intrinsic::amdgcn_fcmp:
    case Intrinsic::amdgcn_ballot:
    case Intrinsic::amdgcn_if_break:
      return true;
    }
  }

  if (const CallInst *CI = dyn_cast<CallInst>(V)) {
    if (CI->isInlineAsm())
      return !isInlineAsmSourceOfDivergence(CI);
    return false;
  }

  const ExtractValueInst *ExtValue = dyn_cast<ExtractValueInst>(V);
  if (!ExtValue)
    return false;

  const CallInst *CI = dyn_cast<CallInst>(ExtValue->getOperand(0));
  if (!CI)
    return false;

  if (const IntrinsicInst *Intrinsic = dyn_cast<IntrinsicInst>(CI)) {
    switch (Intrinsic->getIntrinsicID()) {
    default:
      return false;
    case Intrinsic::amdgcn_if:
    case Intrinsic::amdgcn_else: {
      ArrayRef<unsigned> Indices = ExtValue->getIndices();
      return Indices.size() == 1 && Indices[0] == 1;
    }
    }
  }

  // If we have inline asm returning mixed SGPR and VGPR results, we inferred
  // divergent for the overall struct return. We need to override it in the
  // case we're extracting an SGPR component here.
  if (CI->isInlineAsm())
    return !isInlineAsmSourceOfDivergence(CI, ExtValue->getIndices());

  return false;
}

bool GCNTTIImpl::collectFlatAddressOperands(SmallVectorImpl<int> &OpIndexes,
                                            Intrinsic::ID IID) const {
  switch (IID) {
  case Intrinsic::amdgcn_atomic_inc:
  case Intrinsic::amdgcn_atomic_dec:
  case Intrinsic::amdgcn_ds_fadd:
  case Intrinsic::amdgcn_ds_fmin:
  case Intrinsic::amdgcn_ds_fmax:
  case Intrinsic::amdgcn_is_shared:
  case Intrinsic::amdgcn_is_private:
    OpIndexes.push_back(0);
    return true;
  default:
    return false;
  }
}

Value *GCNTTIImpl::rewriteIntrinsicWithAddressSpace(IntrinsicInst *II,
                                                    Value *OldV,
                                                    Value *NewV) const {
  auto IntrID = II->getIntrinsicID();
  switch (IntrID) {
  case Intrinsic::amdgcn_atomic_inc:
  case Intrinsic::amdgcn_atomic_dec:
  case Intrinsic::amdgcn_ds_fadd:
  case Intrinsic::amdgcn_ds_fmin:
  case Intrinsic::amdgcn_ds_fmax: {
    const ConstantInt *IsVolatile = cast<ConstantInt>(II->getArgOperand(4));
    if (!IsVolatile->isZero())
      return nullptr;
    Module *M = II->getParent()->getParent()->getParent();
    Type *DestTy = II->getType();
    Type *SrcTy = NewV->getType();
    Function *NewDecl =
        Intrinsic::getDeclaration(M, II->getIntrinsicID(), {DestTy, SrcTy});
    II->setArgOperand(0, NewV);
    II->setCalledFunction(NewDecl);
    return II;
  }
  case Intrinsic::amdgcn_is_shared:
  case Intrinsic::amdgcn_is_private: {
    unsigned TrueAS = IntrID == Intrinsic::amdgcn_is_shared ?
      AMDGPUAS::LOCAL_ADDRESS : AMDGPUAS::PRIVATE_ADDRESS;
    unsigned NewAS = NewV->getType()->getPointerAddressSpace();
    LLVMContext &Ctx = NewV->getType()->getContext();
    ConstantInt *NewVal = (TrueAS == NewAS) ?
      ConstantInt::getTrue(Ctx) : ConstantInt::getFalse(Ctx);
    return NewVal;
  }
  case Intrinsic::ptrmask: {
    unsigned OldAS = OldV->getType()->getPointerAddressSpace();
    unsigned NewAS = NewV->getType()->getPointerAddressSpace();
    Value *MaskOp = II->getArgOperand(1);
    Type *MaskTy = MaskOp->getType();

    bool DoTruncate = false;
    if (!getTLI()->isNoopAddrSpaceCast(OldAS, NewAS)) {
      // All valid 64-bit to 32-bit casts work by chopping off the high
      // bits. Any masking only clearing the low bits will also apply in the new
      // address space.
      if (DL.getPointerSizeInBits(OldAS) != 64 ||
          DL.getPointerSizeInBits(NewAS) != 32)
        return nullptr;

      // TODO: Do we need to thread more context in here?
      KnownBits Known = computeKnownBits(MaskOp, DL, 0, nullptr, II);
      if (Known.countMinLeadingOnes() < 32)
        return nullptr;

      DoTruncate = true;
    }

    IRBuilder<> B(II);
    if (DoTruncate) {
      MaskTy = B.getInt32Ty();
      MaskOp = B.CreateTrunc(MaskOp, MaskTy);
    }

    return B.CreateIntrinsic(Intrinsic::ptrmask, {NewV->getType(), MaskTy},
                             {NewV, MaskOp});
  }
  default:
    return nullptr;
  }
}

unsigned GCNTTIImpl::getShuffleCost(TTI::ShuffleKind Kind, VectorType *VT,
                                    int Index, VectorType *SubTp) {
  if (ST->hasVOP3PInsts()) {
    if (cast<FixedVectorType>(VT)->getNumElements() == 2 &&
        DL.getTypeSizeInBits(VT->getElementType()) == 16) {
      // With op_sel VOP3P instructions freely can access the low half or high
      // half of a register, so any swizzle is free.

      switch (Kind) {
      case TTI::SK_Broadcast:
      case TTI::SK_Reverse:
      case TTI::SK_PermuteSingleSrc:
        return 0;
      default:
        break;
      }
    }
  }

  return BaseT::getShuffleCost(Kind, VT, Index, SubTp);
}

bool GCNTTIImpl::areInlineCompatible(const Function *Caller,
                                     const Function *Callee) const {
  const TargetMachine &TM = getTLI()->getTargetMachine();
  const GCNSubtarget *CallerST
    = static_cast<const GCNSubtarget *>(TM.getSubtargetImpl(*Caller));
  const GCNSubtarget *CalleeST
    = static_cast<const GCNSubtarget *>(TM.getSubtargetImpl(*Callee));

  const FeatureBitset &CallerBits = CallerST->getFeatureBits();
  const FeatureBitset &CalleeBits = CalleeST->getFeatureBits();

  FeatureBitset RealCallerBits = CallerBits & ~InlineFeatureIgnoreList;
  FeatureBitset RealCalleeBits = CalleeBits & ~InlineFeatureIgnoreList;
  if ((RealCallerBits & RealCalleeBits) != RealCalleeBits)
    return false;

  // FIXME: dx10_clamp can just take the caller setting, but there seems to be
  // no way to support merge for backend defined attributes.
  AMDGPU::SIModeRegisterDefaults CallerMode(*Caller);
  AMDGPU::SIModeRegisterDefaults CalleeMode(*Callee);
  return CallerMode.isInlineCompatible(CalleeMode);
}

void GCNTTIImpl::getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                                         TTI::UnrollingPreferences &UP) {
  CommonTTI.getUnrollingPreferences(L, SE, UP);
}

unsigned R600TTIImpl::getHardwareNumberOfRegisters(bool Vec) const {
  return 4 * 128; // XXX - 4 channels. Should these count as vector instead?
}

unsigned R600TTIImpl::getNumberOfRegisters(bool Vec) const {
  return getHardwareNumberOfRegisters(Vec);
}

unsigned R600TTIImpl::getRegisterBitWidth(bool Vector) const {
  return 32;
}

unsigned R600TTIImpl::getMinVectorRegisterBitWidth() const {
  return 32;
}

unsigned R600TTIImpl::getLoadStoreVecRegBitWidth(unsigned AddrSpace) const {
  if (AddrSpace == AMDGPUAS::GLOBAL_ADDRESS ||
      AddrSpace == AMDGPUAS::CONSTANT_ADDRESS)
    return 128;
  if (AddrSpace == AMDGPUAS::LOCAL_ADDRESS ||
      AddrSpace == AMDGPUAS::REGION_ADDRESS)
    return 64;
  if (AddrSpace == AMDGPUAS::PRIVATE_ADDRESS)
    return 32;

  if ((AddrSpace == AMDGPUAS::PARAM_D_ADDRESS ||
      AddrSpace == AMDGPUAS::PARAM_I_ADDRESS ||
      (AddrSpace >= AMDGPUAS::CONSTANT_BUFFER_0 &&
      AddrSpace <= AMDGPUAS::CONSTANT_BUFFER_15)))
    return 128;
  llvm_unreachable("unhandled address space");
}

bool R600TTIImpl::isLegalToVectorizeMemChain(unsigned ChainSizeInBytes,
                                             unsigned Alignment,
                                             unsigned AddrSpace) const {
  // We allow vectorization of flat stores, even though we may need to decompose
  // them later if they may access private memory. We don't have enough context
  // here, and legalization can handle it.
  return (AddrSpace != AMDGPUAS::PRIVATE_ADDRESS);
}

bool R600TTIImpl::isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes,
                                              unsigned Alignment,
                                              unsigned AddrSpace) const {
  return isLegalToVectorizeMemChain(ChainSizeInBytes, Alignment, AddrSpace);
}

bool R600TTIImpl::isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes,
                                               unsigned Alignment,
                                               unsigned AddrSpace) const {
  return isLegalToVectorizeMemChain(ChainSizeInBytes, Alignment, AddrSpace);
}

unsigned R600TTIImpl::getMaxInterleaveFactor(unsigned VF) {
  // Disable unrolling if the loop is not vectorized.
  // TODO: Enable this again.
  if (VF == 1)
    return 1;

  return 8;
}

unsigned R600TTIImpl::getCFInstrCost(unsigned Opcode,
                                     TTI::TargetCostKind CostKind) {
  if (CostKind == TTI::TCK_CodeSize || CostKind == TTI::TCK_SizeAndLatency)
    return Opcode == Instruction::PHI ? 0 : 1;

  // XXX - For some reason this isn't called for switch.
  switch (Opcode) {
  case Instruction::Br:
  case Instruction::Ret:
    return 10;
  default:
    return BaseT::getCFInstrCost(Opcode, CostKind);
  }
}

int R600TTIImpl::getVectorInstrCost(unsigned Opcode, Type *ValTy,
                                    unsigned Index) {
  switch (Opcode) {
  case Instruction::ExtractElement:
  case Instruction::InsertElement: {
    unsigned EltSize
      = DL.getTypeSizeInBits(cast<VectorType>(ValTy)->getElementType());
    if (EltSize < 32) {
      return BaseT::getVectorInstrCost(Opcode, ValTy, Index);
    }

    // Extracts are just reads of a subregister, so are free. Inserts are
    // considered free because we don't want to have any cost for scalarizing
    // operations, and we don't have to copy into a different register class.

    // Dynamic indexing isn't free and is best avoided.
    return Index == ~0u ? 2 : 0;
  }
  default:
    return BaseT::getVectorInstrCost(Opcode, ValTy, Index);
  }
}

void R600TTIImpl::getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                                          TTI::UnrollingPreferences &UP) {
  CommonTTI.getUnrollingPreferences(L, SE, UP);
}
