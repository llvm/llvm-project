//===-- PPCTargetTransformInfo.cpp - PPC specific TTI ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PPCTargetTransformInfo.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/CostTable.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "ppctti"

static cl::opt<bool> DisablePPCConstHoist("disable-ppc-constant-hoisting",
cl::desc("disable constant hoisting on PPC"), cl::init(false), cl::Hidden);

// This is currently only used for the data prefetch pass which is only enabled
// for BG/Q by default.
static cl::opt<unsigned>
CacheLineSize("ppc-loop-prefetch-cache-line", cl::Hidden, cl::init(64),
              cl::desc("The loop prefetch cache line size"));

static cl::opt<bool>
EnablePPCColdCC("ppc-enable-coldcc", cl::Hidden, cl::init(false),
                cl::desc("Enable using coldcc calling conv for cold "
                         "internal functions"));

static cl::opt<bool>
LsrNoInsnsCost("ppc-lsr-no-insns-cost", cl::Hidden, cl::init(false),
               cl::desc("Do not add instruction count to lsr cost model"));

// The latency of mtctr is only justified if there are more than 4
// comparisons that will be removed as a result.
static cl::opt<unsigned>
SmallCTRLoopThreshold("min-ctr-loop-threshold", cl::init(4), cl::Hidden,
                      cl::desc("Loops with a constant trip count smaller than "
                               "this value will not use the count register."));

//===----------------------------------------------------------------------===//
//
// PPC cost model.
//
//===----------------------------------------------------------------------===//

TargetTransformInfo::PopcntSupportKind
PPCTTIImpl::getPopcntSupport(unsigned TyWidth) {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  if (ST->hasPOPCNTD() != PPCSubtarget::POPCNTD_Unavailable && TyWidth <= 64)
    return ST->hasPOPCNTD() == PPCSubtarget::POPCNTD_Slow ?
             TTI::PSK_SlowHardware : TTI::PSK_FastHardware;
  return TTI::PSK_Software;
}

int PPCTTIImpl::getIntImmCost(const APInt &Imm, Type *Ty,
                              TTI::TargetCostKind CostKind) {
  if (DisablePPCConstHoist)
    return BaseT::getIntImmCost(Imm, Ty, CostKind);

  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  if (BitSize == 0)
    return ~0U;

  if (Imm == 0)
    return TTI::TCC_Free;

  if (Imm.getBitWidth() <= 64) {
    if (isInt<16>(Imm.getSExtValue()))
      return TTI::TCC_Basic;

    if (isInt<32>(Imm.getSExtValue())) {
      // A constant that can be materialized using lis.
      if ((Imm.getZExtValue() & 0xFFFF) == 0)
        return TTI::TCC_Basic;

      return 2 * TTI::TCC_Basic;
    }
  }

  return 4 * TTI::TCC_Basic;
}

int PPCTTIImpl::getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx,
                                    const APInt &Imm, Type *Ty,
                                    TTI::TargetCostKind CostKind) {
  if (DisablePPCConstHoist)
    return BaseT::getIntImmCostIntrin(IID, Idx, Imm, Ty, CostKind);

  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  if (BitSize == 0)
    return ~0U;

  switch (IID) {
  default:
    return TTI::TCC_Free;
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::usub_with_overflow:
    if ((Idx == 1) && Imm.getBitWidth() <= 64 && isInt<16>(Imm.getSExtValue()))
      return TTI::TCC_Free;
    break;
  case Intrinsic::experimental_stackmap:
    if ((Idx < 2) || (Imm.getBitWidth() <= 64 && isInt<64>(Imm.getSExtValue())))
      return TTI::TCC_Free;
    break;
  case Intrinsic::experimental_patchpoint_void:
  case Intrinsic::experimental_patchpoint_i64:
    if ((Idx < 4) || (Imm.getBitWidth() <= 64 && isInt<64>(Imm.getSExtValue())))
      return TTI::TCC_Free;
    break;
  }
  return PPCTTIImpl::getIntImmCost(Imm, Ty, CostKind);
}

int PPCTTIImpl::getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                  const APInt &Imm, Type *Ty,
                                  TTI::TargetCostKind CostKind) {
  if (DisablePPCConstHoist)
    return BaseT::getIntImmCostInst(Opcode, Idx, Imm, Ty, CostKind);

  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  if (BitSize == 0)
    return ~0U;

  unsigned ImmIdx = ~0U;
  bool ShiftedFree = false, RunFree = false, UnsignedFree = false,
       ZeroFree = false;
  switch (Opcode) {
  default:
    return TTI::TCC_Free;
  case Instruction::GetElementPtr:
    // Always hoist the base address of a GetElementPtr. This prevents the
    // creation of new constants for every base constant that gets constant
    // folded with the offset.
    if (Idx == 0)
      return 2 * TTI::TCC_Basic;
    return TTI::TCC_Free;
  case Instruction::And:
    RunFree = true; // (for the rotate-and-mask instructions)
    LLVM_FALLTHROUGH;
  case Instruction::Add:
  case Instruction::Or:
  case Instruction::Xor:
    ShiftedFree = true;
    LLVM_FALLTHROUGH;
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    ImmIdx = 1;
    break;
  case Instruction::ICmp:
    UnsignedFree = true;
    ImmIdx = 1;
    // Zero comparisons can use record-form instructions.
    LLVM_FALLTHROUGH;
  case Instruction::Select:
    ZeroFree = true;
    break;
  case Instruction::PHI:
  case Instruction::Call:
  case Instruction::Ret:
  case Instruction::Load:
  case Instruction::Store:
    break;
  }

  if (ZeroFree && Imm == 0)
    return TTI::TCC_Free;

  if (Idx == ImmIdx && Imm.getBitWidth() <= 64) {
    if (isInt<16>(Imm.getSExtValue()))
      return TTI::TCC_Free;

    if (RunFree) {
      if (Imm.getBitWidth() <= 32 &&
          (isShiftedMask_32(Imm.getZExtValue()) ||
           isShiftedMask_32(~Imm.getZExtValue())))
        return TTI::TCC_Free;

      if (ST->isPPC64() &&
          (isShiftedMask_64(Imm.getZExtValue()) ||
           isShiftedMask_64(~Imm.getZExtValue())))
        return TTI::TCC_Free;
    }

    if (UnsignedFree && isUInt<16>(Imm.getZExtValue()))
      return TTI::TCC_Free;

    if (ShiftedFree && (Imm.getZExtValue() & 0xFFFF) == 0)
      return TTI::TCC_Free;
  }

  return PPCTTIImpl::getIntImmCost(Imm, Ty, CostKind);
}

unsigned
PPCTTIImpl::getUserCost(const User *U, ArrayRef<const Value *> Operands,
                        TTI::TargetCostKind CostKind) {
  // We already implement getCastInstrCost and getMemoryOpCost where we perform
  // the vector adjustment there.
  if (isa<CastInst>(U) || isa<LoadInst>(U) || isa<StoreInst>(U))
    return BaseT::getUserCost(U, Operands, CostKind);

  if (U->getType()->isVectorTy()) {
    // Instructions that need to be split should cost more.
    std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, U->getType());
    return LT.first * BaseT::getUserCost(U, Operands, CostKind);
  }

  return BaseT::getUserCost(U, Operands, CostKind);
}

bool PPCTTIImpl::mightUseCTR(BasicBlock *BB, TargetLibraryInfo *LibInfo,
                             SmallPtrSetImpl<const Value *> &Visited) {
  const PPCTargetMachine &TM = ST->getTargetMachine();

  // Loop through the inline asm constraints and look for something that
  // clobbers ctr.
  auto asmClobbersCTR = [](InlineAsm *IA) {
    InlineAsm::ConstraintInfoVector CIV = IA->ParseConstraints();
    for (unsigned i = 0, ie = CIV.size(); i < ie; ++i) {
      InlineAsm::ConstraintInfo &C = CIV[i];
      if (C.Type != InlineAsm::isInput)
        for (unsigned j = 0, je = C.Codes.size(); j < je; ++j)
          if (StringRef(C.Codes[j]).equals_lower("{ctr}"))
            return true;
    }
    return false;
  };

  // Determining the address of a TLS variable results in a function call in
  // certain TLS models.
  std::function<bool(const Value *)> memAddrUsesCTR =
      [&memAddrUsesCTR, &TM, &Visited](const Value *MemAddr) -> bool {
    // No need to traverse again if we already checked this operand.
    if (!Visited.insert(MemAddr).second)
      return false;
    const auto *GV = dyn_cast<GlobalValue>(MemAddr);
    if (!GV) {
      // Recurse to check for constants that refer to TLS global variables.
      if (const auto *CV = dyn_cast<Constant>(MemAddr))
        for (const auto &CO : CV->operands())
          if (memAddrUsesCTR(CO))
            return true;

      return false;
    }

    if (!GV->isThreadLocal())
      return false;
    TLSModel::Model Model = TM.getTLSModel(GV);
    return Model == TLSModel::GeneralDynamic ||
      Model == TLSModel::LocalDynamic;
  };

  auto isLargeIntegerTy = [](bool Is32Bit, Type *Ty) {
    if (IntegerType *ITy = dyn_cast<IntegerType>(Ty))
      return ITy->getBitWidth() > (Is32Bit ? 32U : 64U);

    return false;
  };

  for (BasicBlock::iterator J = BB->begin(), JE = BB->end();
       J != JE; ++J) {
    if (CallInst *CI = dyn_cast<CallInst>(J)) {
      // Inline ASM is okay, unless it clobbers the ctr register.
      if (InlineAsm *IA = dyn_cast<InlineAsm>(CI->getCalledOperand())) {
        if (asmClobbersCTR(IA))
          return true;
        continue;
      }

      if (Function *F = CI->getCalledFunction()) {
        // Most intrinsics don't become function calls, but some might.
        // sin, cos, exp and log are always calls.
        unsigned Opcode = 0;
        if (F->getIntrinsicID() != Intrinsic::not_intrinsic) {
          switch (F->getIntrinsicID()) {
          default: continue;
          // If we have a call to ppc_is_decremented_ctr_nonzero, or ppc_mtctr
          // we're definitely using CTR.
          case Intrinsic::set_loop_iterations:
          case Intrinsic::loop_decrement:
            return true;

          // Exclude eh_sjlj_setjmp; we don't need to exclude eh_sjlj_longjmp
          // because, although it does clobber the counter register, the
          // control can't then return to inside the loop unless there is also
          // an eh_sjlj_setjmp.
          case Intrinsic::eh_sjlj_setjmp:

          case Intrinsic::memcpy:
          case Intrinsic::memmove:
          case Intrinsic::memset:
          case Intrinsic::powi:
          case Intrinsic::log:
          case Intrinsic::log2:
          case Intrinsic::log10:
          case Intrinsic::exp:
          case Intrinsic::exp2:
          case Intrinsic::pow:
          case Intrinsic::sin:
          case Intrinsic::cos:
            return true;
          case Intrinsic::copysign:
            if (CI->getArgOperand(0)->getType()->getScalarType()->
                isPPC_FP128Ty())
              return true;
            else
              continue; // ISD::FCOPYSIGN is never a library call.
          case Intrinsic::fma:                Opcode = ISD::FMA;        break;
          case Intrinsic::sqrt:               Opcode = ISD::FSQRT;      break;
          case Intrinsic::floor:              Opcode = ISD::FFLOOR;     break;
          case Intrinsic::ceil:               Opcode = ISD::FCEIL;      break;
          case Intrinsic::trunc:              Opcode = ISD::FTRUNC;     break;
          case Intrinsic::rint:               Opcode = ISD::FRINT;      break;
          case Intrinsic::lrint:              Opcode = ISD::LRINT;      break;
          case Intrinsic::llrint:             Opcode = ISD::LLRINT;     break;
          case Intrinsic::nearbyint:          Opcode = ISD::FNEARBYINT; break;
          case Intrinsic::round:              Opcode = ISD::FROUND;     break;
          case Intrinsic::lround:             Opcode = ISD::LROUND;     break;
          case Intrinsic::llround:            Opcode = ISD::LLROUND;    break;
          case Intrinsic::minnum:             Opcode = ISD::FMINNUM;    break;
          case Intrinsic::maxnum:             Opcode = ISD::FMAXNUM;    break;
          case Intrinsic::umul_with_overflow: Opcode = ISD::UMULO;      break;
          case Intrinsic::smul_with_overflow: Opcode = ISD::SMULO;      break;
          }
        }

        // PowerPC does not use [US]DIVREM or other library calls for
        // operations on regular types which are not otherwise library calls
        // (i.e. soft float or atomics). If adapting for targets that do,
        // additional care is required here.

        LibFunc Func;
        if (!F->hasLocalLinkage() && F->hasName() && LibInfo &&
            LibInfo->getLibFunc(F->getName(), Func) &&
            LibInfo->hasOptimizedCodeGen(Func)) {
          // Non-read-only functions are never treated as intrinsics.
          if (!CI->onlyReadsMemory())
            return true;

          // Conversion happens only for FP calls.
          if (!CI->getArgOperand(0)->getType()->isFloatingPointTy())
            return true;

          switch (Func) {
          default: return true;
          case LibFunc_copysign:
          case LibFunc_copysignf:
            continue; // ISD::FCOPYSIGN is never a library call.
          case LibFunc_copysignl:
            return true;
          case LibFunc_fabs:
          case LibFunc_fabsf:
          case LibFunc_fabsl:
            continue; // ISD::FABS is never a library call.
          case LibFunc_sqrt:
          case LibFunc_sqrtf:
          case LibFunc_sqrtl:
            Opcode = ISD::FSQRT; break;
          case LibFunc_floor:
          case LibFunc_floorf:
          case LibFunc_floorl:
            Opcode = ISD::FFLOOR; break;
          case LibFunc_nearbyint:
          case LibFunc_nearbyintf:
          case LibFunc_nearbyintl:
            Opcode = ISD::FNEARBYINT; break;
          case LibFunc_ceil:
          case LibFunc_ceilf:
          case LibFunc_ceill:
            Opcode = ISD::FCEIL; break;
          case LibFunc_rint:
          case LibFunc_rintf:
          case LibFunc_rintl:
            Opcode = ISD::FRINT; break;
          case LibFunc_round:
          case LibFunc_roundf:
          case LibFunc_roundl:
            Opcode = ISD::FROUND; break;
          case LibFunc_trunc:
          case LibFunc_truncf:
          case LibFunc_truncl:
            Opcode = ISD::FTRUNC; break;
          case LibFunc_fmin:
          case LibFunc_fminf:
          case LibFunc_fminl:
            Opcode = ISD::FMINNUM; break;
          case LibFunc_fmax:
          case LibFunc_fmaxf:
          case LibFunc_fmaxl:
            Opcode = ISD::FMAXNUM; break;
          }
        }

        if (Opcode) {
          EVT EVTy =
              TLI->getValueType(DL, CI->getArgOperand(0)->getType(), true);

          if (EVTy == MVT::Other)
            return true;

          if (TLI->isOperationLegalOrCustom(Opcode, EVTy))
            continue;
          else if (EVTy.isVector() &&
                   TLI->isOperationLegalOrCustom(Opcode, EVTy.getScalarType()))
            continue;

          return true;
        }
      }

      return true;
    } else if (isa<BinaryOperator>(J) &&
               J->getType()->getScalarType()->isPPC_FP128Ty()) {
      // Most operations on ppc_f128 values become calls.
      return true;
    } else if (isa<UIToFPInst>(J) || isa<SIToFPInst>(J) ||
               isa<FPToUIInst>(J) || isa<FPToSIInst>(J)) {
      CastInst *CI = cast<CastInst>(J);
      if (CI->getSrcTy()->getScalarType()->isPPC_FP128Ty() ||
          CI->getDestTy()->getScalarType()->isPPC_FP128Ty() ||
          isLargeIntegerTy(!TM.isPPC64(), CI->getSrcTy()->getScalarType()) ||
          isLargeIntegerTy(!TM.isPPC64(), CI->getDestTy()->getScalarType()))
        return true;
    } else if (isLargeIntegerTy(!TM.isPPC64(),
                                J->getType()->getScalarType()) &&
               (J->getOpcode() == Instruction::UDiv ||
                J->getOpcode() == Instruction::SDiv ||
                J->getOpcode() == Instruction::URem ||
                J->getOpcode() == Instruction::SRem)) {
      return true;
    } else if (!TM.isPPC64() &&
               isLargeIntegerTy(false, J->getType()->getScalarType()) &&
               (J->getOpcode() == Instruction::Shl ||
                J->getOpcode() == Instruction::AShr ||
                J->getOpcode() == Instruction::LShr)) {
      // Only on PPC32, for 128-bit integers (specifically not 64-bit
      // integers), these might be runtime calls.
      return true;
    } else if (isa<IndirectBrInst>(J) || isa<InvokeInst>(J)) {
      // On PowerPC, indirect jumps use the counter register.
      return true;
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(J)) {
      if (SI->getNumCases() + 1 >= (unsigned)TLI->getMinimumJumpTableEntries())
        return true;
    }

    // FREM is always a call.
    if (J->getOpcode() == Instruction::FRem)
      return true;

    if (ST->useSoftFloat()) {
      switch(J->getOpcode()) {
      case Instruction::FAdd:
      case Instruction::FSub:
      case Instruction::FMul:
      case Instruction::FDiv:
      case Instruction::FPTrunc:
      case Instruction::FPExt:
      case Instruction::FPToUI:
      case Instruction::FPToSI:
      case Instruction::UIToFP:
      case Instruction::SIToFP:
      case Instruction::FCmp:
        return true;
      }
    }

    for (Value *Operand : J->operands())
      if (memAddrUsesCTR(Operand))
        return true;
  }

  return false;
}

bool PPCTTIImpl::isHardwareLoopProfitable(Loop *L, ScalarEvolution &SE,
                                          AssumptionCache &AC,
                                          TargetLibraryInfo *LibInfo,
                                          HardwareLoopInfo &HWLoopInfo) {
  const PPCTargetMachine &TM = ST->getTargetMachine();
  TargetSchedModel SchedModel;
  SchedModel.init(ST);

  // Do not convert small short loops to CTR loop.
  unsigned ConstTripCount = SE.getSmallConstantTripCount(L);
  if (ConstTripCount && ConstTripCount < SmallCTRLoopThreshold) {
    SmallPtrSet<const Value *, 32> EphValues;
    CodeMetrics::collectEphemeralValues(L, &AC, EphValues);
    CodeMetrics Metrics;
    for (BasicBlock *BB : L->blocks())
      Metrics.analyzeBasicBlock(BB, *this, EphValues);
    // 6 is an approximate latency for the mtctr instruction.
    if (Metrics.NumInsts <= (6 * SchedModel.getIssueWidth()))
      return false;
  }

  // We don't want to spill/restore the counter register, and so we don't
  // want to use the counter register if the loop contains calls.
  SmallPtrSet<const Value *, 4> Visited;
  for (Loop::block_iterator I = L->block_begin(), IE = L->block_end();
       I != IE; ++I)
    if (mightUseCTR(*I, LibInfo, Visited))
      return false;

  SmallVector<BasicBlock*, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);

  // If there is an exit edge known to be frequently taken,
  // we should not transform this loop.
  for (auto &BB : ExitingBlocks) {
    Instruction *TI = BB->getTerminator();
    if (!TI) continue;

    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      uint64_t TrueWeight = 0, FalseWeight = 0;
      if (!BI->isConditional() ||
          !BI->extractProfMetadata(TrueWeight, FalseWeight))
        continue;

      // If the exit path is more frequent than the loop path,
      // we return here without further analysis for this loop.
      bool TrueIsExit = !L->contains(BI->getSuccessor(0));
      if (( TrueIsExit && FalseWeight < TrueWeight) ||
          (!TrueIsExit && FalseWeight > TrueWeight))
        return false;
    }
  }

  LLVMContext &C = L->getHeader()->getContext();
  HWLoopInfo.CountType = TM.isPPC64() ?
    Type::getInt64Ty(C) : Type::getInt32Ty(C);
  HWLoopInfo.LoopDecrement = ConstantInt::get(HWLoopInfo.CountType, 1);
  return true;
}

void PPCTTIImpl::getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                                         TTI::UnrollingPreferences &UP) {
  if (ST->getCPUDirective() == PPC::DIR_A2) {
    // The A2 is in-order with a deep pipeline, and concatenation unrolling
    // helps expose latency-hiding opportunities to the instruction scheduler.
    UP.Partial = UP.Runtime = true;

    // We unroll a lot on the A2 (hundreds of instructions), and the benefits
    // often outweigh the cost of a division to compute the trip count.
    UP.AllowExpensiveTripCount = true;
  }

  BaseT::getUnrollingPreferences(L, SE, UP);
}

// This function returns true to allow using coldcc calling convention.
// Returning true results in coldcc being used for functions which are cold at
// all call sites when the callers of the functions are not calling any other
// non coldcc functions.
bool PPCTTIImpl::useColdCCForColdCall(Function &F) {
  return EnablePPCColdCC;
}

bool PPCTTIImpl::enableAggressiveInterleaving(bool LoopHasReductions) {
  // On the A2, always unroll aggressively. For QPX unaligned loads, we depend
  // on combining the loads generated for consecutive accesses, and failure to
  // do so is particularly expensive. This makes it much more likely (compared
  // to only using concatenation unrolling).
  if (ST->getCPUDirective() == PPC::DIR_A2)
    return true;

  return LoopHasReductions;
}

PPCTTIImpl::TTI::MemCmpExpansionOptions
PPCTTIImpl::enableMemCmpExpansion(bool OptSize, bool IsZeroCmp) const {
  TTI::MemCmpExpansionOptions Options;
  Options.LoadSizes = {8, 4, 2, 1};
  Options.MaxNumLoads = TLI->getMaxExpandSizeMemcmp(OptSize);
  return Options;
}

bool PPCTTIImpl::enableInterleavedAccessVectorization() {
  return true;
}

unsigned PPCTTIImpl::getNumberOfRegisters(unsigned ClassID) const {
  assert(ClassID == GPRRC || ClassID == FPRRC ||
         ClassID == VRRC || ClassID == VSXRC);
  if (ST->hasVSX()) {
    assert(ClassID == GPRRC || ClassID == VSXRC || ClassID == VRRC);
    return ClassID == VSXRC ? 64 : 32;
  }
  assert(ClassID == GPRRC || ClassID == FPRRC || ClassID == VRRC);
  return 32;
}

unsigned PPCTTIImpl::getRegisterClassForType(bool Vector, Type *Ty) const {
  if (Vector)
    return ST->hasVSX() ? VSXRC : VRRC;
  else if (Ty && (Ty->getScalarType()->isFloatTy() ||
                  Ty->getScalarType()->isDoubleTy()))
    return ST->hasVSX() ? VSXRC : FPRRC;
  else if (Ty && (Ty->getScalarType()->isFP128Ty() ||
                  Ty->getScalarType()->isPPC_FP128Ty()))
    return VRRC;
  else if (Ty && Ty->getScalarType()->isHalfTy())
    return VSXRC;
  else
    return GPRRC;
}

const char* PPCTTIImpl::getRegisterClassName(unsigned ClassID) const {

  switch (ClassID) {
    default:
      llvm_unreachable("unknown register class");
      return "PPC::unknown register class";
    case GPRRC:       return "PPC::GPRRC";
    case FPRRC:       return "PPC::FPRRC";
    case VRRC:        return "PPC::VRRC";
    case VSXRC:       return "PPC::VSXRC";
  }
}

unsigned PPCTTIImpl::getRegisterBitWidth(bool Vector) const {
  if (Vector) {
    if (ST->hasQPX()) return 256;
    if (ST->hasAltivec()) return 128;
    return 0;
  }

  if (ST->isPPC64())
    return 64;
  return 32;

}

unsigned PPCTTIImpl::getCacheLineSize() const {
  // Check first if the user specified a custom line size.
  if (CacheLineSize.getNumOccurrences() > 0)
    return CacheLineSize;

  // Starting with P7 we have a cache line size of 128.
  unsigned Directive = ST->getCPUDirective();
  // Assume that Future CPU has the same cache line size as the others.
  if (Directive == PPC::DIR_PWR7 || Directive == PPC::DIR_PWR8 ||
      Directive == PPC::DIR_PWR9 || Directive == PPC::DIR_PWR10 ||
      Directive == PPC::DIR_PWR_FUTURE)
    return 128;

  // On other processors return a default of 64 bytes.
  return 64;
}

unsigned PPCTTIImpl::getPrefetchDistance() const {
  // This seems like a reasonable default for the BG/Q (this pass is enabled, by
  // default, only on the BG/Q).
  return 300;
}

unsigned PPCTTIImpl::getMaxInterleaveFactor(unsigned VF) {
  unsigned Directive = ST->getCPUDirective();
  // The 440 has no SIMD support, but floating-point instructions
  // have a 5-cycle latency, so unroll by 5x for latency hiding.
  if (Directive == PPC::DIR_440)
    return 5;

  // The A2 has no SIMD support, but floating-point instructions
  // have a 6-cycle latency, so unroll by 6x for latency hiding.
  if (Directive == PPC::DIR_A2)
    return 6;

  // FIXME: For lack of any better information, do no harm...
  if (Directive == PPC::DIR_E500mc || Directive == PPC::DIR_E5500)
    return 1;

  // For P7 and P8, floating-point instructions have a 6-cycle latency and
  // there are two execution units, so unroll by 12x for latency hiding.
  // FIXME: the same for P9 as previous gen until POWER9 scheduling is ready
  // FIXME: the same for P10 as previous gen until POWER10 scheduling is ready
  // Assume that future is the same as the others.
  if (Directive == PPC::DIR_PWR7 || Directive == PPC::DIR_PWR8 ||
      Directive == PPC::DIR_PWR9 || Directive == PPC::DIR_PWR10 ||
      Directive == PPC::DIR_PWR_FUTURE)
    return 12;

  // For most things, modern systems have two execution units (and
  // out-of-order execution).
  return 2;
}

// Adjust the cost of vector instructions on targets which there is overlap
// between the vector and scalar units, thereby reducing the overall throughput
// of vector code wrt. scalar code.
int PPCTTIImpl::vectorCostAdjustment(int Cost, unsigned Opcode, Type *Ty1,
                                     Type *Ty2) {
  if (!ST->vectorsUseTwoUnits() || !Ty1->isVectorTy())
    return Cost;

  std::pair<int, MVT> LT1 = TLI->getTypeLegalizationCost(DL, Ty1);
  // If type legalization involves splitting the vector, we don't want to
  // double the cost at every step - only the last step.
  if (LT1.first != 1 || !LT1.second.isVector())
    return Cost;

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  if (TLI->isOperationExpand(ISD, LT1.second))
    return Cost;

  if (Ty2) {
    std::pair<int, MVT> LT2 = TLI->getTypeLegalizationCost(DL, Ty2);
    if (LT2.first != 1 || !LT2.second.isVector())
      return Cost;
  }

  return Cost * 2;
}

int PPCTTIImpl::getArithmeticInstrCost(unsigned Opcode, Type *Ty,
                                       TTI::TargetCostKind CostKind,
                                       TTI::OperandValueKind Op1Info,
                                       TTI::OperandValueKind Op2Info,
                                       TTI::OperandValueProperties Opd1PropInfo,
                                       TTI::OperandValueProperties Opd2PropInfo,
                                       ArrayRef<const Value *> Args,
                                       const Instruction *CxtI) {
  assert(TLI->InstructionOpcodeToISD(Opcode) && "Invalid opcode");

  // Fallback to the default implementation.
  int Cost = BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info,
                                           Op2Info,
                                           Opd1PropInfo, Opd2PropInfo);
  return vectorCostAdjustment(Cost, Opcode, Ty, nullptr);
}

int PPCTTIImpl::getShuffleCost(TTI::ShuffleKind Kind, Type *Tp, int Index,
                               Type *SubTp) {
  // Legalize the type.
  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Tp);

  // PPC, for both Altivec/VSX and QPX, support cheap arbitrary permutations
  // (at least in the sense that there need only be one non-loop-invariant
  // instruction). We need one such shuffle instruction for each actual
  // register (this is not true for arbitrary shuffles, but is true for the
  // structured types of shuffles covered by TTI::ShuffleKind).
  return vectorCostAdjustment(LT.first, Instruction::ShuffleVector, Tp,
                              nullptr);
}

int PPCTTIImpl::getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src,
                                 TTI::TargetCostKind CostKind,
                                 const Instruction *I) {
  assert(TLI->InstructionOpcodeToISD(Opcode) && "Invalid opcode");

  int Cost = BaseT::getCastInstrCost(Opcode, Dst, Src, CostKind, I);
  Cost = vectorCostAdjustment(Cost, Opcode, Dst, Src);
  // TODO: Allow non-throughput costs that aren't binary.
  if (CostKind != TTI::TCK_RecipThroughput)
    return Cost == 0 ? 0 : 1;
  return Cost;
}

int PPCTTIImpl::getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                                   TTI::TargetCostKind CostKind,
                                   const Instruction *I) {
  int Cost = BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, CostKind, I);
  return vectorCostAdjustment(Cost, Opcode, ValTy, nullptr);
}

int PPCTTIImpl::getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index) {
  assert(Val->isVectorTy() && "This must be a vector type");

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  int Cost = BaseT::getVectorInstrCost(Opcode, Val, Index);
  Cost = vectorCostAdjustment(Cost, Opcode, Val, nullptr);

  if (ST->hasVSX() && Val->getScalarType()->isDoubleTy()) {
    // Double-precision scalars are already located in index #0 (or #1 if LE).
    if (ISD == ISD::EXTRACT_VECTOR_ELT &&
        Index == (ST->isLittleEndian() ? 1 : 0))
      return 0;

    return Cost;

  } else if (ST->hasQPX() && Val->getScalarType()->isFloatingPointTy()) {
    // Floating point scalars are already located in index #0.
    if (Index == 0)
      return 0;

    return Cost;

  } else if (Val->getScalarType()->isIntegerTy() && Index != -1U) {
    if (ST->hasP9Altivec()) {
      if (ISD == ISD::INSERT_VECTOR_ELT)
        // A move-to VSR and a permute/insert.  Assume vector operation cost
        // for both (cost will be 2x on P9).
        return vectorCostAdjustment(2, Opcode, Val, nullptr);

      // It's an extract.  Maybe we can do a cheap move-from VSR.
      unsigned EltSize = Val->getScalarSizeInBits();
      if (EltSize == 64) {
        unsigned MfvsrdIndex = ST->isLittleEndian() ? 1 : 0;
        if (Index == MfvsrdIndex)
          return 1;
      } else if (EltSize == 32) {
        unsigned MfvsrwzIndex = ST->isLittleEndian() ? 2 : 1;
        if (Index == MfvsrwzIndex)
          return 1;
      }

      // We need a vector extract (or mfvsrld).  Assume vector operation cost.
      // The cost of the load constant for a vector extract is disregarded
      // (invariant, easily schedulable).
      return vectorCostAdjustment(1, Opcode, Val, nullptr);
      
    } else if (ST->hasDirectMove())
      // Assume permute has standard cost.
      // Assume move-to/move-from VSR have 2x standard cost.
      return 3;
  }

  // Estimated cost of a load-hit-store delay.  This was obtained
  // experimentally as a minimum needed to prevent unprofitable
  // vectorization for the paq8p benchmark.  It may need to be
  // raised further if other unprofitable cases remain.
  unsigned LHSPenalty = 2;
  if (ISD == ISD::INSERT_VECTOR_ELT)
    LHSPenalty += 7;

  // Vector element insert/extract with Altivec is very expensive,
  // because they require store and reload with the attendant
  // processor stall for load-hit-store.  Until VSX is available,
  // these need to be estimated as very costly.
  if (ISD == ISD::EXTRACT_VECTOR_ELT ||
      ISD == ISD::INSERT_VECTOR_ELT)
    return LHSPenalty + Cost;

  return Cost;
}

int PPCTTIImpl::getMemoryOpCost(unsigned Opcode, Type *Src,
                                MaybeAlign Alignment, unsigned AddressSpace,
                                TTI::TargetCostKind CostKind,
                                const Instruction *I) {
  if (TLI->getValueType(DL, Src,  true) == MVT::Other)
    return BaseT::getMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                  CostKind);
  // Legalize the type.
  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Src);
  assert((Opcode == Instruction::Load || Opcode == Instruction::Store) &&
         "Invalid Opcode");

  int Cost = BaseT::getMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                    CostKind);
  // TODO: Handle other cost kinds.
  if (CostKind != TTI::TCK_RecipThroughput)
    return Cost;

  Cost = vectorCostAdjustment(Cost, Opcode, Src, nullptr);

  bool IsAltivecType = ST->hasAltivec() &&
                       (LT.second == MVT::v16i8 || LT.second == MVT::v8i16 ||
                        LT.second == MVT::v4i32 || LT.second == MVT::v4f32);
  bool IsVSXType = ST->hasVSX() &&
                   (LT.second == MVT::v2f64 || LT.second == MVT::v2i64);
  bool IsQPXType = ST->hasQPX() &&
                   (LT.second == MVT::v4f64 || LT.second == MVT::v4f32);

  // VSX has 32b/64b load instructions. Legalization can handle loading of
  // 32b/64b to VSR correctly and cheaply. But BaseT::getMemoryOpCost and
  // PPCTargetLowering can't compute the cost appropriately. So here we
  // explicitly check this case.
  unsigned MemBytes = Src->getPrimitiveSizeInBits();
  if (Opcode == Instruction::Load && ST->hasVSX() && IsAltivecType &&
      (MemBytes == 64 || (ST->hasP8Vector() && MemBytes == 32)))
    return 1;

  // Aligned loads and stores are easy.
  unsigned SrcBytes = LT.second.getStoreSize();
  if (!SrcBytes || !Alignment || *Alignment >= SrcBytes)
    return Cost;

  // If we can use the permutation-based load sequence, then this is also
  // relatively cheap (not counting loop-invariant instructions): one load plus
  // one permute (the last load in a series has extra cost, but we're
  // neglecting that here). Note that on the P7, we could do unaligned loads
  // for Altivec types using the VSX instructions, but that's more expensive
  // than using the permutation-based load sequence. On the P8, that's no
  // longer true.
  if (Opcode == Instruction::Load &&
      ((!ST->hasP8Vector() && IsAltivecType) || IsQPXType) &&
      *Alignment >= LT.second.getScalarType().getStoreSize())
    return Cost + LT.first; // Add the cost of the permutations.

  // For VSX, we can do unaligned loads and stores on Altivec/VSX types. On the
  // P7, unaligned vector loads are more expensive than the permutation-based
  // load sequence, so that might be used instead, but regardless, the net cost
  // is about the same (not counting loop-invariant instructions).
  if (IsVSXType || (ST->hasVSX() && IsAltivecType))
    return Cost;

  // Newer PPC supports unaligned memory access.
  if (TLI->allowsMisalignedMemoryAccesses(LT.second, 0))
    return Cost;

  // PPC in general does not support unaligned loads and stores. They'll need
  // to be decomposed based on the alignment factor.

  // Add the cost of each scalar load or store.
  assert(Alignment);
  Cost += LT.first * ((SrcBytes / Alignment->value()) - 1);

  // For a vector type, there is also scalarization overhead (only for
  // stores, loads are expanded using the vector-load + permutation sequence,
  // which is much less expensive).
  if (Src->isVectorTy() && Opcode == Instruction::Store)
    for (int i = 0, e = cast<FixedVectorType>(Src)->getNumElements(); i < e;
         ++i)
      Cost += getVectorInstrCost(Instruction::ExtractElement, Src, i);

  return Cost;
}

int PPCTTIImpl::getInterleavedMemoryOpCost(unsigned Opcode, Type *VecTy,
                                           unsigned Factor,
                                           ArrayRef<unsigned> Indices,
                                           unsigned Alignment,
                                           unsigned AddressSpace,
                                           TTI::TargetCostKind CostKind,
                                           bool UseMaskForCond,
                                           bool UseMaskForGaps) {
  if (UseMaskForCond || UseMaskForGaps)
    return BaseT::getInterleavedMemoryOpCost(Opcode, VecTy, Factor, Indices,
                                             Alignment, AddressSpace, CostKind,
                                             UseMaskForCond, UseMaskForGaps);

  assert(isa<VectorType>(VecTy) &&
         "Expect a vector type for interleaved memory op");

  // Legalize the type.
  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, VecTy);

  // Firstly, the cost of load/store operation.
  int Cost =
      getMemoryOpCost(Opcode, VecTy, MaybeAlign(Alignment), AddressSpace,
                      CostKind);

  // PPC, for both Altivec/VSX and QPX, support cheap arbitrary permutations
  // (at least in the sense that there need only be one non-loop-invariant
  // instruction). For each result vector, we need one shuffle per incoming
  // vector (except that the first shuffle can take two incoming vectors
  // because it does not need to take itself).
  Cost += Factor*(LT.first-1);

  return Cost;
}

unsigned PPCTTIImpl::getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                           TTI::TargetCostKind CostKind) {
  return BaseT::getIntrinsicInstrCost(ICA, CostKind);
}

bool PPCTTIImpl::canSaveCmp(Loop *L, BranchInst **BI, ScalarEvolution *SE,
                            LoopInfo *LI, DominatorTree *DT,
                            AssumptionCache *AC, TargetLibraryInfo *LibInfo) {
  // Process nested loops first.
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    if (canSaveCmp(*I, BI, SE, LI, DT, AC, LibInfo))
      return false; // Stop search.

  HardwareLoopInfo HWLoopInfo(L);

  if (!HWLoopInfo.canAnalyze(*LI))
    return false;

  if (!isHardwareLoopProfitable(L, *SE, *AC, LibInfo, HWLoopInfo))
    return false;

  if (!HWLoopInfo.isHardwareLoopCandidate(*SE, *LI, *DT))
    return false;

  *BI = HWLoopInfo.ExitBranch;
  return true;
}

bool PPCTTIImpl::isLSRCostLess(TargetTransformInfo::LSRCost &C1,
                               TargetTransformInfo::LSRCost &C2) {
  // PowerPC default behaviour here is "instruction number 1st priority".
  // If LsrNoInsnsCost is set, call default implementation.
  if (!LsrNoInsnsCost)
    return std::tie(C1.Insns, C1.NumRegs, C1.AddRecCost, C1.NumIVMuls,
                    C1.NumBaseAdds, C1.ScaleCost, C1.ImmCost, C1.SetupCost) <
           std::tie(C2.Insns, C2.NumRegs, C2.AddRecCost, C2.NumIVMuls,
                    C2.NumBaseAdds, C2.ScaleCost, C2.ImmCost, C2.SetupCost);
  else
    return TargetTransformInfoImplBase::isLSRCostLess(C1, C2);
}
