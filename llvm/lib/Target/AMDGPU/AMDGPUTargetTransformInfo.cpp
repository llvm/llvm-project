//===-- AMDGPUTargetTransformInfo.cpp - AMDGPU specific TTI pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/CostTable.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

#define DEBUG_TYPE "AMDGPUtti"


void AMDGPUTTIImpl::getUnrollingPreferences(Loop *L,
                                            TTI::UnrollingPreferences &UP) {
  UP.Threshold = 300; // Twice the default.
  UP.MaxCount = UINT_MAX;
  UP.Partial = true;

  // TODO: Do we want runtime unrolling?

  for (const BasicBlock *BB : L->getBlocks()) {
    const DataLayout &DL = BB->getModule()->getDataLayout();
    for (const Instruction &I : *BB) {
      const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&I);
      if (!GEP || GEP->getAddressSpace() != AMDGPUAS::PRIVATE_ADDRESS)
        continue;

      const Value *Ptr = GEP->getPointerOperand();
      const AllocaInst *Alloca =
          dyn_cast<AllocaInst>(GetUnderlyingObject(Ptr, DL));
      if (Alloca) {
        // We want to do whatever we can to limit the number of alloca
        // instructions that make it through to the code generator.  allocas
        // require us to use indirect addressing, which is slow and prone to
        // compiler bugs.  If this loop does an address calculation on an
        // alloca ptr, then we want to use a higher than normal loop unroll
        // threshold. This will give SROA a better chance to eliminate these
        // allocas.
        //
        // Don't use the maximum allowed value here as it will make some
        // programs way too big.
        UP.Threshold = 800;
      }
    }
  }
}

unsigned AMDGPUTTIImpl::getNumberOfRegisters(bool Vec) {
  if (Vec)
    return 0;

  // Number of VGPRs on SI.
  if (ST->getGeneration() >= AMDGPUSubtarget::SOUTHERN_ISLANDS)
    return 256;

  return 4 * 128; // XXX - 4 channels. Should these count as vector instead?
}

unsigned AMDGPUTTIImpl::getRegisterBitWidth(bool Vector) {
  return Vector ? 0 : 32;
}

unsigned AMDGPUTTIImpl::getMaxInterleaveFactor(unsigned VF) {
  // Semi-arbitrary large amount.
  return 64;
}

int AMDGPUTTIImpl::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, TTI::OperandValueKind Opd1Info,
    TTI::OperandValueKind Opd2Info, TTI::OperandValueProperties Opd1PropInfo,
    TTI::OperandValueProperties Opd2PropInfo) {

  EVT OrigTy = TLI->getValueType(DL, Ty);
  if (!OrigTy.isSimple()) {
    return BaseT::getArithmeticInstrCost(Opcode, Ty, Opd1Info, Opd2Info,
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
  case ISD::SRA: {
    if (SLT == MVT::i64)
      return get64BitInstrCost() * LT.first * NElts;

    // i32
    return getFullRateInstrCost() * LT.first * NElts;
  }
  case ISD::ADD:
  case ISD::SUB:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR: {
    if (SLT == MVT::i64){
      // and, or and xor are typically split into 2 VALU instructions.
      return 2 * getFullRateInstrCost() * LT.first * NElts;
    }

    return LT.first * NElts * getFullRateInstrCost();
  }
  case ISD::MUL: {
    const int QuarterRateCost = getQuarterRateInstrCost();
    if (SLT == MVT::i64) {
      const int FullRateCost = getFullRateInstrCost();
      return (4 * QuarterRateCost + (2 * 2) * FullRateCost) * LT.first * NElts;
    }

    // i32
    return QuarterRateCost * NElts * LT.first;
  }
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
    if (SLT == MVT::f64)
      return LT.first * NElts * get64BitInstrCost();

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
      if (ST->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS)
        Cost += 3 * getFullRateInstrCost();

      return LT.first * Cost * NElts;
    }

    // Assuming no fp32 denormals lowering.
    if (SLT == MVT::f32 || SLT == MVT::f16) {
      assert(!ST->hasFP32Denormals() && "will change when supported");
      int Cost = 7 * getFullRateInstrCost() + 1 * getQuarterRateInstrCost();
      return LT.first * NElts * Cost;
    }

    break;
  default:
    break;
  }

  return BaseT::getArithmeticInstrCost(Opcode, Ty, Opd1Info, Opd2Info,
                                       Opd1PropInfo, Opd2PropInfo);
}

unsigned AMDGPUTTIImpl::getCFInstrCost(unsigned Opcode) {
  // XXX - For some reason this isn't called for switch.
  switch (Opcode) {
  case Instruction::Br:
  case Instruction::Ret:
    return 10;
  default:
    return BaseT::getCFInstrCost(Opcode);
  }
}

int AMDGPUTTIImpl::getVectorInstrCost(unsigned Opcode, Type *ValTy,
                                      unsigned Index) {
  switch (Opcode) {
  case Instruction::ExtractElement:
  case Instruction::InsertElement:
    // Extracts are just reads of a subregister, so are free. Inserts are
    // considered free because we don't want to have any cost for scalarizing
    // operations, and we don't have to copy into a different register class.

    // Dynamic indexing isn't free and is best avoided.
    return Index == ~0u ? 2 : 0;
  default:
    return BaseT::getVectorInstrCost(Opcode, ValTy, Index);
  }
}

static bool isIntrinsicSourceOfDivergence(const TargetIntrinsicInfo *TII,
                                          const IntrinsicInst *I) {
  switch (I->getIntrinsicID()) {
  default:
    return false;
  case Intrinsic::not_intrinsic:
    // This means we have an intrinsic that isn't defined in
    // IntrinsicsAMDGPU.td
    break;

  case Intrinsic::amdgcn_workitem_id_x:
  case Intrinsic::amdgcn_workitem_id_y:
  case Intrinsic::amdgcn_workitem_id_z:
  case Intrinsic::amdgcn_interp_p1:
  case Intrinsic::amdgcn_interp_p2:
  case Intrinsic::amdgcn_mbcnt_hi:
  case Intrinsic::amdgcn_mbcnt_lo:
  case Intrinsic::r600_read_tidig_x:
  case Intrinsic::r600_read_tidig_y:
  case Intrinsic::r600_read_tidig_z:
  case Intrinsic::amdgcn_image_atomic_swap:
  case Intrinsic::amdgcn_image_atomic_add:
  case Intrinsic::amdgcn_image_atomic_sub:
  case Intrinsic::amdgcn_image_atomic_smin:
  case Intrinsic::amdgcn_image_atomic_umin:
  case Intrinsic::amdgcn_image_atomic_smax:
  case Intrinsic::amdgcn_image_atomic_umax:
  case Intrinsic::amdgcn_image_atomic_and:
  case Intrinsic::amdgcn_image_atomic_or:
  case Intrinsic::amdgcn_image_atomic_xor:
  case Intrinsic::amdgcn_image_atomic_inc:
  case Intrinsic::amdgcn_image_atomic_dec:
  case Intrinsic::amdgcn_image_atomic_cmpswap:
  case Intrinsic::amdgcn_buffer_atomic_swap:
  case Intrinsic::amdgcn_buffer_atomic_add:
  case Intrinsic::amdgcn_buffer_atomic_sub:
  case Intrinsic::amdgcn_buffer_atomic_smin:
  case Intrinsic::amdgcn_buffer_atomic_umin:
  case Intrinsic::amdgcn_buffer_atomic_smax:
  case Intrinsic::amdgcn_buffer_atomic_umax:
  case Intrinsic::amdgcn_buffer_atomic_and:
  case Intrinsic::amdgcn_buffer_atomic_or:
  case Intrinsic::amdgcn_buffer_atomic_xor:
  case Intrinsic::amdgcn_buffer_atomic_cmpswap:
  case Intrinsic::amdgcn_ps_live:
    return true;
  }

  StringRef Name = I->getCalledFunction()->getName();
  switch (TII->lookupName((const char *)Name.bytes_begin(), Name.size())) {
  default:
    return false;
  case AMDGPUIntrinsic::SI_fs_interp:
  case AMDGPUIntrinsic::SI_fs_constant:
    return true;
  }
}

static bool isArgPassedInSGPR(const Argument *A) {
  const Function *F = A->getParent();

  // Arguments to compute shaders are never a source of divergence.
  if (!AMDGPU::isShader(F->getCallingConv()))
    return true;

  // For non-compute shaders, SGPR inputs are marked with either inreg or byval.
  if (F->getAttributes().hasAttribute(A->getArgNo() + 1, Attribute::InReg) ||
      F->getAttributes().hasAttribute(A->getArgNo() + 1, Attribute::ByVal))
    return true;

  // Everything else is in VGPRs.
  return false;
}

///
/// \returns true if the result of the value could potentially be
/// different across workitems in a wavefront.
bool AMDGPUTTIImpl::isSourceOfDivergence(const Value *V) const {

  if (const Argument *A = dyn_cast<Argument>(V))
    return !isArgPassedInSGPR(A);

  // Loads from the private address space are divergent, because threads
  // can execute the load instruction with the same inputs and get different
  // results.
  //
  // All other loads are not divergent, because if threads issue loads with the
  // same arguments, they will always get the same result.
  if (const LoadInst *Load = dyn_cast<LoadInst>(V))
    return Load->getPointerAddressSpace() == AMDGPUAS::PRIVATE_ADDRESS;

  // Atomics are divergent because they are executed sequentially: when an
  // atomic operation refers to the same address in each thread, then each
  // thread after the first sees the value written by the previous thread as
  // original value.
  if (isa<AtomicRMWInst>(V) || isa<AtomicCmpXchgInst>(V))
    return true;

  if (const IntrinsicInst *Intrinsic = dyn_cast<IntrinsicInst>(V)) {
    const TargetMachine &TM = getTLI()->getTargetMachine();
    return isIntrinsicSourceOfDivergence(TM.getIntrinsicInfo(), Intrinsic);
  }

  // Assume all function calls are a source of divergence.
  if (isa<CallInst>(V) || isa<InvokeInst>(V))
    return true;

  return false;
}
