//===-- RISCVTargetTransformInfo.cpp - RISC-V specific TTI ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVTargetTransformInfo.h"
#include "MCTargetDesc/RISCVMatInt.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"
using namespace llvm;

#define DEBUG_TYPE "riscvtti"

int RISCVTTIImpl::getIntImmCost(const APInt &Imm, Type *Ty,
                                TTI::TargetCostKind CostKind) {
  assert(Ty->isIntegerTy() &&
         "getIntImmCost can only estimate cost of materialising integers");

  // We have a Zero register, so 0 is always free.
  if (Imm == 0)
    return TTI::TCC_Free;

  // Otherwise, we check how many instructions it will take to materialise.
  const DataLayout &DL = getDataLayout();
  return RISCVMatInt::getIntMatCost(Imm, DL.getTypeSizeInBits(Ty),
                                    getST()->is64Bit());
}

int RISCVTTIImpl::getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                    const APInt &Imm, Type *Ty,
                                    TTI::TargetCostKind CostKind,
                                    Instruction *Inst) {
  assert(Ty->isIntegerTy() &&
         "getIntImmCost can only estimate cost of materialising integers");

  // We have a Zero register, so 0 is always free.
  if (Imm == 0)
    return TTI::TCC_Free;

  // Some instructions in RISC-V can take a 12-bit immediate. Some of these are
  // commutative, in others the immediate comes from a specific argument index.
  bool Takes12BitImm = false;
  unsigned ImmArgIdx = ~0U;

  switch (Opcode) {
  case Instruction::GetElementPtr:
    // Never hoist any arguments to a GetElementPtr. CodeGenPrepare will
    // split up large offsets in GEP into better parts than ConstantHoisting
    // can.
    return TTI::TCC_Free;
  case Instruction::Add:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Mul:
    Takes12BitImm = true;
    break;
  case Instruction::Sub:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    Takes12BitImm = true;
    ImmArgIdx = 1;
    break;
  default:
    break;
  }

  if (Takes12BitImm) {
    // Check immediate is the correct argument...
    if (Instruction::isCommutative(Opcode) || Idx == ImmArgIdx) {
      // ... and fits into the 12-bit immediate.
      if (Imm.getMinSignedBits() <= 64 &&
          getTLI()->isLegalAddImmediate(Imm.getSExtValue())) {
        return TTI::TCC_Free;
      }
    }

    // Otherwise, use the full materialisation cost.
    return getIntImmCost(Imm, Ty, CostKind);
  }

  // By default, prevent hoisting.
  return TTI::TCC_Free;
}

int RISCVTTIImpl::getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx,
                                      const APInt &Imm, Type *Ty,
                                      TTI::TargetCostKind CostKind) {
  // Prevent hoisting in unknown cases.
  return TTI::TCC_Free;
}

bool RISCVTTIImpl::shouldExpandReduction(const IntrinsicInst *II) const {
  // Currently, the ExpandReductions pass can't expand scalable-vector
  // reductions, but we still request expansion as RVV doesn't support certain
  // reductions and the SelectionDAG can't legalize them either.
  switch (II->getIntrinsicID()) {
  default:
    return false;
  // These reductions have no equivalent in RVV
  case Intrinsic::vector_reduce_mul:
  case Intrinsic::vector_reduce_fmul:
  // The fmin and fmax intrinsics are not currently supported due to a
  // discrepancy between the LLVM semantics and the RVV 0.10 ISA behaviour with
  // regards to signaling NaNs: the vector fmin/fmax reduction intrinsics match
  // the behaviour minnum/maxnum intrinsics, whereas the vfredmin/vfredmax
  // instructions match the vfmin/vfmax instructions which match the equivalent
  // scalar fmin/fmax instructions as defined in 2.2 F/D/Q extension (see
  // https://bugs.llvm.org/show_bug.cgi?id=27363).
  // This behaviour is likely fixed in version 2.3 of the RISC-V F/D/Q
  // extension, where fmin/fmax behave like minnum/maxnum, but until then the
  // intrinsics are left unsupported.
  case Intrinsic::vector_reduce_fmax:
  case Intrinsic::vector_reduce_fmin:
    return true;
  }
}

Optional<unsigned> RISCVTTIImpl::getMaxVScale() const {
  // There is no assumption of the maximum vector length in V specification.
  // We use the value specified by users as the maximum vector length.
  // This function will use the assumed maximum vector length to get the
  // maximum vscale for LoopVectorizer.
  // If users do not specify the maximum vector length, we have no way to
  // know whether the LoopVectorizer is safe to do or not.
  // We only consider to use single vector register (LMUL = 1) to vectorize.
  unsigned MaxVectorSizeInBits = ST->getMaxRVVVectorSizeInBits();
  if (ST->hasStdExtV() && MaxVectorSizeInBits != 0)
    return MaxVectorSizeInBits / RISCV::RVVBitsPerBlock;
  return BaseT::getMaxVScale();
}
