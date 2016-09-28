//===-- SystemZTargetTransformInfo.cpp - SystemZ-specific TTI -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a TargetTransformInfo analysis pass specific to the
// SystemZ target machine. It uses the target's detailed information to provide
// more precise answers to certain TTI queries, while letting the target
// independent and default TTI implementations handle the rest.
//
//===----------------------------------------------------------------------===//

#include "SystemZTargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/CostTable.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

#define DEBUG_TYPE "systemztti"

//===----------------------------------------------------------------------===//
//
// SystemZ cost model.
//
//===----------------------------------------------------------------------===//

int SystemZTTIImpl::getIntImmCost(const APInt &Imm, Type *Ty) {
  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  // There is no cost model for constants with a bit size of 0. Return TCC_Free
  // here, so that constant hoisting will ignore this constant.
  if (BitSize == 0)
    return TTI::TCC_Free;
  // No cost model for operations on integers larger than 64 bit implemented yet.
  if (BitSize > 64)
    return TTI::TCC_Free;

  if (Imm == 0)
    return TTI::TCC_Free;

  if (Imm.getBitWidth() <= 64) {
    // Constants loaded via lgfi.
    if (isInt<32>(Imm.getSExtValue()))
      return TTI::TCC_Basic;
    // Constants loaded via llilf.
    if (isUInt<32>(Imm.getZExtValue()))
      return TTI::TCC_Basic;
    // Constants loaded via llihf:
    if ((Imm.getZExtValue() & 0xffffffff) == 0)
      return TTI::TCC_Basic;

    return 2 * TTI::TCC_Basic;
  }

  return 4 * TTI::TCC_Basic;
}

int SystemZTTIImpl::getIntImmCost(unsigned Opcode, unsigned Idx,
                                  const APInt &Imm, Type *Ty) {
  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  // There is no cost model for constants with a bit size of 0. Return TCC_Free
  // here, so that constant hoisting will ignore this constant.
  if (BitSize == 0)
    return TTI::TCC_Free;
  // No cost model for operations on integers larger than 64 bit implemented yet.
  if (BitSize > 64)
    return TTI::TCC_Free;

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
  case Instruction::Store:
    if (Idx == 0 && Imm.getBitWidth() <= 64) {
      // Any 8-bit immediate store can by implemented via mvi.
      if (BitSize == 8)
        return TTI::TCC_Free;
      // 16-bit immediate values can be stored via mvhhi/mvhi/mvghi.
      if (isInt<16>(Imm.getSExtValue()))
        return TTI::TCC_Free;
    }
    break;
  case Instruction::ICmp:
    if (Idx == 1 && Imm.getBitWidth() <= 64) {
      // Comparisons against signed 32-bit immediates implemented via cgfi.
      if (isInt<32>(Imm.getSExtValue()))
        return TTI::TCC_Free;
      // Comparisons against unsigned 32-bit immediates implemented via clgfi.
      if (isUInt<32>(Imm.getZExtValue()))
        return TTI::TCC_Free;
    }
    break;
  case Instruction::Add:
  case Instruction::Sub:
    if (Idx == 1 && Imm.getBitWidth() <= 64) {
      // We use algfi/slgfi to add/subtract 32-bit unsigned immediates.
      if (isUInt<32>(Imm.getZExtValue()))
        return TTI::TCC_Free;
      // Or their negation, by swapping addition vs. subtraction.
      if (isUInt<32>(-Imm.getSExtValue()))
        return TTI::TCC_Free;
    }
    break;
  case Instruction::Mul:
    if (Idx == 1 && Imm.getBitWidth() <= 64) {
      // We use msgfi to multiply by 32-bit signed immediates.
      if (isInt<32>(Imm.getSExtValue()))
        return TTI::TCC_Free;
    }
    break;
  case Instruction::Or:
  case Instruction::Xor:
    if (Idx == 1 && Imm.getBitWidth() <= 64) {
      // Masks supported by oilf/xilf.
      if (isUInt<32>(Imm.getZExtValue()))
        return TTI::TCC_Free;
      // Masks supported by oihf/xihf.
      if ((Imm.getZExtValue() & 0xffffffff) == 0)
        return TTI::TCC_Free;
    }
    break;
  case Instruction::And:
    if (Idx == 1 && Imm.getBitWidth() <= 64) {
      // Any 32-bit AND operation can by implemented via nilf.
      if (BitSize <= 32)
        return TTI::TCC_Free;
      // 64-bit masks supported by nilf.
      if (isUInt<32>(~Imm.getZExtValue()))
        return TTI::TCC_Free;
      // 64-bit masks supported by nilh.
      if ((Imm.getZExtValue() & 0xffffffff) == 0xffffffff)
        return TTI::TCC_Free;
      // Some 64-bit AND operations can be implemented via risbg.
      const SystemZInstrInfo *TII = ST->getInstrInfo();
      unsigned Start, End;
      if (TII->isRxSBGMask(Imm.getZExtValue(), BitSize, Start, End))
        return TTI::TCC_Free;
    }
    break;
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    // Always return TCC_Free for the shift value of a shift instruction.
    if (Idx == 1)
      return TTI::TCC_Free;
    break;
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::IntToPtr:
  case Instruction::PtrToInt:
  case Instruction::BitCast:
  case Instruction::PHI:
  case Instruction::Call:
  case Instruction::Select:
  case Instruction::Ret:
  case Instruction::Load:
    break;
  }

  return SystemZTTIImpl::getIntImmCost(Imm, Ty);
}

int SystemZTTIImpl::getIntImmCost(Intrinsic::ID IID, unsigned Idx,
                                  const APInt &Imm, Type *Ty) {
  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  // There is no cost model for constants with a bit size of 0. Return TCC_Free
  // here, so that constant hoisting will ignore this constant.
  if (BitSize == 0)
    return TTI::TCC_Free;
  // No cost model for operations on integers larger than 64 bit implemented yet.
  if (BitSize > 64)
    return TTI::TCC_Free;

  switch (IID) {
  default:
    return TTI::TCC_Free;
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::usub_with_overflow:
    // These get expanded to include a normal addition/subtraction.
    if (Idx == 1 && Imm.getBitWidth() <= 64) {
      if (isUInt<32>(Imm.getZExtValue()))
        return TTI::TCC_Free;
      if (isUInt<32>(-Imm.getSExtValue()))
        return TTI::TCC_Free;
    }
    break;
  case Intrinsic::smul_with_overflow:
  case Intrinsic::umul_with_overflow:
    // These get expanded to include a normal multiplication.
    if (Idx == 1 && Imm.getBitWidth() <= 64) {
      if (isInt<32>(Imm.getSExtValue()))
        return TTI::TCC_Free;
    }
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
  return SystemZTTIImpl::getIntImmCost(Imm, Ty);
}

TargetTransformInfo::PopcntSupportKind
SystemZTTIImpl::getPopcntSupport(unsigned TyWidth) {
  assert(isPowerOf2_32(TyWidth) && "Type width must be power of 2");
  if (ST->hasPopulationCount() && TyWidth <= 64)
    return TTI::PSK_FastHardware;
  return TTI::PSK_Software;
}

void SystemZTTIImpl::getUnrollingPreferences(Loop *L,
                                             TTI::UnrollingPreferences &UP) {
  // Find out if L contains a call, what the machine instruction count
  // estimate is, and how many stores there are.
  bool HasCall = false;
  unsigned NumStores = 0;
  for (auto &BB : L->blocks())
    for (auto &I : *BB) {
      if (isa<CallInst>(&I) || isa<InvokeInst>(&I)) {
        ImmutableCallSite CS(&I);
        if (const Function *F = CS.getCalledFunction()) {
          if (isLoweredToCall(F))
            HasCall = true;
          if (F->getIntrinsicID() == Intrinsic::memcpy ||
              F->getIntrinsicID() == Intrinsic::memset)
            NumStores++;
        } else { // indirect call.
          HasCall = true;
        }
      }
      if (isa<StoreInst>(&I)) {
        NumStores++;
        Type *MemAccessTy = I.getOperand(0)->getType();
        if((MemAccessTy->isIntegerTy() || MemAccessTy->isFloatingPointTy()) &&
           (getDataLayout().getTypeSizeInBits(MemAccessTy) == 128))
          NumStores++;  // 128 bit fp/int stores get split.
      }
    }

  // The z13 processor will run out of store tags if too many stores
  // are fed into it too quickly. Therefore make sure there are not
  // too many stores in the resulting unrolled loop.
  unsigned const Max = (NumStores ? (12 / NumStores) : UINT_MAX);

  if (HasCall) {
    // Only allow full unrolling if loop has any calls.
    UP.FullUnrollMaxCount = Max;
    UP.MaxCount = 1;
    return;
  }

  UP.MaxCount = Max;
  if (UP.MaxCount <= 1)
    return;

  // Allow partial and runtime trip count unrolling.
  UP.Partial = UP.Runtime = true;

  UP.PartialThreshold = 75;
  UP.DefaultUnrollRuntimeCount = 4;

  // Allow expensive instructions in the pre-header of the loop.
  UP.AllowExpensiveTripCount = true;

  UP.Force = true;
}

unsigned SystemZTTIImpl::getNumberOfRegisters(bool Vector) {
  if (!Vector)
    // Discount the stack pointer.  Also leave out %r0, since it can't
    // be used in an address.
    return 14;
  if (ST->hasVector())
    return 32;
  return 0;
}

unsigned SystemZTTIImpl::getRegisterBitWidth(bool Vector) {
  if (!Vector)
    return 64;
  if (ST->hasVector())
    return 128;
  return 0;
}

