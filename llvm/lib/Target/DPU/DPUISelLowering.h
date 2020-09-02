//===-- DPUISelLowering.h - DPU DAG Lowering Interface ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that DPU uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DPU_DPUISELLOWERING_H
#define LLVM_LIB_TARGET_DPU_DPUISELLOWERING_H

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
// Address spaces known by load/store functions.
namespace DPUADDR_SPACE {
enum { WRAM = 0, MRAM = 255 };
}

namespace DPUISD {
enum {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,
  RET_FLAG, // Return with a flag operand
  CALL,     // Call function
  SetCC,    // SET to a condition
  BrCC,     // Jump and branch with condition
  BrCCi,    // Jump and branch with condition
  BrCCZero, // Jump and branch with condition and one operand equal to zero
  OrJCCZero,
  AndJCCZero,
  XorJCCZero,
  AddJCCZero,
  SubJCCZero,
  Wrapper,    // Global addresses, externals...
  TRUNC64,    // Keep the LSBits register,
  LSL64_32,   // Shift 32 positions to the left
  LSL64_LT32, // Shift by less than 32 positions to the left
  LSL64_GT32, // Shift by more than 32 positions to the left
  LSR64_32,   // Shift 32 positions to the right
  LSR64_LT32, // Shift by less than 32 positions to the right
  LSR64_GT32, // Shift by more than 32 positions to the right
  ASR64_32,   // Shift 32 positions to the right
  ASR64_LT32, // Shift by less than 32 positions to the right
  ASR64_GT32, // Shift by more than 32 positions to the right
  ROL64_32,   // Rotate 32 positions to the left
  ROL64_LT32, // Rotate by less than 32 positions to the left
  ROL64_GT32, // Rotate by more than 32 positions to the left
  ROR64_32,   // Rotate 32 positions to the right
  ROR64_LT32, // Rotate by less than 32 positions to the right
  ROR64_GT32, // Rotate by more than 32 positions to the right
  MUL8_UU,
  MUL8_SU,
  MUL8_SS,
  MUL16_UU,
  MUL16_SU,
  MUL16_SS,

  Addc,
  Subc,
  Rsubc,

  Clo,
  Cls,
  Lslx,
  Lsl1,
  Lsl1x,
  Lsrx,
  Lsr1,
  Lsr1x,

  LslAdd,

  AddJcc,
  AddNullJcc,
  AddcJcc,
  AddcNullJcc,
  AndJcc,
  AndNullJcc,
  OrJcc,
  OrNullJcc,
  XorJcc,
  XorNullJcc,
  NandJcc,
  NandNullJcc,
  NorJcc,
  NorNullJcc,
  NxorJcc,
  NxorNullJcc,
  AndnJcc,
  AndnNullJcc,
  OrnJcc,
  OrnNullJcc,
  LslJcc,
  LslNullJcc,
  LslxJcc,
  LslxNullJcc,
  Lsl1Jcc,
  Lsl1NullJcc,
  Lsl1xJcc,
  Lsl1xNullJcc,
  LsrJcc,
  LsrNullJcc,
  LsrxJcc,
  LsrxNullJcc,
  Lsr1Jcc,
  Lsr1NullJcc,
  Lsr1xJcc,
  Lsr1xNullJcc,
  AsrJcc,
  AsrNullJcc,
  RolJcc,
  RolNullJcc,
  RorJcc,
  RorNullJcc,
  MUL8_UUJcc,
  MUL8_UUNullJcc,
  MUL8_SUJcc,
  MUL8_SUNullJcc,
  MUL8_SSJcc,
  MUL8_SSNullJcc,
  SubJcc,
  SubNullJcc,
  RsubJcc,
  RsubNullJcc,
  SubcJcc,
  SubcNullJcc,
  RsubcJcc,
  RsubcNullJcc,
  CaoJcc,
  CaoNullJcc,
  ClzJcc,
  ClzNullJcc,
  CloJcc,
  CloNullJcc,
  ClsJcc,
  ClsNullJcc,
  MoveJcc,
  MoveNullJcc,
  RolAddJcc,
  RolAddNullJcc,
  LsrAddJcc,
  LsrAddNullJcc,
  LslAddJcc,
  LslAddNullJcc,
  LslSubJcc,
  LslSubNullJcc,

  ADD_VASTART,

  LDMA,
  SDMA,

  SEQREAD_GET,
  SEQREAD_GET_CST,

  LHU_BIG,
  LHS_BIG,
  LW_BIG,
  LD_BIG,

  TEST_NODE
};
}

// Tries to convert a SDValue to a constant, if possible.
//
// Checks whether the provided SDValue represents a constant which can be used
// as an immediate for an operation, without checking the constant size (since
// the length of immediate operands depends on the opcodes). Returns NULL if the
// conversion is not applicable, the constant value otherwise.
static bool canFetchConstantTo(SDValue Value, uint64_t *pValue) {
  if (Value.getOpcode() == ISD::Constant) {
    ConstantSDNode *Constant = cast<ConstantSDNode>(Value);
    if ((!Constant->isOpaque()) && (Constant->getConstantIntValue() != NULL)) {
      *pValue = *(Constant->getConstantIntValue()->getValue().getRawData());
      return true;
    }
  }
  return false;
}

__attribute__((used)) static int32_t getSDValueAlignment(SDValue Op) {
  if (auto *GV = dyn_cast<GlobalAddressSDNode>(Op)) {
    return GV->getGlobal()->getAlignment();
  } else if (Op.getOpcode() == ISD::ADD) {
    SDValue Op0 = Op.getOperand(0);
    SDValue Op1 = Op.getOperand(1);
    uint64_t Size;
    if (auto *GV = dyn_cast<GlobalAddressSDNode>(Op0)) {
      if (canFetchConstantTo(Op1, &Size)) {
        return GV->getGlobal()->getAlignment() + Size;
      }
    } else if (auto *GV = dyn_cast<GlobalAddressSDNode>(Op1)) {
      if (canFetchConstantTo(Op0, &Size)) {
        return GV->getGlobal()->getAlignment() + Size;
      }
    }
  } else if (Op.getOpcode() == ISD::AND) {
    SDValue Op0 = Op.getOperand(0);
    SDValue Op1 = Op.getOperand(1);
    uint64_t Size;
    if (canFetchConstantTo(Op1, &Size)) {
      int32_t ConstantAlign = 1 << __builtin_ctz(Size);
      if (auto *GV = dyn_cast<GlobalAddressSDNode>(Op0)) {
        int32_t OpAlign = GV->getGlobal()->getAlignment();
        return OpAlign > ConstantAlign ? OpAlign : ConstantAlign;
      } else {
        return ConstantAlign;
      }
    } else if (canFetchConstantTo(Op0, &Size)) {
      int32_t ConstantAlign = 1 << __builtin_ctz(Size);
      if (auto *GV = dyn_cast<GlobalAddressSDNode>(Op1)) {
        int32_t OpAlign = GV->getGlobal()->getAlignment();
        return OpAlign > ConstantAlign ? OpAlign : ConstantAlign;
      } else {
        return ConstantAlign;
      }
    }
  }
  return -1;
}

__attribute__((used)) static bool properDMASize(uint64_t Size) {
  return (Size % 8 == 0) && (Size <= 2048);
}

} // namespace llvm

#endif
