//===-- M88kISelLowering.h - M88k DAG lowering interface --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces for the M88kTargetLowering class.
// Only functions required by GlobalISel are implemented.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_M88K_M88KISELLOWERING_H
#define LLVM_LIB_TARGET_M88K_M88KISELLOWERING_H

#include "M88k.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class M88kSubtarget;
class M88kSubtarget;

namespace M88kISD {
enum NodeType : unsigned {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,

  // Return with a flag operand.  Operand 0 is the chain operand.
  RET_FLAG,

  // Calls a function.  Operand 0 is the chain operand and operand 1
  // is the target address.  The arguments start at operand 2.
  // There is an optional glue operand at the end.
  CALL,

  // Get the High 16 bits from a 32-bit immediate.
  Hi16,

  // Get the Lower 16 bits from a 32-bit immediate.
  Lo16,

  // Bit-field instructions.
  CLR,
  SET,
  EXT,
  EXTU,
  MAK,
  ROT,
  FF1,
  FF0,
};
} // end namespace M88kISD

class M88kTargetLowering : public TargetLowering {
  const M88kSubtarget &Subtarget;

public:
  explicit M88kTargetLowering(const TargetMachine &TM,
                              const M88kSubtarget &STI);

  bool isConstantUnsignedBitfieldExtractLegal(unsigned Opc, LLT Ty1,
                                              LLT Ty2) const override;
};

} // end namespace llvm

#endif
