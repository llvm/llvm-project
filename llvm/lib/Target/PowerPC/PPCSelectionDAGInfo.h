//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_PPCSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_POWERPC_PPCSELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "PPCGenSDNodeInfo.inc"

namespace llvm {
namespace PPCISD {

enum NodeType : unsigned {
  /// The result of the mflr at function entry, used for PIC code.
  GlobalBaseReg = GENERATED_OPCODE_END,

  /// The combination of sra[wd]i and addze used to implemented signed
  /// integer division by a power of 2. The first operand is the dividend,
  /// and the second is the constant shift amount (representing the
  /// divisor).
  SRA_ADDZE,

  /// R32 = MFOCRF(CRREG, INFLAG) - Represents the MFOCRF instruction.
  /// This copies the bits corresponding to the specified CRREG into the
  /// resultant GPR.  Bits corresponding to other CR regs are undefined.
  MFOCRF,

  // FIXME: Remove these once the ANDI glue bug is fixed:
  /// i1 = ANDI_rec_1_[EQ|GT]_BIT(i32 or i64 x) - Represents the result of the
  /// eq or gt bit of CR0 after executing andi. x, 1. This is used to
  /// implement truncation of i32 or i64 to i1.
  ANDI_rec_1_EQ_BIT,
  ANDI_rec_1_GT_BIT,

  // READ_TIME_BASE - A read of the 64-bit time-base register on a 32-bit
  // target (returns (Lo, Hi)). It takes a chain operand.
  READ_TIME_BASE,

  /// CHAIN = BDNZ CHAIN, DESTBB - These are used to create counter-based
  /// loops.
  BDNZ,
  BDZ,

  /// GPRC = address of _GLOBAL_OFFSET_TABLE_. Used by general dynamic and
  /// local dynamic TLS and position indendepent code on PPC32.
  PPC32_PICGOT,

  /// VRRC = VADD_SPLAT Elt, EltSize - Temporary node to be expanded
  /// during instruction selection to optimize a BUILD_VECTOR into
  /// operations on splats.  This is necessary to avoid losing these
  /// optimizations due to constant folding.
  VADD_SPLAT,
};

} // namespace PPCISD

class PPCSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  PPCSelectionDAGInfo();

  ~PPCSelectionDAGInfo() override;

  const char *getTargetNodeName(unsigned Opcode) const override;

  void verifyTargetNode(const SelectionDAG &DAG,
                        const SDNode *N) const override;

  std::pair<SDValue, SDValue>
  EmitTargetCodeForMemcmp(SelectionDAG &DAG, const SDLoc &dl, SDValue Chain,
                          SDValue Op1, SDValue Op2, SDValue Op3,
                          const CallInst *CI) const override;
  std::pair<SDValue, SDValue>
  EmitTargetCodeForStrlen(SelectionDAG &DAG, const SDLoc &DL, SDValue Chain,
                          SDValue Src, const CallInst *CI) const override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_POWERPC_PPCSELECTIONDAGINFO_H
