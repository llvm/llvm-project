//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PPCSelectionDAGInfo.h"
#include "PPCISelLowering.h"

using namespace llvm;

PPCSelectionDAGInfo::~PPCSelectionDAGInfo() = default;

bool PPCSelectionDAGInfo::isTargetMemoryOpcode(unsigned Opcode) const {
  return Opcode >= PPCISD::FIRST_MEMORY_OPCODE &&
         Opcode <= PPCISD::LAST_MEMORY_OPCODE;
}

bool PPCSelectionDAGInfo::isTargetStrictFPOpcode(unsigned Opcode) const {
  return Opcode >= PPCISD::FIRST_STRICTFP_OPCODE &&
         Opcode <= PPCISD::LAST_STRICTFP_OPCODE;
}

std::pair<SDValue, SDValue> PPCSelectionDAGInfo::EmitTargetCodeForMemcmp(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Op1, SDValue Op2,
    SDValue Op3, const CallInst *CI) const {
  return DAG.getMemcmp(Chain, dl, Op1, Op2, Op3, CI);
}
