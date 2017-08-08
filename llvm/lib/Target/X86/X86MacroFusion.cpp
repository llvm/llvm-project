//===- X86MacroFusion.cpp - X86 Macro Fusion ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the X86 implementation of the DAG scheduling
/// mutation to pair instructions back to back.
//
//===----------------------------------------------------------------------===//

#include "X86MacroFusion.h"
#include "X86Subtarget.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/CodeGen/MacroFusion.h"

using namespace llvm;

/// \brief Check if the instr pair, FirstMI and SecondMI, should be fused
/// together. Given SecondMI, when FirstMI is unspecified, then check if
/// SecondMI may be part of a fused pair at all.
static bool shouldScheduleAdjacent(const TargetInstrInfo &TII,
                                   const TargetSubtargetInfo &TSI,
                                   const MachineInstr *FirstMI,
                                   const MachineInstr &SecondMI) {
  const X86Subtarget &ST = static_cast<const X86Subtarget&>(TSI);
  // Check if this processor supports macro-fusion. Since this is a minor
  // heuristic, we haven't specifically reserved a feature. hasAVX is a decent
  // proxy for SandyBridge+.
  if (!ST.hasAVX())
    return false;

  enum {
    FuseTest,
    FuseCmp,
    FuseInc
  } FuseKind;

  unsigned FirstOpcode = FirstMI
                         ? FirstMI->getOpcode()
                         : static_cast<unsigned>(X86::INSTRUCTION_LIST_END);
  unsigned SecondOpcode = SecondMI.getOpcode();

  switch (SecondOpcode) {
  default:
    return false;
  case X86::JE_1:
  case X86::JNE_1:
  case X86::JL_1:
  case X86::JLE_1:
  case X86::JG_1:
  case X86::JGE_1:
    FuseKind = FuseInc;
    break;
  case X86::JB_1:
  case X86::JBE_1:
  case X86::JA_1:
  case X86::JAE_1:
    FuseKind = FuseCmp;
    break;
  case X86::JS_1:
  case X86::JNS_1:
  case X86::JP_1:
  case X86::JNP_1:
  case X86::JO_1:
  case X86::JNO_1:
    FuseKind = FuseTest;
    break;
  }

  switch (FirstOpcode) {
  default:
    return false;
  case X86::TEST8rr:
  case X86::TEST16rr:
  case X86::TEST32rr:
  case X86::TEST64rr:
  case X86::TEST8ri:
  case X86::TEST16ri:
  case X86::TEST32ri:
  case X86::TEST32i32:
  case X86::TEST64i32:
  case X86::TEST64ri32:
  case X86::TEST8rm:
  case X86::TEST16rm:
  case X86::TEST32rm:
  case X86::TEST64rm:
  case X86::TEST8ri_NOREX:
  case X86::AND16i16:
  case X86::AND16ri:
  case X86::AND16ri8:
  case X86::AND16rm:
  case X86::AND16rr:
  case X86::AND32i32:
  case X86::AND32ri:
  case X86::AND32ri8:
  case X86::AND32rm:
  case X86::AND32rr:
  case X86::AND64i32:
  case X86::AND64ri32:
  case X86::AND64ri8:
  case X86::AND64rm:
  case X86::AND64rr:
  case X86::AND8i8:
  case X86::AND8ri:
  case X86::AND8rm:
  case X86::AND8rr:
    return true;
  case X86::CMP16i16:
  case X86::CMP16ri:
  case X86::CMP16ri8:
  case X86::CMP16rm:
  case X86::CMP16rr:
  case X86::CMP32i32:
  case X86::CMP32ri:
  case X86::CMP32ri8:
  case X86::CMP32rm:
  case X86::CMP32rr:
  case X86::CMP64i32:
  case X86::CMP64ri32:
  case X86::CMP64ri8:
  case X86::CMP64rm:
  case X86::CMP64rr:
  case X86::CMP8i8:
  case X86::CMP8ri:
  case X86::CMP8rm:
  case X86::CMP8rr:
  case X86::ADD16i16:
  case X86::ADD16ri:
  case X86::ADD16ri8:
  case X86::ADD16ri8_DB:
  case X86::ADD16ri_DB:
  case X86::ADD16rm:
  case X86::ADD16rr:
  case X86::ADD16rr_DB:
  case X86::ADD32i32:
  case X86::ADD32ri:
  case X86::ADD32ri8:
  case X86::ADD32ri8_DB:
  case X86::ADD32ri_DB:
  case X86::ADD32rm:
  case X86::ADD32rr:
  case X86::ADD32rr_DB:
  case X86::ADD64i32:
  case X86::ADD64ri32:
  case X86::ADD64ri32_DB:
  case X86::ADD64ri8:
  case X86::ADD64ri8_DB:
  case X86::ADD64rm:
  case X86::ADD64rr:
  case X86::ADD64rr_DB:
  case X86::ADD8i8:
  case X86::ADD8mi:
  case X86::ADD8mr:
  case X86::ADD8ri:
  case X86::ADD8rm:
  case X86::ADD8rr:
  case X86::SUB16i16:
  case X86::SUB16ri:
  case X86::SUB16ri8:
  case X86::SUB16rm:
  case X86::SUB16rr:
  case X86::SUB32i32:
  case X86::SUB32ri:
  case X86::SUB32ri8:
  case X86::SUB32rm:
  case X86::SUB32rr:
  case X86::SUB64i32:
  case X86::SUB64ri32:
  case X86::SUB64ri8:
  case X86::SUB64rm:
  case X86::SUB64rr:
  case X86::SUB8i8:
  case X86::SUB8ri:
  case X86::SUB8rm:
  case X86::SUB8rr:
    return FuseKind == FuseCmp || FuseKind == FuseInc;
  case X86::INC16r:
  case X86::INC32r:
  case X86::INC64r:
  case X86::INC8r:
  case X86::DEC16r:
  case X86::DEC32r:
  case X86::DEC64r:
  case X86::DEC8r:
    return FuseKind == FuseInc;
  case X86::INSTRUCTION_LIST_END:
    return true;
  }
}

namespace llvm {

std::unique_ptr<ScheduleDAGMutation>
createX86MacroFusionDAGMutation () {
  return createBranchMacroFusionDAGMutation(shouldScheduleAdjacent);
}

} // end namespace llvm
