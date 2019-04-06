//===- X86MacroFusion.cpp - X86 Macro Fusion ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the X86 implementation of the DAG scheduling
/// mutation to pair instructions back to back.
//
//===----------------------------------------------------------------------===//

#include "X86MacroFusion.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MacroFusion.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

using namespace llvm;

namespace {

// The classification for the first instruction.
enum class FirstInstrKind { Test, Cmp, And, ALU, IncDec, Invalid };

// The classification for the second instruction (jump).
enum class JumpKind {
  // JE, JL, JG and variants.
  ELG,
  // JA, JB and variants.
  AB,
  // JS, JP, JO and variants.
  SPO,
  // Not a fusable jump.
  Invalid,
};

} // namespace

static FirstInstrKind classifyFirst(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    return FirstInstrKind::Invalid;
  case X86::TEST8rr:
  case X86::TEST16rr:
  case X86::TEST32rr:
  case X86::TEST64rr:
  case X86::TEST8ri:
  case X86::TEST16ri:
  case X86::TEST32ri:
  case X86::TEST64ri32:
  case X86::TEST8mr:
  case X86::TEST16mr:
  case X86::TEST32mr:
  case X86::TEST64mr:
    return FirstInstrKind::Test;
  case X86::AND16ri:
  case X86::AND16ri8:
  case X86::AND16rm:
  case X86::AND16rr:
  case X86::AND32ri:
  case X86::AND32ri8:
  case X86::AND32rm:
  case X86::AND32rr:
  case X86::AND64ri32:
  case X86::AND64ri8:
  case X86::AND64rm:
  case X86::AND64rr:
  case X86::AND8ri:
  case X86::AND8rm:
  case X86::AND8rr:
    return FirstInstrKind::And;
  case X86::CMP16ri:
  case X86::CMP16ri8:
  case X86::CMP16rm:
  case X86::CMP16rr:
  case X86::CMP16mr:
  case X86::CMP32ri:
  case X86::CMP32ri8:
  case X86::CMP32rm:
  case X86::CMP32rr:
  case X86::CMP32mr:
  case X86::CMP64ri32:
  case X86::CMP64ri8:
  case X86::CMP64rm:
  case X86::CMP64rr:
  case X86::CMP64mr:
  case X86::CMP8ri:
  case X86::CMP8rm:
  case X86::CMP8rr:
  case X86::CMP8mr:
    return FirstInstrKind::Cmp;
  case X86::ADD16ri:
  case X86::ADD16ri8:
  case X86::ADD16ri8_DB:
  case X86::ADD16ri_DB:
  case X86::ADD16rm:
  case X86::ADD16rr:
  case X86::ADD16rr_DB:
  case X86::ADD32ri:
  case X86::ADD32ri8:
  case X86::ADD32ri8_DB:
  case X86::ADD32ri_DB:
  case X86::ADD32rm:
  case X86::ADD32rr:
  case X86::ADD32rr_DB:
  case X86::ADD64ri32:
  case X86::ADD64ri32_DB:
  case X86::ADD64ri8:
  case X86::ADD64ri8_DB:
  case X86::ADD64rm:
  case X86::ADD64rr:
  case X86::ADD64rr_DB:
  case X86::ADD8ri:
  case X86::ADD8ri_DB:
  case X86::ADD8rm:
  case X86::ADD8rr:
  case X86::ADD8rr_DB:
  case X86::SUB16ri:
  case X86::SUB16ri8:
  case X86::SUB16rm:
  case X86::SUB16rr:
  case X86::SUB32ri:
  case X86::SUB32ri8:
  case X86::SUB32rm:
  case X86::SUB32rr:
  case X86::SUB64ri32:
  case X86::SUB64ri8:
  case X86::SUB64rm:
  case X86::SUB64rr:
  case X86::SUB8ri:
  case X86::SUB8rm:
  case X86::SUB8rr:
    return FirstInstrKind::ALU;
  case X86::INC16r:
  case X86::INC32r:
  case X86::INC64r:
  case X86::INC8r:
  case X86::DEC16r:
  case X86::DEC32r:
  case X86::DEC64r:
  case X86::DEC8r:
    return FirstInstrKind::IncDec;
  }
}

static JumpKind classifySecond(const MachineInstr &MI) {
  X86::CondCode CC = X86::getCondFromBranch(MI);
  if (CC == X86::COND_INVALID)
    return JumpKind::Invalid;

  switch (CC) {
  default:
    return JumpKind::Invalid;
  case X86::COND_E:
  case X86::COND_NE:
  case X86::COND_L:
  case X86::COND_LE:
  case X86::COND_G:
  case X86::COND_GE:
    return JumpKind::ELG;
  case X86::COND_B:
  case X86::COND_BE:
  case X86::COND_A:
  case X86::COND_AE:
    return JumpKind::AB;
  case X86::COND_S:
  case X86::COND_NS:
  case X86::COND_P:
  case X86::COND_NP:
  case X86::COND_O:
  case X86::COND_NO:
    return JumpKind::SPO;
  }
}

/// Check if the instr pair, FirstMI and SecondMI, should be fused
/// together. Given SecondMI, when FirstMI is unspecified, then check if
/// SecondMI may be part of a fused pair at all.
static bool shouldScheduleAdjacent(const TargetInstrInfo &TII,
                                   const TargetSubtargetInfo &TSI,
                                   const MachineInstr *FirstMI,
                                   const MachineInstr &SecondMI) {
  const X86Subtarget &ST = static_cast<const X86Subtarget &>(TSI);

  // Check if this processor supports any kind of fusion.
  if (!(ST.hasBranchFusion() || ST.hasMacroFusion()))
    return false;

  const JumpKind BranchKind = classifySecond(SecondMI);

  if (BranchKind == JumpKind::Invalid)
    return false; // Second cannot be fused with anything.

  if (FirstMI == nullptr)
    return true; // We're only checking whether Second can be fused at all.

  const FirstInstrKind TestKind = classifyFirst(*FirstMI);

  if (ST.hasBranchFusion()) {
    // Branch fusion can merge CMP and TEST with all conditional jumps.
    return (TestKind == FirstInstrKind::Cmp ||
            TestKind == FirstInstrKind::Test);
  }

  if (ST.hasMacroFusion()) {
    // Macro Fusion rules are a bit more complex. See Agner Fog's
    // Microarchitecture table 9.2 "Instruction Fusion".
    switch (TestKind) {
    case FirstInstrKind::Test:
    case FirstInstrKind::And:
      return true;
    case FirstInstrKind::Cmp:
    case FirstInstrKind::ALU:
      return BranchKind == JumpKind::ELG || BranchKind == JumpKind::AB;
    case FirstInstrKind::IncDec:
      return BranchKind == JumpKind::ELG;
    case FirstInstrKind::Invalid:
      return false;
    }
  }

  llvm_unreachable("unknown branch fusion type");
}

namespace llvm {

std::unique_ptr<ScheduleDAGMutation>
createX86MacroFusionDAGMutation () {
  return createBranchMacroFusionDAGMutation(shouldScheduleAdjacent);
}

} // end namespace llvm
