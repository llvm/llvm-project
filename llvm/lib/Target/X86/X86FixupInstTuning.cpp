//===-- X86FixupInstTunings.cpp - replace instructions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file does a tuning pass replacing slower machine instructions
// with faster ones. We do this here, as opposed to during normal ISel, as
// attempting to get the "right" instruction can break patterns. This pass
// is not meant search for special cases where an instruction can be transformed
// to another, it is only meant to do transformations where the old instruction
// is always replacable with the new instructions. For example:
//
//      `vpermq ymm` -> `vshufd ymm`
//          -- BAD, not always valid (lane cross/non-repeated mask)
//
//      `vpermilps ymm` -> `vshufd ymm`
//          -- GOOD, always replaceable
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "x86-fixup-inst-tuning"

STATISTIC(NumInstChanges, "Number of instructions changes");

namespace {
class X86FixupInstTuningPass : public MachineFunctionPass {
public:
  static char ID;

  X86FixupInstTuningPass() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "X86 Fixup Inst Tuning"; }

  bool runOnMachineFunction(MachineFunction &MF) override;
  bool processInstruction(MachineFunction &MF, MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator &I);

  // This pass runs after regalloc and doesn't support VReg operands.
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

private:
  const X86InstrInfo *TII = nullptr;
  const X86Subtarget *ST = nullptr;
  const MCSchedModel *SM = nullptr;
};
} // end anonymous namespace

char X86FixupInstTuningPass::ID = 0;

INITIALIZE_PASS(X86FixupInstTuningPass, DEBUG_TYPE, DEBUG_TYPE, false, false)

FunctionPass *llvm::createX86FixupInstTuning() {
  return new X86FixupInstTuningPass();
}

template <typename T>
static std::optional<bool> CmpOptionals(T NewVal, T CurVal) {
  if (NewVal.has_value() && CurVal.has_value() && *NewVal != *CurVal)
    return *NewVal < *CurVal;

  return std::nullopt;
}

bool X86FixupInstTuningPass::processInstruction(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator &I) {
  MachineInstr &MI = *I;
  unsigned Opc = MI.getOpcode();
  unsigned NumOperands = MI.getDesc().getNumOperands();

  auto GetInstTput = [&](unsigned Opcode) -> std::optional<double> {
    // We already checked that SchedModel exists in `NewOpcPreferable`.
    return MCSchedModel::getReciprocalThroughput(
        *ST, *(SM->getSchedClassDesc(TII->get(Opcode).getSchedClass())));
  };

  auto GetInstLat = [&](unsigned Opcode) -> std::optional<double> {
    // We already checked that SchedModel exists in `NewOpcPreferable`.
    return MCSchedModel::computeInstrLatency(
        *ST, *(SM->getSchedClassDesc(TII->get(Opcode).getSchedClass())));
  };

  auto GetInstSize = [&](unsigned Opcode) -> std::optional<unsigned> {
    if (unsigned Size = TII->get(Opcode).getSize())
      return Size;
    // Zero size means we where unable to compute it.
    return std::nullopt;
  };

  auto NewOpcPreferable = [&](unsigned NewOpc,
                              bool ReplaceInTie = true) -> bool {
    std::optional<bool> Res;
    if (SM->hasInstrSchedModel()) {
      // Compare tput -> lat -> code size.
      Res = CmpOptionals(GetInstTput(NewOpc), GetInstTput(Opc));
      if (Res.has_value())
        return *Res;

      Res = CmpOptionals(GetInstLat(NewOpc), GetInstLat(Opc));
      if (Res.has_value())
        return *Res;
    }

    Res = CmpOptionals(GetInstSize(Opc), GetInstSize(NewOpc));
    if (Res.has_value())
      return *Res;

    // We either have either were unable to get tput/lat/codesize or all values
    // were equal. Return specified option for a tie.
    return ReplaceInTie;
  };

  // `vpermilps r, i` -> `vshufps r, r, i`
  // `vpermilps r, i, k` -> `vshufps r, r, i, k`
  // `vshufps` is always as fast or faster than `vpermilps` and takes
  // 1 less byte of code size for VEX and SSE encoding.
  auto ProcessVPERMILPSri = [&](unsigned NewOpc) -> bool {
    if (!NewOpcPreferable(NewOpc))
      return false;
    unsigned MaskImm = MI.getOperand(NumOperands - 1).getImm();
    MI.removeOperand(NumOperands - 1);
    MI.addOperand(MI.getOperand(NumOperands - 2));
    MI.setDesc(TII->get(NewOpc));
    MI.addOperand(MachineOperand::CreateImm(MaskImm));
    return true;
  };

  // `vpermilps m, i` -> `vpshufd m, i` iff no domain delay penalty on shuffles.
  // `vpshufd` is always as fast or faster than `vpermilps` and takes 1 less
  // byte of code size.
  auto ProcessVPERMILPSmi = [&](unsigned NewOpc) -> bool {
    // TODO: Might be work adding bypass delay if -Os/-Oz is enabled as
    // `vpshufd` saves a byte of code size.
    if (!ST->hasNoDomainDelayShuffle() &&
        !NewOpcPreferable(NewOpc, /*ReplaceInTie*/ false))
      return false;
    MI.setDesc(TII->get(NewOpc));
    return true;
  };

  // `vunpcklpd/vmovlhps r, r` -> `vshufps r, r, 0x44`
  // `vunpckhpd/vmovlhps r, r` -> `vshufps r, r, 0xee`
  // `vunpcklpd r, r, k` -> `vshufpd r, r, 0x00`
  // `vunpckhpd r, r, k` -> `vshufpd r, r, 0xff`
  // iff `vshufps` is faster than `vunpck{l|h}pd`. Otherwise stick with
  // `vunpck{l|h}pd` as it uses less code size.
  // TODO: Look into using `{VP}UNPCK{L|H}QDQ{...}` instead of `{V}SHUF{...}PS`
  // as the replacement. `{VP}UNPCK{L|H}QDQ{...}` has no codesize cost.
  auto ProcessUNPCKPD = [&](unsigned NewOpc, unsigned MaskImm) -> bool {
    if (!NewOpcPreferable(NewOpc, /*ReplaceInTie*/ false))
      return false;

    MI.setDesc(TII->get(NewOpc));
    MI.addOperand(MachineOperand::CreateImm(MaskImm));
    return true;
  };

  auto ProcessUNPCKLPDrr = [&](unsigned NewOpc) -> bool {
    return ProcessUNPCKPD(NewOpc, 0x00);
  };
  auto ProcessUNPCKHPDrr = [&](unsigned NewOpc) -> bool {
    return ProcessUNPCKPD(NewOpc, 0xff);
  };

  switch (Opc) {
  case X86::VPERMILPSri:
    return ProcessVPERMILPSri(X86::VSHUFPSrri);
  case X86::VPERMILPSYri:
    return ProcessVPERMILPSri(X86::VSHUFPSYrri);
  case X86::VPERMILPSZ128ri:
    return ProcessVPERMILPSri(X86::VSHUFPSZ128rri);
  case X86::VPERMILPSZ256ri:
    return ProcessVPERMILPSri(X86::VSHUFPSZ256rri);
  case X86::VPERMILPSZri:
    return ProcessVPERMILPSri(X86::VSHUFPSZrri);
  case X86::VPERMILPSZ128rikz:
    return ProcessVPERMILPSri(X86::VSHUFPSZ128rrikz);
  case X86::VPERMILPSZ256rikz:
    return ProcessVPERMILPSri(X86::VSHUFPSZ256rrikz);
  case X86::VPERMILPSZrikz:
    return ProcessVPERMILPSri(X86::VSHUFPSZrrikz);
  case X86::VPERMILPSZ128rik:
    return ProcessVPERMILPSri(X86::VSHUFPSZ128rrik);
  case X86::VPERMILPSZ256rik:
    return ProcessVPERMILPSri(X86::VSHUFPSZ256rrik);
  case X86::VPERMILPSZrik:
    return ProcessVPERMILPSri(X86::VSHUFPSZrrik);
  case X86::VPERMILPSmi:
    return ProcessVPERMILPSmi(X86::VPSHUFDmi);
  case X86::VPERMILPSYmi:
    // TODO: See if there is a more generic way we can test if the replacement
    // instruction is supported.
    return ST->hasAVX2() ? ProcessVPERMILPSmi(X86::VPSHUFDYmi) : false;
  case X86::VPERMILPSZ128mi:
    return ProcessVPERMILPSmi(X86::VPSHUFDZ128mi);
  case X86::VPERMILPSZ256mi:
    return ProcessVPERMILPSmi(X86::VPSHUFDZ256mi);
  case X86::VPERMILPSZmi:
    return ProcessVPERMILPSmi(X86::VPSHUFDZmi);
  case X86::VPERMILPSZ128mikz:
    return ProcessVPERMILPSmi(X86::VPSHUFDZ128mikz);
  case X86::VPERMILPSZ256mikz:
    return ProcessVPERMILPSmi(X86::VPSHUFDZ256mikz);
  case X86::VPERMILPSZmikz:
    return ProcessVPERMILPSmi(X86::VPSHUFDZmikz);
  case X86::VPERMILPSZ128mik:
    return ProcessVPERMILPSmi(X86::VPSHUFDZ128mik);
  case X86::VPERMILPSZ256mik:
    return ProcessVPERMILPSmi(X86::VPSHUFDZ256mik);
  case X86::VPERMILPSZmik:
    return ProcessVPERMILPSmi(X86::VPSHUFDZmik);

    // TODO: {V}UNPCK{L|H}PD{...} is probably safe to transform to
    // `{VP}UNPCK{L|H}QDQ{...}` which gets the same perf benefit as
    // `{V}SHUF{...}PS` but 1) without increasing code size and 2) can also
    // handle the `mr` case. ICL doesn't have a domain penalty for replacing
    // float unpck -> int unpck, but at this time, I haven't verified the set of
    // processors where its safe.
  case X86::MOVLHPSrr:
  case X86::UNPCKLPDrr:
    return ProcessUNPCKLPDrr(X86::SHUFPSrri);
  case X86::VMOVLHPSrr:
  case X86::VUNPCKLPDrr:
    return ProcessUNPCKLPDrr(X86::VSHUFPSrri);
  case X86::VUNPCKLPDYrr:
    return ProcessUNPCKLPDrr(X86::VSHUFPSYrri);
    // VMOVLHPS is always 128 bits.
  case X86::VMOVLHPSZrr:
  case X86::VUNPCKLPDZ128rr:
    return ProcessUNPCKLPDrr(X86::VSHUFPDZ128rri);
  case X86::VUNPCKLPDZ256rr:
    return ProcessUNPCKLPDrr(X86::VSHUFPDZ256rri);
  case X86::VUNPCKLPDZrr:
    return ProcessUNPCKLPDrr(X86::VSHUFPDZrri);
  case X86::VUNPCKLPDZ128rrk:
    return ProcessUNPCKLPDrr(X86::VSHUFPDZ128rrik);
  case X86::VUNPCKLPDZ256rrk:
    return ProcessUNPCKLPDrr(X86::VSHUFPDZ256rrik);
  case X86::VUNPCKLPDZrrk:
    return ProcessUNPCKLPDrr(X86::VSHUFPDZrrik);
  case X86::VUNPCKLPDZ128rrkz:
    return ProcessUNPCKLPDrr(X86::VSHUFPDZ128rrikz);
  case X86::VUNPCKLPDZ256rrkz:
    return ProcessUNPCKLPDrr(X86::VSHUFPDZ256rrikz);
  case X86::VUNPCKLPDZrrkz:
    return ProcessUNPCKLPDrr(X86::VSHUFPDZrrikz);
  case X86::UNPCKHPDrr:
    return ProcessUNPCKHPDrr(X86::SHUFPDrri);
  case X86::VUNPCKHPDrr:
    return ProcessUNPCKHPDrr(X86::VSHUFPDrri);
  case X86::VUNPCKHPDYrr:
    return ProcessUNPCKHPDrr(X86::VSHUFPDYrri);
  case X86::VUNPCKHPDZ128rr:
    return ProcessUNPCKHPDrr(X86::VSHUFPDZ128rri);
  case X86::VUNPCKHPDZ256rr:
    return ProcessUNPCKHPDrr(X86::VSHUFPDZ256rri);
  case X86::VUNPCKHPDZrr:
    return ProcessUNPCKHPDrr(X86::VSHUFPDZrri);
  case X86::VUNPCKHPDZ128rrk:
    return ProcessUNPCKHPDrr(X86::VSHUFPDZ128rrik);
  case X86::VUNPCKHPDZ256rrk:
    return ProcessUNPCKHPDrr(X86::VSHUFPDZ256rrik);
  case X86::VUNPCKHPDZrrk:
    return ProcessUNPCKHPDrr(X86::VSHUFPDZrrik);
  case X86::VUNPCKHPDZ128rrkz:
    return ProcessUNPCKHPDrr(X86::VSHUFPDZ128rrikz);
  case X86::VUNPCKHPDZ256rrkz:
    return ProcessUNPCKHPDrr(X86::VSHUFPDZ256rrikz);
  case X86::VUNPCKHPDZrrkz:
    return ProcessUNPCKHPDrr(X86::VSHUFPDZrrikz);
  default:
    return false;
  }
}

bool X86FixupInstTuningPass::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "Start X86FixupInstTuning\n";);
  bool Changed = false;
  ST = &MF.getSubtarget<X86Subtarget>();
  TII = ST->getInstrInfo();
  SM = &ST->getSchedModel();

  for (MachineBasicBlock &MBB : MF) {
    for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I) {
      if (processInstruction(MF, MBB, I)) {
        ++NumInstChanges;
        Changed = true;
      }
    }
  }
  LLVM_DEBUG(dbgs() << "End X86FixupInstTuning\n";);
  return Changed;
}
