//===-- llvm/CodeGen/GlobalISel/MachineLegalizePass.cpp -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file implements the LegalizeHelper class to legalize individual
/// instructions and the MachineLegalizePass wrapper pass for the primary
/// legalization.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/MachineLegalizePass.h"
#include "llvm/CodeGen/GlobalISel/MachineLegalizeHelper.h"
#include "llvm/CodeGen/GlobalISel/MachineLegalizer.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

#define DEBUG_TYPE "legalize-mir"

using namespace llvm;

char MachineLegalizePass::ID = 0;
INITIALIZE_PASS_BEGIN(MachineLegalizePass, DEBUG_TYPE,
                      "Legalize the Machine IR a function's Machine IR", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(MachineLegalizePass, DEBUG_TYPE,
                    "Legalize the Machine IR a function's Machine IR", false,
                    false)

MachineLegalizePass::MachineLegalizePass() : MachineFunctionPass(ID) {
  initializeMachineLegalizePassPass(*PassRegistry::getPassRegistry());
}

void MachineLegalizePass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void MachineLegalizePass::init(MachineFunction &MF) {
}

bool MachineLegalizePass::combineExtracts(MachineInstr &MI,
                                          MachineRegisterInfo &MRI,
                                          const TargetInstrInfo &TII) {
  bool Changed = false;
  if (MI.getOpcode() != TargetOpcode::G_EXTRACT)
    return Changed;

  unsigned NumDefs = (MI.getNumOperands() - 1) / 2;
  unsigned SrcReg = MI.getOperand(NumDefs).getReg();
  MachineInstr &SeqI = *MRI.def_instr_begin(SrcReg);
  if (SeqI.getOpcode() != TargetOpcode::G_SEQUENCE)
      return Changed;

  unsigned NumSeqSrcs = (SeqI.getNumOperands() - 1) / 2;
  bool AllDefsReplaced = true;

  // Try to match each register extracted with a corresponding insertion formed
  // by the G_SEQUENCE.
  for (unsigned Idx = 0, SeqIdx = 0; Idx < NumDefs; ++Idx) {
    MachineOperand &ExtractMO = MI.getOperand(Idx);
    assert(ExtractMO.isReg() && ExtractMO.isDef() &&
           "unexpected extract operand");

    unsigned ExtractReg = ExtractMO.getReg();
    unsigned ExtractPos = MI.getOperand(NumDefs + Idx + 1).getImm();

    while (SeqIdx < NumSeqSrcs &&
           SeqI.getOperand(2 * SeqIdx + 2).getImm() < ExtractPos)
      ++SeqIdx;

    if (SeqIdx == NumSeqSrcs) {
      AllDefsReplaced = false;
      continue;
    }

    unsigned OrigReg = SeqI.getOperand(2 * SeqIdx + 1).getReg();
    if (SeqI.getOperand(2 * SeqIdx + 2).getImm() != ExtractPos ||
        MRI.getType(OrigReg) != MRI.getType(ExtractReg)) {
      AllDefsReplaced = false;
      continue;
    }

    assert(!TargetRegisterInfo::isPhysicalRegister(OrigReg) &&
           "unexpected physical register in G_SEQUENCE");

    // Finally we can replace the uses.
    for (auto &Use : MRI.use_operands(ExtractReg)) {
      Changed = true;
      Use.setReg(OrigReg);
    }
  }

  if (AllDefsReplaced) {
    // If SeqI was the next instruction in the BB and we removed it, we'd break
    // the outer iteration.
    assert(std::next(MachineBasicBlock::iterator(MI)) != SeqI &&
           "G_SEQUENCE does not dominate G_EXTRACT");

    MI.eraseFromParent();

    if (MRI.use_empty(SrcReg))
      SeqI.eraseFromParent();
    Changed = true;
  }

  return Changed;
}

bool MachineLegalizePass::runOnMachineFunction(MachineFunction &MF) {
  // If the ISel pipeline failed, do not bother running that pass.
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  DEBUG(dbgs() << "Legalize Machine IR for: " << MF.getName() << '\n');
  init(MF);
  const TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  const MachineLegalizer &Legalizer = *MF.getSubtarget().getMachineLegalizer();
  MachineLegalizeHelper Helper(MF);

  // FIXME: an instruction may need more than one pass before it is legal. For
  // example on most architectures <3 x i3> is doubly-illegal. It would
  // typically proceed along a path like: <3 x i3> -> <3 x i8> -> <8 x i8>. We
  // probably want a worklist of instructions rather than naive iterate until
  // convergence for performance reasons.
  bool Changed = false;
  MachineBasicBlock::iterator NextMI;
  for (auto &MBB : MF)
    for (auto MI = MBB.begin(); MI != MBB.end(); MI = NextMI) {
      // Get the next Instruction before we try to legalize, because there's a
      // good chance MI will be deleted.
      NextMI = std::next(MI);

      // Only legalize pre-isel generic instructions: others don't have types
      // and are assumed to be legal.
      if (!isPreISelGenericOpcode(MI->getOpcode()))
        continue;

      auto Res = Helper.legalizeInstr(*MI, Legalizer);

      // Error out if we couldn't legalize this instruction. We may want to fall
      // back to DAG ISel instead in the future.
      if (Res == MachineLegalizeHelper::UnableToLegalize) {
        if (!TPC.isGlobalISelAbortEnabled()) {
          MF.getProperties().set(
              MachineFunctionProperties::Property::FailedISel);
          return false;
        }
        std::string Msg;
        raw_string_ostream OS(Msg);
        OS << "unable to legalize instruction: ";
        MI->print(OS);
        report_fatal_error(OS.str());
      }

      Changed |= Res == MachineLegalizeHelper::Legalized;
    }


  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  for (auto &MBB : MF) {
    for (auto MI = MBB.begin(); MI != MBB.end(); MI = NextMI) {
      // Get the next Instruction before we try to legalize, because there's a
      // good chance MI will be deleted.
      NextMI = std::next(MI);

      Changed |= combineExtracts(*MI, MRI, TII);
    }
  }

  return Changed;
}
