//===-- llvm/CodeGen/GlobalISel/Legalizer.cpp -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file implements the LegalizerHelper class to legalize individual
/// instructions and the LegalizePass wrapper pass for the primary
/// legalization.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/LegalizerHelper.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

#include <iterator>

#define DEBUG_TYPE "legalizer"

using namespace llvm;

char Legalizer::ID = 0;
INITIALIZE_PASS_BEGIN(Legalizer, DEBUG_TYPE,
                      "Legalize the Machine IR a function's Machine IR", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(Legalizer, DEBUG_TYPE,
                    "Legalize the Machine IR a function's Machine IR", false,
                    false)

Legalizer::Legalizer() : MachineFunctionPass(ID) {
  initializeLegalizerPass(*PassRegistry::getPassRegistry());
}

void Legalizer::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void Legalizer::init(MachineFunction &MF) {
}

bool Legalizer::combineExtracts(MachineInstr &MI, MachineRegisterInfo &MRI,
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
    MRI.replaceRegWith(ExtractReg, OrigReg);
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

bool Legalizer::combineMerges(MachineInstr &MI, MachineRegisterInfo &MRI,
                              const TargetInstrInfo &TII) {
  if (MI.getOpcode() != TargetOpcode::G_UNMERGE_VALUES)
    return false;

  unsigned NumDefs = MI.getNumOperands() - 1;
  unsigned SrcReg = MI.getOperand(NumDefs).getReg();
  MachineInstr &MergeI = *MRI.def_instr_begin(SrcReg);
  if (MergeI.getOpcode() != TargetOpcode::G_MERGE_VALUES)
    return false;

  if (MergeI.getNumOperands() - 1 != NumDefs)
    return false;

  // FIXME: is a COPY appropriate if the types mismatch? We know both registers
  // are allocatable by now.
  if (MRI.getType(MI.getOperand(0).getReg()) !=
      MRI.getType(MergeI.getOperand(1).getReg()))
    return false;

  for (unsigned Idx = 0; Idx < NumDefs; ++Idx)
    MRI.replaceRegWith(MI.getOperand(Idx).getReg(),
                       MergeI.getOperand(Idx + 1).getReg());

  MI.eraseFromParent();
  if (MRI.use_empty(MergeI.getOperand(0).getReg()))
    MergeI.eraseFromParent();
  return true;
}

bool Legalizer::runOnMachineFunction(MachineFunction &MF) {
  // If the ISel pipeline failed, do not bother running that pass.
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  DEBUG(dbgs() << "Legalize Machine IR for: " << MF.getName() << '\n');
  init(MF);
  const TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  MachineOptimizationRemarkEmitter MORE(MF, /*MBFI=*/nullptr);
  LegalizerHelper Helper(MF);

  // FIXME: an instruction may need more than one pass before it is legal. For
  // example on most architectures <3 x i3> is doubly-illegal. It would
  // typically proceed along a path like: <3 x i3> -> <3 x i8> -> <8 x i8>. We
  // probably want a worklist of instructions rather than naive iterate until
  // convergence for performance reasons.
  bool Changed = false;
  MachineBasicBlock::iterator NextMI;
  for (auto &MBB : MF) {
    for (auto MI = MBB.begin(); MI != MBB.end(); MI = NextMI) {
      // Get the next Instruction before we try to legalize, because there's a
      // good chance MI will be deleted.
      NextMI = std::next(MI);

      // Only legalize pre-isel generic instructions: others don't have types
      // and are assumed to be legal.
      if (!isPreISelGenericOpcode(MI->getOpcode()))
        continue;
      unsigned NumNewInsns = 0;
      SmallVector<MachineInstr *, 4> WorkList;
      Helper.MIRBuilder.recordInsertions([&](MachineInstr *MI) {
        // Only legalize pre-isel generic instructions.
        // Legalization process could generate Target specific pseudo
        // instructions with generic types. Don't record them
        if (isPreISelGenericOpcode(MI->getOpcode())) {
          ++NumNewInsns;
          WorkList.push_back(MI);
        }
      });
      WorkList.push_back(&*MI);

      bool Changed = false;
      LegalizerHelper::LegalizeResult Res;
      unsigned Idx = 0;
      do {
        Res = Helper.legalizeInstrStep(*WorkList[Idx]);
        // Error out if we couldn't legalize this instruction. We may want to
        // fall back to DAG ISel instead in the future.
        if (Res == LegalizerHelper::UnableToLegalize) {
          Helper.MIRBuilder.stopRecordingInsertions();
          if (Res == LegalizerHelper::UnableToLegalize) {
            reportGISelFailure(MF, TPC, MORE, "gisel-legalize",
                               "unable to legalize instruction",
                               *WorkList[Idx]);
            return false;
          }
        }
        Changed |= Res == LegalizerHelper::Legalized;
        ++Idx;

#ifndef NDEBUG
        if (NumNewInsns) {
          DEBUG(dbgs() << ".. .. Emitted " << NumNewInsns << " insns\n");
          for (auto I = WorkList.end() - NumNewInsns, E = WorkList.end();
               I != E; ++I)
            DEBUG(dbgs() << ".. .. New MI: "; (*I)->print(dbgs()));
          NumNewInsns = 0;
        }
#endif
      } while (Idx < WorkList.size());

      Helper.MIRBuilder.stopRecordingInsertions();
    }
  }

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  for (auto &MBB : MF) {
    for (auto MI = MBB.begin(); MI != MBB.end(); MI = NextMI) {
      // Get the next Instruction before we try to legalize, because there's a
      // good chance MI will be deleted.
      NextMI = std::next(MI);

      // combineExtracts erases MI.
      if (combineExtracts(*MI, MRI, TII)) {
        Changed = true;
        continue;
      }
      Changed |= combineMerges(*MI, MRI, TII);
    }
  }

  return Changed;
}
