//===-- ConnexHazardRecognizer.cpp - Connex Hazard Recognizer Impls -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements hazard recognizer for scheduling on Connex processor.
//
//===----------------------------------------------------------------------===//

// Inspired from llvm/lib/Target/PowerPC/PPCHazardRecognizer.cpp

#include "ConnexHazardRecognizer.h"
#include "Connex.h"
#include "ConnexInstrInfo.h"
#include "ConnexTargetMachine.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
//
#define INCLUDE_SUNIT_DUMP
#include "Misc.h" // For dumpSU()

using namespace llvm;

#define DEBUG_TYPE "post-RA-sched"

// getPredMachineInstr() is declared in ConnexInstrInfo.cpp
extern MachineInstr *getPredMachineInstr(MachineInstr *MI,
                                         MachineInstr **succMI);

/*
 From llvm.org/docs/doxygen/html/ScheduleHazardRecognizer_8h_source.html:
 /// PreEmitNoops - This callback is invoked prior to emitting an instruction.
 /// It should return the number of noops to emit prior to the provided
 /// instruction.
 /// Note: This is only used during PostRA scheduling. EmitNoop is not called
 /// for these noops.
 */
unsigned ConnexDispatchGroupSBHazardRecognizer::PreEmitNoops(SUnit *SU) {
  assert(SU->isInstr() == true);

  if (isDataHazard(SU))
    return 1;

  return ScoreboardHazardRecognizer::PreEmitNoops(SU);
}

bool ConnexDispatchGroupSBHazardRecognizer::isDataHazard(SUnit *SU) {
  // From http://llvm.org/docs/doxygen/html/classllvm_1_1MCInstrDesc.html
  const MCInstrDesc *MCID = DAG->getInstrDesc(SU);
  if (MCID == NULL)
    return false;

  unsigned numUses = MCID->getNumOperands() - MCID->getNumDefs();
  LLVM_DEBUG(dbgs() << "  isDataHazard(): numUses = " << numUses << "\n");
  LLVM_DEBUG(dbgs() << "  isDataHazard(): MCID->getNumOperands() = "
                    << MCID->getNumOperands() << "\n");
  LLVM_DEBUG(dbgs() << "  isDataHazard(): MCID->getNumDefs() = "
                    << MCID->getNumDefs() << "\n");

  assert(SU->isInstr() == true);

  MachineInstr *MI = SU->getInstr();
  LLVM_DEBUG(dbgs() << "  isDataHazard(): MI ="; MI->dump(););

  int MIOpcode = MI->getOpcode();
  LLVM_DEBUG(dbgs() << "  isDataHazard(): MI->getOpcode() = " << MI->getOpcode()
                    << "\n");

  if (MIOpcode == Connex::ST_INDIRECT_H || MIOpcode == Connex::ST_INDIRECT_W ||
      MIOpcode == Connex::ST_INDIRECT_MASKED_H || MIOpcode == Connex::ST_H) {
    /* NOTE: END_REPEAT returns, to my surprise, also mayStore().
       But we should not worry about this since END_REPEAT takes no
       parameter. */
    LLVM_DEBUG(dbgs() << "  isDataHazard(): SU is Store\n");
  } else if (MIOpcode == Connex::LD_INDIRECT_H ||
             MIOpcode == Connex::LD_INDIRECT_W ||
             MIOpcode == Connex::LD_INDIRECT_MASKED_H) {
    LLVM_DEBUG(dbgs() << "  isDataHazard(): SU is Load\n");
  } else if (MIOpcode == Connex::WHEREEQ_BUNDLE_H ||
             MIOpcode == Connex::WHERELT_BUNDLE_H ||
             MIOpcode == Connex::WHEREULT_BUNDLE_H) {
    LLVM_DEBUG(dbgs() << "  isDataHazard(): SU is Where\n");
  } else {
    LLVM_DEBUG(dbgs() << "  isDataHazard(): SU NOT producing data hazard\n");

    // Very important
    return false;
  }

  LLVM_DEBUG(dbgs() << "  isDataHazard(): MI->getNumOperands() = "
                    << MI->getNumOperands() << "\n");

  /*
   Why does getHazardType() find 3 Loads - because I was considering pred in
     DAG (SDNode), not in MachineInstr list, where it should be only 1?

   This should cover these cases described in ConnexISA.docx:
    - (i)write using register defined in the previous instruction:
      LS[R1] = R4
      LS[5] = R1
     and also this slightly different case:
      LS[R10] = R1

    - read using register defined in the previous instruction
      R4 = LS[R1]

    - wherexx using the flag defined in the previous instruction
      R1 = (R2 == R3)
      WHERE_EQUAL
  */

  // small-TODO: understand conceptually what PPC was doing with dispatch group.

  // IMPORTANT: We keep this search for predecessors of SU in the DAG and not
  // for THE only predecessor of the MachineInstr (we are at Post-RA scheduler)
  // contained in SU because MAYBE/it is possible that when doing
  //  ScoreboardHazardRecognizer (out-of-order scheduling to fill delay slots)
  //  we could benefit from the DAG predecessors - QUITE UNLIKELY, but maybe
  //  so. Otherwise, we should ONLY look at the
  //  getPredMachineInstr(MachineInstr *MI).
  //
  //  For any predecessors of SU with which we
  //   have an ordering dependency, return true.
  for (unsigned i = 0, ie = (unsigned)SU->Preds.size(); i != ie; ++i) {
    const MCInstrDesc *PredMCID = DAG->getInstrDesc(SU->Preds[i].getSUnit());

    if (PredMCID == NULL) // || !PredMCID->mayStore())
      continue;

    // SU->Preds is SmallVector of SDep.
    // - see http://llvm.org/docs/doxygen/html/classllvm_1_1SUnit.html
    // - see http://llvm.org/docs/doxygen/html/classllvm_1_1SDep.html
    MachineInstr *PredMI = (SU->Preds[i].getSUnit())->getInstr();
    MachineInstr *tmpNotUsed;

    if (PredMI != getPredMachineInstr(MI, &tmpNotUsed)) {
      LLVM_DEBUG(dbgs() << "  isDataHazard(): jumping DAG predecessor that is "
                           "NOT MachineInstr predecessor: PredMI =";
                 PredMI->dump(); dbgs() << "     for MI ="; MI->dump(););
      continue;
    }

    LLVM_DEBUG(dbgs() << "  isDataHazard(): Found DAG predecessor that is "
                         "MachineInstr predecessor: PredMI =";
               PredMI->dump(); dbgs() << "     for MI ="; MI->dump(););

    LLVM_DEBUG(dbgs() << "  isDataHazard(SU->Preds[" << i << "] = ";
               PredMI->dump();
               // (SU->Preds[i].getSUnit())->dump(DAG);
               // PredMCID->dump(DAG);
               dbgs() << ")\n");

    // TODO: check BETTER we have to check SU->Preds[i] is THE prev
    //              instruction in the list of MachineInstr - .getParent()
    // TODO: we have to check for LD_INDIRECT_H for the memory (offset)
    //   register, not the passthrough (or mask).

    unsigned numDefs = PredMCID->getNumDefs();
    LLVM_DEBUG(dbgs() << "  isDataHazard(): numDefs = " << numDefs << "\n");
    LLVM_DEBUG(dbgs() << "  isDataHazard(): PredMI->getNumOperands() = "
                      << PredMI->getNumOperands() << "\n");
    LLVM_DEBUG(dbgs() << "  isDataHazard(): PredMCID->getNumOperands() = "
                      << PredMCID->getNumOperands() << "\n");
    LLVM_DEBUG(dbgs() << "  isDataHazard(): PredMCID->getNumDefs() = "
                      << PredMCID->getNumDefs() << "\n");

    int idUseStart;
    if (MIOpcode == Connex::LD_INDIRECT_H ||
        MIOpcode == Connex::LD_INDIRECT_W ||
        MIOpcode == Connex::LD_INDIRECT_MASKED_H) {
      LLVM_DEBUG(dbgs() << "  isDataHazard(): PredMI->getOpcode() = "
                        << PredMI->getOpcode() << "\n");

      if (PredMI->isInlineAsm()) {
        LLVM_DEBUG(
            dbgs() << "  isDataHazard(): PredMI is INLINEASM so return true"
                   << "\n");
        // We assume that the PredMI INLINEAASM is NOT a Connex
        // instruction, but a host-side OPINCAA C++ for loop.
        //  In such case, we can have 2 data hazards with MI:
        //      - one with the instruction above this C++ for statement
        //      - one with the instruction at the end of this for loop
        //         when we unroll (if the trip-count of the loop is >1)
        //         this for loop
        //
        // Important TODO: make full checks and
        //                 return true only if it
        //                 is the case, to be more efficient.
        //
        // Important TODO: return true;
      }

      /* %Wh5<def>, %BoolMask1<def,dead> = LD_INDIRECT_MASKED_H %Wh4,
                                           %BoolMask0, %Wh0;
                                           mem:LD256[inttoptr (i16 51 to i16*)]
                                           (tbaa=!12)(alias.scope=!16)
          The arguments ("uses") of LD_INDIRECT_MASKED_H are:
              %Wh4 - I think it is the passthrough register
                 (if mask bit is 0 we use passthrough)
              %BoolMask0 - is the mask
              %Wh0 - the offset register (if mask bit is 0 we use passthrough)
          Note that Connex does NOT support masked gather just with read
              (it requires WHERE also and things become more complex than
                 just masked gather, in principle)
      */

      if (MIOpcode == Connex::LD_INDIRECT_MASKED_H) {
        idUseStart = MCID->getNumDefs() + 2; // 1 for passthrough, 1 bool mask
      } else if (MIOpcode == Connex::LD_INDIRECT_H ||
                 MIOpcode == Connex::LD_INDIRECT_W) {
        idUseStart = MCID->getNumDefs(); // 1 for passthrough, 1 for bool mask
      }
    } else {
      idUseStart = MCID->getNumDefs();
    }

    for (unsigned idUse = idUseStart; idUse < numUses; idUse++) {
      LLVM_DEBUG(dbgs() << "  isDataHazard(): MI->getOperand(" << idUse
                        << ") = " << MI->getOperand(idUse) << "\n");
      for (unsigned idDef = 0; idDef < numDefs; idDef++) {
        // See llvm.org/docs/doxygen/html/classllvm_1_1MachineOperand.html
        const MachineOperand &PredMIMO = PredMI->getOperand(idDef);
        const MachineOperand &MIMO = MI->getOperand(idUse);
        LLVM_DEBUG(dbgs() << "  isDataHazard(): PredMI->getOperand(" << idDef
                          << ") = " << PredMI->getOperand(idDef) << "\n");

        if ((PredMI->getOpcode() != Connex::END_WHERE) &&
            (PredMI->getOpcode() != Connex::WHEREEQ) &&
            (PredMI->getOpcode() != Connex::WHERELT) &&
            (PredMI->getOpcode() != Connex::WHERECRY) && PredMIMO.isReg() &&
            MIMO.isReg() && PredMIMO.getReg() == MIMO.getReg()) {
          LLVM_DEBUG(dbgs()
                     << "  isDataHazard(): found an instr sequence "
                        "(defReg = PredOpcode; write/read/Where useReg;) and "
                        "defReg == useReg. "
                        "This sequence has to be separated by NOP to avoid "
                        "true dependency hazard\n");
          return true;
        }
      }
    }
  }

  return false;
}

ScheduleHazardRecognizer::HazardType
ConnexDispatchGroupSBHazardRecognizer::getHazardType(SUnit *SU, int Stalls) {
  return ScoreboardHazardRecognizer::getHazardType(SU, Stalls);
}

void ConnexDispatchGroupSBHazardRecognizer::EmitInstruction(SUnit *SU) {
  unsigned i, ie;

  LLVM_DEBUG(
      dbgs() << "Entered Connex's "
                "ConnexDispatchGroupSBHazardRecognizer::EmitInstruction(";
      dumpSU(SU, dbgs()); dbgs() << ")\n");
  //
  assert(SU->isInstr() == true);
  MachineInstr *MI = SU->getInstr();
  MachineBasicBlock *MBB = MI->getParent();
  LLVM_DEBUG(dbgs() << "  EmitInstruction(): MBB = " << MBB->getFullName()
                    << "\n"
             // MBB->dump();
  );

  LLVM_DEBUG(dbgs() << "    SU->Succs.size() = " << SU->Succs.size() << "\n");
  LLVM_DEBUG(dbgs() << "    SU->Preds.size() = " << SU->Preds.size() << "\n");

  for (i = 0, ie = (unsigned)SU->Succs.size(); i != ie; ++i) {
    MachineInstr *SuccMI = (SU->Succs[i].getSUnit())->getInstr();
    if (SuccMI == NULL) {
      LLVM_DEBUG(dbgs() << "    SU->Succs[" << i << "] = NULL\n");
    } else {
      LLVM_DEBUG(dbgs() << "    SU->Succs[" << i << "] = "; SuccMI->dump();
                 dbgs() << "\n");
    }
  }
  for (i = 0, ie = (unsigned)SU->Preds.size(); i != ie; ++i) {
    MachineInstr *PredMI = (SU->Preds[i].getSUnit())->getInstr();
    if (PredMI == NULL) {
      LLVM_DEBUG(dbgs() << "    SU->Preds[" << i << "] = NULL\n");
    } else {
      LLVM_DEBUG(dbgs() << "    SU->Preds[" << i << "] = "; PredMI->dump();
                 dbgs() << "\n");
    }
  }

  return ScoreboardHazardRecognizer::EmitInstruction(SU);
}
