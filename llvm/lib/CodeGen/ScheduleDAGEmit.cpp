//===---- ScheduleDAGEmit.cpp - Emit routines for the ScheduleDAG class ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the Emit routines for the ScheduleDAG class, which creates
// MachineInstrs according to the computed schedule.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pre-RA-sched"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

void ScheduleDAG::AddMemOperand(MachineInstr *MI, const MachineMemOperand &MO) {
  MI->addMemOperand(MF, MO);
}

void ScheduleDAG::EmitNoop() {
  TII->insertNoop(*BB, End);
}

void ScheduleDAG::EmitPhysRegCopy(SUnit *SU,
                                  DenseMap<SUnit*, unsigned> &VRBaseMap) {
  for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
       I != E; ++I) {
    if (I->isCtrl()) continue;  // ignore chain preds
    if (I->getSUnit()->CopyDstRC) {
      // Copy to physical register.
      DenseMap<SUnit*, unsigned>::iterator VRI = VRBaseMap.find(I->getSUnit());
      assert(VRI != VRBaseMap.end() && "Node emitted out of order - late");
      // Find the destination physical register.
      unsigned Reg = 0;
      for (SUnit::const_succ_iterator II = SU->Succs.begin(),
             EE = SU->Succs.end(); II != EE; ++II) {
        if (II->getReg()) {
          Reg = II->getReg();
          break;
        }
      }
      TII->copyRegToReg(*BB, End, Reg, VRI->second,
                        SU->CopyDstRC, SU->CopySrcRC);
    } else {
      // Copy from physical register.
      assert(I->getReg() && "Unknown physical register!");
      unsigned VRBase = MRI.createVirtualRegister(SU->CopyDstRC);
      bool isNew = VRBaseMap.insert(std::make_pair(SU, VRBase)).second;
      isNew = isNew; // Silence compiler warning.
      assert(isNew && "Node emitted out of order - early");
      TII->copyRegToReg(*BB, End, VRBase, I->getReg(),
                        SU->CopyDstRC, SU->CopySrcRC);
    }
    break;
  }
}
