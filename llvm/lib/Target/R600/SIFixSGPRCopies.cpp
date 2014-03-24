//===-- SIFixSGPRCopies.cpp - Remove potential VGPR => SGPR copies --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Copies from VGPR to SGPR registers are illegal and the register coalescer
/// will sometimes generate these illegal copies in situations like this:
///
///  Register Class <vsrc> is the union of <vgpr> and <sgpr>
///
/// BB0:
///   %vreg0 <sgpr> = SCALAR_INST
///   %vreg1 <vsrc> = COPY %vreg0 <sgpr>
///    ...
///    BRANCH %cond BB1, BB2
///  BB1:
///    %vreg2 <vgpr> = VECTOR_INST
///    %vreg3 <vsrc> = COPY %vreg2 <vgpr>
///  BB2:
///    %vreg4 <vsrc> = PHI %vreg1 <vsrc>, <BB#0>, %vreg3 <vrsc>, <BB#1>
///    %vreg5 <vgpr> = VECTOR_INST %vreg4 <vsrc>
///
///
/// The coalescer will begin at BB0 and eliminate its copy, then the resulting
/// code will look like this:
///
/// BB0:
///   %vreg0 <sgpr> = SCALAR_INST
///    ...
///    BRANCH %cond BB1, BB2
/// BB1:
///   %vreg2 <vgpr> = VECTOR_INST
///   %vreg3 <vsrc> = COPY %vreg2 <vgpr>
/// BB2:
///   %vreg4 <sgpr> = PHI %vreg0 <sgpr>, <BB#0>, %vreg3 <vsrc>, <BB#1>
///   %vreg5 <vgpr> = VECTOR_INST %vreg4 <sgpr>
///
/// Now that the result of the PHI instruction is an SGPR, the register
/// allocator is now forced to constrain the register class of %vreg3 to
/// <sgpr> so we end up with final code like this:
///
/// BB0:
///   %vreg0 <sgpr> = SCALAR_INST
///    ...
///    BRANCH %cond BB1, BB2
/// BB1:
///   %vreg2 <vgpr> = VECTOR_INST
///   %vreg3 <sgpr> = COPY %vreg2 <vgpr>
/// BB2:
///   %vreg4 <sgpr> = PHI %vreg0 <sgpr>, <BB#0>, %vreg3 <sgpr>, <BB#1>
///   %vreg5 <vgpr> = VECTOR_INST %vreg4 <sgpr>
///
/// Now this code contains an illegal copy from a VGPR to an SGPR.
///
/// In order to avoid this problem, this pass searches for PHI instructions
/// which define a <vsrc> register and constrains its definition class to
/// <vgpr> if the user of the PHI's definition register is a vector instruction.
/// If the PHI's definition class is constrained to <vgpr> then the coalescer
/// will be unable to perform the COPY removal from the above example  which
/// ultimately led to the creation of an illegal COPY.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sgpr-copies"
#include "AMDGPU.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace {

class SIFixSGPRCopies : public MachineFunctionPass {

private:
  static char ID;
  const TargetRegisterClass *inferRegClassFromUses(const SIRegisterInfo *TRI,
                                           const MachineRegisterInfo &MRI,
                                           unsigned Reg,
                                           unsigned SubReg) const;
  const TargetRegisterClass *inferRegClassFromDef(const SIRegisterInfo *TRI,
                                                 const MachineRegisterInfo &MRI,
                                                 unsigned Reg,
                                                 unsigned SubReg) const;
  bool isVGPRToSGPRCopy(const MachineInstr &Copy, const SIRegisterInfo *TRI,
                        const MachineRegisterInfo &MRI) const;

public:
  SIFixSGPRCopies(TargetMachine &tm) : MachineFunctionPass(ID) { }

  virtual bool runOnMachineFunction(MachineFunction &MF);

  const char *getPassName() const {
    return "SI Fix SGPR copies";
  }

};

} // End anonymous namespace

char SIFixSGPRCopies::ID = 0;

FunctionPass *llvm::createSIFixSGPRCopiesPass(TargetMachine &tm) {
  return new SIFixSGPRCopies(tm);
}

static bool hasVGPROperands(const MachineInstr &MI, const SIRegisterInfo *TRI) {
  const MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    if (!MI.getOperand(i).isReg() ||
        !TargetRegisterInfo::isVirtualRegister(MI.getOperand(i).getReg()))
      continue;

    if (TRI->hasVGPRs(MRI.getRegClass(MI.getOperand(i).getReg())))
      return true;
  }
  return false;
}

/// This functions walks the use list of Reg until it finds an Instruction
/// that isn't a COPY returns the register class of that instruction.
/// \return The register defined by the first non-COPY instruction.
const TargetRegisterClass *SIFixSGPRCopies::inferRegClassFromUses(
                                                 const SIRegisterInfo *TRI,
                                                 const MachineRegisterInfo &MRI,
                                                 unsigned Reg,
                                                 unsigned SubReg) const {
  // The Reg parameter to the function must always be defined by either a PHI
  // or a COPY, therefore it cannot be a physical register.
  assert(TargetRegisterInfo::isVirtualRegister(Reg) &&
         "Reg cannot be a physical register");

  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  RC = TRI->getSubRegClass(RC, SubReg);
  for (MachineRegisterInfo::use_iterator I = MRI.use_begin(Reg),
                                         E = MRI.use_end(); I != E; ++I) {
    switch (I->getOpcode()) {
    case AMDGPU::COPY:
      RC = TRI->getCommonSubClass(RC, inferRegClassFromUses(TRI, MRI,
                                  I->getOperand(0).getReg(),
                                  I->getOperand(0).getSubReg()));
      break;
    }
  }

  return RC;
}

const TargetRegisterClass *SIFixSGPRCopies::inferRegClassFromDef(
                                                 const SIRegisterInfo *TRI,
                                                 const MachineRegisterInfo &MRI,
                                                 unsigned Reg,
                                                 unsigned SubReg) const {
  if (!TargetRegisterInfo::isVirtualRegister(Reg)) {
    const TargetRegisterClass *RC = TRI->getPhysRegClass(Reg);
    return TRI->getSubRegClass(RC, SubReg);
  }
  MachineInstr *Def = MRI.getVRegDef(Reg);
  if (Def->getOpcode() != AMDGPU::COPY) {
    return TRI->getSubRegClass(MRI.getRegClass(Reg), SubReg);
  }

  return inferRegClassFromDef(TRI, MRI, Def->getOperand(1).getReg(),
                                   Def->getOperand(1).getSubReg());
}

bool SIFixSGPRCopies::isVGPRToSGPRCopy(const MachineInstr &Copy,
                                      const SIRegisterInfo *TRI,
                                      const MachineRegisterInfo &MRI) const {

  unsigned DstReg = Copy.getOperand(0).getReg();
  unsigned SrcReg = Copy.getOperand(1).getReg();
  unsigned SrcSubReg = Copy.getOperand(1).getSubReg();
  const TargetRegisterClass *DstRC = MRI.getRegClass(DstReg);
  const TargetRegisterClass *SrcRC;

  if (!TargetRegisterInfo::isVirtualRegister(SrcReg) ||
      DstRC == &AMDGPU::M0RegRegClass)
    return false;

  SrcRC = TRI->getSubRegClass(MRI.getRegClass(SrcReg), SrcSubReg);
  return TRI->isSGPRClass(DstRC) && TRI->hasVGPRs(SrcRC);
}

bool SIFixSGPRCopies::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo *TRI = static_cast<const SIRegisterInfo *>(
      MF.getTarget().getRegisterInfo());
  const SIInstrInfo *TII = static_cast<const SIInstrInfo *>(
      MF.getTarget().getInstrInfo());
  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
                                                  BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
                                                      I != E; ++I) {
      MachineInstr &MI = *I;
      if (MI.getOpcode() == AMDGPU::COPY && isVGPRToSGPRCopy(MI, TRI, MRI)) {
        DEBUG(dbgs() << "Fixing VGPR -> SGPR copy:\n");
        DEBUG(MI.print(dbgs()));
        TII->moveToVALU(MI);

      }

      switch (MI.getOpcode()) {
      default: continue;
      case AMDGPU::PHI: {
        DEBUG(dbgs() << " Fixing PHI:\n");
        DEBUG(MI.print(dbgs()));

        for (unsigned i = 1; i < MI.getNumOperands(); i+=2) {
          unsigned Reg = MI.getOperand(i).getReg();
          const TargetRegisterClass *RC = inferRegClassFromDef(TRI, MRI, Reg,
                                                  MI.getOperand(0).getSubReg());
          MRI.constrainRegClass(Reg, RC);
        }
        unsigned Reg = MI.getOperand(0).getReg();
        const TargetRegisterClass *RC = inferRegClassFromUses(TRI, MRI, Reg,
                                                  MI.getOperand(0).getSubReg());
        if (TRI->getCommonSubClass(RC, &AMDGPU::VReg_32RegClass)) {
          MRI.constrainRegClass(Reg, &AMDGPU::VReg_32RegClass);
        }

        if (!TRI->isSGPRClass(MRI.getRegClass(Reg)))
          break;

        // If a PHI node defines an SGPR and any of its operands are VGPRs,
        // then we need to move it to the VALU.
        for (unsigned i = 1; i < MI.getNumOperands(); i+=2) {
          unsigned Reg = MI.getOperand(i).getReg();
          if (TRI->hasVGPRs(MRI.getRegClass(Reg))) {
            TII->moveToVALU(MI);
            break;
          }
        }

        break;
      }
      case AMDGPU::REG_SEQUENCE: {
        if (TRI->hasVGPRs(TII->getOpRegClass(MI, 0)) ||
            !hasVGPROperands(MI, TRI))
          continue;

        DEBUG(dbgs() << "Fixing REG_SEQUENCE:\n");
        DEBUG(MI.print(dbgs()));

        TII->moveToVALU(MI);
        break;
      }
      }
    }
  }
  return false;
}
