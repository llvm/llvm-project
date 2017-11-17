//=- AArch64VectorByElementOpt.cpp - AArch64 vector by element inst opt pass =//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that performs optimization for vector by element
// SIMD instructions.
//
// Certain SIMD instructions with vector element operand are not efficient.
// Rewrite them into SIMD instructions with vector operands. This rewrite
// is driven by the latency of the instructions.
//
// Example:
//    fmla v0.4s, v1.4s, v2.s[1]
//    is rewritten into
//    dup v3.4s, v2.s[1]
//    fmla v0.4s, v1.4s, v3.4s
//
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/Pass.h"
#include <map>

using namespace llvm;

#define DEBUG_TYPE "aarch64-vectorbyelement-opt"

STATISTIC(NumModifiedInstr,
          "Number of vector by element instructions modified");

#define AARCH64_VECTOR_BY_ELEMENT_OPT_NAME                                     \
  "AArch64 vector by element instruction optimization pass"

namespace {

struct AArch64VectorByElementOpt : public MachineFunctionPass {
  static char ID;

  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;
  TargetSchedModel SchedModel;

  AArch64VectorByElementOpt() : MachineFunctionPass(ID) {
    initializeAArch64VectorByElementOptPass(*PassRegistry::getPassRegistry());
  }

  /// Based only on latency of instructions, determine if it is cost efficient
  /// to replace the instruction InstDesc by the two instructions InstDescRep1
  /// and InstDescRep2.
  /// Return true if replacement is recommended.
  bool
  shouldReplaceInstruction(MachineFunction *MF, const MCInstrDesc *InstDesc,
                           const MCInstrDesc *InstDescRep1,
                           const MCInstrDesc *InstDescRep2,
                           std::map<unsigned, bool> &VecInstElemTable) const;

  /// Determine if we need to exit the vector by element instruction
  /// optimization pass early. This makes sure that Targets with no need
  /// for this optimization do not spent any compile time on this pass.
  /// This check is done by comparing the latency of an indexed FMLA
  /// instruction to the latency of the DUP + the latency of a vector
  /// FMLA instruction. We do not check on other related instructions such
  /// as FMLS as we assume that if the situation shows up for one
  /// instruction, then it is likely to show up for the related ones.
  /// Return true if early exit of the pass is recommended.
  bool earlyExitVectElement(MachineFunction *MF);

  /// Check whether an equivalent DUP instruction has already been
  /// created or not.
  /// Return true when the dup instruction already exists. In this case,
  /// DestReg will point to the destination of the already created DUP.
  bool reuseDUP(MachineInstr &MI, unsigned DupOpcode, unsigned SrcReg,
                unsigned LaneNumber, unsigned *DestReg) const;

  /// Certain SIMD instructions with vector element operand are not efficient.
  /// Rewrite them into SIMD instructions with vector operands. This rewrite
  /// is driven by the latency of the instructions.
  /// Return true if the SIMD instruction is modified.
  bool optimizeVectElement(MachineInstr &MI,
                           std::map<unsigned, bool> *VecInstElemTable) const;

  bool runOnMachineFunction(MachineFunction &Fn) override;

  StringRef getPassName() const override {
    return AARCH64_VECTOR_BY_ELEMENT_OPT_NAME;
  }
};

char AArch64VectorByElementOpt::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(AArch64VectorByElementOpt, "aarch64-vectorbyelement-opt",
                AARCH64_VECTOR_BY_ELEMENT_OPT_NAME, false, false)

/// Based only on latency of instructions, determine if it is cost efficient
/// to replace the instruction InstDesc by the two instructions InstDescRep1
/// and InstDescRep2. Note that it is assumed in this fuction that an
/// instruction of type InstDesc is always replaced by the same two
/// instructions as results are cached here.
/// Return true if replacement is recommended.
bool AArch64VectorByElementOpt::shouldReplaceInstruction(
    MachineFunction *MF, const MCInstrDesc *InstDesc,
    const MCInstrDesc *InstDescRep1, const MCInstrDesc *InstDescRep2,
    std::map<unsigned, bool> &VecInstElemTable) const {
  // Check if replacment decision is alredy available in the cached table.
  // if so, return it.
  if (!VecInstElemTable.empty() &&
      VecInstElemTable.find(InstDesc->getOpcode()) != VecInstElemTable.end())
    return VecInstElemTable[InstDesc->getOpcode()];

  unsigned SCIdx = InstDesc->getSchedClass();
  unsigned SCIdxRep1 = InstDescRep1->getSchedClass();
  unsigned SCIdxRep2 = InstDescRep2->getSchedClass();
  const MCSchedClassDesc *SCDesc =
      SchedModel.getMCSchedModel()->getSchedClassDesc(SCIdx);
  const MCSchedClassDesc *SCDescRep1 =
      SchedModel.getMCSchedModel()->getSchedClassDesc(SCIdxRep1);
  const MCSchedClassDesc *SCDescRep2 =
      SchedModel.getMCSchedModel()->getSchedClassDesc(SCIdxRep2);

  // If a subtarget does not define resources for any of the instructions
  // of interest, then return false for no replacement.
  if (!SCDesc->isValid() || SCDesc->isVariant() || !SCDescRep1->isValid() ||
      SCDescRep1->isVariant() || !SCDescRep2->isValid() ||
      SCDescRep2->isVariant()) {
    VecInstElemTable[InstDesc->getOpcode()] = false;
    return false;
  }

  if (SchedModel.computeInstrLatency(InstDesc->getOpcode()) >
      SchedModel.computeInstrLatency(InstDescRep1->getOpcode()) +
          SchedModel.computeInstrLatency(InstDescRep2->getOpcode())) {
    VecInstElemTable[InstDesc->getOpcode()] = true;
    return true;
  }
  VecInstElemTable[InstDesc->getOpcode()] = false;
  return false;
}

/// Determine if we need to exit the vector by element instruction
/// optimization pass early. This makes sure that Targets with no need
/// for this optimization do not spent any compile time on this pass.
/// This check is done by comparing the latency of an indexed FMLA
/// instruction to the latency of the DUP + the latency of a vector
/// FMLA instruction. We do not check on other related instructions such
/// as FMLS as we assume that if the situation shows up for one
/// instruction, then it is likely to show up for the related ones.
/// Return true if early exit of the pass is recommended.
bool AArch64VectorByElementOpt::earlyExitVectElement(MachineFunction *MF) {
  std::map<unsigned, bool> VecInstElemTable;
  const MCInstrDesc *IndexMulMCID = &TII->get(AArch64::FMLAv4i32_indexed);
  const MCInstrDesc *DupMCID = &TII->get(AArch64::DUPv4i32lane);
  const MCInstrDesc *MulMCID = &TII->get(AArch64::FMULv4f32);

  if (!shouldReplaceInstruction(MF, IndexMulMCID, DupMCID, MulMCID,
                                VecInstElemTable))
    return true;
  return false;
}

/// Check whether an equivalent DUP instruction has already been
/// created or not.
/// Return true when the dup instruction already exists. In this case,
/// DestReg will point to the destination of the already created DUP.
bool AArch64VectorByElementOpt::reuseDUP(MachineInstr &MI, unsigned DupOpcode,
                                         unsigned SrcReg, unsigned LaneNumber,
                                         unsigned *DestReg) const {
  for (MachineBasicBlock::iterator MII = MI, MIE = MI.getParent()->begin();
       MII != MIE;) {
    MII--;
    MachineInstr *CurrentMI = &*MII;

    if (CurrentMI->getOpcode() == DupOpcode &&
        CurrentMI->getNumOperands() == 3 &&
        CurrentMI->getOperand(1).getReg() == SrcReg &&
        CurrentMI->getOperand(2).getImm() == LaneNumber) {
      *DestReg = CurrentMI->getOperand(0).getReg();
      return true;
    }
  }

  return false;
}

/// Certain SIMD instructions with vector element operand are not efficient.
/// Rewrite them into SIMD instructions with vector operands. This rewrite
/// is driven by the latency of the instructions.
/// The instruction of concerns are for the time being fmla, fmls, fmul,
/// and fmulx and hence they are hardcoded.
///
/// Example:
///    fmla v0.4s, v1.4s, v2.s[1]
///    is rewritten into
///    dup v3.4s, v2.s[1]           // dup not necessary if redundant
///    fmla v0.4s, v1.4s, v3.4s
/// Return true if the SIMD instruction is modified.
bool AArch64VectorByElementOpt::optimizeVectElement(
    MachineInstr &MI, std::map<unsigned, bool> *VecInstElemTable) const {
  const MCInstrDesc *MulMCID, *DupMCID;
  const TargetRegisterClass *RC = &AArch64::FPR128RegClass;

  switch (MI.getOpcode()) {
  default:
    return false;

  // 4X32 instructions
  case AArch64::FMLAv4i32_indexed:
    DupMCID = &TII->get(AArch64::DUPv4i32lane);
    MulMCID = &TII->get(AArch64::FMLAv4f32);
    break;
  case AArch64::FMLSv4i32_indexed:
    DupMCID = &TII->get(AArch64::DUPv4i32lane);
    MulMCID = &TII->get(AArch64::FMLSv4f32);
    break;
  case AArch64::FMULXv4i32_indexed:
    DupMCID = &TII->get(AArch64::DUPv4i32lane);
    MulMCID = &TII->get(AArch64::FMULXv4f32);
    break;
  case AArch64::FMULv4i32_indexed:
    DupMCID = &TII->get(AArch64::DUPv4i32lane);
    MulMCID = &TII->get(AArch64::FMULv4f32);
    break;

  // 2X64 instructions
  case AArch64::FMLAv2i64_indexed:
    DupMCID = &TII->get(AArch64::DUPv2i64lane);
    MulMCID = &TII->get(AArch64::FMLAv2f64);
    break;
  case AArch64::FMLSv2i64_indexed:
    DupMCID = &TII->get(AArch64::DUPv2i64lane);
    MulMCID = &TII->get(AArch64::FMLSv2f64);
    break;
  case AArch64::FMULXv2i64_indexed:
    DupMCID = &TII->get(AArch64::DUPv2i64lane);
    MulMCID = &TII->get(AArch64::FMULXv2f64);
    break;
  case AArch64::FMULv2i64_indexed:
    DupMCID = &TII->get(AArch64::DUPv2i64lane);
    MulMCID = &TII->get(AArch64::FMULv2f64);
    break;

  // 2X32 instructions
  case AArch64::FMLAv2i32_indexed:
    RC = &AArch64::FPR64RegClass;
    DupMCID = &TII->get(AArch64::DUPv2i32lane);
    MulMCID = &TII->get(AArch64::FMLAv2f32);
    break;
  case AArch64::FMLSv2i32_indexed:
    RC = &AArch64::FPR64RegClass;
    DupMCID = &TII->get(AArch64::DUPv2i32lane);
    MulMCID = &TII->get(AArch64::FMLSv2f32);
    break;
  case AArch64::FMULXv2i32_indexed:
    RC = &AArch64::FPR64RegClass;
    DupMCID = &TII->get(AArch64::DUPv2i32lane);
    MulMCID = &TII->get(AArch64::FMULXv2f32);
    break;
  case AArch64::FMULv2i32_indexed:
    RC = &AArch64::FPR64RegClass;
    DupMCID = &TII->get(AArch64::DUPv2i32lane);
    MulMCID = &TII->get(AArch64::FMULv2f32);
    break;
  }

  if (!shouldReplaceInstruction(MI.getParent()->getParent(),
                                &TII->get(MI.getOpcode()), DupMCID, MulMCID,
                                *VecInstElemTable))
    return false;

  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock &MBB = *MI.getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  // get the operands of the current SIMD arithmetic instruction.
  unsigned MulDest = MI.getOperand(0).getReg();
  unsigned SrcReg0 = MI.getOperand(1).getReg();
  unsigned Src0IsKill = getKillRegState(MI.getOperand(1).isKill());
  unsigned SrcReg1 = MI.getOperand(2).getReg();
  unsigned Src1IsKill = getKillRegState(MI.getOperand(2).isKill());
  unsigned DupDest;

  // Instructions of interest have either 4 or 5 operands.
  if (MI.getNumOperands() == 5) {
    unsigned SrcReg2 = MI.getOperand(3).getReg();
    unsigned Src2IsKill = getKillRegState(MI.getOperand(3).isKill());
    unsigned LaneNumber = MI.getOperand(4).getImm();

    // Create a new DUP instruction. Note that if an equivalent DUP instruction
    // has already been created before, then use that one instread of creating
    // a new one.
    if (!reuseDUP(MI, DupMCID->getOpcode(), SrcReg2, LaneNumber, &DupDest)) {
      DupDest = MRI.createVirtualRegister(RC);
      BuildMI(MBB, MI, DL, *DupMCID, DupDest)
          .addReg(SrcReg2, Src2IsKill)
          .addImm(LaneNumber);
    }
    BuildMI(MBB, MI, DL, *MulMCID, MulDest)
        .addReg(SrcReg0, Src0IsKill)
        .addReg(SrcReg1, Src1IsKill)
        .addReg(DupDest, Src2IsKill);
  } else if (MI.getNumOperands() == 4) {
    unsigned LaneNumber = MI.getOperand(3).getImm();
    if (!reuseDUP(MI, DupMCID->getOpcode(), SrcReg1, LaneNumber, &DupDest)) {
      DupDest = MRI.createVirtualRegister(RC);
      BuildMI(MBB, MI, DL, *DupMCID, DupDest)
          .addReg(SrcReg1, Src1IsKill)
          .addImm(LaneNumber);
    }
    BuildMI(MBB, MI, DL, *MulMCID, MulDest)
        .addReg(SrcReg0, Src0IsKill)
        .addReg(DupDest, Src1IsKill);
  } else {
    return false;
  }

  ++NumModifiedInstr;
  return true;
}

bool AArch64VectorByElementOpt::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;

  TII = MF.getSubtarget().getInstrInfo();
  MRI = &MF.getRegInfo();
  const TargetSubtargetInfo &ST = MF.getSubtarget();
  const AArch64InstrInfo *AAII =
      static_cast<const AArch64InstrInfo *>(ST.getInstrInfo());
  if (!AAII)
    return false;
  SchedModel.init(ST.getSchedModel(), &ST, AAII);
  if (!SchedModel.hasInstrSchedModel())
    return false;

  // A simple check to exit this pass early for targets that do not need it.
  if (earlyExitVectElement(&MF))
    return false;

  bool Changed = false;
  std::map<unsigned, bool> VecInstElemTable;
  SmallVector<MachineInstr *, 8> RemoveMIs;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineBasicBlock::iterator MII = MBB.begin(), MIE = MBB.end();
         MII != MIE;) {
      MachineInstr &MI = *MII;
      if (optimizeVectElement(MI, &VecInstElemTable)) {
        // Add MI to the list of instructions to be removed given that it has
        // been replaced.
        RemoveMIs.push_back(&MI);
        Changed = true;
      }
      ++MII;
    }
  }

  for (MachineInstr *MI : RemoveMIs)
    MI->eraseFromParent();

  return Changed;
}

/// createAArch64VectorByElementOptPass - returns an instance of the
/// vector by element optimization pass.
FunctionPass *llvm::createAArch64VectorByElementOptPass() {
  return new AArch64VectorByElementOpt();
}
