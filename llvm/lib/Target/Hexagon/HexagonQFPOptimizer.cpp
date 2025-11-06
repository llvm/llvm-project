//===----- HexagonQFPOptimizer.cpp - Qualcomm-FP to IEEE-FP conversions
// optimizer ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic infrastructure for optimizing intermediate conversion instructions
// generated while performing vector floating point operations.
// Currently run at the starting of the code generation for Hexagon, cleans
// up redundant conversion instructions and replaces the uses of conversion
// with appropriate machine operand. Liveness is preserved after this pass.
//
// @note: The redundant conversion instructions are not eliminated in this pass.
// In this pass, we are only trying to replace the uses of conversion
// instructions with its appropriate QFP instruction. We are leaving the job to
// Dead instruction Elimination pass to remove redundant conversion
// instructions.
//
// Brief overview of working of this QFP optimizer.
// This version of Hexagon QFP optimizer basically iterates over each
// instruction, checks whether if it belongs to hexagon floating point HVX
// arithmetic instruction category(Add, Sub, Mul). And then it finds the unique
// definition for the machine operands corresponding to the instruction.
//
// Example:
// MachineInstruction *MI be the HVX vadd instruction
// MI -> $v0 = V6_vadd_sf $v1, $v2
// MachineOperand *DefMI1 = MRI->getVRegDef(MI->getOperand(1).getReg());
// MachineOperand *DefMI2 = MRI->getVRegDef(MI->getOperand(2).getReg());
//
// In the above example, DefMI1 and DefMI2 gives the unique definitions
// corresponding to the operands($v1 and &v2 respectively) of instruction MI.
//
// If both of the definitions are not conversion instructions(V6_vconv_sf_qf32,
// V6_vconv_hf_qf16), then it will skip optimizing the current instruction and
// iterates over next instruction.
//
// If one the definitions is conversion instruction then our pass will replace
// the arithmetic instruction with its corresponding mix variant.
// In the above example, if $v1 is conversion instruction
// DefMI1 -> $v1 = V6_vconv_sf_qf32 $v3
// After Transformation:
// MI -> $v0 = V6_vadd_qf32_mix $v3, $v2 ($v1 is replaced with $v3)
//
// If both the definitions are conversion instructions then the instruction will
// be replaced with its qf variant
// In the above example, if $v1 and $v2 are conversion instructions
// DefMI1 -> $v1 = V6_vconv_sf_qf32 $v3
// DefMI2 -> $v2 = V6_vconv_sf_qf32 $v4
// After Transformation:
// MI -> $v0 = V6_vadd_qf32 $v3, $v4 ($v1 is replaced with $v3, $v2 is replaced
// with $v4)
//
// Currently, in this pass, we are not handling the case when the definitions
// are PHI inst.
//
//===----------------------------------------------------------------------===//
#include <unordered_set>
#define HEXAGON_QFP_OPTIMIZER "QFP optimizer pass"

#include "Hexagon.h"
#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <vector>

#define DEBUG_TYPE "hexagon-qfp-optimizer"

using namespace llvm;

cl::opt<bool>
    DisableQFOptimizer("disable-qfp-opt", cl::init(false),
                       cl::desc("Disable optimization of Qfloat operations."));

namespace {
const std::map<unsigned short, unsigned short> QFPInstMap{
    {Hexagon::V6_vadd_hf, Hexagon::V6_vadd_qf16_mix},
    {Hexagon::V6_vadd_qf16_mix, Hexagon::V6_vadd_qf16},
    {Hexagon::V6_vadd_sf, Hexagon::V6_vadd_qf32_mix},
    {Hexagon::V6_vadd_qf32_mix, Hexagon::V6_vadd_qf32},
    {Hexagon::V6_vsub_hf, Hexagon::V6_vsub_qf16_mix},
    {Hexagon::V6_vsub_qf16_mix, Hexagon::V6_vsub_qf16},
    {Hexagon::V6_vsub_sf, Hexagon::V6_vsub_qf32_mix},
    {Hexagon::V6_vsub_qf32_mix, Hexagon::V6_vsub_qf32},
    {Hexagon::V6_vmpy_qf16_hf, Hexagon::V6_vmpy_qf16_mix_hf},
    {Hexagon::V6_vmpy_qf16_mix_hf, Hexagon::V6_vmpy_qf16},
    {Hexagon::V6_vmpy_qf32_hf, Hexagon::V6_vmpy_qf32_mix_hf},
    {Hexagon::V6_vmpy_qf32_mix_hf, Hexagon::V6_vmpy_qf32_qf16},
    {Hexagon::V6_vmpy_qf32_sf, Hexagon::V6_vmpy_qf32}};
} // namespace

namespace {

struct HexagonQFPOptimizer : public MachineFunctionPass {
public:
  static char ID;

  HexagonQFPOptimizer() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  bool optimizeQfp(MachineInstr *MI, MachineBasicBlock *MBB);

  StringRef getPassName() const override { return HEXAGON_QFP_OPTIMIZER; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  const HexagonSubtarget *HST = nullptr;
  const HexagonInstrInfo *HII = nullptr;
  const MachineRegisterInfo *MRI = nullptr;
};

char HexagonQFPOptimizer::ID = 0;
} // namespace

INITIALIZE_PASS(HexagonQFPOptimizer, "hexagon-qfp-optimizer",
                HEXAGON_QFP_OPTIMIZER, false, false)

FunctionPass *llvm::createHexagonQFPOptimizer() {
  return new HexagonQFPOptimizer();
}

bool HexagonQFPOptimizer::optimizeQfp(MachineInstr *MI,
                                      MachineBasicBlock *MBB) {

  // Early exit:
  // - if instruction is invalid or has too few operands (QFP ops need 2 sources
  // + 1 dest),
  // - or does not have a transformation mapping.
  if (MI->getNumOperands() < 3)
    return false;
  auto It = QFPInstMap.find(MI->getOpcode());
  if (It == QFPInstMap.end())
    return false;
  unsigned short InstTy = It->second;

  unsigned Op0F = 0;
  unsigned Op1F = 0;
  // Get the reaching defs of MI, DefMI1 and DefMI2
  MachineInstr *DefMI1 = nullptr;
  MachineInstr *DefMI2 = nullptr;

  if (MI->getOperand(1).isReg())
    DefMI1 = MRI->getVRegDef(MI->getOperand(1).getReg());
  if (MI->getOperand(2).isReg())
    DefMI2 = MRI->getVRegDef(MI->getOperand(2).getReg());
  if (!DefMI1 || !DefMI2)
    return false;

  MachineOperand &Res = MI->getOperand(0);
  MachineInstr *Inst1 = nullptr;
  MachineInstr *Inst2 = nullptr;
  LLVM_DEBUG(dbgs() << "\n[Reaching Defs of operands]: "; DefMI1->dump();
             DefMI2->dump());

  // Get the reaching defs of DefMI
  if (DefMI1->getNumOperands() > 1 && DefMI1->getOperand(1).isReg() &&
      DefMI1->getOperand(1).getReg().isVirtual())
    Inst1 = MRI->getVRegDef(DefMI1->getOperand(1).getReg());

  if (DefMI2->getNumOperands() > 1 && DefMI2->getOperand(1).isReg() &&
      DefMI2->getOperand(1).getReg().isVirtual())
    Inst2 = MRI->getVRegDef(DefMI2->getOperand(1).getReg());

  unsigned Def1OP = DefMI1->getOpcode();
  unsigned Def2OP = DefMI2->getOpcode();

  MachineInstrBuilder MIB;
  // Case 1: Both reaching defs of MI are qf to sf/hf conversions
  if ((Def1OP == Hexagon::V6_vconv_sf_qf32 &&
       Def2OP == Hexagon::V6_vconv_sf_qf32) ||
      (Def1OP == Hexagon::V6_vconv_hf_qf16 &&
       Def2OP == Hexagon::V6_vconv_hf_qf16)) {

    // If the reaching defs of DefMI are W register type, we return
    if ((Inst1 && Inst1->getNumOperands() > 0 && Inst1->getOperand(0).isReg() &&
         MRI->getRegClass(Inst1->getOperand(0).getReg()) ==
             &Hexagon::HvxWRRegClass) ||
        (Inst2 && Inst2->getNumOperands() > 0 && Inst2->getOperand(0).isReg() &&
         MRI->getRegClass(Inst2->getOperand(0).getReg()) ==
             &Hexagon::HvxWRRegClass))
      return false;

    // Analyze the use operands of the conversion to get their KILL status
    MachineOperand &Src1 = DefMI1->getOperand(1);
    MachineOperand &Src2 = DefMI2->getOperand(1);

    Op0F = getKillRegState(Src1.isKill());
    Src1.setIsKill(false);

    Op1F = getKillRegState(Src2.isKill());
    Src2.setIsKill(false);

    if (MI->getOpcode() != Hexagon::V6_vmpy_qf32_sf) {
      auto OuterIt = QFPInstMap.find(MI->getOpcode());
      if (OuterIt == QFPInstMap.end())
        return false;
      auto InnerIt = QFPInstMap.find(OuterIt->second);
      if (InnerIt == QFPInstMap.end())
        return false;
      InstTy = InnerIt->second;
    }

    MIB = BuildMI(*MBB, MI, MI->getDebugLoc(), HII->get(InstTy), Res.getReg())
              .addReg(Src1.getReg(), Op0F, Src1.getSubReg())
              .addReg(Src2.getReg(), Op1F, Src2.getSubReg());
    LLVM_DEBUG(dbgs() << "\n[Inserting]: "; MIB.getInstr()->dump());
    return true;

    // Case 2: Left operand is conversion to sf/hf
  } else if (((Def1OP == Hexagon::V6_vconv_sf_qf32 &&
               Def2OP != Hexagon::V6_vconv_sf_qf32) ||
              (Def1OP == Hexagon::V6_vconv_hf_qf16 &&
               Def2OP != Hexagon::V6_vconv_hf_qf16)) &&
             !DefMI2->isPHI() &&
             (MI->getOpcode() != Hexagon::V6_vmpy_qf32_sf)) {

    if (Inst1 && MRI->getRegClass(Inst1->getOperand(0).getReg()) ==
                     &Hexagon::HvxWRRegClass)
      return false;

    MachineOperand &Src1 = DefMI1->getOperand(1);
    MachineOperand &Src2 = MI->getOperand(2);

    Op0F = getKillRegState(Src1.isKill());
    Src1.setIsKill(false);
    Op1F = getKillRegState(Src2.isKill());
    MIB = BuildMI(*MBB, MI, MI->getDebugLoc(), HII->get(InstTy), Res.getReg())
              .addReg(Src1.getReg(), Op0F, Src1.getSubReg())
              .addReg(Src2.getReg(), Op1F, Src2.getSubReg());
    LLVM_DEBUG(dbgs() << "\n[Inserting]: "; MIB.getInstr()->dump());
    return true;

    // Case 2: Left operand is conversion to sf/hf
  } else if (((Def1OP != Hexagon::V6_vconv_sf_qf32 &&
               Def2OP == Hexagon::V6_vconv_sf_qf32) ||
              (Def1OP != Hexagon::V6_vconv_hf_qf16 &&
               Def2OP == Hexagon::V6_vconv_hf_qf16)) &&
             !DefMI1->isPHI() &&
             (MI->getOpcode() != Hexagon::V6_vmpy_qf32_sf)) {
    // The second operand of original instruction is converted.
    // In "mix" instructions, "qf" operand is always the first operand.

    // Caveat: vsub is not commutative w.r.t operands.
    if (InstTy == Hexagon::V6_vsub_qf16_mix ||
        InstTy == Hexagon::V6_vsub_qf32_mix)
      return false;

    if (Inst2 && MRI->getRegClass(Inst2->getOperand(0).getReg()) ==
                     &Hexagon::HvxWRRegClass)
      return false;

    MachineOperand &Src1 = MI->getOperand(1);
    MachineOperand &Src2 = DefMI2->getOperand(1);

    Op1F = getKillRegState(Src2.isKill());
    Src2.setIsKill(false);
    Op0F = getKillRegState(Src1.isKill());
    MIB = BuildMI(*MBB, MI, MI->getDebugLoc(), HII->get(InstTy), Res.getReg())
              .addReg(Src2.getReg(), Op1F,
                      Src2.getSubReg()) // Notice the operands are flipped.
              .addReg(Src1.getReg(), Op0F, Src1.getSubReg());
    LLVM_DEBUG(dbgs() << "\n[Inserting]: "; MIB.getInstr()->dump());
    return true;
  }

  return false;
}

bool HexagonQFPOptimizer::runOnMachineFunction(MachineFunction &MF) {

  bool Changed = false;

  if (DisableQFOptimizer)
    return Changed;

  HST = &MF.getSubtarget<HexagonSubtarget>();
  if (!HST->useHVXV68Ops() || !HST->usePackets() ||
      skipFunction(MF.getFunction()))
    return false;
  HII = HST->getInstrInfo();
  MRI = &MF.getRegInfo();

  MachineFunction::iterator MBBI = MF.begin();
  LLVM_DEBUG(dbgs() << "\n=== Running QFPOptimzer Pass for : " << MF.getName()
                    << " Optimize intermediate conversions ===\n");
  while (MBBI != MF.end()) {
    MachineBasicBlock *MBB = &*MBBI;
    MachineBasicBlock::iterator MII = MBBI->instr_begin();
    while (MII != MBBI->instr_end()) {
      MachineInstr *MI = &*MII;
      ++MII; // As MI might be removed.

      if (QFPInstMap.count(MI->getOpcode()) &&
          MI->getOpcode() != Hexagon::V6_vconv_sf_qf32 &&
          MI->getOpcode() != Hexagon::V6_vconv_hf_qf16) {
        LLVM_DEBUG(dbgs() << "\n###Analyzing for removal: "; MI->dump());
        if (optimizeQfp(MI, MBB)) {
          MI->eraseFromParent();
          LLVM_DEBUG(dbgs() << "\t....Removing....");
          Changed = true;
        }
      }
    }
    ++MBBI;
  }
  return Changed;
}
