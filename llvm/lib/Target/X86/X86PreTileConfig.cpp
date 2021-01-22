//===-- X86PreTileConfig.cpp - Tile Register Configure---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Pass to pre-config the shape of AMX register
/// AMX register need to be configured before use. The shape of AMX register
/// is encoded in the 1st and 2nd machine operand of AMX pseudo instructions.
/// The pldtilecfg is to config tile registers. It should dominator all AMX
/// instructions. The pldtilecfg produce a virtual cfg register and the cfg
/// register is used by all AMX instructions.
/// This pass is to find the common dominator of all AMX instructions and
/// insert the pldtilecfg instruction. Besides the cfg register that pldtilecfg
/// produces is inserted as the last operand of each AMX instruction. We use
/// this scheme to model the def-use relationship between AMX config instruction
/// and other AMX instructions. Below is an example.
///
///                        ----B1----
///                       /           \
///                      /             \
///                    B2               B3
///    %1:tile = PTILELOADDV        %2:tile = PTILELOADDV
///
///  is transformed to
///
///                            B1
///                 %25:tilecfg = PLDTILECFG
///                       /           \
///                      /             \
///  %1:tile = PTILELOADDV %25    %2:tile = PTILELOADDV %25
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TileShapeInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "tile-pre-config"

namespace {

class X86PreTileConfig : public MachineFunctionPass {
  // context
  MachineFunction *MF = nullptr;
  const X86Subtarget *ST = nullptr;
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  MachineDominatorTree *DomTree = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  MachineInstr *getTileConfigPoint();

public:
  X86PreTileConfig() : MachineFunctionPass(ID) {}

  /// Return the pass name.
  StringRef getPassName() const override {
    return "Tile Register Pre-configure";
  }

  /// X86PreTileConfig analysis usage.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Perform register allocation.
  bool runOnMachineFunction(MachineFunction &mf) override;

  static char ID;
};

} // end anonymous namespace

char X86PreTileConfig::ID = 0;

INITIALIZE_PASS_BEGIN(X86PreTileConfig, "tilepreconfig",
                      "Tile Register Configure", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(X86PreTileConfig, "tilepreconfig",
                    "Tile Register Configure", false, false)

void X86PreTileConfig::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineDominatorTree>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

static Register buildConfigMI(MachineBasicBlock::iterator MI, int FrameIdx,
                              const TargetInstrInfo *TII,
                              MachineRegisterInfo *MRI,
                              const X86Subtarget *ST) {
  auto *MBB = MI->getParent();

  // FIXME: AMX should assume AVX512 enabled.
  if (ST->hasAVX512()) {
    // Zero stack slot.
    Register Zmm = MRI->createVirtualRegister(&X86::VR512RegClass);
    BuildMI(*MBB, MI, DebugLoc(), TII->get(X86::VPXORDZrr), Zmm)
        .addReg(Zmm, RegState::Undef)
        .addReg(Zmm, RegState::Undef);
    addFrameReference(BuildMI(*MBB, MI, DebugLoc(), TII->get(X86::VMOVUPSZmr)),
                      FrameIdx)
        .addReg(Zmm);
  }

  // build psuedo ldtilecfg
  Register VReg = MRI->createVirtualRegister(&X86::TILECFGRegClass);

  addFrameReference(
      BuildMI(*MBB, MI, DebugLoc(), TII->get(X86::PLDTILECFG), VReg), FrameIdx);

  return VReg;
}

static ShapeT getShape(const MachineInstr &MI, MachineRegisterInfo *MRI) {
  unsigned Opcode = MI.getOpcode();
  switch (Opcode) {
  default:
    llvm_unreachable("Unexpected machine instruction on tile");
  case X86::PTILELOADDV:
  case X86::PTDPBSSDV:
  case X86::PTILEZEROV:
    MachineOperand &MO1 = const_cast<MachineOperand &>(MI.getOperand(1));
    MachineOperand &MO2 = const_cast<MachineOperand &>(MI.getOperand(2));
    ShapeT Shape(&MO1, &MO2, MRI);
    return Shape;
  }
}

MachineInstr *X86PreTileConfig::getTileConfigPoint() {
  DenseMap<Register, ShapeT> PhysShapeInfo;
  MachineBasicBlock *MBB = nullptr;
  DenseSet<const MachineInstr *> MIs;
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register VirtReg = Register::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(VirtReg))
      continue;
    const TargetRegisterClass &RC = *MRI->getRegClass(VirtReg);
    if (RC.getID() != X86::TILERegClassID)
      continue;

    // Find the common dominator for all MI that define tile register.
    for (const MachineOperand &MO : MRI->def_operands(VirtReg)) {
      if (MO.isUndef())
        continue;
      const auto *MI = MO.getParent();
      // PHI or IMPLICIT_DEF instructiion.
      // There must be a input tile before PHI instruction.
      if (MI->isTransient())
        continue;
      if (!MBB)
        MBB = const_cast<MachineBasicBlock *>(MI->getParent());
      MBB = DomTree->findNearestCommonDominator(
          MBB, const_cast<MachineBasicBlock *>(MI->getParent()));

      // Collect the instructions that define shape.
      ShapeT Shape = getShape(*MI, MRI);
      std::array<MachineOperand *, 2> ShapeMOs = {Shape.getRow(),
                                                  Shape.getCol()};
      for (auto *ShapeMO : ShapeMOs) {
        Register ShapeReg = ShapeMO->getReg();
        for (const MachineOperand &MO : MRI->def_operands(ShapeReg)) {
          const auto *ShapeMI = MO.getParent();
          MIs.insert(ShapeMI);
        }
      }
    }
  }
  if (!MBB)
    return nullptr;
  // This pass is before the pass of eliminating PHI node, so it
  // is in SSA form.
  assert(MRI->isSSA() && "Not SSA form in pre-tile config");
  // Shape def should dominate tile config MBB.
  //    def s           s1    s2
  //     / \             \   /
  //    /   \             \ /
  //  conf               s3=phi(s1,s2)
  //                       |
  //                       c
  //
  for (const auto *MI : MIs) {
    const MachineBasicBlock *ShapeMBB = MI->getParent();
    if (DomTree->dominates(ShapeMBB, MBB))
      continue;
    if (MI->isMoveImmediate())
      continue;
    report_fatal_error(MF->getName() + ": Failed to config tile register, "
                                       "please define the shape earlier");
  }

  // ldtilecfg should be inserted after the MI that define the shape.
  MachineBasicBlock::reverse_instr_iterator I, E;
  for (I = MBB->instr_rbegin(), E = MBB->instr_rend(); I != E; ++I) {
    auto *MI = &*I;
    if (MIs.count(MI) && (!MI->isMoveImmediate()))
      break;
  }
  MachineBasicBlock::iterator MII;
  if (I == E)
    MII = MBB->getFirstNonPHI();
  else {
    MII = MachineBasicBlock::iterator(&*I);
    MII++;
  }
  return &*MII;
}

static void addTileCFGUse(MachineFunction &MF, Register CFG) {
  for (MachineBasicBlock &MBB : MF) {

    // Traverse the basic block.
    for (MachineInstr &MI : MBB) {
      unsigned Opcode = MI.getOpcode();
      switch (Opcode) {
      default:
        break;
      case X86::PTILELOADDV:
      case X86::PTILESTOREDV:
      case X86::PTDPBSSDV:
      case X86::PTILEZEROV:
        unsigned NumOperands = MI.getNumOperands();
        MI.RemoveOperand(NumOperands - 1);
        MI.addOperand(MF, MachineOperand::CreateReg(CFG, false));
        break;
      }
    }
  }
}

bool X86PreTileConfig::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  MRI = &mf.getRegInfo();
  ST = &mf.getSubtarget<X86Subtarget>();
  TRI = ST->getRegisterInfo();
  TII = mf.getSubtarget().getInstrInfo();
  DomTree = &getAnalysis<MachineDominatorTree>();

  MachineInstr *MI = getTileConfigPoint();
  if (!MI)
    return false;
  unsigned Size = ST->getTileConfigSize();
  Align Alignment = ST->getTileConfigAlignment();
  int SS = mf.getFrameInfo().CreateStackObject(Size, Alignment, false);
  Register CFG = buildConfigMI(MI, SS, TII, MRI, ST);
  addTileCFGUse(mf, CFG);
  return true;
}

FunctionPass *llvm::createX86PreTileConfigPass() {
  return new X86PreTileConfig();
}
