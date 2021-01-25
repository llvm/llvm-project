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
#include "llvm/CodeGen/LiveIntervals.h"
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
  LiveIntervals *LIS = nullptr;
  SmallVector<Register, 16> VTileRegs;
  MachineInstr *TileConfigMI = nullptr;

  void buildConfigMI(MachineBasicBlock::iterator MI, int FrameIdx);
  MachineInstr *getTileConfigPoint();
  void reloadTileConfig(int FI);

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
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(X86PreTileConfig, "tilepreconfig",
                    "Tile Register Configure", false, false)

void X86PreTileConfig::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  AU.addRequired<MachineDominatorTree>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void X86PreTileConfig::buildConfigMI(MachineBasicBlock::iterator MI,
                                     int FrameIdx) {
  auto *MBB = MI->getParent();

  // FIXME: AMX should assume AVX512 enabled.
  if (ST->hasAVX512()) {
    // Zero stack slot.
    Register Zmm = MRI->createVirtualRegister(&X86::VR512RegClass);
    BuildMI(*MBB, MI, DebugLoc(), TII->get(X86::VPXORDZrr), Zmm)
        .addReg(Zmm, RegState::Undef)
        .addReg(Zmm, RegState::Undef);
    TileConfigMI = &*addFrameReference(BuildMI(*MBB, MI, DebugLoc(),
                                               TII->get(X86::VMOVUPSZmr)),
                                       FrameIdx)
                         .addReg(Zmm);
  }

  // build psuedo ldtilecfg
  addFrameReference(BuildMI(*MBB, MI, DebugLoc(), TII->get(X86::LDTILECFG)),
                    FrameIdx);
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
    VTileRegs.push_back(VirtReg);

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

void X86PreTileConfig::reloadTileConfig(int FI) {
  SmallSet<MachineInstr *, 8> MIVisited;
  const TargetRegisterClass *RC = TRI->getRegClass(X86::TILERegClassID);
  auto TileRegNum = RC->getNumRegs();

  for (Register VReg : VTileRegs) {
    BitVector UsableRegs(TRI->getNumRegs());
    for (unsigned I = 0; I < TileRegNum; I++)
      UsableRegs.set(X86::TMM0 + I);
    SmallVector<SlotIndex, 8> RegSlots;
    SmallVector<const uint32_t *, 8> RegMasks;
    LiveInterval &LI = LIS->getInterval(VReg);
    if (!LIS->getInterferenceRegMasks(LI, RegSlots, RegMasks))
      continue;
    for (unsigned I = 0; I < RegSlots.size(); I++) {
      SlotIndex &SI = RegSlots[I];
      MachineInstr *MI = LIS->getInstructionFromIndex(SI);
      // We have reload the tile config register before.
      if (MIVisited.count(MI))
        continue;
      // For inline assembly, we don't reload tile config register.
      // If there is any ldtilecfg instruction in inline assembly,
      // it is user's reponsibility to restore everything.
      if (!MI->isCall())
        continue;
      UsableRegs.clearBitsInMask(RegMasks[I]);
      MIVisited.insert(MI);
      // There is no interference in callee. This is benifited from
      // IPRA.
      if (UsableRegs.none())
        continue;

      // build psuedo ldtilecfg
      auto *MBB = MI->getParent();
      auto MII = MachineBasicBlock::iterator(MI);
      MII++;
      addFrameReference(
          BuildMI(*MBB, *MII, DebugLoc(), TII->get(X86::LDTILECFG)), FI);
    }
  }
  // We just check tile data register interference, we also need check tile
  // config register interference. Since we don't model the config register
  // we should check interference from the ldtilecfg to each tile data register
  // def.
  //              ldtilecfg
  //              /       \
  //             BB1      BB2
  //             /         \
  //            call       BB3
  //            /           \
  //        %1=tileload   %2=tilezero
  // We can start from the instruction of each tile def, and backward to
  // ldtilecfg. If there is any call instruction, and tile data register is
  // not preserved, we should insert ldtilecfg after the call instruction.
  SmallSet<MachineBasicBlock *, 8> MBBVisited;
  for (Register VReg : VTileRegs) {
    for (MachineOperand &MO : MRI->def_operands(VReg)) {
      if (MO.isUndef())
        continue;
      MachineInstr *MI = MO.getParent();
      // May be PHI instructiion.
      // There must be several def tile before PHI instruction.
      if (MI->isTransient())
        continue;

      bool Terminate = false;
      MachineBasicBlock *MBB = MI->getParent();
      // backward to see if there is any call instruction after ldtilecfg.
      std::queue<MachineBasicBlock *> WorkList;
      WorkList.push(MBB);
      bool First = true;
      while (!WorkList.empty()) {
        MBB = WorkList.front();
        WorkList.pop();
        // If we have iterate the basic block before, don't iterate it and
        // its predecessor again. This may be caused by loop, or it has a
        // cross path from several successor, or it has been iterated when
        // handle other tile register. In below example, BB1 hit the condition.
        //               ldtilecfg
        //                  |
        //              ---BB1---
        //              /        \
        //            BB2        BB3
        //            /           \
        //        %1=tileload   %2=tilezero
        if (MBBVisited.count(MBB))
          continue;
        // For the first MBB, we start from the amx instruction which def
        // tile register.
        auto I = (First) ? MI->getReverseIterator() : MBB->instr_rbegin();
        for (auto E = MBB->instr_rend(); I != E; ++I) {
          // If it is inserted point for ldtilecfg, then we've finished
          // backward.
          if (&*I == TileConfigMI) {
            Terminate = true;
            break;
          }
          if (MIVisited.count(&*I))
            continue;
          if (!I->isCall())
            continue;
          BitVector UsableRegs(TRI->getNumRegs());
          for (unsigned I = 0; I < TileRegNum; I++)
            UsableRegs.set(X86::TMM0 + I);
          for (MachineOperand &CallMO : I->operands()) {
            if (CallMO.isRegMask())
              UsableRegs.clearBitsInMask(CallMO.getRegMask());
          }
          // Record the call to avoid double ldtilecfg insert.
          MIVisited.insert(&*I);
          if (UsableRegs.none())
            continue;
          // Insert ldtilecfg after call instruction.
          --I;
          addFrameReference(
              BuildMI(*MBB, *I, DebugLoc(), TII->get(X86::LDTILECFG)), FI);
        }
        // We encounter visited MachineInst, so we don't need to do backward
        // again.
        if (Terminate)
          break;
        // Next we will iterate its predecessor.
        for (MachineBasicBlock::pred_iterator S = MBB->pred_begin(),
                                              E = MBB->pred_end();
             S != E; S++)
          WorkList.push(*S);

        // The first the MBB may be visited for the second time when it is in
        // a loop.
        if (!First)
          MBBVisited.insert(MBB);
        First = false;
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
  LIS = &getAnalysis<LiveIntervals>();

  auto *TileConfigPoint = getTileConfigPoint();
  if (!TileConfigPoint)
    return false;
  unsigned Size = ST->getTileConfigSize();
  Align Alignment = ST->getTileConfigAlignment();
  int SS = mf.getFrameInfo().CreateStackObject(Size, Alignment, false);
  buildConfigMI(TileConfigPoint, SS);
  reloadTileConfig(SS);
  VTileRegs.clear();
  return true;
}

FunctionPass *llvm::createX86PreTileConfigPass() {
  return new X86PreTileConfig();
}
