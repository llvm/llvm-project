//===-- X86TileConfig.cpp - Tile Register Configure----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Pass to config the shape of AMX physical registers
/// AMX register need to be configured before use. In X86PreTileConfig pass
/// the pldtilecfg instruction is inserted, however at that time we don't
/// know the shape of each physical tile registers, because the register
/// allocation is not done yet. This pass runs after egister allocation
/// pass. It collects the shape information of each physical tile register
/// and store the shape in the stack slot that is allocated for load config
/// to tile config register.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86MachineFunctionInfo.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TileShapeInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "tile-config"

namespace {

class X86TileConfig : public MachineFunctionPass {
  // context
  MachineFunction *MF = nullptr;
  const X86Subtarget *ST = nullptr;
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  MachineDominatorTree *DomTree = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  VirtRegMap *VRM = nullptr;
  LiveIntervals *LIS = nullptr;

  MachineInstr *getTileConfigPoint();
  void tileConfig();

public:
  X86TileConfig() : MachineFunctionPass(ID) {}

  /// Return the pass name.
  StringRef getPassName() const override { return "Tile Register Configure"; }

  /// X86TileConfig analysis usage.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Perform register allocation.
  bool runOnMachineFunction(MachineFunction &mf) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoPHIs);
  }

  static char ID;
};

} // end anonymous namespace

char X86TileConfig::ID = 0;

INITIALIZE_PASS_BEGIN(X86TileConfig, "tileconfig", "Tile Register Configure",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_END(X86TileConfig, "tileconfig", "Tile Register Configure",
                    false, false)

void X86TileConfig::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineDominatorTree>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<VirtRegMap>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

static unsigned getTilePhysRegIndex(Register PhysReg) {
  assert((PhysReg >= X86::TMM0 && X86::TMM0 <= X86::TMM7) &&
         "Tile register number is invalid");
  return (PhysReg - X86::TMM0);
}

static MachineInstr *
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                    Register SrcReg, unsigned BitSize, int FrameIdx, int Offset,
                    const TargetInstrInfo *TII, const TargetRegisterClass *RC,
                    const TargetRegisterInfo *TRI) {

  unsigned SubIdx = (BitSize == 8) ? X86::sub_8bit : X86::sub_16bit;
  unsigned Opc = (BitSize == 8) ? X86::MOV8mr : X86::MOV16mr;
  if (BitSize == TRI->getRegSizeInBits(*RC))
    SubIdx = 0;
  MachineInstr *NewMI =
      addFrameReference(BuildMI(MBB, MI, DebugLoc(), TII->get(Opc)), FrameIdx,
                        Offset)
          .addReg(SrcReg, 0, SubIdx);
  return NewMI;
}

static MachineInstr *storeImmToStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                         int64_t Imm, unsigned BitSize,
                                         int FrameIdx, int Offset,
                                         const TargetInstrInfo *TII) {
  unsigned Opc = (BitSize == 8) ? X86::MOV8mi : X86::MOV16mi;
  return addFrameReference(BuildMI(MBB, MI, DebugLoc(), TII->get(Opc)),
                           FrameIdx, Offset)
      .addImm(Imm);
}

MachineInstr *X86TileConfig::getTileConfigPoint() {
  MachineBasicBlock *Entry = &*MF->begin();
  ReversePostOrderTraversal<MachineBasicBlock *> RPOT(Entry);
  for (MachineBasicBlock *MBB : RPOT) {
    for (MachineInstr &MI : *MBB)
      // Refer X86PreTileConfig.cpp.
      // We only support one tile config for now. The other ldtilecfg
      // is for spill purpose and is dominated by the first ldtilecfg.
      if (MI.getOpcode() == X86::LDTILECFG)
        return &MI;
  }

  return nullptr;
}

void X86TileConfig::tileConfig() {
  MachineInstr *MI = getTileConfigPoint();
  if (!MI)
    return;
  MachineBasicBlock *MBB = MI->getParent();
  int SS = MI->getOperand(0).getIndex();
  BitVector PhysRegs(TRI->getNumRegs());

  // Fill in the palette first.
  auto *NewMI = storeImmToStackSlot(*MBB, *MI, 1, 8, SS, 0, TII);
  LIS->InsertMachineInstrInMaps(*NewMI);
  // Fill in the shape of each tile physical register.
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register VirtReg = Register::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(VirtReg))
      continue;
    const TargetRegisterClass &RC = *MRI->getRegClass(VirtReg);
    if (RC.getID() != X86::TILERegClassID)
      continue;
    Register PhysReg = VRM->getPhys(VirtReg);
    if (PhysRegs.test(PhysReg))
      continue;
    PhysRegs.set(PhysReg);
    ShapeT Shape = VRM->getShape(VirtReg);
    Register RowReg = Shape.getRow()->getReg();
    Register ColReg = Shape.getCol()->getReg();

    // Here is the data format for the tile config.
    // 0      palette
    // 1      start_row
    // 2-15   reserved, must be zero
    // 16-17  tile0.colsb Tile 0 bytes per row.
    // 18-19  tile1.colsb Tile 1 bytes per row.
    // 20-21  tile2.colsb Tile 2 bytes per row.
    // ... (sequence continues)
    // 30-31  tile7.colsb Tile 7 bytes per row.
    // 32-47  reserved, must be zero
    // 48     tile0.rows Tile 0 rows.
    // 49     tile1.rows Tile 1 rows.
    // 50     tile2.rows Tile 2 rows.
    // ... (sequence continues)
    // 55     tile7.rows Tile 7 rows.
    // 56-63  reserved, must be zero
    unsigned Index = getTilePhysRegIndex(PhysReg);
    int RowOffset = 48 + Index;
    int ColOffset = 16 + Index * 2;

    unsigned BitSize = 8;
    for (const auto &Pair : {std::make_pair(RowReg, RowOffset),
                             std::make_pair(ColReg, ColOffset)}) {
      int64_t Imm;
      int ImmCount = 0;
      // All def must be the same value, otherwise it is invalid MIs.
      // Immediate is prefered.
      for (const MachineOperand &MO : MRI->def_operands(Pair.first)) {
        const auto *Inst = MO.getParent();
        if (Inst->isMoveImmediate()) {
          ImmCount++;
          Imm = Inst->getOperand(1).getImm();
          break;
        }
      }
      auto StoreConfig = [&](int Offset) {
        MachineInstr *NewMI = nullptr;
        if (ImmCount)
          NewMI = storeImmToStackSlot(*MBB, *MI, Imm, BitSize, SS, Offset, TII);
        else {
          const TargetRegisterClass *RC = MRI->getRegClass(Pair.first);
          NewMI = storeRegToStackSlot(*MBB, *MI, Pair.first, BitSize, SS,
                                      Offset, TII, RC, TRI);
        }
        SlotIndex SIdx = LIS->InsertMachineInstrInMaps(*NewMI);
        if (!ImmCount) {
          // Extend the live interval.
          SmallVector<SlotIndex, 8> EndPoints = {SIdx.getRegSlot()};
          LiveInterval &Int = LIS->getInterval(Pair.first);
          LIS->extendToIndices(Int, EndPoints);
        }
      };
      StoreConfig(Pair.second);
      BitSize += 8;
    }
  }
}

bool X86TileConfig::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  MRI = &mf.getRegInfo();
  ST = &mf.getSubtarget<X86Subtarget>();
  TRI = ST->getRegisterInfo();
  TII = mf.getSubtarget().getInstrInfo();
  DomTree = &getAnalysis<MachineDominatorTree>();
  VRM = &getAnalysis<VirtRegMap>();
  LIS = &getAnalysis<LiveIntervals>();

  if (VRM->isShapeMapEmpty())
    return false;

  tileConfig();
  return true;
}

FunctionPass *llvm::createX86TileConfigPass() { return new X86TileConfig(); }
