//===-- X86FastTileConfig.cpp - Fast Tile Register Configure---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Pass to config the shape of AMX physical registers
/// AMX register need to be configured before use. Before FastRegAllocation pass
/// the ldtilecfg instruction is inserted, however at that time we don't
/// know the shape of each physical tile registers, because the register
/// allocation is not done yet. This pass runs after register allocation
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
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "fasttileconfig"

namespace {

class X86FastTileConfig : public MachineFunctionPass {
  // context
  MachineFunction *MF = nullptr;
  const X86Subtarget *ST = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  X86MachineFunctionInfo *X86FI = nullptr;

  MachineInstr *getTileConfigPoint();
  void tileConfig();

public:
  X86FastTileConfig() : MachineFunctionPass(ID) {}

  bool fastTileConfig();
  bool isTileLoad(MachineInstr &MI);
  bool isTileStore(MachineInstr &MI);
  bool isAMXInstr(MachineInstr &MI);
  void getTileStoreShape(MachineInstr &MI,
                         SmallVector<MachineOperand *> &ShapedTiles);

  MachineInstr *getKeyAMXInstr(MachineInstr *MI);
  void getTileShapesCfg(MachineInstr *MI,
                        SmallVector<MachineOperand *> &ShapedTiles);
  void getShapeCfgInstrs(MachineInstr *MI,
                         std::map<unsigned, MachineInstr *> &RowCfgs,
                         std::map<unsigned, MachineInstr *> &ColCfgs);

  /// Return the pass name.
  StringRef getPassName() const override {
    return "Fast Tile Register Configure";
  }

  void materializeTileCfg(MachineInstr *MI);

  void rewriteTileCfg(SmallVector<MachineOperand *> &ShapedTiles,
                      std::map<unsigned, MachineInstr *> &RowCfgs,
                      std::map<unsigned, MachineInstr *> &ColCfgs);

  /// Perform register allocation.
  bool runOnMachineFunction(MachineFunction &MFunc) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoPHIs);
  }

  static char ID;
};

} // end anonymous namespace

char X86FastTileConfig::ID = 0;

INITIALIZE_PASS_BEGIN(X86FastTileConfig, DEBUG_TYPE,
                      "Fast Tile Register Configure", false, false)
INITIALIZE_PASS_END(X86FastTileConfig, DEBUG_TYPE,
                    "Fast Tile Register Configure", false, false)

static bool isTilePhysReg(MachineOperand &Op) {
  if (!Op.isReg())
    return false;

  Register Reg = Op.getReg();
  if (Reg >= X86::TMM0 && Reg <= X86::TMM7)
    return true;
  return false;
}

static unsigned getTilePhysRegIdx(MachineOperand *Op) {
  assert(isTilePhysReg(*Op) && "Tile Operand is invalid");
  return Op->getReg() - X86::TMM0;
}

static inline void adjustRowCfg(unsigned TIdx, MachineInstr *MI) {
  unsigned Offset = 48 + TIdx;
  MI->getOperand(3).ChangeToImmediate(Offset);
}

static inline void adjustColCfg(unsigned TIdx, MachineInstr *MI) {
  unsigned Offset = 16 + TIdx * 2;
  MI->getOperand(3).ChangeToImmediate(Offset);
}

bool X86FastTileConfig::isTileLoad(MachineInstr &MI) {
  return MI.getOpcode() == X86::PTILELOADDV ||
         MI.getOpcode() == X86::PTILELOADDT1V;
}
bool X86FastTileConfig::isTileStore(MachineInstr &MI) {
  return MI.getOpcode() == X86::PTILESTOREDV;
}
bool X86FastTileConfig::isAMXInstr(MachineInstr &MI) {
  // TODO: May need to handle some special nontile amx instrucion.
  if (MI.getOpcode() == X86::PLDTILECFGV || MI.isDebugInstr())
    return false;

  for (MachineOperand &MO : MI.operands())
    if (isTilePhysReg(MO))
      return true;

  return false;
}

MachineInstr *X86FastTileConfig::getKeyAMXInstr(MachineInstr *MI) {
  auto Cfg = MachineBasicBlock::iterator(MI);
  MachineBasicBlock *MBB = MI->getParent();
  MachineInstr *KeyMI = nullptr;
  int KeyAMXNum = 0;

  for (auto II = Cfg; II != MBB->end(); II++) {
    if (isTileLoad(*II)) {
      KeyMI = &*II;
      continue;
    }

    if (isTileStore(*II)) {
      assert(KeyMI && "Key AMX Should be found before!");
      break;
    }

    if (isAMXInstr(*II)) {
      assert((KeyAMXNum == 0) && "Too many Key AMX instruction!");
      KeyAMXNum++;
      KeyMI = &*II;
    }
  }
  assert(KeyMI && "There must be an AMX instruction.");
  return KeyMI;
}

// Orderly get the tiles in key amx instruction, uses before defs.
void X86FastTileConfig::getTileShapesCfg(
    MachineInstr *CfgMI, SmallVector<MachineOperand *> &ShapedTiles) {
  MachineInstr *KeyMI = getKeyAMXInstr(CfgMI);

  SmallVector<MachineOperand *> DefTiles;
  for (MachineOperand &MO : KeyMI->operands()) {
    if (!isTilePhysReg(MO))
      continue;
    if (MO.isDef())
      DefTiles.push_back(&MO);
    else
      ShapedTiles.push_back(&MO);
  }
  ShapedTiles.append(DefTiles);
}

// We pre-config the shapes at position named with "amx.tmm.N.shape.row* and
// amx.shape.N.col*" at pass "Pre AMX Tile Config".
// The 'N' implies the order of tiles in key amx intrinsic.
void X86FastTileConfig::getShapeCfgInstrs(
    MachineInstr *MI, std::map<unsigned, MachineInstr *> &RowCfgs,
    std::map<unsigned, MachineInstr *> &ColCfgs) {
  auto Cfg = MachineBasicBlock::iterator(MI);
  MachineBasicBlock *MBB = MI->getParent();

  for (auto II = Cfg; II != MBB->begin(); II--) {
    if (isAMXInstr(*II) || II->isTerminator() || II->isCall())
      break;
    if (!II->mayStore() || !II->hasOneMemOperand())
      continue;
    const Value *MemPtr = II->memoperands()[0]->getValue();
    if (!MemPtr)
      continue;

    StringRef Name = MemPtr->getName();
    if (!Name.startswith("amx.tmm."))
      continue;

    // Get the 'N'th tile shape config in key amx instruction.
    auto N = Name.find(".shape");
    StringRef STileIdx = Name.slice(8, N);
    unsigned Idx;
    STileIdx.getAsInteger(10, Idx);

    // And related them with their store instructions.
    if (Name.contains("row"))
      RowCfgs[Idx] = &*II;
    else if (Name.contains("col"))
      ColCfgs[Idx] = &*II;
    else
      llvm_unreachable("Invalid tile shape info!");
  }
  assert((RowCfgs.size() == ColCfgs.size()) &&
         "The number of tile row and col must be equal!");
}

// Here is the data format for the tile config.
// 0      palette   = 1 now.
// 1      start_row = 0 now.
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
void X86FastTileConfig::rewriteTileCfg(
    SmallVector<MachineOperand *> &ShapedTiles,
    std::map<unsigned, MachineInstr *> &RowCfgs,
    std::map<unsigned, MachineInstr *> &ColCfgs) {
  assert((RowCfgs.size() == ShapedTiles.size()) &&
         "The number of tile shapes not equal with the number of tiles!");

  // Orderly get the tiles and adjust the shape config.
  for (unsigned I = 0, E = ShapedTiles.size(); I < E; I++) {
    MachineOperand *MO = ShapedTiles[I];
    unsigned TmmIdx = getTilePhysRegIdx(MO);
    if (I == TmmIdx)
      continue;
    adjustRowCfg(TmmIdx, RowCfgs[I]);
    adjustColCfg(TmmIdx, ColCfgs[I]);
  }
}

// We have already preconfig the shapes before fast register allocation at
// X86PreAMXConfig::preWriteTileCfg(). Now, we have done fast register
// allocation, the shapes pre-written before may not rightly corresponding
// to the correct tmm registers, so we need adjust them.
void X86FastTileConfig::materializeTileCfg(MachineInstr *CfgMI) {
  SmallVector<MachineOperand *> ShapedTiles;
  std::map<unsigned, MachineInstr *> RowCfgs;
  std::map<unsigned, MachineInstr *> ColCfgs;

  // Orderly keep the tile uses and def in ShapedTiles;
  getTileShapesCfg(CfgMI, ShapedTiles);
  assert(ShapedTiles.size() && "Not find shapes config!");

  getShapeCfgInstrs(CfgMI, RowCfgs, ColCfgs);

  rewriteTileCfg(ShapedTiles, RowCfgs, ColCfgs);
}

bool X86FastTileConfig::fastTileConfig() {
  bool Changed = false;

  for (MachineBasicBlock &MBB : *MF) {
    SmallVector<MachineInstr *, 2> CFGs;
    for (MachineInstr &MI : MBB)
      if (MI.getOpcode() == X86::PLDTILECFGV)
        CFGs.push_back(&MI);
    for (auto *MI : CFGs)
      materializeTileCfg(MI);
    if (!CFGs.empty())
      Changed = true;
  }
  if (Changed)
    X86FI->setHasVirtualTileReg(true);
  return Changed;
}

bool X86FastTileConfig::runOnMachineFunction(MachineFunction &MFunc) {
  MF = &MFunc;
  MRI = &MFunc.getRegInfo();
  ST = &MFunc.getSubtarget<X86Subtarget>();
  TRI = ST->getRegisterInfo();
  TII = MFunc.getSubtarget().getInstrInfo();
  X86FI = MFunc.getInfo<X86MachineFunctionInfo>();

  return fastTileConfig();
}

FunctionPass *llvm::createX86FastTileConfigPass() {
  return new X86FastTileConfig();
}
