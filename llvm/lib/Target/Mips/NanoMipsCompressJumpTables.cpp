//===- NanoMipsCompressJumpTables.cpp - nanoMIPS compress JTs  --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a pass that compresses Jump Table entries, whenever
/// possible. Jump table entries used to be fixed size(4B). They used to
/// represent absolute addresses. We want to compress those entries by filling
/// them with specific offsets. Having offsets instead of absolute addresses
/// saves at least 2B per entry. This pass checks if one or two bytes are
/// sufficient for the offset value.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsMachineFunction.h"
#include "MipsSubtarget.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"

#include <cmath>

using namespace llvm;

#define NM_COMPRESS_JUMP_TABLES_OPT_NAME                                       \
  "nanoMIPS compress jump tables optimization pass"

namespace {
struct NMCompressJumpTables : public MachineFunctionPass {
  static char ID;
  const MipsSubtarget *STI;
  const TargetInstrInfo *TII;
  MachineFunction *MF;
  SmallVector<int, 8> BlockInfo;
  SmallVector<int, 8> BrOffsets;

  int computeBlockSize(MachineBasicBlock &MBB);
  void scanFunction();
  bool compressJumpTable(MachineInstr &MI, int Offset);

  NMCompressJumpTables() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override {
    return NM_COMPRESS_JUMP_TABLES_OPT_NAME;
  }
  bool runOnMachineFunction(MachineFunction &Fn) override;
};
} // namespace

char NMCompressJumpTables::ID = 0;

// TODO: Currently, there is no existing LLVM interface which we can use to tell the
// maximum potential size of a MachineInstr. Once we have it, this should be
// enhanced.
int NMCompressJumpTables::computeBlockSize(MachineBasicBlock &MBB) {
  int Size = 0;
  for (const MachineInstr &MI : MBB)
    Size += TII->getInstSizeInBytes(MI);
  return Size;
}

void NMCompressJumpTables::scanFunction() {
  BlockInfo.clear();
  BlockInfo.resize(MF->getNumBlockIDs());
  BrOffsets.clear();
  bool findBR = MF->getJumpTableInfo() &&
                !MF->getJumpTableInfo()->getJumpTables().empty();
  if (findBR)
    BrOffsets.resize(MF->getJumpTableInfo()->getJumpTables().size());
  int Offset = 0;
  for (MachineBasicBlock &MBB : *MF) {
    BlockInfo[MBB.getNumber()] = Offset;
    Offset += computeBlockSize(MBB);
    if (findBR)
      for (auto &MI : MBB) {
        if (MI.getOpcode() == Mips::BRSC_NM) {
          int JTIdx = MI.getOperand(1).getIndex();
          BrOffsets[JTIdx] = Offset;
          break;
        }
      }
  }
}

bool NMCompressJumpTables::compressJumpTable(MachineInstr &MI, int Offset) {
  if (MI.getOpcode() != Mips::LoadJumpTableOffset)
    return false;

  int JTIdx = MI.getOperand(3).getIndex();
  auto &JTInfo = *MF->getJumpTableInfo();
  const MachineJumpTableEntry &JT = JTInfo.getJumpTables()[JTIdx];

  // The jump-table might have been optimized away.
  if (JT.MBBs.empty())
    return false;

  int MaxOffset = std::numeric_limits<int>::min(),
      MinOffset = std::numeric_limits<int>::max();
  int BrOffset = BrOffsets[JTIdx];

  bool Signed = false;
  for (auto Block : JT.MBBs) {
    int BlockOffset = BlockInfo[Block->getNumber()];
    MaxOffset = std::max(MaxOffset, BlockOffset - BrOffset);
    MinOffset = std::min(MinOffset, BlockOffset - BrOffset);
  }
  if (MinOffset < 0)
    Signed = true;

  if (std::max(std::abs(MinOffset), MaxOffset) == MinOffset)
    MaxOffset = MinOffset;

  auto MFI = MF->getInfo<MipsFunctionInfo>();
  MCSymbol *JTS = MFI->getJumpTableSymbol(JTIdx);

  bool EntrySize1 =
      (Signed && isInt<8>(MaxOffset)) || (!Signed && isUInt<8>(MaxOffset));
  bool EntrySize2 =
      (Signed && isInt<16>(MaxOffset)) || (!Signed && isUInt<16>(MaxOffset));
  int EntrySize = EntrySize1 ? 1 : (EntrySize2 ? 2 : 4);
  if (EntrySize1 || EntrySize2)
    MFI->setJumpTableEntryInfo(JTIdx, EntrySize, JTS, Signed);

  return false;
}

bool NMCompressJumpTables::runOnMachineFunction(MachineFunction &Fn) {
  STI = &static_cast<const MipsSubtarget &>(Fn.getSubtarget());
  TII = STI->getInstrInfo();
  bool Modified = false;
  MF = &Fn;

  scanFunction();

  for (MachineBasicBlock &MBB : *MF) {
    int Offset = BlockInfo[MBB.getNumber()];
    for (MachineInstr &MI : MBB) {
      Modified |= compressJumpTable(MI, Offset);
      Offset += TII->getInstSizeInBytes(MI);
    }
  }
  return Modified;
}

namespace llvm {
FunctionPass *createNanoMipsCompressJumpTablesPass() {
  return new NMCompressJumpTables();
}
} // namespace llvm
