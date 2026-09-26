//===-- SuperHExpandPseudoInsts.cpp - Expand pseudo instructions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands pseudo instructions into target
// instructions. This pass should be run after register allocation but before
// the post-regalloc scheduling pass.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "SuperH.h"
#include "SuperHConstantPoolValue.h"
#include "SuperHInstrInfo.h"
#include "SuperHMachineFunctionInfo.h"
#include "SuperHSubtarget.h"
#include "SuperHTargetMachine.h"

#include "llvm/ADT/APInt.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "sh-expand-pseudo"
#define SH_EXPAND_PSEUDO_NAME "SH pseudo instruction expansion pass"

#ifndef NDEBUG
static cl::opt<bool>
    CNoExpand("sh-no-expand", cl::Hidden, cl::init(false),
              cl::desc("Force the backend to not expand instructions."));
static cl::opt<bool>
    CKeepPseudo("sh-keep-pseudo", cl::Hidden, cl::init(false),
                cl::desc("Force the backend to keep pseudos around."));
#endif

namespace {
class SuperHExpandPseudo : public MachineFunctionPass {
public:
  static char ID;

  SuperHExpandPseudo() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return SH_EXPAND_PSEUDO_NAME; }

private:
  typedef MachineBasicBlock Block;
  typedef Block::iterator BlockIt;

  const SuperHSubtarget *STI;
  const SuperHRegisterInfo *TRI;
  const TargetInstrInfo *TII;

  bool expandMBB(Block &MBB);
  bool expandMI(Block &MBB, BlockIt MBBI);
  template <unsigned OP> bool expand(Block &MBB, BlockIt MBBI);
  void getStackOffset(Block &MBB, BlockIt MBBI, Register FrameReg, 
                      int64_t &Offset, uint8_t Bits, uint8_t Scale = 1);

  bool storeToFrame(Block &MBB, BlockIt MBBI, int Scale);
  bool storeToAddress(Block &MBB, BlockIt MBBI);
  bool loadFromFrame(Block &MBB, BlockIt MBBI, int Scale);
  bool loadFromAddress(Block &MBB, BlockIt MBBI);
  bool loadFromImmediate(Block &MBB, BlockIt MBBI);
};




//===----------------------------------------------------------------------===//
//                                Helpers
//===----------------------------------------------------------------------===//

// eraseMI - Helper that erases a machine instruction while respecting
// the debug option to keep them.
static bool eraseMI(MachineInstr &MI) {

#ifndef NDEBUG
  if (CKeepPseudo)
    return false;
#endif

  MI.eraseFromParent();
  return true;
}

// getStackOffset - Sets up the R1 stack reference register to access the 
// stack at the given offset via negative add instructions, Offset is set
// to the needed scaled offset for the stack access.
void SuperHExpandPseudo::getStackOffset(Block &MBB, BlockIt MBBI, Register FrameReg, 
                                        int64_t &Offset, uint8_t Bits, uint8_t Scale) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();


  // Split the stack up into indexable chunks, each instruction has
  // a fixed range that it can access, this range is positive only.
  int64_t StackSize = MFI.getStackSize();
  int64_t AccessRange = ((1 << Bits) * Scale);

  // The stack grows down, so the offset needs to be adjusted so that
  // we index correctly into the negative stack with a positive index.
  int64_t RealOffset = (StackSize-Offset)-Scale;

  // On big endian systems, adjust the pointer for
  // < 32-bit offsets.
  if (!STI->isLittleEndian() && Scale != 4)
    RealOffset -= 4-(Scale-1);

  int64_t SpAdjust = AccessRange * (alignTo(RealOffset, Scale) / AccessRange);

  // Expand sequence to
  // mov      <frame reg>,  r1
  // add      #-SpOffset,   r1
  BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R1).addReg(FrameReg);
  if (SpAdjust != 0)
    BuildMI(MBB, MBBI, DL, TII->get(SH::ADDI), SH::R1)
        .addReg(SH::R1)
        .addImm(-SpAdjust);

  // Adjust the offset to be within the access range.
  Offset = RealOffset % AccessRange;
}




//===----------------------------------------------------------------------===//
//                                Stores
//===----------------------------------------------------------------------===//

bool SuperHExpandPseudo::storeToFrame(Block &MBB, BlockIt MBBI, int Scale) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  const MachineFunction &MF = *MBB.getParent();
  MachineInstr &MI = *MBBI;

  bool SrcIsKill = MI.getOperand(0).isKill();
  auto SrcReg = MI.getOperand(0).getReg();
  auto FrameReg = MI.getOperand(1).getReg();
  auto Offset = MI.getOperand(2).getImm();
  getStackOffset(MBB, MBBI, FrameReg, Offset, 4, Scale);

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Expected valid MOV*SPtr opcode.");
  case SH::MOVBSF: {

    // mov      <src reg>,  r0
    // mov.b    r0, @(offset,r1)
    if (SrcReg != SH::R0)
      BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R0)
          .addReg(SrcReg, getKillRegState(SrcIsKill));
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVBS4))
        .addReg(SH::R1, RegState::Kill)
        .addImm(Offset);
    break;
  }
  case SH::MOVWSF: {

    // mov      <src reg>,  r0
    // mov.w    r0, @(offset,r1)
    if (SrcReg != SH::R0)
      BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R0)
          .addReg(SrcReg, getKillRegState(SrcIsKill));
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVWS4))
        .addReg(SH::R1, RegState::Kill)
        .addImm(Offset);
    break;
  }
  case SH::MOVLSF: {

    // mov.l    <src reg>, @(offset,r1)
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVLS4))
        .addReg(SrcReg, getKillRegState(SrcIsKill))
        .addReg(SH::R1, RegState::Kill)
        .addImm(Offset);
    break;
  }
  }

  return eraseMI(MI);
}

bool SuperHExpandPseudo::storeToAddress(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;
  const MachineFunction &MF = *MBB.getParent();
  const SuperHMachineFunctionInfo *FI = MF.getInfo<SuperHMachineFunctionInfo>();

  bool SrcIsKill = MI.getOperand(0).isKill();
  Register SrcReg = MI.getOperand(0).getReg();
  Register DstReg;

  if (MI.getOperand(1).isGlobal()) {
    if (auto *G = FI->tryGetConstant(MI.getOperand(1).getGlobal(), MF)) {
      BuildMI(MBB, MBBI, DL, TII->get(SH::MOVLI), SH::R1)
          .addConstantPoolIndex(G->getLabelId());
      DstReg = SH::R1;
    }
  } else if (MI.getOperand(1).isReg()) {
    DstReg = MI.getOperand(1).getReg();
  }

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Expected valid MOV*SPtr opcode.");
  case SH::MOVBSP: {
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVBS))
      .addReg(SrcReg, getKillRegState(SrcIsKill))
      .addReg(DstReg);
    break;
  }
  case SH::MOVWSP: {
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVWS))
      .addReg(SrcReg, getKillRegState(SrcIsKill))
      .addReg(DstReg);
    break;
  }
  case SH::MOVLSP: {
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVLS))
      .addReg(SrcReg, getKillRegState(SrcIsKill))
      .addReg(DstReg);
    break;
  }
  }

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVBSF>(Block &MBB, BlockIt MBBI) {

  // Store to stack frame
  return storeToFrame(MBB, MBBI, 1);
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVWSF>(Block &MBB, BlockIt MBBI) {

  // Store to stack frame
  return storeToFrame(MBB, MBBI, 2);
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVLSF>(Block &MBB, BlockIt MBBI) {

  // Store to stack frame
  return storeToFrame(MBB, MBBI, 4);
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVBSP>(Block &MBB, BlockIt MBBI) {
  
  // Store to address.
  return storeToAddress(MBB, MBBI);
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVWSP>(Block &MBB, BlockIt MBBI) {

  // Store to address.
  return storeToAddress(MBB, MBBI);
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVLSP>(Block &MBB, BlockIt MBBI) {

  // Store to address.
  return storeToAddress(MBB, MBBI);
}




//===----------------------------------------------------------------------===//
//                               Loads
//===----------------------------------------------------------------------===//

bool SuperHExpandPseudo::loadFromFrame(Block &MBB, BlockIt MBBI, int Scale) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  const MachineFunction &MF = *MBB.getParent();
  MachineInstr &MI = *MBBI;

  auto DstReg = MI.getOperand(0).getReg();
  bool DstIsKill = MI.getOperand(0).isKill();
  auto FrameReg = MI.getOperand(1).getReg();
  auto Offset = MI.getOperand(2).getImm();
  getStackOffset(MBB, MBBI, FrameReg, Offset, 4, Scale);

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Expected valid MOV*LPtr opcode.");
  case SH::MOVBLF: {

    // mov.b    @(offset,r1), r0
    // mov      r0,           <dst reg>
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVBL4))
        .addReg(SH::R1)
        .addImm(Offset)
        .addReg(SH::R0, RegState::Define);
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOV))
        .addReg(DstReg, getKillRegState(DstIsKill))
        .addReg(SH::R0, RegState::Kill);
    break;
  }
  case SH::MOVWLF: {

    // mov.w    @(offset,r1), r0
    // mov      r0,           <dst reg>
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVWL4))
      .addReg(SH::R1)
      .addImm(Offset);
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOV))
        .addReg(DstReg, getKillRegState(DstIsKill))
        .addReg(SH::R0, RegState::Define);
    break;
  }
  case SH::MOVLLF: {

    // mov.l    @(offset,r1), <dst reg>
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVLL4))
        .addReg(DstReg, getKillRegState(DstIsKill))
        .addReg(SH::R1)
        .addImm(Offset);
    break;
  }
  }

  return eraseMI(MI);
}

bool SuperHExpandPseudo::loadFromAddress(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;
  const MachineFunction &MF = *MBB.getParent();
  const SuperHMachineFunctionInfo *FI = MF.getInfo<SuperHMachineFunctionInfo>();

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg;

  if (MI.getOperand(1).isGlobal()) {
    if (auto *G = FI->tryGetConstant(MI.getOperand(1).getGlobal(), MF)) {
      BuildMI(MBB, MBBI, DL, TII->get(SH::MOVLI), SH::R1)
          .addConstantPoolIndex(G->getLabelId());
      SrcReg = SH::R1;
    }
  } else if (MI.getOperand(1).isReg()) {
    SrcReg = MI.getOperand(1).getReg();
  }

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Expected valid MOV*LPtr opcode.");
  case SH::MOVBLP: {
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVBL), DstReg).addReg(SrcReg);
    break;
  }
  case SH::MOVWLP: {
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVWL), DstReg).addReg(SrcReg);
    break;
  }
  case SH::MOVLLP: {
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVLL), DstReg).addReg(SrcReg);
    break;
  }
  }

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVBLF>(Block &MBB, BlockIt MBBI) {

  // Load from stack frame.
  return loadFromFrame(MBB, MBBI, 1);
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVWLF>(Block &MBB, BlockIt MBBI) {

  // Load from stack frame.
  return loadFromFrame(MBB, MBBI, 2);
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVLLF>(Block &MBB, BlockIt MBBI) {

  // Load from stack frame.
  return loadFromFrame(MBB, MBBI, 4);
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVBLP>(Block &MBB, BlockIt MBBI) {

  // Load from address.
  return loadFromAddress(MBB, MBBI);
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVWLP>(Block &MBB, BlockIt MBBI) {

  // Load from address.
  return loadFromAddress(MBB, MBBI);
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVLLP>(Block &MBB, BlockIt MBBI) {

  // Load from address.
  return loadFromAddress(MBB, MBBI);
}




//===----------------------------------------------------------------------===//
//                              Immediate Loads
//===----------------------------------------------------------------------===//

bool SuperHExpandPseudo::loadFromImmediate(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;
  auto DstReg = MI.getOperand(0).getReg();

  if (MI.getOperand(1).isImm()) {
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVI), DstReg)
        .addImm(MI.getOperand(1).getImm());
  } else if (MI.getOperand(1).isCPI()) {
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVLI), DstReg)
          .addConstantPoolIndex(MI.getOperand(1).getIndex());
  }

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVIB>(Block &MBB, BlockIt MBBI) {
  return loadFromImmediate(MBB, MBBI);
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVIW>(Block &MBB, BlockIt MBBI) {
  return loadFromImmediate(MBB, MBBI);
}

template <>
bool SuperHExpandPseudo::expand<SH::MOVIL>(Block &MBB, BlockIt MBBI) {
  return loadFromImmediate(MBB, MBBI);
}




//===----------------------------------------------------------------------===//
//                              Bit Shifting
//===----------------------------------------------------------------------===//

template <>
bool SuperHExpandPseudo::expand<SH::SHLri>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  auto DstReg = MI.getOperand(0).getReg();
  auto SrcReg = MI.getOperand(1).getReg();
  int64_t Offset = MI.getOperand(2).getImm();

  while (Offset > 0) {

    if (Offset > 16) {
      BuildMI(MBB, MBBI, DL, TII->get(SH::SHLL16), DstReg).addReg(SrcReg);

      Offset -= 16;
      continue;
    }

    if (Offset > 8) {
      BuildMI(MBB, MBBI, DL, TII->get(SH::SHLL8), DstReg).addReg(SrcReg);

      Offset -= 8;
      continue;
    }

    if (Offset > 2) {
      BuildMI(MBB, MBBI, DL, TII->get(SH::SHLL2), DstReg).addReg(SrcReg);

      Offset -= 2;
      continue;
    }

    BuildMI(MBB, MBBI, DL, TII->get(SH::SHLL), DstReg).addReg(SrcReg);
    Offset -= 1;
    continue;
  }

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::SHRri>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  auto DstReg = MI.getOperand(0).getReg();
  auto SrcReg = MI.getOperand(1).getReg();
  int64_t Offset = MI.getOperand(2).getImm();

  while (Offset > 0) {

    if (Offset > 16) {
      BuildMI(MBB, MBBI, DL, TII->get(SH::SHLR16), DstReg).addReg(SrcReg);

      Offset -= 16;
      continue;
    }

    if (Offset > 8) {
      BuildMI(MBB, MBBI, DL, TII->get(SH::SHLR8), DstReg).addReg(SrcReg);

      Offset -= 8;
      continue;
    }

    if (Offset > 2) {
      BuildMI(MBB, MBBI, DL, TII->get(SH::SHLR2), DstReg).addReg(SrcReg);

      Offset -= 2;
      continue;
    }

    BuildMI(MBB, MBBI, DL, TII->get(SH::SHLR), DstReg).addReg(SrcReg);
    Offset -= 1;
    continue;
  }

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::SRAri>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;
  const MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  auto DstReg = MI.getOperand(0).getReg();
  auto SrcReg = MI.getOperand(1).getReg();
  int64_t Offset = MI.getOperand(2).getImm();

  while (Offset > 0) {
    BuildMI(MBB, MBBI, DL, TII->get(SH::SHAR), DstReg).addReg(SrcReg);
    Offset -= 1;
  }

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::SHLrr>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;

  auto Src1Reg = MI.getOperand(1).getReg();
  auto Src2Reg = MI.getOperand(2).getReg();

  BuildMI(MBB, MBBI, DL, TII->get(SH::SHLL)).addReg(Src1Reg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::DT)).addReg(Src2Reg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::BF)).addImm(-4);

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::SHRrr>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;

  auto Src1Reg = MI.getOperand(1).getReg();
  auto Src2Reg = MI.getOperand(2).getReg();

  BuildMI(MBB, MBBI, DL, TII->get(SH::SHLR)).addReg(Src1Reg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::DT)).addReg(Src2Reg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::BF)).addImm(-4);

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::SRArr>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;

  auto Src1Reg = MI.getOperand(1).getReg();
  auto Src2Reg = MI.getOperand(2).getReg();

  BuildMI(MBB, MBBI, DL, TII->get(SH::SHAR)).addReg(Src1Reg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::DT)).addReg(Src2Reg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::BF)).addImm(-4);

  return eraseMI(MI);
}




//===----------------------------------------------------------------------===//
//                            General Interface
//===----------------------------------------------------------------------===//

bool SuperHExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  BlockIt MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    BlockIt NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool SuperHExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "\n********** SuperHExpandPseudo **********\n");
  bool Modified = false;

#ifndef NDEBUG
  if (CNoExpand)
    return false;
#endif

  STI = &MF.getSubtarget<SuperHSubtarget>();
  TRI = STI->getRegisterInfo();
  TII = STI->getInstrInfo();

  for (Block &MBB : MF) {
    bool ContinueExpanding = true;
    unsigned ExpandCount = 0;

    // Continue expanding the block until all pseudos are expanded.
    do {
      assert(ExpandCount < 10 && "pseudo expand limit reached");
      (void)ExpandCount;

      bool BlockModified = expandMBB(MBB);
      Modified |= BlockModified;
      ExpandCount++;

      ContinueExpanding = BlockModified;
    } while (ContinueExpanding);
  }

  return Modified;
}

bool SuperHExpandPseudo::expandMI(Block &MBB, BlockIt MBBI) {
  MachineInstr &MI = *MBBI;
  int Opcode = MBBI->getOpcode();

#define EXPAND(Op)                                                             \
  case Op:                                                                     \
    return expand<Op>(MBB, MI)

  switch (Opcode) {
  default:
    break;
    EXPAND(SH::MOVBSF);
    EXPAND(SH::MOVWSF);
    EXPAND(SH::MOVLSF);
    EXPAND(SH::MOVBLF);
    EXPAND(SH::MOVWLF);
    EXPAND(SH::MOVLLF);
    EXPAND(SH::MOVBSP);
    EXPAND(SH::MOVWSP);
    EXPAND(SH::MOVLSP);
    EXPAND(SH::MOVBLP);
    EXPAND(SH::MOVWLP);
    EXPAND(SH::MOVLLP);
    EXPAND(SH::MOVIB);
    EXPAND(SH::MOVIW);
    EXPAND(SH::MOVIL);
    EXPAND(SH::SHLri);
    EXPAND(SH::SHRri);
    EXPAND(SH::SRAri);
    EXPAND(SH::SHLrr);
    EXPAND(SH::SHRrr);
    EXPAND(SH::SRArr);
  }
#undef EXPAND
  return false;
}

char SuperHExpandPseudo::ID = 0;

} // namespace

INITIALIZE_PASS(SuperHExpandPseudo, "sh-expand-pseudo", SH_EXPAND_PSEUDO_NAME,
                false, false)

FunctionPass *llvm::createSuperHExpandPseudoPass() {
  return new SuperHExpandPseudo();
}