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
#include <cstddef>

using namespace llvm;

#define DEBUG_TYPE "sh-expand-pseudo"
#define SH_EXPAND_PSEUDO_NAME "SH pseudo instruction expansion pass"

static cl::opt<int64_t>
    CFPMaxAdd("sh-fp-max-add", cl::Hidden, cl::init(16),
                cl::desc("Maximum amount of add instructions to generate for heap fp load/stores."));

#ifndef NDEBUG
static cl::opt<bool>
    CNoExpand("sh-no-expand", cl::Hidden, cl::init(false),
              cl::desc("Force the backend to not expand instructions."));
static cl::opt<bool>
    CKeepPseudo("sh-keep-pseudo", cl::Hidden, cl::init(false),
                cl::desc("Force the backend to keep pseudos around."));
#endif

namespace {
struct StackAddr {

  /// Relative base address of the stack index.
  int64_t Base;

  /// < Offset from base address.
  int64_t Offset;
};

struct StackRange {
  int64_t Start;
  int64_t End;

  /// Gets the size of the stack range.
  int64_t stride() { return End - Start; }
};

class SuperHExpandPseudo : public MachineFunctionPass {
public:
  static char ID;

  SuperHExpandPseudo() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return SH_EXPAND_PSEUDO_NAME; }

private:
  using Block = MachineBasicBlock;
  using BlockIt = Block::iterator;

  const SuperHSubtarget *STI;
  const SuperHRegisterInfo *TRI;
  const SuperHInstrInfo *TII;

  /// getStackBucketFor - Gets the enclosing bucket for a given Offset
  /// from the start of a stack frame.
  StackRange getStackBucketFor(const MachineFunction &MF, int64_t Offset, 
                               uint8_t Bits, uint8_t Scale);

  /// getStackOffset - Calculates the base and offset address for accessing
  /// a stack slot.
  StackAddr getStackOffset(const MachineFunction &MF, Register FrameReg,
                           int64_t Offset, uint8_t Bits, uint8_t Scale = 1);

  bool expandMBB(Block &MBB);
  bool expandMI(Block &MBB, BlockIt MBBI);
  template <unsigned OP> bool expand(Block &MBB, BlockIt MBBI);
  void emitStackAdj(Block &MBB, BlockIt MBBI, Register FrameReg, StackAddr Addr);

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

/// truncdiv - Performs a truncating division with 8 bits
/// of fixed point preceision.
static int64_t truncdiv(int64_t V, int64_t D) {
  return ((V << 8LL) / D) >> 8LL;
}



//===----------------------------------------------------------------------===//
//                            Stack Offset Calculation
//===----------------------------------------------------------------------===//
// SuperH instructions have a very limited indexing range for indirect
// operations, additionally indirect loads and stores can only index positively.
// All SuperH ABIs have the stack grow down, as a result indirect loads cannot
// indirectly index into the stack. To access any stack slot a negative offset
// must be applied to a temporary anchor to positively index from it.
//
// The algorithm for calculating stack offsets is roughly designed as such:
//  1.  The stack is split up into buckets, the bucket size is a multiple of
//      the implicit scaling factor for the stack operation multiplied by
//      the amount of bits in the load/store operand.
//  2.  The stack/frame pointer is copied to R1.
//  3.  If the reference point is based on R14 (frame pointer), the size of 
//      the stack is subtracted from the index, ensuring the index is 
//      relative to the start of the stack frame.
//  4.  R1 is adjusted via a sequence of ADDI instructions to the nearest
//      bucket boundary.
//  5.  The output offset is adjusted such that it is indexing relative to the
//      end of the enclosing bucket.
//===----------------------------------------------------------------------===//

StackRange SuperHExpandPseudo::getStackBucketFor(const MachineFunction &MF,
                                                 int64_t Offset, uint8_t Bits, 
                                                 uint8_t Scale) {
  int64_t BucketStride = (1 << Bits) * Scale;
  int64_t BucketIdx = truncdiv(Offset-1, BucketStride);
  return { BucketIdx*BucketStride, ((BucketIdx+1)*BucketStride) };
}

StackAddr SuperHExpandPseudo::getStackOffset(const MachineFunction &MF, 
                                            Register FrameReg, int64_t Offset, 
                                            uint8_t Bits, uint8_t Scale) {
  assert(TRI->getFrameRegister(MF) == FrameReg && "Unexpected frame register!");
  const TargetFrameLowering *TFI = TRI->getFrameLowering(MF);
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  int64_t StackSize = MFI.getStackSize();
  StackRange Bucket = getStackBucketFor(MF, Offset, Bits, Scale);

  // NOTE: The 4 here adjusts out the stack frame pointer, otherwise
  //       the offset will be wrong.
  return {
    (StackSize-Bucket.End),
    (Bucket.stride() - (Offset-Bucket.Start))
  };
}

void SuperHExpandPseudo::emitStackAdj(Block &MBB, BlockIt MBBI, Register FrameReg, 
                                      StackAddr Addr) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  const MachineFunction &MF = *MBB.getParent();
  MachineInstr &MI = *MBBI;

  switch(MI.getOpcode()) {
  case SH::MOVF32SF:
  case SH::MOVF32LF: {

    // Expand sequence to
    // mov      <frame reg>,  r1
    // add      (offset),     r1 ! Repeats until offset is reached.
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R1)
      .addReg(FrameReg);

    TII->emitAddressAdjust(SH::R1, MBB, MBBI, Addr.Base + Addr.Offset, INT_MAX);
    break;
  }
  default: {

    // Expand sequence to
    // mov      <frame reg>,  r1
    // add      #-SpOffset,   r1 ! Repeats until offset is reached.
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R1)
      .addReg(FrameReg);

    TII->emitAddressAdjust(SH::R1, MBB, MBBI, Addr.Base, INT_MAX);
    break;
  }
  }
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

  StackAddr Addr = getStackOffset(MF, FrameReg, Offset, 4, Scale);
  emitStackAdj(MBB, MBBI, FrameReg, Addr);

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Expected valid MOV*SF opcode.");
  case SH::MOVBSF: {

    // mov      <src reg>,  r0
    // mov.b    r0, @(offset,r1)
    if (SrcReg != SH::R0)
      BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R0)
          .addReg(SrcReg, getKillRegState(SrcIsKill));
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVBS4))
        .addReg(SH::R1, RegState::Kill)
        .addImm(Addr.Offset);
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
        .addImm(Addr.Offset);
    break;
  }
  case SH::MOVLSF: {

    // mov.l    <src reg>, @(offset,r1)
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVLS4))
        .addReg(SrcReg, getKillRegState(SrcIsKill))
        .addReg(SH::R1, RegState::Kill)
        .addImm(Addr.Offset);
    break;
  }
  case SH::MOVF32SF: {

    // fmov.s   <src reg>, @r1
    BuildMI(MBB, MBBI, DL, TII->get(SH::FMOV_STORE))
        .addReg(SrcReg, RegState::Kill)
        .addReg(SH::R1, RegState::Kill);
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

  auto Offset = MI.getOperand(2).getImm();
  unsigned Opc;

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Expected valid MOV*SP opcode.");
  case SH::MOVBSP: {
    if (Offset == 0)
      Opc = SH::MOVBS;
    else
      Opc = SH::MOVBS4;
    break;
  }
  case SH::MOVWSP: {
    if (Offset == 0)
      Opc = SH::MOVWS;
    else
      Opc = SH::MOVWS4;
    break;
  }
  case SH::MOVLSP: {
    if (Offset == 0)
      Opc = SH::MOVLS;
    else
      Opc = SH::MOVLS4;
    break;
  }
  case SH::MOVF32SP: {
    if (Offset != 0) {
      BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R1)
        .addReg(DstReg);
      TII->emitAddressAdjust(SH::R1, MBB, MBBI, Offset, CFPMaxAdd);
      DstReg = SH::R1;
    }

    BuildMI(MBB, MBBI, DL, TII->get(SH::FMOV_STORE))
      .addReg(SrcReg, getKillRegState(SrcIsKill))
      .addReg(DstReg);
    return eraseMI(MI);
  }
  }

  // TODO: Allow bigger accesses.
  if (Offset == 0) {
    BuildMI(MBB, MBBI, DL, TII->get(Opc))
      .addReg(SrcReg, getKillRegState(SrcIsKill))
      .addReg(DstReg);
  } else {
    BuildMI(MBB, MBBI, DL, TII->get(Opc))
      .addReg(SrcReg, getKillRegState(SrcIsKill))
      .addReg(DstReg)
      .addImm(Offset);
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
bool SuperHExpandPseudo::expand<SH::MOVF32SF>(Block &MBB, BlockIt MBBI) {

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

template <>
bool SuperHExpandPseudo::expand<SH::MOVF32SP>(Block &MBB, BlockIt MBBI) {

  // Store to stack frame
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

  StackAddr Addr = getStackOffset(MF, FrameReg, Offset, 4, Scale);
  emitStackAdj(MBB, MBBI, FrameReg, Addr);

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Expected valid MOV*LF opcode.");
  case SH::MOVBLF: {

    // mov.b    @(offset,r1), r0
    // mov      r0,           <dst reg>
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVBL4))
        .addReg(SH::R1)
        .addImm(Addr.Offset)
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
      .addImm(Addr.Offset);
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
        .addImm(Addr.Offset);
    break;
  }
  case SH::MOVF32LF: {

    // fmov.s   @r1, <dst reg>
    BuildMI(MBB, MBBI, DL, TII->get(SH::FMOV_LOAD), DstReg)
        .addReg(SH::R1);
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

  auto Offset = MI.getOperand(2).getImm();
  unsigned Opc;

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Expected valid MOV*LP opcode.");
  case SH::MOVBLP: {
    if (Offset == 0)
      Opc = SH::MOVBL;
    else
      Opc = SH::MOVBL4;
    break;
  }
  case SH::MOVWLP: {
    if (Offset == 0)
      Opc = SH::MOVWL;
    else
      Opc = SH::MOVWL4;
    break;
  }
  case SH::MOVLLP: {
    if (Offset == 0)
      Opc = SH::MOVLL;
    else
      Opc = SH::MOVLL4;
    break;
  }
  case SH::MOVF32LP: {
    if (Offset != 0) {
      BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R1)
        .addReg(DstReg);
      TII->emitAddressAdjust(SH::R1, MBB, MBBI, Offset, CFPMaxAdd);
      DstReg = SH::R1;
    }

    BuildMI(MBB, MBBI, DL, TII->get(SH::FMOV_LOAD), DstReg)
      .addReg(SrcReg);
    return eraseMI(MI);
  }
  }

  // TODO: Allow bigger accesses.
  if (Offset == 0) {
    BuildMI(MBB, MBBI, DL, TII->get(Opc), DstReg)
      .addReg(SrcReg);
  } else {
    BuildMI(MBB, MBBI, DL, TII->get(Opc), DstReg)
      .addReg(SrcReg)
      .addImm(Offset);
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
bool SuperHExpandPseudo::expand<SH::MOVF32LF>(Block &MBB, BlockIt MBBI) {

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

template <>
bool SuperHExpandPseudo::expand<SH::MOVF32LP>(Block &MBB, BlockIt MBBI) {

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

template <>
bool SuperHExpandPseudo::expand<SH::MOVIF32>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;
  auto DstReg = MI.getOperand(0).getReg();
  auto Src = MI.getOperand(1);

  if (Src.isFPImm()) {
    if (Src.getFPImm()->isZero()) {
      BuildMI(MBB, MBBI, DL, TII->get(SH::FLDI0), DstReg);
      return eraseMI(MI);
    }

    if (Src.getFPImm()->isOne()) {
      BuildMI(MBB, MBBI, DL, TII->get(SH::FLDI1), DstReg);
      return eraseMI(MI);
    }

    llvm_unreachable("Invalid constant FP.");
  } else if (Src.isCPI()) {

    // Load to FPUL, assuming data is fr32, then move
    // to destination register.
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOVLI), SH::R1)
          .addConstantPoolIndex(MI.getOperand(1).getIndex());
    BuildMI(MBB, MBBI, DL, TII->get(SH::LDSFPUL))
          .addReg(SH::R1);
    BuildMI(MBB, MBBI, DL, TII->get(SH::FSTS), DstReg);
    return eraseMI(MI);
  }
  return false;
}




//===----------------------------------------------------------------------===//
//                              Bit Shifting
//===----------------------------------------------------------------------===//

template <>
bool SuperHExpandPseudo::expand<SH::SHLri>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;

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
//                                Conversion
//===----------------------------------------------------------------------===//

template <>
bool SuperHExpandPseudo::expand<SH::F32TOI32>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;

  auto SrcReg = MI.getOperand(1).getReg();
  auto DstReg = MI.getOperand(2).getReg();

  BuildMI(MBB, MBBI, DL, TII->get(SH::LDSFPUL))
    .addReg(SrcReg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::FLOAT32), DstReg);

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::I32TOF32>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  MachineInstr &MI = *MBBI;

  auto SrcReg = MI.getOperand(1).getReg();
  auto DstReg = MI.getOperand(2).getReg();

  BuildMI(MBB, MBBI, DL, TII->get(SH::FTRC32))
    .addReg(SrcReg);
  BuildMI(MBB, MBBI, DL, TII->get(SH::STSFPUL), DstReg);

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
    EXPAND(SH::MOVF32LF);
    EXPAND(SH::MOVF32LP);
    EXPAND(SH::MOVF32SP);
    EXPAND(SH::MOVF32SF);
    EXPAND(SH::MOVIF32);
    EXPAND(SH::SHLri);
    EXPAND(SH::SHRri);
    EXPAND(SH::SRAri);
    EXPAND(SH::SHLrr);
    EXPAND(SH::SHRrr);
    EXPAND(SH::SRArr);
    EXPAND(SH::F32TOI32);
    EXPAND(SH::I32TOF32);
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