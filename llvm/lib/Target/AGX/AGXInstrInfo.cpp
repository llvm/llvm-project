//===-- AGXInstrInfo.cpp - AGX instruction information --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AGX implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "AGXInstrInfo.h"
#include "AGXSubtarget.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "AGXGenInstrInfo.inc"

namespace llvm {
namespace AGX {
struct AGXPseudoToReal {
  unsigned Pseudo;
  unsigned G13Real;
  unsigned Apple9Real;
};

#define GET_AGXPseudoToRealTable_DECL
#define GET_AGXPseudoToRealTable_IMPL
#include "AGXGenSearchableTables.inc"
} // namespace AGX
} // namespace llvm

// Truth table of bitop selecting the first source.
static constexpr unsigned BitopMov = 0b1010;

AGXInstrInfo::AGXInstrInfo(const AGXSubtarget &ST)
    : AGXGenInstrInfo(ST, RI), RI(ST), ST(ST) {}

void AGXInstrInfo::copy32(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MI, const DebugLoc &DL,
                          MCRegister Dst, MCRegister Src, bool KillSrc) const {
  BuildMI(MBB, MI, DL, get(AGX::BITOP_I32), Dst)
      .addImm(BitopMov)
      .addImm(0)
      .addReg(Src, getKillRegState(KillSrc))
      .addImm(0)
      .addImm(0);
}

void AGXInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MI,
                               const DebugLoc &DL, Register DestReg,
                               Register SrcReg, bool KillSrc,
                               bool RenamableDest, bool RenamableSrc) const {
  const MCRegisterInfo &MRI = *ST.getRegisterInfo();
  AGX::RegInfo D = AGX::getRegInfo(DestReg, MRI);
  AGX::RegInfo S = AGX::getRegInfo(SrcReg, MRI);
  unsigned Bits = D.getSizeInBits();
  if (!D.isGPR() || Bits != S.getSizeInBits())
    report_fatal_error("unsupported AGX register copy");

  if (Bits == 16) {
    BuildMI(MBB, MI, DL, get(AGX::BITOP_I16), DestReg)
        .addImm(BitopMov)
        .addImm(0)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addImm(0)
        .addImm(0);
    return;
  }
  if (Bits == 32) {
    copy32(MBB, MI, DL, DestReg, SrcReg, KillSrc);
    return;
  }

  // Register tuples are copied one 32-bit register at a time, in an order
  // that does not clobber source registers before they are read.
  static const unsigned SubRegs[] = {AGX::sub0, AGX::sub1, AGX::sub2,
                                     AGX::sub3};
  unsigned N = Bits / 32;
  bool Backwards = D.HalfIndex > S.HalfIndex && !S.isUniform();
  for (unsigned I = 0; I < N; ++I) {
    unsigned Idx = SubRegs[Backwards ? N - 1 - I : I];
    MCRegister DstSub = RI.getSubReg(DestReg, Idx);
    MCRegister SrcSub = RI.getSubReg(SrcReg, Idx);
    copy32(MBB, MI, DL, DstSub, SrcSub, KillSrc && I == N - 1);
  }
}

static unsigned getStackOpcode(unsigned Bits, bool Store) {
  switch (Bits) {
  case 16:
    return Store ? AGX::STACK_STORE_16 : AGX::STACK_LOAD_16;
  case 32:
    return Store ? AGX::STACK_STORE_32 : AGX::STACK_LOAD_32;
  case 64:
    return Store ? AGX::STACK_STORE_64 : AGX::STACK_LOAD_64;
  case 96:
    return Store ? AGX::STACK_STORE_96 : AGX::STACK_LOAD_96;
  case 128:
    return Store ? AGX::STACK_STORE_128 : AGX::STACK_LOAD_128;
  default:
    report_fatal_error("unsupported AGX spill size");
  }
}

// Memory format and mask for a stack access of the given size.
static std::pair<unsigned, unsigned> getStackFormat(unsigned Bits) {
  if (Bits == 16)
    return {AGX::FMT_I16, 1};
  return {AGX::FMT_I32, (1u << (Bits / 32)) - 1};
}

void AGXInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MI,
                                       Register SrcReg, bool IsKill,
                                       int FrameIndex,
                                       const TargetRegisterClass *RC,
                                       Register VReg,
                                       MachineInstr::MIFlag Flags) const {
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  unsigned Bits = RI.getRegSizeInBits(*RC);
  auto [Fmt, Mask] = getStackFormat(Bits);
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo::getFixedStack(MF, FrameIndex),
      MachineMemOperand::MOStore, MFI.getObjectSize(FrameIndex),
      MFI.getObjectAlign(FrameIndex));
  BuildMI(MBB, MI, DebugLoc(), get(getStackOpcode(Bits, /*Store=*/true)))
      .addImm(0)
      .addImm(Fmt)
      .addImm(Mask)
      .addReg(SrcReg, getKillRegState(IsKill))
      .addFrameIndex(FrameIndex)
      .addMemOperand(MMO)
      .setMIFlags(Flags);
}

void AGXInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        Register DestReg, int FrameIndex,
                                        const TargetRegisterClass *RC,
                                        Register VReg, unsigned SubReg,
                                        MachineInstr::MIFlag Flags) const {
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  unsigned Bits = RI.getRegSizeInBits(*RC);
  auto [Fmt, Mask] = getStackFormat(Bits);
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo::getFixedStack(MF, FrameIndex),
      MachineMemOperand::MOLoad, MFI.getObjectSize(FrameIndex),
      MFI.getObjectAlign(FrameIndex));
  BuildMI(MBB, MI, DebugLoc(), get(getStackOpcode(Bits, /*Store=*/false)),
          DestReg)
      .addImm(0)
      .addImm(Fmt)
      .addImm(Mask)
      .addFrameIndex(FrameIndex)
      .addMemOperand(MMO)
      .setMIFlags(Flags);
}

static const MachineOperand *getNamedOperand(const MachineInstr &MI,
                                             AGX::OpName Name) {
  int Idx = AGX::getNamedOperandIdx(MI.getOpcode(), Name);
  return Idx < 0 ? nullptr : &MI.getOperand(Idx);
}

static bool isStackLoad(unsigned Opc) {
  switch (Opc) {
  case AGX::STACK_LOAD_16:
  case AGX::STACK_LOAD_32:
  case AGX::STACK_LOAD_64:
  case AGX::STACK_LOAD_96:
  case AGX::STACK_LOAD_128:
    return true;
  default:
    return false;
  }
}

static bool isStackStore(unsigned Opc) {
  switch (Opc) {
  case AGX::STACK_STORE_16:
  case AGX::STACK_STORE_32:
  case AGX::STACK_STORE_64:
  case AGX::STACK_STORE_96:
  case AGX::STACK_STORE_128:
    return true;
  default:
    return false;
  }
}

Register AGXInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                           int &FrameIndex) const {
  if (!isStackLoad(MI.getOpcode()))
    return Register();
  const MachineOperand *O = getNamedOperand(MI, AGX::OpName::O);
  if (!O || !O->isFI())
    return Register();
  FrameIndex = O->getIndex();
  return MI.getOperand(0).getReg();
}

Register AGXInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                          int &FrameIndex) const {
  if (!isStackStore(MI.getOpcode()))
    return Register();
  const MachineOperand *O = getNamedOperand(MI, AGX::OpName::O);
  if (!O || !O->isFI())
    return Register();
  FrameIndex = O->getIndex();
  return getNamedOperand(MI, AGX::OpName::R)->getReg();
}

bool AGXInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                 MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
  MachineBasicBlock::iterator I = MBB.getFirstTerminator();
  if (I == MBB.end())
    return false;

  // Execution mask branches update the nesting counter as well as branching:
  // they cannot be removed or reversed.
  if (I->getOpcode() != AGX::AGX_BR)
    return true;
  if (std::next(I) != MBB.end())
    return true;
  TBB = I->getOperand(0).getMBB();
  return false;
}

unsigned AGXInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                    int *BytesRemoved) const {
  MachineBasicBlock::iterator I = MBB.getFirstTerminator();
  if (I == MBB.end() || I->getOpcode() != AGX::AGX_BR)
    return 0;
  if (BytesRemoved)
    *BytesRemoved = getInstSizeInBytes(*I);
  I->eraseFromParent();
  return 1;
}

unsigned AGXInstrInfo::insertBranch(MachineBasicBlock &MBB,
                                    MachineBasicBlock *TBB,
                                    MachineBasicBlock *FBB,
                                    ArrayRef<MachineOperand> Cond,
                                    const DebugLoc &DL,
                                    int *BytesAdded) const {
  assert(TBB && Cond.empty() && !FBB &&
         "AGX only inserts unconditional branches");
  MachineInstr &MI = *BuildMI(&MBB, DL, get(AGX::AGX_BR)).addMBB(TBB);
  if (BytesAdded)
    *BytesAdded = getInstSizeInBytes(MI);
  return 1;
}

bool AGXInstrInfo::isBasicBlockPrologue(const MachineInstr &MI,
                                        Register Reg) const {
  return MI.getOpcode() == AGX::AGX_END_CF ||
         MI.getOpcode() == AGX::SET_NEST;
}

unsigned AGXInstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  case AGX::AGX_IF:
  case AGX::AGX_ELSE:
  case AGX::AGX_LOOP:
    // An execution mask instruction and a jump.
    return 12;
  case AGX::AGX_BR:
    return 12;
  case AGX::AGX_END_CF:
  case AGX::AGX_LOOP_BEGIN:
    return 6;
  default:
    break;
  }
  if (MI.isMetaInstruction())
    return 0;
  int Real = pseudoToMCOpcode(MI.getOpcode());
  if (Real >= 0)
    return get(Real).getSize();
  return MI.getDesc().getSize();
}

int AGXInstrInfo::pseudoToMCOpcode(unsigned Opcode) const {
  const AGX::AGXPseudoToReal *Entry = AGX::getAGXPseudoToReal(Opcode);
  if (!Entry)
    return -1;
  unsigned Real = ST.getEncodingFamily() == AGXSubtarget::Apple9
                      ? Entry->Apple9Real
                      : Entry->G13Real;
  if (Real == AGX::AGX_NO_REAL)
    return -1;
  return Real;
}
