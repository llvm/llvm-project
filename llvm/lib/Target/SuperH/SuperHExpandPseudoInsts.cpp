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
using namespace SH;

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
  const SuperHMachineFunctionInfo *MFI;

  bool expandMBB(Block &MBB);
  bool expandMI(Block &MBB, BlockIt MBBI);
  template <unsigned OP> bool expand(Block &MBB, BlockIt MBBI);

  void emitStackAdj(Block &MBB, BlockIt MBBI, Register &Base, 
                    bool &BaseIsKill, RelativeAddr Addr);

  bool emitPtrLoad(Block &MBB, BlockIt MBBI, MVT Type,
                   Register &Src, Register &Dst, int64_t Offset);

  bool emitPtrStore(Block &MBB, BlockIt MBBI, MVT Type,
                   Register &Src, Register &Dst, int64_t Offset);

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

void SuperHExpandPseudo::emitStackAdj(Block &MBB, BlockIt MBBI, Register &Base, 
                                      bool &BaseIsKill, RelativeAddr Addr) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  const MachineFunction &MF = *MBB.getParent();
  MachineInstr &MI = *MBBI;

  switch(MI.getOpcode()) {
  case SH::MOVF32SF:
  case SH::MOVF32LF: {
    if (Addr.Base + Addr.Offset == 0)
      break;

    // Expand sequence to
    // mov      <frame reg>,  r1
    // add      (offset),     r1 ! Repeats until offset is reached.
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R1)
      .addReg(Base);

    TII->emitAddressAdjust(SH::R1, MBB, MBBI, Addr.Base + Addr.Offset, INT_MAX);
    Base = SH::R1;
    BaseIsKill = true;
    break;
  }
  default: {
    if (Addr.Base == 0)
      break;

    // Expand sequence to
    // mov      <frame reg>,  r1
    // add      #-SpOffset,   r1 ! Repeats until offset is reached.
    BuildMI(MBB, MBBI, DL, TII->get(SH::MOV), SH::R1)
      .addReg(Base);

    TII->emitAddressAdjust(SH::R1, MBB, MBBI, Addr.Base, INT_MAX);
    Base = SH::R1;
    BaseIsKill = true;
    break;
  }
  }
}




//===----------------------------------------------------------------------===//
//                            Pointer Load-Store
//===----------------------------------------------------------------------===//

enum MemAccessType {
  
  // Direct access to memory location
  ACCESS_DIRECT,

  // Access to memory location via 4-bit displacement
  ACCESS_DISP4,

  // Access to memory location via 4-bit displacement
  // Source/Destination is stored in R0
  ACCESS_DISP4_R0,

  // Access to memory location via R0 displacement
  ACCESS_DISP_R0
};

bool SuperHExpandPseudo::emitPtrLoad(Block &MBB, BlockIt MBBI, MVT Type,
                                     Register &Src, Register &Dst, 
                                     int64_t Offset) {
  const DebugLoc &DL = MBBI->getDebugLoc();

  // Range of an indirect load/store.
  int32_t ILRange = (1 << 4) * Type.getScalarStoreSize();

  // Get the best opcode for the given load.
  MemAccessType AType;
  unsigned Opc;
  switch(Type.SimpleTy) {
  default:
    llvm_unreachable("Unexpected load size!");
  case MVT::i8:
    if (Offset == 0) {
      AType = ACCESS_DIRECT;
      Opc = SH::MOVBL;
    } else if (Offset > 0 && Offset < ILRange) {
      AType = ACCESS_DISP4_R0;
      Opc = SH::MOVBL4;
    } else {
      AType = ACCESS_DISP_R0;
      Opc = SH::MOVBL0;
    }
    break;
  case MVT::i16:
    if (Offset == 0) {
      AType = ACCESS_DIRECT;
      Opc = SH::MOVWL;
    } else if (Offset > 0 && Offset < ILRange) {
      AType = ACCESS_DISP4_R0;
      Opc = SH::MOVWL4;
    } else {
      AType = ACCESS_DISP_R0;
      Opc = SH::MOVWL0;
    }
    break;
  case MVT::i32:
    if (Offset == 0) {
      AType = ACCESS_DIRECT;
      Opc = SH::MOVLL;
    } else if (Offset > 0 && Offset < ILRange) {
      AType = ACCESS_DISP4;
      Opc = SH::MOVLL4;
    } else {
      AType = ACCESS_DISP_R0;
      Opc = SH::MOVLL0;
    }
    break;
  case MVT::f32:
    if (Offset == 0) {
      AType = ACCESS_DIRECT;
      Opc = SH::FMOV_LOAD;
    } else {
      AType = ACCESS_DISP_R0;
      Opc = SH::FMOV_INDEX_LOAD;
    }
  }

  // Generate the appropriate instruction sequence.
  switch(AType) {
  case ACCESS_DIRECT:
      BuildMI(MBB, MBBI, DL, TII->get(Opc), Dst)
          .addReg(Src);
      return true;

  case ACCESS_DISP4:
      BuildMI(MBB, MBBI, DL, TII->get(Opc), Dst)
          .addReg(Src)
          .addImm(Offset);
      return true;

  case ACCESS_DISP4_R0:
      BuildMI(MBB, MBBI, DL, TII->get(Opc))
        .addReg(Src)
        .addImm(Offset);
      BuildMI(MBB, MBBI, DL, TII->get(MOV), Dst)
        .addReg(SH::R0);
      return true;

  case ACCESS_DISP_R0:

      // Add displacement to R0
      if (isInt<8>(Offset)) {
        BuildMI(MBB, MBBI, DL, TII->get(SH::MOVI), SH::R0)
          .addImm(Offset);
      } else {
        BuildMI(MBB, MBBI, DL, TII->get(SH::MOVI), SH::R0)
          .addImm(0);
        TII->emitAddressAdjust(SH::R0, MBB, MBBI, Offset, CFPMaxAdd);
      }

      BuildMI(MBB, MBBI, DL, TII->get(Opc), Dst)
          .addReg(Src)
          .addReg(SH::R0)
          .addImm(Offset);
      return true;
  }
  return false;
}

bool SuperHExpandPseudo::emitPtrStore(Block &MBB, BlockIt MBBI, MVT Type,
                                      Register &Src, Register &Dst, 
                                      int64_t Offset) {
  const DebugLoc &DL = MBBI->getDebugLoc();

  // Range of an indirect load/store.
  int32_t ILRange = (1 << 4) * Type.getScalarStoreSize();

  // Get the best opcode for the given load.
  MemAccessType AType;
  unsigned Opc;
  switch(Type.SimpleTy) {
  default:
    llvm_unreachable("Unexpected load size!");
  case MVT::i8:
    if (Offset == 0) {
      AType = ACCESS_DIRECT;
      Opc = SH::MOVBS;
    } else if (Offset > 0 && Offset < ILRange) {
      AType = ACCESS_DISP4_R0;
      Opc = SH::MOVBS4;
    } else {
      AType = ACCESS_DISP_R0;
      Opc = SH::MOVBS0;
    }
    break;
  case MVT::i16:
    if (Offset == 0) {
      AType = ACCESS_DIRECT;
      Opc = SH::MOVWS;
    } else if (Offset > 0 && Offset < ILRange) {
      AType = ACCESS_DISP4_R0;
      Opc = SH::MOVWS4;
    } else {
      AType = ACCESS_DISP_R0;
      Opc = SH::MOVWS0;
    }
    break;
  case MVT::i32:
    if (Offset == 0) {
      AType = ACCESS_DIRECT;
      Opc = SH::MOVLS;
    } else if (Offset > 0 && Offset < ILRange) {
      AType = ACCESS_DISP4;
      Opc = SH::MOVLS4;
    } else {
      AType = ACCESS_DISP_R0;
      Opc = SH::MOVLS0;
    }
    break;
  case MVT::f32:
    if (Offset == 0) {
      AType = ACCESS_DIRECT;
      Opc = SH::FMOV_STORE;
    } else {
      AType = ACCESS_DISP_R0;
      Opc = SH::FMOV_INDEX_STORE;
    }
  }

  // Generate the appropriate instruction sequence.
  switch(AType) {
  case ACCESS_DIRECT:
      BuildMI(MBB, MBBI, DL, TII->get(Opc))
        .addReg(Src)
        .addReg(Dst);
      return true;
      
  case ACCESS_DISP4:
      BuildMI(MBB, MBBI, DL, TII->get(Opc))
        .addReg(Src)
        .addReg(Dst)
        .addImm(Offset);
      return true;

  case ACCESS_DISP4_R0:
      BuildMI(MBB, MBBI, DL, TII->get(MOV), SH::R0)
        .addReg(Src);
      BuildMI(MBB, MBBI, DL, TII->get(Opc))
        .addReg(SH::R0)
        .addReg(Dst)
        .addImm(Offset);
      return true;

  case ACCESS_DISP_R0:

      // Add displacement to R0
      if (isInt<8>(Offset)) {
        BuildMI(MBB, MBBI, DL, TII->get(SH::MOVI), SH::R0)
          .addImm(Offset);
      } else {
        BuildMI(MBB, MBBI, DL, TII->get(SH::MOVI), SH::R0)
          .addImm(0);
        TII->emitAddressAdjust(SH::R0, MBB, MBBI, Offset, CFPMaxAdd);
      }

      BuildMI(MBB, MBBI, DL, TII->get(Opc))
          .addReg(Dst)
          .addReg(SH::R0)
          .addImm(Offset);
      return true;
  }
  return false;
}




//===----------------------------------------------------------------------===//
//                                Stores
//===----------------------------------------------------------------------===//

bool SuperHExpandPseudo::storeToFrame(Block &MBB, BlockIt MBBI, int Scale) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;
  const MachineFunction &MF = *MBB.getParent();

  bool SrcIsKill = MI.getOperand(0).isKill();
  auto SrcReg = MI.getOperand(0).getReg();
  auto FrameReg = MI.getOperand(1).getReg();
  auto Offset = MI.getOperand(2).getImm();

  bool FrameIsKill;
  auto Addr = TRI->getStackAddress(MF, Offset, 4, Scale);
  emitStackAdj(MBB, II, FrameReg, FrameIsKill, Addr);

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Expected valid MOV*SF opcode.");
  case SH::MOVBSF: {
    if (Addr.Offset == 0) {

      // mov.b    <src reg>, @r1
      BuildMI(MBB, II, DL, TII->get(SH::MOVBS))
        .addReg(SrcReg)
        .addReg(FrameReg);
    } else {

      // mov      <src reg>,  r0
      // mov.b    r0, @(offset,r1)
      if (SrcReg != SH::R0)
        BuildMI(MBB, II, DL, TII->get(SH::MOV), SH::R0)
            .addReg(SrcReg);
      BuildMI(MBB, II, DL, TII->get(SH::MOVBS4))
          .addReg(FrameReg)
          .addImm(Addr.Offset);
    }
    break;
  }
  case SH::MOVWSF: {
    if (Addr.Offset == 0) {

      // mov.w    <src reg>, @r1
      BuildMI(MBB, II, DL, TII->get(SH::MOVWS))
        .addReg(SrcReg)
        .addReg(FrameReg);
    } else {

      // mov      <src reg>,  r0
      // mov.w    r0, @(offset,r1)
      if (SrcReg != SH::R0)
        BuildMI(MBB, II, DL, TII->get(SH::MOV), SH::R0)
            .addReg(SrcReg);
      BuildMI(MBB, II, DL, TII->get(SH::MOVWS4))
          .addReg(FrameReg)
          .addImm(Addr.Offset);
    }
    break;
  }
  case SH::MOVLSF: {
    if (Addr.Offset == 0) {

      // mov.l    <src reg>, @r1
      BuildMI(MBB, II, DL, TII->get(SH::MOVLS))
        .addReg(SrcReg)
        .addReg(FrameReg);
    } else {

      // mov.l    <src reg>, @(offset,r1)
      BuildMI(MBB, II, DL, TII->get(SH::MOVLS4))
          .addReg(SrcReg)
          .addReg(FrameReg)
          .addImm(Addr.Offset);
    }
    break;
  }
  case SH::MOVF32SF: {

    // fmov.s   <src reg>, @r1
    BuildMI(MBB, II, DL, TII->get(SH::FMOV_STORE))
        .addReg(SrcReg)
        .addReg(FrameReg);
    break;
  }
  }

  return eraseMI(MI);
}

bool SuperHExpandPseudo::storeToAddress(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;
  const MachineFunction &MF = *MBB.getParent();

  // bool SrcIsKill = MI.getOperand(0).isKill();
  Register SrcReg = MI.getOperand(0).getReg();
  Register DstReg;
  auto Offset = MI.getOperand(2).getImm();

  if (MI.getOperand(1).isGlobal()) {
    if (auto *G = MFI->tryGetConstant(MI.getOperand(1).getGlobal(), MF)) {
      BuildMI(MBB, II, DL, TII->get(SH::MOVLI), SH::R1)
          .addConstantPoolIndex(G->getLabelId());
      DstReg = SH::R1;
    }
  } else if (MI.getOperand(1).isReg()) {
    DstReg = MI.getOperand(1).getReg();
  }

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Expected valid MOV*SP opcode.");
  case SH::MOVBSP:
    emitPtrStore(MBB, II, MVT::i8, SrcReg, DstReg, Offset);
    break;
  case SH::MOVWSP:
    emitPtrStore(MBB, II, MVT::i16, SrcReg, DstReg, Offset);
    break;
  case SH::MOVLSP:
    emitPtrStore(MBB, II, MVT::i32, SrcReg, DstReg, Offset);
    break;
  case SH::MOVF32SP:
    emitPtrStore(MBB, II, MVT::f32, SrcReg, DstReg, Offset);
    break;
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
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;
  const MachineFunction &MF = *MBB.getParent();

  auto DstReg = MI.getOperand(0).getReg();
  bool DstIsKill = MI.getOperand(0).isKill();
  auto FrameReg = MI.getOperand(1).getReg();
  auto Offset = MI.getOperand(2).getImm();

  bool FrameIsKill;
  auto Addr = TRI->getStackAddress(MF, Offset, 4, Scale);
  emitStackAdj(MBB, II, FrameReg, FrameIsKill, Addr);

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Expected valid MOV*LF opcode.");
  case SH::MOVBLF: {
    if (Addr.Offset == 0) {

      // mov.b  r1, <dst reg>
      BuildMI(MBB, II, DL, TII->get(SH::MOVBL), DstReg)
          .addReg(FrameReg);
    } else {

      // mov.b    @(offset,r1), r0
      // mov      r0,           <dst reg>
      BuildMI(MBB, II, DL, TII->get(SH::MOVBL4))
          .addReg(FrameReg)
          .addImm(Addr.Offset)
          .addReg(SH::R0);
      BuildMI(MBB, II, DL, TII->get(SH::MOV))
          .addReg(DstReg)
          .addReg(SH::R0);
    }
    break;
  }
  case SH::MOVWLF: {
    if (Addr.Offset == 0) {

      // mov.w  r1, <dst reg>
      BuildMI(MBB, II, DL, TII->get(SH::MOVWL), DstReg)
          .addReg(FrameReg);
    } else {

      // mov.w    @(offset,r1), r0
      // mov      r0,           <dst reg>
      BuildMI(MBB, II, DL, TII->get(SH::MOVWL4))
        .addReg(FrameReg)
        .addImm(Addr.Offset);
      BuildMI(MBB, II, DL, TII->get(SH::MOV))
          .addReg(DstReg)
          .addReg(SH::R0);
    }
    break;
  }
  case SH::MOVLLF: {
    if (Addr.Offset == 0) {

      // mov.l  r1, <dst reg>
      BuildMI(MBB, II, DL, TII->get(SH::MOVLL), DstReg)
          .addReg(FrameReg);
    } else {

      // mov.l    @(offset,r1), <dst reg>
      BuildMI(MBB, II, DL, TII->get(SH::MOVLL4))
          .addReg(DstReg)
          .addReg(FrameReg)
          .addImm(Addr.Offset);
    }
    break;
  }
  case SH::MOVF32LF: {

    // fmov.s   @r1, <dst reg>
    BuildMI(MBB, II, DL, TII->get(SH::FMOV_LOAD), DstReg)
        .addReg(FrameReg);
    break;
  }
  }

  return eraseMI(MI);
}

bool SuperHExpandPseudo::loadFromAddress(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;
  const MachineFunction &MF = *MBB.getParent();
  const SuperHMachineFunctionInfo *FI = MF.getInfo<SuperHMachineFunctionInfo>();

  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg;
  auto Offset = MI.getOperand(2).getImm();

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
    llvm_unreachable("Expected valid MOV*LP opcode.");
  case SH::MOVBLP:
    emitPtrLoad(MBB, II, MVT::i8, SrcReg, DstReg, Offset);
    break;
  case SH::MOVWLP:
    emitPtrLoad(MBB, II, MVT::i16, SrcReg, DstReg, Offset);
    break;
  case SH::MOVLLP:
    emitPtrLoad(MBB, II, MVT::i32, SrcReg, DstReg, Offset);
    break;
  case SH::MOVF32LP:
    emitPtrLoad(MBB, II, MVT::f32, SrcReg, DstReg, Offset);
    break;
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
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;
  auto DstReg = MI.getOperand(0).getReg();

  if (MI.getOperand(1).isImm()) {
    BuildMI(MBB, II, DL, TII->get(SH::MOVI), DstReg)
        .addImm(MI.getOperand(1).getImm());
  } else if (MI.getOperand(1).isCPI()) {
    BuildMI(MBB, II, DL, TII->get(SH::MOVLI), DstReg)
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
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;
  auto DstReg = MI.getOperand(0).getReg();
  auto Src = MI.getOperand(1);

  if (Src.isFPImm()) {
    if (Src.getFPImm()->isZero()) {
      BuildMI(MBB, II, DL, TII->get(SH::FLDI0), DstReg);
      return eraseMI(MI);
    }

    if (Src.getFPImm()->isOne()) {
      BuildMI(MBB, II, DL, TII->get(SH::FLDI1), DstReg);
      return eraseMI(MI);
    }

    llvm_unreachable("Invalid constant FP.");
  } else if (Src.isCPI()) {

    // Load to FPUL, assuming data is fr32, then move
    // to destination register.
    BuildMI(MBB, II, DL, TII->get(SH::MOVLI), SH::R1)
      .addConstantPoolIndex(MI.getOperand(1).getIndex());
    BuildMI(MBB, II, DL, TII->get(SH::LDSFPUL))
      .addReg(SH::R1);
    BuildMI(MBB, II, DL, TII->get(SH::FSTS), DstReg);
    return eraseMI(MI);
  }
  return false;
}

//===----------------------------------------------------------------------===//
//                              Bit Manipulation
//===----------------------------------------------------------------------===//

// Optimized pattern for "value not", equvalent to "!value" in C
template <>
bool SuperHExpandPseudo::expand<SH::LNOT>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;

  auto DstReg = MI.getOperand(0).getReg();
  auto SrcReg = MI.getOperand(1).getReg();

  BuildMI(MBB, II, DL, TII->get(SH::TST))
    .addReg(SrcReg)
    .addReg(SrcReg);
  BuildMI(MBB, II, DL, TII->get(SH::MOVT), DstReg);

  return eraseMI(MI);
}



//===----------------------------------------------------------------------===//
//                                Bit Shifting
//===----------------------------------------------------------------------===//

template <>
bool SuperHExpandPseudo::expand<SH::SHLri>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;

  auto DstReg = MI.getOperand(0).getReg();
  int64_t Offset = MI.getOperand(1).getImm();

  if (STI->isSH3() && Offset > 32) {
    BuildMI(MBB, II, DL, TII->get(SH::MOVI), SH::R0)
      .addImm(Offset);
    BuildMI(MBB, II, DL, TII->get(SH::SHAD), DstReg)
      .addReg(SH::R0);
  } else {
    while (Offset > 0) {
      if (Offset >= 16) {
        BuildMI(MBB, II, DL, TII->get(SH::SHLL16), DstReg)
          .addReg(DstReg);

        Offset -= 16;
        continue;
      }

      if (Offset >= 8) {
        BuildMI(MBB, II, DL, TII->get(SH::SHLL8), DstReg)
          .addReg(DstReg);

        Offset -= 8;
        continue;
      }

      if (Offset >= 2) {
        BuildMI(MBB, II, DL, TII->get(SH::SHLL2), DstReg)
          .addReg(DstReg);

        Offset -= 2;
        continue;
      }

      BuildMI(MBB, II, DL, TII->get(SH::SHLL), DstReg)
        .addReg(DstReg);
      Offset -= 1;
      continue;
    }
  }

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::SHRri>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;

  auto DstReg = MI.getOperand(0).getReg();
  int64_t Offset = MI.getOperand(1).getImm();

  if (STI->isSH3() && Offset > 32) {
    BuildMI(MBB, II, DL, TII->get(SH::MOVI), SH::R0)
      .addImm(-Offset);
    BuildMI(MBB, II, DL, TII->get(SH::SHLD), DstReg)
      .addReg(SH::R0);
  } else {
    while (Offset > 0) {

      if (Offset >= 16) {
        BuildMI(MBB, II, DL, TII->get(SH::SHLR16), DstReg)
          .addReg(DstReg);

        Offset -= 16;
        continue;
      }

      if (Offset >= 8) {
        BuildMI(MBB, II, DL, TII->get(SH::SHLR8), DstReg)
          .addReg(DstReg);

        Offset -= 8;
        continue;
      }

      if (Offset >= 2) {
        BuildMI(MBB, II, DL, TII->get(SH::SHLR2), DstReg)
          .addReg(DstReg);

        Offset -= 2;
        continue;
      }

      BuildMI(MBB, II, DL, TII->get(SH::SHLR), DstReg)
          .addReg(DstReg);
      Offset -= 1;
      continue;
    }
  }

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::SRAri>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;

  auto DstReg = MI.getOperand(0).getReg();
  int64_t Offset = MI.getOperand(1).getImm();
  auto SrcReg = MI.getOperand(2).getReg();

  if (STI->isSH3()) {
    BuildMI(MBB, II, DL, TII->get(SH::MOVI), SH::R0)
      .addImm(-Offset);
    BuildMI(MBB, II, DL, TII->get(SH::SHAD), DstReg)
      .addReg(SH::R0)
      .addReg(SrcReg);
  } else {
    while (Offset > 0) {
      BuildMI(MBB, II, DL, TII->get(SH::SHAR), DstReg)
        .addReg(SrcReg);
      Offset -= 1;
    }
  }

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::RTLri>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;

  auto DstReg = MI.getOperand(0).getReg();
  int64_t Offset = MI.getOperand(1).getImm();
  auto SrcReg = MI.getOperand(2).getReg();

  while (Offset > 0) {
    BuildMI(MBB, II, DL, TII->get(SH::ROTL), DstReg)
      .addReg(SrcReg);
    Offset -= 1;
  }

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::RTRri>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;

  auto DstReg = MI.getOperand(0).getReg();
  int64_t Offset = MI.getOperand(1).getImm();
  auto SrcReg = MI.getOperand(2).getReg();

  while (Offset > 0) {
    BuildMI(MBB, II, DL, TII->get(SH::ROTR), DstReg)
      .addReg(SrcReg);
    Offset -= 1;
  }

  return eraseMI(MI);
}

//===----------------------------------------------------------------------===//
// NOTE:  The register-register shifts utilize a small instruction sequence
//        which runs a tight shifting loop utilizing the dt instruction to
//        decrement the "counter" every iteration. All label branches
//        in SuperH are relative to PC + 4, as such the offset for the loop
//        is -8, this is intentional, do not change it, otherwise the check
//        will jump to itself in an infinite loop.
//===----------------------------------------------------------------------===//

template <>
bool SuperHExpandPseudo::expand<SH::SHLrr>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;

  auto Src1Reg = MI.getOperand(1).getReg();
  auto Src2Reg = MI.getOperand(2).getReg();

  if (STI->isSH3()) {
    BuildMI(MBB, II, DL, TII->get(SH::SHAD), Src2Reg)
      .addReg(Src1Reg);
  } else {

    // If a shift by 0 was requested, do nothing.
    BuildMI(MBB, II, DL, TII->get(SH::TST))
      .addReg(Src1Reg)
      .addReg(Src1Reg);
    BuildMI(MBB, II, DL, TII->get(SH::BT))
      .addImm(2);

    // Shifting loop.
    BuildMI(MBB, II, DL, TII->get(SH::SHLL))
      .addReg(Src2Reg);
    BuildMI(MBB, II, DL, TII->get(SH::DT))
      .addReg(Src1Reg);
    BuildMI(MBB, II, DL, TII->get(SH::BF))
      .addImm(-8);
  }

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::SHRrr>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;

  auto Src1Reg = MI.getOperand(1).getReg();
  auto Src2Reg = MI.getOperand(2).getReg();

  if (STI->isSH3()) {
    BuildMI(MBB, II, DL, TII->get(SH::NEG), Src1Reg)
      .addReg(Src1Reg);
    BuildMI(MBB, II, DL, TII->get(SH::SHLD), Src2Reg)
      .addReg(Src1Reg);
  } else {

    // If a shift by 0 was requested, do nothing.
    BuildMI(MBB, II, DL, TII->get(SH::TST))
      .addReg(Src1Reg)
      .addReg(Src1Reg);
    BuildMI(MBB, II, DL, TII->get(SH::BT))
      .addImm(2);

    // Shifting loop.
    BuildMI(MBB, II, DL, TII->get(SH::SHLR))
      .addReg(Src2Reg);
    BuildMI(MBB, II, DL, TII->get(SH::DT))
      .addReg(Src1Reg);
    BuildMI(MBB, II, DL, TII->get(SH::BF))
      .addImm(-8);
  }

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::SRArr>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;

  auto Src1Reg = MI.getOperand(1).getReg();
  auto Src2Reg = MI.getOperand(2).getReg();

  if (STI->isSH3()) {
    BuildMI(MBB, II, DL, TII->get(SH::NEG), Src1Reg)
      .addReg(Src1Reg);
    BuildMI(MBB, II, DL, TII->get(SH::SHAD), Src2Reg)
      .addReg(Src1Reg);
  } else {

    // If a shift by 0 was requested, do nothing.
    BuildMI(MBB, II, DL, TII->get(SH::TST))
      .addReg(Src1Reg)
      .addReg(Src1Reg);
    BuildMI(MBB, II, DL, TII->get(SH::BT))
      .addImm(2);

    // Shifting loop.
    BuildMI(MBB, II, DL, TII->get(SH::SHAR))
      .addReg(Src2Reg);
    BuildMI(MBB, II, DL, TII->get(SH::DT))
      .addReg(Src1Reg);
    BuildMI(MBB, II, DL, TII->get(SH::BF))
      .addImm(-8);
  }

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::RTLrr>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;

  auto Src1Reg = MI.getOperand(1).getReg();
  auto Src2Reg = MI.getOperand(2).getReg();

  // If a shift by 0 was requested, do nothing.
  BuildMI(MBB, II, DL, TII->get(SH::TST))
    .addReg(Src1Reg)
    .addReg(Src1Reg);
  BuildMI(MBB, II, DL, TII->get(SH::BT))
    .addImm(2);

  // Shifting loop.
  BuildMI(MBB, II, DL, TII->get(SH::ROTL))
    .addReg(Src2Reg);
  BuildMI(MBB, II, DL, TII->get(SH::DT))
    .addReg(Src1Reg);
  BuildMI(MBB, II, DL, TII->get(SH::BF))
    .addImm(-8);

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::RTRrr>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;

  auto Src1Reg = MI.getOperand(1).getReg();
  auto Src2Reg = MI.getOperand(2).getReg();

  // If a shift by 0 was requested, do nothing.
  BuildMI(MBB, II, DL, TII->get(SH::TST))
    .addReg(Src1Reg)
    .addReg(Src1Reg);
  BuildMI(MBB, II, DL, TII->get(SH::BT))
    .addImm(2);

  // Shifting loop.
  BuildMI(MBB, II, DL, TII->get(SH::ROTR))
    .addReg(Src2Reg);
  BuildMI(MBB, II, DL, TII->get(SH::DT))
    .addReg(Src1Reg);
  BuildMI(MBB, II, DL, TII->get(SH::BF))
    .addImm(-8);

  return eraseMI(MI);
}




//===----------------------------------------------------------------------===//
//                                Conversion
//===----------------------------------------------------------------------===//

template <>
bool SuperHExpandPseudo::expand<SH::FTOI>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;

  auto SrcReg = MI.getOperand(0).getReg();
  auto DstReg = MI.getOperand(1).getReg();

  BuildMI(MBB, II, DL, TII->get(SH::LDSFPUL))
    .addReg(SrcReg);
  BuildMI(MBB, II, DL, TII->get(SH::FLOAT32), DstReg);

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::ITOF>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;

  auto SrcReg = MI.getOperand(0).getReg();
  auto DstReg = MI.getOperand(1).getReg();

  BuildMI(MBB, II, DL, TII->get(SH::FTRC32))
    .addReg(SrcReg);
  BuildMI(MBB, II, DL, TII->get(SH::STSFPUL), DstReg);

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::FTOIBITS>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;

  auto SrcReg = MI.getOperand(0).getReg();
  auto DstReg = MI.getOperand(1).getReg();

  BuildMI(MBB, II, DL, TII->get(SH::LDSFPUL))
    .addReg(SrcReg);
  BuildMI(MBB, II, DL, TII->get(SH::FSTS), DstReg);

  return eraseMI(MI);
}

template <>
bool SuperHExpandPseudo::expand<SH::ITOFBITS>(Block &MBB, BlockIt MBBI) {
  const DebugLoc &DL = MBBI->getDebugLoc();
  BlockIt II = std::next(MBBI);
  MachineInstr &MI = *MBBI;

  auto SrcReg = MI.getOperand(0).getReg();
  auto DstReg = MI.getOperand(1).getReg();

  BuildMI(MBB, II, DL, TII->get(SH::FLDS))
    .addReg(SrcReg);
  BuildMI(MBB, II, DL, TII->get(SH::STSFPUL), DstReg);

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
  MFI = MF.getInfo<SuperHMachineFunctionInfo>();

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
    EXPAND(SH::LNOT);
    EXPAND(SH::SHLri);
    EXPAND(SH::SHRri);
    EXPAND(SH::SRAri);
    EXPAND(SH::RTLri);
    EXPAND(SH::RTRri);
    EXPAND(SH::SHLrr);
    EXPAND(SH::SHRrr);
    EXPAND(SH::SRArr);
    EXPAND(SH::RTLrr);
    EXPAND(SH::RTRrr);
    EXPAND(SH::FTOI);
    EXPAND(SH::ITOF);
    EXPAND(SH::FTOIBITS);
    EXPAND(SH::ITOFBITS);
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