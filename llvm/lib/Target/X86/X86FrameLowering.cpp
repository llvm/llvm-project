//=======- X86FrameLowering.cpp - X86 Frame Information --------*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "X86FrameLowering.h"
#include "X86InstrBuilder.h"
#include "X86InstrInfo.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallSet.h"

using namespace llvm;

// FIXME: completely move here.
extern cl::opt<bool> ForceStackAlign;

bool X86FrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  return !MF.getFrameInfo()->hasVarSizedObjects();
}

/// hasFP - Return true if the specified function should have a dedicated frame
/// pointer register.  This is true if the function has variable sized allocas
/// or if frame pointer elimination is disabled.
bool X86FrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const MachineModuleInfo &MMI = MF.getMMI();
  const TargetRegisterInfo *RI = TM.getRegisterInfo();

  return (DisableFramePointerElim(MF) ||
          RI->needsStackRealignment(MF) ||
          MFI->hasVarSizedObjects() ||
          MFI->isFrameAddressTaken() ||
          MF.getInfo<X86MachineFunctionInfo>()->getForceFramePointer() ||
          MMI.callsUnwindInit());
}

static unsigned getSUBriOpcode(unsigned is64Bit, int64_t Imm) {
  if (is64Bit) {
    if (isInt<8>(Imm))
      return X86::SUB64ri8;
    return X86::SUB64ri32;
  } else {
    if (isInt<8>(Imm))
      return X86::SUB32ri8;
    return X86::SUB32ri;
  }
}

static unsigned getADDriOpcode(unsigned is64Bit, int64_t Imm) {
  if (is64Bit) {
    if (isInt<8>(Imm))
      return X86::ADD64ri8;
    return X86::ADD64ri32;
  } else {
    if (isInt<8>(Imm))
      return X86::ADD32ri8;
    return X86::ADD32ri;
  }
}

/// findDeadCallerSavedReg - Return a caller-saved register that isn't live
/// when it reaches the "return" instruction. We can then pop a stack object
/// to this register without worry about clobbering it.
static unsigned findDeadCallerSavedReg(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator &MBBI,
                                       const TargetRegisterInfo &TRI,
                                       bool Is64Bit) {
  const MachineFunction *MF = MBB.getParent();
  const Function *F = MF->getFunction();
  if (!F || MF->getMMI().callsEHReturn())
    return 0;

  static const unsigned CallerSavedRegs32Bit[] = {
    X86::EAX, X86::EDX, X86::ECX, 0
  };

  static const unsigned CallerSavedRegs64Bit[] = {
    X86::RAX, X86::RDX, X86::RCX, X86::RSI, X86::RDI,
    X86::R8,  X86::R9,  X86::R10, X86::R11, 0
  };

  unsigned Opc = MBBI->getOpcode();
  switch (Opc) {
  default: return 0;
  case X86::RET:
  case X86::RETI:
  case X86::TCRETURNdi:
  case X86::TCRETURNri:
  case X86::TCRETURNmi:
  case X86::TCRETURNdi64:
  case X86::TCRETURNri64:
  case X86::TCRETURNmi64:
  case X86::EH_RETURN:
  case X86::EH_RETURN64: {
    SmallSet<unsigned, 8> Uses;
    for (unsigned i = 0, e = MBBI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MBBI->getOperand(i);
      if (!MO.isReg() || MO.isDef())
        continue;
      unsigned Reg = MO.getReg();
      if (!Reg)
        continue;
      for (const unsigned *AsI = TRI.getOverlaps(Reg); *AsI; ++AsI)
        Uses.insert(*AsI);
    }

    const unsigned *CS = Is64Bit ? CallerSavedRegs64Bit : CallerSavedRegs32Bit;
    for (; *CS; ++CS)
      if (!Uses.count(*CS))
        return *CS;
  }
  }

  return 0;
}


/// emitSPUpdate - Emit a series of instructions to increment / decrement the
/// stack pointer by a constant value.
static
void emitSPUpdate(MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
                  unsigned StackPtr, int64_t NumBytes,
                  bool Is64Bit, const TargetInstrInfo &TII,
                  const TargetRegisterInfo &TRI) {
  bool isSub = NumBytes < 0;
  uint64_t Offset = isSub ? -NumBytes : NumBytes;
  unsigned Opc = isSub ?
    getSUBriOpcode(Is64Bit, Offset) :
    getADDriOpcode(Is64Bit, Offset);
  uint64_t Chunk = (1LL << 31) - 1;
  DebugLoc DL = MBB.findDebugLoc(MBBI);

  while (Offset) {
    uint64_t ThisVal = (Offset > Chunk) ? Chunk : Offset;
    if (ThisVal == (Is64Bit ? 8 : 4)) {
      // Use push / pop instead.
      unsigned Reg = isSub
        ? (unsigned)(Is64Bit ? X86::RAX : X86::EAX)
        : findDeadCallerSavedReg(MBB, MBBI, TRI, Is64Bit);
      if (Reg) {
        Opc = isSub
          ? (Is64Bit ? X86::PUSH64r : X86::PUSH32r)
          : (Is64Bit ? X86::POP64r  : X86::POP32r);
        MachineInstr *MI = BuildMI(MBB, MBBI, DL, TII.get(Opc))
          .addReg(Reg, getDefRegState(!isSub) | getUndefRegState(isSub));
        if (isSub)
          MI->setFlag(MachineInstr::FrameSetup);
        Offset -= ThisVal;
        continue;
      }
    }

    MachineInstr *MI =
      BuildMI(MBB, MBBI, DL, TII.get(Opc), StackPtr)
      .addReg(StackPtr)
      .addImm(ThisVal);
    if (isSub)
      MI->setFlag(MachineInstr::FrameSetup);
    MI->getOperand(3).setIsDead(); // The EFLAGS implicit def is dead.
    Offset -= ThisVal;
  }
}

/// mergeSPUpdatesUp - Merge two stack-manipulating instructions upper iterator.
static
void mergeSPUpdatesUp(MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
                      unsigned StackPtr, uint64_t *NumBytes = NULL) {
  if (MBBI == MBB.begin()) return;

  MachineBasicBlock::iterator PI = prior(MBBI);
  unsigned Opc = PI->getOpcode();
  if ((Opc == X86::ADD64ri32 || Opc == X86::ADD64ri8 ||
       Opc == X86::ADD32ri || Opc == X86::ADD32ri8) &&
      PI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes += PI->getOperand(2).getImm();
    MBB.erase(PI);
  } else if ((Opc == X86::SUB64ri32 || Opc == X86::SUB64ri8 ||
              Opc == X86::SUB32ri || Opc == X86::SUB32ri8) &&
             PI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes -= PI->getOperand(2).getImm();
    MBB.erase(PI);
  }
}

/// mergeSPUpdatesDown - Merge two stack-manipulating instructions lower iterator.
static
void mergeSPUpdatesDown(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator &MBBI,
                        unsigned StackPtr, uint64_t *NumBytes = NULL) {
  // FIXME: THIS ISN'T RUN!!!
  return;

  if (MBBI == MBB.end()) return;

  MachineBasicBlock::iterator NI = llvm::next(MBBI);
  if (NI == MBB.end()) return;

  unsigned Opc = NI->getOpcode();
  if ((Opc == X86::ADD64ri32 || Opc == X86::ADD64ri8 ||
       Opc == X86::ADD32ri || Opc == X86::ADD32ri8) &&
      NI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes -= NI->getOperand(2).getImm();
    MBB.erase(NI);
    MBBI = NI;
  } else if ((Opc == X86::SUB64ri32 || Opc == X86::SUB64ri8 ||
              Opc == X86::SUB32ri || Opc == X86::SUB32ri8) &&
             NI->getOperand(0).getReg() == StackPtr) {
    if (NumBytes)
      *NumBytes += NI->getOperand(2).getImm();
    MBB.erase(NI);
    MBBI = NI;
  }
}

/// mergeSPUpdates - Checks the instruction before/after the passed
/// instruction. If it is an ADD/SUB instruction it is deleted argument and the
/// stack adjustment is returned as a positive value for ADD and a negative for
/// SUB.
static int mergeSPUpdates(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator &MBBI,
                           unsigned StackPtr,
                           bool doMergeWithPrevious) {
  if ((doMergeWithPrevious && MBBI == MBB.begin()) ||
      (!doMergeWithPrevious && MBBI == MBB.end()))
    return 0;

  MachineBasicBlock::iterator PI = doMergeWithPrevious ? prior(MBBI) : MBBI;
  MachineBasicBlock::iterator NI = doMergeWithPrevious ? 0 : llvm::next(MBBI);
  unsigned Opc = PI->getOpcode();
  int Offset = 0;

  if ((Opc == X86::ADD64ri32 || Opc == X86::ADD64ri8 ||
       Opc == X86::ADD32ri || Opc == X86::ADD32ri8) &&
      PI->getOperand(0).getReg() == StackPtr){
    Offset += PI->getOperand(2).getImm();
    MBB.erase(PI);
    if (!doMergeWithPrevious) MBBI = NI;
  } else if ((Opc == X86::SUB64ri32 || Opc == X86::SUB64ri8 ||
              Opc == X86::SUB32ri || Opc == X86::SUB32ri8) &&
             PI->getOperand(0).getReg() == StackPtr) {
    Offset -= PI->getOperand(2).getImm();
    MBB.erase(PI);
    if (!doMergeWithPrevious) MBBI = NI;
  }

  return Offset;
}

static bool isEAXLiveIn(MachineFunction &MF) {
  for (MachineRegisterInfo::livein_iterator II = MF.getRegInfo().livein_begin(),
       EE = MF.getRegInfo().livein_end(); II != EE; ++II) {
    unsigned Reg = II->first;

    if (Reg == X86::EAX || Reg == X86::AX ||
        Reg == X86::AH || Reg == X86::AL)
      return true;
  }

  return false;
}

void X86FrameLowering::emitCalleeSavedFrameMoves(MachineFunction &MF,
                                                 MCSymbol *Label,
                                                 unsigned FramePtr) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineModuleInfo &MMI = MF.getMMI();

  // Add callee saved registers to move list.
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  if (CSI.empty()) return;

  std::vector<MachineMove> &Moves = MMI.getFrameMoves();
  const TargetData *TD = TM.getTargetData();
  bool HasFP = hasFP(MF);

  // Calculate amount of bytes used for return address storing.
  int stackGrowth = -TD->getPointerSize();

  // FIXME: This is dirty hack. The code itself is pretty mess right now.
  // It should be rewritten from scratch and generalized sometimes.

  // Determine maximum offset (minimum due to stack growth).
  int64_t MaxOffset = 0;
  for (std::vector<CalleeSavedInfo>::const_iterator
         I = CSI.begin(), E = CSI.end(); I != E; ++I)
    MaxOffset = std::min(MaxOffset,
                         MFI->getObjectOffset(I->getFrameIdx()));

  // Calculate offsets.
  int64_t saveAreaOffset = (HasFP ? 3 : 2) * stackGrowth;
  for (std::vector<CalleeSavedInfo>::const_iterator
         I = CSI.begin(), E = CSI.end(); I != E; ++I) {
    int64_t Offset = MFI->getObjectOffset(I->getFrameIdx());
    unsigned Reg = I->getReg();
    Offset = MaxOffset - Offset + saveAreaOffset;

    // Don't output a new machine move if we're re-saving the frame
    // pointer. This happens when the PrologEpilogInserter has inserted an extra
    // "PUSH" of the frame pointer -- the "emitPrologue" method automatically
    // generates one when frame pointers are used. If we generate a "machine
    // move" for this extra "PUSH", the linker will lose track of the fact that
    // the frame pointer should have the value of the first "PUSH" when it's
    // trying to unwind.
    //
    // FIXME: This looks inelegant. It's possibly correct, but it's covering up
    //        another bug. I.e., one where we generate a prolog like this:
    //
    //          pushl  %ebp
    //          movl   %esp, %ebp
    //          pushl  %ebp
    //          pushl  %esi
    //           ...
    //
    //        The immediate re-push of EBP is unnecessary. At the least, it's an
    //        optimization bug. EBP can be used as a scratch register in certain
    //        cases, but probably not when we have a frame pointer.
    if (HasFP && FramePtr == Reg)
      continue;

    MachineLocation CSDst(MachineLocation::VirtualFP, Offset);
    MachineLocation CSSrc(Reg);
    Moves.push_back(MachineMove(Label, CSDst, CSSrc));
  }
}

/// getCompactUnwindRegNum - Get the compact unwind number for a given
/// register. The number corresponds to the enum lists in
/// compact_unwind_encoding.h.
static int getCompactUnwindRegNum(const unsigned *CURegs, unsigned Reg) {
  int Idx = 1;
  for (; *CURegs; ++CURegs, ++Idx)
    if (*CURegs == Reg)
      return Idx;

  return -1;
}

/// encodeCompactUnwindRegistersWithoutFrame - Create the permutation encoding
/// used with frameless stacks. It is passed the number of registers to be saved
/// and an array of the registers saved.
static uint32_t encodeCompactUnwindRegistersWithoutFrame(unsigned SavedRegs[6],
                                                         unsigned RegCount,
                                                         bool Is64Bit) {
  // The saved registers are numbered from 1 to 6. In order to encode the order
  // in which they were saved, we re-number them according to their place in the
  // register order. The re-numbering is relative to the last re-numbered
  // register. E.g., if we have registers {6, 2, 4, 5} saved in that order:
  //
  //    Orig  Re-Num
  //    ----  ------
  //     6       6
  //     2       2
  //     4       3
  //     5       3
  //
  static const unsigned CU32BitRegs[] = {
    X86::EBX, X86::ECX, X86::EDX, X86::EDI, X86::ESI, X86::EBP, 0
  };
  static const unsigned CU64BitRegs[] = {
    X86::RBX, X86::R12, X86::R13, X86::R14, X86::R15, X86::RBP, 0
  };
  const unsigned *CURegs = (Is64Bit ? CU64BitRegs : CU32BitRegs);

  uint32_t RenumRegs[6];
  for (unsigned i = 6 - RegCount; i < 6; ++i) {
    int CUReg = getCompactUnwindRegNum(CURegs, SavedRegs[i]);
    if (CUReg == -1) return ~0U;
    SavedRegs[i] = CUReg;

    unsigned Countless = 0;
    for (unsigned j = 6 - RegCount; j < i; ++j)
      if (SavedRegs[j] < SavedRegs[i])
        ++Countless;

    RenumRegs[i] = SavedRegs[i] - Countless - 1;
  }

  // Take the renumbered values and encode them into a 10-bit number.
  uint32_t permutationEncoding = 0;
  switch (RegCount) {
  case 6:
    permutationEncoding |= 120 * RenumRegs[0] + 24 * RenumRegs[1]
                           + 6 * RenumRegs[2] +  2 * RenumRegs[3]
                           +     RenumRegs[4];
    break;
  case 5:
    permutationEncoding |= 120 * RenumRegs[1] + 24 * RenumRegs[2]
                           + 6 * RenumRegs[3] +  2 * RenumRegs[4]
                           +     RenumRegs[5];
    break;
  case 4:
    permutationEncoding |=  60 * RenumRegs[2] + 12 * RenumRegs[3]
                           + 3 * RenumRegs[4] +      RenumRegs[5];
    break;
  case 3:
    permutationEncoding |=  20 * RenumRegs[3] +  4 * RenumRegs[4]
                           +     RenumRegs[5];
    break;
  case 2:
    permutationEncoding |=   5 * RenumRegs[4] +      RenumRegs[5];
    break;
  case 1:
    permutationEncoding |=       RenumRegs[5];
    break;
  }

  assert((permutationEncoding & 0x3FF) == permutationEncoding &&
         "Invalid compact register encoding!");
  return permutationEncoding;
}

/// encodeCompactUnwindRegistersWithFrame - Return the registers encoded for a
/// compact encoding with a frame pointer.
static uint32_t encodeCompactUnwindRegistersWithFrame(unsigned SavedRegs[6],
                                                      bool Is64Bit) {
  static const unsigned CU32BitRegs[] = {
    X86::EBX, X86::ECX, X86::EDX, X86::EDI, X86::ESI, X86::EBP, 0
  };
  static const unsigned CU64BitRegs[] = {
    X86::RBX, X86::R12, X86::R13, X86::R14, X86::R15, X86::RBP, 0
  };
  const unsigned *CURegs = (Is64Bit ? CU64BitRegs : CU32BitRegs);

  // Encode the registers in the order they were saved, 3-bits per register. The
  // registers are numbered from 1 to 6.
  uint32_t RegEnc = 0;
  for (int I = 5; I >= 0; --I) {
    unsigned Reg = SavedRegs[I];
    if (Reg == 0) break;
    int CURegNum = getCompactUnwindRegNum(CURegs, Reg);
    if (CURegNum == -1)
      return ~0U;
    RegEnc |= (CURegNum & 0x7) << (5 - I);
  }

  assert((RegEnc & 0x7FFF) == RegEnc && "Invalid compact register encoding!");
  return RegEnc;
}

uint32_t X86FrameLowering::getCompactUnwindEncoding(MachineFunction &MF) const {
  const X86RegisterInfo *RegInfo = TM.getRegisterInfo();
  unsigned FramePtr = RegInfo->getFrameRegister(MF);
  unsigned StackPtr = RegInfo->getStackRegister();

  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  int TailCallReturnAddrDelta = X86FI->getTCReturnAddrDelta();

  bool Is64Bit = STI.is64Bit();
  bool HasFP = hasFP(MF);

  unsigned SavedRegs[6] = { 0, 0, 0, 0, 0, 0 };
  int SavedRegIdx = 6;

  unsigned OffsetSize = (Is64Bit ? 8 : 4);

  unsigned PushInstr = (Is64Bit ? X86::PUSH64r : X86::PUSH32r);
  unsigned PushInstrSize = 1;
  unsigned MoveInstr = (Is64Bit ? X86::MOV64rr : X86::MOV32rr);
  unsigned MoveInstrSize = (Is64Bit ? 3 : 2);
  unsigned SubtractInstr = getSUBriOpcode(Is64Bit, -TailCallReturnAddrDelta);
  unsigned SubtractInstrIdx = (Is64Bit ? 3 : 2);

  unsigned StackDivide = (Is64Bit ? 8 : 4);

  unsigned InstrOffset = 0;
  unsigned CFAOffset = 0;
  unsigned StackAdjust = 0;

  MachineBasicBlock &MBB = MF.front(); // Prologue is in entry BB.
  bool ExpectEnd = false;
  for (MachineBasicBlock::iterator
         MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ++MBBI) {
    MachineInstr &MI = *MBBI;
    unsigned Opc = MI.getOpcode();
    if (Opc == X86::PROLOG_LABEL) continue;
    if (!MI.getFlag(MachineInstr::FrameSetup)) break;

    // We don't exect any more prolog instructions.
    if (ExpectEnd) return 0;

    if (Opc == PushInstr) {
      // If there are too many saved registers, we cannot use compact encoding.
      if (--SavedRegIdx < 0) return 0;

      SavedRegs[SavedRegIdx] = MI.getOperand(0).getReg();
      CFAOffset += OffsetSize;
      InstrOffset += PushInstrSize;
    } else if (Opc == MoveInstr) {
      unsigned SrcReg = MI.getOperand(1).getReg();
      unsigned DstReg = MI.getOperand(0).getReg();

      if (DstReg != FramePtr || SrcReg != StackPtr)
        return 0;

      CFAOffset = 0;
      memset(SavedRegs, 0, sizeof(SavedRegs));
      InstrOffset += MoveInstrSize;
    } else if (Opc == SubtractInstr) {
      if (StackAdjust)
        // We all ready have a stack pointer adjustment.
        return 0;

      if (!MI.getOperand(0).isReg() ||
          MI.getOperand(0).getReg() != MI.getOperand(1).getReg() ||
          MI.getOperand(0).getReg() != StackPtr || !MI.getOperand(2).isImm())
        // We need this to be a stack adjustment pointer. Something like:
        //
        //   %RSP<def> = SUB64ri8 %RSP, 48
        return 0;

      StackAdjust = MI.getOperand(2).getImm() / StackDivide;
      SubtractInstrIdx += InstrOffset;
      ExpectEnd = true;
    }
  }

  // Encode that we are using EBP/RBP as the frame pointer.
  uint32_t CompactUnwindEncoding = 0;
  CFAOffset /= StackDivide;
  if (HasFP) {
    if ((CFAOffset & 0xFF) != CFAOffset)
      // Offset was too big for compact encoding.
      return 0;

    // Get the encoding of the saved registers when we have a frame pointer.
    uint32_t RegEnc = encodeCompactUnwindRegistersWithFrame(SavedRegs, Is64Bit);
    if (RegEnc == ~0U)
      return 0;

    CompactUnwindEncoding |= 0x01000000;
    CompactUnwindEncoding |= (CFAOffset & 0xFF) << 16;
    CompactUnwindEncoding |= RegEnc & 0x7FFF;
  } else {
    unsigned FullOffset = CFAOffset + StackAdjust;
    if ((FullOffset & 0xFF) == FullOffset) {
      // Frameless stack.
      CompactUnwindEncoding |= 0x02000000;
      CompactUnwindEncoding |= (FullOffset & 0xFF) << 16;
    } else {
      if ((CFAOffset & 0x7) != CFAOffset)
        // The extra stack adjustments are too big for us to handle.
        return 0;

      // Frameless stack with an offset too large for us to encode compactly.
      CompactUnwindEncoding |= 0x03000000;

      // Encode the offset to the nnnnnn value in the 'subl $nnnnnn, ESP'
      // instruction.
      CompactUnwindEncoding |= (SubtractInstrIdx & 0xFF) << 16;

      // Encode any extra stack stack changes (done via push instructions).
      CompactUnwindEncoding |= (CFAOffset & 0x7) << 13;
    }

    // Get the encoding of the saved registers when we don't have a frame
    // pointer.
    uint32_t RegEnc = encodeCompactUnwindRegistersWithoutFrame(SavedRegs,
                                                               6 - SavedRegIdx,
                                                               Is64Bit);
    if (RegEnc == ~0U) return 0;
    CompactUnwindEncoding |= RegEnc & 0x3FF;
  }

  return CompactUnwindEncoding;
}

/// emitPrologue - Push callee-saved registers onto the stack, which
/// automatically adjust the stack pointer. Adjust the stack pointer to allocate
/// space for local variables. Also emit labels used by the exception handler to
/// generate the exception handling frames.
void X86FrameLowering::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front(); // Prologue goes in entry BB.
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const Function *Fn = MF.getFunction();
  const X86RegisterInfo *RegInfo = TM.getRegisterInfo();
  const X86InstrInfo &TII = *TM.getInstrInfo();
  MachineModuleInfo &MMI = MF.getMMI();
  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  bool needsFrameMoves = MMI.hasDebugInfo() ||
    Fn->needsUnwindTableEntry();
  uint64_t MaxAlign  = MFI->getMaxAlignment(); // Desired stack alignment.
  uint64_t StackSize = MFI->getStackSize();    // Number of bytes to allocate.
  bool HasFP = hasFP(MF);
  bool Is64Bit = STI.is64Bit();
  bool IsWin64 = STI.isTargetWin64();
  unsigned StackAlign = getStackAlignment();
  unsigned SlotSize = RegInfo->getSlotSize();
  unsigned FramePtr = RegInfo->getFrameRegister(MF);
  unsigned StackPtr = RegInfo->getStackRegister();
  DebugLoc DL;

  // If we're forcing a stack realignment we can't rely on just the frame
  // info, we need to know the ABI stack alignment as well in case we
  // have a call out.  Otherwise just make sure we have some alignment - we'll
  // go with the minimum SlotSize.
  if (ForceStackAlign) {
    if (MFI->hasCalls())
      MaxAlign = (StackAlign > MaxAlign) ? StackAlign : MaxAlign;
    else if (MaxAlign < SlotSize)
      MaxAlign = SlotSize;
  }

  // Add RETADDR move area to callee saved frame size.
  int TailCallReturnAddrDelta = X86FI->getTCReturnAddrDelta();
  if (TailCallReturnAddrDelta < 0)
    X86FI->setCalleeSavedFrameSize(
      X86FI->getCalleeSavedFrameSize() - TailCallReturnAddrDelta);

  // If this is x86-64 and the Red Zone is not disabled, if we are a leaf
  // function, and use up to 128 bytes of stack space, don't have a frame
  // pointer, calls, or dynamic alloca then we do not need to adjust the
  // stack pointer (we fit in the Red Zone).
  if (Is64Bit && !Fn->hasFnAttr(Attribute::NoRedZone) &&
      !RegInfo->needsStackRealignment(MF) &&
      !MFI->hasVarSizedObjects() &&                // No dynamic alloca.
      !MFI->adjustsStack() &&                      // No calls.
      !IsWin64 &&                                  // Win64 has no Red Zone
      !EnableSegmentedStacks) {                    // Regular stack
    uint64_t MinSize = X86FI->getCalleeSavedFrameSize();
    if (HasFP) MinSize += SlotSize;
    StackSize = std::max(MinSize, StackSize > 128 ? StackSize - 128 : 0);
    MFI->setStackSize(StackSize);
  }

  // Insert stack pointer adjustment for later moving of return addr.  Only
  // applies to tail call optimized functions where the callee argument stack
  // size is bigger than the callers.
  if (TailCallReturnAddrDelta < 0) {
    MachineInstr *MI =
      BuildMI(MBB, MBBI, DL,
              TII.get(getSUBriOpcode(Is64Bit, -TailCallReturnAddrDelta)),
              StackPtr)
        .addReg(StackPtr)
        .addImm(-TailCallReturnAddrDelta)
        .setMIFlag(MachineInstr::FrameSetup);
    MI->getOperand(3).setIsDead(); // The EFLAGS implicit def is dead.
  }

  // Mapping for machine moves:
  //
  //   DST: VirtualFP AND
  //        SRC: VirtualFP              => DW_CFA_def_cfa_offset
  //        ELSE                        => DW_CFA_def_cfa
  //
  //   SRC: VirtualFP AND
  //        DST: Register               => DW_CFA_def_cfa_register
  //
  //   ELSE
  //        OFFSET < 0                  => DW_CFA_offset_extended_sf
  //        REG < 64                    => DW_CFA_offset + Reg
  //        ELSE                        => DW_CFA_offset_extended

  std::vector<MachineMove> &Moves = MMI.getFrameMoves();
  const TargetData *TD = MF.getTarget().getTargetData();
  uint64_t NumBytes = 0;
  int stackGrowth = -TD->getPointerSize();

  if (HasFP) {
    // Calculate required stack adjustment.
    uint64_t FrameSize = StackSize - SlotSize;
    if (RegInfo->needsStackRealignment(MF))
      FrameSize = (FrameSize + MaxAlign - 1) / MaxAlign * MaxAlign;

    NumBytes = FrameSize - X86FI->getCalleeSavedFrameSize();

    // Get the offset of the stack slot for the EBP register, which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    // Update the frame offset adjustment.
    MFI->setOffsetAdjustment(-NumBytes);

    // Save EBP/RBP into the appropriate stack slot.
    BuildMI(MBB, MBBI, DL, TII.get(Is64Bit ? X86::PUSH64r : X86::PUSH32r))
      .addReg(FramePtr, RegState::Kill)
      .setMIFlag(MachineInstr::FrameSetup);

    if (needsFrameMoves) {
      // Mark the place where EBP/RBP was saved.
      MCSymbol *FrameLabel = MMI.getContext().CreateTempSymbol();
      BuildMI(MBB, MBBI, DL, TII.get(X86::PROLOG_LABEL))
        .addSym(FrameLabel);

      // Define the current CFA rule to use the provided offset.
      if (StackSize) {
        MachineLocation SPDst(MachineLocation::VirtualFP);
        MachineLocation SPSrc(MachineLocation::VirtualFP, 2 * stackGrowth);
        Moves.push_back(MachineMove(FrameLabel, SPDst, SPSrc));
      } else {
        MachineLocation SPDst(StackPtr);
        MachineLocation SPSrc(StackPtr, stackGrowth);
        Moves.push_back(MachineMove(FrameLabel, SPDst, SPSrc));
      }

      // Change the rule for the FramePtr to be an "offset" rule.
      MachineLocation FPDst(MachineLocation::VirtualFP, 2 * stackGrowth);
      MachineLocation FPSrc(FramePtr);
      Moves.push_back(MachineMove(FrameLabel, FPDst, FPSrc));
    }

    // Update EBP with the new base value.
    BuildMI(MBB, MBBI, DL,
            TII.get(Is64Bit ? X86::MOV64rr : X86::MOV32rr), FramePtr)
        .addReg(StackPtr)
        .setMIFlag(MachineInstr::FrameSetup);

    if (needsFrameMoves) {
      // Mark effective beginning of when frame pointer becomes valid.
      MCSymbol *FrameLabel = MMI.getContext().CreateTempSymbol();
      BuildMI(MBB, MBBI, DL, TII.get(X86::PROLOG_LABEL))
        .addSym(FrameLabel);

      // Define the current CFA to use the EBP/RBP register.
      MachineLocation FPDst(FramePtr);
      MachineLocation FPSrc(MachineLocation::VirtualFP);
      Moves.push_back(MachineMove(FrameLabel, FPDst, FPSrc));
    }

    // Mark the FramePtr as live-in in every block except the entry.
    for (MachineFunction::iterator I = llvm::next(MF.begin()), E = MF.end();
         I != E; ++I)
      I->addLiveIn(FramePtr);

    // Realign stack
    if (RegInfo->needsStackRealignment(MF)) {
      MachineInstr *MI =
        BuildMI(MBB, MBBI, DL,
                TII.get(Is64Bit ? X86::AND64ri32 : X86::AND32ri), StackPtr)
        .addReg(StackPtr)
        .addImm(-MaxAlign)
        .setMIFlag(MachineInstr::FrameSetup);

      // The EFLAGS implicit def is dead.
      MI->getOperand(3).setIsDead();
    }
  } else {
    NumBytes = StackSize - X86FI->getCalleeSavedFrameSize();
  }

  // Skip the callee-saved push instructions.
  bool PushedRegs = false;
  int StackOffset = 2 * stackGrowth;

  while (MBBI != MBB.end() &&
         (MBBI->getOpcode() == X86::PUSH32r ||
          MBBI->getOpcode() == X86::PUSH64r)) {
    PushedRegs = true;
    MBBI->setFlag(MachineInstr::FrameSetup);
    ++MBBI;

    if (!HasFP && needsFrameMoves) {
      // Mark callee-saved push instruction.
      MCSymbol *Label = MMI.getContext().CreateTempSymbol();
      BuildMI(MBB, MBBI, DL, TII.get(X86::PROLOG_LABEL)).addSym(Label);

      // Define the current CFA rule to use the provided offset.
      unsigned Ptr = StackSize ? MachineLocation::VirtualFP : StackPtr;
      MachineLocation SPDst(Ptr);
      MachineLocation SPSrc(Ptr, StackOffset);
      Moves.push_back(MachineMove(Label, SPDst, SPSrc));
      StackOffset += stackGrowth;
    }
  }

  DL = MBB.findDebugLoc(MBBI);

  // If there is an SUB32ri of ESP immediately before this instruction, merge
  // the two. This can be the case when tail call elimination is enabled and
  // the callee has more arguments then the caller.
  NumBytes -= mergeSPUpdates(MBB, MBBI, StackPtr, true);

  // If there is an ADD32ri or SUB32ri of ESP immediately after this
  // instruction, merge the two instructions.
  mergeSPUpdatesDown(MBB, MBBI, StackPtr, &NumBytes);

  // Adjust stack pointer: ESP -= numbytes.

  // Windows and cygwin/mingw require a prologue helper routine when allocating
  // more than 4K bytes on the stack.  Windows uses __chkstk and cygwin/mingw
  // uses __alloca.  __alloca and the 32-bit version of __chkstk will probe the
  // stack and adjust the stack pointer in one go.  The 64-bit version of
  // __chkstk is only responsible for probing the stack.  The 64-bit prologue is
  // responsible for adjusting the stack pointer.  Touching the stack at 4K
  // increments is necessary to ensure that the guard pages used by the OS
  // virtual memory manager are allocated in correct sequence.
  if (NumBytes >= 4096 && STI.isTargetCOFF() && !STI.isTargetEnvMacho()) {
    const char *StackProbeSymbol;
    bool isSPUpdateNeeded = false;

    if (Is64Bit) {
      if (STI.isTargetCygMing())
        StackProbeSymbol = "___chkstk";
      else {
        StackProbeSymbol = "__chkstk";
        isSPUpdateNeeded = true;
      }
    } else if (STI.isTargetCygMing())
      StackProbeSymbol = "_alloca";
    else
      StackProbeSymbol = "_chkstk";

    // Check whether EAX is livein for this function.
    bool isEAXAlive = isEAXLiveIn(MF);

    if (isEAXAlive) {
      // Sanity check that EAX is not livein for this function.
      // It should not be, so throw an assert.
      assert(!Is64Bit && "EAX is livein in x64 case!");

      // Save EAX
      BuildMI(MBB, MBBI, DL, TII.get(X86::PUSH32r))
        .addReg(X86::EAX, RegState::Kill)
        .setMIFlag(MachineInstr::FrameSetup);
    }

    if (Is64Bit) {
      // Handle the 64-bit Windows ABI case where we need to call __chkstk.
      // Function prologue is responsible for adjusting the stack pointer.
      BuildMI(MBB, MBBI, DL, TII.get(X86::MOV64ri), X86::RAX)
        .addImm(NumBytes)
        .setMIFlag(MachineInstr::FrameSetup);
    } else {
      // Allocate NumBytes-4 bytes on stack in case of isEAXAlive.
      // We'll also use 4 already allocated bytes for EAX.
      BuildMI(MBB, MBBI, DL, TII.get(X86::MOV32ri), X86::EAX)
        .addImm(isEAXAlive ? NumBytes - 4 : NumBytes)
        .setMIFlag(MachineInstr::FrameSetup);
    }

    BuildMI(MBB, MBBI, DL,
            TII.get(Is64Bit ? X86::W64ALLOCA : X86::CALLpcrel32))
      .addExternalSymbol(StackProbeSymbol)
      .addReg(StackPtr,    RegState::Define | RegState::Implicit)
      .addReg(X86::EFLAGS, RegState::Define | RegState::Implicit)
      .setMIFlag(MachineInstr::FrameSetup);

    // MSVC x64's __chkstk needs to adjust %rsp.
    // FIXME: %rax preserves the offset and should be available.
    if (isSPUpdateNeeded)
      emitSPUpdate(MBB, MBBI, StackPtr, -(int64_t)NumBytes, Is64Bit,
                   TII, *RegInfo);

    if (isEAXAlive) {
        // Restore EAX
        MachineInstr *MI = addRegOffset(BuildMI(MF, DL, TII.get(X86::MOV32rm),
                                                X86::EAX),
                                        StackPtr, false, NumBytes - 4);
        MI->setFlag(MachineInstr::FrameSetup);
        MBB.insert(MBBI, MI);
    }
  } else if (NumBytes)
    emitSPUpdate(MBB, MBBI, StackPtr, -(int64_t)NumBytes, Is64Bit,
                 TII, *RegInfo);

  if (( (!HasFP && NumBytes) || PushedRegs) && needsFrameMoves) {
    // Mark end of stack pointer adjustment.
    MCSymbol *Label = MMI.getContext().CreateTempSymbol();
    BuildMI(MBB, MBBI, DL, TII.get(X86::PROLOG_LABEL))
      .addSym(Label);

    if (!HasFP && NumBytes) {
      // Define the current CFA rule to use the provided offset.
      if (StackSize) {
        MachineLocation SPDst(MachineLocation::VirtualFP);
        MachineLocation SPSrc(MachineLocation::VirtualFP,
                              -StackSize + stackGrowth);
        Moves.push_back(MachineMove(Label, SPDst, SPSrc));
      } else {
        MachineLocation SPDst(StackPtr);
        MachineLocation SPSrc(StackPtr, stackGrowth);
        Moves.push_back(MachineMove(Label, SPDst, SPSrc));
      }
    }

    // Emit DWARF info specifying the offsets of the callee-saved registers.
    if (PushedRegs)
      emitCalleeSavedFrameMoves(MF, Label, HasFP ? FramePtr : StackPtr);
  }

  // Darwin 10.7 and greater has support for compact unwind encoding.
  if (STI.getTargetTriple().isMacOSX() &&
      !STI.getTargetTriple().isMacOSXVersionLT(10, 7))
    MMI.setCompactUnwindEncoding(getCompactUnwindEncoding(MF));
}

void X86FrameLowering::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  const X86RegisterInfo *RegInfo = TM.getRegisterInfo();
  const X86InstrInfo &TII = *TM.getInstrInfo();
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  assert(MBBI != MBB.end() && "Returning block has no instructions");
  unsigned RetOpcode = MBBI->getOpcode();
  DebugLoc DL = MBBI->getDebugLoc();
  bool Is64Bit = STI.is64Bit();
  unsigned StackAlign = getStackAlignment();
  unsigned SlotSize = RegInfo->getSlotSize();
  unsigned FramePtr = RegInfo->getFrameRegister(MF);
  unsigned StackPtr = RegInfo->getStackRegister();

  switch (RetOpcode) {
  default:
    llvm_unreachable("Can only insert epilog into returning blocks");
  case X86::RET:
  case X86::RETI:
  case X86::TCRETURNdi:
  case X86::TCRETURNri:
  case X86::TCRETURNmi:
  case X86::TCRETURNdi64:
  case X86::TCRETURNri64:
  case X86::TCRETURNmi64:
  case X86::EH_RETURN:
  case X86::EH_RETURN64:
    break;  // These are ok
  }

  // Get the number of bytes to allocate from the FrameInfo.
  uint64_t StackSize = MFI->getStackSize();
  uint64_t MaxAlign  = MFI->getMaxAlignment();
  unsigned CSSize = X86FI->getCalleeSavedFrameSize();
  uint64_t NumBytes = 0;

  // If we're forcing a stack realignment we can't rely on just the frame
  // info, we need to know the ABI stack alignment as well in case we
  // have a call out.  Otherwise just make sure we have some alignment - we'll
  // go with the minimum.
  if (ForceStackAlign) {
    if (MFI->hasCalls())
      MaxAlign = (StackAlign > MaxAlign) ? StackAlign : MaxAlign;
    else
      MaxAlign = MaxAlign ? MaxAlign : 4;
  }

  if (hasFP(MF)) {
    // Calculate required stack adjustment.
    uint64_t FrameSize = StackSize - SlotSize;
    if (RegInfo->needsStackRealignment(MF))
      FrameSize = (FrameSize + MaxAlign - 1)/MaxAlign*MaxAlign;

    NumBytes = FrameSize - CSSize;

    // Pop EBP.
    BuildMI(MBB, MBBI, DL,
            TII.get(Is64Bit ? X86::POP64r : X86::POP32r), FramePtr);
  } else {
    NumBytes = StackSize - CSSize;
  }

  // Skip the callee-saved pop instructions.
  MachineBasicBlock::iterator LastCSPop = MBBI;
  while (MBBI != MBB.begin()) {
    MachineBasicBlock::iterator PI = prior(MBBI);
    unsigned Opc = PI->getOpcode();

    if (Opc != X86::POP32r && Opc != X86::POP64r && Opc != X86::DBG_VALUE &&
        !PI->getDesc().isTerminator())
      break;

    --MBBI;
  }

  DL = MBBI->getDebugLoc();

  // If there is an ADD32ri or SUB32ri of ESP immediately before this
  // instruction, merge the two instructions.
  if (NumBytes || MFI->hasVarSizedObjects())
    mergeSPUpdatesUp(MBB, MBBI, StackPtr, &NumBytes);

  // If dynamic alloca is used, then reset esp to point to the last callee-saved
  // slot before popping them off! Same applies for the case, when stack was
  // realigned.
  if (RegInfo->needsStackRealignment(MF)) {
    // We cannot use LEA here, because stack pointer was realigned. We need to
    // deallocate local frame back.
    if (CSSize) {
      emitSPUpdate(MBB, MBBI, StackPtr, NumBytes, Is64Bit, TII, *RegInfo);
      MBBI = prior(LastCSPop);
    }

    BuildMI(MBB, MBBI, DL,
            TII.get(Is64Bit ? X86::MOV64rr : X86::MOV32rr),
            StackPtr).addReg(FramePtr);
  } else if (MFI->hasVarSizedObjects()) {
    if (CSSize) {
      unsigned Opc = Is64Bit ? X86::LEA64r : X86::LEA32r;
      MachineInstr *MI =
        addRegOffset(BuildMI(MF, DL, TII.get(Opc), StackPtr),
                     FramePtr, false, -CSSize);
      MBB.insert(MBBI, MI);
    } else {
      BuildMI(MBB, MBBI, DL,
              TII.get(Is64Bit ? X86::MOV64rr : X86::MOV32rr), StackPtr)
        .addReg(FramePtr);
    }
  } else if (NumBytes) {
    // Adjust stack pointer back: ESP += numbytes.
    emitSPUpdate(MBB, MBBI, StackPtr, NumBytes, Is64Bit, TII, *RegInfo);
  }

  // We're returning from function via eh_return.
  if (RetOpcode == X86::EH_RETURN || RetOpcode == X86::EH_RETURN64) {
    MBBI = MBB.getLastNonDebugInstr();
    MachineOperand &DestAddr  = MBBI->getOperand(0);
    assert(DestAddr.isReg() && "Offset should be in register!");
    BuildMI(MBB, MBBI, DL,
            TII.get(Is64Bit ? X86::MOV64rr : X86::MOV32rr),
            StackPtr).addReg(DestAddr.getReg());
  } else if (RetOpcode == X86::TCRETURNri || RetOpcode == X86::TCRETURNdi ||
             RetOpcode == X86::TCRETURNmi ||
             RetOpcode == X86::TCRETURNri64 || RetOpcode == X86::TCRETURNdi64 ||
             RetOpcode == X86::TCRETURNmi64) {
    bool isMem = RetOpcode == X86::TCRETURNmi || RetOpcode == X86::TCRETURNmi64;
    // Tail call return: adjust the stack pointer and jump to callee.
    MBBI = MBB.getLastNonDebugInstr();
    MachineOperand &JumpTarget = MBBI->getOperand(0);
    MachineOperand &StackAdjust = MBBI->getOperand(isMem ? 5 : 1);
    assert(StackAdjust.isImm() && "Expecting immediate value.");

    // Adjust stack pointer.
    int StackAdj = StackAdjust.getImm();
    int MaxTCDelta = X86FI->getTCReturnAddrDelta();
    int Offset = 0;
    assert(MaxTCDelta <= 0 && "MaxTCDelta should never be positive");

    // Incoporate the retaddr area.
    Offset = StackAdj-MaxTCDelta;
    assert(Offset >= 0 && "Offset should never be negative");

    if (Offset) {
      // Check for possible merge with preceding ADD instruction.
      Offset += mergeSPUpdates(MBB, MBBI, StackPtr, true);
      emitSPUpdate(MBB, MBBI, StackPtr, Offset, Is64Bit, TII, *RegInfo);
    }

    // Jump to label or value in register.
    if (RetOpcode == X86::TCRETURNdi || RetOpcode == X86::TCRETURNdi64) {
      MachineInstrBuilder MIB =
        BuildMI(MBB, MBBI, DL, TII.get((RetOpcode == X86::TCRETURNdi)
                                       ? X86::TAILJMPd : X86::TAILJMPd64));
      if (JumpTarget.isGlobal())
        MIB.addGlobalAddress(JumpTarget.getGlobal(), JumpTarget.getOffset(),
                             JumpTarget.getTargetFlags());
      else {
        assert(JumpTarget.isSymbol());
        MIB.addExternalSymbol(JumpTarget.getSymbolName(),
                              JumpTarget.getTargetFlags());
      }
    } else if (RetOpcode == X86::TCRETURNmi || RetOpcode == X86::TCRETURNmi64) {
      MachineInstrBuilder MIB =
        BuildMI(MBB, MBBI, DL, TII.get((RetOpcode == X86::TCRETURNmi)
                                       ? X86::TAILJMPm : X86::TAILJMPm64));
      for (unsigned i = 0; i != 5; ++i)
        MIB.addOperand(MBBI->getOperand(i));
    } else if (RetOpcode == X86::TCRETURNri64) {
      BuildMI(MBB, MBBI, DL, TII.get(X86::TAILJMPr64)).
        addReg(JumpTarget.getReg(), RegState::Kill);
    } else {
      BuildMI(MBB, MBBI, DL, TII.get(X86::TAILJMPr)).
        addReg(JumpTarget.getReg(), RegState::Kill);
    }

    MachineInstr *NewMI = prior(MBBI);
    for (unsigned i = 2, e = MBBI->getNumOperands(); i != e; ++i)
      NewMI->addOperand(MBBI->getOperand(i));

    // Delete the pseudo instruction TCRETURN.
    MBB.erase(MBBI);
  } else if ((RetOpcode == X86::RET || RetOpcode == X86::RETI) &&
             (X86FI->getTCReturnAddrDelta() < 0)) {
    // Add the return addr area delta back since we are not tail calling.
    int delta = -1*X86FI->getTCReturnAddrDelta();
    MBBI = MBB.getLastNonDebugInstr();

    // Check for possible merge with preceding ADD instruction.
    delta += mergeSPUpdates(MBB, MBBI, StackPtr, true);
    emitSPUpdate(MBB, MBBI, StackPtr, delta, Is64Bit, TII, *RegInfo);
  }
}

int X86FrameLowering::getFrameIndexOffset(const MachineFunction &MF, int FI) const {
  const X86RegisterInfo *RI =
    static_cast<const X86RegisterInfo*>(MF.getTarget().getRegisterInfo());
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  int Offset = MFI->getObjectOffset(FI) - getOffsetOfLocalArea();
  uint64_t StackSize = MFI->getStackSize();

  if (RI->needsStackRealignment(MF)) {
    if (FI < 0) {
      // Skip the saved EBP.
      Offset += RI->getSlotSize();
    } else {
      assert((-(Offset + StackSize)) % MFI->getObjectAlignment(FI) == 0);
      return Offset + StackSize;
    }
    // FIXME: Support tail calls
  } else {
    if (!hasFP(MF))
      return Offset + StackSize;

    // Skip the saved EBP.
    Offset += RI->getSlotSize();

    // Skip the RETADDR move area
    const X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
    int TailCallReturnAddrDelta = X86FI->getTCReturnAddrDelta();
    if (TailCallReturnAddrDelta < 0)
      Offset -= TailCallReturnAddrDelta;
  }

  return Offset;
}

bool X86FrameLowering::spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                             MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                          const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL = MBB.findDebugLoc(MI);

  MachineFunction &MF = *MBB.getParent();

  unsigned SlotSize = STI.is64Bit() ? 8 : 4;
  unsigned FPReg = TRI->getFrameRegister(MF);
  unsigned CalleeFrameSize = 0;

  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();
  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();

  // Push GPRs. It increases frame size.
  unsigned Opc = STI.is64Bit() ? X86::PUSH64r : X86::PUSH32r;
  for (unsigned i = CSI.size(); i != 0; --i) {
    unsigned Reg = CSI[i-1].getReg();
    if (!X86::GR64RegClass.contains(Reg) &&
        !X86::GR32RegClass.contains(Reg))
      continue;
    // Add the callee-saved register as live-in. It's killed at the spill.
    MBB.addLiveIn(Reg);
    if (Reg == FPReg)
      // X86RegisterInfo::emitPrologue will handle spilling of frame register.
      continue;
    CalleeFrameSize += SlotSize;
    BuildMI(MBB, MI, DL, TII.get(Opc)).addReg(Reg, RegState::Kill)
      .setMIFlag(MachineInstr::FrameSetup);
  }

  X86FI->setCalleeSavedFrameSize(CalleeFrameSize);

  // Make XMM regs spilled. X86 does not have ability of push/pop XMM.
  // It can be done by spilling XMMs to stack frame.
  // Note that only Win64 ABI might spill XMMs.
  for (unsigned i = CSI.size(); i != 0; --i) {
    unsigned Reg = CSI[i-1].getReg();
    if (X86::GR64RegClass.contains(Reg) ||
        X86::GR32RegClass.contains(Reg))
      continue;
    // Add the callee-saved register as live-in. It's killed at the spill.
    MBB.addLiveIn(Reg);
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII.storeRegToStackSlot(MBB, MI, Reg, true, CSI[i-1].getFrameIdx(),
                            RC, TRI);
  }

  return true;
}

bool X86FrameLowering::restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                               MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                          const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL = MBB.findDebugLoc(MI);

  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo &TII = *MF.getTarget().getInstrInfo();

  // Reload XMMs from stack frame.
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    if (X86::GR64RegClass.contains(Reg) ||
        X86::GR32RegClass.contains(Reg))
      continue;
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII.loadRegFromStackSlot(MBB, MI, Reg, CSI[i].getFrameIdx(),
                             RC, TRI);
  }

  // POP GPRs.
  unsigned FPReg = TRI->getFrameRegister(MF);
  unsigned Opc = STI.is64Bit() ? X86::POP64r : X86::POP32r;
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    if (!X86::GR64RegClass.contains(Reg) &&
        !X86::GR32RegClass.contains(Reg))
      continue;
    if (Reg == FPReg)
      // X86RegisterInfo::emitEpilogue will handle restoring of frame register.
      continue;
    BuildMI(MBB, MI, DL, TII.get(Opc), Reg);
  }
  return true;
}

void
X86FrameLowering::processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                                   RegScavenger *RS) const {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const X86RegisterInfo *RegInfo = TM.getRegisterInfo();
  unsigned SlotSize = RegInfo->getSlotSize();

  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  int32_t TailCallReturnAddrDelta = X86FI->getTCReturnAddrDelta();

  if (TailCallReturnAddrDelta < 0) {
    // create RETURNADDR area
    //   arg
    //   arg
    //   RETADDR
    //   { ...
    //     RETADDR area
    //     ...
    //   }
    //   [EBP]
    MFI->CreateFixedObject(-TailCallReturnAddrDelta,
                           (-1U*SlotSize)+TailCallReturnAddrDelta, true);
  }

  if (hasFP(MF)) {
    assert((TailCallReturnAddrDelta <= 0) &&
           "The Delta should always be zero or negative");
    const TargetFrameLowering &TFI = *MF.getTarget().getFrameLowering();

    // Create a frame entry for the EBP register that must be saved.
    int FrameIdx = MFI->CreateFixedObject(SlotSize,
                                          -(int)SlotSize +
                                          TFI.getOffsetOfLocalArea() +
                                          TailCallReturnAddrDelta,
                                          true);
    assert(FrameIdx == MFI->getObjectIndexBegin() &&
           "Slot for EBP register must be last in order to be found!");
    (void)FrameIdx;
  }
}

static bool
HasNestArgument(const MachineFunction *MF) {
  const Function *F = MF->getFunction();
  for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
       I != E; I++) {
    if (I->hasNestAttr())
      return true;
  }
  return false;
}

static unsigned
GetScratchRegister(bool Is64Bit, const MachineFunction &MF) {
  if (Is64Bit) {
    return X86::R11;
  } else {
    CallingConv::ID CallingConvention = MF.getFunction()->getCallingConv();
    bool IsNested = HasNestArgument(&MF);

    if (CallingConvention == CallingConv::X86_FastCall) {
      if (IsNested) {
        report_fatal_error("Segmented stacks does not support fastcall with "
                           "nested function.");
        return -1;
      } else {
        return X86::EAX;
      }
    } else {
      if (IsNested)
        return X86::EDX;
      else
        return X86::ECX;
    }
  }
}

void
X86FrameLowering::adjustForSegmentedStacks(MachineFunction &MF) const {
  MachineBasicBlock &prologueMBB = MF.front();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const X86InstrInfo &TII = *TM.getInstrInfo();
  uint64_t StackSize;
  bool Is64Bit = STI.is64Bit();
  unsigned TlsReg, TlsOffset;
  DebugLoc DL;
  const X86Subtarget *ST = &MF.getTarget().getSubtarget<X86Subtarget>();

  unsigned ScratchReg = GetScratchRegister(Is64Bit, MF);
  assert(!MF.getRegInfo().isLiveIn(ScratchReg) &&
         "Scratch register is live-in");

  if (MF.getFunction()->isVarArg())
    report_fatal_error("Segmented stacks do not support vararg functions.");
  if (!ST->isTargetLinux())
    report_fatal_error("Segmented stacks supported only on linux.");

  MachineBasicBlock *allocMBB = MF.CreateMachineBasicBlock();
  MachineBasicBlock *checkMBB = MF.CreateMachineBasicBlock();
  X86MachineFunctionInfo *X86FI = MF.getInfo<X86MachineFunctionInfo>();
  bool IsNested = false;

  // We need to know if the function has a nest argument only in 64 bit mode.
  if (Is64Bit)
    IsNested = HasNestArgument(&MF);

  // The MOV R10, RAX needs to be in a different block, since the RET we emit in
  // allocMBB needs to be last (terminating) instruction.
  MachineBasicBlock *restoreR10MBB = NULL;
  if (IsNested)
    restoreR10MBB = MF.CreateMachineBasicBlock();

  for (MachineBasicBlock::livein_iterator i = prologueMBB.livein_begin(),
         e = prologueMBB.livein_end(); i != e; i++) {
    allocMBB->addLiveIn(*i);
    checkMBB->addLiveIn(*i);

    if (IsNested)
      restoreR10MBB->addLiveIn(*i);
  }

  if (IsNested) {
    allocMBB->addLiveIn(X86::R10);
    restoreR10MBB->addLiveIn(X86::RAX);
  }

  if (IsNested)
    MF.push_front(restoreR10MBB);
  MF.push_front(allocMBB);
  MF.push_front(checkMBB);

  // Eventually StackSize will be calculated by a link-time pass; which will
  // also decide whether checking code needs to be injected into this particular
  // prologue.
  StackSize = MFI->getStackSize();

  // Read the limit off the current stacklet off the stack_guard location.
  if (Is64Bit) {
    TlsReg = X86::FS;
    TlsOffset = 0x70;

    BuildMI(checkMBB, DL, TII.get(X86::LEA64r), ScratchReg).addReg(X86::RSP)
      .addImm(0).addReg(0).addImm(-StackSize).addReg(0);
    BuildMI(checkMBB, DL, TII.get(X86::CMP64rm)).addReg(ScratchReg)
      .addReg(0).addImm(0).addReg(0).addImm(TlsOffset).addReg(TlsReg);
  } else {
    TlsReg = X86::GS;
    TlsOffset = 0x30;

    BuildMI(checkMBB, DL, TII.get(X86::LEA32r), ScratchReg).addReg(X86::ESP)
      .addImm(0).addReg(0).addImm(-StackSize).addReg(0);
    BuildMI(checkMBB, DL, TII.get(X86::CMP32rm)).addReg(ScratchReg)
      .addReg(0).addImm(0).addReg(0).addImm(TlsOffset).addReg(TlsReg);
  }

  // This jump is taken if SP >= (Stacklet Limit + Stack Space required).
  // It jumps to normal execution of the function body.
  BuildMI(checkMBB, DL, TII.get(X86::JG_4)).addMBB(&prologueMBB);

  // On 32 bit we first push the arguments size and then the frame size. On 64
  // bit, we pass the stack frame size in r10 and the argument size in r11.
  if (Is64Bit) {
    // Functions with nested arguments use R10, so it needs to be saved across
    // the call to _morestack

    if (IsNested)
      BuildMI(allocMBB, DL, TII.get(X86::MOV64rr), X86::RAX).addReg(X86::R10);

    BuildMI(allocMBB, DL, TII.get(X86::MOV64ri), X86::R10)
      .addImm(StackSize);
    BuildMI(allocMBB, DL, TII.get(X86::MOV64ri), X86::R11)
      .addImm(X86FI->getArgumentStackSize());
    MF.getRegInfo().setPhysRegUsed(X86::R10);
    MF.getRegInfo().setPhysRegUsed(X86::R11);
  } else {
    // Since we'll call __morestack, stack alignment needs to be preserved.
    BuildMI(allocMBB, DL, TII.get(X86::SUB32ri), X86::ESP).addReg(X86::ESP)
      .addImm(8);
    BuildMI(allocMBB, DL, TII.get(X86::PUSHi32))
      .addImm(X86FI->getArgumentStackSize());
    BuildMI(allocMBB, DL, TII.get(X86::PUSHi32))
      .addImm(StackSize);
  }

  // __morestack is in libgcc
  if (Is64Bit)
    BuildMI(allocMBB, DL, TII.get(X86::CALL64pcrel32))
      .addExternalSymbol("__morestack");
  else
    BuildMI(allocMBB, DL, TII.get(X86::CALLpcrel32))
      .addExternalSymbol("__morestack");

  // __morestack only seems to remove 8 bytes off the stack. Add back the
  // additional 8 bytes we added before pushing the arguments.
  if (!Is64Bit)
    BuildMI(allocMBB, DL, TII.get(X86::ADD32ri), X86::ESP).addReg(X86::ESP)
      .addImm(8);
  BuildMI(allocMBB, DL, TII.get(X86::RET));

  if (IsNested)
    BuildMI(restoreR10MBB, DL, TII.get(X86::MOV64rr), X86::R10)
      .addReg(X86::RAX);

  if (IsNested) {
    allocMBB->addSuccessor(restoreR10MBB);
    restoreR10MBB->addSuccessor(&prologueMBB);
  } else {
    allocMBB->addSuccessor(&prologueMBB);
  }

  checkMBB->addSuccessor(allocMBB);
  checkMBB->addSuccessor(&prologueMBB);

#ifdef XDEBUG
  MF.verify();
#endif
}
