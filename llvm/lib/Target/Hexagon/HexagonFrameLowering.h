//=- HexagonFrameLowering.h - Define frame lowering for Hexagon --*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONFRAMELOWERING_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONFRAMELOWERING_H

#include "Hexagon.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {

class HexagonInstrInfo;
class HexagonRegisterInfo;

class HexagonFrameLowering : public TargetFrameLowering {
public:
  explicit HexagonFrameLowering()
      : TargetFrameLowering(StackGrowsDown, 8, 0, 1, true) {}

  // All of the prolog/epilog functionality, including saving and restoring
  // callee-saved registers is handled in emitPrologue. This is to have the
  // logic for shrink-wrapping in one place.
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const
      override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const
      override {}
  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
      MachineBasicBlock::iterator MI, const std::vector<CalleeSavedInfo> &CSI,
      const TargetRegisterInfo *TRI) const override {
    return true;
  }
  bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
      MachineBasicBlock::iterator MI, const std::vector<CalleeSavedInfo> &CSI,
      const TargetRegisterInfo *TRI) const override {
    return true;
  }

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
      MachineBasicBlock &MBB, MachineBasicBlock::iterator I) const override;
  void processFunctionBeforeFrameFinalized(MachineFunction &MF,
        RegScavenger *RS = nullptr) const override;
  void determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
        RegScavenger *RS) const override;

  bool targetHandlesStackFrameRounding() const override {
    return true;
  }
  int getFrameIndexReference(const MachineFunction &MF, int FI,
                             unsigned &FrameReg) const override;
  bool hasFP(const MachineFunction &MF) const override;

  const SpillSlot *getCalleeSavedSpillSlots(unsigned &NumEntries)
        const override {
    static const SpillSlot Offsets[] = {
      { Hexagon::R17, -4 }, { Hexagon::R16, -8 }, { Hexagon::D8, -8 },
      { Hexagon::R19, -12 }, { Hexagon::R18, -16 }, { Hexagon::D9, -16 },
      { Hexagon::R21, -20 }, { Hexagon::R20, -24 }, { Hexagon::D10, -24 },
      { Hexagon::R23, -28 }, { Hexagon::R22, -32 }, { Hexagon::D11, -32 },
      { Hexagon::R25, -36 }, { Hexagon::R24, -40 }, { Hexagon::D12, -40 },
      { Hexagon::R27, -44 }, { Hexagon::R26, -48 }, { Hexagon::D13, -48 }
    };
    NumEntries = array_lengthof(Offsets);
    return Offsets;
  }

  bool assignCalleeSavedSpillSlots(MachineFunction &MF,
      const TargetRegisterInfo *TRI, std::vector<CalleeSavedInfo> &CSI,
      unsigned &MinCSFrameIndex, unsigned &MaxCSFrameIndex)
      const override;

  bool needsAligna(const MachineFunction &MF) const;
  const MachineInstr *getAlignaInstr(const MachineFunction &MF) const;

  void insertCFIInstructions(MachineFunction &MF) const;

private:
  typedef std::vector<CalleeSavedInfo> CSIVect;

  void expandAlloca(MachineInstr *AI, const HexagonInstrInfo &TII,
      unsigned SP, unsigned CF) const;
  void insertPrologueInBlock(MachineBasicBlock &MBB) const;
  void insertEpilogueInBlock(MachineBasicBlock &MBB) const;
  bool insertCSRSpillsInBlock(MachineBasicBlock &MBB, const CSIVect &CSI,
      const HexagonRegisterInfo &HRI) const;
  bool insertCSRRestoresInBlock(MachineBasicBlock &MBB, const CSIVect &CSI,
      const HexagonRegisterInfo &HRI) const;
  void insertCFIInstructionsAt(MachineBasicBlock &MBB,
      MachineBasicBlock::iterator At) const;

  void adjustForCalleeSavedRegsSpillCall(MachineFunction &MF) const;
  bool replacePseudoRegTransferCode(MachineFunction &MF) const;
  bool replaceVecPredRegPseudoSpillCode(MachineFunction &MF) const;

  void findShrunkPrologEpilog(MachineFunction &MF, MachineBasicBlock *&PrologB,
      MachineBasicBlock *&EpilogB) const;

  bool shouldInlineCSR(llvm::MachineFunction &MF, const CSIVect &CSI) const;
  bool useSpillFunction(MachineFunction &MF, const CSIVect &CSI) const;
  bool useRestoreFunction(MachineFunction &MF, const CSIVect &CSI) const;
};

} // End llvm namespace

#endif
