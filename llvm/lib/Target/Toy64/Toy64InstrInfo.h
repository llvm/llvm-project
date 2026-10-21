//===-- Toy64InstrInfo.h - Toy64 Instruction Information ------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_TOY64_TOY64INSTRINFO_H
#define LLVM_LIB_TARGET_TOY64_TOY64INSTRINFO_H

#include "Toy64RegisterInfo.h"
#include "MCTargetDesc/Toy64MCTargetDesc.h"

#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "Toy64GenInstrInfo.inc"

namespace llvm {

class Toy64Subtarget;

class Toy64InstrInfo : public Toy64GenInstrInfo {
public:
  Toy64InstrInfo(
      const Toy64Subtarget &STI,
      const Toy64RegisterInfo &RI);

  void copyPhysReg(
      MachineBasicBlock &MBB,
      MachineBasicBlock::iterator I,
      const DebugLoc &DL,
      Register DestReg,
      Register SrcReg,
      bool KillSrc,
      bool RenamableDest = false,
      bool RenamableSrc = false) const override;

  void storeRegToStackSlot(
      MachineBasicBlock &MBB,
      MachineBasicBlock::iterator MI,
      Register SrcReg,
      bool IsKill,
      int FrameIndex,
      const TargetRegisterClass *RC,
      Register VReg,
      MachineInstr::MIFlag Flags =
          MachineInstr::NoFlags) const override;

  void loadRegFromStackSlot(
      MachineBasicBlock &MBB,
      MachineBasicBlock::iterator MI,
      Register DestReg,
      int FrameIndex,
      const TargetRegisterClass *RC,
      Register VReg,
      unsigned SubReg = 0,
      MachineInstr::MIFlag Flags =
          MachineInstr::NoFlags) const override;

  Register isStoreToStackSlot(
      const MachineInstr &MI,
      int &FrameIndex) const override;

  Register isLoadFromStackSlot(
      const MachineInstr &MI,
      int &FrameIndex) const override;

};

} // namespace llvm

#endif