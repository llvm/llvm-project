//===-- SIRegisterInfo.h - SI Register Info Interface ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Interface definition for SIRegisterInfo
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_LIB_TARGET_R600_SIREGISTERINFO_H
#define LLVM_LIB_TARGET_R600_SIREGISTERINFO_H

#include "AMDGPURegisterInfo.h"
#include "AMDGPUSubtarget.h"
#include "llvm/Support/Debug.h"

namespace llvm {

struct SIRegisterInfo : public AMDGPURegisterInfo {
private:
  void reserveRegisterTuples(BitVector &, unsigned Reg) const;

public:
  SIRegisterInfo();

  BitVector getReservedRegs(const MachineFunction &MF) const override;

  unsigned getRegPressureSetLimit(const MachineFunction &MF,
                                  unsigned Idx) const override;

  bool requiresRegisterScavenging(const MachineFunction &Fn) const override;

  void eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS) const override;

  unsigned getHWRegIndex(unsigned Reg) const override;

  /// \brief Return the 'base' register class for this register.
  /// e.g. SGPR0 => SReg_32, VGPR => VGPR_32 SGPR0_SGPR1 -> SReg_32, etc.
  const TargetRegisterClass *getPhysRegClass(unsigned Reg) const;

  /// \returns true if this class contains only SGPR registers
  bool isSGPRClass(const TargetRegisterClass *RC) const {
    return !hasVGPRs(RC);
  }

  /// \returns true if this class ID contains only SGPR registers
  bool isSGPRClassID(unsigned RCID) const {
    return isSGPRClass(getRegClass(RCID));
  }

  /// \returns true if this class contains VGPR registers.
  bool hasVGPRs(const TargetRegisterClass *RC) const;

  /// \returns A VGPR reg class with the same width as \p SRC
  const TargetRegisterClass *getEquivalentVGPRClass(
                                          const TargetRegisterClass *SRC) const;

  /// \returns The register class that is used for a sub-register of \p RC for
  /// the given \p SubIdx.  If \p SubIdx equals NoSubRegister, \p RC will
  /// be returned.
  const TargetRegisterClass *getSubRegClass(const TargetRegisterClass *RC,
                                            unsigned SubIdx) const;

  bool shouldRewriteCopySrc(const TargetRegisterClass *DefRC,
                            unsigned DefSubReg,
                            const TargetRegisterClass *SrcRC,
                            unsigned SrcSubReg) const override;

  /// \p Channel This is the register channel (e.g. a value from 0-16), not the
  ///            SubReg index.
  /// \returns The sub-register of Reg that is in Channel.
  unsigned getPhysRegSubReg(unsigned Reg, const TargetRegisterClass *SubRC,
                            unsigned Channel) const;

  /// \returns True if operands defined with this operand type can accept
  /// a literal constant (i.e. any 32-bit immediate).
  bool opCanUseLiteralConstant(unsigned OpType) const;

  /// \returns True if operands defined with this operand type can accept
  /// an inline constant. i.e. An integer value in the range (-16, 64) or
  /// -4.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f. 
  bool opCanUseInlineConstant(unsigned OpType) const;

  enum PreloadedValue {
    // SGPRS:
    SCRATCH_PTR         =  0,
    INPUT_PTR           =  3,
    TGID_X              = 10,
    TGID_Y              = 11,
    TGID_Z              = 12,
    SCRATCH_WAVE_OFFSET = 14,
    // VGPRS:
    FIRST_VGPR_VALUE    = 15,
    TIDIG_X             = FIRST_VGPR_VALUE,
    TIDIG_Y             = 16,
    TIDIG_Z             = 17,
  };

  /// \brief Returns the physical register that \p Value is stored in.
  unsigned getPreloadedValue(const MachineFunction &MF,
                             enum PreloadedValue Value) const;

  /// \brief Give the maximum number of VGPRs that can be used by \p WaveCount
  ///        concurrent waves.
  unsigned getNumVGPRsAllowed(unsigned WaveCount) const;

  /// \brief Give the maximum number of SGPRs that can be used by \p WaveCount
  ///        concurrent waves.
  unsigned getNumSGPRsAllowed(AMDGPUSubtarget::Generation gen,
                              unsigned WaveCount) const;

  unsigned findUnusedRegister(const MachineRegisterInfo &MRI,
                              const TargetRegisterClass *RC) const;

private:
  void buildScratchLoadStore(MachineBasicBlock::iterator MI,
                             unsigned LoadStoreOp, unsigned Value,
                             unsigned ScratchRsrcReg, unsigned ScratchOffset,
                             int64_t Offset, RegScavenger *RS) const;
};

} // End namespace llvm

#endif
