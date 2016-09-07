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

#ifndef LLVM_LIB_TARGET_AMDGPU_SIREGISTERINFO_H
#define LLVM_LIB_TARGET_AMDGPU_SIREGISTERINFO_H

#include "AMDGPURegisterInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

namespace llvm {

class SISubtarget;
class MachineRegisterInfo;

class SIRegisterInfo final : public AMDGPURegisterInfo {
private:
  unsigned SGPRSetID;
  unsigned VGPRSetID;
  BitVector SGPRPressureSets;
  BitVector VGPRPressureSets;

  void reserveRegisterTuples(BitVector &, unsigned Reg) const;
  void classifyPressureSet(unsigned PSetID, unsigned Reg,
                           BitVector &PressureSets) const;

public:
  SIRegisterInfo();

  /// Return the end register initially reserved for the scratch buffer in case
  /// spilling is needed.
  unsigned reservedPrivateSegmentBufferReg(const MachineFunction &MF) const;

  /// Return the end register initially reserved for the scratch wave offset in
  /// case spilling is needed.
  unsigned reservedPrivateSegmentWaveByteOffsetReg(
    const MachineFunction &MF) const;

  BitVector getReservedRegs(const MachineFunction &MF) const override;

  bool requiresRegisterScavenging(const MachineFunction &Fn) const override;


  bool requiresFrameIndexScavenging(const MachineFunction &MF) const override;
  bool requiresVirtualBaseRegisters(const MachineFunction &Fn) const override;
  bool trackLivenessAfterRegAlloc(const MachineFunction &MF) const override;

  int64_t getFrameIndexInstrOffset(const MachineInstr *MI,
                                   int Idx) const override;

  bool needsFrameBaseReg(MachineInstr *MI, int64_t Offset) const override;

  void materializeFrameBaseRegister(MachineBasicBlock *MBB,
                                    unsigned BaseReg, int FrameIdx,
                                    int64_t Offset) const override;

  void resolveFrameIndex(MachineInstr &MI, unsigned BaseReg,
                         int64_t Offset) const override;

  bool isFrameOffsetLegal(const MachineInstr *MI, unsigned BaseReg,
                          int64_t Offset) const override;

  const TargetRegisterClass *getPointerRegClass(
    const MachineFunction &MF, unsigned Kind = 0) const override;

  void eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS) const override;

  unsigned getHWRegIndex(unsigned Reg) const {
    return getEncodingValue(Reg) & 0xff;
  }

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

  bool isSGPRReg(const MachineRegisterInfo &MRI, unsigned Reg) const {
    const TargetRegisterClass *RC;
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      RC = MRI.getRegClass(Reg);
    else
      RC = getPhysRegClass(Reg);
    return isSGPRClass(RC);
  }

  /// \returns true if this class contains VGPR registers.
  bool hasVGPRs(const TargetRegisterClass *RC) const;

  /// returns true if this is a pseudoregister class combination of VGPRs and
  /// SGPRs for operand modeling. FIXME: We should set isAllocatable = 0 on
  /// them.
  static bool isPseudoRegClass(const TargetRegisterClass *RC) {
    return RC == &AMDGPU::VS_32RegClass || RC == &AMDGPU::VS_64RegClass;
  }

  /// \returns A VGPR reg class with the same width as \p SRC
  const TargetRegisterClass *getEquivalentVGPRClass(
                                          const TargetRegisterClass *SRC) const;

  /// \returns A SGPR reg class with the same width as \p SRC
  const TargetRegisterClass *getEquivalentSGPRClass(
                                           const TargetRegisterClass *VRC) const;

  /// \returns The register class that is used for a sub-register of \p RC for
  /// the given \p SubIdx.  If \p SubIdx equals NoSubRegister, \p RC will
  /// be returned.
  const TargetRegisterClass *getSubRegClass(const TargetRegisterClass *RC,
                                            unsigned SubIdx) const;

  bool shouldRewriteCopySrc(const TargetRegisterClass *DefRC,
                            unsigned DefSubReg,
                            const TargetRegisterClass *SrcRC,
                            unsigned SrcSubReg) const override;

  /// \returns True if operands defined with this operand type can accept
  /// a literal constant (i.e. any 32-bit immediate).
  bool opCanUseLiteralConstant(unsigned OpType) const;

  /// \returns True if operands defined with this operand type can accept
  /// an inline constant. i.e. An integer value in the range (-16, 64) or
  /// -4.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f.
  bool opCanUseInlineConstant(unsigned OpType) const;

  enum PreloadedValue {
    // SGPRS:
    PRIVATE_SEGMENT_BUFFER = 0,
    DISPATCH_PTR        =  1,
    QUEUE_PTR           =  2,
    KERNARG_SEGMENT_PTR =  3,
    DISPATCH_ID         =  4,
    FLAT_SCRATCH_INIT   =  5,
    WORKGROUP_ID_X      = 10,
    WORKGROUP_ID_Y      = 11,
    WORKGROUP_ID_Z      = 12,
    PRIVATE_SEGMENT_WAVE_BYTE_OFFSET = 14,

    // VGPRS:
    FIRST_VGPR_VALUE    = 15,
    WORKITEM_ID_X       = FIRST_VGPR_VALUE,
    WORKITEM_ID_Y       = 16,
    WORKITEM_ID_Z       = 17
  };

  /// \brief Returns the physical register that \p Value is stored in.
  unsigned getPreloadedValue(const MachineFunction &MF,
                             enum PreloadedValue Value) const;

  unsigned findUnusedRegister(const MachineRegisterInfo &MRI,
                              const TargetRegisterClass *RC,
                              const MachineFunction &MF) const;

  unsigned getSGPRPressureSet() const { return SGPRSetID; };
  unsigned getVGPRPressureSet() const { return VGPRSetID; };

  bool isVGPR(const MachineRegisterInfo &MRI, unsigned Reg) const;

  bool isSGPRPressureSet(unsigned SetID) const {
    return SGPRPressureSets.test(SetID) && !VGPRPressureSets.test(SetID);
  }
  bool isVGPRPressureSet(unsigned SetID) const {
    return VGPRPressureSets.test(SetID) && !SGPRPressureSets.test(SetID);
  }

  /// \returns SGPR allocation granularity supported by the subtarget.
  unsigned getSGPRAllocGranule() const {
    return 8;
  }

  /// \returns Total number of SGPRs supported by the subtarget.
  unsigned getTotalNumSGPRs(const SISubtarget &ST) const;

  /// \returns Number of addressable SGPRs supported by the subtarget.
  unsigned getNumAddressableSGPRs(const SISubtarget &ST) const;

  /// \returns Number of reserved SGPRs supported by the subtarget.
  unsigned getNumReservedSGPRs(const SISubtarget &ST) const;

  /// \returns Minimum number of SGPRs that meets given number of waves per
  /// execution unit requirement for given subtarget.
  unsigned getMinNumSGPRs(const SISubtarget &ST, unsigned WavesPerEU) const;

  /// \returns Maximum number of SGPRs that meets given number of waves per
  /// execution unit requirement for given subtarget.
  unsigned getMaxNumSGPRs(const SISubtarget &ST, unsigned WavesPerEU) const;

  /// \returns Maximum number of SGPRs that meets number of waves per execution
  /// unit requirement for function \p MF, or number of SGPRs explicitly
  /// requested using "amdgpu-num-sgpr" attribute attached to function \p MF.
  ///
  /// \returns Value that meets number of waves per execution unit requirement
  /// if explicitly requested value cannot be converted to integer, violates
  /// subtarget's specifications, or does not meet number of waves per execution
  /// unit requirement.
  unsigned getMaxNumSGPRs(const MachineFunction &MF) const;

  /// \returns VGPR allocation granularity supported by the subtarget.
  unsigned getVGPRAllocGranule() const {
    return 4;
  }

  /// \returns Total number of VGPRs supported by the subtarget.
  unsigned getTotalNumVGPRs() const {
    return 256;
  }

  /// \returns Number of reserved VGPRs for debugger use supported by the
  /// subtarget.
  unsigned getNumDebuggerReservedVGPRs(const SISubtarget &ST) const;

  /// \returns Minimum number of SGPRs that meets given number of waves per
  /// execution unit requirement.
  unsigned getMinNumVGPRs(unsigned WavesPerEU) const;

  /// \returns Maximum number of VGPRs that meets given number of waves per
  /// execution unit requirement.
  unsigned getMaxNumVGPRs(unsigned WavesPerEU) const;

  /// \returns Maximum number of VGPRs that meets number of waves per execution
  /// unit requirement for function \p MF, or number of VGPRs explicitly
  /// requested using "amdgpu-num-vgpr" attribute attached to function \p MF.
  ///
  /// \returns Value that meets number of waves per execution unit requirement
  /// if explicitly requested value cannot be converted to integer, violates
  /// subtarget's specifications, or does not meet number of waves per execution
  /// unit requirement.
  unsigned getMaxNumVGPRs(const MachineFunction &MF) const;

private:
  void buildScratchLoadStore(MachineBasicBlock::iterator MI,
                             unsigned LoadStoreOp, const MachineOperand *SrcDst,
                             unsigned ScratchRsrcReg, unsigned ScratchOffset,
                             int64_t Offset,
                             RegScavenger *RS) const;
};

} // End namespace llvm

#endif
