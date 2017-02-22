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
#include "SIDefines.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

namespace llvm {

class MachineRegisterInfo;
class SISubtarget;
class SIMachineFunctionInfo;

class SIRegisterInfo final : public AMDGPURegisterInfo {
private:
  unsigned SGPRSetID;
  unsigned VGPRSetID;
  BitVector SGPRPressureSets;
  BitVector VGPRPressureSets;
  bool SpillSGPRToVGPR;
  bool SpillSGPRToSMEM;

  void reserveRegisterTuples(BitVector &, unsigned Reg) const;
  void classifyPressureSet(unsigned PSetID, unsigned Reg,
                           BitVector &PressureSets) const;
public:
  SIRegisterInfo(const SISubtarget &ST);

  bool spillSGPRToVGPR() const {
    return SpillSGPRToVGPR;
  }

  bool spillSGPRToSMEM() const {
    return SpillSGPRToSMEM;
  }

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
  bool requiresFrameIndexReplacementScavenging(
    const MachineFunction &MF) const override;
  bool requiresVirtualBaseRegisters(const MachineFunction &Fn) const override;
  bool trackLivenessAfterRegAlloc(const MachineFunction &MF) const override;

  int64_t getMUBUFInstrOffset(const MachineInstr *MI) const;

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

  /// If \p OnlyToVGPR is true, this will only succeed if this
  bool spillSGPR(MachineBasicBlock::iterator MI,
                 int FI, RegScavenger *RS,
                 bool OnlyToVGPR = false) const;

  bool restoreSGPR(MachineBasicBlock::iterator MI,
                   int FI, RegScavenger *RS,
                   bool OnlyToVGPR = false) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS) const override;

  bool eliminateSGPRToVGPRSpillFrameIndex(MachineBasicBlock::iterator MI,
                                          int FI, RegScavenger *RS) const;

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
  bool opCanUseLiteralConstant(unsigned OpType) const {
    // TODO: 64-bit operands have extending behavior from 32-bit literal.
    return OpType >= AMDGPU::OPERAND_REG_IMM_FIRST &&
           OpType <= AMDGPU::OPERAND_REG_IMM_LAST;
  }

  /// \returns True if operands defined with this operand type can accept
  /// an inline constant. i.e. An integer value in the range (-16, 64) or
  /// -4.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f.
  bool opCanUseInlineConstant(unsigned OpType) const {
    return OpType >= AMDGPU::OPERAND_SRC_FIRST &&
           OpType <= AMDGPU::OPERAND_SRC_LAST;
  }

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

  const TargetRegisterClass *getRegClassForReg(const MachineRegisterInfo &MRI,
                                               unsigned Reg) const;
  bool isVGPR(const MachineRegisterInfo &MRI, unsigned Reg) const;

  bool isSGPRPressureSet(unsigned SetID) const {
    return SGPRPressureSets.test(SetID) && !VGPRPressureSets.test(SetID);
  }
  bool isVGPRPressureSet(unsigned SetID) const {
    return VGPRPressureSets.test(SetID) && !SGPRPressureSets.test(SetID);
  }

  ArrayRef<int16_t> getRegSplitParts(const TargetRegisterClass *RC,
                                     unsigned EltSize) const;

  bool shouldCoalesce(MachineInstr *MI,
                      const TargetRegisterClass *SrcRC,
                      unsigned SubReg,
                      const TargetRegisterClass *DstRC,
                      unsigned DstSubReg,
                      const TargetRegisterClass *NewRC) const override;

  unsigned getRegPressureLimit(const TargetRegisterClass *RC,
                               MachineFunction &MF) const override;

  unsigned getRegPressureSetLimit(const MachineFunction &MF,
                                  unsigned Idx) const override;

  const int *getRegUnitPressureSets(unsigned RegUnit) const override;

private:
  void buildSpillLoadStore(MachineBasicBlock::iterator MI,
                           unsigned LoadStoreOp,
                           int Index,
                           unsigned ValueReg,
                           bool ValueIsKill,
                           unsigned ScratchRsrcReg,
                           unsigned ScratchOffsetReg,
                           int64_t InstrOffset,
                           MachineMemOperand *MMO,
                           RegScavenger *RS) const;
};

} // End namespace llvm

#endif
