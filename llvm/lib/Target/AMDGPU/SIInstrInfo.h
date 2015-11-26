//===-- SIInstrInfo.h - SI Instruction Info Interface -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Interface definition for SIInstrInfo.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_LIB_TARGET_R600_SIINSTRINFO_H
#define LLVM_LIB_TARGET_R600_SIINSTRINFO_H

#include "AMDGPUInstrInfo.h"
#include "SIDefines.h"
#include "SIRegisterInfo.h"

namespace llvm {

class SIInstrInfo : public AMDGPUInstrInfo {
private:
  const SIRegisterInfo RI;

  unsigned buildExtractSubReg(MachineBasicBlock::iterator MI,
                              MachineRegisterInfo &MRI,
                              MachineOperand &SuperReg,
                              const TargetRegisterClass *SuperRC,
                              unsigned SubIdx,
                              const TargetRegisterClass *SubRC) const;
  MachineOperand buildExtractSubRegOrImm(MachineBasicBlock::iterator MI,
                                         MachineRegisterInfo &MRI,
                                         MachineOperand &SuperReg,
                                         const TargetRegisterClass *SuperRC,
                                         unsigned SubIdx,
                                         const TargetRegisterClass *SubRC) const;

  void swapOperands(MachineBasicBlock::iterator Inst) const;

  void lowerScalarAbs(SmallVectorImpl<MachineInstr *> &Worklist,
                      MachineInstr *Inst) const;

  void splitScalar64BitUnaryOp(SmallVectorImpl<MachineInstr *> &Worklist,
                               MachineInstr *Inst, unsigned Opcode) const;

  void splitScalar64BitBinaryOp(SmallVectorImpl<MachineInstr *> &Worklist,
                                MachineInstr *Inst, unsigned Opcode) const;

  void splitScalar64BitBCNT(SmallVectorImpl<MachineInstr *> &Worklist,
                            MachineInstr *Inst) const;
  void splitScalar64BitBFE(SmallVectorImpl<MachineInstr *> &Worklist,
                           MachineInstr *Inst) const;

  void addUsersToMoveToVALUWorklist(
    unsigned Reg, MachineRegisterInfo &MRI,
    SmallVectorImpl<MachineInstr *> &Worklist) const;

  const TargetRegisterClass *
  getDestEquivalentVGPRClass(const MachineInstr &Inst) const;

  bool checkInstOffsetsDoNotOverlap(MachineInstr *MIa,
                                    MachineInstr *MIb) const;

  unsigned findUsedSGPR(const MachineInstr *MI, int OpIndices[3]) const;

protected:
  MachineInstr *commuteInstructionImpl(MachineInstr *MI,
                                       bool NewMI,
                                       unsigned OpIdx0,
                                       unsigned OpIdx1) const override;

public:
  explicit SIInstrInfo(const AMDGPUSubtarget &st);

  const SIRegisterInfo &getRegisterInfo() const override {
    return RI;
  }

  bool isReallyTriviallyReMaterializable(const MachineInstr *MI,
                                         AliasAnalysis *AA) const override;

  bool areLoadsFromSameBasePtr(SDNode *Load1, SDNode *Load2,
                               int64_t &Offset1,
                               int64_t &Offset2) const override;

  bool getMemOpBaseRegImmOfs(MachineInstr *LdSt, unsigned &BaseReg,
                             unsigned &Offset,
                             const TargetRegisterInfo *TRI) const final;

  bool shouldClusterLoads(MachineInstr *FirstLdSt,
                          MachineInstr *SecondLdSt,
                          unsigned NumLoads) const final;

  void copyPhysReg(MachineBasicBlock &MBB,
                   MachineBasicBlock::iterator MI, DebugLoc DL,
                   unsigned DestReg, unsigned SrcReg,
                   bool KillSrc) const override;

  unsigned calculateLDSSpillAddress(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
                                    RegScavenger *RS,
                                    unsigned TmpReg,
                                    unsigned Offset,
                                    unsigned Size) const;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI,
                           unsigned SrcReg, bool isKill, int FrameIndex,
                           const TargetRegisterClass *RC,
                           const TargetRegisterInfo *TRI) const override;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            unsigned DestReg, int FrameIndex,
                            const TargetRegisterClass *RC,
                            const TargetRegisterInfo *TRI) const override;

  bool expandPostRAPseudo(MachineBasicBlock::iterator MI) const override;

  // \brief Returns an opcode that can be used to move a value to a \p DstRC
  // register.  If there is no hardware instruction that can store to \p
  // DstRC, then AMDGPU::COPY is returned.
  unsigned getMovOpcode(const TargetRegisterClass *DstRC) const;

  LLVM_READONLY
  int commuteOpcode(const MachineInstr &MI) const;

  bool findCommutedOpIndices(MachineInstr *MI,
                             unsigned &SrcOpIdx1,
                             unsigned &SrcOpIdx2) const override;

  bool areMemAccessesTriviallyDisjoint(
    MachineInstr *MIa, MachineInstr *MIb,
    AliasAnalysis *AA = nullptr) const override;

  MachineInstr *buildMovInstr(MachineBasicBlock *MBB,
                              MachineBasicBlock::iterator I,
                              unsigned DstReg, unsigned SrcReg) const override;
  bool isMov(unsigned Opcode) const override;

  bool FoldImmediate(MachineInstr *UseMI, MachineInstr *DefMI,
                     unsigned Reg, MachineRegisterInfo *MRI) const final;

  unsigned getMachineCSELookAheadLimit() const override { return 500; }

  MachineInstr *convertToThreeAddress(MachineFunction::iterator &MBB,
                                      MachineBasicBlock::iterator &MI,
                                      LiveVariables *LV) const override;

  static bool isSALU(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::SALU;
  }

  bool isSALU(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::SALU;
  }

  static bool isVALU(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::VALU;
  }

  bool isVALU(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::VALU;
  }

  static bool isSOP1(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::SOP1;
  }

  bool isSOP1(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::SOP1;
  }

  static bool isSOP2(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::SOP2;
  }

  bool isSOP2(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::SOP2;
  }

  static bool isSOPC(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::SOPC;
  }

  bool isSOPC(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::SOPC;
  }

  static bool isSOPK(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::SOPK;
  }

  bool isSOPK(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::SOPK;
  }

  static bool isSOPP(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::SOPP;
  }

  bool isSOPP(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::SOPP;
  }

  static bool isVOP1(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::VOP1;
  }

  bool isVOP1(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::VOP1;
  }

  static bool isVOP2(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::VOP2;
  }

  bool isVOP2(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::VOP2;
  }

  static bool isVOP3(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::VOP3;
  }

  bool isVOP3(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::VOP3;
  }

  static bool isVOPC(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::VOPC;
  }

  bool isVOPC(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::VOPC;
  }

  static bool isMUBUF(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::MUBUF;
  }

  bool isMUBUF(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::MUBUF;
  }

  static bool isMTBUF(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::MTBUF;
  }

  bool isMTBUF(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::MTBUF;
  }

  static bool isSMRD(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::SMRD;
  }

  bool isSMRD(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::SMRD;
  }

  static bool isDS(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::DS;
  }

  bool isDS(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::DS;
  }

  static bool isMIMG(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::MIMG;
  }

  bool isMIMG(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::MIMG;
  }

  static bool isFLAT(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::FLAT;
  }

  bool isFLAT(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::FLAT;
  }

  static bool isWQM(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::WQM;
  }

  bool isWQM(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::WQM;
  }

  static bool isVGPRSpill(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::VGPRSpill;
  }

  bool isVGPRSpill(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::VGPRSpill;
  }

  bool isInlineConstant(const APInt &Imm) const;
  bool isInlineConstant(const MachineOperand &MO, unsigned OpSize) const;
  bool isLiteralConstant(const MachineOperand &MO, unsigned OpSize) const;

  bool isImmOperandLegal(const MachineInstr *MI, unsigned OpNo,
                         const MachineOperand &MO) const;

  /// \brief Return true if this 64-bit VALU instruction has a 32-bit encoding.
  /// This function will return false if you pass it a 32-bit instruction.
  bool hasVALU32BitEncoding(unsigned Opcode) const;

  /// \brief Returns true if this operand uses the constant bus.
  bool usesConstantBus(const MachineRegisterInfo &MRI,
                       const MachineOperand &MO,
                       unsigned OpSize) const;

  /// \brief Return true if this instruction has any modifiers.
  ///  e.g. src[012]_mod, omod, clamp.
  bool hasModifiers(unsigned Opcode) const;

  bool hasModifiersSet(const MachineInstr &MI,
                       unsigned OpName) const;

  bool verifyInstruction(const MachineInstr *MI,
                         StringRef &ErrInfo) const override;

  static unsigned getVALUOp(const MachineInstr &MI);

  bool isSALUOpSupportedOnVALU(const MachineInstr &MI) const;

  /// \brief Return the correct register class for \p OpNo.  For target-specific
  /// instructions, this will return the register class that has been defined
  /// in tablegen.  For generic instructions, like REG_SEQUENCE it will return
  /// the register class of its machine operand.
  /// to infer the correct register class base on the other operands.
  const TargetRegisterClass *getOpRegClass(const MachineInstr &MI,
                                           unsigned OpNo) const;

  /// \brief Return the size in bytes of the operand OpNo on the given
  // instruction opcode.
  unsigned getOpSize(uint16_t Opcode, unsigned OpNo) const {
    const MCOperandInfo &OpInfo = get(Opcode).OpInfo[OpNo];

    if (OpInfo.RegClass == -1) {
      // If this is an immediate operand, this must be a 32-bit literal.
      assert(OpInfo.OperandType == MCOI::OPERAND_IMMEDIATE);
      return 4;
    }

    return RI.getRegClass(OpInfo.RegClass)->getSize();
  }

  /// \brief This form should usually be preferred since it handles operands
  /// with unknown register classes.
  unsigned getOpSize(const MachineInstr &MI, unsigned OpNo) const {
    return getOpRegClass(MI, OpNo)->getSize();
  }

  /// \returns true if it is legal for the operand at index \p OpNo
  /// to read a VGPR.
  bool canReadVGPR(const MachineInstr &MI, unsigned OpNo) const;

  /// \brief Legalize the \p OpIndex operand of this instruction by inserting
  /// a MOV.  For example:
  /// ADD_I32_e32 VGPR0, 15
  /// to
  /// MOV VGPR1, 15
  /// ADD_I32_e32 VGPR0, VGPR1
  ///
  /// If the operand being legalized is a register, then a COPY will be used
  /// instead of MOV.
  void legalizeOpWithMove(MachineInstr *MI, unsigned OpIdx) const;

  /// \brief Check if \p MO is a legal operand if it was the \p OpIdx Operand
  /// for \p MI.
  bool isOperandLegal(const MachineInstr *MI, unsigned OpIdx,
                      const MachineOperand *MO = nullptr) const;

  /// \brief Fix operands in \p MI to satisfy constant bus requirements.
  void legalizeOperandsVOP3(MachineRegisterInfo &MRI, MachineInstr *MI) const;

  /// \brief Legalize all operands in this instruction.  This function may
  /// create new instruction and insert them before \p MI.
  void legalizeOperands(MachineInstr *MI) const;

  /// \brief Split an SMRD instruction into two smaller loads of half the
  //  size storing the results in \p Lo and \p Hi.
  void splitSMRD(MachineInstr *MI, const TargetRegisterClass *HalfRC,
                 unsigned HalfImmOp, unsigned HalfSGPROp,
                 MachineInstr *&Lo, MachineInstr *&Hi) const;

  void moveSMRDToVALU(MachineInstr *MI, MachineRegisterInfo &MRI,
                      SmallVectorImpl<MachineInstr *> &Worklist) const;

  /// \brief Replace this instruction's opcode with the equivalent VALU
  /// opcode.  This function will also move the users of \p MI to the
  /// VALU if necessary.
  void moveToVALU(MachineInstr &MI) const;

  unsigned calculateIndirectAddress(unsigned RegIndex,
                                    unsigned Channel) const override;

  const TargetRegisterClass *getIndirectAddrRegClass() const override;

  MachineInstrBuilder buildIndirectWrite(MachineBasicBlock *MBB,
                                         MachineBasicBlock::iterator I,
                                         unsigned ValueReg,
                                         unsigned Address,
                                         unsigned OffsetReg) const override;

  MachineInstrBuilder buildIndirectRead(MachineBasicBlock *MBB,
                                        MachineBasicBlock::iterator I,
                                        unsigned ValueReg,
                                        unsigned Address,
                                        unsigned OffsetReg) const override;
  void reserveIndirectRegisters(BitVector &Reserved,
                                const MachineFunction &MF) const;

  void LoadM0(MachineInstr *MoveRel, MachineBasicBlock::iterator I,
              unsigned SavReg, unsigned IndexReg) const;

  void insertNOPs(MachineBasicBlock::iterator MI, int Count) const;

  /// \brief Returns the operand named \p Op.  If \p MI does not have an
  /// operand named \c Op, this function returns nullptr.
  LLVM_READONLY
  MachineOperand *getNamedOperand(MachineInstr &MI, unsigned OperandName) const;

  LLVM_READONLY
  const MachineOperand *getNamedOperand(const MachineInstr &MI,
                                        unsigned OpName) const {
    return getNamedOperand(const_cast<MachineInstr &>(MI), OpName);
  }

  /// Get required immediate operand
  int64_t getNamedImmOperand(const MachineInstr &MI, unsigned OpName) const {
    int Idx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), OpName);
    return MI.getOperand(Idx).getImm();
  }

  uint64_t getDefaultRsrcDataFormat() const;
  uint64_t getScratchRsrcWords23() const;
};

namespace AMDGPU {
  LLVM_READONLY
  int getVOPe64(uint16_t Opcode);

  LLVM_READONLY
  int getVOPe32(uint16_t Opcode);

  LLVM_READONLY
  int getCommuteRev(uint16_t Opcode);

  LLVM_READONLY
  int getCommuteOrig(uint16_t Opcode);

  LLVM_READONLY
  int getAddr64Inst(uint16_t Opcode);

  LLVM_READONLY
  int getAtomicRetOp(uint16_t Opcode);

  LLVM_READONLY
  int getAtomicNoRetOp(uint16_t Opcode);

  const uint64_t RSRC_DATA_FORMAT = 0xf00000000000LL;
  const uint64_t RSRC_TID_ENABLE = 1LL << 55;

} // End namespace AMDGPU

namespace SI {
namespace KernelInputOffsets {

/// Offsets in bytes from the start of the input buffer
enum Offsets {
  NGROUPS_X = 0,
  NGROUPS_Y = 4,
  NGROUPS_Z = 8,
  GLOBAL_SIZE_X = 12,
  GLOBAL_SIZE_Y = 16,
  GLOBAL_SIZE_Z = 20,
  LOCAL_SIZE_X = 24,
  LOCAL_SIZE_Y = 28,
  LOCAL_SIZE_Z = 32
};

} // End namespace KernelInputOffsets
} // End namespace SI

} // End namespace llvm

#endif
