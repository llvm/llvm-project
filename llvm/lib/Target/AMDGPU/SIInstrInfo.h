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


#ifndef LLVM_LIB_TARGET_AMDGPU_SIINSTRINFO_H
#define LLVM_LIB_TARGET_AMDGPU_SIINSTRINFO_H

#include "AMDGPUInstrInfo.h"
#include "SIDefines.h"
#include "SIRegisterInfo.h"

namespace llvm {

class SIInstrInfo final : public AMDGPUInstrInfo {
private:
  const SIRegisterInfo RI;
  const SISubtarget &ST;

  // The the inverse predicate should have the negative value.
  enum BranchPredicate {
    INVALID_BR = 0,
    SCC_TRUE = 1,
    SCC_FALSE = -1,
    VCCNZ = 2,
    VCCZ = -2,
    EXECNZ = -3,
    EXECZ = 3
  };

  static unsigned getBranchOpcode(BranchPredicate Cond);
  static BranchPredicate getBranchPredicate(unsigned Opcode);

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

  void swapOperands(MachineInstr &Inst) const;

  void lowerScalarAbs(SmallVectorImpl<MachineInstr *> &Worklist,
                      MachineInstr &Inst) const;

  void splitScalar64BitUnaryOp(SmallVectorImpl<MachineInstr *> &Worklist,
                               MachineInstr &Inst, unsigned Opcode) const;

  void splitScalar64BitBinaryOp(SmallVectorImpl<MachineInstr *> &Worklist,
                                MachineInstr &Inst, unsigned Opcode) const;

  void splitScalar64BitBCNT(SmallVectorImpl<MachineInstr *> &Worklist,
                            MachineInstr &Inst) const;
  void splitScalar64BitBFE(SmallVectorImpl<MachineInstr *> &Worklist,
                           MachineInstr &Inst) const;
  void movePackToVALU(SmallVectorImpl<MachineInstr *> &Worklist,
                      MachineRegisterInfo &MRI,
                      MachineInstr &Inst) const;

  void addUsersToMoveToVALUWorklist(
    unsigned Reg, MachineRegisterInfo &MRI,
    SmallVectorImpl<MachineInstr *> &Worklist) const;

  void
  addSCCDefUsersToVALUWorklist(MachineInstr &SCCDefInst,
                               SmallVectorImpl<MachineInstr *> &Worklist) const;

  const TargetRegisterClass *
  getDestEquivalentVGPRClass(const MachineInstr &Inst) const;

  bool checkInstOffsetsDoNotOverlap(MachineInstr &MIa, MachineInstr &MIb) const;

  unsigned findUsedSGPR(const MachineInstr &MI, int OpIndices[3]) const;

protected:
  bool swapSourceModifiers(MachineInstr &MI,
                           MachineOperand &Src0, unsigned Src0OpName,
                           MachineOperand &Src1, unsigned Src1OpName) const;

  MachineInstr *commuteInstructionImpl(MachineInstr &MI, bool NewMI,
                                       unsigned OpIdx0,
                                       unsigned OpIdx1) const override;

public:

  enum TargetOperandFlags {
    MO_MASK = 0x7,

    MO_NONE = 0,
    // MO_GOTPCREL -> symbol@GOTPCREL -> R_AMDGPU_GOTPCREL.
    MO_GOTPCREL = 1,
    // MO_GOTPCREL32_LO -> symbol@gotpcrel32@lo -> R_AMDGPU_GOTPCREL32_LO.
    MO_GOTPCREL32 = 2,
    MO_GOTPCREL32_LO = 2,
    // MO_GOTPCREL32_HI -> symbol@gotpcrel32@hi -> R_AMDGPU_GOTPCREL32_HI.
    MO_GOTPCREL32_HI = 3,
    // MO_REL32_LO -> symbol@rel32@lo -> R_AMDGPU_REL32_LO.
    MO_REL32 = 4,
    MO_REL32_LO = 4,
    // MO_REL32_HI -> symbol@rel32@hi -> R_AMDGPU_REL32_HI.
    MO_REL32_HI = 5
  };

  explicit SIInstrInfo(const SISubtarget &);

  const SIRegisterInfo &getRegisterInfo() const {
    return RI;
  }

  bool isReallyTriviallyReMaterializable(const MachineInstr &MI,
                                         AliasAnalysis *AA) const override;

  bool areLoadsFromSameBasePtr(SDNode *Load1, SDNode *Load2,
                               int64_t &Offset1,
                               int64_t &Offset2) const override;

  bool getMemOpBaseRegImmOfs(MachineInstr &LdSt, unsigned &BaseReg,
                             int64_t &Offset,
                             const TargetRegisterInfo *TRI) const final;

  bool shouldClusterMemOps(MachineInstr &FirstLdSt, MachineInstr &SecondLdSt,
                           unsigned NumLoads) const final;

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                   const DebugLoc &DL, unsigned DestReg, unsigned SrcReg,
                   bool KillSrc) const override;

  unsigned calculateLDSSpillAddress(MachineBasicBlock &MBB, MachineInstr &MI,
                                    RegScavenger *RS, unsigned TmpReg,
                                    unsigned Offset, unsigned Size) const;

  void materializeImmediate(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            const DebugLoc &DL,
                            unsigned DestReg,
                            int64_t Value) const;

  const TargetRegisterClass *getPreferredSelectRegClass(
                               unsigned Size) const;

  unsigned insertNE(MachineBasicBlock *MBB,
                    MachineBasicBlock::iterator I, const DebugLoc &DL,
                    unsigned SrcReg, int Value)  const;

  unsigned insertEQ(MachineBasicBlock *MBB,
                    MachineBasicBlock::iterator I, const DebugLoc &DL,
                    unsigned SrcReg, int Value)  const;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, unsigned SrcReg,
                           bool isKill, int FrameIndex,
                           const TargetRegisterClass *RC,
                           const TargetRegisterInfo *TRI) const override;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI, unsigned DestReg,
                            int FrameIndex, const TargetRegisterClass *RC,
                            const TargetRegisterInfo *TRI) const override;

  bool expandPostRAPseudo(MachineInstr &MI) const override;

  // \brief Returns an opcode that can be used to move a value to a \p DstRC
  // register.  If there is no hardware instruction that can store to \p
  // DstRC, then AMDGPU::COPY is returned.
  unsigned getMovOpcode(const TargetRegisterClass *DstRC) const;

  LLVM_READONLY
  int commuteOpcode(unsigned Opc) const;

  LLVM_READONLY
  inline int commuteOpcode(const MachineInstr &MI) const {
    return commuteOpcode(MI.getOpcode());
  }

  bool findCommutedOpIndices(MachineInstr &MI, unsigned &SrcOpIdx1,
                             unsigned &SrcOpIdx2) const override;

  bool isBranchOffsetInRange(unsigned BranchOpc,
                             int64_t BrOffset) const override;

  MachineBasicBlock *getBranchDestBlock(const MachineInstr &MI) const override;

  unsigned insertIndirectBranch(MachineBasicBlock &MBB,
                                MachineBasicBlock &NewDestBB,
                                const DebugLoc &DL,
                                int64_t BrOffset,
                                RegScavenger *RS = nullptr) const override;

  bool analyzeBranchImpl(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator I,
                         MachineBasicBlock *&TBB,
                         MachineBasicBlock *&FBB,
                         SmallVectorImpl<MachineOperand> &Cond,
                         bool AllowModify) const;

  bool analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify = false) const override;

  unsigned removeBranch(MachineBasicBlock &MBB,
                        int *BytesRemoved = nullptr) const override;

  unsigned insertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        const DebugLoc &DL,
                        int *BytesAdded = nullptr) const override;

  bool reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const override;


  bool canInsertSelect(const MachineBasicBlock &MBB,
                       ArrayRef<MachineOperand> Cond,
                       unsigned TrueReg, unsigned FalseReg,
                       int &CondCycles,
                       int &TrueCycles, int &FalseCycles) const override;

  void insertSelect(MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator I, const DebugLoc &DL,
                    unsigned DstReg, ArrayRef<MachineOperand> Cond,
                    unsigned TrueReg, unsigned FalseReg) const override;

  void insertVectorSelect(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator I, const DebugLoc &DL,
                          unsigned DstReg, ArrayRef<MachineOperand> Cond,
                          unsigned TrueReg, unsigned FalseReg) const;

  bool
  areMemAccessesTriviallyDisjoint(MachineInstr &MIa, MachineInstr &MIb,
                                  AliasAnalysis *AA = nullptr) const override;

  bool isFoldableCopy(const MachineInstr &MI) const;

  bool FoldImmediate(MachineInstr &UseMI, MachineInstr &DefMI, unsigned Reg,
                     MachineRegisterInfo *MRI) const final;

  unsigned getMachineCSELookAheadLimit() const override { return 500; }

  MachineInstr *convertToThreeAddress(MachineFunction::iterator &MBB,
                                      MachineInstr &MI,
                                      LiveVariables *LV) const override;

  bool isSchedulingBoundary(const MachineInstr &MI,
                            const MachineBasicBlock *MBB,
                            const MachineFunction &MF) const override;

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

  static bool isVMEM(const MachineInstr &MI) {
    return isMUBUF(MI) || isMTBUF(MI) || isMIMG(MI);
  }

  bool isVMEM(uint16_t Opcode) const {
    return isMUBUF(Opcode) || isMTBUF(Opcode) || isMIMG(Opcode);
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

  static bool isSDWA(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::SDWA;
  }

  bool isSDWA(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::SDWA;
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

  static bool isGather4(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::Gather4;
  }

  bool isGather4(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::Gather4;
  }

  static bool isFLAT(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::FLAT;
  }

  bool isFLAT(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::FLAT;
  }

  static bool isEXP(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::EXP;
  }

  bool isEXP(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::EXP;
  }

  static bool isWQM(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::WQM;
  }

  bool isWQM(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::WQM;
  }

  static bool isDisableWQM(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::DisableWQM;
  }

  bool isDisableWQM(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::DisableWQM;
  }

  static bool isVGPRSpill(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::VGPRSpill;
  }

  bool isVGPRSpill(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::VGPRSpill;
  }

  static bool isSGPRSpill(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::SGPRSpill;
  }

  bool isSGPRSpill(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::SGPRSpill;
  }

  static bool isDPP(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::DPP;
  }

  bool isDPP(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::DPP;
  }

  static bool isVOP3P(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::VOP3P;
  }

  bool isVOP3P(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::VOP3P;
  }

  static bool isVINTRP(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::VINTRP;
  }

  bool isVINTRP(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::VINTRP;
  }

  static bool isScalarUnit(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & (SIInstrFlags::SALU | SIInstrFlags::SMRD);
  }

  static bool usesVM_CNT(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::VM_CNT;
  }

  static bool sopkIsZext(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::SOPK_ZEXT;
  }

  bool sopkIsZext(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::SOPK_ZEXT;
  }

  /// \returns true if this is an s_store_dword* instruction. This is more
  /// specific than than isSMEM && mayStore.
  static bool isScalarStore(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::SCALAR_STORE;
  }

  bool isScalarStore(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::SCALAR_STORE;
  }

  static bool isFixedSize(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::FIXED_SIZE;
  }

  bool isFixedSize(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::FIXED_SIZE;
  }

  static bool hasFPClamp(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & SIInstrFlags::HasFPClamp;
  }

  bool hasFPClamp(uint16_t Opcode) const {
    return get(Opcode).TSFlags & SIInstrFlags::HasFPClamp;
  }

  bool isVGPRCopy(const MachineInstr &MI) const {
    assert(MI.isCopy());
    unsigned Dest = MI.getOperand(0).getReg();
    const MachineFunction &MF = *MI.getParent()->getParent();
    const MachineRegisterInfo &MRI = MF.getRegInfo();
    return !RI.isSGPRReg(MRI, Dest);
  }

  bool isInlineConstant(const APInt &Imm) const;

  bool isInlineConstant(const MachineOperand &MO, uint8_t OperandType) const;

  bool isInlineConstant(const MachineOperand &MO,
                        const MCOperandInfo &OpInfo) const {
    return isInlineConstant(MO, OpInfo.OperandType);
  }

  /// \p returns true if \p UseMO is substituted with \p DefMO in \p MI it would
  /// be an inline immediate.
  bool isInlineConstant(const MachineInstr &MI,
                        const MachineOperand &UseMO,
                        const MachineOperand &DefMO) const {
    assert(UseMO.getParent() == &MI);
    int OpIdx = MI.getOperandNo(&UseMO);
    if (!MI.getDesc().OpInfo || OpIdx >= MI.getDesc().NumOperands) {
      return false;
    }

    return isInlineConstant(DefMO, MI.getDesc().OpInfo[OpIdx]);
  }

  /// \p returns true if the operand \p OpIdx in \p MI is a valid inline
  /// immediate.
  bool isInlineConstant(const MachineInstr &MI, unsigned OpIdx) const {
    const MachineOperand &MO = MI.getOperand(OpIdx);
    return isInlineConstant(MO, MI.getDesc().OpInfo[OpIdx].OperandType);
  }

  bool isInlineConstant(const MachineInstr &MI, unsigned OpIdx,
                        const MachineOperand &MO) const {
    if (!MI.getDesc().OpInfo || OpIdx >= MI.getDesc().NumOperands)
      return false;

    if (MI.isCopy()) {
      unsigned Size = getOpSize(MI, OpIdx);
      assert(Size == 8 || Size == 4);

      uint8_t OpType = (Size == 8) ?
        AMDGPU::OPERAND_REG_IMM_INT64 : AMDGPU::OPERAND_REG_IMM_INT32;
      return isInlineConstant(MO, OpType);
    }

    return isInlineConstant(MO, MI.getDesc().OpInfo[OpIdx].OperandType);
  }

  bool isInlineConstant(const MachineOperand &MO) const {
    const MachineInstr *Parent = MO.getParent();
    return isInlineConstant(*Parent, Parent->getOperandNo(&MO));
  }

  bool isLiteralConstant(const MachineOperand &MO,
                         const MCOperandInfo &OpInfo) const {
    return MO.isImm() && !isInlineConstant(MO, OpInfo.OperandType);
  }

  bool isLiteralConstant(const MachineInstr &MI, int OpIdx) const {
    const MachineOperand &MO = MI.getOperand(OpIdx);
    return MO.isImm() && !isInlineConstant(MI, OpIdx);
  }

  // Returns true if this operand could potentially require a 32-bit literal
  // operand, but not necessarily. A FrameIndex for example could resolve to an
  // inline immediate value that will not require an additional 4-bytes; this
  // assumes that it will.
  bool isLiteralConstantLike(const MachineOperand &MO,
                             const MCOperandInfo &OpInfo) const;

  bool isImmOperandLegal(const MachineInstr &MI, unsigned OpNo,
                         const MachineOperand &MO) const;

  /// \brief Return true if this 64-bit VALU instruction has a 32-bit encoding.
  /// This function will return false if you pass it a 32-bit instruction.
  bool hasVALU32BitEncoding(unsigned Opcode) const;

  /// \brief Returns true if this operand uses the constant bus.
  bool usesConstantBus(const MachineRegisterInfo &MRI,
                       const MachineOperand &MO,
                       const MCOperandInfo &OpInfo) const;

  /// \brief Return true if this instruction has any modifiers.
  ///  e.g. src[012]_mod, omod, clamp.
  bool hasModifiers(unsigned Opcode) const;

  bool hasModifiersSet(const MachineInstr &MI,
                       unsigned OpName) const;
  bool hasAnyModifiersSet(const MachineInstr &MI) const;

  bool verifyInstruction(const MachineInstr &MI,
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

    return RI.getRegSizeInBits(*RI.getRegClass(OpInfo.RegClass)) / 8;
  }

  /// \brief This form should usually be preferred since it handles operands
  /// with unknown register classes.
  unsigned getOpSize(const MachineInstr &MI, unsigned OpNo) const {
    return RI.getRegSizeInBits(*getOpRegClass(MI, OpNo)) / 8;
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
  void legalizeOpWithMove(MachineInstr &MI, unsigned OpIdx) const;

  /// \brief Check if \p MO is a legal operand if it was the \p OpIdx Operand
  /// for \p MI.
  bool isOperandLegal(const MachineInstr &MI, unsigned OpIdx,
                      const MachineOperand *MO = nullptr) const;

  /// \brief Check if \p MO would be a valid operand for the given operand
  /// definition \p OpInfo. Note this does not attempt to validate constant bus
  /// restrictions (e.g. literal constant usage).
  bool isLegalVSrcOperand(const MachineRegisterInfo &MRI,
                          const MCOperandInfo &OpInfo,
                          const MachineOperand &MO) const;

  /// \brief Check if \p MO (a register operand) is a legal register for the
  /// given operand description.
  bool isLegalRegOperand(const MachineRegisterInfo &MRI,
                         const MCOperandInfo &OpInfo,
                         const MachineOperand &MO) const;

  /// \brief Legalize operands in \p MI by either commuting it or inserting a
  /// copy of src1.
  void legalizeOperandsVOP2(MachineRegisterInfo &MRI, MachineInstr &MI) const;

  /// \brief Fix operands in \p MI to satisfy constant bus requirements.
  void legalizeOperandsVOP3(MachineRegisterInfo &MRI, MachineInstr &MI) const;

  /// Copy a value from a VGPR (\p SrcReg) to SGPR.  This function can only
  /// be used when it is know that the value in SrcReg is same across all
  /// threads in the wave.
  /// \returns The SGPR register that \p SrcReg was copied to.
  unsigned readlaneVGPRToSGPR(unsigned SrcReg, MachineInstr &UseMI,
                              MachineRegisterInfo &MRI) const;

  void legalizeOperandsSMRD(MachineRegisterInfo &MRI, MachineInstr &MI) const;

  void legalizeGenericOperand(MachineBasicBlock &InsertMBB,
                              MachineBasicBlock::iterator I,
                              const TargetRegisterClass *DstRC,
                              MachineOperand &Op, MachineRegisterInfo &MRI,
                              const DebugLoc &DL) const;

  /// \brief Legalize all operands in this instruction.  This function may
  /// create new instruction and insert them before \p MI.
  void legalizeOperands(MachineInstr &MI) const;

  /// \brief Replace this instruction's opcode with the equivalent VALU
  /// opcode.  This function will also move the users of \p MI to the
  /// VALU if necessary.
  void moveToVALU(MachineInstr &MI) const;

  void insertWaitStates(MachineBasicBlock &MBB,MachineBasicBlock::iterator MI,
                        int Count) const;

  void insertNoop(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator MI) const override;

  void insertReturn(MachineBasicBlock &MBB) const;
  /// \brief Return the number of wait states that result from executing this
  /// instruction.
  unsigned getNumWaitStates(const MachineInstr &MI) const;

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

  bool isLowLatencyInstruction(const MachineInstr &MI) const;
  bool isHighLatencyInstruction(const MachineInstr &MI) const;

  /// \brief Return the descriptor of the target-specific machine instruction
  /// that corresponds to the specified pseudo or native opcode.
  const MCInstrDesc &getMCOpcodeFromPseudo(unsigned Opcode) const {
    return get(pseudoToMCOpcode(Opcode));
  }

  unsigned isStackAccess(const MachineInstr &MI, int &FrameIndex) const;
  unsigned isSGPRStackAccess(const MachineInstr &MI, int &FrameIndex) const;

  unsigned isLoadFromStackSlot(const MachineInstr &MI,
                               int &FrameIndex) const override;
  unsigned isStoreToStackSlot(const MachineInstr &MI,
                              int &FrameIndex) const override;

  unsigned getInstSizeInBytes(const MachineInstr &MI) const override;

  bool mayAccessFlatAddressSpace(const MachineInstr &MI) const;

  bool isNonUniformBranchInstr(MachineInstr &Instr) const;

  void convertNonUniformIfRegion(MachineBasicBlock *IfEntry,
                                 MachineBasicBlock *IfEnd) const;

  void convertNonUniformLoopRegion(MachineBasicBlock *LoopEntry,
                                   MachineBasicBlock *LoopEnd) const;

  std::pair<unsigned, unsigned>
  decomposeMachineOperandsTargetFlags(unsigned TF) const override;

  ArrayRef<std::pair<int, const char *>>
  getSerializableTargetIndices() const override;

  ArrayRef<std::pair<unsigned, const char *>>
  getSerializableDirectMachineOperandTargetFlags() const override;

  ScheduleHazardRecognizer *
  CreateTargetPostRAHazardRecognizer(const InstrItineraryData *II,
                                 const ScheduleDAG *DAG) const override;

  ScheduleHazardRecognizer *
  CreateTargetPostRAHazardRecognizer(const MachineFunction &MF) const override;

  bool isBasicBlockPrologue(const MachineInstr &MI) const override;

  /// \brief Return a partially built integer add instruction without carry.
  /// Caller must add source operands.
  /// For pre-GFX9 it will generate unused carry destination operand.
  /// TODO: After GFX9 it should return a no-carry operation.
  MachineInstrBuilder getAddNoCarry(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I,
                                    const DebugLoc &DL,
                                    unsigned DestReg) const;
};

namespace AMDGPU {
  LLVM_READONLY
  int getVOPe64(uint16_t Opcode);

  LLVM_READONLY
  int getVOPe32(uint16_t Opcode);

  LLVM_READONLY
  int getSDWAOp(uint16_t Opcode);

  LLVM_READONLY
  int getBasicFromSDWAOp(uint16_t Opcode);

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

  LLVM_READONLY
  int getSOPKOp(uint16_t Opcode);

  const uint64_t RSRC_DATA_FORMAT = 0xf00000000000LL;
  const uint64_t RSRC_ELEMENT_SIZE_SHIFT = (32 + 19);
  const uint64_t RSRC_INDEX_STRIDE_SHIFT = (32 + 21);
  const uint64_t RSRC_TID_ENABLE = UINT64_C(1) << (32 + 23);

  // For MachineOperands.
  enum TargetFlags {
    TF_LONG_BRANCH_FORWARD = 1 << 0,
    TF_LONG_BRANCH_BACKWARD = 1 << 1
  };
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
