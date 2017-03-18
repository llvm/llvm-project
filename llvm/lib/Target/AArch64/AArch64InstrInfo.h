//===- AArch64InstrInfo.h - AArch64 Instruction Information -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the AArch64 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64INSTRINFO_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64INSTRINFO_H

#include "AArch64.h"
#include "AArch64RegisterInfo.h"
#include "llvm/CodeGen/MachineCombinerPattern.h"
#include "llvm/Target/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "AArch64GenInstrInfo.inc"

namespace llvm {

class AArch64Subtarget;
class AArch64TargetMachine;

class AArch64InstrInfo final : public AArch64GenInstrInfo {
  const AArch64RegisterInfo RI;
  const AArch64Subtarget &Subtarget;

public:
  explicit AArch64InstrInfo(const AArch64Subtarget &STI);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  const AArch64RegisterInfo &getRegisterInfo() const { return RI; }

  unsigned getInstSizeInBytes(const MachineInstr &MI) const override;

  bool isAsCheapAsAMove(const MachineInstr &MI) const override;

  bool isCoalescableExtInstr(const MachineInstr &MI, unsigned &SrcReg,
                             unsigned &DstReg, unsigned &SubIdx) const override;

  bool
  areMemAccessesTriviallyDisjoint(MachineInstr &MIa, MachineInstr &MIb,
                                  AliasAnalysis *AA = nullptr) const override;

  unsigned isLoadFromStackSlot(const MachineInstr &MI,
                               int &FrameIndex) const override;
  unsigned isStoreToStackSlot(const MachineInstr &MI,
                              int &FrameIndex) const override;

  /// Returns true if there is a shiftable register and that the shift value
  /// is non-zero.
  bool hasShiftedReg(const MachineInstr &MI) const;

  /// Returns true if there is an extendable register and that the extending
  /// value is non-zero.
  bool hasExtendedReg(const MachineInstr &MI) const;

  /// \brief Does this instruction set its full destination register to zero?
  bool isGPRZero(const MachineInstr &MI) const;

  /// \brief Does this instruction rename a GPR without modifying bits?
  bool isGPRCopy(const MachineInstr &MI) const;

  /// \brief Does this instruction rename an FPR without modifying bits?
  bool isFPRCopy(const MachineInstr &MI) const;

  /// Return true if this is load/store scales or extends its register offset.
  /// This refers to scaling a dynamic index as opposed to scaled immediates.
  /// MI should be a memory op that allows scaled addressing.
  bool isScaledAddr(const MachineInstr &MI) const;

  /// Return true if pairing the given load or store is hinted to be
  /// unprofitable.
  bool isLdStPairSuppressed(const MachineInstr &MI) const;

  /// Return true if this is an unscaled load/store.
  bool isUnscaledLdSt(unsigned Opc) const;

  /// Return true if this is an unscaled load/store.
  bool isUnscaledLdSt(MachineInstr &MI) const;

  static bool isPairableLdStInst(const MachineInstr &MI) {
    switch (MI.getOpcode()) {
    default:
      return false;
    // Scaled instructions.
    case AArch64::STRSui:
    case AArch64::STRDui:
    case AArch64::STRQui:
    case AArch64::STRXui:
    case AArch64::STRWui:
    case AArch64::LDRSui:
    case AArch64::LDRDui:
    case AArch64::LDRQui:
    case AArch64::LDRXui:
    case AArch64::LDRWui:
    case AArch64::LDRSWui:
    // Unscaled instructions.
    case AArch64::STURSi:
    case AArch64::STURDi:
    case AArch64::STURQi:
    case AArch64::STURWi:
    case AArch64::STURXi:
    case AArch64::LDURSi:
    case AArch64::LDURDi:
    case AArch64::LDURQi:
    case AArch64::LDURWi:
    case AArch64::LDURXi:
    case AArch64::LDURSWi:
      return true;
    }
  }

  /// Return true if this is a load/store that can be potentially paired/merged.
  bool isCandidateToMergeOrPair(MachineInstr &MI) const;

  /// Hint that pairing the given load or store is unprofitable.
  void suppressLdStPair(MachineInstr &MI) const;

  bool getMemOpBaseRegImmOfs(MachineInstr &LdSt, unsigned &BaseReg,
                             int64_t &Offset,
                             const TargetRegisterInfo *TRI) const override;

  bool getMemOpBaseRegImmOfsWidth(MachineInstr &LdSt, unsigned &BaseReg,
                                  int64_t &Offset, unsigned &Width,
                                  const TargetRegisterInfo *TRI) const;

  /// Return the immediate offset of the base register in a load/store \p LdSt.
  MachineOperand &getMemOpBaseRegImmOfsOffsetOperand(MachineInstr &LdSt) const;

  /// \brief Returns true if opcode \p Opc is a memory operation. If it is, set
  /// \p Scale, \p Width, \p MinOffset, and \p MaxOffset accordingly.
  ///
  /// For unscaled instructions, \p Scale is set to 1.
  bool getMemOpInfo(unsigned Opcode, unsigned &Scale, unsigned &Width,
                    int64_t &MinOffset, int64_t &MaxOffset) const;

  bool shouldClusterMemOps(MachineInstr &FirstLdSt, MachineInstr &SecondLdSt,
                           unsigned NumLoads) const override;

  MachineInstr *emitFrameIndexDebugValue(MachineFunction &MF, int FrameIx,
                                         uint64_t Offset, const MDNode *Var,
                                         const MDNode *Expr,
                                         const DebugLoc &DL) const;
  void copyPhysRegTuple(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                        const DebugLoc &DL, unsigned DestReg, unsigned SrcReg,
                        bool KillSrc, unsigned Opcode,
                        llvm::ArrayRef<unsigned> Indices) const;
  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                   const DebugLoc &DL, unsigned DestReg, unsigned SrcReg,
                   bool KillSrc) const override;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI, unsigned SrcReg,
                           bool isKill, int FrameIndex,
                           const TargetRegisterClass *RC,
                           const TargetRegisterInfo *TRI) const override;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI, unsigned DestReg,
                            int FrameIndex, const TargetRegisterClass *RC,
                            const TargetRegisterInfo *TRI) const override;

  // This tells target independent code that it is okay to pass instructions
  // with subreg operands to foldMemoryOperandImpl.
  bool isSubregFoldable() const override { return true; }

  using TargetInstrInfo::foldMemoryOperandImpl;
  MachineInstr *
  foldMemoryOperandImpl(MachineFunction &MF, MachineInstr &MI,
                        ArrayRef<unsigned> Ops,
                        MachineBasicBlock::iterator InsertPt, int FrameIndex,
                        LiveIntervals *LIS = nullptr) const override;

  /// \returns true if a branch from an instruction with opcode \p BranchOpc
  ///  bytes is capable of jumping to a position \p BrOffset bytes away.
  bool isBranchOffsetInRange(unsigned BranchOpc,
                             int64_t BrOffset) const override;

  MachineBasicBlock *getBranchDestBlock(const MachineInstr &MI) const override;

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
  bool
  reverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const override;
  bool canInsertSelect(const MachineBasicBlock &, ArrayRef<MachineOperand> Cond,
                       unsigned, unsigned, int &, int &, int &) const override;
  void insertSelect(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                    const DebugLoc &DL, unsigned DstReg,
                    ArrayRef<MachineOperand> Cond, unsigned TrueReg,
                    unsigned FalseReg) const override;
  void getNoopForMachoTarget(MCInst &NopInst) const override;

  /// analyzeCompare - For a comparison instruction, return the source registers
  /// in SrcReg and SrcReg2, and the value it compares against in CmpValue.
  /// Return true if the comparison instruction can be analyzed.
  bool analyzeCompare(const MachineInstr &MI, unsigned &SrcReg,
                      unsigned &SrcReg2, int &CmpMask,
                      int &CmpValue) const override;
  /// optimizeCompareInstr - Convert the instruction supplying the argument to
  /// the comparison into one that sets the zero bit in the flags register.
  bool optimizeCompareInstr(MachineInstr &CmpInstr, unsigned SrcReg,
                            unsigned SrcReg2, int CmpMask, int CmpValue,
                            const MachineRegisterInfo *MRI) const override;
  bool optimizeCondBranch(MachineInstr &MI) const override;

  /// Return true when a code sequence can improve throughput. It
  /// should be called only for instructions in loops.
  /// \param Pattern - combiner pattern
  bool isThroughputPattern(MachineCombinerPattern Pattern) const override;
  /// Return true when there is potentially a faster code sequence
  /// for an instruction chain ending in <Root>. All potential patterns are
  /// listed in the <Patterns> array.
  bool getMachineCombinerPatterns(MachineInstr &Root,
                  SmallVectorImpl<MachineCombinerPattern> &Patterns)
      const override;
  /// Return true when Inst is associative and commutative so that it can be
  /// reassociated.
  bool isAssociativeAndCommutative(const MachineInstr &Inst) const override;
  /// When getMachineCombinerPatterns() finds patterns, this function generates
  /// the instructions that could replace the original code sequence
  void genAlternativeCodeSequence(
      MachineInstr &Root, MachineCombinerPattern Pattern,
      SmallVectorImpl<MachineInstr *> &InsInstrs,
      SmallVectorImpl<MachineInstr *> &DelInstrs,
      DenseMap<unsigned, unsigned> &InstrIdxForVirtReg) const override;
  /// AArch64 supports MachineCombiner.
  bool useMachineCombiner() const override;

  bool expandPostRAPseudo(MachineInstr &MI) const override;

  std::pair<unsigned, unsigned>
  decomposeMachineOperandsTargetFlags(unsigned TF) const override;
  ArrayRef<std::pair<unsigned, const char *>>
  getSerializableDirectMachineOperandTargetFlags() const override;
  ArrayRef<std::pair<unsigned, const char *>>
  getSerializableBitmaskMachineOperandTargetFlags() const override;

  bool isFunctionSafeToOutlineFrom(MachineFunction &MF) const override;
  unsigned getOutliningBenefit(size_t SequenceSize, size_t Occurrences,
                               bool CanBeTailCall) const override;
  AArch64GenInstrInfo::MachineOutlinerInstrType
  getOutliningType(MachineInstr &MI) const override;
  void insertOutlinerEpilogue(MachineBasicBlock &MBB,
                              MachineFunction &MF,
                              bool IsTailCall) const override;
  void insertOutlinerPrologue(MachineBasicBlock &MBB,
                              MachineFunction &MF,
                              bool isTailCall) const override;
  MachineBasicBlock::iterator
  insertOutlinedCall(Module &M, MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator &It,
                     MachineFunction &MF,
                     bool IsTailCall) const override;

private:

  /// \brief Sets the offsets on outlined instructions in \p MBB which use SP
  /// so that they will be valid post-outlining.
  ///
  /// \param MBB A \p MachineBasicBlock in an outlined function.
  void fixupPostOutline(MachineBasicBlock &MBB) const;

  void instantiateCondBranch(MachineBasicBlock &MBB, const DebugLoc &DL,
                             MachineBasicBlock *TBB,
                             ArrayRef<MachineOperand> Cond) const;
  bool substituteCmpToZero(MachineInstr &CmpInstr, unsigned SrcReg,
                           const MachineRegisterInfo *MRI) const;
};

/// emitFrameOffset - Emit instructions as needed to set DestReg to SrcReg
/// plus Offset.  This is intended to be used from within the prolog/epilog
/// insertion (PEI) pass, where a virtual scratch register may be allocated
/// if necessary, to be replaced by the scavenger at the end of PEI.
void emitFrameOffset(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                     const DebugLoc &DL, unsigned DestReg, unsigned SrcReg,
                     int Offset, const TargetInstrInfo *TII,
                     MachineInstr::MIFlag = MachineInstr::NoFlags,
                     bool SetNZCV = false);

/// rewriteAArch64FrameIndex - Rewrite MI to access 'Offset' bytes from the
/// FP. Return false if the offset could not be handled directly in MI, and
/// return the left-over portion by reference.
bool rewriteAArch64FrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                            unsigned FrameReg, int &Offset,
                            const AArch64InstrInfo *TII);

/// \brief Use to report the frame offset status in isAArch64FrameOffsetLegal.
enum AArch64FrameOffsetStatus {
  AArch64FrameOffsetCannotUpdate = 0x0, ///< Offset cannot apply.
  AArch64FrameOffsetIsLegal = 0x1,      ///< Offset is legal.
  AArch64FrameOffsetCanUpdate = 0x2     ///< Offset can apply, at least partly.
};

/// \brief Check if the @p Offset is a valid frame offset for @p MI.
/// The returned value reports the validity of the frame offset for @p MI.
/// It uses the values defined by AArch64FrameOffsetStatus for that.
/// If result == AArch64FrameOffsetCannotUpdate, @p MI cannot be updated to
/// use an offset.eq
/// If result & AArch64FrameOffsetIsLegal, @p Offset can completely be
/// rewriten in @p MI.
/// If result & AArch64FrameOffsetCanUpdate, @p Offset contains the
/// amount that is off the limit of the legal offset.
/// If set, @p OutUseUnscaledOp will contain the whether @p MI should be
/// turned into an unscaled operator, which opcode is in @p OutUnscaledOp.
/// If set, @p EmittableOffset contains the amount that can be set in @p MI
/// (possibly with @p OutUnscaledOp if OutUseUnscaledOp is true) and that
/// is a legal offset.
int isAArch64FrameOffsetLegal(const MachineInstr &MI, int &Offset,
                            bool *OutUseUnscaledOp = nullptr,
                            unsigned *OutUnscaledOp = nullptr,
                            int *EmittableOffset = nullptr);

static inline bool isUncondBranchOpcode(int Opc) { return Opc == AArch64::B; }

static inline bool isCondBranchOpcode(int Opc) {
  switch (Opc) {
  case AArch64::Bcc:
  case AArch64::CBZW:
  case AArch64::CBZX:
  case AArch64::CBNZW:
  case AArch64::CBNZX:
  case AArch64::TBZW:
  case AArch64::TBZX:
  case AArch64::TBNZW:
  case AArch64::TBNZX:
    return true;
  default:
    return false;
  }
}

static inline bool isIndirectBranchOpcode(int Opc) { return Opc == AArch64::BR; }

} // end namespace llvm

#endif
