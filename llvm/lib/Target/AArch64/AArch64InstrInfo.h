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

class AArch64InstrInfo : public AArch64GenInstrInfo {
  // Reserve bits in the MachineMemOperand target hint flags, starting at 1.
  // They will be shifted into MOTargetHintStart when accessed.
  enum TargetMemOperandFlags {
    MOSuppressPair = 1
  };

  const AArch64RegisterInfo RI;
  const AArch64Subtarget &Subtarget;

public:
  explicit AArch64InstrInfo(const AArch64Subtarget &STI);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  const AArch64RegisterInfo &getRegisterInfo() const { return RI; }

  unsigned GetInstSizeInBytes(const MachineInstr *MI) const;

  bool isAsCheapAsAMove(const MachineInstr *MI) const override;

  bool isCoalescableExtInstr(const MachineInstr &MI, unsigned &SrcReg,
                             unsigned &DstReg, unsigned &SubIdx) const override;

  bool
  areMemAccessesTriviallyDisjoint(MachineInstr *MIa, MachineInstr *MIb,
                                  AliasAnalysis *AA = nullptr) const override;

  unsigned isLoadFromStackSlot(const MachineInstr *MI,
                               int &FrameIndex) const override;
  unsigned isStoreToStackSlot(const MachineInstr *MI,
                              int &FrameIndex) const override;

  /// Returns true if there is a shiftable register and that the shift value
  /// is non-zero.
  bool hasShiftedReg(const MachineInstr *MI) const;

  /// Returns true if there is an extendable register and that the extending
  /// value is non-zero.
  bool hasExtendedReg(const MachineInstr *MI) const;

  /// \brief Does this instruction set its full destination register to zero?
  bool isGPRZero(const MachineInstr *MI) const;

  /// \brief Does this instruction rename a GPR without modifying bits?
  bool isGPRCopy(const MachineInstr *MI) const;

  /// \brief Does this instruction rename an FPR without modifying bits?
  bool isFPRCopy(const MachineInstr *MI) const;

  /// Return true if this is load/store scales or extends its register offset.
  /// This refers to scaling a dynamic index as opposed to scaled immediates.
  /// MI should be a memory op that allows scaled addressing.
  bool isScaledAddr(const MachineInstr *MI) const;

  /// Return true if pairing the given load or store is hinted to be
  /// unprofitable.
  bool isLdStPairSuppressed(const MachineInstr *MI) const;

  /// Hint that pairing the given load or store is unprofitable.
  void suppressLdStPair(MachineInstr *MI) const;

  bool getMemOpBaseRegImmOfs(MachineInstr *LdSt, unsigned &BaseReg,
                             unsigned &Offset,
                             const TargetRegisterInfo *TRI) const override;

  bool getMemOpBaseRegImmOfsWidth(MachineInstr *LdSt, unsigned &BaseReg,
                                  int &Offset, int &Width,
                                  const TargetRegisterInfo *TRI) const;

  bool enableClusterLoads() const override { return true; }

  bool shouldClusterLoads(MachineInstr *FirstLdSt, MachineInstr *SecondLdSt,
                          unsigned NumLoads) const override;

  bool shouldScheduleAdjacent(MachineInstr *First,
                              MachineInstr *Second) const override;

  MachineInstr *emitFrameIndexDebugValue(MachineFunction &MF, int FrameIx,
                                         uint64_t Offset, const MDNode *Var,
                                         const MDNode *Expr, DebugLoc DL) const;
  void copyPhysRegTuple(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                        DebugLoc DL, unsigned DestReg, unsigned SrcReg,
                        bool KillSrc, unsigned Opcode,
                        llvm::ArrayRef<unsigned> Indices) const;
  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                   DebugLoc DL, unsigned DestReg, unsigned SrcReg,
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

  using TargetInstrInfo::foldMemoryOperandImpl;
  MachineInstr *foldMemoryOperandImpl(MachineFunction &MF, MachineInstr *MI,
                                      ArrayRef<unsigned> Ops,
                                      MachineBasicBlock::iterator InsertPt,
                                      int FrameIndex) const override;

  bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify = false) const override;
  unsigned RemoveBranch(MachineBasicBlock &MBB) const override;
  unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        DebugLoc DL) const override;
  bool
  ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const override;
  bool canInsertSelect(const MachineBasicBlock &, ArrayRef<MachineOperand> Cond,
                       unsigned, unsigned, int &, int &, int &) const override;
  void insertSelect(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                    DebugLoc DL, unsigned DstReg, ArrayRef<MachineOperand> Cond,
                    unsigned TrueReg, unsigned FalseReg) const override;
  void getNoopForMachoTarget(MCInst &NopInst) const override;

  /// analyzeCompare - For a comparison instruction, return the source registers
  /// in SrcReg and SrcReg2, and the value it compares against in CmpValue.
  /// Return true if the comparison instruction can be analyzed.
  bool analyzeCompare(const MachineInstr *MI, unsigned &SrcReg,
                      unsigned &SrcReg2, int &CmpMask,
                      int &CmpValue) const override;
  /// optimizeCompareInstr - Convert the instruction supplying the argument to
  /// the comparison into one that sets the zero bit in the flags register.
  bool optimizeCompareInstr(MachineInstr *CmpInstr, unsigned SrcReg,
                            unsigned SrcReg2, int CmpMask, int CmpValue,
                            const MachineRegisterInfo *MRI) const override;
  bool optimizeCondBranch(MachineInstr *MI) const override;
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
  /// useMachineCombiner - AArch64 supports MachineCombiner
  bool useMachineCombiner() const override;

  bool expandPostRAPseudo(MachineBasicBlock::iterator MI) const override;

  std::pair<unsigned, unsigned>
  decomposeMachineOperandsTargetFlags(unsigned TF) const override;
  ArrayRef<std::pair<unsigned, const char *>>
  getSerializableDirectMachineOperandTargetFlags() const override;
  ArrayRef<std::pair<unsigned, const char *>>
  getSerializableBitmaskMachineOperandTargetFlags() const override;

private:
  void instantiateCondBranch(MachineBasicBlock &MBB, DebugLoc DL,
                             MachineBasicBlock *TBB,
                             ArrayRef<MachineOperand> Cond) const;
};

/// emitFrameOffset - Emit instructions as needed to set DestReg to SrcReg
/// plus Offset.  This is intended to be used from within the prolog/epilog
/// insertion (PEI) pass, where a virtual scratch register may be allocated
/// if necessary, to be replaced by the scavenger at the end of PEI.
void emitFrameOffset(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                     DebugLoc DL, unsigned DestReg, unsigned SrcReg, int Offset,
                     const TargetInstrInfo *TII,
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
