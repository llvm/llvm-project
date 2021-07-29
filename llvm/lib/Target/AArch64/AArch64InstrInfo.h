//===- AArch64InstrInfo.h - AArch64 Instruction Information -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/ADT/Optional.h"
#include "llvm/CodeGen/MachineCombinerPattern.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/TypeSize.h"

#define GET_INSTRINFO_HEADER
#include "AArch64GenInstrInfo.inc"

namespace llvm {

class AArch64Subtarget;
class AArch64TargetMachine;

static const MachineMemOperand::Flags MOSuppressPair =
    MachineMemOperand::MOTargetFlag1;
static const MachineMemOperand::Flags MOStridedAccess =
    MachineMemOperand::MOTargetFlag2;

#define FALKOR_STRIDED_ACCESS_MD "falkor.strided.access"

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

  bool isCoalescableExtInstr(const MachineInstr &MI, Register &SrcReg,
                             Register &DstReg, unsigned &SubIdx) const override;

  bool
  areMemAccessesTriviallyDisjoint(const MachineInstr &MIa,
                                  const MachineInstr &MIb) const override;

  unsigned isLoadFromStackSlot(const MachineInstr &MI,
                               int &FrameIndex) const override;
  unsigned isStoreToStackSlot(const MachineInstr &MI,
                              int &FrameIndex) const override;

  /// Does this instruction set its full destination register to zero?
  static bool isGPRZero(const MachineInstr &MI);

  /// Does this instruction rename a GPR without modifying bits?
  static bool isGPRCopy(const MachineInstr &MI);

  /// Does this instruction rename an FPR without modifying bits?
  static bool isFPRCopy(const MachineInstr &MI);

  /// Return true if pairing the given load or store is hinted to be
  /// unprofitable.
  static bool isLdStPairSuppressed(const MachineInstr &MI);

  /// Return true if the given load or store is a strided memory access.
  static bool isStridedAccess(const MachineInstr &MI);

  /// Return true if it has an unscaled load/store offset.
  static bool hasUnscaledLdStOffset(unsigned Opc);
  static bool hasUnscaledLdStOffset(MachineInstr &MI) {
    return hasUnscaledLdStOffset(MI.getOpcode());
  }

  /// Returns the unscaled load/store for the scaled load/store opcode,
  /// if there is a corresponding unscaled variant available.
  static Optional<unsigned> getUnscaledLdSt(unsigned Opc);

  /// Scaling factor for (scaled or unscaled) load or store.
  static int getMemScale(unsigned Opc);
  static int getMemScale(const MachineInstr &MI) {
    return getMemScale(MI.getOpcode());
  }

  /// Returns whether the instruction is a pre-indexed load.
  static bool isPreLd(const MachineInstr &MI);

  /// Returns whether the instruction is a pre-indexed store.
  static bool isPreSt(const MachineInstr &MI);

  /// Returns whether the instruction is a pre-indexed load/store.
  static bool isPreLdSt(const MachineInstr &MI);

  /// Returns the index for the immediate for a given instruction.
  static unsigned getLoadStoreImmIdx(unsigned Opc);

  /// Return true if pairing the given load or store may be paired with another.
  static bool isPairableLdStInst(const MachineInstr &MI);

  /// Return the opcode that set flags when possible.  The caller is
  /// responsible for ensuring the opc has a flag setting equivalent.
  static unsigned convertToFlagSettingOpc(unsigned Opc, bool &Is64Bit);

  /// Return true if this is a load/store that can be potentially paired/merged.
  bool isCandidateToMergeOrPair(const MachineInstr &MI) const;

  /// Hint that pairing the given load or store is unprofitable.
  static void suppressLdStPair(MachineInstr &MI);

  Optional<ExtAddrMode>
  getAddrModeFromMemoryOp(const MachineInstr &MemI,
                          const TargetRegisterInfo *TRI) const override;

  bool getMemOperandsWithOffsetWidth(
      const MachineInstr &MI, SmallVectorImpl<const MachineOperand *> &BaseOps,
      int64_t &Offset, bool &OffsetIsScalable, unsigned &Width,
      const TargetRegisterInfo *TRI) const override;

  /// If \p OffsetIsScalable is set to 'true', the offset is scaled by `vscale`.
  /// This is true for some SVE instructions like ldr/str that have a
  /// 'reg + imm' addressing mode where the immediate is an index to the
  /// scalable vector located at 'reg + imm * vscale x #bytes'.
  bool getMemOperandWithOffsetWidth(const MachineInstr &MI,
                                    const MachineOperand *&BaseOp,
                                    int64_t &Offset, bool &OffsetIsScalable,
                                    unsigned &Width,
                                    const TargetRegisterInfo *TRI) const;

  /// Return the immediate offset of the base register in a load/store \p LdSt.
  MachineOperand &getMemOpBaseRegImmOfsOffsetOperand(MachineInstr &LdSt) const;

  /// Returns true if opcode \p Opc is a memory operation. If it is, set
  /// \p Scale, \p Width, \p MinOffset, and \p MaxOffset accordingly.
  ///
  /// For unscaled instructions, \p Scale is set to 1.
  static bool getMemOpInfo(unsigned Opcode, TypeSize &Scale, unsigned &Width,
                           int64_t &MinOffset, int64_t &MaxOffset);

  bool shouldClusterMemOps(ArrayRef<const MachineOperand *> BaseOps1,
                           ArrayRef<const MachineOperand *> BaseOps2,
                           unsigned NumLoads, unsigned NumBytes) const override;

  void copyPhysRegTuple(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                        const DebugLoc &DL, MCRegister DestReg,
                        MCRegister SrcReg, bool KillSrc, unsigned Opcode,
                        llvm::ArrayRef<unsigned> Indices) const;
  void copyGPRRegTuple(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                       DebugLoc DL, unsigned DestReg, unsigned SrcReg,
                       bool KillSrc, unsigned Opcode, unsigned ZeroReg,
                       llvm::ArrayRef<unsigned> Indices) const;
  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                   const DebugLoc &DL, MCRegister DestReg, MCRegister SrcReg,
                   bool KillSrc) const override;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI, Register SrcReg,
                           bool isKill, int FrameIndex,
                           const TargetRegisterClass *RC,
                           const TargetRegisterInfo *TRI) const override;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI, Register DestReg,
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
                        LiveIntervals *LIS = nullptr,
                        VirtRegMap *VRM = nullptr) const override;

  /// \returns true if a branch from an instruction with opcode \p BranchOpc
  ///  bytes is capable of jumping to a position \p BrOffset bytes away.
  bool isBranchOffsetInRange(unsigned BranchOpc,
                             int64_t BrOffset) const override;

  MachineBasicBlock *getBranchDestBlock(const MachineInstr &MI) const override;

  bool analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify = false) const override;
  bool analyzeBranchPredicate(MachineBasicBlock &MBB,
                              MachineBranchPredicate &MBP,
                              bool AllowModify) const override;
  unsigned removeBranch(MachineBasicBlock &MBB,
                        int *BytesRemoved = nullptr) const override;
  unsigned insertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        const DebugLoc &DL,
                        int *BytesAdded = nullptr) const override;
  bool
  reverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const override;
  bool canInsertSelect(const MachineBasicBlock &, ArrayRef<MachineOperand> Cond,
                       Register, Register, Register, int &, int &,
                       int &) const override;
  void insertSelect(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                    const DebugLoc &DL, Register DstReg,
                    ArrayRef<MachineOperand> Cond, Register TrueReg,
                    Register FalseReg) const override;
  MCInst getNop() const override;

  bool isSchedulingBoundary(const MachineInstr &MI,
                            const MachineBasicBlock *MBB,
                            const MachineFunction &MF) const override;

  /// analyzeCompare - For a comparison instruction, return the source registers
  /// in SrcReg and SrcReg2, and the value it compares against in CmpValue.
  /// Return true if the comparison instruction can be analyzed.
  bool analyzeCompare(const MachineInstr &MI, Register &SrcReg,
                      Register &SrcReg2, int &CmpMask,
                      int &CmpValue) const override;
  /// optimizeCompareInstr - Convert the instruction supplying the argument to
  /// the comparison into one that sets the zero bit in the flags register.
  bool optimizeCompareInstr(MachineInstr &CmpInstr, Register SrcReg,
                            Register SrcReg2, int CmpMask, int CmpValue,
                            const MachineRegisterInfo *MRI) const override;
  bool optimizeCondBranch(MachineInstr &MI) const override;

  /// Return true when a code sequence can improve throughput. It
  /// should be called only for instructions in loops.
  /// \param Pattern - combiner pattern
  bool isThroughputPattern(MachineCombinerPattern Pattern) const override;
  /// Return true when there is potentially a faster code sequence
  /// for an instruction chain ending in ``Root``. All potential patterns are
  /// listed in the ``Patterns`` array.
  bool
  getMachineCombinerPatterns(MachineInstr &Root,
                             SmallVectorImpl<MachineCombinerPattern> &Patterns,
                             bool DoRegPressureReduce) const override;
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
  ArrayRef<std::pair<MachineMemOperand::Flags, const char *>>
  getSerializableMachineMemOperandTargetFlags() const override;

  bool isFunctionSafeToOutlineFrom(MachineFunction &MF,
                                   bool OutlineFromLinkOnceODRs) const override;
  outliner::OutlinedFunction getOutliningCandidateInfo(
      std::vector<outliner::Candidate> &RepeatedSequenceLocs) const override;
  outliner::InstrType
  getOutliningType(MachineBasicBlock::iterator &MIT, unsigned Flags) const override;
  bool isMBBSafeToOutlineFrom(MachineBasicBlock &MBB,
                              unsigned &Flags) const override;
  void buildOutlinedFrame(MachineBasicBlock &MBB, MachineFunction &MF,
                          const outliner::OutlinedFunction &OF) const override;
  MachineBasicBlock::iterator
  insertOutlinedCall(Module &M, MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator &It, MachineFunction &MF,
                     const outliner::Candidate &C) const override;
  bool shouldOutlineFromFunctionByDefault(MachineFunction &MF) const override;
  /// Returns the vector element size (B, H, S or D) of an SVE opcode.
  uint64_t getElementSizeForOpcode(unsigned Opc) const;
  /// Returns true if the opcode is for an SVE instruction that sets the
  /// condition codes as if it's results had been fed to a PTEST instruction
  /// along with the same general predicate.
  bool isPTestLikeOpcode(unsigned Opc) const;
  /// Returns true if the opcode is for an SVE WHILE## instruction.
  bool isWhileOpcode(unsigned Opc) const;
  /// Returns true if the instruction has a shift by immediate that can be
  /// executed in one cycle less.
  static bool isFalkorShiftExtFast(const MachineInstr &MI);
  /// Return true if the instructions is a SEH instruciton used for unwinding
  /// on Windows.
  static bool isSEHInstruction(const MachineInstr &MI);

  Optional<RegImmPair> isAddImmediate(const MachineInstr &MI,
                                      Register Reg) const override;

  Optional<ParamLoadedValue> describeLoadedValue(const MachineInstr &MI,
                                                 Register Reg) const override;

  unsigned int getTailDuplicateSize(CodeGenOpt::Level OptLevel) const override;

  bool isExtendLikelyToBeFolded(MachineInstr &ExtMI,
                                MachineRegisterInfo &MRI) const override;

  static void decomposeStackOffsetForFrameOffsets(const StackOffset &Offset,
                                                  int64_t &NumBytes,
                                                  int64_t &NumPredicateVectors,
                                                  int64_t &NumDataVectors);
  static void decomposeStackOffsetForDwarfOffsets(const StackOffset &Offset,
                                                  int64_t &ByteSized,
                                                  int64_t &VGSized);
#define GET_INSTRINFO_HELPER_DECLS
#include "AArch64GenInstrInfo.inc"

protected:
  /// If the specific machine instruction is an instruction that moves/copies
  /// value from one register to another register return destination and source
  /// registers as machine operands.
  Optional<DestSourcePair>
  isCopyInstrImpl(const MachineInstr &MI) const override;

private:
  unsigned getInstBundleLength(const MachineInstr &MI) const;

  /// Sets the offsets on outlined instructions in \p MBB which use SP
  /// so that they will be valid post-outlining.
  ///
  /// \param MBB A \p MachineBasicBlock in an outlined function.
  void fixupPostOutline(MachineBasicBlock &MBB) const;

  void instantiateCondBranch(MachineBasicBlock &MBB, const DebugLoc &DL,
                             MachineBasicBlock *TBB,
                             ArrayRef<MachineOperand> Cond) const;
  bool substituteCmpToZero(MachineInstr &CmpInstr, unsigned SrcReg,
                           const MachineRegisterInfo &MRI) const;
  bool removeCmpToZeroOrOne(MachineInstr &CmpInstr, unsigned SrcReg,
                            int CmpValue, const MachineRegisterInfo &MRI) const;

  /// Returns an unused general-purpose register which can be used for
  /// constructing an outlined call if one exists. Returns 0 otherwise.
  unsigned findRegisterToSaveLRTo(const outliner::Candidate &C) const;

  /// Remove a ptest of a predicate-generating operation that already sets, or
  /// can be made to set, the condition codes in an identical manner
  bool optimizePTestInstr(MachineInstr *PTest, unsigned MaskReg,
                          unsigned PredReg,
                          const MachineRegisterInfo *MRI) const;
};

/// Return true if there is an instruction /after/ \p DefMI and before \p UseMI
/// which either reads or clobbers NZCV.
bool isNZCVTouchedInInstructionRange(const MachineInstr &DefMI,
                                     const MachineInstr &UseMI,
                                     const TargetRegisterInfo *TRI);

/// emitFrameOffset - Emit instructions as needed to set DestReg to SrcReg
/// plus Offset.  This is intended to be used from within the prolog/epilog
/// insertion (PEI) pass, where a virtual scratch register may be allocated
/// if necessary, to be replaced by the scavenger at the end of PEI.
void emitFrameOffset(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                     const DebugLoc &DL, unsigned DestReg, unsigned SrcReg,
                     StackOffset Offset, const TargetInstrInfo *TII,
                     MachineInstr::MIFlag = MachineInstr::NoFlags,
                     bool SetNZCV = false, bool NeedsWinCFI = false,
                     bool *HasWinCFI = nullptr);

/// rewriteAArch64FrameIndex - Rewrite MI to access 'Offset' bytes from the
/// FP. Return false if the offset could not be handled directly in MI, and
/// return the left-over portion by reference.
bool rewriteAArch64FrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                              unsigned FrameReg, StackOffset &Offset,
                              const AArch64InstrInfo *TII);

/// Use to report the frame offset status in isAArch64FrameOffsetLegal.
enum AArch64FrameOffsetStatus {
  AArch64FrameOffsetCannotUpdate = 0x0, ///< Offset cannot apply.
  AArch64FrameOffsetIsLegal = 0x1,      ///< Offset is legal.
  AArch64FrameOffsetCanUpdate = 0x2     ///< Offset can apply, at least partly.
};

/// Check if the @p Offset is a valid frame offset for @p MI.
/// The returned value reports the validity of the frame offset for @p MI.
/// It uses the values defined by AArch64FrameOffsetStatus for that.
/// If result == AArch64FrameOffsetCannotUpdate, @p MI cannot be updated to
/// use an offset.eq
/// If result & AArch64FrameOffsetIsLegal, @p Offset can completely be
/// rewritten in @p MI.
/// If result & AArch64FrameOffsetCanUpdate, @p Offset contains the
/// amount that is off the limit of the legal offset.
/// If set, @p OutUseUnscaledOp will contain the whether @p MI should be
/// turned into an unscaled operator, which opcode is in @p OutUnscaledOp.
/// If set, @p EmittableOffset contains the amount that can be set in @p MI
/// (possibly with @p OutUnscaledOp if OutUseUnscaledOp is true) and that
/// is a legal offset.
int isAArch64FrameOffsetLegal(const MachineInstr &MI, StackOffset &Offset,
                              bool *OutUseUnscaledOp = nullptr,
                              unsigned *OutUnscaledOp = nullptr,
                              int64_t *EmittableOffset = nullptr);

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

static inline bool isIndirectBranchOpcode(int Opc) {
  switch (Opc) {
  case AArch64::BR:
  case AArch64::BRAA:
  case AArch64::BRAB:
  case AArch64::BRAAZ:
  case AArch64::BRABZ:
    return true;
  }
  return false;
}

static inline bool isPTrueOpcode(unsigned Opc) {
  switch (Opc) {
  case AArch64::PTRUE_B:
  case AArch64::PTRUE_H:
  case AArch64::PTRUE_S:
  case AArch64::PTRUE_D:
    return true;
  default:
    return false;
  }
}

/// Return opcode to be used for indirect calls.
unsigned getBLRCallOpcode(const MachineFunction &MF);

// struct TSFlags {
#define TSFLAG_ELEMENT_SIZE_TYPE(X)      (X)       // 3-bits
#define TSFLAG_DESTRUCTIVE_INST_TYPE(X) ((X) << 3) // 4-bits
#define TSFLAG_FALSE_LANE_TYPE(X)       ((X) << 7) // 2-bits
#define TSFLAG_INSTR_FLAGS(X)           ((X) << 9) // 2-bits
// }

namespace AArch64 {

enum ElementSizeType {
  ElementSizeMask = TSFLAG_ELEMENT_SIZE_TYPE(0x7),
  ElementSizeNone = TSFLAG_ELEMENT_SIZE_TYPE(0x0),
  ElementSizeB    = TSFLAG_ELEMENT_SIZE_TYPE(0x1),
  ElementSizeH    = TSFLAG_ELEMENT_SIZE_TYPE(0x2),
  ElementSizeS    = TSFLAG_ELEMENT_SIZE_TYPE(0x3),
  ElementSizeD    = TSFLAG_ELEMENT_SIZE_TYPE(0x4),
};

enum DestructiveInstType {
  DestructiveInstTypeMask       = TSFLAG_DESTRUCTIVE_INST_TYPE(0xf),
  NotDestructive                = TSFLAG_DESTRUCTIVE_INST_TYPE(0x0),
  DestructiveOther              = TSFLAG_DESTRUCTIVE_INST_TYPE(0x1),
  DestructiveUnary              = TSFLAG_DESTRUCTIVE_INST_TYPE(0x2),
  DestructiveBinaryImm          = TSFLAG_DESTRUCTIVE_INST_TYPE(0x3),
  DestructiveBinaryShImmUnpred  = TSFLAG_DESTRUCTIVE_INST_TYPE(0x4),
  DestructiveBinary             = TSFLAG_DESTRUCTIVE_INST_TYPE(0x5),
  DestructiveBinaryComm         = TSFLAG_DESTRUCTIVE_INST_TYPE(0x6),
  DestructiveBinaryCommWithRev  = TSFLAG_DESTRUCTIVE_INST_TYPE(0x7),
  DestructiveTernaryCommWithRev = TSFLAG_DESTRUCTIVE_INST_TYPE(0x8),
  DestructiveUnaryPassthru      = TSFLAG_DESTRUCTIVE_INST_TYPE(0x9),
};

enum FalseLaneType {
  FalseLanesMask  = TSFLAG_FALSE_LANE_TYPE(0x3),
  FalseLanesZero  = TSFLAG_FALSE_LANE_TYPE(0x1),
  FalseLanesUndef = TSFLAG_FALSE_LANE_TYPE(0x2),
};

// NOTE: This is a bit field.
static const uint64_t InstrFlagIsWhile     = TSFLAG_INSTR_FLAGS(0x1);
static const uint64_t InstrFlagIsPTestLike = TSFLAG_INSTR_FLAGS(0x2);

#undef TSFLAG_ELEMENT_SIZE_TYPE
#undef TSFLAG_DESTRUCTIVE_INST_TYPE
#undef TSFLAG_FALSE_LANE_TYPE
#undef TSFLAG_INSTR_FLAGS

int getSVEPseudoMap(uint16_t Opcode);
int getSVERevInstr(uint16_t Opcode);
int getSVENonRevInstr(uint16_t Opcode);
}

} // end namespace llvm

#endif
