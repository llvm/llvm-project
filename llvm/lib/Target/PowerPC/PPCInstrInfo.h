//===-- PPCInstrInfo.h - PowerPC Instruction Information --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_PPCINSTRINFO_H
#define LLVM_LIB_TARGET_POWERPC_PPCINSTRINFO_H

#include "PPCRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "PPCGenInstrInfo.inc"

namespace llvm {

/// PPCII - This namespace holds all of the PowerPC target-specific
/// per-instruction flags.  These must match the corresponding definitions in
/// PPC.td and PPCInstrFormats.td.
namespace PPCII {
enum {
  // PPC970 Instruction Flags.  These flags describe the characteristics of the
  // PowerPC 970 (aka G5) dispatch groups and how they are formed out of
  // raw machine instructions.

  /// PPC970_First - This instruction starts a new dispatch group, so it will
  /// always be the first one in the group.
  PPC970_First = 0x1,

  /// PPC970_Single - This instruction starts a new dispatch group and
  /// terminates it, so it will be the sole instruction in the group.
  PPC970_Single = 0x2,

  /// PPC970_Cracked - This instruction is cracked into two pieces, requiring
  /// two dispatch pipes to be available to issue.
  PPC970_Cracked = 0x4,

  /// PPC970_Mask/Shift - This is a bitmask that selects the pipeline type that
  /// an instruction is issued to.
  PPC970_Shift = 3,
  PPC970_Mask = 0x07 << PPC970_Shift
};
enum PPC970_Unit {
  /// These are the various PPC970 execution unit pipelines.  Each instruction
  /// is one of these.
  PPC970_Pseudo = 0 << PPC970_Shift,   // Pseudo instruction
  PPC970_FXU    = 1 << PPC970_Shift,   // Fixed Point (aka Integer/ALU) Unit
  PPC970_LSU    = 2 << PPC970_Shift,   // Load Store Unit
  PPC970_FPU    = 3 << PPC970_Shift,   // Floating Point Unit
  PPC970_CRU    = 4 << PPC970_Shift,   // Control Register Unit
  PPC970_VALU   = 5 << PPC970_Shift,   // Vector ALU
  PPC970_VPERM  = 6 << PPC970_Shift,   // Vector Permute Unit
  PPC970_BRU    = 7 << PPC970_Shift    // Branch Unit
};

enum {
  /// Shift count to bypass PPC970 flags
  NewDef_Shift = 6,

  /// This instruction is an X-Form memory operation.
  XFormMemOp = 0x1 << NewDef_Shift,
  /// This instruction is prefixed.
  Prefixed = 0x1 << (NewDef_Shift+1)
};
} // end namespace PPCII

// Instructions that have an immediate form might be convertible to that
// form if the correct input is a result of a load immediate. In order to
// know whether the transformation is special, we might need to know some
// of the details of the two forms.
struct ImmInstrInfo {
  // Is the immediate field in the immediate form signed or unsigned?
  uint64_t SignedImm : 1;
  // Does the immediate need to be a multiple of some value?
  uint64_t ImmMustBeMultipleOf : 5;
  // Is R0/X0 treated specially by the original r+r instruction?
  // If so, in which operand?
  uint64_t ZeroIsSpecialOrig : 3;
  // Is R0/X0 treated specially by the new r+i instruction?
  // If so, in which operand?
  uint64_t ZeroIsSpecialNew : 3;
  // Is the operation commutative?
  uint64_t IsCommutative : 1;
  // The operand number to check for add-immediate def.
  uint64_t OpNoForForwarding : 3;
  // The operand number for the immediate.
  uint64_t ImmOpNo : 3;
  // The opcode of the new instruction.
  uint64_t ImmOpcode : 16;
  // The size of the immediate.
  uint64_t ImmWidth : 5;
  // The immediate should be truncated to N bits.
  uint64_t TruncateImmTo : 5;
  // Is the instruction summing the operand
  uint64_t IsSummingOperands : 1;
};

// Information required to convert an instruction to just a materialized
// immediate.
struct LoadImmediateInfo {
  unsigned Imm : 16;
  unsigned Is64Bit : 1;
  unsigned SetCR : 1;
};

// Index into the OpcodesForSpill array.
enum SpillOpcodeKey {
  SOK_Int4Spill,
  SOK_Int8Spill,
  SOK_Float8Spill,
  SOK_Float4Spill,
  SOK_CRSpill,
  SOK_CRBitSpill,
  SOK_VRVectorSpill,
  SOK_VSXVectorSpill,
  SOK_VectorFloat8Spill,
  SOK_VectorFloat4Spill,
  SOK_VRSaveSpill,
  SOK_QuadFloat8Spill,
  SOK_QuadFloat4Spill,
  SOK_QuadBitSpill,
  SOK_SpillToVSR,
  SOK_SPESpill,
  SOK_LastOpcodeSpill // This must be last on the enum.
};

// Define list of load and store spill opcodes.
#define Pwr8LoadOpcodes                                                        \
  {                                                                            \
    PPC::LWZ, PPC::LD, PPC::LFD, PPC::LFS, PPC::RESTORE_CR,                    \
        PPC::RESTORE_CRBIT, PPC::LVX, PPC::LXVD2X, PPC::LXSDX, PPC::LXSSPX,    \
        PPC::RESTORE_VRSAVE, PPC::QVLFDX, PPC::QVLFSXs, PPC::QVLFDXb,          \
        PPC::SPILLTOVSR_LD, PPC::EVLDD                                         \
  }

#define Pwr9LoadOpcodes                                                        \
  {                                                                            \
    PPC::LWZ, PPC::LD, PPC::LFD, PPC::LFS, PPC::RESTORE_CR,                    \
        PPC::RESTORE_CRBIT, PPC::LVX, PPC::LXV, PPC::DFLOADf64,                \
        PPC::DFLOADf32, PPC::RESTORE_VRSAVE, PPC::QVLFDX, PPC::QVLFSXs,        \
        PPC::QVLFDXb, PPC::SPILLTOVSR_LD                                       \
  }

#define Pwr8StoreOpcodes                                                       \
  {                                                                            \
    PPC::STW, PPC::STD, PPC::STFD, PPC::STFS, PPC::SPILL_CR, PPC::SPILL_CRBIT, \
        PPC::STVX, PPC::STXVD2X, PPC::STXSDX, PPC::STXSSPX, PPC::SPILL_VRSAVE, \
        PPC::QVSTFDX, PPC::QVSTFSXs, PPC::QVSTFDXb, PPC::SPILLTOVSR_ST,        \
        PPC::EVSTDD                                                            \
  }

#define Pwr9StoreOpcodes                                                       \
  {                                                                            \
    PPC::STW, PPC::STD, PPC::STFD, PPC::STFS, PPC::SPILL_CR, PPC::SPILL_CRBIT, \
        PPC::STVX, PPC::STXV, PPC::DFSTOREf64, PPC::DFSTOREf32,                \
        PPC::SPILL_VRSAVE, PPC::QVSTFDX, PPC::QVSTFSXs, PPC::QVSTFDXb,         \
        PPC::SPILLTOVSR_ST                                                     \
  }

// Initialize arrays for load and store spill opcodes on supported subtargets.
#define StoreOpcodesForSpill                                                   \
  { Pwr8StoreOpcodes, Pwr9StoreOpcodes }
#define LoadOpcodesForSpill                                                    \
  { Pwr8LoadOpcodes, Pwr9LoadOpcodes }

class PPCSubtarget;
class PPCInstrInfo : public PPCGenInstrInfo {
  PPCSubtarget &Subtarget;
  const PPCRegisterInfo RI;
  const unsigned StoreSpillOpcodesArray[2][SOK_LastOpcodeSpill] =
      StoreOpcodesForSpill;
  const unsigned LoadSpillOpcodesArray[2][SOK_LastOpcodeSpill] =
      LoadOpcodesForSpill;

  void StoreRegToStackSlot(MachineFunction &MF, unsigned SrcReg, bool isKill,
                           int FrameIdx, const TargetRegisterClass *RC,
                           SmallVectorImpl<MachineInstr *> &NewMIs) const;
  void LoadRegFromStackSlot(MachineFunction &MF, const DebugLoc &DL,
                            unsigned DestReg, int FrameIdx,
                            const TargetRegisterClass *RC,
                            SmallVectorImpl<MachineInstr *> &NewMIs) const;

  // If the inst has imm-form and one of its operand is produced by a LI,
  // put the imm into the inst directly and remove the LI if possible.
  bool transformToImmFormFedByLI(MachineInstr &MI, const ImmInstrInfo &III,
                                 unsigned ConstantOpNo, MachineInstr &DefMI,
                                 int64_t Imm) const;
  // If the inst has imm-form and one of its operand is produced by an
  // add-immediate, try to transform it when possible.
  bool transformToImmFormFedByAdd(MachineInstr &MI, const ImmInstrInfo &III,
                                  unsigned ConstantOpNo, MachineInstr &DefMI,
                                  bool KillDefMI) const;
  // Try to find that, if the instruction 'MI' contains any operand that
  // could be forwarded from some inst that feeds it. If yes, return the
  // Def of that operand. And OpNoForForwarding is the operand index in
  // the 'MI' for that 'Def'. If we see another use of this Def between
  // the Def and the MI, SeenIntermediateUse becomes 'true'.
  MachineInstr *getForwardingDefMI(MachineInstr &MI,
                                   unsigned &OpNoForForwarding,
                                   bool &SeenIntermediateUse) const;

  // Can the user MI have it's source at index \p OpNoForForwarding
  // forwarded from an add-immediate that feeds it?
  bool isUseMIElgibleForForwarding(MachineInstr &MI, const ImmInstrInfo &III,
                                   unsigned OpNoForForwarding) const;
  bool isDefMIElgibleForForwarding(MachineInstr &DefMI,
                                   const ImmInstrInfo &III,
                                   MachineOperand *&ImmMO,
                                   MachineOperand *&RegMO) const;
  bool isImmElgibleForForwarding(const MachineOperand &ImmMO,
                                 const MachineInstr &DefMI,
                                 const ImmInstrInfo &III,
                                 int64_t &Imm) const;
  bool isRegElgibleForForwarding(const MachineOperand &RegMO,
                                 const MachineInstr &DefMI,
                                 const MachineInstr &MI, bool KillDefMI,
                                 bool &IsFwdFeederRegKilled) const;
  unsigned getSpillTarget() const;
  const unsigned *getStoreOpcodesForSpillArray() const;
  const unsigned *getLoadOpcodesForSpillArray() const;
  virtual void anchor();

protected:
  /// Commutes the operands in the given instruction.
  /// The commutable operands are specified by their indices OpIdx1 and OpIdx2.
  ///
  /// Do not call this method for a non-commutable instruction or for
  /// non-commutable pair of operand indices OpIdx1 and OpIdx2.
  /// Even though the instruction is commutable, the method may still
  /// fail to commute the operands, null pointer is returned in such cases.
  ///
  /// For example, we can commute rlwimi instructions, but only if the
  /// rotate amt is zero.  We also have to munge the immediates a bit.
  MachineInstr *commuteInstructionImpl(MachineInstr &MI, bool NewMI,
                                       unsigned OpIdx1,
                                       unsigned OpIdx2) const override;

public:
  explicit PPCInstrInfo(PPCSubtarget &STI);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  const PPCRegisterInfo &getRegisterInfo() const { return RI; }

  bool isXFormMemOp(unsigned Opcode) const {
    return get(Opcode).TSFlags & PPCII::XFormMemOp;
  }
  bool isPrefixed(unsigned Opcode) const {
    return get(Opcode).TSFlags & PPCII::Prefixed;
  }

  static bool isSameClassPhysRegCopy(unsigned Opcode) {
    unsigned CopyOpcodes[] =
      { PPC::OR, PPC::OR8, PPC::FMR, PPC::VOR, PPC::XXLOR, PPC::XXLORf,
        PPC::XSCPSGNDP, PPC::MCRF, PPC::QVFMR, PPC::QVFMRs, PPC::QVFMRb,
        PPC::CROR, PPC::EVOR, -1U };
    for (int i = 0; CopyOpcodes[i] != -1U; i++)
      if (Opcode == CopyOpcodes[i])
        return true;
    return false;
  }

  ScheduleHazardRecognizer *
  CreateTargetHazardRecognizer(const TargetSubtargetInfo *STI,
                               const ScheduleDAG *DAG) const override;
  ScheduleHazardRecognizer *
  CreateTargetPostRAHazardRecognizer(const InstrItineraryData *II,
                                     const ScheduleDAG *DAG) const override;

  unsigned getInstrLatency(const InstrItineraryData *ItinData,
                           const MachineInstr &MI,
                           unsigned *PredCost = nullptr) const override;

  int getOperandLatency(const InstrItineraryData *ItinData,
                        const MachineInstr &DefMI, unsigned DefIdx,
                        const MachineInstr &UseMI,
                        unsigned UseIdx) const override;
  int getOperandLatency(const InstrItineraryData *ItinData,
                        SDNode *DefNode, unsigned DefIdx,
                        SDNode *UseNode, unsigned UseIdx) const override {
    return PPCGenInstrInfo::getOperandLatency(ItinData, DefNode, DefIdx,
                                              UseNode, UseIdx);
  }

  bool hasLowDefLatency(const TargetSchedModel &SchedModel,
                        const MachineInstr &DefMI,
                        unsigned DefIdx) const override {
    // Machine LICM should hoist all instructions in low-register-pressure
    // situations; none are sufficiently free to justify leaving in a loop
    // body.
    return false;
  }

  bool useMachineCombiner() const override {
    return true;
  }

  /// Return true when there is potentially a faster code sequence
  /// for an instruction chain ending in <Root>. All potential patterns are
  /// output in the <Pattern> array.
  bool getMachineCombinerPatterns(
      MachineInstr &Root,
      SmallVectorImpl<MachineCombinerPattern> &P) const override;

  bool isAssociativeAndCommutative(const MachineInstr &Inst) const override;

  void setSpecialOperandAttr(MachineInstr &OldMI1, MachineInstr &OldMI2,
                             MachineInstr &NewMI1,
                             MachineInstr &NewMI2) const override;

  void setSpecialOperandAttr(MachineInstr &MI, uint16_t Flags) const override;

  bool isCoalescableExtInstr(const MachineInstr &MI,
                             Register &SrcReg, Register &DstReg,
                             unsigned &SubIdx) const override;
  unsigned isLoadFromStackSlot(const MachineInstr &MI,
                               int &FrameIndex) const override;
  bool isReallyTriviallyReMaterializable(const MachineInstr &MI,
                                         AAResults *AA) const override;
  unsigned isStoreToStackSlot(const MachineInstr &MI,
                              int &FrameIndex) const override;

  bool findCommutedOpIndices(const MachineInstr &MI, unsigned &SrcOpIdx1,
                             unsigned &SrcOpIdx2) const override;

  void insertNoop(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator MI) const override;


  // Branch analysis.
  bool analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify) const override;
  unsigned removeBranch(MachineBasicBlock &MBB,
                        int *BytesRemoved = nullptr) const override;
  unsigned insertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        const DebugLoc &DL,
                        int *BytesAdded = nullptr) const override;

  // Select analysis.
  bool canInsertSelect(const MachineBasicBlock &, ArrayRef<MachineOperand> Cond,
                       Register, Register, Register, int &, int &,
                       int &) const override;
  void insertSelect(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                    const DebugLoc &DL, Register DstReg,
                    ArrayRef<MachineOperand> Cond, Register TrueReg,
                    Register FalseReg) const override;

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                   const DebugLoc &DL, MCRegister DestReg, MCRegister SrcReg,
                   bool KillSrc) const override;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           Register SrcReg, bool isKill, int FrameIndex,
                           const TargetRegisterClass *RC,
                           const TargetRegisterInfo *TRI) const override;

  // Emits a register spill without updating the register class for vector
  // registers. This ensures that when we spill a vector register the
  // element order in the register is the same as it was in memory.
  void storeRegToStackSlotNoUpd(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI,
                                unsigned SrcReg, bool isKill, int FrameIndex,
                                const TargetRegisterClass *RC,
                                const TargetRegisterInfo *TRI) const;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI,
                            Register DestReg, int FrameIndex,
                            const TargetRegisterClass *RC,
                            const TargetRegisterInfo *TRI) const override;

  // Emits a register reload without updating the register class for vector
  // registers. This ensures that when we reload a vector register the
  // element order in the register is the same as it was in memory.
  void loadRegFromStackSlotNoUpd(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 unsigned DestReg, int FrameIndex,
                                 const TargetRegisterClass *RC,
                                 const TargetRegisterInfo *TRI) const;

  unsigned getStoreOpcodeForSpill(const TargetRegisterClass *RC) const;

  unsigned getLoadOpcodeForSpill(const TargetRegisterClass *RC) const;

  bool
  reverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const override;

  bool FoldImmediate(MachineInstr &UseMI, MachineInstr &DefMI, Register Reg,
                     MachineRegisterInfo *MRI) const override;

  bool onlyFoldImmediate(MachineInstr &UseMI, MachineInstr &DefMI,
                         Register Reg) const;

  // If conversion by predication (only supported by some branch instructions).
  // All of the profitability checks always return true; it is always
  // profitable to use the predicated branches.
  bool isProfitableToIfCvt(MachineBasicBlock &MBB,
                          unsigned NumCycles, unsigned ExtraPredCycles,
                          BranchProbability Probability) const override {
    return true;
  }

  bool isProfitableToIfCvt(MachineBasicBlock &TMBB,
                           unsigned NumT, unsigned ExtraT,
                           MachineBasicBlock &FMBB,
                           unsigned NumF, unsigned ExtraF,
                           BranchProbability Probability) const override;

  bool isProfitableToDupForIfCvt(MachineBasicBlock &MBB, unsigned NumCycles,
                                 BranchProbability Probability) const override {
    return true;
  }

  bool isProfitableToUnpredicate(MachineBasicBlock &TMBB,
                                 MachineBasicBlock &FMBB) const override {
    return false;
  }

  // Predication support.
  bool isPredicated(const MachineInstr &MI) const override;

  bool PredicateInstruction(MachineInstr &MI,
                            ArrayRef<MachineOperand> Pred) const override;

  bool SubsumesPredicate(ArrayRef<MachineOperand> Pred1,
                         ArrayRef<MachineOperand> Pred2) const override;

  bool DefinesPredicate(MachineInstr &MI,
                        std::vector<MachineOperand> &Pred) const override;

  // Comparison optimization.

  bool analyzeCompare(const MachineInstr &MI, Register &SrcReg,
                      Register &SrcReg2, int &Mask, int &Value) const override;

  bool optimizeCompareInstr(MachineInstr &CmpInstr, Register SrcReg,
                            Register SrcReg2, int Mask, int Value,
                            const MachineRegisterInfo *MRI) const override;


  /// Return true if get the base operand, byte offset of an instruction and
  /// the memory width. Width is the size of memory that is being
  /// loaded/stored (e.g. 1, 2, 4, 8).
  bool getMemOperandWithOffsetWidth(const MachineInstr &LdSt,
                                    const MachineOperand *&BaseOp,
                                    int64_t &Offset, unsigned &Width,
                                    const TargetRegisterInfo *TRI) const;

  /// Return true if two MIs access different memory addresses and false
  /// otherwise
  bool
  areMemAccessesTriviallyDisjoint(const MachineInstr &MIa,
                                  const MachineInstr &MIb) const override;

  /// GetInstSize - Return the number of bytes of code the specified
  /// instruction may be.  This returns the maximum number of bytes.
  ///
  unsigned getInstSizeInBytes(const MachineInstr &MI) const override;

  void getNoop(MCInst &NopInst) const override;

  std::pair<unsigned, unsigned>
  decomposeMachineOperandsTargetFlags(unsigned TF) const override;

  ArrayRef<std::pair<unsigned, const char *>>
  getSerializableDirectMachineOperandTargetFlags() const override;

  ArrayRef<std::pair<unsigned, const char *>>
  getSerializableBitmaskMachineOperandTargetFlags() const override;

  // Expand VSX Memory Pseudo instruction to either a VSX or a FP instruction.
  bool expandVSXMemPseudo(MachineInstr &MI) const;

  // Lower pseudo instructions after register allocation.
  bool expandPostRAPseudo(MachineInstr &MI) const override;

  static bool isVFRegister(unsigned Reg) {
    return Reg >= PPC::VF0 && Reg <= PPC::VF31;
  }
  static bool isVRRegister(unsigned Reg) {
    return Reg >= PPC::V0 && Reg <= PPC::V31;
  }
  const TargetRegisterClass *updatedRC(const TargetRegisterClass *RC) const;
  static int getRecordFormOpcode(unsigned Opcode);

  bool isTOCSaveMI(const MachineInstr &MI) const;

  bool isSignOrZeroExtended(const MachineInstr &MI, bool SignExt,
                            const unsigned PhiDepth) const;

  /// Return true if the output of the instruction is always a sign-extended,
  /// i.e. 0 to 31-th bits are same as 32-th bit.
  bool isSignExtended(const MachineInstr &MI, const unsigned depth = 0) const {
    return isSignOrZeroExtended(MI, true, depth);
  }

  /// Return true if the output of the instruction is always zero-extended,
  /// i.e. 0 to 31-th bits are all zeros
  bool isZeroExtended(const MachineInstr &MI, const unsigned depth = 0) const {
   return isSignOrZeroExtended(MI, false, depth);
  }

  bool convertToImmediateForm(MachineInstr &MI,
                              MachineInstr **KilledDef = nullptr) const;
  bool foldFrameOffset(MachineInstr &MI) const;
  bool isADDIInstrEligibleForFolding(MachineInstr &ADDIMI, int64_t &Imm) const;
  bool isADDInstrEligibleForFolding(MachineInstr &ADDMI) const;
  bool isImmInstrEligibleForFolding(MachineInstr &MI, unsigned &BaseReg,
                                    unsigned &XFormOpcode,
                                    int64_t &OffsetOfImmInstr,
                                    ImmInstrInfo &III) const;
  bool isValidToBeChangedReg(MachineInstr *ADDMI, unsigned Index,
                             MachineInstr *&ADDIMI, int64_t &OffsetAddi,
                             int64_t OffsetImm) const;

  /// Fixup killed/dead flag for register \p RegNo between instructions [\p
  /// StartMI, \p EndMI]. Some PostRA transformations may violate register
  /// killed/dead flags semantics, this function can be called to fix up. Before
  /// calling this function,
  /// 1. Ensure that \p RegNo liveness is killed after instruction \p EndMI.
  /// 2. Ensure that there is no new definition between (\p StartMI, \p EndMI)
  ///    and possible definition for \p RegNo is \p StartMI or \p EndMI.
  /// 3. Ensure that all instructions between [\p StartMI, \p EndMI] are in same
  ///    basic block.
  void fixupIsDeadOrKill(MachineInstr &StartMI, MachineInstr &EndMI,
                         unsigned RegNo) const;
  void replaceInstrWithLI(MachineInstr &MI, const LoadImmediateInfo &LII) const;
  void replaceInstrOperandWithImm(MachineInstr &MI, unsigned OpNo,
                                  int64_t Imm) const;

  bool instrHasImmForm(unsigned Opc, bool IsVFReg, ImmInstrInfo &III,
                       bool PostRA) const;

  // In PostRA phase, try to find instruction defines \p Reg before \p MI.
  // \p SeenIntermediate is set to true if uses between DefMI and \p MI exist.
  MachineInstr *getDefMIPostRA(unsigned Reg, MachineInstr &MI,
                               bool &SeenIntermediateUse) const;

  /// getRegNumForOperand - some operands use different numbering schemes
  /// for the same registers. For example, a VSX instruction may have any of
  /// vs0-vs63 allocated whereas an Altivec instruction could only have
  /// vs32-vs63 allocated (numbered as v0-v31). This function returns the actual
  /// register number needed for the opcode/operand number combination.
  /// The operand number argument will be useful when we need to extend this
  /// to instructions that use both Altivec and VSX numbering (for different
  /// operands).
  static unsigned getRegNumForOperand(const MCInstrDesc &Desc, unsigned Reg,
                                      unsigned OpNo) {
    int16_t regClass = Desc.OpInfo[OpNo].RegClass;
    switch (regClass) {
      // We store F0-F31, VF0-VF31 in MCOperand and it should be F0-F31,
      // VSX32-VSX63 during encoding/disassembling
      case PPC::VSSRCRegClassID:
      case PPC::VSFRCRegClassID:
        if (isVFRegister(Reg))
          return PPC::VSX32 + (Reg - PPC::VF0);
        break;
      // We store VSL0-VSL31, V0-V31 in MCOperand and it should be VSL0-VSL31,
      // VSX32-VSX63 during encoding/disassembling
      case PPC::VSRCRegClassID:
        if (isVRRegister(Reg))
          return PPC::VSX32 + (Reg - PPC::V0);
        break;
      // Other RegClass doesn't need mapping
      default:
        break;
    }
    return Reg;
  }

  /// Check \p Opcode is BDNZ (Decrement CTR and branch if it is still nonzero).
  bool isBDNZ(unsigned Opcode) const;

  /// Find the hardware loop instruction used to set-up the specified loop.
  /// On PPC, we have two instructions used to set-up the hardware loop
  /// (MTCTRloop, MTCTR8loop) with corresponding endloop (BDNZ, BDNZ8)
  /// instructions to indicate the end of a loop.
  MachineInstr *
  findLoopInstr(MachineBasicBlock &PreHeader,
                SmallPtrSet<MachineBasicBlock *, 8> &Visited) const;

  /// Analyze loop L, which must be a single-basic-block loop, and if the
  /// conditions can be understood enough produce a PipelinerLoopInfo object.
  std::unique_ptr<TargetInstrInfo::PipelinerLoopInfo>
  analyzeLoopForPipelining(MachineBasicBlock *LoopBB) const override;
};

}

#endif
