//===-- ARMBaseInstrInfo.h - ARM Base Instruction Information ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Base ARM implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMBASEINSTRUCTIONINFO_H
#define ARMBASEINSTRUCTIONINFO_H

#include "ARM.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"

#define GET_INSTRINFO_HEADER
#include "ARMGenInstrInfo.inc"

namespace llvm {
  class ARMSubtarget;
  class ARMBaseRegisterInfo;

class ARMBaseInstrInfo : public ARMGenInstrInfo {
  const ARMSubtarget &Subtarget;

protected:
  // Can be only subclassed.
  explicit ARMBaseInstrInfo(const ARMSubtarget &STI);

public:
  // Return whether the target has an explicit NOP encoding.
  bool hasNOP() const;

  // Return the non-pre/post incrementing version of 'Opc'. Return 0
  // if there is not such an opcode.
  virtual unsigned getUnindexedOpcode(unsigned Opc) const =0;

  virtual MachineInstr *convertToThreeAddress(MachineFunction::iterator &MFI,
                                              MachineBasicBlock::iterator &MBBI,
                                              LiveVariables *LV) const;

  virtual const ARMBaseRegisterInfo &getRegisterInfo() const =0;
  const ARMSubtarget &getSubtarget() const { return Subtarget; }

  ScheduleHazardRecognizer *
  CreateTargetHazardRecognizer(const TargetMachine *TM,
                               const ScheduleDAG *DAG) const;

  ScheduleHazardRecognizer *
  CreateTargetPostRAHazardRecognizer(const InstrItineraryData *II,
                                     const ScheduleDAG *DAG) const;

  // Branch analysis.
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             SmallVectorImpl<MachineOperand> &Cond,
                             bool AllowModify = false) const;
  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const;
  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const SmallVectorImpl<MachineOperand> &Cond,
                                DebugLoc DL) const;

  virtual
  bool ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const;

  // Predication support.
  bool isPredicated(const MachineInstr *MI) const;

  ARMCC::CondCodes getPredicate(const MachineInstr *MI) const {
    int PIdx = MI->findFirstPredOperandIdx();
    return PIdx != -1 ? (ARMCC::CondCodes)MI->getOperand(PIdx).getImm()
                      : ARMCC::AL;
  }

  virtual
  bool PredicateInstruction(MachineInstr *MI,
                            const SmallVectorImpl<MachineOperand> &Pred) const;

  virtual
  bool SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                         const SmallVectorImpl<MachineOperand> &Pred2) const;

  virtual bool DefinesPredicate(MachineInstr *MI,
                                std::vector<MachineOperand> &Pred) const;

  virtual bool isPredicable(MachineInstr *MI) const;

  /// GetInstSize - Returns the size of the specified MachineInstr.
  ///
  virtual unsigned GetInstSizeInBytes(const MachineInstr* MI) const;

  virtual unsigned isLoadFromStackSlot(const MachineInstr *MI,
                                       int &FrameIndex) const;
  virtual unsigned isStoreToStackSlot(const MachineInstr *MI,
                                      int &FrameIndex) const;
  virtual unsigned isLoadFromStackSlotPostFE(const MachineInstr *MI,
                                             int &FrameIndex) const;
  virtual unsigned isStoreToStackSlotPostFE(const MachineInstr *MI,
                                            int &FrameIndex) const;

  virtual void copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, DebugLoc DL,
                           unsigned DestReg, unsigned SrcReg,
                           bool KillSrc) const;

  virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   unsigned SrcReg, bool isKill, int FrameIndex,
                                   const TargetRegisterClass *RC,
                                   const TargetRegisterInfo *TRI) const;

  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI,
                                    unsigned DestReg, int FrameIndex,
                                    const TargetRegisterClass *RC,
                                    const TargetRegisterInfo *TRI) const;

  virtual bool expandPostRAPseudo(MachineBasicBlock::iterator MI) const;

  virtual MachineInstr *emitFrameIndexDebugValue(MachineFunction &MF,
                                                 int FrameIx,
                                                 uint64_t Offset,
                                                 const MDNode *MDPtr,
                                                 DebugLoc DL) const;

  virtual void reMaterialize(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MI,
                             unsigned DestReg, unsigned SubIdx,
                             const MachineInstr *Orig,
                             const TargetRegisterInfo &TRI) const;

  MachineInstr *duplicate(MachineInstr *Orig, MachineFunction &MF) const;

  MachineInstr *commuteInstruction(MachineInstr*, bool=false) const;

  virtual bool produceSameValue(const MachineInstr *MI0,
                                const MachineInstr *MI1,
                                const MachineRegisterInfo *MRI) const;

  /// areLoadsFromSameBasePtr - This is used by the pre-regalloc scheduler to
  /// determine if two loads are loading from the same base address. It should
  /// only return true if the base pointers are the same and the only
  /// differences between the two addresses is the offset. It also returns the
  /// offsets by reference.
  virtual bool areLoadsFromSameBasePtr(SDNode *Load1, SDNode *Load2,
                                       int64_t &Offset1, int64_t &Offset2)const;

  /// shouldScheduleLoadsNear - This is a used by the pre-regalloc scheduler to
  /// determine (in conjunction with areLoadsFromSameBasePtr) if two loads
  /// should be scheduled togther. On some targets if two loads are loading from
  /// addresses in the same cache line, it's better if they are scheduled
  /// together. This function takes two integers that represent the load offsets
  /// from the common base address. It returns true if it decides it's desirable
  /// to schedule the two loads together. "NumLoads" is the number of loads that
  /// have already been scheduled after Load1.
  virtual bool shouldScheduleLoadsNear(SDNode *Load1, SDNode *Load2,
                                       int64_t Offset1, int64_t Offset2,
                                       unsigned NumLoads) const;

  virtual bool isSchedulingBoundary(const MachineInstr *MI,
                                    const MachineBasicBlock *MBB,
                                    const MachineFunction &MF) const;

  virtual bool isProfitableToIfCvt(MachineBasicBlock &MBB,
                                   unsigned NumCycles, unsigned ExtraPredCycles,
                                   const BranchProbability &Probability) const;

  virtual bool isProfitableToIfCvt(MachineBasicBlock &TMBB,
                                   unsigned NumT, unsigned ExtraT,
                                   MachineBasicBlock &FMBB,
                                   unsigned NumF, unsigned ExtraF,
                                   const BranchProbability &Probability) const;

  virtual bool isProfitableToDupForIfCvt(MachineBasicBlock &MBB,
                                         unsigned NumCycles,
                                         const BranchProbability
                                         &Probability) const {
    return NumCycles == 1;
  }

  virtual bool isProfitableToUnpredicate(MachineBasicBlock &TMBB,
                                         MachineBasicBlock &FMBB) const;

  /// analyzeCompare - For a comparison instruction, return the source registers
  /// in SrcReg and SrcReg2 if having two register operands, and the value it
  /// compares against in CmpValue. Return true if the comparison instruction
  /// can be analyzed.
  virtual bool analyzeCompare(const MachineInstr *MI, unsigned &SrcReg,
                              unsigned &SrcReg2, int &CmpMask,
                              int &CmpValue) const;

  /// optimizeCompareInstr - Convert the instruction to set the zero flag so
  /// that we can remove a "comparison with zero"; Remove a redundant CMP
  /// instruction if the flags can be updated in the same way by an earlier
  /// instruction such as SUB.
  virtual bool optimizeCompareInstr(MachineInstr *CmpInstr, unsigned SrcReg,
                                    unsigned SrcReg2, int CmpMask, int CmpValue,
                                    const MachineRegisterInfo *MRI) const;

  virtual bool analyzeSelect(const MachineInstr *MI,
                             SmallVectorImpl<MachineOperand> &Cond,
                             unsigned &TrueOp, unsigned &FalseOp,
                             bool &Optimizable) const;

  virtual MachineInstr *optimizeSelect(MachineInstr *MI, bool) const;

  /// FoldImmediate - 'Reg' is known to be defined by a move immediate
  /// instruction, try to fold the immediate into the use instruction.
  virtual bool FoldImmediate(MachineInstr *UseMI, MachineInstr *DefMI,
                             unsigned Reg, MachineRegisterInfo *MRI) const;

  virtual unsigned getNumMicroOps(const InstrItineraryData *ItinData,
                                  const MachineInstr *MI) const;

  virtual
  int getOperandLatency(const InstrItineraryData *ItinData,
                        const MachineInstr *DefMI, unsigned DefIdx,
                        const MachineInstr *UseMI, unsigned UseIdx) const;
  virtual
  int getOperandLatency(const InstrItineraryData *ItinData,
                        SDNode *DefNode, unsigned DefIdx,
                        SDNode *UseNode, unsigned UseIdx) const;

  /// VFP/NEON execution domains.
  std::pair<uint16_t, uint16_t>
  getExecutionDomain(const MachineInstr *MI) const;
  void setExecutionDomain(MachineInstr *MI, unsigned Domain) const;

  unsigned getPartialRegUpdateClearance(const MachineInstr*, unsigned,
                                        const TargetRegisterInfo*) const;
  void breakPartialRegDependency(MachineBasicBlock::iterator, unsigned,
                                 const TargetRegisterInfo *TRI) const;
  /// Get the number of addresses by LDM or VLDM or zero for unknown.
  unsigned getNumLDMAddresses(const MachineInstr *MI) const;

private:
  unsigned getInstBundleLength(const MachineInstr *MI) const;

  int getVLDMDefCycle(const InstrItineraryData *ItinData,
                      const MCInstrDesc &DefMCID,
                      unsigned DefClass,
                      unsigned DefIdx, unsigned DefAlign) const;
  int getLDMDefCycle(const InstrItineraryData *ItinData,
                     const MCInstrDesc &DefMCID,
                     unsigned DefClass,
                     unsigned DefIdx, unsigned DefAlign) const;
  int getVSTMUseCycle(const InstrItineraryData *ItinData,
                      const MCInstrDesc &UseMCID,
                      unsigned UseClass,
                      unsigned UseIdx, unsigned UseAlign) const;
  int getSTMUseCycle(const InstrItineraryData *ItinData,
                     const MCInstrDesc &UseMCID,
                     unsigned UseClass,
                     unsigned UseIdx, unsigned UseAlign) const;
  int getOperandLatency(const InstrItineraryData *ItinData,
                        const MCInstrDesc &DefMCID,
                        unsigned DefIdx, unsigned DefAlign,
                        const MCInstrDesc &UseMCID,
                        unsigned UseIdx, unsigned UseAlign) const;

  unsigned getInstrLatency(const InstrItineraryData *ItinData,
                           const MachineInstr *MI,
                           unsigned *PredCost = 0) const;

  int getInstrLatency(const InstrItineraryData *ItinData,
                      SDNode *Node) const;

  bool hasHighOperandLatency(const InstrItineraryData *ItinData,
                             const MachineRegisterInfo *MRI,
                             const MachineInstr *DefMI, unsigned DefIdx,
                             const MachineInstr *UseMI, unsigned UseIdx) const;
  bool hasLowDefLatency(const InstrItineraryData *ItinData,
                        const MachineInstr *DefMI, unsigned DefIdx) const;

  /// verifyInstruction - Perform target specific instruction verification.
  bool verifyInstruction(const MachineInstr *MI, StringRef &ErrInfo) const;

private:
  /// Modeling special VFP / NEON fp MLA / MLS hazards.

  /// MLxEntryMap - Map fp MLA / MLS to the corresponding entry in the internal
  /// MLx table.
  DenseMap<unsigned, unsigned> MLxEntryMap;

  /// MLxHazardOpcodes - Set of add / sub and multiply opcodes that would cause
  /// stalls when scheduled together with fp MLA / MLS opcodes.
  SmallSet<unsigned, 16> MLxHazardOpcodes;

public:
  /// isFpMLxInstruction - Return true if the specified opcode is a fp MLA / MLS
  /// instruction.
  bool isFpMLxInstruction(unsigned Opcode) const {
    return MLxEntryMap.count(Opcode);
  }

  /// isFpMLxInstruction - This version also returns the multiply opcode and the
  /// addition / subtraction opcode to expand to. Return true for 'HasLane' for
  /// the MLX instructions with an extra lane operand.
  bool isFpMLxInstruction(unsigned Opcode, unsigned &MulOpc,
                          unsigned &AddSubOpc, bool &NegAcc,
                          bool &HasLane) const;

  /// canCauseFpMLxStall - Return true if an instruction of the specified opcode
  /// will cause stalls when scheduled after (within 4-cycle window) a fp
  /// MLA / MLS instruction.
  bool canCauseFpMLxStall(unsigned Opcode) const {
    return MLxHazardOpcodes.count(Opcode);
  }
};

static inline
const MachineInstrBuilder &AddDefaultPred(const MachineInstrBuilder &MIB) {
  return MIB.addImm((int64_t)ARMCC::AL).addReg(0);
}

static inline
const MachineInstrBuilder &AddDefaultCC(const MachineInstrBuilder &MIB) {
  return MIB.addReg(0);
}

static inline
const MachineInstrBuilder &AddDefaultT1CC(const MachineInstrBuilder &MIB,
                                          bool isDead = false) {
  return MIB.addReg(ARM::CPSR, getDefRegState(true) | getDeadRegState(isDead));
}

static inline
const MachineInstrBuilder &AddNoT1CC(const MachineInstrBuilder &MIB) {
  return MIB.addReg(0);
}

static inline
bool isUncondBranchOpcode(int Opc) {
  return Opc == ARM::B || Opc == ARM::tB || Opc == ARM::t2B;
}

static inline
bool isCondBranchOpcode(int Opc) {
  return Opc == ARM::Bcc || Opc == ARM::tBcc || Opc == ARM::t2Bcc;
}

static inline
bool isJumpTableBranchOpcode(int Opc) {
  return Opc == ARM::BR_JTr || Opc == ARM::BR_JTm || Opc == ARM::BR_JTadd ||
    Opc == ARM::tBR_JTr || Opc == ARM::t2BR_JT;
}

static inline
bool isIndirectBranchOpcode(int Opc) {
  return Opc == ARM::BX || Opc == ARM::MOVPCRX || Opc == ARM::tBRIND;
}

/// getInstrPredicate - If instruction is predicated, returns its predicate
/// condition, otherwise returns AL. It also returns the condition code
/// register by reference.
ARMCC::CondCodes getInstrPredicate(const MachineInstr *MI, unsigned &PredReg);

int getMatchingCondBranchOpcode(int Opc);

/// Determine if MI can be folded into an ARM MOVCC instruction, and return the
/// opcode of the SSA instruction representing the conditional MI.
unsigned canFoldARMInstrIntoMOVCC(unsigned Reg,
                                  MachineInstr *&MI,
                                  const MachineRegisterInfo &MRI);

/// Map pseudo instructions that imply an 'S' bit onto real opcodes. Whether
/// the instruction is encoded with an 'S' bit is determined by the optional
/// CPSR def operand.
unsigned convertAddSubFlagsOpcode(unsigned OldOpc);

/// emitARMRegPlusImmediate / emitT2RegPlusImmediate - Emits a series of
/// instructions to materializea destreg = basereg + immediate in ARM / Thumb2
/// code.
void emitARMRegPlusImmediate(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator &MBBI, DebugLoc dl,
                             unsigned DestReg, unsigned BaseReg, int NumBytes,
                             ARMCC::CondCodes Pred, unsigned PredReg,
                             const ARMBaseInstrInfo &TII, unsigned MIFlags = 0);

void emitT2RegPlusImmediate(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator &MBBI, DebugLoc dl,
                            unsigned DestReg, unsigned BaseReg, int NumBytes,
                            ARMCC::CondCodes Pred, unsigned PredReg,
                            const ARMBaseInstrInfo &TII, unsigned MIFlags = 0);
void emitThumbRegPlusImmediate(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator &MBBI, DebugLoc dl,
                               unsigned DestReg, unsigned BaseReg,
                               int NumBytes, const TargetInstrInfo &TII,
                               const ARMBaseRegisterInfo& MRI,
                               unsigned MIFlags = 0);


/// rewriteARMFrameIndex / rewriteT2FrameIndex -
/// Rewrite MI to access 'Offset' bytes from the FP. Return false if the
/// offset could not be handled directly in MI, and return the left-over
/// portion by reference.
bool rewriteARMFrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                          unsigned FrameReg, int &Offset,
                          const ARMBaseInstrInfo &TII);

bool rewriteT2FrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                         unsigned FrameReg, int &Offset,
                         const ARMBaseInstrInfo &TII);

} // End llvm namespace

#endif
