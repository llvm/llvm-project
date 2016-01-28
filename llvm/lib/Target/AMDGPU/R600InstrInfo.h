//===-- R600InstrInfo.h - R600 Instruction Info Interface -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Interface definition for R600InstrInfo
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_R600_R600INSTRINFO_H
#define LLVM_LIB_TARGET_R600_R600INSTRINFO_H

#include "AMDGPUInstrInfo.h"
#include "R600Defines.h"
#include "R600RegisterInfo.h"
#include <map>

namespace llvm {

  class AMDGPUTargetMachine;
  class DFAPacketizer;
  class ScheduleDAG;
  class MachineFunction;
  class MachineInstr;
  class MachineInstrBuilder;

  class R600InstrInfo : public AMDGPUInstrInfo {
  private:
  const R600RegisterInfo RI;

  std::vector<std::pair<int, unsigned> >
  ExtractSrcs(MachineInstr *MI, const DenseMap<unsigned, unsigned> &PV, unsigned &ConstCount) const;


  MachineInstrBuilder buildIndirectRead(MachineBasicBlock *MBB,
                                        MachineBasicBlock::iterator I,
                                        unsigned ValueReg, unsigned Address,
                                        unsigned OffsetReg,
                                        unsigned AddrChan) const;

  MachineInstrBuilder buildIndirectWrite(MachineBasicBlock *MBB,
                                        MachineBasicBlock::iterator I,
                                        unsigned ValueReg, unsigned Address,
                                        unsigned OffsetReg,
                                        unsigned AddrChan) const;
  public:
  enum BankSwizzle {
    ALU_VEC_012_SCL_210 = 0,
    ALU_VEC_021_SCL_122,
    ALU_VEC_120_SCL_212,
    ALU_VEC_102_SCL_221,
    ALU_VEC_201,
    ALU_VEC_210
  };

  explicit R600InstrInfo(const AMDGPUSubtarget &st);

  const R600RegisterInfo &getRegisterInfo() const override;
  void copyPhysReg(MachineBasicBlock &MBB,
                   MachineBasicBlock::iterator MI, DebugLoc DL,
                   unsigned DestReg, unsigned SrcReg,
                   bool KillSrc) const override;
  bool isLegalToSplitMBBAt(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI) const override;

  bool isTrig(const MachineInstr &MI) const;
  bool isPlaceHolderOpcode(unsigned opcode) const;
  bool isReductionOp(unsigned opcode) const;
  bool isCubeOp(unsigned opcode) const;

  /// \returns true if this \p Opcode represents an ALU instruction.
  bool isALUInstr(unsigned Opcode) const;
  bool hasInstrModifiers(unsigned Opcode) const;
  bool isLDSInstr(unsigned Opcode) const;
  bool isLDSNoRetInstr(unsigned Opcode) const;
  bool isLDSRetInstr(unsigned Opcode) const;

  /// \returns true if this \p Opcode represents an ALU instruction or an
  /// instruction that will be lowered in ExpandSpecialInstrs Pass.
  bool canBeConsideredALU(const MachineInstr *MI) const;

  bool isTransOnly(unsigned Opcode) const;
  bool isTransOnly(const MachineInstr *MI) const;
  bool isVectorOnly(unsigned Opcode) const;
  bool isVectorOnly(const MachineInstr *MI) const;
  bool isExport(unsigned Opcode) const;

  bool usesVertexCache(unsigned Opcode) const;
  bool usesVertexCache(const MachineInstr *MI) const;
  bool usesTextureCache(unsigned Opcode) const;
  bool usesTextureCache(const MachineInstr *MI) const;

  bool mustBeLastInClause(unsigned Opcode) const;
  bool usesAddressRegister(MachineInstr *MI) const;
  bool definesAddressRegister(MachineInstr *MI) const;
  bool readsLDSSrcReg(const MachineInstr *MI) const;

  /// \returns The operand index for the given source number.  Legal values
  /// for SrcNum are 0, 1, and 2.
  int getSrcIdx(unsigned Opcode, unsigned SrcNum) const;
  /// \returns The operand Index for the Sel operand given an index to one
  /// of the instruction's src operands.
  int getSelIdx(unsigned Opcode, unsigned SrcIdx) const;

  /// \returns a pair for each src of an ALU instructions.
  /// The first member of a pair is the register id.
  /// If register is ALU_CONST, second member is SEL.
  /// If register is ALU_LITERAL, second member is IMM.
  /// Otherwise, second member value is undefined.
  SmallVector<std::pair<MachineOperand *, int64_t>, 3>
      getSrcs(MachineInstr *MI) const;

  unsigned  isLegalUpTo(
    const std::vector<std::vector<std::pair<int, unsigned> > > &IGSrcs,
    const std::vector<R600InstrInfo::BankSwizzle> &Swz,
    const std::vector<std::pair<int, unsigned> > &TransSrcs,
    R600InstrInfo::BankSwizzle TransSwz) const;

  bool FindSwizzleForVectorSlot(
    const std::vector<std::vector<std::pair<int, unsigned> > > &IGSrcs,
    std::vector<R600InstrInfo::BankSwizzle> &SwzCandidate,
    const std::vector<std::pair<int, unsigned> > &TransSrcs,
    R600InstrInfo::BankSwizzle TransSwz) const;

  /// Given the order VEC_012 < VEC_021 < VEC_120 < VEC_102 < VEC_201 < VEC_210
  /// returns true and the first (in lexical order) BankSwizzle affectation
  /// starting from the one already provided in the Instruction Group MIs that
  /// fits Read Port limitations in BS if available. Otherwise returns false
  /// and undefined content in BS.
  /// isLastAluTrans should be set if the last Alu of MIs will be executed on
  /// Trans ALU. In this case, ValidTSwizzle returns the BankSwizzle value to
  /// apply to the last instruction.
  /// PV holds GPR to PV registers in the Instruction Group MIs.
  bool fitsReadPortLimitations(const std::vector<MachineInstr *> &MIs,
                               const DenseMap<unsigned, unsigned> &PV,
                               std::vector<BankSwizzle> &BS,
                               bool isLastAluTrans) const;

  /// An instruction group can only access 2 channel pair (either [XY] or [ZW])
  /// from KCache bank on R700+. This function check if MI set in input meet
  /// this limitations
  bool fitsConstReadLimitations(const std::vector<MachineInstr *> &) const;
  /// Same but using const index set instead of MI set.
  bool fitsConstReadLimitations(const std::vector<unsigned>&) const;

  /// \brief Vector instructions are instructions that must fill all
  /// instruction slots within an instruction group.
  bool isVector(const MachineInstr &MI) const;

  bool isMov(unsigned Opcode) const;

  DFAPacketizer *
  CreateTargetScheduleState(const TargetSubtargetInfo &) const override;

  bool ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const override;

  bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB, MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond, bool AllowModify) const override;

  unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        DebugLoc DL) const override;

  unsigned RemoveBranch(MachineBasicBlock &MBB) const override;

  bool isPredicated(const MachineInstr *MI) const override;

  bool isPredicable(MachineInstr *MI) const override;

  bool
   isProfitableToDupForIfCvt(MachineBasicBlock &MBB, unsigned NumCyles,
                             BranchProbability Probability) const override;

  bool isProfitableToIfCvt(MachineBasicBlock &MBB, unsigned NumCyles,
                           unsigned ExtraPredCycles,
                           BranchProbability Probability) const override ;

  bool
   isProfitableToIfCvt(MachineBasicBlock &TMBB,
                       unsigned NumTCycles, unsigned ExtraTCycles,
                       MachineBasicBlock &FMBB,
                       unsigned NumFCycles, unsigned ExtraFCycles,
                       BranchProbability Probability) const override;

  bool DefinesPredicate(MachineInstr *MI,
                                  std::vector<MachineOperand> &Pred) const override;

  bool SubsumesPredicate(ArrayRef<MachineOperand> Pred1,
                         ArrayRef<MachineOperand> Pred2) const override;

  bool isProfitableToUnpredicate(MachineBasicBlock &TMBB,
                                          MachineBasicBlock &FMBB) const override;

  bool PredicateInstruction(MachineInstr *MI,
                            ArrayRef<MachineOperand> Pred) const override;

  unsigned int getPredicationCost(const MachineInstr *) const override;

  unsigned int getInstrLatency(const InstrItineraryData *ItinData,
                               const MachineInstr *MI,
                               unsigned *PredCost = nullptr) const override;

  int getInstrLatency(const InstrItineraryData *ItinData,
                      SDNode *Node) const override { return 1;}

  bool expandPostRAPseudo(MachineBasicBlock::iterator MI) const override;

  /// \brief Reserve the registers that may be accesed using indirect addressing.
  void reserveIndirectRegisters(BitVector &Reserved,
                                const MachineFunction &MF) const;

  unsigned calculateIndirectAddress(unsigned RegIndex,
                                    unsigned Channel) const override;

  const TargetRegisterClass *getIndirectAddrRegClass() const override;

  MachineInstrBuilder buildIndirectWrite(MachineBasicBlock *MBB,
                          MachineBasicBlock::iterator I,
                          unsigned ValueReg, unsigned Address,
                          unsigned OffsetReg) const override;

  MachineInstrBuilder buildIndirectRead(MachineBasicBlock *MBB,
                                        MachineBasicBlock::iterator I,
                                        unsigned ValueReg, unsigned Address,
                                        unsigned OffsetReg) const override;

  unsigned getMaxAlusPerClause() const;

  ///buildDefaultInstruction - This function returns a MachineInstr with
  /// all the instruction modifiers initialized to their default values.
  /// You can use this function to avoid manually specifying each instruction
  /// modifier operand when building a new instruction.
  ///
  /// \returns a MachineInstr with all the instruction modifiers initialized
  /// to their default values.
  MachineInstrBuilder buildDefaultInstruction(MachineBasicBlock &MBB,
                                              MachineBasicBlock::iterator I,
                                              unsigned Opcode,
                                              unsigned DstReg,
                                              unsigned Src0Reg,
                                              unsigned Src1Reg = 0) const;

  MachineInstr *buildSlotOfVectorInstruction(MachineBasicBlock &MBB,
                                             MachineInstr *MI,
                                             unsigned Slot,
                                             unsigned DstReg) const;

  MachineInstr *buildMovImm(MachineBasicBlock &BB,
                                  MachineBasicBlock::iterator I,
                                  unsigned DstReg,
                                  uint64_t Imm) const;

  MachineInstr *buildMovInstr(MachineBasicBlock *MBB,
                              MachineBasicBlock::iterator I,
                              unsigned DstReg, unsigned SrcReg) const;

  /// \brief Get the index of Op in the MachineInstr.
  ///
  /// \returns -1 if the Instruction does not contain the specified \p Op.
  int getOperandIdx(const MachineInstr &MI, unsigned Op) const;

  /// \brief Get the index of \p Op for the given Opcode.
  ///
  /// \returns -1 if the Instruction does not contain the specified \p Op.
  int getOperandIdx(unsigned Opcode, unsigned Op) const;

  /// \brief Helper function for setting instruction flag values.
  void setImmOperand(MachineInstr *MI, unsigned Op, int64_t Imm) const;

  /// \returns true if this instruction has an operand for storing target flags.
  bool hasFlagOperand(const MachineInstr &MI) const;

  ///\brief Add one of the MO_FLAG* flags to the specified \p Operand.
  void addFlag(MachineInstr *MI, unsigned Operand, unsigned Flag) const;

  ///\brief Determine if the specified \p Flag is set on this \p Operand.
  bool isFlagSet(const MachineInstr &MI, unsigned Operand, unsigned Flag) const;

  /// \param SrcIdx The register source to set the flag on (e.g src0, src1, src2)
  /// \param Flag The flag being set.
  ///
  /// \returns the operand containing the flags for this instruction.
  MachineOperand &getFlagOp(MachineInstr *MI, unsigned SrcIdx = 0,
                            unsigned Flag = 0) const;

  /// \brief Clear the specified flag on the instruction.
  void clearFlag(MachineInstr *MI, unsigned Operand, unsigned Flag) const;

  // Helper functions that check the opcode for status information
  bool isRegisterStore(const MachineInstr &MI) const;
  bool isRegisterLoad(const MachineInstr &MI) const;

};

namespace AMDGPU {

int getLDSNoRetOp(uint16_t Opcode);

} //End namespace AMDGPU

} // End llvm namespace

#endif
