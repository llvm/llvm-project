
//===- HexagonInstrInfo.h - Hexagon Instruction Information -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Hexagon implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONINSTRINFO_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONINSTRINFO_H

#include "HexagonRegisterInfo.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "HexagonGenInstrInfo.inc"

namespace llvm {

struct EVT;
class HexagonSubtarget;
class HexagonInstrInfo : public HexagonGenInstrInfo {
  virtual void anchor();
  const HexagonRegisterInfo RI;
  const HexagonSubtarget &Subtarget;

public:
  typedef unsigned Opcode_t;

  explicit HexagonInstrInfo(HexagonSubtarget &ST);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  const HexagonRegisterInfo &getRegisterInfo() const { return RI; }

  /// isLoadFromStackSlot - If the specified machine instruction is a direct
  /// load from a stack slot, return the virtual or physical register number of
  /// the destination along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than loading from the stack slot.
  unsigned isLoadFromStackSlot(const MachineInstr *MI,
                               int &FrameIndex) const override;

  /// isStoreToStackSlot - If the specified machine instruction is a direct
  /// store to a stack slot, return the virtual or physical register number of
  /// the source reg along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than storing to the stack slot.
  unsigned isStoreToStackSlot(const MachineInstr *MI,
                              int &FrameIndex) const override;


  bool AnalyzeBranch(MachineBasicBlock &MBB,MachineBasicBlock *&TBB,
                         MachineBasicBlock *&FBB,
                         SmallVectorImpl<MachineOperand> &Cond,
                         bool AllowModify) const override;

  unsigned RemoveBranch(MachineBasicBlock &MBB) const override;

  unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        DebugLoc DL) const override;

  bool analyzeCompare(const MachineInstr *MI,
                      unsigned &SrcReg, unsigned &SrcReg2,
                      int &Mask, int &Value) const override;

  void copyPhysReg(MachineBasicBlock &MBB,
                   MachineBasicBlock::iterator I, DebugLoc DL,
                   unsigned DestReg, unsigned SrcReg,
                   bool KillSrc) const override;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           unsigned SrcReg, bool isKill, int FrameIndex,
                           const TargetRegisterClass *RC,
                           const TargetRegisterInfo *TRI) const override;

  void storeRegToAddr(MachineFunction &MF, unsigned SrcReg, bool isKill,
                      SmallVectorImpl<MachineOperand> &Addr,
                      const TargetRegisterClass *RC,
                      SmallVectorImpl<MachineInstr*> &NewMIs) const;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI,
                            unsigned DestReg, int FrameIndex,
                            const TargetRegisterClass *RC,
                            const TargetRegisterInfo *TRI) const override;

  void loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                       SmallVectorImpl<MachineOperand> &Addr,
                       const TargetRegisterClass *RC,
                       SmallVectorImpl<MachineInstr*> &NewMIs) const;

  /// expandPostRAPseudo - This function is called for all pseudo instructions
  /// that remain after register allocation. Many pseudo instructions are
  /// created to help register allocation. This is the place to convert them
  /// into real instructions. The target can edit MI in place, or it can insert
  /// new instructions and erase MI. The function should return true if
  /// anything was changed.
  bool expandPostRAPseudo(MachineBasicBlock::iterator MI) const override;

  MachineInstr *foldMemoryOperandImpl(MachineFunction &MF, MachineInstr *MI,
                                      ArrayRef<unsigned> Ops,
                                      MachineBasicBlock::iterator InsertPt,
                                      int FrameIndex) const override;

  MachineInstr *foldMemoryOperandImpl(MachineFunction &MF, MachineInstr *MI,
                                      ArrayRef<unsigned> Ops,
                                      MachineBasicBlock::iterator InsertPt,
                                      MachineInstr *LoadMI) const override {
    return nullptr;
  }

  unsigned createVR(MachineFunction* MF, MVT VT) const;

  bool isBranch(const MachineInstr *MI) const;
  bool isPredicable(MachineInstr *MI) const override;
  bool PredicateInstruction(MachineInstr *MI,
                            ArrayRef<MachineOperand> Cond) const override;

  bool isProfitableToIfCvt(MachineBasicBlock &MBB, unsigned NumCycles,
                           unsigned ExtraPredCycles,
                           BranchProbability Probability) const override;

  bool isProfitableToIfCvt(MachineBasicBlock &TMBB,
                           unsigned NumTCycles, unsigned ExtraTCycles,
                           MachineBasicBlock &FMBB,
                           unsigned NumFCycles, unsigned ExtraFCycles,
                           BranchProbability Probability) const override;

  bool isPredicated(const MachineInstr *MI) const override;
  bool isPredicated(unsigned Opcode) const;
  bool isPredicatedTrue(const MachineInstr *MI) const;
  bool isPredicatedTrue(unsigned Opcode) const;
  bool isPredicatedNew(const MachineInstr *MI) const;
  bool isPredicatedNew(unsigned Opcode) const;
  bool DefinesPredicate(MachineInstr *MI,
                        std::vector<MachineOperand> &Pred) const override;
  bool SubsumesPredicate(ArrayRef<MachineOperand> Pred1,
                         ArrayRef<MachineOperand> Pred2) const override;

  bool
  ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const override;

  bool isProfitableToDupForIfCvt(MachineBasicBlock &MBB, unsigned NumCycles,
                                 BranchProbability Probability) const override;

  DFAPacketizer *
  CreateTargetScheduleState(const TargetSubtargetInfo &STI) const override;

  bool isSchedulingBoundary(const MachineInstr *MI,
                            const MachineBasicBlock *MBB,
                            const MachineFunction &MF) const override;
  bool isValidOffset(unsigned Opcode, int Offset, bool Extend = true) const;
  bool isValidAutoIncImm(const EVT VT, const int Offset) const;
  bool isMemOp(const MachineInstr *MI) const;
  bool isSpillPredRegOp(const MachineInstr *MI) const;
  bool isU6_3Immediate(const int value) const;
  bool isU6_2Immediate(const int value) const;
  bool isU6_1Immediate(const int value) const;
  bool isU6_0Immediate(const int value) const;
  bool isS4_3Immediate(const int value) const;
  bool isS4_2Immediate(const int value) const;
  bool isS4_1Immediate(const int value) const;
  bool isS4_0Immediate(const int value) const;
  bool isS12_Immediate(const int value) const;
  bool isU6_Immediate(const int value) const;
  bool isS8_Immediate(const int value) const;
  bool isS6_Immediate(const int value) const;

  bool isSaveCalleeSavedRegsCall(const MachineInstr* MI) const;
  bool isConditionalTransfer(const MachineInstr* MI) const;
  bool isConditionalALU32 (const MachineInstr* MI) const;
  bool isConditionalLoad (const MachineInstr* MI) const;
  bool isConditionalStore(const MachineInstr* MI) const;
  bool isNewValueInst(const MachineInstr* MI) const;
  bool isNewValue(const MachineInstr* MI) const;
  bool isNewValue(Opcode_t Opcode) const;
  bool isDotNewInst(const MachineInstr* MI) const;
  int GetDotOldOp(const int opc) const;
  int GetDotNewOp(const MachineInstr* MI) const;
  int GetDotNewPredOp(MachineInstr *MI,
                      const MachineBranchProbabilityInfo
                      *MBPI) const;
  bool mayBeNewStore(const MachineInstr* MI) const;
  bool isDeallocRet(const MachineInstr *MI) const;
  unsigned getInvertedPredicatedOpcode(const int Opc) const;
  bool isExtendable(const MachineInstr* MI) const;
  bool isExtended(const MachineInstr* MI) const;
  bool isPostIncrement(const MachineInstr* MI) const;
  bool isNewValueStore(const MachineInstr* MI) const;
  bool isNewValueStore(unsigned Opcode) const;
  bool isNewValueJump(const MachineInstr* MI) const;
  bool isNewValueJump(Opcode_t Opcode) const;
  bool isNewValueJumpCandidate(const MachineInstr *MI) const;


  void immediateExtend(MachineInstr *MI) const;
  bool isConstExtended(const MachineInstr *MI) const;
  unsigned getSize(const MachineInstr *MI) const;  
  int getDotNewPredJumpOp(MachineInstr *MI,
                      const MachineBranchProbabilityInfo *MBPI) const;
  unsigned getAddrMode(const MachineInstr* MI) const;
  bool isOperandExtended(const MachineInstr *MI,
                         unsigned short OperandNum) const;
  unsigned short getCExtOpNum(const MachineInstr *MI) const;
  int getMinValue(const MachineInstr *MI) const;
  int getMaxValue(const MachineInstr *MI) const;
  bool NonExtEquivalentExists (const MachineInstr *MI) const;
  short getNonExtOpcode(const MachineInstr *MI) const;
  bool PredOpcodeHasJMP_c(Opcode_t Opcode) const;
  bool predOpcodeHasNot(ArrayRef<MachineOperand> Cond) const;
  bool isEndLoopN(Opcode_t Opcode) const;
  bool getPredReg(ArrayRef<MachineOperand> Cond, unsigned &PredReg,
                  unsigned &PredRegPos, unsigned &PredRegFlags) const;
  int getCondOpcode(int Opc, bool sense) const;

};

}

#endif
