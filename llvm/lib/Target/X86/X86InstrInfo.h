//===-- X86InstrInfo.h - X86 Instruction Information ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86INSTRUCTIONINFO_H
#define X86INSTRUCTIONINFO_H

#include "X86.h"
#include "X86RegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Target/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "X86GenInstrInfo.inc"

namespace llvm {
  class X86RegisterInfo;
  class X86TargetMachine;

namespace X86 {
  // X86 specific condition code. These correspond to X86_*_COND in
  // X86InstrInfo.td. They must be kept in synch.
  enum CondCode {
    COND_A  = 0,
    COND_AE = 1,
    COND_B  = 2,
    COND_BE = 3,
    COND_E  = 4,
    COND_G  = 5,
    COND_GE = 6,
    COND_L  = 7,
    COND_LE = 8,
    COND_NE = 9,
    COND_NO = 10,
    COND_NP = 11,
    COND_NS = 12,
    COND_O  = 13,
    COND_P  = 14,
    COND_S  = 15,

    // Artificial condition codes. These are used by AnalyzeBranch
    // to indicate a block terminated with two conditional branches to
    // the same location. This occurs in code using FCMP_OEQ or FCMP_UNE,
    // which can't be represented on x86 with a single condition. These
    // are never used in MachineInstrs.
    COND_NE_OR_P,
    COND_NP_OR_E,

    COND_INVALID
  };

  // Turn condition code into conditional branch opcode.
  unsigned GetCondBranchFromCond(CondCode CC);

  // Turn CMov opcode into condition code.
  CondCode getCondFromCMovOpc(unsigned Opc);

  /// GetOppositeBranchCondition - Return the inverse of the specified cond,
  /// e.g. turning COND_E to COND_NE.
  CondCode GetOppositeBranchCondition(X86::CondCode CC);
}  // end namespace X86;


/// isGlobalStubReference - Return true if the specified TargetFlag operand is
/// a reference to a stub for a global, not the global itself.
inline static bool isGlobalStubReference(unsigned char TargetFlag) {
  switch (TargetFlag) {
  case X86II::MO_DLLIMPORT: // dllimport stub.
  case X86II::MO_GOTPCREL:  // rip-relative GOT reference.
  case X86II::MO_GOT:       // normal GOT reference.
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE:        // Normal $non_lazy_ptr ref.
  case X86II::MO_DARWIN_NONLAZY:                 // Normal $non_lazy_ptr ref.
  case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE: // Hidden $non_lazy_ptr ref.
    return true;
  default:
    return false;
  }
}

/// isGlobalRelativeToPICBase - Return true if the specified global value
/// reference is relative to a 32-bit PIC base (X86ISD::GlobalBaseReg).  If this
/// is true, the addressing mode has the PIC base register added in (e.g. EBX).
inline static bool isGlobalRelativeToPICBase(unsigned char TargetFlag) {
  switch (TargetFlag) {
  case X86II::MO_GOTOFF:                         // isPICStyleGOT: local global.
  case X86II::MO_GOT:                            // isPICStyleGOT: other global.
  case X86II::MO_PIC_BASE_OFFSET:                // Darwin local global.
  case X86II::MO_DARWIN_NONLAZY_PIC_BASE:        // Darwin/32 external global.
  case X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE: // Darwin/32 hidden global.
  case X86II::MO_TLVP:                           // ??? Pretty sure..
    return true;
  default:
    return false;
  }
}

inline static bool isScale(const MachineOperand &MO) {
  return MO.isImm() &&
    (MO.getImm() == 1 || MO.getImm() == 2 ||
     MO.getImm() == 4 || MO.getImm() == 8);
}

inline static bool isLeaMem(const MachineInstr *MI, unsigned Op) {
  if (MI->getOperand(Op).isFI()) return true;
  return Op+4 <= MI->getNumOperands() &&
    MI->getOperand(Op  ).isReg() && isScale(MI->getOperand(Op+1)) &&
    MI->getOperand(Op+2).isReg() &&
    (MI->getOperand(Op+3).isImm() ||
     MI->getOperand(Op+3).isGlobal() ||
     MI->getOperand(Op+3).isCPI() ||
     MI->getOperand(Op+3).isJTI());
}

inline static bool isMem(const MachineInstr *MI, unsigned Op) {
  if (MI->getOperand(Op).isFI()) return true;
  return Op+5 <= MI->getNumOperands() &&
    MI->getOperand(Op+4).isReg() &&
    isLeaMem(MI, Op);
}

class X86InstrInfo : public X86GenInstrInfo {
  X86TargetMachine &TM;
  const X86RegisterInfo RI;

  /// RegOp2MemOpTable3Addr, RegOp2MemOpTable0, RegOp2MemOpTable1,
  /// RegOp2MemOpTable2, RegOp2MemOpTable3 - Load / store folding opcode maps.
  ///
  typedef DenseMap<unsigned,
                   std::pair<unsigned, unsigned> > RegOp2MemOpTableType;
  RegOp2MemOpTableType RegOp2MemOpTable2Addr;
  RegOp2MemOpTableType RegOp2MemOpTable0;
  RegOp2MemOpTableType RegOp2MemOpTable1;
  RegOp2MemOpTableType RegOp2MemOpTable2;
  RegOp2MemOpTableType RegOp2MemOpTable3;

  /// MemOp2RegOpTable - Load / store unfolding opcode map.
  ///
  typedef DenseMap<unsigned,
                   std::pair<unsigned, unsigned> > MemOp2RegOpTableType;
  MemOp2RegOpTableType MemOp2RegOpTable;

  static void AddTableEntry(RegOp2MemOpTableType &R2MTable,
                            MemOp2RegOpTableType &M2RTable,
                            unsigned RegOp, unsigned MemOp, unsigned Flags);

public:
  explicit X86InstrInfo(X86TargetMachine &tm);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const X86RegisterInfo &getRegisterInfo() const { return RI; }

  /// isCoalescableExtInstr - Return true if the instruction is a "coalescable"
  /// extension instruction. That is, it's like a copy where it's legal for the
  /// source to overlap the destination. e.g. X86::MOVSX64rr32. If this returns
  /// true, then it's expected the pre-extension value is available as a subreg
  /// of the result register. This also returns the sub-register index in
  /// SubIdx.
  virtual bool isCoalescableExtInstr(const MachineInstr &MI,
                                     unsigned &SrcReg, unsigned &DstReg,
                                     unsigned &SubIdx) const;

  unsigned isLoadFromStackSlot(const MachineInstr *MI, int &FrameIndex) const;
  /// isLoadFromStackSlotPostFE - Check for post-frame ptr elimination
  /// stack locations as well.  This uses a heuristic so it isn't
  /// reliable for correctness.
  unsigned isLoadFromStackSlotPostFE(const MachineInstr *MI,
                                     int &FrameIndex) const;

  unsigned isStoreToStackSlot(const MachineInstr *MI, int &FrameIndex) const;
  /// isStoreToStackSlotPostFE - Check for post-frame ptr elimination
  /// stack locations as well.  This uses a heuristic so it isn't
  /// reliable for correctness.
  unsigned isStoreToStackSlotPostFE(const MachineInstr *MI,
                                    int &FrameIndex) const;

  bool isReallyTriviallyReMaterializable(const MachineInstr *MI,
                                         AliasAnalysis *AA) const;
  void reMaterialize(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                     unsigned DestReg, unsigned SubIdx,
                     const MachineInstr *Orig,
                     const TargetRegisterInfo &TRI) const;

  /// convertToThreeAddress - This method must be implemented by targets that
  /// set the M_CONVERTIBLE_TO_3_ADDR flag.  When this flag is set, the target
  /// may be able to convert a two-address instruction into a true
  /// three-address instruction on demand.  This allows the X86 target (for
  /// example) to convert ADD and SHL instructions into LEA instructions if they
  /// would require register copies due to two-addressness.
  ///
  /// This method returns a null pointer if the transformation cannot be
  /// performed, otherwise it returns the new instruction.
  ///
  virtual MachineInstr *convertToThreeAddress(MachineFunction::iterator &MFI,
                                              MachineBasicBlock::iterator &MBBI,
                                              LiveVariables *LV) const;

  /// commuteInstruction - We have a few instructions that must be hacked on to
  /// commute them.
  ///
  virtual MachineInstr *commuteInstruction(MachineInstr *MI, bool NewMI) const;

  // Branch analysis.
  virtual bool isUnpredicatedTerminator(const MachineInstr* MI) const;
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             SmallVectorImpl<MachineOperand> &Cond,
                             bool AllowModify) const;
  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const;
  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const SmallVectorImpl<MachineOperand> &Cond,
                                DebugLoc DL) const;
  virtual bool canInsertSelect(const MachineBasicBlock&,
                               const SmallVectorImpl<MachineOperand> &Cond,
                               unsigned, unsigned, int&, int&, int&) const;
  virtual void insertSelect(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI, DebugLoc DL,
                            unsigned DstReg,
                            const SmallVectorImpl<MachineOperand> &Cond,
                            unsigned TrueReg, unsigned FalseReg) const;
  virtual void copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, DebugLoc DL,
                           unsigned DestReg, unsigned SrcReg,
                           bool KillSrc) const;
  virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned SrcReg, bool isKill, int FrameIndex,
                                   const TargetRegisterClass *RC,
                                   const TargetRegisterInfo *TRI) const;

  virtual void storeRegToAddr(MachineFunction &MF, unsigned SrcReg, bool isKill,
                              SmallVectorImpl<MachineOperand> &Addr,
                              const TargetRegisterClass *RC,
                              MachineInstr::mmo_iterator MMOBegin,
                              MachineInstr::mmo_iterator MMOEnd,
                              SmallVectorImpl<MachineInstr*> &NewMIs) const;

  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
                                    unsigned DestReg, int FrameIndex,
                                    const TargetRegisterClass *RC,
                                    const TargetRegisterInfo *TRI) const;

  virtual void loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                               SmallVectorImpl<MachineOperand> &Addr,
                               const TargetRegisterClass *RC,
                               MachineInstr::mmo_iterator MMOBegin,
                               MachineInstr::mmo_iterator MMOEnd,
                               SmallVectorImpl<MachineInstr*> &NewMIs) const;

  virtual bool expandPostRAPseudo(MachineBasicBlock::iterator MI) const;

  virtual
  MachineInstr *emitFrameIndexDebugValue(MachineFunction &MF,
                                         int FrameIx, uint64_t Offset,
                                         const MDNode *MDPtr,
                                         DebugLoc DL) const;

  /// foldMemoryOperand - If this target supports it, fold a load or store of
  /// the specified stack slot into the specified machine instruction for the
  /// specified operand(s).  If this is possible, the target should perform the
  /// folding and return true, otherwise it should return false.  If it folds
  /// the instruction, it is likely that the MachineInstruction the iterator
  /// references has been changed.
  virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                              MachineInstr* MI,
                                           const SmallVectorImpl<unsigned> &Ops,
                                              int FrameIndex) const;

  /// foldMemoryOperand - Same as the previous version except it allows folding
  /// of any load and store from / to any address, not just from a specific
  /// stack slot.
  virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                              MachineInstr* MI,
                                           const SmallVectorImpl<unsigned> &Ops,
                                              MachineInstr* LoadMI) const;

  /// canFoldMemoryOperand - Returns true if the specified load / store is
  /// folding is possible.
  virtual bool canFoldMemoryOperand(const MachineInstr*,
                                    const SmallVectorImpl<unsigned> &) const;

  /// unfoldMemoryOperand - Separate a single instruction which folded a load or
  /// a store or a load and a store into two or more instruction. If this is
  /// possible, returns true as well as the new instructions by reference.
  virtual bool unfoldMemoryOperand(MachineFunction &MF, MachineInstr *MI,
                           unsigned Reg, bool UnfoldLoad, bool UnfoldStore,
                           SmallVectorImpl<MachineInstr*> &NewMIs) const;

  virtual bool unfoldMemoryOperand(SelectionDAG &DAG, SDNode *N,
                           SmallVectorImpl<SDNode*> &NewNodes) const;

  /// getOpcodeAfterMemoryUnfold - Returns the opcode of the would be new
  /// instruction after load / store are unfolded from an instruction of the
  /// specified opcode. It returns zero if the specified unfolding is not
  /// possible. If LoadRegIndex is non-null, it is filled in with the operand
  /// index of the operand which will hold the register holding the loaded
  /// value.
  virtual unsigned getOpcodeAfterMemoryUnfold(unsigned Opc,
                                      bool UnfoldLoad, bool UnfoldStore,
                                      unsigned *LoadRegIndex = 0) const;

  /// areLoadsFromSameBasePtr - This is used by the pre-regalloc scheduler
  /// to determine if two loads are loading from the same base address. It
  /// should only return true if the base pointers are the same and the
  /// only differences between the two addresses are the offset. It also returns
  /// the offsets by reference.
  virtual bool areLoadsFromSameBasePtr(SDNode *Load1, SDNode *Load2,
                                       int64_t &Offset1, int64_t &Offset2) const;

  /// shouldScheduleLoadsNear - This is a used by the pre-regalloc scheduler to
  /// determine (in conjunction with areLoadsFromSameBasePtr) if two loads should
  /// be scheduled togther. On some targets if two loads are loading from
  /// addresses in the same cache line, it's better if they are scheduled
  /// together. This function takes two integers that represent the load offsets
  /// from the common base address. It returns true if it decides it's desirable
  /// to schedule the two loads together. "NumLoads" is the number of loads that
  /// have already been scheduled after Load1.
  virtual bool shouldScheduleLoadsNear(SDNode *Load1, SDNode *Load2,
                                       int64_t Offset1, int64_t Offset2,
                                       unsigned NumLoads) const;

  virtual void getNoopForMachoTarget(MCInst &NopInst) const;

  virtual
  bool ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const;

  /// isSafeToMoveRegClassDefs - Return true if it's safe to move a machine
  /// instruction that defines the specified register class.
  bool isSafeToMoveRegClassDefs(const TargetRegisterClass *RC) const;

  static bool isX86_64ExtendedReg(const MachineOperand &MO) {
    if (!MO.isReg()) return false;
    return X86II::isX86_64ExtendedReg(MO.getReg());
  }

  /// getGlobalBaseReg - Return a virtual register initialized with the
  /// the global base register value. Output instructions required to
  /// initialize the register in the function entry block, if necessary.
  ///
  unsigned getGlobalBaseReg(MachineFunction *MF) const;

  std::pair<uint16_t, uint16_t>
  getExecutionDomain(const MachineInstr *MI) const;

  void setExecutionDomain(MachineInstr *MI, unsigned Domain) const;

  unsigned getPartialRegUpdateClearance(const MachineInstr *MI, unsigned OpNum,
                                        const TargetRegisterInfo *TRI) const;
  void breakPartialRegDependency(MachineBasicBlock::iterator MI, unsigned OpNum,
                                 const TargetRegisterInfo *TRI) const;

  MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                      MachineInstr* MI,
                                      unsigned OpNum,
                                      const SmallVectorImpl<MachineOperand> &MOs,
                                      unsigned Size, unsigned Alignment) const;

  bool isHighLatencyDef(int opc) const;

  bool hasHighOperandLatency(const InstrItineraryData *ItinData,
                             const MachineRegisterInfo *MRI,
                             const MachineInstr *DefMI, unsigned DefIdx,
                             const MachineInstr *UseMI, unsigned UseIdx) const;

  /// analyzeCompare - For a comparison instruction, return the source registers
  /// in SrcReg and SrcReg2 if having two register operands, and the value it
  /// compares against in CmpValue. Return true if the comparison instruction
  /// can be analyzed.
  virtual bool analyzeCompare(const MachineInstr *MI, unsigned &SrcReg,
                              unsigned &SrcReg2,
                              int &CmpMask, int &CmpValue) const;

  /// optimizeCompareInstr - Check if there exists an earlier instruction that
  /// operates on the same source operands and sets flags in the same way as
  /// Compare; remove Compare if possible.
  virtual bool optimizeCompareInstr(MachineInstr *CmpInstr, unsigned SrcReg,
                                    unsigned SrcReg2, int CmpMask, int CmpValue,
                                    const MachineRegisterInfo *MRI) const;

  /// optimizeLoadInstr - Try to remove the load by folding it to a register
  /// operand at the use. We fold the load instructions if and only if the
  /// def and use are in the same BB. We only look at one load and see
  /// whether it can be folded into MI. FoldAsLoadDefReg is the virtual register
  /// defined by the load we are trying to fold. DefMI returns the machine
  /// instruction that defines FoldAsLoadDefReg, and the function returns
  /// the machine instruction generated due to folding.
  virtual MachineInstr* optimizeLoadInstr(MachineInstr *MI,
                        const MachineRegisterInfo *MRI,
                        unsigned &FoldAsLoadDefReg,
                        MachineInstr *&DefMI) const;

private:
  MachineInstr * convertToThreeAddressWithLEA(unsigned MIOpc,
                                              MachineFunction::iterator &MFI,
                                              MachineBasicBlock::iterator &MBBI,
                                              LiveVariables *LV) const;

  /// isFrameOperand - Return true and the FrameIndex if the specified
  /// operand and follow operands form a reference to the stack frame.
  bool isFrameOperand(const MachineInstr *MI, unsigned int Op,
                      int &FrameIndex) const;
};

} // End llvm namespace

#endif
