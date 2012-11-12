//===-- llvm/Target/TargetInstrInfo.h - Instruction Info --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the target machine instruction set to the code generator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETINSTRINFO_H
#define LLVM_TARGET_TARGETINSTRINFO_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class InstrItineraryData;
class LiveVariables;
class MCAsmInfo;
class MachineMemOperand;
class MachineRegisterInfo;
class MDNode;
class MCInst;
class MCSchedModel;
class SDNode;
class ScheduleHazardRecognizer;
class SelectionDAG;
class ScheduleDAG;
class TargetRegisterClass;
class TargetRegisterInfo;
class BranchProbability;

template<class T> class SmallVectorImpl;


//---------------------------------------------------------------------------
///
/// TargetInstrInfo - Interface to description of machine instruction set
///
class TargetInstrInfo : public MCInstrInfo {
  TargetInstrInfo(const TargetInstrInfo &) LLVM_DELETED_FUNCTION;
  void operator=(const TargetInstrInfo &) LLVM_DELETED_FUNCTION;
public:
  TargetInstrInfo(int CFSetupOpcode = -1, int CFDestroyOpcode = -1)
    : CallFrameSetupOpcode(CFSetupOpcode),
      CallFrameDestroyOpcode(CFDestroyOpcode) {
  }

  virtual ~TargetInstrInfo();

  /// getRegClass - Givem a machine instruction descriptor, returns the register
  /// class constraint for OpNum, or NULL.
  const TargetRegisterClass *getRegClass(const MCInstrDesc &TID,
                                         unsigned OpNum,
                                         const TargetRegisterInfo *TRI,
                                         const MachineFunction &MF) const;

  /// isTriviallyReMaterializable - Return true if the instruction is trivially
  /// rematerializable, meaning it has no side effects and requires no operands
  /// that aren't always available.
  bool isTriviallyReMaterializable(const MachineInstr *MI,
                                   AliasAnalysis *AA = 0) const {
    return MI->getOpcode() == TargetOpcode::IMPLICIT_DEF ||
           (MI->getDesc().isRematerializable() &&
            (isReallyTriviallyReMaterializable(MI, AA) ||
             isReallyTriviallyReMaterializableGeneric(MI, AA)));
  }

protected:
  /// isReallyTriviallyReMaterializable - For instructions with opcodes for
  /// which the M_REMATERIALIZABLE flag is set, this hook lets the target
  /// specify whether the instruction is actually trivially rematerializable,
  /// taking into consideration its operands. This predicate must return false
  /// if the instruction has any side effects other than producing a value, or
  /// if it requres any address registers that are not always available.
  virtual bool isReallyTriviallyReMaterializable(const MachineInstr *MI,
                                                 AliasAnalysis *AA) const {
    return false;
  }

private:
  /// isReallyTriviallyReMaterializableGeneric - For instructions with opcodes
  /// for which the M_REMATERIALIZABLE flag is set and the target hook
  /// isReallyTriviallyReMaterializable returns false, this function does
  /// target-independent tests to determine if the instruction is really
  /// trivially rematerializable.
  bool isReallyTriviallyReMaterializableGeneric(const MachineInstr *MI,
                                                AliasAnalysis *AA) const;

public:
  /// getCallFrameSetup/DestroyOpcode - These methods return the opcode of the
  /// frame setup/destroy instructions if they exist (-1 otherwise).  Some
  /// targets use pseudo instructions in order to abstract away the difference
  /// between operating with a frame pointer and operating without, through the
  /// use of these two instructions.
  ///
  int getCallFrameSetupOpcode() const { return CallFrameSetupOpcode; }
  int getCallFrameDestroyOpcode() const { return CallFrameDestroyOpcode; }

  /// isCoalescableExtInstr - Return true if the instruction is a "coalescable"
  /// extension instruction. That is, it's like a copy where it's legal for the
  /// source to overlap the destination. e.g. X86::MOVSX64rr32. If this returns
  /// true, then it's expected the pre-extension value is available as a subreg
  /// of the result register. This also returns the sub-register index in
  /// SubIdx.
  virtual bool isCoalescableExtInstr(const MachineInstr &MI,
                                     unsigned &SrcReg, unsigned &DstReg,
                                     unsigned &SubIdx) const {
    return false;
  }

  /// isLoadFromStackSlot - If the specified machine instruction is a direct
  /// load from a stack slot, return the virtual or physical register number of
  /// the destination along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than loading from the stack slot.
  virtual unsigned isLoadFromStackSlot(const MachineInstr *MI,
                                       int &FrameIndex) const {
    return 0;
  }

  /// isLoadFromStackSlotPostFE - Check for post-frame ptr elimination
  /// stack locations as well.  This uses a heuristic so it isn't
  /// reliable for correctness.
  virtual unsigned isLoadFromStackSlotPostFE(const MachineInstr *MI,
                                             int &FrameIndex) const {
    return 0;
  }

  /// hasLoadFromStackSlot - If the specified machine instruction has
  /// a load from a stack slot, return true along with the FrameIndex
  /// of the loaded stack slot and the machine mem operand containing
  /// the reference.  If not, return false.  Unlike
  /// isLoadFromStackSlot, this returns true for any instructions that
  /// loads from the stack.  This is just a hint, as some cases may be
  /// missed.
  virtual bool hasLoadFromStackSlot(const MachineInstr *MI,
                                    const MachineMemOperand *&MMO,
                                    int &FrameIndex) const {
    return 0;
  }

  /// isStoreToStackSlot - If the specified machine instruction is a direct
  /// store to a stack slot, return the virtual or physical register number of
  /// the source reg along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than storing to the stack slot.
  virtual unsigned isStoreToStackSlot(const MachineInstr *MI,
                                      int &FrameIndex) const {
    return 0;
  }

  /// isStoreToStackSlotPostFE - Check for post-frame ptr elimination
  /// stack locations as well.  This uses a heuristic so it isn't
  /// reliable for correctness.
  virtual unsigned isStoreToStackSlotPostFE(const MachineInstr *MI,
                                            int &FrameIndex) const {
    return 0;
  }

  /// hasStoreToStackSlot - If the specified machine instruction has a
  /// store to a stack slot, return true along with the FrameIndex of
  /// the loaded stack slot and the machine mem operand containing the
  /// reference.  If not, return false.  Unlike isStoreToStackSlot,
  /// this returns true for any instructions that stores to the
  /// stack.  This is just a hint, as some cases may be missed.
  virtual bool hasStoreToStackSlot(const MachineInstr *MI,
                                   const MachineMemOperand *&MMO,
                                   int &FrameIndex) const {
    return 0;
  }

  /// reMaterialize - Re-issue the specified 'original' instruction at the
  /// specific location targeting a new destination register.
  /// The register in Orig->getOperand(0).getReg() will be substituted by
  /// DestReg:SubIdx. Any existing subreg index is preserved or composed with
  /// SubIdx.
  virtual void reMaterialize(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MI,
                             unsigned DestReg, unsigned SubIdx,
                             const MachineInstr *Orig,
                             const TargetRegisterInfo &TRI) const = 0;

  /// duplicate - Create a duplicate of the Orig instruction in MF. This is like
  /// MachineFunction::CloneMachineInstr(), but the target may update operands
  /// that are required to be unique.
  ///
  /// The instruction must be duplicable as indicated by isNotDuplicable().
  virtual MachineInstr *duplicate(MachineInstr *Orig,
                                  MachineFunction &MF) const = 0;

  /// convertToThreeAddress - This method must be implemented by targets that
  /// set the M_CONVERTIBLE_TO_3_ADDR flag.  When this flag is set, the target
  /// may be able to convert a two-address instruction into one or more true
  /// three-address instructions on demand.  This allows the X86 target (for
  /// example) to convert ADD and SHL instructions into LEA instructions if they
  /// would require register copies due to two-addressness.
  ///
  /// This method returns a null pointer if the transformation cannot be
  /// performed, otherwise it returns the last new instruction.
  ///
  virtual MachineInstr *
  convertToThreeAddress(MachineFunction::iterator &MFI,
                   MachineBasicBlock::iterator &MBBI, LiveVariables *LV) const {
    return 0;
  }

  /// commuteInstruction - If a target has any instructions that are
  /// commutable but require converting to different instructions or making
  /// non-trivial changes to commute them, this method can overloaded to do
  /// that.  The default implementation simply swaps the commutable operands.
  /// If NewMI is false, MI is modified in place and returned; otherwise, a
  /// new machine instruction is created and returned.  Do not call this
  /// method for a non-commutable instruction, but there may be some cases
  /// where this method fails and returns null.
  virtual MachineInstr *commuteInstruction(MachineInstr *MI,
                                           bool NewMI = false) const = 0;

  /// findCommutedOpIndices - If specified MI is commutable, return the two
  /// operand indices that would swap value. Return false if the instruction
  /// is not in a form which this routine understands.
  virtual bool findCommutedOpIndices(MachineInstr *MI, unsigned &SrcOpIdx1,
                                     unsigned &SrcOpIdx2) const = 0;

  /// produceSameValue - Return true if two machine instructions would produce
  /// identical values. By default, this is only true when the two instructions
  /// are deemed identical except for defs. If this function is called when the
  /// IR is still in SSA form, the caller can pass the MachineRegisterInfo for
  /// aggressive checks.
  virtual bool produceSameValue(const MachineInstr *MI0,
                                const MachineInstr *MI1,
                                const MachineRegisterInfo *MRI = 0) const = 0;

  /// AnalyzeBranch - Analyze the branching code at the end of MBB, returning
  /// true if it cannot be understood (e.g. it's a switch dispatch or isn't
  /// implemented for a target).  Upon success, this returns false and returns
  /// with the following information in various cases:
  ///
  /// 1. If this block ends with no branches (it just falls through to its succ)
  ///    just return false, leaving TBB/FBB null.
  /// 2. If this block ends with only an unconditional branch, it sets TBB to be
  ///    the destination block.
  /// 3. If this block ends with a conditional branch and it falls through to a
  ///    successor block, it sets TBB to be the branch destination block and a
  ///    list of operands that evaluate the condition. These operands can be
  ///    passed to other TargetInstrInfo methods to create new branches.
  /// 4. If this block ends with a conditional branch followed by an
  ///    unconditional branch, it returns the 'true' destination in TBB, the
  ///    'false' destination in FBB, and a list of operands that evaluate the
  ///    condition.  These operands can be passed to other TargetInstrInfo
  ///    methods to create new branches.
  ///
  /// Note that RemoveBranch and InsertBranch must be implemented to support
  /// cases where this method returns success.
  ///
  /// If AllowModify is true, then this routine is allowed to modify the basic
  /// block (e.g. delete instructions after the unconditional branch).
  ///
  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             SmallVectorImpl<MachineOperand> &Cond,
                             bool AllowModify = false) const {
    return true;
  }

  /// RemoveBranch - Remove the branching code at the end of the specific MBB.
  /// This is only invoked in cases where AnalyzeBranch returns success. It
  /// returns the number of instructions that were removed.
  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const {
    llvm_unreachable("Target didn't implement TargetInstrInfo::RemoveBranch!");
  }

  /// InsertBranch - Insert branch code into the end of the specified
  /// MachineBasicBlock.  The operands to this method are the same as those
  /// returned by AnalyzeBranch.  This is only invoked in cases where
  /// AnalyzeBranch returns success. It returns the number of instructions
  /// inserted.
  ///
  /// It is also invoked by tail merging to add unconditional branches in
  /// cases where AnalyzeBranch doesn't apply because there was no original
  /// branch to analyze.  At least this much must be implemented, else tail
  /// merging needs to be disabled.
  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const SmallVectorImpl<MachineOperand> &Cond,
                                DebugLoc DL) const {
    llvm_unreachable("Target didn't implement TargetInstrInfo::InsertBranch!");
  }

  /// ReplaceTailWithBranchTo - Delete the instruction OldInst and everything
  /// after it, replacing it with an unconditional branch to NewDest. This is
  /// used by the tail merging pass.
  virtual void ReplaceTailWithBranchTo(MachineBasicBlock::iterator Tail,
                                       MachineBasicBlock *NewDest) const = 0;

  /// isLegalToSplitMBBAt - Return true if it's legal to split the given basic
  /// block at the specified instruction (i.e. instruction would be the start
  /// of a new basic block).
  virtual bool isLegalToSplitMBBAt(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI) const {
    return true;
  }

  /// isProfitableToIfCvt - Return true if it's profitable to predicate
  /// instructions with accumulated instruction latency of "NumCycles"
  /// of the specified basic block, where the probability of the instructions
  /// being executed is given by Probability, and Confidence is a measure
  /// of our confidence that it will be properly predicted.
  virtual
  bool isProfitableToIfCvt(MachineBasicBlock &MBB, unsigned NumCycles,
                           unsigned ExtraPredCycles,
                           const BranchProbability &Probability) const {
    return false;
  }

  /// isProfitableToIfCvt - Second variant of isProfitableToIfCvt, this one
  /// checks for the case where two basic blocks from true and false path
  /// of a if-then-else (diamond) are predicated on mutally exclusive
  /// predicates, where the probability of the true path being taken is given
  /// by Probability, and Confidence is a measure of our confidence that it
  /// will be properly predicted.
  virtual bool
  isProfitableToIfCvt(MachineBasicBlock &TMBB,
                      unsigned NumTCycles, unsigned ExtraTCycles,
                      MachineBasicBlock &FMBB,
                      unsigned NumFCycles, unsigned ExtraFCycles,
                      const BranchProbability &Probability) const {
    return false;
  }

  /// isProfitableToDupForIfCvt - Return true if it's profitable for
  /// if-converter to duplicate instructions of specified accumulated
  /// instruction latencies in the specified MBB to enable if-conversion.
  /// The probability of the instructions being executed is given by
  /// Probability, and Confidence is a measure of our confidence that it
  /// will be properly predicted.
  virtual bool
  isProfitableToDupForIfCvt(MachineBasicBlock &MBB, unsigned NumCycles,
                            const BranchProbability &Probability) const {
    return false;
  }

  /// isProfitableToUnpredicate - Return true if it's profitable to unpredicate
  /// one side of a 'diamond', i.e. two sides of if-else predicated on mutually
  /// exclusive predicates.
  /// e.g.
  ///   subeq  r0, r1, #1
  ///   addne  r0, r1, #1
  /// =>
  ///   sub    r0, r1, #1
  ///   addne  r0, r1, #1
  ///
  /// This may be profitable is conditional instructions are always executed.
  virtual bool isProfitableToUnpredicate(MachineBasicBlock &TMBB,
                                         MachineBasicBlock &FMBB) const {
    return false;
  }

  /// canInsertSelect - Return true if it is possible to insert a select
  /// instruction that chooses between TrueReg and FalseReg based on the
  /// condition code in Cond.
  ///
  /// When successful, also return the latency in cycles from TrueReg,
  /// FalseReg, and Cond to the destination register. The Cond latency should
  /// compensate for a conditional branch being removed. For example, if a
  /// conditional branch has a 3 cycle latency from the condition code read,
  /// and a cmov instruction has a 2 cycle latency from the condition code
  /// read, CondCycles should be returned as -1.
  ///
  /// @param MBB         Block where select instruction would be inserted.
  /// @param Cond        Condition returned by AnalyzeBranch.
  /// @param TrueReg     Virtual register to select when Cond is true.
  /// @param FalseReg    Virtual register to select when Cond is false.
  /// @param CondCycles  Latency from Cond+Branch to select output.
  /// @param TrueCycles  Latency from TrueReg to select output.
  /// @param FalseCycles Latency from FalseReg to select output.
  virtual bool canInsertSelect(const MachineBasicBlock &MBB,
                               const SmallVectorImpl<MachineOperand> &Cond,
                               unsigned TrueReg, unsigned FalseReg,
                               int &CondCycles,
                               int &TrueCycles, int &FalseCycles) const {
    return false;
  }

  /// insertSelect - Insert a select instruction into MBB before I that will
  /// copy TrueReg to DstReg when Cond is true, and FalseReg to DstReg when
  /// Cond is false.
  ///
  /// This function can only be called after canInsertSelect() returned true.
  /// The condition in Cond comes from AnalyzeBranch, and it can be assumed
  /// that the same flags or registers required by Cond are available at the
  /// insertion point.
  ///
  /// @param MBB      Block where select instruction should be inserted.
  /// @param I        Insertion point.
  /// @param DL       Source location for debugging.
  /// @param DstReg   Virtual register to be defined by select instruction.
  /// @param Cond     Condition as computed by AnalyzeBranch.
  /// @param TrueReg  Virtual register to copy when Cond is true.
  /// @param FalseReg Virtual register to copy when Cons is false.
  virtual void insertSelect(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator I, DebugLoc DL,
                            unsigned DstReg,
                            const SmallVectorImpl<MachineOperand> &Cond,
                            unsigned TrueReg, unsigned FalseReg) const {
    llvm_unreachable("Target didn't implement TargetInstrInfo::insertSelect!");
  }

  /// analyzeSelect - Analyze the given select instruction, returning true if
  /// it cannot be understood. It is assumed that MI->isSelect() is true.
  ///
  /// When successful, return the controlling condition and the operands that
  /// determine the true and false result values.
  ///
  ///   Result = SELECT Cond, TrueOp, FalseOp
  ///
  /// Some targets can optimize select instructions, for example by predicating
  /// the instruction defining one of the operands. Such targets should set
  /// Optimizable.
  ///
  /// @param         MI Select instruction to analyze.
  /// @param Cond    Condition controlling the select.
  /// @param TrueOp  Operand number of the value selected when Cond is true.
  /// @param FalseOp Operand number of the value selected when Cond is false.
  /// @param Optimizable Returned as true if MI is optimizable.
  /// @returns False on success.
  virtual bool analyzeSelect(const MachineInstr *MI,
                             SmallVectorImpl<MachineOperand> &Cond,
                             unsigned &TrueOp, unsigned &FalseOp,
                             bool &Optimizable) const {
    assert(MI && MI->isSelect() && "MI must be a select instruction");
    return true;
  }

  /// optimizeSelect - Given a select instruction that was understood by
  /// analyzeSelect and returned Optimizable = true, attempt to optimize MI by
  /// merging it with one of its operands. Returns NULL on failure.
  ///
  /// When successful, returns the new select instruction. The client is
  /// responsible for deleting MI.
  ///
  /// If both sides of the select can be optimized, PreferFalse is used to pick
  /// a side.
  ///
  /// @param MI          Optimizable select instruction.
  /// @param PreferFalse Try to optimize FalseOp instead of TrueOp.
  /// @returns Optimized instruction or NULL.
  virtual MachineInstr *optimizeSelect(MachineInstr *MI,
                                       bool PreferFalse = false) const {
    // This function must be implemented if Optimizable is ever set.
    llvm_unreachable("Target must implement TargetInstrInfo::optimizeSelect!");
  }

  /// copyPhysReg - Emit instructions to copy a pair of physical registers.
  ///
  /// This function should support copies within any legal register class as
  /// well as any cross-class copies created during instruction selection.
  ///
  /// The source and destination registers may overlap, which may require a
  /// careful implementation when multiple copy instructions are required for
  /// large registers. See for example the ARM target.
  virtual void copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, DebugLoc DL,
                           unsigned DestReg, unsigned SrcReg,
                           bool KillSrc) const {
    llvm_unreachable("Target didn't implement TargetInstrInfo::copyPhysReg!");
  }

  /// storeRegToStackSlot - Store the specified register of the given register
  /// class to the specified stack frame index. The store instruction is to be
  /// added to the given machine basic block before the specified machine
  /// instruction. If isKill is true, the register operand is the last use and
  /// must be marked kill.
  virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned SrcReg, bool isKill, int FrameIndex,
                                   const TargetRegisterClass *RC,
                                   const TargetRegisterInfo *TRI) const {
    llvm_unreachable("Target didn't implement "
                     "TargetInstrInfo::storeRegToStackSlot!");
  }

  /// loadRegFromStackSlot - Load the specified register of the given register
  /// class from the specified stack frame index. The load instruction is to be
  /// added to the given machine basic block before the specified machine
  /// instruction.
  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
                                    unsigned DestReg, int FrameIndex,
                                    const TargetRegisterClass *RC,
                                    const TargetRegisterInfo *TRI) const {
    llvm_unreachable("Target didn't implement "
                     "TargetInstrInfo::loadRegFromStackSlot!");
  }

  /// expandPostRAPseudo - This function is called for all pseudo instructions
  /// that remain after register allocation. Many pseudo instructions are
  /// created to help register allocation. This is the place to convert them
  /// into real instructions. The target can edit MI in place, or it can insert
  /// new instructions and erase MI. The function should return true if
  /// anything was changed.
  virtual bool expandPostRAPseudo(MachineBasicBlock::iterator MI) const {
    return false;
  }

  /// emitFrameIndexDebugValue - Emit a target-dependent form of
  /// DBG_VALUE encoding the address of a frame index.  Addresses would
  /// normally be lowered the same way as other addresses on the target,
  /// e.g. in load instructions.  For targets that do not support this
  /// the debug info is simply lost.
  /// If you add this for a target you should handle this DBG_VALUE in the
  /// target-specific AsmPrinter code as well; you will probably get invalid
  /// assembly output if you don't.
  virtual MachineInstr *emitFrameIndexDebugValue(MachineFunction &MF,
                                                 int FrameIx,
                                                 uint64_t Offset,
                                                 const MDNode *MDPtr,
                                                 DebugLoc dl) const {
    return 0;
  }

  /// foldMemoryOperand - Attempt to fold a load or store of the specified stack
  /// slot into the specified machine instruction for the specified operand(s).
  /// If this is possible, a new instruction is returned with the specified
  /// operand folded, otherwise NULL is returned.
  /// The new instruction is inserted before MI, and the client is responsible
  /// for removing the old instruction.
  MachineInstr* foldMemoryOperand(MachineBasicBlock::iterator MI,
                                  const SmallVectorImpl<unsigned> &Ops,
                                  int FrameIndex) const;

  /// foldMemoryOperand - Same as the previous version except it allows folding
  /// of any load and store from / to any address, not just from a specific
  /// stack slot.
  MachineInstr* foldMemoryOperand(MachineBasicBlock::iterator MI,
                                  const SmallVectorImpl<unsigned> &Ops,
                                  MachineInstr* LoadMI) const;

protected:
  /// foldMemoryOperandImpl - Target-dependent implementation for
  /// foldMemoryOperand. Target-independent code in foldMemoryOperand will
  /// take care of adding a MachineMemOperand to the newly created instruction.
  virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                          MachineInstr* MI,
                                          const SmallVectorImpl<unsigned> &Ops,
                                          int FrameIndex) const {
    return 0;
  }

  /// foldMemoryOperandImpl - Target-dependent implementation for
  /// foldMemoryOperand. Target-independent code in foldMemoryOperand will
  /// take care of adding a MachineMemOperand to the newly created instruction.
  virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                              MachineInstr* MI,
                                          const SmallVectorImpl<unsigned> &Ops,
                                              MachineInstr* LoadMI) const {
    return 0;
  }

public:
  /// canFoldMemoryOperand - Returns true for the specified load / store if
  /// folding is possible.
  virtual
  bool canFoldMemoryOperand(const MachineInstr *MI,
                            const SmallVectorImpl<unsigned> &Ops) const =0;

  /// unfoldMemoryOperand - Separate a single instruction which folded a load or
  /// a store or a load and a store into two or more instruction. If this is
  /// possible, returns true as well as the new instructions by reference.
  virtual bool unfoldMemoryOperand(MachineFunction &MF, MachineInstr *MI,
                                unsigned Reg, bool UnfoldLoad, bool UnfoldStore,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const{
    return false;
  }

  virtual bool unfoldMemoryOperand(SelectionDAG &DAG, SDNode *N,
                                   SmallVectorImpl<SDNode*> &NewNodes) const {
    return false;
  }

  /// getOpcodeAfterMemoryUnfold - Returns the opcode of the would be new
  /// instruction after load / store are unfolded from an instruction of the
  /// specified opcode. It returns zero if the specified unfolding is not
  /// possible. If LoadRegIndex is non-null, it is filled in with the operand
  /// index of the operand which will hold the register holding the loaded
  /// value.
  virtual unsigned getOpcodeAfterMemoryUnfold(unsigned Opc,
                                      bool UnfoldLoad, bool UnfoldStore,
                                      unsigned *LoadRegIndex = 0) const {
    return 0;
  }

  /// areLoadsFromSameBasePtr - This is used by the pre-regalloc scheduler
  /// to determine if two loads are loading from the same base address. It
  /// should only return true if the base pointers are the same and the
  /// only differences between the two addresses are the offset. It also returns
  /// the offsets by reference.
  virtual bool areLoadsFromSameBasePtr(SDNode *Load1, SDNode *Load2,
                                    int64_t &Offset1, int64_t &Offset2) const {
    return false;
  }

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
                                       unsigned NumLoads) const {
    return false;
  }

  /// ReverseBranchCondition - Reverses the branch condition of the specified
  /// condition list, returning false on success and true if it cannot be
  /// reversed.
  virtual
  bool ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
    return true;
  }

  /// insertNoop - Insert a noop into the instruction stream at the specified
  /// point.
  virtual void insertNoop(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MI) const;


  /// getNoopForMachoTarget - Return the noop instruction to use for a noop.
  virtual void getNoopForMachoTarget(MCInst &NopInst) const {
    // Default to just using 'nop' string.
  }


  /// isPredicated - Returns true if the instruction is already predicated.
  ///
  virtual bool isPredicated(const MachineInstr *MI) const {
    return false;
  }

  /// isUnpredicatedTerminator - Returns true if the instruction is a
  /// terminator instruction that has not been predicated.
  virtual bool isUnpredicatedTerminator(const MachineInstr *MI) const = 0;

  /// PredicateInstruction - Convert the instruction into a predicated
  /// instruction. It returns true if the operation was successful.
  virtual
  bool PredicateInstruction(MachineInstr *MI,
                        const SmallVectorImpl<MachineOperand> &Pred) const = 0;

  /// SubsumesPredicate - Returns true if the first specified predicate
  /// subsumes the second, e.g. GE subsumes GT.
  virtual
  bool SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                         const SmallVectorImpl<MachineOperand> &Pred2) const {
    return false;
  }

  /// DefinesPredicate - If the specified instruction defines any predicate
  /// or condition code register(s) used for predication, returns true as well
  /// as the definition predicate(s) by reference.
  virtual bool DefinesPredicate(MachineInstr *MI,
                                std::vector<MachineOperand> &Pred) const {
    return false;
  }

  /// isPredicable - Return true if the specified instruction can be predicated.
  /// By default, this returns true for every instruction with a
  /// PredicateOperand.
  virtual bool isPredicable(MachineInstr *MI) const {
    return MI->getDesc().isPredicable();
  }

  /// isSafeToMoveRegClassDefs - Return true if it's safe to move a machine
  /// instruction that defines the specified register class.
  virtual bool isSafeToMoveRegClassDefs(const TargetRegisterClass *RC) const {
    return true;
  }

  /// isSchedulingBoundary - Test if the given instruction should be
  /// considered a scheduling boundary. This primarily includes labels and
  /// terminators.
  virtual bool isSchedulingBoundary(const MachineInstr *MI,
                                    const MachineBasicBlock *MBB,
                                    const MachineFunction &MF) const = 0;

  /// Measure the specified inline asm to determine an approximation of its
  /// length.
  virtual unsigned getInlineAsmLength(const char *Str,
                                      const MCAsmInfo &MAI) const;

  /// CreateTargetHazardRecognizer - Allocate and return a hazard recognizer to
  /// use for this target when scheduling the machine instructions before
  /// register allocation.
  virtual ScheduleHazardRecognizer*
  CreateTargetHazardRecognizer(const TargetMachine *TM,
                               const ScheduleDAG *DAG) const = 0;

  /// CreateTargetMIHazardRecognizer - Allocate and return a hazard recognizer
  /// to use for this target when scheduling the machine instructions before
  /// register allocation.
  virtual ScheduleHazardRecognizer*
  CreateTargetMIHazardRecognizer(const InstrItineraryData*,
                                 const ScheduleDAG *DAG) const = 0;

  /// CreateTargetPostRAHazardRecognizer - Allocate and return a hazard
  /// recognizer to use for this target when scheduling the machine instructions
  /// after register allocation.
  virtual ScheduleHazardRecognizer*
  CreateTargetPostRAHazardRecognizer(const InstrItineraryData*,
                                     const ScheduleDAG *DAG) const = 0;

  /// analyzeCompare - For a comparison instruction, return the source registers
  /// in SrcReg and SrcReg2 if having two register operands, and the value it
  /// compares against in CmpValue. Return true if the comparison instruction
  /// can be analyzed.
  virtual bool analyzeCompare(const MachineInstr *MI,
                              unsigned &SrcReg, unsigned &SrcReg2,
                              int &Mask, int &Value) const {
    return false;
  }

  /// optimizeCompareInstr - See if the comparison instruction can be converted
  /// into something more efficient. E.g., on ARM most instructions can set the
  /// flags register, obviating the need for a separate CMP.
  virtual bool optimizeCompareInstr(MachineInstr *CmpInstr,
                                    unsigned SrcReg, unsigned SrcReg2,
                                    int Mask, int Value,
                                    const MachineRegisterInfo *MRI) const {
    return false;
  }

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
                        MachineInstr *&DefMI) const {
    return 0;
  }

  /// FoldImmediate - 'Reg' is known to be defined by a move immediate
  /// instruction, try to fold the immediate into the use instruction.
  virtual bool FoldImmediate(MachineInstr *UseMI, MachineInstr *DefMI,
                             unsigned Reg, MachineRegisterInfo *MRI) const {
    return false;
  }

  /// getNumMicroOps - Return the number of u-operations the given machine
  /// instruction will be decoded to on the target cpu. The itinerary's
  /// IssueWidth is the number of microops that can be dispatched each
  /// cycle. An instruction with zero microops takes no dispatch resources.
  virtual unsigned getNumMicroOps(const InstrItineraryData *ItinData,
                                  const MachineInstr *MI) const = 0;

  /// isZeroCost - Return true for pseudo instructions that don't consume any
  /// machine resources in their current form. These are common cases that the
  /// scheduler should consider free, rather than conservatively handling them
  /// as instructions with no itinerary.
  bool isZeroCost(unsigned Opcode) const {
    return Opcode <= TargetOpcode::COPY;
  }

  virtual int getOperandLatency(const InstrItineraryData *ItinData,
                                SDNode *DefNode, unsigned DefIdx,
                                SDNode *UseNode, unsigned UseIdx) const = 0;

  /// getOperandLatency - Compute and return the use operand latency of a given
  /// pair of def and use.
  /// In most cases, the static scheduling itinerary was enough to determine the
  /// operand latency. But it may not be possible for instructions with variable
  /// number of defs / uses.
  ///
  /// This is a raw interface to the itinerary that may be directly overriden by
  /// a target. Use computeOperandLatency to get the best estimate of latency.
  virtual int getOperandLatency(const InstrItineraryData *ItinData,
                                const MachineInstr *DefMI, unsigned DefIdx,
                                const MachineInstr *UseMI,
                                unsigned UseIdx) const = 0;

  /// computeOperandLatency - Compute and return the latency of the given data
  /// dependent def and use when the operand indices are already known.
  ///
  /// FindMin may be set to get the minimum vs. expected latency.
  unsigned computeOperandLatency(const InstrItineraryData *ItinData,
                                 const MachineInstr *DefMI, unsigned DefIdx,
                                 const MachineInstr *UseMI, unsigned UseIdx,
                                 bool FindMin = false) const;

  /// getInstrLatency - Compute the instruction latency of a given instruction.
  /// If the instruction has higher cost when predicated, it's returned via
  /// PredCost.
  virtual unsigned getInstrLatency(const InstrItineraryData *ItinData,
                                   const MachineInstr *MI,
                                   unsigned *PredCost = 0) const = 0;

  virtual int getInstrLatency(const InstrItineraryData *ItinData,
                              SDNode *Node) const = 0;

  /// Return the default expected latency for a def based on it's opcode.
  unsigned defaultDefLatency(const MCSchedModel *SchedModel,
                             const MachineInstr *DefMI) const;

  int computeDefOperandLatency(const InstrItineraryData *ItinData,
                               const MachineInstr *DefMI, bool FindMin) const;

  /// isHighLatencyDef - Return true if this opcode has high latency to its
  /// result.
  virtual bool isHighLatencyDef(int opc) const { return false; }

  /// hasHighOperandLatency - Compute operand latency between a def of 'Reg'
  /// and an use in the current loop, return true if the target considered
  /// it 'high'. This is used by optimization passes such as machine LICM to
  /// determine whether it makes sense to hoist an instruction out even in
  /// high register pressure situation.
  virtual
  bool hasHighOperandLatency(const InstrItineraryData *ItinData,
                             const MachineRegisterInfo *MRI,
                             const MachineInstr *DefMI, unsigned DefIdx,
                             const MachineInstr *UseMI, unsigned UseIdx) const {
    return false;
  }

  /// hasLowDefLatency - Compute operand latency of a def of 'Reg', return true
  /// if the target considered it 'low'.
  virtual
  bool hasLowDefLatency(const InstrItineraryData *ItinData,
                        const MachineInstr *DefMI, unsigned DefIdx) const = 0;

  /// verifyInstruction - Perform target specific instruction verification.
  virtual
  bool verifyInstruction(const MachineInstr *MI, StringRef &ErrInfo) const {
    return true;
  }

  /// getExecutionDomain - Return the current execution domain and bit mask of
  /// possible domains for instruction.
  ///
  /// Some micro-architectures have multiple execution domains, and multiple
  /// opcodes that perform the same operation in different domains.  For
  /// example, the x86 architecture provides the por, orps, and orpd
  /// instructions that all do the same thing.  There is a latency penalty if a
  /// register is written in one domain and read in another.
  ///
  /// This function returns a pair (domain, mask) containing the execution
  /// domain of MI, and a bit mask of possible domains.  The setExecutionDomain
  /// function can be used to change the opcode to one of the domains in the
  /// bit mask.  Instructions whose execution domain can't be changed should
  /// return a 0 mask.
  ///
  /// The execution domain numbers don't have any special meaning except domain
  /// 0 is used for instructions that are not associated with any interesting
  /// execution domain.
  ///
  virtual std::pair<uint16_t, uint16_t>
  getExecutionDomain(const MachineInstr *MI) const {
    return std::make_pair(0, 0);
  }

  /// setExecutionDomain - Change the opcode of MI to execute in Domain.
  ///
  /// The bit (1 << Domain) must be set in the mask returned from
  /// getExecutionDomain(MI).
  ///
  virtual void setExecutionDomain(MachineInstr *MI, unsigned Domain) const {}


  /// getPartialRegUpdateClearance - Returns the preferred minimum clearance
  /// before an instruction with an unwanted partial register update.
  ///
  /// Some instructions only write part of a register, and implicitly need to
  /// read the other parts of the register.  This may cause unwanted stalls
  /// preventing otherwise unrelated instructions from executing in parallel in
  /// an out-of-order CPU.
  ///
  /// For example, the x86 instruction cvtsi2ss writes its result to bits
  /// [31:0] of the destination xmm register. Bits [127:32] are unaffected, so
  /// the instruction needs to wait for the old value of the register to become
  /// available:
  ///
  ///   addps %xmm1, %xmm0
  ///   movaps %xmm0, (%rax)
  ///   cvtsi2ss %rbx, %xmm0
  ///
  /// In the code above, the cvtsi2ss instruction needs to wait for the addps
  /// instruction before it can issue, even though the high bits of %xmm0
  /// probably aren't needed.
  ///
  /// This hook returns the preferred clearance before MI, measured in
  /// instructions.  Other defs of MI's operand OpNum are avoided in the last N
  /// instructions before MI.  It should only return a positive value for
  /// unwanted dependencies.  If the old bits of the defined register have
  /// useful values, or if MI is determined to otherwise read the dependency,
  /// the hook should return 0.
  ///
  /// The unwanted dependency may be handled by:
  ///
  /// 1. Allocating the same register for an MI def and use.  That makes the
  ///    unwanted dependency identical to a required dependency.
  ///
  /// 2. Allocating a register for the def that has no defs in the previous N
  ///    instructions.
  ///
  /// 3. Calling breakPartialRegDependency() with the same arguments.  This
  ///    allows the target to insert a dependency breaking instruction.
  ///
  virtual unsigned
  getPartialRegUpdateClearance(const MachineInstr *MI, unsigned OpNum,
                               const TargetRegisterInfo *TRI) const {
    // The default implementation returns 0 for no partial register dependency.
    return 0;
  }

  /// breakPartialRegDependency - Insert a dependency-breaking instruction
  /// before MI to eliminate an unwanted dependency on OpNum.
  ///
  /// If it wasn't possible to avoid a def in the last N instructions before MI
  /// (see getPartialRegUpdateClearance), this hook will be called to break the
  /// unwanted dependency.
  ///
  /// On x86, an xorps instruction can be used as a dependency breaker:
  ///
  ///   addps %xmm1, %xmm0
  ///   movaps %xmm0, (%rax)
  ///   xorps %xmm0, %xmm0
  ///   cvtsi2ss %rbx, %xmm0
  ///
  /// An <imp-kill> operand should be added to MI if an instruction was
  /// inserted.  This ties the instructions together in the post-ra scheduler.
  ///
  virtual void
  breakPartialRegDependency(MachineBasicBlock::iterator MI, unsigned OpNum,
                            const TargetRegisterInfo *TRI) const {}

  /// Create machine specific model for scheduling.
  virtual DFAPacketizer*
    CreateTargetScheduleState(const TargetMachine*, const ScheduleDAG*) const {
    return NULL;
  }

private:
  int CallFrameSetupOpcode, CallFrameDestroyOpcode;
};

/// TargetInstrInfoImpl - This is the default implementation of
/// TargetInstrInfo, which just provides a couple of default implementations
/// for various methods.  This separated out because it is implemented in
/// libcodegen, not in libtarget.
class TargetInstrInfoImpl : public TargetInstrInfo {
protected:
  TargetInstrInfoImpl(int CallFrameSetupOpcode = -1,
                      int CallFrameDestroyOpcode = -1)
    : TargetInstrInfo(CallFrameSetupOpcode, CallFrameDestroyOpcode) {}
public:
  virtual void ReplaceTailWithBranchTo(MachineBasicBlock::iterator OldInst,
                                       MachineBasicBlock *NewDest) const;
  virtual MachineInstr *commuteInstruction(MachineInstr *MI,
                                           bool NewMI = false) const;
  virtual bool findCommutedOpIndices(MachineInstr *MI, unsigned &SrcOpIdx1,
                                     unsigned &SrcOpIdx2) const;
  virtual bool canFoldMemoryOperand(const MachineInstr *MI,
                                    const SmallVectorImpl<unsigned> &Ops) const;
  virtual bool hasLoadFromStackSlot(const MachineInstr *MI,
                                    const MachineMemOperand *&MMO,
                                    int &FrameIndex) const;
  virtual bool hasStoreToStackSlot(const MachineInstr *MI,
                                   const MachineMemOperand *&MMO,
                                   int &FrameIndex) const;
  virtual bool isUnpredicatedTerminator(const MachineInstr *MI) const;
  virtual bool PredicateInstruction(MachineInstr *MI,
                            const SmallVectorImpl<MachineOperand> &Pred) const;
  virtual void reMaterialize(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MI,
                             unsigned DestReg, unsigned SubReg,
                             const MachineInstr *Orig,
                             const TargetRegisterInfo &TRI) const;
  virtual MachineInstr *duplicate(MachineInstr *Orig,
                                  MachineFunction &MF) const;
  virtual bool produceSameValue(const MachineInstr *MI0,
                                const MachineInstr *MI1,
                                const MachineRegisterInfo *MRI) const;
  virtual bool isSchedulingBoundary(const MachineInstr *MI,
                                    const MachineBasicBlock *MBB,
                                    const MachineFunction &MF) const;

  virtual int getOperandLatency(const InstrItineraryData *ItinData,
                                SDNode *DefNode, unsigned DefIdx,
                                SDNode *UseNode, unsigned UseIdx) const;

  virtual int getInstrLatency(const InstrItineraryData *ItinData,
                              SDNode *Node) const;

  virtual unsigned getNumMicroOps(const InstrItineraryData *ItinData,
                                  const MachineInstr *MI) const;

  virtual unsigned getInstrLatency(const InstrItineraryData *ItinData,
                                   const MachineInstr *MI,
                                   unsigned *PredCost = 0) const;

  virtual
  bool hasLowDefLatency(const InstrItineraryData *ItinData,
                        const MachineInstr *DefMI, unsigned DefIdx) const;

  virtual int getOperandLatency(const InstrItineraryData *ItinData,
                                const MachineInstr *DefMI, unsigned DefIdx,
                                const MachineInstr *UseMI,
                                unsigned UseIdx) const;

  bool usePreRAHazardRecognizer() const;

  virtual ScheduleHazardRecognizer *
  CreateTargetHazardRecognizer(const TargetMachine*, const ScheduleDAG*) const;

  virtual ScheduleHazardRecognizer *
  CreateTargetMIHazardRecognizer(const InstrItineraryData*,
                                 const ScheduleDAG*) const;

  virtual ScheduleHazardRecognizer *
  CreateTargetPostRAHazardRecognizer(const InstrItineraryData*,
                                     const ScheduleDAG*) const;
};

} // End llvm namespace

#endif
