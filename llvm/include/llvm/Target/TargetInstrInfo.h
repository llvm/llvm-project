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

#include "llvm/Target/TargetInstrDesc.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class TargetRegisterClass;
class LiveVariables;
class CalleeSavedInfo;
class SDNode;
class SelectionDAG;

template<class T> class SmallVectorImpl;


//---------------------------------------------------------------------------
///
/// TargetInstrInfo - Interface to description of machine instruction set
///
class TargetInstrInfo {
  const TargetInstrDesc *Descriptors; // Raw array to allow static init'n
  unsigned NumOpcodes;                // Number of entries in the desc array

  TargetInstrInfo(const TargetInstrInfo &);  // DO NOT IMPLEMENT
  void operator=(const TargetInstrInfo &);   // DO NOT IMPLEMENT
public:
  TargetInstrInfo(const TargetInstrDesc *desc, unsigned NumOpcodes);
  virtual ~TargetInstrInfo();

  // Invariant opcodes: All instruction sets have these as their low opcodes.
  enum { 
    PHI = 0,
    INLINEASM = 1,
    DBG_LABEL = 2,
    EH_LABEL = 3,
    GC_LABEL = 4,
    DECLARE = 5,
    EXTRACT_SUBREG = 6,
    INSERT_SUBREG = 7,
    IMPLICIT_DEF = 8,
    SUBREG_TO_REG = 9
  };

  unsigned getNumOpcodes() const { return NumOpcodes; }

  /// get - Return the machine instruction descriptor that corresponds to the
  /// specified instruction opcode.
  ///
  const TargetInstrDesc &get(unsigned Opcode) const {
    assert(Opcode < NumOpcodes && "Invalid opcode!");
    return Descriptors[Opcode];
  }

  /// isTriviallyReMaterializable - Return true if the instruction is trivially
  /// rematerializable, meaning it has no side effects and requires no operands
  /// that aren't always available.
  bool isTriviallyReMaterializable(const MachineInstr *MI) const {
    return MI->getDesc().isRematerializable() &&
           isReallyTriviallyReMaterializable(MI);
  }

protected:
  /// isReallyTriviallyReMaterializable - For instructions with opcodes for
  /// which the M_REMATERIALIZABLE flag is set, this function tests whether the
  /// instruction itself is actually trivially rematerializable, considering
  /// its operands.  This is used for targets that have instructions that are
  /// only trivially rematerializable for specific uses.  This predicate must
  /// return false if the instruction has any side effects other than
  /// producing a value, or if it requres any address registers that are not
  /// always available.
  virtual bool isReallyTriviallyReMaterializable(const MachineInstr *MI) const {
    return true;
  }

public:
  /// Return true if the instruction is a register to register move and return
  /// the source and dest operands and their sub-register indices by reference.
  virtual bool isMoveInstr(const MachineInstr& MI,
                           unsigned& SrcReg, unsigned& DstReg,
                           unsigned& SrcSubIdx, unsigned& DstSubIdx) const {
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
  
  /// isStoreToStackSlot - If the specified machine instruction is a direct
  /// store to a stack slot, return the virtual or physical register number of
  /// the source reg along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than storing to the stack slot.
  virtual unsigned isStoreToStackSlot(const MachineInstr *MI,
                                      int &FrameIndex) const {
    return 0;
  }

  /// reMaterialize - Re-issue the specified 'original' instruction at the
  /// specific location targeting a new destination register.
  virtual void reMaterialize(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MI,
                             unsigned DestReg,
                             const MachineInstr *Orig) const = 0;

  /// isInvariantLoad - Return true if the specified instruction (which is
  /// marked mayLoad) is loading from a location whose value is invariant across
  /// the function.  For example, loading a value from the constant pool or from
  /// from the argument area of a function if it does not change.  This should
  /// only return true of *all* loads the instruction does are invariant (if it
  /// does multiple loads).
  virtual bool isInvariantLoad(const MachineInstr *MI) const {
    return false;
  }
  
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

  /// commuteInstruction - If a target has any instructions that are commutable,
  /// but require converting to a different instruction or making non-trivial
  /// changes to commute them, this method can overloaded to do this.  The
  /// default implementation of this method simply swaps the first two operands
  /// of MI and returns it.
  ///
  /// If a target wants to make more aggressive changes, they can construct and
  /// return a new machine instruction.  If an instruction cannot commute, it
  /// can also return null.
  ///
  /// If NewMI is true, then a new machine instruction must be created.
  ///
  virtual MachineInstr *commuteInstruction(MachineInstr *MI,
                                           bool NewMI = false) const = 0;

  /// CommuteChangesDestination - Return true if commuting the specified
  /// instruction will also changes the destination operand. Also return the
  /// current operand index of the would be new destination register by
  /// reference. This can happen when the commutable instruction is also a
  /// two-address instruction.
  virtual bool CommuteChangesDestination(MachineInstr *MI,
                                         unsigned &OpIdx) const = 0;

  /// AnalyzeBranch - Analyze the branching code at the end of MBB, returning
  /// true if it cannot be understood (e.g. it's a switch dispatch or isn't
  /// implemented for a target).  Upon success, this returns false and returns
  /// with the following information in various cases:
  ///
  /// 1. If this block ends with no branches (it just falls through to its succ)
  ///    just return false, leaving TBB/FBB null.
  /// 2. If this block ends with only an unconditional branch, it sets TBB to be
  ///    the destination block.
  /// 3. If this block ends with an conditional branch and it falls through to
  ///    an successor block, it sets TBB to be the branch destination block and a
  ///    list of operands that evaluate the condition. These
  ///    operands can be passed to other TargetInstrInfo methods to create new
  ///    branches.
  /// 4. If this block ends with an conditional branch and an unconditional
  ///    block, it returns the 'true' destination in TBB, the 'false' destination
  ///    in FBB, and a list of operands that evaluate the condition. These
  ///    operands can be passed to other TargetInstrInfo methods to create new
  ///    branches.
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
    assert(0 && "Target didn't implement TargetInstrInfo::RemoveBranch!"); 
    return 0;
  }
  
  /// InsertBranch - Insert a branch into the end of the specified
  /// MachineBasicBlock.  This operands to this method are the same as those
  /// returned by AnalyzeBranch.  This is invoked in cases where AnalyzeBranch
  /// returns success and when an unconditional branch (TBB is non-null, FBB is
  /// null, Cond is empty) needs to be inserted. It returns the number of
  /// instructions inserted.
  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                            MachineBasicBlock *FBB,
                            const SmallVectorImpl<MachineOperand> &Cond) const {
    assert(0 && "Target didn't implement TargetInstrInfo::InsertBranch!"); 
    return 0;
  }
  
  /// copyRegToReg - Emit instructions to copy between a pair of registers. It
  /// returns false if the target does not how to copy between the specified
  /// registers.
  virtual bool copyRegToReg(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            unsigned DestReg, unsigned SrcReg,
                            const TargetRegisterClass *DestRC,
                            const TargetRegisterClass *SrcRC) const {
    assert(0 && "Target didn't implement TargetInstrInfo::copyRegToReg!");
    return false;
  }
  
  /// storeRegToStackSlot - Store the specified register of the given register
  /// class to the specified stack frame index. The store instruction is to be
  /// added to the given machine basic block before the specified machine
  /// instruction. If isKill is true, the register operand is the last use and
  /// must be marked kill.
  virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned SrcReg, bool isKill, int FrameIndex,
                                   const TargetRegisterClass *RC) const {
    assert(0 && "Target didn't implement TargetInstrInfo::storeRegToStackSlot!");
  }

  /// storeRegToAddr - Store the specified register of the given register class
  /// to the specified address. The store instruction is to be added to the
  /// given machine basic block before the specified machine instruction. If
  /// isKill is true, the register operand is the last use and must be marked
  /// kill.
  virtual void storeRegToAddr(MachineFunction &MF, unsigned SrcReg, bool isKill,
                              SmallVectorImpl<MachineOperand> &Addr,
                              const TargetRegisterClass *RC,
                              SmallVectorImpl<MachineInstr*> &NewMIs) const {
    assert(0 && "Target didn't implement TargetInstrInfo::storeRegToAddr!");
  }

  /// loadRegFromStackSlot - Load the specified register of the given register
  /// class from the specified stack frame index. The load instruction is to be
  /// added to the given machine basic block before the specified machine
  /// instruction.
  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
                                    unsigned DestReg, int FrameIndex,
                                    const TargetRegisterClass *RC) const {
    assert(0 && "Target didn't implement TargetInstrInfo::loadRegFromStackSlot!");
  }

  /// loadRegFromAddr - Load the specified register of the given register class
  /// class from the specified address. The load instruction is to be added to
  /// the given machine basic block before the specified machine instruction.
  virtual void loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                               SmallVectorImpl<MachineOperand> &Addr,
                               const TargetRegisterClass *RC,
                               SmallVectorImpl<MachineInstr*> &NewMIs) const {
    assert(0 && "Target didn't implement TargetInstrInfo::loadRegFromAddr!");
  }
  
  /// spillCalleeSavedRegisters - Issues instruction(s) to spill all callee
  /// saved registers and returns true if it isn't possible / profitable to do
  /// so by issuing a series of store instructions via
  /// storeRegToStackSlot(). Returns false otherwise.
  virtual bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
    return false;
  }

  /// restoreCalleeSavedRegisters - Issues instruction(s) to restore all callee
  /// saved registers and returns true if it isn't possible / profitable to do
  /// so by issuing a series of load instructions via loadRegToStackSlot().
  /// Returns false otherwise.
  virtual bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
    return false;
  }
  
  /// foldMemoryOperand - Attempt to fold a load or store of the specified stack
  /// slot into the specified machine instruction for the specified operand(s).
  /// If this is possible, a new instruction is returned with the specified
  /// operand folded, otherwise NULL is returned. The client is responsible for
  /// removing the old instruction and adding the new one in the instruction
  /// stream.
  MachineInstr* foldMemoryOperand(MachineFunction &MF,
                                  MachineInstr* MI,
                                  const SmallVectorImpl<unsigned> &Ops,
                                  int FrameIndex) const;

  /// foldMemoryOperand - Same as the previous version except it allows folding
  /// of any load and store from / to any address, not just from a specific
  /// stack slot.
  MachineInstr* foldMemoryOperand(MachineFunction &MF,
                                  MachineInstr* MI,
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
                            const SmallVectorImpl<unsigned> &Ops) const {
    return false;
  }

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
  /// possible.
  virtual unsigned getOpcodeAfterMemoryUnfold(unsigned Opc,
                                      bool UnfoldLoad, bool UnfoldStore) const {
    return 0;
  }
  
  /// BlockHasNoFallThrough - Return true if the specified block does not
  /// fall-through into its successor block.  This is primarily used when a
  /// branch is unanalyzable.  It is useful for things like unconditional
  /// indirect branches (jump tables).
  virtual bool BlockHasNoFallThrough(const MachineBasicBlock &MBB) const {
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
                          MachineBasicBlock::iterator MI) const {
    assert(0 && "Target didn't implement insertNoop!");
    abort();
  }

  /// isPredicated - Returns true if the instruction is already predicated.
  ///
  virtual bool isPredicated(const MachineInstr *MI) const {
    return false;
  }

  /// isUnpredicatedTerminator - Returns true if the instruction is a
  /// terminator instruction that has not been predicated.
  virtual bool isUnpredicatedTerminator(const MachineInstr *MI) const;

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

  /// IgnoreRegisterClassBarriers - Returns true if pre-register allocation
  /// live interval splitting pass should ignore barriers of the specified
  /// register class.
  virtual bool IgnoreRegisterClassBarriers(const TargetRegisterClass *RC) const{
    return true;
  }

  /// getPointerRegClass - Returns a TargetRegisterClass used for pointer
  /// values.
  virtual const TargetRegisterClass *getPointerRegClass() const {
    assert(0 && "Target didn't implement getPointerRegClass!");
    abort();
    return 0; // Must return a value in order to compile with VS 2005
  }

  /// GetInstSize - Returns the size of the specified Instruction.
  /// 
  virtual unsigned GetInstSizeInBytes(const MachineInstr *MI) const {
    assert(0 && "Target didn't implement TargetInstrInfo::GetInstSize!");
    return 0;
  }

  /// GetFunctionSizeInBytes - Returns the size of the specified MachineFunction.
  /// 
  virtual unsigned GetFunctionSizeInBytes(const MachineFunction &MF) const = 0;
};

/// TargetInstrInfoImpl - This is the default implementation of
/// TargetInstrInfo, which just provides a couple of default implementations
/// for various methods.  This separated out because it is implemented in
/// libcodegen, not in libtarget.
class TargetInstrInfoImpl : public TargetInstrInfo {
protected:
  TargetInstrInfoImpl(const TargetInstrDesc *desc, unsigned NumOpcodes)
  : TargetInstrInfo(desc, NumOpcodes) {}
public:
  virtual MachineInstr *commuteInstruction(MachineInstr *MI,
                                           bool NewMI = false) const;
  virtual bool CommuteChangesDestination(MachineInstr *MI,
                                         unsigned &OpIdx) const;
  virtual bool PredicateInstruction(MachineInstr *MI,
                            const SmallVectorImpl<MachineOperand> &Pred) const;
  virtual void reMaterialize(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MI,
                             unsigned DestReg,
                             const MachineInstr *Orig) const;
  virtual unsigned GetFunctionSizeInBytes(const MachineFunction &MF) const;
};

} // End llvm namespace

#endif
