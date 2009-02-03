//===-- llvm/CodeGen/MachineInstr.h - MachineInstr class --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MachineInstr class, which is the
// basic representation for all target dependent machine instructions used by
// the back end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEINSTR_H
#define LLVM_CODEGEN_MACHINEINSTR_H

#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/Target/TargetInstrDesc.h"
#include "llvm/CodeGen/DebugLoc.h"
#include <list>
#include <vector>

namespace llvm {

class TargetInstrDesc;
class TargetInstrInfo;
class TargetRegisterInfo;
class MachineFunction;

//===----------------------------------------------------------------------===//
/// MachineInstr - Representation of each machine instruction.
///
class MachineInstr : public ilist_node<MachineInstr> {
  const TargetInstrDesc *TID;           // Instruction descriptor.
  unsigned short NumImplicitOps;        // Number of implicit operands (which
                                        // are determined at construction time).

  std::vector<MachineOperand> Operands; // the operands
  std::list<MachineMemOperand> MemOperands; // information on memory references
  MachineBasicBlock *Parent;            // Pointer to the owning basic block.
  DebugLoc debugLoc;                    // Source line information.

  // OperandComplete - Return true if it's illegal to add a new operand
  bool OperandsComplete() const;

  MachineInstr(const MachineInstr&);   // DO NOT IMPLEMENT
  void operator=(const MachineInstr&); // DO NOT IMPLEMENT

  // Intrusive list support
  friend struct ilist_traits<MachineInstr>;
  friend struct ilist_traits<MachineBasicBlock>;
  friend struct ilist_sentinel_traits<MachineInstr>;
  void setParent(MachineBasicBlock *P) { Parent = P; }

  /// MachineInstr ctor - This constructor creates a copy of the given
  /// MachineInstr in the given MachineFunction.
  MachineInstr(MachineFunction &, const MachineInstr &);

  /// MachineInstr ctor - This constructor creates a dummy MachineInstr with
  /// TID NULL and no operands.
  MachineInstr();

  // The next two constructors have DebugLoc and non-DebugLoc versions;
  // over time, the non-DebugLoc versions should be phased out and eventually
  // removed.

  /// MachineInstr ctor - This constructor create a MachineInstr and add the
  /// implicit operands.  It reserves space for number of operands specified by
  /// TargetInstrDesc.  The version with a DebugLoc should be preferred.
  explicit MachineInstr(const TargetInstrDesc &TID, bool NoImp = false);

  /// MachineInstr ctor - Work exactly the same as the ctor above, except that
  /// the MachineInstr is created and added to the end of the specified basic
  /// block.  The version with a DebugLoc should be preferred.
  ///
  MachineInstr(MachineBasicBlock *MBB, const TargetInstrDesc &TID);

  /// MachineInstr ctor - This constructor create a MachineInstr and add the
  /// implicit operands.  It reserves space for number of operands specified by
  /// TargetInstrDesc.  An explicit DebugLoc is supplied.
  explicit MachineInstr(const TargetInstrDesc &TID, const DebugLoc dl, 
                        bool NoImp = false);

  /// MachineInstr ctor - Work exactly the same as the ctor above, except that
  /// the MachineInstr is created and added to the end of the specified basic
  /// block.
  ///
  MachineInstr(MachineBasicBlock *MBB, const DebugLoc dl, 
               const TargetInstrDesc &TID);

  ~MachineInstr();

  // MachineInstrs are pool-allocated and owned by MachineFunction.
  friend class MachineFunction;

public:
  const MachineBasicBlock* getParent() const { return Parent; }
  MachineBasicBlock* getParent() { return Parent; }

  /// getDebugLoc - Returns the debug location id of this MachineInstr.
  ///
  const DebugLoc getDebugLoc() const { return debugLoc; }
  
  /// getDesc - Returns the target instruction descriptor of this
  /// MachineInstr.
  const TargetInstrDesc &getDesc() const { return *TID; }

  /// getOpcode - Returns the opcode of this MachineInstr.
  ///
  int getOpcode() const { return TID->Opcode; }

  /// Access to explicit operands of the instruction.
  ///
  unsigned getNumOperands() const { return (unsigned)Operands.size(); }

  const MachineOperand& getOperand(unsigned i) const {
    assert(i < getNumOperands() && "getOperand() out of range!");
    return Operands[i];
  }
  MachineOperand& getOperand(unsigned i) {
    assert(i < getNumOperands() && "getOperand() out of range!");
    return Operands[i];
  }

  /// getNumExplicitOperands - Returns the number of non-implicit operands.
  ///
  unsigned getNumExplicitOperands() const;
  
  /// Access to memory operands of the instruction
  std::list<MachineMemOperand>::iterator memoperands_begin()
  { return MemOperands.begin(); }
  std::list<MachineMemOperand>::iterator memoperands_end()
  { return MemOperands.end(); }
  std::list<MachineMemOperand>::const_iterator memoperands_begin() const
  { return MemOperands.begin(); }
  std::list<MachineMemOperand>::const_iterator memoperands_end() const
  { return MemOperands.end(); }
  bool memoperands_empty() const { return MemOperands.empty(); }

  /// hasOneMemOperand - Return true if this instruction has exactly one
  /// MachineMemOperand.
  bool hasOneMemOperand() const {
    return !memoperands_empty() &&
           next(memoperands_begin()) == memoperands_end();
  }

  /// isIdenticalTo - Return true if this instruction is identical to (same
  /// opcode and same operands as) the specified instruction.
  bool isIdenticalTo(const MachineInstr *Other) const {
    if (Other->getOpcode() != getOpcode() ||
        Other->getNumOperands() != getNumOperands())
      return false;
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
      if (!getOperand(i).isIdenticalTo(Other->getOperand(i)))
        return false;
    return true;
  }

  /// removeFromParent - This method unlinks 'this' from the containing basic
  /// block, and returns it, but does not delete it.
  MachineInstr *removeFromParent();
  
  /// eraseFromParent - This method unlinks 'this' from the containing basic
  /// block and deletes it.
  void eraseFromParent();

  /// isLabel - Returns true if the MachineInstr represents a label.
  ///
  bool isLabel() const;

  /// isDebugLabel - Returns true if the MachineInstr represents a debug label.
  ///
  bool isDebugLabel() const;

  /// readsRegister - Return true if the MachineInstr reads the specified
  /// register. If TargetRegisterInfo is passed, then it also checks if there
  /// is a read of a super-register.
  bool readsRegister(unsigned Reg, const TargetRegisterInfo *TRI = NULL) const {
    return findRegisterUseOperandIdx(Reg, false, TRI) != -1;
  }

  /// killsRegister - Return true if the MachineInstr kills the specified
  /// register. If TargetRegisterInfo is passed, then it also checks if there is
  /// a kill of a super-register.
  bool killsRegister(unsigned Reg, const TargetRegisterInfo *TRI = NULL) const {
    return findRegisterUseOperandIdx(Reg, true, TRI) != -1;
  }

  /// modifiesRegister - Return true if the MachineInstr modifies the
  /// specified register. If TargetRegisterInfo is passed, then it also checks
  /// if there is a def of a super-register.
  bool modifiesRegister(unsigned Reg,
                        const TargetRegisterInfo *TRI = NULL) const {
    return findRegisterDefOperandIdx(Reg, false, TRI) != -1;
  }

  /// registerDefIsDead - Returns true if the register is dead in this machine
  /// instruction. If TargetRegisterInfo is passed, then it also checks
  /// if there is a dead def of a super-register.
  bool registerDefIsDead(unsigned Reg,
                         const TargetRegisterInfo *TRI = NULL) const {
    return findRegisterDefOperandIdx(Reg, true, TRI) != -1;
  }

  /// findRegisterUseOperandIdx() - Returns the operand index that is a use of
  /// the specific register or -1 if it is not found. It further tightening
  /// the search criteria to a use that kills the register if isKill is true.
  int findRegisterUseOperandIdx(unsigned Reg, bool isKill = false,
                                const TargetRegisterInfo *TRI = NULL) const;

  /// findRegisterUseOperand - Wrapper for findRegisterUseOperandIdx, it returns
  /// a pointer to the MachineOperand rather than an index.
  MachineOperand *findRegisterUseOperand(unsigned Reg, bool isKill = false,
                                         const TargetRegisterInfo *TRI = NULL) {
    int Idx = findRegisterUseOperandIdx(Reg, isKill, TRI);
    return (Idx == -1) ? NULL : &getOperand(Idx);
  }
  
  /// findRegisterDefOperandIdx() - Returns the operand index that is a def of
  /// the specified register or -1 if it is not found. If isDead is true, defs
  /// that are not dead are skipped. If TargetRegisterInfo is non-null, then it
  /// also checks if there is a def of a super-register.
  int findRegisterDefOperandIdx(unsigned Reg, bool isDead = false,
                                const TargetRegisterInfo *TRI = NULL) const;

  /// findRegisterDefOperand - Wrapper for findRegisterDefOperandIdx, it returns
  /// a pointer to the MachineOperand rather than an index.
  MachineOperand *findRegisterDefOperand(unsigned Reg, bool isDead = false,
                                         const TargetRegisterInfo *TRI = NULL) {
    int Idx = findRegisterDefOperandIdx(Reg, isDead, TRI);
    return (Idx == -1) ? NULL : &getOperand(Idx);
  }

  /// findFirstPredOperandIdx() - Find the index of the first operand in the
  /// operand list that is used to represent the predicate. It returns -1 if
  /// none is found.
  int findFirstPredOperandIdx() const;
  
  /// isRegReDefinedByTwoAddr - Given the index of a register def operand,
  /// check if the register def is a re-definition due to two addr elimination.
  bool isRegReDefinedByTwoAddr(unsigned DefIdx) const;

  /// copyKillDeadInfo - Copies kill / dead operand properties from MI.
  ///
  void copyKillDeadInfo(const MachineInstr *MI);

  /// copyPredicates - Copies predicate operand(s) from MI.
  void copyPredicates(const MachineInstr *MI);

  /// addRegisterKilled - We have determined MI kills a register. Look for the
  /// operand that uses it and mark it as IsKill. If AddIfNotFound is true,
  /// add a implicit operand if it's not found. Returns true if the operand
  /// exists / is added.
  bool addRegisterKilled(unsigned IncomingReg,
                         const TargetRegisterInfo *RegInfo,
                         bool AddIfNotFound = false);
  
  /// addRegisterDead - We have determined MI defined a register without a use.
  /// Look for the operand that defines it and mark it as IsDead. If
  /// AddIfNotFound is true, add a implicit operand if it's not found. Returns
  /// true if the operand exists / is added.
  bool addRegisterDead(unsigned IncomingReg, const TargetRegisterInfo *RegInfo,
                       bool AddIfNotFound = false);

  /// isSafeToMove - Return true if it is safe to move this instruction. If
  /// SawStore is set to true, it means that there is a store (or call) between
  /// the instruction's location and its intended destination.
  bool isSafeToMove(const TargetInstrInfo *TII, bool &SawStore) const;

  /// isSafeToReMat - Return true if it's safe to rematerialize the specified
  /// instruction which defined the specified register instead of copying it.
  bool isSafeToReMat(const TargetInstrInfo *TII, unsigned DstReg) const;

  /// hasVolatileMemoryRef - Return true if this instruction may have a
  /// volatile memory reference, or if the information describing the
  /// memory reference is not available. Return false if it is known to
  /// have no volatile memory references.
  bool hasVolatileMemoryRef() const;

  //
  // Debugging support
  //
  void print(std::ostream *OS, const TargetMachine *TM) const {
    if (OS) print(*OS, TM);
  }
  void print(std::ostream &OS, const TargetMachine *TM = 0) const;
  void print(std::ostream *OS) const { if (OS) print(*OS); }
  void print(raw_ostream *OS, const TargetMachine *TM) const {
    if (OS) print(*OS, TM);
  }
  void print(raw_ostream &OS, const TargetMachine *TM = 0) const;
  void print(raw_ostream *OS) const { if (OS) print(*OS); }
  void dump() const;

  //===--------------------------------------------------------------------===//
  // Accessors used to build up machine instructions.

  /// addOperand - Add the specified operand to the instruction.  If it is an
  /// implicit operand, it is added to the end of the operand list.  If it is
  /// an explicit operand it is added at the end of the explicit operand list
  /// (before the first implicit operand). 
  void addOperand(const MachineOperand &Op);
  
  /// setDesc - Replace the instruction descriptor (thus opcode) of
  /// the current instruction with a new one.
  ///
  void setDesc(const TargetInstrDesc &tid) { TID = &tid; }

  /// setDebugLoc - Replace current source information with new such.
  /// Avoid using this, the constructor argument is preferable.
  ///
  void setDebugLoc(const DebugLoc dl) { debugLoc = dl; }

  /// RemoveOperand - Erase an operand  from an instruction, leaving it with one
  /// fewer operand than it started with.
  ///
  void RemoveOperand(unsigned i);

  /// addMemOperand - Add a MachineMemOperand to the machine instruction,
  /// referencing arbitrary storage.
  void addMemOperand(MachineFunction &MF,
                     const MachineMemOperand &MO);

  /// clearMemOperands - Erase all of this MachineInstr's MachineMemOperands.
  void clearMemOperands(MachineFunction &MF);

private:
  /// getRegInfo - If this instruction is embedded into a MachineFunction,
  /// return the MachineRegisterInfo object for the current function, otherwise
  /// return null.
  MachineRegisterInfo *getRegInfo();

  /// addImplicitDefUseOperands - Add all implicit def and use operands to
  /// this instruction.
  void addImplicitDefUseOperands();
  
  /// RemoveRegOperandsFromUseLists - Unlink all of the register operands in
  /// this instruction from their respective use lists.  This requires that the
  /// operands already be on their use lists.
  void RemoveRegOperandsFromUseLists();
  
  /// AddRegOperandsToUseLists - Add all of the register operands in
  /// this instruction from their respective use lists.  This requires that the
  /// operands not be on their use lists yet.
  void AddRegOperandsToUseLists(MachineRegisterInfo &RegInfo);
};

//===----------------------------------------------------------------------===//
// Debugging Support

inline std::ostream& operator<<(std::ostream &OS, const MachineInstr &MI) {
  MI.print(OS);
  return OS;
}

inline raw_ostream& operator<<(raw_ostream &OS, const MachineInstr &MI) {
  MI.print(OS);
  return OS;
}

} // End llvm namespace

#endif
