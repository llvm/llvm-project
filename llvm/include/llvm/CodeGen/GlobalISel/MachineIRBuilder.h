//===-- llvm/CodeGen/GlobalISel/MachineIRBuilder.h - MIBuilder --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares the MachineIRBuilder class.
/// This is a helper class to build MachineInstr.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_MACHINEIRBUILDER_H
#define LLVM_CODEGEN_GLOBALISEL_MACHINEIRBUILDER_H

#include "llvm/CodeGen/GlobalISel/Types.h"

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/IR/DebugLoc.h"

namespace llvm {

// Forward declarations.
class MachineFunction;
class MachineInstr;
class TargetInstrInfo;

/// Helper class to build MachineInstr.
/// It keeps internally the insertion point and debug location for all
/// the new instructions we want to create.
/// This information can be modify via the related setters.
class MachineIRBuilder {
  /// MachineFunction under construction.
  MachineFunction *MF;
  /// Information used to access the description of the opcodes.
  const TargetInstrInfo *TII;
  /// Debug location to be set to any instruction we create.
  DebugLoc DL;

  /// Fields describing the insertion point.
  /// @{
  MachineBasicBlock *MBB;
  MachineInstr *MI;
  bool Before;
  /// @}

  const TargetInstrInfo &getTII() {
    assert(TII && "TargetInstrInfo is not set");
    return *TII;
  }

public:
  /// Getter for the function we currently build.
  MachineFunction &getMF() {
    assert(MF && "MachineFunction is not set");
    return *MF;
  }

  /// Getter for the basic block we currently build.
  MachineBasicBlock &getMBB() {
    assert(MBB && "MachineBasicBlock is not set");
    return *MBB;
  }

  /// Current insertion point for new instructions.
  MachineBasicBlock::iterator getInsertPt();

  /// Setters for the insertion point.
  /// @{
  /// Set the MachineFunction where to build instructions.
  void setMF(MachineFunction &);

  /// Set the insertion point to the beginning (\p Beginning = true) or end
  /// (\p Beginning = false) of \p MBB.
  /// \pre \p MBB must be contained by getMF().
  void setMBB(MachineBasicBlock &MBB, bool Beginning = false);

  /// Set the insertion point to before (\p Before = true) or after
  /// (\p Before = false) \p MI.
  /// \pre MI must be in getMF().
  void setInstr(MachineInstr &MI, bool Before = false);
  /// @}

  /// Set the debug location to \p DL for all the next build instructions.
  void setDebugLoc(const DebugLoc &DL) { this->DL = DL; }

  /// Build and insert <empty> = \p Opcode [\p Ty] <empty>.
  /// \p Ty is the type of the instruction if \p Opcode describes
  /// a generic machine instruction. \p Ty must be nullptr if \p Opcode
  /// does not describe a generic instruction.
  /// The insertion point is the one set by the last call of either
  /// setBasicBlock or setMI.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre Ty == nullptr or isPreISelGenericOpcode(Opcode)
  ///
  /// \return The newly created instruction.
  MachineInstr *buildInstr(unsigned Opcode, Type *Ty);

  /// Build and insert <empty> = \p Opcode [\p Ty] \p BB.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre Ty == nullptr or isPreISelGenericOpcode(Opcode)
  ///
  /// \return The newly created instruction.
  MachineInstr *buildInstr(unsigned Opcode, Type *Ty, MachineBasicBlock &BB);

  /// Build and insert \p Res<def> = \p Opcode [\p Ty] \p Op0, \p Op1.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre Ty == nullptr or isPreISelGenericOpcode(Opcode)
  ///
  /// \return The newly created instruction.
  MachineInstr *buildInstr(unsigned Opcode, Type *Ty, unsigned Res,
                           unsigned Op0, unsigned Op1);

  /// Build and insert \p Res<def> = \p Opcode \p Op0, \p Op1.
  /// I.e., instruction with a non-generic opcode.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre not isPreISelGenericOpcode(\p Opcode)
  ///
  /// \return The newly created instruction.
  MachineInstr *buildInstr(unsigned Opcode, unsigned Res, unsigned Op0,
                           unsigned Op1);

  /// Build and insert \p Res<def> = \p Opcode \p Op0.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre not isPreISelGenericOpcode(\p Opcode)
  ///
  /// \return The newly created instruction.
  MachineInstr *buildInstr(unsigned Opcode, unsigned Res, unsigned Op0);

  /// Build and insert <empty> = \p Opcode <empty>.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre not isPreISelGenericOpcode(\p Opcode)
  ///
  /// \return The newly created instruction.
  MachineInstr *buildInstr(unsigned Opcode);
};

} // End namespace llvm.
#endif // LLVM_CODEGEN_GLOBALISEL_MACHINEIRBUILDER_H
