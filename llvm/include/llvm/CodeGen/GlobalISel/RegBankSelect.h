//== llvm/CodeGen/GlobalISel/RegBankSelect.h - Reg Bank Selector -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file describes the interface of the MachineFunctionPass
/// responsible for assigning the generic virtual registers to register bank.

/// By default, the reg bank selector relies on local decisions to
/// assign the register bank. In other words, it looks at one instruction
/// at a time to decide where the operand of that instruction should live.
///
/// At higher optimization level, we could imagine that the reg bank selector
/// would use more global analysis and do crazier thing like duplicating
/// instructions and so on. This is future work.
///
/// For now, the pass uses a greedy algorithm to decide where the operand
/// of an instruction should live. It asks the target which banks may be
/// used for each operand of the instruction and what is the cost. Then,
/// it chooses the solution which minimize the cost of the instruction plus
/// the cost of any move that may be needed to to the values into the right
/// register bank.
/// In other words, the cost for an instruction on a register bank RegBank
/// is: Cost of I on RegBank plus the sum of the cost for bringing the
/// input operands from their current register bank to RegBank.
/// Thus, the following formula:
/// cost(I, RegBank) = cost(I.Opcode, RegBank) +
///    sum(for each arg in I.arguments: costCrossCopy(arg.RegBank, RegBank))
///
/// E.g., Let say we are assigning the register bank for the instruction
/// defining v2.
/// v0(A_REGBANK) = ...
/// v1(A_REGBANK) = ...
/// v2 = G_ADD i32 v0, v1 <-- MI
///
/// The target may say it can generate G_ADD i32 on register bank A and B
/// with a cost of respectively 5 and 1.
/// Then, let say the cost of a cross register bank copies from A to B is 1.
/// The reg bank selector would compare the following two costs:
/// cost(MI, A_REGBANK) = cost(G_ADD, A_REGBANK) + cost(v0.RegBank, A_REGBANK) +
///    cost(v1.RegBank, A_REGBANK)
///                     = 5 + cost(A_REGBANK, A_REGBANK) + cost(A_REGBANK,
///                                                             A_REGBANK)
///                     = 5 + 0 + 0 = 5
/// cost(MI, B_REGBANK) = cost(G_ADD, B_REGBANK) + cost(v0.RegBank, B_REGBANK) +
///    cost(v1.RegBank, B_REGBANK)
///                     = 1 + cost(A_REGBANK, B_REGBANK) + cost(A_REGBANK,
///                                                             B_REGBANK)
///                     = 1 + 1 + 1 = 3
/// Therefore, in this specific example, the reg bank selector would choose
/// bank B for MI.
/// v0(A_REGBANK) = ...
/// v1(A_REGBANK) = ...
/// tmp0(B_REGBANK) = COPY v0
/// tmp1(B_REGBANK) = COPY v1
/// v2(B_REGBANK) = G_ADD i32 tmp0, tmp1
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_REGBANKSELECT_H
#define LLVM_CODEGEN_GLOBALISEL_REGBANKSELECT_H

#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/RegisterBankInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {
// Forward declarations.
class MachineRegisterInfo;
class TargetRegisterInfo;

/// This pass implements the reg bank selector pass used in the GlobalISel
/// pipeline. At the end of this pass, all register operands have been assigned
class RegBankSelect : public MachineFunctionPass {
public:
  static char ID;

private:
  /// Interface to the target lowering info related
  /// to register banks.
  const RegisterBankInfo *RBI;

  /// MRI contains all the register class/bank information that this
  /// pass uses and updates.
  MachineRegisterInfo *MRI;

  /// Information on the register classes for the current function.
  const TargetRegisterInfo *TRI;

  /// Helper class used for every code morphing.
  MachineIRBuilder MIRBuilder;

  /// Assign the register bank of each operand of \p MI.
  void assignInstr(MachineInstr &MI);

  /// Initialize the field members using \p MF.
  void init(MachineFunction &MF);

  /// Check if \p Reg is already assigned what is described by \p ValMapping.
  bool assignmentMatch(unsigned Reg,
                       const RegisterBankInfo::ValueMapping &ValMapping) const;

  /// Insert repairing code for \p Reg as specified by \p ValMapping.
  /// The repairing code is inserted before \p DefUseMI if \p IsDef is false
  /// and after otherwise.
  /// The transformation could be sketched as:
  /// \code
  /// ... = op Reg
  /// \endcode
  /// Becomes
  /// \code
  /// <returned reg> = COPY Reg
  /// ... = op Reg
  /// \endcode
  ///
  /// \note This is the responsability of the caller to replace \p Reg
  /// by the returned register.
  ///
  /// \return The register of the properly mapped value.
  unsigned repairReg(unsigned Reg,
                     const RegisterBankInfo::ValueMapping &ValMapping,
                     MachineInstr &DefUseMI, bool IsDef);

  /// Set the insertion point of the MIRBuilder to a safe point
  /// to insert instructions before (\p Before == true) or after
  /// \p InsertPt.
  void setSafeInsertionPoint(MachineInstr &InsertPt, bool Before);

public:
  // Ctor, nothing fancy.
  RegBankSelect();

  const char *getPassName() const override {
    return "RegBankSelect";
  }

  /// Walk through \p MF and assign a register bank to every virtual register
  /// that are still mapped to nothing.
  /// The target needs to provide a RegisterBankInfo and in particular
  /// override RegisterBankInfo::getInstrMapping.
  ///
  /// Simplified algo:
  /// \code
  ///   RBI = MF.subtarget.getRegBankInfo()
  ///   MIRBuilder.setMF(MF)
  ///   for each bb in MF
  ///     for each inst in bb
  ///       MIRBuilder.setInstr(inst)
  ///       MappingCosts = RBI.getMapping(inst);
  ///       Idx = findIdxOfMinCost(MappingCosts)
  ///       CurRegBank = MappingCosts[Idx].RegBank
  ///       MRI.setRegBank(inst.getOperand(0).getReg(), CurRegBank)
  ///       for each argument in inst
  ///         if (CurRegBank != argument.RegBank)
  ///           ArgReg = argument.getReg()
  ///           Tmp = MRI.createNewVirtual(MRI.getSize(ArgReg), CurRegBank)
  ///           MIRBuilder.buildInstr(COPY, Tmp, ArgReg)
  ///           inst.getOperand(argument.getOperandNo()).setReg(Tmp)
  /// \endcode
  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // End namespace llvm.

#endif
