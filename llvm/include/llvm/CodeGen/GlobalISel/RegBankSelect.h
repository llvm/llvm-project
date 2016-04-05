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

#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {
// Forward declarations.
class RegisterBankInfo;

/// This pass implements the reg bank selector pass used in the GlobalISel
/// pipeline. At the end of this pass, all register operands have been assigned
class RegBankSelect : public MachineFunctionPass {
public:
  static char ID;

private:
  /// Interface to the target lowering info related
  /// to register banks.
  const RegisterBankInfo *RBI;

public:
  // Ctor, nothing fancy.
  RegBankSelect();

  const char *getPassName() const override {
    return "RegBankSelect";
  }

  // Simplified algo:
  //   RBI = MF.subtarget.getRegBankInfo()
  //   MIRBuilder.reset(MF)
  //   for each bb in MF
  //     for each inst in bb
  //       MappingCosts = RBI.getMapping(inst);
  //       Idx = findIdxOfMinCost(MappingCosts)
  //       CurRegBank = MappingCosts[Idx].RegBank
  //       MRI.setRegBank(inst.getOperand(0).getReg(), CurRegBank)
  //       for each argument in inst
  //         if (CurRegBank != argument.RegBank)
  //           ArgReg = argument.getReg()
  //           Tmp = MRI.createNewVirtual(MRI.getSize(ArgReg), CurRegBank)
  //           MIRBuilder.buildInstr(COPY, Tmp, ArgReg)
  //           inst.getOperand(argument.getOperandNo()).setReg(Tmp)
  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // End namespace llvm.

#endif
