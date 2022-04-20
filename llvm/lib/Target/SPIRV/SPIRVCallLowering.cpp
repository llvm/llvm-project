//===--- SPIRVCallLowering.cpp - Call lowering ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lowering of LLVM calls to machine code calls for
// GlobalISel.
//
//===----------------------------------------------------------------------===//

#include "SPIRVCallLowering.h"
#include "SPIRV.h"
#include "SPIRVISelLowering.h"
#include "SPIRVRegisterInfo.h"
#include "SPIRVSubtarget.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

using namespace llvm;

SPIRVCallLowering::SPIRVCallLowering(const SPIRVTargetLowering &TLI)
    : CallLowering(&TLI) {}

bool SPIRVCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                    const Value *Val, ArrayRef<Register> VRegs,
                                    FunctionLoweringInfo &FLI,
                                    Register SwiftErrorVReg) const {
  // Currently all return types should use a single register.
  // TODO: handle the case of multiple registers.
  if (VRegs.size() > 1)
    return false;
  if (Val) {
    MIRBuilder.buildInstr(SPIRV::OpReturnValue).addUse(VRegs[0]);
    return true;
  }
  MIRBuilder.buildInstr(SPIRV::OpReturn);
  return true;
}

bool SPIRVCallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                             const Function &F,
                                             ArrayRef<ArrayRef<Register>> VRegs,
                                             FunctionLoweringInfo &FLI) const {
  auto MRI = MIRBuilder.getMRI();
  // Assign types and names to all args, and store their types for later.
  SmallVector<Register, 4> ArgTypeVRegs;
  if (VRegs.size() > 0) {
    unsigned i = 0;
    for (const auto &Arg : F.args()) {
      // Currently formal args should use single registers.
      // TODO: handle the case of multiple registers.
      if (VRegs[i].size() > 1)
        return false;
      ArgTypeVRegs.push_back(
          MRI->createGenericVirtualRegister(LLT::scalar(32)));
      ++i;
    }
  }

  // Generate a SPIR-V type for the function.
  Register FuncVReg = MRI->createGenericVirtualRegister(LLT::scalar(32));
  MRI->setRegClass(FuncVReg, &SPIRV::IDRegClass);

  MIRBuilder.buildInstr(SPIRV::OpFunction)
      .addDef(FuncVReg)
      .addUse(MRI->createGenericVirtualRegister(LLT::scalar(32)))
      .addImm(0)
      .addUse(MRI->createGenericVirtualRegister(LLT::scalar(32)));

  // Add OpFunctionParameters.
  const unsigned NumArgs = ArgTypeVRegs.size();
  for (unsigned i = 0; i < NumArgs; ++i) {
    assert(VRegs[i].size() == 1 && "Formal arg has multiple vregs");
    MRI->setRegClass(VRegs[i][0], &SPIRV::IDRegClass);
    MIRBuilder.buildInstr(SPIRV::OpFunctionParameter)
        .addDef(VRegs[i][0])
        .addUse(ArgTypeVRegs[i]);
  }
  return true;
}

bool SPIRVCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                  CallLoweringInfo &Info) const {
  // Currently call returns should have single vregs.
  // TODO: handle the case of multiple registers.
  if (Info.OrigRet.Regs.size() > 1)
    return false;

  Register ResVReg =
      Info.OrigRet.Regs.empty() ? Register(0) : Info.OrigRet.Regs[0];
  // Make sure there's a valid return reg, even for functions returning void.
  if (!ResVReg.isValid()) {
    ResVReg = MIRBuilder.getMRI()->createVirtualRegister(&SPIRV::IDRegClass);
  }
  // Emit the OpFunctionCall and its args.
  auto MIB = MIRBuilder.buildInstr(SPIRV::OpFunctionCall)
                 .addDef(ResVReg)
                 .addUse(MIRBuilder.getMRI()->createVirtualRegister(
                     &SPIRV::IDRegClass))
                 .add(Info.Callee);

  for (const auto &Arg : Info.OrigArgs) {
    // Currently call args should have single vregs.
    if (Arg.Regs.size() > 1)
      return false;
    MIB.addUse(Arg.Regs[0]);
  }
  return true;
}
