//===-- SnippetGenerator.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <array>
#include <string>

#include "Assembler.h"
#include "MCInstrDescView.h"
#include "SnippetGenerator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"

namespace exegesis {

SnippetGeneratorFailure::SnippetGeneratorFailure(const llvm::Twine &S)
    : llvm::StringError(S, llvm::inconvertibleErrorCode()) {}

SnippetGenerator::SnippetGenerator(const LLVMState &State)
    : State(State), RATC(State.getRegInfo(),
                         getFunctionReservedRegs(State.getTargetMachine())) {}

SnippetGenerator::~SnippetGenerator() = default;

llvm::Expected<std::vector<BenchmarkCode>>
SnippetGenerator::generateConfigurations(unsigned Opcode) const {
  if (auto E = generateCodeTemplate(Opcode)) {
    CodeTemplate &CT = E.get();
    std::vector<BenchmarkCode> Output;
    // TODO: Generate as many BenchmarkCode as needed.
    {
      BenchmarkCode BC;
      BC.Info = CT.Info;
      for (InstructionBuilder &IB : CT.Instructions) {
        IB.randomizeUnsetVariables(
            CT.ScratchSpacePointerInReg
                ? RATC.getRegister(CT.ScratchSpacePointerInReg).aliasedBits()
                : RATC.emptyRegisters());
        BC.Instructions.push_back(IB.build());
      }
      if (CT.ScratchSpacePointerInReg)
        BC.LiveIns.push_back(CT.ScratchSpacePointerInReg);
      BC.RegisterInitialValues = computeRegisterInitialValues(CT.Instructions);
      Output.push_back(std::move(BC));
    }
    return Output;
  } else
    return E.takeError();
}

std::vector<RegisterValue> SnippetGenerator::computeRegisterInitialValues(
    const std::vector<InstructionBuilder> &Instructions) const {
  // Collect all register uses and create an assignment for each of them.
  // Ignore memory operands which are handled separately.
  // Loop invariant: DefinedRegs[i] is true iif it has been set at least once
  // before the current instruction.
  llvm::BitVector DefinedRegs = RATC.emptyRegisters();
  std::vector<RegisterValue> RIV;
  for (const InstructionBuilder &IB : Instructions) {
    // Returns the register that this Operand sets or uses, or 0 if this is not
    // a register.
    const auto GetOpReg = [&IB](const Operand &Op) -> unsigned {
      if (Op.IsMem)
        return 0;
      if (Op.ImplicitReg)
        return *Op.ImplicitReg;
      if (Op.IsExplicit && IB.getValueFor(Op).isReg())
        return IB.getValueFor(Op).getReg();
      return 0;
    };
    // Collect used registers that have never been def'ed.
    for (const Operand &Op : IB.Instr.Operands) {
      if (!Op.IsDef) {
        const unsigned Reg = GetOpReg(Op);
        if (Reg > 0 && !DefinedRegs.test(Reg)) {
          RIV.push_back(RegisterValue{Reg, llvm::APInt()});
          DefinedRegs.set(Reg);
        }
      }
    }
    // Mark defs as having been def'ed.
    for (const Operand &Op : IB.Instr.Operands) {
      if (Op.IsDef) {
        const unsigned Reg = GetOpReg(Op);
        if (Reg > 0)
          DefinedRegs.set(Reg);
      }
    }
  }
  return RIV;
}

llvm::Expected<CodeTemplate> SnippetGenerator::generateSelfAliasingCodeTemplate(
    const Instruction &Instr) const {
  const AliasingConfigurations SelfAliasing(Instr, Instr);
  if (SelfAliasing.empty()) {
    return llvm::make_error<SnippetGeneratorFailure>("empty self aliasing");
  }
  CodeTemplate CT;
  InstructionBuilder IB(Instr);
  if (SelfAliasing.hasImplicitAliasing()) {
    CT.Info = "implicit Self cycles, picking random values.";
  } else {
    CT.Info = "explicit self cycles, selecting one aliasing Conf.";
    // This is a self aliasing instruction so defs and uses are from the same
    // instance, hence twice IB in the following call.
    setRandomAliasing(SelfAliasing, IB, IB);
  }
  CT.Instructions.push_back(std::move(IB));
  return std::move(CT);
}

llvm::Expected<CodeTemplate>
SnippetGenerator::generateUnconstrainedCodeTemplate(const Instruction &Instr,
                                                    llvm::StringRef Msg) const {
  CodeTemplate CT;
  CT.Info = llvm::formatv("{0}, repeating an unconstrained assignment", Msg);
  CT.Instructions.emplace_back(Instr);
  return std::move(CT);
}
} // namespace exegesis
