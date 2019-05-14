//===-- SnippetGenerator.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <array>
#include <string>

#include "Assembler.h"
#include "MCInstrDescView.h"
#include "SnippetGenerator.h"
#include "Target.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"

namespace llvm {
namespace exegesis {

std::vector<CodeTemplate> getSingleton(CodeTemplate &&CT) {
  std::vector<CodeTemplate> Result;
  Result.push_back(std::move(CT));
  return Result;
}

SnippetGeneratorFailure::SnippetGeneratorFailure(const llvm::Twine &S)
    : llvm::StringError(S, llvm::inconvertibleErrorCode()) {}

SnippetGenerator::SnippetGenerator(const LLVMState &State) : State(State) {}

SnippetGenerator::~SnippetGenerator() = default;

llvm::Expected<std::vector<BenchmarkCode>>
SnippetGenerator::generateConfigurations(const Instruction &Instr) const {
  if (auto E = generateCodeTemplates(Instr)) {
    const auto &RATC = State.getRATC();
    std::vector<BenchmarkCode> Output;
    for (CodeTemplate &CT : E.get()) {
      const llvm::BitVector &ForbiddenRegs =
          CT.ScratchSpacePointerInReg
              ? RATC.getRegister(CT.ScratchSpacePointerInReg).aliasedBits()
              : RATC.emptyRegisters();
      // TODO: Generate as many BenchmarkCode as needed.
      {
        BenchmarkCode BC;
        BC.Info = CT.Info;
        for (InstructionTemplate &IT : CT.Instructions) {
          randomizeUnsetVariables(State.getExegesisTarget(), ForbiddenRegs, IT);
          BC.Instructions.push_back(IT.build());
        }
        if (CT.ScratchSpacePointerInReg)
          BC.LiveIns.push_back(CT.ScratchSpacePointerInReg);
        BC.RegisterInitialValues =
            computeRegisterInitialValues(CT.Instructions);
        Output.push_back(std::move(BC));
      }
    }
    return Output;
  } else
    return E.takeError();
}

std::vector<RegisterValue> SnippetGenerator::computeRegisterInitialValues(
    const std::vector<InstructionTemplate> &Instructions) const {
  // Collect all register uses and create an assignment for each of them.
  // Ignore memory operands which are handled separately.
  // Loop invariant: DefinedRegs[i] is true iif it has been set at least once
  // before the current instruction.
  llvm::BitVector DefinedRegs = State.getRATC().emptyRegisters();
  std::vector<RegisterValue> RIV;
  for (const InstructionTemplate &IT : Instructions) {
    // Returns the register that this Operand sets or uses, or 0 if this is not
    // a register.
    const auto GetOpReg = [&IT](const Operand &Op) -> unsigned {
      if (Op.isMemory())
        return 0;
      if (Op.isImplicitReg())
        return Op.getImplicitReg();
      if (Op.isExplicit() && IT.getValueFor(Op).isReg())
        return IT.getValueFor(Op).getReg();
      return 0;
    };
    // Collect used registers that have never been def'ed.
    for (const Operand &Op : IT.Instr.Operands) {
      if (Op.isUse()) {
        const unsigned Reg = GetOpReg(Op);
        if (Reg > 0 && !DefinedRegs.test(Reg)) {
          RIV.push_back(RegisterValue::zero(Reg));
          DefinedRegs.set(Reg);
        }
      }
    }
    // Mark defs as having been def'ed.
    for (const Operand &Op : IT.Instr.Operands) {
      if (Op.isDef()) {
        const unsigned Reg = GetOpReg(Op);
        if (Reg > 0)
          DefinedRegs.set(Reg);
      }
    }
  }
  return RIV;
}

llvm::Expected<std::vector<CodeTemplate>>
generateSelfAliasingCodeTemplates(const Instruction &Instr) {
  const AliasingConfigurations SelfAliasing(Instr, Instr);
  if (SelfAliasing.empty())
    return llvm::make_error<SnippetGeneratorFailure>("empty self aliasing");
  std::vector<CodeTemplate> Result;
  Result.emplace_back();
  CodeTemplate &CT = Result.back();
  InstructionTemplate IT(Instr);
  if (SelfAliasing.hasImplicitAliasing()) {
    CT.Info = "implicit Self cycles, picking random values.";
  } else {
    CT.Info = "explicit self cycles, selecting one aliasing Conf.";
    // This is a self aliasing instruction so defs and uses are from the same
    // instance, hence twice IT in the following call.
    setRandomAliasing(SelfAliasing, IT, IT);
  }
  CT.Instructions.push_back(std::move(IT));
  return std::move(Result);
}

llvm::Expected<std::vector<CodeTemplate>>
generateUnconstrainedCodeTemplates(const Instruction &Instr,
                                   llvm::StringRef Msg) {
  std::vector<CodeTemplate> Result;
  Result.emplace_back();
  CodeTemplate &CT = Result.back();
  CT.Info = llvm::formatv("{0}, repeating an unconstrained assignment", Msg);
  CT.Instructions.emplace_back(Instr);
  return std::move(Result);
}

std::mt19937 &randomGenerator() {
  static std::random_device RandomDevice;
  static std::mt19937 RandomGenerator(RandomDevice());
  return RandomGenerator;
}

size_t randomIndex(size_t Max) {
  std::uniform_int_distribution<> Distribution(0, Max);
  return Distribution(randomGenerator());
}

template <typename C>
static auto randomElement(const C &Container) -> decltype(Container[0]) {
  assert(!Container.empty() &&
         "Can't pick a random element from an empty container)");
  return Container[randomIndex(Container.size() - 1)];
}

static void setRegisterOperandValue(const RegisterOperandAssignment &ROV,
                                    InstructionTemplate &IB) {
  assert(ROV.Op);
  if (ROV.Op->isExplicit()) {
    auto &AssignedValue = IB.getValueFor(*ROV.Op);
    if (AssignedValue.isValid()) {
      assert(AssignedValue.isReg() && AssignedValue.getReg() == ROV.Reg);
      return;
    }
    AssignedValue = llvm::MCOperand::createReg(ROV.Reg);
  } else {
    assert(ROV.Op->isImplicitReg());
    assert(ROV.Reg == ROV.Op->getImplicitReg());
  }
}

size_t randomBit(const llvm::BitVector &Vector) {
  assert(Vector.any());
  auto Itr = Vector.set_bits_begin();
  for (size_t I = randomIndex(Vector.count() - 1); I != 0; --I)
    ++Itr;
  return *Itr;
}

void setRandomAliasing(const AliasingConfigurations &AliasingConfigurations,
                       InstructionTemplate &DefIB, InstructionTemplate &UseIB) {
  assert(!AliasingConfigurations.empty());
  assert(!AliasingConfigurations.hasImplicitAliasing());
  const auto &RandomConf = randomElement(AliasingConfigurations.Configurations);
  setRegisterOperandValue(randomElement(RandomConf.Defs), DefIB);
  setRegisterOperandValue(randomElement(RandomConf.Uses), UseIB);
}

void randomizeUnsetVariables(const ExegesisTarget &Target,
                             const llvm::BitVector &ForbiddenRegs,
                             InstructionTemplate &IT) {
  for (const Variable &Var : IT.Instr.Variables) {
    llvm::MCOperand &AssignedValue = IT.getValueFor(Var);
    if (!AssignedValue.isValid())
      Target.randomizeMCOperand(IT.Instr, Var, AssignedValue, ForbiddenRegs);
  }
}

} // namespace exegesis
} // namespace llvm
