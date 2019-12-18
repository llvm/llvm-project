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
#include "Error.h"
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

SnippetGeneratorFailure::SnippetGeneratorFailure(const Twine &S)
    : StringError(S, inconvertibleErrorCode()) {}

SnippetGenerator::SnippetGenerator(const LLVMState &State, const Options &Opts)
    : State(State), Opts(Opts) {}

SnippetGenerator::~SnippetGenerator() = default;

Expected<std::vector<BenchmarkCode>> SnippetGenerator::generateConfigurations(
    const Instruction &Instr, const BitVector &ExtraForbiddenRegs) const {
  BitVector ForbiddenRegs = State.getRATC().reservedRegisters();
  ForbiddenRegs |= ExtraForbiddenRegs;
  // If the instruction has memory registers, prevent the generator from
  // using the scratch register and its aliasing registers.
  if (Instr.hasMemoryOperands()) {
    const auto &ET = State.getExegesisTarget();
    unsigned ScratchSpacePointerInReg =
        ET.getScratchMemoryRegister(State.getTargetMachine().getTargetTriple());
    if (ScratchSpacePointerInReg == 0)
      return make_error<Failure>(
          "Infeasible : target does not support memory instructions");
    const auto &ScratchRegAliases =
        State.getRATC().getRegister(ScratchSpacePointerInReg).aliasedBits();
    // If the instruction implicitly writes to ScratchSpacePointerInReg , abort.
    // FIXME: We could make a copy of the scratch register.
    for (const auto &Op : Instr.Operands) {
      if (Op.isDef() && Op.isImplicitReg() &&
          ScratchRegAliases.test(Op.getImplicitReg()))
        return make_error<Failure>(
            "Infeasible : memory instruction uses scratch memory register");
    }
    ForbiddenRegs |= ScratchRegAliases;
  }

  if (auto E = generateCodeTemplates(Instr, ForbiddenRegs)) {
    std::vector<BenchmarkCode> Output;
    for (CodeTemplate &CT : E.get()) {
      // TODO: Generate as many BenchmarkCode as needed.
      {
        BenchmarkCode BC;
        BC.Info = CT.Info;
        for (InstructionTemplate &IT : CT.Instructions) {
          randomizeUnsetVariables(State.getExegesisTarget(), ForbiddenRegs, IT);
          BC.Key.Instructions.push_back(IT.build());
        }
        if (CT.ScratchSpacePointerInReg)
          BC.LiveIns.push_back(CT.ScratchSpacePointerInReg);
        BC.Key.RegisterInitialValues =
            computeRegisterInitialValues(CT.Instructions);
        BC.Key.Config = CT.Config;
        Output.push_back(std::move(BC));
        if (Output.size() >= Opts.MaxConfigsPerOpcode)
          return Output; // Early exit if we exceeded the number of allowed
                         // configs.
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
  BitVector DefinedRegs = State.getRATC().emptyRegisters();
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
    for (const Operand &Op : IT.getInstr().Operands) {
      if (Op.isUse()) {
        const unsigned Reg = GetOpReg(Op);
        if (Reg > 0 && !DefinedRegs.test(Reg)) {
          RIV.push_back(RegisterValue::zero(Reg));
          DefinedRegs.set(Reg);
        }
      }
    }
    // Mark defs as having been def'ed.
    for (const Operand &Op : IT.getInstr().Operands) {
      if (Op.isDef()) {
        const unsigned Reg = GetOpReg(Op);
        if (Reg > 0)
          DefinedRegs.set(Reg);
      }
    }
  }
  return RIV;
}

Expected<std::vector<CodeTemplate>>
generateSelfAliasingCodeTemplates(const Instruction &Instr) {
  const AliasingConfigurations SelfAliasing(Instr, Instr);
  if (SelfAliasing.empty())
    return make_error<SnippetGeneratorFailure>("empty self aliasing");
  std::vector<CodeTemplate> Result;
  Result.emplace_back();
  CodeTemplate &CT = Result.back();
  InstructionTemplate IT(&Instr);
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

Expected<std::vector<CodeTemplate>>
generateUnconstrainedCodeTemplates(const Instruction &Instr, StringRef Msg) {
  std::vector<CodeTemplate> Result;
  Result.emplace_back();
  CodeTemplate &CT = Result.back();
  CT.Info = formatv("{0}, repeating an unconstrained assignment", Msg);
  CT.Instructions.emplace_back(&Instr);
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
    AssignedValue = MCOperand::createReg(ROV.Reg);
  } else {
    assert(ROV.Op->isImplicitReg());
    assert(ROV.Reg == ROV.Op->getImplicitReg());
  }
}

size_t randomBit(const BitVector &Vector) {
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
                             const BitVector &ForbiddenRegs,
                             InstructionTemplate &IT) {
  for (const Variable &Var : IT.getInstr().Variables) {
    MCOperand &AssignedValue = IT.getValueFor(Var);
    if (!AssignedValue.isValid())
      Target.randomizeMCOperand(IT.getInstr(), Var, AssignedValue,
                                ForbiddenRegs);
  }
}

} // namespace exegesis
} // namespace llvm
