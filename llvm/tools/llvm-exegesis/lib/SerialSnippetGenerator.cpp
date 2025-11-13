//===-- SerialSnippetGenerator.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SerialSnippetGenerator.h"

#include "CodeTemplate.h"
#include "MCInstrDescView.h"
#include "Target.h"
#include <algorithm>
#include <numeric>
#include <vector>

namespace llvm {
namespace exegesis {

struct ExecutionClass {
  ExecutionMode Mask;
  const char *Description;
} static const kExecutionClasses[] = {
    {ExecutionMode::ALWAYS_SERIAL_IMPLICIT_REGS_ALIAS |
         ExecutionMode::ALWAYS_SERIAL_TIED_REGS_ALIAS,
     "Repeating a single implicitly serial instruction"},
    {ExecutionMode::SERIAL_VIA_EXPLICIT_REGS,
     "Repeating a single explicitly serial instruction"},
    {ExecutionMode::SERIAL_VIA_MEMORY_INSTR |
         ExecutionMode::SERIAL_VIA_NON_MEMORY_INSTR,
     "Repeating two instructions"},
};

static constexpr size_t kMaxAliasingInstructions = 10;

static std::vector<const Instruction *>
computeAliasingInstructions(const LLVMState &State, const Instruction *Instr,
                            size_t MaxAliasingInstructions,
                            const BitVector &ForbiddenRegisters) {
  const auto &ET = State.getExegesisTarget();
  const auto AvailableFeatures = State.getSubtargetInfo().getFeatureBits();
  // Randomly iterate the set of instructions.
  std::vector<unsigned> Opcodes;
  Opcodes.resize(State.getInstrInfo().getNumOpcodes());
  std::iota(Opcodes.begin(), Opcodes.end(), 0U);
  llvm::shuffle(Opcodes.begin(), Opcodes.end(), randomGenerator());

  std::vector<const Instruction *> AliasingInstructions;
  for (const unsigned OtherOpcode : Opcodes) {
    if (!ET.isOpcodeAvailable(OtherOpcode, AvailableFeatures))
      continue;
    if (OtherOpcode == Instr->Description.getOpcode())
      continue;
    const Instruction &OtherInstr = State.getIC().getInstr(OtherOpcode);
    if (ET.getIgnoredOpcodeReasonOrNull(State, OtherInstr.getOpcode()))
      continue;
    if (OtherInstr.hasMemoryOperands())
      continue;
    // Filtering out loads/stores might belong in hasMemoryOperands(), but that
    // complicates things as there are instructions with may load/store that
    // don't have operands (e.g. X86's CLUI instruction). So, it's easier to
    // filter them out here.
    if (OtherInstr.Description.mayLoad() || OtherInstr.Description.mayStore())
      continue;
    if (!ET.allowAsBackToBack(OtherInstr))
      continue;
    if (Instr->hasAliasingRegistersThrough(OtherInstr, ForbiddenRegisters))
      AliasingInstructions.push_back(&OtherInstr);
    if (AliasingInstructions.size() >= MaxAliasingInstructions)
      break;
  }
  return AliasingInstructions;
}

static ExecutionMode getExecutionModes(const Instruction &Instr,
                                       const BitVector &ForbiddenRegisters) {
  ExecutionMode EM = ExecutionMode::UNKNOWN;
  if (Instr.hasAliasingImplicitRegisters())
    EM |= ExecutionMode::ALWAYS_SERIAL_IMPLICIT_REGS_ALIAS;
  if (Instr.hasTiedRegisters())
    EM |= ExecutionMode::ALWAYS_SERIAL_TIED_REGS_ALIAS;
  if (Instr.hasMemoryOperands())
    EM |= ExecutionMode::SERIAL_VIA_MEMORY_INSTR;
  if (Instr.hasAliasingNotMemoryRegisters(ForbiddenRegisters))
    EM |= ExecutionMode::SERIAL_VIA_EXPLICIT_REGS;
  if (Instr.hasOneUseOrOneDef())
    EM |= ExecutionMode::SERIAL_VIA_NON_MEMORY_INSTR;
  return EM;
}

static void appendCodeTemplates(const LLVMState &State,
                                InstructionTemplate Variant,
                                const BitVector &ForbiddenRegisters,
                                ExecutionMode ExecutionModeBit,
                                StringRef ExecutionClassDescription,
                                std::vector<CodeTemplate> &CodeTemplates) {
  assert(isEnumValue(ExecutionModeBit) && "Bit must be a power of two");
  switch (ExecutionModeBit) {
  case ExecutionMode::ALWAYS_SERIAL_IMPLICIT_REGS_ALIAS:
    // Nothing to do, the instruction is always serial.
    [[fallthrough]];
  case ExecutionMode::ALWAYS_SERIAL_TIED_REGS_ALIAS: {
    // Picking whatever value for the tied variable will make the instruction
    // serial.
    CodeTemplate CT;
    CT.Execution = ExecutionModeBit;
    CT.Info = std::string(ExecutionClassDescription);
    CT.Instructions.push_back(std::move(Variant));
    CodeTemplates.push_back(std::move(CT));
    return;
  }
  case ExecutionMode::SERIAL_VIA_MEMORY_INSTR: {
    // Select back-to-back memory instruction.

    auto &I = Variant.getInstr();
    if (I.Description.mayLoad()) {
      // If instruction is load, we can self-alias it in case when instruction
      // overrides whole address register. For that we use provided scratch
      // memory.

      // TODO: now it is not checked if load writes the whole register.

      auto DefOpIt = find_if(I.Operands, [](Operand const &Op) {
        return Op.isDef() && Op.isReg();
      });

      if (DefOpIt == I.Operands.end())
        return;

      const Operand &DefOp = *DefOpIt;
      const ExegesisTarget &ET = State.getExegesisTarget();
      unsigned ScratchMemoryRegister = ET.getScratchMemoryRegister(
          State.getTargetMachine().getTargetTriple());
      const llvm::MCRegisterClass &RegClass =
          State.getTargetMachine().getMCRegisterInfo()->getRegClass(
              DefOp.getExplicitOperandInfo().RegClass);

      // Register classes of def operand and memory operand must be the same
      // to perform aliasing.
      if (!RegClass.contains(ScratchMemoryRegister))
        return;

      ET.fillMemoryOperands(Variant, ScratchMemoryRegister, 0);
      Variant.getValueFor(DefOp) = MCOperand::createReg(ScratchMemoryRegister);

      CodeTemplate CT;
      CT.Execution = ExecutionModeBit;
      CT.ScratchSpacePointerInReg = ScratchMemoryRegister;

      CT.Info = std::string(ExecutionClassDescription);
      CT.Instructions.push_back(std::move(Variant));
      CodeTemplates.push_back(std::move(CT));
    }

    // TODO: implement more cases
    return;
  }
  case ExecutionMode::SERIAL_VIA_EXPLICIT_REGS: {
    // Making the execution of this instruction serial by selecting one def
    // register to alias with one use register.
    const AliasingConfigurations SelfAliasing(
        Variant.getInstr(), Variant.getInstr(), ForbiddenRegisters);
    assert(!SelfAliasing.empty() && !SelfAliasing.hasImplicitAliasing() &&
           "Instr must alias itself explicitly");
    // This is a self aliasing instruction so defs and uses are from the same
    // instance, hence twice Variant in the following call.
    setRandomAliasing(SelfAliasing, Variant, Variant);
    CodeTemplate CT;
    CT.Execution = ExecutionModeBit;
    CT.Info = std::string(ExecutionClassDescription);
    CT.Instructions.push_back(std::move(Variant));
    CodeTemplates.push_back(std::move(CT));
    return;
  }
  case ExecutionMode::SERIAL_VIA_NON_MEMORY_INSTR: {
    const Instruction &Instr = Variant.getInstr();
    // Select back-to-back non-memory instruction.
    for (const auto *OtherInstr : computeAliasingInstructions(
             State, &Instr, kMaxAliasingInstructions, ForbiddenRegisters)) {
      const AliasingConfigurations Forward(Instr, *OtherInstr,
                                           ForbiddenRegisters);
      const AliasingConfigurations Back(*OtherInstr, Instr, ForbiddenRegisters);
      InstructionTemplate ThisIT(Variant);
      InstructionTemplate OtherIT(OtherInstr);
      if (!Forward.hasImplicitAliasing())
        setRandomAliasing(Forward, ThisIT, OtherIT);
      else if (!Back.hasImplicitAliasing())
        setRandomAliasing(Back, OtherIT, ThisIT);
      CodeTemplate CT;
      CT.Execution = ExecutionModeBit;
      CT.Info = std::string(ExecutionClassDescription);
      CT.Instructions.push_back(std::move(ThisIT));
      CT.Instructions.push_back(std::move(OtherIT));
      CodeTemplates.push_back(std::move(CT));
    }
    return;
  }
  default:
    llvm_unreachable("Unhandled enum value");
  }
}

SerialSnippetGenerator::~SerialSnippetGenerator() = default;

Expected<std::vector<CodeTemplate>>
SerialSnippetGenerator::generateCodeTemplates(
    InstructionTemplate Variant, const BitVector &ForbiddenRegisters) const {
  std::vector<CodeTemplate> Results;
  const ExecutionMode EM =
      getExecutionModes(Variant.getInstr(), ForbiddenRegisters);
  for (const auto EC : kExecutionClasses) {
    for (const auto ExecutionModeBit : getExecutionModeBits(EM & EC.Mask))
      appendCodeTemplates(State, Variant, ForbiddenRegisters, ExecutionModeBit,
                          EC.Description, Results);
    if (!Results.empty())
      break;
  }
  if (Results.empty())
    return make_error<Failure>(
        "No strategy found to make the execution serial");
  return std::move(Results);
}

} // namespace exegesis
} // namespace llvm
