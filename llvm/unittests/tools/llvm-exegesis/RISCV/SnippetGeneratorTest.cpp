//===-- SnippetGeneratorTest.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../Common/AssemblerUtils.h"
#include "LlvmState.h"
#include "MCInstrDescView.h"
#include "ParallelSnippetGenerator.h"
#include "RISCVInstrInfo.h"
#include "RegisterAliasing.h"
#include "SerialSnippetGenerator.h"
#include "TestBase.h"

namespace llvm {
namespace exegesis {
namespace {

using testing::AnyOf;
using testing::ElementsAre;
using testing::HasSubstr;
using testing::SizeIs;

MATCHER(IsInvalid, "") { return !arg.isValid(); }
MATCHER(IsReg, "") { return arg.isReg(); }

template <typename SnippetGeneratorT>
class RISCVSnippetGeneratorTest : public RISCVTestBase {
protected:
  RISCVSnippetGeneratorTest() : Generator(State, SnippetGenerator::Options()) {}

  std::vector<CodeTemplate> checkAndGetCodeTemplates(unsigned Opcode) {
    randomGenerator().seed(0); // Initialize seed.
    const Instruction &Instr = State.getIC().getInstr(Opcode);
    auto CodeTemplateOrError = Generator.generateCodeTemplates(
        &Instr, State.getRATC().emptyRegisters());
    EXPECT_FALSE(CodeTemplateOrError.takeError()); // Valid configuration.
    return std::move(CodeTemplateOrError.get());
  }

  SnippetGeneratorT Generator;
};

using RISCVSerialSnippetGeneratorTest =
    RISCVSnippetGeneratorTest<SerialSnippetGenerator>;

using RISCVParallelSnippetGeneratorTest =
    RISCVSnippetGeneratorTest<ParallelSnippetGenerator>;

TEST_F(RISCVSerialSnippetGeneratorTest,
       ImplicitSelfDependencyThroughExplicitRegs) {
  // - ADD
  // - Op0 Explicit Def RegClass(GPR)
  // - Op1 Explicit Use RegClass(GPR)
  // - Op2 Explicit Use RegClass(GPR)
  // - Var0 [Op0]
  // - Var1 [Op1]
  // - Var2 [Op2]
  // - hasAliasingRegisters
  const unsigned Opcode = RISCV::ADD;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  const auto &CT = CodeTemplates[0];
  EXPECT_THAT(CT.Execution, ExecutionMode::SERIAL_VIA_EXPLICIT_REGS);
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionTemplate &IT = CT.Instructions[0];
  EXPECT_THAT(IT.getOpcode(), Opcode);
  ASSERT_THAT(IT.getVariableValues(), SizeIs(3));
  EXPECT_THAT(IT.getVariableValues(),
              AnyOf(ElementsAre(IsReg(), IsInvalid(), IsReg()),
                    ElementsAre(IsReg(), IsReg(), IsInvalid())))
      << "Op0 is either set to Op1 or to Op2";
}

TEST_F(RISCVSerialSnippetGeneratorTest,
       ImplicitSelfDependencyThroughExplicitRegsForbidAll) {
  // - XOR
  // - Op0 Explicit Def RegClass(GPR)
  // - Op1 Explicit Use RegClass(GPR)
  // - Op2 Explicit Use RegClass(GPR)
  // - Var0 [Op0]
  // - Var1 [Op1]
  // - Var2 [Op2]
  // - hasAliasingRegisters
  randomGenerator().seed(0); // Initialize seed.
  const Instruction &Instr = State.getIC().getInstr(RISCV::XOR);
  auto AllRegisters = State.getRATC().emptyRegisters();
  AllRegisters.flip();
  EXPECT_TRUE(errorToBool(
      Generator.generateCodeTemplates(&Instr, AllRegisters).takeError()));
}

TEST_F(RISCVParallelSnippetGeneratorTest, MemoryUse) {
  // LB reads from memory.
  // - LB
  // - Op0 Explicit Def RegClass(GPR)
  // - Op1 Explicit Use Memory RegClass(GPR)
  // - Op2 Explicit Use Memory
  // - Var0 [Op0]
  // - Var1 [Op1]
  // - Var2 [Op2]
  // - hasMemoryOperands
  const unsigned Opcode = RISCV::LB;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  const auto &CT = CodeTemplates[0];
  EXPECT_THAT(CT.Info, HasSubstr("instruction has no tied variables"));
  EXPECT_THAT(CT.Execution, ExecutionMode::UNKNOWN);
  ASSERT_THAT(CT.Instructions,
              SizeIs(ParallelSnippetGenerator::kMinNumDifferentAddresses));
  const InstructionTemplate &IT = CT.Instructions[0];
  EXPECT_THAT(IT.getOpcode(), Opcode);
  ASSERT_THAT(IT.getVariableValues(), SizeIs(3));
  EXPECT_EQ(IT.getVariableValues()[1].getReg(), RISCV::X10);
}

} // namespace
} // namespace exegesis
} // namespace llvm
