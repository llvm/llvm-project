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
#include "PPCInstrInfo.h"
#include "ParallelSnippetGenerator.h"
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
class PPCSnippetGeneratorTest : public PPCTestBase {
protected:
  PPCSnippetGeneratorTest() : Generator(State, SnippetGenerator::Options()) {}

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

using PPCSerialSnippetGeneratorTest = PPCSnippetGeneratorTest<SerialSnippetGenerator>;

using PPCParallelSnippetGeneratorTest =
    PPCSnippetGeneratorTest<ParallelSnippetGenerator>;

TEST_F(PPCSerialSnippetGeneratorTest, ImplicitSelfDependencyThroughExplicitRegs) {
  // - ADD8
  // - Op0 Explicit Def RegClass(G8RC)
  // - Op1 Explicit Use RegClass(G8RC)
  // - Op2 Explicit Use RegClass(G8RC)
  // - Var0 [Op0]
  // - Var1 [Op1]
  // - Var2 [Op2]
  // - hasAliasingRegisters
  const unsigned Opcode = PPC::ADD8;
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

TEST_F(PPCSerialSnippetGeneratorTest, ImplicitSelfDependencyThroughTiedRegs) {

  // - RLDIMI
  // - Op0 Explicit Def RegClass(G8RC)
  // - Op1 Explicit Use RegClass(G8RC) TiedToOp0
  // - Op2 Explicit Use RegClass(G8RC)
  // - Op3 Explicit Use Immediate
  // - Op4 Explicit Use Immediate
  // - Var0 [Op0,Op1]
  // - Var1 [Op2]
  // - Var2 [Op3]
  // - Var3 [Op4]
  // - hasTiedRegisters (execution is always serial)
  // - hasAliasingRegisters
  // - RLDIMI
  const unsigned Opcode = PPC::RLDIMI;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  const auto &CT = CodeTemplates[0];
  EXPECT_THAT(CT.Execution, ExecutionMode::ALWAYS_SERIAL_TIED_REGS_ALIAS);
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionTemplate &IT = CT.Instructions[0];
  EXPECT_THAT(IT.getOpcode(), Opcode);
  ASSERT_THAT(IT.getVariableValues(), SizeIs(4));
  EXPECT_THAT(IT.getVariableValues()[2], IsInvalid()) << "Operand 1 is not set";
  EXPECT_THAT(IT.getVariableValues()[3], IsInvalid()) << "Operand 2 is not set";
}

TEST_F(PPCParallelSnippetGeneratorTest, MemoryUse) {
  // - LDX
  // - Op0 Explicit Def RegClass(G8RC)
  // - Op1 Explicit Use Memory RegClass(GPRC)
  // - Op2 Explicit Use Memory RegClass(VSSRC)
  // - Var0 [Op0]
  // - Var1 [Op1]
  // - Var2 [Op2]
  // - hasMemoryOperands
  // - hasAliasingRegisters
  const unsigned Opcode = PPC::LDX;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  for (const auto &CT : CodeTemplates) {
    EXPECT_THAT(CT.Info, HasSubstr("instruction has no tied variables"));
    EXPECT_THAT(CT.Execution, ExecutionMode::UNKNOWN);
    ASSERT_THAT(CT.Instructions,
                SizeIs(ParallelSnippetGenerator::kMinNumDifferentAddresses));
    const InstructionTemplate &IT = CT.Instructions[0];
    EXPECT_THAT(IT.getOpcode(), Opcode);
    ASSERT_THAT(IT.getVariableValues(), SizeIs(3));
    EXPECT_EQ(IT.getVariableValues()[1].getReg(), PPC::X1);
    EXPECT_EQ(IT.getVariableValues()[2].getReg(), PPC::X13);
  }
}

} // namespace
} // namespace exegesis
} // namespace llvm
