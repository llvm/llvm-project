//===-- SnippetGeneratorTest.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../Common/AssemblerUtils.h"
#include "Latency.h"
#include "LlvmState.h"
#include "MCInstrDescView.h"
#include "RegisterAliasing.h"
#include "MipsInstrInfo.h"

#include <unordered_set>

namespace llvm {
namespace exegesis {

void InitializeMipsExegesisTarget();

namespace {

using testing::AnyOf;
using testing::ElementsAre;
using testing::SizeIs;

MATCHER(IsInvalid, "") { return !arg.isValid(); }
MATCHER(IsReg, "") { return arg.isReg(); }

class MipsSnippetGeneratorTest : public ::testing::Test {
protected:
  MipsSnippetGeneratorTest() : State("mips-unknown-linux", "mips32"),
	InstrInfo(State.getInstrInfo()) {}

  static void SetUpTestCase() {
    LLVMInitializeMipsTargetInfo();
    LLVMInitializeMipsTarget();
    LLVMInitializeMipsTargetMC();
    InitializeMipsExegesisTarget();
  }

  LLVMState State;
  const MCInstrInfo &InstrInfo;
};

template <typename SnippetGeneratorT>
class SnippetGeneratorTest : public MipsSnippetGeneratorTest {
protected:
  SnippetGeneratorTest() : Generator(State, SnippetGenerator::Options()) {}

  std::vector<CodeTemplate> checkAndGetCodeTemplates(unsigned Opcode) {
    randomGenerator().seed(0); // Initialize seed.
    const Instruction &Instr = State.getIC().getInstr(Opcode);
    auto CodeTemplateOrError = Generator.generateCodeTemplates(
        Instr, State.getRATC().emptyRegisters());
    EXPECT_FALSE(CodeTemplateOrError.takeError()); // Valid configuration.
    return std::move(CodeTemplateOrError.get());
  }

  SnippetGeneratorT Generator;
};

using LatencySnippetGeneratorTest =
    SnippetGeneratorTest<LatencySnippetGenerator>;

TEST_F(LatencySnippetGeneratorTest, ImplicitSelfDependencyThroughExplicitRegs) {
  // - ADD
  // - Op0 Explicit Def RegClass(GPR32)
  // - Op1 Explicit Use RegClass(GPR32)
  // - Op2 Explicit Use RegClass(GPR32)
  // - Var0 [Op0]
  // - Var1 [Op1]
  // - Var2 [Op2]
  const unsigned Opcode = Mips::ADD;
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

TEST_F(LatencySnippetGeneratorTest,
 ImplicitSelfDependencyThroughExplicitRegsForbidAll) {
  // - XOR
  // - Op0 Explicit Def RegClass(GPR32)
  // - Op1 Explicit Use RegClass(GPR32)
  // - Op2 Explicit Use RegClass(GPR32)
  // - Var0 [Op0]
  // - Var1 [Op1]
  // - Var2 [Op2]
  randomGenerator().seed(0); // Initialize seed.
  const Instruction &Instr = State.getIC().getInstr(Mips::XOR);
  auto AllRegisters = State.getRATC().emptyRegisters();
  AllRegisters.flip();
  auto Error = Generator.generateCodeTemplates(Instr, AllRegisters).takeError();
  EXPECT_TRUE((bool)Error);
  consumeError(std::move(Error));
}

} // namespace
} // namespace exegesis
} // namespace llvm
