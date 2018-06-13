//===-- SnippetGeneratorTest.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../Common/AssemblerUtils.h"
#include "Latency.h"
#include "LlvmState.h"
#include "MCInstrDescView.h"
#include "RegisterAliasing.h"
#include "Uops.h"
#include "X86InstrInfo.h"

#include <unordered_set>

namespace exegesis {
namespace {

class X86SnippetGeneratorTest : public ::testing::Test {
protected:
  X86SnippetGeneratorTest()
      : State("x86_64-unknown-linux", "haswell"),
        MCInstrInfo(State.getInstrInfo()), MCRegisterInfo(State.getRegInfo()) {}

  static void SetUpTestCase() {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86Target();
    LLVMInitializeX86AsmPrinter();
  }

  const LLVMState State;
  const llvm::MCInstrInfo &MCInstrInfo;
  const llvm::MCRegisterInfo &MCRegisterInfo;
};

class LatencySnippetGeneratorTest : public X86SnippetGeneratorTest {
protected:
  LatencySnippetGeneratorTest() : Runner(State) {}

  BenchmarkConfiguration checkAndGetConfiguration(unsigned Opcode) {
    randomGenerator().seed(0); // Initialize seed.
    auto ConfOrError = Runner.generateConfiguration(Opcode);
    EXPECT_FALSE(ConfOrError.takeError()); // Valid configuration.
    return ConfOrError.get();
  }

  LatencyBenchmarkRunner Runner;
};

TEST_F(LatencySnippetGeneratorTest, ImplicitSelfDependency) {
  // ADC16i16 self alias because of implicit use and def.

  // explicit use 0       : imm
  // implicit def         : AX
  // implicit def         : EFLAGS
  // implicit use         : AX
  // implicit use         : EFLAGS
  const unsigned Opcode = llvm::X86::ADC16i16;
  auto Conf = checkAndGetConfiguration(Opcode);
  EXPECT_THAT(Conf.Info, testing::HasSubstr("implicit"));
  ASSERT_THAT(Conf.Snippet, testing::SizeIs(1));
  const llvm::MCInst Instr = Conf.Snippet[0];
  EXPECT_THAT(Instr.getOpcode(), Opcode);
  EXPECT_THAT(Instr.getNumOperands(), 1);
  EXPECT_TRUE(Instr.getOperand(0).isImm()); // Use
  EXPECT_THAT(MCInstrInfo.get(Opcode).getImplicitDefs()[0], llvm::X86::AX);
  EXPECT_THAT(MCInstrInfo.get(Opcode).getImplicitDefs()[1], llvm::X86::EFLAGS);
  EXPECT_THAT(MCInstrInfo.get(Opcode).getImplicitUses()[0], llvm::X86::AX);
  EXPECT_THAT(MCInstrInfo.get(Opcode).getImplicitUses()[1], llvm::X86::EFLAGS);
}

TEST_F(LatencySnippetGeneratorTest, ExplicitSelfDependency) {
  // ADD16ri self alias because Op0 and Op1 are tied together.

  // explicit def 0       : reg RegClass=GR16
  // explicit use 1       : reg RegClass=GR16 | TIED_TO:0
  // explicit use 2       : imm
  // implicit def         : EFLAGS
  const unsigned Opcode = llvm::X86::ADD16ri;
  auto Conf = checkAndGetConfiguration(Opcode);
  EXPECT_THAT(Conf.Info, testing::HasSubstr("explicit"));
  ASSERT_THAT(Conf.Snippet, testing::SizeIs(1));
  const llvm::MCInst Instr = Conf.Snippet[0];
  EXPECT_THAT(Instr.getOpcode(), Opcode);
  EXPECT_THAT(Instr.getNumOperands(), 3);
  EXPECT_TRUE(Instr.getOperand(0).isReg());
  EXPECT_TRUE(Instr.getOperand(1).isReg());
  EXPECT_THAT(Instr.getOperand(0).getReg(), Instr.getOperand(1).getReg())
      << "Op0 and Op1 should have the same value";
  EXPECT_TRUE(Instr.getOperand(2).isImm());
  EXPECT_THAT(MCInstrInfo.get(Opcode).getImplicitDefs()[0], llvm::X86::EFLAGS);
}

TEST_F(LatencySnippetGeneratorTest, DependencyThroughOtherOpcode) {
  // CMP64rr
  // explicit use 0       : reg RegClass=GR64
  // explicit use 1       : reg RegClass=GR64
  // implicit def         : EFLAGS

  const unsigned Opcode = llvm::X86::CMP64rr;
  auto Conf = checkAndGetConfiguration(Opcode);
  EXPECT_THAT(Conf.Info, testing::HasSubstr("cycle through"));
  ASSERT_THAT(Conf.Snippet, testing::SizeIs(2));
  const llvm::MCInst Instr = Conf.Snippet[0];
  EXPECT_THAT(Instr.getOpcode(), Opcode);
  // TODO: check that the two instructions alias each other.
}

TEST_F(LatencySnippetGeneratorTest, LAHF) {
  const unsigned Opcode = llvm::X86::LAHF;
  auto Conf = checkAndGetConfiguration(Opcode);
  EXPECT_THAT(Conf.Info, testing::HasSubstr("cycle through"));
  ASSERT_THAT(Conf.Snippet, testing::SizeIs(2));
  const llvm::MCInst Instr = Conf.Snippet[0];
  EXPECT_THAT(Instr.getOpcode(), Opcode);
}

class UopsSnippetGeneratorTest : public X86SnippetGeneratorTest {
protected:
  UopsSnippetGeneratorTest() : Runner(State) {}

  BenchmarkConfiguration checkAndGetConfiguration(unsigned Opcode) {
    randomGenerator().seed(0); // Initialize seed.
    auto ConfOrError = Runner.generateConfiguration(Opcode);
    EXPECT_FALSE(ConfOrError.takeError()); // Valid configuration.
    return ConfOrError.get();
  }

  UopsBenchmarkRunner Runner;
};

TEST_F(UopsSnippetGeneratorTest, ParallelInstruction) {
  // BNDCL32rr is parallelno matter what.

  // explicit use 0       : reg RegClass=BNDR
  // explicit use 1       : reg RegClass=GR32

  const unsigned Opcode = llvm::X86::BNDCL32rr;
  auto Conf = checkAndGetConfiguration(Opcode);
  EXPECT_THAT(Conf.Info, testing::HasSubstr("parallel"));
  ASSERT_THAT(Conf.Snippet, testing::SizeIs(1));
  const llvm::MCInst Instr = Conf.Snippet[0];
  EXPECT_THAT(Instr.getOpcode(), Opcode);
}

TEST_F(UopsSnippetGeneratorTest, SerialInstruction) {
  // CDQ is serial no matter what.

  // implicit def         : EAX
  // implicit def         : EDX
  // implicit use         : EAX
  const unsigned Opcode = llvm::X86::CDQ;
  auto Conf = checkAndGetConfiguration(Opcode);
  EXPECT_THAT(Conf.Info, testing::HasSubstr("serial"));
  ASSERT_THAT(Conf.Snippet, testing::SizeIs(1));
  const llvm::MCInst Instr = Conf.Snippet[0];
  EXPECT_THAT(Instr.getOpcode(), Opcode);
}

TEST_F(UopsSnippetGeneratorTest, StaticRenaming) {
  // CMOVA32rr has tied variables, we enumarate the possible values to execute
  // as many in parallel as possible.

  // explicit def 0       : reg RegClass=GR32
  // explicit use 1       : reg RegClass=GR32 | TIED_TO:0
  // explicit use 2       : reg RegClass=GR32
  // implicit use         : EFLAGS
  const unsigned Opcode = llvm::X86::CMOVA32rr;
  auto Conf = checkAndGetConfiguration(Opcode);
  EXPECT_THAT(Conf.Info, testing::HasSubstr("static renaming"));
  constexpr const unsigned kInstructionCount = 15;
  ASSERT_THAT(Conf.Snippet, testing::SizeIs(kInstructionCount));
  std::unordered_set<unsigned> AllDefRegisters;
  for (const auto &Inst : Conf.Snippet)
    AllDefRegisters.insert(Inst.getOperand(0).getReg());
  EXPECT_THAT(AllDefRegisters, testing::SizeIs(kInstructionCount))
      << "Each instruction writes to a different register";
}

TEST_F(UopsSnippetGeneratorTest, NoTiedVariables) {
  // CMOV_GR32 has no tied variables, we make sure def and use are different
  // from each other.

  // explicit def 0       : reg RegClass=GR32
  // explicit use 1       : reg RegClass=GR32
  // explicit use 2       : reg RegClass=GR32
  // explicit use 3       : imm
  // implicit use         : EFLAGS
  const unsigned Opcode = llvm::X86::CMOV_GR32;
  auto Conf = checkAndGetConfiguration(Opcode);
  EXPECT_THAT(Conf.Info, testing::HasSubstr("no tied variables"));
  ASSERT_THAT(Conf.Snippet, testing::SizeIs(1));
  const llvm::MCInst Instr = Conf.Snippet[0];
  EXPECT_THAT(Instr.getOpcode(), Opcode);
  EXPECT_THAT(Instr.getNumOperands(), 4);
  EXPECT_THAT(Instr.getOperand(0).getReg(),
              testing::Not(Instr.getOperand(1).getReg()))
      << "Def is different from first Use";
  EXPECT_THAT(Instr.getOperand(0).getReg(),
              testing::Not(Instr.getOperand(2).getReg()))
      << "Def is different from second Use";
  EXPECT_THAT(Instr.getOperand(3).getImm(), 1);
}

} // namespace
} // namespace exegesis
