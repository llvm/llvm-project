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
#include "RegisterAliasing.h"
#include "SerialSnippetGenerator.h"
#include "TestBase.h"
#include "X86InstrInfo.h"
#include "llvm/ADT/SetOperations.h"

#include <unordered_set>

namespace llvm {
namespace exegesis {

void InitializeX86ExegesisTarget();

namespace {

using testing::AnyOf;
using testing::ElementsAre;
using testing::Ge;
using testing::Gt;
using testing::HasSubstr;
using testing::IsEmpty;
using testing::Not;
using testing::SizeIs;
using testing::UnorderedElementsAre;

MATCHER(IsInvalid, "") { return !arg.isValid(); }
MATCHER(IsReg, "") { return arg.isReg(); }

template <typename SnippetGeneratorT>
class X86SnippetGeneratorTest : public X86TestBase {
protected:
  X86SnippetGeneratorTest() : Generator(State, SnippetGenerator::Options()),
	                      InstrInfo(State.getInstrInfo()) {}

  std::vector<CodeTemplate> checkAndGetCodeTemplates(unsigned Opcode) {
    randomGenerator().seed(0); // Initialize seed.
    const Instruction &Instr = State.getIC().getInstr(Opcode);
    auto CodeTemplateOrError = Generator.generateCodeTemplates(
        &Instr, State.getRATC().emptyRegisters());
    EXPECT_FALSE(CodeTemplateOrError.takeError()); // Valid configuration.
    return std::move(CodeTemplateOrError.get());
  }

  SnippetGeneratorT Generator;
  const MCInstrInfo &InstrInfo;
};

using X86SerialSnippetGeneratorTest = X86SnippetGeneratorTest<SerialSnippetGenerator>;

using X86ParallelSnippetGeneratorTest =
    X86SnippetGeneratorTest<ParallelSnippetGenerator>;

TEST_F(X86SerialSnippetGeneratorTest, ImplicitSelfDependencyThroughImplicitReg) {
  // - ADC16i16
  // - Op0 Explicit Use Immediate
  // - Op1 Implicit Def Reg(AX)
  // - Op2 Implicit Def Reg(EFLAGS)
  // - Op3 Implicit Use Reg(AX)
  // - Op4 Implicit Use Reg(EFLAGS)
  // - Var0 [Op0]
  // - hasAliasingImplicitRegisters (execution is always serial)
  // - hasAliasingRegisters
  const unsigned Opcode = X86::ADC16i16;
  EXPECT_THAT(InstrInfo.get(Opcode).getImplicitDefs()[0], X86::AX);
  EXPECT_THAT(InstrInfo.get(Opcode).getImplicitDefs()[1], X86::EFLAGS);
  EXPECT_THAT(InstrInfo.get(Opcode).getImplicitUses()[0], X86::AX);
  EXPECT_THAT(InstrInfo.get(Opcode).getImplicitUses()[1], X86::EFLAGS);
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  const auto &CT = CodeTemplates[0];
  EXPECT_THAT(CT.Execution, ExecutionMode::ALWAYS_SERIAL_IMPLICIT_REGS_ALIAS);
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionTemplate &IT = CT.Instructions[0];
  EXPECT_THAT(IT.getOpcode(), Opcode);
  ASSERT_THAT(IT.getVariableValues(), SizeIs(1)); // Imm.
  EXPECT_THAT(IT.getVariableValues()[0], IsInvalid()) << "Immediate is not set";
}

TEST_F(X86SerialSnippetGeneratorTest, ImplicitSelfDependencyThroughTiedRegs) {
  // - ADD16ri
  // - Op0 Explicit Def RegClass(GR16)
  // - Op1 Explicit Use RegClass(GR16) TiedToOp0
  // - Op2 Explicit Use Immediate
  // - Op3 Implicit Def Reg(EFLAGS)
  // - Var0 [Op0,Op1]
  // - Var1 [Op2]
  // - hasTiedRegisters (execution is always serial)
  // - hasAliasingRegisters
  const unsigned Opcode = X86::ADD16ri;
  EXPECT_THAT(InstrInfo.get(Opcode).getImplicitDefs()[0], X86::EFLAGS);
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  const auto &CT = CodeTemplates[0];
  EXPECT_THAT(CT.Execution, ExecutionMode::ALWAYS_SERIAL_TIED_REGS_ALIAS);
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionTemplate &IT = CT.Instructions[0];
  EXPECT_THAT(IT.getOpcode(), Opcode);
  ASSERT_THAT(IT.getVariableValues(), SizeIs(2));
  EXPECT_THAT(IT.getVariableValues()[0], IsInvalid()) << "Operand 1 is not set";
  EXPECT_THAT(IT.getVariableValues()[1], IsInvalid()) << "Operand 2 is not set";
}

TEST_F(X86SerialSnippetGeneratorTest, ImplicitSelfDependencyThroughExplicitRegs) {
  // - VXORPSrr
  // - Op0 Explicit Def RegClass(VR128)
  // - Op1 Explicit Use RegClass(VR128)
  // - Op2 Explicit Use RegClass(VR128)
  // - Var0 [Op0]
  // - Var1 [Op1]
  // - Var2 [Op2]
  // - hasAliasingRegisters
  const unsigned Opcode = X86::VXORPSrr;
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

TEST_F(X86SerialSnippetGeneratorTest,
       ImplicitSelfDependencyThroughExplicitRegsForbidAll) {
  // - VXORPSrr
  // - Op0 Explicit Def RegClass(VR128)
  // - Op1 Explicit Use RegClass(VR128)
  // - Op2 Explicit Use RegClass(VR128)
  // - Var0 [Op0]
  // - Var1 [Op1]
  // - Var2 [Op2]
  // - hasAliasingRegisters
  const unsigned Opcode = X86::VXORPSrr;
  randomGenerator().seed(0); // Initialize seed.
  const Instruction &Instr = State.getIC().getInstr(Opcode);
  auto AllRegisters = State.getRATC().emptyRegisters();
  AllRegisters.flip();
  auto Error =
      Generator.generateCodeTemplates(&Instr, AllRegisters).takeError();
  EXPECT_TRUE((bool)Error);
  consumeError(std::move(Error));
}

TEST_F(X86SerialSnippetGeneratorTest, DependencyThroughOtherOpcode) {
  // - CMP64rr
  // - Op0 Explicit Use RegClass(GR64)
  // - Op1 Explicit Use RegClass(GR64)
  // - Op2 Implicit Def Reg(EFLAGS)
  // - Var0 [Op0]
  // - Var1 [Op1]
  const unsigned Opcode = X86::CMP64rr;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(Gt(1U))) << "Many templates are available";
  for (const auto &CT : CodeTemplates) {
    EXPECT_THAT(CT.Execution, ExecutionMode::SERIAL_VIA_NON_MEMORY_INSTR);
    ASSERT_THAT(CT.Instructions, SizeIs(2));
    const InstructionTemplate &IT = CT.Instructions[0];
    EXPECT_THAT(IT.getOpcode(), Opcode);
    ASSERT_THAT(IT.getVariableValues(), SizeIs(2));
    EXPECT_THAT(IT.getVariableValues(),
                AnyOf(ElementsAre(IsReg(), IsInvalid()),
                      ElementsAre(IsInvalid(), IsReg())));
    EXPECT_THAT(CT.Instructions[1].getOpcode(), Not(Opcode));
    // TODO: check that the two instructions alias each other.
  }
}

TEST_F(X86SerialSnippetGeneratorTest, LAHF) {
  // - LAHF
  // - Op0 Implicit Def Reg(AH)
  // - Op1 Implicit Use Reg(EFLAGS)
  const unsigned Opcode = X86::LAHF;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(Gt(1U))) << "Many templates are available";
  for (const auto &CT : CodeTemplates) {
    EXPECT_THAT(CT.Execution, ExecutionMode::SERIAL_VIA_NON_MEMORY_INSTR);
    ASSERT_THAT(CT.Instructions, SizeIs(2));
    const InstructionTemplate &IT = CT.Instructions[0];
    EXPECT_THAT(IT.getOpcode(), Opcode);
    ASSERT_THAT(IT.getVariableValues(), SizeIs(0));
  }
}

TEST_F(X86SerialSnippetGeneratorTest, VCVTUSI642SDZrrb_Int) {
  // - VCVTUSI642SDZrrb_Int
  // - Op0 Explicit Def RegClass(VR128X)
  // - Op1 Explicit Use RegClass(VR128X)
  // - Op2 Explicit Use STATIC_ROUNDING
  // - Op2 Explicit Use RegClass(GR64)
  // - Op4 Implicit Use Reg(MXSCR)
  const unsigned Opcode = X86::VCVTUSI642SDZrrb_Int;
  const Instruction &Instr = State.getIC().getInstr(Opcode);
  std::vector<BenchmarkCode> Configs;
  auto Error = Generator.generateConfigurations(
      &Instr, Configs, State.getRATC().emptyRegisters());
  ASSERT_FALSE(Error);
  ASSERT_THAT(Configs, SizeIs(1));
  const BenchmarkCode &BC = Configs[0];
  ASSERT_THAT(BC.Key.Instructions, SizeIs(1));
  ASSERT_TRUE(BC.Key.Instructions[0].getOperand(3).isImm());
}

TEST_F(X86ParallelSnippetGeneratorTest, SerialInstruction) {
  // - CDQ
  // - Op0 Implicit Def Reg(EAX)
  // - Op1 Implicit Def Reg(EDX)
  // - Op2 Implicit Use Reg(EAX)
  // - hasAliasingImplicitRegisters (execution is always serial)
  // - hasAliasingRegisters
  const unsigned Opcode = X86::CDQ;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  const auto &CT = CodeTemplates[0];
  EXPECT_THAT(CT.Info, HasSubstr("serial"));
  EXPECT_THAT(CT.Execution, ExecutionMode::UNKNOWN);
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionTemplate &IT = CT.Instructions[0];
  EXPECT_THAT(IT.getOpcode(), Opcode);
  ASSERT_THAT(IT.getVariableValues(), SizeIs(0));
}

TEST_F(X86ParallelSnippetGeneratorTest, ReadAfterWrite_CMOV32rr) {
  // CMOV32rr has tied variables, we enumerate the possible values to execute
  // as many in parallel as possible.

  // - CMOV32rr
  // - Op0 Explicit Def RegClass(GR32)
  // - Op1 Explicit Use RegClass(GR32) TiedToOp0
  // - Op2 Explicit Use RegClass(GR32)
  // - Op3 Explicit Use Immediate
  // - Op3 Implicit Use Reg(EFLAGS)
  // - Var0 [Op0,Op1]
  // - Var1 [Op2]
  // - hasTiedRegisters (execution is always serial)
  // - hasAliasingRegisters
  const unsigned Opcode = X86::CMOV32rr;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(2));
  for (const auto &CT : CodeTemplates) {
    EXPECT_THAT(CT.Info, HasSubstr("avoiding Read-After-Write issue"));
    EXPECT_THAT(CT.Execution, ExecutionMode::UNKNOWN);
    ASSERT_GT(CT.Instructions.size(), 1U);
    std::unordered_set<unsigned> AllDefRegisters;
    std::unordered_set<unsigned> AllUseRegisters;
    for (const auto &IT : CT.Instructions) {
      ASSERT_THAT(IT.getVariableValues(), SizeIs(3));
      AllDefRegisters.insert(IT.getVariableValues()[0].getReg());
      AllUseRegisters.insert(IT.getVariableValues()[1].getReg());
    }
    EXPECT_THAT(AllDefRegisters, SizeIs(CT.Instructions.size()))
        << "Each instruction writes to a different register";
    EXPECT_THAT(AllUseRegisters, Not(IsEmpty()))
        << "In total, some other registers are used";
    auto AllDefAndUseRegs = AllUseRegisters;
    llvm::set_intersect(AllDefAndUseRegs, AllDefRegisters); // A := A ^ B
    EXPECT_THAT(AllDefAndUseRegs, IsEmpty())
        << "No instruction uses any register defined by any of the "
           "instructions";
  }
}

TEST_F(X86ParallelSnippetGeneratorTest, ReadAfterWrite_VFMADD132PDr) {
  // VFMADD132PDr has tied variables, we enumerate the possible values
  // to execute as many in parallel as possible.

  // - VFMADD132PDr
  // - Op0 Explicit Def RegClass(XMM)
  // - Op1 Explicit Use RegClass(XMM) TiedToOp0
  // - Op2 Explicit Use RegClass(XMM)
  // - Op3 Explicit Use RegClass(XMM)
  // - Var0 [Op0,Op1]
  // - Var1 [Op2]
  // - Var2 [Op3]
  // - hasTiedRegisters (execution is always serial)
  // - hasAliasingRegisters
  const unsigned Opcode = X86::VFMADD132PDr;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(3));
  for (const auto &CT : CodeTemplates) {
    EXPECT_THAT(CT.Info, HasSubstr("avoiding Read-After-Write issue"));
    EXPECT_THAT(CT.Execution, ExecutionMode::UNKNOWN);
    ASSERT_GT(CT.Instructions.size(), 1U);
    std::unordered_set<unsigned> AllDefRegisters;
    std::unordered_set<unsigned> AllUseRegisters;
    for (const auto &IT : CT.Instructions) {
      ASSERT_THAT(IT.getVariableValues(), SizeIs(3));
      AllDefRegisters.insert(IT.getVariableValues()[0].getReg());
      AllUseRegisters.insert(IT.getVariableValues()[1].getReg());
      AllUseRegisters.insert(IT.getVariableValues()[2].getReg());
    }
    EXPECT_THAT(AllDefRegisters, SizeIs(CT.Instructions.size()))
        << "Each instruction writes to a different register";
    EXPECT_THAT(AllUseRegisters, Not(IsEmpty()))
        << "In total, some other registers are used";
    auto AllDefAndUseRegs = AllUseRegisters;
    llvm::set_intersect(AllDefAndUseRegs, AllDefRegisters); // A := A ^ B
    EXPECT_THAT(AllDefAndUseRegs, IsEmpty())
        << "No instruction uses any register defined by any of the "
           "instructions";
  }
}

TEST_F(X86ParallelSnippetGeneratorTest, NoTiedVariables) {
  // CMOV_GR32 has no tied variables, we make sure def and use are different
  // from each other.

  // - CMOV_GR32
  // - Op0 Explicit Def RegClass(GR32)
  // - Op1 Explicit Use RegClass(GR32)
  // - Op2 Explicit Use RegClass(GR32)
  // - Op3 Explicit Use Immediate
  // - Op4 Implicit Use Reg(EFLAGS)
  // - Var0 [Op0]
  // - Var1 [Op1]
  // - Var2 [Op2]
  // - Var3 [Op3]
  // - hasAliasingRegisters
  const unsigned Opcode = X86::CMOV_GR32;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(2));
  for (const auto &CT : CodeTemplates) {
    EXPECT_THAT(CT.Info, HasSubstr("no tied variables"));
    EXPECT_THAT(CT.Execution, ExecutionMode::UNKNOWN);
    ASSERT_THAT(CT.Instructions, SizeIs(1));
    const InstructionTemplate &IT = CT.Instructions[0];
    EXPECT_THAT(IT.getOpcode(), Opcode);
    ASSERT_THAT(IT.getVariableValues(), SizeIs(4));
    EXPECT_THAT(IT.getVariableValues()[0].getReg(),
                Not(IT.getVariableValues()[1].getReg()))
        << "Def is different from first Use";
    EXPECT_THAT(IT.getVariableValues()[0].getReg(),
                Not(IT.getVariableValues()[2].getReg()))
        << "Def is different from second Use";
    EXPECT_THAT(IT.getVariableValues()[3], IsInvalid());
  }
}

TEST_F(X86ParallelSnippetGeneratorTest, MemoryUse) {
  // Mov32rm reads from memory.
  // - MOV32rm
  // - Op0 Explicit Def RegClass(GR32)
  // - Op1 Explicit Use Memory RegClass(GR8)
  // - Op2 Explicit Use Memory
  // - Op3 Explicit Use Memory RegClass(GRH8)
  // - Op4 Explicit Use Memory
  // - Op5 Explicit Use Memory RegClass(SEGMENT_REG)
  // - Var0 [Op0]
  // - Var1 [Op1]
  // - Var2 [Op2]
  // - Var3 [Op3]
  // - Var4 [Op4]
  // - Var5 [Op5]
  // - hasMemoryOperands
  // - hasAliasingRegisters
  const unsigned Opcode = X86::MOV32rm;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  for (const auto &CT : CodeTemplates) {
    EXPECT_THAT(CT.Info, HasSubstr("no tied variables"));
    EXPECT_THAT(CT.Execution, ExecutionMode::UNKNOWN);
    ASSERT_THAT(CT.Instructions,
                SizeIs(ParallelSnippetGenerator::kMinNumDifferentAddresses));
    const InstructionTemplate &IT = CT.Instructions[0];
    EXPECT_THAT(IT.getOpcode(), Opcode);
    ASSERT_THAT(IT.getVariableValues(), SizeIs(6));
    EXPECT_EQ(IT.getVariableValues()[2].getImm(), 1);
    EXPECT_EQ(IT.getVariableValues()[3].getReg(), 0u);
    EXPECT_EQ(IT.getVariableValues()[4].getImm(), 0);
    EXPECT_EQ(IT.getVariableValues()[5].getReg(), 0u);
  }
}

TEST_F(X86ParallelSnippetGeneratorTest, MOV16ms) {
  const unsigned Opcode = X86::MOV16ms;
  const Instruction &Instr = State.getIC().getInstr(Opcode);
  std::vector<BenchmarkCode> Benchmarks;
  auto Err = Generator.generateConfigurations(&Instr, Benchmarks,
                                              State.getRATC().emptyRegisters());
  EXPECT_TRUE((bool)Err);
  EXPECT_THAT(toString(std::move(Err)),
              testing::HasSubstr("no available registers"));
}

TEST_F(X86ParallelSnippetGeneratorTest,
       AvoidSerializingThroughImplicitRegisters) {
  // MULX32rr implicitly uses EDX. We should not select that register to avoid
  // serialization.
  const unsigned Opcode = X86::MULX32rr;
  randomGenerator().seed(0); // Initialize seed.
  const Instruction &Instr = State.getIC().getInstr(Opcode);
  // Forbid all registers but RDX/EDX/DX/DH/DL. The only option would be to
  // choose that register, but that would serialize the instruction, so we
  // should be returning an error.
  auto AllRegisters = State.getRATC().emptyRegisters();
  AllRegisters.flip();
  AllRegisters.reset(X86::RDX);
  AllRegisters.reset(X86::EDX);
  AllRegisters.reset(X86::DX);
  AllRegisters.reset(X86::DH);
  AllRegisters.reset(X86::DL);
  auto Err = Generator.generateCodeTemplates(&Instr, AllRegisters);
  EXPECT_FALSE((bool)Err);
  EXPECT_THAT(toString(Err.takeError()),
              testing::HasSubstr("Failed to produce any snippet"));
}

class X86FakeSnippetGenerator : public SnippetGenerator {
public:
  X86FakeSnippetGenerator(const LLVMState &State, const Options &Opts)
      : SnippetGenerator(State, Opts) {}

  const Instruction &getInstr(unsigned Opcode) {
    return State.getIC().getInstr(Opcode);
  }

  InstructionTemplate getInstructionTemplate(unsigned Opcode) {
    return {&getInstr(Opcode)};
  }

private:
  Expected<std::vector<CodeTemplate>>
  generateCodeTemplates(InstructionTemplate, const BitVector &) const override {
    return make_error<StringError>("not implemented", inconvertibleErrorCode());
  }
};

using X86FakeSnippetGeneratorTest = X86SnippetGeneratorTest<X86FakeSnippetGenerator>;

testing::Matcher<const RegisterValue &> IsRegisterValue(unsigned Reg,
                                                        APInt Value) {
  return testing::AllOf(testing::Field(&RegisterValue::Register, Reg),
                        testing::Field(&RegisterValue::Value, Value));
}

TEST_F(X86FakeSnippetGeneratorTest, MemoryUse_Movsb) {
  // MOVSB writes to scratch memory register.
  // - MOVSB
  // - Op0 Explicit Use Memory RegClass(GR8)
  // - Op1 Explicit Use Memory RegClass(GR8)
  // - Op2 Explicit Use Memory RegClass(SEGMENT_REG)
  // - Op3 Implicit Def Reg(EDI)
  // - Op4 Implicit Def Reg(ESI)
  // - Op5 Implicit Use Reg(EDI)
  // - Op6 Implicit Use Reg(ESI)
  // - Op7 Implicit Use Reg(DF)
  // - Var0 [Op0]
  // - Var1 [Op1]
  // - Var2 [Op2]
  // - hasMemoryOperands
  // - hasAliasingImplicitRegisters (execution is always serial)
  // - hasAliasingRegisters
  const unsigned Opcode = X86::MOVSB;
  const Instruction &Instr = State.getIC().getInstr(Opcode);
  std::vector<BenchmarkCode> Benchmarks;
  auto Error = Generator.generateConfigurations(
      &Instr, Benchmarks, State.getRATC().emptyRegisters());
  EXPECT_TRUE((bool)Error);
  consumeError(std::move(Error));
}

TEST_F(X86FakeSnippetGeneratorTest, ComputeRegisterInitialValuesAdd16ri) {
  // ADD16ri:
  // explicit def 0       : reg RegClass=GR16
  // explicit use 1       : reg RegClass=GR16 | TIED_TO:0
  // explicit use 2       : imm
  // implicit def         : EFLAGS
  InstructionTemplate IT = Generator.getInstructionTemplate(X86::ADD16ri);
  IT.getValueFor(IT.getInstr().Variables[0]) = MCOperand::createReg(X86::AX);
  std::vector<InstructionTemplate> Snippet;
  Snippet.push_back(std::move(IT));
  const auto RIV = Generator.computeRegisterInitialValues(Snippet);
  EXPECT_THAT(RIV, ElementsAre(IsRegisterValue(X86::AX, APInt())));
}

TEST_F(X86FakeSnippetGeneratorTest, ComputeRegisterInitialValuesAdd64rr) {
  // ADD64rr:
  //  mov64ri rax, 42
  //  add64rr rax, rax, rbx
  // -> only rbx needs defining.
  std::vector<InstructionTemplate> Snippet;
  {
    InstructionTemplate Mov = Generator.getInstructionTemplate(X86::MOV64ri);
    Mov.getValueFor(Mov.getInstr().Variables[0]) =
        MCOperand::createReg(X86::RAX);
    Mov.getValueFor(Mov.getInstr().Variables[1]) = MCOperand::createImm(42);
    Snippet.push_back(std::move(Mov));
  }
  {
    InstructionTemplate Add = Generator.getInstructionTemplate(X86::ADD64rr);
    Add.getValueFor(Add.getInstr().Variables[0]) =
        MCOperand::createReg(X86::RAX);
    Add.getValueFor(Add.getInstr().Variables[1]) =
        MCOperand::createReg(X86::RBX);
    Snippet.push_back(std::move(Add));
  }

  const auto RIV = Generator.computeRegisterInitialValues(Snippet);
  EXPECT_THAT(RIV, ElementsAre(IsRegisterValue(X86::RBX, APInt())));
}

} // namespace
} // namespace exegesis
} // namespace llvm
