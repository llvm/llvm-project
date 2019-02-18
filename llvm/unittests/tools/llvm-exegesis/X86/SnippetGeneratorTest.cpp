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

namespace llvm {
namespace exegesis {

void InitializeX86ExegesisTarget();

namespace {

using testing::AnyOf;
using testing::ElementsAre;
using testing::Gt;
using testing::HasSubstr;
using testing::Not;
using testing::SizeIs;
using testing::UnorderedElementsAre;

MATCHER(IsInvalid, "") { return !arg.isValid(); }
MATCHER(IsReg, "") { return arg.isReg(); }

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
    InitializeX86ExegesisTarget();
  }

  const LLVMState State;
  const llvm::MCInstrInfo &MCInstrInfo;
  const llvm::MCRegisterInfo &MCRegisterInfo;
};

template <typename SnippetGeneratorT>
class SnippetGeneratorTest : public X86SnippetGeneratorTest {
protected:
  SnippetGeneratorTest() : Generator(State) {}

  std::vector<CodeTemplate> checkAndGetCodeTemplates(unsigned Opcode) {
    randomGenerator().seed(0); // Initialize seed.
    const Instruction &Instr = State.getIC().getInstr(Opcode);
    auto CodeTemplateOrError = Generator.generateCodeTemplates(Instr);
    EXPECT_FALSE(CodeTemplateOrError.takeError()); // Valid configuration.
    return std::move(CodeTemplateOrError.get());
  }

  SnippetGeneratorT Generator;
};

using LatencySnippetGeneratorTest =
    SnippetGeneratorTest<LatencySnippetGenerator>;

using UopsSnippetGeneratorTest = SnippetGeneratorTest<UopsSnippetGenerator>;

TEST_F(LatencySnippetGeneratorTest, ImplicitSelfDependencyThroughImplicitReg) {
  // - ADC16i16
  // - Op0 Explicit Use Immediate
  // - Op1 Implicit Def Reg(AX)
  // - Op2 Implicit Def Reg(EFLAGS)
  // - Op3 Implicit Use Reg(AX)
  // - Op4 Implicit Use Reg(EFLAGS)
  // - Var0 [Op0]
  // - hasAliasingImplicitRegisters (execution is always serial)
  // - hasAliasingRegisters
  const unsigned Opcode = llvm::X86::ADC16i16;
  EXPECT_THAT(MCInstrInfo.get(Opcode).getImplicitDefs()[0], llvm::X86::AX);
  EXPECT_THAT(MCInstrInfo.get(Opcode).getImplicitDefs()[1], llvm::X86::EFLAGS);
  EXPECT_THAT(MCInstrInfo.get(Opcode).getImplicitUses()[0], llvm::X86::AX);
  EXPECT_THAT(MCInstrInfo.get(Opcode).getImplicitUses()[1], llvm::X86::EFLAGS);
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  const auto &CT = CodeTemplates[0];
  EXPECT_THAT(CT.Execution, ExecutionMode::ALWAYS_SERIAL_IMPLICIT_REGS_ALIAS);
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionTemplate &IT = CT.Instructions[0];
  EXPECT_THAT(IT.getOpcode(), Opcode);
  ASSERT_THAT(IT.VariableValues, SizeIs(1)); // Imm.
  EXPECT_THAT(IT.VariableValues[0], IsInvalid()) << "Immediate is not set";
}

TEST_F(LatencySnippetGeneratorTest, ImplicitSelfDependencyThroughTiedRegs) {
  // - ADD16ri
  // - Op0 Explicit Def RegClass(GR16)
  // - Op1 Explicit Use RegClass(GR16) TiedToOp0
  // - Op2 Explicit Use Immediate
  // - Op3 Implicit Def Reg(EFLAGS)
  // - Var0 [Op0,Op1]
  // - Var1 [Op2]
  // - hasTiedRegisters (execution is always serial)
  // - hasAliasingRegisters
  const unsigned Opcode = llvm::X86::ADD16ri;
  EXPECT_THAT(MCInstrInfo.get(Opcode).getImplicitDefs()[0], llvm::X86::EFLAGS);
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  const auto &CT = CodeTemplates[0];
  EXPECT_THAT(CT.Execution, ExecutionMode::ALWAYS_SERIAL_TIED_REGS_ALIAS);
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionTemplate &IT = CT.Instructions[0];
  EXPECT_THAT(IT.getOpcode(), Opcode);
  ASSERT_THAT(IT.VariableValues, SizeIs(2));
  EXPECT_THAT(IT.VariableValues[0], IsInvalid()) << "Operand 1 is not set";
  EXPECT_THAT(IT.VariableValues[1], IsInvalid()) << "Operand 2 is not set";
}

TEST_F(LatencySnippetGeneratorTest, ImplicitSelfDependencyThroughExplicitRegs) {
  // - VXORPSrr
  // - Op0 Explicit Def RegClass(VR128)
  // - Op1 Explicit Use RegClass(VR128)
  // - Op2 Explicit Use RegClass(VR128)
  // - Var0 [Op0]
  // - Var1 [Op1]
  // - Var2 [Op2]
  // - hasAliasingRegisters
  const unsigned Opcode = llvm::X86::VXORPSrr;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  const auto &CT = CodeTemplates[0];
  EXPECT_THAT(CT.Execution, ExecutionMode::SERIAL_VIA_EXPLICIT_REGS);
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionTemplate &IT = CT.Instructions[0];
  EXPECT_THAT(IT.getOpcode(), Opcode);
  ASSERT_THAT(IT.VariableValues, SizeIs(3));
  EXPECT_THAT(IT.VariableValues,
              AnyOf(ElementsAre(IsReg(), IsInvalid(), IsReg()),
                    ElementsAre(IsReg(), IsReg(), IsInvalid())))
      << "Op0 is either set to Op1 or to Op2";
}

TEST_F(LatencySnippetGeneratorTest, DependencyThroughOtherOpcode) {
  // - CMP64rr
  // - Op0 Explicit Use RegClass(GR64)
  // - Op1 Explicit Use RegClass(GR64)
  // - Op2 Implicit Def Reg(EFLAGS)
  // - Var0 [Op0]
  // - Var1 [Op1]
  const unsigned Opcode = llvm::X86::CMP64rr;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(Gt(1U))) << "Many templates are available";
  for (const auto &CT : CodeTemplates) {
    EXPECT_THAT(CT.Execution, ExecutionMode::SERIAL_VIA_NON_MEMORY_INSTR);
    ASSERT_THAT(CT.Instructions, SizeIs(2));
    const InstructionTemplate &IT = CT.Instructions[0];
    EXPECT_THAT(IT.getOpcode(), Opcode);
    ASSERT_THAT(IT.VariableValues, SizeIs(2));
    EXPECT_THAT(IT.VariableValues, AnyOf(ElementsAre(IsReg(), IsInvalid()),
                                         ElementsAre(IsInvalid(), IsReg())));
    EXPECT_THAT(CT.Instructions[1].getOpcode(), Not(Opcode));
    // TODO: check that the two instructions alias each other.
  }
}

TEST_F(LatencySnippetGeneratorTest, LAHF) {
  // - LAHF
  // - Op0 Implicit Def Reg(AH)
  // - Op1 Implicit Use Reg(EFLAGS)
  const unsigned Opcode = llvm::X86::LAHF;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(Gt(1U))) << "Many templates are available";
  for (const auto &CT : CodeTemplates) {
    EXPECT_THAT(CT.Execution, ExecutionMode::SERIAL_VIA_NON_MEMORY_INSTR);
    ASSERT_THAT(CT.Instructions, SizeIs(2));
    const InstructionTemplate &IT = CT.Instructions[0];
    EXPECT_THAT(IT.getOpcode(), Opcode);
    ASSERT_THAT(IT.VariableValues, SizeIs(0));
  }
}

TEST_F(UopsSnippetGeneratorTest, ParallelInstruction) {
  // - BNDCL32rr
  // - Op0 Explicit Use RegClass(BNDR)
  // - Op1 Explicit Use RegClass(GR32)
  // - Var0 [Op0]
  // - Var1 [Op1]
  const unsigned Opcode = llvm::X86::BNDCL32rr;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  const auto &CT = CodeTemplates[0];
  EXPECT_THAT(CT.Info, HasSubstr("parallel"));
  EXPECT_THAT(CT.Execution, ExecutionMode::UNKNOWN);
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionTemplate &IT = CT.Instructions[0];
  EXPECT_THAT(IT.getOpcode(), Opcode);
  ASSERT_THAT(IT.VariableValues, SizeIs(2));
  EXPECT_THAT(IT.VariableValues[0], IsInvalid());
  EXPECT_THAT(IT.VariableValues[1], IsInvalid());
}

TEST_F(UopsSnippetGeneratorTest, SerialInstruction) {
  // - CDQ
  // - Op0 Implicit Def Reg(EAX)
  // - Op1 Implicit Def Reg(EDX)
  // - Op2 Implicit Use Reg(EAX)
  // - hasAliasingImplicitRegisters (execution is always serial)
  // - hasAliasingRegisters
  const unsigned Opcode = llvm::X86::CDQ;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  const auto &CT = CodeTemplates[0];
  EXPECT_THAT(CT.Info, HasSubstr("serial"));
  EXPECT_THAT(CT.Execution, ExecutionMode::UNKNOWN);
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionTemplate &IT = CT.Instructions[0];
  EXPECT_THAT(IT.getOpcode(), Opcode);
  ASSERT_THAT(IT.VariableValues, SizeIs(0));
}

TEST_F(UopsSnippetGeneratorTest, StaticRenaming) {
  // CMOVA32rr has tied variables, we enumerate the possible values to execute
  // as many in parallel as possible.

  // - CMOVA32rr
  // - Op0 Explicit Def RegClass(GR32)
  // - Op1 Explicit Use RegClass(GR32) TiedToOp0
  // - Op2 Explicit Use RegClass(GR32)
  // - Op3 Implicit Use Reg(EFLAGS)
  // - Var0 [Op0,Op1]
  // - Var1 [Op2]
  // - hasTiedRegisters (execution is always serial)
  // - hasAliasingRegisters
  const unsigned Opcode = llvm::X86::CMOVA32rr;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  const auto &CT = CodeTemplates[0];
  EXPECT_THAT(CT.Info, HasSubstr("static renaming"));
  EXPECT_THAT(CT.Execution, ExecutionMode::UNKNOWN);
  constexpr const unsigned kInstructionCount = 15;
  ASSERT_THAT(CT.Instructions, SizeIs(kInstructionCount));
  std::unordered_set<unsigned> AllDefRegisters;
  for (const auto &IT : CT.Instructions) {
    ASSERT_THAT(IT.VariableValues, SizeIs(2));
    AllDefRegisters.insert(IT.VariableValues[0].getReg());
  }
  EXPECT_THAT(AllDefRegisters, SizeIs(kInstructionCount))
      << "Each instruction writes to a different register";
}

TEST_F(UopsSnippetGeneratorTest, NoTiedVariables) {
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
  const unsigned Opcode = llvm::X86::CMOV_GR32;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  const auto &CT = CodeTemplates[0];
  EXPECT_THAT(CT.Info, HasSubstr("no tied variables"));
  EXPECT_THAT(CT.Execution, ExecutionMode::UNKNOWN);
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionTemplate &IT = CT.Instructions[0];
  EXPECT_THAT(IT.getOpcode(), Opcode);
  ASSERT_THAT(IT.VariableValues, SizeIs(4));
  EXPECT_THAT(IT.VariableValues[0].getReg(), Not(IT.VariableValues[1].getReg()))
      << "Def is different from first Use";
  EXPECT_THAT(IT.VariableValues[0].getReg(), Not(IT.VariableValues[2].getReg()))
      << "Def is different from second Use";
  EXPECT_THAT(IT.VariableValues[3], IsInvalid());
}

TEST_F(UopsSnippetGeneratorTest, MemoryUse) {
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
  const unsigned Opcode = llvm::X86::MOV32rm;
  const auto CodeTemplates = checkAndGetCodeTemplates(Opcode);
  ASSERT_THAT(CodeTemplates, SizeIs(1));
  const auto &CT = CodeTemplates[0];
  EXPECT_THAT(CT.Info, HasSubstr("no tied variables"));
  EXPECT_THAT(CT.Execution, ExecutionMode::UNKNOWN);
  ASSERT_THAT(CT.Instructions,
              SizeIs(UopsSnippetGenerator::kMinNumDifferentAddresses));
  const InstructionTemplate &IT = CT.Instructions[0];
  EXPECT_THAT(IT.getOpcode(), Opcode);
  ASSERT_THAT(IT.VariableValues, SizeIs(6));
  EXPECT_EQ(IT.VariableValues[2].getImm(), 1);
  EXPECT_EQ(IT.VariableValues[3].getReg(), 0u);
  EXPECT_EQ(IT.VariableValues[4].getImm(), 0);
  EXPECT_EQ(IT.VariableValues[5].getReg(), 0u);
}

TEST_F(UopsSnippetGeneratorTest, MemoryUse_Movsb) {
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
  const unsigned Opcode = llvm::X86::MOVSB;
  const Instruction &Instr = State.getIC().getInstr(Opcode);
  auto Error = Generator.generateCodeTemplates(Instr).takeError();
  EXPECT_TRUE((bool)Error);
  llvm::consumeError(std::move(Error));
}

class FakeSnippetGenerator : public SnippetGenerator {
public:
  FakeSnippetGenerator(const LLVMState &State) : SnippetGenerator(State) {}

  Instruction createInstruction(unsigned Opcode) {
    return State.getIC().getInstr(Opcode);
  }

private:
  llvm::Expected<std::vector<CodeTemplate>>
  generateCodeTemplates(const Instruction &Instr) const override {
    return llvm::make_error<llvm::StringError>("not implemented",
                                               llvm::inconvertibleErrorCode());
  }
};

using FakeSnippetGeneratorTest = SnippetGeneratorTest<FakeSnippetGenerator>;

testing::Matcher<const RegisterValue &> IsRegisterValue(unsigned Reg,
                                                        llvm::APInt Value) {
  return testing::AllOf(testing::Field(&RegisterValue::Register, Reg),
                        testing::Field(&RegisterValue::Value, Value));
}

TEST_F(FakeSnippetGeneratorTest, ComputeRegisterInitialValuesAdd16ri) {
  // ADD16ri:
  // explicit def 0       : reg RegClass=GR16
  // explicit use 1       : reg RegClass=GR16 | TIED_TO:0
  // explicit use 2       : imm
  // implicit def         : EFLAGS
  InstructionTemplate IT(Generator.createInstruction(llvm::X86::ADD16ri));
  IT.getValueFor(IT.Instr.Variables[0]) =
      llvm::MCOperand::createReg(llvm::X86::AX);
  std::vector<InstructionTemplate> Snippet;
  Snippet.push_back(std::move(IT));
  const auto RIV = Generator.computeRegisterInitialValues(Snippet);
  EXPECT_THAT(RIV, ElementsAre(IsRegisterValue(llvm::X86::AX, llvm::APInt())));
}

TEST_F(FakeSnippetGeneratorTest, ComputeRegisterInitialValuesAdd64rr) {
  // ADD64rr:
  //  mov64ri rax, 42
  //  add64rr rax, rax, rbx
  // -> only rbx needs defining.
  std::vector<InstructionTemplate> Snippet;
  {
    InstructionTemplate Mov(Generator.createInstruction(llvm::X86::MOV64ri));
    Mov.getValueFor(Mov.Instr.Variables[0]) =
        llvm::MCOperand::createReg(llvm::X86::RAX);
    Mov.getValueFor(Mov.Instr.Variables[1]) = llvm::MCOperand::createImm(42);
    Snippet.push_back(std::move(Mov));
  }
  {
    InstructionTemplate Add(Generator.createInstruction(llvm::X86::ADD64rr));
    Add.getValueFor(Add.Instr.Variables[0]) =
        llvm::MCOperand::createReg(llvm::X86::RAX);
    Add.getValueFor(Add.Instr.Variables[1]) =
        llvm::MCOperand::createReg(llvm::X86::RBX);
    Snippet.push_back(std::move(Add));
  }

  const auto RIV = Generator.computeRegisterInitialValues(Snippet);
  EXPECT_THAT(RIV, ElementsAre(IsRegisterValue(llvm::X86::RBX, llvm::APInt())));
}

} // namespace
} // namespace exegesis
} // namespace llvm
