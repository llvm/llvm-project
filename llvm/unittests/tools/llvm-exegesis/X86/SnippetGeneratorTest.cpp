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

void InitializeX86ExegesisTarget();

namespace {

using testing::AnyOf;
using testing::ElementsAre;
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

  CodeTemplate checkAndGetCodeTemplate(unsigned Opcode) {
    randomGenerator().seed(0); // Initialize seed.
    auto CodeTemplateOrError = Generator.generateCodeTemplate(Opcode);
    EXPECT_FALSE(CodeTemplateOrError.takeError()); // Valid configuration.
    return std::move(CodeTemplateOrError.get());
  }

  SnippetGeneratorT Generator;
};

using LatencySnippetGeneratorTest =
    SnippetGeneratorTest<LatencySnippetGenerator>;

using UopsSnippetGeneratorTest = SnippetGeneratorTest<UopsSnippetGenerator>;

TEST_F(LatencySnippetGeneratorTest, ImplicitSelfDependency) {
  // ADC16i16 self alias because of implicit use and def.

  // explicit use 0       : imm
  // implicit def         : AX
  // implicit def         : EFLAGS
  // implicit use         : AX
  // implicit use         : EFLAGS
  const unsigned Opcode = llvm::X86::ADC16i16;
  EXPECT_THAT(MCInstrInfo.get(Opcode).getImplicitDefs()[0], llvm::X86::AX);
  EXPECT_THAT(MCInstrInfo.get(Opcode).getImplicitDefs()[1], llvm::X86::EFLAGS);
  EXPECT_THAT(MCInstrInfo.get(Opcode).getImplicitUses()[0], llvm::X86::AX);
  EXPECT_THAT(MCInstrInfo.get(Opcode).getImplicitUses()[1], llvm::X86::EFLAGS);
  const CodeTemplate CT = checkAndGetCodeTemplate(Opcode);
  EXPECT_THAT(CT.Info, HasSubstr("implicit"));
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionBuilder &IB = CT.Instructions[0];
  EXPECT_THAT(IB.getOpcode(), Opcode);
  ASSERT_THAT(IB.VariableValues, SizeIs(1)); // Imm.
  EXPECT_THAT(IB.VariableValues[0], IsInvalid()) << "Immediate is not set";
}

TEST_F(LatencySnippetGeneratorTest, ExplicitSelfDependency) {
  // ADD16ri self alias because Op0 and Op1 are tied together.

  // explicit def 0       : reg RegClass=GR16
  // explicit use 1       : reg RegClass=GR16 | TIED_TO:0
  // explicit use 2       : imm
  // implicit def         : EFLAGS
  const unsigned Opcode = llvm::X86::ADD16ri;
  EXPECT_THAT(MCInstrInfo.get(Opcode).getImplicitDefs()[0], llvm::X86::EFLAGS);
  const CodeTemplate CT = checkAndGetCodeTemplate(Opcode);
  EXPECT_THAT(CT.Info, HasSubstr("explicit"));
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionBuilder &IB = CT.Instructions[0];
  EXPECT_THAT(IB.getOpcode(), Opcode);
  ASSERT_THAT(IB.VariableValues, SizeIs(2));
  EXPECT_THAT(IB.VariableValues[0], IsReg()) << "Operand 0 and 1";
  EXPECT_THAT(IB.VariableValues[1], IsInvalid()) << "Operand 2 is not set";
}

TEST_F(LatencySnippetGeneratorTest, DependencyThroughOtherOpcode) {
  // CMP64rr
  // explicit use 0       : reg RegClass=GR64
  // explicit use 1       : reg RegClass=GR64
  // implicit def         : EFLAGS

  const unsigned Opcode = llvm::X86::CMP64rr;
  const CodeTemplate CT = checkAndGetCodeTemplate(Opcode);
  EXPECT_THAT(CT.Info, HasSubstr("cycle through"));
  ASSERT_THAT(CT.Instructions, SizeIs(2));
  const InstructionBuilder &IB = CT.Instructions[0];
  EXPECT_THAT(IB.getOpcode(), Opcode);
  ASSERT_THAT(IB.VariableValues, SizeIs(2));
  EXPECT_THAT(IB.VariableValues, AnyOf(ElementsAre(IsReg(), IsInvalid()),
                                       ElementsAre(IsInvalid(), IsReg())));
  EXPECT_THAT(CT.Instructions[1].getOpcode(), Not(Opcode));
  // TODO: check that the two instructions alias each other.
}

TEST_F(LatencySnippetGeneratorTest, LAHF) {
  const unsigned Opcode = llvm::X86::LAHF;
  const CodeTemplate CT = checkAndGetCodeTemplate(Opcode);
  EXPECT_THAT(CT.Info, HasSubstr("cycle through"));
  ASSERT_THAT(CT.Instructions, SizeIs(2));
  const InstructionBuilder &IB = CT.Instructions[0];
  EXPECT_THAT(IB.getOpcode(), Opcode);
  ASSERT_THAT(IB.VariableValues, SizeIs(0));
}

TEST_F(UopsSnippetGeneratorTest, ParallelInstruction) {
  // BNDCL32rr is parallel no matter what.

  // explicit use 0       : reg RegClass=BNDR
  // explicit use 1       : reg RegClass=GR32

  const unsigned Opcode = llvm::X86::BNDCL32rr;
  const CodeTemplate CT = checkAndGetCodeTemplate(Opcode);
  EXPECT_THAT(CT.Info, HasSubstr("parallel"));
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionBuilder &IB = CT.Instructions[0];
  EXPECT_THAT(IB.getOpcode(), Opcode);
  ASSERT_THAT(IB.VariableValues, SizeIs(2));
  EXPECT_THAT(IB.VariableValues[0], IsInvalid());
  EXPECT_THAT(IB.VariableValues[1], IsInvalid());
}

TEST_F(UopsSnippetGeneratorTest, SerialInstruction) {
  // CDQ is serial no matter what.

  // implicit def         : EAX
  // implicit def         : EDX
  // implicit use         : EAX
  const unsigned Opcode = llvm::X86::CDQ;
  const CodeTemplate CT = checkAndGetCodeTemplate(Opcode);
  EXPECT_THAT(CT.Info, HasSubstr("serial"));
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionBuilder &IB = CT.Instructions[0];
  EXPECT_THAT(IB.getOpcode(), Opcode);
  ASSERT_THAT(IB.VariableValues, SizeIs(0));
}

TEST_F(UopsSnippetGeneratorTest, StaticRenaming) {
  // CMOVA32rr has tied variables, we enumarate the possible values to execute
  // as many in parallel as possible.

  // explicit def 0       : reg RegClass=GR32
  // explicit use 1       : reg RegClass=GR32 | TIED_TO:0
  // explicit use 2       : reg RegClass=GR32
  // implicit use         : EFLAGS
  const unsigned Opcode = llvm::X86::CMOVA32rr;
  const CodeTemplate CT = checkAndGetCodeTemplate(Opcode);
  EXPECT_THAT(CT.Info, HasSubstr("static renaming"));
  constexpr const unsigned kInstructionCount = 15;
  ASSERT_THAT(CT.Instructions, SizeIs(kInstructionCount));
  std::unordered_set<unsigned> AllDefRegisters;
  for (const auto &IB : CT.Instructions) {
    ASSERT_THAT(IB.VariableValues, SizeIs(2));
    AllDefRegisters.insert(IB.VariableValues[0].getReg());
  }
  EXPECT_THAT(AllDefRegisters, SizeIs(kInstructionCount))
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
  const CodeTemplate CT = checkAndGetCodeTemplate(Opcode);
  EXPECT_THAT(CT.Info, HasSubstr("no tied variables"));
  ASSERT_THAT(CT.Instructions, SizeIs(1));
  const InstructionBuilder &IB = CT.Instructions[0];
  EXPECT_THAT(IB.getOpcode(), Opcode);
  ASSERT_THAT(IB.VariableValues, SizeIs(4));
  EXPECT_THAT(IB.VariableValues[0].getReg(), Not(IB.VariableValues[1].getReg()))
      << "Def is different from first Use";
  EXPECT_THAT(IB.VariableValues[0].getReg(), Not(IB.VariableValues[2].getReg()))
      << "Def is different from second Use";
  EXPECT_THAT(IB.VariableValues[3], IsInvalid());
}

TEST_F(UopsSnippetGeneratorTest, MemoryUse) {
  // Mov32rm reads from memory.
  const unsigned Opcode = llvm::X86::MOV32rm;
  const CodeTemplate CT = checkAndGetCodeTemplate(Opcode);
  EXPECT_THAT(CT.Info, HasSubstr("no tied variables"));
  ASSERT_THAT(CT.Instructions,
              SizeIs(UopsSnippetGenerator::kMinNumDifferentAddresses));
  const InstructionBuilder &IB = CT.Instructions[0];
  EXPECT_THAT(IB.getOpcode(), Opcode);
  ASSERT_THAT(IB.VariableValues, SizeIs(6));
  EXPECT_EQ(IB.VariableValues[2].getImm(), 1);
  EXPECT_EQ(IB.VariableValues[3].getReg(), 0u);
  EXPECT_EQ(IB.VariableValues[4].getImm(), 0);
  EXPECT_EQ(IB.VariableValues[5].getReg(), 0u);
}

TEST_F(UopsSnippetGeneratorTest, MemoryUse_Movsb) {
  // MOVSB writes to scratch memory register.
  const unsigned Opcode = llvm::X86::MOVSB;
  auto Error = Generator.generateCodeTemplate(Opcode).takeError();
  EXPECT_TRUE((bool)Error);
  llvm::consumeError(std::move(Error));
}

class FakeSnippetGenerator : public SnippetGenerator {
public:
  FakeSnippetGenerator(const LLVMState &State) : SnippetGenerator(State) {}

  Instruction createInstruction(unsigned Opcode) {
    return Instruction(State.getInstrInfo().get(Opcode), RATC);
  }

private:
  llvm::Expected<CodeTemplate>
  generateCodeTemplate(unsigned Opcode) const override {
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
  InstructionBuilder IB(Generator.createInstruction(llvm::X86::ADD16ri));
  IB.getValueFor(IB.Instr.Variables[0]) =
      llvm::MCOperand::createReg(llvm::X86::AX);
  std::vector<InstructionBuilder> Snippet;
  Snippet.push_back(std::move(IB));
  const auto RIV = Generator.computeRegisterInitialValues(Snippet);
  EXPECT_THAT(RIV, ElementsAre(IsRegisterValue(llvm::X86::AX, llvm::APInt())));
}

TEST_F(FakeSnippetGeneratorTest, ComputeRegisterInitialValuesAdd64rr) {
  // ADD64rr:
  //  mov64ri rax, 42
  //  add64rr rax, rax, rbx
  // -> only rbx needs defining.
  std::vector<InstructionBuilder> Snippet;
  {
    InstructionBuilder Mov(Generator.createInstruction(llvm::X86::MOV64ri));
    Mov.getValueFor(Mov.Instr.Variables[0]) =
        llvm::MCOperand::createReg(llvm::X86::RAX);
    Mov.getValueFor(Mov.Instr.Variables[1]) = llvm::MCOperand::createImm(42);
    Snippet.push_back(std::move(Mov));
  }
  {
    InstructionBuilder Add(Generator.createInstruction(llvm::X86::ADD64rr));
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
