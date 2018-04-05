//===-- InstructionSnippetGeneratorTest.cpp ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InstructionSnippetGenerator.h"
#include "X86InstrInfo.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <set>

namespace llvm {

bool operator==(const MCOperand &A, const MCOperand &B) {
  if ((A.isValid() == false) && (B.isValid() == false))
    return true;
  if (A.isReg() && B.isReg())
    return A.getReg() == B.getReg();
  if (A.isImm() && B.isImm())
    return A.getImm() == B.getImm();
  return false;
}

} // namespace llvm

namespace exegesis {
namespace {

using testing::_;
using testing::AllOf;
using testing::AnyOf;
using testing::Contains;
using testing::ElementsAre;
using testing::Eq;
using testing::Field;
using testing::Not;
using testing::SizeIs;
using testing::UnorderedElementsAre;
using testing::Value;

using llvm::X86::AL;
using llvm::X86::AX;
using llvm::X86::EFLAGS;
using llvm::X86::RAX;

class MCInstrDescViewTest : public ::testing::Test {
protected:
  MCInstrDescViewTest()
      : TheTriple(llvm::sys::getProcessTriple()),
        CpuName(llvm::sys::getHostCPUName().str()) {}

  void SetUp() override {
    llvm::InitializeNativeTarget();

    std::string Error;
    const auto *Target = llvm::TargetRegistry::lookupTarget(TheTriple, Error);
    InstrInfo.reset(Target->createMCInstrInfo());
    RegInfo.reset(Target->createMCRegInfo(TheTriple));
  }

  const std::string TheTriple;
  const std::string CpuName;
  std::unique_ptr<const llvm::MCInstrInfo> InstrInfo;
  std::unique_ptr<const llvm::MCRegisterInfo> RegInfo;
};

MATCHER(IsDef, "") { return arg.IsDef; }
MATCHER(IsUse, "") { return arg.IsUse; }
MATCHER_P2(EqVarAssignement, VariableIndexMatcher, AssignedRegisterMatcher,
           "") {
  return Value(
      arg,
      AllOf(Field(&VariableAssignment::VarIdx, VariableIndexMatcher),
            Field(&VariableAssignment::AssignedReg, AssignedRegisterMatcher)));
}

size_t returnIndexZero(const size_t UpperBound) { return 0; }

TEST_F(MCInstrDescViewTest, DISABLED_XOR64rr) {
  const llvm::MCInstrDesc &InstrDesc = InstrInfo->get(llvm::X86::XOR64rr);
  const auto Vars =
      getVariables(*RegInfo, InstrDesc, llvm::BitVector(RegInfo->getNumRegs()));

  // XOR64rr has the following operands:
  //  0. out register
  //  1. in register (tied to out)
  //  2. in register
  //  3. out EFLAGS (implicit)
  //
  // This translates to 3 variables, one for 0 and 1, one for 2, one for 3.
  ASSERT_THAT(Vars, SizeIs(3));

  EXPECT_THAT(Vars[0].ExplicitOperands, ElementsAre(0, 1));
  EXPECT_THAT(Vars[1].ExplicitOperands, ElementsAre(2));
  EXPECT_THAT(Vars[2].ExplicitOperands, ElementsAre()); // implicit

  EXPECT_THAT(Vars[0], AllOf(IsUse(), IsDef()));
  EXPECT_THAT(Vars[1], AllOf(IsUse(), Not(IsDef())));
  EXPECT_THAT(Vars[2], AllOf(Not(IsUse()), IsDef()));

  EXPECT_THAT(Vars[0].PossibleRegisters, Contains(RAX));
  EXPECT_THAT(Vars[1].PossibleRegisters, Contains(RAX));
  EXPECT_THAT(Vars[2].PossibleRegisters, ElementsAre(EFLAGS));

  // Computing chains.
  const auto Chains = computeSequentialAssignmentChains(*RegInfo, Vars);

  // Because operands 0 and 1 are tied together any possible value for variable
  // 0 would do.
  for (const auto &Reg : Vars[0].PossibleRegisters) {
    EXPECT_THAT(Chains, Contains(ElementsAre(EqVarAssignement(0, Reg))));
  }

  // We also have chains going through operand 0 to 2 (i.e. Vars 0 and 1).
  EXPECT_THAT(Vars[0].PossibleRegisters, Eq(Vars[1].PossibleRegisters))
      << "Variables 0 and 1 are of the same class";
  for (const auto &Reg : Vars[0].PossibleRegisters) {
    EXPECT_THAT(Chains,
                Contains(UnorderedElementsAre(EqVarAssignement(0, Reg),
                                              EqVarAssignement(1, Reg))));
  }

  // EFLAGS does not appear as an input therefore no chain can contain EFLAGS.
  EXPECT_THAT(Chains, Not(Contains(Contains(EqVarAssignement(_, EFLAGS)))));

  // Computing assignment.
  const auto Regs = getRandomAssignment(Vars, Chains, &returnIndexZero);
  EXPECT_THAT(Regs, ElementsAre(RAX, RAX, EFLAGS));

  // Generating assembler representation.
  const llvm::MCInst Inst = generateMCInst(InstrDesc, Vars, Regs);
  EXPECT_THAT(Inst.getOpcode(), llvm::X86::XOR64rr);
  EXPECT_THAT(Inst.getNumOperands(), 3);
  EXPECT_THAT(Inst.getOperand(0), llvm::MCOperand::createReg(RAX));
  EXPECT_THAT(Inst.getOperand(1), llvm::MCOperand::createReg(RAX));
  EXPECT_THAT(Inst.getOperand(2), llvm::MCOperand::createReg(RAX));
}

TEST_F(MCInstrDescViewTest, DISABLED_AAA) {
  const llvm::MCInstrDesc &InstrDesc = InstrInfo->get(llvm::X86::AAA);
  const auto Vars =
      getVariables(*RegInfo, InstrDesc, llvm::BitVector(RegInfo->getNumRegs()));

  // AAA has the following operands:
  //  0. out AX      (implicit)
  //  1. out EFLAGS  (implicit)
  //  2. in AL       (implicit)
  //  3. in EFLAGS   (implicit)
  //
  // This translates to 4 Vars (non are tied together).
  ASSERT_THAT(Vars, SizeIs(4));

  EXPECT_THAT(Vars[0].ExplicitOperands, ElementsAre()); // implicit
  EXPECT_THAT(Vars[1].ExplicitOperands, ElementsAre()); // implicit
  EXPECT_THAT(Vars[2].ExplicitOperands, ElementsAre()); // implicit
  EXPECT_THAT(Vars[3].ExplicitOperands, ElementsAre()); // implicit

  EXPECT_THAT(Vars[0], AllOf(Not(IsUse()), IsDef()));
  EXPECT_THAT(Vars[1], AllOf(Not(IsUse()), IsDef()));
  EXPECT_THAT(Vars[2], AllOf(IsUse(), Not(IsDef())));
  EXPECT_THAT(Vars[3], AllOf(IsUse(), Not(IsDef())));

  EXPECT_THAT(Vars[0].PossibleRegisters, ElementsAre(AX));
  EXPECT_THAT(Vars[1].PossibleRegisters, ElementsAre(EFLAGS));
  EXPECT_THAT(Vars[2].PossibleRegisters, ElementsAre(AL));
  EXPECT_THAT(Vars[3].PossibleRegisters, ElementsAre(EFLAGS));

  const auto Chains = computeSequentialAssignmentChains(*RegInfo, Vars);
  EXPECT_THAT(Chains,
              ElementsAre(UnorderedElementsAre(EqVarAssignement(0, AX),
                                               EqVarAssignement(2, AL)),
                          UnorderedElementsAre(EqVarAssignement(1, EFLAGS),
                                               EqVarAssignement(3, EFLAGS))));

  // Computing assignment.
  const auto Regs = getRandomAssignment(Vars, Chains, &returnIndexZero);
  EXPECT_THAT(Regs, ElementsAre(AX, EFLAGS, AL, EFLAGS));

  // Generating assembler representation.
  const llvm::MCInst Inst = generateMCInst(InstrDesc, Vars, Regs);
  EXPECT_THAT(Inst.getOpcode(), llvm::X86::AAA);
  EXPECT_THAT(Inst.getNumOperands(), 0) << "All operands are implicit";
}

TEST_F(MCInstrDescViewTest, DISABLED_ReservedRegisters) {
  llvm::BitVector ReservedRegisters(RegInfo->getNumRegs());

  const llvm::MCInstrDesc &InstrDesc = InstrInfo->get(llvm::X86::XOR64rr);
  {
    const auto Vars = getVariables(*RegInfo, InstrDesc, ReservedRegisters);
    ASSERT_THAT(Vars, SizeIs(3));
    EXPECT_THAT(Vars[0].PossibleRegisters, Contains(RAX));
    EXPECT_THAT(Vars[1].PossibleRegisters, Contains(RAX));
  }

  // Disable RAX.
  ReservedRegisters.set(RAX);
  {
    const auto Vars = getVariables(*RegInfo, InstrDesc, ReservedRegisters);
    ASSERT_THAT(Vars, SizeIs(3));
    EXPECT_THAT(Vars[0].PossibleRegisters, Not(Contains(RAX)));
    EXPECT_THAT(Vars[1].PossibleRegisters, Not(Contains(RAX)));
  }
}

Variable makeVariableWithRegisters(bool IsReg,
                                   std::initializer_list<int> Regs) {
  assert((IsReg || (Regs.size() == 0)) && "IsReg => !(Regs.size() == 0)");
  Variable Var;
  Var.IsReg = IsReg;
  Var.PossibleRegisters.insert(Regs.begin(), Regs.end());
  return Var;
}

TEST(getExclusiveAssignment, TriviallyFeasible) {
  const std::vector<Variable> Vars = {
      makeVariableWithRegisters(true, {3}),
      makeVariableWithRegisters(false, {}),
      makeVariableWithRegisters(true, {4}),
      makeVariableWithRegisters(true, {5}),
  };
  const auto Regs = getExclusiveAssignment(Vars);
  EXPECT_THAT(Regs, ElementsAre(3, 0, 4, 5));
}

TEST(getExclusiveAssignment, TriviallyInfeasible1) {
  const std::vector<Variable> Vars = {
      makeVariableWithRegisters(true, {3}),
      makeVariableWithRegisters(true, {}),
      makeVariableWithRegisters(true, {4}),
      makeVariableWithRegisters(true, {5}),
  };
  const auto Regs = getExclusiveAssignment(Vars);
  EXPECT_THAT(Regs, ElementsAre());
}

TEST(getExclusiveAssignment, TriviallyInfeasible) {
  const std::vector<Variable> Vars = {
      makeVariableWithRegisters(true, {4}),
      makeVariableWithRegisters(true, {4}),
  };
  const auto Regs = getExclusiveAssignment(Vars);
  EXPECT_THAT(Regs, ElementsAre());
}

TEST(getExclusiveAssignment, Feasible1) {
  const std::vector<Variable> Vars = {
      makeVariableWithRegisters(true, {4, 3}),
      makeVariableWithRegisters(true, {6, 3}),
      makeVariableWithRegisters(true, {6, 4}),
  };
  const auto Regs = getExclusiveAssignment(Vars);
  ASSERT_THAT(Regs, AnyOf(ElementsAre(3, 6, 4), ElementsAre(4, 3, 6)));
}

TEST(getExclusiveAssignment, Feasible2) {
  const std::vector<Variable> Vars = {
      makeVariableWithRegisters(true, {1, 2}),
      makeVariableWithRegisters(true, {3, 4}),
  };
  const auto Regs = getExclusiveAssignment(Vars);
  ASSERT_THAT(Regs, AnyOf(ElementsAre(1, 3), ElementsAre(1, 4),
                          ElementsAre(2, 3), ElementsAre(2, 4)));
}

TEST(getGreedyAssignment, Infeasible) {
  const std::vector<Variable> Vars = {
      makeVariableWithRegisters(true, {}),
      makeVariableWithRegisters(true, {1, 2}),
  };
  const auto Regs = getGreedyAssignment(Vars);
  ASSERT_THAT(Regs, ElementsAre());
}

TEST(getGreedyAssignment, FeasibleNoFallback) {
  const std::vector<Variable> Vars = {
      makeVariableWithRegisters(true, {1, 2}),
      makeVariableWithRegisters(false, {}),
      makeVariableWithRegisters(true, {2, 3}),
  };
  const auto Regs = getGreedyAssignment(Vars);
  ASSERT_THAT(Regs, ElementsAre(1, 0, 2));
}

TEST(getGreedyAssignment, Feasible) {
  const std::vector<Variable> Vars = {
      makeVariableWithRegisters(false, {}),
      makeVariableWithRegisters(true, {1, 2}),
      makeVariableWithRegisters(true, {2, 3}),
      makeVariableWithRegisters(true, {2, 3}),
      makeVariableWithRegisters(true, {2, 3}),
  };
  const auto Regs = getGreedyAssignment(Vars);
  ASSERT_THAT(Regs, ElementsAre(0, 1, 2, 3, 2));
}

} // namespace
} // namespace exegesis
