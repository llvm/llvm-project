//===-- TargetTest.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Target.h"

#include <cassert>
#include <memory>

#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "llvm/MC/MCInstPrinter.h"

namespace llvm {

bool operator==(const MCOperand &a, const MCOperand &b) {
  if (a.isImm() && b.isImm())
    return a.getImm() == b.getImm();
  if (a.isReg() && b.isReg())
    return a.getReg() == b.getReg();
  return false;
}

bool operator==(const MCInst &a, const MCInst &b) {
  if (a.getOpcode() != b.getOpcode())
    return false;
  if (a.getNumOperands() != b.getNumOperands())
    return false;
  for (unsigned I = 0; I < a.getNumOperands(); ++I) {
    if (!(a.getOperand(I) == b.getOperand(I)))
      return false;
  }
  return true;
}

} // namespace llvm

namespace llvm {
namespace exegesis {

void InitializeX86ExegesisTarget();

namespace {

using testing::AllOf;
using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::Eq;
using testing::Gt;
using testing::Matcher;
using testing::NotNull;
using testing::Property;
using testing::SizeIs;

Matcher<MCOperand> IsImm(int64_t Value) {
  return AllOf(Property(&MCOperand::isImm, Eq(true)),
               Property(&MCOperand::getImm, Eq(Value)));
}

Matcher<MCOperand> IsReg(unsigned Reg) {
  return AllOf(Property(&MCOperand::isReg, Eq(true)),
               Property(&MCOperand::getReg, Eq(Reg)));
}

Matcher<MCInst> OpcodeIs(unsigned Opcode) {
  return Property(&MCInst::getOpcode, Eq(Opcode));
}

Matcher<MCInst> IsMovImmediate(unsigned Opcode, int64_t Reg, int64_t Value) {
  return AllOf(OpcodeIs(Opcode), ElementsAre(IsReg(Reg), IsImm(Value)));
}

Matcher<MCInst> IsMovValueToStack(unsigned Opcode, int64_t Value,
                                  size_t Offset) {
  return AllOf(OpcodeIs(Opcode),
               ElementsAre(IsReg(X86::RSP), IsImm(1), IsReg(0), IsImm(Offset),
                           IsReg(0), IsImm(Value)));
}

Matcher<MCInst> IsMovValueFromStack(unsigned Opcode, unsigned Reg) {
  return AllOf(OpcodeIs(Opcode),
               ElementsAre(IsReg(Reg), IsReg(X86::RSP), IsImm(1), IsReg(0),
                           IsImm(0), IsReg(0)));
}

Matcher<MCInst> IsStackAllocate(unsigned Size) {
  return AllOf(OpcodeIs(X86::SUB64ri8),
               ElementsAre(IsReg(X86::RSP), IsReg(X86::RSP), IsImm(Size)));
}

Matcher<MCInst> IsStackDeallocate(unsigned Size) {
  return AllOf(OpcodeIs(X86::ADD64ri8),
               ElementsAre(IsReg(X86::RSP), IsReg(X86::RSP), IsImm(Size)));
}

constexpr const char kTriple[] = "x86_64-unknown-linux";

class X86TargetTest : public ::testing::Test {
protected:
  X86TargetTest(const char *Features)
      : State(cantFail(LLVMState::Create(kTriple, "core2", Features))) {}

  static void SetUpTestCase() {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    InitializeX86ExegesisTarget();
  }

  std::vector<MCInst> setRegTo(unsigned Reg, const APInt &Value) {
    return State.getExegesisTarget().setRegTo(State.getSubtargetInfo(), Reg,
                                              Value);
  }

  const Instruction &getInstr(unsigned OpCode) {
    return State.getIC().getInstr(OpCode);
  }

  LLVMState State;
};

class X86Core2TargetTest : public X86TargetTest {
public:
  X86Core2TargetTest() : X86TargetTest("") {}
};

class X86Core2AvxTargetTest : public X86TargetTest {
public:
  X86Core2AvxTargetTest() : X86TargetTest("+avx") {}
};

class X86Core2Avx512TargetTest : public X86TargetTest {
public:
  X86Core2Avx512TargetTest() : X86TargetTest("+avx512vl") {}
};

TEST_F(X86Core2TargetTest, NoHighByteRegs) {
  EXPECT_TRUE(State.getRATC().reservedRegisters().test(X86::AH));
}

TEST_F(X86Core2TargetTest, SetFlags) {
  const unsigned Reg = X86::EFLAGS;
  EXPECT_THAT(setRegTo(Reg, APInt(64, 0x1111222233334444ULL)),
              ElementsAre(IsStackAllocate(8),
                          IsMovValueToStack(X86::MOV32mi, 0x33334444UL, 0),
                          IsMovValueToStack(X86::MOV32mi, 0x11112222UL, 4),
                          OpcodeIs(X86::POPF64)));
}

TEST_F(X86Core2TargetTest, SetRegToGR8Value) {
  const uint8_t Value = 0xFFU;
  const unsigned Reg = X86::AL;
  EXPECT_THAT(setRegTo(Reg, APInt(8, Value)),
              ElementsAre(IsMovImmediate(X86::MOV8ri, Reg, Value)));
}

TEST_F(X86Core2TargetTest, SetRegToGR16Value) {
  const uint16_t Value = 0xFFFFU;
  const unsigned Reg = X86::BX;
  EXPECT_THAT(setRegTo(Reg, APInt(16, Value)),
              ElementsAre(IsMovImmediate(X86::MOV16ri, Reg, Value)));
}

TEST_F(X86Core2TargetTest, SetRegToGR32Value) {
  const uint32_t Value = 0x7FFFFU;
  const unsigned Reg = X86::ECX;
  EXPECT_THAT(setRegTo(Reg, APInt(32, Value)),
              ElementsAre(IsMovImmediate(X86::MOV32ri, Reg, Value)));
}

TEST_F(X86Core2TargetTest, SetRegToGR64Value) {
  const uint64_t Value = 0x7FFFFFFFFFFFFFFFULL;
  const unsigned Reg = X86::RDX;
  EXPECT_THAT(setRegTo(Reg, APInt(64, Value)),
              ElementsAre(IsMovImmediate(X86::MOV64ri, Reg, Value)));
}

TEST_F(X86Core2TargetTest, SetRegToVR64Value) {
  EXPECT_THAT(setRegTo(X86::MM0, APInt(64, 0x1111222233334444ULL)),
              ElementsAre(IsStackAllocate(8),
                          IsMovValueToStack(X86::MOV32mi, 0x33334444UL, 0),
                          IsMovValueToStack(X86::MOV32mi, 0x11112222UL, 4),
                          IsMovValueFromStack(X86::MMX_MOVQ64rm, X86::MM0),
                          IsStackDeallocate(8)));
}

TEST_F(X86Core2TargetTest, SetRegToVR128Value_Use_MOVDQUrm) {
  EXPECT_THAT(
      setRegTo(X86::XMM0, APInt(128, "11112222333344445555666677778888", 16)),
      ElementsAre(IsStackAllocate(16),
                  IsMovValueToStack(X86::MOV32mi, 0x77778888UL, 0),
                  IsMovValueToStack(X86::MOV32mi, 0x55556666UL, 4),
                  IsMovValueToStack(X86::MOV32mi, 0x33334444UL, 8),
                  IsMovValueToStack(X86::MOV32mi, 0x11112222UL, 12),
                  IsMovValueFromStack(X86::MOVDQUrm, X86::XMM0),
                  IsStackDeallocate(16)));
}

TEST_F(X86Core2AvxTargetTest, SetRegToVR128Value_Use_VMOVDQUrm) {
  EXPECT_THAT(
      setRegTo(X86::XMM0, APInt(128, "11112222333344445555666677778888", 16)),
      ElementsAre(IsStackAllocate(16),
                  IsMovValueToStack(X86::MOV32mi, 0x77778888UL, 0),
                  IsMovValueToStack(X86::MOV32mi, 0x55556666UL, 4),
                  IsMovValueToStack(X86::MOV32mi, 0x33334444UL, 8),
                  IsMovValueToStack(X86::MOV32mi, 0x11112222UL, 12),
                  IsMovValueFromStack(X86::VMOVDQUrm, X86::XMM0),
                  IsStackDeallocate(16)));
}

TEST_F(X86Core2Avx512TargetTest, SetRegToVR128Value_Use_VMOVDQU32Z128rm) {
  EXPECT_THAT(
      setRegTo(X86::XMM0, APInt(128, "11112222333344445555666677778888", 16)),
      ElementsAre(IsStackAllocate(16),
                  IsMovValueToStack(X86::MOV32mi, 0x77778888UL, 0),
                  IsMovValueToStack(X86::MOV32mi, 0x55556666UL, 4),
                  IsMovValueToStack(X86::MOV32mi, 0x33334444UL, 8),
                  IsMovValueToStack(X86::MOV32mi, 0x11112222UL, 12),
                  IsMovValueFromStack(X86::VMOVDQU32Z128rm, X86::XMM0),
                  IsStackDeallocate(16)));
}

TEST_F(X86Core2AvxTargetTest, SetRegToVR256Value_Use_VMOVDQUYrm) {
  const char ValueStr[] =
      "1111111122222222333333334444444455555555666666667777777788888888";
  EXPECT_THAT(
      setRegTo(X86::YMM0, APInt(256, ValueStr, 16)),
      ElementsAreArray({IsStackAllocate(32),
                        IsMovValueToStack(X86::MOV32mi, 0x88888888UL, 0),
                        IsMovValueToStack(X86::MOV32mi, 0x77777777UL, 4),
                        IsMovValueToStack(X86::MOV32mi, 0x66666666UL, 8),
                        IsMovValueToStack(X86::MOV32mi, 0x55555555UL, 12),
                        IsMovValueToStack(X86::MOV32mi, 0x44444444UL, 16),
                        IsMovValueToStack(X86::MOV32mi, 0x33333333UL, 20),
                        IsMovValueToStack(X86::MOV32mi, 0x22222222UL, 24),
                        IsMovValueToStack(X86::MOV32mi, 0x11111111UL, 28),
                        IsMovValueFromStack(X86::VMOVDQUYrm, X86::YMM0),
                        IsStackDeallocate(32)}));
}

TEST_F(X86Core2Avx512TargetTest, SetRegToVR256Value_Use_VMOVDQU32Z256rm) {
  const char ValueStr[] =
      "1111111122222222333333334444444455555555666666667777777788888888";
  EXPECT_THAT(
      setRegTo(X86::YMM0, APInt(256, ValueStr, 16)),
      ElementsAreArray({IsStackAllocate(32),
                        IsMovValueToStack(X86::MOV32mi, 0x88888888UL, 0),
                        IsMovValueToStack(X86::MOV32mi, 0x77777777UL, 4),
                        IsMovValueToStack(X86::MOV32mi, 0x66666666UL, 8),
                        IsMovValueToStack(X86::MOV32mi, 0x55555555UL, 12),
                        IsMovValueToStack(X86::MOV32mi, 0x44444444UL, 16),
                        IsMovValueToStack(X86::MOV32mi, 0x33333333UL, 20),
                        IsMovValueToStack(X86::MOV32mi, 0x22222222UL, 24),
                        IsMovValueToStack(X86::MOV32mi, 0x11111111UL, 28),
                        IsMovValueFromStack(X86::VMOVDQU32Z256rm, X86::YMM0),
                        IsStackDeallocate(32)}));
}

TEST_F(X86Core2Avx512TargetTest, SetRegToVR512Value) {
  const char ValueStr[] =
      "1111111122222222333333334444444455555555666666667777777788888888"
      "99999999AAAAAAAABBBBBBBBCCCCCCCCDDDDDDDDEEEEEEEEFFFFFFFF00000000";
  EXPECT_THAT(
      setRegTo(X86::ZMM0, APInt(512, ValueStr, 16)),
      ElementsAreArray({IsStackAllocate(64),
                        IsMovValueToStack(X86::MOV32mi, 0x00000000UL, 0),
                        IsMovValueToStack(X86::MOV32mi, 0xFFFFFFFFUL, 4),
                        IsMovValueToStack(X86::MOV32mi, 0xEEEEEEEEUL, 8),
                        IsMovValueToStack(X86::MOV32mi, 0xDDDDDDDDUL, 12),
                        IsMovValueToStack(X86::MOV32mi, 0xCCCCCCCCUL, 16),
                        IsMovValueToStack(X86::MOV32mi, 0xBBBBBBBBUL, 20),
                        IsMovValueToStack(X86::MOV32mi, 0xAAAAAAAAUL, 24),
                        IsMovValueToStack(X86::MOV32mi, 0x99999999UL, 28),
                        IsMovValueToStack(X86::MOV32mi, 0x88888888UL, 32),
                        IsMovValueToStack(X86::MOV32mi, 0x77777777UL, 36),
                        IsMovValueToStack(X86::MOV32mi, 0x66666666UL, 40),
                        IsMovValueToStack(X86::MOV32mi, 0x55555555UL, 44),
                        IsMovValueToStack(X86::MOV32mi, 0x44444444UL, 48),
                        IsMovValueToStack(X86::MOV32mi, 0x33333333UL, 52),
                        IsMovValueToStack(X86::MOV32mi, 0x22222222UL, 56),
                        IsMovValueToStack(X86::MOV32mi, 0x11111111UL, 60),
                        IsMovValueFromStack(X86::VMOVDQU32Zrm, X86::ZMM0),
                        IsStackDeallocate(64)}));
}

// Note: We always put 80 bits on the stack independently of the size of the
// value. This uses a bit more space but makes the code simpler.

TEST_F(X86Core2TargetTest, SetRegToST0_32Bits) {
  EXPECT_THAT(setRegTo(X86::ST0, APInt(32, 0x11112222ULL)),
              ElementsAre(IsStackAllocate(10),
                          IsMovValueToStack(X86::MOV32mi, 0x11112222UL, 0),
                          IsMovValueToStack(X86::MOV32mi, 0x00000000UL, 4),
                          IsMovValueToStack(X86::MOV16mi, 0x0000UL, 8),
                          OpcodeIs(X86::LD_F80m), IsStackDeallocate(10)));
}

TEST_F(X86Core2TargetTest, SetRegToST1_32Bits) {
  const MCInst CopySt0ToSt1 = MCInstBuilder(X86::ST_Frr).addReg(X86::ST1);
  EXPECT_THAT(setRegTo(X86::ST1, APInt(32, 0x11112222ULL)),
              ElementsAre(IsStackAllocate(10),
                          IsMovValueToStack(X86::MOV32mi, 0x11112222UL, 0),
                          IsMovValueToStack(X86::MOV32mi, 0x00000000UL, 4),
                          IsMovValueToStack(X86::MOV16mi, 0x0000UL, 8),
                          OpcodeIs(X86::LD_F80m), CopySt0ToSt1,
                          IsStackDeallocate(10)));
}

TEST_F(X86Core2TargetTest, SetRegToST0_64Bits) {
  EXPECT_THAT(setRegTo(X86::ST0, APInt(64, 0x1111222233334444ULL)),
              ElementsAre(IsStackAllocate(10),
                          IsMovValueToStack(X86::MOV32mi, 0x33334444UL, 0),
                          IsMovValueToStack(X86::MOV32mi, 0x11112222UL, 4),
                          IsMovValueToStack(X86::MOV16mi, 0x0000UL, 8),
                          OpcodeIs(X86::LD_F80m), IsStackDeallocate(10)));
}

TEST_F(X86Core2TargetTest, SetRegToST0_80Bits) {
  EXPECT_THAT(setRegTo(X86::ST0, APInt(80, "11112222333344445555", 16)),
              ElementsAre(IsStackAllocate(10),
                          IsMovValueToStack(X86::MOV32mi, 0x44445555UL, 0),
                          IsMovValueToStack(X86::MOV32mi, 0x22223333UL, 4),
                          IsMovValueToStack(X86::MOV16mi, 0x1111UL, 8),
                          OpcodeIs(X86::LD_F80m), IsStackDeallocate(10)));
}

TEST_F(X86Core2TargetTest, SetRegToFP0_80Bits) {
  EXPECT_THAT(setRegTo(X86::FP0, APInt(80, "11112222333344445555", 16)),
              ElementsAre(IsStackAllocate(10),
                          IsMovValueToStack(X86::MOV32mi, 0x44445555UL, 0),
                          IsMovValueToStack(X86::MOV32mi, 0x22223333UL, 4),
                          IsMovValueToStack(X86::MOV16mi, 0x1111UL, 8),
                          OpcodeIs(X86::LD_Fp80m), IsStackDeallocate(10)));
}

TEST_F(X86Core2TargetTest, SetRegToFP1_32Bits) {
  EXPECT_THAT(setRegTo(X86::FP1, APInt(32, 0x11112222ULL)),
              ElementsAre(IsStackAllocate(10),
                          IsMovValueToStack(X86::MOV32mi, 0x11112222UL, 0),
                          IsMovValueToStack(X86::MOV32mi, 0x00000000UL, 4),
                          IsMovValueToStack(X86::MOV16mi, 0x0000UL, 8),
                          OpcodeIs(X86::LD_Fp80m), IsStackDeallocate(10)));
}

TEST_F(X86Core2TargetTest, SetRegToFP1_4Bits) {
  EXPECT_THAT(setRegTo(X86::FP1, APInt(4, 0x1ULL)),
              ElementsAre(IsStackAllocate(10),
                          IsMovValueToStack(X86::MOV32mi, 0x00000001UL, 0),
                          IsMovValueToStack(X86::MOV32mi, 0x00000000UL, 4),
                          IsMovValueToStack(X86::MOV16mi, 0x0000UL, 8),
                          OpcodeIs(X86::LD_Fp80m), IsStackDeallocate(10)));
}

TEST_F(X86Core2Avx512TargetTest, FillMemoryOperands_ADD64rm) {
  const Instruction &I = getInstr(X86::ADD64rm);
  InstructionTemplate IT(&I);
  constexpr const int kOffset = 42;
  State.getExegesisTarget().fillMemoryOperands(IT, X86::RDI, kOffset);
  // Memory is operands 2-6.
  EXPECT_THAT(IT.getValueFor(I.Operands[2]), IsReg(X86::RDI));
  EXPECT_THAT(IT.getValueFor(I.Operands[3]), IsImm(1));
  EXPECT_THAT(IT.getValueFor(I.Operands[4]), IsReg(0));
  EXPECT_THAT(IT.getValueFor(I.Operands[5]), IsImm(kOffset));
  EXPECT_THAT(IT.getValueFor(I.Operands[6]), IsReg(0));
}

TEST_F(X86Core2Avx512TargetTest, FillMemoryOperands_VGATHERDPSZ128rm) {
  const Instruction &I = getInstr(X86::VGATHERDPSZ128rm);
  InstructionTemplate IT(&I);
  constexpr const int kOffset = 42;
  State.getExegesisTarget().fillMemoryOperands(IT, X86::RDI, kOffset);
  // Memory is operands 4-8.
  EXPECT_THAT(IT.getValueFor(I.Operands[4]), IsReg(X86::RDI));
  EXPECT_THAT(IT.getValueFor(I.Operands[5]), IsImm(1));
  EXPECT_THAT(IT.getValueFor(I.Operands[6]), IsReg(0));
  EXPECT_THAT(IT.getValueFor(I.Operands[7]), IsImm(kOffset));
  EXPECT_THAT(IT.getValueFor(I.Operands[8]), IsReg(0));
}

TEST_F(X86Core2TargetTest, AllowAsBackToBack) {
  EXPECT_TRUE(
      State.getExegesisTarget().allowAsBackToBack(getInstr(X86::ADD64rr)));
  EXPECT_FALSE(
      State.getExegesisTarget().allowAsBackToBack(getInstr(X86::LEA64r)));
}

} // namespace
} // namespace exegesis
} // namespace llvm
