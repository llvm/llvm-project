//===- X86MCInstrAnalysisTest.cpp - Tests for X86 MC instruction semantics
//-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

#include <memory>

using namespace llvm;

namespace {

struct Context {
  static constexpr char TripleName[] = "x86_64-unknown-elf";
  const Triple TheTriple;
  std::unique_ptr<MCInstrInfo> Info;
  std::unique_ptr<MCInstrAnalysis> Analysis;

  Context() : TheTriple(TripleName) {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86TargetMC();

    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TheTriple, Error);
    if (!TheTarget)
      return;

    Info.reset(TheTarget->createMCInstrInfo());
    Analysis.reset(TheTarget->createMCInstrAnalysis(Info.get()));
  }
};

Context &getContext() {
  static Context Ctxt;
  return Ctxt;
}

} // namespace

TEST(X86MCInstrAnalysisTest, PushRegister) {
  const auto &Analysis = getContext().Analysis;
  ASSERT_NE(Analysis, nullptr);
  MCInst Inst = MCInstBuilder(X86::PUSH64r).addReg(X86::RDI);

  EXPECT_EQ(*Analysis->evaluateDefinedValue(Inst, X86::RSP),
            MCSemExpr::createReg(1, X86::RSP, -8));
  EXPECT_FALSE(Analysis->evaluateDefinedValue(Inst, X86::RDI));

  const auto Addresses = Analysis->evaluateStoreAddress(Inst);
  ASSERT_EQ(Addresses.size(), 1u);
  EXPECT_EQ(Addresses[0], MCSemAddrExpr::createReg(1, X86::RSP, -8));

  const auto Values = Analysis->getStoredValue(Inst);
  ASSERT_EQ(Values.size(), 1u);
  EXPECT_EQ(Values[0], MCSemExpr::createReg(1, X86::RDI, 0));
}

TEST(X86MCInstrAnalysisTest, PopRegister) {
  const auto &Analysis = getContext().Analysis;
  ASSERT_NE(Analysis, nullptr);
  MCInst Inst = MCInstBuilder(X86::POP64r).addReg(X86::RDI);

  EXPECT_EQ(
      *Analysis->evaluateDefinedValue(Inst, X86::RDI),
      MCSemExpr::createMem(1, MCSemAddrExpr::createReg(1, X86::RSP, 0), 0));
  EXPECT_EQ(*Analysis->evaluateDefinedValue(Inst, X86::RSP),
            MCSemExpr::createReg(1, X86::RSP, 8));
  EXPECT_FALSE(Analysis->evaluateDefinedValue(Inst, X86::RAX));
  EXPECT_TRUE(Analysis->evaluateStoreAddress(Inst).empty());
  EXPECT_TRUE(Analysis->getStoredValue(Inst).empty());
}

TEST(X86MCInstrAnalysisTest, RegisterMove) {
  const auto &Analysis = getContext().Analysis;
  ASSERT_NE(Analysis, nullptr);
  MCInst Inst = MCInstBuilder(X86::MOV64rr).addReg(X86::RBP).addReg(X86::RSP);

  EXPECT_EQ(*Analysis->evaluateDefinedValue(Inst, X86::RBP),
            MCSemExpr::createReg(1, X86::RSP, 0));
  EXPECT_FALSE(Analysis->evaluateDefinedValue(Inst, X86::RSP));
}

TEST(X86MCInstrAnalysisTest, StackStore) {
  const auto &Analysis = getContext().Analysis;
  ASSERT_NE(Analysis, nullptr);
  MCInst Inst = MCInstBuilder(X86::MOV64mr)
                    .addReg(X86::RSP)
                    .addImm(1)
                    .addReg(X86::NoRegister)
                    .addImm(-16)
                    .addReg(X86::NoRegister)
                    .addReg(X86::RDI);

  const auto Addresses = Analysis->evaluateStoreAddress(Inst);
  ASSERT_EQ(Addresses.size(), 1u);
  EXPECT_EQ(Addresses[0], MCSemAddrExpr::createReg(1, X86::RSP, -16));

  const auto Values = Analysis->getStoredValue(Inst);
  ASSERT_EQ(Values.size(), 1u);
  EXPECT_EQ(Values[0], MCSemExpr::createReg(1, X86::RDI, 0));
}

TEST(X86MCInstrAnalysisTest, StackLoad) {
  const auto &Analysis = getContext().Analysis;
  ASSERT_NE(Analysis, nullptr);
  MCInst Inst = MCInstBuilder(X86::MOV64rm)
                    .addReg(X86::RDI)
                    .addReg(X86::RBP)
                    .addImm(1)
                    .addReg(X86::NoRegister)
                    .addImm(-24)
                    .addReg(X86::NoRegister);

  EXPECT_EQ(
      *Analysis->evaluateDefinedValue(Inst, X86::RDI),
      MCSemExpr::createMem(1, MCSemAddrExpr::createReg(1, X86::RBP, -24), 0));
  EXPECT_FALSE(Analysis->evaluateDefinedValue(Inst, X86::RAX));
}

// Creates the specified memory inst loading into destination RDI
static MCInst memoryInst(unsigned Opcode, MCRegister Base, int64_t Scale,
                         MCRegister Index, int64_t Disp,
                         MCRegister Segment = X86::NoRegister,
                         unsigned Flags = 0) {
  MCInstBuilder Builder(Opcode);
  if (Opcode == X86::MOV64rm)
    Builder.addReg(X86::RDI);
  Builder.addReg(Base).addImm(Scale).addReg(Index).addImm(Disp).addReg(Segment);
  if (Opcode == X86::MOV64mr)
    Builder.addReg(X86::RDI);
  MCInst Inst = Builder;
  Inst.setFlags(Flags);
  return Inst;
}

// Creates the memory inst from parameters and tests for correct SemExprs
// Loads and stores are from/to RDI.
static void expectMemoryAddress(MCRegister Base, int64_t Scale,
                                MCRegister Index, int64_t Disp,
                                MCSemAddrExpr Expected) {
  const auto &Analysis = getContext().Analysis;
  ASSERT_NE(Analysis, nullptr);
  MCInst Load = memoryInst(X86::MOV64rm, Base, Scale, Index, Disp);
  auto Value = Analysis->evaluateDefinedValue(Load, X86::RDI);
  ASSERT_TRUE(Value);
  EXPECT_EQ(*Value, MCSemExpr::createMem(1, Expected, 0));

  MCInst Store = memoryInst(X86::MOV64mr, Base, Scale, Index, Disp);
  auto Addresses = Analysis->evaluateStoreAddress(Store);
  auto Values = Analysis->getStoredValue(Store);
  ASSERT_EQ(Addresses.size(), 1u);
  ASSERT_EQ(Values.size(), Addresses.size());
  EXPECT_EQ(Addresses[0], Expected);
  EXPECT_EQ(Values[0], MCSemExpr::createReg(1, X86::RDI, 0));
}

TEST(X86MCInstrAnalysisTest, BaseOnlyAddress) {
  for (MCRegister Base : {X86::RSP, X86::RBP, X86::RAX, X86::RDI, X86::R12})
    expectMemoryAddress(Base, 1, X86::NoRegister, -16,
                        MCSemAddrExpr::createReg(1, Base, -16));
}

TEST(X86MCInstrAnalysisTest, ScaleIgnoredWithoutIndex) {
  for (int64_t Scale : {1, 2, 4, 8})
    expectMemoryAddress(X86::RAX, Scale, X86::NoRegister, 16,
                        MCSemAddrExpr::createReg(1, X86::RAX, 16));
}

TEST(X86MCInstrAnalysisTest, IndexOnlyAddress) {
  for (int64_t Scale : {1, 2, 4, 8})
    for (int64_t Disp : {-16, 0, 16})
      expectMemoryAddress(X86::NoRegister, Scale, X86::RCX, Disp,
                          MCSemAddrExpr::createReg(Scale, X86::RCX, Disp));
}

TEST(X86MCInstrAnalysisTest, AbsoluteAddress) {
  for (int64_t Disp : {-16, 0, 16})
    expectMemoryAddress(X86::NoRegister, 1, X86::NoRegister, Disp,
                        MCSemAddrExpr::createConst(Disp));
}

// To test cases that are not handled yet.
// Loads and stores are from/to RDI as well.
static void expectUnknownMemoryAddress(MCRegister Base, int64_t Scale,
                                       MCRegister Index, int64_t Disp,
                                       MCRegister Segment = X86::NoRegister,
                                       unsigned Flags = 0) {
  const auto &Analysis = getContext().Analysis;
  ASSERT_NE(Analysis, nullptr);
  MCInst Load =
      memoryInst(X86::MOV64rm, Base, Scale, Index, Disp, Segment, Flags);
  EXPECT_FALSE(Analysis->evaluateDefinedValue(Load, X86::RDI));

  MCInst Store =
      memoryInst(X86::MOV64mr, Base, Scale, Index, Disp, Segment, Flags);
  EXPECT_TRUE(Analysis->evaluateStoreAddress(Store).empty());
  EXPECT_TRUE(Analysis->getStoredValue(Store).empty());
}

TEST(X86MCInstrAnalysisTest, UnsupportedMemoryAddresses) {
  // A base plus an index is outside the helper's supported subset for now.
  expectUnknownMemoryAddress(X86::RBP, 8, X86::RCX, 16);
  // Does not support RIP-relative addressing yet due to variable length inst
  // encoding.
  expectUnknownMemoryAddress(X86::RIP, 1, X86::NoRegister, 16);
  // Does not support segment registers yet.
  expectUnknownMemoryAddress(X86::RAX, 1, X86::NoRegister, 16, X86::FS);
  // Does not support 32 bit address registers yet.
  expectUnknownMemoryAddress(X86::EAX, 1, X86::NoRegister, 16);
  expectUnknownMemoryAddress(X86::NoRegister, 8, X86::ECX, 16);
  expectUnknownMemoryAddress(X86::NoRegister, 1, X86::NoRegister, -16,
                             X86::NoRegister, X86::IP_HAS_AD_SIZE);
}
