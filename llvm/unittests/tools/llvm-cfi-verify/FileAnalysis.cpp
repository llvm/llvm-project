//===- llvm/unittests/tools/llvm-cfi-verify/FileAnalysis.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../tools/llvm-cfi-verify/lib/FileAnalysis.h"
#include "../tools/llvm-cfi-verify/lib/GraphBuilder.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using Instr = ::llvm::cfi_verify::FileAnalysis::Instr;
using ::testing::Eq;
using ::testing::Field;

namespace llvm {
namespace cfi_verify {
namespace {
class ELFTestFileAnalysis : public FileAnalysis {
public:
  ELFTestFileAnalysis(StringRef Trip)
      : FileAnalysis(Triple(Trip), SubtargetFeatures()) {}

  // Expose this method publicly for testing.
  void parseSectionContents(ArrayRef<uint8_t> SectionBytes,
                            object::SectionedAddress Address) {
    FileAnalysis::parseSectionContents(SectionBytes, Address);
  }

  Error initialiseDisassemblyMembers() {
    return FileAnalysis::initialiseDisassemblyMembers();
  }
};

class BasicFileAnalysisTest : public ::testing::Test {
public:
  BasicFileAnalysisTest(StringRef Trip)
      : SuccessfullyInitialised(false), Analysis(Trip) {}
protected:
  virtual void SetUp() {
    IgnoreDWARFFlag = true;
    SuccessfullyInitialised = true;
    if (auto Err = Analysis.initialiseDisassemblyMembers()) {
      handleAllErrors(std::move(Err), [&](const UnsupportedDisassembly &E) {
        SuccessfullyInitialised = false;
        outs()
            << "Note: CFIVerifyTests are disabled due to lack of support "
               "on this build.\n";
      });
    }
  }

  bool SuccessfullyInitialised;
  ELFTestFileAnalysis Analysis;
};

class BasicX86FileAnalysisTest : public BasicFileAnalysisTest {
public:
  BasicX86FileAnalysisTest() : BasicFileAnalysisTest("x86_64--") {}
};

class BasicAArch64FileAnalysisTest : public BasicFileAnalysisTest {
public:
  BasicAArch64FileAnalysisTest() : BasicFileAnalysisTest("aarch64--") {}
};

TEST_F(BasicX86FileAnalysisTest, BasicDisassemblyTraversalTest) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x90,                   // 0: nop
          0xb0, 0x00,             // 1: mov $0x0, %al
          0x48, 0x89, 0xe5,       // 3: mov %rsp, %rbp
          0x48, 0x83, 0xec, 0x18, // 6: sub $0x18, %rsp
          0x48, 0xbe, 0xc4, 0x07, 0x40,
          0x00, 0x00, 0x00, 0x00, 0x00, // 10: movabs $0x4007c4, %rsi
          0x2f,                         // 20: (bad)
          0x41, 0x0e,                   // 21: rex.B (bad)
          0x62, 0x72, 0x65, 0x61, 0x6b, // 23: (bad) {%k1}
      },
      {0xDEADBEEF, 0x0});

  EXPECT_EQ(nullptr, Analysis.getInstruction(0x0));
  EXPECT_EQ(nullptr, Analysis.getInstruction(0x1000));

  // 0xDEADBEEF: nop
  const auto *InstrMeta = Analysis.getInstruction(0xDEADBEEF);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(0xDEADBEEF, InstrMeta->VMAddress);
  EXPECT_EQ(1u, InstrMeta->InstructionSize);
  EXPECT_TRUE(InstrMeta->Valid);

  const auto *NextInstrMeta = Analysis.getNextInstructionSequential(*InstrMeta);
  EXPECT_EQ(nullptr, Analysis.getPrevInstructionSequential(*InstrMeta));
  const auto *PrevInstrMeta = InstrMeta;

  // 0xDEADBEEF + 1: mov $0x0, %al
  InstrMeta = Analysis.getInstruction(0xDEADBEEF + 1);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(NextInstrMeta, InstrMeta);
  EXPECT_EQ(0xDEADBEEF + 1, InstrMeta->VMAddress);
  EXPECT_EQ(2u, InstrMeta->InstructionSize);
  EXPECT_TRUE(InstrMeta->Valid);

  NextInstrMeta = Analysis.getNextInstructionSequential(*InstrMeta);
  EXPECT_EQ(PrevInstrMeta, Analysis.getPrevInstructionSequential(*InstrMeta));
  PrevInstrMeta = InstrMeta;

  // 0xDEADBEEF + 3: mov %rsp, %rbp
  InstrMeta = Analysis.getInstruction(0xDEADBEEF + 3);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(NextInstrMeta, InstrMeta);
  EXPECT_EQ(0xDEADBEEF + 3, InstrMeta->VMAddress);
  EXPECT_EQ(3u, InstrMeta->InstructionSize);
  EXPECT_TRUE(InstrMeta->Valid);

  NextInstrMeta = Analysis.getNextInstructionSequential(*InstrMeta);
  EXPECT_EQ(PrevInstrMeta, Analysis.getPrevInstructionSequential(*InstrMeta));
  PrevInstrMeta = InstrMeta;

  // 0xDEADBEEF + 6: sub $0x18, %rsp
  InstrMeta = Analysis.getInstruction(0xDEADBEEF + 6);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(NextInstrMeta, InstrMeta);
  EXPECT_EQ(0xDEADBEEF + 6, InstrMeta->VMAddress);
  EXPECT_EQ(4u, InstrMeta->InstructionSize);
  EXPECT_TRUE(InstrMeta->Valid);

  NextInstrMeta = Analysis.getNextInstructionSequential(*InstrMeta);
  EXPECT_EQ(PrevInstrMeta, Analysis.getPrevInstructionSequential(*InstrMeta));
  PrevInstrMeta = InstrMeta;

  // 0xDEADBEEF + 10: movabs $0x4007c4, %rsi
  InstrMeta = Analysis.getInstruction(0xDEADBEEF + 10);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(NextInstrMeta, InstrMeta);
  EXPECT_EQ(0xDEADBEEF + 10, InstrMeta->VMAddress);
  EXPECT_EQ(10u, InstrMeta->InstructionSize);
  EXPECT_TRUE(InstrMeta->Valid);

  EXPECT_EQ(nullptr, Analysis.getNextInstructionSequential(*InstrMeta));
  EXPECT_EQ(PrevInstrMeta, Analysis.getPrevInstructionSequential(*InstrMeta));
  PrevInstrMeta = InstrMeta;

  // 0xDEADBEEF + 20: (bad)
  InstrMeta = Analysis.getInstruction(0xDEADBEEF + 20);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(0xDEADBEEF + 20, InstrMeta->VMAddress);
  EXPECT_EQ(1u, InstrMeta->InstructionSize);
  EXPECT_FALSE(InstrMeta->Valid);

  EXPECT_EQ(nullptr, Analysis.getNextInstructionSequential(*InstrMeta));
  EXPECT_EQ(PrevInstrMeta, Analysis.getPrevInstructionSequential(*InstrMeta));

  // 0xDEADBEEF + 21: rex.B (bad)
  InstrMeta = Analysis.getInstruction(0xDEADBEEF + 21);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(0xDEADBEEF + 21, InstrMeta->VMAddress);
  EXPECT_EQ(2u, InstrMeta->InstructionSize);
  EXPECT_FALSE(InstrMeta->Valid);

  EXPECT_EQ(nullptr, Analysis.getNextInstructionSequential(*InstrMeta));
  EXPECT_EQ(nullptr, Analysis.getPrevInstructionSequential(*InstrMeta));

  // 0xDEADBEEF + 6: (bad) {%k1}
  InstrMeta = Analysis.getInstruction(0xDEADBEEF + 23);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(0xDEADBEEF + 23, InstrMeta->VMAddress);
  EXPECT_EQ(5u, InstrMeta->InstructionSize);
  EXPECT_FALSE(InstrMeta->Valid);

  EXPECT_EQ(nullptr, Analysis.getNextInstructionSequential(*InstrMeta));
  EXPECT_EQ(nullptr, Analysis.getPrevInstructionSequential(*InstrMeta));
}

TEST_F(BasicX86FileAnalysisTest, PrevAndNextFromBadInst) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x90, // 0: nop
          0x2f, // 1: (bad)
          0x90  // 2: nop
      },
      {0xDEADBEEF, 0x0});
  const auto &BadInstrMeta = Analysis.getInstructionOrDie(0xDEADBEEF + 1);
  const auto *GoodInstrMeta =
      Analysis.getPrevInstructionSequential(BadInstrMeta);
  EXPECT_NE(nullptr, GoodInstrMeta);
  EXPECT_EQ(0xDEADBEEF, GoodInstrMeta->VMAddress);
  EXPECT_EQ(1u, GoodInstrMeta->InstructionSize);

  GoodInstrMeta = Analysis.getNextInstructionSequential(BadInstrMeta);
  EXPECT_NE(nullptr, GoodInstrMeta);
  EXPECT_EQ(0xDEADBEEF + 2, GoodInstrMeta->VMAddress);
  EXPECT_EQ(1u, GoodInstrMeta->InstructionSize);
}

TEST_F(BasicX86FileAnalysisTest, CFITrapTest) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x90,                   // 0: nop
          0xb0, 0x00,             // 1: mov $0x0, %al
          0x48, 0x89, 0xe5,       // 3: mov %rsp, %rbp
          0x48, 0x83, 0xec, 0x18, // 6: sub $0x18, %rsp
          0x48, 0xbe, 0xc4, 0x07, 0x40,
          0x00, 0x00, 0x00, 0x00, 0x00, // 10: movabs $0x4007c4, %rsi
          0x2f,                         // 20: (bad)
          0x41, 0x0e,                   // 21: rex.B (bad)
          0x62, 0x72, 0x65, 0x61, 0x6b, // 23: (bad) {%k1}
          0x0f, 0x0b                    // 28: ud2
      },
      {0xDEADBEEF, 0x0});

  EXPECT_FALSE(Analysis.isCFITrap(Analysis.getInstructionOrDie(0xDEADBEEF)));
  EXPECT_FALSE(
      Analysis.isCFITrap(Analysis.getInstructionOrDie(0xDEADBEEF + 3)));
  EXPECT_FALSE(
      Analysis.isCFITrap(Analysis.getInstructionOrDie(0xDEADBEEF + 6)));
  EXPECT_FALSE(
      Analysis.isCFITrap(Analysis.getInstructionOrDie(0xDEADBEEF + 10)));
  EXPECT_FALSE(
      Analysis.isCFITrap(Analysis.getInstructionOrDie(0xDEADBEEF + 20)));
  EXPECT_FALSE(
      Analysis.isCFITrap(Analysis.getInstructionOrDie(0xDEADBEEF + 21)));
  EXPECT_FALSE(
      Analysis.isCFITrap(Analysis.getInstructionOrDie(0xDEADBEEF + 23)));
  EXPECT_TRUE(
      Analysis.isCFITrap(Analysis.getInstructionOrDie(0xDEADBEEF + 28)));
}

TEST_F(BasicX86FileAnalysisTest, FallThroughTest) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x90,                         // 0: nop
          0xb0, 0x00,                   // 1: mov $0x0, %al
          0x2f,                         // 3: (bad)
          0x0f, 0x0b,                   // 4: ud2
          0xff, 0x20,                   // 6: jmpq *(%rax)
          0xeb, 0x00,                   // 8: jmp +0
          0xe8, 0x45, 0xfe, 0xff, 0xff, // 10: callq [some loc]
          0xff, 0x10,                   // 15: callq *(rax)
          0x75, 0x00,                   // 17: jne +0
          0xc3,                         // 19: retq
      },
      {0xDEADBEEF, 0x0});

  EXPECT_TRUE(
      Analysis.canFallThrough(Analysis.getInstructionOrDie(0xDEADBEEF)));
  EXPECT_TRUE(
      Analysis.canFallThrough(Analysis.getInstructionOrDie(0xDEADBEEF + 1)));
  EXPECT_FALSE(
      Analysis.canFallThrough(Analysis.getInstructionOrDie(0xDEADBEEF + 3)));
  EXPECT_FALSE(
      Analysis.canFallThrough(Analysis.getInstructionOrDie(0xDEADBEEF + 4)));
  EXPECT_FALSE(
      Analysis.canFallThrough(Analysis.getInstructionOrDie(0xDEADBEEF + 6)));
  EXPECT_FALSE(
      Analysis.canFallThrough(Analysis.getInstructionOrDie(0xDEADBEEF + 8)));
  EXPECT_FALSE(
      Analysis.canFallThrough(Analysis.getInstructionOrDie(0xDEADBEEF + 10)));
  EXPECT_FALSE(
      Analysis.canFallThrough(Analysis.getInstructionOrDie(0xDEADBEEF + 15)));
  EXPECT_TRUE(
      Analysis.canFallThrough(Analysis.getInstructionOrDie(0xDEADBEEF + 17)));
  EXPECT_FALSE(
      Analysis.canFallThrough(Analysis.getInstructionOrDie(0xDEADBEEF + 19)));
}

TEST_F(BasicX86FileAnalysisTest, DefiniteNextInstructionTest) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x90,                         // 0: nop
          0xb0, 0x00,                   // 1: mov $0x0, %al
          0x2f,                         // 3: (bad)
          0x0f, 0x0b,                   // 4: ud2
          0xff, 0x20,                   // 6: jmpq *(%rax)
          0xeb, 0x00,                   // 8: jmp 10 [+0]
          0xeb, 0x05,                   // 10: jmp 17 [+5]
          0xe8, 0x00, 0x00, 0x00, 0x00, // 12: callq 17 [+0]
          0xe8, 0x78, 0x56, 0x34, 0x12, // 17: callq 0x1234569f [+0x12345678]
          0xe8, 0x04, 0x00, 0x00, 0x00, // 22: callq 31 [+4]
          0xff, 0x10,                   // 27: callq *(rax)
          0x75, 0x00,                   // 29: jne 31 [+0]
          0x75, 0xe0,                   // 31: jne 1 [-32]
          0xc3,                         // 33: retq
          0xeb, 0xdd,                   // 34: jmp 1 [-35]
          0xeb, 0xdd,                   // 36: jmp 3 [-35]
          0xeb, 0xdc,                   // 38: jmp 4 [-36]
      },
      {0xDEADBEEF, 0x0});

  const auto *Current = Analysis.getInstruction(0xDEADBEEF);
  const auto *Next = Analysis.getDefiniteNextInstruction(*Current);
  EXPECT_NE(nullptr, Next);
  EXPECT_EQ(0xDEADBEEF + 1, Next->VMAddress);

  Current = Analysis.getInstruction(0xDEADBEEF + 1);
  EXPECT_EQ(nullptr, Analysis.getDefiniteNextInstruction(*Current));

  Current = Analysis.getInstruction(0xDEADBEEF + 3);
  EXPECT_EQ(nullptr, Analysis.getDefiniteNextInstruction(*Current));

  Current = Analysis.getInstruction(0xDEADBEEF + 4);
  EXPECT_EQ(nullptr, Analysis.getDefiniteNextInstruction(*Current));

  Current = Analysis.getInstruction(0xDEADBEEF + 6);
  EXPECT_EQ(nullptr, Analysis.getDefiniteNextInstruction(*Current));

  Current = Analysis.getInstruction(0xDEADBEEF + 8);
  Next = Analysis.getDefiniteNextInstruction(*Current);
  EXPECT_NE(nullptr, Next);
  EXPECT_EQ(0xDEADBEEF + 10, Next->VMAddress);

  Current = Analysis.getInstruction(0xDEADBEEF + 10);
  Next = Analysis.getDefiniteNextInstruction(*Current);
  EXPECT_NE(nullptr, Next);
  EXPECT_EQ(0xDEADBEEF + 17, Next->VMAddress);

  Current = Analysis.getInstruction(0xDEADBEEF + 12);
  Next = Analysis.getDefiniteNextInstruction(*Current);
  EXPECT_NE(nullptr, Next);
  EXPECT_EQ(0xDEADBEEF + 17, Next->VMAddress);

  Current = Analysis.getInstruction(0xDEADBEEF + 17);
  // Note, definite next instruction address is out of range and should fail.
  EXPECT_EQ(nullptr, Analysis.getDefiniteNextInstruction(*Current));
  Next = Analysis.getDefiniteNextInstruction(*Current);

  Current = Analysis.getInstruction(0xDEADBEEF + 22);
  Next = Analysis.getDefiniteNextInstruction(*Current);
  EXPECT_NE(nullptr, Next);
  EXPECT_EQ(0xDEADBEEF + 31, Next->VMAddress);

  Current = Analysis.getInstruction(0xDEADBEEF + 27);
  EXPECT_EQ(nullptr, Analysis.getDefiniteNextInstruction(*Current));
  Current = Analysis.getInstruction(0xDEADBEEF + 29);
  EXPECT_EQ(nullptr, Analysis.getDefiniteNextInstruction(*Current));
  Current = Analysis.getInstruction(0xDEADBEEF + 31);
  EXPECT_EQ(nullptr, Analysis.getDefiniteNextInstruction(*Current));
  Current = Analysis.getInstruction(0xDEADBEEF + 33);
  EXPECT_EQ(nullptr, Analysis.getDefiniteNextInstruction(*Current));

  Current = Analysis.getInstruction(0xDEADBEEF + 34);
  Next = Analysis.getDefiniteNextInstruction(*Current);
  EXPECT_NE(nullptr, Next);
  EXPECT_EQ(0xDEADBEEF + 1, Next->VMAddress);

  Current = Analysis.getInstruction(0xDEADBEEF + 36);
  EXPECT_EQ(nullptr, Analysis.getDefiniteNextInstruction(*Current));

  Current = Analysis.getInstruction(0xDEADBEEF + 38);
  Next = Analysis.getDefiniteNextInstruction(*Current);
  EXPECT_NE(nullptr, Next);
  EXPECT_EQ(0xDEADBEEF + 4, Next->VMAddress);
}

TEST_F(BasicX86FileAnalysisTest, ControlFlowXRefsTest) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x90,                         // 0: nop
          0xb0, 0x00,                   // 1: mov $0x0, %al
          0x2f,                         // 3: (bad)
          0x0f, 0x0b,                   // 4: ud2
          0xff, 0x20,                   // 6: jmpq *(%rax)
          0xeb, 0x00,                   // 8: jmp 10 [+0]
          0xeb, 0x05,                   // 10: jmp 17 [+5]
          0xe8, 0x00, 0x00, 0x00, 0x00, // 12: callq 17 [+0]
          0xe8, 0x78, 0x56, 0x34, 0x12, // 17: callq 0x1234569f [+0x12345678]
          0xe8, 0x04, 0x00, 0x00, 0x00, // 22: callq 31 [+4]
          0xff, 0x10,                   // 27: callq *(rax)
          0x75, 0x00,                   // 29: jne 31 [+0]
          0x75, 0xe0,                   // 31: jne 1 [-32]
          0xc3,                         // 33: retq
          0xeb, 0xdd,                   // 34: jmp 1 [-35]
          0xeb, 0xdd,                   // 36: jmp 3 [-35]
          0xeb, 0xdc,                   // 38: jmp 4 [-36]
      },
      {0xDEADBEEF, 0x0});
  const auto *InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF);
  std::set<const Instr *> XRefs =
      Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_TRUE(XRefs.empty());

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 1);
  XRefs = Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_THAT(XRefs, UnorderedElementsAre(
                         Field(&Instr::VMAddress, Eq(0xDEADBEEF)),
                         Field(&Instr::VMAddress, Eq(0xDEADBEEF + 31)),
                         Field(&Instr::VMAddress, Eq(0xDEADBEEF + 34))));

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 3);
  XRefs = Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_THAT(XRefs, UnorderedElementsAre(
                         Field(&Instr::VMAddress, Eq(0xDEADBEEF + 1)),
                         Field(&Instr::VMAddress, Eq(0xDEADBEEF + 36))));

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 4);
  XRefs = Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_THAT(XRefs, UnorderedElementsAre(
                         Field(&Instr::VMAddress, Eq(0xDEADBEEF + 38))));

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 6);
  EXPECT_TRUE(Analysis.getDirectControlFlowXRefs(*InstrMetaPtr).empty());

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 8);
  XRefs = Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_TRUE(Analysis.getDirectControlFlowXRefs(*InstrMetaPtr).empty());

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 10);
  XRefs = Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_THAT(XRefs, UnorderedElementsAre(
                         Field(&Instr::VMAddress, Eq(0xDEADBEEF + 8))));

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 12);
  XRefs = Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_TRUE(Analysis.getDirectControlFlowXRefs(*InstrMetaPtr).empty());

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 17);
  XRefs = Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_THAT(XRefs, UnorderedElementsAre(
                         Field(&Instr::VMAddress, Eq(0xDEADBEEF + 10)),
                         Field(&Instr::VMAddress, Eq(0xDEADBEEF + 12))));

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 22);
  XRefs = Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_TRUE(Analysis.getDirectControlFlowXRefs(*InstrMetaPtr).empty());

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 27);
  XRefs = Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_TRUE(Analysis.getDirectControlFlowXRefs(*InstrMetaPtr).empty());

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 29);
  XRefs = Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_TRUE(Analysis.getDirectControlFlowXRefs(*InstrMetaPtr).empty());

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 31);
  XRefs = Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_THAT(XRefs, UnorderedElementsAre(
                         Field(&Instr::VMAddress, Eq(0xDEADBEEF + 22)),
                         Field(&Instr::VMAddress, Eq(0xDEADBEEF + 29))));

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 33);
  XRefs = Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_THAT(XRefs, UnorderedElementsAre(
                         Field(&Instr::VMAddress, Eq(0xDEADBEEF + 31))));

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 34);
  XRefs = Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_TRUE(Analysis.getDirectControlFlowXRefs(*InstrMetaPtr).empty());

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 36);
  XRefs = Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_TRUE(Analysis.getDirectControlFlowXRefs(*InstrMetaPtr).empty());

  InstrMetaPtr = &Analysis.getInstructionOrDie(0xDEADBEEF + 38);
  XRefs = Analysis.getDirectControlFlowXRefs(*InstrMetaPtr);
  EXPECT_TRUE(Analysis.getDirectControlFlowXRefs(*InstrMetaPtr).empty());
}

TEST_F(BasicX86FileAnalysisTest, CFIProtectionInvalidTargets) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x90,       // 0: nop
          0x0f, 0x0b, // 1: ud2
          0x75, 0x00, // 3: jne 5 [+0]
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_NOT_INDIRECT_CF,
            Analysis.validateCFIProtection(Result));
  Result = GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 1, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_NOT_INDIRECT_CF,
            Analysis.validateCFIProtection(Result));
  Result = GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 3, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_NOT_INDIRECT_CF,
            Analysis.validateCFIProtection(Result));
  Result = GraphBuilder::buildFlowGraph(Analysis, {0x12345678, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_INVALID_INSTRUCTION,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicX86FileAnalysisTest, CFIProtectionBasicFallthroughToUd2) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x02, // 0: jne 4 [+2]
          0x0f, 0x0b, // 2: ud2
          0xff, 0x10, // 4: callq *(%rax)
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 4, 0x0});
  EXPECT_EQ(CFIProtectionStatus::PROTECTED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicX86FileAnalysisTest, CFIProtectionBasicJumpToUd2) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x02, // 0: jne 4 [+2]
          0xff, 0x10, // 2: callq *(%rax)
          0x0f, 0x0b, // 4: ud2
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 2, 0x0});
  EXPECT_EQ(CFIProtectionStatus::PROTECTED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicX86FileAnalysisTest, CFIProtectionDualPathUd2) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x03, // 0: jne 5 [+3]
          0x90,       // 2: nop
          0xff, 0x10, // 3: callq *(%rax)
          0x0f, 0x0b, // 5: ud2
          0x75, 0xf9, // 7: jne 2 [-7]
          0x0f, 0x0b, // 9: ud2
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 3, 0x0});
  EXPECT_EQ(CFIProtectionStatus::PROTECTED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicX86FileAnalysisTest, CFIProtectionDualPathSingleUd2) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x05, // 0: jne 7 [+5]
          0x90,       // 2: nop
          0xff, 0x10, // 3: callq *(%rax)
          0x75, 0xfb, // 5: jne 2 [-5]
          0x0f, 0x0b, // 7: ud2
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 3, 0x0});
  EXPECT_EQ(CFIProtectionStatus::PROTECTED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicX86FileAnalysisTest, CFIProtectionDualFailLimitUpwards) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x06, // 0: jne 8 [+6]
          0x90,       // 2: nop
          0x90,       // 3: nop
          0x90,       // 4: nop
          0x90,       // 5: nop
          0xff, 0x10, // 6: callq *(%rax)
          0x0f, 0x0b, // 8: ud2
      },
      {0xDEADBEEF, 0x0});
  uint64_t PrevSearchLengthForConditionalBranch =
      SearchLengthForConditionalBranch;
  SearchLengthForConditionalBranch = 2;

  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 6, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_ORPHANS,
            Analysis.validateCFIProtection(Result));

  SearchLengthForConditionalBranch = PrevSearchLengthForConditionalBranch;
}

TEST_F(BasicX86FileAnalysisTest, CFIProtectionDualFailLimitDownwards) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x02, // 0: jne 4 [+2]
          0xff, 0x10, // 2: callq *(%rax)
          0x90,       // 4: nop
          0x90,       // 5: nop
          0x90,       // 6: nop
          0x90,       // 7: nop
          0x0f, 0x0b, // 8: ud2
      },
      {0xDEADBEEF, 0x0});
  uint64_t PrevSearchLengthForUndef = SearchLengthForUndef;
  SearchLengthForUndef = 2;

  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 2, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_BAD_CONDITIONAL_BRANCH,
            Analysis.validateCFIProtection(Result));

  SearchLengthForUndef = PrevSearchLengthForUndef;
}

TEST_F(BasicX86FileAnalysisTest, CFIProtectionGoodAndBadPaths) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0xeb, 0x02, // 0: jmp 4 [+2]
          0x75, 0x02, // 2: jne 6 [+2]
          0xff, 0x10, // 4: callq *(%rax)
          0x0f, 0x0b, // 6: ud2
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 4, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_ORPHANS,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicX86FileAnalysisTest, CFIProtectionWithUnconditionalJumpInFallthrough) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x04, // 0: jne 6 [+4]
          0xeb, 0x00, // 2: jmp 4 [+0]
          0xff, 0x10, // 4: callq *(%rax)
          0x0f, 0x0b, // 6: ud2
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 4, 0x0});
  EXPECT_EQ(CFIProtectionStatus::PROTECTED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicX86FileAnalysisTest, CFIProtectionComplexExample) {
  if (!SuccessfullyInitialised)
    return;
  // See unittests/GraphBuilder.cpp::BuildFlowGraphComplexExample for this
  // graph.
  Analysis.parseSectionContents(
      {
          0x75, 0x12,                   // 0: jne 20 [+18]
          0xeb, 0x03,                   // 2: jmp 7 [+3]
          0x75, 0x10,                   // 4: jne 22 [+16]
          0x90,                         // 6: nop
          0x90,                         // 7: nop
          0x90,                         // 8: nop
          0xff, 0x10,                   // 9: callq *(%rax)
          0xeb, 0xfc,                   // 11: jmp 9 [-4]
          0x75, 0xfa,                   // 13: jne 9 [-6]
          0xe8, 0x78, 0x56, 0x34, 0x12, // 15: callq OUTOFBOUNDS [+0x12345678]
          0x90,                         // 20: nop
          0x90,                         // 21: nop
          0x0f, 0x0b,                   // 22: ud2
      },
      {0xDEADBEEF, 0x0});
  uint64_t PrevSearchLengthForUndef = SearchLengthForUndef;
  SearchLengthForUndef = 5;
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 9, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_ORPHANS,
            Analysis.validateCFIProtection(Result));
  SearchLengthForUndef = PrevSearchLengthForUndef;
}

TEST_F(BasicX86FileAnalysisTest, UndefSearchLengthOneTest) {
  Analysis.parseSectionContents(
      {
          0x77, 0x0d,                   // 0x688118: ja 0x688127 [+12]
          0x48, 0x89, 0xdf,             // 0x68811a: mov %rbx, %rdi
          0xff, 0xd0,                   // 0x68811d: callq *%rax
          0x48, 0x89, 0xdf,             // 0x68811f: mov %rbx, %rdi
          0xe8, 0x09, 0x00, 0x00, 0x00, // 0x688122: callq 0x688130
          0x0f, 0x0b,                   // 0x688127: ud2
      },
      {0x688118, 0x0});
  uint64_t PrevSearchLengthForUndef = SearchLengthForUndef;
  SearchLengthForUndef = 1;
  GraphResult Result = GraphBuilder::buildFlowGraph(Analysis, {0x68811d, 0x0});
  EXPECT_EQ(CFIProtectionStatus::PROTECTED,
            Analysis.validateCFIProtection(Result));
  SearchLengthForUndef = PrevSearchLengthForUndef;
}

TEST_F(BasicX86FileAnalysisTest, UndefSearchLengthOneTestFarAway) {
  Analysis.parseSectionContents(
      {
          0x74, 0x73,                         // 0x7759eb: je 0x775a60
          0xe9, 0x1c, 0x04, 0x00, 0x00, 0x00, // 0x7759ed: jmpq 0x775e0e
      },
      {0x7759eb, 0x0});

  Analysis.parseSectionContents(
      {
          0x0f, 0x85, 0xb2, 0x03, 0x00, 0x00, // 0x775a56: jne    0x775e0e
          0x48, 0x83, 0xc3, 0xf4, // 0x775a5c: add    $0xfffffffffffffff4,%rbx
          0x48, 0x8b, 0x7c, 0x24, 0x10, // 0x775a60: mov    0x10(%rsp),%rdi
          0x48, 0x89, 0xde,             // 0x775a65: mov    %rbx,%rsi
          0xff, 0xd1,                   // 0x775a68: callq  *%rcx
      },
      {0x775a56, 0x0});

  Analysis.parseSectionContents(
      {
          0x0f, 0x0b, // 0x775e0e: ud2
      },
      {0x775e0e, 0x0});
  uint64_t PrevSearchLengthForUndef = SearchLengthForUndef;
  SearchLengthForUndef = 1;
  GraphResult Result = GraphBuilder::buildFlowGraph(Analysis, {0x775a68, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_BAD_CONDITIONAL_BRANCH,
            Analysis.validateCFIProtection(Result));
  SearchLengthForUndef = 2;
  Result = GraphBuilder::buildFlowGraph(Analysis, {0x775a68, 0x0});
  EXPECT_EQ(CFIProtectionStatus::PROTECTED,
            Analysis.validateCFIProtection(Result));
  SearchLengthForUndef = 3;
  Result = GraphBuilder::buildFlowGraph(Analysis, {0x775a68, 0x0});
  EXPECT_EQ(CFIProtectionStatus::PROTECTED,
            Analysis.validateCFIProtection(Result));
  SearchLengthForUndef = PrevSearchLengthForUndef;
}

TEST_F(BasicX86FileAnalysisTest, CFIProtectionClobberSinglePathExplicit) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x02,                         // 0: jne 4 [+2]
          0x0f, 0x0b,                         // 2: ud2
          0x48, 0x05, 0x00, 0x00, 0x00, 0x00, // 4: add $0x0, %rax
          0xff, 0x10,                         // 10: callq *(%rax)
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 10, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_REGISTER_CLOBBERED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicX86FileAnalysisTest, CFIProtectionClobberSinglePathExplicit2) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x02,             // 0: jne 4 [+2]
          0x0f, 0x0b,             // 2: ud2
          0x48, 0x83, 0xc0, 0x00, // 4: add $0x0, %rax
          0xff, 0x10,             // 8: callq *(%rax)
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 8, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_REGISTER_CLOBBERED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicX86FileAnalysisTest, CFIProtectionClobberSinglePathImplicit) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x02,                   // 0: jne 4 [+2]
          0x0f, 0x0b,                   // 2: ud2
          0x05, 0x00, 0x00, 0x00, 0x00, // 4: add $0x0, %eax
          0xff, 0x10,                   // 9: callq *(%rax)
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 9, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_REGISTER_CLOBBERED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicX86FileAnalysisTest, CFIProtectionClobberDualPathImplicit) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x75, 0x04, // 0: jne 6 [+4]
          0x0f, 0x31, // 2: rdtsc (note: affects eax)
          0xff, 0x10, // 4: callq *(%rax)
          0x0f, 0x0b, // 6: ud2
          0x75, 0xf9, // 8: jne 2 [-7]
          0x0f, 0x0b, // 10: ud2
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 4, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_REGISTER_CLOBBERED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicAArch64FileAnalysisTest, AArch64BasicUnprotected) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x00, 0x01, 0x3f, 0xd6, // 0: blr x8
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_ORPHANS,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicAArch64FileAnalysisTest, AArch64BasicProtected) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x49, 0x00, 0x00, 0x54, // 0: b.ls 8
          0x20, 0x00, 0x20, 0xd4, // 4: brk #0x1
          0x00, 0x01, 0x3f, 0xd6, // 8: blr x8
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 8, 0x0});
  EXPECT_EQ(CFIProtectionStatus::PROTECTED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicAArch64FileAnalysisTest, AArch64ClobberBasic) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x49, 0x00, 0x00, 0x54, // 0: b.ls 8
          0x20, 0x00, 0x20, 0xd4, // 4: brk #0x1
          0x08, 0x05, 0x00, 0x91, // 8: add x8, x8, #1
          0x00, 0x01, 0x3f, 0xd6, // 12: blr x8
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 12, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_REGISTER_CLOBBERED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicAArch64FileAnalysisTest, AArch64ClobberOneLoad) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x49, 0x00, 0x00, 0x54, // 0: b.ls 8
          0x20, 0x00, 0x20, 0xd4, // 4: brk #0x1
          0x21, 0x09, 0x40, 0xf9, // 8: ldr x1, [x9,#16]
          0x20, 0x00, 0x1f, 0xd6, // 12: br x1
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 12, 0x0});
  EXPECT_EQ(CFIProtectionStatus::PROTECTED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicAArch64FileAnalysisTest, AArch64ClobberLoadAddGood) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x49, 0x00, 0x00, 0x54, // 0: b.ls 8
          0x20, 0x00, 0x20, 0xd4, // 4: brk #0x1
          0x21, 0x04, 0x00, 0x91, // 8: add x1, x1, #1
          0x21, 0x09, 0x40, 0xf9, // 12: ldr x1, [x9,#16]
          0x20, 0x00, 0x1f, 0xd6, // 16: br x1
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 16, 0x0});
  EXPECT_EQ(CFIProtectionStatus::PROTECTED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicAArch64FileAnalysisTest, AArch64ClobberLoadAddBad) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x49, 0x00, 0x00, 0x54, // 0: b.ls 8
          0x20, 0x00, 0x20, 0xd4, // 4: brk #0x1
          0x21, 0x09, 0x40, 0xf9, // 8: ldr x1, [x9,#16]
          0x21, 0x04, 0x00, 0x91, // 12: add x1, x1, #1
          0x20, 0x00, 0x1f, 0xd6, // 16: br x1
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 16, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_REGISTER_CLOBBERED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicAArch64FileAnalysisTest, AArch64ClobberLoadAddBad2) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x49, 0x00, 0x00, 0x54, // 0: b.ls 8
          0x20, 0x00, 0x20, 0xd4, // 4: brk #0x1
          0x29, 0x04, 0x00, 0x91, // 16: add x9, x1, #1
          0x21, 0x09, 0x40, 0xf9, // 12: ldr x1, [x9,#16]
          0x20, 0x00, 0x1f, 0xd6, // 16: br x1
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 16, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_REGISTER_CLOBBERED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicAArch64FileAnalysisTest, AArch64ClobberTwoLoads) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x49, 0x00, 0x00, 0x54, // 0: b.ls 8
          0x20, 0x00, 0x20, 0xd4, // 4: brk #0x1
          0x21, 0x09, 0x40, 0xf9, // 8: ldr x1, [x9,#16]
          0x21, 0x08, 0x40, 0xf9, // 12: ldr x1, [x1,#16]
          0x20, 0x00, 0x1f, 0xd6, // 16: br x1
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 16, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_REGISTER_CLOBBERED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicAArch64FileAnalysisTest, AArch64ClobberUnrelatedSecondLoad) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x49, 0x00, 0x00, 0x54, // 0: b.ls 8
          0x20, 0x00, 0x20, 0xd4, // 4: brk #0x1
          0x21, 0x09, 0x40, 0xf9, // 8: ldr x1, [x9,#16]
          0x21, 0x09, 0x40, 0xf9, // 12: ldr x1, [x9,#16]
          0x20, 0x00, 0x1f, 0xd6, // 16: br x1
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 16, 0x0});
  EXPECT_EQ(CFIProtectionStatus::PROTECTED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicAArch64FileAnalysisTest, AArch64ClobberUnrelatedLoads) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x49, 0x00, 0x00, 0x54, // 0: b.ls 8
          0x20, 0x00, 0x20, 0xd4, // 4: brk #0x1
          0x22, 0x09, 0x40, 0xf9, // 8: ldr x2, [x9,#16]
          0x22, 0x08, 0x40, 0xf9, // 12: ldr x2, [x1,#16]
          0x20, 0x00, 0x1f, 0xd6, // 16: br x1
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 16, 0x0});
  EXPECT_EQ(CFIProtectionStatus::PROTECTED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicAArch64FileAnalysisTest, AArch64GoodAndBadPaths) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0x03, 0x00, 0x00, 0x14, // 0: b 12
          0x49, 0x00, 0x00, 0x54, // 4: b.ls 8
          0x20, 0x00, 0x20, 0xd4, // 8: brk #0x1
          0x20, 0x00, 0x1f, 0xd6, // 12: br x1
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 12, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_ORPHANS,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicAArch64FileAnalysisTest, AArch64TwoPaths) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0xc9, 0x00, 0x00, 0x54, // 0: b.ls 24
          0x21, 0x08, 0x40, 0xf9, // 4: ldr x1, [x1,#16]
          0x03, 0x00, 0x00, 0x14, // 8: b 12
          0x69, 0x00, 0x00, 0x54, // 12: b.ls 12
          0x21, 0x08, 0x40, 0xf9, // 16: ldr x1, [x1,#16]
          0x20, 0x00, 0x1f, 0xd6, // 20: br x1
          0x20, 0x00, 0x20, 0xd4, // 24: brk #0x1
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 20, 0x0});
  EXPECT_EQ(CFIProtectionStatus::PROTECTED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicAArch64FileAnalysisTest, AArch64TwoPathsBadLoad1) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0xe9, 0x00, 0x00, 0x54, // 0: b.ls 28
          0x21, 0x08, 0x40, 0xf9, // 4: ldr x1, [x1,#16]
          0x21, 0x08, 0x40, 0xf9, // 8: ldr x1, [x1,#16]
          0x03, 0x00, 0x00, 0x14, // 12: b 12
          0x69, 0x00, 0x00, 0x54, // 16: b.ls 12
          0x21, 0x08, 0x40, 0xf9, // 20: ldr x1, [x1,#16]
          0x20, 0x00, 0x1f, 0xd6, // 24: br x1
          0x20, 0x00, 0x20, 0xd4, // 28: brk #0x1
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 24, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_REGISTER_CLOBBERED,
            Analysis.validateCFIProtection(Result));
}

TEST_F(BasicAArch64FileAnalysisTest, AArch64TwoPathsBadLoad2) {
  if (!SuccessfullyInitialised)
    return;
  Analysis.parseSectionContents(
      {
          0xe9, 0x00, 0x00, 0x54, // 0: b.ls 28
          0x21, 0x08, 0x40, 0xf9, // 4: ldr x1, [x1,#16]
          0x03, 0x00, 0x00, 0x14, // 8: b 12
          0x89, 0x00, 0x00, 0x54, // 12: b.ls 16
          0x21, 0x08, 0x40, 0xf9, // 16: ldr x1, [x1,#16]
          0x21, 0x08, 0x40, 0xf9, // 20: ldr x1, [x1,#16]
          0x20, 0x00, 0x1f, 0xd6, // 24: br x1
          0x20, 0x00, 0x20, 0xd4, // 28: brk #0x1
      },
      {0xDEADBEEF, 0x0});
  GraphResult Result =
      GraphBuilder::buildFlowGraph(Analysis, {0xDEADBEEF + 24, 0x0});
  EXPECT_EQ(CFIProtectionStatus::FAIL_REGISTER_CLOBBERED,
            Analysis.validateCFIProtection(Result));
}

} // anonymous namespace
} // end namespace cfi_verify
} // end namespace llvm

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();

  return RUN_ALL_TESTS();
}
