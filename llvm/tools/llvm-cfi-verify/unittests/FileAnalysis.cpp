//===- llvm/tools/llvm-cfi-verify/unittests/FileAnalysis.cpp --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../FileAnalysis.h"
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

namespace llvm {
namespace cfi_verify {
namespace {
class ELFx86TestFileAnalysis : public FileAnalysis {
public:
  ELFx86TestFileAnalysis()
      : FileAnalysis(Triple("x86_64--"), SubtargetFeatures()) {}

  // Expose this method publicly for testing.
  void parseSectionContents(ArrayRef<uint8_t> SectionBytes,
                            uint64_t SectionAddress) {
    FileAnalysis::parseSectionContents(SectionBytes, SectionAddress);
  }

  Error initialiseDisassemblyMembers() {
    return FileAnalysis::initialiseDisassemblyMembers();
  }
};

class BasicFileAnalysisTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    if (Verifier.initialiseDisassemblyMembers()) {
      FAIL() << "Failed to initialise FileAnalysis.";
    }
  }

  ELFx86TestFileAnalysis Verifier;
};

TEST_F(BasicFileAnalysisTest, BasicDisassemblyTraversalTest) {
  Verifier.parseSectionContents(
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
      0xDEADBEEF);

  EXPECT_EQ(nullptr, Verifier.getInstruction(0x0));
  EXPECT_EQ(nullptr, Verifier.getInstruction(0x1000));

  // 0xDEADBEEF: nop
  const auto *InstrMeta = Verifier.getInstruction(0xDEADBEEF);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(0xDEADBEEF, InstrMeta->VMAddress);
  EXPECT_EQ(1u, InstrMeta->InstructionSize);
  EXPECT_TRUE(InstrMeta->Valid);

  const auto *NextInstrMeta = Verifier.getNextInstructionSequential(*InstrMeta);
  EXPECT_EQ(nullptr, Verifier.getPrevInstructionSequential(*InstrMeta));
  const auto *PrevInstrMeta = InstrMeta;

  // 0xDEADBEEF + 1: mov $0x0, %al
  InstrMeta = Verifier.getInstruction(0xDEADBEEF + 1);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(NextInstrMeta, InstrMeta);
  EXPECT_EQ(0xDEADBEEF + 1, InstrMeta->VMAddress);
  EXPECT_EQ(2u, InstrMeta->InstructionSize);
  EXPECT_TRUE(InstrMeta->Valid);

  NextInstrMeta = Verifier.getNextInstructionSequential(*InstrMeta);
  EXPECT_EQ(PrevInstrMeta, Verifier.getPrevInstructionSequential(*InstrMeta));
  PrevInstrMeta = InstrMeta;

  // 0xDEADBEEF + 3: mov %rsp, %rbp
  InstrMeta = Verifier.getInstruction(0xDEADBEEF + 3);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(NextInstrMeta, InstrMeta);
  EXPECT_EQ(0xDEADBEEF + 3, InstrMeta->VMAddress);
  EXPECT_EQ(3u, InstrMeta->InstructionSize);
  EXPECT_TRUE(InstrMeta->Valid);

  NextInstrMeta = Verifier.getNextInstructionSequential(*InstrMeta);
  EXPECT_EQ(PrevInstrMeta, Verifier.getPrevInstructionSequential(*InstrMeta));
  PrevInstrMeta = InstrMeta;

  // 0xDEADBEEF + 6: sub $0x18, %rsp
  InstrMeta = Verifier.getInstruction(0xDEADBEEF + 6);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(NextInstrMeta, InstrMeta);
  EXPECT_EQ(0xDEADBEEF + 6, InstrMeta->VMAddress);
  EXPECT_EQ(4u, InstrMeta->InstructionSize);
  EXPECT_TRUE(InstrMeta->Valid);

  NextInstrMeta = Verifier.getNextInstructionSequential(*InstrMeta);
  EXPECT_EQ(PrevInstrMeta, Verifier.getPrevInstructionSequential(*InstrMeta));
  PrevInstrMeta = InstrMeta;

  // 0xDEADBEEF + 10: movabs $0x4007c4, %rsi
  InstrMeta = Verifier.getInstruction(0xDEADBEEF + 10);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(NextInstrMeta, InstrMeta);
  EXPECT_EQ(0xDEADBEEF + 10, InstrMeta->VMAddress);
  EXPECT_EQ(10u, InstrMeta->InstructionSize);
  EXPECT_TRUE(InstrMeta->Valid);

  EXPECT_EQ(nullptr, Verifier.getNextInstructionSequential(*InstrMeta));
  EXPECT_EQ(PrevInstrMeta, Verifier.getPrevInstructionSequential(*InstrMeta));
  PrevInstrMeta = InstrMeta;

  // 0xDEADBEEF + 20: (bad)
  InstrMeta = Verifier.getInstruction(0xDEADBEEF + 20);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(0xDEADBEEF + 20, InstrMeta->VMAddress);
  EXPECT_EQ(1u, InstrMeta->InstructionSize);
  EXPECT_FALSE(InstrMeta->Valid);

  EXPECT_EQ(nullptr, Verifier.getNextInstructionSequential(*InstrMeta));
  EXPECT_EQ(PrevInstrMeta, Verifier.getPrevInstructionSequential(*InstrMeta));

  // 0xDEADBEEF + 21: rex.B (bad)
  InstrMeta = Verifier.getInstruction(0xDEADBEEF + 21);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(0xDEADBEEF + 21, InstrMeta->VMAddress);
  EXPECT_EQ(2u, InstrMeta->InstructionSize);
  EXPECT_FALSE(InstrMeta->Valid);

  EXPECT_EQ(nullptr, Verifier.getNextInstructionSequential(*InstrMeta));
  EXPECT_EQ(nullptr, Verifier.getPrevInstructionSequential(*InstrMeta));

  // 0xDEADBEEF + 6: (bad) {%k1}
  InstrMeta = Verifier.getInstruction(0xDEADBEEF + 23);
  EXPECT_NE(nullptr, InstrMeta);
  EXPECT_EQ(0xDEADBEEF + 23, InstrMeta->VMAddress);
  EXPECT_EQ(5u, InstrMeta->InstructionSize);
  EXPECT_FALSE(InstrMeta->Valid);

  EXPECT_EQ(nullptr, Verifier.getNextInstructionSequential(*InstrMeta));
  EXPECT_EQ(nullptr, Verifier.getPrevInstructionSequential(*InstrMeta));
}

TEST_F(BasicFileAnalysisTest, PrevAndNextFromBadInst) {
  Verifier.parseSectionContents(
      {
          0x90, // 0: nop
          0x2f, // 1: (bad)
          0x90  // 2: nop
      },
      0xDEADBEEF);
  const auto &BadInstrMeta = Verifier.getInstructionOrDie(0xDEADBEEF + 1);
  const auto *GoodInstrMeta =
      Verifier.getPrevInstructionSequential(BadInstrMeta);
  EXPECT_NE(nullptr, GoodInstrMeta);
  EXPECT_EQ(0xDEADBEEF, GoodInstrMeta->VMAddress);
  EXPECT_EQ(1u, GoodInstrMeta->InstructionSize);

  GoodInstrMeta = Verifier.getNextInstructionSequential(BadInstrMeta);
  EXPECT_NE(nullptr, GoodInstrMeta);
  EXPECT_EQ(0xDEADBEEF + 2, GoodInstrMeta->VMAddress);
  EXPECT_EQ(1u, GoodInstrMeta->InstructionSize);
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
