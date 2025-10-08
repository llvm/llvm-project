//===- MIR2VecTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MIR2Vec.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace mir2vec;
using VocabMap = std::map<std::string, ir2vec::Embedding>;

namespace {

TEST(MIR2VecTest, RegexExtraction) {
  // Test simple instruction names
  EXPECT_EQ(MIRVocabulary::extractBaseOpcodeName("NOP"), "NOP");
  EXPECT_EQ(MIRVocabulary::extractBaseOpcodeName("RET"), "RET");
  EXPECT_EQ(MIRVocabulary::extractBaseOpcodeName("ADD16ri"), "ADD");
  EXPECT_EQ(MIRVocabulary::extractBaseOpcodeName("ADD32rr"), "ADD");
  EXPECT_EQ(MIRVocabulary::extractBaseOpcodeName("ADD64rm"), "ADD");
  EXPECT_EQ(MIRVocabulary::extractBaseOpcodeName("MOV8ri"), "MOV");
  EXPECT_EQ(MIRVocabulary::extractBaseOpcodeName("MOV32mr"), "MOV");
  EXPECT_EQ(MIRVocabulary::extractBaseOpcodeName("PUSH64r"), "PUSH");
  EXPECT_EQ(MIRVocabulary::extractBaseOpcodeName("POP64r"), "POP");
  EXPECT_EQ(MIRVocabulary::extractBaseOpcodeName("JMP_4"), "JMP");
  EXPECT_EQ(MIRVocabulary::extractBaseOpcodeName("CALL64pcrel32"), "CALL");
  EXPECT_EQ(MIRVocabulary::extractBaseOpcodeName("SOME_INSTR_123"),
            "SOME_INSTR");
  EXPECT_EQ(MIRVocabulary::extractBaseOpcodeName("123ADD"), "ADD");
  EXPECT_FALSE(MIRVocabulary::extractBaseOpcodeName("123").empty());
}

class MIR2VecVocabTestFixture : public ::testing::Test {
protected:
  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<Module> M;
  std::unique_ptr<TargetMachine> TM;
  const TargetInstrInfo *TII;

  static void SetUpTestCase() {
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  void SetUp() override {
    Triple TargetTriple("x86_64-unknown-linux-gnu");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
    if (!T) {
      GTEST_SKIP() << "x86_64-unknown-linux-gnu target triple not available; "
                      "Skipping test";
      return;
    }

    Ctx = std::make_unique<LLVMContext>();
    M = std::make_unique<Module>("test", *Ctx);
    M->setTargetTriple(TargetTriple);

    TargetOptions Options;
    TM = std::unique_ptr<TargetMachine>(
        T->createTargetMachine(TargetTriple, "", "", Options, std::nullopt));
    if (!TM) {
      GTEST_SKIP() << "Failed to create X86 target machine; Skipping test";
      return;
    }

    // Create a dummy function to get subtarget info
    FunctionType *FT = FunctionType::get(Type::getVoidTy(*Ctx), false);
    Function *F =
        Function::Create(FT, Function::ExternalLinkage, "test", M.get());

    // Get the target instruction info
    TII = TM->getSubtargetImpl(*F)->getInstrInfo();
    if (!TII) {
      GTEST_SKIP() << "Failed to get target instruction info; Skipping test";
      return;
    }
  }
};

// Function to find an opcode by name
static int findOpcodeByName(const TargetInstrInfo *TII, StringRef Name) {
  for (unsigned Opcode = 1; Opcode < TII->getNumOpcodes(); ++Opcode) {
    if (TII->getName(Opcode) == Name)
      return Opcode;
  }
  return -1; // Not found
}

TEST_F(MIR2VecVocabTestFixture, CanonicalOpcodeMappingTest) {
  // Test that same base opcodes get same canonical indices
  std::string BaseName1 = MIRVocabulary::extractBaseOpcodeName("ADD16ri");
  std::string BaseName2 = MIRVocabulary::extractBaseOpcodeName("ADD32rr");
  std::string BaseName3 = MIRVocabulary::extractBaseOpcodeName("ADD64rm");

  EXPECT_EQ(BaseName1, BaseName2);
  EXPECT_EQ(BaseName2, BaseName3);

  // Create a MIRVocabulary instance to test the mapping
  // Use a minimal MIRVocabulary to trigger canonical mapping construction
  VocabMap VMap;
  Embedding Val = Embedding(64, 1.0f);
  VMap["ADD"] = Val;
  MIRVocabulary TestVocab(std::move(VMap), TII);

  unsigned Index1 = TestVocab.getCanonicalIndexForBaseName(BaseName1);
  unsigned Index2 = TestVocab.getCanonicalIndexForBaseName(BaseName2);
  unsigned Index3 = TestVocab.getCanonicalIndexForBaseName(BaseName3);
  EXPECT_EQ(Index1, Index2);
  EXPECT_EQ(Index2, Index3);

  // Test that different base opcodes get different canonical indices
  std::string AddBase = MIRVocabulary::extractBaseOpcodeName("ADD32rr");
  std::string SubBase = MIRVocabulary::extractBaseOpcodeName("SUB32rr");
  std::string MovBase = MIRVocabulary::extractBaseOpcodeName("MOV32rr");

  unsigned AddIndex = TestVocab.getCanonicalIndexForBaseName(AddBase);
  unsigned SubIndex = TestVocab.getCanonicalIndexForBaseName(SubBase);
  unsigned MovIndex = TestVocab.getCanonicalIndexForBaseName(MovBase);

  EXPECT_NE(AddIndex, SubIndex);
  EXPECT_NE(SubIndex, MovIndex);
  EXPECT_NE(AddIndex, MovIndex);

  // Even though we only added "ADD" to the vocab, the canonical mapping
  // should assign unique indices to all the base opcodes of the target
  // Ideally, we would check against the exact number of unique base opcodes
  // for X86, but that would make the test brittle. So we just check that
  // the number is reasonably closer to the expected number (>6880) and not just
  // opcodes that we added.
  EXPECT_GT(TestVocab.getCanonicalSize(),
            6880u); // X86 has >6880 unique base opcodes

  // Check that the embeddings for opcodes not in the vocab are zero vectors
  int Add32rrOpcode = findOpcodeByName(TII, "ADD32rr");
  ASSERT_NE(Add32rrOpcode, -1) << "ADD32rr opcode not found";
  EXPECT_TRUE(TestVocab[Add32rrOpcode].approximatelyEquals(Val));

  int Sub32rrOpcode = findOpcodeByName(TII, "SUB32rr");
  ASSERT_NE(Sub32rrOpcode, -1) << "SUB32rr opcode not found";
  EXPECT_TRUE(
      TestVocab[Sub32rrOpcode].approximatelyEquals(Embedding(64, 0.0f)));

  int Mov32rrOpcode = findOpcodeByName(TII, "MOV32rr");
  ASSERT_NE(Mov32rrOpcode, -1) << "MOV32rr opcode not found";
  EXPECT_TRUE(
      TestVocab[Mov32rrOpcode].approximatelyEquals(Embedding(64, 0.0f)));
}

// Test deterministic mapping
TEST_F(MIR2VecVocabTestFixture, DeterministicMapping) {
  // Test that the same base name always maps to the same canonical index
  std::string BaseName = "ADD";

  // Create a MIRVocabulary instance to test deterministic mapping
  // Use a minimal MIRVocabulary to trigger canonical mapping construction
  VocabMap VMap;
  VMap["ADD"] = Embedding(64, 1.0f);
  MIRVocabulary TestVocab(std::move(VMap), TII);

  unsigned Index1 = TestVocab.getCanonicalIndexForBaseName(BaseName);
  unsigned Index2 = TestVocab.getCanonicalIndexForBaseName(BaseName);
  unsigned Index3 = TestVocab.getCanonicalIndexForBaseName(BaseName);

  EXPECT_EQ(Index1, Index2);
  EXPECT_EQ(Index2, Index3);

  // Test across multiple runs
  for (int Pos = 0; Pos < 100; ++Pos) {
    unsigned Index = TestVocab.getCanonicalIndexForBaseName(BaseName);
    EXPECT_EQ(Index, Index1);
  }
}

// Test MIRVocabulary construction
TEST_F(MIR2VecVocabTestFixture, VocabularyConstruction) {
  VocabMap VMap;
  VMap["ADD"] = Embedding(128, 1.0f); // Dimension 128, all values 1.0
  VMap["SUB"] = Embedding(128, 2.0f); // Dimension 128, all values 2.0

  MIRVocabulary Vocab(std::move(VMap), TII);
  EXPECT_TRUE(Vocab.isValid());
  EXPECT_EQ(Vocab.getDimension(), 128u);

  // Test iterator - iterates over individual embeddings
  auto IT = Vocab.begin();
  EXPECT_NE(IT, Vocab.end());

  // Check first embedding exists and has correct dimension
  EXPECT_EQ((*IT).size(), 128u);

  size_t Count = 0;
  for (auto IT = Vocab.begin(); IT != Vocab.end(); ++IT) {
    EXPECT_EQ((*IT).size(), 128u);
    ++Count;
  }
  EXPECT_GT(Count, 0u);
}

} // namespace