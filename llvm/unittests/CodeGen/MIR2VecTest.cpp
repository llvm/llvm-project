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
#include "llvm/Support/raw_ostream.h"
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
  const TargetInstrInfo *TII = nullptr;

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

    // Set the data layout to match the target machine
    M->setDataLayout(TM->createDataLayout());

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

  void TearDown() override { TII = nullptr; }

  // Find an opcode by name
  int findOpcodeByName(StringRef Name) {
    for (unsigned Opcode = 1; Opcode < TII->getNumOpcodes(); ++Opcode) {
      if (TII->getName(Opcode) == Name)
        return Opcode;
    }
    return -1; // Not found
  }

  // Create a vocabulary with specific opcodes and embeddings
  Expected<MIRVocabulary>
  createTestVocab(std::initializer_list<std::pair<const char *, float>> opcodes,
                  unsigned dimension = 2) {
    assert(TII && "TargetInstrInfo not initialized");
    VocabMap VMap;
    for (const auto &[name, value] : opcodes)
      VMap[name] = Embedding(dimension, value);
    return MIRVocabulary::create(std::move(VMap), *TII);
  }
};

TEST_F(MIR2VecVocabTestFixture, CanonicalOpcodeMappingTest) {
  // Test that same base opcodes get same canonical indices
  std::string BaseName1 = MIRVocabulary::extractBaseOpcodeName("ADD16ri");
  std::string BaseName2 = MIRVocabulary::extractBaseOpcodeName("ADD32rr");
  std::string BaseName3 = MIRVocabulary::extractBaseOpcodeName("ADD64rm");

  EXPECT_EQ(BaseName1, BaseName2);
  EXPECT_EQ(BaseName2, BaseName3);

  // Create a MIRVocabulary instance to test the mapping
  // Use a minimal MIRVocabulary to trigger canonical mapping construction
  Embedding Val = Embedding(64, 1.0f);
  auto TestVocabOrErr = createTestVocab({{"ADD", 1.0f}}, 64);
  ASSERT_TRUE(static_cast<bool>(TestVocabOrErr))
      << "Failed to create vocabulary: "
      << toString(TestVocabOrErr.takeError());
  auto &TestVocab = *TestVocabOrErr;

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
  int Add32rrOpcode = findOpcodeByName("ADD32rr");
  ASSERT_NE(Add32rrOpcode, -1) << "ADD32rr opcode not found";
  EXPECT_TRUE(TestVocab[Add32rrOpcode].approximatelyEquals(Val));

  int Sub32rrOpcode = findOpcodeByName("SUB32rr");
  ASSERT_NE(Sub32rrOpcode, -1) << "SUB32rr opcode not found";
  EXPECT_TRUE(
      TestVocab[Sub32rrOpcode].approximatelyEquals(Embedding(64, 0.0f)));

  int Mov32rrOpcode = findOpcodeByName("MOV32rr");
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
  auto TestVocabOrErr = createTestVocab({{"ADD", 1.0f}}, 64);
  ASSERT_TRUE(static_cast<bool>(TestVocabOrErr))
      << "Failed to create vocabulary: "
      << toString(TestVocabOrErr.takeError());
  auto &TestVocab = *TestVocabOrErr;

  unsigned Index1 = TestVocab.getCanonicalIndexForBaseName(BaseName);
  unsigned Index2 = TestVocab.getCanonicalIndexForBaseName(BaseName);
  unsigned Index3 = TestVocab.getCanonicalIndexForBaseName(BaseName);
  EXPECT_EQ(Index2, Index3);

  // Test across multiple runs
  for (int Pos = 0; Pos < 100; ++Pos) {
    unsigned Index = TestVocab.getCanonicalIndexForBaseName(BaseName);
    EXPECT_EQ(Index, Index1);
  }
}

// Test MIRVocabulary construction
TEST_F(MIR2VecVocabTestFixture, VocabularyConstruction) {
  auto VocabOrErr = createTestVocab({{"ADD", 1.0f}, {"SUB", 2.0f}}, 128);
  ASSERT_TRUE(static_cast<bool>(VocabOrErr))
      << "Failed to create vocabulary: " << toString(VocabOrErr.takeError());
  auto &Vocab = *VocabOrErr;
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

// Test factory method with empty vocabulary
TEST_F(MIR2VecVocabTestFixture, EmptyVocabularyCreation) {
  VocabMap EmptyVMap;

  auto VocabOrErr = MIRVocabulary::create(std::move(EmptyVMap), *TII);
  EXPECT_FALSE(static_cast<bool>(VocabOrErr))
      << "Factory method should fail with empty vocabulary";

  // Consume the error
  if (!VocabOrErr) {
    auto Err = VocabOrErr.takeError();
    std::string ErrorMsg = toString(std::move(Err));
    EXPECT_FALSE(ErrorMsg.empty());
  }
}

// Fixture for embedding related tests
class MIR2VecEmbeddingTestFixture : public MIR2VecVocabTestFixture {
protected:
  std::unique_ptr<MachineModuleInfo> MMI;
  MachineFunction *MF = nullptr;

  void SetUp() override {
    MIR2VecVocabTestFixture::SetUp();

    // Create a dummy function for MachineFunction
    FunctionType *FT = FunctionType::get(Type::getVoidTy(*Ctx), false);
    Function *F =
        Function::Create(FT, Function::ExternalLinkage, "test", M.get());

    MMI = std::make_unique<MachineModuleInfo>(TM.get());
    MF = &MMI->getOrCreateMachineFunction(*F);
  }

  void TearDown() override { MIR2VecVocabTestFixture::TearDown(); }

  // Create a machine instruction
  MachineInstr *createMachineInstr(MachineBasicBlock &MBB, unsigned Opcode) {
    const MCInstrDesc &Desc = TII->get(Opcode);
    // Create instruction - operands don't affect opcode-based embeddings
    MachineInstr *MI = BuildMI(MBB, MBB.end(), DebugLoc(), Desc);
    return MI;
  }

  MachineInstr *createMachineInstr(MachineBasicBlock &MBB,
                                   const char *OpcodeName) {
    int Opcode = findOpcodeByName(OpcodeName);
    if (Opcode == -1)
      return nullptr;
    return createMachineInstr(MBB, Opcode);
  }

  void createMachineInstrs(MachineBasicBlock &MBB,
                           std::initializer_list<const char *> Opcodes) {
    for (const char *OpcodeName : Opcodes) {
      MachineInstr *MI = createMachineInstr(MBB, OpcodeName);
      ASSERT_TRUE(MI != nullptr);
    }
  }
};

// Test factory method for creating embedder
TEST_F(MIR2VecEmbeddingTestFixture, CreateSymbolicEmbedder) {
  auto VocabOrErr = MIRVocabulary::createDummyVocabForTest(*TII, 1);
  ASSERT_TRUE(static_cast<bool>(VocabOrErr))
      << "Failed to create vocabulary: " << toString(VocabOrErr.takeError());
  auto &V = *VocabOrErr;
  auto Emb = MIREmbedder::create(MIR2VecKind::Symbolic, *MF, V);
  EXPECT_NE(Emb, nullptr);
}

TEST_F(MIR2VecEmbeddingTestFixture, CreateInvalidMode) {
  auto VocabOrErr = MIRVocabulary::createDummyVocabForTest(*TII, 1);
  ASSERT_TRUE(static_cast<bool>(VocabOrErr))
      << "Failed to create vocabulary: " << toString(VocabOrErr.takeError());
  auto &V = *VocabOrErr;
  auto Result = MIREmbedder::create(static_cast<MIR2VecKind>(-1), *MF, V);
  EXPECT_FALSE(static_cast<bool>(Result));
}

// Test SymbolicMIREmbedder with simple target opcodes
TEST_F(MIR2VecEmbeddingTestFixture, TestSymbolicEmbedder) {
  // Create a test vocabulary with specific values
  auto VocabOrErr = createTestVocab(
      {
          {"NOOP", 1.0f}, // [1.0, 1.0, 1.0, 1.0]
          {"RET", 2.0f},  // [2.0, 2.0, 2.0, 2.0]
          {"TRAP", 3.0f}  // [3.0, 3.0, 3.0, 3.0]
      },
      4);
  ASSERT_TRUE(static_cast<bool>(VocabOrErr))
      << "Failed to create vocabulary: " << toString(VocabOrErr.takeError());
  auto &Vocab = *VocabOrErr;
  // Create a basic block using fixture's MF
  MachineBasicBlock *MBB = MF->CreateMachineBasicBlock();
  MF->push_back(MBB);

  // Use real X86 opcodes that should exist and not be pseudo
  auto NoopInst = createMachineInstr(*MBB, "NOOP");
  ASSERT_TRUE(NoopInst != nullptr);

  auto RetInst = createMachineInstr(*MBB, "RET64");
  ASSERT_TRUE(RetInst != nullptr);

  auto TrapInst = createMachineInstr(*MBB, "TRAP");
  ASSERT_TRUE(TrapInst != nullptr);

  // Verify these are not pseudo instructions
  ASSERT_FALSE(NoopInst->isPseudo()) << "NOOP is marked as pseudo instruction";
  ASSERT_FALSE(RetInst->isPseudo()) << "RET is marked as pseudo instruction";
  ASSERT_FALSE(TrapInst->isPseudo()) << "TRAP is marked as pseudo instruction";

  // Create embedder
  auto Embedder = SymbolicMIREmbedder::create(*MF, Vocab);
  ASSERT_TRUE(Embedder != nullptr);

  // Test instruction embeddings
  auto NoopEmb = Embedder->getMInstVector(*NoopInst);
  auto RetEmb = Embedder->getMInstVector(*RetInst);
  auto TrapEmb = Embedder->getMInstVector(*TrapInst);

  // Verify embeddings match expected values (accounting for weight scaling)
  float ExpectedWeight = mir2vec::OpcWeight; // Global weight from command line
  EXPECT_TRUE(NoopEmb.approximatelyEquals(Embedding(4, 1.0f * ExpectedWeight)));
  EXPECT_TRUE(RetEmb.approximatelyEquals(Embedding(4, 2.0f * ExpectedWeight)));
  EXPECT_TRUE(TrapEmb.approximatelyEquals(Embedding(4, 3.0f * ExpectedWeight)));

  // Test basic block embedding (should be sum of instruction embeddings)
  auto MBBVector = Embedder->getMBBVector(*MBB);

  // Expected BB vector: NOOP + RET + TRAP = [1+2+3, 1+2+3, 1+2+3, 1+2+3] *
  // weight = [6, 6, 6, 6] * weight
  Embedding ExpectedMBBVector(4, 6.0f * ExpectedWeight);
  EXPECT_TRUE(MBBVector.approximatelyEquals(ExpectedMBBVector));

  // Test function embedding (should equal MBB embedding since we have one MBB)
  auto MFuncVector = Embedder->getMFunctionVector();
  EXPECT_TRUE(MFuncVector.approximatelyEquals(ExpectedMBBVector));
}

// Test embedder with multiple basic blocks
TEST_F(MIR2VecEmbeddingTestFixture, MultipleBasicBlocks) {
  // Create a test vocabulary
  auto VocabOrErr = createTestVocab({{"NOOP", 1.0f}, {"TRAP", 2.0f}});
  ASSERT_TRUE(static_cast<bool>(VocabOrErr))
      << "Failed to create vocabulary: " << toString(VocabOrErr.takeError());
  auto &Vocab = *VocabOrErr;

  // Create two basic blocks using fixture's MF
  MachineBasicBlock *MBB1 = MF->CreateMachineBasicBlock();
  MachineBasicBlock *MBB2 = MF->CreateMachineBasicBlock();
  MF->push_back(MBB1);
  MF->push_back(MBB2);

  createMachineInstrs(*MBB1, {"NOOP", "NOOP"});
  createMachineInstr(*MBB2, "TRAP");

  // Create embedder
  auto Embedder = SymbolicMIREmbedder::create(*MF, Vocab);
  ASSERT_TRUE(Embedder != nullptr);

  // Test basic block embeddings
  auto MBB1Vector = Embedder->getMBBVector(*MBB1);
  auto MBB2Vector = Embedder->getMBBVector(*MBB2);

  float ExpectedWeight = mir2vec::OpcWeight;
  // BB1: NOOP + NOOP = 2 * ([1, 1] * weight)
  Embedding ExpectedMBB1Vector(2, 2.0f * ExpectedWeight);
  EXPECT_TRUE(MBB1Vector.approximatelyEquals(ExpectedMBB1Vector));

  // BB2: TRAP = [2, 2] * weight
  Embedding ExpectedMBB2Vector(2, 2.0f * ExpectedWeight);
  EXPECT_TRUE(MBB2Vector.approximatelyEquals(ExpectedMBB2Vector));

  // Function embedding: BB1 + BB2 = [2+2, 2+2] * weight = [4, 4] * weight
  // Function embedding should be just the first BB embedding as the second BB
  // is unreachable
  auto MFuncVector = Embedder->getMFunctionVector();
  EXPECT_TRUE(MFuncVector.approximatelyEquals(ExpectedMBB1Vector));

  // Add a branch from BB1 to BB2 to make both reachable; now function embedding
  // should be MBB1 + MBB2
  MBB1->addSuccessor(MBB2);
  auto NewMFuncVector = Embedder->getMFunctionVector(); // Recompute embeddings
  Embedding ExpectedFuncVector = MBB1Vector + MBB2Vector;
  EXPECT_TRUE(NewMFuncVector.approximatelyEquals(ExpectedFuncVector));
}

// Test embedder with empty basic block
TEST_F(MIR2VecEmbeddingTestFixture, EmptyBasicBlock) {

  // Create an empty basic block
  MachineBasicBlock *MBB = MF->CreateMachineBasicBlock();
  MF->push_back(MBB);

  // Create embedder
  auto VocabOrErr = MIRVocabulary::createDummyVocabForTest(*TII, 2);
  ASSERT_TRUE(static_cast<bool>(VocabOrErr))
      << "Failed to create vocabulary: " << toString(VocabOrErr.takeError());
  auto &V = *VocabOrErr;
  auto Embedder = SymbolicMIREmbedder::create(*MF, V);
  ASSERT_TRUE(Embedder != nullptr);

  // Test that empty BB has zero embedding
  auto MBBVector = Embedder->getMBBVector(*MBB);
  Embedding ExpectedBBVector(2, 0.0f);
  EXPECT_TRUE(MBBVector.approximatelyEquals(ExpectedBBVector));

  // Function embedding should also be zero
  auto MFuncVector = Embedder->getMFunctionVector();
  EXPECT_TRUE(MFuncVector.approximatelyEquals(ExpectedBBVector));
}

// Test embedder with opcodes not in vocabulary
TEST_F(MIR2VecEmbeddingTestFixture, UnknownOpcodes) {
  // Create a test vocabulary with limited entries
  // SUB is intentionally not included
  auto VocabOrErr = createTestVocab({{"ADD", 1.0f}});
  ASSERT_TRUE(static_cast<bool>(VocabOrErr))
      << "Failed to create vocabulary: " << toString(VocabOrErr.takeError());
  auto &Vocab = *VocabOrErr;

  // Create a basic block
  MachineBasicBlock *MBB = MF->CreateMachineBasicBlock();
  MF->push_back(MBB);

  // Find opcodes
  int AddOpcode = findOpcodeByName("ADD32rr");
  int SubOpcode = findOpcodeByName("SUB32rr");

  ASSERT_NE(AddOpcode, -1) << "ADD32rr opcode not found";
  ASSERT_NE(SubOpcode, -1) << "SUB32rr opcode not found";

  // Create instructions
  MachineInstr *AddInstr = createMachineInstr(*MBB, AddOpcode);
  MachineInstr *SubInstr = createMachineInstr(*MBB, SubOpcode);

  // Create embedder
  auto Embedder = SymbolicMIREmbedder::create(*MF, Vocab);
  ASSERT_TRUE(Embedder != nullptr);

  // Test instruction embeddings
  auto AddVector = Embedder->getMInstVector(*AddInstr);
  auto SubVector = Embedder->getMInstVector(*SubInstr);

  float ExpectedWeight = mir2vec::OpcWeight;
  // ADD should have the embedding from vocabulary
  EXPECT_TRUE(
      AddVector.approximatelyEquals(Embedding(2, 1.0f * ExpectedWeight)));

  // SUB should have zero embedding (not in vocabulary)
  EXPECT_TRUE(SubVector.approximatelyEquals(Embedding(2, 0.0f)));

  // Basic block embedding should be ADD + SUB = [1.0, 1.0] * weight + [0.0,
  // 0.0] = [1.0, 1.0] * weight
  const auto &MBBVector = Embedder->getMBBVector(*MBB);
  Embedding ExpectedBBVector(2, 1.0f * ExpectedWeight);
  EXPECT_TRUE(MBBVector.approximatelyEquals(ExpectedBBVector));
}
} // namespace
