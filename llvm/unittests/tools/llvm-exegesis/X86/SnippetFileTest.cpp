//===-- SnippetFileTest.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SnippetFile.h"

#include "LlvmState.h"
#include "TestBase.h"
#include "X86InstrInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {

void InitializeX86ExegesisTarget();

namespace {

using testing::ElementsAre;
using testing::Eq;
using testing::Property;
using testing::SizeIs;
using testing::UnorderedElementsAre;

using llvm::unittest::TempDir;

class X86SnippetFileTest : public X86TestBase {
protected:
  Expected<std::vector<BenchmarkCode>> TestCommon(StringRef Contents) {
    TempDir TestDirectory("SnippetFileTestDir", /*Unique*/ true);
    SmallString<64> Filename(TestDirectory.path());
    sys::path::append(Filename, "snippet.s");
    errs() << Filename << "-------\n";
    {
      std::error_code EC;
      raw_fd_ostream FOS(Filename, EC);
      FOS << Contents;
      EXPECT_FALSE(EC);
    }
    return readSnippets(State, Filename);
  }
};

// FIXME: Refactor these to ../Common/Matchers.h
static auto HasOpcode = [](unsigned Opcode) {
  return Property(&MCInst::getOpcode, Eq(Opcode));
};

MATCHER_P2(RegisterInitialValueIs, Reg, Val, "") {
  if (arg.Register == Reg &&
      arg.Value.getLimitedValue() == static_cast<uint64_t>(Val))
    return true;
  *result_listener << "expected: {" << Reg << ", " << Val << "} ";
  *result_listener << "actual: {" << arg.Register << ", "
                   << arg.Value.getLimitedValue() << "}";
  return false;
}

MATCHER_P3(MemoryDefinitionIs, Name, Value, Size, "") {
  if (arg.second.Value.getLimitedValue() == static_cast<uint64_t>(Value) &&
      arg.second.SizeBytes == static_cast<size_t>(Size) && arg.first == Name)
    return true;
  *result_listener << "expected: {" << Name << ", " << Value << ", " << Size
                   << "} ";
  *result_listener << "actual: {" << arg.first << ", "
                   << arg.second.Value.getLimitedValue() << ", "
                   << arg.second.SizeBytes << "}";
  return false;
}

MATCHER_P2(MemoryMappingIs, Address, Name, "") {
  if (arg.Address == Address && arg.MemoryValueName == Name)
    return true;
  *result_listener << "expected: {" << Address << ", " << Name << "} ";
  *result_listener << "actual: {" << arg.Address << ", " << arg.MemoryValueName
                   << "}";
  return false;
}

TEST_F(X86SnippetFileTest, Works) {
  auto Snippets = TestCommon(R"(
    # LLVM-EXEGESIS-DEFREG RAX 0f
    # LLVM-EXEGESIS-DEFREG SIL 0
    # LLVM-EXEGESIS-LIVEIN RDI
    # LLVM-EXEGESIS-LIVEIN DL
    incq %rax
  )");
  EXPECT_FALSE((bool)Snippets.takeError());
  ASSERT_THAT(*Snippets, SizeIs(1));
  const auto &Snippet = (*Snippets)[0];
  ASSERT_THAT(Snippet.Key.Instructions, ElementsAre(HasOpcode(X86::INC64r)));
  ASSERT_THAT(Snippet.Key.RegisterInitialValues,
              ElementsAre(RegisterInitialValueIs(X86::RAX, 15),
                          RegisterInitialValueIs(X86::SIL, 0)));
  ASSERT_THAT(Snippet.LiveIns, ElementsAre(X86::RDI, X86::DL));
}

TEST_F(X86SnippetFileTest, BadDefregParam) {
  auto Error = TestCommon(R"(
    # LLVM-EXEGESIS-DEFREG DOESNOEXIST 0
    incq %rax
  )")
                   .takeError();
  EXPECT_TRUE((bool)Error);
  consumeError(std::move(Error));
}

TEST_F(X86SnippetFileTest, NoDefregValue) {
  auto Error = TestCommon(R"(
    # LLVM-EXEGESIS-DEFREG RAX
    incq %rax
  )")
                   .takeError();
  EXPECT_TRUE((bool)Error);
  consumeError(std::move(Error));
}

TEST_F(X86SnippetFileTest, MissingParam) {
  auto Error = TestCommon(R"(
    # LLVM-EXEGESIS-LIVEIN
    incq %rax
  )")
                   .takeError();
  EXPECT_TRUE((bool)Error);
  consumeError(std::move(Error));
}

TEST_F(X86SnippetFileTest, NoAsmStreamer) {
  auto Snippets = TestCommon(R"(
    .cv_fpo_proc foo 4
  )");
  EXPECT_FALSE((bool)Snippets.takeError());
}

TEST_F(X86SnippetFileTest, MemoryDefinitionTestSingleDef) {
  auto Snippets = TestCommon(R"(
    # LLVM-EXEGESIS-MEM-DEF test1 4096 ff
    # LLVM-EXEGESIS-MEM-MAP test1 65536
    # LLVM-EXEGESIS-MEM-MAP test1 131072
    movq $8192, %r10
    movq (%r10), %r11
  )");
  EXPECT_FALSE((bool)Snippets.takeError());
  ASSERT_THAT(*Snippets, SizeIs(1));
  const auto &Snippet = (*Snippets)[0];
  ASSERT_THAT(Snippet.Key.MemoryValues,
              UnorderedElementsAre(MemoryDefinitionIs("test1", 255, 4096)));
  ASSERT_THAT(Snippet.Key.MemoryMappings,
              ElementsAre(MemoryMappingIs(65536, "test1"),
                          MemoryMappingIs(131072, "test1")));
}

TEST_F(X86SnippetFileTest, MemoryDefinitionsTestTwoDef) {
  auto Snippets = TestCommon(R"(
    # LLVM-EXEGESIS-MEM-DEF test1 4096 ff
    # LLVM-EXEGESIS-MEM-DEF test2 4096 100
    # LLVM-EXEGESIS-MEM-MAP test1 65536
    # LLVM-EXEGESIS-MEM-MAP test2 131072
    movq $8192, %r10
    movq (%r10), %r11
  )");
  EXPECT_FALSE((bool)Snippets.takeError());
  ASSERT_THAT(*Snippets, SizeIs(1));
  const auto &Snippet = (*Snippets)[0];
  ASSERT_THAT(Snippet.Key.MemoryValues,
              UnorderedElementsAre(MemoryDefinitionIs("test1", 255, 4096),
                                   MemoryDefinitionIs("test2", 256, 4096)));
  ASSERT_THAT(Snippet.Key.MemoryMappings,
              ElementsAre(MemoryMappingIs(65536, "test1"),
                          MemoryMappingIs(131072, "test2")));
}

TEST_F(X86SnippetFileTest, MemoryDefinitionMissingParameter) {
  auto Error = TestCommon(R"(
    # LLVM-EXEGESIS-MEM-DEF test1 4096
  )")
                   .takeError();
  EXPECT_TRUE((bool)Error);
  consumeError(std::move(Error));
}

TEST_F(X86SnippetFileTest, MemoryMappingMissingParameters) {
  auto Error = TestCommon(R"(
    # LLVM-EXEGESIS-MEM-MAP test1
  )")
                   .takeError();
  EXPECT_TRUE((bool)Error);
  consumeError(std::move(Error));
}

TEST_F(X86SnippetFileTest, MemoryMappingNoDefinition) {
  auto Error = TestCommon(R"(
    # LLVM-EXEGESIS-MEM-MAP test1 65536
  )")
                   .takeError();
  EXPECT_TRUE((bool)Error);
  consumeError(std::move(Error));
}

TEST_F(X86SnippetFileTest, SnippetAddress) {
  auto Snippets = TestCommon(R"(
    # LLVM-EXEGESIS-SNIPPET-ADDRESS 10000
  )");
  ASSERT_TRUE(static_cast<bool>(Snippets));
  EXPECT_THAT(*Snippets, SizeIs(1));
  const auto &Snippet = (*Snippets)[0];
  EXPECT_EQ(Snippet.Key.SnippetAddress, 0x10000);
}

TEST_F(X86SnippetFileTest, LoopRegister) {
  auto Snippets = TestCommon(R"(
    # LLVM-EXEGESIS-LOOP-REGISTER R11
  )");
  ASSERT_TRUE(static_cast<bool>(Snippets));
  EXPECT_THAT(*Snippets, SizeIs(1));
  const auto &Snippet = (*Snippets)[0];
  EXPECT_EQ(Snippet.Key.LoopRegister, X86::R11);
}

TEST_F(X86SnippetFileTest, LoopRegisterInvalidRegister) {
  auto Error = TestCommon(R"(
    # LLVM-EXEGESIS-LOOP-REGISTER INVALID
  )")
                   .takeError();
  EXPECT_TRUE(static_cast<bool>(Error));
  consumeError(std::move(Error));
}

} // namespace
} // namespace exegesis
} // namespace llvm
