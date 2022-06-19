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

using testing::AllOf;
using testing::ElementsAre;
using testing::Eq;
using testing::Field;
using testing::Property;
using testing::SizeIs;

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

} // namespace
} // namespace exegesis
} // namespace llvm
