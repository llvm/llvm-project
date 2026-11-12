//===-- LineEditor.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"

using namespace llvm;

class LineEditorTest : public testing::Test {
public:
  SmallString<64> HistPath;
  LineEditor *LE;

  LineEditorTest() {
    init();
  }

  void init() {
    sys::fs::createTemporaryFile("temp", "history", HistPath);
    ASSERT_FALSE(HistPath.empty());
    LE = new LineEditor("test", HistPath);
  }

  ~LineEditorTest() override {
    delete LE;
    sys::fs::remove(HistPath.str());
  }
};

TEST_F(LineEditorTest, HistorySaveLoad) {
  LE->saveHistory();
  LE->loadHistory();
}

struct TestListCompleter {
  std::vector<LineEditor::Completion> Completions;

  TestListCompleter(const std::vector<LineEditor::Completion> &Completions)
      : Completions(Completions) {}

  std::vector<LineEditor::Completion> operator()(StringRef Buffer,
                                                 size_t Pos) const {
    EXPECT_TRUE(Buffer.empty());
    EXPECT_EQ(0u, Pos);
    return Completions;
  }
};

TEST_F(LineEditorTest, ListCompleters) {
  std::vector<LineEditor::Completion> Comps;

  Comps.push_back(LineEditor::Completion("foo", "int foo()"));
  LE->setListCompleter(TestListCompleter(Comps));
  LineEditor::CompletionAction CA = LE->getCompletionAction("", 0);
  EXPECT_EQ(LineEditor::CompletionAction::AK_Insert, CA.Kind);
  EXPECT_EQ("foo", CA.Text);

  Comps.push_back(LineEditor::Completion("bar", "int bar()"));
  LE->setListCompleter(TestListCompleter(Comps));
  CA = LE->getCompletionAction("", 0);
  EXPECT_EQ(LineEditor::CompletionAction::AK_ShowCompletions, CA.Kind);
  ASSERT_EQ(2u, CA.Completions.size());
  ASSERT_EQ("int foo()", CA.Completions[0]);
  ASSERT_EQ("int bar()", CA.Completions[1]);

  Comps.clear();
  Comps.push_back(LineEditor::Completion("fee", "int fee()"));
  Comps.push_back(LineEditor::Completion("fi", "int fi()"));
  Comps.push_back(LineEditor::Completion("foe", "int foe()"));
  Comps.push_back(LineEditor::Completion("fum", "int fum()"));
  LE->setListCompleter(TestListCompleter(Comps));
  CA = LE->getCompletionAction("", 0);
  EXPECT_EQ(LineEditor::CompletionAction::AK_Insert, CA.Kind);
  EXPECT_EQ("f", CA.Text);
}
