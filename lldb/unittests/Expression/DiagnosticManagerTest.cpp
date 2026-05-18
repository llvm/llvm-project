//===-- DiagnosticManagerTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/DiagnosticManager.h"
#include "gtest/gtest.h"

using namespace lldb_private;

static const uint32_t custom_diag_id = 42;

namespace {
class FixItDiag : public Diagnostic {
  bool m_has_fixits;

public:
  FixItDiag(llvm::StringRef msg, bool has_fixits)
      : Diagnostic(DiagnosticOrigin::eDiagnosticOriginLLDB, custom_diag_id,
                   DiagnosticDetail{{}, lldb::eSeverityError, msg.str(), {}}),
        m_has_fixits(has_fixits) {}
  bool HasFixIts() const override { return m_has_fixits; }
};
} // namespace

namespace {
class TextDiag : public Diagnostic {
public:
  TextDiag(llvm::StringRef msg, lldb::Severity severity)
      : Diagnostic(DiagnosticOrigin::eDiagnosticOriginLLDB, custom_diag_id,
                   DiagnosticDetail{{}, severity, msg.str(), msg.str()}) {}
};
} // namespace

TEST(DiagnosticManagerTest, AddDiagnostic) {
  DiagnosticManager mgr;
  EXPECT_EQ(0U, mgr.Diagnostics().size());

  std::string msg = "foo bar has happened";
  lldb::Severity severity = lldb::eSeverityError;
  DiagnosticOrigin origin = DiagnosticOrigin::eDiagnosticOriginLLDB;
  auto diag = std::make_unique<Diagnostic>(
      origin, custom_diag_id, DiagnosticDetail{{}, severity, msg, {}});
  mgr.AddDiagnostic(std::move(diag));
  EXPECT_EQ(1U, mgr.Diagnostics().size());
  const Diagnostic *got = mgr.Diagnostics().front().get();
  EXPECT_EQ(DiagnosticOrigin::eDiagnosticOriginLLDB, got->getKind());
  EXPECT_EQ(msg, got->GetMessage());
  EXPECT_EQ(severity, got->GetSeverity());
  EXPECT_EQ(custom_diag_id, got->GetCompilerID());
  EXPECT_EQ(false, got->HasFixIts());
}

TEST(DiagnosticManagerTest, HasFixits) {
  DiagnosticManager mgr;
  // By default we shouldn't have any fixits.
  EXPECT_FALSE(mgr.HasFixIts());
  // Adding a diag without fixits shouldn't make HasFixIts return true.
  mgr.AddDiagnostic(std::make_unique<FixItDiag>("no fixit", false));
  EXPECT_FALSE(mgr.HasFixIts());
  // Adding a diag with fixits will mark the manager as containing fixits.
  mgr.AddDiagnostic(std::make_unique<FixItDiag>("fixit", true));
  EXPECT_TRUE(mgr.HasFixIts());
  // Adding another diag without fixit shouldn't make it return false.
  mgr.AddDiagnostic(std::make_unique<FixItDiag>("no fixit", false));
  EXPECT_TRUE(mgr.HasFixIts());
  // Adding a diag with fixits. The manager should still return true.
  mgr.AddDiagnostic(std::make_unique<FixItDiag>("fixit", true));
  EXPECT_TRUE(mgr.HasFixIts());
}

static std::string toString(DiagnosticManager &mgr) {
  // The error code doesn't really matter since we just convert the
  // diagnostics to a string.
  auto result = lldb::eExpressionCompleted;
  return llvm::toString(mgr.GetAsError(result));
}

TEST(DiagnosticManagerTest, GetStringNoDiags) {
  DiagnosticManager mgr;
  EXPECT_EQ("", toString(mgr));
  std::unique_ptr<Diagnostic> empty;
  mgr.AddDiagnostic(std::move(empty));
  EXPECT_EQ("", toString(mgr));
}

TEST(DiagnosticManagerTest, GetStringBasic) {
  DiagnosticManager mgr;
  mgr.AddDiagnostic(std::make_unique<TextDiag>("abc", lldb::eSeverityError));
  EXPECT_EQ("error: abc\n", toString(mgr));
}

TEST(DiagnosticManagerTest, GetStringMultiline) {
  DiagnosticManager mgr;

  // Multiline diagnostics should only get one severity label.
  mgr.AddDiagnostic(std::make_unique<TextDiag>("b\nc", lldb::eSeverityError));
  EXPECT_EQ("error: b\nc\n", toString(mgr));
}

TEST(DiagnosticManagerTest, GetStringMultipleDiags) {
  DiagnosticManager mgr;
  mgr.AddDiagnostic(std::make_unique<TextDiag>("abc", lldb::eSeverityError));
  EXPECT_EQ("error: abc\n", toString(mgr));
  mgr.AddDiagnostic(std::make_unique<TextDiag>("def", lldb::eSeverityError));
  EXPECT_EQ("error: abc\nerror: def\n", toString(mgr));
}

TEST(DiagnosticManagerTest, GetStringSeverityLabels) {
  DiagnosticManager mgr;

  // Different severities should cause different labels.
  mgr.AddDiagnostic(std::make_unique<TextDiag>("foo", lldb::eSeverityError));
  mgr.AddDiagnostic(std::make_unique<TextDiag>("bar", lldb::eSeverityWarning));
  // Remarks have no labels.
  mgr.AddDiagnostic(std::make_unique<TextDiag>("baz", lldb::eSeverityInfo));
  EXPECT_EQ("error: foo\nwarning: bar\nbaz\n", toString(mgr));
}

TEST(DiagnosticManagerTest, GetStringPreserveOrder) {
  DiagnosticManager mgr;

  // Make sure we preserve the diagnostic order and do not sort them in any way.
  mgr.AddDiagnostic(std::make_unique<TextDiag>("baz", lldb::eSeverityInfo));
  mgr.AddDiagnostic(std::make_unique<TextDiag>("bar", lldb::eSeverityWarning));
  mgr.AddDiagnostic(std::make_unique<TextDiag>("foo", lldb::eSeverityError));
  EXPECT_EQ("baz\nwarning: bar\nerror: foo\n", toString(mgr));
}

TEST(DiagnosticManagerTest, AppendMessageNoDiag) {
  DiagnosticManager mgr;

  // FIXME: This *really* should not just fail silently.
  mgr.AppendMessageToDiagnostic("no diag has been pushed yet");
  EXPECT_EQ(0U, mgr.Diagnostics().size());
}

TEST(DiagnosticManagerTest, AppendMessageAttachToLastDiag) {
  DiagnosticManager mgr;

  mgr.AddDiagnostic(std::make_unique<TextDiag>("foo", lldb::eSeverityError));
  mgr.AddDiagnostic(std::make_unique<TextDiag>("bar", lldb::eSeverityError));
  // This should append to 'bar' and not to 'foo'.
  mgr.AppendMessageToDiagnostic("message text");

  EXPECT_EQ("error: foo\nerror: bar\nmessage text\n", toString(mgr));
}

TEST(DiagnosticManagerTest, AppendMessageSubsequentDiags) {
  DiagnosticManager mgr;

  mgr.AddDiagnostic(std::make_unique<TextDiag>("bar", lldb::eSeverityError));
  mgr.AppendMessageToDiagnostic("message text");
  // Pushing another diag after the message should work fine.
  mgr.AddDiagnostic(std::make_unique<TextDiag>("foo", lldb::eSeverityError));

  EXPECT_EQ("error: bar\nmessage text\nerror: foo\n", toString(mgr));
}

TEST(DiagnosticManagerTest, PutString) {
  DiagnosticManager mgr;

  mgr.PutString(lldb::eSeverityError, "foo");
  EXPECT_EQ(1U, mgr.Diagnostics().size());
  EXPECT_EQ(eDiagnosticOriginLLDB, mgr.Diagnostics().front()->getKind());
  EXPECT_EQ("error: foo\n", toString(mgr));
}

TEST(DiagnosticManagerTest, PutStringMultiple) {
  DiagnosticManager mgr;

  // Multiple PutString should behave like multiple diagnostics.
  mgr.PutString(lldb::eSeverityError, "foo");
  mgr.PutString(lldb::eSeverityError, "bar");
  EXPECT_EQ(2U, mgr.Diagnostics().size());
  EXPECT_EQ("error: foo\nerror: bar\n", toString(mgr));
}

TEST(DiagnosticManagerTest, PutStringSeverities) {
  DiagnosticManager mgr;

  // Multiple PutString with different severities should behave like we
  // created multiple diagnostics.
  mgr.PutString(lldb::eSeverityError, "foo");
  mgr.PutString(lldb::eSeverityWarning, "bar");
  EXPECT_EQ(2U, mgr.Diagnostics().size());
  EXPECT_EQ("error: foo\nwarning: bar\n", toString(mgr));
}

TEST(DiagnosticManagerTest, FixedExpression) {
  DiagnosticManager mgr;

  // By default there should be no fixed expression.
  EXPECT_EQ("", mgr.GetFixedExpression());

  // Setting the fixed expression should change it.
  mgr.SetFixedExpression("foo");
  EXPECT_EQ("foo", mgr.GetFixedExpression());

  // Setting the fixed expression again should also change it.
  mgr.SetFixedExpression("bar");
  EXPECT_EQ("bar", mgr.GetFixedExpression());
}

TEST(DiagnosticManagerTest, StatusConversion) {
  DiagnosticManager mgr;
  mgr.AddDiagnostic(std::make_unique<TextDiag>("abc", lldb::eSeverityError));
  mgr.AddDiagnostic(std::make_unique<TextDiag>("def", lldb::eSeverityWarning));
  Status status =
      Status::FromError(mgr.GetAsError(lldb::eExpressionParseError));
  EXPECT_EQ(std::string("error: abc\nwarning: def\n"),
            std::string(status.AsCString()));
}
