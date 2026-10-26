//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandReturnObject.h"

#include "lldb/Host/common/DiagnosticsRendering.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <system_error>

using namespace lldb;
using namespace lldb_private;

namespace {
/// A DiagnosticError with no structured DiagnosticDetail entries, used to
/// exercise the plain-message fallback in CommandReturnObject::SetError.
class NoDetailsDiagnosticError
    : public llvm::ErrorInfo<NoDetailsDiagnosticError, DiagnosticError> {
public:
  static char ID;

  explicit NoDetailsDiagnosticError(std::string message)
      : ErrorInfo(std::make_error_code(std::errc::invalid_argument)),
        m_message(std::move(message)) {}

  std::string message() const override { return m_message; }
  llvm::ArrayRef<DiagnosticDetail> GetDetails() const override { return {}; }
  std::unique_ptr<CloneableError> Clone() const override {
    return std::make_unique<NoDetailsDiagnosticError>(m_message);
  }

private:
  std::string m_message;
};
char NoDetailsDiagnosticError::ID;
} // namespace

TEST(CommandReturnObjectTest, DefaultStatusIsInvalid) {
  CommandReturnObject result(/*colors=*/false);
  EXPECT_EQ(result.GetStatus(), eReturnStatusInvalid);
}

TEST(CommandReturnObjectTest, SetStatusUpdatesStatus) {
  CommandReturnObject result(false);
  result.SetStatus(eReturnStatusSuccessFinishResult);
  EXPECT_EQ(result.GetStatus(), eReturnStatusSuccessFinishResult);
}

TEST(CommandReturnObjectTest, AppendErrorSetsFailed) {
  CommandReturnObject result(false);
  result.AppendError("boom");
  EXPECT_EQ(result.GetStatus(), eReturnStatusFailed);
}

TEST(CommandReturnObjectTest, ClearResetsToInvalid) {
  CommandReturnObject result(false);
  result.SetStatus(eReturnStatusSuccessFinishResult);
  ASSERT_EQ(result.GetStatus(), eReturnStatusSuccessFinishResult);
  result.Clear();
  EXPECT_EQ(result.GetStatus(), eReturnStatusInvalid);
}

TEST(CommandReturnObjectTest, SetErrorFromDiagnosticErrorWithoutDetails) {
  CommandReturnObject result(false);
  result.SetError(llvm::make_error<NoDetailsDiagnosticError>("boom"));
  EXPECT_EQ(result.GetStatus(), eReturnStatusFailed);
  EXPECT_NE(result.GetErrorString().find("boom"), std::string::npos);
}
