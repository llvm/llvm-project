#include "lldb/Utility/DiagnosticsRendering.h"
#include "lldb/Utility/StreamString.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;
using llvm::StringRef;
namespace {
class ErrorDisplayTest : public ::testing::Test {};

std::string Render(std::vector<DiagnosticDetail> details) {
  StreamString stream;
  RenderDiagnosticDetails(stream, 0, true, details);
  return stream.GetData();
}
} // namespace

TEST_F(ErrorDisplayTest, RenderStatus) {
  using SourceLocation = DiagnosticDetail::SourceLocation;
  {
    SourceLocation inline_loc;
    inline_loc.in_user_input = true;
    std::string result =
        Render({DiagnosticDetail{inline_loc, eSeverityError, "foo", ""}});
    ASSERT_TRUE(StringRef(result).contains("error:"));
    ASSERT_TRUE(StringRef(result).contains("foo"));
  }

  {
    // Test that diagnostics on the same column can be handled and all
    // three errors are diagnosed.
    SourceLocation loc1 = {FileSpec{"a.c"}, 13, 11, 0, false, true};
    SourceLocation loc2 = {FileSpec{"a.c"}, 13, 13, 0, false, true};
    std::string result =
        Render({DiagnosticDetail{loc1, eSeverityError, "1", "1"},
                DiagnosticDetail{loc1, eSeverityError, "2", "2"},
                DiagnosticDetail{loc2, eSeverityError, "3", "3"}});
    ASSERT_TRUE(StringRef(result).contains("error: 1"));
    ASSERT_TRUE(StringRef(result).contains("error: 2"));
    ASSERT_TRUE(StringRef(result).contains("error: 3"));
  }
  {
    // Test that diagnostics in reverse order are emitted correctly.
    SourceLocation loc1 = {FileSpec{"a.c"}, 1, 20, 0, false, true};
    SourceLocation loc2 = {FileSpec{"a.c"}, 2, 10, 0, false, true};
    std::string result =
        Render({DiagnosticDetail{loc2, eSeverityError, "X", "X"},
                DiagnosticDetail{loc1, eSeverityError, "Y", "Y"}});
    ASSERT_LT(StringRef(result).find("Y"), StringRef(result).find("X"));
  }
  {
    // Test that diagnostics in reverse order are emitted correctly.
    SourceLocation loc1 = {FileSpec{"a.c"}, 2, 10, 0, false, true};
    SourceLocation loc2 = {FileSpec{"a.c"}, 1, 20, 0, false, true};
    std::string result =
        Render({DiagnosticDetail{loc2, eSeverityError, "X", "X"},
                DiagnosticDetail{loc1, eSeverityError, "Y", "Y"}});
    ASSERT_LT(StringRef(result).find("Y"), StringRef(result).find("X"));
  }
  {
    // Test that range diagnostics are emitted correctly.
    SourceLocation loc1 = {FileSpec{"a.c"}, 1, 1, 3, false, true};
    SourceLocation loc2 = {FileSpec{"a.c"}, 1, 5, 3, false, true};
    std::string result =
        Render({DiagnosticDetail{loc1, eSeverityError, "X", "X"},
                DiagnosticDetail{loc2, eSeverityError, "Y", "Y"}});
    auto lines = StringRef(result).split('\n');
    auto line1 = lines.first;
    lines = lines.second.split('\n');
    auto line2 = lines.first;
    lines = lines.second.split('\n');
    auto line3 = lines.first;
    //               1234567
    ASSERT_EQ(line1, "^~~ ^~~");
    ASSERT_EQ(line2, "|   error: Y");
    ASSERT_EQ(line3, "error: X");
  }
}
