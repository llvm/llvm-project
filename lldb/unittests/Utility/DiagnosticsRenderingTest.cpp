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
    SourceLocation loc1 = {FileSpec{"a.c"}, 13, 5, 0, false, true};
    SourceLocation loc2 = {FileSpec{"a.c"}, 13, 7, 0, false, true};
    SourceLocation loc3 = {FileSpec{"a.c"}, 13, 9, 0, false, true};
    std::string result =
        Render({DiagnosticDetail{loc1, eSeverityError, "1", "1"},
                DiagnosticDetail{loc2, eSeverityError, "2a", "2a"},
                DiagnosticDetail{loc2, eSeverityInfo, "2b", "2b"},
                DiagnosticDetail{loc3, eSeverityError, "3", "3"}});
    llvm::SmallVector<StringRef> lines;
    StringRef(result).split(lines, '\n');
    //                1234567890123
    ASSERT_EQ(lines[0], "    ^ ^ ^");
    ASSERT_EQ(lines[1], "    | | error: 3");
    ASSERT_EQ(lines[2], "    | error: 2a");
    ASSERT_EQ(lines[3], "    | note: 2b");
    ASSERT_EQ(lines[4], "    error: 1");
  }
  {
    // Test that diagnostics in reverse order are emitted correctly.
    SourceLocation loc1 = {FileSpec{"a.c"}, 1, 20, 0, false, true};
    SourceLocation loc2 = {FileSpec{"a.c"}, 2, 10, 0, false, true};
    std::string result =
        Render({DiagnosticDetail{loc2, eSeverityError, "X", "X"},
                DiagnosticDetail{loc1, eSeverityError, "Y", "Y"}});
    // Unintuitively the later diagnostic appears first in the string:
    //    ^   ^
    //    |   second
    //    first
    ASSERT_GT(StringRef(result).find("Y"), StringRef(result).find("X"));
  }
  {
    // Test that diagnostics in reverse order are emitted correctly.
    SourceLocation loc1 = {FileSpec{"a.c"}, 1, 10, 0, false, true};
    SourceLocation loc2 = {FileSpec{"a.c"}, 1, 20, 0, false, true};
    std::string result =
        Render({DiagnosticDetail{loc2, eSeverityError, "X", "X"},
                DiagnosticDetail{loc1, eSeverityError, "Y", "Y"}});
    ASSERT_GT(StringRef(result).find("Y"), StringRef(result).find("X"));
  }
  {
    // Test that range diagnostics are emitted correctly.
    SourceLocation loc1 = {FileSpec{"a.c"}, 1, 1, 3, false, true};
    SourceLocation loc2 = {FileSpec{"a.c"}, 1, 5, 3, false, true};
    std::string result =
        Render({DiagnosticDetail{loc1, eSeverityError, "X", "X"},
                DiagnosticDetail{loc2, eSeverityError, "Y", "Y"}});
    llvm::SmallVector<StringRef> lines;
    StringRef(result).split(lines, '\n');
    //                1234567
    ASSERT_EQ(lines[0], "^~~ ^~~");
    ASSERT_EQ(lines[1], "|   error: Y");
    ASSERT_EQ(lines[2], "error: X");
  }
  {
    // Test diagnostics on the same line are emitted correctly.
    SourceLocation loc1 = {FileSpec{"a.c"}, 1, 2, 0, false, true};
    SourceLocation loc2 = {FileSpec{"a.c"}, 1, 6, 0, false, true};
    std::string result =
        Render({DiagnosticDetail{loc1, eSeverityError, "X", "X"},
                DiagnosticDetail{loc2, eSeverityError, "Y", "Y"}});
    llvm::SmallVector<StringRef> lines;
    StringRef(result).split(lines, '\n');
    //                1234567
    ASSERT_EQ(lines[0], " ^   ^");
    ASSERT_EQ(lines[1], " |   error: Y");
    ASSERT_EQ(lines[2], " error: X");
  }
}
