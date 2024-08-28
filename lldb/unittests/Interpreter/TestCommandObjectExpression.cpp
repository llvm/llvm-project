#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Utility/StreamString.h"
#include "gtest/gtest.h"

namespace lldb_private {
std::string RenderDiagnosticDetails(Stream &stream,
                                    std::optional<uint16_t> offset_in_command,
                                    bool multiline,
                                    llvm::ArrayRef<DiagnosticDetail> details);
}

using namespace lldb_private;
using namespace lldb;
using llvm::StringRef;
namespace {
class ErrorDisplayTest : public ::testing::Test {};
} // namespace

static std::string Render(std::vector<DiagnosticDetail> details) {
  StreamString stream;
  RenderDiagnosticDetails(stream, 0, true, details);
  return stream.GetData();
}

TEST_F(ErrorDisplayTest, RenderStatus) {
  DiagnosticDetail::SourceLocation inline_loc;
  inline_loc.in_user_input = true;
  {
    std::string result =
        Render({DiagnosticDetail{inline_loc, eSeverityError, "foo", ""}});
    ASSERT_TRUE(StringRef(result).contains("error:"));
    ASSERT_TRUE(StringRef(result).contains("foo"));
  }
}
