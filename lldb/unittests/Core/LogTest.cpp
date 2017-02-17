//===-- LogTest.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Core/Log.h"
#include "lldb/Host/Host.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/Support/ManagedStatic.h"

using namespace lldb;
using namespace lldb_private;

enum { FOO = 1, BAR = 2 };
static constexpr Log::Category test_categories[] = {
    {"foo", "log foo", FOO}, {"bar", "log bar", BAR},
};
static constexpr uint32_t default_flags = FOO;

static Log::Channel test_channel(test_categories, default_flags);

struct LogChannelTest : public ::testing::Test {
  static void SetUpTestCase() {
    Log::Register("chan", test_channel);
  }

  static void TearDownTestCase() {
    Log::Unregister("chan");
    llvm::llvm_shutdown();
  }
};

static std::string GetLogString(uint32_t log_options, const char *format,
                                int arg) {
  std::string stream_string;
  std::shared_ptr<llvm::raw_string_ostream> stream_sp(
      new llvm::raw_string_ostream(stream_string));
  Log log_(stream_sp);
  log_.GetOptions().Reset(log_options);
  Log *log = &log_;
  LLDB_LOG(log, format, arg);
  return stream_sp->str();
}

TEST(LogTest, LLDB_LOG_nullptr) {
  Log *log = nullptr;
  LLDB_LOG(log, "{0}", 0); // Shouldn't crash
}

TEST(LogTest, log_options) {
  EXPECT_EQ("Hello World 47\n", GetLogString(0, "Hello World {0}", 47));
  EXPECT_EQ("Hello World 47\n",
            GetLogString(LLDB_LOG_OPTION_THREADSAFE, "Hello World {0}", 47));

  {
    std::string msg =
        GetLogString(LLDB_LOG_OPTION_PREPEND_SEQUENCE, "Hello World {0}", 47);
    int seq_no;
    EXPECT_EQ(1, sscanf(msg.c_str(), "%d Hello World 47", &seq_no));
  }

  EXPECT_EQ(
      "LogTest.cpp:GetLogString                                     Hello "
      "World 47\n",
      GetLogString(LLDB_LOG_OPTION_PREPEND_FILE_FUNCTION, "Hello World {0}", 47));

  EXPECT_EQ(llvm::formatv("[{0,0+4}/{1,0+4}] Hello World 47\n",
                          Host::GetCurrentProcessID(),
                          Host::GetCurrentThreadID())
                .str(),
            GetLogString(LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD,
                         "Hello World {0}", 47));
}

TEST(LogTest, Register) {
  llvm::llvm_shutdown_obj obj;
  Log::Register("chan", test_channel);
  Log::Unregister("chan");
  Log::Register("chan", test_channel);
  Log::Unregister("chan");
}

TEST(LogTest, Unregister) {
  llvm::llvm_shutdown_obj obj;
  Log::Register("chan", test_channel);
  EXPECT_EQ(nullptr, test_channel.GetLogIfAny(FOO));
  const char *cat1[] = {"foo", nullptr};
  std::string message;
  std::shared_ptr<llvm::raw_string_ostream> stream_sp(
      new llvm::raw_string_ostream(message));
  StreamString err;
  EXPECT_TRUE(Log::EnableLogChannel(stream_sp, 0, "chan", cat1, err));
  EXPECT_NE(nullptr, test_channel.GetLogIfAny(FOO));
  Log::Unregister("chan");
  EXPECT_EQ(nullptr, test_channel.GetLogIfAny(FOO));
}

TEST_F(LogChannelTest, Enable) {
  EXPECT_EQ(nullptr, test_channel.GetLogIfAll(FOO));
  std::string message;
  std::shared_ptr<llvm::raw_string_ostream> stream_sp(
      new llvm::raw_string_ostream(message));
  StreamString err;
  EXPECT_FALSE(Log::EnableLogChannel(stream_sp, 0, "chanchan", nullptr, err));
  EXPECT_EQ("Invalid log channel 'chanchan'.\n", err.GetString());
  err.Clear();

  EXPECT_TRUE(Log::EnableLogChannel(stream_sp, 0, "chan", nullptr, err));
  EXPECT_EQ("", err.GetString());
  EXPECT_NE(nullptr, test_channel.GetLogIfAll(FOO));
  EXPECT_EQ(nullptr, test_channel.GetLogIfAll(BAR));

  const char *cat2[] = {"bar", nullptr};
  EXPECT_TRUE(Log::EnableLogChannel(stream_sp, 0, "chan", cat2, err));
  EXPECT_NE(nullptr, test_channel.GetLogIfAll(FOO | BAR));

  const char *cat3[] = {"baz", nullptr};
  EXPECT_TRUE(Log::EnableLogChannel(stream_sp, 0, "chan", cat3, err));
  EXPECT_TRUE(err.GetString().contains("unrecognized log category 'baz'"))
      << "err: " << err.GetString().str();
  EXPECT_NE(nullptr, test_channel.GetLogIfAll(FOO | BAR));
}

TEST_F(LogChannelTest, Disable) {
  EXPECT_EQ(nullptr, test_channel.GetLogIfAll(FOO));
  const char *cat12[] = {"foo", "bar", nullptr};
  std::string message;
  std::shared_ptr<llvm::raw_string_ostream> stream_sp(
      new llvm::raw_string_ostream(message));
  StreamString err;
  EXPECT_TRUE(Log::EnableLogChannel(stream_sp, 0, "chan", cat12, err));
  EXPECT_NE(nullptr, test_channel.GetLogIfAll(FOO | BAR));

  const char *cat2[] = {"bar", nullptr};
  EXPECT_TRUE(Log::DisableLogChannel("chan", cat2, err));
  EXPECT_NE(nullptr, test_channel.GetLogIfAll(FOO));
  EXPECT_EQ(nullptr, test_channel.GetLogIfAll(BAR));

  const char *cat3[] = {"baz", nullptr};
  EXPECT_TRUE(Log::DisableLogChannel("chan", cat3, err));
  EXPECT_TRUE(err.GetString().contains("unrecognized log category 'baz'"))
      << "err: " << err.GetString().str();
  EXPECT_NE(nullptr, test_channel.GetLogIfAll(FOO));
  EXPECT_EQ(nullptr, test_channel.GetLogIfAll(BAR));
  err.Clear();

  EXPECT_TRUE(Log::DisableLogChannel("chan", nullptr, err));
  EXPECT_EQ(nullptr, test_channel.GetLogIfAny(FOO | BAR));
}

TEST_F(LogChannelTest, List) {
  StreamString str;
  EXPECT_TRUE(Log::ListChannelCategories("chan", str));
  std::string expected =
      R"(Logging categories for 'chan':
  all - all available logging categories
  default - default set of logging categories
  foo - log foo
  bar - log bar
)";
  EXPECT_EQ(expected, str.GetString().str());
  str.Clear();

  EXPECT_FALSE(Log::ListChannelCategories("chanchan", str));
  EXPECT_EQ("Invalid log channel 'chanchan'.\n", str.GetString().str());
}
