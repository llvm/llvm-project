//===-- LogTest.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Threading.h"
#include <thread>

using namespace lldb;
using namespace lldb_private;

enum { FOO = 1, BAR = 2 };
static constexpr Log::Category test_categories[] = {
    {{"foo"}, {"log foo"}, FOO}, {{"bar"}, {"log bar"}, BAR},
};
static constexpr uint32_t default_flags = FOO;

static Log::Channel test_channel(test_categories, default_flags);

struct LogChannelTest : public ::testing::Test {
  void TearDown() override { Log::DisableAllLogChannels(); }

  static void SetUpTestCase() {
    Log::Register("chan", test_channel);
  }

  static void TearDownTestCase() {
    Log::Unregister("chan");
    llvm::llvm_shutdown();
  }
};

// Wrap enable, disable and list functions to make them easier to test.
static bool EnableChannel(std::shared_ptr<llvm::raw_ostream> stream_sp,
                          uint32_t log_options, llvm::StringRef channel,
                          llvm::ArrayRef<const char *> categories,
                          std::string &error) {
  error.clear();
  llvm::raw_string_ostream error_stream(error);
  return Log::EnableLogChannel(stream_sp, log_options, channel, categories,
                               error_stream);
}

static bool DisableChannel(llvm::StringRef channel,
                           llvm::ArrayRef<const char *> categories,
                           std::string &error) {
  error.clear();
  llvm::raw_string_ostream error_stream(error);
  return Log::DisableLogChannel(channel, categories, error_stream);
}

static bool ListCategories(llvm::StringRef channel, std::string &result) {
  result.clear();
  llvm::raw_string_ostream result_stream(result);
  return Log::ListChannelCategories(channel, result_stream);
}

TEST(LogTest, LLDB_LOG_nullptr) {
  Log *log = nullptr;
  LLDB_LOG(log, "{0}", 0); // Shouldn't crash
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
  std::string message;
  std::shared_ptr<llvm::raw_string_ostream> stream_sp(
      new llvm::raw_string_ostream(message));
  EXPECT_TRUE(Log::EnableLogChannel(stream_sp, 0, "chan", {"foo"}, llvm::nulls()));
  EXPECT_NE(nullptr, test_channel.GetLogIfAny(FOO));
  Log::Unregister("chan");
  EXPECT_EQ(nullptr, test_channel.GetLogIfAny(FOO));
}

TEST_F(LogChannelTest, Enable) {
  EXPECT_EQ(nullptr, test_channel.GetLogIfAll(FOO));
  std::string message;
  std::shared_ptr<llvm::raw_string_ostream> stream_sp(
      new llvm::raw_string_ostream(message));
  std::string error;
  ASSERT_FALSE(EnableChannel(stream_sp, 0, "chanchan", {}, error));
  EXPECT_EQ("Invalid log channel 'chanchan'.\n", error);

  EXPECT_TRUE(EnableChannel(stream_sp, 0, "chan", {}, error));
  EXPECT_NE(nullptr, test_channel.GetLogIfAll(FOO));
  EXPECT_EQ(nullptr, test_channel.GetLogIfAll(BAR));

  EXPECT_TRUE(EnableChannel(stream_sp, 0, "chan", {"bar"}, error));
  EXPECT_NE(nullptr, test_channel.GetLogIfAll(FOO | BAR));

  EXPECT_TRUE(EnableChannel(stream_sp, 0, "chan", {"baz"}, error));
  EXPECT_NE(std::string::npos, error.find("unrecognized log category 'baz'"))
      << "error: " << error;
  EXPECT_NE(nullptr, test_channel.GetLogIfAll(FOO | BAR));
}

TEST_F(LogChannelTest, EnableOptions) {
  EXPECT_EQ(nullptr, test_channel.GetLogIfAll(FOO));
  std::string message;
  std::shared_ptr<llvm::raw_string_ostream> stream_sp(
      new llvm::raw_string_ostream(message));
  std::string error;
  EXPECT_TRUE(
      EnableChannel(stream_sp, LLDB_LOG_OPTION_VERBOSE, "chan", {}, error));

  Log *log = test_channel.GetLogIfAll(FOO);
  ASSERT_NE(nullptr, log);
  EXPECT_TRUE(log->GetVerbose());
}

TEST_F(LogChannelTest, Disable) {
  EXPECT_EQ(nullptr, test_channel.GetLogIfAll(FOO));
  std::string message;
  std::shared_ptr<llvm::raw_string_ostream> stream_sp(
      new llvm::raw_string_ostream(message));
  std::string error;
  EXPECT_TRUE(EnableChannel(stream_sp, 0, "chan", {"foo", "bar"}, error));
  EXPECT_NE(nullptr, test_channel.GetLogIfAll(FOO | BAR));

  EXPECT_TRUE(DisableChannel("chan", {"bar"}, error));
  EXPECT_NE(nullptr, test_channel.GetLogIfAll(FOO));
  EXPECT_EQ(nullptr, test_channel.GetLogIfAll(BAR));

  EXPECT_TRUE(DisableChannel("chan", {"baz"}, error));
  EXPECT_NE(std::string::npos, error.find("unrecognized log category 'baz'"))
      << "error: " << error;
  EXPECT_NE(nullptr, test_channel.GetLogIfAll(FOO));
  EXPECT_EQ(nullptr, test_channel.GetLogIfAll(BAR));

  EXPECT_TRUE(DisableChannel("chan", {}, error));
  EXPECT_EQ(nullptr, test_channel.GetLogIfAny(FOO | BAR));
}

TEST_F(LogChannelTest, List) {
  std::string list;
  EXPECT_TRUE(ListCategories("chan", list));
  std::string expected =
      R"(Logging categories for 'chan':
  all - all available logging categories
  default - default set of logging categories
  foo - log foo
  bar - log bar
)";
  EXPECT_EQ(expected, list);

  EXPECT_FALSE(ListCategories("chanchan", list));
  EXPECT_EQ("Invalid log channel 'chanchan'.\n", list);
}

static std::string GetLogString(uint32_t log_options, const char *format,
                                int arg) {
  std::string message;
  std::shared_ptr<llvm::raw_string_ostream> stream_sp(
      new llvm::raw_string_ostream(message));
  std::string error;
  llvm::raw_string_ostream error_stream(error);
  EXPECT_TRUE(
      Log::EnableLogChannel(stream_sp, log_options, "chan", {}, error_stream));

  Log *log = test_channel.GetLogIfAll(FOO);
  EXPECT_NE(nullptr, log);

  LLDB_LOG(log, format, arg);
  EXPECT_TRUE(Log::DisableLogChannel("chan", {}, error_stream));

  return stream_sp->str();
}

TEST_F(LogChannelTest, log_options) {
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

  EXPECT_EQ(llvm::formatv("[{0,0+4}/{1,0+4}] Hello World 47\n", ::getpid(),
                          llvm::get_threadid())
                .str(),
            GetLogString(LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD,
                         "Hello World {0}", 47));
}

TEST_F(LogChannelTest, LogThread) {
  // Test that we are able to concurrently write to a log channel and disable
  // it.
  std::string message;
  std::shared_ptr<llvm::raw_string_ostream> stream_sp(
      new llvm::raw_string_ostream(message));
  std::string err;
  EXPECT_TRUE(EnableChannel(stream_sp, 0, "chan", {}, err));

  Log *log = test_channel.GetLogIfAll(FOO);

  // Start logging on one thread. Concurrently, try disabling the log channel.
  std::thread log_thread([log] { LLDB_LOG(log, "Hello World"); });
  EXPECT_TRUE(DisableChannel("chan", {}, err));
  log_thread.join();

  // The log thread either managed to write to the log in time, or it didn't. In
  // either case, we should not trip any undefined behavior (run the test under
  // TSAN to verify this).
  EXPECT_TRUE(stream_sp->str() == "" || stream_sp->str() == "Hello World\n")
      << "str(): " << stream_sp->str();
}

TEST_F(LogChannelTest, LogVerboseThread) {
  // Test that we are able to concurrently check the verbose flag of a log
  // channel and enable it.
  std::string message;
  std::shared_ptr<llvm::raw_string_ostream> stream_sp(
      new llvm::raw_string_ostream(message));
  std::string err;
  EXPECT_TRUE(EnableChannel(stream_sp, 0, "chan", {}, err));

  Log *log = test_channel.GetLogIfAll(FOO);

  // Start logging on one thread. Concurrently, try enabling the log channel
  // (with different log options).
  std::thread log_thread([log] { LLDB_LOGV(log, "Hello World"); });
  EXPECT_TRUE(EnableChannel(stream_sp, LLDB_LOG_OPTION_VERBOSE, "chan",
                                    {}, err));
  log_thread.join();
  EXPECT_TRUE(DisableChannel("chan", {}, err));

  // The log thread either managed to write to the log, or it didn't. In either
  // case, we should not trip any undefined behavior (run the test under TSAN to
  // verify this).
  EXPECT_TRUE(stream_sp->str() == "" || stream_sp->str() == "Hello World\n")
      << "str(): " << stream_sp->str();
}

TEST_F(LogChannelTest, LogGetLogThread) {
  // Test that we are able to concurrently get mask of a Log object and disable
  // it.
  std::string message;
  std::shared_ptr<llvm::raw_string_ostream> stream_sp(
      new llvm::raw_string_ostream(message));
  std::string err;
  EXPECT_TRUE(EnableChannel(stream_sp, 0, "chan", {}, err));
  Log *log = test_channel.GetLogIfAll(FOO);

  // Try fetching the log on one thread. Concurrently, try disabling the log
  // channel.
  uint32_t mask;
  std::thread log_thread([log, &mask] { mask = log->GetMask().Get(); });
  EXPECT_TRUE(DisableChannel("chan", {}, err));
  log_thread.join();

  // The mask should be either zero of "FOO". In either case, we should not trip
  // any undefined behavior (run the test under TSAN to verify this).
  EXPECT_TRUE(mask == 0 || mask == FOO) << "mask: " << mask;
}
