//===-- LogTest.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gmock/gmock.h"
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

namespace {
// A test fixture which provides tests with a pre-registered channel.
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

// A test fixture which provides tests with a pre-registered and pre-enabled
// channel. Additionally, the messages written to that channel are captured and
// made available via getMessage().
class LogChannelEnabledTest : public LogChannelTest {
  llvm::SmallString<0> m_messages;
  std::shared_ptr<llvm::raw_svector_ostream> m_stream_sp =
      std::make_shared<llvm::raw_svector_ostream>(m_messages);
  Log *m_log;
  size_t m_consumed_bytes = 0;

protected:
  std::shared_ptr<llvm::raw_ostream> getStream() { return m_stream_sp; }
  Log *getLog() { return m_log; }
  llvm::StringRef takeOutput();
  llvm::StringRef logAndTakeOutput(llvm::StringRef Message);

public:
  void SetUp() override;
};
} // end anonymous namespace

void LogChannelEnabledTest::SetUp() {
  LogChannelTest::SetUp();

  std::string error;
  ASSERT_TRUE(EnableChannel(m_stream_sp, 0, "chan", {}, error));

  m_log = test_channel.GetLogIfAll(FOO);
  ASSERT_NE(nullptr, m_log);
}

llvm::StringRef LogChannelEnabledTest::takeOutput() {
  llvm::StringRef result = m_stream_sp->str().drop_front(m_consumed_bytes);
  m_consumed_bytes+= result.size();
  return result;
}

llvm::StringRef LogChannelEnabledTest::logAndTakeOutput(llvm::StringRef Message) {
  LLDB_LOG(m_log, "{0}", Message);
  return takeOutput();
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

TEST_F(LogChannelEnabledTest, log_options) {
  std::string Err;
  EXPECT_EQ("Hello World\n", logAndTakeOutput("Hello World"));
  EXPECT_TRUE(EnableChannel(getStream(), LLDB_LOG_OPTION_THREADSAFE, "chan", {},
                            Err));
  EXPECT_EQ("Hello World\n", logAndTakeOutput("Hello World"));

  {
    EXPECT_TRUE(EnableChannel(getStream(), LLDB_LOG_OPTION_PREPEND_SEQUENCE,
                              "chan", {}, Err));
    llvm::StringRef Msg = logAndTakeOutput("Hello World");
    int seq_no;
    EXPECT_EQ(1, sscanf(Msg.str().c_str(), "%d Hello World", &seq_no));
  }

  {
    EXPECT_TRUE(EnableChannel(getStream(), LLDB_LOG_OPTION_PREPEND_FILE_FUNCTION,
                              "chan", {}, Err));
    llvm::StringRef Msg = logAndTakeOutput("Hello World");
    char File[12];
    char Function[17];
      
    sscanf(Msg.str().c_str(), "%[^:]:%s                                 Hello World", File, Function);
    EXPECT_STRCASEEQ("LogTest.cpp", File);
    EXPECT_STREQ("logAndTakeOutput", Function);
  }

  EXPECT_TRUE(EnableChannel(
      getStream(), LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD, "chan", {}, Err));
  EXPECT_EQ(llvm::formatv("[{0,0+4}/{1,0+4}] Hello World\n", ::getpid(),
                          llvm::get_threadid())
                .str(),
            logAndTakeOutput("Hello World"));
}

TEST_F(LogChannelEnabledTest, LLDB_LOG_ERROR) {
  LLDB_LOG_ERROR(getLog(), llvm::Error::success(), "Foo failed: {0}");
  ASSERT_EQ("", takeOutput());

  LLDB_LOG_ERROR(getLog(),
                 llvm::make_error<llvm::StringError>(
                     "My Error", llvm::inconvertibleErrorCode()),
                 "Foo failed: {0}");
  ASSERT_EQ("Foo failed: My Error\n", takeOutput());

  // Doesn't log, but doesn't assert either
  LLDB_LOG_ERROR(nullptr,
                 llvm::make_error<llvm::StringError>(
                     "My Error", llvm::inconvertibleErrorCode()),
                 "Foo failed: {0}");
}

TEST_F(LogChannelEnabledTest, LogThread) {
  // Test that we are able to concurrently write to a log channel and disable
  // it.
  std::string err;

  // Start logging on one thread. Concurrently, try disabling the log channel.
  std::thread log_thread([this] { LLDB_LOG(getLog(), "Hello World"); });
  EXPECT_TRUE(DisableChannel("chan", {}, err));
  log_thread.join();

  // The log thread either managed to write to the log in time, or it didn't. In
  // either case, we should not trip any undefined behavior (run the test under
  // TSAN to verify this).
  EXPECT_THAT(takeOutput(), testing::AnyOf("", "Hello World\n"));
}

TEST_F(LogChannelEnabledTest, LogVerboseThread) {
  // Test that we are able to concurrently check the verbose flag of a log
  // channel and enable it.
  std::string err;

  // Start logging on one thread. Concurrently, try enabling the log channel
  // (with different log options).
  std::thread log_thread([this] { LLDB_LOGV(getLog(), "Hello World"); });
  EXPECT_TRUE(
      EnableChannel(getStream(), LLDB_LOG_OPTION_VERBOSE, "chan", {}, err));
  log_thread.join();

  // The log thread either managed to write to the log, or it didn't. In either
  // case, we should not trip any undefined behavior (run the test under TSAN to
  // verify this).
  EXPECT_THAT(takeOutput(), testing::AnyOf("", "Hello World\n"));
}

TEST_F(LogChannelEnabledTest, LogGetLogThread) {
  // Test that we are able to concurrently get mask of a Log object and disable
  // it.
  std::string err;

  // Try fetching the log mask on one thread. Concurrently, try disabling the
  // log channel.
  uint32_t mask;
  std::thread log_thread([this, &mask] { mask = getLog()->GetMask().Get(); });
  EXPECT_TRUE(DisableChannel("chan", {}, err));
  log_thread.join();

  // The mask should be either zero of "FOO". In either case, we should not trip
  // any undefined behavior (run the test under TSAN to verify this).
  EXPECT_THAT(mask, testing::AnyOf(0, FOO));
}
