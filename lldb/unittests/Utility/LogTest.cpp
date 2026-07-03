//===-- LogTest.cpp -------------------------------------------------------===//
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
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Threading.h"
#include "llvm/Testing/Support/Error.h"
#include <thread>

using namespace lldb;
using namespace lldb_private;

enum class TestChannel : Log::MaskType {
  FOO = Log::ChannelFlag<0>,
  BAR = Log::ChannelFlag<1>,
  LLVM_MARK_AS_BITMASK_ENUM(BAR),
};

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

static constexpr Log::Category test_categories[] = {
    {{"foo"}, {"log foo"}, TestChannel::FOO},
    {{"bar"}, {"log bar"}, TestChannel::BAR},
};

static Log::Channel test_channel(test_categories, TestChannel::FOO);

namespace lldb_private {
template <> Log::Channel &LogChannelFor<TestChannel>() { return test_channel; }
} // namespace lldb_private

// Wrap list function to make it easier to test.
static bool ListCategories(llvm::StringRef channel, std::string &result) {
  result.clear();
  llvm::raw_string_ostream result_stream(result);
  return Log::ListChannelCategories(channel, result_stream);
}

namespace {
// A test fixture which provides tests with a pre-registered channel.
struct LogChannelTest : public ::testing::Test {
  void TearDown() override { Log::DisableAllLogChannels(); }

  static void SetUpTestCase() { Log::Register("chan", test_channel); }

  static void TearDownTestCase() {
    Log::Unregister("chan");
    llvm::llvm_shutdown();
  }
};

class TestLogHandler : public LogHandler {
public:
  TestLogHandler() : m_messages(), m_stream(m_messages) {}

  void Emit(llvm::StringRef message) override { m_stream << message; }

  llvm::SmallString<0> m_messages;
  llvm::raw_svector_ostream m_stream;
};

// A test fixture which provides tests with a pre-registered and pre-enabled
// channel. Additionally, the messages written to that channel are captured and
// made available via getMessage().
class LogChannelEnabledTest : public LogChannelTest {
  std::shared_ptr<TestLogHandler> m_log_handler_sp =
      std::make_shared<TestLogHandler>();
  Log *m_log;
  size_t m_consumed_bytes = 0;

protected:
  std::shared_ptr<LogHandler> getLogHandler() { return m_log_handler_sp; }
  Log *getLog() { return m_log; }
  llvm::StringRef takeOutput();
  llvm::StringRef logAndTakeOutput(llvm::StringRef Message);
  llvm::StringRef logAndTakeOutputf(llvm::StringRef Message);

public:
  void SetUp() override;
};

static std::string GetDumpAsString(const RotatingLogHandler &handler) {
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  handler.Dump(stream);
  return buffer;
}
} // end anonymous namespace

void LogChannelEnabledTest::SetUp() {
  LogChannelTest::SetUp();

  ASSERT_THAT_ERROR(Log::EnableLogChannel(m_log_handler_sp, 0, "chan", {}),
                    llvm::Succeeded());

  m_log = GetLog(TestChannel::FOO);
  ASSERT_NE(nullptr, m_log);
}

llvm::StringRef LogChannelEnabledTest::takeOutput() {
  llvm::StringRef result =
      m_log_handler_sp->m_stream.str().drop_front(m_consumed_bytes);
  m_consumed_bytes += result.size();
  return result;
}

llvm::StringRef
LogChannelEnabledTest::logAndTakeOutput(llvm::StringRef Message) {
  LLDB_LOG(m_log, "{0}", Message);
  return takeOutput();
}

llvm::StringRef
LogChannelEnabledTest::logAndTakeOutputf(llvm::StringRef Message) {
  LLDB_LOGF(m_log, "%s", Message.str().c_str());
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
  EXPECT_EQ(nullptr, GetLog(TestChannel::FOO));
  auto log_handler_sp = std::make_shared<TestLogHandler>();
  EXPECT_THAT_ERROR(Log::EnableLogChannel(log_handler_sp, 0, "chan", {"foo"}),
                    llvm::Succeeded());
  EXPECT_NE(nullptr, GetLog(TestChannel::FOO));
  Log::Unregister("chan");
  EXPECT_EQ(nullptr, GetLog(TestChannel::FOO));
}

namespace {
static char test_baton;
static size_t callback_count = 0;
static void TestCallback(const char *data, void *baton) {
  EXPECT_STREQ("Foobar", data);
  EXPECT_EQ(&test_baton, baton);
  ++callback_count;
}
} // namespace

TEST(LogTest, CallbackLogHandler) {
  CallbackLogHandler handler(TestCallback, &test_baton);
  handler.Emit("Foobar");
  EXPECT_EQ(1u, callback_count);
}

TEST(LogHandlerTest, RotatingLogHandler) {
  RotatingLogHandler handler(3);

  handler.Emit("foo");
  handler.Emit("bar");
  EXPECT_EQ(GetDumpAsString(handler), "foobar");

  handler.Emit("baz");
  handler.Emit("qux");
  EXPECT_EQ(GetDumpAsString(handler), "barbazqux");

  handler.Emit("quux");
  EXPECT_EQ(GetDumpAsString(handler), "bazquxquux");
}

TEST(LogHandlerTest, TeeLogHandler) {
  auto handler1 = std::make_shared<RotatingLogHandler>(2);
  auto handler2 = std::make_shared<RotatingLogHandler>(2);
  TeeLogHandler handler(handler1, handler2);

  handler.Emit("foo");
  handler.Emit("bar");

  EXPECT_EQ(GetDumpAsString(*handler1), "foobar");
  EXPECT_EQ(GetDumpAsString(*handler2), "foobar");
}

TEST_F(LogChannelTest, Enable) {
  EXPECT_EQ(nullptr, GetLog(TestChannel::FOO));
  auto log_handler_sp = std::make_shared<TestLogHandler>();
  ASSERT_THAT_ERROR(
      Log::EnableLogChannel(log_handler_sp, 0, "chanchan", {}),
      llvm::FailedWithMessage("Invalid log channel 'chanchan'.\n"));

  EXPECT_THAT_ERROR(Log::EnableLogChannel(log_handler_sp, 0, "chan", {}),
                    llvm::Succeeded());
  EXPECT_NE(nullptr, GetLog(TestChannel::FOO));
  EXPECT_EQ(nullptr, GetLog(TestChannel::BAR));
  EXPECT_NE(nullptr, GetLog(TestChannel::FOO | TestChannel::BAR));

  EXPECT_THAT_ERROR(Log::EnableLogChannel(log_handler_sp, 0, "chan", {"bar"}),
                    llvm::Succeeded());
  EXPECT_NE(nullptr, GetLog(TestChannel::FOO));
  EXPECT_NE(nullptr, GetLog(TestChannel::BAR));

  EXPECT_THAT_ERROR(
      Log::EnableLogChannel(log_handler_sp, 0, "chan", {"baz"}),
      llvm::FailedWithMessage("error: unrecognized log category 'baz'\n"
                              "Logging categories for 'chan':\n"
                              "  all - all available logging categories\n"
                              "  default - default set of logging categories\n"
                              "  foo - log foo\n"
                              "  bar - log bar\n"));

  EXPECT_THAT_ERROR(
      Log::EnableLogChannel(log_handler_sp, 0, "chan", {"baz", "bar", "abc"}),
      llvm::FailedWithMessage(
          "error: unrecognized log categories 'baz', 'abc'\n"
          "Logging categories for 'chan':\n"
          "  all - all available logging categories\n"
          "  default - default set of logging categories\n"
          "  foo - log foo\n"
          "  bar - log bar\n"));
}

TEST_F(LogChannelTest, EnableOptions) {
  EXPECT_EQ(nullptr, GetLog(TestChannel::FOO));
  auto log_handler_sp = std::make_shared<TestLogHandler>();
  EXPECT_THAT_ERROR(Log::EnableLogChannel(log_handler_sp,
                                          LLDB_LOG_OPTION_VERBOSE, "chan", {}),
                    llvm::Succeeded());

  Log *log = GetLog(TestChannel::FOO);
  ASSERT_NE(nullptr, log);
  EXPECT_TRUE(log->GetVerbose());
}

TEST_F(LogChannelTest, Disable) {
  EXPECT_EQ(nullptr, GetLog(TestChannel::FOO));
  auto log_handler_sp = std::make_shared<TestLogHandler>();
  EXPECT_THAT_ERROR(
      Log::EnableLogChannel(log_handler_sp, 0, "chan", {"foo", "bar"}),
      llvm::Succeeded());
  EXPECT_NE(nullptr, GetLog(TestChannel::FOO));
  EXPECT_NE(nullptr, GetLog(TestChannel::BAR));

  std::string error;
  EXPECT_THAT_ERROR(Log::DisableLogChannel("chan", {"bar"}), llvm::Succeeded());
  EXPECT_NE(nullptr, GetLog(TestChannel::FOO));
  EXPECT_EQ(nullptr, GetLog(TestChannel::BAR));

  EXPECT_THAT_ERROR(
      Log::DisableLogChannel("chan", {"baz"}),
      llvm::FailedWithMessage("error: unrecognized log category 'baz'\n"
                              "Logging categories for 'chan':\n"
                              "  all - all available logging categories\n"
                              "  default - default set of logging categories\n"
                              "  foo - log foo\n"
                              "  bar - log bar\n"));
  EXPECT_NE(nullptr, GetLog(TestChannel::FOO));
  EXPECT_EQ(nullptr, GetLog(TestChannel::BAR));

  EXPECT_THAT_ERROR(Log::DisableLogChannel("chan", {}), llvm::Succeeded());
  EXPECT_EQ(nullptr, GetLog(TestChannel::FOO | TestChannel::BAR));
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
  EXPECT_EQ("Hello World\n", logAndTakeOutput("Hello World"));
  EXPECT_THAT_ERROR(Log::EnableLogChannel(getLogHandler(), 0, "chan", {}),
                    llvm::Succeeded());
  EXPECT_EQ("Hello World\n", logAndTakeOutput("Hello World"));

  {
    EXPECT_THAT_ERROR(Log::EnableLogChannel(getLogHandler(),
                                            LLDB_LOG_OPTION_PREPEND_SEQUENCE,
                                            "chan", {}),
                      llvm::Succeeded());
    llvm::StringRef Msg = logAndTakeOutput("Hello World");
    int seq_no;
    EXPECT_EQ(1, sscanf(Msg.str().c_str(), "%d Hello World", &seq_no));
  }

  {
    EXPECT_THAT_ERROR(
        Log::EnableLogChannel(
            getLogHandler(), LLDB_LOG_OPTION_PREPEND_FILE_FUNCTION, "chan", {}),
        llvm::Succeeded());
    llvm::StringRef Msg = logAndTakeOutput("Hello World");
    char File[12];
    char Function[17];

    sscanf(Msg.str().c_str(),
           "%[^:]:%s                                 Hello World", File,
           Function);
    EXPECT_STRCASEEQ("LogTest.cpp", File);
    EXPECT_STREQ("logAndTakeOutput", Function);
  }

  {
    EXPECT_THAT_ERROR(
        Log::EnableLogChannel(
            getLogHandler(), LLDB_LOG_OPTION_PREPEND_FILE_FUNCTION, "chan", {}),
        llvm::Succeeded());
    llvm::StringRef Msg = logAndTakeOutputf("Hello World");
    char File[12];
    char Function[18];

    sscanf(Msg.str().c_str(),
           "%[^:]:%s                                 Hello World", File,
           Function);
    EXPECT_STRCASEEQ("LogTest.cpp", File);
    EXPECT_STREQ("logAndTakeOutputf", Function);
  }

  EXPECT_THAT_ERROR(
      Log::EnableLogChannel(
          getLogHandler(), LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD, "chan", {}),
      llvm::Succeeded());
  EXPECT_EQ(llvm::formatv("[{0,0+4}/{1,0+4}] Hello World\n", ::getpid(),
                          llvm::get_threadid())
                .str(),
            logAndTakeOutput("Hello World"));
}

TEST_F(LogChannelEnabledTest, JSONLOutput) {
  // With JSON, every line must parse as JSON and contain exactly the message we
  // logged.
  EXPECT_THAT_ERROR(Log::EnableLogChannel(getLogHandler(),
                                          /*log_options=*/LLDB_LOG_OPTION_JSON,
                                          "chan", {}),
                    llvm::Succeeded());
  llvm::StringRef Msg = logAndTakeOutput("Hello \"World\"\nsecond line");
  ASSERT_TRUE(Msg.ends_with("\n"));
  ASSERT_EQ(Msg.count('\n'), 1u) << "expected one JSON object per line";
  llvm::Expected<llvm::json::Value> Parsed = llvm::json::parse(Msg);
  ASSERT_TRUE(static_cast<bool>(Parsed)) << llvm::toString(Parsed.takeError());
  llvm::json::Object *Obj = Parsed->getAsObject();
  ASSERT_NE(Obj, nullptr);
  EXPECT_EQ(Obj->getString("message").value_or(""),
            "Hello \"World\"\nsecond line");
  EXPECT_EQ(Obj->size(), 1u);

  // Combining JSON with metadata flags: each metadata flag becomes a JSON
  // field instead of a prefix on the line.
  uint32_t Opts = LLDB_LOG_OPTION_JSON | LLDB_LOG_OPTION_PREPEND_SEQUENCE |
                  LLDB_LOG_OPTION_PREPEND_TIMESTAMP |
                  LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD |
                  LLDB_LOG_OPTION_PREPEND_THREAD_NAME |
                  LLDB_LOG_OPTION_BACKTRACE |
                  LLDB_LOG_OPTION_PREPEND_FILE_FUNCTION;
  EXPECT_THAT_ERROR(Log::EnableLogChannel(getLogHandler(), Opts, "chan", {}),
                    llvm::Succeeded());
  Msg = logAndTakeOutput("payload");
  Parsed = llvm::json::parse(Msg);
  ASSERT_TRUE(static_cast<bool>(Parsed)) << llvm::toString(Parsed.takeError());
  Obj = Parsed->getAsObject();
  ASSERT_NE(Obj, nullptr);
  EXPECT_EQ(Obj->getString("message").value_or(""), "payload");
  EXPECT_TRUE(Obj->getInteger("sequence").has_value());
  EXPECT_TRUE(Obj->getNumber("timestamp").has_value());
  EXPECT_EQ(Obj->getInteger("pid").value_or(-1),
            static_cast<int64_t>(::getpid()));
  EXPECT_EQ(Obj->getInteger("tid").value_or(-1),
            static_cast<int64_t>(llvm::get_threadid()));
  EXPECT_TRUE(Obj->getString("thread_name").has_value());
  EXPECT_TRUE(Obj->getString("backtrace").has_value());
  EXPECT_EQ(Obj->getString("file").value_or(""), "LogTest.cpp");
  EXPECT_EQ(Obj->getString("function").value_or(""), "logAndTakeOutput");
}

TEST_F(LogChannelEnabledTest, LLDB_LOG_ERROR) {
  LLDB_LOG_ERROR(getLog(), llvm::Error::success(), "Foo failed: {0}");
  ASSERT_EQ("", takeOutput());

  LLDB_LOG_ERROR(getLog(), llvm::createStringError("My Error"),
                 "Foo failed: {0}");
  ASSERT_EQ("Foo failed: My Error\n", takeOutput());

  // Doesn't log, but doesn't assert either
  LLDB_LOG_ERROR(nullptr, llvm::createStringError("My Error"),
                 "Foo failed: {0}");
}

TEST_F(LogChannelEnabledTest, LogThread) {
  // Test that we are able to concurrently write to a log channel and disable
  // it.
  // Start logging on one thread. Concurrently, try disabling the log channel.
  std::thread log_thread([this] { LLDB_LOG(getLog(), "Hello World"); });
  EXPECT_THAT_ERROR(Log::DisableLogChannel("chan", {}), llvm::Succeeded());
  log_thread.join();

  // The log thread either managed to write to the log in time, or it didn't. In
  // either case, we should not trip any undefined behavior (run the test under
  // TSAN to verify this).
  EXPECT_THAT(takeOutput(), testing::AnyOf("", "Hello World\n"));
}

TEST_F(LogChannelEnabledTest, LogVerboseThread) {
  // Test that we are able to concurrently check the verbose flag of a log
  // channel and enable it.

  // Start logging on one thread. Concurrently, try enabling the log channel
  // (with different log options).
  std::thread log_thread([this] { LLDB_LOG_VERBOSE(getLog(), "Hello World"); });
  EXPECT_THAT_ERROR(Log::EnableLogChannel(getLogHandler(),
                                          LLDB_LOG_OPTION_VERBOSE, "chan", {}),
                    llvm::Succeeded());
  log_thread.join();

  // The log thread either managed to write to the log, or it didn't. In either
  // case, we should not trip any undefined behavior (run the test under TSAN to
  // verify this).
  EXPECT_THAT(takeOutput(), testing::AnyOf("", "Hello World\n"));
}

TEST_F(LogChannelEnabledTest, LogGetLogThread) {
  // Test that we are able to concurrently get mask of a Log object and disable
  // it.

  // Try fetching the log mask on one thread. Concurrently, try disabling the
  // log channel.
  uint64_t mask;
  std::thread log_thread([this, &mask] { mask = getLog()->GetMask(); });
  EXPECT_THAT_ERROR(Log::DisableLogChannel("chan", {}), llvm::Succeeded());
  log_thread.join();

  // The mask should be either zero of "FOO". In either case, we should not trip
  // any undefined behavior (run the test under TSAN to verify this).
  EXPECT_THAT(mask, testing::AnyOf(0, Log::MaskType(TestChannel::FOO)));
}
