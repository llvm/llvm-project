//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProgressEvent.h"
#include "Protocol/ProtocolBase.h"
#include "TestingSupport/TestUtilities.h"
#include "llvm/Support/JSON.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <chrono>
#include <string>
#include <utility>
#include <vector>

using namespace lldb_dap;
using namespace lldb_dap::protocol;
using namespace std::chrono_literals;

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

namespace lldb_dap::protocol {
// For formatted error messages.
// see https://google.github.io/googletest/advanced.html.
void PrintTo(const Event &e, std::ostream *os) {
  *os << lldb_private::PrettyPrint(toJSON(e));
}
} // namespace lldb_dap::protocol

namespace {

const llvm::json::Value *BodyField(const Event &e, llvm::StringRef key) {
  if (!e.body)
    return nullptr;
  const auto *obj = e.body->getAsObject();
  if (!obj)
    return nullptr;
  return obj->get(key);
}

std::string BodyFieldStr(const Event &e, llvm::StringRef key) {
  const auto *v = BodyField(e, key);
  if (!v)
    return {};
  return v->getAsString().value_or("").str();
}

MATCHER_P(ProgressKind, kind,
          "event.event is " + testing::PrintToString(kind)) {
  return arg.event == std::string(kind);
}

MATCHER_P(HasId, id, "body.progressId is " + testing::PrintToString(id)) {
  return BodyFieldStr(arg, "progressId") == std::to_string(id);
}

MATCHER_P(HasMessage, message,
          "body.message is " + testing::PrintToString(message)) {
  return BodyFieldStr(arg, "message") == std::string(message);
}

MATCHER_P(HasTitle, title, "body.title is " + testing::PrintToString(title)) {
  return BodyFieldStr(arg, "title") == std::string(title);
}

MATCHER(HasNoPercentage, "body has no percentage field") {
  return BodyField(arg, "percentage") == nullptr;
}

MATCHER_P(HasPercentage, expected,
          "body.percentage is " + testing::PrintToString(expected)) {
  const auto *v = BodyField(arg, "percentage");
  if (!v)
    return false;
  auto n = v->getAsNumber();
  return n && *n == expected;
}

template <typename... Fields>
auto Progress(llvm::StringRef kind, Fields &&...fields) {
  return AllOf(ProgressKind(kind), std::forward<Fields>(fields)...);
}

class ProgressEventReporterTest : public ::testing::Test {
protected:
  ProgressEventReporter::TimePoint now{std::chrono::seconds(0)};
  std::vector<Event> sent_events;
  protocol::Id seq_id = 0;

  ProgressEventReporter m_reporter{[this](Event e) {
    e.seq = ++seq_id;
    sent_events.push_back(std::move(e));
  }};

  void IncreaseTime(std::chrono::milliseconds d) { now += d; }

  // Advance the fake clock by `offset`, then Report.
  void ReportAt(std::chrono::milliseconds offset, uint64_t id,
                llvm::StringRef title, llvm::StringRef details,
                uint64_t completed, uint64_t total) {
    IncreaseTime(offset);
    m_reporter.Report(id, title.str(), details.str(), completed, total, now);
  }
};

/// Progress reported before the start delay is not sent.
TEST_F(ProgressEventReporterTest, DropsFastProgress) {
  const llvm::StringRef title = "Parsing Compile Units";
  ReportAt(0ms, 1, title, "", 0, 4);
  ReportAt(ProgressEventReporter::k_start_delay - 500ms, 1, title, "foo.cpp", 1,
           4);
  ReportAt(0ms, 1, title, "bar.cpp", 2, 4);
  ReportAt(0ms, 1, title, "baz.cpp", 3, 4);

  EXPECT_THAT(sent_events, IsEmpty());
}

/// Progress ending without an update after the start delay, sends both the
/// start and end progressEvents.
TEST_F(ProgressEventReporterTest, EmitsStartAndEndProgressAfterStartDelay) {
  const llvm::StringRef title = "Reading all files";
  const llvm::StringRef end_detail = "done reading.";
  ReportAt(1ms, 1, title, "in folder", 0, 3);
  ReportAt(ProgressEventReporter::k_start_delay + 500ms, 1, title, end_detail,
           3, 3);

  auto expected_events = ElementsAre(
      Progress("progressStart", HasTitle(title), HasMessage(end_detail)),
      Progress("progressEnd", HasMessage(end_detail)));
  EXPECT_THAT(sent_events, expected_events);
}

TEST_F(ProgressEventReporterTest, DrainEmitsStartForIdleProgress) {
  ReportAt(0ms, 1, "title", "", 0, 3);
  EXPECT_THAT(sent_events, IsEmpty()); // buffered

  IncreaseTime(ProgressEventReporter::k_start_delay + 500ms);
  m_reporter.Drain(now);

  EXPECT_THAT(sent_events, ElementsAre(Progress("progressStart", HasId(1))));

  // Calling drain multiple times does not create new events.
  for (int i = 0; i < 10; i++)
    m_reporter.Drain(now);
  EXPECT_TRUE(m_reporter.HasPending());
  EXPECT_THAT(sent_events, ElementsAre(Progress("progressStart")));
}

/// Multiple update progessEvents are reported until after the update_interval.
TEST_F(ProgressEventReporterTest, ThrottlesUpdatesWithinInterval) {
  // Send the start.
  const uint32_t id = 10;
  ReportAt(0ms, id, "throttled", "d", 0, 10);
  ReportAt(ProgressEventReporter::k_start_delay + 10ms, id, "throttled", "d1",
           1, 10);
  ASSERT_EQ(sent_events.size(), 1U);

  // Report an update before the update_internal.
  ReportAt(ProgressEventReporter::k_update_interval - 50ms, id, "throttled",
           "d2", 2, 10);
  ASSERT_EQ(sent_events.size(), 1U);

  // Report an update after the update_interval.
  ReportAt(ProgressEventReporter::k_update_interval * 2, id, "throttled", "d3",
           3, 10);

  // Report end.
  ReportAt(0ms, id, "", "throttled end", 10, 10);

  auto expected_events = ElementsAre(
      Progress("progressStart", HasTitle("throttled")),
      Progress("progressUpdate", HasMessage("d3")),
      Progress("progressEnd", HasId(id), HasMessage("throttled end")));
  ASSERT_THAT(sent_events, expected_events);
}

TEST_F(ProgressEventReporterTest, EndFlushesLatestSnapshotEvenWhenThrottled) {
  const uint32_t id = 32;
  ReportAt(0ms, id, "t", "d0", 0, 10);
  ReportAt(ProgressEventReporter::k_start_delay + 100ms, id, "t", "d1", 1, 10);
  ASSERT_EQ(sent_events.size(), 1U);
  ASSERT_THAT(sent_events[0], Progress("progressStart", HasTitle("t")));

  // Throttled update.
  ReportAt(50ms, id, "", "d2", 2, 10);
  ASSERT_EQ(sent_events.size(), 1U);

  // Report end.
  ReportAt(0ms, id, "t", "final", 10, 10);

  EXPECT_THAT(sent_events,
              ElementsAre(Progress("progressStart"),
                          Progress("progressEnd", HasMessage("final"))));
}

TEST_F(ProgressEventReporterTest, IndeterminateProgressOmitsPercentage) {
  const uint64_t total = UINT64_MAX;
  const std::string title = "Indeterminate";

  ReportAt(0ms, 1, title, "d", 0, total);
  ReportAt(ProgressEventReporter::k_start_delay + 500ms, 1, title, "d", 5,
           total);
  // Report End.
  ReportAt(10ms, 1, "", "end", total, total);

  EXPECT_THAT(sent_events,
              ElementsAre(Progress("progressStart", HasNoPercentage()),
                          Progress("progressEnd", HasMessage("end"))));
}

TEST_F(ProgressEventReporterTest, DeterministicProgressEmitsPercentage) {
  ReportAt(0ms, 1, "t", "d", 0, 4);
  ReportAt(ProgressEventReporter::k_start_delay + 500ms, 1, "t", "d", 2, 4);

  EXPECT_THAT(sent_events,
              ElementsAre(Progress("progressStart", HasPercentage(50))));
}

TEST_F(ProgressEventReporterTest, IndependentIdsAreTrackedSeparately) {
  const uint32_t id_1 = 1;
  const uint32_t id_2 = 2;
  ReportAt(0ms, id_1, "one", "d1", 0, 3);
  ReportAt(0ms, id_2, "two", "d2", 0, 3);
  ReportAt(ProgressEventReporter::k_start_delay + 500ms, id_1, "one", "d1", 3,
           3);
  ReportAt(0ms, id_2, "two", "d2", 3, 3);
  EXPECT_FALSE(m_reporter.HasPending());

  EXPECT_THAT(
      sent_events,
      ElementsAre(Progress("progressStart", HasId(1)),
                  Progress("progressEnd", HasId(id_1), HasMessage("d1")),
                  Progress("progressStart", HasId(id_2)),
                  Progress("progressEnd", HasId(id_2), HasMessage("d2"))));
}

/// A new progress reported without a title is dropped.
/// It cannot reach this state based on the current lldb_private::Progress
/// implementation because `completed` always starts at 0.
TEST_F(ProgressEventReporterTest, NewProgressWithoutTitleIsDropped) {
  m_reporter.Report(/*progress_id=*/1, /*title=*/std::nullopt, /*details=*/"d",
                    /*completed=*/0, /*total=*/3, now);
  EXPECT_FALSE(m_reporter.HasPending());
  EXPECT_THAT(sent_events, IsEmpty());
}

} // namespace
