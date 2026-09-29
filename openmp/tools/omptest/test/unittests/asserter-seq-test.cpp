#include "OmptAliases.h"
#include "OmptAsserter.h"
#include <omp-tools.h>
#include <sstream>

#include "gtest/gtest.h"

using namespace omptest;
using OAE = omptest::OmptAssertEvent;
using OS = omptest::ObserveState;

/// SequencedAsserter test-fixture class to avoid code duplication among tests.
class OmptSequencedAsserterTest : public testing::Test {
protected:
  OmptSequencedAsserterTest() {
    // Construct default sequenced asserter
    SeqAsserter = std::make_unique<omptest::OmptSequencedAsserter>();

    // Silence all potential log prints
    SeqAsserter->getLog()->setLoggingLevel(logging::Level::Critical);
  }

  std::unique_ptr<omptest::OmptSequencedAsserter> SeqAsserter;
};

TEST_F(OmptSequencedAsserterTest, DefaultState) {
  // Assertion should neither start as 'deactivated' nor 'suspended'
  ASSERT_EQ(SeqAsserter->isActive(), true);
  ASSERT_EQ(SeqAsserter->AssertionSuspended, false);

  // Assertion should begin with event ID zero
  ASSERT_EQ(SeqAsserter->NextEvent, 0);

  // Assertion should begin without previous notifications or assertions
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 0);
  ASSERT_EQ(SeqAsserter->getSuccessfulAssertionCount(), 0);

  // There should be no expected events
  ASSERT_EQ(SeqAsserter->Events.empty(), true);
  ASSERT_EQ(SeqAsserter->getRemainingEventCount(), 0);

  // Default mode should be Strict
  ASSERT_NE(SeqAsserter->getOperationMode(), AssertMode::Relaxed);
  ASSERT_EQ(SeqAsserter->getOperationMode(), AssertMode::Strict);

  // Default state should be passing
  ASSERT_NE(SeqAsserter->getState(), AssertState::Fail);
  ASSERT_EQ(SeqAsserter->getState(), AssertState::Pass);
  ASSERT_NE(SeqAsserter->checkState(), AssertState::Fail);
  ASSERT_EQ(SeqAsserter->checkState(), AssertState::Pass);
}

TEST_F(OmptSequencedAsserterTest, IgnoreNotificationsWhenEmpty) {
  // ParallelBegin events are suppressed by default
  auto SuppressedEvent = OAE::ParallelBegin(
      /*Name=*/"ParBegin", /*Group=*/"", /*Expected=*/OS::Always,
      /*NumThreads=*/3);

  // DeviceFinalize events are not ignored by default
  auto IgnoredEvent = OAE::DeviceFinalize(
      /*Name=*/"DevFini", /*Group=*/"", /*Expected=*/OS::Always,
      /*DeviceNum=*/7);

  // Situation: There is nothing to assert.
  // Result: All notifications are ignored.
  // Hence, check that the perceived count of notifications remains unchanged
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 0);

  SeqAsserter->notify(std::move(SuppressedEvent));

  ASSERT_EQ(SeqAsserter->getNotificationCount(), 0);
  SeqAsserter->notify(std::move(IgnoredEvent));
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 0);
  ASSERT_EQ(SeqAsserter->getSuccessfulAssertionCount(), 0);
  ASSERT_EQ(SeqAsserter->checkState(), AssertState::Pass);
}

TEST_F(OmptSequencedAsserterTest, IgnoreNotificationsWhileDeactivated) {
  auto ExpectedEvent = OAE::DeviceUnload(
      /*Name=*/"DevUnload", /*Group=*/"", /*Expected=*/OS::Always);
  SeqAsserter->insert(std::move(ExpectedEvent));
  ASSERT_EQ(SeqAsserter->Events.empty(), false);

  // Deactivate asserter, effectively ignoring notifications
  SeqAsserter->setActive(false);
  ASSERT_EQ(SeqAsserter->isActive(), false);
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 0);

  // DeviceFinalize events are not ignored by default
  auto IgnoredEvent = OAE::DeviceFinalize(
      /*Name=*/"DevFini", /*Group=*/"", /*Expected=*/OS::Always,
      /*DeviceNum=*/7);
  SeqAsserter->notify(std::move(IgnoredEvent));

  // Assertion was deactivated: No change
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 0);
  ASSERT_EQ(SeqAsserter->getSuccessfulAssertionCount(), 0);

  SeqAsserter->setActive(true);
  ASSERT_EQ(SeqAsserter->isActive(), true);

  auto ObservedEvent = OAE::DeviceUnload(
      /*Name=*/"DevFini", /*Group=*/"", /*Expected=*/OS::Always);
  SeqAsserter->notify(std::move(ObservedEvent));

  // Assertion was activated, one notification expected
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 1);
  ASSERT_EQ(SeqAsserter->getSuccessfulAssertionCount(), 1);
  ASSERT_EQ(SeqAsserter->checkState(), AssertState::Pass);
}

TEST_F(OmptSequencedAsserterTest, AddEvent) {
  ASSERT_EQ(SeqAsserter->getRemainingEventCount(), 0);
  auto ExpectedEvent = OAE::DeviceFinalize(
      /*Name=*/"DevFini", /*Group=*/"", /*Expected=*/OS::Always,
      /*DeviceNum=*/7);
  SeqAsserter->insert(std::move(ExpectedEvent));
  // Sanity check: Notifications should not be triggered
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 0);
  // Adding an expected event must change the event count but not the state
  ASSERT_EQ(SeqAsserter->getRemainingEventCount(), 1);
  ASSERT_EQ(SeqAsserter->getSuccessfulAssertionCount(), 0);
  ASSERT_EQ(SeqAsserter->getState(), AssertState::Pass);
}

TEST_F(OmptSequencedAsserterTest, AddEventIgnoreSuppressed) {
  auto ExpectedEvent = OAE::DeviceFinalize(
      /*Name=*/"DevFini", /*Group=*/"", /*Expected=*/OS::Always,
      /*DeviceNum=*/7);
  SeqAsserter->insert(std::move(ExpectedEvent));
  // ParallelBegin events are suppressed by default
  auto SuppressedEvent = OAE::ParallelBegin(
      /*Name=*/"ParBegin", /*Group=*/"", /*Expected=*/OS::Always,
      /*NumThreads=*/3);
  // Situation: There is one expected event and ParallelBegins are suppressed.
  // Notification count remains unchanged for suppressed events
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 0);
  SeqAsserter->notify(std::move(SuppressedEvent));
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 0);
  ASSERT_EQ(SeqAsserter->getSuccessfulAssertionCount(), 0);
  ASSERT_EQ(SeqAsserter->getState(), AssertState::Pass);
}

TEST_F(OmptSequencedAsserterTest, AddEventObservePass) {
  auto ExpectedEvent = OAE::DeviceFinalize(
      /*Name=*/"DevFini", /*Group=*/"", /*Expected=*/OS::Always,
      /*DeviceNum=*/7);
  SeqAsserter->insert(std::move(ExpectedEvent));
  // DeviceFinalize events are not ignored by default
  auto ObservedEvent = OAE::DeviceFinalize(
      /*Name=*/"DevFini", /*Group=*/"", /*Expected=*/OS::Always,
      /*DeviceNum=*/7);
  SeqAsserter->notify(std::move(ObservedEvent));
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 1);
  ASSERT_EQ(SeqAsserter->getSuccessfulAssertionCount(), 1);
  ASSERT_EQ(SeqAsserter->checkState(), AssertState::Pass);
}

TEST_F(OmptSequencedAsserterTest, AddEventObserveFail) {
  auto ExpectedEvent = OAE::DeviceFinalize(
      /*Name=*/"DevFini", /*Group=*/"", /*Expected=*/OS::Always,
      /*DeviceNum=*/7);
  SeqAsserter->insert(std::move(ExpectedEvent));
  // DeviceFinalize events are not ignored by default
  // Provide wrong DeviceNum
  auto ObservedEvent = OAE::DeviceFinalize(
      /*Name=*/"DevFini", /*Group=*/"", /*Expected=*/OS::Always,
      /*DeviceNum=*/23);

  SeqAsserter->notify(std::move(ObservedEvent));
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 1);
  // Observed and expected event do not match: Fail
  ASSERT_EQ(SeqAsserter->getSuccessfulAssertionCount(), 0);
  ASSERT_EQ(SeqAsserter->checkState(), AssertState::Fail);
}

TEST_F(OmptSequencedAsserterTest, AddEventObserveDifferentType) {
  auto ExpectedEvent = OAE::DeviceUnload(
      /*Name=*/"DevUnload", /*Group=*/"", /*Expected=*/OS::Always);
  SeqAsserter->insert(std::move(ExpectedEvent));
  // DeviceFinalize events are not ignored by default
  auto ObservedEvent = OAE::DeviceFinalize(
      /*Name=*/"DevFini", /*Group=*/"", /*Expected=*/OS::Always,
      /*DeviceNum=*/7);

  SeqAsserter->notify(std::move(ObservedEvent));
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 1);
  // Observed and expected event do not match: Fail
  ASSERT_EQ(SeqAsserter->getSuccessfulAssertionCount(), 0);
  ASSERT_EQ(SeqAsserter->checkState(), AssertState::Fail);
}

TEST_F(OmptSequencedAsserterTest, CheckTargetGroupNoEffect) {
  // Situation: Groups are designed to be used as an indicator -WITHIN- target
  // regions. Hence, comparing two target regions w.r.t. their groups has no
  // effect on pass or fail.

  auto ExpectedEvent = OAE::Target(
      /*Name=*/"Target", /*Group=*/"MyTargetGroup", /*Expected=*/OS::Always,
      /*Kind=*/TARGET, /*Endpoint=*/BEGIN,
      /*DeviceNum=*/7, /*TaskData=*/nullptr, /*TargetId=*/23,
      /*CodeptrRA=*/nullptr);
  SeqAsserter->insert(std::move(ExpectedEvent));
  ASSERT_EQ(SeqAsserter->Events.empty(), false);

  // Deactivate asserter, effectively ignoring notifications
  SeqAsserter->setActive(false);
  ASSERT_EQ(SeqAsserter->isActive(), false);
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 0);

  // Target events are not ignored by default
  auto ObservedEvent = OAE::Target(
      /*Name=*/"Target", /*Group=*/"MyTargetGroup", /*Expected=*/OS::Always,
      /*Kind=*/TARGET, /*Endpoint=*/BEGIN, /*DeviceNum=*/7,
      /*TaskData=*/nullptr, /*TargetId=*/23, /*CodeptrRA=*/nullptr);
  SeqAsserter->notify(std::move(ObservedEvent));

  // Assertion was deactivated: No change
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 0);
  ASSERT_EQ(SeqAsserter->getSuccessfulAssertionCount(), 0);
  ASSERT_EQ(SeqAsserter->getRemainingEventCount(), 1);

  // Re-activate asserter
  SeqAsserter->setActive(true);
  ASSERT_EQ(SeqAsserter->isActive(), true);
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 0);

  // Actually observe a target event from "AnotherGroup"
  auto AnotherObservedEvent = OAE::Target(
      /*Name=*/"Target", /*Group=*/"AnotherGroup", /*Expected=*/OS::Always,
      /*Kind=*/TARGET, /*Endpoint=*/BEGIN, /*DeviceNum=*/7,
      /*TaskData=*/nullptr, /*TargetId=*/23, /*CodeptrRA=*/nullptr);
  SeqAsserter->notify(std::move(AnotherObservedEvent));

  // Observed all expected events; groups of target regions do not affect pass
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 1);
  ASSERT_EQ(SeqAsserter->getSuccessfulAssertionCount(), 1);
  ASSERT_EQ(SeqAsserter->getRemainingEventCount(), 0);

  ASSERT_EQ(SeqAsserter->checkState(), AssertState::Pass);
}

TEST_F(OmptSequencedAsserterTest, CheckSyncPoint) {
  auto ExpectedEvent = OAE::Target(
      /*Name=*/"Target", /*Group=*/"MyTargetGroup", /*Expected=*/OS::Always,
      /*Kind=*/TARGET, /*Endpoint=*/BEGIN,
      /*DeviceNum=*/7, /*TaskData=*/nullptr, /*TargetId=*/23,
      /*CodeptrRA=*/nullptr);
  SeqAsserter->insert(std::move(ExpectedEvent));
  ASSERT_EQ(SeqAsserter->Events.empty(), false);
  ASSERT_EQ(SeqAsserter->getRemainingEventCount(), 1);

  // Target events are not ignored by default
  auto ObservedEvent = OAE::Target(
      /*Name=*/"Target", /*Group=*/"MyTargetGroup", /*Expected=*/OS::Always,
      /*Kind=*/TARGET, /*Endpoint=*/BEGIN, /*DeviceNum=*/7,
      /*TaskData=*/nullptr, /*TargetId=*/23, /*CodeptrRA=*/nullptr);
  SeqAsserter->notify(std::move(ObservedEvent));
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 1);

  SeqAsserter->notify(OAE::AssertionSyncPoint(
      /*Name=*/"", /*Group=*/"", /*Expected=*/OS::Always,
      /*SyncPointName=*/"SyncPoint 1"));
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 2);
  ASSERT_EQ(SeqAsserter->getSuccessfulAssertionCount(), 1);

  // All events processed: SyncPoint "passes"
  ASSERT_EQ(SeqAsserter->checkState(), AssertState::Pass);

  auto AnotherExpectedEvent = OAE::Target(
      /*Name=*/"Target", /*Group=*/"MyTargetGroup", /*Expected=*/OS::Always,
      /*Kind=*/TARGET, /*Endpoint=*/BEGIN,
      /*DeviceNum=*/7, /*TaskData=*/nullptr, /*TargetId=*/23,
      /*CodeptrRA=*/nullptr);

  ASSERT_EQ(SeqAsserter->getRemainingEventCount(), 0);
  SeqAsserter->insert(std::move(AnotherExpectedEvent));
  ASSERT_EQ(SeqAsserter->getRemainingEventCount(), 1);

  // Remaining events present: SyncPoint "fails"
  SeqAsserter->notify(OAE::AssertionSyncPoint(
      /*Name=*/"", /*Group=*/"", /*Expected=*/OS::Always,
      /*SyncPointName=*/"SyncPoint 2"));
  ASSERT_EQ(SeqAsserter->checkState(), AssertState::Fail);
}

TEST_F(OmptSequencedAsserterTest, CheckExcessNotify) {
  auto ExpectedEvent = OAE::Target(
      /*Name=*/"Target", /*Group=*/"MyTargetGroup", /*Expected=*/OS::Always,
      /*Kind=*/TARGET, /*Endpoint=*/BEGIN,
      /*DeviceNum=*/7, /*TaskData=*/nullptr, /*TargetId=*/23,
      /*CodeptrRA=*/nullptr);
  SeqAsserter->insert(std::move(ExpectedEvent));
  ASSERT_EQ(SeqAsserter->Events.empty(), false);
  ASSERT_EQ(SeqAsserter->getRemainingEventCount(), 1);

  // Target events are not ignored by default
  auto ObservedEvent = OAE::Target(
      /*Name=*/"Target", /*Group=*/"MyTargetGroup", /*Expected=*/OS::Always,
      /*Kind=*/TARGET, /*Endpoint=*/BEGIN, /*DeviceNum=*/7,
      /*TaskData=*/nullptr, /*TargetId=*/23, /*CodeptrRA=*/nullptr);
  SeqAsserter->notify(std::move(ObservedEvent));
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 1);

  // All events processed: pass
  ASSERT_EQ(SeqAsserter->checkState(), AssertState::Pass);

  // Target events are not ignored by default
  auto AnotherObservedEvent = OAE::Target(
      /*Name=*/"Target", /*Group=*/"MyTargetGroup", /*Expected=*/OS::Always,
      /*Kind=*/TARGET, /*Endpoint=*/BEGIN, /*DeviceNum=*/7,
      /*TaskData=*/nullptr, /*TargetId=*/23, /*CodeptrRA=*/nullptr);

  // No more events expected: notify "fails"
  SeqAsserter->notify(std::move(AnotherObservedEvent));
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 2);
  ASSERT_EQ(SeqAsserter->checkState(), AssertState::Fail);
}

TEST_F(OmptSequencedAsserterTest, CheckSuspend) {
  SeqAsserter->insert(OAE::AssertionSuspend(
      /*Name=*/"", /*Group=*/"", /*Expected=*/OS::Never));
  ASSERT_EQ(SeqAsserter->Events.empty(), false);

  // Being notified while the next expected event is a "suspend" should change
  // the asserter's state
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 0);
  ASSERT_EQ(SeqAsserter->AssertionSuspended, false);
  SeqAsserter->notify(OAE::DeviceFinalize(
      /*Name=*/"DevFini", /*Group=*/"", /*Expected=*/OS::Always,
      /*DeviceNum=*/7));
  ASSERT_EQ(SeqAsserter->AssertionSuspended, true);
  ASSERT_EQ(SeqAsserter->getNotificationCount(), 1);

  ASSERT_EQ(SeqAsserter->getSuccessfulAssertionCount(), 0);
  ASSERT_EQ(SeqAsserter->checkState(), AssertState::Pass);

  auto ExpectedEvent = OAE::Target(
      /*Name=*/"Target", /*Group=*/"MyTargetGroup", /*Expected=*/OS::Always,
      /*Kind=*/TARGET, /*Endpoint=*/BEGIN,
      /*DeviceNum=*/7, /*TaskData=*/nullptr, /*TargetId=*/23,
      /*CodeptrRA=*/nullptr);
  SeqAsserter->insert(std::move(ExpectedEvent));

  // Being notified with an observed event, which matches the next expected
  // event, resumes assertion (suspended = false)
  ASSERT_EQ(SeqAsserter->AssertionSuspended, true);
  SeqAsserter->notify(OAE::Target(
      /*Name=*/"Target", /*Group=*/"MyTargetGroup", /*Expected=*/OS::Always,
      /*Kind=*/TARGET, /*Endpoint=*/BEGIN,
      /*DeviceNum=*/7, /*TaskData=*/nullptr, /*TargetId=*/23,
      /*CodeptrRA=*/nullptr));
  ASSERT_EQ(SeqAsserter->AssertionSuspended, false);

  ASSERT_EQ(SeqAsserter->getNotificationCount(), 2);
  ASSERT_EQ(SeqAsserter->getSuccessfulAssertionCount(), 1);
  ASSERT_EQ(SeqAsserter->checkState(), AssertState::Pass);
}
