#include "OmptAssertEvent.h"
#include <omp-tools.h>

#include "gtest/gtest.h"

TEST(CompareOperatorTests, BufferRecordIdentity) {

  using namespace omptest;

  // Default, no time limit or anything
  auto BRDflt = OmptAssertEvent::BufferRecord("dflt", "", ObserveState::always,
                                              ompt_callback_target_submit);

  // Minimum time set, no max time
  auto BRMinSet = OmptAssertEvent::BufferRecord(
      "minset", "", ObserveState::always, ompt_callback_target_submit, 10);

  // Minimum time and maximum time set
  auto BRMinMaxSet =
      OmptAssertEvent::BufferRecord("minmaxset", "", ObserveState::always,
                                    ompt_callback_target_submit, {10, 100});

  ASSERT_EQ(BRDflt, BRDflt);
  ASSERT_EQ(BRMinSet, BRMinSet);
  ASSERT_EQ(BRMinMaxSet, BRMinMaxSet);
}
