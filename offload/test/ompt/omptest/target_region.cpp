// clang-format off
// RUN: %libomptarget-compilexx-generic %omptest-flags && %libomptarget-run-generic | %fcheck-generic
// REQUIRES: omptest
// REQUIRES: gpu
// clang-format on

// Minimal end-to-end check of omptest integration: (i) the tool attaches,
// OMPT callbacks reach the asserters, assertions can fail, and
// test cases run. Checks non-EMI target callbacks only.

#include "OmptTester.h"

using namespace omptest;

// Tests are defined in the GoogleTest way.
// omptest provides the additional macros to formulate assertions on the
// sequence of OMPT callbacks observed by the tool.
// Thus, we establish a series of events that need to be observed from the
// OpenMP runtime.

TEST(OmptestIntegration, TargetRegion) {
  // The runtime emits device-init/load and data-op callbacks before the
  // target region; strict mode would fail on those, so relax the asserter.
  OMPT_ASSERT_SEQUENCE_MODE_RELAXED()
  // Required events: TearDown fails the case if either is never observed,
  // so a pass here proves the tool attached and callbacks reached it.
  OMPT_ASSERT_SEQUENCE(Target, TARGET, BEGIN)
  OMPT_ASSERT_SEQUENCE(Target, TARGET, END)
  // Stop checking after END; callbacks the runtime emits afterwards would
  // otherwise fail the test as unexpected extra events.
  OMPT_ASSERT_SEQUENCE_SUSPEND()
#pragma omp target
  {
  }
}

// Negative control: proves that a failing assertion actually fails the
// test. A plain target region never emits an exit-data event, so the
// event below is never matched, the sequence asserter fails, and
// TEST_XFAIL requires exactly that failure in order to pass.
TEST_XFAIL(OmptestIntegrationXFail, NegativeControl) {
  OMPT_ASSERT_SEQUENCE_MODE_RELAXED()
  OMPT_ASSERT_SEQUENCE(Target, ompt_target_exit_data, BEGIN)
#pragma omp target
  {
  }
}

OMPTEST_TESTSUITE_MAIN()

// Guards against a vacuous pass: GoogleTest still exits 0 and prints
// "0 tests" if no case is registered or run.
// CHECK: [  PASSED  ] 2 tests.
