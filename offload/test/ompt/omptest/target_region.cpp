// clang-format off
// RUN: %libomptarget-compilexx-generic %omptest-flags && %libomptarget-run-generic | %fcheck-generic
// REQUIRES: omptest
// REQUIRES: gpu
// clang-format on

#include "OmptTester.h"

using namespace omptest;

TEST(OmptestIntegration, TargetRegion) {
  OMPT_ASSERT_SEQUENCE_MODE_RELAXED()
  OMPT_ASSERT_SEQUENCE(Target, TARGET, BEGIN)
  OMPT_ASSERT_SEQUENCE(Target, TARGET, END)
  OMPT_ASSERT_SEQUENCE_SUSPEND()
#pragma omp target
  {
  }
}

TEST_XFAIL(OmptestIntegrationXFail, NegativeControl) {
  OMPT_ASSERT_SEQUENCE_MODE_RELAXED()
  OMPT_ASSERT_SEQUENCE(Target, ompt_target_exit_data, BEGIN)
#pragma omp target
  {
  }
}

OMPTEST_TESTSUITE_MAIN()

// CHECK: [  PASSED  ] 2 tests.
