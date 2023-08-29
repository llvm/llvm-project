#include "../include/OmptTester.h"

/// Example definition on how to write a test case
OMPTTESTCASE(TestSuiteOne, ShowCaseTest) {
  /* The Test Body */
  int arr[10] = {0};

  /// OMPT_SEQ_ASSERT are meant to occur in order
  OMPT_SEQ_ASSERT(DataAlloc, 10, int, arr, "NoUSM")
  // OmptSequenceAsserter.insert(OmptAssertEvent::DataAlloc(&arr, 10,
  // sizeof(int), "NoUSM"));
  OMPT_SEQ_ASSERT_NOT(DataAlloc, 1, int, arr, "NoUSM")
  OMPT_SEQ_ASSERT(DataMap, H2D, 10, int, arr, "NoUSM")
  // OmptSequenceAsserter.insert(OmptAssertEvent::DataMap(H2D, &arr, 10,
  // sizeof(int), "NoUSM"));
  OMPT_SEQ_ASSERT(KernelLaunch, SGN::SPMD, "NoUSM, USM")
  // OmptSequenceAsserter.insert(OmptAssertEvent::KernelLaunch(SGN::SPMD,
  // "NoUSM, USM"));

  /// OMPT_EVENT_ASSERT are meant to just occur (in no specified order)
  OMPT_EVENT_ASSERT("MyId", DataMap, H2D, "asd")
}

OMPTTESTCASE(ManualSuite, ParallelFor) {
  /* The Test Body */
  int arr[10] = {0};
  SequenceAsserter.insert(omptest::OmptAssertEvent::ParallelBegin(
      /*NumThreads=*/2, "User Parallel Begin"));
  SequenceAsserter.insert(
      omptest::OmptAssertEvent::ThreadBegin("User Thread Begin"));
  SequenceAsserter.insert(
      omptest::OmptAssertEvent::ParallelEnd("User Parallel End"));

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;
}

int main(int argc, char **argv) {
  std::cout << "Starting" << std::endl;

  Runner Run;
  Run.run();

  std::cout << "Ending" << std::endl;
  return 0;
}