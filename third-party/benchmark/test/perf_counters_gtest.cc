#include <random>
#include <thread>

#include "../src/perf_counters.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#ifndef GTEST_SKIP
struct MsgHandler {
  void operator=(std::ostream&) {}
};
#define GTEST_SKIP() return MsgHandler() = std::cout
#endif

using benchmark::internal::PerfCounters;
using benchmark::internal::PerfCountersMeasurement;
using benchmark::internal::PerfCounterValues;
using ::testing::AllOf;
using ::testing::Gt;
using ::testing::Lt;

namespace {
const char kGenericPerfEvent1[] = "CYCLES";
const char kGenericPerfEvent2[] = "INSTRUCTIONS";

TEST(PerfCountersTest, Init) {
  EXPECT_EQ(PerfCounters::Initialize(), PerfCounters::kSupported);
}

TEST(PerfCountersTest, OneCounter) {
  if (!PerfCounters::kSupported) {
    GTEST_SKIP() << "Performance counters not supported.\n";
  }
  EXPECT_TRUE(PerfCounters::Initialize());
  EXPECT_EQ(PerfCounters::Create({kGenericPerfEvent1}).num_counters(), 1);
}

TEST(PerfCountersTest, NegativeTest) {
  if (!PerfCounters::kSupported) {
    EXPECT_FALSE(PerfCounters::Initialize());
    return;
  }
  EXPECT_TRUE(PerfCounters::Initialize());
  // Safety checks
  // Create() will always create a valid object, even if passed no or
  // wrong arguments as the new behavior is to warn and drop unsupported
  // counters
  EXPECT_EQ(PerfCounters::Create({}).num_counters(), 0);
  EXPECT_EQ(PerfCounters::Create({""}).num_counters(), 0);
  EXPECT_EQ(PerfCounters::Create({"not a counter name"}).num_counters(), 0);
  {
    // Try sneaking in a bad egg to see if it is filtered out. The
    // number of counters has to be two, not zero
    auto counter =
        PerfCounters::Create({kGenericPerfEvent2, "", kGenericPerfEvent1});
    EXPECT_EQ(counter.num_counters(), 2);
    EXPECT_EQ(counter.names(), std::vector<std::string>(
                                   {kGenericPerfEvent2, kGenericPerfEvent1}));
  }
  {
    // Try sneaking in an outrageous counter, like a fat finger mistake
    auto counter = PerfCounters::Create(
        {kGenericPerfEvent2, "not a counter name", kGenericPerfEvent1});
    EXPECT_EQ(counter.num_counters(), 2);
    EXPECT_EQ(counter.names(), std::vector<std::string>(
                                   {kGenericPerfEvent2, kGenericPerfEvent1}));
  }
  {
    // Finally try a golden input - it should like both of them
    EXPECT_EQ(PerfCounters::Create({kGenericPerfEvent1, kGenericPerfEvent2})
                  .num_counters(),
              2);
  }
  {
    // Add a bad apple in the end of the chain to check the edges
    auto counter = PerfCounters::Create(
        {kGenericPerfEvent1, kGenericPerfEvent2, "bad event name"});
    EXPECT_EQ(counter.num_counters(), 2);
    EXPECT_EQ(counter.names(), std::vector<std::string>(
                                   {kGenericPerfEvent1, kGenericPerfEvent2}));
  }
}

TEST(PerfCountersTest, Read1Counter) {
  if (!PerfCounters::kSupported) {
    GTEST_SKIP() << "Test skipped because libpfm is not supported.\n";
  }
  EXPECT_TRUE(PerfCounters::Initialize());
  auto counters = PerfCounters::Create({kGenericPerfEvent1});
  EXPECT_EQ(counters.num_counters(), 1);
  PerfCounterValues values1(1);
  EXPECT_TRUE(counters.Snapshot(&values1));
  EXPECT_GT(values1[0], 0);
  PerfCounterValues values2(1);
  EXPECT_TRUE(counters.Snapshot(&values2));
  EXPECT_GT(values2[0], 0);
  EXPECT_GT(values2[0], values1[0]);
}

TEST(PerfCountersTest, Read2Counters) {
  if (!PerfCounters::kSupported) {
    GTEST_SKIP() << "Test skipped because libpfm is not supported.\n";
  }
  EXPECT_TRUE(PerfCounters::Initialize());
  auto counters =
      PerfCounters::Create({kGenericPerfEvent1, kGenericPerfEvent2});
  EXPECT_EQ(counters.num_counters(), 2);
  PerfCounterValues values1(2);
  EXPECT_TRUE(counters.Snapshot(&values1));
  EXPECT_GT(values1[0], 0);
  EXPECT_GT(values1[1], 0);
  PerfCounterValues values2(2);
  EXPECT_TRUE(counters.Snapshot(&values2));
  EXPECT_GT(values2[0], 0);
  EXPECT_GT(values2[1], 0);
}

TEST(PerfCountersTest, ReopenExistingCounters) {
  // This test works in recent and old Intel hardware, Pixel 3, and Pixel 6.
  // However we cannot make assumptions beyond 2 HW counters due to Pixel 6.
  if (!PerfCounters::kSupported) {
    GTEST_SKIP() << "Test skipped because libpfm is not supported.\n";
  }
  EXPECT_TRUE(PerfCounters::Initialize());
  std::vector<std::string> kMetrics({kGenericPerfEvent1});
  std::vector<PerfCounters> counters(2);
  for (auto& counter : counters) {
    counter = PerfCounters::Create(kMetrics);
  }
  PerfCounterValues values(1);
  EXPECT_TRUE(counters[0].Snapshot(&values));
  EXPECT_TRUE(counters[1].Snapshot(&values));
}

TEST(PerfCountersTest, CreateExistingMeasurements) {
  // The test works (i.e. causes read to fail) for the assumptions
  // about hardware capabilities (i.e. small number (2) hardware
  // counters) at this date,
  // the same as previous test ReopenExistingCounters.
  if (!PerfCounters::kSupported) {
    GTEST_SKIP() << "Test skipped because libpfm is not supported.\n";
  }
  EXPECT_TRUE(PerfCounters::Initialize());

  // This means we will try 10 counters but we can only guarantee
  // for sure at this time that only 3 will work. Perhaps in the future
  // we could use libpfm to query for the hardware limits on this
  // particular platform.
  const int kMaxCounters = 10;
  const int kMinValidCounters = 2;

  // Let's use a ubiquitous counter that is guaranteed to work
  // on all platforms
  const std::vector<std::string> kMetrics{"cycles"};

  // Cannot create a vector of actual objects because the
  // copy constructor of PerfCounters is deleted - and so is
  // implicitly deleted on PerfCountersMeasurement too
  std::vector<std::unique_ptr<PerfCountersMeasurement>>
      perf_counter_measurements;

  perf_counter_measurements.reserve(kMaxCounters);
  for (int j = 0; j < kMaxCounters; ++j) {
    perf_counter_measurements.emplace_back(
        new PerfCountersMeasurement(kMetrics));
  }

  std::vector<std::pair<std::string, double>> measurements;

  // Start all counters together to see if they hold
  size_t max_counters = kMaxCounters;
  for (size_t i = 0; i < kMaxCounters; ++i) {
    auto& counter(*perf_counter_measurements[i]);
    EXPECT_EQ(counter.num_counters(), 1);
    if (!counter.Start()) {
      max_counters = i;
      break;
    };
  }

  ASSERT_GE(max_counters, kMinValidCounters);

  // Start all together
  for (size_t i = 0; i < max_counters; ++i) {
    auto& counter(*perf_counter_measurements[i]);
    EXPECT_TRUE(counter.Stop(measurements) || (i >= kMinValidCounters));
  }

  // Start/stop individually
  for (size_t i = 0; i < max_counters; ++i) {
    auto& counter(*perf_counter_measurements[i]);
    measurements.clear();
    counter.Start();
    EXPECT_TRUE(counter.Stop(measurements) || (i >= kMinValidCounters));
  }
}

// We try to do some meaningful work here but the compiler
// insists in optimizing away our loop so we had to add a
// no-optimize macro. In case it fails, we added some entropy
// to this pool as well.

BENCHMARK_DONT_OPTIMIZE size_t do_work() {
  static std::mt19937 rd{std::random_device{}()};
  static std::uniform_int_distribution<size_t> mrand(0, 10);
  const size_t kNumLoops = 1000000;
  size_t sum = 0;
  for (size_t j = 0; j < kNumLoops; ++j) {
    sum += mrand(rd);
  }
  benchmark::DoNotOptimize(sum);
  return sum;
}

void measure(size_t threadcount, PerfCounterValues* before,
             PerfCounterValues* after) {
  BM_CHECK_NE(before, nullptr);
  BM_CHECK_NE(after, nullptr);
  std::vector<std::thread> threads(threadcount);
  auto work = [&]() { BM_CHECK(do_work() > 1000); };

  // We need to first set up the counters, then start the threads, so the
  // threads would inherit the counters. But later, we need to first destroy
  // the thread pool (so all the work finishes), then measure the counters. So
  // the scopes overlap, and we need to explicitly control the scope of the
  // threadpool.
  auto counters =
      PerfCounters::Create({kGenericPerfEvent1, kGenericPerfEvent2});
  for (auto& t : threads) t = std::thread(work);
  counters.Snapshot(before);
  for (auto& t : threads) t.join();
  counters.Snapshot(after);
}

TEST(PerfCountersTest, MultiThreaded) {
  if (!PerfCounters::kSupported) {
    GTEST_SKIP() << "Test skipped because libpfm is not supported.";
  }
  EXPECT_TRUE(PerfCounters::Initialize());
  PerfCounterValues before(2);
  PerfCounterValues after(2);

  // Notice that this test will work even if we taskset it to a single CPU
  // In this case the threads will run sequentially
  // Start two threads and measure the number of combined cycles and
  // instructions
  measure(2, &before, &after);
  std::vector<double> Elapsed2Threads{
      static_cast<double>(after[0] - before[0]),
      static_cast<double>(after[1] - before[1])};

  // Start four threads and measure the number of combined cycles and
  // instructions
  measure(4, &before, &after);
  std::vector<double> Elapsed4Threads{
      static_cast<double>(after[0] - before[0]),
      static_cast<double>(after[1] - before[1])};

  // The following expectations fail (at least on a beefy workstation with lots
  // of cpus) - it seems that in some circumstances the runtime of 4 threads
  // can even be better than with 2.
  // So instead of expecting 4 threads to be slower, let's just make sure they
  // do not differ too much in general (one is not more than 10x than the
  // other).
  EXPECT_THAT(Elapsed4Threads[0] / Elapsed2Threads[0], AllOf(Gt(0.1), Lt(10)));
  EXPECT_THAT(Elapsed4Threads[1] / Elapsed2Threads[1], AllOf(Gt(0.1), Lt(10)));
}

TEST(PerfCountersTest, HardwareLimits) {
  // The test works (i.e. causes read to fail) for the assumptions
  // about hardware capabilities (i.e. small number (3-4) hardware
  // counters) at this date,
  // the same as previous test ReopenExistingCounters.
  if (!PerfCounters::kSupported) {
    GTEST_SKIP() << "Test skipped because libpfm is not supported.\n";
  }
  EXPECT_TRUE(PerfCounters::Initialize());

  // Taken from `perf list`, but focusses only on those HW events that actually
  // were reported when running `sudo perf stat -a sleep 10`, intersected over
  // several platforms. All HW events listed in the first command not reported
  // in the second seem to not work. This is sad as we don't really get to test
  // the grouping here (groups can contain up to 6 members)...
  std::vector<std::string> counter_names{
      "cycles",         // leader
      "instructions",   //
      "branch-misses",  //
  };

  // In the off-chance that some of these values are not supported,
  // we filter them out so the test will complete without failure
  // albeit it might not actually test the grouping on that platform
  std::vector<std::string> valid_names;
  for (const std::string& name : counter_names) {
    if (PerfCounters::IsCounterSupported(name)) {
      valid_names.push_back(name);
    }
  }
  PerfCountersMeasurement counter(valid_names);

  std::vector<std::pair<std::string, double>> measurements;

  counter.Start();
  EXPECT_TRUE(counter.Stop(measurements));
}

}  // namespace
