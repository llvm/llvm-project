
#undef NDEBUG

#include "benchmark/benchmark.h"
#include "output_test.h"

// ========================================================================= //
// ---------------------- Testing Prologue Output -------------------------- //
// ========================================================================= //

// clang-format off

ADD_CASES(TC_ConsoleOut,
          {{"^[-]+$", MR_Next},
           {"^Benchmark %s Time %s CPU %s Iterations UserCounters...$", MR_Next},
           {"^[-]+$", MR_Next}});
ADD_CASES(TC_CSVOut, {{"%csv_header,\"bar\",\"foo\""}});

// clang-format on

// ========================================================================= //
// ------------------------- Simple Counters Output ------------------------ //
// ========================================================================= //

namespace {
void BM_Counters_Simple(benchmark::State& state) {
  for (auto _ : state) {
  }
  state.counters["foo"] = 1;
  state.counters["bar"] = 2 * static_cast<double>(state.iterations());
}
BENCHMARK(BM_Counters_Simple)->ThreadRange(1, 8);
ADD_CASES(TC_ConsoleOut, {{"^BM_Counters_Simple/threads:%int %console_report "
                           "bar=%hrfloat foo=%hrfloat$"}});
ADD_CASES(TC_JSONOut,
          {{"\"name\": \"BM_Counters_Simple/threads:%int\",$"},
           {"\"family_index\": 0,$", MR_Next},
           {"\"per_family_instance_index\": 0,$", MR_Next},
           {"\"run_name\": \"BM_Counters_Simple/threads:%int\",$", MR_Next},
           {"\"run_type\": \"iteration\",$", MR_Next},
           {"\"repetitions\": 1,$", MR_Next},
           {"\"repetition_index\": 0,$", MR_Next},
           {"\"threads\": %int,$", MR_Next},
           {"\"iterations\": %int,$", MR_Next},
           {"\"real_time\": %float,$", MR_Next},
           {"\"cpu_time\": %float,$", MR_Next},
           {"\"time_unit\": \"ns\",$", MR_Next},
           {"\"bar\": %float,$", MR_Next},
           {"\"foo\": %float$", MR_Next},
           {"}", MR_Next}});
ADD_CASES(
    TC_CSVOut,
    {{"^\"BM_Counters_Simple/threads:%int\",%csv_report,%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckSimple(Results const& e) {
  double its = e.NumIterations();
  CHECK_COUNTER_VALUE(e, int, "foo", EQ, 1 * e.NumThreads());
  // check that the value of bar is within 0.1% of the expected value
  CHECK_FLOAT_COUNTER_VALUE(e, "bar", EQ, 2. * its, 0.001);
}
CHECK_BENCHMARK_RESULTS("BM_Counters_Simple/threads:%int", &CheckSimple);
}  // end namespace

// ========================================================================= //
// --------------------- Counters+Items+Bytes/s Output --------------------- //
// ========================================================================= //

namespace {
void BM_Counters_WithBytesAndItemsPSec(benchmark::State& state) {
  for (auto _ : state) {
    // This test requires a non-zero CPU time to avoid divide-by-zero
    auto iterations = static_cast<double>(state.iterations()) *
                      static_cast<double>(state.iterations());
    benchmark::DoNotOptimize(iterations);
  }
  state.counters["foo"] = 1;
  state.SetBytesProcessed(364);
  state.SetItemsProcessed(150);
}
BENCHMARK(BM_Counters_WithBytesAndItemsPSec)->ThreadRange(1, 8);
ADD_CASES(TC_ConsoleOut,
          {{"^BM_Counters_WithBytesAndItemsPSec/threads:%int %console_report "
            "bytes_per_second=%hrfloat/s "
            "foo=%hrfloat items_per_second=%hrfloat/s$"}});
ADD_CASES(
    TC_JSONOut,
    {{"\"name\": \"BM_Counters_WithBytesAndItemsPSec/threads:%int\",$"},
     {"\"family_index\": 1,$", MR_Next},
     {"\"per_family_instance_index\": 0,$", MR_Next},
     {"\"run_name\": \"BM_Counters_WithBytesAndItemsPSec/threads:%int\",$",
      MR_Next},
     {"\"run_type\": \"iteration\",$", MR_Next},
     {"\"repetitions\": 1,$", MR_Next},
     {"\"repetition_index\": 0,$", MR_Next},
     {"\"threads\": %int,$", MR_Next},
     {"\"iterations\": %int,$", MR_Next},
     {"\"real_time\": %float,$", MR_Next},
     {"\"cpu_time\": %float,$", MR_Next},
     {"\"time_unit\": \"ns\",$", MR_Next},
     {"\"bytes_per_second\": %float,$", MR_Next},
     {"\"foo\": %float,$", MR_Next},
     {"\"items_per_second\": %float$", MR_Next},
     {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_Counters_WithBytesAndItemsPSec/threads:%int\","
                       "%csv_bytes_items_report,,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckBytesAndItemsPSec(Results const& e) {
  // this (and not real time) is the time used
  double t = e.DurationCPUTime() / e.NumThreads();
  CHECK_COUNTER_VALUE(e, int, "foo", EQ, 1 * e.NumThreads());
  // check that the values are within 0.1% of the expected values
  CHECK_FLOAT_RESULT_VALUE(e, "bytes_per_second", EQ,
                           (364. * e.NumThreads()) / t, 0.001);
  CHECK_FLOAT_RESULT_VALUE(e, "items_per_second", EQ,
                           (150. * e.NumThreads()) / t, 0.001);
}
CHECK_BENCHMARK_RESULTS("BM_Counters_WithBytesAndItemsPSec/threads:%int",
                        &CheckBytesAndItemsPSec);
}  // end namespace

// ========================================================================= //
// ------------------------- Rate Counters Output -------------------------- //
// ========================================================================= //
namespace {
void BM_Counters_Rate(benchmark::State& state) {
  for (auto _ : state) {
    // This test requires a non-zero CPU time to avoid divide-by-zero
    auto iterations = static_cast<double>(state.iterations()) *
                      static_cast<double>(state.iterations());
    benchmark::DoNotOptimize(iterations);
  }
  namespace bm = benchmark;
  state.counters["foo"] = bm::Counter{1, bm::Counter::kIsRate};
  state.counters["bar"] = bm::Counter{2, bm::Counter::kIsRate};
}
BENCHMARK(BM_Counters_Rate)->ThreadRange(1, 8);
ADD_CASES(TC_ConsoleOut, {{"^BM_Counters_Rate/threads:%int %console_report "
                           "bar=%hrfloat/s foo=%hrfloat/s$"}});
ADD_CASES(TC_JSONOut,
          {{"\"name\": \"BM_Counters_Rate/threads:%int\",$"},
           {"\"family_index\": 2,$", MR_Next},
           {"\"per_family_instance_index\": 0,$", MR_Next},
           {"\"run_name\": \"BM_Counters_Rate/threads:%int\",$", MR_Next},
           {"\"run_type\": \"iteration\",$", MR_Next},
           {"\"repetitions\": 1,$", MR_Next},
           {"\"repetition_index\": 0,$", MR_Next},
           {"\"threads\": %int,$", MR_Next},
           {"\"iterations\": %int,$", MR_Next},
           {"\"real_time\": %float,$", MR_Next},
           {"\"cpu_time\": %float,$", MR_Next},
           {"\"time_unit\": \"ns\",$", MR_Next},
           {"\"bar\": %float,$", MR_Next},
           {"\"foo\": %float$", MR_Next},
           {"}", MR_Next}});
ADD_CASES(TC_CSVOut,
          {{"^\"BM_Counters_Rate/threads:%int\",%csv_report,%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckRate(Results const& e) {
  // this (and not real time) is the time used
  double t = e.DurationCPUTime() / e.NumThreads();
  // check that the values are within 0.1% of the expected values
  CHECK_FLOAT_COUNTER_VALUE(e, "foo", EQ, (1. * e.NumThreads()) / t, 0.001);
  CHECK_FLOAT_COUNTER_VALUE(e, "bar", EQ, (2. * e.NumThreads()) / t, 0.001);
}
CHECK_BENCHMARK_RESULTS("BM_Counters_Rate/threads:%int", &CheckRate);
}  // end namespace

// ========================================================================= //
// ----------------------- Inverted Counters Output ------------------------ //
// ========================================================================= //

namespace {
void BM_Invert(benchmark::State& state) {
  for (auto _ : state) {
    // This test requires a non-zero CPU time to avoid divide-by-zero
    auto iterations = static_cast<double>(state.iterations()) *
                      static_cast<double>(state.iterations());
    benchmark::DoNotOptimize(iterations);
  }
  namespace bm = benchmark;
  state.counters["foo"] = bm::Counter{0.0001, bm::Counter::kInvert};
  state.counters["bar"] = bm::Counter{10000, bm::Counter::kInvert};
}
BENCHMARK(BM_Invert)->ThreadRange(1, 8);
ADD_CASES(
    TC_ConsoleOut,
    {{"^BM_Invert/threads:%int %console_report bar=%hrfloatu foo=%hrfloatk$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_Invert/threads:%int\",$"},
                       {"\"family_index\": 3,$", MR_Next},
                       {"\"per_family_instance_index\": 0,$", MR_Next},
                       {"\"run_name\": \"BM_Invert/threads:%int\",$", MR_Next},
                       {"\"run_type\": \"iteration\",$", MR_Next},
                       {"\"repetitions\": 1,$", MR_Next},
                       {"\"repetition_index\": 0,$", MR_Next},
                       {"\"threads\": %int,$", MR_Next},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\",$", MR_Next},
                       {"\"bar\": %float,$", MR_Next},
                       {"\"foo\": %float$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut,
          {{"^\"BM_Invert/threads:%int\",%csv_report,%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckInvert(Results const& e) {
  CHECK_FLOAT_COUNTER_VALUE(e, "foo", EQ, 1. / (0.0001 * e.NumThreads()),
                            0.0001);
  CHECK_FLOAT_COUNTER_VALUE(e, "bar", EQ, 1. / (10000 * e.NumThreads()),
                            0.0001);
}
CHECK_BENCHMARK_RESULTS("BM_Invert/threads:%int", &CheckInvert);
}  // end namespace

// ========================================================================= //
// --------------------- InvertedRate Counters Output ---------------------- //
// ========================================================================= //

namespace {
void BM_Counters_InvertedRate(benchmark::State& state) {
  for (auto _ : state) {
    // This test requires a non-zero CPU time to avoid divide-by-zero
    auto iterations = static_cast<double>(state.iterations()) *
                      static_cast<double>(state.iterations());
    benchmark::DoNotOptimize(iterations);
  }
  namespace bm = benchmark;
  state.counters["foo"] =
      bm::Counter{1, bm::Counter::kIsRate | bm::Counter::kInvert};
  state.counters["bar"] =
      bm::Counter{8192, bm::Counter::kIsRate | bm::Counter::kInvert};
}
BENCHMARK(BM_Counters_InvertedRate)->ThreadRange(1, 8);
ADD_CASES(TC_ConsoleOut,
          {{"^BM_Counters_InvertedRate/threads:%int %console_report "
            "bar=%hrfloats foo=%hrfloats$"}});
ADD_CASES(TC_JSONOut,
          {{"\"name\": \"BM_Counters_InvertedRate/threads:%int\",$"},
           {"\"family_index\": 4,$", MR_Next},
           {"\"per_family_instance_index\": 0,$", MR_Next},
           {"\"run_name\": \"BM_Counters_InvertedRate/threads:%int\",$",
            MR_Next},
           {"\"run_type\": \"iteration\",$", MR_Next},
           {"\"repetitions\": 1,$", MR_Next},
           {"\"repetition_index\": 0,$", MR_Next},
           {"\"threads\": %int,$", MR_Next},
           {"\"iterations\": %int,$", MR_Next},
           {"\"real_time\": %float,$", MR_Next},
           {"\"cpu_time\": %float,$", MR_Next},
           {"\"time_unit\": \"ns\",$", MR_Next},
           {"\"bar\": %float,$", MR_Next},
           {"\"foo\": %float$", MR_Next},
           {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_Counters_InvertedRate/"
                       "threads:%int\",%csv_report,%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckInvertedRate(Results const& e) {
  // this (and not real time) is the time used
  double t = e.DurationCPUTime() / e.NumThreads();
  // check that the values are within 0.1% of the expected values
  CHECK_FLOAT_COUNTER_VALUE(e, "foo", EQ, t / (e.NumThreads()), 0.001);
  CHECK_FLOAT_COUNTER_VALUE(e, "bar", EQ, t / (8192.0 * e.NumThreads()), 0.001);
}
CHECK_BENCHMARK_RESULTS("BM_Counters_InvertedRate/threads:%int",
                        &CheckInvertedRate);
}  // end namespace

// ========================================================================= //
// ------------------------- Thread Counters Output ------------------------ //
// ========================================================================= //

namespace {
void BM_Counters_Threads(benchmark::State& state) {
  for (auto _ : state) {
  }
  state.counters["foo"] = 1;
  state.counters["bar"] = 2;
}
BENCHMARK(BM_Counters_Threads)->ThreadRange(1, 8);
ADD_CASES(TC_ConsoleOut, {{"^BM_Counters_Threads/threads:%int %console_report "
                           "bar=%hrfloat foo=%hrfloat$"}});
ADD_CASES(TC_JSONOut,
          {{"\"name\": \"BM_Counters_Threads/threads:%int\",$"},
           {"\"family_index\": 5,$", MR_Next},
           {"\"per_family_instance_index\": 0,$", MR_Next},
           {"\"run_name\": \"BM_Counters_Threads/threads:%int\",$", MR_Next},
           {"\"run_type\": \"iteration\",$", MR_Next},
           {"\"repetitions\": 1,$", MR_Next},
           {"\"repetition_index\": 0,$", MR_Next},
           {"\"threads\": %int,$", MR_Next},
           {"\"iterations\": %int,$", MR_Next},
           {"\"real_time\": %float,$", MR_Next},
           {"\"cpu_time\": %float,$", MR_Next},
           {"\"time_unit\": \"ns\",$", MR_Next},
           {"\"bar\": %float,$", MR_Next},
           {"\"foo\": %float$", MR_Next},
           {"}", MR_Next}});
ADD_CASES(
    TC_CSVOut,
    {{"^\"BM_Counters_Threads/threads:%int\",%csv_report,%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckThreads(Results const& e) {
  CHECK_COUNTER_VALUE(e, int, "foo", EQ, e.NumThreads());
  CHECK_COUNTER_VALUE(e, int, "bar", EQ, 2 * e.NumThreads());
}
CHECK_BENCHMARK_RESULTS("BM_Counters_Threads/threads:%int", &CheckThreads);
}  // end namespace

// ========================================================================= //
// ---------------------- ThreadAvg Counters Output ------------------------ //
// ========================================================================= //

namespace {
void BM_Counters_AvgThreads(benchmark::State& state) {
  for (auto _ : state) {
  }
  namespace bm = benchmark;
  state.counters["foo"] = bm::Counter{1, bm::Counter::kAvgThreads};
  state.counters["bar"] = bm::Counter{2, bm::Counter::kAvgThreads};
}
BENCHMARK(BM_Counters_AvgThreads)->ThreadRange(1, 8);
ADD_CASES(TC_ConsoleOut, {{"^BM_Counters_AvgThreads/threads:%int "
                           "%console_report bar=%hrfloat foo=%hrfloat$"}});
ADD_CASES(TC_JSONOut,
          {{"\"name\": \"BM_Counters_AvgThreads/threads:%int\",$"},
           {"\"family_index\": 6,$", MR_Next},
           {"\"per_family_instance_index\": 0,$", MR_Next},
           {"\"run_name\": \"BM_Counters_AvgThreads/threads:%int\",$", MR_Next},
           {"\"run_type\": \"iteration\",$", MR_Next},
           {"\"repetitions\": 1,$", MR_Next},
           {"\"repetition_index\": 0,$", MR_Next},
           {"\"threads\": %int,$", MR_Next},
           {"\"iterations\": %int,$", MR_Next},
           {"\"real_time\": %float,$", MR_Next},
           {"\"cpu_time\": %float,$", MR_Next},
           {"\"time_unit\": \"ns\",$", MR_Next},
           {"\"bar\": %float,$", MR_Next},
           {"\"foo\": %float$", MR_Next},
           {"}", MR_Next}});
ADD_CASES(
    TC_CSVOut,
    {{"^\"BM_Counters_AvgThreads/threads:%int\",%csv_report,%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckAvgThreads(Results const& e) {
  CHECK_COUNTER_VALUE(e, int, "foo", EQ, 1);
  CHECK_COUNTER_VALUE(e, int, "bar", EQ, 2);
}
CHECK_BENCHMARK_RESULTS("BM_Counters_AvgThreads/threads:%int",
                        &CheckAvgThreads);
}  // end namespace

// ========================================================================= //
// ---------------------- ThreadAvg Counters Output ------------------------ //
// ========================================================================= //

namespace {
void BM_Counters_AvgThreadsRate(benchmark::State& state) {
  for (auto _ : state) {
    // This test requires a non-zero CPU time to avoid divide-by-zero
    auto iterations = static_cast<double>(state.iterations()) *
                      static_cast<double>(state.iterations());
    benchmark::DoNotOptimize(iterations);
  }
  namespace bm = benchmark;
  state.counters["foo"] = bm::Counter{1, bm::Counter::kAvgThreadsRate};
  state.counters["bar"] = bm::Counter{2, bm::Counter::kAvgThreadsRate};
}
BENCHMARK(BM_Counters_AvgThreadsRate)->ThreadRange(1, 8);
ADD_CASES(TC_ConsoleOut, {{"^BM_Counters_AvgThreadsRate/threads:%int "
                           "%console_report bar=%hrfloat/s foo=%hrfloat/s$"}});
ADD_CASES(TC_JSONOut,
          {{"\"name\": \"BM_Counters_AvgThreadsRate/threads:%int\",$"},
           {"\"family_index\": 7,$", MR_Next},
           {"\"per_family_instance_index\": 0,$", MR_Next},
           {"\"run_name\": \"BM_Counters_AvgThreadsRate/threads:%int\",$",
            MR_Next},
           {"\"run_type\": \"iteration\",$", MR_Next},
           {"\"repetitions\": 1,$", MR_Next},
           {"\"repetition_index\": 0,$", MR_Next},
           {"\"threads\": %int,$", MR_Next},
           {"\"iterations\": %int,$", MR_Next},
           {"\"real_time\": %float,$", MR_Next},
           {"\"cpu_time\": %float,$", MR_Next},
           {"\"time_unit\": \"ns\",$", MR_Next},
           {"\"bar\": %float,$", MR_Next},
           {"\"foo\": %float$", MR_Next},
           {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_Counters_AvgThreadsRate/"
                       "threads:%int\",%csv_report,%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckAvgThreadsRate(Results const& e) {
  // this (and not real time) is the time used
  double t = e.DurationCPUTime() / e.NumThreads();
  CHECK_FLOAT_COUNTER_VALUE(e, "foo", EQ, 1. / t, 0.001);
  CHECK_FLOAT_COUNTER_VALUE(e, "bar", EQ, 2. / t, 0.001);
}
CHECK_BENCHMARK_RESULTS("BM_Counters_AvgThreadsRate/threads:%int",
                        &CheckAvgThreadsRate);
}  // end namespace

// ========================================================================= //
// ------------------- IterationInvariant Counters Output ------------------ //
// ========================================================================= //

namespace {
void BM_Counters_IterationInvariant(benchmark::State& state) {
  for (auto _ : state) {
  }
  namespace bm = benchmark;
  state.counters["foo"] = bm::Counter{1, bm::Counter::kIsIterationInvariant};
  state.counters["bar"] = bm::Counter{2, bm::Counter::kIsIterationInvariant};
}
BENCHMARK(BM_Counters_IterationInvariant)->ThreadRange(1, 8);
ADD_CASES(TC_ConsoleOut,
          {{"^BM_Counters_IterationInvariant/threads:%int %console_report "
            "bar=%hrfloat foo=%hrfloat$"}});
ADD_CASES(TC_JSONOut,
          {{"\"name\": \"BM_Counters_IterationInvariant/threads:%int\",$"},
           {"\"family_index\": 8,$", MR_Next},
           {"\"per_family_instance_index\": 0,$", MR_Next},
           {"\"run_name\": \"BM_Counters_IterationInvariant/threads:%int\",$",
            MR_Next},
           {"\"run_type\": \"iteration\",$", MR_Next},
           {"\"repetitions\": 1,$", MR_Next},
           {"\"repetition_index\": 0,$", MR_Next},
           {"\"threads\": %int,$", MR_Next},
           {"\"iterations\": %int,$", MR_Next},
           {"\"real_time\": %float,$", MR_Next},
           {"\"cpu_time\": %float,$", MR_Next},
           {"\"time_unit\": \"ns\",$", MR_Next},
           {"\"bar\": %float,$", MR_Next},
           {"\"foo\": %float$", MR_Next},
           {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_Counters_IterationInvariant/"
                       "threads:%int\",%csv_report,%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckIterationInvariant(Results const& e) {
  double its = e.NumIterations();
  // check that the values are within 0.1% of the expected value
  CHECK_FLOAT_COUNTER_VALUE(e, "foo", EQ, its * e.NumThreads(), 0.001);
  CHECK_FLOAT_COUNTER_VALUE(e, "bar", EQ, 2. * its * e.NumThreads(), 0.001);
}
CHECK_BENCHMARK_RESULTS("BM_Counters_IterationInvariant/threads:%int",
                        &CheckIterationInvariant);
}  // end namespace

// ========================================================================= //
// ----------------- IterationInvariantRate Counters Output ---------------- //
// ========================================================================= //

namespace {
void BM_Counters_kIsIterationInvariantRate(benchmark::State& state) {
  for (auto _ : state) {
    // This test requires a non-zero CPU time to avoid divide-by-zero
    auto iterations = static_cast<double>(state.iterations()) *
                      static_cast<double>(state.iterations());
    benchmark::DoNotOptimize(iterations);
  }
  namespace bm = benchmark;
  state.counters["foo"] =
      bm::Counter{1, bm::Counter::kIsIterationInvariantRate};
  state.counters["bar"] =
      bm::Counter{2, bm::Counter::kIsRate | bm::Counter::kIsIterationInvariant};
}
BENCHMARK(BM_Counters_kIsIterationInvariantRate)->ThreadRange(1, 8);
ADD_CASES(TC_ConsoleOut,
          {{"^BM_Counters_kIsIterationInvariantRate/threads:%int "
            "%console_report bar=%hrfloat/s foo=%hrfloat/s$"}});
ADD_CASES(
    TC_JSONOut,
    {{"\"name\": \"BM_Counters_kIsIterationInvariantRate/threads:%int\",$"},
     {"\"family_index\": 9,$", MR_Next},
     {"\"per_family_instance_index\": 0,$", MR_Next},
     {"\"run_name\": \"BM_Counters_kIsIterationInvariantRate/threads:%int\",$",
      MR_Next},
     {"\"run_type\": \"iteration\",$", MR_Next},
     {"\"repetitions\": 1,$", MR_Next},
     {"\"repetition_index\": 0,$", MR_Next},
     {"\"threads\": %int,$", MR_Next},
     {"\"iterations\": %int,$", MR_Next},
     {"\"real_time\": %float,$", MR_Next},
     {"\"cpu_time\": %float,$", MR_Next},
     {"\"time_unit\": \"ns\",$", MR_Next},
     {"\"bar\": %float,$", MR_Next},
     {"\"foo\": %float$", MR_Next},
     {"}", MR_Next}});
ADD_CASES(
    TC_CSVOut,
    {{"^\"BM_Counters_kIsIterationInvariantRate/threads:%int\",%csv_report,"
      "%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckIsIterationInvariantRate(Results const& e) {
  double its = e.NumIterations();
  // this (and not real time) is the time used
  double t = e.DurationCPUTime() / e.NumThreads();
  // check that the values are within 0.1% of the expected values
  CHECK_FLOAT_COUNTER_VALUE(e, "foo", EQ, its * 1. * e.NumThreads() / t, 0.001);
  CHECK_FLOAT_COUNTER_VALUE(e, "bar", EQ, its * 2. * e.NumThreads() / t, 0.001);
}
CHECK_BENCHMARK_RESULTS("BM_Counters_kIsIterationInvariantRate/threads:%int",
                        &CheckIsIterationInvariantRate);
}  // end namespace

// ========================================================================= //
// --------------------- AvgIterations Counters Output --------------------- //
// ========================================================================= //

namespace {
void BM_Counters_AvgIterations(benchmark::State& state) {
  for (auto _ : state) {
  }
  namespace bm = benchmark;
  state.counters["foo"] = bm::Counter{1, bm::Counter::kAvgIterations};
  state.counters["bar"] = bm::Counter{2, bm::Counter::kAvgIterations};
}
BENCHMARK(BM_Counters_AvgIterations)->ThreadRange(1, 8);
ADD_CASES(TC_ConsoleOut,
          {{"^BM_Counters_AvgIterations/threads:%int %console_report "
            "bar=%hrfloat foo=%hrfloat$"}});
ADD_CASES(TC_JSONOut,
          {{"\"name\": \"BM_Counters_AvgIterations/threads:%int\",$"},
           {"\"family_index\": 10,$", MR_Next},
           {"\"per_family_instance_index\": 0,$", MR_Next},
           {"\"run_name\": \"BM_Counters_AvgIterations/threads:%int\",$",
            MR_Next},
           {"\"run_type\": \"iteration\",$", MR_Next},
           {"\"repetitions\": 1,$", MR_Next},
           {"\"repetition_index\": 0,$", MR_Next},
           {"\"threads\": %int,$", MR_Next},
           {"\"iterations\": %int,$", MR_Next},
           {"\"real_time\": %float,$", MR_Next},
           {"\"cpu_time\": %float,$", MR_Next},
           {"\"time_unit\": \"ns\",$", MR_Next},
           {"\"bar\": %float,$", MR_Next},
           {"\"foo\": %float$", MR_Next},
           {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_Counters_AvgIterations/"
                       "threads:%int\",%csv_report,%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckAvgIterations(Results const& e) {
  double its = e.NumIterations();
  // check that the values are within 0.1% of the expected value
  CHECK_FLOAT_COUNTER_VALUE(e, "foo", EQ, 1. * e.NumThreads() / its, 0.001);
  CHECK_FLOAT_COUNTER_VALUE(e, "bar", EQ, 2. * e.NumThreads() / its, 0.001);
}
CHECK_BENCHMARK_RESULTS("BM_Counters_AvgIterations/threads:%int",
                        &CheckAvgIterations);
}  // end namespace

// ========================================================================= //
// ------------------- AvgIterationsRate Counters Output ------------------- //
// ========================================================================= //

namespace {
void BM_Counters_kAvgIterationsRate(benchmark::State& state) {
  for (auto _ : state) {
    // This test requires a non-zero CPU time to avoid divide-by-zero
    auto iterations = static_cast<double>(state.iterations()) *
                      static_cast<double>(state.iterations());
    benchmark::DoNotOptimize(iterations);
  }
  namespace bm = benchmark;
  state.counters["foo"] = bm::Counter{1, bm::Counter::kAvgIterationsRate};
  state.counters["bar"] =
      bm::Counter{2, bm::Counter::kIsRate | bm::Counter::kAvgIterations};
}
BENCHMARK(BM_Counters_kAvgIterationsRate)->ThreadRange(1, 8);
ADD_CASES(TC_ConsoleOut, {{"^BM_Counters_kAvgIterationsRate/threads:%int "
                           "%console_report bar=%hrfloat/s foo=%hrfloat/s$"}});
ADD_CASES(TC_JSONOut,
          {{"\"name\": \"BM_Counters_kAvgIterationsRate/threads:%int\",$"},
           {"\"family_index\": 11,$", MR_Next},
           {"\"per_family_instance_index\": 0,$", MR_Next},
           {"\"run_name\": \"BM_Counters_kAvgIterationsRate/threads:%int\",$",
            MR_Next},
           {"\"run_type\": \"iteration\",$", MR_Next},
           {"\"repetitions\": 1,$", MR_Next},
           {"\"repetition_index\": 0,$", MR_Next},
           {"\"threads\": %int,$", MR_Next},
           {"\"iterations\": %int,$", MR_Next},
           {"\"real_time\": %float,$", MR_Next},
           {"\"cpu_time\": %float,$", MR_Next},
           {"\"time_unit\": \"ns\",$", MR_Next},
           {"\"bar\": %float,$", MR_Next},
           {"\"foo\": %float$", MR_Next},
           {"}", MR_Next}});
ADD_CASES(TC_CSVOut,
          {{"^\"BM_Counters_kAvgIterationsRate/threads:%int\",%csv_report,"
            "%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckAvgIterationsRate(Results const& e) {
  double its = e.NumIterations();
  // this (and not real time) is the time used
  double t = e.DurationCPUTime() / e.NumThreads();
  // check that the values are within 0.1% of the expected values
  CHECK_FLOAT_COUNTER_VALUE(e, "foo", EQ, 1. * e.NumThreads() / its / t, 0.001);
  CHECK_FLOAT_COUNTER_VALUE(e, "bar", EQ, 2. * e.NumThreads() / its / t, 0.001);
}
CHECK_BENCHMARK_RESULTS("BM_Counters_kAvgIterationsRate/threads:%int",
                        &CheckAvgIterationsRate);
}  // end namespace

// ========================================================================= //
// --------------------------- TEST CASES END ------------------------------ //
// ========================================================================= //

int main(int argc, char* argv[]) {
  benchmark::MaybeReenterWithoutASLR(argc, argv);
  RunOutputTests(argc, argv);
}
