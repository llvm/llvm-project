
#undef NDEBUG

#include "benchmark/benchmark.h"
#include "output_test.h"

// ========================================================================= //
// ---------------------- Testing Prologue Output -------------------------- //
// ========================================================================= //

ADD_CASES(TC_ConsoleOut,
          {{"^[-]+$", MR_Next},
           {"^Benchmark %s Time %s CPU %s Iterations UserCounters...$", MR_Next},
           {"^[-]+$", MR_Next}});
ADD_CASES(TC_CSVOut, {{"%csv_header,\"bar\",\"foo\""}});

// ========================================================================= //
// ------------------------- Simple Counters Output ------------------------ //
// ========================================================================= //

void BM_Counters_Simple(benchmark::State& state) {
  for (auto _ : state) {
  }
  state.counters["foo"] = 1;
  state.counters["bar"] = 2 * (double)state.iterations();
}
BENCHMARK(BM_Counters_Simple);
ADD_CASES(TC_ConsoleOut, {{"^BM_Counters_Simple %console_report bar=%hrfloat foo=%hrfloat$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_Counters_Simple\",$"},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\",$", MR_Next},
                       {"\"bar\": %float,$", MR_Next},
                       {"\"foo\": %float$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_Counters_Simple\",%csv_report,%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckSimple(Results const& e) {
  double its = e.GetAs< double >("iterations");
  CHECK_COUNTER_VALUE(e, int, "foo", EQ, 1);
  // check that the value of bar is within 0.1% of the expected value
  CHECK_FLOAT_COUNTER_VALUE(e, "bar", EQ, 2.*its, 0.001);
}
CHECK_BENCHMARK_RESULTS("BM_Counters_Simple", &CheckSimple);

// ========================================================================= //
// --------------------- Counters+Items+Bytes/s Output --------------------- //
// ========================================================================= //

namespace { int num_calls1 = 0; }
void BM_Counters_WithBytesAndItemsPSec(benchmark::State& state) {
  for (auto _ : state) {
  }
  state.counters["foo"] = 1;
  state.counters["bar"] = ++num_calls1;
  state.SetBytesProcessed(364);
  state.SetItemsProcessed(150);
}
BENCHMARK(BM_Counters_WithBytesAndItemsPSec);
ADD_CASES(TC_ConsoleOut,
          {{"^BM_Counters_WithBytesAndItemsPSec %console_report "
            "bar=%hrfloat foo=%hrfloat +%hrfloatB/s +%hrfloat items/s$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_Counters_WithBytesAndItemsPSec\",$"},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\",$", MR_Next},
                       {"\"bytes_per_second\": %float,$", MR_Next},
                       {"\"items_per_second\": %float,$", MR_Next},
                       {"\"bar\": %float,$", MR_Next},
                       {"\"foo\": %float$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_Counters_WithBytesAndItemsPSec\","
                       "%csv_bytes_items_report,%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckBytesAndItemsPSec(Results const& e) {
  double t = e.DurationCPUTime(); // this (and not real time) is the time used
  CHECK_COUNTER_VALUE(e, int, "foo", EQ, 1);
  CHECK_COUNTER_VALUE(e, int, "bar", EQ, num_calls1);
  // check that the values are within 0.1% of the expected values
  CHECK_FLOAT_RESULT_VALUE(e, "bytes_per_second", EQ, 364./t, 0.001);
  CHECK_FLOAT_RESULT_VALUE(e, "items_per_second", EQ, 150./t, 0.001);
}
CHECK_BENCHMARK_RESULTS("BM_Counters_WithBytesAndItemsPSec",
                        &CheckBytesAndItemsPSec);

// ========================================================================= //
// ------------------------- Rate Counters Output -------------------------- //
// ========================================================================= //

void BM_Counters_Rate(benchmark::State& state) {
  for (auto _ : state) {
  }
  namespace bm = benchmark;
  state.counters["foo"] = bm::Counter{1, bm::Counter::kIsRate};
  state.counters["bar"] = bm::Counter{2, bm::Counter::kIsRate};
}
BENCHMARK(BM_Counters_Rate);
ADD_CASES(TC_ConsoleOut, {{"^BM_Counters_Rate %console_report bar=%hrfloat/s foo=%hrfloat/s$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_Counters_Rate\",$"},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\",$", MR_Next},
                       {"\"bar\": %float,$", MR_Next},
                       {"\"foo\": %float$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_Counters_Rate\",%csv_report,%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckRate(Results const& e) {
  double t = e.DurationCPUTime(); // this (and not real time) is the time used
  // check that the values are within 0.1% of the expected values
  CHECK_FLOAT_COUNTER_VALUE(e, "foo", EQ, 1./t, 0.001);
  CHECK_FLOAT_COUNTER_VALUE(e, "bar", EQ, 2./t, 0.001);
}
CHECK_BENCHMARK_RESULTS("BM_Counters_Rate", &CheckRate);

// ========================================================================= //
// ------------------------- Thread Counters Output ------------------------ //
// ========================================================================= //

void BM_Counters_Threads(benchmark::State& state) {
  for (auto _ : state) {
  }
  state.counters["foo"] = 1;
  state.counters["bar"] = 2;
}
BENCHMARK(BM_Counters_Threads)->ThreadRange(1, 8);
ADD_CASES(TC_ConsoleOut, {{"^BM_Counters_Threads/threads:%int %console_report bar=%hrfloat foo=%hrfloat$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_Counters_Threads/threads:%int\",$"},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\",$", MR_Next},
                       {"\"bar\": %float,$", MR_Next},
                       {"\"foo\": %float$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_Counters_Threads/threads:%int\",%csv_report,%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckThreads(Results const& e) {
  CHECK_COUNTER_VALUE(e, int, "foo", EQ, e.NumThreads());
  CHECK_COUNTER_VALUE(e, int, "bar", EQ, 2 * e.NumThreads());
}
CHECK_BENCHMARK_RESULTS("BM_Counters_Threads/threads:%int", &CheckThreads);

// ========================================================================= //
// ---------------------- ThreadAvg Counters Output ------------------------ //
// ========================================================================= //

void BM_Counters_AvgThreads(benchmark::State& state) {
  for (auto _ : state) {
  }
  namespace bm = benchmark;
  state.counters["foo"] = bm::Counter{1, bm::Counter::kAvgThreads};
  state.counters["bar"] = bm::Counter{2, bm::Counter::kAvgThreads};
}
BENCHMARK(BM_Counters_AvgThreads)->ThreadRange(1, 8);
ADD_CASES(TC_ConsoleOut, {{"^BM_Counters_AvgThreads/threads:%int %console_report bar=%hrfloat foo=%hrfloat$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_Counters_AvgThreads/threads:%int\",$"},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\",$", MR_Next},
                       {"\"bar\": %float,$", MR_Next},
                       {"\"foo\": %float$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_Counters_AvgThreads/threads:%int\",%csv_report,%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckAvgThreads(Results const& e) {
  CHECK_COUNTER_VALUE(e, int, "foo", EQ, 1);
  CHECK_COUNTER_VALUE(e, int, "bar", EQ, 2);
}
CHECK_BENCHMARK_RESULTS("BM_Counters_AvgThreads/threads:%int",
                        &CheckAvgThreads);

// ========================================================================= //
// ---------------------- ThreadAvg Counters Output ------------------------ //
// ========================================================================= //

void BM_Counters_AvgThreadsRate(benchmark::State& state) {
  for (auto _ : state) {
  }
  namespace bm = benchmark;
  state.counters["foo"] = bm::Counter{1, bm::Counter::kAvgThreadsRate};
  state.counters["bar"] = bm::Counter{2, bm::Counter::kAvgThreadsRate};
}
BENCHMARK(BM_Counters_AvgThreadsRate)->ThreadRange(1, 8);
ADD_CASES(TC_ConsoleOut, {{"^BM_Counters_AvgThreadsRate/threads:%int %console_report bar=%hrfloat/s foo=%hrfloat/s$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_Counters_AvgThreadsRate/threads:%int\",$"},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\",$", MR_Next},
                       {"\"bar\": %float,$", MR_Next},
                       {"\"foo\": %float$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_Counters_AvgThreadsRate/threads:%int\",%csv_report,%float,%float$"}});
// VS2013 does not allow this function to be passed as a lambda argument
// to CHECK_BENCHMARK_RESULTS()
void CheckAvgThreadsRate(Results const& e) {
  CHECK_FLOAT_COUNTER_VALUE(e, "foo", EQ, 1./e.DurationCPUTime(), 0.001);
  CHECK_FLOAT_COUNTER_VALUE(e, "bar", EQ, 2./e.DurationCPUTime(), 0.001);
}
CHECK_BENCHMARK_RESULTS("BM_Counters_AvgThreadsRate/threads:%int",
                        &CheckAvgThreadsRate);

// ========================================================================= //
// --------------------------- TEST CASES END ------------------------------ //
// ========================================================================= //

int main(int argc, char* argv[]) { RunOutputTests(argc, argv); }
