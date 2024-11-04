// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "benchmark_runner.h"

#include "benchmark/benchmark.h"
#include "benchmark_api_internal.h"
#include "internal_macros.h"

#ifndef BENCHMARK_OS_WINDOWS
#if !defined(BENCHMARK_OS_FUCHSIA) && !defined(BENCHMARK_OS_QURT)
#include <sys/resource.h>
#endif
#include <sys/time.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "check.h"
#include "colorprint.h"
#include "commandlineflags.h"
#include "complexity.h"
#include "counter.h"
#include "internal_macros.h"
#include "log.h"
#include "mutex.h"
#include "perf_counters.h"
#include "re.h"
#include "statistics.h"
#include "string_util.h"
#include "thread_manager.h"
#include "thread_timer.h"

namespace benchmark {

namespace internal {

MemoryManager* memory_manager = nullptr;

namespace {

static constexpr IterationCount kMaxIterations = 1000000000000;
const double kDefaultMinTime =
    std::strtod(::benchmark::kDefaultMinTimeStr, /*p_end*/ nullptr);

BenchmarkReporter::Run CreateRunReport(
    const benchmark::internal::BenchmarkInstance& b,
    const internal::ThreadManager::Result& results,
    IterationCount memory_iterations,
    const MemoryManager::Result* memory_result, double seconds,
    int64_t repetition_index, int64_t repeats) {
  // Create report about this benchmark run.
  BenchmarkReporter::Run report;

  report.run_name = b.name();
  report.family_index = b.family_index();
  report.per_family_instance_index = b.per_family_instance_index();
  report.skipped = results.skipped_;
  report.skip_message = results.skip_message_;
  report.report_label = results.report_label_;
  // This is the total iterations across all threads.
  report.iterations = results.iterations;
  report.time_unit = b.time_unit();
  report.threads = b.threads();
  report.repetition_index = repetition_index;
  report.repetitions = repeats;

  if (!report.skipped) {
    if (b.use_manual_time()) {
      report.real_accumulated_time = results.manual_time_used;
    } else {
      report.real_accumulated_time = results.real_time_used;
    }
    report.use_real_time_for_initial_big_o = b.use_manual_time();
    report.cpu_accumulated_time = results.cpu_time_used;
    report.complexity_n = results.complexity_n;
    report.complexity = b.complexity();
    report.complexity_lambda = b.complexity_lambda();
    report.statistics = &b.statistics();
    report.counters = results.counters;

    if (memory_iterations > 0) {
      assert(memory_result != nullptr);
      report.memory_result = memory_result;
      report.allocs_per_iter =
          memory_iterations ? static_cast<double>(memory_result->num_allocs) /
                                  static_cast<double>(memory_iterations)
                            : 0;
    }

    internal::Finish(&report.counters, results.iterations, seconds,
                     b.threads());
  }
  return report;
}

// Execute one thread of benchmark b for the specified number of iterations.
// Adds the stats collected for the thread into manager->results.
void RunInThread(const BenchmarkInstance* b, IterationCount iters,
                 int thread_id, ThreadManager* manager,
                 PerfCountersMeasurement* perf_counters_measurement) {
  internal::ThreadTimer timer(
      b->measure_process_cpu_time()
          ? internal::ThreadTimer::CreateProcessCpuTime()
          : internal::ThreadTimer::Create());

  State st =
      b->Run(iters, thread_id, &timer, manager, perf_counters_measurement);
  BM_CHECK(st.skipped() || st.iterations() >= st.max_iterations)
      << "Benchmark returned before State::KeepRunning() returned false!";
  {
    MutexLock l(manager->GetBenchmarkMutex());
    internal::ThreadManager::Result& results = manager->results;
    results.iterations += st.iterations();
    results.cpu_time_used += timer.cpu_time_used();
    results.real_time_used += timer.real_time_used();
    results.manual_time_used += timer.manual_time_used();
    results.complexity_n += st.complexity_length_n();
    internal::Increment(&results.counters, st.counters);
  }
  manager->NotifyThreadComplete();
}

double ComputeMinTime(const benchmark::internal::BenchmarkInstance& b,
                      const BenchTimeType& iters_or_time) {
  if (!IsZero(b.min_time())) return b.min_time();
  // If the flag was used to specify number of iters, then return the default
  // min_time.
  if (iters_or_time.tag == BenchTimeType::ITERS) return kDefaultMinTime;

  return iters_or_time.time;
}

IterationCount ComputeIters(const benchmark::internal::BenchmarkInstance& b,
                            const BenchTimeType& iters_or_time) {
  if (b.iterations() != 0) return b.iterations();

  // We've already concluded that this flag is currently used to pass
  // iters but do a check here again anyway.
  BM_CHECK(iters_or_time.tag == BenchTimeType::ITERS);
  return iters_or_time.iters;
}

}  // end namespace

BenchTimeType ParseBenchMinTime(const std::string& value) {
  BenchTimeType ret;

  if (value.empty()) {
    ret.tag = BenchTimeType::TIME;
    ret.time = 0.0;
    return ret;
  }

  if (value.back() == 'x') {
    char* p_end;
    // Reset errno before it's changed by strtol.
    errno = 0;
    IterationCount num_iters = std::strtol(value.c_str(), &p_end, 10);

    // After a valid parse, p_end should have been set to
    // point to the 'x' suffix.
    BM_CHECK(errno == 0 && p_end != nullptr && *p_end == 'x')
        << "Malformed iters value passed to --benchmark_min_time: `" << value
        << "`. Expected --benchmark_min_time=<integer>x.";

    ret.tag = BenchTimeType::ITERS;
    ret.iters = num_iters;
    return ret;
  }

  bool has_suffix = value.back() == 's';
  if (!has_suffix) {
    BM_VLOG(0) << "Value passed to --benchmark_min_time should have a suffix. "
                  "Eg., `30s` for 30-seconds.";
  }

  char* p_end;
  // Reset errno before it's changed by strtod.
  errno = 0;
  double min_time = std::strtod(value.c_str(), &p_end);

  // After a successful parse, p_end should point to the suffix 's',
  // or the end of the string if the suffix was omitted.
  BM_CHECK(errno == 0 && p_end != nullptr &&
           ((has_suffix && *p_end == 's') || *p_end == '\0'))
      << "Malformed seconds value passed to --benchmark_min_time: `" << value
      << "`. Expected --benchmark_min_time=<float>x.";

  ret.tag = BenchTimeType::TIME;
  ret.time = min_time;

  return ret;
}

BenchmarkRunner::BenchmarkRunner(
    const benchmark::internal::BenchmarkInstance& b_,
    PerfCountersMeasurement* pcm_,
    BenchmarkReporter::PerFamilyRunReports* reports_for_family_)
    : b(b_),
      reports_for_family(reports_for_family_),
      parsed_benchtime_flag(ParseBenchMinTime(FLAGS_benchmark_min_time)),
      min_time(ComputeMinTime(b_, parsed_benchtime_flag)),
      min_warmup_time((!IsZero(b.min_time()) && b.min_warmup_time() > 0.0)
                          ? b.min_warmup_time()
                          : FLAGS_benchmark_min_warmup_time),
      warmup_done(!(min_warmup_time > 0.0)),
      repeats(b.repetitions() != 0 ? b.repetitions()
                                   : FLAGS_benchmark_repetitions),
      has_explicit_iteration_count(b.iterations() != 0 ||
                                   parsed_benchtime_flag.tag ==
                                       BenchTimeType::ITERS),
      pool(b.threads() - 1),
      iters(has_explicit_iteration_count
                ? ComputeIters(b_, parsed_benchtime_flag)
                : 1),
      perf_counters_measurement_ptr(pcm_) {
  run_results.display_report_aggregates_only =
      (FLAGS_benchmark_report_aggregates_only ||
       FLAGS_benchmark_display_aggregates_only);
  run_results.file_report_aggregates_only =
      FLAGS_benchmark_report_aggregates_only;
  if (b.aggregation_report_mode() != internal::ARM_Unspecified) {
    run_results.display_report_aggregates_only =
        (b.aggregation_report_mode() &
         internal::ARM_DisplayReportAggregatesOnly);
    run_results.file_report_aggregates_only =
        (b.aggregation_report_mode() & internal::ARM_FileReportAggregatesOnly);
    BM_CHECK(FLAGS_benchmark_perf_counters.empty() ||
             (perf_counters_measurement_ptr->num_counters() == 0))
        << "Perf counters were requested but could not be set up.";
  }
}

BenchmarkRunner::IterationResults BenchmarkRunner::DoNIterations() {
  BM_VLOG(2) << "Running " << b.name().str() << " for " << iters << "\n";

  std::unique_ptr<internal::ThreadManager> manager;
  manager.reset(new internal::ThreadManager(b.threads()));

  // Run all but one thread in separate threads
  for (std::size_t ti = 0; ti < pool.size(); ++ti) {
    pool[ti] = std::thread(&RunInThread, &b, iters, static_cast<int>(ti + 1),
                           manager.get(), perf_counters_measurement_ptr);
  }
  // And run one thread here directly.
  // (If we were asked to run just one thread, we don't create new threads.)
  // Yes, we need to do this here *after* we start the separate threads.
  RunInThread(&b, iters, 0, manager.get(), perf_counters_measurement_ptr);

  // The main thread has finished. Now let's wait for the other threads.
  manager->WaitForAllThreads();
  for (std::thread& thread : pool) thread.join();

  IterationResults i;
  // Acquire the measurements/counters from the manager, UNDER THE LOCK!
  {
    MutexLock l(manager->GetBenchmarkMutex());
    i.results = manager->results;
  }

  // And get rid of the manager.
  manager.reset();

  // Adjust real/manual time stats since they were reported per thread.
  i.results.real_time_used /= b.threads();
  i.results.manual_time_used /= b.threads();
  // If we were measuring whole-process CPU usage, adjust the CPU time too.
  if (b.measure_process_cpu_time()) i.results.cpu_time_used /= b.threads();

  BM_VLOG(2) << "Ran in " << i.results.cpu_time_used << "/"
             << i.results.real_time_used << "\n";

  // By using KeepRunningBatch a benchmark can iterate more times than
  // requested, so take the iteration count from i.results.
  i.iters = i.results.iterations / b.threads();

  // Base decisions off of real time if requested by this benchmark.
  i.seconds = i.results.cpu_time_used;
  if (b.use_manual_time()) {
    i.seconds = i.results.manual_time_used;
  } else if (b.use_real_time()) {
    i.seconds = i.results.real_time_used;
  }

  return i;
}

IterationCount BenchmarkRunner::PredictNumItersNeeded(
    const IterationResults& i) const {
  // See how much iterations should be increased by.
  // Note: Avoid division by zero with max(seconds, 1ns).
  double multiplier = GetMinTimeToApply() * 1.4 / std::max(i.seconds, 1e-9);
  // If our last run was at least 10% of FLAGS_benchmark_min_time then we
  // use the multiplier directly.
  // Otherwise we use at most 10 times expansion.
  // NOTE: When the last run was at least 10% of the min time the max
  // expansion should be 14x.
  const bool is_significant = (i.seconds / GetMinTimeToApply()) > 0.1;
  multiplier = is_significant ? multiplier : 10.0;

  // So what seems to be the sufficiently-large iteration count? Round up.
  const IterationCount max_next_iters = static_cast<IterationCount>(
      std::llround(std::max(multiplier * static_cast<double>(i.iters),
                            static_cast<double>(i.iters) + 1.0)));
  // But we do have *some* limits though..
  const IterationCount next_iters = std::min(max_next_iters, kMaxIterations);

  BM_VLOG(3) << "Next iters: " << next_iters << ", " << multiplier << "\n";
  return next_iters;  // round up before conversion to integer.
}

bool BenchmarkRunner::ShouldReportIterationResults(
    const IterationResults& i) const {
  // Determine if this run should be reported;
  // Either it has run for a sufficient amount of time
  // or because an error was reported.
  return i.results.skipped_ ||
         i.iters >= kMaxIterations ||  // Too many iterations already.
         i.seconds >=
             GetMinTimeToApply() ||  // The elapsed time is large enough.
         // CPU time is specified but the elapsed real time greatly exceeds
         // the minimum time.
         // Note that user provided timers are except from this test.
         ((i.results.real_time_used >= 5 * GetMinTimeToApply()) &&
          !b.use_manual_time());
}

double BenchmarkRunner::GetMinTimeToApply() const {
  // In order to re-use functionality to run and measure benchmarks for running
  // a warmup phase of the benchmark, we need a way of telling whether to apply
  // min_time or min_warmup_time. This function will figure out if we are in the
  // warmup phase and therefore need to apply min_warmup_time or if we already
  // in the benchmarking phase and min_time needs to be applied.
  return warmup_done ? min_time : min_warmup_time;
}

void BenchmarkRunner::FinishWarmUp(const IterationCount& i) {
  warmup_done = true;
  iters = i;
}

void BenchmarkRunner::RunWarmUp() {
  // Use the same mechanisms for warming up the benchmark as used for actually
  // running and measuring the benchmark.
  IterationResults i_warmup;
  // Dont use the iterations determined in the warmup phase for the actual
  // measured benchmark phase. While this may be a good starting point for the
  // benchmark and it would therefore get rid of the need to figure out how many
  // iterations are needed if min_time is set again, this may also be a complete
  // wrong guess since the warmup loops might be considerably slower (e.g
  // because of caching effects).
  const IterationCount i_backup = iters;

  for (;;) {
    b.Setup();
    i_warmup = DoNIterations();
    b.Teardown();

    const bool finish = ShouldReportIterationResults(i_warmup);

    if (finish) {
      FinishWarmUp(i_backup);
      break;
    }

    // Although we are running "only" a warmup phase where running enough
    // iterations at once without measuring time isn't as important as it is for
    // the benchmarking phase, we still do it the same way as otherwise it is
    // very confusing for the user to know how to choose a proper value for
    // min_warmup_time if a different approach on running it is used.
    iters = PredictNumItersNeeded(i_warmup);
    assert(iters > i_warmup.iters &&
           "if we did more iterations than we want to do the next time, "
           "then we should have accepted the current iteration run.");
  }
}

void BenchmarkRunner::DoOneRepetition() {
  assert(HasRepeatsRemaining() && "Already done all repetitions?");

  const bool is_the_first_repetition = num_repetitions_done == 0;

  // In case a warmup phase is requested by the benchmark, run it now.
  // After running the warmup phase the BenchmarkRunner should be in a state as
  // this warmup never happened except the fact that warmup_done is set. Every
  // other manipulation of the BenchmarkRunner instance would be a bug! Please
  // fix it.
  if (!warmup_done) RunWarmUp();

  IterationResults i;
  // We *may* be gradually increasing the length (iteration count)
  // of the benchmark until we decide the results are significant.
  // And once we do, we report those last results and exit.
  // Please do note that the if there are repetitions, the iteration count
  // is *only* calculated for the *first* repetition, and other repetitions
  // simply use that precomputed iteration count.
  for (;;) {
    b.Setup();
    i = DoNIterations();
    b.Teardown();

    // Do we consider the results to be significant?
    // If we are doing repetitions, and the first repetition was already done,
    // it has calculated the correct iteration time, so we have run that very
    // iteration count just now. No need to calculate anything. Just report.
    // Else, the normal rules apply.
    const bool results_are_significant = !is_the_first_repetition ||
                                         has_explicit_iteration_count ||
                                         ShouldReportIterationResults(i);

    if (results_are_significant) break;  // Good, let's report them!

    // Nope, bad iteration. Let's re-estimate the hopefully-sufficient
    // iteration count, and run the benchmark again...

    iters = PredictNumItersNeeded(i);
    assert(iters > i.iters &&
           "if we did more iterations than we want to do the next time, "
           "then we should have accepted the current iteration run.");
  }

  // Oh, one last thing, we need to also produce the 'memory measurements'..
  MemoryManager::Result* memory_result = nullptr;
  IterationCount memory_iterations = 0;
  if (memory_manager != nullptr) {
    // TODO(vyng): Consider making BenchmarkReporter::Run::memory_result an
    // optional so we don't have to own the Result here.
    // Can't do it now due to cxx03.
    memory_results.push_back(MemoryManager::Result());
    memory_result = &memory_results.back();
    // Only run a few iterations to reduce the impact of one-time
    // allocations in benchmarks that are not properly managed.
    memory_iterations = std::min<IterationCount>(16, iters);
    memory_manager->Start();
    std::unique_ptr<internal::ThreadManager> manager;
    manager.reset(new internal::ThreadManager(1));
    b.Setup();
    RunInThread(&b, memory_iterations, 0, manager.get(),
                perf_counters_measurement_ptr);
    manager->WaitForAllThreads();
    manager.reset();
    b.Teardown();
    memory_manager->Stop(*memory_result);
  }

  // Ok, now actually report.
  BenchmarkReporter::Run report =
      CreateRunReport(b, i.results, memory_iterations, memory_result, i.seconds,
                      num_repetitions_done, repeats);

  if (reports_for_family) {
    ++reports_for_family->num_runs_done;
    if (!report.skipped) reports_for_family->Runs.push_back(report);
  }

  run_results.non_aggregates.push_back(report);

  ++num_repetitions_done;
}

RunResults&& BenchmarkRunner::GetResults() {
  assert(!HasRepeatsRemaining() && "Did not run all repetitions yet?");

  // Calculate additional statistics over the repetitions of this instance.
  run_results.aggregates_only = ComputeStats(run_results.non_aggregates);

  return std::move(run_results);
}

}  // end namespace internal

}  // end namespace benchmark
