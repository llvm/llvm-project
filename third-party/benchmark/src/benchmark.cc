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

#include "benchmark/benchmark.h"

#include "benchmark_api_internal.h"
#include "benchmark_runner.h"
#include "internal_macros.h"

#ifndef BENCHMARK_OS_WINDOWS
#ifndef BENCHMARK_OS_FUCHSIA
#include <sys/resource.h>
#endif
#include <sys/time.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
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
// Print a list of benchmarks. This option overrides all other options.
BM_DEFINE_bool(benchmark_list_tests, false);

// A regular expression that specifies the set of benchmarks to execute.  If
// this flag is empty, or if this flag is the string \"all\", all benchmarks
// linked into the binary are run.
BM_DEFINE_string(benchmark_filter, "");

// Minimum number of seconds we should run benchmark before results are
// considered significant.  For cpu-time based tests, this is the lower bound
// on the total cpu time used by all threads that make up the test.  For
// real-time based tests, this is the lower bound on the elapsed time of the
// benchmark execution, regardless of number of threads.
BM_DEFINE_double(benchmark_min_time, 0.5);

// The number of runs of each benchmark. If greater than 1, the mean and
// standard deviation of the runs will be reported.
BM_DEFINE_int32(benchmark_repetitions, 1);

// If set, enable random interleaving of repetitions of all benchmarks.
// See http://github.com/google/benchmark/issues/1051 for details.
BM_DEFINE_bool(benchmark_enable_random_interleaving, false);

// Report the result of each benchmark repetitions. When 'true' is specified
// only the mean, standard deviation, and other statistics are reported for
// repeated benchmarks. Affects all reporters.
BM_DEFINE_bool(benchmark_report_aggregates_only, false);

// Display the result of each benchmark repetitions. When 'true' is specified
// only the mean, standard deviation, and other statistics are displayed for
// repeated benchmarks. Unlike benchmark_report_aggregates_only, only affects
// the display reporter, but  *NOT* file reporter, which will still contain
// all the output.
BM_DEFINE_bool(benchmark_display_aggregates_only, false);

// The format to use for console output.
// Valid values are 'console', 'json', or 'csv'.
BM_DEFINE_string(benchmark_format, "console");

// The format to use for file output.
// Valid values are 'console', 'json', or 'csv'.
BM_DEFINE_string(benchmark_out_format, "json");

// The file to write additional output to.
BM_DEFINE_string(benchmark_out, "");

// Whether to use colors in the output.  Valid values:
// 'true'/'yes'/1, 'false'/'no'/0, and 'auto'. 'auto' means to use colors if
// the output is being sent to a terminal and the TERM environment variable is
// set to a terminal type that supports colors.
BM_DEFINE_string(benchmark_color, "auto");

// Whether to use tabular format when printing user counters to the console.
// Valid values: 'true'/'yes'/1, 'false'/'no'/0.  Defaults to false.
BM_DEFINE_bool(benchmark_counters_tabular, false);

// List of additional perf counters to collect, in libpfm format. For more
// information about libpfm: https://man7.org/linux/man-pages/man3/libpfm.3.html
BM_DEFINE_string(benchmark_perf_counters, "");

// Extra context to include in the output formatted as comma-separated key-value
// pairs. Kept internal as it's only used for parsing from env/command line.
BM_DEFINE_kvpairs(benchmark_context, {});

// The level of verbose logging to output
BM_DEFINE_int32(v, 0);

namespace internal {

std::map<std::string, std::string>* global_context = nullptr;

// FIXME: wouldn't LTO mess this up?
void UseCharPointer(char const volatile*) {}

}  // namespace internal

State::State(IterationCount max_iters, const std::vector<int64_t>& ranges,
             int thread_i, int n_threads, internal::ThreadTimer* timer,
             internal::ThreadManager* manager,
             internal::PerfCountersMeasurement* perf_counters_measurement)
    : total_iterations_(0),
      batch_leftover_(0),
      max_iterations(max_iters),
      started_(false),
      finished_(false),
      error_occurred_(false),
      range_(ranges),
      complexity_n_(0),
      thread_index_(thread_i),
      threads_(n_threads),
      timer_(timer),
      manager_(manager),
      perf_counters_measurement_(perf_counters_measurement) {
  BM_CHECK(max_iterations != 0) << "At least one iteration must be run";
  BM_CHECK_LT(thread_index_, threads_)
      << "thread_index must be less than threads";

  // Note: The use of offsetof below is technically undefined until C++17
  // because State is not a standard layout type. However, all compilers
  // currently provide well-defined behavior as an extension (which is
  // demonstrated since constexpr evaluation must diagnose all undefined
  // behavior). However, GCC and Clang also warn about this use of offsetof,
  // which must be suppressed.
#if defined(__INTEL_COMPILER)
#pragma warning push
#pragma warning(disable : 1875)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winvalid-offsetof"
#endif
  // Offset tests to ensure commonly accessed data is on the first cache line.
  const int cache_line_size = 64;
  static_assert(offsetof(State, error_occurred_) <=
                    (cache_line_size - sizeof(error_occurred_)),
                "");
#if defined(__INTEL_COMPILER)
#pragma warning pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif
}

void State::PauseTiming() {
  // Add in time accumulated so far
  BM_CHECK(started_ && !finished_ && !error_occurred_);
  timer_->StopTimer();
  if (perf_counters_measurement_) {
    auto measurements = perf_counters_measurement_->StopAndGetMeasurements();
    for (const auto& name_and_measurement : measurements) {
      auto name = name_and_measurement.first;
      auto measurement = name_and_measurement.second;
      BM_CHECK_EQ(counters[name], 0.0);
      counters[name] = Counter(measurement, Counter::kAvgIterations);
    }
  }
}

void State::ResumeTiming() {
  BM_CHECK(started_ && !finished_ && !error_occurred_);
  timer_->StartTimer();
  if (perf_counters_measurement_) {
    perf_counters_measurement_->Start();
  }
}

void State::SkipWithError(const char* msg) {
  BM_CHECK(msg);
  error_occurred_ = true;
  {
    MutexLock l(manager_->GetBenchmarkMutex());
    if (manager_->results.has_error_ == false) {
      manager_->results.error_message_ = msg;
      manager_->results.has_error_ = true;
    }
  }
  total_iterations_ = 0;
  if (timer_->running()) timer_->StopTimer();
}

void State::SetIterationTime(double seconds) {
  timer_->SetIterationTime(seconds);
}

void State::SetLabel(const char* label) {
  MutexLock l(manager_->GetBenchmarkMutex());
  manager_->results.report_label_ = label;
}

void State::StartKeepRunning() {
  BM_CHECK(!started_ && !finished_);
  started_ = true;
  total_iterations_ = error_occurred_ ? 0 : max_iterations;
  manager_->StartStopBarrier();
  if (!error_occurred_) ResumeTiming();
}

void State::FinishKeepRunning() {
  BM_CHECK(started_ && (!finished_ || error_occurred_));
  if (!error_occurred_) {
    PauseTiming();
  }
  // Total iterations has now wrapped around past 0. Fix this.
  total_iterations_ = 0;
  finished_ = true;
  manager_->StartStopBarrier();
}

namespace internal {
namespace {

// Flushes streams after invoking reporter methods that write to them. This
// ensures users get timely updates even when streams are not line-buffered.
void FlushStreams(BenchmarkReporter* reporter) {
  if (!reporter) return;
  std::flush(reporter->GetOutputStream());
  std::flush(reporter->GetErrorStream());
}

// Reports in both display and file reporters.
void Report(BenchmarkReporter* display_reporter,
            BenchmarkReporter* file_reporter, const RunResults& run_results) {
  auto report_one = [](BenchmarkReporter* reporter, bool aggregates_only,
                       const RunResults& results) {
    assert(reporter);
    // If there are no aggregates, do output non-aggregates.
    aggregates_only &= !results.aggregates_only.empty();
    if (!aggregates_only) reporter->ReportRuns(results.non_aggregates);
    if (!results.aggregates_only.empty())
      reporter->ReportRuns(results.aggregates_only);
  };

  report_one(display_reporter, run_results.display_report_aggregates_only,
             run_results);
  if (file_reporter)
    report_one(file_reporter, run_results.file_report_aggregates_only,
               run_results);

  FlushStreams(display_reporter);
  FlushStreams(file_reporter);
}

void RunBenchmarks(const std::vector<BenchmarkInstance>& benchmarks,
                   BenchmarkReporter* display_reporter,
                   BenchmarkReporter* file_reporter) {
  // Note the file_reporter can be null.
  BM_CHECK(display_reporter != nullptr);

  // Determine the width of the name field using a minimum width of 10.
  bool might_have_aggregates = FLAGS_benchmark_repetitions > 1;
  size_t name_field_width = 10;
  size_t stat_field_width = 0;
  for (const BenchmarkInstance& benchmark : benchmarks) {
    name_field_width =
        std::max<size_t>(name_field_width, benchmark.name().str().size());
    might_have_aggregates |= benchmark.repetitions() > 1;

    for (const auto& Stat : benchmark.statistics())
      stat_field_width = std::max<size_t>(stat_field_width, Stat.name_.size());
  }
  if (might_have_aggregates) name_field_width += 1 + stat_field_width;

  // Print header here
  BenchmarkReporter::Context context;
  context.name_field_width = name_field_width;

  // Keep track of running times of all instances of each benchmark family.
  std::map<int /*family_index*/, BenchmarkReporter::PerFamilyRunReports>
      per_family_reports;

  if (display_reporter->ReportContext(context) &&
      (!file_reporter || file_reporter->ReportContext(context))) {
    FlushStreams(display_reporter);
    FlushStreams(file_reporter);

    size_t num_repetitions_total = 0;

    std::vector<internal::BenchmarkRunner> runners;
    runners.reserve(benchmarks.size());
    for (const BenchmarkInstance& benchmark : benchmarks) {
      BenchmarkReporter::PerFamilyRunReports* reports_for_family = nullptr;
      if (benchmark.complexity() != oNone)
        reports_for_family = &per_family_reports[benchmark.family_index()];

      runners.emplace_back(benchmark, reports_for_family);
      int num_repeats_of_this_instance = runners.back().GetNumRepeats();
      num_repetitions_total += num_repeats_of_this_instance;
      if (reports_for_family)
        reports_for_family->num_runs_total += num_repeats_of_this_instance;
    }
    assert(runners.size() == benchmarks.size() && "Unexpected runner count.");

    std::vector<size_t> repetition_indices;
    repetition_indices.reserve(num_repetitions_total);
    for (size_t runner_index = 0, num_runners = runners.size();
         runner_index != num_runners; ++runner_index) {
      const internal::BenchmarkRunner& runner = runners[runner_index];
      std::fill_n(std::back_inserter(repetition_indices),
                  runner.GetNumRepeats(), runner_index);
    }
    assert(repetition_indices.size() == num_repetitions_total &&
           "Unexpected number of repetition indexes.");

    if (FLAGS_benchmark_enable_random_interleaving) {
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(repetition_indices.begin(), repetition_indices.end(), g);
    }

    for (size_t repetition_index : repetition_indices) {
      internal::BenchmarkRunner& runner = runners[repetition_index];
      runner.DoOneRepetition();
      if (runner.HasRepeatsRemaining()) continue;
      // FIXME: report each repetition separately, not all of them in bulk.

      RunResults run_results = runner.GetResults();

      // Maybe calculate complexity report
      if (const auto* reports_for_family = runner.GetReportsForFamily()) {
        if (reports_for_family->num_runs_done ==
            reports_for_family->num_runs_total) {
          auto additional_run_stats = ComputeBigO(reports_for_family->Runs);
          run_results.aggregates_only.insert(run_results.aggregates_only.end(),
                                             additional_run_stats.begin(),
                                             additional_run_stats.end());
          per_family_reports.erase(
              static_cast<int>(reports_for_family->Runs.front().family_index));
        }
      }

      Report(display_reporter, file_reporter, run_results);
    }
  }
  display_reporter->Finalize();
  if (file_reporter) file_reporter->Finalize();
  FlushStreams(display_reporter);
  FlushStreams(file_reporter);
}

// Disable deprecated warnings temporarily because we need to reference
// CSVReporter but don't want to trigger -Werror=-Wdeprecated-declarations
BENCHMARK_DISABLE_DEPRECATED_WARNING

std::unique_ptr<BenchmarkReporter> CreateReporter(
    std::string const& name, ConsoleReporter::OutputOptions output_opts) {
  typedef std::unique_ptr<BenchmarkReporter> PtrType;
  if (name == "console") {
    return PtrType(new ConsoleReporter(output_opts));
  } else if (name == "json") {
    return PtrType(new JSONReporter);
  } else if (name == "csv") {
    return PtrType(new CSVReporter);
  } else {
    std::cerr << "Unexpected format: '" << name << "'\n";
    std::exit(1);
  }
}

BENCHMARK_RESTORE_DEPRECATED_WARNING

}  // end namespace

bool IsZero(double n) {
  return std::abs(n) < std::numeric_limits<double>::epsilon();
}

ConsoleReporter::OutputOptions GetOutputOptions(bool force_no_color) {
  int output_opts = ConsoleReporter::OO_Defaults;
  auto is_benchmark_color = [force_no_color]() -> bool {
    if (force_no_color) {
      return false;
    }
    if (FLAGS_benchmark_color == "auto") {
      return IsColorTerminal();
    }
    return IsTruthyFlagValue(FLAGS_benchmark_color);
  };
  if (is_benchmark_color()) {
    output_opts |= ConsoleReporter::OO_Color;
  } else {
    output_opts &= ~ConsoleReporter::OO_Color;
  }
  if (FLAGS_benchmark_counters_tabular) {
    output_opts |= ConsoleReporter::OO_Tabular;
  } else {
    output_opts &= ~ConsoleReporter::OO_Tabular;
  }
  return static_cast<ConsoleReporter::OutputOptions>(output_opts);
}

}  // end namespace internal

size_t RunSpecifiedBenchmarks() {
  return RunSpecifiedBenchmarks(nullptr, nullptr, FLAGS_benchmark_filter);
}

size_t RunSpecifiedBenchmarks(std::string spec) {
  return RunSpecifiedBenchmarks(nullptr, nullptr, std::move(spec));
}

size_t RunSpecifiedBenchmarks(BenchmarkReporter* display_reporter) {
  return RunSpecifiedBenchmarks(display_reporter, nullptr,
                                FLAGS_benchmark_filter);
}

size_t RunSpecifiedBenchmarks(BenchmarkReporter* display_reporter,
                              std::string spec) {
  return RunSpecifiedBenchmarks(display_reporter, nullptr, std::move(spec));
}

size_t RunSpecifiedBenchmarks(BenchmarkReporter* display_reporter,
                              BenchmarkReporter* file_reporter) {
  return RunSpecifiedBenchmarks(display_reporter, file_reporter,
                                FLAGS_benchmark_filter);
}

size_t RunSpecifiedBenchmarks(BenchmarkReporter* display_reporter,
                              BenchmarkReporter* file_reporter,
                              std::string spec) {
  if (spec.empty() || spec == "all")
    spec = ".";  // Regexp that matches all benchmarks

  // Setup the reporters
  std::ofstream output_file;
  std::unique_ptr<BenchmarkReporter> default_display_reporter;
  std::unique_ptr<BenchmarkReporter> default_file_reporter;
  if (!display_reporter) {
    default_display_reporter = internal::CreateReporter(
        FLAGS_benchmark_format, internal::GetOutputOptions());
    display_reporter = default_display_reporter.get();
  }
  auto& Out = display_reporter->GetOutputStream();
  auto& Err = display_reporter->GetErrorStream();

  std::string const& fname = FLAGS_benchmark_out;
  if (fname.empty() && file_reporter) {
    Err << "A custom file reporter was provided but "
           "--benchmark_out=<file> was not specified."
        << std::endl;
    std::exit(1);
  }
  if (!fname.empty()) {
    output_file.open(fname);
    if (!output_file.is_open()) {
      Err << "invalid file name: '" << fname << "'" << std::endl;
      std::exit(1);
    }
    if (!file_reporter) {
      default_file_reporter = internal::CreateReporter(
          FLAGS_benchmark_out_format, ConsoleReporter::OO_None);
      file_reporter = default_file_reporter.get();
    }
    file_reporter->SetOutputStream(&output_file);
    file_reporter->SetErrorStream(&output_file);
  }

  std::vector<internal::BenchmarkInstance> benchmarks;
  if (!FindBenchmarksInternal(spec, &benchmarks, &Err)) return 0;

  if (benchmarks.empty()) {
    Err << "Failed to match any benchmarks against regex: " << spec << "\n";
    return 0;
  }

  if (FLAGS_benchmark_list_tests) {
    for (auto const& benchmark : benchmarks)
      Out << benchmark.name().str() << "\n";
  } else {
    internal::RunBenchmarks(benchmarks, display_reporter, file_reporter);
  }

  return benchmarks.size();
}

std::string GetBenchmarkFilter() { return FLAGS_benchmark_filter; }

void RegisterMemoryManager(MemoryManager* manager) {
  internal::memory_manager = manager;
}

void AddCustomContext(const std::string& key, const std::string& value) {
  if (internal::global_context == nullptr) {
    internal::global_context = new std::map<std::string, std::string>();
  }
  if (!internal::global_context->emplace(key, value).second) {
    std::cerr << "Failed to add custom context \"" << key << "\" as it already "
              << "exists with value \"" << value << "\"\n";
  }
}

namespace internal {

void PrintUsageAndExit() {
  fprintf(stdout,
          "benchmark"
          " [--benchmark_list_tests={true|false}]\n"
          "          [--benchmark_filter=<regex>]\n"
          "          [--benchmark_min_time=<min_time>]\n"
          "          [--benchmark_repetitions=<num_repetitions>]\n"
          "          [--benchmark_enable_random_interleaving={true|false}]\n"
          "          [--benchmark_report_aggregates_only={true|false}]\n"
          "          [--benchmark_display_aggregates_only={true|false}]\n"
          "          [--benchmark_format=<console|json|csv>]\n"
          "          [--benchmark_out=<filename>]\n"
          "          [--benchmark_out_format=<json|console|csv>]\n"
          "          [--benchmark_color={auto|true|false}]\n"
          "          [--benchmark_counters_tabular={true|false}]\n"
          "          [--benchmark_perf_counters=<counter>,...]\n"
          "          [--benchmark_context=<key>=<value>,...]\n"
          "          [--v=<verbosity>]\n");
  exit(0);
}

void ParseCommandLineFlags(int* argc, char** argv) {
  using namespace benchmark;
  BenchmarkReporter::Context::executable_name =
      (argc && *argc > 0) ? argv[0] : "unknown";
  for (int i = 1; argc && i < *argc; ++i) {
    if (ParseBoolFlag(argv[i], "benchmark_list_tests",
                      &FLAGS_benchmark_list_tests) ||
        ParseStringFlag(argv[i], "benchmark_filter", &FLAGS_benchmark_filter) ||
        ParseDoubleFlag(argv[i], "benchmark_min_time",
                        &FLAGS_benchmark_min_time) ||
        ParseInt32Flag(argv[i], "benchmark_repetitions",
                       &FLAGS_benchmark_repetitions) ||
        ParseBoolFlag(argv[i], "benchmark_enable_random_interleaving",
                      &FLAGS_benchmark_enable_random_interleaving) ||
        ParseBoolFlag(argv[i], "benchmark_report_aggregates_only",
                      &FLAGS_benchmark_report_aggregates_only) ||
        ParseBoolFlag(argv[i], "benchmark_display_aggregates_only",
                      &FLAGS_benchmark_display_aggregates_only) ||
        ParseStringFlag(argv[i], "benchmark_format", &FLAGS_benchmark_format) ||
        ParseStringFlag(argv[i], "benchmark_out", &FLAGS_benchmark_out) ||
        ParseStringFlag(argv[i], "benchmark_out_format",
                        &FLAGS_benchmark_out_format) ||
        ParseStringFlag(argv[i], "benchmark_color", &FLAGS_benchmark_color) ||
        ParseBoolFlag(argv[i], "benchmark_counters_tabular",
                      &FLAGS_benchmark_counters_tabular) ||
        ParseStringFlag(argv[i], "benchmark_perf_counters",
                        &FLAGS_benchmark_perf_counters) ||
        ParseKeyValueFlag(argv[i], "benchmark_context",
                          &FLAGS_benchmark_context) ||
        ParseInt32Flag(argv[i], "v", &FLAGS_v)) {
      for (int j = i; j != *argc - 1; ++j) argv[j] = argv[j + 1];

      --(*argc);
      --i;
    } else if (IsFlag(argv[i], "help")) {
      PrintUsageAndExit();
    }
  }
  for (auto const* flag :
       {&FLAGS_benchmark_format, &FLAGS_benchmark_out_format}) {
    if (*flag != "console" && *flag != "json" && *flag != "csv") {
      PrintUsageAndExit();
    }
  }
  if (FLAGS_benchmark_color.empty()) {
    PrintUsageAndExit();
  }
  for (const auto& kv : FLAGS_benchmark_context) {
    AddCustomContext(kv.first, kv.second);
  }
}

int InitializeStreams() {
  static std::ios_base::Init init;
  return 0;
}

}  // end namespace internal

void Initialize(int* argc, char** argv) {
  internal::ParseCommandLineFlags(argc, argv);
  internal::LogLevel() = FLAGS_v;
}

void Shutdown() { delete internal::global_context; }

bool ReportUnrecognizedArguments(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    fprintf(stderr, "%s: error: unrecognized command-line flag: %s\n", argv[0],
            argv[i]);
  }
  return argc > 1;
}

}  // end namespace benchmark
