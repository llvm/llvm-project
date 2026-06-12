// Copyright 2016 Ismael Jimenez Martinez. All rights reserved.
// Copyright 2017 Roman Lebedev. All rights reserved.
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

#include "statistics.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "check.h"

namespace benchmark {

const auto StatisticsSum = [](const std::vector<double>& v) {
  return std::accumulate(v.begin(), v.end(), 0.0);
};

double StatisticsMean(const std::vector<double>& v) {
  if (v.empty()) {
    return 0.0;
  }
  return StatisticsSum(v) * (1.0 / static_cast<double>(v.size()));
}

double StatisticsMedian(const std::vector<double>& v) {
  if (v.size() < 3) {
    return StatisticsMean(v);
  }
  std::vector<double> copy(v);

  auto center = copy.begin() + v.size() / 2;
  std::nth_element(copy.begin(), center, copy.end());

  // Did we have an odd number of samples?  If yes, then center is the median.
  // If not, then we are looking for the average between center and the value
  // before.  Instead of resorting, we just look for the max value before it,
  // which is not necessarily the element immediately preceding `center` Since
  // `copy` is only partially sorted by `nth_element`.
  if (v.size() % 2 == 1) {
    return *center;
  }
  auto center2 = std::max_element(copy.begin(), center);
  return (*center + *center2) / 2.0;
}

// Return the sum of the squares of this sample set
const auto SumSquares = [](const std::vector<double>& v) {
  return std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
};

const auto Sqr = [](const double dat) { return dat * dat; };
const auto Sqrt = [](const double dat) {
  // Avoid NaN due to imprecision in the calculations
  if (dat < 0.0) {
    return 0.0;
  }
  return std::sqrt(dat);
};

double StatisticsStdDev(const std::vector<double>& v) {
  const auto mean = StatisticsMean(v);
  if (v.empty()) {
    return mean;
  }

  // Sample standard deviation is undefined for n = 1
  if (v.size() == 1) {
    return 0.0;
  }

  const double avg_squares =
      SumSquares(v) * (1.0 / static_cast<double>(v.size()));
  return Sqrt(static_cast<double>(v.size()) /
              (static_cast<double>(v.size()) - 1.0) *
              (avg_squares - Sqr(mean)));
}

double StatisticsCV(const std::vector<double>& v) {
  if (v.size() < 2) {
    return 0.0;
  }

  const auto stddev = StatisticsStdDev(v);
  const auto mean = StatisticsMean(v);

  if (std::fpclassify(mean) == FP_ZERO) {
    return 0.0;
  }

  return stddev / mean;
}

std::vector<BenchmarkReporter::Run> ComputeStats(
    const std::vector<BenchmarkReporter::Run>& reports) {
  typedef BenchmarkReporter::Run Run;
  std::vector<Run> results;

  auto error_count = std::count_if(reports.begin(), reports.end(),
                                   [](Run const& run) { return run.skipped; });

  if (reports.size() - static_cast<size_t>(error_count) < 2) {
    // We don't report aggregated data if there was a single run.
    return results;
  }

  // Accumulators.
  std::vector<double> real_accumulated_time_stat;
  std::vector<double> cpu_accumulated_time_stat;

  real_accumulated_time_stat.reserve(reports.size());
  cpu_accumulated_time_stat.reserve(reports.size());

  // All repetitions should be run with the same number of iterations so we
  // can take this information from the first benchmark.
  const IterationCount run_iterations = reports.front().iterations;
  // create stats for user counters
  struct CounterStat {
    Counter c;
    std::vector<double> s;
  };
  std::map<std::string, CounterStat> counter_stats;
  for (Run const& r : reports) {
    for (auto const& cnt : r.counters) {
      auto it = counter_stats.find(cnt.first);
      if (it == counter_stats.end()) {
        it = counter_stats
                 .emplace(cnt.first,
                          CounterStat{cnt.second, std::vector<double>{}})
                 .first;
        it->second.s.reserve(reports.size());
      } else {
        BM_CHECK_EQ(it->second.c.flags, cnt.second.flags);
      }
    }
  }

  // Populate the accumulators.
  for (Run const& run : reports) {
    BM_CHECK_EQ(reports[0].benchmark_name(), run.benchmark_name());
    BM_CHECK_EQ(run_iterations, run.iterations);
    if (run.skipped != 0u) {
      continue;
    }
    real_accumulated_time_stat.emplace_back(run.real_accumulated_time);
    cpu_accumulated_time_stat.emplace_back(run.cpu_accumulated_time);
    // user counters
    for (auto const& cnt : run.counters) {
      auto it = counter_stats.find(cnt.first);
      BM_CHECK_NE(it, counter_stats.end());
      it->second.s.emplace_back(cnt.second);
    }
  }

  // Only add label if it is same for all runs
  std::string report_label = reports[0].report_label;
  for (std::size_t i = 1; i < reports.size(); i++) {
    if (reports[i].report_label != report_label) {
      report_label = "";
      break;
    }
  }

  const double iteration_rescale_factor =
      static_cast<double>(reports.size()) / static_cast<double>(run_iterations);

  for (const auto& Stat : *reports[0].statistics) {
    // Get the data from the accumulator to BenchmarkReporter::Run's.
    Run data;
    data.run_name = reports[0].run_name;
    data.family_index = reports[0].family_index;
    data.per_family_instance_index = reports[0].per_family_instance_index;
    data.run_type = BenchmarkReporter::Run::RT_Aggregate;
    data.threads = reports[0].threads;
    data.repetitions = reports[0].repetitions;
    data.repetition_index = Run::no_repetition_index;
    data.aggregate_name = Stat.name_;
    data.aggregate_unit = Stat.unit_;
    data.report_label = report_label;

    // It is incorrect to say that an aggregate is computed over
    // run's iterations, because those iterations already got averaged.
    // Similarly, if there are N repetitions with 1 iterations each,
    // an aggregate will be computed over N measurements, not 1.
    // Thus it is best to simply use the count of separate reports.
    data.iterations = static_cast<IterationCount>(reports.size());

    data.real_accumulated_time = Stat.compute_(real_accumulated_time_stat);
    data.cpu_accumulated_time = Stat.compute_(cpu_accumulated_time_stat);

    if (data.aggregate_unit == StatisticUnit::kTime) {
      // We will divide these times by data.iterations when reporting, but the
      // data.iterations is not necessarily the scale of these measurements,
      // because in each repetition, these timers are sum over all the iters.
      // And if we want to say that the stats are over N repetitions and not
      // M iterations, we need to multiply these by (N/M).
      data.real_accumulated_time *= iteration_rescale_factor;
      data.cpu_accumulated_time *= iteration_rescale_factor;
    }

    data.time_unit = reports[0].time_unit;

    // user counters
    for (auto const& kv : counter_stats) {
      // Do *NOT* rescale the custom counters. They are already properly scaled.
      const auto uc_stat = Stat.compute_(kv.second.s);
      auto c = Counter(uc_stat, counter_stats[kv.first].c.flags,
                       counter_stats[kv.first].c.oneK);
      data.counters[kv.first] = c;
    }

    results.push_back(data);
  }

  return results;
}

}  // end namespace benchmark
