// FIXME: WIP

#include <cassert>
#include <memory>

#include "benchmark/benchmark.h"
#include "output_test.h"

namespace {
class TestProfilerManager : public benchmark::ProfilerManager {
 public:
  void AfterSetupStart() override { ++start_called; }
  void BeforeTeardownStop() override { ++stop_called; }

  int start_called = 0;
  int stop_called = 0;
};

void BM_empty(benchmark::State& state) {
  for (auto _ : state) {
    auto iterations = state.iterations();
    benchmark::DoNotOptimize(iterations);
  }
}
BENCHMARK(BM_empty);

ADD_CASES(TC_ConsoleOut, {{"^BM_empty %console_report$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_empty\",$"},
                       {"\"family_index\": 0,$", MR_Next},
                       {"\"per_family_instance_index\": 0,$", MR_Next},
                       {"\"run_name\": \"BM_empty\",$", MR_Next},
                       {"\"run_type\": \"iteration\",$", MR_Next},
                       {"\"repetitions\": 1,$", MR_Next},
                       {"\"repetition_index\": 0,$", MR_Next},
                       {"\"threads\": 1,$", MR_Next},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\"$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_empty\",%csv_report$"}});
}  // end namespace

int main(int argc, char* argv[]) {
  benchmark::MaybeReenterWithoutASLR(argc, argv);
  std::unique_ptr<TestProfilerManager> pm(new TestProfilerManager());

  benchmark::RegisterProfilerManager(pm.get());
  RunOutputTests(argc, argv);
  benchmark::RegisterProfilerManager(nullptr);

  assert(pm->start_called == 1);
  assert(pm->stop_called == 1);
}
