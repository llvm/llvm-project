
#include <memory>
#undef NDEBUG

#include <chrono>
#include <thread>

#include "../src/timers.h"
#include "benchmark/benchmark.h"

namespace {

const std::chrono::duration<double, std::milli> time_frame(50);
const double time_frame_in_sec(
    std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1>>>(
        time_frame)
        .count());

void MyBusySpinwait() {
  const auto start = benchmark::ChronoClockNow();

  while (true) {
    const auto now = benchmark::ChronoClockNow();
    const auto elapsed = now - start;

    if (std::chrono::duration<double, std::chrono::seconds::period>(elapsed) >=
        time_frame) {
      return;
    }
  }
}

int numRunThreadsCalled_ = 0;

class ManualThreadRunner : public benchmark::ThreadRunnerBase {
 public:
  explicit ManualThreadRunner(int num_threads)
      : pool(static_cast<size_t>(num_threads - 1)) {}

  void RunThreads(const std::function<void(int)>& fn) final {
    for (std::size_t ti = 0; ti < pool.size(); ++ti) {
      pool[ti] = std::thread(fn, static_cast<int>(ti + 1));
    }

    fn(0);

    for (std::thread& thread : pool) {
      thread.join();
    }

    ++numRunThreadsCalled_;
  }

 private:
  std::vector<std::thread> pool;
};

// ========================================================================= //
// --------------------------- TEST CASES BEGIN ---------------------------- //
// ========================================================================= //

// ========================================================================= //
// BM_ManualThreading
// Creation of threads is done before the start of the measurement,
// joining after the finish of the measurement.
void BM_ManualThreading(benchmark::State& state) {
  for (auto _ : state) {
    MyBusySpinwait();
    state.SetIterationTime(time_frame_in_sec);
  }
  state.counters["invtime"] =
      benchmark::Counter{1, benchmark::Counter::kIsRate};
}

}  // end namespace

BENCHMARK(BM_ManualThreading)
    ->Iterations(1)
    ->ThreadRunner([](int num_threads) {
      return std::make_unique<ManualThreadRunner>(num_threads);
    })
    ->Threads(1);
BENCHMARK(BM_ManualThreading)
    ->Iterations(1)
    ->ThreadRunner([](int num_threads) {
      return std::make_unique<ManualThreadRunner>(num_threads);
    })
    ->Threads(1)
    ->UseRealTime();
BENCHMARK(BM_ManualThreading)
    ->Iterations(1)
    ->ThreadRunner([](int num_threads) {
      return std::make_unique<ManualThreadRunner>(num_threads);
    })
    ->Threads(1)
    ->UseManualTime();
BENCHMARK(BM_ManualThreading)
    ->Iterations(1)
    ->ThreadRunner([](int num_threads) {
      return std::make_unique<ManualThreadRunner>(num_threads);
    })
    ->Threads(1)
    ->MeasureProcessCPUTime();
BENCHMARK(BM_ManualThreading)
    ->Iterations(1)
    ->ThreadRunner([](int num_threads) {
      return std::make_unique<ManualThreadRunner>(num_threads);
    })
    ->Threads(1)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK(BM_ManualThreading)
    ->Iterations(1)
    ->ThreadRunner([](int num_threads) {
      return std::make_unique<ManualThreadRunner>(num_threads);
    })
    ->Threads(1)
    ->MeasureProcessCPUTime()
    ->UseManualTime();

BENCHMARK(BM_ManualThreading)
    ->Iterations(1)
    ->ThreadRunner([](int num_threads) {
      return std::make_unique<ManualThreadRunner>(num_threads);
    })
    ->Threads(2);
BENCHMARK(BM_ManualThreading)
    ->Iterations(1)
    ->ThreadRunner([](int num_threads) {
      return std::make_unique<ManualThreadRunner>(num_threads);
    })
    ->Threads(2)
    ->UseRealTime();
BENCHMARK(BM_ManualThreading)
    ->Iterations(1)
    ->ThreadRunner([](int num_threads) {
      return std::make_unique<ManualThreadRunner>(num_threads);
    })
    ->Threads(2)
    ->UseManualTime();
BENCHMARK(BM_ManualThreading)
    ->Iterations(1)
    ->ThreadRunner([](int num_threads) {
      return std::make_unique<ManualThreadRunner>(num_threads);
    })
    ->Threads(2)
    ->MeasureProcessCPUTime();
BENCHMARK(BM_ManualThreading)
    ->Iterations(1)
    ->ThreadRunner([](int num_threads) {
      return std::make_unique<ManualThreadRunner>(num_threads);
    })
    ->Threads(2)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK(BM_ManualThreading)
    ->Iterations(1)
    ->ThreadRunner([](int num_threads) {
      return std::make_unique<ManualThreadRunner>(num_threads);
    })
    ->Threads(2)
    ->MeasureProcessCPUTime()
    ->UseManualTime();

// ========================================================================= //
// ---------------------------- TEST CASES END ----------------------------- //
// ========================================================================= //

int main(int argc, char* argv[]) {
  benchmark::MaybeReenterWithoutASLR(argc, argv);
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  assert(numRunThreadsCalled_ > 0);
}
