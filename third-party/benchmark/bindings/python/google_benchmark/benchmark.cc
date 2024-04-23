// Benchmark for Python.

#include "benchmark/benchmark.h"

#include "nanobind/nanobind.h"
#include "nanobind/operators.h"
#include "nanobind/stl/bind_map.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

NB_MAKE_OPAQUE(benchmark::UserCounters);

namespace {
namespace nb = nanobind;

std::vector<std::string> Initialize(const std::vector<std::string>& argv) {
  // The `argv` pointers here become invalid when this function returns, but
  // benchmark holds the pointer to `argv[0]`. We create a static copy of it
  // so it persists, and replace the pointer below.
  static std::string executable_name(argv[0]);
  std::vector<char*> ptrs;
  ptrs.reserve(argv.size());
  for (auto& arg : argv) {
    ptrs.push_back(const_cast<char*>(arg.c_str()));
  }
  ptrs[0] = const_cast<char*>(executable_name.c_str());
  int argc = static_cast<int>(argv.size());
  benchmark::Initialize(&argc, ptrs.data());
  std::vector<std::string> remaining_argv;
  remaining_argv.reserve(argc);
  for (int i = 0; i < argc; ++i) {
    remaining_argv.emplace_back(ptrs[i]);
  }
  return remaining_argv;
}

benchmark::internal::Benchmark* RegisterBenchmark(const std::string& name,
                                                  nb::callable f) {
  return benchmark::RegisterBenchmark(
      name, [f](benchmark::State& state) { f(&state); });
}

NB_MODULE(_benchmark, m) {

  using benchmark::TimeUnit;
  nb::enum_<TimeUnit>(m, "TimeUnit")
      .value("kNanosecond", TimeUnit::kNanosecond)
      .value("kMicrosecond", TimeUnit::kMicrosecond)
      .value("kMillisecond", TimeUnit::kMillisecond)
      .value("kSecond", TimeUnit::kSecond)
      .export_values();

  using benchmark::BigO;
  nb::enum_<BigO>(m, "BigO")
      .value("oNone", BigO::oNone)
      .value("o1", BigO::o1)
      .value("oN", BigO::oN)
      .value("oNSquared", BigO::oNSquared)
      .value("oNCubed", BigO::oNCubed)
      .value("oLogN", BigO::oLogN)
      .value("oNLogN", BigO::oNLogN)
      .value("oAuto", BigO::oAuto)
      .value("oLambda", BigO::oLambda)
      .export_values();

  using benchmark::internal::Benchmark;
  nb::class_<Benchmark>(m, "Benchmark")
      // For methods returning a pointer to the current object, reference
      // return policy is used to ask nanobind not to take ownership of the
      // returned object and avoid calling delete on it.
      // https://pybind11.readthedocs.io/en/stable/advanced/functions.html#return-value-policies
      //
      // For methods taking a const std::vector<...>&, a copy is created
      // because a it is bound to a Python list.
      // https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
      .def("unit", &Benchmark::Unit, nb::rv_policy::reference)
      .def("arg", &Benchmark::Arg, nb::rv_policy::reference)
      .def("args", &Benchmark::Args, nb::rv_policy::reference)
      .def("range", &Benchmark::Range, nb::rv_policy::reference,
           nb::arg("start"), nb::arg("limit"))
      .def("dense_range", &Benchmark::DenseRange,
           nb::rv_policy::reference, nb::arg("start"),
           nb::arg("limit"), nb::arg("step") = 1)
      .def("ranges", &Benchmark::Ranges, nb::rv_policy::reference)
      .def("args_product", &Benchmark::ArgsProduct,
           nb::rv_policy::reference)
      .def("arg_name", &Benchmark::ArgName, nb::rv_policy::reference)
      .def("arg_names", &Benchmark::ArgNames,
           nb::rv_policy::reference)
      .def("range_pair", &Benchmark::RangePair,
           nb::rv_policy::reference, nb::arg("lo1"), nb::arg("hi1"),
           nb::arg("lo2"), nb::arg("hi2"))
      .def("range_multiplier", &Benchmark::RangeMultiplier,
           nb::rv_policy::reference)
      .def("min_time", &Benchmark::MinTime, nb::rv_policy::reference)
      .def("min_warmup_time", &Benchmark::MinWarmUpTime,
           nb::rv_policy::reference)
      .def("iterations", &Benchmark::Iterations,
           nb::rv_policy::reference)
      .def("repetitions", &Benchmark::Repetitions,
           nb::rv_policy::reference)
      .def("report_aggregates_only", &Benchmark::ReportAggregatesOnly,
           nb::rv_policy::reference, nb::arg("value") = true)
      .def("display_aggregates_only", &Benchmark::DisplayAggregatesOnly,
           nb::rv_policy::reference, nb::arg("value") = true)
      .def("measure_process_cpu_time", &Benchmark::MeasureProcessCPUTime,
           nb::rv_policy::reference)
      .def("use_real_time", &Benchmark::UseRealTime,
           nb::rv_policy::reference)
      .def("use_manual_time", &Benchmark::UseManualTime,
           nb::rv_policy::reference)
      .def(
          "complexity",
          (Benchmark * (Benchmark::*)(benchmark::BigO)) & Benchmark::Complexity,
          nb::rv_policy::reference,
          nb::arg("complexity") = benchmark::oAuto);

  using benchmark::Counter;
  nb::class_<Counter> py_counter(m, "Counter");

  nb::enum_<Counter::Flags>(py_counter, "Flags")
      .value("kDefaults", Counter::Flags::kDefaults)
      .value("kIsRate", Counter::Flags::kIsRate)
      .value("kAvgThreads", Counter::Flags::kAvgThreads)
      .value("kAvgThreadsRate", Counter::Flags::kAvgThreadsRate)
      .value("kIsIterationInvariant", Counter::Flags::kIsIterationInvariant)
      .value("kIsIterationInvariantRate",
             Counter::Flags::kIsIterationInvariantRate)
      .value("kAvgIterations", Counter::Flags::kAvgIterations)
      .value("kAvgIterationsRate", Counter::Flags::kAvgIterationsRate)
      .value("kInvert", Counter::Flags::kInvert)
      .export_values()
      .def(nb::self | nb::self);

  nb::enum_<Counter::OneK>(py_counter, "OneK")
      .value("kIs1000", Counter::OneK::kIs1000)
      .value("kIs1024", Counter::OneK::kIs1024)
      .export_values();

  py_counter
      .def(nb::init<double, Counter::Flags, Counter::OneK>(),
           nb::arg("value") = 0., nb::arg("flags") = Counter::kDefaults,
           nb::arg("k") = Counter::kIs1000)
      .def("__init__", ([](Counter *c, double value) { new (c) Counter(value); }))
      .def_rw("value", &Counter::value)
      .def_rw("flags", &Counter::flags)
      .def_rw("oneK", &Counter::oneK)
      .def(nb::init_implicit<double>());

  nb::implicitly_convertible<nb::int_, Counter>();

  nb::bind_map<benchmark::UserCounters>(m, "UserCounters");

  using benchmark::State;
  nb::class_<State>(m, "State")
      .def("__bool__", &State::KeepRunning)
      .def_prop_ro("keep_running", &State::KeepRunning)
      .def("pause_timing", &State::PauseTiming)
      .def("resume_timing", &State::ResumeTiming)
      .def("skip_with_error", &State::SkipWithError)
      .def_prop_ro("error_occurred", &State::error_occurred)
      .def("set_iteration_time", &State::SetIterationTime)
      .def_prop_rw("bytes_processed", &State::bytes_processed,
                    &State::SetBytesProcessed)
      .def_prop_rw("complexity_n", &State::complexity_length_n,
                    &State::SetComplexityN)
      .def_prop_rw("items_processed", &State::items_processed,
                   &State::SetItemsProcessed)
      .def("set_label", &State::SetLabel)
      .def("range", &State::range, nb::arg("pos") = 0)
      .def_prop_ro("iterations", &State::iterations)
      .def_prop_ro("name", &State::name)
      .def_rw("counters", &State::counters)
      .def_prop_ro("thread_index", &State::thread_index)
      .def_prop_ro("threads", &State::threads);

  m.def("Initialize", Initialize);
  m.def("RegisterBenchmark", RegisterBenchmark,
        nb::rv_policy::reference);
  m.def("RunSpecifiedBenchmarks",
        []() { benchmark::RunSpecifiedBenchmarks(); });
  m.def("ClearRegisteredBenchmarks", benchmark::ClearRegisteredBenchmarks);
};
}  // namespace
