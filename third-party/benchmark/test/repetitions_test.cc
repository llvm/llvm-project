
#include "benchmark/benchmark.h"
#include "output_test.h"

// ========================================================================= //
// ------------------------ Testing Basic Output --------------------------- //
// ========================================================================= //

static void BM_ExplicitRepetitions(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_ExplicitRepetitions)->Repetitions(2);

ADD_CASES(TC_ConsoleOut,
          {{"^BM_ExplicitRepetitions/repeats:2 %console_report$"}});
ADD_CASES(TC_ConsoleOut,
          {{"^BM_ExplicitRepetitions/repeats:2 %console_report$"}});
ADD_CASES(TC_ConsoleOut,
          {{"^BM_ExplicitRepetitions/repeats:2_mean %console_report$"}});
ADD_CASES(TC_ConsoleOut,
          {{"^BM_ExplicitRepetitions/repeats:2_median %console_report$"}});
ADD_CASES(TC_ConsoleOut,
          {{"^BM_ExplicitRepetitions/repeats:2_stddev %console_report$"}});
ADD_CASES(TC_JSONOut,
          {{"\"name\": \"BM_ExplicitRepetitions/repeats:2\",$"},
           {"\"family_index\": 0,$", MR_Next},
           {"\"per_family_instance_index\": 0,$", MR_Next},
           {"\"run_name\": \"BM_ExplicitRepetitions/repeats:2\",$", MR_Next},
           {"\"run_type\": \"iteration\",$", MR_Next},
           {"\"repetitions\": 2,$", MR_Next},
           {"\"repetition_index\": 0,$", MR_Next},
           {"\"threads\": 1,$", MR_Next},
           {"\"iterations\": %int,$", MR_Next},
           {"\"real_time\": %float,$", MR_Next},
           {"\"cpu_time\": %float,$", MR_Next},
           {"\"time_unit\": \"ns\"$", MR_Next},
           {"}", MR_Next}});
ADD_CASES(TC_JSONOut,
          {{"\"name\": \"BM_ExplicitRepetitions/repeats:2\",$"},
           {"\"family_index\": 0,$", MR_Next},
           {"\"per_family_instance_index\": 0,$", MR_Next},
           {"\"run_name\": \"BM_ExplicitRepetitions/repeats:2\",$", MR_Next},
           {"\"run_type\": \"iteration\",$", MR_Next},
           {"\"repetitions\": 2,$", MR_Next},
           {"\"repetition_index\": 1,$", MR_Next},
           {"\"threads\": 1,$", MR_Next},
           {"\"iterations\": %int,$", MR_Next},
           {"\"real_time\": %float,$", MR_Next},
           {"\"cpu_time\": %float,$", MR_Next},
           {"\"time_unit\": \"ns\"$", MR_Next},
           {"}", MR_Next}});
ADD_CASES(TC_JSONOut,
          {{"\"name\": \"BM_ExplicitRepetitions/repeats:2_mean\",$"},
           {"\"family_index\": 0,$", MR_Next},
           {"\"per_family_instance_index\": 0,$", MR_Next},
           {"\"run_name\": \"BM_ExplicitRepetitions/repeats:2\",$", MR_Next},
           {"\"run_type\": \"aggregate\",$", MR_Next},
           {"\"repetitions\": 2,$", MR_Next},
           {"\"threads\": 1,$", MR_Next},
           {"\"aggregate_name\": \"mean\",$", MR_Next},
           {"\"aggregate_unit\": \"time\",$", MR_Next},
           {"\"iterations\": %int,$", MR_Next},
           {"\"real_time\": %float,$", MR_Next},
           {"\"cpu_time\": %float,$", MR_Next},
           {"\"time_unit\": \"ns\"$", MR_Next},
           {"}", MR_Next}});
ADD_CASES(TC_JSONOut,
          {{"\"name\": \"BM_ExplicitRepetitions/repeats:2_median\",$"},
           {"\"family_index\": 0,$", MR_Next},
           {"\"per_family_instance_index\": 0,$", MR_Next},
           {"\"run_name\": \"BM_ExplicitRepetitions/repeats:2\",$", MR_Next},
           {"\"run_type\": \"aggregate\",$", MR_Next},
           {"\"repetitions\": 2,$", MR_Next},
           {"\"threads\": 1,$", MR_Next},
           {"\"aggregate_name\": \"median\",$", MR_Next},
           {"\"aggregate_unit\": \"time\",$", MR_Next},
           {"\"iterations\": %int,$", MR_Next},
           {"\"real_time\": %float,$", MR_Next},
           {"\"cpu_time\": %float,$", MR_Next},
           {"\"time_unit\": \"ns\"$", MR_Next},
           {"}", MR_Next}});
ADD_CASES(TC_JSONOut,
          {{"\"name\": \"BM_ExplicitRepetitions/repeats:2_stddev\",$"},
           {"\"family_index\": 0,$", MR_Next},
           {"\"per_family_instance_index\": 0,$", MR_Next},
           {"\"run_name\": \"BM_ExplicitRepetitions/repeats:2\",$", MR_Next},
           {"\"run_type\": \"aggregate\",$", MR_Next},
           {"\"repetitions\": 2,$", MR_Next},
           {"\"threads\": 1,$", MR_Next},
           {"\"aggregate_name\": \"stddev\",$", MR_Next},
           {"\"aggregate_unit\": \"time\",$", MR_Next},
           {"\"iterations\": %int,$", MR_Next},
           {"\"real_time\": %float,$", MR_Next},
           {"\"cpu_time\": %float,$", MR_Next},
           {"\"time_unit\": \"ns\"$", MR_Next},
           {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_ExplicitRepetitions/repeats:2\",%csv_report$"}});
ADD_CASES(TC_CSVOut, {{"^\"BM_ExplicitRepetitions/repeats:2\",%csv_report$"}});
ADD_CASES(TC_CSVOut,
          {{"^\"BM_ExplicitRepetitions/repeats:2_mean\",%csv_report$"}});
ADD_CASES(TC_CSVOut,
          {{"^\"BM_ExplicitRepetitions/repeats:2_median\",%csv_report$"}});
ADD_CASES(TC_CSVOut,
          {{"^\"BM_ExplicitRepetitions/repeats:2_stddev\",%csv_report$"}});

// ========================================================================= //
// ------------------------ Testing Basic Output --------------------------- //
// ========================================================================= //

static void BM_ImplicitRepetitions(benchmark::State& state) {
  for (auto _ : state) {
  }
}
BENCHMARK(BM_ImplicitRepetitions);

ADD_CASES(TC_ConsoleOut, {{"^BM_ImplicitRepetitions %console_report$"}});
ADD_CASES(TC_ConsoleOut, {{"^BM_ImplicitRepetitions %console_report$"}});
ADD_CASES(TC_ConsoleOut, {{"^BM_ImplicitRepetitions %console_report$"}});
ADD_CASES(TC_ConsoleOut, {{"^BM_ImplicitRepetitions_mean %console_report$"}});
ADD_CASES(TC_ConsoleOut, {{"^BM_ImplicitRepetitions_median %console_report$"}});
ADD_CASES(TC_ConsoleOut, {{"^BM_ImplicitRepetitions_stddev %console_report$"}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_ImplicitRepetitions\",$"},
                       {"\"family_index\": 1,$", MR_Next},
                       {"\"per_family_instance_index\": 0,$", MR_Next},
                       {"\"run_name\": \"BM_ImplicitRepetitions\",$", MR_Next},
                       {"\"run_type\": \"iteration\",$", MR_Next},
                       {"\"repetitions\": 3,$", MR_Next},
                       {"\"repetition_index\": 0,$", MR_Next},
                       {"\"threads\": 1,$", MR_Next},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\"$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_ImplicitRepetitions\",$"},
                       {"\"family_index\": 1,$", MR_Next},
                       {"\"per_family_instance_index\": 0,$", MR_Next},
                       {"\"run_name\": \"BM_ImplicitRepetitions\",$", MR_Next},
                       {"\"run_type\": \"iteration\",$", MR_Next},
                       {"\"repetitions\": 3,$", MR_Next},
                       {"\"repetition_index\": 1,$", MR_Next},
                       {"\"threads\": 1,$", MR_Next},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\"$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_ImplicitRepetitions\",$"},
                       {"\"family_index\": 1,$", MR_Next},
                       {"\"per_family_instance_index\": 0,$", MR_Next},
                       {"\"run_name\": \"BM_ImplicitRepetitions\",$", MR_Next},
                       {"\"run_type\": \"iteration\",$", MR_Next},
                       {"\"repetitions\": 3,$", MR_Next},
                       {"\"repetition_index\": 2,$", MR_Next},
                       {"\"threads\": 1,$", MR_Next},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\"$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_ImplicitRepetitions_mean\",$"},
                       {"\"family_index\": 1,$", MR_Next},
                       {"\"per_family_instance_index\": 0,$", MR_Next},
                       {"\"run_name\": \"BM_ImplicitRepetitions\",$", MR_Next},
                       {"\"run_type\": \"aggregate\",$", MR_Next},
                       {"\"repetitions\": 3,$", MR_Next},
                       {"\"threads\": 1,$", MR_Next},
                       {"\"aggregate_name\": \"mean\",$", MR_Next},
                       {"\"aggregate_unit\": \"time\",$", MR_Next},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\"$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_ImplicitRepetitions_median\",$"},
                       {"\"family_index\": 1,$", MR_Next},
                       {"\"per_family_instance_index\": 0,$", MR_Next},
                       {"\"run_name\": \"BM_ImplicitRepetitions\",$", MR_Next},
                       {"\"run_type\": \"aggregate\",$", MR_Next},
                       {"\"repetitions\": 3,$", MR_Next},
                       {"\"threads\": 1,$", MR_Next},
                       {"\"aggregate_name\": \"median\",$", MR_Next},
                       {"\"aggregate_unit\": \"time\",$", MR_Next},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\"$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_JSONOut, {{"\"name\": \"BM_ImplicitRepetitions_stddev\",$"},
                       {"\"family_index\": 1,$", MR_Next},
                       {"\"per_family_instance_index\": 0,$", MR_Next},
                       {"\"run_name\": \"BM_ImplicitRepetitions\",$", MR_Next},
                       {"\"run_type\": \"aggregate\",$", MR_Next},
                       {"\"repetitions\": 3,$", MR_Next},
                       {"\"threads\": 1,$", MR_Next},
                       {"\"aggregate_name\": \"stddev\",$", MR_Next},
                       {"\"aggregate_unit\": \"time\",$", MR_Next},
                       {"\"iterations\": %int,$", MR_Next},
                       {"\"real_time\": %float,$", MR_Next},
                       {"\"cpu_time\": %float,$", MR_Next},
                       {"\"time_unit\": \"ns\"$", MR_Next},
                       {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_ImplicitRepetitions\",%csv_report$"}});
ADD_CASES(TC_CSVOut, {{"^\"BM_ImplicitRepetitions\",%csv_report$"}});
ADD_CASES(TC_CSVOut, {{"^\"BM_ImplicitRepetitions_mean\",%csv_report$"}});
ADD_CASES(TC_CSVOut, {{"^\"BM_ImplicitRepetitions_median\",%csv_report$"}});
ADD_CASES(TC_CSVOut, {{"^\"BM_ImplicitRepetitions_stddev\",%csv_report$"}});

// ========================================================================= //
// --------------------------- TEST CASES END ------------------------------ //
// ========================================================================= //

int main(int argc, char* argv[]) { RunOutputTests(argc, argv); }
