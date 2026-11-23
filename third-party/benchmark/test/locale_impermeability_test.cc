#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdlib>

#include "benchmark/benchmark.h"
#include "output_test.h"

namespace {
void BM_ostream(benchmark::State& state) {
#if !defined(__MINGW64__) || defined(__clang__)
  // GCC-based versions of MINGW64 do not support locale manipulations,
  // don't run the test under them.
  std::locale::global(std::locale("en_US.UTF-8"));
#endif
  while (state.KeepRunning()) {
    state.SetIterationTime(1e-6);
  }
}
BENCHMARK(BM_ostream)->UseManualTime()->Iterations(1000000);

ADD_CASES(TC_ConsoleOut, {{"^BM_ostream/iterations:1000000/manual_time"
                           " %console_report$"}});
ADD_CASES(TC_JSONOut,
          {{"\"name\": \"BM_ostream/iterations:1000000/manual_time\",$"},
           {"\"family_index\": 0,$", MR_Next},
           {"\"per_family_instance_index\": 0,$", MR_Next},
           {"\"run_name\": "
            "\"BM_ostream/iterations:1000000/manual_time\",$",
            MR_Next},
           {"\"run_type\": \"iteration\",$", MR_Next},
           {"\"repetitions\": 1,$", MR_Next},
           {"\"repetition_index\": 0,$", MR_Next},
           {"\"threads\": 1,$", MR_Next},
           {"\"iterations\": 1000000,$", MR_Next},
           {"\"real_time\": %float,$", MR_Next},
           {"\"cpu_time\": %float,$", MR_Next},
           {"\"time_unit\": \"ns\"$", MR_Next},
           {"}", MR_Next}});
ADD_CASES(TC_CSVOut, {{"^\"BM_ostream/iterations:1000000/"
                       "manual_time\",1000000,%float,%float,ns,,,,,$"}});
}  // end namespace

int main(int argc, char* argv[]) {
  benchmark::MaybeReenterWithoutASLR(argc, argv);
  RunOutputTests(argc, argv);
}
