#include "benchmark/benchmark.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;
using namespace Intrinsic;

// Benchmark intrinsic lookup from a variety of targets.
static void BM_GetIntrinsicForClangBuiltin(benchmark::State &state) {
  static const char *Builtins[] = {
      "__builtin_adjust_trampoline",
      "__builtin_trap",
      "__builtin_arm_ttest",
      "__builtin_amdgcn_cubetc",
      "__builtin_amdgcn_udot2",
      "__builtin_arm_stc",
      "__builtin_bpf_compare",
      "__builtin_HEXAGON_A2_max",
      "__builtin_lasx_xvabsd_b",
      "__builtin_mips_dlsa",
      "__nvvm_floor_f",
      "__builtin_altivec_vslb",
      "__builtin_r600_read_tgid_x",
      "__builtin_riscv_aes64im",
      "__builtin_s390_vcksm",
      "__builtin_ve_vl_pvfmksge_Mvl",
      "__builtin_ia32_axor64",
      "__builtin_bitrev",
  };
  static const char *Targets[] = {"",     "aarch64", "amdgcn", "mips",
                                  "nvvm", "r600",    "riscv"};

  for (auto _ : state) {
    for (auto Builtin : Builtins)
      for (auto Target : Targets)
        getIntrinsicForClangBuiltin(Target, Builtin);
  }
}

static void
BM_GetIntrinsicForClangBuiltinHexagonFirst(benchmark::State &state) {
  // Exercise the worst case by looking for the first builtin for a target
  // that has a lot of builtins.
  for (auto _ : state)
    getIntrinsicForClangBuiltin("hexagon", "__builtin_HEXAGON_A2_abs");
}

BENCHMARK(BM_GetIntrinsicForClangBuiltin);
BENCHMARK(BM_GetIntrinsicForClangBuiltinHexagonFirst);

BENCHMARK_MAIN();
