// clang-format off
// RUN: %libomptarget-compile-generic -DREQ=1 && %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefix=GOOD
// RUN: %libomptarget-compile-generic -DREQ=2 && not %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefix=BAD
// clang-format on

/*
  Test for the 'requires' clause check.
  When a target region is used, the requires flags are set in the
  runtime for the entire compilation unit. If the flags are set again,
  (for whatever reason) the set must be consistent with previously
  set values.
*/
#include <omp.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Various definitions copied from OpenMP RTL

extern void __tgt_register_requires(int64_t);

// End of definitions copied from OpenMP RTL.
// ---------------------------------------------------------------------------

void run_reg_requires() {
  // Before the target region is registered, the requires registers the status
  // of the requires clauses. Since there are no requires clauses in this file
  // the flags state can only be OMP_REQ_NONE i.e. 1.

  // This is the 2nd time this function is called so it should print SUCCESS if
  // REQ is compatible with `1` and otherwise cause an error.
  __tgt_register_requires(1);
  __tgt_register_requires(REQ);

  printf("SUCCESS");

  // clang-format off
  // GOOD: SUCCESS
  // BAD: omptarget fatal error 2: '#pragma omp requires reverse_offload' not used consistently!
  // clang-format on
}

// ---------------------------------------------------------------------------
int main() {
  run_reg_requires();

// This also runs reg requires for the first time.
#pragma omp target
  {}

  return 0;
}
