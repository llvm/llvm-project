// RUN: %libomptarget-compile-generic && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 | %fcheck-generic -allow-empty -check-prefix=DEBUG
// REQUIRES: libomptarget-debug

/*
  Test for the 'requires' clause check.
  When a target region is used, the requires flags are set in the
  runtime for the entire compilation unit. If the flags are set again,
  (for whatever reason) the set must be consistent with previously
  set values.
*/
#include <stdio.h>
#include <omp.h>

// ---------------------------------------------------------------------------
// Various definitions copied from OpenMP RTL

extern void __tgt_register_requires(int64_t);

// End of definitions copied from OpenMP RTL.
// ---------------------------------------------------------------------------

void run_reg_requires() {
  // Before the target region is registered, the requires registers the status
  // of the requires clauses. Since there are no requires clauses in this file
  // the flags state can only be OMP_REQ_NONE i.e. 1.

  // This is the 2nd time this function is called so it should print the debug
  // info belonging to the check.
  __tgt_register_requires(1);
  __tgt_register_requires(1);
  // DEBUG: New requires flags 1 compatible with existing 1!
}

// ---------------------------------------------------------------------------
int main() {
  run_reg_requires();

// This also runs reg requires for the first time.
#pragma omp target
  {}

  return 0;
}
