// RUN: mlir-opt %s --split-input-file --verify-diagnostics

// expected-error @below {{domain must be any SMT value type}}
func.func @array_domain_no_smt_type(%arg0: !smt.array<[i32 -> !smt.bool]>) {
  return
}

// -----

// expected-error @below {{range must be any SMT value type}}
func.func @array_range_no_smt_type(%arg0: !smt.array<[!smt.bool -> i32]>) {
  return
}
