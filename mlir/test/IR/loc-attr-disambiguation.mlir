// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// Verify that forward-referencing loc() aliases are not consumed by
// parseOptionalAttribute when probing for optional attribute groups.

// CHECK-LABEL: @loc_forward_ref_with_optional_group
func.func @loc_forward_ref_with_optional_group() {
  // CHECK: test.optional_loc_group
  test.optional_loc_group loc(#loc_fwd)
  // CHECK: test.optional_loc_group 42 : i64
  test.optional_loc_group 42 : i64 loc(#loc_fwd)
  return
} loc(#loc_base)
#loc_base = loc(unknown)
#loc_fwd = loc("forward_ref_test"(#loc_base))
