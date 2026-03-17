// RUN: mlir-opt %s -mlir-print-debuginfo | mlir-opt -mlir-print-debuginfo | FileCheck %s

// Verify that location attributes use #loc(...) and do not conflict with
// trailing loc(...) specifiers in optional assembly format groups.

// CHECK-LABEL: @loc_attr_disambiguation
func.func @loc_attr_disambiguation() {
  // CHECK: test.optional_loc_group #{{.*}} loc(#{{.*}})
  test.optional_loc_group #loc("attr_loc") loc(#loc_base)
  // CHECK: test.optional_loc_group #{{.*}} loc(#{{.*}})
  test.optional_loc_group #loc("attr_loc") loc(#loc_trailing)
  // CHECK: test.optional_loc_group loc(#{{.*}})
  test.optional_loc_group loc(#loc_trailing)
  // CHECK: test.optional_loc_group 42 : i64 loc(#{{.*}})
  test.optional_loc_group 42 : i64 loc(#loc_trailing)
  return
} loc(#loc_base)
#loc_base = #loc(unknown)
#loc_trailing = #loc("trailing_loc")
