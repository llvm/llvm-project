// RUN: mlir-opt %s -pass-pipeline='builtin.func(canonicalize)' | FileCheck %s --check-prefix=NO_FILTER
// RUN: mlir-opt %s -pass-pipeline='builtin.func(canonicalize{enable-patterns=TestRemoveOpWithInnerOps})' | FileCheck %s --check-prefix=FILTER_ENABLE
// RUN: mlir-opt %s -pass-pipeline='builtin.func(canonicalize{disable-patterns=TestRemoveOpWithInnerOps})' | FileCheck %s --check-prefix=FILTER_DISABLE

// NO_FILTER-LABEL: func @remove_op_with_inner_ops_pattern
// NO_FILTER-NEXT: return
// FILTER_ENABLE-LABEL: func @remove_op_with_inner_ops_pattern
// FILTER_ENABLE-NEXT: return
// FILTER_DISABLE-LABEL: func @remove_op_with_inner_ops_pattern
// FILTER_DISABLE-NEXT: "test.op_with_region_pattern"()
func @remove_op_with_inner_ops_pattern() {
  "test.op_with_region_pattern"() ({
    "test.op_with_region_terminator"() : () -> ()
  }) : () -> ()
  return
}
