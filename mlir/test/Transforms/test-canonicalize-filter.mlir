// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s --check-prefix=NO_FILTER
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{enable-patterns=TestRemoveOpWithInnerOps}))' | FileCheck %s --check-prefix=FILTER_ENABLE
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{disable-patterns=TestRemoveOpWithInnerOps}))' | FileCheck %s --check-prefix=FILTER_DISABLE
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{disable-patterns=FoldToCallOpPattern}))' | FileCheck %s --check-prefix=DISABLE_ANON
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{enable-patterns=FoldToCallOpPattern}))' | FileCheck %s --check-prefix=ENABLE_ANON
// A label containing "::" is treated as qualified and requires exact match;
// a wrong-namespace suffix must NOT accidentally match.
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{disable-patterns=foo::FoldToCallOpPattern}))' | FileCheck %s --check-prefix=NO_FILTER
// A non-existent label is a no-op (no pattern gets disabled).
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{disable-patterns=DoesNotExist}))' | FileCheck %s --check-prefix=NO_FILTER
// Mixing multiple labels (both unqualified) in one list.
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{disable-patterns=FoldToCallOpPattern,TestRemoveOpWithInnerOps}))' | FileCheck %s --check-prefix=DISABLE_BOTH

// NO_FILTER-LABEL: func @remove_op_with_inner_ops_pattern
// NO_FILTER-NEXT: return
// FILTER_ENABLE-LABEL: func @remove_op_with_inner_ops_pattern
// FILTER_ENABLE-NEXT: return
// FILTER_DISABLE-LABEL: func @remove_op_with_inner_ops_pattern
// FILTER_DISABLE-NEXT: "test.op_with_region_pattern"()
// DISABLE_BOTH-LABEL: func @remove_op_with_inner_ops_pattern
// DISABLE_BOTH-NEXT: "test.op_with_region_pattern"()
func.func @remove_op_with_inner_ops_pattern() {
  "test.op_with_region_pattern"() ({
    "test.op_with_region_terminator"() : () -> ()
  }) : () -> ()
  return
}

// FoldToCallOpPattern lives in an anonymous namespace; its debug name is
// "(anonymous namespace)::FoldToCallOpPattern". Unqualified filter labels
// match against the suffix after the last "::".

// NO_FILTER-LABEL: func @fold_to_call_unqualified_filter
// NO_FILTER-NEXT: call @callee
// DISABLE_ANON-LABEL: func @fold_to_call_unqualified_filter
// DISABLE_ANON-NEXT: "test.fold_to_call_op"
// DISABLE_BOTH-LABEL: func @fold_to_call_unqualified_filter
// DISABLE_BOTH-NEXT: "test.fold_to_call_op"
// ENABLE_ANON-LABEL: func @fold_to_call_unqualified_filter
// ENABLE_ANON-NEXT: call @callee
func.func private @callee()
func.func @fold_to_call_unqualified_filter() {
  "test.fold_to_call_op"() {callee = @callee} : () -> ()
  return
}
