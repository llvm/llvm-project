// RUN: mlir-opt %s -test-analysis-deps-multi-requester -verify-diagnostics

// When multiple loaded analyses all depend on the same missing analysis, the
// solver should emit one error with one note per requester.

// expected-error @below {{DataFlowSolver: missing required analyses}}
// expected-note-re @below {{FooDependentAnalysis' requires '{{.*}}FooAnalysis' (not loaded)}}
// expected-note-re @below {{BazDependentAnalysis' requires '{{.*}}FooAnalysis' (not loaded)}}
func.func @missing_dep_multi_requester() {
  return
}
