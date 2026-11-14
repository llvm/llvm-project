// RUN: mlir-opt %s -test-analysis-deps -verify-diagnostics

// The `FooDependentAnalysis` test analysis declares a dependency on
// `FooAnalysis` but the test pass only loads `FooDependentAnalysis`, so
// `initializeAndRun` must fail with a diagnostic naming both the requester and
// the missing dep.

// expected-error @below {{DataFlowSolver: missing required analyses}}
// expected-note-re @below {{FooDependentAnalysis' requires '{{.*}}FooAnalysis' (not loaded)}}
func.func @missing_dep() {
  return
}
