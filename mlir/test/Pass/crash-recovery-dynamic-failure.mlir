// Check that local reproducers will also traverse dynamic pass pipelines.
// RUN: mlir-opt %s -pass-pipeline='test-module-pass,test-dynamic-pipeline{op-name=inner_mod1 run-on-nested-operations=1 dynamic-pipeline=test-pass-failure}' -mlir-pass-pipeline-crash-reproducer=%t -verify-diagnostics -mlir-pass-pipeline-local-reproducer --mlir-disable-threading
// RUN: cat %t | FileCheck -check-prefix=REPRO_LOCAL_DYNAMIC_FAILURE %s

// The crash recovery mechanism will leak memory allocated in the crashing thread.
// UNSUPPORTED: asan

module @inner_mod1 {
  // expected-error@below {{Failures have been detected while processing an MLIR pass pipeline}}
  // expected-note@below {{Pipeline failed while executing}}
  module @foo {}
}


// REPRO_LOCAL_DYNAMIC_FAILURE: module @inner_mod1
// REPRO_LOCAL_DYNAMIC_FAILURE: module @foo {

// REPRO_LOCAL_DYNAMIC_FAILURE: pipeline: "builtin.module(test-pass-failure)"
