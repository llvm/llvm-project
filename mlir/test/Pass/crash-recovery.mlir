// RUN: mlir-opt %s -pass-pipeline='builtin.module(test-module-pass, test-pass-crash)' -pass-pipeline-crash-reproducer=%t -verify-diagnostics
// RUN: cat %t | FileCheck -check-prefix=REPRO %s
// RUN: mlir-opt %s -pass-pipeline='builtin.module(test-module-pass, test-pass-crash)' -pass-pipeline-crash-reproducer=%t -verify-diagnostics -pass-pipeline-local-reproducer -mlir-disable-threading
// RUN: cat %t | FileCheck -check-prefix=REPRO_LOCAL %s

// Check that we correctly handle verifiers passes with local reproducer, this used to crash.
// RUN: mlir-opt %s -test-module-pass -test-module-pass  -test-module-pass -pass-pipeline-crash-reproducer=%t -pass-pipeline-local-reproducer -mlir-disable-threading
// RUN: cat %t | FileCheck -check-prefix=REPRO_LOCAL %s

// Check that local reproducers will also traverse dynamic pass pipelines.
// RUN: mlir-opt %s -pass-pipeline='test-module-pass,test-dynamic-pipeline{op-name=inner_mod1 run-on-nested-operations=1 dynamic-pipeline=test-pass-crash}' -pass-pipeline-crash-reproducer=%t -verify-diagnostics -pass-pipeline-local-reproducer --mlir-disable-threading
// RUN: cat %t | FileCheck -check-prefix=REPRO_LOCAL_DYNAMIC %s

// expected-error@below {{Failures have been detected while processing an MLIR pass pipeline}}
// expected-note@below {{Pipeline failed while executing}}
module @inner_mod1 {
  module @foo {}
}

// REPRO: configuration: -pass-pipeline='builtin.module(test-module-pass, test-pass-crash)'

// REPRO: module @inner_mod1
// REPRO: module @foo {

// REPRO_LOCAL: configuration: -pass-pipeline='builtin.module(test-pass-crash)'

// REPRO_LOCAL: module @inner_mod1
// REPRO_LOCAL: module @foo {

// REPRO_LOCAL_DYNAMIC: configuration: -pass-pipeline='builtin.module(test-pass-crash)'

// REPRO_LOCAL_DYNAMIC: module @inner_mod1
// REPRO_LOCAL_DYNAMIC: module @foo {
