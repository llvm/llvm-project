// RUN: mlir-opt %s -pass-pipeline='builtin.module(builtin.module(test-module-pass))' --mlir-generate-reproducer=%t -verify-diagnostics
// RUN: cat %t | FileCheck -check-prefix=REPRO %s

module @inner_mod1 {
  module @foo {}
}

// REPRO: module @inner_mod1
// REPRO: module @foo {
// REPRO: pipeline: "builtin.module(any(builtin.module(test-module-pass)))"
