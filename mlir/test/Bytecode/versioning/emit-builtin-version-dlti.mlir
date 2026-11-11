// RUN: mlir-opt %s -split-input-file -emit-bytecode -emit-bytecode-dialect-version=builtin=0 -verify-diagnostics
// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-dialect-version=builtin=1 | mlir-opt | FileCheck %s

// A property introduced in version 1 cannot be silently dropped for version 0.
// CHECK: module <dlti = #dlti.map<"chip" = "sm_90">>
// expected-error@+1 {{cannot emit builtin dialect version 0 with a 'dlti' property}}
module <{dlti = #dlti.map<"chip" = "sm_90">}> {}

// -----

// Also reject DLTI properties on nested modules.
// CHECK: module @nested <dlti = #dlti.map<"chip" = "sm_90">>
module {
  // expected-error@+1 {{cannot emit builtin dialect version 0 with a 'dlti' property}}
  module @nested <{dlti = #dlti.map<"chip" = "sm_90">}> {}
}
