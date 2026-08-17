// RUN: mlir-opt %s --test-bytecode-roundtrip="test-kind=4" | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/163337
//
// When using test-kind=4, the attribute callback calls the builtin dialect
// reader for each attribute. For dialect-specific attributes (e.g., spirv.*),
// the builtin reader fails and the callback falls through to the regular reader.
// During the failing read, the callback may add entries to the deferred parsing
// worklist. These stale entries must be discarded when the reader position is
// reset, otherwise the `deferredWorklist.empty()` assertion fires in debug
// builds and may corrupt subsequent attribute resolution in release builds.

spirv.module Logical GLSL450 {
  spirv.func @callee() -> () "None" {
    spirv.Kill
  }
  spirv.func @caller() -> () "None" {
    spirv.FunctionCall @callee() : () -> ()
    spirv.Return
  }
}

// CHECK-LABEL: spirv.module Logical GLSL450
// CHECK: spirv.func @callee
// CHECK: spirv.func @caller

// Verify the custom test attribute roundtrips correctly through the callback.
"test.versionedC"() <{attribute = #test.attr_params<42, 24>}> : () -> ()

// CHECK: "test.versionedC"() <{attribute = #test.attr_params<42, 24>}> : () -> ()
