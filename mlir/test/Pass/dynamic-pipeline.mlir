// RUN: mlir-opt %s -pass-pipeline='builtin.module(builtin.module(test-dynamic-pipeline{op-name=inner_mod1, dynamic-pipeline=func.func(cse,canonicalize)}))'  --mlir-disable-threading  -mlir-print-ir-before-all 2>&1 | FileCheck %s --check-prefix=MOD1 --check-prefix=MOD1-ONLY --check-prefix=CHECK
// RUN: mlir-opt %s -pass-pipeline='builtin.module(builtin.module(test-dynamic-pipeline{op-name=inner_mod2, dynamic-pipeline=func.func(cse,canonicalize)}))'  --mlir-disable-threading  -mlir-print-ir-before-all 2>&1 | FileCheck %s --check-prefix=MOD2 --check-prefix=MOD2-ONLY --check-prefix=CHECK
// RUN: mlir-opt %s -pass-pipeline='builtin.module(builtin.module(test-dynamic-pipeline{op-name=inner_mod1,inner_mod2, dynamic-pipeline=func.func(cse,canonicalize)}))'  --mlir-disable-threading  -mlir-print-ir-before-all 2>&1 | FileCheck %s --check-prefix=MOD1 --check-prefix=MOD2 --check-prefix=CHECK
// RUN: mlir-opt %s -pass-pipeline='builtin.module(builtin.module(test-dynamic-pipeline{dynamic-pipeline=func.func(cse,canonicalize)}))'  --mlir-disable-threading  -mlir-print-ir-before-all 2>&1 | FileCheck %s --check-prefix=MOD1 --check-prefix=MOD2 --check-prefix=CHECK


func.func @f() {
  return
}

// CHECK: IR Dump Before
// CHECK-SAME: TestDynamicPipelinePass
// CHECK: module @inner_mod1
// MOD2-ONLY: dynamic-pipeline skip op name: inner_mod1
module @inner_mod1 {
// MOD1: Dump Before CSE
// MOD1: @foo
// MOD1: Dump Before Canonicalizer
// MOD1: @foo
  func.func @foo() {
    return
  }
// MOD1: Dump Before CSE
// MOD1: @baz
// MOD1: Dump Before Canonicalizer
// MOD1: @baz
  func.func @baz() {
    return
  }
}

// CHECK: IR Dump Before
// CHECK-SAME: TestDynamicPipelinePass
// CHECK: module @inner_mod2
// MOD1-ONLY: dynamic-pipeline skip op name: inner_mod2
module @inner_mod2 {
// MOD2: Dump Before CSE
// MOD2: @foo
// MOD2: Dump Before Canonicalizer
// MOD2: @foo
  func.func @foo() {
    return
  }
}
