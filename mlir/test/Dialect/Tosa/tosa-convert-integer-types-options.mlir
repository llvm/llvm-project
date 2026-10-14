// RUN: mlir-opt -tosa-convert-integer-types='bitwidth-mappings=4:8,48:64' %s | FileCheck %s '-D$I4=8' '-D$I48=64'
// RUN: mlir-opt -tosa-convert-integer-types='bitwidth-mappings=48:64,4:8,48:64' %s | FileCheck %s '-D$I4=8' '-D$I48=64'
// RUN: mlir-opt -tosa-convert-integer-types='bitwidth-mappings=4:8' %s | FileCheck %s '-D$I4=8' '-D$I48=48'
// RUN: mlir-opt -tosa-convert-integer-types='bitwidth-mappings=48:64' %s | FileCheck %s '-D$I4=4' '-D$I48=64'
// RUN: mlir-opt -tosa-convert-integer-types='bitwidth-mappings=32:32' %s | FileCheck %s '-D$I4=4' '-D$I48=48'
// RUN: mlir-opt -tosa-convert-integer-types='bitwidth-mappings=4:8' %s | mlir-opt -tosa-convert-integer-types='bitwidth-mappings=48:64' | FileCheck %s '-D$I4=8' '-D$I48=64'
// RUN: mlir-opt -tosa-convert-integer-types='bitwidth-mappings=48:64' %s | mlir-opt -tosa-convert-integer-types='bitwidth-mappings=4:8' | FileCheck %s '-D$I4=8' '-D$I48=64'
//
// RUN: not mlir-opt -tosa-convert-integer-types %s 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not mlir-opt -pass-pipeline='builtin.module(tosa-convert-integer-types)' %s 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not mlir-opt -tosa-convert-integer-types='bitwidth-mappings=' %s 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not mlir-opt -tosa-convert-integer-types='bitwidth-mappings=0:8,4:8' %s 2>&1 | FileCheck %s --check-prefix=ZERO
// RUN: not mlir-opt -tosa-convert-integer-types='bitwidth-mappings=16777216:16777216' %s 2>&1 | FileCheck %s --check-prefix=TOO-LARGE
// RUN: not mlir-opt -tosa-convert-integer-types='bitwidth-mappings=8388609:16777216' %s 2>&1 | FileCheck %s --check-prefix=DESTINATION
// RUN: not mlir-opt -tosa-convert-integer-types='bitwidth-mappings=4:0' %s 2>&1 | FileCheck %s --check-prefix=DESTINATION-ZERO
// RUN: not mlir-opt -tosa-convert-integer-types='bitwidth-mappings=4:8,4:16' %s 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not mlir-opt -tosa-convert-integer-types='bitwidth-mappings=4' %s 2>&1 | FileCheck %s --check-prefix=SYNTAX
// RUN: not mlir-opt -tosa-convert-integer-types='bitwidth-mappings=4:' %s 2>&1 | FileCheck %s --check-prefix=SYNTAX
// RUN: not mlir-opt -tosa-convert-integer-types='bitwidth-mappings=4:8:16' %s 2>&1 | FileCheck %s --check-prefix=SYNTAX
// RUN: not mlir-opt -tosa-convert-integer-types='bitwidth-mappings=-4:8' %s 2>&1 | FileCheck %s --check-prefix=SYNTAX
// RUN: not mlir-opt -tosa-convert-integer-types='bitwidth-mappings=4:foo' %s 2>&1 | FileCheck %s --check-prefix=SYNTAX

// Both conversions must update function signatures, calls, region arguments and
// nested attributes consistently. The options also control constant payloads.
// The i32 metadata exercises an explicit identity mapping as well as unmapped types.
// CHECK: func.func private @callee(tensor<2xi[[$I4]]>, tensor<2xi[[$I48]]>) -> (tensor<2xi[[$I4]]>, tensor<2xi[[$I48]]>)
func.func private @callee(tensor<2xi4>, tensor<2xi48>) -> (tensor<2xi4>, tensor<2xi48>)

// CHECK-LABEL: func.func @caller(
// CHECK-SAME: %{{.*}}: tensor<i1>) -> (tensor<2xi[[$I4]]>, tensor<2xi[[$I48]]>)
func.func @caller(%cond: tensor<i1>) -> (tensor<2xi4>, tensor<2xi48>) {
  // CHECK: tosa.const values(dense<[-7, 7]> : tensor<2xi[[$I4]]>)
  %a = tosa.const values(dense<[-7, 7]> : tensor<2xi4>) : () -> tensor<2xi4>
  // CHECK: tosa.const values(dense<[-1, 140737488355327]> : tensor<2xi[[$I48]]>)
  %b = tosa.const values(dense<[-1, 140737488355327]> : tensor<2xi48>) : () -> tensor<2xi48>
  // CHECK: call @callee(%{{.*}}, %{{.*}}) {test.types = [i[[$I4]], i[[$I48]], i32]} : (tensor<2xi[[$I4]]>, tensor<2xi[[$I48]]>) -> (tensor<2xi[[$I4]]>, tensor<2xi[[$I48]]>)
  %c:2 = func.call @callee(%a, %b) {test.types = [i4, i48, i32]} : (tensor<2xi4>, tensor<2xi48>) -> (tensor<2xi4>, tensor<2xi48>)
  %d:2 = tosa.cond_if %cond (%x = %c#0, %y = %c#1) : tensor<i1> (tensor<2xi4>, tensor<2xi48>) -> (tensor<2xi4>, tensor<2xi48>) {
  // CHECK: ^bb0(%{{.*}}: tensor<2xi[[$I4]]>, %{{.*}}: tensor<2xi[[$I48]]>):
  ^bb0(%x: tensor<2xi4>, %y: tensor<2xi48>):
    tosa.yield %x, %y : tensor<2xi4>, tensor<2xi48>
  } else {
  ^bb0(%x: tensor<2xi4>, %y: tensor<2xi48>):
    tosa.yield %x, %y : tensor<2xi4>, tensor<2xi48>
  }
  return %d#0, %d#1 : tensor<2xi4>, tensor<2xi48>
}

// MISSING: error: The 'bitwidth-mappings' option must be specified and non-empty.
// ZERO: error: invalid source integer bitwidth 0: expected a value between 1 and 16777215
// TOO-LARGE: error: invalid source integer bitwidth 16777216: expected a value between 1 and 16777215
// DESTINATION: error: invalid destination integer bitwidth 16777216: expected a value between 1 and 16777215
// DESTINATION-ZERO: error: invalid destination integer bitwidth 0: expected a value between 1 and 16777215
// CONFLICT: error: conflicting mappings for source integer bitwidth 4: 8 and 16
// SYNTAX: expected <source>:<destination> integer bitwidths
