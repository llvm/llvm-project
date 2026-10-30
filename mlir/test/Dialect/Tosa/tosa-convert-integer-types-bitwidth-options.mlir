// RUN: mlir-opt -tosa-convert-integer-types-bitwidth='bitwidth-mappings=4:8,48:64' %s | FileCheck %s '-D$I4=8' '-D$I48=64'
// RUN: mlir-opt -tosa-convert-integer-types-bitwidth='bitwidth-mappings=4:8' %s | FileCheck %s '-D$I4=8' '-D$I48=48'
// RUN: mlir-opt -tosa-convert-integer-types-bitwidth='bitwidth-mappings=48:64' %s | FileCheck %s '-D$I4=4' '-D$I48=64'
//
// RUN: not mlir-opt -tosa-convert-integer-types-bitwidth %s 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not mlir-opt -tosa-convert-integer-types-bitwidth='bitwidth-mappings=4:16' %s 2>&1 | FileCheck %s --check-prefix=BAD-MAPPING
// RUN: not mlir-opt -tosa-convert-integer-types-bitwidth='bitwidth-mappings=4' %s 2>&1 | FileCheck %s --check-prefix=SYNTAX

// Check selected and unselected widths in both types and constant values.
// CHECK-LABEL: module attributes {
// CHECK-SAME: test.i4 = dense<[-7, 7]> : tensor<2xi[[$I4]]>
// CHECK-SAME: test.i48 = dense<[-1, 140737488355327]> : tensor<2xi[[$I48]]>
// CHECK-SAME: test.types = [i[[$I4]], i[[$I48]], i1, i32, i64]
module attributes {
  test.i4 = dense<[-7, 7]> : tensor<2xi4>,
  test.i48 = dense<[-1, 140737488355327]> : tensor<2xi48>,
  test.types = [i4, i48, i1, i32, i64]
} {}

// MISSING: error: The 'bitwidth-mappings' option must be specified and non-empty.
// BAD-MAPPING: error: unsupported integer bitwidth mapping 4:16: expected 4:8 or 48:64
// SYNTAX: expected <source>:<destination> integer bitwidths
