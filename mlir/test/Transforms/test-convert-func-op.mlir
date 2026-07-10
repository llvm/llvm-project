// RUN: mlir-opt %s -test-convert-func-op --split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @add
func.func @add(%arg0: i32, %arg1: i32) -> i32 attributes { llvm.emit_c_interface } {
  %res = arith.addi %arg0, %arg1 : i32
  return %res : i32
}
// CHECK-LABEL: llvm.func @_mlir_ciface_add
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]+]]: i32
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]+]]: i32
// CHECK-NEXT: [[RES:%.*]] = llvm.call @add([[ARG0]], [[ARG1]])
// CHECK-NEXT: llvm.return [[RES]]

// -----

// Test that `llvm.byval` arguments are converted to `llvm.ptr` and the actual
// value is retrieved within the `llvm.func`.

// CHECK-LABEL: llvm.func @byval
func.func @byval(%arg0: !test.smpla {llvm.byval = !test.smpla}) -> !test.smpla {
  return %arg0 : !test.smpla
}

// CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr {llvm.byval = !llvm.struct<(i8, i8)>}) -> !llvm.struct<(i8, i8)>
//      CHECK: %[[LD:.*]] = llvm.load %[[ARG0]] : !llvm.ptr -> !llvm.struct<(i8, i8)>
//      CHECK: llvm.return %[[LD]] : !llvm.struct<(i8, i8)>

// -----

// Test that `llvm.byref` arguments are converted to `llvm.ptr` and the actual
// value is retrieved within the `llvm.func`.

// CHECK-LABEL: llvm.func @byref
func.func @byref(%arg0: !test.smpla {llvm.byref = !test.smpla}) -> !test.smpla {
  return %arg0 : !test.smpla
}

// CHECK-SAME: (%[[ARG0:.*]]: !llvm.ptr {llvm.byref = !llvm.struct<(i8, i8)>}) -> !llvm.struct<(i8, i8)>
//      CHECK: %[[LD:.*]] = llvm.load %[[ARG0]] : !llvm.ptr -> !llvm.struct<(i8, i8)>
//      CHECK: llvm.return %[[LD]] : !llvm.struct<(i8, i8)>

// -----

// CHECK-LABEL: llvm.func @affine_for_iter_args
func.func @affine_for_iter_args(%arg0: memref<10xf32>) {
  %0 = affine.for %arg1 = 0 to 10 iter_args(%arg2 = %arg0) -> (memref<10xf32>) {
    affine.yield %arg0 : memref<10xf32>
  }
  return
}

// CHECK: builtin.unrealized_conversion_cast
// CHECK: affine.for
// CHECK: affine.yield
// CHECK: llvm.return
