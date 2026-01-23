// RUN: mlir-opt -drop-equivalent-buffer-results -split-input-file %s | FileCheck %s
// RUN: mlir-opt -drop-equivalent-buffer-results=modify-public-functions=1 -split-input-file %s | \
// RUN:   FileCheck %s --check-prefix=MODIFY-PUBLIC


// CHECK-LABEL: func private @single_buffer_return({{.*}}) {
// CHECK: return

!type = memref<?xf32, strided<[?], offset: ?>>
func.func private @single_buffer_return(%buf: !type, %val: f32, %idx: index) -> !type {
  memref.store %val, %buf[%idx] : !type
  return %buf : !type
}

// -----

// CHECK-LABEL: func private @multiple_buffer_returns({{.*}}) {
// CHECK: return

!type = memref<?xf32, strided<[?], offset: ?>>
!type1 = memref<?x?xf32>
func.func private @multiple_buffer_returns(
    %buf: !type, %buf1: !type1, %val: f32, %idx: index) -> (!type1, !type) {
  memref.store %val, %buf[%idx] : !type
  memref.store %val, %buf1[%idx, %idx] : !type1
  return %buf1, %buf : !type1, !type
}

// -----

// CHECK-LABEL: func private @multiple_mixed_returns({{.*}}) -> i32 {
// CHECK: %[[CST:.+]] = arith.constant 1 : i32
// CHECK: return %[[CST]] : i32

!type = memref<?xf32, strided<[?], offset: ?>>
!type1 = memref<?x?xf32>
func.func private @multiple_mixed_returns(
    %buf: !type, %buf1: !type1, %val: f32, %idx: index) -> (!type1, i32, !type) {
  memref.store %val, %buf[%idx] : !type
  memref.store %val, %buf1[%idx, %idx] : !type1
  %cst = arith.constant 1 : i32
  return %buf1, %cst, %buf : !type1, i32, !type
}

// -----

// Ensure public functions remain unchanged by default.
// CHECK-LABEL: func @public_function(
// CHECK-SAME:    %[[BUF:.+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME:    ) -> memref<?xf32, strided<[?], offset: ?>> {
// CHECK: return %[[BUF]]

// When explicitly requested, public functions can be modified.
// MODIFY-PUBLIC-LABEL: func @public_function(
// MODIFY-PUBLIC-SAME:    %[[BUF:.+]]: memref<?xf32, strided<[?], offset: ?>>,
// MODIFY-PUBLIC-SAME:    ) {
// MODIFY-PUBLIC: return

!type = memref<?xf32, strided<[?], offset: ?>>
func.func @public_function(
    %buf: !type, %val: f32, %idx: index) -> !type {
  memref.store %val, %buf[%idx] : !type
  return %buf : !type
}

// -----

// CHECK-LABEL: func private @negative_external_function(
// CHECK-SAME:    -> memref<?xf32, strided<[?], offset: ?>>

// Ensure external function remains unchanged.
// MODIFY-PUBLIC-LABEL: func private @negative_external_function(
// MODIFY-PUBLIC-SAME:    -> memref<?xf32, strided<[?], offset: ?>>

!type = memref<?xf32, strided<[?], offset: ?>>
func.func private @negative_external_function(%arg0: !type) -> !type
