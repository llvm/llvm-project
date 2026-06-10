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

// CHECK-LABEL: func @caller(
// CHECK-SAME:     %[[BUF:.+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK: call @single_buffer_return(%[[BUF]]{{.*}}-> ()
// CHECK: %[[LOADED:.+]] = memref.load %[[BUF]]
// CHECK: return %[[LOADED]]

func.func @caller(%buf: !type, %val: f32, %idx: index) -> f32 {
  %0 = call @single_buffer_return(%buf, %val, %idx) : (!type, f32, index) -> (!type)
  %1 = memref.load %0[%idx] : !type
  return %1 : f32
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

// CHECK-LABEL: func @caller(
// CHECK-SAME:     %[[IN_BUF:.+]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK: %[[RET_VAL:.+]] = call @public_function(%[[IN_BUF]]{{.*}}-> memref
// CHECK: %[[LOADED:.+]] = memref.load %[[RET_VAL]]
// CHECK: return %[[LOADED]]

// MODIFY-PUBLIC-LABEL: func @caller(
// MODIFY-PUBLIC-SAME:    %[[IN_BUF:.+]]: memref<?xf32, strided<[?], offset: ?>>,
// MODIFY-PUBLIC: call @public_function(%[[IN_BUF]]{{.*}}-> ()
// MODIFY-PUBLIC: %[[LOADED:.*]] = memref.load %[[IN_BUF]]
// MODIFY-PUBLIC: return %[[LOADED]]

func.func @caller(%buf: !type, %val: f32, %idx: index) -> f32 {
  %0 = call @public_function(%buf, %val, %idx) : (!type, f32, index) -> (!type)
  %1 = memref.load %0[%idx] : !type
  return %1 : f32
}

// -----

// CHECK-LABEL: func private @negative_external_function(
// CHECK-SAME:    -> memref<?xf32, strided<[?], offset: ?>>

// Ensure external function remains unchanged.
// MODIFY-PUBLIC-LABEL: func private @negative_external_function(
// MODIFY-PUBLIC-SAME:    -> memref<?xf32, strided<[?], offset: ?>>

!type = memref<?xf32, strided<[?], offset: ?>>
func.func private @negative_external_function(%arg0: !type) -> !type
