// RUN: mlir-opt -convert-std-to-llvm -convert-std-to-llvm-emit-c-wrappers %s | FileCheck %s

// This tests the default memref calling convention and the emission of C
// wrappers. We don't need to separate runs because the wrapper-emission
// version subsumes the calling convention and only adds new functions, that we
// can also file-check in the same run.

// An external function is transformed into the glue around calling an interface function.
// CHECK-LABEL: @external
// CHECK: %[[ALLOC0:.*]]: !llvm<"float*">, %[[ALIGN0:.*]]: !llvm<"float*">, %[[OFFSET0:.*]]: !llvm.i64, %[[SIZE00:.*]]: !llvm.i64, %[[SIZE01:.*]]: !llvm.i64, %[[STRIDE00:.*]]: !llvm.i64, %[[STRIDE01:.*]]: !llvm.i64,
// CHECK: %[[ALLOC1:.*]]: !llvm<"float*">, %[[ALIGN1:.*]]: !llvm<"float*">, %[[OFFSET1:.*]]: !llvm.i64)
func @external(%arg0: memref<?x?xf32>, %arg1: memref<f32>)
  // Populate the descriptor for arg0.
  // CHECK: %[[DESC00:.*]] = llvm.mlir.undef : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: %[[DESC01:.*]] = llvm.insertvalue %arg0, %[[DESC00]][0]
  // CHECK: %[[DESC02:.*]] = llvm.insertvalue %arg1, %[[DESC01]][1]
  // CHECK: %[[DESC03:.*]] = llvm.insertvalue %arg2, %[[DESC02]][2]
  // CHECK: %[[DESC04:.*]] = llvm.insertvalue %arg3, %[[DESC03]][3, 0]
  // CHECK: %[[DESC05:.*]] = llvm.insertvalue %arg5, %[[DESC04]][4, 0]
  // CHECK: %[[DESC06:.*]] = llvm.insertvalue %arg4, %[[DESC05]][3, 1]
  // CHECK: %[[DESC07:.*]] = llvm.insertvalue %arg6, %[[DESC06]][4, 1]

  // Allocate on stack and store to comply with C calling convention.
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : index)
  // CHECK: %[[DESC0_ALLOCA:.*]] = llvm.alloca %[[C1]] x !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: llvm.store %[[DESC07]], %[[DESC0_ALLOCA]]

  // Populate the descriptor for arg1.
  // CHECK: %[[DESC10:.*]] = llvm.mlir.undef : !llvm<"{ float*, float*, i64 }">
  // CHECK: %[[DESC11:.*]] = llvm.insertvalue %arg7, %[[DESC10]][0] : !llvm<"{ float*, float*, i64 }">
  // CHECK: %[[DESC12:.*]] = llvm.insertvalue %arg8, %[[DESC11]][1] : !llvm<"{ float*, float*, i64 }">
  // CHECK: %[[DESC13:.*]] = llvm.insertvalue %arg9, %[[DESC12]][2] : !llvm<"{ float*, float*, i64 }">

  // Allocate on stack and store to comply with C calling convention.
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : index)
  // CHECK: %[[DESC1_ALLOCA:.*]] = llvm.alloca %[[C1]] x !llvm<"{ float*, float*, i64 }">
  // CHECK: llvm.store %[[DESC13]], %[[DESC1_ALLOCA]]

  // Call the interface function.
  // CHECK: llvm.call @_mlir_ciface_external

// Verify that an interface function is emitted.
// CHECK-LABEL: llvm.func @_mlir_ciface_external
// CHECK: (!llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">, !llvm<"{ float*, float*, i64 }*">)

// Verify that the return value is not affected.
// CHECK-LABEL: @returner
// CHECK: -> !llvm<"{ { float*, float*, i64, [2 x i64], [2 x i64] }, { float*, float*, i64 } }">
func @returner() -> (memref<?x?xf32>, memref<f32>)

// CHECK-LABEL: @caller
func @caller() {
  %0:2 = call @returner() : () -> (memref<?x?xf32>, memref<f32>)
  // Extract individual values from the descriptor for the first memref.
  // CHECK: %[[ALLOC0:.*]] = llvm.extractvalue %[[DESC0:.*]][0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: %[[ALIGN0:.*]] = llvm.extractvalue %[[DESC0]][1]
  // CHECK: %[[OFFSET0:.*]] = llvm.extractvalue %[[DESC0]][2]
  // CHECK: %[[SIZE00:.*]] = llvm.extractvalue %[[DESC0]][3, 0]
  // CHECK: %[[SIZE01:.*]] = llvm.extractvalue %[[DESC0]][3, 1]
  // CHECK: %[[STRIDE00:.*]] = llvm.extractvalue %[[DESC0]][4, 0]
  // CHECK: %[[STRIDE01:.*]] = llvm.extractvalue %[[DESC0]][4, 1]

  // Extract individual values from the descriptor for the second memref.
  // CHECK: %[[ALLOC1:.*]] = llvm.extractvalue %[[DESC1:.*]][0] : !llvm<"{ float*, float*, i64 }">
  // CHECK: %[[ALIGN1:.*]] = llvm.extractvalue %[[DESC1]][1]
  // CHECK: %[[OFFSET1:.*]] = llvm.extractvalue %[[DESC1]][2]

  // Forward the values to the call.
  // CHECK: llvm.call @external(%[[ALLOC0]], %[[ALIGN0]], %[[OFFSET0]], %[[SIZE00]], %[[SIZE01]], %[[STRIDE00]], %[[STRIDE01]], %[[ALLOC1]], %[[ALIGN1]], %[[OFFSET1]]) : (!llvm<"float*">, !llvm<"float*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64, !llvm<"float*">, !llvm<"float*">, !llvm.i64) -> ()
  call @external(%0#0, %0#1) : (memref<?x?xf32>, memref<f32>) -> ()
  return
}

// CHECK-LABEL: @callee
func @callee(%arg0: memref<?xf32>, %arg1: index) {
  %0 = load %arg0[%arg1] : memref<?xf32>
  return
}

// Verify that an interface function is emitted.
// CHECK-LABEL: @_mlir_ciface_callee
// CHECK: %[[ARG0:.*]]: !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">
  // Load the memref descriptor pointer.
  // CHECK: %[[DESC:.*]] = llvm.load %[[ARG0]] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">

  // Extract individual components of the descriptor.
  // CHECK: %[[ALLOC:.*]] = llvm.extractvalue %[[DESC]][0]
  // CHECK: %[[ALIGN:.*]] = llvm.extractvalue %[[DESC]][1]
  // CHECK: %[[OFFSET:.*]] = llvm.extractvalue %[[DESC]][2]
  // CHECK: %[[SIZE:.*]] = llvm.extractvalue %[[DESC]][3, 0]
  // CHECK: %[[STRIDE:.*]] = llvm.extractvalue %[[DESC]][4, 0]

  // Forward the descriptor components to the call.
  // CHECK: llvm.call @callee(%[[ALLOC]], %[[ALIGN]], %[[OFFSET]], %[[SIZE]], %[[STRIDE]], %{{.*}}) : (!llvm<"float*">, !llvm<"float*">, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> ()

