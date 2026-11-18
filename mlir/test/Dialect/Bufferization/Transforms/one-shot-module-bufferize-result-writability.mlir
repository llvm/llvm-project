// RUN: mlir-opt %s -split-input-file -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" | FileCheck %s
// RUN: mlir-opt %s -split-input-file -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map analysis-heuristic=fuzzer analysis-fuzzer-seed=23" | FileCheck %s

// A function result backed by a constant global must be copied before writing.
// CHECK: memref.global "private" constant
// CHECK-LABEL: func.func private @constant_tensor()
// CHECK:         %[[GLOBAL:.*]] = memref.get_global
// CHECK-NEXT:    return %[[GLOBAL]] : memref<1xi32>
func.func private @constant_tensor() -> tensor<1xi32> {
  %c = arith.constant dense<1> : tensor<1xi32>
  return %c : tensor<1xi32>
}

// CHECK-LABEL: func.func @write_constant_result()
// CHECK:         %[[BASE:.*]] = call @constant_tensor() : () -> memref<1xi32>
// CHECK:         %[[COPY:.*]] = memref.alloc() {{.*}} : memref<1xi32>
// CHECK:         memref.copy %[[BASE]], %[[COPY]]
// CHECK:         vector.scatter %[[COPY]]
// CHECK:         memref.load %[[COPY]]
func.func @write_constant_result() -> i32 {
  %c0 = arith.constant 0 : index
  %base = call @constant_tensor() : () -> tensor<1xi32>
  %indices = arith.constant dense<0> : vector<1xindex>
  %mask = arith.constant dense<true> : vector<1xi1>
  %value = arith.constant dense<2> : vector<1xi32>
  %updated = vector.scatter %base[%c0][%indices], %mask, %value : tensor<1xi32>, vector<1xindex>, vector<1xi1>, vector<1xi32> -> tensor<1xi32>
  %result = tensor.extract %updated[%c0] : tensor<1xi32>
  return %result : i32
}

// -----

// Inspect aliases of return values: tensor.cast itself does not report that its
// result is read-only.
func.func private @constant_cast() -> tensor<?xi32> {
  %c = arith.constant dense<1> : tensor<1xi32>
  %cast = tensor.cast %c : tensor<1xi32> to tensor<?xi32>
  return %cast : tensor<?xi32>
}

// CHECK-LABEL: func.func @write_cast_result()
// CHECK:         %[[BASE:.*]] = call @constant_cast() : () -> memref<?xi32>
// CHECK:         %[[COPY:.*]] = memref.alloc(%{{.*}}) {{.*}} : memref<?xi32>
// CHECK:         memref.copy %[[BASE]], %[[COPY]]
// CHECK:         memref.store %{{.*}}, %[[COPY]]
func.func @write_cast_result() -> i32 {
  %c0 = arith.constant 0 : index
  %v = arith.constant 2 : i32
  %base = call @constant_cast() : () -> tensor<?xi32>
  %updated = tensor.insert %v into %base[%c0] : tensor<?xi32>
  %result = tensor.extract %updated[%c0] : tensor<?xi32>
  return %result : i32
}

// -----

// Propagate read-only results through a chain of calls, including an identity
// function whose result aliases the constant argument at the call site.
func.func private @constant_tensor() -> tensor<1xi32> {
  %c = arith.constant dense<1> : tensor<1xi32>
  return %c : tensor<1xi32>
}
func.func private @identity(%arg: tensor<1xi32>) -> tensor<1xi32> {
  return %arg : tensor<1xi32>
}
func.func private @relay() -> tensor<1xi32> {
  %c = call @constant_tensor() : () -> tensor<1xi32>
  %r = call @identity(%c) : (tensor<1xi32>) -> tensor<1xi32>
  return %r : tensor<1xi32>
}

// CHECK-LABEL: func.func @write_relay_result()
// CHECK:         %[[BASE:.*]] = call @relay() : () -> memref<1xi32>
// CHECK-NEXT:    %[[COPY:.*]] = memref.alloc() {{.*}} : memref<1xi32>
// CHECK-NEXT:    memref.copy %[[BASE]], %[[COPY]]
// CHECK-NEXT:    memref.store %{{.*}}, %[[COPY]]
func.func @write_relay_result() -> i32 {
  %c0 = arith.constant 0 : index
  %v = arith.constant 2 : i32
  %base = call @relay() : () -> tensor<1xi32>
  %updated = tensor.insert %v into %base[%c0] : tensor<1xi32>
  %result = tensor.extract %updated[%c0] : tensor<1xi32>
  return %result : i32
}

// -----

// The write may happen in another callee instead of at the call site itself.
func.func private @constant_tensor() -> tensor<1xi32> {
  %c = arith.constant dense<1> : tensor<1xi32>
  return %c : tensor<1xi32>
}
func.func private @writer(%arg: tensor<1xi32>) -> tensor<1xi32> {
  %c0 = arith.constant 0 : index
  %v = arith.constant 2 : i32
  %updated = tensor.insert %v into %arg[%c0] : tensor<1xi32>
  return %updated : tensor<1xi32>
}

// CHECK-LABEL: func.func @write_in_callee()
// CHECK:         %[[BASE:.*]] = call @constant_tensor() : () -> memref<1xi32>
// CHECK-NEXT:    %[[COPY:.*]] = memref.alloc() {{.*}} : memref<1xi32>
// CHECK-NEXT:    memref.copy %[[BASE]], %[[COPY]]
// CHECK-NEXT:    %{{.*}} = call @writer(%[[COPY]])
func.func @write_in_callee() -> i32 {
  %c0 = arith.constant 0 : index
  %base = call @constant_tensor() : () -> tensor<1xi32>
  %updated = call @writer(%base) : (tensor<1xi32>) -> tensor<1xi32>
  %result = tensor.extract %updated[%c0] : tensor<1xi32>
  return %result : i32
}

// -----

// The first return is writable, but the second may return a constant. All
// return sites must agree before a result can be considered writable.
func.func private @choose(%cond: i1) -> tensor<1xi32> {
  %c = arith.constant dense<1> : tensor<1xi32>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %copy = bufferization.alloc_tensor() copy(%c) : tensor<1xi32>
  return %copy : tensor<1xi32>
^bb2:
  return %c : tensor<1xi32>
}

// CHECK-LABEL: func.func @write_multiple_returns(
// CHECK:         %[[BASE:.*]] = call @choose(%{{.*}}) : (i1) -> memref<1xi32>
// CHECK-NEXT:    %[[COPY:.*]] = memref.alloc() {{.*}} : memref<1xi32>
// CHECK-NEXT:    memref.copy %[[BASE]], %[[COPY]]
// CHECK-NEXT:    memref.store %{{.*}}, %[[COPY]]
func.func @write_multiple_returns(%cond: i1) -> i32 {
  %c0 = arith.constant 0 : index
  %v = arith.constant 2 : i32
  %base = call @choose(%cond) : (i1) -> tensor<1xi32>
  %updated = tensor.insert %v into %base[%c0] : tensor<1xi32>
  %result = tensor.extract %updated[%c0] : tensor<1xi32>
  return %result : i32
}

// -----

// Track writability per result, skipping non-tensor results. A freshly allocated
// result must remain writable without an extra copy at the caller.
func.func private @mixed() -> (i32, tensor<1xi32>, tensor<1xi32>) {
  %v = arith.constant 42 : i32
  %c = arith.constant dense<1> : tensor<1xi32>
  %copy = bufferization.alloc_tensor() copy(%c) : tensor<1xi32>
  return %v, %c, %copy : i32, tensor<1xi32>, tensor<1xi32>
}

// CHECK-LABEL: func.func @write_mixed_results()
// CHECK:         %[[RESULTS:.*]]:3 = call @mixed()
// CHECK-NEXT:    %[[COPY:.*]] = memref.alloc() {{.*}} : memref<1xi32>
// CHECK-NEXT:    memref.copy %[[RESULTS]]#1, %[[COPY]]
// CHECK-NEXT:    memref.store %{{.*}}, %[[COPY]]
// CHECK-NEXT:    memref.store %{{.*}}, %[[RESULTS]]#2
// CHECK-NOT:     memref.alloc
// CHECK-NOT:     memref.copy
// CHECK:         return
func.func @write_mixed_results() -> i32 {
  %c0 = arith.constant 0 : index
  %v = arith.constant 2 : i32
  %results:3 = call @mixed() : () -> (i32, tensor<1xi32>, tensor<1xi32>)
  %a = tensor.insert %v into %results#1[%c0] : tensor<1xi32>
  %b = tensor.insert %v into %results#2[%c0] : tensor<1xi32>
  %x = tensor.extract %a[%c0] : tensor<1xi32>
  %y = tensor.extract %b[%c0] : tensor<1xi32>
  %sum = arith.addi %x, %y : i32
  return %sum : i32
}

// -----

// With no function body, the returned buffer may be read-only.
func.func private @external_tensor() -> tensor<1xi32>

// CHECK-LABEL: func.func @write_external_result()
// CHECK:         %[[BASE:.*]] = call @external_tensor() : () -> memref<1xi32>
// CHECK-NEXT:    %[[COPY:.*]] = memref.alloc() {{.*}} : memref<1xi32>
// CHECK-NEXT:    memref.copy %[[BASE]], %[[COPY]]
// CHECK-NEXT:    memref.store %{{.*}}, %[[COPY]]
func.func @write_external_result() -> i32 {
  %c0 = arith.constant 0 : index
  %v = arith.constant 2 : i32
  %base = call @external_tensor() : () -> tensor<1xi32>
  %updated = tensor.insert %v into %base[%c0] : tensor<1xi32>
  %result = tensor.extract %updated[%c0] : tensor<1xi32>
  return %result : i32
}

// -----

// Function-boundary analysis does not handle recursive calls. Conservatively
// treat their results as read-only as well.
func.func private @recursive(%cond: i1) -> tensor<1xi32> {
  %result = scf.if %cond -> tensor<1xi32> {
    %c = arith.constant dense<1> : tensor<1xi32>
    scf.yield %c : tensor<1xi32>
  } else {
    %r = func.call @recursive(%cond) : (i1) -> tensor<1xi32>
    scf.yield %r : tensor<1xi32>
  }
  return %result : tensor<1xi32>
}

// CHECK-LABEL: func.func @write_recursive_result(
// CHECK:         %[[BASE:.*]] = call @recursive(%{{.*}}) : (i1) -> memref<1xi32>
// CHECK-NEXT:    %[[COPY:.*]] = memref.alloc() {{.*}} : memref<1xi32>
// CHECK-NEXT:    memref.copy %[[BASE]], %[[COPY]]
// CHECK-NEXT:    memref.store %{{.*}}, %[[COPY]]
func.func @write_recursive_result(%cond: i1) -> i32 {
  %c0 = arith.constant 0 : index
  %v = arith.constant 2 : i32
  %base = call @recursive(%cond) : (i1) -> tensor<1xi32>
  %updated = tensor.insert %v into %base[%c0] : tensor<1xi32>
  %result = tensor.extract %updated[%c0] : tensor<1xi32>
  return %result : i32
}
