// RUN: mlir-opt %s -one-shot-bufferize="allow-unknown-ops" -verify-diagnostics -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-heuristic=fuzzer analysis-fuzzer-seed=23" -verify-diagnostics -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-heuristic=fuzzer analysis-fuzzer-seed=59" -verify-diagnostics -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-heuristic=fuzzer analysis-fuzzer-seed=91" -verify-diagnostics -split-input-file -o /dev/null

// Run with top-down analysis.
// RUN: mlir-opt %s -one-shot-bufferize="allow-unknown-ops analysis-heuristic=top-down" -verify-diagnostics -split-input-file | FileCheck %s --check-prefix=CHECK-TOP-DOWN-ANALYSIS

// Test without analysis: Insert a copy on every buffer write.
// RUN: mlir-opt %s -allow-unregistered-dialect -one-shot-bufferize="allow-unknown-ops copy-before-write" -split-input-file | FileCheck %s --check-prefix=CHECK-COPY-BEFORE-WRITE

// CHECK-LABEL: func @no_conflict
//       CHECK:   memref.alloc
//       CHECK:   memref.store
//  CHECK-NEXT:   memref.store
//  CHECK-NEXT:   memref.store
//  CHECK-NEXT:   memref.store
// CHECK-COPY-BEFORE-WRITE-LABEL: func @no_conflict
//       CHECK-COPY-BEFORE-WRITE:   memref.alloc
//       CHECK-COPY-BEFORE-WRITE:   memref.store
//       CHECK-COPY-BEFORE-WRITE:   memref.store
//       CHECK-COPY-BEFORE-WRITE:   memref.store
//       CHECK-COPY-BEFORE-WRITE:   memref.alloc
//       CHECK-COPY-BEFORE-WRITE:   memref.copy
//       CHECK-COPY-BEFORE-WRITE:   memref.store
func.func @no_conflict(%fill: f32, %f: f32, %idx: index) -> tensor<3xf32> {
  %t = tensor.from_elements %fill, %fill, %fill : tensor<3xf32>
  %i = tensor.insert %f into %t[%idx] : tensor<3xf32>
  return %i : tensor<3xf32>
}

// -----

// CHECK-LABEL: func @use_tensor_func_arg(
//  CHECK-SAME:     %[[A:.*]]: tensor<?xf32>
func.func @use_tensor_func_arg(%A : tensor<?xf32>) -> (vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32

  // CHECK: %[[A_memref:.*]] = bufferization.to_buffer %[[A]]
  // CHECK: %[[res:.*]] = vector.transfer_read %[[A_memref]]
  %0 = vector.transfer_read %A[%c0], %f0 : tensor<?xf32>, vector<4xf32>

  // CHECK: return %[[res]]
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: func @return_tensor(
//  CHECK-SAME:     %[[A:.*]]: tensor<?xf32>
func.func @return_tensor(%A : tensor<?xf32>, %v : vector<4xf32>) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index

  // CHECK: %[[A_memref:.*]] = bufferization.to_buffer %[[A]]
  // CHECK: %[[dim:.*]] = memref.dim %[[A_memref]]
  // CHECK: %[[alloc:.*]] = memref.alloc(%[[dim]])
  // CHECK: memref.copy %[[A_memref]], %[[alloc]]
  // CHECK: vector.transfer_write %{{.*}}, %[[alloc]]
  // CHECK: %[[res_tensor:.*]] = bufferization.to_tensor %[[alloc]]
  %0 = vector.transfer_write %v, %A[%c0] : vector<4xf32>, tensor<?xf32>

  // CHECK: return %[[res_tensor]]
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @func_without_tensor_args
func.func @func_without_tensor_args(%v : vector<10xf32>) -> () {
  // CHECK: %[[alloc:.*]] = memref.alloc()
  %0 = bufferization.alloc_tensor() : tensor<10xf32>

  %c0 = arith.constant 0 : index
  // CHECK: vector.transfer_write %{{.*}}, %[[alloc]]
  %1 = vector.transfer_write %v, %0[%c0] : vector<10xf32>, tensor<10xf32>

  %cst = arith.constant 0.0 : f32
  // CHECK: vector.transfer_read %[[alloc]]
  %r = vector.transfer_read %1[%c0], %cst : tensor<10xf32>, vector<11xf32>

  vector.print %r : vector<11xf32>
  return
}

// -----

// CHECK-LABEL: func private @private_func
func.func private @private_func(tensor<?xf32>) -> ()

// CHECK-LABEL: func @empty_func()
func.func @empty_func() -> () {
  return
}

// -----

// CHECK-LABEL: func @read_after_write_conflict(
func.func @read_after_write_conflict(%cst : f32, %idx : index, %idx2 : index)
    -> (f32, f32) {
  // CHECK-DAG: %[[alloc:.*]] = memref.alloc
  // CHECK-DAG: %[[dummy:.*]] = "test.dummy_op"
  // CHECK-DAG: %[[dummy_m:.*]] = bufferization.to_buffer %[[dummy]]
  %t = "test.dummy_op"() : () -> (tensor<10xf32>)

  // CHECK: memref.copy %[[dummy_m]], %[[alloc]]
  // CHECK: memref.store %{{.*}}, %[[alloc]]
  %write = tensor.insert %cst into %t[%idx2] : tensor<10xf32>

  // CHECK: %[[read:.*]] = "test.some_use"(%[[dummy]])
  %read = "test.some_use"(%t) : (tensor<10xf32>) -> (f32)
  // CHECK: %[[read2:.*]] = memref.load %[[alloc]]
  %read2 = tensor.extract %write[%idx] : tensor<10xf32>

  // CHECK: return %[[read]], %[[read2]]
  return %read, %read2 : f32, f32
}

// -----

// CHECK-LABEL: func @copy_deallocated(
func.func @copy_deallocated() -> tensor<10xf32> {
  // CHECK: %[[alloc:.*]] = memref.alloc()
  %0 = bufferization.alloc_tensor() : tensor<10xf32>
  // CHECK: %[[alloc_tensor:.*]] = bufferization.to_tensor %[[alloc]]
  // CHECK: return %[[alloc_tensor]]
  return %0 : tensor<10xf32>
}

// -----

// CHECK-LABEL: func @select_different_tensors(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
func.func @select_different_tensors(%t: tensor<?xf32>, %sz: index, %pos: index, %c: i1) -> f32 {
  // CHECK-DAG: %[[m:.*]] = bufferization.to_buffer %[[t]] : tensor<?xf32> to memref<?xf32, strided{{.*}}>
  // CHECK-DAG: %[[alloc:.*]] = memref.alloc(%{{.*}}) {{.*}} : memref<?xf32>
  %0 = bufferization.alloc_tensor(%sz) : tensor<?xf32>

  // A cast must be inserted because %t and %0 have different memref types.
  // CHECK: %[[casted:.*]] = memref.cast %[[alloc]] : memref<?xf32> to memref<?xf32, strided{{.*}}>
  // CHECK: arith.select %{{.*}}, %[[casted]], %[[m]]
  %1 = arith.select %c, %0, %t : tensor<?xf32>
  %2 = tensor.extract %1[%pos] : tensor<?xf32>
  return %2 : f32
}

// -----

// CHECK-LABEL: func @alloc_tensor_with_copy(
//  CHECK-SAME:     %[[t:.*]]: tensor<5xf32>)
// TODO: Add a test case with dynamic dim size. This is not possible at the
// moment because this would create a tensor op during bufferization. That is
// currently forbidden.
func.func @alloc_tensor_with_copy(%t: tensor<5xf32>) -> tensor<5xf32> {
  // CHECK: %[[m:.*]] = bufferization.to_buffer %[[t]]
  // CHECK: %[[alloc:.*]] = memref.alloc() {{.*}} : memref<5xf32>
  // CHECK: memref.copy %[[m]], %[[alloc]]
  %0 = bufferization.alloc_tensor() copy(%t) : tensor<5xf32>
  // CHECK: %[[r:.*]] = bufferization.to_tensor %[[alloc]]
  // CHECK: return %[[r]]
  return %0 : tensor<5xf32>
}

// -----

// CHECK-LABEL: func @alloc_tensor_with_memory_space()
func.func @alloc_tensor_with_memory_space() -> tensor<5xf32> {
  // CHECK: %[[alloc:.*]] = memref.alloc() {{.*}} : memref<5xf32, 1>
  %0 = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<5xf32>
  // CHECK: %[[r:.*]] = bufferization.to_tensor %[[alloc]]
  // CHECK: return %[[r]]
  return %0 : tensor<5xf32>
}

// -----

// CHECK-LABEL: func @read_of_alias
// CHECK-TOP-DOWN-ANALYSIS-LABEL: func @read_of_alias
func.func @read_of_alias(%t: tensor<100xf32>, %pos1: index, %pos2: index,
                         %pos3: index, %pos4: index, %sz: index, %f: f32)
  -> (f32, f32)
{
  // CHECK: %[[alloc:.*]] = memref.alloc
  // CHECK: memref.copy
  // CHECK: memref.store %{{.*}}, %[[alloc]]
  // CHECK-TOP-DOWN-ANALYSIS: %[[alloc:.*]] = memref.alloc
  // CHECK-TOP-DOWN-ANALYSIS: memref.copy
  // CHECK-TOP-DOWN-ANALYSIS: memref.store %{{.*}}, %[[alloc]]
  %0 = tensor.insert %f into %t[%pos1] : tensor<100xf32>
  %1 = tensor.extract_slice %t[%pos2][%sz][1] : tensor<100xf32> to tensor<?xf32>
  %2 = tensor.extract %1[%pos3] : tensor<?xf32>
  %3 = tensor.extract %0[%pos3] : tensor<100xf32>
  return %2, %3 : f32, f32
}

// -----

// CHECK-LABEL: func @from_unranked_to_unranked(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<*xi32>
func.func @from_unranked_to_unranked(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK: %[[m:.*]] = bufferization.to_buffer %[[arg0]] : tensor<*xi32> to memref<*xi32>
  // CHECK: %[[t:.*]] = bufferization.to_tensor %[[m]]
  // CHECK: return %[[t]] : tensor<*xi32>
  %0 = tensor.cast %arg0 : tensor<*xi32> to tensor<*xi32>
  return %0 : tensor<*xi32>
}

// -----

// CHECK-LABEL: func @tensor_copy(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<5xf32>)
func.func @tensor_copy(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  // CHECK: %[[m:.*]] = bufferization.to_buffer %[[arg0]]
  // CHECK: %[[alloc:.*]] = memref.alloc() {{.*}} : memref<5xf32>
  // CHECK: memref.copy %[[m]], %[[alloc]]
  // CHECK: %[[r:.*]] = bufferization.to_tensor %[[alloc]]
  // CHECK: return %[[r]]
  %dest = bufferization.alloc_tensor() : tensor<5xf32>
  %0 = bufferization.materialize_in_destination %arg0 in %dest
      : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

// CHECK-LABEL: func @materialize_in_destination_buffer(
//  CHECK-SAME:     %[[t:.*]]: tensor<5xf32>, %[[m:.*]]: memref<5xf32>)
//       CHECK:   %[[b:.*]] = bufferization.to_buffer %[[t]] : tensor<5xf32> to memref<5xf32, strided<[?], offset: ?>>
//       CHECK:   memref.copy %[[b]], %[[m]]
func.func @materialize_in_destination_buffer(%t: tensor<5xf32>, %m: memref<5xf32>) {
  bufferization.materialize_in_destination %t in restrict writable %m
      : (tensor<5xf32>, memref<5xf32>) -> ()
  return
}

// -----

func.func @materialize_in_func_bbarg(%t: tensor<?xf32>, %dest: tensor<?xf32>)
    -> tensor<?xf32> {
  // This op is not bufferizable because function block arguments are
  // read-only in regular One-Shot Bufferize. (Run One-Shot Module
  // Bufferization instead.)
  // expected-error @below{{not bufferizable under the given constraints: would write to read-only buffer}}
  %0 = bufferization.materialize_in_destination %t in %dest
      : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @materialize_in_dest_raw(%f: f32, %f2: f32, %idx: index) -> (tensor<5xf32>, f32) {
  %dest = bufferization.alloc_tensor() : tensor<5xf32>
  // Note: The location of the RaW conflict may not be accurate (such as in this
  // example). This is because the analysis operates on "alias sets" and not
  // single SSA values. The location may point to any SSA value in the alias set
  // that participates in the conflict.
  // expected-error @below{{not bufferizable under the given constraints: cannot avoid RaW conflict}}
  %dest_filled = linalg.fill ins(%f : f32) outs(%dest : tensor<5xf32>) -> tensor<5xf32>
  %src = bufferization.alloc_tensor() : tensor<5xf32>
  %src_filled = linalg.fill ins(%f2 : f32) outs(%src : tensor<5xf32>) -> tensor<5xf32>

  %0 = bufferization.materialize_in_destination %src_filled in %dest_filled
      : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  // Read from %dest_filled, which makes it impossible to bufferize the
  // materialize_in_destination op in-place.
  %r = tensor.extract %dest_filled[%idx] : tensor<5xf32>

  return %0, %r : tensor<5xf32>, f32
}

// -----

// CHECK:       func.func @custom_op(
// CHECK-SAME:    %[[ARG:.*]]: !test.test_tensor<[32, 64], f64>
// CHECK-SAME:  ) -> !test.test_tensor<[32, 128], f64> {
func.func @custom_op(%arg: !test.test_tensor<[32, 64], f64>)
    -> !test.test_tensor<[32, 128], f64> {
  // CHECK: %[[MEMREF:.*]] = bufferization.to_buffer %[[ARG]]
  // CHECK: %[[DUMMY:.*]] = "test.dummy_memref_op"(%[[MEMREF]])
  // CHECK-SAME: : (!test.test_memref<[32, 64], f64>)
  // CHECK-SAME: -> !test.test_memref<[32, 128], f64>
  // CHECK: %[[OUT:.*]] = bufferization.to_tensor %[[DUMMY]]
  %out = "test.dummy_tensor_op"(%arg) : (!test.test_tensor<[32, 64], f64>)
    -> !test.test_tensor<[32, 128], f64>

  // CHECK: return %[[OUT]]
  return %out : !test.test_tensor<[32, 128], f64>
}

// -----

// CHECK:       func.func @custom_origin_op()
// CHECK-SAME:  -> !test.test_tensor<[42], f64> {
func.func @custom_origin_op() -> !test.test_tensor<[42], f64> {
  // CHECK: %[[MEMREF:.*]] = "test.create_memref_op"() : ()
  // CHECK-SAME: -> !test.test_memref<[21], f64>
  // CHECK: %[[DUMMY:.*]] = "test.dummy_memref_op"(%[[MEMREF]])
  // CHECK-SAME: : (!test.test_memref<[21], f64>)
  // CHECK-SAME: -> !test.test_memref<[42], f64>
  %in = "test.create_tensor_op"() : () -> !test.test_tensor<[21], f64>
  %out = "test.dummy_tensor_op"(%in) : (!test.test_tensor<[21], f64>)
    -> !test.test_tensor<[42], f64>

  // CHECK: %[[OUT:.*]] = bufferization.to_tensor %[[DUMMY]]
  // CHECK: return %[[OUT]]
  return %out : !test.test_tensor<[42], f64>
}
