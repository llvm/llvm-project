// Note: Default is function-boundary-type-conversion=infer-layout-map
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs" -canonicalize -drop-equivalent-buffer-results -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs test-analysis-only analysis-fuzzer-seed=23" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs test-analysis-only analysis-fuzzer-seed=59" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs test-analysis-only analysis-fuzzer-seed=91" -split-input-file -o /dev/null

// Test bufferization using memref types that have no layout map.
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map" -split-input-file | FileCheck %s --check-prefix=CHECK-NO-LAYOUT-MAP

// Test bufferization using memref types that have fully dynamic layout maps.
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs function-boundary-type-conversion=fully-dynamic-layout-map" -split-input-file | FileCheck %s --check-prefix=CHECK-FULLY-DYNAMIC-LAYOUT-MAP


// Bufferization of bodiless function with no tensor return value.

// CHECK-LABEL: func private @private_func(memref<?xf32, strided<[?], offset: ?>>
// CHECK-NO-LAYOUT-MAP-LABEL: func private @private_func(memref<?xf32>)
func.func private @private_func(tensor<?xf32>) -> ()

// CHECK-LABEL: func private @private_func_2d(memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK-NO-LAYOUT-MAP-LABEL: func private @private_func_2d(memref<?x?xf32>)
func.func private @private_func_2d(tensor<?x?xf32>) -> ()

// CHECK-LABEL: func @empty_func() {
// CHECK-NO-LAYOUT-MAP-LABEL: func @empty_func() {
// CHECK-FULLY-DYNAMIC-LAYOUT-MAP-LABEL: func @empty_func() {
func.func @empty_func() -> () {
  return
}

// -----

// A bodiless function that returns something that is not a tensor.

// CHECK: func private @external_func_with_return_val(memref<4xi32, strided{{.*}}>) -> f32
// CHECK-FULLY-DYNAMIC-LAYOUT-MAP-LABEL: func private @external_func_with_return_val(memref<4xi32,
// CHECK-FULLY-DYNAMIC-LAYOUT-MAP-SAME: strided<[?], offset: ?>>
// CHECK-NO-LAYOUT-MAP-LABEL: func private @external_func_with_return_val(memref<4xi32>)
func.func private @external_func_with_return_val(tensor<4xi32>) -> f32

// -----

// A function that returns a non-equivalent tensor with layout map.

// CHECK-LABEL: func @return_extract_slice(%{{.*}}) -> memref<2x?xf32, strided<[10, 1], offset: ?>>
//       CHECK:   %[[alloc:.*]] = memref.alloc() {{.*}} : memref<20x10xf32>
//       CHECK:   %[[subview:.*]] = memref.subview {{.*}} : memref<20x10xf32> to memref<2x?xf32, strided<[10, 1], offset: ?>>
//       CHECK:   return %[[subview]]

// CHECK-NO-LAYOUT-MAP-LABEL: func @return_extract_slice(%{{.*}}) -> memref<2x?xf32>
//       CHECK-NO-LAYOUT-MAP:   %[[alloc:.*]] = memref.alloc() {{.*}} : memref<20x10xf32>
//       CHECK-NO-LAYOUT-MAP:   %[[subview:.*]] = memref.subview {{.*}} : memref<20x10xf32> to memref<2x?xf32, strided<[10, 1], offset: ?>>
//       CHECK-NO-LAYOUT-MAP:   %[[alloc_no_layout:.*]] = memref.alloc(%{{.*}}) : memref<2x?xf32>
//       CHECK-NO-LAYOUT-MAP:   memref.copy %[[subview]], %[[alloc_no_layout]]
// TODO: %alloc should be deallocated here, but we currently do not dealloc
// buffers that are inserted due to to_tensor/to_memref canonicalization (when
// the buffer types have different layout maps).
//       CHECK-NO-LAYOUT-MAP:   return %[[alloc_no_layout]]

// CHECK-FULLY-DYNAMIC-LAYOUT-MAP-LABEL: func @return_extract_slice(%{{.*}}) -> memref<2x?xf32,
//  CHECK-FULLY-DYNAMIC-LAYOUT-MAP-SAME: strided<[?, ?], offset: ?>> {
func.func @return_extract_slice(%idx: index, %sz: index) -> (tensor<2x?xf32>)
{
  %t = bufferization.alloc_tensor() : tensor<20x10xf32>
  %0 = tensor.extract_slice %t[%idx, %idx][2, %sz][1, 1]
      : tensor<20x10xf32> to tensor<2x?xf32>
  return %0 : tensor<2x?xf32>
}

// -----

// CHECK-LABEL: func private @private_func
// CHECK-NO-LAYOUT-MAP-LABEL: func private @private_func(memref<?xf32>) -> f32
func.func private @private_func(tensor<?xf32>) -> (f32)

// private_func may modify the buffer arg, but that's OK because %t is writable.
// No alloc/copy should be inserted.

// CHECK-LABEL: func @main(
//  CHECK-SAME:     %[[t:.*]]: memref<?xf32
//   CHECK-NOT: alloc
//   CHECK-NOT: copy
//       CHECK: call @private_func(%[[t]])
func.func @main(%t: tensor<?xf32> {bufferization.writable = true}) -> (f32) {
  %0 = call @private_func(%t) : (tensor<?xf32>) -> (f32)
  return %0 : f32
}

// -----

// CHECK-LABEL: func private @private_func
func.func private @private_func(tensor<?xf32>) -> (f32)

// private_func may modify the buffer arg, %t is not writable. A copy is needed.

// CHECK-LABEL: func @main(
//  CHECK-SAME:     %[[t:.*]]: memref<?xf32
//       CHECK: %[[alloc:.*]] = memref.alloc
//   CHECK-DAG: memref.copy %[[t]], %[[alloc]]
//   CHECK-DAG: %[[casted:.*]] = memref.cast %[[alloc]]
//       CHECK: call @private_func(%[[casted]])
//       CHECK: memref.dealloc %[[alloc]]
func.func @main(%t: tensor<?xf32> {bufferization.writable = false}) -> (f32) {
  %0 = call @private_func(%t) : (tensor<?xf32>) -> (f32)
  return %0 : f32
}

// -----

// Test bufferization of a function without tensor args.

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

// Bufferization of a function that is reading and writing. %t0 is writable, so
// no copy should be inserted.

// CHECK-LABEL: func @inner_func(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
func.func @inner_func(%t: tensor<?xf32>) -> (tensor<?xf32>, f32) {
  // CHECK-NOT: copy
  %f = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: memref.store %{{.*}}, %[[arg0]]
  %0 = tensor.insert %f into %t[%c0] : tensor<?xf32>
  // CHECK: %[[load:.*]] = memref.load %[[arg0]]
  %1 = tensor.extract %0[%c1] : tensor<?xf32>
  // CHECK: return %[[load]] : f32
  return %0, %1 : tensor<?xf32>, f32
}

// CHECK-LABEL: func @call_func_with_non_tensor_return(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
func.func @call_func_with_non_tensor_return(
    %t0: tensor<?xf32> {bufferization.writable = true}) -> (f32, tensor<?xf32>) {
  // CHECK-NOT: alloc
  // CHECK-NOT: copy
  // CHECK: %[[call:.*]] = call @inner_func(%[[arg0]])
  %0, %1 = call @inner_func(%t0) : (tensor<?xf32>) -> (tensor<?xf32>, f32)
  // CHECK: return %[[call]] : f32
  return %1, %0 : f32, tensor<?xf32>
}

// -----

// Bufferization of a function that is reading and writing. %t0 is not writable,
// so a copy is needed.

// CHECK-LABEL: func @inner_func(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
func.func @inner_func(%t: tensor<?xf32>) -> (tensor<?xf32>, f32) {
  // CHECK-NOT: copy
  %f = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: memref.store %{{.*}}, %[[arg0]]
  %0 = tensor.insert %f into %t[%c0] : tensor<?xf32>
  // CHECK: %[[load:.*]] = memref.load %[[arg0]]
  %1 = tensor.extract %0[%c1] : tensor<?xf32>
  // CHECK: return %[[load]] : f32
  return %0, %1 : tensor<?xf32>, f32
}

// CHECK-LABEL: func @call_func_with_non_tensor_return(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
func.func @call_func_with_non_tensor_return(
    %t0: tensor<?xf32> {bufferization.writable = false}) -> (f32, tensor<?xf32>) {
  // CHECK: %[[alloc:.*]] = memref.alloc
  // CHECK-DAG: memref.copy %[[arg0]], %[[alloc]]
  // CHECK-DAG: %[[casted:.*]] = memref.cast %[[alloc]]
  // CHECK: %[[call:.*]] = call @inner_func(%[[casted]])
  %0, %1 = call @inner_func(%t0) : (tensor<?xf32>) -> (tensor<?xf32>, f32)

  // Note: The tensor return value cannot fold away because the CallOp
  // bufferized out-of-place.
  // CHECK: return %[[call]], %[[casted]] : f32, memref<?xf32
  return %1, %0 : f32, tensor<?xf32>
}

// -----

// A chain of function calls. The last function f0 is potentially writing to the
// buffer. This becomes a problem when bufferizing main and a copy must be
// inserted then. (No copies in the other functions.)

// CHECK-LABEL: func private @f0(
func.func private @f0(tensor<?xf32>) -> (f32)

// CHECK-LABEL: func @f1(
//  CHECK-SAME:     %[[t1:.*]]: memref<?xf32
//       CHECK:   %[[r1:.*]] = call @f0(%[[t1]])
//       CHECK:   return %[[r1]]
func.func @f1(%t: tensor<?xf32>) -> (f32) {
  %0 = call @f0(%t) : (tensor<?xf32>) -> (f32)
  return %0 : f32
}

// CHECK-LABEL: func @f2(
//  CHECK-SAME:     %[[t2:.*]]: memref<?xf32
//       CHECK:   %[[r2:.*]] = call @f1(%[[t2]])
//       CHECK:   return %[[r2]]
func.func @f2(%t: tensor<?xf32>) -> (f32) {
  %0 = call @f1(%t) : (tensor<?xf32>) -> (f32)
  return %0 : f32
}

// CHECK-LABEL: func @main(
//  CHECK-SAME:     %[[t3:.*]]: memref<?xf32
//       CHECK: %[[alloc:.*]] = memref.alloc
//   CHECK-DAG: memref.copy %[[t3]], %[[alloc]]
//   CHECK-DAG: %[[casted:.*]] = memref.cast %[[alloc]]
//       CHECK: call @f2(%[[casted]])
//       CHECK: memref.dealloc %[[alloc]]
func.func @main(%t: tensor<?xf32> {bufferization.writable = false}) -> (f32) {
  %0 = call @f2(%t) : (tensor<?xf32>) -> (f32)
  return %0 : f32
}

// -----

// This function does not read, just write. We need an alloc, but no copy.

// CHECK-LABEL: func @does_not_read(
//   CHECK-NOT:   alloc
//   CHECK-NOT:   copy
func.func @does_not_read(%t: tensor<?xf32>) -> tensor<?xf32> {
  %f0 = arith.constant 0.0 : f32
  %r = linalg.fill ins(%f0 : f32) outs(%t : tensor<?xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// CHECK-LABEL: func @main(
//  CHECK-SAME:     %[[t:.*]]: memref<?xf32
//       CHECK:   %[[alloc:.*]] = memref.alloc
//   CHECK-NOT:   copy
//       CHECK: %[[casted:.*]] = memref.cast %[[alloc]]
//   CHECK-NOT:   copy
//       CHECK:   call @does_not_read(%[[casted]])
//       CHECK:   %[[r:.*]] = memref.load %[[casted]]
//       CHECK:   memref.dealloc %[[alloc]]
func.func @main(%t: tensor<?xf32> {bufferization.writable = false}) -> f32 {
  %0 = call @does_not_read(%t) : (tensor<?xf32>) -> (tensor<?xf32>)
  %idx = arith.constant 4 : index
  %r = tensor.extract %0[%idx] : tensor<?xf32>
  return %r : f32
}

// -----

// Alloc and copy must be inserted because the arith.constant is read-only.

//      CHECK: memref.global "private" constant @__constant_4xi32 : memref<4xi32> = dense<[1, 2, 3, 4]>
//      CHECK: func private @some_external_func(memref<4xi32, strided<[?], offset: ?>>)
func.func private @some_external_func(tensor<4xi32>)

//      CHECK: func @main()
func.func @main() {
//  CHECK-DAG:   %[[A:.*]] = memref.get_global @__constant_4xi32 : memref<4xi32>
  %A = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

//  CHECK-DAG:   %[[alloc:.*]] = memref.alloc
//  CHECK-DAG:   %[[B:.*]] = memref.cast %[[alloc]] : memref<4xi32> to memref<4xi32, strided<[?], offset: ?>>
//  CHECK-DAG:   memref.copy %[[A]], %[[alloc]]
//      CHECK:   call @some_external_func(%[[B]]) : (memref<4xi32, strided<[?], offset: ?>>) -> ()
  call @some_external_func(%A) : (tensor<4xi32>) -> ()

//      CHECK: memref.dealloc %[[alloc]]
  return
}

// -----

// Alloc and copy must be inserted because the arith.constant is read-only. The
// function call is inside of an scf.execute_region.

//      CHECK: memref.global "private" constant @__constant_4xi32 : memref<4xi32> = dense<[1, 2, 3, 4]>
//      CHECK: func private @some_external_func_within_scf_execute(memref<4xi32, strided<[?], offset: ?>>)
func.func private @some_external_func_within_scf_execute(tensor<4xi32>)

//      CHECK: func @main()
func.func @main() {
//  CHECK-DAG:   %[[A:.*]] = memref.get_global @__constant_4xi32 : memref<4xi32>
  %A = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

// Note: The scf.execute_region canonicalizes away.

//  CHECK-DAG:   %[[alloc:.*]] = memref.alloc
//  CHECK-DAG:   %[[B:.*]] = memref.cast %[[alloc]] : memref<4xi32> to memref<4xi32, strided<[?], offset: ?>>
//  CHECK-DAG:   memref.copy %[[A]], %[[alloc]]
//      CHECK:   call @some_external_func_within_scf_execute(%[[B]]) : (memref<4xi32, strided<[?], offset: ?>>) -> ()
  scf.execute_region {
    func.call @some_external_func_within_scf_execute(%A) : (tensor<4xi32>) -> ()
    scf.yield
  }

//      CHECK:   memref.dealloc %[[alloc]]
  return
}

// -----

// A write inside an scf.execute_region. An equivalent tensor is yielded.

// CHECK-LABEL: func @execute_region_test(
//  CHECK-SAME:     %[[m1:.*]]: memref<?xf32
func.func @execute_region_test(%t1 : tensor<?xf32>)
    -> (f32, tensor<?xf32>, f32)
{
  %f1 = arith.constant 0.0 : f32
  %f2 = arith.constant 1.0 : f32
  %idx = arith.constant 7 : index

  // scf.execute_region is canonicalized away after bufferization. So just the
  // memref.store is left over.

  // CHECK-NOT: alloc
  // CHECK-NOT: copy
  // CHECK: memref.store %{{.*}}, %[[m1]][%{{.*}}]
  %0, %1, %2 = scf.execute_region -> (f32, tensor<?xf32>, f32) {
    %t2 = tensor.insert %f2 into %t1[%idx] : tensor<?xf32>
    scf.yield %f1, %t2, %f2 : f32, tensor<?xf32>, f32
  }

  // CHECK: return %{{.*}}, %{{.*}} : f32, f32
  return %0, %1, %2 : f32, tensor<?xf32>, f32
}

// -----

//      CHECK:  func private @some_external_func(memref<?xf32, strided<[?], offset: ?>>)
func.func private @some_external_func(tensor<?xf32>)

//      CHECK:  func @scf_for_with_tensor_insert_slice(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>
// CHECK-SAME:    %[[C:[a-zA-Z0-9]*]]: memref<4xf32, strided<[?], offset: ?>>
func.func @scf_for_with_tensor_insert_slice(
    %A : tensor<?xf32>, %B : tensor<?xf32>, %C : tensor<4xf32>,
    %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // CHECK-NEXT: scf.for
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    // CHECK-NEXT:   %[[SVA:.*]] = memref.subview %[[A]]
    // CHECK-NEXT:   memref.copy %[[C]], %[[SVA]] : memref<4xf32, strided<[?], offset: ?>> to memref<4xf32, strided<[?], offset: ?>>
    %ttA = tensor.insert_slice %C into %tA[%i][4][1] : tensor<4xf32> into tensor<?xf32>

    // CHECK-NEXT:   %[[SVB:.*]] = memref.subview %[[B]]
    // CHECK-NEXT:   memref.copy %[[C]], %[[SVB]] : memref<4xf32, strided<[?], offset: ?>> to memref<4xf32, strided<[?], offset: ?>>
    %ttB = tensor.insert_slice %C into %tB[%i][4][1] : tensor<4xf32> into tensor<?xf32>

    // scf.yield is empty and is elided
    //  CHECK-NOT:   scf.yield
    scf.yield %ttA, %ttB : tensor<?xf32>, tensor<?xf32>
  }

  // Swaparoo requires bufferizing the whole function to figure out who's who.
  return %r0#1, %r0#0: tensor<?xf32>, tensor<?xf32>
}

//      CHECK:  func @bar(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>
// CHECK-SAME:    %[[C:[a-zA-Z0-9]*]]: memref<4xf32, strided<[?], offset: ?>>
func.func @bar(
    %A : tensor<?xf32> {bufferization.writable = true},
    %B : tensor<?xf32> {bufferization.writable = true},
    %C : tensor<4xf32> {bufferization.writable = true},
    %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
//  CHECK-DAG:   call @scf_for_with_tensor_insert_slice(%[[A]], %[[B]], %[[C]]
  %r0:2 = call @scf_for_with_tensor_insert_slice(%A, %B, %C, %lb, %ub, %step) :
      (tensor<?xf32>, tensor<?xf32>, tensor<4xf32>, index, index, index)
        -> (tensor<?xf32>, tensor<?xf32>)

  // %r0#0 requires a copy because we have no idea what the function is doing.
//  CHECK-DAG:   %[[alloc:.*]] = memref.alloc
//  CHECK-DAG:   %[[casted:.*]] = memref.cast %[[alloc]]
//  CHECK-DAG:   memref.copy %[[B]], %[[alloc]]
// CHECK-NEXT:   call @some_external_func(%[[casted]]) : (memref<?xf32, strided<[?], offset: ?>>) -> ()
  call @some_external_func(%r0#0) : (tensor<?xf32>) -> ()

//      CHECK:   return
  return %r0#0, %r0#1: tensor<?xf32>, tensor<?xf32>
}

// -----

//      CHECK:  func @init_and_dot(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<64xf32, strided<[?], offset: ?>>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]*]]: memref<64xf32, strided<[?], offset: ?>>
// CHECK-SAME:    %[[C:[a-zA-Z0-9]*]]: memref<f32, strided<[], offset: ?>>
func.func @init_and_dot(%a: tensor<64xf32>, %b: tensor<64xf32>, %c: tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT:   %[[C0:.*]] = arith.constant 0{{.*}} : f32
  %v0 = arith.constant 0.0 : f32

  // CHECK-NEXT:   linalg.fill ins(%[[C0]] : f32) outs(%[[C]] : memref<f32, strided<[], offset: ?>>)
  %d = linalg.fill ins(%v0 : f32) outs(%c : tensor<f32>) -> tensor<f32>

  // CHECK-NEXT:   linalg.dot ins(%[[A]], %[[B]] : memref<64xf32, strided<[?], offset: ?>>, memref<64xf32, strided<[?], offset: ?>>) outs(%[[C]] : memref<f32, strided<[], offset: ?>>)
  %e = linalg.dot ins(%a, %b : tensor<64xf32>,tensor<64xf32>)
    outs(%d: tensor<f32>) -> tensor<f32>

  // CHECK-NEXT:   return
  return %e : tensor<f32>
}

//      CHECK:  func @main()
func.func @main() {
  //  CHECK-DAG:   %[[C0:.*]] = arith.constant 0{{.*}} : f32
  //  CHECK-DAG:   %[[C1:.*]] = arith.constant 1{{.*}} : f32
  //  CHECK-DAG:   %[[C2:.*]] = arith.constant 2{{.*}} : f32
  %v0 = arith.constant 0.0 : f32
  %v1 = arith.constant 1.0 : f32
  %v2 = arith.constant 2.0 : f32

  // CHECK-NEXT:   %[[A:.*]] = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  // CHECK-NEXT:   %[[B:.*]] = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  // CHECK-NEXT:   %[[C:.*]] = memref.alloc() {alignment = 64 : i64} : memref<f32>
  //  CHECK-DAG:   %[[cA:.*]] = memref.cast %[[A]] : memref<64xf32> to memref<64xf32, strided<[?], offset: ?>>
  //  CHECK-DAG:   %[[cB:.*]] = memref.cast %[[B]] : memref<64xf32> to memref<64xf32, strided<[?], offset: ?>>
  //  CHECK-DAG:   %[[cC:.*]] = memref.cast %[[C]] : memref<f32> to memref<f32, strided<[], offset: ?>>
  %A = bufferization.alloc_tensor() : tensor<64xf32>
  %B = bufferization.alloc_tensor() : tensor<64xf32>
  %C = bufferization.alloc_tensor() : tensor<f32>

  //  CHECK-DAG:   linalg.fill ins(%[[C1]] : f32) outs(%[[A]] : memref<64xf32>)
  //  CHECK-DAG:   linalg.fill ins(%[[C2]] : f32) outs(%[[B]] : memref<64xf32>)
  //  CHECK-DAG:   linalg.fill ins(%[[C0]] : f32) outs(%[[C]] : memref<f32>)
  %AA = linalg.fill ins(%v1 : f32) outs(%A : tensor<64xf32>) -> tensor<64xf32>
  %BB = linalg.fill ins(%v2 : f32) outs(%B : tensor<64xf32>) -> tensor<64xf32>
  %CC = linalg.fill ins(%v0 : f32) outs(%C : tensor<f32>) -> tensor<f32>

  // CHECK-NEXT:   call @init_and_dot(%[[cA]], %[[cB]], %[[cC]])
  %res = call @init_and_dot(%AA, %BB, %CC) :
    (tensor<64xf32>, tensor<64xf32>, tensor<f32>) -> tensor<f32>

  // CHECK-NEXT:   %[[dC:.*]] = memref.cast %[[cC]] : memref<f32, {{.*}}> to memref<*xf32>
  %res2 = tensor.cast %res: tensor<f32> to tensor<*xf32>

  // CHECK-NEXT:   call @printMemrefF32(%[[dC]]) : (memref<*xf32>) -> ()
  call @printMemrefF32(%res2) : (tensor<*xf32>) -> ()

  // CHECK-DAG:   memref.dealloc %[[A]] : memref<64xf32>
  // CHECK-DAG:   memref.dealloc %[[B]] : memref<64xf32>
  // CHECK-DAG:   memref.dealloc %[[C]] : memref<f32>
  // CHECK-NEXT:   return
  return
}

//     CHECK:   func private @printMemrefF32(memref<*xf32>)
func.func private @printMemrefF32(tensor<*xf32>)

// -----

// CHECK: func private @external_func(memref<?xf32, strided<[?], offset: ?>>)
func.func private @external_func(tensor<?xf32>)

//      CHECK: func @callee(
// CHECK-SAME:   %[[A:[0-9a-zA-Z]*]]: memref<?xf32>
// CHECK-SAME:   %[[B:[0-9a-zA-Z]*]]: memref<?xf32, strided<[?], offset: ?>>
// CHECK-SAME:   %[[C:[0-9a-zA-Z]*]]: memref<?xf32, strided<[?], offset: ?>>
func.func @callee(
    %A : tensor<?xf32> {bufferization.buffer_layout = affine_map<(i)[s0, s1] -> (i)>},
    %B : tensor<?xf32>,
    %C : tensor<?xf32>) {
// CHECK-NEXT: %[[CASTED:.*]] = memref.cast %[[A]] : memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK-NEXT: call @external_func(%[[CASTED]]) : (memref<?xf32, strided<[?], offset: ?>>) -> ()
  call @external_func(%A) : (tensor<?xf32>) -> ()

// CHECK-NEXT: call @external_func(%[[B]]) : (memref<?xf32, strided<[?], offset: ?>>) -> ()
  call @external_func(%B) : (tensor<?xf32>) -> ()

// CHECK-NEXT: call @external_func(%[[C]]) : (memref<?xf32, strided<[?], offset: ?>>) -> ()
  call @external_func(%C) : (tensor<?xf32>) -> ()

  return
}

//      CHECK: func @entry(
// CHECK-SAME:   %[[A:[0-9a-zA-Z]*]]: memref<?xf32>
// CHECK-SAME:   %[[B:[0-9a-zA-Z]*]]: memref<?xf32>
// CHECK-SAME:   %[[C:[0-9a-zA-Z]*]]: memref<?xf32, strided<[?], offset: ?>>
func.func @entry(%A : tensor<?xf32> {bufferization.buffer_layout = affine_map<(i)[s0, s1] -> (i)>, bufferization.writable = false},
                 %B : tensor<?xf32> {bufferization.buffer_layout = affine_map<(i)[s0, s1] -> (i)>, bufferization.writable = false},
                 %C : tensor<?xf32> {bufferization.writable = false}) {
// Note: `callee` does not write to its bbArg directly, but `external_func`
// does. Inside `callee`, the writes via `external_func` do not cause a
// conflict. However, inside `entry`, the writes do cause a conflict because
// %A, %B and %C are not inplaceable. This test case shows that this kind of
// conflict detection has a "transitive" nature.
//  CHECK-DAG: %[[ALLOC_A:.*]] = memref.alloc
//  CHECK-DAG: %[[CASTED_A:.*]] = memref.cast %[[ALLOC_A]]
//  CHECK-DAG: %[[ALLOC_B:.*]] = memref.alloc
//  CHECK-DAG: %[[CASTED_B:.*]] = memref.cast %[[ALLOC_B]]
//  CHECK-DAG: %[[ALLOC_C:.*]] = memref.alloc
//  CHECK-DAG: %[[CASTED_C:.*]] = memref.cast %[[ALLOC_C]]
//  CHECK-DAG: memref.copy %[[A]], %[[ALLOC_A]]
//  CHECK-DAG: memref.copy %[[B]], %[[ALLOC_B]]
//  CHECK-DAG: memref.copy %[[C]], %[[ALLOC_C]]
// CHECK-NEXT: call @callee(%[[CASTED_A]], %[[CASTED_B]], %[[CASTED_C]])
  call @callee(%A, %B, %C) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> ()
  return
}

// -----

// No alloc or copy inside of the loop.

// CHECK-LABEL: func @inner_func(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
func.func @inner_func(%t: tensor<?xf32>) -> tensor<?xf32> {
  %f = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK: memref.store %{{.*}}, %[[arg0]]
  %0 = tensor.insert %f into %t[%c0] : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @equivalent_func_arg(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
func.func @equivalent_func_arg(%t0: tensor<?xf32> {bufferization.writable = true},
                               %c0: index, %c10: index, %c1: index) -> tensor<?xf32> {
  // CHECK-NOT: alloc
  // CHECK-NOT: copy
  // CHECK: scf.for {{.*}} iter_args(%[[t1:.*]] = %[[arg0]])
  %1 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%t1 = %t0) -> (tensor<?xf32>) {
    // CHECK: call @inner_func(%[[t1]])
    %3 = func.call @inner_func(%t1) : (tensor<?xf32>) -> tensor<?xf32>
    // CHECK: scf.yield %[[t1]]
    scf.yield %3 : tensor<?xf32>
  }
  return %1: tensor<?xf32>
}

// -----

// inner_func_2 modifies the bbArg, but the loop yields the original value. A
// buffer copy must be inserted inside the loop.

// CHECK-LABEL: func @inner_func_2(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
func.func @inner_func_2(%t: tensor<?xf32>) -> tensor<?xf32> {
  %f = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK: memref.store %{{.*}}, %[[arg0]]
  %0 = tensor.insert %f into %t[%c0] : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @equivalent_func_arg_2(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32
func.func @equivalent_func_arg_2(%t0: tensor<?xf32> {bufferization.writable = true},
                                 %c0: index, %c10: index, %c1: index) -> tensor<?xf32> {
  // CHECK: scf.for {{.*}} {
  %1 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%t1 = %t0) -> (tensor<?xf32>) {
    // CHECK: %[[alloc:.*]] = memref.alloc
    // CHECK-DAG: %[[casted:.*]] = memref.cast %[[alloc]]
    // CHECK-DAG: memref.copy %[[arg0]], %[[alloc]]
    // CHECK: call @inner_func_2(%[[casted]])
    // CHECK: memref.dealloc %[[alloc]]
    // CHECK-NOT: scf.yield
    %3 = func.call @inner_func_2(%t1) : (tensor<?xf32>) -> tensor<?xf32>
    scf.yield %t1 : tensor<?xf32>
  }
  return %1: tensor<?xf32>
}

// -----

// Bufferize without fully dynamic layout maps.

// CHECK-LABEL: func @transfer_read(%{{.*}}: memref<?xf32, strided{{.*}}>) -> vector<4xf32> {
// CHECK-NO-LAYOUT-MAP-LABEL: func @transfer_read(%{{.*}}: memref<?xf32>) -> vector<4xf32>
func.func @transfer_read(
    %A : tensor<?xf32> {bufferization.writable = false})
  -> (vector<4xf32>)
{
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32

//       CHECK: %[[RES:.*]] = vector.transfer_read {{.*}} : memref<?xf32, strided{{.*}}>, vector<4xf32>
  %0 = vector.transfer_read %A[%c0], %f0 : tensor<?xf32>, vector<4xf32>

//       CHECK: return %[[RES]] : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: func @main(
func.func @main() {
  // CHECK: %[[const:.*]] = memref.get_global
  %t = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  // CHECK: %[[alloc:.*]] = memref.alloc
  // CHECK: memref.copy %[[const]], %[[alloc]]
  // CHECK: %[[casted:.*]] = memref.cast %[[alloc]] : memref<3xf32> to memref<*xf32>
  %unranked = tensor.cast %t : tensor<3xf32> to tensor<*xf32>
  // CHECK: call @maybe_writing_func(%[[casted]])
  func.call @maybe_writing_func(%unranked) : (tensor<*xf32>) -> ()
  return
}

// This function may write to buffer(%ptr).
func.func private @maybe_writing_func(%ptr : tensor<*xf32>)

// -----

// Test if other callables are left intact and don't cause trouble.

llvm.func @llvm_func()

func.func @call_llvm_func() {
  llvm.call @llvm_func() : () -> ()
  return
}

// -----

// CHECK-LABEL: func @to_memref_op_unsupported(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xf32,
func.func @to_memref_op_unsupported(
    %t1: tensor<?xf32> {bufferization.writable = true}, %idx1: index,
    %idx2: index, %idx3: index, %v1: vector<5xf32>) -> (vector<5xf32>) {

  // Insert a copy because we cannot analyze what happens with the result of a
  // to_memref op.
  // CHECK: %[[alloc:.*]] = memref.alloc
  // CHECK: memref.copy %[[arg0]], %[[alloc]]
  %0 = bufferization.to_memref %t1 : memref<?xf32>
  // CHECK: "test.foo"(%[[alloc]])
  "test.foo"(%0) : (memref<?xf32>) -> ()

  // CHECK: vector.transfer_read %[[arg0]]
  %cst = arith.constant 0.0 : f32
  %r1 = vector.transfer_read %t1[%idx3], %cst : tensor<?xf32>, vector<5xf32>

  return %r1 : vector<5xf32>
}
