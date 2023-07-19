// RUN: mlir-opt -verify-diagnostics -convert-bufferization-to-memref -split-input-file %s | FileCheck %s

// CHECK-LABEL: @conversion_static
func.func @conversion_static(%arg0 : memref<2xf32>) -> memref<2xf32> {
  %0 = bufferization.clone %arg0 : memref<2xf32> to memref<2xf32>
  memref.dealloc %arg0 : memref<2xf32>
  return %0 : memref<2xf32>
}

// CHECK:      %[[ALLOC:.*]] = memref.alloc
// CHECK-NEXT: memref.copy %[[ARG:.*]], %[[ALLOC]]
// CHECK-NEXT: memref.dealloc %[[ARG]]
// CHECK-NEXT: return %[[ALLOC]]

// -----

// CHECK-LABEL: @conversion_dynamic
func.func @conversion_dynamic(%arg0 : memref<?xf32>) -> memref<?xf32> {
  %1 = bufferization.clone %arg0 : memref<?xf32> to memref<?xf32>
  memref.dealloc %arg0 : memref<?xf32>
  return %1 : memref<?xf32>
}

// CHECK:      %[[CONST:.*]] = arith.constant
// CHECK-NEXT: %[[DIM:.*]] = memref.dim %[[ARG:.*]], %[[CONST]]
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc(%[[DIM]])
// CHECK-NEXT: memref.copy %[[ARG]], %[[ALLOC]]
// CHECK-NEXT: memref.dealloc %[[ARG]]
// CHECK-NEXT: return %[[ALLOC]]

// -----

func.func @conversion_unknown(%arg0 : memref<*xf32>) -> memref<*xf32> {
// expected-error@+1 {{failed to legalize operation 'bufferization.clone' that was explicitly marked illegal}}
  %1 = bufferization.clone %arg0 : memref<*xf32> to memref<*xf32>
  memref.dealloc %arg0 : memref<*xf32>
  return %1 : memref<*xf32>
}

// -----

// CHECK-LABEL: func @conversion_with_layout_map(
//  CHECK-SAME:     %[[ARG:.*]]: memref<?xf32, strided<[?], offset: ?>>
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[DIM:.*]] = memref.dim %[[ARG]], %[[C0]]
//       CHECK:   %[[ALLOC:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
//       CHECK:   %[[CASTED:.*]] = memref.cast %[[ALLOC]] : memref<?xf32> to memref<?xf32, strided<[?], offset: ?>>
//       CHECK:   memref.copy
//       CHECK:   memref.dealloc
//       CHECK:   return %[[CASTED]]
func.func @conversion_with_layout_map(%arg0 : memref<?xf32, strided<[?], offset: ?>>) -> memref<?xf32, strided<[?], offset: ?>> {
  %1 = bufferization.clone %arg0 : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
  memref.dealloc %arg0 : memref<?xf32, strided<[?], offset: ?>>
  return %1 : memref<?xf32, strided<[?], offset: ?>>
}

// -----

// This bufferization.clone cannot be lowered because a buffer with this layout
// map cannot be allocated (or casted to).

func.func @conversion_with_invalid_layout_map(%arg0 : memref<?xf32, strided<[10], offset: ?>>)
    -> memref<?xf32, strided<[10], offset: ?>> {
// expected-error@+1 {{failed to legalize operation 'bufferization.clone' that was explicitly marked illegal}}
  %1 = bufferization.clone %arg0 : memref<?xf32, strided<[10], offset: ?>> to memref<?xf32, strided<[10], offset: ?>>
  memref.dealloc %arg0 : memref<?xf32, strided<[10], offset: ?>>
  return %1 : memref<?xf32, strided<[10], offset: ?>>
}
// -----

// CHECK-LABEL: func @conversion_dealloc_empty
func.func @conversion_dealloc_empty() {
  // CHECK-NEXT: return
  bufferization.dealloc
  return
}

// -----

// CHECK-NOT: func @deallocHelper
// CHECK-LABEL: func @conversion_dealloc_simple
// CHECK-SAME: [[ARG0:%.+]]: memref<2xf32>
// CHECK-SAME: [[ARG1:%.+]]: i1
func.func @conversion_dealloc_simple(%arg0: memref<2xf32>, %arg1: i1) -> i1 {
  %0 = bufferization.dealloc (%arg0 : memref<2xf32>) if (%arg1)
  return %0 : i1
}

//      CHECk: scf.if [[ARG1]] {
// CHECk-NEXT:   memref.dealloc [[ARG0]] : memref<2xf32>
// CHECk-NEXT: }
// CHECk-NEXT: [[FALSE:%.+]] = arith.constant false
// CHECk-NEXT: return [[FALSE]] : i1

// -----

func.func @conversion_dealloc_multiple_memrefs_and_retained(%arg0: memref<2xf32>, %arg1: memref<5xf32>, %arg2: memref<1xf32>, %arg3: i1, %arg4: i1) -> (i1, i1) {
  %0:2 = bufferization.dealloc (%arg0, %arg1 : memref<2xf32>, memref<5xf32>) if (%arg3, %arg4) retain (%arg2 : memref<1xf32>)
  return %0#0, %0#1 : i1, i1
}

// CHECK-LABEL: func @conversion_dealloc_multiple_memrefs_and_retained
// CHECK-SAME: [[ARG0:%.+]]: memref<2xf32>,
// CHECK-SAME: [[ARG1:%.+]]: memref<5xf32>,
// CHECK-SAME: [[ARG2:%.+]]: memref<1xf32>,
// CHECK-SAME: [[ARG3:%.+]]: i1,
// CHECK-SAME: [[ARG4:%.+]]: i1
//      CHECK: [[TO_DEALLOC_MR:%.+]] = memref.alloc() : memref<2xindex>
//      CHECK: [[TO_RETAIN_MR:%.+]] = memref.alloc() : memref<1xindex>
//  CHECK-DAG: [[V0:%.+]] = memref.extract_aligned_pointer_as_index [[ARG0]]
//  CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
//  CHECK-DAG: memref.store [[V0]], [[TO_DEALLOC_MR]][[[C0]]]
//  CHECK-DAG: [[V1:%.+]] = memref.extract_aligned_pointer_as_index [[ARG1]]
//  CHECK-DAG: [[C1:%.+]] = arith.constant 1 : index
//  CHECK-DAG: memref.store [[V1]], [[TO_DEALLOC_MR]][[[C1]]]
//  CHECK-DAG: [[V2:%.+]] = memref.extract_aligned_pointer_as_index [[ARG2]]
//  CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
//  CHECK-DAG: memref.store [[V2]], [[TO_RETAIN_MR]][[[C0]]]
//  CHECK-DAG: [[CAST_DEALLOC:%.+]] = memref.cast [[TO_DEALLOC_MR]] : memref<2xindex> to memref<?xindex>
//  CHECK-DAG: [[CAST_RETAIN:%.+]] = memref.cast [[TO_RETAIN_MR]] : memref<1xindex> to memref<?xindex>
//  CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
//      CHECK: [[RES0:%.+]]:2 = call @dealloc_helper([[CAST_DEALLOC]], [[CAST_RETAIN]], [[C0]])
//      CHECK: [[SHOULD_DEALLOC_0:%.+]] = arith.andi [[RES0]]#0, [[ARG3]]
//      CHECK: [[OWNERSHIP0:%.+]] = arith.andi [[RES0]]#1, [[ARG3]]
//      CHECK: scf.if [[SHOULD_DEALLOC_0]] {
//      CHECK:   memref.dealloc %arg0
//      CHECK: }
//      CHECK: [[C1:%.+]] = arith.constant 1 : index
//      CHECK: [[RES1:%.+]]:2 = call @dealloc_helper([[CAST_DEALLOC]], [[CAST_RETAIN]], [[C1]])
//      CHECK: [[SHOULD_DEALLOC_1:%.+]] = arith.andi [[RES1:%.+]]#0, [[ARG4]]
//      CHECK: [[OWNERSHIP1:%.+]] = arith.andi [[RES1:%.+]]#1, [[ARG4]]
//      CHECK: scf.if [[SHOULD_DEALLOC_1]]
//      CHECK:   memref.dealloc [[ARG1]]
//      CHECK: }
//      CHECK: memref.dealloc [[TO_DEALLOC_MR]]
//      CHECK: memref.dealloc [[TO_RETAIN_MR]]
//      CHECK: return [[OWNERSHIP0]], [[OWNERSHIP1]]

//      CHECK: func @dealloc_helper
// CHECK-SAME: [[ARG0:%.+]]: memref<?xindex>, [[ARG1:%.+]]: memref<?xindex>
// CHECK-SAME: [[ARG2:%.+]]: index
// CHECK-SAME:  -> (i1, i1)
//      CHECK:   [[TO_RETAIN_SIZE:%.+]] = memref.dim [[ARG1]], %c0
//      CHECK:   [[TO_DEALLOC:%.+]] = memref.load [[ARG0]][[[ARG2]]] : memref<?xindex>
// CHECK-NEXT:   [[NO_RETAIN_ALIAS:%.+]] = scf.for [[ITER:%.+]] = %c0 to [[TO_RETAIN_SIZE]] step %c1 iter_args([[ITER_ARG:%.+]] = %true) -> (i1) {
// CHECK-NEXT:     [[RETAIN_VAL:%.+]] = memref.load [[ARG1]][[[ITER]]] : memref<?xindex>
// CHECK-NEXT:     [[DOES_ALIAS:%.+]] = arith.cmpi ne, [[RETAIN_VAL]], [[TO_DEALLOC]] : index
// CHECK-NEXT:     [[AGG_DOES_ALIAS:%.+]] = arith.andi [[ITER_ARG]], [[DOES_ALIAS]] : i1
// CHECK-NEXT:     scf.yield [[AGG_DOES_ALIAS]] : i1
// CHECK-NEXT:   }
// CHECK-NEXT:   [[SHOULD_DEALLOC:%.+]] = scf.for [[ITER:%.+]] = %c0 to [[ARG2]] step %c1 iter_args([[ITER_ARG:%.+]] = [[NO_RETAIN_ALIAS]]) -> (i1) {
// CHECK-NEXT:     [[OTHER_DEALLOC_VAL:%.+]] = memref.load [[ARG0]][[[ITER]]] : memref<?xindex>
// CHECK-NEXT:     [[DOES_ALIAS:%.+]] = arith.cmpi ne, [[OTHER_DEALLOC_VAL]], [[TO_DEALLOC]] : index
// CHECK-NEXT:     [[AGG_DOES_ALIAS:%.+]] = arith.andi [[ITER_ARG]], [[DOES_ALIAS]] : i1
// CHECK-NEXT:     scf.yield [[AGG_DOES_ALIAS]] : i1
// CHECK-NEXT:   }
// CHECK-NEXT:   [[OWNERSHIP:%.+]] = arith.xori [[NO_RETAIN_ALIAS]], %true : i1
// CHECK-NEXT:   return [[SHOULD_DEALLOC]], [[OWNERSHIP]] : i1, i1
