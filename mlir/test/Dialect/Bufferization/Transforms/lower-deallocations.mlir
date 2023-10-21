// RUN: mlir-opt -verify-diagnostics -bufferization-lower-deallocations -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @conversion_dealloc_empty
func.func @conversion_dealloc_empty() {
  // CHECK-NOT: bufferization.dealloc
  bufferization.dealloc
  return
}

// -----

func.func @conversion_dealloc_empty_but_retains(%arg0: memref<2xi32>, %arg1: memref<2xi32>) -> (i1, i1) {
  %0:2 = bufferization.dealloc retain (%arg0, %arg1 : memref<2xi32>, memref<2xi32>)
  return %0#0, %0#1 : i1, i1
}

// CHECK-LABEL: func @conversion_dealloc_empty
//  CHECK-NEXT: [[FALSE:%.+]] = arith.constant false
//  CHECK-NEXT: return [[FALSE]], [[FALSE]] :

// -----

// CHECK-NOT: func @deallocHelper
// CHECK-LABEL: func @conversion_dealloc_simple
// CHECK-SAME: [[ARG0:%.+]]: memref<2xf32>
// CHECK-SAME: [[ARG1:%.+]]: i1
func.func @conversion_dealloc_simple(%arg0: memref<2xf32>, %arg1: i1) {
  bufferization.dealloc (%arg0 : memref<2xf32>) if (%arg1)
  return
}

//      CHECk: scf.if [[ARG1]] {
// CHECk-NEXT:   memref.dealloc [[ARG0]] : memref<2xf32>
// CHECk-NEXT: }
// CHECk-NEXT: return

// -----

func.func @conversion_dealloc_one_memref_and_multiple_retained(%arg0: memref<2xf32>, %arg1: memref<1xf32>, %arg2: i1, %arg3: memref<2xf32>) -> (i1, i1) {
  %0:2 = bufferization.dealloc (%arg0 : memref<2xf32>) if (%arg2) retain (%arg1, %arg3 : memref<1xf32>, memref<2xf32>)
  return %0#0, %0#1 : i1, i1
}

// CHECK-LABEL: func @conversion_dealloc_one_memref_and_multiple_retained
//  CHECK-SAME: ([[ARG0:%.+]]: memref<2xf32>, [[ARG1:%.+]]: memref<1xf32>, [[ARG2:%.+]]: i1, [[ARG3:%.+]]: memref<2xf32>)
//   CHECK-DAG: [[M0:%.+]] = memref.extract_aligned_pointer_as_index [[ARG0]]
//   CHECK-DAG: [[R0:%.+]] = memref.extract_aligned_pointer_as_index [[ARG1]]
//   CHECK-DAG: [[R1:%.+]] = memref.extract_aligned_pointer_as_index [[ARG3]]
//   CHECK-DAG: [[DOES_NOT_ALIAS_R0:%.+]] = arith.cmpi ne, [[M0]], [[R0]] : index
//   CHECK-DAG: [[DOES_NOT_ALIAS_R1:%.+]] = arith.cmpi ne, [[M0]], [[R1]] : index
//       CHECK: [[NOT_RETAINED:%.+]] = arith.andi [[DOES_NOT_ALIAS_R0]], [[DOES_NOT_ALIAS_R1]]
//       CHECK: [[SHOULD_DEALLOC:%.+]] = arith.andi [[NOT_RETAINED]], [[ARG2]]
//       CHECK: scf.if [[SHOULD_DEALLOC]]
//       CHECK:   memref.dealloc [[ARG0]]
//       CHECK: }
//   CHECK-DAG: [[ALIASES_R0:%.+]] = arith.xori [[DOES_NOT_ALIAS_R0]], %true
//   CHECK-DAG: [[ALIASES_R1:%.+]] = arith.xori [[DOES_NOT_ALIAS_R1]], %true
//   CHECK-DAG: [[RES0:%.+]] = arith.andi [[ALIASES_R0]], [[ARG2]]
//   CHECK-DAG: [[RES1:%.+]] = arith.andi [[ALIASES_R1]], [[ARG2]]
//       CHECK: return [[RES0]], [[RES1]]

// CHECK-NOT: func @dealloc_helper

// -----

func.func @conversion_dealloc_multiple_memrefs_and_retained(%arg0: memref<2xf32>, %arg1: memref<5xf32>, %arg2: memref<1xf32>, %arg3: i1, %arg4: i1, %arg5: memref<2xf32>) -> (i1, i1) {
  %0:2 = bufferization.dealloc (%arg0, %arg1 : memref<2xf32>, memref<5xf32>) if (%arg3, %arg4) retain (%arg2, %arg5 : memref<1xf32>, memref<2xf32>)
  return %0#0, %0#1 : i1, i1
}

// CHECK-LABEL: func @conversion_dealloc_multiple_memrefs_and_retained
// CHECK-SAME: ([[ARG0:%.+]]: memref<2xf32>, [[ARG1:%.+]]: memref<5xf32>,
// CHECK-SAME: [[ARG2:%.+]]: memref<1xf32>, [[ARG3:%.+]]: i1, [[ARG4:%.+]]: i1,
// CHECK-SAME: [[ARG5:%.+]]: memref<2xf32>)
//      CHECK: [[TO_DEALLOC_MR:%.+]] = memref.alloca() : memref<2xindex>
//      CHECK: [[CONDS:%.+]] = memref.alloca() : memref<2xi1>
//      CHECK: [[TO_RETAIN_MR:%.+]] = memref.alloca() : memref<2xindex>
//  CHECK-DAG: [[V0:%.+]] = memref.extract_aligned_pointer_as_index [[ARG0]]
//  CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
//  CHECK-DAG: memref.store [[V0]], [[TO_DEALLOC_MR]][[[C0]]]
//  CHECK-DAG: [[V1:%.+]] = memref.extract_aligned_pointer_as_index [[ARG1]]
//  CHECK-DAG: [[C1:%.+]] = arith.constant 1 : index
//  CHECK-DAG: memref.store [[V1]], [[TO_DEALLOC_MR]][[[C1]]]
//  CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
//  CHECK-DAG: memref.store [[ARG3]], [[CONDS]][[[C0]]]
//  CHECK-DAG: [[C1:%.+]] = arith.constant 1 : index
//  CHECK-DAG: memref.store [[ARG4]], [[CONDS]][[[C1]]]
//  CHECK-DAG: [[V2:%.+]] = memref.extract_aligned_pointer_as_index [[ARG2]]
//  CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
//  CHECK-DAG: memref.store [[V2]], [[TO_RETAIN_MR]][[[C0]]]
//  CHECK-DAG: [[V3:%.+]] = memref.extract_aligned_pointer_as_index [[ARG5]]
//  CHECK-DAG: [[C1:%.+]] = arith.constant 1 : index
//  CHECK-DAG: memref.store [[V3]], [[TO_RETAIN_MR]][[[C1]]]
//  CHECK-DAG: [[CAST_DEALLOC:%.+]] = memref.cast [[TO_DEALLOC_MR]] : memref<2xindex> to memref<?xindex>
//  CHECK-DAG: [[CAST_CONDS:%.+]] = memref.cast [[CONDS]] : memref<2xi1> to memref<?xi1>
//  CHECK-DAG: [[CAST_RETAIN:%.+]] = memref.cast [[TO_RETAIN_MR]] : memref<2xindex> to memref<?xindex>
//      CHECK: [[DEALLOC_CONDS:%.+]] = memref.alloca() : memref<2xi1>
//      CHECK: [[RETAIN_CONDS:%.+]] = memref.alloca() : memref<2xi1>
//      CHECK: [[CAST_DEALLOC_CONDS:%.+]] = memref.cast [[DEALLOC_CONDS]] : memref<2xi1> to memref<?xi1>
//      CHECK: [[CAST_RETAIN_CONDS:%.+]] = memref.cast [[RETAIN_CONDS]] : memref<2xi1> to memref<?xi1>
//      CHECK: call @dealloc_helper([[CAST_DEALLOC]], [[CAST_RETAIN]], [[CAST_CONDS]], [[CAST_DEALLOC_CONDS]], [[CAST_RETAIN_CONDS]])
//      CHECK: [[C0:%.+]] = arith.constant 0 : index
//      CHECK: [[SHOULD_DEALLOC_0:%.+]] = memref.load [[DEALLOC_CONDS]][[[C0]]]
//      CHECK: scf.if [[SHOULD_DEALLOC_0]] {
//      CHECK:   memref.dealloc %arg0
//      CHECK: }
//      CHECK: [[C1:%.+]] = arith.constant 1 : index
//      CHECK: [[SHOULD_DEALLOC_1:%.+]] = memref.load [[DEALLOC_CONDS]][[[C1]]]
//      CHECK: scf.if [[SHOULD_DEALLOC_1]]
//      CHECK:   memref.dealloc [[ARG1]]
//      CHECK: }
//      CHECK: [[C0:%.+]] = arith.constant 0 : index
//      CHECK: [[OWNERSHIP0:%.+]] = memref.load [[RETAIN_CONDS]][[[C0]]]
//      CHECK: [[C1:%.+]] = arith.constant 1 : index
//      CHECK: [[OWNERSHIP1:%.+]] = memref.load [[RETAIN_CONDS]][[[C1]]]
//      CHECK: return [[OWNERSHIP0]], [[OWNERSHIP1]]

//      CHECK: func private @dealloc_helper
// CHECK-SAME: ([[TO_DEALLOC_MR:%.+]]: memref<?xindex>, [[TO_RETAIN_MR:%.+]]: memref<?xindex>,
// CHECK-SAME: [[CONDS:%.+]]: memref<?xi1>, [[DEALLOC_CONDS_OUT:%.+]]: memref<?xi1>,
// CHECK-SAME: [[RETAIN_CONDS_OUT:%.+]]: memref<?xi1>)
//      CHECK:   [[TO_DEALLOC_SIZE:%.+]] = memref.dim [[TO_DEALLOC_MR]], %c0
//      CHECK:   [[TO_RETAIN_SIZE:%.+]] = memref.dim [[TO_RETAIN_MR]], %c0
//      CHECK:   scf.for [[ITER:%.+]] = %c0 to [[TO_RETAIN_SIZE]] step %c1 {
// CHECK-NEXT:     memref.store %false, [[RETAIN_CONDS_OUT]][[[ITER]]]
// CHECK-NEXT:   }
//      CHECK:   scf.for [[OUTER_ITER:%.+]] = %c0 to [[TO_DEALLOC_SIZE]] step %c1 {
//      CHECK:     [[TO_DEALLOC:%.+]] = memref.load [[TO_DEALLOC_MR]][[[OUTER_ITER]]]
// CHECK-NEXT:     [[COND:%.+]] = memref.load [[CONDS]][[[OUTER_ITER]]]
// CHECK-NEXT:     [[NO_RETAIN_ALIAS:%.+]] = scf.for [[ITER:%.+]] = %c0 to [[TO_RETAIN_SIZE]] step %c1 iter_args([[ITER_ARG:%.+]] = %true) -> (i1) {
// CHECK-NEXT:       [[RETAIN_VAL:%.+]] = memref.load [[TO_RETAIN_MR]][[[ITER]]] : memref<?xindex>
// CHECK-NEXT:       [[DOES_ALIAS:%.+]] = arith.cmpi eq, [[RETAIN_VAL]], [[TO_DEALLOC]] : index
// CHECK-NEXT:       scf.if [[DOES_ALIAS]]
// CHECK-NEXT:         [[RETAIN_COND:%.+]] = memref.load [[RETAIN_CONDS_OUT]][[[ITER]]]
// CHECK-NEXT:         [[AGG_RETAIN_COND:%.+]] = arith.ori [[RETAIN_COND]], [[COND]] : i1
// CHECK-NEXT:         memref.store [[AGG_RETAIN_COND]], [[RETAIN_CONDS_OUT]][[[ITER]]]
// CHECK-NEXT:       }
// CHECK-NEXT:       [[DOES_NOT_ALIAS:%.+]] = arith.cmpi ne, [[RETAIN_VAL]], [[TO_DEALLOC]] : index
// CHECK-NEXT:       [[AGG_DOES_NOT_ALIAS:%.+]] = arith.andi [[ITER_ARG]], [[DOES_NOT_ALIAS]] : i1
// CHECK-NEXT:       scf.yield [[AGG_DOES_NOT_ALIAS]] : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     [[SHOULD_DEALLOC:%.+]] = scf.for [[ITER:%.+]] = %c0 to [[OUTER_ITER]] step %c1 iter_args([[ITER_ARG:%.+]] = [[NO_RETAIN_ALIAS]]) -> (i1) {
// CHECK-NEXT:       [[OTHER_DEALLOC_VAL:%.+]] = memref.load [[ARG0]][[[ITER]]] : memref<?xindex>
// CHECK-NEXT:       [[DOES_ALIAS:%.+]] = arith.cmpi ne, [[OTHER_DEALLOC_VAL]], [[TO_DEALLOC]] : index
// CHECK-NEXT:       [[AGG_DOES_ALIAS:%.+]] = arith.andi [[ITER_ARG]], [[DOES_ALIAS]] : i1
// CHECK-NEXT:       scf.yield [[AGG_DOES_ALIAS]] : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     [[DEALLOC_COND:%.+]] = arith.andi [[SHOULD_DEALLOC]], [[COND]] : i1
// CHECK-NEXT:     memref.store [[DEALLOC_COND]], [[DEALLOC_CONDS_OUT]][[[OUTER_ITER]]]
// CHECK-NEXT:   }
// CHECK-NEXT:   return
