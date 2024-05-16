// RUN: mlir-opt -verify-diagnostics -ownership-based-buffer-deallocation \
// RUN:   --buffer-deallocation-simplification -split-input-file %s | FileCheck %s
// RUN: mlir-opt -verify-diagnostics -ownership-based-buffer-deallocation=private-function-dynamic-ownership=true -split-input-file %s > /dev/null

// RUN: mlir-opt %s -buffer-deallocation-pipeline --split-input-file > /dev/null

// Test Case: Dead operations in a single block.
// BufferDeallocation expected behavior: It only inserts the two missing
// DeallocOps after the last BufferBasedOp.

// CHECK-LABEL: func @redundantOperations
func.func @redundantOperations(%arg0: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  %1 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%0: memref<2xf32>) out(%1: memref<2xf32>)
  return
}

//      CHECK: (%[[ARG0:.*]]: {{.*}})
//      CHECK: %[[FIRST_ALLOC:.*]] = memref.alloc()
//  CHECK-NOT: bufferization.dealloc
//      CHECK: test.buffer_based in(%[[ARG0]]{{.*}}out(%[[FIRST_ALLOC]]
//  CHECK-NOT: bufferization.dealloc
//      CHECK: %[[SECOND_ALLOC:.*]] = memref.alloc()
//  CHECK-NOT: bufferization.dealloc
//      CHECK: test.buffer_based in(%[[FIRST_ALLOC]]{{.*}}out(%[[SECOND_ALLOC]]
//      CHECK: bufferization.dealloc (%[[FIRST_ALLOC]] :{{.*}}) if (%true{{[0-9_]*}})
//      CHECK: bufferization.dealloc (%[[SECOND_ALLOC]] :{{.*}}) if (%true{{[0-9_]*}})
// CHECK-NEXT: return

// TODO: The dealloc could be split in two to avoid runtime aliasing checks
// since we can be sure at compile time that they will never alias.

// -----

// CHECK-LABEL: func @allocaIsNotDeallocated
func.func @allocaIsNotDeallocated(%arg0: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  %1 = memref.alloca() : memref<2xf32>
  test.buffer_based in(%0: memref<2xf32>) out(%1: memref<2xf32>)
  return
}

//      CHECK: (%[[ARG0:.*]]: {{.*}})
//      CHECK: %[[FIRST_ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: test.buffer_based in(%[[ARG0]]{{.*}}out(%[[FIRST_ALLOC]]
// CHECK-NEXT: %[[SECOND_ALLOC:.*]] = memref.alloca()
// CHECK-NEXT: test.buffer_based in(%[[FIRST_ALLOC]]{{.*}}out(%[[SECOND_ALLOC]]
//      CHECK: bufferization.dealloc (%[[FIRST_ALLOC]] :{{.*}}) if (%true{{[0-9_]*}})
// CHECK-NEXT: return

// -----

// Test Case: Inserting missing DeallocOp in a single block.

// CHECK-LABEL: func @inserting_missing_dealloc_simple
func.func @inserting_missing_dealloc_simple(
  %arg0 : memref<2xf32>,
  %arg1: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  test.copy(%0, %arg1) : (memref<2xf32>, memref<2xf32>)
  return
}

//      CHECK: %[[ALLOC0:.*]] = memref.alloc()
//      CHECK: test.copy
//      CHECK: bufferization.dealloc (%[[ALLOC0]] :{{.*}}) if (%true{{[0-9_]*}})

// -----

// Test Case: The ownership indicator is set to false for alloca

// CHECK-LABEL: func @alloca_ownership_indicator_is_false
func.func @alloca_ownership_indicator_is_false() {
  %0 = memref.alloca() : memref<2xf32>
  cf.br ^bb1(%0: memref<2xf32>)
^bb1(%arg0 : memref<2xf32>):
  return
}

//      CHECK:  %[[ALLOC0:.*]] = memref.alloca()
// CHECK-NEXT:   cf.br ^bb1(%[[ALLOC0]], %false :
// CHECK-NEXT: ^bb1([[A0:%.+]]: memref<2xf32>, [[COND0:%.+]]: i1):
//      CHECK:   [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A0]]
//      CHECK:   bufferization.dealloc ([[BASE]] : {{.*}}) if ([[COND0]])
// CHECK-NEXT:   return

// -----

func.func @dealloc_existing_clones(%arg0: memref<?x?xf64>, %arg1: memref<?x?xf64>) -> memref<?x?xf64> {
  %0 = bufferization.clone %arg0 : memref<?x?xf64> to memref<?x?xf64>
  %1 = bufferization.clone %arg1 : memref<?x?xf64> to memref<?x?xf64>
  return %0 : memref<?x?xf64>
}

// CHECK-LABEL: func @dealloc_existing_clones
//       CHECK: (%[[ARG0:.*]]: memref<?x?xf64>, %[[ARG1:.*]]: memref<?x?xf64>)
//       CHECK: %[[RES0:.*]] = bufferization.clone %[[ARG0]]
//       CHECK: %[[RES1:.*]] = bufferization.clone %[[ARG1]]
//  CHECK-NEXT: bufferization.dealloc (%[[RES1]] :{{.*}}) if (%true{{[0-9_]*}})
//   CHECK-NOT: retain
//  CHECK-NEXT: return %[[RES0]]

// TODO: The retain operand could be dropped to avoid runtime aliasing checks
// since We can guarantee at compile-time that it will never alias with the
// dealloc operand

// -----

memref.global "private" constant @__constant_4xf32 : memref<4xf32> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]>

func.func @op_without_aliasing_and_allocation() -> memref<4xf32> {
  %0 = memref.get_global @__constant_4xf32 : memref<4xf32>
  return %0 : memref<4xf32>
}

// CHECK-LABEL: func @op_without_aliasing_and_allocation
//       CHECK:   [[GLOBAL:%.+]] = memref.get_global @__constant_4xf32
//       CHECK:   [[RES:%.+]] = scf.if %false
//       CHECK:     scf.yield [[GLOBAL]] :
//       CHECK:     [[CLONE:%.+]] = bufferization.clone [[GLOBAL]]
//       CHECK:     scf.yield [[CLONE]] :
//       CHECK:   return [[RES]] :

// -----

// Allocations with "bufferization.manual_deallocation" are assigned an
// ownership of "false".

func.func @manual_deallocation(%c: i1, %f: f32, %idx: index) -> f32 {
  %0 = memref.alloc() {bufferization.manual_deallocation} : memref<5xf32>
  linalg.fill ins(%f : f32) outs(%0 : memref<5xf32>)
  %1 = memref.alloc() : memref<5xf32>
  linalg.fill ins(%f : f32) outs(%1 : memref<5xf32>)
  %2 = arith.select %c, %0, %1 : memref<5xf32>
  %3 = memref.load %2[%idx] : memref<5xf32>

  // Only buffers that are under "manual deallocation" are allowed to be
  // deallocated with memref.dealloc. For consistency reasons, the
  // manual_deallocation attribute must also be specified. A runtime insertion
  // is inserted to ensure that we do not have ownership. (This is not a
  // bulletproof check, but covers some cases of invalid IR.)
  memref.dealloc %0 {bufferization.manual_deallocation} : memref<5xf32>

  return %3 : f32
}

// CHECK-LABEL: func @manual_deallocation(
//       CHECK:   %[[true:.*]] = arith.constant true
//       CHECK:   %[[manual_alloc:.*]] = memref.alloc() {bufferization.manual_deallocation} : memref<5xf32>
//       CHECK:   %[[managed_alloc:.*]] = memref.alloc() : memref<5xf32>
//       CHECK:   %[[selected:.*]] = arith.select
//       CHECK:   cf.assert %[[true]], "expected that the block does not have ownership"
//       CHECK:   memref.dealloc %[[manual_alloc]]
//       CHECK:   bufferization.dealloc (%[[managed_alloc]] : memref<5xf32>) if (%[[true]])
