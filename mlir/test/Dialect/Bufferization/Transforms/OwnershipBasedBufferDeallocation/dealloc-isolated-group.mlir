// RUN: mlir-opt -verify-diagnostics -ownership-based-buffer-deallocation \
// RUN:   -buffer-deallocation-simplification -split-input-file %s | FileCheck %s

func.func @function_call() {
  %alloc = memref.alloc() : memref<f64>
  %alloc2 = memref.alloc() : memref<f64>
  %ret = test.isolated_one_region_with_recursive_memory_effects %alloc {
  ^bb0(%arg1: memref<f64>):
    test.region_yield %arg1 : memref<f64>
  } : (memref<f64>) -> memref<f64>
  test.copy(%ret, %alloc2) : (memref<f64>, memref<f64>)
  return
}

// CHECK-LABEL: func @function_call()
//       CHECK: [[ALLOC0:%.+]] = memref.alloc(
//  CHECK-NEXT: [[ALLOC1:%.+]] = memref.alloc(
//  CHECK-NEXT: [[RET:%.+]]:2 = test.isolated_one_region_with_recursive_memory_effects [[ALLOC0]]
//  CHECK-NEXT: ^bb0([[ARG:%.+]]: memref<f64>)
//       CHECK:   test.region_yield [[ARG]]
//   CHECK-NOT:   bufferization.dealloc
//       CHECK: }
//  CHECK-NEXT: test.copy
//  CHECK-NEXT: [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[RET]]#0
//  CHECK-NEXT: bufferization.dealloc ([[ALLOC0]], [[ALLOC1]], [[BASE]] :{{.*}}) if (%true, %true, [[RET]]#1)

// -----

func.func @function_call_requries_merged_ownership_mid_block(%arg0: i1) {
  %alloc = memref.alloc() : memref<f64>
  %alloc2 = memref.alloca() : memref<f64>
  %0 = arith.select %arg0, %alloc, %alloc2 : memref<f64>
  %ret = test.isolated_one_region_with_recursive_memory_effects %0 {
  ^bb0(%arg1: memref<f64>):
    test.region_yield %arg1 : memref<f64>
  } : (memref<f64>) -> (memref<f64>)
  test.copy(%ret, %alloc) : (memref<f64>, memref<f64>)
  return
}

// CHECK-LABEL: func @function_call_requries_merged_ownership_mid_block
//       CHECK:   [[ALLOC0:%.+]] = memref.alloc(
//  CHECK-NEXT:   [[ALLOC1:%.+]] = memref.alloca(
//  CHECK-NEXT:   [[SELECT:%.+]] = arith.select{{.*}}[[ALLOC0]], [[ALLOC1]]
//  CHECK-NEXT:   [[RET:%.+]]:2 = test.isolated_one_region_with_recursive_memory_effects [[SELECT]]
//  CHECK-NEXT:   ^bb0([[ARG:%.+]]: memref<f64>)
//       CHECK:     test.region_yield [[ARG]]
//   CHECK-NOT:     bufferization.dealloc
//       CHECK:   }
//  CHECK-NEXT:   test.copy
//  CHECK-NEXT:   [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[RET]]#0
//  CHECK-NEXT:   bufferization.dealloc ([[ALLOC0]], [[BASE]] :
//  CHECK-SAME:     if (%true, [[RET]]#1)
//   CHECK-NOT:     retain
//  CHECK-NEXT:   return

// -----

func.func @g(%arg0: memref<f32>) -> memref<f32> {
  %0 = test.isolated_one_region_with_recursive_memory_effects %arg0 {
  ^bb0(%arg1: memref<f32>):
    test.region_yield %arg1 : memref<f32>
  } : (memref<f32>) -> (memref<f32>)
  return %0 : memref<f32>
}

// CHECK-LABEL:   func.func @g(
// CHECK-SAME:                 %[[VAL_0:.*]]: memref<f32>) -> memref<f32> {
// CHECK:           %[[BLOCK:.*]]:2 = test.isolated_one_region_with_recursive_memory_effects %[[VAL_0]] {
// CHECK:           ^bb0(%[[ARG:.*]]: memref<f32>):
// CHECK:             test.region_yield %[[ARG]], %false : memref<f32>, i1
// CHECK:           } : (memref<f32>) -> (memref<f32>, i1)
// CHECK:           %[[VAL_4:.*]] = scf.if %[[BLOCK]]#1 -> (memref<f32>) {
// CHECK:             scf.yield %[[BLOCK]]#0 : memref<f32>
// CHECK:           } else {
// CHECK:             %[[VAL_6:.*]] = bufferization.clone %[[BLOCK]]#0 : memref<f32> to memref<f32>
// CHECK:             scf.yield %[[VAL_6]] : memref<f32>
// CHECK:           }
// CHECK:           %[[BUF:.*]], %[[OFFSET:.*]] = memref.extract_strided_metadata %[[BLOCK]]#0 : memref<f32> -> memref<f32>, index
// CHECK:           %[[VAL_11:.*]] = bufferization.dealloc (%[[BUF]] : memref<f32>) if (%[[BLOCK]]#1) retain (%[[VAL_4]] : memref<f32>)
// CHECK:           return %[[VAL_4]] : memref<f32>
// CHECK:         }

// -----

func.func @alloc_yielded_from_block() {
  %alloc = memref.alloc() : memref<f64>
  %alloc2 = memref.alloc() : memref<f64>
  %ret = test.isolated_one_region_with_recursive_memory_effects %alloc {
  ^bb0(%arg1: memref<f64>):
    %0 = memref.load %arg1[] : memref<f64>
    %c1 = arith.constant 1.0 : f64
    %r0 = arith.cmpf oeq, %0, %c1 : f64
    %1 = scf.if %r0 -> memref<f64> {
      %alloc3 = memref.alloc() : memref<f64>
      scf.yield %alloc3 : memref<f64>
    } else {
      scf.yield %arg1 : memref<f64>
    }
    test.region_yield %1 : memref<f64>
  } : (memref<f64>) -> memref<f64>
  test.copy(%ret, %alloc2) : (memref<f64>, memref<f64>)
  return
}

// CHECK-LABEL:   func.func @alloc_yielded_from_block() {
// CHECK:           %true = arith.constant true
// CHECK:           %[[ALLOC:.*]] = memref.alloc() : memref<f64>
// CHECK:           %[[BLOCK:.*]]:2 = test.isolated_one_region_with_recursive_memory_effects %[[ALLOC]] {
// CHECK:           ^bb0(%[[ARG:.*]]: memref<f64>):
// CHECK:             %[[VAL_9:.*]] = arith.cmpf oeq
// CHECK:             %[[VAL_10:.*]]:2 = scf.if %[[VAL_9]] -> (memref<f64>, i1) {
// CHECK:               %[[BLOCK_ALLOC:.*]] = memref.alloc() : memref<f64>
// CHECK:               scf.yield %[[BLOCK_ALLOC]], %true_{{[0-9]*}} : memref<f64>, i1
// CHECK:             } else {
// CHECK:               scf.yield %[[ARG]], %false : memref<f64>, i1
// CHECK:             }
// CHECK:             test.region_yield %[[VAL_10]]#0, %[[VAL_10]]#1 : memref<f64>, i1
// CHECK:           } : (memref<f64>) -> (memref<f64>, i1)
// CHECK:           test.copy
// CHECK:           %[[BUF:.*]], %[[OFFSET:.*]] = memref.extract_strided_metadata %[[BLOCK]]#0 : memref<f64> -> memref<f64>, index
// CHECK:           bufferization.dealloc (%[[ALLOC]], %{{.*}}, %[[BUF]] : memref<f64>, memref<f64>, memref<f64>) if (%true, %true, %[[BLOCK]]#1)
// CHECK:           return
// CHECK:         }

