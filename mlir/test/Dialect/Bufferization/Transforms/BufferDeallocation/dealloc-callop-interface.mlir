// RUN: mlir-opt -verify-diagnostics -buffer-deallocation=private-function-dynamic-ownership=false \
// RUN:   -buffer-deallocation-simplification -split-input-file %s | FileCheck %s
// RUN: mlir-opt -verify-diagnostics -buffer-deallocation=private-function-dynamic-ownership=true \
// RUN:   --buffer-deallocation-simplification -split-input-file %s | FileCheck %s --check-prefix=CHECK-DYNAMIC

func.func private @f(%arg0: memref<f64>) -> memref<f64> {
  return %arg0 : memref<f64>
}

func.func @function_call() {
  %alloc = memref.alloc() : memref<f64>
  %alloc2 = memref.alloc() : memref<f64>
  %ret = call @f(%alloc) : (memref<f64>) -> memref<f64>
  test.copy(%ret, %alloc2) : (memref<f64>, memref<f64>)
  return
}

// CHECK-LABEL: func @function_call()
//       CHECK: [[ALLOC0:%.+]] = memref.alloc(
//  CHECK-NEXT: [[ALLOC1:%.+]] = memref.alloc(
//  CHECK-NEXT: [[RET:%.+]] = call @f([[ALLOC0]]) : (memref<f64>) -> memref<f64>
//  CHECK-NEXT: test.copy
//  CHECK-NEXT: [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[RET]]
// COM: the following dealloc operation should be split into three since we can
// COM: be sure that the memrefs will never alias according to the buffer
// COM: deallocation ABI, however, the local alias analysis is not powerful
// COM: enough to detect this yet.
//  CHECK-NEXT: bufferization.dealloc ([[ALLOC0]], [[ALLOC1]], [[BASE]] :{{.*}}) if (%true{{[0-9_]*}}, %true{{[0-9_]*}}, %true{{[0-9_]*}})

// CHECK-DYNAMIC-LABEL: func @function_call()
//       CHECK-DYNAMIC: [[ALLOC0:%.+]] = memref.alloc(
//  CHECK-DYNAMIC-NEXT: [[ALLOC1:%.+]] = memref.alloc(
//  CHECK-DYNAMIC-NEXT: [[RET:%.+]]:2 = call @f([[ALLOC0]], %true{{[0-9_]*}}) : (memref<f64>, i1) -> (memref<f64>, i1)
//  CHECK-DYNAMIC-NEXT: test.copy
//  CHECK-DYNAMIC-NEXT: [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[RET]]#0
//  CHECK-DYNAMIC-NEXT: bufferization.dealloc ([[ALLOC0]], [[ALLOC1]], [[BASE]] :{{.*}}) if (%true{{[0-9_]*}}, %true{{[0-9_]*}}, [[RET]]#1)

// -----

func.func @f(%arg0: memref<f64>) -> memref<f64> {
  return %arg0 : memref<f64>
}

func.func @function_call_non_private() {
  %alloc = memref.alloc() : memref<f64>
  %alloc2 = memref.alloc() : memref<f64>
  %ret = call @f(%alloc) : (memref<f64>) -> memref<f64>
  test.copy(%ret, %alloc2) : (memref<f64>, memref<f64>)
  return
}

// CHECK-LABEL: func @function_call_non_private
//       CHECK: [[ALLOC0:%.+]] = memref.alloc(
//       CHECK: [[ALLOC1:%.+]] = memref.alloc(
//       CHECK: [[RET:%.+]] = call @f([[ALLOC0]]) : (memref<f64>) -> memref<f64>
//  CHECK-NEXT: test.copy
//  CHECK-NEXT: [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[RET]]
//  CHECK-NEXT: bufferization.dealloc ([[ALLOC0]], [[ALLOC1]], [[BASE]] :{{.*}}) if (%true{{[0-9_]*}}, %true{{[0-9_]*}}, %true{{[0-9_]*}})
//  CHECK-NEXT: return

// CHECK-DYNAMIC-LABEL: func @function_call_non_private
//       CHECK-DYNAMIC: [[ALLOC0:%.+]] = memref.alloc(
//       CHECK-DYNAMIC: [[ALLOC1:%.+]] = memref.alloc(
//       CHECK-DYNAMIC: [[RET:%.+]] = call @f([[ALLOC0]]) : (memref<f64>) -> memref<f64>
//  CHECK-DYNAMIC-NEXT: test.copy
//  CHECK-DYNAMIC-NEXT: [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[RET]]
//  CHECK-DYNAMIC-NEXT: bufferization.dealloc ([[ALLOC0]], [[ALLOC1]], [[BASE]] :{{.*}}) if (%true{{[0-9_]*}}, %true{{[0-9_]*}}, %true{{[0-9_]*}})
//  CHECK-DYNAMIC-NEXT: return

// -----

func.func private @f(%arg0: memref<f64>) -> memref<f64> {
  return %arg0 : memref<f64>
}

func.func @function_call_requries_merged_ownership_mid_block(%arg0: i1) {
  %alloc = memref.alloc() : memref<f64>
  %alloc2 = memref.alloca() : memref<f64>
  %0 = arith.select %arg0, %alloc, %alloc2 : memref<f64>
  %ret = call @f(%0) : (memref<f64>) -> memref<f64>
  test.copy(%ret, %alloc) : (memref<f64>, memref<f64>)
  return
}

// CHECK-LABEL: func @function_call_requries_merged_ownership_mid_block
//       CHECK:   [[ALLOC0:%.+]] = memref.alloc(
//  CHECK-NEXT:   [[ALLOC1:%.+]] = memref.alloca(
//  CHECK-NEXT:   [[SELECT:%.+]] = arith.select{{.*}}[[ALLOC0]], [[ALLOC1]]
//  CHECK-NEXT:   [[RET:%.+]] = call @f([[SELECT]])
//  CHECK-NEXT:   test.copy
//  CHECK-NEXT:   [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[RET]]
//  CHECK-NEXT:   bufferization.dealloc ([[ALLOC0]], [[BASE]] :
//  CHECK-SAME:     if (%true{{[0-9_]*}}, %true{{[0-9_]*}})
//   CHECK-NOT:     retain
//  CHECK-NEXT:   return

// CHECK-DYNAMIC-LABEL: func @function_call_requries_merged_ownership_mid_block
//       CHECK-DYNAMIC:   [[ALLOC0:%.+]] = memref.alloc(
//  CHECK-DYNAMIC-NEXT:   [[ALLOC1:%.+]] = memref.alloca(
//  CHECK-DYNAMIC-NEXT:   [[SELECT:%.+]] = arith.select{{.*}}[[ALLOC0]], [[ALLOC1]]
//  CHECK-DYNAMIC-NEXT:   [[CLONE:%.+]] = bufferization.clone [[SELECT]]
//  CHECK-DYNAMIC-NEXT:   [[RET:%.+]]:2 = call @f([[CLONE]], %true{{[0-9_]*}})
//  CHECK-DYNAMIC-NEXT:   test.copy
//  CHECK-DYNAMIC-NEXT:   [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[RET]]#0
//  CHECK-DYNAMIC-NEXT:   bufferization.dealloc ([[ALLOC0]], [[CLONE]], [[BASE]] :
//  CHECK-DYNAMIC-SAME:     if (%true{{[0-9_]*}}, %true{{[0-9_]*}}, [[RET]]#1)
//   CHECK-DYNAMIC-NOT:     retain
//  CHECK-DYNAMIC-NEXT:   return

// TODO: the inserted clone is not necessary, we just have to know which of the
// two allocations was selected, either by checking aliasing of the result at
// runtime or by extracting the select condition using an OpInterface or by
// hardcoding the select op
