// RUN: mlir-opt --allow-unregistered-dialect -verify-diagnostics -buffer-deallocation=private-function-dynamic-ownership=false \
// RUN:  --buffer-deallocation-simplification -split-input-file %s | FileCheck %s
// RUN: mlir-opt --allow-unregistered-dialect -verify-diagnostics -buffer-deallocation=private-function-dynamic-ownership=true \
// RUN:  --buffer-deallocation-simplification -split-input-file %s | FileCheck %s --check-prefix=CHECK-DYNAMIC

// RUN: mlir-opt %s -buffer-deallocation-pipeline --split-input-file > /dev/null
// RUN: mlir-opt %s -buffer-deallocation-pipeline=private-function-dynamic-ownership --split-input-file > /dev/null

// Test Case: Existing AllocOp with no users.
// BufferDeallocation expected behavior: It should insert a DeallocOp right
// before ReturnOp.

func.func private @emptyUsesValue(%arg0: memref<4xf32>) {
  %0 = memref.alloc() : memref<4xf32>
  "test.memref_user"(%0) : (memref<4xf32>) -> ()
  return
}

// CHECK-LABEL: func private @emptyUsesValue(
//       CHECK: [[ALLOC:%.*]] = memref.alloc()
//       CHECK: bufferization.dealloc ([[ALLOC]] :
//  CHECK-SAME:   if (%true{{[0-9_]*}}) 
//   CHECK-NOT:   retain
//  CHECK-NEXT: return

// CHECK-DYNAMIC-LABEL: func private @emptyUsesValue(
//  CHECK-DYNAMIC-SAME: [[ARG0:%.+]]: memref<4xf32>, [[ARG1:%.+]]: i1)
//       CHECK-DYNAMIC: [[ALLOC:%.*]] = memref.alloc()
//      CHECK-DYNAMIC: [[BASE:%[a-zA-Z0-9_]+]], {{.*}} = memref.extract_strided_metadata [[ARG0]]
//  CHECK-DYNAMIC-NEXT: bufferization.dealloc ([[BASE]] :{{.*}}) if ([[ARG1]]) 
//   CHECK-DYNAMIC-NOT:   retain
//  CHECK-DYNAMIC-NEXT: bufferization.dealloc ([[ALLOC]] :{{.*}}) if (%true{{[0-9_]*}}) 
//   CHECK-DYNAMIC-NOT:   retain
//  CHECK-DYNAMIC-NEXT: return

// -----

func.func @emptyUsesValue(%arg0: memref<4xf32>) {
  %0 = memref.alloc() : memref<4xf32>
  "test.memref_user"(%0) : (memref<4xf32>) -> ()
  return
}

// CHECK-LABEL: func @emptyUsesValue(

// CHECK-DYNAMIC-LABEL: func @emptyUsesValue(
//       CHECK-DYNAMIC: [[ALLOC:%.*]] = memref.alloc()
//       CHECK-DYNAMIC: bufferization.dealloc ([[ALLOC]] :{{.*}}) if (%true{{[0-9_]*}}) 
//   CHECK-DYNAMIC-NOT:   retain
//  CHECK-DYNAMIC-NEXT: return

// -----

// Test Case: Dead operations in a single block.
// BufferDeallocation expected behavior: It only inserts the two missing
// DeallocOps after the last BufferBasedOp.

func.func private @redundantOperations(%arg0: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  %1 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%0: memref<2xf32>) out(%1: memref<2xf32>)
  return
}

// CHECK-LABEL: func private @redundantOperations
//      CHECK: (%[[ARG0:.*]]: {{.*}})
//      CHECK: %[[FIRST_ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: test.buffer_based
//      CHECK: %[[SECOND_ALLOC:.*]] = memref.alloc()
// CHECK-NEXT: test.buffer_based
// CHECK-NEXT: bufferization.dealloc (%[[FIRST_ALLOC]] : {{.*}}) if (%true{{[0-9_]*}})
// CHECK-NEXT: bufferization.dealloc (%[[SECOND_ALLOC]] : {{.*}}) if (%true{{[0-9_]*}})
// CHECK-NEXT: return

// CHECK-DYNAMIC-LABEL: func private @redundantOperations
//      CHECK-DYNAMIC: (%[[ARG0:.*]]: memref{{.*}}, %[[ARG1:.*]]: i1)
//      CHECK-DYNAMIC: %[[FIRST_ALLOC:.*]] = memref.alloc()
// CHECK-DYNAMIC-NEXT: test.buffer_based
//      CHECK-DYNAMIC: %[[SECOND_ALLOC:.*]] = memref.alloc()
// CHECK-DYNAMIC-NEXT: test.buffer_based
// CHECK-DYNAMIC-NEXT: %[[BASE:[a-zA-Z0-9_]+]], {{.*}} = memref.extract_strided_metadata %[[ARG0]]
// CHECK-DYNAMIC-NEXT: bufferization.dealloc (%[[BASE]] : {{.*}}) if (%[[ARG1]])
// CHECK-DYNAMIC-NEXT: bufferization.dealloc (%[[FIRST_ALLOC]] : {{.*}}) if (%true{{[0-9_]*}})
// CHECK-DYNAMIC-NEXT: bufferization.dealloc (%[[SECOND_ALLOC]] : {{.*}}) if (%true{{[0-9_]*}})
// CHECK-DYNAMIC-NEXT: return

// -----

// Test Case: buffer deallocation escaping
// BufferDeallocation expected behavior: It must not dealloc %arg1 and %x
// since they are operands of return operation and should escape from
// deallocating. It should dealloc %y after CopyOp.

func.func private @memref_in_function_results(
  %arg0: memref<5xf32>,
  %arg1: memref<10xf32>,
  %arg2: memref<5xf32>) -> (memref<10xf32>, memref<15xf32>) {
  %x = memref.alloc() : memref<15xf32>
  %y = memref.alloc() : memref<5xf32>
  test.buffer_based in(%arg0: memref<5xf32>) out(%y: memref<5xf32>)
  test.copy(%y, %arg2) : (memref<5xf32>, memref<5xf32>)
  return %arg1, %x : memref<10xf32>, memref<15xf32>
}

// CHECK-LABEL: func private @memref_in_function_results
//       CHECK: (%[[ARG0:.*]]: memref<5xf32>, %[[ARG1:.*]]: memref<10xf32>,
//  CHECK-SAME: %[[RESULT:.*]]: memref<5xf32>)
//       CHECK: %[[X:.*]] = memref.alloc()
//       CHECK: %[[Y:.*]] = memref.alloc()
//       CHECK: test.copy
//  CHECK-NEXT: %[[V0:.+]] = scf.if %false
//  CHECK-NEXT:   scf.yield %[[ARG1]]
//  CHECK-NEXT: } else {
//  CHECK-NEXT:   %[[CLONE:.+]] = bufferization.clone %[[ARG1]]
//  CHECK-NEXT:   scf.yield %[[CLONE]]
//  CHECK-NEXT: }
//       CHECK: bufferization.dealloc (%[[Y]] : {{.*}}) if (%true{{[0-9_]*}})
//   CHECK-NOT: retain
//       CHECK: return %[[V0]], %[[X]]

// CHECK-DYNAMIC-LABEL: func private @memref_in_function_results
//       CHECK-DYNAMIC: (%[[ARG0:.*]]: memref<5xf32>, %[[ARG1:.*]]: memref<10xf32>,
//  CHECK-DYNAMIC-SAME: %[[RESULT:.*]]: memref<5xf32>, %[[ARG3:.*]]: i1, %[[ARG4:.*]]: i1, %[[ARG5:.*]]: i1)
//       CHECK-DYNAMIC: %[[X:.*]] = memref.alloc()
//       CHECK-DYNAMIC: %[[Y:.*]] = memref.alloc()
//       CHECK-DYNAMIC: test.copy
//       CHECK-DYNAMIC: %[[BASE0:[a-zA-Z0-9_]+]], {{.+}} = memref.extract_strided_metadata %[[ARG0]]
//       CHECK-DYNAMIC: %[[BASE1:[a-zA-Z0-9_]+]], {{.+}} = memref.extract_strided_metadata %[[RESULT]]
//       CHECK-DYNAMIC: bufferization.dealloc (%[[Y]] : {{.*}}) if (%true{{[0-9_]*}})
//   CHECK-DYNAMIC-NOT: retain
//       CHECK-DYNAMIC: [[OWN:%.+]] = bufferization.dealloc (%[[BASE0]], %[[BASE1]] : {{.*}}) if (%[[ARG3]], %[[ARG5]]) retain (%[[ARG1]] :
//       CHECK-DYNAMIC: [[OR:%.+]] = arith.ori [[OWN]], %[[ARG4]]
//       CHECK-DYNAMIC: return %[[ARG1]], %[[X]], [[OR]], %true
