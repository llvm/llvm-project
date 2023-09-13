// RUN: mlir-opt -verify-diagnostics -buffer-deallocation \
// RUN:  -buffer-deallocation-simplification -split-input-file %s | FileCheck %s
// RUN: mlir-opt -verify-diagnostics -buffer-deallocation=private-function-dynamic-ownership=true -split-input-file %s > /dev/null

// RUN: mlir-opt %s -buffer-deallocation-pipeline --split-input-file > /dev/null

// Test Case:
//    bb0
//   /   \
//  bb1  bb2 <- Initial position of AllocOp
//   \   /
//    bb3
// BufferDeallocation expected behavior: bb2 contains an AllocOp which is
// passed to bb3. In the latter block, there should be a deallocation.
// Since bb1 does not contain an adequate alloc, the deallocation has to be
// made conditional on the branch taken in bb0.

func.func @condBranch(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cf.cond_br %arg0, ^bb2(%arg1 : memref<2xf32>), ^bb1
^bb1:
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cf.br ^bb2(%0 : memref<2xf32>)
^bb2(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func @condBranch
//  CHECK-SAME: ([[ARG0:%.+]]: i1,
//  CHECK-SAME: [[ARG1:%.+]]: memref<2xf32>,
//  CHECK-SAME: [[ARG2:%.+]]: memref<2xf32>)
//   CHECK-NOT: bufferization.dealloc
//       CHECK: cf.cond_br{{.*}}, ^bb2([[ARG1]], %false{{[0-9_]*}} :{{.*}}), ^bb1
//       CHECK: ^bb1:
//       CHECK: %[[ALLOC1:.*]] = memref.alloc
//  CHECK-NEXT: test.buffer_based
//  CHECK-NEXT: cf.br ^bb2(%[[ALLOC1]], %true
//  CHECK-NEXT: ^bb2([[ALLOC2:%.+]]: memref<2xf32>, [[COND1:%.+]]: i1):
//       CHECK: test.copy
//  CHECK-NEXT: [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[ALLOC2]]
//  CHECK-NEXT: bufferization.dealloc ([[BASE]] : {{.*}}) if ([[COND1]])
//  CHECK-NEXT: return

// -----

// Test Case:
//    bb0
//   /   \
//  bb1  bb2 <- Initial position of AllocOp
//   \   /
//    bb3
// BufferDeallocation expected behavior: The existing AllocOp has a dynamic
// dependency to block argument %0 in bb2. Since the dynamic type is passed
// to bb3 via the block argument %2, it is currently required to allocate a
// temporary buffer for %2 that gets copies of %arg0 and %1 with their
// appropriate shape dimensions. The copy buffer deallocation will be applied
// to %2 in block bb3.

func.func @condBranchDynamicType(
  %arg0: i1,
  %arg1: memref<?xf32>,
  %arg2: memref<?xf32>,
  %arg3: index) {
  cf.cond_br %arg0, ^bb2(%arg1 : memref<?xf32>), ^bb1(%arg3: index)
^bb1(%0: index):
  %1 = memref.alloc(%0) : memref<?xf32>
  test.buffer_based in(%arg1: memref<?xf32>) out(%1: memref<?xf32>)
  cf.br ^bb2(%1 : memref<?xf32>)
^bb2(%2: memref<?xf32>):
  test.copy(%2, %arg2) : (memref<?xf32>, memref<?xf32>)
  return
}

// CHECK-LABEL: func @condBranchDynamicType
//  CHECK-SAME: ([[ARG0:%.+]]: i1, [[ARG1:%.+]]: memref<?xf32>, [[ARG2:%.+]]: memref<?xf32>, [[ARG3:%.+]]: index)
//   CHECK-NOT: bufferization.dealloc
//       CHECK: cf.cond_br{{.*}}^bb2(%arg1, %false{{[0-9_]*}} :{{.*}}), ^bb1
//       CHECK: ^bb1([[IDX:%.*]]:{{.*}})
//       CHECK: [[ALLOC1:%.*]] = memref.alloc([[IDX]])
//  CHECK-NEXT: test.buffer_based
//  CHECK-NEXT: cf.br ^bb2([[ALLOC1]], %true
//  CHECK-NEXT: ^bb2([[ALLOC3:%.*]]:{{.*}}, [[COND:%.+]]:{{.*}})
//       CHECK: test.copy([[ALLOC3]],
//  CHECK-NEXT: [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[ALLOC3]]
//  CHECK-NEXT: bufferization.dealloc ([[BASE]] : {{.*}}) if ([[COND]])
//  CHECK-NEXT: return

// -----

// Test case: See above.

func.func @condBranchUnrankedType(
  %arg0: i1,
  %arg1: memref<*xf32>,
  %arg2: memref<*xf32>,
  %arg3: index) {
  cf.cond_br %arg0, ^bb2(%arg1 : memref<*xf32>), ^bb1(%arg3: index)
^bb1(%0: index):
  %1 = memref.alloc(%0) : memref<?xf32>
  %2 = memref.cast %1 : memref<?xf32> to memref<*xf32>
  test.buffer_based in(%arg1: memref<*xf32>) out(%2: memref<*xf32>)
  cf.br ^bb2(%2 : memref<*xf32>)
^bb2(%3: memref<*xf32>):
  test.copy(%3, %arg2) : (memref<*xf32>, memref<*xf32>)
  return
}

// CHECK-LABEL: func @condBranchUnrankedType
//  CHECK-SAME: ([[ARG0:%.+]]: i1, [[ARG1:%.+]]: memref<*xf32>, [[ARG2:%.+]]: memref<*xf32>, [[ARG3:%.+]]: index)
//   CHECK-NOT: bufferization.dealloc
//       CHECK: cf.cond_br{{.*}}^bb2([[ARG1]], %false{{[0-9_]*}} :{{.*}}), ^bb1
//       CHECK: ^bb1([[IDX:%.*]]:{{.*}})
//       CHECK: [[ALLOC1:%.*]] = memref.alloc([[IDX]])
//  CHECK-NEXT: [[CAST:%.+]] = memref.cast [[ALLOC1]]
//  CHECK-NEXT: test.buffer_based
//  CHECK-NEXT: cf.br ^bb2([[CAST]], %true
//  CHECK-NEXT: ^bb2([[ALLOC3:%.*]]:{{.*}}, [[COND:%.+]]:{{.*}})
//       CHECK: test.copy([[ALLOC3]],
//  CHECK-NEXT: [[CAST:%.+]] = memref.reinterpret_cast [[ALLOC3]]
//  CHECK-NEXT: [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[CAST]]
//  CHECK-NEXT: bufferization.dealloc ([[BASE]] : {{.*}}) if ([[COND]])
//  CHECK-NEXT: return

// TODO: we can get rid of first dealloc by doing some must-alias analysis

// -----

// Test Case:
//      bb0
//     /    \
//   bb1    bb2 <- Initial position of AllocOp
//    |     /  \
//    |   bb3  bb4
//    |     \  /
//    \     bb5
//     \    /
//       bb6
//        |
//       bb7
// BufferDeallocation expected behavior: The existing AllocOp has a dynamic
// dependency to block argument %0 in bb2.  Since the dynamic type is passed to
// bb5 via the block argument %2 and to bb6 via block argument %3, it is
// currently required to pass along the condition under which the newly
// allocated buffer should be deallocated, since the path via bb1 does not
// allocate a buffer.

func.func @condBranchDynamicTypeNested(
  %arg0: i1,
  %arg1: memref<?xf32>,
  %arg2: memref<?xf32>,
  %arg3: index) {
  cf.cond_br %arg0, ^bb1, ^bb2(%arg3: index)
^bb1:
  cf.br ^bb6(%arg1 : memref<?xf32>)
^bb2(%0: index):
  %1 = memref.alloc(%0) : memref<?xf32>
  test.buffer_based in(%arg1: memref<?xf32>) out(%1: memref<?xf32>)
  cf.cond_br %arg0, ^bb3, ^bb4
^bb3:
  cf.br ^bb5(%1 : memref<?xf32>)
^bb4:
  cf.br ^bb5(%1 : memref<?xf32>)
^bb5(%2: memref<?xf32>):
  cf.br ^bb6(%2 : memref<?xf32>)
^bb6(%3: memref<?xf32>):
  cf.br ^bb7(%3 : memref<?xf32>)
^bb7(%4: memref<?xf32>):
  test.copy(%4, %arg2) : (memref<?xf32>, memref<?xf32>)
  return
}

// CHECK-LABEL: func @condBranchDynamicTypeNested
//  CHECK-SAME: ([[ARG0:%.+]]: i1, [[ARG1:%.+]]: memref<?xf32>, [[ARG2:%.+]]: memref<?xf32>, [[ARG3:%.+]]: index)
//   CHECK-NOT: bufferization.dealloc
//   CHECK-NOT: bufferization.clone
//       CHECK: cf.cond_br{{.*}}
//  CHECK-NEXT: ^bb1
//   CHECK-NOT: bufferization.dealloc
//   CHECK-NOT: bufferization.clone
//       CHECK: cf.br ^bb5([[ARG1]], %false{{[0-9_]*}} :
//       CHECK: ^bb2([[IDX:%.*]]:{{.*}})
//       CHECK: [[ALLOC1:%.*]] = memref.alloc([[IDX]])
//  CHECK-NEXT: test.buffer_based
//  CHECK-NEXT: [[NOT_ARG0:%.+]] = arith.xori [[ARG0]], %true
//  CHECK-NEXT: [[OWN:%.+]] = arith.select [[ARG0]], [[ARG0]], [[NOT_ARG0]]
//   CHECK-NOT: bufferization.dealloc
//   CHECK-NOT: bufferization.clone
//       CHECK: cf.cond_br{{.*}}, ^bb3, ^bb3
//  CHECK-NEXT: ^bb3:
//   CHECK-NOT: bufferization.dealloc
//   CHECK-NOT: bufferization.clone
//       CHECK: cf.br ^bb4([[ALLOC1]], [[OWN]]
//  CHECK-NEXT: ^bb4([[ALLOC2:%.*]]:{{.*}}, [[COND1:%.+]]:{{.*}})
//   CHECK-NOT: bufferization.dealloc
//   CHECK-NOT: bufferization.clone
//       CHECK: cf.br ^bb5([[ALLOC2]], [[COND1]]
//  CHECK-NEXT: ^bb5([[ALLOC4:%.*]]:{{.*}}, [[COND2:%.+]]:{{.*}})
//  CHECK-NEXT: [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[ALLOC4]]
//  CHECK-NEXT: [[OWN:%.+]]:2 = bufferization.dealloc ([[BASE]] :{{.*}}) if ([[COND2]]) retain ([[ALLOC4]], [[ARG2]] :
//       CHECK: cf.br ^bb6([[ALLOC4]], [[OWN]]#0
//  CHECK-NEXT: ^bb6([[ALLOC5:%.*]]:{{.*}}, [[COND3:%.+]]:{{.*}})
//       CHECK: test.copy
//       CHECK: [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[ALLOC5]]
//  CHECK-NEXT: bufferization.dealloc ([[BASE]] : {{.*}}) if ([[COND3]])
//  CHECK-NEXT: return

// TODO: the dealloc in bb5 can be optimized away by adding another
// canonicalization pattern

// -----

// Test Case:
//    bb0
//   /   \
//  |    bb1 <- Initial position of AllocOp
//   \   /
//    bb2
// BufferDeallocation expected behavior: It should insert a DeallocOp at the
// exit block after CopyOp since %1 is an alias for %0 and %arg1.

func.func @criticalEdge(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  cf.cond_br %arg0, ^bb1, ^bb2(%arg1 : memref<2xf32>)
^bb1:
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cf.br ^bb2(%0 : memref<2xf32>)
^bb2(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func @criticalEdge
//  CHECK-SAME: ([[ARG0:%.+]]: i1, [[ARG1:%.+]]: memref<2xf32>, [[ARG2:%.+]]: memref<2xf32>)
//   CHECK-NOT: bufferization.dealloc
//   CHECK-NOT: bufferization.clone
//       CHECK: cf.cond_br{{.*}}, ^bb1, ^bb2([[ARG1]], %false
//       CHECK: [[ALLOC1:%.*]] = memref.alloc()
//  CHECK-NEXT: test.buffer_based
//  CHECK-NEXT: cf.br ^bb2([[ALLOC1]], %true
//  CHECK-NEXT: ^bb2([[ALLOC2:%.+]]:{{.*}}, [[COND:%.+]]: {{.*}})
//       CHECK: test.copy
//  CHECK-NEXT: [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[ALLOC2]]
//  CHECK-NEXT: bufferization.dealloc ([[BASE]] : {{.*}}) if ([[COND]])
//  CHECK-NEXT: return

// -----

// Test Case:
//    bb0 <- Initial position of AllocOp
//   /   \
//  |    bb1
//   \   /
//    bb2
// BufferDeallocation expected behavior: It only inserts a DeallocOp at the
// exit block after CopyOp since %1 is an alias for %0 and %arg1.

func.func @invCriticalEdge(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cf.cond_br %arg0, ^bb1, ^bb2(%arg1 : memref<2xf32>)
^bb1:
  cf.br ^bb2(%0 : memref<2xf32>)
^bb2(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func @invCriticalEdge
//  CHECK-SAME:  ([[ARG0:%.+]]: i1, [[ARG1:%.+]]: memref<2xf32>, [[ARG2:%.+]]: memref<2xf32>)
//       CHECK:   [[ALLOC:%.+]] = memref.alloc()
//  CHECK-NEXT:   test.buffer_based
//  CHECK-NEXT:   [[NOT_ARG0:%.+]] = arith.xori [[ARG0]], %true
//  CHECK-NEXT:   bufferization.dealloc ([[ALLOC]] : {{.*}}) if ([[NOT_ARG0]])
//  CHECK-NEXT:   cf.cond_br{{.*}}^bb1, ^bb2([[ARG1]], %false
//  CHECK-NEXT: ^bb1:
//   CHECK-NOT:   bufferization.dealloc
//   CHECK-NOT:   bufferization.clone
//       CHECK:   cf.br ^bb2([[ALLOC]], [[ARG0]]
//  CHECK-NEXT: ^bb2([[ALLOC1:%.+]]:{{.*}}, [[COND:%.+]]:{{.*}})
//       CHECK:   test.copy
//  CHECK-NEXT:   [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[ALLOC1]]
//  CHECK-NEXT:   bufferization.dealloc ([[BASE]] : {{.*}}) if ([[COND]])
//  CHECK-NEXT:   return

// -----

// Test Case:
//    bb0 <- Initial position of the first AllocOp
//   /   \
//  bb1  bb2
//   \   /
//    bb3 <- Initial position of the second AllocOp
// BufferDeallocation expected behavior: It only inserts two missing
// DeallocOps in the exit block. %5 is an alias for %0. Therefore, the
// DeallocOp for %0 should occur after the last BufferBasedOp. The Dealloc for
// %7 should happen after CopyOp.

func.func @ifElse(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cf.cond_br %arg0,
    ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>),
    ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  cf.br ^bb3(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  cf.br ^bb3(%3, %4 : memref<2xf32>, memref<2xf32>)
^bb3(%5: memref<2xf32>, %6: memref<2xf32>):
  %7 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%5: memref<2xf32>) out(%7: memref<2xf32>)
  test.copy(%7, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func @ifElse
//  CHECK-SAME: ([[ARG0:%.+]]: i1, [[ARG1:%.+]]: memref<2xf32>, [[ARG2:%.+]]: memref<2xf32>)
//       CHECK:   [[ALLOC0:%.+]] = memref.alloc()
//  CHECK-NEXT:   test.buffer_based
//   CHECK-NOT:   bufferization.dealloc
//   CHECK-NOT:   bufferization.clone
//  CHECK-NEXT:   [[NOT_ARG0:%.+]] = arith.xori [[ARG0]], %true
//  CHECK-NEXT:   cf.cond_br {{.*}}^bb1([[ARG1]], [[ALLOC0]], %false{{[0-9_]*}}, [[ARG0]] : {{.*}}), ^bb2([[ALLOC0]], [[ARG1]], [[NOT_ARG0]], %false{{[0-9_]*}} : {{.*}})
//       CHECK: ^bb3([[A0:%.+]]:{{.*}}, [[A1:%.+]]:{{.*}}, [[COND0:%.+]]: i1, [[COND1:%.+]]: i1):
//       CHECK:   [[ALLOC1:%.+]] = memref.alloc()
//  CHECK-NEXT:   test.buffer_based
//  CHECK-NEXT:   test.copy
//  CHECK-NEXT:   [[BASE0:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A0]]
//  CHECK-NEXT:   [[BASE1:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A1]]
//  CHECK-NEXT:   bufferization.dealloc ([[ALLOC1]] : {{.*}}) if (%true
//   CHECK-NOT:   retain
//  CHECK-NEXT:   bufferization.dealloc ([[BASE0]], [[BASE1]] : {{.*}}) if ([[COND0]], [[COND1]])
//   CHECK-NOT:   retain
//  CHECK-NEXT:   return

// TODO: Instead of deallocating the bbarg memrefs, a slightly better analysis
// could do an unconditional deallocation on ALLOC0 and move it before the
// test.copy (dealloc of ALLOC1 would remain after the copy)

// -----

// Test Case: No users for buffer in if-else CFG
//    bb0 <- Initial position of AllocOp
//   /   \
//  bb1  bb2
//   \   /
//    bb3
// BufferDeallocation expected behavior: It only inserts a missing DeallocOp
// in the exit block since %5 or %6 are the latest aliases of %0.

func.func @ifElseNoUsers(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cf.cond_br %arg0,
    ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>),
    ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  cf.br ^bb3(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  cf.br ^bb3(%3, %4 : memref<2xf32>, memref<2xf32>)
^bb3(%5: memref<2xf32>, %6: memref<2xf32>):
  test.copy(%arg1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func @ifElseNoUsers
//  CHECK-SAME: ([[ARG0:%.+]]: i1, [[ARG1:%.+]]: memref<2xf32>, [[ARG2:%.+]]: memref<2xf32>)
//       CHECK:   [[ALLOC:%.+]] = memref.alloc()
//  CHECK-NEXT:   test.buffer_based
//  CHECK-NEXT:   [[NOT_ARG0:%.+]] = arith.xori [[ARG0]], %true
//  CHECK-NEXT:   cf.cond_br {{.*}}^bb1([[ARG1]], [[ALLOC]], %false{{[0-9_]*}}, [[ARG0]] : {{.*}}), ^bb2([[ALLOC]], [[ARG1]], [[NOT_ARG0]], %false{{[0-9_]*}} : {{.*}})
//       CHECK: ^bb3([[A0:%.+]]:{{.*}}, [[A1:%.+]]:{{.*}}, [[COND0:%.+]]: i1, [[COND1:%.+]]: i1):
//       CHECK:   test.copy
//  CHECK-NEXT:   [[BASE0:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A0]]
//  CHECK-NEXT:   [[BASE1:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A1]]
//  CHECK-NEXT:   bufferization.dealloc ([[BASE0]], [[BASE1]] : {{.*}}) if ([[COND0]], [[COND1]])
//   CHECK-NOT:   retain
//  CHECK-NEXT:   return

// TODO: slightly better analysis could just insert an unconditional dealloc on %0

// -----

// Test Case:
//      bb0 <- Initial position of the first AllocOp
//     /    \
//   bb1    bb2
//    |     /  \
//    |   bb3  bb4
//    \     \  /
//     \     /
//       bb5 <- Initial position of the second AllocOp
// BufferDeallocation expected behavior: Two missing DeallocOps should be
// inserted in the exit block.

func.func @ifElseNested(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  cf.cond_br %arg0,
    ^bb1(%arg1, %0 : memref<2xf32>, memref<2xf32>),
    ^bb2(%0, %arg1 : memref<2xf32>, memref<2xf32>)
^bb1(%1: memref<2xf32>, %2: memref<2xf32>):
  cf.br ^bb5(%1, %2 : memref<2xf32>, memref<2xf32>)
^bb2(%3: memref<2xf32>, %4: memref<2xf32>):
  cf.cond_br %arg0, ^bb3(%3 : memref<2xf32>), ^bb4(%4 : memref<2xf32>)
^bb3(%5: memref<2xf32>):
  cf.br ^bb5(%5, %3 : memref<2xf32>, memref<2xf32>)
^bb4(%6: memref<2xf32>):
  cf.br ^bb5(%3, %6 : memref<2xf32>, memref<2xf32>)
^bb5(%7: memref<2xf32>, %8: memref<2xf32>):
  %9 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%7: memref<2xf32>) out(%9: memref<2xf32>)
  test.copy(%9, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func @ifElseNested
//  CHECK-SAME: ([[ARG0:%.+]]: i1, [[ARG1:%.+]]: memref<2xf32>, [[ARG2:%.+]]: memref<2xf32>)
//       CHECK:   [[ALLOC0:%.+]] = memref.alloc()
//  CHECK-NEXT:   test.buffer_based
//  CHECK-NEXT:   [[NOT_ARG0:%.+]] = arith.xori [[ARG0]], %true
//  CHECK-NEXT:   cf.cond_br {{.*}}^bb1([[ARG1]], [[ALLOC0]], %false{{[0-9_]*}}, [[ARG0]] : {{.*}}), ^bb2([[ALLOC0]], [[ARG1]], [[NOT_ARG0]], %false{{[0-9_]*}} :
//       CHECK: ^bb5([[A0:%.+]]: memref<2xf32>, [[A1:%.+]]: memref<2xf32>, [[COND0:%.+]]: i1, [[COND1:%.+]]: i1):
//       CHECK:   [[ALLOC1:%.+]] = memref.alloc()
//  CHECK-NEXT:   test.buffer_based
//  CHECK-NEXT:   test.copy
//  CHECK-NEXT:   [[BASE0:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A0]]
//  CHECK-NEXT:   [[BASE1:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A1]]
//  CHECK-NEXT:   bufferization.dealloc ([[ALLOC1]] : {{.*}}) if (%true
//   CHECK-NOT:   retain
//  CHECK-NEXT:   bufferization.dealloc ([[BASE0]], [[BASE1]] : {{.*}}) if ([[COND0]], [[COND1]])
//   CHECK-NOT:   retain
//  CHECK-NEXT:   return

// TODO: Instead of deallocating the bbarg memrefs, a slightly better analysis
// could do an unconditional deallocation on ALLOC0 and move it before the
// test.copy (dealloc of ALLOC1 would remain after the copy)

// -----

// Test Case:
//                                     bb0
//                                    /   \
// Initial pos of the 1st AllocOp -> bb1  bb2 <- Initial pos of the 2nd AllocOp
//                                    \   /
//                                     bb3
// BufferDeallocation expected behavior: We need to introduce a copy for each
// buffer since the buffers are passed to bb3. The both missing DeallocOps are
// inserted in the respective block of the allocs. The copy is freed in the exit
// block.

func.func @moving_alloc_and_inserting_missing_dealloc(
  %cond: i1,
    %arg0: memref<2xf32>,
    %arg1: memref<2xf32>) {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %0 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  cf.br ^exit(%0 : memref<2xf32>)
^bb2:
  %1 = memref.alloc() : memref<2xf32>
  test.buffer_based in(%1: memref<2xf32>) out(%arg0: memref<2xf32>)
  cf.br ^exit(%1 : memref<2xf32>)
^exit(%arg2: memref<2xf32>):
  test.copy(%arg2, %arg1) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func @moving_alloc_and_inserting_missing_dealloc
//  CHECK-SAME: ([[ARG0:%.+]]: i1, [[ARG0:%.+]]: memref<2xf32>, [[ARG0:%.+]]: memref<2xf32>)
//       CHECK: ^bb1:
//       CHECK:   [[ALLOC0:%.+]] = memref.alloc()
//  CHECK-NEXT:   test.buffer_based
//  CHECK-NEXT:   cf.br ^bb3([[ALLOC0]], %true
//       CHECK: ^bb2:
//       CHECK:   [[ALLOC1:%.+]] = memref.alloc()
//  CHECK-NEXT:   test.buffer_based
//  CHECK-NEXT:   cf.br ^bb3([[ALLOC1]], %true
//       CHECK: ^bb3([[A0:%.+]]: memref<2xf32>, [[COND0:%.+]]: i1):
//       CHECK:   test.copy
//  CHECK-NEXT:   [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A0]]
//  CHECK-NEXT:   bufferization.dealloc ([[BASE]] : {{.*}}) if ([[COND0]])
//  CHECK-NEXT:   return

// -----

func.func @select_aliases(%arg0: index, %arg1: memref<?xi8>, %arg2: i1) {
  %0 = memref.alloc(%arg0) : memref<?xi8>
  %1 = memref.alloc(%arg0) : memref<?xi8>
  %2 = arith.select %arg2, %0, %1 : memref<?xi8>
  test.copy(%2, %arg1) : (memref<?xi8>, memref<?xi8>)
  return
}

// CHECK-LABEL: func @select_aliases
// CHECK: [[ALLOC0:%.+]] = memref.alloc(
// CHECK: [[ALLOC1:%.+]] = memref.alloc(
// CHECK: arith.select
// CHECK: test.copy
// CHECK: bufferization.dealloc ([[ALLOC0]] : {{.*}}) if (%true
// CHECK-NOT: retain
// CHECK: bufferization.dealloc ([[ALLOC1]] : {{.*}}) if (%true
// CHECK-NOT: retain

// -----

func.func @select_aliases_not_same_ownership(%arg0: index, %arg1: memref<?xi8>, %arg2: i1) {
  %0 = memref.alloc(%arg0) : memref<?xi8>
  %1 = memref.alloca(%arg0) : memref<?xi8>
  %2 = arith.select %arg2, %0, %1 : memref<?xi8>
  cf.br ^bb1(%2 : memref<?xi8>)
^bb1(%arg3: memref<?xi8>):
  test.copy(%arg3, %arg1) : (memref<?xi8>, memref<?xi8>)
  return
}

// CHECK-LABEL: func @select_aliases_not_same_ownership
// CHECK: ([[ARG0:%.+]]: index, [[ARG1:%.+]]: memref<?xi8>, [[ARG2:%.+]]: i1)
// CHECK: [[ALLOC0:%.+]] = memref.alloc(
// CHECK: [[ALLOC1:%.+]] = memref.alloca(
// CHECK: [[SELECT:%.+]] = arith.select
// CHECK: [[OWN:%.+]] = bufferization.dealloc ([[ALLOC0]] :{{.*}}) if (%true{{[0-9_]*}}) retain ([[SELECT]] :
// CHECK: cf.br ^bb1([[SELECT]], [[OWN]] :
// CHECK: ^bb1([[A0:%.+]]: memref<?xi8>, [[COND:%.+]]: i1)
// CHECK: test.copy
// CHECK: [[BASE0:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A0]]
// CHECK: bufferization.dealloc ([[BASE0]] : {{.*}}) if ([[COND]])
// CHECK-NOT: retain

// -----

func.func @select_captured_in_next_block(%arg0: index, %arg1: memref<?xi8>, %arg2: i1, %arg3: i1) {
  %0 = memref.alloc(%arg0) : memref<?xi8>
  %1 = memref.alloca(%arg0) : memref<?xi8>
  %2 = arith.select %arg2, %0, %1 : memref<?xi8>
  cf.cond_br %arg3, ^bb1(%0 : memref<?xi8>), ^bb1(%arg1 : memref<?xi8>)
^bb1(%arg4: memref<?xi8>):
  test.copy(%arg4, %2) : (memref<?xi8>, memref<?xi8>)
  return
}

// CHECK-LABEL: func @select_captured_in_next_block
// CHECK: ([[ARG0:%.+]]: index, [[ARG1:%.+]]: memref<?xi8>, [[ARG2:%.+]]: i1, [[ARG3:%.+]]: i1)
// CHECK: [[ALLOC0:%.+]] = memref.alloc(
// CHECK: [[ALLOC1:%.+]] = memref.alloca(
// CHECK: [[SELECT:%.+]] = arith.select
// CHECK: [[OWN0:%.+]]:2 = bufferization.dealloc ([[ALLOC0]] :{{.*}}) if ([[ARG3]]) retain ([[ALLOC0]], [[SELECT]] :
// CHECK: [[NOT_ARG3:%.+]] = arith.xori [[ARG3]], %true
// CHECK: [[OWN1:%.+]] = bufferization.dealloc ([[ALLOC0]] :{{.*}}) if ([[NOT_ARG3]]) retain ([[SELECT]] :
// CHECK: [[MERGED_OWN:%.+]] = arith.select [[ARG3]], [[OWN0]]#1, [[OWN1]]
// CHECK: cf.cond_br{{.*}}^bb1([[ALLOC0]], [[OWN0]]#0 :{{.*}}), ^bb1([[ARG1]], %false
// CHECK: ^bb1([[A0:%.+]]: memref<?xi8>, [[COND:%.+]]: i1)
// CHECK: test.copy
// CHECK: [[BASE0:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[SELECT]]
// CHECK: [[BASE1:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A0]]
// CHECK: bufferization.dealloc ([[BASE0]], [[BASE1]] : {{.*}}) if ([[MERGED_OWN]], [[COND]])

// There are two interesting parts here:
// * The dealloc condition of %0 in the second block should be the corresponding
// result of the dealloc operation of the first block, because %0 has unknown
// ownership status and thus would other wise require a clone in the first
// block.
// * The dealloc of the first block must make sure that the branch condition and
// respective retained values are handled correctly, i.e., only the ones for the
// actual branch taken have to be retained.

// -----

func.func @blocks_not_preordered_by_dominance() {
  cf.br ^bb1
^bb2:
  "test.memref_user"(%alloc) : (memref<2xi32>) -> ()
  return
^bb1:
  %alloc = memref.alloc() : memref<2xi32>
  cf.br ^bb2
}

// CHECK-LABEL: func @blocks_not_preordered_by_dominance
//  CHECK-NEXT:   [[TRUE:%.+]] = arith.constant true
//  CHECK-NEXT:   cf.br [[BB1:\^.+]]
//  CHECK-NEXT: [[BB2:\^[a-zA-Z0-9_]+]]:
//  CHECK-NEXT:   "test.memref_user"([[ALLOC:%[a-zA-Z0-9_]+]])
//  CHECK-NEXT:   bufferization.dealloc ([[ALLOC]] : {{.*}}) if ([[TRUE]])
//   CHECK-NOT: retain
//  CHECK-NEXT:   return
//  CHECK-NEXT: [[BB1]]:
//  CHECK-NEXT:   [[ALLOC]] = memref.alloc()
//  CHECK-NEXT:   cf.br [[BB2]]
//  CHECK-NEXT: }
