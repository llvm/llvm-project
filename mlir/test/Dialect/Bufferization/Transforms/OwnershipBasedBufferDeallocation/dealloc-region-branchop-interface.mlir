// RUN: mlir-opt -allow-unregistered-dialect -verify-diagnostics -ownership-based-buffer-deallocation \
// RUN:  --buffer-deallocation-simplification -split-input-file %s | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect -verify-diagnostics -ownership-based-buffer-deallocation=private-function-dynamic-ownership=true -split-input-file %s > /dev/null

// RUN: mlir-opt %s -buffer-deallocation-pipeline --split-input-file --verify-diagnostics > /dev/null

// Test Case: Nested regions - This test defines a BufferBasedOp inside the
// region of a RegionBufferBasedOp.
// BufferDeallocation expected behavior: The AllocOp for the BufferBasedOp
// should remain inside the region of the RegionBufferBasedOp and it should insert
// the missing DeallocOp in the same region. The missing DeallocOp should be
// inserted after CopyOp.

func.func @nested_regions_and_cond_branch(
  %arg0: i1,
  %arg1: memref<2xf32>,
  %arg2: memref<2xf32>) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = memref.alloc() : memref<2xf32>
  test.region_buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>) {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %1 = memref.alloc() : memref<2xf32>
    test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
    %tmp1 = math.exp %gen1_arg0 : f32
    test.region_yield %tmp1 : f32
  }
  cf.br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func @nested_regions_and_cond_branch
//  CHECK-SAME: ([[ARG0:%.+]]: i1, [[ARG1:%.+]]: memref<2xf32>, [[ARG2:%.+]]: memref<2xf32>)
//       CHECK: ^bb1:
//   CHECK-NOT:   bufferization.clone
//   CHECK-NOT:   bufferization.dealloc
//       CHECK:   cf.br ^bb3([[ARG1]], %false
//       CHECK: ^bb2:
//       CHECK:   [[ALLOC0:%.+]] = memref.alloc()
//       CHECK:   test.region_buffer_based
//       CHECK:     [[ALLOC1:%.+]] = memref.alloc()
//       CHECK:     test.buffer_based
//       CHECK:     bufferization.dealloc ([[ALLOC1]] : memref<2xf32>) if (%true
//  CHECK-NEXT:     test.region_yield
//   CHECK-NOT:   bufferization.clone
//   CHECK-NOT:   bufferization.dealloc
//       CHECK:   cf.br ^bb3([[ALLOC0]], %true
//       CHECK: ^bb3([[A0:%.+]]: memref<2xf32>, [[COND0:%.+]]: i1):
//       CHECK:   test.copy
//  CHECK-NEXT:   [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A0]]
//  CHECK-NEXT:   bufferization.dealloc ([[BASE]] : {{.*}}) if ([[COND0]])
//       CHECK:   return

// -----

// Test Case: nested region control flow
// The alloc %1 flows through both if branches until it is finally returned.
// Hence, it does not require a specific dealloc operation. However, %3
// requires a dealloc.

func.func @nested_region_control_flow(
  %arg0 : index,
  %arg1 : index) -> memref<?x?xf32> {
  %0 = arith.cmpi eq, %arg0, %arg1 : index
  %1 = memref.alloc(%arg0, %arg0) : memref<?x?xf32>
  %2 = scf.if %0 -> (memref<?x?xf32>) {
    scf.yield %1 : memref<?x?xf32>
  } else {
    %3 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    "test.read_buffer"(%3) : (memref<?x?xf32>) -> ()
    scf.yield %1 : memref<?x?xf32>
  }
  return %2 : memref<?x?xf32>
}

// CHECK-LABEL: func @nested_region_control_flow
//       CHECK:   [[ALLOC:%.+]] = memref.alloc(
//       CHECK:   [[V0:%.+]]:2 = scf.if
//       CHECK:     scf.yield [[ALLOC]], %false
//       CHECK:     [[ALLOC1:%.+]] = memref.alloc(
//       CHECK:     bufferization.dealloc ([[ALLOC1]] :{{.*}}) if (%true{{[0-9_]*}})
//   CHECK-NOT: retain
//       CHECK:     scf.yield [[ALLOC]], %false
//       CHECK:   [[V1:%.+]] = scf.if [[V0]]#1
//       CHECK:     scf.yield [[V0]]#0
//       CHECK:     [[CLONE:%.+]] = bufferization.clone [[V0]]#0
//       CHECK:     scf.yield [[CLONE]]
//       CHECK:   [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#0
//       CHECK:   bufferization.dealloc ([[ALLOC]], [[BASE]] : {{.*}}) if (%true{{[0-9_]*}}, [[V0]]#1) retain ([[V1]] :
//       CHECK:   return [[V1]]

// -----

// Test Case: nested region control flow with a nested buffer allocation in a
// divergent branch.
// Buffer deallocation places a copy for both  %1 and %3, since they are
// returned in the end.

func.func @nested_region_control_flow_div(
  %arg0 : index,
  %arg1 : index) -> memref<?x?xf32> {
  %0 = arith.cmpi eq, %arg0, %arg1 : index
  %1 = memref.alloc(%arg0, %arg0) : memref<?x?xf32>
  %2 = scf.if %0 -> (memref<?x?xf32>) {
    scf.yield %1 : memref<?x?xf32>
  } else {
    %3 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
    scf.yield %3 : memref<?x?xf32>
  }
  return %2 : memref<?x?xf32>
}

// CHECK-LABEL: func @nested_region_control_flow_div
//       CHECK:   [[ALLOC:%.+]] = memref.alloc(
//       CHECK:   [[V0:%.+]]:2 = scf.if
//       CHECK:     scf.yield [[ALLOC]], %false
//       CHECK:     [[ALLOC1:%.+]] = memref.alloc(
//       CHECK:     scf.yield [[ALLOC1]], %true
//       CHECK:   [[V1:%.+]] = scf.if [[V0]]#1
//       CHECK:     scf.yield [[V0]]#0
//       CHECK:     [[CLONE:%.+]] = bufferization.clone [[V0]]#0
//       CHECK:     scf.yield [[CLONE]]
//       CHECK:   [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#0
//       CHECK:   bufferization.dealloc ([[ALLOC]], [[BASE]] :{{.*}}) if (%true{{[0-9_]*}}, [[V0]]#1) retain ([[V1]] :
//       CHECK:   return [[V1]]

// -----

// Test Case: nested region control flow within a region interface.
// No copies are required in this case since the allocation finally escapes
// the method.

func.func @inner_region_control_flow(%arg0 : index) -> memref<?x?xf32> {
  %0 = memref.alloc(%arg0, %arg0) : memref<?x?xf32>
  %1 = test.region_if %0 : memref<?x?xf32> -> (memref<?x?xf32>) then {
    ^bb0(%arg1 : memref<?x?xf32>):
      test.region_if_yield %arg1 : memref<?x?xf32>
  } else {
    ^bb0(%arg1 : memref<?x?xf32>):
      test.region_if_yield %arg1 : memref<?x?xf32>
  } join {
    ^bb0(%arg1 : memref<?x?xf32>):
      test.region_if_yield %arg1 : memref<?x?xf32>
  }
  return %1 : memref<?x?xf32>
}

// CHECK-LABEL: func.func @inner_region_control_flow
//       CHECK:   [[ALLOC:%.+]] = memref.alloc(
//       CHECK:   [[V0:%.+]]:2 = test.region_if [[ALLOC]], %false
//       CHECK:   ^bb0([[ARG1:%.+]]: memref<?x?xf32>, [[ARG2:%.+]]: i1):
//       CHECK:     test.region_if_yield [[ARG1]], [[ARG2]]
//       CHECK:   ^bb0([[ARG1:%.+]]: memref<?x?xf32>, [[ARG2:%.+]]: i1):
//       CHECK:     test.region_if_yield [[ARG1]], [[ARG2]]
//       CHECK:   ^bb0([[ARG1:%.+]]: memref<?x?xf32>, [[ARG2:%.+]]: i1):
//       CHECK:     test.region_if_yield [[ARG1]], [[ARG2]]
//       CHECK:   [[V1:%.+]] = scf.if [[V0]]#1
//       CHECK:     scf.yield [[V0]]#0
//       CHECK:     [[CLONE:%.+]] = bufferization.clone [[V0]]#0
//       CHECK:     scf.yield [[CLONE]]
//       CHECK:   [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#0
//       CHECK:   bufferization.dealloc ([[ALLOC]], [[BASE]] :{{.*}}) if (%true{{[0-9_]*}}, [[V0]]#1) retain ([[V1]] :
//       CHECK:   return [[V1]]

// -----

func.func @nestedRegionsAndCondBranchAlloca(
  %arg0: i1,
  %arg1: memref<2xf32>,
  %arg2: memref<2xf32>) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  cf.br ^bb3(%arg1 : memref<2xf32>)
^bb2:
  %0 = memref.alloc() : memref<2xf32>
  test.region_buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>) {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %1 = memref.alloca() : memref<2xf32>
    test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
    %tmp1 = math.exp %gen1_arg0 : f32
    test.region_yield %tmp1 : f32
  }
  cf.br ^bb3(%0 : memref<2xf32>)
^bb3(%1: memref<2xf32>):
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func @nestedRegionsAndCondBranchAlloca
//  CHECK-SAME: ([[ARG0:%.+]]: i1, [[ARG1:%.+]]: memref<2xf32>, [[ARG2:%.+]]: memref<2xf32>)
//       CHECK: ^bb1:
//       CHECK:   cf.br ^bb3([[ARG1]], %false
//       CHECK: ^bb2:
//       CHECK:   [[ALLOC:%.+]] = memref.alloc()
//       CHECK:   test.region_buffer_based
//       CHECK:     memref.alloca()
//       CHECK:     test.buffer_based
//   CHECK-NOT:     bufferization.dealloc
//   CHECK-NOT:     bufferization.clone
//       CHECK:     test.region_yield
//       CHECK:   }
//       CHECK:   cf.br ^bb3([[ALLOC]], %true
//       CHECK: ^bb3([[A0:%.+]]: memref<2xf32>, [[COND:%.+]]: i1):
//       CHECK:   test.copy
//       CHECK:   [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[A0]]
//       CHECK:   bufferization.dealloc ([[BASE]] :{{.*}}) if ([[COND]])

// -----

func.func @nestedRegionControlFlowAlloca(
  %arg0 : index, %arg1 : index, %arg2: f32) -> memref<?x?xf32> {
  %0 = arith.cmpi eq, %arg0, %arg1 : index
  %1 = memref.alloc(%arg0, %arg0) : memref<?x?xf32>
  %2 = scf.if %0 -> (memref<?x?xf32>) {
    scf.yield %1 : memref<?x?xf32>
  } else {
    %3 = memref.alloca(%arg0, %arg1) : memref<?x?xf32>
    %c0 = arith.constant 0 : index
    memref.store %arg2, %3[%c0, %c0] : memref<?x?xf32>
    scf.yield %1 : memref<?x?xf32>
  }
  return %2 : memref<?x?xf32>
}

// CHECK-LABEL: func @nestedRegionControlFlowAlloca
//       CHECK: [[ALLOC:%.+]] = memref.alloc(
//       CHECK: [[V0:%.+]]:2 = scf.if
//       CHECK:   scf.yield [[ALLOC]], %false
//       CHECK:   memref.alloca(
//       CHECK:   scf.yield [[ALLOC]], %false
//       CHECK: [[V1:%.+]] = scf.if [[V0]]#1
//       CHECK:   scf.yield [[V0]]#0
//       CHECK:   [[CLONE:%.+]] = bufferization.clone [[V0]]#0
//       CHECK:   scf.yield [[CLONE]]
//       CHECK: [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#0
//       CHECK: bufferization.dealloc ([[ALLOC]], [[BASE]] :{{.*}}) if (%true{{[0-9_]*}}, [[V0]]#1) retain ([[V1]] :
//       CHECK: return [[V1]]

// -----

// Test Case: structured control-flow loop using a nested alloc.
// The iteration argument %iterBuf has to be freed before yielding %3 to avoid
// memory leaks.

func.func @loop_alloc(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  "test.read_buffer"(%0) : (memref<2xf32>) -> ()
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %2 = arith.cmpi eq, %i, %ub : index
    %3 = memref.alloc() : memref<2xf32>
    scf.yield %3 : memref<2xf32>
  }
  test.copy(%1, %res) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func @loop_alloc
//  CHECK-SAME: ([[ARG0:%.+]]: index, [[ARG1:%.+]]: index, [[ARG2:%.+]]: index, [[ARG3:%.+]]: memref<2xf32>, [[ARG4:%.+]]: memref<2xf32>)
//       CHECK: [[ALLOC:%.+]] = memref.alloc()
//       CHECK: [[V0:%.+]]:2 = scf.for {{.*}} iter_args([[ARG6:%.+]] = [[ARG3]], [[ARG7:%.+]] = %false
//       CHECK:   [[ALLOC1:%.+]] = memref.alloc()
//       CHECK:   [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[ARG6]]
//       CHECK:   bufferization.dealloc ([[BASE]] :{{.*}}) if ([[ARG7]])
//   CHECK-NOT:       retain
//       CHECK:   scf.yield [[ALLOC1]], %true
//       CHECK: test.copy
//       CHECK: [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#0
//       CHECK: bufferization.dealloc ([[ALLOC]] :{{.*}}) if (%true
//   CHECK-NOT: retain
//       CHECK: bufferization.dealloc ([[BASE]] :{{.*}}) if ([[V0]]#1)
//   CHECK-NOT: retain

// -----

// Test Case: structured control-flow loop with a nested if operation.
// The loop yields buffers that have been defined outside of the loop and the
// backedges only use the iteration arguments (or one of its aliases).
// Therefore, we do not have to (and are not allowed to) free any buffers
// that are passed via the backedges.

func.func @loop_nested_if_no_alloc(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %2 = arith.cmpi eq, %i, %ub : index
    %3 = scf.if %2 -> (memref<2xf32>) {
      scf.yield %0 : memref<2xf32>
    } else {
      scf.yield %iterBuf : memref<2xf32>
    }
    scf.yield %3 : memref<2xf32>
  }
  test.copy(%1, %res) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func @loop_nested_if_no_alloc
//  CHECK-SAME: ({{.*}}, [[ARG3:%.+]]: memref<2xf32>, [[ARG4:%.+]]: memref<2xf32>)
//       CHECK: [[ALLOC:%.+]] = memref.alloc()
//       CHECK: [[V0:%.+]]:2 = scf.for {{.*}} iter_args([[ARG6:%.+]] = [[ARG3]], [[ARG7:%.+]] = %false
//       CHECK:   [[V1:%.+]]:2 = scf.if
//       CHECK:     scf.yield [[ALLOC]], %false
//       CHECK:     scf.yield [[ARG6]], %false
//       CHECK:   [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[ARG6]]
//       CHECK:   [[OWN:%.+]] = bufferization.dealloc ([[BASE]] :{{.*}}) if ([[ARG7]]) retain ([[V1]]#0 :
//       CHECK:   [[OWN_AGG:%.+]] = arith.ori [[OWN]], [[V1]]#1
//       CHECK:   scf.yield [[V1]]#0, [[OWN_AGG]]
//       CHECK: test.copy
//       CHECK: [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#0
//       CHECK: bufferization.dealloc ([[ALLOC]], [[BASE]] :{{.*}}) if (%true{{[0-9_]*}}, [[V0]]#1)

// TODO: we know statically that the inner dealloc will never deallocate
//       anything, i.e., we can optimize it away

// -----

// Test Case: structured control-flow loop with a nested if operation using
// a deeply nested buffer allocation.

func.func @loop_nested_if_alloc(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>) -> memref<2xf32> {
  %0 = memref.alloc() : memref<2xf32>
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %2 = arith.cmpi eq, %i, %ub : index
    %3 = scf.if %2 -> (memref<2xf32>) {
      %4 = memref.alloc() : memref<2xf32>
      scf.yield %4 : memref<2xf32>
    } else {
      scf.yield %0 : memref<2xf32>
    }
    scf.yield %3 : memref<2xf32>
  }
  return %1 : memref<2xf32>
}

// CHECK-LABEL: func @loop_nested_if_alloc
//  CHECK-SAME: ({{.*}}, [[ARG3:%.+]]: memref<2xf32>)
//       CHECK: [[ALLOC:%.+]] = memref.alloc()
//       CHECK: [[V0:%.+]]:2 = scf.for {{.*}} iter_args([[ARG5:%.+]] = [[ARG3]], [[ARG6:%.+]] = %false
//       CHECK:   [[V1:%.+]]:2 = scf.if
//       CHECK:     [[ALLOC1:%.+]] = memref.alloc()
//       CHECK:     scf.yield [[ALLOC1]], %true
//       CHECK:     scf.yield [[ALLOC]], %false
//       CHECK:   [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[ARG5]]
//       CHECK:   [[OWN:%.+]] = bufferization.dealloc ([[BASE]] :{{.*}}) if ([[ARG6]]) retain ([[V1]]#0 :
//       CHECK:   [[OWN_AGG:%.+]] = arith.ori [[OWN]], [[V1]]#1
//       CHECK:   scf.yield [[V1]]#0, [[OWN_AGG]]
//       CHECK: }
//       CHECK: [[V2:%.+]] = scf.if [[V0]]#1
//       CHECK:   scf.yield [[V0]]#0
//       CHECK:   [[CLONE:%.+]] = bufferization.clone [[V0]]#0
//       CHECK:   scf.yield [[CLONE]]
//       CHECK: [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#0
//       CHECK: bufferization.dealloc ([[ALLOC]], [[BASE]] :{{.*}}) if (%true{{[0-9_]*}}, [[V0]]#1) retain ([[V2]] :
//       CHECK: return [[V2]]

// -----

// Test Case: several nested structured control-flow loops with a deeply nested
// buffer allocation inside an if operation.

func.func @loop_nested_alloc(
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>,
  %res: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  "test.read_buffer"(%0) : (memref<2xf32>) -> ()
  %1 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %2 = scf.for %i2 = %lb to %ub step %step
      iter_args(%iterBuf2 = %iterBuf) -> memref<2xf32> {
      %3 = scf.for %i3 = %lb to %ub step %step
        iter_args(%iterBuf3 = %iterBuf2) -> memref<2xf32> {
        %4 = memref.alloc() : memref<2xf32>
        "test.read_buffer"(%4) : (memref<2xf32>) -> ()
        %5 = arith.cmpi eq, %i, %ub : index
        %6 = scf.if %5 -> (memref<2xf32>) {
          %7 = memref.alloc() : memref<2xf32>
          scf.yield %7 : memref<2xf32>
        } else {
          scf.yield %iterBuf3 : memref<2xf32>
        }
        scf.yield %6 : memref<2xf32>
      }
      scf.yield %3 : memref<2xf32>
    }
    scf.yield %2 : memref<2xf32>
  }
  test.copy(%1, %res) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func @loop_nested_alloc
//       CHECK: ({{.*}}, [[ARG3:%.+]]: memref<2xf32>, {{.*}}: memref<2xf32>)
//       CHECK: [[ALLOC:%.+]] = memref.alloc()
//       CHECK: [[V0:%.+]]:2 = scf.for {{.*}} iter_args([[ARG6:%.+]] = [[ARG3]], [[ARG7:%.+]] = %false
//       CHECK:   [[V1:%.+]]:2 = scf.for {{.*}} iter_args([[ARG9:%.+]] = [[ARG6]], [[ARG10:%.+]] = %false
//       CHECK:     [[V2:%.+]]:2 = scf.for {{.*}} iter_args([[ARG12:%.+]] = [[ARG9]], [[ARG13:%.+]] = %false
//       CHECK:       [[ALLOC1:%.+]] = memref.alloc()
//       CHECK:       [[V3:%.+]]:2 = scf.if
//       CHECK:         [[ALLOC2:%.+]] = memref.alloc()
//       CHECK:         scf.yield [[ALLOC2]], %true
//       CHECK:       } else {
//       CHECK:         scf.yield [[ARG12]], %false
//       CHECK:       }
//       CHECK:       [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[ARG12]]
//       CHECK:       [[OWN:%.+]] = bufferization.dealloc ([[BASE]] :{{.*}}) if ([[ARG13]]) retain ([[V3]]#0 :
//       CHECK:       bufferization.dealloc ([[ALLOC1]] :{{.*}}) if (%true{{[0-9_]*}})
//   CHECK-NOT: retain
//       CHECK:       [[OWN_AGG:%.+]] = arith.ori [[OWN]], [[V3]]#1
//       CHECK:       scf.yield [[V3]]#0, [[OWN_AGG]]
//       CHECK:     }
//       CHECK:     [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[ARG9]]
//       CHECK:     [[OWN:%.+]] = bufferization.dealloc ([[BASE]] :{{.*}}) if ([[ARG10]]) retain ([[V2]]#0 :
//       CHECK:     [[OWN_AGG:%.+]] = arith.ori [[OWN]], [[V2]]#1
//       CHECK:     scf.yield [[V2]]#0, [[OWN_AGG]]
//       CHECK:   }
//       CHECK:   [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[ARG6]]
//       CHECK:   [[OWN:%.+]] = bufferization.dealloc ([[BASE]] :{{.*}}) if ([[ARG7]]) retain ([[V1]]#0 :
//       CHECK:   [[OWN_AGG:%.+]] = arith.ori [[OWN]], [[V1]]#1
//       CHECK:   scf.yield [[V1]]#0, [[OWN_AGG]]
//       CHECK: }
//       CHECK: test.copy
//       CHECK: [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#0
//       CHECK: bufferization.dealloc ([[ALLOC]] :{{.*}}) if (%true
//       CHECK: bufferization.dealloc ([[BASE]] :{{.*}}) if ([[V0]]#1)

// TODO: all the retain operands could be removed by doing some more thorough analysis

// -----

func.func @affine_loop() -> f32 {
  %buffer = memref.alloc() : memref<1024xf32>
  %sum_init_0 = arith.constant 0.0 : f32
  %res = affine.for %i = 0 to 10 step 2 iter_args(%sum_iter = %sum_init_0) -> f32 {
    %t = affine.load %buffer[%i] : memref<1024xf32>
    %sum_next = arith.addf %sum_iter, %t : f32
    affine.yield %sum_next : f32
  }
  return %res : f32
}

// CHECK-LABEL: func @affine_loop
//       CHECK: [[ALLOC:%.+]] = memref.alloc()
//       CHECK: affine.for {{.*}} iter_args(%arg1 = %cst)
//       CHECK:   affine.yield
//       CHECK: bufferization.dealloc ([[ALLOC]] :{{.*}}) if (%true

// -----

func.func @assumingOp(
  %arg0: !shape.witness,
  %arg2: memref<2xf32>,
  %arg3: memref<2xf32>) {
  // Confirm the alloc will be dealloc'ed in the block.
  %1 = shape.assuming %arg0 -> memref<2xf32> {
    %0 = memref.alloc() : memref<2xf32>
    "test.read_buffer"(%0) : (memref<2xf32>) -> ()
    shape.assuming_yield %arg2 : memref<2xf32>
  }
  // Confirm the alloc will be returned and dealloc'ed after its use.
  %3 = shape.assuming %arg0 -> memref<2xf32> {
    %2 = memref.alloc() : memref<2xf32>
    shape.assuming_yield %2 : memref<2xf32>
  }
  test.copy(%3, %arg3) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func @assumingOp
//       CHECK: ({{.*}}, [[ARG1:%.+]]: memref<2xf32>, {{.*}}: memref<2xf32>)
//       CHECK: [[V0:%.+]]:2 = shape.assuming
//       CHECK:   [[ALLOC:%.+]] = memref.alloc()
//       CHECK:   bufferization.dealloc ([[ALLOC]] :{{.*}}) if (%true{{[0-9_]*}})
//   CHECK-NOT: retain
//       CHECK:   shape.assuming_yield [[ARG1]], %false
//       CHECK: }
//       CHECK: [[V1:%.+]]:2 = shape.assuming
//       CHECK:   [[ALLOC:%.+]] = memref.alloc()
//       CHECK:   shape.assuming_yield [[ALLOC]], %true
//       CHECK: }
//       CHECK: test.copy
//       CHECK: [[BASE0:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#0
//       CHECK: [[BASE1:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V1]]#0
//       CHECK: bufferization.dealloc ([[BASE1]] :{{.*}}) if ([[V1]]#1)
//   CHECK-NOT: retain
//       CHECK: bufferization.dealloc ([[BASE0]] :{{.*}}) if ([[V0]]#1)
//   CHECK-NOT: retain
//       CHECK: return

// -----

// Test Case: The op "test.one_region_with_recursive_memory_effects" does not
// implement the RegionBranchOpInterface. This is allowed during buffer
// deallocation because the operation's region does not deal with any MemRef
// values.

func.func @noRegionBranchOpInterface() {
  %0 = "test.one_region_with_recursive_memory_effects"() ({
    %1 = "test.one_region_with_recursive_memory_effects"() ({
      %2 = memref.alloc() : memref<2xi32>
      "test.read_buffer"(%2) : (memref<2xi32>) -> ()
      "test.return"() : () -> ()
    }) : () -> (i32)
    "test.return"() : () -> ()
  }) : () -> (i32)
  "test.return"() : () -> ()
}

// -----

// Test Case: The second op "test.one_region_with_recursive_memory_effects" does
// not implement the RegionBranchOpInterface but has buffer semantics. This is
// not allowed during buffer deallocation.

func.func @noRegionBranchOpInterface() {
  %0 = "test.one_region_with_recursive_memory_effects"() ({
    // expected-error@+1 {{All operations with attached regions need to implement the RegionBranchOpInterface.}}
    %1 = "test.one_region_with_recursive_memory_effects"() ({
      %2 = memref.alloc() : memref<2xi32>
      "test.read_buffer"(%2) : (memref<2xi32>) -> ()
      "test.return"(%2) : (memref<2xi32>) -> ()
    }) : () -> (memref<2xi32>)
    "test.return"() : () -> ()
  }) : () -> (i32)
  "test.return"() : () -> ()
}

// -----

func.func @while_two_arg(%arg0: index) {
  %a = memref.alloc(%arg0) : memref<?xf32>
  scf.while (%arg1 = %a, %arg2 = %a) : (memref<?xf32>, memref<?xf32>) -> (memref<?xf32>, memref<?xf32>) {
    // This op has a side effect, but it's not an allocate/free side effect.
    %0 = "test.side_effect_op"() {effects = [{effect="read"}]} : () -> i1
    scf.condition(%0) %arg1, %arg2 : memref<?xf32>, memref<?xf32>
  } do {
  ^bb0(%arg1: memref<?xf32>, %arg2: memref<?xf32>):
    %b = memref.alloc(%arg0) : memref<?xf32>
    scf.yield %arg1, %b : memref<?xf32>, memref<?xf32>
  }
  return
}

// CHECK-LABEL: func @while_two_arg
//       CHECK: [[ALLOC:%.+]] = memref.alloc(
//       CHECK: [[V0:%.+]]:4 = scf.while ({{.*}} = [[ALLOC]], {{.*}} = [[ALLOC]], {{.*}} = %false{{[0-9_]*}}, {{.*}} = %false{{[0-9_]*}})
//       CHECK:   scf.condition
//       CHECK: ^bb0([[ARG1:%.+]]: memref<?xf32>, [[ARG2:%.+]]: memref<?xf32>, [[ARG3:%.+]]: i1, [[ARG4:%.+]]: i1):
//       CHECK:   [[ALLOC1:%.+]] = memref.alloc(
//       CHECK:   [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[ARG2]]
//       CHECK:   [[OWN:%.+]] = bufferization.dealloc ([[BASE]] :{{.*}}) if ([[ARG4]]) retain ([[ARG1]] :
//       CHECK:   [[OWN_AGG:%.+]] = arith.ori [[OWN]], [[ARG3]]
//       CHECK:   scf.yield [[ARG1]], [[ALLOC1]], [[OWN_AGG]], %true
//       CHECK: [[BASE0:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#0
//       CHECK: [[BASE1:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#1
//       CHECK: bufferization.dealloc ([[ALLOC]], [[BASE0]], [[BASE1]] :{{.*}}) if (%true{{[0-9_]*}}, [[V0]]#2, [[V0]]#3)

// -----

func.func @while_three_arg(%arg0: index) {
  %a = memref.alloc(%arg0) : memref<?xf32>
  scf.while (%arg1 = %a, %arg2 = %a, %arg3 = %a) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> (memref<?xf32>, memref<?xf32>, memref<?xf32>) {
    // This op has a side effect, but it's not an allocate/free side effect.
    %0 = "test.side_effect_op"() {effects = [{effect="read"}]} : () -> i1
    scf.condition(%0) %arg1, %arg2, %arg3 : memref<?xf32>, memref<?xf32>, memref<?xf32>
  } do {
  ^bb0(%arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>):
    %b = memref.alloc(%arg0) : memref<?xf32>
    %q = memref.alloc(%arg0) : memref<?xf32>
    scf.yield %q, %b, %arg2: memref<?xf32>, memref<?xf32>, memref<?xf32>
  }
  return
}

// CHECK-LABEL: func @while_three_arg
//       CHECK: [[ALLOC:%.+]] = memref.alloc(
//       CHECK: [[V0:%.+]]:6 = scf.while ({{.*}} = [[ALLOC]], {{.*}} = [[ALLOC]], {{.*}} = [[ALLOC]], {{.*}} = %false{{[0-9_]*}}, {{.*}} = %false{{[0-9_]*}}, {{.*}} = %false
//       CHECK:   scf.condition
//       CHECK: ^bb0([[ARG1:%.+]]: memref<?xf32>, [[ARG2:%.+]]: memref<?xf32>, [[ARG3:%.+]]: memref<?xf32>, [[ARG4:%.+]]: i1, [[ARG5:%.+]]: i1, [[ARG6:%.+]]: i1):
//       CHECK:   [[ALLOC1:%.+]] = memref.alloc(
//       CHECK:   [[ALLOC2:%.+]] = memref.alloc(
//       CHECK:   [[BASE0:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[ARG1]]
//       CHECK:   [[BASE2:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[ARG3]]
//       CHECK:   [[OWN:%.+]] = bufferization.dealloc ([[BASE0]], [[BASE2]] :{{.*}}) if ([[ARG4]], [[ARG6]]) retain ([[ARG2]] :
//       CHECK:   [[OWN_AGG:%.+]] = arith.ori [[OWN]], [[ARG5]]
//       CHECK:   scf.yield [[ALLOC2]], [[ALLOC1]], [[ARG2]], %true{{[0-9_]*}}, %true{{[0-9_]*}}, [[OWN_AGG]] :
//       CHECK: }
//       CHECK: [[BASE0:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#0
//       CHECK: [[BASE1:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#1
//       CHECK: [[BASE2:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#2
//       CHECK: bufferization.dealloc ([[ALLOC]], [[BASE0]], [[BASE1]], [[BASE2]] :{{.*}}) if (%true{{[0-9_]*}}, [[V0]]#3, [[V0]]#4, [[V0]]#5)

// TODO: better alias analysis could simplify the dealloc inside the body further

// -----

// Memref allocated in `then` region and passed back to the parent if op.
#set = affine_set<() : (0 >= 0)>
func.func @test_affine_if_1(%arg0: memref<10xf32>) -> memref<10xf32> {
  %0 = affine.if #set() -> memref<10xf32> {
    %alloc = memref.alloc() : memref<10xf32>
    affine.yield %alloc : memref<10xf32>
  } else {
    affine.yield %arg0 : memref<10xf32>
  }
  return %0 : memref<10xf32>
}

// CHECK-LABEL: func @test_affine_if_1
//  CHECK-SAME: ([[ARG0:%.*]]: memref<10xf32>)
//       CHECK: [[V0:%.+]]:2 = affine.if
//       CHECK:   [[ALLOC:%.+]] = memref.alloc()
//       CHECK:   affine.yield [[ALLOC]], %true
//       CHECK:   affine.yield [[ARG0]], %false
//       CHECK: [[V1:%.+]] = scf.if [[V0]]#1
//       CHECK:   scf.yield [[V0]]#0
//       CHECK:   [[CLONE:%.+]] = bufferization.clone [[V0]]#0
//       CHECK:   scf.yield [[CLONE]]
//       CHECK: [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#0
//       CHECK: bufferization.dealloc ([[BASE]] :{{.*}}) if ([[V0]]#1) retain ([[V1]] :
//       CHECK: return [[V1]]

// TODO: the dealloc could be optimized away since the memref to be deallocated
//       either aliases with V1 or the condition is false

// -----

// Memref allocated before parent IfOp and used in `then` region.
// Expected result: deallocation should happen after affine.if op.
#set = affine_set<() : (0 >= 0)>
func.func @test_affine_if_2() -> memref<10xf32> {
  %alloc0 = memref.alloc() : memref<10xf32>
  %0 = affine.if #set() -> memref<10xf32> {
    affine.yield %alloc0 : memref<10xf32>
  } else {
    %alloc = memref.alloc() : memref<10xf32>
    affine.yield %alloc : memref<10xf32>
  }
  return %0 : memref<10xf32>
}
// CHECK-LABEL: func @test_affine_if_2
//       CHECK: [[ALLOC:%.+]] = memref.alloc()
//       CHECK: [[V0:%.+]]:2 = affine.if
//       CHECK:   affine.yield [[ALLOC]], %false
//       CHECK:   [[ALLOC1:%.+]] = memref.alloc()
//       CHECK:   affine.yield [[ALLOC1]], %true
//       CHECK: [[V1:%.+]] = scf.if [[V0]]#1
//       CHECK:   scf.yield [[V0]]#0
//       CHECK:   [[CLONE:%.+]] = bufferization.clone [[V0]]#0
//       CHECK:   scf.yield [[CLONE]]
//       CHECK: [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#0
//       CHECK: bufferization.dealloc ([[ALLOC]], [[BASE]] :{{.*}}) if (%true{{[0-9_]*}}, [[V0]]#1) retain ([[V1]] :
//       CHECK: return [[V1]]

// -----

// Memref allocated before parent IfOp and used in `else` region.
// Expected result: deallocation should happen after affine.if op.
#set = affine_set<() : (0 >= 0)>
func.func @test_affine_if_3() -> memref<10xf32> {
  %alloc0 = memref.alloc() : memref<10xf32>
  %0 = affine.if #set() -> memref<10xf32> {
    %alloc = memref.alloc() : memref<10xf32>
    affine.yield %alloc : memref<10xf32>
  } else {
    affine.yield %alloc0 : memref<10xf32>
  }
  return %0 : memref<10xf32>
}

// CHECK-LABEL: func @test_affine_if_3
//       CHECK: [[ALLOC:%.+]] = memref.alloc()
//       CHECK: [[V0:%.+]]:2 = affine.if
//       CHECK:   [[ALLOC1:%.+]] = memref.alloc()
//       CHECK:   affine.yield [[ALLOC1]], %true
//       CHECK:   affine.yield [[ALLOC]], %false
//       CHECK: [[V1:%.+]] = scf.if [[V0]]#1
//       CHECK:   scf.yield [[V0]]#0
//       CHECK:   [[CLONE:%.+]] = bufferization.clone [[V0]]#0
//       CHECK:   scf.yield [[CLONE]]
//       CHECK: [[BASE:%[a-zA-Z0-9_]+]],{{.*}} = memref.extract_strided_metadata [[V0]]#0
//       CHECK: bufferization.dealloc ([[ALLOC]], [[BASE]] :{{.*}}) if (%true{{[0-9_]*}}, [[V0]]#1) retain ([[V1]]
//       CHECK: return [[V1]]
