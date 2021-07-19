// RUN: mlir-opt %s -test-scf-pipelining -split-input-file | FileCheck %s

// CHECK-LABEL: simple_pipeline(
//  CHECK-SAME:   %[[A:.*]]: memref<?xf32>, %[[R:.*]]: memref<?xf32>) {
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = constant 1 : index
//   CHECK-DAG:   %[[C3:.*]] = constant 3 : index
// Prologue:
//       CHECK:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
// Kernel:
//  CHECK-NEXT:   %[[L1:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C3]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[LARG:.*]] = %[[L0]]) -> (f32) {
//  CHECK-NEXT:     %[[ADD0:.*]] = addf %[[LARG]], %{{.*}} : f32
//  CHECK-NEXT:     memref.store %[[ADD0]], %[[R]][%[[IV]]] : memref<?xf32>
//  CHECK-NEXT:     %[[IV1:.*]] = addi %[[IV]], %[[C1]] : index
//  CHECK-NEXT:     %[[LR:.*]] = memref.load %[[A]][%[[IV1]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[LR]] : f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:   %[[ADD1:.*]] = addf %[[L1]], %{{.*}} : f32
//  CHECK-NEXT:   memref.store %[[ADD1]], %[[R]][%[[C3]]] : memref<?xf32>
func @simple_pipeline(%A: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %cf = constant 1.0 : f32
  scf.for %i0 = %c0 to %c4 step %c1 {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 2 } : memref<?xf32>
    %A1_elem = addf %A_elem, %cf { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 0 } : f32
    memref.store %A1_elem, %result[%i0] { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 1 } : memref<?xf32>
  }  { __test_pipelining_loop__ }
  return
}

// -----

// CHECK-LABEL: three_stage(
//  CHECK-SAME:   %[[A:.*]]: memref<?xf32>, %[[R:.*]]: memref<?xf32>) {
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = constant 3 : index
// Prologue:
//       CHECK:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
//  CHECK-NEXT:   %[[ADD0:.*]] = addf %[[L0]], %{{.*}} : f32
//  CHECK-NEXT:   %[[L1:.*]] = memref.load %[[A]][%[[C1]]] : memref<?xf32>
// Kernel:
//  CHECK-NEXT:   %[[LR:.*]]:2 = scf.for %[[IV:.*]] = %[[C0]] to %[[C2]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[ADDARG:.*]] = %[[ADD0]],
//  CHECK-SAME:     %[[LARG:.*]] = %[[L1]]) -> (f32, f32) {
//  CHECK-NEXT:     memref.store %[[ADDARG]], %[[R]][%[[IV]]] : memref<?xf32>
//  CHECK-NEXT:     %[[ADD1:.*]] = addf %[[LARG]], %{{.*}} : f32
//  CHECK-NEXT:     %[[IV2:.*]] = addi %[[IV]], %[[C2]] : index
//  CHECK-NEXT:     %[[L3:.*]] = memref.load %[[A]][%[[IV2]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[ADD1]], %[[L3]] : f32, f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:   memref.store %[[LR]]#0, %[[R]][%[[C2]]] : memref<?xf32>
//  CHECK-NEXT:   %[[ADD2:.*]] = addf %[[LR]]#1, %{{.*}} : f32
//  CHECK-NEXT:   memref.store %[[ADD2]], %[[R]][%[[C3]]] : memref<?xf32>
func @three_stage(%A: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %cf = constant 1.0 : f32
  scf.for %i0 = %c0 to %c4 step %c1 {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 2 } : memref<?xf32>
    %A1_elem = addf %A_elem, %cf { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 1 } : f32
    memref.store %A1_elem, %result[%i0] { __test_pipelining_stage__ = 2, __test_pipelining_op_order__ = 0 } : memref<?xf32>
  } { __test_pipelining_loop__ }
  return
}

// -----
// CHECK-LABEL: long_liverange(
//  CHECK-SAME:   %[[A:.*]]: memref<?xf32>, %[[R:.*]]: memref<?xf32>) {
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = constant 3 : index
//   CHECK-DAG:   %[[C4:.*]] = constant 4 : index
//   CHECK-DAG:   %[[C6:.*]] = constant 6 : index
//   CHECK-DAG:   %[[C7:.*]] = constant 7 : index
//   CHECK-DAG:   %[[C8:.*]] = constant 8 : index
//   CHECK-DAG:   %[[C9:.*]] = constant 9 : index
// Prologue:
//       CHECK:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
//  CHECK-NEXT:   %[[L1:.*]] = memref.load %[[A]][%[[C1]]] : memref<?xf32>
//  CHECK-NEXT:   %[[L2:.*]] = memref.load %[[A]][%[[C2]]] : memref<?xf32>
//  CHECK-NEXT:   %[[L3:.*]] = memref.load %[[A]][%[[C3]]] : memref<?xf32>
// Kernel:
//  CHECK-NEXT:   %[[LR:.*]]:4 = scf.for %[[IV:.*]] = %[[C0]] to %[[C6]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[LA0:.*]] = %[[L0]],
//  CHECK-SAME:     %[[LA1:.*]] = %[[L1]], %[[LA2:.*]] = %[[L2]],
//  CHECK-SAME:     %[[LA3:.*]] = %[[L3]]) -> (f32, f32, f32, f32) {
//  CHECK-NEXT:     %[[ADD0:.*]] = addf %[[LA0]], %{{.*}} : f32
//  CHECK-NEXT:     memref.store %[[ADD0]], %[[R]][%[[IV]]] : memref<?xf32>
//  CHECK-NEXT:     %[[IV4:.*]] = addi %[[IV]], %[[C4]] : index
//  CHECK-NEXT:     %[[L4:.*]] = memref.load %[[A]][%[[IV4]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[LA1]], %[[LA2]], %[[LA3]], %[[L4]] : f32, f32, f32, f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:  %[[ADD1:.*]] = addf %[[LR]]#0, %{{.*}} : f32
//  CHECK-NEXT:  memref.store %[[ADD1]], %[[R]][%[[C6]]] : memref<?xf32>
//  CHECK-NEXT:  %[[ADD2:.*]] = addf %[[LR]]#1, %{{.*}} : f32
//  CHECK-NEXT:  memref.store %[[ADD2]], %[[R]][%[[C7]]] : memref<?xf32>
//  CHECK-NEXT:  %[[ADD3:.*]] = addf %[[LR]]#2, %{{.*}} : f32
//  CHECK-NEXT:  memref.store %[[ADD3]], %[[R]][%[[C8]]] : memref<?xf32>
//  CHECK-NEXT:  %[[ADD4:.*]] = addf %[[LR]]#3, %{{.*}} : f32
//  CHECK-NEXT:  memref.store %[[ADD4]], %[[R]][%[[C9]]] : memref<?xf32>
func @long_liverange(%A: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %cf = constant 1.0 : f32
  scf.for %i0 = %c0 to %c10 step %c1 {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 2 } : memref<?xf32>
    %A1_elem = addf %A_elem, %cf { __test_pipelining_stage__ = 4, __test_pipelining_op_order__ = 0 } : f32
    memref.store %A1_elem, %result[%i0] { __test_pipelining_stage__ = 4, __test_pipelining_op_order__ = 1 } : memref<?xf32>
  } { __test_pipelining_loop__ }
  return
}

// -----

// CHECK-LABEL: multiple_uses(
//  CHECK-SAME:   %[[A:.*]]: memref<?xf32>, %[[R:.*]]: memref<?xf32>) {
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = constant 3 : index
//   CHECK-DAG:   %[[C7:.*]] = constant 7 : index
//   CHECK-DAG:   %[[C8:.*]] = constant 8 : index
//   CHECK-DAG:   %[[C9:.*]] = constant 9 : index
// Prologue:
//       CHECK:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
//  CHECK-NEXT:   %[[ADD0:.*]] = addf %[[L0]], %{{.*}} : f32
//  CHECK-NEXT:   %[[L1:.*]] = memref.load %[[A]][%[[C1]]] : memref<?xf32>
//  CHECK-NEXT:   %[[ADD1:.*]] = addf %[[L1]], %{{.*}} : f32
//  CHECK-NEXT:   %[[MUL0:.*]] = mulf %[[ADD0]], %[[L0]] : f32
//  CHECK-NEXT:   %[[L2:.*]] = memref.load %[[A]][%[[C2]]] : memref<?xf32>
// Kernel:
//  CHECK-NEXT:   %[[LR:.*]]:4 = scf.for %[[IV:.*]] = %[[C0]] to %[[C7]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[LA1:.*]] = %[[L1]],
//  CHECK-SAME:     %[[LA2:.*]] = %[[L2]], %[[ADDARG1:.*]] = %[[ADD1]],
//  CHECK-SAME:     %[[MULARG0:.*]] = %[[MUL0]]) -> (f32, f32, f32, f32) {
//  CHECK-NEXT:     %[[ADD2:.*]] = addf %[[LA2]], %{{.*}} : f32
//  CHECK-NEXT:     %[[MUL1:.*]] = mulf %[[ADDARG1]], %[[LA1]] : f32
//  CHECK-NEXT:     memref.store %[[MULARG0]], %[[R]][%[[IV]]] : memref<?xf32>
//  CHECK-NEXT:     %[[IV3:.*]] = addi %[[IV]], %[[C3]] : index
//  CHECK-NEXT:     %[[L3:.*]] = memref.load %[[A]][%[[IV3]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[LA2]], %[[L3]], %[[ADD2]], %[[MUL1]] : f32, f32, f32, f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:   %[[ADD3:.*]] = addf %[[LR]]#1, %{{.*}} : f32
//  CHECK-NEXT:   %[[MUL2:.*]] = mulf %[[LR]]#2, %[[LR]]#0 : f32
//  CHECK-NEXT:   memref.store %[[LR]]#3, %[[R]][%[[C7]]] : memref<?xf32>
//  CHECK-NEXT:   %[[MUL3:.*]] = mulf %[[ADD3]], %[[LR]]#1 : f32
//  CHECK-NEXT:   memref.store %[[MUL2]], %[[R]][%[[C8]]] : memref<?xf32>
//  CHECK-NEXT:   memref.store %[[MUL3]], %[[R]][%[[C9]]] : memref<?xf32>
func @multiple_uses(%A: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %cf = constant 1.0 : f32
  scf.for %i0 = %c0 to %c10 step %c1 {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 3 } : memref<?xf32>
    %A1_elem = addf %A_elem, %cf { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 0 } : f32
    %A2_elem = mulf %A1_elem, %A_elem { __test_pipelining_stage__ = 2, __test_pipelining_op_order__ = 1 } : f32
    memref.store %A2_elem, %result[%i0] { __test_pipelining_stage__ = 3, __test_pipelining_op_order__ = 2 } : memref<?xf32>
  } { __test_pipelining_loop__ }
  return
}

// -----

// CHECK-LABEL: loop_carried(
//  CHECK-SAME:   %[[A:.*]]: memref<?xf32>, %[[R:.*]]: memref<?xf32>) {
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = constant 1 : index
//   CHECK-DAG:   %[[C3:.*]] = constant 3 : index
//   CHECK-DAG:   %[[CSTF:.*]] = constant 1.000000e+00 : f32
// Prologue:
//       CHECK:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
// Kernel:
//  CHECK-NEXT:   %[[LR:.*]]:2 = scf.for %[[IV:.*]] = %[[C0]] to %[[C3]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[C:.*]] = %[[CSTF]],
//  CHECK-SAME:     %[[LARG:.*]] = %[[L0]]) -> (f32, f32) {
//  CHECK-NEXT:     %[[ADD0:.*]] = addf %[[LARG]], %[[C]] : f32
//  CHECK-NEXT:     %[[IV1:.*]] = addi %[[IV]], %[[C1]] : index
//  CHECK-NEXT:     %[[L1:.*]] = memref.load %[[A]][%[[IV1]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[ADD0]], %[[L1]] : f32, f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:   %[[ADD1:.*]] = addf %[[LR]]#1, %[[LR]]#0 : f32
//  CHECK-NEXT:   memref.store %[[ADD1]], %[[R]][%[[C0]]] : memref<?xf32>
func @loop_carried(%A: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %cf = constant 1.0 : f32
  %r = scf.for %i0 = %c0 to %c4 step %c1 iter_args(%arg0 = %cf) -> (f32) {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 1 } : memref<?xf32>
    %A1_elem = addf %A_elem, %arg0 { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 0 } : f32
    scf.yield %A1_elem : f32
  }  { __test_pipelining_loop__ }
  memref.store %r, %result[%c0] : memref<?xf32>
  return
}

// -----

// CHECK-LABEL: backedge_different_stage
//  CHECK-SAME:   (%[[A:.*]]: memref<?xf32>) -> f32 {
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//   CHECK-DAG:   %[[CSTF:.*]] = constant 1.000000e+00 : f32
// Prologue:
//       CHECK:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
//  CHECK-NEXT:   %[[ADD0:.*]] = addf %[[L0]], %[[CSTF]] : f32
//  CHECK-NEXT:   %[[L1:.*]] = memref.load %[[A]][%[[C1]]] : memref<?xf32>
// Kernel:
//  CHECK-NEXT:   %[[R:.*]]:3 = scf.for %[[IV:.*]] = %[[C0]] to %[[C2]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[C:.*]] = %[[CSTF]],
//  CHECK-SAME:     %[[ADDARG:.*]] = %[[ADD0]], %[[LARG:.*]] = %[[L1]]) -> (f32, f32, f32) {
//  CHECK-NEXT:     %[[MUL0:.*]] = mulf %[[CSTF]], %[[ADDARG]] : f32
//  CHECK-NEXT:     %[[ADD1:.*]] = addf %[[LARG]], %[[MUL0]] : f32
//  CHECK-NEXT:     %[[IV2:.*]] = addi %[[IV]], %[[C2]] : index
//  CHECK-NEXT:     %[[L2:.*]] = memref.load %[[A]][%[[IV2]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[MUL0]], %[[ADD1]], %[[L2]] : f32, f32, f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:   %[[MUL1:.*]] = mulf %[[CSTF]], %[[R]]#1 : f32
//  CHECK-NEXT:   %[[ADD2:.*]] = addf %[[R]]#2, %[[MUL1]] : f32
//  CHECK-NEXT:   %[[MUL2:.*]] = mulf %[[CSTF]], %[[ADD2]] : f32
//  CHECK-NEXT:   return %[[MUL2]] : f32
func @backedge_different_stage(%A: memref<?xf32>) -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %cf = constant 1.0 : f32
  %r = scf.for %i0 = %c0 to %c4 step %c1 iter_args(%arg0 = %cf) -> (f32) {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 2 } : memref<?xf32>
    %A1_elem = addf %A_elem, %arg0 { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 1 } : f32
    %A2_elem = mulf %cf, %A1_elem { __test_pipelining_stage__ = 2, __test_pipelining_op_order__ = 0 } : f32
    scf.yield %A2_elem : f32
  }  { __test_pipelining_loop__ }
  return %r : f32
}

// -----

// CHECK-LABEL: backedge_same_stage
//  CHECK-SAME:   (%[[A:.*]]: memref<?xf32>) -> f32 {
//   CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = constant 1 : index
//   CHECK-DAG:   %[[C3:.*]] = constant 3 : index
//   CHECK-DAG:   %[[CSTF:.*]] = constant 1.000000e+00 : f32
// Prologue:
//       CHECK:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
// Kernel:
//  CHECK-NEXT:   %[[R:.*]]:2 = scf.for %[[IV:.*]] = %[[C0]] to %[[C3]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[C:.*]] = %[[CSTF]],
//  CHECK-SAME:     %[[LARG:.*]] = %[[L0]]) -> (f32, f32) {
//  CHECK-NEXT:     %[[ADD0:.*]] = addf %[[LARG]], %[[C]] : f32
//  CHECK-NEXT:     %[[MUL0:.*]] = mulf %[[CSTF]], %[[ADD0]] : f32
//  CHECK-NEXT:     %[[IV1:.*]] = addi %[[IV]], %[[C1]] : index
//  CHECK-NEXT:     %[[L2:.*]] = memref.load %[[A]][%[[IV1]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[MUL0]], %[[L2]] : f32, f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:   %[[ADD1:.*]] = addf %[[R]]#1, %[[R]]#0 : f32
//  CHECK-NEXT:   %[[MUL1:.*]] = mulf %[[CSTF]], %[[ADD1]] : f32
//  CHECK-NEXT:   return %[[MUL1]] : f32
func @backedge_same_stage(%A: memref<?xf32>) -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %cf = constant 1.0 : f32
  %r = scf.for %i0 = %c0 to %c4 step %c1 iter_args(%arg0 = %cf) -> (f32) {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 2 } : memref<?xf32>
    %A1_elem = addf %A_elem, %arg0 { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 0 } : f32
    %A2_elem = mulf %cf, %A1_elem { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 1 } : f32
    scf.yield %A2_elem : f32
  }  { __test_pipelining_loop__ }
  return %r : f32
}
