// RUN: mlir-opt %s -test-scf-pipelining -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-scf-pipelining=annotate -split-input-file | FileCheck %s --check-prefix ANNOTATE
// RUN: mlir-opt %s -test-scf-pipelining=no-epilogue-peeling -split-input-file | FileCheck %s --check-prefix NOEPILOGUE

// CHECK-LABEL: simple_pipeline(
//  CHECK-SAME:   %[[A:.*]]: memref<?xf32>, %[[R:.*]]: memref<?xf32>) {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// Prologue:
//       CHECK:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
// Kernel:
//  CHECK-NEXT:   %[[L1:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C3]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[LARG:.*]] = %[[L0]]) -> (f32) {
//  CHECK-NEXT:     %[[ADD0:.*]] = arith.addf %[[LARG]], %{{.*}} : f32
//  CHECK-NEXT:     memref.store %[[ADD0]], %[[R]][%[[IV]]] : memref<?xf32>
//  CHECK-NEXT:     %[[IV1:.*]] = arith.addi %[[IV]], %[[C1]] : index
//  CHECK-NEXT:     %[[LR:.*]] = memref.load %[[A]][%[[IV1]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[LR]] : f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:   %[[ADD1:.*]] = arith.addf %[[L1]], %{{.*}} : f32
//  CHECK-NEXT:   memref.store %[[ADD1]], %[[R]][%[[C3]]] : memref<?xf32>
func.func @simple_pipeline(%A: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cf = arith.constant 1.0 : f32
  scf.for %i0 = %c0 to %c4 step %c1 {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 2 } : memref<?xf32>
    %A1_elem = arith.addf %A_elem, %cf { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 0 } : f32
    memref.store %A1_elem, %result[%i0] { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 1 } : memref<?xf32>
  }  { __test_pipelining_loop__ }
  return
}


// -----

// CHECK-LABEL: simple_pipeline_region(
//  CHECK-SAME:   %[[A:.*]]: memref<?xf32>, %[[R:.*]]: memref<?xf32>) {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// Prologue:
//       CHECK:   %[[L0:.*]] = scf.execute_region
//  CHECK-NEXT:     memref.load %[[A]][%[[C0]]] : memref<?xf32>
// Kernel:
//       CHECK:   %[[L1:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C3]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[LARG:.*]] = %[[L0]]) -> (f32) {
//  CHECK-NEXT:     %[[ADD0:.*]] = scf.execute_region
//  CHECK-NEXT:       arith.addf %[[LARG]], %{{.*}} : f32
//       CHECK:     memref.store %[[ADD0]], %[[R]][%[[IV]]] : memref<?xf32>
//  CHECK-NEXT:     %[[IV1:.*]] = arith.addi %[[IV]], %[[C1]] : index
//  CHECK-NEXT:     %[[LR:.*]] = scf.execute_region
//  CHECK-NEXT:       memref.load %[[A]][%[[IV1]]] : memref<?xf32>
//       CHECK:     scf.yield %[[LR]] : f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:   %[[ADD1:.*]] = scf.execute_region
//  CHECK-NEXT:     arith.addf %[[L1]], %{{.*}} : f32
//       CHECK:   memref.store %[[ADD1]], %[[R]][%[[C3]]] : memref<?xf32>
func.func @simple_pipeline_region(%A: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cf = arith.constant 1.0 : f32
  scf.for %i0 = %c0 to %c4 step %c1 {

    %A_elem = scf.execute_region -> f32 {
      %A_elem1 = memref.load %A[%i0]  : memref<?xf32>
      scf.yield %A_elem1 : f32
    } { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 2 }

    %A1_elem = scf.execute_region -> f32 {
      %A1_elem1 = arith.addf %A_elem, %cf  : f32
      scf.yield %A1_elem1 : f32
    } { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 0 }

    memref.store %A1_elem, %result[%i0] { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 1 } : memref<?xf32>
  }  { __test_pipelining_loop__ }
  return
}

// -----

// CHECK-LABEL: simple_pipeline_step(
//  CHECK-SAME:   %[[A:.*]]: memref<?xf32>, %[[R:.*]]: memref<?xf32>) {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
//   CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
//   CHECK-DAG:   %[[C9:.*]] = arith.constant 9 : index
// Prologue:
//       CHECK:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
//       CHECK:   %[[L1:.*]] = memref.load %[[A]][%[[C3]]] : memref<?xf32>
// Kernel:
//  CHECK-NEXT:   %[[L2:.*]]:2 = scf.for %[[IV:.*]] = %[[C0]] to %[[C5]]
//  CHECK-SAME:     step %[[C3]] iter_args(%[[LARG0:.*]] = %[[L0]], %[[LARG1:.*]] = %[[L1]]) -> (f32, f32) {
//  CHECK-NEXT:     %[[ADD0:.*]] = arith.addf %[[LARG0]], %{{.*}} : f32
//  CHECK-NEXT:     memref.store %[[ADD0]], %[[R]][%[[IV]]] : memref<?xf32>
//  CHECK-NEXT:     %[[IV1:.*]] = arith.addi %[[IV]], %[[C6]] : index
//  CHECK-NEXT:     %[[LR:.*]] = memref.load %[[A]][%[[IV1]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[LARG1]], %[[LR]] : f32, f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:   %[[ADD1:.*]] = arith.addf %[[L2]]#0, %{{.*}} : f32
//  CHECK-NEXT:   memref.store %[[ADD1]], %[[R]][%[[C6]]] : memref<?xf32>
//  CHECK-NEXT:   %[[ADD2:.*]] = arith.addf %[[L2]]#1, %{{.*}} : f32
//  CHECK-NEXT:   memref.store %[[ADD2]], %[[R]][%[[C9]]] : memref<?xf32>
func.func @simple_pipeline_step(%A: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c11 = arith.constant 11 : index
  %cf = arith.constant 1.0 : f32
  scf.for %i0 = %c0 to %c11 step %c3 {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 2 } : memref<?xf32>
    %A1_elem = arith.addf %A_elem, %cf { __test_pipelining_stage__ = 2, __test_pipelining_op_order__ = 0 } : f32
    memref.store %A1_elem, %result[%i0] { __test_pipelining_stage__ = 2, __test_pipelining_op_order__ = 1 } : memref<?xf32>
  }  { __test_pipelining_loop__ }
  return
}

// -----

// CHECK-LABEL: three_stage(
//  CHECK-SAME:   %[[A:.*]]: memref<?xf32>, %[[R:.*]]: memref<?xf32>) {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// Prologue:
//       CHECK:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
//  CHECK-NEXT:   %[[ADD0:.*]] = arith.addf %[[L0]], %{{.*}} : f32
//  CHECK-NEXT:   %[[L1:.*]] = memref.load %[[A]][%[[C1]]] : memref<?xf32>
// Kernel:
//  CHECK-NEXT:   %[[LR:.*]]:2 = scf.for %[[IV:.*]] = %[[C0]] to %[[C2]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[ADDARG:.*]] = %[[ADD0]],
//  CHECK-SAME:     %[[LARG:.*]] = %[[L1]]) -> (f32, f32) {
//  CHECK-NEXT:     memref.store %[[ADDARG]], %[[R]][%[[IV]]] : memref<?xf32>
//  CHECK-NEXT:     %[[ADD1:.*]] = arith.addf %[[LARG]], %{{.*}} : f32
//  CHECK-NEXT:     %[[IV2:.*]] = arith.addi %[[IV]], %[[C2]] : index
//  CHECK-NEXT:     %[[L3:.*]] = memref.load %[[A]][%[[IV2]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[ADD1]], %[[L3]] : f32, f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:   memref.store %[[LR]]#0, %[[R]][%[[C2]]] : memref<?xf32>
//  CHECK-NEXT:   %[[ADD2:.*]] = arith.addf %[[LR]]#1, %{{.*}} : f32
//  CHECK-NEXT:   memref.store %[[ADD2]], %[[R]][%[[C3]]] : memref<?xf32>

// Prologue:
//  ANNOTATE:   memref.load {{.*}} {__test_pipelining_iteration = 0 : i32, __test_pipelining_part = "prologue"}
//  ANNOTATE:   memref.load {{.*}} {__test_pipelining_iteration = 1 : i32, __test_pipelining_part = "prologue"}
// Kernel:
//  ANNOTATE:   scf.for
//  ANNOTATE:     memref.store {{.*}} {__test_pipelining_iteration = 0 : i32, __test_pipelining_part = "kernel"}
//  ANNOTATE:     arith.addf {{.*}} {__test_pipelining_iteration = 0 : i32, __test_pipelining_part = "kernel"}
//  ANNOTATE:     memref.load {{.*}} {__test_pipelining_iteration = 0 : i32, __test_pipelining_part = "kernel"}
//  ANNOTATE:     scf.yield
//  ANNOTATE:   }
// Epilogue:
//  ANNOTATE:   memref.store {{.*}} {__test_pipelining_iteration = 0 : i32, __test_pipelining_part = "epilogue"}
//  ANNOTATE:   arith.addf {{.*}} {__test_pipelining_iteration = 0 : i32, __test_pipelining_part = "epilogue"}
//  ANNOTATE:   memref.store {{.*}} {__test_pipelining_iteration = 1 : i32, __test_pipelining_part = "epilogue"}

// NOEPILOGUE-LABEL: three_stage(
//  NOEPILOGUE-SAME:   %[[A:.*]]: memref<?xf32>, %[[R:.*]]: memref<?xf32>) {
//   NOEPILOGUE-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   NOEPILOGUE-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   NOEPILOGUE-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   NOEPILOGUE-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   NOEPILOGUE-DAG:   %[[C4:.*]] = arith.constant 4 : index
//   NOEPILOGUE-DAG:   %[[CF:.*]] = arith.constant 0.000000e+00 : f32
// Prologue:
//       NOEPILOGUE:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
//  NOEPILOGUE-NEXT:   %[[ADD0:.*]] = arith.addf %[[L0]], %{{.*}} : f32
//  NOEPILOGUE-NEXT:   %[[L1:.*]] = memref.load %[[A]][%[[C1]]] : memref<?xf32>
// Kernel:
//  NOEPILOGUE-NEXT:   %[[LR:.*]]:2 = scf.for %[[IV:.*]] = %[[C0]] to %[[C4]]
//  NOEPILOGUE-SAME:     step %[[C1]] iter_args(%[[ADDARG:.*]] = %[[ADD0]],
//  NOEPILOGUE-SAME:     %[[LARG:.*]] = %[[L1]]) -> (f32, f32) {
//   NOEPILOGUE-DAG:     %[[S0:.*]] = arith.cmpi slt, %[[IV]], %[[C2]] : index
//   NOEPILOGUE-DAG:     %[[S1:.*]] = arith.cmpi slt, %[[IV]], %[[C3]] : index
//  NOEPILOGUE-NEXT:     memref.store %[[ADDARG]], %[[R]][%[[IV]]] : memref<?xf32>
//  NOEPILOGUE-NEXT:     %[[ADD1:.*]] = scf.if %[[S1]] -> (f32) {
//  NOEPILOGUE-NEXT:       %[[PADD:.*]] = arith.addf %[[LARG]], %{{.*}} : f32
//  NOEPILOGUE-NEXT:       scf.yield %[[PADD]] : f32
//  NOEPILOGUE-NEXT:     } else {
//  NOEPILOGUE-NEXT:       scf.yield %[[CF]] : f32
//  NOEPILOGUE-NEXT:     }
//  NOEPILOGUE-NEXT:     %[[IV2:.*]] = arith.addi %[[IV]], %[[C2]] : index
//  NOEPILOGUE-NEXT:     %[[L3:.*]] = scf.if %[[S0]] -> (f32) {
//  NOEPILOGUE-NEXT:       %[[PL:.*]] = memref.load %[[A]][%[[IV2]]] : memref<?xf32>
//  NOEPILOGUE-NEXT:       scf.yield %[[PL]] : f32
//  NOEPILOGUE-NEXT:     } else {
//  NOEPILOGUE-NEXT:       scf.yield %[[CF]] : f32
//  NOEPILOGUE-NEXT:     }
//  NOEPILOGUE-NEXT:     scf.yield %[[ADD1]], %[[L3]] : f32, f32
//  NOEPILOGUE-NEXT:   }
// No epilogue should be generated.
//   NOEPILOGUE-NOT:   memref.store
//       NOEPILOGUE:   return

func.func @three_stage(%A: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cf = arith.constant 1.0 : f32
  scf.for %i0 = %c0 to %c4 step %c1 {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 2 } : memref<?xf32>
    %A1_elem = arith.addf %A_elem, %cf { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 1 } : f32
    memref.store %A1_elem, %result[%i0] { __test_pipelining_stage__ = 2, __test_pipelining_op_order__ = 0 } : memref<?xf32>
  } { __test_pipelining_loop__ }
  return
}

// -----
// CHECK-LABEL: long_liverange(
//  CHECK-SAME:   %[[A:.*]]: memref<?xf32>, %[[R:.*]]: memref<?xf32>) {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
//   CHECK-DAG:   %[[C7:.*]] = arith.constant 7 : index
//   CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
//   CHECK-DAG:   %[[C9:.*]] = arith.constant 9 : index
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
//  CHECK-NEXT:     %[[ADD0:.*]] = arith.addf %[[LA0]], %{{.*}} : f32
//  CHECK-NEXT:     memref.store %[[ADD0]], %[[R]][%[[IV]]] : memref<?xf32>
//  CHECK-NEXT:     %[[IV4:.*]] = arith.addi %[[IV]], %[[C4]] : index
//  CHECK-NEXT:     %[[L4:.*]] = memref.load %[[A]][%[[IV4]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[LA1]], %[[LA2]], %[[LA3]], %[[L4]] : f32, f32, f32, f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:  %[[ADD1:.*]] = arith.addf %[[LR]]#0, %{{.*}} : f32
//  CHECK-NEXT:  memref.store %[[ADD1]], %[[R]][%[[C6]]] : memref<?xf32>
//  CHECK-NEXT:  %[[ADD2:.*]] = arith.addf %[[LR]]#1, %{{.*}} : f32
//  CHECK-NEXT:  memref.store %[[ADD2]], %[[R]][%[[C7]]] : memref<?xf32>
//  CHECK-NEXT:  %[[ADD3:.*]] = arith.addf %[[LR]]#2, %{{.*}} : f32
//  CHECK-NEXT:  memref.store %[[ADD3]], %[[R]][%[[C8]]] : memref<?xf32>
//  CHECK-NEXT:  %[[ADD4:.*]] = arith.addf %[[LR]]#3, %{{.*}} : f32
//  CHECK-NEXT:  memref.store %[[ADD4]], %[[R]][%[[C9]]] : memref<?xf32>
func.func @long_liverange(%A: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cf = arith.constant 1.0 : f32
  scf.for %i0 = %c0 to %c10 step %c1 {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 2 } : memref<?xf32>
    %A1_elem = arith.addf %A_elem, %cf { __test_pipelining_stage__ = 4, __test_pipelining_op_order__ = 0 } : f32
    memref.store %A1_elem, %result[%i0] { __test_pipelining_stage__ = 4, __test_pipelining_op_order__ = 1 } : memref<?xf32>
  } { __test_pipelining_loop__ }
  return
}

// -----

// CHECK-LABEL: multiple_uses(
//  CHECK-SAME:   %[[A:.*]]: memref<?xf32>, %[[R:.*]]: memref<?xf32>) {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C7:.*]] = arith.constant 7 : index
//   CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
//   CHECK-DAG:   %[[C9:.*]] = arith.constant 9 : index
// Prologue:
//       CHECK:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
//  CHECK-NEXT:   %[[ADD0:.*]] = arith.addf %[[L0]], %{{.*}} : f32
//  CHECK-NEXT:   %[[L1:.*]] = memref.load %[[A]][%[[C1]]] : memref<?xf32>
//  CHECK-NEXT:   %[[ADD1:.*]] = arith.addf %[[L1]], %{{.*}} : f32
//  CHECK-NEXT:   %[[MUL0:.*]] = arith.mulf %[[ADD0]], %[[L0]] : f32
//  CHECK-NEXT:   %[[L2:.*]] = memref.load %[[A]][%[[C2]]] : memref<?xf32>
// Kernel:
//  CHECK-NEXT:   %[[LR:.*]]:4 = scf.for %[[IV:.*]] = %[[C0]] to %[[C7]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[LA1:.*]] = %[[L1]],
//  CHECK-SAME:     %[[LA2:.*]] = %[[L2]], %[[ADDARG1:.*]] = %[[ADD1]],
//  CHECK-SAME:     %[[MULARG0:.*]] = %[[MUL0]]) -> (f32, f32, f32, f32) {
//  CHECK-NEXT:     %[[ADD2:.*]] = arith.addf %[[LA2]], %{{.*}} : f32
//  CHECK-NEXT:     %[[MUL1:.*]] = arith.mulf %[[ADDARG1]], %[[LA1]] : f32
//  CHECK-NEXT:     memref.store %[[MULARG0]], %[[R]][%[[IV]]] : memref<?xf32>
//  CHECK-NEXT:     %[[IV3:.*]] = arith.addi %[[IV]], %[[C3]] : index
//  CHECK-NEXT:     %[[L3:.*]] = memref.load %[[A]][%[[IV3]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[LA2]], %[[L3]], %[[ADD2]], %[[MUL1]] : f32, f32, f32, f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:   %[[ADD3:.*]] = arith.addf %[[LR]]#1, %{{.*}} : f32
//  CHECK-NEXT:   %[[MUL2:.*]] = arith.mulf %[[LR]]#2, %[[LR]]#0 : f32
//  CHECK-NEXT:   memref.store %[[LR]]#3, %[[R]][%[[C7]]] : memref<?xf32>
//  CHECK-NEXT:   %[[MUL3:.*]] = arith.mulf %[[ADD3]], %[[LR]]#1 : f32
//  CHECK-NEXT:   memref.store %[[MUL2]], %[[R]][%[[C8]]] : memref<?xf32>
//  CHECK-NEXT:   memref.store %[[MUL3]], %[[R]][%[[C9]]] : memref<?xf32>
func.func @multiple_uses(%A: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cf = arith.constant 1.0 : f32
  scf.for %i0 = %c0 to %c10 step %c1 {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 3 } : memref<?xf32>
    %A1_elem = arith.addf %A_elem, %cf { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 0 } : f32
    %A2_elem = arith.mulf %A1_elem, %A_elem { __test_pipelining_stage__ = 2, __test_pipelining_op_order__ = 1 } : f32
    memref.store %A2_elem, %result[%i0] { __test_pipelining_stage__ = 3, __test_pipelining_op_order__ = 2 } : memref<?xf32>
  } { __test_pipelining_loop__ }
  return
}

// -----

// CHECK-LABEL: region_multiple_uses(
//  CHECK-SAME:   %[[A:.*]]: memref<?xf32>, %[[R:.*]]: memref<?xf32>) {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C7:.*]] = arith.constant 7 : index
//   CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
//   CHECK-DAG:   %[[C9:.*]] = arith.constant 9 : index
// Prologue:
//       CHECK:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
//  CHECK-NEXT:   %[[ADD0:.*]] = arith.addf %[[L0]], %{{.*}} : f32
//  CHECK-NEXT:   %[[L1:.*]] = memref.load %[[A]][%[[C1]]] : memref<?xf32>
//  CHECK-NEXT:   %[[ADD1:.*]] = arith.addf %[[L1]], %{{.*}} : f32
//  CHECK-NEXT:   %[[MUL0:.*]] = scf.execute_region
// arith.mulf %[[ADD0]], %[[L0]] : f32
//  CHECK:   %[[L2:.*]] = memref.load %[[A]][%[[C2]]] : memref<?xf32>
// Kernel:
//  CHECK-NEXT:   %[[LR:.*]]:4 = scf.for %[[IV:.*]] = %[[C0]] to %[[C7]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[LA1:.*]] = %[[L1]],
//  CHECK-SAME:     %[[LA2:.*]] = %[[L2]], %[[ADDARG1:.*]] = %[[ADD1]],
//  CHECK-SAME:     %[[MULARG0:.*]] = %[[MUL0]]) -> (f32, f32, f32, f32) {
//  CHECK-NEXT:     %[[ADD2:.*]] = arith.addf %[[LA2]], %{{.*}} : f32
//  CHECK-NEXT:     %[[MUL1:.*]] = scf.execute_region
// arith.mulf %[[ADDARG1]], %[[LA1]] : f32
//       CHECK:     memref.store %[[MULARG0]], %[[R]][%[[IV]]] : memref<?xf32>
//  CHECK-NEXT:     %[[IV3:.*]] = arith.addi %[[IV]], %[[C3]] : index
//  CHECK-NEXT:     %[[L3:.*]] = memref.load %[[A]][%[[IV3]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[LA2]], %[[L3]], %[[ADD2]], %[[MUL1]] : f32, f32, f32, f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:   %[[ADD3:.*]] = arith.addf %[[LR]]#1, %{{.*}} : f32
//  CHECK-NEXT:   %[[MUL2:.*]] = scf.execute_region
// arith.mulf %[[LR]]#2, %[[LR]]#0 : f32
//       CHECK:   memref.store %[[LR]]#3, %[[R]][%[[C7]]] : memref<?xf32>
//  CHECK-NEXT:   %[[MUL3:.*]] = scf.execute_region
/// %[[ADD3]], %[[LR]]#1 : f32
//       CHECK:   memref.store %[[MUL2]], %[[R]][%[[C8]]] : memref<?xf32>
//  CHECK-NEXT:   memref.store %[[MUL3]], %[[R]][%[[C9]]] : memref<?xf32>

func.func @region_multiple_uses(%A: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cf = arith.constant 1.0 : f32
  scf.for %i0 = %c0 to %c10 step %c1 {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 3 } : memref<?xf32>
    %A1_elem = arith.addf %A_elem, %cf { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 0 } : f32
    %A2_elem = scf.execute_region -> f32 {
      %A2_elem1 = arith.mulf %A1_elem, %A_elem : f32
      scf.yield %A2_elem1 : f32
    } { __test_pipelining_stage__ = 2, __test_pipelining_op_order__ = 1 }
    memref.store %A2_elem, %result[%i0] { __test_pipelining_stage__ = 3, __test_pipelining_op_order__ = 2 } : memref<?xf32>
  } { __test_pipelining_loop__ }
  return
}

// -----

// CHECK-LABEL: loop_carried(
//  CHECK-SAME:   %[[A:.*]]: memref<?xf32>, %[[R:.*]]: memref<?xf32>) {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[CSTF:.*]] = arith.constant 1.000000e+00 : f32
// Prologue:
//       CHECK:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
// Kernel:
//  CHECK-NEXT:   %[[LR:.*]]:2 = scf.for %[[IV:.*]] = %[[C0]] to %[[C3]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[C:.*]] = %[[CSTF]],
//  CHECK-SAME:     %[[LARG:.*]] = %[[L0]]) -> (f32, f32) {
//  CHECK-NEXT:     %[[ADD0:.*]] = arith.addf %[[LARG]], %[[C]] : f32
//  CHECK-NEXT:     %[[IV1:.*]] = arith.addi %[[IV]], %[[C1]] : index
//  CHECK-NEXT:     %[[L1:.*]] = memref.load %[[A]][%[[IV1]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[ADD0]], %[[L1]] : f32, f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:   %[[ADD1:.*]] = arith.addf %[[LR]]#1, %[[LR]]#0 : f32
//  CHECK-NEXT:   memref.store %[[ADD1]], %[[R]][%[[C0]]] : memref<?xf32>
func.func @loop_carried(%A: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cf = arith.constant 1.0 : f32
  %r = scf.for %i0 = %c0 to %c4 step %c1 iter_args(%arg0 = %cf) -> (f32) {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 1 } : memref<?xf32>
    %A1_elem = arith.addf %A_elem, %arg0 { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 0 } : f32
    scf.yield %A1_elem : f32
  }  { __test_pipelining_loop__ }
  memref.store %r, %result[%c0] : memref<?xf32>
  return
}

// -----

// CHECK-LABEL: backedge_different_stage
//  CHECK-SAME:   (%[[A:.*]]: memref<?xf32>) -> f32 {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[CSTF:.*]] = arith.constant 1.000000e+00 : f32
// Prologue:
//       CHECK:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
//  CHECK-NEXT:   %[[ADD0:.*]] = arith.addf %[[L0]], %[[CSTF]] : f32
//  CHECK-NEXT:   %[[L1:.*]] = memref.load %[[A]][%[[C1]]] : memref<?xf32>
// Kernel:
//  CHECK-NEXT:   %[[R:.*]]:3 = scf.for %[[IV:.*]] = %[[C0]] to %[[C2]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[C:.*]] = %[[CSTF]],
//  CHECK-SAME:     %[[ADDARG:.*]] = %[[ADD0]], %[[LARG:.*]] = %[[L1]]) -> (f32, f32, f32) {
//  CHECK-NEXT:     %[[ADD1:.*]] = arith.addf %[[LARG]], %[[ADDARG]] : f32
//  CHECK-NEXT:     %[[IV2:.*]] = arith.addi %[[IV]], %[[C2]] : index
//  CHECK-NEXT:     %[[L2:.*]] = memref.load %[[A]][%[[IV2]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[ADDARG]], %[[ADD1]], %[[L2]] : f32, f32, f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:   %[[ADD2:.*]] = arith.addf %[[R]]#2, %[[R]]#1 : f32
//  CHECK-NEXT:   return %[[ADD2]] : f32
func.func @backedge_different_stage(%A: memref<?xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cf = arith.constant 1.0 : f32
  %r = scf.for %i0 = %c0 to %c4 step %c1 iter_args(%arg0 = %cf) -> (f32) {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 2 } : memref<?xf32>
    %A1_elem = arith.addf %A_elem, %arg0 { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 1 } : f32
    %A2_elem = arith.mulf %cf, %A1_elem { __test_pipelining_stage__ = 2, __test_pipelining_op_order__ = 0 } : f32
    scf.yield %A2_elem : f32
  }  { __test_pipelining_loop__ }
  return %r : f32
}

// -----

// CHECK-LABEL: region_backedge_different_stage
//  CHECK-SAME:   (%[[A:.*]]: memref<?xf32>) -> f32 {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[CSTF:.*]] = arith.constant 1.000000e+00 : f32
// Prologue:
//       CHECK:   %[[L0:.*]] = scf.execute_region
//  CHECK-NEXT:     memref.load %[[A]][%[[C0]]] : memref<?xf32>
//       CHECK:   %[[ADD0:.*]] = scf.execute_region
//  CHECK-NEXT:   arith.addf %[[L0]], %[[CSTF]] : f32
//       CHECK:   %[[L1:.*]] = scf.execute_region
//  CHECK-NEXT:     memref.load %[[A]][%[[C1]]] : memref<?xf32>
// Kernel:
//       CHECK:   %[[R:.*]]:3 = scf.for %[[IV:.*]] = %[[C0]] to %[[C2]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[C:.*]] = %[[CSTF]],
//  CHECK-SAME:     %[[ADDARG:.*]] = %[[ADD0]], %[[LARG:.*]] = %[[L1]]) -> (f32, f32, f32) {
//       CHECK:     %[[ADD1:.*]] = scf.execute_region
//  CHECK-NEXT:       arith.addf %[[LARG]], %[[ADDARG]] : f32
//       CHECK:     %[[IV2:.*]] = arith.addi %[[IV]], %[[C2]] : index
//       CHECK:     %[[L2:.*]] = scf.execute_region
//  CHECK-NEXT:       memref.load %[[A]][%[[IV2]]] : memref<?xf32>
//       CHECK:     scf.yield %[[ADDARG]], %[[ADD1]], %[[L2]] : f32, f32, f32
//  CHECK-NEXT:   }
// Epilogue:
//       CHECK:   %[[ADD2:.*]] = scf.execute_region
//  CHECK-NEXT:    arith.addf %[[R]]#2, %[[R]]#1 : f32
//       CHECK:   return %[[ADD2]] : f32

func.func @region_backedge_different_stage(%A: memref<?xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cf = arith.constant 1.0 : f32
  %r = scf.for %i0 = %c0 to %c4 step %c1 iter_args(%arg0 = %cf) -> (f32) {
    %A_elem = scf.execute_region -> f32 {
      %A_elem1 = memref.load %A[%i0] : memref<?xf32>
      scf.yield %A_elem1 : f32
    } { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 2 }
    %A1_elem = scf.execute_region -> f32 {
      %inner = arith.addf %A_elem, %arg0 : f32
      scf.yield %inner : f32
    }  { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 1 }
    %A2_elem = arith.mulf %cf, %A1_elem { __test_pipelining_stage__ = 2, __test_pipelining_op_order__ = 0 } : f32
    scf.yield %A2_elem : f32
  }  { __test_pipelining_loop__ }
  return %r : f32
}


// -----

// CHECK-LABEL: backedge_same_stage
//  CHECK-SAME:   (%[[A:.*]]: memref<?xf32>) -> f32 {
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[CSTF:.*]] = arith.constant 1.000000e+00 : f32
// Prologue:
//       CHECK:   %[[L0:.*]] = memref.load %[[A]][%[[C0]]] : memref<?xf32>
// Kernel:
//  CHECK-NEXT:   %[[R:.*]]:2 = scf.for %[[IV:.*]] = %[[C0]] to %[[C3]]
//  CHECK-SAME:     step %[[C1]] iter_args(%[[C:.*]] = %[[CSTF]],
//  CHECK-SAME:     %[[LARG:.*]] = %[[L0]]) -> (f32, f32) {
//  CHECK-NEXT:     %[[ADD0:.*]] = arith.addf %[[LARG]], %[[C]] : f32
//  CHECK-NEXT:     %[[IV1:.*]] = arith.addi %[[IV]], %[[C1]] : index
//  CHECK-NEXT:     %[[L2:.*]] = memref.load %[[A]][%[[IV1]]] : memref<?xf32>
//  CHECK-NEXT:     scf.yield %[[ADD0]], %[[L2]] : f32, f32
//  CHECK-NEXT:   }
// Epilogue:
//  CHECK-NEXT:   %[[ADD1:.*]] = arith.addf %[[R]]#1, %[[R]]#0 : f32
//  CHECK-NEXT:   return %[[ADD1]] : f32
func.func @backedge_same_stage(%A: memref<?xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cf = arith.constant 1.0 : f32
  %r = scf.for %i0 = %c0 to %c4 step %c1 iter_args(%arg0 = %cf) -> (f32) {
    %A_elem = memref.load %A[%i0] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 2 } : memref<?xf32>
    %A1_elem = arith.addf %A_elem, %arg0 { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 0 } : f32
    %A2_elem = arith.mulf %cf, %A1_elem { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 1 } : f32
    scf.yield %A2_elem : f32
  }  { __test_pipelining_loop__ }
  return %r : f32
}

// -----

// CHECK: @pipeline_op_with_region(%[[ARG0:.+]]: memref<?xf32>, %[[ARG1:.+]]: memref<?xf32>, %[[ARG2:.+]]: memref<?xf32>) {
// CHECK:   %[[C0:.+]] = arith.constant 0 :
// CHECK:   %[[C3:.+]] = arith.constant 3 :
// CHECK:   %[[C1:.+]] = arith.constant 1 :
// CHECK:   %[[APRO:.+]] = memref.alloc() :
// CHECK:   %[[BPRO:.+]] = memref.alloc() :
// CHECK:   %[[ASV0:.+]] = memref.subview %[[ARG0]][%[[C0]]] [8] [1] :
// CHECK:   %[[BSV0:.+]] = memref.subview %[[ARG1]][%[[C0]]] [8] [1] :

// Prologue:
// CHECK:   %[[PAV0:.+]] = memref.subview %[[APRO]][%[[C0]], 0] [1, 8] [1, 1] :
// CHECK:   %[[PBV0:.+]] = memref.subview %[[BPRO]][%[[C0]], 0] [1, 8] [1, 1] :
// CHECK:   memref.copy %[[ASV0]], %[[PAV0]] :
// CHECK:   memref.copy %[[BSV0]], %[[PBV0]] :

// Kernel:
// CHECK:   %[[R:.+]]:2 = scf.for %[[IV:.+]] = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK-SAME: iter_args(%[[IA:.+]] = %[[PAV0]], %[[IB:.+]] = %[[PBV0:.+]])
// CHECK:     %[[CV:.+]] = memref.subview %[[ARG2]]
// CHECK:     linalg.generic
// CHECK-SAME:  ins(%[[IA]], %[[IB]], %{{.*}} : {{.*}}) outs(%[[CV]] :
// CHECK:     %[[NEXT:.+]] = arith.addi %[[IV]], %[[C1]]
// CHECK:     %[[ASV:.+]] = memref.subview %[[ARG0]][%[[NEXT]]] [8] [1] :
// CHECK:     %[[NEXT:.+]] = arith.addi %[[IV]], %[[C1]] :
// CHECK:     %[[BSV:.+]] = memref.subview %[[ARG1]][%[[NEXT]]] [8] [1] :
// CHECK:     %[[NEXT:.+]] = arith.addi %[[IV]], %[[C1]] :
// CHECK:     %[[BUFIDX:.+]] = affine.apply
// CHECK:     %[[APROSV:.+]] = memref.subview %[[APRO]][%[[BUFIDX]], 0] [1, 8] [1, 1] :
// CHECK:     %[[BPROSV:.+]] = memref.subview %[[BPRO]][%[[BUFIDX]], 0] [1, 8] [1, 1] :
// CHECK:     memref.copy %[[ASV]], %[[APROSV]] :
// CHECK:     memref.copy %[[BSV]], %[[BPROSV]] :
// CHECK:     scf.yield %[[APROSV]], %[[BPROSV]] :
// CHECK:   }
// CHECK:   %[[CV:.+]] = memref.subview %[[ARG2]][%[[C3]]] [8] [1] :
// CHECK:   linalg.generic
// CHECK-SAME: ins(%[[R]]#0, %[[R]]#1, %{{.*}} : {{.*}}) outs(%[[CV]] :


#map = affine_map<(d0)[s0]->(d0 + s0)>
#map1 = affine_map<(d0)->(d0)>
#map2 = affine_map<(d0)->()>
#linalg_attrs = {
  indexing_maps = [
      #map1,
      #map1,
      #map2,
      #map1
    ],
  iterator_types = ["parallel"],
  __test_pipelining_stage__ = 1,
  __test_pipelining_op_order__ = 2
}
func.func @pipeline_op_with_region(%A: memref<?xf32>, %B: memref<?xf32>, %result: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cf = arith.constant 1.0 : f32
  %a_buf = memref.alloc() : memref<2x8xf32>
  %b_buf = memref.alloc() : memref<2x8xf32>
  scf.for %i0 = %c0 to %c4 step %c1 {
    %A_view = memref.subview %A[%i0][8][1] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 3 } : memref<?xf32> to memref<8xf32, #map>
    %B_view = memref.subview %B[%i0][8][1] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 4 } : memref<?xf32> to memref<8xf32, #map>
    %buf_idx = affine.apply  affine_map<(d0)->(d0 mod 2)> (%i0)[] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 5 }
    %a_buf_view = memref.subview %a_buf[%buf_idx,0][1,8][1,1] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 6 } : memref<2x8xf32> to memref<8xf32, #map>
    %b_buf_view = memref.subview %b_buf[%buf_idx,0][1,8][1,1] { __test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 7 } : memref<2x8xf32> to memref<8xf32, #map>
    memref.copy %A_view , %a_buf_view {__test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 8} : memref<8xf32, #map> to memref<8xf32, #map>
    memref.copy %B_view , %b_buf_view {__test_pipelining_stage__ = 0, __test_pipelining_op_order__ = 9} : memref<8xf32, #map> to memref<8xf32, #map>
    %C_view = memref.subview %result[%i0][8][1] { __test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 0 } : memref<?xf32> to memref<8xf32, #map>
    %scalar = arith.addf %cf, %cf {__test_pipelining_stage__ = 1, __test_pipelining_op_order__ = 1} : f32
    linalg.generic #linalg_attrs ins(%a_buf_view, %b_buf_view, %scalar : memref<8xf32, #map>, memref<8xf32, #map>, f32)
      outs(%C_view: memref<8xf32, #map>) {
      ^bb0(%a: f32, %b: f32, %s: f32, %c: f32):
        %add = arith.addf %a, %b : f32
        %accum = arith.addf %add, %c : f32
        %accum1 = arith.addf %scalar, %accum : f32
        %accum2 = arith.addf %s, %accum1 : f32
        linalg.yield %accum2 : f32
    }
    scf.yield
  }  { __test_pipelining_loop__ }
  return
}
