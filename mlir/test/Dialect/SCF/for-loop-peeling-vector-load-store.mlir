// RUN: mlir-opt %s -scf-for-loop-peeling -canonicalize -verify-diagnostics | FileCheck %s

func.func @vector_read_write(%a : memref<100xi32>, %b : memref<100xi32>, %ub: index) {
// %LB to %NEW_UB will be multiple of STEP after peeling.
// So vector.transfer_write could be transferred to vector.store to avoid
// generating mask when lowering to LLVM.
//
//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 64)>
//      CHECK: func @vector_read_write(
// CHECK-SAME:   %[[A:.*]]: memref<100xi32>, %[[B:.*]]: memref<100xi32>, %[[UB:.*]]: index
//      CHECK:   %[[LB:.*]] = arith.constant 0 : index
//      CHECK:   %[[STEP:.*]] = arith.constant 64 : index
//      CHECK:   %[[NEW_UB:.*]] = affine.apply #[[MAP0]]
//      CHECK:   scf.for %[[IV:.*]] = %[[LB]] to %[[NEW_UB]] step %[[STEP]] {
//      CHECK:     %[[VAL:.*]] = vector.load %[[B]][%[[IV]]]
//      CHECK:     vector.store %[[VAL]], %[[A]][%[[IV]]]
//      CHECK:   }
//      CHECK:   scf.for %[[IV:.*]] = %[[NEW_UB]] to %[[UB]] step %[[STEP]] {
//      CHECK:     %[[VAL:.*]] = vector.transfer_read %[[B]][%[[IV]]]
//      CHECK:     vector.transfer_write %[[VAL]], %[[A]][%[[IV]]]
//      CHECK:   }
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %pad = arith.constant 0 : i32
  scf.for %i = %c0 to %ub step %c64 {
    %val = vector.transfer_read %b[%i], %pad : memref<100xi32>, vector<64xi32>
    vector.transfer_write %val, %a[%i] : vector<64xi32>, memref<100xi32>
  }
  return
}
