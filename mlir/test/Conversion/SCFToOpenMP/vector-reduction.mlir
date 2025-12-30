// RUN: mlir-opt %s --convert-scf-to-openmp | FileCheck %s

// CHECK-LABEL: omp.declare_reduction @__scf_reduction : vector<2xi1> init
// CHECK-NEXT:  ^bb0(%arg0: vector<2xi1>):
// CHECK-NEXT:    %[[CONST:.*]] = llvm.mlir.constant(dense<true> : vector<2xi1>) : vector<2xi1>
// CHECK-NEXT:    omp.yield(%[[CONST]] : vector<2xi1>)

func.func @vector_and_reduction() {
  %v_mask = vector.constant_mask [1] : vector<2xi1>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %result = scf.parallel (%i) = (%c0) to (%c2) step (%c1) init(%v_mask) -> vector<2xi1> {
    %val = vector.constant_mask [1] : vector<2xi1>
    scf.reduce (%val : vector<2xi1>) {
    ^bb0(%lhs: vector<2xi1>, %rhs: vector<2xi1>):
      %0 = arith.andi %lhs, %rhs : vector<2xi1>
      scf.reduce.return %0 : vector<2xi1>
    }
  }
  return
}