// RUN: tr-opt %s --tr-lower-affine | FileCheck %s

// Milestone 15: standard Affine -> SCF + Arith lowering.
// Before: affine.for / affine.apply. After: scf.for / arith.muli+addi.

#map = affine_map<()[s0] -> (s0 * 128)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 128)>
#map2 = affine_map<()[s0, s1, s2] -> (s0 * 32 + s1 + s2 * 128)>
#map3 = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0 * 128)>

// CHECK-LABEL: func.func @row_base
// CHECK-SAME: (%[[PID:.*]]: index)
func.func @row_base(%pid: index) -> index {
  // CHECK: arith.constant 128 : index
  // CHECK: arith.muli %[[PID]], %{{.*}}
  // CHECK-NOT: affine.apply
  %0 = affine.apply #map()[%pid]
  return %0 : index
}

// CHECK-LABEL: func.func @global_row
// CHECK-SAME: (%[[PID:.*]]: index, %[[LOCAL:.*]]: index)
func.func @global_row(%pid: index, %local: index) -> index {
  // CHECK: arith.muli %[[PID]], %{{.*}}
  // CHECK: arith.addi %[[LOCAL]], %{{.*}}
  // CHECK-NOT: affine.apply
  %0 = affine.apply #map1()[%local, %pid]
  return %0 : index
}

// CHECK-LABEL: func.func @global_col
// CHECK-SAME: (%[[KT:.*]]: index, %[[LANE:.*]]: index, %[[J:.*]]: index)
func.func @global_col(%kt: index, %lane: index, %j: index) -> index {
  // CHECK: arith.muli %[[J]], %{{.*}}
  // CHECK: arith.addi %{{.*}}, %[[LANE]]
  // CHECK: arith.muli %[[KT]], %{{.*}}
  // CHECK: arith.addi
  // CHECK-NOT: affine.apply
  %0 = affine.apply #map2()[%j, %lane, %kt]
  return %0 : index
}

// CHECK-LABEL: func.func @local_rows
func.func @local_rows(%pid: index) -> index {
  %c0 = arith.constant 0 : index
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args
  // CHECK: arith.addi
  // CHECK: arith.muli
  // CHECK: scf.yield
  // CHECK-NOT: affine.for
  // CHECK-NOT: affine.apply
  %r = affine.for %local = 0 to 16 iter_args(%acc = %c0) -> index {
    %g = affine.apply #map3(%acc, %local)[%pid]
    affine.yield %g : index
  }
  return %r : index
}
