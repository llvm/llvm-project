// RUN: mlir-opt %s -scf-for-loop-continuous-peeling=convert-single-iter-loops-to-if=true -split-input-file | FileCheck %s

#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func @foo(%ub: index) -> index {
  %c0 = arith.constant 0 : index
  %step = arith.constant 8 : index
  %0 = scf.for %iv = %c0 to %ub step %step iter_args(%arg = %c0) -> (index) {
    %1 = affine.min #map(%ub, %iv)[%step]
    %2 = index.add %1, %arg
    scf.yield %2 : index
  }
  return %0 : index
}

// CHECK:   #[[MAP:.*]] = affine_map<()[s0, s1, s2] -> (s1 - s1 mod s2)>
// CHECK: func.func @foo(%[[UB:.*]]: index) -> index {
// CHECK: %[[STEP1:.*]] = arith.constant 1 : index
// CHECK: %[[STEP2:.*]] = arith.constant 2 : index
// CHECK: %[[STEP4:.*]] = arith.constant 4 : index
// CHECK:    %[[LB:.*]] = arith.constant 0 : index
// CHECK: %[[STEP8:.*]] = arith.constant 8 : index
// CHECK:    %[[I0:.*]] = affine.apply #[[MAP]]()[%[[LB]], %[[UB]], %[[STEP8]]]
// CHECK:    %[[I1:.*]] = scf.for %{{.*}} = %[[LB]] to %[[I0]] step %[[STEP8]] iter_args(%[[ALB:.*]] = %[[LB]]) -> (index) {
// CHECK:   %[[SUM:.*]] =   index.add %[[ALB]], %[[STEP8]]
// CHECK:                   scf.yield %[[SUM]] : index
// CHECK:    %[[I2:.*]] = affine.apply #[[MAP]]()[%[[I0]], %[[UB]], %[[STEP4]]]
// CHECK:    %[[I3:.*]] = arith.cmpi slt, %[[I0]], %[[I2]] : index
// CHECK:    %[[I4:.*]] = scf.if %[[I3]] -> (index) {
// CHECK:   %[[SUM:.*]] =   index.add %[[I1]], %[[STEP4]]
// CHECK:                   scf.yield %[[SUM]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[I1]] : index
// CHECK:    %[[I5:.*]] = affine.apply #[[MAP]]()[%[[I2]], %[[UB]], %[[STEP2]]]
// CHECK:    %[[I6:.*]] = arith.cmpi slt, %[[I2]], %[[I5]] : index
// CHECK:    %[[I7:.*]] = scf.if %[[I6]] -> (index) {
// CHECK:   %[[SUM:.*]] =   index.add %[[I4]], %[[STEP2]]
// CHECK:                   scf.yield %[[SUM]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[I4]] : index
// CHECK:    %[[I8:.*]] = arith.cmpi slt, %[[I5]], %[[UB]] : index
// CHECK:    %[[I9:.*]] = scf.if %[[I8]] -> (index) {
// CHECK:   %[[SUM:.*]] =   index.add %[[I7]], %[[STEP1]]
// CHECK:                   scf.yield %[[SUM]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[I7]] : index
// CHECK:                 return %[[I9]] : index

// -----

#map1 = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func @foo1(%ub: index) -> index {
  %c0 = arith.constant 1 : index
  %step = arith.constant 8 : index
  %0 = scf.for %iv = %c0 to %ub step %step iter_args(%arg = %c0) -> (index) {
    %1 = affine.min #map1(%ub, %iv)[%step]
    %2 = index.add %1, %arg
    scf.yield %2 : index
  }
  return %0 : index
}

// CHECK:   #[[MAP:.*]] = affine_map<()[s0, s1, s2] -> (s1 - (s1 - s0) mod s2)>
// CHECK: func.func @foo1(%[[UB:.*]]: index) -> index {
// CHECK: %[[STEP2:.*]] = arith.constant 2 : index
// CHECK: %[[STEP4:.*]] = arith.constant 4 : index
// CHECK: %[[STEP1:.*]] = arith.constant 1 : index
// CHECK: %[[STEP8:.*]] = arith.constant 8 : index
// CHECK:    %[[I0:.*]] = affine.apply #[[MAP]]()[%[[STEP1]], %[[UB]], %[[STEP8]]]
// CHECK:    %[[I1:.*]] = scf.for %{{.*}} = %[[STEP1]] to %[[I0]] step %[[STEP8]] iter_args(%[[ALB:.*]] = %[[STEP1]]) -> (index) {
// CHECK:   %[[SUM:.*]] =   index.add %[[ALB]], %[[STEP8]]
// CHECK:                   scf.yield %[[SUM]] : index
// CHECK:    %[[I2:.*]] = affine.apply #[[MAP]]()[%[[I0]], %[[UB]], %[[STEP4]]]
// CHECK:    %[[I3:.*]] = arith.cmpi slt, %[[I0]], %[[I2]] : index
// CHECK:    %[[I4:.*]] = scf.if %[[I3]] -> (index) {
// CHECK:   %[[SUM:.*]] =   index.add %[[I1]], %[[STEP4]]
// CHECK:                   scf.yield %[[SUM]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[I1]] : index
// CHECK:    %[[I5:.*]] = affine.apply #[[MAP]]()[%[[I2]], %[[UB]], %[[STEP2]]]
// CHECK:    %[[I6:.*]] = arith.cmpi slt, %[[I2]], %[[I5]] : index
// CHECK:    %[[I7:.*]] = scf.if %[[I6]] -> (index) {
// CHECK:   %[[SUM:.*]] =   index.add %[[I4]], %[[STEP2]]
// CHECK:                   scf.yield %[[SUM]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[I4]] : index
// CHECK:    %[[I8:.*]] = arith.cmpi slt, %[[I5]], %[[UB]] : index
// CHECK:    %[[I9:.*]] = scf.if %[[I8]] -> (index) {
// CHECK:   %[[SUM:.*]] =   index.add %[[I7]], %[[STEP1]]
// CHECK:                   scf.yield %[[SUM]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[I7]] : index
// CHECK:                 return %[[I9]] : index