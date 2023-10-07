// RUN: mlir-opt  -test-transform-dialect-interpreter -canonicalize --split-input-file --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func @hoist_vector_transfer_pairs(
//  CHECK-SAME:   %[[MEMREF0:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF1:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF2:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF3:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF4:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF5:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[VAL:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[LB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[UB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[STEP:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[CMP:[a-zA-Z0-9]*]]: i1
func.func @hoist_vector_transfer_pairs(
    %memref0: memref<?x?xf32>, %memref1: memref<?x?xf32>, %memref2: memref<?x?xf32>,
    %memref3: memref<?x?xf32>, %memref4: memref<?x?xf32>, %memref5: memref<?x?xf32>,
    %val: index, %lb : index, %ub : index, %step: index, %cmp: i1) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32

// CHECK: vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<1xf32>
// CHECK: scf.for %[[I:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}}) -> (vector<1xf32>) {
// CHECK:   vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<2xf32>
// CHECK:   scf.for %[[J:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}}) -> (vector<1xf32>, vector<2xf32>) {
// CHECK:     vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<3xf32>
// CHECK:     vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<4xf32>
// CHECK:     "some_crippling_use"(%[[MEMREF4]]) : (memref<?x?xf32>) -> ()
// CHECK:     vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<5xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<1xf32>) -> vector<1xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%[[MEMREF2]], %{{.*}}) : (memref<?x?xf32>, vector<3xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<5xf32>) -> vector<5xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<3xf32>, memref<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<4xf32>, memref<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<5xf32>, memref<?x?xf32>
// CHECK:     "some_crippling_use"(%[[MEMREF3]]) : (memref<?x?xf32>) -> ()
// CHECK:     scf.yield {{.*}} : vector<1xf32>, vector<2xf32>
// CHECK:   }
// CHECK:   vector.transfer_write %{{.*}} : vector<2xf32>, memref<?x?xf32>
// CHECK:   "unrelated_use"(%[[MEMREF0]]) : (memref<?x?xf32>) -> ()
// CHECK:   scf.yield {{.*}} : vector<1xf32>
// CHECK: }
// CHECK: vector.transfer_write %{{.*}} : vector<1xf32>, memref<?x?xf32>
// CHECK: "unrelated_use"(%[[MEMREF1]]) : (memref<?x?xf32>) -> ()
  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      %r0 = vector.transfer_read %memref1[%c0, %c0], %cst: memref<?x?xf32>, vector<1xf32>
      %r1 = vector.transfer_read %memref0[%i, %i], %cst: memref<?x?xf32>, vector<2xf32>
      %r2 = vector.transfer_read %memref2[%c0, %c0], %cst: memref<?x?xf32>, vector<3xf32>
      %r3 = vector.transfer_read %memref3[%c0, %c0], %cst: memref<?x?xf32>, vector<4xf32>
      "some_crippling_use"(%memref4) : (memref<?x?xf32>) -> ()
      %r4 = vector.transfer_read %memref4[%c0, %c0], %cst: memref<?x?xf32>, vector<5xf32>
      %r5 = vector.transfer_read %memref5[%c0, %c0], %cst: memref<?x?xf32>, vector<6xf32>
      "some_crippling_use"(%memref5) : (memref<?x?xf32>) -> ()
      %u0 = "some_use"(%r0) : (vector<1xf32>) -> vector<1xf32>
      %u1 = "some_use"(%r1) : (vector<2xf32>) -> vector<2xf32>
      %u2 = "some_use"(%memref2, %r2) : (memref<?x?xf32>, vector<3xf32>) -> vector<3xf32>
      %u3 = "some_use"(%r3) : (vector<4xf32>) -> vector<4xf32>
      %u4 = "some_use"(%r4) : (vector<5xf32>) -> vector<5xf32>
      %u5 = "some_use"(%r5) : (vector<6xf32>) -> vector<6xf32>
      vector.transfer_write %u0, %memref1[%c0, %c0] : vector<1xf32>, memref<?x?xf32>
      vector.transfer_write %u1, %memref0[%i, %i] : vector<2xf32>, memref<?x?xf32>
      vector.transfer_write %u2, %memref2[%c0, %c0] : vector<3xf32>, memref<?x?xf32>
      vector.transfer_write %u3, %memref3[%c0, %c0] : vector<4xf32>, memref<?x?xf32>
      vector.transfer_write %u4, %memref4[%c0, %c0] : vector<5xf32>, memref<?x?xf32>
      vector.transfer_write %u5, %memref5[%c0, %c0] : vector<6xf32>, memref<?x?xf32>
      "some_crippling_use"(%memref3) : (memref<?x?xf32>) -> ()
    }
    "unrelated_use"(%memref0) : (memref<?x?xf32>) -> ()
  }
  "unrelated_use"(%memref1) : (memref<?x?xf32>) -> ()
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_disjoint(
//  CHECK-SAME:   %[[MEMREF0:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF1:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF2:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF3:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[VAL:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[LB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[UB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[STEP:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[RANDOM:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[CMP:[a-zA-Z0-9]*]]: i1
func.func @hoist_vector_transfer_pairs_disjoint(
    %memref0: memref<?x?xf32>, %memref1: memref<?x?xf32>,
    %memref2: memref<?x?xf32>, %memref3: memref<?x?xf32>, %val: index, %lb : index, %ub : index,
    %step: index, %random_index : index, %cmp: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.0 : f32

// CHECK: vector.transfer_read %[[MEMREF2]]{{.*}} : memref<?x?xf32>, vector<3xf32>
// CHECK: vector.transfer_read %[[MEMREF2]]{{.*}} : memref<?x?xf32>, vector<3xf32>
// CHECK: vector.transfer_read %[[MEMREF3]]{{.*}} : memref<?x?xf32>, vector<4xf32>
// CHECK: vector.transfer_read %[[MEMREF3]]{{.*}} : memref<?x?xf32>, vector<4xf32>
// CHECK: scf.for %[[I:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}}) ->
//  CHECK-SAME: (vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>) {
// CHECK:   scf.for %[[J:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}}) ->
//  CHECK-SAME: (vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>) {
// CHECK:     vector.transfer_read %[[MEMREF1]]{{.*}} : memref<?x?xf32>, vector<2xf32>
// CHECK:     vector.transfer_read %[[MEMREF1]]{{.*}} : memref<?x?xf32>, vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<3xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<3xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     vector.transfer_write %{{.*}}, %[[MEMREF1]]{{.*}} : vector<2xf32>, memref<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}}, %[[MEMREF1]]{{.*}} : vector<2xf32>, memref<?x?xf32>
// CHECK:     scf.yield {{.*}} : vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>
// CHECK:   }
// CHECK:   scf.yield {{.*}} : vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>
// CHECK: }
// CHECK: vector.transfer_write %{{.*}}, %[[MEMREF3]]{{.*}} : vector<4xf32>, memref<?x?xf32>
// CHECK: vector.transfer_write %{{.*}}, %[[MEMREF3]]{{.*}} : vector<4xf32>, memref<?x?xf32>
// CHECK: vector.transfer_write %{{.*}}, %[[MEMREF2]]{{.*}} : vector<3xf32>, memref<?x?xf32>
// CHECK: vector.transfer_write %{{.*}}, %[[MEMREF2]]{{.*}} : vector<3xf32>, memref<?x?xf32>
  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      %r00 = vector.transfer_read %memref1[%c0, %c0], %cst: memref<?x?xf32>, vector<2xf32>
      %r01 = vector.transfer_read %memref1[%c0, %c1], %cst: memref<?x?xf32>, vector<2xf32>
      %r20 = vector.transfer_read %memref2[%c0, %c0], %cst: memref<?x?xf32>, vector<3xf32>
      %r21 = vector.transfer_read %memref2[%c0, %c3], %cst: memref<?x?xf32>, vector<3xf32>
      %r30 = vector.transfer_read %memref3[%c0, %random_index], %cst: memref<?x?xf32>, vector<4xf32>
      %r31 = vector.transfer_read %memref3[%c1, %random_index], %cst: memref<?x?xf32>, vector<4xf32>
      %r10 = vector.transfer_read %memref0[%i, %i], %cst: memref<?x?xf32>, vector<2xf32>
      %r11 = vector.transfer_read %memref0[%random_index, %random_index], %cst: memref<?x?xf32>, vector<2xf32>
      %u00 = "some_use"(%r00) : (vector<2xf32>) -> vector<2xf32>
      %u01 = "some_use"(%r01) : (vector<2xf32>) -> vector<2xf32>
      %u20 = "some_use"(%r20) : (vector<3xf32>) -> vector<3xf32>
      %u21 = "some_use"(%r21) : (vector<3xf32>) -> vector<3xf32>
      %u30 = "some_use"(%r30) : (vector<4xf32>) -> vector<4xf32>
      %u31 = "some_use"(%r31) : (vector<4xf32>) -> vector<4xf32>
      %u10 = "some_use"(%r10) : (vector<2xf32>) -> vector<2xf32>
      %u11 = "some_use"(%r11) : (vector<2xf32>) -> vector<2xf32>
      vector.transfer_write %u00, %memref1[%c0, %c0] : vector<2xf32>, memref<?x?xf32>
      vector.transfer_write %u01, %memref1[%c0, %c1] : vector<2xf32>, memref<?x?xf32>
      vector.transfer_write %u20, %memref2[%c0, %c0] : vector<3xf32>, memref<?x?xf32>
      vector.transfer_write %u21, %memref2[%c0, %c3] : vector<3xf32>, memref<?x?xf32>
      vector.transfer_write %u30, %memref3[%c0, %random_index] : vector<4xf32>, memref<?x?xf32>
      vector.transfer_write %u31, %memref3[%c1, %random_index] : vector<4xf32>, memref<?x?xf32>
      vector.transfer_write %u10, %memref0[%i, %i] : vector<2xf32>, memref<?x?xf32>
      vector.transfer_write %u11, %memref0[%random_index, %random_index] : vector<2xf32>, memref<?x?xf32>
    }
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_in_affine_loops(
//  CHECK-SAME:   %[[MEMREF0:[a-zA-Z0-9]+]]: memref<64x64xi32>,
//  CHECK-SAME:   %[[MEMREF1:[a-zA-Z0-9]+]]: memref<64x64xi32>,
//  CHECK-SAME:   %[[MEMREF2:[a-zA-Z0-9]+]]: memref<64x64xi32>) {
//       CHECK:   %[[C0:.*]] = arith.constant 0 : i32
//       CHECK:   affine.for %[[I:.*]] = 0 to 64 {
//       CHECK:     affine.for %[[J:.*]] = 0 to 64 step 16 {
//       CHECK:       %[[R0:.*]] = vector.transfer_read %[[MEMREF2]][%[[I]], %[[J]]], %[[C0]] : memref<64x64xi32>, vector<16xi32>
//       CHECK:       %[[R:.*]] = affine.for %[[K:.*]] = 0 to 64 iter_args(%[[ACC:.*]] = %[[R0]]) -> (vector<16xi32>) {
//       CHECK:         %[[AV:.*]] = vector.transfer_read %[[MEMREF0]][%[[I]], %[[K]]], %[[C0]] {{.*}}: memref<64x64xi32>, vector<16xi32>
//       CHECK:         %[[BV:.*]] = vector.transfer_read %[[MEMREF1]][%[[K]], %[[J]]], %[[C0]] {{.*}}: memref<64x64xi32>, vector<16xi32>
//       CHECK:         %[[T0:.*]] = arith.muli %[[AV]], %[[BV]] : vector<16xi32>
//       CHECK:         %[[T1:.*]] = arith.addi %[[ACC]], %[[T0]] : vector<16xi32>
//       CHECK:         affine.yield %[[T1]] : vector<16xi32>
//       CHECK:       }
//       CHECK:       vector.transfer_write %[[R]], %[[MEMREF2]][%[[I]], %[[J]]] : vector<16xi32>, memref<64x64xi32>
//       CHECK:     }
//       CHECK:   }
func.func @hoist_vector_transfer_pairs_in_affine_loops(%memref0: memref<64x64xi32>, %memref1: memref<64x64xi32>, %memref2: memref<64x64xi32>) {
  %c0_i32 = arith.constant 0 : i32
  affine.for %arg3 = 0 to 64 {
    affine.for %arg4 = 0 to 64 step 16 {
      affine.for %arg5 = 0 to 64 {
        %0 = vector.transfer_read %memref0[%arg3, %arg5], %c0_i32 {permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
        %1 = vector.transfer_read %memref1[%arg5, %arg4], %c0_i32 : memref<64x64xi32>, vector<16xi32>
        %2 = vector.transfer_read %memref2[%arg3, %arg4], %c0_i32 : memref<64x64xi32>, vector<16xi32>
        %3 = arith.muli %0, %1 : vector<16xi32>
        %4 = arith.addi %2, %3 : vector<16xi32>
        vector.transfer_write %4, %memref2[%arg3, %arg4] : vector<16xi32>, memref<64x64xi32>
      }
    }
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_tensor
func.func @hoist_vector_transfer_pairs_tensor(
    %tensor0: tensor<?x?xf32>, %tensor1: tensor<?x?xf32>, %tensor2: tensor<?x?xf32>,
    %tensor3: tensor<?x?xf32>, %tensor4: tensor<?x?xf32>, %tensor5: tensor<?x?xf32>,
    %val: index, %lb : index, %ub : index, %step: index) ->
    (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
     tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32

// CHECK: vector.transfer_read %{{.*}} : tensor<?x?xf32>, vector<1xf32>
// CHECK: scf.for {{.*}} iter_args({{.*}}) ->
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<1xf32>) {
// CHECK:   vector.transfer_read %{{.*}} : tensor<?x?xf32>, vector<2xf32>
// CHECK:   scf.for {{.*}} iter_args({{.*}}) ->
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<2xf32>, vector<1xf32>) {
// CHECK:     vector.transfer_read %{{.*}} : tensor<?x?xf32>, vector<4xf32>
// CHECK:     "some_crippling_use"(%{{.*}}) : (tensor<?x?xf32>) -> ()
// CHECK:     vector.transfer_read %{{.*}} : tensor<?x?xf32>, vector<5xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<1xf32>) -> vector<1xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (tensor<?x?xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<5xf32>) -> vector<5xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<3xf32>, tensor<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<4xf32>, tensor<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<5xf32>, tensor<?x?xf32>
// CHECK:     "some_crippling_use"(%{{.*}}) : (tensor<?x?xf32>) -> ()
// CHECK:     scf.yield {{.*}} :
// CHECK-SAME: tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<2xf32>, vector<1xf32>
// CHECK:   }
// CHECK:   vector.transfer_write %{{.*}} : vector<2xf32>, tensor<?x?xf32>
// CHECK:   scf.yield {{.*}} :
// CHECK-SAME: tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<1xf32>
// CHECK: }
// CHECK: vector.transfer_write %{{.*}} : vector<1xf32>, tensor<?x?xf32>
  %0:6 = scf.for %i = %lb to %ub step %step
  iter_args(%arg0 = %tensor0, %arg1 = %tensor1, %arg2 = %tensor2,
            %arg3 = %tensor3,  %arg4 = %tensor4, %arg5 = %tensor5)
  -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
     tensor<?x?xf32>, tensor<?x?xf32>)  {
    %1:6 = scf.for %j = %lb to %ub step %step
    iter_args(%arg6 = %arg0, %arg7 = %arg1, %arg8 = %arg2,
              %arg9 = %arg3,  %arg10 = %arg4, %arg11 = %arg5)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
       tensor<?x?xf32>, tensor<?x?xf32>)  {
      %r0 = vector.transfer_read %arg7[%c0, %c0], %cst: tensor<?x?xf32>, vector<1xf32>
      %r1 = vector.transfer_read %arg6[%i, %i], %cst: tensor<?x?xf32>, vector<2xf32>
      %r3 = vector.transfer_read %arg9[%c0, %c0], %cst: tensor<?x?xf32>, vector<4xf32>
      "some_crippling_use"(%arg10) : (tensor<?x?xf32>) -> ()
      %r4 = vector.transfer_read %arg10[%c0, %c0], %cst: tensor<?x?xf32>, vector<5xf32>
      %r5 = vector.transfer_read %arg11[%c0, %c0], %cst: tensor<?x?xf32>, vector<6xf32>
      "some_crippling_use"(%arg11) : (tensor<?x?xf32>) -> ()
      %u0 = "some_use"(%r0) : (vector<1xf32>) -> vector<1xf32>
      %u1 = "some_use"(%r1) : (vector<2xf32>) -> vector<2xf32>
      %u2 = "some_use"(%arg8) : (tensor<?x?xf32>) -> vector<3xf32>
      %u3 = "some_use"(%r3) : (vector<4xf32>) -> vector<4xf32>
      %u4 = "some_use"(%r4) : (vector<5xf32>) -> vector<5xf32>
      %u5 = "some_use"(%r5) : (vector<6xf32>) -> vector<6xf32>
      %w1 = vector.transfer_write %u0, %arg7[%c0, %c0] : vector<1xf32>, tensor<?x?xf32>
      %w0 = vector.transfer_write %u1, %arg6[%i, %i] : vector<2xf32>, tensor<?x?xf32>
      %w2 = vector.transfer_write %u2, %arg8[%c0, %c0] : vector<3xf32>, tensor<?x?xf32>
      %w3 = vector.transfer_write %u3, %arg9[%c0, %c0] : vector<4xf32>, tensor<?x?xf32>
      %w4 = vector.transfer_write %u4, %arg10[%c0, %c0] : vector<5xf32>, tensor<?x?xf32>
      %w5 = vector.transfer_write %u5, %arg11[%c0, %c0] : vector<6xf32>, tensor<?x?xf32>
      "some_crippling_use"(%w3) : (tensor<?x?xf32>) -> ()
      scf.yield %w0, %w1, %w2, %w3, %w4, %w5 :
        tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
        tensor<?x?xf32>, tensor<?x?xf32>
      }
      scf.yield %1#0,  %1#1, %1#2, %1#3, %1#4, %1#5 :
        tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
        tensor<?x?xf32>, tensor<?x?xf32>
  }
  return %0#0,  %0#1, %0#2, %0#3, %0#4,  %0#5 :
        tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
        tensor<?x?xf32>, tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_tensor_subsets %0
    : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_disjoint_tensor(
//  CHECK-SAME:   %[[TENSOR0:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR1:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR2:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR3:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
func.func @hoist_vector_transfer_pairs_disjoint_tensor(
    %tensor0: tensor<?x?xf32>, %tensor1: tensor<?x?xf32>,
    %tensor2: tensor<?x?xf32>, %tensor3: tensor<?x?xf32>,
    %val: index, %lb : index, %ub : index, %step: index,
    %random_index : index) ->
    (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.0 : f32

// CHECK: vector.transfer_read %[[TENSOR2]]{{.*}} : tensor<?x?xf32>, vector<3xf32>
// CHECK: vector.transfer_read %[[TENSOR2]]{{.*}} : tensor<?x?xf32>, vector<3xf32>
// CHECK: vector.transfer_read %[[TENSOR3]]{{.*}} : tensor<?x?xf32>, vector<4xf32>
// CHECK: vector.transfer_read %[[TENSOR3]]{{.*}} : tensor<?x?xf32>, vector<4xf32>
// CHECK: %[[R:.*]]:6 = scf.for {{.*}} iter_args({{.*}}) ->
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>) {
// CHECK:   scf.for {{.*}} iter_args({{.*}}) ->
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>) {
// CHECK:     vector.transfer_read %[[TENSOR1]]{{.*}} : tensor<?x?xf32>, vector<2xf32>
// CHECK:     vector.transfer_read %[[TENSOR1]]{{.*}} : tensor<?x?xf32>, vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<3xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<3xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     vector.transfer_write %{{.*}}, %{{.*}}{{.*}} : vector<2xf32>, tensor<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}}, %{{.*}}{{.*}} : vector<2xf32>, tensor<?x?xf32>
// CHECK:     scf.yield {{.*}} :
// CHECK-SAME: tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>
// CHECK:   }
// CHECK:   scf.yield {{.*}} :
// CHECK-SAME: tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>
// CHECK: }
// CHECK: %[[TENSOR4:.*]] = vector.transfer_write %[[R]]#5, %[[TENSOR3]]{{.*}} : vector<4xf32>, tensor<?x?xf32>
// CHECK:                   vector.transfer_write %[[R]]#4, %[[TENSOR4]]{{.*}} : vector<4xf32>, tensor<?x?xf32>
// CHECK: %[[TENSOR5:.*]] = vector.transfer_write %[[R]]#3, %[[TENSOR2]]{{.*}} : vector<3xf32>, tensor<?x?xf32>
// CHECK:                   vector.transfer_write %[[R]]#2, %[[TENSOR5]]{{.*}} : vector<3xf32>, tensor<?x?xf32>
  %0:4 = scf.for %i = %lb to %ub step %step
  iter_args(%arg0 = %tensor0, %arg1 = %tensor1, %arg2 = %tensor2,
            %arg3 = %tensor3)
  -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
    %1:4 = scf.for %j = %lb to %ub step %step
    iter_args(%arg4 = %arg0, %arg5 = %arg1, %arg6 = %arg2,
              %arg7 = %arg3)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
      %r00 = vector.transfer_read %arg5[%c0, %c0], %cst: tensor<?x?xf32>, vector<2xf32>
      %r01 = vector.transfer_read %arg5[%c0, %c1], %cst: tensor<?x?xf32>, vector<2xf32>
      %r20 = vector.transfer_read %arg6[%c0, %c0], %cst: tensor<?x?xf32>, vector<3xf32>
      %r21 = vector.transfer_read %arg6[%c0, %c3], %cst: tensor<?x?xf32>, vector<3xf32>
      %r30 = vector.transfer_read %arg7[%c0, %random_index], %cst: tensor<?x?xf32>, vector<4xf32>
      %r31 = vector.transfer_read %arg7[%c1, %random_index], %cst: tensor<?x?xf32>, vector<4xf32>
      %r10 = vector.transfer_read %arg4[%i, %i], %cst: tensor<?x?xf32>, vector<2xf32>
      %r11 = vector.transfer_read %arg4[%random_index, %random_index], %cst: tensor<?x?xf32>, vector<2xf32>
      %u00 = "some_use"(%r00) : (vector<2xf32>) -> vector<2xf32>
      %u01 = "some_use"(%r01) : (vector<2xf32>) -> vector<2xf32>
      %u20 = "some_use"(%r20) : (vector<3xf32>) -> vector<3xf32>
      %u21 = "some_use"(%r21) : (vector<3xf32>) -> vector<3xf32>
      %u30 = "some_use"(%r30) : (vector<4xf32>) -> vector<4xf32>
      %u31 = "some_use"(%r31) : (vector<4xf32>) -> vector<4xf32>
      %u10 = "some_use"(%r10) : (vector<2xf32>) -> vector<2xf32>
      %u11 = "some_use"(%r11) : (vector<2xf32>) -> vector<2xf32>
      %w10 = vector.transfer_write %u00, %arg5[%c0, %c0] : vector<2xf32>, tensor<?x?xf32>
      %w11 = vector.transfer_write %u01, %w10[%c0, %c1] : vector<2xf32>, tensor<?x?xf32>
      %w20 = vector.transfer_write %u20, %arg6[%c0, %c0] : vector<3xf32>, tensor<?x?xf32>
      %w21 = vector.transfer_write %u21, %w20[%c0, %c3] : vector<3xf32>, tensor<?x?xf32>
      %w30 = vector.transfer_write %u30, %arg7[%c0, %random_index] : vector<4xf32>, tensor<?x?xf32>
      %w31 = vector.transfer_write %u31, %w30[%c1, %random_index] : vector<4xf32>, tensor<?x?xf32>
      %w00 = vector.transfer_write %u10, %arg4[%i, %i] : vector<2xf32>, tensor<?x?xf32>
      %w01 = vector.transfer_write %u11, %w00[%random_index, %random_index] : vector<2xf32>, tensor<?x?xf32>
      scf.yield %w01, %w11, %w21, %w31 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
    }
    scf.yield %1#0,  %1#1, %1#2, %1#3 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
  }
  return %0#0,  %0#1, %0#2, %0#3 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_tensor_subsets %0
    : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_tensor_and_slices
//  CHECK-SAME:   %[[TENSOR0:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR1:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR2:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR3:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR4:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR5:[a-zA-Z0-9]*]]: tensor<?x?xf32>
func.func @hoist_vector_transfer_pairs_tensor_and_slices(
    %tensor0: tensor<?x?xf32>, %tensor1: tensor<?x?xf32>, %tensor2: tensor<?x?xf32>,
    %tensor3: tensor<?x?xf32>, %tensor4: tensor<?x?xf32>, %tensor5: tensor<?x?xf32>,
    %val: index, %lb : index, %ub : index, %step: index) ->
    (
      tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>//, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
    ) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32

  //      CHECK: scf.for %[[I:.*]] = {{.*}} iter_args(
  // CHECK-SAME:   %[[TENSOR0_ARG:[0-9a-zA-Z]+]] = %[[TENSOR0]],
  // CHECK-SAME:   %[[TENSOR1_ARG:[0-9a-zA-Z]+]] = %[[TENSOR1]],
  // CHECK-SAME:   %[[TENSOR2_ARG:[0-9a-zA-Z]+]] = %[[TENSOR2]]
  // CHECK-SAME: ) ->
  // CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
  %0:3 = scf.for %i = %lb to %ub step %step
  iter_args(%arg0 = %tensor0, %arg1 = %tensor1, %arg2 = %tensor2)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)  {

    // Hoisted
    // CHECK:   %[[ST0:.*]] = tensor.extract_slice %[[TENSOR0_ARG]][%[[I]], %[[I]]]{{.*}}: tensor<?x?xf32> to tensor<?x?xf32>
    // CHECK:   %[[V0:.*]] = vector.transfer_read %[[ST0]]{{.*}} : tensor<?x?xf32>, vector<1xf32>

    //      CHECK:   %[[R:.*]]:3 = scf.for %[[J:.*]] = {{.*}} iter_args(
    // CHECK-SAME:   %[[TENSOR1_ARG_L2:[0-9a-zA-Z]+]] = %[[TENSOR1_ARG]]
    // CHECK-SAME:   %[[TENSOR2_ARG_L2:[0-9a-zA-Z]+]] = %[[TENSOR2_ARG]]
    // CHECK-SAME:   %[[V0_ARG_L2:[0-9a-zA-Z]+]] = %[[V0]]
    // CHECK-SAME: ) ->
    // CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, vector<1xf32>
    %1:3 = scf.for %j = %lb to %ub step %step
    iter_args(%arg6 = %arg0, %arg7 = %arg1, %arg8 = %arg2)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)  {
      // Hoists.
      %st0 = tensor.extract_slice %arg6[%i, %i][%step, %step][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %r0 = vector.transfer_read %st0[%c0, %c0], %cst: tensor<?x?xf32>, vector<1xf32>

      // CHECK:     %[[ST1:.*]] = tensor.extract_slice %[[TENSOR1_ARG_L2]][%[[J]],{{.*}}: tensor<?x?xf32> to tensor<?x?xf32>
      // CHECK:     %[[V1:.*]] = vector.transfer_read %[[ST1]]{{.*}} : tensor<?x?xf32>, vector<2xf32>
      // Does not hoist (slice depends on %j)
      %st1 = tensor.extract_slice %arg7[%j, %c0][%step, %step][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %r1 = vector.transfer_read %st1[%c0, %c0], %cst: tensor<?x?xf32>, vector<2xf32>

      // CHECK:     %[[ST2:.*]] = tensor.extract_slice %[[TENSOR2_ARG_L2]][%[[I]],{{.*}}: tensor<?x?xf32> to tensor<?x?xf32>
      // CHECK:     %[[V2:.*]] = vector.transfer_read %[[ST2]]{{.*}} : tensor<?x?xf32>, vector<3xf32>
      // Does not hoist, 2 slice %arg8.
      %st2 = tensor.extract_slice %arg8[%i, %c0][%step, %step][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %r2 = vector.transfer_read %st2[%c0, %c0], %cst: tensor<?x?xf32>, vector<3xf32>

      // CHECK:     %[[U0:.*]] = "some_use"(%[[V0_ARG_L2]]) : (vector<1xf32>) -> vector<1xf32>
      // CHECK:     %[[U1:.*]] = "some_use"(%[[V1]]) : (vector<2xf32>) -> vector<2xf32>
      // CHECK:     %[[U2:.*]] = "some_use"(%[[V2]]) : (vector<3xf32>) -> vector<3xf32>
      %u0 = "some_use"(%r0) : (vector<1xf32>) -> vector<1xf32>
      %u1 = "some_use"(%r1) : (vector<2xf32>) -> vector<2xf32>
      %u2 = "some_use"(%r2) : (vector<3xf32>) -> vector<3xf32>

      // Hoists
      %w0 = vector.transfer_write %u0, %st0[%c0, %c0] : vector<1xf32>, tensor<?x?xf32>

      // CHECK-DAG:     %[[STI1:.*]] = vector.transfer_write %[[U1]], %{{.*}} : vector<2xf32>, tensor<?x?xf32>
      // Does not hoist (associated slice depends on %j).
      %w1 = vector.transfer_write %u1, %st1[%i, %i] : vector<2xf32>, tensor<?x?xf32>

      // CHECK-DAG:     %[[STI2:.*]] = vector.transfer_write %[[U2]], %{{.*}} : vector<3xf32>, tensor<?x?xf32>
      // Does not hoist, 2 slice / insert_slice for %arg8.
      %w2 = vector.transfer_write %u2, %st2[%c0, %c0] : vector<3xf32>, tensor<?x?xf32>

      // Hoists.
      %sti0 = tensor.insert_slice %w0 into %arg6[%i, %i][%step, %step][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

      // CHECK-DAG:     tensor.insert_slice %[[STI1]] into %[[TENSOR1_ARG_L2]][%[[J]],{{.*}}: tensor<?x?xf32> into tensor<?x?xf32>
      // Does not hoist (depends on %j).
      %sti1 = tensor.insert_slice %w1 into %arg7[%j, %c0][%step, %step][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

      // CHECK-DAG:     tensor.insert_slice %[[STI2]] into %[[TENSOR2_ARG_L2]][%[[I]],{{.*}}: tensor<?x?xf32> into tensor<?x?xf32>
      // Does not hoist, 2 slice / insert_slice for %arg8.
      %sti2 = tensor.insert_slice %w2 into %arg8[%i, %c0][%step, %step][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
      // Extract with a different stride to make sure we cannot fold this extract with the above insert.
      %st22 = tensor.extract_slice %sti2[%i, %c0][%step, %step][2, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %sti22 = tensor.insert_slice %st22 into %arg8[%i, %c0][%step, %step][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

      // CHECK:     scf.yield {{.*}} : tensor<?x?xf32>, tensor<?x?xf32>, vector<1xf32>
      // CHECK:   }
      scf.yield %sti0, %sti1, %sti22:
        tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
    }

    // Hoisted
    // CHECK:   %[[STI0:.*]] = vector.transfer_write %[[R]]#2, %[[ST0]]{{.*}} : vector<1xf32>, tensor<?x?xf32>
    // CHECK:   tensor.insert_slice %[[STI0]] into %[[TENSOR0_ARG]][%[[I]], %[[I]]]{{.*}} : tensor<?x?xf32> into tensor<?x?xf32>

    // CHECK:   scf.yield {{.*}} : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
    scf.yield %1#0, %1#1, %1#2 :
      tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>

    // CHECK: }
  }
  return %0#0, %0#1, %0#2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_tensor_subsets %0
    : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_write_pairs_disjoint_tensor(
//  CHECK-SAME:   %[[T:.*]]: tensor<?x?xf32>,
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[R0:.*]] = vector.transfer_read %[[T]][%[[C0]], %[[C0]]], %{{.*}} : tensor<?x?xf32>, vector<2xf32>
//   CHECK-DAG:   %[[R1:.*]] = vector.transfer_read %[[T]][%[[C0]], %[[C3]]], %{{.*}} : tensor<?x?xf32>, vector<2xf32>
//       CHECK:   %[[F:.*]]:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[R3:.*]] = %[[R1:.*]], %[[R2:.*]] = %[[R0]]) -> (vector<2xf32>, vector<2xf32>) {
//       CHECK:     %[[R4:.*]] = "some_use"(%[[R2]]) : (vector<2xf32>) -> vector<2xf32>
//       CHECK:     %[[R5:.*]] = "some_use"(%[[R3]]) : (vector<2xf32>) -> vector<2xf32>
//       CHECK:     scf.yield %[[R5]], %[[R4]] : vector<2xf32>, vector<2xf32>
//       CHECK:   }
//       CHECK:   %[[W0:.*]] = vector.transfer_write %[[F]]#1, %[[T]][%[[C0]], %[[C0]]] : vector<2xf32>, tensor<?x?xf32>
//       CHECK:   %[[W1:.*]] = vector.transfer_write %[[F]]#0, %[[W0]][%[[C0]], %[[C3]]] : vector<2xf32>, tensor<?x?xf32>
//       CHECK:  return %[[W1]] : tensor<?x?xf32>
func.func @hoist_vector_transfer_write_pairs_disjoint_tensor(
    %tensor: tensor<?x?xf32>,
    %val: index, %lb : index, %ub : index, %step: index) ->
    (tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.0 : f32
  %1 = scf.for %j = %lb to %ub step %step iter_args(%arg5 = %tensor)
    -> (tensor<?x?xf32>) {
    %r00 = vector.transfer_read %arg5[%c0, %c0], %cst: tensor<?x?xf32>, vector<2xf32>
    %u00 = "some_use"(%r00) : (vector<2xf32>) -> vector<2xf32>
    %w10 = vector.transfer_write %u00, %arg5[%c0, %c0] : vector<2xf32>, tensor<?x?xf32>

    // Hoist by properly bypassing the disjoint write %w10.
    %r01 = vector.transfer_read %w10[%c0, %c3], %cst: tensor<?x?xf32>, vector<2xf32>
    %u01 = "some_use"(%r01) : (vector<2xf32>) -> vector<2xf32>
    %w11 = vector.transfer_write %u01, %w10[%c0, %c3] : vector<2xf32>, tensor<?x?xf32>
    scf.yield %w11 : tensor<?x?xf32>
  }
  return %1 : tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_tensor_subsets %0
    : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_tensor_and_slices_static_large_tensor
//  CHECK-SAME:   %[[TENSOR0:[a-zA-Z0-9]*]]: tensor<100x100xf32>,
//  CHECK-SAME:   %[[TENSOR1:[a-zA-Z0-9]*]]: tensor<200x200xf32>,
//  CHECK-SAME:   %[[TENSOR2:[a-zA-Z0-9]*]]: tensor<300x300xf32>
func.func @hoist_vector_transfer_pairs_tensor_and_slices_static_large_tensor(
    %tensor0: tensor<100x100xf32>, %tensor1: tensor<200x200xf32>, %tensor2: tensor<300x300xf32>,
    %val: index, %lb : index, %ub : index, %step: index) ->
    (
      tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>
    ) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32

  //      CHECK: scf.for %[[I:.*]] = {{.*}} iter_args(
  // CHECK-SAME:   %[[TENSOR0_ARG:[0-9a-zA-Z]+]] = %[[TENSOR0]],
  // CHECK-SAME:   %[[TENSOR1_ARG:[0-9a-zA-Z]+]] = %[[TENSOR1]],
  // CHECK-SAME:   %[[TENSOR2_ARG:[0-9a-zA-Z]+]] = %[[TENSOR2]]
  // CHECK-SAME: ) ->
  // CHECK-SAME: (tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>
  %0:3 = scf.for %i = %lb to %ub step %step
  iter_args(%arg0 = %tensor0, %arg1 = %tensor1, %arg2 = %tensor2)
    -> (tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>)  {

    // Hoisted
    // CHECK:   %[[ST0:.*]] = tensor.extract_slice %[[TENSOR0_ARG]][%[[I]], %[[I]]]{{.*}}: tensor<100x100xf32> to tensor<?x?xf32>
    // CHECK:   %[[V0:.*]] = vector.transfer_read %[[ST0]]{{.*}} : tensor<?x?xf32>, vector<1xf32>

    //      CHECK:   %[[R:.*]]:3 = scf.for %[[J:.*]] = {{.*}} iter_args(
    // CHECK-SAME:   %[[TENSOR1_ARG_L2:[0-9a-zA-Z]+]] = %[[TENSOR1_ARG]]
    // CHECK-SAME:   %[[TENSOR2_ARG_L2:[0-9a-zA-Z]+]] = %[[TENSOR2_ARG]]
    // CHECK-SAME:   %[[V0_ARG_L2:[0-9a-zA-Z]+]] = %[[V0]]
    // CHECK-SAME: ) ->
    // CHECK-SAME: (tensor<200x200xf32>, tensor<300x300xf32>, vector<1xf32>
    %1:3 = scf.for %j = %lb to %ub step %step
    iter_args(%arg6 = %arg0, %arg7 = %arg1, %arg8 = %arg2)
    -> (tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>)  {
      // Hoists.
      %st0 = tensor.extract_slice %arg6[%i, %i][%step, %step][1, 1] : tensor<100x100xf32> to tensor<?x?xf32>
      %r0 = vector.transfer_read %st0[%c0, %c0], %cst: tensor<?x?xf32>, vector<1xf32>

      // CHECK:     %[[ST1:.*]] = tensor.extract_slice %[[TENSOR1_ARG_L2]][%[[J]],{{.*}}: tensor<200x200xf32> to tensor<?x?xf32>
      // CHECK:     %[[V1:.*]] = vector.transfer_read %[[ST1]]{{.*}} : tensor<?x?xf32>, vector<2xf32>
      // Does not hoist (slice depends on %j)
      %st1 = tensor.extract_slice %arg7[%j, %c0][%step, %step][1, 1] : tensor<200x200xf32> to tensor<?x?xf32>
      %r1 = vector.transfer_read %st1[%c0, %c0], %cst: tensor<?x?xf32>, vector<2xf32>

      // CHECK:     %[[ST2:.*]] = tensor.extract_slice %[[TENSOR2_ARG_L2]][%[[I]],{{.*}}: tensor<300x300xf32> to tensor<?x?xf32>
      // CHECK:     %[[V2:.*]] = vector.transfer_read %[[ST2]]{{.*}} : tensor<?x?xf32>, vector<3xf32>
      // Does not hoist, 2 slice %arg8.
      %st2 = tensor.extract_slice %arg8[%i, %c0][%step, %step][1, 1] : tensor<300x300xf32> to tensor<?x?xf32>
      %r2 = vector.transfer_read %st2[%c0, %c0], %cst: tensor<?x?xf32>, vector<3xf32>

      // CHECK:     %[[U0:.*]] = "some_use"(%[[V0_ARG_L2]]) : (vector<1xf32>) -> vector<1xf32>
      // CHECK:     %[[U1:.*]] = "some_use"(%[[V1]]) : (vector<2xf32>) -> vector<2xf32>
      // CHECK:     %[[U2:.*]] = "some_use"(%[[V2]]) : (vector<3xf32>) -> vector<3xf32>
      %u0 = "some_use"(%r0) : (vector<1xf32>) -> vector<1xf32>
      %u1 = "some_use"(%r1) : (vector<2xf32>) -> vector<2xf32>
      %u2 = "some_use"(%r2) : (vector<3xf32>) -> vector<3xf32>

      // Hoists
      %w0 = vector.transfer_write %u0, %st0[%c0, %c0] : vector<1xf32>, tensor<?x?xf32>

      // CHECK-DAG:     %[[STI1:.*]] = vector.transfer_write %[[U1]], %{{.*}} : vector<2xf32>, tensor<?x?xf32>
      // Does not hoist (associated slice depends on %j).
      %w1 = vector.transfer_write %u1, %st1[%i, %i] : vector<2xf32>, tensor<?x?xf32>

      // CHECK-DAG:     %[[STI2:.*]] = vector.transfer_write %[[U2]], %{{.*}} : vector<3xf32>, tensor<?x?xf32>
      // Does not hoist, 2 slice / insert_slice for %arg8.
      %w2 = vector.transfer_write %u2, %st2[%c0, %c0] : vector<3xf32>, tensor<?x?xf32>

      // Hoists.
      %sti0 = tensor.insert_slice %w0 into %arg6[%i, %i][%step, %step][1, 1] : tensor<?x?xf32> into tensor<100x100xf32>

      // CHECK-DAG:     tensor.insert_slice %[[STI1]] into %[[TENSOR1_ARG_L2]][%[[J]],{{.*}}: tensor<?x?xf32> into tensor<200x200xf32>
      // Does not hoist (depends on %j).
      %sti1 = tensor.insert_slice %w1 into %arg7[%j, %c0][%step, %step][1, 1] : tensor<?x?xf32> into tensor<200x200xf32>

      // CHECK-DAG:     tensor.insert_slice %[[STI2]] into %[[TENSOR2_ARG_L2]][%[[I]],{{.*}}: tensor<?x?xf32> into tensor<300x300xf32>
      // Does not hoist, 2 slice / insert_slice for %arg8.
      %sti2 = tensor.insert_slice %w2 into %arg8[%i, %c0][%step, %step][1, 1] : tensor<?x?xf32> into tensor<300x300xf32>
      // Extract with a different stride to make sure we cannot fold this extract with the above insert.
      %st22 = tensor.extract_slice %sti2[%i, %c0][%step, %step][2, 1] : tensor<300x300xf32> to tensor<?x?xf32>
      %sti22 = tensor.insert_slice %st22 into %arg8[%i, %c0][%step, %step][1, 1] : tensor<?x?xf32> into tensor<300x300xf32>

      // CHECK:     scf.yield {{.*}} : tensor<200x200xf32>, tensor<300x300xf32>, vector<1xf32>
      // CHECK:   }
      scf.yield %sti0, %sti1, %sti22:
        tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>
    }

    // Hoisted
    // CHECK:   %[[STI0:.*]] = vector.transfer_write %[[R]]#2, %[[ST0]]{{.*}} : vector<1xf32>, tensor<?x?xf32>
    // CHECK:   tensor.insert_slice %[[STI0]] into %[[TENSOR0_ARG]][%[[I]], %[[I]]]{{.*}} : tensor<?x?xf32> into tensor<100x100xf32>

    // CHECK:   scf.yield {{.*}} : tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>
    scf.yield %1#0, %1#1, %1#2 :
      tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>

    // CHECK: }
  }
  return %0#0, %0#1, %0#2 : tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_tensor_subsets %0
    : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL:  func.func @hoist_vector_transfer_read(
// CHECK-DAG:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:      %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:      %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:      %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:          %[[ALLOC:.+]] = memref.alloc() : memref<32x64xf32>
// CHECK:          %[[ALLOC_0:.+]] = memref.alloc() : memref<32x128xf32>
// CHECK:          %[[CAST:.+]] = memref.cast %[[ALLOC_0]] : memref<32x128xf32> to memref<32x128xf32, strided<[128, 1],
// CHECK-SAME:       offset: ?>>
// CHECK:          %[[D0:.+]] = vector.transfer_read %[[ALLOC]][%[[C0]], %[[C0]]], %[[CST]] {in_bounds = [true, true]} :
// CHECK-SAME:       memref<32x64xf32>, vector<32x64xf32>
// CHECK:          scf.for %[[ARG0:.+]] = %[[C0]] to %[[C1024]] step %[[C128]] {
// CHECK:            %[[D1:.+]] = vector.transfer_read %[[ALLOC_0]][%[[C0]], %[[C0]]], %[[CST]] {in_bounds = [true, true]}
// CHECK-SAME:         : memref<32x128xf32>, vector<32x128xf32>
// CHECK:            "some_use"(%[[D0]], %[[D1]], %[[CAST]]) : (vector<32x64xf32>, vector<32x128xf32>, memref<32x128xf32,
// CHECK-SAME:         strided<[128, 1], offset: ?>>) -> ()
// CHECK:          }
// CHECK:          memref.dealloc %[[ALLOC]] : memref<32x64xf32>
// CHECK:          return
func.func @hoist_vector_transfer_read() {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %cst_2 = arith.constant 0.000000e+00 : f32
  %memref0 = memref.alloc() : memref<32x64xf32>
  %memref2 = memref.alloc() : memref<32x128xf32>
  %subview2 = memref.subview %memref2[%c0, %c0] [32, 128] [1, 1]: memref<32x128xf32> to memref<32x128xf32, strided<[128, 1], offset: ?>>
  scf.for %arg0 = %c0 to %c1024 step %c128 {
    %2 = vector.transfer_read %memref2[%c0, %c0], %cst_2 {in_bounds = [true, true]} : memref<32x128xf32>, vector<32x128xf32>
    %3 = vector.transfer_read %memref0[%c0, %c0], %cst_2 {in_bounds = [true, true]} : memref<32x64xf32>, vector<32x64xf32>
    "some_use"(%3, %2, %subview2) : (vector<32x64xf32>, vector<32x128xf32>, memref<32x128xf32, strided<[128, 1], offset: ?>>) -> ()
  }
  memref.dealloc %memref0 : memref<32x64xf32>
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}

// -----

// The transfers in this test case cannot be hoisted and replaced by a vector
// iter_arg because they do not match.

// CHECK-LABEL:  func.func @non_matching_transfers(
//       CHECK:    scf.for {{.*}} {
//       CHECK:      vector.transfer_read
//       CHECK:      vector.transfer_write
//       CHECK:    }
func.func @non_matching_transfers(%m: memref<6x1x7x32xf32>) {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant dense<5.5> : vector<6x7x32xf32>
  %cst_0 = arith.constant 0.0 : f32
  scf.for %iv = %c0 to %c1024 step %c128 {
    %read = vector.transfer_read %m[%c0, %c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>} : memref<6x1x7x32xf32>, vector<6x7x32xf32>
    %added = arith.addf %read, %cst : vector<6x7x32xf32>
    %bc = vector.broadcast %added : vector<6x7x32xf32> to vector<1x6x7x32xf32>
    %tr = vector.transpose %bc, [1, 0, 2, 3] : vector<1x6x7x32xf32> to vector<6x1x7x32xf32>
    vector.transfer_write %tr, %m[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<6x1x7x32xf32>, memref<6x1x7x32xf32>
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}

// -----

// Regression test - `vector.transfer_read` below should not be hoisted.
// Indeed, %collapse_shape (written to by `vector.transfer_write`) and %alloca
// (read by `vector.transfer_read`) alias.

// CHECK-LABEL:  func.func @no_hoisting_collapse_shape
//       CHECK:    scf.for {{.*}} {
//       CHECK:      vector.transfer_write {{.*}} : vector<4xi32>, memref<4xi32>
//       CHECK-NEXT:      vector.transfer_read {{.*}} : memref<1x4x1xi32>, vector<1x4x1xi32>
//       CHECK-NEXT:      vector.transfer_write {{.*}} : vector<1x4x1xi32>, memref<1x4x1xi32, strided<[20, 1, 1], offset: ?>>
//       CHECK-NEXT:    }

func.func @no_hoisting_collapse_shape(%in_0: memref<1x20x1xi32>, %1: memref<9x1xi32>, %vec: vector<4xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c20 = arith.constant 20 : index
  %alloca = memref.alloca() {alignment = 64 : i64} : memref<1x4x1xi32>
  scf.for %arg0 = %c0 to %c20 step %c4 {
    %subview = memref.subview %in_0[0, %arg0, 0] [1, 4, 1] [1, 1, 1] : memref<1x20x1xi32> to memref<1x4x1xi32, strided<[20, 1, 1], offset: ?>>
    %collapse_shape = memref.collapse_shape %alloca [[0, 1, 2]] : memref<1x4x1xi32> into memref<4xi32>
    vector.transfer_write %vec, %collapse_shape[%c0] {in_bounds = [true]} : vector<4xi32>, memref<4xi32>
    %read = vector.transfer_read %alloca[%c0, %c0, %c0], %c0_i32 {in_bounds = [true, true, true]} : memref<1x4x1xi32>, vector<1x4x1xi32>
    vector.transfer_write %read, %subview[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x4x1xi32>, memref<1x4x1xi32, strided<[20, 1, 1], offset: ?>>
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}

// -----

// Regression test - `vector.transfer_read` below should not be hoisted.
// Indeed, %collapse_shape (read by `vector.transfer_read`) and %alloca
// (written to by `vector.transfer_write`) alias.

// CHECK-LABEL:  func.func @no_hoisting_collapse_shape_2
//       CHECK:    scf.for {{.*}} {
//       CHECK:      vector.transfer_write
//       CHECK:      vector.transfer_read

func.func @no_hoisting_collapse_shape_2(%vec: vector<1x12x1xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c20 = arith.constant 20 : index
  %alloca = memref.alloca() {alignment = 64 : i64} : memref<1x12x1xi32>
  scf.for %arg0 = %c0 to %c20 step %c4 {
    %collapse_shape = memref.collapse_shape %alloca [[0, 1, 2]] : memref<1x12x1xi32> into memref<12xi32>
    vector.transfer_write %vec, %alloca[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<1x12x1xi32>, memref<1x12x1xi32>
    %read = vector.transfer_read %collapse_shape[%c0], %c0_i32 {in_bounds = [true]} : memref<12xi32>, vector<12xi32>
    "prevent.dce"(%read) : (vector<12xi32>) ->()
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}

// -----

// Regression test - hoisting the following `vector.transfer_{read|write}` pair
// would not be safe:
//    %lhs = vector.transfer_read %collapsed_1[%c0]
//    vector.transfer_write %op, %collapsed_1[%c0]
// That's because the following `vector.transfer_read` reads from the same
// memory (i.e. `%collapsed_1` and `%collapsed_2` alias):
//    %acc = vector.transfer_read %collapsed_2[%c0]

// CHECK-LABEL:  func.func @no_hoisting_write_to_memref
//       CHECK:    scf.for {{.*}} {
//       CHECK:      vector.transfer_read {{.*}} :  memref<2xi32>, vector<1xi32>
//       CHECK-NEXT:      vector.transfer_read {{.*}} :  memref<2xi32>, vector<1xi32>
//       CHECK-NEXT:      vector.outerproduct {{.*}} : vector<1xi32>, i32
//       CHECK-NEXT:      vector.transfer_write {{.*}} : vector<1xi32>, memref<2xi32>
//       CHECK-NEXT:    }

func.func @no_hoisting_write_to_memref(%rhs: i32, %arg1: vector<1xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c20 = arith.constant 20 : index
  %alloca = memref.alloca() {alignment = 64 : i64} : memref<1x1x2xi32>
  %cast = memref.cast %alloca : memref<1x1x2xi32> to memref<1x1x2xi32>
  %collapsed_1 = memref.collapse_shape %alloca [[0, 1, 2]] : memref<1x1x2xi32> into memref<2xi32>
  scf.for %_ = %c0 to %c20 step %c4 {
    %collapsed_2 = memref.collapse_shape %alloca [[0, 1, 2]] : memref<1x1x2xi32> into memref<2xi32>
    %lhs = vector.transfer_read %collapsed_1[%c0], %c0_i32 {in_bounds = [true]} : memref<2xi32>, vector<1xi32>
    %acc = vector.transfer_read %collapsed_2[%c0], %c0_i32 {in_bounds = [true]} : memref<2xi32>, vector<1xi32>
    %op = vector.outerproduct %lhs, %rhs, %acc {kind = #vector.kind<add>} : vector<1xi32>, i32
    vector.transfer_write %op, %collapsed_1[%c0] {in_bounds = [true]} : vector<1xi32>, memref<2xi32>
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}

// -----

// Test that we can hoist out 1-D read-write pairs whose indices are dynamic values.

// CHECK: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 + 1)>
// CHECK: #[[$MAP4:.+]] = affine_map<()[s0] -> (s0 + 4)>

//   CHECK-LABEL: func.func @hoist_vector_transfer_pairs_disjoint_dynamic
//    CHECK-SAME: (%[[BUFFER:.+]]: memref<?x?xf32>, %{{.+}}: index, %{{.+}}: index, %{{.+}}: index, %[[I0:.+]]: index)

//         CHECK:   %[[PLUS1:.+]] = affine.apply #[[$MAP1]]()[%[[I0]]]
//         CHECK:   %[[PLUS4:.+]] = affine.apply #[[$MAP4]]()[%[[I0]]]
//         CHECK:   %2 = vector.transfer_read %[[BUFFER]][%[[I0]], %[[I0]]]
//         CHECK:   %3 = vector.transfer_read %[[BUFFER]][%[[PLUS1]], %[[I0]]]
//         CHECK:   %4 = vector.transfer_read %[[BUFFER]][%[[PLUS1]], %[[PLUS4]]]
// CHECK-COUNT-2:   scf.for %{{.+}} = {{.+}} -> (vector<4xf32>, vector<4xf32>, vector<4xf32>)
// CHECK-COUNT-3:     "some_use"
// CHECK-COUNT-2:   scf.yield {{.+}} : vector<4xf32>, vector<4xf32>, vector<4xf32>
//         CHECK:   vector.transfer_write %{{.+}}, %[[BUFFER]][%[[PLUS1]], %[[PLUS4]]]
//         CHECK:   vector.transfer_write %{{.+}}, %[[BUFFER]][%[[PLUS1]], %[[I0]]]
//         CHECK:   vector.transfer_write %{{.+}}, %[[BUFFER]][%[[I0]], %[[I0]]]

func.func @hoist_vector_transfer_pairs_disjoint_dynamic(
    %buffer: memref<?x?xf32>, %lb : index, %ub : index, %step: index, %i0 : index) {
  %cst = arith.constant 0.0 : f32
  %i1 = affine.apply affine_map<(d0) -> (d0 + 1)>(%i0)
  %i2 = affine.apply affine_map<(d0) -> (d0 + 4)>(%i0)

  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      %r0 = vector.transfer_read %buffer[%i0, %i0], %cst: memref<?x?xf32>, vector<4xf32>
      // Disjoint leading dim
      %r1 = vector.transfer_read %buffer[%i1, %i0], %cst: memref<?x?xf32>, vector<4xf32>
      // Non-overlap trailing dim
      %r2 = vector.transfer_read %buffer[%i1, %i2], %cst: memref<?x?xf32>, vector<4xf32>
      %u0 = "some_use"(%r0) : (vector<4xf32>) -> vector<4xf32>
      %u1 = "some_use"(%r1) : (vector<4xf32>) -> vector<4xf32>
      %u2 = "some_use"(%r2) : (vector<4xf32>) -> vector<4xf32>
      vector.transfer_write %u0, %buffer[%i0, %i0] : vector<4xf32>, memref<?x?xf32>
      vector.transfer_write %u1, %buffer[%i1, %i0] : vector<4xf32>, memref<?x?xf32>
      vector.transfer_write %u2, %buffer[%i1, %i2] : vector<4xf32>, memref<?x?xf32>
    }
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}

// -----

// Test that we cannot hoist out read-write pairs whose indices are overlapping.

//   CHECK-LABEL: func.func @hoist_vector_transfer_pairs_overlapping_dynamic
// CHECK-COUNT-2:   scf.for
// CHECK-COUNT-2:     vector.transfer_read
// CHECK-COUNT-2:     vector.transfer_write

func.func @hoist_vector_transfer_pairs_overlapping_dynamic(
    %buffer: memref<?x?xf32>, %lb : index, %ub : index, %step: index, %i0 : index) {
  %cst = arith.constant 0.0 : f32
  %i1 = affine.apply affine_map<(d0) -> (d0 + 3)>(%i0)

  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      %r0 = vector.transfer_read %buffer[%i0, %i0], %cst: memref<?x?xf32>, vector<4xf32>
      // Overlapping range with the above
      %r1 = vector.transfer_read %buffer[%i0, %i1], %cst: memref<?x?xf32>, vector<4xf32>
      %u0 = "some_use"(%r0) : (vector<4xf32>) -> vector<4xf32>
      %u1 = "some_use"(%r1) : (vector<4xf32>) -> vector<4xf32>
      vector.transfer_write %u0, %buffer[%i0, %i0] : vector<4xf32>, memref<?x?xf32>
      vector.transfer_write %u1, %buffer[%i0, %i1] : vector<4xf32>, memref<?x?xf32>
    }
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}

// -----

// Test that we can hoist out 2-D read-write pairs whose indices are dynamic values.

//   CHECK-LABEL: func.func @hoist_vector_transfer_pairs_disjoint_dynamic
// CHECK-COUNT-2:   vector.transfer_read
// CHECK-COUNT-2:   %{{.+}}:2 = scf.for {{.+}} -> (vector<16x8xf32>, vector<16x8xf32>)
// CHECK-COUNT-2:   scf.yield {{.+}} : vector<16x8xf32>, vector<16x8xf32>
// CHECK-COUNT-2:   vector.transfer_write
//         CHECK:   return

func.func @hoist_vector_transfer_pairs_disjoint_dynamic(
    %buffer: memref<?x?xf32>, %lb : index, %ub : index, %step: index, %i0 : index, %i1 : index) {
  %cst = arith.constant 0.0 : f32
  %i2 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 16)>(%i1)
  %i3 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 16 + 8)>(%i1)

  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      %r0 = vector.transfer_read %buffer[%i0, %i2], %cst: memref<?x?xf32>, vector<16x8xf32>
      %r1 = vector.transfer_read %buffer[%i0, %i3], %cst: memref<?x?xf32>, vector<16x8xf32>
      %u0 = "some_use"(%r0) : (vector<16x8xf32>) -> vector<16x8xf32>
      %u1 = "some_use"(%r1) : (vector<16x8xf32>) -> vector<16x8xf32>
      vector.transfer_write %u0, %buffer[%i0, %i2] : vector<16x8xf32>, memref<?x?xf32>
      vector.transfer_write %u1, %buffer[%i0, %i3] : vector<16x8xf32>, memref<?x?xf32>
    }
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}
