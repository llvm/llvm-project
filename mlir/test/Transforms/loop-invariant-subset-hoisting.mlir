// RUN: mlir-opt %s  -split-input-file -loop-invariant-subset-hoisting | FileCheck %s

// CHECK-LABEL: func @hoist_matching_extract_insert(
//  CHECK-SAME:     %[[arg:.*]]: tensor<?xf32>
func.func @hoist_matching_extract_insert(%arg: tensor<?xf32>) -> tensor<?xf32> {
  %lb = "test.foo"() : () -> (index)
  %ub = "test.foo"() : () -> (index)
  %step = "test.foo"() : () -> (index)

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %add = arith.addi %c0, %c1 : index
  %sub = arith.subi %add, %c1 : index

  // CHECK: %[[extract:.*]] = tensor.extract_slice %[[arg]]
  // CHECK: %[[for:.*]]:2 = scf.for {{.*}} iter_args(%[[t:.*]] = %[[arg]], %[[hoisted:.*]] = %[[extract]])
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%t = %arg) -> (tensor<?xf32>) {
    // CHECK: tensor.extract_slice %[[t]][9] [5] [1]
    %standalone = tensor.extract_slice %t[9][5][1] : tensor<?xf32> to tensor<5xf32>
    "test.foo"(%standalone) : (tensor<5xf32>) -> ()

    %1 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
    // CHECK: %[[foo:.*]] = "test.foo"(%[[hoisted]])
    %2 = "test.foo"(%1) : (tensor<5xf32>) -> (tensor<5xf32>)
    // Obfuscate the IR by inserting at offset %sub instead of 0; both of them
    // have the same value.
    %3 = tensor.insert_slice %2 into %t[%sub][5][1] : tensor<5xf32> into tensor<?xf32>
    // CHECK: scf.yield %[[t]], %[[foo]]
    scf.yield %3 : tensor<?xf32>
  }
  // CHECK: %[[insert:.*]] = tensor.insert_slice %[[for]]#1 into %[[for]]#0

  // CHECK: return %[[insert]]
  return %0 : tensor<?xf32>
}

// -----

func.func @subset_of_subset(%arg: tensor<?xf32>) -> tensor<?xf32> {
  %lb = "test.foo"() : () -> (index)
  %ub = "test.foo"() : () -> (index)
  %step = "test.foo"() : () -> (index)

  // CHECK: %[[extract1:.*]] = tensor.extract_slice %[[arg]]
  // CHECK: %[[extract2:.*]] = tensor.extract_slice %[[extract1]]
  // CHECK: %[[for:.*]]:3 = scf.for {{.*}} iter_args(%[[t:.*]] = %[[arg]], %[[hoisted1:.*]] = %[[extract1]], %[[hoisted2:.*]] = %[[extract2]])
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%t = %arg) -> (tensor<?xf32>) {
    %extract1 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
    %extract2 = tensor.extract_slice %extract1[1][2][1] : tensor<5xf32> to tensor<2xf32>

    // CHECK: %[[foo:.*]] = "test.foo"(%[[hoisted2]])
    %2 = "test.foo"(%extract2) : (tensor<2xf32>) -> (tensor<2xf32>)

    %insert1 = tensor.insert_slice %2 into %extract1[1][2][1] : tensor<2xf32> into tensor<5xf32>
    %insert2 = tensor.insert_slice %insert1 into %t[0][5][1] : tensor<5xf32> into tensor<?xf32>

    // CHECK: scf.yield %[[t]], %[[hoisted1]], %[[foo]]
    scf.yield %insert2 : tensor<?xf32>
  }
  // CHECK: %[[insert2:.*]] = tensor.insert_slice %[[for]]#2 into %[[for]]#1[1] [2] [1]
  // CHECK: %[[insert1:.*]] = tensor.insert_slice %[[insert2]] into %[[for]]#0[0] [5] [1]

  // CHECK: return %[[insert1]]
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @hoist_matching_chain(
//  CHECK-SAME:     %[[arg:.*]]: tensor<?xf32>
func.func @hoist_matching_chain(%arg: tensor<?xf32>) -> tensor<?xf32> {
  %lb = "test.foo"() : () -> (index)
  %ub = "test.foo"() : () -> (index)
  %step = "test.foo"() : () -> (index)
  %sz = "test.foo"() : () -> (index)

  // CHECK: %[[extract2:.*]] = tensor.extract_slice %[[arg]][%{{.*}}] [5] [1]
  // CHECK: %[[extract1:.*]] = tensor.extract_slice %[[arg]][0] [%{{.*}}] [1]
  // CHECK: %[[for:.*]]:3 = scf.for {{.*}} iter_args(%[[t:.*]] = %[[arg]], %[[hoisted2:.*]] = %[[extract2]], %[[hoisted1:.*]] = %[[extract1]])
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%t = %arg) -> (tensor<?xf32>) {
    %1 = tensor.extract_slice %t[0][%sz][1] : tensor<?xf32> to tensor<?xf32>
    %2 = tensor.extract_slice %t[%sz][5][1] : tensor<?xf32> to tensor<5xf32>
    // CHECK-DAG: %[[foo1:.*]] = "test.foo"(%[[hoisted1]])
    // CHECK-DAG: %[[foo2:.*]] = "test.foo"(%[[hoisted2]])
    %foo1 = "test.foo"(%1) : (tensor<?xf32>) -> (tensor<?xf32>)
    %foo2 = "test.foo"(%2) : (tensor<5xf32>) -> (tensor<5xf32>)
    %5 = tensor.insert_slice %foo2 into %t[%sz][5][1] : tensor<5xf32> into tensor<?xf32>
    %6 = tensor.insert_slice %foo1 into %5[0][%sz][1] : tensor<?xf32> into tensor<?xf32>
    // CHECK: scf.yield %[[t]], %[[foo2]], %[[foo1]]
    scf.yield %6 : tensor<?xf32>
  }
  // CHECK: %[[insert2:.*]] = tensor.insert_slice %[[for]]#2 into %[[for]]#0[0] [%{{.*}}] [1]
  // CHECK: %[[insert1:.*]] = tensor.insert_slice %[[for]]#1 into %[[insert2]][%{{.*}}] [5] [1]

  // CHECK: return %[[insert1]]
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @do_not_hoist_overlapping_subsets(
func.func @do_not_hoist_overlapping_subsets(%arg: tensor<?xf32>) -> tensor<?xf32> {
  %lb = "test.foo"() : () -> (index)
  %ub = "test.foo"() : () -> (index)
  %step = "test.foo"() : () -> (index)
  %sz1 = "test.foo"() : () -> (index)
  %sz2 = "test.foo"() : () -> (index)

  // CHECK: scf.for
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%t = %arg) -> (tensor<?xf32>) {
    // These two slices are potentially overlapping. Do not hoist.
    // CHECK: tensor.extract_slice
    // CHECK: tensor.extract_slice
    %1 = tensor.extract_slice %t[0][%sz1][1] : tensor<?xf32> to tensor<?xf32>
    %2 = tensor.extract_slice %t[10][%sz2][1] : tensor<?xf32> to tensor<?xf32>
    // CHECK: "test.foo"
    // CHECK: "test.foo"
    %foo1 = "test.foo"(%1) : (tensor<?xf32>) -> (tensor<?xf32>)
    %foo2 = "test.foo"(%2) : (tensor<?xf32>) -> (tensor<?xf32>)
    // CHECK: tensor.insert_slice
    // CHECK: tensor.insert_slice
    %5 = tensor.insert_slice %foo2 into %t[0][%sz1][1] : tensor<?xf32> into tensor<?xf32>
    %6 = tensor.insert_slice %foo1 into %5[10][%sz2][1] : tensor<?xf32> into tensor<?xf32>
    // CHECK: scf.yield
    scf.yield %6 : tensor<?xf32>
  }

  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @multiple_yields(
//  CHECK-SAME:     %[[arg:.*]]: tensor<?xf32>
func.func @multiple_yields(%arg: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  %lb = "test.foo"() : () -> (index)
  %ub = "test.foo"() : () -> (index)
  %step = "test.foo"() : () -> (index)

  // CHECK: %[[extract1:.*]] = tensor.extract_slice
  // CHECK: %[[extract2:.*]] = tensor.extract_slice
  // CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[arg]], %{{.*}} = %[[arg]], %{{.*}} = %[[extract1]], %{{.*}} = %[[extract2]])
  %0:2 = scf.for %iv = %lb to %ub step %step iter_args(%t1 = %arg, %t2 = %arg)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %1 = tensor.extract_slice %t1[0][5][1] : tensor<?xf32> to tensor<5xf32>
    %2 = tensor.extract_slice %t2[5][5][1] : tensor<?xf32> to tensor<5xf32>
    // CHECK: "test.foo"
    // CHECK: "test.foo"
    %foo1 = "test.foo"(%1) : (tensor<5xf32>) -> (tensor<5xf32>)
    %foo2 = "test.foo"(%2) : (tensor<5xf32>) -> (tensor<5xf32>)
    %5 = tensor.insert_slice %foo2 into %t1[0][5][1] : tensor<5xf32> into tensor<?xf32>
    %6 = tensor.insert_slice %foo1 into %t2[5][5][1] : tensor<5xf32> into tensor<?xf32>
    // CHECK: scf.yield
    scf.yield %5, %6 : tensor<?xf32>, tensor<?xf32>
  }
  // CHECK: tensor.insert_slice
  // CHECK: tensor.insert_slice

  return %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @do_not_hoist_swapping_yields(
func.func @do_not_hoist_swapping_yields(%arg: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  %lb = "test.foo"() : () -> (index)
  %ub = "test.foo"() : () -> (index)
  %step = "test.foo"() : () -> (index)

  // CHECK: scf.for
  %0:2 = scf.for %iv = %lb to %ub step %step iter_args(%t1 = %arg, %t2 = %arg)
      -> (tensor<?xf32>, tensor<?xf32>) {
    // CHECK: tensor.extract_slice
    // CHECK: tensor.extract_slice
    %1 = tensor.extract_slice %t1[0][5][1] : tensor<?xf32> to tensor<5xf32>
    %2 = tensor.extract_slice %t2[5][5][1] : tensor<?xf32> to tensor<5xf32>
    // CHECK: "test.foo"
    // CHECK: "test.foo"
    %foo1 = "test.foo"(%1) : (tensor<5xf32>) -> (tensor<5xf32>)
    %foo2 = "test.foo"(%2) : (tensor<5xf32>) -> (tensor<5xf32>)
    // CHECK: tensor.insert_slice
    // CHECK: tensor.insert_slice
    %5 = tensor.insert_slice %foo2 into %t1[0][5][1] : tensor<5xf32> into tensor<?xf32>
    %6 = tensor.insert_slice %foo1 into %t2[5][5][1] : tensor<5xf32> into tensor<?xf32>
    // Swapping yields: do not hoist.
    // CHECK: scf.yield
    scf.yield %6, %5 : tensor<?xf32>, tensor<?xf32>
  }

  return %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @non_subset_op(
func.func @non_subset_op(%arg: tensor<?xf32>) -> tensor<?xf32> {
  %lb = "test.foo"() : () -> (index)
  %ub = "test.foo"() : () -> (index)
  %step = "test.foo"() : () -> (index)

  // CHECK: scf.for
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%t = %arg) -> (tensor<?xf32>) {
    // If any value along the use-def chain from the region iter_arg to the
    // terminator is used by a non-subset op, no subset op along that chain can
    // be hoisted. That is because it is unknown which parts of the value are
    // accessed by the non-subset op.
    // CHECK: "test.non_subset_op"
    "test.non_subset_op"(%t) : (tensor<?xf32>) -> ()
    // CHECK: tensor.extract_slice
    %1 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
    // CHECK: "test.foo"
    %2 = "test.foo"(%1) : (tensor<5xf32>) -> (tensor<5xf32>)
    // CHECK: tensor.insert_slice
    %3 = tensor.insert_slice %2 into %t[0][5][1] : tensor<5xf32> into tensor<?xf32>
    // CHECK: scf.yield
    scf.yield %3 : tensor<?xf32>
  }

  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @non_loop_invariant_subset_op(
func.func @non_loop_invariant_subset_op(%arg: tensor<?xf32>) -> tensor<?xf32> {
  %lb = "test.foo"() : () -> (index)
  %ub = "test.foo"() : () -> (index)
  %step = "test.foo"() : () -> (index)

  // CHECK: scf.for
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%t = %arg) -> (tensor<?xf32>) {
    // Subset ops that are not loop-invariant cannot be hoisted.
    // CHECK: tensor.extract_slice
    %1 = tensor.extract_slice %t[%iv][5][1] : tensor<?xf32> to tensor<5xf32>
    // CHECK: "test.foo"
    %2 = "test.foo"(%1) : (tensor<5xf32>) -> (tensor<5xf32>)
    // CHECK: tensor.insert_slice
    %3 = tensor.insert_slice %2 into %t[%iv][5][1] : tensor<5xf32> into tensor<?xf32>
    // CHECK: scf.yield
    scf.yield %3 : tensor<?xf32>
  }

  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @nested_hoisting(
//  CHECK-SAME:     %[[arg:.*]]: tensor<?xf32>
func.func @nested_hoisting(%arg: tensor<?xf32>) -> tensor<?xf32> {
  %lb = "test.foo"() : () -> (index)
  %ub = "test.foo"() : () -> (index)
  %step = "test.foo"() : () -> (index)

  // CHECK: %[[extract:.*]] = tensor.extract_slice %[[arg]][0] [5] [1]
  // CHECK: %[[extract2:.*]] = tensor.extract_slice %[[arg]][5] [5] [1]
  // CHECK: %[[for:.*]]:3 = scf.for {{.*}} iter_args(%[[t:.*]] = %[[arg]], %[[hoisted:.*]] = %[[extract]], %[[hoisted2:.*]] = %[[extract2]])
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%t = %arg) -> (tensor<?xf32>) {
    %1 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
    // CHECK: %[[foo:.*]] = "test.foo"(%[[hoisted]])
    %2 = "test.foo"(%1) : (tensor<5xf32>) -> (tensor<5xf32>)
    %3 = tensor.insert_slice %2 into %t[0][5][1] : tensor<5xf32> into tensor<?xf32>
    // CHECK: %[[for2:.*]]:2 = {{.*}} iter_args(%[[t2:.*]] = %[[t]], %[[hoisted2_nested:.*]] = %[[hoisted2]])
    %4 = scf.for %iv2 = %lb to %ub step %step iter_args(%t2 = %3) -> (tensor<?xf32>) {
      %5 = tensor.extract_slice %t2[5][5][1] : tensor<?xf32> to tensor<5xf32>
      // CHECK: %[[foo2:.*]] = "test.foo"(%[[hoisted2_nested]])
      %6 = "test.foo"(%5) : (tensor<5xf32>) -> (tensor<5xf32>)
      %7 = tensor.insert_slice %6 into %t2[5][5][1] : tensor<5xf32> into tensor<?xf32>
      // CHECK: scf.yield %[[t2]], %[[foo2]]
      scf.yield %7 : tensor<?xf32>
    }
    // CHECK: scf.yield %[[for2]]#0, %[[foo]], %[[for2]]#1
    scf.yield %4 : tensor<?xf32>
  }
  // CHECK: %[[insert:.*]] = tensor.insert_slice %[[for]]#2 into %[[for]]#0[5] [5] [1]
  // CHECK: %[[insert2:.*]] = tensor.insert_slice %[[for]]#1 into %[[insert]][0] [5] [1]
  // CHECK: return %[[insert2]]
  return %0 : tensor<?xf32>
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
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<1xf32>) {
// CHECK:   vector.transfer_read %{{.*}} : tensor<?x?xf32>, vector<2xf32>
// CHECK:   scf.for {{.*}} iter_args({{.*}}) ->
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<2xf32>, vector<1xf32>) {
// CHECK:     vector.transfer_read %{{.*}} : tensor<?x?xf32>, vector<4xf32>
// CHECK:     "test.some_crippling_use"(%{{.*}}) : (tensor<?x?xf32>) -> ()
// CHECK:     vector.transfer_read %{{.*}} : tensor<?x?xf32>, vector<5xf32>
// CHECK:     "test.some_use"(%{{.*}}) : (vector<1xf32>) -> vector<1xf32>
// CHECK:     "test.some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "test.some_use"(%{{.*}}) : (tensor<?x?xf32>) -> vector<3xf32>
// CHECK:     "test.some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "test.some_use"(%{{.*}}) : (vector<5xf32>) -> vector<5xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<3xf32>, tensor<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<4xf32>, tensor<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<5xf32>, tensor<?x?xf32>
// CHECK:     "test.some_crippling_use"(%{{.*}}) : (tensor<?x?xf32>) -> ()
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
      "test.some_crippling_use"(%arg10) : (tensor<?x?xf32>) -> ()
      %r4 = vector.transfer_read %arg10[%c0, %c0], %cst: tensor<?x?xf32>, vector<5xf32>
      %r5 = vector.transfer_read %arg11[%c0, %c0], %cst: tensor<?x?xf32>, vector<6xf32>
      "test.some_crippling_use"(%arg11) : (tensor<?x?xf32>) -> ()
      %u0 = "test.some_use"(%r0) : (vector<1xf32>) -> vector<1xf32>
      %u1 = "test.some_use"(%r1) : (vector<2xf32>) -> vector<2xf32>
      %u2 = "test.some_use"(%arg8) : (tensor<?x?xf32>) -> vector<3xf32>
      %u3 = "test.some_use"(%r3) : (vector<4xf32>) -> vector<4xf32>
      %u4 = "test.some_use"(%r4) : (vector<5xf32>) -> vector<5xf32>
      %u5 = "test.some_use"(%r5) : (vector<6xf32>) -> vector<6xf32>
      %w1 = vector.transfer_write %u0, %arg7[%c0, %c0] : vector<1xf32>, tensor<?x?xf32>
      %w0 = vector.transfer_write %u1, %arg6[%i, %i] : vector<2xf32>, tensor<?x?xf32>
      %w2 = vector.transfer_write %u2, %arg8[%c0, %c0] : vector<3xf32>, tensor<?x?xf32>
      %w3 = vector.transfer_write %u3, %arg9[%c0, %c0] : vector<4xf32>, tensor<?x?xf32>
      %w4 = vector.transfer_write %u4, %arg10[%c0, %c0] : vector<5xf32>, tensor<?x?xf32>
      %w5 = vector.transfer_write %u5, %arg11[%c0, %c0] : vector<6xf32>, tensor<?x?xf32>
      "test.some_crippling_use"(%w3) : (tensor<?x?xf32>) -> ()
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
// CHECK: %[[R:.*]]:8 = scf.for {{.*}} iter_args({{.*}}) ->
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>) {
// CHECK:   scf.for {{.*}} iter_args({{.*}}) ->
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>) {
// CHECK:     vector.transfer_read %[[TENSOR1]]{{.*}} : tensor<?x?xf32>, vector<2xf32>
// CHECK:     vector.transfer_read %[[TENSOR1]]{{.*}} : tensor<?x?xf32>, vector<2xf32>
// CHECK:     "test.some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "test.some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "test.some_use"(%{{.*}}) : (vector<3xf32>) -> vector<3xf32>
// CHECK:     "test.some_use"(%{{.*}}) : (vector<3xf32>) -> vector<3xf32>
// CHECK:     "test.some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "test.some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "test.some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "test.some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     vector.transfer_write %{{.*}}, %{{.*}}{{.*}} : vector<2xf32>, tensor<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}}, %{{.*}}{{.*}} : vector<2xf32>, tensor<?x?xf32>
// CHECK:     scf.yield {{.*}} :
// CHECK-SAME: tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>
// CHECK:   }
// CHECK:   scf.yield {{.*}} :
// CHECK-SAME: tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>
// CHECK: }
// CHECK: %[[TENSOR4:.*]] = vector.transfer_write %[[R]]#7, %[[R]]#3{{.*}} : vector<4xf32>, tensor<?x?xf32>
// CHECK:                   vector.transfer_write %[[R]]#6, %[[TENSOR4]]{{.*}} : vector<4xf32>, tensor<?x?xf32>
// CHECK: %[[TENSOR5:.*]] = vector.transfer_write %[[R]]#5, %[[R]]#2{{.*}} : vector<3xf32>, tensor<?x?xf32>
// CHECK:                   vector.transfer_write %[[R]]#4, %[[TENSOR5]]{{.*}} : vector<3xf32>, tensor<?x?xf32>
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
      %u00 = "test.some_use"(%r00) : (vector<2xf32>) -> vector<2xf32>
      %u01 = "test.some_use"(%r01) : (vector<2xf32>) -> vector<2xf32>
      %u20 = "test.some_use"(%r20) : (vector<3xf32>) -> vector<3xf32>
      %u21 = "test.some_use"(%r21) : (vector<3xf32>) -> vector<3xf32>
      %u30 = "test.some_use"(%r30) : (vector<4xf32>) -> vector<4xf32>
      %u31 = "test.some_use"(%r31) : (vector<4xf32>) -> vector<4xf32>
      %u10 = "test.some_use"(%r10) : (vector<2xf32>) -> vector<2xf32>
      %u11 = "test.some_use"(%r11) : (vector<2xf32>) -> vector<2xf32>
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

    //      CHECK:   %[[R:.*]]:5 = scf.for %[[J:.*]] = {{.*}} iter_args(
    // CHECK-SAME:   %[[TENSOR0_ARG_L2:[0-9a-zA-Z]+]] = %[[TENSOR0_ARG]]
    // CHECK-SAME:   %[[TENSOR1_ARG_L2:[0-9a-zA-Z]+]] = %[[TENSOR1_ARG]]
    // CHECK-SAME:   %[[TENSOR2_ARG_L2:[0-9a-zA-Z]+]] = %[[TENSOR2_ARG]]
    // CHECK-SAME:   %[[ST0_ARG_L2:[0-9a-zA-Z]+]] = %[[ST0]]
    // CHECK-SAME:   %[[V0_ARG_L2:[0-9a-zA-Z]+]] = %[[V0]]
    // CHECK-SAME: ) ->
    // CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<1xf32>)
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

      // CHECK:     %[[U0:.*]] = "test.some_use"(%[[V0_ARG_L2]]) : (vector<1xf32>) -> vector<1xf32>
      // CHECK:     %[[U1:.*]] = "test.some_use"(%[[V1]]) : (vector<2xf32>) -> vector<2xf32>
      // CHECK:     %[[U2:.*]] = "test.some_use"(%[[V2]]) : (vector<3xf32>) -> vector<3xf32>
      %u0 = "test.some_use"(%r0) : (vector<1xf32>) -> vector<1xf32>
      %u1 = "test.some_use"(%r1) : (vector<2xf32>) -> vector<2xf32>
      %u2 = "test.some_use"(%r2) : (vector<3xf32>) -> vector<3xf32>

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

      // CHECK:     scf.yield {{.*}} : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<1xf32>
      // CHECK:   }
      scf.yield %sti0, %sti1, %sti22:
        tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
    }

    // Hoisted
    // CHECK:   %[[STI0:.*]] = vector.transfer_write %[[R]]#4, %[[R]]#3{{.*}} : vector<1xf32>, tensor<?x?xf32>
    // CHECK:   tensor.insert_slice %[[STI0]] into %[[R]]#0[%[[I]], %[[I]]]{{.*}} : tensor<?x?xf32> into tensor<?x?xf32>

    // CHECK:   scf.yield {{.*}} : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
    scf.yield %1#0, %1#1, %1#2 :
      tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>

    // CHECK: }
  }
  return %0#0, %0#1, %0#2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_write_pairs_disjoint_tensor(
//  CHECK-SAME:   %[[T:.*]]: tensor<?x?xf32>,
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[R0:.*]] = vector.transfer_read %[[T]][%[[C0]], %[[C0]]], %{{.*}} : tensor<?x?xf32>, vector<2xf32>
//   CHECK-DAG:   %[[R1:.*]] = vector.transfer_read %[[T]][%[[C0]], %[[C3]]], %{{.*}} : tensor<?x?xf32>, vector<2xf32>
//       CHECK:   %[[F:.*]]:3 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[TL:.*]] = %[[T]], %[[R2:.*]] = %[[R0]], %[[R3:.*]] = %[[R1]]) -> (tensor<?x?xf32>, vector<2xf32>, vector<2xf32>) {
//       CHECK:     %[[R4:.*]] = "test.some_use"(%[[R2]]) : (vector<2xf32>) -> vector<2xf32>
//       CHECK:     %[[R5:.*]] = "test.some_use"(%[[R3]]) : (vector<2xf32>) -> vector<2xf32>
//       CHECK:     scf.yield %[[TL]], %[[R4]], %[[R5]] : tensor<?x?xf32>, vector<2xf32>, vector<2xf32>
//       CHECK:   }
//       CHECK:   %[[W0:.*]] = vector.transfer_write %[[F]]#2, %[[F]]#0[%[[C0]], %[[C3]]] : vector<2xf32>, tensor<?x?xf32>
//       CHECK:   %[[W1:.*]] = vector.transfer_write %[[F]]#1, %[[W0]][%[[C0]], %[[C0]]] : vector<2xf32>, tensor<?x?xf32>
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
    %u00 = "test.some_use"(%r00) : (vector<2xf32>) -> vector<2xf32>
    %w10 = vector.transfer_write %u00, %arg5[%c0, %c0] : vector<2xf32>, tensor<?x?xf32>

    // Hoist by properly bypassing the disjoint write %w10.
    %r01 = vector.transfer_read %w10[%c0, %c3], %cst: tensor<?x?xf32>, vector<2xf32>
    %u01 = "test.some_use"(%r01) : (vector<2xf32>) -> vector<2xf32>
    %w11 = vector.transfer_write %u01, %w10[%c0, %c3] : vector<2xf32>, tensor<?x?xf32>
    scf.yield %w11 : tensor<?x?xf32>
  }
  return %1 : tensor<?x?xf32>
}
