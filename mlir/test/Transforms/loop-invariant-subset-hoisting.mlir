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
