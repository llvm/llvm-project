// RUN: mlir-opt %s -one-shot-bufferize="allow-return-allocs-from-loops bufferize-function-boundaries test-analysis-only" -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -one-shot-bufferize="allow-return-allocs-from-loops bufferize-function-boundaries test-analysis-only analysis-fuzzer-seed=23" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="allow-return-allocs-from-loops bufferize-function-boundaries test-analysis-only analysis-fuzzer-seed=59" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="allow-return-allocs-from-loops bufferize-function-boundaries test-analysis-only analysis-fuzzer-seed=91" -split-input-file -o /dev/null

// CHECK-LABEL: func @scf_for_yield_only
func.func @scf_for_yield_only(
    %A : tensor<?xf32> {bufferization.writable = false},
    %B : tensor<?xf32> {bufferization.writable = true},
    %lb : index,
    %ub : index,
    %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  //      CHECK: scf.for
  // CHECK-NEXT: scf.yield
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  //      CHECK: } {__inplace_operands_attr__ = ["none", "none", "none", "false"]}
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%t = %A) -> (tensor<?xf32>) {
    scf.yield %t : tensor<?xf32>
  }

  //      CHECK: scf.for
  // CHECK-NEXT: scf.yield
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  //      CHECK: } {__inplace_operands_attr__ = ["none", "none", "none", "true"]}
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%t = %B) -> (tensor<?xf32>) {
    scf.yield %t : tensor<?xf32>
  }

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [-1, 1]
  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_for_with_tensor.insert_slice
func.func @scf_for_with_tensor.insert_slice(
    %A : tensor<?xf32> {bufferization.writable = false},
    %B : tensor<?xf32> {bufferization.writable = true},
    %C : tensor<4xf32> {bufferization.writable = false},
    %lb : index,
    %ub : index,
    %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  //      CHECK: scf.for
  // scf.for bbArgs are always inplaceable seen from ops inside the body:
  //   1. Either the matching tensor is not inplaceable and an alloc occurs
  //      which makes bbArg inplaceable.
  //   2. Or it is already inplaceable and so is bbArg.
  // CHECK-NEXT:   tensor.insert_slice
  // CHECK-SAME:     {__inplace_operands_attr__ = ["true", "true"]}
  // CHECK-NEXT:   tensor.insert_slice
  // CHECK-SAME:     {__inplace_operands_attr__ = ["true", "true"]}
  // CHECK-NEXT:   scf.yield {__inplace_operands_attr__ = ["true", "true"]}
  // CHECK-NEXT: } {__inplace_operands_attr__ = ["none", "none", "none", "false", "true"]}
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    %ttA = tensor.insert_slice %C into %tA[0][4][1] : tensor<4xf32> into tensor<?xf32>
    %ttB = tensor.insert_slice %C into %tB[0][4][1] : tensor<4xf32> into tensor<?xf32>
    scf.yield %ttA, %ttB : tensor<?xf32>, tensor<?xf32>
  }

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [-1, 1]
  return %r0#0, %r0#1: tensor<?xf32>, tensor<?xf32>
}

// -----

func.func private @some_use(tensor<?xf32>) -> ()

// CHECK-LABEL: func @scf_for_deps
func.func @scf_for_deps(
    %A : tensor<?xf32> {bufferization.writable = true},
    %B : tensor<?xf32> {bufferization.writable = true},
    %lb : index,
    %ub : index,
    %step : index)
  -> (tensor<?xf32>)
{
  // %r0 must be out of place because one use of %t in the subsequent production
  // of %r1 is read.
  //      CHECK: scf.for
  // CHECK-NEXT: call
  // CHECK-SAME: {__inplace_operands_attr__ = ["false"]}
  // CHECK-NEXT: scf.yield
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  //      CHECK: } {__inplace_operands_attr__ = ["none", "none", "none", "false"]}
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%t = %A) -> (tensor<?xf32>) {
    func.call @some_use(%t) : (tensor<?xf32>) -> ()
    scf.yield %t : tensor<?xf32>
  }

  // %r1 bufferizes inplace fine.
  //      CHECK: scf.for
  // CHECK-NEXT: call
  // CHECK-SAME: {__inplace_operands_attr__ = ["false"]}
  // CHECK-NEXT: scf.yield
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  //      CHECK: } {__inplace_operands_attr__ = ["none", "none", "none", "true"]}
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%t = %A) -> (tensor<?xf32>) {
    func.call @some_use(%t) : (tensor<?xf32>) -> ()
    scf.yield %t : tensor<?xf32>
  }

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0]
  return %r1: tensor<?xf32>
}

// -----

#accesses = [
  affine_map<(i) -> (i)>
]
#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

// CHECK-LABEL: func @reading_scf_for
func.func @reading_scf_for(%t1: tensor<?xf32> {bufferization.writable = true},
                           %s: index, %v: vector<5xf32>) -> (tensor<?xf32>, vector<5xf32>) {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32

  // Write to %t1.
  // CHECK:      vector.transfer_write
  // CHECK-SAME: __inplace_operands_attr__ = ["none", "false", "none"]
  %t3 = vector.transfer_write %v, %t1[%s] : vector<5xf32>, tensor<?xf32>

  // Read the old value of %t1 inside the loop via an alias.
  // CHECK: scf.for {{.*}} {
  %r, %v3 = scf.for %i = %c0 to %s step %c1 iter_args(%t2 = %t1, %v0 = %v) -> (tensor<?xf32>, vector<5xf32>) {
    // CHECK:      tensor.extract_slice
    // CHECK-SAME: __inplace_operands_attr__ = ["true", "none", "none"]
    %e = tensor.extract_slice %t2[%s][%s][1] : tensor<?xf32> to tensor<?xf32>

    // Read from %t1 via alias %e.
    %v2 = vector.transfer_read %e[%s], %cst : tensor<?xf32>, vector<5xf32>
    scf.yield %t2, %v2 : tensor<?xf32>, vector<5xf32>
  }
  // CHECK: } {__inplace_operands_attr__ = ["none", "none", "none", "true", "none"]}

  // Use %t3 in some way without reading it, so that it does not get DCE'd.
  // CHECK:      linalg.generic
  // CHECK-SAME: __inplace_operands_attr__ = ["true"]
  %o = linalg.generic #trait outs (%t3 : tensor<?xf32>) {
      ^bb(%0: f32) :
        linalg.yield %cst : f32
    } -> (tensor<?xf32>)

  return %o, %v3 : tensor<?xf32>, vector<5xf32>
}

// -----

#accesses = [
  affine_map<(i) -> (i)>
]
#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

// CHECK-LABEL: func @non_reading_scf_for
func.func @non_reading_scf_for(%t1: tensor<?xf32> {bufferization.writable = true},
                               %s: index, %v: vector<5xf32>) -> (tensor<?xf32>, vector<5xf32>) {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 0.0 : f32

  // Write to %t1.
  // CHECK:      vector.transfer_write
  // CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
  %t3 = vector.transfer_write %v, %t1[%s] : vector<5xf32>, tensor<?xf32>

  // This loop does not read from %t1. It only writes to it.
  // CHECK:      scf.for
  %r, %v3 = scf.for %i = %c0 to %c10 step %c1 iter_args(%t2 = %t1, %v0 = %v) -> (tensor<?xf32>, vector<5xf32>) {
    // Write to %t1 via %t2. (Overwrite %t3.)
    // CHECK:      linalg.generic
    // CHECK-SAME: __inplace_operands_attr__ = ["true"]
    %o2 = linalg.generic #trait outs (%t2 : tensor<?xf32>) {
        ^bb(%0: f32) :
          linalg.yield %cst : f32
      } -> (tensor<?xf32>)

    // Read overwritten value. This is not a read of %t1.
    %v2 = vector.transfer_read %o2[%s], %cst : tensor<?xf32>, vector<5xf32>
    scf.yield %o2, %v2 : tensor<?xf32>, vector<5xf32>
  }

  // Use %t3 in some way without reading it, so that it does not get DCE'd.
  // CHECK:      linalg.generic
  // CHECK-SAME: __inplace_operands_attr__ = ["true"]
  %o = linalg.generic #trait outs (%t3 : tensor<?xf32>) {
      ^bb(%0: f32) :
        linalg.yield %cst : f32
    } -> (tensor<?xf32>)

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0, -1]
  return %o, %v3 : tensor<?xf32>, vector<5xf32>
}

// -----

//===----------------------------------------------------------------------===//
// scf.if cases
//===----------------------------------------------------------------------===//

// This example passes analysis, but it fails when bufferizing.
// CHECK-LABEL: func @scf_if_inplace1
func.func @scf_if_inplace1(%t1: tensor<?xf32> {bufferization.writable = true},
                           %t2: tensor<?xf32> {bufferization.writable = true},
                           %cond: i1) -> tensor<?xf32> {
  %r = scf.if %cond -> (tensor<?xf32>) {
    // CHECK:      scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %t1 : tensor<?xf32>
  } else {
    // CHECK:      scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %t2 : tensor<?xf32>
  }
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_if_inplace2
func.func @scf_if_inplace2(%t1: tensor<?xf32> {bufferization.writable = true},
                           %v: vector<5xf32>, %idx: index,
                           %cond: i1) -> tensor<?xf32> {
  %r = scf.if %cond -> (tensor<?xf32>) {
    // CHECK:      scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %t1 : tensor<?xf32>
  } else {
    //      CHECK: vector.transfer_write
    // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none"]
    %t2 = vector.transfer_write %v, %t1[%idx] : vector<5xf32>, tensor<?xf32>
    scf.yield %t2 : tensor<?xf32>
  }
  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0]
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_if_inplace3
func.func @scf_if_inplace3(%t1: tensor<?xf32> {bufferization.writable = true},
                           %v1: vector<5xf32>, %v2: vector<5xf32>, %idx: index,
                           %cond: i1) -> tensor<?xf32> {
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none", "none"]
  %e = tensor.extract_slice %t1[%idx][%idx][1] : tensor<?xf32> to tensor<?xf32>
  %r = scf.if %cond -> (tensor<?xf32>) {
    //      CHECK: vector.transfer_write
    // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none"]
    %t2 = vector.transfer_write %v1, %e[%idx] : vector<5xf32>, tensor<?xf32>
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %t2 : tensor<?xf32>
  } else {
    // Writing the same tensor through an alias. This is OK.
    //      CHECK: vector.transfer_write
    // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none"]
    %t3 = vector.transfer_write %v2, %t1[%idx] : vector<5xf32>, tensor<?xf32>
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %t3 : tensor<?xf32>
  }
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_if_in_place4
func.func @scf_if_in_place4(%t1: tensor<?xf32> {bufferization.writable = true},
                            %v: vector<5xf32>, %idx: index,
                            %cond: i1, %cond2: i1) -> (tensor<?xf32>, vector<10xf32>) {
  %cst = arith.constant 0.0 : f32
  %r = scf.if %cond -> (tensor<?xf32>) {
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %t1 : tensor<?xf32>
  } else {
    //      CHECK: vector.transfer_write
    // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none"]
    %t2 = vector.transfer_write %v, %t1[%idx] : vector<5xf32>, tensor<?xf32>
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %t2 : tensor<?xf32>
  }
  %r_alias = scf.if %cond2 -> (tensor<?xf32>) {
    // Reading %r is OK. No conflict.
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %r : tensor<?xf32>
  } else {
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %r : tensor<?xf32>
  }
  %v2 = vector.transfer_read %r_alias[%idx], %cst : tensor<?xf32>, vector<10xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0, -1]
  return %r_alias, %v2 : tensor<?xf32>, vector<10xf32>
}

// -----

// CHECK-LABEL: func @scf_if_inplace5
func.func @scf_if_inplace5(%t1: tensor<?xf32> {bufferization.writable = true},
                           %idx: index, %cond: i1) -> tensor<?xf32> {
  %r = scf.if %cond -> (tensor<?xf32>) {
    //      CHECK: tensor.extract_slice
    // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none", "none"]
    %e = tensor.extract_slice %t1[%idx][%idx][1] : tensor<?xf32> to tensor<?xf32>
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %e : tensor<?xf32>
  } else {
    //      CHECK: tensor.extract_slice
    // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none", "none"]
    %f = tensor.extract_slice %t1[%idx][%idx][1] : tensor<?xf32> to tensor<?xf32>
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %f : tensor<?xf32>
  }

  // Inserting into an equivalent tensor at the same offset. This bufferizes
  // inplace.
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none", "none"]
  %r2 = tensor.insert_slice %r into %t1[%idx][%idx][1] : tensor<?xf32> into tensor<?xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0]
  return %r2 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_if_inplace6
func.func @scf_if_inplace6(%t1: tensor<?xf32> {bufferization.writable = true},
                           %v1: vector<5xf32>, %v2: vector<5xf32>,
                           %v3: vector<5xf32>, %idx: index,
                           %cond: i1, %cond2: i1) -> tensor<?xf32> {
  // Test nested scf.if ops.
  %r = scf.if %cond -> (tensor<?xf32>) {
    %t2 = scf.if %cond2 -> (tensor<?xf32>) {
      //      CHECK: vector.transfer_write
      // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none"]
      %t3 = vector.transfer_write %v1, %t1[%idx] : vector<5xf32>, tensor<?xf32>
      //      CHECK: scf.yield
      // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
      scf.yield %t3 : tensor<?xf32>
    } else {
      //      CHECK: vector.transfer_write
      // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none"]
      %t4 = vector.transfer_write %v3, %t1[%idx] : vector<5xf32>, tensor<?xf32>
      //      CHECK: scf.yield
      // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
      scf.yield %t4 : tensor<?xf32>
    }
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %t2 : tensor<?xf32>
  } else {
    //      CHECK: vector.transfer_write
    // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none"]
    %t3 = vector.transfer_write %v2, %t1[%idx] : vector<5xf32>, tensor<?xf32>
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %t3 : tensor<?xf32>
  }

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0]
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_if_inplace7
func.func @scf_if_inplace7(%t1: tensor<?xf32> {bufferization.writable = true},
                           %v1: vector<5xf32>, %v2: vector<5xf32>, %idx: index,
                           %idx2: index, %cond: i1) -> (tensor<?xf32>, vector<5xf32>) {
  %cst = arith.constant 0.0 : f32
  %r, %v_r2 = scf.if %cond -> (tensor<?xf32>, vector<5xf32>) {
    //      CHECK: vector.transfer_write
    // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none"]
    %t2 = vector.transfer_write %v1, %t1[%idx] : vector<5xf32>, tensor<?xf32>
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none"]}
    scf.yield %t2, %v1 : tensor<?xf32>, vector<5xf32>
  } else {
    // Writing the same tensor through an alias.
    //      CHECK: vector.transfer_write
    // CHECK-SAME: {__inplace_operands_attr__ = ["none", "false", "none"]
    %t3 = vector.transfer_write %v2, %t1[%idx] : vector<5xf32>, tensor<?xf32>
    // Read the original value of %t1. This requires the write in this branch
    // to be out-of-place. But the write in the other branch can still be
    // inplace.
    %v_r = vector.transfer_read %t1[%idx2], %cst : tensor<?xf32>, vector<5xf32>
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none"]}
    scf.yield %t3, %v_r : tensor<?xf32>, vector<5xf32>
  }
  return %r, %v_r2 : tensor<?xf32>, vector<5xf32>
}

// -----

// CHECK-LABEL: func @scf_if_out_of_place1a
func.func @scf_if_out_of_place1a(%t1: tensor<?xf32> {bufferization.writable = true},
                                 %idx: index, %idx2: index,
                                 %cond: i1) -> tensor<?xf32> {
  %r = scf.if %cond -> (tensor<?xf32>) {
    //      CHECK: tensor.extract_slice
    // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none", "none"]
    %e = tensor.extract_slice %t1[%idx][%idx][1] : tensor<?xf32> to tensor<?xf32>
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %e : tensor<?xf32>
  } else {
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %t1 : tensor<?xf32>
  }

  // Reading from and writing to the same tensor via different args. This is a
  // conflict.
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "false", "none", "none"]
  %r2 = tensor.insert_slice %r into %t1[%idx2][%idx2][1] : tensor<?xf32> into tensor<?xf32>
  return %r2 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_if_out_of_place1b
func.func @scf_if_out_of_place1b(%t1: tensor<?xf32> {bufferization.writable = true},
                                 %idx: index, %idx2: index, %idx3: index,
                                 %cond: i1) -> tensor<?xf32> {
  %r = scf.if %cond -> (tensor<?xf32>) {
    //      CHECK: tensor.extract_slice
    // CHECK-SAME: {__inplace_operands_attr__ = ["false", "none", "none"]
    %e = tensor.extract_slice %t1[%idx][%idx][1] : tensor<?xf32> to tensor<?xf32>
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %e : tensor<?xf32>
  } else {
    //      CHECK: tensor.extract_slice
    // CHECK-SAME: {__inplace_operands_attr__ = ["false", "none", "none"]
    %f = tensor.extract_slice %t1[%idx2][%idx2][1] : tensor<?xf32> to tensor<?xf32>
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %f : tensor<?xf32>
  }

  // Reading from and writing to the same tensor via different args. This is a
  // conflict. In contrast to scf_if_out_of_place1a, the fact that %r aliases
  // with %t1 is only detected when analyzing the tensor.extract_slices. That's
  // why the tensor.insert_slice is inplace and the two extract_slices are
  // out-of-place.
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none", "none"]
  %r2 = tensor.insert_slice %r into %t1[%idx3][%idx3][1] : tensor<?xf32> into tensor<?xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0]
  return %r2 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_if_out_of_place1c
func.func @scf_if_out_of_place1c(%t1: tensor<?xf32> {bufferization.writable = true},
                                 %idx: index, %idx2: index, %cond: i1) -> tensor<?xf32> {
  %r = scf.if %cond -> (tensor<?xf32>) {
    //      CHECK: tensor.extract_slice
    // CHECK-SAME: {__inplace_operands_attr__ = ["false", "none", "none"]
    %e = tensor.extract_slice %t1[%idx][%idx][1] : tensor<?xf32> to tensor<?xf32>
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %e : tensor<?xf32>
  } else {
    // TODO: This one could bufferize inplace, but the analysis is too restrictive.
    //      CHECK: tensor.extract_slice
    // CHECK-SAME: {__inplace_operands_attr__ = ["false", "none", "none"]
    %f = tensor.extract_slice %t1[%idx2][%idx2][1] : tensor<?xf32> to tensor<?xf32>
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %f : tensor<?xf32>
  }

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none", "none"]
  %r2 = tensor.insert_slice %r into %t1[%idx2][%idx2][1] : tensor<?xf32> into tensor<?xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0]
  return %r2 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_if_out_of_place2
func.func @scf_if_out_of_place2(%t1: tensor<?xf32> {bufferization.writable = true},
                                %v: vector<5xf32>, %idx: index,
                                %cond: i1) -> (tensor<?xf32>, vector<10xf32>) {
  %cst = arith.constant 0.0 : f32
  %r = scf.if %cond -> (tensor<?xf32>) {
    scf.yield %t1 : tensor<?xf32>
  } else {
    //      CHECK: vector.transfer_write
    // CHECK-SAME: {__inplace_operands_attr__ = ["none", "false", "none"]
    %t2 = vector.transfer_write %v, %t1[%idx] : vector<5xf32>, tensor<?xf32>
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %t2 : tensor<?xf32>
  }

  // Read the old value of %t1. Forces the transfer_write to bufferize
  // out-of-place.
  %v2 = vector.transfer_read %t1[%idx], %cst : tensor<?xf32>, vector<10xf32>
  return %r, %v2 : tensor<?xf32>, vector<10xf32>
}

// -----

// CHECK-LABEL: func @scf_if_out_of_place3
func.func @scf_if_out_of_place3(%t1: tensor<?xf32> {bufferization.writable = true},
                                %v: vector<5xf32>, %idx: index,
                                %cond: i1, %cond2: i1) -> (tensor<?xf32>, vector<10xf32>) {
  %cst = arith.constant 0.0 : f32
  %r = scf.if %cond -> (tensor<?xf32>) {
    scf.yield %t1 : tensor<?xf32>
  } else {
    //      CHECK: vector.transfer_write
    // CHECK-SAME: {__inplace_operands_attr__ = ["none", "false", "none"]
    %t2 = vector.transfer_write %v, %t1[%idx] : vector<5xf32>, tensor<?xf32>
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %t2 : tensor<?xf32>
  }
  %t1_alias = scf.if %cond2 -> (tensor<?xf32>) {
    // scf.yield bufferizes to a read. That is a conflict in this example.
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %t1 : tensor<?xf32>
  } else {
    //      CHECK: scf.yield
    // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
    scf.yield %t1 : tensor<?xf32>
  }
  %v2 = vector.transfer_read %t1_alias[%idx], %cst : tensor<?xf32>, vector<10xf32>
  return %r, %v2 : tensor<?xf32>, vector<10xf32>
}

// -----

// CHECK-LABEL: func @write_to_same_tensor_in_loop_in_place(
func.func @write_to_same_tensor_in_loop_in_place(
    %A : tensor<?xf32> {bufferization.writable = true},
    %lb : index, %ub : index, %step : index, %sz: index)
  -> (tensor<?xf32>)
{
  // CHECK: scf.for {{.*}} {
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%t = %A) -> (tensor<?xf32>) {
    %B = bufferization.alloc_tensor(%sz) : tensor<?xf32>
    %i2 = arith.index_cast %i : index to i32
    %i3 = arith.sitofp %i2 : i32 to f32
    // The tensor.insert is in-place because the %B is defined inside the loop.
    //      CHECK: tensor.insert
    // CHECK-SAME:   {__inplace_operands_attr__ = ["none", "true", "none"]}
    %B2 = tensor.insert %i3 into %B[%i] : tensor<?xf32>
    //      CHECK: tensor.insert_slice
    // CHECK-SAME:   {__inplace_operands_attr__ = ["true", "true", "none", "none"]}
    %A2 = tensor.insert_slice %B2 into %t[%i][%sz][1] : tensor<?xf32> into tensor<?xf32>
    scf.yield %A2 : tensor<?xf32>
  }
  // CHECK: } {__inplace_operands_attr__ = ["none", "none", "none", "true"]}

  return %r0 : tensor<?xf32>
}

// -----

// This is a regression test. Everything can bufferize in-place because %7 and
// %arg1 are in the same repetitive region.

// CHECK-LABEL: func @same_enclosing_repetitive_region
func.func @same_enclosing_repetitive_region(%2: tensor<320xf32>,
                                            %3: tensor<320x10240xf32>)
  -> tensor<320xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant -0.000000e+00 : f32
  %c320 = arith.constant 320 : index
  %4 = scf.forall (%arg0) in (%c320) shared_outs(%arg1 = %2) -> (tensor<320xf32>) {
    // CHECK: tensor.extract_slice {{.*}} {__inplace_operands_attr__ = ["true", "none"]}
    %5 = tensor.extract_slice %3[%arg0, 0] [1, 10240] [1, 1]  : tensor<320x10240xf32> to tensor<1x10240xf32>
    // CHECK: tensor.extract_slice {{.*}} {__inplace_operands_attr__ = ["true", "none"]}
    %6 = tensor.extract_slice %arg1[%arg0] [1] [1] : tensor<320xf32> to tensor<1xf32>
    // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<1xf32>) -> tensor<1xf32>
    // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>

    scf.forall.in_parallel {
      // CHECK: tensor.parallel_insert_slice {{.*}} {__inplace_operands_attr__ = ["true", "true", "none"]}
      tensor.parallel_insert_slice %8 into %arg1[%arg0] [1] [1] : tensor<1xf32> into tensor<320xf32>
    }
  }
  return %4 : tensor<320xf32>
}

// -----

// CHECK-LABEL: different_repetitive_region_via_alias
func.func @different_repetitive_region_via_alias(%arg0: tensor<4xf32>,
                                                 %arg1: tensor<4xf32>,
                                                 %arg2: index,
                                                 %arg3: index,
                                                 %arg4: index)
  -> (tensor<4xf32>)
{
  %cst = arith.constant 0.000000e+00 : f32
  %cst2 = arith.constant 1.000000e+00 : f32
  %0 = bufferization.alloc_tensor() : tensor<4xf32>

  // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "false"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4xf32>) -> tensor<4xf32>

  %2 = scf.for %arg5 = %arg2 to %arg3 step %arg4 iter_args(%arg6 = %arg1) -> (tensor<4xf32>) {
    // CHECK: tensor.extract {{.*}} {__inplace_operands_attr__ = ["true", "none"]}
    %4 = tensor.extract %1[%arg4] : tensor<4xf32>
    vector.print %4 : f32
    // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
    %5 = linalg.fill ins(%cst2 : f32) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
    scf.yield %5 : tensor<4xf32>
  }

  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: no_raw_conflict_after_repetitive_use
func.func @no_raw_conflict_after_repetitive_use(%arg0: tensor<4xf32>,
                                                %arg1: tensor<4xf32>,
                                                %arg2: index,
                                                %arg3: index,
                                                %arg4: index)
  -> (tensor<4xf32>, tensor<4xf32>)
{
  %cst = arith.constant 0.000000e+00 : f32
  %cst2 = arith.constant 1.000000e+00 : f32
  %cst3 = arith.constant 2.000000e+00 : f32
  %0 = bufferization.alloc_tensor() : tensor<4xf32>

  // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4xf32>) -> tensor<4xf32>

  %2 = scf.for %arg5 = %arg2 to %arg3 step %arg4 iter_args(%arg6 = %arg1) -> (tensor<4xf32>) {
    // CHECK: tensor.extract {{.*}} {__inplace_operands_attr__ = ["true", "none"]}
    %4 = tensor.extract %1[%arg4] : tensor<4xf32>
    vector.print %4 : f32
    // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "false"]}
    %5 = linalg.fill ins(%cst2 : f32) outs(%1 : tensor<4xf32>) -> tensor<4xf32>
    scf.yield %5 : tensor<4xf32>
  }

  // The following is *not* a RaW conflict.
  // CHECK: tensor.extract {{.*}} {__inplace_operands_attr__ = ["true", "none"]}
  %6 = tensor.extract %1[%arg4] : tensor<4xf32>
  vector.print %6 : f32
  // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
  %7 = linalg.fill ins(%cst3 : f32) outs(%1 : tensor<4xf32>) -> tensor<4xf32>

  return %2, %7 : tensor<4xf32>, tensor<4xf32>
}

// -----

// CHECK-LABEL: func @read_of_bbarg_in_repetitive_region(
func.func @read_of_bbarg_in_repetitive_region(
    %t: tensor<10xf32>, %a: index, %b: index, %c: index, %cst: f32) {
  // CHECK: scf.for
  scf.for %iv = %a to %b step %c {
    // Must bufferize out-of-place because definition of read is in a different
    // repetitive region.
    // CHECK: tensor.extract_slice {{.*}} {__inplace_operands_attr__ = ["true"]}
    %2 = tensor.extract_slice %t[0][4][1] : tensor<10xf32> to tensor<4xf32>
    %3 = tensor.extract %2[%a] : tensor<4xf32>
    vector.print %3 : f32
    // CHECK: tensor.insert {{.*}} {__inplace_operands_attr__ = ["none", "false", "none"]}
    %4 = tensor.insert %cst into %2[%a] : tensor<4xf32>
    %5 = tensor.extract %4[%a] : tensor<4xf32>
    vector.print %5 : f32
  }
  return
}

// -----

// CHECK-LABEL: func @read_definition_in_same_repetitive_region_as_write(
func.func @read_definition_in_same_repetitive_region_as_write(
    %t: tensor<10xf32>, %a: index, %b: index, %c: index, %cst: f32) {
  // CHECK: tensor.insert {{.*}} {__inplace_operands_attr__ = ["none", "true", "none"]}
  %1 = tensor.insert %cst into %t[%a] : tensor<10xf32>
  // CHECK: scf.for
  scf.for %iv = %a to %b step %c {
    // Can bufferize in-place.
    // CHECK: tensor.extract_slice {{.*}} {__inplace_operands_attr__ = ["true"]}
    %2 = tensor.extract_slice %1[0][4][1] : tensor<10xf32> to tensor<4xf32>
    %3 = tensor.extract %2[%a] : tensor<4xf32>
    vector.print %3 : f32
  }
  return
}

// -----

// CHECK-LABEL: func @read_definition_in_same_repetitive_region_as_conflicting_write(
func.func @read_definition_in_same_repetitive_region_as_conflicting_write(
    %t: tensor<10xf32>, %a: index, %b: index, %c: index, %cst: f32) {
  // Cannot bufferize in-place according to normal op dominance rules.
  // CHECK: tensor.insert {{.*}} {__inplace_operands_attr__ = ["none", "false", "none"]}
  %1 = tensor.insert %cst into %t[%a] : tensor<10xf32>
  // CHECK: scf.for
  scf.for %iv = %a to %b step %c {
    // CHECK: tensor.extract_slice {{.*}} {__inplace_operands_attr__ = ["true"]}
    %2 = tensor.extract_slice %t[0][4][1] : tensor<10xf32> to tensor<4xf32>
    %3 = tensor.extract %2[%a] : tensor<4xf32>
    vector.print %3 : f32
  }
  return
}

// -----

// CHECK: func @write_value_in_repetitive_region(
func.func @write_value_in_repetitive_region(
    %t: tensor<10xf32>, %a: index, %b: index, %c: index, %cst: f32) {
  %0 = tensor.extract %t[%a] : tensor<10xf32>
  vector.print %0 : f32

  scf.for %iv = %a to %b step %c {
    // No further read of %0, so this can bufferize in-place.
    // CHECK: tensor.extract_slice {{.*}} {__inplace_operands_attr__ = ["true"]}
    %2 = tensor.extract_slice %t[0][4][1] : tensor<10xf32> to tensor<4xf32>
    // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
    %filled = linalg.fill ins(%cst : f32) outs(%2 : tensor<4xf32>) -> tensor<4xf32>
    %3 = tensor.extract %filled[%a] : tensor<4xf32>
    vector.print %3 : f32
  }
  return
}

// -----

// CHECK-LABEL: func @nesting_op_repetitive_regions(
func.func @nesting_op_repetitive_regions(
    %t: tensor<10xf32>, %a: index, %b: index, %c: index, %cst: f32) {
  // Cannot bufferize in-place according to normal op dominance rules.
  // CHECK: tensor.insert {{.*}} {__inplace_operands_attr__ = ["none", "false", "none"]}
  %1 = tensor.insert %cst into %t[%a] : tensor<10xf32>
  // CHECK: scf.for
  scf.for %iv1 = %a to %b step %c {
    // CHECK: scf.for
    scf.for %iv2 = %a to %b step %c {
      // CHECK: scf.for
      scf.for %iv3 = %a to %b step %c {
        // CHECK: tensor.extract_slice {{.*}} {__inplace_operands_attr__ = ["true"]}
        %2 = tensor.extract_slice %t[0][4][1] : tensor<10xf32> to tensor<4xf32>
        %3 = tensor.extract %2[%a] : tensor<4xf32>
        vector.print %3 : f32
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: func @parallel_region()
func.func @parallel_region() -> tensor<320xf32>
{
  %alloc0 = bufferization.alloc_tensor() : tensor<320xf32>
  %alloc1 = bufferization.alloc_tensor() : tensor<1xf32>
  %c320 = arith.constant 320 : index
  // CHECK: scf.forall
  %0 = scf.forall (%arg0) in (%c320) shared_outs(%arg1 = %alloc0) -> (tensor<320xf32>) {
    %val = "test.foo"() : () -> (f32)
    // linalg.fill must bufferize out-of-place because every thread needs a
    // private copy of %alloc1.
    // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "false"]}
    %fill = linalg.fill ins(%val : f32) outs(%alloc1 : tensor<1xf32>) -> tensor<1xf32>
    scf.forall.in_parallel {
      // CHECK: tensor.parallel_insert_slice {{.*}} {__inplace_operands_attr__ = ["true", "true", "none"]}
      tensor.parallel_insert_slice %fill into %arg1[%arg0] [1] [1] : tensor<1xf32> into tensor<320xf32>
    }
  }
  // CHECK: } {__inplace_operands_attr__ = ["none", "true"]}
  return %0 : tensor<320xf32>
}

// -----

// CHECK-LABEL: func @parallel_region_mixed_def(
func.func @parallel_region_mixed_def(%c: i1) -> tensor<320xf32>
{
  %alloc0 = bufferization.alloc_tensor() : tensor<320xf32>
  %alloc1 = bufferization.alloc_tensor() : tensor<1xf32>
  %c320 = arith.constant 320 : index
  // CHECK: scf.forall
  %0 = scf.forall (%arg0) in (%c320) shared_outs(%arg1 = %alloc0) -> (tensor<320xf32>) {
    %alloc2 = bufferization.alloc_tensor() : tensor<1xf32>
    %selected = scf.if %c -> tensor<1xf32> {
      scf.yield %alloc1 : tensor<1xf32>
    } else {
      scf.yield %alloc2 : tensor<1xf32>
    }
    %val = "test.foo"() : () -> (f32)
    // linalg.fill must bufferize out-of-place because every thread needs a
    // private copy of %alloc1.
    // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "false"]}
    %fill = linalg.fill ins(%val : f32) outs(%selected : tensor<1xf32>) -> tensor<1xf32>
    scf.forall.in_parallel {
      // CHECK: tensor.parallel_insert_slice {{.*}} {__inplace_operands_attr__ = ["true", "true", "none"]}
      tensor.parallel_insert_slice %fill into %arg1[%arg0] [1] [1] : tensor<1xf32> into tensor<320xf32>
    }
  }
  // CHECK: } {__inplace_operands_attr__ = ["none", "true"]}
  return %0 : tensor<320xf32>
}

// -----

// CHECK-LABEL: func @parallel_region_two_writes(
func.func @parallel_region_two_writes(%f: f32) -> tensor<320xf32>
{
  %alloc0 = bufferization.alloc_tensor() : tensor<320xf32>
  %alloc1 = bufferization.alloc_tensor() : tensor<1xf32>
  %c320 = arith.constant 320 : index
  %c0 = arith.constant 0 : index
  // CHECK: scf.forall
  %0 = scf.forall (%arg0) in (%c320) shared_outs(%arg1 = %alloc0) -> (tensor<320xf32>) {
    %val = "test.foo"() : () -> (f32)
    // linalg.fill must bufferize out-of-place because every thread needs a
    // private copy of %alloc1.
    // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "false"]}
    %fill = linalg.fill ins(%val : f32) outs(%alloc1 : tensor<1xf32>) -> tensor<1xf32>
    // CHECK: tensor.insert
    // CHECK-SAME: __inplace_operands_attr__ = ["none", "true", "none"]
    %inserted = tensor.insert %f into %fill[%c0] : tensor<1xf32>

    scf.forall.in_parallel {
      // CHECK: tensor.parallel_insert_slice {{.*}} {__inplace_operands_attr__ = ["true", "true", "none"]}
      tensor.parallel_insert_slice %inserted into %arg1[%arg0] [1] [1] : tensor<1xf32> into tensor<320xf32>
    }
  }
  // CHECK: } {__inplace_operands_attr__ = ["none", "true"]}
  return %0 : tensor<320xf32>
}

// -----

// CHECK-LABEL: func @parallel_region_no_read()
func.func @parallel_region_no_read()
{
  %alloc0 = bufferization.alloc_tensor() : tensor<320xf32>
  %alloc1 = bufferization.alloc_tensor() : tensor<1xf32>
  %c320 = arith.constant 320 : index
  // CHECK: scf.forall
  scf.forall (%arg0) in (%c320) {
    %val = "test.foo"() : () -> (f32)
    // linalg.fill can bufferize in-place because no alias of %alloc1 is read.
    // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
    %fill = linalg.fill ins(%val : f32) outs(%alloc1 : tensor<1xf32>) -> tensor<1xf32>
    scf.forall.in_parallel {
    }
  }
  return
}
