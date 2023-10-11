// RUN: mlir-opt %s -test-transform-dialect-interpreter -verify-diagnostics -allow-unregistered-dialect -split-input-file | FileCheck %s

// CHECK-DAG: memref.global "private" @[[ALLOC0:alloc.*]] : memref<2x32xf32>
// CHECK-DAG: memref.global "private" @[[ALLOC1:alloc.*]] : memref<2x32xf32>

// CHECK-DAG: func.func @func(%[[LB:.*]]: index, %[[UB:.*]]: index)
func.func @func(%lb: index, %ub: index) {
  // CHECK-DAG: scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (%[[LB]], %[[UB]])
  scf.forall (%arg0, %arg1) in (%lb, %ub) {
    // CHECK-DAG: %[[MR0:.*]] = memref.get_global @[[ALLOC0]] : memref<2x32xf32>
    // CHECK-DAG: %[[MR1:.*]] = memref.get_global @[[ALLOC1]] : memref<2x32xf32>
    // CHECK-DAG: memref.store %{{.*}}, %[[MR0]][%{{.*}}, %{{.*}}] : memref<2x32xf32>
    // CHECK-DAG: memref.store %{{.*}}, %[[MR1]][%{{.*}}, %{{.*}}] : memref<2x32xf32>
    %cst = arith.constant 0.0 : f32
    %mr0 = memref.alloca() : memref<2x32xf32>
    %mr1 = memref.alloca() : memref<2x32xf32>
    memref.store %cst, %mr0[%arg0, %arg1] : memref<2x32xf32>
    memref.store %cst, %mr1[%arg0, %arg1] : memref<2x32xf32>
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %alloca = transform.structured.match ops{["memref.alloca"]} in %arg0
      : (!transform.any_op) -> !transform.op<"memref.alloca">
  %get_global, %global = transform.memref.alloca_to_global %alloca
        : (!transform.op<"memref.alloca">)
          -> (!transform.any_op, !transform.any_op)
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0) -> ((d0 floordiv 4) mod 2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

// CHECK-LABEL: func @multi_buffer
func.func @multi_buffer(%in: memref<16xf32>) {
  // CHECK: %[[A:.*]] = memref.alloc() : memref<2x4xf32>
  // expected-remark @below {{transformed}}
  %tmp = memref.alloc() : memref<4xf32>

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index

  // CHECK: scf.for %[[IV:.*]] = %[[C0]]
  scf.for %i0 = %c0 to %c16 step %c4 {
    // CHECK: %[[I:.*]] = affine.apply #[[$MAP0]](%[[IV]])
    // CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: ?>>
    %1 = memref.subview %in[%i0] [4] [1] : memref<16xf32> to memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
    // CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4xf32, #[[$MAP1]]> to memref<4xf32, strided<[1], offset: ?>>
    memref.copy %1, %tmp :  memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>> to memref<4xf32>

    "some_use"(%tmp) : (memref<4xf32>) ->()
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["memref.alloc"]} in %arg1 : (!transform.any_op) -> !transform.op<"memref.alloc">
  %1 = transform.memref.multibuffer %0 {factor = 2 : i64} : (!transform.op<"memref.alloc">) -> !transform.any_op
  // Verify that the returned handle is usable.
  transform.test_print_remark_at_operand %1, "transformed" : !transform.any_op
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0) -> ((d0 floordiv 4) mod 2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

// CHECK-LABEL: func @multi_buffer_on_affine_loop
func.func @multi_buffer_on_affine_loop(%in: memref<16xf32>) {
  // CHECK: %[[A:.*]] = memref.alloc() : memref<2x4xf32>
  // expected-remark @below {{transformed}}
  %tmp = memref.alloc() : memref<4xf32>

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index

  // CHECK: affine.for %[[IV:.*]] = 0
  affine.for %i0 = 0 to 16 step 4 {
    // CHECK: %[[I:.*]] = affine.apply #[[$MAP0]](%[[IV]])
    // CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: ?>>
    %1 = memref.subview %in[%i0] [4] [1] : memref<16xf32> to memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
    // CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4xf32, #[[$MAP1]]> to memref<4xf32, strided<[1], offset: ?>>
    memref.copy %1, %tmp :  memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>> to memref<4xf32>

    "some_use"(%tmp) : (memref<4xf32>) ->()
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["memref.alloc"]} in %arg1 : (!transform.any_op) -> !transform.op<"memref.alloc">
  %1 = transform.memref.multibuffer %0 {factor = 2 : i64} : (!transform.op<"memref.alloc">) -> !transform.any_op
  // Verify that the returned handle is usable.
  transform.test_print_remark_at_operand %1, "transformed" : !transform.any_op
}

// -----

// Trying to use multibuffer on allocs that are used in different loops
// with none dominating the other is going to fail.
// Check that we emit a proper error for that.
func.func @multi_buffer_uses_with_no_loop_dominator(%in: memref<16xf32>, %cond: i1) {
  // expected-error @below {{op failed to multibuffer}}
  %tmp = memref.alloc() : memref<4xf32>

  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  scf.if %cond {
    scf.for %i0 = %c0 to %c16 step %c4 {
      %var = memref.subview %in[%i0] [4] [1] : memref<16xf32> to memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
      memref.copy %var, %tmp :  memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>> to memref<4xf32>

      "some_use"(%tmp) : (memref<4xf32>) ->()
    }
  }

  scf.for %i0 = %c0 to %c16 step %c4 {
    %1 = memref.subview %in[%i0] [4] [1] : memref<16xf32> to memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
    memref.copy %1, %tmp :  memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>> to memref<4xf32>

    "some_use"(%tmp) : (memref<4xf32>) ->()
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["memref.alloc"]} in %arg1 : (!transform.any_op) -> !transform.op<"memref.alloc">
  %1 = transform.memref.multibuffer %0 {factor = 2 : i64} : (!transform.op<"memref.alloc">) -> !transform.any_op
}

// -----

// Make sure the multibuffer operation is typed so that it only supports
// memref.alloc.
// Check that we emit an error if we try to match something else.
func.func @multi_buffer_reject_alloca(%in: memref<16xf32>, %cond: i1) {
  %tmp = memref.alloca() : memref<4xf32>

  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  scf.if %cond {
    scf.for %i0 = %c0 to %c16 step %c4 {
      %var = memref.subview %in[%i0] [4] [1] : memref<16xf32> to memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
      memref.copy %var, %tmp :  memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>> to memref<4xf32>

      "some_use"(%tmp) : (memref<4xf32>) ->()
    }
  }

  scf.for %i0 = %c0 to %c16 step %c4 {
    %1 = memref.subview %in[%i0] [4] [1] : memref<16xf32> to memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
    memref.copy %1, %tmp :  memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>> to memref<4xf32>

    "some_use"(%tmp) : (memref<4xf32>) ->()
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["memref.alloca"]} in %arg1 : (!transform.any_op) -> !transform.op<"memref.alloca">
  // expected-error @below {{'transform.memref.multibuffer' op operand #0 must be Transform IR handle to memref.alloc operations, but got '!transform.op<"memref.alloca">'}}
  %1 = transform.memref.multibuffer %0 {factor = 2 : i64} : (!transform.op<"memref.alloca">) -> !transform.any_op
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0) -> ((d0 floordiv 4) mod 2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

// CHECK-LABEL: func @multi_buffer_one_alloc_with_use_outside_of_loop
// Make sure we manage to apply multi_buffer to the memref that is used in
// the loop (%tmp) and don't error out for the one that is not (%tmp2).
func.func @multi_buffer_one_alloc_with_use_outside_of_loop(%in: memref<16xf32>) {
  // CHECK: %[[A:.*]] = memref.alloc() : memref<2x4xf32>
  // expected-remark @below {{transformed}}
  %tmp = memref.alloc() : memref<4xf32>
  %tmp2 = memref.alloc() : memref<4xf32>

  "some_use_outside_of_loop"(%tmp2) : (memref<4xf32>) -> ()

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index

  // CHECK: scf.for %[[IV:.*]] = %[[C0]]
  scf.for %i0 = %c0 to %c16 step %c4 {
    // CHECK: %[[I:.*]] = affine.apply #[[$MAP0]](%[[IV]])
    // CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: ?>>
    %1 = memref.subview %in[%i0] [4] [1] : memref<16xf32> to memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
    // CHECK: memref.copy %{{.*}}, %[[SV]] : memref<4xf32, #[[$MAP1]]> to memref<4xf32, strided<[1], offset: ?>>
    memref.copy %1, %tmp :  memref<4xf32, affine_map<(d0)[s0] -> (d0 + s0)>> to memref<4xf32>

    "some_use"(%tmp) : (memref<4xf32>) ->()
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["memref.alloc"]} in %arg1 : (!transform.any_op) -> !transform.op<"memref.alloc">
  %1 = transform.memref.multibuffer %0 {factor = 2 : i64} : (!transform.op<"memref.alloc">) -> !transform.any_op
  // Verify that the returned handle is usable.
  transform.test_print_remark_at_operand %1, "transformed" : !transform.any_op
}

// -----


// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0) -> ((d0 floordiv 4) mod 2)>

// CHECK-LABEL: func @multi_buffer
func.func @multi_buffer_no_analysis(%in: memref<16xf32>) {
  // CHECK: %[[A:.*]] = memref.alloc() : memref<2x4xf32>
  // expected-remark @below {{transformed}}
  %tmp = memref.alloc() : memref<4xf32>

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index

  // CHECK: scf.for %[[IV:.*]] = %[[C0]]
  scf.for %i0 = %c0 to %c16 step %c4 {
  // CHECK: %[[I:.*]] = affine.apply #[[$MAP0]](%[[IV]])
  // CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: ?>>
    "some_write_read"(%tmp) : (memref<4xf32>) ->()
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["memref.alloc"]} in %arg1 : (!transform.any_op) -> !transform.op<"memref.alloc">
  %1 = transform.memref.multibuffer %0 {factor = 2 : i64, skip_analysis} : (!transform.op<"memref.alloc">) -> !transform.any_op
  // Verify that the returned handle is usable.
  transform.test_print_remark_at_operand %1, "transformed" : !transform.any_op
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0) -> ((d0 floordiv 4) mod 2)>

// CHECK-LABEL: func @multi_buffer_dealloc
func.func @multi_buffer_dealloc(%in: memref<16xf32>) {
  // CHECK: %[[A:.*]] = memref.alloc() : memref<2x4xf32>
  // expected-remark @below {{transformed}}
  %tmp = memref.alloc() : memref<4xf32>

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index

  // CHECK: scf.for %[[IV:.*]] = %[[C0]]
  scf.for %i0 = %c0 to %c16 step %c4 {
  // CHECK: %[[I:.*]] = affine.apply #[[$MAP0]](%[[IV]])
  // CHECK: %[[SV:.*]] = memref.subview %[[A]][%[[I]], 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: ?>>
    "some_write_read"(%tmp) : (memref<4xf32>) ->()
  }

  // CHECK-NOT: memref.dealloc {{.*}} : memref<4xf32>
  // CHECK: memref.dealloc %[[A]] : memref<2x4xf32>
  memref.dealloc %tmp : memref<4xf32>
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["memref.alloc"]} in %arg1 : (!transform.any_op) -> !transform.op<"memref.alloc">
  %1 = transform.memref.multibuffer %0 {factor = 2 : i64, skip_analysis} : (!transform.op<"memref.alloc">) -> !transform.any_op
  // Verify that the returned handle is usable.
  transform.test_print_remark_at_operand %1, "transformed" : !transform.any_op
}

// -----

// CHECK-LABEL: func.func @dead_alloc
func.func @dead_alloc() {
  // CHECK-NOT: %{{.+}} = memref.alloc
  %0 = memref.alloc() : memref<8x64xf32, 3>
  %1 = memref.subview %0[0, 0] [8, 4] [1, 1] : memref<8x64xf32, 3> to
    memref<8x4xf32, affine_map<(d0, d1) -> (d0 * 64 + d1)>, 3>
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant dense<0.000000e+00> : vector<1x4xf32>
  vector.transfer_write %cst_0, %1[%c0, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, memref<8x4xf32, affine_map<(d0, d1) -> (d0 * 64 + d1)>, 3>
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.memref.erase_dead_alloc_and_stores %0 : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: @store_to_load
//  CHECK-SAME:   (%[[ARG:.+]]: vector<4xf32>)
//   CHECK-NOT:   memref.alloc()
//   CHECK-NOT:   vector.transfer_write
//   CHECK-NOT:   vector.transfer_read
//       CHECK:   return %[[ARG]] : vector<4xf32>
func.func @store_to_load(%arg: vector<4xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %cst_1 = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  vector.transfer_write %arg, %alloc[%c0] {in_bounds = [true]} : vector<4xf32>, memref<64xf32>
  %r = vector.transfer_read %alloc[%c0], %cst_1 {in_bounds = [true]} : memref<64xf32>, vector<4xf32>
  return %r : vector<4xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.memref.erase_dead_alloc_and_stores %0 : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: func @lower_to_llvm
//   CHECK-NOT:   memref.alloc
//       CHECK:   llvm.call @malloc
func.func @lower_to_llvm() {
  %0 = memref.alloc() : memref<2048xi8>
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_conversion_patterns to %0 {
    transform.apply_conversion_patterns.dialect_to_llvm "memref"
  } with type_converter {
    transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
  } {legal_dialects = ["func", "llvm"]} : !transform.any_op
}
