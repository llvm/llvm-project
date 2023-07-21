// RUN: mlir-opt -test-transform-dialect-interpreter -split-input-file -verify-diagnostics -allow-unregistered-dialect %s | FileCheck %s


!tt = tensor<8xf16>

// CHECK-LABEL: func @copy_1d_8xf16
func.func @copy_1d_8xf16(%t0: !tt, %out: !tt) -> !tt {
  /// Too little data for all threads, needs predication, while keeping most
  /// minor transfer size -> 1 thread.
  // CHECK: scf.forall {{.*}} in (1) {{.*}}
  // CHECK:   linalg.copy {{.*}} -> tensor<8xf16>
  // CHECK: {mapping = [#gpu.linear<x>]}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 32 desired_bit_alignment = 128
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}

// -----

!tt = tensor<8xf16>
!tin = tensor<?xf16>

// CHECK-LABEL: func @pad_1d_8xf16
func.func @pad_1d_8xf16(%t0: !tin, %sz: index) -> !tt {
  %cst = arith.constant 0.0 : f16
  /// Too little data for all threads, needs predication, while keeping most
  /// minor transfer size -> 1 thread.
  // CHECK: scf.forall {{.*}} in (1) {{.*}}
  // CHECK:   %[[padded:.*]] = tensor.pad {{.*}}
  // CHECK:   tensor.cast %[[padded]] : tensor<?xf16> to tensor<8xf16>
  // CHECK: {mapping = [#gpu.linear<x>]}
  %0 = tensor.pad %t0 low[0] high[%sz] {
  ^bb0(%arg0: index):
    tensor.yield %cst : f16
  } : !tin to !tt
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["tensor.pad"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.gpu.map_copy_to_threads %0
    total_num_threads = 32 desired_bit_alignment = 128
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"tensor.pad">)
}

// -----

!tt = tensor<16xf16>

// CHECK-LABEL: func @copy_1d_16xf16
func.func @copy_1d_16xf16(%t0: !tt, %out: !tt) -> !tt {
  /// Too little data for all threads, needs predication, while keeping most
  /// minor transfer size -> 2 threads.
  // CHECK: scf.forall {{.*}} in (2) {{.*}}
  // CHECK:   linalg.copy {{.*}} -> tensor<8xf16>
  // CHECK: {mapping = [#gpu.linear<x>]}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 32 desired_bit_alignment = 128
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}

// -----

!tt = tensor<20xf16>

// CHECK-LABEL: func @copy_1d_20xf16
func.func @copy_1d_20xf16(%t0: !tt, %out: !tt) -> !tt {
  /// Too little data for all threads, needs predication, while keeping most
  /// minor transfer size -> 5 threads.
  // CHECK: scf.forall {{.*}} in (5) {{.*}}
  // CHECK:   linalg.copy {{.*}} -> tensor<4xf16>
  // CHECK: {mapping = [#gpu.linear<x>]}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 32 desired_bit_alignment = 128
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}


// -----

!tt = tensor<20xf16>

// CHECK-LABEL: func @copy_1d_20xf16
func.func @copy_1d_20xf16(%t0: !tt, %out: !tt) -> !tt {
  /// Too little data for all threads, needs predication, while keeping most
  /// minor transfer size -> 5 threads.
  // CHECK: scf.forall {{.*}} in (5) {{.*}}
  // CHECK:   linalg.copy {{.*}} -> tensor<4xf16>
  // CHECK: {mapping = [#gpu.linear<x>]}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 32 desired_bit_alignment = 128
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}

// -----

!tt = tensor<128xf16>

// CHECK-LABEL: func @copy_1d_128xf16
func.func @copy_1d_128xf16(%t0: !tt, %out: !tt) -> !tt {
  /// Enough data for all threads and no need for predication but we must reduce
  /// the transfer size to 4xf16.
  // CHECK: scf.forall {{.*}} in (32) {{.*}}
  // CHECK:   linalg.copy {{.*}} -> tensor<4xf16>
  // CHECK: {mapping = [#gpu.linear<x>]}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 32 desired_bit_alignment = 128
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}

// -----

!tt = tensor<256xf16>

// CHECK-LABEL: func @copy_1d_256xf16
func.func @copy_1d_256xf16(%t0: !tt, %out: !tt) -> !tt {
  /// Enough data for all threads and no need for predication.
  // CHECK: scf.forall {{.*}} in (32) {{.*}}
  // CHECK:   linalg.copy {{.*}} -> tensor<8xf16>
  // CHECK: {mapping = [#gpu.linear<x>]}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 32 desired_bit_alignment = 128
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}

// -----

!tt = tensor<16x32x64xi8>

// CHECK-LABEL: func @copy_3d_16x32x64xi8
func.func @copy_3d_16x32x64xi8(%t0: !tt, %out: !tt) -> !tt {
  // CHECK: scf.forall {{.*}} in (1, 8, 4) {{.*}}
  // CHECK:   linalg.copy {{.*}} -> tensor<16x4x16xi8>
  // CHECK: {mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 32 desired_bit_alignment = 128
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}

// -----

!tt = tensor<16x32x64xi8>

// CHECK-LABEL: func @copy_3d_16x32x64xi8
func.func @copy_3d_16x32x64xi8(%t0: !tt, %out: !tt) -> !tt {
  // CHECK: scf.forall {{.*}} in (1, 4, 8) {{.*}}
  // CHECK:   linalg.copy {{.*}} -> tensor<16x8x8xi8>
  // CHECK: {mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 32 desired_bit_alignment = 64
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}

// -----

!tt = tensor<4x8x16xi8>

// CHECK-LABEL: func @copy_3d_4x8x16xi8
func.func @copy_3d_4x8x16xi8(%t0: !tt, %out: !tt) -> !tt {
  // CHECK: scf.forall {{.*}} in (4, 8, 1) {{.*}}
  // CHECK:   linalg.copy {{.*}} -> tensor<1x1x16xi8>
  // CHECK: {mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 32 desired_bit_alignment = 128
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}

// -----

!tt = tensor<4x8x16xi8>

// CHECK-LABEL: func @copy_3d_4x8x16xi8
func.func @copy_3d_4x8x16xi8(%t0: !tt, %out: !tt) -> !tt {
  // CHECK: scf.forall {{.*}} in (1, 2, 16) {{.*}}
  // CHECK:   linalg.copy {{.*}} -> tensor<4x4x1xi8>
  // CHECK: {mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 32 desired_bit_alignment = 8
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}

// -----

!tt = tensor<3x5x7xi8>

// CHECK-LABEL: func @copy_3d_3x5x7xi8
func.func @copy_3d_3x5x7xi8(%t0: !tt, %out: !tt) -> !tt {
  // Best effort greedy mapping: first 7, then skip 5 (as 7*5 overflows 32), then
  // take 3.
  // DP mapping: 7 mandated most minor, then skip 5  (as 7*5 overflows 32), then
  // take 3.
  // CHECK: scf.forall {{.*}} in (3, 1, 7) {{.*}}
  // CHECK:   linalg.copy {{.*}} -> tensor<1x5x1xi8>
  // CHECK: {mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 32 desired_bit_alignment = 8
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}

// -----

!tt = tensor<16x15x5xi8>

// CHECK-LABEL: func @copy_3d_16x15x5xi8
func.func @copy_3d_16x15x5xi8(%t0: !tt, %out: !tt) -> !tt {
  // DP mapping: 5 mandated most minor, then 3 to allow 8 on the outermost.
  // CHECK: scf.forall {{.*}} in (8, 3, 5) {{.*}}
  // CHECK:   linalg.copy {{.*}} -> tensor<2x5x1xi8>
  // CHECK: {mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 128 desired_bit_alignment = 8
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}

// -----

!tt = tensor<16x15x40xi8>

// CHECK-LABEL: func @copy_3d_16x15x40xi8
func.func @copy_3d_16x15x40xi8(%t0: !tt, %out: !tt) -> !tt {
  // DP mapping: 5 mandated most minor, then 3 to allow 8 on the outermost.
  // CHECK: scf.forall {{.*}} in (8, 3, 5) {{.*}}
  // CHECK:   linalg.copy {{.*}} -> tensor<2x5x8xi8>
  // CHECK: {mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 128 desired_bit_alignment = 64
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}


////////////////////////////////////////////////////////////////////////////////
// Tests below are expected to fail.
////////////////////////////////////////////////////////////////////////////////

// -----

!tt = tensor<1024xf16>

// NO-CHECK-LABEL-ON-EXPECTED-ERROR
func.func @copy_1d_1024xf16(%t0: !tt, %out: !tt) -> !tt {
  /// Too much data for all threads, we do not try to recover here, this is the
  /// job of higher-level transformations to select better tile sizes and number
  /// of threads.

  // expected-note @below {{target op}}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{too few threads to map copy op to threads on the most minor dimension, given alignment and vector size constraints}}
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 32 desired_bit_alignment = 128
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}

// -----

!tt = tensor<257xf16>

// NO-CHECK-LABEL-ON-EXPECTED-ERROR
func.func @copy_1d_257xf16(%t0: !tt, %out: !tt) -> !tt {
  /// Too much data for all threads, we do not try to recover here, this is the
  /// job of higher-level transformations to select better tile sizes and number
  /// of threads.
  
  // expected-note @below {{target op}}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{too few threads to map copy op to threads on the most minor dimension, given alignment and vector size constraints}}
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 32 desired_bit_alignment = 128
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}

// -----

!tt = tensor<512xi8>

// NO-CHECK-LABEL-ON-EXPECTED-ERROR
func.func @copy_1d_512xi8(%t0: !tt, %out: !tt) -> !tt {
  /// Too much data for all threads given the forced alignment to 8b, 
  /// we do not try to recover here, this is the job of higher-level 
  /// transformations to select better tile sizes and number of threads.
  // expected-note @below {{target op}}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{too few threads to map copy op to threads on the most minor dimension, given alignment and vector size constraints}}
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 32 desired_bit_alignment = 8
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}

// -----

!tt = tensor<16x32x64xi8>

// NO-CHECK-LABEL-ON-EXPECTED-ERROR
func.func @copy_3d_16x32x64xi8(%t0: !tt, %out: !tt) -> !tt {
  /// Too much data for all threads given the forced alignment to 8b, 
  /// we do not try to recover here, this is the job of higher-level 
  /// transformations to select better tile sizes and number of threads.
  // expected-note @below {{target op}}
  %0 = linalg.copy ins(%t0: !tt) outs(%out: !tt) -> !tt 
  return %0 : !tt
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{too few threads to map copy op to threads on the most minor dimension, given alignment and vector size constraints}}
  transform.structured.gpu.map_copy_to_threads %0 
    total_num_threads = 32 desired_bit_alignment = 8
      : (!transform.any_op) -> (!transform.op<"scf.forall">, !transform.op<"linalg.copy">)
}
