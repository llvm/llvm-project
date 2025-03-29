// DEFINE: %{entry_point} = entry
// DEFINE: %{compile} = mlir-opt %s -test-lower-to-arm-sme -test-lower-to-llvm
// DEFINE: %{run} = %mcr_aarch64_cmd \
// DEFINE:  -march=aarch64 -mattr=+sve,+sme \
// DEFINE:  -e %{entry_point} -entry-point-result=void \
// DEFINE:  -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils,%native_arm_sme_abi_shlib

// RUN: %{compile} | %{run} | FileCheck %s

// Vector store.
func.func @transfer_write_2d(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %c0 = arith.constant 0.0 : f32
  %zero = vector.splat %c0 : vector<[4]x[4]xf32>
  vector.transfer_write %zero, %A[%base1, %base2] {in_bounds=[true, true]} :
    vector<[4]x[4]xf32>, memref<?x?xf32>
  return
}

// Masked vector store.
func.func @transfer_write_2d_mask(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %c0 = arith.constant 0.0 : f32
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %mask = vector.create_mask %c2, %c3 : vector<[4]x[4]xi1>
  %zero = vector.splat %c0 : vector<[4]x[4]xf32>
  vector.transfer_write %zero, %A[%base1, %base2], %mask {in_bounds=[true, true]} :
    vector<[4]x[4]xf32>, memref<?x?xf32>
  return
}

// Vector transpose + store.
func.func @transfer_write_2d_transposed(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %0 = vector.load %A[%base1, %base2] : memref<?x?xf32>, vector<[4]x[4]xf32>
  vector.transfer_write %0, %A[%base1, %base2] {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds=[true, true]} :
    vector<[4]x[4]xf32>, memref<?x?xf32>
  return
}

// Vector transpose + masked store.
func.func @transfer_write_2d_mask_transposed(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %mask = vector.create_mask %c4, %c2 : vector<[4]x[4]xi1>
  %0 = vector.load %A[%base1, %base2] : memref<?x?xf32>, vector<[4]x[4]xf32>
  vector.transfer_write %0, %A[%base1, %base2], %mask {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds=[true, true]} :
    vector<[4]x[4]xf32>, memref<?x?xf32>
  return
}

// Vector load + print.
func.func @load_and_print(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %0 = vector.load %A[%base1, %base2] : memref<?x?xf32>, vector<[4]x[4]xf32>

  vector.print str "TILE BEGIN:\n"
  vector.print %0: vector<[4]x[4]xf32>

  return
}

// Allocate heap memory of size 'd0' x 'd1' and initialize.
//
// Example:
//
// initialize_memory(%c4, %c5)
//
//    0,  1,  2,  3,  4
//   10, 11, 12, 13, 14
//   20, 21, 22, 23, 24
//   30, 31, 32, 33, 34
//
// Returns dynamic memref. It's the callers responsiblity to free the returned
// memref.
func.func @initialize_memory(%d0 : index, %d1 : index) -> memref<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_f32 = arith.constant 1.0 : f32
  %c10_f32 = arith.constant 10.0 : f32

  %A = memref.alloc(%d0, %d1) : memref<?x?xf32>

  %init = arith.constant 0.0 : f32
  scf.for %i = %c0 to %d0 step %c1 iter_args(%val = %init) -> f32 {
    scf.for %j = %c0 to %d1 step %c1 iter_args(%inner_val = %val) -> f32 {
      memref.store %inner_val, %A[%i, %j] : memref<?x?xf32>
      %inner_val_next = arith.addf %inner_val, %c1_f32 : f32
      scf.yield %inner_val_next : f32
    }
    %val_next = arith.addf %val, %c10_f32 : f32
    scf.yield %val_next : f32
  }

  return %A : memref<?x?xf32>
}

func.func @entry() {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  // 1. Initialize memory
  //
  // Allocate enough memory to load a 32-bit tile plus a tiny bit more to test
  // non-zero offsets while remaining inbounds.
  %svl_s = arm_sme.streaming_vl <word>
  %svl_s_plus_two = arith.addi %svl_s, %c2 : index
  %A = call @initialize_memory(%svl_s_plus_two, %svl_s_plus_two) : (index, index) -> memref<?x?xf32>

  // CHECK-LABEL: TILE BEGIN:
  // CHECK-NEXT: ( 0, 1, 2, 3
  // CHECK-NEXT: ( 10, 11, 12, 13
  // CHECK-NEXT: ( 20, 21, 22, 23
  // CHECK-NEXT: ( 30, 31, 32, 33
  call @load_and_print(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()

  // 2. Write 2-D vector of zeroes to 1. at offset [2, 2].
  // CHECK-LABEL: TILE BEGIN:
  // CHECK-NEXT: ( 0, 1, 2, 3
  // CHECK-NEXT: ( 10, 11, 12, 13
  // CHECK-NEXT: ( 20, 21, 0, 0
  // CHECK-NEXT: ( 30, 31, 0, 0
  call @transfer_write_2d(%A, %c2, %c2) : (memref<?x?xf32>, index, index) -> ()
  call @load_and_print(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()

  // 3. Write 2-D vector of zeroes to 2. but with mask (nrows=2, ncols=3).
  // CHECK-LABEL: TILE BEGIN:
  // CHECK-NEXT: ( 0, 0, 0, 3
  // CHECK-NEXT: ( 0, 0, 0, 13
  // CHECK-NEXT: ( 20, 21, 0, 0
  // CHECK-NEXT: ( 30, 31, 0, 0
  call @transfer_write_2d_mask(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()
  call @load_and_print(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()

  // 4. Reload 3. + transpose + store.
  // CHECK-LABEL: TILE BEGIN:
  // CHECK-NEXT: ( 0, 0, 20, 30
  // CHECK-NEXT: ( 0, 0, 21, 31
  // CHECK-NEXT: ( 0, 0, 0, 0
  // CHECK-NEXT: ( 3, 13, 0, 0
  call @transfer_write_2d_transposed(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()
  call @load_and_print(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()

  // 5. Reload 4. + transpose + masked store (nrows=4, ncols=2).
  // The mask applies after permutation. Columns 2 and 3 (from 4.) are
  // preserved.
  // CHECK-LABEL: TILE BEGIN:
  // CHECK-NEXT: ( 0, 0, 20, 30
  // CHECK-NEXT: ( 0, 0, 21, 31
  // CHECK-NEXT: ( 20, 21, 0, 0
  // CHECK-NEXT: ( 30, 31, 0, 0
  call @transfer_write_2d_mask_transposed(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()
  call @load_and_print(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()

  memref.dealloc %A : memref<?x?xf32>

  return
}
