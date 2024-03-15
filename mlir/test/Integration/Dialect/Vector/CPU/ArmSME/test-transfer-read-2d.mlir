// DEFINE: %{entry_point} = entry
// DEFINE: %{compile} = mlir-opt %s -test-lower-to-arm-sme -test-lower-to-llvm
// DEFINE: %{run} = %mcr_aarch64_cmd \
// DEFINE:  -march=aarch64 -mattr=+sve,+sme \
// DEFINE:  -e %{entry_point} -entry-point-result=void \
// DEFINE:  -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%arm_sme_abi_shlib

// RUN: %{compile} | %{run} | FileCheck %s

// 2-D vector load (SME tile).
func.func @transfer_read_2d(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %c4 = arith.constant 4 : index
  %pad = arith.constant 0.0 : f32
  %0 = vector.transfer_read %A[%base1, %base2], %pad {in_bounds=[true, true]} :
    memref<?x?xf32>, vector<[4]x[4]xf32>

  vector.print str "TILE BEGIN:\n"
  vector.print %0: vector<[4]x[4]xf32>

  return
}

// 2-D vector load (SME tile) + transpose.
func.func @transfer_read_2d_transposed(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %pad = arith.constant 0.0 : f32
  %0 = vector.transfer_read %A[%base1, %base2], %pad
    {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds=[true, true]}
      : memref<?x?xf32>, vector<[4]x[4]xf32>

  vector.print str "TILE BEGIN:\n"
  vector.print %0 : vector<[4]x[4]xf32>

  return
}

// 2-D vector load (SME tile) with mask and pad of zero.
func.func @transfer_read_2d_mask(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %pad = arith.constant 0.0 : f32
  %mask = vector.create_mask %c2, %c3 : vector<[4]x[4]xi1>
  %0 = vector.transfer_read %A[%base1, %base2], %pad, %mask
    {in_bounds = [true, true]} : memref<?x?xf32>, vector<[4]x[4]xf32>

  vector.print str "TILE BEGIN:\n"
  vector.print %0: vector<[4]x[4]xf32>

  return
}

// 2-D vector load (SME tile) with mask and pad of zero + transpose.
func.func @transfer_read_2d_mask_transposed(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %pad = arith.constant 0.0 : f32
  %mask = vector.create_mask %c2, %c3 : vector<[4]x[4]xi1>
  %0 = vector.transfer_read %A[%base1, %base2], %pad, %mask
    {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds=[true, true]}
      : memref<?x?xf32>, vector<[4]x[4]xf32>

  vector.print str "TILE BEGIN:\n"
  vector.print %0: vector<[4]x[4]xf32>

  return
}

// 2-D vector load (SME tile) with mask and non-zero pad.
func.func @transfer_read_2d_mask_non_zero_pad(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %pad = arith.constant -42.0 : f32
  %mask = vector.create_mask %c2, %c3 : vector<[4]x[4]xi1>
  %0 = vector.transfer_read %A[%base1, %base2], %pad, %mask
    {in_bounds = [true, true]} : memref<?x?xf32>, vector<[4]x[4]xf32>

  vector.print str "TILE BEGIN:\n"
  vector.print %0: vector<[4]x[4]xf32>

  return
}

// 2-D vector load (SME tile) with mask and non-zero pad + transpose.
func.func @transfer_read_2d_mask_non_zero_pad_transposed(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %pad = arith.constant -42.0 : f32
  %mask = vector.create_mask %c2, %c3 : vector<[4]x[4]xi1>
  %0 = vector.transfer_read %A[%base1, %base2], %pad, %mask
    {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds=[true, true]}
      : memref<?x?xf32>, vector<[4]x[4]xf32>

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
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  // Allocate enough memory to load a 32-bit tile plus a tiny bit more to test
  // non-zero offsets while remaining inbounds.
  %svl_s = arm_sme.streaming_vl <word>
  %svl_s_plus_two = arith.addi %svl_s, %c2 : index

  %A = call @initialize_memory(%svl_s_plus_two, %svl_s_plus_two) : (index, index) -> memref<?x?xf32>

  // 1.a. Read 2D vector from 2D memref.
  //
  // CHECK-LABEL: TILE BEGIN:
  // CHECK-NEXT: ( 0, 1, 2, 3
  // CHECK-NEXT: ( 10, 11, 12, 13
  // CHECK-NEXT: ( 20, 21, 22, 23
  // CHECK-NEXT: ( 30, 31, 32, 33
  call @transfer_read_2d(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()

  // 1.b. Same as 1.a., but with non-zero offsets.
  //
  // CHECK-LABEL: TILE BEGIN:
  // CHECK-NEXT: ( 12, 13, 14, 15
  // CHECK-NEXT: ( 22, 23, 24, 25
  // CHECK-NEXT: ( 32, 33, 34, 35
  // CHECK-NEXT: ( 42, 43, 44, 45
  call @transfer_read_2d(%A, %c1, %c2) : (memref<?x?xf32>, index, index) -> ()

  // 2. Same as 1.a., but with mask and a pad of constant zero.
  // CHECK-LABEL: TILE BEGIN:
  // CHECK-NEXT: ( 0, 1, 2, 0
  // CHECK-NEXT: ( 10, 11, 12, 0
  // CHECK-NEXT: ( 0, 0, 0, 0
  // CHECK-NEXT: ( 0, 0, 0, 0
  call @transfer_read_2d_mask(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()

  // 3. Same as 1.a., but with mask and non-zero pad.
  // CHECK-LABEL: TILE BEGIN:
  // CHECK-NEXT: ( 0, 1, 2, -42
  // CHECK-NEXT: ( 10, 11, 12, -42
  // CHECK-NEXT: ( -42, -42, -42, -42
  // CHECK-NEXT: ( -42, -42, -42, -42
  call @transfer_read_2d_mask_non_zero_pad(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()

  // 4. Same as 1.a., but transpose the result.
  // CHECK-LABEL: TILE BEGIN:
  // CHECK-NEXT: ( 0, 10, 20, 30
  // CHECK-NEXT: ( 1, 11, 21, 31
  // CHECK-NEXT: ( 2, 12, 22, 32
  // CHECK-NEXT: ( 3, 13, 23, 33
  call @transfer_read_2d_transposed(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()

  // 5. Same as 2., but transpose the result.
  // CHECK-LABEL: TILE BEGIN:
  // CHECK-NEXT: ( 0, 10, 0, 0
  // CHECK-NEXT: ( 1, 11, 0, 0
  // CHECK-NEXT: ( 2, 12, 0, 0
  // CHECK-NEXT: ( 0, 0, 0, 0
  call @transfer_read_2d_mask_transposed(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()

  // 5. Same as 3, but transpose the result.
  // CHECK-LABEL: TILE BEGIN:
  // CHECK-NEXT: ( 0, 10, -42, -42
  // CHECK-NEXT: ( 1, 11, -42, -42
  // CHECK-NEXT: ( 2, 12, -42, -42
  // CHECK-NEXT: ( -42, -42, -42, -42
  call @transfer_read_2d_mask_non_zero_pad_transposed(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()

  memref.dealloc %A : memref<?x?xf32>

  return
}
