// RUN: mlir-opt %s -convert-vector-to-scf -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -convert-vector-to-scf=full-unroll=true -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @transfer_read_1d(%A : memref<?xf32>, %base: index) {
  %fm42 = arith.constant -42.0: f32
  %f = vector.transfer_read %A[%base], %fm42
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32>, vector<13xf32>
  vector.print %f: vector<13xf32>
  return
}

func @transfer_read_mask_1d(%A : memref<?xf32>, %base: index) {
  %fm42 = arith.constant -42.0: f32
  %m = arith.constant dense<[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]> : vector<13xi1>
  %f = vector.transfer_read %A[%base], %fm42, %m : memref<?xf32>, vector<13xf32>
  vector.print %f: vector<13xf32>
  return
}

func @transfer_read_inbounds_4(%A : memref<?xf32>, %base: index) {
  %fm42 = arith.constant -42.0: f32
  %f = vector.transfer_read %A[%base], %fm42
      {permutation_map = affine_map<(d0) -> (d0)>, in_bounds = [true]} :
    memref<?xf32>, vector<4xf32>
  vector.print %f: vector<4xf32>
  return
}

func @transfer_read_mask_inbounds_4(%A : memref<?xf32>, %base: index) {
  %fm42 = arith.constant -42.0: f32
  %m = arith.constant dense<[0, 1, 0, 1]> : vector<4xi1>
  %f = vector.transfer_read %A[%base], %fm42, %m {in_bounds = [true]}
      : memref<?xf32>, vector<4xf32>
  vector.print %f: vector<4xf32>
  return
}

func @transfer_write_1d(%A : memref<?xf32>, %base: index) {
  %f0 = arith.constant 0.0 : f32
  %vf0 = splat %f0 : vector<4xf32>
  vector.transfer_write %vf0, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<4xf32>, memref<?xf32>
  return
}

func @entry() {
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %c3 = arith.constant 3: index
  %c4 = arith.constant 4: index
  %c5 = arith.constant 5: index
  %A = memref.alloc(%c5) : memref<?xf32>
  scf.for %i = %c0 to %c5 step %c1 {
    %i32 = arith.index_cast %i : index to i32
    %fi = arith.sitofp %i32 : i32 to f32
    memref.store %fi, %A[%i] : memref<?xf32>
  }
  // On input, memory contains [[ 0, 1, 2, 3, 4, xxx garbage xxx ]]
  // Read shifted by 2 and pad with -42:
  //   ( 2, 3, 4, -42, ..., -42)
  call @transfer_read_1d(%A, %c2) : (memref<?xf32>, index) -> ()
  // Read with mask and out-of-bounds access.
  call @transfer_read_mask_1d(%A, %c2) : (memref<?xf32>, index) -> ()
  // Write into memory shifted by 3
  //   memory contains [[ 0, 1, 2, 0, 0, xxx garbage xxx ]]
  call @transfer_write_1d(%A, %c3) : (memref<?xf32>, index) -> ()
  // Read shifted by 0 and pad with -42:
  //   ( 0, 1, 2, 0, 0, -42, ..., -42)
  call @transfer_read_1d(%A, %c0) : (memref<?xf32>, index) -> ()
  // Read in-bounds 4 @ 1, guaranteed to not overflow.
  // Exercises proper alignment.
  call @transfer_read_inbounds_4(%A, %c1) : (memref<?xf32>, index) -> ()
  // Read in-bounds with mask.
  call @transfer_read_mask_inbounds_4(%A, %c1) : (memref<?xf32>, index) -> ()

  memref.dealloc %A : memref<?xf32>

  return
}

// CHECK: ( 2, 3, 4, -42, -42, -42, -42, -42, -42, -42, -42, -42, -42 )
// CHECK: ( -42, -42, 4, -42, -42, -42, -42, -42, -42, -42, -42, -42, -42 )
// CHECK: ( 0, 1, 2, 0, 0, -42, -42, -42, -42, -42, -42, -42, -42 )
// CHECK: ( 1, 2, 0, 0 )
// CHECK: ( -42, 2, -42, 0 )
