// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @transfer_read_1d(%A : memref<?xf32>, %base: index) {
  %fm42 = constant -42.0: f32
  %f = vector.transfer_read %A[%base], %fm42
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32>, vector<13xf32>
  vector.print %f: vector<13xf32>
  return
}

func @transfer_read_unmasked_4(%A : memref<?xf32>, %base: index) {
  %fm42 = constant -42.0: f32
  %f = vector.transfer_read %A[%base], %fm42
      {permutation_map = affine_map<(d0) -> (d0)>, masked = [false]} :
    memref<?xf32>, vector<4xf32>
  vector.print %f: vector<4xf32>
  return
}

func @transfer_write_1d(%A : memref<?xf32>, %base: index) {
  %f0 = constant 0.0 : f32
  %vf0 = splat %f0 : vector<4xf32>
  vector.transfer_write %vf0, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<4xf32>, memref<?xf32>
  return
}

func @entry() {
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c2 = constant 2: index
  %c3 = constant 3: index
  %c4 = constant 4: index
  %c5 = constant 5: index
  %A = alloc(%c5) : memref<?xf32>
  scf.for %i = %c0 to %c5 step %c1 {
    %i32 = index_cast %i : index to i32
    %fi = sitofp %i32 : i32 to f32
    store %fi, %A[%i] : memref<?xf32>
  }
  // On input, memory contains [[ 0, 1, 2, 3, 4, xxx garbage xxx ]]
  // Read shifted by 2 and pad with -42:
  //   ( 2, 3, 4, -42, ..., -42)
  call @transfer_read_1d(%A, %c2) : (memref<?xf32>, index) -> ()
  // Write into memory shifted by 3
  //   memory contains [[ 0, 1, 2, 0, 0, xxx garbage xxx ]]
  call @transfer_write_1d(%A, %c3) : (memref<?xf32>, index) -> ()
  // Read shifted by 0 and pad with -42:
  //   ( 0, 1, 2, 0, 0, -42, ..., -42)
  call @transfer_read_1d(%A, %c0) : (memref<?xf32>, index) -> ()
  // Read unmasked 4 @ 1, guaranteed to not overflow.
  // Exercises proper alignment.
  call @transfer_read_unmasked_4(%A, %c1) : (memref<?xf32>, index) -> ()
  return
}

// CHECK: ( 2, 3, 4, -42, -42, -42, -42, -42, -42, -42, -42, -42, -42 )
// CHECK: ( 0, 1, 2, 0, 0, -42, -42, -42, -42, -42, -42, -42, -42 )
// CHECK: ( 1, 2, 0, 0 )
