// RUN: mlir-opt %s -convert-vector-to-scf -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @transfer_read_2d(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %fm42 = constant -42.0: f32
  %f = vector.transfer_read %A[%base1, %base2], %fm42
      {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} :
    memref<?x?xf32>, vector<4x9xf32>
  vector.print %f: vector<4x9xf32>
  return
}

func @transfer_write_2d(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %fn1 = constant -1.0 : f32
  %vf0 = splat %fn1 : vector<1x4xf32>
  vector.transfer_write %vf0, %A[%base1, %base2]
    {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} :
    vector<1x4xf32>, memref<?x?xf32>
  return
}

func @entry() {
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c2 = constant 2: index
  %c3 = constant 3: index
  %c4 = constant 4: index
  %c5 = constant 5: index
  %c8 = constant 5: index
  %f10 = constant 10.0: f32
  // work with dims of 4, not of 3
  %first = constant 3: index
  %second = constant 4: index
  %A = memref.alloc(%first, %second) : memref<?x?xf32>
  scf.for %i = %c0 to %first step %c1 {
    %i32 = index_cast %i : index to i32
    %fi = sitofp %i32 : i32 to f32
    %fi10 = mulf %fi, %f10 : f32
    scf.for %j = %c0 to %second step %c1 {
        %j32 = index_cast %j : index to i32
        %fj = sitofp %j32 : i32 to f32
        %fres = addf %fi10, %fj : f32
        memref.store %fres, %A[%i, %j] : memref<?x?xf32>
    }
  }
  // On input, memory contains [[ 0, 1, 2, ...], [10, 11, 12, ...], ...]
  // Read shifted by 2 and pad with -42:
  call @transfer_read_2d(%A, %c1, %c2) : (memref<?x?xf32>, index, index) -> ()
  // Write into memory shifted by 3
  call @transfer_write_2d(%A, %c3, %c1) : (memref<?x?xf32>, index, index) -> ()
  // Read shifted by 0 and pad with -42:
  call @transfer_read_2d(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()
  return
}

// CHECK: ( ( 12, 13, -42, -42, -42, -42, -42, -42, -42 ), ( 22, 23, -42, -42, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ) )
// CHECK: ( ( 0, 1, 2, 3, -42, -42, -42, -42, -42 ), ( 10, 11, 12, 13, -42, -42, -42, -42, -42 ), ( 20, 21, 22, 23, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ) )
