// Run test with and without test-vector-transfer-lowering-patterns.

// RUN: mlir-opt %s -convert-vector-to-scf -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -test-vector-transfer-lowering-patterns -convert-vector-to-scf -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s


memref.global "private" @gv : memref<3x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                       [10., 11., 12., 13.],
                                                       [20., 21., 22., 23.]]>

// Vector load with transpose.
func @transfer_read_2d(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %fm42 = constant -42.0: f32
  %f = vector.transfer_read %A[%base1, %base2], %fm42
      {in_bounds = [true, false], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} :
    memref<?x?xf32>, vector<3x9xf32>
  vector.print %f: vector<3x9xf32>
  return
}

func @entry() {
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c2 = constant 2: index
  %c3 = constant 3: index
  %0 = memref.get_global @gv : memref<3x4xf32>
  %A = memref.cast %0 : memref<3x4xf32> to memref<?x?xf32>

  // 1. Read 2D vector from 2D memref with transpose.
  call @transfer_read_2d(%A, %c1, %c2) : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( ( 12, 22, -42, -42, -42, -42, -42, -42, -42 ), ( 13, 23, -42, -42, -42, -42, -42, -42, -42 ), ( 20, 0, -42, -42, -42, -42, -42, -42, -42 ) )

  return
}
