// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#dotp_accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> ()>
]
#dotp_trait = {
  indexing_maps = #dotp_accesses,
  iterator_types = ["reduction"]
}

#matvec_accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (j)>,
  affine_map<(i, j) -> (i)>
]
#matvec_trait = {
  indexing_maps = #matvec_accesses,
  iterator_types = ["parallel", "reduction"]
}

#mattransvec_accesses = [
  affine_map<(i, j) -> (j, i)>,
  affine_map<(i, j) -> (j)>,
  affine_map<(i, j) -> (i)>
]
#mattransvec_trait = {
  indexing_maps = #mattransvec_accesses,
  iterator_types = ["parallel", "reduction"]
}

#matmat_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

#mattransmat_accesses = [
  affine_map<(i, j, k) -> (k, i)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#mattransmat_trait = {
  indexing_maps = #mattransmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

#matmattrans_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (j, k)>,
  affine_map<(i, j, k) -> (i, j)>
]
#matmattrans_trait = {
  indexing_maps = #matmattrans_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

#mattransmattrans_accesses = [
  affine_map<(i, j, k) -> (k, i)>,
  affine_map<(i, j, k) -> (j, k)>,
  affine_map<(i, j, k) -> (i, j)>
]
#mattransmattrans_trait = {
  indexing_maps = #mattransmattrans_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

#matmat_then_trans_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (j, i)>
]
#matmat_then_trans_trait = {
  indexing_maps = #matmat_then_trans_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

#contract2d_accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> ()>
]
#contract2d_trait = {
  indexing_maps = #contract2d_accesses,
  iterator_types = ["reduction", "reduction"]
}

#contract2d_alt_accesses = [
  affine_map<(i, j) -> (j, i)>,
  affine_map<(i, j) -> (j, i)>,
  affine_map<(i, j) -> ()>
]
#contract2d_alt_trait = {
  indexing_maps = #contract2d_alt_accesses,
  iterator_types = ["reduction", "reduction"]
}

#contract2d_trans_accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (j, i)>,
  affine_map<(i, j) -> ()>
]
#contract2d_trans_trait = {
  indexing_maps = #contract2d_trans_accesses,
  iterator_types = ["reduction", "reduction"]
}

#contract2d_trans_alt_accesses = [
  affine_map<(i, j) -> (j, i)>,
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> ()>
]
#contract2d_trans_alt_trait = {
  indexing_maps = #contract2d_trans_alt_accesses,
  iterator_types = ["reduction", "reduction"]
}

#column_major_matmat_accesses = [
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (j, i)>
]
#column_major_matmat_trait = {
  indexing_maps = #column_major_matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

func @entry() {
  %f0 = arith.constant 0.0: f32
  %f1 = arith.constant 1.0: f32
  %f2 = arith.constant 2.0: f32
  %f3 = arith.constant 3.0: f32
  %f4 = arith.constant 4.0: f32
  %f5 = arith.constant 5.0: f32
  %f6 = arith.constant 6.0: f32
  %f7 = arith.constant 7.0: f32
  %f8 = arith.constant 8.0: f32

  // Zero vectors.
  %z1 = vector.broadcast %f0 : f32 to vector<2xf32>
  %z2 = vector.broadcast %f0 : f32 to vector<2x2xf32>
  %z3 = vector.broadcast %f0 : f32 to vector<3x4xf32>

  // Construct test vectors.
  %0 = vector.broadcast %f1 : f32 to vector<2xf32>
  %a = vector.insert %f2, %0[1] : f32 into vector<2xf32>
  %1 = vector.broadcast %f3 : f32 to vector<2xf32>
  %b = vector.insert %f4, %1[1] : f32 into vector<2xf32>
  %2 = vector.broadcast %f5 : f32 to vector<2xf32>
  %c = vector.insert %f6, %2[1] : f32 into vector<2xf32>
  %3 = vector.broadcast %f7 : f32 to vector<2xf32>
  %d = vector.insert %f8, %3[1] : f32 into vector<2xf32>

  vector.print %a : vector<2xf32>
  vector.print %b : vector<2xf32>
  vector.print %c : vector<2xf32>
  vector.print %d : vector<2xf32>
  //
  // test vectors:
  //
  // CHECK: ( 1, 2 )
  // CHECK: ( 3, 4 )
  // CHECK: ( 5, 6 )
  // CHECK: ( 7, 8 )

  // Construct test matrices.
  %4 = vector.broadcast %f0 : f32 to vector<2x2xf32>
  %5 = vector.insert %a, %4[0] : vector<2xf32> into vector<2x2xf32>
  %A = vector.insert %b, %5[1] : vector<2xf32> into vector<2x2xf32>
  %6 = vector.broadcast %f0 : f32 to vector<2x2xf32>
  %7 = vector.insert %c, %6[0] : vector<2xf32> into vector<2x2xf32>
  %B = vector.insert %d, %7[1] : vector<2xf32> into vector<2x2xf32>
  %8 = vector.broadcast %f0 : f32 to vector<3x2xf32>
  %9 = vector.insert %a, %8[0] : vector<2xf32> into vector<3x2xf32>
  %10 = vector.insert %b, %9[1] : vector<2xf32> into vector<3x2xf32>
  %C = vector.insert %c, %10[2] : vector<2xf32> into vector<3x2xf32>
  %cst = arith.constant dense<0.000000e+00> : vector<2x4xf32>
  %11 = vector.insert_strided_slice %A, %cst {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<2x4xf32>
  %D = vector.insert_strided_slice %B, %11 {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<2x4xf32>

  vector.print %A : vector<2x2xf32>
  vector.print %B : vector<2x2xf32>
  vector.print %C : vector<3x2xf32>
  vector.print %D : vector<2x4xf32>
  //
  // test matrices:
  //
  // CHECK: ( ( 1, 2 ), ( 3, 4 ) )
  // CHECK: ( ( 5, 6 ), ( 7, 8 ) )
  // CHECK: ( ( 1, 2 ), ( 3, 4 ), ( 5, 6 ) )
  // CHECK: ( ( 1, 2, 5, 6 ), ( 3, 4, 7, 8 ) )

  // Contraction: dot-product a x b
  %dp1 = vector.contract #dotp_trait %a, %b, %f0
    : vector<2xf32>, vector<2xf32> into f32
  %dp2 = vector.contract #dotp_trait %a, %b, %f1
    : vector<2xf32>, vector<2xf32> into f32

  vector.print %dp1 : f32
  vector.print %dp2 : f32
  //
  // dot products:
  //
  // CHECK: 11
  // CHECK: 12

  // Contraction: matrix-vector A x c
  %mv1 = vector.contract #matvec_trait %A, %c, %z1
    : vector<2x2xf32>, vector<2xf32> into vector<2xf32>
  %mv2 = vector.contract #matvec_trait %A, %c, %a
    : vector<2x2xf32>, vector<2xf32> into vector<2xf32>

  vector.print %mv1 : vector<2xf32>
  vector.print %mv2 : vector<2xf32>
  //
  // matrix x vector:
  //
  // CHECK: ( 17, 39 )
  // CHECK: ( 18, 41 )

  // Contraction: matrix-trans-vector A^T x c
  %mv3 = vector.contract #mattransvec_trait %A, %c, %z1
    : vector<2x2xf32>, vector<2xf32> into vector<2xf32>
  %mv4 = vector.contract #mattransvec_trait %A, %c, %a
    : vector<2x2xf32>, vector<2xf32> into vector<2xf32>

  vector.print %mv3 : vector<2xf32>
  vector.print %mv4 : vector<2xf32>
  //
  // matrix x vector:
  //
  // CHECK: ( 23, 34 )
  // CHECK: ( 24, 36 )

  // Contraction: matrix-matrix A x B
  %mm1 = vector.contract #matmat_trait %A, %B, %z2
    : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
  %mm2 = vector.contract #matmat_trait %A, %B, %A
    : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

  vector.print %mm1 : vector<2x2xf32>
  vector.print %mm2 : vector<2x2xf32>
  //
  // matrix x matrix:
  //
  // CHECK: ( ( 19, 22 ), ( 43, 50 ) )
  // CHECK: ( ( 20, 24 ), ( 46, 54 ) )

  // Contraction: matrix-matrix A x B where A, B, C have column-major layout.
  // ( 1 * 5 + 3 * 6 = 23, 2 * 5 + 4 * 6 = 34)
  // ( 1 * 7 + 3 * 8 = 31, 2 * 7 + 4 * 8 = 46)
  // +
  // ( ( 1, 2 ), ( 3, 4 ) )
  %llvm_matrix_column_major_mm0 =
    vector.contract #column_major_matmat_trait %A, %B, %z2
      : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
  %llvm_matrix_column_major_mm1 =
    vector.contract #column_major_matmat_trait %A, %B, %A
      : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

  vector.print %llvm_matrix_column_major_mm0 : vector<2x2xf32>
  vector.print %llvm_matrix_column_major_mm1 : vector<2x2xf32>
  //
  // matrix x matrix:
  //
  // CHECK: ( ( 23, 31 ), ( 34, 46 ) )
  // CHECK: ( ( 24, 33 ), ( 37, 50 ) )

  // Contraction: matrix-trans-matrix A^T x B
  %mm3 = vector.contract #mattransmat_trait %A, %B, %z2
    : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
  %mm4 = vector.contract #mattransmat_trait %A, %B, %A
    : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

  vector.print %mm3 : vector<2x2xf32>
  vector.print %mm4 : vector<2x2xf32>
  //
  // matrix x matrix:
  //
  // CHECK: ( ( 26, 30 ), ( 38, 44 ) )
  // CHECK: ( ( 27, 32 ), ( 41, 48 ) )

  // Contraction: matrix-matrix-trans A x B^T
  %mm5 = vector.contract #matmattrans_trait %A, %B, %z2
    : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
  %mm6 = vector.contract #matmattrans_trait %A, %B, %A
    : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

  vector.print %mm5 : vector<2x2xf32>
  vector.print %mm6 : vector<2x2xf32>
  //
  // matrix x matrix:
  //
  // CHECK: ( ( 17, 23 ), ( 39, 53 ) )
  // CHECK: ( ( 18, 25 ), ( 42, 57 ) )

  // Contraction: matrix-trans-matrix-trans A^T x B^T
  %mm7 = vector.contract #mattransmattrans_trait %A, %B, %z2
    : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
  %mm8 = vector.contract #mattransmattrans_trait %A, %B, %A
    : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

  vector.print %mm7 : vector<2x2xf32>
  vector.print %mm8 : vector<2x2xf32>
  //
  // matrix x matrix:
  //
  // CHECK: ( ( 23, 31 ), ( 34, 46 ) )
  // CHECK: ( ( 24, 33 ), ( 37, 50 ) )

  // Contraction: matrix-matrix-then-trans (A x B)^T
  %mm9 = vector.contract #matmat_then_trans_trait %A, %B, %z2
    : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
  %mm10 = vector.contract #matmat_then_trans_trait %A, %B, %A
    : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>

  vector.print %mm9 : vector<2x2xf32>
  vector.print %mm10 : vector<2x2xf32>
  //
  // matrix x matrix:
  //
  // CHECK: ( ( 19, 43 ), ( 22, 50 ) )
  // CHECK: ( ( 20, 45 ), ( 25, 54 ) )

  // Contraction: matrix-matrix C x D
  %mm11 = vector.contract #matmat_trait %C, %D, %z3
    : vector<3x2xf32>, vector<2x4xf32> into vector<3x4xf32>
  %mm12 = vector.contract #matmat_trait %C, %D, %mm11
    : vector<3x2xf32>, vector<2x4xf32> into vector<3x4xf32>

  vector.print %mm11 : vector<3x4xf32>
  vector.print %mm12 : vector<3x4xf32>
  // CHECK: ( ( 7, 10, 19, 22 ), ( 15, 22, 43, 50 ), ( 23, 34, 67, 78 ) )
  // CHECK: ( ( 14, 20, 38, 44 ), ( 30, 44, 86, 100 ), ( 46, 68, 134, 156 ) )

  // Contractions in 2D.
  %c1 = vector.contract #contract2d_trait %A, %B, %f0
    : vector<2x2xf32>, vector<2x2xf32> into f32
  %c2 = vector.contract #contract2d_trait %A, %B, %f1
    : vector<2x2xf32>, vector<2x2xf32> into f32
  %c3 = vector.contract #contract2d_alt_trait %A, %B, %f0
    : vector<2x2xf32>, vector<2x2xf32> into f32
  %c4 = vector.contract #contract2d_alt_trait %A, %B, %f1
    : vector<2x2xf32>, vector<2x2xf32> into f32
  %c5 = vector.contract #contract2d_trans_trait %A, %B, %f0
    : vector<2x2xf32>, vector<2x2xf32> into f32
  %c6 = vector.contract #contract2d_trans_trait %A, %B, %f1
    : vector<2x2xf32>, vector<2x2xf32> into f32
  %c7 = vector.contract #contract2d_trans_alt_trait %A, %B, %f0
    : vector<2x2xf32>, vector<2x2xf32> into f32
  %c8 = vector.contract #contract2d_trans_alt_trait %A, %B, %f1
    : vector<2x2xf32>, vector<2x2xf32> into f32

  vector.print %c1 : f32
  vector.print %c2 : f32
  vector.print %c3 : f32
  vector.print %c4 : f32
  vector.print %c5 : f32
  vector.print %c6 : f32
  vector.print %c7 : f32
  vector.print %c8 : f32
  //
  // 2D contractions:
  //
  // CHECK: 70
  // CHECK: 71
  // CHECK: 70
  // CHECK: 71
  // CHECK: 69
  // CHECK: 70
  // CHECK: 69
  // CHECK: 70

  return
}
