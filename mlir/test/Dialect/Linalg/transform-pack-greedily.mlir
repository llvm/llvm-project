// RUN: mlir-opt %s -test-transform-dialect-interpreter --split-input-file | FileCheck %s

!A_mk = tensor<1023x255xf32>
!B_kn = tensor<255x127xf32>
!C_mn = tensor<1023x127xf32>

// Normalized dims are:                     ( k,  m,  n)(kk, mm, nn)
// CHECK-DAG: #[[$mk_kkmm:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
// CHECK-DAG: #[[$kn_kknn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[$mn_mmnn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>

// CHECK-LABEL: @matmul_mk_kn_mn(
func.func @matmul_mk_kn_mn(%A : !A_mk, %B : !B_kn, %C : !C_mn) -> !C_mn {
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$mk_kkmm]], #[[$kn_kknn]], #[[$mn_mmnn]]]
  // CHECK-SAME:   ["reduction", "parallel", "parallel", "reduction", "parallel", "parallel"]} 
  // CHECK-SAME:   ins(%{{.*}} : tensor<128x8x32x8xf32>, tensor<8x8x32x16xf32>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<128x8x8x16xf32>)
  %0 = linalg.matmul ins(%A, %B : !A_mk, !B_kn) outs(%C : !C_mn) -> !C_mn
  return %0 : !C_mn
}

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %module_op 
    : (!pdl.operation) -> !transform.op<"linalg.matmul">
  transform.structured.pack_greedily %matmul 
      gemm_packed_sizes = [8, 16, 32] gemm_inner_dims_order = [1, 2, 0]
    : (!transform.op<"linalg.matmul">) -> !transform.op<"linalg.generic">
}

// -----

!A_mk = tensor<1023x255xf32>
!B_nk = tensor<127x255xf32>
!C_nm = tensor<127x1023xf32>

#mkn_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (n, m)>
]
#mkn_trait = {
  indexing_maps = #mkn_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// Normalized dims are:                     ( k,  m,  n)(kk, mm, nn)
// CHECK-DAG: #[[$km_kkmm:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
// CHECK-DAG: #[[$kn_kknn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
// CHECK-DAG: #[[$mn_mmnn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>

// CHECK-LABEL: @matmul_mk_nk_nm(
func.func @matmul_mk_nk_nm(%A : !A_mk, %B : !B_nk, %C : !C_nm) -> !C_nm {
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$mk_kkmm]], #[[$kn_kknn]], #[[$mn_mmnn]]]
  // CHECK-SAME:   ["reduction", "parallel", "parallel", "reduction", "parallel", "parallel"]} 
  // CHECK-SAME:   ins(%{{.*}} : tensor<128x8x32x8xf32>, tensor<8x8x32x16xf32>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<8x128x8x16xf32>)
  %0 = linalg.generic #mkn_trait ins(%A, %B : !A_mk, !B_nk) outs(%C : !C_nm) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %d = arith.mulf %a, %b : f32
      %e = arith.addf %c, %d : f32
      linalg.yield %e : f32
  } -> !C_nm
  return %0 : !C_nm
}

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %generic = transform.structured.match ops{["linalg.generic"]} in %module_op : (!pdl.operation) -> !transform.op<"linalg.generic">
  transform.structured.pack_greedily %generic
      gemm_packed_sizes = [8, 16, 32] gemm_inner_dims_order = [1, 2, 0]
    : (!transform.op<"linalg.generic">) -> !transform.op<"linalg.generic">
}

// -----

!A_mk = tensor<1023x255xf32>
!B_nk = tensor<127x255xf32>
!C_nm = tensor<127x1023xf32>

#mkn_accesses = [
  affine_map<(k, m, n) -> (m, k)>,
  affine_map<(k, m, n) -> (n, k)>,
  affine_map<(k, m, n) -> (n, m)>
]
#mkn_trait = {
  indexing_maps = #mkn_accesses,
  iterator_types = ["reduction", "parallel", "parallel"]
}

// Normalized dims are:                     ( k,  m,  n)(kk, mm, nn)
// CHECK-DAG: #[[$mk_kkmm:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
// CHECK-DAG: #[[$kn_kknn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
// CHECK-DAG: #[[$mn_mmnn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>

// CHECK-LABEL: @matmul_mk_nk_nm_transposed(
func.func @matmul_mk_nk_nm_transposed(%A : !A_mk, %B : !B_nk, %C : !C_nm) -> !C_nm {
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$mk_kkmm]], #[[$kn_kknn]], #[[$mn_mmnn]]]
  // CHECK-SAME:   ["reduction", "parallel", "parallel", "reduction", "parallel", "parallel"]} 
  // CHECK-SAME:   ins(%{{.*}} : tensor<128x8x32x8xf32>, tensor<8x8x32x16xf32>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<8x128x8x16xf32>)
  %0 = linalg.generic #mkn_trait ins(%A, %B : !A_mk, !B_nk) outs(%C : !C_nm) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %d = arith.mulf %a, %b : f32
      %e = arith.addf %c, %d : f32
      linalg.yield %e : f32
  } -> !C_nm
  return %0 : !C_nm
}

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %generic = transform.structured.match ops{["linalg.generic"]} in %module_op : (!pdl.operation) -> !transform.op<"linalg.generic">
  transform.structured.pack_greedily %generic
      gemm_packed_sizes = [8, 16, 32] gemm_inner_dims_order = [1, 2, 0]
    : (!transform.op<"linalg.generic">) -> !transform.op<"linalg.generic">
}

// -----

!A_bmkm2 = tensor<42x1023x255x33xf32>
!B_nkb = tensor<127x255x42xf32>
!C_nbm = tensor<127x42x1023xf32>

#mkn_accesses = [
  affine_map<(k, m, n, b, m2) -> (b, m, k, m2)>,
  affine_map<(k, m, n, b, m2) -> (n, k, b)>,
  affine_map<(k, m, n, b, m2) -> (n, b, m)>
]
#mkn_trait = {
  indexing_maps = #mkn_accesses,
  iterator_types = ["reduction", "parallel", "parallel", "parallel", "parallel"]
}

// Normalized dims are:                        ( ?,  ?,  k,  m,  n)(kk, mm, nn)
// CHECK-DAG: #[[$bmkm2_kkmm:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d3, d2, d1, d5, d6)>
// CHECK-DAG:   #[[$nkb_kknn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d2, d0, d5, d7)>
// CHECK-DAG:   #[[$nbm_mmnn:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d0, d3, d6, d7)>

// CHECK-LABEL: @contraction_bmkm2_nkb_nbm(
func.func @contraction_bmkm2_nkb_nbm(%A : !A_bmkm2, %B : !B_nkb, %C : !C_nbm) -> !C_nbm {
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$bmkm2_kkmm]], #[[$nkb_kknn]], #[[$nbm_mmnn]]]
  // CHECK-SAME:   ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel"]} 
  // CHECK-SAME:   ins(%{{.*}} : tensor<42x128x8x33x32x8xf32>, tensor<8x8x42x32x16xf32>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<8x42x128x8x16xf32>)
  %0 = linalg.generic #mkn_trait ins(%A, %B : !A_bmkm2, !B_nkb) outs(%C : !C_nbm) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %d = arith.mulf %a, %b : f32
      %e = arith.addf %c, %d : f32
      linalg.yield %e : f32
  } -> !C_nbm
  return %0 : !C_nbm
}

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %generic = transform.structured.match ops{["linalg.generic"]} in %module_op : (!pdl.operation) -> !transform.op<"linalg.generic">
  transform.structured.pack_greedily %generic
      gemm_packed_sizes = [8, 16, 32] gemm_inner_dims_order = [1, 2, 0]
    : (!transform.op<"linalg.generic">) -> !transform.op<"linalg.generic">
}

// -----

// Conv linguo:                          h   w  kh  kw   c   n   f  cc  nn  ff
// Normalized dims are:                ( ?,  ?,  ?,  ?,  k,  m,  n)(kk, mm, nn)
//                                                                                   n   c   h + kh   w + kw  cc  nn
// CHECK-DAG: #[[$M1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d5, d4, d0 + d2, d1 + d3, d7, d8)>
//                                                                                   f   c  kh  kw  cc  ff
// CHECK-DAG: #[[$M2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d6, d4, d2, d3, d7, d9)>
//                                                                                   n   f   h   w  nn  ff
// CHECK-DAG: #[[$M3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d5, d6, d0, d1, d8, d9)>

// CHECK-LABEL: @conv_2d_nchw_fchw
func.func @conv_2d_nchw_fchw(%arg0: tensor<?x47x16x16xf32>, %arg2: tensor<?x16x14x14xf32>) -> tensor<?x16x14x14xf32> {
  %c0 = arith.constant dense<0.1> : tensor<16x47x3x3xf32>
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$M1]], #[[$M2]], #[[$M3]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel"]
  // CHECK-SAME:  ins(%{{.*}} : tensor<?x2x16x16x32x8xf32>, tensor<1x2x3x3x32x16xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<?x1x14x14x8x16xf32>)
  %0 = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %c0: tensor<?x47x16x16xf32>, tensor<16x47x3x3xf32>)
    outs(%arg2: tensor<?x16x14x14xf32>) -> tensor<?x16x14x14xf32>
  return %0 : tensor<?x16x14x14xf32>
}

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %conv = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %module_op 
    : (!pdl.operation) -> !transform.op<"linalg.conv_2d_nchw_fchw">
  transform.structured.pack_greedily %conv
      gemm_packed_sizes = [8, 16, 32] gemm_inner_dims_order = [1, 2, 0]
    : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> !transform.op<"linalg.generic">
}


// -----

// These should fail to pack for now as they don't contain a contraction.
// CHECK-LABEL: @reduce_and_map
func.func @reduce_and_map(%arg0: tensor<10x100xf32>,
    %arg1: tensor<10x100xf32>, %output: tensor<10xf32>) -> tensor<10xf32> {
  %map_init = tensor.empty() : tensor<10x100xf32>
  // CHECK: linalg.map
  %mapped = linalg.map { arith.addf }
              ins(%arg0, %arg1 : tensor<10x100xf32>, tensor<10x100xf32>)
              outs(%map_init : tensor<10x100xf32>)
  // CHECK: linalg.reduce
  %res = linalg.reduce { arith.addf }
           ins(%mapped: tensor<10x100xf32>)
           outs(%output: tensor<10xf32>)
           dimensions = [1]
  return %res : tensor<10xf32>
}

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %generic = transform.structured.match ops{["linalg.generic"]} in %module_op : (!pdl.operation) -> !transform.op<"linalg.generic">
  transform.structured.pack_greedily %generic
      gemm_packed_sizes = [8, 16, 32] gemm_inner_dims_order = [1, 2, 0]
    : (!transform.op<"linalg.generic">) -> !transform.op<"linalg.generic">
}
