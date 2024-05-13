// RUN: mlir-opt %s -transform-interpreter --split-input-file | FileCheck %s

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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %module_op
      : (!transform.any_op) -> !transform.op<"linalg.matmul">
    transform.structured.pack_greedily %matmul
        matmul_packed_sizes = [8, 16, 32] matmul_inner_dims_order = [1, 2, 0]
      : (!transform.op<"linalg.matmul">) -> !transform.op<"linalg.generic">
      transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %module_op : (!transform.any_op) -> !transform.op<"linalg.generic">
    transform.structured.pack_greedily %generic
        matmul_packed_sizes = [8, 16, 32] matmul_inner_dims_order = [1, 2, 0]
      : (!transform.op<"linalg.generic">) -> !transform.op<"linalg.generic">
      transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %module_op : (!transform.any_op) -> !transform.op<"linalg.generic">
    transform.structured.pack_greedily %generic
        matmul_packed_sizes = [8, 16, 32] matmul_inner_dims_order = [1, 2, 0]
      : (!transform.op<"linalg.generic">) -> !transform.op<"linalg.generic">
      transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %module_op : (!transform.any_op) -> !transform.op<"linalg.generic">
    transform.structured.pack_greedily %generic
        matmul_packed_sizes = [8, 16, 32] matmul_inner_dims_order = [1, 2, 0]
      : (!transform.op<"linalg.generic">) -> !transform.op<"linalg.generic">
      transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %conv = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %module_op
      : (!transform.any_op) -> !transform.op<"linalg.conv_2d_nchw_fchw">
    transform.structured.pack_greedily %conv
        matmul_packed_sizes = [8, 16, 32] matmul_inner_dims_order = [1, 2, 0]
      : (!transform.op<"linalg.conv_2d_nchw_fchw">) -> !transform.op<"linalg.generic">
      transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %module_op : (!transform.any_op) -> !transform.op<"linalg.generic">
    transform.structured.pack_greedily %generic
        matmul_packed_sizes = [8, 16, 32] matmul_inner_dims_order = [1, 2, 0]
      : (!transform.op<"linalg.generic">) -> !transform.op<"linalg.generic">
      transform.yield
  }
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
  // CHECK-SAME:   ins(%{{.*}} : tensor<128x8x32x8xf32>, tensor<1x8x32x130xf32>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<1x128x8x130xf32>)
  %0 = linalg.generic #mkn_trait ins(%A, %B : !A_mk, !B_nk) outs(%C : !C_nm) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %d = arith.mulf %a, %b : f32
      %e = arith.addf %c, %d : f32
      linalg.yield %e : f32
  } -> !C_nm
  return %0 : !C_nm
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %module_op : (!transform.any_op) -> !transform.op<"linalg.generic">
    transform.structured.pack_greedily %generic
        // In this spec, the "k" dimension is not packed but rather padded to the
        // next multiple of 10 (i.e. 130).
        matmul_packed_sizes = [8, 0, 32]
        matmul_padded_sizes_next_multiple_of = [0, 10, 0]
        matmul_inner_dims_order = [1, 2, 0]
      : (!transform.op<"linalg.generic">) -> !transform.op<"linalg.generic">
      transform.yield
  }
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

// Normalized dims are:                     ( k,  m,  n)(kk, mm)
// CHECK-DAG: #[[$km_kkmm:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d0, d3)>
// CHECK-DAG: #[[$kn_kknn:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d3, d4)>
// CHECK-DAG: #[[$mn_mmnn:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d1, d4)>

// CHECK-LABEL: @matmul_mk_nk_nm(
func.func @matmul_mk_nk_nm(%A : !A_mk, %B : !B_nk, %C : !C_nm) -> !C_nm {
  //      CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$mk_kkmm]], #[[$kn_kknn]], #[[$mn_mmnn]]]
  // CHECK-SAME:   ["reduction", "parallel", "parallel", "reduction", "parallel"]}
  // CHECK-SAME:   ins(%{{.*}} : tensor<1023x8x32xf32>, tensor<1x8x32x130xf32>)
  // CHECK-SAME:  outs(%{{.*}} : tensor<1x1023x130xf32>)
  %0 = linalg.generic #mkn_trait ins(%A, %B : !A_mk, !B_nk) outs(%C : !C_nm) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %d = arith.mulf %a, %b : f32
      %e = arith.addf %c, %d : f32
      linalg.yield %e : f32
  } -> !C_nm
  return %0 : !C_nm
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %module_op : (!transform.any_op) -> !transform.op<"linalg.generic">
    transform.structured.pack_greedily %generic
        // In this spec, the "n" dimension is neither packed not unpacked.
        // We don't end up with an innermost matmul after packing but only with an
        // innermost matvec.
        matmul_packed_sizes = [0, 0, 32]
        matmul_padded_sizes_next_multiple_of = [0, 10, 0]
        matmul_inner_dims_order = [1, 2, 0]
      : (!transform.op<"linalg.generic">) -> !transform.op<"linalg.generic">
      transform.yield
  }
}

// -----

!A = tensor<1023x255xf32>
!X = tensor<255xf32>
!Y = tensor<1023xf32>

// CHECK-LABEL: @matvec_fail(
func.func @matvec_fail(%A : !A, %x : !X, %y : !Y) -> !Y {
  //      CHECK: linalg.matvec
  %0 = linalg.matvec ins(%A, %x : !A, !X) outs(%y : !Y) -> !Y
  return %0 : !Y
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matvec"]} in %module_op
      : (!transform.any_op) -> !transform.op<"linalg.matvec">
    transform.structured.pack_greedily %matmul
        matmul_packed_sizes = [8, 16, 32] matmul_inner_dims_order = [1, 2, 0]
      : (!transform.op<"linalg.matvec">) -> !transform.any_op
      transform.yield
  }
}

// -----

func.func @no_padding_on_packs(%A: tensor<32x32xf32>, %B: tensor<32x32xf32>, %C: tensor<32x32xf32>)
    -> tensor<32x32xf32> {
  %0 = linalg.matmul  ins(%A, %B: tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%C: tensor<32x32xf32>)
    -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: no_padding_on_packs
// CHECK: tensor.pack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [8, 4]
// CHECK-SAME:  into %{{.+}} : tensor<32x32xf32> -> tensor<4x8x8x4xf32>
// CHECK: tensor.pack %{{.+}} outer_dims_perm = [1, 0]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [4, 16] into %{{.+}} : tensor<32x32xf32> -> tensor<2x8x4x16xf32>
// CHECK: tensor.pack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [8, 16]
// CHECK-SAME:  into %{{.+}} : tensor<32x32xf32> -> tensor<4x2x8x16xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
        : (!transform.any_op) -> !transform.op<"linalg.matmul">
      %1 = transform.structured.pack_greedily %0
          matmul_packed_sizes = [8, 16, 4] matmul_inner_dims_order = [0, 1, 2]
        : (!transform.op<"linalg.matmul">) -> !transform.op<"linalg.generic">
      %pack = transform.get_producer_of_operand %1[1]
      : (!transform.op<"linalg.generic">) -> (!transform.op<"tensor.pack">)
      %2, %pack_2, %empty_unpack_2 =
      transform.structured.pack_transpose %pack with_compute_op(%1)
      outer_perm = [1, 0] inner_perm = [1, 0]
       : (!transform.op<"tensor.pack">, !transform.op<"linalg.generic">)
      -> (!transform.op<"linalg.generic">, !transform.op<"tensor.pack">, !transform.any_op)
      transform.yield
  }
}
