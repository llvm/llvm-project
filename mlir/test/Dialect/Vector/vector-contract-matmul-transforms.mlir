// RUN: mlir-opt %s -test-vector-contraction-prepare-for-mmt-lowering | FileCheck %s

// CHECK-LABEL: func.func @not_matmul
// CHECK-SAME:    ([[ARG0:%.+]]: vector<4xf32>, [[ARG1:%.+]]: vector<4xf32>, [[ARG2:%.+]]: f32)
// CHECK-NEXT:    vector.contract
// CHECK-NEXT:    return
func.func @not_matmul(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: f32) -> f32 {
  %0 = vector.contract {indexing_maps = [affine_map<(d0) -> (d0)>,
                                         affine_map<(d0) -> (d0)>,
                                         affine_map<(d0) -> ()>],
                        iterator_types = ["reduction"],
                        kind = #vector.kind<add>} %arg0, %arg1, %arg2 :
         vector<4xf32>, vector<4xf32> into f32
  return %0 : f32
}

// This contraction is already in the canonical form.
// CHECK-LABEL: func.func @matmul_mk_nk_mn_4x4xi32
// CHECK-SAME:    ([[ARG0:%.+]]: vector<4x4xi32>, [[ARG1:%.+]]: vector<4x4xi32>, [[ARG2:%.+]]: vector<4x4xi32>)
// CHECK-NEXT:    [[RES:%.+]]   = vector.contract {{.+}} [[ARG0]], [[ARG1]], [[ARG2]]
// CHECK-NEXT:    return [[RES]]
func.func @matmul_mk_nk_mn_4x4xi32(%arg0: vector<4x4xi32>, %arg1: vector<4x4xi32>, %arg2: vector<4x4xi32>) -> vector<4x4xi32> {
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                           affine_map<(d0, d1, d2) -> (d1, d2)>,
                                           affine_map<(d0, d1, d2) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<4x4xi32>, vector<4x4xi32> into vector<4x4xi32>
  return %res : vector<4x4xi32>
}

// CHECK-LABEL: func.func @matmul_mk_kn_mn_4x4xi32
// CHECK-SAME:    ([[ARG0:%.+]]: vector<4x4xi32>, [[ARG1:%.+]]: vector<4x4xi32>, [[ARG2:%.+]]: vector<4x4xi32>)
// CHECK-NEXT:    [[TRANS:%.+]] = vector.transpose [[ARG1]], [1, 0] : vector<4x4xi32> to vector<4x4xi32>
// CHECK-NEXT:    [[RES:%.+]]   = vector.contract {{.+}} [[ARG0]], [[TRANS]], [[ARG2]]
// CHECK-NEXT:    return [[RES]]
func.func @matmul_mk_kn_mn_4x4xi32(%arg0: vector<4x4xi32>, %arg1: vector<4x4xi32>, %arg2: vector<4x4xi32>) -> vector<4x4xi32> {
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                                           affine_map<(d0, d1, d2) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<4x4xi32>, vector<4x4xi32> into vector<4x4xi32>
  return %res : vector<4x4xi32>
}

// CHECK-LABEL: func.func @matmul_mk_kn_mn_4x4xi8_extsi_i32
// CHECK-SAME:    ([[ARG0:%.+]]: vector<4x4xi8>, [[ARG1:%.+]]: vector<4x4xi8>, [[ARG2:%.+]]: vector<4x4xi32>)
// CHECK-NEXT:    [[LHS:%.+]]   = arith.extsi [[ARG0]] : vector<4x4xi8> to vector<4x4xi32>
// CHECK-NEXT:    [[TRANS:%.+]] = vector.transpose [[ARG1]], [1, 0] : vector<4x4xi8> to vector<4x4xi8>
// CHECK-NEXT:    [[RHS:%.+]]   = arith.extsi [[TRANS]] : vector<4x4xi8> to vector<4x4xi32>
// CHECK-NEXT:    [[RES:%.+]]   = vector.contract {{.+}} [[LHS]], [[RHS]], [[ARG2]]
// CHECK-NEXT:    return [[RES]]
func.func @matmul_mk_kn_mn_4x4xi8_extsi_i32(%arg0: vector<4x4xi8>, %arg1: vector<4x4xi8>, %arg2: vector<4x4xi32>) -> vector<4x4xi32> {
  %lhs = arith.extsi %arg0: vector<4x4xi8> to vector<4x4xi32>
  %rhs = arith.extsi %arg1: vector<4x4xi8> to vector<4x4xi32>
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                                           affine_map<(d0, d1, d2) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>} %lhs, %rhs, %arg2 : vector<4x4xi32>, vector<4x4xi32> into vector<4x4xi32>
  return %res : vector<4x4xi32>
}

// Check that non-square shapes are also handled.
// CHECK-LABEL: func.func @matmul_mk_kn_mn_4x16xi32
// CHECK-SAME:    ([[ARG0:%.+]]: vector<4x16xi32>, [[ARG1:%.+]]: vector<16x4xi32>, [[ARG2:%.+]]: vector<4x4xi32>)
// CHECK-NEXT:    [[TRANS:%.+]] = vector.transpose [[ARG1]], [1, 0] : vector<16x4xi32> to vector<4x16xi32>
// CHECK-NEXT:    [[RES:%.+]]   = vector.contract {{.+}} [[ARG0]], [[TRANS]], [[ARG2]]
// CHECK-NEXT:    return [[RES]]
func.func @matmul_mk_kn_mn_4x16xi32(%arg0: vector<4x16xi32>, %arg1: vector<16x4xi32>, %arg2: vector<4x4xi32>) -> vector<4x4xi32> {
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                                           affine_map<(d0, d1, d2) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<4x16xi32>, vector<16x4xi32> into vector<4x4xi32>
  return %res : vector<4x4xi32>
}

// CHECK-LABEL: func.func @matmul_mk_kn_mn_4x4xi8_extui_i32
// CHECK-SAME:    ([[ARG0:%.+]]: vector<4x4xi8>, [[ARG1:%.+]]: vector<4x4xi8>, [[ARG2:%.+]]: vector<4x4xi32>)
// CHECK-NEXT:    [[LHS:%.+]]   = arith.extui [[ARG0]] : vector<4x4xi8> to vector<4x4xi32>
// CHECK-NEXT:    [[TRANS:%.+]] = vector.transpose [[ARG1]], [1, 0] : vector<4x4xi8> to vector<4x4xi8>
// CHECK-NEXT:    [[RHS:%.+]]   = arith.extui [[TRANS]] : vector<4x4xi8> to vector<4x4xi32>
// CHECK-NEXT:    [[RES:%.+]]   = vector.contract {{.+}} [[LHS]], [[RHS]], [[ARG2]]
// CHECK-NEXT:    return [[RES]]
func.func @matmul_mk_kn_mn_4x4xi8_extui_i32(%arg0: vector<4x4xi8>, %arg1: vector<4x4xi8>, %arg2: vector<4x4xi32>) -> vector<4x4xi32> {
  %lhs = arith.extui %arg0: vector<4x4xi8> to vector<4x4xi32>
  %rhs = arith.extui %arg1: vector<4x4xi8> to vector<4x4xi32>
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                                           affine_map<(d0, d1, d2) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>} %lhs, %rhs, %arg2 : vector<4x4xi32>, vector<4x4xi32> into vector<4x4xi32>
  return %res : vector<4x4xi32>
}

// CHECK-LABEL: func.func @matmul_km_nk_mn_4x4xi32
// CHECK-SAME:    ([[ARG0:%.+]]: vector<4x4xi32>, [[ARG1:%.+]]: vector<4x4xi32>, [[ARG2:%.+]]: vector<4x4xi32>)
// CHECK-NEXT:    [[TRANS:%.+]] = vector.transpose [[ARG0]], [1, 0] : vector<4x4xi32> to vector<4x4xi32>
// CHECK-NEXT:    [[RES:%.+]]   = vector.contract {{.+}} [[TRANS]], [[ARG1]], [[ARG2]]
// CHECK-NEXT:    return [[RES]]
func.func @matmul_km_nk_mn_4x4xi32(%arg0: vector<4x4xi32>, %arg1: vector<4x4xi32>, %arg2: vector<4x4xi32>) -> vector<4x4xi32> {
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>,
                                           affine_map<(d0, d1, d2) -> (d1, d2)>,
                                           affine_map<(d0, d1, d2) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<4x4xi32>, vector<4x4xi32> into vector<4x4xi32>
  return %res : vector<4x4xi32>
}

// CHECK-LABEL: func.func @matmul_km_kn_mn_4x4xi32
// CHECK-SAME:    ([[ARG0:%.+]]: vector<4x4xi32>, [[ARG1:%.+]]: vector<4x4xi32>, [[ARG2:%.+]]: vector<4x4xi32>)
// CHECK-DAG:     [[LHS:%.+]] = vector.transpose [[ARG0]], [1, 0] : vector<4x4xi32> to vector<4x4xi32>
// CHECK-DAG:     [[RHS:%.+]] = vector.transpose [[ARG1]], [1, 0] : vector<4x4xi32> to vector<4x4xi32>
// CHECK-NEXT:    [[RES:%.+]] = vector.contract {{.+}} [[LHS]], [[RHS]], [[ARG2]]
// CHECK-NEXT:    return [[RES]]
func.func @matmul_km_kn_mn_4x4xi32(%arg0: vector<4x4xi32>, %arg1: vector<4x4xi32>, %arg2: vector<4x4xi32>) -> vector<4x4xi32> {
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>,
                                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                                           affine_map<(d0, d1, d2) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<4x4xi32>, vector<4x4xi32> into vector<4x4xi32>
  return %res : vector<4x4xi32>
}

// CHECK-LABEL: func.func @matmul_km_kn_mn_4x4xi8_mixed_ext_i32
// CHECK-SAME:    ([[ARG0:%.+]]: vector<4x4xi8>, [[ARG1:%.+]]: vector<4x4xi8>, [[ARG2:%.+]]: vector<4x4xi32>)
// CHECK-DAG:     [[LHST:%.+]] = vector.transpose [[ARG0]], [1, 0] : vector<4x4xi8> to vector<4x4xi8>
// CHECK-DAG:     [[LHS:%.+]]  = arith.extsi [[LHST]] : vector<4x4xi8> to vector<4x4xi32>
// CHECK-DAG:     [[RHST:%.+]] = vector.transpose [[ARG1]], [1, 0] : vector<4x4xi8> to vector<4x4xi8>
// CHECK-DAG:     [[RHS:%.+]]  = arith.extui [[RHST]] : vector<4x4xi8> to vector<4x4xi32>
// CHECK-NEXT:    [[RES:%.+]]  = vector.contract {{.+}} [[LHS]], [[RHS]], [[ARG2]]
// CHECK-NEXT:    return [[RES]]
func.func @matmul_km_kn_mn_4x4xi8_mixed_ext_i32(%arg0: vector<4x4xi8>, %arg1: vector<4x4xi8>, %arg2: vector<4x4xi32>) -> vector<4x4xi32> {
  %lhs = arith.extsi %arg0 : vector<4x4xi8> to vector<4x4xi32>
  %rhs = arith.extui %arg1 : vector<4x4xi8> to vector<4x4xi32>
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>,
                                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                                           affine_map<(d0, d1, d2) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>} %lhs, %rhs, %arg2 : vector<4x4xi32>, vector<4x4xi32> into vector<4x4xi32>
  return %res : vector<4x4xi32>
}

// CHECK-LABEL: func.func @matmul_mk_nk_nm_4x4xi32
// CHECK-SAME:    ([[ARG0:%.+]]: vector<4x4xi32>, [[ARG1:%.+]]: vector<4x4xi32>, [[ARG2:%.+]]: vector<4x4xi32>)
// CHECK-NEXT:    [[RES:%.+]]   = vector.contract {{.+}} [[ARG1]], [[ARG0]], [[ARG2]]
// CHECK-NEXT:    return [[RES]]
func.func @matmul_mk_nk_nm_4x4xi32(%arg0: vector<4x4xi32>, %arg1: vector<4x4xi32>, %arg2: vector<4x4xi32>) -> vector<4x4xi32> {
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                           affine_map<(d0, d1, d2) -> (d1, d2)>,
                                           affine_map<(d0, d1, d2) -> (d1, d0)>],
                          iterator_types = ["parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<4x4xi32>, vector<4x4xi32> into vector<4x4xi32>
  return %res : vector<4x4xi32>
}

// CHECK-LABEL: func.func @matmul_km_kn_nm_4x4xi32
// CHECK-SAME:    ([[ARG0:%.+]]: vector<4x4xi32>, [[ARG1:%.+]]: vector<4x4xi32>, [[ARG2:%.+]]: vector<4x4xi32>)
// CHECK-DAG:     [[LHS:%.+]] = vector.transpose [[ARG0]], [1, 0] : vector<4x4xi32> to vector<4x4xi32>
// CHECK-DAG:     [[RHS:%.+]] = vector.transpose [[ARG1]], [1, 0] : vector<4x4xi32> to vector<4x4xi32>
// CHECK-NEXT:    [[RES:%.+]] = vector.contract {{.+}} [[RHS]], [[LHS]], [[ARG2]]
// CHECK-NEXT:    return [[RES]]
func.func @matmul_km_kn_nm_4x4xi32(%arg0: vector<4x4xi32>, %arg1: vector<4x4xi32>, %arg2: vector<4x4xi32>) -> vector<4x4xi32> {
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>,
                                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                                           affine_map<(d0, d1, d2) -> (d1, d0)>],
                          iterator_types = ["parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<4x4xi32>, vector<4x4xi32> into vector<4x4xi32>
  return %res : vector<4x4xi32>
}

// CHECK-LABEL: func.func @matmul_mk_kn_nm_4x4xi32
// CHECK-SAME:    ([[ARG0:%.+]]: vector<4x4xi32>, [[ARG1:%.+]]: vector<4x4xi32>, [[ARG2:%.+]]: vector<4x4xi32>)
// CHECK-DAG:     [[RHS:%.+]] = vector.transpose [[ARG1]], [1, 0] : vector<4x4xi32> to vector<4x4xi32>
// CHECK-NEXT:    [[RES:%.+]] = vector.contract {{.+}} [[RHS]], [[ARG0]], [[ARG2]]
// CHECK-NEXT:    return [[RES]]
func.func @matmul_mk_kn_nm_4x4xi32(%arg0: vector<4x4xi32>, %arg1: vector<4x4xi32>, %arg2: vector<4x4xi32>) -> vector<4x4xi32> {
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                           affine_map<(d0, d1, d2) -> (d2, d1)>,
                                           affine_map<(d0, d1, d2) -> (d1, d0)>],
                          iterator_types = ["parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<4x4xi32>, vector<4x4xi32> into vector<4x4xi32>
  return %res : vector<4x4xi32>
}

// CHECK-LABEL: func.func @matmul_km_nk_nm_4x4xi32
// CHECK-SAME:    ([[ARG0:%.+]]: vector<4x4xi32>, [[ARG1:%.+]]: vector<4x4xi32>, [[ARG2:%.+]]: vector<4x4xi32>)
// CHECK-DAG:     [[LHS:%.+]] = vector.transpose [[ARG0]], [1, 0] : vector<4x4xi32> to vector<4x4xi32>
// CHECK-NEXT:    [[RES:%.+]] = vector.contract {{.+}} [[ARG1]], [[LHS]], [[ARG2]]
// CHECK-NEXT:    return [[RES]]
func.func @matmul_km_nk_nm_4x4xi32(%arg0: vector<4x4xi32>, %arg1: vector<4x4xi32>, %arg2: vector<4x4xi32>) -> vector<4x4xi32> {
  %res = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0)>,
                                           affine_map<(d0, d1, d2) -> (d1, d2)>,
                                           affine_map<(d0, d1, d2) -> (d1, d0)>],
                          iterator_types = ["parallel", "parallel", "reduction"],
                          kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<4x4xi32>, vector<4x4xi32> into vector<4x4xi32>
  return %res : vector<4x4xi32>
}
