// RUN: mlir-opt %s -arm-sme-outer-product-fusion -cse -split-input-file | FileCheck %s

// CHECK-LABEL: @outerproduct_add_widening_2way_f16f16f32
// CHECK-SAME:    %[[A0:.*]]: vector<[4]xf16>, %[[B0:.*]]: vector<[4]xf16>, %[[A1:.*]]: vector<[4]xf16>, %[[B1:.*]]: vector<[4]xf16>,
// CHECK-SAME:    %[[A0_MASK:.*]]: vector<[4]xi1>, %[[B0_MASK:.*]]: vector<[4]xi1>, %[[A1_MASK:.*]]: vector<[4]xi1>, %[[B1_MASK:.*]]: vector<[4]xi1>
// CHECK-DAG: %[[ACC:.*]] = arith.constant dense<0.000000e+00> : vector<[4]x[4]xf32>
// CHECK-DAG: %[[LHS:.*]] = vector.interleave %[[A0]], %[[A1]] : vector<[4]xf16> -> vector<[8]xf16>
// CHECK-DAG: %[[RHS:.*]] = vector.interleave %[[B0]], %[[B1]] : vector<[4]xf16> -> vector<[8]xf16>
// CHECK-DAG: %[[LHS_MASK:.*]] = vector.interleave %[[A0_MASK]], %[[A1_MASK]] : vector<[4]xi1> -> vector<[8]xi1>
// CHECK-DAG: %[[RHS_MASK:.*]] = vector.interleave %[[B0_MASK]], %[[B1_MASK]] : vector<[4]xi1> -> vector<[8]xi1>
// CHECK-DAG: arm_sme.fmopa_2way %[[LHS]], %[[RHS]] acc(%[[ACC]]) masks(%[[LHS_MASK]], %[[RHS_MASK]]) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
func.func @outerproduct_add_widening_2way_f16f16f32(
    %a0 : vector<[4]xf16>, %b0 : vector<[4]xf16>,
    %a1 : vector<[4]xf16>, %b1 : vector<[4]xf16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xf32> {
  %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
  %a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
  %b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

  %acc = arith.constant dense<0.0> : vector<[4]x[4]xf32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xf32>, vector<[4]xf32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xf32>, vector<[4]xf32>

  return %1 : vector<[4]x[4]xf32>
}

// -----

/// Verify chain of 4 outer products are fused into 2 2-way widening outer
/// products.

// CHECK-LABEL: @outerproduct_x2_add_widening_2way_f16f16f32
// CHECK-COUNT-2: arm_sme.fmopa_2way
func.func @outerproduct_x2_add_widening_2way_f16f16f32(
    %a0 : vector<[4]xf16>, %b0 : vector<[4]xf16>,
    %a1 : vector<[4]xf16>, %b1 : vector<[4]xf16>,
    %a2 : vector<[4]xf16>, %b2 : vector<[4]xf16>,
    %a3 : vector<[4]xf16>, %b3 : vector<[4]xf16>) -> vector<[4]x[4]xf32> {
  %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>

  %a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
  %b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

  %a2_ext = arith.extf %a2 : vector<[4]xf16> to vector<[4]xf32>
  %b2_ext = arith.extf %b2 : vector<[4]xf16> to vector<[4]xf32>

  %a3_ext = arith.extf %a3 : vector<[4]xf16> to vector<[4]xf32>
  %b3_ext = arith.extf %b3 : vector<[4]xf16> to vector<[4]xf32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xf32>, vector<[4]xf32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xf32>, vector<[4]xf32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) : vector<[4]xf32>, vector<[4]xf32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) : vector<[4]xf32>, vector<[4]xf32>

  return %3 : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_2way_f16f16f32
// CHECK: arm_sme.fmops_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
func.func @outerproduct_sub_widening_2way_f16f16f32(
    %a0 : vector<[4]xf16>, %b0 : vector<[4]xf16>,
    %a1 : vector<[4]xf16>, %b1 : vector<[4]xf16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xf32> {
  %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
  %a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
  %b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

  %acc = arith.constant dense<0.0> : vector<[4]x[4]xf32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xf32>, vector<[4]xf32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xf32>, vector<[4]xf32>

  return %1 : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_2way_bf16bf16f32
// CHECK: arm_sme.fmopa_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
func.func @outerproduct_add_widening_2way_bf16bf16f32(
    %a0 : vector<[4]xbf16>, %b0 : vector<[4]xbf16>,
    %a1 : vector<[4]xbf16>, %b1 : vector<[4]xbf16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xf32> {
  %a0_ext = arith.extf %a0 : vector<[4]xbf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xbf16> to vector<[4]xf32>
  %a1_ext = arith.extf %a1 : vector<[4]xbf16> to vector<[4]xf32>
  %b1_ext = arith.extf %b1 : vector<[4]xbf16> to vector<[4]xf32>

  %acc = arith.constant dense<0.0> : vector<[4]x[4]xf32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xf32>, vector<[4]xf32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xf32>, vector<[4]xf32>

  return %1 : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_2way_bf16bf16f32
// CHECK: arm_sme.fmops_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
func.func @outerproduct_sub_widening_2way_bf16bf16f32(
    %a0 : vector<[4]xbf16>, %b0 : vector<[4]xbf16>,
    %a1 : vector<[4]xbf16>, %b1 : vector<[4]xbf16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xf32> {
  %a0_ext = arith.extf %a0 : vector<[4]xbf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xbf16> to vector<[4]xf32>
  %a1_ext = arith.extf %a1 : vector<[4]xbf16> to vector<[4]xf32>
  %b1_ext = arith.extf %b1 : vector<[4]xbf16> to vector<[4]xf32>

  %acc = arith.constant dense<0.0> : vector<[4]x[4]xf32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xf32>, vector<[4]xf32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xf32>, vector<[4]xf32>

  return %1 : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_2way_signed_i16i16i32
// CHECK: arm_sme.smopa_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
func.func @outerproduct_add_widening_2way_signed_i16i16i32(
    %a0 : vector<[4]xi16>, %b0 : vector<[4]xi16>,
    %a1 : vector<[4]xi16>, %b1 : vector<[4]xi16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi16> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi16> to vector<[4]xi32>
  %a1_ext = arith.extsi %a1 : vector<[4]xi16> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi16> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %1 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_2way_signed_i16i16i32
// CHECK: arm_sme.smops_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
func.func @outerproduct_sub_widening_2way_signed_i16i16i32(
    %a0 : vector<[4]xi16>, %b0 : vector<[4]xi16>,
    %a1 : vector<[4]xi16>, %b1 : vector<[4]xi16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi16> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi16> to vector<[4]xi32>
  %a1_ext = arith.extsi %a1 : vector<[4]xi16> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi16> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %1 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_2way_unsigned_i16i16i32
// CHECK: arm_sme.umopa_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
func.func @outerproduct_add_widening_2way_unsigned_i16i16i32(
    %a0 : vector<[4]xi16>, %b0 : vector<[4]xi16>,
    %a1 : vector<[4]xi16>, %b1 : vector<[4]xi16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extui %a0 : vector<[4]xi16> to vector<[4]xi32>
  %b0_ext = arith.extui %b0 : vector<[4]xi16> to vector<[4]xi32>
  %a1_ext = arith.extui %a1 : vector<[4]xi16> to vector<[4]xi32>
  %b1_ext = arith.extui %b1 : vector<[4]xi16> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %1 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_2way_unsigned_i16i16i32
// CHECK: arm_sme.umops_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
func.func @outerproduct_sub_widening_2way_unsigned_i16i16i32(
    %a0 : vector<[4]xi16>, %b0 : vector<[4]xi16>,
    %a1 : vector<[4]xi16>, %b1 : vector<[4]xi16>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extui %a0 : vector<[4]xi16> to vector<[4]xi32>
  %b0_ext = arith.extui %b0 : vector<[4]xi16> to vector<[4]xi32>
  %a1_ext = arith.extui %a1 : vector<[4]xi16> to vector<[4]xi32>
  %b1_ext = arith.extui %b1 : vector<[4]xi16> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %1 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_signed_i8i8i32
// CHECK-SAME:    %[[A0:[a-z0-9]+]]: vector<[4]xi8>, %[[B0:[a-z0-9]+]]: vector<[4]xi8>,
// CHECK-SAME:    %[[A1:[a-z0-9]+]]: vector<[4]xi8>, %[[B1:[a-z0-9]+]]: vector<[4]xi8>,
// CHECK-SAME:    %[[A2:[a-z0-9]+]]: vector<[4]xi8>, %[[B2:[a-z0-9]+]]: vector<[4]xi8>,
// CHECK-SAME:    %[[A3:[a-z0-9]+]]: vector<[4]xi8>, %[[B3:[a-z0-9]+]]: vector<[4]xi8>,
// CHECK-SAME:    %[[A0_MASK:[a-z0-9]+]]: vector<[4]xi1>, %[[B0_MASK:[a-z0-9]+]]: vector<[4]xi1>,
// CHECK-SAME:    %[[A1_MASK:[a-z0-9]+]]: vector<[4]xi1>, %[[B1_MASK:[a-z0-9]+]]: vector<[4]xi1>,
// CHECK-SAME:    %[[A2_MASK:[a-z0-9]+]]: vector<[4]xi1>, %[[B2_MASK:[a-z0-9]+]]: vector<[4]xi1>,
// CHECK-SAME:    %[[A3_MASK:[a-z0-9]+]]: vector<[4]xi1>, %[[B3_MASK:[a-z0-9]+]]: vector<[4]xi1>
// CHECK-DAG: %[[ACC:.*]] = arith.constant dense<0> : vector<[4]x[4]xi32>
// CHECK-DAG: %[[LHS0:.*]] = vector.interleave %[[A0]], %[[A2]] : vector<[4]xi8> -> vector<[8]xi8>
// CHECK-DAG: %[[LHS1:.*]] = vector.interleave %[[A1]], %[[A3]] : vector<[4]xi8> -> vector<[8]xi8>
// CHECK-DAG: %[[RHS0:.*]] = vector.interleave %[[B0]], %[[B2]] : vector<[4]xi8> -> vector<[8]xi8>
// CHECK-DAG: %[[RHS1:.*]] = vector.interleave %[[B1]], %[[B3]] : vector<[4]xi8> -> vector<[8]xi8>
// CHECK-DAG: %[[LHS:.*]] = vector.interleave %[[LHS0]], %[[LHS1]] : vector<[8]xi8> -> vector<[16]xi8>
// CHECK-DAG: %[[RHS:.*]] = vector.interleave %[[RHS0]], %[[RHS1]] : vector<[8]xi8> -> vector<[16]xi8>
// CHECK-DAG: %[[LHS0_MASK:.*]] = vector.interleave %[[A0_MASK]], %[[A2_MASK]] : vector<[4]xi1> -> vector<[8]xi1>
// CHECK-DAG: %[[LHS1_MASK:.*]] = vector.interleave %[[A1_MASK]], %[[A3_MASK]] : vector<[4]xi1> -> vector<[8]xi1>
// CHECK-DAG: %[[RHS0_MASK:.*]] = vector.interleave %[[B0_MASK]], %[[B2_MASK]] : vector<[4]xi1> -> vector<[8]xi1>
// CHECK-DAG: %[[RHS1_MASK:.*]] = vector.interleave %[[B1_MASK]], %[[B3_MASK]] : vector<[4]xi1> -> vector<[8]xi1>
// CHECK-DAG: %[[LHS_MASK:.*]] = vector.interleave %[[LHS0_MASK]], %[[LHS1_MASK]] : vector<[8]xi1> -> vector<[16]xi1>
// CHECK-DAG: %[[RHS_MASK:.*]] = vector.interleave %[[RHS0_MASK]], %[[RHS1_MASK]] : vector<[8]xi1> -> vector<[16]xi1>
// CHECK-DAG: arm_sme.smopa_4way %[[LHS]], %[[RHS]] acc(%[[ACC]]) masks(%[[LHS_MASK]], %[[RHS_MASK]]) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_add_widening_4way_signed_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_signed_i8i8i32
// CHECK: arm_sme.smops_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_sub_widening_4way_signed_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_signed_i16i16i64
// CHECK: arm_sme.smopa_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_add_widening_4way_signed_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_ext = arith.extsi %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_ext = arith.extsi %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_ext = arith.extsi %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_ext = arith.extsi %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_ext = arith.extsi %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_ext = arith.extsi %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_ext = arith.extsi %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_ext = arith.extsi %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_signed_i16i16i64
// CHECK: arm_sme.smops_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_sub_widening_4way_signed_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_ext = arith.extsi %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_ext = arith.extsi %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_ext = arith.extsi %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_ext = arith.extsi %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_ext = arith.extsi %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_ext = arith.extsi %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_ext = arith.extsi %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_ext = arith.extsi %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_unsigned_i8i8i32
// CHECK: arm_sme.umopa_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_add_widening_4way_unsigned_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extui %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extui %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extui %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extui %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extui %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extui %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extui %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extui %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_unsigned_i8i8i32
// CHECK: arm_sme.umops_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_sub_widening_4way_unsigned_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extui %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extui %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extui %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extui %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extui %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extui %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extui %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extui %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_unsigned_i16i16i64
// CHECK: arm_sme.umopa_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_add_widening_4way_unsigned_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_ext = arith.extui %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_ext = arith.extui %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_ext = arith.extui %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_ext = arith.extui %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_ext = arith.extui %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_ext = arith.extui %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_ext = arith.extui %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_ext = arith.extui %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_unsigned_i16i16i64
// CHECK: arm_sme.umops_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_sub_widening_4way_unsigned_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_ext = arith.extui %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_ext = arith.extui %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_ext = arith.extui %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_ext = arith.extui %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_ext = arith.extui %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_ext = arith.extui %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_ext = arith.extui %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_ext = arith.extui %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_signed_by_unsigned_i8i8i32
// CHECK: arm_sme.sumopa_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_add_widening_4way_signed_by_unsigned_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_sext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_zext = arith.extui %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_sext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_zext = arith.extui %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_sext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_zext = arith.extui %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_sext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_zext = arith.extui %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_sext, %b0_zext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_sext, %b1_zext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_sext, %b2_zext acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_sext, %b3_zext acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_signed_by_unsigned_i8i8i32
// CHECK: arm_sme.sumops_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_sub_widening_4way_signed_by_unsigned_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_sext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_zext = arith.extui %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_sext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_zext = arith.extui %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_sext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_zext = arith.extui %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_sext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_zext = arith.extui %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_sext, %b0_zext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_sext, %b1_zext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_sext, %b2_zext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_sext, %b3_zext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_signed_by_unsigned_i16i16i64
// CHECK: arm_sme.sumopa_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_add_widening_4way_signed_by_unsigned_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_sext = arith.extsi %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_zext = arith.extui %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_sext = arith.extsi %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_zext = arith.extui %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_sext = arith.extsi %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_zext = arith.extui %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_sext = arith.extsi %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_zext = arith.extui %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_sext, %b0_zext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_sext, %b1_zext acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_sext, %b2_zext acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_sext, %b3_zext acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_signed_by_unsigned_i16i16i64
// CHECK: arm_sme.sumops_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_sub_widening_4way_signed_by_unsigned_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_sext = arith.extsi %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_zext = arith.extui %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_sext = arith.extsi %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_zext = arith.extui %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_sext = arith.extsi %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_zext = arith.extui %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_sext = arith.extsi %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_zext = arith.extui %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_sext, %b0_zext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_sext, %b1_zext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_sext, %b2_zext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_sext, %b3_zext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_unsigned_by_signed_i8i8i32
// CHECK: arm_sme.usmopa_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_add_widening_4way_unsigned_by_signed_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_zext = arith.extui %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_sext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_zext = arith.extui %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_sext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_zext = arith.extui %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_sext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_zext = arith.extui %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_sext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_zext, %b0_sext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_zext, %b1_sext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_zext, %b2_sext acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_zext, %b3_sext acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_unsigned_by_signed_i8i8i32
// CHECK: arm_sme.usmops_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[16]xi8>, vector<[16]xi8> into vector<[4]x[4]xi32>
func.func @outerproduct_sub_widening_4way_unsigned_by_signed_i8i8i32(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a0_mask : vector<[4]xi1>, %b0_mask : vector<[4]xi1>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>,
    %a3_mask : vector<[4]xi1>, %b3_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_zext = arith.extui %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_sext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_zext = arith.extui %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_sext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_zext = arith.extui %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_sext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_zext = arith.extui %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_sext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

  %acc = arith.constant dense<0> : vector<[4]x[4]xi32>

  %0 = arm_sme.outerproduct %a0_zext, %b0_sext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_zext, %b1_sext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_zext, %b2_sext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_zext, %b3_sext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_add_widening_4way_unsigned_by_signed_i16i16i64
// CHECK: arm_sme.usmopa_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_add_widening_4way_unsigned_by_signed_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_zext = arith.extui %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_sext = arith.extsi %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_zext = arith.extui %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_sext = arith.extsi %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_zext = arith.extui %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_sext = arith.extsi %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_zext = arith.extui %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_sext = arith.extsi %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_zext, %b0_sext acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_zext, %b1_sext acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_zext, %b2_sext acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_zext, %b3_sext acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}

// -----

// CHECK-LABEL: @outerproduct_sub_widening_4way_unsigned_by_signed_i16i16i64
// CHECK: arm_sme.usmops_4way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[2]x[2]xi64>
func.func @outerproduct_sub_widening_4way_unsigned_by_signed_i16i16i64(
    %a0 : vector<[2]xi16>, %b0 : vector<[2]xi16>,
    %a1 : vector<[2]xi16>, %b1 : vector<[2]xi16>,
    %a2 : vector<[2]xi16>, %b2 : vector<[2]xi16>,
    %a3 : vector<[2]xi16>, %b3 : vector<[2]xi16>,
    %a0_mask : vector<[2]xi1>, %b0_mask : vector<[2]xi1>,
    %a1_mask : vector<[2]xi1>, %b1_mask : vector<[2]xi1>,
    %a2_mask : vector<[2]xi1>, %b2_mask : vector<[2]xi1>,
    %a3_mask : vector<[2]xi1>, %b3_mask : vector<[2]xi1>) -> vector<[2]x[2]xi64> {
  %a0_zext = arith.extui %a0 : vector<[2]xi16> to vector<[2]xi64>
  %b0_sext = arith.extsi %b0 : vector<[2]xi16> to vector<[2]xi64>

  %a1_zext = arith.extui %a1 : vector<[2]xi16> to vector<[2]xi64>
  %b1_sext = arith.extsi %b1 : vector<[2]xi16> to vector<[2]xi64>

  %a2_zext = arith.extui %a2 : vector<[2]xi16> to vector<[2]xi64>
  %b2_sext = arith.extsi %b2 : vector<[2]xi16> to vector<[2]xi64>

  %a3_zext = arith.extui %a3 : vector<[2]xi16> to vector<[2]xi64>
  %b3_sext = arith.extsi %b3 : vector<[2]xi16> to vector<[2]xi64>

  %acc = arith.constant dense<0> : vector<[2]x[2]xi64>

  %0 = arm_sme.outerproduct %a0_zext, %b0_sext kind<sub> acc(%acc) masks(%a0_mask, %b0_mask) : vector<[2]xi64>, vector<[2]xi64>
  %1 = arm_sme.outerproduct %a1_zext, %b1_sext kind<sub> acc(%0) masks(%a1_mask, %b1_mask) : vector<[2]xi64>, vector<[2]xi64>
  %2 = arm_sme.outerproduct %a2_zext, %b2_sext kind<sub> acc(%1) masks(%a2_mask, %b2_mask) : vector<[2]xi64>, vector<[2]xi64>
  %3 = arm_sme.outerproduct %a3_zext, %b3_sext kind<sub> acc(%2) masks(%a3_mask, %b3_mask) : vector<[2]xi64>, vector<[2]xi64>

  return %3 : vector<[2]x[2]xi64>
}

/// Tests for related patterns.

// -----

// CHECK-LABEL: @extract_from_arith_ext(
// CHECK-SAME:                          %[[SRC:.*]]: vector<4x[8]xi8>
// CHECK: %[[EXTRACT:.*]] = vector.extract %[[SRC]][0] : vector<[8]xi8> from vector<4x[8]xi8>
// CHECK: %[[EXTEND:.*]] = arith.extsi %[[EXTRACT]] : vector<[8]xi8> to vector<[8]xi32>
// CHECK: return %[[EXTEND]]
func.func @extract_from_arith_ext(%src: vector<4x[8]xi8>) -> vector<[8]xi32> {
  %0 = arith.extsi %src : vector<4x[8]xi8> to vector<4x[8]xi32>
  %1 = vector.extract %0[0] : vector<[8]xi32> from vector<4x[8]xi32>
  return %1 : vector<[8]xi32>
}

// -----

// CHECK-LABEL: @non_constant_extract_from_arith_ext(
// CHECK-SAME:                                       %[[SRC:[a-z0-9]+]]: vector<4x[8]xi8>,
// CHECK-SAME:                                       %[[DIM:[a-z0-9]+]]: index
// CHECK: %[[EXTRACT:.*]] = vector.extract %[[SRC]][%[[DIM]]] : vector<[8]xi8> from vector<4x[8]xi8>
// CHECK: %[[EXTEND:.*]] = arith.extui %[[EXTRACT]] : vector<[8]xi8> to vector<[8]xi32>
// CHECK: return %[[EXTEND]]
func.func @non_constant_extract_from_arith_ext(%src: vector<4x[8]xi8>, %dim: index) -> vector<[8]xi32> {
  %0 = arith.extui %src : vector<4x[8]xi8> to vector<4x[8]xi32>
  %1 = vector.extract %0[%dim] : vector<[8]xi32> from vector<4x[8]xi32>
  return %1 : vector<[8]xi32>
}

// -----

// CHECK-LABEL: @scalable_extract_from_arith_ext(
// CHECK-SAME:                                   %[[SRC:.*]]: vector<[8]xf16>
// CHECK: %[[EXTRACT:.*]] = vector.scalable.extract %[[SRC]][0] : vector<[4]xf16> from vector<[8]xf16>
// CHECK: %[[EXTEND:.*]] = arith.extf %[[EXTRACT]] : vector<[4]xf16> to vector<[4]xf32>
// CHECK: return %[[EXTEND]]
func.func @scalable_extract_from_arith_ext(%src: vector<[8]xf16>) -> vector<[4]xf32> {
  %0 = arith.extf %src : vector<[8]xf16> to vector<[8]xf32>
  %1 = vector.scalable.extract %0[0] : vector<[4]xf32> from vector<[8]xf32>
  return %1 : vector<[4]xf32>
}

/// Negative tests

// -----

// CHECK-LABEL: @outerproduct_widening_2way__no_acc
// CHECK-NOT: arm_sme.fmopa_2way
// CHECK: arm_sme.outerproduct
// CHECK-NOT: arm_sme.fmopa_2way
func.func @outerproduct_widening_2way__no_acc(%a0 : vector<[4]xf16>, %b0 : vector<[4]xf16>) -> vector<[4]x[4]xf32> {
  %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xf32>, vector<[4]xf32>

  return %0 : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_widening_4way__no_acc
// CHECK-NOT: arm_sme.fmopa_4way
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK-NOT: arm_sme.fmopa_4way
func.func @outerproduct_widening_4way__no_acc(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) : vector<[4]xi32>, vector<[4]xi32>

  return %2 : vector<[4]x[4]xi32>
}

// -----

/// Defining op of accumulator operand must be an 'arm_sme.outerproduct'.

// CHECK-LABEL: @outerproduct_widening_2way__bad_acc
// CHECK-NOT: arm_sme.fmopa_2way
// CHECK: arm_sme.outerproduct
// CHECK-NOT: arm_sme.fmopa_2way
func.func @outerproduct_widening_2way__bad_acc(%a0 : vector<[4]xf16>, %b0 : vector<[4]xf16>, %acc : vector<[4]x[4]xf32>) -> vector<[4]x[4]xf32> {
  %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) : vector<[4]xf32>, vector<[4]xf32>

  return %0 : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_widening_4way__missing_acc
// CHECK-NOT: arm_sme.fmopa_4way
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK-NOT: arm_sme.fmopa_4way
func.func @outerproduct_widening_4way__missing_acc(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) : vector<[4]xi32>, vector<[4]xi32>
  // Missing accumulator breaks use-def chain.
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext : vector<[4]xi32>, vector<[4]xi32>
  "test.some_use"(%2) : (vector<[4]x[4]xi32>) -> ()

  return %3 : vector<[4]x[4]xi32>
}

// -----

/// Combining kinds of outer products must match to be fused.

// CHECK-LABEL: @outerproduct_widening_2way__bad_combining_kind
// CHECK-NOT: arm_sme.fmopa_2way
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK-NOT: arm_sme.fmopa_2way
func.func @outerproduct_widening_2way__bad_combining_kind(
    %a0 : vector<[4]xf16>, %b0 : vector<[4]xf16>,
    %a1 : vector<[4]xf16>, %b1 : vector<[4]xf16>) -> vector<[4]x[4]xf32> {
  %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
  %a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
  %b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<add> : vector<[4]xf32>, vector<[4]xf32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<sub> acc(%0) : vector<[4]xf32>, vector<[4]xf32>

  return %1 : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_widening_4way__inconsistent_combining_kind
// CHECK-NOT: arm_sme.fmopa_4way
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK-NOT: arm_sme.fmopa_4way
func.func @outerproduct_widening_4way__inconsistent_combining_kind(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext kind<sub> : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext kind<add> acc(%0) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext kind<add> acc(%1) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext kind<add> acc(%2) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

/// If the first outer product has uses other than as the input to another
/// outer product, it can't be erased after fusion. This is a problem when
/// it also has an accumulator as this will be used as the root for tile
/// allocation and since the widening outer product uses the same
/// accumulator it will get assigned the same tile ID, resulting in 3
/// outer products and incorrect results. Check this is prevented.

// CHECK-LABEL: @outerproduct_widening_2way__cant_erase
// CHECK-NOT: arm_sme.fmopa_2way
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK-NOT: arm_sme.fmopa_2way
func.func @outerproduct_widening_2way__cant_erase(
    %a0 : vector<[4]xf16>, %b0 : vector<[4]xf16>,
    %a1 : vector<[4]xf16>, %b1 : vector<[4]xf16>) -> vector<[4]x[4]xf32> {
  %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
  %a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
  %b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

  %acc = arith.constant dense<1.0> : vector<[4]x[4]xf32>
  %0 = arm_sme.outerproduct %a0_ext, %b0_ext acc(%acc) : vector<[4]xf32>, vector<[4]xf32>
  "test.some_use"(%0) : (vector<[4]x[4]xf32>) -> ()
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xf32>, vector<[4]xf32>

  return %1 : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_widening_4way__multi_use_cant_erase
// CHECK-NOT: arm_sme.fmopa_4way
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK-NOT: arm_sme.fmopa_4way
func.func @outerproduct_widening_4way__multi_use_cant_erase(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xi32>, vector<[4]xi32>
  "test.some_use"(%1) : (vector<[4]x[4]xi32>) -> ()
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

// CHECK-LABEL: @outerproduct_widening_2way__unsupported_type_f32f32f64
// CHECK-NOT: arm_sme.fmopa_2way
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK-NOT: arm_sme.fmopa_2way
func.func @outerproduct_widening_2way__unsupported_type_f32f32f64(
    %a0 : vector<[2]xf32>, %b0 : vector<[2]xf32>,
    %a1 : vector<[2]xf32>, %b1 : vector<[2]xf32>) -> vector<[2]x[2]xf64> {
  %a0_ext = arith.extf %a0 : vector<[2]xf32> to vector<[2]xf64>
  %b0_ext = arith.extf %b0 : vector<[2]xf32> to vector<[2]xf64>
  %a1_ext = arith.extf %a1 : vector<[2]xf32> to vector<[2]xf64>
  %b1_ext = arith.extf %b1 : vector<[2]xf32> to vector<[2]xf64>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[2]xf64>, vector<[2]xf64>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[2]xf64>, vector<[2]xf64>

  return %1 : vector<[2]x[2]xf64>
}

// -----

// CHECK-LABEL: @outerproduct_widening_4way__unsupported_type_f16f16f64
// CHECK-NOT: arm_sme.fmopa_4way
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK-NOT: arm_sme.fmopa_4way
func.func @outerproduct_widening_4way__unsupported_type_f16f16f64(
    %a0 : vector<[2]xf16>, %b0 : vector<[2]xf16>,
    %a1 : vector<[2]xf16>, %b1 : vector<[2]xf16>,
    %a2 : vector<[2]xf16>, %b2 : vector<[2]xf16>,
    %a3 : vector<[2]xf16>, %b3 : vector<[2]xf16>) -> vector<[2]x[2]xf64> {
  %a0_ext = arith.extf %a0 : vector<[2]xf16> to vector<[2]xf64>
  %b0_ext = arith.extf %b0 : vector<[2]xf16> to vector<[2]xf64>

  %a1_ext = arith.extf %a1 : vector<[2]xf16> to vector<[2]xf64>
  %b1_ext = arith.extf %b1 : vector<[2]xf16> to vector<[2]xf64>

  %a2_ext = arith.extf %a2 : vector<[2]xf16> to vector<[2]xf64>
  %b2_ext = arith.extf %b2 : vector<[2]xf16> to vector<[2]xf64>

  %a3_ext = arith.extf %a3 : vector<[2]xf16> to vector<[2]xf64>
  %b3_ext = arith.extf %b3 : vector<[2]xf16> to vector<[2]xf64>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[2]xf64>, vector<[2]xf64>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[2]xf64>, vector<[2]xf64>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) : vector<[2]xf64>, vector<[2]xf64>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) : vector<[2]xf64>, vector<[2]xf64>

  return %3 : vector<[2]x[2]xf64>
}

// -----

/// Fusion only occurs if either both outer products are masked, or neither.

// CHECK-LABEL: @outerproduct_widening_2way__bad_masking
// CHECK-NOT: arm_sme.fmopa_2way
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK-NOT: arm_sme.fmopa_2way
func.func @outerproduct_widening_2way__bad_masking(
    %a0 : vector<[4]xf16>, %b0 : vector<[4]xf16>,
    %a1 : vector<[4]xf16>, %b1 : vector<[4]xf16>,
    %a1_mask : vector<[4]xi1>, %b1_mask : vector<[4]xi1>) -> vector<[4]x[4]xf32> {
  %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
  %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
  %a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
  %b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xf32>, vector<[4]xf32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) masks(%a1_mask, %b1_mask) : vector<[4]xf32>, vector<[4]xf32>

  return %1 : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_widening_4way__inconsistent_masking
// CHECK-NOT: arm_sme.fmopa_4way
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK-NOT: arm_sme.fmopa_4way
func.func @outerproduct_widening_4way__inconsistent_masking(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi8>, %b2 : vector<[4]xi8>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>,
    %a2_mask : vector<[4]xi1>, %b2_mask : vector<[4]xi1>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a2_ext = arith.extsi %a2 : vector<[4]xi8> to vector<[4]xi32>
  %b2_ext = arith.extsi %b2 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xi32>, vector<[4]xi32>
  %2 = arm_sme.outerproduct %a2_ext, %b2_ext acc(%1) masks(%a2_mask, %b2_mask) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

// -----

/// Defining op of outer product must be a supported extension op.

// CHECK-LABEL: @outerproduct_widening_2way__bad_defining_op
// CHECK-NOT: arm_sme.fmopa_2way
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK-NOT: arm_sme.fmopa_2way
func.func @outerproduct_widening_2way__bad_defining_op(
    %a0 : vector<[4]xf32>, %b0 : vector<[4]xf32>,
    %a1 : vector<[4]xf32>, %b1 : vector<[4]xf32>) -> vector<[4]x[4]xf32> {
  %0 = arm_sme.outerproduct %a0, %b0 : vector<[4]xf32>, vector<[4]xf32>
  %1 = arm_sme.outerproduct %a1, %b1 acc(%0) : vector<[4]xf32>, vector<[4]xf32>

  return %1 : vector<[4]x[4]xf32>
}

// -----

// CHECK-LABEL: @outerproduct_widening_4way__bad_defining_op
// CHECK-NOT: arm_sme.fmopa_4way
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK: arm_sme.outerproduct
// CHECK-NOT: arm_sme.fmopa_4way
func.func @outerproduct_widening_4way__bad_defining_op(
    %a0 : vector<[4]xi8>, %b0 : vector<[4]xi8>,
    %a1 : vector<[4]xi8>, %b1 : vector<[4]xi8>,
    %a2 : vector<[4]xi32>, %b2 : vector<[4]xi32>,
    %a3 : vector<[4]xi8>, %b3 : vector<[4]xi8>) -> vector<[4]x[4]xi32> {
  %a0_ext = arith.extsi %a0 : vector<[4]xi8> to vector<[4]xi32>
  %b0_ext = arith.extsi %b0 : vector<[4]xi8> to vector<[4]xi32>

  %a1_ext = arith.extsi %a1 : vector<[4]xi8> to vector<[4]xi32>
  %b1_ext = arith.extsi %b1 : vector<[4]xi8> to vector<[4]xi32>

  %a3_ext = arith.extsi %a3 : vector<[4]xi8> to vector<[4]xi32>
  %b3_ext = arith.extsi %b3 : vector<[4]xi8> to vector<[4]xi32>

  %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xi32>, vector<[4]xi32>
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xi32>, vector<[4]xi32>
  /// Inputs must come from an arith.ext.
  %2 = arm_sme.outerproduct %a2, %b2 acc(%1) : vector<[4]xi32>, vector<[4]xi32>
  %3 = arm_sme.outerproduct %a3_ext, %b3_ext acc(%2) : vector<[4]xi32>, vector<[4]xi32>

  return %3 : vector<[4]x[4]xi32>
}

/// Negative tests for related patterns.

// -----

/// Non-vector extracts should be ignored.

// CHECK-LABEL: @extract_scalar_from_arith_ext
// CHECK-NEXT: arith.extsi
// CHECK-NEXT: vector.extract
func.func @extract_scalar_from_arith_ext(%src: vector<4x[8]xi8>) -> i32 {
  %0 = arith.extsi %src : vector<4x[8]xi8> to vector<4x[8]xi32>
  %1 = vector.extract %0[0, 0] : i32 from vector<4x[8]xi32>
  return %1 : i32
}

// -----

/// Extracted type should be a 1-D scalable vector type.

// CHECK-LABEL: @extract_fixed_1d_vec_from_arith_ext
// CHECK-NEXT: arith.extsi
// CHECK-NEXT: vector.extract
func.func @extract_fixed_1d_vec_from_arith_ext(%src: vector<4x8xi8>) -> vector<8xi32> {
  %0 = arith.extsi %src : vector<4x8xi8> to vector<4x8xi32>
  %1 = vector.extract %0[0] : vector<8xi32> from vector<4x8xi32>
  return %1 : vector<8xi32>
}

// -----

/// Extract must come from an arith extend.

// CHECK-LABEL: @extract_from_non_arith_ext
// CHECK-NEXT: vector.extract
// CHECK-NEXT: return
func.func @extract_from_non_arith_ext(%src: vector<4x[8]xi32>) -> vector<[8]xi32> {
  %0 = vector.extract %src[0] : vector<[8]xi32> from vector<4x[8]xi32>
  return %0 : vector<[8]xi32>
}

// -----

/// Scalable extract must come from an arith extend.

// CHECK-LABEL: @scalable_extract_from_non_arith_ext
// CHECK-NEXT: vector.scalable.extract
// CHECK-NEXT: return
func.func @scalable_extract_from_non_arith_ext(%src: vector<[8]xf32>) -> vector<[4]xf32> {
  %0 = vector.scalable.extract %src[0] : vector<[4]xf32> from vector<[8]xf32>
  return %0 : vector<[4]xf32>
}
