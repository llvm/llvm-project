// RUN: mlir-opt %s -arm-sme-outer-product-widening -cse -split-input-file | FileCheck %s

// CHECK-LABEL: @outerproduct_add_widening_2way_f16f16f32
// CHECK-SAME:    %[[A0:.*]]: vector<[4]xf16>, %[[B0:.*]]: vector<[4]xf16>, %[[A1:.*]]: vector<[4]xf16>, %[[B1:.*]]: vector<[4]xf16>,
// CHECK-SAME:    %[[A0_MASK:.*]]: vector<[4]xi1>, %[[B0_MASK:.*]]: vector<[4]xi1>, %[[A1_MASK:.*]]: vector<[4]xi1>, %[[B1_MASK:.*]]: vector<[4]xi1>
// CHECK-DAG: %[[ACC:.*]] = arith.constant dense<0.000000e+00> : vector<[4]x[4]xf32>
// CHECK-DAG: %[[LHS:.*]] = "llvm.intr.experimental.vector.interleave2"(%[[A0]], %[[A1]]) : (vector<[4]xf16>, vector<[4]xf16>) -> vector<[8]xf16>
// CHECK-DAG: %[[RHS:.*]] = "llvm.intr.experimental.vector.interleave2"(%[[B0]], %[[B1]]) : (vector<[4]xf16>, vector<[4]xf16>) -> vector<[8]xf16>
// CHECK-DAG: %[[LHS_MASK:.*]] = "llvm.intr.experimental.vector.interleave2"(%[[A0_MASK]], %[[A1_MASK]]) : (vector<[4]xi1>, vector<[4]xi1>) -> vector<[8]xi1>
// CHECK-DAG: %[[RHS_MASK:.*]] = "llvm.intr.experimental.vector.interleave2"(%[[B0_MASK]], %[[B1_MASK]]) : (vector<[4]xi1>, vector<[4]xi1>) -> vector<[8]xi1>
// CHECK-DAG: arm_sme.fmopa_wide_2way %[[LHS]], %[[RHS]] acc(%[[ACC]]) masks(%[[LHS_MASK]], %[[RHS_MASK]]) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
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

// CHECK-LABEL: @outerproduct_sub_widening_2way_f16f16f32
// CHECK: arm_sme.fmops_wide_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
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
// CHECK: arm_sme.fmopa_wide_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
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
// CHECK: arm_sme.fmops_wide_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xbf16>, vector<[8]xbf16> into vector<[4]x[4]xf32>
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
// CHECK: arm_sme.smopa_wide_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
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
// CHECK: arm_sme.smops_wide_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
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
// CHECK: arm_sme.umopa_wide_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
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
// CHECK: arm_sme.umops_wide_2way %{{.*}}, %{{.*}} acc(%{{.*}}) masks(%{{.*}}, %{{.*}}) : vector<[8]xi16>, vector<[8]xi16> into vector<[4]x[4]xi32>
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
