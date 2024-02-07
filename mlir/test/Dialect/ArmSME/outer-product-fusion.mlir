// RUN: mlir-opt %s -arm-sme-outer-product-fusion -cse -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @outerproduct_add_widening_2way_f16f16f32
// CHECK-SAME:    %[[A0:.*]]: vector<[4]xf16>, %[[B0:.*]]: vector<[4]xf16>, %[[A1:.*]]: vector<[4]xf16>, %[[B1:.*]]: vector<[4]xf16>,
// CHECK-SAME:    %[[A0_MASK:.*]]: vector<[4]xi1>, %[[B0_MASK:.*]]: vector<[4]xi1>, %[[A1_MASK:.*]]: vector<[4]xi1>, %[[B1_MASK:.*]]: vector<[4]xi1>
// CHECK-DAG: %[[ACC:.*]] = arith.constant dense<0.000000e+00> : vector<[4]x[4]xf32>
// CHECK-DAG: %[[LHS:.*]] = "llvm.intr.experimental.vector.interleave2"(%[[A0]], %[[A1]]) : (vector<[4]xf16>, vector<[4]xf16>) -> vector<[8]xf16>
// CHECK-DAG: %[[RHS:.*]] = "llvm.intr.experimental.vector.interleave2"(%[[B0]], %[[B1]]) : (vector<[4]xf16>, vector<[4]xf16>) -> vector<[8]xf16>
// CHECK-DAG: %[[LHS_MASK:.*]] = "llvm.intr.experimental.vector.interleave2"(%[[A0_MASK]], %[[A1_MASK]]) : (vector<[4]xi1>, vector<[4]xi1>) -> vector<[8]xi1>
// CHECK-DAG: %[[RHS_MASK:.*]] = "llvm.intr.experimental.vector.interleave2"(%[[B0_MASK]], %[[B1_MASK]]) : (vector<[4]xi1>, vector<[4]xi1>) -> vector<[8]xi1>
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
  "fake.use"(%0) : (vector<[4]x[4]xf32>) -> ()
  %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xf32>, vector<[4]xf32>

  return %1 : vector<[4]x[4]xf32>
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
