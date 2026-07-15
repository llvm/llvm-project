// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

///----------------------------------------------------------------------------------------
/// arith.extsi
///
/// [Pattern: RewriteAlignedSubByteIntExt]
///----------------------------------------------------------------------------------------
// Negative test - the trailing dim 1 is not a multiple of 2 (i.e. 8 / 4).
// CHECK-LABEL: func.func @unaligned_extsi_i4_to_i8(
func.func @unaligned_extsi_i4_to_i8(%a: vector<1xi4>) -> vector<1xi8> {
  // CHECK-NOT: arith.bitcast
  // CHECK: arith.extsi %[[IN:.*]] : vector<1xi4> to vector<1xi8>
  %0 = arith.extsi %a : vector<1xi4> to vector<1xi8>
  return %0 : vector<1xi8>
}

// Negative test - the trailing dim 2 is not a multiple of 4 (i.e. 8 / 2).
// CHECK-LABEL: func.func @unaligned_extsi_i2_to_i8(
func.func @unaligned_extsi_i2_to_i8(%a: vector<2xi2>) -> vector<2xi8> {
  // CHECK-NOT: arith.bitcast
  // CHECK: arith.extsi %[[IN:.*]] : vector<2xi2> to vector<2xi8>
  %0 = arith.extsi %a : vector<2xi2> to vector<2xi8>
  return %0 : vector<2xi8>
}

// CHECK-LABEL: func.func @aligned_extsi_i4_to_i8(
func.func @aligned_extsi_i4_to_i8(%a: vector<8xi4>) -> vector<8xi8> {
// CHECK-SAME:    %[[IN:.*]]: vector<8xi4>) -> vector<8xi8> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<4xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8xi4> to vector<4xi8>
// CHECK:           %[[SHL_LOW:.*]] = arith.shli %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[LOW:.*]] = arith.shrsi %[[SHL_LOW]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrsi %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<4xi8>
  %0 = arith.extsi %a : vector<8xi4> to vector<8xi8>
  return %0 : vector<8xi8>
}

// CHECK-LABEL: func.func @aligned_extsi_i2_to_i8(
func.func @aligned_extsi_i2_to_i8(%a: vector<8xi2>) -> vector<8xi8> {
// CHECK-SAME:      %[[IN:.*]]: vector<8xi2>) -> vector<8xi8> {
// CHECK:           %[[CST_2:.*]] = arith.constant dense<2> : vector<2xi8>
// CHECK:           %[[CST_4:.*]] = arith.constant dense<4> : vector<2xi8>
// CHECK:           %[[CST_6:.*]] = arith.constant dense<6> : vector<2xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8xi2> to vector<2xi8>
// Extract bits 0-1
// CHECK:           %[[SHL_6:.*]] = arith.shli %[[BITCAST]], %[[CST_6]] : vector<2xi8>
// CHECK:           %[[ELEM0:.*]] = arith.shrsi %[[SHL_6]], %[[CST_6]] : vector<2xi8>
// Extract bits 2-3
// CHECK:           %[[SHL_4:.*]] = arith.shli %[[BITCAST]], %[[CST_4]] : vector<2xi8>
// CHECK:           %[[ELEM1:.*]] = arith.shrsi %[[SHL_4]], %[[CST_6]] : vector<2xi8>
// Extract bits 4-5
// CHECK:           %[[SHL_2:.*]] = arith.shli %[[BITCAST]], %[[CST_2]] : vector<2xi8>
// CHECK:           %[[ELEM2:.*]] = arith.shrsi %[[SHL_2]], %[[CST_6]] : vector<2xi8>
// Extract bits 6-7
// CHECK:           %[[ELEM3:.*]] = arith.shrsi %[[BITCAST]], %[[CST_6]] : vector<2xi8>
// CHECK:           %[[INTERLEAVE02:.*]] = vector.interleave %[[ELEM0]], %[[ELEM2]] : vector<2xi8>
// CHECK:           %[[INTERLEAVE13:.*]] = vector.interleave %[[ELEM1]], %[[ELEM3]] : vector<2xi8>
// CHECK:           %[[RESULT:.*]] = vector.interleave %[[INTERLEAVE02]], %[[INTERLEAVE13]] : vector<4xi8>
  %0 = arith.extsi %a : vector<8xi2> to vector<8xi8>
  return %0 : vector<8xi8>
}

// CHECK-LABEL: func.func @aligned_extsi_i4_to_i32(
func.func @aligned_extsi_i4_to_i32(%a: vector<8xi4>) -> vector<8xi32> {
// CHECK-SAME:    %[[IN:.*]]: vector<8xi4>) -> vector<8xi32> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<4xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8xi4> to vector<4xi8>
// CHECK:           %[[SHL_LOW:.*]] = arith.shli %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[LOW:.*]] = arith.shrsi %[[SHL_LOW]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrsi %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<4xi8>
// CHECK:           %[[I32:.*]] = arith.extsi %[[INTERLEAVE]] : vector<8xi8> to vector<8xi32>
  %0 = arith.extsi %a : vector<8xi4> to vector<8xi32>
  return %0 : vector<8xi32>
}

// CHECK-LABEL: func.func @aligned_extsi_i2_to_i32(
func.func @aligned_extsi_i2_to_i32(%a: vector<8xi2>) -> vector<8xi32> {
// CHECK-SAME:      %[[IN:.*]]: vector<8xi2>) -> vector<8xi32> {
// CHECK:           %[[CST_2:.*]] = arith.constant dense<2> : vector<2xi8>
// CHECK:           %[[CST_4:.*]] = arith.constant dense<4> : vector<2xi8>
// CHECK:           %[[CST_6:.*]] = arith.constant dense<6> : vector<2xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8xi2> to vector<2xi8>
// Extract bits 0-1
// CHECK:           %[[SHL_6:.*]] = arith.shli %[[BITCAST]], %[[CST_6]] : vector<2xi8>
// CHECK:           %[[ELEM0:.*]] = arith.shrsi %[[SHL_6]], %[[CST_6]] : vector<2xi8>
// Extract bits 2-3
// CHECK:           %[[SHL_4:.*]] = arith.shli %[[BITCAST]], %[[CST_4]] : vector<2xi8>
// CHECK:           %[[ELEM1:.*]] = arith.shrsi %[[SHL_4]], %[[CST_6]] : vector<2xi8>
// Extract bits 4-5
// CHECK:           %[[SHL_2:.*]] = arith.shli %[[BITCAST]], %[[CST_2]] : vector<2xi8>
// CHECK:           %[[ELEM2:.*]] = arith.shrsi %[[SHL_2]], %[[CST_6]] : vector<2xi8>
// Extract bits 6-7
// CHECK:           %[[ELEM3:.*]] = arith.shrsi %[[BITCAST]], %[[CST_6]] : vector<2xi8>
// CHECK:           %[[INTERLEAVE02:.*]] = vector.interleave %[[ELEM0]], %[[ELEM2]] : vector<2xi8>
// CHECK:           %[[INTERLEAVE13:.*]] = vector.interleave %[[ELEM1]], %[[ELEM3]] : vector<2xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[INTERLEAVE02]], %[[INTERLEAVE13]] : vector<4xi8>
// CHECK:           %[[RESULT:.*]] = arith.extsi %[[INTERLEAVE]] : vector<8xi8> to vector<8xi32>
  %0 = arith.extsi %a : vector<8xi2> to vector<8xi32>
  return %0 : vector<8xi32>
}

// CHECK-LABEL: func.func @aligned_extsi_i4_to_i32_2d(
func.func @aligned_extsi_i4_to_i32_2d(%a: vector<8x32xi4>) -> vector<8x32xi32> {
// CHECK-SAME:    %[[IN:.*]]: vector<8x32xi4>) -> vector<8x32xi32> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<8x16xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8x32xi4> to vector<8x16xi8>
// CHECK:           %[[SHL_LOW:.*]] = arith.shli %[[BITCAST]], %[[I4_BITS]] : vector<8x16xi8>
// CHECK:           %[[LOW:.*]] = arith.shrsi %[[SHL_LOW]], %[[I4_BITS]] : vector<8x16xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrsi %[[BITCAST]], %[[I4_BITS]] : vector<8x16xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<8x16xi8>
// CHECK:           %[[I32:.*]] = arith.extsi %[[INTERLEAVE]] : vector<8x32xi8> to vector<8x32xi32>
  %0 = arith.extsi %a : vector<8x32xi4> to vector<8x32xi32>
  return %0 : vector<8x32xi32>
}

// CHECK-LABEL: func.func @aligned_extsi_i2_to_i32_2d(
func.func @aligned_extsi_i2_to_i32_2d(%a: vector<8x32xi2>) -> vector<8x32xi32> {
// CHECK-SAME:      %[[IN:.*]]: vector<8x32xi2>) -> vector<8x32xi32> {
// CHECK:           %[[CST_2:.*]] = arith.constant dense<2> : vector<8x8xi8>
// CHECK:           %[[CST_4:.*]] = arith.constant dense<4> : vector<8x8xi8>
// CHECK:           %[[CST_6:.*]] = arith.constant dense<6> : vector<8x8xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8x32xi2> to vector<8x8xi8>
// Extract bits 0-1
// CHECK:           %[[SHL_6:.*]] = arith.shli %[[BITCAST]], %[[CST_6]] : vector<8x8xi8>
// CHECK:           %[[ELEM0:.*]] = arith.shrsi %[[SHL_6]], %[[CST_6]] : vector<8x8xi8>
// Extract bits 2-3
// CHECK:           %[[SHL_4:.*]] = arith.shli %[[BITCAST]], %[[CST_4]] : vector<8x8xi8>
// CHECK:           %[[ELEM1:.*]] = arith.shrsi %[[SHL_4]], %[[CST_6]] : vector<8x8xi8>
// Extract bits 4-5
// CHECK:           %[[SHL_2:.*]] = arith.shli %[[BITCAST]], %[[CST_2]] : vector<8x8xi8>
// CHECK:           %[[ELEM2:.*]] = arith.shrsi %[[SHL_2]], %[[CST_6]] : vector<8x8xi8>
// Extract bits 6-7
// CHECK:           %[[ELEM3:.*]] = arith.shrsi %[[BITCAST]], %[[CST_6]] : vector<8x8xi8>
// CHECK:           %[[INTERLEAVE02:.*]] = vector.interleave %[[ELEM0]], %[[ELEM2]] : vector<8x8xi8>
// CHECK:           %[[INTERLEAVE13:.*]] = vector.interleave %[[ELEM1]], %[[ELEM3]] : vector<8x8xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[INTERLEAVE02]], %[[INTERLEAVE13]] : vector<8x16xi8>
// CHECK:           %[[RESULT:.*]] = arith.extsi %[[INTERLEAVE]] : vector<8x32xi8> to vector<8x32xi32>
  %0 = arith.extsi %a : vector<8x32xi2> to vector<8x32xi32>
  return %0 : vector<8x32xi32>
}

///----------------------------------------------------------------------------------------
/// arith.trunci
///
/// [Pattern: RewriteAlignedSubByteIntTrunc]
///----------------------------------------------------------------------------------------
// CHECK-LABEL: func.func @aligned_trunci_i8_to_i4(
func.func @aligned_trunci_i8_to_i4(%a: vector<8xi8>) -> vector<8xi4> {
// CHECK-SAME:    %[[IN:.*]]: vector<8xi8>) -> vector<8xi4> {
// CHECK-DAG:       %[[LOW_MASK:.*]] = arith.constant dense<15> : vector<4xi8>
// CHECK-DAG:       %[[I4_BITS:.*]] = arith.constant dense<4> : vector<4xi8>
// CHECK:           %[[LOW:.*]], %[[HIGH:.*]] = vector.deinterleave %[[IN]] : vector<8xi8> -> vector<4xi8>
// CHECK:           %[[ZEROED_LOW:.*]] = arith.andi %[[LOW]], %[[LOW_MASK]] : vector<4xi8>
// CHECK:           %[[SHL_HIGH:.*]] = arith.shli %[[HIGH]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[MERGED:.*]] = arith.ori %[[ZEROED_LOW]], %[[SHL_HIGH]] : vector<4xi8>
// CHECK:           %[[I4:.*]] = vector.bitcast %[[MERGED]] : vector<4xi8> to vector<8xi4>
  %0 = arith.trunci %a : vector<8xi8> to vector<8xi4>
  return %0 : vector<8xi4>
}

// CHECK-LABEL: func.func @aligned_trunci_i32_to_i4(
func.func @aligned_trunci_i32_to_i4(%a: vector<8xi32>) -> vector<8xi4> {
// CHECK-SAME:    %[[IN:.*]]: vector<8xi32>) -> vector<8xi4> {
// CHECK-DAG:       %[[LOW_MASK:.*]] = arith.constant dense<15> : vector<4xi8>
// CHECK-DAG:       %[[I4_BITS:.*]] = arith.constant dense<4> : vector<4xi8>
// CHECK:           %[[I8:.*]] = arith.trunci %[[IN]] : vector<8xi32> to vector<8xi8>
// CHECK:           %[[LOW:.*]], %[[HIGH:.*]] = vector.deinterleave %[[I8]] : vector<8xi8> -> vector<4xi8>
// CHECK:           %[[ZEROED_LOW:.*]] = arith.andi %[[LOW]], %[[LOW_MASK]] : vector<4xi8>
// CHECK:           %[[SHL_HIGH:.*]] = arith.shli %[[HIGH]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[MERGED:.*]] = arith.ori %[[ZEROED_LOW]], %[[SHL_HIGH]] : vector<4xi8>
// CHECK:           %[[I4:.*]] = vector.bitcast %[[MERGED]] : vector<4xi8> to vector<8xi4>
  %0 = arith.trunci %a : vector<8xi32> to vector<8xi4>
  return %0 : vector<8xi4>
}

// CHECK-LABEL: func.func @aligned_trunci_2d(
func.func @aligned_trunci_2d(%a: vector<8x32xi32>) -> vector<8x32xi4> {
// CHECK-NOT:       vector.shuffle
// CHECK-NOT:       vector.andi
// CHECK-NOT:       vector.shli
// CHECK-NOT:       vector.ori
// CHECK:           arith.trunci {{.*}} : vector<8x32xi32> to vector<8x32xi8>
// CHECK-NOT:       arith.trunci {{.*}} : vector<8x32xi8> to vector<8x32xi4>
// CHECK:           vector.deinterleave
  %0 = arith.trunci %a : vector<8x32xi32> to vector<8x32xi4>
  return %0 : vector<8x32xi4>
}

// CHECK-LABEL: func.func @aligned_trunci_nd(
// CHECK-SAME: %[[IN:.*]]: vector<3x8x32xi32>) -> vector<3x8x32xi4> {
func.func @aligned_trunci_nd(%a: vector<3x8x32xi32>) -> vector<3x8x32xi4> {
  // CHECK: %[[LEFT_SHIFT_BITS:.*]] = arith.constant dense<4> : vector<3x8x16xi8>
  // CHECK: %[[I4_MASK:.*]] = arith.constant dense<15> : vector<3x8x16xi8>
  // CHECK: %[[I8:.*]] = arith.trunci %[[IN]] : vector<3x8x32xi32> to vector<3x8x32xi8>
  // CHECK: %[[LOW:.*]], %[[HIGH:.*]] = vector.deinterleave %[[I8]] : vector<3x8x32xi8> -> vector<3x8x16xi8>
  // CHECK: %[[ZEROED_LOW:.*]] = arith.andi %[[LOW]], %[[I4_MASK]] : vector<3x8x16xi8>
  // CHECK: %[[SHL_HIGH:.*]] = arith.shli %[[HIGH]], %[[LEFT_SHIFT_BITS]] : vector<3x8x16xi8>
  // CHECK: %[[MERGED:.*]] = arith.ori %[[ZEROED_LOW]], %[[SHL_HIGH]] : vector<3x8x16xi8>
  // CHECK: %[[I4:.*]] = vector.bitcast %[[MERGED]] : vector<3x8x16xi8> to vector<3x8x32xi4>
  %0 = arith.trunci %a : vector<3x8x32xi32> to vector<3x8x32xi4>
  return %0 : vector<3x8x32xi4>
}

func.func @aligned_trunci_i8_to_i2_no_match(%a: vector<8xi8>) -> vector<8xi2> {
  // CHECK-NOT: arith.bitcast
  // CHECK: arith.trunci %[[IN:.*]] : vector<8xi8> to vector<8xi2>
  %0 = arith.trunci %a : vector<8xi8> to vector<8xi2>
  return %0 : vector<8xi2>
}

///----------------------------------------------------------------------------------------
/// arith.extui
///
/// [Pattern: RewriteAlignedSubByteIntExt]
///----------------------------------------------------------------------------------------

// CHECK-LABEL: func.func @aligned_extui_i4_to_i8(
func.func @aligned_extui_i4_to_i8(%a: vector<8xi4>) -> vector<8xi8> {
// CHECK-SAME:                             %[[IN:.*]]: vector<8xi4>) -> vector<8xi8> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<4xi8>
// CHECK:           %[[LOWBITS_MASK:.*]] = arith.constant dense<15> : vector<4xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8xi4> to vector<4xi8>
// CHECK:           %[[LOW:.*]] = arith.andi %[[BITCAST]], %[[LOWBITS_MASK]] : vector<4xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrui %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<4xi8>
  %0 = arith.extui %a : vector<8xi4> to vector<8xi8>
  return %0 : vector<8xi8>
}

// CHECK-LABEL: func.func @aligned_extui_i2_to_i8(
func.func @aligned_extui_i2_to_i8(%a: vector<8xi2>) -> vector<8xi8> {
// CHECK-SAME:      %[[IN:.*]]: vector<8xi2>) -> vector<8xi8> {
// CHECK:           %[[CST_6:.*]] = arith.constant dense<6> : vector<2xi8>
// CHECK:           %[[CST_4:.*]] = arith.constant dense<4> : vector<2xi8>
// CHECK:           %[[CST_2:.*]] = arith.constant dense<2> : vector<2xi8>
// CHECK:           %[[LOWBITS_MASK:.*]] = arith.constant dense<3> : vector<2xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8xi2> to vector<2xi8>
// Extract bits 0-1
// CHECK:           %[[ELEM0:.*]] = arith.andi %[[BITCAST]], %[[LOWBITS_MASK]] : vector<2xi8>
// Extract bits 2-3
// CHECK:           %[[SHR_2:.*]] = arith.shrui %[[BITCAST]], %[[CST_2]] : vector<2xi8>
// CHECK:           %[[ELEM1:.*]] = arith.andi %[[SHR_2]], %[[LOWBITS_MASK]] : vector<2xi8>
// Extract bits 4-5
// CHECK:           %[[SHR_4:.*]] = arith.shrui %[[BITCAST]], %[[CST_4]] : vector<2xi8>
// CHECK:           %[[ELEM2:.*]] = arith.andi %[[SHR_4]], %[[LOWBITS_MASK]] : vector<2xi8>
// Extract bits 6-7
// CHECK:           %[[ELEM3:.*]] = arith.shrui %[[BITCAST]], %[[CST_6]] : vector<2xi8>
// CHECK:           %[[INTERLEAVE02:.*]] = vector.interleave %[[ELEM0]], %[[ELEM2]] : vector<2xi8>
// CHECK:           %[[INTERLEAVE13:.*]] = vector.interleave %[[ELEM1]], %[[ELEM3]] : vector<2xi8>
// CHECK:           %[[RESULT:.*]] = vector.interleave %[[INTERLEAVE02]], %[[INTERLEAVE13]] : vector<4xi8>
  %0 = arith.extui %a : vector<8xi2> to vector<8xi8>
  return %0 : vector<8xi8>
}

// CHECK-LABEL: func.func @aligned_extui_i4_to_i32(
func.func @aligned_extui_i4_to_i32(%a: vector<8xi4>) -> vector<8xi32> {
// CHECK-SAME:                             %[[IN:.*]]: vector<8xi4>) -> vector<8xi32> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<4xi8>
// CHECK:           %[[LOWBITS_MASK:.*]] = arith.constant dense<15> : vector<4xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8xi4> to vector<4xi8>
// CHECK:           %[[LOW:.*]] = arith.andi %[[BITCAST]], %[[LOWBITS_MASK]] : vector<4xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrui %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<4xi8>
// CHECK:           %[[I32:.*]] = arith.extui %[[INTERLEAVE]] : vector<8xi8> to vector<8xi32>
  %0 = arith.extui %a : vector<8xi4> to vector<8xi32>
  return %0 : vector<8xi32>
}

// CHECK-LABEL: func.func @aligned_extui_i2_to_i32(
func.func @aligned_extui_i2_to_i32(%a: vector<8xi2>) -> vector<8xi32> {
// CHECK-SAME:      %[[IN:.*]]: vector<8xi2>) -> vector<8xi32> {
// CHECK:           %[[CST_6:.*]] = arith.constant dense<6> : vector<2xi8>
// CHECK:           %[[CST_4:.*]] = arith.constant dense<4> : vector<2xi8>
// CHECK:           %[[CST_2:.*]] = arith.constant dense<2> : vector<2xi8>
// CHECK:           %[[LOWBITS_MASK:.*]] = arith.constant dense<3> : vector<2xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8xi2> to vector<2xi8>
// Extract bits 0-1
// CHECK:           %[[ELEM0:.*]] = arith.andi %[[BITCAST]], %[[LOWBITS_MASK]] : vector<2xi8>
// Extract bits 2-3
// CHECK:           %[[SHR_2:.*]] = arith.shrui %[[BITCAST]], %[[CST_2]] : vector<2xi8>
// CHECK:           %[[ELEM1:.*]] = arith.andi %[[SHR_2]], %[[LOWBITS_MASK]] : vector<2xi8>
// Extract bits 4-5
// CHECK:           %[[SHR_4:.*]] = arith.shrui %[[BITCAST]], %[[CST_4]] : vector<2xi8>
// CHECK:           %[[ELEM2:.*]] = arith.andi %[[SHR_4]], %[[LOWBITS_MASK]] : vector<2xi8>
// Extract bits 6-7
// CHECK:           %[[ELEM3:.*]] = arith.shrui %[[BITCAST]], %[[CST_6]] : vector<2xi8>
// CHECK:           %[[INTERLEAVE02:.*]] = vector.interleave %[[ELEM0]], %[[ELEM2]] : vector<2xi8>
// CHECK:           %[[INTERLEAVE13:.*]] = vector.interleave %[[ELEM1]], %[[ELEM3]] : vector<2xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[INTERLEAVE02]], %[[INTERLEAVE13]] : vector<4xi8>
// CHECK:           %[[RESULT:.*]] = arith.extui %[[INTERLEAVE]] : vector<8xi8> to vector<8xi32>
  %0 = arith.extui %a : vector<8xi2> to vector<8xi32>
  return %0 : vector<8xi32>
}

// CHECK-LABEL: func.func @aligned_extui_i4_to_i32_2d(
func.func @aligned_extui_i4_to_i32_2d(%a: vector<8x32xi4>) -> vector<8x32xi32> {
// CHECK-SAME:                                %[[VAL_0:.*]]: vector<8x32xi4>) -> vector<8x32xi32> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<8x16xi8>
// CHECK:           %[[LOWBITS_MASK:.*]] = arith.constant dense<15> : vector<8x16xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[VAL_0]] : vector<8x32xi4> to vector<8x16xi8>
// CHECK:           %[[LOW:.*]] = arith.andi %[[BITCAST]], %[[LOWBITS_MASK]] : vector<8x16xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrui %[[BITCAST]], %[[I4_BITS]] : vector<8x16xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<8x16xi8>
// CHECK:           %[[I32:.*]] = arith.extui %[[INTERLEAVE]] : vector<8x32xi8> to vector<8x32xi32>
  %0 = arith.extui %a : vector<8x32xi4> to vector<8x32xi32>
  return %0 : vector<8x32xi32>
}

// CHECK-LABEL: func.func @aligned_extui_i2_to_i32_2d(
func.func @aligned_extui_i2_to_i32_2d(%a: vector<8x32xi2>) -> vector<8x32xi32> {
// CHECK-SAME:      %[[IN:.*]]: vector<8x32xi2>) -> vector<8x32xi32> {
// CHECK:           %[[CST_6:.*]] = arith.constant dense<6> : vector<8x8xi8>
// CHECK:           %[[CST_4:.*]] = arith.constant dense<4> : vector<8x8xi8>
// CHECK:           %[[CST_2:.*]] = arith.constant dense<2> : vector<8x8xi8>
// CHECK:           %[[LOWBITS_MASK:.*]] = arith.constant dense<3> : vector<8x8xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8x32xi2> to vector<8x8xi8>
// Extract bits 0-1
// CHECK:           %[[ELEM0:.*]] = arith.andi %[[BITCAST]], %[[LOWBITS_MASK]] : vector<8x8xi8>
// Extract bits 2-3
// CHECK:           %[[SHR_2:.*]] = arith.shrui %[[BITCAST]], %[[CST_2]] : vector<8x8xi8>
// CHECK:           %[[ELEM1:.*]] = arith.andi %[[SHR_2]], %[[LOWBITS_MASK]] : vector<8x8xi8>
// Extract bits 4-5
// CHECK:           %[[SHR_4:.*]] = arith.shrui %[[BITCAST]], %[[CST_4]] : vector<8x8xi8>
// CHECK:           %[[ELEM2:.*]] = arith.andi %[[SHR_4]], %[[LOWBITS_MASK]] : vector<8x8xi8>
// Extract bits 6-7
// CHECK:           %[[ELEM3:.*]] = arith.shrui %[[BITCAST]], %[[CST_6]] : vector<8x8xi8>
// CHECK:           %[[INTERLEAVE02:.*]] = vector.interleave %[[ELEM0]], %[[ELEM2]] : vector<8x8xi8>
// CHECK:           %[[INTERLEAVE13:.*]] = vector.interleave %[[ELEM1]], %[[ELEM3]] : vector<8x8xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[INTERLEAVE02]], %[[INTERLEAVE13]] : vector<8x16xi8>
// CHECK:           %[[RESULT:.*]] = arith.extui %[[INTERLEAVE]] : vector<8x32xi8> to vector<8x32xi32>
  %0 = arith.extui %a : vector<8x32xi2> to vector<8x32xi32>
  return %0 : vector<8x32xi32>
}

///----------------------------------------------------------------------------------------
/// arith.sitofp
///
/// [Pattern: RewriteAlignedSubByteIntExt]
///----------------------------------------------------------------------------------------

// CHECK-LABEL: func.func @aligned_sitofp(
func.func @aligned_sitofp(%a: vector<8xi4>) -> vector<8xf32> {
// CHECK-SAME:    %[[IN:.*]]: vector<8xi4>) -> vector<8xf32> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<4xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8xi4> to vector<4xi8>
// CHECK:           %[[SHL_LOW:.*]] = arith.shli %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[LOW:.*]] = arith.shrsi %[[SHL_LOW]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrsi %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<4xi8>
// CHECK:           %[[F32:.*]] = arith.sitofp %[[INTERLEAVE]] : vector<8xi8> to vector<8xf32>
  %0 = arith.sitofp %a : vector<8xi4> to vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func.func @aligned_sitofp_2d(
func.func @aligned_sitofp_2d(%a: vector<8x32xi4>) -> vector<8x32xf32> {
// CHECK-SAME:    %[[IN:.*]]: vector<8x32xi4>) -> vector<8x32xf32> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<8x16xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8x32xi4> to vector<8x16xi8>
// CHECK:           %[[SHL_LOW:.*]] = arith.shli %[[BITCAST]], %[[I4_BITS]] : vector<8x16xi8>
// CHECK:           %[[LOW:.*]] = arith.shrsi %[[SHL_LOW]], %[[I4_BITS]] : vector<8x16xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrsi %[[BITCAST]], %[[I4_BITS]] : vector<8x16xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<8x16xi8>
// CHECK:           %[[F32:.*]] = arith.sitofp %[[INTERLEAVE]] : vector<8x32xi8> to vector<8x32xf32>
  %0 = arith.sitofp %a : vector<8x32xi4> to vector<8x32xf32>
  return %0 : vector<8x32xf32>
}

///----------------------------------------------------------------------------------------
/// arith.uitofp
///
/// [Pattern: RewriteAlignedSubByteIntExt]
///----------------------------------------------------------------------------------------

// CHECK-LABEL: func.func @aligned_uitofp(
func.func @aligned_uitofp(%a: vector<8xi4>) -> vector<8xf32> {
// CHECK-SAME:    %[[IN:.*]]: vector<8xi4>) -> vector<8xf32> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<4xi8>
// CHECK:           %[[LOWBITS_MASK:.*]] = arith.constant dense<15> : vector<4xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8xi4> to vector<4xi8>
// CHECK:           %[[LOW:.*]] = arith.andi %[[BITCAST]], %[[LOWBITS_MASK]] : vector<4xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrui %[[BITCAST]], %[[I4_BITS]] : vector<4xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<4xi8>
// CHECK:           %[[F32:.*]] = arith.uitofp %[[INTERLEAVE]] : vector<8xi8> to vector<8xf32>
  %0 = arith.uitofp %a : vector<8xi4> to vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func.func @aligned_uitofp_2d(
func.func @aligned_uitofp_2d(%a: vector<8x32xi4>) -> vector<8x32xf32> {
// CHECK-SAME:    %[[IN:.*]]: vector<8x32xi4>) -> vector<8x32xf32> {
// CHECK:           %[[I4_BITS:.*]] = arith.constant dense<4> : vector<8x16xi8>
// CHECK:           %[[LOWBITS_MASK:.*]] = arith.constant dense<15> : vector<8x16xi8>
// CHECK:           %[[BITCAST:.*]] = vector.bitcast %[[IN]] : vector<8x32xi4> to vector<8x16xi8>
// CHECK:           %[[LOW:.*]] = arith.andi %[[BITCAST]], %[[LOWBITS_MASK]] : vector<8x16xi8>
// CHECK:           %[[HIGH:.*]] = arith.shrui %[[BITCAST]], %[[I4_BITS]] : vector<8x16xi8>
// CHECK:           %[[INTERLEAVE:.*]] = vector.interleave %[[LOW]], %[[HIGH]] : vector<8x16xi8>
// CHECK:           %[[F32:.*]] = arith.uitofp %[[INTERLEAVE]] : vector<8x32xi8> to vector<8x32xf32>
  %0 = arith.uitofp %a : vector<8x32xi4> to vector<8x32xf32>
  return %0 : vector<8x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
        : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.rewrite_narrow_types
    } : !transform.any_op
    transform.yield
  }
}
