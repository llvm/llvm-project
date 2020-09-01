// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.BitCount
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitcount_scalar
spv.func @bitcount_scalar(%arg0: i16) "None" {
  // CHECK: "llvm.intr.ctpop"(%{{.*}}) : (!llvm.i16) -> !llvm.i16
  %0 = spv.BitCount %arg0: i16
  spv.Return
}

// CHECK-LABEL: @bitcount_vector
spv.func @bitcount_vector(%arg0: vector<3xi32>) "None" {
  // CHECK: "llvm.intr.ctpop"(%{{.*}}) : (!llvm.vec<3 x i32>) -> !llvm.vec<3 x i32>
  %0 = spv.BitCount %arg0: vector<3xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.BitReverse
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitreverse_scalar
spv.func @bitreverse_scalar(%arg0: i64) "None" {
  // CHECK: "llvm.intr.bitreverse"(%{{.*}}) : (!llvm.i64) -> !llvm.i64
  %0 = spv.BitReverse %arg0: i64
  spv.Return
}

// CHECK-LABEL: @bitreverse_vector
spv.func @bitreverse_vector(%arg0: vector<4xi32>) "None" {
  // CHECK: "llvm.intr.bitreverse"(%{{.*}}) : (!llvm.vec<4 x i32>) -> !llvm.vec<4 x i32>
  %0 = spv.BitReverse %arg0: vector<4xi32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.BitFieldInsert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitfield_insert_scalar_same_bit_width
//  CHECK-SAME: %[[BASE:.*]]: !llvm.i32, %[[INSERT:.*]]: !llvm.i32, %[[OFFSET:.*]]: !llvm.i32, %[[COUNT:.*]]: !llvm.i32
spv.func @bitfield_insert_scalar_same_bit_width(%base: i32, %insert: i32, %offset: i32, %count: i32) "None" {
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(-1 : i32) : !llvm.i32
  // CHECK: %[[T0:.*]] = llvm.shl %[[MINUS_ONE]], %[[COUNT]] : !llvm.i32
  // CHECK: %[[T1:.*]] = llvm.xor %[[T0]], %[[MINUS_ONE]] : !llvm.i32
  // CHECK: %[[T2:.*]] = llvm.shl %[[T1]], %[[OFFSET]] : !llvm.i32
  // CHECK: %[[MASK:.*]] = llvm.xor %[[T2]], %[[MINUS_ONE]] : !llvm.i32
  // CHECK: %[[NEW_BASE:.*]] = llvm.and %[[BASE]], %[[MASK]] : !llvm.i32
  // CHECK: %[[SHIFTED_INSERT:.*]] = llvm.shl %[[INSERT]], %[[OFFSET]] : !llvm.i32
  // CHECK: llvm.or %[[NEW_BASE]], %[[SHIFTED_INSERT]] : !llvm.i32
  %0 = spv.BitFieldInsert %base, %insert, %offset, %count : i32, i32, i32
  spv.Return
}

// CHECK-LABEL: @bitfield_insert_scalar_smaller_bit_width
//  CHECK-SAME: %[[BASE:.*]]: !llvm.i64, %[[INSERT:.*]]: !llvm.i64, %[[OFFSET:.*]]: !llvm.i8, %[[COUNT:.*]]: !llvm.i8
spv.func @bitfield_insert_scalar_smaller_bit_width(%base: i64, %insert: i64, %offset: i8, %count: i8) "None" {
  // CHECK: %[[EXT_OFFSET:.*]] = llvm.zext %[[OFFSET]] : !llvm.i8 to !llvm.i64
  // CHECK: %[[EXT_COUNT:.*]] = llvm.zext %[[COUNT]] : !llvm.i8 to !llvm.i64
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(-1 : i64) : !llvm.i64
  // CHECK: %[[T0:.*]] = llvm.shl %[[MINUS_ONE]], %[[EXT_COUNT]] : !llvm.i64
  // CHECK: %[[T1:.*]] = llvm.xor %[[T0]], %[[MINUS_ONE]] : !llvm.i64
  // CHECK: %[[T2:.*]] = llvm.shl %[[T1]], %[[EXT_OFFSET]] : !llvm.i64
  // CHECK: %[[MASK:.*]] = llvm.xor %[[T2]], %[[MINUS_ONE]] : !llvm.i64
  // CHECK: %[[NEW_BASE:.*]] = llvm.and %[[BASE]], %[[MASK]] : !llvm.i64
  // CHECK: %[[SHIFTED_INSERT:.*]] = llvm.shl %[[INSERT]], %[[EXT_OFFSET]] : !llvm.i64
  // CHECK: llvm.or %[[NEW_BASE]], %[[SHIFTED_INSERT]] : !llvm.i64
  %0 = spv.BitFieldInsert %base, %insert, %offset, %count : i64, i8, i8
  spv.Return
}

// CHECK-LABEL: @bitfield_insert_scalar_greater_bit_width
//  CHECK-SAME: %[[BASE:.*]]: !llvm.i16, %[[INSERT:.*]]: !llvm.i16, %[[OFFSET:.*]]: !llvm.i32, %[[COUNT:.*]]: !llvm.i64
spv.func @bitfield_insert_scalar_greater_bit_width(%base: i16, %insert: i16, %offset: i32, %count: i64) "None" {
  // CHECK: %[[TRUNC_OFFSET:.*]] = llvm.trunc %[[OFFSET]] : !llvm.i32 to !llvm.i16
  // CHECK: %[[TRUNC_COUNT:.*]] = llvm.trunc %[[COUNT]] : !llvm.i64 to !llvm.i16
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(-1 : i16) : !llvm.i16
  // CHECK: %[[T0:.*]] = llvm.shl %[[MINUS_ONE]], %[[TRUNC_COUNT]] : !llvm.i16
  // CHECK: %[[T1:.*]] = llvm.xor %[[T0]], %[[MINUS_ONE]] : !llvm.i16
  // CHECK: %[[T2:.*]] = llvm.shl %[[T1]], %[[TRUNC_OFFSET]] : !llvm.i16
  // CHECK: %[[MASK:.*]] = llvm.xor %[[T2]], %[[MINUS_ONE]] : !llvm.i16
  // CHECK: %[[NEW_BASE:.*]] = llvm.and %[[BASE]], %[[MASK]] : !llvm.i16
  // CHECK: %[[SHIFTED_INSERT:.*]] = llvm.shl %[[INSERT]], %[[TRUNC_OFFSET]] : !llvm.i16
  // CHECK: llvm.or %[[NEW_BASE]], %[[SHIFTED_INSERT]] : !llvm.i16
  %0 = spv.BitFieldInsert %base, %insert, %offset, %count : i16, i32, i64
  spv.Return
}

// CHECK-LABEL: @bitfield_insert_vector
//  CHECK-SAME: %[[BASE:.*]]: !llvm.vec<2 x i32>, %[[INSERT:.*]]: !llvm.vec<2 x i32>, %[[OFFSET:.*]]: !llvm.i32, %[[COUNT:.*]]: !llvm.i32
spv.func @bitfield_insert_vector(%base: vector<2xi32>, %insert: vector<2xi32>, %offset: i32, %count: i32) "None" {
  // CHECK: %[[OFFSET_V0:.*]] = llvm.mlir.undef : !llvm.vec<2 x i32>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
  // CHECK: %[[OFFSET_V1:.*]] = llvm.insertelement %[[OFFSET]], %[[OFFSET_V0]][%[[ZERO]] : !llvm.i32] : !llvm.vec<2 x i32>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
  // CHECK: %[[OFFSET_V2:.*]] = llvm.insertelement  %[[OFFSET]], %[[OFFSET_V1]][%[[ONE]] : !llvm.i32] : !llvm.vec<2 x i32>
  // CHECK: %[[COUNT_V0:.*]] = llvm.mlir.undef : !llvm.vec<2 x i32>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
  // CHECK: %[[COUNT_V1:.*]] = llvm.insertelement %[[COUNT]], %[[COUNT_V0]][%[[ZERO]] : !llvm.i32] : !llvm.vec<2 x i32>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
  // CHECK: %[[COUNT_V2:.*]] = llvm.insertelement %[[COUNT]], %[[COUNT_V1]][%[[ONE]] : !llvm.i32] : !llvm.vec<2 x i32>
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(dense<-1> : vector<2xi32>) : !llvm.vec<2 x i32>
  // CHECK: %[[T0:.*]] = llvm.shl %[[MINUS_ONE]], %[[COUNT_V2]] : !llvm.vec<2 x i32>
  // CHECK: %[[T1:.*]] = llvm.xor %[[T0]], %[[MINUS_ONE]] : !llvm.vec<2 x i32>
  // CHECK: %[[T2:.*]] = llvm.shl %[[T1]], %[[OFFSET_V2]] : !llvm.vec<2 x i32>
  // CHECK: %[[MASK:.*]] = llvm.xor %[[T2]], %[[MINUS_ONE]] : !llvm.vec<2 x i32>
  // CHECK: %[[NEW_BASE:.*]] = llvm.and %[[BASE]], %[[MASK]] : !llvm.vec<2 x i32>
  // CHECK: %[[SHIFTED_INSERT:.*]] = llvm.shl %[[INSERT]], %[[OFFSET_V2]] : !llvm.vec<2 x i32>
  // CHECK: llvm.or %[[NEW_BASE]], %[[SHIFTED_INSERT]] : !llvm.vec<2 x i32>
  %0 = spv.BitFieldInsert %base, %insert, %offset, %count : vector<2xi32>, i32, i32
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.BitFieldSExtract
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitfield_sextract_scalar_same_bit_width
//  CHECK-SAME: %[[BASE:.*]]: !llvm.i64, %[[OFFSET:.*]]: !llvm.i64, %[[COUNT:.*]]: !llvm.i64
spv.func @bitfield_sextract_scalar_same_bit_width(%base: i64, %offset: i64, %count: i64) "None" {
  // CHECK: %[[SIZE:.]] = llvm.mlir.constant(64 : i64) : !llvm.i64
  // CHECK: %[[T0:.*]] = llvm.add %[[COUNT]], %[[OFFSET]] : !llvm.i64
  // CHECK: %[[T1:.*]] = llvm.sub %[[SIZE]], %[[T0]] : !llvm.i64
  // CHECK: %[[SHIFTED_LEFT:.*]] = llvm.shl %[[BASE]], %[[T1]] : !llvm.i64
  // CHECK: %[[T2:.*]] = llvm.add %[[OFFSET]], %[[T1]] : !llvm.i64
  // CHECK: llvm.ashr %[[SHIFTED_LEFT]], %[[T2]] : !llvm.i64
  %0 = spv.BitFieldSExtract %base, %offset, %count : i64, i64, i64
  spv.Return
}

// CHECK-LABEL: @bitfield_sextract_scalar_smaller_bit_width
//  CHECK-SAME: %[[BASE:.*]]: !llvm.i32, %[[OFFSET:.*]]: !llvm.i8, %[[COUNT:.*]]: !llvm.i8
spv.func @bitfield_sextract_scalar_smaller_bit_width(%base: i32, %offset: i8, %count: i8) "None" {
  // CHECK: %[[EXT_OFFSET:.*]] = llvm.zext %[[OFFSET]] : !llvm.i8 to !llvm.i32
  // CHECK: %[[EXT_COUNT:.*]] = llvm.zext %[[COUNT]] : !llvm.i8 to !llvm.i32
  // CHECK: %[[SIZE:.]] = llvm.mlir.constant(32 : i32) : !llvm.i32
  // CHECK: %[[T0:.*]] = llvm.add %[[EXT_COUNT]], %[[EXT_OFFSET]] : !llvm.i32
  // CHECK: %[[T1:.*]] = llvm.sub %[[SIZE]], %[[T0]] : !llvm.i32
  // CHECK: %[[SHIFTED_LEFT:.*]] = llvm.shl %[[BASE]], %[[T1]] : !llvm.i32
  // CHECK: %[[T2:.*]] = llvm.add %[[EXT_OFFSET]], %[[T1]] : !llvm.i32
  // CHECK: llvm.ashr %[[SHIFTED_LEFT]], %[[T2]] : !llvm.i32
  %0 = spv.BitFieldSExtract %base, %offset, %count : i32, i8, i8
  spv.Return
}

// CHECK-LABEL: @bitfield_sextract_scalar_greater_bit_width
//  CHECK-SAME: %[[BASE:.*]]: !llvm.i32, %[[OFFSET:.*]]: !llvm.i64, %[[COUNT:.*]]: !llvm.i64
spv.func @bitfield_sextract_scalar_greater_bit_width(%base: i32, %offset: i64, %count: i64) "None" {
  // CHECK: %[[TRUNC_OFFSET:.*]] = llvm.trunc %[[OFFSET]] : !llvm.i64 to !llvm.i32
  // CHECK: %[[TRUNC_COUNT:.*]] = llvm.trunc %[[COUNT]] : !llvm.i64 to !llvm.i32
  // CHECK: %[[SIZE:.]] = llvm.mlir.constant(32 : i32) : !llvm.i32
  // CHECK: %[[T0:.*]] = llvm.add %[[TRUNC_COUNT]], %[[TRUNC_OFFSET]] : !llvm.i32
  // CHECK: %[[T1:.*]] = llvm.sub %[[SIZE]], %[[T0]] : !llvm.i32
  // CHECK: %[[SHIFTED_LEFT:.*]] = llvm.shl %[[BASE]], %[[T1]] : !llvm.i32
  // CHECK: %[[T2:.*]] = llvm.add %[[TRUNC_OFFSET]], %[[T1]] : !llvm.i32
  // CHECK: llvm.ashr %[[SHIFTED_LEFT]], %[[T2]] : !llvm.i32
  %0 = spv.BitFieldSExtract %base, %offset, %count : i32, i64, i64
  spv.Return
}

// CHECK-LABEL: @bitfield_sextract_vector
//  CHECK-SAME: %[[BASE:.*]]: !llvm.vec<2 x i32>, %[[OFFSET:.*]]: !llvm.i32, %[[COUNT:.*]]: !llvm.i32
spv.func @bitfield_sextract_vector(%base: vector<2xi32>, %offset: i32, %count: i32) "None" {
  // CHECK: %[[OFFSET_V0:.*]] = llvm.mlir.undef : !llvm.vec<2 x i32>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
  // CHECK: %[[OFFSET_V1:.*]] = llvm.insertelement %[[OFFSET]], %[[OFFSET_V0]][%[[ZERO]] : !llvm.i32] : !llvm.vec<2 x i32>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
  // CHECK: %[[OFFSET_V2:.*]] = llvm.insertelement  %[[OFFSET]], %[[OFFSET_V1]][%[[ONE]] : !llvm.i32] : !llvm.vec<2 x i32>
  // CHECK: %[[COUNT_V0:.*]] = llvm.mlir.undef : !llvm.vec<2 x i32>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
  // CHECK: %[[COUNT_V1:.*]] = llvm.insertelement %[[COUNT]], %[[COUNT_V0]][%[[ZERO]] : !llvm.i32] : !llvm.vec<2 x i32>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
  // CHECK: %[[COUNT_V2:.*]] = llvm.insertelement %[[COUNT]], %[[COUNT_V1]][%[[ONE]] : !llvm.i32] : !llvm.vec<2 x i32>
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(dense<32> : vector<2xi32>) : !llvm.vec<2 x i32>
  // CHECK: %[[T0:.*]] = llvm.add %[[COUNT_V2]], %[[OFFSET_V2]] : !llvm.vec<2 x i32>
  // CHECK: %[[T1:.*]] = llvm.sub %[[SIZE]], %[[T0]] : !llvm.vec<2 x i32>
  // CHECK: %[[SHIFTED_LEFT:.*]] = llvm.shl %[[BASE]], %[[T1]] : !llvm.vec<2 x i32>
  // CHECK: %[[T2:.*]] = llvm.add %[[OFFSET_V2]], %[[T1]] : !llvm.vec<2 x i32>
  // CHECK: llvm.ashr %[[SHIFTED_LEFT]], %[[T2]] : !llvm.vec<2 x i32>
  %0 = spv.BitFieldSExtract %base, %offset, %count : vector<2xi32>, i32, i32
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.BitFieldUExtract
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitfield_uextract_scalar_same_bit_width
//  CHECK-SAME: %[[BASE:.*]]: !llvm.i32, %[[OFFSET:.*]]: !llvm.i32, %[[COUNT:.*]]: !llvm.i32
spv.func @bitfield_uextract_scalar_same_bit_width(%base: i32, %offset: i32, %count: i32) "None" {
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(-1 : i32) : !llvm.i32
  // CHECK: %[[T0:.*]] = llvm.shl %[[MINUS_ONE]], %[[COUNT]] : !llvm.i32
  // CHECK: %[[MASK:.*]] = llvm.xor %[[T0]], %[[MINUS_ONE]] : !llvm.i32
  // CHECK: %[[SHIFTED_BASE:.*]] = llvm.lshr %[[BASE]], %[[OFFSET]] : !llvm.i32
  // CHECK: llvm.and %[[SHIFTED_BASE]], %[[MASK]] : !llvm.i32
  %0 = spv.BitFieldUExtract %base, %offset, %count : i32, i32, i32
  spv.Return
}

// CHECK-LABEL: @bitfield_uextract_scalar_smaller_bit_width
//  CHECK-SAME: %[[BASE:.*]]: !llvm.i32, %[[OFFSET:.*]]: !llvm.i16, %[[COUNT:.*]]: !llvm.i8
spv.func @bitfield_uextract_scalar_smaller_bit_width(%base: i32, %offset: i16, %count: i8) "None" {
  // CHECK: %[[EXT_OFFSET:.*]] = llvm.zext %[[OFFSET]] : !llvm.i16 to !llvm.i32
  // CHECK: %[[EXT_COUNT:.*]] = llvm.zext %[[COUNT]] : !llvm.i8 to !llvm.i32
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(-1 : i32) : !llvm.i32
  // CHECK: %[[T0:.*]] = llvm.shl %[[MINUS_ONE]], %[[EXT_COUNT]] : !llvm.i32
  // CHECK: %[[MASK:.*]] = llvm.xor %[[T0]], %[[MINUS_ONE]] : !llvm.i32
  // CHECK: %[[SHIFTED_BASE:.*]] = llvm.lshr %[[BASE]], %[[EXT_OFFSET]] : !llvm.i32
  // CHECK: llvm.and %[[SHIFTED_BASE]], %[[MASK]] : !llvm.i32
  %0 = spv.BitFieldUExtract %base, %offset, %count : i32, i16, i8
  spv.Return
}

// CHECK-LABEL: @bitfield_uextract_scalar_greater_bit_width
//  CHECK-SAME: %[[BASE:.*]]: !llvm.i8, %[[OFFSET:.*]]: !llvm.i16, %[[COUNT:.*]]: !llvm.i8
spv.func @bitfield_uextract_scalar_greater_bit_width(%base: i8, %offset: i16, %count: i8) "None" {
  // CHECK: %[[TRUNC_OFFSET:.*]] = llvm.trunc %[[OFFSET]] : !llvm.i16 to !llvm.i8
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(-1 : i8) : !llvm.i8
  // CHECK: %[[T0:.*]] = llvm.shl %[[MINUS_ONE]], %[[COUNT]] : !llvm.i8
  // CHECK: %[[MASK:.*]] = llvm.xor %[[T0]], %[[MINUS_ONE]] : !llvm.i8
  // CHECK: %[[SHIFTED_BASE:.*]] = llvm.lshr %[[BASE]], %[[TRUNC_OFFSET]] : !llvm.i8
  // CHECK: llvm.and %[[SHIFTED_BASE]], %[[MASK]] : !llvm.i8
  %0 = spv.BitFieldUExtract %base, %offset, %count : i8, i16, i8
  spv.Return
}

// CHECK-LABEL: @bitfield_uextract_vector
//  CHECK-SAME: %[[BASE:.*]]: !llvm.vec<2 x i32>, %[[OFFSET:.*]]: !llvm.i32, %[[COUNT:.*]]: !llvm.i32
spv.func @bitfield_uextract_vector(%base: vector<2xi32>, %offset: i32, %count: i32) "None" {
  // CHECK: %[[OFFSET_V0:.*]] = llvm.mlir.undef : !llvm.vec<2 x i32>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
  // CHECK: %[[OFFSET_V1:.*]] = llvm.insertelement %[[OFFSET]], %[[OFFSET_V0]][%[[ZERO]] : !llvm.i32] : !llvm.vec<2 x i32>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
  // CHECK: %[[OFFSET_V2:.*]] = llvm.insertelement  %[[OFFSET]], %[[OFFSET_V1]][%[[ONE]] : !llvm.i32] : !llvm.vec<2 x i32>
  // CHECK: %[[COUNT_V0:.*]] = llvm.mlir.undef : !llvm.vec<2 x i32>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
  // CHECK: %[[COUNT_V1:.*]] = llvm.insertelement %[[COUNT]], %[[COUNT_V0]][%[[ZERO]] : !llvm.i32] : !llvm.vec<2 x i32>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
  // CHECK: %[[COUNT_V2:.*]] = llvm.insertelement %[[COUNT]], %[[COUNT_V1]][%[[ONE]] : !llvm.i32] : !llvm.vec<2 x i32>
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(dense<-1> : vector<2xi32>) : !llvm.vec<2 x i32>
  // CHECK: %[[T0:.*]] = llvm.shl %[[MINUS_ONE]], %[[COUNT_V2]] : !llvm.vec<2 x i32>
  // CHECK: %[[MASK:.*]] = llvm.xor %[[T0]], %[[MINUS_ONE]] : !llvm.vec<2 x i32>
  // CHECK: %[[SHIFTED_BASE:.*]] = llvm.lshr %[[BASE]], %[[OFFSET_V2]] : !llvm.vec<2 x i32>
  // CHECK: llvm.and %[[SHIFTED_BASE]], %[[MASK]] : !llvm.vec<2 x i32>
  %0 = spv.BitFieldUExtract %base, %offset, %count : vector<2xi32>, i32, i32
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.BitwiseAnd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitwise_and_scalar
spv.func @bitwise_and_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.and %{{.*}}, %{{.*}} : !llvm.i32
  %0 = spv.BitwiseAnd %arg0, %arg1 : i32
  spv.Return
}

// CHECK-LABEL: @bitwise_and_vector
spv.func @bitwise_and_vector(%arg0: vector<4xi64>, %arg1: vector<4xi64>) "None" {
  // CHECK: llvm.and %{{.*}}, %{{.*}} : !llvm.vec<4 x i64>
  %0 = spv.BitwiseAnd %arg0, %arg1 : vector<4xi64>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.BitwiseOr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitwise_or_scalar
spv.func @bitwise_or_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.or %{{.*}}, %{{.*}} : !llvm.i64
  %0 = spv.BitwiseOr %arg0, %arg1 : i64
  spv.Return
}

// CHECK-LABEL: @bitwise_or_vector
spv.func @bitwise_or_vector(%arg0: vector<3xi8>, %arg1: vector<3xi8>) "None" {
  // CHECK: llvm.or %{{.*}}, %{{.*}} : !llvm.vec<3 x i8>
  %0 = spv.BitwiseOr %arg0, %arg1 : vector<3xi8>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.BitwiseXor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitwise_xor_scalar
spv.func @bitwise_xor_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.xor %{{.*}}, %{{.*}} : !llvm.i32
  %0 = spv.BitwiseXor %arg0, %arg1 : i32
  spv.Return
}

// CHECK-LABEL: @bitwise_xor_vector
spv.func @bitwise_xor_vector(%arg0: vector<2xi16>, %arg1: vector<2xi16>) "None" {
  // CHECK: llvm.xor %{{.*}}, %{{.*}} : !llvm.vec<2 x i16>
  %0 = spv.BitwiseXor %arg0, %arg1 : vector<2xi16>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.Not
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @not_scalar
spv.func @not_scalar(%arg0: i32) "None" {
  // CHECK: %[[CONST:.*]] = llvm.mlir.constant(-1 : i32) : !llvm.i32
  // CHECK: llvm.xor %{{.*}}, %[[CONST]] : !llvm.i32
  %0 = spv.Not %arg0 : i32
  spv.Return
}

// CHECK-LABEL: @not_vector
spv.func @not_vector(%arg0: vector<2xi16>) "None" {
  // CHECK: %[[CONST:.*]] = llvm.mlir.constant(dense<-1> : vector<2xi16>) : !llvm.vec<2 x i16>
  // CHECK: llvm.xor %{{.*}}, %[[CONST]] : !llvm.vec<2 x i16>
  %0 = spv.Not %arg0 : vector<2xi16>
  spv.Return
}
