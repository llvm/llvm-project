// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s
// RUN: mlir-opt -convert-spirv-to-llvm='allow-pattern-rollback=0' %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.BitCount
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitcount_scalar
spirv.func @bitcount_scalar(%arg0: i16) "None" {
  // CHECK: llvm.intr.ctpop(%{{.*}}) : (i16) -> i16
  %0 = spirv.BitCount %arg0: i16
  spirv.Return
}

// CHECK-LABEL: @bitcount_vector
spirv.func @bitcount_vector(%arg0: vector<3xi32>) "None" {
  // CHECK: llvm.intr.ctpop(%{{.*}}) : (vector<3xi32>) -> vector<3xi32>
  %0 = spirv.BitCount %arg0: vector<3xi32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.BitReverse
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitreverse_scalar
spirv.func @bitreverse_scalar(%arg0: i64) "None" {
  // CHECK: llvm.intr.bitreverse(%{{.*}}) : (i64) -> i64
  %0 = spirv.BitReverse %arg0: i64
  spirv.Return
}

// CHECK-LABEL: @bitreverse_vector
spirv.func @bitreverse_vector(%arg0: vector<4xi32>) "None" {
  // CHECK: llvm.intr.bitreverse(%{{.*}}) : (vector<4xi32>) -> vector<4xi32>
  %0 = spirv.BitReverse %arg0: vector<4xi32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.BitFieldInsert
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitfield_insert_scalar_same_bit_width
//  CHECK-SAME: %[[BASE:.*]]: i32, %[[INSERT:.*]]: i32, %[[OFFSET:.*]]: i32, %[[COUNT:.*]]: i32
spirv.func @bitfield_insert_scalar_same_bit_width(%base: i32, %insert: i32, %offset: i32, %count: i32) "None" {
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK: %[[T0:.*]] = llvm.sub %[[SIZE]], %[[COUNT]] : i32
  // CHECK: %[[LOW_MASK:.*]] = llvm.lshr %[[MINUS_ONE]], %[[T0]] : i32
  // CHECK: %[[FIELD_MASK:.*]] = llvm.shl %[[LOW_MASK]], %[[OFFSET]] : i32
  // CHECK: %[[BASE_MASK:.*]] = llvm.xor %[[FIELD_MASK]], %[[MINUS_ONE]] : i32
  // CHECK: %[[NEW_BASE:.*]] = llvm.and %[[BASE]], %[[BASE_MASK]] : i32
  // CHECK: %[[SHIFTED_INSERT:.*]] = llvm.shl %[[INSERT]], %[[OFFSET]] : i32
  // CHECK: %[[NEW_INSERT:.*]] = llvm.and %[[SHIFTED_INSERT]], %[[FIELD_MASK]] : i32
  // CHECK: %[[RESULT:.*]] = llvm.or %[[NEW_BASE]], %[[NEW_INSERT]] : i32
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[COUNT_IS_ZERO:.*]] = llvm.icmp "eq" %[[COUNT]], %[[C0]] : i32
  // CHECK: llvm.select %[[COUNT_IS_ZERO]], %[[BASE]], %[[RESULT]] : i1, i32
  %0 = spirv.BitFieldInsert %base, %insert, %offset, %count : i32, i32, i32
  spirv.Return
}

// CHECK-LABEL: @bitfield_insert_scalar_smaller_bit_width
//  CHECK-SAME: %[[BASE:.*]]: i64, %[[INSERT:.*]]: i64, %[[OFFSET:.*]]: i8, %[[COUNT:.*]]: i8
spirv.func @bitfield_insert_scalar_smaller_bit_width(%base: i64, %insert: i64, %offset: i8, %count: i8) "None" {
  // CHECK: %[[EXT_OFFSET:.*]] = llvm.zext %[[OFFSET]] : i8 to i64
  // CHECK: %[[EXT_COUNT:.*]] = llvm.zext %[[COUNT]] : i8 to i64
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(-1 : i64) : i64
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(64 : i64) : i64
  // CHECK: %[[T0:.*]] = llvm.sub %[[SIZE]], %[[EXT_COUNT]] : i64
  // CHECK: %[[LOW_MASK:.*]] = llvm.lshr %[[MINUS_ONE]], %[[T0]] : i64
  // CHECK: %[[FIELD_MASK:.*]] = llvm.shl %[[LOW_MASK]], %[[EXT_OFFSET]] : i64
  // CHECK: %[[BASE_MASK:.*]] = llvm.xor %[[FIELD_MASK]], %[[MINUS_ONE]] : i64
  // CHECK: %[[NEW_BASE:.*]] = llvm.and %[[BASE]], %[[BASE_MASK]] : i64
  // CHECK: %[[SHIFTED_INSERT:.*]] = llvm.shl %[[INSERT]], %[[EXT_OFFSET]] : i64
  // CHECK: %[[NEW_INSERT:.*]] = llvm.and %[[SHIFTED_INSERT]], %[[FIELD_MASK]] : i64
  // CHECK: %[[RESULT:.*]] = llvm.or %[[NEW_BASE]], %[[NEW_INSERT]] : i64
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[COUNT_IS_ZERO:.*]] = llvm.icmp "eq" %[[EXT_COUNT]], %[[C0]] : i64
  // CHECK: llvm.select %[[COUNT_IS_ZERO]], %[[BASE]], %[[RESULT]] : i1, i64
  %0 = spirv.BitFieldInsert %base, %insert, %offset, %count : i64, i8, i8
  spirv.Return
}

// CHECK-LABEL: @bitfield_insert_scalar_greater_bit_width
//  CHECK-SAME: %[[BASE:.*]]: i16, %[[INSERT:.*]]: i16, %[[OFFSET:.*]]: i32, %[[COUNT:.*]]: i64
spirv.func @bitfield_insert_scalar_greater_bit_width(%base: i16, %insert: i16, %offset: i32, %count: i64) "None" {
  // CHECK: %[[TRUNC_OFFSET:.*]] = llvm.trunc %[[OFFSET]] : i32 to i16
  // CHECK: %[[TRUNC_COUNT:.*]] = llvm.trunc %[[COUNT]] : i64 to i16
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(-1 : i16) : i16
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(16 : i16) : i16
  // CHECK: %[[T0:.*]] = llvm.sub %[[SIZE]], %[[TRUNC_COUNT]] : i16
  // CHECK: %[[LOW_MASK:.*]] = llvm.lshr %[[MINUS_ONE]], %[[T0]] : i16
  // CHECK: %[[FIELD_MASK:.*]] = llvm.shl %[[LOW_MASK]], %[[TRUNC_OFFSET]] : i16
  // CHECK: %[[BASE_MASK:.*]] = llvm.xor %[[FIELD_MASK]], %[[MINUS_ONE]] : i16
  // CHECK: %[[NEW_BASE:.*]] = llvm.and %[[BASE]], %[[BASE_MASK]] : i16
  // CHECK: %[[SHIFTED_INSERT:.*]] = llvm.shl %[[INSERT]], %[[TRUNC_OFFSET]] : i16
  // CHECK: %[[NEW_INSERT:.*]] = llvm.and %[[SHIFTED_INSERT]], %[[FIELD_MASK]] : i16
  // CHECK: %[[RESULT:.*]] = llvm.or %[[NEW_BASE]], %[[NEW_INSERT]] : i16
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i16) : i16
  // CHECK: %[[COUNT_IS_ZERO:.*]] = llvm.icmp "eq" %[[TRUNC_COUNT]], %[[C0]] : i16
  // CHECK: llvm.select %[[COUNT_IS_ZERO]], %[[BASE]], %[[RESULT]] : i1, i16
  %0 = spirv.BitFieldInsert %base, %insert, %offset, %count : i16, i32, i64
  spirv.Return
}

// CHECK-LABEL: @bitfield_insert_vector
//  CHECK-SAME: %[[BASE:.*]]: vector<2xi32>, %[[INSERT:.*]]: vector<2xi32>, %[[OFFSET:.*]]: i32, %[[COUNT:.*]]: i32
spirv.func @bitfield_insert_vector(%base: vector<2xi32>, %insert: vector<2xi32>, %offset: i32, %count: i32) "None" {
  // CHECK: %[[OFFSET_V0:.*]] = llvm.mlir.poison : vector<2xi32>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[OFFSET_V1:.*]] = llvm.insertelement %[[OFFSET]], %[[OFFSET_V0]][%[[ZERO]] : i32] : vector<2xi32>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[OFFSET_V2:.*]] = llvm.insertelement  %[[OFFSET]], %[[OFFSET_V1]][%[[ONE]] : i32] : vector<2xi32>
  // CHECK: %[[COUNT_V0:.*]] = llvm.mlir.poison : vector<2xi32>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[COUNT_V1:.*]] = llvm.insertelement %[[COUNT]], %[[COUNT_V0]][%[[ZERO]] : i32] : vector<2xi32>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[COUNT_V2:.*]] = llvm.insertelement %[[COUNT]], %[[COUNT_V1]][%[[ONE]] : i32] : vector<2xi32>
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(dense<-1> : vector<2xi32>) : vector<2xi32>
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(dense<32> : vector<2xi32>) : vector<2xi32>
  // CHECK: %[[T0:.*]] = llvm.sub %[[SIZE]], %[[COUNT_V2]] : vector<2xi32>
  // CHECK: %[[LOW_MASK:.*]] = llvm.lshr %[[MINUS_ONE]], %[[T0]] : vector<2xi32>
  // CHECK: %[[FIELD_MASK:.*]] = llvm.shl %[[LOW_MASK]], %[[OFFSET_V2]] : vector<2xi32>
  // CHECK: %[[BASE_MASK:.*]] = llvm.xor %[[FIELD_MASK]], %[[MINUS_ONE]] : vector<2xi32>
  // CHECK: %[[NEW_BASE:.*]] = llvm.and %[[BASE]], %[[BASE_MASK]] : vector<2xi32>
  // CHECK: %[[SHIFTED_INSERT:.*]] = llvm.shl %[[INSERT]], %[[OFFSET_V2]] : vector<2xi32>
  // CHECK: %[[NEW_INSERT:.*]] = llvm.and %[[SHIFTED_INSERT]], %[[FIELD_MASK]] : vector<2xi32>
  // CHECK: %[[RESULT:.*]] = llvm.or %[[NEW_BASE]], %[[NEW_INSERT]] : vector<2xi32>
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
  // CHECK: %[[COUNT_IS_ZERO:.*]] = llvm.icmp "eq" %[[COUNT_V2]], %[[C0]] : vector<2xi32>
  // CHECK: llvm.select %[[COUNT_IS_ZERO]], %[[BASE]], %[[RESULT]] : vector<2xi1>, vector<2xi32>
  %0 = spirv.BitFieldInsert %base, %insert, %offset, %count : vector<2xi32>, i32, i32
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.BitFieldSExtract
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitfield_sextract_scalar_same_bit_width
//  CHECK-SAME: %[[BASE:.*]]: i64, %[[OFFSET:.*]]: i64, %[[COUNT:.*]]: i64
spirv.func @bitfield_sextract_scalar_same_bit_width(%base: i64, %offset: i64, %count: i64) "None" {
  // CHECK: %[[SIZE:.]] = llvm.mlir.constant(64 : i64) : i64
  // CHECK: %[[T0:.*]] = llvm.add %[[COUNT]], %[[OFFSET]] : i64
  // CHECK: %[[T1:.*]] = llvm.sub %[[SIZE]], %[[T0]] : i64
  // CHECK: %[[SHIFTED_LEFT:.*]] = llvm.shl %[[BASE]], %[[T1]] : i64
  // CHECK: %[[T2:.*]] = llvm.add %[[OFFSET]], %[[T1]] : i64
  // CHECK: llvm.ashr %[[SHIFTED_LEFT]], %[[T2]] : i64
  %0 = spirv.BitFieldSExtract %base, %offset, %count : i64, i64, i64
  spirv.Return
}

// CHECK-LABEL: @bitfield_sextract_scalar_smaller_bit_width
//  CHECK-SAME: %[[BASE:.*]]: i32, %[[OFFSET:.*]]: i8, %[[COUNT:.*]]: i8
spirv.func @bitfield_sextract_scalar_smaller_bit_width(%base: i32, %offset: i8, %count: i8) "None" {
  // CHECK: %[[EXT_OFFSET:.*]] = llvm.zext %[[OFFSET]] : i8 to i32
  // CHECK: %[[EXT_COUNT:.*]] = llvm.zext %[[COUNT]] : i8 to i32
  // CHECK: %[[SIZE:.]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK: %[[T0:.*]] = llvm.add %[[EXT_COUNT]], %[[EXT_OFFSET]] : i32
  // CHECK: %[[T1:.*]] = llvm.sub %[[SIZE]], %[[T0]] : i32
  // CHECK: %[[SHIFTED_LEFT:.*]] = llvm.shl %[[BASE]], %[[T1]] : i32
  // CHECK: %[[T2:.*]] = llvm.add %[[EXT_OFFSET]], %[[T1]] : i32
  // CHECK: llvm.ashr %[[SHIFTED_LEFT]], %[[T2]] : i32
  %0 = spirv.BitFieldSExtract %base, %offset, %count : i32, i8, i8
  spirv.Return
}

// CHECK-LABEL: @bitfield_sextract_scalar_greater_bit_width
//  CHECK-SAME: %[[BASE:.*]]: i32, %[[OFFSET:.*]]: i64, %[[COUNT:.*]]: i64
spirv.func @bitfield_sextract_scalar_greater_bit_width(%base: i32, %offset: i64, %count: i64) "None" {
  // CHECK: %[[TRUNC_OFFSET:.*]] = llvm.trunc %[[OFFSET]] : i64 to i32
  // CHECK: %[[TRUNC_COUNT:.*]] = llvm.trunc %[[COUNT]] : i64 to i32
  // CHECK: %[[SIZE:.]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK: %[[T0:.*]] = llvm.add %[[TRUNC_COUNT]], %[[TRUNC_OFFSET]] : i32
  // CHECK: %[[T1:.*]] = llvm.sub %[[SIZE]], %[[T0]] : i32
  // CHECK: %[[SHIFTED_LEFT:.*]] = llvm.shl %[[BASE]], %[[T1]] : i32
  // CHECK: %[[T2:.*]] = llvm.add %[[TRUNC_OFFSET]], %[[T1]] : i32
  // CHECK: llvm.ashr %[[SHIFTED_LEFT]], %[[T2]] : i32
  %0 = spirv.BitFieldSExtract %base, %offset, %count : i32, i64, i64
  spirv.Return
}

// CHECK-LABEL: @bitfield_sextract_vector
//  CHECK-SAME: %[[BASE:.*]]: vector<2xi32>, %[[OFFSET:.*]]: i32, %[[COUNT:.*]]: i32
spirv.func @bitfield_sextract_vector(%base: vector<2xi32>, %offset: i32, %count: i32) "None" {
  // CHECK: %[[OFFSET_V0:.*]] = llvm.mlir.poison : vector<2xi32>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[OFFSET_V1:.*]] = llvm.insertelement %[[OFFSET]], %[[OFFSET_V0]][%[[ZERO]] : i32] : vector<2xi32>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[OFFSET_V2:.*]] = llvm.insertelement  %[[OFFSET]], %[[OFFSET_V1]][%[[ONE]] : i32] : vector<2xi32>
  // CHECK: %[[COUNT_V0:.*]] = llvm.mlir.poison : vector<2xi32>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[COUNT_V1:.*]] = llvm.insertelement %[[COUNT]], %[[COUNT_V0]][%[[ZERO]] : i32] : vector<2xi32>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[COUNT_V2:.*]] = llvm.insertelement %[[COUNT]], %[[COUNT_V1]][%[[ONE]] : i32] : vector<2xi32>
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(dense<32> : vector<2xi32>) : vector<2xi32>
  // CHECK: %[[T0:.*]] = llvm.add %[[COUNT_V2]], %[[OFFSET_V2]] : vector<2xi32>
  // CHECK: %[[T1:.*]] = llvm.sub %[[SIZE]], %[[T0]] : vector<2xi32>
  // CHECK: %[[SHIFTED_LEFT:.*]] = llvm.shl %[[BASE]], %[[T1]] : vector<2xi32>
  // CHECK: %[[T2:.*]] = llvm.add %[[OFFSET_V2]], %[[T1]] : vector<2xi32>
  // CHECK: llvm.ashr %[[SHIFTED_LEFT]], %[[T2]] : vector<2xi32>
  %0 = spirv.BitFieldSExtract %base, %offset, %count : vector<2xi32>, i32, i32
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.BitFieldUExtract
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitfield_uextract_scalar_same_bit_width
//  CHECK-SAME: %[[BASE:.*]]: i32, %[[OFFSET:.*]]: i32, %[[COUNT:.*]]: i32
spirv.func @bitfield_uextract_scalar_same_bit_width(%base: i32, %offset: i32, %count: i32) "None" {
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: %[[T0:.*]] = llvm.shl %[[MINUS_ONE]], %[[COUNT]] : i32
  // CHECK: %[[MASK:.*]] = llvm.xor %[[T0]], %[[MINUS_ONE]] : i32
  // CHECK: %[[SHIFTED_BASE:.*]] = llvm.lshr %[[BASE]], %[[OFFSET]] : i32
  // CHECK: llvm.and %[[SHIFTED_BASE]], %[[MASK]] : i32
  %0 = spirv.BitFieldUExtract %base, %offset, %count : i32, i32, i32
  spirv.Return
}

// CHECK-LABEL: @bitfield_uextract_scalar_smaller_bit_width
//  CHECK-SAME: %[[BASE:.*]]: i32, %[[OFFSET:.*]]: i16, %[[COUNT:.*]]: i8
spirv.func @bitfield_uextract_scalar_smaller_bit_width(%base: i32, %offset: i16, %count: i8) "None" {
  // CHECK: %[[EXT_OFFSET:.*]] = llvm.zext %[[OFFSET]] : i16 to i32
  // CHECK: %[[EXT_COUNT:.*]] = llvm.zext %[[COUNT]] : i8 to i32
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: %[[T0:.*]] = llvm.shl %[[MINUS_ONE]], %[[EXT_COUNT]] : i32
  // CHECK: %[[MASK:.*]] = llvm.xor %[[T0]], %[[MINUS_ONE]] : i32
  // CHECK: %[[SHIFTED_BASE:.*]] = llvm.lshr %[[BASE]], %[[EXT_OFFSET]] : i32
  // CHECK: llvm.and %[[SHIFTED_BASE]], %[[MASK]] : i32
  %0 = spirv.BitFieldUExtract %base, %offset, %count : i32, i16, i8
  spirv.Return
}

// CHECK-LABEL: @bitfield_uextract_scalar_greater_bit_width
//  CHECK-SAME: %[[BASE:.*]]: i8, %[[OFFSET:.*]]: i16, %[[COUNT:.*]]: i8
spirv.func @bitfield_uextract_scalar_greater_bit_width(%base: i8, %offset: i16, %count: i8) "None" {
  // CHECK: %[[TRUNC_OFFSET:.*]] = llvm.trunc %[[OFFSET]] : i16 to i8
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(-1 : i8) : i8
  // CHECK: %[[T0:.*]] = llvm.shl %[[MINUS_ONE]], %[[COUNT]] : i8
  // CHECK: %[[MASK:.*]] = llvm.xor %[[T0]], %[[MINUS_ONE]] : i8
  // CHECK: %[[SHIFTED_BASE:.*]] = llvm.lshr %[[BASE]], %[[TRUNC_OFFSET]] : i8
  // CHECK: llvm.and %[[SHIFTED_BASE]], %[[MASK]] : i8
  %0 = spirv.BitFieldUExtract %base, %offset, %count : i8, i16, i8
  spirv.Return
}

// CHECK-LABEL: @bitfield_uextract_vector
//  CHECK-SAME: %[[BASE:.*]]: vector<2xi32>, %[[OFFSET:.*]]: i32, %[[COUNT:.*]]: i32
spirv.func @bitfield_uextract_vector(%base: vector<2xi32>, %offset: i32, %count: i32) "None" {
  // CHECK: %[[OFFSET_V0:.*]] = llvm.mlir.poison : vector<2xi32>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[OFFSET_V1:.*]] = llvm.insertelement %[[OFFSET]], %[[OFFSET_V0]][%[[ZERO]] : i32] : vector<2xi32>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[OFFSET_V2:.*]] = llvm.insertelement  %[[OFFSET]], %[[OFFSET_V1]][%[[ONE]] : i32] : vector<2xi32>
  // CHECK: %[[COUNT_V0:.*]] = llvm.mlir.poison : vector<2xi32>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[COUNT_V1:.*]] = llvm.insertelement %[[COUNT]], %[[COUNT_V0]][%[[ZERO]] : i32] : vector<2xi32>
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[COUNT_V2:.*]] = llvm.insertelement %[[COUNT]], %[[COUNT_V1]][%[[ONE]] : i32] : vector<2xi32>
  // CHECK: %[[MINUS_ONE:.*]] = llvm.mlir.constant(dense<-1> : vector<2xi32>) : vector<2xi32>
  // CHECK: %[[T0:.*]] = llvm.shl %[[MINUS_ONE]], %[[COUNT_V2]] : vector<2xi32>
  // CHECK: %[[MASK:.*]] = llvm.xor %[[T0]], %[[MINUS_ONE]] : vector<2xi32>
  // CHECK: %[[SHIFTED_BASE:.*]] = llvm.lshr %[[BASE]], %[[OFFSET_V2]] : vector<2xi32>
  // CHECK: llvm.and %[[SHIFTED_BASE]], %[[MASK]] : vector<2xi32>
  %0 = spirv.BitFieldUExtract %base, %offset, %count : vector<2xi32>, i32, i32
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.BitwiseAnd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitwise_and_scalar
spirv.func @bitwise_and_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.and %{{.*}}, %{{.*}} : i32
  %0 = spirv.BitwiseAnd %arg0, %arg1 : i32
  spirv.Return
}

// CHECK-LABEL: @bitwise_and_vector
spirv.func @bitwise_and_vector(%arg0: vector<4xi64>, %arg1: vector<4xi64>) "None" {
  // CHECK: llvm.and %{{.*}}, %{{.*}} : vector<4xi64>
  %0 = spirv.BitwiseAnd %arg0, %arg1 : vector<4xi64>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.BitwiseOr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitwise_or_scalar
spirv.func @bitwise_or_scalar(%arg0: i64, %arg1: i64) "None" {
  // CHECK: llvm.or %{{.*}}, %{{.*}} : i64
  %0 = spirv.BitwiseOr %arg0, %arg1 : i64
  spirv.Return
}

// CHECK-LABEL: @bitwise_or_vector
spirv.func @bitwise_or_vector(%arg0: vector<3xi8>, %arg1: vector<3xi8>) "None" {
  // CHECK: llvm.or %{{.*}}, %{{.*}} : vector<3xi8>
  %0 = spirv.BitwiseOr %arg0, %arg1 : vector<3xi8>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.BitwiseXor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitwise_xor_scalar
spirv.func @bitwise_xor_scalar(%arg0: i32, %arg1: i32) "None" {
  // CHECK: llvm.xor %{{.*}}, %{{.*}} : i32
  %0 = spirv.BitwiseXor %arg0, %arg1 : i32
  spirv.Return
}

// CHECK-LABEL: @bitwise_xor_vector
spirv.func @bitwise_xor_vector(%arg0: vector<2xi16>, %arg1: vector<2xi16>) "None" {
  // CHECK: llvm.xor %{{.*}}, %{{.*}} : vector<2xi16>
  %0 = spirv.BitwiseXor %arg0, %arg1 : vector<2xi16>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.Not
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @not_scalar
spirv.func @not_scalar(%arg0: i32) "None" {
  // CHECK: %[[CONST:.*]] = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: llvm.xor %{{.*}}, %[[CONST]] : i32
  %0 = spirv.Not %arg0 : i32
  spirv.Return
}

// CHECK-LABEL: @not_vector
spirv.func @not_vector(%arg0: vector<2xi16>) "None" {
  // CHECK: %[[CONST:.*]] = llvm.mlir.constant(dense<-1> : vector<2xi16>) : vector<2xi16>
  // CHECK: llvm.xor %{{.*}}, %[[CONST]] : vector<2xi16>
  %0 = spirv.Not %arg0 : vector<2xi16>
  spirv.Return
}
