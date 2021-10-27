// RUN: mlir-opt -split-input-file -convert-memref-to-spirv="bool-num-bits=8" %s -o - | FileCheck %s

// Check that with proper compute and storage extensions, we don't need to
// perform special tricks.

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0,
      [
        Shader, Int8, Int16, Int64, Float16, Float64,
        StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16,
        StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8
      ],
      [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class]>, {}>
} {

// CHECK-LABEL: @load_store_zero_rank_float
func @load_store_zero_rank_float(%arg0: memref<f32>, %arg1: memref<f32>) {
  //      CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast {{.+}} : memref<f32> to !spv.ptr<!spv.struct<(!spv.array<1 x f32, stride=4> [0])>, StorageBuffer>
  //      CHECK: [[ARG1:%.*]] = builtin.unrealized_conversion_cast {{.+}} : memref<f32> to !spv.ptr<!spv.struct<(!spv.array<1 x f32, stride=4> [0])>, StorageBuffer>
  //      CHECK: [[ZERO1:%.*]] = spv.Constant 0 : i32
  //      CHECK: spv.AccessChain [[ARG0]][
  // CHECK-SAME: [[ZERO1]], [[ZERO1]]
  // CHECK-SAME: ] :
  //      CHECK: spv.Load "StorageBuffer" %{{.*}} : f32
  %0 = memref.load %arg0[] : memref<f32>
  //      CHECK: [[ZERO2:%.*]] = spv.Constant 0 : i32
  //      CHECK: spv.AccessChain [[ARG1]][
  // CHECK-SAME: [[ZERO2]], [[ZERO2]]
  // CHECK-SAME: ] :
  //      CHECK: spv.Store "StorageBuffer" %{{.*}} : f32
  memref.store %0, %arg1[] : memref<f32>
  return
}

// CHECK-LABEL: @load_store_zero_rank_int
func @load_store_zero_rank_int(%arg0: memref<i32>, %arg1: memref<i32>) {
  //      CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast {{.+}} : memref<i32> to !spv.ptr<!spv.struct<(!spv.array<1 x i32, stride=4> [0])>, StorageBuffer>
  //      CHECK: [[ARG1:%.*]] = builtin.unrealized_conversion_cast {{.+}} : memref<i32> to !spv.ptr<!spv.struct<(!spv.array<1 x i32, stride=4> [0])>, StorageBuffer>
  //      CHECK: [[ZERO1:%.*]] = spv.Constant 0 : i32
  //      CHECK: spv.AccessChain [[ARG0]][
  // CHECK-SAME: [[ZERO1]], [[ZERO1]]
  // CHECK-SAME: ] :
  //      CHECK: spv.Load "StorageBuffer" %{{.*}} : i32
  %0 = memref.load %arg0[] : memref<i32>
  //      CHECK: [[ZERO2:%.*]] = spv.Constant 0 : i32
  //      CHECK: spv.AccessChain [[ARG1]][
  // CHECK-SAME: [[ZERO2]], [[ZERO2]]
  // CHECK-SAME: ] :
  //      CHECK: spv.Store "StorageBuffer" %{{.*}} : i32
  memref.store %0, %arg1[] : memref<i32>
  return
}

// CHECK-LABEL: func @load_store_unknown_dim
func @load_store_unknown_dim(%i: index, %source: memref<?xi32>, %dest: memref<?xi32>) {
  // CHECK: %[[SRC:.+]] = builtin.unrealized_conversion_cast {{.+}} : memref<?xi32> to !spv.ptr<!spv.struct<(!spv.rtarray<i32, stride=4> [0])>, StorageBuffer>
  // CHECK: %[[DST:.+]] = builtin.unrealized_conversion_cast {{.+}} : memref<?xi32> to !spv.ptr<!spv.struct<(!spv.rtarray<i32, stride=4> [0])>, StorageBuffer>
  // CHECK: %[[AC0:.+]] = spv.AccessChain %[[SRC]]
  // CHECK: spv.Load "StorageBuffer" %[[AC0]]
  %0 = memref.load %source[%i] : memref<?xi32>
  // CHECK: %[[AC1:.+]] = spv.AccessChain %[[DST]]
  // CHECK: spv.Store "StorageBuffer" %[[AC1]]
  memref.store %0, %dest[%i]: memref<?xi32>
  return
}

// CHECK-LABEL: func @load_i1
//  CHECK-SAME: (%[[SRC:.+]]: memref<4xi1>, %[[IDX:.+]]: index)
func @load_i1(%src: memref<4xi1>, %i : index) -> i1 {
  // CHECK-DAG: %[[SRC_CAST:.+]] = builtin.unrealized_conversion_cast %[[SRC]] : memref<4xi1> to !spv.ptr<!spv.struct<(!spv.array<4 x i8, stride=1> [0])>, StorageBuffer>
  // CHECK-DAG: %[[IDX_CAST:.+]] = builtin.unrealized_conversion_cast %[[IDX]]
  // CHECK: %[[ZERO_0:.+]] = spv.Constant 0 : i32
  // CHECK: %[[ZERO_1:.+]] = spv.Constant 0 : i32
  // CHECK: %[[ONE:.+]] = spv.Constant 1 : i32
  // CHECK: %[[MUL:.+]] = spv.IMul %[[ONE]], %[[IDX_CAST]] : i32
  // CHECK: %[[ADD:.+]] = spv.IAdd %[[ZERO_1]], %[[MUL]] : i32
  // CHECK: %[[ADDR:.+]] = spv.AccessChain %[[SRC_CAST]][%[[ZERO_0]], %[[ADD]]]
  // CHECK: %[[VAL:.+]] = spv.Load "StorageBuffer" %[[ADDR]] : i8
  // CHECK: %[[ONE_I8:.+]] = spv.Constant 1 : i8
  // CHECK: %[[BOOL:.+]] = spv.IEqual %[[VAL]], %[[ONE_I8]] : i8
  %0 = memref.load %src[%i] : memref<4xi1>
  // CHECK: return %[[BOOL]]
  return %0: i1
}

// CHECK-LABEL: func @store_i1
//  CHECK-SAME: %[[DST:.+]]: memref<4xi1>,
//  CHECK-SAME: %[[IDX:.+]]: index
func @store_i1(%dst: memref<4xi1>, %i: index) {
  %true = arith.constant true
  // CHECK-DAG: %[[DST_CAST:.+]] = builtin.unrealized_conversion_cast %[[DST]] : memref<4xi1> to !spv.ptr<!spv.struct<(!spv.array<4 x i8, stride=1> [0])>, StorageBuffer>
  // CHECK-DAG: %[[IDX_CAST:.+]] = builtin.unrealized_conversion_cast %[[IDX]]
  // CHECK: %[[ZERO_0:.+]] = spv.Constant 0 : i32
  // CHECK: %[[ZERO_1:.+]] = spv.Constant 0 : i32
  // CHECK: %[[ONE:.+]] = spv.Constant 1 : i32
  // CHECK: %[[MUL:.+]] = spv.IMul %[[ONE]], %[[IDX_CAST]] : i32
  // CHECK: %[[ADD:.+]] = spv.IAdd %[[ZERO_1]], %[[MUL]] : i32
  // CHECK: %[[ADDR:.+]] = spv.AccessChain %[[DST_CAST]][%[[ZERO_0]], %[[ADD]]]
  // CHECK: %[[ZERO_I8:.+]] = spv.Constant 0 : i8
  // CHECK: %[[ONE_I8:.+]] = spv.Constant 1 : i8
  // CHECK: %[[RES:.+]] = spv.Select %{{.+}}, %[[ONE_I8]], %[[ZERO_I8]] : i1, i8
  // CHECK: spv.Store "StorageBuffer" %[[ADDR]], %[[RES]] : i8
  memref.store %true, %dst[%i]: memref<4xi1>
  return
}

} // end module

// -----

// Check that access chain indices are properly adjusted if non-32-bit types are
// emulated via 32-bit types.
// TODO: Test i64 types.
module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
} {

// CHECK-LABEL: @load_i1
func @load_i1(%arg0: memref<i1>) -> i1 {
  //     CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[FOUR1:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[QUOTIENT:.+]] = spv.SDiv %[[ZERO]], %[[FOUR1]] : i32
  //     CHECK: %[[PTR:.+]] = spv.AccessChain %{{.+}}[%[[ZERO]], %[[QUOTIENT]]]
  //     CHECK: %[[LOAD:.+]] = spv.Load  "StorageBuffer" %[[PTR]]
  //     CHECK: %[[FOUR2:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[EIGHT:.+]] = spv.Constant 8 : i32
  //     CHECK: %[[IDX:.+]] = spv.UMod %[[ZERO]], %[[FOUR2]] : i32
  //     CHECK: %[[BITS:.+]] = spv.IMul %[[IDX]], %[[EIGHT]] : i32
  //     CHECK: %[[VALUE:.+]] = spv.ShiftRightArithmetic %[[LOAD]], %[[BITS]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spv.Constant 255 : i32
  //     CHECK: %[[T1:.+]] = spv.BitwiseAnd %[[VALUE]], %[[MASK]] : i32
  //     CHECK: %[[T2:.+]] = spv.Constant 24 : i32
  //     CHECK: %[[T3:.+]] = spv.ShiftLeftLogical %[[T1]], %[[T2]] : i32, i32
  //     CHECK: %[[T4:.+]] = spv.ShiftRightArithmetic %[[T3]], %[[T2]] : i32, i32
  // Convert to i1 type.
  //     CHECK: %[[ONE:.+]] = spv.Constant 1 : i32
  //     CHECK: %[[RES:.+]]  = spv.IEqual %[[T4]], %[[ONE]] : i32
  //     CHECK: return %[[RES]]
  %0 = memref.load %arg0[] : memref<i1>
  return %0 : i1
}

// CHECK-LABEL: @load_i8
func @load_i8(%arg0: memref<i8>) {
  //     CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[FOUR1:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[QUOTIENT:.+]] = spv.SDiv %[[ZERO]], %[[FOUR1]] : i32
  //     CHECK: %[[PTR:.+]] = spv.AccessChain %{{.+}}[%[[ZERO]], %[[QUOTIENT]]]
  //     CHECK: %[[LOAD:.+]] = spv.Load  "StorageBuffer" %[[PTR]]
  //     CHECK: %[[FOUR2:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[EIGHT:.+]] = spv.Constant 8 : i32
  //     CHECK: %[[IDX:.+]] = spv.UMod %[[ZERO]], %[[FOUR2]] : i32
  //     CHECK: %[[BITS:.+]] = spv.IMul %[[IDX]], %[[EIGHT]] : i32
  //     CHECK: %[[VALUE:.+]] = spv.ShiftRightArithmetic %[[LOAD]], %[[BITS]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spv.Constant 255 : i32
  //     CHECK: %[[T1:.+]] = spv.BitwiseAnd %[[VALUE]], %[[MASK]] : i32
  //     CHECK: %[[T2:.+]] = spv.Constant 24 : i32
  //     CHECK: %[[T3:.+]] = spv.ShiftLeftLogical %[[T1]], %[[T2]] : i32, i32
  //     CHECK: spv.ShiftRightArithmetic %[[T3]], %[[T2]] : i32, i32
  %0 = memref.load %arg0[] : memref<i8>
  return
}

// CHECK-LABEL: @load_i16
//       CHECK: (%[[ARG0:.+]]: {{.*}}, %[[ARG1:.+]]: index)
func @load_i16(%arg0: memref<10xi16>, %index : index) {
  //     CHECK: %[[ARG1_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : index to i32
  //     CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[OFFSET:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[ONE:.+]] = spv.Constant 1 : i32
  //     CHECK: %[[UPDATE:.+]] = spv.IMul %[[ONE]], %[[ARG1_CAST]] : i32
  //     CHECK: %[[FLAT_IDX:.+]] = spv.IAdd %[[OFFSET]], %[[UPDATE]] : i32
  //     CHECK: %[[TWO1:.+]] = spv.Constant 2 : i32
  //     CHECK: %[[QUOTIENT:.+]] = spv.SDiv %[[FLAT_IDX]], %[[TWO1]] : i32
  //     CHECK: %[[PTR:.+]] = spv.AccessChain %{{.+}}[%[[ZERO]], %[[QUOTIENT]]]
  //     CHECK: %[[LOAD:.+]] = spv.Load  "StorageBuffer" %[[PTR]]
  //     CHECK: %[[TWO2:.+]] = spv.Constant 2 : i32
  //     CHECK: %[[SIXTEEN:.+]] = spv.Constant 16 : i32
  //     CHECK: %[[IDX:.+]] = spv.UMod %[[FLAT_IDX]], %[[TWO2]] : i32
  //     CHECK: %[[BITS:.+]] = spv.IMul %[[IDX]], %[[SIXTEEN]] : i32
  //     CHECK: %[[VALUE:.+]] = spv.ShiftRightArithmetic %[[LOAD]], %[[BITS]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spv.Constant 65535 : i32
  //     CHECK: %[[T1:.+]] = spv.BitwiseAnd %[[VALUE]], %[[MASK]] : i32
  //     CHECK: %[[T2:.+]] = spv.Constant 16 : i32
  //     CHECK: %[[T3:.+]] = spv.ShiftLeftLogical %[[T1]], %[[T2]] : i32, i32
  //     CHECK: spv.ShiftRightArithmetic %[[T3]], %[[T2]] : i32, i32
  %0 = memref.load %arg0[%index] : memref<10xi16>
  return
}

// CHECK-LABEL: @load_i32
func @load_i32(%arg0: memref<i32>) {
  // CHECK-NOT: spv.SDiv
  //     CHECK: spv.Load
  // CHECK-NOT: spv.ShiftRightArithmetic
  %0 = memref.load %arg0[] : memref<i32>
  return
}

// CHECK-LABEL: @load_f32
func @load_f32(%arg0: memref<f32>) {
  // CHECK-NOT: spv.SDiv
  //     CHECK: spv.Load
  // CHECK-NOT: spv.ShiftRightArithmetic
  %0 = memref.load %arg0[] : memref<f32>
  return
}

// CHECK-LABEL: @store_i1
//       CHECK: (%[[ARG0:.+]]: {{.*}}, %[[ARG1:.+]]: i1)
func @store_i1(%arg0: memref<i1>, %value: i1) {
  //     CHECK: %[[ARG0_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG0]]
  //     CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[FOUR:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[EIGHT:.+]] = spv.Constant 8 : i32
  //     CHECK: %[[IDX:.+]] = spv.UMod %[[ZERO]], %[[FOUR]] : i32
  //     CHECK: %[[OFFSET:.+]] = spv.IMul %[[IDX]], %[[EIGHT]] : i32
  //     CHECK: %[[MASK1:.+]] = spv.Constant 255 : i32
  //     CHECK: %[[TMP1:.+]] = spv.ShiftLeftLogical %[[MASK1]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spv.Not %[[TMP1]] : i32
  //     CHECK: %[[ZERO1:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[ONE1:.+]] = spv.Constant 1 : i32
  //     CHECK: %[[CASTED_ARG1:.+]] = spv.Select %[[ARG1]], %[[ONE1]], %[[ZERO1]] : i1, i32
  //     CHECK: %[[CLAMPED_VAL:.+]] = spv.BitwiseAnd %[[CASTED_ARG1]], %[[MASK1]] : i32
  //     CHECK: %[[STORE_VAL:.+]] = spv.ShiftLeftLogical %[[CLAMPED_VAL]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[FOUR2:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[ACCESS_IDX:.+]] = spv.SDiv %[[ZERO]], %[[FOUR2]] : i32
  //     CHECK: %[[PTR:.+]] = spv.AccessChain %[[ARG0_CAST]][%[[ZERO]], %[[ACCESS_IDX]]]
  //     CHECK: spv.AtomicAnd "Device" "AcquireRelease" %[[PTR]], %[[MASK]]
  //     CHECK: spv.AtomicOr "Device" "AcquireRelease" %[[PTR]], %[[STORE_VAL]]
  memref.store %value, %arg0[] : memref<i1>
  return
}

// CHECK-LABEL: @store_i8
//       CHECK: (%[[ARG0:.+]]: {{.*}}, %[[ARG1:.+]]: i8)
func @store_i8(%arg0: memref<i8>, %value: i8) {
  //     CHECK-DAG: %[[ARG1_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : i8 to i32
  //     CHECK-DAG: %[[ARG0_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG0]]
  //     CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[FOUR:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[EIGHT:.+]] = spv.Constant 8 : i32
  //     CHECK: %[[IDX:.+]] = spv.UMod %[[ZERO]], %[[FOUR]] : i32
  //     CHECK: %[[OFFSET:.+]] = spv.IMul %[[IDX]], %[[EIGHT]] : i32
  //     CHECK: %[[MASK1:.+]] = spv.Constant 255 : i32
  //     CHECK: %[[TMP1:.+]] = spv.ShiftLeftLogical %[[MASK1]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spv.Not %[[TMP1]] : i32
  //     CHECK: %[[CLAMPED_VAL:.+]] = spv.BitwiseAnd %[[ARG1_CAST]], %[[MASK1]] : i32
  //     CHECK: %[[STORE_VAL:.+]] = spv.ShiftLeftLogical %[[CLAMPED_VAL]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[FOUR2:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[ACCESS_IDX:.+]] = spv.SDiv %[[ZERO]], %[[FOUR2]] : i32
  //     CHECK: %[[PTR:.+]] = spv.AccessChain %[[ARG0_CAST]][%[[ZERO]], %[[ACCESS_IDX]]]
  //     CHECK: spv.AtomicAnd "Device" "AcquireRelease" %[[PTR]], %[[MASK]]
  //     CHECK: spv.AtomicOr "Device" "AcquireRelease" %[[PTR]], %[[STORE_VAL]]
  memref.store %value, %arg0[] : memref<i8>
  return
}

// CHECK-LABEL: @store_i16
//       CHECK: (%[[ARG0:.+]]: memref<10xi16>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: i16)
func @store_i16(%arg0: memref<10xi16>, %index: index, %value: i16) {
  //     CHECK-DAG: %[[ARG2_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG2]] : i16 to i32
  //     CHECK-DAG: %[[ARG0_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG0]]
  //     CHECK-DAG: %[[ARG1_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : index to i32
  //     CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[OFFSET:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[ONE:.+]] = spv.Constant 1 : i32
  //     CHECK: %[[UPDATE:.+]] = spv.IMul %[[ONE]], %[[ARG1_CAST]] : i32
  //     CHECK: %[[FLAT_IDX:.+]] = spv.IAdd %[[OFFSET]], %[[UPDATE]] : i32
  //     CHECK: %[[TWO:.+]] = spv.Constant 2 : i32
  //     CHECK: %[[SIXTEEN:.+]] = spv.Constant 16 : i32
  //     CHECK: %[[IDX:.+]] = spv.UMod %[[FLAT_IDX]], %[[TWO]] : i32
  //     CHECK: %[[OFFSET:.+]] = spv.IMul %[[IDX]], %[[SIXTEEN]] : i32
  //     CHECK: %[[MASK1:.+]] = spv.Constant 65535 : i32
  //     CHECK: %[[TMP1:.+]] = spv.ShiftLeftLogical %[[MASK1]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spv.Not %[[TMP1]] : i32
  //     CHECK: %[[CLAMPED_VAL:.+]] = spv.BitwiseAnd %[[ARG2_CAST]], %[[MASK1]] : i32
  //     CHECK: %[[STORE_VAL:.+]] = spv.ShiftLeftLogical %[[CLAMPED_VAL]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[TWO2:.+]] = spv.Constant 2 : i32
  //     CHECK: %[[ACCESS_IDX:.+]] = spv.SDiv %[[FLAT_IDX]], %[[TWO2]] : i32
  //     CHECK: %[[PTR:.+]] = spv.AccessChain %[[ARG0_CAST]][%[[ZERO]], %[[ACCESS_IDX]]]
  //     CHECK: spv.AtomicAnd "Device" "AcquireRelease" %[[PTR]], %[[MASK]]
  //     CHECK: spv.AtomicOr "Device" "AcquireRelease" %[[PTR]], %[[STORE_VAL]]
  memref.store %value, %arg0[%index] : memref<10xi16>
  return
}

// CHECK-LABEL: @store_i32
func @store_i32(%arg0: memref<i32>, %value: i32) {
  //     CHECK: spv.Store
  // CHECK-NOT: spv.AtomicAnd
  // CHECK-NOT: spv.AtomicOr
  memref.store %value, %arg0[] : memref<i32>
  return
}

// CHECK-LABEL: @store_f32
func @store_f32(%arg0: memref<f32>, %value: f32) {
  //     CHECK: spv.Store
  // CHECK-NOT: spv.AtomicAnd
  // CHECK-NOT: spv.AtomicOr
  memref.store %value, %arg0[] : memref<f32>
  return
}

} // end module

// -----

// Check that access chain indices are properly adjusted if non-16/32-bit types
// are emulated via 32-bit types.
module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Int16, StorageBuffer16BitAccess, Shader],
    [SPV_KHR_storage_buffer_storage_class, SPV_KHR_16bit_storage]>, {}>
} {

// CHECK-LABEL: @load_i8
func @load_i8(%arg0: memref<i8>) {
  //     CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[FOUR1:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[QUOTIENT:.+]] = spv.SDiv %[[ZERO]], %[[FOUR1]] : i32
  //     CHECK: %[[PTR:.+]] = spv.AccessChain %{{.+}}[%[[ZERO]], %[[QUOTIENT]]]
  //     CHECK: %[[LOAD:.+]] = spv.Load  "StorageBuffer" %[[PTR]]
  //     CHECK: %[[FOUR2:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[EIGHT:.+]] = spv.Constant 8 : i32
  //     CHECK: %[[IDX:.+]] = spv.UMod %[[ZERO]], %[[FOUR2]] : i32
  //     CHECK: %[[BITS:.+]] = spv.IMul %[[IDX]], %[[EIGHT]] : i32
  //     CHECK: %[[VALUE:.+]] = spv.ShiftRightArithmetic %[[LOAD]], %[[BITS]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spv.Constant 255 : i32
  //     CHECK: %[[T1:.+]] = spv.BitwiseAnd %[[VALUE]], %[[MASK]] : i32
  //     CHECK: %[[T2:.+]] = spv.Constant 24 : i32
  //     CHECK: %[[T3:.+]] = spv.ShiftLeftLogical %[[T1]], %[[T2]] : i32, i32
  //     CHECK: spv.ShiftRightArithmetic %[[T3]], %[[T2]] : i32, i32
  %0 = memref.load %arg0[] : memref<i8>
  return
}

// CHECK-LABEL: @load_i16
func @load_i16(%arg0: memref<i16>) {
  // CHECK-NOT: spv.SDiv
  //     CHECK: spv.Load
  // CHECK-NOT: spv.ShiftRightArithmetic
  %0 = memref.load %arg0[] : memref<i16>
  return
}

// CHECK-LABEL: @store_i8
//       CHECK: (%[[ARG0:.+]]: {{.*}}, %[[ARG1:.+]]: i8)
func @store_i8(%arg0: memref<i8>, %value: i8) {
  //     CHECK-DAG: %[[ARG1_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : i8 to i32
  //     CHECK-DAG: %[[ARG0_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG0]]
  //     CHECK: %[[ZERO:.+]] = spv.Constant 0 : i32
  //     CHECK: %[[FOUR:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[EIGHT:.+]] = spv.Constant 8 : i32
  //     CHECK: %[[IDX:.+]] = spv.UMod %[[ZERO]], %[[FOUR]] : i32
  //     CHECK: %[[OFFSET:.+]] = spv.IMul %[[IDX]], %[[EIGHT]] : i32
  //     CHECK: %[[MASK1:.+]] = spv.Constant 255 : i32
  //     CHECK: %[[TMP1:.+]] = spv.ShiftLeftLogical %[[MASK1]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spv.Not %[[TMP1]] : i32
  //     CHECK: %[[CLAMPED_VAL:.+]] = spv.BitwiseAnd %[[ARG1_CAST]], %[[MASK1]] : i32
  //     CHECK: %[[STORE_VAL:.+]] = spv.ShiftLeftLogical %[[CLAMPED_VAL]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[FOUR2:.+]] = spv.Constant 4 : i32
  //     CHECK: %[[ACCESS_IDX:.+]] = spv.SDiv %[[ZERO]], %[[FOUR2]] : i32
  //     CHECK: %[[PTR:.+]] = spv.AccessChain %[[ARG0_CAST]][%[[ZERO]], %[[ACCESS_IDX]]]
  //     CHECK: spv.AtomicAnd "Device" "AcquireRelease" %[[PTR]], %[[MASK]]
  //     CHECK: spv.AtomicOr "Device" "AcquireRelease" %[[PTR]], %[[STORE_VAL]]
  memref.store %value, %arg0[] : memref<i8>
  return
}

// CHECK-LABEL: @store_i16
func @store_i16(%arg0: memref<10xi16>, %index: index, %value: i16) {
  //     CHECK: spv.Store
  // CHECK-NOT: spv.AtomicAnd
  // CHECK-NOT: spv.AtomicOr
  memref.store %value, %arg0[%index] : memref<10xi16>
  return
}

} // end module
