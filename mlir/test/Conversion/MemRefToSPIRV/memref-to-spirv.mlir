// RUN: mlir-opt -split-input-file -convert-memref-to-spirv="bool-num-bits=8" -cse %s -o - | FileCheck %s

// Check that with proper compute and storage extensions, we don't need to
// perform special tricks.

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0,
      [
        Shader, Int8, Int16, Int64, Float16, Float64,
        StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16,
        StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8
      ],
      [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @load_store_zero_rank_float
func.func @load_store_zero_rank_float(%arg0: memref<f32, #spirv.storage_class<StorageBuffer>>, %arg1: memref<f32, #spirv.storage_class<StorageBuffer>>) {
  //      CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast {{.+}} : memref<f32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>
  //      CHECK: [[ARG1:%.*]] = builtin.unrealized_conversion_cast {{.+}} : memref<f32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>
  //      CHECK: [[ZERO:%.*]] = spirv.Constant 0 : i32
  //      CHECK: spirv.AccessChain [[ARG0]][
  // CHECK-SAME: [[ZERO]], [[ZERO]]
  // CHECK-SAME: ] :
  //      CHECK: spirv.Load "StorageBuffer" %{{.*}} : f32
  %0 = memref.load %arg0[] : memref<f32, #spirv.storage_class<StorageBuffer>>
  //      CHECK: spirv.AccessChain [[ARG1]][
  // CHECK-SAME: [[ZERO]], [[ZERO]]
  // CHECK-SAME: ] :
  //      CHECK: spirv.Store "StorageBuffer" %{{.*}} : f32
  memref.store %0, %arg1[] : memref<f32, #spirv.storage_class<StorageBuffer>>
  return
}

// CHECK-LABEL: @load_store_zero_rank_int
func.func @load_store_zero_rank_int(%arg0: memref<i32, #spirv.storage_class<StorageBuffer>>, %arg1: memref<i32, #spirv.storage_class<StorageBuffer>>) {
  //      CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast {{.+}} : memref<i32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>, StorageBuffer>
  //      CHECK: [[ARG1:%.*]] = builtin.unrealized_conversion_cast {{.+}} : memref<i32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>, StorageBuffer>
  //      CHECK: [[ZERO:%.*]] = spirv.Constant 0 : i32
  //      CHECK: spirv.AccessChain [[ARG0]][
  // CHECK-SAME: [[ZERO]], [[ZERO]]
  // CHECK-SAME: ] :
  //      CHECK: spirv.Load "StorageBuffer" %{{.*}} : i32
  %0 = memref.load %arg0[] : memref<i32, #spirv.storage_class<StorageBuffer>>
  //      CHECK: spirv.AccessChain [[ARG1]][
  // CHECK-SAME: [[ZERO]], [[ZERO]]
  // CHECK-SAME: ] :
  //      CHECK: spirv.Store "StorageBuffer" %{{.*}} : i32
  memref.store %0, %arg1[] : memref<i32, #spirv.storage_class<StorageBuffer>>
  return
}

// CHECK-LABEL: func @load_store_unknown_dim
func.func @load_store_unknown_dim(%i: index, %source: memref<?xi32, #spirv.storage_class<StorageBuffer>>, %dest: memref<?xi32, #spirv.storage_class<StorageBuffer>>) {
  // CHECK: %[[SRC:.+]] = builtin.unrealized_conversion_cast {{.+}} : memref<?xi32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
  // CHECK: %[[DST:.+]] = builtin.unrealized_conversion_cast {{.+}} : memref<?xi32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
  // CHECK: %[[AC0:.+]] = spirv.AccessChain %[[SRC]]
  // CHECK: spirv.Load "StorageBuffer" %[[AC0]]
  %0 = memref.load %source[%i] : memref<?xi32, #spirv.storage_class<StorageBuffer>>
  // CHECK: %[[AC1:.+]] = spirv.AccessChain %[[DST]]
  // CHECK: spirv.Store "StorageBuffer" %[[AC1]]
  memref.store %0, %dest[%i]: memref<?xi32, #spirv.storage_class<StorageBuffer>>
  return
}

// CHECK-LABEL: func @load_i1
//  CHECK-SAME: (%[[SRC:.+]]: memref<4xi1, #spirv.storage_class<StorageBuffer>>, %[[IDX:.+]]: index)
func.func @load_i1(%src: memref<4xi1, #spirv.storage_class<StorageBuffer>>, %i : index) -> i1 {
  // CHECK-DAG: %[[SRC_CAST:.+]] = builtin.unrealized_conversion_cast %[[SRC]] : memref<4xi1, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<4 x i8, stride=1> [0])>, StorageBuffer>
  // CHECK-DAG: %[[IDX_CAST:.+]] = builtin.unrealized_conversion_cast %[[IDX]]
  // CHECK: %[[ZERO:.+]] = spirv.Constant 0 : i32
  // CHECK: %[[ONE:.+]] = spirv.Constant 1 : i32
  // CHECK: %[[MUL:.+]] = spirv.IMul %[[ONE]], %[[IDX_CAST]] : i32
  // CHECK: %[[ADD:.+]] = spirv.IAdd %[[ZERO]], %[[MUL]] : i32
  // CHECK: %[[ADDR:.+]] = spirv.AccessChain %[[SRC_CAST]][%[[ZERO]], %[[ADD]]]
  // CHECK: %[[VAL:.+]] = spirv.Load "StorageBuffer" %[[ADDR]] : i8
  // CHECK: %[[ONE_I8:.+]] = spirv.Constant 1 : i8
  // CHECK: %[[BOOL:.+]] = spirv.IEqual %[[VAL]], %[[ONE_I8]] : i8
  %0 = memref.load %src[%i] : memref<4xi1, #spirv.storage_class<StorageBuffer>>
  // CHECK: return %[[BOOL]]
  return %0: i1
}

// CHECK-LABEL: func @store_i1
//  CHECK-SAME: %[[DST:.+]]: memref<4xi1, #spirv.storage_class<StorageBuffer>>,
//  CHECK-SAME: %[[IDX:.+]]: index
func.func @store_i1(%dst: memref<4xi1, #spirv.storage_class<StorageBuffer>>, %i: index) {
  %true = arith.constant true
  // CHECK-DAG: %[[DST_CAST:.+]] = builtin.unrealized_conversion_cast %[[DST]] : memref<4xi1, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<4 x i8, stride=1> [0])>, StorageBuffer>
  // CHECK-DAG: %[[IDX_CAST:.+]] = builtin.unrealized_conversion_cast %[[IDX]]
  // CHECK: %[[ZERO:.+]] = spirv.Constant 0 : i32
  // CHECK: %[[ONE:.+]] = spirv.Constant 1 : i32
  // CHECK: %[[MUL:.+]] = spirv.IMul %[[ONE]], %[[IDX_CAST]] : i32
  // CHECK: %[[ADD:.+]] = spirv.IAdd %[[ZERO]], %[[MUL]] : i32
  // CHECK: %[[ADDR:.+]] = spirv.AccessChain %[[DST_CAST]][%[[ZERO]], %[[ADD]]]
  // CHECK: %[[ZERO_I8:.+]] = spirv.Constant 0 : i8
  // CHECK: %[[ONE_I8:.+]] = spirv.Constant 1 : i8
  // CHECK: %[[RES:.+]] = spirv.Select %{{.+}}, %[[ONE_I8]], %[[ZERO_I8]] : i1, i8
  // CHECK: spirv.Store "StorageBuffer" %[[ADDR]], %[[RES]] : i8
  memref.store %true, %dst[%i]: memref<4xi1, #spirv.storage_class<StorageBuffer>>
  return
}

// CHECK-LABEL: @load_i16
func.func @load_i16(%arg0: memref<i16, #spirv.storage_class<StorageBuffer>>) {
  // CHECK-NOT: spirv.SDiv
  //     CHECK: spirv.Load
  // CHECK-NOT: spirv.ShiftRightArithmetic
  %0 = memref.load %arg0[] : memref<i16, #spirv.storage_class<StorageBuffer>>
  return
}

// CHECK-LABEL: @store_i16
func.func @store_i16(%arg0: memref<10xi16, #spirv.storage_class<StorageBuffer>>, %index: index, %value: i16) {
  //     CHECK: spirv.Store
  // CHECK-NOT: spirv.AtomicAnd
  // CHECK-NOT: spirv.AtomicOr
  memref.store %value, %arg0[%index] : memref<10xi16, #spirv.storage_class<StorageBuffer>>
  return
}

} // end module

// -----

// Check for Kernel capability, that with proper compute and storage extensions, we don't need to
// perform special tricks.

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0,
      [
        Kernel, Addresses, Int8, Int16, Int64, Float16, Float64], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @load_store_zero_rank_float
func.func @load_store_zero_rank_float(%arg0: memref<f32, #spirv.storage_class<CrossWorkgroup>>, %arg1: memref<f32, #spirv.storage_class<CrossWorkgroup>>) {
  //      CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast {{.+}} : memref<f32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<!spirv.array<1 x f32>, CrossWorkgroup>
  //      CHECK: [[ARG1:%.*]] = builtin.unrealized_conversion_cast {{.+}} : memref<f32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<!spirv.array<1 x f32>, CrossWorkgroup>
  //      CHECK: [[ZERO:%.*]] = spirv.Constant 0 : i32
  //      CHECK: spirv.AccessChain [[ARG0]][
  // CHECK-SAME: [[ZERO]]
  // CHECK-SAME: ] :
  //      CHECK: spirv.Load "CrossWorkgroup" %{{.*}} : f32
  %0 = memref.load %arg0[] : memref<f32, #spirv.storage_class<CrossWorkgroup>>
  //      CHECK: spirv.AccessChain [[ARG1]][
  // CHECK-SAME: [[ZERO]]
  // CHECK-SAME: ] :
  //      CHECK: spirv.Store "CrossWorkgroup" %{{.*}} : f32
  memref.store %0, %arg1[] : memref<f32, #spirv.storage_class<CrossWorkgroup>>
  return
}

// CHECK-LABEL: @load_store_zero_rank_int
func.func @load_store_zero_rank_int(%arg0: memref<i32, #spirv.storage_class<CrossWorkgroup>>, %arg1: memref<i32, #spirv.storage_class<CrossWorkgroup>>) {
  //      CHECK: [[ARG0:%.*]] = builtin.unrealized_conversion_cast {{.+}} : memref<i32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<!spirv.array<1 x i32>, CrossWorkgroup>
  //      CHECK: [[ARG1:%.*]] = builtin.unrealized_conversion_cast {{.+}} : memref<i32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<!spirv.array<1 x i32>, CrossWorkgroup>
  //      CHECK: [[ZERO:%.*]] = spirv.Constant 0 : i32
  //      CHECK: spirv.AccessChain [[ARG0]][
  // CHECK-SAME: [[ZERO]]
  // CHECK-SAME: ] :
  //      CHECK: spirv.Load "CrossWorkgroup" %{{.*}} : i32
  %0 = memref.load %arg0[] : memref<i32, #spirv.storage_class<CrossWorkgroup>>
  //      CHECK: spirv.AccessChain [[ARG1]][
  // CHECK-SAME: [[ZERO]]
  // CHECK-SAME: ] :
  //      CHECK: spirv.Store "CrossWorkgroup" %{{.*}} : i32
  memref.store %0, %arg1[] : memref<i32, #spirv.storage_class<CrossWorkgroup>>
  return
}

// CHECK-LABEL: func @load_store_unknown_dim
func.func @load_store_unknown_dim(%i: index, %source: memref<?xi32, #spirv.storage_class<CrossWorkgroup>>, %dest: memref<?xi32, #spirv.storage_class<CrossWorkgroup>>) {
  // CHECK: %[[SRC:.+]] = builtin.unrealized_conversion_cast {{.+}} : memref<?xi32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<i32, CrossWorkgroup>
  // CHECK: %[[DST:.+]] = builtin.unrealized_conversion_cast {{.+}} : memref<?xi32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<i32, CrossWorkgroup>
  // CHECK: %[[AC0:.+]] = spirv.PtrAccessChain %[[SRC]]
  // CHECK: spirv.Load "CrossWorkgroup" %[[AC0]]
  %0 = memref.load %source[%i] : memref<?xi32, #spirv.storage_class<CrossWorkgroup>>
  // CHECK: %[[AC1:.+]] = spirv.PtrAccessChain %[[DST]]
  // CHECK: spirv.Store "CrossWorkgroup" %[[AC1]]
  memref.store %0, %dest[%i]: memref<?xi32, #spirv.storage_class<CrossWorkgroup>>
  return
}

// CHECK-LABEL: func @load_i1
//  CHECK-SAME: (%[[SRC:.+]]: memref<4xi1, #spirv.storage_class<CrossWorkgroup>>, %[[IDX:.+]]: index)
func.func @load_i1(%src: memref<4xi1, #spirv.storage_class<CrossWorkgroup>>, %i : index) -> i1 {
  // CHECK-DAG: %[[SRC_CAST:.+]] = builtin.unrealized_conversion_cast %[[SRC]] : memref<4xi1, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<!spirv.array<4 x i8>, CrossWorkgroup>
  // CHECK-DAG: %[[IDX_CAST:.+]] = builtin.unrealized_conversion_cast %[[IDX]]
  // CHECK: %[[ZERO:.+]] = spirv.Constant 0 : i32
  // CHECK: %[[ONE:.+]] = spirv.Constant 1 : i32
  // CHECK: %[[MUL:.+]] = spirv.IMul %[[ONE]], %[[IDX_CAST]] : i32
  // CHECK: %[[ADD:.+]] = spirv.IAdd %[[ZERO]], %[[MUL]] : i32
  // CHECK: %[[ADDR:.+]] = spirv.AccessChain %[[SRC_CAST]][%[[ADD]]]
  // CHECK: %[[VAL:.+]] = spirv.Load "CrossWorkgroup" %[[ADDR]] : i8
  // CHECK: %[[ONE_I8:.+]] = spirv.Constant 1 : i8
  // CHECK: %[[BOOL:.+]] = spirv.IEqual %[[VAL]], %[[ONE_I8]] : i8
  %0 = memref.load %src[%i] : memref<4xi1, #spirv.storage_class<CrossWorkgroup>>
  // CHECK: return %[[BOOL]]
  return %0: i1
}

// CHECK-LABEL: func @store_i1
//  CHECK-SAME: %[[DST:.+]]: memref<4xi1, #spirv.storage_class<CrossWorkgroup>>,
//  CHECK-SAME: %[[IDX:.+]]: index
func.func @store_i1(%dst: memref<4xi1, #spirv.storage_class<CrossWorkgroup>>, %i: index) {
  %true = arith.constant true
  // CHECK-DAG: %[[DST_CAST:.+]] = builtin.unrealized_conversion_cast %[[DST]] : memref<4xi1, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<!spirv.array<4 x i8>, CrossWorkgroup>
  // CHECK-DAG: %[[IDX_CAST:.+]] = builtin.unrealized_conversion_cast %[[IDX]]
  // CHECK: %[[ZERO:.+]] = spirv.Constant 0 : i32
  // CHECK: %[[ONE:.+]] = spirv.Constant 1 : i32
  // CHECK: %[[MUL:.+]] = spirv.IMul %[[ONE]], %[[IDX_CAST]] : i32
  // CHECK: %[[ADD:.+]] = spirv.IAdd %[[ZERO]], %[[MUL]] : i32
  // CHECK: %[[ADDR:.+]] = spirv.AccessChain %[[DST_CAST]][%[[ADD]]]
  // CHECK: %[[ZERO_I8:.+]] = spirv.Constant 0 : i8
  // CHECK: %[[ONE_I8:.+]] = spirv.Constant 1 : i8
  // CHECK: %[[RES:.+]] = spirv.Select %{{.+}}, %[[ONE_I8]], %[[ZERO_I8]] : i1, i8
  // CHECK: spirv.Store "CrossWorkgroup" %[[ADDR]], %[[RES]] : i8
  memref.store %true, %dst[%i]: memref<4xi1, #spirv.storage_class<CrossWorkgroup>>
  return
}

} // end module

// -----

// Check address space casts

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0,
      [
        Kernel, Addresses, GenericPointer], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: func.func @memory_space_cast
func.func @memory_space_cast(%arg: memref<4xf32, #spirv.storage_class<CrossWorkgroup>>)
    -> memref<4xf32, #spirv.storage_class<Function>> {
  // CHECK: %[[ARG_CAST:.+]] = builtin.unrealized_conversion_cast {{.*}} to !spirv.ptr<!spirv.array<4 x f32>, CrossWorkgroup>
  // CHECK: %[[TO_GENERIC:.+]] = spirv.PtrCastToGeneric %[[ARG_CAST]] : !spirv.ptr<!spirv.array<4 x f32>, CrossWorkgroup> to !spirv.ptr<!spirv.array<4 x f32>, Generic>
  // CHECK: %[[TO_PRIVATE:.+]] = spirv.GenericCastToPtr %[[TO_GENERIC]] : !spirv.ptr<!spirv.array<4 x f32>, Generic> to !spirv.ptr<!spirv.array<4 x f32>, Function>
  // CHECK: %[[RET:.+]] = builtin.unrealized_conversion_cast %[[TO_PRIVATE]]
  // CHECK: return %[[RET]]
  %ret = memref.memory_space_cast %arg : memref<4xf32, #spirv.storage_class<CrossWorkgroup>>
    to memref<4xf32, #spirv.storage_class<Function>>
  return %ret : memref<4xf32, #spirv.storage_class<Function>>
}

} // end module

// -----

// Check that access chain indices are properly adjusted if non-32-bit types are
// emulated via 32-bit types.
// TODO: Test i64 types.
module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @load_i1
func.func @load_i1(%arg0: memref<i1, #spirv.storage_class<StorageBuffer>>) -> i1 {
  //     CHECK: %[[ZERO:.+]] = spirv.Constant 0 : i32
  //     CHECK: %[[FOUR:.+]] = spirv.Constant 4 : i32
  //     CHECK: %[[QUOTIENT:.+]] = spirv.SDiv %[[ZERO]], %[[FOUR]] : i32
  //     CHECK: %[[PTR:.+]] = spirv.AccessChain %{{.+}}[%[[ZERO]], %[[QUOTIENT]]]
  //     CHECK: %[[LOAD:.+]] = spirv.Load  "StorageBuffer" %[[PTR]]
  //     CHECK: %[[EIGHT:.+]] = spirv.Constant 8 : i32
  //     CHECK: %[[IDX:.+]] = spirv.UMod %[[ZERO]], %[[FOUR]] : i32
  //     CHECK: %[[BITS:.+]] = spirv.IMul %[[IDX]], %[[EIGHT]] : i32
  //     CHECK: %[[VALUE:.+]] = spirv.ShiftRightArithmetic %[[LOAD]], %[[BITS]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spirv.Constant 255 : i32
  //     CHECK: %[[T1:.+]] = spirv.BitwiseAnd %[[VALUE]], %[[MASK]] : i32
  //     CHECK: %[[T2:.+]] = spirv.Constant 24 : i32
  //     CHECK: %[[T3:.+]] = spirv.ShiftLeftLogical %[[T1]], %[[T2]] : i32, i32
  //     CHECK: %[[T4:.+]] = spirv.ShiftRightArithmetic %[[T3]], %[[T2]] : i32, i32
  // Convert to i1 type.
  //     CHECK: %[[ONE:.+]] = spirv.Constant 1 : i32
  //     CHECK: %[[RES:.+]]  = spirv.IEqual %[[T4]], %[[ONE]] : i32
  //     CHECK: return %[[RES]]
  %0 = memref.load %arg0[] : memref<i1, #spirv.storage_class<StorageBuffer>>
  return %0 : i1
}

// CHECK-LABEL: @load_i8
func.func @load_i8(%arg0: memref<i8, #spirv.storage_class<StorageBuffer>>) -> i8 {
  //     CHECK: %[[ZERO:.+]] = spirv.Constant 0 : i32
  //     CHECK: %[[FOUR:.+]] = spirv.Constant 4 : i32
  //     CHECK: %[[QUOTIENT:.+]] = spirv.SDiv %[[ZERO]], %[[FOUR]] : i32
  //     CHECK: %[[PTR:.+]] = spirv.AccessChain %{{.+}}[%[[ZERO]], %[[QUOTIENT]]]
  //     CHECK: %[[LOAD:.+]] = spirv.Load  "StorageBuffer" %[[PTR]]
  //     CHECK: %[[EIGHT:.+]] = spirv.Constant 8 : i32
  //     CHECK: %[[IDX:.+]] = spirv.UMod %[[ZERO]], %[[FOUR]] : i32
  //     CHECK: %[[BITS:.+]] = spirv.IMul %[[IDX]], %[[EIGHT]] : i32
  //     CHECK: %[[VALUE:.+]] = spirv.ShiftRightArithmetic %[[LOAD]], %[[BITS]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spirv.Constant 255 : i32
  //     CHECK: %[[T1:.+]] = spirv.BitwiseAnd %[[VALUE]], %[[MASK]] : i32
  //     CHECK: %[[T2:.+]] = spirv.Constant 24 : i32
  //     CHECK: %[[T3:.+]] = spirv.ShiftLeftLogical %[[T1]], %[[T2]] : i32, i32
  //     CHECK: %[[SR:.+]] = spirv.ShiftRightArithmetic %[[T3]], %[[T2]] : i32, i32
  //     CHECK: builtin.unrealized_conversion_cast %[[SR]]
  %0 = memref.load %arg0[] : memref<i8, #spirv.storage_class<StorageBuffer>>
  return %0 : i8
}

// CHECK-LABEL: @load_i16
//       CHECK: (%[[ARG0:.+]]: {{.*}}, %[[ARG1:.+]]: index)
func.func @load_i16(%arg0: memref<10xi16, #spirv.storage_class<StorageBuffer>>, %index : index) -> i16 {
  //     CHECK: %[[ARG1_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : index to i32
  //     CHECK: %[[ZERO:.+]] = spirv.Constant 0 : i32
  //     CHECK: %[[ONE:.+]] = spirv.Constant 1 : i32
  //     CHECK: %[[UPDATE:.+]] = spirv.IMul %[[ONE]], %[[ARG1_CAST]] : i32
  //     CHECK: %[[FLAT_IDX:.+]] = spirv.IAdd %[[ZERO]], %[[UPDATE]] : i32
  //     CHECK: %[[TWO:.+]] = spirv.Constant 2 : i32
  //     CHECK: %[[QUOTIENT:.+]] = spirv.SDiv %[[FLAT_IDX]], %[[TWO]] : i32
  //     CHECK: %[[PTR:.+]] = spirv.AccessChain %{{.+}}[%[[ZERO]], %[[QUOTIENT]]]
  //     CHECK: %[[LOAD:.+]] = spirv.Load  "StorageBuffer" %[[PTR]]
  //     CHECK: %[[SIXTEEN:.+]] = spirv.Constant 16 : i32
  //     CHECK: %[[IDX:.+]] = spirv.UMod %[[FLAT_IDX]], %[[TWO]] : i32
  //     CHECK: %[[BITS:.+]] = spirv.IMul %[[IDX]], %[[SIXTEEN]] : i32
  //     CHECK: %[[VALUE:.+]] = spirv.ShiftRightArithmetic %[[LOAD]], %[[BITS]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spirv.Constant 65535 : i32
  //     CHECK: %[[T1:.+]] = spirv.BitwiseAnd %[[VALUE]], %[[MASK]] : i32
  //     CHECK: %[[T3:.+]] = spirv.ShiftLeftLogical %[[T1]], %[[SIXTEEN]] : i32, i32
  //     CHECK: %[[SR:.+]] = spirv.ShiftRightArithmetic %[[T3]], %[[SIXTEEN]] : i32, i32
  //     CHECK: builtin.unrealized_conversion_cast %[[SR]]
  %0 = memref.load %arg0[%index] : memref<10xi16, #spirv.storage_class<StorageBuffer>>
  return %0: i16
}

// CHECK-LABEL: @load_f32
func.func @load_f32(%arg0: memref<f32, #spirv.storage_class<StorageBuffer>>) {
  // CHECK-NOT: spirv.SDiv
  //     CHECK: spirv.Load
  // CHECK-NOT: spirv.ShiftRightArithmetic
  %0 = memref.load %arg0[] : memref<f32, #spirv.storage_class<StorageBuffer>>
  return
}

// CHECK-LABEL: @store_i1
//       CHECK: (%[[ARG0:.+]]: {{.*}}, %[[ARG1:.+]]: i1)
func.func @store_i1(%arg0: memref<i1, #spirv.storage_class<StorageBuffer>>, %value: i1) {
  //     CHECK: %[[ARG0_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG0]]
  //     CHECK: %[[ZERO:.+]] = spirv.Constant 0 : i32
  //     CHECK: %[[FOUR:.+]] = spirv.Constant 4 : i32
  //     CHECK: %[[EIGHT:.+]] = spirv.Constant 8 : i32
  //     CHECK: %[[IDX:.+]] = spirv.UMod %[[ZERO]], %[[FOUR]] : i32
  //     CHECK: %[[OFFSET:.+]] = spirv.IMul %[[IDX]], %[[EIGHT]] : i32
  //     CHECK: %[[MASK1:.+]] = spirv.Constant 255 : i32
  //     CHECK: %[[TMP1:.+]] = spirv.ShiftLeftLogical %[[MASK1]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spirv.Not %[[TMP1]] : i32
  //     CHECK: %[[ONE:.+]] = spirv.Constant 1 : i32
  //     CHECK: %[[CASTED_ARG1:.+]] = spirv.Select %[[ARG1]], %[[ONE]], %[[ZERO]] : i1, i32
  //     CHECK: %[[CLAMPED_VAL:.+]] = spirv.BitwiseAnd %[[CASTED_ARG1]], %[[MASK1]] : i32
  //     CHECK: %[[STORE_VAL:.+]] = spirv.ShiftLeftLogical %[[CLAMPED_VAL]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[ACCESS_IDX:.+]] = spirv.SDiv %[[ZERO]], %[[FOUR]] : i32
  //     CHECK: %[[PTR:.+]] = spirv.AccessChain %[[ARG0_CAST]][%[[ZERO]], %[[ACCESS_IDX]]]
  //     CHECK: spirv.AtomicAnd "Device" "AcquireRelease" %[[PTR]], %[[MASK]]
  //     CHECK: spirv.AtomicOr "Device" "AcquireRelease" %[[PTR]], %[[STORE_VAL]]
  memref.store %value, %arg0[] : memref<i1, #spirv.storage_class<StorageBuffer>>
  return
}

// CHECK-LABEL: @store_i8
//       CHECK: (%[[ARG0:.+]]: {{.*}}, %[[ARG1:.+]]: i8)
func.func @store_i8(%arg0: memref<i8, #spirv.storage_class<StorageBuffer>>, %value: i8) {
  //     CHECK-DAG: %[[ARG1_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : i8 to i32
  //     CHECK-DAG: %[[ARG0_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG0]]
  //     CHECK: %[[ZERO:.+]] = spirv.Constant 0 : i32
  //     CHECK: %[[FOUR:.+]] = spirv.Constant 4 : i32
  //     CHECK: %[[EIGHT:.+]] = spirv.Constant 8 : i32
  //     CHECK: %[[IDX:.+]] = spirv.UMod %[[ZERO]], %[[FOUR]] : i32
  //     CHECK: %[[OFFSET:.+]] = spirv.IMul %[[IDX]], %[[EIGHT]] : i32
  //     CHECK: %[[MASK1:.+]] = spirv.Constant 255 : i32
  //     CHECK: %[[TMP1:.+]] = spirv.ShiftLeftLogical %[[MASK1]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spirv.Not %[[TMP1]] : i32
  //     CHECK: %[[CLAMPED_VAL:.+]] = spirv.BitwiseAnd %[[ARG1_CAST]], %[[MASK1]] : i32
  //     CHECK: %[[STORE_VAL:.+]] = spirv.ShiftLeftLogical %[[CLAMPED_VAL]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[ACCESS_IDX:.+]] = spirv.SDiv %[[ZERO]], %[[FOUR]] : i32
  //     CHECK: %[[PTR:.+]] = spirv.AccessChain %[[ARG0_CAST]][%[[ZERO]], %[[ACCESS_IDX]]]
  //     CHECK: spirv.AtomicAnd "Device" "AcquireRelease" %[[PTR]], %[[MASK]]
  //     CHECK: spirv.AtomicOr "Device" "AcquireRelease" %[[PTR]], %[[STORE_VAL]]
  memref.store %value, %arg0[] : memref<i8, #spirv.storage_class<StorageBuffer>>
  return
}

// CHECK-LABEL: @store_i16
//       CHECK: (%[[ARG0:.+]]: memref<10xi16, #spirv.storage_class<StorageBuffer>>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: i16)
func.func @store_i16(%arg0: memref<10xi16, #spirv.storage_class<StorageBuffer>>, %index: index, %value: i16) {
  //     CHECK-DAG: %[[ARG2_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG2]] : i16 to i32
  //     CHECK-DAG: %[[ARG0_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG0]]
  //     CHECK-DAG: %[[ARG1_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : index to i32
  //     CHECK: %[[ZERO:.+]] = spirv.Constant 0 : i32
  //     CHECK: %[[ONE:.+]] = spirv.Constant 1 : i32
  //     CHECK: %[[UPDATE:.+]] = spirv.IMul %[[ONE]], %[[ARG1_CAST]] : i32
  //     CHECK: %[[FLAT_IDX:.+]] = spirv.IAdd %[[ZERO]], %[[UPDATE]] : i32
  //     CHECK: %[[TWO:.+]] = spirv.Constant 2 : i32
  //     CHECK: %[[SIXTEEN:.+]] = spirv.Constant 16 : i32
  //     CHECK: %[[IDX:.+]] = spirv.UMod %[[FLAT_IDX]], %[[TWO]] : i32
  //     CHECK: %[[OFFSET:.+]] = spirv.IMul %[[IDX]], %[[SIXTEEN]] : i32
  //     CHECK: %[[MASK1:.+]] = spirv.Constant 65535 : i32
  //     CHECK: %[[TMP1:.+]] = spirv.ShiftLeftLogical %[[MASK1]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[MASK:.+]] = spirv.Not %[[TMP1]] : i32
  //     CHECK: %[[CLAMPED_VAL:.+]] = spirv.BitwiseAnd %[[ARG2_CAST]], %[[MASK1]] : i32
  //     CHECK: %[[STORE_VAL:.+]] = spirv.ShiftLeftLogical %[[CLAMPED_VAL]], %[[OFFSET]] : i32, i32
  //     CHECK: %[[ACCESS_IDX:.+]] = spirv.SDiv %[[FLAT_IDX]], %[[TWO]] : i32
  //     CHECK: %[[PTR:.+]] = spirv.AccessChain %[[ARG0_CAST]][%[[ZERO]], %[[ACCESS_IDX]]]
  //     CHECK: spirv.AtomicAnd "Device" "AcquireRelease" %[[PTR]], %[[MASK]]
  //     CHECK: spirv.AtomicOr "Device" "AcquireRelease" %[[PTR]], %[[STORE_VAL]]
  memref.store %value, %arg0[%index] : memref<10xi16, #spirv.storage_class<StorageBuffer>>
  return
}

// CHECK-LABEL: @store_f32
func.func @store_f32(%arg0: memref<f32, #spirv.storage_class<StorageBuffer>>, %value: f32) {
  //     CHECK: spirv.Store
  // CHECK-NOT: spirv.AtomicAnd
  // CHECK-NOT: spirv.AtomicOr
  memref.store %value, %arg0[] : memref<f32, #spirv.storage_class<StorageBuffer>>
  return
}

} // end module

// -----

// Check that access chain indices are properly adjusted if sub-byte types are
// emulated via 32-bit types.
module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @load_i4
func.func @load_i4(%arg0: memref<?xi4, #spirv.storage_class<StorageBuffer>>, %i: index) -> i4 {
  // CHECK: %[[INDEX:.+]] = builtin.unrealized_conversion_cast %{{.+}} : index to i32
  // CHECK: %[[ZERO:.+]] = spirv.Constant 0 : i32
  // CHECK: %[[ONE:.+]] = spirv.Constant 1 : i32
  // CHECK: %[[MUL:.+]] = spirv.IMul %[[ONE]], %[[INDEX]] : i32
  // CHECK: %[[OFFSET:.+]] = spirv.IAdd %[[ZERO]], %[[MUL]] : i32
  // CHECK: %[[EIGHT:.+]] = spirv.Constant 8 : i32
  // CHECK: %[[QUOTIENT:.+]] = spirv.SDiv %[[OFFSET]], %[[EIGHT]] : i32
  // CHECK: %[[PTR:.+]] = spirv.AccessChain %{{.+}}[%[[ZERO]], %[[QUOTIENT]]]
  // CHECK: %[[LOAD:.+]] = spirv.Load "StorageBuffer" %[[PTR]] : i32
  // CHECK: %[[FOUR:.+]] = spirv.Constant 4 : i32
  // CHECK: %[[IDX:.+]] = spirv.UMod %[[OFFSET]], %[[EIGHT]] : i32
  // CHECK: %[[BITS:.+]] = spirv.IMul %[[IDX]], %[[FOUR]] : i32
  // CHECK: %[[VALUE:.+]] = spirv.ShiftRightArithmetic %[[LOAD]], %[[BITS]] : i32, i32
  // CHECK: %[[MASK:.+]] = spirv.Constant 15 : i32
  // CHECK: %[[AND:.+]] = spirv.BitwiseAnd %[[VALUE]], %[[MASK]] : i32
  // CHECK: %[[C28:.+]] = spirv.Constant 28 : i32
  // CHECK: %[[SL:.+]] = spirv.ShiftLeftLogical %[[AND]], %[[C28]] : i32, i32
  // CHECK: %[[SR:.+]] = spirv.ShiftRightArithmetic %[[SL]], %[[C28]] : i32, i32
  // CHECK: builtin.unrealized_conversion_cast %[[SR]]
  %0 = memref.load %arg0[%i] : memref<?xi4, #spirv.storage_class<StorageBuffer>>
  return %0 : i4
}

// CHECK-LABEL: @store_i4
func.func @store_i4(%arg0: memref<?xi4, #spirv.storage_class<StorageBuffer>>, %value: i4, %i: index) {
  // CHECK: %[[VAL:.+]] = builtin.unrealized_conversion_cast %{{.+}} : i4 to i32
  // CHECK: %[[INDEX:.+]] = builtin.unrealized_conversion_cast %{{.+}} : index to i32
  // CHECK: %[[ZERO:.+]] = spirv.Constant 0 : i32
  // CHECK: %[[ONE:.+]] = spirv.Constant 1 : i32
  // CHECK: %[[MUL:.+]] = spirv.IMul %[[ONE]], %[[INDEX]] : i32
  // CHECK: %[[OFFSET:.+]] = spirv.IAdd %[[ZERO]], %[[MUL]] : i32
  // CHECK: %[[EIGHT:.+]] = spirv.Constant 8 : i32
  // CHECK: %[[FOUR:.+]] = spirv.Constant [[OFFSET]] : i32
  // CHECK: %[[IDX:.+]] = spirv.UMod %[[OFFSET]], %[[EIGHT]] : i32
  // CHECK: %[[BITS:.+]] = spirv.IMul %[[IDX]], %[[FOUR]] : i32
  // CHECK: %[[MASK1:.+]] = spirv.Constant 15 : i32
  // CHECK: %[[SL:.+]] = spirv.ShiftLeftLogical %[[MASK1]], %[[BITS]] : i32, i32
  // CHECK: %[[MASK2:.+]] = spirv.Not %[[SL]] : i32
  // CHECK: %[[CLAMPED_VAL:.+]] = spirv.BitwiseAnd %[[VAL]], %[[MASK1]] : i32
  // CHECK: %[[STORE_VAL:.+]] = spirv.ShiftLeftLogical %[[CLAMPED_VAL]], %[[BITS]] : i32, i32
  // CHECK: %[[ACCESS_INDEX:.+]] = spirv.SDiv %[[OFFSET]], %[[EIGHT]] : i32
  // CHECK: %[[PTR:.+]] = spirv.AccessChain %{{.+}}[%[[ZERO]], %[[ACCESS_INDEX]]]
  // CHECK: spirv.AtomicAnd "Device" "AcquireRelease" %[[PTR]], %[[MASK2]]
  // CHECK: spirv.AtomicOr "Device" "AcquireRelease" %[[PTR]], %[[STORE_VAL]]
  memref.store %value, %arg0[%i] : memref<?xi4, #spirv.storage_class<StorageBuffer>>
  return
}

} // end module

// -----

// Check that casts are properly inserted if the corresponding **compute**
// capability is allowed.
module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader, Int8, Int16], [
      SPV_KHR_8bit_storage, SPV_KHR_16bit_storage, SPV_KHR_storage_buffer_storage_class
      ]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @load_i1
func.func @load_i1(%arg0: memref<i1, #spirv.storage_class<StorageBuffer>>) -> i1 {
  //     CHECK: %[[ONE:.+]] = spirv.Constant 1 : i32
  //     CHECK: %[[RES:.+]]  = spirv.IEqual %{{.+}}, %[[ONE]] : i32
  //     CHECK: return %[[RES]]
  %0 = memref.load %arg0[] : memref<i1, #spirv.storage_class<StorageBuffer>>
  return %0 : i1
}

// CHECK-LABEL: @load_i8
func.func @load_i8(%arg0: memref<i8, #spirv.storage_class<StorageBuffer>>) -> i8 {
  //     CHECK: %[[RES:.+]] = spirv.UConvert %{{.+}} : i32 to i8
  //     CHECK: return %[[RES]]
  %0 = memref.load %arg0[] : memref<i8, #spirv.storage_class<StorageBuffer>>
  return %0 : i8
}

// CHECK-LABEL: @load_i16
func.func @load_i16(%arg0: memref<10xi16, #spirv.storage_class<StorageBuffer>>, %index : index) -> i16 {
  //     CHECK: %[[RES:.+]] = spirv.UConvert %{{.+}} : i32 to i16
  //     CHECK: return %[[RES]]
  %0 = memref.load %arg0[%index] : memref<10xi16, #spirv.storage_class<StorageBuffer>>
  return %0: i16
}

} // end module

// -----

// Check reinterpret_casts

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0,
      [Kernel, Addresses, GenericPointer], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: func.func @reinterpret_cast
//  CHECK-SAME:  (%[[MEM:.*]]: memref<?xf32, #spirv.storage_class<CrossWorkgroup>>, %[[OFF:.*]]: index)
func.func @reinterpret_cast(%arg: memref<?xf32, #spirv.storage_class<CrossWorkgroup>>, %arg1: index) -> memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>> {
//       CHECK:  %[[MEM1:.*]] = builtin.unrealized_conversion_cast %[[MEM]] : memref<?xf32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<f32, CrossWorkgroup>
//       CHECK:  %[[OFF1:.*]] = builtin.unrealized_conversion_cast %[[OFF]] : index to i32
//       CHECK:  %[[RET:.*]] = spirv.InBoundsPtrAccessChain %[[MEM1]][%[[OFF1]]] : !spirv.ptr<f32, CrossWorkgroup>, i32
//       CHECK:  %[[RET1:.*]] = builtin.unrealized_conversion_cast %[[RET]] : !spirv.ptr<f32, CrossWorkgroup> to memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>>
//       CHECK:  return %[[RET1]]
  %ret = memref.reinterpret_cast %arg to offset: [%arg1], sizes: [10], strides: [1] : memref<?xf32, #spirv.storage_class<CrossWorkgroup>> to memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>>
  return %ret : memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>>
}

// CHECK-LABEL: func.func @reinterpret_cast_0
//  CHECK-SAME:  (%[[MEM:.*]]: memref<?xf32, #spirv.storage_class<CrossWorkgroup>>)
func.func @reinterpret_cast_0(%arg: memref<?xf32, #spirv.storage_class<CrossWorkgroup>>) -> memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>> {
//       CHECK:  %[[MEM1:.*]] = builtin.unrealized_conversion_cast %[[MEM]] : memref<?xf32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<f32, CrossWorkgroup>
//       CHECK:  %[[RET:.*]] = builtin.unrealized_conversion_cast %[[MEM1]] : !spirv.ptr<f32, CrossWorkgroup> to memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>>
//       CHECK:  return %[[RET]]
  %ret = memref.reinterpret_cast %arg to offset: [0], sizes: [10], strides: [1] : memref<?xf32, #spirv.storage_class<CrossWorkgroup>> to memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>>
  return %ret : memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>>
}

// CHECK-LABEL: func.func @reinterpret_cast_5
//  CHECK-SAME:  (%[[MEM:.*]]: memref<?xf32, #spirv.storage_class<CrossWorkgroup>>)
func.func @reinterpret_cast_5(%arg: memref<?xf32, #spirv.storage_class<CrossWorkgroup>>) -> memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>> {
//       CHECK:  %[[MEM1:.*]] = builtin.unrealized_conversion_cast %[[MEM]] : memref<?xf32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<f32, CrossWorkgroup>
//       CHECK:  %[[OFF:.*]] = spirv.Constant 5 : i32
//       CHECK:  %[[RET:.*]] = spirv.InBoundsPtrAccessChain %[[MEM1]][%[[OFF]]] : !spirv.ptr<f32, CrossWorkgroup>, i32
//       CHECK:  %[[RET1:.*]] = builtin.unrealized_conversion_cast %[[RET]] : !spirv.ptr<f32, CrossWorkgroup> to memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>>
//       CHECK:  return %[[RET1]]
  %ret = memref.reinterpret_cast %arg to offset: [5], sizes: [10], strides: [1] : memref<?xf32, #spirv.storage_class<CrossWorkgroup>> to memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>>
  return %ret : memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>>
}

} // end module
