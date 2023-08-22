// RUN: mlir-opt -split-input-file -convert-memref-to-spirv="bool-num-bits=8" -cse %s -o - | FileCheck %s

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
