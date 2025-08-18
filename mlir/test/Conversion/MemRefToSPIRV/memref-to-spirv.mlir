// RUN: mlir-opt --split-input-file -pass-pipeline="builtin.module(convert-memref-to-spirv{bool-num-bits=8}, cse)" %s | FileCheck %s
// RUN: mlir-opt --split-input-file -pass-pipeline="builtin.module(convert-memref-to-spirv{bool-num-bits=8 use-64bit-index=true}, cse)" %s \
// RUN: | FileCheck --check-prefix=CHECK64 %s

// Check that with proper compute and storage extensions, we don't need to
// perform special tricks.

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.5,
      [
        Shader, Int8, Int16, Int64, Float16, Float64,
        StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16,
        StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8,
        PhysicalStorageBufferAddresses
      ],
      [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_physical_storage_buffer]>,
      #spirv.resource_limits<>>
} {

// CHECK-LABEL: @load_store_zero_rank_float(
//  CHECK-SAME:     %[[OARG0:.*]]: memref{{.*}}, %[[OARG1:.*]]: memref
func.func @load_store_zero_rank_float(%arg0: memref<f32, #spirv.storage_class<StorageBuffer>>, %arg1: memref<f32, #spirv.storage_class<StorageBuffer>>) {
  //  CHECK-DAG: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %[[OARG0]] : memref<f32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>
  //  CHECK-DAG: [[ARG1:%.*]] = builtin.unrealized_conversion_cast %[[OARG1]] : memref<f32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>
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
//  CHECK-SAME:     %[[OARG0:.*]]: memref{{.*}}, %[[OARG1:.*]]: memref
func.func @load_store_zero_rank_int(%arg0: memref<i32, #spirv.storage_class<StorageBuffer>>, %arg1: memref<i32, #spirv.storage_class<StorageBuffer>>) {
  //  CHECK-DAG: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %[[OARG0]] : memref<i32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>, StorageBuffer>
  //  CHECK-DAG: [[ARG1:%.*]] = builtin.unrealized_conversion_cast %[[OARG1]] : memref<i32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>, StorageBuffer>
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
//  CHECK-SAME:     %[[OARG0:.*]]: index, %[[OARG1:.*]]: memref{{.*}}, %[[OARG2:.*]]: memref
func.func @load_store_unknown_dim(%i: index, %source: memref<?xi32, #spirv.storage_class<StorageBuffer>>, %dest: memref<?xi32, #spirv.storage_class<StorageBuffer>>) {
  // CHECK-DAG: %[[SRC:.+]] = builtin.unrealized_conversion_cast %[[OARG1]] : memref<?xi32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
  // CHECK-DAG: %[[DST:.+]] = builtin.unrealized_conversion_cast %[[OARG2]] : memref<?xi32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
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
  // CHECK: %[[ZERO:.*]] = spirv.Constant 0 : i32
  // CHECK: %[[ADDR:.+]] = spirv.AccessChain %[[SRC_CAST]][%[[ZERO]], %[[IDX_CAST]]]
  // CHECK: %[[VAL:.+]] = spirv.Load "StorageBuffer" %[[ADDR]] : i8
  // CHECK: %[[ZERO_I8:.+]] = spirv.Constant 0 : i8
  // CHECK: %[[BOOL:.+]] = spirv.INotEqual %[[VAL]], %[[ZERO_I8]] : i8
  %0 = memref.load %src[%i] : memref<4xi1, #spirv.storage_class<StorageBuffer>>
  // CHECK: return %[[BOOL]]
  return %0: i1
}

// CHECK-LABEL: func @load_aligned
//  CHECK-SAME: (%[[SRC:.+]]: memref<4xi1, #spirv.storage_class<StorageBuffer>>, %[[IDX:.+]]: index)
func.func @load_aligned(%src: memref<4xi1, #spirv.storage_class<StorageBuffer>>, %i : index) -> i1 {
  // CHECK: spirv.Load "StorageBuffer" {{.*}} ["Aligned", 32] : i8
  %0 = memref.load %src[%i] { alignment = 32 } : memref<4xi1, #spirv.storage_class<StorageBuffer>>
  return %0: i1
}

// CHECK-LABEL: func @load_aligned_nontemporal
func.func @load_aligned_nontemporal(%src: memref<4xi1, #spirv.storage_class<StorageBuffer>>, %i : index) -> i1 {
  // CHECK: spirv.Load "StorageBuffer" {{.*}} ["Aligned|Nontemporal", 32] : i8
  %0 = memref.load %src[%i] { alignment = 32, nontemporal = true } : memref<4xi1, #spirv.storage_class<StorageBuffer>>
  return %0: i1
}

// CHECK-LABEL: func @load_aligned_psb
func.func @load_aligned_psb(%src: memref<4xi1, #spirv.storage_class<PhysicalStorageBuffer>>, %i : index) -> i1 {
  // CHECK: %[[VAL:.+]] = spirv.Load "PhysicalStorageBuffer" {{.*}} ["Aligned", 32] : i8
  %0 = memref.load %src[%i] { alignment = 32 } : memref<4xi1, #spirv.storage_class<PhysicalStorageBuffer>>
  return %0: i1
}

// CHECK-LABEL: func @store_i1
//  CHECK-SAME: %[[DST:.+]]: memref<4xi1, #spirv.storage_class<StorageBuffer>>,
//  CHECK-SAME: %[[IDX:.+]]: index
func.func @store_i1(%dst: memref<4xi1, #spirv.storage_class<StorageBuffer>>, %i: index) {
  %true = arith.constant true
  // CHECK-DAG: %[[DST_CAST:.+]] = builtin.unrealized_conversion_cast %[[DST]] : memref<4xi1, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<4 x i8, stride=1> [0])>, StorageBuffer>
  // CHECK-DAG: %[[IDX_CAST:.+]] = builtin.unrealized_conversion_cast %[[IDX]]
  // CHECK: %[[ZERO:.*]] = spirv.Constant 0 : i32
  // CHECK: %[[ADDR:.+]] = spirv.AccessChain %[[DST_CAST]][%[[ZERO]], %[[IDX_CAST]]]
  // CHECK: %[[ONE_I8:.+]] = spirv.Constant 1 : i8
  // CHECK: spirv.Store "StorageBuffer" %[[ADDR]], %[[ONE_I8]] : i8
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

// CHECK-LABEL: @load_store_i32_physical
func.func @load_store_i32_physical(%arg0: memref<i32, #spirv.storage_class<PhysicalStorageBuffer>>) {
  //     CHECK: spirv.Load "PhysicalStorageBuffer" %{{.+}} ["Aligned", 4] : i32
  //     CHECK: spirv.Store "PhysicalStorageBuffer" %{{.+}}, %{{.+}} ["Aligned", 4] : i32
  %0 = memref.load %arg0[] : memref<i32, #spirv.storage_class<PhysicalStorageBuffer>>
  memref.store %0, %arg0[] : memref<i32, #spirv.storage_class<PhysicalStorageBuffer>>
  return
}

// CHECK-LABEL: @load_store_i8_physical
func.func @load_store_i8_physical(%arg0: memref<i8, #spirv.storage_class<PhysicalStorageBuffer>>) {
  //     CHECK: spirv.Load "PhysicalStorageBuffer" %{{.+}} ["Aligned", 1] : i8
  //     CHECK: spirv.Store "PhysicalStorageBuffer" %{{.+}}, %{{.+}} ["Aligned", 1] : i8
  %0 = memref.load %arg0[] : memref<i8, #spirv.storage_class<PhysicalStorageBuffer>>
  memref.store %0, %arg0[] : memref<i8, #spirv.storage_class<PhysicalStorageBuffer>>
  return
}

// CHECK-LABEL: @load_store_i1_physical
func.func @load_store_i1_physical(%arg0: memref<i1, #spirv.storage_class<PhysicalStorageBuffer>>) {
  //     CHECK: spirv.Load "PhysicalStorageBuffer" %{{.+}} ["Aligned", 1] : i8
  //     CHECK: spirv.Store "PhysicalStorageBuffer" %{{.+}}, %{{.+}} ["Aligned", 1] : i8
  %0 = memref.load %arg0[] : memref<i1, #spirv.storage_class<PhysicalStorageBuffer>>
  memref.store %0, %arg0[] : memref<i1, #spirv.storage_class<PhysicalStorageBuffer>>
  return
}

// CHECK-LABEL: @load_store_f32_physical
func.func @load_store_f32_physical(%arg0: memref<f32, #spirv.storage_class<PhysicalStorageBuffer>>) {
  //     CHECK: spirv.Load "PhysicalStorageBuffer" %{{.+}} ["Aligned", 4] : f32
  //     CHECK: spirv.Store "PhysicalStorageBuffer" %{{.+}}, %{{.+}} ["Aligned", 4] : f32
  %0 = memref.load %arg0[] : memref<f32, #spirv.storage_class<PhysicalStorageBuffer>>
  memref.store %0, %arg0[] : memref<f32, #spirv.storage_class<PhysicalStorageBuffer>>
  return
}

// CHECK-LABEL: @load_store_f16_physical
func.func @load_store_f16_physical(%arg0: memref<f16, #spirv.storage_class<PhysicalStorageBuffer>>) {
  //     CHECK: spirv.Load "PhysicalStorageBuffer" %{{.+}} ["Aligned", 2] : f16
  //     CHECK: spirv.Store "PhysicalStorageBuffer" %{{.+}}, %{{.+}} ["Aligned", 2] : f16
  %0 = memref.load %arg0[] : memref<f16, #spirv.storage_class<PhysicalStorageBuffer>>
  memref.store %0, %arg0[] : memref<f16, #spirv.storage_class<PhysicalStorageBuffer>>
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
//  CHECK-SAME:     %[[OARG0:.*]]: memref{{.*}}, %[[OARG1:.*]]: memref
func.func @load_store_zero_rank_float(%arg0: memref<f32, #spirv.storage_class<CrossWorkgroup>>, %arg1: memref<f32, #spirv.storage_class<CrossWorkgroup>>) {
  //  CHECK-DAG: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %[[OARG0]] : memref<f32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<!spirv.array<1 x f32>, CrossWorkgroup>
  //  CHECK-DAG: [[ARG1:%.*]] = builtin.unrealized_conversion_cast %[[OARG1]] : memref<f32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<!spirv.array<1 x f32>, CrossWorkgroup>
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
//  CHECK-SAME:     %[[OARG0:.*]]: memref{{.*}}, %[[OARG1:.*]]: memref
func.func @load_store_zero_rank_int(%arg0: memref<i32, #spirv.storage_class<CrossWorkgroup>>, %arg1: memref<i32, #spirv.storage_class<CrossWorkgroup>>) {
  //  CHECK-DAG: [[ARG0:%.*]] = builtin.unrealized_conversion_cast %[[OARG0]] : memref<i32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<!spirv.array<1 x i32>, CrossWorkgroup>
  //  CHECK-DAG: [[ARG1:%.*]] = builtin.unrealized_conversion_cast %[[OARG1]] : memref<i32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<!spirv.array<1 x i32>, CrossWorkgroup>
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
//  CHECK-SAME:     %[[OARG0:.*]]: index, %[[OARG1:.*]]: memref{{.*}}, %[[OARG2:.*]]: memref
func.func @load_store_unknown_dim(%i: index, %source: memref<?xi32, #spirv.storage_class<CrossWorkgroup>>, %dest: memref<?xi32, #spirv.storage_class<CrossWorkgroup>>) {
  // CHECK-DAG: %[[SRC:.+]] = builtin.unrealized_conversion_cast %[[OARG1]] : memref<?xi32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<i32, CrossWorkgroup>
  // CHECK-DAG: %[[DST:.+]] = builtin.unrealized_conversion_cast %[[OARG2]] : memref<?xi32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<i32, CrossWorkgroup>
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
  // CHECK: %[[ADDR:.+]] = spirv.AccessChain %[[SRC_CAST]][%[[IDX_CAST]]]
  // CHECK: %[[VAL:.+]] = spirv.Load "CrossWorkgroup" %[[ADDR]] : i8
  // CHECK: %[[ZERO_I8:.+]] = spirv.Constant 0 : i8
  // CHECK: %[[BOOL:.+]] = spirv.INotEqual %[[VAL]], %[[ZERO_I8]] : i8
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
  // CHECK: %[[ADDR:.+]] = spirv.AccessChain %[[DST_CAST]][%[[IDX_CAST]]]
  // CHECK: %[[ONE_I8:.+]] = spirv.Constant 1 : i8
  // CHECK: spirv.Store "CrossWorkgroup" %[[ADDR]], %[[ONE_I8]] : i8
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
//   CHECK-DAG:  %[[MEM1:.*]] = builtin.unrealized_conversion_cast %[[MEM]] : memref<?xf32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<f32, CrossWorkgroup>
//   CHECK-DAG:  %[[OFF1:.*]] = builtin.unrealized_conversion_cast %[[OFF]] : index to i32
//       CHECK:  %[[RET:.*]] = spirv.InBoundsPtrAccessChain %[[MEM1]][%[[OFF1]]] : !spirv.ptr<f32, CrossWorkgroup>, i32
//       CHECK:  %[[RET1:.*]] = builtin.unrealized_conversion_cast %[[RET]] : !spirv.ptr<f32, CrossWorkgroup> to memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>>
//       CHECK:  return %[[RET1]]
  %ret = memref.reinterpret_cast %arg to offset: [%arg1], sizes: [10], strides: [1] : memref<?xf32, #spirv.storage_class<CrossWorkgroup>> to memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>>
  return %ret : memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>>
}

// CHECK-LABEL: func.func @reinterpret_cast_0
//  CHECK-SAME:  (%[[MEM:.*]]: memref<?xf32, #spirv.storage_class<CrossWorkgroup>>)
func.func @reinterpret_cast_0(%arg: memref<?xf32, #spirv.storage_class<CrossWorkgroup>>) -> memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>> {
//   CHECK-DAG:  %[[MEM1:.*]] = builtin.unrealized_conversion_cast %[[MEM]] : memref<?xf32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<f32, CrossWorkgroup>
//   CHECK-DAG:  %[[RET:.*]] = builtin.unrealized_conversion_cast %[[MEM1]] : !spirv.ptr<f32, CrossWorkgroup> to memref<?xf32, strided<[1], offset: ?>, #spirv.storage_class<CrossWorkgroup>>
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


// -----

// Check casts

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0,
      [Kernel, Addresses, GenericPointer], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: func.func @cast
//  CHECK-SAME:  (%[[MEM:.*]]: memref<4x?xf32, #spirv.storage_class<CrossWorkgroup>>)
func.func @cast(%arg: memref<4x?xf32, #spirv.storage_class<CrossWorkgroup>>) -> memref<?x4xf32, #spirv.storage_class<CrossWorkgroup>> {
//   CHECK-DAG:  %[[MEM1:.*]] = builtin.unrealized_conversion_cast %[[MEM]] : memref<4x?xf32, #spirv.storage_class<CrossWorkgroup>> to !spirv.ptr<f32, CrossWorkgroup>
//   CHECK-DAG:  %[[MEM2:.*]] = builtin.unrealized_conversion_cast %[[MEM1]] : !spirv.ptr<f32, CrossWorkgroup> to memref<?x4xf32, #spirv.storage_class<CrossWorkgroup>>
//       CHECK:  return %[[MEM2]]
  %ret = memref.cast %arg : memref<4x?xf32, #spirv.storage_class<CrossWorkgroup>> to memref<?x4xf32, #spirv.storage_class<CrossWorkgroup>>
  return %ret : memref<?x4xf32, #spirv.storage_class<CrossWorkgroup>>
}

// TODO: Not supported yet
// CHECK-LABEL: func.func @cast_from_static
//  CHECK-SAME:  (%[[MEM:.*]]: memref<4x4xf32, #spirv.storage_class<CrossWorkgroup>>)
func.func @cast_from_static(%arg: memref<4x4xf32, #spirv.storage_class<CrossWorkgroup>>) -> memref<?x4xf32, #spirv.storage_class<CrossWorkgroup>> {
//       CHECK:  %[[MEM1:.*]] =  memref.cast %[[MEM]] : memref<4x4xf32, #spirv.storage_class<CrossWorkgroup>> to memref<?x4xf32, #spirv.storage_class<CrossWorkgroup>>
//       CHECK:  return %[[MEM1]]
  %ret = memref.cast %arg : memref<4x4xf32, #spirv.storage_class<CrossWorkgroup>> to memref<?x4xf32, #spirv.storage_class<CrossWorkgroup>>
  return %ret : memref<?x4xf32, #spirv.storage_class<CrossWorkgroup>>
}

// TODO: Not supported yet
// CHECK-LABEL: func.func @cast_to_static
//  CHECK-SAME:  (%[[MEM:.*]]: memref<4x?xf32, #spirv.storage_class<CrossWorkgroup>>)
func.func @cast_to_static(%arg: memref<4x?xf32, #spirv.storage_class<CrossWorkgroup>>) -> memref<4x4xf32, #spirv.storage_class<CrossWorkgroup>> {
//       CHECK:  %[[MEM1:.*]] =  memref.cast %[[MEM]] : memref<4x?xf32, #spirv.storage_class<CrossWorkgroup>> to memref<4x4xf32, #spirv.storage_class<CrossWorkgroup>>
//       CHECK:  return %[[MEM1]]
  %ret = memref.cast %arg : memref<4x?xf32, #spirv.storage_class<CrossWorkgroup>> to memref<4x4xf32, #spirv.storage_class<CrossWorkgroup>>
  return %ret : memref<4x4xf32, #spirv.storage_class<CrossWorkgroup>>
}

// TODO: Not supported yet
// CHECK-LABEL: func.func @cast_to_static_zero_elems
//  CHECK-SAME:  (%[[MEM:.*]]: memref<?xf32, #spirv.storage_class<CrossWorkgroup>>)
func.func @cast_to_static_zero_elems(%arg: memref<?xf32, #spirv.storage_class<CrossWorkgroup>>) -> memref<0xf32, #spirv.storage_class<CrossWorkgroup>> {
//       CHECK:  %[[MEM1:.*]] =  memref.cast %[[MEM]] : memref<?xf32, #spirv.storage_class<CrossWorkgroup>> to memref<0xf32, #spirv.storage_class<CrossWorkgroup>>
//       CHECK:  return %[[MEM1]]
  %ret = memref.cast %arg : memref<?xf32, #spirv.storage_class<CrossWorkgroup>> to memref<0xf32, #spirv.storage_class<CrossWorkgroup>>
  return %ret : memref<0xf32, #spirv.storage_class<CrossWorkgroup>>
}

}

// -----

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.5, [Kernel, Int64, Addresses, PhysicalStorageBufferAddresses], []>, #spirv.resource_limits<>>
} {
// CHECK-LABEL: func @extract_aligned_pointer_as_index_kernel
func.func @extract_aligned_pointer_as_index_kernel(%m: memref<?xf32, #spirv.storage_class<CrossWorkgroup>>) -> index {
  %0 = memref.extract_aligned_pointer_as_index %m: memref<?xf32, #spirv.storage_class<CrossWorkgroup>> -> index
  // CHECK: %[[I32:.*]] = spirv.ConvertPtrToU {{%.*}} : !spirv.ptr<f32, CrossWorkgroup> to i32
  // CHECK: %[[R:.*]] = builtin.unrealized_conversion_cast %[[I32]] : i32 to index
  // CHECK64: %[[I64:.*]] = spirv.ConvertPtrToU {{%.*}} : !spirv.ptr<f32, CrossWorkgroup> to i64
  // CHECK64: %[[R:.*]] = builtin.unrealized_conversion_cast %[[I64]] : i64 to index

  // CHECK: return %[[R:.*]] : index
  return %0: index
}
}

// -----

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.5, [Shader, Int64, Addresses, PhysicalStorageBufferAddresses], []>, #spirv.resource_limits<>>
} {
// CHECK-LABEL: func @extract_aligned_pointer_as_index_shader
func.func @extract_aligned_pointer_as_index_shader(%m: memref<?xf32, #spirv.storage_class<CrossWorkgroup>>) -> index {
  %0 = memref.extract_aligned_pointer_as_index %m: memref<?xf32, #spirv.storage_class<CrossWorkgroup>> -> index
  // CHECK: %[[I32:.*]] = spirv.ConvertPtrToU {{%.*}} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32>)>, CrossWorkgroup> to i32
  // CHECK: %[[R:.*]] = builtin.unrealized_conversion_cast %[[I32]] : i32 to index
  // CHECK64: %[[I64:.*]] = spirv.ConvertPtrToU {{%.*}} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32>)>, CrossWorkgroup> to i64
  // CHECK64: %[[R:.*]] = builtin.unrealized_conversion_cast %[[I64]] : i64 to index

  // CHECK: return %[[R:.*]] : index
  return %0: index
}
}


// -----

// Check nontemporal attribute

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [
    Shader,
    PhysicalStorageBufferAddresses
  ], [
    SPV_KHR_storage_buffer_storage_class,
    SPV_KHR_physical_storage_buffer
  ]>, #spirv.resource_limits<>>
} {
  func.func @load_nontemporal(%arg0: memref<f32, #spirv.storage_class<StorageBuffer>>) {
    %0 = memref.load %arg0[] {nontemporal = true} : memref<f32, #spirv.storage_class<StorageBuffer>>
//       CHECK:  spirv.Load "StorageBuffer" %{{.+}} ["Nontemporal"] : f32
    memref.store %0, %arg0[] {nontemporal = true} : memref<f32, #spirv.storage_class<StorageBuffer>>
//       CHECK:  spirv.Store "StorageBuffer" %{{.+}}, %{{.+}} ["Nontemporal"] : f32
    return
  }

  func.func @load_nontemporal_aligned(%arg0: memref<f32, #spirv.storage_class<PhysicalStorageBuffer>>) {
    %0 = memref.load %arg0[] {nontemporal = true} : memref<f32, #spirv.storage_class<PhysicalStorageBuffer>>
//       CHECK:  spirv.Load "PhysicalStorageBuffer" %{{.+}} ["Aligned|Nontemporal", 4] : f32
    memref.store %0, %arg0[] {nontemporal = true} : memref<f32, #spirv.storage_class<PhysicalStorageBuffer>>
//       CHECK:  spirv.Store "PhysicalStorageBuffer" %{{.+}}, %{{.+}} ["Aligned|Nontemporal", 4] : f32
    return
  }
}

// -----

// Check Image Support.

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [
    Shader,
    PhysicalStorageBufferAddresses,
    Image1D,
    StorageImageExtendedFormats,
    Float16,
    StorageBuffer16BitAccess,
    StorageUniform16,
    Int16
  ], [
    SPV_KHR_storage_buffer_storage_class,
    SPV_KHR_physical_storage_buffer,
    SPV_KHR_16bit_storage
  ]>, #spirv.resource_limits<>>
} {
  // CHECK-LABEL: @load_from_image_1D(
  // CHECK-SAME: %[[ARG0:.*]]: memref<1xf32, #spirv.storage_class<Image>>, %[[ARG1:.*]]: memref<1xf32, #spirv.storage_class<StorageBuffer>>
  func.func @load_from_image_1D(%arg0: memref<1xf32, #spirv.storage_class<Image>>, %arg1: memref<1xf32, #spirv.storage_class<StorageBuffer>>) {
// CHECK-DAG: %[[SB:.*]] = builtin.unrealized_conversion_cast %arg1 : memref<1xf32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>
// CHECK-DAG: %[[IMAGE_PTR:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<1xf32, #spirv.storage_class<Image>> to !spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim1D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>>, UniformConstant>
    %cst = arith.constant 0 : index
    // CHECK: %[[COORDS:.*]] = builtin.unrealized_conversion_cast %{{.*}} : index to i32
    // CHECK: %[[SIMAGE:.*]] = spirv.Load "UniformConstant" %[[IMAGE_PTR]] : !spirv.sampled_image<!spirv.image<f32, Dim1D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>>
    // CHECK: %[[IMAGE:.*]] = spirv.Image %[[SIMAGE]] : !spirv.sampled_image<!spirv.image<f32, Dim1D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>>
    // CHECK-NOT: spirv.CompositeConstruct
    // CHECK: %[[RES_VEC:.*]] =  spirv.ImageFetch %[[IMAGE]], %[[COORDS]]  : !spirv.image<f32, Dim1D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>, i32 -> vector<4xf32>
    // CHECK: %[[RESULT:.*]] = spirv.CompositeExtract %[[RES_VEC]][0 : i32] : vector<4xf32>
    %0 = memref.load %arg0[%cst] : memref<1xf32, #spirv.storage_class<Image>>
    // CHECK: spirv.Store "StorageBuffer" %{{.*}}, %[[RESULT]] : f32
    memref.store %0, %arg1[%cst] : memref<1xf32, #spirv.storage_class<StorageBuffer>>
    return
  }

  // CHECK-LABEL: @load_from_image_2D(
  // CHECK-SAME: %[[ARG0:.*]]: memref<1x1xf32, #spirv.storage_class<Image>>, %[[ARG1:.*]]: memref<1x1xf32, #spirv.storage_class<StorageBuffer>>
  func.func @load_from_image_2D(%arg0: memref<1x1xf32, #spirv.storage_class<Image>>, %arg1: memref<1x1xf32, #spirv.storage_class<StorageBuffer>>) {
// CHECK-DAG: %[[SB:.*]] = builtin.unrealized_conversion_cast %arg1 : memref<1x1xf32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>
// CHECK-DAG: %[[IMAGE_PTR:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<1x1xf32, #spirv.storage_class<Image>> to !spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>>, UniformConstant>
    %cst = arith.constant 0 : index
    // CHECK: %[[SIMAGE:.*]] = spirv.Load "UniformConstant" %[[IMAGE_PTR]] : !spirv.sampled_image<!spirv.image<f32, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>>
    // CHECK: %[[IMAGE:.*]] = spirv.Image %[[SIMAGE]] : !spirv.sampled_image<!spirv.image<f32, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>>
    // CHECK: %[[COORDS:.*]] = spirv.CompositeConstruct %{{.*}}, %{{.*}} : (i32, i32) -> vector<2xi32>
    // CHECK: %[[RES_VEC:.*]] =  spirv.ImageFetch %[[IMAGE]], %[[COORDS]]  : !spirv.image<f32, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>, vector<2xi32> -> vector<4xf32>
    // CHECK: %[[RESULT:.*]] = spirv.CompositeExtract %[[RES_VEC]][0 : i32] : vector<4xf32>
    %0 = memref.load %arg0[%cst, %cst] : memref<1x1xf32, #spirv.storage_class<Image>>
    // CHECK: spirv.Store "StorageBuffer" %{{.*}}, %[[RESULT]] : f32
    memref.store %0, %arg1[%cst, %cst] : memref<1x1xf32, #spirv.storage_class<StorageBuffer>>
    return
  }

  // CHECK-LABEL: @load_from_image_3D(
  // CHECK-SAME: %[[ARG0:.*]]: memref<1x1x1xf32, #spirv.storage_class<Image>>, %[[ARG1:.*]]: memref<1x1x1xf32, #spirv.storage_class<StorageBuffer>>
  func.func @load_from_image_3D(%arg0: memref<1x1x1xf32, #spirv.storage_class<Image>>, %arg1: memref<1x1x1xf32, #spirv.storage_class<StorageBuffer>>) {
// CHECK-DAG: %[[SB:.*]] = builtin.unrealized_conversion_cast %arg1 : memref<1x1x1xf32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x f32, stride=4> [0])>, StorageBuffer>
// CHECK-DAG: %[[IMAGE_PTR:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<1x1x1xf32, #spirv.storage_class<Image>> to !spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim3D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>>, UniformConstant>
    %cst = arith.constant 0 : index
    // CHECK: %[[SIMAGE:.*]] = spirv.Load "UniformConstant" %[[IMAGE_PTR]] : !spirv.sampled_image<!spirv.image<f32, Dim3D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>>
    // CHECK: %[[IMAGE:.*]] = spirv.Image %[[SIMAGE]] : !spirv.sampled_image<!spirv.image<f32, Dim3D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>>
    // CHECK: %[[COORDS:.*]] = spirv.CompositeConstruct %{{.*}}, %{{.*}}, %{{.*}} : (i32, i32, i32) -> vector<3xi32>
    // CHECK: %[[RES_VEC:.*]] =  spirv.ImageFetch %[[IMAGE]], %[[COORDS]]  : !spirv.image<f32, Dim3D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32f>, vector<3xi32> -> vector<4xf32>
    // CHECK: %[[RESULT:.*]] = spirv.CompositeExtract %[[RES_VEC]][0 : i32] : vector<4xf32>
    %0 = memref.load %arg0[%cst, %cst, %cst] : memref<1x1x1xf32, #spirv.storage_class<Image>>
    // CHECK: spirv.Store "StorageBuffer" %{{.*}}, %[[RESULT]] : f32
    memref.store %0, %arg1[%cst, %cst, %cst] : memref<1x1x1xf32, #spirv.storage_class<StorageBuffer>>
    return
  }

  // CHECK-LABEL: @load_from_image_2D_f16(
  // CHECK-SAME: %[[ARG0:.*]]: memref<1x1xf16, #spirv.storage_class<Image>>, %[[ARG1:.*]]: memref<1x1xf16, #spirv.storage_class<StorageBuffer>>
  func.func @load_from_image_2D_f16(%arg0: memref<1x1xf16, #spirv.storage_class<Image>>, %arg1: memref<1x1xf16, #spirv.storage_class<StorageBuffer>>) {
// CHECK-DAG: %[[SB:.*]] = builtin.unrealized_conversion_cast %arg1 : memref<1x1xf16, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x f16, stride=2> [0])>, StorageBuffer>
// CHECK-DAG: %[[IMAGE_PTR:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<1x1xf16, #spirv.storage_class<Image>> to !spirv.ptr<!spirv.sampled_image<!spirv.image<f16, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R16f>>, UniformConstant>
    %cst = arith.constant 0 : index
    // CHECK: %[[SIMAGE:.*]] = spirv.Load "UniformConstant" %[[IMAGE_PTR]] : !spirv.sampled_image<!spirv.image<f16, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R16f>>
    // CHECK: %[[IMAGE:.*]] = spirv.Image %[[SIMAGE]] : !spirv.sampled_image<!spirv.image<f16, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R16f>>
    // CHECK: %[[COORDS:.*]] = spirv.CompositeConstruct %{{.*}}, %{{.*}} : (i32, i32) -> vector<2xi32>
    // CHECK: %[[RES_VEC:.*]] =  spirv.ImageFetch %[[IMAGE]], %[[COORDS]]  : !spirv.image<f16, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R16f>, vector<2xi32> -> vector<4xf16>
    // CHECK: %[[RESULT:.*]] = spirv.CompositeExtract %[[RES_VEC]][0 : i32] : vector<4xf16>
    %0 = memref.load %arg0[%cst, %cst] : memref<1x1xf16, #spirv.storage_class<Image>>
    // CHECK: spirv.Store "StorageBuffer" %{{.*}}, %[[RESULT]] : f16
    memref.store %0, %arg1[%cst, %cst] : memref<1x1xf16, #spirv.storage_class<StorageBuffer>>
    return
  }

  // CHECK-LABEL: @load_from_image_2D_i32(
  // CHECK-SAME: %[[ARG0:.*]]: memref<1x1xi32, #spirv.storage_class<Image>>, %[[ARG1:.*]]: memref<1x1xi32, #spirv.storage_class<StorageBuffer>>
  func.func @load_from_image_2D_i32(%arg0: memref<1x1xi32, #spirv.storage_class<Image>>, %arg1: memref<1x1xi32, #spirv.storage_class<StorageBuffer>>) {
// CHECK-DAG: %[[SB:.*]] = builtin.unrealized_conversion_cast %arg1 : memref<1x1xi32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x i32, stride=4> [0])>, StorageBuffer>
// CHECK-DAG: %[[IMAGE_PTR:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<1x1xi32, #spirv.storage_class<Image>> to !spirv.ptr<!spirv.sampled_image<!spirv.image<i32, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32i>>, UniformConstant>
    %cst = arith.constant 0 : index
    // CHECK: %[[SIMAGE:.*]] = spirv.Load "UniformConstant" %[[IMAGE_PTR]] : !spirv.sampled_image<!spirv.image<i32, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32i>>
    // CHECK: %[[IMAGE:.*]] = spirv.Image %[[SIMAGE]] : !spirv.sampled_image<!spirv.image<i32, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32i>>
    // CHECK: %[[COORDS:.*]] = spirv.CompositeConstruct %{{.*}}, %{{.*}} : (i32, i32) -> vector<2xi32>
    // CHECK: %[[RES_VEC:.*]] =  spirv.ImageFetch %[[IMAGE]], %[[COORDS]]  : !spirv.image<i32, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32i>, vector<2xi32> -> vector<4xi32>
    // CHECK: %[[RESULT:.*]] = spirv.CompositeExtract %[[RES_VEC]][0 : i32] : vector<4xi32>
    %0 = memref.load %arg0[%cst, %cst] : memref<1x1xi32, #spirv.storage_class<Image>>
    // CHECK: spirv.Store "StorageBuffer" %{{.*}}, %[[RESULT]] : i32
    memref.store %0, %arg1[%cst, %cst] : memref<1x1xi32, #spirv.storage_class<StorageBuffer>>
    return
  }

  // CHECK-LABEL: @load_from_image_2D_ui32(
  // CHECK-SAME: %[[ARG0:.*]]: memref<1x1xui32, #spirv.storage_class<Image>>, %[[ARG1:.*]]: memref<1x1xui32, #spirv.storage_class<StorageBuffer>>
  func.func @load_from_image_2D_ui32(%arg0: memref<1x1xui32, #spirv.storage_class<Image>>, %arg1: memref<1x1xui32, #spirv.storage_class<StorageBuffer>>) {
// CHECK-DAG: %[[SB:.*]] = builtin.unrealized_conversion_cast %arg1 : memref<1x1xui32, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x ui32, stride=4> [0])>, StorageBuffer>
// CHECK-DAG: %[[IMAGE_PTR:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<1x1xui32, #spirv.storage_class<Image>> to !spirv.ptr<!spirv.sampled_image<!spirv.image<ui32, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32ui>>, UniformConstant>
    %cst = arith.constant 0 : index
    // CHECK: %[[SIMAGE:.*]] = spirv.Load "UniformConstant" %[[IMAGE_PTR]] : !spirv.sampled_image<!spirv.image<ui32, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32ui>>
    // CHECK: %[[IMAGE:.*]] = spirv.Image %[[SIMAGE]] : !spirv.sampled_image<!spirv.image<ui32, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32ui>>
    // CHECK: %[[COORDS:.*]] = spirv.CompositeConstruct %{{.*}}, %{{.*}} : (i32, i32) -> vector<2xi32>
    // CHECK: %[[RES_VEC:.*]] =  spirv.ImageFetch %[[IMAGE]], %[[COORDS]]  : !spirv.image<ui32, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R32ui>, vector<2xi32> -> vector<4xui32>
    // CHECK: %[[RESULT:.*]] = spirv.CompositeExtract %[[RES_VEC]][0 : i32] : vector<4xui32>
    %0 = memref.load %arg0[%cst, %cst] : memref<1x1xui32, #spirv.storage_class<Image>>
    // CHECK: spirv.Store "StorageBuffer" %{{.*}}, %[[RESULT]] : ui32
    memref.store %0, %arg1[%cst, %cst] : memref<1x1xui32, #spirv.storage_class<StorageBuffer>>
    return
  }

  // CHECK-LABEL: @load_from_image_2D_i16(
  // CHECK-SAME: %[[ARG0:.*]]: memref<1x1xi16, #spirv.storage_class<Image>>, %[[ARG1:.*]]: memref<1x1xi16, #spirv.storage_class<StorageBuffer>>
  func.func @load_from_image_2D_i16(%arg0: memref<1x1xi16, #spirv.storage_class<Image>>, %arg1: memref<1x1xi16, #spirv.storage_class<StorageBuffer>>) {
// CHECK-DAG: %[[SB:.*]] = builtin.unrealized_conversion_cast %arg1 : memref<1x1xi16, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x i16, stride=2> [0])>, StorageBuffer>
// CHECK-DAG: %[[IMAGE_PTR:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<1x1xi16, #spirv.storage_class<Image>> to !spirv.ptr<!spirv.sampled_image<!spirv.image<i16, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R16i>>, UniformConstant>
    %cst = arith.constant 0 : index
    // CHECK: %[[SIMAGE:.*]] = spirv.Load "UniformConstant" %[[IMAGE_PTR]] : !spirv.sampled_image<!spirv.image<i16, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R16i>>
    // CHECK: %[[IMAGE:.*]] = spirv.Image %[[SIMAGE]] : !spirv.sampled_image<!spirv.image<i16, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R16i>>
    // CHECK: %[[COORDS:.*]] = spirv.CompositeConstruct %{{.*}}, %{{.*}} : (i32, i32) -> vector<2xi32>
    // CHECK: %[[RES_VEC:.*]] =  spirv.ImageFetch %[[IMAGE]], %[[COORDS]]  : !spirv.image<i16, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R16i>, vector<2xi32> -> vector<4xi16>
    // CHECK: %[[RESULT:.*]] = spirv.CompositeExtract %[[RES_VEC]][0 : i32] : vector<4xi16>
    %0 = memref.load %arg0[%cst, %cst] : memref<1x1xi16, #spirv.storage_class<Image>>
    // CHECK: spirv.Store "StorageBuffer" %{{.*}}, %[[RESULT]] : i16
    memref.store %0, %arg1[%cst, %cst] : memref<1x1xi16, #spirv.storage_class<StorageBuffer>>
    return
  }

  // CHECK-LABEL: @load_from_image_2D_ui16(
  // CHECK-SAME: %[[ARG0:.*]]: memref<1x1xui16, #spirv.storage_class<Image>>, %[[ARG1:.*]]: memref<1x1xui16, #spirv.storage_class<StorageBuffer>>
  func.func @load_from_image_2D_ui16(%arg0: memref<1x1xui16, #spirv.storage_class<Image>>, %arg1: memref<1x1xui16, #spirv.storage_class<StorageBuffer>>) {
// CHECK-DAG: %[[SB:.*]] = builtin.unrealized_conversion_cast %arg1 : memref<1x1xui16, #spirv.storage_class<StorageBuffer>> to !spirv.ptr<!spirv.struct<(!spirv.array<1 x ui16, stride=2> [0])>, StorageBuffer>
// CHECK-DAG: %[[IMAGE_PTR:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<1x1xui16, #spirv.storage_class<Image>> to !spirv.ptr<!spirv.sampled_image<!spirv.image<ui16, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R16ui>>, UniformConstant>
    %cst = arith.constant 0 : index
    // CHECK: %[[SIMAGE:.*]] = spirv.Load "UniformConstant" %[[IMAGE_PTR]] : !spirv.sampled_image<!spirv.image<ui16, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R16ui>>
    // CHECK: %[[IMAGE:.*]] = spirv.Image %[[SIMAGE]] : !spirv.sampled_image<!spirv.image<ui16, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R16ui>>
    // CHECK: %[[COORDS:.*]] = spirv.CompositeConstruct %{{.*}}, %{{.*}} : (i32, i32) -> vector<2xi32>
    // CHECK: %[[RES_VEC:.*]] =  spirv.ImageFetch %[[IMAGE]], %[[COORDS]]  : !spirv.image<ui16, Dim2D, DepthUnknown, NonArrayed, SingleSampled, NeedSampler, R16ui>, vector<2xi32> -> vector<4xui16>
    // CHECK: %[[RESULT:.*]] = spirv.CompositeExtract %[[RES_VEC]][0 : i32] : vector<4xui16>
    %0 = memref.load %arg0[%cst, %cst] : memref<1x1xui16, #spirv.storage_class<Image>>
    // CHECK: spirv.Store "StorageBuffer" %{{.*}}, %[[RESULT]] : ui16
    memref.store %0, %arg1[%cst, %cst] : memref<1x1xui16, #spirv.storage_class<StorageBuffer>>
    return
  }

  // CHECK-LABEL: @load_from_image_2D_rank0(
  func.func @load_from_image_2D_rank0(%arg0: memref<f32, #spirv.storage_class<Image>>, %arg1: memref<f32, #spirv.storage_class<StorageBuffer>>) {
    %cst = arith.constant 0 : index
    // CHECK-NOT: spirv.Image
    // CHECK-NOT: spirv.ImageFetch
    %0 = memref.load %arg0[] : memref<f32, #spirv.storage_class<Image>>
    memref.store %0, %arg1[] : memref<f32, #spirv.storage_class<StorageBuffer>>
    return
  }

  // CHECK-LABEL: @load_from_image_2D_rank4(
  func.func @load_from_image_2D_rank4(%arg0: memref<1x1x1x1xf32, #spirv.storage_class<Image>>, %arg1: memref<1x1x1x1xf32, #spirv.storage_class<StorageBuffer>>) {
    %cst = arith.constant 0 : index
    // CHECK-NOT: spirv.Image
    // CHECK-NOT: spirv.ImageFetch
    %0 = memref.load %arg0[%cst, %cst, %cst, %cst] : memref<1x1x1x1xf32, #spirv.storage_class<Image>>
    memref.store %0, %arg1[%cst, %cst, %cst, %cst] : memref<1x1x1x1xf32, #spirv.storage_class<StorageBuffer>>
    return
  }

  // CHECK-LABEL: @load_from_image_2D_vector(
  func.func @load_from_image_2D_vector(%arg0: memref<1xvector<1xf32>, #spirv.storage_class<Image>>, %arg1: memref<1xvector<1xf32>, #spirv.storage_class<StorageBuffer>>) {
    %cst = arith.constant 0 : index
    // CHECK-NOT: spirv.Image
    // CHECK-NOT: spirv.ImageFetch
    %0 = memref.load %arg0[%cst] : memref<1xvector<1xf32>, #spirv.storage_class<Image>>
    memref.store %0, %arg1[%cst] : memref<1xvector<1xf32>, #spirv.storage_class<StorageBuffer>>
    return
  }
}
