// RUN: mlir-opt -split-input-file -convert-memref-to-spirv %s -o - | FileCheck %s

module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, #spirv.resource_limits<>>} {

//      CHECK: func.func @atomic_addi_storage_buffer
// CHECK-SAME: (%[[VAL:.+]]: i32,
func.func @atomic_addi_storage_buffer(%value: i32, %memref: memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>, %i0: index, %i1: index, %i2: index) -> i32 {
  // CHECK: %[[AC:.+]] = spirv.AccessChain
  // CHECK: %[[ATOMIC:.+]] = spirv.AtomicIAdd <Device> <AcquireRelease> %[[AC]], %[[VAL]] : !spirv.ptr<i32, StorageBuffer>
  // CHECK: return %[[ATOMIC]]
  %0 = memref.atomic_rmw "addi" %value, %memref[%i0, %i1, %i2] : (i32, memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>) -> i32
  return %0: i32
}

//      CHECK: func.func @atomic_maxs_workgroup
// CHECK-SAME: (%[[VAL:.+]]: i32,
func.func @atomic_maxs_workgroup(%value: i32, %memref: memref<2x3x4xi32, #spirv.storage_class<Workgroup>>, %i0: index, %i1: index, %i2: index) -> i32 {
  // CHECK: %[[AC:.+]] = spirv.AccessChain
  // CHECK: %[[ATOMIC:.+]] = spirv.AtomicSMax <Workgroup> <AcquireRelease> %[[AC]], %[[VAL]] : !spirv.ptr<i32, Workgroup>
  // CHECK: return %[[ATOMIC]]
  %0 = memref.atomic_rmw "maxs" %value, %memref[%i0, %i1, %i2] : (i32, memref<2x3x4xi32, #spirv.storage_class<Workgroup>>) -> i32
  return %0: i32
}

//      CHECK: func.func @atomic_maxu_storage_buffer
// CHECK-SAME: (%[[VAL:.+]]: i32,
func.func @atomic_maxu_storage_buffer(%value: i32, %memref: memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>, %i0: index, %i1: index, %i2: index) -> i32 {
  // CHECK: %[[AC:.+]] = spirv.AccessChain
  // CHECK: %[[ATOMIC:.+]] = spirv.AtomicUMax <Device> <AcquireRelease> %[[AC]], %[[VAL]] : !spirv.ptr<i32, StorageBuffer>
  // CHECK: return %[[ATOMIC]]
  %0 = memref.atomic_rmw "maxu" %value, %memref[%i0, %i1, %i2] : (i32, memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>) -> i32
  return %0: i32
}

//      CHECK: func.func @atomic_mins_workgroup
// CHECK-SAME: (%[[VAL:.+]]: i32,
func.func @atomic_mins_workgroup(%value: i32, %memref: memref<2x3x4xi32, #spirv.storage_class<Workgroup>>, %i0: index, %i1: index, %i2: index) -> i32 {
  // CHECK: %[[AC:.+]] = spirv.AccessChain
  // CHECK: %[[ATOMIC:.+]] = spirv.AtomicSMin <Workgroup> <AcquireRelease> %[[AC]], %[[VAL]] : !spirv.ptr<i32, Workgroup>
  // CHECK: return %[[ATOMIC]]
  %0 = memref.atomic_rmw "mins" %value, %memref[%i0, %i1, %i2] : (i32, memref<2x3x4xi32, #spirv.storage_class<Workgroup>>) -> i32
  return %0: i32
}

//      CHECK: func.func @atomic_minu_storage_buffer
// CHECK-SAME: (%[[VAL:.+]]: i32,
func.func @atomic_minu_storage_buffer(%value: i32, %memref: memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>, %i0: index, %i1: index, %i2: index) -> i32 {
  // CHECK: %[[AC:.+]] = spirv.AccessChain
  // CHECK: %[[ATOMIC:.+]] = spirv.AtomicUMin <Device> <AcquireRelease> %[[AC]], %[[VAL]] : !spirv.ptr<i32, StorageBuffer>
  // CHECK: return %[[ATOMIC]]
  %0 = memref.atomic_rmw "minu" %value, %memref[%i0, %i1, %i2] : (i32, memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>) -> i32
  return %0: i32
}

//      CHECK: func.func @atomic_ori_workgroup
// CHECK-SAME: (%[[VAL:.+]]: i32,
func.func @atomic_ori_workgroup(%value: i32, %memref: memref<2x3x4xi32, #spirv.storage_class<Workgroup>>, %i0: index, %i1: index, %i2: index) -> i32 {
  // CHECK: %[[AC:.+]] = spirv.AccessChain
  // CHECK: %[[ATOMIC:.+]] = spirv.AtomicOr <Workgroup> <AcquireRelease> %[[AC]], %[[VAL]] : !spirv.ptr<i32, Workgroup>
  // CHECK: return %[[ATOMIC]]
  %0 = memref.atomic_rmw "ori" %value, %memref[%i0, %i1, %i2] : (i32, memref<2x3x4xi32, #spirv.storage_class<Workgroup>>) -> i32
  return %0: i32
}

//      CHECK: func.func @atomic_andi_storage_buffer
// CHECK-SAME: (%[[VAL:.+]]: i32,
func.func @atomic_andi_storage_buffer(%value: i32, %memref: memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>, %i0: index, %i1: index, %i2: index) -> i32 {
  // CHECK: %[[AC:.+]] = spirv.AccessChain
  // CHECK: %[[ATOMIC:.+]] = spirv.AtomicAnd <Device> <AcquireRelease> %[[AC]], %[[VAL]] : !spirv.ptr<i32, StorageBuffer>
  // CHECK: return %[[ATOMIC]]
  %0 = memref.atomic_rmw "andi" %value, %memref[%i0, %i1, %i2] : (i32, memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>) -> i32
  return %0: i32
}

}

// -----

// Check sub-element-width atomic ori on i8 memref (stored as i32 in SPIR-V).
// The byte index must be divided by 4 to get the i32 index, and the value
// must be shifted to the correct byte position within the i32.

module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, #spirv.resource_limits<>>} {

// CHECK-LABEL: func.func @atomic_ori_i8_storage_buffer
func.func @atomic_ori_i8_storage_buffer(%value: i8, %memref: memref<16xi8, #spirv.storage_class<StorageBuffer>>, %i0: index) -> i8 {
  //      CHECK:     %[[IDX:.+]] = builtin.unrealized_conversion_cast %{{.*}} : index to i32
  //      CHECK:     %[[MEM:.+]] = builtin.unrealized_conversion_cast %{{.*}} : memref{{.*}} to !spirv.ptr
  //      CHECK:     %[[VAL:.+]] = builtin.unrealized_conversion_cast %{{.*}} : i8 to i32
  // Compute bit offset: (idx % 4) * 8
  //  CHECK-DAG:     %[[C4:.+]] = spirv.Constant 4 : i32
  //  CHECK-DAG:     %[[C8:.+]] = spirv.Constant 8 : i32
  //      CHECK:     %[[MOD:.+]] = spirv.UMod %[[IDX]], %[[C4]]
  //      CHECK:     %[[OFFSET:.+]] = spirv.IMul %[[MOD]], %[[C8]]
  // Adjust the access chain index: idx / 4
  //      CHECK:     %[[DIV:.+]] = spirv.SDiv %[[IDX]], %{{.*}}
  //      CHECK:     %[[AC:.+]] = spirv.AccessChain %[[MEM]][%{{.*}}, %[[DIV]]]
  // Mask and shift the value
  //      CHECK:     %[[C255:.+]] = spirv.Constant 255 : i32
  //      CHECK:     %[[MASKED:.+]] = spirv.BitwiseAnd %[[VAL]], %[[C255]]
  //      CHECK:     %[[SHIFTED:.+]] = spirv.ShiftLeftLogical %[[MASKED]], %[[OFFSET]]
  // Atomic OR
  //      CHECK:     %[[ATOMIC:.+]] = spirv.AtomicOr <Device> <AcquireRelease> %[[AC]], %[[SHIFTED]]
  // Extract old value from result
  //      CHECK:     spirv.ShiftRightLogical %[[ATOMIC]], %[[OFFSET]]
  //      CHECK:     spirv.BitwiseAnd
  //      CHECK:     return
  %0 = memref.atomic_rmw "ori" %value, %memref[%i0] : (i8, memref<16xi8, #spirv.storage_class<StorageBuffer>>) -> i8
  return %0: i8
}

}

// -----

// Check sub-element-width atomic andi on i8 memref.

module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, #spirv.resource_limits<>>} {

// CHECK-LABEL: func.func @atomic_andi_i8_storage_buffer
func.func @atomic_andi_i8_storage_buffer(%value: i8, %memref: memref<16xi8, #spirv.storage_class<StorageBuffer>>, %i0: index) -> i8 {
  //      CHECK:     %[[IDX:.+]] = builtin.unrealized_conversion_cast %{{.*}} : index to i32
  //      CHECK:     %[[MEM:.+]] = builtin.unrealized_conversion_cast %{{.*}} : memref{{.*}} to !spirv.ptr
  //      CHECK:     %[[VAL:.+]] = builtin.unrealized_conversion_cast %{{.*}} : i8 to i32
  //  CHECK-DAG:     %[[C4:.+]] = spirv.Constant 4 : i32
  //  CHECK-DAG:     %[[C8:.+]] = spirv.Constant 8 : i32
  //      CHECK:     %[[MOD:.+]] = spirv.UMod %[[IDX]], %[[C4]]
  //      CHECK:     %[[OFFSET:.+]] = spirv.IMul %[[MOD]], %[[C8]]
  //      CHECK:     %[[DIV:.+]] = spirv.SDiv %[[IDX]], %{{.*}}
  //      CHECK:     %[[AC:.+]] = spirv.AccessChain %[[MEM]][%{{.*}}, %[[DIV]]]
  // Build the AND mask: (val << offset) | ~(0xFF << offset)
  //      CHECK:     %[[C255:.+]] = spirv.Constant 255 : i32
  //      CHECK:     %[[MASKED:.+]] = spirv.BitwiseAnd %[[VAL]], %[[C255]]
  //      CHECK:     %[[SHIFTED:.+]] = spirv.ShiftLeftLogical %[[MASKED]], %[[OFFSET]]
  //      CHECK:     %[[ELEM_SHIFTED:.+]] = spirv.ShiftLeftLogical %[[C255]], %[[OFFSET]]
  //      CHECK:     %[[NOT_ELEM:.+]] = spirv.Not %[[ELEM_SHIFTED]]
  //      CHECK:     %[[MASK:.+]] = spirv.BitwiseOr %[[SHIFTED]], %[[NOT_ELEM]]
  // Atomic AND
  //      CHECK:     %[[ATOMIC:.+]] = spirv.AtomicAnd <Device> <AcquireRelease> %[[AC]], %[[MASK]]
  // Extract old value
  //      CHECK:     spirv.ShiftRightLogical %[[ATOMIC]], %[[OFFSET]]
  //      CHECK:     spirv.BitwiseAnd
  //      CHECK:     return
  %0 = memref.atomic_rmw "andi" %value, %memref[%i0] : (i8, memref<16xi8, #spirv.storage_class<StorageBuffer>>) -> i8
  return %0: i8
}

}
