// RUN: mlir-opt -split-input-file -convert-memref-to-spirv %s -o - | FileCheck %s

module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, #spirv.resource_limits<>>} {

//      CHECK: func.func @atomic_addi_storage_buffer
// CHECK-SAME: (%[[VAL:.+]]: i32,
func.func @atomic_addi_storage_buffer(%value: i32, %memref: memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>, %i0: index, %i1: index, %i2: index) -> i32 {
  // CHECK: %[[AC:.+]] = spirv.AccessChain
  // CHECK: %[[ATOMIC:.+]] = spirv.AtomicIAdd "Device" "AcquireRelease" %[[AC]], %[[VAL]] : !spirv.ptr<i32, StorageBuffer>
  // CHECK: return %[[ATOMIC]]
  %0 = memref.atomic_rmw "addi" %value, %memref[%i0, %i1, %i2] : (i32, memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>) -> i32
  return %0: i32
}

//      CHECK: func.func @atomic_maxs_workgroup
// CHECK-SAME: (%[[VAL:.+]]: i32,
func.func @atomic_maxs_workgroup(%value: i32, %memref: memref<2x3x4xi32, #spirv.storage_class<Workgroup>>, %i0: index, %i1: index, %i2: index) -> i32 {
  // CHECK: %[[AC:.+]] = spirv.AccessChain
  // CHECK: %[[ATOMIC:.+]] = spirv.AtomicSMax "Workgroup" "AcquireRelease" %[[AC]], %[[VAL]] : !spirv.ptr<i32, Workgroup>
  // CHECK: return %[[ATOMIC]]
  %0 = memref.atomic_rmw "maxs" %value, %memref[%i0, %i1, %i2] : (i32, memref<2x3x4xi32, #spirv.storage_class<Workgroup>>) -> i32
  return %0: i32
}

//      CHECK: func.func @atomic_maxu_storage_buffer
// CHECK-SAME: (%[[VAL:.+]]: i32,
func.func @atomic_maxu_storage_buffer(%value: i32, %memref: memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>, %i0: index, %i1: index, %i2: index) -> i32 {
  // CHECK: %[[AC:.+]] = spirv.AccessChain
  // CHECK: %[[ATOMIC:.+]] = spirv.AtomicUMax "Device" "AcquireRelease" %[[AC]], %[[VAL]] : !spirv.ptr<i32, StorageBuffer>
  // CHECK: return %[[ATOMIC]]
  %0 = memref.atomic_rmw "maxu" %value, %memref[%i0, %i1, %i2] : (i32, memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>) -> i32
  return %0: i32
}

//      CHECK: func.func @atomic_mins_workgroup
// CHECK-SAME: (%[[VAL:.+]]: i32,
func.func @atomic_mins_workgroup(%value: i32, %memref: memref<2x3x4xi32, #spirv.storage_class<Workgroup>>, %i0: index, %i1: index, %i2: index) -> i32 {
  // CHECK: %[[AC:.+]] = spirv.AccessChain
  // CHECK: %[[ATOMIC:.+]] = spirv.AtomicSMin "Workgroup" "AcquireRelease" %[[AC]], %[[VAL]] : !spirv.ptr<i32, Workgroup>
  // CHECK: return %[[ATOMIC]]
  %0 = memref.atomic_rmw "mins" %value, %memref[%i0, %i1, %i2] : (i32, memref<2x3x4xi32, #spirv.storage_class<Workgroup>>) -> i32
  return %0: i32
}

//      CHECK: func.func @atomic_minu_storage_buffer
// CHECK-SAME: (%[[VAL:.+]]: i32,
func.func @atomic_minu_storage_buffer(%value: i32, %memref: memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>, %i0: index, %i1: index, %i2: index) -> i32 {
  // CHECK: %[[AC:.+]] = spirv.AccessChain
  // CHECK: %[[ATOMIC:.+]] = spirv.AtomicUMin "Device" "AcquireRelease" %[[AC]], %[[VAL]] : !spirv.ptr<i32, StorageBuffer>
  // CHECK: return %[[ATOMIC]]
  %0 = memref.atomic_rmw "minu" %value, %memref[%i0, %i1, %i2] : (i32, memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>) -> i32
  return %0: i32
}

//      CHECK: func.func @atomic_ori_workgroup
// CHECK-SAME: (%[[VAL:.+]]: i32,
func.func @atomic_ori_workgroup(%value: i32, %memref: memref<2x3x4xi32, #spirv.storage_class<Workgroup>>, %i0: index, %i1: index, %i2: index) -> i32 {
  // CHECK: %[[AC:.+]] = spirv.AccessChain
  // CHECK: %[[ATOMIC:.+]] = spirv.AtomicOr "Workgroup" "AcquireRelease" %[[AC]], %[[VAL]] : !spirv.ptr<i32, Workgroup>
  // CHECK: return %[[ATOMIC]]
  %0 = memref.atomic_rmw "ori" %value, %memref[%i0, %i1, %i2] : (i32, memref<2x3x4xi32, #spirv.storage_class<Workgroup>>) -> i32
  return %0: i32
}

//      CHECK: func.func @atomic_andi_storage_buffer
// CHECK-SAME: (%[[VAL:.+]]: i32,
func.func @atomic_andi_storage_buffer(%value: i32, %memref: memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>, %i0: index, %i1: index, %i2: index) -> i32 {
  // CHECK: %[[AC:.+]] = spirv.AccessChain
  // CHECK: %[[ATOMIC:.+]] = spirv.AtomicAnd "Device" "AcquireRelease" %[[AC]], %[[VAL]] : !spirv.ptr<i32, StorageBuffer>
  // CHECK: return %[[ATOMIC]]
  %0 = memref.atomic_rmw "andi" %value, %memref[%i0, %i1, %i2] : (i32, memref<2x3x4xi32, #spirv.storage_class<StorageBuffer>>) -> i32
  return %0: i32
}

}

