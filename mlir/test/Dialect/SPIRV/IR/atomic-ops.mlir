// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.AtomicAnd
//===----------------------------------------------------------------------===//

func @atomic_and(%ptr : !spv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spv.AtomicAnd "Device" "None" %{{.*}}, %{{.*}} : !spv.ptr<i32, StorageBuffer>
  %0 = spv.AtomicAnd "Device" "None" %ptr, %value : !spv.ptr<i32, StorageBuffer>
  return %0 : i32
}

// -----

func @atomic_and(%ptr : !spv.ptr<f32, StorageBuffer>, %value : i32) -> i32 {
  // expected-error @+1 {{pointer operand must point to an integer value, found 'f32'}}
  %0 = "spv.AtomicAnd"(%ptr, %value) {memory_scope = 4: i32, semantics = 0x4 : i32} : (!spv.ptr<f32, StorageBuffer>, i32) -> (i32)
  return %0 : i32
}


// -----

func @atomic_and(%ptr : !spv.ptr<i32, StorageBuffer>, %value : i64) -> i64 {
  // expected-error @+1 {{expected value to have the same type as the pointer operand's pointee type 'i32', but found 'i64'}}
  %0 = "spv.AtomicAnd"(%ptr, %value) {memory_scope = 2: i32, semantics = 0x8 : i32} : (!spv.ptr<i32, StorageBuffer>, i64) -> (i64)
  return %0 : i64
}

// -----

func @atomic_and(%ptr : !spv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // expected-error @+1 {{expected at most one of these four memory constraints to be set: `Acquire`, `Release`,`AcquireRelease` or `SequentiallyConsistent`}}
  %0 = spv.AtomicAnd "Device" "Acquire|Release" %ptr, %value : !spv.ptr<i32, StorageBuffer>
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.AtomicCompareExchangeWeak
//===----------------------------------------------------------------------===//

func @atomic_compare_exchange_weak(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 {
  // CHECK: spv.AtomicCompareExchangeWeak "Workgroup" "Release" "Acquire" %{{.*}}, %{{.*}}, %{{.*}} : !spv.ptr<i32, Workgroup>
  %0 = spv.AtomicCompareExchangeWeak "Workgroup" "Release" "Acquire" %ptr, %value, %comparator: !spv.ptr<i32, Workgroup>
  return %0: i32
}

// -----

func @atomic_compare_exchange_weak(%ptr: !spv.ptr<i32, Workgroup>, %value: i64, %comparator: i32) -> i32 {
  // expected-error @+1 {{value operand must have the same type as the op result, but found 'i64' vs 'i32'}}
  %0 = "spv.AtomicCompareExchangeWeak"(%ptr, %value, %comparator) {memory_scope = 4: i32, equal_semantics = 0x4: i32, unequal_semantics = 0x2:i32} : (!spv.ptr<i32, Workgroup>, i64, i32) -> (i32)
  return %0: i32
}

// -----

func @atomic_compare_exchange_weak(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i16) -> i32 {
  // expected-error @+1 {{comparator operand must have the same type as the op result, but found 'i16' vs 'i32'}}
  %0 = "spv.AtomicCompareExchangeWeak"(%ptr, %value, %comparator) {memory_scope = 4: i32, equal_semantics = 0x4: i32, unequal_semantics = 0x2:i32} : (!spv.ptr<i32, Workgroup>, i32, i16) -> (i32)
  return %0: i32
}

// -----

func @atomic_compare_exchange_weak(%ptr: !spv.ptr<i64, Workgroup>, %value: i32, %comparator: i32) -> i32 {
  // expected-error @+1 {{pointer operand's pointee type must have the same as the op result type, but found 'i64' vs 'i32'}}
  %0 = "spv.AtomicCompareExchangeWeak"(%ptr, %value, %comparator) {memory_scope = 4: i32, equal_semantics = 0x4: i32, unequal_semantics = 0x2:i32} : (!spv.ptr<i64, Workgroup>, i32, i32) -> (i32)
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.AtomicIAdd
//===----------------------------------------------------------------------===//

func @atomic_iadd(%ptr : !spv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spv.AtomicIAdd "Workgroup" "None" %{{.*}}, %{{.*}} : !spv.ptr<i32, StorageBuffer>
  %0 = spv.AtomicIAdd "Workgroup" "None" %ptr, %value : !spv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spv.AtomicIDecrement
//===----------------------------------------------------------------------===//

func @atomic_idecrement(%ptr : !spv.ptr<i32, StorageBuffer>) -> i32 {
  // CHECK: spv.AtomicIDecrement "Workgroup" "None" %{{.*}} : !spv.ptr<i32, StorageBuffer>
  %0 = spv.AtomicIDecrement "Workgroup" "None" %ptr : !spv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spv.AtomicIIncrement
//===----------------------------------------------------------------------===//

func @atomic_iincrement(%ptr : !spv.ptr<i32, StorageBuffer>) -> i32 {
  // CHECK: spv.AtomicIIncrement "Workgroup" "None" %{{.*}} : !spv.ptr<i32, StorageBuffer>
  %0 = spv.AtomicIIncrement "Workgroup" "None" %ptr : !spv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spv.AtomicISub
//===----------------------------------------------------------------------===//

func @atomic_isub(%ptr : !spv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spv.AtomicISub "Workgroup" "None" %{{.*}}, %{{.*}} : !spv.ptr<i32, StorageBuffer>
  %0 = spv.AtomicISub "Workgroup" "None" %ptr, %value : !spv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spv.AtomicOr
//===----------------------------------------------------------------------===//

func @atomic_or(%ptr : !spv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spv.AtomicOr "Workgroup" "None" %{{.*}}, %{{.*}} : !spv.ptr<i32, StorageBuffer>
  %0 = spv.AtomicOr "Workgroup" "None" %ptr, %value : !spv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spv.AtomicSMax
//===----------------------------------------------------------------------===//

func @atomic_smax(%ptr : !spv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spv.AtomicSMax "Workgroup" "None" %{{.*}}, %{{.*}} : !spv.ptr<i32, StorageBuffer>
  %0 = spv.AtomicSMax "Workgroup" "None" %ptr, %value : !spv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spv.AtomicSMin
//===----------------------------------------------------------------------===//

func @atomic_smin(%ptr : !spv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spv.AtomicSMin "Workgroup" "None" %{{.*}}, %{{.*}} : !spv.ptr<i32, StorageBuffer>
  %0 = spv.AtomicSMin "Workgroup" "None" %ptr, %value : !spv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spv.AtomicUMax
//===----------------------------------------------------------------------===//

func @atomic_umax(%ptr : !spv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spv.AtomicUMax "Workgroup" "None" %{{.*}}, %{{.*}} : !spv.ptr<i32, StorageBuffer>
  %0 = spv.AtomicUMax "Workgroup" "None" %ptr, %value : !spv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spv.AtomicUMin
//===----------------------------------------------------------------------===//

func @atomic_umin(%ptr : !spv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spv.AtomicUMin "Workgroup" "None" %{{.*}}, %{{.*}} : !spv.ptr<i32, StorageBuffer>
  %0 = spv.AtomicUMin "Workgroup" "None" %ptr, %value : !spv.ptr<i32, StorageBuffer>
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spv.AtomicXor
//===----------------------------------------------------------------------===//

func @atomic_xor(%ptr : !spv.ptr<i32, StorageBuffer>, %value : i32) -> i32 {
  // CHECK: spv.AtomicXor "Workgroup" "None" %{{.*}}, %{{.*}} : !spv.ptr<i32, StorageBuffer>
  %0 = spv.AtomicXor "Workgroup" "None" %ptr, %value : !spv.ptr<i32, StorageBuffer>
  return %0 : i32
}
