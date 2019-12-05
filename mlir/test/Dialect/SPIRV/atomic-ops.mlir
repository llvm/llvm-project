// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.AtomicCompareExchangeWeak
//===----------------------------------------------------------------------===//

func @atomic_compare_exchange_weak(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 {
  // CHECK: %{{.*}} = spv.AtomicCompareExchangeWeak "Workgroup" "Release" "Acquire" %{{.*}}, %{{.*}}, %{{.*}} : !spv.ptr<i32, Workgroup>
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
