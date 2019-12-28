// RUN: mlir-opt -disable-pass-threading -test-spirv-op-availability %s | FileCheck %s

// CHECK-LABEL: iadd
func @iadd(%arg: i32) -> i32 {
  // CHECK: min version: V_1_0
  // CHECK: max version: V_1_5
  // CHECK: extensions: [ ]
  // CHECK: capabilities: [ ]
  %0 = spv.IAdd %arg, %arg: i32
  return %0: i32
}

// CHECK: atomic_compare_exchange_weak
func @atomic_compare_exchange_weak(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 {
  // CHECK: min version: V_1_0
  // CHECK: max version: V_1_3
  // CHECK: extensions: [ ]
  // CHECK: capabilities: [ [Kernel] ]
  %0 = spv.AtomicCompareExchangeWeak "Workgroup" "Release" "Acquire" %ptr, %value, %comparator: !spv.ptr<i32, Workgroup>
  return %0: i32
}

// CHECK-LABEL: subgroup_ballot
func @subgroup_ballot(%predicate: i1) -> vector<4xi32> {
  // CHECK: min version: V_1_3
  // CHECK: max version: V_1_5
  // CHECK: extensions: [ ]
  // CHECK: capabilities: [ [GroupNonUniformBallot] ]
  %0 = spv.GroupNonUniformBallot "Workgroup" %predicate : vector<4xi32>
  return %0: vector<4xi32>
}
