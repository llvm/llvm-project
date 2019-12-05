// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  // CHECK-LABEL: @atomic_compare_exchange_weak
  func @atomic_compare_exchange_weak(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 {
    // CHECK: %{{.*}} = spv.AtomicCompareExchangeWeak "Workgroup" "Release" "Acquire" %{{.*}}, %{{.*}}, %{{.*}} : !spv.ptr<i32, Workgroup>
    %0 = spv.AtomicCompareExchangeWeak "Workgroup" "Release" "Acquire" %ptr, %value, %comparator: !spv.ptr<i32, Workgroup>
    spv.ReturnValue %0: i32
  }
}
