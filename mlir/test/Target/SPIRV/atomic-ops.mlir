// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK-LABEL: @test_int_atomics
  spv.func @test_int_atomics(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 "None" {
    // CHECK: spv.AtomicCompareExchangeWeak "Workgroup" "Release" "Acquire" %{{.*}}, %{{.*}}, %{{.*}} : !spv.ptr<i32, Workgroup>
    %0 = spv.AtomicCompareExchangeWeak "Workgroup" "Release" "Acquire" %ptr, %value, %comparator: !spv.ptr<i32, Workgroup>
    // CHECK: spv.AtomicAnd "Device" "None" %{{.*}}, %{{.*}} : !spv.ptr<i32, Workgroup>
    %1 = spv.AtomicAnd "Device" "None" %ptr, %value : !spv.ptr<i32, Workgroup>
    // CHECK: spv.AtomicIAdd "Workgroup" "Acquire" %{{.*}}, %{{.*}} : !spv.ptr<i32, Workgroup>
    %2 = spv.AtomicIAdd "Workgroup" "Acquire" %ptr, %value : !spv.ptr<i32, Workgroup>
    // CHECK: spv.AtomicIDecrement "Workgroup" "Acquire" %{{.*}} : !spv.ptr<i32, Workgroup>
    %3 = spv.AtomicIDecrement "Workgroup" "Acquire" %ptr : !spv.ptr<i32, Workgroup>
    // CHECK: spv.AtomicIIncrement "Device" "Release" %{{.*}} : !spv.ptr<i32, Workgroup>
    %4 = spv.AtomicIIncrement "Device" "Release" %ptr : !spv.ptr<i32, Workgroup>
    // CHECK: spv.AtomicISub "Workgroup" "Acquire" %{{.*}}, %{{.*}} : !spv.ptr<i32, Workgroup>
    %5 = spv.AtomicISub "Workgroup" "Acquire" %ptr, %value : !spv.ptr<i32, Workgroup>
    // CHECK: spv.AtomicOr "Workgroup" "AcquireRelease" %{{.*}}, %{{.*}} : !spv.ptr<i32, Workgroup>
    %6 = spv.AtomicOr "Workgroup" "AcquireRelease" %ptr, %value : !spv.ptr<i32, Workgroup>
    // CHECK: spv.AtomicSMax "Subgroup" "None" %{{.*}}, %{{.*}} : !spv.ptr<i32, Workgroup>
    %7 = spv.AtomicSMax "Subgroup" "None" %ptr, %value : !spv.ptr<i32, Workgroup>
    // CHECK: spv.AtomicSMin "Device" "Release" %{{.*}}, %{{.*}} : !spv.ptr<i32, Workgroup>
    %8 = spv.AtomicSMin "Device" "Release" %ptr, %value : !spv.ptr<i32, Workgroup>
    // CHECK: spv.AtomicUMax "Subgroup" "None" %{{.*}}, %{{.*}} : !spv.ptr<i32, Workgroup>
    %9 = spv.AtomicUMax "Subgroup" "None" %ptr, %value : !spv.ptr<i32, Workgroup>
    // CHECK: spv.AtomicUMin "Device" "Release" %{{.*}}, %{{.*}} : !spv.ptr<i32, Workgroup>
    %10 = spv.AtomicUMin "Device" "Release" %ptr, %value : !spv.ptr<i32, Workgroup>
    // CHECK: spv.AtomicXor "Workgroup" "AcquireRelease" %{{.*}}, %{{.*}} : !spv.ptr<i32, Workgroup>
    %11 = spv.AtomicXor "Workgroup" "AcquireRelease" %ptr, %value : !spv.ptr<i32, Workgroup>
    // CHECK: spv.AtomicCompareExchange "Workgroup" "Release" "Acquire" %{{.*}}, %{{.*}}, %{{.*}} : !spv.ptr<i32, Workgroup>
    %12 = spv.AtomicCompareExchange "Workgroup" "Release" "Acquire" %ptr, %value, %comparator: !spv.ptr<i32, Workgroup>
    // CHECK: spv.AtomicExchange "Workgroup" "Release" %{{.*}}, %{{.*}} : !spv.ptr<i32, Workgroup>
    %13 = spv.AtomicExchange "Workgroup" "Release" %ptr, %value: !spv.ptr<i32, Workgroup>
    spv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @test_float_atomics
  spv.func @test_float_atomics(%ptr: !spv.ptr<f32, Workgroup>, %value: f32) -> f32 "None" {
    // CHECK: spv.AtomicFAddEXT "Workgroup" "Acquire" %{{.*}}, %{{.*}} : !spv.ptr<f32, Workgroup>
    %0 = spv.AtomicFAddEXT "Workgroup" "Acquire" %ptr, %value : !spv.ptr<f32, Workgroup>
    spv.ReturnValue %0: f32
  }
}
