// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK-LABEL: @test_int_atomics
  spirv.func @test_int_atomics(%ptr: !spirv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 "None" {
    // CHECK: spirv.AtomicCompareExchangeWeak "Workgroup" "Release" "Acquire" %{{.*}}, %{{.*}}, %{{.*}} : !spirv.ptr<i32, Workgroup>
    %0 = spirv.AtomicCompareExchangeWeak "Workgroup" "Release" "Acquire" %ptr, %value, %comparator: !spirv.ptr<i32, Workgroup>
    // CHECK: spirv.AtomicAnd "Device" "None" %{{.*}}, %{{.*}} : !spirv.ptr<i32, Workgroup>
    %1 = spirv.AtomicAnd "Device" "None" %ptr, %value : !spirv.ptr<i32, Workgroup>
    // CHECK: spirv.AtomicIAdd "Workgroup" "Acquire" %{{.*}}, %{{.*}} : !spirv.ptr<i32, Workgroup>
    %2 = spirv.AtomicIAdd "Workgroup" "Acquire" %ptr, %value : !spirv.ptr<i32, Workgroup>
    // CHECK: spirv.AtomicIDecrement "Workgroup" "Acquire" %{{.*}} : !spirv.ptr<i32, Workgroup>
    %3 = spirv.AtomicIDecrement "Workgroup" "Acquire" %ptr : !spirv.ptr<i32, Workgroup>
    // CHECK: spirv.AtomicIIncrement "Device" "Release" %{{.*}} : !spirv.ptr<i32, Workgroup>
    %4 = spirv.AtomicIIncrement "Device" "Release" %ptr : !spirv.ptr<i32, Workgroup>
    // CHECK: spirv.AtomicISub "Workgroup" "Acquire" %{{.*}}, %{{.*}} : !spirv.ptr<i32, Workgroup>
    %5 = spirv.AtomicISub "Workgroup" "Acquire" %ptr, %value : !spirv.ptr<i32, Workgroup>
    // CHECK: spirv.AtomicOr "Workgroup" "AcquireRelease" %{{.*}}, %{{.*}} : !spirv.ptr<i32, Workgroup>
    %6 = spirv.AtomicOr "Workgroup" "AcquireRelease" %ptr, %value : !spirv.ptr<i32, Workgroup>
    // CHECK: spirv.AtomicSMax "Subgroup" "None" %{{.*}}, %{{.*}} : !spirv.ptr<i32, Workgroup>
    %7 = spirv.AtomicSMax "Subgroup" "None" %ptr, %value : !spirv.ptr<i32, Workgroup>
    // CHECK: spirv.AtomicSMin "Device" "Release" %{{.*}}, %{{.*}} : !spirv.ptr<i32, Workgroup>
    %8 = spirv.AtomicSMin "Device" "Release" %ptr, %value : !spirv.ptr<i32, Workgroup>
    // CHECK: spirv.AtomicUMax "Subgroup" "None" %{{.*}}, %{{.*}} : !spirv.ptr<i32, Workgroup>
    %9 = spirv.AtomicUMax "Subgroup" "None" %ptr, %value : !spirv.ptr<i32, Workgroup>
    // CHECK: spirv.AtomicUMin "Device" "Release" %{{.*}}, %{{.*}} : !spirv.ptr<i32, Workgroup>
    %10 = spirv.AtomicUMin "Device" "Release" %ptr, %value : !spirv.ptr<i32, Workgroup>
    // CHECK: spirv.AtomicXor "Workgroup" "AcquireRelease" %{{.*}}, %{{.*}} : !spirv.ptr<i32, Workgroup>
    %11 = spirv.AtomicXor "Workgroup" "AcquireRelease" %ptr, %value : !spirv.ptr<i32, Workgroup>
    // CHECK: spirv.AtomicCompareExchange "Workgroup" "Release" "Acquire" %{{.*}}, %{{.*}}, %{{.*}} : !spirv.ptr<i32, Workgroup>
    %12 = spirv.AtomicCompareExchange "Workgroup" "Release" "Acquire" %ptr, %value, %comparator: !spirv.ptr<i32, Workgroup>
    // CHECK: spirv.AtomicExchange "Workgroup" "Release" %{{.*}}, %{{.*}} : !spirv.ptr<i32, Workgroup>
    %13 = spirv.AtomicExchange "Workgroup" "Release" %ptr, %value: !spirv.ptr<i32, Workgroup>
    spirv.ReturnValue %0: i32
  }

  // CHECK-LABEL: @test_float_atomics
  spirv.func @test_float_atomics(%ptr: !spirv.ptr<f32, Workgroup>, %value: f32) -> f32 "None" {
    // CHECK: spirv.EXT.AtomicFAdd "Workgroup" "Acquire" %{{.*}}, %{{.*}} : !spirv.ptr<f32, Workgroup>
    %0 = spirv.EXT.AtomicFAdd "Workgroup" "Acquire" %ptr, %value : !spirv.ptr<f32, Workgroup>
    spirv.ReturnValue %0: f32
  }
}
