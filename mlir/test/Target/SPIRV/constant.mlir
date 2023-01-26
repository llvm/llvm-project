// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> {
  // CHECK-LABEL: @bool_const
  spirv.func @bool_const() -> () "None" {
    // CHECK: spirv.Constant true
    %0 = spirv.Constant true
    // CHECK: spirv.Constant false
    %1 = spirv.Constant false

    %2 = spirv.Variable init(%0): !spirv.ptr<i1, Function>
    %3 = spirv.Variable init(%1): !spirv.ptr<i1, Function>
    spirv.Return
  }

  // CHECK-LABEL: @i32_const
  spirv.func @i32_const() -> () "None" {
    // CHECK: spirv.Constant 0 : i32
    %0 = spirv.Constant  0 : i32
    // CHECK: spirv.Constant 10 : i32
    %1 = spirv.Constant 10 : i32
    // CHECK: spirv.Constant -5 : i32
    %2 = spirv.Constant -5 : i32

    %3 = spirv.IAdd %0, %1 : i32
    %4 = spirv.IAdd %2, %3 : i32
    spirv.Return
  }

  // CHECK-LABEL: @si32_const
  spirv.func @si32_const() -> () "None" {
    // CHECK: spirv.Constant 0 : si32
    %0 = spirv.Constant  0 : si32
    // CHECK: spirv.Constant 10 : si32
    %1 = spirv.Constant 10 : si32
    // CHECK: spirv.Constant -5 : si32
    %2 = spirv.Constant -5 : si32

    %3 = spirv.IAdd %0, %1 : si32
    %4 = spirv.IAdd %2, %3 : si32
    spirv.Return
  }

  // CHECK-LABEL: @ui32_const
  // We cannot differentiate signless vs. unsigned integers in SPIR-V blob
  // because they all use 1 as the signedness bit. So we always treat them
  // as signless integers.
  spirv.func @ui32_const() -> () "None" {
    // CHECK: spirv.Constant 0 : i32
    %0 = spirv.Constant  0 : ui32
    // CHECK: spirv.Constant 10 : i32
    %1 = spirv.Constant 10 : ui32
    // CHECK: spirv.Constant -5 : i32
    %2 = spirv.Constant 4294967291 : ui32

    %3 = spirv.IAdd %0, %1 : ui32
    %4 = spirv.IAdd %2, %3 : ui32
    spirv.Return
  }

  // CHECK-LABEL: @i64_const
  spirv.func @i64_const() -> () "None" {
    // CHECK: spirv.Constant 4294967296 : i64
    %0 = spirv.Constant           4294967296 : i64 //  2^32
    // CHECK: spirv.Constant -4294967296 : i64
    %1 = spirv.Constant          -4294967296 : i64 // -2^32
    // CHECK: spirv.Constant 9223372036854775807 : i64
    %2 = spirv.Constant  9223372036854775807 : i64 //  2^63 - 1
    // CHECK: spirv.Constant -9223372036854775808 : i64
    %3 = spirv.Constant -9223372036854775808 : i64 // -2^63

    %4 = spirv.IAdd %0, %1 : i64
    %5 = spirv.IAdd %2, %3 : i64
    spirv.Return
  }

  // CHECK-LABEL: @i16_const
  spirv.func @i16_const() -> () "None" {
    // CHECK: spirv.Constant -32768 : i16
    %0 = spirv.Constant -32768 : i16 // -2^15
    // CHECK: spirv.Constant 32767 : i16
    %1 = spirv.Constant 32767 : i16 //  2^15 - 1

    %2 = spirv.IAdd %0, %1 : i16
    spirv.Return
  }

  // CHECK-LABEL: @i8_const
  spirv.func @i8_const() -> () "None" {
    // CHECK: spirv.Constant 0 : i8
    %0 = spirv.Constant 0 : i8
    // CHECK: spirv.Constant -1 : i8
    %1 = spirv.Constant 255 : i8

    // CHECK: spirv.Constant 0 : si8
    %2 = spirv.Constant 0 : si8
    // CHECK: spirv.Constant 127 : si8
    %3 = spirv.Constant 127 : si8
    // CHECK: spirv.Constant -128 : si8
    %4 = spirv.Constant -128 : si8

    // CHECK: spirv.Constant 0 : i8
    %5 = spirv.Constant 0 : ui8
    // CHECK: spirv.Constant -1 : i8
    %6 = spirv.Constant 255 : ui8

    %10 = spirv.IAdd %0, %1: i8
    %11 = spirv.IAdd %2, %3: si8
    %12 = spirv.IAdd %3, %4: si8
    %13 = spirv.IAdd %5, %6: ui8
    spirv.Return
  }

  // CHECK-LABEL: @float_const
  spirv.func @float_const() -> () "None" {
    // CHECK: spirv.Constant 0.000000e+00 : f32
    %0 = spirv.Constant 0. : f32
    // CHECK: spirv.Constant 1.000000e+00 : f32
    %1 = spirv.Constant 1. : f32
    // CHECK: spirv.Constant -0.000000e+00 : f32
    %2 = spirv.Constant -0. : f32
    // CHECK: spirv.Constant -1.000000e+00 : f32
    %3 = spirv.Constant -1. : f32
    // CHECK: spirv.Constant 7.500000e-01 : f32
    %4 = spirv.Constant 0.75 : f32
    // CHECK: spirv.Constant -2.500000e-01 : f32
    %5 = spirv.Constant -0.25 : f32

    %6 = spirv.FAdd %0, %1 : f32
    %7 = spirv.FAdd %2, %3 : f32
    %8 = spirv.FAdd %4, %5 : f32
    spirv.Return
  }

  // CHECK-LABEL: @double_const
  spirv.func @double_const() -> () "None" {
    // TODO: test range boundary values
    // CHECK: spirv.Constant 1.024000e+03 : f64
    %0 = spirv.Constant 1024. : f64
    // CHECK: spirv.Constant -1.024000e+03 : f64
    %1 = spirv.Constant -1024. : f64

    %2 = spirv.FAdd %0, %1 : f64
    spirv.Return
  }

  // CHECK-LABEL: @half_const
  spirv.func @half_const() -> () "None" {
    // CHECK: spirv.Constant 5.120000e+02 : f16
    %0 = spirv.Constant 512. : f16
    // CHECK: spirv.Constant -5.120000e+02 : f16
    %1 = spirv.Constant -512. : f16

    %2 = spirv.FAdd %0, %1 : f16
    spirv.Return
  }

  // CHECK-LABEL: @bool_vector_const
  spirv.func @bool_vector_const() -> () "None" {
    // CHECK: spirv.Constant dense<false> : vector<2xi1>
    %0 = spirv.Constant dense<false> : vector<2xi1>
    // CHECK: spirv.Constant dense<true> : vector<3xi1>
    %1 = spirv.Constant dense<true> : vector<3xi1>
    // CHECK: spirv.Constant dense<[false, true]> : vector<2xi1>
    %2 = spirv.Constant dense<[false, true]> : vector<2xi1>

    %3 = spirv.Variable init(%0): !spirv.ptr<vector<2xi1>, Function>
    %4 = spirv.Variable init(%1): !spirv.ptr<vector<3xi1>, Function>
    %5 = spirv.Variable init(%2): !spirv.ptr<vector<2xi1>, Function>
    spirv.Return
  }

  // CHECK-LABEL: @int_vector_const
  spirv.func @int_vector_const() -> () "None" {
    // CHECK: spirv.Constant dense<0> : vector<3xi32>
    %0 = spirv.Constant dense<0> : vector<3xi32>
    // CHECK: spirv.Constant dense<1> : vector<3xi32>
    %1 = spirv.Constant dense<1> : vector<3xi32>
    // CHECK: spirv.Constant dense<[2, -3, 4]> : vector<3xi32>
    %2 = spirv.Constant dense<[2, -3, 4]> : vector<3xi32>

    %3 = spirv.IAdd %0, %1 : vector<3xi32>
    %4 = spirv.IAdd %2, %3 : vector<3xi32>
    spirv.Return
  }

  // CHECK-LABEL: @fp_vector_const
  spirv.func @fp_vector_const() -> () "None" {
    // CHECK: spirv.Constant dense<0.000000e+00> : vector<4xf32>
    %0 = spirv.Constant dense<0.> : vector<4xf32>
    // CHECK: spirv.Constant dense<-1.500000e+01> : vector<4xf32>
    %1 = spirv.Constant dense<-15.> : vector<4xf32>
    // CHECK: spirv.Constant dense<[7.500000e-01, -2.500000e-01, 1.000000e+01, 4.200000e+01]> : vector<4xf32>
    %2 = spirv.Constant dense<[0.75, -0.25, 10., 42.]> : vector<4xf32>

    %3 = spirv.FAdd %0, %1 : vector<4xf32>
    %4 = spirv.FAdd %2, %3 : vector<4xf32>
    spirv.Return
  }

  // CHECK-LABEL: @ui64_array_const
  spirv.func @ui64_array_const() -> (!spirv.array<3xui64>) "None" {
    // CHECK: spirv.Constant [5, 6, 7] : !spirv.array<3 x i64>
    %0 = spirv.Constant [5 : ui64, 6 : ui64, 7 : ui64] : !spirv.array<3 x ui64>

    spirv.ReturnValue %0: !spirv.array<3xui64>
  }

  // CHECK-LABEL: @si32_array_const
  spirv.func @si32_array_const() -> (!spirv.array<3xsi32>) "None" {
    // CHECK: spirv.Constant [5 : si32, 6 : si32, 7 : si32] : !spirv.array<3 x si32>
    %0 = spirv.Constant [5 : si32, 6 : si32, 7 : si32] : !spirv.array<3 x si32>

    spirv.ReturnValue %0 : !spirv.array<3xsi32>
  }
  // CHECK-LABEL: @float_array_const
  spirv.func @float_array_const() -> (!spirv.array<2 x vector<2xf32>>) "None" {
    // CHECK: spirv.Constant [dense<3.000000e+00> : vector<2xf32>, dense<[4.000000e+00, 5.000000e+00]> : vector<2xf32>] : !spirv.array<2 x vector<2xf32>>
    %0 = spirv.Constant [dense<3.0> : vector<2xf32>, dense<[4., 5.]> : vector<2xf32>] : !spirv.array<2 x vector<2xf32>>

    spirv.ReturnValue %0 : !spirv.array<2 x vector<2xf32>>
  }

  // CHECK-LABEL: @ignore_not_used_const
  spirv.func @ignore_not_used_const() -> () "None" {
    %0 = spirv.Constant false
    // CHECK-NEXT: spirv.Return
    spirv.Return
  }

  // CHECK-LABEL: @materialize_const_at_each_use
  spirv.func @materialize_const_at_each_use() -> (i32) "None" {
    // CHECK: %[[USE1:.*]] = spirv.Constant 42 : i32
    // CHECK: %[[USE2:.*]] = spirv.Constant 42 : i32
    // CHECK: spirv.IAdd %[[USE1]], %[[USE2]]
    %0 = spirv.Constant 42 : i32
    %1 = spirv.IAdd %0, %0 : i32
    spirv.ReturnValue %1 : i32
  }

  // CHECK-LABEL: @const_variable
  spirv.func @const_variable(%arg0 : i32, %arg1 : i32) -> () "None" {
    // CHECK: %[[CONST:.*]] = spirv.Constant 5 : i32
    // CHECK: spirv.Variable init(%[[CONST]]) : !spirv.ptr<i32, Function>
    // CHECK: spirv.IAdd %arg0, %arg1
    %0 = spirv.IAdd %arg0, %arg1 : i32
    %1 = spirv.Constant 5 : i32
    %2 = spirv.Variable init(%1) : !spirv.ptr<i32, Function>
    %3 = spirv.Load "Function" %2 : i32
    %4 = spirv.IAdd %0, %3 : i32
    spirv.Return
  }

  // CHECK-LABEL: @multi_dimensions_const
  spirv.func @multi_dimensions_const() -> (!spirv.array<2 x !spirv.array<2 x !spirv.array<3 x i32, stride=4>, stride=12>, stride=24>) "None" {
    // CHECK: spirv.Constant {{\[}}{{\[}}[1 : i32, 2 : i32, 3 : i32], [4 : i32, 5 : i32, 6 : i32]], {{\[}}[7 : i32, 8 : i32, 9 : i32], [10 : i32, 11 : i32, 12 : i32]]] : !spirv.array<2 x !spirv.array<2 x !spirv.array<3 x i32, stride=4>, stride=12>, stride=24>
    %0 = spirv.Constant dense<[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]> : tensor<2x2x3xi32> : !spirv.array<2 x !spirv.array<2 x !spirv.array<3 x i32, stride=4>, stride=12>, stride=24>
    spirv.ReturnValue %0 : !spirv.array<2 x !spirv.array<2 x !spirv.array<3 x i32, stride=4>, stride=12>, stride=24>
  }

  // CHECK-LABEL: @multi_dimensions_splat_const
  spirv.func @multi_dimensions_splat_const() -> (!spirv.array<2 x !spirv.array<2 x !spirv.array<3 x i32, stride=4>, stride=12>, stride=24>) "None" {
    // CHECK: spirv.Constant {{\[}}{{\[}}[1 : i32, 1 : i32, 1 : i32], [1 : i32, 1 : i32, 1 : i32]], {{\[}}[1 : i32, 1 : i32, 1 : i32], [1 : i32, 1 : i32, 1 : i32]]] : !spirv.array<2 x !spirv.array<2 x !spirv.array<3 x i32, stride=4>, stride=12>, stride=24>
    %0 = spirv.Constant dense<1> : tensor<2x2x3xi32> : !spirv.array<2 x !spirv.array<2 x !spirv.array<3 x i32, stride=4>, stride=12>, stride=24>
    spirv.ReturnValue %0 : !spirv.array<2 x !spirv.array<2 x !spirv.array<3 x i32, stride=4>, stride=12>, stride=24>
  }
}
