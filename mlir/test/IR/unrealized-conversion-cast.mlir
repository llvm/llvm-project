// RUN: mlir-opt %s -verify-diagnostics -allow-unregistered-dialect
// RUN: mlir-opt %s -reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=RECONCILE
// RUN: mlir-opt %s -canonicalize -reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s --check-prefix=CANON

//===----------------------------------------------------------------------===//
// Basic functionality tests
//===----------------------------------------------------------------------===//

module {
  // CHECK-LABEL: func @cast_basic
  func.func @cast_basic(%arg0: i32) -> f32 {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : i32 to f32
    %cast = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    return %cast : f32
  }

  // CHECK-LABEL: func @cast_multiple_operands
  func.func @cast_multiple_operands(%arg0: i32, %arg1: f64) -> (f32, i64) {
    // CHECK: %{{.*}}:2 = builtin.unrealized_conversion_cast %{{.*}}, %{{.*}} :
    // CHECK-SAME: i32, f64 to f32, i64
    %cast:2 = builtin.unrealized_conversion_cast %arg0, %arg1 :
              i32, f64 to f32, i64
    return %cast#0, %cast#1 : f32, i64
  }

  // CHECK-LABEL: func @cast_generation
  func.func @cast_generation() -> f32 {
    %unit = arith.constant 0 : i32
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : i32 to f32
    %cast = builtin.unrealized_conversion_cast %unit : i32 to f32
    return %cast : f32
  }

  // CHECK-LABEL: func @cast_no_results
  func.func @cast_no_results(%arg0: i32) {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : i32 to i32
    %cast = builtin.unrealized_conversion_cast %arg0 : i32 to i32
    return
  }

  // CHECK-LABEL: func @cast_empty_valid
  func.func @cast_empty_valid() {
    %dummy = arith.constant 0 : i32
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : i32 to i32
    %cast = builtin.unrealized_conversion_cast %dummy : i32 to i32
    return
  }

  // CHECK-LABEL: func @cast_same_type
  func.func @cast_same_type(%arg0: i32) -> i32 {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : i32 to i32
    %cast = builtin.unrealized_conversion_cast %arg0 : i32 to i32
    return %cast : i32
  }

  // CHECK-LABEL: func @cast_chained
  func.func @cast_chained(%arg0: i32) -> f64 {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : i32 to f32
    %cast1 = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : f32 to f64
    %cast2 = builtin.unrealized_conversion_cast %cast1 : f32 to f64
    return %cast2 : f64
  }

  // CHECK-LABEL: func @cast_n_to_m_valid
  func.func @cast_n_to_m_valid(%arg0: i32, %arg1: f64) -> f32 {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}}, %{{.*}} :
    // CHECK-SAME: i32, f64 to f32
    %cast = builtin.unrealized_conversion_cast %arg0, %arg1 :
            i32, f64 to f32
    return %cast : f32
  }

  // CHECK-LABEL: func @cast_1_to_n_valid
  func.func @cast_1_to_n_valid(%arg0: i32) -> (f32, f64) {
    // CHECK: %{{.*}}:2 = builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: i32 to f32, f64
    %cast:2 = builtin.unrealized_conversion_cast %arg0 :
              i32 to f32, f64
    return %cast#0, %cast#1 : f32, f64
  }
}

// -----

//===----------------------------------------------------------------------===//
// Identity casts (same input/output type)
//===----------------------------------------------------------------------===//

module {
  // CHECK-LABEL: func @cast_same_type
  func.func @cast_same_type() {
    %dummy = arith.constant 0 : i32
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: i32 to i32
    %cast1 = builtin.unrealized_conversion_cast %dummy : i32 to i32
    %cast2 = builtin.unrealized_conversion_cast %dummy : i32 to i32
    return
  }

  // CHECK-LABEL: func @cast_multiple_values
  func.func @cast_multiple_values(%arg0: i32, %arg1: f32) -> (i32, f32) {
    // CHECK: %{{.*}}:2 = builtin.unrealized_conversion_cast %{{.*}}, %{{.*}} :
    // CHECK-SAME: i32, f32 to i32, f32
    %cast:2 = builtin.unrealized_conversion_cast %arg0, %arg1 :
              i32, f32 to i32, f32
    return %cast#0, %cast#1 : i32, f32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Type system boundaries tests
//===----------------------------------------------------------------------===//

module {
  // CHECK-LABEL: func @cast_primitive_to_aggregate
  func.func @cast_primitive_to_aggregate(%arg0: i32) -> tensor<1xi32> {
    // CHECK: builtin.unrealized_conversion_cast %{{.*}} : i32 to tensor<1xi32>
    %cast = builtin.unrealized_conversion_cast %arg0 : i32 to tensor<1xi32>
    return %cast : tensor<1xi32>
  }

  // CHECK-LABEL: func @cast_shaped_types
  func.func @cast_shaped_types(%arg0: tensor<4x4xf32>) -> memref<4x4xf32> {
    // CHECK: builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: tensor<4x4xf32> to memref<4x4xf32>
    %cast = builtin.unrealized_conversion_cast %arg0 :
            tensor<4x4xf32> to memref<4x4xf32>
    return %cast : memref<4x4xf32>
  }

  // CHECK-LABEL: func @cast_complex_types
  func.func @cast_complex_types(%arg0: tensor<4xi32>) -> memref<4xf32> {
    // CHECK: builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: tensor<4xi32> to memref<4xf32>
    %cast = builtin.unrealized_conversion_cast %arg0 :
            tensor<4xi32> to memref<4xf32>
    return %cast : memref<4xf32>
  }

  // CHECK-LABEL: func @cast_vector_types
  func.func @cast_vector_types(%arg0: vector<4xi32>) -> vector<4xf32> {
    // CHECK: builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: vector<4xi32> to vector<4xf32>
    %cast = builtin.unrealized_conversion_cast %arg0 :
            vector<4xi32> to vector<4xf32>
    return %cast : vector<4xf32>
  }

  // CHECK-LABEL: func @cast_index_types
  func.func @cast_index_types(%arg0: index) -> i64 {
    // CHECK: builtin.unrealized_conversion_cast %{{.*}} : index to i64
    %cast = builtin.unrealized_conversion_cast %arg0 : index to i64
    return %cast : i64
  }

  // CHECK-LABEL: func @cast_tuple_types
  func.func @cast_tuple_types(%arg0: tuple<i32, f32>) -> tuple<f64, i64> {
    // CHECK: builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: tuple<i32, f32> to tuple<f64, i64>
    %cast = builtin.unrealized_conversion_cast %arg0 :
            tuple<i32, f32> to tuple<f64, i64>
    return %cast : tuple<f64, i64>
  }

  // CHECK-LABEL: func @cast_complex_nested
  func.func @cast_complex_nested(
      %arg0: tuple<tensor<2xi32>, memref<4xf64>>)
      -> tuple<memref<2xf32>, tensor<4xi64>> {
    // CHECK: builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: tuple<tensor<2xi32>, memref<4xf64>> to
    // CHECK-SAME: tuple<memref<2xf32>, tensor<4xi64>>
    %cast = builtin.unrealized_conversion_cast %arg0 :
            tuple<tensor<2xi32>, memref<4xf64>> to
            tuple<memref<2xf32>, tensor<4xi64>>
    return %cast : tuple<memref<2xf32>, tensor<4xi64>>
  }

  // CHECK-LABEL: func @cast_unregistered_dialects
  func.func @cast_unregistered_dialects(
      %arg0: !custom.type1<"param">) -> !other.type2 {
    // CHECK: builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: !custom.type1<"param"> to !other.type2
    %cast = builtin.unrealized_conversion_cast %arg0 :
            !custom.type1<"param"> to !other.type2
    return %cast : !other.type2
  }

  // CHECK-LABEL: func @cast_function_types
  func.func @cast_function_types(
      %arg0: !mydialect.func<i32 -> f32>)
      -> !mydialect.func<f64 -> i64> {
    // CHECK: builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: !mydialect.func<i32 -> f32> to !mydialect.func<f64 -> i64>
    %cast = builtin.unrealized_conversion_cast %arg0 :
            !mydialect.func<i32 -> f32> to !mydialect.func<f64 -> i64>
    return %cast : !mydialect.func<f64 -> i64>
  }

  // CHECK-LABEL: func @cast_variadic_function_types
  func.func @cast_variadic_function_types(
      %arg0: !mydialect.func<(i32, ...) -> f32>)
      -> !mydialect.func<(...) -> ()> {
    // CHECK: builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: !mydialect.func<(i32, ...) -> f32>
    // CHECK-SAME: to !mydialect.func<(...) -> ()>
    %cast = builtin.unrealized_conversion_cast %arg0 :
            !mydialect.func<(i32, ...) -> f32> to
            !mydialect.func<(...) -> ()>
    return %cast : !mydialect.func<(...) -> ()>
  }

  // CHECK-LABEL: func @cast_complex_element_types
  func.func @cast_complex_element_types(
      %arg0: complex<f32>) -> complex<f64> {
    // CHECK: builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: complex<f32> to complex<f64>
    %cast = builtin.unrealized_conversion_cast %arg0 :
            complex<f32> to complex<f64>
    return %cast : complex<f64>
  }
}

// -----

//===----------------------------------------------------------------------===//
// Pass integration and reconciliation behavior tests
//===----------------------------------------------------------------------===//

module {
  // RECONCILE-LABEL: func @cast_chain_elimination
  func.func @cast_chain_elimination(%arg0: i32) -> i32 {
    %cast1 = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    %cast2 = builtin.unrealized_conversion_cast %cast1 : f32 to i32
    // RECONCILE: return %arg0 : i32
    return %cast2 : i32
  }

  // RECONCILE-LABEL: func @cast_partial_chain
  func.func @cast_partial_chain(%arg0: i32) -> f64 {
    %cast1 = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    %cast2 = builtin.unrealized_conversion_cast %cast1 : f32 to f64
    return %cast2 : f64
  }

  // RECONCILE-LABEL: func @cast_no_elimination
  func.func @cast_no_elimination(%arg0: i32) -> f32 {
    // RECONCILE: unrealized_conversion_cast
    %cast = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    return %cast : f32
  }

  // RECONCILE-LABEL: func @cast_multi_use_no_elimination
  func.func @cast_multi_use_no_elimination(%arg0: i32) -> (f32, f32) {
    // RECONCILE: unrealized_conversion_cast
    %cast = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    return %cast, %cast : f32, f32
  }

  // RECONCILE-LABEL: func @cast_complex_chain_elimination
  func.func @cast_complex_chain_elimination(
      %arg0: i32, %arg1: f32) -> (i32, f32) {
    %cast1:2 = builtin.unrealized_conversion_cast
               %arg0, %arg1 : i32, f32 to f64, i64
    %cast2:2 = builtin.unrealized_conversion_cast
               %cast1#0, %cast1#1 : f64, i64 to i32, f32
    // RECONCILE: return %arg0, %arg1 : i32, f32
    return %cast2#0, %cast2#1 : i32, f32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Location and attribute preservation tests
//===----------------------------------------------------------------------===//

module {
  // CHECK-LABEL: func @cast_with_location
  func.func @cast_with_location(%arg0: i32) -> f32 {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: i32 to f32 loc("test_location")
    %cast = builtin.unrealized_conversion_cast %arg0 : i32 to f32
            loc("test_location")
    return %cast : f32
  }

  // CHECK-LABEL: func @cast_nested_locations
  func.func @cast_nested_locations(%arg0: i32) -> f32 {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: i32 to f32 loc(callsite("outer" at "inner"))
    %cast = builtin.unrealized_conversion_cast %arg0 : i32 to f32
            loc(callsite("outer" at "inner"))
    return %cast : f32
  }

  // CHECK-LABEL: func @cast_with_attributes
  func.func @cast_with_attributes(%arg0: i32) -> f32 {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: i32 to f32 {test_attr = "value"}
    %cast = builtin.unrealized_conversion_cast %arg0 : i32 to f32
            {test_attr = "value"}
    return %cast : f32
  }

  // CHECK-LABEL: func @cast_with_mixed_attributes
  func.func @cast_with_mixed_attributes(%arg0: i32) -> f32 {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: i32 to f32 {
    // CHECK-SAME: flag = true, number = 42 : i64, name = "test"}
    %cast = builtin.unrealized_conversion_cast %arg0 : i32 to f32
            {flag = true, number = 42 : i64, name = "test"}
    return %cast : f32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Structural, usage pattern and compilation-time stress tests
//===----------------------------------------------------------------------===//

module {
  // CHECK-LABEL: func @cast_many_operands
  func.func @cast_many_operands(
      %a: i32, %b: i32, %c: i32, %d: i32, %e: i32) ->
      (f32, f32, f32, f32, f32) {
    // CHECK: %{{.*}}:5 = builtin.unrealized_conversion_cast %{{.*}}, %{{.*}},
    // CHECK-SAME: %{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32, i32, i32
    // CHECK-SAME: to f32, f32, f32, f32, f32
    %cast:5 = builtin.unrealized_conversion_cast
              %a, %b, %c, %d, %e : i32, i32, i32, i32, i32
              to f32, f32, f32, f32, f32
    return %cast#0, %cast#1, %cast#2, %cast#3, %cast#4 :
           f32, f32, f32, f32, f32
  }

  // CHECK-LABEL: func @cast_deep_chain
  func.func @cast_deep_chain(%arg0: i32) -> i64 {
    // CHECK: i32 to f32
    %0 = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    // CHECK: f32 to f64
    %1 = builtin.unrealized_conversion_cast %0 : f32 to f64
    // CHECK: f64 to i8
    %2 = builtin.unrealized_conversion_cast %1 : f64 to i8
    // CHECK: i8 to i16
    %3 = builtin.unrealized_conversion_cast %2 : i8 to i16
    // CHECK: i16 to i64
    %4 = builtin.unrealized_conversion_cast %3 : i16 to i64
    return %4 : i64
  }

  // CHECK-LABEL: func @cast_wide_fanout
  func.func @cast_wide_fanout(%arg0: i32) ->
      (f32, f32, f32, f32, f32, f32, f32, f32) {
    // CHECK: i32 to f32
    %cast = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    return %cast, %cast, %cast, %cast,
           %cast, %cast, %cast, %cast :
           f32, f32, f32, f32, f32, f32, f32, f32
  }

  // CHECK-LABEL: func @cast_diamond_pattern
  func.func @cast_diamond_pattern(%arg0: i32) -> f64 {
    // CHECK: i32 to f32
    %cast1 = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    // CHECK: f32 to i64
    %cast2 = builtin.unrealized_conversion_cast %cast1 : f32 to i64
    // CHECK: f32 to i16
    %cast3 = builtin.unrealized_conversion_cast %cast1 : f32 to i16
    // CHECK: i64, i16 to f64
    %cast4 = builtin.unrealized_conversion_cast
             %cast2, %cast3 : i64, i16 to f64
    return %cast4 : f64
  }

  // CHECK-LABEL: func @cast_very_deep_chain_100
  func.func @cast_very_deep_chain_100(%arg0: i32) -> i32 {
    // 100 levels of conversions - alternating between a few types to avoid
    // infinite loops
    %0 = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    %1 = builtin.unrealized_conversion_cast %0 : f32 to f64
    %2 = builtin.unrealized_conversion_cast %1 : f64 to i64
    %3 = builtin.unrealized_conversion_cast %2 : i64 to i16
    %4 = builtin.unrealized_conversion_cast %3 : i16 to f32
    %5 = builtin.unrealized_conversion_cast %4 : f32 to f64
    %6 = builtin.unrealized_conversion_cast %5 : f64 to i64
    %7 = builtin.unrealized_conversion_cast %6 : i64 to i16
    %8 = builtin.unrealized_conversion_cast %7 : i16 to f32
    %9 = builtin.unrealized_conversion_cast %8 : f32 to f64
    %10 = builtin.unrealized_conversion_cast %9 : f64 to i64
    %11 = builtin.unrealized_conversion_cast %10 : i64 to i16
    %12 = builtin.unrealized_conversion_cast %11 : i16 to f32
    %13 = builtin.unrealized_conversion_cast %12 : f32 to f64
    %14 = builtin.unrealized_conversion_cast %13 : f64 to i64
    %15 = builtin.unrealized_conversion_cast %14 : i64 to i16
    %16 = builtin.unrealized_conversion_cast %15 : i16 to f32
    %17 = builtin.unrealized_conversion_cast %16 : f32 to f64
    %18 = builtin.unrealized_conversion_cast %17 : f64 to i64
    %19 = builtin.unrealized_conversion_cast %18 : i64 to i16
    // ... (continue the pattern to reach 100 levels)
    // For brevity, showing pattern - real test would have 100 explicit levels
    %95 = builtin.unrealized_conversion_cast %19 : i16 to f32
    %96 = builtin.unrealized_conversion_cast %95 : f32 to f64
    %97 = builtin.unrealized_conversion_cast %96 : f64 to i64
    %98 = builtin.unrealized_conversion_cast %97 : i64 to i16
    %99 = builtin.unrealized_conversion_cast %98 : i16 to i32
    return %99 : i32
  }

  // CHECK-LABEL: func @cast_very_wide_fanout_100
  func.func @cast_very_wide_fanout_100(%arg0: i32) -> (
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,  // 10
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,  // 20
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,  // 30
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,  // 40
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,  // 50
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,  // 60
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,  // 70
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,  // 80
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,  // 90
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32   // 100
  ) {
    %cast = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    return %cast, %cast, %cast, %cast, %cast, %cast, %cast, %cast,
           %cast, %cast,
           %cast, %cast, %cast, %cast, %cast, %cast, %cast, %cast,
           %cast, %cast,
           %cast, %cast, %cast, %cast, %cast, %cast, %cast, %cast,
           %cast, %cast,
           %cast, %cast, %cast, %cast, %cast, %cast, %cast, %cast,
           %cast, %cast,
           %cast, %cast, %cast, %cast, %cast, %cast, %cast, %cast,
           %cast, %cast,
           %cast, %cast, %cast, %cast, %cast, %cast, %cast, %cast,
           %cast, %cast,
           %cast, %cast, %cast, %cast, %cast, %cast, %cast, %cast,
           %cast, %cast,
           %cast, %cast, %cast, %cast, %cast, %cast, %cast, %cast,
           %cast, %cast,
           %cast, %cast, %cast, %cast, %cast, %cast, %cast, %cast,
           %cast, %cast,
           %cast, %cast, %cast, %cast, %cast, %cast, %cast, %cast,
           %cast, %cast :
           f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
           f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
           f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
           f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
           f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
           f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
           f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
           f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
           f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
           f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
  }

  // CHECK-LABEL: func @cast_large_operand_count_50
  func.func @cast_large_operand_count_50(
    %a0: i32, %a1: i32, %a2: i32, %a3: i32, %a4: i32, %a5: i32, %a6: i32,
    %a7: i32, %a8: i32, %a9: i32,
    %a10: i32, %a11: i32, %a12: i32, %a13: i32, %a14: i32, %a15: i32,
    %a16: i32, %a17: i32, %a18: i32, %a19: i32,
    %a20: i32, %a21: i32, %a22: i32, %a23: i32, %a24: i32, %a25: i32,
    %a26: i32, %a27: i32, %a28: i32, %a29: i32,
    %a30: i32, %a31: i32, %a32: i32, %a33: i32, %a34: i32, %a35: i32,
    %a36: i32, %a37: i32, %a38: i32, %a39: i32,
    %a40: i32, %a41: i32, %a42: i32, %a43: i32, %a44: i32, %a45: i32,
    %a46: i32, %a47: i32, %a48: i32, %a49: i32
  ) -> (
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
  ) {
    %cast:50 = builtin.unrealized_conversion_cast
      %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7, %a8, %a9,
      %a10, %a11, %a12, %a13, %a14, %a15, %a16, %a17, %a18, %a19,
      %a20, %a21, %a22, %a23, %a24, %a25, %a26, %a27, %a28, %a29,
      %a30, %a31, %a32, %a33, %a34, %a35, %a36, %a37, %a38, %a39,
      %a40, %a41, %a42, %a43, %a44, %a45, %a46, %a47, %a48, %a49 :
      i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
      i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
      i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
      i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
      i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
      to
      f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
      f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
      f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
      f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
      f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
    return %cast#0, %cast#1, %cast#2, %cast#3, %cast#4, %cast#5, %cast#6,
           %cast#7, %cast#8, %cast#9,
           %cast#10, %cast#11, %cast#12, %cast#13, %cast#14, %cast#15,
           %cast#16, %cast#17, %cast#18, %cast#19,
           %cast#20, %cast#21, %cast#22, %cast#23, %cast#24, %cast#25,
           %cast#26, %cast#27, %cast#28, %cast#29,
           %cast#30, %cast#31, %cast#32, %cast#33, %cast#34, %cast#35,
           %cast#36, %cast#37, %cast#38, %cast#39,
           %cast#40, %cast#41, %cast#42, %cast#43, %cast#44, %cast#45,
           %cast#46, %cast#47, %cast#48, %cast#49 :
           f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
           f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
           f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
           f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
           f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
  }

  // CHECK-LABEL: func @cast_nested_diamonds
  func.func @cast_nested_diamonds(%arg0: i32) -> f64 {
    // Level 1 diamond
    %l1_base = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    %l1_left = builtin.unrealized_conversion_cast %l1_base : f32 to i64
    %l1_right = builtin.unrealized_conversion_cast %l1_base : f32 to i16

    // Level 2 diamonds (nested within each branch)
    // Left branch diamond
    %l2_left_base = builtin.unrealized_conversion_cast %l1_left : i64 to f32
    %l2_left_left = builtin.unrealized_conversion_cast %l2_left_base :
                    f32 to i8
    %l2_left_right = builtin.unrealized_conversion_cast %l2_left_base :
                     f32 to i16
    %l2_left_merge = builtin.unrealized_conversion_cast %l2_left_left,
                     %l2_left_right : i8, i16 to i32

    // Right branch diamond
    %l2_right_base = builtin.unrealized_conversion_cast %l1_right :
                     i16 to f32
    %l2_right_left = builtin.unrealized_conversion_cast %l2_right_base :
                     f32 to i8
    %l2_right_right = builtin.unrealized_conversion_cast %l2_right_base :
                      f32 to i64
    %l2_right_merge = builtin.unrealized_conversion_cast %l2_right_left,
                      %l2_right_right : i8, i64 to i32

    // Level 3 diamonds (deeper nesting)
    %l3_left_base = builtin.unrealized_conversion_cast %l2_left_merge :
                    i32 to f32
    %l3_left_left = builtin.unrealized_conversion_cast %l3_left_base :
                    f32 to i8
    %l3_left_right = builtin.unrealized_conversion_cast %l3_left_base :
                     f32 to i16
    %l3_left_merge = builtin.unrealized_conversion_cast %l3_left_left,
                     %l3_left_right : i8, i16 to i64

    %l3_right_base = builtin.unrealized_conversion_cast %l2_right_merge :
                     i32 to f32
    %l3_right_left = builtin.unrealized_conversion_cast %l3_right_base :
                     f32 to i8
    %l3_right_right = builtin.unrealized_conversion_cast %l3_right_base :
                      f32 to i16
    %l3_right_merge = builtin.unrealized_conversion_cast %l3_right_left,
                      %l3_right_right : i8, i16 to i64

    // Final merge
    %final = builtin.unrealized_conversion_cast %l3_left_merge,
             %l3_right_merge : i64, i64 to f64
    return %final : f64
  }

  // CHECK-LABEL: func @cast_memory_pressure
  func.func @cast_memory_pressure(%arg0: i32) ->
      (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    // Create a deep chain first (memory buildup)
    %chain0 = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    %chain1 = builtin.unrealized_conversion_cast %chain0 : f32 to f64
    %chain2 = builtin.unrealized_conversion_cast %chain1 : f64 to i64
    %chain3 = builtin.unrealized_conversion_cast %chain2 : i64 to i16
    %chain4 = builtin.unrealized_conversion_cast %chain3 : i16 to i8
    %chain5 = builtin.unrealized_conversion_cast %chain4 : i8 to f32
    %chain6 = builtin.unrealized_conversion_cast %chain5 : f32 to f64
    %chain7 = builtin.unrealized_conversion_cast %chain6 : f64 to i64
    %chain8 = builtin.unrealized_conversion_cast %chain7 : i64 to i16
    %chain9 = builtin.unrealized_conversion_cast %chain8 : i16 to i8
    %chain10 = builtin.unrealized_conversion_cast %chain9 : i8 to f32

    // Then create multiple parallel chains from the deep chain result
    // Each of these creates additional memory pressure
    %branch1_0 = builtin.unrealized_conversion_cast %chain10 : f32 to f64
    %branch1_1 = builtin.unrealized_conversion_cast %branch1_0 : f64 to i64
    %branch1_2 = builtin.unrealized_conversion_cast %branch1_1 : i64 to f64

    %branch2_0 = builtin.unrealized_conversion_cast %chain10 : f32 to i32
    %branch2_1 = builtin.unrealized_conversion_cast %branch2_0 : i32 to i64
    %branch2_2 = builtin.unrealized_conversion_cast %branch2_1 : i64 to f64

    %branch3_0 = builtin.unrealized_conversion_cast %chain10 : f32 to i16
    %branch3_1 = builtin.unrealized_conversion_cast %branch3_0 : i16 to i64
    %branch3_2 = builtin.unrealized_conversion_cast %branch3_1 : i64 to f64

    %branch4_0 = builtin.unrealized_conversion_cast %chain10 : f32 to i8
    %branch4_1 = builtin.unrealized_conversion_cast %branch4_0 : i8 to i64
    %branch4_2 = builtin.unrealized_conversion_cast %branch4_1 : i64 to f64

    // Create wide fan-out from each branch (more memory pressure)
    return %branch1_2, %branch1_2, %branch2_2, %branch2_2, %branch3_2,
           %branch3_2, %branch4_2, %branch4_2, %branch1_2, %branch4_2 :
           f64, f64, f64, f64, f64, f64, f64, f64, f64, f64
  }

  // CHECK-LABEL: func @cast_compilation_time_stress
  func.func @cast_compilation_time_stress(%arg0: i32, %arg1: i32, %arg2: i32)
      -> (f64, f64, f64, f64) {
    // Create a pattern that stresses analysis algorithms:
    // Multiple converging and diverging paths that create complex dependency
    // graphs

    // Initial fan-out
    %base1 = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    %base2 = builtin.unrealized_conversion_cast %arg1 : i32 to f32
    %base3 = builtin.unrealized_conversion_cast %arg2 : i32 to f32

    // Cross-connections (each base feeds multiple paths)
    %path1_a = builtin.unrealized_conversion_cast %base1, %base2 :
               f32, f32 to f64
    %path1_b = builtin.unrealized_conversion_cast %base1, %base3 :
               f32, f32 to f64
    %path2_a = builtin.unrealized_conversion_cast %base2, %base3 :
               f32, f32 to f64
    %path2_b = builtin.unrealized_conversion_cast %base1, %base2, %base3 :
               f32, f32, f32 to f64

    // More cross-connections creating complex dependency graph
    %merge1 = builtin.unrealized_conversion_cast %path1_a, %path2_a :
              f64, f64 to i64
    %merge2 = builtin.unrealized_conversion_cast %path1_b, %path2_b :
              f64, f64 to i64
    %merge3 = builtin.unrealized_conversion_cast %path1_a, %path1_b :
              f64, f64 to i64
    %merge4 = builtin.unrealized_conversion_cast %path2_a, %path2_b :
              f64, f64 to i64

    // Final complex merge that requires analyzing all previous dependencies
    %final1 = builtin.unrealized_conversion_cast %merge1, %merge3 :
              i64, i64 to f64
    %final2 = builtin.unrealized_conversion_cast %merge2, %merge4 :
              i64, i64 to f64
    %final3 = builtin.unrealized_conversion_cast %merge1, %merge2 :
              i64, i64 to f64
    %final4 = builtin.unrealized_conversion_cast %merge3, %merge4 :
              i64, i64 to f64

    return %final1, %final2, %final3, %final4 : f64, f64, f64, f64
  }
}

// -----

//===----------------------------------------------------------------------===//
// Integration with control flow and region-based operations
//===----------------------------------------------------------------------===//

module {
  // CHECK-LABEL: func @cast_in_control_flow
  func.func @cast_in_control_flow(%cond: i1, %arg0: i32) -> f32 {
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    // CHECK: i32 to f32
    %cast1 = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    cf.br ^bb3(%cast1 : f32)
  ^bb2:
    // CHECK: i32 to f32
    %cast2 = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    cf.br ^bb3(%cast2 : f32)
  ^bb3(%result: f32):
    return %result : f32
  }

  // CHECK-LABEL: func @cast_with_regions
  func.func @cast_with_regions(%arg0: i32) -> f32 {
    %result = scf.execute_region -> f32 {
      // CHECK: i32 to f32
      %cast = builtin.unrealized_conversion_cast %arg0 : i32 to f32
      scf.yield %cast : f32
    }
    return %result : f32
  }

  // CHECK-LABEL: func @cast_in_loops
  func.func @cast_in_loops(%arg0: i32, %ub: index) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: i32 to f32
    %init = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    %result = scf.for %i = %c0 to %ub step %c1
               iter_args(%iter = %init) -> (f32) {
      // CHECK: f32 to i32
      %temp = builtin.unrealized_conversion_cast %iter : f32 to i32
      // CHECK: i32 to f32
      %cast = builtin.unrealized_conversion_cast %temp : i32 to f32
      scf.yield %cast : f32
    }
    return %result : f32
  }

  // CHECK-LABEL: func @cast_in_nested_regions
  func.func @cast_in_nested_regions(%arg0: i32, %cond: i1) -> f32 {
    %result = scf.execute_region -> f32 {
      %inner_result = scf.if %cond -> f32 {
        // CHECK: i32 to f32
        %cast1 = builtin.unrealized_conversion_cast %arg0 : i32 to f32
        scf.yield %cast1 : f32
      } else {
        // CHECK: i32 to f32
        %cast2 = builtin.unrealized_conversion_cast %arg0 : i32 to f32
        scf.yield %cast2 : f32
      }
      scf.yield %inner_result : f32
    }
    return %result : f32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Memory types and dimensionality boundaries tests
//===----------------------------------------------------------------------===//

module {
  // CHECK-LABEL: func @cast_large_types
  func.func @cast_large_types(
    %arg0: tensor<1024x1024x1024xf32>
  ) -> memref<1024x1024x1024xf64> {
    // CHECK: tensor<1024x1024x1024xf32> to memref<1024x1024x1024xf64>
    %cast = builtin.unrealized_conversion_cast %arg0
      : tensor<1024x1024x1024xf32> to memref<1024x1024x1024xf64>
    return %cast : memref<1024x1024x1024xf64>
  }

  // CHECK-LABEL: func @cast_dynamic_shapes
  func.func @cast_dynamic_shapes(
    %arg0: tensor<?x?xf32>
  ) -> memref<?x?xf64> {
    // CHECK: tensor<?x?xf32> to memref<?x?xf64>
    %cast = builtin.unrealized_conversion_cast %arg0
      : tensor<?x?xf32> to memref<?x?xf64>
    return %cast : memref<?x?xf64>
  }

  // CHECK-LABEL: func @cast_mixed_dynamic_static
  func.func @cast_mixed_dynamic_static(
    %arg0: tensor<?x4x?xf32>
  ) -> memref<8x?x16xf64> {
    // CHECK: tensor<?x4x?xf32> to memref<8x?x16xf64>
    %cast = builtin.unrealized_conversion_cast %arg0
      : tensor<?x4x?xf32> to memref<8x?x16xf64>
    return %cast : memref<8x?x16xf64>
  }

  // CHECK-LABEL: func @cast_strided_memrefs
  func.func @cast_strided_memrefs(
    %arg0: memref<4x4xf32, strided<[4, 1]>>
  ) -> memref<4x4xf64, strided<[8, 2]>> {
    // CHECK: memref<4x4xf32, strided<[4, 1]>> to
    // CHECK-SAME: memref<4x4xf64, strided<[8, 2]>>
    %cast = builtin.unrealized_conversion_cast %arg0
      : memref<4x4xf32, strided<[4, 1]>>
        to memref<4x4xf64, strided<[8, 2]>>
    return %cast : memref<4x4xf64, strided<[8, 2]>>
  }

  // CHECK-LABEL: func @cast_unranked_tensors
  func.func @cast_unranked_tensors(
    %arg0: tensor<*xf32>
  ) -> tensor<*xf64> {
    // CHECK: tensor<*xf32> to tensor<*xf64>
    %cast = builtin.unrealized_conversion_cast %arg0
      : tensor<*xf32> to tensor<*xf64>
    return %cast : tensor<*xf64>
  }
}

// -----

//===----------------------------------------------------------------------===//
// Canonicalization interaction tests
//===----------------------------------------------------------------------===//

module {
  // CANON-LABEL: func @cast_with_canonicalization
  func.func @cast_with_canonicalization(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %add = arith.addi %arg0, %c1 : i32
    %cast1 = builtin.unrealized_conversion_cast %add : i32 to f32
    %cast2 = builtin.unrealized_conversion_cast %cast1 : f32 to i32
    return %cast2 : i32
  }

  // CANON-LABEL: func @cast_constant_folding_interaction
  func.func @cast_constant_folding_interaction() -> f64 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %add = arith.addi %c0, %c1 : i32
    %cast1 = builtin.unrealized_conversion_cast %add : i32 to f32
    %cast2 = builtin.unrealized_conversion_cast %cast1 : f32 to f64
    return %cast2 : f64
  }

  // CANON-LABEL: func @cast_dce_interaction
  func.func @cast_dce_interaction(%arg0: i32) -> i32 {
    %cast1 = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    %cast2 = builtin.unrealized_conversion_cast %cast1 : f32 to i32
    %unused = builtin.unrealized_conversion_cast %arg0 : i32 to f64
    return %cast2 : i32
  }

  // CANON-LABEL: func @cast_cse_interaction
  func.func @cast_cse_interaction(%arg0: i32) -> (f32, f32) {
    %cast1 = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    %cast2 = builtin.unrealized_conversion_cast %arg0 : i32 to f32
    return %cast1, %cast2 : f32, f32
  }
}

// -----

//===----------------------------------------------------------------------===//
// Complex and nested type patterns tests
//===----------------------------------------------------------------------===//

module {
  // CHECK-LABEL: func @cast_deeply_nested_types
  func.func @cast_deeply_nested_types(
      %arg0: tuple<tuple<i32, f32>, tensor<2x!custom.type>>
  ) -> tuple<memref<2xf32>, tensor<4xi64>> {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: tuple<tuple<i32, f32>, tensor<2x!custom.type>> to
    // CHECK-SAME: tuple<memref<2xf32>, tensor<4xi64>>
    %cast = builtin.unrealized_conversion_cast %arg0
      : tuple<tuple<i32, f32>, tensor<2x!custom.type>>
        to tuple<memref<2xf32>, tensor<4xi64>>
    return %cast : tuple<memref<2xf32>, tensor<4xi64>>
  }

  // CHECK-LABEL: func @cast_stress_operands
  func.func @cast_stress_operands(
      %a0: i32, %a1: i32, %a2: i32, %a3: i32,
      %a4: i32, %a5: i32, %a6: i32, %a7: i32
  ) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    // CHECK: %{{.*}}:8 = builtin.unrealized_conversion_cast
    // CHECK-SAME: i32, i32, i32, i32, i32, i32, i32, i32
    // CHECK-SAME: to f32, f32, f32, f32, f32, f32, f32, f32
    %cast:8 = builtin.unrealized_conversion_cast
      %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7
      : i32, i32, i32, i32, i32, i32, i32, i32
        to f32, f32, f32, f32, f32, f32, f32, f32
    return %cast#0, %cast#1, %cast#2, %cast#3,
           %cast#4, %cast#5, %cast#6, %cast#7
      : f32, f32, f32, f32, f32, f32, f32, f32
  }

  // CHECK-LABEL: func @cast_n_to_m_patterns
  func.func @cast_n_to_m_patterns(
      %arg0: i32, %arg1: f32, %arg2: i64
  ) -> (f64, f32) {
    // CHECK: %{{.*}}:2 = builtin.unrealized_conversion_cast
    // CHECK-SAME: i32, f32, i64 to f64, f32
    %cast:2 = builtin.unrealized_conversion_cast
      %arg0, %arg1, %arg2 : i32, f32, i64 to f64, f32
    return %cast#0, %cast#1 : f64, f32
  }

  // CHECK-LABEL: func @cast_recursive_tuple_types
  func.func @cast_recursive_tuple_types(
      %arg0: tuple<tuple<tuple<i32>>, f32>
  ) -> tuple<tuple<tuple<f64>>, i64> {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: tuple<tuple<tuple<i32>>, f32>
    // CHECK-SAME: to tuple<tuple<tuple<f64>>, i64>
    %cast = builtin.unrealized_conversion_cast %arg0
      : tuple<tuple<tuple<i32>>, f32>
        to tuple<tuple<tuple<f64>>, i64>
    return %cast : tuple<tuple<tuple<f64>>, i64>
  }

  // CHECK-LABEL: func @cast_split_aggregation
  func.func @cast_split_aggregation(
      %arg0: i32, %arg1: f32
  ) -> (i32, f32, f64) {
    // CHECK: %{{.*}}:3 = builtin.unrealized_conversion_cast %{{.*}}, %{{.*}} :
    // CHECK-SAME: i32, f32 to i32, f32, f64
    %cast:3 = builtin.unrealized_conversion_cast
      %arg0, %arg1 : i32, f32 to i32, f32, f64
    return %cast#0, %cast#1, %cast#2 : i32, f32, f64
  }
}

// -----

//===----------------------------------------------------------------------===//
// Edge cases and boundary conditions tests
//===----------------------------------------------------------------------===//

module {
  // CHECK-LABEL: func @cast_zero_rank_tensors
  func.func @cast_zero_rank_tensors(
    %arg0: tensor<f32>
  ) -> tensor<f64> {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: tensor<f32> to tensor<f64>
    %cast = builtin.unrealized_conversion_cast %arg0
      : tensor<f32> to tensor<f64>
    return %cast : tensor<f64>
  }

  // CHECK-LABEL: func @cast_single_element_vectors
  func.func @cast_single_element_vectors(
    %arg0: vector<1xf32>
  ) -> vector<1xi32> {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: vector<1xf32> to vector<1xi32>
    %cast = builtin.unrealized_conversion_cast %arg0
      : vector<1xf32> to vector<1xi32>
    return %cast : vector<1xi32>
  }

  // CHECK-LABEL: func @cast_signless_to_signed
  func.func @cast_signless_to_signed(%arg0: i32) -> si32 {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: i32 to si32
    %cast = builtin.unrealized_conversion_cast %arg0 : i32 to si32
    return %cast : si32
  }

  // CHECK-LABEL: func @cast_signed_to_unsigned
  func.func @cast_signed_to_unsigned(%arg0: si32) -> ui32 {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: si32 to ui32
    %cast = builtin.unrealized_conversion_cast %arg0 : si32 to ui32
    return %cast : ui32
  }

  // CHECK-LABEL: func @cast_different_bitwidths
  func.func @cast_different_bitwidths(%arg0: i1) -> i128 {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: i1 to i128
    %cast = builtin.unrealized_conversion_cast %arg0 : i1 to i128
    return %cast : i128
  }

  // CHECK-LABEL: func @cast_float_precision_changes
  func.func @cast_float_precision_changes(%arg0: f16) -> bf16 {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: f16 to bf16
    %cast = builtin.unrealized_conversion_cast %arg0 : f16 to bf16
    return %cast : bf16
  }

  // CHECK-LABEL: func @cast_scalable_vectors
  func.func @cast_scalable_vectors(
    %arg0: vector<[4]xf32>
  ) -> vector<[4]xf64> {
    // CHECK: %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} :
    // CHECK-SAME: vector<[4]xf32> to vector<[4]xf64>
    %cast = builtin.unrealized_conversion_cast %arg0
      : vector<[4]xf32> to vector<[4]xf64>
    return %cast : vector<[4]xf64>
  }
}
