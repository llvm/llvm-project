// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// This test covers the Integer Dot Product ops defined in the
// SPV_KHR_integer_dot_product extension.

//===----------------------------------------------------------------------===//
// spirv.SDot
//===----------------------------------------------------------------------===//

// CHECK: @sdot_scalar_i32
func.func @sdot_scalar_i32(%a: i32, %b: i32) -> i32 {
  // CHECK-NEXT: spirv.SDot
  %r = spirv.SDot %a, %b {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> i32
  return %r : i32
}

// CHECK: @sdot_scalar_i64
func.func @sdot_scalar_i64(%a: i32, %b: i32) -> i64 {
  // CHECK-NEXT: spirv.SDot
  %r = spirv.SDot %a, %b {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> i64
  return %r : i64
}

// CHECK: @sdot_vector_4xi8
func.func @sdot_vector_4xi8(%a: vector<4xi8>, %b: vector<4xi8>) -> i32 {
  // CHECK-NEXT: spirv.SDot
  %r = spirv.SDot %a, %b : (vector<4xi8>, vector<4xi8>) -> i32
  return %r : i32
}

// CHECK: @sdot_vector_4xi16
func.func @sdot_vector_4xi16(%a: vector<4xi16>, %b: vector<4xi16>) -> i64 {
  // CHECK-NEXT: spirv.SDot
  %r = spirv.SDot %a, %b : (vector<4xi16>, vector<4xi16>) -> i64
  return %r : i64
}

// CHECK: @sdot_vector_8xi8
func.func @sdot_vector_8xi8(%a: vector<8xi8>, %b: vector<8xi8>) -> i64 {
  // CHECK-NEXT: spirv.SDot
  %r = spirv.SDot %a, %b : (vector<8xi8>, vector<8xi8>) -> i64
  return %r : i64
}

// -----

func.func @sdot_scalar_bad_types(%a: i32, %b: i64) -> i32 {
  // expected-error @+1 {{op requires the same type for both vector operands}}
  %r = spirv.SDot %a, %b : (i32, i64) -> i32
  return %r : i32
}
// -----

func.func @sdot_vector_4xi8_bad_attr(%a: vector<4xi8>, %b: vector<4xi8>) -> i32 {
  // expected-error @+1 {{op with invalid format attribute for vector operands of type 'vector<4xi8>'}}
  %r = spirv.SDot %a, %b {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}:
        (vector<4xi8>, vector<4xi8>) -> i32
  return %r : i32
}

// -----

func.func @sdot_scalar_i32_bad_attr(%a: i32, %b: i32) -> i32 {
  // expected-error @+1 {{op only supports the 'format' #spirv.packed_vector_format attribute}}
  %r = spirv.SDot %a, %b {volatile = #spirv.decoration<Volatile>,
                          format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> i32
  return %r : i32
}

// -----

func.func @udot_vector_4xi8_bad_attr(%a: vector<4xi8>, %b: vector<4xi8>) -> i32 {
  // expected-error @+1 {{op only supports the 'format' #spirv.packed_vector_format attribute}}
  %r = spirv.UDot %a, %b {volatile = #spirv.decoration<Volatile>}: (vector<4xi8>, vector<4xi8>) -> i32
  return %r : i32
}

// -----

func.func @sdot_scalar_bad_types(%a: i32, %b: i32) -> i16 {
  // expected-error @+1 {{op result type has insufficient bit-width (16 bits) for the specified vector operand type (32 bits)}}
  %r = spirv.SDot %a, %b {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> i16
  return %r : i16
}

// -----

func.func @sdot_scalar_bad_types(%a: i64, %b: i64) -> i64 {
  // expected-error @+1 {{op with specified Packed Vector Format (PackedVectorFormat4x8Bit) requires integer vector operands to be 32-bits wide}}
  %r = spirv.SDot %a, %b {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i64, i64) -> i64
  return %r : i64
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SUDot
//===----------------------------------------------------------------------===//

// CHECK: @sudot_scalar_i32
func.func @sudot_scalar_i32(%a: i32, %b: i32) -> i32 {
  // CHECK-NEXT: spirv.SUDot
  %r = spirv.SUDot %a, %b {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> i32
  return %r : i32
}

// CHECK: @sudot_scalar_i64
func.func @sudot_scalar_i64(%a: i32, %b: i32) -> i64 {
  // CHECK-NEXT: spirv.SUDot
  %r = spirv.SUDot %a, %b {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> i64
  return %r : i64
}

// CHECK: @sudot_vector_4xi8
func.func @sudot_vector_4xi8(%a: vector<4xi8>, %b: vector<4xi8>) -> i32 {
  // CHECK-NEXT: spirv.SUDot
  %r = spirv.SUDot %a, %b : (vector<4xi8>, vector<4xi8>) -> i32
  return %r : i32
}

// CHECK: @sudot_vector_4xi16
func.func @sudot_vector_4xi16(%a: vector<4xi16>, %b: vector<4xi16>) -> i64 {
  // CHECK-NEXT: spirv.SUDot
  %r = spirv.SUDot %a, %b : (vector<4xi16>, vector<4xi16>) -> i64
  return %r : i64
}

// CHECK: @sudot_vector_8xi8
func.func @sudot_vector_8xi8(%a: vector<8xi8>, %b: vector<8xi8>) -> i64 {
  // CHECK-NEXT: spirv.SUDot
  %r = spirv.SUDot %a, %b : (vector<8xi8>, vector<8xi8>) -> i64
  return %r : i64
}

// -----

//===----------------------------------------------------------------------===//
// spirv.UDot
//===----------------------------------------------------------------------===//

// CHECK: @udot_scalar_i32
func.func @udot_scalar_i32(%a: i32, %b: i32) -> i32 {
  // CHECK-NEXT: spirv.UDot
  %r = spirv.UDot %a, %b {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> i32
  return %r : i32
}

// CHECK: @udot_scalar_i64
func.func @udot_scalar_i64(%a: i32, %b: i32) -> i64 {
  // CHECK-NEXT: spirv.UDot
  %r = spirv.UDot %a, %b {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> i64
  return %r : i64
}

// CHECK: @udot_vector_4xi8
func.func @udot_vector_4xi8(%a: vector<4xi8>, %b: vector<4xi8>) -> i32 {
  // CHECK-NEXT: spirv.UDot
  %r = spirv.UDot %a, %b : (vector<4xi8>, vector<4xi8>) -> i32
  return %r : i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SDotAccSat
//===----------------------------------------------------------------------===//

// CHECK: @sdot_acc_sat_scalar_i32
func.func @sdot_acc_sat_scalar_i32(%a: i32, %b: i32, %acc: i32) -> i32 {
  // CHECK-NEXT: spirv.SDotAccSat
  %r = spirv.SDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32, i32) -> i32
  return %r : i32
}

// CHECK: @sdot_acc_sat_scalar_i64
func.func @sdot_acc_sat_scalar_i64(%a: i32, %b: i32, %acc: i64) -> i64 {
  // CHECK-NEXT: spirv.SDotAccSat
  %r = spirv.SDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32, i64) -> i64
  return %r : i64
}

// CHECK: @sdot_acc_sat_vector_4xi8
func.func @sdot_acc_sat_vector_4xi8(%a: vector<4xi8>, %b: vector<4xi8>, %acc: i32) -> i32 {
  // CHECK-NEXT: spirv.SDotAccSat
  %r = spirv.SDotAccSat %a, %b, %acc : (vector<4xi8>, vector<4xi8>, i32) -> i32
  return %r : i32
}

// CHECK: @sdot_acc_sat_vector_4xi16
func.func @sdot_acc_sat_vector_4xi16(%a: vector<4xi16>, %b: vector<4xi16>, %acc: i64) -> i64 {
  // CHECK-NEXT: spirv.SDotAccSat
  %r = spirv.SDotAccSat %a, %b, %acc : (vector<4xi16>, vector<4xi16>, i64) -> i64
  return %r : i64
}

// CHECK: @sdot_acc_sat_vector_8xi8
func.func @sdot_acc_sat_vector_8xi8(%a: vector<8xi8>, %b: vector<8xi8>, %acc: i64) -> i64 {
  // CHECK-NEXT: spirv.SDotAccSat
  %r = spirv.SDotAccSat %a, %b, %acc : (vector<8xi8>, vector<8xi8>, i64) -> i64
  return %r : i64
}

// -----

func.func @sdot_acc_sat_scalar_bad_types(%a: i32, %b: i64, %acc: i32) -> i32 {
  // expected-error @+1 {{op requires the same type for both vector operands}}
  %r = spirv.SDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i64, i32) -> i32
  return %r : i32
}

// -----

func.func @sdot_acc_sat_scalar_bad_types(%a: i32, %b: i32, %acc: i16) -> i16 {
  // expected-error @+1 {{op result type has insufficient bit-width (16 bits) for the specified vector operand type (32 bits)}}
  %r = spirv.SDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32, i16) -> i16
  return %r : i16
}

// -----

func.func @sdot_acc_sat_scalar_bad_types(%a: i64, %b: i64, %acc: i64) -> i64 {
  // expected-error @+1 {{op with specified Packed Vector Format (PackedVectorFormat4x8Bit) requires integer vector operands to be 32-bits wide}}
  %r = spirv.SDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i64, i64, i64) -> i64
  return %r : i64
}

// -----

func.func @sdot_acc_sat_scalar_bad_accumulator(%a: i32, %b: i32, %acc: i32) -> i64 {
  // expected-error @+1 {{requires the same accumulator operand and result types}}
  %r = spirv.SDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32, i32) -> i64
  return %r : i64
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SUDotAccSat
//===----------------------------------------------------------------------===//

// CHECK: @sudot_acc_sat_scalar_i32
func.func @sudot_acc_sat_scalar_i32(%a: i32, %b: i32, %acc: i32) -> i32 {
  // CHECK-NEXT: spirv.SUDotAccSat
  %r = spirv.SUDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32, i32) -> i32
  return %r : i32
}

// CHECK: @sudot_acc_sat_scalar_i64
func.func @sudot_acc_sat_scalar_i64(%a: i32, %b: i32, %acc: i64) -> i64 {
  // CHECK-NEXT: spirv.SUDotAccSat
  %r = spirv.SUDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32, i64) -> i64
  return %r : i64
}

// CHECK: @sudot_acc_sat_vector_4xi8
func.func @sudot_acc_sat_vector_4xi8(%a: vector<4xi8>, %b: vector<4xi8>, %acc: i32) -> i32 {
  // CHECK-NEXT: spirv.SUDotAccSat
  %r = spirv.SUDotAccSat %a, %b, %acc : (vector<4xi8>, vector<4xi8>, i32) -> i32
  return %r : i32
}

// CHECK: @sudot_acc_sat_vector_4xi16
func.func @sudot_acc_sat_vector_4xi16(%a: vector<4xi16>, %b: vector<4xi16>, %acc: i64) -> i64 {
  // CHECK-NEXT: spirv.SUDotAccSat
  %r = spirv.SUDotAccSat %a, %b, %acc : (vector<4xi16>, vector<4xi16>, i64) -> i64
  return %r : i64
}

// CHECK: @sudot_acc_sat_vector_8xi8
func.func @sudot_acc_sat_vector_8xi8(%a: vector<8xi8>, %b: vector<8xi8>, %acc: i64) -> i64 {
  // CHECK-NEXT: spirv.SUDotAccSat
  %r = spirv.SUDotAccSat %a, %b, %acc : (vector<8xi8>, vector<8xi8>, i64) -> i64
  return %r : i64
}

// -----

//===----------------------------------------------------------------------===//
// spirv.UDotAccSat
//===----------------------------------------------------------------------===//

// CHECK: @udot_acc_sat_scalar_i32
func.func @udot_acc_sat_scalar_i32(%a: i32, %b: i32, %acc: i32) -> i32 {
  // CHECK-NEXT: spirv.UDotAccSat
  %r = spirv.UDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32, i32) -> i32
  return %r : i32
}

// CHECK: @udot_acc_sat_scalar_i64
func.func @udot_acc_sat_scalar_i64(%a: i32, %b: i32, %acc: i64) -> i64 {
  // CHECK-NEXT: spirv.UDotAccSat
  %r = spirv.UDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32, i64) -> i64
  return %r : i64
}

// CHECK: @udot_acc_sat_vector_4xi8
func.func @udot_acc_sat_vector_4xi8(%a: vector<4xi8>, %b: vector<4xi8>, %acc: i32) -> i32 {
  // CHECK-NEXT: spirv.UDotAccSat
  %r = spirv.UDotAccSat %a, %b, %acc : (vector<4xi8>, vector<4xi8>, i32) -> i32
  return %r : i32
}

// CHECK: @udot_acc_sat_vector_4xi16
func.func @udot_acc_sat_vector_4xi16(%a: vector<4xi16>, %b: vector<4xi16>, %acc: i64) -> i64 {
  // CHECK-NEXT: spirv.UDotAccSat
  %r = spirv.UDotAccSat %a, %b, %acc : (vector<4xi16>, vector<4xi16>, i64) -> i64
  return %r : i64
}

// CHECK: @udot_acc_sat_vector_8xi8
func.func @udot_acc_sat_vector_8xi8(%a: vector<8xi8>, %b: vector<8xi8>, %acc: i64) -> i64 {
  // CHECK-NEXT: spirv.UDotAccSat
  %r = spirv.UDotAccSat %a, %b, %acc : (vector<8xi8>, vector<8xi8>, i64) -> i64
  return %r : i64
}
