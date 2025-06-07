// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.FAdd
//===----------------------------------------------------------------------===//

func.func @fadd_scalar(%arg: f32) -> f32 {
  // CHECK: spirv.FAdd
  %0 = spirv.FAdd %arg, %arg : f32
  return %0 : f32
}

// -----

func.func @fadd_bf16_scalar(%arg: bf16) -> bf16 {
  // expected-error @+1 {{operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %0 = spirv.FAdd %arg, %arg : bf16
  return %0 : bf16
}

// -----

//===----------------------------------------------------------------------===//
// spirv.FDiv
//===----------------------------------------------------------------------===//

func.func @fdiv_scalar(%arg: f32) -> f32 {
  // CHECK: spirv.FDiv
  %0 = spirv.FDiv %arg, %arg : f32
  return %0 : f32
}

// -----

func.func @fdiv_bf16_scalar(%arg: bf16) -> bf16 {
  // expected-error @+1 {{operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %0 = spirv.FDiv %arg, %arg : bf16
  return %0 : bf16
}

// -----

//===----------------------------------------------------------------------===//
// spirv.FMod
//===----------------------------------------------------------------------===//

func.func @fmod_scalar(%arg: f32) -> f32 {
  // CHECK: spirv.FMod
  %0 = spirv.FMod %arg, %arg : f32
  return %0 : f32
}

// -----

func.func @fmod_bf16_scalar(%arg: bf16) -> bf16 {
  // expected-error @+1 {{operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %0 = spirv.FMod %arg, %arg : bf16
  return %0 : bf16
}

// -----

//===----------------------------------------------------------------------===//
// spirv.FMul
//===----------------------------------------------------------------------===//

func.func @fmul_scalar(%arg: f32) -> f32 {
  // CHECK: spirv.FMul
  %0 = spirv.FMul %arg, %arg : f32
  return %0 : f32
}

func.func @fmul_vector(%arg: vector<4xf32>) -> vector<4xf32> {
  // CHECK: spirv.FMul
  %0 = spirv.FMul %arg, %arg : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

func.func @fmul_i32(%arg: i32) -> i32 {
  // expected-error @+1 {{operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %0 = spirv.FMul %arg, %arg : i32
  return %0 : i32
}

// -----

func.func @fmul_bf16(%arg: bf16) -> bf16 {
  // expected-error @+1 {{operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %0 = spirv.FMul %arg, %arg : bf16
  return %0 : bf16
}

// -----

func.func @fmul_bf16_vector(%arg: vector<4xbf16>) -> vector<4xbf16> {
  // expected-error @+1 {{operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %0 = spirv.FMul %arg, %arg : vector<4xbf16>
  return %0 : vector<4xbf16>
}

// -----

func.func @fmul_tensor(%arg: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %0 = spirv.FMul %arg, %arg : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.FNegate
//===----------------------------------------------------------------------===//

func.func @fnegate_scalar(%arg: f32) -> f32 {
  // CHECK: spirv.FNegate
  %0 = spirv.FNegate %arg : f32
  return %0 : f32
}

// -----

func.func @fnegate_bf16_scalar(%arg: bf16) -> bf16 {
  // expected-error @+1 {{operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %0 = spirv.FNegate %arg : bf16
  return %0 : bf16
}

// -----

//===----------------------------------------------------------------------===//
// spirv.FRem
//===----------------------------------------------------------------------===//

func.func @frem_scalar(%arg: f32) -> f32 {
  // CHECK: spirv.FRem
  %0 = spirv.FRem %arg, %arg : f32
  return %0 : f32
}

// -----

func.func @frem_bf16_scalar(%arg: bf16) -> bf16 {
  // expected-error @+1 {{operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %0 = spirv.FRem %arg, %arg : bf16
  return %0 : bf16
}

// -----

//===----------------------------------------------------------------------===//
// spirv.FSub
//===----------------------------------------------------------------------===//

func.func @fsub_scalar(%arg: f32) -> f32 {
  // CHECK: spirv.FSub
  %0 = spirv.FSub %arg, %arg : f32
  return %0 : f32
}

// -----

func.func @fsub_bf16_scalar(%arg: bf16) -> bf16 {
  // expected-error @+1 {{operand #0 must be 16/32/64-bit float or vector of 16/32/64-bit float values}}
  %0 = spirv.FSub %arg, %arg : bf16
  return %0 : bf16
}

// -----

//===----------------------------------------------------------------------===//
// spirv.IAdd
//===----------------------------------------------------------------------===//

func.func @iadd_scalar(%arg: i32) -> i32 {
  // CHECK: spirv.IAdd
  %0 = spirv.IAdd %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.IMul
//===----------------------------------------------------------------------===//

func.func @imul_scalar(%arg: i32) -> i32 {
  // CHECK: spirv.IMul
  %0 = spirv.IMul %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ISub
//===----------------------------------------------------------------------===//

func.func @isub_scalar(%arg: i32) -> i32 {
  // CHECK: spirv.ISub
  %0 = spirv.ISub %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.IAddCarry
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @iadd_carry_scalar
func.func @iadd_carry_scalar(%arg: i32) -> !spirv.struct<(i32, i32)> {
  // CHECK: spirv.IAddCarry %{{.+}}, %{{.+}} : !spirv.struct<(i32, i32)>
  %0 = spirv.IAddCarry %arg, %arg : !spirv.struct<(i32, i32)>
  return %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: @iadd_carry_vector
func.func @iadd_carry_vector(%arg: vector<3xi32>) -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> {
  // CHECK: spirv.IAddCarry %{{.+}}, %{{.+}} : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  %0 = spirv.IAddCarry %arg, %arg : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  return %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
}

// -----

func.func @iadd_carry(%arg: i32) -> !spirv.struct<(i32, i32, i32)> {
  // expected-error @+1 {{expected spirv.struct type with two members}}
  %0 = spirv.IAddCarry %arg, %arg : !spirv.struct<(i32, i32, i32)>
  return %0 : !spirv.struct<(i32, i32, i32)>
}

// -----

func.func @iadd_carry(%arg: i32) -> !spirv.struct<(i32)> {
  // expected-error @+1 {{expected result struct type containing two members}}
  %0 = "spirv.IAddCarry"(%arg, %arg): (i32, i32) -> !spirv.struct<(i32)>
  return %0 : !spirv.struct<(i32)>
}

// -----

func.func @iadd_carry(%arg: i32) -> !spirv.struct<(i32, i64)> {
  // expected-error @+1 {{expected all operand types and struct member types are the same}}
  %0 = "spirv.IAddCarry"(%arg, %arg): (i32, i32) -> !spirv.struct<(i32, i64)>
  return %0 : !spirv.struct<(i32, i64)>
}

// -----

func.func @iadd_carry(%arg: i64) -> !spirv.struct<(i32, i32)> {
  // expected-error @+1 {{expected all operand types and struct member types are the same}}
  %0 = "spirv.IAddCarry"(%arg, %arg): (i64, i64) -> !spirv.struct<(i32, i32)>
  return %0 : !spirv.struct<(i32, i32)>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ISubBorrow
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @isub_borrow_scalar
func.func @isub_borrow_scalar(%arg: i32) -> !spirv.struct<(i32, i32)> {
  // CHECK: spirv.ISubBorrow %{{.+}}, %{{.+}} : !spirv.struct<(i32, i32)>
  %0 = spirv.ISubBorrow %arg, %arg : !spirv.struct<(i32, i32)>
  return %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: @isub_borrow_vector
func.func @isub_borrow_vector(%arg: vector<3xi32>) -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> {
  // CHECK: spirv.ISubBorrow %{{.+}}, %{{.+}} : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  %0 = spirv.ISubBorrow %arg, %arg : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  return %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
}

// -----

func.func @isub_borrow(%arg: i32) -> !spirv.struct<(i32, i32, i32)> {
  // expected-error @+1 {{expected spirv.struct type with two members}}
  %0 = spirv.ISubBorrow %arg, %arg : !spirv.struct<(i32, i32, i32)>
  return %0 : !spirv.struct<(i32, i32, i32)>
}

// -----

func.func @isub_borrow(%arg: i32) -> !spirv.struct<(i32)> {
  // expected-error @+1 {{expected result struct type containing two members}}
  %0 = "spirv.ISubBorrow"(%arg, %arg): (i32, i32) -> !spirv.struct<(i32)>
  return %0 : !spirv.struct<(i32)>
}

// -----

func.func @isub_borrow(%arg: i32) -> !spirv.struct<(i32, i64)> {
  // expected-error @+1 {{expected all operand types and struct member types are the same}}
  %0 = "spirv.ISubBorrow"(%arg, %arg): (i32, i32) -> !spirv.struct<(i32, i64)>
  return %0 : !spirv.struct<(i32, i64)>
}

// -----

func.func @isub_borrow(%arg: i64) -> !spirv.struct<(i32, i32)> {
  // expected-error @+1 {{expected all operand types and struct member types are the same}}
  %0 = "spirv.ISubBorrow"(%arg, %arg): (i64, i64) -> !spirv.struct<(i32, i32)>
  return %0 : !spirv.struct<(i32, i32)>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.Dot
//===----------------------------------------------------------------------===//

func.func @dot(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> f32 {
  %0 = spirv.Dot %arg0, %arg1 : vector<4xf32> -> f32
  return %0 : f32
}

// -----

// expected-note @+1 {{prior use here}}
func.func @dot(%arg0: vector<4xf32>, %arg1: vector<3xf32>) -> f32 {
  // expected-error @+1 {{use of value '%arg1' expects different type than prior uses}}
  %0 = spirv.Dot %arg0, %arg1 : vector<4xf32> -> f32
  return %0 : f32
}

// -----

func.func @dot(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> f16 {
  // expected-error @+1 {{'spirv.Dot' op failed to verify that all of {vector1, result} have same element type}}
  %0 = spirv.Dot %arg0, %arg1 : vector<4xf32> -> f16
  return %0 : f16
}

// -----

func.func @dot(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> i32 {
  // expected-error @+1 {{'spirv.Dot' op operand #0 must be vector of 16/32/64-bit float values of length 2/3/4/8/16}}
  %0 = spirv.Dot %arg0, %arg1 : vector<4xi32> -> i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SMulExtended
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @smul_extended_scalar
func.func @smul_extended_scalar(%arg: i32) -> !spirv.struct<(i32, i32)> {
  // CHECK: spirv.SMulExtended %{{.+}}, %{{.+}} : !spirv.struct<(i32, i32)>
  %0 = spirv.SMulExtended %arg, %arg : !spirv.struct<(i32, i32)>
  return %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: @smul_extended_vector
func.func @smul_extended_vector(%arg: vector<3xi32>) -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> {
  // CHECK: spirv.SMulExtended %{{.+}}, %{{.+}} : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  %0 = spirv.SMulExtended %arg, %arg : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  return %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
}

// -----

func.func @smul_extended(%arg: i32) -> !spirv.struct<(i32, i32, i32)> {
  // expected-error @+1 {{expected spirv.struct type with two members}}
  %0 = spirv.SMulExtended %arg, %arg : !spirv.struct<(i32, i32, i32)>
  return %0 : !spirv.struct<(i32, i32, i32)>
}

// -----

func.func @smul_extended(%arg: i32) -> !spirv.struct<(i32)> {
  // expected-error @+1 {{expected result struct type containing two members}}
  %0 = "spirv.SMulExtended"(%arg, %arg): (i32, i32) -> !spirv.struct<(i32)>
  return %0 : !spirv.struct<(i32)>
}

// -----

func.func @smul_extended(%arg: i32) -> !spirv.struct<(i32, i64)> {
  // expected-error @+1 {{expected all operand types and struct member types are the same}}
  %0 = "spirv.SMulExtended"(%arg, %arg): (i32, i32) -> !spirv.struct<(i32, i64)>
  return %0 : !spirv.struct<(i32, i64)>
}

// -----

func.func @smul_extended(%arg: i64) -> !spirv.struct<(i32, i32)> {
  // expected-error @+1 {{expected all operand types and struct member types are the same}}
  %0 = "spirv.SMulExtended"(%arg, %arg): (i64, i64) -> !spirv.struct<(i32, i32)>
  return %0 : !spirv.struct<(i32, i32)>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.UMulExtended
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @umul_extended_scalar
func.func @umul_extended_scalar(%arg: i32) -> !spirv.struct<(i32, i32)> {
  // CHECK: spirv.UMulExtended %{{.+}}, %{{.+}} : !spirv.struct<(i32, i32)>
  %0 = spirv.UMulExtended %arg, %arg : !spirv.struct<(i32, i32)>
  return %0 : !spirv.struct<(i32, i32)>
}

// CHECK-LABEL: @umul_extended_vector
func.func @umul_extended_vector(%arg: vector<3xi32>) -> !spirv.struct<(vector<3xi32>, vector<3xi32>)> {
  // CHECK: spirv.UMulExtended %{{.+}}, %{{.+}} : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  %0 = spirv.UMulExtended %arg, %arg : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
  return %0 : !spirv.struct<(vector<3xi32>, vector<3xi32>)>
}

// -----

func.func @umul_extended(%arg: i32) -> !spirv.struct<(i32, i32, i32)> {
  // expected-error @+1 {{expected spirv.struct type with two members}}
  %0 = spirv.UMulExtended %arg, %arg : !spirv.struct<(i32, i32, i32)>
  return %0 : !spirv.struct<(i32, i32, i32)>
}

// -----

func.func @umul_extended(%arg: i32) -> !spirv.struct<(i32)> {
  // expected-error @+1 {{expected result struct type containing two members}}
  %0 = "spirv.UMulExtended"(%arg, %arg): (i32, i32) -> !spirv.struct<(i32)>
  return %0 : !spirv.struct<(i32)>
}

// -----

func.func @umul_extended(%arg: i32) -> !spirv.struct<(i32, i64)> {
  // expected-error @+1 {{expected all operand types and struct member types are the same}}
  %0 = "spirv.UMulExtended"(%arg, %arg): (i32, i32) -> !spirv.struct<(i32, i64)>
  return %0 : !spirv.struct<(i32, i64)>
}

// -----

func.func @umul_extended(%arg: i64) -> !spirv.struct<(i32, i32)> {
  // expected-error @+1 {{expected all operand types and struct member types are the same}}
  %0 = "spirv.UMulExtended"(%arg, %arg): (i64, i64) -> !spirv.struct<(i32, i32)>
  return %0 : !spirv.struct<(i32, i32)>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SDiv
//===----------------------------------------------------------------------===//

func.func @sdiv_scalar(%arg: i32) -> i32 {
  // CHECK: spirv.SDiv
  %0 = spirv.SDiv %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SMod
//===----------------------------------------------------------------------===//

func.func @smod_scalar(%arg: i32) -> i32 {
  // CHECK: spirv.SMod
  %0 = spirv.SMod %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SNegate
//===----------------------------------------------------------------------===//

func.func @snegate_scalar(%arg: i32) -> i32 {
  // CHECK: spirv.SNegate
  %0 = spirv.SNegate %arg : i32
  return %0 : i32
}

// -----
//===----------------------------------------------------------------------===//
// spirv.SRem
//===----------------------------------------------------------------------===//

func.func @srem_scalar(%arg: i32) -> i32 {
  // CHECK: spirv.SRem
  %0 = spirv.SRem %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.UDiv
//===----------------------------------------------------------------------===//

func.func @udiv_scalar(%arg: i32) -> i32 {
  // CHECK: spirv.UDiv
  %0 = spirv.UDiv %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.UMod
//===----------------------------------------------------------------------===//

func.func @umod_scalar(%arg: i32) -> i32 {
  // CHECK: spirv.UMod
  %0 = spirv.UMod %arg, %arg : i32
  return %0 : i32
}

// -----
//===----------------------------------------------------------------------===//
// spirv.VectorTimesScalar
//===----------------------------------------------------------------------===//

func.func @vector_times_scalar(%vector: vector<4xf32>, %scalar: f32) -> vector<4xf32> {
  // CHECK: spirv.VectorTimesScalar %{{.+}}, %{{.+}} : (vector<4xf32>, f32) -> vector<4xf32>
  %0 = spirv.VectorTimesScalar %vector, %scalar : (vector<4xf32>, f32) -> vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

func.func @vector_times_scalar(%vector: vector<4xf32>, %scalar: f16) -> vector<4xf32> {
  // expected-error @+1 {{scalar operand and result element type match}}
  %0 = spirv.VectorTimesScalar %vector, %scalar : (vector<4xf32>, f16) -> vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

func.func @vector_times_scalar(%vector: vector<4xf32>, %scalar: f32) -> vector<3xf32> {
  // expected-error @+1 {{vector operand and result type mismatch}}
  %0 = spirv.VectorTimesScalar %vector, %scalar : (vector<4xf32>, f32) -> vector<3xf32>
  return %0 : vector<3xf32>
}

// -----

func.func @vector_bf16_times_scalar_bf16(%vector: vector<4xbf16>, %scalar: bf16) -> vector<4xbf16> {
  // expected-error @+1 {{op operand #0 must be vector of 16/32/64-bit float values of length 2/3/4}}
  %0 = spirv.VectorTimesScalar %vector, %scalar : (vector<4xbf16>, bf16) -> vector<4xbf16>
  return %0 : vector<4xbf16>
}
