// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Bitcast
//===----------------------------------------------------------------------===//

func @cast1(%arg0 : f32) {
  // CHECK: {{%.*}} = spv.Bitcast {{%.*}} : f32 to i32
  %0 = spv.Bitcast %arg0 : f32 to i32
  return
}

func @cast2(%arg0 : vector<2xf32>) {
  // CHECK: {{%.*}} = spv.Bitcast {{%.*}} : vector<2xf32> to vector<2xi32>
  %0 = spv.Bitcast %arg0 : vector<2xf32> to vector<2xi32>
  return
}

func @cast3(%arg0 : vector<2xf32>) {
  // CHECK: {{%.*}} = spv.Bitcast {{%.*}} : vector<2xf32> to i64
  %0 = spv.Bitcast %arg0 : vector<2xf32> to i64
  return
}

func @cast4(%arg0 : !spv.ptr<f32, Function>) {
  // CHECK: {{%.*}} = spv.Bitcast {{%.*}} : !spv.ptr<f32, Function> to !spv.ptr<i32, Function>
  %0 = spv.Bitcast %arg0 : !spv.ptr<f32, Function> to !spv.ptr<i32, Function>
  return
}

func @cast5(%arg0 : !spv.ptr<f32, Function>) {
  // CHECK: {{%.*}} = spv.Bitcast {{%.*}} : !spv.ptr<f32, Function> to !spv.ptr<vector<2xi32>, Function>
  %0 = spv.Bitcast %arg0 : !spv.ptr<f32, Function> to !spv.ptr<vector<2xi32>, Function>
  return
}

func @cast6(%arg0 : vector<4xf32>) {
  // CHECK: {{%.*}} = spv.Bitcast {{%.*}} : vector<4xf32> to vector<2xi64>
  %0 = spv.Bitcast %arg0 : vector<4xf32> to vector<2xi64>
  return
}

// -----

func @cast1(%arg0 : f32) {
  // expected-error @+1 {{result type must be different from operand type}}
  %0 = spv.Bitcast %arg0 : f32 to f32
  return
}

// -----

func @cast1(%arg0 : f32) {
  // expected-error @+1 {{mismatch in result type bitwidth 64 and operand type bitwidth 32}}
  %0 = spv.Bitcast %arg0 : f32 to i64
  return
}

// -----

func @cast1(%arg0 : vector<2xf32>) {
  // expected-error @+1 {{mismatch in result type bitwidth 96 and operand type bitwidth 64}}
  %0 = spv.Bitcast %arg0 : vector<2xf32> to vector<3xf32>
  return
}

// -----

func @cast3(%arg0 : !spv.ptr<f32, Function>) {
  // expected-error @+1 {{unhandled bit cast conversion from pointer type to non-pointer type}}
  %0 = spv.Bitcast %arg0 : !spv.ptr<f32, Function> to i64
  return
}

// -----

func @cast3(%arg0 : i64) {
  // expected-error @+1 {{unhandled bit cast conversion from non-pointer type to pointer type}}
  %0 = spv.Bitcast %arg0 : i64 to !spv.ptr<f32, Function>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.ConvertFToS
//===----------------------------------------------------------------------===//

func @convert_f_to_s_scalar(%arg0 : f32) -> i32 {
  // CHECK: {{%.*}} = spv.ConvertFToS {{%.*}} : f32 to i32
  %0 = spv.ConvertFToS %arg0 : f32 to i32
  spv.ReturnValue %0 : i32
}

// -----

func @convert_f64_to_s32_scalar(%arg0 : f64) -> i32 {
  // CHECK: {{%.*}} = spv.ConvertFToS {{%.*}} : f64 to i32
  %0 = spv.ConvertFToS %arg0 : f64 to i32
  spv.ReturnValue %0 : i32
}

// -----

func @convert_f_to_s_vector(%arg0 : vector<3xf32>) -> vector<3xi32> {
  // CHECK: {{%.*}} = spv.ConvertFToS {{%.*}} : vector<3xf32> to vector<3xi32>
  %0 = spv.ConvertFToS %arg0 : vector<3xf32> to vector<3xi32>
  spv.ReturnValue %0 : vector<3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.ConvertFToU
//===----------------------------------------------------------------------===//

func @convert_f_to_u_scalar(%arg0 : f32) -> i32 {
  // CHECK: {{%.*}} = spv.ConvertFToU {{%.*}} : f32 to i32
  %0 = spv.ConvertFToU %arg0 : f32 to i32
  spv.ReturnValue %0 : i32
}

// -----

func @convert_f64_to_u32_scalar(%arg0 : f64) -> i32 {
  // CHECK: {{%.*}} = spv.ConvertFToU {{%.*}} : f64 to i32
  %0 = spv.ConvertFToU %arg0 : f64 to i32
  spv.ReturnValue %0 : i32
}

// -----

func @convert_f_to_u_vector(%arg0 : vector<3xf32>) -> vector<3xi32> {
  // CHECK: {{%.*}} = spv.ConvertFToU {{%.*}} : vector<3xf32> to vector<3xi32>
  %0 = spv.ConvertFToU %arg0 : vector<3xf32> to vector<3xi32>
  spv.ReturnValue %0 : vector<3xi32>
}

// -----

func @convert_f_to_u_coopmatrix(%arg0 : !spv.coopmatrix<8x16xf32, Subgroup>) {
  // CHECK: {{%.*}} = spv.ConvertFToU {{%.*}} : !spv.coopmatrix<8x16xf32, Subgroup> to !spv.coopmatrix<8x16xi32, Subgroup>
  %0 = spv.ConvertFToU %arg0 : !spv.coopmatrix<8x16xf32, Subgroup> to !spv.coopmatrix<8x16xi32, Subgroup>
  spv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spv.ConvertSToF
//===----------------------------------------------------------------------===//

func @convert_s_to_f_scalar(%arg0 : i32) -> f32 {
  // CHECK: {{%.*}} = spv.ConvertSToF {{%.*}} : i32 to f32
  %0 = spv.ConvertSToF %arg0 : i32 to f32
  spv.ReturnValue %0 : f32
}

// -----

func @convert_s64_to_f32_scalar(%arg0 : i64) -> f32 {
  // CHECK: {{%.*}} = spv.ConvertSToF {{%.*}} : i64 to f32
  %0 = spv.ConvertSToF %arg0 : i64 to f32
  spv.ReturnValue %0 : f32
}

// -----

func @convert_s_to_f_vector(%arg0 : vector<3xi32>) -> vector<3xf32> {
  // CHECK: {{%.*}} = spv.ConvertSToF {{%.*}} : vector<3xi32> to vector<3xf32>
  %0 = spv.ConvertSToF %arg0 : vector<3xi32> to vector<3xf32>
  spv.ReturnValue %0 : vector<3xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.ConvertUToF
//===----------------------------------------------------------------------===//

func @convert_u_to_f_scalar(%arg0 : i32) -> f32 {
  // CHECK: {{%.*}} = spv.ConvertUToF {{%.*}} : i32 to f32
  %0 = spv.ConvertUToF %arg0 : i32 to f32
  spv.ReturnValue %0 : f32
}

// -----

func @convert_u64_to_f32_scalar(%arg0 : i64) -> f32 {
  // CHECK: {{%.*}} = spv.ConvertUToF {{%.*}} : i64 to f32
  %0 = spv.ConvertUToF %arg0 : i64 to f32
  spv.ReturnValue %0 : f32
}

// -----

func @convert_u_to_f_vector(%arg0 : vector<3xi32>) -> vector<3xf32> {
  // CHECK: {{%.*}} = spv.ConvertUToF {{%.*}} : vector<3xi32> to vector<3xf32>
  %0 = spv.ConvertUToF %arg0 : vector<3xi32> to vector<3xf32>
  spv.ReturnValue %0 : vector<3xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.FConvert
//===----------------------------------------------------------------------===//

func @f_convert_scalar(%arg0 : f32) -> f64 {
  // CHECK: {{%.*}} = spv.FConvert {{%.*}} : f32 to f64
  %0 = spv.FConvert %arg0 : f32 to f64
  spv.ReturnValue %0 : f64
}

// -----

func @f_convert_vector(%arg0 : vector<3xf32>) -> vector<3xf64> {
  // CHECK: {{%.*}} = spv.FConvert {{%.*}} : vector<3xf32> to vector<3xf64>
  %0 = spv.FConvert %arg0 : vector<3xf32> to vector<3xf64>
  spv.ReturnValue %0 : vector<3xf64>
}

// -----

func @f_convert_coop_matrix(%arg0 : !spv.coopmatrix<8x16xf32, Subgroup>) {
  // CHECK: {{%.*}} = spv.FConvert {{%.*}} : !spv.coopmatrix<8x16xf32, Subgroup> to !spv.coopmatrix<8x16xf64, Subgroup>
  %0 = spv.FConvert %arg0 : !spv.coopmatrix<8x16xf32, Subgroup> to !spv.coopmatrix<8x16xf64, Subgroup>
  spv.Return
}

// -----

func @f_convert_vector(%arg0 : f32) -> f32 {
  // expected-error @+1 {{expected the different bit widths for operand type and result type, but provided 'f32' and 'f32'}}
  %0 = spv.FConvert %arg0 : f32 to f32
  spv.ReturnValue %0 : f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.SConvert
//===----------------------------------------------------------------------===//

func @s_convert_scalar(%arg0 : i32) -> i64 {
  // CHECK: {{%.*}} = spv.SConvert {{%.*}} : i32 to i64
  %0 = spv.SConvert %arg0 : i32 to i64
  spv.ReturnValue %0 : i64
}

// -----

//===----------------------------------------------------------------------===//
// spv.UConvert
//===----------------------------------------------------------------------===//

func @u_convert_scalar(%arg0 : i32) -> i64 {
  // CHECK: {{%.*}} = spv.UConvert {{%.*}} : i32 to i64
  %0 = spv.UConvert %arg0 : i32 to i64
  spv.ReturnValue %0 : i64
}

