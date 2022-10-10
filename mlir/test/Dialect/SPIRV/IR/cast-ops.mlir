// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.Bitcast
//===----------------------------------------------------------------------===//

func.func @cast1(%arg0 : f32) {
  // CHECK: {{%.*}} = spirv.Bitcast {{%.*}} : f32 to i32
  %0 = spirv.Bitcast %arg0 : f32 to i32
  return
}

func.func @cast2(%arg0 : vector<2xf32>) {
  // CHECK: {{%.*}} = spirv.Bitcast {{%.*}} : vector<2xf32> to vector<2xi32>
  %0 = spirv.Bitcast %arg0 : vector<2xf32> to vector<2xi32>
  return
}

func.func @cast3(%arg0 : vector<2xf32>) {
  // CHECK: {{%.*}} = spirv.Bitcast {{%.*}} : vector<2xf32> to i64
  %0 = spirv.Bitcast %arg0 : vector<2xf32> to i64
  return
}

func.func @cast4(%arg0 : !spirv.ptr<f32, Function>) {
  // CHECK: {{%.*}} = spirv.Bitcast {{%.*}} : !spirv.ptr<f32, Function> to !spirv.ptr<i32, Function>
  %0 = spirv.Bitcast %arg0 : !spirv.ptr<f32, Function> to !spirv.ptr<i32, Function>
  return
}

func.func @cast5(%arg0 : !spirv.ptr<f32, Function>) {
  // CHECK: {{%.*}} = spirv.Bitcast {{%.*}} : !spirv.ptr<f32, Function> to !spirv.ptr<vector<2xi32>, Function>
  %0 = spirv.Bitcast %arg0 : !spirv.ptr<f32, Function> to !spirv.ptr<vector<2xi32>, Function>
  return
}

func.func @cast6(%arg0 : vector<4xf32>) {
  // CHECK: {{%.*}} = spirv.Bitcast {{%.*}} : vector<4xf32> to vector<2xi64>
  %0 = spirv.Bitcast %arg0 : vector<4xf32> to vector<2xi64>
  return
}

// -----

func.func @cast1(%arg0 : f32) {
  // expected-error @+1 {{result type must be different from operand type}}
  %0 = spirv.Bitcast %arg0 : f32 to f32
  return
}

// -----

func.func @cast1(%arg0 : f32) {
  // expected-error @+1 {{mismatch in result type bitwidth 64 and operand type bitwidth 32}}
  %0 = spirv.Bitcast %arg0 : f32 to i64
  return
}

// -----

func.func @cast1(%arg0 : vector<2xf32>) {
  // expected-error @+1 {{mismatch in result type bitwidth 96 and operand type bitwidth 64}}
  %0 = spirv.Bitcast %arg0 : vector<2xf32> to vector<3xf32>
  return
}

// -----

func.func @cast3(%arg0 : !spirv.ptr<f32, Function>) {
  // expected-error @+1 {{unhandled bit cast conversion from pointer type to non-pointer type}}
  %0 = spirv.Bitcast %arg0 : !spirv.ptr<f32, Function> to i64
  return
}

// -----

func.func @cast3(%arg0 : i64) {
  // expected-error @+1 {{unhandled bit cast conversion from non-pointer type to pointer type}}
  %0 = spirv.Bitcast %arg0 : i64 to !spirv.ptr<f32, Function>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ConvertFToS
//===----------------------------------------------------------------------===//

func.func @convert_f_to_s_scalar(%arg0 : f32) -> i32 {
  // CHECK: {{%.*}} = spirv.ConvertFToS {{%.*}} : f32 to i32
  %0 = spirv.ConvertFToS %arg0 : f32 to i32
  spirv.ReturnValue %0 : i32
}

// -----

func.func @convert_f64_to_s32_scalar(%arg0 : f64) -> i32 {
  // CHECK: {{%.*}} = spirv.ConvertFToS {{%.*}} : f64 to i32
  %0 = spirv.ConvertFToS %arg0 : f64 to i32
  spirv.ReturnValue %0 : i32
}

// -----

func.func @convert_f_to_s_vector(%arg0 : vector<3xf32>) -> vector<3xi32> {
  // CHECK: {{%.*}} = spirv.ConvertFToS {{%.*}} : vector<3xf32> to vector<3xi32>
  %0 = spirv.ConvertFToS %arg0 : vector<3xf32> to vector<3xi32>
  spirv.ReturnValue %0 : vector<3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ConvertFToU
//===----------------------------------------------------------------------===//

func.func @convert_f_to_u_scalar(%arg0 : f32) -> i32 {
  // CHECK: {{%.*}} = spirv.ConvertFToU {{%.*}} : f32 to i32
  %0 = spirv.ConvertFToU %arg0 : f32 to i32
  spirv.ReturnValue %0 : i32
}

// -----

func.func @convert_f64_to_u32_scalar(%arg0 : f64) -> i32 {
  // CHECK: {{%.*}} = spirv.ConvertFToU {{%.*}} : f64 to i32
  %0 = spirv.ConvertFToU %arg0 : f64 to i32
  spirv.ReturnValue %0 : i32
}

// -----

func.func @convert_f_to_u_vector(%arg0 : vector<3xf32>) -> vector<3xi32> {
  // CHECK: {{%.*}} = spirv.ConvertFToU {{%.*}} : vector<3xf32> to vector<3xi32>
  %0 = spirv.ConvertFToU %arg0 : vector<3xf32> to vector<3xi32>
  spirv.ReturnValue %0 : vector<3xi32>
}

// -----

func.func @convert_f_to_u_coopmatrix(%arg0 : !spirv.coopmatrix<8x16xf32, Subgroup>) {
  // CHECK: {{%.*}} = spirv.ConvertFToU {{%.*}} : !spirv.coopmatrix<8x16xf32, Subgroup> to !spirv.coopmatrix<8x16xi32, Subgroup>
  %0 = spirv.ConvertFToU %arg0 : !spirv.coopmatrix<8x16xf32, Subgroup> to !spirv.coopmatrix<8x16xi32, Subgroup>
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ConvertSToF
//===----------------------------------------------------------------------===//

func.func @convert_s_to_f_scalar(%arg0 : i32) -> f32 {
  // CHECK: {{%.*}} = spirv.ConvertSToF {{%.*}} : i32 to f32
  %0 = spirv.ConvertSToF %arg0 : i32 to f32
  spirv.ReturnValue %0 : f32
}

// -----

func.func @convert_s64_to_f32_scalar(%arg0 : i64) -> f32 {
  // CHECK: {{%.*}} = spirv.ConvertSToF {{%.*}} : i64 to f32
  %0 = spirv.ConvertSToF %arg0 : i64 to f32
  spirv.ReturnValue %0 : f32
}

// -----

func.func @convert_s_to_f_vector(%arg0 : vector<3xi32>) -> vector<3xf32> {
  // CHECK: {{%.*}} = spirv.ConvertSToF {{%.*}} : vector<3xi32> to vector<3xf32>
  %0 = spirv.ConvertSToF %arg0 : vector<3xi32> to vector<3xf32>
  spirv.ReturnValue %0 : vector<3xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ConvertUToF
//===----------------------------------------------------------------------===//

func.func @convert_u_to_f_scalar(%arg0 : i32) -> f32 {
  // CHECK: {{%.*}} = spirv.ConvertUToF {{%.*}} : i32 to f32
  %0 = spirv.ConvertUToF %arg0 : i32 to f32
  spirv.ReturnValue %0 : f32
}

// -----

func.func @convert_u64_to_f32_scalar(%arg0 : i64) -> f32 {
  // CHECK: {{%.*}} = spirv.ConvertUToF {{%.*}} : i64 to f32
  %0 = spirv.ConvertUToF %arg0 : i64 to f32
  spirv.ReturnValue %0 : f32
}

// -----

func.func @convert_u_to_f_vector(%arg0 : vector<3xi32>) -> vector<3xf32> {
  // CHECK: {{%.*}} = spirv.ConvertUToF {{%.*}} : vector<3xi32> to vector<3xf32>
  %0 = spirv.ConvertUToF %arg0 : vector<3xi32> to vector<3xf32>
  spirv.ReturnValue %0 : vector<3xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.FConvert
//===----------------------------------------------------------------------===//

func.func @f_convert_scalar(%arg0 : f32) -> f64 {
  // CHECK: {{%.*}} = spirv.FConvert {{%.*}} : f32 to f64
  %0 = spirv.FConvert %arg0 : f32 to f64
  spirv.ReturnValue %0 : f64
}

// -----

func.func @f_convert_vector(%arg0 : vector<3xf32>) -> vector<3xf64> {
  // CHECK: {{%.*}} = spirv.FConvert {{%.*}} : vector<3xf32> to vector<3xf64>
  %0 = spirv.FConvert %arg0 : vector<3xf32> to vector<3xf64>
  spirv.ReturnValue %0 : vector<3xf64>
}

// -----

func.func @f_convert_coop_matrix(%arg0 : !spirv.coopmatrix<8x16xf32, Subgroup>) {
  // CHECK: {{%.*}} = spirv.FConvert {{%.*}} : !spirv.coopmatrix<8x16xf32, Subgroup> to !spirv.coopmatrix<8x16xf64, Subgroup>
  %0 = spirv.FConvert %arg0 : !spirv.coopmatrix<8x16xf32, Subgroup> to !spirv.coopmatrix<8x16xf64, Subgroup>
  spirv.Return
}

// -----

func.func @f_convert_vector(%arg0 : f32) -> f32 {
  // expected-error @+1 {{expected the different bit widths for operand type and result type, but provided 'f32' and 'f32'}}
  %0 = spirv.FConvert %arg0 : f32 to f32
  spirv.ReturnValue %0 : f32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.SConvert
//===----------------------------------------------------------------------===//

func.func @s_convert_scalar(%arg0 : i32) -> i64 {
  // CHECK: {{%.*}} = spirv.SConvert {{%.*}} : i32 to i64
  %0 = spirv.SConvert %arg0 : i32 to i64
  spirv.ReturnValue %0 : i64
}

// -----

//===----------------------------------------------------------------------===//
// spirv.UConvert
//===----------------------------------------------------------------------===//

func.func @u_convert_scalar(%arg0 : i32) -> i64 {
  // CHECK: {{%.*}} = spirv.UConvert {{%.*}} : i32 to i64
  %0 = spirv.UConvert %arg0 : i32 to i64
  spirv.ReturnValue %0 : i64
}

// -----

//===----------------------------------------------------------------------===//
// spirv.PtrCastToGeneric
//===----------------------------------------------------------------------===//

func.func @ptrcasttogeneric1(%arg0 : !spirv.ptr<f32, CrossWorkgroup>) {
  // CHECK: {{%.*}} = spirv.PtrCastToGeneric {{%.*}} : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<f32, Generic>
  %0 = spirv.PtrCastToGeneric %arg0 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<f32, Generic>
  return
}
// -----

func.func @ptrcasttogeneric2(%arg0 : !spirv.ptr<f32, StorageBuffer>) {
  // expected-error @+1 {{pointer must point to the Workgroup, CrossWorkgroup, or Function Storage Class}}
  %0 = spirv.PtrCastToGeneric %arg0 : !spirv.ptr<f32, StorageBuffer> to !spirv.ptr<f32, Generic>
  return
}

// -----

func.func @ptrcasttogeneric3(%arg0 : !spirv.ptr<f32, CrossWorkgroup>) {
  // expected-error @+1 {{result type must be of storage class Generic}}
  %0 = spirv.PtrCastToGeneric %arg0 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<f32, Uniform>
  return
}

// -----

func.func @ptrcasttogeneric4(%arg0 : !spirv.ptr<f32, CrossWorkgroup>) {
  // expected-error @+1 {{pointee type must have the same as the op result type}}
  %0 = spirv.PtrCastToGeneric %arg0 : !spirv.ptr<f32, CrossWorkgroup> to !spirv.ptr<vector<2xi32>, Generic>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.GenericCastToPtr
//===----------------------------------------------------------------------===//

func.func @genericcasttoptr1(%arg0 : !spirv.ptr<vector<2xi32>, Generic>) {
  // CHECK: {{%.*}} = spirv.GenericCastToPtr {{%.*}} : !spirv.ptr<vector<2xi32>, Generic> to !spirv.ptr<vector<2xi32>, CrossWorkgroup>
  %0 = spirv.GenericCastToPtr %arg0 : !spirv.ptr<vector<2xi32>, Generic> to !spirv.ptr<vector<2xi32>, CrossWorkgroup>
  return
}
// -----

func.func @genericcasttoptr2(%arg0 : !spirv.ptr<f32, Uniform>) {
  // expected-error @+1 {{pointer type must be of storage class Generic}}
  %0 = spirv.GenericCastToPtr %arg0 : !spirv.ptr<f32, Uniform> to !spirv.ptr<f32, Workgroup>
  return
}

// -----

func.func @genericcasttoptr3(%arg0 : !spirv.ptr<f32, Generic>) {
  // expected-error @+1 {{result must point to the Workgroup, CrossWorkgroup, or Function Storage Class}}
  %0 = spirv.GenericCastToPtr %arg0 : !spirv.ptr<f32, Generic> to !spirv.ptr<f32, Uniform>
  return
}

// -----

func.func @genericcasttoptr4(%arg0 : !spirv.ptr<f32, Generic>) {
  // expected-error @+1 {{pointee type must have the same as the op result type}}
  %0 = spirv.GenericCastToPtr %arg0 : !spirv.ptr<f32, Generic> to !spirv.ptr<vector<2xi32>, Workgroup>
  return
}
// -----

//===----------------------------------------------------------------------===//
// spirv.GenericCastToPtrExplicit
//===----------------------------------------------------------------------===//

func.func @genericcasttoptrexplicit1(%arg0 : !spirv.ptr<vector<2xi32>, Generic>) {
  // CHECK: {{%.*}} = spirv.GenericCastToPtrExplicit {{%.*}} : !spirv.ptr<vector<2xi32>, Generic> to !spirv.ptr<vector<2xi32>, CrossWorkgroup>
  %0 = spirv.GenericCastToPtrExplicit %arg0 : !spirv.ptr<vector<2xi32>, Generic> to !spirv.ptr<vector<2xi32>, CrossWorkgroup>
  return
}
// -----

func.func @genericcasttoptrexplicit2(%arg0 : !spirv.ptr<f32, Uniform>) {
  // expected-error @+1 {{pointer type must be of storage class Generic}}
  %0 = spirv.GenericCastToPtrExplicit %arg0 : !spirv.ptr<f32, Uniform> to !spirv.ptr<f32, Workgroup>
  return
}

// -----

func.func @genericcasttoptrexplicit3(%arg0 : !spirv.ptr<f32, Generic>) {
  // expected-error @+1 {{result must point to the Workgroup, CrossWorkgroup, or Function Storage Class}}
  %0 = spirv.GenericCastToPtrExplicit %arg0 : !spirv.ptr<f32, Generic> to !spirv.ptr<f32, Uniform>
  return
}

// -----

func.func @genericcasttoptrexplicit4(%arg0 : !spirv.ptr<f32, Generic>) {
  // expected-error @+1 {{pointee type must have the same as the op result type}}
  %0 = spirv.GenericCastToPtrExplicit %arg0 : !spirv.ptr<f32, Generic> to !spirv.ptr<vector<2xi32>, Workgroup>
  return
}
