// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Bitcast
//===----------------------------------------------------------------------===//

func @bitcast_float_to_integer_scalar(%arg0 : f32) {
	// CHECK: {{.*}} = llvm.bitcast {{.*}} : !llvm.float to !llvm.i32
	%0 = spv.Bitcast %arg0: f32 to i32
	return
}

func @bitcast_float_to_integer_vector(%arg0 : vector<3xf32>) {
	// CHECK: {{.*}} = llvm.bitcast {{.*}} : !llvm<"<3 x float>"> to !llvm<"<3 x i32>">
	%0 = spv.Bitcast %arg0: vector<3xf32> to vector<3xi32>
	return
}

func @bitcast_vector_to_scalar(%arg0 : vector<2xf32>) {
	// CHECK: {{.*}} = llvm.bitcast {{.*}} : !llvm<"<2 x float>"> to !llvm.i64
	%0 = spv.Bitcast %arg0: vector<2xf32> to i64
	return
}

func @bitcast_scalar_to_vector(%arg0 : f64) {
	// CHECK: {{.*}} = llvm.bitcast {{.*}} : !llvm.double to !llvm<"<2 x i32>">
	%0 = spv.Bitcast %arg0: f64 to vector<2xi32>
	return
}

func @bitcast_vector_to_vector(%arg0 : vector<4xf32>) {
	// CHECK: {{.*}} = llvm.bitcast {{.*}} : !llvm<"<4 x float>"> to !llvm<"<2 x i64>">
	%0 = spv.Bitcast %arg0: vector<4xf32> to vector<2xi64>
	return
}

func @bitcast_pointer(%arg0: !spv.ptr<f32, Function>) {
	// CHECK: %{{.*}} = llvm.bitcast %{{.*}} : !llvm<"float*"> to !llvm<"i32*">
	%0 = spv.Bitcast %arg0 : !spv.ptr<f32, Function> to !spv.ptr<i32, Function>
	return
}

//===----------------------------------------------------------------------===//
// spv.ConvertFToS
//===----------------------------------------------------------------------===//

func @convert_float_to_signed_scalar(%arg0: f32) {
	// CHECK: %{{.*}} = llvm.fptosi %{{.*}} : !llvm.float to !llvm.i32
    %0 = spv.ConvertFToS %arg0: f32 to i32
	return
}

func @convert_float_to_signed_vector(%arg0: vector<2xf32>) {
	// CHECK: %{{.*}} = llvm.fptosi %{{.*}} : !llvm<"<2 x float>"> to !llvm<"<2 x i32>">
    %0 = spv.ConvertFToS %arg0: vector<2xf32> to vector<2xi32>
	return
}

//===----------------------------------------------------------------------===//
// spv.ConvertFToU
//===----------------------------------------------------------------------===//

func @convert_float_to_unsigned_scalar(%arg0: f32) {
	// CHECK: %{{.*}} = llvm.fptoui %{{.*}} : !llvm.float to !llvm.i32
    %0 = spv.ConvertFToU %arg0: f32 to i32
	return
}

func @convert_float_to_unsigned_vector(%arg0: vector<2xf32>) {
	// CHECK: %{{.*}} = llvm.fptoui %{{.*}} : !llvm<"<2 x float>"> to !llvm<"<2 x i32>">
    %0 = spv.ConvertFToU %arg0: vector<2xf32> to vector<2xi32>
	return
}

//===----------------------------------------------------------------------===//
// spv.ConvertSToF
//===----------------------------------------------------------------------===//

func @convert_signed_to_float_scalar(%arg0: i32) {
	// CHECK: %{{.*}} = llvm.sitofp %{{.*}} : !llvm.i32 to !llvm.float
    %0 = spv.ConvertSToF %arg0: i32 to f32
	return
}

func @convert_signed_to_float_vector(%arg0: vector<3xi32>) {
	// CHECK: %{{.*}} = llvm.sitofp %{{.*}} : !llvm<"<3 x i32>"> to !llvm<"<3 x float>">
    %0 = spv.ConvertSToF %arg0: vector<3xi32> to vector<3xf32>
	return
}

//===----------------------------------------------------------------------===//
// spv.ConvertUToF
//===----------------------------------------------------------------------===//

func @convert_unsigned_to_float_scalar(%arg0: i32) {
	// CHECK: %{{.*}} = llvm.uitofp %{{.*}} : !llvm.i32 to !llvm.float
    %0 = spv.ConvertUToF %arg0: i32 to f32
	return
}

func @convert_unsigned_to_float_vector(%arg0: vector<3xi32>) {
	// CHECK: %{{.*}} = llvm.uitofp %{{.*}} : !llvm<"<3 x i32>"> to !llvm<"<3 x float>">
    %0 = spv.ConvertUToF %arg0: vector<3xi32> to vector<3xf32>
	return
}

//===----------------------------------------------------------------------===//
// spv.FConvert
//===----------------------------------------------------------------------===//

func @fconvert_scalar(%arg0: f32, %arg1: f64) {
	// CHECK: %{{.*}} = llvm.fpext %{{.*}} : !llvm.float to !llvm.double
    %0 = spv.FConvert %arg0: f32 to f64

    // CHECK: %{{.*}} = llvm.fptrunc %{{.*}} : !llvm.double to !llvm.float
    %1 = spv.FConvert %arg1: f64 to f32
	return
}

func @fconvert_vector(%arg0: vector<2xf32>, %arg1: vector<2xf64>) {
	// CHECK: %{{.*}} = llvm.fpext %{{.*}} : !llvm<"<2 x float>"> to !llvm<"<2 x double>">
    %0 = spv.FConvert %arg0: vector<2xf32> to vector<2xf64>

    // CHECK: %{{.*}} = llvm.fptrunc %{{.*}} : !llvm<"<2 x double>"> to !llvm<"<2 x float>">
    %1 = spv.FConvert %arg1: vector<2xf64> to vector<2xf32>
	return
}

//===----------------------------------------------------------------------===//
// spv.SConvert
//===----------------------------------------------------------------------===//

func @sconvert_scalar(%arg0: i32, %arg1: i64) {
	// CHECK: %{{.*}} = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %0 = spv.SConvert %arg0: i32 to i64

    // CHECK: %{{.*}} = llvm.trunc %{{.*}} : !llvm.i64 to !llvm.i32
    %1 = spv.SConvert %arg1: i64 to i32
	return
}

func @sconvert_vector(%arg0: vector<3xi32>, %arg1: vector<3xi64>) {
	// CHECK: %{{.*}} = llvm.sext %{{.*}} : !llvm<"<3 x i32>"> to !llvm<"<3 x i64>">
    %0 = spv.SConvert %arg0: vector<3xi32> to vector<3xi64>

    // CHECK: %{{.*}} = llvm.trunc %{{.*}} : !llvm<"<3 x i64>"> to !llvm<"<3 x i32>">
    %1 = spv.SConvert %arg1: vector<3xi64> to vector<3xi32>
	return
}

//===----------------------------------------------------------------------===//
// spv.UConvert
//===----------------------------------------------------------------------===//

func @uconvert_scalar(%arg0: i32, %arg1: i64) {
	// CHECK: %{{.*}} = llvm.zext %{{.*}} : !llvm.i32 to !llvm.i64
    %0 = spv.UConvert %arg0: i32 to i64

    // CHECK: %{{.*}} = llvm.trunc %{{.*}} : !llvm.i64 to !llvm.i32
    %1 = spv.UConvert %arg1: i64 to i32
	return
}

func @uconvert_vector(%arg0: vector<3xi32>, %arg1: vector<3xi64>) {
	// CHECK: %{{.*}} = llvm.zext %{{.*}} : !llvm<"<3 x i32>"> to !llvm<"<3 x i64>">
    %0 = spv.UConvert %arg0: vector<3xi32> to vector<3xi64>

    // CHECK: %{{.*}} = llvm.trunc %{{.*}} : !llvm<"<3 x i64>"> to !llvm<"<3 x i32>">
    %1 = spv.UConvert %arg1: vector<3xi64> to vector<3xi32>
	return
}
