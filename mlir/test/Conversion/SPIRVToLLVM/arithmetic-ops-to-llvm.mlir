// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.IAdd
//===----------------------------------------------------------------------===//

func @iadd_scalar(%arg0: i32, %arg1: i32) {
	// CHECK: %{{.*}} = llvm.add %{{.*}}, %{{.*}} : !llvm.i32
	%0 = spv.IAdd %arg0, %arg1 : i32
	return
}

func @iadd_vector(%arg0: vector<4xi64>, %arg1: vector<4xi64>) {
	// CHECK: %{{.*}} = llvm.add %{{.*}}, %{{.*}} : !llvm<"<4 x i64>">
	%0 = spv.IAdd %arg0, %arg1 : vector<4xi64>
	return
}

//===----------------------------------------------------------------------===//
// spv.ISub
//===----------------------------------------------------------------------===//

func @isub_scalar(%arg0: i8, %arg1: i8) {
	// CHECK: %{{.*}} = llvm.sub %{{.*}}, %{{.*}} : !llvm.i8
	%0 = spv.ISub %arg0, %arg1 : i8
	return
}

func @isub_vector(%arg0: vector<2xi16>, %arg1: vector<2xi16>) {
	// CHECK: %{{.*}} = llvm.sub %{{.*}}, %{{.*}} : !llvm<"<2 x i16>">
	%0 = spv.ISub %arg0, %arg1 : vector<2xi16>
	return
}

//===----------------------------------------------------------------------===//
// spv.IMul
//===----------------------------------------------------------------------===//

func @imul_scalar(%arg0: i32, %arg1: i32) {
	// CHECK: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : !llvm.i32
	%0 = spv.IMul %arg0, %arg1 : i32
	return
}

func @imul_vector(%arg0: vector<3xi32>, %arg1: vector<3xi32>) {
	// CHECK: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : !llvm<"<3 x i32>">
	%0 = spv.IMul %arg0, %arg1 : vector<3xi32>
	return
}

//===----------------------------------------------------------------------===//
// spv.FAdd
//===----------------------------------------------------------------------===//

func @fadd_scalar(%arg0: f16, %arg1: f16) {
	// CHECK: %{{.*}} = llvm.fadd %{{.*}}, %{{.*}} : !llvm.half
	%0 = spv.FAdd %arg0, %arg1 : f16
	return
}

func @fadd_vector(%arg0: vector<4xf32>, %arg1: vector<4xf32>) {
	// CHECK: %{{.*}} = llvm.fadd %{{.*}}, %{{.*}} : !llvm<"<4 x float>">
	%0 = spv.FAdd %arg0, %arg1 : vector<4xf32>
	return
}

//===----------------------------------------------------------------------===//
// spv.FSub
//===----------------------------------------------------------------------===//

func @fsub_scalar(%arg0: f32, %arg1: f32) {
	// CHECK: %{{.*}} = llvm.fsub %{{.*}}, %{{.*}} : !llvm.float
	%0 = spv.FSub %arg0, %arg1 : f32
	return
}

func @fsub_vector(%arg0: vector<2xf32>, %arg1: vector<2xf32>) {
	// CHECK: %{{.*}} = llvm.fsub %{{.*}}, %{{.*}} : !llvm<"<2 x float>">
	%0 = spv.FSub %arg0, %arg1 : vector<2xf32>
	return
}

//===----------------------------------------------------------------------===//
// spv.FDiv
//===----------------------------------------------------------------------===//

func @fdiv_scalar(%arg0: f32, %arg1: f32) {
	// CHECK: %{{.*}} = llvm.fdiv %{{.*}}, %{{.*}} : !llvm.float
	%0 = spv.FDiv %arg0, %arg1 : f32
	return
}

func @fdiv_vector(%arg0: vector<3xf64>, %arg1: vector<3xf64>) {
	// CHECK: %{{.*}} = llvm.fdiv %{{.*}}, %{{.*}} : !llvm<"<3 x double>">
	%0 = spv.FDiv %arg0, %arg1 : vector<3xf64>
	return
}

//===----------------------------------------------------------------------===//
// spv.FMul
//===----------------------------------------------------------------------===//

func @fmul_scalar(%arg0: f32, %arg1: f32) {
	// CHECK: %{{.*}} = llvm.fmul %{{.*}}, %{{.*}} : !llvm.float
	%0 = spv.FMul %arg0, %arg1 : f32
	return
}

func @fmul_vector(%arg0: vector<2xf32>, %arg1: vector<2xf32>) {
	// CHECK: %{{.*}} = llvm.fmul %{{.*}}, %{{.*}} : !llvm<"<2 x float>">
	%0 = spv.FMul %arg0, %arg1 : vector<2xf32>
	return
}

//===----------------------------------------------------------------------===//
// spv.FRem
//===----------------------------------------------------------------------===//

func @frem_scalar(%arg0: f32, %arg1: f32) {
	// CHECK: %{{.*}} = llvm.frem %{{.*}}, %{{.*}} : !llvm.float
	%0 = spv.FRem %arg0, %arg1 : f32
	return
}

func @frem_vector(%arg0: vector<3xf64>, %arg1: vector<3xf64>) {
	// CHECK: %{{.*}} = llvm.frem %{{.*}}, %{{.*}} : !llvm<"<3 x double>">
	%0 = spv.FRem %arg0, %arg1 : vector<3xf64>
	return
}

//===----------------------------------------------------------------------===//
// spv.FNegate
//===----------------------------------------------------------------------===//

func @fneg_scalar(%arg: f64) {
	// CHECK: %{{.*}} = llvm.fneg %{{.*}} : !llvm.double
	%0 = spv.FNegate %arg : f64
	return
}

func @fneg_vector(%arg: vector<2xf32>) {
	// CHECK: %{{.*}} = llvm.fneg %{{.*}} : !llvm<"<2 x float>">
	%0 = spv.FNegate %arg : vector<2xf32>
	return
}

//===----------------------------------------------------------------------===//
// spv.UDiv
//===----------------------------------------------------------------------===//

func @udiv_scalar(%arg0: i32, %arg1: i32) {
	// CHECK: %{{.*}} = llvm.udiv %{{.*}}, %{{.*}} : !llvm.i32
	%0 = spv.UDiv %arg0, %arg1 : i32
	return
}

func @udiv_vector(%arg0: vector<3xi64>, %arg1: vector<3xi64>) {
	// CHECK: %{{.*}} = llvm.udiv %{{.*}}, %{{.*}} : !llvm<"<3 x i64>">
	%0 = spv.UDiv %arg0, %arg1 : vector<3xi64>
	return
}

//===----------------------------------------------------------------------===//
// spv.UMod
//===----------------------------------------------------------------------===//

func @umod_scalar(%arg0: i32, %arg1: i32) {
	// CHECK: %{{.*}} = llvm.urem %{{.*}}, %{{.*}} : !llvm.i32
	%0 = spv.UMod %arg0, %arg1 : i32
	return
}

func @umod_vector(%arg0: vector<3xi64>, %arg1: vector<3xi64>) {
	// CHECK: %{{.*}} = llvm.urem %{{.*}}, %{{.*}} : !llvm<"<3 x i64>">
	%0 = spv.UMod %arg0, %arg1 : vector<3xi64>
	return
}

//===----------------------------------------------------------------------===//
// spv.SDiv
//===----------------------------------------------------------------------===//

func @sdiv_scalar(%arg0: i16, %arg1: i16) {
	// CHECK: %{{.*}} = llvm.sdiv %{{.*}}, %{{.*}} : !llvm.i16
	%0 = spv.SDiv %arg0, %arg1 : i16
	return
}

func @sdiv_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) {
	// CHECK: %{{.*}} = llvm.sdiv %{{.*}}, %{{.*}} : !llvm<"<2 x i64>">
	%0 = spv.SDiv %arg0, %arg1 : vector<2xi64>
	return
}

//===----------------------------------------------------------------------===//
// spv.SRem
//===----------------------------------------------------------------------===//

func @srem_scalar(%arg0: i32, %arg1: i32) {
	// CHECK: %{{.*}} = llvm.srem %{{.*}}, %{{.*}} : !llvm.i32
	%0 = spv.SRem %arg0, %arg1 : i32
	return
}

func @srem_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) {
	// CHECK: %{{.*}} = llvm.srem %{{.*}}, %{{.*}} : !llvm<"<4 x i32>">
	%0 = spv.SRem %arg0, %arg1 : vector<4xi32>
	return
}
