// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.IEqual
//===----------------------------------------------------------------------===//

func @i_equal_scalar(%arg0: i32, %arg1: i32) {
	// CHECK: %{{.*}} = llvm.icmp "eq" %{{.*}}, %{{.*}} : !llvm.i32
	%0 = spv.IEqual %arg0, %arg1 : i32
	return
}

func @i_equal_vector(%arg0: vector<4xi64>, %arg1: vector<4xi64>) {
	// CHECK: %{{.*}} = llvm.icmp "eq" %{{.*}}, %{{.*}} : !llvm<"<4 x i64>">
	%0 = spv.IEqual %arg0, %arg1 : vector<4xi64>
	return
}

//===----------------------------------------------------------------------===//
// spv.INotEqual
//===----------------------------------------------------------------------===//

func @i_not_equal_scalar(%arg0: i64, %arg1: i64) {
	// CHECK: %{{.*}} = llvm.icmp "ne" %{{.*}}, %{{.*}} : !llvm.i64
	%0 = spv.INotEqual %arg0, %arg1 : i64
	return
}

func @i_not_equal_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) {
	// CHECK: %{{.*}} = llvm.icmp "ne" %{{.*}}, %{{.*}} : !llvm<"<2 x i64>">
	%0 = spv.INotEqual %arg0, %arg1 : vector<2xi64>
	return
}

//===----------------------------------------------------------------------===//
// spv.SGreaterThanEqual
//===----------------------------------------------------------------------===//

func @s_greater_than_equal_scalar(%arg0: i64, %arg1: i64) {
	// CHECK: %{{.*}} = llvm.icmp "sge" %{{.*}}, %{{.*}} : !llvm.i64
	%0 = spv.SGreaterThanEqual %arg0, %arg1 : i64
	return
}

func @s_greater_than_equal_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) {
	// CHECK: %{{.*}} = llvm.icmp "sge" %{{.*}}, %{{.*}} : !llvm<"<2 x i64>">
	%0 = spv.SGreaterThanEqual %arg0, %arg1 : vector<2xi64>
	return
}

//===----------------------------------------------------------------------===//
// spv.SGreaterThan
//===----------------------------------------------------------------------===//

func @s_greater_than_scalar(%arg0: i64, %arg1: i64) {
	// CHECK: %{{.*}} = llvm.icmp "sgt" %{{.*}}, %{{.*}} : !llvm.i64
	%0 = spv.SGreaterThan %arg0, %arg1 : i64
	return
}

func @s_greater_than_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) {
	// CHECK: %{{.*}} = llvm.icmp "sgt" %{{.*}}, %{{.*}} : !llvm<"<2 x i64>">
	%0 = spv.SGreaterThan %arg0, %arg1 : vector<2xi64>
	return
}

//===----------------------------------------------------------------------===//
// spv.SLessThanEqual
//===----------------------------------------------------------------------===//

func @s_less_than_equal_scalar(%arg0: i64, %arg1: i64) {
	// CHECK: %{{.*}} = llvm.icmp "sle" %{{.*}}, %{{.*}} : !llvm.i64
	%0 = spv.SLessThanEqual %arg0, %arg1 : i64
	return
}

func @s_less_than_equal_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) {
	// CHECK: %{{.*}} = llvm.icmp "sle" %{{.*}}, %{{.*}} : !llvm<"<2 x i64>">
	%0 = spv.SLessThanEqual %arg0, %arg1 : vector<2xi64>
	return
}

//===----------------------------------------------------------------------===//
// spv.SLessThan
//===----------------------------------------------------------------------===//

func @s_less_than_scalar(%arg0: i64, %arg1: i64) {
	// CHECK: %{{.*}} = llvm.icmp "slt" %{{.*}}, %{{.*}} : !llvm.i64
	%0 = spv.SLessThan %arg0, %arg1 : i64
	return
}

func @s_less_than_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) {
	// CHECK: %{{.*}} = llvm.icmp "slt" %{{.*}}, %{{.*}} : !llvm<"<2 x i64>">
	%0 = spv.SLessThan %arg0, %arg1 : vector<2xi64>
	return
}

//===----------------------------------------------------------------------===//
// spv.UGreaterThanEqual
//===----------------------------------------------------------------------===//

func @u_greater_than_equal_scalar(%arg0: i64, %arg1: i64) {
	// CHECK: %{{.*}} = llvm.icmp "uge" %{{.*}}, %{{.*}} : !llvm.i64
	%0 = spv.UGreaterThanEqual %arg0, %arg1 : i64
	return
}

func @u_greater_than_equal_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) {
	// CHECK: %{{.*}} = llvm.icmp "uge" %{{.*}}, %{{.*}} : !llvm<"<2 x i64>">
	%0 = spv.UGreaterThanEqual %arg0, %arg1 : vector<2xi64>
	return
}

//===----------------------------------------------------------------------===//
// spv.UGreaterThan
//===----------------------------------------------------------------------===//

func @u_greater_than_scalar(%arg0: i64, %arg1: i64) {
	// CHECK: %{{.*}} = llvm.icmp "ugt" %{{.*}}, %{{.*}} : !llvm.i64
	%0 = spv.UGreaterThan %arg0, %arg1 : i64
	return
}

func @u_greater_than_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) {
	// CHECK: %{{.*}} = llvm.icmp "ugt" %{{.*}}, %{{.*}} : !llvm<"<2 x i64>">
	%0 = spv.UGreaterThan %arg0, %arg1 : vector<2xi64>
	return
}

//===----------------------------------------------------------------------===//
// spv.ULessThanEqual
//===----------------------------------------------------------------------===//

func @u_less_than_equal_scalar(%arg0: i64, %arg1: i64) {
	// CHECK: %{{.*}} = llvm.icmp "ule" %{{.*}}, %{{.*}} : !llvm.i64
	%0 = spv.ULessThanEqual %arg0, %arg1 : i64
	return
}

func @u_less_than_equal_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) {
	// CHECK: %{{.*}} = llvm.icmp "ule" %{{.*}}, %{{.*}} : !llvm<"<2 x i64>">
	%0 = spv.ULessThanEqual %arg0, %arg1 : vector<2xi64>
	return
}

//===----------------------------------------------------------------------===//
// spv.ULessThan
//===----------------------------------------------------------------------===//

func @u_less_than_scalar(%arg0: i64, %arg1: i64) {
	// CHECK: %{{.*}} = llvm.icmp "ult" %{{.*}}, %{{.*}} : !llvm.i64
	%0 = spv.ULessThan %arg0, %arg1 : i64
	return
}

func @u_less_than_vector(%arg0: vector<2xi64>, %arg1: vector<2xi64>) {
	// CHECK: %{{.*}} = llvm.icmp "ult" %{{.*}}, %{{.*}} : !llvm<"<2 x i64>">
	%0 = spv.ULessThan %arg0, %arg1 : vector<2xi64>
	return
}

//===----------------------------------------------------------------------===//
// spv.FOrdEqual
//===----------------------------------------------------------------------===//

func @f_ord_equal_scalar(%arg0: f32, %arg1: f32) {
	// CHECK: %{{.*}} = llvm.fcmp "oeq" %{{.*}}, %{{.*}} : !llvm.float
	%0 = spv.FOrdEqual %arg0, %arg1 : f32
	return
}

func @f_ord_equal_vector(%arg0: vector<4xf64>, %arg1: vector<4xf64>) {
	// CHECK: %{{.*}} = llvm.fcmp "oeq" %{{.*}}, %{{.*}} : !llvm<"<4 x double>">
	%0 = spv.FOrdEqual %arg0, %arg1 : vector<4xf64>
	return
}

//===----------------------------------------------------------------------===//
// spv.FOrdGreaterThanEqual
//===----------------------------------------------------------------------===//

func @f_ord_greater_than_equal_scalar(%arg0: f64, %arg1: f64) {
	// CHECK: %{{.*}} = llvm.fcmp "oge" %{{.*}}, %{{.*}} : !llvm.double
	%0 = spv.FOrdGreaterThanEqual %arg0, %arg1 : f64
	return
}

func @f_ord_greater_than_equal_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) {
	// CHECK: %{{.*}} = llvm.fcmp "oge" %{{.*}}, %{{.*}} : !llvm<"<2 x double>">
	%0 = spv.FOrdGreaterThanEqual %arg0, %arg1 : vector<2xf64>
	return
}

//===----------------------------------------------------------------------===//
// spv.FOrdGreaterThan
//===----------------------------------------------------------------------===//

func @f_ord_greater_than_scalar(%arg0: f64, %arg1: f64) {
	// CHECK: %{{.*}} = llvm.fcmp "ogt" %{{.*}}, %{{.*}} : !llvm.double
	%0 = spv.FOrdGreaterThan %arg0, %arg1 : f64
	return
}

func @f_ord_greater_than_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) {
	// CHECK: %{{.*}} = llvm.fcmp "ogt" %{{.*}}, %{{.*}} : !llvm<"<2 x double>">
	%0 = spv.FOrdGreaterThan %arg0, %arg1 : vector<2xf64>
	return
}

//===----------------------------------------------------------------------===//
// spv.FOrdLessThan
//===----------------------------------------------------------------------===//

func @f_ord_less_than_scalar(%arg0: f64, %arg1: f64) {
	// CHECK: %{{.*}} = llvm.fcmp "olt" %{{.*}}, %{{.*}} : !llvm.double
	%0 = spv.FOrdLessThan %arg0, %arg1 : f64
	return
}

func @f_ord_less_than_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) {
	// CHECK: %{{.*}} = llvm.fcmp "olt" %{{.*}}, %{{.*}} : !llvm<"<2 x double>">
	%0 = spv.FOrdLessThan %arg0, %arg1 : vector<2xf64>
	return
}

//===----------------------------------------------------------------------===//
// spv.FOrdLessThanEqual
//===----------------------------------------------------------------------===//

func @f_ord_less_than_equal_scalar(%arg0: f64, %arg1: f64) {
	// CHECK: %{{.*}} = llvm.fcmp "ole" %{{.*}}, %{{.*}} : !llvm.double
	%0 = spv.FOrdLessThanEqual %arg0, %arg1 : f64
	return
}

func @f_ord_less_than_equal_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) {
	// CHECK: %{{.*}} = llvm.fcmp "ole" %{{.*}}, %{{.*}} : !llvm<"<2 x double>">
	%0 = spv.FOrdLessThanEqual %arg0, %arg1 : vector<2xf64>
	return
}

//===----------------------------------------------------------------------===//
// spv.FOrdNotEqual
//===----------------------------------------------------------------------===//

func @f_ord_not_equal_scalar(%arg0: f32, %arg1: f32) {
	// CHECK: %{{.*}} = llvm.fcmp "one" %{{.*}}, %{{.*}} : !llvm.float
	%0 = spv.FOrdNotEqual %arg0, %arg1 : f32
	return
}

func @f_ord_not_equal_vector(%arg0: vector<4xf64>, %arg1: vector<4xf64>) {
	// CHECK: %{{.*}} = llvm.fcmp "one" %{{.*}}, %{{.*}} : !llvm<"<4 x double>">
	%0 = spv.FOrdNotEqual %arg0, %arg1 : vector<4xf64>
	return
}

//===----------------------------------------------------------------------===//
// spv.FUnordEqual
//===----------------------------------------------------------------------===//

func @f_unord_equal_scalar(%arg0: f32, %arg1: f32) {
	// CHECK: %{{.*}} = llvm.fcmp "ueq" %{{.*}}, %{{.*}} : !llvm.float
	%0 = spv.FUnordEqual %arg0, %arg1 : f32
	return
}

func @f_unord_equal_vector(%arg0: vector<4xf64>, %arg1: vector<4xf64>) {
	// CHECK: %{{.*}} = llvm.fcmp "ueq" %{{.*}}, %{{.*}} : !llvm<"<4 x double>">
	%0 = spv.FUnordEqual %arg0, %arg1 : vector<4xf64>
	return
}

//===----------------------------------------------------------------------===//
// spv.FUnordGreaterThanEqual
//===----------------------------------------------------------------------===//

func @f_unord_greater_than_equal_scalar(%arg0: f64, %arg1: f64) {
	// CHECK: %{{.*}} = llvm.fcmp "uge" %{{.*}}, %{{.*}} : !llvm.double
	%0 = spv.FUnordGreaterThanEqual %arg0, %arg1 : f64
	return
}

func @f_unord_greater_than_equal_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) {
	// CHECK: %{{.*}} = llvm.fcmp "uge" %{{.*}}, %{{.*}} : !llvm<"<2 x double>">
	%0 = spv.FUnordGreaterThanEqual %arg0, %arg1 : vector<2xf64>
	return
}

//===----------------------------------------------------------------------===//
// spv.FUnordGreaterThan
//===----------------------------------------------------------------------===//

func @f_unord_greater_than_scalar(%arg0: f64, %arg1: f64) {
	// CHECK: %{{.*}} = llvm.fcmp "ugt" %{{.*}}, %{{.*}} : !llvm.double
	%0 = spv.FUnordGreaterThan %arg0, %arg1 : f64
	return
}

func @f_unord_greater_than_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) {
	// CHECK: %{{.*}} = llvm.fcmp "ugt" %{{.*}}, %{{.*}} : !llvm<"<2 x double>">
	%0 = spv.FUnordGreaterThan %arg0, %arg1 : vector<2xf64>
	return
}

//===----------------------------------------------------------------------===//
// spv.FUnordLessThan
//===----------------------------------------------------------------------===//

func @f_unord_less_than_scalar(%arg0: f64, %arg1: f64) {
	// CHECK: %{{.*}} = llvm.fcmp "ult" %{{.*}}, %{{.*}} : !llvm.double
	%0 = spv.FUnordLessThan %arg0, %arg1 : f64
	return
}

func @f_unord_less_than_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) {
	// CHECK: %{{.*}} = llvm.fcmp "ult" %{{.*}}, %{{.*}} : !llvm<"<2 x double>">
	%0 = spv.FUnordLessThan %arg0, %arg1 : vector<2xf64>
	return
}

//===----------------------------------------------------------------------===//
// spv.FUnordLessThanEqual
//===----------------------------------------------------------------------===//

func @f_unord_less_than_equal_scalar(%arg0: f64, %arg1: f64) {
	// CHECK: %{{.*}} = llvm.fcmp "ule" %{{.*}}, %{{.*}} : !llvm.double
	%0 = spv.FUnordLessThanEqual %arg0, %arg1 : f64
	return
}

func @f_unord_less_than_equal_vector(%arg0: vector<2xf64>, %arg1: vector<2xf64>) {
	// CHECK: %{{.*}} = llvm.fcmp "ule" %{{.*}}, %{{.*}} : !llvm<"<2 x double>">
	%0 = spv.FUnordLessThanEqual %arg0, %arg1 : vector<2xf64>
	return
}

//===----------------------------------------------------------------------===//
// spv.FUnordNotEqual
//===----------------------------------------------------------------------===//

func @f_unord_not_equal_scalar(%arg0: f32, %arg1: f32) {
	// CHECK: %{{.*}} = llvm.fcmp "une" %{{.*}}, %{{.*}} : !llvm.float
	%0 = spv.FUnordNotEqual %arg0, %arg1 : f32
	return
}

func @f_unord_not_equal_vector(%arg0: vector<4xf64>, %arg1: vector<4xf64>) {
	// CHECK: %{{.*}} = llvm.fcmp "une" %{{.*}}, %{{.*}} : !llvm<"<4 x double>">
	%0 = spv.FUnordNotEqual %arg0, %arg1 : vector<4xf64>
	return
}
