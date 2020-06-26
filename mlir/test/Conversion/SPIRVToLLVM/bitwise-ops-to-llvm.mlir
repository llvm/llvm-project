// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.BitCount
//===----------------------------------------------------------------------===//

func @bitcount_scalar(%arg0: i16) {
	// CHECK: %{{.*}} = "llvm.intr.ctpop"(%{{.*}}) : (!llvm.i16) -> !llvm.i16
	%0 = spv.BitCount %arg0: i16
	return
}

func @bitcount_vector(%arg0: vector<3xi32>) {
	// CHECK: %{{.*}} = "llvm.intr.ctpop"(%{{.*}}) : (!llvm<"<3 x i32>">) -> !llvm<"<3 x i32>">
	%0 = spv.BitCount %arg0: vector<3xi32>
	return
}

//===----------------------------------------------------------------------===//
// spv.BitReverse
//===----------------------------------------------------------------------===//

func @bitreverse_scalar(%arg0: i64) {
	// CHECK: %{{.*}} = "llvm.intr.bitreverse"(%{{.*}}) : (!llvm.i64) -> !llvm.i64
	%0 = spv.BitReverse %arg0: i64
	return
}

func @bitreverse_vector(%arg0: vector<4xi32>) {
	// CHECK: %{{.*}} = "llvm.intr.bitreverse"(%{{.*}}) : (!llvm<"<4 x i32>">) -> !llvm<"<4 x i32>">
	%0 = spv.BitReverse %arg0: vector<4xi32>
	return
}

//===----------------------------------------------------------------------===//
// spv.BitwiseAnd
//===----------------------------------------------------------------------===//

func @bitwise_and_scalar(%arg0: i32, %arg1: i32) {
	// CHECK: %{{.*}} = llvm.and %{{.*}}, %{{.*}} : !llvm.i32
	%0 = spv.BitwiseAnd %arg0, %arg1 : i32
	return
}

func @bitwise_and_vector(%arg0: vector<4xi64>, %arg1: vector<4xi64>) {
	// CHECK: %{{.*}} = llvm.and %{{.*}}, %{{.*}} : !llvm<"<4 x i64>">
	%0 = spv.BitwiseAnd %arg0, %arg1 : vector<4xi64>
	return
}

//===----------------------------------------------------------------------===//
// spv.BitwiseOr
//===----------------------------------------------------------------------===//

func @bitwise_or_scalar(%arg0: i64, %arg1: i64) {
	// CHECK: %{{.*}} = llvm.or %{{.*}}, %{{.*}} : !llvm.i64
	%0 = spv.BitwiseOr %arg0, %arg1 : i64
	return
}

func @bitwise_or_vector(%arg0: vector<3xi8>, %arg1: vector<3xi8>) {
	// CHECK: %{{.*}} = llvm.or %{{.*}}, %{{.*}} : !llvm<"<3 x i8>">
	%0 = spv.BitwiseOr %arg0, %arg1 : vector<3xi8>
	return
}

//===----------------------------------------------------------------------===//
// spv.BitwiseXor
//===----------------------------------------------------------------------===//

func @bitwise_xor_scalar(%arg0: i32, %arg1: i32) {
	// CHECK: %{{.*}} = llvm.xor %{{.*}}, %{{.*}} : !llvm.i32
	%0 = spv.BitwiseXor %arg0, %arg1 : i32
	return
}

func @bitwise_xor_vector(%arg0: vector<2xi16>, %arg1: vector<2xi16>) {
	// CHECK: %{{.*}} = llvm.xor %{{.*}}, %{{.*}} : !llvm<"<2 x i16>">
	%0 = spv.BitwiseXor %arg0, %arg1 : vector<2xi16>
	return
}
