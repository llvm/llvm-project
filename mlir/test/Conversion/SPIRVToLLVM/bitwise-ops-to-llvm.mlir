// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

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
