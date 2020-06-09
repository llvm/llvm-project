// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

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
