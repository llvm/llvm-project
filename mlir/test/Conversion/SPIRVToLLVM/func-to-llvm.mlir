// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Return
//===----------------------------------------------------------------------===//

func @return() {
	// CHECK: llvm.return
	spv.Return
}

//===----------------------------------------------------------------------===//
// spv.ReturnValue
//===----------------------------------------------------------------------===//

func @return_value(%arg: i32) {
	// CHECK: llvm.return %{{.*}} : !llvm.i32
	spv.ReturnValue %arg : i32
}

//===----------------------------------------------------------------------===//
// spv.func
//===----------------------------------------------------------------------===//

// CHECK-LABEL: llvm.func @none()
spv.func @none() -> () "None" {
	spv.Return
}

// CHECK-LABEL: llvm.func @inline() attributes {passthrough = ["alwaysinline"]}
spv.func @inline() -> () "Inline" {
	spv.Return
}

// CHECK-LABEL: llvm.func @dont_inline() attributes {passthrough = ["noinline"]}
spv.func @dont_inline() -> () "DontInline" {
	spv.Return
}

// CHECK-LABEL: llvm.func @pure() attributes {passthrough = ["readonly"]}
spv.func @pure() -> () "Pure" {
	spv.Return
}

// CHECK-LABEL: llvm.func @const() attributes {passthrough = ["readnone"]}
spv.func @const() -> () "Const" {
	spv.Return
}

// CHECK-LABEL: llvm.func @scalar_types(%arg0: !llvm.i32, %arg1: !llvm.i1, %arg2: !llvm.double, %arg3: !llvm.float)
spv.func @scalar_types(%arg0: i32, %arg1: i1, %arg2: f64, %arg3: f32) -> () "None" {
	spv.Return
}

// CHECK-LABEL: llvm.func @vector_types(%arg0: !llvm<"<2 x i64>">, %arg1: !llvm<"<2 x i64>">) -> !llvm<"<2 x i64>">
spv.func @vector_types(%arg0: vector<2xi64>, %arg1: vector<2xi64>) -> vector<2xi64> "None" {
	%0 = spv.IAdd %arg0, %arg1 : vector<2xi64>
	spv.ReturnValue %0 : vector<2xi64>
}



