// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Load
//===----------------------------------------------------------------------===//

func @load() {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  //  CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm<"float*">
  %1 = spv.Load "Function" %0 : f32
  return
}

func @load_none() {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  //  CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm<"float*">
  %1 = spv.Load "Function" %0 ["None"] : f32
  return
}

func @load_with_alignment() {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: %{{.*}} = llvm.load %{{.*}} {alignment = 4 : i64} : !llvm<"float*">
  %1 = spv.Load "Function" %0 ["Aligned", 4] : f32
  return
}

//===----------------------------------------------------------------------===//
// spv.Store
//===----------------------------------------------------------------------===//

func @store(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: llvm.store %{{.*}}, %{{.*}} : !llvm<"float*">
  spv.Store "Function" %0, %arg0 : f32
  return
}

func @store_composite(%arg0 : !spv.struct<f64>) -> () {
  %0 = spv.Variable : !spv.ptr<!spv.struct<f64>, Function>
  // CHECK: llvm.store %{{.*}}, %{{.*}} : !llvm<"<{ double }>*">
  spv.Store "Function" %0, %arg0 : !spv.struct<f64>
  return
}

func @store_with_alignment(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : !llvm<"float*">
  spv.Store "Function" %0, %arg0 ["Aligned", 4] : f32
  return
}

//===----------------------------------------------------------------------===//
// spv.Variable
//===----------------------------------------------------------------------===//

func @variable_scalar() {
	// CHECK: %[[SIZE1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
  // CHECK: %{{.*}} = llvm.alloca %[[SIZE1]] x !llvm.float : (!llvm.i32) -> !llvm<"float*">
	%0 = spv.Variable : !spv.ptr<f32, Function>
	// CHECK: %[[SIZE2:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
  // CHECK: %{{.*}} = llvm.alloca %[[SIZE2]] x !llvm.i8 : (!llvm.i32) -> !llvm<"i8*">
	%1 = spv.Variable : !spv.ptr<i8, Function>
  return
}

func @variable_scalar_with_initialization() {
	// CHECK: %[[VALUE:.*]] = llvm.mlir.constant(0 : i64) : !llvm.i64
	// CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
  // CHECK: %[[ALLOCATED:.*]] = llvm.alloca %[[SIZE]] x !llvm.i64 : (!llvm.i32) -> !llvm<"i64*">
	// CHECK: llvm.store %[[VALUE]], %[[ALLOCATED]] : !llvm<"i64*">
	%c = spv.constant 0 : i64
	%0 = spv.Variable init(%c) : !spv.ptr<i64, Function>
  return
}

func @variable_vector() {
	// CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
	// CHECK: %{{.*}} = llvm.alloca  %[[SIZE]] x !llvm<"<3 x float>"> : (!llvm.i32) -> !llvm<"<3 x float>*">
	%0 = spv.Variable : !spv.ptr<vector<3xf32>, Function>
	return
}

func @variable_vector_with_initialization() {
	// CHECK: %[[VALUE:.*]] = llvm.mlir.constant(dense<false> : vector<3xi1>) : !llvm<"<3 x i1>">
	// CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
  // CHECK: %[[ALLOCATED:.*]] = llvm.alloca %[[SIZE]] x !llvm<"<3 x i1>"> : (!llvm.i32) -> !llvm<"<3 x i1>*">
	// CHECK: llvm.store %[[VALUE]], %[[ALLOCATED]] : !llvm<"<3 x i1>*">
	%c = spv.constant dense<false> : vector<3xi1>
	%0 = spv.Variable init(%c) : !spv.ptr<vector<3xi1>, Function>
  return
}

func @variable_array() {
	// CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
	// CHECK: %{{.*}} = llvm.alloca %[[SIZE]] x !llvm<"[10 x i32]"> : (!llvm.i32) -> !llvm<"[10 x i32]*">
	%0 = spv.Variable : !spv.ptr<!spv.array<10 x i32>, Function>
	return
}
