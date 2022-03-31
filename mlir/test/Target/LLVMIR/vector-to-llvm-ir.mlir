// RUN: mlir-opt %s -pass-pipeline="convert-vector-to-llvm,func.func(convert-arith-to-llvm),convert-func-to-llvm,reconcile-unrealized-casts" | mlir-translate -mlir-to-llvmir | FileCheck %s

func @genbool_1d() -> vector<8xi1> {
  %0 = vector.constant_mask [4] : vector<8xi1>
  return %0 : vector<8xi1>
}
// CHECK-LABEL: @genbool_1d()
// CHECK-NEXT: ret <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false>

func @genbool_2d() -> vector<4x4xi1> {
  %v = vector.constant_mask [2, 2] : vector<4x4xi1>
  return %v: vector<4x4xi1>
}
// CHECK-LABEL: @genbool_2d()
// CHECK-NEXT: ret [4 x <4 x i1>] [<4 x i1> <i1 true, i1 true, i1 false, i1 false>, <4 x i1> <i1 true, i1 true, i1 false, i1 false>, <4 x i1> zeroinitializer, <4 x i1> zeroinitializer]

func @genbool_3d() -> vector<2x3x4xi1> {
  %v = vector.constant_mask [1, 1, 3] : vector<2x3x4xi1>
  return %v: vector<2x3x4xi1>
}
// CHECK-LABEL: @genbool_3d()
// CHECK-NEXT: ret [2 x [3 x <4 x i1>]] {{\[+}}3 x <4 x i1>] [<4 x i1> <i1 true, i1 true, i1 true, i1 false>, <4 x i1> zeroinitializer, <4 x i1> zeroinitializer], [3 x <4 x i1>] zeroinitializer]
// note: awkward syntax to match [[

func @genbool_1d_var_but_constant() -> vector<8xi1> {
  %i = arith.constant 0 : index
  %v = vector.create_mask %i : vector<8xi1>
  return %v : vector<8xi1>
}
// CHECK-LABEL: @genbool_1d_var_but_constant()
// CHECK-NEXT: ret <8 x i1> zeroinitializer
