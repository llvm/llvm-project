// RUN: mlir-opt %s | mlir-rewrite --simple-rename --simple-rename-op-name="test.concat" --simple-rename-match="axis" --simple-rename-replace="bxis" | FileCheck %s -check-prefix=RENAME
// RUN: mlir-opt %s | mlir-rewrite --mark-ranges | FileCheck %s -check-prefix=RANGE
// Note: running through mlir-opt to just strip out comments & avoid self matches.

func.func @two_dynamic_one_direct_shape(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) -> tensor<?x4x?xf32> {
  // RENAME: "test.concat"({{.*}}) {bxis = 0 : i64}
  // RANGE: <%{{.*}} = ["test.concat"]({{.*}}) {axis = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x4x?xf32>) -> tensor<?x4x?xf32>>
  %5 = "test.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
  return %5 : tensor<?x4x?xf32>
}
