// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s


func.func @roundtripOperandSizeAttr(%arg0: i32) {
  // CHECK: odsOperandSegmentSizes = array<i32: 0, 2, 1, 1>}>
  "test.attr_sized_operands"(%arg0, %arg0, %arg0, %arg0) <{odsOperandSegmentSizes = array<i32: 0, 2, 1, 1>}> : (i32, i32, i32, i32) -> ()
  return
}
