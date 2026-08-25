// The memref descriptor fields use the converted index type, which is not
// necessarily `i64`.

// RUN: mlir-opt %s --convert-to-llvm | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<index, 32>,
>} {
  // CHECK-LABEL: llvm.func @type_cast
  //       CHECK:   %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : i32
  //       CHECK:   llvm.insertvalue %[[OFFSET]], %{{.*}}[2] : !llvm.struct<(ptr, ptr, i32)>
  func.func @type_cast(%arg0: memref<8x8x8xf32>) -> memref<vector<8x8x8xf32>> {
    %0 = vector.type_cast %arg0 : memref<8x8x8xf32> to memref<vector<8x8x8xf32>>
    return %0 : memref<vector<8x8x8xf32>>
  }
}
