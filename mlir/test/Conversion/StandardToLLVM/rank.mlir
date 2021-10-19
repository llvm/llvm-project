// RUN: mlir-opt -convert-std-to-llvm %s -split-input-file | FileCheck %s
// RUN: mlir-opt -convert-std-to-llvm='index-bitwidth=32' %s -split-input-file | FileCheck --check-prefix=CHECK32 %s

// CHECK-LABEL: func @rank_of_unranked
// CHECK32-LABEL: func @rank_of_unranked
func @rank_of_unranked(%unranked: memref<*xi32>) {
  %rank = rank %unranked : memref<*xi32>
  return
}
// CHECK-NEXT: llvm.mlir.undef
// CHECK-NEXT: llvm.insertvalue
// CHECK-NEXT: llvm.insertvalue
// CHECK-NEXT: llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i64, ptr<i8>)>
// CHECK32: llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i32, ptr<i8>)>

// CHECK-LABEL: func @rank_of_ranked
// CHECK32-LABEL: func @rank_of_ranked
func @rank_of_ranked(%ranked: memref<?xi32>) {
  %rank = rank %ranked : memref<?xi32>
  return
}
// CHECK: llvm.mlir.constant(1 : index) : i64
// CHECK32: llvm.mlir.constant(1 : index) : i32
