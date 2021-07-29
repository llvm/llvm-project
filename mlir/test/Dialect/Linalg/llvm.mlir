// RUN: mlir-opt %s -convert-linalg-to-llvm | FileCheck %s

func @range(%arg0: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %R = linalg.range %c0:%arg0:%c1 : !linalg.range
  return
}
// CHECK-LABEL: func @range
//       CHECK:   constant 0 : index
//       CHECK:   constant 1 : index
//       CHECK:   llvm.mlir.undef : !llvm.struct<(i64, i64, i64)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(i64, i64, i64)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(i64, i64, i64)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(i64, i64, i64)>
