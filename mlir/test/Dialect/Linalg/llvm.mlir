// RUN: mlir-opt %s -convert-linalg-to-llvm | FileCheck %s

func @range(%arg0: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %R = linalg.range %c0:%arg0:%c1 : !linalg.range
  return
}
// CHECK-LABEL: func @range(%{{.*}}: i64) {
//       CHECK:   llvm.mlir.constant(0 : index) : i64
//  CHECK-NEXT:   llvm.mlir.constant(1 : index) : i64
//  CHECK-NEXT:   llvm.mlir.undef : !llvm.struct<(i64, i64, i64)>
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(i64, i64, i64)>
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(i64, i64, i64)>
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(i64, i64, i64)>

func @reshape_static_expand(%arg0: memref<3x4x5xf32>) -> memref<1x3x4x1x5xf32> {
  // Reshapes that expand a contiguous tensor with some 1's.
  %0 = linalg.reshape %arg0 [affine_map<(i, j, k, l, m) -> (i, j)>,
                             affine_map<(i, j, k, l, m) -> (k)>,
                             affine_map<(i, j, k, l, m) -> (l, m)>] :
    memref<3x4x5xf32> into memref<1x3x4x1x5xf32>
  return %0 : memref<1x3x4x1x5xf32>
}
// CHECK-LABEL: func @reshape_static_expand
//       CHECK:    llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(1 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(3 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(4 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(1 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(5 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 4] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(60 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(20 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(5 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(5 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(1 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 4] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>

func @reshape_static_collapse(%arg0: memref<1x3x4x1x5xf32>) -> memref<3x4x5xf32> {
  %0 = linalg.reshape %arg0 [affine_map<(i, j, k, l, m) -> (i, j)>,
                             affine_map<(i, j, k, l, m) -> (k)>,
                             affine_map<(i, j, k, l, m) -> (l, m)>] :
    memref<1x3x4x1x5xf32> into memref<3x4x5xf32>
  return %0 : memref<3x4x5xf32>
}
// CHECK-LABEL: func @reshape_static_collapse
//       CHECK:    llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.mlir.constant(3 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.mlir.constant(4 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.mlir.constant(5 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.mlir.constant(20 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.mlir.constant(5 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.mlir.constant(1 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>

func @reshape_fold_zero_dim(%arg0 : memref<1x1xf32>) -> memref<f32> {
  %0 = linalg.reshape %arg0 [] : memref<1x1xf32> into memref<f32>
  return %0 : memref<f32>
}
// CHECK-LABEL: func @reshape_fold_zero_dim
//       CHECK:   llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
//       CHECK:   llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
//       CHECK:   llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
//       CHECK:   llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>

func @reshape_expand_zero_dim(%arg0 : memref<f32>) -> memref<1x1xf32> {
  %0 = linalg.reshape %arg0 [] : memref<f32> into memref<1x1xf32>
  return %0 : memref<1x1xf32>
}
// CHECK-LABEL: func @reshape_expand_zero_dim
//       CHECK:   llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.mlir.constant(1 : index) : i64
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.mlir.constant(1 : index) : i64
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.mlir.constant(1 : index) : i64
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.mlir.constant(1 : index) : i64
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
