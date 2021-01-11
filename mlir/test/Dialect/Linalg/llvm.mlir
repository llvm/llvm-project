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

func @slice(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: !linalg.range) {
  %1 = linalg.slice %arg0[%arg1] : memref<?xf32, offset: ?, strides: [1]>, !linalg.range, memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @slice
//   insert data ptr for slice op
//       CHECK:   llvm.extractvalue %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i64, i64, i64)>
//  CHECK-NEXT:   llvm.mul %{{.*}}, %{{.*}} : i64
//  CHECK-NEXT:   llvm.add %{{.*}}, %{{.*}} : i64
//    insert offset
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//  CHECK-NEXT:   llvm.mlir.constant(0 : index)
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i64, i64, i64)>
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[1] : !llvm.struct<(i64, i64, i64)>
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[2] : !llvm.struct<(i64, i64, i64)>
//    get size[0] from parent view
//  CHECK-NEXT:   llvm.extractvalue %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//  CHECK-NEXT:   llvm.icmp "slt" %{{.*}}, %{{.*}} : i64
//  CHECK-NEXT:   llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, i64
//    compute size[0] bounded by parent view's size[0]
//  CHECK-NEXT:   llvm.sub %{{.*}}, %{{.*}} : i64
//    bound below by 0
//  CHECK-NEXT:   llvm.icmp "slt" %{{.*}}, %{{.*}} : i64
//  CHECK-NEXT:   llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, i64
//    compute stride[0] using bounded size
//  CHECK-NEXT:   llvm.mul %{{.*}}, %{{.*}} : i64
//    insert size and stride
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//  CHECK-NEXT:   llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>

func @slice_with_range_and_index(%arg0: memref<?x?xf64, offset: ?, strides: [?, 1]>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %R = linalg.range %c0:%c1:%c1 : !linalg.range
  scf.for %i0 = %c0 to %c1 step %c1 {
    %1 = linalg.slice %arg0[%i0, %R] : memref<?x?xf64, offset: ?, strides: [?, 1]>, index, !linalg.range, memref<?xf64, offset: ?, strides: [1]>
  }
  return
}
// CHECK-LABEL: func @slice_with_range_and_index
// loop-body.
//       CHECK:   llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   llvm.extractvalue %{{.*}}[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.extractvalue %{{.*}}[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   llvm.insertvalue %{{.*}}[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i64, i64, i64)>
//       CHECK:   llvm.extractvalue %{{.*}}[1] : !llvm.struct<(i64, i64, i64)>
//       CHECK:   llvm.insertvalue %{{.*}}[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
//       CHECK:   llvm.insertvalue %{{.*}}[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>

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
