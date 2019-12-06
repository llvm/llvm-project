// RUN: mlir-opt %s -convert-vector-to-llvm | FileCheck %s

func @broadcast_vec1d_from_scalar(%arg0: f32) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2xf32>
  return %0 : vector<2xf32>
}
//    CHECK-LABEL: broadcast_vec1d_from_scalar
//          CHECK:   llvm.mlir.undef : !llvm<"<2 x float>">
//          CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//          CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<2 x float>">
//          CHECK:   llvm.shufflevector {{.*}}, {{.*}}[0 : i32, 0 : i32] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//          CHECK:   llvm.return {{.*}} : !llvm<"<2 x float>">

func @broadcast_vec2d_from_scalar(%arg0: f32) -> vector<2x3xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3xf32>
  return %0 : vector<2x3xf32>
}
//    CHECK-LABEL: broadcast_vec2d_from_scalar
//          CHECK:   llvm.mlir.undef : !llvm<"<3 x float>">
//          CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//          CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<3 x float>">
//          CHECK:   llvm.shufflevector {{.*}}, {{.*}}[0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
//          CHECK:   llvm.mlir.undef : !llvm<"[2 x <3 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[2 x <3 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[2 x <3 x float>]">
//          CHECK:   llvm.return {{.*}} : !llvm<"[2 x <3 x float>]">

func @broadcast_vec3d_from_scalar(%arg0: f32) -> vector<2x3x4xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3x4xf32>
  return %0 : vector<2x3x4xf32>
}
//    CHECK-LABEL: broadcast_vec3d_from_scalar
//          CHECK:   llvm.mlir.undef : !llvm<"<4 x float>">
//          CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//          CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<4 x float>">
//          CHECK:   llvm.shufflevector {{.*}}, {{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
//          CHECK:   llvm.mlir.undef : !llvm<"[3 x <4 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <4 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <4 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <4 x float>]">
//          CHECK:   llvm.mlir.undef : !llvm<"[2 x [3 x <4 x float>]]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[2 x [3 x <4 x float>]]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[2 x [3 x <4 x float>]]">
//          CHECK:   llvm.return {{.*}} : !llvm<"[2 x [3 x <4 x float>]]">

func @broadcast_vec1d_from_vec1d(%arg0: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<2xf32>
  return %0 : vector<2xf32>
}
//    CHECK-LABEL: broadcast_vec1d_from_vec1d
//          CHECK:   llvm.return {{.*}} : !llvm<"<2 x float>">

func @broadcast_vec2d_from_vec1d(%arg0: vector<2xf32>) -> vector<3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}
//    CHECK-LABEL: broadcast_vec2d_from_vec1d
//          CHECK:   llvm.mlir.undef : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.return {{.*}} : !llvm<"[3 x <2 x float>]">

func @broadcast_vec3d_from_vec1d(%arg0: vector<2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}
//    CHECK-LABEL: broadcast_vec3d_from_vec1d
//          CHECK:   llvm.mlir.undef : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.mlir.undef : !llvm<"[4 x [3 x <2 x float>]]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[4 x [3 x <2 x float>]]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[4 x [3 x <2 x float>]]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[4 x [3 x <2 x float>]]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3] : !llvm<"[4 x [3 x <2 x float>]]">
//          CHECK:   llvm.return {{.*}} : !llvm<"[4 x [3 x <2 x float>]]">

func @broadcast_vec3d_from_vec2d(%arg0: vector<3x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<3x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}
//    CHECK-LABEL: broadcast_vec3d_from_vec2d
//          CHECK:   llvm.mlir.undef : !llvm<"[4 x [3 x <2 x float>]]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[4 x [3 x <2 x float>]]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[4 x [3 x <2 x float>]]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[4 x [3 x <2 x float>]]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3] : !llvm<"[4 x [3 x <2 x float>]]">
//          CHECK:   llvm.return {{.*}} : !llvm<"[4 x [3 x <2 x float>]]">

func @broadcast_stretch(%arg0: vector<1xf32>) -> vector<4xf32> {
  %0 = vector.broadcast %arg0 : vector<1xf32> to vector<4xf32>
  return %0 : vector<4xf32>
}
//    CHECK-LABEL: broadcast_stretch
//          CHECK:   llvm.mlir.undef : !llvm<"<4 x float>">
//          CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//          CHECK:   llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<1 x float>">
//          CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//          CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<4 x float>">
//          CHECK:   llvm.shufflevector {{.*}}, {{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
//          CHECK:   llvm.return {{.*}} : !llvm<"<4 x float>">

func @broadcast_stretch_at_start(%arg0: vector<1x4xf32>) -> vector<3x4xf32> {
  %0 = vector.broadcast %arg0 : vector<1x4xf32> to vector<3x4xf32>
  return %0 : vector<3x4xf32>
}
//    CHECK-LABEL: broadcast_stretch_at_start
//          CHECK:   llvm.mlir.undef : !llvm<"[3 x <4 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <4 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <4 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <4 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <4 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <4 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <4 x float>]">
//          CHECK:   llvm.return {{.*}} : !llvm<"[3 x <4 x float>]">

func @broadcast_stretch_at_end(%arg0: vector<4x1xf32>) -> vector<4x3xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1xf32> to vector<4x3xf32>
  return %0 : vector<4x3xf32>
}
//    CHECK-LABEL: broadcast_stretch_at_end
//          CHECK:   llvm.mlir.undef : !llvm<"[4 x <3 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[4 x <1 x float>]">
//          CHECK:   llvm.mlir.undef : !llvm<"<3 x float>">
//          CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//          CHECK:   llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<1 x float>">
//          CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//          CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<3 x float>">
//          CHECK:   llvm.shufflevector {{.*}}, {{.*}} [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[4 x <3 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[1] : !llvm<"[4 x <1 x float>]">
//          CHECK:   llvm.mlir.undef : !llvm<"<3 x float>">
//          CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//          CHECK:   llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<1 x float>">
//          CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//          CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<3 x float>">
//          CHECK:   llvm.shufflevector {{.*}}, {{.*}} [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[4 x <3 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[2] : !llvm<"[4 x <1 x float>]">
//          CHECK:   llvm.mlir.undef : !llvm<"<3 x float>">
//          CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//          CHECK:   llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<1 x float>">
//          CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//          CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<3 x float>">
//          CHECK:   llvm.shufflevector {{.*}}, {{.*}} [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[4 x <3 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[3] : !llvm<"[4 x <1 x float>]">
//          CHECK:   llvm.mlir.undef : !llvm<"<3 x float>">
//          CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//          CHECK:   llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<1 x float>">
//          CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//          CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<3 x float>">
//          CHECK:   llvm.shufflevector {{.*}}, {{.*}} [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3] : !llvm<"[4 x <3 x float>]">
//          CHECK:   llvm.return {{.*}} : !llvm<"[4 x <3 x float>]">

func @broadcast_stretch_in_middle(%arg0: vector<4x1x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}
//    CHECK-LABEL: broadcast_stretch_in_middle
//          CHECK:   llvm.mlir.undef : !llvm<"[4 x [3 x <2 x float>]]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[4 x [1 x <2 x float>]]">
//          CHECK:   llvm.mlir.undef : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[4 x [3 x <2 x float>]]">
//          CHECK:   llvm.extractvalue {{.*}}[1] : !llvm<"[4 x [1 x <2 x float>]]">
//          CHECK:   llvm.mlir.undef : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[4 x [3 x <2 x float>]]">
//          CHECK:   llvm.extractvalue {{.*}}[2] : !llvm<"[4 x [1 x <2 x float>]]">
//          CHECK:   llvm.mlir.undef : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[4 x [3 x <2 x float>]]">
//          CHECK:   llvm.extractvalue {{.*}}[3] : !llvm<"[4 x [1 x <2 x float>]]">
//          CHECK:   llvm.mlir.undef : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <2 x float>]">
//          CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3] : !llvm<"[4 x [3 x <2 x float>]]">
//          CHECK:   llvm.return {{.*}} : !llvm<"[4 x [3 x <2 x float>]]">

func @outerproduct(%arg0: vector<2xf32>, %arg1: vector<3xf32>) -> vector<2x3xf32> {
  %2 = vector.outerproduct %arg0, %arg1 : vector<2xf32>, vector<3xf32>
  return %2 : vector<2x3xf32>
}
//    CHECK-LABEL: outerproduct
//          CHECK:   llvm.mlir.undef : !llvm<"[2 x <3 x float>]">
//          CHECK:   llvm.shufflevector {{.*}} [0 : i32, 0 : i32, 0 : i32] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//          CHECK:   llvm.fmul {{.*}}, {{.*}} : !llvm<"<3 x float>">
//          CHECK:   llvm.insertvalue {{.*}}[0] : !llvm<"[2 x <3 x float>]">
//          CHECK:   llvm.shufflevector {{.*}} [1 : i32, 1 : i32, 1 : i32] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//          CHECK:   llvm.fmul {{.*}}, {{.*}} : !llvm<"<3 x float>">
//          CHECK:   llvm.insertvalue {{.*}}[1] : !llvm<"[2 x <3 x float>]">
//          CHECK:   llvm.return {{.*}} : !llvm<"[2 x <3 x float>]">

func @outerproduct_add(%arg0: vector<2xf32>, %arg1: vector<3xf32>, %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %2 = vector.outerproduct %arg0, %arg1, %arg2 : vector<2xf32>, vector<3xf32>
  return %2 : vector<2x3xf32>
}
//    CHECK-LABEL: outerproduct_add
//          CHECK:   llvm.mlir.undef : !llvm<"[2 x <3 x float>]">
//          CHECK:   llvm.shufflevector {{.*}} [0 : i32, 0 : i32, 0 : i32] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[2 x <3 x float>]">
//          CHECK:   "llvm.intr.fmuladd"({{.*}}) : (!llvm<"<3 x float>">, !llvm<"<3 x float>">, !llvm<"<3 x float>">) -> !llvm<"<3 x float>">
//          CHECK:   llvm.insertvalue {{.*}}[0] : !llvm<"[2 x <3 x float>]">
//          CHECK:   llvm.shufflevector {{.*}} [1 : i32, 1 : i32, 1 : i32] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//          CHECK:   llvm.extractvalue {{.*}}[1] : !llvm<"[2 x <3 x float>]">
//          CHECK:   "llvm.intr.fmuladd"({{.*}}) : (!llvm<"<3 x float>">, !llvm<"<3 x float>">, !llvm<"<3 x float>">) -> !llvm<"<3 x float>">
//          CHECK:   llvm.insertvalue {{.*}}[1] : !llvm<"[2 x <3 x float>]">
//          CHECK:   llvm.return {{.*}} : !llvm<"[2 x <3 x float>]">

func @extract_vec_2d_from_vec_3d(%arg0: vector<4x3x16xf32>) -> vector<3x16xf32> {
  %0 = vector.extract %arg0[0 : i32]: vector<4x3x16xf32>
  return %0 : vector<3x16xf32>
}
// CHECK-LABEL: extract_vec_2d_from_vec_3d
//       CHECK:   llvm.extractvalue %{{.*}}[0 : i32] : !llvm<"[4 x [3 x <16 x float>]]">
//       CHECK:   llvm.return %{{.*}} : !llvm<"[3 x <16 x float>]">

func @extract_element_from_vec_3d(%arg0: vector<4x3x16xf32>) -> f32 {
  %0 = vector.extract %arg0[0 : i32, 0 : i32, 0 : i32]: vector<4x3x16xf32>
  return %0 : f32
}
// CHECK-LABEL: extract_element_from_vec_3d
//       CHECK:   llvm.extractvalue %{{.*}}[0 : i32, 0 : i32] : !llvm<"[4 x [3 x <16 x float>]]">
//       CHECK:   llvm.mlir.constant(0 : i32) : !llvm.i32
//       CHECK:   llvm.extractelement %{{.*}}[%{{.*}} : !llvm.i32] : !llvm<"<16 x float>">
//       CHECK:   llvm.return %{{.*}} : !llvm.float

func @vector_type_cast(%arg0: memref<8x8x8xf32>) -> memref<vector<8x8x8xf32>> {
  %0 = vector.type_cast %arg0: memref<8x8x8xf32> to memref<vector<8x8x8xf32>>
  return %0 : memref<vector<8x8x8xf32>>
}
// CHECK-LABEL: vector_type_cast
//       CHECK:   llvm.mlir.undef : !llvm<"{ [8 x [8 x <8 x float>]]*, [8 x [8 x <8 x float>]]*, i64 }">
//       CHECK:   %[[allocated:.*]] = llvm.extractvalue {{.*}}[0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   %[[allocatedBit:.*]] = llvm.bitcast %[[allocated]] : !llvm<"float*"> to !llvm<"[8 x [8 x <8 x float>]]*">
//       CHECK:   llvm.insertvalue %[[allocatedBit]], {{.*}}[0] : !llvm<"{ [8 x [8 x <8 x float>]]*, [8 x [8 x <8 x float>]]*, i64 }">
//       CHECK:   %[[aligned:.*]] = llvm.extractvalue {{.*}}[1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   %[[alignedBit:.*]] = llvm.bitcast %[[aligned]] : !llvm<"float*"> to !llvm<"[8 x [8 x <8 x float>]]*">
//       CHECK:   llvm.insertvalue %[[alignedBit]], {{.*}}[1] : !llvm<"{ [8 x [8 x <8 x float>]]*, [8 x [8 x <8 x float>]]*, i64 }">
//       CHECK:   llvm.mlir.constant(0 : index
//       CHECK:   llvm.insertvalue {{.*}}[2] : !llvm<"{ [8 x [8 x <8 x float>]]*, [8 x [8 x <8 x float>]]*, i64 }">
