// RUN: mlir-opt %s -convert-vector-to-llvm | FileCheck %s

func @broadcast_vec1d_from_scalar(%arg0: f32) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2xf32>
  return %0 : vector<2xf32>
}
// CHECK-LABEL: broadcast_vec1d_from_scalar
//       CHECK:   llvm.mlir.undef : !llvm<"<2 x float>">
//       CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<2 x float>">
//       CHECK:   llvm.shufflevector {{.*}}, {{.*}}[0 : i32, 0 : i32] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//       CHECK:   llvm.return {{.*}} : !llvm<"<2 x float>">

func @broadcast_vec2d_from_scalar(%arg0: f32) -> vector<2x3xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3xf32>
  return %0 : vector<2x3xf32>
}
// CHECK-LABEL: broadcast_vec2d_from_scalar
//       CHECK:   llvm.mlir.undef : !llvm<"<3 x float>">
//       CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<3 x float>">
//       CHECK:   llvm.shufflevector {{.*}}, {{.*}}[0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
//       CHECK:   llvm.mlir.undef : !llvm<"[2 x <3 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[2 x <3 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[2 x <3 x float>]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[2 x <3 x float>]">

func @broadcast_vec3d_from_scalar(%arg0: f32) -> vector<2x3x4xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3x4xf32>
  return %0 : vector<2x3x4xf32>
}
// CHECK-LABEL: broadcast_vec3d_from_scalar
//       CHECK:   llvm.mlir.undef : !llvm<"<4 x float>">
//       CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<4 x float>">
//       CHECK:   llvm.shufflevector {{.*}}, {{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
//       CHECK:   llvm.mlir.undef : !llvm<"[3 x <4 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <4 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <4 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <4 x float>]">
//       CHECK:   llvm.mlir.undef : !llvm<"[2 x [3 x <4 x float>]]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[2 x [3 x <4 x float>]]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[2 x [3 x <4 x float>]]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[2 x [3 x <4 x float>]]">

func @broadcast_vec1d_from_vec1d(%arg0: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<2xf32>
  return %0 : vector<2xf32>
}
// CHECK-LABEL: broadcast_vec1d_from_vec1d
//       CHECK:   llvm.return {{.*}} : !llvm<"<2 x float>">

func @broadcast_vec2d_from_vec1d(%arg0: vector<2xf32>) -> vector<3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}
// CHECK-LABEL: broadcast_vec2d_from_vec1d
//       CHECK:   llvm.mlir.undef : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[3 x <2 x float>]">

func @broadcast_vec3d_from_vec1d(%arg0: vector<2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}
// CHECK-LABEL: broadcast_vec3d_from_vec1d
//       CHECK:   llvm.mlir.undef : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.mlir.undef : !llvm<"[4 x [3 x <2 x float>]]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[4 x [3 x <2 x float>]]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[4 x [3 x <2 x float>]]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[4 x [3 x <2 x float>]]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3] : !llvm<"[4 x [3 x <2 x float>]]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[4 x [3 x <2 x float>]]">

func @broadcast_vec3d_from_vec2d(%arg0: vector<3x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<3x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}
// CHECK-LABEL: broadcast_vec3d_from_vec2d
//       CHECK:   llvm.mlir.undef : !llvm<"[4 x [3 x <2 x float>]]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[4 x [3 x <2 x float>]]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[4 x [3 x <2 x float>]]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[4 x [3 x <2 x float>]]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3] : !llvm<"[4 x [3 x <2 x float>]]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[4 x [3 x <2 x float>]]">

func @broadcast_stretch(%arg0: vector<1xf32>) -> vector<4xf32> {
  %0 = vector.broadcast %arg0 : vector<1xf32> to vector<4xf32>
  return %0 : vector<4xf32>
}
// CHECK-LABEL: broadcast_stretch
//       CHECK:   llvm.mlir.undef : !llvm<"<4 x float>">
//       CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<1 x float>">
//       CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<4 x float>">
//       CHECK:   llvm.shufflevector {{.*}}, {{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
//       CHECK:   llvm.return {{.*}} : !llvm<"<4 x float>">

func @broadcast_stretch_at_start(%arg0: vector<1x4xf32>) -> vector<3x4xf32> {
  %0 = vector.broadcast %arg0 : vector<1x4xf32> to vector<3x4xf32>
  return %0 : vector<3x4xf32>
}
// CHECK-LABEL: broadcast_stretch_at_start
//       CHECK:   llvm.mlir.undef : !llvm<"[3 x <4 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <4 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <4 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <4 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <4 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <4 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <4 x float>]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[3 x <4 x float>]">

func @broadcast_stretch_at_end(%arg0: vector<4x1xf32>) -> vector<4x3xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1xf32> to vector<4x3xf32>
  return %0 : vector<4x3xf32>
}
// CHECK-LABEL: broadcast_stretch_at_end
//       CHECK:   llvm.mlir.undef : !llvm<"[4 x <3 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[4 x <1 x float>]">
//       CHECK:   llvm.mlir.undef : !llvm<"<3 x float>">
//       CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<1 x float>">
//       CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<3 x float>">
//       CHECK:   llvm.shufflevector {{.*}}, {{.*}} [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[4 x <3 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[1] : !llvm<"[4 x <1 x float>]">
//       CHECK:   llvm.mlir.undef : !llvm<"<3 x float>">
//       CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<1 x float>">
//       CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<3 x float>">
//       CHECK:   llvm.shufflevector {{.*}}, {{.*}} [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[4 x <3 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[2] : !llvm<"[4 x <1 x float>]">
//       CHECK:   llvm.mlir.undef : !llvm<"<3 x float>">
//       CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<1 x float>">
//       CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<3 x float>">
//       CHECK:   llvm.shufflevector {{.*}}, {{.*}} [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[4 x <3 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[3] : !llvm<"[4 x <1 x float>]">
//       CHECK:   llvm.mlir.undef : !llvm<"<3 x float>">
//       CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<1 x float>">
//       CHECK:   llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<3 x float>">
//       CHECK:   llvm.shufflevector {{.*}}, {{.*}} [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3] : !llvm<"[4 x <3 x float>]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[4 x <3 x float>]">

func @broadcast_stretch_in_middle(%arg0: vector<4x1x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}
// CHECK-LABEL: broadcast_stretch_in_middle
//       CHECK:   llvm.mlir.undef : !llvm<"[4 x [3 x <2 x float>]]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[4 x [1 x <2 x float>]]">
//       CHECK:   llvm.mlir.undef : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[4 x [3 x <2 x float>]]">
//       CHECK:   llvm.extractvalue {{.*}}[1] : !llvm<"[4 x [1 x <2 x float>]]">
//       CHECK:   llvm.mlir.undef : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[4 x [3 x <2 x float>]]">
//       CHECK:   llvm.extractvalue {{.*}}[2] : !llvm<"[4 x [1 x <2 x float>]]">
//       CHECK:   llvm.mlir.undef : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[4 x [3 x <2 x float>]]">
//       CHECK:   llvm.extractvalue {{.*}}[3] : !llvm<"[4 x [1 x <2 x float>]]">
//       CHECK:   llvm.mlir.undef : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[1 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[3 x <2 x float>]">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3] : !llvm<"[4 x [3 x <2 x float>]]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[4 x [3 x <2 x float>]]">

func @outerproduct(%arg0: vector<2xf32>, %arg1: vector<3xf32>) -> vector<2x3xf32> {
  %2 = vector.outerproduct %arg0, %arg1 : vector<2xf32>, vector<3xf32>
  return %2 : vector<2x3xf32>
}
// CHECK-LABEL: outerproduct
//       CHECK:   llvm.mlir.undef : !llvm<"[2 x <3 x float>]">
//       CHECK:   llvm.shufflevector {{.*}} [0 : i32, 0 : i32, 0 : i32] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//       CHECK:   llvm.fmul {{.*}}, {{.*}} : !llvm<"<3 x float>">
//       CHECK:   llvm.insertvalue {{.*}}[0] : !llvm<"[2 x <3 x float>]">
//       CHECK:   llvm.shufflevector {{.*}} [1 : i32, 1 : i32, 1 : i32] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//       CHECK:   llvm.fmul {{.*}}, {{.*}} : !llvm<"<3 x float>">
//       CHECK:   llvm.insertvalue {{.*}}[1] : !llvm<"[2 x <3 x float>]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[2 x <3 x float>]">

func @outerproduct_add(%arg0: vector<2xf32>, %arg1: vector<3xf32>, %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %2 = vector.outerproduct %arg0, %arg1, %arg2 : vector<2xf32>, vector<3xf32>
  return %2 : vector<2x3xf32>
}
// CHECK-LABEL: outerproduct_add
//       CHECK:   llvm.mlir.undef : !llvm<"[2 x <3 x float>]">
//       CHECK:   llvm.shufflevector {{.*}} [0 : i32, 0 : i32, 0 : i32] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[2 x <3 x float>]">
//       CHECK:   "llvm.intr.fmuladd"({{.*}}) : (!llvm<"<3 x float>">, !llvm<"<3 x float>">, !llvm<"<3 x float>">) -> !llvm<"<3 x float>">
//       CHECK:   llvm.insertvalue {{.*}}[0] : !llvm<"[2 x <3 x float>]">
//       CHECK:   llvm.shufflevector {{.*}} [1 : i32, 1 : i32, 1 : i32] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//       CHECK:   llvm.extractvalue {{.*}}[1] : !llvm<"[2 x <3 x float>]">
//       CHECK:   "llvm.intr.fmuladd"({{.*}}) : (!llvm<"<3 x float>">, !llvm<"<3 x float>">, !llvm<"<3 x float>">) -> !llvm<"<3 x float>">
//       CHECK:   llvm.insertvalue {{.*}}[1] : !llvm<"[2 x <3 x float>]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[2 x <3 x float>]">

func @shuffle_1D_direct(%arg0: vector<2xf32>, %arg1: vector<2xf32>) -> vector<2xf32> {
  %1 = vector.shuffle %arg0, %arg1 [0, 1] : vector<2xf32>, vector<2xf32>
  return %1 : vector<2xf32>
}
// CHECK-LABEL: shuffle_1D_direct(%arg0: !llvm<"<2 x float>">, %arg1: !llvm<"<2 x float>">)
//       CHECK:   %[[s:.*]] = llvm.shufflevector %arg0, %arg1 [0, 1] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//       CHECK:   llvm.return %[[s]] : !llvm<"<2 x float>">

func @shuffle_1D(%arg0: vector<2xf32>, %arg1: vector<3xf32>) -> vector<5xf32> {
  %1 = vector.shuffle %arg0, %arg1 [4, 3, 2, 1, 0] : vector<2xf32>, vector<3xf32>
  return %1 : vector<5xf32>
}
// CHECK-LABEL: shuffle_1D(%arg0: !llvm<"<2 x float>">, %arg1: !llvm<"<3 x float>">)
//       CHECK:   %[[u0:.*]] = llvm.mlir.undef : !llvm<"<5 x float>">
//       CHECK:   %[[c2:.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
//       CHECK:   %[[e1:.*]] = llvm.extractelement %arg1[%[[c2]] : !llvm.i64] : !llvm<"<3 x float>">
//       CHECK:   %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   %[[i1:.*]] = llvm.insertelement %[[e1]], %[[u0]][%[[c0]] : !llvm.i64] : !llvm<"<5 x float>">
//       CHECK:   %[[c1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:   %[[e2:.*]] = llvm.extractelement %arg1[%[[c1]] : !llvm.i64] : !llvm<"<3 x float>">
//       CHECK:   %[[c1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:   %[[i2:.*]] = llvm.insertelement %[[e2]], %[[i1]][%[[c1]] : !llvm.i64] : !llvm<"<5 x float>">
//       CHECK:   %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   %[[e3:.*]] = llvm.extractelement %arg1[%[[c0]] : !llvm.i64] : !llvm<"<3 x float>">
//       CHECK:   %[[c2:.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
//       CHECK:   %[[i3:.*]] = llvm.insertelement %[[e3]], %[[i2]][%[[c2]] : !llvm.i64] : !llvm<"<5 x float>">
//       CHECK:   %[[c1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:   %[[e4:.*]] = llvm.extractelement %arg0[%[[c1]] : !llvm.i64] : !llvm<"<2 x float>">
//       CHECK:   %[[c3:.*]] = llvm.mlir.constant(3 : index) : !llvm.i64
//       CHECK:   %[[i4:.*]] = llvm.insertelement %[[e4]], %[[i3]][%[[c3]] : !llvm.i64] : !llvm<"<5 x float>">
//       CHECK:   %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   %[[e5:.*]] = llvm.extractelement %arg0[%[[c0]] : !llvm.i64] : !llvm<"<2 x float>">
//       CHECK:   %[[c4:.*]] = llvm.mlir.constant(4 : index) : !llvm.i64
//       CHECK:   %[[i5:.*]] = llvm.insertelement %[[e5]], %[[i4]][%[[c4]] : !llvm.i64] : !llvm<"<5 x float>">
//       CHECK:   llvm.return %[[i5]] : !llvm<"<5 x float>">

func @shuffle_2D(%a: vector<1x4xf32>, %b: vector<2x4xf32>) -> vector<3x4xf32> {
  %1 = vector.shuffle %a, %b[1, 0, 2] : vector<1x4xf32>, vector<2x4xf32>
  return %1 : vector<3x4xf32>
}
// CHECK-LABEL: shuffle_2D(%arg0: !llvm<"[1 x <4 x float>]">, %arg1: !llvm<"[2 x <4 x float>]">)
//       CHECK:   %[[u0:.*]] = llvm.mlir.undef : !llvm<"[3 x <4 x float>]">
//       CHECK:   %[[e1:.*]] = llvm.extractvalue %arg1[0] : !llvm<"[2 x <4 x float>]">
//       CHECK:   %[[i1:.*]] = llvm.insertvalue %[[e1]], %[[u0]][0] : !llvm<"[3 x <4 x float>]">
//       CHECK:   %[[e2:.*]] = llvm.extractvalue %arg0[0] : !llvm<"[1 x <4 x float>]">
//       CHECK:   %[[i2:.*]] = llvm.insertvalue %[[e2]], %[[i1]][1] : !llvm<"[3 x <4 x float>]">
//       CHECK:   %[[e3:.*]] = llvm.extractvalue %arg1[1] : !llvm<"[2 x <4 x float>]">
//       CHECK:   %[[i3:.*]] = llvm.insertvalue %[[e3]], %[[i2]][2] : !llvm<"[3 x <4 x float>]">
//       CHECK:   llvm.return %[[i3]] : !llvm<"[3 x <4 x float>]">

func @extract_element(%arg0: vector<16xf32>) -> f32 {
  %0 = constant 15 : i32
  %1 = vector.extractelement %arg0[%0 : i32]: vector<16xf32>
  return %1 : f32
}
// CHECK-LABEL: extract_element(%arg0: !llvm<"<16 x float>">)
//       CHECK:   %[[c:.*]] = llvm.mlir.constant(15 : i32) : !llvm.i32
//       CHECK:   %[[x:.*]] = llvm.extractelement %arg0[%[[c]] : !llvm.i32] : !llvm<"<16 x float>">
//       CHECK:   llvm.return %[[x]] : !llvm.float

func @extract_element_from_vec_1d(%arg0: vector<16xf32>) -> f32 {
  %0 = vector.extract %arg0[15]: vector<16xf32>
  return %0 : f32
}
// CHECK-LABEL: extract_element_from_vec_1d
//       CHECK:   llvm.mlir.constant(15 : i64) : !llvm.i64
//       CHECK:   llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<16 x float>">
//       CHECK:   llvm.return {{.*}} : !llvm.float

func @extract_vec_2d_from_vec_3d(%arg0: vector<4x3x16xf32>) -> vector<3x16xf32> {
  %0 = vector.extract %arg0[0]: vector<4x3x16xf32>
  return %0 : vector<3x16xf32>
}
// CHECK-LABEL: extract_vec_2d_from_vec_3d
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[4 x [3 x <16 x float>]]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[3 x <16 x float>]">

func @extract_vec_1d_from_vec_3d(%arg0: vector<4x3x16xf32>) -> vector<16xf32> {
  %0 = vector.extract %arg0[0, 0]: vector<4x3x16xf32>
  return %0 : vector<16xf32>
}
// CHECK-LABEL: extract_vec_1d_from_vec_3d
//       CHECK:   llvm.extractvalue {{.*}}[0, 0] : !llvm<"[4 x [3 x <16 x float>]]">
//       CHECK:   llvm.return {{.*}} : !llvm<"<16 x float>">

func @extract_element_from_vec_3d(%arg0: vector<4x3x16xf32>) -> f32 {
  %0 = vector.extract %arg0[0, 0, 0]: vector<4x3x16xf32>
  return %0 : f32
}
// CHECK-LABEL: extract_element_from_vec_3d
//       CHECK:   llvm.extractvalue {{.*}}[0, 0] : !llvm<"[4 x [3 x <16 x float>]]">
//       CHECK:   llvm.mlir.constant(0 : i64) : !llvm.i64
//       CHECK:   llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<16 x float>">
//       CHECK:   llvm.return {{.*}} : !llvm.float

func @insert_element(%arg0: f32, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = constant 3 : i32
  %1 = vector.insertelement %arg0, %arg1[%0 : i32] : vector<4xf32>
  return %1 : vector<4xf32>
}
// CHECK-LABEL: insert_element(%arg0: !llvm.float, %arg1: !llvm<"<4 x float>">)
//       CHECK:   %[[c:.*]] = llvm.mlir.constant(3 : i32) : !llvm.i32
//       CHECK:   %[[x:.*]] = llvm.insertelement %arg0, %arg1[%[[c]] : !llvm.i32] : !llvm<"<4 x float>">
//       CHECK:   llvm.return %[[x]] : !llvm<"<4 x float>">

func @insert_element_into_vec_1d(%arg0: f32, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.insert %arg0, %arg1[3] : f32 into vector<4xf32>
  return %0 : vector<4xf32>
}
// CHECK-LABEL: insert_element_into_vec_1d
//       CHECK:   llvm.mlir.constant(3 : i64) : !llvm.i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<4 x float>">
//       CHECK:   llvm.return {{.*}} : !llvm<"<4 x float>">

func @insert_vec_2d_into_vec_3d(%arg0: vector<8x16xf32>, %arg1: vector<4x8x16xf32>) -> vector<4x8x16xf32> {
  %0 = vector.insert %arg0, %arg1[3] : vector<8x16xf32> into vector<4x8x16xf32>
  return %0 : vector<4x8x16xf32>
}
// CHECK-LABEL: insert_vec_2d_into_vec_3d
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3] : !llvm<"[4 x [8 x <16 x float>]]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[4 x [8 x <16 x float>]]">

func @insert_vec_1d_into_vec_3d(%arg0: vector<16xf32>, %arg1: vector<4x8x16xf32>) -> vector<4x8x16xf32> {
  %0 = vector.insert %arg0, %arg1[3, 7] : vector<16xf32> into vector<4x8x16xf32>
  return %0 : vector<4x8x16xf32>
}
// CHECK-LABEL: insert_vec_1d_into_vec_3d
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3, 7] : !llvm<"[4 x [8 x <16 x float>]]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[4 x [8 x <16 x float>]]">

func @insert_element_into_vec_3d(%arg0: f32, %arg1: vector<4x8x16xf32>) -> vector<4x8x16xf32> {
  %0 = vector.insert %arg0, %arg1[3, 7, 15] : f32 into vector<4x8x16xf32>
  return %0 : vector<4x8x16xf32>
}
// CHECK-LABEL: insert_element_into_vec_3d
//       CHECK:   llvm.extractvalue {{.*}}[3, 7] : !llvm<"[4 x [8 x <16 x float>]]">
//       CHECK:   llvm.mlir.constant(15 : i64) : !llvm.i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<16 x float>">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3, 7] : !llvm<"[4 x [8 x <16 x float>]]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[4 x [8 x <16 x float>]]">

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

func @vector_print_scalar(%arg0: f32) {
  vector.print %arg0 : f32
  return
}
// CHECK-LABEL: vector_print_scalar(%arg0: !llvm.float)
//       CHECK:    llvm.call @print_f32(%arg0) : (!llvm.float) -> ()
//       CHECK:    llvm.call @print_newline() : () -> ()

func @vector_print_vector(%arg0: vector<2x2xf32>) {
  vector.print %arg0 : vector<2x2xf32>
  return
}
// CHECK-LABEL: vector_print_vector(%arg0: !llvm<"[2 x <2 x float>]">)
//       CHECK:    llvm.call @print_open() : () -> ()
//       CHECK:    %[[x0:.*]] = llvm.extractvalue %arg0[0] : !llvm<"[2 x <2 x float>]">
//       CHECK:    llvm.call @print_open() : () -> ()
//       CHECK:    %[[x1:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:    %[[x2:.*]] = llvm.extractelement %[[x0]][%[[x1]] : !llvm.i64] : !llvm<"<2 x float>">
//       CHECK:    llvm.call @print_f32(%[[x2]]) : (!llvm.float) -> ()
//       CHECK:    llvm.call @print_comma() : () -> ()
//       CHECK:    %[[x3:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:    %[[x4:.*]] = llvm.extractelement %[[x0]][%[[x3]] : !llvm.i64] : !llvm<"<2 x float>">
//       CHECK:    llvm.call @print_f32(%[[x4]]) : (!llvm.float) -> ()
//       CHECK:    llvm.call @print_close() : () -> ()
//       CHECK:    llvm.call @print_comma() : () -> ()
//       CHECK:    %[[x5:.*]] = llvm.extractvalue %arg0[1] : !llvm<"[2 x <2 x float>]">
//       CHECK:    llvm.call @print_open() : () -> ()
//       CHECK:    %[[x6:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:    %[[x7:.*]] = llvm.extractelement %[[x5]][%[[x6]] : !llvm.i64] : !llvm<"<2 x float>">
//       CHECK:    llvm.call @print_f32(%[[x7]]) : (!llvm.float) -> ()
//       CHECK:    llvm.call @print_comma() : () -> ()
//       CHECK:    %[[x8:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:    %[[x9:.*]] = llvm.extractelement %[[x5]][%[[x8]] : !llvm.i64] : !llvm<"<2 x float>">
//       CHECK:    llvm.call @print_f32(%[[x9]]) : (!llvm.float) -> ()
//       CHECK:    llvm.call @print_close() : () -> ()
//       CHECK:    llvm.call @print_close() : () -> ()
//       CHECK:    llvm.call @print_newline() : () -> ()
