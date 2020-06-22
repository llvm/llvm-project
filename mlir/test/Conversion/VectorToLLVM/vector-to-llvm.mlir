// RUN: mlir-opt %s -convert-vector-to-llvm | FileCheck %s

func @broadcast_vec1d_from_scalar(%arg0: f32) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2xf32>
  return %0 : vector<2xf32>
}
// CHECK-LABEL: llvm.func @broadcast_vec1d_from_scalar(
// CHECK-SAME:  %[[A:.*]]: !llvm.float)
// CHECK:       %[[T0:.*]] = llvm.mlir.undef : !llvm<"<2 x float>">
// CHECK:       %[[T1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:       %[[T2:.*]] = llvm.insertelement %[[A]], %[[T0]][%[[T1]] : !llvm.i32] : !llvm<"<2 x float>">
// CHECK:       %[[T3:.*]] = llvm.shufflevector %[[T2]], %[[T0]] [0 : i32, 0 : i32] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
// CHECK:       llvm.return %[[T3]] : !llvm<"<2 x float>">

func @broadcast_vec2d_from_scalar(%arg0: f32) -> vector<2x3xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3xf32>
  return %0 : vector<2x3xf32>
}
// CHECK-LABEL: llvm.func @broadcast_vec2d_from_scalar(
// CHECK-SAME:  %[[A:.*]]: !llvm.float)
// CHECK:       %[[T0:.*]] = llvm.mlir.undef : !llvm<"[2 x <3 x float>]">
// CHECK:       %[[T1:.*]] = llvm.mlir.undef : !llvm<"<3 x float>">
// CHECK:       %[[T2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:       %[[T3:.*]] = llvm.insertelement %[[A]], %[[T1]][%[[T2]] : !llvm.i32] : !llvm<"<3 x float>">
// CHECK:       %[[T4:.*]] = llvm.shufflevector %[[T3]], %[[T3]] [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
// CHECK:       %[[T5:.*]] = llvm.insertvalue %[[T4]], %[[T0]][0] : !llvm<"[2 x <3 x float>]">
// CHECK:       %[[T6:.*]] = llvm.insertvalue %[[T4]], %[[T5]][1] : !llvm<"[2 x <3 x float>]">
// CHECK:       llvm.return %[[T6]] : !llvm<"[2 x <3 x float>]">

func @broadcast_vec3d_from_scalar(%arg0: f32) -> vector<2x3x4xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3x4xf32>
  return %0 : vector<2x3x4xf32>
}
// CHECK-LABEL: llvm.func @broadcast_vec3d_from_scalar(
// CHECK-SAME:  %[[A:.*]]: !llvm.float)
// CHECK:       %[[T0:.*]] = llvm.mlir.undef : !llvm<"[2 x [3 x <4 x float>]]">
// CHECK:       %[[T1:.*]] = llvm.mlir.undef : !llvm<"<4 x float>">
// CHECK:       %[[T2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:       %[[T3:.*]] = llvm.insertelement %[[A]], %[[T1]][%[[T2]] : !llvm.i32] : !llvm<"<4 x float>">
// CHECK:       %[[T4:.*]] = llvm.shufflevector %[[T3]], %[[T3]] [0 : i32, 0 : i32, 0 : i32, 0 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
// CHECK:       %[[T5:.*]] = llvm.insertvalue %[[T4]], %[[T0]][0, 0] : !llvm<"[2 x [3 x <4 x float>]]">
// CHECK:       %[[T6:.*]] = llvm.insertvalue %[[T4]], %[[T5]][0, 1] : !llvm<"[2 x [3 x <4 x float>]]">
// CHECK:       %[[T7:.*]] = llvm.insertvalue %[[T4]], %[[T6]][0, 2] : !llvm<"[2 x [3 x <4 x float>]]">
// CHECK:       %[[T8:.*]] = llvm.insertvalue %[[T4]], %[[T7]][1, 0] : !llvm<"[2 x [3 x <4 x float>]]">
// CHECK:       %[[T9:.*]] = llvm.insertvalue %[[T4]], %[[T8]][1, 1] : !llvm<"[2 x [3 x <4 x float>]]">
// CHECK:       %[[T10:.*]] = llvm.insertvalue %[[T4]], %[[T9]][1, 2] : !llvm<"[2 x [3 x <4 x float>]]">
// CHECK:       llvm.return %[[T10]] : !llvm<"[2 x [3 x <4 x float>]]">

func @broadcast_vec1d_from_vec1d(%arg0: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<2xf32>
  return %0 : vector<2xf32>
}
// CHECK-LABEL: llvm.func @broadcast_vec1d_from_vec1d(
// CHECK-SAME:  %[[A:.*]]: !llvm<"<2 x float>">)
// CHECK:       llvm.return %[[A]] : !llvm<"<2 x float>">

func @broadcast_vec2d_from_vec1d(%arg0: vector<2xf32>) -> vector<3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}
// CHECK-LABEL: llvm.func @broadcast_vec2d_from_vec1d(
// CHECK-SAME:  %[[A:.*]]: !llvm<"<2 x float>">)
// CHECK:       %[[T0:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<3x2xf32>) : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T1:.*]] = llvm.insertvalue %[[A]], %[[T0]][0] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T2:.*]] = llvm.insertvalue %[[A]], %[[T1]][1] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[A]], %[[T2]][2] : !llvm<"[3 x <2 x float>]">
// CHECK:       llvm.return %[[T3]] : !llvm<"[3 x <2 x float>]">

func @broadcast_vec3d_from_vec1d(%arg0: vector<2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}
// CHECK-LABEL: llvm.func @broadcast_vec3d_from_vec1d(
// CHECK-SAME:  %[[A:.*]]: !llvm<"<2 x float>">)
// CHECK:       %[[T0:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<3x2xf32>) : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T1:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<4x3x2xf32>) : !llvm<"[4 x [3 x <2 x float>]]">
// CHECK:       %[[T2:.*]] = llvm.insertvalue %[[A]], %[[T0]][0] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[A]], %[[T2]][1] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T4:.*]] = llvm.insertvalue %[[A]], %[[T3]][2] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T5:.*]] = llvm.insertvalue %[[T4]], %[[T1]][0] : !llvm<"[4 x [3 x <2 x float>]]">
// CHECK:       %[[T6:.*]] = llvm.insertvalue %[[T4]], %[[T5]][1] : !llvm<"[4 x [3 x <2 x float>]]">
// CHECK:       %[[T7:.*]] = llvm.insertvalue %[[T4]], %[[T6]][2] : !llvm<"[4 x [3 x <2 x float>]]">
// CHECK:       %[[T8:.*]] = llvm.insertvalue %[[T4]], %[[T7]][3] : !llvm<"[4 x [3 x <2 x float>]]">
// CHECK:       llvm.return %[[T8]] : !llvm<"[4 x [3 x <2 x float>]]">

func @broadcast_vec3d_from_vec2d(%arg0: vector<3x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<3x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}
// CHECK-LABEL: llvm.func @broadcast_vec3d_from_vec2d(
// CHECK-SAME:  %[[A:.*]]: !llvm<"[3 x <2 x float>]">)
// CHECK:       %[[T0:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<4x3x2xf32>) : !llvm<"[4 x [3 x <2 x float>]]">
// CHECK:       %[[T1:.*]] = llvm.insertvalue %[[A]], %[[T0]][0] : !llvm<"[4 x [3 x <2 x float>]]">
// CHECK:       %[[T2:.*]] = llvm.insertvalue %[[A]], %[[T1]][1] : !llvm<"[4 x [3 x <2 x float>]]">
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[A]], %[[T2]][2] : !llvm<"[4 x [3 x <2 x float>]]">
// CHECK:       %[[T4:.*]] = llvm.insertvalue %[[A]], %[[T3]][3] : !llvm<"[4 x [3 x <2 x float>]]">
// CHECK:       llvm.return %[[T4]] : !llvm<"[4 x [3 x <2 x float>]]">

func @broadcast_stretch(%arg0: vector<1xf32>) -> vector<4xf32> {
  %0 = vector.broadcast %arg0 : vector<1xf32> to vector<4xf32>
  return %0 : vector<4xf32>
}
// CHECK-LABEL: llvm.func @broadcast_stretch(
// CHECK-SAME:  %[[A:.*]]: !llvm<"<1 x float>">)
// CHECK:       %[[T0:.*]] = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK:       %[[T1:.*]] = llvm.extractelement %[[A]][%[[T0]] : !llvm.i64] : !llvm<"<1 x float>">
// CHECK:       %[[T2:.*]] = llvm.mlir.undef : !llvm<"<4 x float>">
// CHECK:       %[[T3:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:       %[[T4:.*]] = llvm.insertelement %[[T1]], %[[T2]][%3 : !llvm.i32] : !llvm<"<4 x float>">
// CHECK:       %[[T5:.*]] = llvm.shufflevector %[[T4]], %[[T2]] [0 : i32, 0 : i32, 0 : i32, 0 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
// CHECK:       llvm.return %[[T5]] : !llvm<"<4 x float>">

func @broadcast_stretch_at_start(%arg0: vector<1x4xf32>) -> vector<3x4xf32> {
  %0 = vector.broadcast %arg0 : vector<1x4xf32> to vector<3x4xf32>
  return %0 : vector<3x4xf32>
}
// CHECK-LABEL: llvm.func @broadcast_stretch_at_start(
// CHECK-SAME:  %[[A:.*]]: !llvm<"[1 x <4 x float>]">)
// CHECK:       %[[T0:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<3x4xf32>) : !llvm<"[3 x <4 x float>]">
// CHECK:       %[[T1:.*]] = llvm.extractvalue %[[A]][0] : !llvm<"[1 x <4 x float>]">
// CHECK:       %[[T2:.*]] = llvm.insertvalue %[[T1]], %[[T0]][0] : !llvm<"[3 x <4 x float>]">
// CHECK:       %[[T3:.*]] = llvm.insertvalue %[[T1]], %[[T2]][1] : !llvm<"[3 x <4 x float>]">
// CHECK:       %[[T4:.*]] = llvm.insertvalue %[[T1]], %[[T3]][2] : !llvm<"[3 x <4 x float>]">
// CHECK:       llvm.return %[[T4]] : !llvm<"[3 x <4 x float>]">

func @broadcast_stretch_at_end(%arg0: vector<4x1xf32>) -> vector<4x3xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1xf32> to vector<4x3xf32>
  return %0 : vector<4x3xf32>
}
// CHECK-LABEL: llvm.func @broadcast_stretch_at_end(
// CHECK-SAME:  %[[A:.*]]: !llvm<"[4 x <1 x float>]">)
// CHECK:       %[[T0:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<4x3xf32>) : !llvm<"[4 x <3 x float>]">
// CHECK:       %[[T1:.*]] = llvm.extractvalue %[[A]][0] : !llvm<"[4 x <1 x float>]">
// CHECK:       %[[T2:.*]] = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK:       %[[T3:.*]] = llvm.extractelement %[[T1]][%[[T2]] : !llvm.i64] : !llvm<"<1 x float>">
// CHECK:       %[[T4:.*]] = llvm.mlir.undef : !llvm<"<3 x float>">
// CHECK:       %[[T5:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:       %[[T6:.*]] = llvm.insertelement %[[T3]], %[[T4]][%[[T5]] : !llvm.i32] : !llvm<"<3 x float>">
// CHECK:       %[[T7:.*]] = llvm.shufflevector %[[T6]], %[[T4]] [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
// CHECK:       %[[T8:.*]] = llvm.insertvalue %[[T7]], %[[T0]][0] : !llvm<"[4 x <3 x float>]">
// CHECK:       %[[T9:.*]] = llvm.extractvalue %[[A]][1] : !llvm<"[4 x <1 x float>]">
// CHECK:       %[[T10:.*]] = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK:       %[[T11:.*]] = llvm.extractelement %[[T9]][%[[T10]] : !llvm.i64] : !llvm<"<1 x float>">
// CHECK:       %[[T12:.*]] = llvm.mlir.undef : !llvm<"<3 x float>">
// CHECK:       %[[T13:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:       %[[T14:.*]] = llvm.insertelement %[[T11]], %[[T12]][%[[T13]] : !llvm.i32] : !llvm<"<3 x float>">
// CHECK:       %[[T15:.*]] = llvm.shufflevector %[[T14]], %[[T12]] [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
// CHECK:       %[[T16:.*]] = llvm.insertvalue %[[T15]], %[[T8]][1] : !llvm<"[4 x <3 x float>]">
// CHECK:       %[[T17:.*]] = llvm.extractvalue %[[A]][2] : !llvm<"[4 x <1 x float>]">
// CHECK:       %[[T18:.*]] = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK:       %[[T19:.*]] = llvm.extractelement %[[T17]][%[[T18]] : !llvm.i64] : !llvm<"<1 x float>">
// CHECK:       %[[T20:.*]] = llvm.mlir.undef : !llvm<"<3 x float>">
// CHECK:       %[[T21:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:       %[[T22:.*]] = llvm.insertelement %[[T19]], %[[T20]][%[[T21]] : !llvm.i32] : !llvm<"<3 x float>">
// CHECK:       %[[T23:.*]] = llvm.shufflevector %[[T22]], %[[T20]] [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
// CHECK:       %[[T24:.*]] = llvm.insertvalue %[[T23]], %[[T16]][2] : !llvm<"[4 x <3 x float>]">
// CHECK:       %[[T25:.*]] = llvm.extractvalue %[[A]][3] : !llvm<"[4 x <1 x float>]">
// CHECK:       %[[T26:.*]] = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK:       %[[T27:.*]] = llvm.extractelement %[[T25]][%[[T26]] : !llvm.i64] : !llvm<"<1 x float>">
// CHECK:       %[[T28:.*]] = llvm.mlir.undef : !llvm<"<3 x float>">
// CHECK:       %[[T29:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:       %[[T30:.*]] = llvm.insertelement %[[T27]], %[[T28]][%[[T29]] : !llvm.i32] : !llvm<"<3 x float>">
// CHECK:       %[[T31:.*]] = llvm.shufflevector %[[T30]], %[[T28]] [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
// CHECK:       %[[T32:.*]] = llvm.insertvalue %[[T31]], %[[T24]][3] : !llvm<"[4 x <3 x float>]">
// CHECK:       llvm.return %[[T32]] : !llvm<"[4 x <3 x float>]">

func @broadcast_stretch_in_middle(%arg0: vector<4x1x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}
// CHECK-LABEL: llvm.func @broadcast_stretch_in_middle(
// CHECK-SAME:  %[[A:.*]]: !llvm<"[4 x [1 x <2 x float>]]">)
// CHECK:       %[[T0:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<4x3x2xf32>) : !llvm<"[4 x [3 x <2 x float>]]">
// CHECK:       %[[T1:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<3x2xf32>) : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T2:.*]] = llvm.extractvalue %[[A]][0] : !llvm<"[4 x [1 x <2 x float>]]">
// CHECK:       %[[T3:.*]] = llvm.extractvalue %[[T2]][0] : !llvm<"[1 x <2 x float>]">
// CHECK:       %[[T4:.*]] = llvm.insertvalue %[[T3]], %[[T1]][0] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T5:.*]] = llvm.insertvalue %[[T3]], %[[T4]][1] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T6:.*]] = llvm.insertvalue %[[T3]], %[[T5]][2] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T7:.*]] = llvm.insertvalue %[[T6]], %[[T0]][0] : !llvm<"[4 x [3 x <2 x float>]]">
// CHECK:       %[[T8:.*]] = llvm.extractvalue %[[A]][1] : !llvm<"[4 x [1 x <2 x float>]]">
// CHECK:       %[[T9:.*]] = llvm.extractvalue %[[T8]][0] : !llvm<"[1 x <2 x float>]">
// CHECK:       %[[T10:.*]] = llvm.insertvalue %[[T9]], %[[T1]][0] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T11:.*]] = llvm.insertvalue %[[T9]], %[[T10]][1] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T12:.*]] = llvm.insertvalue %[[T9]], %[[T11]][2] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T13:.*]] = llvm.insertvalue %[[T12]], %[[T7]][1] : !llvm<"[4 x [3 x <2 x float>]]">
// CHECK:       %[[T14:.*]] = llvm.extractvalue %[[A]][2] : !llvm<"[4 x [1 x <2 x float>]]">
// CHECK:       %[[T15:.*]] = llvm.extractvalue %[[T14]][0] : !llvm<"[1 x <2 x float>]">
// CHECK:       %[[T16:.*]] = llvm.insertvalue %[[T15]], %[[T1]][0] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T17:.*]] = llvm.insertvalue %[[T15]], %[[T16]][1] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T18:.*]] = llvm.insertvalue %[[T15]], %[[T17]][2] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T19:.*]] = llvm.insertvalue %[[T18]], %[[T13]][2] : !llvm<"[4 x [3 x <2 x float>]]">
// CHECK:       %[[T20:.*]] = llvm.extractvalue %[[A]][3] : !llvm<"[4 x [1 x <2 x float>]]">
// CHECK:       %[[T21:.*]] = llvm.extractvalue %[[T20]][0] : !llvm<"[1 x <2 x float>]">
// CHECK:       %[[T22:.*]] = llvm.insertvalue %[[T21]], %[[T1]][0] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T23:.*]] = llvm.insertvalue %[[T21]], %[[T22]][1] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T24:.*]] = llvm.insertvalue %[[T21]], %[[T23]][2] : !llvm<"[3 x <2 x float>]">
// CHECK:       %[[T25:.*]] = llvm.insertvalue %[[T24]], %[[T19]][3] : !llvm<"[4 x [3 x <2 x float>]]">
// CHECK:       llvm.return %[[T25]] : !llvm<"[4 x [3 x <2 x float>]]">

func @outerproduct(%arg0: vector<2xf32>, %arg1: vector<3xf32>) -> vector<2x3xf32> {
  %2 = vector.outerproduct %arg0, %arg1 : vector<2xf32>, vector<3xf32>
  return %2 : vector<2x3xf32>
}
// CHECK-LABEL: llvm.func @outerproduct(
// CHECK-SAME: %[[A:.*]]: !llvm<"<2 x float>">,
// CHECK-SAME: %[[B:.*]]: !llvm<"<3 x float>">)
//      CHECK: %[[T0:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<2x3xf32>)
//      CHECK: %[[T1:.*]] = llvm.mlir.constant(0 : i64) : !llvm.i64
//      CHECK: %[[T2:.*]] = llvm.extractelement %[[A]][%[[T1]] : !llvm.i64] : !llvm<"<2 x float>">
//      CHECK: %[[T3:.*]] = llvm.mlir.undef : !llvm<"<3 x float>">
//      CHECK: %[[T4:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
//      CHECK: %[[T5:.*]] = llvm.insertelement %[[T2]], %[[T3]][%4 : !llvm.i32] : !llvm<"<3 x float>">
//      CHECK: %[[T6:.*]] = llvm.shufflevector %[[T5]], %[[T3]] [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
//      CHECK: %[[T7:.*]] = llvm.fmul %[[T6]], %[[B]] : !llvm<"<3 x float>">
//      CHECK: %[[T8:.*]] = llvm.insertvalue %[[T7]], %[[T0]][0] : !llvm<"[2 x <3 x float>]">
//      CHECK: %[[T9:.*]] = llvm.mlir.constant(1 : i64) : !llvm.i64
//      CHECK: %[[T10:.*]] = llvm.extractelement %[[A]][%9 : !llvm.i64] : !llvm<"<2 x float>">
//      CHECK: %[[T11:.*]] = llvm.mlir.undef : !llvm<"<3 x float>">
//      CHECK: %[[T12:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
//      CHECK: %[[T13:.*]] = llvm.insertelement %[[T10]], %[[T11]][%12 : !llvm.i32] : !llvm<"<3 x float>">
//      CHECK: %[[T14:.*]] = llvm.shufflevector %[[T13]], %[[T11]] [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
//      CHECK: %[[T15:.*]] = llvm.fmul %[[T14]], %[[B]] : !llvm<"<3 x float>">
//      CHECK: %[[T16:.*]] = llvm.insertvalue %[[T15]], %[[T8]][1] : !llvm<"[2 x <3 x float>]">
//      CHECK: llvm.return %[[T16]] : !llvm<"[2 x <3 x float>]">

func @outerproduct_add(%arg0: vector<2xf32>, %arg1: vector<3xf32>, %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %2 = vector.outerproduct %arg0, %arg1, %arg2 : vector<2xf32>, vector<3xf32>
  return %2 : vector<2x3xf32>
}
// CHECK-LABEL: llvm.func @outerproduct_add(
// CHECK-SAME: %[[A:.*]]: !llvm<"<2 x float>">,
// CHECK-SAME: %[[B:.*]]: !llvm<"<3 x float>">,
// CHECK-SAME: %[[C:.*]]: !llvm<"[2 x <3 x float>]">)
//      CHECK: %[[T0:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<2x3xf32>)
//      CHECK: %[[T1:.*]] = llvm.mlir.constant(0 : i64) : !llvm.i64
//      CHECK: %[[T2:.*]] = llvm.extractelement %[[A]][%[[T1]] : !llvm.i64] : !llvm<"<2 x float>">
//      CHECK: %[[T3:.*]] = llvm.mlir.undef : !llvm<"<3 x float>">
//      CHECK: %[[T4:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
//      CHECK: %[[T5:.*]] = llvm.insertelement %[[T2]], %[[T3]][%[[T4]] : !llvm.i32] : !llvm<"<3 x float>">
//      CHECK: %[[T6:.*]] = llvm.shufflevector %[[T5]], %[[T3]] [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
//      CHECK: %[[T7:.*]] = llvm.extractvalue %[[C]][0] : !llvm<"[2 x <3 x float>]">
//      CHECK: %[[T8:.*]] = "llvm.intr.fma"(%[[T6]], %[[B]], %[[T7]]) : (!llvm<"<3 x float>">, !llvm<"<3 x float>">, !llvm<"<3 x float>">)
//      CHECK: %[[T9:.*]] = llvm.insertvalue %[[T8]], %[[T0]][0] : !llvm<"[2 x <3 x float>]">
//      CHECK: %[[T10:.*]] = llvm.mlir.constant(1 : i64) : !llvm.i64
//      CHECK: %[[T11:.*]] = llvm.extractelement %[[A]][%[[T10]] : !llvm.i64] : !llvm<"<2 x float>">
//      CHECK: %[[T12:.*]] = llvm.mlir.undef : !llvm<"<3 x float>">
//      CHECK: %[[T13:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
//      CHECK: %[[T14:.*]] = llvm.insertelement %[[T11]], %[[T12]][%[[T13]] : !llvm.i32] : !llvm<"<3 x float>">
//      CHECK: %[[T15:.*]] = llvm.shufflevector %[[T14]], %[[T12]] [0 : i32, 0 : i32, 0 : i32] : !llvm<"<3 x float>">, !llvm<"<3 x float>">
//      CHECK: %[[T16:.*]] = llvm.extractvalue %[[C]][1] : !llvm<"[2 x <3 x float>]">
//      CHECK: %[[T17:.*]] = "llvm.intr.fma"(%[[T15]], %[[B]], %[[T16]]) : (!llvm<"<3 x float>">, !llvm<"<3 x float>">, !llvm<"<3 x float>">)
//      CHECK: %[[T18:.*]] = llvm.insertvalue %[[T17]], %[[T9]][1] : !llvm<"[2 x <3 x float>]">
//      CHECK: llvm.return %[[T18]] : !llvm<"[2 x <3 x float>]">

func @shuffle_1D_direct(%arg0: vector<2xf32>, %arg1: vector<2xf32>) -> vector<2xf32> {
  %1 = vector.shuffle %arg0, %arg1 [0, 1] : vector<2xf32>, vector<2xf32>
  return %1 : vector<2xf32>
}
// CHECK-LABEL: llvm.func @shuffle_1D_direct(
// CHECK-SAME: %[[A:.*]]: !llvm<"<2 x float>">,
// CHECK-SAME: %[[B:.*]]: !llvm<"<2 x float>">)
//       CHECK:   %[[s:.*]] = llvm.shufflevector %[[A]], %[[B]] [0, 1] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//       CHECK:   llvm.return %[[s]] : !llvm<"<2 x float>">

func @shuffle_1D(%arg0: vector<2xf32>, %arg1: vector<3xf32>) -> vector<5xf32> {
  %1 = vector.shuffle %arg0, %arg1 [4, 3, 2, 1, 0] : vector<2xf32>, vector<3xf32>
  return %1 : vector<5xf32>
}
// CHECK-LABEL: llvm.func @shuffle_1D(
// CHECK-SAME: %[[A:.*]]: !llvm<"<2 x float>">,
// CHECK-SAME: %[[B:.*]]: !llvm<"<3 x float>">)
//       CHECK:   %[[u0:.*]] = llvm.mlir.undef : !llvm<"<5 x float>">
//       CHECK:   %[[c2:.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
//       CHECK:   %[[e1:.*]] = llvm.extractelement %[[B]][%[[c2]] : !llvm.i64] : !llvm<"<3 x float>">
//       CHECK:   %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   %[[i1:.*]] = llvm.insertelement %[[e1]], %[[u0]][%[[c0]] : !llvm.i64] : !llvm<"<5 x float>">
//       CHECK:   %[[c1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:   %[[e2:.*]] = llvm.extractelement %[[B]][%[[c1]] : !llvm.i64] : !llvm<"<3 x float>">
//       CHECK:   %[[c1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:   %[[i2:.*]] = llvm.insertelement %[[e2]], %[[i1]][%[[c1]] : !llvm.i64] : !llvm<"<5 x float>">
//       CHECK:   %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   %[[e3:.*]] = llvm.extractelement %[[B]][%[[c0]] : !llvm.i64] : !llvm<"<3 x float>">
//       CHECK:   %[[c2:.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
//       CHECK:   %[[i3:.*]] = llvm.insertelement %[[e3]], %[[i2]][%[[c2]] : !llvm.i64] : !llvm<"<5 x float>">
//       CHECK:   %[[c1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:   %[[e4:.*]] = llvm.extractelement %[[A]][%[[c1]] : !llvm.i64] : !llvm<"<2 x float>">
//       CHECK:   %[[c3:.*]] = llvm.mlir.constant(3 : index) : !llvm.i64
//       CHECK:   %[[i4:.*]] = llvm.insertelement %[[e4]], %[[i3]][%[[c3]] : !llvm.i64] : !llvm<"<5 x float>">
//       CHECK:   %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:   %[[e5:.*]] = llvm.extractelement %[[A]][%[[c0]] : !llvm.i64] : !llvm<"<2 x float>">
//       CHECK:   %[[c4:.*]] = llvm.mlir.constant(4 : index) : !llvm.i64
//       CHECK:   %[[i5:.*]] = llvm.insertelement %[[e5]], %[[i4]][%[[c4]] : !llvm.i64] : !llvm<"<5 x float>">
//       CHECK:   llvm.return %[[i5]] : !llvm<"<5 x float>">

func @shuffle_2D(%a: vector<1x4xf32>, %b: vector<2x4xf32>) -> vector<3x4xf32> {
  %1 = vector.shuffle %a, %b[1, 0, 2] : vector<1x4xf32>, vector<2x4xf32>
  return %1 : vector<3x4xf32>
}
// CHECK-LABEL: llvm.func @shuffle_2D(
// CHECK-SAME: %[[A:.*]]: !llvm<"[1 x <4 x float>]">,
// CHECK-SAME: %[[B:.*]]: !llvm<"[2 x <4 x float>]">)
//       CHECK:   %[[u0:.*]] = llvm.mlir.undef : !llvm<"[3 x <4 x float>]">
//       CHECK:   %[[e1:.*]] = llvm.extractvalue %[[B]][0] : !llvm<"[2 x <4 x float>]">
//       CHECK:   %[[i1:.*]] = llvm.insertvalue %[[e1]], %[[u0]][0] : !llvm<"[3 x <4 x float>]">
//       CHECK:   %[[e2:.*]] = llvm.extractvalue %[[A]][0] : !llvm<"[1 x <4 x float>]">
//       CHECK:   %[[i2:.*]] = llvm.insertvalue %[[e2]], %[[i1]][1] : !llvm<"[3 x <4 x float>]">
//       CHECK:   %[[e3:.*]] = llvm.extractvalue %[[B]][1] : !llvm<"[2 x <4 x float>]">
//       CHECK:   %[[i3:.*]] = llvm.insertvalue %[[e3]], %[[i2]][2] : !llvm<"[3 x <4 x float>]">
//       CHECK:   llvm.return %[[i3]] : !llvm<"[3 x <4 x float>]">

func @extract_element(%arg0: vector<16xf32>) -> f32 {
  %0 = constant 15 : i32
  %1 = vector.extractelement %arg0[%0 : i32]: vector<16xf32>
  return %1 : f32
}
// CHECK-LABEL: llvm.func @extract_element(
// CHECK-SAME: %[[A:.*]]: !llvm<"<16 x float>">)
//       CHECK:   %[[c:.*]] = llvm.mlir.constant(15 : i32) : !llvm.i32
//       CHECK:   %[[x:.*]] = llvm.extractelement %[[A]][%[[c]] : !llvm.i32] : !llvm<"<16 x float>">
//       CHECK:   llvm.return %[[x]] : !llvm.float

func @extract_element_from_vec_1d(%arg0: vector<16xf32>) -> f32 {
  %0 = vector.extract %arg0[15]: vector<16xf32>
  return %0 : f32
}
// CHECK-LABEL: llvm.func @extract_element_from_vec_1d
//       CHECK:   llvm.mlir.constant(15 : i64) : !llvm.i64
//       CHECK:   llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<16 x float>">
//       CHECK:   llvm.return {{.*}} : !llvm.float

func @extract_vec_2d_from_vec_3d(%arg0: vector<4x3x16xf32>) -> vector<3x16xf32> {
  %0 = vector.extract %arg0[0]: vector<4x3x16xf32>
  return %0 : vector<3x16xf32>
}
// CHECK-LABEL: llvm.func @extract_vec_2d_from_vec_3d
//       CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[4 x [3 x <16 x float>]]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[3 x <16 x float>]">

func @extract_vec_1d_from_vec_3d(%arg0: vector<4x3x16xf32>) -> vector<16xf32> {
  %0 = vector.extract %arg0[0, 0]: vector<4x3x16xf32>
  return %0 : vector<16xf32>
}
// CHECK-LABEL: llvm.func @extract_vec_1d_from_vec_3d
//       CHECK:   llvm.extractvalue {{.*}}[0, 0] : !llvm<"[4 x [3 x <16 x float>]]">
//       CHECK:   llvm.return {{.*}} : !llvm<"<16 x float>">

func @extract_element_from_vec_3d(%arg0: vector<4x3x16xf32>) -> f32 {
  %0 = vector.extract %arg0[0, 0, 0]: vector<4x3x16xf32>
  return %0 : f32
}
// CHECK-LABEL: llvm.func @extract_element_from_vec_3d
//       CHECK:   llvm.extractvalue {{.*}}[0, 0] : !llvm<"[4 x [3 x <16 x float>]]">
//       CHECK:   llvm.mlir.constant(0 : i64) : !llvm.i64
//       CHECK:   llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<16 x float>">
//       CHECK:   llvm.return {{.*}} : !llvm.float

func @insert_element(%arg0: f32, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = constant 3 : i32
  %1 = vector.insertelement %arg0, %arg1[%0 : i32] : vector<4xf32>
  return %1 : vector<4xf32>
}
// CHECK-LABEL: llvm.func @insert_element(
// CHECK-SAME: %[[A:.*]]: !llvm.float,
// CHECK-SAME: %[[B:.*]]: !llvm<"<4 x float>">)
//       CHECK:   %[[c:.*]] = llvm.mlir.constant(3 : i32) : !llvm.i32
//       CHECK:   %[[x:.*]] = llvm.insertelement %[[A]], %[[B]][%[[c]] : !llvm.i32] : !llvm<"<4 x float>">
//       CHECK:   llvm.return %[[x]] : !llvm<"<4 x float>">

func @insert_element_into_vec_1d(%arg0: f32, %arg1: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.insert %arg0, %arg1[3] : f32 into vector<4xf32>
  return %0 : vector<4xf32>
}
// CHECK-LABEL: llvm.func @insert_element_into_vec_1d
//       CHECK:   llvm.mlir.constant(3 : i64) : !llvm.i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<4 x float>">
//       CHECK:   llvm.return {{.*}} : !llvm<"<4 x float>">

func @insert_vec_2d_into_vec_3d(%arg0: vector<8x16xf32>, %arg1: vector<4x8x16xf32>) -> vector<4x8x16xf32> {
  %0 = vector.insert %arg0, %arg1[3] : vector<8x16xf32> into vector<4x8x16xf32>
  return %0 : vector<4x8x16xf32>
}
// CHECK-LABEL: llvm.func @insert_vec_2d_into_vec_3d
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3] : !llvm<"[4 x [8 x <16 x float>]]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[4 x [8 x <16 x float>]]">

func @insert_vec_1d_into_vec_3d(%arg0: vector<16xf32>, %arg1: vector<4x8x16xf32>) -> vector<4x8x16xf32> {
  %0 = vector.insert %arg0, %arg1[3, 7] : vector<16xf32> into vector<4x8x16xf32>
  return %0 : vector<4x8x16xf32>
}
// CHECK-LABEL: llvm.func @insert_vec_1d_into_vec_3d
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3, 7] : !llvm<"[4 x [8 x <16 x float>]]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[4 x [8 x <16 x float>]]">

func @insert_element_into_vec_3d(%arg0: f32, %arg1: vector<4x8x16xf32>) -> vector<4x8x16xf32> {
  %0 = vector.insert %arg0, %arg1[3, 7, 15] : f32 into vector<4x8x16xf32>
  return %0 : vector<4x8x16xf32>
}
// CHECK-LABEL: llvm.func @insert_element_into_vec_3d
//       CHECK:   llvm.extractvalue {{.*}}[3, 7] : !llvm<"[4 x [8 x <16 x float>]]">
//       CHECK:   llvm.mlir.constant(15 : i64) : !llvm.i64
//       CHECK:   llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<16 x float>">
//       CHECK:   llvm.insertvalue {{.*}}, {{.*}}[3, 7] : !llvm<"[4 x [8 x <16 x float>]]">
//       CHECK:   llvm.return {{.*}} : !llvm<"[4 x [8 x <16 x float>]]">

func @vector_type_cast(%arg0: memref<8x8x8xf32>) -> memref<vector<8x8x8xf32>> {
  %0 = vector.type_cast %arg0: memref<8x8x8xf32> to memref<vector<8x8x8xf32>>
  return %0 : memref<vector<8x8x8xf32>>
}
// CHECK-LABEL: llvm.func @vector_type_cast
//       CHECK:   llvm.mlir.undef : !llvm<"{ [8 x [8 x <8 x float>]]*, [8 x [8 x <8 x float>]]*, i64 }">
//       CHECK:   %[[allocated:.*]] = llvm.extractvalue {{.*}}[0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   %[[allocatedBit:.*]] = llvm.bitcast %[[allocated]] : !llvm<"float*"> to !llvm<"[8 x [8 x <8 x float>]]*">
//       CHECK:   llvm.insertvalue %[[allocatedBit]], {{.*}}[0] : !llvm<"{ [8 x [8 x <8 x float>]]*, [8 x [8 x <8 x float>]]*, i64 }">
//       CHECK:   %[[aligned:.*]] = llvm.extractvalue {{.*}}[1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   %[[alignedBit:.*]] = llvm.bitcast %[[aligned]] : !llvm<"float*"> to !llvm<"[8 x [8 x <8 x float>]]*">
//       CHECK:   llvm.insertvalue %[[alignedBit]], {{.*}}[1] : !llvm<"{ [8 x [8 x <8 x float>]]*, [8 x [8 x <8 x float>]]*, i64 }">
//       CHECK:   llvm.mlir.constant(0 : index
//       CHECK:   llvm.insertvalue {{.*}}[2] : !llvm<"{ [8 x [8 x <8 x float>]]*, [8 x [8 x <8 x float>]]*, i64 }">

func @vector_type_cast_non_zero_addrspace(%arg0: memref<8x8x8xf32, 3>) -> memref<vector<8x8x8xf32>, 3> {
  %0 = vector.type_cast %arg0: memref<8x8x8xf32, 3> to memref<vector<8x8x8xf32>, 3>
  return %0 : memref<vector<8x8x8xf32>, 3>
}
// CHECK-LABEL: llvm.func @vector_type_cast_non_zero_addrspace
//       CHECK:   llvm.mlir.undef : !llvm<"{ [8 x [8 x <8 x float>]] addrspace(3)*, [8 x [8 x <8 x float>]] addrspace(3)*, i64 }">
//       CHECK:   %[[allocated:.*]] = llvm.extractvalue {{.*}}[0] : !llvm<"{ float addrspace(3)*, float addrspace(3)*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   %[[allocatedBit:.*]] = llvm.bitcast %[[allocated]] : !llvm<"float addrspace(3)*"> to !llvm<"[8 x [8 x <8 x float>]] addrspace(3)*">
//       CHECK:   llvm.insertvalue %[[allocatedBit]], {{.*}}[0] : !llvm<"{ [8 x [8 x <8 x float>]] addrspace(3)*, [8 x [8 x <8 x float>]] addrspace(3)*, i64 }">
//       CHECK:   %[[aligned:.*]] = llvm.extractvalue {{.*}}[1] : !llvm<"{ float addrspace(3)*, float addrspace(3)*, i64, [3 x i64], [3 x i64] }">
//       CHECK:   %[[alignedBit:.*]] = llvm.bitcast %[[aligned]] : !llvm<"float addrspace(3)*"> to !llvm<"[8 x [8 x <8 x float>]] addrspace(3)*">
//       CHECK:   llvm.insertvalue %[[alignedBit]], {{.*}}[1] : !llvm<"{ [8 x [8 x <8 x float>]] addrspace(3)*, [8 x [8 x <8 x float>]] addrspace(3)*, i64 }">
//       CHECK:   llvm.mlir.constant(0 : index
//       CHECK:   llvm.insertvalue {{.*}}[2] : !llvm<"{ [8 x [8 x <8 x float>]] addrspace(3)*, [8 x [8 x <8 x float>]] addrspace(3)*, i64 }">

func @vector_print_scalar_i1(%arg0: i1) {
  vector.print %arg0 : i1
  return
}
// CHECK-LABEL: llvm.func @vector_print_scalar_i1(
// CHECK-SAME: %[[A:.*]]: !llvm.i1)
//       CHECK: %[[T:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
//       CHECK: %[[F:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
//       CHECK: %[[S:.*]] = llvm.select %[[A]], %[[T]], %[[F]] : !llvm.i1, !llvm.i32
//       CHECK: llvm.call @print_i32(%[[S]]) : (!llvm.i32) -> ()
//       CHECK: llvm.call @print_newline() : () -> ()

func @vector_print_scalar_i32(%arg0: i32) {
  vector.print %arg0 : i32
  return
}
// CHECK-LABEL: llvm.func @vector_print_scalar_i32(
// CHECK-SAME: %[[A:.*]]: !llvm.i32)
//       CHECK:    llvm.call @print_i32(%[[A]]) : (!llvm.i32) -> ()
//       CHECK:    llvm.call @print_newline() : () -> ()

func @vector_print_scalar_i64(%arg0: i64) {
  vector.print %arg0 : i64
  return
}
// CHECK-LABEL: llvm.func @vector_print_scalar_i64(
// CHECK-SAME: %[[A:.*]]: !llvm.i64)
//       CHECK:    llvm.call @print_i64(%[[A]]) : (!llvm.i64) -> ()
//       CHECK:    llvm.call @print_newline() : () -> ()

func @vector_print_scalar_f32(%arg0: f32) {
  vector.print %arg0 : f32
  return
}
// CHECK-LABEL: llvm.func @vector_print_scalar_f32(
// CHECK-SAME: %[[A:.*]]: !llvm.float)
//       CHECK:    llvm.call @print_f32(%[[A]]) : (!llvm.float) -> ()
//       CHECK:    llvm.call @print_newline() : () -> ()

func @vector_print_scalar_f64(%arg0: f64) {
  vector.print %arg0 : f64
  return
}
// CHECK-LABEL: llvm.func @vector_print_scalar_f64(
// CHECK-SAME: %[[A:.*]]: !llvm.double)
//       CHECK:    llvm.call @print_f64(%[[A]]) : (!llvm.double) -> ()
//       CHECK:    llvm.call @print_newline() : () -> ()

func @vector_print_vector(%arg0: vector<2x2xf32>) {
  vector.print %arg0 : vector<2x2xf32>
  return
}
// CHECK-LABEL: llvm.func @vector_print_vector(
// CHECK-SAME: %[[A:.*]]: !llvm<"[2 x <2 x float>]">)
//       CHECK:    llvm.call @print_open() : () -> ()
//       CHECK:    %[[x0:.*]] = llvm.extractvalue %[[A]][0] : !llvm<"[2 x <2 x float>]">
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
//       CHECK:    %[[x5:.*]] = llvm.extractvalue %[[A]][1] : !llvm<"[2 x <2 x float>]">
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

func @extract_strided_slice1(%arg0: vector<4xf32>) -> vector<2xf32> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
  return %0 : vector<2xf32>
}
// CHECK-LABEL: llvm.func @extract_strided_slice1
//       CHECK:    llvm.mlir.constant(0.000000e+00 : f32) : !llvm.float
//       CHECK:    llvm.mlir.constant(dense<0.000000e+00> : vector<2xf32>) : !llvm<"<2 x float>">
//       CHECK:    llvm.mlir.constant(2 : index) : !llvm.i64
//       CHECK:    llvm.extractelement %{{.*}}[%{{.*}} : !llvm.i64] : !llvm<"<4 x float>">
//       CHECK:    llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:    llvm.insertelement %{{.*}}, %{{.*}}[%{{.*}} : !llvm.i64] : !llvm<"<2 x float>">
//       CHECK:    llvm.mlir.constant(3 : index) : !llvm.i64
//       CHECK:    llvm.extractelement %{{.*}}[%{{.*}} : !llvm.i64] : !llvm<"<4 x float>">
//       CHECK:    llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:    llvm.insertelement %{{.*}}, %{{.*}}[%{{.*}} : !llvm.i64] : !llvm<"<2 x float>">

func @extract_strided_slice2(%arg0: vector<4x8xf32>) -> vector<2x8xf32> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]} : vector<4x8xf32> to vector<2x8xf32>
  return %0 : vector<2x8xf32>
}
// CHECK-LABEL: llvm.func @extract_strided_slice2
//       CHECK:    llvm.mlir.constant(0.000000e+00 : f32) : !llvm.float
//       CHECK:    llvm.mlir.constant(dense<0.000000e+00> : vector<2x8xf32>) : !llvm<"[2 x <8 x float>]">
//       CHECK:    llvm.extractvalue %{{.*}}[2] : !llvm<"[4 x <8 x float>]">
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"[2 x <8 x float>]">
//       CHECK:    llvm.extractvalue %{{.*}}[3] : !llvm<"[4 x <8 x float>]">
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"[2 x <8 x float>]">

func @extract_strided_slice3(%arg0: vector<4x8xf32>) -> vector<2x2xf32> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x8xf32> to vector<2x2xf32>
  return %0 : vector<2x2xf32>
}
// CHECK-LABEL: llvm.func @extract_strided_slice3
//       CHECK:    llvm.mlir.constant(0.000000e+00 : f32) : !llvm.float
//       CHECK:    llvm.mlir.constant(dense<0.000000e+00> : vector<2x2xf32>) : !llvm<"[2 x <2 x float>]">
//
// Subvector vector<8xf32> @2
//       CHECK:    llvm.extractvalue {{.*}}[2] : !llvm<"[4 x <8 x float>]">
//       CHECK:    llvm.mlir.constant(0.000000e+00 : f32) : !llvm.float
//       CHECK:    llvm.mlir.constant(dense<0.000000e+00> : vector<2xf32>) : !llvm<"<2 x float>">
//       CHECK:    llvm.mlir.constant(2 : index) : !llvm.i64
//       CHECK:    llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<8 x float>">
//       CHECK:    llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:    llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<2 x float>">
//       CHECK:    llvm.mlir.constant(3 : index) : !llvm.i64
//       CHECK:    llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<8 x float>">
//       CHECK:    llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:    llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<2 x float>">
//       CHECK:    llvm.insertvalue {{.*}}, {{.*}}[0] : !llvm<"[2 x <2 x float>]">
//
// Subvector vector<8xf32> @3
//       CHECK:    llvm.extractvalue {{.*}}[3] : !llvm<"[4 x <8 x float>]">
//       CHECK:    llvm.mlir.constant(0.000000e+00 : f32) : !llvm.float
//       CHECK:    llvm.mlir.constant(dense<0.000000e+00> : vector<2xf32>) : !llvm<"<2 x float>">
//       CHECK:    llvm.mlir.constant(2 : index) : !llvm.i64
//       CHECK:    llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<8 x float>">
//       CHECK:    llvm.mlir.constant(0 : index) : !llvm.i64
//       CHECK:    llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<2 x float>">
//       CHECK:    llvm.mlir.constant(3 : index) : !llvm.i64
//       CHECK:    llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<8 x float>">
//       CHECK:    llvm.mlir.constant(1 : index) : !llvm.i64
//       CHECK:    llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<2 x float>">
//       CHECK:    llvm.insertvalue {{.*}}, {{.*}}[1] : !llvm<"[2 x <2 x float>]">

func @insert_strided_slice1(%b: vector<4x4xf32>, %c: vector<4x4x4xf32>) -> vector<4x4x4xf32> {
  %0 = vector.insert_strided_slice %b, %c {offsets = [2, 0, 0], strides = [1, 1]} : vector<4x4xf32> into vector<4x4x4xf32>
  return %0 : vector<4x4x4xf32>
}
// CHECK-LABEL: llvm.func @insert_strided_slice1
//       CHECK:    llvm.extractvalue {{.*}}[2] : !llvm<"[4 x [4 x <4 x float>]]">
//  CHECK-NEXT:    llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[4 x [4 x <4 x float>]]">

func @insert_strided_slice2(%a: vector<2x2xf32>, %b: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}
// CHECK-LABEL: llvm.func @insert_strided_slice2
//
// Subvector vector<2xf32> @0 into vector<4xf32> @2
//       CHECK:    llvm.extractvalue {{.*}}[0] : !llvm<"[2 x <2 x float>]">
//  CHECK-NEXT:    llvm.extractvalue {{.*}}[2] : !llvm<"[4 x <4 x float>]">
// Element @0 -> element @2
//  CHECK-NEXT:    llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:    llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<2 x float>">
//  CHECK-NEXT:    llvm.mlir.constant(2 : index) : !llvm.i64
//  CHECK-NEXT:    llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<4 x float>">
// Element @1 -> element @3
//  CHECK-NEXT:    llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:    llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<2 x float>">
//  CHECK-NEXT:    llvm.mlir.constant(3 : index) : !llvm.i64
//  CHECK-NEXT:    llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<4 x float>">
//  CHECK-NEXT:    llvm.insertvalue {{.*}}, {{.*}}[2] : !llvm<"[4 x <4 x float>]">
//
// Subvector vector<2xf32> @1 into vector<4xf32> @3
//       CHECK:    llvm.extractvalue {{.*}}[1] : !llvm<"[2 x <2 x float>]">
//  CHECK-NEXT:    llvm.extractvalue {{.*}}[3] : !llvm<"[4 x <4 x float>]">
// Element @0 -> element @2
//  CHECK-NEXT:    llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:    llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<2 x float>">
//  CHECK-NEXT:    llvm.mlir.constant(2 : index) : !llvm.i64
//  CHECK-NEXT:    llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<4 x float>">
// Element @1 -> element @3
//  CHECK-NEXT:    llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:    llvm.extractelement {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<2 x float>">
//  CHECK-NEXT:    llvm.mlir.constant(3 : index) : !llvm.i64
//  CHECK-NEXT:    llvm.insertelement {{.*}}, {{.*}}[{{.*}} : !llvm.i64] : !llvm<"<4 x float>">
//  CHECK-NEXT:    llvm.insertvalue {{.*}}, {{.*}}[3] : !llvm<"[4 x <4 x float>]">

func @insert_strided_slice3(%arg0: vector<2x4xf32>, %arg1: vector<16x4x8xf32>) -> vector<16x4x8xf32> {
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [0, 0, 2], strides = [1, 1]}:
        vector<2x4xf32> into vector<16x4x8xf32>
  return %0 : vector<16x4x8xf32>
}
// CHECK-LABEL: llvm.func @insert_strided_slice3(
// CHECK-SAME: %[[A:.*]]: !llvm<"[2 x <4 x float>]">,
// CHECK-SAME: %[[B:.*]]: !llvm<"[16 x [4 x <8 x float>]]">)
//      CHECK: %[[s0:.*]] = llvm.extractvalue %[[B]][0] : !llvm<"[16 x [4 x <8 x float>]]">
//      CHECK: %[[s1:.*]] = llvm.extractvalue %[[A]][0] : !llvm<"[2 x <4 x float>]">
//      CHECK: %[[s2:.*]] = llvm.extractvalue %[[s0]][0] : !llvm<"[4 x <8 x float>]">
//      CHECK: %[[s3:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//      CHECK: %[[s4:.*]] = llvm.extractelement %[[s1]][%[[s3]] : !llvm.i64] : !llvm<"<4 x float>">
//      CHECK: %[[s5:.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
//      CHECK: %[[s6:.*]] = llvm.insertelement %[[s4]], %[[s2]][%[[s5]] : !llvm.i64] : !llvm<"<8 x float>">
//      CHECK: %[[s7:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//      CHECK: %[[s8:.*]] = llvm.extractelement %[[s1]][%[[s7]] : !llvm.i64] : !llvm<"<4 x float>">
//      CHECK: %[[s9:.*]] = llvm.mlir.constant(3 : index) : !llvm.i64
//      CHECK: %[[s10:.*]] = llvm.insertelement %[[s8]], %[[s6]][%[[s9]] : !llvm.i64] : !llvm<"<8 x float>">
//      CHECK: %[[s11:.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
//      CHECK: %[[s12:.*]] = llvm.extractelement %[[s1]][%[[s11]] : !llvm.i64] : !llvm<"<4 x float>">
//      CHECK: %[[s13:.*]] = llvm.mlir.constant(4 : index) : !llvm.i64
//      CHECK: %[[s14:.*]] = llvm.insertelement %[[s12]], %[[s10]][%[[s13]] : !llvm.i64] : !llvm<"<8 x float>">
//      CHECK: %[[s15:.*]] = llvm.mlir.constant(3 : index) : !llvm.i64
//      CHECK: %[[s16:.*]] = llvm.extractelement %[[s1]][%[[s15]] : !llvm.i64] : !llvm<"<4 x float>">
//      CHECK: %[[s17:.*]] = llvm.mlir.constant(5 : index) : !llvm.i64
//      CHECK: %[[s18:.*]] = llvm.insertelement %[[s16]], %[[s14]][%[[s17]] : !llvm.i64] : !llvm<"<8 x float>">
//      CHECK: %[[s19:.*]] = llvm.insertvalue %[[s18]], %[[s0]][0] : !llvm<"[4 x <8 x float>]">
//      CHECK: %[[s20:.*]] = llvm.extractvalue %[[A]][1] : !llvm<"[2 x <4 x float>]">
//      CHECK: %[[s21:.*]] = llvm.extractvalue %[[s0]][1] : !llvm<"[4 x <8 x float>]">
//      CHECK: %[[s22:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//      CHECK: %[[s23:.*]] = llvm.extractelement %[[s20]][%[[s22]] : !llvm.i64] : !llvm<"<4 x float>">
//      CHECK: %[[s24:.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
//      CHECK: %[[s25:.*]] = llvm.insertelement %[[s23]], %[[s21]][%[[s24]] : !llvm.i64] : !llvm<"<8 x float>">
//      CHECK: %[[s26:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//      CHECK: %[[s27:.*]] = llvm.extractelement %[[s20]][%[[s26]] : !llvm.i64] : !llvm<"<4 x float>">
//      CHECK: %[[s28:.*]] = llvm.mlir.constant(3 : index) : !llvm.i64
//      CHECK: %[[s29:.*]] = llvm.insertelement %[[s27]], %[[s25]][%[[s28]] : !llvm.i64] : !llvm<"<8 x float>">
//      CHECK: %[[s30:.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
//      CHECK: %[[s31:.*]] = llvm.extractelement %[[s20]][%[[s30]] : !llvm.i64] : !llvm<"<4 x float>">
//      CHECK: %[[s32:.*]] = llvm.mlir.constant(4 : index) : !llvm.i64
//      CHECK: %[[s33:.*]] = llvm.insertelement %[[s31]], %[[s29]][%[[s32]] : !llvm.i64] : !llvm<"<8 x float>">
//      CHECK: %[[s34:.*]] = llvm.mlir.constant(3 : index) : !llvm.i64
//      CHECK: %[[s35:.*]] = llvm.extractelement %[[s20]][%[[s34]] : !llvm.i64] : !llvm<"<4 x float>">
//      CHECK: %[[s36:.*]] = llvm.mlir.constant(5 : index) : !llvm.i64
//      CHECK: %[[s37:.*]] = llvm.insertelement %[[s35]], %[[s33]][%[[s36]] : !llvm.i64] : !llvm<"<8 x float>">
//      CHECK: %[[s38:.*]] = llvm.insertvalue %[[s37]], %[[s19]][1] : !llvm<"[4 x <8 x float>]">
//      CHECK: %[[s39:.*]] = llvm.insertvalue %[[s38]], %[[B]][0] : !llvm<"[16 x [4 x <8 x float>]]">
//      CHECK: llvm.return %[[s39]] : !llvm<"[16 x [4 x <8 x float>]]">

func @extract_strides(%arg0: vector<3x3xf32>) -> vector<1x1xf32> {
  %0 = vector.extract_slices %arg0, [2, 2], [1, 1]
    : vector<3x3xf32> into tuple<vector<2x2xf32>, vector<2x1xf32>, vector<1x2xf32>, vector<1x1xf32>>
  %1 = vector.tuple_get %0, 3 : tuple<vector<2x2xf32>, vector<2x1xf32>, vector<1x2xf32>, vector<1x1xf32>>
  return %1 : vector<1x1xf32>
}
// CHECK-LABEL: llvm.func @extract_strides(
// CHECK-SAME: %[[A:.*]]: !llvm<"[3 x <3 x float>]">)
//      CHECK: %[[s0:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<1x1xf32>) : !llvm<"[1 x <1 x float>]">
//      CHECK: %[[s1:.*]] = llvm.extractvalue %[[A]][2] : !llvm<"[3 x <3 x float>]">
//      CHECK: %[[s3:.*]] = llvm.mlir.constant(dense<0.000000e+00> : vector<1xf32>) : !llvm<"<1 x float>">
//      CHECK: %[[s4:.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
//      CHECK: %[[s5:.*]] = llvm.extractelement %[[s1]][%[[s4]] : !llvm.i64] : !llvm<"<3 x float>">
//      CHECK: %[[s6:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//      CHECK: %[[s7:.*]] = llvm.insertelement %[[s5]], %[[s3]][%[[s6]] : !llvm.i64] : !llvm<"<1 x float>">
//      CHECK: %[[s8:.*]] = llvm.insertvalue %[[s7]], %[[s0]][0] : !llvm<"[1 x <1 x float>]">
//      CHECK: llvm.return %[[s8]] : !llvm<"[1 x <1 x float>]">

// CHECK-LABEL: llvm.func @vector_fma(
//  CHECK-SAME: %[[A:.*]]: !llvm<"<8 x float>">, %[[B:.*]]: !llvm<"[2 x <4 x float>]">)
//  CHECK-SAME: -> !llvm<"{ <8 x float>, [2 x <4 x float>] }"> {
func @vector_fma(%a: vector<8xf32>, %b: vector<2x4xf32>) -> (vector<8xf32>, vector<2x4xf32>) {
  //         CHECK: "llvm.intr.fma"(%[[A]], %[[A]], %[[A]]) :
  //    CHECK-SAME:   (!llvm<"<8 x float>">, !llvm<"<8 x float>">, !llvm<"<8 x float>">) -> !llvm<"<8 x float>">
  %0 = vector.fma %a, %a, %a : vector<8xf32>

  //       CHECK: %[[b00:.*]] = llvm.extractvalue %[[B]][0] : !llvm<"[2 x <4 x float>]">
  //       CHECK: %[[b01:.*]] = llvm.extractvalue %[[B]][0] : !llvm<"[2 x <4 x float>]">
  //       CHECK: %[[b02:.*]] = llvm.extractvalue %[[B]][0] : !llvm<"[2 x <4 x float>]">
  //       CHECK: %[[B0:.*]] = "llvm.intr.fma"(%[[b00]], %[[b01]], %[[b02]]) :
  //  CHECK-SAME: (!llvm<"<4 x float>">, !llvm<"<4 x float>">, !llvm<"<4 x float>">) -> !llvm<"<4 x float>">
  //       CHECK: llvm.insertvalue %[[B0]], {{.*}}[0] : !llvm<"[2 x <4 x float>]">
  //       CHECK: %[[b10:.*]] = llvm.extractvalue %[[B]][1] : !llvm<"[2 x <4 x float>]">
  //       CHECK: %[[b11:.*]] = llvm.extractvalue %[[B]][1] : !llvm<"[2 x <4 x float>]">
  //       CHECK: %[[b12:.*]] = llvm.extractvalue %[[B]][1] : !llvm<"[2 x <4 x float>]">
  //       CHECK: %[[B1:.*]] = "llvm.intr.fma"(%[[b10]], %[[b11]], %[[b12]]) :
  //  CHECK-SAME: (!llvm<"<4 x float>">, !llvm<"<4 x float>">, !llvm<"<4 x float>">) -> !llvm<"<4 x float>">
  //       CHECK: llvm.insertvalue %[[B1]], {{.*}}[1] : !llvm<"[2 x <4 x float>]">
  %1 = vector.fma %b, %b, %b : vector<2x4xf32>

  return %0, %1: vector<8xf32>, vector<2x4xf32>
}

func @reduce_f32(%arg0: vector<16xf32>) -> f32 {
  %0 = vector.reduction "add", %arg0 : vector<16xf32> into f32
  return %0 : f32
}
// CHECK-LABEL: llvm.func @reduce_f32(
// CHECK-SAME: %[[A:.*]]: !llvm<"<16 x float>">)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : !llvm.float
//      CHECK: %[[V:.*]] = "llvm.intr.experimental.vector.reduce.v2.fadd"(%[[C]], %[[A]])
//      CHECK: llvm.return %[[V]] : !llvm.float

func @reduce_f64(%arg0: vector<16xf64>) -> f64 {
  %0 = vector.reduction "add", %arg0 : vector<16xf64> into f64
  return %0 : f64
}
// CHECK-LABEL: llvm.func @reduce_f64(
// CHECK-SAME: %[[A:.*]]: !llvm<"<16 x double>">)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f64) : !llvm.double
//      CHECK: %[[V:.*]] = "llvm.intr.experimental.vector.reduce.v2.fadd"(%[[C]], %[[A]])
//      CHECK: llvm.return %[[V]] : !llvm.double

func @reduce_i32(%arg0: vector<16xi32>) -> i32 {
  %0 = vector.reduction "add", %arg0 : vector<16xi32> into i32
  return %0 : i32
}
// CHECK-LABEL: llvm.func @reduce_i32(
// CHECK-SAME: %[[A:.*]]: !llvm<"<16 x i32>">)
//      CHECK: %[[V:.*]] = "llvm.intr.experimental.vector.reduce.add"(%[[A]])
//      CHECK: llvm.return %[[V]] : !llvm.i32

func @reduce_i64(%arg0: vector<16xi64>) -> i64 {
  %0 = vector.reduction "add", %arg0 : vector<16xi64> into i64
  return %0 : i64
}
// CHECK-LABEL: llvm.func @reduce_i64(
// CHECK-SAME: %[[A:.*]]: !llvm<"<16 x i64>">)
//      CHECK: %[[V:.*]] = "llvm.intr.experimental.vector.reduce.add"(%[[A]])
//      CHECK: llvm.return %[[V]] : !llvm.i64


//                          4x16                16x3               4x3
func @matrix_ops(%A: vector<64xf64>, %B: vector<48xf64>) -> vector<12xf64> {
  %C = vector.matrix_multiply %A, %B
    { lhs_rows = 4: i32, lhs_columns = 16: i32 , rhs_columns = 3: i32 } :
    (vector<64xf64>, vector<48xf64>) -> vector<12xf64>
  return %C: vector<12xf64>
}
// CHECK-LABEL: llvm.func @matrix_ops
//       CHECK:   llvm.intr.matrix.multiply %{{.*}}, %{{.*}} {
//  CHECK-SAME: lhs_columns = 16 : i32, lhs_rows = 4 : i32, rhs_columns = 3 : i32
//  CHECK-SAME: } : (!llvm<"<64 x double>">, !llvm<"<48 x double>">) -> !llvm<"<12 x double>">

func @transfer_read_1d(%A : memref<?xf32>, %base: index) -> vector<17xf32> {
  %f7 = constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32>, vector<17xf32>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<17xf32>, memref<?xf32>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_1d
//  CHECK-SAME: %[[BASE:[a-zA-Z0-9]*]]: !llvm.i64) -> !llvm<"<17 x float>">
//
// 1. Bitcast to vector form.
//       CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
//       CHECK: %[[vecPtr:.*]] = llvm.bitcast %[[gep]] :
//  CHECK-SAME: !llvm<"float*"> to !llvm<"<17 x float>*">
//
// 2. Create a vector with linear indices [ 0 .. vector_length - 1 ].
//       CHECK: %[[linearIndex:.*]] = llvm.mlir.constant(
//  CHECK-SAME: dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> :
//  CHECK-SAME: vector<17xi64>) : !llvm<"<17 x i64>">
//
// 3. Create offsetVector = [ offset + 0 .. offset + vector_length - 1 ].
//       CHECK: %[[offsetVec:.*]] = llvm.mlir.undef : !llvm<"<17 x i64>">
//       CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
//       CHECK: %[[offsetVec2:.*]] = llvm.insertelement %[[BASE]], %[[offsetVec]][%[[c0]] :
//  CHECK-SAME: !llvm.i32] : !llvm<"<17 x i64>">
//       CHECK: %[[offsetVec3:.*]] = llvm.shufflevector %[[offsetVec2]], %{{.*}} [
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32,
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32,
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32] :
//  CHECK-SAME: !llvm<"<17 x i64>">, !llvm<"<17 x i64>">
//       CHECK: %[[offsetVec4:.*]] = llvm.add %[[offsetVec3]], %[[linearIndex]] :
//  CHECK-SAME: !llvm<"<17 x i64>">
//
// 4. Let dim the memref dimension, compute the vector comparison mask:
//    [ offset + 0 .. offset + vector_length - 1 ] < [ dim .. dim ]
//       CHECK: %[[DIM:.*]] = llvm.extractvalue %{{.*}}[3, 0] :
//  CHECK-SAME: !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
//       CHECK: %[[dimVec:.*]] = llvm.mlir.undef : !llvm<"<17 x i64>">
//       CHECK: %[[c01:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
//       CHECK: %[[dimVec2:.*]] = llvm.insertelement %[[DIM]], %[[dimVec]][%[[c01]] :
//  CHECK-SAME:  !llvm.i32] : !llvm<"<17 x i64>">
//       CHECK: %[[dimVec3:.*]] = llvm.shufflevector %[[dimVec2]], %{{.*}} [
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32,
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32,
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32] :
//  CHECK-SAME: !llvm<"<17 x i64>">, !llvm<"<17 x i64>">
//       CHECK: %[[mask:.*]] = llvm.icmp "slt" %[[offsetVec4]], %[[dimVec3]] :
//  CHECK-SAME: !llvm<"<17 x i64>">
//
// 5. Rewrite as a masked read.
//       CHECK: %[[PASS_THROUGH:.*]] =  llvm.mlir.constant(dense<7.000000e+00> :
//  CHECK-SAME:  vector<17xf32>) : !llvm<"<17 x float>">
//       CHECK: %[[loaded:.*]] = llvm.intr.masked.load %[[vecPtr]], %[[mask]],
//  CHECK-SAME: %[[PASS_THROUGH]] {alignment = 4 : i32} :
//  CHECK-SAME: (!llvm<"<17 x float>*">, !llvm<"<17 x i1>">, !llvm<"<17 x float>">) -> !llvm<"<17 x float>">

//
// 1. Bitcast to vector form.
//       CHECK: %[[gep_b:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
//       CHECK: %[[vecPtr_b:.*]] = llvm.bitcast %[[gep_b]] :
//  CHECK-SAME: !llvm<"float*"> to !llvm<"<17 x float>*">
//
// 2. Create a vector with linear indices [ 0 .. vector_length - 1 ].
//       CHECK: %[[linearIndex_b:.*]] = llvm.mlir.constant(
//  CHECK-SAME: dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> :
//  CHECK-SAME:  vector<17xi64>) : !llvm<"<17 x i64>">
//
// 3. Create offsetVector = [ offset + 0 .. offset + vector_length - 1 ].
//       CHECK: llvm.shufflevector {{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32,
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32,
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] :
//  CHECK-SAME: !llvm<"<17 x i64>">, !llvm<"<17 x i64>">
//       CHECK: llvm.add
//
// 4. Let dim the memref dimension, compute the vector comparison mask:
//    [ offset + 0 .. offset + vector_length - 1 ] < [ dim .. dim ]
//       CHECK: llvm.shufflevector {{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32,
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32,
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] :
//  CHECK-SAME: !llvm<"<17 x i64>">, !llvm<"<17 x i64>">
//       CHECK: %[[mask_b:.*]] = llvm.icmp "slt" {{.*}} : !llvm<"<17 x i64>">
//
// 5. Rewrite as a masked write.
//       CHECK: llvm.intr.masked.store %[[loaded]], %[[vecPtr_b]], %[[mask_b]]
//  CHECK-SAME: {alignment = 4 : i32} :
//  CHECK-SAME: !llvm<"<17 x float>">, !llvm<"<17 x i1>"> into !llvm<"<17 x float>*">

func @transfer_read_2d_to_1d(%A : memref<?x?xf32>, %base0: index, %base1: index) -> vector<17xf32> {
  %f7 = constant 7.0: f32
  %f = vector.transfer_read %A[%base0, %base1], %f7
      {permutation_map = affine_map<(d0, d1) -> (d1)>} :
    memref<?x?xf32>, vector<17xf32>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_2d_to_1d
//  CHECK-SAME: %[[BASE_0:[a-zA-Z0-9]*]]: !llvm.i64, %[[BASE_1:[a-zA-Z0-9]*]]: !llvm.i64) -> !llvm<"<17 x float>">
//
// Create offsetVector = [ offset + 0 .. offset + vector_length - 1 ].
//       CHECK: %[[offsetVec:.*]] = llvm.mlir.undef : !llvm<"<17 x i64>">
//       CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// Here we check we properly use %BASE_1
//       CHECK: %[[offsetVec2:.*]] = llvm.insertelement %[[BASE_1]], %[[offsetVec]][%[[c0]] :
//  CHECK-SAME: !llvm.i32] : !llvm<"<17 x i64>">
//       CHECK: %[[offsetVec3:.*]] = llvm.shufflevector %[[offsetVec2]], %{{.*}} [
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32,
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32,
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32] :
//
// Let dim the memref dimension, compute the vector comparison mask:
//    [ offset + 0 .. offset + vector_length - 1 ] < [ dim .. dim ]
// Here we check we properly use %DIM[1]
//       CHECK: %[[DIM:.*]] = llvm.extractvalue %{{.*}}[3, 1] :
//  CHECK-SAME: !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//       CHECK: %[[dimVec:.*]] = llvm.mlir.undef : !llvm<"<17 x i64>">
//       CHECK: %[[c01:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
//       CHECK: %[[dimVec2:.*]] = llvm.insertelement %[[DIM]], %[[dimVec]][%[[c01]] :
//  CHECK-SAME:  !llvm.i32] : !llvm<"<17 x i64>">
//       CHECK: %[[dimVec3:.*]] = llvm.shufflevector %[[dimVec2]], %{{.*}} [
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32,
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32,
//  CHECK-SAME:  0 : i32, 0 : i32, 0 : i32] :
//  CHECK-SAME: !llvm<"<17 x i64>">, !llvm<"<17 x i64>">

func @transfer_read_1d_non_zero_addrspace(%A : memref<?xf32, 3>, %base: index) -> vector<17xf32> {
  %f7 = constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32, 3>, vector<17xf32>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<17xf32>, memref<?xf32, 3>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_1d_non_zero_addrspace
//  CHECK-SAME: %[[BASE:[a-zA-Z0-9]*]]: !llvm.i64) -> !llvm<"<17 x float>">
//
// 1. Check address space for GEP is correct.
//       CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm<"float addrspace(3)*">, !llvm.i64) -> !llvm<"float addrspace(3)*">
//       CHECK: %[[vecPtr:.*]] = llvm.addrspacecast %[[gep]] :
//  CHECK-SAME: !llvm<"float addrspace(3)*"> to !llvm<"<17 x float>*">
//
// 2. Check address space of the memref is correct.
//       CHECK: %[[DIM:.*]] = llvm.extractvalue %{{.*}}[3, 0] :
//  CHECK-SAME: !llvm<"{ float addrspace(3)*, float addrspace(3)*, i64, [1 x i64], [1 x i64] }">
//
// 3. Check address apce for GEP is correct.
//       CHECK: %[[gep_b:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm<"float addrspace(3)*">, !llvm.i64) -> !llvm<"float addrspace(3)*">
//       CHECK: %[[vecPtr_b:.*]] = llvm.addrspacecast %[[gep_b]] :
//  CHECK-SAME: !llvm<"float addrspace(3)*"> to !llvm<"<17 x float>*">

func @transfer_read_1d_not_masked(%A : memref<?xf32>, %base: index) -> vector<17xf32> {
  %f7 = constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7 {masked = [false]} :
    memref<?xf32>, vector<17xf32>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_1d_not_masked
//  CHECK-SAME: %[[BASE:[a-zA-Z0-9]*]]: !llvm.i64) -> !llvm<"<17 x float>">
//
// 1. Bitcast to vector form.
//       CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
//       CHECK: %[[vecPtr:.*]] = llvm.bitcast %[[gep]] :
//  CHECK-SAME: !llvm<"float*"> to !llvm<"<17 x float>*">
//
// 2. Rewrite as a load.
//       CHECK: %[[loaded:.*]] = llvm.load %[[vecPtr]] : !llvm<"<17 x float>*">

func @genbool_1d() -> vector<8xi1> {
  %0 = vector.constant_mask [4] : vector<8xi1>
  return %0 : vector<8xi1>
}
// CHECK-LABEL: func @genbool_1d
// CHECK: %[[C1:.*]] = llvm.mlir.constant(dense<[true, true, true, true, false, false, false, false]> : vector<8xi1>) : !llvm<"<8 x i1>">
// CHECK: llvm.return %[[C1]] : !llvm<"<8 x i1>">

func @genbool_2d() -> vector<4x4xi1> {
  %v = vector.constant_mask [2, 2] : vector<4x4xi1>
  return %v: vector<4x4xi1>
}

// CHECK-LABEL: func @genbool_2d
// CHECK: %[[C1:.*]] = llvm.mlir.constant(dense<[true, true, false, false]> : vector<4xi1>) : !llvm<"<4 x i1>">
// CHECK: %[[C2:.*]] = llvm.mlir.constant(dense<false> : vector<4x4xi1>) : !llvm<"[4 x <4 x i1>]">
// CHECK: %[[T0:.*]] = llvm.insertvalue %[[C1]], %[[C2]][0] : !llvm<"[4 x <4 x i1>]">
// CHECK: %[[T1:.*]] = llvm.insertvalue %[[C1]], %[[T0]][1] : !llvm<"[4 x <4 x i1>]">
// CHECK: llvm.return %[[T1]] : !llvm<"[4 x <4 x i1>]">

func @flat_transpose(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = vector.flat_transpose %arg0 { rows = 4: i32, columns = 4: i32 }
     : vector<16xf32> -> vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @flat_transpose
// CHECK-SAME:  %[[A:.*]]: !llvm<"<16 x float>">
// CHECK:       %[[T:.*]] = llvm.intr.matrix.transpose %[[A]]
// CHECK-SAME:      {columns = 4 : i32, rows = 4 : i32} :
// CHECK-SAME:      !llvm<"<16 x float>"> into !llvm<"<16 x float>">
// CHECK:       llvm.return %[[T]] : !llvm<"<16 x float>">
