// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func @broadcast_vec1d_from_scalar
// CHECK-SAME: %[[A:.*0]]: f32
// CHECK:      %[[T0:.*]] = vector.splat %[[A]] : vector<2xf32>
// CHECK:      return %[[T0]] : vector<2xf32>

func.func @broadcast_vec1d_from_scalar(%arg0: f32) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: func @broadcast_vec2d_from_scalar
// CHECK-SAME: %[[A:.*0]]: f32
// CHECK:      %[[T0:.*]] = vector.splat %[[A]] : vector<2x3xf32>
// CHECK:      return %[[T0]] : vector<2x3xf32>

func.func @broadcast_vec2d_from_scalar(%arg0: f32) -> vector<2x3xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// CHECK-LABEL: func @broadcast_vec3d_from_scalar
// CHECK-SAME: %[[A:.*0]]: f32
// CHECK:      %[[T0:.*]] = vector.splat %[[A]] : vector<2x3x4xf32>
// CHECK:      return %[[T0]] : vector<2x3x4xf32>

func.func @broadcast_vec3d_from_scalar(%arg0: f32) -> vector<2x3x4xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<2x3x4xf32>
  return %0 : vector<2x3x4xf32>
}

// CHECK-LABEL: func @broadcast_vec1d_from_vec1d
// CHECK-SAME: %[[A:.*0]]: vector<2xf32>
// CHECK:      return %[[A]] : vector<2xf32>

func.func @broadcast_vec1d_from_vec1d(%arg0: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: func @broadcast_vec2d_from_vec1d
// CHECK-SAME: %[[A:.*0]]: vector<2xf32>
// CHECK:      %[[U0:.*]] = ub.poison : vector<3x2xf32>
// CHECK:      %[[T0:.*]] = vector.insert %[[A]], %[[U0]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T1:.*]] = vector.insert %[[A]], %[[T0]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T2:.*]] = vector.insert %[[A]], %[[T1]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      return %[[T2]] : vector<3x2xf32>

func.func @broadcast_vec2d_from_vec1d(%arg0: vector<2xf32>) -> vector<3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

// CHECK-LABEL: func @broadcast_vec3d_from_vec1d
// CHECK-SAME: %[[A:.*0]]: vector<2xf32>
// CHECK-DAG:  %[[U0:.*]] = ub.poison : vector<3x2xf32>
// CHECK-DAG:  %[[U1:.*]] = ub.poison : vector<4x3x2xf32>
// CHECK:      %[[T0:.*]] = vector.insert %[[A]], %[[U0]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T1:.*]] = vector.insert %[[A]], %[[T0]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T2:.*]] = vector.insert %[[A]], %[[T1]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[U1]] [0] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T4:.*]] = vector.insert %[[T2]], %[[T3]] [1] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T5:.*]] = vector.insert %[[T2]], %[[T4]] [2] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T6:.*]] = vector.insert %[[T2]], %[[T5]] [3] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:       return %[[T6]] : vector<4x3x2xf32>

func.func @broadcast_vec3d_from_vec1d(%arg0: vector<2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}

// CHECK-LABEL: func @broadcast_vec3d_from_vec2d
// CHECK-SAME: %[[A:.*0]]: vector<3x2xf32>
// CHECK:      %[[U0:.*]] = ub.poison : vector<4x3x2xf32>
// CHECK:      %[[T0:.*]] = vector.insert %[[A]], %[[U0]] [0] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T1:.*]] = vector.insert %[[A]], %[[T0]] [1] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T2:.*]] = vector.insert %[[A]], %[[T1]] [2] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[A]], %[[T2]] [3] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      return %[[T3]] : vector<4x3x2xf32>

func.func @broadcast_vec3d_from_vec2d(%arg0: vector<3x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<3x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}

// CHECK-LABEL: func @broadcast_stretch
// CHECK-SAME: %[[A:.*0]]: vector<1xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : f32 from vector<1xf32>
// CHECK:      %[[T1:.*]] = vector.splat %[[T0]] : vector<4xf32>
// CHECK:      return %[[T1]] : vector<4xf32>

func.func @broadcast_stretch(%arg0: vector<1xf32>) -> vector<4xf32> {
  %0 = vector.broadcast %arg0 : vector<1xf32> to vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func @broadcast_stretch_at_start
// CHECK-SAME: %[[A:.*0]]: vector<1x4xf32>
// CHECK:      %[[U0:.*]] = ub.poison : vector<3x4xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<4xf32> from vector<1x4xf32>
// CHECK:      %[[T1:.*]] = vector.insert %[[T0]], %[[U0]] [0] : vector<4xf32> into vector<3x4xf32>
// CHECK:      %[[T2:.*]] = vector.insert %[[T0]], %[[T1]] [1] : vector<4xf32> into vector<3x4xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T0]], %[[T2]] [2] : vector<4xf32> into vector<3x4xf32>
// CHECK:      return %[[T3]] : vector<3x4xf32>

func.func @broadcast_stretch_at_start(%arg0: vector<1x4xf32>) -> vector<3x4xf32> {
  %0 = vector.broadcast %arg0 : vector<1x4xf32> to vector<3x4xf32>
  return %0 : vector<3x4xf32>
}

// CHECK-LABEL: func @broadcast_stretch_at_end
// CHECK-SAME: %[[A:.*0]]: vector<4x1xf32>
// CHECK:      %[[U0:.*]] = ub.poison : vector<4x3xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0, 0] : f32 from vector<4x1xf32>
// CHECK:      %[[T2:.*]] = vector.splat %[[T0]] : vector<3xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[U0]] [0] : vector<3xf32> into vector<4x3xf32>
// CHECK:      %[[T4:.*]] = vector.extract %[[A]][1, 0] : f32 from vector<4x1xf32>
// CHECK:      %[[T6:.*]] = vector.splat %[[T4]] : vector<3xf32>
// CHECK:      %[[T7:.*]] = vector.insert %[[T6]], %[[T3]] [1] : vector<3xf32> into vector<4x3xf32>
// CHECK:      %[[T8:.*]] = vector.extract %[[A]][2, 0] : f32 from vector<4x1xf32>
// CHECK:      %[[T10:.*]] = vector.splat %[[T8]] : vector<3xf32>
// CHECK:      %[[T11:.*]] = vector.insert %[[T10]], %[[T7]] [2] : vector<3xf32> into vector<4x3xf32>
// CHECK:      %[[T12:.*]] = vector.extract %[[A]][3, 0] : f32 from vector<4x1xf32>
// CHECK:      %[[T14:.*]] = vector.splat %[[T12]] : vector<3xf32>
// CHECK:      %[[T15:.*]] = vector.insert %[[T14]], %[[T11]] [3] : vector<3xf32> into vector<4x3xf32>
// CHECK:      return %[[T15]] : vector<4x3xf32>

func.func @broadcast_stretch_at_end(%arg0: vector<4x1xf32>) -> vector<4x3xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1xf32> to vector<4x3xf32>
  return %0 : vector<4x3xf32>
}

// CHECK-LABEL: func @broadcast_stretch_in_middle
// CHECK-SAME: %[[A:.*0]]: vector<4x1x2xf32>
// CHECK:      %[[U0:.*]] = ub.poison : vector<4x3x2xf32>
// CHECK:      %[[U1:.*]] = ub.poison : vector<3x2xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0, 0] : vector<2xf32> from vector<4x1x2xf32>
// CHECK:      %[[T2:.*]] = vector.insert %[[T0]], %[[U1]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T0]], %[[T2]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T4:.*]] = vector.insert %[[T0]], %[[T3]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T5:.*]] = vector.insert %[[T4]], %[[U0]] [0] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T6:.*]] = vector.extract %[[A]][1, 0] : vector<2xf32> from vector<4x1x2xf32>
// CHECK:      %[[T8:.*]] = vector.insert %[[T6]], %[[U1]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T9:.*]] = vector.insert %[[T6]], %[[T8]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T10:.*]] = vector.insert %[[T6]], %[[T9]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T11:.*]] = vector.insert %[[T10]], %[[T5]] [1] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T12:.*]] = vector.extract %[[A]][2, 0] : vector<2xf32> from vector<4x1x2xf32>
// CHECK:      %[[T14:.*]] = vector.insert %[[T12]], %[[U1]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T15:.*]] = vector.insert %[[T12]], %[[T14]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T16:.*]] = vector.insert %[[T12]], %[[T15]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T17:.*]] = vector.insert %[[T16]], %[[T11]] [2] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      %[[T18:.*]] = vector.extract %[[A]][3, 0] : vector<2xf32> from vector<4x1x2xf32>
// CHECK:      %[[T20:.*]] = vector.insert %[[T18]], %[[U1]] [0] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T21:.*]] = vector.insert %[[T18]], %[[T20]] [1] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T22:.*]] = vector.insert %[[T18]], %[[T21]] [2] : vector<2xf32> into vector<3x2xf32>
// CHECK:      %[[T23:.*]] = vector.insert %[[T22]], %[[T17]] [3] : vector<3x2xf32> into vector<4x3x2xf32>
// CHECK:      return %[[T23]] : vector<4x3x2xf32>

func.func @broadcast_stretch_in_middle(%arg0: vector<4x1x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}

// CHECK-LABEL:   func.func @broadcast_scalable_duplication
// CHECK-SAME:      %[[ARG0:.*]]: vector<[32]xf32>)
// CHECK:           %[[INIT:.*]] = ub.poison : vector<1x[32]xf32>
// CHECK:           %[[RES:.*]] = vector.insert %[[ARG0]], %[[INIT]] [0] : vector<[32]xf32> into vector<1x[32]xf32>
// CHECK:           return %[[RES]] : vector<1x[32]xf32>

func.func @broadcast_scalable_duplication(%arg0: vector<[32]xf32>) -> vector<1x[32]xf32> {
  %res = vector.broadcast %arg0 : vector<[32]xf32> to vector<1x[32]xf32>
  return %res : vector<1x[32]xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_broadcast
    } : !transform.any_op
    transform.yield
  }
}
