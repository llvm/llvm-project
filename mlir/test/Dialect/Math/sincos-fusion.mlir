// RUN: mlir-opt -math-sincos-fusion %s | FileCheck %s

// CHECK-LABEL:   func.func @sincos_fusion(
// CHECK-SAME:      %[[ARG0:.*]]: f32,
// CHECK-SAME:      %[[ARG1:.*]]: f32) -> (f32, f32, f32, f32) {
// CHECK:           %[[VAL_0:.*]], %[[VAL_1:.*]] = math.sincos %[[ARG0]] : f32
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = math.sincos %[[ARG1]] : f32
// CHECK:           return %[[VAL_0]], %[[VAL_1]], %[[VAL_3]], %[[VAL_2]] : f32, f32, f32, f32
// CHECK:         }
func.func @sincos_fusion(%arg0 : f32, %arg1 : f32) -> (f32, f32, f32, f32) {
    %0 = math.sin %arg0 : f32
    %1 = math.cos %arg0 : f32

    %2 = math.cos %arg1 : f32
    %3 = math.sin %arg1 : f32

    func.return %0, %1, %2, %3 : f32, f32, f32, f32
}

func.func private @sink(%arg0 : f32)

// CHECK:         func.func private @sink(f32)
// CHECK-LABEL:   func.func @sincos_ensure_ssa_dominance(
// CHECK-SAME:      %[[ARG0:.*]]: f32,
// CHECK-SAME:      %[[ARG1:.*]]: f32) -> (f32, f32, f32, f32) {
// CHECK:           %[[VAL_0:.*]], %[[VAL_1:.*]] = math.sincos %[[ARG0]] : f32
// CHECK:           call @sink(%[[VAL_0]]) : (f32) -> ()
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = math.sincos %[[ARG1]] : f32
// CHECK:           call @sink(%[[VAL_3]]) : (f32) -> ()
// CHECK:           return %[[VAL_0]], %[[VAL_1]], %[[VAL_3]], %[[VAL_2]] : f32, f32, f32, f32
// CHECK:         }
func.func @sincos_ensure_ssa_dominance(%arg0 : f32, %arg1 : f32) -> (f32, f32, f32, f32) {
    %0 = math.sin %arg0 : f32
    func.call @sink(%0) : (f32) -> ()
    %1 = math.cos %arg0 : f32
    %2 = math.cos %arg1 : f32
    func.call @sink(%2) : (f32) -> ()
    %3 = math.sin %arg1 : f32
    func.return %0, %1, %2, %3 : f32, f32, f32, f32
}

// CHECK-LABEL:   func.func @sincos_fusion_no_match_fmf(
// CHECK-SAME:      %[[ARG0:.*]]: f32) -> (f32, f32) {
// CHECK:           %[[VAL_0:.*]] = math.sin %[[ARG0]] fastmath<contract> : f32
// CHECK:           %[[VAL_1:.*]] = math.cos %[[ARG0]] : f32
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : f32, f32
// CHECK:         }
func.func @sincos_fusion_no_match_fmf(%arg0 : f32) -> (f32, f32) {
    %0 = math.sin %arg0 fastmath<contract> : f32
    %1 = math.cos %arg0 : f32
    func.return %0, %1 : f32, f32
}

// CHECK-LABEL:   func.func @sincos_no_fusion_different_block(
// CHECK-SAME:      %[[ARG0:.*]]: f32,
// CHECK-SAME:      %[[ARG1:.*]]: i1) -> f32 {
// CHECK:           %[[VAL_0:.*]] = scf.if %[[ARG1]] -> (f32) {
// CHECK:             %[[VAL_1:.*]] = math.sin %[[ARG0]] : f32
// CHECK:             scf.yield %[[VAL_1]] : f32
// CHECK:           } else {
// CHECK:             %[[VAL_2:.*]] = math.cos %[[ARG0]] : f32
// CHECK:             scf.yield %[[VAL_2]] : f32
// CHECK:           }
// CHECK:           return %[[VAL_0]] : f32
// CHECK:         }
func.func @sincos_no_fusion_different_block(%arg0 : f32, %flag : i1) -> f32 {
  %0 = scf.if %flag -> f32 {
    %s = math.sin %arg0 : f32
    scf.yield %s : f32
  } else {
    %c = math.cos %arg0 : f32
    scf.yield %c : f32
  }
  func.return %0 : f32
}

// CHECK-LABEL:   func.func @sincos_fusion_preserve_fastmath(
// CHECK-SAME:      %[[ARG0:.*]]: f32) -> (f32, f32) {
// CHECK:           %[[VAL_0:.*]], %[[VAL_1:.*]] = math.sincos %[[ARG0]] fastmath<contract> : f32
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : f32, f32
// CHECK:         }
func.func @sincos_fusion_preserve_fastmath(%arg0 : f32) -> (f32, f32) {
    %0 = math.sin %arg0 fastmath<contract> : f32
    %1 = math.cos %arg0 fastmath<contract> : f32
    func.return %0, %1 : f32, f32
}
