// RUN: mlir-opt %s -cse | FileCheck %s

// Reduced from the CLIP input in llvm/llvm-project#243418, with IREE
// util.global.load adapted to ml_program.global_load. Both have a Read effect
// on the default resource without an SSA value. The indices and output buffer
// of the original position-embedding gather are exposed as function arguments.
// Four loads and this live gather are sufficient to expose overlapping scans.
// The gather must be checked once across both redundant-load queries.
// EFFECTS: CSE checking effects: ml_program.global_load
// EFFECTS-NEXT: CSE checking effects: linalg.generic

#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
ml_program.global private mutable @token : tensor<49408x768xf16>
ml_program.global private mutable @position : tensor<77x768xf16>

// CHECK-LABEL: func.func @overlapping_reads
// CHECK: %[[TOKEN:.*]] = ml_program.global_load @token
// CHECK: %[[POSITION:.*]] = ml_program.global_load @position
// CHECK-NOT: ml_program.global_load
// CHECK: %[[GATHER:.*]] = linalg.generic
// CHECK: tensor.extract %[[POSITION]]
// CHECK-NOT: ml_program.global_load
// CHECK: return %[[TOKEN]], %[[POSITION]], %[[GATHER]], %[[TOKEN]], %[[POSITION]]
func.func @overlapping_reads(%indices: tensor<64xi64>, %init: tensor<64x768xf16>) -> (tensor<49408x768xf16>, tensor<77x768xf16>, tensor<64x768xf16>, tensor<49408x768xf16>, tensor<77x768xf16>) {
  %token = ml_program.global_load @token : tensor<49408x768xf16>
  %position = ml_program.global_load @position : tensor<77x768xf16>
  %gather = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%indices : tensor<64xi64>) outs(%init : tensor<64x768xf16>) {
  ^bb0(%in: i64, %out: f16):
    %row = arith.index_cast %in : i64 to index
    %col = linalg.index 1 : index
    %extracted = tensor.extract %position[%row, %col] : tensor<77x768xf16>
    linalg.yield %extracted : f16
  } -> tensor<64x768xf16>
  %token_0 = ml_program.global_load @token : tensor<49408x768xf16>
  %position_1 = ml_program.global_load @position : tensor<77x768xf16>
  return %token, %position, %gather, %token_0, %position_1 : tensor<49408x768xf16>, tensor<77x768xf16>, tensor<64x768xf16>, tensor<49408x768xf16>, tensor<77x768xf16>
}
