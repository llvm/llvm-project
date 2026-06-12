// RUN:  mlir-opt %s -transform-interpreter --split-input-file | FileCheck %s

#packed_maps = [
  affine_map<(d0, d1, d2) -> (d0, d2)>,
  affine_map<(d0, d1, d2) -> (d1, d2)>,
  affine_map<(d0, d1, d2) -> (d0, d1)>
]

// CHECK-LABEL: @test_vector_contract_to_ummla

// CHECK-SAME: %[[LHS:arg0]]: vector<4x8xi8>
// CHECK-SAME: %[[RHS:arg1]]: vector<[4]x8xi8>
// CHECK-SAME: %[[ACC:arg2]]: vector<4x[4]xi32>

// CHECK:         [[P0:[0-9]+]] = ub.poison : vector<[8]xi32>
// CHECK-NEXT:    [[P1:[0-9]+]] = ub.poison : vector<4x[4]xi32>
// CHECK-NEXT:    [[P2:[0-9]+]] = ub.poison : vector<[16]xi8>

// Extract LHS rows 0 and 1, concatenate, turn into scalable vector
// CHECK:         %[[T3:[0-9]+]] = vector.extract %[[LHS]][0] : vector<8xi8> from vector<4x8xi8>
// CHECK-NEXT:    %[[T4:[0-9]+]] = vector.extract %[[LHS]][1] : vector<8xi8> from vector<4x8xi8>
// CHECK-NEXT:    %[[T5:[0-9]+]] = vector.shuffle %[[T3]], %[[T4]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xi8>, vector<8xi8>
// CHECK-NEXT:    %[[T6:[0-9]+]] = vector.scalable.insert %[[T5]], %[[P2]][0] : vector<16xi8> into vector<[16]xi8>

// Replicate across the entire length of the scalable vector
// CHECK-NEXT:    %[[LHS_0:[0-9]+]] = arm_sve.dupq_lane %[[T6]][0] : vector<[16]xi8>

// Same for LHS rows 2 and 3
// CHECK-NEXT:    %[[T8:[0-9]+]] = vector.extract %[[LHS]][2] : vector<8xi8> from vector<4x8xi8>
// CHECK-NEXT:    %[[T9:[0-9]+]] = vector.extract %[[LHS]][3] : vector<8xi8> from vector<4x8xi8>
// CHECK-NEXT:    %[[T10:[0-9]+]] = vector.shuffle %[[T8]], %[[T9]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xi8>, vector<8xi8>
// CHECK-NEXT:    %[[T11:[0-9]+]] = vector.scalable.insert %[[T10]], %[[P2]][0] : vector<16xi8> into vector<[16]xi8>
// CHECK-NEXT:    %[[LHS_1:[0-9]+]] = arm_sve.dupq_lane %[[T11]][0] : vector<[16]xi8>

// Extract sub-tiles from the RHS
// CHECK-NEXT:    %[[T13:[0-9]+]] = vector.shape_cast %[[RHS]] : vector<[4]x8xi8> to vector<[32]xi8>
// CHECK-NEXT:    %[[RHS_0:[0-9]+]] = vector.scalable.extract %[[T13]][0] : vector<[16]xi8> from vector<[32]xi8>
// CHECK-NEXT:    %[[RHS_1:[0-9]+]] = vector.scalable.extract %[[T13]][16] : vector<[16]xi8> from vector<[32]xi8>

// Extract accumulator rows 0 and 1 and pack (into "registers")
// CHECK-NEXT:    %[[T16:[0-9]+]] = vector.extract %[[ACC]][0] : vector<[4]xi32> from vector<4x[4]xi32>
// CHECK-NEXT:    %[[T17:[0-9]+]] = vector.extract %[[ACC]][1] : vector<[4]xi32> from vector<4x[4]xi32>
// CHECK-NEXT:    %[[T18:[0-9]+]] = vector.bitcast %[[T16]] : vector<[4]xi32> to vector<[2]xi64>
// CHECK-NEXT:    %[[T19:[0-9]+]] = vector.bitcast %[[T17]] : vector<[4]xi32> to vector<[2]xi64>
// CHECK-NEXT:    %[[T20:[0-9]+]] = vector.interleave %[[T18]], %[[T19]] : vector<[2]xi64> -> vector<[4]xi64>
// CHECK-NEXT:    %[[T21:[0-9]+]] = vector.bitcast %[[T20]] : vector<[4]xi64> to vector<[8]xi32>
// CHECK-NEXT:    %[[ACC_0:[0-9]+]] = vector.scalable.extract %[[T21]][0] : vector<[4]xi32> from vector<[8]xi32>
// CHECK-NEXT:    %[[ACC_1:[0-9]+]] = vector.scalable.extract %[[T21]][4] : vector<[4]xi32> from vector<[8]xi32>

// Same for accumulator rows 2 and 3
// CHECK-NEXT:    %[[T24:[0-9]+]] = vector.extract %[[ACC]][2] : vector<[4]xi32> from vector<4x[4]xi32>
// CHECK-NEXT:    %[[T25:[0-9]+]] = vector.extract %[[ACC]][3] : vector<[4]xi32> from vector<4x[4]xi32>
// CHECK-NEXT:    %[[T26:[0-9]+]] = vector.bitcast %[[T24]] : vector<[4]xi32> to vector<[2]xi64>
// CHECK-NEXT:    %[[T27:[0-9]+]] = vector.bitcast %[[T25]] : vector<[4]xi32> to vector<[2]xi64>
// CHECK-NEXT:    %[[T28:[0-9]+]] = vector.interleave %[[T26]], %[[T27]] : vector<[2]xi64> -> vector<[4]xi64>
// CHECK-NEXT:    %[[T29:[0-9]+]] = vector.bitcast %[[T28]] : vector<[4]xi64> to vector<[8]xi32>
// CHECK-NEXT:    %[[ACC_2:[0-9]+]] = vector.scalable.extract %[[T29]][0] : vector<[4]xi32> from vector<[8]xi32>
// CHECK-NEXT:    %[[ACC_3:[0-9]+]] = vector.scalable.extract %[[T29]][4] : vector<[4]xi32> from vector<[8]xi32>

// Do the sub-tile matrix multiplications
// CHECK-NEXT:    %[[PACK_RES_00:[0-9]+]] = arm_sve.ummla %[[ACC_0]], %[[LHS_0]], %[[RHS_0]] : vector<[16]xi8> to vector<[4]xi32>
// CHECK-NEXT:    %[[PACK_RES_01:[0-9]+]] = arm_sve.ummla %[[ACC_1]], %[[LHS_0]], %[[RHS_1]] : vector<[16]xi8> to vector<[4]xi32>
// CHECK-NEXT:    %[[PACK_RES_10:[0-9]+]] = arm_sve.ummla %[[ACC_2]], %[[LHS_1]], %[[RHS_0]] : vector<[16]xi8> to vector<[4]xi32>
// CHECK-NEXT:    %[[PACK_RES_11:[0-9]+]] = arm_sve.ummla %[[ACC_3]], %[[LHS_1]], %[[RHS_1]] : vector<[16]xi8> to vector<[4]xi32>

// Unpack (from "registers") and insert in the output result rows 0 and 1
// CHECK-NEXT:    %[[T36:[0-9]+]] = vector.scalable.insert %[[PACK_RES_00]], %[[P0]][0] : vector<[4]xi32> into vector<[8]xi32>
// CHECK-NEXT:    %[[T37:[0-9]+]] = vector.scalable.insert %[[PACK_RES_01]], %[[T36]][4] : vector<[4]xi32> into vector<[8]xi32>
// CHECK-NEXT:    %[[T38:[0-9]+]] = vector.bitcast %[[T37]] : vector<[8]xi32> to vector<[4]xi64>
// CHECK-NEXT:    %res1, %res2 = vector.deinterleave %[[T38]] : vector<[4]xi64> -> vector<[2]xi64>
// CHECK-NEXT:    %[[UNPACK_RES_0:[0-9]+]] = vector.bitcast %res1 : vector<[2]xi64> to vector<[4]xi32>
// CHECK-NEXT:    %[[UNPACK_RES_1:[0-9]+]] = vector.bitcast %res2 : vector<[2]xi64> to vector<[4]xi32>
// CHECK-NEXT:    %[[TMP_OUT_0:[0-9]+]] = vector.insert %[[UNPACK_RES_0]], %[[P1]] [0] : vector<[4]xi32> into vector<4x[4]xi32>
// CHECK-NEXT:    %[[TMP_OUT_1:[0-9]+]] = vector.insert %[[UNPACK_RES_1]], %[[TMP_OUT_0]] [1] : vector<[4]xi32> into vector<4x[4]xi32>

// Same for result rows 2 and 3
// CHECK-NEXT:    %[[T43:[0-9]+]] = vector.scalable.insert %[[PACK_RES_10]], %[[P0]][0] : vector<[4]xi32> into vector<[8]xi32>
// CHECK-NEXT:    %[[T44:[0-9]+]] = vector.scalable.insert %[[PACK_RES_11]], %[[T43]][4] : vector<[4]xi32> into vector<[8]xi32>
// CHECK-NEXT:    %[[T45:[0-9]+]] = vector.bitcast %[[T44]] : vector<[8]xi32> to vector<[4]xi64>
// CHECK-NEXT:    %res1_0, %res2_1 = vector.deinterleave %[[T45]] : vector<[4]xi64> -> vector<[2]xi64>
// CHECK-NEXT:    %[[UNPACK_RES_2:[0-9]+]] = vector.bitcast %res1_0 : vector<[2]xi64> to vector<[4]xi32>
// CHECK-NEXT:    %[[UNPACK_RES_3:[0-9]+]] = vector.bitcast %res2_1 : vector<[2]xi64> to vector<[4]xi32>
// CHECK-NEXT:    %[[TMP_OUT_2:[0-9]+]] = vector.insert %[[UNPACK_RES_2]], %[[TMP_OUT_1]] [2] : vector<[4]xi32> into vector<4x[4]xi32>
// CHECK-NEXT:    %[[OUT:[0-9]+]] = vector.insert %[[UNPACK_RES_3]], %[[TMP_OUT_2]] [3] : vector<[4]xi32> into vector<4x[4]xi32>

// CHECK-NEXT:    return %[[OUT]] : vector<4x[4]xi32>

func.func @test_vector_contract_to_ummla(%lhs: vector<4x8xi8>,
              %rhs: vector<[4]x8xi8>,
              %acc: vector<4x[4]xi32>) -> vector<4x[4]xi32> {

  %0 = arith.extui %lhs : vector<4x8xi8> to vector<4x8xi32>
  %1 = arith.extui %rhs : vector<[4]x8xi8> to vector<[4]x8xi32>
  %2 = vector.contract {indexing_maps = #packed_maps,
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>} %0, %1, %acc
    : vector<4x8xi32>, vector<[4]x8xi32> into vector<4x[4]xi32>

  return %2 : vector<4x[4]xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.op<"func.func">

    transform.apply_patterns to %func {
      transform.apply_patterns.arm_sve.vector_contract_to_i8mm
    } : !transform.op<"func.func">

    transform.yield
  }
}
