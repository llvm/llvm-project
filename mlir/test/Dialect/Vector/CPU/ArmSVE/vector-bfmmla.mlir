// RUN:  mlir-opt %s --transform-interpreter | FileCheck %s

#attrs = {
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

// CHECK-LABEL: @test_vector_contract_to_bfmmla
// CHECK-SAME:    %[[LHS:.+]]: vector<4x4xbf16>, %[[RHS:.+]]: vector<[4]x4xbf16>, %[[ACC:.+]]: vector<4x[4]xf32>) -> vector<4x[4]xf32> {
// CHECK-NEXT:    %[[T0:.+]]  = ub.poison : vector<[8]xf32>
// CHECK-NEXT:    %[[UB:.+]] = ub.poison : vector<4x[4]xf32>
// CHECK-NEXT:    %[[T2:.+]]  = ub.poison : vector<[8]xbf16>

// Extract rows 0 and 1 of the LHS, concatenate them, and replicate the resulting 8xbf16 vector
// VSCALE times to obtain a [8]xbf16 vector.
// CHECK-NEXT:    %[[T3:.+]]  = vector.extract %[[LHS]][0] : vector<4xbf16> from vector<4x4xbf16>
// CHECK-NEXT:    %[[T4:.+]]  = vector.extract %[[LHS]][1] : vector<4xbf16> from vector<4x4xbf16>
// CHECK-NEXT:    %[[T5:.+]]  = vector.shuffle %[[T3]], %[[T4]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
// CHECK-NEXT:    %[[T6:.+]]  = vector.scalable.insert %[[T5]], %[[T2]][0] : vector<8xbf16> into vector<[8]xbf16>
// CHECK-NEXT:    %[[LHS_00:.+]] = arm_sve.dupq_lane %[[T6]][0] : vector<[8]xbf16>

// Same for rows 2 and 3 of the LHS.
// CHECK-NEXT:    %[[T8:.+]]  = vector.extract %[[LHS]][2] : vector<4xbf16> from vector<4x4xbf16>
// CHECK-NEXT:    %[[T9:.+]]  = vector.extract %[[LHS]][3] : vector<4xbf16> from vector<4x4xbf16>
// CHECK-NEXT:    %[[T10:.+]]  = vector.shuffle %[[T8]], %[[T9]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
// CHECK-NEXT:    %[[T11:.+]]  = vector.scalable.insert %[[T10]], %[[T2]][0] : vector<8xbf16> into vector<[8]xbf16>
// CHECK-NEXT:    %[[LHS_10:.+]] = arm_sve.dupq_lane %[[T11]][0] : vector<[8]xbf16>

// Extract sub-tiles from the RHS
// CHECK-NEXT:    %[[T13:.+]]  = vector.shape_cast %[[RHS]] : vector<[4]x4xbf16> to vector<[16]xbf16>
// CHECK-NEXT:    %[[RHS_00:.+]] = vector.scalable.extract %[[T13]][0] : vector<[8]xbf16> from vector<[16]xbf16>
// CHECK-NEXT:    %[[RHS_01:.+]] = vector.scalable.extract %[[T13]][8] : vector<[8]xbf16> from vector<[16]xbf16>


// Extract accumulator rows 0 and 1 and pack (into "registers")
// CHECK-NEXT:    %[[T16:.+]]  = vector.extract %[[ACC]][0] : vector<[4]xf32> from vector<4x[4]xf32>
// CHECK-NEXT:    %[[T17:.+]]  = vector.extract %[[ACC]][1] : vector<[4]xf32> from vector<4x[4]xf32>
// CHECK-NEXT:    %[[T18:.+]]  = vector.bitcast %[[T16]] : vector<[4]xf32> to vector<[2]xi64>
// CHECK-NEXT:    %[[T19:.+]]  = vector.bitcast %[[T17]] : vector<[4]xf32> to vector<[2]xi64>
// CHECK-NEXT:    %[[T20:.+]]  = vector.interleave %[[T18]], %[[T19]] : vector<[2]xi64> -> vector<[4]xi64>
// CHECK-NEXT:    %[[T21:.+]]  = vector.bitcast %[[T20]] : vector<[4]xi64> to vector<[8]xf32>
// CHECK-NEXT:    %[[ACC_00:.+]] = vector.scalable.extract %[[T21]][0] : vector<[4]xf32> from vector<[8]xf32>
// CHECK-NEXT:    %[[ACC_01:.+]] = vector.scalable.extract %[[T21]][4] : vector<[4]xf32> from vector<[8]xf32>

// Same for accumulator rows 2 and 3
// CHECK-NEXT:    %[[T24:.+]]  = vector.extract %[[ACC]][2] : vector<[4]xf32> from vector<4x[4]xf32>
// CHECK-NEXT:    %[[T25:.+]]  = vector.extract %[[ACC]][3] : vector<[4]xf32> from vector<4x[4]xf32>
// CHECK-NEXT:    %[[T26:.+]]  = vector.bitcast %[[T24]] : vector<[4]xf32> to vector<[2]xi64>
// CHECK-NEXT:    %[[T27:.+]]  = vector.bitcast %[[T25]] : vector<[4]xf32> to vector<[2]xi64>
// CHECK-NEXT:    %[[T28:.+]]  = vector.interleave %[[T26]], %[[T27]] : vector<[2]xi64> -> vector<[4]xi64>
// CHECK-NEXT:    %[[T29:.+]]  = vector.bitcast %[[T28]] : vector<[4]xi64> to vector<[8]xf32>
// CHECK-NEXT:    %[[ACC_10:.+]] = vector.scalable.extract %[[T29]][0] : vector<[4]xf32> from vector<[8]xf32>
// CHECK-NEXT:    %[[ACC_11:.+]] = vector.scalable.extract %[[T29]][4] : vector<[4]xf32> from vector<[8]xf32>

// Do the sub-tile matrix multiplications
// CHECK-NEXT:    %[[PACK_RES_00:.+]] = arm_sve.intr.bfmmla %[[ACC_00]], %[[LHS_00]], %[[RHS_00]] : vector<[8]xbf16> to vector<[4]xf32>
// CHECK-NEXT:    %[[PACK_RES_01:.+]] = arm_sve.intr.bfmmla %[[ACC_01]], %[[LHS_00]], %[[RHS_01]] : vector<[8]xbf16> to vector<[4]xf32>
// CHECK-NEXT:    %[[PACK_RES_10:.+]] = arm_sve.intr.bfmmla %[[ACC_10]], %[[LHS_10]], %[[RHS_00]] : vector<[8]xbf16> to vector<[4]xf32>
// CHECK-NEXT:    %[[PACK_RES_11:.+]] = arm_sve.intr.bfmmla %[[ACC_11]], %[[LHS_10]], %[[RHS_01]] : vector<[8]xbf16> to vector<[4]xf32>

// Unpack (from "registers") and insert in the output result rows 0 and 1
// CHECK-NEXT:    %[[T36:.+]]  = vector.scalable.insert %[[PACK_RES_00]], %[[T0]][0] : vector<[4]xf32> into vector<[8]xf32>
// CHECK-NEXT:    %[[T37:.+]]  = vector.scalable.insert %[[PACK_RES_01]], %[[T36]][4] : vector<[4]xf32> into vector<[8]xf32>
// CHECK-NEXT:    %[[T38:.+]]  = vector.bitcast %[[T37]] : vector<[8]xf32> to vector<[4]xi64>
// CHECK-NEXT:    %res1, %res2 = vector.deinterleave %[[T38]] : vector<[4]xi64> -> vector<[2]xi64>
// CHECK-NEXT:    %[[UNPACK_RES_00:.+]] = vector.bitcast %res1 : vector<[2]xi64> to vector<[4]xf32>
// CHECK-NEXT:    %[[UNPACK_RES_01:.+]] = vector.bitcast %res2 : vector<[2]xi64> to vector<[4]xf32>
// CHECK-NEXT:    %[[TMP_OUT_0:.+]] = vector.insert %[[UNPACK_RES_00]], %[[UB]] [0] : vector<[4]xf32> into vector<4x[4]xf32>
// CHECK-NEXT:    %[[TMP_OUT_1:.+]] = vector.insert %[[UNPACK_RES_01]], %[[TMP_OUT_0]] [1] : vector<[4]xf32> into vector<4x[4]xf32>

// Same for result rows 2 and 3
// CHECK-NEXT:    %[[T43:.+]]  = vector.scalable.insert %[[PACK_RES_10]], %[[T0]][0] : vector<[4]xf32> into vector<[8]xf32>
// CHECK-NEXT:    %[[T44:.+]]  = vector.scalable.insert %[[PACK_RES_11]], %[[T43]][4] : vector<[4]xf32> into vector<[8]xf32>
// CHECK-NEXT:    %[[T45:.+]]  = vector.bitcast %[[T44]] : vector<[8]xf32> to vector<[4]xi64>
// CHECK-NEXT:    %res1_0, %res2_1 = vector.deinterleave %[[T45]] : vector<[4]xi64> -> vector<[2]xi64>
// CHECK-NEXT:    %[[UNPACK_RES_10:.+]] = vector.bitcast %res1_0 : vector<[2]xi64> to vector<[4]xf32>
// CHECK-NEXT:    %[[UNPACK_RES_11:.+]] = vector.bitcast %res2_1 : vector<[2]xi64> to vector<[4]xf32>
// CHECK-NEXT:    %[[TMP_OUT_2:.+]] = vector.insert %[[UNPACK_RES_10]], %[[TMP_OUT_1]] [2] : vector<[4]xf32> into vector<4x[4]xf32>
// CHECK-NEXT:    %[[OUT:.+]] = vector.insert %[[UNPACK_RES_11]], %[[TMP_OUT_2]] [3] : vector<[4]xf32> into vector<4x[4]xf32>
// CHECK-NEXT:    return %[[OUT]] : vector<4x[4]xf32>
func.func @test_vector_contract_to_bfmmla(%lhs: vector<4x4xbf16>,
                                          %rhs: vector<[4]x4xbf16>,
                                          %acc: vector<4x[4]xf32>) -> vector<4x[4]xf32> {
  %0 = vector.contract #attrs %lhs, %rhs, %acc
    : vector<4x4xbf16>, vector<[4]x4xbf16> into vector<4x[4]xf32>

  return %0 : vector<4x[4]xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.op<"func.func">

    transform.apply_patterns to %func {
      transform.apply_patterns.arm_sve.vector_contract_to_bfmmla
    } : !transform.op<"func.func">

    transform.yield
  }
}
