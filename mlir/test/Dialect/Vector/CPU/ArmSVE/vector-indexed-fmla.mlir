// RUN:  mlir-opt %s --transform-interpreter | FileCheck %s

#attrs = {
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d2, d1)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  kind = #vector.kind<add>
}

// CHECK-LABEL:   func.func @test_vector_contract_to_indexed_fmla(
// CHECK-SAME:      %[[LHS:.*]]: vector<1x4xf32>,
// CHECK-SAME:      %[[RHS:.*]]: vector<4x[8]xf32>,
// CHECK-SAME:      %[[ACC:.*]]: vector<1x[8]xf32>) -> vector<1x[8]xf32> {

/// Lanes
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 3 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : i32

/// Broadcast LHS
// CHECK:           %[[EXTRACT_0:.*]] = vector.extract %[[LHS]][0] : vector<4xf32> from vector<1x4xf32>
// CHECK:           %[[LHS_BCAST:.*]] = llvm.intr.vector.repeat(%[[EXTRACT_0]]) : (vector<4xf32>) -> vector<[4]xf32>
// CHECK:           %[[EXTRACT_1:.*]] = vector.extract %[[ACC]][0] : vector<[8]xf32> from vector<1x[8]xf32>
// CHECK:           %[[ACC_0:.*]] = vector.scalable.extract %[[EXTRACT_1]][0] : vector<[4]xf32> from vector<[8]xf32>
// CHECK:           %[[ACC_1:.*]] = vector.scalable.extract %[[EXTRACT_1]][4] : vector<[4]xf32> from vector<[8]xf32>

/// FMLA: Lane 0
// CHECK:           %[[EXTRACT_2:.*]] = vector.extract %[[RHS]][0] : vector<[8]xf32> from vector<4x[8]xf32>
// CHECK:           %[[SCALABLE_2:.*]] = vector.scalable.extract %[[EXTRACT_2]][0] : vector<[4]xf32> from vector<[8]xf32>
// CHECK:           %[[SCALABLE_3:.*]] = vector.scalable.extract %[[EXTRACT_2]][4] : vector<[4]xf32> from vector<[8]xf32>
// CHECK:           %[[FMLA_0:.*]] = arm_sve.fmla %[[ACC_0]], %[[SCALABLE_2]], %[[LHS_BCAST]], %[[CONSTANT_3]] : vector<[4]xf32> to vector<[4]xf32>
// CHECK:           %[[FMLA_1:.*]] = arm_sve.fmla %[[ACC_1]], %[[SCALABLE_3]], %[[LHS_BCAST]], %[[CONSTANT_3]] : vector<[4]xf32> to vector<[4]xf32>

/// FMLA: Lane 1
// CHECK:           %[[EXTRACT_3:.*]] = vector.extract %[[RHS]][1] : vector<[8]xf32> from vector<4x[8]xf32>
// CHECK:           %[[SCALABLE_4:.*]] = vector.scalable.extract %[[EXTRACT_3]][0] : vector<[4]xf32> from vector<[8]xf32>
// CHECK:           %[[SCALABLE_5:.*]] = vector.scalable.extract %[[EXTRACT_3]][4] : vector<[4]xf32> from vector<[8]xf32>
// CHECK:           %[[FMLA_2:.*]] = arm_sve.fmla %[[FMLA_0]], %[[SCALABLE_4]], %[[LHS_BCAST]], %[[CONSTANT_2]] : vector<[4]xf32> to vector<[4]xf32>
// CHECK:           %[[FMLA_3:.*]] = arm_sve.fmla %[[FMLA_1]], %[[SCALABLE_5]], %[[LHS_BCAST]], %[[CONSTANT_2]] : vector<[4]xf32> to vector<[4]xf32>

/// FMLA: Lane 2
// CHECK:           %[[EXTRACT_4:.*]] = vector.extract %[[RHS]][2] : vector<[8]xf32> from vector<4x[8]xf32>
// CHECK:           %[[SCALABLE_6:.*]] = vector.scalable.extract %[[EXTRACT_4]][0] : vector<[4]xf32> from vector<[8]xf32>
// CHECK:           %[[SCALABLE_7:.*]] = vector.scalable.extract %[[EXTRACT_4]][4] : vector<[4]xf32> from vector<[8]xf32>
// CHECK:           %[[FMLA_4:.*]] = arm_sve.fmla %[[FMLA_2]], %[[SCALABLE_6]], %[[LHS_BCAST]], %[[CONSTANT_1]] : vector<[4]xf32> to vector<[4]xf32>
// CHECK:           %[[FMLA_5:.*]] = arm_sve.fmla %[[FMLA_3]], %[[SCALABLE_7]], %[[LHS_BCAST]], %[[CONSTANT_1]] : vector<[4]xf32> to vector<[4]xf32>

/// FMLA: Lane 3
// CHECK:           %[[EXTRACT_5:.*]] = vector.extract %[[RHS]][3] : vector<[8]xf32> from vector<4x[8]xf32>
// CHECK:           %[[SCALABLE_8:.*]] = vector.scalable.extract %[[EXTRACT_5]][0] : vector<[4]xf32> from vector<[8]xf32>
// CHECK:           %[[SCALABLE_9:.*]] = vector.scalable.extract %[[EXTRACT_5]][4] : vector<[4]xf32> from vector<[8]xf32>
// CHECK:           %[[FMLA_6:.*]] = arm_sve.fmla %[[FMLA_4]], %[[SCALABLE_8]], %[[LHS_BCAST]], %[[CONSTANT_0]] : vector<[4]xf32> to vector<[4]xf32>
// CHECK:           %[[FMLA_7:.*]] = arm_sve.fmla %[[FMLA_5]], %[[SCALABLE_9]], %[[LHS_BCAST]], %[[CONSTANT_0]] : vector<[4]xf32> to vector<[4]xf32>

/// Store the result back
// CHECK:           %[[SCALABLE_10:.*]] = vector.scalable.insert %[[FMLA_6]], %[[EXTRACT_1]][0] : vector<[4]xf32> into vector<[8]xf32>
// CHECK:           %[[SCALABLE_11:.*]] = vector.scalable.insert %[[FMLA_7]], %[[SCALABLE_10]][4] : vector<[4]xf32> into vector<[8]xf32>
// CHECK:           %[[SHAPE_CAST_0:.*]] = vector.shape_cast %[[SCALABLE_11]] : vector<[8]xf32> to vector<1x[8]xf32>
// CHECK:           return %[[SHAPE_CAST_0]] : vector<1x[8]xf32>
func.func @test_vector_contract_to_indexed_fmla(%lhs: vector<1x4xf32>,
                                          %rhs: vector<4x[8]xf32>,
                                          %acc: vector<1x[8]xf32>) -> vector<1x[8]xf32> {
  %0 = vector.contract #attrs %lhs, %rhs, %acc
    : vector<1x4xf32>, vector<4x[8]xf32> into vector<1x[8]xf32>

  return %0 : vector<1x[8]xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.op<"func.func">

    transform.apply_patterns to %func {
      transform.apply_patterns.arm_sve.vector_contract_to_bfmmla
      transform.apply_patterns.arm_sve.vector_contract_to_indexed_fmla
    } : !transform.op<"func.func">

    transform.yield
  }
}
