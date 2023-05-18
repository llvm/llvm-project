// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

#dotp_accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> ()>
]
#dotp_trait = {
  indexing_maps = #dotp_accesses,
  iterator_types = ["reduction"]
}

// CHECK-LABEL: func @extract_contract1
// CHECK-SAME: %[[A:.*0]]: vector<4xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<4xf32>,
// CHECK-SAME: %[[C:.*2]]: f32
// CHECK:      %[[F:.*]] = arith.mulf %[[A]], %[[B]] : vector<4xf32>
// CHECK:      %[[R:.*]] = vector.reduction <add>, %[[F]], %[[C]] : vector<4xf32> into f32
// CHECK:      return %[[R]] : f32

func.func @extract_contract1(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: f32) -> f32 {
  %0 = vector.contract #dotp_trait %arg0, %arg1, %arg2
    : vector<4xf32>, vector<4xf32> into f32
  return %0 : f32
}

// CHECK-LABEL: func @masked_extract_contract1
//  CHECK-SAME:   %[[A:.*0]]: vector<4xf32>, %[[B:.*1]]: vector<4xf32>, %[[C:.*2]]: f32
//  CHECK-SAME:   %[[M:.*]]: vector<4xi1>
//       CHECK:   %[[F:.*]] = arith.mulf %[[A]], %[[B]] : vector<4xf32>
//       CHECK:   %[[R:.*]] = vector.mask %[[M]] { vector.reduction <add>, %0, %arg2 : vector<4xf32> into f32 } : vector<4xi1> -> f32
//       CHECK:   return %[[R]] : f32

func.func @masked_extract_contract1(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: f32, %mask: vector<4xi1>) -> f32 {
  %0 = vector.mask %mask { vector.contract #dotp_trait %arg0, %arg1, %arg2 : vector<4xf32>, vector<4xf32> into f32 } : vector<4xi1> -> f32
  return %0 : f32
}

// CHECK-LABEL: func @extract_contract1_int
// CHECK-SAME: %[[A:.*0]]: vector<4xi32>,
// CHECK-SAME: %[[B:.*1]]: vector<4xi32>,
// CHECK-SAME: %[[C:.*2]]: i32
// CHECK:      %[[F:.*]] = arith.muli %[[A]], %[[B]] : vector<4xi32>
// CHECK:      %[[R:.*]] = vector.reduction <add>, %[[F]], %[[C]] : vector<4xi32> into i32
// CHECK:      return %[[R]] : i32

func.func @extract_contract1_int(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: i32) -> i32 {
  %0 = vector.contract #dotp_trait %arg0, %arg1, %arg2
    : vector<4xi32>, vector<4xi32> into i32
  return %0 : i32
}

#matvec_accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (j)>,
  affine_map<(i, j) -> (i)>
]
#matvec_trait = {
  indexing_maps = #matvec_accesses,
  iterator_types = ["parallel", "reduction"]
}

// CHECK-LABEL: func @extract_contract2
// CHECK-SAME: %[[A:.*0]]: vector<2x3xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<3xf32>,
// CHECK-SAME: %[[C:.*2]]: vector<2xf32>
// CHECK:      %[[R:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<3xf32> from vector<2x3xf32>
// CHECK:      %[[T2:.*]] = arith.mulf %[[T0]], %[[B]] : vector<3xf32>
// CHECK:      %[[T3:.*]] = vector.reduction <add>, %[[T2]] : vector<3xf32> into f32
// CHECK:      %[[T4:.*]] = vector.insert %[[T3]], %[[R]] [0] : f32 into vector<2xf32>
// CHECK:      %[[T5:.*]] = vector.extract %[[A]][1] : vector<3xf32> from vector<2x3xf32>
// CHECK:      %[[T7:.*]] = arith.mulf %[[T5]], %[[B]] : vector<3xf32>
// CHECK:      %[[T8:.*]] = vector.reduction <add>, %[[T7]] : vector<3xf32> into f32
// CHECK:      %[[T9:.*]] = vector.insert %[[T8]], %[[T4]] [1] : f32 into vector<2xf32>
// CHECK:      %[[T10:.*]] = arith.addf %[[T9]], %[[C]] : vector<2xf32>
// CHECK:      return %[[T10]] : vector<2xf32>

func.func @extract_contract2(%arg0: vector<2x3xf32>,
                        %arg1: vector<3xf32>,
                        %arg2: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #matvec_trait %arg0, %arg1, %arg2
    : vector<2x3xf32>, vector<3xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: func @extract_contract2_int
// CHECK-SAME: %[[A:.*0]]: vector<2x3xi32>,
// CHECK-SAME: %[[B:.*1]]: vector<3xi32>,
// CHECK-SAME: %[[C:.*2]]: vector<2xi32>
// CHECK:      %[[R:.*]] = arith.constant dense<0> : vector<2xi32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<3xi32> from vector<2x3xi32>
// CHECK:      %[[T2:.*]] = arith.muli %[[T0]], %[[B]] : vector<3xi32>
// CHECK:      %[[T3:.*]] = vector.reduction <add>, %[[T2]] : vector<3xi32> into i32
// CHECK:      %[[T4:.*]] = vector.insert %[[T3]], %[[R]] [0] : i32 into vector<2xi32>
// CHECK:      %[[T5:.*]] = vector.extract %[[A]][1] : vector<3xi32> from vector<2x3xi32>
// CHECK:      %[[T7:.*]] = arith.muli %[[T5]], %[[B]] : vector<3xi32>
// CHECK:      %[[T8:.*]] = vector.reduction <add>, %[[T7]] : vector<3xi32> into i32
// CHECK:      %[[T9:.*]] = vector.insert %[[T8]], %[[T4]] [1] : i32 into vector<2xi32>
// CHECK:      %[[T10:.*]] = arith.addi %[[T9]], %[[C]] : vector<2xi32>
// CHECK:      return %[[T10]] : vector<2xi32>
func.func @extract_contract2_int(%arg0: vector<2x3xi32>,
                        %arg1: vector<3xi32>,
                        %arg2: vector<2xi32>) -> vector<2xi32> {
  %0 = vector.contract #matvec_trait %arg0, %arg1, %arg2
    : vector<2x3xi32>, vector<3xi32> into vector<2xi32>
  return %0 : vector<2xi32>
}

#vecmat_accesses = [
  affine_map<(i, j) -> (j)>,
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (i)>
]
#vecmat_trait = {
  indexing_maps = #vecmat_accesses,
  iterator_types = ["parallel", "reduction"]
}

// CHECK-LABEL: func @extract_contract3
// CHECK-SAME: %[[A:.*0]]: vector<3xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<2x3xf32>,
// CHECK-SAME: %[[C:.*2]]: vector<2xf32>
// CHECK:      %[[R:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[B]][0] : vector<3xf32> from vector<2x3xf32>
// CHECK:      %[[T2:.*]] = arith.mulf %[[T0]], %[[A]] : vector<3xf32>
// CHECK:      %[[T3:.*]] = vector.reduction <add>, %[[T2]] : vector<3xf32> into f32
// CHECK:      %[[T4:.*]] = vector.insert %[[T3]], %[[R]] [0] : f32 into vector<2xf32>
// CHECK:      %[[T5:.*]] = vector.extract %[[B]][1] : vector<3xf32> from vector<2x3xf32>
// CHECK:      %[[T7:.*]] = arith.mulf %[[T5]], %[[A]] : vector<3xf32>
// CHECK:      %[[T8:.*]] = vector.reduction <add>, %[[T7]] : vector<3xf32> into f32
// CHECK:      %[[T9:.*]] = vector.insert %[[T8]], %[[T4]] [1] : f32 into vector<2xf32>
// CHECK:      %[[T10:.*]] = arith.addf %[[T9]], %[[C]] : vector<2xf32>
// CHECK:      return %[[T10]] : vector<2xf32>

func.func @extract_contract3(%arg0: vector<3xf32>,
                        %arg1: vector<2x3xf32>,
                        %arg2: vector<2xf32>) -> vector<2xf32> {
  %0 = vector.contract #vecmat_trait %arg0, %arg1, %arg2
    : vector<3xf32>, vector<2x3xf32> into vector<2xf32>
  return %0 : vector<2xf32>
}

#matmat_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#matmat_trait = {
  indexing_maps = #matmat_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @extract_contract4
// CHECK-SAME: %[[A:.*0]]: vector<2x2xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<2x2xf32>,
// CHECK-SAME: %[[C:.*2]]: vector<2x2xf32>
// CHECK:    %[[R:.*]] = arith.constant dense<0.000000e+00> : vector<2x2xf32>
// CHECK:    %[[Bt:.*]] = vector.transpose %arg1, [1, 0] : vector<2x2xf32> to vector<2x2xf32>
// CHECK:    %[[T0:.*]] = vector.extract %[[A]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK:    %[[T2:.*]] = vector.extract %[[Bt]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK:    %[[T9:.*]] = arith.mulf %[[T0]], %[[T2]] : vector<2xf32>
// CHECK:    %[[T10:.*]] = vector.reduction <add>, %[[T9]] : vector<2xf32> into f32
// CHECK:    %[[T11:.*]] = vector.insert %[[T10]], %[[R]] [0, 0] : f32 into vector<2x2xf32>
//
// CHECK:    %[[T12:.*]] = vector.extract %[[Bt]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK:    %[[T19:.*]] = arith.mulf %[[T0]], %[[T12]] : vector<2xf32>
// CHECK:    %[[T20:.*]] = vector.reduction <add>, %[[T19]] : vector<2xf32> into f32
// CHECK:    %[[T21:.*]] = vector.insert %[[T20]], %[[T11]] [0, 1] : f32 into vector<2x2xf32>
//
// CHECK:    %[[T23:.*]] = vector.extract %[[A]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK:    %[[T24:.*]] = vector.extract %[[Bt]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK:    %[[T32:.*]] = arith.mulf %[[T23]], %[[T24]] : vector<2xf32>
// CHECK:    %[[T33:.*]] = vector.reduction <add>, %[[T32]] : vector<2xf32> into f32
// CHECK:    %[[T34:.*]] = vector.insert %[[T33]], %[[T21]] [1, 0] : f32 into vector<2x2xf32>
//
// CHECK:    %[[T40:.*]] = vector.extract %[[Bt]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK:    %[[T41:.*]] = arith.mulf %[[T23]], %[[T40]] : vector<2xf32>
// CHECK:    %[[T42:.*]] = vector.reduction <add>, %[[T41]] : vector<2xf32> into f32
// CHECK:    %[[T43:.*]] = vector.insert %[[T42]], %[[T34]] [1, 1] : f32 into vector<2x2xf32>
//
// CHECK:    %[[T52:.*]] = arith.addf %[[T43]], %[[C]] : vector<2x2xf32>
// CHECK:    return %[[T52]] : vector<2x2xf32>

func.func @extract_contract4(%arg0: vector<2x2xf32>,
                        %arg1: vector<2x2xf32>,
                        %arg2: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract #matmat_trait %arg0, %arg1, %arg2
    : vector<2x2xf32>, vector<2x2xf32> into vector<2x2xf32>
  return %0 : vector<2x2xf32>
}


#contraction2d_accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> ()>
]
#contraction2d_trait = {
  indexing_maps = #contraction2d_accesses,
  iterator_types = ["reduction", "reduction"]
}

// CHECK-LABEL: func @full_contract1
// CHECK-SAME: %[[A:.*0]]: vector<2x3xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<2x3xf32>,
// CHECK-SAME: %[[C:.*2]]: f32
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<3xf32> from vector<2x3xf32>
// CHECK:      %[[T1:.*]] = vector.extract %[[B]][0] : vector<3xf32> from vector<2x3xf32>
// CHECK:      %[[T2:.*]] = arith.mulf %[[T0]], %[[T1]] : vector<3xf32>
// CHECK:      %[[T3:.*]] = vector.reduction <add>, %[[T2]], %[[C]] : vector<3xf32> into f32
// CHECK:      %[[T5:.*]] = vector.extract %[[A]][1] : vector<3xf32> from vector<2x3xf32>
// CHECK:      %[[T6:.*]] = vector.extract %[[B]][1] : vector<3xf32> from vector<2x3xf32>
// CHECK:      %[[T7:.*]] = arith.mulf %[[T5]], %[[T6]] : vector<3xf32>
// CHECK:      %[[T8:.*]] = vector.reduction <add>, %[[T7]], %[[T3]] : vector<3xf32> into f32
// CHECK:      return %[[T8]] : f32

func.func @full_contract1(%arg0: vector<2x3xf32>,
                     %arg1: vector<2x3xf32>,
                     %arg2: f32) -> f32 {
  %0 = vector.contract #contraction2d_trait %arg0, %arg1, %arg2
    : vector<2x3xf32>, vector<2x3xf32> into f32
  return %0 : f32
}

#contraction2d_trans_accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (j, i)>,
  affine_map<(i, j) -> ()>
]
#contraction2d_trans_trait = {
  indexing_maps = #contraction2d_trans_accesses,
  iterator_types = ["reduction", "reduction"]
}

// CHECK-LABEL: func @full_contract2
// CHECK-SAME: %[[A:.*0]]: vector<2x3xf32>,
// CHECK-SAME: %[[B:.*1]]: vector<3x2xf32>,
// CHECK-SAME: %[[C:.*2]]: f32
// CHECK:      %[[Z:.*]] = arith.constant dense<0.000000e+00> : vector<3xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0] : vector<3xf32> from vector<2x3xf32>
// CHECK:      %[[T1:.*]] = vector.extract %[[B]][0, 0] : f32 from vector<3x2xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T1]], %[[Z]] [0] : f32 into vector<3xf32>
// CHECK:      %[[T4:.*]] = vector.extract %[[B]][1, 0] : f32 from vector<3x2xf32>
// CHECK:      %[[T6:.*]] = vector.insert %[[T4]], %[[T3]] [1] : f32 into vector<3xf32>
// CHECK:      %[[T7:.*]] = vector.extract %[[B]][2, 0] : f32 from vector<3x2xf32>
// CHECK:      %[[T9:.*]] = vector.insert %[[T7]], %[[T6]] [2] : f32 into vector<3xf32>
// CHECK:      %[[T10:.*]] = arith.mulf %[[T0]], %[[T9]] : vector<3xf32>
// CHECK:      %[[T11:.*]] = vector.reduction <add>, %[[T10]], %[[C]] : vector<3xf32> into f32
//
// CHECK:      %[[T12:.*]] = vector.extract %[[A]][1] : vector<3xf32> from vector<2x3xf32>
// CHECK:      %[[T13:.*]] = vector.extract %[[B]][0, 1] : f32 from vector<3x2xf32>
// CHECK:      %[[T15:.*]] = vector.insert %[[T13]], %[[Z]] [0] : f32 into vector<3xf32>
// CHECK:      %[[T16:.*]] = vector.extract %[[B]][1, 1] : f32 from vector<3x2xf32>
// CHECK:      %[[T18:.*]] = vector.insert %[[T16]], %[[T15]] [1] : f32 into vector<3xf32>
// CHECK:      %[[T19:.*]] = vector.extract %[[B]][2, 1] : f32 from vector<3x2xf32>
// CHECK:      %[[T21:.*]] = vector.insert %[[T19]], %[[T18]] [2] : f32 into vector<3xf32>
// CHECK:      %[[T22:.*]] = arith.mulf %[[T12]], %[[T21]] : vector<3xf32>
// CHECK:      %[[T23:.*]] = vector.reduction <add>, %[[T22]], %[[T11]] : vector<3xf32> into f32
// CHECK:      return %[[T23]] : f32

func.func @full_contract2(%arg0: vector<2x3xf32>,
                     %arg1: vector<3x2xf32>,
                     %arg2: f32) -> f32 {
  %0 = vector.contract #contraction2d_trans_trait %arg0, %arg1, %arg2
    : vector<2x3xf32>, vector<3x2xf32> into f32
  return %0 : f32
}

// CHECK-LABEL: @contract_one_sided_unit_reduction_dim
// CHECK-SAME: (%[[A0:.+]]: vector<1x2xi32>, %[[A1:.+]]: vector<2x2xi32>, %[[A2:.+]]: vector<2xi32>)
// CHECK-DAG: %[[C:.+]] = arith.constant dense<0> : vector<2xi32>
// CHECK-DAG: %[[E00:.+]] = vector.extract %[[A0]][0] : vector<2xi32> from vector<1x2xi32>
// CHECK-DAG: %[[E10:.+]] = vector.extract %[[A1]][0] : vector<2xi32> from vector<2x2xi32>
// CHECK:     %[[M0:.+]] = arith.muli %[[E10]], %[[E00]] : vector<2xi32>
// CHECK:     %[[R0:.+]] = vector.reduction <add>, %[[M0]] : vector<2xi32> into i32
// CHECK:     %[[I0:.+]] = vector.insert %[[R0]], %[[C]] [0] : i32 into vector<2xi32>
// CHECK:     %[[E11:.+]] = vector.extract %[[A1]][1] : vector<2xi32> from vector<2x2xi32>
// CHECK:     %[[M1:.+]] = arith.muli %[[E11]], %[[E00]] : vector<2xi32>
// CHECK:     %[[R1:.+]] = vector.reduction <add>, %[[M1]] : vector<2xi32> into i32
// CHECK:     %[[I1:.+]] = vector.insert %[[R1]], %[[I0]] [1] : i32 into vector<2xi32>
// CHECK:     %[[S:.+]] = arith.addi %[[I1]], %[[A2]] : vector<2xi32>
// CHECK:     return %[[S]] : vector<2xi32>

func.func @contract_one_sided_unit_reduction_dim(%arg0 : vector<1x2xi32>, %arg1 : vector<2x2xi32>, %arg2 : vector<2xi32>) -> vector<2xi32> {
  %res = vector.contract {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d1)>
    ],
    iterator_types = ["reduction", "parallel", "reduction"],
    kind = #vector.kind<add>
  } %arg0, %arg1, %arg2 : vector<1x2xi32>, vector<2x2xi32>, vector<2xi32> into vector<2xi32>
  return %res : vector<2xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "dot"
    } : !transform.any_op
    transform.yield
  }
}
