// RUN: mlir-opt %s --test-transform-dialect-interpreter --split-input-file | FileCheck %s

#matvec_accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (j)>,
  affine_map<(i, j) -> (i)>
]
#matvec_trait = {
  indexing_maps = #matvec_accesses,
  iterator_types = ["parallel", "reduction"]
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

#matmat_accesses_0 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmat_trait_0 = {
  indexing_maps = #matmat_accesses_0,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL:   func.func @masked_extract_contract2(
// CHECK-SAME:                                      %[[VAL_0:.*]]: vector<2x3xf32>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: vector<3xf32>,
// CHECK-SAME:                                      %[[VAL_2:.*]]: vector<2xf32>,
// CHECK-SAME:                                      %[[IN_MASK:.*]]: vector<2x3xi1>) -> vector<2xf32>
// CHECK:           %[[T_MASK:.*]] = vector.transpose %[[IN_MASK]], [1, 0] : vector<2x3xi1> to vector<3x2xi1>
// CHECK:           %[[MASK0:.*]] = vector.extract %[[T_MASK]][0] : vector<3x2xi1>
// CHECK:           vector.mask %[[MASK0]] { vector.outerproduct

// CHECK:           %[[MASK1:.*]] = vector.extract %[[T_MASK]][1] : vector<3x2xi1>
// CHECK:           vector.mask %[[MASK1]] { vector.outerproduct

// CHECK:           %[[MASK2:.*]] = vector.extract %[[T_MASK]][2] : vector<3x2xi1>
// CHECK:           vector.mask %[[MASK2]] { vector.outerproduct

func.func @masked_extract_contract2(%arg0: vector<2x3xf32>,
                                    %arg1: vector<3xf32>,
                                    %arg2: vector<2xf32>,
                                    %m: vector<2x3xi1>) -> vector<2xf32> {
  %0 = vector.mask %m { vector.contract #matvec_trait %arg0, %arg1, %arg2
          : vector<2x3xf32>, vector<3xf32> into vector<2xf32> } : vector<2x3xi1> -> vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: func.func @masked_extract_contract4(
// CHECK-SAME:                                      %[[VAL_0:.*]]: vector<3x5xf32>,
// CHECK-SAME:                                      %[[VAL_1:.*]]: vector<5x7xf32>,
// CHECK-SAME:                                      %[[VAL_2:.*]]: vector<3x7xf32>,
// CHECK-SAME:                                      %[[VAL_3:.*]]: vector<3x7x5xi1>) -> vector<3x7xf32> {
// CHECK:         %[[VAL_5:.*]] = vector.transpose %[[VAL_3]], [2, 0, 1] : vector<3x7x5xi1> to vector<5x3x7xi1>
// CHECK:         %[[VAL_8:.*]] = vector.extract %[[VAL_5]][0] : vector<5x3x7xi1>
// CHECK:         %[[VAL_9:.*]] = vector.mask %[[VAL_8]] { vector.outerproduct %{{.*}} {kind = #vector.kind<add>} : vector<3xf32>, vector<7xf32> } : vector<3x7xi1> -> vector<3x7xf32>
// CHECK:         %[[VAL_12:.*]] = vector.extract %[[VAL_5]][1] : vector<5x3x7xi1>
// CHECK:         %[[VAL_13:.*]] = vector.mask %[[VAL_12]] { vector.outerproduct %{{.*}} {kind = #vector.kind<add>} : vector<3xf32>, vector<7xf32> } : vector<3x7xi1> -> vector<3x7xf32>
// CHECK:         %[[VAL_16:.*]] = vector.extract %[[VAL_5]][2] : vector<5x3x7xi1>
// CHECK:         %[[VAL_17:.*]] = vector.mask %[[VAL_16]] { vector.outerproduct %{{.*}} {kind = #vector.kind<add>} : vector<3xf32>, vector<7xf32> } : vector<3x7xi1> -> vector<3x7xf32>
// CHECK:         %[[VAL_20:.*]] = vector.extract %[[VAL_5]][3] : vector<5x3x7xi1>
// CHECK:         %[[VAL_21:.*]] = vector.mask %[[VAL_20]] { vector.outerproduct %{{.*}} {kind = #vector.kind<add>} : vector<3xf32>, vector<7xf32> } : vector<3x7xi1> -> vector<3x7xf32>
// CHECK:         %[[VAL_24:.*]] = vector.extract %[[VAL_5]][4] : vector<5x3x7xi1>
// CHECK:         %[[VAL_25:.*]] = vector.mask %[[VAL_24]] { vector.outerproduct %{{.*}} {kind = #vector.kind<add>} : vector<3xf32>, vector<7xf32> } : vector<3x7xi1> -> vector<3x7xf32>

func.func @masked_extract_contract4(%arg0: vector<3x5xf32>,
                                    %arg1: vector<5x7xf32>,
                                    %arg2: vector<3x7xf32>,
                                    %m : vector<3x7x5xi1>) -> vector<3x7xf32> {
  %0 = vector.mask %m { vector.contract #matmat_trait %arg0, %arg1, %arg2
  : vector<3x5xf32>, vector<5x7xf32> into vector<3x7xf32> } : vector<3x7x5xi1> -> vector<3x7xf32>
  return %0 : vector<3x7xf32>
}

// CHECK-LABEL: func @matmul
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x4xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<4x3xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
// CHECK-SAME:  : vector<2x4xf32> to vector<4x2xf32>
//
//      CHECK: %[[a0:.*]] = vector.extract %[[At]][0] : vector<4x2xf32>
//      CHECK: %[[b0:.*]] = vector.extract %[[B]][0] : vector<4x3xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
// CHECK-SAME:  : vector<2xf32>, vector<3xf32>
//
//      CHECK: %[[a1:.*]] = vector.extract %[[At]][1] : vector<4x2xf32>
//      CHECK: %[[b1:.*]] = vector.extract %[[B]][1] : vector<4x3xf32>
//      CHECK: %[[c1:.*]] = vector.outerproduct %[[a1]], %[[b1]], %[[c0]]
// CHECK-SAME:  : vector<2xf32>, vector<3xf32>
//
//      CHECK: %[[a2:.*]] = vector.extract %[[At]][2] : vector<4x2xf32>
//      CHECK: %[[b2:.*]] = vector.extract %[[B]][2] : vector<4x3xf32>
//      CHECK: %[[c2:.*]] = vector.outerproduct %[[a2]], %[[b2]], %[[c1]]
// CHECK-SAME:  : vector<2xf32>, vector<3xf32>
//
//      CHECK: %[[a3:.*]] = vector.extract %[[At]][3] : vector<4x2xf32>
//      CHECK: %[[b3:.*]] = vector.extract %[[B]][3] : vector<4x3xf32>
//      CHECK: %[[c3:.*]] = vector.outerproduct %[[a3]], %[[b3]], %[[c2]]
// CHECK-SAME:  : vector<2xf32>, vector<3xf32>
//
//      CHECK: return %[[c3]] : vector<2x3xf32>
func.func @matmul(%arg0: vector<2x4xf32>,
                          %arg1: vector<4x3xf32>,
                          %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %0 = vector.contract #matmat_trait %arg0, %arg1, %arg2
    : vector<2x4xf32>, vector<4x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// CHECK-LABEL: func @matmul_0
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      CHECK: %[[a0:.*]] = vector.extract %[[At]][0] : vector<1x2xf32>
//      CHECK: %[[b0:.*]] = vector.extract %[[B]][0] : vector<1x3xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
//      CHECK: return %[[c0]] : vector<2x3xf32>
func.func @matmul_0(%arg0: vector<2x1xf32>, %arg1: vector<1x3xf32>, %arg2: vector<2x3xf32>)
-> vector<2x3xf32>
{
  %0 = vector.contract #matmat_trait_0 %arg0, %arg1, %arg2
    : vector<2x1xf32>, vector<1x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// CHECK-LABEL: func @matmul_0_mixed
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf16>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf16>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      CHECK: %[[a0:.*]] = vector.extract %[[At]][0] : vector<1x2xf16>
//      CHECK: %[[b0:.*]] = vector.extract %[[B]][0] : vector<1x3xf16>
//      CHECK: %[[a1:.*]] = arith.extf %[[a0]] : vector<2xf16> to vector<2xf32>
//      CHECK: %[[b1:.*]] = arith.extf %[[b0]] : vector<3xf16> to vector<3xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a1]], %[[b1]], %[[C]]
//      CHECK: return %[[c0]] : vector<2x3xf32>
func.func @matmul_0_mixed(%arg0: vector<2x1xf16>, %arg1: vector<1x3xf16>, %arg2: vector<2x3xf32>)
-> vector<2x3xf32>
{
  %0 = vector.contract #matmat_trait_0 %arg0, %arg1, %arg2
    : vector<2x1xf16>, vector<1x3xf16> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

#matmat_accesses_1 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmat_trait_1 = {
  indexing_maps = #matmat_accesses_1,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @matmul_1
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<3x1xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      CHECK: %[[Bt:.*]] = vector.transpose %[[B]], [1, 0]
//      CHECK: %[[a0:.*]] = vector.extract %[[At]][0] : vector<1x2xf32>
//      CHECK: %[[b0:.*]] = vector.extract %[[Bt]][0] : vector<1x3xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
//      CHECK: return %[[c0]] : vector<2x3xf32>
func.func @matmul_1(%arg0: vector<2x1xf32>, %arg1: vector<3x1xf32>, %arg2: vector<2x3xf32>)
-> vector<2x3xf32>
{
  %0 = vector.contract #matmat_trait_1 %arg0, %arg1, %arg2
    : vector<2x1xf32>, vector<3x1xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

#matmat_accesses_2 = [
  affine_map<(m, n, k) -> (k, m)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmat_trait_2 = {
  indexing_maps = #matmat_accesses_2,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @matmul_2
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<1x2xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      CHECK: %[[a0:.*]] = vector.extract %[[A]][0] : vector<1x2xf32>
//      CHECK: %[[b0:.*]] = vector.extract %[[B]][0] : vector<1x3xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
//      CHECK: return %[[c0]] : vector<2x3xf32>
func.func @matmul_2(%arg0: vector<1x2xf32>, %arg1: vector<1x3xf32>, %arg2: vector<2x3xf32>)
-> vector<2x3xf32>
{
  %0 = vector.contract #matmat_trait_2 %arg0, %arg1, %arg2
    : vector<1x2xf32>, vector<1x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

#matmat_accesses_3 = [
  affine_map<(m, n, k) -> (k, m)>,
  affine_map<(m, n, k) -> (n, k)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmat_trait_3 = {
  indexing_maps = #matmat_accesses_3,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @matmul_3
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<1x2xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<3x1xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<2x3xf32>
//      CHECK: %[[Bt:.*]] = vector.transpose %[[B]], [1, 0]
//      CHECK: %[[a0:.*]] = vector.extract %[[A]][0] : vector<1x2xf32>
//      CHECK: %[[b0:.*]] = vector.extract %[[Bt]][0] : vector<1x3xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[a0]], %[[b0]], %[[C]]
//      CHECK: return %[[c0]] : vector<2x3xf32>
func.func @matmul_3(%arg0: vector<1x2xf32>, %arg1: vector<3x1xf32>, %arg2: vector<2x3xf32>)
-> vector<2x3xf32>
{
  %0 = vector.contract #matmat_trait_3 %arg0, %arg1, %arg2
    : vector<1x2xf32>, vector<3x1xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

#matmat_accesses_4 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (n, m)>
]
#matmat_trait_4 = {
  indexing_maps = #matmat_accesses_4,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @matmul_4
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<3x2xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      CHECK: %[[b0:.*]] = vector.extract %[[B]][0] : vector<1x3xf32>
//      CHECK: %[[a0:.*]] = vector.extract %[[At]][0] : vector<1x2xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[b0]], %[[a0]], %[[C]]
//      CHECK: return %[[c0]] : vector<3x2xf32>
func.func @matmul_4(%arg0: vector<2x1xf32>, %arg1: vector<1x3xf32>, %arg2: vector<3x2xf32>)
-> vector<3x2xf32>
{
  %0 = vector.contract #matmat_trait_4 %arg0, %arg1, %arg2
    : vector<2x1xf32>, vector<1x3xf32> into vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

#matmat_accesses_5 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (n, m)>
]
#matmat_trait_5 = {
  indexing_maps = #matmat_accesses_5,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @matmul_5
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<3x2xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      CHECK-DAG: %[[a0:.*]] = vector.extract %[[At]][0] : vector<1x2xf32>
//      CHECK-DAG: %[[b0:.*]] = vector.extract %[[B]][0] : vector<1x3xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[b0]], %[[a0]], %[[C]]
//      CHECK: return %[[c0]] : vector<3x2xf32>
func.func @matmul_5(%arg0: vector<2x1xf32>, %arg1: vector<1x3xf32>, %arg2: vector<3x2xf32>)
-> vector<3x2xf32>
{
  %0 = vector.contract #matmat_trait_5 %arg0, %arg1, %arg2
    : vector<2x1xf32>, vector<1x3xf32> into vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

#matmat_accesses_6 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (n, m)>
]
#matmat_trait_6 = {
  indexing_maps = #matmat_accesses_6,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @matmul_6
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<3x2xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      CHECK-DAG: %[[a0:.*]] = vector.extract %[[At]][0] : vector<1x2xf32>
//      CHECK-DAG: %[[b0:.*]] = vector.extract %[[B]][0] : vector<1x3xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[b0]], %[[a0]], %[[C]]
//      CHECK: return %[[c0]] : vector<3x2xf32>
func.func @matmul_6(%arg0: vector<2x1xf32>, %arg1: vector<1x3xf32>, %arg2: vector<3x2xf32>)
-> vector<3x2xf32>
{
  %0 = vector.contract #matmat_trait_6 %arg0, %arg1, %arg2
    : vector<2x1xf32>, vector<1x3xf32> into vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

#matmat_accesses_7 = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (n, m)>
]
#matmat_trait_7 = {
  indexing_maps = #matmat_accesses_7,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @matmul_7
// CHECK-SAME: %[[A:[a-zA-Z0-9]*]]: vector<2x1xf32>,
// CHECK-SAME: %[[B:[a-zA-Z0-9]*]]: vector<1x3xf32>,
// CHECK-SAME: %[[C:[a-zA-Z0-9]*]]: vector<3x2xf32>
//      CHECK: %[[At:.*]] = vector.transpose %[[A]], [1, 0]
//      CHECK-DAG: %[[a0:.*]] = vector.extract %[[At]][0] : vector<1x2xf32>
//      CHECK-DAG: %[[b0:.*]] = vector.extract %[[B]][0] : vector<1x3xf32>
//      CHECK: %[[c0:.*]] = vector.outerproduct %[[b0]], %[[a0]], %[[C]]
//      CHECK: return %[[c0]] : vector<3x2xf32>
func.func @matmul_7(%arg0: vector<2x1xf32>, %arg1: vector<1x3xf32>, %arg2: vector<3x2xf32>)
-> vector<3x2xf32>
{
  %0 = vector.contract #matmat_trait_7 %arg0, %arg1, %arg2
    : vector<2x1xf32>, vector<1x3xf32> into vector<3x2xf32>
  return %0 : vector<3x2xf32>
}


transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %f = transform.structured.match ops{["func.func"]} in %module_op 
    : (!pdl.operation) -> !pdl.operation

  %f2 = transform.vector.lower_contraction %f
    lowering_strategy = "outerproduct"
      : (!pdl.operation) -> !pdl.operation
}
