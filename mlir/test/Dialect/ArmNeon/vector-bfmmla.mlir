// RUN:  mlir-opt %s --transform-interpreter | FileCheck %s

// Test lowering of vector.contract to BFMMLA operations.
// For each iteration [I, J, K] sub-tiles are extracted from offsets as follows:
//   LHS: [2*I, 4*K]
//   RHS: [2*J, 4*K]
//   ACC: [2*I, 2*J]
// Sub-tile insert offsets for the result are as like ACC (there are redundant
// inserts).

// CHECK-LABEL: func.func @vector_contract_to_bfmmla
// CHECK-SAME:    %[[LHS:.+]]: vector<4x8xbf16>, %[[RHS:.+]]: vector<4x8xbf16>, %[[ACC:.+]]: vector<4x4xf32>

// %[[INIT_RES:.+]] = arith.constant dense<0.000000e+00> : vector<4x4xf32>

// Iteration [0, 0, 0]
// Extract sib-tiles from each of LHS, RHS and ACC
// %[[T0:.+]] = vector.extract_strided_slice %[[LHS]] {offsets = [0, 0], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T1:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [0, 0], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T2:.+]] = vector.extract_strided_slice %[[ACC]] {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>

// Flatten the operands to fit the `bfmmla` operation types
// %[[T3:.+]] = vector.shape_cast %[[T0]] : vector<2x4xbf16> to vector<8xbf16>
// %[[T4:.+]] = vector.shape_cast %[[T1]] : vector<2x4xbf16> to vector<8xbf16>
// %[[T5:.+]] = vector.shape_cast %[[T2]] : vector<2x2xf32> to vector<4xf32>

// Perform the matrix multiply and accumulate
// %[[K_ACC_0:.+]] = arm_neon.intr.bfmmla %[[T5]], %[[T3]], %[[T4]] : vector<8xbf16> to vector<4xf32>

// Un-flatten the output sub-tile and inserr into the result
// %[[T7:.+]] = vector.shape_cast %[[K_ACC_0]] : vectK_ACCor<4xf32> to vector<2x2xf32>
// %[[TMP_RES_0:.+]] = vector.insert_strided_slice %[[T7]], %[[INIT_RES]] {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>

// Iteration [0, 0, 1]
// %[[T9:.+]] = vector.extract_strided_slice %[[LHS]] {offsets = [0, 4], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T10:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [0, 4], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T11:.+]] = vector.shape_cast %[[T9]] : vector<2x4xbf16> to vector<8xbf16>
// %[[T12:.+]] = vector.shape_cast %[[T1]]0 : vector<2x4xbf16> to vector<8xbf16>
// %[[T13:.+]] = arm_neon.intr.bfmmla %[[K_ACC_0]], %[[T1]]1, %[[T1]]2 : vector<8xbf16> to vector<4xf32>
// %[[T14:.+]] = vector.shape_cast %[[T1]]3 : vector<4xf32> to vector<2x2xf32>
// %[[TMP_RES_1:.+]] = vector.insert_strided_slice %[[T1]]4, %[[TMP_RES_0]] {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>

// Iteration [0, 1, 0]
// %[[T16:.+]] = vector.extract_strided_slice %[[LHS]] {offsets = [0, 0], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T17:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [2, 0], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T18:.+]] = vector.extract_strided_slice %[[ACC]] {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// %[[T19:.+]] = vector.shape_cast %[[T1]]6 : vector<2x4xbf16> to vector<8xbf16>
// %[[T20:.+]] = vector.shape_cast %[[T1]]7 : vector<2x4xbf16> to vector<8xbf16>
// %[[T21:.+]] = vector.shape_cast %[[T1]]8 : vector<2x2xf32> to vector<4xf32>
// %[[K_ACC_1:.+]] = arm_neon.intr.bfmmla %[[T2]]1, %[[T1]]9, %[[T2]]0 : vector<8xbf16> to vector<4xf32>
// %[[T23:.+]] = vector.shape_cast %[[K_ACC_1]] : vector<4xf32> to vector<2x2xf32>
// %[[TMP_RES_2:.+]] = vector.insert_strided_slice %[[T2]]3, %[[TMP_RES_1]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>

// Iteration [0, 1, 1]
// %[[T25:.+]] = vector.extract_strided_slice %[[LHS]] {offsets = [0, 4], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T26:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [2, 4], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T27:.+]] = vector.shape_cast %[[T2]]5 : vector<2x4xbf16> to vector<8xbf16>
// %[[T28:.+]] = vector.shape_cast %[[T2]]6 : vector<2x4xbf16> to vector<8xbf16>
// %[[T29:.+]] = arm_neon.intr.bfmmla %[[K_ACC_1]], %[[T2]]7, %[[T2]]8 : vector<8xbf16> to vector<4xf32>
// %[[T30:.+]] = vector.shape_cast %[[T2]]9 : vector<4xf32> to vector<2x2xf32>
// %[[TMP_RES_3:.+]] = vector.insert_strided_slice %[[T3]]0, %[[TMP_RES_2]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>

// Iteration [1, 0, 0]
// %[[T32:.+]] = vector.extract_strided_slice %[[LHS]] {offsets = [2, 0], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T33:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [0, 0], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T34:.+]] = vector.extract_strided_slice %[[ACC]] {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// %[[T35:.+]] = vector.shape_cast %[[T3]]2 : vector<2x4xbf16> to vector<8xbf16>
// %[[T36:.+]] = vector.shape_cast %[[T3]]3 : vector<2x4xbf16> to vector<8xbf16>
// %[[T37:.+]] = vector.shape_cast %[[T3]]4 : vector<2x2xf32> to vector<4xf32>
// %[[K_ACC_2:.+]] = arm_neon.intr.bfmmla %[[T3]]7, %[[T3]]5, %[[T3]]6 : vector<8xbf16> to vector<4xf32>
// %[[T39:.+]] = vector.shape_cast %[[K_ACC_2]] : vector<4xf32> to vector<2x2xf32>
//%[[TMP_RES_4:.+]] = vector.insert_strided_slice %[[T3]]9, %[[TMP_RES_3]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>

// Iteration [1, 0, 1]
// %[[T41:.+]] = vector.extract_strided_slice %[[LHS]] {offsets = [2, 4], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T42:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [0, 4], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T43:.+]] = vector.shape_cast %[[T4]]1 : vector<2x4xbf16> to vector<8xbf16>
// %[[T44:.+]] = vector.shape_cast %[[T4]]2 : vector<2x4xbf16> to vector<8xbf16>
// %[[T45:.+]] = arm_neon.intr.bfmmla %[[K_ACC_2]], %[[T4]]3, %[[T4]]4 : vector<8xbf16> to vector<4xf32>
// %[[T46:.+]] = vector.shape_cast %[[T4]]5 : vector<4xf32> to vector<2x2xf32>
//%[[TMP_RES_5:.+]] = vector.insert_strided_slice %[[T4]]6,%[[TMP_RES_4]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>

// Iteration [1, 1, 0]
// %[[T48:.+]] = vector.extract_strided_slice %[[LHS]] {offsets = [2, 0], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T49:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [2, 0], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T50:.+]] = vector.extract_strided_slice %[[ACC]] {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// %[[T51:.+]] = vector.shape_cast %[[T4]]8 : vector<2x4xbf16> to vector<8xbf16>
// %[[T52:.+]] = vector.shape_cast %[[T4]]9 : vector<2x4xbf16> to vector<8xbf16>
// %[[T53:.+]] = vector.shape_cast %[[T5]]0 : vector<2x2xf32> to vector<4xf32>
// %[[K_ACC_3:.+]] = arm_neon.intr.bfmmla %[[T5]]3, %[[T5]]1, %[[T5]]2 : vector<8xbf16> to vector<4xf32>
// %[[T55:.+]] = vector.shape_cast %[[K_ACC_3]] : vector<4xf32> to vector<2x2xf32>
//%[[TMP_RES_6:.+]] = vector.insert_strided_slice %[[T5]]5,%[[TMP_RES_5]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>

// Iteration [1, 1, 1]
// %[[T57:.+]] = vector.extract_strided_slice %[[LHS]] {offsets = [2, 4], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T58:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [2, 4], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// %[[T59:.+]] = vector.shape_cast %[[T5]]7 : vector<2x4xbf16> to vector<8xbf16>
// %[[T60:.+]] = vector.shape_cast %[[T5]]8 : vector<2x4xbf16> to vector<8xbf16>
// %[[T61:.+]] = arm_neon.intr.bfmmla %[[K_ACC_3]], %[[T5]]9, %[[T6]]0 : vector<8xbf16> to vector<4xf32>
// %[[T62:.+]] = vector.shape_cast %[[T6]]1 : vector<4xf32> to vector<2x2xf32>
// %[[RESULT:.+]] = vector.insert_strided_slice %[[T6]]2,%[[TMP_RES_6]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>

// return %[[RESULT]] : vector<4x4xf32>

func.func @vector_contract_to_bfmmla(%lhs: vector<4x8xbf16>,
                                     %rhs: vector<4x8xbf16>,
                                     %acc: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.contract { indexing_maps = [
                          affine_map<(m, n, k) -> (m, k)>,
                          affine_map<(m, n, k) -> (n, k)>,
                          affine_map<(m, n, k) -> (m, n)>
                        ],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>
                      }
    %lhs, %rhs, %acc : vector<4x8xbf16>, vector<4x8xbf16> into vector<4x4xf32>

  return %0 : vector<4x4xf32>
}

// Test lowering of vector.contract, representing vector by matrix multiply and
// accumulate, to BFMMLA operations.

// For each iteration [J, K] sub-tiles are extracted from offsets as follows:
//   LHS: [4*K]
//   RHS: [2*J, 4*K]
//   ACC: [2*J]
// Sub-tile insert offsets for the result are as like ACC (there are redundant
// inserts).
// CHECK-LABEL: func.func @vector_contract_vecmat_to_bfmmla
// CHECK-SAME:   %[[LHS:.+]]: vector<8xbf16>, %[[RHS:.+]]: vector<4x8xbf16>, %[[ACC:.+]]: vector<4xf32>) -> vector<4xf32> {
// CHECK: %[[ACC_PAD_Z:.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf32>
// CHECK: %[[LHS_PAD_Z:.+]] = arith.constant dense<0.000000e+00> : vector<2x4xbf16>
// CHECK: %[[RES_INIT:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>

// Iteration [0, 0]
// Extract sub-tiles
// CHECK: %[[T0:.+]] = vector.extract_strided_slice %[[LHS]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
// CHECK: %[[T1:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [0, 0], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// CHECK: %[[T2:.+]] = vector.extract_strided_slice %[[ACC]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>

// Pad LHS sub-tile/vector with an extra row of zeroes
// CHECK: %[[T3:.+]] = vector.insert_strided_slice %[[T0]], %[[LHS_PAD_Z]] {offsets = [0, 0], strides = [1]} : vector<4xbf16> into vector<2x4xbf16>

// Pad ACC sub-tile/vector with an extra row of zeroes
// CHECK: %[[T4:.+]] = vector.insert_strided_slice %[[T2]], %[[ACC_PAD_Z]] {offsets = [0, 0], strides = [1]} : vector<2xf32> into vector<2x2xf32>

// Flatten the operands to fit the `bfmmla` operation types
// CHECK: %[[T5:.+]] = vector.shape_cast %[[T3]] : vector<2x4xbf16> to vector<8xbf16>
// CHECK: %[[T6:.+]] = vector.shape_cast %[[T1]] : vector<2x4xbf16> to vector<8xbf16>
// CHECK: %[[T7:.+]] = vector.shape_cast %[[T4]] : vector<2x2xf32> to vector<4xf32>

// Perform the matrix multiply and accumulate
// CHECK: %[[K_ACC_0:.+]] = arm_neon.intr.bfmmla %[[T7]], %[[T5]], %[[T6]] : vector<8xbf16> to vector<4xf32>

// Un-flatten the output sub-tile
// CHECK: %[[T9:.+]] = vector.shape_cast %[[K_ACC_0]] : vector<4xf32> to vector<2x2xf32>

// Extract the first rows (the second row is padding) and insert into the result
// CHECK: %[[T10:.+]] = vector.extract %[[T9]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[TMP_RES_0:.+]] = vector.insert_strided_slice %[[T10]], %[[RES_INIT]] {offsets = [0], strides = [1]} : vector<2xf32> into vector<4xf32>

// Iteration [0, 1]
// CHECK: %[[T12:.+]] = vector.extract_strided_slice %[[LHS]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
// CHECK: %[[T13:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [0, 4], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// CHECK: %[[T14:.+]] = vector.insert_strided_slice %[[T12]], %[[LHS_PAD_Z]] {offsets = [0, 0], strides = [1]} : vector<4xbf16> into vector<2x4xbf16>
// CHECK: %[[T15:.+]] = vector.shape_cast %[[T14]] : vector<2x4xbf16> to vector<8xbf16>
// CHECK: %[[T16:.+]] = vector.shape_cast %[[T13]] : vector<2x4xbf16> to vector<8xbf16>
// CHECK: %[[T17:.+]] = arm_neon.intr.bfmmla %[[K_ACC_0]], %[[T15]], %[[T16]] : vector<8xbf16> to vector<4xf32>
// CHECK: %[[T18:.+]] = vector.shape_cast %[[T17]] : vector<4xf32> to vector<2x2xf32>
// CHECK: %[[T19:.+]] = vector.extract %[[T18]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[TMP_RES_1:.+]] = vector.insert_strided_slice %[[T19]], %[[TMP_RES_0]] {offsets = [0], strides = [1]} : vector<2xf32> into vector<4xf32>

// Iteration [1, 0]
// CHECK: %[[T21:.+]] = vector.extract_strided_slice %[[LHS]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
// CHECK: %[[T22:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [2, 0], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// CHECK: %[[T23:.+]] = vector.extract_strided_slice %[[ACC]] {offsets = [2], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
// CHECK: %[[T24:.+]] = vector.insert_strided_slice %[[T21]], %[[LHS_PAD_Z]] {offsets = [0, 0], strides = [1]} : vector<4xbf16> into vector<2x4xbf16>
// CHECK: %[[T25:.+]] = vector.insert_strided_slice %[[T23]], %[[ACC_PAD_Z]] {offsets = [0, 0], strides = [1]} : vector<2xf32> into vector<2x2xf32>
// CHECK: %[[T26:.+]] = vector.shape_cast %[[T24]] : vector<2x4xbf16> to vector<8xbf16>
// CHECK: %[[T27:.+]] = vector.shape_cast %[[T22]] : vector<2x4xbf16> to vector<8xbf16>
// CHECK: %[[T28:.+]] = vector.shape_cast %[[T25]] : vector<2x2xf32> to vector<4xf32>
// CHECK: %[[K_ACC_1:.+]] = arm_neon.intr.bfmmla %[[T28]], %[[T26]], %[[T27]] : vector<8xbf16> to vector<4xf32>
// CHECK: %[[T30:.+]] = vector.shape_cast %[[K_ACC_1]] : vector<4xf32> to vector<2x2xf32>
// CHECK: %[[T31:.+]] = vector.extract %[[T30]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[TMP_RES_2:.+]] = vector.insert_strided_slice %[[T31]], %[[TMP_RES_1]] {offsets = [2], strides = [1]} : vector<2xf32> into vector<4xf32>

// Iteration [1, 1]
// CHECK: %[[T33:.+]] = vector.extract_strided_slice %[[LHS]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
// CHECK: %[[T34:.+]] = vector.extract_strided_slice %[[RHS]] {offsets = [2, 4], sizes = [2, 4], strides = [1, 1]} : vector<4x8xbf16> to vector<2x4xbf16>
// CHECK: %[[T35:.+]] = vector.insert_strided_slice %[[T33]], %[[LHS_PAD_Z]] {offsets = [0, 0], strides = [1]} : vector<4xbf16> into vector<2x4xbf16>
// CHECK: %[[T36:.+]] = vector.shape_cast %[[T35]] : vector<2x4xbf16> to vector<8xbf16>
// CHECK: %[[T37:.+]] = vector.shape_cast %[[T34]] : vector<2x4xbf16> to vector<8xbf16>
// CHECK: %[[T38:.+]] = arm_neon.intr.bfmmla %[[K_ACC_1]], %[[T36]], %[[T37]] : vector<8xbf16> to vector<4xf32>
// CHECK: %[[T39:.+]] = vector.shape_cast %[[T38]] : vector<4xf32> to vector<2x2xf32>
// CHECK: %[[T40:.+]] = vector.extract %[[T39]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[RESULT:.+]] = vector.insert_strided_slice %[[T40]], %[[TMP_RES_2]] {offsets = [2], strides = [1]} : vector<2xf32> into vector<4xf32>
// CHECK: return %[[RESULT]] : vector<4xf32>
func.func @vector_contract_vecmat_to_bfmmla(%lhs: vector<8xbf16>,
                                            %rhs: vector<4x8xbf16>,
                                            %acc: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.contract { indexing_maps = [
                          affine_map<(n, k) -> (k)>,
                          affine_map<(n, k) -> (n, k)>,
                          affine_map<(n, k) -> (n)>
                        ],
                        iterator_types = ["parallel", "reduction"],
                        kind = #vector.kind<add>
                      }
    %lhs, %rhs, %acc : vector<8xbf16>, vector<4x8xbf16> into vector<4xf32>

  return %0 : vector<4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.op<"func.func">

    transform.apply_patterns to %func {
      transform.apply_patterns.arm_neon.vector_contract_to_bfmmla
    } : !transform.op<"func.func">

    transform.yield
  }
}
