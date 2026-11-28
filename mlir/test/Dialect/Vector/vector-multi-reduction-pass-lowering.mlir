// RUN: mlir-opt -lower-vector-multi-reduction="lowering-strategy=inner-reduction" -split-input-file %s | FileCheck %s --check-prefixes=ALL,INNER-REDUCTION
// RUN: mlir-opt -lower-vector-multi-reduction="lowering-strategy=inner-parallel" -split-input-file %s | FileCheck %s --check-prefixes=ALL,INNER-PARALLEL
// RUN: mlir-opt -lower-vector-multi-reduction -split-input-file %s | FileCheck %s --check-prefixes=ALL,INNER-PARALLEL

func.func @vector_multi_reduction(%arg0: vector<2x4xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}
//           ALL-LABEL: func @vector_multi_reduction
//            ALL-SAME: %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.*]]: vector<2xf32>)
// INNER-REDUCTION-DAG: %[[RESULT_VEC_0:.+]] = arith.constant dense<{{.*}}> : vector<2xf32>
//     INNER-REDUCTION: %[[V0:.+]] = vector.extract %[[INPUT]][0]
//     INNER-REDUCTION: %[[ACC0:.+]] = vector.extract %[[ACC]][0]
//     INNER-REDUCTION: %[[RV0:.+]] = vector.reduction <mul>, %[[V0]], %[[ACC0]] : vector<4xf32> into f32
//     INNER-REDUCTION: %[[RESULT_VEC_1:.+]] = vector.insert %[[RV0:.+]], %[[RESULT_VEC_0]] [0] : f32 into vector<2xf32>
//     INNER-REDUCTION: %[[V1:.+]] = vector.extract %[[INPUT]][1]
//     INNER-REDUCTION: %[[ACC1:.+]] = vector.extract %[[ACC]][1]
//     INNER-REDUCTION: %[[RV1:.+]] = vector.reduction <mul>, %[[V1]], %[[ACC1]] : vector<4xf32> into f32
//     INNER-REDUCTION: %[[RESULT_VEC:.+]] = vector.insert %[[RV1:.+]], %[[RESULT_VEC_1]] [1] : f32 into vector<2xf32>
//     INNER-REDUCTION: return %[[RESULT_VEC]]

//      INNER-PARALLEL: %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xf32> to vector<4x2xf32>
//      INNER-PARALLEL: %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<2xf32> from vector<4x2xf32>
//      INNER-PARALLEL: %[[RV0:.+]] = arith.mulf %[[V0]], %[[ACC]] : vector<2xf32>
//      INNER-PARALLEL: %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<2xf32> from vector<4x2xf32>
//      INNER-PARALLEL: %[[RV01:.+]] = arith.mulf %[[V1]], %[[RV0]] : vector<2xf32>
//      INNER-PARALLEL: %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<2xf32> from vector<4x2xf32>
//      INNER-PARALLEL: %[[RV012:.+]] = arith.mulf %[[V2]], %[[RV01]] : vector<2xf32>
//      INNER-PARALLEL: %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<2xf32> from vector<4x2xf32>
//      INNER-PARALLEL: %[[RESULT_VEC:.+]] = arith.mulf %[[V3]], %[[RV012]] : vector<2xf32>
//      INNER-PARALLEL: return %[[RESULT_VEC]] : vector<2xf32>

// -----

func.func @vector_multi_reduction_parallel_middle(%arg0: vector<3x4x5xf32>, %acc: vector<4xf32>) -> vector<4xf32> {
    %0 = vector.multi_reduction <add>, %arg0, %acc [0, 2] : vector<3x4x5xf32> to vector<4xf32>
    return %0 : vector<4xf32>
}

//       ALL-LABEL: func @vector_multi_reduction_parallel_middle
//        ALL-SAME: %[[INPUT:.+]]: vector<3x4x5xf32>, %[[ACC:.+]]: vector<4xf32>
// INNER-REDUCTION: vector.transpose %[[INPUT]], [1, 0, 2] : vector<3x4x5xf32> to vector<4x3x5xf32>
//  INNER-PARALLEL: vector.transpose %[[INPUT]], [0, 2, 1] : vector<3x4x5xf32> to vector<3x5x4xf32>

// -----

func.func @vector_multi_reduction_masked(%arg0: vector<2x4xf32>, %acc: vector<2xf32>, %mask: vector<2x4xi1>) -> vector<2xf32> {
    %0 = vector.mask %mask { vector.multi_reduction <mul>, %arg0, %acc [1] : vector<2x4xf32> to vector<2xf32> } : vector<2x4xi1> -> vector<2xf32>
    return %0 : vector<2xf32>
}

//       ALL-LABEL: func @vector_multi_reduction_masked
//        ALL-SAME: %[[INPUT:.+]]: vector<2x4xf32>, %[[ACC:.+]]: vector<2xf32>, %[[MASK:.+]]: vector<2x4xi1>
// INNER-REDUCTION: %[[INNERVEC:.+]] = vector.extract %[[INPUT]][0] : vector<4xf32> from vector<2x4xf32>
// INNER-REDUCTION: %[[INNERACC:.+]] = vector.extract %[[ACC]][0] : f32 from vector<2xf32>
// INNER-REDUCTION: %[[INNERMASK:.+]] = vector.extract %[[MASK]][0] : vector<4xi1> from vector<2x4xi1>
// INNER-REDUCTION: vector.mask %[[INNERMASK]] { vector.reduction <mul>, %[[INNERVEC]], %[[INNERACC]] : vector<4xf32> into f32 } : vector<4xi1> -> f32
//  INNER-PARALLEL: %[[TPMASK:.+]] = vector.transpose %[[MASK]], [1, 0] : vector<2x4xi1> to vector<4x2xi1>
//  INNER-PARALLEL: %[[TPINPUT:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<2x4xf32> to vector<4x2xf32>
//  INNER-PARALLEL: %[[INNERVEC:.+]] = vector.extract %[[TPINPUT]][0] : vector<2xf32> from vector<4x2xf32>
//  INNER-PARALLEL: %[[INNERMASK:.+]] = vector.extract %[[TPMASK]][0] : vector<2xi1> from vector<4x2xi1>
//  INNER-PARALLEL: %[[REDUCED:.+]] = arith.mulf %[[INNERVEC]], %[[ACC]] : vector<2xf32>
//  INNER-PARALLEL: arith.select %[[INNERMASK]], %[[REDUCED]], %[[ACC]] : vector<2xi1>, vector<2xf32>

// -----

// Reduces the fixed inner dim, the scalable outer dim is parallel.
//  * inner-reduction: TwoDimMultiReductionToReduction would have to unroll the
//    scalable outer dim, so it bails and the op is left as is.
//  * inner-parallel: the op is transposed to vector<4x[2]xf32>, and the fixed
//    outer dim is then unrolled by TwoDimMultiReductionToElementWise.
func.func @vector_multi_reduction_scalable_parallel_dim(%arg0: vector<[2]x4xf32>, %acc: vector<[2]xf32>) -> vector<[2]xf32> {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [1] : vector<[2]x4xf32> to vector<[2]xf32>
    return %0 : vector<[2]xf32>
}
//       ALL-LABEL: func @vector_multi_reduction_scalable_parallel_dim
//        ALL-SAME: %[[INPUT:.+]]: vector<[2]x4xf32>, %[[ACC:.+]]: vector<[2]xf32>
// INNER-REDUCTION: %[[RES:.+]] = vector.multi_reduction <mul>, %[[INPUT]], %[[ACC]] [1] : vector<[2]x4xf32> to vector<[2]xf32>
// INNER-REDUCTION: return %[[RES]] : vector<[2]xf32>

//  INNER-PARALLEL: %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<[2]x4xf32> to vector<4x[2]xf32>
//  INNER-PARALLEL: %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<[2]xf32> from vector<4x[2]xf32>
//  INNER-PARALLEL: %[[RV0:.+]] = arith.mulf %[[V0]], %[[ACC]] : vector<[2]xf32>
//  INNER-PARALLEL: %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<[2]xf32> from vector<4x[2]xf32>
//  INNER-PARALLEL: %[[RV01:.+]] = arith.mulf %[[V1]], %[[RV0]] : vector<[2]xf32>
//  INNER-PARALLEL: %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<[2]xf32> from vector<4x[2]xf32>
//  INNER-PARALLEL: %[[RV012:.+]] = arith.mulf %[[V2]], %[[RV01]] : vector<[2]xf32>
//  INNER-PARALLEL: %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<[2]xf32> from vector<4x[2]xf32>
//  INNER-PARALLEL: %[[RESULT_VEC:.+]] = arith.mulf %[[V3]], %[[RV012]] : vector<[2]xf32>
//  INNER-PARALLEL: return %[[RESULT_VEC]] : vector<[2]xf32>

// -----

// Reduces the scalable outer dim, the fixed inner dim is parallel.
//  * inner-reduction: the op is transposed to vector<4x[2]xf32>, and the fixed
//    outer dim is then unrolled by TwoDimMultiReductionToReduction.
//  * inner-parallel: TwoDimMultiReductionToElementWise would have to unroll the
//    scalable outer dim, so it bails and the op is left as is.
func.func @vector_multi_reduction_scalable_reduction_dim(%arg0: vector<[2]x4xf32>, %acc: vector<4xf32>) -> vector<4xf32> {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0] : vector<[2]x4xf32> to vector<4xf32>
    return %0 : vector<4xf32>
}
//       ALL-LABEL: func @vector_multi_reduction_scalable_reduction_dim
//        ALL-SAME: %[[INPUT:.+]]: vector<[2]x4xf32>, %[[ACC:.+]]: vector<4xf32>
// INNER-REDUCTION: %[[TRANSPOSED:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<[2]x4xf32> to vector<4x[2]xf32>
// INNER-REDUCTION: %[[V0:.+]] = vector.extract %[[TRANSPOSED]][0] : vector<[2]xf32> from vector<4x[2]xf32>
// INNER-REDUCTION: %[[ACC0:.+]] = vector.extract %[[ACC]][0] : f32 from vector<4xf32>
// INNER-REDUCTION: %[[RV0:.+]] = vector.reduction <mul>, %[[V0]], %[[ACC0]] : vector<[2]xf32> into f32
// INNER-REDUCTION: %[[RESULT_VEC_0:.+]] = vector.insert %[[RV0]], %{{.+}} [0] : f32 into vector<4xf32>
// INNER-REDUCTION: %[[V1:.+]] = vector.extract %[[TRANSPOSED]][1] : vector<[2]xf32> from vector<4x[2]xf32>
// INNER-REDUCTION: %[[ACC1:.+]] = vector.extract %[[ACC]][1] : f32 from vector<4xf32>
// INNER-REDUCTION: %[[RV1:.+]] = vector.reduction <mul>, %[[V1]], %[[ACC1]] : vector<[2]xf32> into f32
// INNER-REDUCTION: %[[RESULT_VEC_1:.+]] = vector.insert %[[RV1]], %[[RESULT_VEC_0]] [1] : f32 into vector<4xf32>
// INNER-REDUCTION: %[[V2:.+]] = vector.extract %[[TRANSPOSED]][2] : vector<[2]xf32> from vector<4x[2]xf32>
// INNER-REDUCTION: %[[ACC2:.+]] = vector.extract %[[ACC]][2] : f32 from vector<4xf32>
// INNER-REDUCTION: %[[RV2:.+]] = vector.reduction <mul>, %[[V2]], %[[ACC2]] : vector<[2]xf32> into f32
// INNER-REDUCTION: %[[RESULT_VEC_2:.+]] = vector.insert %[[RV2]], %[[RESULT_VEC_1]] [2] : f32 into vector<4xf32>
// INNER-REDUCTION: %[[V3:.+]] = vector.extract %[[TRANSPOSED]][3] : vector<[2]xf32> from vector<4x[2]xf32>
// INNER-REDUCTION: %[[ACC3:.+]] = vector.extract %[[ACC]][3] : f32 from vector<4xf32>
// INNER-REDUCTION: %[[RV3:.+]] = vector.reduction <mul>, %[[V3]], %[[ACC3]] : vector<[2]xf32> into f32
// INNER-REDUCTION: %[[RESULT_VEC:.+]] = vector.insert %[[RV3]], %[[RESULT_VEC_2]] [3] : f32 into vector<4xf32>
// INNER-REDUCTION: return %[[RESULT_VEC]] : vector<4xf32>

//  INNER-PARALLEL: %[[RES:.+]] = vector.multi_reduction <mul>, %[[INPUT]], %[[ACC]] [0] : vector<[2]x4xf32> to vector<4xf32>
//  INNER-PARALLEL: return %[[RES]] : vector<4xf32>

// -----

// Masked version of @vector_multi_reduction_scalable_parallel_dim.
func.func @vector_multi_reduction_scalable_parallel_dim_masked(%arg0: vector<[2]x4xf32>, %acc: vector<[2]xf32>, %mask: vector<[2]x4xi1>) -> vector<[2]xf32> {
    %0 = vector.mask %mask { vector.multi_reduction <mul>, %arg0, %acc [1] : vector<[2]x4xf32> to vector<[2]xf32> } : vector<[2]x4xi1> -> vector<[2]xf32>
    return %0 : vector<[2]xf32>
}
//       ALL-LABEL: func @vector_multi_reduction_scalable_parallel_dim_masked
//        ALL-SAME: %[[INPUT:.+]]: vector<[2]x4xf32>, %[[ACC:.+]]: vector<[2]xf32>, %[[MASK:.+]]: vector<[2]x4xi1>
// INNER-REDUCTION: %[[RES:.+]] = vector.mask %[[MASK]] { vector.multi_reduction <mul>, %[[INPUT]], %[[ACC]] [1] : vector<[2]x4xf32> to vector<[2]xf32> } : vector<[2]x4xi1> -> vector<[2]xf32>
// INNER-REDUCTION: return %[[RES]] : vector<[2]xf32>

//  INNER-PARALLEL: %[[TPMASK:.+]] = vector.transpose %[[MASK]], [1, 0] : vector<[2]x4xi1> to vector<4x[2]xi1>
//  INNER-PARALLEL: %[[TPINPUT:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<[2]x4xf32> to vector<4x[2]xf32>
//  INNER-PARALLEL: %[[INNERVEC:.+]] = vector.extract %[[TPINPUT]][0] : vector<[2]xf32> from vector<4x[2]xf32>
//  INNER-PARALLEL: %[[INNERMASK:.+]] = vector.extract %[[TPMASK]][0] : vector<[2]xi1> from vector<4x[2]xi1>
//  INNER-PARALLEL: %[[REDUCED:.+]] = arith.mulf %[[INNERVEC]], %[[ACC]] : vector<[2]xf32>
//  INNER-PARALLEL: arith.select %[[INNERMASK]], %[[REDUCED]], %[[ACC]] : vector<[2]xi1>, vector<[2]xf32>
//  INNER-PARALLEL-COUNT-2: arith.select %{{.+}}, %{{.+}}, %{{.+}} : vector<[2]xi1>, vector<[2]xf32>
//  INNER-PARALLEL: %[[RES:.+]] = arith.select %{{.+}}, %{{.+}}, %{{.+}} : vector<[2]xi1>, vector<[2]xf32>
//  INNER-PARALLEL: return %[[RES]] : vector<[2]xf32>

// -----

// Masked version of @vector_multi_reduction_scalable_reduction_dim.
func.func @vector_multi_reduction_scalable_reduction_dim_masked(%arg0: vector<[2]x4xf32>, %acc: vector<4xf32>, %mask: vector<[2]x4xi1>) -> vector<4xf32> {
    %0 = vector.mask %mask { vector.multi_reduction <mul>, %arg0, %acc [0] : vector<[2]x4xf32> to vector<4xf32> } : vector<[2]x4xi1> -> vector<4xf32>
    return %0 : vector<4xf32>
}
//       ALL-LABEL: func @vector_multi_reduction_scalable_reduction_dim_masked
//        ALL-SAME: %[[INPUT:.+]]: vector<[2]x4xf32>, %[[ACC:.+]]: vector<4xf32>, %[[MASK:.+]]: vector<[2]x4xi1>
// INNER-REDUCTION: %[[TPMASK:.+]] = vector.transpose %[[MASK]], [1, 0] : vector<[2]x4xi1> to vector<4x[2]xi1>
// INNER-REDUCTION: %[[TPINPUT:.+]] = vector.transpose %[[INPUT]], [1, 0] : vector<[2]x4xf32> to vector<4x[2]xf32>
// INNER-REDUCTION: %[[INNERVEC:.+]] = vector.extract %[[TPINPUT]][0] : vector<[2]xf32> from vector<4x[2]xf32>
// INNER-REDUCTION: %[[INNERACC:.+]] = vector.extract %[[ACC]][0] : f32 from vector<4xf32>
// INNER-REDUCTION: %[[INNERMASK:.+]] = vector.extract %[[TPMASK]][0] : vector<[2]xi1> from vector<4x[2]xi1>
// INNER-REDUCTION: vector.mask %[[INNERMASK]] { vector.reduction <mul>, %[[INNERVEC]], %[[INNERACC]] : vector<[2]xf32> into f32 } : vector<[2]xi1> -> f32
// INNER-REDUCTION-COUNT-3: vector.mask %{{.+}} { vector.reduction <mul>, %{{.+}}, %{{.+}} : vector<[2]xf32> into f32 } : vector<[2]xi1> -> f32
// INNER-REDUCTION: %[[RES:.+]] = vector.insert %{{.+}}, %{{.+}} [3] : f32 into vector<4xf32>
// INNER-REDUCTION: return %[[RES]] : vector<4xf32>

//  INNER-PARALLEL: %[[RES:.+]] = vector.mask %[[MASK]] { vector.multi_reduction <mul>, %[[INPUT]], %[[ACC]] [0] : vector<[2]x4xf32> to vector<4xf32> } : vector<[2]x4xi1> -> vector<4xf32>
//  INNER-PARALLEL: return %[[RES]] : vector<4xf32>

// -----

// Covers TwoDimMultiReductionToReduction (inner-reduction), and
// InnerOuterDimReductionConversion + TwoDimMultiReductionToElementWise
// (inner-parallel).
func.func @vector_multi_reduction_fastmath(%arg0: vector<2x4xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <maximumf>, %arg0, %acc fastmath<nnan,nsz> [1] : vector<2x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}
//              ALL-LABEL: func @vector_multi_reduction_fastmath
// INNER-REDUCTION-COUNT-2: vector.reduction <maximumf>, %{{.+}}, %{{.+}} fastmath<nnan,nsz> : vector<4xf32> into f32
//         INNER-PARALLEL: vector.transpose
//  INNER-PARALLEL-COUNT-4: arith.maximumf %{{.+}}, %{{.+}} fastmath<nnan,nsz> : vector<2xf32>

// -----

// Covers FlattenMultiReduction.
func.func @vector_multi_reduction_rank3_fastmath(%arg0: vector<2x3x4xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <add>, %arg0, %acc fastmath<reassoc> [1, 2] : vector<2x3x4xf32> to vector<2xf32>
    return %0 : vector<2xf32>
}
//               ALL-LABEL: func @vector_multi_reduction_rank3_fastmath
// INNER-REDUCTION-COUNT-2: vector.reduction <add>, %{{.+}}, %{{.+}} fastmath<reassoc> : vector<12xf32> into f32
// INNER-PARALLEL-COUNT-12: arith.addf %{{.+}}, %{{.+}} fastmath<reassoc> : vector<2xf32>

// -----

// Covers OneDimMultiReductionToReduction.
func.func @vector_multi_reduction_1d_fastmath(%arg0: vector<8xf32>, %acc: f32) -> f32 {
    %0 = vector.multi_reduction <maxnumf>, %arg0, %acc fastmath<nnan> [0] : vector<8xf32> to f32
    return %0 : f32
}
// ALL-LABEL: func @vector_multi_reduction_1d_fastmath
//       ALL: vector.reduction <maxnumf>, %{{.+}}, %{{.+}} fastmath<nnan> : vector<8xf32> into f32
