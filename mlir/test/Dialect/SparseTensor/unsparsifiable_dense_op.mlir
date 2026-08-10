// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification | FileCheck %s

#trait = {
  indexing_maps = [
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, 0)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, 0)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, 0)>,
    affine_map<(d0, d1, d2, d3) -> (d3)>,
    affine_map<(d0, d1, d2, d3) -> (d3)>,
    affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
  ],
  iterator_types = ["parallel", "parallel", "parallel", "reduction"]
}

#VEC = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed), posWidth = 32, crdWidth = 32 }>
#COO = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton), posWidth = 32, crdWidth = 32 }>
#CCC = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : compressed), posWidth = 32, crdWidth = 32 }>

//
// This kernel can be sparsified as all unsparsifiable operations'
// operands are loaded from dense tensors.
//
// CHECK-LABEL: func @dense_op_without_sp_dep
// CHECK-NOT:   linalg.generic {{.*}}
func.func @dense_op_without_sp_dep(%169: tensor<2x10x8xf32>,
                                   %expanded_54: tensor<2x10x1xf32>,
                                   %expanded_56: tensor<2x10x1xf32>,
                                   %expanded_57: tensor<2x10x1xf32>,
                                   %176: tensor<8xf32, #VEC>,
                                   %177: tensor<8xf32, #VEC>,
                                   %9: tensor<100x8xf32, #COO>) ->  tensor<2x10x100xf32> {
    %cst_13 = arith.constant -3.40282347E+38 : f32
    %178 = tensor.empty() : tensor<2x10x100xf32>
    %179 = linalg.generic #trait
    ins(%169, %expanded_54, %expanded_56, %expanded_57, %176, %177, %9 :
        tensor<2x10x8xf32>, tensor<2x10x1xf32>, tensor<2x10x1xf32>, tensor<2x10x1xf32>,
        tensor<8xf32, #VEC>, tensor<8xf32, #VEC>, tensor<100x8xf32, #COO>)
    outs(%178 : tensor<2x10x100xf32>) {
    ^bb0(%in: f32, %in_58: f32, %in_59: f32, %in_60: f32, %in_61: f32, %in_62: f32, %in_63: f32, %out: f32):
      %180 = arith.mulf %in_60, %in_60 : f32
      %181 = arith.mulf %in_59, %cst_13 : f32
      %182 = arith.subf %181, %180 : f32
      %183 = arith.maximumf %182, %cst_13 : f32
      %184 = arith.addf %183, %cst_13 : f32
      %185 = math.rsqrt %184 : f32 // data dependent on sparse value.
      %186 = arith.mulf %185, %in_61 : f32
      %187 = arith.subf %in, %in_58 : f32
      %188 = arith.mulf %187, %186 : f32
      %189 = arith.addf %188, %in_62 : f32
      %190 = arith.mulf %189, %in_63 : f32
      %191 = arith.addf %out, %190 : f32
      linalg.yield %191 : f32
    } -> tensor<2x10x100xf32>
   return %179 : tensor<2x10x100xf32>
}

//
// This kernel cannot be sparsified as some unsparsifiable operations'
// operands are loaded from sparse tensors.
//
// CHECK-LABEL: func @dense_op_with_sp_dep
// CHECK:       linalg.generic {{.*}}
func.func @dense_op_with_sp_dep(%169: tensor<2x10x8xf32>,
                                %expanded_54: tensor<2x10x1xf32, #CCC>,
                                %expanded_56: tensor<2x10x1xf32, #CCC>,
                                %expanded_57: tensor<2x10x1xf32, #CCC>,
                                %176: tensor<8xf32, #VEC>,
                                %177: tensor<8xf32, #VEC>,
                                %9: tensor<100x8xf32, #COO>) ->  tensor<2x10x100xf32> {
    %cst_13 = arith.constant -3.40282347E+38 : f32
    %178 = tensor.empty() : tensor<2x10x100xf32>
    %179 = linalg.generic #trait
    ins(%169, %expanded_54, %expanded_56, %expanded_57, %176, %177, %9 :
        tensor<2x10x8xf32>, tensor<2x10x1xf32, #CCC>, tensor<2x10x1xf32, #CCC>, tensor<2x10x1xf32, #CCC>,
        tensor<8xf32, #VEC>, tensor<8xf32, #VEC>, tensor<100x8xf32, #COO>)
    outs(%178 : tensor<2x10x100xf32>) {
    ^bb0(%in: f32, %in_58: f32, %in_59: f32, %in_60: f32, %in_61: f32, %in_62: f32, %in_63: f32, %out: f32):
      %180 = arith.mulf %in_60, %in_60 : f32
      %181 = arith.mulf %in_59, %cst_13 : f32
      %182 = arith.subf %181, %180 : f32
      %183 = arith.maximumf %182, %cst_13 : f32
      %184 = arith.addf %183, %cst_13 : f32
      %185 = math.rsqrt %184 : f32
      %186 = arith.mulf %185, %in_61 : f32
      %187 = arith.subf %in, %in_58 : f32
      %188 = arith.mulf %187, %186 : f32
      %189 = arith.addf %188, %in_62 : f32
      %190 = arith.mulf %189, %in_63 : f32
      %191 = arith.addf %out, %190 : f32
      linalg.yield %191 : f32
    } -> tensor<2x10x100xf32>
   return %179 : tensor<2x10x100xf32>
}
