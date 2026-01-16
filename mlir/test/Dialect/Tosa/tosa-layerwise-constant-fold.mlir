// RUN: mlir-opt --split-input-file --tosa-layerwise-constant-fold %s | FileCheck %s
// RUN: mlir-opt --split-input-file --tosa-layerwise-constant-fold="aggressive-reduce-constant=true" %s | FileCheck %s --check-prefix=AGGRESIVE

// CHECK-LABEL: @armax_fold_dim_size_1
func.func @armax_fold_dim_size_1(%arg0: tensor<2x1x3xf32>) -> tensor<2x3xi32> {
  // CHECK: "tosa.const"() <{values = dense<0> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
  %0 = tosa.argmax %arg0 {axis = 1 : i32}: (tensor<2x1x3xf32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @argmax_dynamic_shape_no_fold_dim_size_1
func.func @argmax_dynamic_shape_no_fold_dim_size_1(%arg0: tensor<?x1x3xf32>) -> tensor<?x3xi32> {
  // CHECK: tosa.argmax
  %0 = tosa.argmax %arg0 {axis = 1 : i32}: (tensor<?x1x3xf32>) -> tensor<?x3xi32>
  return %0 : tensor<?x3xi32>
}

// -----

// CHECK-LABEL: @transpose_fold
func.func @transpose_fold(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK: return %arg0
  %1 = tosa.transpose %arg0 { perms = array<i32: 0, 1> }: (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @transpose_nofold
func.func @transpose_nofold(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // CHECK: tosa.transpose
  %1 = tosa.transpose %arg0 { perms = array<i32: 1, 0> }: (tensor<3x3xf32>) -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}

// -----

// CHECK-LABEL: @transpose_nofold_shape
func.func @transpose_nofold_shape(%arg0: tensor<3x4xf32>) -> tensor<?x?xf32> {
  // CHECK: tosa.transpose
  %1 = tosa.transpose %arg0 { perms = array<i32: 1, 0> }: (tensor<3x4xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @transpose_fold_splat
func.func @transpose_fold_splat() -> tensor<3x2xf32> {
  %input = "tosa.const"() {values = dense<4.0> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  //               CHECK: %[[CST:.+]] = "tosa.const"() <{
  // CHECK-SAME{LITERAL}: values = dense<4.000000e+00> : tensor<3x2xf32>
  %1 = tosa.transpose %input { perms = array<i32: 1, 0> }: (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @transpose_fold_2d_float
func.func @transpose_fold_2d_float() -> tensor<3x2xf32> {
  %input = "tosa.const"() {values = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  //               CHECK: %[[CST:.+]] = "tosa.const"() <{
  // CHECK-SAME{LITERAL}: values = dense<[[0.000000e+00, 3.000000e+00], [1.000000e+00, 4.000000e+00], [2.000000e+00, 5.000000e+00]]> : tensor<3x2xf32>
  %1 = tosa.transpose %input { perms = array<i32: 1, 0> }: (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @transpose_fold_2d_bool
func.func @transpose_fold_2d_bool() -> tensor<3x2xi1> {
  %input = "tosa.const"() {values = dense<[[true, false, false], [false, false, true]]> : tensor<2x3xi1>} : () -> tensor<2x3xi1>
  //               CHECK: %[[CST:.+]] = "tosa.const"() <{
  // CHECK-SAME{LITERAL}: values = dense<[[true, false], [false, false], [false, true]]> : tensor<3x2xi1>
  %1 = tosa.transpose %input { perms = array<i32: 1, 0> }: (tensor<2x3xi1>) -> tensor<3x2xi1>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x2xi1>
}

// -----

// CHECK-LABEL: @transpose_fold_4d_int
func.func @transpose_fold_4d_int() -> tensor<3x1x4x2xi32> {
  %input = "tosa.const"() {values = dense<[[
    [[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11]],
    [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
  ]]> : tensor<1x2x3x4xi32>} : () -> tensor<1x2x3x4xi32>
  //               CHECK: %[[CST:.+]] = "tosa.const"() <{
  // CHECK-SAME{LITERAL}: values = dense<[
  // CHECK-SAME{LITERAL}:   [[[0, 12], [1, 13], [2, 14], [3, 15]]],
  // CHECK-SAME{LITERAL}:   [[[4, 16], [5, 17], [6, 18], [7, 19]]],
  // CHECK-SAME{LITERAL}:   [[[8, 20], [9, 21], [10, 22], [11, 23]]]
  // CHECK-SAME{LITERAL}: ]>
  %1 = tosa.transpose %input { perms = array<i32: 2, 0, 3, 1> }: (tensor<1x2x3x4xi32>) -> tensor<3x1x4x2xi32>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x1x4x2xi32>
}

// -----

// CHECK-LABEL: @transpose_nofold_non_cst_input
func.func @transpose_nofold_non_cst_input(%input: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // CHECK: tosa.transpose
  %1 = tosa.transpose %input { perms = array<i32: 1, 0> }: (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @transpose_nofold_multi_users
func.func @transpose_nofold_multi_users() -> (tensor<3x2xf32>, tensor<2x3xf32>) {
  %input = "tosa.const"() {values = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  // CHECK: tosa.transpose
  %1 = tosa.transpose %input { perms = array<i32: 1, 0> }: (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %1, %input : tensor<3x2xf32>, tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @transpose_nofold_quantized_types
func.func @transpose_nofold_quantized_types() -> tensor<1x1x2x2x!quant.uniform<i8<-127:127>:f32:3, {1.000000e-01,1.000000e-01}>> {
  %input = "tosa.const"() {values = dense<-127> : tensor<2x1x1x2xi8>} : () -> tensor<2x1x1x2x!quant.uniform<i8<-127:127>:f32:3, {1.000000e-01,1.000000e-01}>>
  // CHECK: tosa.transpose
  %0 = tosa.transpose %input { perms = array<i32: 1, 2, 3, 0> }: (tensor<2x1x1x2x!quant.uniform<i8<-127:127>:f32:3, {1.000000e-01,1.000000e-01}>>) -> tensor<1x1x2x2x!quant.uniform<i8<-127:127>:f32:3, {1.000000e-01,1.000000e-01}>>
  return %0: tensor<1x1x2x2x!quant.uniform<i8<-127:127>:f32:3, {1.000000e-01,1.000000e-01}>>
}

// -----

// CHECK-LABEL: @transpose_fold_dense_resource
func.func @transpose_fold_dense_resource() -> tensor<2x2xf32> {
  %0 = "tosa.const"() <{values = dense_resource<resource> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>

  // CHECK-NOT: tosa.transpose
  %2 = tosa.transpose %0 { perms = array<i32: 1, 0> }: (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}
{-#
  dialect_resources: {
    builtin: {
      resource: "0x040000003f800000400000004040000040800000"
    }
  }
#-}

// -----

  func.func @reduce_sum_constant() -> tensor<1x3xi32> {
    // CHECK-LABEL:   func.func @reduce_sum_constant() -> tensor<1x3xi32> {
    // CHECK:    %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}5, 7, 9]]> : tensor<1x3xi32>}> : () -> tensor<1x3xi32>
    // CHECK:         return %[[VAL_0]] : tensor<1x3xi32>

    %const = "tosa.const"() {values = dense<[[1,2,3], [4,5,6]]> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
    %0 = tosa.reduce_sum %const {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<1x3xi32>
    return %0 : tensor<1x3xi32>
  }

// -----

  func.func @reduce_sum_constant() -> tensor<2x1xi32> {
  // CHECK-LABEL:   func.func @reduce_sum_constant() -> tensor<2x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}6], [15]]> : tensor<2x1xi32>}> : () -> tensor<2x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<2x1xi32>
  // CHECK:         }
    %const = "tosa.const"() <{values = dense<[[1,2,3], [4,5,6]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    %0 = tosa.reduce_sum %const {axis = 1 : i32} : (tensor<2x3xi32>) -> tensor<2x1xi32>
    return %0 : tensor<2x1xi32>
  }


// -----

func.func @reduce_sum_constant() -> tensor<3x1xi32> {
  // CHECK-LABEL:   func.func @reduce_sum_constant() -> tensor<3x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}6], [15], [24]]> : tensor<3x1xi32>}> : () -> tensor<3x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<3x1xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>}> : () -> tensor<3x3xi32>
  %0 = tosa.reduce_sum %const {axis = 1 : i32} : (tensor<3x3xi32>) -> tensor<3x1xi32>
  return %0 : tensor<3x1xi32>
}

// -----

func.func @reduce_sum_constant() -> tensor<2x1x4xi32> {
  // CHECK-LABEL:   func.func @reduce_sum_constant() -> tensor<2x1x4xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}[15, 18, 21, 24]], {{\[\[}}51, 54, 57, 60]]]> : tensor<2x1x4xi32>}> : () -> tensor<2x1x4xi32>
  // CHECK:           return %[[VAL_0]] : tensor<2x1x4xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]> : tensor<2x3x4xi32>}> : () -> tensor<2x3x4xi32>
  %0 = tosa.reduce_sum %const {axis = 1 : i32} : (tensor<2x3x4xi32>) -> tensor<2x1x4xi32>
  return %0 : tensor<2x1x4xi32>
}

// -----

func.func @reduce_sum_constant() -> tensor<1x3x3xi32> {
  // CHECK-LABEL:   func.func @reduce_sum_constant() -> tensor<1x3x3xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}[30, 33, 36], [39, 42, 45], [48, 51, 54]]]> : tensor<1x3x3xi32>}> : () -> tensor<1x3x3xi32>
  // CHECK:           return %[[VAL_0]] : tensor<1x3x3xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]> : tensor<3x3x3xi32>}> : () -> tensor<3x3x3xi32>
  %0 = tosa.reduce_sum %const {axis = 0 : i32} : (tensor<3x3x3xi32>) -> tensor<1x3x3xi32>
  return %0 : tensor<1x3x3xi32>
}

// -----

func.func @reduce_sum_constant() -> tensor<2x2x2x1xi32> {
  // CHECK-LABEL:   func.func @reduce_sum_constant() -> tensor<2x2x2x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}{{\[\[}}3], [7]], {{\[\[}}11], [15]]], {{\[\[}}[19], [23]], {{\[\[}}27], [31]]]]> : tensor<2x2x2x1xi32>}> : () -> tensor<2x2x2x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<2x2x2x1xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]> : tensor<2x2x2x2xi32>}> : () -> tensor<2x2x2x2xi32>
  %0 = tosa.reduce_sum %const {axis = 3 : i32} : (tensor<2x2x2x2xi32>) -> tensor<2x2x2x1xi32>
  return %0 : tensor<2x2x2x1xi32>
}

// -----

func.func @reduce_sum_constant() -> tensor<1x1x1xi32> {
  // CHECK-LABEL:   func.func @reduce_sum_constant() -> tensor<1x1x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<42> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<1x1x1xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[42]]]> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
  %0 = tosa.reduce_sum %const {axis = 0 : i32} : (tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
  return %0 : tensor<1x1x1xi32>
}

// -----

func.func @reduce_sum_constant() -> tensor<2x3x1x5xi32> {
  // CHECK-LABEL:   func.func @reduce_sum_constant() -> tensor<2x3x1x5xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}{{\[\[}}34, 38, 42, 46, 50]], {{\[\[}}114, 118, 122, 126, 130]], {{\[\[}}194, 198, 202, 206, 210]]], {{\[\[}}[274, 278, 282, 286, 290]], {{\[\[}}354, 358, 362, 366, 370]], {{\[\[}}434, 438, 442, 446, 450]]]]> : tensor<2x3x1x5xi32>}> : () -> tensor<2x3x1x5xi32>
  // CHECK:           return %[[VAL_0]] : tensor<2x3x1x5xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]], [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30], [31, 32, 33, 34, 35], [36, 37, 38, 39, 40]], [[41, 42, 43, 44, 45], [46, 47, 48, 49, 50], [51, 52, 53, 54, 55], [56, 57, 58, 59, 60]]], [[[61, 62, 63, 64, 65], [66, 67, 68, 69, 70], [71, 72, 73, 74, 75], [76, 77, 78, 79, 80]], [[81, 82, 83, 84, 85], [86, 87, 88, 89, 90], [91, 92, 93, 94, 95], [96, 97, 98, 99, 100]], [[101, 102, 103, 104, 105], [106, 107, 108, 109, 110], [111, 112, 113, 114, 115], [116, 117, 118, 119, 120]]]]> : tensor<2x3x4x5xi32>}> : () -> tensor<2x3x4x5xi32>
  %0 = tosa.reduce_sum %const {axis = 2 : i32} : (tensor<2x3x4x5xi32>) -> tensor<2x3x1x5xi32>
  return %0 : tensor<2x3x1x5xi32>
}

// -----

  func.func @reduce_prod_constant() -> tensor<1x3xi32> {
    // CHECK-LABEL:   func.func @reduce_prod_constant() -> tensor<1x3xi32> {
    // CHECK:    %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}4, 10, 18]]> : tensor<1x3xi32>}> : () -> tensor<1x3xi32>
    // CHECK:         return %[[VAL_0]] : tensor<1x3xi32>

    %const = "tosa.const"() <{values = dense<[[1,2,3], [4,5,6]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    %0 = tosa.reduce_product %const {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<1x3xi32>
    return %0 : tensor<1x3xi32>
  }

// -----

  func.func @reduce_prod_constant() -> tensor<2x1xi32> {
  // CHECK-LABEL:   func.func @reduce_prod_constant() -> tensor<2x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}6], [120]]> : tensor<2x1xi32>}> : () -> tensor<2x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<2x1xi32>
  // CHECK:         }

    %const = "tosa.const"() <{values = dense<[[1,2,3], [4,5,6]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    %0 = tosa.reduce_product %const {axis = 1 : i32} : (tensor<2x3xi32>) -> tensor<2x1xi32>
    return %0 : tensor<2x1xi32>
  }

// -----

func.func @reduce_prod_constant() -> tensor<3x1xi32> {
  // CHECK-LABEL:   func.func @reduce_prod_constant() -> tensor<3x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}6], [120], [504]]> : tensor<3x1xi32>}> : () -> tensor<3x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<3x1xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>}> : () -> tensor<3x3xi32>
  %0 = tosa.reduce_product %const {axis = 1 : i32} : (tensor<3x3xi32>) -> tensor<3x1xi32>
  return %0 : tensor<3x1xi32>
}

// -----

func.func @reduce_prod_constant() -> tensor<2x1x4xi32> {
  // CHECK-LABEL:   func.func @reduce_prod_constant() -> tensor<2x1x4xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}[45, 120, 231, 384]], {{\[\[}}4641, 5544, 6555, 7680]]]> : tensor<2x1x4xi32>}> : () -> tensor<2x1x4xi32>
  // CHECK:           return %[[VAL_0]] : tensor<2x1x4xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]> : tensor<2x3x4xi32>}> : () -> tensor<2x3x4xi32>
  %0 = tosa.reduce_product %const {axis = 1 : i32} : (tensor<2x3x4xi32>) -> tensor<2x1x4xi32>
  return %0 : tensor<2x1x4xi32>
}

// -----

func.func @reduce_prod_constant() -> tensor<1x3x3xi32> {
  // CHECK-LABEL:   func.func @reduce_prod_constant() -> tensor<1x3x3xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}[190, 440, 756], [1144, 1610, 2160], [2800, 3536, 4374]]]> : tensor<1x3x3xi32>}> : () -> tensor<1x3x3xi32>
  // CHECK:           return %[[VAL_0]] : tensor<1x3x3xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]> : tensor<3x3x3xi32>}> : () -> tensor<3x3x3xi32>
  %0 = tosa.reduce_product %const {axis = 0 : i32} : (tensor<3x3x3xi32>) -> tensor<1x3x3xi32>
  return %0 : tensor<1x3x3xi32>
}

// -----

func.func @reduce_prod_constant() -> tensor<2x2x2x1xi32> {
  // CHECK-LABEL:   func.func @reduce_prod_constant() -> tensor<2x2x2x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}{{\[\[}}2], [12]], {{\[\[}}30], [56]]], {{\[\[}}[90], [132]], {{\[\[}}182], [240]]]]> : tensor<2x2x2x1xi32>}> : () -> tensor<2x2x2x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<2x2x2x1xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]> : tensor<2x2x2x2xi32>}> : () -> tensor<2x2x2x2xi32>
  %0 = tosa.reduce_product %const {axis = 3 : i32} : (tensor<2x2x2x2xi32>) -> tensor<2x2x2x1xi32>
  return %0 : tensor<2x2x2x1xi32>
}

// -----

func.func @reduce_prod_constant() -> tensor<1x1x1xi32> {
  // CHECK-LABEL:   func.func @reduce_prod_constant() -> tensor<1x1x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<42> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<1x1x1xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[42]]]> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
  %0 = tosa.reduce_product %const {axis = 0 : i32} : (tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
  return %0 : tensor<1x1x1xi32>
}

// -----

  func.func @reduce_max_constant() -> tensor<1x3xi32> {
    // CHECK-LABEL:   func.func @reduce_max_constant() -> tensor<1x3xi32> {
    // CHECK:    %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}4, 5, 6]]> : tensor<1x3xi32>}> : () -> tensor<1x3xi32>
    // CHECK:         return %[[VAL_0]] : tensor<1x3xi32>

    %const = "tosa.const"() <{values = dense<[[1,2,3], [4,5,6]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    %0 = tosa.reduce_max %const {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<1x3xi32>
    return %0 : tensor<1x3xi32>
  }

// -----

  func.func @reduce_max_constant() -> tensor<2x1xi32> {
  // CHECK-LABEL:   func.func @reduce_max_constant() -> tensor<2x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}3], [6]]> : tensor<2x1xi32>}> : () -> tensor<2x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<2x1xi32>
  // CHECK:         }

    %const = "tosa.const"() <{values = dense<[[1,2,3], [4,5,6]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    %0 = tosa.reduce_max %const {axis = 1 : i32} : (tensor<2x3xi32>) -> tensor<2x1xi32>
    return %0 : tensor<2x1xi32>
  }

// -----

func.func @reduce_max_constant() -> tensor<3x1xi32> {
  // CHECK-LABEL:   func.func @reduce_max_constant() -> tensor<3x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}3], [6], [9]]> : tensor<3x1xi32>}> : () -> tensor<3x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<3x1xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>}> : () -> tensor<3x3xi32>
  %0 = tosa.reduce_max %const {axis = 1 : i32} : (tensor<3x3xi32>) -> tensor<3x1xi32>
  return %0 : tensor<3x1xi32>
}

// -----

func.func @reduce_max_constant() -> tensor<2x1x4xi32> {
  // CHECK-LABEL:   func.func @reduce_max_constant() -> tensor<2x1x4xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}[9, 10, 11, 12]], {{\[\[}}21, 22, 23, 24]]]> : tensor<2x1x4xi32>}> : () -> tensor<2x1x4xi32>
  // CHECK:           return %[[VAL_0]] : tensor<2x1x4xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]> : tensor<2x3x4xi32>}> : () -> tensor<2x3x4xi32>
  %0 = tosa.reduce_max %const {axis = 1 : i32} : (tensor<2x3x4xi32>) -> tensor<2x1x4xi32>
  return %0 : tensor<2x1x4xi32>
}

// -----

func.func @reduce_max_constant() -> tensor<1x3x3xi32> {
  // CHECK-LABEL:   func.func @reduce_max_constant() -> tensor<1x3x3xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}[19, 20, 21], [22, 23, 24], [25, 26, 27]]]> : tensor<1x3x3xi32>}> : () -> tensor<1x3x3xi32>
  // CHECK:           return %[[VAL_0]] : tensor<1x3x3xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]> : tensor<3x3x3xi32>}> : () -> tensor<3x3x3xi32>
  %0 = tosa.reduce_max %const {axis = 0 : i32} : (tensor<3x3x3xi32>) -> tensor<1x3x3xi32>
  return %0 : tensor<1x3x3xi32>
}

// -----

func.func @reduce_max_constant() -> tensor<2x2x2x1xi32> {
  // CHECK-LABEL:   func.func @reduce_max_constant() -> tensor<2x2x2x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}{{\[\[}}2], [4]], {{\[\[}}6], [8]]], {{\[\[}}[10], [12]], {{\[\[}}14], [16]]]]> : tensor<2x2x2x1xi32>}> : () -> tensor<2x2x2x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<2x2x2x1xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]> : tensor<2x2x2x2xi32>}> : () -> tensor<2x2x2x2xi32>
  %0 = tosa.reduce_max %const {axis = 3 : i32} : (tensor<2x2x2x2xi32>) -> tensor<2x2x2x1xi32>
  return %0 : tensor<2x2x2x1xi32>
}

// -----

func.func @reduce_max_constant() -> tensor<1x1x1xi32> {
  // CHECK-LABEL:   func.func @reduce_max_constant() -> tensor<1x1x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<42> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<1x1x1xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[42]]]> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
  %0 = tosa.reduce_max %const {axis = 0 : i32} : (tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
  return %0 : tensor<1x1x1xi32>
}

// -----

func.func @reduce_max_constant_no_overflow() -> tensor<1xi8> {
  // CHECK-LABEL:   func.func @reduce_max_constant_no_overflow() -> tensor<1xi8> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<120> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK:           return %[[VAL_0]] : tensor<1xi8>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[-127, 120, -126]> : tensor<3xi8>}> : () -> tensor<3xi8>
  %0 = tosa.reduce_max %const {axis = 0 : i32} : (tensor<3xi8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// -----

  func.func @reduce_min_constant() -> tensor<1x3xi32> {
    // CHECK-LABEL:   func.func @reduce_min_constant() -> tensor<1x3xi32> {
    // CHECK:    %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}1, 2, 3]]> : tensor<1x3xi32>}> : () -> tensor<1x3xi32>
    // CHECK:         return %[[VAL_0]] : tensor<1x3xi32>
    %const = "tosa.const"() <{values = dense<[[1,2,3], [4,5,6]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    %0 = tosa.reduce_min %const {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<1x3xi32>
    return %0 : tensor<1x3xi32>
  }


// -----

  func.func @reduce_min_constant() -> tensor<2x1xi32> {
  // CHECK-LABEL:   func.func @reduce_min_constant() -> tensor<2x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}1], [4]]> : tensor<2x1xi32>}> : () -> tensor<2x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<2x1xi32>
  // CHECK:         }

    %const = "tosa.const"() <{values = dense<[[1,2,3], [4,5,6]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    %0 = tosa.reduce_min %const {axis = 1 : i32} : (tensor<2x3xi32>) -> tensor<2x1xi32>
    return %0 : tensor<2x1xi32>
  }

// -----

func.func @reduce_min_constant() -> tensor<3x1xi32> {
  // CHECK-LABEL:   func.func @reduce_min_constant() -> tensor<3x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}1], [4], [7]]> : tensor<3x1xi32>}> : () -> tensor<3x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<3x1xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>}> : () -> tensor<3x3xi32>
  %0 = tosa.reduce_min %const {axis = 1 : i32} : (tensor<3x3xi32>) -> tensor<3x1xi32>
  return %0 : tensor<3x1xi32>
}

// -----

func.func @reduce_min_constant() -> tensor<2x1x4xi32> {
  // CHECK-LABEL:   func.func @reduce_min_constant() -> tensor<2x1x4xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}[1, 2, 3, 4]], {{\[\[}}13, 14, 15, 16]]]> : tensor<2x1x4xi32>}> : () -> tensor<2x1x4xi32>
  // CHECK:           return %[[VAL_0]] : tensor<2x1x4xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]> : tensor<2x3x4xi32>}> : () -> tensor<2x3x4xi32>
  %0 = tosa.reduce_min %const {axis = 1 : i32} : (tensor<2x3x4xi32>) -> tensor<2x1x4xi32>
  return %0 : tensor<2x1x4xi32>
}

// -----

func.func @reduce_min_constant() -> tensor<1x3x3xi32> {
  // CHECK-LABEL:   func.func @reduce_min_constant() -> tensor<1x3x3xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}[1, 2, 3], [4, 5, 6], [7, 8, 9]]]> : tensor<1x3x3xi32>}> : () -> tensor<1x3x3xi32>
  // CHECK:           return %[[VAL_0]] : tensor<1x3x3xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]> : tensor<3x3x3xi32>}> : () -> tensor<3x3x3xi32>
  %0 = tosa.reduce_min %const {axis = 0 : i32} : (tensor<3x3x3xi32>) -> tensor<1x3x3xi32>
  return %0 : tensor<1x3x3xi32>
}

// -----

func.func @reduce_min_constant() -> tensor<2x2x2x1xi32> {
  // CHECK-LABEL:   func.func @reduce_min_constant() -> tensor<2x2x2x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}{{\[\[}}1], [3]], {{\[\[}}5], [7]]], {{\[\[}}[9], [11]], {{\[\[}}13], [15]]]]> : tensor<2x2x2x1xi32>}> : () -> tensor<2x2x2x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<2x2x2x1xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]> : tensor<2x2x2x2xi32>}> : () -> tensor<2x2x2x2xi32>
  %0 = tosa.reduce_min %const {axis = 3 : i32} : (tensor<2x2x2x2xi32>) -> tensor<2x2x2x1xi32>
  return %0 : tensor<2x2x2x1xi32>
}

// -----

func.func @reduce_min_constant() -> tensor<1x1x1xi32> {
  // CHECK-LABEL:   func.func @reduce_min_constant() -> tensor<1x1x1xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<42> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
  // CHECK:           return %[[VAL_0]] : tensor<1x1x1xi32>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[42]]]> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
  %0 = tosa.reduce_min %const {axis = 0 : i32} : (tensor<1x1x1xi32>) -> tensor<1x1x1xi32>
  return %0 : tensor<1x1x1xi32>
}

// -----

func.func @reduce_min_constant_no_overflow() -> tensor<1xi8> {
  // CHECK-LABEL:   func.func @reduce_min_constant_no_overflow() -> tensor<1xi8> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<-127> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK:           return %[[VAL_0]] : tensor<1xi8>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[-127, 120, -126]> : tensor<3xi8>}> : () -> tensor<3xi8>
  %0 = tosa.reduce_min %const {axis = 0 : i32} : (tensor<3xi8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}


// -----

func.func @reduce_any_constant() -> tensor<1x3xi1> {
  // CHECK-LABEL:   func.func @reduce_any_constant() -> tensor<1x3xi1> {
  // CHECK:    %[[VAL_0:.*]] = "tosa.const"() <{values = dense<true> : tensor<1x3xi1>}> : () -> tensor<1x3xi1>
  // CHECK:         return %[[VAL_0]] : tensor<1x3xi1>

  %const = "tosa.const"() <{values = dense<[[true,true,true], [true,false,true]]> : tensor<2x3xi1>}> : () -> tensor<2x3xi1>
  %0 = tosa.reduce_any %const {axis = 0 : i32} : (tensor<2x3xi1>) -> tensor<1x3xi1>
  return %0 : tensor<1x3xi1>
}


// -----

func.func @reduce_any_constant() -> tensor<2x1xi1> {
// CHECK-LABEL:   func.func @reduce_any_constant() -> tensor<2x1xi1> {
// CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<true> : tensor<2x1xi1>}> : () -> tensor<2x1xi1>
// CHECK:           return %[[VAL_0]] : tensor<2x1xi1>
// CHECK:         }

  %const = "tosa.const"() <{values = dense<[[true,true,true], [true,false,true]]> : tensor<2x3xi1>}> : () -> tensor<2x3xi1>
  %0 = tosa.reduce_any %const {axis = 1 : i32} : (tensor<2x3xi1>) -> tensor<2x1xi1>
  return %0 : tensor<2x1xi1>
}

// -----

func.func @reduce_any_constant() -> tensor<3x1xi1> {
  // CHECK-LABEL:   func.func @reduce_any_constant() -> tensor<3x1xi1> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}true], [false], [true]]> : tensor<3x1xi1>}> : () -> tensor<3x1xi1>
  // CHECK:           return %[[VAL_0]] : tensor<3x1xi1>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[true, false, false], [false, false, false], [false, false, true]]> : tensor<3x3xi1>}> : () -> tensor<3x3xi1>
  %0 = tosa.reduce_any %const {axis = 1 : i32} : (tensor<3x3xi1>) -> tensor<3x1xi1>
  return %0 : tensor<3x1xi1>
}

// -----

func.func @reduce_any_constant() -> tensor<2x1x4xi1> {
  // CHECK-LABEL:   func.func @reduce_any_constant() -> tensor<2x1x4xi1> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}[true, false, true, true]], {{\[\[}}true, false, true, false]]]> : tensor<2x1x4xi1>}> : () -> tensor<2x1x4xi1>
  // CHECK:           return %[[VAL_0]] : tensor<2x1x4xi1>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[true, false, false, true], [false, false, true, false], [true, false, true, true]], [[false, false, false, false], [false, false, true, false], [true, false, true, false]]]> : tensor<2x3x4xi1>}> : () -> tensor<2x3x4xi1>
  %0 = tosa.reduce_any %const {axis = 1 : i32} : (tensor<2x3x4xi1>) -> tensor<2x1x4xi1>
  return %0 : tensor<2x1x4xi1>
}

// -----

  func.func @reduce_all_constant() -> tensor<1x3xi1> {
  // CHECK-LABEL:   func.func @reduce_all_constant() -> tensor<1x3xi1> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}true, false, true]]> : tensor<1x3xi1>}> : () -> tensor<1x3xi1>
  // CHECK:           return %[[VAL_0]] : tensor<1x3xi1>
  // CHECK:         }
    %const = "tosa.const"() <{values = dense<[[true,true,true], [true,false,true]]> : tensor<2x3xi1>}> : () -> tensor<2x3xi1>
    %0 = tosa.reduce_all %const {axis = 0 : i32} : (tensor<2x3xi1>) -> tensor<1x3xi1>
    return %0 : tensor<1x3xi1>
  }

// -----

  func.func @reduce_all_constant() -> tensor<2x1xi1> {
  // CHECK-LABEL:   func.func @reduce_all_constant() -> tensor<2x1xi1> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}true], [false]]> : tensor<2x1xi1>}> : () -> tensor<2x1xi1>
  // CHECK:           return %[[VAL_0]] : tensor<2x1xi1>
  // CHECK:         }
    %const = "tosa.const"() <{values = dense<[[true,true,true], [true,false,true]]> : tensor<2x3xi1>}> : () -> tensor<2x3xi1>
    %0 = tosa.reduce_all %const {axis = 1 : i32} : (tensor<2x3xi1>) -> tensor<2x1xi1>
    return %0 : tensor<2x1xi1>
  }

// -----

func.func @reduce_all_constant() -> tensor<3x1xi1> {
  // CHECK-LABEL:   func.func @reduce_all_constant() -> tensor<3x1xi1> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<false> : tensor<3x1xi1>}> : () -> tensor<3x1xi1>
  // CHECK:           return %[[VAL_0]] : tensor<3x1xi1>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[true, false, false], [false, false, false], [false, false, true]]> : tensor<3x3xi1>}> : () -> tensor<3x3xi1>
  %0 = tosa.reduce_all %const {axis = 1 : i32} : (tensor<3x3xi1>) -> tensor<3x1xi1>
  return %0 : tensor<3x1xi1>
}

// -----

func.func @reduce_all_constant() -> tensor<2x1x4xi1> {
  // CHECK-LABEL:   func.func @reduce_all_constant() -> tensor<2x1x4xi1> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<false> : tensor<2x1x4xi1>}> : () -> tensor<2x1x4xi1>
  // CHECK:           return %[[VAL_0]] : tensor<2x1x4xi1>
  // CHECK:         }
  %const = "tosa.const"() <{values = dense<[[[true, false, false, true], [false, false, true, false], [true, false, true, true]], [[false, false, false, false], [false, false, true, false], [true, false, true, false]]]> : tensor<2x3x4xi1>}> : () -> tensor<2x3x4xi1>
  %0 = tosa.reduce_all %const {axis = 1 : i32} : (tensor<2x3x4xi1>) -> tensor<2x1x4xi1>
  return %0 : tensor<2x1x4xi1>
}

// -----

func.func @reduce_sum_constant() -> tensor<1x3xi32> {
// CHECK-LABEL:   func.func @reduce_sum_constant() -> tensor<1x3xi32> {
// CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<2> : tensor<1x3xi32>}> : () -> tensor<1x3xi32>
// CHECK:           return %[[VAL_0]] : tensor<1x3xi32>
// CHECK:         }
  %const = "tosa.const"() <{values = dense<1> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
  %0 = tosa.reduce_sum %const {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<1x3xi32>
  return %0 : tensor<1x3xi32>
}

// -----

func.func @reduce_sum_constant() -> tensor<1x3xi32> {
  // CHECK-LABEL:     func.func @reduce_sum_constant() -> tensor<1x3xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
  // CHECK:           %[[VAL_1:.*]] = "tosa.const"() <{values = dense<{{\[\[}}1, 2, 3], [4, 5, 7]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
  // CHECK:           %[[VAL_2:.*]] = tosa.add %[[VAL_0]], %[[VAL_1]] : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK:           %[[VAL_3:.*]] = tosa.reduce_sum %[[VAL_2]] {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<1x3xi32>
  // CHECK:           return %[[VAL_3]] : tensor<1x3xi32>
  %arg0 = "tosa.const"() <{values = dense<[[1,2,3], [4,5,6]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
  %arg1 = "tosa.const"() <{values = dense<[[1,2,3], [4,5,7]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
  %arg2 = tosa.add %arg0, %arg1 : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  %0 = tosa.reduce_sum %arg2 {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<1x3xi32>
  return %0 : tensor<1x3xi32>
}

// -----

func.func @reduce_sum_constant_aggressive() -> tensor<1x3xi32> {
  // AGGRESIVE-LABEL: func.func @reduce_sum_constant_aggressive() -> tensor<1x3xi32> {
  // AGGRESIVE:       %[[VAL_0:.*]] = "tosa.const"() <{values = dense<4> : tensor<1x3xi32>}> : () -> tensor<1x3xi32>
  // AGGRESIVE:       return %[[VAL_0:.*]] : tensor<1x3xi32>

  // CHECK-LABEL:     func.func @reduce_sum_constant_aggressive() -> tensor<1x3xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<1> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
  // CHECK:           %[[VAL_1:.*]] = tosa.reduce_sum %[[VAL_0]] {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<1x3xi32>
  // CHECK:           %[[VAL_2:.*]] = tosa.reduce_sum %[[VAL_0]] {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<1x3xi32>
  // CHECK:           %[[VAL_3:.*]] = tosa.add %[[VAL_1]], %[[VAL_2]] : (tensor<1x3xi32>, tensor<1x3xi32>) -> tensor<1x3xi32>
  // CHECK:           return %[[VAL_3]] : tensor<1x3xi32>

  %const = "tosa.const"() {values = dense<1> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
  %0 = tosa.reduce_sum %const {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<1x3xi32>
  %1 = tosa.reduce_sum %const {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<1x3xi32>
  %res = tosa.add %0, %1 : (tensor<1x3xi32>, tensor<1x3xi32>) -> tensor<1x3xi32>
  return %res : tensor<1x3xi32>
}

// -----

func.func @reduce_sum_constant_aggressive() -> tensor<2x3xi32> {
  // AGGRESIVE-LABEL:     func.func @reduce_sum_constant_aggressive() -> tensor<2x3xi32> {
  // AGGRESIVE-DAG:       %[[VAL_0:.*]] = "tosa.const"() <{values = dense<2> : tensor<1x2x3xi32>}> : () -> tensor<1x2x3xi32>
  // AGGRESIVE-DAG:       %[[VAL_1:.*]] = "tosa.const"() <{values = dense<1> : tensor<2x2x3xi32>}> : () -> tensor<2x2x3xi32>
  // AGGRESIVE-DAG:       %[[VAL_2:.*]] = "tosa.const"() <{values = dense<2> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
  // AGGRESIVE:           %[[VAL_3:.*]] = tosa.argmax %[[VAL_0]] {axis = 1 : i32} : (tensor<1x2x3xi32>) -> tensor<1x3xi32>
  // AGGRESIVE:           %[[VAL_4:.*]] = tosa.argmax %[[VAL_1]] {axis = 0 : i32} : (tensor<2x2x3xi32>) -> tensor<2x3xi32>
  // AGGRESIVE:           %[[VAL_5:.*]] = tosa.add %[[VAL_3]], %[[VAL_2]] : (tensor<1x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // AGGRESIVE:           %[[VAL_6:.*]] = tosa.add %[[VAL_5]], %[[VAL_4]] : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // AGGRESIVE:           return %[[VAL_6]] : tensor<2x3xi32>

  // CHECK-LABEL:     func.func @reduce_sum_constant_aggressive() -> tensor<2x3xi32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<1> : tensor<2x2x3xi32>}> : () -> tensor<2x2x3xi32>
  // CHECK:           %[[VAL_1:.*]] = "tosa.const"() <{values = dense<2> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
  // CHECK:           %[[VAL_2:.*]] = tosa.reduce_sum %[[VAL_0]] {axis = 0 : i32} : (tensor<2x2x3xi32>) -> tensor<1x2x3xi32>
  // CHECK:           %[[VAL_3:.*]] = tosa.argmax %[[VAL_2]] {axis = 1 : i32} : (tensor<1x2x3xi32>) -> tensor<1x3xi32>
  // CHECK:           %[[VAL_4:.*]] = tosa.argmax %[[VAL_0]] {axis = 0 : i32} : (tensor<2x2x3xi32>) -> tensor<2x3xi32>
  // CHECK:           %[[VAL_5:.*]] = tosa.add %[[VAL_3]], %[[VAL_1]] : (tensor<1x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK:           %[[VAL_6:.*]] = tosa.add %[[VAL_5]], %[[VAL_4]] : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK:           return %[[VAL_6]] : tensor<2x3xi32>

  %const0 = "tosa.const"() {values = dense<1> : tensor<2x2x3xi32>} : () -> tensor<2x2x3xi32>
  %const1 = "tosa.const"() {values = dense<2> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
  %reduce0 = tosa.reduce_sum %const0 {axis = 0 : i32} : (tensor<2x2x3xi32>) -> tensor<1x2x3xi32>
  %argmax0 = tosa.argmax %reduce0 {axis = 1 : i32} : (tensor<1x2x3xi32>) -> tensor<1x3xi32>
  %argmax1 = tosa.argmax %const0 {axis = 0 : i32} : (tensor<2x2x3xi32>) -> tensor<2x3xi32>
  %res0 = tosa.add %argmax0, %const1 : (tensor<1x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  %res1 = tosa.add %res0, %argmax1 : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %res1 : tensor<2x3xi32>
}
