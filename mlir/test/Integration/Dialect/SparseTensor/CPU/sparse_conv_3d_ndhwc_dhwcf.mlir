//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparsifier_opts} = enable-runtime-library=true
// DEFINE: %{sparsifier_opts_sve} = enable-arm-sve=true %{sparsifier_opts}
// DEFINE: %{compile} = mlir-opt %s --sparsifier="%{sparsifier_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparsifier="%{sparsifier_opts_sve}"
// DEFINE: %{run_libs} = -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
// DEFINE: %{run_opts} = -e main -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#CCCCC = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3, d4) -> (d0 : compressed, d1 : compressed, d2 : compressed, d3 : compressed, d4 : compressed)
}>

#CDCDC = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3, d4) -> (d0 : compressed, d1 : dense, d2 : compressed, d3 : dense, d4 : compressed)
}>

// Creates and returns 5-D buffer of size (%s1, %s2, %s3, %s4, %s5) filled with the value %f
func.func @alloc_5d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %s4 : index, %s5 : index, %f : f32) -> tensor<?x?x?x?x?xf32> {
  %buf = tensor.empty(%s1, %s2, %s3, %s4, %s5) : tensor<?x?x?x?x?xf32>
  %ret = linalg.fill ins(%f : f32) outs(%buf : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %ret : tensor<?x?x?x?x?xf32>
}

func.func @conv_3d_ndhwc_dhwcf(%arg0: tensor<?x?x?x?x?xf32>,
                               %arg1: tensor<?x?x?x?x?xf32>,
                               %arg2: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> {
  %ret = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>,
                                       strides = dense<1> : tensor<3xi64>}
     ins (%arg0, %arg1: tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>)
    outs (%arg2: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
  return %ret : tensor<?x?x?x?x?xf32>
}

func.func @conv_3d_ndhwc_dhwcf_CCCCC(%arg0: tensor<?x?x?x?x?xf32, #CCCCC>,
                                     %arg1: tensor<?x?x?x?x?xf32>)
                                     -> tensor<?x?x?x?x?xf32, #CCCCC> {
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %s = tensor.empty(%c1, %c6, %c6, %c6, %c1)
    : tensor<?x?x?x?x?xf32, #CCCCC>
  %ret = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>,
                                       strides = dense<1> : tensor<3xi64>}
     ins (%arg0, %arg1: tensor<?x?x?x?x?xf32, #CCCCC>, tensor<?x?x?x?x?xf32>)
    outs (%s: tensor<?x?x?x?x?xf32, #CCCCC>) -> tensor<?x?x?x?x?xf32, #CCCCC>
  return %ret : tensor<?x?x?x?x?xf32, #CCCCC>
}

func.func @conv_3d_ndhwc_dhwcf_CDCDC(%arg0: tensor<?x?x?x?x?xf32, #CDCDC>,
                                     %arg1: tensor<?x?x?x?x?xf32>)
                                     -> tensor<?x?x?x?x?xf32, #CDCDC> {
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %s = tensor.empty(%c1, %c6, %c6, %c6, %c1)
    : tensor<?x?x?x?x?xf32, #CDCDC>
  %ret = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>,
                                       strides = dense<1> : tensor<3xi64>}
     ins (%arg0, %arg1: tensor<?x?x?x?x?xf32, #CDCDC>, tensor<?x?x?x?x?xf32>)
    outs (%s: tensor<?x?x?x?x?xf32, #CDCDC>) -> tensor<?x?x?x?x?xf32, #CDCDC>
  return %ret : tensor<?x?x?x?x?xf32, #CDCDC>
}

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  %f10 = arith.constant 10.00000e+00 : f32
  %val = arith.constant 2.00000e+00 : f32
  %zero = arith.constant 0.00000e+00 : f32

  %in3D_tmp = call @alloc_5d_filled_f32(%c1, %c8, %c8, %c8, %c1, %val) : (index, index, index, index, index, f32) -> (tensor<?x?x?x?x?xf32>)
  %in3D_ndhwc = tensor.insert %f10 into %in3D_tmp[%c0, %c0, %c0, %c3, %c0] : tensor<?x?x?x?x?xf32>

  %filter3D_ndhwc = call @alloc_5d_filled_f32(%c3, %c3, %c3, %c1, %c1, %val) : (index, index, index, index, index, f32) -> (tensor<?x?x?x?x?xf32>)
  %out3D_ndhwc = call @alloc_5d_filled_f32(%c1, %c6, %c6, %c6, %c1, %zero) : (index, index, index, index, index, f32) -> (tensor<?x?x?x?x?xf32>)

  %in3D_ndhwc_CCCCC = sparse_tensor.convert %in3D_ndhwc
    : tensor<?x?x?x?x?xf32> to tensor<?x?x?x?x?xf32, #CCCCC>
  %in3D_ndhwc_CDCDC = sparse_tensor.convert %in3D_ndhwc
    : tensor<?x?x?x?x?xf32> to tensor<?x?x?x?x?xf32, #CDCDC>

  //      CHECK:( ( ( ( ( 108 ), ( 124 ), ( 124 ), ( 124 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ),
  // CHECK-SAME:    ( ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ),
  // CHECK-SAME:      ( ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ), ( 108 ) ) ) ) )
  %dense_ret = call @conv_3d_ndhwc_dhwcf(%in3D_ndhwc, %filter3D_ndhwc, %out3D_ndhwc)
      : (tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) -> (tensor<?x?x?x?x?xf32>)
  %dense_v = vector.transfer_read %dense_ret[%c0, %c0, %c0, %c0, %c0], %zero
      : tensor<?x?x?x?x?xf32>, vector<1x6x6x6x1xf32>
  vector.print %dense_v : vector<1x6x6x6x1xf32>

  %CCCCC_ret = call @conv_3d_ndhwc_dhwcf_CCCCC(%in3D_ndhwc_CCCCC, %filter3D_ndhwc)
      : (tensor<?x?x?x?x?xf32, #CCCCC>,
         tensor<?x?x?x?x?xf32>) -> (tensor<?x?x?x?x?xf32, #CCCCC>)

  //
  // CHECK:      ---- Sparse Tensor ----
  // CHECK-NEXT: nse = 216
  // CHECK-NEXT: dim = ( 1, 6, 6, 6, 1 )
  // CHECK-NEXT: lvl = ( 1, 6, 6, 6, 1 )
  // CHECK-NEXT: pos[0] : ( 0, 1 )
  // CHECK-NEXT: crd[0] : ( 0 )
  // CHECK-NEXT: pos[1] : ( 0, 6 )
  // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 4, 5 )
  // CHECK-NEXT: pos[2] : ( 0, 6, 12, 18, 24, 30, 36 )
  // CHECK-NEXT: crd[2] : ( 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
  // CHECK-SAME:            4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5 )
  // CHECK-NEXT: pos[3] : ( 0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96,
  // CHECK-SAME:            102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174,
  // CHECK-SAME:            180, 186, 192, 198, 204, 210, 216 )
  // CHECK-NEXT: crd[3] : ( 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
  // CHECK-SAME:            4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1,
  // CHECK-SAME:            2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
  // CHECK-SAME:            0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
  // CHECK-SAME:            4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1,
  // CHECK-SAME:            2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
  // CHECK-SAME:            0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
  // CHECK-SAME:            4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1,
  // CHECK-SAME:            2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
  // CHECK-SAME:            0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5 )
  // CHECK-NEXT: pos[4] : ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
  // CHECK-SAME:            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
  // CHECK-SAME:            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
  // CHECK-SAME:            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
  // CHECK-SAME:            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
  // CHECK-SAME:            87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
  // CHECK-SAME:            103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
  // CHECK-SAME:            117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
  // CHECK-SAME:            131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
  // CHECK-SAME:            145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
  // CHECK-SAME:            159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
  // CHECK-SAME:            173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
  // CHECK-SAME:            187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
  // CHECK-SAME:            201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214,
  // CHECK-SAME:            215, 216 )
  // CHECK-NEXT: crd[4] : ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: values : ( 108, 124, 124, 124, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108 )
  // CHECK-NEXT: ----
  //
  sparse_tensor.print %CCCCC_ret : tensor<?x?x?x?x?xf32, #CCCCC>

  %CDCDC_ret = call @conv_3d_ndhwc_dhwcf_CDCDC(%in3D_ndhwc_CDCDC, %filter3D_ndhwc)
      : (tensor<?x?x?x?x?xf32, #CDCDC>,
         tensor<?x?x?x?x?xf32>) -> (tensor<?x?x?x?x?xf32, #CDCDC>)

  //
  // CHECK:      ---- Sparse Tensor ----
  // CHECK-NEXT: nse = 216
  // CHECK-NEXT: dim = ( 1, 6, 6, 6, 1 )
  // CHECK-NEXT: lvl = ( 1, 6, 6, 6, 1 )
  // CHECK-NEXT: pos[0] : ( 0, 1 )
  // CHECK-NEXT: crd[0] : ( 0 )
  // CHECK-NEXT: pos[2] : ( 0, 6, 12, 18, 24, 30, 36 )
  // CHECK-NEXT: crd[2] : ( 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3,
  // CHECK-SAME:            4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5 )
  // CHECK-NEXT: pos[4] : ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
  // CHECK-SAME:            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
  // CHECK-SAME:            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
  // CHECK-SAME:            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
  // CHECK-SAME:            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
  // CHECK-SAME:            87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
  // CHECK-SAME:            103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
  // CHECK-SAME:            117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
  // CHECK-SAME:            131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
  // CHECK-SAME:            145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
  // CHECK-SAME:            159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
  // CHECK-SAME:            173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
  // CHECK-SAME:            187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
  // CHECK-SAME:            201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214,
  // CHECK-SAME:            215, 216 )
  // CHECK-NEXT: crd[4] : ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME:            0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: values : ( 108, 124, 124, 124, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
  // CHECK-SAME:            108, 108, 108, 108, 108, 108 )
  // CHECK-NEXT: ----
  //
  sparse_tensor.print %CDCDC_ret : tensor<?x?x?x?x?xf32, #CDCDC>

  // Free the resources
  bufferization.dealloc_tensor %in3D_ndhwc : tensor<?x?x?x?x?xf32>
  bufferization.dealloc_tensor %filter3D_ndhwc : tensor<?x?x?x?x?xf32>
  bufferization.dealloc_tensor %out3D_ndhwc : tensor<?x?x?x?x?xf32>

  bufferization.dealloc_tensor %in3D_ndhwc_CDCDC : tensor<?x?x?x?x?xf32, #CDCDC>
  bufferization.dealloc_tensor %in3D_ndhwc_CCCCC : tensor<?x?x?x?x?xf32, #CCCCC>

  bufferization.dealloc_tensor %CCCCC_ret : tensor<?x?x?x?x?xf32, #CCCCC>
  bufferization.dealloc_tensor %CDCDC_ret : tensor<?x?x?x?x?xf32, #CDCDC>

  return
}
