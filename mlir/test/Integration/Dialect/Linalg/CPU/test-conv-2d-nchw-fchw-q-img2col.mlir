// UNSUPPORTED: asan
// DEFINE: %{lower} = -test-transform-dialect-erase-schedule \
// DEFINE:   -one-shot-bufferize="bufferize-function-boundaries" \
// DEFINE:   -buffer-deallocation-pipeline -convert-bufferization-to-memref \
// DEFINE:   -convert-linalg-to-loops -convert-scf-to-cf \
// DEFINE:   -expand-strided-metadata -lower-affine -convert-arith-to-llvm \
// DEFINE:   -convert-scf-to-cf -finalize-memref-to-llvm -convert-func-to-llvm \
// DEFINE:   -convert-cf-to-llvm -reconcile-unrealized-casts
// DEFINE: %{run} = mlir-runner -e main -entry-point-result=void \
// DEFINE:   -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils

// Check the reference convolution
// RUN: mlir-opt %s %{lower} | %{run} | FileCheck %s

// RUN: mlir-opt %s -transform-interpreter -test-transform-dialect-erase-schedule -o %t
// RUN: FileCheck %s --check-prefix=IM2COL --implicit-check-not=linalg.conv_2d_nchw_fchw_q < %t
// RUN: mlir-opt %t %{lower} | %{run} | FileCheck %s

func.func private @printMemrefI32(tensor<*xi32>)

// IM2COL-LABEL: func.func @conv(
// IM2COL: linalg.generic
// IM2COL: linalg.generic
// IM2COL: tensor.expand_shape
// IM2COL: return
func.func @conv(%input: tensor<2x2x3x3xi8>,
                %filter: tensor<3x2x2x2xi8>,
                %init: tensor<2x3x2x2xi32>,
                %input_zp: i32, %filter_zp: i32) -> tensor<2x3x2x2xi32> {
  %result = linalg.conv_2d_nchw_fchw_q
      {strides = dense<1> : tensor<2xi64>, dilations = dense<1> : tensor<2xi64>}
      ins(%input, %filter, %input_zp, %filter_zp :
          tensor<2x2x3x3xi8>, tensor<3x2x2x2xi8>, i32, i32)
      outs(%init : tensor<2x3x2x2xi32>) -> tensor<2x3x2x2xi32>
  return %result : tensor<2x3x2x2xi32>
}

func.func @main() {
  %input = arith.constant dense<[
    [[[-4, -3, -2],[-1, 0, 1], [2, 3, 4]],
     [[5, 4, 3], [2, 1, 0], [-1, -2, -3]]],
    [[[6, -5, 4], [-3, 2, -1], [0, 1, -2]],
     [[-7, 6, -5], [4, -3, 2], [-1, 0, 1]]]
  ]> : tensor<2x2x3x3xi8>
  %filter = arith.constant dense<[
    [[[1, -2], [3, 0]], [[-1, 2], [-3, 4]]],
    [[[-4, 1], [2, -1]], [[3, -2], [0, 1]]],
    [[[2, 0], [-1, -3]], [[-2, 4], [1, -4]]]
  ]> : tensor<3x2x2x2xi8>
  %init = arith.constant dense<7> : tensor<2x3x2x2xi32>
  %input_zp = arith.constant 3 : i32
  %filter_zp = arith.constant -2 : i32
  %result = func.call @conv(%input, %filter, %init, %input_zp, %filter_zp) :
      (tensor<2x2x3x3xi8>, tensor<3x2x2x2xi8>, tensor<2x3x2x2xi32>, i32, i32)
      -> tensor<2x3x2x2xi32>

  %flat = tensor.collapse_shape %result [[0], [1, 2, 3]] :
      tensor<2x3x2x2xi32> into tensor<2x12xi32>
  %unranked = tensor.cast %flat : tensor<2x12xi32> to tensor<*xi32>
  func.call @printMemrefI32(%unranked) : (tensor<*xi32>) -> ()
  return
}

// CHECK: Unranked Memref
// CHECK-SAME: rank = 2 offset = 0 sizes = [2, 12] strides = [12, 1] data =
// CHECK-NEXT: [-45, -45, -45, -45, -14, -18, -26, -30, -27, -28, -30, -31]
// CHECK-NEXT: [-51, -59, -67, -35, -114, 18, -10, -58, 31, -84, -62, -13]
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %conv = transform.structured.match ops{["linalg.conv_2d_nchw_fchw_q"]}
        in %root : (!transform.any_op) -> !transform.any_op
    %col, %result = transform.structured.convert_conv2d_to_img2col %conv :
        (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
