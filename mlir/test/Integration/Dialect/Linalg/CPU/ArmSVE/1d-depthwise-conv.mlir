// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:    -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:    -one-shot-bufferize="bufferize-function-boundaries" -lower-vector-mask -buffer-deallocation-pipeline -cse -canonicalize -convert-vector-to-scf -arm-sve-legalize-vector-storage \
// DEFINE:    -convert-vector-to-llvm="enable-arm-sve" -test-lower-to-llvm -o %t
// DEFINE: %{entry_point} = conv
// DEFINE: %{run} = %mcr_aarch64_cmd %t -e %{entry_point} -entry-point-result=void --march=aarch64 --mattr="+sve"\
// DEFINE:    -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils

// RUN: rm -f %t && %{compile} && %{run} | FileCheck %s

func.func @conv() {
  // Define input/output tensors
  %input_init = tensor.empty() : tensor<1x8x6xi32>
  %output_init = tensor.empty() : tensor<1x7x6xi32>

  %five = arith.constant 5 : i32
  %zero = arith.constant 0 : i32
  %input = linalg.fill ins(%five : i32) outs(%input_init : tensor<1x8x6xi32>) -> tensor<1x8x6xi32>
  %output = linalg.fill ins(%zero : i32) outs(%output_init : tensor<1x7x6xi32>) -> tensor<1x7x6xi32>

  // Define the filter tensor
  %filter = arith.constant dense<[
    [ 1,  2, 3, 4, 5, 6],
    [ 11, 12, 13, 14, 15, 16]
  ]> : tensor<2x6xi32>

  // static sizes -> dynamic sizes
  %input_dyn = tensor.cast %input_init : tensor<1x8x6xi32> to tensor<1x8x?xi32>
  %output_dyn = tensor.cast %output : tensor<1x7x6xi32> to tensor<1x7x?xi32>
  %filter_dyn = tensor.cast %filter : tensor<2x6xi32> to tensor<2x?xi32>

  // Run the convolution
  %res = linalg.depthwise_conv_1d_nwc_wc
    ins(%input_dyn, %filter_dyn : tensor<1x8x?xi32>, tensor<2x?xi32>)
    outs(%output_dyn : tensor<1x7x?xi32>) -> tensor<1x7x?xi32>

  // Print the results
  // CHECK: SVE: START OF TEST OUTPUT
  vector.print str "SVE: START OF TEST OUTPUT\n"

  // CHECK-NEXT: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [1, 7, 6] strides = [42, 6, 1] data =
  // CHECK-COUNT-7: [60, 70, 80, 90, 100, 110]
  %xf = tensor.cast %res : tensor<1x7x?xi32> to tensor<*xi32>
  call @printMemrefI32(%xf) : (tensor<*xi32>) -> ()

  // CHECK-NEXT: SVE: END OF TEST OUTPUT
  vector.print str "SVE: END OF TEST OUTPUT\n"

  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [1, 7, [8], 2] : !transform.any_op
    transform.yield
  }
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)
