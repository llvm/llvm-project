// RUN: mlir-opt %s --pre-sparsification-rewrite | FileCheck %s

#CCCD = #sparse_tensor.encoding<{ map = (d0, d1, d2, d3) -> (d0 : compressed, d1 : compressed, d2 : compressed, d3 : dense) }>



// CHECK-LABEL: func.func @fuse_concat_with_extract(
// CHECK-SAME:    %[[VAL_0:.*0]]: tensor<128x32x32x1xf32, #sparse{{[0-9]*}}>,
// CHECK-SAME:    %[[VAL_1:.*1]]: tensor<128x32x32x1xf32, #sparse{{[0-9]*}}>,
// CHECK-SAME:    %[[VAL_2:.*2]]: tensor<128x32x32x1xf32, #sparse{{[0-9]*}}>)
// CHECK-NOT:     tensor.concat
// CHECK-NOT:     tensor.extract_slice
// CHECK:         return %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]
// CHECK:       }
func.func @fuse_concat_with_extract(%t0 : tensor<128x32x32x1xf32, #CCCD>,
                                    %t1 : tensor<128x32x32x1xf32, #CCCD>,
                                    %t2 : tensor<128x32x32x1xf32, #CCCD>) -> (tensor<128x32x32x1xf32, #CCCD>, tensor<128x32x32x1xf32, #CCCD>, tensor<128x32x32x1xf32, #CCCD>) {
  %concat = tensor.concat dim(3) %t0, %t1, %t2 : (tensor<128x32x32x1xf32, #CCCD>, tensor<128x32x32x1xf32, #CCCD>, tensor<128x32x32x1xf32, #CCCD>) -> tensor<128x32x32x3xf32, #CCCD>
  %r0 = tensor.extract_slice %concat[0, 0, 0, 0] [128, 32, 32, 1] [1, 1, 1, 1] : tensor<128x32x32x3xf32, #CCCD> to tensor<128x32x32x1xf32, #CCCD>
  %r1 = tensor.extract_slice %concat[0, 0, 0, 1] [128, 32, 32, 1] [1, 1, 1, 1] : tensor<128x32x32x3xf32, #CCCD> to tensor<128x32x32x1xf32, #CCCD>
  %r2 = tensor.extract_slice %concat[0, 0, 0, 2] [128, 32, 32, 1] [1, 1, 1, 1] : tensor<128x32x32x3xf32, #CCCD> to tensor<128x32x32x1xf32, #CCCD>
  return %r0, %r1, %r2 : tensor<128x32x32x1xf32, #CCCD>, tensor<128x32x32x1xf32, #CCCD>, tensor<128x32x32x1xf32, #CCCD>
}
