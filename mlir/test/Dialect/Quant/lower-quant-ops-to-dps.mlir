// RUN: mlir-opt %s -lower-quant-ops -linalg-convert-to-dps \
// RUN:             -linalg-specialize-generic-ops -cse | FileCheck %s

// CHECK-LABEL: func.func @lower_qcast_to_dps(
// CHECK-SAME:    %[[X:.+]]: tensor<10xf32>) -> tensor<10x!quant.uniform<i8:f32, 2.000000e+00:10>>
// CHECK-DAG:     %[[CST_10I:.+]] = arith.constant dense<10> : tensor<10xi8>
// CHECK-DAG:     %[[CST_2F:.+]] = arith.constant dense<2.000000e+00> : tensor<10xf32>
// CHECK:         %[[E:.+]] = tensor.empty() : tensor<10xf32>
// CHECK:         %[[DIV:.+]] = linalg.div ins(%[[X]], %[[CST_2F]] : tensor<10xf32>, tensor<10xf32>)
// CHECK-SAME:                             outs(%[[E]] : tensor<10xf32>) -> tensor<10xf32>
//
// CHECK:         %[[SITOFP:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[CST_10I]] : tensor<10xi8>) outs(%[[E]] : tensor<10xf32>)
// CHECK:            %{{.*}} = arith.sitofp %{{.*}} : i8 to f32
//
// CHECK:         %[[ADD:.+]] = linalg.add ins(%[[DIV]], %[[SITOFP]] : tensor<10xf32>, tensor<10xf32>)
// CHECK:         %{{.*}} = linalg.generic
// CHECK-SAME:       ins(%[[ADD]] : tensor<10xf32>)
// CHECK:            %{{.*}} = arith.fptosi %{{.*}} : f32 to i8


!qalias = !quant.uniform<i8:f32, 2.0:10>
func.func @lower_qcast_to_dps(%arg0: tensor<10xf32>) -> tensor<10x!qalias> {
  %0 = quant.qcast %arg0 : tensor<10xf32> to tensor<10x!qalias>
  return %0 : tensor<10x!qalias>
}
