// RUN: mlir-opt -one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" --merge-alloc %s | FileCheck %s

func.func @mlp(%x: tensor<128x128xf32>, %y: tensor<128x128xf32>) -> tensor<128x128xf32> {
   // CHECK-DAG:  %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<131072xi8>
   // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
   // CHECK-DAG:  %[[VIEW_A:.*]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<131072xi8> to memref<128x128xf32>
   %a0 = tensor.empty() : tensor<128x128xf32>
   // CHECK:      linalg.matmul ins
   // CHECK-SAME: outs(%[[VIEW_A]] : memref<128x128xf32>)
   %a = linalg.matmul ins(%x, %y: tensor<128x128xf32>, tensor<128x128xf32>) outs(%a0: tensor<128x128xf32>) -> tensor<128x128xf32>
   // CHECK-DAG:  %[[C65536:.*]] = arith.constant 65536 : index
   // CHECK-DAG:  %[[VIEW_B:.*]] = memref.view %[[ALLOC]][%[[C65536]]][] : memref<131072xi8> to memref<128x128xf32>
   %b0 = tensor.empty() : tensor<128x128xf32>
   // CHECK:      linalg.matmul ins(%[[VIEW_A]],
   // CHECK-SAME: outs(%[[VIEW_B]] : memref<128x128xf32>)
   %b = linalg.matmul ins(%a, %y: tensor<128x128xf32>, tensor<128x128xf32>) outs(%b0: tensor<128x128xf32>) -> tensor<128x128xf32>
   // CHECK-DAG:  %[[C0_2:.*]] = arith.constant 0 : index
   // CHECK-DAG:  %[[VIEW_C:.*]] = memref.view %[[ALLOC]][%[[C0_2]]][] : memref<131072xi8> to memref<128x128xf32>
   %c0 = tensor.empty() : tensor<128x128xf32>
   // CHECK:      linalg.matmul ins(%[[VIEW_B]],
   // CHECK-SAME: outs(%[[VIEW_C]] : memref<128x128xf32>)
   %c = linalg.matmul ins(%b, %y: tensor<128x128xf32>, tensor<128x128xf32>) outs(%c0: tensor<128x128xf32>) -> tensor<128x128xf32>
   // CHECK-DAG:  %[[D:.*]] = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
   // CHECK:      linalg.matmul ins(%[[VIEW_C]],
   // CHECK-SAME: outs(%[[D]] : memref<128x128xf32>)
   %d0 = tensor.empty() : tensor<128x128xf32>
   %d = linalg.matmul ins(%c, %y: tensor<128x128xf32>, tensor<128x128xf32>) outs(%d0: tensor<128x128xf32>) -> tensor<128x128xf32>
   // CHECK:      return %[[D]]
   return %d : tensor<128x128xf32>
}