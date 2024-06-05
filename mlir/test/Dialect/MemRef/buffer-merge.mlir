// RUN: mlir-opt -allow-unregistered-dialect -one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" --merge-alloc %s | FileCheck %s

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

// CHECK-LABEL: @basic
func.func @basic() -> memref<8x64xf32> {
  // CHECK-DAG: %[[BASE:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4096xi8>
  // b is used in return, complex lifetime
  // CHECK-DAG: %[[B:.*]] = memref.alloc()
  %b = memref.alloc() : memref<8x64xf32>
  // CHECK:     "test.source"(%[[B]])
  "test.source"(%b)  : (memref<8x64xf32>) -> ()
  // c and d has overlapping lifetime
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C:.*]] = memref.view %[[BASE]][%[[C0]]][] : memref<4096xi8> to memref<8x64xf32>
  %c = memref.alloc() : memref<8x64xf32>
  // CHECK:     "test.source"(%[[C]])
  "test.source"(%c)  : (memref<8x64xf32>) -> ()
  // CHECK-DAG: %[[C2048:.*]] = arith.constant 2048 : index
  // CHECK-DAG: %[[D:.*]] = memref.view %[[BASE]][%[[C2048]]][] : memref<4096xi8> to memref<8x64xf32>
  %d = memref.alloc() : memref<8x64xf32>
  // CHECK:     "test.source"(%[[D]])
  "test.source"(%d)  : (memref<8x64xf32>) -> ()
  // CHECK:     "test.source"(%[[C]])
  "test.source"(%c)  : (memref<8x64xf32>) -> ()
  // e can reuse the above memory
  // CHECK-DAG: %[[C0_2:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[E:.*]] = memref.view %[[BASE]][%[[C0_2]]][] : memref<4096xi8> to memref<8x64xf32>
  %e = memref.alloc() : memref<8x64xf32>
  // CHECK:     "test.source"(%[[E]])
  "test.source"(%e)  : (memref<8x64xf32>) -> ()
  // CHECK:     return %[[B]]
  return %b : memref<8x64xf32>
}

// CHECK-LABEL: @withloop
func.func @withloop() {
  // CHECK-DAG: %[[BASE2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<6144xi8>
  // CHECK-DAG: %[[C2048:.*]] = arith.constant 2048 : index
  // CHECK-DAG: %[[F:.*]] = memref.view %[[BASE2]][%[[C2048]]][] : memref<6144xi8> to memref<8x64xf32>
  %f = memref.alloc() : memref<8x64xf32>
  // CHECK-DAG: %[[C033:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[G:.*]] = memref.view %[[BASE2]][%[[C033]]][] : memref<6144xi8> to memref<8x64xf32>
  %g = memref.alloc() : memref<8x64xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  // CHECK: scf.for
  scf.for %i = %c0 to %c5 step %c1 {
      // CHECK:     "test.source"(%[[F]])
      "test.source"(%f)  : (memref<8x64xf32>) -> ()
      // CHECK:     "test.source"(%[[G]])
      "test.source"(%g)  : (memref<8x64xf32>) -> ()
      // CHECK-DAG: %[[C4096:.*]] = arith.constant 4096 : index
      // CHECK-DAG: %[[H:.*]] = memref.view %[[BASE2]][%[[C4096]]][] : memref<6144xi8> to memref<8x64xf32>
      %h = memref.alloc() : memref<8x64xf32>
      // CHECK:     "test.source"(%[[H]])
      "test.source"(%h)  : (memref<8x64xf32>) -> ()
      // CHECK-DAG: %[[C4096_3:.*]] = arith.constant 4096 : index
      // CHECK-DAG: %[[J:.*]] = memref.view %[[BASE2]][%[[C4096_3]]][] : memref<6144xi8> to memref<8x64xf32>
      %j = memref.alloc() : memref<8x64xf32>
      // CHECK:     "test.source"(%[[J]])
      "test.source"(%j)  : (memref<8x64xf32>) -> ()
  }
  return
}