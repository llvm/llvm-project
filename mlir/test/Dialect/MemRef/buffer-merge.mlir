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

// CHECK-LABEL: @different_size_and_dtype
func.func @different_size_and_dtype() {
  // CHECK-DAG: %[[BASE_3:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3072xi8>
  // c and d has overlapping lifetime
  // CHECK-DAG: %[[C0_3:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C_3:.*]] = memref.view %[[BASE_3]][%[[C0_3]]][] : memref<3072xi8> to memref<8x64xf32>
  %c = memref.alloc() : memref<8x64xf32>
  // CHECK:     "test.source"(%[[C_3]])
  "test.source"(%c)  : (memref<8x64xf32>) -> ()
  // CHECK-DAG: %[[C2048_3:.*]] = arith.constant 2048 : index
  // CHECK-DAG: %[[D_3:.*]] = memref.view %[[BASE_3]][%[[C2048_3]]][] : memref<3072xi8> to memref<64xf32>
  %d = memref.alloc() : memref<64xf32>
  // CHECK:     "test.source"(%[[D_3]])
  "test.source"(%d)  : (memref<64xf32>) -> ()
  // last use of d
  // e can reuse the d's memory
  // CHECK-DAG: %[[C2048_3_2:.*]] = arith.constant 2048 : index
  // CHECK-DAG: %[[E_3:.*]] = memref.view %[[BASE_3]][%[[C2048_3_2]]][] : memref<3072xi8> to memref<2x2x32xf64>
  %e = memref.alloc() : memref<2x2x32xf64>
  // CHECK:     "test.source"(%[[E_3]])
  "test.source"(%e)  : (memref<2x2x32xf64>) -> ()
  // CHECK:     "test.source"(%[[C_3]])
  "test.source"(%c)  : (memref<8x64xf32>) -> ()

  // e and c are free'd. f can reuse the memory across e and c
  // CHECK-DAG: %[[C0_3_2:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[F_3:.*]] = memref.view %[[BASE_3]][%[[C0_3_2]]][] : memref<3072xi8> to memref<2x4x32xf64>
  %f = memref.alloc() : memref<2x4x32xf64>
  // CHECK:     "test.source"(%[[F_3]])
  "test.source"(%f)  : (memref<2x4x32xf64>) -> ()
  // CHECK:     return
  return
}

// CHECK-LABEL: @withloop
func.func @withloop() {
  // CHECK-DAG: %[[BASE2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<6144xi8>
  // CHECK-DAG: %[[C033:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[F:.*]] = memref.view %[[BASE2]][%[[C033]]][] : memref<6144xi8> to memref<8x64xf32>
  %f = memref.alloc() : memref<8x64xf32>
  // CHECK-DAG: %[[C2048:.*]] = arith.constant 2048 : index
  // CHECK-DAG: %[[G:.*]] = memref.view %[[BASE2]][%[[C2048]]][] : memref<6144xi8> to memref<8x64xf32>
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


// CHECK-LABEL: @different_mem_space
func.func @different_mem_space() {
  // CHECK-DAG: %[[BASE_DEFAULT:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4096xi8>
  // CHECK-DAG: %[[BASE_WG:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3584xi8, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[BASE_PRIVATE:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1280xi8, #gpu.address_space<private>>
  // CHECK-DAG: %[[C0_4:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[defaultv:.*]] = memref.view %[[BASE_DEFAULT]][%[[C0_4]]][] : memref<4096xi8> to memref<8x64xf32>
  %defaultv = memref.alloc() : memref<8x64xf32>
  // CHECK:     "test.source"(%[[defaultv]])
  "test.source"(%defaultv)  : (memref<8x64xf32>) -> ()
  // CHECK-DAG: %[[C2048_4:.*]] = arith.constant 2048 : index
  // CHECK-DAG: %[[defaultv2:.*]] = memref.view %[[BASE_DEFAULT]][%[[C2048_4]]][] : memref<4096xi8> to memref<8x64xf32>
  %defaultv2 = memref.alloc() : memref<8x64xf32>
  // CHECK-DAG: %[[C2048_4_2:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[priv:.*]] = memref.view %[[BASE_PRIVATE]][%[[C2048_4_2]]][] : memref<1280xi8, #gpu.address_space<private>> to memref<4x64xf32, #gpu.address_space<private>>
  %priv = memref.alloc() : memref<4x64xf32, #gpu.address_space<private>>
  // CHECK:     "test.source"(%[[priv]])
  "test.source"(%priv)  : (memref<4x64xf32, #gpu.address_space<private>>) -> ()
  // CHECK-DAG: %[[C0_4_2:.*]] = arith.constant 1024 : index
  // CHECK-DAG: %[[priv2:.*]] = memref.view %[[BASE_PRIVATE]][%[[C0_4_2]]][] : memref<1280xi8, #gpu.address_space<private>> to memref<64xf32, #gpu.address_space<private>>
  %priv2 = memref.alloc() : memref<64xf32,  #gpu.address_space<private>>
  // CHECK:     "test.source"(%[[priv2]])
  "test.source"(%priv2)  : (memref<64xf32, #gpu.address_space<private>>) -> ()
  // CHECK:     "test.source"(%[[priv]])
  "test.source"(%priv)  : (memref<4x64xf32, #gpu.address_space<private>>) -> ()


  // CHECK-DAG: %[[C2048_4_3:.*]] = arith.constant 1536 : index
  // CHECK-DAG: %[[sharedv:.*]] = memref.view %[[BASE_WG]][%[[C2048_4_3]]][] : memref<3584xi8, #gpu.address_space<workgroup>> to memref<8x64xf32, #gpu.address_space<workgroup>>
  %sharedv = memref.alloc() : memref<8x64xf32, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[C0_4_3:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[sharedv2:.*]] = memref.view %[[BASE_WG]][%[[C0_4_3]]][] : memref<3584xi8, #gpu.address_space<workgroup>> to memref<6x64xf32, #gpu.address_space<workgroup>>
  %sharedv2 = memref.alloc() : memref<6x64xf32, #gpu.address_space<workgroup>>
  // CHECK:     "test.source"(%[[sharedv2]])
  // CHECK:     "test.source"(%[[sharedv]])
  // CHECK:     "test.source"(%[[sharedv2]])
  "test.source"(%sharedv2)  : (memref<6x64xf32, #gpu.address_space<workgroup>>) -> ()
  "test.source"(%sharedv)  : (memref<8x64xf32, #gpu.address_space<workgroup>>) -> ()
  "test.source"(%sharedv2)  : (memref<6x64xf32, #gpu.address_space<workgroup>>) -> ()


  // CHECK:     "test.source"(%[[defaultv2]])
  // CHECK:     "test.source"(%[[defaultv]])
  "test.source"(%defaultv2)  : (memref<8x64xf32>) -> ()
  "test.source"(%defaultv)  : (memref<8x64xf32>) -> ()
  return
}
