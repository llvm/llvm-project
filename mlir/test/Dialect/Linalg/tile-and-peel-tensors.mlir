// RUN: mlir-opt %s -test-linalg-transform-patterns="test-tile-pattern tile-sizes=256,128,512 peeled-loops=0" -canonicalize | \
// RUN:     FileCheck %s -check-prefix=CHECK-PEEL-0

// RUN: mlir-opt %s -test-linalg-transform-patterns="test-tile-pattern tile-sizes=256,128,512 peeled-loops=1,2" -canonicalize | \
// RUN:     FileCheck %s -check-prefix=CHECK-PEEL-12

//     CHECK-PEEL-0: func @matmul_static_tensor
// CHECK-PEEL-0-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-PEEL-0-DAG:   %[[c128:.*]] = arith.constant 128 : index
// CHECK-PEEL-0-DAG:   %[[c256:.*]] = arith.constant 256 : index
// CHECK-PEEL-0-DAG:   %[[c512:.*]] = arith.constant 512 : index
// CHECK-PEEL-0-DAG:   %[[c1280:.*]] = arith.constant 1280 : index
// CHECK-PEEL-0-DAG:   %[[c1600:.*]] = arith.constant 1600 : index
// CHECK-PEEL-0-DAG:   %[[c1700:.*]] = arith.constant 1700 : index
//     CHECK-PEEL-0:   scf.for %{{.*}} = %[[c0]] to %[[c1280]] step %[[c256]] {{.*}} {
//     CHECK-PEEL-0:     scf.for %{{.*}} = %[[c0]] to %[[c1700]] step %[[c128]] {{.*}} {
//     CHECK-PEEL-0:       scf.for %{{.*}} = %[[c0]] to %[[c1600]] step %[[c512]] {{.*}} {
//     CHECK-PEEL-0:         linalg.matmul ins({{.*}} : tensor<256x?xf32>, tensor<?x?xf32>) outs({{.*}} : tensor<256x?xf32>)
//     CHECK-PEEL-0:       }
//     CHECK-PEEL-0:     }
//     CHECK-PEEL-0:   }
//     CHECK-PEEL-0:   scf.for %{{.*}} = %[[c0]] to %[[c1700]] step %[[c128]] {{.*}} {
//     CHECK-PEEL-0:     scf.for %{{.*}} = %[[c0]] to %[[c1600]] step %[[c512]] {{.*}} {
//     CHECK-PEEL-0:       linalg.matmul ins({{.*}} : tensor<220x?xf32>, tensor<?x?xf32>) outs({{.*}} : tensor<220x?xf32>)
//     CHECK-PEEL-0:     }
//     CHECK-PEEL-0:   }

//     CHECK-PEEL-12: func @matmul_static_tensor
// CHECK-PEEL-12-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-PEEL-12-DAG:   %[[c128:.*]] = arith.constant 128 : index
// CHECK-PEEL-12-DAG:   %[[c256:.*]] = arith.constant 256 : index
// CHECK-PEEL-12-DAG:   %[[c512:.*]] = arith.constant 512 : index
// CHECK-PEEL-12-DAG:   %[[c1500:.*]] = arith.constant 1500 : index
// CHECK-PEEL-12-DAG:   %[[c1536:.*]] = arith.constant 1536 : index
// CHECK-PEEL-12-DAG:   %[[c1600:.*]] = arith.constant 1600 : index
// CHECK-PEEL-12-DAG:   %[[c1664:.*]] = arith.constant 1664 : index
//     CHECK-PEEL-12:   scf.for %{{.*}} = %[[c0]] to %[[c1500]] step %[[c256]] {{.*}} {
//     CHECK-PEEL-12:     scf.for %{{.*}} = %[[c0]] to %[[c1664]] step %[[c128]] {{.*}} {
//     CHECK-PEEL-12:       scf.for %{{.*}} = %[[c0]] to %[[c1536]] step %[[c512]] {{.*}} {
//     CHECK-PEEL-12:         linalg.matmul ins({{.*}} : tensor<?x512xf32>, tensor<512x128xf32>) outs({{.*}} : tensor<?x128xf32>)
//     CHECK-PEEL-12:       }
//     CHECK-PEEL-12:       linalg.matmul ins({{.*}} : tensor<?x64xf32>, tensor<64x128xf32>) outs({{.*}} : tensor<?x128xf32>)
//     CHECK-PEEL-12:     }
//     CHECK-PEEL-12:     scf.for %{{.*}} = %[[c0]] to %[[c1600]] step %[[c512]] {{.*}} {
//     CHECK-PEEL-12:       linalg.matmul ins({{.*}} : tensor<?x?xf32>, tensor<?x36xf32>) outs({{.*}} : tensor<?x36xf32>)
//     CHECK-PEEL-12:     }
//     CHECK-PEEL-12:   }
func.func @matmul_static_tensor(%arg0: tensor<1500x1600xf32>, %arg1: tensor<1600x1700xf32>)
    -> tensor<1500x1700xf32> {
  %out = linalg.init_tensor [1500, 1700] : tensor<1500x1700xf32>
  %r = linalg.matmul {__internal_linalg_transform__ = "tile"}
      ins(%arg0, %arg1: tensor<1500x1600xf32>, tensor<1600x1700xf32>)
      outs(%out: tensor<1500x1700xf32>) -> tensor<1500x1700xf32>
  return %r : tensor<1500x1700xf32>
}

// -----

//     CHECK-PEEL-0: func @matmul_dynamic_tensor
// CHECK-PEEL-0-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-PEEL-0-DAG:   %[[c128:.*]] = arith.constant 128 : index
// CHECK-PEEL-0-DAG:   %[[c256:.*]] = arith.constant 256 : index
// CHECK-PEEL-0-DAG:   %[[c512:.*]] = arith.constant 512 : index
//     CHECK-PEEL-0:   scf.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c256]] {{.*}} {
//     CHECK-PEEL-0:     scf.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c128]] {{.*}} {
//     CHECK-PEEL-0:       scf.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c512]] {{.*}} {
//     CHECK-PEEL-0:         linalg.matmul ins({{.*}} : tensor<256x?xf32>, tensor<?x?xf32>) outs({{.*}} : tensor<256x?xf32>)
//     CHECK-PEEL-0:       }
//     CHECK-PEEL-0:     }
//     CHECK-PEEL-0:   }
//     CHECK-PEEL-0:   scf.for %{{.*}} {
//     CHECK-PEEL-0:     scf.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c128]] {{.*}} {
//     CHECK-PEEL-0:       scf.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c512]] {{.*}} {
//     CHECK-PEEL-0:         linalg.matmul ins({{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs({{.*}} : tensor<?x?xf32>)
//     CHECK-PEEL-0:       }
//     CHECK-PEEL-0:     }
//     CHECK-PEEL-0:   }

//     CHECK-PEEL-12: func @matmul_dynamic_tensor
// CHECK-PEEL-12-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-PEEL-12-DAG:   %[[c128:.*]] = arith.constant 128 : index
// CHECK-PEEL-12-DAG:   %[[c256:.*]] = arith.constant 256 : index
// CHECK-PEEL-12-DAG:   %[[c512:.*]] = arith.constant 512 : index
//     CHECK-PEEL-12:   scf.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c256]] {{.*}} {
//     CHECK-PEEL-12:     scf.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c128]] {{.*}} {
//     CHECK-PEEL-12:       scf.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c512]] {{.*}} {
//     CHECK-PEEL-12:         linalg.matmul ins({{.*}} : tensor<?x512xf32>, tensor<512x128xf32>) outs({{.*}} : tensor<?x128xf32>)
//     CHECK-PEEL-12:       }
//     CHECK-PEEL-12:       scf.for %{{.*}} {
//     CHECK-PEEL-12:         linalg.matmul ins({{.*}} : tensor<?x?xf32>, tensor<?x128xf32>) outs({{.*}} : tensor<?x128xf32>)
//     CHECK-PEEL-12:       }
//     CHECK-PEEL-12:     }
//     CHECK-PEEL-12:     scf.for %{{.*}} {
//     CHECK-PEEL-12:       scf.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c512]] {{.*}} {
//     CHECK-PEEL-12:         linalg.matmul ins({{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs({{.*}} : tensor<?x?xf32>)
//     CHECK-PEEL-12:       }
//     CHECK-PEEL-12:     }
//     CHECK-PEEL-12:   }
func.func @matmul_dynamic_tensor(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %out = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %r = linalg.matmul {__internal_linalg_transform__ = "tile"}
      ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%out: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}
