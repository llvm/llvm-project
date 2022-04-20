// RUN: mlir-opt %s -pass-pipeline="linalg-comprehensive-module-bufferize{allow-return-allocs use-alloca}" -split-input-file | FileCheck %s

//  CHECK-DAG: #[[$DYN_0D_MAP:.*]] = affine_map<()[s0] -> (s0)>
//  CHECK-DAG: #[[$DYN_1D_MAP:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

//      CHECK:  func @init_and_dot(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<64xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]*]]: memref<64xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[C:[a-zA-Z0-9]*]]: memref<f32, #[[$DYN_0D_MAP]]>
func.func @init_and_dot(%a: tensor<64xf32>, %b: tensor<64xf32>, %c: tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT:   %[[C0:.*]] = arith.constant 0{{.*}} : f32
  %v0 = arith.constant 0.0 : f32

  // CHECK-NEXT:   linalg.fill ins(%[[C0]] : f32) outs(%[[C]] : memref<f32, #[[$DYN_0D_MAP]]>)
  %d = linalg.fill ins(%v0 : f32) outs(%c : tensor<f32>) -> tensor<f32>

  // CHECK-NEXT:   linalg.dot ins(%[[A]], %[[B]] : memref<64xf32, #[[$DYN_1D_MAP]]>, memref<64xf32, #[[$DYN_1D_MAP]]>) outs(%[[C]] : memref<f32, #[[$DYN_0D_MAP]]>)
  %e = linalg.dot ins(%a, %b : tensor<64xf32>,tensor<64xf32>)
    outs(%d: tensor<f32>) -> tensor<f32>

  // CHECK-NEXT:   return
  return %e : tensor<f32>
}

//      CHECK:  func @main()
func.func @main() {
  //  CHECK-DAG:   %[[C0:.*]] = arith.constant 0{{.*}} : f32
  //  CHECK-DAG:   %[[C1:.*]] = arith.constant 1{{.*}} : f32
  //  CHECK-DAG:   %[[C2:.*]] = arith.constant 2{{.*}} : f32
  %v0 = arith.constant 0.0 : f32
  %v1 = arith.constant 1.0 : f32
  %v2 = arith.constant 2.0 : f32

  // CHECK-NEXT:   %[[A:.*]] = memref.alloca() {alignment = 128 : i64} : memref<64xf32>
  // CHECK-NEXT:   %[[B:.*]] = memref.alloca() {alignment = 128 : i64} : memref<64xf32>
  // CHECK-NEXT:   %[[C:.*]] = memref.alloca() {alignment = 128 : i64} : memref<f32>
  //  CHECK-DAG:   %[[cA:.*]] = memref.cast %[[A]] : memref<64xf32> to memref<64xf32, #[[$DYN_1D_MAP]]>
  //  CHECK-DAG:   %[[cB:.*]] = memref.cast %[[B]] : memref<64xf32> to memref<64xf32, #[[$DYN_1D_MAP]]>
  //  CHECK-DAG:   %[[cC:.*]] = memref.cast %[[C]] : memref<f32> to memref<f32, #[[$DYN_0D_MAP]]>
  %A = linalg.init_tensor [64] : tensor<64xf32>
  %B = linalg.init_tensor [64] : tensor<64xf32>
  %C = linalg.init_tensor [] : tensor<f32>

  //  CHECK-DAG:   linalg.fill ins(%[[C1]] : f32) outs(%[[A]] : memref<64xf32>)
  //  CHECK-DAG:   linalg.fill ins(%[[C2]] : f32) outs(%[[B]] : memref<64xf32>)
  //  CHECK-DAG:   linalg.fill ins(%[[C0]] : f32) outs(%[[C]] : memref<f32>)
  %AA = linalg.fill ins(%v1 : f32) outs(%A : tensor<64xf32>) -> tensor<64xf32>
  %BB = linalg.fill ins(%v2 : f32) outs(%B : tensor<64xf32>) -> tensor<64xf32>
  %CC = linalg.fill ins(%v0 : f32) outs(%C : tensor<f32>) -> tensor<f32>

  // CHECK-NEXT:   call @init_and_dot(%[[cA]], %[[cB]], %[[cC]])
  %res = call @init_and_dot(%AA, %BB, %CC) :
    (tensor<64xf32>, tensor<64xf32>, tensor<f32>) -> tensor<f32>

  // CHECK-NEXT:   %[[dC:.*]] = memref.cast %[[C]] : memref<f32> to memref<*xf32>
  %res2 = tensor.cast %res: tensor<f32> to tensor<*xf32>

  // CHECK-NEXT:   call @print_memref_f32(%[[dC]]) : (memref<*xf32>) -> ()
  call @print_memref_f32(%res2) : (tensor<*xf32>) -> ()

  return
}

//     CHECK:   func private @print_memref_f32(memref<*xf32>)
func.func private @print_memref_f32(tensor<*xf32>)
