// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize -split-input-file | FileCheck %s

//      CHECK: #[[$DYN_1D_MAP:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

//      CHECK:  func private @some_external_func(memref<?xf32, #[[$DYN_1D_MAP]]>)
func private @some_external_func(tensor<?xf32>)

//      CHECK:  func @scf_for_with_tensor_insert_slice(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]*]]: memref<?xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[C:[a-zA-Z0-9]*]]: memref<4xf32, #[[$DYN_1D_MAP]]>
func @scf_for_with_tensor_insert_slice(
    %A : tensor<?xf32>, %B : tensor<?xf32>, %C : tensor<4xf32>,
    %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // CHECK-NEXT: scf.for
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    // CHECK-NEXT:   %[[SVA:.*]] = memref.subview %[[A]]
    // CHECK-NEXT:   linalg.copy(%[[C]], %[[SVA]]) : memref<4xf32, #[[$DYN_1D_MAP]]>, memref<4xf32, #[[$DYN_1D_MAP]]>
    %ttA = tensor.insert_slice %C into %tA[%i][4][1] : tensor<4xf32> into tensor<?xf32>

    // CHECK-NEXT:   %[[SVB:.*]] = memref.subview %[[B]]
    // CHECK-NEXT:   linalg.copy(%[[C]], %[[SVB]]) : memref<4xf32, #[[$DYN_1D_MAP]]>, memref<4xf32, #[[$DYN_1D_MAP]]>
    %ttB = tensor.insert_slice %C into %tB[%i][4][1] : tensor<4xf32> into tensor<?xf32>

    // scf.yield is empty and is elided
    //  CHECK-NOT:   scf.yield
    scf.yield %ttA, %ttB : tensor<?xf32>, tensor<?xf32>
  }

  // Swaparoo requires bufferizing the whole function to figure out who's who.
  return %r0#1, %r0#0: tensor<?xf32>, tensor<?xf32>
}

//      CHECK:  func @bar(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]*]]: memref<?xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[C:[a-zA-Z0-9]*]]: memref<4xf32, #[[$DYN_1D_MAP]]>
func @bar(
    %A : tensor<?xf32> {linalg.inplaceable = true},
    %B : tensor<?xf32> {linalg.inplaceable = true},
    %C : tensor<4xf32> {linalg.inplaceable = true},
    %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
// CHECK-NEXT:   call @scf_for_with_tensor_insert_slice(%[[A]], %[[B]], %[[C]]
  %r0:2 = call @scf_for_with_tensor_insert_slice(%A, %B, %C, %lb, %ub, %step) :
      (tensor<?xf32>, tensor<?xf32>, tensor<4xf32>, index, index, index)
        -> (tensor<?xf32>, tensor<?xf32>)

  // %r0#0 is actually %B after inplaceable results are swapped in the callee.
// CHECK-NEXT:   call @some_external_func(%[[B]]) : (memref<?xf32, #[[$DYN_1D_MAP]]>) -> ()
  call @some_external_func(%r0#0) : (tensor<?xf32>) -> ()

// CHECK-NEXT:   return
  return %r0#0, %r0#1: tensor<?xf32>, tensor<?xf32>
}

// -----

//  CHECK-DAG: #[[$DYN_0D_MAP:.*]] = affine_map<()[s0] -> (s0)>
//  CHECK-DAG: #[[$DYN_1D_MAP:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

//      CHECK:  func @init_and_dot(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<64xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]*]]: memref<64xf32, #[[$DYN_1D_MAP]]>
// CHECK-SAME:    %[[C:[a-zA-Z0-9]*]]: memref<f32, #[[$DYN_0D_MAP]]>
func @init_and_dot(%a: tensor<64xf32>, %b: tensor<64xf32>, %c: tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT:   %[[C0:.*]] = constant 0{{.*}} : f32
  %v0 = constant 0.0 : f32

  // CHECK-NEXT:   linalg.fill(%[[C0]], %[[C]]) : f32, memref<f32, #[[$DYN_0D_MAP]]>
  %d = linalg.fill(%v0, %c) : f32, tensor<f32> -> tensor<f32>

  // CHECK-NEXT:   linalg.dot ins(%[[A]], %[[B]] : memref<64xf32, #[[$DYN_1D_MAP]]>, memref<64xf32, #[[$DYN_1D_MAP]]>) outs(%[[C]] : memref<f32, #[[$DYN_0D_MAP]]>)
  %e = linalg.dot ins(%a, %b : tensor<64xf32>,tensor<64xf32>)
    outs(%d: tensor<f32>) -> tensor<f32>

  // CHECK-NEXT:   return
  return %e : tensor<f32>
}

//      CHECK:  func @main()
func @main() {
  //  CHECK-DAG:   %[[C0:.*]] = constant 0{{.*}} : f32
  //  CHECK-DAG:   %[[C1:.*]] = constant 1{{.*}} : f32
  //  CHECK-DAG:   %[[C2:.*]] = constant 2{{.*}} : f32
  %v0 = constant 0.0 : f32
  %v1 = constant 1.0 : f32
  %v2 = constant 2.0 : f32

  // CHECK-NEXT:   %[[A:.*]] = memref.alloc() : memref<64xf32>
  // CHECK-NEXT:   %[[B:.*]] = memref.alloc() : memref<64xf32>
  // CHECK-NEXT:   %[[C:.*]] = memref.alloc() : memref<f32>
  %A = linalg.init_tensor [64] : tensor<64xf32>
  %B = linalg.init_tensor [64] : tensor<64xf32>
  %C = linalg.init_tensor [] : tensor<f32>

  // CHECK-NEXT:   linalg.fill(%[[C1]], %[[A]]) : f32, memref<64xf32>
  // CHECK-NEXT:   linalg.fill(%[[C2]], %[[B]]) : f32, memref<64xf32>
  // CHECK-NEXT:   linalg.fill(%[[C0]], %[[C]]) : f32, memref<f32>
  %AA = linalg.fill(%v1, %A) : f32, tensor<64xf32> -> tensor<64xf32>
  %BB = linalg.fill(%v2, %B) : f32, tensor<64xf32> -> tensor<64xf32>
  %CC = linalg.fill(%v0, %C) : f32, tensor<f32> -> tensor<f32>

  // CHECK-NEXT:   %[[cA:.*]] = memref.cast %[[A]] : memref<64xf32> to memref<64xf32, #[[$DYN_1D_MAP]]>
  // CHECK-NEXT:   %[[cB:.*]] = memref.cast %[[B]] : memref<64xf32> to memref<64xf32, #[[$DYN_1D_MAP]]>
  // CHECK-NEXT:   %[[cC:.*]] = memref.cast %[[C]] : memref<f32> to memref<f32, #[[$DYN_0D_MAP]]>
  // CHECK-NEXT:   call @init_and_dot(%[[cA]], %[[cB]], %[[cC]])
  %res = call @init_and_dot(%AA, %BB, %CC) :
    (tensor<64xf32>, tensor<64xf32>, tensor<f32>) -> tensor<f32>

  // CHECK-NEXT:   %[[dC:.*]] = memref.cast %[[C]] : memref<f32> to memref<*xf32>
  %res2 = tensor.cast %res: tensor<f32> to tensor<*xf32>

  // CHECK-NEXT:   call @print_memref_f32(%[[dC]]) : (memref<*xf32>) -> ()
  call @print_memref_f32(%res2) : (tensor<*xf32>) -> ()

  // CHECK-DAG:   memref.dealloc %[[A]] : memref<64xf32>
  // CHECK-DAG:   memref.dealloc %[[B]] : memref<64xf32>
  // CHECK-DAG:   memref.dealloc %[[C]] : memref<f32>
  // CHECK-NEXT:   return
  return
}

//     CHECK:   func private @print_memref_f32(memref<*xf32>)
func private @print_memref_f32(tensor<*xf32>)
