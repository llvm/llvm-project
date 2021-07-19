// RUN: mlir-opt %s -canonicalize -cse -linalg-comprehensive-module-bufferize |\
// RUN: mlir-opt -convert-vector-to-scf -lower-affine -convert-linalg-to-loops |\
// RUN: mlir-opt -canonicalize -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm | \

// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext |\
// RUN: FileCheck %s

#map0 = affine_map<(d0, d1)[s0] -> ((d1 - d0) ceildiv s0)>
#map1 = affine_map<(d0, d1)[s0] -> ((d0 - d1) ceildiv s0)>

func @init_and_dot(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<f32> {linalg.inplaceable = true}) -> tensor<f32> {
  %c64 = constant 64 : index
  %cst = constant 0.000000e+00 : f32
  %c2 = constant 2 : index
  %c0 = constant 0 : index
  %0 = linalg.fill(%cst, %arg2) : f32, tensor<f32> -> tensor<f32>
  %1 = affine.apply #map0(%c0, %c64)[%c2]
  %2 = linalg.init_tensor [%1, 2] : tensor<?x2xf32>
  %3 = scf.for %arg3 = %c0 to %c64 step %c2 iter_args(%arg4 = %2) -> (tensor<?x2xf32>) {
    %8 = affine.apply #map1(%arg3, %c0)[%c2]
    %9 = tensor.extract_slice %arg1[%arg3] [2] [1] : tensor<64xf32> to tensor<2xf32>
    %10 = tensor.cast %9 : tensor<2xf32> to tensor<?xf32>
    %11 = linalg.pad_tensor %10 low[%c0] high[%c0]  {
    ^bb0(%arg5: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<?xf32> to tensor<2xf32>
    %12 = tensor.insert_slice %11 into %arg4[%8, 0] [1, 2] [1, 1] : tensor<2xf32> into tensor<?x2xf32>
    scf.yield %12 : tensor<?x2xf32>
  }

  // %B = tensor.cast %3 : tensor<?x2xf32> to tensor<*xf32>
  // call @print_memref_f32(%B) : (tensor<*xf32>) -> ()

  %4 = affine.apply #map0(%c0, %c64)[%c2]
  %5 = linalg.init_tensor [%4, 2] : tensor<?x2xf32>
  %6 = scf.for %arg3 = %c0 to %c64 step %c2 iter_args(%arg4 = %5) -> (tensor<?x2xf32>) {
    %8 = affine.apply #map1(%arg3, %c0)[%c2]
    %9 = tensor.extract_slice %arg0[%arg3] [2] [1] : tensor<64xf32> to tensor<2xf32>
    %10 = tensor.cast %9 : tensor<2xf32> to tensor<?xf32>
    %11 = linalg.pad_tensor %10 low[%c0] high[%c0]  {
    ^bb0(%arg5: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<?xf32> to tensor<2xf32>
    %12 = tensor.insert_slice %11 into %arg4[%8, 0] [1, 2] [1, 1] : tensor<2xf32> into tensor<?x2xf32>
    scf.yield %12 : tensor<?x2xf32>
  }

  // %A = tensor.cast %6 : tensor<?x2xf32> to tensor<*xf32>
  // call @print_memref_f32(%A) : (tensor<*xf32>) -> ()

  // %C = tensor.cast %0 : tensor<f32> to tensor<*xf32>
  // call @print_memref_f32(%C) : (tensor<*xf32>) -> ()

  %7 = scf.for %arg3 = %c0 to %c64 step %c2 iter_args(%arg4 = %0) -> (tensor<f32>) {
    %8 = tensor.extract_slice %arg0[%arg3] [2] [1] : tensor<64xf32> to tensor<2xf32>
    %9 = tensor.cast %8 : tensor<2xf32> to tensor<?xf32>
    %10 = tensor.extract_slice %arg1[%arg3] [2] [1] : tensor<64xf32> to tensor<2xf32>
    %11 = tensor.cast %10 : tensor<2xf32> to tensor<?xf32>
    %12 = affine.apply #map1(%arg3, %c0)[%c2]
    %13 = tensor.extract_slice %6[%12, 0] [1, 2] [1, 1] : tensor<?x2xf32> to tensor<2xf32>
    %14 = affine.apply #map1(%arg3, %c0)[%c2]
    %15 = tensor.extract_slice %3[%14, 0] [1, 2] [1, 1] : tensor<?x2xf32> to tensor<2xf32>
    %16 = linalg.dot ins(%13, %15 : tensor<2xf32>, tensor<2xf32>) outs(%arg4 : tensor<f32>) -> tensor<f32>

    // %AA = tensor.cast %13 : tensor<2xf32> to tensor<*xf32>
    // call @print_memref_f32(%AA) : (tensor<*xf32>) -> ()
    // %BB = tensor.cast %15 : tensor<2xf32> to tensor<*xf32>
    // call @print_memref_f32(%BB) : (tensor<*xf32>) -> ()
    // %CC = tensor.cast %16 : tensor<f32> to tensor<*xf32>
    // call @print_memref_f32(%CC) : (tensor<*xf32>) -> ()

    scf.yield %16 : tensor<f32>
  }
  return %7 : tensor<f32>
}

func @main() {
  %v0 = constant 0.0 : f32
  %v1 = constant 1.0 : f32
  %v2 = constant 2.0 : f32

  %A = linalg.init_tensor [64] : tensor<64xf32>
  %B = linalg.init_tensor [64] : tensor<64xf32>
  %C = linalg.init_tensor [] : tensor<f32>
  %AA = linalg.fill(%v1, %A) : f32, tensor<64xf32> -> tensor<64xf32>
  %BB = linalg.fill(%v2, %B) : f32, tensor<64xf32> -> tensor<64xf32>
  %CC = linalg.fill(%v0, %C) : f32, tensor<f32> -> tensor<f32>

  %res = call @init_and_dot(%AA, %BB, %CC) :
    (tensor<64xf32>, tensor<64xf32>, tensor<f32>) -> tensor<f32>

  %res2 = tensor.cast %res: tensor<f32> to tensor<*xf32>

//      CHECK: Unranked Memref base@ = {{.*}} rank = 0 offset = 0 sizes = [] strides = [] data =
// CHECK-NEXT: [128]
  call @print_memref_f32(%res2) : (tensor<*xf32>) -> ()

  return
}

func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }
