// RUN: mlir-opt %s -one-shot-bufferize="allow-unknown-ops" -verify-diagnostics -split-input-file | FileCheck %s

func.func @test(%14: index, %0 : memref<8x16xf16>, %1 : memref<8xi32>, %2 : memref<?x16xf16>) {
  %16 = bufferization.to_tensor %0 restrict : memref<8x16xf16> to tensor<8x16xf16>
  %17 = bufferization.to_tensor %1 restrict : memref<8xi32> to tensor<8xi32>
  %18 = bufferization.to_tensor %2 restrict : memref<?x16xf16> to tensor<?x16xf16>
  %cst = arith.constant 123.4 : f32
  %19 = scf.forall (%arg0) in (2) shared_outs(%arg1 = %18) -> (tensor<?x16xf16>) {
    %20 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg0)
    %extracted_slice = tensor.extract_slice %arg1[0, %20] [%14, 8] [1, 1] : tensor<?x16xf16> to tensor<?x8xf16>
    %21 = scf.forall (%arg2, %arg3) in (8, 1) shared_outs(%arg4 = %extracted_slice) -> (tensor<?x8xf16>) {
      %extracted_slice_0 = tensor.extract_slice %16[%arg2, %20] [1, 8] [1, 1] : tensor<8x16xf16> to tensor<1x8xf16>
      %extracted_slice_1 = tensor.extract_slice %17[%arg2] [1] [1] : tensor<8xi32> to tensor<1xi32>
      %22 = linalg.fill ins(%cst : f32) outs(%arg4 : tensor<?x8xf16>) -> tensor<?x8xf16>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %22 into %arg4[0, 0] [%14, 8] [1, 1] : tensor<?x8xf16> into tensor<?x8xf16>
      }
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %21 into %arg1[0, %20] [%14, 8] [1, 1] : tensor<?x8xf16> into tensor<?x16xf16>
    }
  }
  bufferization.materialize_in_destination
    %19 in restrict writable %2 : (tensor<?x16xf16>, memref<?x16xf16>) -> ()
  return
}
