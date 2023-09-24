// RUN: mlir-opt %s --post-sparsification-rewrite="enable-runtime-library=false" \
// RUN:    --sparse-tensor-codegen=create-sparse-deallocs=false \
// RUN:    --canonicalize --cse | FileCheck %s -check-prefix=CHECK-NO-DEALLOC

// RUN: mlir-opt %s --post-sparsification-rewrite="enable-runtime-library=false" \
// RUN:    --sparse-tensor-codegen=create-sparse-deallocs=true \
// RUN:    --canonicalize --cse | FileCheck %s -check-prefix=CHECK-DEALLOC

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed)}>
#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed),
}>

//
// No memref.dealloc is user-requested so
// CHECK-NO-DEALLOC-LABEL: @sparse_convert_permuted
// CHECK-NO-DEALLOC-NOT: memref.dealloc
//
// Otherwise memref.dealloc is created to free temporary sparse buffers.
// CHECK-DEALLOC-LABEL: @sparse_convert_permuted
// CHECK-DEALLOC: memref.dealloc
//
func.func @sparse_convert_permuted(%arg0: tensor<?x?xf32, #CSR>) -> tensor<?x?xf32, #CSC> {
  %0 = sparse_tensor.convert %arg0 : tensor<?x?xf32, #CSR> to tensor<?x?xf32, #CSC>
  return %0 : tensor<?x?xf32, #CSC>
}
