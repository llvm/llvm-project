// RUN: mlir-opt %s -post-sparsification-rewrite="enable-runtime-library=false enable-convert=false" | \
// RUN: FileCheck %s

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

#COO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)
}>

// CHECK-LABEL:   func.func @sparse_new(
// CHECK-SAME:    %[[A:.*]]: !llvm.ptr<i8>) -> tensor<?x?xf32, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:         %[[COO:.*]] = sparse_tensor.new %[[A]] : !llvm.ptr<i8> to tensor<?x?xf32, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK:         %[[R:.*]] = sparse_tensor.convert %[[COO]]
// CHECK:         bufferization.dealloc_tensor %[[COO]]
// CHECK:         return %[[R]]
func.func @sparse_new(%arg0: !llvm.ptr<i8>) -> tensor<?x?xf32, #CSR> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<?x?xf32, #CSR>
  return %0 : tensor<?x?xf32, #CSR>
}

// CHECK-LABEL:   func.func @sparse_new_csc(
// CHECK-SAME:    %[[A:.*]]: !llvm.ptr<i8>) -> tensor<?x?xf32, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:         %[[COO:.*]] = sparse_tensor.new %[[A]] : !llvm.ptr<i8> to tensor<?x?xf32, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK:         %[[R:.*]] = sparse_tensor.convert %[[COO]]
// CHECK:         bufferization.dealloc_tensor %[[COO]]
// CHECK:         return %[[R]]
func.func @sparse_new_csc(%arg0: !llvm.ptr<i8>) -> tensor<?x?xf32, #CSC> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<?x?xf32, #CSC>
  return %0 : tensor<?x?xf32, #CSC>
}

// CHECK-LABEL:   func.func @sparse_new_coo(
// CHECK-SAME:    %[[A:.*]]: !llvm.ptr<i8>) -> tensor<?x?xf32, #sparse_tensor.encoding<{{{.*}}}>> {
// CHECK:         %[[COO:.*]] = sparse_tensor.new %[[A]] : !llvm.ptr<i8> to tensor<?x?xf32, #sparse_tensor.encoding<{{{.*}}}>>
// CHECK:         return %[[COO]]
func.func @sparse_new_coo(%arg0: !llvm.ptr<i8>) -> tensor<?x?xf32, #COO> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<?x?xf32, #COO>
  return %0 : tensor<?x?xf32, #COO>
}

// CHECK-LABEL:   func.func @sparse_out(
// CHECK-SAME:    %[[A:.*]]: tensor<10x20xf32, #sparse_tensor.encoding<{{{.*}}}>>,
// CHECK-SAME:    %[[B:.*]]: !llvm.ptr<i8>) {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG:     %[[C20:.*]] = arith.constant 20 : index
// CHECK:         %[[NNZ:.*]] = sparse_tensor.number_of_entries %[[A]]
// CHECK:         %[[DS:.*]] = memref.alloca(%[[C2]]) : memref<?xindex>
// CHECK:         memref.store %[[C10]], %[[DS]]{{\[}}%[[C0]]] : memref<?xindex>
// CHECK:         memref.store %[[C20]], %[[DS]]{{\[}}%[[C1]]] : memref<?xindex>
// CHECK:         %[[W:.*]] = call @createSparseTensorWriter(%[[B]])
// CHECK:         call @outSparseTensorWriterMetaData(%[[W]], %[[C2]], %[[NNZ]], %[[DS]])
// CHECK:         %[[V:.*]] = memref.alloca() : memref<f32>
// CHECK:         scf.for  %{{.*}} = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK:           scf.for  {{.*}} {
// CHECK:             func.call @outSparseTensorWriterNextF32(%[[W]], %[[C2]], %[[DS]], %[[V]])
// CHECK:           }
// CHECK:         }
// CHECK:         call @delSparseTensorWriter(%[[W]])
// CHECK:         return
// CHECK:         }
func.func @sparse_out( %arg0: tensor<10x20xf32, #CSR>, %arg1: !llvm.ptr<i8>) -> () {
  sparse_tensor.out %arg0, %arg1 : tensor<10x20xf32, #CSR>, !llvm.ptr<i8>
  return
}
