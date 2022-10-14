// RUN: mlir-opt %s -sparse-tensor-rewrite=enable-runtime-library=false  | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

// CHECK-LABEL:   func.func @sparse_new(
// CHECK-SAME:    %[[A:.*]]: !llvm.ptr<i8>) -> tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> {
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[R:.*]] = call @createSparseTensorReader(%[[A]])
// CHECK:         %[[DS:.*]] = memref.alloc(%[[C2]]) : memref<?xindex>
// CHECK:         call @getSparseTensorReaderDimSizes(%[[R]], %[[DS]])
// CHECK:         %[[D0:.*]] = memref.load %[[DS]]{{\[}}%[[C0]]]
// CHECK:         %[[D1:.*]] = memref.load %[[DS]]{{\[}}%[[C1]]]
// CHECK:         %[[T:.*]] = bufferization.alloc_tensor(%[[D0]], %[[D1]])
// CHECK:         %[[N:.*]] = call @getSparseTensorReaderNNZ(%[[R]])
// CHECK:         scf.for %{{.*}} = %[[C0]] to %[[N]] step %[[C1]] {
// CHECK:           %[[V:.*]] = func.call @getSparseTensorReaderNextF32(%[[R]], %[[DS]])
// CHECK:           %[[E0:.*]] = memref.load %[[DS]]{{\[}}%[[C0]]]
// CHECK:           %[[E1:.*]] = memref.load %[[DS]]{{\[}}%[[C1]]]
// CHECK:           sparse_tensor.insert %[[V]] into %[[T]]{{\[}}%[[E0]], %[[E1]]]
// CHECK:         }
// CHECK:         memref.dealloc %[[DS]]
// CHECK:         call @delSparseTensorReader(%[[R]])
// CHECK:         %[[R:.*]] = sparse_tensor.convert %[[T]]
// CHECK:         bufferization.dealloc_tensor %[[T]]
// CHECK:         return %[[R]]
// CHECK:         }
func.func @sparse_new(%arg0: !llvm.ptr<i8>) -> tensor<?x?xf32, #CSR> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<?x?xf32, #CSR>
  return %0 : tensor<?x?xf32, #CSR>
}
