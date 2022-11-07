// RUN: mlir-opt %s -sparse-tensor-rewrite="enable-runtime-library=false enable-convert=false" |\
// RUN: FileCheck %s

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

// CHECK-LABEL:   func.func @sparse_new(
// CHECK-SAME:    %[[A:.*]]: !llvm.ptr<i8>) -> tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> {
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[R:.*]] = call @createSparseTensorReader(%[[A]])
// CHECK:         %[[DS:.*]] = memref.alloca(%[[C2]]) : memref<?xindex>
// CHECK:         call @getSparseTensorReaderDimSizes(%[[R]], %[[DS]])
// CHECK:         %[[D0:.*]] = memref.load %[[DS]]{{\[}}%[[C0]]]
// CHECK:         %[[D1:.*]] = memref.load %[[DS]]{{\[}}%[[C1]]]
// CHECK:         %[[T:.*]] = bufferization.alloc_tensor(%[[D0]], %[[D1]])
// CHECK:         %[[N:.*]] = call @getSparseTensorReaderNNZ(%[[R]])
// CHECK:         %[[VB:.*]] = memref.alloca()
// CHECK:         scf.for %{{.*}} = %[[C0]] to %[[N]] step %[[C1]] {
// CHECK:           func.call @getSparseTensorReaderNextF32(%[[R]], %[[DS]], %[[VB]])
// CHECK:           %[[E0:.*]] = memref.load %[[DS]]{{\[}}%[[C0]]]
// CHECK:           %[[E1:.*]] = memref.load %[[DS]]{{\[}}%[[C1]]]
// CHECK:           %[[V:.*]] = memref.load %[[VB]][]
// CHECK:           sparse_tensor.insert %[[V]] into %[[T]]{{\[}}%[[E0]], %[[E1]]]
// CHECK:         }
// CHECK:         call @delSparseTensorReader(%[[R]])
// CHECK:         %[[R:.*]] = sparse_tensor.convert %[[T]]
// CHECK:         bufferization.dealloc_tensor %[[T]]
// CHECK:         return %[[R]]
// CHECK:         }
func.func @sparse_new(%arg0: !llvm.ptr<i8>) -> tensor<?x?xf32, #CSR> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<?x?xf32, #CSR>
  return %0 : tensor<?x?xf32, #CSR>
}

// CHECK-LABEL:   func.func @sparse_out(
// CHECK-SAME:    %[[A:.*]]: tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>,
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
