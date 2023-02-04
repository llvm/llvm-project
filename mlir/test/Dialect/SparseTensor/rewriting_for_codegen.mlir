// RUN: mlir-opt %s -post-sparsification-rewrite="enable-runtime-library=false enable-convert=false" | \
// RUN: FileCheck %s

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

#CSC = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i, j) -> (j, i)>
}>

// CHECK-LABEL:   func.func @sparse_new_symmetry(
// CHECK-SAME:    %[[A:.*]]: !llvm.ptr<i8>) -> tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> {
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[R:.*]] = call @createSparseTensorReader(%[[A]])
// CHECK:         %[[DS:.*]] = memref.alloca(%[[C2]]) : memref<?xindex>
// CHECK:         call @copySparseTensorReaderDimSizes(%[[R]], %[[DS]])
// CHECK:         %[[D0:.*]] = memref.load %[[DS]]{{\[}}%[[C0]]]
// CHECK:         %[[D1:.*]] = memref.load %[[DS]]{{\[}}%[[C1]]]
// CHECK:         %[[N:.*]] = call @getSparseTensorReaderNNZ(%[[R]])
// CHECK:         %[[T:.*]] = bufferization.alloc_tensor(%[[D0]], %[[D1]]) size_hint=%[[N]]
// CHECK:         %[[S:.*]] = call @getSparseTensorReaderIsSymmetric(%[[R]])
// CHECK:         %[[VB:.*]] = memref.alloca()
// CHECK:         %[[T2:.*]] = scf.for %{{.*}} = %[[C0]] to %[[N]] step %[[C1]] iter_args(%[[A2:.*]] = %[[T]])
// CHECK:           func.call @getSparseTensorReaderNextF32(%[[R]], %[[DS]], %[[VB]])
// CHECK:           %[[E0:.*]] = memref.load %[[DS]]{{\[}}%[[C0]]]
// CHECK:           %[[E1:.*]] = memref.load %[[DS]]{{\[}}%[[C1]]]
// CHECK:           %[[V:.*]] = memref.load %[[VB]][]
// CHECK:           %[[T1:.*]] = sparse_tensor.insert %[[V]] into %[[A2]]{{\[}}%[[E0]], %[[E1]]]
// CHECK:           %[[NE:.*]] = arith.cmpi ne, %[[E0]], %[[E1]]
// CHECK:           %[[COND:.*]] = arith.andi %[[S]], %[[NE]]
// CHECK:           %[[T3:.*]] = scf.if %[[COND]]
// CHECK:             %[[T4:.*]] = sparse_tensor.insert %[[V]] into %[[T1]]{{\[}}%[[E1]], %[[E0]]]
// CHECK:             scf.yield %[[T4]]
// CHECK:           else
// CHECK:             scf.yield %[[T1]]
// CHECK:           scf.yield %[[T3]]
// CHECK:         }
// CHECK:         call @delSparseTensorReader(%[[R]])
// CHECK:         %[[T5:.*]] = sparse_tensor.load %[[T2]] hasInserts
// CHECK:         %[[R:.*]] = sparse_tensor.convert %[[T5]]
// CHECK:         bufferization.dealloc_tensor %[[T5]]
// CHECK:         return %[[R]]
func.func @sparse_new_symmetry(%arg0: !llvm.ptr<i8>) -> tensor<?x?xf32, #CSR> {
  %0 = sparse_tensor.new expand_symmetry %arg0 : !llvm.ptr<i8> to tensor<?x?xf32, #CSR>
  return %0 : tensor<?x?xf32, #CSR>
}

// CHECK-LABEL:   func.func @sparse_new(
// CHECK-SAME:    %[[A:.*]]: !llvm.ptr<i8>) -> tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> {
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[R:.*]] = call @createSparseTensorReader(%[[A]])
// CHECK:         %[[DS:.*]] = memref.alloca(%[[C2]]) : memref<?xindex>
// CHECK:         call @copySparseTensorReaderDimSizes(%[[R]], %[[DS]])
// CHECK:         %[[D0:.*]] = memref.load %[[DS]]{{\[}}%[[C0]]]
// CHECK:         %[[D1:.*]] = memref.load %[[DS]]{{\[}}%[[C1]]]
// CHECK:         %[[N:.*]] = call @getSparseTensorReaderNNZ(%[[R]])
// CHECK:         %[[T:.*]] = bufferization.alloc_tensor(%[[D0]], %[[D1]]) size_hint=%[[N]]
// CHECK:         %[[VB:.*]] = memref.alloca()
// CHECK:         %[[T2:.*]] = scf.for %{{.*}} = %[[C0]] to %[[N]] step %[[C1]] iter_args(%[[A2:.*]] = %[[T]])
// CHECK:           func.call @getSparseTensorReaderNextF32(%[[R]], %[[DS]], %[[VB]])
// CHECK:           %[[E0:.*]] = memref.load %[[DS]]{{\[}}%[[C0]]]
// CHECK:           %[[E1:.*]] = memref.load %[[DS]]{{\[}}%[[C1]]]
// CHECK:           %[[V:.*]] = memref.load %[[VB]][]
// CHECK:           %[[T1:.*]] = sparse_tensor.insert %[[V]] into %[[A2]]{{\[}}%[[E0]], %[[E1]]]
// CHECK:           scf.yield %[[T1]]
// CHECK:         }
// CHECK:         call @delSparseTensorReader(%[[R]])
// CHECK:         %[[T4:.*]] = sparse_tensor.load %[[T2]] hasInserts
// CHECK:         %[[R:.*]] = sparse_tensor.convert %[[T4]]
// CHECK:         bufferization.dealloc_tensor %[[T4]]
// CHECK:         return %[[R]]
func.func @sparse_new(%arg0: !llvm.ptr<i8>) -> tensor<?x?xf32, #CSR> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<?x?xf32, #CSR>
  return %0 : tensor<?x?xf32, #CSR>
}

// CHECK-LABEL:   func.func @sparse_new_csc(
// CHECK-SAME:    %[[A:.*]]: !llvm.ptr<i8>) -> tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)> }>> {
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[R:.*]] = call @createSparseTensorReader(%[[A]])
// CHECK:         %[[DS:.*]] = memref.alloca(%[[C2]]) : memref<?xindex>
// CHECK:         call @copySparseTensorReaderDimSizes(%[[R]], %[[DS]])
// CHECK:         %[[D0:.*]] = memref.load %[[DS]]{{\[}}%[[C0]]]
// CHECK:         %[[D1:.*]] = memref.load %[[DS]]{{\[}}%[[C1]]]
// CHECK:         %[[N:.*]] = call @getSparseTensorReaderNNZ(%[[R]])
// CHECK:         %[[T:.*]] = bufferization.alloc_tensor(%[[D0]], %[[D1]]) size_hint=%[[N]]
// CHECK:         %[[VB:.*]] = memref.alloca()
// CHECK:         %[[T2:.*]] = scf.for %{{.*}} = %[[C0]] to %[[N]] step %[[C1]] iter_args(%[[A2:.*]] = %[[T]])
// CHECK:           func.call @getSparseTensorReaderNextF32(%[[R]], %[[DS]], %[[VB]])
// CHECK:           %[[E0:.*]] = memref.load %[[DS]]{{\[}}%[[C0]]]
// CHECK:           %[[E1:.*]] = memref.load %[[DS]]{{\[}}%[[C1]]]
// CHECK:           %[[V:.*]] = memref.load %[[VB]][]
// CHECK:           %[[T1:.*]] = sparse_tensor.insert %[[V]] into %[[A2]]{{\[}}%[[E1]], %[[E0]]]
// CHECK:           scf.yield %[[T1]]
// CHECK:         }
// CHECK:         call @delSparseTensorReader(%[[R]])
// CHECK:         %[[T4:.*]] = sparse_tensor.load %[[T2]] hasInserts
// CHECK:         %[[R:.*]] = sparse_tensor.convert %[[T4]]
// CHECK:         bufferization.dealloc_tensor %[[T4]]
// CHECK:         return %[[R]]
func.func @sparse_new_csc(%arg0: !llvm.ptr<i8>) -> tensor<?x?xf32, #CSC> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<?x?xf32, #CSC>
  return %0 : tensor<?x?xf32, #CSC>
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
