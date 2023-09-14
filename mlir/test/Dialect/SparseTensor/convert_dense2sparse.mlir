// RUN: mlir-opt %s --sparse-tensor-conversion --canonicalize --cse | FileCheck %s
// RUN: mlir-opt %s --post-sparsification-rewrite="enable-runtime-library=false enable-foreach=false" \
// RUN: --canonicalize --cse | FileCheck %s --check-prefix=CHECK-RWT

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

#CSR = #sparse_tensor.encoding<{
  lvlTypes = ["dense", "compressed"]
}>

#CSC = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ],
  dimToLvl = affine_map<(i, j) -> (j, i)>
}>

#SparseTensor = #sparse_tensor.encoding<{
  lvlTypes = ["dense", "compressed", "compressed"],
  dimToLvl = affine_map<(i,j,k) -> (k,i,j)>
}>

// CHECK-LABEL: func @sparse_convert_1d(
//  CHECK-SAME: %[[A:.*]]: tensor<?xi32>) -> !llvm.ptr<i8> {
//   CHECK-DAG: %[[EmptyCOO:.*]] = arith.constant 4 : i32
//   CHECK-DAG: %[[FromCOO:.*]] = arith.constant 2 : i32
//   CHECK-DAG: %[[I0:.*]] = arith.constant 0 : i32
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[U:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?xi32>
//   CHECK-DAG: %[[LvlTypes:.*]] = memref.alloca() : memref<1xi8>
//   CHECK-DAG: %[[DimSizes:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[LvlSizes:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[Iota:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[LvlTypesP:.*]] = memref.cast %[[LvlTypes]] : memref<1xi8> to memref<?xi8>
//   CHECK-DAG: %[[DimSizesP:.*]] = memref.cast %[[DimSizes]] : memref<1xindex> to memref<?xindex>
//   CHECK-DAG: %[[LvlSizesP:.*]] = memref.cast %[[LvlSizes]] : memref<1xindex> to memref<?xindex>
//   CHECK-DAG: %[[IotaP:.*]] = memref.cast %[[Iota]] : memref<1xindex> to memref<?xindex>
//       CHECK: %[[NP:.*]] = llvm.mlir.null : !llvm.ptr<i8>
//       CHECK: %[[C:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[IotaP]], %[[IotaP]], %{{.*}}, %{{.*}}, %{{.*}}, %[[EmptyCOO]], %[[NP]])
//       CHECK: %[[M:.*]] = memref.alloca() : memref<1xindex>
//       CHECK: %[[T:.*]] = memref.cast %[[M]] : memref<1xindex> to memref<?xindex>
//       CHECK: %[[BUF:.*]] = memref.alloca() : memref<i32>
//       CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[U]] step %[[C1]] {
//       CHECK:   %[[E:.*]] = tensor.extract %[[A]][%[[I]]] : tensor<?xi32>
//       CHECK:   %[[N:.*]] = arith.cmpi ne, %[[E]], %[[I0]] : i32
//       CHECK:   scf.if %[[N]] {
//       CHECK:     memref.store %[[I]], %[[M]][%[[C0]]] : memref<1xindex>
//       CHECK:     memref.store %[[E]], %[[BUF]][] : memref<i32>
//       CHECK:     call @addEltI32(%[[C]], %[[BUF]], %[[T]], %[[IotaP]])
//       CHECK:   }
//       CHECK: }
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[IotaP]], %[[IotaP]], %{{.*}}, %{{.*}}, %{{.*}}, %[[FromCOO]], %[[C]])
//       CHECK: call @delSparseTensorCOOI32(%[[C]])
//       CHECK: return %[[T]] : !llvm.ptr<i8>
func.func @sparse_convert_1d(%arg0: tensor<?xi32>) -> tensor<?xi32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xi32> to tensor<?xi32, #SparseVector>
  return %0 : tensor<?xi32, #SparseVector>
}

// CHECK-LABEL: func @sparse_convert_complex(
//  CHECK-SAME: %[[A:.*]]: tensor<100xcomplex<f64>>) -> !llvm.ptr<i8> {
//   CHECK-DAG: %[[CC:.*]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C100:.*]] = arith.constant 100 : index
//       CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C100]] step %[[C1]] {
//       CHECK:   %[[E:.*]] = tensor.extract %[[A]][%[[I]]] : tensor<100xcomplex<f64>>
//       CHECK:   %[[N:.*]] = complex.neq %[[E]], %[[CC]] : complex<f64>
//       CHECK:   scf.if %[[N]] {
//       CHECK:     memref.store %[[I]], %{{.*}}[%[[C0]]] : memref<1xindex>
//       CHECK:     call @addEltC64
//       CHECK:   }
//       CHECK: }
//       CHECK: %[[T:.*]] = call @newSparseTensor
//       CHECK: call @delSparseTensorCOOC64
//       CHECK: return %[[T]] : !llvm.ptr<i8>
func.func @sparse_convert_complex(%arg0: tensor<100xcomplex<f64>>) -> tensor<100xcomplex<f64>, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<100xcomplex<f64>> to tensor<100xcomplex<f64>, #SparseVector>
  return %0 : tensor<100xcomplex<f64>, #SparseVector>
}

// CHECK-LABEL: func @sparse_convert_2d(
//  CHECK-SAME: %[[A:.*]]: tensor<2x4xf64>) -> !llvm.ptr<i8>
//   CHECK-DAG: %[[EmptyCOO:.*]] = arith.constant 4 : i32
//   CHECK-DAG: %[[FromCOO:.*]] = arith.constant 2 : i32
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[LvlTypes:.*]] = memref.alloca() : memref<2xi8>
//   CHECK-DAG: %[[DimSizes:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[LvlSizes:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[Iota:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[LvlTypesP:.*]] = memref.cast %[[LvlTypes]] : memref<2xi8> to memref<?xi8>
//   CHECK-DAG: %[[DimSizesP:.*]] = memref.cast %[[DimSizes]] : memref<2xindex> to memref<?xindex>
//   CHECK-DAG: %[[LvlSizesP:.*]] = memref.cast %[[LvlSizes]] : memref<2xindex> to memref<?xindex>
//   CHECK-DAG: %[[IotaP:.*]] = memref.cast %[[Iota]] : memref<2xindex> to memref<?xindex>
//       CHECK: %[[NP:.*]] = llvm.mlir.null : !llvm.ptr<i8>
//       CHECK: %[[C:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[IotaP]], %[[IotaP]], %{{.*}}, %{{.*}}, %{{.*}}, %[[EmptyCOO]], %[[NP]])
//       CHECK: %[[M:.*]] = memref.alloca() : memref<2xindex>
//       CHECK: %[[T:.*]] = memref.cast %[[M]] : memref<2xindex> to memref<?xindex>
//       CHECK: %[[BUF:.*]] = memref.alloca() : memref<f64>
//       CHECK: scf.for %[[I:.*]] = %[[C0]] to %{{.*}} step %[[C1]] {
//       CHECK:   scf.for %[[J:.*]] = %[[C0]] to %{{.*}} step %[[C1]] {
//       CHECK:     %[[E:.*]] = tensor.extract %[[A]][%[[I]], %[[J]]] : tensor<2x4xf64>
//       CHECK:     memref.store %[[I]], %[[M]][%[[C0]]] : memref<2xindex>
//       CHECK:     memref.store %[[J]], %[[M]][%[[C1]]] : memref<2xindex>
//       CHECK:     memref.store %[[E]], %[[BUF]][] : memref<f64>
//       CHECK:     call @addEltF64(%[[C]], %[[BUF]], %[[T]], %[[IotaP]])
//       CHECK:   }
//       CHECK: }
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[IotaP]], %[[IotaP]], %{{.*}}, %{{.*}}, %{{.*}}, %[[FromCOO]], %[[C]])
//       CHECK: call @delSparseTensorCOOF64(%[[C]])
//       CHECK: return %[[T]] : !llvm.ptr<i8>

// CHECK-RWT-LABEL:   func.func @sparse_convert_2d(
//  CHECK-RWT-SAME:     %[[T0:.*]]: tensor<2x4xf64>) -> tensor<2x4xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> {
//       CHECK-RWT:     %[[T1:.*]] = bufferization.alloc_tensor()
//       CHECK-RWT:     %[[T2:.*]] = sparse_tensor.foreach in %[[T0]] init(%[[T1]])
//       CHECK-RWT:     ^bb0(%[[L0I0:.*]]: index, %[[L0I1:.*]]: index, %[[L0V:.*]]: f64, %[[L0T:.*]]: tensor
//       CHECK-RWT:        %[[CMP:.*]] = arith.cmpf une, %[[L0V]]
//       CHECK-RWT:        %[[IFR:.*]] = scf.if %[[CMP]]
//       CHECK-RWT:          %[[L0T2:.*]] = sparse_tensor.insert %[[L0V]] into %[[L0T]]{{\[}}%[[L0I0]], %[[L0I1]]]
//       CHECK-RWT:          scf.yield %[[L0T2]]
//       CHECK-RWT:        } else {
//       CHECK-RWT:          scf.yield %[[L0T]]
//       CHECK-RWT:        }
//       CHECK-RWT:        sparse_tensor.yield %[[IFR]]
//       CHECK-RWT:     }
//       CHECK-RWT:     %[[R:.*]] = sparse_tensor.load %[[T2]] hasInserts
//       CHECK-RWT:     return %[[R]]
//       CHECK-RWT:   }
func.func @sparse_convert_2d(%arg0: tensor<2x4xf64>) -> tensor<2x4xf64, #CSR> {
  %0 = sparse_tensor.convert %arg0 : tensor<2x4xf64> to tensor<2x4xf64, #CSR>
  return %0 : tensor<2x4xf64, #CSR>
}

// CHECK-LABEL: func @sparse_constant() -> !llvm.ptr<i8> {
//   CHECK-DAG: %[[EmptyCOO:.*]] = arith.constant 4 : i32
//   CHECK-DAG: %[[FromCOO:.*]] = arith.constant 2 : i32
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[LvlTypes:.*]] = memref.alloca() : memref<2xi8>
//   CHECK-DAG: %[[DimSizes:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[LvlSizes:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[Iota:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[LvlTypesP:.*]] = memref.cast %[[LvlTypes]] : memref<2xi8> to memref<?xi8>
//   CHECK-DAG: %[[DimSizesP:.*]] = memref.cast %[[DimSizes]] : memref<2xindex> to memref<?xindex>
//   CHECK-DAG: %[[LvlSizesP:.*]] = memref.cast %[[LvlSizes]] : memref<2xindex> to memref<?xindex>
//   CHECK-DAG: %[[IotaP:.*]] = memref.cast %[[Iota]] : memref<2xindex> to memref<?xindex>
//       CHECK: %[[NP:.*]] = llvm.mlir.null : !llvm.ptr<i8>
//       CHECK: %[[C:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[IotaP]], %[[IotaP]], %{{.*}}, %{{.*}}, %{{.*}}, %[[EmptyCOO]], %[[NP]])
//       CHECK: %[[M:.*]] = memref.alloca() : memref<2xindex>
//       CHECK: %[[N:.*]] = memref.cast %[[M]] : memref<2xindex> to memref<?xindex>
//       CHECK: %[[BUF:.*]] = memref.alloca() : memref<f32>
//       CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
//   CHECK-DAG:   memref.store %{{.*}}, %[[M]][%[[C0]]] : memref<2xindex>
//   CHECK-DAG:   memref.store %{{.*}}, %[[M]][%[[C1]]] : memref<2xindex>
//   CHECK-DAG:   %[[V:.*]] = tensor.extract %{{.*}}[%[[I]]] : tensor<2xf32>
//       CHECK:   memref.store %[[V]], %[[BUF]][] : memref<f32>
//       CHECK:   call @addEltF32(%{{.*}}, %[[BUF]], %[[N]], %[[IotaP]])
//       CHECK: }
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[IotaP]], %[[IotaP]], %{{.*}}, %{{.*}}, %{{.*}}, %[[FromCOO]], %[[C]])
//       CHECK: call @delSparseTensorCOOF32(%[[C]])
//       CHECK: return %[[T]] : !llvm.ptr<i8>

// CHECK-RWT-LABEL:   func.func @sparse_constant() -> tensor<8x7xf32, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> {
//       CHECK-RWT:     %[[F0:.*]] = arith.constant sparse<{{\[\[}}0, 0], [1, 6]], [1.000000e+00, 5.000000e+00]> : tensor<8x7xf32>
//       CHECK-RWT:     %[[T0:.*]] = bufferization.alloc_tensor()
//       CHECK-RWT:     %[[T1:.*]] = sparse_tensor.foreach in %[[F0]] init(%[[T0]])
//       CHECK-RWT:     ^bb0(%[[L0I0:.*]]: index, %[[L0I1:.*]]: index, %[[L0V:.*]]: f32, %[[L0T:.*]]: tensor
//       CHECK-RWT:       %[[L0T2:.*]] = sparse_tensor.insert %[[L0V]] into %[[L0T]]{{\[}}%[[L0I0]], %[[L0I1]]]
//       CHECK-RWT:       sparse_tensor.yield %[[L0T2]]
//       CHECK-RWT:     }
//       CHECK-RWT:     %[[R:.*]] = sparse_tensor.load %[[T1]] hasInserts
//       CHECK-RWT:     return %[[R]]
//       CHECK-RWT:   }
func.func @sparse_constant() -> tensor<8x7xf32, #CSR>{
  // Initialize a tensor.
  %0 = arith.constant sparse<[[0, 0], [1, 6]], [1.0, 5.0]> : tensor<8x7xf32>
  // Convert the tensor to a sparse tensor.
  %1 = sparse_tensor.convert %0 : tensor<8x7xf32> to tensor<8x7xf32, #CSR>
  return %1 : tensor<8x7xf32, #CSR>
}

// CHECK-RWT-LABEL:   func.func @sparse_constant_csc() -> tensor<8x7xf32,
//       CHECK-RWT:    %[[VAL_0:.*]] = arith.constant sparse<{{\[\[}}0, 0], [1, 6]], [1.000000e+00, 5.000000e+00]> : tensor<8x7xf32>
//       CHECK-RWT:    %[[VAL_1:.*]] = bufferization.alloc_tensor() :
//       CHECK-RWT:    %[[VAL_2:.*]] = sparse_tensor.foreach in %[[VAL_0]] init(%[[VAL_1]]) {order = #map} : tensor<8x7xf32>,
//       CHECK-RWT:    ^bb0(%[[VAL_3:.*]]: index, %[[VAL_4:.*]]: index, %[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: tensor
//       CHECK-RWT:      %[[VAL_7:.*]] = sparse_tensor.insert %[[VAL_5]] into %[[VAL_6]]{{\[}}%[[VAL_4]], %[[VAL_3]]] :
//       CHECK-RWT:      sparse_tensor.yield %[[VAL_7]] :
//       CHECK-RWT:    }
//       CHECK-RWT:    %[[VAL_8:.*]] = sparse_tensor.load %[[VAL_9:.*]] hasInserts :
//       CHECK-RWT:    return %[[VAL_8]] :
//       CHECK-RWT:  }
func.func @sparse_constant_csc() -> tensor<8x7xf32, #CSC>{
  // Initialize a tensor.
  %0 = arith.constant sparse<[[0, 0], [1, 6]], [1.0, 5.0]> : tensor<8x7xf32>
  // Convert the tensor to a sparse tensor.
  %1 = sparse_tensor.convert %0 : tensor<8x7xf32> to tensor<8x7xf32, #CSC>
  return %1 : tensor<8x7xf32, #CSC>
}

// CHECK-LABEL: func @sparse_convert_3d(
//  CHECK-SAME: %[[A:.*]]: tensor<?x?x?xf64>) -> !llvm.ptr<i8>
//   CHECK-DAG: %[[EmptyCOO:.*]] = arith.constant 4 : i32
//   CHECK-DAG: %[[FromCOO:.*]] = arith.constant 2 : i32
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[U1:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?x?x?xf64>
//   CHECK-DAG: %[[U2:.*]] = tensor.dim %[[A]], %[[C1]] : tensor<?x?x?xf64>
//   CHECK-DAG: %[[U3:.*]] = tensor.dim %[[A]], %[[C2]] : tensor<?x?x?xf64>
//   CHECK-DAG: %[[LvlTypes:.*]] = memref.alloca() : memref<3xi8>
//   CHECK-DAG: %[[DimSizes:.*]] = memref.alloca() : memref<3xindex>
//   CHECK-DAG: %[[LvlSizes:.*]] = memref.alloca() : memref<3xindex>
//   CHECK-DAG: %[[Lvl2Dim:.*]] = memref.alloca() : memref<3xindex>
//   CHECK-DAG: %[[Dim2Lvl:.*]] = memref.alloca() : memref<3xindex>
//   CHECK-DAG: %[[LvlTypesP:.*]] = memref.cast %[[LvlTypes]] : memref<3xi8> to memref<?xi8>
//   CHECK-DAG: %[[DimSizesP:.*]] = memref.cast %[[DimSizes]] : memref<3xindex> to memref<?xindex>
//   CHECK-DAG: %[[LvlSizesP:.*]] = memref.cast %[[LvlSizes]] : memref<3xindex> to memref<?xindex>
//   CHECK-DAG: %[[Lvl2DimP:.*]] = memref.cast %[[Lvl2Dim]] : memref<3xindex> to memref<?xindex>
//   CHECK-DAG: %[[Dim2LvlP:.*]] = memref.cast %[[Dim2Lvl]] : memref<3xindex> to memref<?xindex>
//       CHECK: %[[NP:.*]] = llvm.mlir.null : !llvm.ptr<i8>
//       CHECK: %[[C:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[Lvl2DimP]], %[[Dim2LvlP]], %{{.*}}, %{{.*}}, %{{.*}}, %[[EmptyCOO]], %[[NP]])
//       CHECK: %[[M:.*]] = memref.alloca() : memref<3xindex>
//       CHECK: %[[N:.*]] = memref.cast %[[M]] : memref<3xindex> to memref<?xindex>
//       CHECK: %[[BUF:.*]] = memref.alloca() : memref<f64>
//       CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[U1]] step %[[C1]] {
//       CHECK:   scf.for %[[J:.*]] = %[[C0]] to %[[U2]] step %[[C1]] {
//       CHECK:     scf.for %[[K:.*]] = %[[C0]] to %[[U3]] step %[[C1]] {
//       CHECK:       %[[E:.*]] = tensor.extract %[[A]][%[[I]], %[[J]], %[[K]]] : tensor<?x?x?xf64>
//       CHECK:       memref.store %[[I]], %[[M]][%[[C0]]] : memref<3xindex>
//       CHECK:       memref.store %[[J]], %[[M]][%[[C1]]] : memref<3xindex>
//       CHECK:       memref.store %[[K]], %[[M]][%[[C2]]] : memref<3xindex>
//       CHECK:       memref.store %[[E]], %[[BUF]][] : memref<f64>
//       CHECK:       call @addEltF64(%[[C]], %[[BUF]], %[[N]], %[[Dim2LvlP]])
//       CHECK:     }
//       CHECK:   }
//       CHECK: }
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[DimSizesP]], %[[LvlSizesP]], %[[LvlTypesP]], %[[Lvl2DimP]], %[[Dim2LvlP]], %{{.*}}, %{{.*}}, %{{.*}}, %[[FromCOO]], %[[C]])
//       CHECK: call @delSparseTensorCOOF64(%[[C]])
//       CHECK: return %[[T]] : !llvm.ptr<i8>
func.func @sparse_convert_3d(%arg0: tensor<?x?x?xf64>) -> tensor<?x?x?xf64, #SparseTensor> {
  %0 = sparse_tensor.convert %arg0 : tensor<?x?x?xf64> to tensor<?x?x?xf64, #SparseTensor>
  return %0 : tensor<?x?x?xf64, #SparseTensor>
}
