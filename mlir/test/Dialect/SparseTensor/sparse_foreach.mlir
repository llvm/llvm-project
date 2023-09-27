// RUN: mlir-opt %s --post-sparsification-rewrite="enable-runtime-library=false enable-foreach=true" --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @sparse_foreach_constant
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[V1:.*]] = arith.constant 5.000000e+00 : f32
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[V3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:   %[[V4:.*]] = arith.constant 6.000000e+00 : f32
//               (1, 1) -> (2, 1) -> (2, 2)
// CHECK-NEXT:  "test.use"(%[[C1]], %[[C1]], %[[V1]])
// CHECK-NEXT:  "test.use"(%[[C2]], %[[C1]], %[[V3]])
// CHECK-NEXT:  "test.use"(%[[C1]], %[[C2]], %[[V4]])
//               (1, 1) -> (1, 2) -> (2, 1)
// CHECK-NEXT:  "test.use"(%[[C1]], %[[C1]], %[[V1]])
// CHECK-NEXT:  "test.use"(%[[C1]], %[[C2]], %[[V4]])
// CHECK-NEXT:  "test.use"(%[[C2]], %[[C1]], %[[V3]])
func.func @sparse_foreach_constant() -> () {
  %cst = arith.constant sparse<[[2, 1], [1, 1], [1, 2]], [1.0, 5.0, 6.0]> : tensor<8x7xf32>
  // Make use the sparse constant are properly sorted based on the requested order.
  sparse_tensor.foreach in %cst { order = affine_map<(d0, d1) -> (d1, d0)> } : tensor<8x7xf32> do {
  ^bb0(%arg0: index, %arg1: index, %arg2: f32):
    "test.use" (%arg0, %arg1, %arg2): (index,index,f32)->()
  }
  sparse_tensor.foreach in %cst : tensor<8x7xf32> do {
  ^bb0(%arg0: index, %arg1: index, %arg2: f32):
    "test.use" (%arg0, %arg1, %arg2): (index,index,f32)->()
  }
  return
}

#CSR_SLICE = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "compressed" ],
  dimSlices = [ (0, 4, 1), (2, 4, 1) ]
}>

#CSR_SLICE_DYN = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "compressed" ],
  dimSlices = [ (?, ?, ?), (?, ?, ?) ]
}>


// CHECK-LABEL:   func.func @foreach_print_slice_dyn(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tensor<?x?xf64,
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<?x?xf64,
// CHECK-DAG:       %[[VAL_4:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<?x?xf64,
// CHECK-DAG:       %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_1]] : tensor<?x?xf64,
// CHECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.slice.offset %[[VAL_0]] at 0 : tensor<?x?xf64,
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.slice.stride %[[VAL_0]] at 0 : tensor<?x?xf64,
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<?x?xf64,
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<?x?xf64,
// CHECK-DAG:       %[[VAL_10:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64,
// CHECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.slice.offset %[[VAL_0]] at 1 : tensor<?x?xf64,
// CHECK-DAG:       %[[VAL_12:.*]] = sparse_tensor.slice.stride %[[VAL_0]] at 1 : tensor<?x?xf64,
// CHECK-DAG:       %[[VAL_13:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64,
// CHECK:           %[[VAL_14:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_1]]] : memref<?xindex>
// CHECK:           %[[VAL_15:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_16:.*]] = %[[VAL_14]] to %[[VAL_15]] step %[[VAL_2]] {
// CHECK:             %[[VAL_17:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK:             %[[VAL_18:.*]] = arith.subi %[[VAL_17]], %[[VAL_6]] : index
// CHECK:             %[[VAL_19:.*]] = arith.remui %[[VAL_18]], %[[VAL_7]] : index
// CHECK:             %[[VAL_20:.*]] = arith.divui %[[VAL_18]], %[[VAL_7]] : index
// CHECK:             %[[VAL_21:.*]] = arith.cmpi uge, %[[VAL_17]], %[[VAL_6]] : index
// CHECK:             %[[VAL_22:.*]] = arith.cmpi ult, %[[VAL_20]], %[[VAL_5]] : index
// CHECK:             %[[VAL_23:.*]] = arith.cmpi eq, %[[VAL_19]], %[[VAL_1]] : index
// CHECK:             %[[VAL_24:.*]] = arith.andi %[[VAL_21]], %[[VAL_22]] : i1
// CHECK:             %[[VAL_25:.*]] = arith.andi %[[VAL_24]], %[[VAL_23]] : i1
// CHECK:             scf.if %[[VAL_25]] {
// CHECK:               %[[VAL_26:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_16]], %[[VAL_2]] : index
// CHECK:               %[[VAL_28:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_27]]] : memref<?xindex>
// CHECK:               scf.for %[[VAL_29:.*]] = %[[VAL_26]] to %[[VAL_28]] step %[[VAL_2]] {
// CHECK:                 %[[VAL_30:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_29]]] : memref<?xindex>
// CHECK:                 %[[VAL_31:.*]] = arith.subi %[[VAL_30]], %[[VAL_11]] : index
// CHECK:                 %[[VAL_32:.*]] = arith.remui %[[VAL_31]], %[[VAL_12]] : index
// CHECK:                 %[[VAL_33:.*]] = arith.divui %[[VAL_31]], %[[VAL_12]] : index
// CHECK:                 %[[VAL_34:.*]] = arith.cmpi uge, %[[VAL_30]], %[[VAL_11]] : index
// CHECK:                 %[[VAL_35:.*]] = arith.cmpi ult, %[[VAL_33]], %[[VAL_10]] : index
// CHECK:                 %[[VAL_36:.*]] = arith.cmpi eq, %[[VAL_32]], %[[VAL_1]] : index
// CHECK:                 %[[VAL_37:.*]] = arith.andi %[[VAL_34]], %[[VAL_35]] : i1
// CHECK:                 %[[VAL_38:.*]] = arith.andi %[[VAL_37]], %[[VAL_36]] : i1
// CHECK:                 scf.if %[[VAL_38]] {
// CHECK:                   %[[VAL_39:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_29]]] : memref<?xf64>
// CHECK:                   "test.use"(%[[VAL_39]]) : (f64) -> ()
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
//
func.func @foreach_print_slice_dyn(%A: tensor<?x?xf64, #CSR_SLICE_DYN>) {
  sparse_tensor.foreach in %A : tensor<?x?xf64, #CSR_SLICE_DYN> do {
  ^bb0(%1: index, %2: index, %v: f64) :
    "test.use" (%v) : (f64) -> ()
  }
  return
}

// CHECK-LABEL:   func.func @foreach_print_slice(
// CHECK-SAME:                                   %[[VAL_0:.*]]: tensor<4x4xf64,
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<4x4xf64,
// CHECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<4x4xf64,
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<4x4xf64,
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<4x4xf64,
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<4x4xf64,
// CHECK-DAG:       %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_12:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_4]] {
// CHECK:             %[[VAL_13:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_12]]] : memref<?xindex>
// CHECK:             %[[VAL_14:.*]] = arith.cmpi ult, %[[VAL_13]], %[[VAL_1]] : index
// CHECK:             scf.if %[[VAL_14]] {
// CHECK:               %[[VAL_15:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_12]]] : memref<?xindex>
// CHECK:               %[[VAL_16:.*]] = arith.addi %[[VAL_12]], %[[VAL_4]] : index
// CHECK:               %[[VAL_17:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK:               scf.for %[[VAL_18:.*]] = %[[VAL_15]] to %[[VAL_17]] step %[[VAL_4]] {
// CHECK:                 %[[VAL_19:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// CHECK:                 %[[VAL_20:.*]] = arith.subi %[[VAL_19]], %[[VAL_2]] : index
// CHECK:                 %[[VAL_21:.*]] = arith.cmpi uge, %[[VAL_19]], %[[VAL_2]] : index
// CHECK:                 %[[VAL_22:.*]] = arith.cmpi ult, %[[VAL_20]], %[[VAL_1]] : index
// CHECK:                 %[[VAL_23:.*]] = arith.andi %[[VAL_21]], %[[VAL_22]] : i1
// CHECK:                 scf.if %[[VAL_23]] {
// CHECK:                   %[[VAL_24:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK:                   "test.use"(%[[VAL_24]]) : (f64) -> ()
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
//
func.func @foreach_print_slice(%A: tensor<4x4xf64, #CSR_SLICE>) {
  sparse_tensor.foreach in %A : tensor<4x4xf64, #CSR_SLICE> do {
  ^bb0(%1: index, %2: index, %v: f64) :
    "test.use" (%v) : (f64) -> ()
  }
  return
}

#BCOO = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : loose_compressed(nonunique), d2 : singleton)
}>

// CHECK-LABEL:   func.func @foreach_bcoo(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<4x4x4xf64, #{{.*}}>>) {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_5:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<4x4x4xf64, #{{.*}}>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<4x4x4xf64, #{{.*}}>> to memref<?xf64>
// CHECK:           scf.for %[[VAL_7:.*]] = %[[VAL_2]] to %[[VAL_1]] step %[[VAL_3]] {
// CHECK:             %[[VAL_8:.*]] = arith.muli %[[VAL_7]], %[[VAL_4]] : index
// CHECK:             %[[VAL_9:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_8]]] : memref<?xindex>
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_8]], %[[VAL_3]] : index
// CHECK:             %[[VAL_11:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_10]]] : memref<?xindex>
// CHECK:             scf.for %[[VAL_12:.*]] = %[[VAL_9]] to %[[VAL_11]] step %[[VAL_3]] {
// CHECK:               %[[VAL_13:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_12]]] : memref<?xf64>
// CHECK:               "test.use"(%[[VAL_13]]) : (f64) -> ()
// CHECK:             } {"Emitted from" = "sparse_tensor.foreach"}
// CHECK:           } {"Emitted from" = "sparse_tensor.foreach"}
// CHECK:           return
// CHECK:         }
func.func @foreach_bcoo(%A: tensor<4x4x4xf64, #BCOO>) {
  sparse_tensor.foreach in %A : tensor<4x4x4xf64, #BCOO> do {
  ^bb0(%1: index, %2: index, %3: index,  %v: f64) :
    "test.use" (%v) : (f64) -> ()
  }
  return
}
