// RUN: mlir-opt %s --lower-sparse-foreach-to-scf --canonicalize | FileCheck %s

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
  map = (d0 : #sparse_tensor<slice(0, 4, 1)>, d1 : #sparse_tensor<slice(2, 4, 1)>) -> (d0 : compressed, d1 : compressed)
}>

#CSR_SLICE_DYN = #sparse_tensor.encoding<{
  map = (d0 : #sparse_tensor<slice(?, ?, ?)>, d1 : #sparse_tensor<slice(?, ?, ?)>) -> (d0 : compressed, d1 : compressed)
}>

// TODO: re-enable after lowering coo.next to function call (such that loop structure is more clear).

// C_HECK-LABEL:   func.func @foreach_print_slice_dyn(
// C_HECK-SAME:                                       %[[VAL_0:.*]]: tensor<?x?xf64,
// C_HECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// C_HECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// C_HECK-DAG:       %[[VAL_3:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<?x?xf64,
// C_HECK-DAG:       %[[VAL_4:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<?x?xf64,
// C_HECK-DAG:       %[[VAL_5:.*]] = sparse_tensor.lvl %[[VAL_0]], %[[VAL_1]] : tensor<?x?xf64,
// C_HECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.slice.offset %[[VAL_0]] at 0 : tensor<?x?xf64,
// C_HECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.slice.stride %[[VAL_0]] at 0 : tensor<?x?xf64,
// C_HECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<?x?xf64,
// C_HECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<?x?xf64,
// C_HECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.lvl %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf64,
// C_HECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.slice.offset %[[VAL_0]] at 1 : tensor<?x?xf64,
// C_HECK-DAG:       %[[VAL_12:.*]] = sparse_tensor.slice.stride %[[VAL_0]] at 1 : tensor<?x?xf64,
// C_HECK-DAG:       %[[VAL_13:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64,
// C_HECK:           %[[VAL_14:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_1]]] : memref<?xindex>
// C_HECK:           %[[VAL_15:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// C_HECK:           scf.for %[[VAL_16:.*]] = %[[VAL_14]] to %[[VAL_15]] step %[[VAL_2]] {
// C_HECK:             %[[VAL_17:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// C_HECK:             %[[VAL_18:.*]] = arith.subi %[[VAL_17]], %[[VAL_6]] : index
// C_HECK:             %[[VAL_19:.*]] = arith.remui %[[VAL_18]], %[[VAL_7]] : index
// C_HECK:             %[[VAL_20:.*]] = arith.divui %[[VAL_18]], %[[VAL_7]] : index
// C_HECK:             %[[VAL_21:.*]] = arith.cmpi uge, %[[VAL_17]], %[[VAL_6]] : index
// C_HECK:             %[[VAL_22:.*]] = arith.cmpi ult, %[[VAL_20]], %[[VAL_5]] : index
// C_HECK:             %[[VAL_23:.*]] = arith.cmpi eq, %[[VAL_19]], %[[VAL_1]] : index
// C_HECK:             %[[VAL_24:.*]] = arith.andi %[[VAL_21]], %[[VAL_22]] : i1
// C_HECK:             %[[VAL_25:.*]] = arith.andi %[[VAL_24]], %[[VAL_23]] : i1
// C_HECK:             scf.if %[[VAL_25]] {
// C_HECK:               %[[VAL_26:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// C_HECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_16]], %[[VAL_2]] : index
// C_HECK:               %[[VAL_28:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_27]]] : memref<?xindex>
// C_HECK:               scf.for %[[VAL_29:.*]] = %[[VAL_26]] to %[[VAL_28]] step %[[VAL_2]] {
// C_HECK:                 %[[VAL_30:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_29]]] : memref<?xindex>
// C_HECK:                 %[[VAL_31:.*]] = arith.subi %[[VAL_30]], %[[VAL_11]] : index
// C_HECK:                 %[[VAL_32:.*]] = arith.remui %[[VAL_31]], %[[VAL_12]] : index
// C_HECK:                 %[[VAL_33:.*]] = arith.divui %[[VAL_31]], %[[VAL_12]] : index
// C_HECK:                 %[[VAL_34:.*]] = arith.cmpi uge, %[[VAL_30]], %[[VAL_11]] : index
// C_HECK:                 %[[VAL_35:.*]] = arith.cmpi ult, %[[VAL_33]], %[[VAL_10]] : index
// C_HECK:                 %[[VAL_36:.*]] = arith.cmpi eq, %[[VAL_32]], %[[VAL_1]] : index
// C_HECK:                 %[[VAL_37:.*]] = arith.andi %[[VAL_34]], %[[VAL_35]] : i1
// C_HECK:                 %[[VAL_38:.*]] = arith.andi %[[VAL_37]], %[[VAL_36]] : i1
// C_HECK:                 scf.if %[[VAL_38]] {
// C_HECK:                   %[[VAL_39:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_29]]] : memref<?xf64>
// C_HECK:                   "test.use"(%[[VAL_39]]) : (f64) -> ()
// C_HECK:                 }
// C_HECK:               }
// C_HECK:             }
// C_HECK:           }
// C_HECK:           return
//
func.func @foreach_print_slice_dyn(%A: tensor<?x?xf64, #CSR_SLICE_DYN>) {
  sparse_tensor.foreach in %A : tensor<?x?xf64, #CSR_SLICE_DYN> do {
  ^bb0(%1: index, %2: index, %v: f64) :
    "test.use" (%v) : (f64) -> ()
  }
  return
}

// C_HECK-LABEL:   func.func @foreach_print_slice(
// C_HECK-SAME:                                   %[[VAL_0:.*]]: tensor<4x4xf64,
// C_HECK-DAG:       %[[VAL_1:.*]] = arith.constant 4 : index
// C_HECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : index
// C_HECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// C_HECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : index
// C_HECK-DAG:       %[[VAL_5:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<4x4xf64,
// C_HECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<4x4xf64,
// C_HECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<4x4xf64,
// C_HECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<4x4xf64,
// C_HECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<4x4xf64,
// C_HECK-DAG:       %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// C_HECK:           %[[VAL_11:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// C_HECK:           scf.for %[[VAL_12:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_4]] {
// C_HECK:             %[[VAL_13:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_12]]] : memref<?xindex>
// C_HECK:             %[[VAL_14:.*]] = arith.cmpi ult, %[[VAL_13]], %[[VAL_1]] : index
// C_HECK:             scf.if %[[VAL_14]] {
// C_HECK:               %[[VAL_15:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_12]]] : memref<?xindex>
// C_HECK:               %[[VAL_16:.*]] = arith.addi %[[VAL_12]], %[[VAL_4]] : index
// C_HECK:               %[[VAL_17:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// C_HECK:               scf.for %[[VAL_18:.*]] = %[[VAL_15]] to %[[VAL_17]] step %[[VAL_4]] {
// C_HECK:                 %[[VAL_19:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// C_HECK:                 %[[VAL_20:.*]] = arith.subi %[[VAL_19]], %[[VAL_2]] : index
// C_HECK:                 %[[VAL_21:.*]] = arith.cmpi uge, %[[VAL_19]], %[[VAL_2]] : index
// C_HECK:                 %[[VAL_22:.*]] = arith.cmpi ult, %[[VAL_20]], %[[VAL_1]] : index
// C_HECK:                 %[[VAL_23:.*]] = arith.andi %[[VAL_21]], %[[VAL_22]] : i1
// C_HECK:                 scf.if %[[VAL_23]] {
// C_HECK:                   %[[VAL_24:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// C_HECK:                   "test.use"(%[[VAL_24]]) : (f64) -> ()
// C_HECK:                 }
// C_HECK:               }
// C_HECK:             }
// C_HECK:           }
// C_HECK:           return
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

// C_HECK-LABEL:   func.func @foreach_bcoo(
// C_HECK-SAME:      %[[VAL_0:.*]]: tensor<4x4x4xf64, #sparse{{[0-9]*}}>) {
// C_HECK-DAG:       %[[VAL_1:.*]] = arith.constant 4 : index
// C_HECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// C_HECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// C_HECK-DAG:       %[[VAL_4:.*]] = arith.constant 2 : index
// C_HECK-DAG:       %[[VAL_5:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<4x4x4xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// C_HECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<4x4x4xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// C_HECK:           scf.for %[[VAL_7:.*]] = %[[VAL_2]] to %[[VAL_1]] step %[[VAL_3]] {
// C_HECK:             %[[VAL_8:.*]] = arith.muli %[[VAL_7]], %[[VAL_4]] : index
// C_HECK:             %[[VAL_9:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_8]]] : memref<?xindex>
// C_HECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_8]], %[[VAL_3]] : index
// C_HECK:             %[[VAL_11:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_10]]] : memref<?xindex>
// C_HECK:             scf.for %[[VAL_12:.*]] = %[[VAL_9]] to %[[VAL_11]] step %[[VAL_3]] {
// C_HECK:               %[[VAL_13:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_12]]] : memref<?xf64>
// C_HECK:               "test.use"(%[[VAL_13]]) : (f64) -> ()
// C_HECK:             } {"Emitted from" = "sparse_tensor.foreach"}
// C_HECK:           } {"Emitted from" = "sparse_tensor.foreach"}
// C_HECK:           return
// C_HECK:         }
func.func @foreach_bcoo(%A: tensor<4x4x4xf64, #BCOO>) {
  sparse_tensor.foreach in %A : tensor<4x4x4xf64, #BCOO> do {
  ^bb0(%1: index, %2: index, %3: index,  %v: f64) :
    "test.use" (%v) : (f64) -> ()
  }
  return
}
