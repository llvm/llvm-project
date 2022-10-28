// RUN: mlir-opt %s  --tensor-copy-insertion --sparse-tensor-rewrite --sparsification --cse | FileCheck %s

#SM = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>

#trait_matmul = {
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d1, d0)>,
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d1, d2)>
  ],
  iterator_types = ["reduction", "parallel", "parallel"]
}

#trait_scale = {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>
  ],
  iterator_types = ["parallel", "parallel"]
}

// CHECK-LABEL: func.func @fold_yield_arg_zero() -> tensor<1024x1024xf64> {
// CHECK:         %[[VAL_0:.*]] = arith.constant dense<0.000000e+00> : tensor<1024x1024xf64>
// CHECK:         %[[VAL_1:.*]] = bufferization.alloc_tensor() copy(%[[VAL_0]]) {bufferization.escape = [false], memory_space = 0 : ui64} : tensor<1024x1024xf64>
// CHECK:         return %[[VAL_1]] : tensor<1024x1024xf64>
// CHECK:       }
func.func @fold_yield_arg_zero() -> tensor<1024x1024xf64> {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = tensor.empty() : tensor<1024x1024xf64>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>,
                                        affine_map<(d0, d1) -> (d0, d1)>],
                                        iterator_types = ["parallel", "parallel"]}
                                        ins(%cst : f64)
                                        outs(%0 : tensor<1024x1024xf64>) {
    ^bb0(%a: f64, %x: f64):
      linalg.yield %a : f64
    } -> tensor<1024x1024xf64>
  return %1 : tensor<1024x1024xf64>
}

// CHECK-LABEL: func.func @fold_yield_direct_zero() -> tensor<32xf64> {
// CHECK:         %[[VAL_0:.*]] = arith.constant dense<0.000000e+00> : tensor<32xf64>
// CHECK:         %[[VAL_1:.*]] = bufferization.alloc_tensor() copy(%[[VAL_0]]) {bufferization.escape = [false], memory_space = 0 : ui64} : tensor<32xf64>
// CHECK:         return %[[VAL_1]] : tensor<32xf64>
// CHECK:       }
func.func @fold_yield_direct_zero() -> tensor<32xf64> {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = tensor.empty() : tensor<32xf64>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>],
                                        iterator_types = ["parallel"]}
                                        outs(%0 : tensor<32xf64>) {
    ^bb0(%x: f64):
      linalg.yield %cst : f64
    } -> tensor<32xf64>
  return %1 : tensor<32xf64>
}

// CHECK-LABEL: func.func @sampled_dd_unfused(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<8x8xf64, #sparse_tensor.encoding<{{.*}}>>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<8x8xf64>,
// CHECK-SAME:    %[[VAL_2:.*]]: tensor<8x8xf64>) -> tensor<8x8xf64> {
// CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[VAL_6:.*]] = arith.constant dense<0.000000e+00> : tensor<8x8xf64>
// CHECK:         %[[VAL_7:.*]] = bufferization.alloc_tensor() copy(%[[VAL_6]]) {bufferization.escape = [false]} : tensor<8x8xf64>
// CHECK:         %[[VAL_8:.*]] = bufferization.alloc_tensor() copy(%[[VAL_6]]) {bufferization.escape = [false], memory_space = 0 : ui64} : tensor<8x8xf64>
// CHECK:         %[[VAL_9:.*]] = bufferization.to_memref %[[VAL_1]] : memref<8x8xf64>
// CHECK:         %[[VAL_10:.*]] = bufferization.to_memref %[[VAL_2]] : memref<8x8xf64>
// CHECK:         %[[VAL_11:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 0 : index} : tensor<8x8xf64, #sparse_tensor.encoding<{{.*}}>> to memref<?xindex>
// CHECK:         %[[VAL_12:.*]] = sparse_tensor.indices %[[VAL_0]] {dimension = 0 : index} : tensor<8x8xf64, #sparse_tensor.encoding<{{.*}}>> to memref<?xindex>
// CHECK:         %[[VAL_13:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 1 : index} : tensor<8x8xf64, #sparse_tensor.encoding<{{.*}}>> to memref<?xindex>
// CHECK:         %[[VAL_14:.*]] = sparse_tensor.indices %[[VAL_0]] {dimension = 1 : index} : tensor<8x8xf64, #sparse_tensor.encoding<{{.*}}>> to memref<?xindex>
// CHECK:         %[[VAL_15:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<8x8xf64, #sparse_tensor.encoding<{{.*}}>> to memref<?xf64>
// CHECK:         %[[VAL_16:.*]] = bufferization.to_memref %[[VAL_8]] : memref<8x8xf64>
// CHECK:         %[[VAL_17:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:         %[[VAL_18:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:         scf.for %[[VAL_19:.*]] = %[[VAL_17]] to %[[VAL_18]] step %[[VAL_5]] {
// CHECK:           %[[VAL_20:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_19]]] : memref<?xindex>
// CHECK:           %[[VAL_21:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_19]]] : memref<?xindex>
// CHECK:           %[[VAL_22:.*]] = arith.addi %[[VAL_19]], %[[VAL_5]] : index
// CHECK:           %[[VAL_23:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_22]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_24:.*]] = %[[VAL_21]] to %[[VAL_23]] step %[[VAL_5]] {
// CHECK:             %[[VAL_25:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_24]]] : memref<?xindex>
// CHECK:             %[[VAL_26:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_20]], %[[VAL_25]]] : memref<8x8xf64>
// CHECK:             %[[VAL_27:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_24]]] : memref<?xf64>
// CHECK:             %[[VAL_28:.*]] = scf.for %[[VAL_29:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] iter_args(%[[VAL_30:.*]] = %[[VAL_26]]) -> (f64) {
// CHECK:               %[[VAL_31:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_20]], %[[VAL_29]]] : memref<8x8xf64>
// CHECK:               %[[VAL_32:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_29]], %[[VAL_25]]] : memref<8x8xf64>
// CHECK:               %[[VAL_33:.*]] = arith.mulf %[[VAL_31]], %[[VAL_32]] : f64
// CHECK:               %[[VAL_34:.*]] = arith.mulf %[[VAL_33]], %[[VAL_27]] : f64
// CHECK:               %[[VAL_35:.*]] = arith.addf %[[VAL_30]], %[[VAL_34]] : f64
// CHECK:               scf.yield %[[VAL_35]] : f64
// CHECK:             }
// CHECK:             memref.store %[[VAL_24:.*]], %[[VAL_16]]{{\[}}%[[VAL_20]], %[[VAL_25]]] : memref<8x8xf64>
// CHECK:           }
// CHECK:         }
// CHECK:         %[[VAL_37:.*]] = bufferization.to_tensor %[[VAL_16]] : memref<8x8xf64>
// CHECK:         return %[[VAL_37]] : tensor<8x8xf64>
// CHECK:       }
func.func @sampled_dd_unfused(%args: tensor<8x8xf64, #SM>,
                              %arga: tensor<8x8xf64>,
                              %argb: tensor<8x8xf64>) -> tensor<8x8xf64> {
  // Perform dense-dense matrix matrix multiplication.
  %1 = arith.constant dense<0.0> : tensor<8x8xf64>
  %2 = linalg.generic #trait_matmul
    ins(%arga, %argb : tensor<8x8xf64>, tensor<8x8xf64>)
    outs(%1 : tensor<8x8xf64>) {
      ^bb0(%a: f64, %b: f64, %x: f64):
        %p = arith.mulf %a, %b : f64
        %q = arith.addf %x, %p : f64
        linalg.yield %q : f64
  } -> tensor<8x8xf64>
  // Sample the result with elements-wise multiplication with sparse matrix.
  %3 = linalg.generic #trait_scale
    ins(%2, %args : tensor<8x8xf64>, tensor<8x8xf64, #SM>)
    outs(%1 : tensor<8x8xf64>) {
      ^bb0(%t: f64, %s: f64, %x: f64):
        %r = arith.mulf %t, %s : f64
        linalg.yield %r : f64
  } -> tensor<8x8xf64>
  return %3 : tensor<8x8xf64>
}

// CHECK-LABEL:   func.func @sparse_sampled_dd_unfused(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<8x8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<8x8xf64>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<8x8xf64>) -> tensor<8x8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant dense<0.000000e+00> : tensor<8x8xf64>
// CHECK-DAG:       %[[VAL_9:.*]] = bufferization.alloc_tensor() copy(%[[VAL_8]]) {bufferization.escape = [false]} : tensor<8x8xf64>
// CHECK-DAG:       %[[VAL_10:.*]] = bufferization.alloc_tensor() {bufferization.escape = [false]} : tensor<8x8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK-DAG:       %[[VAL_11:.*]] = bufferization.to_memref %[[VAL_1]] : memref<8x8xf64>
// CHECK-DAG:       %[[VAL_12:.*]] = bufferization.to_memref %[[VAL_2]] : memref<8x8xf64>
// CHECK-DAG:       %[[VAL_13:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 0 : index} : tensor<8x8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_14:.*]] = sparse_tensor.indices %[[VAL_0]] {dimension = 0 : index} : tensor<8x8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_15:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 1 : index} : tensor<8x8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_16:.*]] = sparse_tensor.indices %[[VAL_0]] {dimension = 1 : index} : tensor<8x8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_17:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<8x8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xf64>
// CHECK:           %[[VAL_18:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:           %[[VAL_19:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:           %[[VAL_20:.*]] = scf.for %[[VAL_21:.*]] = %[[VAL_18]] to %[[VAL_19]] step %[[VAL_5]] iter_args(%[[VAL_22:.*]] = %[[VAL_10]]) -> (tensor<8x8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>) {
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_21]]] : memref<?xindex>
// CHECK:             %[[VAL_24:.*]], %[[VAL_25:.*]], %[[VAL_26:.*]], %[[VAL_27:.*]] = sparse_tensor.expand %[[VAL_10]] : tensor<8x8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xf64>, memref<?xi1>, memref<?xindex>
// CHECK:             %[[VAL_28:.*]] = scf.for %[[VAL_29:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] iter_args(%[[VAL_30:.*]] = %[[VAL_27]]) -> (index) {
// CHECK:               %[[VAL_31:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_23]], %[[VAL_29]]] : memref<8x8xf64>
// CHECK:               %[[VAL_32:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_21]]] : memref<?xindex>
// CHECK:               %[[VAL_33:.*]] = arith.addi %[[VAL_21]], %[[VAL_5]] : index
// CHECK:               %[[VAL_34:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_33]]] : memref<?xindex>
// CHECK:               %[[VAL_35:.*]] = scf.for %[[VAL_36:.*]] = %[[VAL_32]] to %[[VAL_34]] step %[[VAL_5]] iter_args(%[[VAL_37:.*]] = %[[VAL_30]]) -> (index) {
// CHECK:                 %[[VAL_38:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_36]]] : memref<?xindex>
// CHECK:                 %[[VAL_39:.*]] = memref.load %[[VAL_24]]{{\[}}%[[VAL_38]]] : memref<?xf64>
// CHECK:                 %[[VAL_40:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_29]], %[[VAL_38]]] : memref<8x8xf64>
// CHECK:                 %[[VAL_41:.*]] = arith.mulf %[[VAL_31]], %[[VAL_40]] : f64
// CHECK:                 %[[VAL_42:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_36]]] : memref<?xf64>
// CHECK:                 %[[VAL_43:.*]] = arith.mulf %[[VAL_41]], %[[VAL_42]] : f64
// CHECK:                 %[[VAL_44:.*]] = arith.addf %[[VAL_39]], %[[VAL_43]] : f64
// CHECK:                 %[[VAL_45:.*]] = memref.load %[[VAL_25]]{{\[}}%[[VAL_38]]] : memref<?xi1>
// CHECK:                 %[[VAL_46:.*]] = arith.cmpi eq, %[[VAL_45]], %[[VAL_6]] : i1
// CHECK:                 %[[VAL_47:.*]] = scf.if %[[VAL_46]] -> (index) {
// CHECK:                   memref.store %[[VAL_7]], %[[VAL_25]]{{\[}}%[[VAL_38]]] : memref<?xi1>
// CHECK:                   memref.store %[[VAL_38]], %[[VAL_26]]{{\[}}%[[VAL_37]]] : memref<?xindex>
// CHECK:                   %[[VAL_48:.*]] = arith.addi %[[VAL_37]], %[[VAL_5]] : index
// CHECK:                   scf.yield %[[VAL_48]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_37]] : index
// CHECK:                 }
// CHECK:                 memref.store %[[VAL_44]], %[[VAL_24]]{{\[}}%[[VAL_38]]] : memref<?xf64>
// CHECK:                 scf.yield %[[VAL_49:.*]] : index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_50:.*]] : index
// CHECK:             }
// CHECK:             %[[VAL_51:.*]] = sparse_tensor.compress %[[VAL_24]], %[[VAL_25]], %[[VAL_26]], %[[VAL_52:.*]] into %[[VAL_22]]{{\[}}%[[VAL_23]]] : memref<?xf64>, memref<?xi1>, memref<?xindex>, tensor<8x8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:             scf.yield %[[VAL_51]] : tensor<8x8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:           }
// CHECK:           %[[VAL_53:.*]] = sparse_tensor.load %[[VAL_54:.*]] hasInserts : tensor<8x8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:           return %[[VAL_53]] : tensor<8x8xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:         }
func.func @sparse_sampled_dd_unfused(%args: tensor<8x8xf64, #SM>,
                                     %arga: tensor<8x8xf64>,
                                     %argb: tensor<8x8xf64>) -> tensor<8x8xf64, #SM> {
  // Perform dense-dense matrix matrix multiplication.
  %1 = arith.constant dense<0.0> : tensor<8x8xf64>
  %2 = linalg.generic #trait_matmul
    ins(%arga, %argb : tensor<8x8xf64>, tensor<8x8xf64>)
    outs(%1 : tensor<8x8xf64>) {
      ^bb0(%a: f64, %b: f64, %x: f64):
        %p = arith.mulf %a, %b : f64
        %q = arith.addf %x, %p : f64
        linalg.yield %q : f64
  } -> tensor<8x8xf64>
  // Sample the result with elements-wise multiplication with sparse matrix.
  %3 = bufferization.alloc_tensor() : tensor<8x8xf64, #SM>
  %4 = linalg.generic #trait_scale
    ins(%2, %args : tensor<8x8xf64>, tensor<8x8xf64, #SM>)
    outs(%3 : tensor<8x8xf64, #SM>) {
      ^bb0(%t: f64, %s: f64, %x: f64):
        %r = arith.mulf %t, %s : f64
        linalg.yield %r : f64
  } -> tensor<8x8xf64, #SM>
  return %4 : tensor<8x8xf64, #SM>
}
