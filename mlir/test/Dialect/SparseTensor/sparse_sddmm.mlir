// RUN: mlir-opt %s  --tensor-copy-insertion --sparsification --cse | FileCheck %s

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
  %0 = linalg.init_tensor [1024, 1024] : tensor<1024x1024xf64>
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
  %0 = linalg.init_tensor [32] : tensor<32xf64>
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


// CHECK-LABEL:  func @sparse_sampled_dd_unfused(
// CHECK-SAME:   %[[TMP_arg0:.*]]: tensor<8x8xf64, #sparse_tensor.encoding
// CHECK-SAME:   %[[TMP_arg1:.*]]: tensor<8x8xf64>,
// CHECK-SAME:   %[[TMP_arg2:.*]]: tensor<8x8xf64>)
// CHECK-DAG:    %[[TMP_c8:.*]] = arith.constant 8 : index
// CHECK-DAG:    %[[TMP_c2:.*]] = arith.constant 2 : index
// CHECK-DAG:    %[[TMP_c0:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[TMP_c1:.*]] = arith.constant 1 : index
// CHECK-DAG:    %[[TMP_false:.*]] = arith.constant false
// CHECK-DAG:    %[[TMP_true:.*]] = arith.constant true
// CHECK-DAG:    %[[TMP_cst:.*]] = arith.constant dense<0.000000e+00> : tensor<8x8xf64>
// CHECK:        %[[TMP_0:.*]] = bufferization.alloc_tensor() copy(%[[TMP_cst]]) {bufferization.escape = [false]}
// CHECK:        %[[TMP_1:.*]] = bufferization.alloc_tensor() {bufferization.escape = [false]}
// CHECK:        %[[TMP_2:.*]] = bufferization.to_memref %[[TMP_arg1]] : memref<8x8xf64>
// CHECK:        %[[TMP_3:.*]] = bufferization.to_memref %[[TMP_arg2]] : memref<8x8xf64>
// CHECK:        %[[TMP_4:.*]] = sparse_tensor.pointers %[[TMP_arg0]] {dimension = 0 : index}
// CHECK:        %[[TMP_5:.*]] = sparse_tensor.indices %[[TMP_arg0]] {dimension = 0 : index}
// CHECK:        %[[TMP_6:.*]] = sparse_tensor.pointers %[[TMP_arg0]] {dimension = 1 : index}
// CHECK:        %[[TMP_7:.*]] = sparse_tensor.indices %[[TMP_arg0]] {dimension = 1 : index}
// CHECK:        %[[TMP_8:.*]] = sparse_tensor.values %[[TMP_arg0]]
// CHECK:        %[[TMP_9:.*]] = memref.alloca(%[[TMP_c2]]) : memref<?xindex>
// CHECK:        %[[TMP_10:.*]] = memref.load %[[TMP_4]][%[[TMP_c0]]] : memref<?xindex>
// CHECK:        %[[TMP_11:.*]] = memref.load %[[TMP_4]][%[[TMP_c1]]] : memref<?xindex>
// CHECK:        scf.for %[[TMP_arg3:.*]] = %[[TMP_10]] to %[[TMP_11]] step %[[TMP_c1]] {
// CHECK:          %[[TMP_13:.*]] = memref.load %[[TMP_5]][%[[TMP_arg3]]] : memref<?xindex>
// CHECK:          memref.store %[[TMP_13]], %[[TMP_9]][%[[TMP_c0]]] : memref<?xindex>
// CHECK:          %[[TMP_values:.*]], %[[TMP_filled:.*]], %[[TMP_added:.*]], %[[TMP_count:.*]] = sparse_tensor.expand %[[TMP_1]]
// CHECK:          %[[TMP_14:.*]] = scf.for %[[TMP_arg4:.*]] = %[[TMP_c0]] to %[[TMP_c8]] step %[[TMP_c1]] iter_args(%[[TMP_arg5:.*]] = %[[TMP_count]]) -> (index) {
// CHECK:            %[[TMP_15:.*]] = memref.load %[[TMP_2]][%[[TMP_13]], %[[TMP_arg4]]] : memref<8x8xf64>
// CHECK:            %[[TMP_16:.*]] = memref.load %[[TMP_6]][%[[TMP_arg3]]] : memref<?xindex>
// CHECK:            %[[TMP_17:.*]] = arith.addi %[[TMP_arg3]], %[[TMP_c1]] : index
// CHECK:            %[[TMP_18:.*]] = memref.load %[[TMP_6]][%[[TMP_17]]] : memref<?xindex>
// CHECK:            %[[TMP_19:.*]] = scf.for %[[TMP_arg6:.*]] = %[[TMP_16]] to %[[TMP_18]] step %[[TMP_c1]] iter_args(%[[TMP_arg7:.*]] = %[[TMP_arg5]]) -> (index) {
// CHECK:              %[[TMP_20:.*]] = memref.load %[[TMP_7]][%[[TMP_arg6]]] : memref<?xindex>
// CHECK:              %[[TMP_21:.*]] = memref.load %[[TMP_values]][%[[TMP_20]]] : memref<?xf64>
// CHECK:              %[[TMP_22:.*]] = memref.load %[[TMP_3]][%[[TMP_arg4]], %[[TMP_20]]] : memref<8x8xf64>
// CHECK:              %[[TMP_23:.*]] = arith.mulf %[[TMP_15]], %[[TMP_22]] : f64
// CHECK:              %[[TMP_24:.*]] = memref.load %[[TMP_8]][%[[TMP_arg6]]] : memref<?xf64>
// CHECK:              %[[TMP_25:.*]] = arith.mulf %[[TMP_23]], %[[TMP_24]] : f64
// CHECK:              %[[TMP_26:.*]] = arith.addf %[[TMP_21]], %[[TMP_25]] : f64
// CHECK:              %[[TMP_27:.*]] = memref.load %[[TMP_filled]][%[[TMP_20]]] : memref<?xi1>
// CHECK:              %[[TMP_28:.*]] = arith.cmpi eq, %[[TMP_27]], %[[TMP_false]] : i1
// CHECK:              %[[TMP_29:.*]] = scf.if %[[TMP_28]] -> (index) {
// CHECK:                memref.store %[[TMP_true]], %[[TMP_filled]][%[[TMP_20]]] : memref<?xi1>
// CHECK:                memref.store %[[TMP_20]], %[[TMP_added]][%[[TMP_arg7]]] : memref<?xindex>
// CHECK:                %[[TMP_30:.*]] = arith.addi %[[TMP_arg7]], %[[TMP_c1]] : index
// CHECK:                scf.yield %[[TMP_30]] : index
// CHECK:              } else {
// CHECK:                scf.yield %[[TMP_arg7]] : index
// CHECK:              }
// CHECK:              memref.store %[[TMP_26]], %[[TMP_values]][%[[TMP_20]]] : memref<?xf64>
// CHECK:              scf.yield %[[TMP_29]] : index
// CHECK:            }
// CHECK:            scf.yield %[[TMP_19]] : index
// CHECK:          }
// CHECK:          sparse_tensor.compress %[[TMP_1]], %[[TMP_9]], %[[TMP_values]], %[[TMP_filled]], %[[TMP_added]], %[[TMP_14]]
// CHECK:        }
// CHECK:        %[[TMP_12:.*]] = sparse_tensor.load %[[TMP_1]] hasInserts 
// CHECK:        return %[[TMP_12]] : tensor<8x8xf64, #sparse_tensor.encoding
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
