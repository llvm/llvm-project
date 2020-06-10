// RUN: mlir-opt %s -pass-pipeline='func(parallel-loop-tiling{parallel-loop-tile-sizes=1,4})' -split-input-file | FileCheck %s

func @parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index,
                    %arg3 : index, %arg4 : index, %arg5 : index,
		    %A: memref<?x?xf32>, %B: memref<?x?xf32>,
                    %C: memref<?x?xf32>, %result: memref<?x?xf32>) {
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3) step (%arg4, %arg5) {
    %B_elem = load %B[%i0, %i1] : memref<?x?xf32>
    %C_elem = load %C[%i0, %i1] : memref<?x?xf32>
    %sum_elem = addf %B_elem, %C_elem : f32
    store %sum_elem, %result[%i0, %i1] : memref<?x?xf32>
  }
  return
}

// CHECK:       #map0 = affine_map<(d0, d1, d2) -> (d0, d1 - d2)>
// CHECK-LABEL:   func @parallel_loop(
// CHECK-SAME:                        [[VAL_0:%.*]]: index, [[VAL_1:%.*]]: index, [[VAL_2:%.*]]: index, [[VAL_3:%.*]]: index, [[VAL_4:%.*]]: index, [[VAL_5:%.*]]: index, [[VAL_6:%.*]]: memref<?x?xf32>, [[VAL_7:%.*]]: memref<?x?xf32>, [[VAL_8:%.*]]: memref<?x?xf32>, [[VAL_9:%.*]]: memref<?x?xf32>) {
// CHECK:           [[VAL_10:%.*]] = constant 0 : index
// CHECK:           [[VAL_11:%.*]] = constant 1 : index
// CHECK:           [[VAL_12:%.*]] = constant 4 : index
// CHECK:           [[VAL_13:%.*]] = muli [[VAL_4]], [[VAL_11]] : index
// CHECK:           [[VAL_14:%.*]] = muli [[VAL_5]], [[VAL_12]] : index
// CHECK:           scf.parallel ([[VAL_15:%.*]], [[VAL_16:%.*]]) = ([[VAL_0]], [[VAL_1]]) to ([[VAL_2]], [[VAL_3]]) step ([[VAL_13]], [[VAL_14]]) {
// CHECK:             [[VAL_17:%.*]] = affine.min #map0([[VAL_11]], [[VAL_2]], [[VAL_15]])
// CHECK:             [[VAL_18:%.*]] = affine.min #map0([[VAL_12]], [[VAL_3]], [[VAL_16]])
// CHECK:             scf.parallel ([[VAL_19:%.*]], [[VAL_20:%.*]]) = ([[VAL_10]], [[VAL_10]]) to ([[VAL_17]], [[VAL_18]]) step ([[VAL_4]], [[VAL_5]]) {
// CHECK:               [[VAL_21:%.*]] = load [[VAL_7]]{{\[}}[[VAL_19]], [[VAL_20]]] : memref<?x?xf32>
// CHECK:               [[VAL_22:%.*]] = load [[VAL_8]]{{\[}}[[VAL_19]], [[VAL_20]]] : memref<?x?xf32>
// CHECK:               [[VAL_23:%.*]] = addf [[VAL_21]], [[VAL_22]] : f32
// CHECK:               store [[VAL_23]], [[VAL_9]]{{\[}}[[VAL_19]], [[VAL_20]]] : memref<?x?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return

// -----

func @tile_nested_innermost() {
  %c2 = constant 2 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    scf.parallel (%k, %l) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    }
  }
  scf.parallel (%i, %j) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
  }
  return
}

// CHECK-LABEL:   func @tile_nested_innermost() {
// CHECK:           [[VAL_24:%.*]] = constant 2 : index
// CHECK:           [[VAL_25:%.*]] = constant 0 : index
// CHECK:           [[VAL_26:%.*]] = constant 1 : index
// CHECK:           scf.parallel ([[VAL_27:%.*]], [[VAL_28:%.*]]) = ([[VAL_25]], [[VAL_25]]) to ([[VAL_24]], [[VAL_24]]) step ([[VAL_26]], [[VAL_26]]) {
// CHECK:             [[VAL_29:%.*]] = constant 0 : index
// CHECK:             [[VAL_30:%.*]] = constant 1 : index
// CHECK:             [[VAL_31:%.*]] = constant 4 : index
// CHECK:             [[VAL_32:%.*]] = muli [[VAL_26]], [[VAL_30]] : index
// CHECK:             [[VAL_33:%.*]] = muli [[VAL_26]], [[VAL_31]] : index
// CHECK:             scf.parallel ([[VAL_34:%.*]], [[VAL_35:%.*]]) = ([[VAL_25]], [[VAL_25]]) to ([[VAL_24]], [[VAL_24]]) step ([[VAL_32]], [[VAL_33]]) {
// CHECK:               [[VAL_36:%.*]] = affine.min #map0([[VAL_30]], [[VAL_24]], [[VAL_34]])
// CHECK:               [[VAL_37:%.*]] = affine.min #map0([[VAL_31]], [[VAL_24]], [[VAL_35]])
// CHECK:               scf.parallel ([[VAL_38:%.*]], [[VAL_39:%.*]]) = ([[VAL_29]], [[VAL_29]]) to ([[VAL_36]], [[VAL_37]]) step ([[VAL_26]], [[VAL_26]]) {
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[VAL_40:%.*]] = constant 0 : index
// CHECK:           [[VAL_41:%.*]] = constant 1 : index
// CHECK:           [[VAL_42:%.*]] = constant 4 : index
// CHECK:           [[VAL_43:%.*]] = muli [[VAL_26]], [[VAL_41]] : index
// CHECK:           [[VAL_44:%.*]] = muli [[VAL_26]], [[VAL_42]] : index
// CHECK:           scf.parallel ([[VAL_45:%.*]], [[VAL_46:%.*]]) = ([[VAL_25]], [[VAL_25]]) to ([[VAL_24]], [[VAL_24]]) step ([[VAL_43]], [[VAL_44]]) {
// CHECK:             [[VAL_47:%.*]] = affine.min #map0([[VAL_41]], [[VAL_24]], [[VAL_45]])
// CHECK:             [[VAL_48:%.*]] = affine.min #map0([[VAL_42]], [[VAL_24]], [[VAL_46]])
// CHECK:             scf.parallel ([[VAL_49:%.*]], [[VAL_50:%.*]]) = ([[VAL_40]], [[VAL_40]]) to ([[VAL_47]], [[VAL_48]]) step ([[VAL_26]], [[VAL_26]]) {
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
