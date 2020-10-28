// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=10,25" | FileCheck %s -check-prefix=TILE-10n25
// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=25,0" | FileCheck %s -check-prefix=TILE-25n0
// RUN: mlir-opt %s -linalg-tile="linalg-tile-sizes=0,25" | FileCheck %s -check-prefix=TILE-0n25

#id_1d = affine_map<(i) -> (i)>
#pointwise_1d_trait = {
  args_in = 1,
  args_out = 1,
  indexing_maps = [#id_1d, #id_1d],
  iterator_types = ["parallel"]
}
func @indexed_generic_vector(%operand: memref<50xf32>, %result: memref<50xf32>) {
  linalg.indexed_generic #pointwise_1d_trait
      ins(%operand :memref<50xf32>)
     outs(%result : memref<50xf32>) {
    ^bb0(%i: index, %operand_in: f32, %result_in: f32):
      %i_int = index_cast %i: index to i32
      %i_float = sitofp %i_int : i32 to f32
      %out = addf %operand_in, %i_float : f32
      linalg.yield %out : f32
  }
  return
}
// TILE-10n25-LABEL: func @indexed_generic_vector
// TILE-10n25: %[[C10:.*]] = constant 10 : index
// TILE-10n25: scf.for %[[J:.*]] = {{.*}} step %[[C10]]
// TILE-10n25:   linalg.indexed_generic
// TILE-10n25:   ^bb0(%[[I:.*]]: index, %[[IN:.*]]: f32, %[[OUT:.*]]: f32)
// TILE-10n25:     %[[NEW_I:.*]] = addi %[[I]], %[[J]] : index
// TILE-10n25:     %[[NEW_I_INT:.*]] = index_cast %[[NEW_I]] : index to i32
// TILE-10n25:     %[[NEW_I_FLOAT:.*]] = sitofp %[[NEW_I_INT]] : i32 to f32
// TILE-10n25:     %[[OUT:.*]] = addf %[[IN]], %[[NEW_I_FLOAT]] : f32

// TILE-25n0-LABEL: func @indexed_generic_vector
// TILE-25n0: %[[C25:.*]] = constant 25 : index
// TILE-25n0: scf.for %[[J:.*]] = {{.*}} step %[[C25]]
// TILE-25n0:   linalg.indexed_generic
// TILE-25n0:   ^bb0(%[[I:.*]]: index, %[[IN:.*]]: f32, %[[OUT:.*]]: f32)
// TILE-25n0:     %[[NEW_I:.*]] = addi %[[I]], %[[J]] : index
// TILE-25n0:     %[[NEW_I_INT:.*]] = index_cast %[[NEW_I]] : index to i32
// TILE-25n0:     %[[NEW_I_FLOAT:.*]] = sitofp %[[NEW_I_INT]] : i32 to f32
// TILE-25n0:     %[[OUT:.*]] = addf %[[IN]], %[[NEW_I_FLOAT]] : f32

// TILE-0n25-LABEL: func @indexed_generic_vector
// TILE-0n25-NOT: scf.for %[[J:.*]] = {{.*}} step %[[C25]]
// TILE-0n25: linalg.indexed_generic

#combined_indices_trait = {
  args_in = 1,
  args_out = 1,
  indexing_maps = [
    affine_map<(i, j) -> (j, i + j)>,
    affine_map<(i, j) -> (i, j)>
  ],
  iterator_types = ["parallel", "parallel"]
}
func @indexed_generic_matrix(%operand: memref<50x100xf32>, %result: memref<50x100xf32>) {
  linalg.indexed_generic #combined_indices_trait
     ins(%operand : memref<50x100xf32>)
    outs(%result : memref<50x100xf32>) {
    ^bb0(%i: index, %j: index, %operand_in: f32, %result_in: f32):
      %i_int = index_cast %i: index to i32
      %i_float = sitofp %i_int : i32 to f32
      %j_int = index_cast %j: index to i32
      %j_float = sitofp %j_int : i32 to f32
      %out = addf %i_float, %j_float : f32
      linalg.yield %out : f32
  }
  return
}
// TILE-10n25-LABEL: func @indexed_generic_matrix
// TILE-10n25-DAG: %[[C25:.*]] = constant 25 : index
// TILE-10n25-DAG: %[[C10:.*]] = constant 10 : index
// TILE-10n25: scf.for %[[K:.*]] = {{.*}} step %[[C10]]
// TILE-10n25:   scf.for %[[L:.*]] = {{.*}} step %[[C25]]
// TILE-10n25:     linalg.indexed_generic
// TILE-10n25:     ^bb0(%[[I:.*]]: index, %[[J:.*]]: index, %[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// TILE-10n25:       %[[NEW_I:.*]] = addi %[[I]], %[[K]] : index
// TILE-10n25:       %[[NEW_J:.*]] = addi %[[J]], %[[L]] : index
// TILE-10n25:       %[[NEW_INT_I:.*]] = index_cast %[[NEW_I]] : index to i32
// TILE-10n25:       %[[NEW_FLOAT_I:.*]] = sitofp %[[NEW_INT_I]] : i32 to f32
// TILE-10n25:       %[[NEW_INT_J:.*]] = index_cast %[[NEW_J]] : index to i32
// TILE-10n25:       %[[NEW_FLOAT_J:.*]] = sitofp %[[NEW_INT_J]] : i32 to f32
// TILE-10n25:       %[[OUT:.*]] = addf %[[NEW_FLOAT_I]], %[[NEW_FLOAT_J]] : f32

// TILE-25n0-LABEL: func @indexed_generic_matrix
// TILE-25n0: %[[C25:.*]] = constant 25 : index
// TILE-25n0: scf.for %[[L:.*]] = {{.*}} step %[[C25]]
// TILE-25n0:   linalg.indexed_generic
// TILE-25n0:   ^bb0(%[[I:.*]]: index, %[[J:.*]]: index, %[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// TILE-25n0:     %[[NEW_I:.*]] = addi %[[I]], %[[L]] : index
// TILE-25n0:     %[[NEW_INT_I:.*]] = index_cast %[[NEW_I]] : index to i32
// TILE-25n0:     %[[NEW_FLOAT_I:.*]] = sitofp %[[NEW_INT_I]] : i32 to f32
// TILE-25n0:     %[[INT_J:.*]] = index_cast %[[J]] : index to i32
// TILE-25n0:     %[[FLOAT_J:.*]] = sitofp %[[INT_J]] : i32 to f32
// TILE-25n0:     %[[OUT:.*]] = addf %[[NEW_FLOAT_I]], %[[FLOAT_J]] : f32

// TILE-0n25-LABEL: func @indexed_generic_matrix
// TILE-0n25: %[[C25:.*]] = constant 25 : index
// TILE-0n25: scf.for %[[L:.*]] = {{.*}} step %[[C25]]
// TILE-0n25:   linalg.indexed_generic
// TILE-0n25:   ^bb0(%[[I:.*]]: index, %[[J:.*]]: index, %[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// TILE-0n25:     %[[NEW_J:.*]] = addi %[[J]], %[[L]] : index
// TILE-0n25:     %[[INT_I:.*]] = index_cast %[[I]] : index to i32
// TILE-0n25:     %[[FLOAT_I:.*]] = sitofp %[[INT_I]] : i32 to f32
// TILE-0n25:     %[[NEW_INT_J:.*]] = index_cast %[[NEW_J]] : index to i32
// TILE-0n25:     %[[NEW_FLOAT_J:.*]] = sitofp %[[NEW_INT_J]] : i32 to f32
// TILE-0n25:     %[[OUT:.*]] = addf %[[FLOAT_I]], %[[NEW_FLOAT_J]] : f32
