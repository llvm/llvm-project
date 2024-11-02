// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification | FileCheck %s

#DenseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : dense)
}>

#SparseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

#trait = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) * i * j"
}

// CHECK-LABEL:   func.func @dense_index(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = sparse_tensor.lvl %[[VAL_0]], %[[VAL_1]] : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_4:.*]] = sparse_tensor.lvl %[[VAL_0]], %[[VAL_1]] : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_5:.*]] = tensor.empty(%[[VAL_3]], %[[VAL_4]]) : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.lvl %[[VAL_0]], %[[VAL_1]] : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.lvl %[[VAL_0]], %[[VAL_2]] : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_24:.*]] = sparse_tensor.lvl %[[VAL_5]], %[[VAL_2]] : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.values %[[VAL_5]] : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK:           scf.for %[[VAL_10:.*]] = %[[VAL_1]] to %[[VAL_7]] step %[[VAL_2]] {
// CHECK:             %[[VAL_12:.*]] = arith.muli %[[VAL_10]], %[[VAL_8]] : index
// CHECK:             %[[VAL_14:.*]] = arith.muli %[[VAL_10]], %[[VAL_24]] : index
// CHECK:             scf.for %[[VAL_11:.*]] = %[[VAL_1]] to %[[VAL_8]] step %[[VAL_2]] {
// CHECK:               %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : index
// CHECK:               %[[VAL_15:.*]] = arith.addi %[[VAL_11]], %[[VAL_14]] : index
// CHECK:               %[[VAL_16:.*]] = arith.index_cast %[[VAL_11]] : index to i64
// CHECK:               %[[VAL_17:.*]] = arith.index_cast %[[VAL_10]] : index to i64
// CHECK:               %[[VAL_18:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_13]]] : memref<?xi64>
// CHECK:               %[[VAL_19:.*]] = arith.muli %[[VAL_17]], %[[VAL_18]] : i64
// CHECK:               %[[VAL_20:.*]] = arith.muli %[[VAL_16]], %[[VAL_19]] : i64
// CHECK:               memref.store %[[VAL_20]], %[[VAL_9]]{{\[}}%[[VAL_15]]] : memref<?xi64>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = sparse_tensor.load %[[VAL_5]] : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK:           return %[[VAL_21]] : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK:         }
func.func @dense_index(%arga: tensor<?x?xi64, #DenseMatrix>)
                      -> tensor<?x?xi64, #DenseMatrix> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  %0 = sparse_tensor.lvl %arga, %c0 : tensor<?x?xi64, #DenseMatrix>
  %1 = sparse_tensor.lvl %arga, %c1 : tensor<?x?xi64, #DenseMatrix>
  %init = tensor.empty(%0, %1) : tensor<?x?xi64, #DenseMatrix>
  %r = linalg.generic #trait
      ins(%arga: tensor<?x?xi64, #DenseMatrix>)
     outs(%init: tensor<?x?xi64, #DenseMatrix>) {
      ^bb(%a: i64, %x: i64):
        %i = linalg.index 0 : index
        %j = linalg.index 1 : index
        %ii = arith.index_cast %i : index to i64
        %jj = arith.index_cast %j : index to i64
        %m1 = arith.muli %ii, %a : i64
        %m2 = arith.muli %jj, %m1 : i64
        linalg.yield %m2 : i64
  } -> tensor<?x?xi64, #DenseMatrix>
  return %r : tensor<?x?xi64, #DenseMatrix>
}


// CHECK-LABEL:   func.func @sparse_index(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = sparse_tensor.lvl %[[VAL_0]], %[[VAL_1]] : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_4:.*]] = sparse_tensor.lvl %[[VAL_0]], %[[VAL_1]] : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_5:.*]] = tensor.empty(%[[VAL_3]], %[[VAL_4]]) : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_1]]] : memref<?xindex>
// CHECK:           %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[T:.*]] = scf.for %[[VAL_13:.*]] = %[[VAL_11]] to %[[VAL_12]] step %[[VAL_2]] {{.*}} {
// CHECK:             %[[VAL_14:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_13]]] : memref<?xindex>
// CHECK:             %[[VAL_15:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_13]]] : memref<?xindex>
// CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_13]], %[[VAL_2]] : index
// CHECK:             %[[VAL_17:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK:             %[[L:.*]] = scf.for %[[VAL_18:.*]] = %[[VAL_15]] to %[[VAL_17]] step %[[VAL_2]] {{.*}} {
// CHECK:               %[[VAL_19:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// CHECK:               %[[VAL_20:.*]] = arith.index_cast %[[VAL_19]] : index to i64
// CHECK:               %[[VAL_21:.*]] = arith.index_cast %[[VAL_14]] : index to i64
// CHECK:               %[[VAL_22:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_18]]] : memref<?xi64>
// CHECK:               %[[VAL_23:.*]] = arith.muli %[[VAL_21]], %[[VAL_22]] : i64
// CHECK:               %[[VAL_24:.*]] = arith.muli %[[VAL_20]], %[[VAL_23]] : i64
// CHECK:               %[[Y:.*]] = sparse_tensor.insert %[[VAL_24]] into %{{.*}}[%[[VAL_14]], %[[VAL_19]]] : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK:               scf.yield %[[Y]]
// CHECK:             }
// CHECK:             scf.yield %[[L]]
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = sparse_tensor.load %[[T]] hasInserts : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK:           return %[[VAL_25]] : tensor<?x?xi64, #sparse{{[0-9]*}}>
// CHECK:         }
func.func @sparse_index(%arga: tensor<?x?xi64, #SparseMatrix>)
                       -> tensor<?x?xi64, #SparseMatrix> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  %0 = sparse_tensor.lvl %arga, %c0 : tensor<?x?xi64, #SparseMatrix>
  %1 = sparse_tensor.lvl %arga, %c1 : tensor<?x?xi64, #SparseMatrix>
  %init = tensor.empty(%0, %1) : tensor<?x?xi64, #SparseMatrix>
  %r = linalg.generic #trait
      ins(%arga: tensor<?x?xi64, #SparseMatrix>)
     outs(%init: tensor<?x?xi64, #SparseMatrix>) {
      ^bb(%a: i64, %x: i64):
        %i = linalg.index 0 : index
        %j = linalg.index 1 : index
        %ii = arith.index_cast %i : index to i64
        %jj = arith.index_cast %j : index to i64
        %m1 = arith.muli %ii, %a : i64
        %m2 = arith.muli %jj, %m1 : i64
        linalg.yield %m2 : i64
  } -> tensor<?x?xi64, #SparseMatrix>
  return %r : tensor<?x?xi64, #SparseMatrix>
}
