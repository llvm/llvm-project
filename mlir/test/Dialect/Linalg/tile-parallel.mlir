// RUN: mlir-opt %s -linalg-tile-to-parallel-loops="linalg-tile-sizes=2" | FileCheck %s -check-prefix=TILE-2
// RUN: mlir-opt %s -linalg-tile-to-parallel-loops="linalg-tile-sizes=0,2" | FileCheck %s -check-prefix=TILE-02
// RUN: mlir-opt %s -linalg-tile-to-parallel-loops="linalg-tile-sizes=0,0,2" | FileCheck %s -check-prefix=TILE-002
// RUN: mlir-opt %s -linalg-tile-to-parallel-loops="linalg-tile-sizes=2,3,4" | FileCheck %s -check-prefix=TILE-234

#id_2d = affine_map<(i, j) -> (i, j)>
#pointwise_2d_trait = {
  args_in = 2,
  args_out = 1,
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  iterator_types = ["parallel", "parallel"]
}

func @sum(%lhs: memref<?x?xf32, offset: ?, strides: [?, 1]>,
          %rhs: memref<?x?xf32, offset: ?, strides: [?, 1]>,
          %sum: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.generic #pointwise_2d_trait
     ins(%lhs, %rhs: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                     memref<?x?xf32, offset: ?, strides: [?, 1]>)
    outs(%sum : memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  ^bb0(%lhs_in: f32, %rhs_in: f32, %sum_out: f32):
    %result = addf %lhs_in, %rhs_in : f32
    linalg.yield %result : f32
  }
  return
}
// TILE-2-LABEL: func @sum(
// TILE-2-SAME:    [[LHS:%.*]]: {{.*}}, [[RHS:%.*]]: {{.*}}, [[SUM:%.*]]: {{.*}}) {
// TILE-2-DAG: [[C0:%.*]] = constant 0 : index
// TILE-2-DAG: [[C2:%.*]] = constant 2 : index
// TILE-2: [[LHS_ROWS:%.*]] = memref.dim [[LHS]], %c0
// TILE-2: scf.parallel ([[I:%.*]]) = ([[C0]]) to ([[LHS_ROWS]]) step ([[C2]]) {
// TILE-2-NO: scf.parallel
// TILE-2:   [[LHS_SUBVIEW:%.*]] = memref.subview [[LHS]]
// TILE-2:   [[RHS_SUBVIEW:%.*]] = memref.subview [[RHS]]
// TILE-2:   [[SUM_SUBVIEW:%.*]] = memref.subview [[SUM]]
// TILE-2:   linalg.generic {{.*}} ins([[LHS_SUBVIEW]], [[RHS_SUBVIEW]]{{.*}} outs([[SUM_SUBVIEW]]

// TILE-02-LABEL: func @sum(
// TILE-02-SAME:    [[LHS:%.*]]: {{.*}}, [[RHS:%.*]]: {{.*}}, [[SUM:%.*]]: {{.*}}) {
// TILE-02-DAG: [[C0:%.*]] = constant 0 : index
// TILE-02-DAG: [[C2:%.*]] = constant 2 : index
// TILE-02: [[LHS_COLS:%.*]] = memref.dim [[LHS]], %c1
// TILE-02: scf.parallel ([[I:%.*]]) = ([[C0]]) to ([[LHS_COLS]]) step ([[C2]]) {
// TILE-02-NO: scf.parallel
// TILE-02:   [[LHS_SUBVIEW:%.*]] = memref.subview [[LHS]]
// TILE-02:   [[RHS_SUBVIEW:%.*]] = memref.subview [[RHS]]
// TILE-02:   [[SUM_SUBVIEW:%.*]] = memref.subview [[SUM]]
// TILE-02:    linalg.generic {{.*}} ins([[LHS_SUBVIEW]], [[RHS_SUBVIEW]]{{.*}} outs([[SUM_SUBVIEW]]

// TILE-002-LABEL: func @sum(
// TILE-002-SAME:    [[LHS:%.*]]: {{.*}}, [[RHS:%.*]]: {{.*}}, [[SUM:%.*]]: {{.*}}) {
// TILE-002-NO: scf.parallel
// TILE-002:   linalg.generic {{.*}} ins([[LHS]], [[RHS]]{{.*}} outs([[SUM]]

// TILE-234-LABEL: func @sum(
// TILE-234-SAME:    [[LHS:%.*]]: {{.*}}, [[RHS:%.*]]: {{.*}}, [[SUM:%.*]]: {{.*}}) {
// TILE-234-DAG: [[C0:%.*]] = constant 0 : index
// TILE-234-DAG: [[C2:%.*]] = constant 2 : index
// TILE-234-DAG: [[C3:%.*]] = constant 3 : index
// TILE-234: [[LHS_ROWS:%.*]] = memref.dim [[LHS]], %c0
// TILE-234: [[LHS_COLS:%.*]] = memref.dim [[LHS]], %c1
// TILE-234: scf.parallel ([[I:%.*]], [[J:%.*]]) = ([[C0]], [[C0]]) to ([[LHS_ROWS]], [[LHS_COLS]]) step ([[C2]], [[C3]]) {
// TILE-234-NO: scf.parallel
// TILE-234:   [[LHS_SUBVIEW:%.*]] = memref.subview [[LHS]]
// TILE-234:   [[RHS_SUBVIEW:%.*]] = memref.subview [[RHS]]
// TILE-234:   [[SUM_SUBVIEW:%.*]] = memref.subview [[SUM]]
// TILE-234:   linalg.generic {{.*}} ins([[LHS_SUBVIEW]], [[RHS_SUBVIEW]]{{.*}} outs([[SUM_SUBVIEW]]
