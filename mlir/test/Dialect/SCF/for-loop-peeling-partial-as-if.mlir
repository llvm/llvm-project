// RUN: mlir-opt %s -scf-for-loop-peeling=partial-as-if=true -canonicalize -split-input-file | FileCheck %s
// RUN: mlir-opt %s -scf-for-loop-peeling="partial-as-if=true peel-front=true" -canonicalize -split-input-file | FileCheck %s --check-prefix=FRONT
// RUN: mlir-opt %s -scf-for-loop-peeling="partial-as-if=true skip-partial=false" -canonicalize -split-input-file | FileCheck %s --check-prefix=NO-SKIP

// The partial iteration becomes an scf.if. Its body uses the lower bound as
// the induction variable, and the else branch forwards the init args. The
// affine.min in the partial iteration is still simplified.

//  CHECK-DAG: #[[$MAP:.*]] = affine_map<()[s0, s1] -> (s0 - s1)>
//      CHECK: func @iter_args(
// CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[INIT:.*]]: f32
//  CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//      CHECK:   %[[MAIN:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[SPLIT:.*]] step %[[C4]] iter_args(%[[ACC:.*]] = %[[INIT]]) -> (f32) {
//      CHECK:     %[[V:.*]] = func.call @use(%[[IV]], %[[C4]])
//      CHECK:     %[[SUM:.*]] = arith.addf %[[ACC]], %[[V]]
//      CHECK:     scf.yield %[[SUM]]
//      CHECK:   }
//      CHECK:   %[[COND:.*]] = arith.cmpi slt, %[[SPLIT]], %[[UB]] : index
//      CHECK:   %[[RES:.*]] = scf.if %[[COND]] -> (f32) {
//      CHECK:     %[[SIZE:.*]] = affine.apply #[[$MAP]]()[%[[UB]], %[[SPLIT]]]
//      CHECK:     %[[V2:.*]] = func.call @use(%[[SPLIT]], %[[SIZE]])
//      CHECK:     %[[SUM2:.*]] = arith.addf %[[MAIN]], %[[V2]]
//      CHECK:     scf.yield %[[SUM2]]
//      CHECK:   } else {
//      CHECK:     scf.yield %[[MAIN]]
//      CHECK:   }
//      CHECK:   return %[[RES]]

// With peel-front, the first iteration becomes the scf.if.

//      FRONT: func @iter_args(
// FRONT-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[INIT:.*]]: f32
//      FRONT:   %[[SPLIT:.*]] = arith.maxsi %[[LB]], %{{.*}} : index
//      FRONT:   %[[COND:.*]] = arith.cmpi slt, %[[LB]], %[[SPLIT]] : index
//      FRONT:   %[[FIRST:.*]] = scf.if %[[COND]] -> (f32) {
//      FRONT:     %[[V:.*]] = func.call @use(%[[LB]], %{{.*}})
//      FRONT:     %[[SUM:.*]] = arith.addf %[[INIT]], %[[V]]
//      FRONT:     scf.yield %[[SUM]]
//      FRONT:   } else {
//      FRONT:     scf.yield %[[INIT]]
//      FRONT:   }
//      FRONT:   %[[RES:.*]] = scf.for %{{.*}} = %[[SPLIT]] to %[[UB]] step %{{.*}} iter_args(%{{.*}} = %[[FIRST]]) -> (f32) {
//      FRONT:   return %[[RES]]

#map = affine_map<(d0)[s0, s1] -> (s0, s1 - d0)>
func.func private @use(index, index) -> f32

func.func @iter_args(%lb: index, %ub: index, %init: f32) -> f32 {
  %c4 = arith.constant 4 : index
  %r = scf.for %iv = %lb to %ub step %c4 iter_args(%acc = %init) -> f32 {
    %size = affine.min #map(%iv)[%c4, %ub]
    %v = func.call @use(%iv, %size) : (index, index) -> f32
    %sum = arith.addf %acc, %v : f32
    scf.yield %sum : f32
  }
  return %r : f32
}

// -----

// An unsigned loop gets an unsigned compare. Without results, there is no
// else branch.

//      CHECK: func @unsigned(
// CHECK-SAME:     %[[UB:.*]]: index
//      CHECK:   scf.for unsigned %{{.*}} = %{{.*}} to %[[SPLIT:.*]] step
//      CHECK:   %[[COND:.*]] = arith.cmpi ult, %[[SPLIT]], %[[UB]] : index
//      CHECK:   scf.if %[[COND]] {
//      CHECK:     func.call @use(%[[SPLIT]])
//  CHECK-NOT:   else
//      CHECK:   return

//      FRONT: func @unsigned(
// FRONT-SAME:     %[[UB:.*]]: index
//  FRONT-DAG:   %[[C0:.*]] = arith.constant 0 : index
//      FRONT:   %[[SPLIT:.*]] = arith.minui %[[UB]], %{{.*}} : index
//      FRONT:   %[[COND:.*]] = arith.cmpi ugt, %[[SPLIT]], %[[C0]] : index
//      FRONT:   scf.if %[[COND]] {
//      FRONT:     func.call @use(%[[C0]])
//      FRONT:   scf.for unsigned %{{.*}} = %[[SPLIT]] to %[[UB]] step

func.func private @use(index)

func.func @unsigned(%ub: index) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  scf.for unsigned %iv = %c0 to %ub step %c4 {
    func.call @use(%iv) : (index) -> ()
  }
  return
}

// -----

// With skip-partial (the default), the inner loop is not peeled inside the
// scf.if of the outer partial iteration.

//      CHECK: func @nested(
//      CHECK:   scf.for
//      CHECK:     scf.for
//      CHECK:     scf.if
//      CHECK:   }
//      CHECK:   scf.if
//      CHECK:     scf.for
//  CHECK-NOT:     scf.if
//      CHECK:   return

//      NO-SKIP: func @nested(
//      NO-SKIP:   scf.for
//      NO-SKIP:     scf.for
//      NO-SKIP:     scf.if
//      NO-SKIP:   }
//      NO-SKIP:   scf.if
//      NO-SKIP:     scf.for
//      NO-SKIP:     scf.if
//      NO-SKIP:   }
//      NO-SKIP:   return

func.func private @use(index, index)

func.func @nested(%ub0: index, %ub1: index) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %ub0 step %c4 {
    scf.for %j = %c0 to %ub1 step %c8 {
      func.call @use(%i, %j) : (index, index) -> ()
    }
  }
  return
}
