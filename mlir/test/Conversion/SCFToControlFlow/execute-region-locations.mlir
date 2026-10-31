// RUN: mlir-opt -convert-scf-to-cf -mlir-print-debuginfo -mlir-print-local-scope %s | FileCheck %s

// The branch entering the region takes the location of the
// scf.execute_region, while each branch leaving it keeps the location of the
// scf.yield it replaces.

// CHECK-LABEL: func @execute_region_locations
// CHECK:         cf.br ^{{.*}} loc("op")
// CHECK:         cf.cond_br {{.*}} loc("cond")
// CHECK:         cf.br ^[[CONT:bb[0-9]+]] loc("yield1")
// CHECK:         cf.br ^[[CONT]] loc("yield2")
// CHECK:       ^[[CONT]]:
// CHECK:         return
func.func @execute_region_locations(%cond: i1) {
  scf.execute_region {
    cf.cond_br %cond, ^bb1, ^bb2 loc("cond")
  ^bb1:
    scf.yield loc("yield1")
  ^bb2:
    scf.yield loc("yield2")
  } loc("op")
  return
}
