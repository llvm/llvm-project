// RUN: mlir-opt %s --pass-pipeline='builtin.module(any(mem2reg))' | FileCheck %s --check-prefix=MEM2REG
// RUN: mlir-opt %s --pass-pipeline='builtin.module(any(mem2reg,canonicalize))' | FileCheck %s --check-prefix=CANON

// Two loop nests share the same alloca, causing the first loop's result
// to chain into the second loop's init -- demonstrating the cross-loop
// use chain that the generic canonicalization patterns cannot handle.

// MEM2REG-LABEL: func.func @redundant_iter_args
//       MEM2REG:   %[[POISON:.*]] = ub.poison : f32
//       MEM2REG:   %[[CST:.*]] = arith.constant 1.000000e+00 : f32
//       MEM2REG:   %[[R1:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[POISON]]) -> (f32) {
//       MEM2REG:     %[[R1I:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (f32) {
//       MEM2REG:       memref.store %[[CST]],
//       MEM2REG:       scf.yield %[[CST]] : f32
//       MEM2REG:     }
//       MEM2REG:     scf.yield %[[R1I]] : f32
//       MEM2REG:   }
//       MEM2REG:   scf.for {{.*}} iter_args(%{{.*}} = %[[R1]]) -> (f32) {
//       MEM2REG:     scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (f32) {
//       MEM2REG:       memref.store %[[CST]],
//       MEM2REG:       scf.yield %[[CST]] : f32
//       MEM2REG:     }
//       MEM2REG:   }

// CANON-LABEL: func.func @redundant_iter_args
//  CANON-SAME:   (%[[N:.*]]: index, %[[MEM:.*]]: memref<f32>)
//       CANON:   %[[CST:.*]] = arith.constant 1.000000e+00 : f32
//       CANON:   scf.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} {
//       CANON:     scf.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} {
//   CANON-NOT:       iter_args
//       CANON:       memref.store %[[CST]], %[[MEM]][] : memref<f32>
//       CANON:     }
//       CANON:   }
//       CANON:   scf.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} {
//       CANON:     scf.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} {
//   CANON-NOT:       iter_args
//       CANON:       memref.store %[[CST]], %[[MEM]][] : memref<f32>
//       CANON:     }
//       CANON:   }
//       CANON:   return

func.func @redundant_iter_args(%n: index, %mem: memref<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.0 : f32
  %tmp = memref.alloca() : memref<f32>
  scf.for %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      memref.store %cst, %tmp[] : memref<f32>
      %v = memref.load %tmp[] : memref<f32>
      memref.store %v, %mem[] : memref<f32>
    }
  }
  scf.for %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      memref.store %cst, %tmp[] : memref<f32>
      %v = memref.load %tmp[] : memref<f32>
      memref.store %v, %mem[] : memref<f32>
    }
  }
  return
}
