// RUN: mlir-opt -pass-pipeline='builtin.module(func.func(llvm-infer-alias-scopes-attrs))' %s | FileCheck %s

// CHECK-LABEL:  distinct_objects
func.func @distinct_objects(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %1, %2, %3 = memref.distinct_objects %arg0, %arg1, %arg2 : memref<?xf32>, memref<?xf32>, memref<?xf32>
//       CHECK:    memref.load {{.*}} {alias_scopes = [#[[SCOPE0:.*]]], llvm.noalias = [#[[SCOPE1:.*]], #[[SCOPE2:.*]]]}
//       CHECK:    memref.store {{.*}} {alias_scopes = [#[[SCOPE1]]], llvm.noalias = [#[[SCOPE0]], #[[SCOPE2]]]} : memref<?xf32>
//       CHECK:    memref.store {{.*}} {alias_scopes = [#[[SCOPE2]]], llvm.noalias = [#[[SCOPE0]], #[[SCOPE1]]]} : memref<?xf32>
  %4 = memref.load %1[%c0] : memref<?xf32>
  memref.store %4, %2[%c0] : memref<?xf32>
  memref.store %4, %3[%c0] : memref<?xf32>
  return
}
