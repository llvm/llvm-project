// RUN: mlir-opt -one-shot-bufferize="bufferize-function-boundaries" -split-input-file %s | FileCheck %s
// RUN: mlir-opt -one-shot-bufferize -split-input-file %s | FileCheck %s --check-prefix=CHECK-NO-FUNC

// CHECK-NO-FUNC-LABEL: func @br(
//  CHECK-NO-FUNC-SAME:     %[[t:.*]]: tensor<5xf32>)
//       CHECK-NO-FUNC:   %[[m:.*]] = bufferization.to_memref %[[t]] : memref<5xf32, strided<[?], offset: ?>>
//       CHECK-NO-FUNC:   %[[r:.*]] = scf.execute_region -> memref<5xf32, strided<[?], offset: ?>> {
//       CHECK-NO-FUNC:     cf.br ^[[block:.*]](%[[m]]
//       CHECK-NO-FUNC:   ^[[block]](%[[arg1:.*]]: memref<5xf32, strided<[?], offset: ?>>):
//       CHECK-NO-FUNC:     scf.yield %[[arg1]]
//       CHECK-NO-FUNC:   }
//       CHECK-NO-FUNC:   return
func.func @br(%t: tensor<5xf32>) {
  %0 = scf.execute_region -> tensor<5xf32> {
    cf.br ^bb1(%t : tensor<5xf32>)
  ^bb1(%arg1 : tensor<5xf32>):
    scf.yield %arg1 : tensor<5xf32>
  }
  return
}

// -----

// CHECK-NO-FUNC-LABEL: func @cond_br(
//  CHECK-NO-FUNC-SAME:     %[[t1:.*]]: tensor<5xf32>,
//       CHECK-NO-FUNC:   %[[m1:.*]] = bufferization.to_memref %[[t1]] : memref<5xf32, strided<[?], offset: ?>>
//       CHECK-NO-FUNC:   %[[alloc:.*]] = memref.alloc() {{.*}} : memref<5xf32>
//       CHECK-NO-FUNC:   %[[r:.*]] = scf.execute_region -> memref<5xf32, strided<[?], offset: ?>> {
//       CHECK-NO-FUNC:     cf.cond_br %{{.*}}, ^[[block1:.*]](%[[m1]] : {{.*}}), ^[[block2:.*]](%[[alloc]] : {{.*}})
//       CHECK-NO-FUNC:   ^[[block1]](%[[arg1:.*]]: memref<5xf32, strided<[?], offset: ?>>):
//       CHECK-NO-FUNC:     scf.yield %[[arg1]]
//       CHECK-NO-FUNC:   ^[[block2]](%[[arg2:.*]]: memref<5xf32>):
//       CHECK-NO-FUNC:     %[[cast:.*]] = memref.cast %[[arg2]] : memref<5xf32> to memref<5xf32, strided<[?], offset: ?>
//       CHECK-NO-FUNC:     cf.br ^[[block1]](%[[cast]] : {{.*}})
//       CHECK-NO-FUNC:   }
//       CHECK-NO-FUNC:   return
func.func @cond_br(%t1: tensor<5xf32>, %c: i1) {
  // Use an alloc for the second block instead of a function block argument.
  // A cast must be inserted because the two will have different layout maps.
  %t0 = bufferization.alloc_tensor() : tensor<5xf32>
  %0 = scf.execute_region -> tensor<5xf32> {
    cf.cond_br %c, ^bb1(%t1 : tensor<5xf32>), ^bb2(%t0 : tensor<5xf32>)
  ^bb1(%arg1 : tensor<5xf32>):
    scf.yield %arg1 : tensor<5xf32>
  ^bb2(%arg2 : tensor<5xf32>):
    cf.br ^bb1(%arg2 : tensor<5xf32>)
  }
  return
}

// -----

// CHECK-LABEL: func @looping_branches(
func.func @looping_branches() -> tensor<5xf32> {
// CHECK: %[[alloc:.*]] = memref.alloc
  %0 = bufferization.alloc_tensor() : tensor<5xf32>
// CHECK: cf.br {{.*}}(%[[alloc]]
  cf.br ^bb1(%0: tensor<5xf32>)
// CHECK: ^{{.*}}(%[[arg1:.*]]: memref<5xf32>):
^bb1(%arg1: tensor<5xf32>):
  %pos = "test.foo"() : () -> (index)
  %val = "test.bar"() : () -> (f32)
// CHECK: memref.store %{{.*}}, %[[arg1]]
  %inserted = tensor.insert %val into %arg1[%pos] : tensor<5xf32>
  %cond = "test.qux"() : () -> (i1)
// CHECK: cf.cond_br {{.*}}(%[[arg1]] {{.*}}(%[[arg1]]
  cf.cond_br %cond, ^bb1(%inserted: tensor<5xf32>), ^bb2(%inserted: tensor<5xf32>)
// CHECK: ^{{.*}}(%[[arg2:.*]]: memref<5xf32>):
^bb2(%arg2: tensor<5xf32>):
// CHECK: return %[[arg2]]
  func.return %arg2 : tensor<5xf32>
}
