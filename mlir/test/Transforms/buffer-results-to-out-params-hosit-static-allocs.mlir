// RUN: mlir-opt -allow-unregistered-dialect -p 'builtin.module(buffer-results-to-out-params{hoist-static-allocs})'  %s | FileCheck %s

// CHECK-LABEL:   func private @basic(
// CHECK-SAME:                %[[ARG:.*]]: memref<8x64xf32>) {
// CHECK-NOT:        memref.alloc()
// CHECK:           "test.source"(%[[ARG]])  : (memref<8x64xf32>) -> ()
// CHECK:           return
// CHECK:         }
func.func private @basic() -> (memref<8x64xf32>) {
  %b = memref.alloc() : memref<8x64xf32>
  "test.source"(%b)  : (memref<8x64xf32>) -> ()
  return %b : memref<8x64xf32>
}

// CHECK-LABEL:   func private @basic_no_change(
// CHECK-SAME:                %[[ARG:.*]]: memref<f32>) {
// CHECK:           %[[RESULT:.*]] = "test.source"() : () -> memref<f32>
// CHECK:           memref.copy %[[RESULT]], %[[ARG]]  : memref<f32> to memref<f32>
// CHECK:           return
// CHECK:         }
func.func private @basic_no_change() -> (memref<f32>) {
  %0 = "test.source"() : () -> (memref<f32>)
  return %0 : memref<f32>
}

// CHECK-LABEL:   func private @basic_dynamic(
// CHECK-SAME:                %[[D:.*]]: index, %[[ARG:.*]]: memref<?xf32>) {
// CHECK:           %[[RESULT:.*]] = memref.alloc(%[[D]]) : memref<?xf32>
// CHECK:           "test.source"(%[[RESULT]])  : (memref<?xf32>) -> ()
// CHECK:           memref.copy %[[RESULT]], %[[ARG]]
// CHECK:           return
// CHECK:         }
func.func private @basic_dynamic(%d: index) -> (memref<?xf32>) {
  %b = memref.alloc(%d) : memref<?xf32>
  "test.source"(%b)  : (memref<?xf32>) -> ()
  return %b : memref<?xf32>
}

// -----

// no change due to writing to func args
// CHECK-LABEL:   func private @return_arg(
// CHECK-SAME:        %[[ARG0:.*]]: memref<128x256xf32>, %[[ARG1:.*]]: memref<128x256xf32>, %[[ARG2:.*]]: memref<128x256xf32>) {
// CHECK:           "test.source"(%[[ARG0]], %[[ARG1]])
// CHECK:           memref.copy
// CHECK:           return
// CHECK:         }
func.func private @return_arg(%arg0: memref<128x256xf32>, %arg1: memref<128x256xf32>) -> memref<128x256xf32> {
  "test.source"(%arg0, %arg1)  : (memref<128x256xf32>, memref<128x256xf32>) -> ()
  return %arg0 : memref<128x256xf32>
}

// CHECK-LABEL: func.func private @duplicate_return_value(
// CHECK-SAME:    %[[ARG0:.*]]: memref<4xf32>, %[[ARG1:.*]]: memref<4xf32>) {
// CHECK-NOT:     memref.alloc
// CHECK:         memref.copy %[[ARG0]], %[[ARG1]] : memref<4xf32> to memref<4xf32>
// CHECK:         return
func.func private @duplicate_return_value()
    -> (memref<4xf32>, memref<4xf32>) {
  %a = memref.alloc() : memref<4xf32>
  return %a, %a : memref<4xf32>, memref<4xf32>
}

// -----

// CHECK-LABEL: func.func private @multiple_duplicate_returns(
// CHECK-SAME:    %[[COND:.*]]: i1, %[[OUT0:.*]]: memref<4xf32>, %[[OUT1:.*]]: memref<4xf32>) {
// CHECK:        cf.cond_br %[[COND]], ^[[THEN:.*]], ^[[ELSE:.*]]
// CHECK:      ^[[THEN]]:
// CHECK:        memref.copy %[[OUT0]], %[[OUT1]] : memref<4xf32> to memref<4xf32>
// CHECK:        return
// CHECK:      ^[[ELSE]]:
// CHECK:        memref.copy %[[OUT0]], %[[OUT1]] : memref<4xf32> to memref<4xf32>
// CHECK:        return
func.func private @multiple_duplicate_returns(%cond: i1)
    -> (memref<4xf32>, memref<4xf32>) {
  %a = memref.alloc() : memref<4xf32>
  cf.cond_br %cond, ^then, ^else
^then:
  return %a, %a : memref<4xf32>, memref<4xf32>
^else:
  return %a, %a : memref<4xf32>, memref<4xf32>
}

// -----

// CHECK-LABEL: func.func private @multiple_returns_with_non_alloc(
// CHECK-SAME:    %[[INPUT:.*]]: memref<4xf32>, %{{.*}}: i1, %[[OUT0:.*]]: memref<4xf32>, %[[OUT1:.*]]: memref<4xf32>) {
// CHECK:        cf.cond_br %{{.*}}, ^[[THEN:.*]], ^[[ELSE:.*]]
// CHECK:      ^[[THEN]]:
// CHECK:        memref.copy %[[OUT0]], %[[OUT1]] : memref<4xf32> to memref<4xf32>
// CHECK:        return
// CHECK:      ^[[ELSE]]:
// CHECK:        memref.copy %[[INPUT]], %[[OUT0]] : memref<4xf32> to memref<4xf32>
// CHECK:        memref.copy %[[INPUT]], %[[OUT1]] : memref<4xf32> to memref<4xf32>
// CHECK:        return
func.func private @multiple_returns_with_non_alloc(%input: memref<4xf32>, %cond: i1)
    -> (memref<4xf32>, memref<4xf32>) {
  %a = memref.alloc() : memref<4xf32>
  cf.cond_br %cond, ^then, ^else
^then:
  return %a, %a : memref<4xf32>, memref<4xf32>
^else:
  return %input, %input : memref<4xf32>, memref<4xf32>
}
