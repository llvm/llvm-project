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
