// RUN: tr-opt %s --tr-recognize-load-reduce | FileCheck %s

// CHECK-LABEL: func.func @row_partial
func.func @row_partial(%in: !tr.buffer<MxKxf32>, %i: index, %j: index)
    -> !tr.tile<128xf32> {
  %t = tr.load %in[%i, %j] : !tr.buffer<MxKxf32>, !tr.tile<128x128xf32>
  // CHECK: tr.reduce_sum {{.*}} {tr.load_reduce}
  %p = tr.reduce_sum %t, axis = 1 : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
  return %p : !tr.tile<128xf32>
}

// CHECK-LABEL: func.func @multi_use
func.func @multi_use(%in: !tr.buffer<MxKxf32>, %i: index, %j: index)
    -> (!tr.tile<128x128xf32>, !tr.tile<128xf32>) {
  %t = tr.load %in[%i, %j] : !tr.buffer<MxKxf32>, !tr.tile<128x128xf32>
  %p = tr.reduce_sum %t, axis = 1 : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
  // CHECK-NOT: tr.load_reduce
  return %t, %p : !tr.tile<128x128xf32>, !tr.tile<128xf32>
}
