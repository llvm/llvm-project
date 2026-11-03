// RUN: tr-opt %s --tr-annotate-reduction-plan | FileCheck %s

// CHECK-LABEL: func.func @row
func.func @row(%t: !tr.tile<128x128xf32>) -> !tr.tile<128xf32> {
  // CHECK: tr.reduce_sum {{.*}} {tr.plan = "row"}
  %r = tr.reduce_sum %t, axis = 1 : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
  return %r : !tr.tile<128xf32>
}

// CHECK-LABEL: func.func @col
func.func @col(%t: !tr.tile<128x128xf32>) -> !tr.tile<128xf32> {
  // CHECK: tr.reduce_sum {{.*}} {tr.plan = "column"}
  %r = tr.reduce_sum %t, axis = 0 : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
  return %r : !tr.tile<128xf32>
}

// CHECK-LABEL: func.func @full
func.func @full(%t: !tr.tile<128xf32>) -> !tr.tile<f32> {
  // CHECK: tr.reduce_sum {{.*}} {tr.plan = "full"}
  %r = tr.reduce_sum %t, axis = 0 : !tr.tile<128xf32> -> !tr.tile<f32>
  return %r : !tr.tile<f32>
}
