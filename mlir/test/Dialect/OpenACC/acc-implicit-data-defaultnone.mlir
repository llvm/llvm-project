// RUN: mlir-opt %s -acc-implicit-data=ignore-default-none=false -split-input-file | FileCheck %s
// RUN: mlir-opt %s -acc-implicit-data=ignore-default-none=true -split-input-file | FileCheck %s --check-prefix=ENABLED

// -----

// Test scalar in parallel with default(none) - should NOT generate implicit
// data when ignore-default-none is disabled. Otherwise, it should generate
// implicit firstprivate.
func.func @test_scalar_parallel_defaultnone() {
  %alloc = memref.alloca() : memref<f32>
  acc.parallel {
    %load = memref.load %alloc[] : memref<f32>
    acc.yield
  } attributes {defaultAttr = #acc<defaultvalue none>}
  return
}

// CHECK-LABEL: func.func @test_scalar_parallel_defaultnone
// CHECK-NOT: acc.firstprivate
// CHECK-NOT: acc.copyin

// ENABLED-LABEL: func.func @test_scalar_parallel_defaultnone
// ENABLED: acc.firstprivate varPtr({{.*}} : memref<f32>) recipe({{.*}}) -> memref<f32> {implicit = true, name = ""}
