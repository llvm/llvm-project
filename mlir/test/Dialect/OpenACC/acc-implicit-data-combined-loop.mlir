// RUN: mlir-opt %s -acc-implicit-data -split-input-file | FileCheck %s
// RUN: mlir-opt %s -acc-implicit-data=enable-combined-loop-implicit-firstprivate=false -split-input-file | FileCheck %s --check-prefix=FLAG-OFF

// Combined parallel loop: implicit scalar firstprivate is on the loop only.
func.func @test_scalar_combined_parallel_loop() {
  %alloc = memref.alloca() : memref<f32>
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  acc.parallel combined(loop) {
    acc.loop combined(parallel) control(%iv : index) = (%c0 : index) to (%c10 : index) step (%c1 : index) {
      %load = memref.load %alloc[] : memref<f32>
      acc.yield
    } independent
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @test_scalar_combined_parallel_loop
// CHECK: %[[FP:.*]] = acc.firstprivate varPtr({{.*}} : memref<f32>) recipe({{.*}}) implicit(true) name("") -> memref<f32>
// CHECK-NOT: acc.parallel{{.*}}firstprivate
// CHECK: acc.parallel combined(loop)
// CHECK: acc.loop combined(parallel) {{.*}}firstprivate(%[[FP]] : memref<f32>)

// FLAG-OFF-LABEL: func.func @test_scalar_combined_parallel_loop
// FLAG-OFF: %[[FP:.*]] = acc.firstprivate varPtr({{.*}} : memref<f32>) recipe({{.*}}) implicit(true) name("") -> memref<f32>
// FLAG-OFF: acc.parallel combined(loop) firstprivate(%[[FP]] : memref<f32>)
// FLAG-OFF-NOT: acc.loop{{.*}}firstprivate

// -----

// Combined serial loop: same loop-only firstprivate.
func.func @test_scalar_combined_serial_loop() {
  %alloc = memref.alloca() : memref<i64>
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  acc.serial combined(loop) {
    acc.loop combined(serial) control(%iv : index) = (%c0 : index) to (%c10 : index) step (%c1 : index) {
      %load = memref.load %alloc[] : memref<i64>
      acc.yield
    } seq
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @test_scalar_combined_serial_loop
// CHECK: %[[FP:.*]] = acc.firstprivate varPtr({{.*}} : memref<i64>) recipe({{.*}}) implicit(true) name("") -> memref<i64>
// CHECK-NOT: acc.serial{{.*}}firstprivate
// CHECK: acc.serial combined(loop)
// CHECK: acc.loop combined(serial) {{.*}}firstprivate(%[[FP]] : memref<i64>)

// FLAG-OFF-LABEL: func.func @test_scalar_combined_serial_loop
// FLAG-OFF: %[[FP:.*]] = acc.firstprivate varPtr({{.*}} : memref<i64>) recipe({{.*}}) implicit(true) name("") -> memref<i64>
// FLAG-OFF: acc.serial combined(loop) firstprivate(%[[FP]] : memref<i64>)
// FLAG-OFF-NOT: acc.loop{{.*}}firstprivate

// -----

// Non-combined parallel + nested loop keeps firstprivate on the compute op.
func.func @test_scalar_noncombined_parallel_nested_loop() {
  %alloc = memref.alloca() : memref<f32>
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  acc.parallel {
    acc.loop control(%iv : index) = (%c0 : index) to (%c10 : index) step (%c1 : index) {
      %load = memref.load %alloc[] : memref<f32>
      acc.yield
    } independent
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @test_scalar_noncombined_parallel_nested_loop
// CHECK: %[[FP:.*]] = acc.firstprivate varPtr({{.*}} : memref<f32>) recipe({{.*}}) implicit(true) name("") -> memref<f32>
// CHECK: acc.parallel firstprivate(%[[FP]] : memref<f32>)
// CHECK-NOT: acc.loop{{.*}}firstprivate

// FLAG-OFF-LABEL: func.func @test_scalar_noncombined_parallel_nested_loop
// FLAG-OFF: %[[FP:.*]] = acc.firstprivate varPtr({{.*}} : memref<f32>) recipe({{.*}}) implicit(true) name("") -> memref<f32>
// FLAG-OFF: acc.parallel firstprivate(%[[FP]] : memref<f32>)
// FLAG-OFF-NOT: acc.loop{{.*}}firstprivate
