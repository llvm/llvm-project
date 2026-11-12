// RUN: mlir-opt %s -offload-livein-value-canonicalization -split-input-file | FileCheck %s

// -----

// Test constant sinking: when all uses are inside the region, sink the op.
func.func private @use_i64(i64) -> ()

func.func @test_constant_sink() {
  %c1 = arith.constant 1 : i64
  acc.serial {
    func.call @use_i64(%c1) : (i64) -> ()
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_constant_sink
// CHECK-NEXT: acc.serial {
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:   func.call @use_i64(%[[C1]])

// -----

// Test constant rematerialization: when uses exist both inside and outside,
// clone the op inside the region.
func.func private @use_i64(i64) -> ()

func.func @test_constant_rematerialize() {
  %c1 = arith.constant 1 : i64
  func.call @use_i64(%c1) : (i64) -> ()
  acc.serial {
    func.call @use_i64(%c1) : (i64) -> ()
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_constant_rematerialize
// CHECK: %[[C1_OUTER:.*]] = arith.constant 1 : i64
// CHECK: call @use_i64(%[[C1_OUTER]])
// CHECK: acc.serial {
// CHECK:   %[[C1_INNER:.*]] = arith.constant 1 : i64
// CHECK:   func.call @use_i64(%[[C1_INNER]])

// -----

// Test acc.bounds sinking
// Note: Using orphan acc.copyin inside compute region is not strictly valid IR,
// but using acc.private or similar requires recipe declarations which
// complicates the test. The important thing is testing bounds sinking.
func.func @test_accbounds_sink(%arg0: memref<10xf32>) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %bounds = acc.bounds lowerbound(%c0 : index) upperbound(%c10 : index)
  acc.serial {
    %copy = acc.copyin varPtr(%arg0 : memref<10xf32>) bounds(%bounds) -> memref<10xf32>
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_accbounds_sink
// CHECK: acc.serial {
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C10:.*]] = arith.constant 10 : index
// CHECK:   %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[C0]] : index) upperbound(%[[C10]] : index)
// CHECK:   acc.copyin varPtr({{.*}}) bounds(%[[BOUNDS]])

// -----

// Test acc.bounds rematerialization (bounds used both inside and outside)
// Note: Using orphan acc.copyin is not strictly valid IR (see comment above).
func.func @test_accbounds_rematerialize(%arg0: memref<10xf32>) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %bounds = acc.bounds lowerbound(%c0 : index) upperbound(%c10 : index)
  %copy_outer = acc.copyin varPtr(%arg0 : memref<10xf32>) bounds(%bounds) -> memref<10xf32>
  acc.serial {
    %copy_inner = acc.copyin varPtr(%arg0 : memref<10xf32>) bounds(%bounds) -> memref<10xf32>
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_accbounds_rematerialize
// CHECK: %[[BOUNDS_OUTER:.*]] = acc.bounds
// CHECK: acc.copyin varPtr({{.*}}) bounds(%[[BOUNDS_OUTER]])
// CHECK: acc.serial {
// CHECK:   %[[BOUNDS_INNER:.*]] = acc.bounds
// CHECK:   acc.copyin varPtr({{.*}}) bounds(%[[BOUNDS_INNER]])

// -----

// Test memref.get_global with acc.declare sinking
memref.global @memref_global_with_declare : memref<10xf32> = dense<0.0> {acc.declare = #acc.declare<dataClause = acc_copyin>}

func.func private @use_memref(memref<10xf32>) -> ()

func.func @test_memref_get_global_sink() {
  %memref = memref.get_global @memref_global_with_declare : memref<10xf32>
  acc.serial {
    func.call @use_memref(%memref) : (memref<10xf32>) -> ()
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_memref_get_global_sink
// CHECK: acc.serial {
// CHECK:   %[[MEM:.*]] = memref.get_global @memref_global_with_declare
// CHECK:   func.call @use_memref(%[[MEM]])

// -----

// Test memref.reinterpret_cast traces through to get_global
memref.global @memref_global_reinterpret : memref<2x5xf32> = dense<0.0> {acc.declare = #acc.declare<dataClause = acc_copyin>}

func.func private @use_memref_1d(memref<10xf32>) -> ()

func.func @test_memref_reinterpret_cast_sink() {
  %memref = memref.get_global @memref_global_reinterpret : memref<2x5xf32>
  %reinterpreted = memref.reinterpret_cast %memref to offset: [0], sizes: [10], strides: [1] : memref<2x5xf32> to memref<10xf32>
  acc.serial {
    func.call @use_memref_1d(%reinterpreted) : (memref<10xf32>) -> ()
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_memref_reinterpret_cast_sink
// CHECK: acc.serial {
// CHECK:   memref.get_global @memref_global_reinterpret
// CHECK:   memref.reinterpret_cast

// -----

// Test with acc.parallel (another OffloadRegionOpInterface)
func.func private @use_i32(i32) -> ()

func.func @test_parallel_region() {
  %c42 = arith.constant 42 : i32
  acc.parallel {
    func.call @use_i32(%c42) : (i32) -> ()
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_parallel_region
// CHECK: acc.parallel {
// CHECK:   %[[C42:.*]] = arith.constant 42 : i32
// CHECK:   func.call @use_i32(%[[C42]])

// -----

// Test with acc.kernels (another OffloadRegionOpInterface)
func.func private @use_f32(f32) -> ()

func.func @test_kernels_region() {
  %cst = arith.constant 3.14 : f32
  acc.kernels {
    func.call @use_f32(%cst) : (f32) -> ()
    acc.terminator
  }
  return
}

// CHECK-LABEL: @test_kernels_region
// CHECK: acc.kernels {
// CHECK:   %[[CST:.*]] = arith.constant 3.14{{.*}} : f32
// CHECK:   func.call @use_f32(%[[CST]])

// -----

// Test multiple constants with mixed sinking/rematerialization
func.func private @use_index(index) -> ()

func.func @test_multiple_constants() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  func.call @use_index(%c1) : (index) -> ()
  acc.serial {
    func.call @use_index(%c1) : (index) -> ()
    func.call @use_index(%c2) : (index) -> ()
    acc.yield
  }
  return
}

// CHECK-LABEL: @test_multiple_constants
// CHECK: %[[C1_OUTER:.*]] = arith.constant 1 : index
// CHECK: call @use_index(%[[C1_OUTER]])
// CHECK: acc.serial {
// CHECK-DAG: arith.constant 1 : index
// CHECK-DAG: arith.constant 2 : index

// -----

// Test with gpu.launch (another OffloadRegionOpInterface)
func.func private @use_index(index) -> ()

func.func @test_gpu_launch_region() {
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
    func.call @use_index(%c42) : (index) -> ()
    gpu.terminator
  }
  return
}

// CHECK-LABEL: @test_gpu_launch_region
// CHECK: gpu.launch
// CHECK:   %[[C42:.*]] = arith.constant 42 : index
// CHECK:   func.call @use_index(%[[C42]])

// -----

// Test gpu.launch with constant rematerialization
func.func private @use_index(index) -> ()

func.func @test_gpu_launch_rematerialize() {
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : index
  func.call @use_index(%c42) : (index) -> ()
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
    func.call @use_index(%c42) : (index) -> ()
    gpu.terminator
  }
  return
}

// CHECK-LABEL: @test_gpu_launch_rematerialize
// CHECK: %[[C42_OUTER:.*]] = arith.constant 42 : index
// CHECK: call @use_index(%[[C42_OUTER]])
// CHECK: gpu.launch
// CHECK:   %[[C42_INNER:.*]] = arith.constant 42 : index
// CHECK:   func.call @use_index(%[[C42_INNER]])
