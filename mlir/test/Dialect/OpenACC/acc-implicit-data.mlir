// RUN: mlir-opt %s -acc-implicit-data -split-input-file | FileCheck %s

// -----

// Test scalar in serial construct - should generate firstprivate
func.func @test_scalar_in_serial() {
  %alloc = memref.alloca() : memref<i64>
  acc.serial {
    %load = memref.load %alloc[] : memref<i64>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @test_scalar_in_serial
// CHECK: acc.firstprivate varPtr({{.*}} : memref<i64>) -> memref<i64> {implicit = true, name = ""}

// -----

// Test scalar in parallel construct - should generate firstprivate
func.func @test_scalar_in_parallel() {
  %alloc = memref.alloca() : memref<f32>
  acc.parallel {
    %load = memref.load %alloc[] : memref<f32>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @test_scalar_in_parallel
// CHECK: acc.firstprivate varPtr({{.*}} : memref<f32>) -> memref<f32> {implicit = true, name = ""}

// -----

// Test scalar in kernels construct - should generate copyin/copyout
func.func @test_scalar_in_kernels() {
  %alloc = memref.alloca() : memref<f64>
  acc.kernels {
    %load = memref.load %alloc[] : memref<f64>
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @test_scalar_in_kernels
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr({{.*}} : memref<f64>) -> memref<f64> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = ""}
// CHECK: acc.copyout accPtr(%[[COPYIN]] : memref<f64>) to varPtr({{.*}} : memref<f64>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = ""}

// -----

// Test scalar in parallel with default(none) - should NOT generate implicit data
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

// -----

// Test array in parallel - should generate copyin/copyout
func.func @test_array_in_parallel() {
  %alloc = memref.alloca() : memref<10xf32>
  acc.parallel {
    %c0 = arith.constant 0 : index
    %load = memref.load %alloc[%c0] : memref<10xf32>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @test_array_in_parallel
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr({{.*}} : memref<10xf32>) -> memref<10xf32> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = ""}
// CHECK: acc.copyout accPtr(%[[COPYIN]] : memref<10xf32>) to varPtr({{.*}} : memref<10xf32>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = ""}

// -----

// Test array in kernels - should generate copyin/copyout
func.func @test_array_in_kernels() {
  %alloc = memref.alloca() : memref<20xi32>
  acc.kernels {
    %c0 = arith.constant 0 : index
    %load = memref.load %alloc[%c0] : memref<20xi32>
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @test_array_in_kernels
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr({{.*}} : memref<20xi32>) -> memref<20xi32> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = ""}
// CHECK: acc.copyout accPtr(%[[COPYIN]] : memref<20xi32>) to varPtr({{.*}} : memref<20xi32>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = ""}

// -----

// Test array with default(present) - should generate present
func.func @test_array_parallel_defaultpresent() {
  %alloc = memref.alloca() : memref<10xf32>
  acc.parallel {
    %c0 = arith.constant 0 : index
    %load = memref.load %alloc[%c0] : memref<10xf32>
    acc.yield
  } attributes {defaultAttr = #acc<defaultvalue present>}
  return
}

// CHECK-LABEL: func.func @test_array_parallel_defaultpresent
// CHECK: %[[PRESENT:.*]] = acc.present varPtr({{.*}} : memref<10xf32>) -> memref<10xf32> {implicit = true, name = ""}
// CHECK: acc.delete accPtr(%[[PRESENT]] : memref<10xf32>) {dataClause = #acc<data_clause acc_present>, implicit = true, name = ""}

// -----

// Test scalar with default(present) - should still generate firstprivate (scalars ignore default(present))
func.func @test_scalar_parallel_defaultpresent() {
  %alloc = memref.alloca() : memref<f32>
  acc.parallel {
    %load = memref.load %alloc[] : memref<f32>
    acc.yield
  } attributes {defaultAttr = #acc<defaultvalue present>}
  return
}

// CHECK-LABEL: func.func @test_scalar_parallel_defaultpresent
// CHECK: acc.firstprivate varPtr({{.*}} : memref<f32>) -> memref<f32> {implicit = true, name = ""}

// -----

// Test multidimensional array
func.func @test_multidim_array_in_parallel() {
  %alloc = memref.alloca() : memref<8x16xf32>
  acc.parallel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %load = memref.load %alloc[%c0, %c1] : memref<8x16xf32>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @test_multidim_array_in_parallel
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr({{.*}} : memref<8x16xf32>) -> memref<8x16xf32> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = ""}
// CHECK: acc.copyout accPtr(%[[COPYIN]] : memref<8x16xf32>) to varPtr({{.*}} : memref<8x16xf32>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = ""}

// -----

// Test dynamic size array
func.func @test_dynamic_array(%size: index) {
  %alloc = memref.alloca(%size) : memref<?xf64>
  acc.parallel {
    %c0 = arith.constant 0 : index
    %load = memref.load %alloc[%c0] : memref<?xf64>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @test_dynamic_array
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr({{.*}} : memref<?xf64>) -> memref<?xf64> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = ""}
// CHECK: acc.copyout accPtr(%[[COPYIN]] : memref<?xf64>) to varPtr({{.*}} : memref<?xf64>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = ""}

// -----

// Test variable with explicit data clause - implicit should recognize it
func.func @test_with_explicit_copyin() {
  %alloc = memref.alloca() : memref<100xf32>
  %copyin = acc.copyin varPtr(%alloc : memref<100xf32>) -> memref<100xf32> {name = "explicit"}
  acc.parallel dataOperands(%copyin : memref<100xf32>) {
    %c0 = arith.constant 0 : index
    %load = memref.load %alloc[%c0] : memref<100xf32>
    acc.yield
  }
  acc.copyout accPtr(%copyin : memref<100xf32>) to varPtr(%alloc : memref<100xf32>) {name = "explicit"}
  return
}

// CHECK-LABEL: func.func @test_with_explicit_copyin
// CHECK: acc.present varPtr({{.*}} : memref<100xf32>) -> memref<100xf32> {implicit = true, name = ""}

// -----

// Test multiple variables
func.func @test_multiple_variables() {
  %alloc1 = memref.alloca() : memref<f32>
  %alloc2 = memref.alloca() : memref<10xi32>
  acc.parallel {
    %load1 = memref.load %alloc1[] : memref<f32>
    %c0 = arith.constant 0 : index
    %load2 = memref.load %alloc2[%c0] : memref<10xi32>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @test_multiple_variables
// CHECK: acc.firstprivate varPtr({{.*}} : memref<f32>) -> memref<f32> {implicit = true, name = ""}
// CHECK: %[[COPYIN:.*]] = acc.copyin varPtr({{.*}} : memref<10xi32>) -> memref<10xi32> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = ""}
// CHECK: acc.copyout accPtr(%[[COPYIN]] : memref<10xi32>) to varPtr({{.*}} : memref<10xi32>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = ""}

// -----

// Test memref.view aliasing - view of explicitly copied buffer should generate present
func.func @test_memref_view(%size: index) {
  %c0 = arith.constant 0 : index
  %buffer = memref.alloca(%size) : memref<?xi8>
  %copyin = acc.copyin varPtr(%buffer : memref<?xi8>) -> memref<?xi8> {name = "buffer"}
  %view = memref.view %buffer[%c0][] : memref<?xi8> to memref<8x64xf32>
  acc.kernels dataOperands(%copyin : memref<?xi8>) {
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %load = memref.load %view[%c0_0, %c0_1] : memref<8x64xf32>
    acc.terminator
  }
  acc.copyout accPtr(%copyin : memref<?xi8>) to varPtr(%buffer : memref<?xi8>) {name = "buffer"}
  return
}

// CHECK-LABEL: func.func @test_memref_view
// CHECK: acc.present varPtr({{.*}} : memref<8x64xf32>) -> memref<8x64xf32> {implicit = true, name = ""}

