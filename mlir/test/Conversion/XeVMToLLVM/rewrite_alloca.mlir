// RUN: mlir-opt --convert-xevm-to-llvm --split-input-file %s | FileCheck %s

module {
  // CHECK: llvm.mlir.global internal @__global_alloca_[[G2:.*]]() {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<10 x i32>
  // CHECK: llvm.mlir.global internal @__global_alloca_[[G1:.*]]() {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<10 x i32>
  // CHECK: llvm.mlir.global internal @__global_alloca_[[G0:.*]]() {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<10 x i32>
  // CHECK: llvm.func @test_with_parent_module()
  llvm.func @test_with_parent_module() -> !llvm.ptr<3> {
    %0 = llvm.mlir.constant(10 : i32) : i32
    // CHECK: %[[VAR0:.*]] = llvm.mlir.addressof @__global_alloca_[[G0]] : !llvm.ptr<3>
    // CHECK: %[[VAR1:.*]] = llvm.mlir.addressof @__global_alloca_[[G1]] : !llvm.ptr<3>
    // CHECK: %[[VAR2:.*]] = llvm.mlir.addressof @__global_alloca_[[G2]] : !llvm.ptr<3>
    %1 = llvm.alloca %0 x i32 {alignment = 8 : i64} : (i32) -> !llvm.ptr<3>
    %2 = llvm.alloca %0 x i32 {alignment = 8 : i64} : (i32) -> !llvm.ptr<3>
    %3 = llvm.alloca %0 x i32 {alignment = 8 : i64} : (i32) -> !llvm.ptr<3>
    // CHECK: %[[VAR3:.*]] = llvm.load %[[VAR1:.*]] : !llvm.ptr<3> -> i32
    // CHECK: %[[VAR4:.*]] = llvm.load %[[VAR2:.*]] : !llvm.ptr<3> -> i32
    // CHECK: %[[VAR5:.*]] = llvm.add %[[VAR3]], %[[VAR4]] : i32
    %4 = llvm.load %2 : !llvm.ptr<3> -> i32
    %5 = llvm.load %3 : !llvm.ptr<3> -> i32
    %6 = llvm.add %4, %5 : i32
    // CHECK: llvm.store %[[VAR5]], %[[VAR0]] : i32, !llvm.ptr<3>
    // CHECK: llvm.return %[[VAR0]] : !llvm.ptr<3>
    llvm.store %6, %1 : i32, !llvm.ptr<3>
    llvm.return %1 : !llvm.ptr<3>
  }
}

// -----

// CHECK-LABEL: gpu.module @test
gpu.module @test {
  // CHECK: llvm.mlir.global internal @__global_alloca_[[G2:.*]]() {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<10 x i32>
  // CHECK: llvm.mlir.global internal @__global_alloca_[[G1:.*]]() {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<10 x i32>
  // CHECK: llvm.mlir.global internal @__global_alloca_[[G0:.*]]() {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<10 x i32>
  // CHECK: llvm.func @test_with_parent_gpu_module()
  llvm.func @test_with_parent_gpu_module() -> !llvm.ptr<3> {
    %0 = llvm.mlir.constant(10 : i32) : i32
    // CHECK: %[[VAR0:.*]] = llvm.mlir.addressof @__global_alloca_[[G0]] : !llvm.ptr<3>
    // CHECK: %[[VAR1:.*]] = llvm.mlir.addressof @__global_alloca_[[G1]] : !llvm.ptr<3>
    // CHECK: %[[VAR2:.*]] = llvm.mlir.addressof @__global_alloca_[[G2]] : !llvm.ptr<3>
    %1 = llvm.alloca %0 x i32 {alignment = 8 : i64} : (i32) -> !llvm.ptr<3>
    %2 = llvm.alloca %0 x i32 {alignment = 8 : i64} : (i32) -> !llvm.ptr<3>
    %3 = llvm.alloca %0 x i32 {alignment = 8 : i64} : (i32) -> !llvm.ptr<3>
    // CHECK: %[[VAR3:.*]] = llvm.load %[[VAR1:.*]] : !llvm.ptr<3> -> i32
    // CHECK: %[[VAR4:.*]] = llvm.load %[[VAR2:.*]] : !llvm.ptr<3> -> i32
    // CHECK: %[[VAR5:.*]] = llvm.add %[[VAR3]], %[[VAR4]] : i32
    %4 = llvm.load %2 : !llvm.ptr<3> -> i32
    %5 = llvm.load %3 : !llvm.ptr<3> -> i32
    %6 = llvm.add %4, %5 : i32
    // CHECK: llvm.store %[[VAR5]], %[[VAR0]] : i32, !llvm.ptr<3>
    // CHECK: llvm.return %[[VAR0]] : !llvm.ptr<3>
    llvm.store %6, %1 : i32, !llvm.ptr<3>
    llvm.return %1 : !llvm.ptr<3>
  }
}

// -----

module {
  // CHECK-LABEL: llvm.func @test_with_default_addr_space()
  llvm.func @test_with_default_addr_space() -> !llvm.ptr {
    %0 = llvm.mlir.constant(10 : i32) : i32
    // CHECK: %[[VAR1:.*]] = llvm.alloca %0 x i32 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    // CHECK: %[[VAR2:.*]] = llvm.alloca %0 x i32 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    // CHECK: %[[VAR3:.*]] = llvm.alloca %0 x i32 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %1 = llvm.alloca %0 x i32 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x i32 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x i32 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.load %2 : !llvm.ptr -> i32
    %5 = llvm.load %3 : !llvm.ptr -> i32
    %6 = llvm.add %4, %5 : i32
    llvm.store %6, %1 : i32, !llvm.ptr
    llvm.return %1 : !llvm.ptr
  }
}
