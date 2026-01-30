// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Test that omp.target operations marked with omp.target_unreachable
// do not generate any kernel code in LLVM IR

module attributes {omp.is_target_device = false} {
  // Test 1: Target with unreachable attribute - should generate NO kernel code
  llvm.func @test_unreachable_target() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    %map = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "a"}
    omp.target map_entries(%map -> %arg0 : !llvm.ptr) {
      %2 = llvm.mlir.constant(42 : i32) : i32
      llvm.store %2, %arg0 : i32, !llvm.ptr
      omp.terminator
    } {omp.target_unreachable}
    llvm.return
  }
}

// CHECK-NOT: @__omp_offloading_{{.*}}_test_unreachable_target
