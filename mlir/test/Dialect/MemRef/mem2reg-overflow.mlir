# RUN: mlir-opt %s -mem2reg -verify-diagnostics | FileCheck %s

# CHECK: module
module {
  // This used to crash mlir-opt with the assertion in BuiltinTypeInterfaces.cpp:86
  // (9223372036854775807 * 3 overflowing)
  func.func @alloca_unconvertable_memory_space() {
    return
  ^bb1(%0: index):
    %alloca = memref.alloca() : memref<9223372036854775807x3xi32>
    return
  }

  // This is a copy of the crashing input from the bug report
  // It should now run successfully with no assertion and no crash
  func.func @test_mem2reg_works() {
    %0 = arith.constant 42 : i32
    %1 = memref.alloca() : memref<9223372036854775807x3xi32>
    memref.store %0, %1[%0] : memref<9223372036854775807x3xi32>
    return
  }
}
