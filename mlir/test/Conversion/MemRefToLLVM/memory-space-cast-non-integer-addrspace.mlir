// RUN: mlir-opt %s -finalize-memref-to-llvm | FileCheck %s
// Regression test for https://github.com/llvm/llvm-project/issues/186041.
// memref.memory_space_cast to a non-integer address space should not crash when
// the MemRefToLLVM type conversion fails for the result type. The op is left
// unconverted (partial conversion succeeds, exit 0).

// CHECK-LABEL: @memory_space_cast_non_integer_addrspace
// CHECK: memref.memory_space_cast
func.func @memory_space_cast_non_integer_addrspace(%arg0: memref<128xi32, 3>) {
  %cast = memref.memory_space_cast %arg0 : memref<128xi32, 3> to memref<128xi32, #gpu.address_space<workgroup>>
  return
}
