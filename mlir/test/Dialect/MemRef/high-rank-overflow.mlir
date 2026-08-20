// RUN: mlir-opt %s --convert-to-llvm --split-input-file --verify-diagnostics | FileCheck %s

// Test that extremely high-rank memrefs with overflow in stride calculation
// are handled gracefully instead of crashing (issue #177816).

// CHECK-LABEL: func @high_rank_memref_overflow
func.func @high_rank_memref_overflow() {
  // This creates a memref with 64 dimensions of size 2, resulting in 2^64 elements
  // which overflows int64_t. The stride calculation should handle this gracefully.
  %0 = memref.alloc() : memref<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2xi32>
  return
}

// -----

// CHECK-LABEL: func @high_rank_memref_max_dim
func.func @high_rank_memref_max_dim() {
  // Test with fewer dimensions but larger sizes that also cause overflow
  %0 = memref.alloc() : memref<9223372036854775807x2xi32>
  return
}
