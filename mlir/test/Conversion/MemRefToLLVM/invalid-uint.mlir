// RUN: mlir-opt %s -finalize-memref-to-llvm -verify-diagnostics

// CHECK-LABEL: @invalid_int_conversion
func.func @invalid_int_conversion() {
     // expected-error@+1 {{conversion of memref memory space 1 : ui64 to integer address space failed. Consider adding memory space conversions.}}
     %alloc = memref.alloc() {alignment = 64 : i64} : memref<10xf32, 1 : ui64> 
    return
}
