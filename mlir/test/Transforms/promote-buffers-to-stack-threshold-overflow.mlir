// RUN: mlir-opt -promote-buffers-to-stack="max-alloc-size-in-bytes=536870912" %s | FileCheck %s

// CHECK-LABEL: func @thresholdOverflow
func.func @thresholdOverflow(%value: i8) -> i8 {
  %c0 = arith.constant 0 : index
  // CHECK: memref.alloca()
  %buffer = memref.alloc() : memref<1xi8>
  memref.store %value, %buffer[%c0] : memref<1xi8>
  %result = memref.load %buffer[%c0] : memref<1xi8>
  return %result : i8
}
