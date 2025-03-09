// RUN: mlir-opt %s -generate-runtime-verification \
// RUN:     -test-cf-assert \
// RUN:     -convert-to-llvm | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:     -shared-libs=%mlir_runner_utils 2>&1 | \
// RUN: FileCheck %s

func.func private @notifyMemrefAllocated(%ptr : memref<*xf32>, %msg : memref<*xi8>) attributes { llvm.emit_c_interface }
func.func private @notifyMemrefDeallocated(%ptr : memref<*xf32>, %msg : memref<*xi8>) attributes { llvm.emit_c_interface }

func.func @main() {
  %c4 = arith.constant 4 : index
  %alloc = memref.alloc() : memref<1xf32>
  // memref.dealloc %alloc : memref<1xf32>
  return
}
