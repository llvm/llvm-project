// RUN: mlir-opt %s -test-lower-to-llvm | \
// RUN: mlir-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils | \
// RUN: FileCheck %s

/// This tests printing (multiple) string literals works.

func.func @entry() {
   // CHECK: Hello, World!
   vector.print str "Hello, World!\n"
   // CHECK-NEXT: Bye!
   vector.print str "Bye!\n"
   return
}
