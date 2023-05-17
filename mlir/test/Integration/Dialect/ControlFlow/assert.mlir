// RUN: mlir-opt %s -test-cf-assert \
// RUN:     -convert-func-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void | \
// RUN: FileCheck %s

func.func @main() {
  %a = arith.constant 0 : i1
  %b = arith.constant 1 : i1
  // CHECK: assertion foo
  cf.assert %a, "assertion foo"
  // CHECK-NOT: assertion bar
  cf.assert %b, "assertion bar"
  return
}
