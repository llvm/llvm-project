// RUN: mlir-opt %s -test-lower-to-llvm  | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

//
// Test various floating-point types.
//
func.func @entry() {
  %0 = arith.constant dense<[-1000.0, -1.1, 0.0, 1.1, 1000.0]> : vector<5xf64>
  vector.print %0 : vector<5xf64>
  // CHECK: ( -1000, -1.1, 0, 1.1, 1000 )

  %1 = arith.constant dense<[-1000.0, -1.1, 0.0, 1.1, 1000.0]> : vector<5xf32>
  vector.print %1 : vector<5xf32>
  // CHECK: ( -1000, -1.1, 0, 1.1, 1000 )

  %2 = arith.constant dense<[-1000.0, -1.1, 0.0, 1.1, 1000.0]> : vector<5xf16>
  vector.print %2 : vector<5xf16>
  // CHECK: ( -1000, -1.09961, 0, 1.09961, 1000 )

  %3 = arith.constant dense<[-1000.0, -1.1, 0.0, 1.1, 1000.0]> : vector<5xbf16>
  vector.print %3 : vector<5xbf16>
  // CHECK: ( -1000, -1.10156, 0, 1.10156, 1000 )

  return
}
