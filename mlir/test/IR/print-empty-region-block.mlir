// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// test.switch_with_no_break requires exactly one block per case region. Check
// that empty blocks are printed, so the printed IR parses and verifies again.
// CHECK-LABEL: func.func @empty_case_blocks
func.func @empty_case_blocks(%arg: index) {
  // CHECK: test.switch_with_no_break
  test.switch_with_no_break %arg
  // CHECK: case 0 {
  // CHECK-NEXT: ^bb0:
  // CHECK-NEXT: }
  case 0 {
  ^bb:
  }
  // CHECK: case 1 {
  // CHECK-NEXT: ^bb0:
  // CHECK-NEXT: }
  case 1 {
  ^bb:
  }
  return
}
