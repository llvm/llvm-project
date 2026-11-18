// RUN: mlir-opt -allow-unregistered-dialect %s -mlir-print-debuginfo | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s -mlir-print-debuginfo | mlir-opt -allow-unregistered-dialect -mlir-print-debuginfo | FileCheck %s
// Tests that ArtificialLoc round-trips correctly through the MLIR parser/printer.
// Locations shared across multiple ops are printed as aliases; verify the alias
// definition contains the expected keyword.

// CHECK-DAG: = loc(artificial)
// CHECK-DAG: = loc(unknown)

func.func @artificial_loc_on_ops() -> i32 {
  %0 = "test.op"() : () -> i32 loc(artificial)
  return %0 : i32 loc(artificial)
} loc(artificial)

func.func @mix_with_unknown() {
  // ArtificialLoc and UnknownLoc are distinct types.
  "test.op"() : () -> () loc(artificial)
  "test.op"() : () -> () loc(unknown)
  return
}
