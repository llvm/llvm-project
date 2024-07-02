// RUN: mlir-opt -convert-to-spirv %s | FileCheck %s

// CHECK-LABEL: @ub
// CHECK: %[[UNDEF:.*]] = spirv.Undef : i32
// CHECK: spirv.ReturnValue %[[UNDEF]] : i32
func.func @ub() -> index {
  %0 = ub.poison : index
  return %0 : index
}
