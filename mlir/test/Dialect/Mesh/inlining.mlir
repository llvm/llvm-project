// RUN: mlir-opt -inline %s | FileCheck %s

mesh.mesh @mesh0(shape = 4x?x2)

func.func private @mesh_to_inline() -> (index, index) {
  %0:2 = mesh.mesh_shape @mesh0 axes = [2, 1] : index, index
  return %0#0, %0#1 : index, index
}
// CHECK-LABEL: func.func @main
func.func @main() -> (index, index) {
  // CHECK-NEXT: %[[AXIS_SIZE:.*]]:2 = mesh.mesh_shape @mesh0 axes = [2, 1] : index
  %0:2 = func.call @mesh_to_inline() : () -> (index, index)
  // CHECK-NEXT: return %[[AXIS_SIZE]]#0, %[[AXIS_SIZE]]#1
  return %0#0, %0#1 : index, index
}
