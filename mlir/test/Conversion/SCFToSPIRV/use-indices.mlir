// RUN: mlir-opt -convert-scf-to-spirv %s -o - | FileCheck %s

// CHECK-LABEL: @forward
func.func @forward() {
  // CHECK: %[[LB:.*]] = spirv.Constant 0 : i32
  %c0 = arith.constant 0 : index
  // CHECK: %[[UB:.*]] = spirv.Constant 32 : i32
  %c32 = arith.constant 32 : index
  // CHECK: %[[STEP:.*]] = spirv.Constant 1 : i32
  %c1 = arith.constant 1 : index

  // CHECK:      spirv.mlir.loop {
  // CHECK-NEXT:   spirv.Branch ^[[HEADER:.*]](%[[LB]] : i32)
  // CHECK:      ^[[HEADER]](%[[INDVAR:.*]]: i32):
  // CHECK:        %[[CMP:.*]] = spirv.SLessThan %[[INDVAR]], %[[UB]] : i32
  // CHECK:        spirv.BranchConditional %[[CMP]], ^[[BODY:.*]], ^[[MERGE:.*]]
  // CHECK:      ^[[BODY]]:
  // CHECK:        %[[X:.*]] = spirv.IAdd %[[INDVAR]], %[[INDVAR]] : i32
  // CHECK:        %[[INDNEXT:.*]] = spirv.IAdd %[[INDVAR]], %[[STEP]] : i32
  // CHECK:        spirv.Branch ^[[HEADER]](%[[INDNEXT]] : i32)
  // CHECK:      ^[[MERGE]]:
  // CHECK:        spirv.mlir.merge
  // CHECK:      }
  scf.for %arg2 = %c0 to %c32 step %c1 {
      %1 = index.add %arg2, %arg2
  }
  return
}
