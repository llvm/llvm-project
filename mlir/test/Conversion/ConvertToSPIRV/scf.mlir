// RUN: mlir-opt -convert-to-spirv %s | FileCheck %s

// CHECK-LABEL: @if_yield
// CHECK: %[[VAR:.*]] = spirv.Variable : !spirv.ptr<f32, Function>
// CHECK:       spirv.mlir.selection {
// CHECK-NEXT:    spirv.BranchConditional {{%.*}}, [[TRUE:\^.*]], [[FALSE:\^.*]]
// CHECK-NEXT:  [[TRUE]]:
// CHECK:         %[[C0TRUE:.*]] = spirv.Constant 0.000000e+00 : f32
// CHECK:         %[[RETTRUE:.*]] = spirv.Constant 0.000000e+00 : f32
// CHECK-DAG:     spirv.Store "Function" %[[VAR]], %[[RETTRUE]] : f32
// CHECK:         spirv.Branch ^[[MERGE:.*]]
// CHECK-NEXT:  [[FALSE]]:
// CHECK:         %[[C0FALSE:.*]] = spirv.Constant 1.000000e+00 : f32
// CHECK:         %[[RETFALSE:.*]] = spirv.Constant 2.71828175 : f32
// CHECK-DAG:     spirv.Store "Function" %[[VAR]], %[[RETFALSE]] : f32
// CHECK:         spirv.Branch ^[[MERGE]]
// CHECK-NEXT:  ^[[MERGE]]:
// CHECK:         spirv.mlir.merge
// CHECK-NEXT:  }
// CHECK-DAG:   %[[OUT:.*]] = spirv.Load "Function" %[[VAR]] : f32
// CHECK:       spirv.ReturnValue %[[OUT]] : f32
func.func @if_yield(%arg0: i1) -> f32 {
  %0 = scf.if %arg0 -> f32 {
    %c0 = arith.constant 0.0 : f32
    %res = math.sqrt %c0 : f32
    scf.yield %res : f32
  } else {
    %c0 = arith.constant 1.0 : f32
    %res = math.exp %c0 : f32
    scf.yield %res : f32
  }
  return %0 : f32
}

// CHECK-LABEL: @while
func.func @while(%arg0: i32, %arg1: i32) -> i32 {
  %c2_i32 = arith.constant 2 : i32
  %0 = scf.while (%arg3 = %arg0) : (i32) -> (i32) {
    %1 = arith.cmpi slt, %arg3, %arg1 : i32
    scf.condition(%1) %arg3 : i32
  } do {
  ^bb0(%arg5: i32):
    %1 = arith.muli %arg5, %c2_i32 : i32
    scf.yield %1 : i32
  }
  return %0 : i32
}
