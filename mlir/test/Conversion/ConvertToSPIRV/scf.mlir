// RUN: mlir-opt -convert-to-spirv="run-signature-conversion=false run-vector-unrolling=false" -split-input-file %s | FileCheck %s

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
// CHECK-SAME: (%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32)
// CHECK:       %[[INITVAR:.*]] = spirv.Constant 2 : i32
// CHECK:       %[[VAR1:.*]] = spirv.Variable : !spirv.ptr<i32, Function>
// CHECK:       spirv.mlir.loop {
// CHECK:         spirv.Branch ^[[HEADER:.*]](%[[ARG1]] : i32)
// CHECK:       ^[[HEADER]](%[[INDVAR1:.*]]: i32):
// CHECK:         %[[CMP:.*]] = spirv.SLessThan %[[INDVAR1]], %[[ARG2]] : i32
// CHECK:         spirv.Store "Function" %[[VAR1]], %[[INDVAR1]] : i32
// CHECK:         spirv.BranchConditional %[[CMP]], ^[[BODY:.*]](%[[INDVAR1]] : i32), ^[[MERGE:.*]]
// CHECK:       ^[[BODY]](%[[INDVAR2:.*]]: i32):
// CHECK:         %[[UPDATED:.*]] = spirv.IMul %[[INDVAR2]], %[[INITVAR]] : i32
// CHECK:       spirv.Branch ^[[HEADER]](%[[UPDATED]] : i32)
// CHECK:       ^[[MERGE]]:
// CHECK:         spirv.mlir.merge
// CHECK:       }
// CHECK:       %[[OUT:.*]] = spirv.Load "Function" %[[VAR1]] : i32
// CHECK:       spirv.ReturnValue %[[OUT]] : i32
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

// CHECK-LABEL: @for
// CHECK:       %[[LB:.*]] = spirv.Constant 4 : i32
// CHECK:       %[[UB:.*]] = spirv.Constant 42 : i32
// CHECK:       %[[STEP:.*]] = spirv.Constant 2 : i32
// CHECK:       %[[INITVAR1:.*]] = spirv.Constant 0.000000e+00 : f32
// CHECK:       %[[INITVAR2:.*]] = spirv.Constant 1.000000e+00 : f32
// CHECK:       %[[VAR1:.*]] = spirv.Variable : !spirv.ptr<f32, Function>
// CHECK:       %[[VAR2:.*]] = spirv.Variable : !spirv.ptr<f32, Function>
// CHECK:       spirv.mlir.loop {
// CHECK:         spirv.Branch ^[[HEADER:.*]](%[[LB]], %[[INITVAR1]], %[[INITVAR2]] : i32, f32, f32)
// CHECK:       ^[[HEADER]](%[[INDVAR:.*]]: i32, %[[CARRIED1:.*]]: f32, %[[CARRIED2:.*]]: f32):
// CHECK:         %[[CMP:.*]] = spirv.SLessThan %[[INDVAR]], %[[UB]] : i32
// CHECK:         spirv.BranchConditional %[[CMP]], ^[[BODY:.*]], ^[[MERGE:.*]]
// CHECK:       ^[[BODY]]:
// CHECK:         %[[UPDATED:.*]] = spirv.FAdd %[[CARRIED1]], %[[CARRIED1]] : f32
// CHECK-DAG:     %[[INCREMENT:.*]] = spirv.IAdd %[[INDVAR]], %[[STEP]] : i32
// CHECK-DAG:     spirv.Store "Function" %[[VAR1]], %[[UPDATED]] : f32
// CHECK-DAG:     spirv.Store "Function" %[[VAR2]], %[[UPDATED]] : f32
// CHECK:       spirv.Branch ^[[HEADER]](%[[INCREMENT]], %[[UPDATED]], %[[UPDATED]] : i32, f32, f32)
// CHECK:       ^[[MERGE]]:
// CHECK:         spirv.mlir.merge
// CHECK:       }
// CHECK-DAG:  %[[OUT1:.*]] = spirv.Load "Function" %[[VAR1]] : f32
// CHECK-DAG:  %[[OUT2:.*]] = spirv.Load "Function" %[[VAR2]] : f32
func.func @for() {
  %lb = arith.constant 4 : index
  %ub = arith.constant 42 : index
  %step = arith.constant 2 : index
  %s0 = arith.constant 0.0 : f32
  %s1 = arith.constant 1.0 : f32
  %result:2 = scf.for %i0 = %lb to %ub step %step iter_args(%si = %s0, %sj = %s1) -> (f32, f32) {
    %sn = arith.addf %si, %si : f32
    scf.yield %sn, %sn: f32, f32
  }
  return
}
