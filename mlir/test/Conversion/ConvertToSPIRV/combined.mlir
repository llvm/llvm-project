// RUN: mlir-opt -convert-to-spirv %s | FileCheck %s

// CHECK-LABEL: @combined
// CHECK: %[[C0_F32:.*]] = spirv.Constant 0.000000e+00 : f32
// CHECK: %[[C1_F32:.*]]  = spirv.Constant 1.000000e+00 : f32
// CHECK: %[[C0_I32:.*]] = spirv.Constant 0 : i32
// CHECK: %[[C4_I32:.*]] = spirv.Constant 4 : i32
// CHECK: %[[C0_I32_0:.*]] = spirv.Constant 0 : i32
// CHECK: %[[C4_I32_0:.*]] = spirv.Constant 4 : i32
// CHECK: %[[C1_I32:.*]] = spirv.Constant 1 : i32
// CHECK: %[[VEC:.*]] = spirv.Constant dense<1.000000e+00> : vector<4xf32>
// CHECK: %[[VARIABLE:.*]] = spirv.Variable : !spirv.ptr<f32, Function>
// CHECK: spirv.mlir.loop {
// CHECK:    spirv.Branch ^[[HEADER:.*]](%[[C0_I32_0]], %[[C0_F32]] : i32, f32)
// CHECK:  ^[[HEADER]](%[[INDVAR_0:.*]]: i32, %[[INDVAR_1:.*]]: f32):
// CHECK:    %[[SLESSTHAN:.*]] = spirv.SLessThan %[[INDVAR_0]], %[[C4_I32_0]] : i32
// CHECK:    spirv.BranchConditional %[[SLESSTHAN]], ^[[BODY:.*]], ^[[MERGE:.*]]
// CHECK:  ^[[BODY]]:
// CHECK:    %[[FADD:.*]] = spirv.FAdd %[[INDVAR_1]], %[[C1_F32]]  : f32
// CHECK:    %[[INSERT:.*]] = spirv.CompositeInsert %[[FADD]], %[[VEC]][0 : i32] : f32 into vector<4xf32>
// CHECK:    spirv.Store "Function" %[[VARIABLE]], %[[FADD]] : f32
// CHECK:    %[[IADD:.*]] = spirv.IAdd %[[INDVAR_0]], %[[C1_I32]] : i32
// CHECK:    spirv.Branch ^[[HEADER]](%[[IADD]], %[[FADD]] : i32, f32)
// CHECK:  ^[[MERGE]]:
// CHECK:    spirv.mlir.merge
// CHECK:  }
// CHECK:  %[[LOAD:.*]] = spirv.Load "Function" %[[VARIABLE]] : f32
// CHECK:  %[[UNDEF:.*]] = spirv.Undef : f32
// CHECK:  spirv.ReturnValue %[[UNDEF]] : f32
func.func @combined() -> f32 {
  %c0_f32 = arith.constant 0.0 : f32
  %c1_f32 = arith.constant 1.0 : f32
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %lb = index.casts %c0_i32 : i32 to index
  %ub = index.casts %c4_i32 : i32 to index
  %step = arith.constant 1 : index
  %buf = vector.broadcast %c1_f32 : f32 to vector<4xf32>
  scf.for %iv = %lb to %ub step %step iter_args(%sum_iter = %c0_f32) -> f32 {
    %t = vector.extract %buf[0] : f32 from vector<4xf32>
    %sum_next = arith.addf %sum_iter, %t : f32
    vector.insert %sum_next, %buf[0] : f32 into vector<4xf32>
    scf.yield %sum_next : f32
  }
  %ret = ub.poison : f32
  return %ret : f32
}
