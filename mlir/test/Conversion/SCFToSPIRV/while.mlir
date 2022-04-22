// RUN: mlir-opt -allow-unregistered-dialect -convert-scf-to-spirv %s -o - | FileCheck %s

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader, Int64], [SPV_KHR_storage_buffer_storage_class]>, {}>
} {

// CHECK-LABEL: @while_loop1
func.func @while_loop1(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-SAME: (%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32)
  // CHECK: %[[INITVAR:.*]] = spv.Constant 2 : i32
  // CHECK: %[[VAR1:.*]] = spv.Variable : !spv.ptr<i32, Function>
  // CHECK: spv.mlir.loop {
  // CHECK:   spv.Branch ^[[HEADER:.*]](%[[ARG1]] : i32)
  // CHECK: ^[[HEADER]](%[[INDVAR1:.*]]: i32):
  // CHECK:   %[[CMP:.*]] = spv.SLessThan %[[INDVAR1]], %[[ARG2]] : i32
  // CHECK:   spv.Store "Function" %[[VAR1]], %[[INDVAR1]] : i32
  // CHECK:   spv.BranchConditional %[[CMP]], ^[[BODY:.*]](%[[INDVAR1]] : i32), ^[[MERGE:.*]]
  // CHECK: ^[[BODY]](%[[INDVAR2:.*]]: i32):
  // CHECK:   %[[UPDATED:.*]] = spv.IMul %[[INDVAR2]], %[[INITVAR]] : i32
  // CHECK: spv.Branch ^[[HEADER]](%[[UPDATED]] : i32)
  // CHECK: ^[[MERGE]]:
  // CHECK:   spv.mlir.merge
  // CHECK: }
  %c2_i32 = arith.constant 2 : i32
  %0 = scf.while (%arg3 = %arg0) : (i32) -> (i32) {
    %1 = arith.cmpi slt, %arg3, %arg1 : i32
    scf.condition(%1) %arg3 : i32
  } do {
  ^bb0(%arg5: i32):
    %1 = arith.muli %arg5, %c2_i32 : i32
    scf.yield %1 : i32
  }
  // CHECK: %[[OUT:.*]] = spv.Load "Function" %[[VAR1]] : i32
  // CHECK: spv.ReturnValue %[[OUT]] : i32
  return %0 : i32
}

// -----

// CHECK-LABEL: @while_loop2
func.func @while_loop2(%arg0: f32) -> i64 {
  // CHECK-SAME: (%[[ARG:.*]]: f32)
  // CHECK: %[[VAR:.*]] = spv.Variable : !spv.ptr<i64, Function>
  // CHECK: spv.mlir.loop {
  // CHECK:   spv.Branch ^[[HEADER:.*]](%[[ARG]] : f32)
  // CHECK: ^[[HEADER]](%[[INDVAR1:.*]]: f32):
  // CHECK:   %[[SHARED:.*]] = "foo.shared_compute"(%[[INDVAR1]]) : (f32) -> i64
  // CHECK:   %[[CMP:.*]] = "foo.evaluate_condition"(%[[INDVAR1]], %[[SHARED]]) : (f32, i64) -> i1
  // CHECK:   spv.Store "Function" %[[VAR]], %[[SHARED]] : i64
  // CHECK:   spv.BranchConditional %[[CMP]], ^[[BODY:.*]](%[[SHARED]] : i64), ^[[MERGE:.*]]
  // CHECK: ^[[BODY]](%[[INDVAR2:.*]]: i64):
  // CHECK:   %[[UPDATED:.*]] = "foo.payload"(%[[INDVAR2]]) : (i64) -> f32
  // CHECK: spv.Branch ^[[HEADER]](%[[UPDATED]] : f32)
  // CHECK: ^[[MERGE]]:
  // CHECK:   spv.mlir.merge
  // CHECK: }
  %res = scf.while (%arg1 = %arg0) : (f32) -> i64 {
    %shared = "foo.shared_compute"(%arg1) : (f32) -> i64
    %condition = "foo.evaluate_condition"(%arg1, %shared) : (f32, i64) -> i1
    scf.condition(%condition) %shared : i64
  } do {
  ^bb0(%arg2: i64):
    %res = "foo.payload"(%arg2) : (i64) -> f32
    scf.yield %res : f32
  }
  // CHECK: %[[OUT:.*]] = spv.Load "Function" %[[VAR]] : i64
  // CHECK: spv.ReturnValue %[[OUT]] : i64
  return %res : i64
}

} // end module
