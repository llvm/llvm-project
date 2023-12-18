// RUN: mlir-opt -allow-unregistered-dialect -convert-scf-to-spirv %s -o - | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader, Int64], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: @while_loop1
func.func @while_loop1(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK-SAME: (%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32)
  // CHECK: %[[INITVAR:.*]] = spirv.Constant 2 : i32
  // CHECK: %[[VAR1:.*]] = spirv.Variable : !spirv.ptr<i32, Function>
  // CHECK: spirv.mlir.loop {
  // CHECK:   spirv.Branch ^[[HEADER:.*]](%[[ARG1]] : i32)
  // CHECK: ^[[HEADER]](%[[INDVAR1:.*]]: i32):
  // CHECK:   %[[CMP:.*]] = spirv.SLessThan %[[INDVAR1]], %[[ARG2]] : i32
  // CHECK:   spirv.Store "Function" %[[VAR1]], %[[INDVAR1]] : i32
  // CHECK:   spirv.BranchConditional %[[CMP]], ^[[BODY:.*]](%[[INDVAR1]] : i32), ^[[MERGE:.*]]
  // CHECK: ^[[BODY]](%[[INDVAR2:.*]]: i32):
  // CHECK:   %[[UPDATED:.*]] = spirv.IMul %[[INDVAR2]], %[[INITVAR]] : i32
  // CHECK: spirv.Branch ^[[HEADER]](%[[UPDATED]] : i32)
  // CHECK: ^[[MERGE]]:
  // CHECK:   spirv.mlir.merge
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
  // CHECK: %[[OUT:.*]] = spirv.Load "Function" %[[VAR1]] : i32
  // CHECK: spirv.ReturnValue %[[OUT]] : i32
  return %0 : i32
}

// -----

// CHECK-LABEL: @while_loop2
func.func @while_loop2(%arg0: f32) -> i64 {
  // CHECK-SAME: (%[[ARG:.*]]: f32)
  // CHECK: %[[VAR:.*]] = spirv.Variable : !spirv.ptr<i64, Function>
  // CHECK: spirv.mlir.loop {
  // CHECK:   spirv.Branch ^[[HEADER:.*]](%[[ARG]] : f32)
  // CHECK: ^[[HEADER]](%[[INDVAR1:.*]]: f32):
  // CHECK:   %[[SHARED:.*]] = "foo.shared_compute"(%[[INDVAR1]]) : (f32) -> i64
  // CHECK:   %[[CMP:.*]] = "foo.evaluate_condition"(%[[INDVAR1]], %[[SHARED]]) : (f32, i64) -> i1
  // CHECK:   spirv.Store "Function" %[[VAR]], %[[SHARED]] : i64
  // CHECK:   spirv.BranchConditional %[[CMP]], ^[[BODY:.*]](%[[SHARED]] : i64), ^[[MERGE:.*]]
  // CHECK: ^[[BODY]](%[[INDVAR2:.*]]: i64):
  // CHECK:   %[[UPDATED:.*]] = "foo.payload"(%[[INDVAR2]]) : (i64) -> f32
  // CHECK: spirv.Branch ^[[HEADER]](%[[UPDATED]] : f32)
  // CHECK: ^[[MERGE]]:
  // CHECK:   spirv.mlir.merge
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
  // CHECK: %[[OUT:.*]] = spirv.Load "Function" %[[VAR]] : i64
  // CHECK: spirv.ReturnValue %[[OUT]] : i64
  return %res : i64
}

// -----

// CHECK-LABEL: @while_loop_before_typeconv
func.func @while_loop_before_typeconv(%arg0: index) -> i64 {
  // CHECK-SAME: (%[[ARG:.*]]: i32)
  // CHECK: %[[VAR:.*]] = spirv.Variable : !spirv.ptr<i64, Function>
  // CHECK: spirv.mlir.loop {
  // CHECK:   spirv.Branch ^[[HEADER:.*]](%[[ARG]] : i32)
  // CHECK: ^[[HEADER]](%[[INDVAR1:.*]]: i32):
  // CHECK:   spirv.BranchConditional %{{.*}}, ^[[BODY:.*]](%{{.*}} : i64), ^[[MERGE:.*]]
  // CHECK: ^[[BODY]](%[[INDVAR2:.*]]: i64):
  // CHECK: spirv.Branch ^[[HEADER]](%{{.*}} : i32)
  // CHECK: ^[[MERGE]]:
  // CHECK:   spirv.mlir.merge
  // CHECK: }
  %res = scf.while (%arg1 = %arg0) : (index) -> i64 {
    %shared = "foo.shared_compute"(%arg1) : (index) -> i64
    %condition = "foo.evaluate_condition"(%arg1, %shared) : (index, i64) -> i1
    scf.condition(%condition) %shared : i64
  } do {
  ^bb0(%arg2: i64):
    %res = "foo.payload"(%arg2) : (i64) -> index
    scf.yield %res : index
  }
  // CHECK: %[[OUT:.*]] = spirv.Load "Function" %[[VAR]] : i64
  // CHECK: spirv.ReturnValue %[[OUT]] : i64
  return %res : i64
}

// -----

// CHECK-LABEL: @while_loop_after_typeconv
func.func @while_loop_after_typeconv(%arg0: f32) -> index {
  // CHECK-SAME: (%[[ARG:.*]]: f32)
  // CHECK: %[[VAR:.*]] = spirv.Variable : !spirv.ptr<i32, Function>
  // CHECK: spirv.mlir.loop {
  // CHECK:   spirv.Branch ^[[HEADER:.*]](%[[ARG]] : f32)
  // CHECK: ^[[HEADER]](%[[INDVAR1:.*]]: f32):
  // CHECK:   spirv.BranchConditional %{{.*}}, ^[[BODY:.*]](%{{.*}} : i32), ^[[MERGE:.*]]
  // CHECK: ^[[BODY]](%[[INDVAR2:.*]]: i32):
  // CHECK: spirv.Branch ^[[HEADER]](%{{.*}} : f32)
  // CHECK: ^[[MERGE]]:
  // CHECK:   spirv.mlir.merge
  // CHECK: }
  %res = scf.while (%arg1 = %arg0) : (f32) -> index {
    %shared = "foo.shared_compute"(%arg1) : (f32) -> index
    %condition = "foo.evaluate_condition"(%arg1, %shared) : (f32, index) -> i1
    scf.condition(%condition) %shared : index
  } do {
  ^bb0(%arg2: index):
    %res = "foo.payload"(%arg2) : (index) -> f32
    scf.yield %res : f32
  }
  // CHECK: %[[OUT:.*]] = spirv.Load "Function" %[[VAR]] : i32
  // CHECK: spirv.ReturnValue %[[OUT]] : i32
  return %res : index
}

} // end module
