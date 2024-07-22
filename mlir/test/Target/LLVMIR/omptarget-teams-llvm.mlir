// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// The aim of the test is to check the LLVM IR codegen for the device
// for omp teams construct

module attributes {omp.is_target_device = true} {
  llvm.func @foo(i32)
  llvm.func @omp_target_teams_shared_simple(%arg0 : i32)  attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
    omp.teams {
      llvm.call @foo(%arg0) : (i32) -> ()
      omp.terminator
    }
  llvm.return
  }
}

// CHECK-LABEL: @omp_target_teams_shared_simple
// CHECK-SAME: (i32 [[ARG0:%.+]])
// CHECK: call void @[[OUTLINED_FN:.*]](
// CHECK-NOT: call {{.+}} @__kmpc_fork_teams
// CHECK: ret void

//CHECK: define internal void @[[OUTLINED_FN]](
//CHECK: call void @foo(i32 %[[FOO_ARG:.*]])
//CHECK: ret void
