// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// The aim of the test is to check the LLVM IR codegen for the device
// for omp teams construct

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true, omp.target = #omp.target<target_cpu = "gfx90a", target_features = "">} {
  llvm.func @foo(i32)
  llvm.func @omp_target_teams_shared_simple(%arg0 : i32)  {
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
// CHECK-NOT: call void {{.+}} @__kmpc_fork_teams(ptr @{{.+}}, i32 1, ptr [[OUTLINED_FN:.+]], ptr [[STRUCT_ARG:.*]])
// CHECK: ret void

//CHECK: define internal void @[[OUTLINED_FN]](
//CHECK: call void @foo(i32 %[[FOO_ARG:.*]])
//CHECK: ret void
