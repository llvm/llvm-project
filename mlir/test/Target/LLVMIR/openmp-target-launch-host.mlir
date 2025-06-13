// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK: define void @main(i32 %[[NUM_TEAMS_ARG:.*]])
// CHECK: %[[KERNEL_ARGS:.*]] = alloca %struct.__tgt_kernel_arguments
// CHECK: %[[NUM_TEAMS:.*]] = insertvalue [3 x i32] zeroinitializer, i32 %[[NUM_TEAMS_ARG]], 0

// CHECK: %[[NUM_TEAMS_KARG:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[KERNEL_ARGS]], i32 0, i32 10
// CHECK: store [3 x i32] %[[NUM_TEAMS]], ptr %[[NUM_TEAMS_KARG]], align 4

// CHECK: %[[NUM_THREADS_ARG:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[KERNEL_ARGS]], i32 0, i32 11
// CHECK: store [3 x i32] [i32 10, i32 0, i32 0], ptr %[[NUM_THREADS_ARG]], align 4

// CHECK: %{{.*}} = call i32 @__tgt_target_kernel(ptr {{.*}}, i64 -1, i32 %[[NUM_TEAMS_ARG]], i32 [[NUM_THREADS:10]], ptr @.[[OUTLINED_FN:.*]].region_id, ptr %[[KERNEL_ARGS]])
// CHECK: call void @[[OUTLINED_FN]](i32 %[[NUM_TEAMS_ARG]])

// CHECK: define internal void @[[OUTLINED_FN]](i32 %[[NUM_TEAMS_OUTLINED:.*]])
// CHECK: call void @__kmpc_push_num_teams_51(ptr {{.*}}, i32 {{.*}}, i32 %[[NUM_TEAMS_OUTLINED]], i32 %[[NUM_TEAMS_OUTLINED]], i32 [[NUM_THREADS]])
module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @main(%num_teams : i32) {
    %target_threads = llvm.mlir.constant(20) : i32
    %teams_threads = llvm.mlir.constant(10) : i32
    omp.target thread_limit(%target_threads : i32)
               host_eval(%num_teams -> %arg_teams, %teams_threads -> %arg_teams_threads : i32, i32) {
      omp.teams num_teams(to %arg_teams : i32) thread_limit(%arg_teams_threads : i32) {
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}
