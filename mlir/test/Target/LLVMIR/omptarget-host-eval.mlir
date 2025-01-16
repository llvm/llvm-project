// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @omp_target_region_() {
    %out_teams = llvm.mlir.constant(1000 : i32) : i32
    %out_threads = llvm.mlir.constant(2000 : i32) : i32
    %out_lb = llvm.mlir.constant(0 : i32) : i32
    %out_ub = llvm.mlir.constant(3000 : i32) : i32
    %out_step = llvm.mlir.constant(1 : i32) : i32

    omp.target
      host_eval(%out_teams -> %teams, %out_threads -> %threads,
                %out_lb -> %lb, %out_ub -> %ub, %out_step -> %step :
                i32, i32, i32, i32, i32) {
      omp.teams num_teams(to %teams : i32) thread_limit(%threads : i32) {
        omp.parallel {
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define void @omp_target_region_
// CHECK: %[[ARGS:.*]] = alloca %struct.__tgt_kernel_arguments

// CHECK: %[[TRIPCOUNT_ADDR:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[ARGS]], i32 0, i32 8
// CHECK: store i64 3000, ptr %[[TRIPCOUNT_ADDR]]

// CHECK: %[[TEAMS_ADDR:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[ARGS]], i32 0, i32 10
// CHECK: store [3 x i32] [i32 1000, i32 0, i32 0], ptr %[[TEAMS_ADDR]]

// CHECK: %[[THREADS_ADDR:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[ARGS]], i32 0, i32 11
// CHECK: store [3 x i32] [i32 2000, i32 0, i32 0], ptr %[[THREADS_ADDR]]

// CHECK: call i32 @__tgt_target_kernel(ptr @{{.*}}, i64 {{.*}}, i32 1000, i32 2000, ptr @{{.*}}, ptr %[[ARGS]])
