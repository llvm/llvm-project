// RUN: split-file %s %t
// RUN: mlir-translate -mlir-to-llvmir %t/host.mlir | FileCheck %s --check-prefix=HOST
// RUN: mlir-translate -mlir-to-llvmir %t/device.mlir | FileCheck %s --check-prefix=DEVICE

//--- host.mlir

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @main(%x : i32) {
    omp.target host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
      omp.teams {
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

// HOST-LABEL: define void @main
// HOST:         %omp_loop.tripcount = {{.*}}
// HOST-NEXT:    br label %[[ENTRY:.*]]
// HOST:       [[ENTRY]]:
// HOST-NEXT:    %[[TRIPCOUNT:.*]] = zext i32 %omp_loop.tripcount to i64
// HOST:         %[[TRIPCOUNT_KARG:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[KARGS:.*]], i32 0, i32 8
// HOST-NEXT:    store i64 %[[TRIPCOUNT]], ptr %[[TRIPCOUNT_KARG]]
// HOST:         %[[RESULT:.*]] = call i32 @__tgt_target_kernel({{.*}}, ptr %[[KARGS]])
// HOST-NEXT:    %[[CMP:.*]] = icmp ne i32 %[[RESULT]], 0
// HOST-NEXT:    br i1 %[[CMP]], label %[[OFFLOAD_FAILED:.*]], label %{{.*}}
// HOST:       [[OFFLOAD_FAILED]]:
// HOST:         call void @[[TARGET_OUTLINE:.*]]({{.*}})

// HOST:       define internal void @[[TARGET_OUTLINE]]
// HOST:         call void{{.*}}@__kmpc_fork_teams({{.*}}, ptr @[[TEAMS_OUTLINE:.*]], {{.*}})

// HOST:       define internal void @[[TEAMS_OUTLINE]]
// HOST:         call void{{.*}}@__kmpc_fork_call({{.*}}, ptr @[[PARALLEL_OUTLINE:.*]], {{.*}})

// HOST:       define internal void @[[PARALLEL_OUTLINE]]
// HOST:         call void @[[DISTRIBUTE_OUTLINE:.*]]({{.*}})

// HOST:       define internal void @[[DISTRIBUTE_OUTLINE]]
// HOST:         call void @__kmpc_dist_for_static_init{{.*}}(ptr {{.*}}, i32 {{.*}}, i32 34, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, i32 {{.*}}, i32 {{.*}})

//--- device.mlir

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true, omp.is_gpu = true} {
  llvm.func @main(%x : i32) {
    omp.target host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) {
      omp.teams {
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

// DEVICE:      @[[KERNEL_NAME:.*]]_exec_mode = weak protected constant i8 2
// DEVICE:      @llvm.compiler.used = appending global [1 x ptr] [ptr @[[KERNEL_NAME]]_exec_mode], section "llvm.metadata"
// DEVICE:      @[[KERNEL_NAME]]_kernel_environment = weak_odr protected constant %struct.KernelEnvironmentTy {
// DEVICE-SAME: %struct.ConfigurationEnvironmentTy { i8 0, i8 1, i8 [[EXEC_MODE:2]], {{.*}}},
// DEVICE-SAME: ptr @{{.*}}, ptr @{{.*}} }

// DEVICE:      define weak_odr protected amdgpu_kernel void @[[KERNEL_NAME]]({{.*}})
// DEVICE:        %{{.*}} = call i32 @__kmpc_target_init(ptr @[[KERNEL_NAME]]_kernel_environment, {{.*}})
// DEVICE:        call void @[[TARGET_OUTLINE:.*]]({{.*}})
// DEVICE:        call void @__kmpc_target_deinit()

// DEVICE:      define internal void @[[TARGET_OUTLINE]]({{.*}})
// DEVICE:        call void @__kmpc_parallel_51(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, ptr @[[PARALLEL_OUTLINE:.*]], ptr {{.*}}, ptr {{.*}}, i64 {{.*}})

// DEVICE:      define internal void @[[PARALLEL_OUTLINE]]({{.*}})
// DEVICE:        call void @[[DISTRIBUTE_OUTLINE:.*]]({{.*}})

// DEVICE:      define internal void @[[DISTRIBUTE_OUTLINE]]({{.*}})
// DEVICE:        call void @__kmpc_distribute_for_static_loop{{.*}}({{.*}})
