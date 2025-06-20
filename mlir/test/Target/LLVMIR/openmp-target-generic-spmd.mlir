// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @host(%arg0 : !llvm.ptr) {
    %x = llvm.load %arg0 : !llvm.ptr -> i32
    %0 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(to) capture(ByCopy) -> !llvm.ptr
    omp.target host_eval(%x -> %lb, %x -> %ub, %x -> %step : i32, i32, i32) map_entries(%0 -> %ptr : !llvm.ptr) {
      %x.map = llvm.load %ptr : !llvm.ptr -> i32
      omp.teams {
        omp.distribute {
          omp.loop_nest (%iv1) : i32 = (%lb) to (%ub) step (%step) {
            omp.parallel {
              omp.wsloop {
                omp.loop_nest (%iv2) : i32 = (%x.map) to (%x.map) step (%x.map) {
                  omp.yield
                }
              }
              omp.terminator
            }
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define void @host
// CHECK:         %omp_loop.tripcount = {{.*}}
// CHECK-NEXT:    br label %[[ENTRY:.*]]
// CHECK:       [[ENTRY]]:
// CHECK:         %[[TRIPCOUNT:.*]] = zext i32 %omp_loop.tripcount to i64
// CHECK:         %[[TRIPCOUNT_KARG:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[KARGS:.*]], i32 0, i32 8
// CHECK-NEXT:    store i64 %[[TRIPCOUNT]], ptr %[[TRIPCOUNT_KARG]]
// CHECK:         %[[RESULT:.*]] = call i32 @__tgt_target_kernel({{.*}}, ptr %[[KARGS]])
// CHECK-NEXT:    %[[CMP:.*]] = icmp ne i32 %[[RESULT]], 0
// CHECK-NEXT:    br i1 %[[CMP]], label %[[OFFLOAD_FAILED:.*]], label %{{.*}}
// CHECK:       [[OFFLOAD_FAILED]]:
// CHECK:         call void @[[TARGET_OUTLINE:.*]]({{.*}})

// CHECK:       define internal void @[[TARGET_OUTLINE]]
// CHECK:         call void{{.*}}@__kmpc_fork_teams({{.*}}, ptr @[[TEAMS_OUTLINE:.*]], {{.*}})

// CHECK:       define internal void @[[TEAMS_OUTLINE]]
// CHECK:         call void @[[DISTRIBUTE_OUTLINE:.*]]({{.*}})

// CHECK:       define internal void @[[DISTRIBUTE_OUTLINE]]
// CHECK:         call void @__kmpc_for_static_init{{.*}}(ptr {{.*}}, i32 {{.*}}, i32 92, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, i32 {{.*}}, i32 {{.*}})
// CHECK:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call({{.*}}, ptr @[[PARALLEL_OUTLINE:.*]], {{.*}})

// CHECK:       define internal void @[[PARALLEL_OUTLINE]]
// CHECK:         call void @__kmpc_for_static_init{{.*}}(ptr {{.*}}, i32 {{.*}}, i32 34, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, i32 {{.*}}, i32 {{.*}})

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true, omp.is_gpu = true} {
  llvm.func @device(%arg0 : !llvm.ptr) {
    %0 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(to) capture(ByCopy) -> !llvm.ptr
    omp.target map_entries(%0 -> %ptr : !llvm.ptr) {
      %x = llvm.load %ptr : !llvm.ptr -> i32
      omp.teams {
        omp.distribute {
          omp.loop_nest (%iv1) : i32 = (%x) to (%x) step (%x) {
            omp.parallel {
              omp.wsloop {
                omp.loop_nest (%iv2) : i32 = (%x) to (%x) step (%x) {
                  omp.yield
                }
              }
              omp.terminator
            }
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK:      @[[KERNEL_NAME:.*]]_exec_mode = weak protected constant i8 [[EXEC_MODE:3]]
// CHECK:      @llvm.compiler.used = appending global [1 x ptr] [ptr @[[KERNEL_NAME]]_exec_mode], section "llvm.metadata"
// CHECK:      @[[KERNEL_NAME]]_kernel_environment = weak_odr protected constant %struct.KernelEnvironmentTy {
// CHECK-SAME: %struct.ConfigurationEnvironmentTy { i8 1, i8 1, i8 [[EXEC_MODE]], {{.*}}},
// CHECK-SAME: ptr @{{.*}}, ptr @{{.*}} }

// CHECK:      define weak_odr protected amdgpu_kernel void @[[KERNEL_NAME]]({{.*}})
// CHECK:        %{{.*}} = call i32 @__kmpc_target_init(ptr @[[KERNEL_NAME]]_kernel_environment, {{.*}})
// CHECK:        call void @[[TARGET_OUTLINE:.*]]({{.*}})
// CHECK:        call void @__kmpc_target_deinit()

// CHECK:      define internal void @[[TARGET_OUTLINE]]({{.*}})
// CHECK:        call void @[[TEAMS_OUTLINE:.*]]({{.*}})

// CHECK:      define internal void @[[TEAMS_OUTLINE]]({{.*}})
// CHECK:        call void @__kmpc_distribute_static_loop{{.*}}({{.*}}, ptr @[[DISTRIBUTE_OUTLINE:[^,]*]], {{.*}})

// CHECK:      define internal void @[[DISTRIBUTE_OUTLINE]]({{.*}})
// CHECK:        call void @__kmpc_parallel_51(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, ptr @[[PARALLEL_OUTLINE:.*]], ptr {{.*}}, ptr {{.*}}, i64 {{.*}})

// CHECK:      define internal void @[[PARALLEL_OUTLINE]]({{.*}})
// CHECK:        call void @__kmpc_for_static_loop{{.*}}({{.*}})

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true, omp.is_gpu = true} {
  llvm.func @device2(%arg0 : !llvm.ptr) {
    %0 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(to) capture(ByCopy) -> !llvm.ptr
    omp.target map_entries(%0 -> %ptr : !llvm.ptr) {
      %x = llvm.load %ptr : !llvm.ptr -> i32
      omp.teams {
        omp.distribute {
          omp.loop_nest (%iv1) : i32 = (%x) to (%x) step (%x) {
            omp.parallel {
              omp.terminator
            }
            llvm.br ^bb2
          ^bb1:
            omp.parallel {
              omp.terminator
            }
            omp.yield
          ^bb2:
            llvm.br ^bb1
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK:      @[[KERNEL_NAME:.*]]_exec_mode = weak protected constant i8 [[EXEC_MODE:3]]
// CHECK:      @llvm.compiler.used = appending global [1 x ptr] [ptr @[[KERNEL_NAME]]_exec_mode], section "llvm.metadata"
// CHECK:      @[[KERNEL_NAME]]_kernel_environment = weak_odr protected constant %struct.KernelEnvironmentTy {
// CHECK-SAME: %struct.ConfigurationEnvironmentTy { i8 1, i8 1, i8 [[EXEC_MODE]], {{.*}}},
// CHECK-SAME: ptr @{{.*}}, ptr @{{.*}} }

// CHECK:      define weak_odr protected amdgpu_kernel void @[[KERNEL_NAME]]({{.*}})
// CHECK:        %{{.*}} = call i32 @__kmpc_target_init(ptr @[[KERNEL_NAME]]_kernel_environment, {{.*}})
// CHECK:        call void @[[TARGET_OUTLINE:.*]]({{.*}})
// CHECK:        call void @__kmpc_target_deinit()

// CHECK:      define internal void @[[TARGET_OUTLINE]]({{.*}})
// CHECK:        call void @[[TEAMS_OUTLINE:.*]]({{.*}})

// CHECK:      define internal void @[[TEAMS_OUTLINE]]({{.*}})
// CHECK:        call void @__kmpc_distribute_static_loop{{.*}}({{.*}}, ptr @[[DISTRIBUTE_OUTLINE:[^,]*]], {{.*}})

// CHECK:      define internal void @[[DISTRIBUTE_OUTLINE]]({{.*}})
// CHECK:        call void @__kmpc_parallel_51(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, ptr @[[PARALLEL_OUTLINE0:.*]], ptr {{.*}}, ptr {{.*}}, i64 {{.*}})
// CHECK:        call void @__kmpc_parallel_51(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, ptr @[[PARALLEL_OUTLINE1:.*]], ptr {{.*}}, ptr {{.*}}, i64 {{.*}})

// CHECK:      define internal void @[[PARALLEL_OUTLINE1]]({{.*}})
// CHECK:      define internal void @[[PARALLEL_OUTLINE0]]({{.*}})
