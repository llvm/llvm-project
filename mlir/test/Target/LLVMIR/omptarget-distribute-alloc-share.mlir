// RUN: mlir-opt %s -convert-openmp-to-llvm | mlir-translate -mlir-to-llvmir | FileCheck %s

// This file contains MLIR test cases for __kmpc_alloc_shared and __kmpc_free_shared

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  llvm.func @dist_collapse() {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %a = llvm.alloca %c1 x i32 {bindc_name = "y"} : (i64) -> !llvm.ptr<5>
    %y = llvm.addrspacecast %a : !llvm.ptr<5> to !llvm.ptr
    %m = omp.map.info var_ptr(%y : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("y") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%m -> %arg0 : !llvm.ptr) {
      %n = llvm.mlir.constant(100 : i32) : i32
      %one = llvm.mlir.constant(1 : i32) : i32
      omp.teams {
        omp.distribute {
          omp.loop_nest (%i, %j) : i32 = (%one, %one) to (%n, %n) inclusive step (%one, %one) collapse(2) {
            %v = llvm.add %i, %j : i32
            omp.parallel {
              llvm.store %v, %arg0 : i32, !llvm.ptr
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

  llvm.func @dist_simple() {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %a = llvm.alloca %c1 x i32 {bindc_name = "y"} : (i64) -> !llvm.ptr<5>
    %y = llvm.addrspacecast %a : !llvm.ptr<5> to !llvm.ptr
    %m = omp.map.info var_ptr(%y : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("y") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%m -> %arg0 : !llvm.ptr) {
      %n = llvm.mlir.constant(100 : i32) : i32
      %one = llvm.mlir.constant(1 : i32) : i32
      omp.teams {
        omp.distribute {
          omp.loop_nest (%i) : i32 = (%one) to (%n) inclusive step (%one) {
            %v = llvm.add %i, %i : i32
            omp.parallel {
              llvm.store %v, %arg0 : i32, !llvm.ptr
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

  llvm.func @dist_double() {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %a = llvm.alloca %c1 x i32 {bindc_name = "y"} : (i64) -> !llvm.ptr<5>
    %y = llvm.addrspacecast %a : !llvm.ptr<5> to !llvm.ptr
    %m = omp.map.info var_ptr(%y : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("y") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%m -> %arg0 : !llvm.ptr) {
      %n = llvm.mlir.constant(100 : i32) : i32
      %one = llvm.mlir.constant(1 : i32) : i32
      omp.teams {
        omp.distribute {
          omp.loop_nest (%i) : i32 = (%one) to (%n) inclusive step (%one) {
            %v = llvm.add %i, %i : i32
            omp.parallel {
              llvm.store %v, %arg0 : i32, !llvm.ptr
              omp.terminator
            }
            omp.yield
          }
        }
        omp.distribute {
          omp.loop_nest (%i) : i32 = (%one) to (%n) inclusive step (%one) {
            %v = llvm.add %i, %i : i32
            omp.parallel {
              llvm.store %v, %arg0 : i32, !llvm.ptr
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

  llvm.func @dist_if() {
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %a = llvm.alloca %c1 x i32 {bindc_name = "y"} : (i64) -> !llvm.ptr<5>
    %y = llvm.addrspacecast %a : !llvm.ptr<5> to !llvm.ptr
    %m = omp.map.info var_ptr(%y : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("y") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%m -> %arg0 : !llvm.ptr) {
      %n = llvm.mlir.constant(100 : i32) : i32
      %one = llvm.mlir.constant(1 : i32) : i32
      %zero = llvm.mlir.constant(0 : i32) : i32
      omp.teams {
        %ld = llvm.load %arg0 : !llvm.ptr -> i32
        %cond = llvm.icmp "sgt" %ld, %zero : i32
        llvm.cond_br %cond, ^then, ^endif
      ^then:
        omp.distribute {
          omp.loop_nest (%i) : i32 = (%one) to (%n) inclusive step (%one) {
            %v = llvm.add %i, %i : i32
            omp.parallel {
              llvm.store %v, %arg0 : i32, !llvm.ptr
              omp.terminator
            }
            omp.yield
          }
        }
        llvm.br ^endif
      ^endif:
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_dist_collapse_l7..omp_par.4
// CHECK:         %[[STRUCT_ARG:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 16)
// CHECK:         %[[RELOADED:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 4)
// CHECK:         br label %[[LOOP_PREHEADER:.*]]
// CHECK:         [[LOOP_PREHEADER]]:
// CHECK:         br label %[[LOOP_BODY:.*]]
// CHECK:         [[LOOP_BODY]]:
// CHECK:         br label %[[LOOP_NEST:.*]]
// CHECK:         [[LOOP_NEST]]:
// CHECK:         br label %[[LOOP_PARALLEL:.*]]
// CHECK:         [[LOOP_PARALLEL]]:
// CHECK:         call void @__kmpc_parallel_60(
// CHECK:         br label %[[LOOP_EXIT:.*]]
// CHECK:         [[LOOP_EXIT]]:
// CHECK-NEXT:    call void @__kmpc_free_shared(ptr %[[STRUCT_ARG]], i64 16)
// CHECK:         br label %[[LOOP_EXIT:.*]]
// CHECK:         [[LOOP_EXIT]]:
// CHECK:         br label %[[LOOP_AFTER:.*]]
// CHECK:         [[LOOP_AFTER]]:
// CHECK:         br label %[[DEALLOC:.*]]
// CHECK:         [[DEALLOC]]:
// CHECK-NEXT:    call void @__kmpc_free_shared(ptr %[[RELOADED]], i64 4)

// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_dist_simple_l32..omp_par.7
// CHECK:         %[[STRUCT_ARG1:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 16)
// CHECK:         %[[RELOADED1:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 4)
// CHECK:         br label %[[LOOP_NEST1:.*]]
// CHECK:         [[LOOP_NEST1]]:
// CHECK:         br label %[[LOOP_PARALLEL1:.*]]
// CHECK:         [[LOOP_PARALLEL1]]:
// CHECK:         call void @__kmpc_parallel_60(
// CHECK:         br label %[[LOOP_EXIT1:.*]]
// CHECK:         [[LOOP_EXIT1]]:
// CHECK-NEXT:    call void @__kmpc_free_shared(ptr %[[STRUCT_ARG1]], i64 16)
// CHECK:         br label %[[REGION_CONT1:.*]]
// CHECK:         [[REGION_CONT1]]:
// CHECK:         br label %[[DEALLOC1:.*]]
// CHECK:         [[DEALLOC1]]:
// CHECK-NEXT:    call void @__kmpc_free_shared(ptr %[[RELOADED1]], i64 4)

// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_dist_double_l57..omp_par.13
// CHECK:         %[[STRUCT_ARG2:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 16)
// CHECK:         %[[RELOADED2:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 4)
// CHECK:         br label %[[LOOP_NEST2:.*]]
// CHECK:         [[LOOP_NEST2]]:
// CHECK:         br label %[[LOOP_PARALLEL2:.*]]
// CHECK:         [[LOOP_PARALLEL2]]:
// CHECK:         call void @__kmpc_parallel_60(
// CHECK:         br label %[[LOOP_EXIT2:.*]]
// CHECK:         [[LOOP_EXIT2]]:
// CHECK-NEXT:    call void @__kmpc_free_shared(ptr %[[STRUCT_ARG2]], i64 16)
// CHECK:         br label %[[REGION_CONT2:.*]]
// CHECK:         [[REGION_CONT2]]:
// CHECK:         br label %[[DEALLOC2:.*]]
// CHECK:         [[DEALLOC2]]:
// CHECK-NEXT:    call void @__kmpc_free_shared(ptr %[[RELOADED2]], i64 4)

// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_dist_double_l57..omp_par.10
// CHECK:         %[[STRUCT_ARG3:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 16)
// CHECK:         %[[RELOADED3:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 4)
// CHECK:         br label %[[LOOP_NEST3:.*]]
// CHECK:         [[LOOP_NEST3]]:
// CHECK:         br label %[[LOOP_PARALLEL3:.*]]
// CHECK:         [[LOOP_PARALLEL3]]:
// CHECK:         call void @__kmpc_parallel_60(
// CHECK:         br label %[[LOOP_EXIT3:.*]]
// CHECK:         [[LOOP_EXIT3]]:
// CHECK-NEXT:    call void @__kmpc_free_shared(ptr %[[STRUCT_ARG3]], i64 16)
// CHECK:         br label %[[REGION_CONT3:.*]]
// CHECK:         [[REGION_CONT3]]:
// CHECK:         br label %[[DEALLOC3:.*]]
// CHECK:         [[DEALLOC3]]:
// CHECK-NEXT:    call void @__kmpc_free_shared(ptr %[[RELOADED3]], i64 4)

// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_dist_if_l92..omp_par.16
// CHECK:         %[[STRUCT_ARG4:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 16)
// CHECK:         %[[RELOADED4:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 4)
// CHECK:         br label %[[LOOP_NEST4:.*]]
// CHECK:         [[LOOP_NEST4]]:
// CHECK:         br label %[[LOOP_PARALLEL4:.*]]
// CHECK:         [[LOOP_PARALLEL4]]:
// CHECK:         call void @__kmpc_parallel_60(
// CHECK:         br label %[[LOOP_EXIT4:.*]]
// CHECK:         [[LOOP_EXIT4]]:
// CHECK-NEXT:    call void @__kmpc_free_shared(ptr %[[STRUCT_ARG4]], i64 16)
// CHECK:         br label %[[REGION_CONT4:.*]]
// CHECK:         [[REGION_CONT4]]:
// CHECK:         br label %[[DEALLOC4:.*]]
// CHECK:         [[DEALLOC4]]:
// CHECK-NEXT:    call void @__kmpc_free_shared(ptr %[[RELOADED4]], i64 4)

