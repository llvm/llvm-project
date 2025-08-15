// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @target_if_nowait(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %0 x i32 {bindc_name = "cond"} : (i64) -> !llvm.ptr
    %6 = llvm.load %3 : !llvm.ptr -> i32
    %7 = llvm.mlir.constant(0 : i64) : i32
    %8 = llvm.icmp "ne" %6, %7 : i32
    %9 = omp.map.info var_ptr(%3 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "cond"}
    %10 = omp.map.info var_ptr(%arg0 : !llvm.ptr, f32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "var"}
    %11 = omp.map.info var_ptr(%arg1 : !llvm.ptr, f32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "val"}
    omp.target if(%8) nowait map_entries(%10 -> %arg3, %11 -> %arg4 : !llvm.ptr, !llvm.ptr) {
      %12 = llvm.load %arg4 : !llvm.ptr -> f32
      llvm.store %12, %arg3 : f32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: define void @target_if_nowait{{.*}} {
// CHECK: omp_if.then:
// CHECK:   br label %[[TARGET_TASK_BB:.*]]

// CHECK: [[TARGET_TASK_BB]]:
// CHECK:   call ptr @__kmpc_omp_target_task_alloc
// CHECK:   br label %[[OFFLOAD_CONT:.*]]

// CHECK: [[OFFLOAD_CONT]]:
// CHECK:   br label %omp_if.end

// CHECK: omp_if.else:
// CHECK:   br label %[[HOST_TASK_BB:.*]]

// CHECK: [[HOST_TASK_BB]]:
// CHECK:   call ptr @__kmpc_omp_task_alloc
// CHECK:   br label %[[HOST_TASK_CONT:.*]]

// CHECK: [[HOST_TASK_CONT]]:
// CHECK:   br label %omp_if.end

// CHECK: omp_if.end:
// CHECK:   ret void
// CHECK: }

