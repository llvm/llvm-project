// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test checks that, when compiling for an offloading target, device shared
// memory will be used in place of allocas for certain private variables.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  omp.private {type = private} @privatizer : i32
  omp.declare_reduction @reduction : i32 init {
  ^bb0(%arg0: i32):
    %0 = llvm.mlir.constant(0 : i32) : i32
    omp.yield(%0 : i32)
  } combiner {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = llvm.add %arg0, %arg1 : i32
    omp.yield(%0 : i32)
  }
  llvm.func @main() {
    %c0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %c0 x i32 {bindc_name = "x"} : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    %3 = llvm.alloca %c0 x i32 {bindc_name = "y"} : (i64) -> !llvm.ptr<5>
    %4 = llvm.addrspacecast %3 : !llvm.ptr<5> to !llvm.ptr
    %5 = llvm.alloca %c0 x i32 {bindc_name = "z"} : (i64) -> !llvm.ptr<5>
    %6 = llvm.addrspacecast %5 : !llvm.ptr<5> to !llvm.ptr
    %7 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "x"}
    %8 = omp.map.info var_ptr(%4 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "y"}
    %9 = omp.map.info var_ptr(%6 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "z"}
    omp.target map_entries(%7 -> %arg0, %8 -> %arg1, %9 -> %arg2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      %11 = llvm.mlir.constant(10000 : i32) : i32
      %12 = llvm.mlir.constant(1 : i32) : i32
      omp.teams reduction(@reduction %arg0 -> %arg3 : !llvm.ptr) {
        omp.distribute private(@privatizer %arg1 -> %arg4, @privatizer %arg2 -> %arg5 : !llvm.ptr, !llvm.ptr) {
          omp.loop_nest (%arg6) : i32 = (%12) to (%11) inclusive step (%12) {
            llvm.store %arg6, %arg4 : i32, !llvm.ptr
            %13 = llvm.load %arg3 : !llvm.ptr -> i32
            %14 = llvm.add %13, %12 : i32
            llvm.store %14, %arg3 : i32, !llvm.ptr
            omp.parallel reduction(@reduction %arg5 -> %arg7 : !llvm.ptr) {
              %15 = llvm.load %arg4 : !llvm.ptr -> i32
              %16 = llvm.load %arg7 : !llvm.ptr -> i32
              %17 = llvm.add %15, %16 : i32
              llvm.store %17, %arg7 : i32, !llvm.ptr
              omp.terminator
            }
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    // CHECK: call i32 @__kmpc_target_init
    // CHECK: call void @[[OUTLINED_TARGET:__omp_offloading_[A-Za-z0-9_.]*]]

    // CHECK: define internal void @[[OUTLINED_TARGET]]
    // CHECK: %[[X_PRIV:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 4)
    // CHECK: %[[GEP_X:.*]] = getelementptr { {{.*}} }, ptr addrspace(5) %structArg
    // CHECK-NEXT: store ptr %[[X_PRIV]], ptr addrspace(5) %[[GEP_X]]
    // CHECK-NEXT: call void @[[OUTLINED_TEAMS:__omp_offloading_[A-Za-z0-9_.]*]](ptr %structArg.ascast)

    // CHECK: [[REDUCE_FINALIZE_BB:reduce\.finalize.*]]:
    // CHECK-NEXT: %{{.*}} = call i32 @__kmpc_global_thread_num
    // CHECK-NEXT: call void @__kmpc_barrier
    // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[X_PRIV]], i64 4)

    // CHECK: define internal void @[[OUTLINED_TEAMS]]
    // CHECK: %[[Y_PRIV:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 4)
    // CHECK: %[[Z_PRIV:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 4)

    // %[[GEP_Y:.*]] = getelementptr { {{.*}} }, ptr addrspace(5) %structArg
    // store ptr %[[Y_PRIV]], ptr addrspace(5) %[[GEP_Y]], align 8
    // %[[GEP_Z:.*]] = getelementptr { {{.*}} }, ptr addrspace(5) %structArg
    // store ptr %[[Z_PRIV]], ptr addrspace(5) %[[GEP_Z]], align 8

    // CHECK: call void @__kmpc_free_shared(ptr %[[Y_PRIV]], i64 4)
    // CHECK-NEXT: call void @__kmpc_free_shared(ptr %[[Z_PRIV]], i64 4)
    // CHECK-NEXT: br label %[[EXIT_BB:.*]]

    // CHECK: [[EXIT_BB]]:
    // CHECK-NEXT: ret void
    llvm.return
  }
}
