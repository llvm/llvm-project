// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Only check the overall shape of the code and the presence of relevant
// runtime calls. Actual IR checking is done at the OpenMPIRBuilder level.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true } {
  omp.declare_reduction @add_reduction_i32 : i32 init {
  ^bb0(%arg0: i32):
    %0 = llvm.mlir.constant(0 : i32) : i32
    omp.yield(%0 : i32)
  } combiner {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = llvm.add %arg0, %arg1 : i32
    omp.yield(%0 : i32)
  }
  llvm.func @simple_target_teams_only_reduction_() attributes {fir.internal_name = "_QPsimple_target_teams_only_reduction", frame_pointer = #llvm.framePointerKind<all>, omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, target_cpu = "gfx1030", target_features = #llvm.target_features<["+16-bit-insts", "+ci-insts", "+dl-insts", "+dot1-insts", "+dot10-insts", "+dot2-insts", "+dot5-insts", "+dot6-insts", "+dot7-insts", "+dpp", "+gfx10-3-insts", "+gfx10-insts", "+gfx8-insts", "+gfx9-insts", "+gws", "+image-insts", "+s-memrealtime", "+s-memtime-inst", "+wavefrontsize32"]>} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 {bindc_name = "sum"} : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x i32 {bindc_name = "index_"} : (i64) -> !llvm.ptr<5>
    %5 = llvm.addrspacecast %4 : !llvm.ptr<5> to !llvm.ptr
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.mlir.constant(1 : i64) : i64
    llvm.store %6, %2 : i32, !llvm.ptr
    %9 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "sum"}
    %10 = omp.map.info var_ptr(%5 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "index_"}
    omp.target map_entries(%9 -> %arg0, %10 -> %arg1 : !llvm.ptr, !llvm.ptr) {
      %11 = llvm.mlir.constant(0 : index) : i64
      %12 = llvm.mlir.constant(10000 : index) : i64
      %13 = llvm.mlir.constant(1 : index) : i64
      omp.teams reduction(@add_reduction_i32 %arg0 -> %arg2 : !llvm.ptr) {
        %14 = llvm.trunc %13 : i64 to i32
        llvm.br ^bb1(%14, %12 : i32, i64)
      ^bb1(%15: i32, %16: i64):  // 2 preds: ^bb0, ^bb2
        %17 = llvm.icmp "sgt" %16, %11 : i64
        llvm.cond_br %17, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        llvm.store %15, %arg1 : i32, !llvm.ptr
        %18 = llvm.load %arg2 : !llvm.ptr -> i32
        %19 = llvm.load %arg1 : !llvm.ptr -> i32
        %20 = llvm.add %18, %19 : i32
        llvm.store %20, %arg2 : i32, !llvm.ptr
        %21 = llvm.load %arg1 : !llvm.ptr -> i32
        %22 = llvm.add %21, %14 overflow<nsw> : i32
        %23 = llvm.sub %16, %13 : i64
        llvm.br ^bb1(%22, %23 : i32, i64)
      ^bb3:  // pred: ^bb1
        llvm.store %15, %arg1 : i32, !llvm.ptr
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: call i32 @__kmpc_target_init
// CHECK: call void @[[OUTLINED:__omp_offloading_[A-Za-z0-9_.]*]]
// CHECK: %[[MASTER:.+]] = call i32 @__kmpc_nvptx_teams_reduce_nowait_v2
// CHECK: icmp eq i32 %[[MASTER]], 1
// CHECK: i1 %{{.+}}, label %[[THEN:[A-Za-z0-9_.]*]], label %[[DONE:[A-Za-z0-9_.]*]]
// CHECK: [[THEN]]:
// CHECK-NEXT: %[[FINAL_RHS:[A-Za-z0-9_.]*]] = load i32
// CHECK-NEXT: %[[FINAL_LHS:[A-Za-z0-9_.]*]] = load i32
// CHECK-NEXT: %[[FINAL_RESULT:[A-Za-z0-9_.]*]] = add i32 %[[FINAL_LHS]], %[[FINAL_RHS]]
// CHECK-NEXT: store i32 %[[FINAL_RESULT]]

// CHECK: call void @__kmpc_barrier
// CHECK: call void @__kmpc_target_deinit

// CHECK: define internal void @[[OUTLINED]]
// Skip to the loop
// CHECK: br i1
// CHECK: %[[TEAM_RHS:[A-Za-z0-9_.]*]] = load i32
// CHECK-NEXT: %[[TEAM_LHS:[A-Za-z0-9_.]*]] = load i32
// CHECK-NEXT: %[[TEAM_RESULT:[A-Za-z0-9_.]*]] = add i32 %[[TEAM_RHS]], %[[TEAM_LHS]]
// CHECK-NEXT: store i32 %[[TEAM_RESULT]]
