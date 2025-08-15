// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Only check the overall shape of the code and the presence of relevant
// runtime calls. Actual IR checking is done at the OpenMPIRBuilder level.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true } {
  omp.private {type = private} @_QFEj_private_i32 : i32
  omp.declare_reduction @add_reduction_f32 : f32 init {
  ^bb0(%arg0: f32):
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    omp.yield(%0 : f32)
  } combiner {
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = llvm.fadd %arg0, %arg1 {fastmathFlags = #llvm.fastmath<contract>} : f32
    omp.yield(%0 : f32)
  }
  omp.declare_reduction @add_reduction_f64 : f64 init {
  ^bb0(%arg0: f64):
    %0 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    omp.yield(%0 : f64)
  } combiner {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = llvm.fadd %arg0, %arg1 {fastmathFlags = #llvm.fastmath<contract>} : f64
    omp.yield(%0 : f64)
  }
  llvm.func @_QQmain() attributes {fir.bindc_name = "reduction", frame_pointer = #llvm.framePointerKind<all>, omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, target_cpu = "gfx1030", target_features = #llvm.target_features<["+16-bit-insts", "+ci-insts", "+dl-insts", "+dot1-insts", "+dot10-insts", "+dot2-insts", "+dot5-insts", "+dot6-insts", "+dot7-insts", "+dpp", "+gfx10-3-insts", "+gfx10-insts", "+gfx8-insts", "+gfx9-insts", "+gws", "+image-insts", "+s-memrealtime", "+s-memtime-inst", "+vmem-to-lds-load-insts", "+wavefrontsize32"]>} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 {bindc_name = "k"} : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %3 x i32 {bindc_name = "j"} : (i64) -> !llvm.ptr<5>
    %5 = llvm.addrspacecast %4 : !llvm.ptr<5> to !llvm.ptr
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.alloca %6 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr<5>
    %8 = llvm.addrspacecast %7 : !llvm.ptr<5> to !llvm.ptr
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.alloca %9 x f32 {bindc_name = "ce4"} : (i64) -> !llvm.ptr<5>
    %11 = llvm.addrspacecast %10 : !llvm.ptr<5> to !llvm.ptr
    %12 = llvm.mlir.constant(1 : i64) : i64
    %13 = llvm.alloca %12 x f32 {bindc_name = "ce3"} : (i64) -> !llvm.ptr<5>
    %14 = llvm.addrspacecast %13 : !llvm.ptr<5> to !llvm.ptr
    %15 = llvm.mlir.constant(1 : i64) : i64
    %16 = llvm.alloca %15 x f64 {bindc_name = "ce2"} : (i64) -> !llvm.ptr<5>
    %17 = llvm.addrspacecast %16 : !llvm.ptr<5> to !llvm.ptr
    %18 = llvm.mlir.constant(1 : i64) : i64
    %19 = llvm.alloca %18 x f64 {bindc_name = "ce1"} : (i64) -> !llvm.ptr<5>
    %20 = llvm.addrspacecast %19 : !llvm.ptr<5> to !llvm.ptr
    %21 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %22 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    %23 = llvm.mlir.constant(1 : i64) : i64
    %24 = llvm.mlir.constant(1 : i64) : i64
    %25 = llvm.mlir.constant(1 : i64) : i64
    %26 = llvm.mlir.constant(1 : i64) : i64
    %27 = llvm.mlir.constant(1 : i64) : i64
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.mlir.constant(1 : i64) : i64
    llvm.store %22, %20 : f64, !llvm.ptr
    llvm.store %22, %17 : f64, !llvm.ptr
    llvm.store %21, %14 : f32, !llvm.ptr
    llvm.store %21, %11 : f32, !llvm.ptr
    %30 = omp.map.info var_ptr(%20 : !llvm.ptr, f64) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "ce1"}
    %31 = omp.map.info var_ptr(%17 : !llvm.ptr, f64) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "ce2"}
    %32 = omp.map.info var_ptr(%14 : !llvm.ptr, f32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "ce3"}
    %33 = omp.map.info var_ptr(%11 : !llvm.ptr, f32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "ce4"}
    %34 = omp.map.info var_ptr(%5 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "j"}
    omp.target map_entries(%30 -> %arg0, %31 -> %arg1, %32 -> %arg2, %33 -> %arg3, %34 -> %arg4 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      %35 = llvm.mlir.constant(1.000000e+00 : f32) : f32
      %36 = llvm.mlir.constant(1.000000e+00 : f64) : f64
      %37 = llvm.mlir.constant(1000 : i32) : i32
      %38 = llvm.mlir.constant(1 : i32) : i32
      omp.teams reduction(@add_reduction_f64 %arg0 -> %arg5, @add_reduction_f64 %arg1 -> %arg6, @add_reduction_f32 %arg2 -> %arg7, @add_reduction_f32 %arg3 -> %arg8 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) {
        omp.parallel {
          omp.distribute {
            omp.wsloop reduction(@add_reduction_f64 %arg5 -> %arg9, @add_reduction_f64 %arg6 -> %arg10, @add_reduction_f32 %arg7 -> %arg11, @add_reduction_f32 %arg8 -> %arg12 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) {
              omp.simd private(@_QFEj_private_i32 %arg4 -> %arg13 : !llvm.ptr) reduction(@add_reduction_f64 %arg9 -> %arg14, @add_reduction_f64 %arg10 -> %arg15, @add_reduction_f32 %arg11 -> %arg16, @add_reduction_f32 %arg12 -> %arg17 : !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) {
                omp.loop_nest (%arg18) : i32 = (%38) to (%37) inclusive step (%38) {
                  llvm.store %arg18, %arg13 : i32, !llvm.ptr
                  %39 = llvm.load %arg14 : !llvm.ptr -> f64
                  %40 = llvm.fadd %39, %36 {fastmathFlags = #llvm.fastmath<contract>} : f64
                  llvm.store %40, %arg14 : f64, !llvm.ptr
                  %41 = llvm.load %arg15 : !llvm.ptr -> f64
                  %42 = llvm.fadd %41, %36 {fastmathFlags = #llvm.fastmath<contract>} : f64
                  llvm.store %42, %arg15 : f64, !llvm.ptr
                  %43 = llvm.load %arg16 : !llvm.ptr -> f32
                  %44 = llvm.fadd %43, %35 {fastmathFlags = #llvm.fastmath<contract>} : f32
                  llvm.store %44, %arg16 : f32, !llvm.ptr
                  %45 = llvm.load %arg17 : !llvm.ptr -> f32
                  %46 = llvm.fadd %45, %35 {fastmathFlags = #llvm.fastmath<contract>} : f32
                  llvm.store %46, %arg17 : f32, !llvm.ptr
                  omp.yield
                }
              } {omp.composite}
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

// CHECK: kernel_environment =
// CHECK-SAME: i32 24, i32 1024
// CHECK: call void @[[OUTLINED:__omp_offloading_[A-Za-z0-9_.]*]]
// CHECK: %[[MASTER:.+]] = call i32 @__kmpc_nvptx_teams_reduce_nowait_v2
// CHECK: icmp eq i32 %[[MASTER]], 1
// CHECK: i1 %{{.+}}, label %[[THEN:[A-Za-z0-9_.]*]], label %[[DONE:[A-Za-z0-9_.]*]]
// CHECK: [[THEN]]:
// CHECK-NEXT: %[[FINAL_RHS0:[A-Za-z0-9_.]*]] = load double
// CHECK-NEXT: %[[FINAL_LHS0:[A-Za-z0-9_.]*]] = load double
// CHECK-NEXT: %[[FINAL_RESULT0:[A-Za-z0-9_.]*]] = fadd contract double %[[FINAL_LHS0]], %[[FINAL_RHS0]]
// CHECK-NEXT: store double %[[FINAL_RESULT0]]
// CHECK-NEXT: %[[FINAL_RHS1:[A-Za-z0-9_.]*]] = load double
// CHECK-NEXT: %[[FINAL_LHS1:[A-Za-z0-9_.]*]] = load double
// CHECK-NEXT: %[[FINAL_RESULT1:[A-Za-z0-9_.]*]] = fadd contract double %[[FINAL_LHS1]], %[[FINAL_RHS1]]
// CHECK-NEXT: store double %[[FINAL_RESULT1]]
// CHECK-NEXT: %[[FINAL_RHS2:[A-Za-z0-9_.]*]] = load float
// CHECK-NEXT: %[[FINAL_LHS2:[A-Za-z0-9_.]*]] = load float
// CHECK-NEXT: %[[FINAL_RESULT2:[A-Za-z0-9_.]*]] = fadd contract float %[[FINAL_LHS2]], %[[FINAL_RHS2]]
// CHECK-NEXT: store float %[[FINAL_RESULT2]]
// CHECK-NEXT: %[[FINAL_RHS3:[A-Za-z0-9_.]*]] = load float
// CHECK-NEXT: %[[FINAL_LHS3:[A-Za-z0-9_.]*]] = load float
// CHECK-NEXT: %[[FINAL_RESULT3:[A-Za-z0-9_.]*]] = fadd contract float %[[FINAL_LHS3]], %[[FINAL_RHS3]]
// CHECK-NEXT: store float %[[FINAL_RESULT3]]
